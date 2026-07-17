"""
talmud_eval.py
Section-aware Talmud evaluation — extends genizah_fragment_eval.py.

Adds:
  - Ground truth parsing from --- Section --- delimited text files
  - Section-aware Gemini prompts (JSON output: gemara / rashi / tosafot)
  - Kraken / MiDRASH as a fourth model in the benchmark
  - LM Studio local models (any number; configurable via --lm_studio_models)
  - Per-section CER logged to W&B alongside the existing per-model metrics
  - Levenshtein-based CER (standard for HTR papers) alongside the existing
    SequenceMatcher CER so results are directly comparable to OCRBench / MiDRASH paper

Reuses from genizah_fragment_eval.py:
  BatchManager, EvalConfig, save_raw_output, save_incremental_result,
  evaluate_transcription, log_to_wandb (for base metrics), create_comparison_html

Run:
    python talmud_eval.py --max_documents 5 --skip_pro            # smoke test (Flash + Kraken only)
    python talmud_eval.py --lm_studio_models qwen/qwen3-vl-8b    # add Qwen3-VL
    python talmud_eval.py                                          # full run with all configured models
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from google.cloud import vision as gcloud_vision

import wandb

# Reuse the existing eval infrastructure
from src.datasets.evaluations.fragment_evals import (
    EvalConfig,
    BatchManager,
    evaluate_transcription,   # SequenceMatcher-based — kept for cross-run consistency
    save_raw_output,
    save_incremental_result,
    log_to_wandb,
    create_comparison_html,
)

# Agent imports (same path as fragment eval)
from src.models.llm.transcription.genizah_fragment_agent import (
    AgentConfig,
    call_gemini_with_retry,
    call_gemini_text_only,
)

# Talmud GT parser
from src.datasets.document_models.talmud_gt_parser import (
    load_gt_directory,
    get_section_summary,
    SECTION_KEYS,           # ('gemara', 'rashi', 'tosafot')
)

# Shared benchmark metrics (single source of truth for CER/WER)
from src.datasets.evaluations.metrics import (
    strip_nikud,
    normalize_whitespace,
    cer_levenshtein,
    cer_pair,
    char_count_ratio,
    flag_failure_modes,
)

# Domain-specific HTR
from src.models.ocr.kraken_transcriber import preload_kraken_model, transcribe_with_kraken

# Local VLMs via LM Studio
from src.models.ocr.lms_transcriber import (
    transcribe_with_lm_studio,
    check_lm_studio_health,
    LMStudioConfig,
)

# Claude (flag-gated — expensive, off by default)
from src.models.llm.transcription.claude_transcriber import (
    transcribe_with_claude,
    ClaudeConfig,
)

# LLM-judge error-type classification
from src.datasets.evaluations.error_classifier import (
    classify_transcription_errors,
    classification_table_rows,
    aggregate_error_stats,
    TABLE_COLUMNS as ERROR_TABLE_COLUMNS,
)


# ============================================================================
# Talmud-specific config (extends EvalConfig without modifying it)
# ============================================================================

TALMUD_IMAGES_DIR = Path(
    "/Users/isaac/Documents/GitHub/historical-document-analysis/"
    "src/datasets/raw_data/cairo_genizah/evaluations/talmud_sample/converted_images"
)
TALMUD_GT_DIR = Path(
    "/Users/isaac/Documents/GitHub/historical-document-analysis/"
    "src/datasets/raw_data/cairo_genizah/evaluations/talmud_sample/texts"
)
TALMUD_RESULTS_DIR = Path("./talmud_results")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

_MAX_CELL_CHARS = 3000  # W&B renders long strings slowly; cap per cell

# ============================================================================
# LM Studio / Claude runtime state
# Set once at the top of run_talmud_eval(); read throughout the module.
# ============================================================================

# Populated by run_talmud_eval() before any documents are evaluated.
_lm_studio_models: List[str] = []
_lm_studio_base_url: str = LMStudioConfig.base_url  # "http://localhost:1234/v1"
_claude_models: List[str] = []   # e.g. ["claude-opus-4-8", "claude-sonnet-5"]

# LLM judge for error-type classification (text-only; Gemini Flash default).
_judge_model: str = AgentConfig.GEMINI_ANALYSIS_MODEL
_judge_base_url: Optional[str] = None
_run_error_analysis: bool = True
_error_classifications: List[Dict] = []   # accumulated across documents


def _lm_key(model_id: str) -> str:
    """Stable short identifier for a model ID, safe for W&B column names.

    e.g. "qwen/qwen3-vl-8b" → "qwen3_vl_8b"
    """
    name = model_id.split("/")[-1]          # drop provider prefix
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# ============================================================================
# W&B table column construction
#
# Three tables only: gemara / rashi / tosafot  (no full_page table).
# Every table includes the full-page image so you can see what page a row
# refers to.  Column order per table:
#
#   image | doc_id | ground_truth
#   ── VLMs (text + CER per model) ──────────────────────────────────────────
#   gemini_pro | pro_cer_strict | pro_cer_lenient
#   gemini_flash | flash_cer_strict | flash_cer_lenient
#   {lm_key} | {lm_key}_cer_strict | {lm_key}_cer_lenient   (per LM Studio model)
#   ── OCR raw (flat full-page text — shown for inspection, no per-section CER) ──
#   vision_ocr_raw | kraken_raw
#   ── OCR segmented (text-LLM extracted section + CER) ─────────────────────
#   vision_ocr_segmented | vision_ocr_seg_cer_strict | vision_ocr_seg_cer_lenient
#   kraken_segmented     | kraken_seg_cer_strict     | kraken_seg_cer_lenient
#   ── metadata ──────────────────────────────────────────────────────────────
#   corpus | track | script_type | catalog_metadata_provided | gt_chars
# ============================================================================

def _section_columns() -> List[str]:
    """Column list for each of the three section tables.

    Layout: identity | ground_truth | all text outputs | all metrics | metadata
    Text outputs are kept together so you can scan across and compare visually.
    Metrics follow so they don't interrupt the reading flow.
    """
    lm_keys = [_lm_key(mid) for mid in _lm_studio_models]

    # ── Identity ──────────────────────────────────────────────────────────────
    cols = ["image", "doc_id", "ground_truth"]

    claude_keys = [_lm_key(m) for m in _claude_models]

    # ── Text outputs (all models together) ───────────────────────────────────
    cols += ["gemini_pro", "gemini_flash"]
    cols += claude_keys                    # Claude tiers side by side
    cols += lm_keys                        # open-weight VLMs
    cols += [
        "vision_ocr_raw",                  # flat full-page OCR (no section split)
        "kraken_raw",                      # flat full-page HTR
        "vision_ocr_segmented",            # after text-LLM segmentation
        "kraken_segmented",
    ]

    # ── Metrics (all models together, same order as text above) ──────────────
    cols += ["pro_cer_strict", "pro_cer_lenient"]
    cols += ["flash_cer_strict", "flash_cer_lenient"]
    for k in claude_keys:
        cols += [f"{k}_cer_strict", f"{k}_cer_lenient"]
    for k in lm_keys:
        cols += [f"{k}_cer_strict", f"{k}_cer_lenient"]
    cols += [
        # raw OCR has no meaningful per-section CER (flat text vs section GT)
        # so we skip raw metrics and go straight to segmented
        "vision_ocr_seg_cer_strict", "vision_ocr_seg_cer_lenient",
        "kraken_seg_cer_strict",     "kraken_seg_cer_lenient",
    ]

    # ── Metadata ──────────────────────────────────────────────────────────────
    cols += ["corpus", "track", "script_type", "catalog_metadata_provided", "gt_chars"]

    return cols


# ============================================================================
# Accumulated W&B table rows  (three section tables only)
# ============================================================================

_table_rows: Dict[str, List[List]] = {
    "gemara":  [],
    "rashi":   [],
    "tosafot": [],
}


def _cell(text: str) -> str:
    """Truncate text for a W&B table cell with a length indicator."""
    if len(text) <= _MAX_CELL_CHARS:
        return text
    return text[:_MAX_CELL_CHARS] + f"\n… [{len(text)} chars total]"


def _append_section_table_rows(
        doc_id: str,
        image_path: str,
        gt_sections: Dict[str, str],
        # VLM outputs — plain text per section, one call per section
        pro_sections: Optional[Dict[str, str]],
        flash_sections: Optional[Dict[str, str]],
        pro_section_metrics: Optional[Dict],
        flash_section_metrics: Optional[Dict],
        # Claude models (flag-gated) — keyed by model ID, like LM Studio
        claude_sections: Optional[Dict[str, Dict[str, str]]] = None,
        claude_section_metrics: Optional[Dict[str, Dict]] = None,
        # LM Studio VLMs
        lm_studio_sections: Optional[Dict[str, Dict[str, str]]] = None,
        lm_studio_section_metrics: Optional[Dict[str, Dict]] = None,
        # Traditional OCR — flat (raw) text from full page
        vision_ocr_raw: str = "",
        kraken_raw: str = "",
        # Traditional OCR — after text-LLM segmentation
        vision_sections: Optional[Dict[str, str]] = None,
        vision_section_metrics: Optional[Dict] = None,
        kraken_sections: Optional[Dict[str, str]] = None,
        kraken_section_metrics: Optional[Dict] = None,
        script_type: str = "talmud_vilna",
) -> None:
    """Append one row to each of the three section tables.

    Column order: image | doc_id | ground_truth
                  ── all text outputs ──
                  gemini_pro | gemini_flash | {lm_keys} |
                  vision_ocr_raw | kraken_raw |
                  vision_ocr_segmented | kraken_segmented
                  ── all metrics ──
                  pro_cer | flash_cer | {lm_key}_cer |
                  vision_ocr_seg_cer | kraken_seg_cer
                  ── metadata ──
    """
    claude_sections           = claude_sections           or {}
    claude_section_metrics    = claude_section_metrics    or {}
    lm_studio_sections        = lm_studio_sections        or {}
    lm_studio_section_metrics = lm_studio_section_metrics or {}
    vision_sections           = vision_sections           or {}
    vision_section_metrics    = vision_section_metrics    or {}
    kraken_sections           = kraken_sections           or {}
    kraken_section_metrics    = kraken_section_metrics    or {}

    def _cer(metrics: Dict, section: str, key: str):
        return (metrics.get(section) or {}).get(key) if metrics else None

    wb_image = wandb.Image(image_path, caption=doc_id) if Path(image_path).exists() else None

    for section in SECTION_KEYS:
        gt_text    = gt_sections.get(section, "")
        pro_text   = (pro_sections   or {}).get(section, "")
        flash_text = (flash_sections or {}).get(section, "")
        vis_seg    = vision_sections.get(section, "")
        krak_seg   = kraken_sections.get(section, "")

        # ── TEXT block ────────────────────────────────────────────────────
        row = [
            wb_image,
            doc_id,
            _cell(gt_text),
            # VLMs
            _cell(pro_text),
            _cell(flash_text),
        ]
        for model_id in _claude_models:
            c_text = (claude_sections.get(model_id) or {}).get(section, "")
            row.append(_cell(c_text))
        for model_id in _lm_studio_models:
            lm_text = (lm_studio_sections.get(model_id) or {}).get(section, "")
            row.append(_cell(lm_text))
        row += [
            # OCR raw (flat full-page — shown for inspection only)
            _cell(vision_ocr_raw),
            _cell(kraken_raw),
            # OCR segmented (section extracted by text LLM)
            _cell(vis_seg),
            _cell(krak_seg),
        ]

        # ── METRICS block ─────────────────────────────────────────────────
        row += [
            _cer(pro_section_metrics,   section, "cer_strict"),
            _cer(pro_section_metrics,   section, "cer_lenient"),
            _cer(flash_section_metrics, section, "cer_strict"),
            _cer(flash_section_metrics, section, "cer_lenient"),
        ]
        for model_id in _claude_models:
            c_mets = claude_section_metrics.get(model_id) or {}
            row += [
                _cer(c_mets, section, "cer_strict"),
                _cer(c_mets, section, "cer_lenient"),
            ]
        for model_id in _lm_studio_models:
            lm_mets = lm_studio_section_metrics.get(model_id) or {}
            row += [
                _cer(lm_mets, section, "cer_strict"),
                _cer(lm_mets, section, "cer_lenient"),
            ]
        row += [
            _cer(vision_section_metrics, section, "cer_strict"),
            _cer(vision_section_metrics, section, "cer_lenient"),
            _cer(kraken_section_metrics, section, "cer_strict"),
            _cer(kraken_section_metrics, section, "cer_lenient"),
        ]

        # ── METADATA block ────────────────────────────────────────────────
        row += ["talmud", "track2_fullpage", script_type, False, len(gt_text)]

        _table_rows[section].append(row)


def log_section_tables(step: int) -> None:
    """Re-log the three section tables to W&B after each document.

    Table names are stable so W&B replaces rather than appends.
    """
    cols = _section_columns()
    for section in SECTION_KEYS:
        rows = _table_rows[section]
        if rows:
            wandb.log(
                {f"transcriptions/{section}": wandb.Table(columns=cols, data=rows)},
                step=step,
            )


# cer_levenshtein, cer_pair, strip_nikud, normalize_whitespace
# are imported from src.datasets.evaluations.metrics


# ============================================================================
# Section-aware Gemini prompts
# ============================================================================

_LAYOUT = """This is a full page from the Babylonian Talmud (Vilna edition).
The page has three spatially distinct sections:
  • GEMARA (גמרא)    — centre column, large square Hebrew/Aramaic script
  • RASHI (רש"י)     — inner margin, smaller semi-cursive Rashi script
  • TOSAFOT (תוספות) — outer margin, smaller square script"""

# ── Per-section extraction prompts (plain text, no JSON) ─────────────────────
# Each prompt names only the target section.  The model must locate it on the
# full page and return plain text — no JSON wrapper, no parsing failures, and
# one section per call keeps output well within token limits.

SECTION_EXTRACT_PROMPTS: Dict[str, str] = {
    "gemara": f"""{_LAYOUT}

Extract ONLY the main Gemara text from the centre column.
Do not include Rashi or Tosafot commentary.
Transcribe exactly as written. Mark unclear characters with [?].
Preserve nikud and line breaks. Do NOT correct or complete from memory.

Return ONLY the Gemara transcription.""",

    "rashi": f"""{_LAYOUT}

Extract ONLY the Rashi commentary from the inner margin (semi-cursive Rashi script).
Do not include the Gemara or Tosafot.
Transcribe exactly as written. Mark unclear characters with [?].
Preserve nikud and line breaks. Do NOT correct or complete from memory.

Return ONLY the Rashi transcription.""",

    "tosafot": f"""{_LAYOUT}

Extract ONLY the Tosafot commentary from the outer margin (small square script).
Do not include the Gemara or Rashi.
Transcribe exactly as written. Mark unclear characters with [?].
Preserve nikud and line breaks. Do NOT correct or complete from memory.

Return ONLY the Tosafot transcription.""",
}

# ── OCR segmentation prompt (text-only — no image) ────────────────────────────
# Fed to a configurable LLM after flat OCR output is obtained from Cloud Vision,
# Kraken, or Tesseract.  The LLM identifies and separates the three sections
# from the unsegmented OCR stream using its knowledge of Talmud page structure.

# Structural framing (column order) — empirically the strongest prior for
# gemara/tosafot extraction: it tells the model WHERE the text lives.
# Kept byte-identical to the long-proven wording — paraphrasing it made
# gemara extraction bimodal (occasional collapse to ~1/3 of the section).
# NOTE: the rashi prompt deliberately does NOT use this framing — the
# labelled sub-sections it describes are the marginal apparatus, and
# conflating them with Rashi's commentary produced rashi CER > 1.0.
_OCR_STRUCTURE = """\
The OCR was run on a full Talmud page and reads COLUMN BY COLUMN.
The output has this structure:

  FIRST (~50–80 lines): Right column — the Rashi margin area, printed as
  labelled sub-sections. These labels all belong to the Rashi column:
    מסורת השיים, הגהות הב"ח, גליון השים, מוסף רש"י / מוסף ר'ש"י, רב ניסים גאון

  SEPARATOR: possibly a garbled/corrupted line at the column boundary.

  REMAINDER: page header (פרק … ברכות etc.) then center + left columns —
  continuous Gemara text (Aramaic/Hebrew discourse) mixed with Tosafot
  entries (start with ד"ה, contain וא"ת / ויש לומר / ותימה)."""

# Content taxonomy — used for the rashi call, where content shape (not
# position) is what identifies the text.  Tested on saved raw OCR: this
# dropped kraken rashi CER from 1.10/2.27 to 0.05/0.10 on the worst pages.
_CONTENT_TAXONOMY = """\
The OCR was run on a full page of the Babylonian Talmud (Vilna edition).
It reads COLUMN BY COLUMN, so the page's sections appear interleaved and out
of reading order, with OCR errors throughout.  The stream contains FOUR kinds
of text:

1. MARGINAL APPARATUS — short labelled reference blocks. Their (possibly
   garbled) labels include: מסורת הש"ס, הגהות הב"ח, גליון הש"ס, תורה אור,
   עין משפט, רב ניסים גאון, מוסף רש"י — plus the page header (e.g. פרק ... ברכות).
   These are NOT Rashi's commentary.
2. GEMARA — the main text: continuous Talmudic Aramaic/Hebrew discourse
   (dialectic with terms like אמר רב, תנו רבנן, מאי, תניא).
3. RASHI — Rashi's running commentary: a chain of SHORT glosses. Each gloss
   opens with a headword quoted from the Gemara, usually ending with a
   period, followed by a brief explanation, usually ending with a colon.
   Example shape:  בריוני. פריצים:  שפיל. השפיל עצמך לסוף המקרא ... :
4. TOSAFOT — longer dialectical entries. They also open with a headword, but
   contain extended argumentation with phrases like וא"ת (ואם תאמר),
   וי"ל / ויש לומר, ותימה, וקשה, אומר ר"ת."""

# One prompt per section — each call only needs to output ~1/3 of the text,
# avoiding the finish_reason=2 truncation that kills the single-prompt approach.
_SEGMENTATION_PROMPTS: Dict[str, str] = {
    "gemara": f"""{_OCR_STRUCTURE}

Extract ONLY the main Gemara text (continuous Talmudic Aramaic/Hebrew discourse).
Do NOT include the marginal apparatus, Rashi, or Tosafot commentary.
Return ONLY the raw text — no label, no explanation.""",

    "rashi": f"""{_CONTENT_TAXONOMY}

Extract ONLY kind 3 — Rashi's running commentary: the chain of short
headword-period + explanation-colon glosses.
Do NOT include the labelled marginal apparatus blocks (מסורת הש"ס,
הגהות הב"ח, גליון הש"ס, רב ניסים גאון etc.) — those are NOT Rashi.
Do NOT include the Gemara or the longer dialectical Tosafot entries.
Copy the text EXACTLY as the OCR produced it — do NOT correct, complete, or
normalize anything, even obvious OCR errors.
Return ONLY the raw extracted text — no label, no explanation.""",

    "tosafot": f"""{_OCR_STRUCTURE}

Extract ONLY the Tosafot commentary entries from the OCR output.
Tosafot entries start with ד"ה and contain analytical phrases like
וא"ת (ואם תאמר), ויש לומר, ותימה.
Return ONLY the raw text — no label, no explanation.""",
}

# Keep the old name as an alias so nothing else breaks if it's referenced
_SEGMENTATION_PROMPT = _SEGMENTATION_PROMPTS["gemara"]


# ============================================================================
# JSON parsing (shared by all models)
# ============================================================================

def _parse_section_json(raw: str, model_label: str = "") -> Optional[Dict[str, str]]:
    """Extract {gemara, rashi, tosafot} from a VLM response.

    Tries four strategies in order:
      1. Entire response is valid JSON
      2. JSON inside markdown fences (```json ... ```)
      3. Largest bare JSON object in the response
      4. Plain-text fallback: labelled sections (--- Gemara --- / **Gemara:**)

    Logs the raw response on failure for diagnosis.
    """
    if not raw:
        return None

    _KEY_ALIASES: Dict[str, str] = {
        "gemara": "gemara", "talmud": "gemara", "main text": "gemara",
        "גמרא": "gemara",
        "rashi": "rashi", "rashi commentary": "rashi", "commentary": "rashi",
        'rashi (רש"י)': "rashi", "rashi (רשי)": "rashi",
        "רשי": "rashi", 'רש"י': "rashi",
        "tosafot": "tosafot", "tosafos": "tosafot", "tosfot": "tosafot",
        "tosafot commentary": "tosafot",
        "תוספות": "tosafot",
    }

    def _normalize_keys(d: dict) -> Optional[Dict[str, str]]:
        out: Dict[str, str] = {}
        for raw_key, val in d.items():
            canonical = _KEY_ALIASES.get(raw_key.strip().lower())
            if canonical and canonical not in out:
                out[canonical] = str(val).strip()
        return out if out else None

    # Strategy 1: entire response is valid JSON
    try:
        data = json.loads(raw.strip())
        if isinstance(data, dict):
            result = _normalize_keys(data)
            if result:
                return {k: result.get(k, "") for k in SECTION_KEYS}
    except json.JSONDecodeError:
        pass

    # Strategy 2: JSON inside markdown fences (complete or truncated).
    # First try a complete fence pair; if absent, try everything after an
    # opening fence — handles finish_reason=2 (MAX_TOKENS) truncation where
    # the model never writes the closing ```.
    fence_body: Optional[str] = None
    complete_fence = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if complete_fence:
        fence_body = complete_fence.group(1)
    else:
        open_fence = re.search(r"```(?:json)?\s*(.*)", raw, re.DOTALL)
        if open_fence:
            fence_body = open_fence.group(1)
    if fence_body:
        try:
            data = json.loads(fence_body)
            if isinstance(data, dict):
                result = _normalize_keys(data)
                if result:
                    return {k: result.get(k, "") for k in SECTION_KEYS}
        except json.JSONDecodeError:
            pass

    # Strategy 3: largest balanced JSON object in response
    json_candidates = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", raw, re.DOTALL)
    json_candidates.sort(key=len, reverse=True)
    for candidate in json_candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                result = _normalize_keys(data)
                if result:
                    return {k: result.get(k, "") for k in SECTION_KEYS}
        except json.JSONDecodeError:
            continue

    # Strategy 4: plain-text section headers
    section_pattern = re.compile(
        r"(?:^|\n)\s*(?:\*\*|###|---\s*)?"
        r"(gemara|rashi|tosafot|tosafos|גמרא|רשי|תוספות)"
        r"(?:\s*\(.*?\))?"
        r"(?:\*\*|:|\s*---)*\s*\n"
        r"(.*?)(?=\n\s*(?:\*\*|###|---\s*)?(?:gemara|rashi|tosafot|tosafos|גמרא|רשי|תוספות)|$)",
        re.IGNORECASE | re.DOTALL,
    )
    plain_sections: Dict[str, str] = {}
    for m in section_pattern.finditer(raw):
        key = _KEY_ALIASES.get(m.group(1).strip().lower())
        if key and key not in plain_sections:
            plain_sections[key] = m.group(2).strip()
    if plain_sections:
        return {k: plain_sections.get(k, "") for k in SECTION_KEYS}

    # Strategy 5: partial JSON key extraction — salvages sections that completed
    # (or partially completed) before a MAX_TOKENS truncation or decoder-collapse.
    #
    # Two sub-patterns per key:
    #   5a — value string is properly closed:  "key": "value"
    #   5b — value string is truncated at EOF:  "key": "[text without closing quote
    #
    # Cap value length at 20 000 chars to avoid catastrophic backtracking on
    # collapsed outputs.
    partial: Dict[str, str] = {}
    for raw_key, canonical in _KEY_ALIASES.items():
        if canonical in partial:
            continue
        esc_key = re.escape(raw_key)

        # 5a: properly closed value
        pat_closed = re.compile(
            rf'"{esc_key}"\s*:\s*"((?:[^"\\]|\\.){{0,20000}})"',
            re.DOTALL | re.IGNORECASE,
        )
        m = pat_closed.search(raw)
        if m:
            value = m.group(1)
            try:
                value = json.loads(f'"{value}"')
            except (json.JSONDecodeError, ValueError):
                pass
            partial[canonical] = value.strip()
            continue

        # 5b: value string truncated — no closing quote, grab everything to EOF
        pat_trunc = re.compile(
            rf'"{esc_key}"\s*:\s*"((?:[^"\\]|\\.){{0,20000}})\Z',
            re.DOTALL | re.IGNORECASE,
        )
        m = pat_trunc.search(raw)
        if m:
            value = m.group(1)
            try:
                value = json.loads(f'"{value}"')
            except (json.JSONDecodeError, ValueError):
                value = value.replace("\\n", "\n").replace('\\"', '"')
            partial[canonical] = value.strip()

    if partial:
        print(
            f"    ⚠️  [{model_label}] Partial JSON recovery — "
            f"salvaged: {sorted(partial.keys())}"
        )
        return {k: partial.get(k, "") for k in SECTION_KEYS}

    label = f"[{model_label}] " if model_label else ""
    print(f"    ✗ {label}JSON parse failed on {len(raw)}-char response.")
    print(f"      First 400 chars: {repr(raw[:400])}")
    return None


# ============================================================================
# Section-level metrics
# ============================================================================

def compute_section_metrics(
        predicted: Dict[str, str],
        gt: Dict[str, str],
) -> Dict[str, Dict[str, Optional[float]]]:
    """CER per section + weighted overall."""
    metrics: Dict[str, Dict] = {}
    total_ref_chars: List[str] = []
    total_hyp_chars: List[str] = []
    total_ref_lenient: List[str] = []
    total_hyp_lenient: List[str] = []

    for section in SECTION_KEYS:
        ref = normalize_whitespace(gt.get(section, ""))
        hyp = normalize_whitespace(predicted.get(section, ""))
        if not ref:
            metrics[section] = {"cer_strict": None, "cer_lenient": None}
            continue
        s, l = cer_pair(hyp, ref)
        metrics[section] = {"cer_strict": s, "cer_lenient": l}
        total_ref_chars.extend(ref)
        total_hyp_chars.extend(hyp)
        total_ref_lenient.extend(strip_nikud(ref))
        total_hyp_lenient.extend(strip_nikud(hyp))

    if total_ref_chars:
        metrics["overall"] = {
            "cer_strict": cer_levenshtein(
                "".join(total_hyp_chars), "".join(total_ref_chars)
            ),
            "cer_lenient": cer_levenshtein(
                "".join(total_hyp_lenient), "".join(total_ref_lenient)
            ),
        }
    else:
        metrics["overall"] = {"cer_strict": 1.0, "cer_lenient": 1.0}

    return metrics


def _score_section_output(section: str, text: str, gt_text: str) -> Dict[str, Optional[float]]:
    """Score one section's output against its ground truth, guarding empty cases.

    Returns ``None`` metrics when there is no model output or no GT for the
    section — consistent with :func:`compute_section_metrics`, so unscorable
    sections don't pollute per-section means with artificial CER values.

    :param section: Section name (gemara / rashi / tosafot), for logging
    :type section: str
    :param text: Model output for the section
    :type text: str
    :param gt_text: Ground-truth text for the section
    :type gt_text: str
    :return: Dict with ``cer_strict`` / ``cer_lenient`` (floats or None)
    :rtype: Dict[str, Optional[float]]
    """
    if text and gt_text.strip():
        s, l = cer_pair(text, gt_text)
        print(f"    [{section}] {len(text)} chars  CER strict={s:.3f}  lenient={l:.3f}")
        return {"cer_strict": s, "cer_lenient": l}
    if text:
        print(f"    [{section}] {len(text)} chars — no GT for this section, not scored")
    else:
        print(f"    [{section}] no output")
    return {"cer_strict": None, "cer_lenient": None}


def _wandb_section_payload(model: str, metrics: Dict) -> Dict[str, float]:
    """Flatten section metrics to a W&B log dict."""
    out = {}
    for section, vals in metrics.items():
        for k, v in vals.items():
            if v is not None:
                out[f"{model}/{section}/{k}"] = v
    return out


# ============================================================================
# VLM call helpers
# ============================================================================

async def _call_vision_ocr(image_path: str) -> Tuple[Optional[str], float]:
    """Run Google Cloud Vision OCR and return (text, elapsed_s).

    Vision OCR returns a flat string with no section awareness.  The raw
    output is never scored (block order is not reading order) — it is shown
    for inspection and scored only after the segmentation-LLM step.
    Mirrors the vision_ocr_node() implementation in genizah_fragment_agent.py.
    """
    t0 = time.time()
    try:
        client = gcloud_vision.ImageAnnotatorClient()
        with open(image_path, "rb") as f:
            image = gcloud_vision.Image(content=f.read())
        response = client.text_detection(image=image)
        elapsed = time.time() - t0

        if response.text_annotations:
            text = response.text_annotations[0].description
            print(f"    ✓ {len(text)} chars (raw — scored after segmentation only)")
        else:
            text = ""
            print("    ⚠ No text detected by Vision OCR")

        return text, elapsed

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"    ✗ Vision OCR failed: {type(exc).__name__}: {str(exc)[:120]}")
        return None, elapsed


async def _call_section_extraction(
    model_name: str,
    image_path: str,
    section: str,
    timeout: int,
) -> Tuple[str, float]:
    """Extract one section from a full Talmud page using a VLM.

    One call per section — plain text output, no JSON, no parsing failures.
    Each call is ~1/3 the output tokens of the old all-sections approach,
    eliminating finish_reason=2 (MAX_TOKENS) truncations on Flash.
    """
    t0 = time.time()
    text = await call_gemini_with_retry(
        model_name, image_path,
        SECTION_EXTRACT_PROMPTS[section],
        temperature=0.1, timeout=timeout,
    )
    return (text or "").strip(), time.time() - t0


async def _call_lms_text_only(
    prompt: str,
    model_id: str,
    base_url: str,
    timeout: float = 120.0,
) -> Optional[str]:
    """Text-only call to an LM Studio model — no image.

    Used when an open-source text model is preferred over Gemini Flash
    for the OCR segmentation step.
    """
    import aiohttp as _aiohttp
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "temperature": 0.1,
    }
    try:
        async with _aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/chat/completions",
                json=payload,
                timeout=_aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return (data["choices"][0]["message"]["content"] or "").strip() or None
    except Exception as e:
        print(f"    ✗ LMS text-only failed ({model_id}): {e}")
        return None


async def _call_ocr_segmentation(
    flat_text: str,
    segmentation_model: str,
    segmentation_base_url: Optional[str],
    gt_sections: Dict[str, str],
) -> Tuple[Dict[str, str], Dict]:
    """Segment flat OCR output into Gemara / Rashi / Tosafot, with QC.

    ⚠️  This is NOT a single "segment this" call — full design rationale,
    empirical numbers, and editing guidance live in
    ``docs/talmud_ocr_segmentation.md``. Summary:

    **Stage 1 — extraction:** THREE separate calls, one per section, so each
    response is ~1/3 the size (avoids finish_reason=2 truncation). Gemara and
    Tosafot use the byte-frozen positional prompt (``_OCR_STRUCTURE`` —
    paraphrasing it causes bimodal collapse); Rashi uses the content-shape
    prompt (``_CONTENT_TAXONOMY``).

    **Stage 2 — quality control:** each sample is checked by two detectors
    (plain statistics with thresholds, resample at most once, keep the
    better-scoring sample):

    * *Collapse/over-extraction:* output length outside [0.75, 2.0] × the GT
      section length. NOTE: uses GT *length* only — an eval-harness
      stabilizer, disclosed in the docs.
    * *Canonical recitation (GT-free):* fraction of the output's 8-char
      shingles found verbatim in the OCR input. Faithful copies score
      0.92-0.95; samples where the LLM recites the memorized Vilna text
      (adding punctuation, "fixing" OCR errors) score ~0.55. Flag < 0.75.

    :param flat_text: Raw column-by-column OCR output (Vision or Kraken)
    :type flat_text: str
    :param segmentation_model: Gemini model name, or LM Studio model ID when
        ``segmentation_base_url`` is set
    :type segmentation_model: str
    :param segmentation_base_url: LM Studio URL, or None for Gemini
    :type segmentation_base_url: Optional[str]
    :param gt_sections: Ground-truth section texts (used for CER and for the
        length-based detector threshold)
    :type gt_sections: Dict[str, str]
    :return: (per-section extracted text, per-section CER metrics)
    :rtype: Tuple[Dict[str, str], Dict]
    """
    ocr_input = flat_text[:15_000]
    sections: Dict[str, str] = {}

    async def _call_one(section: str) -> str:
        prompt = _SEGMENTATION_PROMPTS[section] + f"\n\nOCR text:\n{ocr_input}"
        if segmentation_base_url:
            raw = await _call_lms_text_only(
                prompt, segmentation_model, segmentation_base_url
            )
        else:
            raw = await call_gemini_text_only(
                segmentation_model, prompt, timeout=90
            )
        return (raw or "").strip()

    def _shingle_overlap(out: str, n: int = 8) -> float:
        """Fraction of the output's n-char shingles present in the OCR input.

        The segmenter is supposed to COPY text from the OCR stream, so a
        faithful extraction scores ~0.9+.  When the LLM instead recites the
        canonical text from memory (adding punctuation, fixing OCR errors),
        overlap drops sharply (~0.5-0.6) — measurable without ground truth.
        """
        o = re.sub(r"\s+", " ", out).strip()
        s = re.sub(r"\s+", " ", ocr_input).strip()
        if len(o) < n:
            return 1.0
        src_shingles = {s[i:i + n] for i in range(len(s) - n + 1)}
        samples = [o[i:i + n] for i in range(0, len(o) - n + 1, 4)]
        return sum(sh in src_shingles for sh in samples) / len(samples)

    def _sample_score(out: str, gt_len: int) -> float:
        """Badness score for one segmentation sample (lower is better).

        Combines relative length deviation from the GT (collapse /
        over-extraction) with recitation evidence (1 - shingle overlap).
        """
        return abs(len(out) - gt_len) / max(gt_len, 1) + (1.0 - _shingle_overlap(out))

    for section in SECTION_KEYS:
        text = await _call_one(section)
        gt_len = len(gt_sections.get(section, ""))
        # The segmentation LLM misfires in two bimodal ways even at
        # temperature 0.1: (a) collapsing to a fraction of the section or
        # dumping several sections into one — caught by length vs GT; and
        # (b) reciting the memorized canonical text instead of copying the
        # OCR — caught by low shingle overlap with the input.  One resample
        # usually recovers; keep the better-scoring sample.
        if gt_len > 200:
            overlap = _shingle_overlap(text)
            bad_length = not (0.75 * gt_len <= len(text) <= 2.0 * gt_len)
            reciting = overlap < 0.75
            if bad_length or reciting:
                reason = "suspicious length" if bad_length else f"low OCR overlap ({overlap:.2f} — reciting?)"
                print(f"      [{section}] {reason} ({len(text)} chars vs "
                      f"gt={gt_len}) — resampling once...")
                retry = await _call_one(section)
                if _sample_score(retry, gt_len) < _sample_score(text, gt_len):
                    text = retry
        sections[section] = text
        print(f"      [{section}] extracted {len(text)} chars  "
              f"(gt={gt_len} chars)")
        if section != SECTION_KEYS[-1]:
            await asyncio.sleep(1)   # brief spacing between calls

    metrics = compute_section_metrics(sections, gt_sections)
    om = metrics.get("overall", {})
    print(
        f"    ✓ Segmented overall — "
        f"CER strict={om.get('cer_strict', 0):.3f}  "
        f"lenient={om.get('cer_lenient', 0):.3f}"
    )
    for sec in SECTION_KEYS:
        sec_m = metrics.get(sec, {})
        cer_s = sec_m.get("cer_strict")
        cer_str = f"CER={cer_s:.3f}" if cer_s is not None else "CER=n/a"
        gt_len  = len(gt_sections.get(sec, ""))
        flag    = "  ⚠️  segmentation failure?" if (
            cer_s is not None and cer_s >= 0.99 and gt_len > 50
        ) else ""
        print(f"      [{sec}] {cer_str}{flag}")
    return sections, metrics


# ============================================================================
# Per-document evaluation
# ============================================================================

async def evaluate_talmud_document(
        doc_id: str,
        image_path: Path,
        gt_sections: Dict[str, str],
        kraken_model_path: str,
        skip_flash: bool = False,
        skip_pro: bool = False,
        skip_kraken: bool = False,
        skip_lm_studio: bool = False,
        skip_vision_ocr: bool = False,
        claude_models: Optional[List[str]] = None,
        batch_num: Optional[int] = None,
        segmentation_model: str = AgentConfig.GEMINI_FLASH_MODEL,
        segmentation_base_url: Optional[str] = None,
) -> Optional[Dict]:
    """Run all models on one Talmud page and return a result dict.

    VLMs (Flash, Pro, LM Studio): three separate plain-text extraction calls,
    one per section. No JSON output, no parsing failures, no token-limit
    truncations.

    Traditional OCR (Vision, Kraken): one flat-text call → one text-only
    segmentation call to split the output into sections.  The segmentation
    model is configurable (default: Gemini Flash; set segmentation_base_url
    to use an open-source model via LM Studio instead).
    """
    print(f"\n📄 {doc_id}")
    print(get_section_summary(gt_sections))

    full_gt = "\n".join(
        gt_sections.get(k, "") for k in SECTION_KEYS if gt_sections.get(k)
    )
    if not full_gt.strip():
        print(
            f"  ⚠️  No GT text — gt_sections keys: {list(gt_sections.keys())}, "
            f"SECTION_KEYS: {SECTION_KEYS}"
        )
        return None

    state: Dict = {
        "doc_id": doc_id,
        "image_path": str(image_path),
        "ground_truth": full_gt,
        "processing_time": 0.0,
        "model_times": {},
        # Per-section outputs keyed by model
        "_pro_sections":          {},   # {section: text}
        "_flash_sections":        {},
        "_claude_sections":       {},
        "_lm_sections":           {},   # {model_id: {section: text}}
        "_vision_sections":       {},   # after segmentation
        "_kraken_sections":       {},   # after segmentation
        # Section metrics
        "_pro_section_metrics":   {},
        "_flash_section_metrics": {},
        "_claude_section_metrics": {},
        "_lm_section_metrics":    {},   # {model_id: metrics_dict}
        "_vision_section_metrics":{},
        "_kraken_section_metrics":{},
        # Flat texts for HTML / raw output
        "_vision_flat":  "",
        "_kraken_flat":  "",
        # Consensus (Pro preferred, Flash fallback)
        "final_transcription": "",
        "consensus_strategy":  "none",
    }

    t_doc_start = time.time()

    # ── Google Cloud Vision (flat OCR → segmentation) ─────────────────────────
    vision_text: Optional[str] = None
    if not skip_vision_ocr:
        print("  🔍 Vision OCR...")
        vision_text, vision_elapsed = await _call_vision_ocr(str(image_path))
        state["model_times"]["vision_ocr"] = vision_elapsed
        state["_vision_flat"] = vision_text or ""
        if vision_text:
            print(f"    → segmenting with {segmentation_model}...")
            vsec, vmets = await _call_ocr_segmentation(
                vision_text, segmentation_model, segmentation_base_url, gt_sections
            )
            state["_vision_sections"]        = vsec
            state["_vision_section_metrics"] = vmets

    # ── Gemini Pro — 3 separate section-extraction calls ─────────────────────
    if not skip_pro:
        print("  🎯 Gemini Pro (per-section extraction)...")
        pro_t_total = 0.0
        for section in SECTION_KEYS:
            text, elapsed = await _call_section_extraction(
                AgentConfig.GEMINI_PRO_MODEL, str(image_path),
                section, AgentConfig.GEMINI_PRO_TIMEOUT,
            )
            state["_pro_sections"][section] = text
            pro_t_total += elapsed
            state["_pro_section_metrics"][section] = _score_section_output(
                section, text, gt_sections.get(section, "")
            )
            await asyncio.sleep(2)
        state["model_times"]["gemini_pro"] = pro_t_total
        # Overall CER across all sections
        pro_overall = compute_section_metrics(state["_pro_sections"], gt_sections)
        state["_pro_section_metrics"]["overall"] = pro_overall.get("overall", {})
        pro_full = "\n".join(state["_pro_sections"].get(k, "") for k in SECTION_KEYS)
        state["final_transcription"] = pro_full
        state["consensus_strategy"]  = "pro_preferred"

    # ── Gemini Flash — 3 separate section-extraction calls ───────────────────
    if not skip_flash:
        await asyncio.sleep(3)
        print("  ⚡ Gemini Flash (per-section extraction)...")
        flash_t_total = 0.0
        for section in SECTION_KEYS:
            text, elapsed = await _call_section_extraction(
                AgentConfig.GEMINI_FLASH_MODEL, str(image_path),
                section, AgentConfig.GEMINI_FLASH_TIMEOUT,
            )
            state["_flash_sections"][section] = text
            flash_t_total += elapsed
            state["_flash_section_metrics"][section] = _score_section_output(
                section, text, gt_sections.get(section, "")
            )
            await asyncio.sleep(2)
        state["model_times"]["gemini_flash"] = flash_t_total
        flash_overall = compute_section_metrics(state["_flash_sections"], gt_sections)
        state["_flash_section_metrics"]["overall"] = flash_overall.get("overall", {})
        flash_full = "\n".join(state["_flash_sections"].get(k, "") for k in SECTION_KEYS)
        if not state["final_transcription"]:
            state["final_transcription"] = flash_full
            state["consensus_strategy"]  = "flash_fallback"

    # ── Claude — 3 section-extraction calls per model (flag-gated) ───────────
    for claude_model in (claude_models or []):
        ckey = _lm_key(claude_model)
        print(f"  🤖 Claude ({claude_model}, per-section extraction)...")
        claude_t_total = 0.0
        c_sections: Dict[str, str] = {}
        c_metrics: Dict = {}
        for section in SECTION_KEYS:
            t0 = time.time()
            text = await transcribe_with_claude(
                str(image_path), SECTION_EXTRACT_PROMPTS[section],
                model=claude_model,
            )
            text = (text or "").strip()
            claude_t_total += time.time() - t0
            c_sections[section] = text
            c_metrics[section] = _score_section_output(
                section, text, gt_sections.get(section, "")
            )
            await asyncio.sleep(1)
        c_overall = compute_section_metrics(c_sections, gt_sections)
        c_metrics["overall"] = c_overall.get("overall", {})
        state["_claude_sections"][claude_model] = c_sections
        state["_claude_section_metrics"][claude_model] = c_metrics
        state["model_times"][ckey] = claude_t_total

    # ── Kraken (flat HTR → segmentation) ─────────────────────────────────────
    kraken_flat = ""
    if not skip_kraken:
        print("  🐙 Kraken / MiDRASH...")
        t0 = time.time()
        kraken_flat = await transcribe_with_kraken(
            kraken_model_path, str(image_path), timeout=120.0
        ) or ""
        elapsed = time.time() - t0
        state["model_times"]["kraken"] = elapsed
        state["_kraken_flat"] = kraken_flat
        if kraken_flat:
            print(f"    {len(kraken_flat)} chars — segmenting with {segmentation_model}...")
            ksec, kmets = await _call_ocr_segmentation(
                kraken_flat, segmentation_model, segmentation_base_url, gt_sections
            )
            state["_kraken_sections"]        = ksec
            state["_kraken_section_metrics"] = kmets
        else:
            print("    ✗ Kraken returned nothing")

    # ── LM Studio — 3 per-section calls, MODEL-MAJOR order ────────────────────
    # All sections for one model before switching models: LM Studio holds one
    # large model in memory at a time, so interleaving models per section
    # forces an evict/reload on every call (and requests to the evicted model
    # 400 with "Model unloaded").
    if not skip_lm_studio and _lm_studio_models:
        print(f"  🖥  LM Studio: {', '.join(_lm_key(m) for m in _lm_studio_models)}...")
        for mid in _lm_studio_models:
            k = _lm_key(mid)
            t0 = time.time()
            sections: Dict[str, str] = {}
            for section in SECTION_KEYS:
                text = await transcribe_with_lm_studio(
                    mid, str(image_path), SECTION_EXTRACT_PROMPTS[section],
                    base_url=_lm_studio_base_url,
                )
                sections[section] = (text or "").strip()
                await asyncio.sleep(1)
            state["model_times"][k] = time.time() - t0
            mets = compute_section_metrics(sections, gt_sections)
            state["_lm_sections"][mid]        = sections
            state["_lm_section_metrics"][mid] = mets
            om = mets.get("overall", {})
            print(
                f"    [{k}] overall CER strict={om.get('cer_strict', 1.0):.3f}  "
                f"lenient={om.get('cer_lenient', 1.0):.3f}"
            )

    state["processing_time"] = time.time() - t_doc_start

    # ── W&B logging ───────────────────────────────────────────────────────────
    _step = len(_table_rows["gemara"])  # use gemara as the step counter

    pro_sections_d   = state["_pro_sections"]
    flash_sections_d = state["_flash_sections"]
    pro_full   = "\n".join(pro_sections_d.get(k, "")   for k in SECTION_KEYS)
    flash_full = "\n".join(flash_sections_d.get(k, "") for k in SECTION_KEYS)

    # ── LLM-judge error-type classification (full page per model) ────────────
    if _run_error_analysis:
        def _join_sections(secs: Dict[str, str]) -> str:
            return "\n".join(secs.get(k, "") for k in SECTION_KEYS)

        judge_candidates: Dict[str, str] = {}
        if pro_full.strip():
            judge_candidates["pro"] = pro_full
        if flash_full.strip():
            judge_candidates["flash"] = flash_full
        for mid, secs in state["_claude_sections"].items():
            judge_candidates[_lm_key(mid)] = _join_sections(secs)
        for mid, secs in state["_lm_sections"].items():
            judge_candidates[_lm_key(mid)] = _join_sections(secs)
        if state["_vision_sections"]:
            judge_candidates["vision_ocr_seg"] = _join_sections(state["_vision_sections"])
        if state["_kraken_sections"]:
            judge_candidates["kraken_seg"] = _join_sections(state["_kraken_sections"])

        print(f"  🧪 Error classification (judge: {_judge_model})...")
        for label, hyp_text in judge_candidates.items():
            if not hyp_text.strip():
                continue
            classification = await classify_transcription_errors(
                hyp_text, full_gt,
                judge_model=_judge_model, judge_base_url=_judge_base_url,
            )
            if classification:
                _error_classifications.append(
                    {"doc_id": doc_id, "model": label, **classification}
                )
                types = [e["type"] for e in classification["errors"]] or ["none"]
                print(f"    [{label}] {classification['overall_quality']}: "
                      f"{', '.join(types)}")

    # For the section tables we show the segmented OCR text per section;
    # for the full-page table we show the raw flat text.
    _append_section_table_rows(
        doc_id=doc_id,
        image_path=str(image_path),
        gt_sections=gt_sections,
        pro_sections=pro_sections_d or None,
        flash_sections=flash_sections_d or None,
        pro_section_metrics=state["_pro_section_metrics"] or None,
        flash_section_metrics=state["_flash_section_metrics"] or None,
        claude_sections=state["_claude_sections"] or None,
        claude_section_metrics=state["_claude_section_metrics"] or None,
        lm_studio_sections=state["_lm_sections"] or None,
        lm_studio_section_metrics=state["_lm_section_metrics"] or None,
        vision_ocr_raw=state["_vision_flat"],
        kraken_raw=state["_kraken_flat"],
        vision_sections=state["_vision_sections"] or None,
        vision_section_metrics=state["_vision_section_metrics"] or None,
        kraken_sections=state["_kraken_sections"] or None,
        kraken_section_metrics=state["_kraken_section_metrics"] or None,
        script_type="talmud_vilna",
    )
    log_section_tables(step=_step)

    # Scalar metrics
    scalar: Dict = {"doc_id": doc_id}
    for label, mets in [
        ("vision_ocr", state["_vision_section_metrics"]),
        ("flash",      state["_flash_section_metrics"]),
        ("pro",        state["_pro_section_metrics"]),
        ("kraken",     state["_kraken_section_metrics"]),
    ]:
        if mets:
            scalar.update(_wandb_section_payload(label, mets))
    for mid, mets in state["_claude_section_metrics"].items():
        scalar.update(_wandb_section_payload(_lm_key(mid), mets))
    for mid, mets in state["_lm_section_metrics"].items():
        scalar.update(_wandb_section_payload(_lm_key(mid), mets))
    for key, t in state["model_times"].items():
        scalar[f"timing/{key}_s"] = t
    wandb.log({k: v for k, v in scalar.items() if v is not None}, step=_step)

    # Raw outputs + incremental JSONL
    save_raw_output(doc_id, "ground_truth", full_gt, full_gt)
    if state["_vision_flat"]:
        save_raw_output(doc_id, "vision_ocr_flat", state["_vision_flat"])
    for section, text in state["_pro_sections"].items():
        if text:
            save_raw_output(doc_id, f"gemini_pro_{section}", text)
    for section, text in state["_flash_sections"].items():
        if text:
            save_raw_output(doc_id, f"gemini_flash_{section}", text)
    for mid, secs in state["_claude_sections"].items():
        for section, text in secs.items():
            if text:
                save_raw_output(doc_id, f"{_lm_key(mid)}_{section}", text)
    if state["_kraken_flat"]:
        save_raw_output(doc_id, "kraken_flat", state["_kraken_flat"])
    for mid, secs in state["_lm_sections"].items():
        for section, text in secs.items():
            if text:
                save_raw_output(doc_id, f"{_lm_key(mid)}_{section}", text)
    save_incremental_result(state, batch_num)

    return state


# ============================================================================
# Main evaluation loop
# ============================================================================

async def run_talmud_eval(
        images_dir: Path = TALMUD_IMAGES_DIR,
        gt_dir: Path = TALMUD_GT_DIR,
        kraken_model_path: str = "MiDRASH_Gen_01.mlmodel",
        lm_studio_models: Optional[List[str]] = None,
        lm_studio_base_url: str = "http://localhost:1234/v1",
        max_documents: Optional[int] = None,
        skip_vision_ocr: bool = False,
        skip_flash: bool = False,
        skip_pro: bool = False,
        skip_kraken: bool = False,
        skip_lm_studio: bool = False,
        use_claude: bool = False,
        claude_models: Optional[List[str]] = None,
        segmentation_model: str = AgentConfig.GEMINI_FLASH_MODEL,
        segmentation_base_url: Optional[str] = None,
        judge_model: str = AgentConfig.GEMINI_ANALYSIS_MODEL,
        judge_base_url: Optional[str] = None,
        skip_error_analysis: bool = False,
        wandb_project: str = EvalConfig.WANDB_PROJECT,
        wandb_run_name: Optional[str] = None,
) -> None:
    global _lm_studio_models, _lm_studio_base_url, _claude_models
    global _judge_model, _judge_base_url, _run_error_analysis

    # ── Set module-level config (drives dynamic column generation) ────────────
    _lm_studio_models = [] if (skip_lm_studio or not lm_studio_models) else lm_studio_models
    _lm_studio_base_url = lm_studio_base_url
    _claude_models = list(claude_models) if (use_claude and claude_models) else []
    _judge_model = judge_model
    _judge_base_url = judge_base_url
    _run_error_analysis = not skip_error_analysis

    if _claude_models and not os.getenv("ANTHROPIC_API_KEY"):
        print(
            "⚠️  ANTHROPIC_API_KEY is not set — Claude auth will fall back to "
            "an `ant auth login` profile if one exists."
        )

    TALMUD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EvalConfig.RAW_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Verify LM Studio is reachable before processing any documents ─────────
    if _lm_studio_models:
        print(f"🖥  Checking LM Studio at {_lm_studio_base_url} ...")
        try:
            available = await check_lm_studio_health(_lm_studio_base_url)
            print(f"   Available models: {available}")
            missing = [m for m in _lm_studio_models if m not in available]
            if missing:
                print(
                    f"   ⚠️  These models are not loaded in LM Studio: {missing}\n"
                    f"   Load them in LM Studio or they will produce empty results."
                )
        except RuntimeError as exc:
            print(f"   ⚠️  {exc}\n   Disabling LM Studio for this run.")
            _lm_studio_models = []

    # ── Match images ↔ GT by filename stem ───────────────────────────────────
    _page_suffix_re = re.compile(r"_page_\d+$", re.IGNORECASE)

    gt_by_stem = load_gt_directory(gt_dir)
    image_map: Dict[str, Path] = {}

    for ext in IMAGE_EXTENSIONS:
        for p in images_dir.glob(f"*{ext}"):
            if p.stem in gt_by_stem:
                image_map[p.stem] = p
                continue
            gt_key = _page_suffix_re.sub("", p.stem)
            if gt_key in gt_by_stem:
                image_map[p.stem] = p

    doc_ids = sorted(image_map.keys())
    if not doc_ids:
        raise RuntimeError(
            f"No images in {images_dir} matched GT stems in {gt_dir}.\n"
            f"GT stems found: {sorted(gt_by_stem.keys())}\n"
            f"Example image stems: "
            + ", ".join(
                p.stem
                for ext in IMAGE_EXTENSIONS
                for p in list(images_dir.glob(f"*{ext}"))[:3]
            )
        )

    if max_documents:
        doc_ids = doc_ids[:max_documents]

    print(f"Matched {len(doc_ids)} documents: {doc_ids}")

    if not skip_kraken:
        preload_kraken_model(kraken_model_path)

    wandb.init(
        project=wandb_project,
        name=wandb_run_name or f"talmud-{time.strftime('%Y%m%d-%H%M%S')}",
        config={
            "eval_type": "talmud_section_aware",
            "num_documents": len(doc_ids),
            "doc_ids": doc_ids,
            "kraken_model": kraken_model_path if not skip_kraken else None,
            "gemini_flash_model": AgentConfig.GEMINI_FLASH_MODEL,
            "gemini_pro_model": AgentConfig.GEMINI_PRO_MODEL,
            # LM Studio config captured for reproducibility
            "lm_studio_models": _lm_studio_models,
            "lm_studio_base_url": _lm_studio_base_url if _lm_studio_models else None,
            "claude_models": _claude_models,
            "error_judge_model": _judge_model if _run_error_analysis else None,
        },
        tags=["talmud", "section-eval", "kraken"]
             + ([f"lm-studio:{_lm_key(m)}" for m in _lm_studio_models])
             + ([f"claude:{_lm_key(m)}" for m in _claude_models]),
    )

    all_results: List[Dict] = []

    for i, doc_id in enumerate(doc_ids):
        gt_key = _page_suffix_re.sub("", doc_id)
        try:
            result = await evaluate_talmud_document(
                doc_id=doc_id,
                image_path=image_map[doc_id],
                gt_sections=gt_by_stem[gt_key],
                kraken_model_path=kraken_model_path,
                skip_vision_ocr=skip_vision_ocr,
                skip_flash=skip_flash,
                skip_pro=skip_pro,
                skip_kraken=skip_kraken,
                skip_lm_studio=not _lm_studio_models,
                claude_models=_claude_models,
                batch_num=1,
                segmentation_model=segmentation_model,
                segmentation_base_url=segmentation_base_url,
            )
            if result:
                all_results.append(result)
        except Exception as exc:
            print(f"  ❌ {doc_id}: {exc}")
            wandb.log({"error": str(exc), "error_doc": doc_id})

        if i < len(doc_ids) - 1:
            await asyncio.sleep(AgentConfig.DELAY_BETWEEN_DOCS)

    _log_summary(all_results)
    wandb.finish()


def _log_summary(results: List[Dict]) -> None:
    """Mean CER per model/section across all documents → wandb.summary.

    Reads directly from the accumulated _table_rows using the column index
    map from _section_columns() so this stays in sync with the table schema.
    """
    from collections import defaultdict
    accs: Dict[str, List[float]] = defaultdict(list)

    col_idx = {c: i for i, c in enumerate(_section_columns())}

    # All models and their CER column names in the section tables
    model_cer_cols = [
        ("pro",            "pro_cer_strict",              "pro_cer_lenient"),
        ("flash",          "flash_cer_strict",             "flash_cer_lenient"),
        ("vision_ocr_seg", "vision_ocr_seg_cer_strict",    "vision_ocr_seg_cer_lenient"),
        ("kraken_seg",     "kraken_seg_cer_strict",        "kraken_seg_cer_lenient"),
    ]
    for model_id in _claude_models + _lm_studio_models:
        k = _lm_key(model_id)
        model_cer_cols.append((k, f"{k}_cer_strict", f"{k}_cer_lenient"))

    for section in SECTION_KEYS:
        for row in _table_rows[section]:
            for model_label, col_strict, col_lenient in model_cer_cols:
                for col, suffix in [(col_strict, "cer_strict"), (col_lenient, "cer_lenient")]:
                    idx = col_idx.get(col)
                    if idx is not None:
                        v = row[idx]
                        if v is not None:
                            accs[f"mean/{model_label}/{section}/{suffix}"].append(v)

    summary = {k: sum(v) / len(v) for k, v in accs.items()}
    summary["num_documents"] = len(results)

    print("\nAggregate summary:")
    for k, v in sorted(summary.items()):
        wandb.summary[k] = v
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ── Error-type classification: per-doc table + benchmark summary ────────
    if _error_classifications:
        wandb.log({
            "error_classification": wandb.Table(
                columns=ERROR_TABLE_COLUMNS,
                data=classification_table_rows(_error_classifications),
            )
        })
        error_stats = aggregate_error_stats(_error_classifications)
        print("\n🧪 Error-type summary (fraction of docs affected):")
        for k, v in sorted(error_stats.items()):
            wandb.summary[k] = v
            if k.endswith("/pct_docs"):
                print(f"  {k.removeprefix('error_types/').removesuffix('/pct_docs')}: {v:.0%}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir",       type=Path, default=TALMUD_IMAGES_DIR)
    p.add_argument("--gt_dir",           type=Path, default=TALMUD_GT_DIR)
    p.add_argument("--kraken_model",     default="MiDRASH_Gen_01.mlmodel")
    p.add_argument("--max_documents",    type=int,  default=None)
    p.add_argument("--wandb_project",    default=EvalConfig.WANDB_PROJECT)
    p.add_argument("--wandb_run_name",   default=None)
    p.add_argument("--skip_flash",       action="store_true")
    p.add_argument("--skip_pro",         action="store_true")
    p.add_argument("--skip_kraken",      action="store_true")
    p.add_argument("--skip_lm_studio",   action="store_true")

    # LM Studio VLM models
    p.add_argument(
        "--lm_studio_models",
        type=lambda s: [m.strip() for m in s.split(",") if m.strip()],
        default=["qwen/qwen3-vl-8b"],
        metavar="MODEL[,MODEL,...]",
        help="LM Studio VLM model IDs (comma-separated). "
             "To add Gemma 4, use gemma-4-31b-it-mlx (correct but SLOW: "
             "~9 min/section on Apple Silicon — plan an overnight run). "
             "Do NOT use the QAT GGUF builds (google/gemma-4-31b-qat, "
             "google/gemma-4-26b-a4b-qat) — runaway thinking produces empty "
             "output and the 31B's crashes poison other models' requests.",
    )

    # Claude (flag-gated — expensive, off by default)
    p.add_argument(
        "--use_claude",
        action="store_true",
        help="Include Claude in the benchmark (off by default — tokens are expensive).",
    )
    p.add_argument(
        "--claude_models",
        type=lambda s: [m.strip() for m in s.split(",") if m.strip()],
        default=["claude-opus-4-8", "claude-sonnet-5"],
        metavar="MODEL[,MODEL,...]",
        help="Claude model IDs to score side by side when --use_claude is set "
             "(comma-separated). Default: claude-opus-4-8,claude-sonnet-5",
    )

    # LLM-judge error-type classification
    p.add_argument(
        "--judge_model",
        default=AgentConfig.GEMINI_ANALYSIS_MODEL,
        help="Text-only LLM that classifies error types vs the ground truth. "
             "Default: Gemini Flash. Set to an LM Studio model ID and provide "
             "--judge_base_url to use a local model instead.",
    )
    p.add_argument(
        "--judge_base_url",
        default=None,
        help="If set, the judge model is served via LM Studio at this URL.",
    )
    p.add_argument(
        "--skip_error_analysis",
        action="store_true",
        help="Disable the LLM-judge error-type classification.",
    )
    p.add_argument(
        "--lm_studio_base_url",
        default="http://localhost:1234/v1",
        help="LM Studio API base URL for VLM inference.",
    )

    # OCR segmentation model (text-only LLM step)
    p.add_argument(
        "--segmentation_model",
        default=AgentConfig.GEMINI_FLASH_MODEL,
        help="Model used to segment flat OCR output into Gemara/Rashi/Tosafot. "
             "Default: Gemini Flash (text-only call). "
             "Set to an LM Studio model ID and provide --segmentation_base_url "
             "to use an open-source text model instead.",
    )
    p.add_argument(
        "--segmentation_base_url",
        default=None,
        help="If set, the segmentation model is served via LM Studio at this URL. "
             "Leave unset to use Gemini (default).",
    )

    args = p.parse_args()

    asyncio.run(run_talmud_eval(
        images_dir=args.images_dir,
        gt_dir=args.gt_dir,
        kraken_model_path=(
            "/Users/isaac/Documents/GitHub/historical-document-analysis"
            "/src/datasets/raw_data/cairo_genizah/custom_model_weights"
            "/MiDRASH_Gen_01.mlmodel"
        ),
        lm_studio_models=args.lm_studio_models,
        lm_studio_base_url=args.lm_studio_base_url,
        max_documents=args.max_documents,
        skip_flash=args.skip_flash,
        skip_pro=args.skip_pro,
        skip_kraken=args.skip_kraken,
        skip_lm_studio=args.skip_lm_studio,
        use_claude=args.use_claude,
        claude_models=args.claude_models,
        segmentation_model=args.segmentation_model,
        segmentation_base_url=args.segmentation_base_url,
        judge_model=args.judge_model,
        judge_base_url=args.judge_base_url,
        skip_error_analysis=args.skip_error_analysis,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    ))