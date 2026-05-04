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
from src.models.llm.genizah_fragment_agent import AgentConfig, call_gemini_with_retry

# Talmud GT parser
from src.datasets.document_models.talmud_gt_parser import (
    load_gt_directory,
    strip_nikud,
    normalize_whitespace,
    get_section_summary,
    SECTION_KEYS,           # ('gemara', 'rashi', 'tosafot')
)

# Domain-specific HTR
from src.models.ocr.kraken_transcriber import preload_kraken_model, transcribe_with_kraken

# Local VLMs via LM Studio
from src.models.ocr.lms_transcriber import (
    transcribe_batch as lm_studio_transcribe_batch,
    check_lm_studio_health,
    LMStudioConfig,
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
_MAX_CELL_CHARS = 3000  # W&B renders long strings slowly; cap per cell

# ============================================================================
# LM Studio runtime state
# Set once at the top of run_talmud_eval(); read throughout the module.
# ========

# ============================================================================
# LM Studio runtime state
# Set once at the top of run_talmud_eval(); read throughout the module.
# ============================================================================

# Populated by run_talmud_eval() before any documents are evaluated.
_lm_studio_models: List[str] = []
_lm_studio_base_url: str = LMStudioConfig.base_url  # "http://localhost:1234/v1"


def _lm_key(model_id: str) -> str:
    """Stable short identifier for a model ID, safe for W&B column names.

    e.g. "qwen/qwen3-vl-8b" → "qwen3_vl_8b"
    """
    name = model_id.split("/")[-1]          # drop provider prefix
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# ============================================================================
# Dynamic W&B table column construction
#
# Columns depend on which LM Studio models are configured, so we build the
# lists lazily after _lm_studio_models is set.  Both _section_columns() and
# _full_page_columns() are pure functions — call them whenever you need a
# fresh column list (e.g. when constructing a wandb.Table).
# ============================================================================

def _section_columns() -> List[str]:
    """Per-section table columns (gemara / rashi / tosafot tables)."""
    cols = [
        "doc_id",
        "ground_truth",
        "vision_ocr",
        "gemini_flash",
        "gemini_pro",
        "vision_ocr_cer_strict",
        "vision_ocr_cer_lenient",
        "flash_cer_strict",
        "flash_cer_lenient",
        "pro_cer_strict",
        "pro_cer_lenient",
        "gt_chars",
        "vision_ocr_chars",
        "flash_chars",
        "pro_chars",
    ]
    for model_id in _lm_studio_models:
        k = _lm_key(model_id)
        cols += [k, f"{k}_cer_strict", f"{k}_cer_lenient", f"{k}_chars"]
    return cols


def _full_page_columns() -> List[str]:
    """Full-page table columns — image first so it renders leftmost in W&B."""
    cols = [
        "image",           # wandb.Image — rendered inline in the W&B table
        "doc_id",
        "ground_truth",
        "vision_ocr",
        "gemini_flash",
        "gemini_pro",
        "kraken",
        "vision_ocr_cer_strict",
        "vision_ocr_cer_lenient",
        "flash_cer_strict",
        "flash_cer_lenient",
        "pro_cer_strict",
        "pro_cer_lenient",
        "kraken_cer_strict",
        "kraken_cer_lenient",
        "gt_chars",
        "vision_ocr_chars",
        "flash_chars",
        "pro_chars",
        "kraken_chars",
    ]
    for model_id in _lm_studio_models:
        k = _lm_key(model_id)
        cols += [k, f"{k}_cer_strict", f"{k}_cer_lenient", f"{k}_chars"]
    return cols


# ============================================================================
# Accumulated W&B table rows
# ============================================================================

# Keyed by section name; "full_page" for the fourth table.
# Each inner list holds rows whose column order matches _section_columns() /
# _full_page_columns() at the time the rows were built.
_table_rows: Dict[str, List[List]] = {
    "gemara": [],
    "rashi": [],
    "tosafot": [],
    "full_page": [],
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
        vision_ocr_text: Optional[str],
        vision_ocr_section_metrics: Optional[Dict],
        flash_sections: Optional[Dict[str, str]],
        pro_sections: Optional[Dict[str, str]],
        flash_section_metrics: Optional[Dict],
        pro_section_metrics: Optional[Dict],
        kraken_text: Optional[str],
        kraken_metrics: Optional[Dict],
        full_gt: str,
        vision_ocr_full: str,
        flash_full: str,
        pro_full: str,
        lm_studio_sections: Optional[Dict[str, Dict[str, str]]] = None,
        lm_studio_section_metrics: Optional[Dict[str, Dict]] = None,
        lm_studio_full_texts: Optional[Dict[str, str]] = None,
) -> None:
    """Build one row per section table and one full-page row, then append.

    vision_ocr_text:            flat string from Cloud Vision (no section structure)
    vision_ocr_section_metrics: overall + per-section CER computed against full GT
    lm_studio_sections:         {model_id: {section_key: text}}
    lm_studio_section_metrics:  {model_id: {section_key: {cer_strict, cer_lenient}}}
    lm_studio_full_texts:       {model_id: full_concatenated_text}
    """
    lm_studio_sections = lm_studio_sections or {}
    lm_studio_section_metrics = lm_studio_section_metrics or {}
    lm_studio_full_texts = lm_studio_full_texts or {}
    vom = vision_ocr_section_metrics or {}

    def _cer(metrics, section, key):
        if not metrics:
            return None
        return (metrics.get(section) or {}).get(key)

    # ── Per-section rows (gemara, rashi, tosafot) ──────────────────────────
    # Vision OCR returns a flat string with no section awareness, so we show
    # the full OCR text in every section row and use overall CER for each.
    ocr_text = vision_ocr_text or ""
    for section in SECTION_KEYS:
        gt_text  = gt_sections.get(section, "")
        fl_text  = (flash_sections or {}).get(section, "")
        pro_text = (pro_sections or {}).get(section, "")

        row = [
            doc_id,
            _cell(gt_text),
            _cell(ocr_text),                                    # same flat text per section
            _cell(fl_text),
            _cell(pro_text),
            _cer(vom, "overall", "cer_strict"),                 # overall OCR CER (no section split)
            _cer(vom, "overall", "cer_lenient"),
            _cer(flash_section_metrics, section, "cer_strict"),
            _cer(flash_section_metrics, section, "cer_lenient"),
            _cer(pro_section_metrics, section, "cer_strict"),
            _cer(pro_section_metrics, section, "cer_lenient"),
            len(gt_text),
            len(ocr_text),
            len(fl_text),
            len(pro_text),
        ]
        # Append one group of columns per LM Studio model
        for model_id in _lm_studio_models:
            # Use `or {}` not `.get(id, {})` — the key may exist with value None
            # (parse failed) and dict.get() returns None in that case, not the default.
            lm_text = (lm_studio_sections.get(model_id) or {}).get(section, "")
            lm_metrics = lm_studio_section_metrics.get(model_id) or {}
            row += [
                _cell(lm_text),
                (lm_metrics.get(section) or {}).get("cer_strict"),
                (lm_metrics.get(section) or {}).get("cer_lenient"),
                len(lm_text),
            ]
        _table_rows[section].append(row)

    # ── Full-page row ──────────────────────────────────────────────────────
    km = kraken_metrics or {}

    # wandb.Image renders the thumbnail inline; caption shown on hover.
    wb_image = wandb.Image(image_path, caption=doc_id) if Path(image_path).exists() else None

    full_row = [
        wb_image,
        doc_id,
        _cell(full_gt),
        _cell(vision_ocr_full),
        _cell(flash_full),
        _cell(pro_full),
        _cell(kraken_text or ""),
        _cer(vom, "overall", "cer_strict"),
        _cer(vom, "overall", "cer_lenient"),
        _cer(flash_section_metrics, "overall", "cer_strict"),
        _cer(flash_section_metrics, "overall", "cer_lenient"),
        _cer(pro_section_metrics, "overall", "cer_strict"),
        _cer(pro_section_metrics, "overall", "cer_lenient"),
        km.get("cer_levenshtein"),
        km.get("cer_levenshtein_lenient"),
        len(full_gt),
        len(vision_ocr_full),
        len(flash_full),
        len(pro_full),
        len(kraken_text or ""),
    ]
    for model_id in _lm_studio_models:
        lm_full = lm_studio_full_texts.get(model_id) or ""
        lm_overall = (lm_studio_section_metrics.get(model_id) or {}).get("overall") or {}
        full_row += [
            _cell(lm_full),
            lm_overall.get("cer_strict"),
            lm_overall.get("cer_lenient"),
            len(lm_full),
        ]
    _table_rows["full_page"].append(full_row)


def log_section_tables(step: int) -> None:
    """Re-log all four section tables to W&B with the current accumulated rows.

    Called after every document so the tables update live in the W&B dashboard.
    Table names are stable across calls so W&B replaces rather than duplicates.
    """
    sec_cols = _section_columns()
    for section in SECTION_KEYS:
        rows = _table_rows[section]
        if rows:
            wandb.log(
                {f"transcriptions/{section}": wandb.Table(
                    columns=sec_cols,
                    data=rows,
                )},
                step=step,
            )

    full_rows = _table_rows["full_page"]
    if full_rows:
        wandb.log(
            {"transcriptions/full_page": wandb.Table(
                columns=_full_page_columns(),
                data=full_rows,
            )},
            step=step,
        )


# ============================================================================
# Levenshtein CER — standard for HTR benchmark papers
# ============================================================================

def _levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for ca in a:
        curr = [prev[0] + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def cer_levenshtein(hypothesis: str, reference: str) -> float:
    """Standard Levenshtein CER: edit_distance / len(reference).

    Use this for numbers reported in the paper — it matches OCRBench / MiDRASH
    paper conventions.  The existing calculate_cer() uses SequenceMatcher which
    gives slightly different values; we keep both for cross-run consistency.
    """
    ref = normalize_whitespace(reference)
    hyp = normalize_whitespace(hypothesis)
    if not ref:
        return 1.0
    return min(_levenshtein(hyp, ref) / len(ref), 1.0)


def cer_pair(hypothesis: str, reference: str) -> Tuple[float, float]:
    """Return (cer_strict, cer_lenient) — with and without nikud."""
    return (
        cer_levenshtein(hypothesis, reference),
        cer_levenshtein(strip_nikud(hypothesis), strip_nikud(reference)),
    )


# ============================================================================
# Section-aware Gemini prompts
# ============================================================================

_LAYOUT = """This is a page from the Babylonian Talmud (תלמוד בבלי).
The page has three spatially distinct sections:
  • GEMARA (גמרא)   — center, large square script (Babylonian Aramaic / Hebrew)
  • RASHI (רש"י)    — inner margin (right), smaller semi-cursive Rashi script
  • TOSAFOT (תוספות) — outer margin (left), smaller square script"""

_SCHEMA = """\
Return ONLY valid JSON — no markdown, no commentary:
{"gemara": "...", "rashi": "...", "tosafot": "..."}
Use empty string for any absent or fully illegible section."""

FLASH_PROMPT = f"""{_LAYOUT}
 
Transcribe each section EXACTLY as visible. Do NOT complete from memory — variants matter.
Mark unclear characters with [?]. Preserve nikud and line breaks (\\n).
{_SCHEMA}"""

PRO_PROMPT = f"""{_LAYOUT}
 
Step 1 — Locate each section by position and script style.
Step 2 — Transcribe what you see, NOT what you know the text should say.
  • Textual variants and scribal errors are the scholarly signal — preserve them.
  • Mark damaged/unclear chars with [?].
  • Preserve nikud and layout.
Step 3 — Output the JSON below.
{_SCHEMA}"""

# Qwen3-VL and other open-weight VLMs sometimes struggle with nested JSON
# inside complex RTL layouts, so the LM Studio prompt is slightly simplified
# while preserving the core instruction about transcribing variants faithfully.
LM_STUDIO_PROMPT = f"""{_LAYOUT}
 
Transcribe each section EXACTLY as it appears in the image.
CRITICAL: Do NOT correct or complete text from memory — every variant and
unusual spelling is intentional scholarly evidence.
Mark unreadable characters with [?]. Preserve nikud and line breaks (\\n).
{_SCHEMA}"""


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

    # Strategy 2: JSON inside markdown fences
    # Capture everything between the fences and let json.loads delimit the object —
    # the previous {.*?} regex stopped at the first } which truncated mid-object.
    fence_m = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fence_m:
        try:
            data = json.loads(fence_m.group(1))
            if isinstance(data, dict):
                result = _normalize_keys(data)
                if result:
                    return {k: result.get(k, "") for k in SECTION_KEYS}
        except json.JSONDecodeError:
            pass

    # Strategy 3: largest JSON object in response
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

async def _call_vision_ocr(image_path: str, full_gt: str) -> Tuple[Optional[str], Dict, float]:
    """Run Google Cloud Vision OCR and return (text, metrics, elapsed_s).

    Vision OCR returns a flat string with no section awareness — the same text
    is shown in every section table row, and CER is computed against the full GT.
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
        else:
            text = ""

        if text:
            strict, lenient = cer_pair(text, full_gt)
            metrics = {
                "overall": {"cer_strict": strict, "cer_lenient": lenient},
                **{s: {"cer_strict": None, "cer_lenient": None} for s in SECTION_KEYS},
            }
            print(f"    ✓ {len(text)} chars  CER strict={strict:.3f}  lenient={lenient:.3f}")
        else:
            metrics = {
                "overall": {"cer_strict": 1.0, "cer_lenient": 1.0},
                **{s: {"cer_strict": None, "cer_lenient": None} for s in SECTION_KEYS},
            }
            print("    ⚠ No text detected by Vision OCR")

        return text, metrics, elapsed

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"    ✗ Vision OCR failed: {type(exc).__name__}: {str(exc)[:120]}")
        return None, {}, elapsed


async def _call_section_aware(
    model_name: str,
    image_path: str,
    prompt: str,
    timeout: int,
) -> Tuple[Optional[Dict[str, str]], float]:
    """Call a Gemini model and return (parsed_sections, elapsed_s)."""
    t0 = time.time()
    raw = await call_gemini_with_retry(
        model_name, image_path, prompt, temperature=0.1, timeout=timeout
    )
    label = model_name.split("-")[0]
    return _parse_section_json(raw, model_label=label), time.time() - t0


async def _call_lm_studio_models(
    image_path: str,
    gt_sections: Dict[str, str],
) -> Tuple[
    Dict[str, Optional[Dict[str, str]]],   # model_id → parsed sections (None if parse failed)
    Dict[str, Dict],                        # model_id → section metrics
    Dict[str, str],                         # model_id → full text (raw fallback if parse failed)
    Dict[str, float],                       # model_id → elapsed_s
]:
    """Run all configured LM Studio models on one image concurrently.

    Even when JSON parsing fails (e.g. Qwen repetition loops, truncated output),
    we preserve the raw response as the full text and compute CER/WER against it.
    A parse failure with terrible CER is a valid benchmark result — it documents
    the failure mode.  Models that return nothing at all get None full text and
    no metrics.
    """
    if not _lm_studio_models:
        return {}, {}, {}, {}

    t0 = time.time()
    raw_results: Dict[str, Optional[str]] = await lm_studio_transcribe_batch(
        model_ids=_lm_studio_models,
        image_path=image_path,
        prompt=LM_STUDIO_PROMPT,
        base_url=_lm_studio_base_url,
    )
    elapsed = time.time() - t0

    full_gt = "\n".join(gt_sections.get(k, "") for k in SECTION_KEYS if gt_sections.get(k))

    parsed: Dict[str, Optional[Dict[str, str]]] = {}
    section_metrics: Dict[str, Dict] = {}
    full_texts: Dict[str, str] = {}
    timings: Dict[str, float] = {}

    for model_id, raw in raw_results.items():
        k = _lm_key(model_id)
        timings[model_id] = elapsed

        if not raw:
            # Model returned nothing — no text, no metrics
            parsed[model_id] = None
            full_texts[model_id] = ""
            section_metrics[model_id] = {}
            print(f"    ✗ [{k}] No response from model")
            continue

        sections = _parse_section_json(raw, model_label=k)
        parsed[model_id] = sections

        if sections:
            # Clean JSON parse — compute per-section metrics
            full_texts[model_id] = "\n".join(
                sections.get(s, "") for s in SECTION_KEYS if sections.get(s)
            )
            section_metrics[model_id] = compute_section_metrics(sections, gt_sections)
            m = section_metrics[model_id]
            print(
                f"    ✓ [{k}] CER strict={m['overall']['cer_strict']:.3f} "
                f"lenient={m['overall']['cer_lenient']:.3f}"
            )
        else:
            # Parse failed (malformed JSON, repetition loop, etc.) — use raw text.
            # Section-level breakdown is unavailable, but we can still compute
            # overall CER/WER against the full GT.  This is the data point that
            # documents the failure mode in the benchmark.
            full_texts[model_id] = raw
            strict, lenient = cer_pair(raw, full_gt)
            section_metrics[model_id] = {
                "overall": {"cer_strict": strict, "cer_lenient": lenient},
                # Per-section metrics are None — parse failed so we can't attribute
                **{s: {"cer_strict": None, "cer_lenient": None} for s in SECTION_KEYS},
            }
            print(
                f"    ⚠ [{k}] JSON parse failed — logging raw output as fallback "
                f"({len(raw)} chars, CER strict={strict:.3f} lenient={lenient:.3f})"
            )

    return parsed, section_metrics, full_texts, timings


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
        batch_num: Optional[int] = None,
) -> Optional[Dict]:
    """Run all models on one Talmud page and return a result dict."""
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

    state = {
        "doc_id": doc_id,
        "image_path": str(image_path),
        "catalog_metadata": {"description": f"Talmud Bavli — {doc_id}"},
        "ground_truth": full_gt,
        "vision_ocr_result": None,
        "gemini_flash_result": None,
        "gemini_pro_result": None,
        "kraken_result": None,
        "vision_ocr_metrics": None,
        "gemini_flash_metrics": None,
        "gemini_pro_metrics": None,
        "kraken_metrics": None,
        "all_results": [],
        "disagreements": [],
        "needs_review": False,
        "final_transcription": "",
        "confidence_score": 0.0,
        "consensus_strategy": "pro_preferred",
        "consensus_metrics": None,
        "analysis_result": None,
        "processing_time": 0.0,
        "model_times": {},
        # Section-level data (not in the base schema)
        "_flash_sections": None,
        "_pro_sections": None,
        "_flash_section_metrics": None,
        "_pro_section_metrics": None,
        # LM Studio — {model_id: ...}
        "_lm_sections": {},
        "_lm_section_metrics": {},
        "_lm_full_texts": {},
        "_lm_results": {},
    }

    t_doc_start = time.time()

    # ── Google Cloud Vision OCR ───────────────────────────────────────────────
    if not skip_vision_ocr:
        print("  🔍 Vision OCR...")
        vision_text, vision_section_metrics, vision_elapsed = await _call_vision_ocr(
            str(image_path), full_gt
        )
        state["model_times"]["vision_ocr"] = vision_elapsed
        if vision_text is not None:
            state["vision_ocr_result"] = {
                "text": vision_text, "model": "google_vision_ocr",
                "char_count": len(vision_text), "processing_time": vision_elapsed,
            }
            state["vision_ocr_metrics"] = evaluate_transcription(full_gt, vision_text, "vision_ocr")
            state["vision_ocr_metrics"]["processing_time"] = vision_elapsed
    else:
        vision_text, vision_section_metrics, vision_elapsed = None, {}, 0.0

    # ── Gemini Pro ────────────────────────────────────────────────────────────
    if not skip_pro:
        print("  🎯 Gemini Pro (section-aware)...")
        t0 = time.time()
        raw_pro = await call_gemini_with_retry(
            AgentConfig.GEMINI_PRO_MODEL, str(image_path),
            PRO_PROMPT, temperature=0.1, timeout=AgentConfig.GEMINI_PRO_TIMEOUT,
        )
        elapsed = time.time() - t0
        state["model_times"]["gemini_pro"] = elapsed

        sections = _parse_section_json(raw_pro or "", model_label="pro") if raw_pro else None

        if sections:
            reconstructed = "\n".join(
                sections.get(k, "") for k in SECTION_KEYS if sections.get(k)
            )
        elif raw_pro:
            reconstructed = raw_pro
            print(f"    ⚠ [pro] JSON parse failed — using raw output for metrics ({len(raw_pro)} chars)")
        else:
            reconstructed = ""

        if reconstructed:
            state["gemini_pro_result"] = {
                "text": reconstructed, "model": "gemini_pro_talmud_section",
                "char_count": len(reconstructed), "processing_time": elapsed,
                "parse_failed": sections is None,
            }
            state["gemini_pro_metrics"] = evaluate_transcription(
                full_gt, reconstructed, "gemini_pro"
            )
            state["gemini_pro_metrics"]["processing_time"] = elapsed
            state["final_transcription"] = reconstructed
            state["consensus_metrics"] = evaluate_transcription(
                full_gt, reconstructed, "consensus_pro_preferred"
            )

            if sections:
                section_m = compute_section_metrics(sections, gt_sections)
                state["_pro_sections"] = sections
                state["_pro_section_metrics"] = section_m
                print(
                    f"    ✓ overall CER strict={section_m['overall']['cer_strict']:.3f} "
                    f"lenient={section_m['overall']['cer_lenient']:.3f}"
                )
            else:
                strict, lenient = cer_pair(reconstructed, full_gt)
                state["_pro_section_metrics"] = {
                    "overall": {"cer_strict": strict, "cer_lenient": lenient},
                    **{s: {"cer_strict": None, "cer_lenient": None} for s in SECTION_KEYS},
                }
                print(f"    ⚠ overall CER strict={strict:.3f} lenient={lenient:.3f} (raw fallback)")

    # Fallback consensus to Flash if Pro failed
    if not state["final_transcription"] and state.get("gemini_flash_result"):
        state["final_transcription"] = state["gemini_flash_result"]["text"]
        state["consensus_strategy"] = "flash_fallback"
        state["consensus_metrics"] = state["gemini_flash_metrics"]

    # ── Gemini Flash ──────────────────────────────────────────────────────────
    if not skip_flash:
        await asyncio.sleep(3)
        print("  ⚡ Gemini Flash (section-aware)...")
        t0 = time.time()
        raw_flash = await call_gemini_with_retry(
            AgentConfig.GEMINI_FLASH_MODEL, str(image_path),
            FLASH_PROMPT, temperature=0.1, timeout=AgentConfig.GEMINI_FLASH_TIMEOUT,
        )
        elapsed = time.time() - t0
        state["model_times"]["gemini_flash"] = elapsed

        sections = _parse_section_json(raw_flash or "", model_label="flash") if raw_flash else None

        if sections:
            reconstructed = "\n".join(
                sections.get(k, "") for k in SECTION_KEYS if sections.get(k)
            )
        elif raw_flash:
            # JSON parse failed — use raw response so we still get CER metrics
            reconstructed = raw_flash
            print(f"    ⚠ [flash] JSON parse failed — using raw output for metrics ({len(raw_flash)} chars)")
        else:
            reconstructed = ""

        if reconstructed:
            state["gemini_flash_result"] = {
                "text": reconstructed, "model": "gemini_flash_talmud_section",
                "char_count": len(reconstructed), "processing_time": elapsed,
                "parse_failed": sections is None,
            }
            state["gemini_flash_metrics"] = evaluate_transcription(
                full_gt, reconstructed, "gemini_flash"
            )
            state["gemini_flash_metrics"]["processing_time"] = elapsed

            if sections:
                section_m = compute_section_metrics(sections, gt_sections)
                state["_flash_sections"] = sections
                state["_flash_section_metrics"] = section_m
                print(
                    f"    ✓ overall CER strict={section_m['overall']['cer_strict']:.3f} "
                    f"lenient={section_m['overall']['cer_lenient']:.3f}"
                )
            else:
                # Compute overall-only metrics from raw text; sections stay None
                strict, lenient = cer_pair(reconstructed, full_gt)
                state["_flash_section_metrics"] = {
                    "overall": {"cer_strict": strict, "cer_lenient": lenient},
                    **{s: {"cer_strict": None, "cer_lenient": None} for s in SECTION_KEYS},
                }
                print(f"    ⚠ overall CER strict={strict:.3f} lenient={lenient:.3f} (raw fallback)")

    # ── Kraken ────────────────────────────────────────────────────────────────
    if not skip_kraken:
        print("  🐙 Kraken / MiDRASH...")
        t0 = time.time()
        kraken_text = await transcribe_with_kraken(
            kraken_model_path, str(image_path), timeout=120.0
        )
        elapsed = time.time() - t0
        state["model_times"]["kraken"] = elapsed
        if kraken_text:
            state["kraken_result"] = {
                "text": kraken_text, "model": "kraken_midrash_gen_01",
                "char_count": len(kraken_text), "processing_time": elapsed,
            }
            state["kraken_metrics"] = evaluate_transcription(
                full_gt, kraken_text, "kraken"
            )
            state["kraken_metrics"]["processing_time"] = elapsed
            (
                state["kraken_metrics"]["cer_levenshtein"],
                state["kraken_metrics"]["cer_levenshtein_lenient"],
            ) = cer_pair(kraken_text, full_gt)
            print(
                f"    ✓ {len(kraken_text)} chars  "
                f"CER strict={state['kraken_metrics']['cer_levenshtein']:.3f}  "
                f"lenient={state['kraken_metrics']['cer_levenshtein_lenient']:.3f}"
            )
        else:
            print("    ✗ Kraken failed")

    # ── LM Studio models ──────────────────────────────────────────────────────
    if not skip_lm_studio and _lm_studio_models:
        print(f"  🖥  LM Studio: {', '.join(_lm_key(m) for m in _lm_studio_models)}...")
        try:
            (
                lm_sections,
                lm_section_metrics,
                lm_full_texts,
                lm_timings,
            ) = await _call_lm_studio_models(str(image_path), gt_sections)
        except Exception as exc:
            print(f"    ✗ LM Studio batch failed: {exc}")
            lm_sections, lm_section_metrics, lm_full_texts, lm_timings = {}, {}, {}, {}

        state["_lm_sections"] = lm_sections
        state["_lm_section_metrics"] = lm_section_metrics
        state["_lm_full_texts"] = lm_full_texts

        for model_id, elapsed in lm_timings.items():
            state["model_times"][_lm_key(model_id)] = elapsed

        # Build result dicts compatible with save_raw_output
        for model_id, full_text in lm_full_texts.items():
            if full_text:
                k = _lm_key(model_id)
                state["_lm_results"][model_id] = {
                    "text": full_text,
                    "model": model_id,
                    "char_count": len(full_text),
                    "processing_time": lm_timings.get(model_id, 0.0),
                }
                # Also store aggregate metrics for the existing log_to_wandb path
                lm_agg = evaluate_transcription(full_gt, full_text, k)
                lm_agg["processing_time"] = lm_timings.get(model_id, 0.0)
                state["_lm_results"][model_id]["metrics"] = lm_agg

    state["processing_time"] = time.time() - t_doc_start

    # ── W&B logging ───────────────────────────────────────────────────────────
    _step = len(_table_rows["full_page"])   # before appending the new row

    flash_sections = state.get("_flash_sections") or {}
    pro_sections   = state.get("_pro_sections") or {}
    # Fall back to the raw result text when JSON parse failed and _flash/_pro_sections
    # are empty — otherwise the table shows an empty cell next to a real CER value.
    flash_full = (
        "\n".join(flash_sections.get(k, "") for k in SECTION_KEYS)
        if flash_sections
        else (state.get("gemini_flash_result") or {}).get("text", "")
    )
    pro_full = (
        "\n".join(pro_sections.get(k, "") for k in SECTION_KEYS)
        if pro_sections
        else (state.get("gemini_pro_result") or {}).get("text", "")
    )
    kraken_text    = (state.get("kraken_result") or {}).get("text", "")
    kraken_metrics = state.get("kraken_metrics")

    # 1. Append to section tables + re-log immediately for live W&B updates
    _append_section_table_rows(
        doc_id=doc_id,
        image_path=str(image_path),
        gt_sections=gt_sections,
        vision_ocr_text=vision_text,
        vision_ocr_section_metrics=vision_section_metrics,
        flash_sections=flash_sections or None,
        pro_sections=pro_sections or None,
        flash_section_metrics=state.get("_flash_section_metrics"),
        pro_section_metrics=state.get("_pro_section_metrics"),
        kraken_text=kraken_text or None,
        kraken_metrics=kraken_metrics,
        full_gt=full_gt,
        vision_ocr_full=vision_text or "",
        flash_full=flash_full,
        pro_full=pro_full,
        lm_studio_sections=state.get("_lm_sections"),
        lm_studio_section_metrics=state.get("_lm_section_metrics"),
        lm_studio_full_texts=state.get("_lm_full_texts"),
    )
    log_section_tables(step=_step)

    # 2. Scalar metrics for chart views
    scalar_payload: Dict = {"doc_id": doc_id}
    if vision_section_metrics:
        scalar_payload.update(_wandb_section_payload("vision_ocr", vision_section_metrics))
        scalar_payload["vision_ocr/processing_time_s"] = vision_elapsed
    if state.get("_flash_section_metrics"):
        scalar_payload.update(_wandb_section_payload("flash", state["_flash_section_metrics"]))
    if state.get("_pro_section_metrics"):
        scalar_payload.update(_wandb_section_payload("pro", state["_pro_section_metrics"]))
    if kraken_metrics:
        scalar_payload.update({
            "kraken/overall/cer_strict": kraken_metrics.get("cer_levenshtein"),
            "kraken/overall/cer_lenient": kraken_metrics.get("cer_levenshtein_lenient"),
            "kraken/overall/wer": kraken_metrics.get("wer"),
            "kraken/processing_time_s": kraken_metrics.get("processing_time"),
        })
    for model_id, m in state.get("_lm_section_metrics", {}).items():
        k = _lm_key(model_id)
        scalar_payload.update(_wandb_section_payload(k, m))
        scalar_payload[f"{k}/processing_time_s"] = state["model_times"].get(k)

    for model_key in ("vision_ocr", "gemini_flash", "gemini_pro", "kraken"):
        t = state["model_times"].get(model_key)
        if t is not None:
            scalar_payload[f"timing/{model_key}_s"] = t

    wandb.log({k: v for k, v in scalar_payload.items() if v is not None}, step=_step)

    # 3. HTML comparison
    if Path(state["image_path"]).exists():
        _update_comparison_html(state)

    # 4. Raw text files + incremental JSONL
    for model_key, label in [
        ("vision_ocr_result", "vision_ocr"),
        ("gemini_flash_result", "gemini_flash"),
        ("gemini_pro_result", "gemini_pro"),
        ("kraken_result", "kraken"),
    ]:
        r = state.get(model_key)
        if r:
            save_raw_output(doc_id, label, r["text"])

    for model_id, result in state.get("_lm_results", {}).items():
        if result:
            save_raw_output(doc_id, _lm_key(model_id), result["text"])

    save_raw_output(doc_id, "ground_truth", full_gt, full_gt)
    save_incremental_result(state, batch_num)

    return state


def _update_comparison_html(state: Dict) -> None:
    """Generate comparison HTML, injecting Kraken and LM Studio blocks."""
    html = create_comparison_html(
        state["doc_id"],
        state.get("ground_truth", ""),
        (state.get("vision_ocr_result") or {}).get("text", ""),
        (state.get("gemini_flash_result") or {}).get("text", ""),
        (state.get("gemini_pro_result") or {}).get("text", ""),
        state["final_transcription"],
        state["consensus_strategy"],
    )

    extra_blocks = ""

    kraken_text = (state.get("kraken_result") or {}).get("text", "")
    if kraken_text:
        extra_blocks += f"""
        <div class="section" style="grid-column: span 2;">
            <h3>Kraken / MiDRASH Gen 01 ({len(kraken_text)} chars)</h3>
            <div class="text">{kraken_text[:800]}...</div>
        </div>"""

    for model_id, result in state.get("_lm_results", {}).items():
        text = (result or {}).get("text", "")
        if text:
            k = _lm_key(model_id)
            extra_blocks += f"""
        <div class="section" style="grid-column: span 2;">
            <h3>LM Studio: {model_id} ({len(text)} chars)</h3>
            <div class="text">{text[:800]}...</div>
        </div>"""

    if extra_blocks:
        html = html.replace("</body>", extra_blocks + "\n</body>")

    html_path = EvalConfig.RAW_OUTPUTS_DIR / state["doc_id"] / "comparison.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")
    wandb.log({f"comparison_html/{state['doc_id']}": wandb.Html(html)})


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
        wandb_project: str = EvalConfig.WANDB_PROJECT,
        wandb_run_name: Optional[str] = None,
) -> None:
    global _lm_studio_models, _lm_studio_base_url

    # ── Set module-level LM Studio config (drives dynamic column generation) ──
    _lm_studio_models = [] if (skip_lm_studio or not lm_studio_models) else lm_studio_models
    _lm_studio_base_url = lm_studio_base_url

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
        },
        tags=["talmud", "section-eval", "kraken"]
             + ([f"lm-studio:{_lm_key(m)}" for m in _lm_studio_models]),
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
                batch_num=1,
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
    """Mean CER per model/section across all documents → wandb.summary."""
    from collections import defaultdict
    accs: Dict[str, List[float]] = defaultdict(list)

    sec_cols = _section_columns()
    fp_cols  = _full_page_columns()
    sec_col  = {c: i for i, c in enumerate(sec_cols)}
    fp_col   = {c: i for i, c in enumerate(fp_cols)}

    for section in SECTION_KEYS:
        for row in _table_rows[section]:
            for model, cer_key in [
                ("vision_ocr", "vision_ocr_cer_strict"), ("vision_ocr", "vision_ocr_cer_lenient"),
                ("flash", "flash_cer_strict"), ("flash", "flash_cer_lenient"),
                ("pro",   "pro_cer_strict"),   ("pro",   "pro_cer_lenient"),
            ]:
                idx = sec_col.get(cer_key)
                if idx is not None:
                    v = row[idx]
                    if v is not None:
                        accs[f"mean/{model}/{section}/{cer_key.split('_', 1)[1]}"].append(v)

            # LM Studio per-section means
            for model_id in _lm_studio_models:
                k = _lm_key(model_id)
                for suffix in ("cer_strict", "cer_lenient"):
                    col_name = f"{k}_{suffix}"
                    idx = sec_col.get(col_name)
                    if idx is not None:
                        v = row[idx]
                        if v is not None:
                            accs[f"mean/{k}/{section}/{suffix}"].append(v)

    for row in _table_rows["full_page"]:
        for col_key in ("vision_ocr_cer_strict", "vision_ocr_cer_lenient",
                        "kraken_cer_strict", "kraken_cer_lenient"):
            idx = fp_col.get(col_key)
            if idx is not None:
                v = row[idx]
                if v is not None:
                    model_prefix = "vision_ocr" if col_key.startswith("vision") else "kraken"
                    metric_suffix = col_key.replace(f"{model_prefix}_", "")
                    accs[f"mean/{model_prefix}/overall/{metric_suffix}"].append(v)

        for model_id in _lm_studio_models:
            k = _lm_key(model_id)
            for suffix in ("cer_strict", "cer_lenient"):
                col_name = f"{k}_{suffix}"
                idx = fp_col.get(col_name)
                if idx is not None:
                    v = row[idx]
                    if v is not None:
                        accs[f"mean/{k}/overall/{suffix}"].append(v)

    summary = {k: sum(v) / len(v) for k, v in accs.items()}
    summary["num_documents"] = len(results)

    print("\nAggregate summary:")
    for k, v in sorted(summary.items()):
        wandb.summary[k] = v
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


# ============================================================================
# CLI
# ============================================================================
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

    # LM Studio: comma-separated list of model IDs exactly as shown in LM Studio
    p.add_argument(
        "--lm_studio_models",
        type=lambda s: [m.strip() for m in s.split(",") if m.strip()],
        default=["qwen/qwen3-vl-8b"],
        metavar="MODEL[,MODEL,...]",
        help="LM Studio model IDs to include.  Comma-separated. "
             'Example: "qwen/qwen3-vl-8b,llava/llava-1.6-mistral-7b"',
    )
    p.add_argument(
        "--lm_studio_base_url",
        default="http://localhost:1234/v1",
        help="LM Studio API base URL (default: http://localhost:1234/v1)",
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
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    ))