"""
talmud_eval.py
Section-aware Talmud evaluation — extends genizah_fragment_eval.py.

Adds:
  - Ground truth parsing from --- Section --- delimited text files
  - Section-aware Gemini prompts (JSON output: gemara / rashi / tosafot)
  - Kraken / MiDRASH as a fourth model in the benchmark
  - Per-section CER logged to W&B alongside the existing per-model metrics
  - Levenshtein-based CER (standard for HTR papers) alongside the existing
    SequenceMatcher CER so results are directly comparable to OCRBench / MiDRASH paper

Reuses from genizah_fragment_eval.py:
  BatchManager, EvalConfig, save_raw_output, save_incremental_result,
  evaluate_transcription, log_to_wandb (for base metrics), create_comparison_html

Run:
    python talmud_eval.py --max_documents 5 --skip_pro   # smoke test
    python talmud_eval.py                                 # full run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import wandb

# Reuse the existing eval infrastructure
from src.datasets.evaluations.fragment_evals  import (
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

# New modules
from src.datasets.document_models.talmud_gt_parser import (
    load_gt_directory,
    strip_nikud,
    normalize_whitespace,
    get_section_summary,
    SECTION_KEYS,           # ('gemara', 'rashi', 'tosafot')
)

from src.models.ocr.kraken_transcriber import preload_kraken_model, transcribe_with_kraken

# ============================================================================
# Talmud-specific config (extends EvalConfig without modifying it)
# ============================================================================

TALMUD_IMAGES_DIR = Path(
    "/Users/isaac1/Documents/GitHub/multimodal-document-analysis"
    "/src/datasets/raw_data/talmud/bavli/talmud_complete/converted_images"
)
TALMUD_GT_DIR = Path(
    "/Users/isaac1/Documents/GitHub/multimodal-document-analysis"
    "/src/datasets/raw_data/talmud/bavli/talmud_complete/texts"
)
TALMUD_RESULTS_DIR = Path("./talmud_results")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

# ============================================================================
# Per-section W&B tables
#
# One table per Talmud section (gemara / rashi / tosafot) plus a full-page
# table that includes Kraken.  Rows accumulate in module-level lists and the
# full table is re-logged after every document so W&B shows live updates.
#
# Column layout per section table:
#   doc_id | ground_truth | flash | pro
#   | flash_cer_strict | flash_cer_lenient | pro_cer_strict | pro_cer_lenient
#
# Full-page table (whole-document concat, Kraken included):
#   doc_id | ground_truth | flash | pro | kraken
#   | flash_cer | pro_cer | kraken_cer_strict | kraken_cer_lenient
# ============================================================================

_MAX_CELL_CHARS = 3000  # W&B renders long strings slowly; cap per cell

_SECTION_COLUMNS = [
    "doc_id",
    "ground_truth",
    "gemini_flash",
    "gemini_pro",
    "flash_cer_strict",
    "flash_cer_lenient",
    "pro_cer_strict",
    "pro_cer_lenient",
    "gt_chars",
    "flash_chars",
    "pro_chars",
]

_FULL_PAGE_COLUMNS = [
    "doc_id",
    "ground_truth",
    "gemini_flash",
    "gemini_pro",
    "kraken",
    "flash_cer_strict",
    "flash_cer_lenient",
    "pro_cer_strict",
    "pro_cer_lenient",
    "kraken_cer_strict",
    "kraken_cer_lenient",
    "gt_chars",
    "flash_chars",
    "pro_chars",
    "kraken_chars",
]

# Accumulated rows — keyed by section name; "full_page" for the fourth table
_table_rows: Dict[str, List[List]] = {
    "gemara": [],
    "rashi": [],
    "tosafot": [],
    "full_page": [],
}


def _cell(text: str) -> str:
    """Truncate text for W&B table cell with a length indicator."""
    if len(text) <= _MAX_CELL_CHARS:
        return text
    return text[:_MAX_CELL_CHARS] + f"\n… [{len(text)} chars total]"


def _append_section_table_rows(
        doc_id: str,
        gt_sections: Dict[str, str],
        flash_sections: Optional[Dict[str, str]],
        pro_sections: Optional[Dict[str, str]],
        flash_section_metrics: Optional[Dict],
        pro_section_metrics: Optional[Dict],
        kraken_text: Optional[str],
        kraken_metrics: Optional[Dict],
        full_gt: str,
        flash_full: str,
        pro_full: str,
) -> None:
    """Build one row per section table and append to module-level accumulators."""

    def _cer(metrics, section, key):
        if not metrics:
            return None
        return (metrics.get(section) or {}).get(key)

    # ── Per-section rows (gemara, rashi, tosafot) ──────────────────────────
    for section in SECTION_KEYS:
        gt_text = gt_sections.get(section, "")
        fl_text = (flash_sections or {}).get(section, "")
        pro_text = (pro_sections or {}).get(section, "")

        row = [
            doc_id,
            _cell(gt_text),
            _cell(fl_text),
            _cell(pro_text),
            _cer(flash_section_metrics, section, "cer_strict"),
            _cer(flash_section_metrics, section, "cer_lenient"),
            _cer(pro_section_metrics, section, "cer_strict"),
            _cer(pro_section_metrics, section, "cer_lenient"),
            len(gt_text),
            len(fl_text),
            len(pro_text),
        ]
        _table_rows[section].append(row)

    # ── Full-page row (Kraken lives here) ─────────────────────────────────
    km = kraken_metrics or {}
    full_row = [
        doc_id,
        _cell(full_gt),
        _cell(flash_full),
        _cell(pro_full),
        _cell(kraken_text or ""),
        _cer(flash_section_metrics, "overall", "cer_strict"),
        _cer(flash_section_metrics, "overall", "cer_lenient"),
        _cer(pro_section_metrics, "overall", "cer_strict"),
        _cer(pro_section_metrics, "overall", "cer_lenient"),
        km.get("cer_levenshtein"),
        km.get("cer_levenshtein_lenient"),
        len(full_gt),
        len(flash_full),
        len(pro_full),
        len(kraken_text or ""),
    ]
    _table_rows["full_page"].append(full_row)


def log_section_tables(step: int) -> None:
    """Re-log all four section tables to W&B with the current accumulated rows.

    Called after every document so the tables update live in the W&B dashboard.
    Table names are stable across calls so W&B replaces rather than duplicates.
    """
    for section in SECTION_KEYS:
        rows = _table_rows[section]
        if rows:
            wandb.log(
                {f"transcriptions/{section}": wandb.Table(
                    columns=_SECTION_COLUMNS,
                    data=rows,
                )},
                step=step,
            )

    full_rows = _table_rows["full_page"]
    if full_rows:
        wandb.log(
            {"transcriptions/full_page": wandb.Table(
                columns=_FULL_PAGE_COLUMNS,
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
    paper conventions. The existing calculate_cer() uses SequenceMatcher which
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


def _parse_section_json(raw: str, model_label: str = "") -> Optional[Dict[str, str]]:
    """Extract {gemara, rashi, tosafot} from a VLM response.

    Tries four strategies in order:
      1. JSON inside markdown fences (```json ... ```)
      2. First bare JSON object in the response
      3. Case-insensitive key matching (model used "Gemara" instead of "gemara")
      4. Plain-text fallback: look for labelled sections (--- Gemara --- / **Gemara:**)

    Always logs the raw response on failure so you can see exactly what came back.
    """
    if not raw:
        return None

    # Canonical aliases — covers common model variations
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
        """Map whatever keys the model used to our canonical set."""
        out: Dict[str, str] = {}
        for raw_key, val in d.items():
            canonical = _KEY_ALIASES.get(raw_key.strip().lower())
            if canonical and canonical not in out:
                out[canonical] = str(val).strip()
        # Accept result even if only some sections are present
        return out if out else None

    # ── Strategy 1: entire response is valid JSON ─────────────────────────────
    # Try this first — it's the happy path and handles escaped chars correctly.
    # Regex strategies below fail on `\"` inside Hebrew string values.
    try:
        data = json.loads(raw.strip())
        if isinstance(data, dict):
            result = _normalize_keys(data)
            if result:
                return {k: result.get(k, "") for k in SECTION_KEYS}
    except json.JSONDecodeError:
        pass

    # ── Strategy 2: JSON inside markdown fences ───────────────────────────────
    fence_m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence_m:
        try:
            data = json.loads(fence_m.group(1))
            result = _normalize_keys(data)
            if result:
                return {k: result.get(k, "") for k in SECTION_KEYS}
        except json.JSONDecodeError:
            pass

    # ── Strategy 3: Largest JSON object in the response ───────────────────────
    # Find ALL {...} spans and try the longest one (avoids tiny nested objects).
    # Note: this regex cannot handle escaped quotes inside string values, which
    # is why Strategy 1 runs first for clean JSON responses.
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

    # ── Strategy 4: Plain-text fallback ──────────────────────────────────────
    # Handles responses like:
    #   **Gemara:**\ntext...\n\n**Rashi:**\ntext...
    #   --- GEMARA ---\ntext...\n--- RASHI ---
    section_pattern = re.compile(
        r"(?:^|\n)\s*(?:\*\*|###|---\s*)?"  # optional markdown/delimiter
        r"(gemara|rashi|tosafot|tosafos|גמרא|רשי|תוספות)"
        r"(?:\s*\(.*?\))?"  # optional Hebrew in parens
        r"(?:\*\*|:|\s*---)*\s*\n"  # closing marker + newline
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

    # ── All strategies failed — log what we got so you can diagnose it ────────
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
    """CER per section + weighted overall.

    Returns {section: {cer_strict, cer_lenient}} for each section present in GT,
    plus an 'overall' key weighted by GT character count.
    """
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
            "cer_strict": cer_levenshtein("".join(total_hyp_chars), "".join(total_ref_chars)),
            "cer_lenient": cer_levenshtein("".join(total_hyp_lenient), "".join(total_ref_lenient)),
        }
    else:
        metrics["overall"] = {"cer_strict": 1.0, "cer_lenient": 1.0}

    return metrics


def _wandb_section_payload(model: str, metrics: Dict) -> Dict[str, float]:
    """Flatten section metrics to W&B log dict."""
    out = {}
    for section, vals in metrics.items():
        for k, v in vals.items():
            if v is not None:
                out[f"{model}/{section}/{k}"] = v
    return out


# ============================================================================
# Talmud VLM calls
# ============================================================================

async def _call_section_aware(model_name: str, image_path: str, prompt: str, timeout: int):
    t0 = time.time()
    raw = await call_gemini_with_retry(model_name, image_path, prompt, temperature=0.1, timeout=timeout)
    label = model_name.split("-")[0]  # e.g. "gemini" — keep it short
    return _parse_section_json(raw, model_label=label), time.time() - t0


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
        batch_num: Optional[int] = None,
) -> Optional[Dict]:
    """
    Run all models on one Talmud page and return a result dict compatible
    with the existing W&B table schema, plus extended section metrics.
    """
    print(f"\n📄 {doc_id}")
    print(get_section_summary(gt_sections))

    # Concatenated GT for whole-document metrics (used for Kraken + existing log_to_wandb)
    full_gt = "\n".join(gt_sections.get(k, "") for k in SECTION_KEYS if gt_sections.get(k))
    if not full_gt.strip():
        print(f"  ⚠️  No GT text — gt_sections keys: {list(gt_sections.keys())}, SECTION_KEYS: {SECTION_KEYS}")
        return None

    # Build a minimal TranscriptionState stub so log_to_wandb can be reused
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
        # Talmud-specific additions
        "_flash_sections": None,
        "_pro_sections": None,
        "_flash_section_metrics": None,
        "_pro_section_metrics": None,
    }

    t_doc_start = time.time()

    # ── Gemini Flash ─────────────────────────────────────────────────────────
    if not skip_flash:
        print("  ⚡ Gemini Flash (section-aware)...")
        sections, elapsed = await _call_section_aware(
            AgentConfig.GEMINI_FLASH_MODEL, str(image_path),
            FLASH_PROMPT, AgentConfig.GEMINI_FLASH_TIMEOUT,
        )
        state["model_times"]["gemini_flash"] = elapsed

        if sections:
            reconstructed = "\n".join(sections.get(k, "") for k in SECTION_KEYS if sections.get(k))
            state["gemini_flash_result"] = {
                "text": reconstructed, "model": "gemini_flash_talmud_section",
                "char_count": len(reconstructed), "processing_time": elapsed,
            }
            state["gemini_flash_metrics"] = evaluate_transcription(full_gt, reconstructed, "gemini_flash")
            state["gemini_flash_metrics"]["processing_time"] = elapsed

            section_m = compute_section_metrics(sections, gt_sections)
            state["_flash_sections"] = sections
            state["_flash_section_metrics"] = section_m

            print(f"    ✓ overall CER strict={section_m['overall']['cer_strict']:.3f} "
                  f"lenient={section_m['overall']['cer_lenient']:.3f}")
        else:
            pass  # _parse_section_json already printed the raw response

    # ── Gemini Pro (after a short pause) ─────────────────────────────────────
    if not skip_pro:
        await asyncio.sleep(3)
        print("  🎯 Gemini Pro (section-aware)...")
        sections, elapsed = await _call_section_aware(
            AgentConfig.GEMINI_PRO_MODEL, str(image_path),
            PRO_PROMPT, AgentConfig.GEMINI_PRO_TIMEOUT,
        )
        state["model_times"]["gemini_pro"] = elapsed

        if sections:
            reconstructed = "\n".join(sections.get(k, "") for k in SECTION_KEYS if sections.get(k))
            state["gemini_pro_result"] = {
                "text": reconstructed, "model": "gemini_pro_talmud_section",
                "char_count": len(reconstructed), "processing_time": elapsed,
            }
            state["gemini_pro_metrics"] = evaluate_transcription(full_gt, reconstructed, "gemini_pro")
            state["gemini_pro_metrics"]["processing_time"] = elapsed

            section_m = compute_section_metrics(sections, gt_sections)
            state["_pro_sections"] = sections
            state["_pro_section_metrics"] = section_m

            # Pro output drives consensus transcription
            state["final_transcription"] = reconstructed
            state["consensus_metrics"] = evaluate_transcription(
                full_gt, reconstructed, "consensus_pro_preferred"
            )

            print(f"    ✓ overall CER strict={section_m['overall']['cer_strict']:.3f} "
                  f"lenient={section_m['overall']['cer_lenient']:.3f}")
        else:
            pass  # _parse_section_json already printed the raw response

    # Fallback consensus to Flash if Pro failed
    if not state["final_transcription"] and state.get("gemini_flash_result"):
        state["final_transcription"] = state["gemini_flash_result"]["text"]
        state["consensus_strategy"] = "flash_fallback"
        state["consensus_metrics"] = state["gemini_flash_metrics"]

    # ── Kraken ───────────────────────────────────────────────────────────────
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
            # Reuse existing evaluate_transcription for SequenceMatcher metrics
            state["kraken_metrics"] = evaluate_transcription(full_gt, kraken_text, "kraken")
            state["kraken_metrics"]["processing_time"] = elapsed
            # Also compute proper Levenshtein CER for the paper
            state["kraken_metrics"]["cer_levenshtein"], state["kraken_metrics"]["cer_levenshtein_lenient"] = \
                cer_pair(kraken_text, full_gt)

            print(f"    ✓ {len(kraken_text)} chars  "
                  f"CER strict={state['kraken_metrics']['cer_levenshtein']:.3f}  "
                  f"lenient={state['kraken_metrics']['cer_levenshtein_lenient']:.3f}")
        else:
            print("    ✗ Kraken failed")

    state["processing_time"] = time.time() - t_doc_start

    # ── W&B logging ──────────────────────────────────────────────────────────
    # Step counter = total rows logged so far (used so W&B time-series is ordered)
    _step = len(_table_rows["full_page"])  # before we append the new row

    flash_sections = state.get("_flash_sections") or {}
    pro_sections = state.get("_pro_sections") or {}
    flash_full = "\n".join(flash_sections.get(k, "") for k in SECTION_KEYS)
    pro_full = "\n".join(pro_sections.get(k, "") for k in SECTION_KEYS)
    kraken_text = (state.get("kraken_result") or {}).get("text", "")
    kraken_metrics = state.get("kraken_metrics")

    # 1. Append to the four section tables and immediately re-log them so
    #    W&B shows live updates after every document.
    _append_section_table_rows(
        doc_id=doc_id,
        gt_sections=gt_sections,
        flash_sections=flash_sections or None,
        pro_sections=pro_sections or None,
        flash_section_metrics=state.get("_flash_section_metrics"),
        pro_section_metrics=state.get("_pro_section_metrics"),
        kraken_text=kraken_text or None,
        kraken_metrics=kraken_metrics,
        full_gt=full_gt,
        flash_full=flash_full,
        pro_full=pro_full,
    )
    log_section_tables(step=_step)

    # 2. Scalar metrics (CER timeseries + timing) — kept for chart views
    scalar_payload: Dict = {"doc_id": doc_id}

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
    for model_key in ("gemini_flash", "gemini_pro", "kraken"):
        t = state["model_times"].get(model_key)
        if t is not None:
            scalar_payload[f"timing/{model_key}_s"] = t

    wandb.log({k: v for k, v in scalar_payload.items() if v is not None}, step=_step)

    # 3. HTML comparison (kept for quick visual spot-checks)
    if Path(state["image_path"]).exists():
        _update_comparison_html_with_kraken(state)

    # 4. Raw text files + incremental JSONL
    for model_key, label in [
        ("gemini_flash_result", "gemini_flash"),
        ("gemini_pro_result", "gemini_pro"),
        ("kraken_result", "kraken"),
    ]:
        r = state.get(model_key)
        if r:
            save_raw_output(doc_id, label, r["text"])

    save_raw_output(doc_id, "ground_truth", full_gt, full_gt)
    save_incremental_result(state, batch_num)

    return state


def _update_comparison_html_with_kraken(state: Dict) -> None:
    """Extend the existing comparison HTML to include a Kraken column."""
    kraken_text = (state.get("kraken_result") or {}).get("text", "")
    if not kraken_text:
        return

    html = create_comparison_html(
        state["doc_id"],
        state.get("ground_truth", ""),
        "",  # Vision OCR (not run)
        (state.get("gemini_flash_result") or {}).get("text", ""),
        (state.get("gemini_pro_result") or {}).get("text", ""),
        state["final_transcription"],
        state["consensus_strategy"],
    )

    # Inject Kraken section before </body>
    kraken_block = f"""
        <div class="section" style="grid-column: span 2;">
            <h3>Kraken / MiDRASH Gen 01 ({len(kraken_text)} chars)</h3>
            <div class="text">{kraken_text[:800]}...</div>
        </div>"""
    html = html.replace("</body>", kraken_block + "\n</body>")

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
        max_documents: Optional[int] = None,
        skip_flash: bool = False,
        skip_pro: bool = False,
        skip_kraken: bool = False,
        wandb_project: str = EvalConfig.WANDB_PROJECT,
        wandb_run_name: Optional[str] = None,
) -> None:
    TALMUD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EvalConfig.RAW_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Match images ↔ GT by filename stem.
    #
    # Image names follow the pattern:  01_2_page_001.png
    # GT stems follow the pattern:     01_2
    #
    # Strategy: strip a trailing _page_NNN (or _NNN) suffix from the image stem
    # to recover the GT key, then group all pages of the same GT document together.
    # When multiple pages exist for one GT file we use all of them independently
    # (each page gets its own eval row keyed as "01_2_page_001", "01_2_page_002", …).
    _page_suffix_re = re.compile(r"_page_\d+$", re.IGNORECASE)

    gt_by_stem = load_gt_directory(gt_dir)
    image_map: Dict[str, Path] = {}  # doc_id (image stem) → image path

    for ext in IMAGE_EXTENSIONS:
        for p in images_dir.glob(f"*{ext}"):
            # Try exact match first (stem == GT key)
            if p.stem in gt_by_stem:
                image_map[p.stem] = p
                continue
            # Strip _page_NNN suffix and try again
            gt_key = _page_suffix_re.sub("", p.stem)
            if gt_key in gt_by_stem:
                image_map[p.stem] = p  # keep full stem as unique doc_id

    doc_ids = sorted(image_map.keys())
    if not doc_ids:
        raise RuntimeError(
            f"No images in {images_dir} matched GT stems in {gt_dir}.\n"
            f"GT stems found: {sorted(gt_by_stem.keys())}\n"
            f"Example image stems: "
            + ", ".join(
                p.stem for ext in IMAGE_EXTENSIONS
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
        },
        tags=["talmud", "section-eval", "kraken"],
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
                skip_flash=skip_flash,
                skip_pro=skip_pro,
                skip_kraken=skip_kraken,
                batch_num=1,
            )
            if result:
                all_results.append(result)

        except Exception as exc:
            print(f"  ❌ {doc_id}: {exc}")
            wandb.log({"error": str(exc), "error_doc": doc_id})

        if i < len(doc_ids) - 1:
            await asyncio.sleep(AgentConfig.DELAY_BETWEEN_DOCS)

    # Section tables are already live in W&B (re-logged after every document).
    # Just log the aggregate summary to wandb.summary and finish.
    _log_summary(all_results)
    wandb.finish()


def _log_summary(results: List[Dict]) -> None:
    """Compute mean CER per model/section across all documents and write to wandb.summary."""
    from collections import defaultdict
    accs: Dict[str, List[float]] = defaultdict(list)

    # Pull from the accumulated table rows (columns are stable, index them by name)
    sec_col = {c: i for i, c in enumerate(_SECTION_COLUMNS)}
    fp_col = {c: i for i, c in enumerate(_FULL_PAGE_COLUMNS)}

    for section in SECTION_KEYS:
        for row in _table_rows[section]:
            for model, cer_key in [("flash", "flash_cer_strict"), ("flash", "flash_cer_lenient"),
                                   ("pro", "pro_cer_strict"), ("pro", "pro_cer_lenient")]:
                v = row[sec_col[cer_key]]
                if v is not None:
                    accs[f"mean/{model}/{section}/{cer_key.split('_', 1)[1]}"].append(v)

    for row in _table_rows["full_page"]:
        for col_key in ("kraken_cer_strict", "kraken_cer_lenient"):
            v = row[fp_col[col_key]]
            if v is not None:
                accs[f"mean/kraken/overall/{col_key.replace('kraken_', '')}"].append(v)

    summary = {k: sum(v) / len(v) for k, v in accs.items()}
    summary["num_documents"] = len(results)

    print("\nAggregate summary:")
    for k, v in sorted(summary.items()):
        wandb.summary[k] = v
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", type=Path, default=TALMUD_IMAGES_DIR)
    p.add_argument("--gt_dir", type=Path, default=TALMUD_GT_DIR)
    p.add_argument("--kraken_model", default="MiDRASH_Gen_01.mlmodel")
    p.add_argument("--max_documents", type=int, default=None)
    p.add_argument("--wandb_project", default=EvalConfig.WANDB_PROJECT)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--skip_flash", action="store_true")
    p.add_argument("--skip_pro", action="store_true")
    p.add_argument("--skip_kraken", action="store_true")
    args = p.parse_args()

    asyncio.run(run_talmud_eval(
        images_dir=args.images_dir,
        gt_dir=args.gt_dir,
        kraken_model_path="/Users/isaac1/Documents/historical-document-analysis/src/datasets/raw_data/cairo_genizah/custom_model_weights/MiDRASH_Gen_01.mlmodel",
        max_documents=args.max_documents,
        skip_flash=args.skip_flash,
        skip_pro=args.skip_pro,
        skip_kraken=args.skip_kraken,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    ))