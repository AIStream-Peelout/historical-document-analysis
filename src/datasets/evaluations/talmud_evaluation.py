"""
talmud_eval.py
Section-aware evaluation runner for Talmud pages.

Adds two models to the existing pipeline:
  1. Kraken/MiDRASH — domain-specific HTR model (whole-page CER)
  2. Section-aware Gemini calls — Flash + Pro prompted to output JSON
     with gemara / rashi / tosafot keys, enabling per-section CER

W&B metric key structure:
    {model}/{section}/cer_strict    — CER computed with nikud included
    {model}/{section}/cer_lenient   — CER computed after stripping nikud
    {model}/overall/cer_strict
    {model}/overall/cer_lenient
    kraken/overall/cer_strict       — Kraken is whole-page only (no layout split)
    kraken/overall/cer_lenient

Run directly:
    python talmud_eval.py \
        --images_dir .../converted_images \
        --gt_dir     .../texts \
        --kraken_model MiDRASH_Gen_01.mlmodel \
        --wandb_project cairo-genizah-vlm-eval

Or import and call run_talmud_evaluation() programmatically.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import wandb

# Internal imports (same package)
from src.models.llm.genizah_fragment_agent import AgentConfig, call_gemini_with_retry
from src.models.ocr.kraken_transcriber import preload_kraken_model, transcribe_with_kraken
from src.datasets.document_models.talmud_gt_parser import (
    load_gt_directory,
    strip_nikud,
    normalize_whitespace,
    get_section_summary,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# CER / WER utilities                                                          #
# --------------------------------------------------------------------------- #

def _edit_distance(a: str, b: str) -> int:
    """Standard Levenshtein edit distance — O(len(a)*len(b))."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def compute_cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate = edit_distance(hyp, ref) / len(ref).

    Returns 1.0 (100 % error) when reference is empty to avoid ZeroDivisionError.
    """
    ref = normalize_whitespace(reference)
    hyp = normalize_whitespace(hypothesis)
    if not ref:
        return 1.0
    return min(_edit_distance(hyp, ref) / len(ref), 1.0)


def compute_cer_pair(hypothesis: str, reference: str) -> Tuple[float, float]:
    """Return (cer_strict, cer_lenient) — with and without nikud."""
    cer_strict = compute_cer(hypothesis, reference)
    cer_lenient = compute_cer(strip_nikud(hypothesis), strip_nikud(reference))
    return cer_strict, cer_lenient


# --------------------------------------------------------------------------- #
# Section-aware Gemini prompts                                                 #
# --------------------------------------------------------------------------- #

_TALMUD_LAYOUT_DESCRIPTION = """
This is a page from the Babylonian Talmud (תלמוד בבלי). The page is divided into 
three distinct textual sections arranged spatially:

1. GEMARA (גמרא) — The main Talmudic text in the CENTER of the page, printed in 
   larger square Hebrew/Aramaic script.
2. RASHI (רש"י) — Rabbi Shlomo Yitzchaki's commentary, printed in the INNER MARGIN 
   (typically the right margin), in a distinctive smaller semi-cursive Rashi script.
3. TOSAFOT (תוספות) — Later medieval commentary, printed in the OUTER MARGIN 
   (typically the left margin), in smaller square script.
"""

_SECTION_JSON_SCHEMA = """\
Return ONLY a JSON object (no markdown, no commentary) with this exact structure:
{
    "gemara": "<full gemara transcription>",
    "rashi": "<full rashi transcription>",
    "tosafot": "<full tosafot transcription>"
}
If a section is absent or illegible, use an empty string for that key.
"""

TALMUD_FLASH_PROMPT = f"""{_TALMUD_LAYOUT_DESCRIPTION}

TRANSCRIPTION INSTRUCTIONS:
1. Identify each of the three sections by its physical location on the page.
2. Transcribe each section EXACTLY as it appears — character by character.
3. Do NOT normalize, correct, or complete text from memory.
4. Preserve vocalization marks (nikud) where visible.
5. Mark unclear characters with [?].
6. Preserve line breaks within each section using \\n.

{_SECTION_JSON_SCHEMA}"""

TALMUD_PRO_PROMPT = f"""{_TALMUD_LAYOUT_DESCRIPTION}

TRANSCRIPTION INSTRUCTIONS:
Think step-by-step:
  Step 1 — Identify the three spatial regions of the page.
  Step 2 — For each region, determine the script style (square vs. Rashi script) to confirm 
            which commentary it contains.
  Step 3 — Transcribe EXACTLY what is visible in each region.
            • Do NOT draw on memorized versions of the Talmud — textual variants matter.
            • Do NOT fill in damaged or obscured text.
            • Mark ambiguous characters with [?].
            • Preserve nikud where present.
            • Preserve line structure within each section.

{_SECTION_JSON_SCHEMA}"""


# --------------------------------------------------------------------------- #
# Section JSON parser                                                          #
# --------------------------------------------------------------------------- #

def parse_section_json(response_text: str) -> Optional[Dict[str, str]]:
    """Extract and validate the section JSON from a VLM response.

    Handles:
      - Clean JSON response
      - JSON wrapped in markdown code fences (```json ... ```)
      - Missing keys (fills with empty string)

    Returns None if parsing fails entirely.
    """
    if not response_text:
        return None

    # Strip markdown fences if present
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    raw_json = fence_match.group(1) if fence_match else response_text.strip()

    # Attempt to isolate a JSON object if there's surrounding prose
    obj_match = re.search(r"\{.*\}", raw_json, re.DOTALL)
    if obj_match:
        raw_json = obj_match.group(0)

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        logger.warning(f"JSON parse failed: {exc}. Raw: {raw_json[:300]}")
        return None

    # Normalize keys and fill missing sections
    normalized: Dict[str, str] = {}
    for key in ("gemara", "rashi", "tosafot"):
        normalized[key] = str(data.get(key, "")).strip()

    return normalized


# --------------------------------------------------------------------------- #
# Per-section metrics computation                                              #
# --------------------------------------------------------------------------- #

SECTION_KEYS = ("gemara", "rashi", "tosafot")


def compute_section_metrics(
    section_transcriptions: Dict[str, str],
    gt_sections: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    """Compute CER per section and overall.

    Args:
        section_transcriptions: {'gemara': '...', 'rashi': '...', 'tosafot': '...'}
        gt_sections: same structure from ground truth

    Returns:
        {
            'gemara':  {'cer_strict': float, 'cer_lenient': float},
            'rashi':   {...},
            'tosafot': {...},
            'overall': {'cer_strict': float, 'cer_lenient': float},
        }
    """
    metrics: Dict[str, Dict[str, float]] = {}

    all_hyp_chars: List[str] = []
    all_ref_chars: List[str] = []
    all_hyp_chars_lenient: List[str] = []
    all_ref_chars_lenient: List[str] = []

    for section in SECTION_KEYS:
        hyp = normalize_whitespace(section_transcriptions.get(section, ""))
        ref = normalize_whitespace(gt_sections.get(section, ""))

        if not ref:
            # Section absent in GT — skip from per-section metrics but note it
            metrics[section] = {"cer_strict": None, "cer_lenient": None, "skipped": True}
            continue

        cer_strict, cer_lenient = compute_cer_pair(hyp, ref)
        metrics[section] = {"cer_strict": cer_strict, "cer_lenient": cer_lenient}

        all_hyp_chars.extend(list(hyp))
        all_ref_chars.extend(list(ref))
        all_hyp_chars_lenient.extend(list(strip_nikud(hyp)))
        all_ref_chars_lenient.extend(list(strip_nikud(ref)))

    # Overall CER computed over concatenated chars (weighted by section length)
    if all_ref_chars:
        overall_strict = compute_cer("".join(all_hyp_chars), "".join(all_ref_chars))
        overall_lenient = compute_cer(
            "".join(all_hyp_chars_lenient), "".join(all_ref_chars_lenient)
        )
    else:
        overall_strict = overall_lenient = 1.0

    metrics["overall"] = {"cer_strict": overall_strict, "cer_lenient": overall_lenient}
    return metrics


# --------------------------------------------------------------------------- #
# W&B logging helpers                                                          #
# --------------------------------------------------------------------------- #

def _flatten_metrics_for_wandb(
    model_name: str, metrics: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """Convert nested metrics dict to flat W&B-compatible key/value pairs.

    e.g.  flash/gemara/cer_strict = 0.12
    """
    flat: Dict[str, float] = {}
    for section, values in metrics.items():
        for metric_name, value in values.items():
            if value is not None and metric_name != "skipped":
                flat[f"{model_name}/{section}/{metric_name}"] = value
    return flat


def log_document_to_wandb(
    doc_id: str,
    step: int,
    model_results: Dict[str, Dict],  # model_name → {section_metrics, raw_text, ...}
) -> None:
    """Log per-document metrics to the active W&B run."""
    log_payload: Dict[str, object] = {"doc_id": doc_id, "step": step}

    for model_name, result in model_results.items():
        section_metrics = result.get("section_metrics")
        if section_metrics:
            log_payload.update(_flatten_metrics_for_wandb(model_name, section_metrics))

        # Also log character counts for diagnostics
        for section in SECTION_KEYS:
            raw = result.get("sections", {}).get(section, "")
            log_payload[f"{model_name}/{section}/char_count"] = len(raw)

        processing_time = result.get("processing_time")
        if processing_time is not None:
            log_payload[f"{model_name}/processing_time_s"] = processing_time

    wandb.log(log_payload, step=step)


# --------------------------------------------------------------------------- #
# Section-aware VLM transcription calls                                        #
# --------------------------------------------------------------------------- #

async def transcribe_talmud_flash(
    image_path: str,
) -> Tuple[Optional[Dict[str, str]], float]:
    """Run Gemini Flash on a Talmud page, requesting section JSON output.

    Returns (section_dict_or_None, elapsed_seconds).
    """
    t0 = time.time()
    raw = await call_gemini_with_retry(
        AgentConfig.GEMINI_FLASH_MODEL,
        image_path,
        TALMUD_FLASH_PROMPT,
        temperature=0.1,
        timeout=AgentConfig.GEMINI_FLASH_TIMEOUT,
    )
    elapsed = time.time() - t0
    sections = parse_section_json(raw) if raw else None
    return sections, elapsed


async def transcribe_talmud_pro(
    image_path: str,
) -> Tuple[Optional[Dict[str, str]], float]:
    """Run Gemini Pro on a Talmud page, requesting section JSON output."""
    t0 = time.time()
    raw = await call_gemini_with_retry(
        AgentConfig.GEMINI_PRO_MODEL,
        image_path,
        TALMUD_PRO_PROMPT,
        temperature=0.1,
        timeout=AgentConfig.GEMINI_PRO_TIMEOUT,
    )
    elapsed = time.time() - t0
    sections = parse_section_json(raw) if raw else None
    return sections, elapsed


async def transcribe_talmud_kraken(
    model_path: str,
    image_path: str,
) -> Tuple[Optional[str], float]:
    """Run Kraken on a Talmud page. Returns (whole_page_text, elapsed_seconds).

    NOTE: Kraken produces a single text stream — it does not separate sections.
    We log whole-page CER only. The physical layout (center vs. margins) could
    be used in future work to classify lines by section using baseline x-coordinates.
    """
    t0 = time.time()
    text = await transcribe_with_kraken(model_path, image_path)
    elapsed = time.time() - t0
    return text, elapsed


# --------------------------------------------------------------------------- #
# Per-document evaluation                                                      #
# --------------------------------------------------------------------------- #

async def evaluate_document(
    doc_id: str,
    image_path: Path,
    gt_sections: Dict[str, str],
    kraken_model_path: str,
    skip_flash: bool = False,
    skip_pro: bool = False,
    skip_kraken: bool = False,
) -> Dict[str, Dict]:
    """Run all models on one document and compute section-level metrics.

    Returns a dict: model_name → {sections, section_metrics, processing_time}
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Document: {doc_id}")
    logger.info(f"GT summary:\n{get_section_summary(gt_sections)}")

    results: Dict[str, Dict] = {}

    # --- Gemini Flash ---
    if not skip_flash:
        logger.info("  ⚡ Gemini Flash (section-aware)...")
        sections, elapsed = await transcribe_talmud_flash(str(image_path))
        if sections:
            section_metrics = compute_section_metrics(sections, gt_sections)
            results["flash"] = {
                "sections": sections,
                "section_metrics": section_metrics,
                "processing_time": elapsed,
            }
            _log_section_results("flash", sections, section_metrics)
        else:
            logger.warning("  ✗ Flash returned no parseable section JSON")
            results["flash"] = {"sections": {}, "section_metrics": None, "processing_time": elapsed}

    # --- Gemini Pro (sequential, with delay to avoid rate limiting) ---
    if not skip_pro:
        await asyncio.sleep(3)
        logger.info("  🎯 Gemini Pro (section-aware)...")
        sections, elapsed = await transcribe_talmud_pro(str(image_path))
        if sections:
            section_metrics = compute_section_metrics(sections, gt_sections)
            results["pro"] = {
                "sections": sections,
                "section_metrics": section_metrics,
                "processing_time": elapsed,
            }
            _log_section_results("pro", sections, section_metrics)
        else:
            logger.warning("  ✗ Pro returned no parseable section JSON")
            results["pro"] = {"sections": {}, "section_metrics": None, "processing_time": elapsed}

    # --- Kraken ---
    if not skip_kraken:
        logger.info("  🐙 Kraken / MiDRASH...")
        whole_page_text, elapsed = await transcribe_talmud_kraken(
            kraken_model_path, str(image_path)
        )
        if whole_page_text:
            # Compute CER against concatenated GT (all sections joined)
            full_gt = "\n".join(
                gt_sections.get(k, "") for k in SECTION_KEYS if gt_sections.get(k)
            )
            cer_strict, cer_lenient = compute_cer_pair(whole_page_text, full_gt)
            kraken_metrics = {
                "overall": {"cer_strict": cer_strict, "cer_lenient": cer_lenient}
            }
            results["kraken"] = {
                "sections": {"whole_page": whole_page_text},
                "section_metrics": kraken_metrics,
                "processing_time": elapsed,
            }
            logger.info(
                f"    ✓ Kraken overall CER: strict={cer_strict:.3f}  lenient={cer_lenient:.3f}"
            )
        else:
            logger.warning("  ✗ Kraken failed")
            results["kraken"] = {"sections": {}, "section_metrics": None, "processing_time": elapsed}

    return results


def _log_section_results(
    model_name: str,
    sections: Dict[str, str],
    metrics: Dict[str, Dict[str, float]],
) -> None:
    """Pretty-print per-section CER to the console."""
    for section in list(SECTION_KEYS) + ["overall"]:
        m = metrics.get(section, {})
        if m.get("skipped"):
            logger.info(f"    {model_name}/{section}: GT absent — skipped")
        elif m.get("cer_strict") is not None:
            logger.info(
                f"    {model_name}/{section}: "
                f"CER strict={m['cer_strict']:.3f}  lenient={m['cer_lenient']:.3f}  "
                f"({len(sections.get(section,''))} chars transcribed)"
            )


# --------------------------------------------------------------------------- #
# Main evaluation loop                                                         #
# --------------------------------------------------------------------------- #

async def run_talmud_evaluation(
    images_dir: Path,
    gt_dir: Path,
    kraken_model_path: str,
    output_dir: Optional[Path] = None,
    wandb_project: str = "cairo-genizah-vlm-eval",
    wandb_run_name: Optional[str] = None,
    max_documents: Optional[int] = None,
    skip_flash: bool = False,
    skip_pro: bool = False,
    skip_kraken: bool = False,
    image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff"),
) -> None:
    """
    Main evaluation loop. Matches images to GT files by filename stem, runs all
    models, computes per-section CER, and logs everything to W&B.

    Image/GT matching:
        images_dir/01_2.jpg  ←→  gt_dir/01_2.txt
        (stem must match exactly)
    """
    # --- Validate paths ---
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")

    # --- Load all ground truth ---
    logger.info(f"Loading ground truth from {gt_dir} ...")
    gt_by_stem = load_gt_directory(gt_dir)
    logger.info(f"  Found {len(gt_by_stem)} GT files: {sorted(gt_by_stem.keys())}")

    # --- Find matching images ---
    image_files: Dict[str, Path] = {}
    for ext in image_extensions:
        for img_path in images_dir.glob(f"*{ext}"):
            stem = img_path.stem
            if stem in gt_by_stem:
                image_files[stem] = img_path

    if not image_files:
        raise RuntimeError(
            f"No images in {images_dir} matched GT stems {sorted(gt_by_stem.keys())}"
        )

    doc_ids = sorted(image_files.keys())
    if max_documents:
        doc_ids = doc_ids[:max_documents]

    logger.info(f"Matched {len(doc_ids)} documents for evaluation: {doc_ids}")

    # --- Preload Kraken model (amortize cost across all documents) ---
    if not skip_kraken:
        logger.info(f"Preloading Kraken model: {kraken_model_path}")
        preload_kraken_model(kraken_model_path)

    # --- Initialize W&B ---
    run = wandb.init(
        project=wandb_project,
        name=wandb_run_name or f"talmud-eval-{time.strftime('%Y%m%d-%H%M%S')}",
        config={
            "eval_type": "talmud_section_aware",
            "num_documents": len(doc_ids),
            "doc_ids": doc_ids,
            "kraken_model": kraken_model_path if not skip_kraken else None,
            "gemini_flash_model": AgentConfig.GEMINI_FLASH_MODEL,
            "gemini_pro_model": AgentConfig.GEMINI_PRO_MODEL,
            "skip_flash": skip_flash,
            "skip_pro": skip_pro,
            "skip_kraken": skip_kraken,
        },
        tags=["talmud", "section-eval"],
    )

    # --- Output dir for transcription artifacts ---
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # --- Per-document evaluation ---
    all_results: List[Dict] = []

    for step, doc_id in enumerate(doc_ids):
        image_path = image_files[doc_id]
        gt_sections = gt_by_stem[doc_id]

        try:
            model_results = await evaluate_document(
                doc_id=doc_id,
                image_path=image_path,
                gt_sections=gt_sections,
                kraken_model_path=kraken_model_path,
                skip_flash=skip_flash,
                skip_pro=skip_pro,
                skip_kraken=skip_kraken,
            )

            # Log to W&B
            log_document_to_wandb(doc_id=doc_id, step=step, model_results=model_results)
            all_results.append({"doc_id": doc_id, "results": model_results})

            # Optionally save transcription artifacts
            if output_dir:
                _save_document_output(output_dir, doc_id, model_results, gt_sections)

        except Exception as exc:
            logger.error(f"Failed on {doc_id}: {exc}", exc_info=True)
            wandb.log({"error": str(exc), "error_doc": doc_id}, step=step)

        # Delay between documents to respect API rate limits
        if step < len(doc_ids) - 1:
            await asyncio.sleep(AgentConfig.DELAY_BETWEEN_DOCS)

    # --- Compute and log aggregate summary ---
    summary = _compute_aggregate_summary(all_results)
    for key, value in summary.items():
        wandb.summary[key] = value

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE — AGGREGATE SUMMARY")
    for key, value in sorted(summary.items()):
        logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    wandb.finish()


def _save_document_output(
    output_dir: Path,
    doc_id: str,
    model_results: Dict[str, Dict],
    gt_sections: Dict[str, str],
) -> None:
    """Save per-document transcription results as JSON for offline inspection."""
    out = {
        "doc_id": doc_id,
        "ground_truth": gt_sections,
        "model_outputs": {
            model: {
                "sections": result.get("sections", {}),
                "section_metrics": {
                    sec: {k: v for k, v in vals.items() if k != "skipped"}
                    for sec, vals in (result.get("section_metrics") or {}).items()
                },
                "processing_time_s": result.get("processing_time"),
            }
            for model, result in model_results.items()
        },
    }
    (output_dir / f"{doc_id}_eval.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _compute_aggregate_summary(all_results: List[Dict]) -> Dict[str, float]:
    """Average per-section CER across all documents for each model."""
    from collections import defaultdict

    accumulators: Dict[str, List[float]] = defaultdict(list)

    for entry in all_results:
        for model_name, result in entry["results"].items():
            section_metrics = result.get("section_metrics") or {}
            for section, vals in section_metrics.items():
                for metric, value in vals.items():
                    if value is not None and metric != "skipped":
                        accumulators[f"{model_name}/{section}/{metric}"].append(value)

    summary: Dict[str, float] = {}
    for key, values in accumulators.items():
        summary[f"mean_{key}"] = sum(values) / len(values)

    summary["num_documents_evaluated"] = len(all_results)
    return summary


# --------------------------------------------------------------------------- #
# CLI entry point                                                              #
# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Talmud section-aware evaluation with per-section CER logged to W&B"
    )
    parser.add_argument(
        "--images_dir",
        type=Path,
        default=Path(
            "/Users/isaac1/Documents/GitHub/multimodal-document-analysis"
            "/src/datasets/raw_data/talmud/bavli/talmud_complete/converted_images"
        ),
        help="Directory containing Talmud page images",
    )
    parser.add_argument(
        "--gt_dir",
        type=Path,
        default=Path(
            "/Users/isaac1/Documents/GitHub/multimodal-document-analysis"
            "/src/datasets/raw_data/talmud/bavli/talmud_complete/texts"
        ),
        help="Directory containing ground truth .txt files",
    )
    parser.add_argument(
        "--kraken_model",
        type=str,
        default="MiDRASH_Gen_01.mlmodel",
        help="Path to the MiDRASH Kraken .mlmodel file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional directory to save per-document JSON artifacts",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="cairo-genizah-vlm-eval",
    )
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument(
        "--max_documents",
        type=int,
        default=None,
        help="Cap number of documents (useful for quick smoke tests)",
    )
    parser.add_argument("--skip_flash", action="store_true")
    parser.add_argument("--skip_pro", action="store_true")
    parser.add_argument("--skip_kraken", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(
        run_talmud_evaluation(
            images_dir=args.images_dir,
            gt_dir=args.gt_dir,
            kraken_model_path=args.kraken_model,
            output_dir=args.output_dir,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            max_documents=args.max_documents,
            skip_flash=args.skip_flash,
            skip_pro=args.skip_pro,
            skip_kraken=args.skip_kraken,
        )
    )