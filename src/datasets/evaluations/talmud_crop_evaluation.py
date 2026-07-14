"""Crop-track benchmark: pure-reading CER on the benchmark section crops.

Sibling of ``talmud_evaluation.py`` that scores models on the 195 pre-cropped
section images in ``talmud_sample/crops/`` instead of full pages. Because
each crop contains exactly one section, this isolates *reading* ability from
*layout localization* — the two skills the full-page benchmark conflates.
Reporting both tracks before/after fine-tuning shows where a model's errors
come from.

Only LM Studio models are supported here (the crop track exists to measure
local fine-tunes; cloud models are already scored on the full-page track).

Usage
-----
  python -m src.datasets.evaluations.talmud_crop_evaluation \\
      --lm_studio_models qwen/qwen3-vl-8b,<finetuned-id> [--max_pages 10] [--wandb]
"""

import argparse
import asyncio
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from src.datasets.document_models.talmud_gt_parser import load_gt_directory
from src.datasets.evaluations.metrics import (
    cer_pair,
    char_count_ratio,
    flag_failure_modes,
    normalize_whitespace,
)
from src.finetuning.qwen_hebrew.prompts import CROP_TRANSCRIBE_PROMPTS, SECTIONS
from src.models.ocr.lms_transcriber import (
    check_lm_studio_health,
    transcribe_with_lm_studio,
)

_REPO = Path(__file__).resolve().parents[3]
SAMPLE_DIR = _REPO / "src/datasets/raw_data/cairo_genizah/evaluations/talmud_sample"
CROPS_DIR = SAMPLE_DIR / "crops"
TEXTS_DIR = SAMPLE_DIR / "texts"
RESULTS_PATH = Path(__file__).parent / "transcription_results/crop_track_results.jsonl"

MAX_TOKENS = 4500


def collect_crop_jobs(max_pages: Optional[int] = None) -> List[Dict]:
    """Pair benchmark crops with their GT sections.

    :param max_pages: Optional cap on the number of pages.
    :returns: List of jobs (``stem``, ``section``, ``crop_path``, ``gt``).
    """
    gt_by_stem = load_gt_directory(TEXTS_DIR)
    stems = sorted(gt_by_stem)
    if max_pages:
        stems = stems[:max_pages]

    jobs = []
    for stem in stems:
        for section in SECTIONS:
            gt = gt_by_stem[stem].get(section)
            crop_path = CROPS_DIR / f"{stem}_page_001_{section}.png"
            if gt and gt.strip() and crop_path.exists():
                jobs.append(
                    {"stem": stem, "section": section,
                     "crop_path": str(crop_path), "gt": gt}
                )
    return jobs


async def evaluate_model(model_id: str, jobs: List[Dict]) -> List[Dict]:
    """Score one LM Studio model on all crop jobs (sequential — LM Studio lock).

    :param model_id: LM Studio model id.
    :param jobs: Output of :func:`collect_crop_jobs`.
    :returns: Per-crop result dicts.
    """
    results = []
    for i, job in enumerate(jobs, 1):
        start = time.time()
        text = await transcribe_with_lm_studio(
            model_id, job["crop_path"], CROP_TRANSCRIBE_PROMPTS[job["section"]],
            max_tokens=MAX_TOKENS,
        )
        elapsed = time.time() - start

        hyp = normalize_whitespace(text or "")
        ref = normalize_whitespace(job["gt"])
        strict, lenient = cer_pair(hyp, ref)
        result = {
            "model": model_id,
            "stem": job["stem"],
            "section": job["section"],
            "cer_strict": strict,
            "cer_lenient": lenient,
            "char_ratio": char_count_ratio(hyp, ref),
            "flags": flag_failure_modes(hyp, ref),
            "seconds": round(elapsed, 1),
            "output": text or "",
        }
        results.append(result)
        print(
            f"  [{i}/{len(jobs)}] {job['stem']}/{job['section']}: "
            f"CER {strict:.3f} (lenient {lenient:.3f}) in {elapsed:.0f}s"
        )
    return results


def summarize_model(results: List[Dict]) -> Dict:
    """Aggregate one model's crop results.

    :param results: Per-crop results for a single model.
    :returns: Summary dict with overall and per-section CER means.
    """
    by_section = defaultdict(list)
    for r in results:
        by_section[r["section"]].append(r)
    return {
        "n": len(results),
        "cer_strict": statistics.mean(r["cer_strict"] for r in results),
        "cer_lenient": statistics.mean(r["cer_lenient"] for r in results),
        "per_section": {
            s: {
                "cer_strict": statistics.mean(r["cer_strict"] for r in rs),
                "cer_lenient": statistics.mean(r["cer_lenient"] for r in rs),
            }
            for s, rs in sorted(by_section.items())
        },
    }


async def run(models: List[str], max_pages: Optional[int], use_wandb: bool) -> None:
    """Run the crop-track benchmark, model-major (one LM Studio model at a time).

    :param models: LM Studio model ids to score.
    :param max_pages: Optional page cap.
    :param use_wandb: Log summaries and a per-crop table to W&B.
    """
    available = await check_lm_studio_health()
    print(f"LM Studio models available: {available}")

    jobs = collect_crop_jobs(max_pages)
    print(f"Scoring {len(jobs)} crops × {len(models)} models\n")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    wandb_run = None
    if use_wandb:
        import wandb

        wandb_run = wandb.init(
            project="talmud-transcription-eval", job_type="crop_track",
            config={"models": models, "n_crops": len(jobs)},
        )

    all_results: Dict[str, List[Dict]] = {}
    with open(RESULTS_PATH, "a", encoding="utf-8") as sink:
        for model_id in models:
            print(f"── {model_id} ──")
            results = await evaluate_model(model_id, jobs)
            all_results[model_id] = results
            for r in results:
                sink.write(json.dumps(r, ensure_ascii=False) + "\n")
            sink.flush()

    print("\n══ Crop-track summary (pure reading, no layout) ══")
    for model_id, results in all_results.items():
        s = summarize_model(results)
        print(f"\n{model_id}  (n={s['n']})")
        print(f"  overall CER: {s['cer_strict']:.4f} strict / {s['cer_lenient']:.4f} lenient")
        for section, m in s["per_section"].items():
            print(f"  {section:8s}: {m['cer_strict']:.4f} / {m['cer_lenient']:.4f}")
        if wandb_run:
            key = model_id.replace("/", "_")
            wandb_run.summary[f"crop_track/{key}/cer_strict"] = s["cer_strict"]
            wandb_run.summary[f"crop_track/{key}/cer_lenient"] = s["cer_lenient"]
            for section, m in s["per_section"].items():
                wandb_run.summary[f"crop_track/{key}/{section}/cer_strict"] = m["cer_strict"]

    if wandb_run:
        import wandb

        table = wandb.Table(
            columns=["model", "stem", "section", "cer_strict", "cer_lenient",
                     "char_ratio", "flags", "output"],
        )
        for results in all_results.values():
            for r in results:
                table.add_data(r["model"], r["stem"], r["section"], r["cer_strict"],
                               r["cer_lenient"], r["char_ratio"],
                               ",".join(r["flags"]), r["output"][:2000])
        wandb_run.log({"transcriptions/crops": table})
        wandb_run.finish()

    print(f"\nPer-crop results appended → {RESULTS_PATH}")


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point.

    :param argv: Optional argument list (defaults to ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description="Crop-track (pure reading) benchmark.")
    parser.add_argument(
        "--lm_studio_models", type=str, required=True,
        help="Comma-separated LM Studio model ids (scored one at a time).",
    )
    parser.add_argument("--max_pages", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args(argv)

    models = [m.strip() for m in args.lm_studio_models.split(",") if m.strip()]
    asyncio.run(run(models, args.max_pages, args.wandb))


if __name__ == "__main__":
    main()
