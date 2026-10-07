"""Benchmark line-wise v18b on the verified 131-fragment set.

Writes each assembled page to ``transcription_raw_outputs/<doc>/
qwen3_vl_8b_heb_v18b_linewise.txt`` so the offline scorer
(``score_genizah_offline.py``) picks it up as just another model row —
coverage, F1, ngramP and per-script splits come for free, directly
comparable to the whole-page ``qwen3_vl_8b_heb_v18b_step700`` row.

Per-line sidecars (kraken text + VLM text + bbox per line) go to
``transcription_results/linewise_details/<doc>.jsonl`` for reconciliation
experiments.  Resumable: docs with an existing output file are skipped.

Usage (from repo root):
    PYTHONPATH=. python -m src.datasets.consensus.run_linewise_benchmark \\
        [--mode blind] [--limit 0]
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from src.datasets.consensus.linewise import (
    transcribe_linewise,
    transcribe_sections,
)
from src.datasets.evaluations.helper_eval_scripts.score_genizah_offline import (
    load_benchmark,
)
from src.models.ocr.lms_transcriber import check_lm_studio_health

_REPO = Path(__file__).resolve().parents[3]
_EVAL_DIR = _REPO / "src/datasets/evaluations"
_BENCH_DIR = _REPO / "src/datasets/raw_data/cairo_genizah/evaluations/genizah_test_v1"
_OUTPUTS = _EVAL_DIR / "transcription_raw_outputs"
_DETAILS = _EVAL_DIR / "transcription_results/linewise_details"

VLM_MODEL = "qwen3-vl-8b-heb-v18b-step700"
OUTPUT_MODEL_KEY = "qwen3_vl_8b_heb_v18b_linewise"


def _image_for_doc(doc: dict) -> Path:
    """Resolve the staged benchmark image for a verified-benchmark document.

    The benchmark run staged every fragment image as ``images/<doc_id>.jpg``.

    :param doc: Benchmark document dict.
    :type doc: dict
    :return: Path to the local image file.
    :rtype: Path
    """
    return _BENCH_DIR / "images" / f"{doc['doc_id']}.jpg"


async def run(mode: str, limit: int) -> None:
    """Process the verified benchmark docs line-wise.

    :param mode: ``blind`` or ``adversarial`` (changes the output key too).
    :type mode: str
    :param limit: Max docs this run (0 = all).
    :type limit: int
    """
    models = await check_lm_studio_health()
    if VLM_MODEL not in models:
        raise RuntimeError(f"{VLM_MODEL} not served by LM Studio")

    if mode == "sections":
        out_key = "qwen3_vl_8b_heb_v18b_sections"
    else:
        out_key = OUTPUT_MODEL_KEY + ("_adv" if mode == "adversarial" else "")
    _DETAILS.mkdir(parents=True, exist_ok=True)
    docs = load_benchmark("verified")
    todo = []
    for d in docs:
        out_path = _OUTPUTS / d["doc_id"] / f"{out_key}.txt"
        img = _image_for_doc(d)
        if out_path.exists() or not img.exists():
            continue
        todo.append((d, img, out_path))
    if limit:
        todo = todo[:limit]
    print(f"benchmark docs: {len(docs)}, to process: {len(todo)} (mode={mode})")

    for i, (d, img, out_path) in enumerate(todo, 1):
        t0 = time.time()
        try:
            if mode == "sections":
                result = await transcribe_sections(str(img), d["doc_id"],
                                                   VLM_MODEL)
                units = result["bands"]
            else:
                result = await transcribe_linewise(str(img), d["doc_id"],
                                                   VLM_MODEL, mode=mode)
                units = result["lines"]
        except Exception as exc:
            print(f"[{i}/{len(todo)}] {d['doc_id']} FAILED: {exc}", flush=True)
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result["text"], encoding="utf-8")
        detail_path = _DETAILS / f"{d['doc_id']}.{mode}.jsonl"
        with open(detail_path, "w", encoding="utf-8") as fh:
            for unit in units:
                fh.write(json.dumps(unit, ensure_ascii=False) + "\n")
        print(f"[{i}/{len(todo)}] {d['doc_id']}: {len(units)} units, "
              f"{len(result['text'])} chars in {time.time() - t0:.0f}s",
              flush=True)
    print("done")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("blind", "adversarial", "sections"),
                        default="sections")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(run(args.mode, args.limit))


if __name__ == "__main__":
    main()
