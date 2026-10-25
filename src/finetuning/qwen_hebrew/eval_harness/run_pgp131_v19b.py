"""Transcribe the frozen PGP-131 benchmark with v19b-step1300 via LM Studio.

Mirrors the conditions of the existing v19a PGP outputs exactly: same frozen
fragment list (genizah_test_v1_verified.json), same benchmarked prompt
(``build_fragment_prompt`` — byte-identical to fragment_evals'), same
max_tokens (8192, the fragment-track default).  Outputs land beside the v19a
files as ``transcription_raw_outputs/<doc_id>/qwen3_vl_8b_heb_v19b_step1300.txt``
so ``score_genizah_offline.py --benchmark verified`` picks them up unchanged.
Resumable: existing output files are skipped.

Usage::

    .venv/bin/python src/finetuning/qwen_hebrew/eval_harness/run_pgp131_v19b.py [MODEL] \\
        [--spec SPEC.json --images-dir DIR --outputs-dir DIR]

The defaults read PGP-131; another spec in the same verified-JSON format (e.g. the sealed
confirmation set) is read with ``--spec`` / ``--images-dir`` / ``--outputs-dir``.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

REPO = Path("/Users/isaac/Documents/GitHub/historical-document-analysis")
sys.path.insert(0, str(REPO))

from src.datasets.consensus.consensus_gate import build_fragment_prompt  # noqa: E402
from src.models.ocr.lms_transcriber import (  # noqa: E402
    check_lm_studio_health,
    transcribe_with_lm_studio,
)

DEFAULT_MODEL = "qwen3-vl-8b-heb-v19b-step1300"
BENCH = REPO / "src/datasets/raw_data/cairo_genizah/evaluations/genizah_test_v1"
OUT = REPO / "src/datasets/evaluations/transcription_raw_outputs"
MAX_TOKENS = 8192


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command line: the model id plus optional spec / image / output overrides.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :type argv: Optional[List[str]]
    :return: ``model``, ``spec``, ``images_dir``, ``outputs_dir``.
    :rtype: argparse.Namespace
    """
    ap = argparse.ArgumentParser(description="Transcribe a verified-JSON benchmark via LM Studio.")
    ap.add_argument("model", nargs="?", default=DEFAULT_MODEL, help="LM Studio model id")
    ap.add_argument("--spec", type=Path, default=BENCH / "genizah_test_v1_verified.json",
                    help="benchmark JSON ({'docs': [{'doc_id': ...}]})")
    ap.add_argument("--images-dir", type=Path, default=BENCH / "images", help="folder of <doc_id>.jpg")
    ap.add_argument("--outputs-dir", type=Path, default=OUT, help="output root (<dir>/<doc_id>/<model>.txt)")
    return ap.parse_args(argv)


async def main(args: argparse.Namespace) -> None:
    """Run the benchmark fragments sequentially, skipping finished ones.

    :param args: :func:`parse_args` result.
    :type args: argparse.Namespace
    """
    docs = json.load(open(args.spec))["docs"]
    key = args.model.replace("-", "_")
    todo = [d for d in docs
            if not (args.outputs_dir / d["doc_id"] / f"{key}.txt").exists()]
    print(f"{len(docs)} fragments, {len(todo)} to transcribe", flush=True)
    if not todo:
        return
    models = await check_lm_studio_health()
    if args.model not in models:
        raise RuntimeError(f"{args.model} not served by LM Studio: {models}")
    for i, d in enumerate(todo, 1):
        image = args.images_dir / f"{d['doc_id']}.jpg"
        if not image.exists():
            print(f"  MISSING IMAGE {d['doc_id']}", flush=True)
            continue
        txt = await transcribe_with_lm_studio(
            args.model, str(image), build_fragment_prompt(d["doc_id"]),
            max_tokens=MAX_TOKENS)
        outdir = args.outputs_dir / d["doc_id"]
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"{key}.txt").write_text(txt or "", encoding="utf-8")
        if i % 10 == 0:
            print(f"  {i}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
