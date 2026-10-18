"""Transcribe the frozen PGP-131 benchmark with v19b-step1300 via LM Studio.

Mirrors the conditions of the existing v19a PGP outputs exactly: same frozen
fragment list (genizah_test_v1_verified.json), same benchmarked prompt
(``build_fragment_prompt`` — byte-identical to fragment_evals'), same
max_tokens (8192, the fragment-track default).  Outputs land beside the v19a
files as ``transcription_raw_outputs/<doc_id>/qwen3_vl_8b_heb_v19b_step1300.txt``
so ``score_genizah_offline.py --benchmark verified`` picks them up unchanged.
Resumable: existing output files are skipped.
"""
import asyncio
import json
import sys
from pathlib import Path

REPO = Path("/Users/isaac/Documents/GitHub/historical-document-analysis")
sys.path.insert(0, str(REPO))

from src.datasets.consensus.consensus_gate import build_fragment_prompt  # noqa: E402
from src.models.ocr.lms_transcriber import (  # noqa: E402
    check_lm_studio_health,
    transcribe_with_lm_studio,
)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3-vl-8b-heb-v19b-step1300"
BENCH = REPO / "src/datasets/raw_data/cairo_genizah/evaluations/genizah_test_v1"
OUT = REPO / "src/datasets/evaluations/transcription_raw_outputs"
MAX_TOKENS = 8192


async def main() -> None:
    """Run the 131 fragments sequentially, skipping finished ones."""
    docs = json.load(open(BENCH / "genizah_test_v1_verified.json"))["docs"]
    key = MODEL.replace("-", "_")
    todo = [d for d in docs
            if not (OUT / d["doc_id"] / f"{key}.txt").exists()]
    print(f"{len(docs)} fragments, {len(todo)} to transcribe", flush=True)
    if not todo:
        return
    models = await check_lm_studio_health()
    if MODEL not in models:
        raise RuntimeError(f"{MODEL} not served by LM Studio: {models}")
    for i, d in enumerate(todo, 1):
        image = BENCH / "images" / f"{d['doc_id']}.jpg"
        if not image.exists():
            print(f"  MISSING IMAGE {d['doc_id']}", flush=True)
            continue
        txt = await transcribe_with_lm_studio(
            MODEL, str(image), build_fragment_prompt(d["doc_id"]),
            max_tokens=MAX_TOKENS)
        outdir = OUT / d["doc_id"]
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"{key}.txt").write_text(txt or "", encoding="utf-8")
        if i % 10 == 0:
            print(f"  {i}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
