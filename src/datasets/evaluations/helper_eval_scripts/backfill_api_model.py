"""Backfill one API model's transcriptions for fragments that are missing it.

Benchmark runs can lose a model partway through for reasons unrelated to the
model itself — a provider spending cap, an exhausted credit balance, a
transient outage.  Re-running the whole harness would re-spend on every model
that already succeeded and would contend for LM Studio with a running judge
phase, so this script calls exactly one model on exactly the fragments that
lack its output and writes the result where the offline scorer expects it
(``transcription_raw_outputs/<fragment>/<model_key>.txt``).

Because scoring is decoupled from collection, no W&B run or merge step is
needed: the next ``score_genizah_offline.py`` pass picks the new files up.

Usage (from src/datasets/evaluations):
    PYTHONPATH=<repo> python helper_eval_scripts/backfill_api_model.py \\
        --provider claude --model claude-opus-4-8 [--limit N] [--dry-run]
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import dotenv

dotenv.load_dotenv()

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fragment_evals import _fragment_prompt, model_key_from_id  # noqa: E402
from src.models.llm.transcription.claude_transcriber import (  # noqa: E402
    transcribe_with_claude,
)
from src.models.llm.transcription.openai_transcriber import (  # noqa: E402
    transcribe_with_openai,
)

_BENCH = (_REPO / "src/datasets/raw_data/cairo_genizah/evaluations/genizah_test_v1"
          / "genizah_test_v1.json")
_IMAGES = _REPO / "src/datasets/evaluations/genizah_images"
_OUTPUTS = _REPO / "src/datasets/evaluations/transcription_raw_outputs"


def missing_fragments(model_key: str) -> list:
    """List benchmark fragments that have no output for this model.

    :param model_key: Sanitised model key used as the output filename.
    :type model_key: str
    :return: List of (doc_id, image_path) for fragments still missing.
    :rtype: list
    """
    spec = json.load(open(_BENCH))
    out = []
    for d in spec["docs"]:
        doc_id = d["doc_id"]
        if (_OUTPUTS / doc_id / f"{model_key}.txt").exists():
            continue
        basename = d["image_url"].rsplit("/", 1)[-1]
        img = _IMAGES / doc_id / basename
        if not img.exists():
            candidates = list((_IMAGES / doc_id).glob("*.jpg")) if (_IMAGES / doc_id).is_dir() else []
            if not candidates:
                print(f"  ⚠️  no local image for {doc_id} — skipping")
                continue
            img = candidates[0]
        out.append((doc_id, img))
    return out


async def main() -> None:
    """Call the requested model on every fragment missing its output."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("claude", "openai"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between calls (be polite to the provider)")
    args = parser.parse_args()

    key = model_key_from_id(args.model)
    todo = missing_fragments(key)
    if args.limit:
        todo = todo[:args.limit]
    print(f"{args.model} → {len(todo)} fragments missing output")
    if args.dry_run:
        for doc_id, _ in todo[:10]:
            print("   ", doc_id)
        return

    call = transcribe_with_claude if args.provider == "claude" else transcribe_with_openai
    ok = fail = 0
    t0 = time.time()
    for i, (doc_id, img) in enumerate(todo, 1):
        prompt = _fragment_prompt(doc_id)
        text = await call(str(img), prompt, model=args.model)
        if text:
            dest = _OUTPUTS / doc_id / f"{key}.txt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")
            ok += 1
            print(f"[{i}/{len(todo)}] ✓ {doc_id}: {len(text)} chars")
        else:
            fail += 1
            print(f"[{i}/{len(todo)}] ✗ {doc_id}: no output")
        await asyncio.sleep(args.delay)

    print(f"\n{args.model}: {ok} written, {fail} failed, "
          f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    asyncio.run(main())
