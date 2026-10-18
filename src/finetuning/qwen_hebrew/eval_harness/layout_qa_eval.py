"""Layout-QA (the VQA task family) eval on held-out ktiv_v2 val rows.

The val split is held out by manuscript — the same rows the v20a notebook's
eval loss samples from — so no page here was trained on. Three templated
question types from ``build_ktiv_dataset``:

* ``columns``   — "How many columns ... number only" → integer exact-match
* ``edge_line`` — "Transcribe ONLY the first/last line" → CER vs GT line
* ``find_line`` — "Which line contains the phrase ..." → CER vs GT line,
  plus whether the answer contains the queried phrase at all

Answers come from LM Studio at the harness's standard decode (temperature
0.1, single sample), so per-question results carry the usual decode-noise
caveat; the per-type aggregates over ~40 questions each are the readout.
Results land beside the grounding results as
``layout_qa_eval_<model>_results.json``.
"""
import argparse
import asyncio
import json
import os
import random
import re
import statistics
import sys
import tempfile
from pathlib import Path

REPO = Path("/Users/isaac/Documents/GitHub/historical-document-analysis")
sys.path.insert(0, str(REPO))
# Dataset cache stays on the LOCAL disk: the SMB-mounted NAS throws EBADF on
# the download's append-open / file locks (2026-09-04). The val split is small.
os.environ["HF_HOME"] = str(Path.home() / ".cache/huggingface")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")
from datasets import load_dataset  # noqa: E402

from src.datasets.evaluations.metrics import cer_pair  # noqa: E402
from src.models.ocr.lms_transcriber import (  # noqa: E402
    check_lm_studio_health,
    transcribe_with_lm_studio,
)

KTIV2 = "isaacmg/genizah_ktiv_v2"
REV = "1bb20e209a3b3095fc84dad7c928da4310269fd3"
OUT_DIR = REPO / "src/datasets/evaluations/grounding_eval"
MAX_TOKENS = 512


def qtype(question: str) -> str:
    """Classify a layout-QA question into its template family.

    :param question: The prompt text of the row.
    :return: One of ``columns``, ``edge_line``, ``find_line``.
    """
    if question.startswith("How many columns"):
        return "columns"
    if question.startswith("Transcribe ONLY the"):
        return "edge_line"
    if question.startswith("Which line of this manuscript page contains"):
        return "find_line"
    raise ValueError(f"unknown layout_qa template: {question[:60]!r}")


def norm(text: str) -> str:
    """Whitespace-normalize an answer for comparison.

    :param text: Raw model or GT text.
    :return: Stripped text with runs of whitespace collapsed.
    """
    return re.sub(r"\s+", " ", (text or "").strip())


def first_int(text: str):
    """Extract the first integer in a string, or ``None``.

    :param text: Model answer.
    :return: The integer, or ``None`` when no digits are present.
    """
    m = re.search(r"\d+", text or "")
    return int(m.group()) if m else None


def score(row: dict, pred: str) -> dict:
    """Score one prediction against its GT answer by question type.

    :param row: Dataset row (``question``, ``answer``).
    :param pred: Raw model output.
    :return: Per-question record with type-specific metrics.
    """
    t = qtype(row["question"])
    gt, p = norm(row["answer"]), norm(pred)
    rec = {"stem": row["stem"], "qtype": t, "question": row["question"],
           "gt": gt, "pred": p}
    if t == "columns":
        rec["correct"] = first_int(p) == first_int(gt)
    else:
        rec["cer"] = cer_pair(p, gt)[0] if gt else None   # strict CER, hyp-then-ref
        rec["exact"] = p == gt
        if t == "find_line":
            phrase = re.search(r'"(.+?)"', row["question"]).group(1)
            rec["has_phrase"] = norm(phrase) in p
    return rec


async def main() -> None:
    """Run the sampled val questions through LM Studio and summarize."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # STREAM the val split: a non-streaming load_dataset prepares EVERY split
    # of the config first (the 33 GB train shards), which filled the local
    # disk on 2026-09-04. Streaming touches only the 4 val shards (~1.7 GB)
    # and caches nothing.
    stream = load_dataset(KTIV2, split="val", revision=REV,
                          token=os.environ["HF1_TOKEN"], streaming=True)
    pool = [row for row in stream if row["task"] == "layout_qa"]
    random.Random(args.seed).shuffle(pool)
    qa = pool[:args.n]
    print(f"layout_qa val rows: {len(pool)} total, evaluating {len(qa)} for {args.model}",
          flush=True)

    served = await check_lm_studio_health()
    if args.model not in served:
        raise RuntimeError(f"{args.model} not served by LM Studio: {served}")

    recs = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, row in enumerate(qa, 1):
            img = Path(tmp) / f"{i}.jpg"
            row["image"].convert("RGB").save(img, quality=95)
            pred = await transcribe_with_lm_studio(
                args.model, str(img), row["question"], max_tokens=MAX_TOKENS)
            recs.append(score(row, pred or ""))
            if i % 20 == 0:
                print(f"  {i}/{len(qa)}", flush=True)

    by = {}
    for r in recs:
        by.setdefault(r["qtype"], []).append(r)
    summary = {}
    print(f"\n== layout_qa: {args.model}")
    for t, rs in sorted(by.items()):
        if t == "columns":
            acc = sum(r["correct"] for r in rs) / len(rs)
            summary[t] = {"n": len(rs), "accuracy": acc}
            print(f"   {t:9s} n={len(rs):3d}  accuracy {acc:.3f}")
        else:
            cers = [r["cer"] for r in rs if r["cer"] is not None]
            med = statistics.median(cers) if cers else None
            exact = sum(r["exact"] for r in rs) / len(rs)
            summary[t] = {"n": len(rs), "median_cer": med, "exact_rate": exact}
            line = f"   {t:9s} n={len(rs):3d}  median CER {med:.3f}  exact {exact:.3f}"
            if t == "find_line":
                hp = sum(r["has_phrase"] for r in rs) / len(rs)
                summary[t]["has_phrase_rate"] = hp
                line += f"  contains-phrase {hp:.3f}"
            print(line)
    out = OUT_DIR / f"layout_qa_eval_{args.model}_results.json"
    out.write_text(json.dumps({"model": args.model, "summary": summary, "rows": recs},
                              ensure_ascii=False, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
