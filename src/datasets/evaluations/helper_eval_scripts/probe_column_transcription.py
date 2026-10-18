# File name: probe_column_transcription.py
# Date: 9/11/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Probe: does section-based prompting fix the two-column under-read on Talmud manuscripts?

The CER breakdown (2026-09-10) showed the religious-140 two-column pages at median CER 0.278 with 36 %
under-read (hyp/gt letter ratio < 0.8) versus 0.203 / 12 % on single-column pages — coverage, not glyph
quality, is the lever. This probe re-transcribes every two-column benchmark page with the flagship
(v2.0a-1800 via LM Studio) under three conditions and scores them with the benchmark's own metrics:

    A  page      — the benchmarked full-page prompt (existing outputs reused, never re-decoded)
    B  prompt    — the same page image, two calls: "ONLY the right column" then "ONLY the left column"
    C  crop      — the page split at its widest vertical ink gap, each half transcribed with the page prompt

Hebrew reading order: right column first, so B and C concatenate right then left. Outputs are cached per
(doc, condition) so the probe is resumable; nothing else touches LM Studio's prod state.

Usage (repo root):
    .venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.probe_column_transcription \\
        [--model qwen3-vl-8b-heb-v20a-step1800] [--limit N]
"""
import argparse
import asyncio
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

REPO = Path("/Users/isaac/Documents/GitHub/historical-document-analysis")
sys.path.insert(0, str(REPO))
from src.datasets.consensus.consensus_gate import build_fragment_prompt  # noqa: E402
from src.datasets.evaluations.helper_eval_scripts.run_religious_benchmark import _OUT  # noqa: E402
from src.datasets.evaluations.metrics import (  # noqa: E402
    cer_pair, genizah_visible_ink_gt, normalize_ink_hypothesis)
from src.models.ocr.lms_transcriber import check_lm_studio_health, transcribe_with_lm_studio  # noqa: E402

BENCH = REPO / "src/datasets/raw_data/cairo_genizah/evaluations/genizah_religious_v1/genizah_religious_v1.json"
PROBE_DIR = REPO / "src/datasets/evaluations/probes"
CACHE = REPO / "src/datasets/evaluations/transcription_raw_outputs_probe_columns"
MAX_TOKENS = 2500
COLUMN_HINT = ("\n\nThis page is written in TWO columns. Transcribe ONLY the {side} column "
               "(the {order} column in reading order) and ignore the other column entirely.")


def letters(text: str) -> int:
    """Count Hebrew letters in ``text``.

    :param text: Any string.
    :returns: Number of characters in the Hebrew letter block.
    """
    return sum("א" <= ch <= "ת" for ch in text)


def split_x(image_path: str) -> Optional[Tuple[int, int]]:
    """Find the widest vertical ink gap in the central band of a two-column page.

    :param image_path: Page image path.
    :returns: ``(split_x, width)`` in original pixels, or None when no clear gap exists.
    """
    im = Image.open(image_path).convert("L")
    w, h = im.size
    small = im.resize((max(200, w // 8), max(200, h // 8)))
    a = 255 - np.asarray(small, dtype=np.float32)          # ink is bright
    a = a[int(a.shape[0] * 0.08): int(a.shape[0] * 0.92)]   # drop top/bottom margins
    profile = a.mean(axis=0)
    lo, hi = int(profile.size * 0.35), int(profile.size * 0.65)
    band = profile[lo:hi]
    thresh = np.percentile(profile, 20) + 0.15 * (np.percentile(profile, 80) - np.percentile(profile, 20))
    low = band <= thresh
    best, cur_start, best_span = None, None, 0
    for i, flag in enumerate(list(low) + [False]):
        if flag and cur_start is None:
            cur_start = i
        elif not flag and cur_start is not None:
            if i - cur_start > best_span:
                best_span, best = i - cur_start, (cur_start + i) // 2
            cur_start = None
    if best is None or best_span < max(3, band.size // 40):
        return None
    return int((lo + best) * w / profile.size), w


async def decode(model: str, image: str, prompt: str, cache_file: Path) -> str:
    """Cached LM Studio decode.

    :param model: LM Studio model id.
    :param image: Image path.
    :param prompt: Prompt text.
    :param cache_file: Where the raw text is cached.
    :returns: Raw model text ('' on failure).
    """
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8", errors="replace")
    txt = await transcribe_with_lm_studio(model, image, prompt, max_tokens=MAX_TOKENS) or ""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(txt, encoding="utf-8")
    return txt


def score(hyp: str, gt: str) -> Dict[str, float]:
    """Benchmark-consistent CER plus the coverage ratio.

    :param hyp: Model text.
    :param gt: Ground truth.
    :returns: Dict with ``cer``, ``len_ratio``, ``hyp_letters``, ``gt_letters``.
    """
    gt_ink = genizah_visible_ink_gt(gt)
    cer, _ = cer_pair(normalize_ink_hypothesis(hyp), gt_ink)
    return {"cer": round(float(cer), 4), "len_ratio": round(letters(hyp) / max(1, letters(gt)), 3),
            "hyp_letters": letters(hyp), "gt_letters": letters(gt)}


async def run(model: str, limit: int) -> None:
    """Run all three conditions over the two-column pages and write CSV + summary."""
    docs = [d for d in json.load(open(BENCH))["docs"] if d["n_columns"] >= 2]
    if limit:
        docs = docs[:limit]
    served = await check_lm_studio_health()
    assert model in served, f"{model} not served: {served}"
    key = model.replace("-", "_")
    rows: List[Dict] = []
    for i, d in enumerate(docs, 1):
        gt, image, did = d["gt"], d["image"], d["doc_id"]
        base_prompt = build_fragment_prompt(did)
        a_file = _OUT / did / f"{key}.txt"
        a_txt = a_file.read_text(encoding="utf-8", errors="replace") if a_file.exists() else ""
        right = await decode(model, image, base_prompt + COLUMN_HINT.format(side="RIGHT", order="first"),
                             CACHE / did / f"{key}__B_right.txt")
        left = await decode(model, image, base_prompt + COLUMN_HINT.format(side="LEFT", order="second"),
                            CACHE / did / f"{key}__B_left.txt")
        b_txt = right.strip() + "\n" + left.strip()
        c_txt, split_info = "", None
        sp = split_x(image)
        if sp:
            sx, w = sp
            im = Image.open(image)
            crops = {"right": im.crop((sx, 0, w, im.height)), "left": im.crop((0, 0, sx, im.height))}
            parts = []
            for side, crop in crops.items():
                cp = CACHE / did / f"crop_{side}.jpg"
                if not cp.exists():
                    cp.parent.mkdir(parents=True, exist_ok=True)
                    crop.convert("RGB").save(cp, quality=92)
                parts.append((await decode(model, str(cp), base_prompt, CACHE / did / f"{key}__C_{side}.txt")).strip())
            c_txt = "\n".join(parts)
            split_info = round(sx / w, 3)
        for cond, txt in (("A_page", a_txt), ("B_prompt", b_txt), ("C_crop", c_txt)):
            if cond == "C_crop" and not sp:
                continue
            s = score(txt, gt)
            rows.append({"doc_id": did, "condition": cond, "split_frac": split_info if cond == "C_crop" else "", **s})
        last = {r["condition"]: r["cer"] for r in rows if r["doc_id"] == did}
        print(f"  {i}/{len(docs)} {did[-24:]} " + " ".join(f"{k}={v:.3f}" for k, v in last.items()), flush=True)
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PROBE_DIR / f"column_transcription_{key}.csv"
    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    lines = [f"# Column-transcription probe — {model} on the {len(docs)} two-column religious-140 pages", "",
             "| condition | n | median CER | mean CER | under-read (<0.8) | over-gen (>1.2) | median len_ratio |",
             "|---|---|---|---|---|---|---|"]
    for cond in ("A_page", "B_prompt", "C_crop"):
        rs = [r for r in rows if r["condition"] == cond]
        if not rs:
            continue
        cers = [r["cer"] for r in rs]; lr = [r["len_ratio"] for r in rs]
        lines.append(f"| {cond} | {len(rs)} | {statistics.median(cers):.3f} | {statistics.mean(cers):.3f} | "
                     f"{sum(x < 0.8 for x in lr)}/{len(rs)} | {sum(x > 1.2 for x in lr)}/{len(rs)} | {statistics.median(lr):.2f} |")
    paired = {}
    for r in rows:
        paired.setdefault(r["doc_id"], {})[r["condition"]] = r["cer"]
    for cond in ("B_prompt", "C_crop"):
        deltas = [v[cond] - v["A_page"] for v in paired.values() if cond in v and "A_page" in v]
        if deltas:
            lines.append(f"\n{cond} vs A_page: paired median ΔCER {statistics.median(deltas):+.3f}, "
                         f"better on {sum(x < -0.02 for x in deltas)}, worse on {sum(x > 0.02 for x in deltas)} of {len(deltas)}")
    md = PROBE_DIR / f"column_transcription_{key}.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"wrote {csv_path} and {md}", flush=True)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen3-vl-8b-heb-v20a-step1800")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    asyncio.run(run(a.model, a.limit))


if __name__ == "__main__":
    main()
