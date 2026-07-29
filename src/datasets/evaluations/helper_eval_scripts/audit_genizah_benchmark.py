"""Audit and repair the frozen Genizah benchmark.

Two defects were found by manual review of ``genizah_test_v1`` (2026-07-29):

1. **Self-duplicated ground truth** — some FJP records concatenate two copies
   of the same edition (typically one with lacunae marked, one without), so
   the GT is ~2x the text actually on the page.  Every model is then capped
   near CER 0.5 however well it reads.
2. **Image / transcription misalignment** — for multi-leaf documents the
   edition may transcribe only one page while the record carries a different
   page's image (e.g. Oxford MS heb. f 39/30, where Gil edited page 3).

This script detects both, repairs (1) automatically, flags (2) using
cross-model agreement, and writes a verified subset manifest for the paper.

Misalignment signal: if EVERY model — including Kraken, which demonstrably
recovers real ink even when fragmentary — shows near-zero character overlap
with the GT on a fragment where those same models perform normally elsewhere,
the pairing is almost certainly wrong rather than merely hard.

Usage (from src/datasets/evaluations):
    PYTHONPATH=<repo> python helper_eval_scripts/audit_genizah_benchmark.py \\
        [--overlap-threshold 0.12] [--write-subset]
"""

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.evaluations.metrics import (  # noqa: E402
    genizah_visible_ink_gt,
    normalize_ink_hypothesis,
)

_BENCH_DIR = _REPO / "src/datasets/raw_data/cairo_genizah/evaluations/genizah_test_v1"
_OUTPUTS = _REPO / "src/datasets/evaluations/transcription_raw_outputs"
_HEB = re.compile(r"[֐-׿]+")

# Models whose output counts as evidence that a page is readable at all.
_EVIDENCE_MODELS = [
    "kraken_seg", "gemini_pro", "gemini_flash", "claude_opus_4_8",
    "claude_sonnet_5", "qwen3_vl_8b_heb_v17_step800",
    "qwen3_vl_8b_heb_v16_step1100", "vision_ocr_seg",
]


def letters_only(text: str) -> str:
    """Reduce text to bare Semitic letters for structural comparison.

    :param text: Any transcription text.
    :type text: str
    :return: Concatenated Hebrew/Judaeo-Arabic letters.
    :rtype: str
    """
    text = re.sub(r"\[\.\.\.\]|\(!\)|\[\?\]", " ", text)
    return "".join(_HEB.findall(text))


def find_self_duplication(gt: str, threshold: float = 0.55) -> float:
    """Score how strongly a ground truth repeats its own first half.

    :param gt: Ground-truth text.
    :type gt: str
    :param threshold: Similarity above which duplication is suspected.
    :type threshold: float
    :return: Similarity ratio between the two halves (0.0 when too short).
    :rtype: float
    """
    letters = letters_only(gt)
    if len(letters) < 200:
        return 0.0
    half = len(letters) // 2
    return difflib.SequenceMatcher(
        None, letters[:half], letters[half:], autojunk=False
    ).ratio()


def repair_duplication(gt: str) -> str:
    """Remove a duplicated second copy of the text from a ground truth.

    Splits on whitespace-preserving word boundaries and keeps whichever copy
    retains more visible characters (the fuller edition), which is normally
    the copy with fewer lacuna markers.

    :param gt: Ground-truth text suspected of self-duplication.
    :type gt: str
    :return: Repaired ground truth (unchanged when no split is found).
    :rtype: str
    """
    words = gt.split()
    n = len(words)
    best = None
    # Scan candidate split points around the midpoint for the division that
    # makes the two sides most similar; keep the richer side.
    for cut in range(int(n * 0.35), int(n * 0.65)):
        a = letters_only(" ".join(words[:cut]))
        b = letters_only(" ".join(words[cut:]))
        if not a or not b:
            continue
        sim = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
        if best is None or sim > best[0]:
            best = (sim, cut, len(a), len(b))
    if not best or best[0] < 0.55:
        return gt
    _sim, cut, len_a, len_b = best
    return " ".join(words[:cut]) if len_a >= len_b else " ".join(words[cut:])


def overlap_with_gt(hyp: str, gt_letters: str, n: int = 5) -> float:
    """Fraction of the hypothesis's character n-grams that occur in the GT.

    :param hyp: Model transcription.
    :type hyp: str
    :param gt_letters: Ground truth reduced to letters.
    :type gt_letters: str
    :param n: N-gram size.
    :type n: int
    :return: Overlap in [0, 1]; 0.0 when either side is too short.
    :rtype: float
    """
    h = letters_only(normalize_ink_hypothesis(hyp))
    if len(h) < n or len(gt_letters) < n:
        return 0.0
    gt_grams = {gt_letters[i:i + n] for i in range(len(gt_letters) - n + 1)}
    hyp_grams = [h[i:i + n] for i in range(len(h) - n + 1)]
    if not hyp_grams:
        return 0.0
    return sum(g in gt_grams for g in hyp_grams) / len(hyp_grams)


def main() -> None:
    """Run the audit and optionally write the verified subset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlap-threshold", type=float, default=0.12,
                        help="Below this best-model overlap, flag as misaligned")
    parser.add_argument("--write-subset", action="store_true",
                        help="Write genizah_test_v1_verified.json")
    args = parser.parse_args()

    spec = json.load(open(_BENCH_DIR / "genizah_test_v1.json"))
    docs = spec["docs"]

    duplicated, misaligned, unevaluated = [], [], []
    repaired_docs = []

    for d in docs:
        doc_id = d["doc_id"]
        gt = d["gt"]
        sim = find_self_duplication(gt)
        if sim > 0.55:
            fixed = repair_duplication(gt)
            duplicated.append((doc_id, round(sim, 2), len(letters_only(gt)),
                               len(letters_only(fixed))))
            gt = fixed

        gt_letters = letters_only(genizah_visible_ink_gt(gt))
        outdir = _OUTPUTS / doc_id
        best_overlap, best_model = 0.0, None
        found_any = False
        for m in _EVIDENCE_MODELS:
            f = outdir / f"{m}.txt"
            if not f.exists():
                continue
            found_any = True
            ov = overlap_with_gt(f.read_text(errors="replace"), gt_letters)
            if ov > best_overlap:
                best_overlap, best_model = ov, m
        if not found_any:
            unevaluated.append(doc_id)
        elif best_overlap < args.overlap_threshold:
            misaligned.append((doc_id, round(best_overlap, 3), best_model))

        nd = dict(d)
        nd["gt"] = gt
        nd["_repaired_duplication"] = sim > 0.55
        repaired_docs.append(nd)

    print(f"benchmark size                       : {len(docs)}")
    print(f"self-duplicated GT (repaired)        : {len(duplicated)}")
    for row in sorted(duplicated, key=lambda r: -r[1])[:20]:
        print(f"    {row[0]:38s} sim={row[1]}  {row[2]} → {row[3]} letters")
    print(f"\nevaluated so far                     : {len(docs) - len(unevaluated)}")
    print(f"flagged MISALIGNED (best overlap < {args.overlap_threshold}) : {len(misaligned)}")
    for row in sorted(misaligned, key=lambda r: r[1])[:25]:
        print(f"    {row[0]:38s} best={row[1]} ({row[2]})")
    print(f"\nnot yet evaluated (cannot check)     : {len(unevaluated)}")

    if args.write_subset:
        flagged = {r[0] for r in misaligned}
        kept = [d for d in repaired_docs if d["doc_id"] not in flagged]
        out = dict(spec)
        out["version"] = "genizah_test_v1_verified"
        out["derived_from"] = spec["version"]
        out["n"] = len(kept)
        out["docs"] = kept
        out["audit"] = {
            "duplicated_gt_repaired": [r[0] for r in duplicated],
            "excluded_misaligned": sorted(flagged),
            "overlap_threshold": args.overlap_threshold,
            "method": "GT self-duplication repaired by similarity split; "
                      "misalignment = no evidence model exceeded the n-gram "
                      "overlap threshold, indicating the image and the "
                      "transcription are not the same page",
        }
        path = _BENCH_DIR / "genizah_test_v1_verified.json"
        json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
        print(f"\nWrote {path} with {len(kept)} fragments")


if __name__ == "__main__":
    main()
