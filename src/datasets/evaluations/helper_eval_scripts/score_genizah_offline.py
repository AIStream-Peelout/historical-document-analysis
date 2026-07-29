"""Offline scoring pipeline — the single source of truth for Genizah results.

Decouples SCORING from COLLECTION.  The benchmark run saves each model's raw
transcription to ``transcription_raw_outputs/<fragment>/<model>.txt``; this
script recomputes every reported number from those files against the repaired
ground truth, so ground-truth corrections never require re-running a model.

Why not plain CER?  On Genizah fragments a single CER column entangles four
different things: reading accuracy, the editor's choice of which text layer to
transcribe, reading-order decisions on non-linear pages, and ground-truth
quality.  Models also fail in categorically different ways — abstention, loop
collapse, fluent hallucination, partial reads — and averaging those produces a
number that ranks the *shape of failure* rather than reading ability.  So this
pipeline reports:

* **Failure mode** per fragment (abstained / loop_collapse / hallucinated /
  substantive) — the behavioural table is the primary result.
* **Aligned precision / recall / F1** — separates "read a little, accurately"
  from "read everything, noisily", which CER cannot.
* **Order-independent n-gram precision** — is what the model wrote actually on
  this page?  Robust to layer choice and reading order, so it remains
  meaningful on palimpsests where CER does not.  This is the hallucination
  measure.
* **CER / WER conditional on a substantive attempt**, never as a bare mean.
* **Tier A / B**: Tier A fragments are those where at least two independent
  systems (one of them order-blind Kraken) converge on the ground truth —
  evidence the GT is the right layer in the right order, so CER is meaningful.
  Tier B fragments report precision-family metrics only.

Usage (from src/datasets/evaluations):
    PYTHONPATH=<repo> python helper_eval_scripts/score_genizah_offline.py \\
        [--benchmark verified|frozen] [--no-wandb]
"""

import argparse
import collections
import csv
import difflib
import json
import re
import statistics
import sys
from pathlib import Path

import dotenv

dotenv.load_dotenv()

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.evaluations.metrics import (  # noqa: E402
    cer_pair,
    genizah_visible_ink_gt,
    normalize_ink_hypothesis,
    wer_pair,
)
from src.datasets.evaluations.helper_eval_scripts.audit_genizah_benchmark import (  # noqa: E402
    find_self_duplication,
    letters_only,
    repair_duplication,
)

_BENCH_DIR = _REPO / "src/datasets/raw_data/cairo_genizah/evaluations/genizah_test_v1"
_OUTPUTS = _REPO / "src/datasets/evaluations/transcription_raw_outputs"

# Thresholds (documented in the paper's methods section).
ABSTAIN_MIN_CHARS = 25       # below this the model effectively declined
LOOP_REPEAT_RATIO = 0.45     # share of text covered by one repeated 12-gram
HALLUCINATION_NGRAM = 0.10   # n-gram precision below this = not this page
TIER_A_OVERLAP = 0.25        # per-system overlap needed to count as convergent
NGRAM_N = 5

_REFUSAL_RE = re.compile(
    r"can(no|')?t (help|assist|transcribe|read|comply)|unable to (transcribe|read|comply)"
    r"|sorry[—,\- ]|not able to transcribe|cannot comply", re.I)
# Order-blind evidence system: reads ink without a language prior.
_ORDER_BLIND = "kraken_seg"
_VLM_EVIDENCE = [
    "gemini_pro", "gemini_flash", "claude_opus_4_8", "claude_sonnet_5",
    "gpt_5_6_sol", "qwen3_vl_8b_heb_v17_step800", "qwen3_vl_8b_heb_v16_step1100",
]


def ngram_precision(hyp_letters: str, gt_letters: str, n: int = NGRAM_N) -> float:
    """Fraction of the hypothesis's n-grams that appear anywhere in the GT.

    Order-independent, so it stays meaningful on multi-layer pages where the
    reference linearisation is arbitrary.  Low values mean invented text.

    :param hyp_letters: Hypothesis reduced to letters.
    :type hyp_letters: str
    :param gt_letters: Ground truth reduced to letters.
    :type gt_letters: str
    :param n: N-gram size.
    :type n: int
    :return: Precision in [0, 1].
    :rtype: float
    """
    if len(hyp_letters) < n or len(gt_letters) < n:
        return 0.0
    gt_grams = {gt_letters[i:i + n] for i in range(len(gt_letters) - n + 1)}
    hyp_grams = [hyp_letters[i:i + n] for i in range(len(hyp_letters) - n + 1)]
    return sum(g in gt_grams for g in hyp_grams) / len(hyp_grams)


def loop_ratio(letters: str, span: int = 12) -> float:
    """Share of the text accounted for by its single most repeated span.

    :param letters: Hypothesis reduced to letters.
    :type letters: str
    :param span: Length of the repeated unit to look for.
    :type span: int
    :return: Ratio in [0, 1]; 0.0 for texts shorter than the span.
    :rtype: float
    """
    if len(letters) < span * 3:
        return 0.0
    counts = collections.Counter(
        letters[i:i + span] for i in range(len(letters) - span + 1)
    )
    top, freq = counts.most_common(1)[0]
    return (freq * span) / len(letters)


def aligned_prf(hyp: str, gt: str) -> tuple:
    """Character-level precision, recall and F1 from a sequence alignment.

    :param hyp: Normalised hypothesis text.
    :type hyp: str
    :param gt: Visible-ink ground truth.
    :type gt: str
    :return: (precision, recall, f1), each in [0, 1].
    :rtype: tuple
    """
    if not hyp or not gt:
        return 0.0, 0.0, 0.0
    matcher = difflib.SequenceMatcher(None, hyp, gt, autojunk=False)
    matched = sum(b.size for b in matcher.get_matching_blocks())
    precision = matched / len(hyp)
    recall = matched / len(gt)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def classify(raw: str, hyp_letters: str, gt_letters: str, ngram_p: float) -> str:
    """Assign a failure mode to one model output.

    :param raw: Raw model output as saved.
    :type raw: str
    :param hyp_letters: Normalised hypothesis reduced to letters.
    :type hyp_letters: str
    :param gt_letters: Ground truth reduced to letters.
    :type gt_letters: str
    :param ngram_p: Order-independent n-gram precision.
    :type ngram_p: float
    :return: One of abstained / loop_collapse / hallucinated / substantive.
    :rtype: str
    """
    if len(hyp_letters) < ABSTAIN_MIN_CHARS or _REFUSAL_RE.search(raw[:400]):
        return "abstained"
    if loop_ratio(hyp_letters) > LOOP_REPEAT_RATIO:
        return "loop_collapse"
    if ngram_p < HALLUCINATION_NGRAM:
        return "hallucinated"
    return "substantive"


def load_benchmark(which: str) -> list:
    """Load benchmark documents with duplicated ground truths repaired.

    :param which: ``verified`` to use the audited subset when present,
        otherwise the frozen 150.
    :type which: str
    :return: List of document dicts with repaired ``gt``.
    :rtype: list
    """
    path = _BENCH_DIR / ("genizah_test_v1_verified.json" if which == "verified"
                         else "genizah_test_v1.json")
    if not path.exists():
        path = _BENCH_DIR / "genizah_test_v1.json"
    spec = json.load(open(path))
    docs = []
    for d in spec["docs"]:
        nd = dict(d)
        if not d.get("_repaired_duplication") and find_self_duplication(d["gt"]) > 0.55:
            nd["gt"] = repair_duplication(d["gt"])
            nd["_repaired_duplication"] = True
        docs.append(nd)
    return docs


def main() -> None:
    """Score every saved output and emit the paper tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("frozen", "verified"), default="verified")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("transcription_results/paper_table"))
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-run-name", default="genizah-offline-scoring-v1")
    args = parser.parse_args()

    docs = load_benchmark(args.benchmark)
    rows = []
    tiers = {}

    for d in docs:
        doc_id = d["doc_id"]
        outdir = _OUTPUTS / doc_id
        if not outdir.is_dir():
            continue
        gt_ink = genizah_visible_ink_gt(d["gt"])
        gt_letters = letters_only(gt_ink)
        if len(gt_letters) < 50:
            continue

        per_model = {}
        for f in sorted(outdir.glob("*.txt")):
            model = f.stem
            if model in ("ground_truth", "consensus") or model.endswith(("_raw", "_flat")):
                continue
            raw = f.read_text(errors="replace")
            hyp = normalize_ink_hypothesis(raw)
            hyp_letters = letters_only(hyp)
            ngram_p = ngram_precision(hyp_letters, gt_letters)
            mode = classify(raw, hyp_letters, gt_letters, ngram_p)
            precision, recall, f1 = aligned_prf(hyp, gt_ink)
            cer_s, cer_l = cer_pair(hyp, gt_ink)
            wer_s, _ = wer_pair(hyp, gt_ink)
            per_model[model] = dict(
                fragment_id=doc_id, model=model, failure_mode=mode,
                ngram_precision=round(ngram_p, 4),
                aligned_precision=round(precision, 4),
                aligned_recall=round(recall, 4),
                aligned_f1=round(f1, 4),
                cer=round(cer_s, 4), cer_lenient=round(cer_l, 4), wer=round(wer_s, 4),
                len_ratio=round(len(hyp) / len(gt_ink), 3) if gt_ink else None,
                gt_letters=len(gt_letters), hyp_letters=len(hyp_letters),
                repaired_gt=bool(d.get("_repaired_duplication")),
            )

        # Tier A requires convergence of the order-blind system AND a VLM.
        kraken_ok = per_model.get(_ORDER_BLIND, {}).get("ngram_precision", 0) >= TIER_A_OVERLAP
        vlm_ok = any(per_model.get(m, {}).get("ngram_precision", 0) >= TIER_A_OVERLAP
                     for m in _VLM_EVIDENCE)
        tier = "A" if (kraken_ok and vlm_ok) else "B"
        tiers[doc_id] = tier
        for r in per_model.values():
            r["tier"] = tier
            rows.append(r)

    if not rows:
        print("No saved outputs to score yet.")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    long_csv = args.out_dir / "genizah_offline_long.csv"
    with open(long_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_frag = len({r["fragment_id"] for r in rows})
    n_a = sum(1 for t in tiers.values() if t == "A")
    print(f"scored {n_frag} fragments   Tier A: {n_a}   Tier B: {len(tiers) - n_a}\n")

    by_model = collections.defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    summary = []
    print(f"{'model':30s} {'n':>3s} {'abst':>5s} {'loop':>5s} {'hall':>5s} "
          f"{'subst':>5s} {'ngramP':>7s} {'F1':>6s} {'CER*':>6s}")
    for model, rs in sorted(by_model.items()):
        modes = collections.Counter(r["failure_mode"] for r in rs)
        subst = [r for r in rs if r["failure_mode"] == "substantive"]
        cer_sub = statistics.median([r["cer"] for r in subst]) if subst else None
        rec = dict(
            model=model, n=len(rs),
            pct_abstained=round(modes["abstained"] / len(rs), 3),
            pct_loop_collapse=round(modes["loop_collapse"] / len(rs), 3),
            pct_hallucinated=round(modes["hallucinated"] / len(rs), 3),
            pct_substantive=round(modes["substantive"] / len(rs), 3),
            ngram_precision_median=round(statistics.median(
                [r["ngram_precision"] for r in rs]), 4),
            aligned_f1_median=round(statistics.median([r["aligned_f1"] for r in rs]), 4),
            cer_median_substantive=round(cer_sub, 4) if cer_sub is not None else None,
            n_substantive=len(subst),
        )
        summary.append(rec)
        print(f"{model:30s} {rec['n']:3d} {rec['pct_abstained']:5.0%} "
              f"{rec['pct_loop_collapse']:5.0%} {rec['pct_hallucinated']:5.0%} "
              f"{rec['pct_substantive']:5.0%} {rec['ngram_precision_median']:7.3f} "
              f"{rec['aligned_f1_median']:6.3f} "
              f"{(f'{cer_sub:.3f}' if cer_sub is not None else '  —'):>6s}")
    print("\n* CER median over substantive attempts only; Tier B fragments should be "
          "read via precision-family columns, not CER.")

    summary_csv = args.out_dir / "genizah_offline_summary.csv"
    with open(summary_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\nWrote {long_csv} and {summary_csv}")

    if args.no_wandb:
        return
    import wandb
    run = wandb.init(
        project="cairo-genizah-transcription", entity="igodfried",
        name=args.wandb_run_name, job_type="offline-scoring",
        config=dict(benchmark=args.benchmark, fragments=n_frag, tier_a=n_a,
                    abstain_min_chars=ABSTAIN_MIN_CHARS,
                    loop_repeat_ratio=LOOP_REPEAT_RATIO,
                    hallucination_ngram=HALLUCINATION_NGRAM,
                    tier_a_overlap=TIER_A_OVERLAP),
        tags=["paper", "genizah", "offline-scoring"],
    )
    wandb.log({
        "paper/genizah_offline_long": wandb.Table(
            columns=list(rows[0].keys()), data=[list(r.values()) for r in rows]),
        "paper/genizah_offline_summary": wandb.Table(
            columns=list(summary[0].keys()), data=[list(r.values()) for r in summary]),
    })
    print(f"W&B: {run.url}")
    wandb.finish()


if __name__ == "__main__":
    main()
