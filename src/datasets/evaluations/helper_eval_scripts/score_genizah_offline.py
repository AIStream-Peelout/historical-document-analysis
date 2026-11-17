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

Another spec in the same format (e.g. the sealed confirmation set) is scored with
``--spec <file> --outputs-dir <dir> --script-tags <sidecar> --out-dir <dir>``.
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
from typing import Optional

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
# Period-aware loop rule (2026-11-08). The 12-gram rule above sees only loops whose repeated unit is short: in an
# output that repeats one 45-letter line, any single 12-gram covers 12/45 of the text and stays under 0.45, so a
# runaway of thousands of letters was scored as a substantive answer. The second rule looks at long windows instead.
LOOP_WINDOW = 30             # letters per window
LOOP_WINDOW_REPEATS = 3      # a window seen at least this often is repeated
LOOP_WINDOW_SHARE = 0.5      # share of the output inside repeated windows ...
LOOP_LENGTH_RATIO = 2.0      # ... when the output holds at least this many times the ground truth's letters
LOOP_RULES = ("legacy", "period_aware")
#: Default loop rule. ``legacy`` (the 12-gram rule alone) reproduces every score computed before 2026-11-08,
#: including the paper's tables; ``period_aware`` adds the long-window rule.
LOOP_RULE = "period_aware"
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


def repeated_window_share(letters: str, window: int = LOOP_WINDOW, min_repeats: int = LOOP_WINDOW_REPEATS) -> float:
    """Share of the letters covered by windows of ``window`` letters that occur at least ``min_repeats`` times.

    Unlike :func:`loop_ratio`, the measure does not depend on the period of a loop: when one line of any length is
    repeated, almost every window inside the repeated part recurs. Ordinary text, formulae included, rarely holds a
    30-letter run three times.

    :param letters: Hypothesis reduced to letters.
    :type letters: str
    :param window: Window length in letters.
    :type window: int
    :param min_repeats: Occurrences from which a window counts as repeated.
    :type min_repeats: int
    :return: Ratio in [0, 1]; 0.0 for texts shorter than ``window * min_repeats``.
    :rtype: float
    """
    if window <= 0 or len(letters) < window * min_repeats:
        return 0.0
    grams = [letters[i:i + window] for i in range(len(letters) - window + 1)]
    counts = collections.Counter(grams)
    delta = [0] * (len(letters) + 1)
    for i, gram in enumerate(grams):
        if counts[gram] >= min_repeats:
            delta[i] += 1
            delta[i + window] -= 1
    covered = depth = 0
    for step in delta[:-1]:
        depth += step
        covered += depth > 0
    return covered / len(letters)


def is_long_period_loop(hyp_asserted: str, gt_letters: str) -> bool:
    """The period-aware loop rule: an output much longer than its page and made mostly of repeated windows.

    The length condition keeps a page whose own text repeats (a refrain, a list of like items) from counting as a loop
    when it is read at its true length. Without a ground truth the rule does not apply.

    :param hyp_asserted: Hypothesis reduced to asserted letters (Hebrew + Arabic).
    :type hyp_asserted: str
    :param gt_letters: Ground truth reduced to letters (empty: unknown).
    :type gt_letters: str
    :return: True when the output holds at least :data:`LOOP_LENGTH_RATIO` times the ground truth's letters and
        :func:`repeated_window_share` reaches :data:`LOOP_WINDOW_SHARE`.
    :rtype: bool
    """
    if not gt_letters or len(hyp_asserted) < LOOP_LENGTH_RATIO * len(gt_letters):
        return False
    return repeated_window_share(hyp_asserted) >= LOOP_WINDOW_SHARE


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


# Script-agnostic assertion test: abstention means the model asserted no
# page content in ANY plausible script.  Judging assertion by Hebrew letters
# alone silently reclassifies Arabic-script output (script confusion, e.g.
# looping Basmala formulae) as abstention, hiding hallucinated content.
# Scoring stays Hebrew-side: wrong-script assertions fall through to the
# loop/hallucination checks because their Hebrew n-gram precision is ~0.
_ASSERTED_RE = re.compile(r"[֐-׿؀-ۿݐ-ݿ]")


def asserted_letters(text: str) -> str:
    """Reduce text to letters in any page-plausible script (Hebrew + Arabic).

    :param text: Normalised hypothesis text.
    :type text: str
    :return: Concatenated Hebrew and Arabic letters, in order.
    :rtype: str
    """
    text = re.sub(r"\[\.\.\.\]|\(!\)|\[\?\]", " ", text)
    return "".join(_ASSERTED_RE.findall(text))


def classify(raw: str, hyp_asserted: str, gt_letters: str, ngram_p: float, loop_rule: str = LOOP_RULE) -> str:
    """Assign a failure mode to one model output.

    :param raw: Raw model output as saved.
    :type raw: str
    :param hyp_asserted: Normalised hypothesis reduced to script-agnostic
        asserted letters (Hebrew + Arabic).
    :type hyp_asserted: str
    :param gt_letters: Ground truth reduced to letters.
    :type gt_letters: str
    :param ngram_p: Order-independent n-gram precision.
    :type ngram_p: float
    :param loop_rule: ``legacy`` (the 12-gram rule) or ``period_aware`` (also :func:`is_long_period_loop`).
    :type loop_rule: str
    :return: One of abstained / loop_collapse / hallucinated / substantive.
    :rtype: str
    :raises ValueError: On an unknown loop rule.
    """
    if loop_rule not in LOOP_RULES:
        raise ValueError(f"loop rule {loop_rule!r}: expected one of {LOOP_RULES}")
    if len(hyp_asserted) < ABSTAIN_MIN_CHARS or _REFUSAL_RE.search(raw[:400]):
        return "abstained"
    if loop_ratio(hyp_asserted) > LOOP_REPEAT_RATIO:
        return "loop_collapse"
    if loop_rule == "period_aware" and is_long_period_loop(hyp_asserted, gt_letters):
        return "loop_collapse"
    if ngram_p < HALLUCINATION_NGRAM:
        return "hallucinated"
    return "substantive"


def load_benchmark(which: str, spec_path: Optional[Path] = None) -> list:
    """Load benchmark documents with duplicated ground truths repaired.

    :param which: ``verified`` to use the audited subset when present,
        otherwise the frozen 150.
    :type which: str
    :param spec_path: Another spec in the same format (``{"docs": [...]}``); overrides ``which``.
    :type spec_path: Optional[Path]
    :return: List of document dicts with repaired ``gt``.
    :rtype: list
    """
    path = _BENCH_DIR / ("genizah_test_v1_verified.json" if which == "verified"
                         else "genizah_test_v1.json")
    if not path.exists():
        path = _BENCH_DIR / "genizah_test_v1.json"
    if spec_path is not None:
        path = spec_path
    spec = json.load(open(path))
    docs = []
    for d in spec["docs"]:
        nd = dict(d)
        if not d.get("_repaired_duplication") and find_self_duplication(d["gt"]) > 0.55:
            nd["gt"] = repair_duplication(d["gt"])
            nd["_repaired_duplication"] = True
        docs.append(nd)
    return docs


def scoring_rule(loop_rule: str) -> dict:
    """The failure-mode thresholds a scoring run used, for the record next to its tables.

    :param loop_rule: The loop rule of the run.
    :type loop_rule: str
    :return: ``{loop_rule, abstain_min_chars, loop_repeat_ratio, loop_window, loop_window_repeats,
        loop_window_share, loop_length_ratio, hallucination_ngram}``.
    :rtype: dict
    """
    return {"loop_rule": loop_rule, "abstain_min_chars": ABSTAIN_MIN_CHARS, "loop_repeat_ratio": LOOP_REPEAT_RATIO,
            "loop_window": LOOP_WINDOW, "loop_window_repeats": LOOP_WINDOW_REPEATS,
            "loop_window_share": LOOP_WINDOW_SHARE, "loop_length_ratio": LOOP_LENGTH_RATIO,
            "hallucination_ngram": HALLUCINATION_NGRAM}


def write_scoring_rule(out_dir: Path, loop_rule: str, name: str = "scoring_rule.json") -> Path:
    """Write :func:`scoring_rule` next to a run's tables.

    :param out_dir: The run's output folder.
    :type out_dir: Path
    :param loop_rule: The loop rule of the run.
    :type loop_rule: str
    :param name: File name.
    :type name: str
    :return: The written file.
    :rtype: Path
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(scoring_rule(loop_rule), indent=1) + "\n", encoding="utf-8")
    return path


def main() -> None:
    """Score every saved output and emit the paper tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("frozen", "verified"), default="verified")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("transcription_results/paper_table"))
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-run-name", default="genizah-offline-scoring-v1")
    parser.add_argument("--script-tags", type=Path,
                        default=_BENCH_DIR / "genizah_test_v1_script_tags.json",
                        help="EVAL-ONLY language/script sidecar "
                             "(build_script_tags.py); enables the per-script "
                             "stratification table")
    parser.add_argument("--spec", type=Path, default=None,
                        help="score another spec in the verified-JSON format "
                             "(overrides --benchmark)")
    parser.add_argument("--outputs-dir", type=Path, default=_OUTPUTS,
                        help="per-fragment output folders (<dir>/<doc_id>/<model>.txt)")
    parser.add_argument("--loop-rule", choices=LOOP_RULES, default=LOOP_RULE,
                        help="loop detection: legacy (one repeated 12-gram, as scored before 2026-11-08) or "
                             "period_aware (also long repeated windows in an output twice the page's length)")
    args = parser.parse_args()
    print(f"loop rule: {args.loop_rule}")

    script_tags = {}
    if args.script_tags.exists():
        script_tags = {k: v.get("bucket", "untagged")
                       for k, v in json.load(open(args.script_tags)).items()}

    docs = load_benchmark(args.benchmark, args.spec)
    rows = []
    tiers = {}

    for d in docs:
        doc_id = d["doc_id"]
        outdir = args.outputs_dir / doc_id
        if not outdir.is_dir():
            continue
        gt_ink = genizah_visible_ink_gt(d["gt"])
        gt_letters = letters_only(gt_ink)
        if len(gt_letters) < 50:
            continue

        per_model = {}
        for f in sorted(outdir.glob("*.txt")):
            model = f.stem
            if model in ("ground_truth", "consensus") or model.endswith("_flat"):
                continue
            raw = f.read_text(errors="replace")
            hyp = normalize_ink_hypothesis(raw)
            hyp_letters = letters_only(hyp)
            ngram_p = ngram_precision(hyp_letters, gt_letters)
            mode = classify(raw, asserted_letters(hyp), gt_letters, ngram_p, loop_rule=args.loop_rule)
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
                script_bucket=script_tags.get(doc_id, "untagged"),
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
    write_scoring_rule(args.out_dir, args.loop_rule)
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

    by_script = []
    if script_tags:
        groups = collections.defaultdict(list)
        for r in rows:
            groups[(r["model"], r["script_bucket"])].append(r)
        print(f"\nPer-script stratification (eval-only sidecar; "
              f"{len(set(script_tags.values()))} buckets):")
        print(f"{'model':30s} {'bucket':14s} {'n':>3s} {'subst':>6s} "
              f"{'ngramP':>7s} {'F1':>6s}")
        for (model, bucket), rs in sorted(groups.items()):
            modes = collections.Counter(r["failure_mode"] for r in rs)
            rec = dict(
                model=model, script_bucket=bucket, n=len(rs),
                pct_substantive=round(modes["substantive"] / len(rs), 3),
                pct_abstained=round(modes["abstained"] / len(rs), 3),
                pct_loop_collapse=round(modes["loop_collapse"] / len(rs), 3),
                pct_hallucinated=round(modes["hallucinated"] / len(rs), 3),
                ngram_precision_median=round(statistics.median(
                    [r["ngram_precision"] for r in rs]), 4),
                aligned_f1_median=round(statistics.median(
                    [r["aligned_f1"] for r in rs]), 4),
            )
            by_script.append(rec)
            print(f"{model:30s} {bucket:14s} {rec['n']:3d} "
                  f"{rec['pct_substantive']:6.0%} "
                  f"{rec['ngram_precision_median']:7.3f} "
                  f"{rec['aligned_f1_median']:6.3f}")
        by_script_csv = args.out_dir / "genizah_offline_by_script.csv"
        with open(by_script_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(by_script[0].keys()))
            w.writeheader()
            w.writerows(by_script)
        print(f"Wrote {by_script_csv}")

    if args.no_wandb:
        return
    import wandb
    run = wandb.init(
        project="cairo-genizah-transcription", entity="igodfried",
        name=args.wandb_run_name, job_type="offline-scoring",
        config=dict(benchmark=args.spec.stem if args.spec else args.benchmark,
                    fragments=n_frag, tier_a=n_a,
                    abstain_min_chars=ABSTAIN_MIN_CHARS,
                    loop_repeat_ratio=LOOP_REPEAT_RATIO,
                    hallucination_ngram=HALLUCINATION_NGRAM,
                    tier_a_overlap=TIER_A_OVERLAP),
        tags=["paper", "genizah", "offline-scoring"],
    )
    tables = {
        "paper/genizah_offline_long": wandb.Table(
            columns=list(rows[0].keys()), data=[list(r.values()) for r in rows]),
        "paper/genizah_offline_summary": wandb.Table(
            columns=list(summary[0].keys()), data=[list(r.values()) for r in summary]),
    }
    if by_script:
        tables["paper/genizah_offline_by_script"] = wandb.Table(
            columns=list(by_script[0].keys()),
            data=[list(r.values()) for r in by_script])
    wandb.log(tables)
    print(f"W&B: {run.url}")
    wandb.finish()


if __name__ == "__main__":
    main()
