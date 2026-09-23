"""Parameterised re-scorer for the Genizah benchmark (n-gram metric audit).

A flag-driven variant of ``score_genizah_offline.py``.  With no flags it must
reproduce the paper summary table exactly; ``--check-paper`` asserts that.
Never writes into the paper table directory.

Usage (from the repo root):
    .venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.ngram_audit.score_variant \\
        --check-paper
    .venv/bin/python -m ...score_variant --n 5 --clip --letter-set semitic \\
        --halluc-cutoff 0.10 --tier-cutoff 0.25 --out-dir /path/to/scratch
"""

import argparse
import csv
import sys
from pathlib import Path

from src.datasets.evaluations.helper_eval_scripts.ngram_audit.common import (
    PAPER_SUMMARY,
    PAPER_SYSTEMS,
    ScoringConfig,
    headline_claims,
    load_features,
    load_fragments,
    score,
    summarise,
    tier_a_count,
)


def check_against_paper(summary: dict, tier_a: int) -> bool:
    """Compare a default-config summary with the committed paper CSV.

    :param summary: Output of :func:`summarise`.
    :type summary: dict
    :param tier_a: Tier A count from the variant scorer.
    :type tier_a: int
    :return: True when every shared cell matches to the CSV's rounding.
    :rtype: bool
    """
    ok = True
    with open(PAPER_SUMMARY) as fh:
        for row in csv.DictReader(fh):
            m = row["model"]
            if m not in summary:
                print(f"  missing in variant: {m}")
                ok = False
                continue
            s = summary[m]
            for key, nd in (("pct_abstained", 3), ("pct_loop_collapse", 3),
                            ("pct_hallucinated", 3), ("pct_substantive", 3),
                            ("ngram_precision_median", 4), ("aligned_f1_median", 4),
                            ("cer_median_substantive", 4)):
                want = row[key]
                got = s[key]
                got_s = "" if got is None else f"{round(got, nd)}"
                if want != got_s and not (want and got is not None
                                          and abs(float(want) - got) < 10 ** -nd + 1e-9):
                    print(f"  MISMATCH {m} {key}: paper={want} variant={got_s}")
                    ok = False
            if int(row["n"]) != s["n"] or int(row["n_substantive"]) != s["n_substantive"]:
                print(f"  MISMATCH {m} n/n_substantive")
                ok = False
    if tier_a != 55:
        print(f"  MISMATCH tier A: paper=55 variant={tier_a}")
        ok = False
    return ok


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=("verified", "frozen"), default="verified")
    ap.add_argument("--n", type=int, default=ScoringConfig.n)
    ap.add_argument("--clip", action="store_true")
    ap.add_argument("--letter-set", choices=("hebrew", "semitic"), default="hebrew")
    ap.add_argument("--halluc-cutoff", type=float, default=ScoringConfig.halluc_cutoff)
    ap.add_argument("--tier-cutoff", type=float, default=ScoringConfig.tier_cutoff)
    ap.add_argument("--loop-cutoff", type=float, default=ScoringConfig.loop_cutoff)
    ap.add_argument("--loop-span", type=int, default=ScoringConfig.loop_span)
    ap.add_argument("--abstain-chars", type=int, default=ScoringConfig.abstain_chars)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write long/summary CSVs here (never the paper dir)")
    ap.add_argument("--check-paper", action="store_true",
                    help="assert the default config reproduces the paper CSV")
    ap.add_argument("--paper-systems-only", action="store_true")
    args = ap.parse_args()

    cfg = ScoringConfig(n=args.n, clip=args.clip, letter_set=args.letter_set,
                        halluc_cutoff=args.halluc_cutoff, tier_cutoff=args.tier_cutoff,
                        loop_cutoff=args.loop_cutoff, loop_span=args.loop_span,
                        abstain_chars=args.abstain_chars)
    frags = load_fragments(args.benchmark)
    outs = load_features(frags, PAPER_SYSTEMS if args.paper_systems_only else None)
    rows = score(frags, outs, cfg)
    summary = summarise(rows)
    n_a = tier_a_count(rows)
    print(f"config: {cfg.label()}   fragments={len(frags)}  Tier A={n_a}")
    print(f"{'model':30s} {'n':>3s} {'abst':>5s} {'loop':>5s} {'hall':>5s} {'subst':>5s} "
          f"{'ngramP':>7s} {'F1':>6s} {'CER*':>6s}")
    for m, s in summary.items():
        cer = s["cer_median_substantive"]
        print(f"{m:30s} {s['n']:3d} {s['pct_abstained']:5.0%} {s['pct_loop_collapse']:5.0%} "
              f"{s['pct_hallucinated']:5.0%} {s['pct_substantive']:5.0%} "
              f"{s['ngram_precision_median']:7.3f} {s['aligned_f1_median']:6.3f} "
              f"{(f'{cer:.3f}' if cer is not None else '  —'):>6s}")
    if all(m in summary for m in PAPER_SYSTEMS):
        print("claims:", headline_claims(summary))

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        with open(args.out_dir / "long.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with open(args.out_dir / "summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(next(iter(summary.values())).keys()))
            w.writeheader()
            w.writerows(summary.values())
        print(f"wrote {args.out_dir}/long.csv and summary.csv")

    if args.check_paper:
        if cfg != ScoringConfig():
            print("--check-paper requires the default config")
            sys.exit(2)
        ok = check_against_paper(summary, n_a)
        print("PAPER REPRODUCTION:", "OK" if ok else "FAILED")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
