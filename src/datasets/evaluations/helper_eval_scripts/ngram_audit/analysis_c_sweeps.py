"""Analysis C: threshold and n sweeps for the behaviour classification, Tier A
and the paper's headline claims.

Every setting is one-factor-at-a-time from :data:`common.PAPER_CONFIG` (items
1-5), plus a 27-point grid over the two factors that matter most (item 6:
``n`` x ``halluc_cutoff``, in the unclipped/Hebrew, clipped/Hebrew and
unclipped/semitic variants).

Outputs (all under ``$SCRATCH/C/``):

* ``C_sweeps.md`` -- per-setting behaviour tables, Tier A counts, claim
  verdicts, the two condensed appendix tables (T1, T2), claim margins at the
  paper setting and the first-failure hallucination cutoff per claim.
* ``sweep_summary_long.csv`` -- one row per (setting, system) with every
  summary column.
* ``tier_changes.csv`` -- one row per (setting, fragment) whose tier differs
  from the paper setting.

Run::

    .venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.ngram_audit.analysis_c_sweeps
"""

import csv
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.datasets.evaluations.helper_eval_scripts.ngram_audit.common import (
    FRONTIER_VLMS,
    HEBVL17,
    PAPER_CONFIG,
    PAPER_SYSTEMS,
    ScoringConfig,
    cached_features,
    headline_claims,
    score,
    summarise,
    tier_a_count,
)

SCRATCH = Path("/private/tmp/claude-501/-Users-isaac-Documents-GitHub-historical-document-analysis"
               "/30e3ec54-dba4-4651-b60d-6ac04c3a0d9c/scratchpad/ngram_audit")
OUT_DIR = SCRATCH / "C"
CACHE = SCRATCH / "features_paper_systems.pkl"

#: Display order and short labels for the five claims carried through every table.
CLAIM_KEYS = [
    "kraken_ge_frontier_substantive",
    "hebvl17_gt_frontier_substantive",
    "gemini_pro_best_frontier_substantive",
    "kraken_ngram_gt_frontier",
    "hebvl17_ngram_gt_frontier",
]
CLAIM_LABELS = {
    "kraken_ge_frontier_substantive": "C1 kraken_seg %subst >= best frontier",
    "hebvl17_gt_frontier_substantive": "C2 HebVL-1.7 %subst > best frontier",
    "gemini_pro_best_frontier_substantive": "C3 gemini_pro %subst = best frontier",
    "kraken_ngram_gt_frontier": "C4 kraken_seg ngram median > every frontier",
    "hebvl17_ngram_gt_frontier": "C5 HebVL-1.7 ngram median > every frontier",
}


def all_claims(summary: Dict[str, dict]) -> Dict[str, bool]:
    """The three paper claims plus the two n-gram-median claims.

    The extra claims are computed exactly like the substantive-share ones:
    strict ``>`` against the best frontier VLM.

    :param summary: Output of :func:`common.summarise`.
    :type summary: dict
    :return: claim key -> holds?
    :rtype: dict
    """
    claims = dict(headline_claims(summary))
    frontier_ngram = [summary[m]["ngram_precision_median"]
                      for m in FRONTIER_VLMS if m in summary]
    best = max(frontier_ngram)
    claims["kraken_ngram_gt_frontier"] = (
        summary["kraken_seg"]["ngram_precision_median"] > best)
    claims["hebvl17_ngram_gt_frontier"] = (
        summary[HEBVL17]["ngram_precision_median"] > best)
    return claims


def claim_margins(summary: Dict[str, dict]) -> Dict[str, Tuple[float, float, float, str]]:
    """Margin in percentage points behind each claim.

    For claims whose subject is itself a frontier VLM (C3) the comparison is
    against the best *other* frontier system, so the margin is informative.

    :param summary: Output of :func:`common.summarise`.
    :type summary: dict
    :return: claim key -> (subject value, comparison value, gap in pp, comparison system).
    :rtype: dict
    """
    def best_other(field: str, exclude: Optional[str] = None) -> Tuple[float, str]:
        pairs = [(summary[m][field], m) for m in FRONTIER_VLMS
                 if m in summary and m != exclude]
        return max(pairs)

    margins: Dict[str, Tuple[float, float, float, str]] = {}
    for key, subject, field, exclude in [
        ("kraken_ge_frontier_substantive", "kraken_seg", "pct_substantive", None),
        ("hebvl17_gt_frontier_substantive", HEBVL17, "pct_substantive", None),
        ("gemini_pro_best_frontier_substantive", "gemini_pro", "pct_substantive",
         "gemini_pro"),
        ("kraken_ngram_gt_frontier", "kraken_seg", "ngram_precision_median", None),
        ("hebvl17_ngram_gt_frontier", HEBVL17, "ngram_precision_median", None),
    ]:
        val = summary[subject][field]
        cmp_val, cmp_model = best_other(field, exclude)
        margins[key] = (val, cmp_val, 100.0 * (val - cmp_val), cmp_model)
    return margins


def single_factor_settings() -> List[Tuple[str, ScoringConfig]]:
    """Items 1-5: one-factor-at-a-time variations of the paper config.

    :return: (column label, config) in table order, paper baseline first.
    :rtype: list
    """
    settings: List[Tuple[str, ScoringConfig]] = [("paper", PAPER_CONFIG)]
    for n in (3, 4, 5, 6, 8):
        settings.append((f"n={n}", ScoringConfig(n=n)))
    for h in (0.05, 0.10, 0.15, 0.20, 0.30):
        settings.append((f"h={h:.2f}", ScoringConfig(halluc_cutoff=h)))
    for t in (0.15, 0.20, 0.25, 0.30, 0.40):
        settings.append((f"t={t:.2f}", ScoringConfig(tier_cutoff=t)))
    for lo in (0.30, 0.45, 0.60):
        settings.append((f"l={lo:.2f}", ScoringConfig(loop_cutoff=lo)))
    for a in (15, 25, 50):
        settings.append((f"a={a}", ScoringConfig(abstain_chars=a)))
    return settings


def grid_settings() -> List[Tuple[str, ScoringConfig]]:
    """Item 6: n x halluc_cutoff grid in three metric variants (27 settings).

    :return: (column label, config) in table order.
    :rtype: list
    """
    settings: List[Tuple[str, ScoringConfig]] = []
    for variant, kwargs in [("unclip", {}),
                            ("clip", {"clip": True}),
                            ("sem", {"letter_set": "semitic"})]:
        for n in (3, 5, 8):
            for h in (0.05, 0.10, 0.20):
                label = f"{variant} n{n} h{h:.2f}"
                settings.append((label, ScoringConfig(n=n, halluc_cutoff=h, **kwargs)))
    return settings


def tier_map(rows: List[dict]) -> Dict[str, str]:
    """Fragment -> tier for one scored row set.

    :param rows: Output of :func:`common.score`.
    :type rows: list
    :return: fragment id -> ``A`` or ``B``.
    :rtype: dict
    """
    return {r["fragment_id"]: r["tier"] for r in rows}


def fmt(value: Optional[float], places: int = 3) -> str:
    """Round for markdown, rendering ``None``/NaN as an em dash.

    :param value: Number or ``None``.
    :type value: float or None
    :param places: Decimal places.
    :type places: int
    :return: Formatted cell.
    :rtype: str
    """
    if value is None:
        return "--"
    if value != value:  # NaN
        return "--"
    return f"{value:.{places}f}"


def behaviour_table(summary: Dict[str, dict]) -> List[str]:
    """Markdown behaviour-distribution table for one setting.

    :param summary: Output of :func:`common.summarise`.
    :type summary: dict
    :return: Markdown lines.
    :rtype: list
    """
    lines = ["| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m in PAPER_SYSTEMS:
        s = summary[m]
        lines.append(
            f"| {m} | {s['n']} | {fmt(s['pct_abstained'])} | {fmt(s['pct_loop_collapse'])} "
            f"| {fmt(s['pct_hallucinated'])} | {fmt(s['pct_substantive'])} "
            f"| {fmt(s['ngram_precision_median'])} | {fmt(s['aligned_f1_median'])} "
            f"| {fmt(s['cer_median_substantive'])} |")
    return lines


def condensed_table(labels: Sequence[str], summaries: Dict[str, Dict[str, dict]],
                    tiers: Dict[str, int], claims: Dict[str, Dict[str, bool]],
                    tier_changed: Dict[str, int],
                    blank_labels: Sequence[str] = ()) -> List[str]:
    """Build a condensed table: systems x settings with %substantive cells.

    :param labels: Column labels in order.
    :type labels: Sequence[str]
    :param summaries: label -> summary dict.
    :type summaries: dict
    :param tiers: label -> Tier A count.
    :type tiers: dict
    :param claims: label -> claim verdicts.
    :type claims: dict
    :param tier_changed: label -> number of fragments whose tier moved vs paper.
    :type tier_changed: dict
    :param blank_labels: Columns where the %substantive cells are, by
        construction, identical to the paper column and are therefore blanked.
    :type blank_labels: Sequence[str]
    :return: Markdown lines.
    :rtype: list
    """
    head = "| system | " + " | ".join(labels) + " |"
    rule = "|---" * (len(labels) + 1) + "|"
    lines = [head, rule]
    for m in PAPER_SYSTEMS:
        cells = []
        for lb in labels:
            if lb in blank_labels:
                cells.append("=")
            else:
                cells.append(fmt(summaries[lb][m]["pct_substantive"]))
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append("| **Tier A count** | " + " | ".join(str(tiers[lb]) for lb in labels) + " |")
    lines.append("| **Tier changes vs paper** | "
                 + " | ".join(str(tier_changed[lb]) for lb in labels) + " |")
    for key in CLAIM_KEYS:
        row = " | ".join("TRUE" if claims[lb][key] else "FALSE" for lb in labels)
        lines.append(f"| {CLAIM_LABELS[key]} | {row} |")
    return lines


def first_failure_cutoff(frags: list, outs: list,
                         cutoffs: Sequence[float]) -> Dict[str, Optional[float]]:
    """Lowest hallucination cutoff (n=5, unclipped) at which each claim fails.

    :param frags: Fragments.
    :type frags: list
    :param outs: Outputs.
    :type outs: list
    :param cutoffs: Cutoffs to test, ascending.
    :type cutoffs: Sequence[float]
    :return: claim key -> first failing cutoff, or ``None`` if it never fails.
    :rtype: dict
    """
    first: Dict[str, Optional[float]] = {k: None for k in CLAIM_KEYS}
    for c in cutoffs:
        cfg = ScoringConfig(halluc_cutoff=c)
        verdicts = all_claims(summarise(score(frags, outs, cfg), PAPER_SYSTEMS))
        for key, holds in verdicts.items():
            if not holds and first[key] is None:
                first[key] = c
    return first


def run() -> None:
    """Execute every sweep and write the CSVs and the markdown report.

    :return: ``None``.
    :rtype: None
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frags, outs = cached_features(CACHE)

    paper_rows = score(frags, outs, PAPER_CONFIG)
    paper_tiers = tier_map(paper_rows)
    paper_summary = summarise(paper_rows, PAPER_SYSTEMS)

    singles = single_factor_settings()
    grid = grid_settings()
    all_settings = singles + grid

    summaries: Dict[str, Dict[str, dict]] = {}
    tiers: Dict[str, int] = {}
    claims: Dict[str, Dict[str, bool]] = {}
    changed_counts: Dict[str, int] = {}
    changed_rows: List[dict] = []
    long_rows: List[dict] = []
    cfg_by_label: Dict[str, ScoringConfig] = {}

    for label, cfg in all_settings:
        rows = score(frags, outs, cfg)
        summary = summarise(rows, PAPER_SYSTEMS)
        summaries[label] = summary
        tiers[label] = tier_a_count(rows)
        claims[label] = all_claims(summary)
        cfg_by_label[label] = cfg

        tmap = tier_map(rows)
        moved = [(fid, paper_tiers[fid], tmap[fid]) for fid in sorted(tmap)
                 if tmap[fid] != paper_tiers[fid]]
        changed_counts[label] = len(moved)
        for fid, old, new in moved:
            changed_rows.append(dict(setting=label, config=cfg.label(),
                                     fragment_id=fid, paper_tier=old, new_tier=new))

        for m in PAPER_SYSTEMS:
            s = summary[m]
            long_rows.append(dict(setting=label, config=cfg.label(), model=m, n=s["n"],
                                  pct_abstained=s["pct_abstained"],
                                  pct_loop_collapse=s["pct_loop_collapse"],
                                  pct_hallucinated=s["pct_hallucinated"],
                                  pct_substantive=s["pct_substantive"],
                                  ngram_precision_median=s["ngram_precision_median"],
                                  aligned_f1_median=s["aligned_f1_median"],
                                  cer_median_substantive=s["cer_median_substantive"],
                                  n_substantive=s["n_substantive"],
                                  tier_a_count=tiers[label],
                                  tier_changes_vs_paper=changed_counts[label]))

    with open(OUT_DIR / "sweep_summary_long.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(long_rows[0].keys()))
        writer.writeheader()
        writer.writerows(long_rows)

    with open(OUT_DIR / "tier_changes.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["setting", "config", "fragment_id",
                                                "paper_tier", "new_tier"])
        writer.writeheader()
        writer.writerows(changed_rows)

    margins = claim_margins(paper_summary)
    cutoffs = [round(0.05 * i, 2) for i in range(1, 13)]
    first_fail = first_failure_cutoff(frags, outs, cutoffs)

    md: List[str] = []
    md.append("# Analysis C — threshold and n sweeps")
    md.append("")
    md.append(f"Benchmark: {len(frags)} verified fragments x {len(PAPER_SYSTEMS)} "
              f"paper systems = {len(outs)} outputs. "
              f"Paper config: `{PAPER_CONFIG.label()}` (Tier A = {tiers['paper']}).")
    md.append("")
    md.append("Claim keys used throughout:")
    md.append("")
    for key in CLAIM_KEYS:
        md.append(f"* **{CLAIM_LABELS[key]}** (`{key}`)")
    md.append("")
    md.append("`%subst` = share of that system's outputs classified substantive; "
              "`ngram med` = median n-gram precision over all its outputs; `F1 med` = "
              "median aligned F1 (config-independent, shown for reference); "
              "`CER med (subst)` = median CER over substantive outputs only.")
    md.append("")
    coverage = ", ".join(f"{m} {paper_summary[m]['n']}" for m in PAPER_SYSTEMS)
    md.append("**Coverage caveat:** systems do not all cover all 131 fragments — "
              f"{coverage}. Every share below is over the system's own denominator, "
              "exactly as in the paper.")
    md.append("")

    md.append("## T1 — sensitivity of substantive share (single-factor settings)")
    md.append("")
    md.append("Cells = %substantive. `=` marks columns that cannot change the behaviour "
              "classification by construction (`tier_cutoff` only feeds tier assignment).")
    md.append("")
    single_labels = [lb for lb, _ in singles]
    blank = [lb for lb in single_labels if lb.startswith("t=")]
    md.extend(condensed_table(single_labels, summaries, tiers, claims,
                              changed_counts, blank))
    md.append("")

    md.append("## T2 — Tier A count and claims under the 27-setting n x halluc grid")
    md.append("")
    grid_labels = [lb for lb, _ in grid]
    for variant, title in [("unclip", "unclipped, Hebrew letter set (paper metric family)"),
                           ("clip", "BLEU-style clipped counts, Hebrew letter set"),
                           ("sem", "unclipped, semitic (Hebrew+Arabic) letter set")]:
        cols = [lb for lb in grid_labels if lb.startswith(variant + " ")]
        md.append(f"### {title}")
        md.append("")
        md.extend(condensed_table(cols, summaries, tiers, claims, changed_counts))
        md.append("")

    md.append("## Claim margins at the paper setting")
    md.append("")
    md.append("| claim | subject value | best comparison | comparison system | gap (pp) |")
    md.append("|---|---|---|---|---|")
    for key in CLAIM_KEYS:
        val, cmp_val, gap, cmp_model = margins[key]
        md.append(f"| {CLAIM_LABELS[key]} | {fmt(val)} | {fmt(cmp_val)} | {cmp_model} "
                  f"| {gap:+.1f} |")
    md.append("")
    md.append("C3's comparison excludes gemini_pro itself (it is a frontier VLM), so the "
              "gap is against the runner-up frontier system.")
    md.append("")

    md.append("## First hallucination cutoff at which each claim fails")
    md.append("")
    md.append(f"Searched {cutoffs[0]:.2f}-{cutoffs[-1]:.2f} in steps of 0.05, n=5 unclipped, "
              "all other thresholds at the paper values.")
    md.append("")
    md.append("| claim | holds at paper setting | first failing cutoff |")
    md.append("|---|---|---|")
    paper_claims = claims["paper"]
    for key in CLAIM_KEYS:
        c = first_fail[key]
        md.append(f"| {CLAIM_LABELS[key]} | "
                  f"{'TRUE' if paper_claims[key] else 'FALSE'} | "
                  + (f"{c:.2f}" if c is not None else "never fails in range") + " |")
    md.append("")
    md.append("C4 and C5 compare medians of n-gram precision, which do not depend on the "
              "hallucination cutoff at all: C4 holds and C5 fails at every cutoff, "
              "including the paper's.")
    md.append("")

    md.append("## Per-setting detail")
    md.append("")
    for label, cfg in all_settings:
        md.append(f"### {label} — `{cfg.label()}`")
        md.append("")
        md.extend(behaviour_table(summaries[label]))
        md.append("")
        md.append(f"Tier A = **{tiers[label]}** (paper = {tiers['paper']}); "
                  f"fragments whose tier changed vs paper: **{changed_counts[label]}** "
                  "(ids in `tier_changes.csv`).")
        md.append("")
        verdicts = " · ".join(
            f"{CLAIM_LABELS[k].split(' ', 1)[0]}: {'TRUE' if claims[label][k] else 'FALSE'}"
            for k in CLAIM_KEYS)
        md.append(f"Claims — {verdicts}")
        md.append("")

    (OUT_DIR / "C_sweeps.md").write_text("\n".join(md) + "\n")

    n_frag = len({r["fragment_id"] for r in paper_rows})
    print(f"settings scored: {len(all_settings)}  outputs: {len(outs)}  fragments: {n_frag}")
    print(f"paper Tier A: {tiers['paper']}  "
          f"kraken %subst: {paper_summary['kraken_seg']['pct_substantive']:.3f}  "
          f"gemini_pro %subst: {paper_summary['gemini_pro']['pct_substantive']:.3f}  "
          f"HebVL1.7 %subst: {paper_summary[HEBVL17]['pct_substantive']:.3f}")
    print("tier A range across settings: "
          f"{min(tiers.values())}-{max(tiers.values())}")
    print("claim failures: " + ", ".join(
        f"{k}={sum(1 for lb in summaries if not claims[lb][k])}/{len(summaries)}"
        for k in CLAIM_KEYS))
    print(f"median tier-change count: "
          f"{statistics.median(changed_counts.values()):.1f}")
    print(f"wrote {OUT_DIR}/C_sweeps.md, sweep_summary_long.csv "
          f"({len(long_rows)} rows), tier_changes.csv ({len(changed_rows)} rows)")


if __name__ == "__main__":
    run()
