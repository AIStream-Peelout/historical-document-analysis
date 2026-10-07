"""Analysis B: clipped vs unclipped n-gram precision.

The paper's hallucination gate counts a hypothesis 5-gram as "supported" if it
appears anywhere in the reference, no matter how many times the hypothesis
repeats it.  BLEU-style clipping caps each window's credit at its reference
count, so text copied repeatedly (a repeated formula, a duplicated Kraken line,
a degenerate loop that survived the loop gate) stops earning precision once the
reference's occurrences are used up.

This script quantifies the resulting gap (unclipped - clipped) per system, the
behaviour-classification flips it causes at the paper's 0.10 cutoff, the Tier A
count under clipped scoring, per-fragment worst offenders with a repetition
signature, the Spearman correlation between the gap and hypothesis length
inflation, and the same gap at n=3 / n=8.

Outputs (all under ``$SCRATCH/B``): ``B_clip.md``, ``clip_long.csv`` and
``pairs/<system>/<rank>_<doc_id>.txt``.
"""

import collections
import csv
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.datasets.evaluations.helper_eval_scripts.ngram_audit.common import (
    Fragment,
    Output,
    PAPER_CONFIG,
    PAPER_SYSTEMS,
    ScoringConfig,
    cached_features,
    ngram_precision,
    ngrams,
    score,
    tier_a_count,
)

SCRATCH = Path("/private/tmp/claude-501/-Users-isaac-Documents-GitHub-historical-document-analysis"
               "/30e3ec54-dba4-4651-b60d-6ac04c3a0d9c/scratchpad/ngram_audit")
OUT_DIR = SCRATCH / "B"
CACHE = SCRATCH / "features_paper_systems.pkl"

LETTER_SET = "hebrew"
MAIN_N = 5
EXTRA_NS = (3, 8)
GAP_BANDS = (0.05, 0.10, 0.20)
DETAIL_SYSTEMS = ["kraken_seg", "gemini_pro", "qwen3_vl_8b_heb_v17_step800"]
SHORT_SYSTEMS = ["claude_opus_4_8", "gemini_flash"]
TOP_K = 15
PAPER_TIER_A = 55


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of ``values``.

    :param values: Sample values.
    :type values: Sequence[float]
    :param q: Percentile in [0, 100].
    :type q: float
    :return: Percentile, or ``nan`` for an empty sample.
    :rtype: float
    """
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    """Spearman rank correlation with a two-sided p-value.

    ``scipy`` is present in this repo's virtualenv (checked: 1.15.3) and is used
    for the p-value; the rho is cross-checked against a numpy rank correlation
    by :func:`spearman_rho_numpy` in the self-test at the bottom of ``main``.

    :param xs: First sample.
    :type xs: Sequence[float]
    :param ys: Second sample, same length as ``xs``.
    :type ys: Sequence[float]
    :return: ``(rho, p_value)``.
    :rtype: tuple
    """
    if len(xs) < 3:
        return float("nan"), float("nan")
    from scipy import stats
    res = stats.spearmanr(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))
    return float(res.statistic), float(res.pvalue)


def spearman_rho_numpy(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rho computed from average ranks with numpy only.

    :param xs: First sample.
    :type xs: Sequence[float]
    :param ys: Second sample, same length as ``xs``.
    :type ys: Sequence[float]
    :return: Rho, or ``nan`` when a sample is constant or too short.
    :rtype: float
    """
    if len(xs) < 3:
        return float("nan")
    rx, ry = _rankdata(xs), _rankdata(ys)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _rankdata(values: Sequence[float]) -> np.ndarray:
    """Average-tie ranks of ``values`` (numpy-only ``scipy.stats.rankdata``).

    :param values: Sample values.
    :type values: Sequence[float]
    :return: Ranks, 1-based, ties averaged.
    :rtype: numpy.ndarray
    """
    arr = np.asarray(values, dtype=float)
    order = arr.argsort(kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
    # Average the ranks inside each tie group.
    srt = arr[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def overused_windows(hyp_letters: str, gt_letters: str, n: int) -> List[Tuple[str, int, int]]:
    """Hypothesis n-grams used more often than the reference contains them.

    :param hyp_letters: Hypothesis reduced to letters.
    :type hyp_letters: str
    :param gt_letters: Reference reduced to letters.
    :type gt_letters: str
    :param n: Window size.
    :type n: int
    :return: ``(window, hyp_count, gt_count)`` sorted by excess then hyp count.
    :rtype: list
    """
    hyp_counts = collections.Counter(ngrams(hyp_letters, n))
    gt_counts = collections.Counter(ngrams(gt_letters, n))
    over = [(g, c, gt_counts.get(g, 0)) for g, c in hyp_counts.items()
            if c > gt_counts.get(g, 0)]
    over.sort(key=lambda t: (t[1] - t[2], t[1]), reverse=True)
    return over


def repetition_signature(hyp_letters: str, gt_letters: str, n: int) -> Dict[str, object]:
    """Most repeated hypothesis window plus the count of over-used windows.

    :param hyp_letters: Hypothesis reduced to letters.
    :type hyp_letters: str
    :param gt_letters: Reference reduced to letters.
    :type gt_letters: str
    :param n: Window size.
    :type n: int
    :return: ``top_gram``, ``top_hyp_count``, ``top_gt_count``,
        ``n_overused_grams`` and ``excess_grams`` (total clipped-away windows).
    :rtype: dict
    """
    hyp_counts = collections.Counter(ngrams(hyp_letters, n))
    gt_counts = collections.Counter(ngrams(gt_letters, n))
    if not hyp_counts:
        return dict(top_gram="", top_hyp_count=0, top_gt_count=0,
                    n_overused_grams=0, excess_grams=0)
    top_gram, top_hyp = hyp_counts.most_common(1)[0]
    over = [(c - gt_counts.get(g, 0)) for g, c in hyp_counts.items()
            if c > gt_counts.get(g, 0)]
    return dict(top_gram=top_gram, top_hyp_count=top_hyp,
                top_gt_count=gt_counts.get(top_gram, 0),
                n_overused_grams=len(over), excess_grams=sum(over))


def build_rows(frags: List[Fragment], outs: List[Output]) -> List[dict]:
    """Per-output long rows with clipped/unclipped precision at n=5, 3 and 8.

    :param frags: Benchmark fragments.
    :type frags: list
    :param outs: Outputs with cached features.
    :type outs: list
    :return: One dict per (fragment, system).
    :rtype: list
    """
    paper_rows = {(r["fragment_id"], r["model"]): r
                  for r in score(frags, outs, PAPER_CONFIG)}
    clip_cfg = ScoringConfig(clip=True)
    clip_rows = {(r["fragment_id"], r["model"]): r
                 for r in score(frags, outs, clip_cfg)}
    by_doc = {fr.doc_id: fr for fr in frags}

    rows: List[dict] = []
    for o in outs:
        fr = by_doc[o.doc_id]
        gt_l = fr.gt_letters[LETTER_SET]
        hyp_l = o.hyp_letters[LETTER_SET]
        pr = paper_rows[(o.doc_id, o.model)]
        cr = clip_rows[(o.doc_id, o.model)]
        unclipped = ngram_precision(hyp_l, gt_l, MAIN_N, clip=False)
        clipped = ngram_precision(hyp_l, gt_l, MAIN_N, clip=True)
        row = dict(
            doc_id=o.doc_id, model=o.model, n=MAIN_N,
            unclipped=unclipped, clipped=clipped, gap=unclipped - clipped,
            hyp_letters=len(hyp_l), gt_letters=len(gt_l),
            len_ratio=(len(hyp_l) / len(gt_l)) if gt_l else float("nan"),
            len_diff=len(hyp_l) - len(gt_l),
            aligned_precision=o.aligned_precision, aligned_recall=o.aligned_recall,
            cer=o.cer, tier=pr["tier"], tier_clipped=cr["tier"],
            script_bucket=fr.script_bucket,
            mode_paper=pr["failure_mode"], mode_clipped=cr["failure_mode"],
        )
        row.update(repetition_signature(hyp_l, gt_l, MAIN_N))
        for n in EXTRA_NS:
            u = ngram_precision(hyp_l, gt_l, n, clip=False)
            c = ngram_precision(hyp_l, gt_l, n, clip=True)
            row[f"unclipped_n{n}"] = u
            row[f"clipped_n{n}"] = c
            row[f"gap_n{n}"] = u - c
        rows.append(row)
    return rows


def gap_stats(gaps: Sequence[float]) -> Dict[str, float]:
    """Summary statistics of a gap sample.

    :param gaps: Gap values.
    :type gaps: Sequence[float]
    :return: ``n``, ``mean``, ``median``, ``p90``, ``max`` and the band counts.
    :rtype: dict
    """
    if not gaps:
        return dict(n=0, mean=float("nan"), median=float("nan"),
                    p90=float("nan"), max=float("nan"),
                    **{f"ge_{b:.2f}": 0 for b in GAP_BANDS})
    out = dict(n=len(gaps), mean=statistics.fmean(gaps), median=statistics.median(gaps),
               p90=percentile(gaps, 90), max=max(gaps))
    for b in GAP_BANDS:
        out[f"ge_{b:.2f}"] = sum(1 for g in gaps if g >= b)
    return out


def write_long_csv(rows: List[dict], path: Path) -> None:
    """Write every per-output row to CSV.

    :param rows: Rows from :func:`build_rows`.
    :type rows: list
    :param path: Destination CSV path.
    :type path: Path
    :return: ``None``
    :rtype: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in sorted(rows, key=lambda r: (r["model"], -r["gap"])):
            writer.writerow(r)


def write_pair_file(row: dict, frag: Fragment, out: Output, rank: int, path: Path) -> None:
    """Dump one hypothesis/reference pair plus its over-used windows.

    :param row: The row for this (fragment, system).
    :type row: dict
    :param frag: The fragment (for ``gt_ink``).
    :type frag: Fragment
    :param out: The output (for ``hyp``).
    :type out: Output
    :param rank: 1-based rank within the system's gap ordering.
    :type rank: int
    :param path: Destination text file.
    :type path: Path
    :return: ``None``
    :rtype: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    over = overused_windows(out.hyp_letters[LETTER_SET], frag.gt_letters[LETTER_SET], MAIN_N)
    lines = [
        f"rank={rank} doc_id={row['doc_id']} system={row['model']}",
        (f"unclipped={row['unclipped']:.3f} clipped={row['clipped']:.3f} "
         f"gap={row['gap']:.3f} tier={row['tier']} bucket={row['script_bucket']}"),
        (f"hyp_letters={row['hyp_letters']} gt_letters={row['gt_letters']} "
         f"len_ratio={row['len_ratio']:.3f} aligned_P={row['aligned_precision']:.3f} "
         f"aligned_R={row['aligned_recall']:.3f} cer={row['cer']:.3f}"),
        (f"mode_paper={row['mode_paper']} mode_clipped={row['mode_clipped']} "
         f"distinct_overused_5grams={row['n_overused_grams']} "
         f"excess_5gram_windows={row['excess_grams']}"),
        "",
        "=== TOP 10 OVER-USED 5-GRAMS (hyp_count > gt_count) ===",
    ]
    for gram, hc, gc in over[:10]:
        lines.append(f"  {gram!r}  hyp={hc}  gt={gc}  excess={hc - gc}")
    lines += ["", "=== HYPOTHESIS (normalised) ===", out.hyp,
              "", "=== REFERENCE (visible ink) ===", frag.gt_ink, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _md_table(header: Sequence[str], body: Sequence[Sequence[object]]) -> str:
    """Render a GitHub markdown table.

    :param header: Column headings.
    :type header: Sequence[str]
    :param body: Row cells (already formatted).
    :type body: Sequence[Sequence[object]]
    :return: Markdown text.
    :rtype: str
    """
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in body:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _f(x: Optional[float], nd: int = 3) -> str:
    """Round a float for a markdown cell.

    :param x: Value (may be ``None``/``nan``).
    :type x: float or None
    :param nd: Decimal places.
    :type nd: int
    :return: Formatted string.
    :rtype: str
    """
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    return f"{x:.{nd}f}"


def build_markdown(rows: List[dict], systems: Sequence[str],
                   tier_a_clipped: int) -> str:
    """Assemble the full report markdown.

    :param rows: Rows from :func:`build_rows`.
    :type rows: list
    :param systems: System order.
    :type systems: Sequence[str]
    :param tier_a_clipped: Tier A fragment count under clipped scoring.
    :type tier_a_clipped: int
    :return: Markdown document.
    :rtype: str
    """
    by_model: Dict[str, List[dict]] = collections.defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    parts = [
        "# Analysis B — clipped vs unclipped n-gram precision",
        "",
        (f"Metric: character {MAIN_N}-gram precision, `{LETTER_SET}` letter set, "
         "131 verified fragments x 11 paper systems. "
         "`unclipped` = paper metric (window ∈ set of reference windows); "
         "`clipped` = BLEU-style modified precision (each window earns credit at "
         "most as often as the reference contains it). `gap = unclipped - clipped`."),
        "",
        "## 1a. Gap over all outputs, per system",
        "",
    ]
    head = ["system", "n", "mean gap", "median gap", "p90 gap", "max gap",
            "gap>=0.05", "gap>=0.10", "gap>=0.20"]
    body = []
    for m in systems:
        st = gap_stats([r["gap"] for r in by_model.get(m, [])])
        body.append([m, st["n"], _f(st["mean"]), _f(st["median"]), _f(st["p90"]),
                     _f(st["max"]), st["ge_0.05"], st["ge_0.10"], st["ge_0.20"]])
    parts += [_md_table(head, body), "",
              "## 1b. Same, restricted to outputs substantive under PAPER_CONFIG", ""]
    body = []
    for m in systems:
        subs = [r for r in by_model.get(m, []) if r["mode_paper"] == "substantive"]
        st = gap_stats([r["gap"] for r in subs])
        body.append([m, st["n"], _f(st["mean"]), _f(st["median"]), _f(st["p90"]),
                     _f(st["max"]), st["ge_0.05"], st["ge_0.10"], st["ge_0.20"]])
    parts += [_md_table(head, body), "",
              "## 1c. Behaviour re-classification under clipping (same 0.10 cutoff)", ""]
    flip_head = ["system", "substantive (paper)", "substantive (clipped)",
                 "subst -> hallucinated flips", "flipped doc_ids"]
    flip_body = []
    for m in systems:
        rs = by_model.get(m, [])
        flips = [r for r in rs if r["mode_paper"] == "substantive"
                 and r["mode_clipped"] == "hallucinated"]
        flip_body.append([
            m,
            sum(1 for r in rs if r["mode_paper"] == "substantive"),
            sum(1 for r in rs if r["mode_clipped"] == "substantive"),
            len(flips),
            ", ".join(sorted(r["doc_id"] for r in flips)) or "—",
        ])
    parts += [_md_table(flip_head, flip_body), "",
              "## 1d. Tier A (0.25 convergence cutoff)", "",
              _md_table(["scoring", "Tier A fragments"],
                        [["unclipped (paper)", PAPER_TIER_A],
                         ["clipped", tier_a_clipped]]),
              ""]

    parts += ["## 2. Largest-gap fragments per system", ""]
    detail_head = ["#", "doc_id", "unclipped", "clipped", "gap", "hyp L", "gt L",
                   "len ratio", "aligned P", "aligned R", "tier", "bucket",
                   "top 5-gram hyp/gt", "distinct over-used 5-grams"]
    for m in DETAIL_SYSTEMS:
        top = sorted(by_model.get(m, []), key=lambda r: -r["gap"])[:TOP_K]
        body = []
        for i, r in enumerate(top, 1):
            body.append([i, r["doc_id"], _f(r["unclipped"]), _f(r["clipped"]),
                         _f(r["gap"]), r["hyp_letters"], r["gt_letters"],
                         _f(r["len_ratio"]), _f(r["aligned_precision"]),
                         _f(r["aligned_recall"]), r["tier"], r["script_bucket"],
                         f"{r['top_hyp_count']}/{r['top_gt_count']}",
                         r["n_overused_grams"]])
        parts += [f"### {m}", "",
                  ("Top 5-gram column = occurrences of the hypothesis's single most "
                   "repeated 5-gram in the hypothesis / in the reference. "
                   "The window itself (Hebrew text) is in the pair files."),
                  "", _md_table(detail_head, body), ""]

    short_head = ["#", "doc_id", "unclipped", "clipped", "gap", "len ratio",
                  "top 5-gram hyp/gt", "distinct over-used 5-grams"]
    for m in SHORT_SYSTEMS:
        top = sorted(by_model.get(m, []), key=lambda r: -r["gap"])[:TOP_K]
        body = [[i, r["doc_id"], _f(r["unclipped"]), _f(r["clipped"]), _f(r["gap"]),
                 _f(r["len_ratio"]), f"{r['top_hyp_count']}/{r['top_gt_count']}",
                 r["n_overused_grams"]]
                for i, r in enumerate(top, 1)]
        parts += [f"### {m} (short form)", "", _md_table(short_head, body), ""]

    parts += ["## 4. What drives the gap", "",
              ("Spearman rho of gap against hypothesis length inflation, per system "
               "(all 131 outputs each). `len_ratio` = hyp letters / gt letters; "
               "`len_diff` = hyp letters - gt letters."), ""]
    corr_head = ["system", "rho(gap, len_ratio)", "p", "rho(gap, len_diff)", "p",
                 "mean gap, len_ratio<=1.05", "mean gap, len_ratio>1.05",
                 "n short/normal", "n long"]
    corr_body = []
    for m in systems:
        rs = by_model.get(m, [])
        gaps = [r["gap"] for r in rs]
        rho1, p1 = spearman_rho(gaps, [r["len_ratio"] for r in rs])
        rho2, p2 = spearman_rho(gaps, [float(r["len_diff"]) for r in rs])
        short = [r["gap"] for r in rs if r["len_ratio"] <= 1.05]
        long_ = [r["gap"] for r in rs if r["len_ratio"] > 1.05]
        corr_body.append([m, _f(rho1), _f(p1, 4), _f(rho2), _f(p2, 4),
                          _f(statistics.fmean(short) if short else float("nan")),
                          _f(statistics.fmean(long_) if long_ else float("nan")),
                          len(short), len(long_)])
    parts += [_md_table(corr_head, corr_body), ""]

    pooled = rows
    rho_all, p_all = spearman_rho([r["gap"] for r in pooled],
                                  [r["len_ratio"] for r in pooled])
    rho_all2, p_all2 = spearman_rho([r["gap"] for r in pooled],
                                    [float(r["len_diff"]) for r in pooled])
    short_all = [r["gap"] for r in pooled if r["len_ratio"] <= 1.05]
    long_all = [r["gap"] for r in pooled if r["len_ratio"] > 1.05]
    parts += [
        (f"Pooled over all {len(pooled)} outputs: rho(gap, len_ratio) = {_f(rho_all)} "
         f"(p = {_f(p_all, 4)}), rho(gap, len_diff) = {_f(rho_all2)} "
         f"(p = {_f(p_all2, 4)}). Mean gap for len_ratio <= 1.05: "
         f"{_f(statistics.fmean(short_all) if short_all else float('nan'))} "
         f"(n = {len(short_all)}); for len_ratio > 1.05: "
         f"{_f(statistics.fmean(long_all) if long_all else float('nan'))} "
         f"(n = {len(long_all)})."),
        "",
    ]

    big = [r for r in pooled if r["gap"] >= 0.05]
    big_long = [r for r in big if r["len_ratio"] > 1.05]
    big_short = [r for r in big if r["len_ratio"] <= 1.05]
    by_sys_short = collections.Counter(r["model"] for r in big_short)
    parts += [
        (f"**Reading.** Of the {len(big)} outputs with gap >= 0.05, {len(big_long)} "
         f"({len(big_long) / len(big):.0%}) are longer than the reference "
         f"(len_ratio > 1.05) — the over-generation / loop channel, dominated by "
         f"HebVL-1.7 — but {len(big_short)} are NOT: "
         + ", ".join(f"{m} x{c}" for m, c in by_sys_short.most_common())
         + ". Those are short or normal-length outputs (len_ratio "
         f"{min(r['len_ratio'] for r in big_short):.2f}-"
         f"{max(r['len_ratio'] for r in big_short):.2f}) that recycle a formula "
         "internally, so clipping bites there too. The correlation is real but "
         "modest (pooled rho = "
         f"{_f(rho_all)}), and the per-system rho is driven by a handful of long "
         "outputs: for the two Qwen baselines, where most outputs are long for "
         "unrelated reasons, rho is ~0."),
        "",
        "## 5. Gap at n=3 and n=8", "",
    ]
    n_head = ["system"]
    for n in (EXTRA_NS[0], MAIN_N, EXTRA_NS[1]):
        n_head += [f"mean gap n={n}", f"p90 gap n={n}"]
    n_body = []
    for m in systems:
        rs = by_model.get(m, [])
        cells: List[object] = [m]
        for n in (EXTRA_NS[0], MAIN_N, EXTRA_NS[1]):
            key = "gap" if n == MAIN_N else f"gap_n{n}"
            gs = [r[key] for r in rs]
            cells += [_f(statistics.fmean(gs) if gs else float("nan")),
                      _f(percentile(gs, 90))]
        n_body.append(cells)
    parts += [_md_table(n_head, n_body), ""]
    return "\n".join(parts)


def main() -> None:
    """Run analysis B end to end and write every output file.

    :return: ``None``
    :rtype: None
    """
    frags, outs = cached_features(CACHE)
    rows = build_rows(frags, outs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    write_long_csv(rows, OUT_DIR / "clip_long.csv")

    tier_a_clipped = tier_a_count(score(frags, outs, ScoringConfig(clip=True)))

    by_doc = {fr.doc_id: fr for fr in frags}
    by_key = {(o.doc_id, o.model): o for o in outs}
    by_model: Dict[str, List[dict]] = collections.defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    n_pairs = 0
    for m in DETAIL_SYSTEMS:
        top = sorted(by_model.get(m, []), key=lambda r: -r["gap"])[:TOP_K]
        for i, r in enumerate(top, 1):
            write_pair_file(r, by_doc[r["doc_id"]], by_key[(r["doc_id"], r["model"])],
                            i, OUT_DIR / "pairs" / m / f"{i:02d}_{r['doc_id']}.txt")
            n_pairs += 1

    md = build_markdown(rows, PAPER_SYSTEMS, tier_a_clipped)
    (OUT_DIR / "B_clip.md").write_text(md, encoding="utf-8")
    rho_scipy, _ = spearman_rho([r["gap"] for r in rows], [r["len_ratio"] for r in rows])
    rho_np = spearman_rho_numpy([r["gap"] for r in rows], [r["len_ratio"] for r in rows])
    print(f"rows={len(rows)} fragments={len(frags)} pairs={n_pairs} "
          f"tier_a_clipped={tier_a_clipped} "
          f"rho_check scipy={rho_scipy:.6f} numpy={rho_np:.6f}")
    print(md)


if __name__ == "__main__":
    main()
