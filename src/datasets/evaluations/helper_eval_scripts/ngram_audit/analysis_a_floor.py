"""Analysis A: the cross-document "formulaic floor" of the paper's n-gram metric.

The paper's behaviour gate scores a transcription by the share of its character
5-grams (Hebrew letters only, order-independent, unclipped) that occur anywhere
in the reference of the *same* fragment.  Because Genizah texts are highly
formulaic, a hypothesis may earn a sizeable score against references it has
never seen.  This script measures that floor directly: every substantive output
(under :data:`common.PAPER_CONFIG`) is scored against every *other* fragment's
ground truth, and the resulting wrong-reference distribution is compared with
the own-reference score.

Outputs (written to ``$SCRATCH/A``):

* ``pairs_n5.csv`` -- the full (doc_id, model, ref_doc_id, precision) matrix at
  n=5 unclipped, own-reference pairs included.
* ``A_floor.md`` -- the self-contained markdown tables.

Run with::

    .venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.ngram_audit.analysis_a_floor
"""

import collections
import csv
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.datasets.evaluations.helper_eval_scripts.ngram_audit.common import (
    Fragment,
    Output,
    PAPER_CONFIG,
    PAPER_SYSTEMS,
    cached_features,
    ngram_precision,
    ngrams,
    score,
)

SCRATCH = Path("/private/tmp/claude-501/-Users-isaac-Documents-GitHub-historical-document-analysis"
               "/30e3ec54-dba4-4651-b60d-6ac04c3a0d9c/scratchpad/ngram_audit")
OUT_DIR = SCRATCH / "A"
CACHE = SCRATCH / "features_paper_systems.pkl"

LETTER_SET = "hebrew"
N_SWEEP = [3, 4, 6, 8]
HALLUC_CUTOFF = PAPER_CONFIG.halluc_cutoff      # 0.10
TIER_CUTOFF = PAPER_CONFIG.tier_cutoff          # 0.25


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (numpy convention) without numpy.

    :param values: Sample values (need not be sorted); must be non-empty.
    :type values: Sequence[float]
    :param q: Percentile in [0, 100].
    :type q: float
    :return: The interpolated percentile.
    :rtype: float
    """
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


class RefIndex:
    """Reference-side n-gram index for one window size.

    :param frags: Benchmark fragments.
    :type frags: List[Fragment]
    :param n: Window size.
    :type n: int
    :param letter_set: Letter set key into ``Fragment.gt_letters``.
    :type letter_set: str
    """

    def __init__(self, frags: List[Fragment], n: int, letter_set: str = LETTER_SET) -> None:
        self.n = n
        self.doc_ids: List[str] = [fr.doc_id for fr in frags]
        self.sets: Dict[str, set] = {}
        self.counts: Dict[str, collections.Counter] = {}
        self.too_short: Dict[str, bool] = {}
        for fr in frags:
            gl = fr.gt_letters[letter_set]
            self.too_short[fr.doc_id] = len(gl) < n
            grams = ngrams(gl, n)
            self.counts[fr.doc_id] = collections.Counter(grams)
            self.sets[fr.doc_id] = set(grams)


class HypIndex:
    """Hypothesis-side n-gram summary for one output at one window size.

    Stores the unique-gram set plus the duplicate tail so that the precision
    against any reference reduces to one C-level set intersection plus a short
    loop over the (rare) repeated grams.

    :param hyp_letters: Hypothesis reduced to letters.
    :type hyp_letters: str
    :param n: Window size.
    :type n: int
    """

    def __init__(self, hyp_letters: str, n: int) -> None:
        self.n = n
        self.total = max(len(hyp_letters) - n + 1, 0)
        self.too_short = len(hyp_letters) < n
        counts = collections.Counter(ngrams(hyp_letters, n))
        self.uniq: set = set(counts)
        self.dups: List[Tuple[str, int]] = [(g, c) for g, c in counts.items() if c > 1]

    def precision(self, ref: RefIndex, ref_doc: str, clip: bool = False) -> float:
        """Order-independent n-gram precision against one reference.

        Byte-equivalent to :func:`common.ngram_precision` at the same ``n`` /
        ``clip``, including the ``0.0`` short-input convention.

        :param ref: Reference index at the same ``n``.
        :type ref: RefIndex
        :param ref_doc: Reference fragment id.
        :type ref_doc: str
        :param clip: Apply BLEU-style count clipping.
        :type clip: bool
        :return: Precision in [0, 1].
        :rtype: float
        """
        if self.too_short or ref.too_short[ref_doc]:
            return 0.0
        gt_set = ref.sets[ref_doc]
        matched = len(self.uniq & gt_set)
        if self.dups:
            if clip:
                gt_counts = ref.counts[ref_doc]
                for gram, cnt in self.dups:
                    gt_c = gt_counts.get(gram, 0)
                    if gt_c:
                        matched += min(cnt, gt_c) - 1
            else:
                for gram, cnt in self.dups:
                    if gram in gt_set:
                        matched += cnt - 1
        return matched / self.total


def substantive_outputs(frags: List[Fragment], outs: List[Output]) -> List[Output]:
    """Select the outputs classified ``substantive`` under the paper config.

    :param frags: Benchmark fragments.
    :type frags: List[Fragment]
    :param outs: All (fragment, system) outputs.
    :type outs: List[Output]
    :return: The substantive subset, in input order.
    :rtype: List[Output]
    """
    rows = score(frags, outs, PAPER_CONFIG)
    keep = {(r["fragment_id"], r["model"]) for r in rows
            if r["failure_mode"] == "substantive"}
    return [o for o in outs if (o.doc_id, o.model) in keep]


def verify_fast_path(frags: List[Fragment], subs: List[Output], n: int = 5,
                     sample: int = 40) -> None:
    """Assert the indexed precision equals :func:`common.ngram_precision`.

    :param frags: Benchmark fragments.
    :type frags: List[Fragment]
    :param subs: Substantive outputs.
    :type subs: List[Output]
    :param n: Window size to check.
    :type n: int
    :param sample: Number of (output, reference) pairs to check per clip mode.
    :type sample: int
    :return: Nothing; raises ``AssertionError`` on mismatch.
    :rtype: None
    """
    ref = RefIndex(frags, n)
    step = max(len(subs) // sample, 1)
    for clip in (False, True):
        for out in subs[::step]:
            hyp = HypIndex(out.hyp_letters[LETTER_SET], n)
            for fr in frags[::max(len(frags) // 5, 1)]:
                fast = hyp.precision(ref, fr.doc_id, clip=clip)
                slow = ngram_precision(out.hyp_letters[LETTER_SET],
                                       fr.gt_letters[LETTER_SET], n, clip=clip)
                assert abs(fast - slow) < 1e-12, (out.doc_id, out.model, fr.doc_id, clip)


def compute_pairs(frags: List[Fragment], subs: List[Output], n: int,
                  clip: bool) -> List[Tuple[str, str, str, float]]:
    """Score every substantive output against every fragment's reference.

    :param frags: Benchmark fragments (the 131 references).
    :type frags: List[Fragment]
    :param subs: Substantive outputs.
    :type subs: List[Output]
    :param n: Window size.
    :type n: int
    :param clip: Apply count clipping.
    :type clip: bool
    :return: Rows of (doc_id, model, ref_doc_id, precision), own-ref included.
    :rtype: List[Tuple[str, str, str, float]]
    """
    ref = RefIndex(frags, n)
    pairs: List[Tuple[str, str, str, float]] = []
    for out in subs:
        hyp = HypIndex(out.hyp_letters[LETTER_SET], n)
        for ref_doc in ref.doc_ids:
            pairs.append((out.doc_id, out.model, ref_doc,
                          hyp.precision(ref, ref_doc, clip=clip)))
    return pairs


def system_rows(pairs: Sequence[Tuple[str, str, str, float]],
                systems: Sequence[str], full: bool = True) -> List[dict]:
    """Aggregate a pair matrix into per-system wrong-/own-reference statistics.

    :param pairs: Rows from :func:`compute_pairs`.
    :type pairs: Sequence[tuple]
    :param systems: System order.
    :type systems: Sequence[str]
    :param full: Include the per-output max-over-wrong-refs columns.
    :type full: bool
    :return: One dict per system.
    :rtype: List[dict]
    """
    wrong: Dict[str, List[float]] = collections.defaultdict(list)
    own: Dict[str, List[float]] = collections.defaultdict(list)
    per_out_max: Dict[str, Dict[str, float]] = collections.defaultdict(dict)
    for doc_id, model, ref_doc, p in pairs:
        if ref_doc == doc_id:
            own[model].append(p)
            continue
        wrong[model].append(p)
        cur = per_out_max[model].get(doc_id)
        if cur is None or p > cur:
            per_out_max[model][doc_id] = p
    rows = []
    for m in systems:
        w = wrong.get(m, [])
        if not w:
            continue
        o = own.get(m, [])
        maxes = list(per_out_max[m].values())
        row = dict(
            model=m, n_substantive=len(o), n_wrong_pairs=len(w),
            wrong_median=statistics.median(w), wrong_p90=percentile(w, 90),
            wrong_p99=percentile(w, 99), wrong_max=max(w),
            share_wrong_ge_halluc=sum(p >= HALLUC_CUTOFF for p in w) / len(w),
            share_wrong_ge_tier=sum(p >= TIER_CUTOFF for p in w) / len(w),
            own_median=statistics.median(o) if o else float("nan"),
        )
        if full:
            row.update(
                maxwrong_median=statistics.median(maxes),
                maxwrong_p90=percentile(maxes, 90),
                maxwrong_max=max(maxes),
                n_maxwrong_ge_halluc=sum(p >= HALLUC_CUTOFF for p in maxes),
                n_maxwrong_ge_tier=sum(p >= TIER_CUTOFF for p in maxes),
            )
        rows.append(row)
    return rows


def reference_rows(pairs: Sequence[Tuple[str, str, str, float]],
                   frags: List[Fragment], tier_a: set) -> List[dict]:
    """Aggregate the pair matrix by reference fragment (the per-reference floor).

    :param pairs: Rows from :func:`compute_pairs`.
    :type pairs: Sequence[tuple]
    :param frags: Benchmark fragments.
    :type frags: List[Fragment]
    :param tier_a: Fragment ids that are Tier A under the paper config.
    :type tier_a: set
    :return: One dict per reference, sorted by mean wrong-source precision.
    :rtype: List[dict]
    """
    by_ref: Dict[str, List[float]] = collections.defaultdict(list)
    for doc_id, _model, ref_doc, p in pairs:
        if ref_doc != doc_id:
            by_ref[ref_doc].append(p)
    meta = {fr.doc_id: fr for fr in frags}
    rows = []
    for ref_doc, vals in by_ref.items():
        fr = meta[ref_doc]
        rows.append(dict(
            ref_doc_id=ref_doc, script_bucket=fr.script_bucket,
            gt_letters=len(fr.gt_letters[LETTER_SET]),
            tier_a=ref_doc in tier_a, n_pairs=len(vals),
            mean_wrong=statistics.fmean(vals), max_wrong=max(vals),
            share_ge_halluc=sum(p >= HALLUC_CUTOFF for p in vals) / len(vals),
        ))
    rows.sort(key=lambda r: r["mean_wrong"], reverse=True)
    return rows


def md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a GitHub markdown table.

    :param headers: Column headers.
    :type headers: Sequence[str]
    :param rows: Row cells (already formatted or plain scalars).
    :type rows: Sequence[Sequence[object]]
    :return: Markdown text ending in a newline.
    :rtype: str
    """
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def f3(x: Optional[float]) -> str:
    """Format a float to 3 dp.

    :param x: Value (``None``/NaN tolerated).
    :type x: float or None
    :return: Formatted string.
    :rtype: str
    """
    return "n/a" if x is None else f"{x:.3f}"


def write_pairs_csv(path: Path, pairs: Sequence[Tuple[str, str, str, float]]) -> None:
    """Write the full pair matrix to CSV.

    :param path: Destination file.
    :type path: Path
    :param pairs: Rows from :func:`compute_pairs`.
    :type pairs: Sequence[tuple]
    :return: Nothing.
    :rtype: None
    """
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["doc_id", "model", "ref_doc_id", "precision"])
        for doc_id, model, ref_doc, p in pairs:
            w.writerow([doc_id, model, ref_doc, f"{p:.6f}"])


def build_markdown(frags: List[Fragment], subs: List[Output],
                   sys_rows: List[dict], ref_rows: List[dict],
                   sweep: Dict[int, List[dict]], clipped_rows: List[dict]) -> str:
    """Assemble the self-contained report.

    :param frags: Benchmark fragments.
    :type frags: List[Fragment]
    :param subs: Substantive outputs.
    :type subs: List[Output]
    :param sys_rows: Per-system rows at n=5 unclipped.
    :type sys_rows: List[dict]
    :param ref_rows: Per-reference rows (sorted descending by mean).
    :type ref_rows: List[dict]
    :param sweep: n -> per-system rows for the n sweep.
    :type sweep: Dict[int, List[dict]]
    :param clipped_rows: Per-system rows at n=5 clipped.
    :type clipped_rows: List[dict]
    :return: Markdown document.
    :rtype: str
    """
    n_sub = len(subs)
    n_ref = len(frags)
    parts: List[str] = []
    parts.append(
        "# Analysis A — the formulaic floor of the order-independent 5-gram metric\n\n"
        f"Every output classified **substantive** under the paper config "
        f"(`{PAPER_CONFIG.label()}`) is scored with the exact paper metric "
        "(character n-grams over Hebrew-block letters, set membership in the "
        "reference, unclipped) against **every other** benchmark fragment's "
        "ground truth.  A wrong-reference score is pure floor: the hypothesis "
        "and the reference come from different manuscripts.\n\n"
        f"- substantive outputs: **{n_sub}** across {len(PAPER_SYSTEMS)} systems\n"
        f"- references: **{n_ref}** fragments (each output scored against "
        f"{n_ref - 1} wrong references + its own)\n"
        f"- wrong-reference pairs at n=5: **{n_sub * (n_ref - 1):,}**\n"
        f"- cutoffs: hallucination `< {HALLUC_CUTOFF:.2f}`, Tier A "
        f"`>= {TIER_CUTOFF:.2f}`\n")

    parts.append("\n## 1. Per-system wrong-reference floor (n=5, unclipped)\n\n")
    parts.append(md_table(
        ["system", "n subst.", "wrong pairs", "wrong med", "wrong p90",
         "wrong p99", "wrong max", "share >= 0.10", "share >= 0.25", "own med"],
        [[r["model"], r["n_substantive"], f"{r['n_wrong_pairs']:,}",
          f3(r["wrong_median"]), f3(r["wrong_p90"]), f3(r["wrong_p99"]),
          f3(r["wrong_max"]), f3(r["share_wrong_ge_halluc"]),
          f3(r["share_wrong_ge_tier"]), f3(r["own_median"])] for r in sys_rows]))

    parts.append("\n### 1b. Per-output maximum over the 130 wrong references\n\n"
                 "\"Would this output have passed the gate against the best "
                 "wrong manuscript in the benchmark?\"\n\n")
    parts.append(md_table(
        ["system", "n subst.", "max-wrong med", "max-wrong p90", "max-wrong max",
         "n outputs max >= 0.10", "n outputs max >= 0.25"],
        [[r["model"], r["n_substantive"], f3(r["maxwrong_median"]),
          f3(r["maxwrong_p90"]), f3(r["maxwrong_max"]),
          f"{r['n_maxwrong_ge_halluc']} ({r['n_maxwrong_ge_halluc'] / r['n_substantive']:.3f})",
          f"{r['n_maxwrong_ge_tier']} ({r['n_maxwrong_ge_tier'] / r['n_substantive']:.3f})"]
         for r in sys_rows]))

    parts.append("\n## 2. Per-reference floor (n=5, unclipped)\n\n"
                 "For each fragment used as a reference: mean and max n-gram "
                 "precision of all substantive outputs **of other fragments** "
                 "(all systems pooled).\n\n")
    ref_cols = ["ref doc_id", "script bucket", "gt letters", "Tier A", "pairs",
                "mean wrong", "max wrong", "share >= 0.10"]

    def ref_cells(r: dict) -> List[object]:
        return [r["ref_doc_id"], r["script_bucket"], r["gt_letters"],
                "yes" if r["tier_a"] else "no", r["n_pairs"],
                f3(r["mean_wrong"]), f3(r["max_wrong"]), f3(r["share_ge_halluc"])]

    parts.append("**15 most formulaic references (highest mean wrong-source "
                 "precision)**\n\n")
    parts.append(md_table(ref_cols, [ref_cells(r) for r in ref_rows[:15]]))
    parts.append("\n**15 least formulaic references**\n\n")
    parts.append(md_table(ref_cols, [ref_cells(r) for r in ref_rows[-15:][::-1]]))
    all_means = [r["mean_wrong"] for r in ref_rows]
    all_maxes = [r["max_wrong"] for r in ref_rows]
    parts.append(f"\nAcross all {len(ref_rows)} references: mean-of-means "
                 f"{f3(statistics.fmean(all_means))}, median-of-means "
                 f"{f3(statistics.median(all_means))}, "
                 f"p90-of-means {f3(percentile(all_means, 90))}, "
                 f"max-of-means {f3(max(all_means))}; "
                 f"{sum(m >= HALLUC_CUTOFF for m in all_maxes)} of "
                 f"{len(ref_rows)} references are matched at >= "
                 f"{HALLUC_CUTOFF:.2f} by at least one foreign output "
                 f"({sum(m >= TIER_CUTOFF for m in all_maxes)} at >= "
                 f"{TIER_CUTOFF:.2f}).\n")

    parts.append("\n## 3. How the floor moves with n (unclipped, same outputs)\n\n"
                 "Wrong-reference pairs only.  n=5 is the paper's setting.\n\n")
    ns = sorted(set(N_SWEEP) | {5})
    headers = ["system"] + [f"n={n} med / p90 / share>=0.10" for n in ns]
    by_n = {n: {r["model"]: r for r in rows} for n, rows in sweep.items()}
    by_n[5] = {r["model"]: r for r in sys_rows}
    rows_out = []
    for m in [r["model"] for r in sys_rows]:
        cells: List[object] = [m]
        for n in ns:
            r = by_n[n].get(m)
            cells.append("n/a" if r is None else
                         f"{f3(r['wrong_median'])} / {f3(r['wrong_p90'])} / "
                         f"{f3(r['share_wrong_ge_halluc'])}")
        rows_out.append(cells)
    parts.append(md_table(headers, rows_out))

    parts.append("\n## 4. Same table with clipping (n=5, clip=True)\n\n")
    parts.append(md_table(
        ["system", "n subst.", "wrong med", "wrong p90", "wrong p99",
         "wrong max", "share >= 0.10", "share >= 0.25", "own med"],
        [[r["model"], r["n_substantive"], f3(r["wrong_median"]),
          f3(r["wrong_p90"]), f3(r["wrong_p99"]), f3(r["wrong_max"]),
          f3(r["share_wrong_ge_halluc"]), f3(r["share_wrong_ge_tier"]),
          f3(r["own_median"])] for r in clipped_rows]))
    clip_map = {r["model"]: r for r in clipped_rows}
    deltas = [clip_map[r["model"]]["wrong_p90"] - r["wrong_p90"]
              for r in sys_rows if r["model"] in clip_map]
    parts.append(f"\nLargest absolute change in wrong-reference p90 from "
                 f"clipping: {f3(max(abs(d) for d in deltas))}.\n")

    parts.append("\n## 5. Floor vs signal — own-reference distribution "
                 "(n=5, unclipped)\n\n")
    parts.append(md_table(
        ["system", "n subst.", "own med", "own p10", "own min", "own max",
         "wrong p90", "own med / wrong p90"],
        [[r["model"], r["n_substantive"], f3(r["own_median"]), f3(r["own_p10"]),
          f3(r["own_min"]), f3(r["own_max"]), f3(r["wrong_p90"]),
          f3(r["own_median"] / r["wrong_p90"]) if r["wrong_p90"] > 0 else "inf"]
         for r in sys_rows]))
    return "".join(parts)


def notable_section(pairs: Sequence[Tuple[str, str, str, float]],
                    subs: List[Output], row_by_key: Dict[Tuple[str, str], dict],
                    top: int = 15) -> str:
    """Report the worst wrong-reference pairs and the short-hypothesis effect.

    :param pairs: Rows from :func:`compute_pairs` at n=5 unclipped.
    :type pairs: Sequence[tuple]
    :param subs: Substantive outputs.
    :type subs: List[Output]
    :param row_by_key: (doc_id, model) -> scored row under the paper config.
    :type row_by_key: Dict[Tuple[str, str], dict]
    :param top: Number of pairs to list.
    :type top: int
    :return: Markdown section.
    :rtype: str
    """
    hyp_len = {(o.doc_id, o.model): len(o.hyp_letters[LETTER_SET]) for o in subs}
    wrong = [p for p in pairs if p[0] != p[2]]
    wrong.sort(key=lambda t: t[3], reverse=True)
    cells = []
    for doc_id, model, ref_doc, p in wrong[:top]:
        r = row_by_key[(doc_id, model)]
        cells.append([doc_id, model, ref_doc, f3(p), hyp_len[(doc_id, model)],
                      r["gt_letters"], f3(r["ngram_precision"]), f3(r["cer"]),
                      f3(r["aligned_f1"])])
    out = ["\n## 6. Where the floor bites\n\n",
           f"Top {top} wrong-reference pairs (n=5, unclipped).  `own n-gram P` "
           "is the score the paper actually reports for that output; `CER` and "
           "`aligned F1` are the order-aware metrics on its own reference.\n\n"]
    out.append(md_table(
        ["source doc", "system", "wrong reference", "wrong P", "hyp letters",
         "ref gt letters", "own n-gram P", "own CER", "own aligned F1"], cells))
    per_out_max: Dict[Tuple[str, str], float] = collections.defaultdict(float)
    for doc_id, model, ref_doc, p in wrong:
        key = (doc_id, model)
        if p > per_out_max[key]:
            per_out_max[key] = p
    short = [v for k, v in per_out_max.items() if hyp_len[k] < 300]
    long_ = [v for k, v in per_out_max.items() if hyp_len[k] >= 300]
    out.append("\nShort hypotheses carry most of the floor risk:\n\n")
    out.append(md_table(
        ["hypothesis length", "n outputs", "median max-wrong", "share max-wrong >= 0.10",
         "share max-wrong >= 0.25"],
        [["< 300 Hebrew letters", len(short), f3(statistics.median(short)),
          f3(sum(v >= HALLUC_CUTOFF for v in short) / len(short)),
          f3(sum(v >= TIER_CUTOFF for v in short) / len(short))],
         [">= 300 Hebrew letters", len(long_), f3(statistics.median(long_)),
          f3(sum(v >= HALLUC_CUTOFF for v in long_) / len(long_)),
          f3(sum(v >= TIER_CUTOFF for v in long_) / len(long_))]]))
    return "".join(out)


def add_own_stats(sys_rows: List[dict],
                  pairs: Sequence[Tuple[str, str, str, float]]) -> None:
    """Attach own-reference percentiles to the per-system rows in place.

    :param sys_rows: Rows from :func:`system_rows`.
    :type sys_rows: List[dict]
    :param pairs: Rows from :func:`compute_pairs` at the same config.
    :type pairs: Sequence[tuple]
    :return: Nothing.
    :rtype: None
    """
    own: Dict[str, List[float]] = collections.defaultdict(list)
    for doc_id, model, ref_doc, p in pairs:
        if ref_doc == doc_id:
            own[model].append(p)
    for r in sys_rows:
        o = own[r["model"]]
        r["own_p10"] = percentile(o, 10)
        r["own_min"] = min(o)
        r["own_max"] = max(o)


def main() -> None:
    """Run analysis A end to end and write the CSV + markdown.

    :return: Nothing.
    :rtype: None
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frags, outs = cached_features(CACHE)
    rows = score(frags, outs, PAPER_CONFIG)
    tier_a = {r["fragment_id"] for r in rows if r["tier"] == "A"}
    subs = substantive_outputs(frags, outs)
    print(f"fragments={len(frags)} outputs={len(outs)} substantive={len(subs)} "
          f"tier_a={len(tier_a)}")

    verify_fast_path(frags, subs)
    print("fast-path precision verified against common.ngram_precision")

    pairs5 = compute_pairs(frags, subs, 5, clip=False)
    write_pairs_csv(OUT_DIR / "pairs_n5.csv", pairs5)
    sys_rows = system_rows(pairs5, PAPER_SYSTEMS)
    add_own_stats(sys_rows, pairs5)
    ref_rows = reference_rows(pairs5, frags, tier_a)

    sweep = {n: system_rows(compute_pairs(frags, subs, n, clip=False),
                            PAPER_SYSTEMS, full=False) for n in N_SWEEP}
    clipped_rows = system_rows(compute_pairs(frags, subs, 5, clip=True),
                               PAPER_SYSTEMS)

    md = build_markdown(frags, subs, sys_rows, ref_rows, sweep, clipped_rows)
    md += notable_section(pairs5, subs,
                          {(r["fragment_id"], r["model"]): r for r in rows})
    (OUT_DIR / "A_floor.md").write_text(md)

    with open(OUT_DIR / "per_reference_floor_n5.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ref_rows[0].keys()))
        w.writeheader()
        w.writerows(ref_rows)
    print(f"wrote {OUT_DIR/'pairs_n5.csv'} ({len(pairs5):,} rows), "
          f"{OUT_DIR/'A_floor.md'}, {OUT_DIR/'per_reference_floor_n5.csv'}")


if __name__ == "__main__":
    main()
