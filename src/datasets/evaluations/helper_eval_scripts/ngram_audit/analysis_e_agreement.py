"""Analysis E: does the 5-gram precision agree with the other signals?

Four parts:

1. Correlation of ``ngram_precision`` with ``aligned_precision``,
   ``aligned_f1`` and ``cer_lenient`` (Spearman + Pearson), per system and
   pooled, for all outputs and for Tier A outputs.
2. Disagreement lists in both directions, with the (hyp, gt) pairs dumped for
   manual reading and shorter-window n-gram precision for the direction where
   near-miss character confusions are the suspected cause.
3. Conditioning on the Gemini Flash judge labels: distributions, a 2x2 with
   Cohen's kappa, rank AUC for three competing predictors, and the
   canonical-completion outputs.
4. A stratified ~120-output sample for the scholar validation workstream.

Everything is scored under :data:`common.PAPER_CONFIG`; no file in the paper's
reproduction path is touched.
"""

import collections
import csv
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO))

from src.datasets.evaluations.helper_eval_scripts.ngram_audit.common import (  # noqa: E402
    PAPER_CONFIG,
    PAPER_SYSTEMS,
    Fragment,
    Output,
    cached_features,
    load_judge,
    ngram_precision,
    score,
)

SCRATCH = Path("/private/tmp/claude-501/-Users-isaac-Documents-GitHub-historical-document-analysis"
               "/30e3ec54-dba4-4651-b60d-6ac04c3a0d9c/scratchpad/ngram_audit")
OUT_DIR = SCRATCH / "E"
VERIFIED_JSON = (_REPO / "src/datasets/raw_data/cairo_genizah/evaluations/genizah_test_v1"
                 / "genizah_test_v1_verified.json")

SAMPLE_SEED = 20260916
BINS: List[Tuple[str, float, float]] = [
    ("[0,0.05)", 0.0, 0.05),
    ("[0.05,0.10)", 0.05, 0.10),
    ("[0.10,0.20)", 0.10, 0.20),
    ("[0.20,0.35)", 0.20, 0.35),
    ("[0.35,0.60)", 0.35, 0.60),
    ("[0.60,1]", 0.60, 1.0001),
]
BIN_TARGETS = {"[0,0.05)": 10, "[0.05,0.10)": 30, "[0.10,0.20)": 30,
               "[0.20,0.35)": 30, "[0.35,0.60)": 10, "[0.60,1]": 10}
# Tokens that mark a "hallucination" example as an apparatus-mark complaint
# rather than an invented-word complaint.
APPARATUS_TOKENS = ["rafeh", "rafe", "[?]", "nikud", "niqqud", "vowel", "mark",
                    "diacrit", "dagesh", "cantillation", "punctuation",
                    "ֿ", "ְ"]


# --------------------------------------------------------------------------
# small stats helpers
# --------------------------------------------------------------------------
def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (scipy, tie-corrected).

    :param x: First variable.
    :type x: Sequence[float]
    :param y: Second variable.
    :type y: Sequence[float]
    :return: rho, or NaN when undefined (n < 3 or a constant vector).
    :rtype: float
    """
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan")
    return float(scipy_stats.spearmanr(x, y).statistic)


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson product-moment correlation.

    :param x: First variable.
    :type x: Sequence[float]
    :param y: Second variable.
    :type y: Sequence[float]
    :return: r, or NaN when undefined.
    :rtype: float
    """
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan")
    return float(np.corrcoef(np.asarray(x, dtype=float), np.asarray(y, dtype=float))[0, 1])


def rank_auc(scores: Sequence[float], labels: Sequence[int]) -> Tuple[float, int, int]:
    """Rank-based AUC (Mann-Whitney U / (n_pos * n_neg)) with tie averaging.

    :param scores: Predictor values (higher = more positive).
    :type scores: Sequence[float]
    :param labels: 1 for positive, 0 for negative.
    :type labels: Sequence[int]
    :return: (AUC, n_pos, n_neg); AUC is NaN when a class is empty.
    :rtype: tuple
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan"), n_pos, n_neg
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc), n_pos, n_neg


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> Tuple[float, float]:
    """Cohen's kappa and raw agreement for two binary label vectors.

    :param a: First rater's binary labels.
    :type a: Sequence[int]
    :param b: Second rater's binary labels.
    :type b: Sequence[int]
    :return: (kappa, observed agreement); kappa is NaN when chance agreement
        is 1 (both raters constant and identical).
    :rtype: tuple
    """
    n = len(a)
    if n == 0:
        return float("nan"), float("nan")
    a_arr = np.asarray(a, dtype=int)
    b_arr = np.asarray(b, dtype=int)
    po = float((a_arr == b_arr).mean())
    pa1, pb1 = a_arr.mean(), b_arr.mean()
    pe = float(pa1 * pb1 + (1 - pa1) * (1 - pb1))
    if math.isclose(pe, 1.0):
        return float("nan"), po
    return float((po - pe) / (1 - pe)), po


def quartiles(values: Sequence[float]) -> Tuple[float, float, float]:
    """p25 / median / p75 of a value list.

    :param values: Numbers.
    :type values: Sequence[float]
    :return: (p25, median, p75); NaNs when empty.
    :rtype: tuple
    """
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    return (float(np.percentile(arr, 25)), float(np.median(arr)),
            float(np.percentile(arr, 75)))


def fmt(x: Optional[float], dp: int = 3) -> str:
    """Round for a markdown cell, rendering NaN/None as an em dash.

    :param x: Value.
    :type x: float or None
    :param dp: Decimal places.
    :type dp: int
    :return: Cell text.
    :rtype: str
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{dp}f}"


def md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a GitHub markdown table.

    :param header: Column names.
    :type header: Sequence[str]
    :param rows: Row cells (already stringified).
    :type rows: Sequence[Sequence[str]]
    :return: Table text ending in a newline.
    :rtype: str
    """
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def write_csv(path: Path, rows: List[dict], fields: Sequence[str]) -> None:
    """Write dict rows to CSV.

    :param path: Destination file.
    :type path: Path
    :param rows: Rows.
    :type rows: list
    :param fields: Column order.
    :type fields: Sequence[str]
    :return: None
    :rtype: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


# --------------------------------------------------------------------------
# data assembly
# --------------------------------------------------------------------------
def build_rows(frags: List[Fragment], outs: List[Output]) -> List[dict]:
    """Score under PAPER_CONFIG and attach judge labels and text handles.

    :param frags: Benchmark fragments.
    :type frags: list
    :param outs: Per-(fragment, system) features.
    :type outs: list
    :return: Scored rows restricted to :data:`PAPER_SYSTEMS`.
    :rtype: list
    """
    rows = [r for r in score(frags, outs, PAPER_CONFIG) if r["model"] in PAPER_SYSTEMS]
    judge = load_judge()
    out_by_key = {(o.doc_id, o.model): o for o in outs}
    frag_by_id = {f.doc_id: f for f in frags}
    for r in rows:
        key = (r["fragment_id"], r["model"])
        o = out_by_key[key]
        fr = frag_by_id[r["fragment_id"]]
        r["hyp"] = o.hyp
        r["gt_ink"] = fr.gt_ink
        r["hyp_letters_str"] = o.hyp_letters[PAPER_CONFIG.letter_set]
        r["gt_letters_str"] = fr.gt_letters[PAPER_CONFIG.letter_set]
        r["len_ratio"] = (r["hyp_letters"] / r["gt_letters"]) if r["gt_letters"] else float("nan")
        rec = judge.get(key)
        r["judged"] = rec is not None
        errs = (rec or {}).get("errors") or []
        r["judge_errors"] = errs
        r["judged_halluc_severe"] = any(
            e.get("type") == "hallucination" and e.get("severity") == "severe" for e in errs)
        r["judged_halluc_any"] = any(e.get("type") == "hallucination" for e in errs)
        r["judged_canonical"] = any(e.get("type") == "canonical_completion" for e in errs)
        r["overall_quality"] = (rec or {}).get("overall_quality")
    return rows


def load_image_urls() -> Dict[str, str]:
    """Map doc_id -> image_url from the verified benchmark JSON.

    :return: Mapping (missing docs simply absent).
    :rtype: dict
    """
    data = json.load(open(VERIFIED_JSON))
    return {d["doc_id"]: d.get("image_url", "") for d in data["docs"]}


# --------------------------------------------------------------------------
# Part 1
# --------------------------------------------------------------------------
def correlation_rows(rows: List[dict]) -> List[dict]:
    """Per-system and pooled correlations of the n-gram metric with alignment.

    :param rows: Scored rows.
    :type rows: list
    :return: One record per (subset, system) with six coefficients.
    :rtype: list
    """
    out: List[dict] = []
    subsets = [("all", rows), ("tier_a", [r for r in rows if r["tier"] == "A"])]
    for subset_name, subset in subsets:
        by_model = collections.defaultdict(list)
        for r in subset:
            by_model[r["model"]].append(r)
        groups = [(m, by_model.get(m, [])) for m in PAPER_SYSTEMS]
        groups.append(("POOLED", subset))
        for model, rs in groups:
            ng = [r["ngram_precision"] for r in rs]
            rec = dict(subset=subset_name, model=model, n=len(rs))
            for label, key in (("aligned_p", "aligned_precision"),
                               ("aligned_f1", "aligned_f1"),
                               ("cer_lenient", "cer_lenient")):
                other = [r[key] for r in rs]
                rec[f"spearman_{label}"] = spearman(ng, other)
                rec[f"pearson_{label}"] = pearson(ng, other)
            out.append(rec)
    return out


# --------------------------------------------------------------------------
# Part 2
# --------------------------------------------------------------------------
def disagreements(rows: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Split out the two disagreement directions.

    :param rows: Scored rows.
    :type rows: list
    :return: (list A rows, list B rows) sorted for reporting.
    :rtype: tuple
    """
    list_a = [r for r in rows
              if r["ngram_precision"] >= 0.25 and r["aligned_precision"] <= 0.5]
    list_b = [r for r in rows
              if r["ngram_precision"] < 0.10 and r["aligned_precision"] >= 0.5]
    list_a.sort(key=lambda r: (-r["ngram_precision"], r["aligned_precision"]))
    list_b.sort(key=lambda r: (-r["aligned_precision"], r["ngram_precision"]))
    return list_a, list_b


def dump_pairs(rows: List[dict], directory: Path, limit: Optional[int] = None) -> int:
    """Write hypothesis / ground-truth-ink pairs to text files.

    :param rows: Rows to dump (already ordered).
    :type rows: list
    :param directory: Destination folder (created).
    :type directory: Path
    :param limit: Write at most this many rows (``None`` = all).
    :type limit: int or None
    :return: Number of files written.
    :rtype: int
    """
    directory.mkdir(parents=True, exist_ok=True)
    selected = rows if limit is None else rows[:limit]
    for r in selected:
        path = directory / f"{r['fragment_id']}_{r['model']}.txt"
        body = (
            f"doc_id: {r['fragment_id']}\nmodel: {r['model']}\n"
            f"ngram_precision(n=5): {r['ngram_precision']:.4f}\n"
            f"aligned_precision: {r['aligned_precision']:.4f}\n"
            f"aligned_recall: {r['aligned_recall']:.4f}\n"
            f"cer_lenient: {r['cer_lenient']:.4f}\n"
            f"len_ratio(hyp/gt letters): {r['len_ratio']:.4f}\n"
            f"tier: {r['tier']}  script_bucket: {r['script_bucket']}  "
            f"failure_mode: {r['failure_mode']}\n"
            f"judged_halluc_severe: {r['judged_halluc_severe']}  "
            f"judged_canonical: {r['judged_canonical']}  "
            f"overall_quality: {r['overall_quality']}\n"
            "\n===== HYPOTHESIS (normalised ink) =====\n"
            f"{r['hyp']}\n"
            "\n===== GROUND TRUTH (visible ink) =====\n"
            f"{r['gt_ink']}\n"
        )
        path.write_text(body)
    return len(selected)


def short_window_rescue(rows: List[dict]) -> None:
    """Add n=3 and n=4 n-gram precision in place (list B diagnosis).

    :param rows: Rows carrying ``hyp_letters_str`` / ``gt_letters_str``.
    :type rows: list
    :return: None
    :rtype: None
    """
    for r in rows:
        for n in (3, 4):
            r[f"ngram_precision_n{n}"] = ngram_precision(
                r["hyp_letters_str"], r["gt_letters_str"], n=n, clip=PAPER_CONFIG.clip)


# --------------------------------------------------------------------------
# Part 3
# --------------------------------------------------------------------------
def conditioned_distribution(rows: List[dict], flag_key: str) -> List[dict]:
    """n-gram precision distribution conditioned on a boolean judge flag.

    :param rows: Judged rows only.
    :type rows: list
    :param flag_key: Row key holding the boolean.
    :type flag_key: str
    :return: Per-system and pooled records for flag True / False.
    :rtype: list
    """
    out: List[dict] = []
    by_model = collections.defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    groups = [(m, by_model.get(m, [])) for m in PAPER_SYSTEMS] + [("POOLED", rows)]
    for model, rs in groups:
        rec = dict(model=model, n_total=len(rs))
        for flag in (True, False):
            vals = [r["ngram_precision"] for r in rs if bool(r[flag_key]) is flag]
            p25, med, p75 = quartiles(vals)
            suffix = "true" if flag else "false"
            rec[f"n_{suffix}"] = len(vals)
            rec[f"median_{suffix}"] = med
            rec[f"p25_{suffix}"] = p25
            rec[f"p75_{suffix}"] = p75
        out.append(rec)
    return out


def quality_distribution(rows: List[dict]) -> List[dict]:
    """n-gram precision distribution by ``overall_quality``, per system+pooled.

    :param rows: Judged rows only.
    :type rows: list
    :return: Records keyed by (model, quality).
    :rtype: list
    """
    order = ["unusable", "poor", "fair", "good"]
    out: List[dict] = []
    by_model = collections.defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    groups = [(m, by_model.get(m, [])) for m in PAPER_SYSTEMS] + [("POOLED", rows)]
    for model, rs in groups:
        for q in order:
            vals = [r["ngram_precision"] for r in rs if r["overall_quality"] == q]
            p25, med, p75 = quartiles(vals)
            out.append(dict(model=model, quality=q, n=len(vals),
                            median=med, p25=p25, p75=p75))
    return out


def two_by_two(rows: List[dict]) -> List[dict]:
    """2x2 of metric class vs judged severe hallucination, per system+pooled.

    Metric "positive" is ``failure_mode == 'hallucinated'``; abstained and
    loop-collapse rows are excluded because the metric never reaches the
    n-gram cutoff for them.

    :param rows: Judged rows only.
    :type rows: list
    :return: Contingency records with agreement and Cohen's kappa.
    :rtype: list
    """
    usable = [r for r in rows if r["failure_mode"] in ("hallucinated", "substantive")]
    by_model = collections.defaultdict(list)
    for r in usable:
        by_model[r["model"]].append(r)
    groups = [(m, by_model.get(m, [])) for m in PAPER_SYSTEMS] + [("POOLED", usable)]
    out: List[dict] = []
    for model, rs in groups:
        metric = [1 if r["failure_mode"] == "hallucinated" else 0 for r in rs]
        judged = [1 if r["judged_halluc_severe"] else 0 for r in rs]
        kappa, agree = cohen_kappa(metric, judged)
        out.append(dict(
            model=model, n=len(rs),
            both=sum(1 for m, j in zip(metric, judged) if m and j),
            metric_only=sum(1 for m, j in zip(metric, judged) if m and not j),
            judge_only=sum(1 for m, j in zip(metric, judged) if j and not m),
            neither=sum(1 for m, j in zip(metric, judged) if not m and not j),
            agreement=agree, kappa=kappa,
        ))
    return out


def auc_table(rows: List[dict]) -> List[dict]:
    """Rank AUC for three predictors of judged severe hallucination.

    ``cer_lenient`` is negated so that, like the other two, higher means
    "better transcription" and the AUC is comparable in direction.

    :param rows: Judged rows only.
    :type rows: list
    :return: Per-system and pooled AUC records.
    :rtype: list
    """
    by_model = collections.defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    groups = [(m, by_model.get(m, [])) for m in PAPER_SYSTEMS] + [("POOLED", rows)]
    out: List[dict] = []
    for model, rs in groups:
        labels = [1 if r["judged_halluc_severe"] else 0 for r in rs]
        rec = dict(model=model, n=len(rs))
        for name, getter in (("ngram_precision", lambda r: -r["ngram_precision"]),
                             ("aligned_precision", lambda r: -r["aligned_precision"]),
                             ("cer_lenient", lambda r: r["cer_lenient"])):
            auc, n_pos, n_neg = rank_auc([getter(r) for r in rs], labels)
            rec[f"auc_{name}"] = auc
            rec["n_pos"] = n_pos
            rec["n_neg"] = n_neg
        out.append(rec)
    return out


def apparatus_breakdown(rows: List[dict]) -> Dict[str, int]:
    """Count severe-hallucination examples that complain about marks.

    Examples whose text carries almost no Latin prose (a bare
    ``<gt> -> <hyp>`` pair in Hebrew) are counted separately, because the
    token list is English and cannot adjudicate them.

    :param rows: Judged rows only.
    :type rows: list
    :return: Counts of mark-flavoured vs other severe hallucination examples.
    :rtype: dict
    """
    counts: collections.Counter = collections.Counter()
    for r in rows:
        for e in r["judge_errors"]:
            if e.get("type") != "hallucination" or e.get("severity") != "severe":
                continue
            example = (e.get("example") or "").lower()
            if not example:
                counts["no_example"] += 1
            elif any(tok in example for tok in APPARATUS_TOKENS):
                counts["apparatus_marks"] += 1
            elif sum(c.isascii() and c.isalpha() for c in example) < 4:
                # A bare Hebrew "<gt> -> <hyp>" substitution pair: invented text,
                # counted apart because the token list cannot inspect it.
                counts["invented_text_hebrew_pair"] += 1
            else:
                counts["invented_text_described"] += 1
    return dict(counts)


def length_bands(rows: List[dict]) -> List[dict]:
    """Group list-(b) rows by hypothesis/reference length ratio.

    Separates "aligned precision is high because the output is a stub" from
    genuine full-length near misses.

    :param rows: List (b) rows carrying ``len_ratio`` and shorter-window
        precisions.
    :type rows: list
    :return: One record per length band.
    :rtype: list
    """
    bands = [("len ratio < 0.15 (stub)", 0.0, 0.15),
             ("0.15-0.50", 0.15, 0.50),
             ("0.50-0.80", 0.50, 0.80),
             (">= 0.80 (full length)", 0.80, float("inf"))]
    out: List[dict] = []
    for label, lo, hi in bands:
        rs = [r for r in rows if lo <= r["len_ratio"] < hi]
        out.append(dict(
            band=label, n=len(rs),
            median_aligned_r=(statistics.median(r["aligned_recall"] for r in rs)
                              if rs else float("nan")),
            median_n5=(statistics.median(r["ngram_precision"] for r in rs)
                       if rs else float("nan")),
            median_n4=(statistics.median(r["ngram_precision_n4"] for r in rs)
                       if rs else float("nan")),
            median_n3=(statistics.median(r["ngram_precision_n3"] for r in rs)
                       if rs else float("nan")),
            rescued_n3=sum(1 for r in rs if r["ngram_precision_n3"] >= 0.10),
            abstained=sum(1 for r in rs if r["failure_mode"] == "abstained"),
        ))
    return out


# --------------------------------------------------------------------------
# Part 4
# --------------------------------------------------------------------------
def bin_of(value: float) -> str:
    """Bin label for an n-gram precision value.

    :param value: Precision in [0, 1].
    :type value: float
    :return: Bin label.
    :rtype: str
    """
    for label, lo, hi in BINS:
        if lo <= value < hi:
            return label
    return BINS[-1][0]


def stratified_sample(rows: List[dict], rng: random.Random) -> List[dict]:
    """Draw the scholar-validation sample with per-system quotas.

    Within each bin, no system may exceed 25 % of the bin's target while
    candidates from other systems remain; the remainder is then filled
    without the cap so the target size is met when the bin is large enough.

    :param rows: Scored rows for the paper systems.
    :type rows: list
    :param rng: Seeded RNG.
    :type rng: random.Random
    :return: Sampled rows with ``sample_bin`` / ``sample_reason`` set.
    :rtype: list
    """
    by_bin = collections.defaultdict(list)
    for r in rows:
        by_bin[bin_of(r["ngram_precision"])].append(r)
    chosen: List[dict] = []
    chosen_keys = set()
    for label, _lo, _hi in BINS:
        target = BIN_TARGETS[label]
        pool = list(by_bin.get(label, []))
        rng.shuffle(pool)
        cap = max(1, math.ceil(0.25 * target))
        per_system = collections.Counter()
        picked: List[dict] = []
        for r in pool:
            if len(picked) >= target:
                break
            if per_system[r["model"]] < cap:
                picked.append(r)
                per_system[r["model"]] += 1
        if len(picked) < target:
            picked_keys = {(r["fragment_id"], r["model"]) for r in picked}
            for r in pool:
                if len(picked) >= target:
                    break
                if (r["fragment_id"], r["model"]) not in picked_keys:
                    picked.append(r)
                    picked_keys.add((r["fragment_id"], r["model"]))
        for r in picked:
            r["sample_bin"] = label
            r["sample_reason"] = "bin_quota"
            chosen.append(r)
            chosen_keys.add((r["fragment_id"], r["model"]))
    for r in rows:
        key = (r["fragment_id"], r["model"])
        if r["judged_canonical"] and key not in chosen_keys:
            r["sample_bin"] = bin_of(r["ngram_precision"])
            r["sample_reason"] = "canonical_completion"
            chosen.append(r)
            chosen_keys.add(key)
    chosen.sort(key=lambda r: (r["sample_bin"], r["model"], r["fragment_id"]))
    return chosen


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def build_report(rows: List[dict], corr: List[dict], list_a: List[dict],
                 list_b: List[dict], judged: List[dict], sample: List[dict]) -> str:
    """Assemble the self-contained markdown report.

    :param rows: All scored rows.
    :type rows: list
    :param corr: Part 1 records.
    :type corr: list
    :param list_a: Part 2 list (a).
    :type list_a: list
    :param list_b: Part 2 list (b).
    :type list_b: list
    :param judged: Rows with a judge record.
    :type judged: list
    :param sample: Part 4 sample rows.
    :type sample: list
    :return: Markdown text.
    :rtype: str
    """
    md: List[str] = []
    n_docs = len({r["fragment_id"] for r in rows})
    n_tier_a = len({r["fragment_id"] for r in rows if r["tier"] == "A"})
    md.append("# Analysis E — agreement of the 5-gram precision with other signals "
              "and with the LLM judge\n")
    md.append(f"Config: `{PAPER_CONFIG.label()}` (paper reproduction). "
              f"{len(rows)} outputs = {n_docs} verified fragments x "
              f"{len(PAPER_SYSTEMS)} paper systems (Tier A fragments: {n_tier_a}). "
              "scipy 1.15.3 was available in `.venv`, so Spearman uses "
              "`scipy.stats.spearmanr` (tie-corrected); AUC and kappa are implemented "
              "with numpy in this script.\n")

    # Part 1
    md.append("## Part 1 — correlation with the alignment metrics\n")
    md.append("`cer_lenient` correlations are expected negative (higher n-gram "
              "precision = lower CER).\n")
    for subset, title in (("all", "All outputs"), ("tier_a", "Tier A outputs only")):
        md.append(f"### {title}\n")
        header = ["system", "n", "rho(aligned P)", "r(aligned P)", "rho(aligned F1)",
                  "r(aligned F1)", "rho(CER-lenient)", "r(CER-lenient)"]
        body = []
        for rec in corr:
            if rec["subset"] != subset:
                continue
            body.append([rec["model"], rec["n"],
                         fmt(rec["spearman_aligned_p"]), fmt(rec["pearson_aligned_p"]),
                         fmt(rec["spearman_aligned_f1"]), fmt(rec["pearson_aligned_f1"]),
                         fmt(rec["spearman_cer_lenient"]), fmt(rec["pearson_cer_lenient"])])
        md.append(md_table(header, body))
    zero_share = {m: sum(1 for r in rows if r["model"] == m and r["ngram_precision"] == 0.0)
                  / max(1, sum(1 for r in rows if r["model"] == m))
                  for m in PAPER_SYSTEMS}
    worst = sorted(zero_share.items(), key=lambda kv: -kv[1])[:3]
    md.append("Tie caveat: systems whose n-gram precision is exactly 0 for most outputs "
              "have a near-degenerate rank vector, so Spearman collapses while Pearson "
              "stays high (the zeros still line up with low alignment scores). Share of "
              "outputs at exactly 0: "
              + ", ".join(f"{m} {s:.1%}" for m, s in worst) + ".\n")

    # Part 2
    md.append("## Part 2 — disagreement lists\n")
    md.append(f"**List (a)** n-gram >= 0.25 but aligned P <= 0.5: **{len(list_a)}** outputs "
              f"({len(list_a) / len(rows):.1%} of all).\n")
    md.append(f"**List (b)** n-gram < 0.10 but aligned P >= 0.5: **{len(list_b)}** outputs "
              f"({len(list_b) / len(rows):.1%} of all).\n")
    for name, lst in (("a", list_a), ("b", list_b)):
        counts = collections.Counter(r["model"] for r in lst)
        body = [[m, counts.get(m, 0),
                 f"{counts.get(m, 0) / sum(1 for r in rows if r['model'] == m):.3f}"]
                for m in PAPER_SYSTEMS]
        body.append(["TOTAL", len(lst), f"{len(lst) / len(rows):.3f}"])
        md.append(f"### List ({name}) — per-system counts\n")
        md.append(md_table(["system", "count", "share of that system's outputs"], body))
    md.append("### List (a) rows\n")
    header_a = ["doc_id", "model", "ngram", "aligned P", "aligned R", "CER-len",
                "len ratio", "tier", "script", "failure_mode"]
    md.append(md_table(header_a, [
        [r["fragment_id"], r["model"], fmt(r["ngram_precision"]),
         fmt(r["aligned_precision"]), fmt(r["aligned_recall"]), fmt(r["cer_lenient"]),
         fmt(r["len_ratio"], 2), r["tier"], r["script_bucket"], r["failure_mode"]]
        for r in list_a]))
    md.append("### List (b) rows (with shorter windows)\n")
    header_b = header_a + ["ngram n=4", "ngram n=3"]
    md.append(md_table(header_b, [
        [r["fragment_id"], r["model"], fmt(r["ngram_precision"]),
         fmt(r["aligned_precision"]), fmt(r["aligned_recall"]), fmt(r["cer_lenient"]),
         fmt(r["len_ratio"], 2), r["tier"], r["script_bucket"], r["failure_mode"],
         fmt(r["ngram_precision_n4"]), fmt(r["ngram_precision_n3"])]
        for r in list_b]))
    if list_b:
        md.append("#### List (b) split by hypothesis length\n")
        md.append("Aligned precision is length-blind: a 20-character stub that happens to "
                  "match scores ~0.9. Splitting list (b) by `len(hyp letters)/len(GT letters)` "
                  "separates those stubs from genuine full-length near misses.\n")
        md.append(md_table(
            ["band", "n", "median aligned R", "median n=5", "median n=4", "median n=3",
             "n=3 >= 0.10", "metric-abstained"],
            [[b["band"], b["n"], fmt(b["median_aligned_r"]), fmt(b["median_n5"]),
              fmt(b["median_n4"]), fmt(b["median_n3"]), b["rescued_n3"], b["abstained"]]
             for b in length_bands(list_b)]))
        md.append("Shorter-window rescue, list (b): median n=5 "
                  f"{fmt(statistics.median(r['ngram_precision'] for r in list_b))}, "
                  f"n=4 {fmt(statistics.median(r['ngram_precision_n4'] for r in list_b))}, "
                  f"n=3 {fmt(statistics.median(r['ngram_precision_n3'] for r in list_b))}; "
                  f"{sum(1 for r in list_b if r['ngram_precision_n3'] >= 0.10)}/{len(list_b)} "
                  "clear the 0.10 hallucination cutoff at n=3 and "
                  f"{sum(1 for r in list_b if r['ngram_precision_n4'] >= 0.10)}/{len(list_b)} "
                  "at n=4.\n")

    # Part 3
    md.append("## Part 3 — conditioning on the Gemini Flash judge\n")
    judged_keys = {(r["fragment_id"], r["model"]) for r in judged}
    missing = collections.Counter(r["model"] for r in rows
                                 if (r["fragment_id"], r["model"]) not in judged_keys)
    md.append(f"Judge coverage: {len(judged)}/{len(rows)} outputs "
              f"({len(judged) / len(rows):.1%}). Unjudged per system: "
              + ", ".join(f"{m} {missing.get(m, 0)}" for m in PAPER_SYSTEMS if missing.get(m))
              + ". All tables below use judged rows only.\n")
    n_sev = sum(1 for r in judged if r["judged_halluc_severe"])
    n_any = sum(1 for r in judged if r["judged_halluc_any"])
    n_can = sum(1 for r in judged if r["judged_canonical"])
    md.append(f"Judged flags: severe hallucination {n_sev} ({n_sev / len(judged):.1%}), "
              f"any-severity hallucination {n_any} ({n_any / len(judged):.1%}), "
              f"canonical_completion {n_can}.\n")

    md.append("### (i) n-gram precision conditioned on judged hallucination\n")
    for flag_key, title in (("judged_halluc_severe", "severe hallucination"),
                            ("judged_halluc_any", "hallucination, any severity")):
        md.append(f"**{title}**\n")
        recs = conditioned_distribution(judged, flag_key)
        md.append(md_table(
            ["system", "n flagged", "median", "p25", "p75", "n not flagged",
             "median", "p25", "p75"],
            [[rec["model"], rec["n_true"], fmt(rec["median_true"]), fmt(rec["p25_true"]),
              fmt(rec["p75_true"]), rec["n_false"], fmt(rec["median_false"]),
              fmt(rec["p25_false"]), fmt(rec["p75_false"])] for rec in recs]))

    md.append("### (ii) n-gram precision conditioned on overall_quality\n")
    qrecs = quality_distribution(judged)
    md.append(md_table(["system", "quality", "n", "median", "p25", "p75"],
                       [[q["model"], q["quality"], q["n"], fmt(q["median"]),
                         fmt(q["p25"]), fmt(q["p75"])] for q in qrecs if q["n"]]))

    md.append("### (iii) 2x2: metric class vs judged severe hallucination\n")
    md.append("Metric positive = `failure_mode == hallucinated`; abstained and "
              "loop-collapse rows excluded.\n")
    trecs = two_by_two(judged)
    md.append(md_table(
        ["system", "n", "both halluc", "metric only", "judge only", "neither",
         "agreement", "kappa"],
        [[t["model"], t["n"], t["both"], t["metric_only"], t["judge_only"], t["neither"],
          fmt(t["agreement"]), fmt(t["kappa"])] for t in trecs]))

    md.append("### (iv) rank AUC for predicting judged severe hallucination\n")
    md.append("Predictors oriented so that higher = more likely hallucination "
              "(n-gram precision and aligned precision negated; CER-lenient as is). "
              "AUC 0.5 = no signal.\n")
    arecs = auc_table(judged)
    md.append(md_table(["system", "n", "pos", "neg", "AUC n-gram P", "AUC aligned P",
                        "AUC CER-lenient"],
                       [[a["model"], a["n"], a["n_pos"], a["n_neg"],
                         fmt(a["auc_ngram_precision"]), fmt(a["auc_aligned_precision"]),
                         fmt(a["auc_cer_lenient"])] for a in arecs]))

    md.append("### (v) canonical_completion outputs\n")
    can = sorted((r for r in judged if r["judged_canonical"]),
                 key=lambda r: r["ngram_precision"])
    md.append(md_table(
        ["doc_id", "model", "ngram", "aligned P", "CER-len", "failure_mode",
         "substantive?", "quality", "tier"],
        [[r["fragment_id"], r["model"], fmt(r["ngram_precision"]),
          fmt(r["aligned_precision"]), fmt(r["cer_lenient"]), r["failure_mode"],
          "yes" if r["failure_mode"] == "substantive" else "no",
          r["overall_quality"], r["tier"]] for r in can]))
    if can:
        n_sub = sum(1 for r in can if r["failure_mode"] == "substantive")
        md.append(f"{n_sub}/{len(can)} canonical-completion outputs are classed "
                  "**substantive** by the metric; median n-gram precision "
                  f"{fmt(statistics.median(r['ngram_precision'] for r in can))} vs "
                  f"{fmt(statistics.median(r['ngram_precision'] for r in judged))} over all "
                  "judged outputs.\n")

    md.append("### Judge 'hallucination' label: marks vs invented text\n")
    ab = apparatus_breakdown(judged)
    total_ab = sum(ab.values())
    md.append(md_table(["example flavour", "count", "share"],
                       [[k, v, f"{v / total_ab:.3f}" if total_ab else "—"]
                        for k, v in sorted(ab.items(), key=lambda kv: -kv[1])]))
    invented = (ab.get("invented_text_hebrew_pair", 0)
                + ab.get("invented_text_described", 0))
    md.append(f"Apparatus-mark complaints: {ab.get('apparatus_marks', 0)}/{total_ab} "
              f"({ab.get('apparatus_marks', 0) / total_ab:.1%}) of severe-hallucination "
              f"examples; invented text (both flavours) {invented}/{total_ab} "
              f"({invented / total_ab:.1%}). Matched tokens: "
              + ", ".join(f"`{t}`" for t in APPARATUS_TOKENS) + ".\n")

    # Part 4
    md.append("## Part 4 — scholar validation sample\n")
    bcounts = collections.Counter(r["sample_bin"] for r in sample)
    rcounts = collections.Counter(r["sample_reason"] for r in sample)
    pop = collections.Counter(bin_of(r["ngram_precision"]) for r in rows)
    md.append(md_table(["bin", "population", "target", "sampled"],
                       [[label, pop.get(label, 0), BIN_TARGETS[label], bcounts.get(label, 0)]
                        for label, _lo, _hi in BINS]
                       + [["TOTAL", len(rows), sum(BIN_TARGETS.values()), len(sample)]]))
    md.append("Draw reasons: " + ", ".join(f"{k} {v}" for k, v in rcounts.items())
              + f"; seed {SAMPLE_SEED}. Bins can exceed their target because every judged "
              "canonical_completion output is added on top of the quota draw.\n")
    quota_only = [r for r in sample if r["sample_reason"] == "bin_quota"]
    cap_rows = []
    for label, _lo, _hi in BINS:
        in_bin = [r for r in quota_only if r["sample_bin"] == label]
        counts = collections.Counter(r["model"] for r in in_bin)
        top_model, top_n = (counts.most_common(1)[0] if counts else ("—", 0))
        cap_rows.append([label, len(in_bin), f"ceil(0.25 x {BIN_TARGETS[label]}) = "
                         f"{max(1, math.ceil(0.25 * BIN_TARGETS[label]))}",
                         top_model, top_n,
                         f"{top_n / len(in_bin):.3f}" if in_bin else "—"])
    md.append("Secondary stratification by system (quota draws only):\n")
    md.append(md_table(["bin", "quota drawn", "per-system cap", "heaviest system",
                        "its count", "its share of bin"], cap_rows))
    scounts = collections.Counter(r["model"] for r in sample)
    md.append(md_table(["system", "in sample", "share"],
                       [[m, scounts.get(m, 0), f"{scounts.get(m, 0) / len(sample):.3f}"]
                        for m in PAPER_SYSTEMS]))
    md.append("Written to `E/scholar_sample.csv` with hyp/GT text files in "
              "`E/scholar_sample_pairs/`.\n")
    return "\n".join(md)


def main() -> None:
    """Run all four parts and write every artefact under ``$SCRATCH/E``.

    :return: None
    :rtype: None
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frags, outs = cached_features(SCRATCH / "features_paper_systems.pkl")
    rows = build_rows(frags, outs)

    corr = correlation_rows(rows)
    write_csv(OUT_DIR / "correlations.csv", corr,
              ["subset", "model", "n", "spearman_aligned_p", "pearson_aligned_p",
               "spearman_aligned_f1", "pearson_aligned_f1",
               "spearman_cer_lenient", "pearson_cer_lenient"])

    list_a, list_b = disagreements(rows)
    short_window_rescue(list_b)
    dump_pairs(list_a, OUT_DIR / "pairs" / "a_ngram_high_aligned_low", limit=10)
    dump_pairs(list_b, OUT_DIR / "pairs" / "b_ngram_low_aligned_high", limit=10)
    row_fields = ["fragment_id", "model", "ngram_precision", "aligned_precision",
                  "aligned_recall", "cer_lenient", "len_ratio", "tier", "script_bucket",
                  "failure_mode", "judged_halluc_severe", "judged_canonical",
                  "overall_quality"]
    write_csv(OUT_DIR / "disagreement_a.csv", list_a, row_fields)
    write_csv(OUT_DIR / "disagreement_b.csv", list_b,
              row_fields + ["ngram_precision_n4", "ngram_precision_n3"])

    judged = [r for r in rows if r["judged"]]
    write_csv(OUT_DIR / "judge_conditioned_halluc_severe.csv",
              conditioned_distribution(judged, "judged_halluc_severe"),
              ["model", "n_total", "n_true", "median_true", "p25_true", "p75_true",
               "n_false", "median_false", "p25_false", "p75_false"])
    write_csv(OUT_DIR / "judge_quality_distribution.csv", quality_distribution(judged),
              ["model", "quality", "n", "median", "p25", "p75"])
    write_csv(OUT_DIR / "judge_2x2_kappa.csv", two_by_two(judged),
              ["model", "n", "both", "metric_only", "judge_only", "neither",
               "agreement", "kappa"])
    write_csv(OUT_DIR / "judge_auc.csv", auc_table(judged),
              ["model", "n", "n_pos", "n_neg", "auc_ngram_precision",
               "auc_aligned_precision", "auc_cer_lenient"])

    rng = random.Random(SAMPLE_SEED)
    sample = stratified_sample(rows, rng)
    urls = load_image_urls()
    sample_rows = []
    for r in sample:
        sample_rows.append(dict(
            doc_id=r["fragment_id"], model=r["model"], bin=r["sample_bin"],
            draw_reason=r["sample_reason"], ngram=round(r["ngram_precision"], 4),
            aligned_p=round(r["aligned_precision"], 4),
            cer_lenient=round(r["cer_lenient"], 4), tier=r["tier"],
            script_bucket=r["script_bucket"], failure_mode=r["failure_mode"],
            judged=r["judged"], judged_halluc_severe=r["judged_halluc_severe"],
            judged_halluc_any=r["judged_halluc_any"],
            judged_canonical=r["judged_canonical"],
            overall_quality=r["overall_quality"] or "",
            image_url=urls.get(r["fragment_id"], "")))
    write_csv(OUT_DIR / "scholar_sample.csv", sample_rows,
              ["doc_id", "model", "bin", "draw_reason", "ngram", "aligned_p",
               "cer_lenient", "tier", "script_bucket", "failure_mode", "judged",
               "judged_halluc_severe", "judged_halluc_any", "judged_canonical",
               "overall_quality", "image_url"])
    dump_pairs(sample, OUT_DIR / "scholar_sample_pairs")

    report = build_report(rows, corr, list_a, list_b, judged, sample)
    (OUT_DIR / "E_agreement.md").write_text(report)
    print(f"rows={len(rows)} judged={len(judged)} list_a={len(list_a)} "
          f"list_b={len(list_b)} sample={len(sample)}")
    print(f"wrote {OUT_DIR}/E_agreement.md")


if __name__ == "__main__":
    main()
