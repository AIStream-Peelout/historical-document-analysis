"""Analysis F — simulated in-page substitution vs cross-page hallucination.

Question: if a model silently patches a hard passage with text copied from
ELSEWHERE ON THE SAME PAGE (a closing formula, a repeated refrain, a line it
already read), does any of the paper's metrics notice?  The paper's guard is
set-membership 5-gram precision against the page reference, which by
construction cannot notice: the copied text *is* on the page.

The simulation takes the ``kraken_seg`` output on Tier A fragments (it is the
only system with real, per-line segmentation), replaces a band of its lines
with pseudo-lines cut from the page reference, re-normalises exactly as the
scorer would, and recomputes every metric before / after.  Controls: copying
from a DIFFERENT page (classic cross-page hallucination, what the metric was
designed for), different band sizes, different source regions, n = 3 / 5 / 8,
and a pure insertion that models "closing formula emitted mid-page".

Usage (from the repo root):
    .venv/bin/python -m \\
        src.datasets.evaluations.helper_eval_scripts.ngram_audit.analysis_f_substitution
"""

import argparse
import csv
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.datasets.evaluations.metrics import cer_pair, normalize_ink_hypothesis
from src.datasets.evaluations.helper_eval_scripts.score_genizah_offline import (
    aligned_prf,
    asserted_letters,
)
from src.datasets.evaluations.helper_eval_scripts.ngram_audit.common import (
    PAPER_CONFIG,
    Fragment,
    Output,
    ScoringConfig,
    cached_features,
    letters,
    loop_ratio,
    ngram_precision,
    score,
)

SCRATCH = Path("/private/tmp/claude-501/-Users-isaac-Documents-GitHub-historical-"
               "document-analysis/30e3ec54-dba4-4651-b60d-6ac04c3a0d9c/scratchpad/"
               "ngram_audit")
ORDER_BLIND = "kraken_seg"
NGRAM_SIZES = (3, 5, 8)

#: Metric key -> (pretty label, direction that means "the metric noticed a
#: problem"): -1 = a DROP is the alarm, +1 = a RISE is the alarm.
METRIC_DIRECTION: Dict[str, Tuple[str, int]] = {
    "ngram_p_unclipped_n5": ("5-gram precision (unclipped, paper)", -1),
    "ngram_p_clipped_n5": ("5-gram precision (clipped)", -1),
    "aligned_precision": ("aligned precision", -1),
    "aligned_recall": ("aligned recall", -1),
    "aligned_f1": ("aligned F1", -1),
    "cer": ("CER strict", +1),
    "cer_lenient": ("CER lenient", +1),
    "loop_ratio": ("loop ratio (asserted)", +1),
}
ALARM_DELTA = 0.02


def kraken_lines(raw: str) -> List[str]:
    """Split a Kraken raw output into its segmented lines.

    :param raw: Raw saved model output (one physical line per segmented line).
    :type raw: str
    :return: Non-empty lines, stripped of trailing whitespace.
    :rtype: list
    """
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]


def pseudo_lines(gt_ink: str, n_lines: int) -> List[str]:
    """Chunk a line-break-free reference into ``n_lines`` equal-word pseudo-lines.

    The benchmark ground truth carries no line breaks, so a like-for-like
    "reference line" is built by splitting on whitespace and dealing the words
    into ``n_lines`` consecutive chunks of as-equal-as-possible size.

    :param gt_ink: Visible-ink reference text (no line breaks).
    :type gt_ink: str
    :param n_lines: Number of chunks to produce (the hypothesis' line count).
    :type n_lines: int
    :return: ``n_lines`` chunks (possibly empty when words are scarce).
    :rtype: list
    """
    words = gt_ink.split()
    if n_lines <= 0:
        return []
    base, rem = divmod(len(words), n_lines)
    chunks, pos = [], 0
    for i in range(n_lines):
        take = base + (1 if i < rem else 0)
        chunks.append(" ".join(words[pos:pos + take]))
        pos += take
    return chunks


def source_band(chunks: Sequence[str], where: str, frac: float = 0.2) -> List[str]:
    """Take the first or last ``frac`` of a pseudo-line list, dropping blanks.

    :param chunks: Pseudo-lines from :func:`pseudo_lines`.
    :type chunks: Sequence[str]
    :param where: ``first`` or ``last``.
    :type where: str
    :param frac: Share of the page to take.
    :type frac: float
    :return: Non-empty source lines (falls back to every non-empty chunk).
    :rtype: list
    """
    non_empty = [c for c in chunks if c]
    if not non_empty:
        return []
    k = max(1, int(round(frac * len(non_empty))))
    band = non_empty[:k] if where == "first" else non_empty[-k:]
    return band or non_empty


def substitute(lines: Sequence[str], source: Sequence[str], frac: float) -> List[str]:
    """Replace the central ``frac`` band of ``lines`` with cycled ``source`` lines.

    ``frac=0.30`` replaces indices ``[0.35 * L, 0.65 * L)`` — the middle 30 %.

    :param lines: Hypothesis lines.
    :type lines: Sequence[str]
    :param source: Replacement lines, cycled when the band is longer.
    :type source: Sequence[str]
    :param frac: Share of lines to replace.
    :type frac: float
    :return: New line list of the same length.
    :rtype: list
    """
    n = len(lines)
    if not source or n == 0:
        return list(lines)
    start = int((0.5 - frac / 2) * n)
    end = int((0.5 + frac / 2) * n)
    if end <= start:
        end = min(n, start + 1)
    out = list(lines)
    for j, i in enumerate(range(start, end)):
        out[i] = source[j % len(source)]
    return out


def insert_once(lines: Sequence[str], extra: str) -> List[str]:
    """Insert one extra line at the midpoint, removing nothing.

    :param lines: Hypothesis lines.
    :type lines: Sequence[str]
    :param extra: Line to insert (e.g. the reference's closing pseudo-line).
    :type extra: str
    :return: Line list one longer than the input.
    :rtype: list
    """
    out = list(lines)
    if extra:
        out.insert(len(out) // 2, extra)
    return out


def measure(text_lines: Sequence[str], fr: Fragment,
            cfg: ScoringConfig = PAPER_CONFIG) -> Dict[str, float]:
    """Score a candidate hypothesis exactly as the paper scorer would.

    The lines are re-joined with newlines and pushed through
    ``normalize_ink_hypothesis`` first, so the measurement path is identical
    to the one the real scorer uses on a saved output.

    :param text_lines: Hypothesis lines (perturbed or not).
    :type text_lines: Sequence[str]
    :param fr: Reference fragment.
    :type fr: Fragment
    :param cfg: Scoring configuration (letter set / paper n).
    :type cfg: ScoringConfig
    :return: Metric name -> value, plus the normalised text under ``_text``.
    :rtype: dict
    """
    hyp = normalize_ink_hypothesis("\n".join(text_lines))
    hyp_l = letters(hyp, cfg.letter_set)
    gt_l = fr.gt_letters[cfg.letter_set]
    vals: Dict[str, float] = {}
    for n in NGRAM_SIZES:
        vals[f"ngram_p_unclipped_n{n}"] = ngram_precision(hyp_l, gt_l, n, clip=False)
        vals[f"ngram_p_clipped_n{n}"] = ngram_precision(hyp_l, gt_l, n, clip=True)
    p, r, f1 = aligned_prf(hyp, fr.gt_ink)
    vals["aligned_precision"], vals["aligned_recall"], vals["aligned_f1"] = p, r, f1
    cer_s, cer_l = cer_pair(hyp, fr.gt_ink)
    vals["cer"], vals["cer_lenient"] = cer_s, cer_l
    vals["loop_ratio"] = loop_ratio(asserted_letters(hyp), cfg.loop_span)
    vals["hyp_letters"] = float(len(hyp_l))
    vals["_text"] = hyp
    return vals


def tier_a_kraken(frags: List[Fragment], outs: List[Output]
                  ) -> List[Tuple[Fragment, Output]]:
    """Tier A fragments that have a substantive ``kraken_seg`` output.

    :param frags: Benchmark fragments.
    :type frags: list
    :param outs: Cached per-(fragment, system) outputs.
    :type outs: list
    :return: (fragment, kraken output) pairs in benchmark order.
    :rtype: list
    """
    rows = score(frags, outs, PAPER_CONFIG)
    tier_a = {r["fragment_id"] for r in rows if r["tier"] == "A"}
    substantive = {r["fragment_id"] for r in rows
                   if r["model"] == ORDER_BLIND and r["failure_mode"] == "substantive"}
    by_doc = {(o.doc_id, o.model): o for o in outs}
    pairs = []
    for fr in frags:
        if fr.doc_id not in tier_a or fr.doc_id not in substantive:
            continue
        out = by_doc.get((fr.doc_id, ORDER_BLIND))
        if out is not None and kraken_lines(out.raw):
            pairs.append((fr, out))
    return pairs


def run_setting(pairs: List[Tuple[Fragment, Output]], mode: str, frac: float,
                where: str = "last", cross: bool = False
                ) -> List[Dict[str, object]]:
    """Run one perturbation setting over every fragment.

    :param pairs: (fragment, kraken output) pairs from :func:`tier_a_kraken`.
    :type pairs: list
    :param mode: ``substitute`` (replace a band) or ``insert`` (append one
        closing pseudo-line mid-page, removing nothing).
    :type mode: str
    :param frac: Share of lines replaced (ignored for ``insert``).
    :type frac: float
    :param where: Source region of the donor page, ``first`` or ``last``.
    :type where: str
    :param cross: Draw the donor lines from a DIFFERENT Tier A page.
    :type cross: bool
    :return: One record per fragment with before / after / delta values.
    :rtype: list
    """
    recs: List[Dict[str, object]] = []
    for idx, (fr, out) in enumerate(pairs):
        lines = kraken_lines(out.raw)
        donor_fr, donor_out = pairs[(idx + 1) % len(pairs)] if cross else (fr, out)
        donor_lines = pseudo_lines(donor_fr.gt_ink, len(kraken_lines(donor_out.raw)))
        source = source_band(donor_lines, where)
        if mode == "insert":
            after_lines = insert_once(lines, source[-1] if source else "")
        else:
            after_lines = substitute(lines, source, frac)
        before = measure(lines, fr)
        after = measure(after_lines, fr)
        rec: Dict[str, object] = {
            "fragment_id": fr.doc_id,
            "n_lines": len(lines),
            "n_changed": sum(1 for a, b in zip(lines, after_lines) if a != b)
            if mode != "insert" else 1,
            "donor_id": donor_fr.doc_id,
            "_before_text": before.pop("_text"),
            "_after_text": after.pop("_text"),
        }
        for key in before:
            rec[f"{key}_before"] = before[key]
            rec[f"{key}_after"] = after[key]
            rec[f"{key}_delta"] = after[key] - before[key]
        recs.append(rec)
    return recs


def aggregate(recs: List[Dict[str, object]], keys: Optional[Sequence[str]] = None
              ) -> Dict[str, Dict[str, float]]:
    """Mean / median delta and alarm counts per metric.

    :param recs: Records from :func:`run_setting`.
    :type recs: list
    :param keys: Metric keys to aggregate (default: :data:`METRIC_DIRECTION`).
    :type keys: Sequence[str] or None
    :return: metric key -> {mean_delta, median_delta, n_alarm, n_opposite}.
    :rtype: dict
    """
    names = list(METRIC_DIRECTION) if keys is None else list(keys)
    agg: Dict[str, Dict[str, float]] = {}
    for key in names:
        deltas = [float(r[f"{key}_delta"]) for r in recs]
        direction = METRIC_DIRECTION.get(key, ("", -1))[1]
        agg[key] = {
            "mean_before": statistics.fmean(float(r[f"{key}_before"]) for r in recs),
            "mean_after": statistics.fmean(float(r[f"{key}_after"]) for r in recs),
            "mean_delta": statistics.fmean(deltas),
            "median_delta": statistics.median(deltas),
            "n_alarm": sum(1 for d in deltas if d * direction > ALARM_DELTA),
            "n_opposite": sum(1 for d in deltas if d * direction < -ALARM_DELTA),
        }
    return agg


def md_table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a GitHub markdown table.

    :param header: Column names.
    :type header: Sequence[str]
    :param rows: Cell values (already formatted or numeric).
    :type rows: Sequence[Sequence[object]]
    :return: Markdown table text.
    :rtype: str
    """
    def cell(v: object) -> str:
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def agg_table(agg: Dict[str, Dict[str, float]], n_frag: int) -> str:
    """Format one setting's aggregate block.

    :param agg: Output of :func:`aggregate`.
    :type agg: dict
    :param n_frag: Fragment count (for the alarm-count denominator).
    :type n_frag: int
    :return: Markdown table.
    :rtype: str
    """
    rows = []
    for key, stats in agg.items():
        label, direction = METRIC_DIRECTION.get(key, (key, -1))
        rows.append([label, stats["mean_before"], stats["mean_after"],
                     stats["mean_delta"], stats["median_delta"],
                     f"{stats['n_alarm']}/{n_frag}",
                     f"{stats['n_opposite']}/{n_frag}"])
    return md_table(["metric", "mean before", "mean after", "mean Δ", "median Δ",
                     f"n alarmed (Δ>{ALARM_DELTA} wrong way)", "n moved the other way"],
                    rows)


def ngram_compare(settings: Dict[str, List[Dict[str, object]]]) -> str:
    """Three-way n-gram comparison (baseline / in-page / cross-page) at n=3,5,8.

    :param settings: Setting label -> records; must contain ``in-page`` and
        ``cross-page`` keys.
    :type settings: dict
    :return: Markdown table.
    :rtype: str
    """
    rows = []
    for n in NGRAM_SIZES:
        for clip in (False, True):
            key = f"ngram_p_{'clipped' if clip else 'unclipped'}_n{n}"
            row: List[object] = [n, "clipped" if clip else "unclipped"]
            base = statistics.fmean(float(r[f"{key}_before"])
                                    for r in settings["in-page"])
            row.append(base)
            for label in ("in-page", "cross-page"):
                recs = settings[label]
                after = statistics.fmean(float(r[f"{key}_after"]) for r in recs)
                delta = statistics.fmean(float(r[f"{key}_delta"]) for r in recs)
                row += [after, delta]
            rows.append(row)
    return md_table(["n", "counting", "baseline mean", "in-page mean", "in-page Δ",
                     "cross-page mean", "cross-page Δ"], rows)


def gate_table(settings: Dict[str, List[Dict[str, object]]]) -> str:
    """How often the paper's own gates fire after each perturbation.

    The paper calls an output ``hallucinated`` below ``halluc_cutoff`` and
    denies it Tier A evidence below ``tier_cutoff``, both on the unclipped
    n=5 precision.  This asks whether a perturbed output ever trips them.

    :param settings: Setting label -> records from :func:`run_setting`.
    :type settings: dict
    :return: Markdown table.
    :rtype: str
    """
    key = "ngram_p_unclipped_n5"
    rows = []
    for label, recs in settings.items():
        n = len(recs)
        rows.append([
            label,
            f"{sum(1 for r in recs if float(r[f'{key}_before']) < PAPER_CONFIG.halluc_cutoff)}/{n}",
            f"{sum(1 for r in recs if float(r[f'{key}_after']) < PAPER_CONFIG.halluc_cutoff)}/{n}",
            f"{sum(1 for r in recs if float(r[f'{key}_before']) < PAPER_CONFIG.tier_cutoff)}/{n}",
            f"{sum(1 for r in recs if float(r[f'{key}_after']) < PAPER_CONFIG.tier_cutoff)}/{n}",
        ])
    return md_table(["setting",
                     f"hallucinated before (<{PAPER_CONFIG.halluc_cutoff:.2f})",
                     "hallucinated after",
                     f"below Tier-A cutoff before (<{PAPER_CONFIG.tier_cutoff:.2f})",
                     "below Tier-A cutoff after"], rows)


def write_examples(recs: List[Dict[str, object]], out_dir: Path, k: int = 5) -> List[str]:
    """Dump before/after text for the ``k`` fragments with the largest CER rise.

    :param recs: Records from the main setting.
    :type recs: list
    :param out_dir: ``examples`` directory (created if absent).
    :type out_dir: Path
    :param k: Number of examples.
    :type k: int
    :return: Fragment ids written.
    :rtype: list
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = sorted(recs, key=lambda r: -float(r["cer_delta"]))[:k]
    ids = []
    for r in chosen:
        fid = str(r["fragment_id"]).replace("/", "_")
        body = [
            f"fragment: {r['fragment_id']}",
            f"kraken lines: {r['n_lines']}   lines replaced: {r['n_changed']}",
            "metrics  before -> after",
            f"  5-gram precision (unclipped): {r['ngram_p_unclipped_n5_before']:.3f}"
            f" -> {r['ngram_p_unclipped_n5_after']:.3f}",
            f"  5-gram precision (clipped):   {r['ngram_p_clipped_n5_before']:.3f}"
            f" -> {r['ngram_p_clipped_n5_after']:.3f}",
            f"  aligned F1:                   {r['aligned_f1_before']:.3f}"
            f" -> {r['aligned_f1_after']:.3f}",
            f"  CER strict:                   {r['cer_before']:.3f}"
            f" -> {r['cer_after']:.3f}",
            f"  loop ratio:                   {r['loop_ratio_before']:.3f}"
            f" -> {r['loop_ratio_after']:.3f}",
            "", "=== BEFORE (normalised) ===", str(r["_before_text"]),
            "", "=== AFTER (normalised) ===", str(r["_after_text"]), "",
        ]
        (out_dir / f"{fid}.txt").write_text("\n".join(body))
        ids.append(str(r["fragment_id"]))
    return ids


def write_csv(recs: List[Dict[str, object]], path: Path) -> None:
    """Write the per-fragment table of the main setting.

    :param recs: Records from :func:`run_setting`.
    :type recs: list
    :param path: Destination CSV path.
    :type path: Path
    :return: None
    :rtype: None
    """
    cols = ["fragment_id", "donor_id", "n_lines", "n_changed"]
    for key in METRIC_DIRECTION:
        cols += [f"{key}_before", f"{key}_after", f"{key}_delta"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in recs:
            writer.writerow({c: (round(r[c], 4) if isinstance(r.get(c), float) else r.get(c))
                             for c in cols})


def per_fragment_table(recs: List[Dict[str, object]]) -> str:
    """Per-fragment before/after table for the main setting.

    :param recs: Records from :func:`run_setting`.
    :type recs: list
    :return: Markdown table.
    :rtype: str
    """
    rows = []
    for r in sorted(recs, key=lambda x: str(x["fragment_id"])):
        rows.append([
            r["fragment_id"], r["n_lines"], r["n_changed"],
            r["ngram_p_unclipped_n5_before"], r["ngram_p_unclipped_n5_after"],
            r["ngram_p_clipped_n5_before"], r["ngram_p_clipped_n5_after"],
            r["aligned_f1_before"], r["aligned_f1_after"],
            r["cer_before"], r["cer_after"],
            r["loop_ratio_before"], r["loop_ratio_after"],
        ])
    return md_table(["fragment", "lines", "subst.", "5g uncl. before", "5g uncl. after",
                     "5g clip. before", "5g clip. after", "F1 before", "F1 after",
                     "CER before", "CER after", "loop before", "loop after"], rows)


def build_report(pairs: List[Tuple[Fragment, Output]], out_dir: Path) -> Path:
    """Run every setting, write the CSV / examples / markdown report.

    :param pairs: (fragment, kraken output) pairs.
    :type pairs: list
    :param out_dir: ``$SCRATCH/F``.
    :type out_dir: Path
    :return: Path of the markdown report.
    :rtype: Path
    """
    n = len(pairs)
    main = run_setting(pairs, "substitute", 0.30, "last")
    settings = {
        "in-page": main,
        "in-page, donor = FIRST 20 %": run_setting(pairs, "substitute", 0.30, "first"),
        "in-page, 15 % of lines": run_setting(pairs, "substitute", 0.15, "last"),
        "in-page, 50 % of lines": run_setting(pairs, "substitute", 0.50, "last"),
        "cross-page": run_setting(pairs, "substitute", 0.30, "last", cross=True),
        "insertion (closing formula mid-page)": run_setting(pairs, "insert", 0.0, "last"),
    }
    write_csv(main, out_dir / "per_fragment.csv")
    example_ids = write_examples(main, out_dir / "examples")

    parts = [
        "# Analysis F — simulated in-page substitution vs cross-page hallucination",
        "",
        f"Scoring: `PAPER_CONFIG` ({PAPER_CONFIG.label()}).  System: `{ORDER_BLIND}` "
        f"(the only system with real per-line segmentation).  Fragments: {n} Tier A "
        "fragments whose kraken_seg output is substantive and non-empty.",
        "",
        "Hypothesis lines = `Output.raw.splitlines()` with blanks dropped.  Reference "
        "pseudo-lines = `Fragment.gt_ink` split on whitespace and dealt into the same "
        "number of equal-word chunks.  After perturbation the lines are re-joined with "
        "newlines and pushed through `normalize_ink_hypothesis`, i.e. the measurement "
        "path is the scorer's.  \"Alarm\" = the metric moved more than "
        f"{ALARM_DELTA} in the direction that signals a problem (precision / F1 down, "
        "CER / loop ratio up).",
        "",
        "## 1. Main setting — middle 30 % of lines replaced by the reference's LAST 20 %",
        "",
        agg_table(aggregate(main), n),
        "",
        "## 2. Variants (aggregate only)",
        "",
    ]
    for label in ("in-page, donor = FIRST 20 %", "in-page, 15 % of lines",
                  "in-page, 50 % of lines", "cross-page"):
        note = (" — donor pseudo-lines come from a DIFFERENT Tier A page "
                "(classic cross-page hallucination; this is what the n-gram guard was "
                "designed to catch)" if label == "cross-page" else "")
        parts += [f"### {label}{note}", "", agg_table(aggregate(settings[label]), n), ""]
    parts += [
        "## 3. n-gram precision at n = 3 / 5 / 8 — baseline vs in-page vs cross-page",
        "",
        "Both perturbations replace the same 30 % band of lines; only the donor page "
        "differs.",
        "",
        ngram_compare(settings),
        "",
        "## 4. Insertion simulation — the reference's closing pseudo-line emitted "
        "once more mid-page (nothing removed)",
        "",
        agg_table(aggregate(settings["insertion (closing formula mid-page)"]), n),
        "",
        "## 5. Do the paper's own gates ever fire?",
        "",
        "Counts of fragments whose unclipped n=5 precision falls below the "
        "hallucination cutoff / below the Tier-A evidence cutoff.",
        "",
        gate_table(settings),
        "",
        "## 6. Per-fragment table (main setting)",
        "",
        per_fragment_table(main),
        "",
        "## Files",
        "",
        "- `per_fragment.csv` — every metric, before / after / delta, main setting.",
        "- `examples/` — before/after normalised text for the 5 fragments with the "
        "largest CER rise: " + ", ".join(example_ids) + ".",
        "",
    ]
    report = out_dir / "F_substitution.md"
    report.write_text("\n".join(parts))
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point.

    :param argv: Command-line arguments (default ``sys.argv[1:]``).
    :type argv: Sequence[str] or None
    :return: Process exit status.
    :rtype: int
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=SCRATCH / "F")
    ap.add_argument("--cache", type=Path,
                    default=SCRATCH / "features_paper_systems.pkl")
    args = ap.parse_args(argv)

    frags, outs = cached_features(args.cache)
    pairs = tier_a_kraken(frags, outs)
    print(f"Tier A fragments with substantive kraken_seg: {len(pairs)}")
    report = build_report(pairs, args.out_dir)
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
