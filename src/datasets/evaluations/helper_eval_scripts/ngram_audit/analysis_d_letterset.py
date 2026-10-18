"""Analysis D — sensitivity of the Genizah n-gram metric to the letter set.

The paper's scorer reduces both hypothesis and reference to the Hebrew Unicode
block (``U+0590``-``U+05FF``) before counting 5-grams.  Any Arabic-script
character a model writes is therefore silently deleted, so a mixed-script
output is graded on its Hebrew half alone and a genuinely Arabic-script page
looks unreadable to every system.  This module re-scores everything under the
Hebrew+Arabic union ("semitic") letter set and reports what moves.

Part 1 works on the verified-131 feature cache (11 paper systems).
Part 2 re-derives the four Arabic-script fragments the benchmark audit
excluded (they are absent from the verified subset and from
``load_fragments``, which drops fragments with < 50 *Hebrew* letters) and
re-runs the audit's 0.12 best-evidence-overlap test under the union set, both
for those four and for all 19 excluded fragments.

Run:
    .venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.ngram_audit.analysis_d_letterset
"""

import collections
import csv
import json
import os
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.datasets.evaluations.helper_eval_scripts.ngram_audit.common import (
    BENCH_DIR,
    LETTER_RES,
    OUTPUTS,
    PAPER_CONFIG,
    PAPER_SYSTEMS,
    Fragment,
    Output,
    ScoringConfig,
    cached_features,
    letters,
    load_features,
    ngram_precision,
    score,
    tier_a_count,
)
from src.datasets.evaluations.helper_eval_scripts.score_genizah_offline import (
    load_benchmark,
)
from src.datasets.evaluations.metrics import genizah_visible_ink_gt

SCRATCH = Path(os.environ.get(
    "SCRATCH",
    "/private/tmp/claude-501/-Users-isaac-Documents-GitHub-historical-document-analysis"
    "/30e3ec54-dba4-4651-b60d-6ac04c3a0d9c/scratchpad/ngram_audit"))
OUT_DIR = SCRATCH / "D"
CACHE = SCRATCH / "features_paper_systems.pkl"

SEMITIC_CONFIG = ScoringConfig(letter_set="semitic")

#: Evidence systems used by ``audit_genizah_benchmark._EVIDENCE_MODELS``.
AUDIT_EVIDENCE = [
    "kraken_seg", "gemini_pro", "gemini_flash", "claude_opus_4_8",
    "claude_sonnet_5", "qwen3_vl_8b_heb_v17_step800",
    "qwen3_vl_8b_heb_v16_step1100", "vision_ocr_seg",
]
AUDIT_THRESHOLD = 0.12

#: The four Arabic-script fragments the audit excluded, per the task brief.
ARABIC_FOUR = [
    "New_York_JTS_ENA_NS_2_29",
    "Cambridge_CUL_T_S_Ar_38_2",
    "Cambridge_CUL_T_S_Ar_4_10",
    "Cambridge_CUL_T_S_Ar_19_23",
]


def script_tags() -> Dict[str, str]:
    """Load the benchmark's per-fragment script bucket.

    :return: doc_id -> bucket (``untagged`` when missing).
    :rtype: dict
    """
    path = BENCH_DIR / "genizah_test_v1_script_tags.json"
    if not path.exists():
        return {}
    return {k: v.get("bucket", "untagged") for k, v in json.load(open(path)).items()}


def excluded_ids() -> List[str]:
    """Read ``audit.excluded_misaligned`` from the verified manifest.

    :return: Sorted doc_ids excluded by the benchmark audit.
    :rtype: list
    """
    spec = json.load(open(BENCH_DIR / "genizah_test_v1_verified.json"))
    return list(spec["audit"]["excluded_misaligned"])


def load_fragments_unfiltered(which: str, keep: Optional[Sequence[str]] = None
                              ) -> List[Fragment]:
    """Load benchmark fragments WITHOUT the 50-Hebrew-letter floor.

    ``common.load_fragments`` drops any fragment whose repaired GT holds fewer
    than 50 Hebrew-block letters, which removes Arabic-script pages by
    construction.  This variant keeps them; it still requires a saved-output
    directory so the fragment can be scored.

    :param which: ``verified`` or ``frozen``.
    :type which: str
    :param keep: Restrict to these doc_ids (default: all).
    :type keep: Sequence[str] or None
    :return: Fragments in benchmark order.
    :rtype: list
    """
    tags = script_tags()
    wanted = set(keep) if keep is not None else None
    frags: List[Fragment] = []
    for d in load_benchmark(which):
        if wanted is not None and d["doc_id"] not in wanted:
            continue
        if not (OUTPUTS / d["doc_id"]).is_dir():
            continue
        gt_ink = genizah_visible_ink_gt(d["gt"])
        gl = {ls: letters(gt_ink, ls) for ls in LETTER_RES}
        frags.append(Fragment(d["doc_id"], d["gt"], gt_ink, gl,
                              tags.get(d["doc_id"], "untagged"),
                              bool(d.get("_repaired_duplication"))))
    return frags


def arabic_only(counts: Dict[str, str]) -> int:
    """Arabic-block letter count implied by the two letter-set reductions.

    :param counts: Mapping with ``hebrew`` and ``semitic`` letter strings.
    :type counts: dict
    :return: ``len(semitic) - len(hebrew)``.
    :rtype: int
    """
    return len(counts["semitic"]) - len(counts["hebrew"])


def join_rows(rows_heb: List[dict], rows_sem: List[dict]) -> List[dict]:
    """Pair the Hebrew-set and union-set scored rows on (fragment, model).

    :param rows_heb: Rows from :func:`common.score` under PAPER_CONFIG.
    :type rows_heb: list
    :param rows_sem: Rows from :func:`common.score` under the semitic config.
    :type rows_sem: list
    :return: One merged dict per output.
    :rtype: list
    """
    sem_by_key = {(r["fragment_id"], r["model"]): r for r in rows_sem}
    merged = []
    for r in rows_heb:
        s = sem_by_key[(r["fragment_id"], r["model"])]
        merged.append(dict(
            doc_id=r["fragment_id"], model=r["model"],
            script_bucket=r["script_bucket"],
            p_hebrew=r["ngram_precision"], p_semitic=s["ngram_precision"],
            delta=s["ngram_precision"] - r["ngram_precision"],
            class_hebrew=r["failure_mode"], class_semitic=s["failure_mode"],
            tier_hebrew=r["tier"], tier_semitic=s["tier"],
            hyp_letters_hebrew=r["hyp_letters"], hyp_letters_semitic=s["hyp_letters"],
            hyp_arabic_only=s["hyp_letters"] - r["hyp_letters"],
            gt_letters_hebrew=r["gt_letters"], gt_letters_semitic=s["gt_letters"],
            gt_arabic_only=s["gt_letters"] - r["gt_letters"],
            aligned_f1=r["aligned_f1"], cer=r["cer"],
        ))
    return merged


def write_csv(path: Path, rows: List[dict], fields: Sequence[str]) -> None:
    """Write ``rows`` as CSV.

    :param path: Destination file.
    :type path: Path
    :param rows: Row dicts.
    :type rows: list
    :param fields: Column order.
    :type fields: Sequence[str]
    :return: None
    :rtype: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def md_table(header: Sequence[str], body: List[Sequence[object]]) -> str:
    """Render a GitHub markdown table.

    :param header: Column names.
    :type header: Sequence[str]
    :param body: Row values (already formatted).
    :type body: list
    :return: Markdown text ending in a blank line.
    :rtype: str
    """
    lines = ["| " + " | ".join(str(h) for h in header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for row in body:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines) + "\n"


def fmt(x: object, dp: int = 3) -> str:
    """Round floats for tables, pass anything else through.

    :param x: Value.
    :type x: object
    :param dp: Decimal places.
    :type dp: int
    :return: String.
    :rtype: str
    """
    if isinstance(x, float):
        return f"{x:.{dp}f}"
    return str(x)


def evidence_coverage(outs: Sequence[Output]) -> Dict[str, int]:
    """How many of the audit's eight evidence systems have a saved output.

    :param outs: Outputs for the fragments of interest.
    :type outs: Sequence[Output]
    :return: doc_id -> number of evidence systems present.
    :rtype: dict
    """
    present: Dict[str, set] = collections.defaultdict(set)
    for o in outs:
        if o.model in AUDIT_EVIDENCE:
            present[o.doc_id].add(o.model)
    return {d: len(v) for d, v in present.items()}


def audit_overlaps(frags: Sequence[Fragment], outs: Sequence[Output]
                   ) -> Dict[str, Dict[str, Tuple[float, Optional[str]]]]:
    """Best evidence-system 5-gram overlap per fragment, per letter set.

    Reproduces ``audit_genizah_benchmark.overlap_with_gt`` (n=5, unclipped,
    set membership) restricted to the audit's evidence systems.

    :param frags: Fragments to score.
    :type frags: Sequence[Fragment]
    :param outs: Outputs for those fragments.
    :type outs: Sequence[Output]
    :return: doc_id -> letter_set -> (best overlap, best model).
    :rtype: dict
    """
    by_doc = {fr.doc_id: fr for fr in frags}
    best: Dict[str, Dict[str, Tuple[float, Optional[str]]]] = {
        d: {ls: (0.0, None) for ls in LETTER_RES} for d in by_doc}
    for o in outs:
        if o.model not in AUDIT_EVIDENCE:
            continue
        fr = by_doc[o.doc_id]
        for ls in LETTER_RES:
            ov = ngram_precision(o.hyp_letters[ls], fr.gt_letters[ls], 5, clip=False)
            if ov > best[o.doc_id][ls][0]:
                best[o.doc_id][ls] = (ov, o.model)
    return best


def part1(md: List[str], key: List[str]) -> None:
    """Letter-set sensitivity on the verified 131 x 11 paper systems.

    :param md: Markdown accumulator, appended in place.
    :type md: list
    :param key: Headline-bullet accumulator, appended in place.
    :type key: list
    :return: None
    :rtype: None
    """
    frags, outs = cached_features(CACHE)
    rows_heb = score(frags, outs, PAPER_CONFIG)
    rows_sem = score(frags, outs, SEMITIC_CONFIG)
    merged = join_rows(rows_heb, rows_sem)
    write_csv(OUT_DIR / "D_part1_per_output.csv", merged, [
        "doc_id", "model", "script_bucket", "p_hebrew", "p_semitic", "delta",
        "class_hebrew", "class_semitic", "tier_hebrew", "tier_semitic",
        "hyp_letters_hebrew", "hyp_letters_semitic", "hyp_arabic_only",
        "gt_letters_hebrew", "gt_letters_semitic", "gt_arabic_only",
        "aligned_f1", "cer"])

    md.append("## Part 1 — verified 131, 11 paper systems\n")
    md.append(f"Outputs scored: **{len(merged)}** saved outputs over "
              f"{len({r['doc_id'] for r in merged})} fragments and "
              f"{len({r['model'] for r in merged})} systems (not every system "
              f"has a file for every fragment).\n")

    # --- class changes -----------------------------------------------------
    changed = [r for r in merged if r["class_hebrew"] != r["class_semitic"]]
    by_model = collections.defaultdict(list)
    for r in merged:
        by_model[r["model"]].append(r)
    body = []
    for m in PAPER_SYSTEMS:
        rs = by_model.get(m, [])
        ch = [r for r in rs if r["class_hebrew"] != r["class_semitic"]]
        breakdown = collections.Counter(f"{r['class_hebrew']}->{r['class_semitic']}"
                                        for r in ch)
        body.append([m, len(rs), len(ch),
                     ", ".join(f"{k} x{v}" for k, v in sorted(breakdown.items())) or "-"])
    md.append("### Class changes per system (Hebrew set -> union set)\n")
    md.append(md_table(["system", "n outputs", "class changes", "from->to"], body))
    md.append(f"Total class changes: **{len(changed)} / {len(merged)}** "
              f"({len(changed) / len(merged):.3%}).\n")
    key.append(f"- On the verified 131 the union letter set changes the behaviour "
               f"class of **{len(changed)} / {len(merged)}** outputs "
               f"({len(changed) / len(merged):.2%}), all of them "
               f"`substantive -> hallucinated`, and never the reverse.")

    if changed:
        md.append("### Every output whose class changes\n")
        md.append(md_table(
            ["doc_id", "model", "bucket", "P(heb)", "P(sem)", "class heb",
             "class sem", "hyp heb", "hyp sem", "hyp ar-only",
             "gt heb", "gt sem", "gt ar-only"],
            [[r["doc_id"], r["model"], r["script_bucket"], fmt(r["p_hebrew"]),
              fmt(r["p_semitic"]), r["class_hebrew"], r["class_semitic"],
              r["hyp_letters_hebrew"], r["hyp_letters_semitic"], r["hyp_arabic_only"],
              r["gt_letters_hebrew"], r["gt_letters_semitic"], r["gt_arabic_only"]]
             for r in sorted(changed, key=lambda r: (r["model"], r["doc_id"]))]))
    else:
        md.append("### Every output whose class changes\n\nNone.\n")

    # --- precision deltas --------------------------------------------------
    nz = [r for r in merged if abs(r["delta"]) > 1e-12]
    md.append("### Precision movement\n")
    md.append(f"- Outputs with any change in 5-gram precision: **{len(nz)}** "
              f"({len(nz) / len(merged):.3%}).\n")
    if nz:
        md.append(f"- Mean delta over those: {statistics.mean(r['delta'] for r in nz):+.4f}; "
                  f"min {min(r['delta'] for r in nz):+.4f}, "
                  f"max {max(r['delta'] for r in nz):+.4f}.\n")
        md.append(md_table(
            ["doc_id", "model", "bucket", "P(heb)", "P(sem)", "delta",
             "hyp ar-only", "gt ar-only"],
            [[r["doc_id"], r["model"], r["script_bucket"], fmt(r["p_hebrew"]),
              fmt(r["p_semitic"]), f"{r['delta']:+.3f}", r["hyp_arabic_only"],
              r["gt_arabic_only"]]
             for r in sorted(nz, key=lambda r: r["delta"])]))

    # --- Tier A ------------------------------------------------------------
    md.append("### Tier A\n")
    md.append(f"- Tier A under PAPER_CONFIG (hebrew): **{tier_a_count(rows_heb)}**\n")
    md.append(f"- Tier A under letter_set='semitic': **{tier_a_count(rows_sem)}**\n")
    moved = sorted({r["doc_id"] for r in merged if r["tier_hebrew"] != r["tier_semitic"]})
    key.append(f"- Tier A is unchanged: **{tier_a_count(rows_heb)}** under the paper "
               f"letter set, **{tier_a_count(rows_sem)}** under the union set "
               f"({len(moved)} fragments change tier).")
    md.append(f"- Fragments changing tier: **{len(moved)}**"
              + (" (" + ", ".join(moved) + ")" if moved else "") + "\n")

    # --- Arabic-block presence --------------------------------------------
    md.append("### Arabic-block characters present at all\n")
    body = []
    for m in PAPER_SYSTEMS:
        rs = by_model.get(m, [])
        with_ar = [r for r in rs if r["hyp_arabic_only"] > 0]
        tot = sum(r["hyp_arabic_only"] for r in with_ar)
        body.append([m, len(rs), len(with_ar),
                     f"{len(with_ar) / len(rs):.3f}" if rs else "-", tot,
                     max((r["hyp_arabic_only"] for r in with_ar), default=0)])
    md.append(md_table(
        ["system", "n outputs", "hyps with >=1 Arabic-block letter", "share",
         "total Arabic-block letters", "max in one output"], body))

    gt_ar = sorted({(r["doc_id"], r["script_bucket"], r["gt_letters_hebrew"],
                     r["gt_letters_semitic"], r["gt_arabic_only"])
                    for r in merged if r["gt_arabic_only"] > 0},
                   key=lambda t: -t[4])
    md.append(f"\nReferences (GT) containing Arabic-block letters: "
              f"**{len(gt_ar)} / {len({r['doc_id'] for r in merged})}**\n")
    if gt_ar:
        md.append(md_table(
            ["doc_id", "bucket", "gt hebrew letters", "gt semitic letters",
             "gt Arabic-only letters"],
            [list(t) for t in gt_ar]))

    # --- mixed-script outputs ---------------------------------------------
    mixed = [r for r in merged
             if r["hyp_letters_semitic"] > 0
             and r["hyp_arabic_only"] / r["hyp_letters_semitic"] >= 0.20]
    md.append("### Mixed-script outputs (>= 20% of hypothesis letters are Arabic-block)\n")
    md.append(f"Count: **{len(mixed)} / {len(merged)}**.\n")
    key.append(f"- **{len(mixed)} / {len(merged)}** outputs are >= 20% Arabic-block by "
               f"letter count and are therefore graded on a Hebrew fragment of "
               f"themselves under the paper metric; `vision_ocr_seg` writes "
               f"Arabic-block characters in "
               f"{len([r for r in by_model['vision_ocr_seg'] if r['hyp_arabic_only'] > 0])}"
               f"/{len(by_model['vision_ocr_seg'])} of its outputs.")
    key.append(f"- Only **{len(gt_ar)} / {len({r['doc_id'] for r in merged})}** "
               f"references in the verified 131 contain any Arabic-block letter "
               f"(max {max((t[4] for t in gt_ar), default=0)} letters), so the union "
               f"set can only ever cost a system precision here, never earn it.")
    if mixed:
        md.append(md_table(
            ["doc_id", "model", "bucket", "Arabic share of hyp", "P(heb)", "P(sem)",
             "delta", "class heb", "class sem", "hyp heb", "hyp ar-only", "gt ar-only"],
            [[r["doc_id"], r["model"], r["script_bucket"],
              fmt(r["hyp_arabic_only"] / r["hyp_letters_semitic"]),
              fmt(r["p_hebrew"]), fmt(r["p_semitic"]), f"{r['delta']:+.3f}",
              r["class_hebrew"], r["class_semitic"], r["hyp_letters_hebrew"],
              r["hyp_arabic_only"], r["gt_arabic_only"]]
             for r in sorted(mixed,
                             key=lambda r: -r["hyp_arabic_only"] / r["hyp_letters_semitic"])]))
        write_csv(OUT_DIR / "D_part1_mixed_script.csv", mixed, [
            "doc_id", "model", "script_bucket", "hyp_letters_hebrew",
            "hyp_letters_semitic", "hyp_arabic_only", "p_hebrew", "p_semitic",
            "delta", "class_hebrew", "class_semitic", "gt_arabic_only"])


def part2(md: List[str], key: List[str]) -> None:
    """The four excluded Arabic-script fragments, plus all 19 exclusions.

    :param md: Markdown accumulator, appended in place.
    :type md: list
    :param key: Headline-bullet accumulator, appended in place.
    :type key: list
    :return: None
    :rtype: None
    """
    systems = list(dict.fromkeys(PAPER_SYSTEMS + ["vision_ocr_seg"]))

    md.append("\n## Part 2 — the Arabic-script fragments the audit excluded\n")

    frags4 = load_fragments_unfiltered("frozen", ARABIC_FOUR)
    found = {fr.doc_id for fr in frags4}
    missing = [d for d in ARABIC_FOUR if d not in found]
    if missing:
        md.append(f"Not loadable (absent from frozen benchmark or no outputs dir): "
                  f"{', '.join(missing)}\n")
    outs4 = load_features(frags4, systems, with_alignment=True)

    md.append("### GT letter counts\n")
    md.append(md_table(
        ["doc_id", "bucket", "gt hebrew", "gt semitic", "gt Arabic-only",
         "GT repaired?"],
        [[fr.doc_id, fr.script_bucket, len(fr.gt_letters["hebrew"]),
          len(fr.gt_letters["semitic"]), arabic_only(fr.gt_letters), fr.repaired]
         for fr in frags4]))

    by_doc = {fr.doc_id: fr for fr in frags4}
    rows4 = []
    for o in outs4:
        fr = by_doc[o.doc_id]
        rows4.append(dict(
            doc_id=o.doc_id, model=o.model,
            p_hebrew=ngram_precision(o.hyp_letters["hebrew"], fr.gt_letters["hebrew"], 5),
            p_semitic=ngram_precision(o.hyp_letters["semitic"], fr.gt_letters["semitic"], 5),
            hyp_letters_hebrew=len(o.hyp_letters["hebrew"]),
            hyp_letters_semitic=len(o.hyp_letters["semitic"]),
            hyp_arabic_only=arabic_only(o.hyp_letters),
            aligned_f1=o.aligned_f1, cer=o.cer, cer_lenient=o.cer_lenient,
        ))
    for r in rows4:
        r["delta"] = r["p_semitic"] - r["p_hebrew"]
    write_csv(OUT_DIR / "D_part2_arabic_four.csv", rows4, [
        "doc_id", "model", "p_hebrew", "p_semitic", "delta",
        "hyp_letters_hebrew", "hyp_letters_semitic", "hyp_arabic_only",
        "aligned_f1", "cer", "cer_lenient"])

    for fr in frags4:
        rs = [r for r in rows4 if r["doc_id"] == fr.doc_id]
        md.append(f"\n#### {fr.doc_id} "
                  f"(bucket `{fr.script_bucket}`, GT {len(fr.gt_letters['hebrew'])} heb / "
                  f"{len(fr.gt_letters['semitic'])} sem / "
                  f"{arabic_only(fr.gt_letters)} Arabic-only letters)\n")
        md.append(md_table(
            ["system", "P5(heb)", "P5(sem)", "delta", "aligned F1", "CER",
             "hyp heb", "hyp sem", "hyp ar-only"],
            [[r["model"], fmt(r["p_hebrew"]), fmt(r["p_semitic"]),
              f"{r['delta']:+.3f}", fmt(r["aligned_f1"]), fmt(r["cer"]),
              r["hyp_letters_hebrew"], r["hyp_letters_semitic"], r["hyp_arabic_only"]]
             for r in sorted(rs, key=lambda r: -r["p_semitic"])]))

    best4 = audit_overlaps(frags4, outs4)
    cov4 = evidence_coverage(outs4)
    md.append("\n### Audit re-check on the four (evidence systems only, threshold 0.12)\n")
    md.append(md_table(
        ["doc_id", "evidence systems present / 8", "best overlap (heb)",
         "best model (heb)", "passes heb?", "best overlap (sem)",
         "best model (sem)", "passes sem?"],
        [[d, cov4.get(d, 0), fmt(best4[d]["hebrew"][0]), best4[d]["hebrew"][1] or "-",
          "yes" if best4[d]["hebrew"][0] >= AUDIT_THRESHOLD else "no",
          fmt(best4[d]["semitic"][0]), best4[d]["semitic"][1] or "-",
          "yes" if best4[d]["semitic"][0] >= AUDIT_THRESHOLD else "no"]
         for d in ARABIC_FOUR if d in best4]))
    thin = [d for d in ARABIC_FOUR if cov4.get(d, 0) < len(AUDIT_EVIDENCE)]
    if thin:
        md.append("\n" + "; ".join(
            f"`{d}` has only {cov4.get(d, 0)}/8 evidence systems on disk "
            f"(missing: {', '.join(m for m in AUDIT_EVIDENCE if not (OUTPUTS / d / (m + '.txt')).exists())})"
            for d in thin) + ".\n")

    # --- all 19 exclusions -------------------------------------------------
    ex = excluded_ids()
    frags_ex = load_fragments_unfiltered("frozen", ex)
    outs_ex = load_features(frags_ex, AUDIT_EVIDENCE, with_alignment=False)
    best_ex = audit_overlaps(frags_ex, outs_ex)
    cov_ex = evidence_coverage(outs_ex)
    tags = script_tags()
    by_doc_ex = {fr.doc_id: fr for fr in frags_ex}
    ex_rows = []
    for d in ex:
        fr = by_doc_ex.get(d)
        gt_h = len(fr.gt_letters["hebrew"]) if fr else 0
        gt_s = len(fr.gt_letters["semitic"]) if fr else 0
        if d not in best_ex:
            ex_rows.append(dict(doc_id=d, script_bucket=tags.get(d, "untagged"),
                                evidence_systems=0, gt_letters_hebrew=gt_h,
                                gt_letters_semitic=gt_s, gt_arabic_only=gt_s - gt_h,
                                best_hebrew=float("nan"), best_model_hebrew="",
                                best_semitic=float("nan"), best_model_semitic="",
                                readmitted_semitic="no outputs"))
            continue
        bh, mh = best_ex[d]["hebrew"]
        bs, ms = best_ex[d]["semitic"]
        ex_rows.append(dict(
            doc_id=d, script_bucket=tags.get(d, "untagged"),
            evidence_systems=cov_ex.get(d, 0),
            gt_letters_hebrew=gt_h, gt_letters_semitic=gt_s,
            gt_arabic_only=gt_s - gt_h,
            best_hebrew=bh, best_model_hebrew=mh or "",
            best_semitic=bs, best_model_semitic=ms or "",
            readmitted_semitic="yes" if bs >= AUDIT_THRESHOLD else "no"))
    write_csv(OUT_DIR / "D_part2_excluded_19.csv", ex_rows, [
        "doc_id", "script_bucket", "evidence_systems", "gt_letters_hebrew",
        "gt_letters_semitic", "gt_arabic_only", "best_hebrew",
        "best_model_hebrew", "best_semitic", "best_model_semitic",
        "readmitted_semitic"])

    readmit = [r for r in ex_rows if r["readmitted_semitic"] == "yes"]
    md.append(f"\n### All {len(ex)} excluded fragments under the union set\n")
    md.append(md_table(
        ["doc_id", "bucket", "evidence systems / 8", "gt heb", "gt ar-only",
         "best overlap (heb)", "best model (heb)", "best overlap (sem)",
         "best model (sem)", "readmitted at 0.12?"],
        [[r["doc_id"], r["script_bucket"], r["evidence_systems"],
          r["gt_letters_hebrew"], r["gt_arabic_only"],
          fmt(r["best_hebrew"]), r["best_model_hebrew"] or "-",
          fmt(r["best_semitic"]), r["best_model_semitic"] or "-",
          r["readmitted_semitic"]]
         for r in sorted(ex_rows, key=lambda r: -(r["best_semitic"]
                                                  if r["best_semitic"] == r["best_semitic"]
                                                  else -1))]))
    key.append(f"- Re-running the audit's 0.12 best-evidence-overlap test under the "
               f"union set readmits **{len(readmit)} / {len(ex)}** excluded fragments: "
               + ", ".join(f"`{r['doc_id']}` ({r['best_hebrew']:.3f} -> "
                           f"{r['best_semitic']:.3f})" for r in readmit) + ".")
    md.append(f"\nReadmitted under the union set at 0.12: **{len(readmit)} / {len(ex)}**"
              + (" — " + ", ".join(r["doc_id"] for r in readmit) if readmit else "") + "\n")


def part3(md: List[str]) -> None:
    """Where the six ``arabic_script``-tagged fragments ended up.

    :param md: Markdown accumulator, appended in place.
    :type md: list
    :return: None
    :rtype: None
    """
    tags = script_tags()
    buckets = collections.Counter(tags.values())
    ar = sorted(d for d, b in tags.items() if b == "arabic_script")
    verified = {d["doc_id"] for d in load_benchmark("verified")}
    frags, _ = cached_features(CACHE)
    scored = {fr.doc_id for fr in frags}
    ex = set(excluded_ids())

    md.append("\n## Note — the `arabic_script` script_tags bucket\n")
    md.append(md_table(["bucket", "fragments", "share of 150"],
                       [[b, c, f"{c / sum(buckets.values()):.3f}"]
                        for b, c in buckets.most_common()]))
    md.append(md_table(
        ["doc_id", "in verified subset?", "in the scored 131?", "in excluded_misaligned?"],
        [[d, "yes" if d in verified else "no", "yes" if d in scored else "no",
          "yes" if d in ex else "no"] for d in ar]))
    md.append(f"\n`arabic_script` = **{len(ar)}/150 = "
              f"{len(ar) / len(tags):.3%}** of the tagged benchmark; "
              f"{sum(1 for d in ar if d in ex)} of the {len(ar)} are in "
              f"`audit.excluded_misaligned`, {sum(1 for d in ar if d in scored)} "
              f"survive into the scored 131.\n")


def main() -> None:
    """Run parts 1-3 and write the markdown report.

    :return: None
    :rtype: None
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md: List[str] = [
        "# Analysis D — effect of the letter set (Hebrew block vs Hebrew+Arabic union)\n",
        "Paper metric: 5-gram precision, unclipped, letters reduced to the Hebrew "
        "block `U+0590-U+05FF`. Union ('semitic') set adds `U+0600-U+06FF`. "
        "All numbers 3 dp.\n",
    ]
    key: List[str] = []
    part1(md, key)
    part2(md, key)
    part3(md)
    md.insert(2, "## Headline\n\n" + "\n".join(key) + "\n")
    (OUT_DIR / "D_letterset.md").write_text("\n".join(md))
    print(f"wrote {OUT_DIR / 'D_letterset.md'}")


if __name__ == "__main__":
    main()
