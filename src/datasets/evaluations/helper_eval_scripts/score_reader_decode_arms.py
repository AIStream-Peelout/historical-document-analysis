# File name: score_reader_decode_arms.py
# Date: 10/29/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Score the page reader's decode arms (``run_reader_decode_arms.py``) on the coverage map (offline, no model).

Every arm found in ``--out-dir`` (and the saved trained-prompt reads of the same checkpoint, ``trained2500``) is scored
with ``score_arabic_covered`` on ``coverage_v2.jsonl``, on the paired set (the 58 documents of the earlier v2.5a
reports) and on all 60 documents:

* **Table A, recognition** (covered images only): median 5-gram F1 and LER over the documents with a covered image,
  and the paired comparison of each arm against arm 0 (per-document deltas, 20,000 bootstrap resamples of
  documents, win / loss / tie counts);
* **Table B, over-reading** (uncovered and blank images): answered images, images with >= 40 Arabic letters by side
  script (Hebrew-script sides, Arabic sides, blank / faint images), Hebrew output on Hebrew sides;
* **Reader behaviour** over all 117 images: the scorer's loop flag and ``[...]`` floods in the stored text, first
  requests that hit the token limit, final ``length`` finishes, retries, requests, gated / failed images, images the
  script filter cleaned, turned images and accepted band reads;
* the **7 rotated images** one by one (each scored alone against the sections it shows; a copy against the sections of
  the image it copies) and the **4 under-read documents** one by one;
* per arm, the documents whose F1 moved by at least 0.05 against arm 0 (what the arm fixed and broke).

Outputs: the Markdown tables (``--report``), ``scores/<arm>[.paired].csv`` and ``scores/summary.json`` in the out dir.

Usage::

    .venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.score_reader_decode_arms \\
        [--out-dir <benchmark>/outputs_decode_1029] [--report logs/next_round/arabic_decode_1029/tables_auto.md]
"""
import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.datasets.evaluations import arabic_script as ar
from src.datasets.evaluations.helper_eval_scripts import score_arabic_benchmark as sab
from src.datasets.evaluations.helper_eval_scripts import score_arabic_covered as sac
from src.models.ocr import reader_options as ro

ARM_ORDER = ("trained2500", "arm0", "arm1", "arm2", "arm2_gated", "arm3_all", "arm3", "arm4", "arm5_all",
             "arm5_gatefirst", "arm5", "arm0_rerun")
ARM_LABELS = {"trained2500": "saved reads: trained prompt, 2,500 tokens, no guard",
              "arm0": "production reader (baseline)", "arm1": "+ marker-aware loop detector",
              "arm2": "+ orientation probe on every page (calibrated margin)",
              "arm2_gated": "+ orientation probe after a suspect upright read only", "arm3_all": "+ band crops on every page", "arm3": "+ band crops, auto trigger",
              "arm4": "+ script post-filter", "arm5_all": "all options, crops on every page", "arm5_gatefirst": "all options, orientation gate before the band crops",
              "arm5": "all options (band crops, then the orientation gate)",
              "arm0_rerun": "production reader re-read (no cache): noise"}
TRAINED_SOURCE = "outputs/qwen3-vl-8b-heb-v25a-step1200__trained.jsonl"
PAIRED_IDS = "scores/qwen3-vl-8b-heb-v25a-step1200__trained.v25a.csv"
BASELINE = "arm0"
ROTATED = (("Cambridge_CUL_T_S_NS_38a_126", 1, "90", None), ("Oxford_Bodleian_Bodl_MS_heb_a_3_35", 0, "90", None),
           ("Oxford_Bodleian_Bodl_MS_heb_a_3_35", 2, "270", 0), ("Cambridge_Mosseri_IV_249", 1, "180", None),
           ("Cambridge_CUL_Or_1080_15_77", 1, "180", 2), ("Cambridge_CUL_T_S_H15_62", 1, "180", 2),
           ("Cambridge_CUL_T_S_NS_98_22", 1, "180", 2))          # (doc, image, turn, image it copies)
UNDER_READ = ("Cambridge_CUL_T_S_Ar_53_45", "Cambridge_CUL_T_S_12_172", "Cambridge_CUL_T_S_Misc_29_8",
              "Cambridge_CUL_T_S_Ar_42_143")
FLOOD_MARKERS = 20
MOVE = 0.05


def load_arm_rows(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """An arm's rows by image (the last row of an image wins).

    :param path: Arm JSONL.
    :type path: Path
    :return: ``{(doc_id, image_index): row}``.
    :rtype: Dict[Tuple[str, int], Dict[str, Any]]
    """
    rows: Dict[Tuple[str, int], Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                rows[(row["doc_id"], int(row["image_index"]))] = row
    return rows


def answers_of(rows: Mapping[Tuple[str, int], Mapping[str, Any]]) -> Dict[str, Dict[int, str]]:
    """The scorer's answer layout: text per document and image (rows without text are left out).

    :param rows: Rows by image.
    :type rows: Mapping[Tuple[str, int], Mapping[str, Any]]
    :return: ``{doc_id: {image_index: text}}``.
    :rtype: Dict[str, Dict[int, str]]
    """
    out: Dict[str, Dict[int, str]] = {}
    for (doc, index), row in rows.items():
        if isinstance(row.get("text"), str):
            out.setdefault(doc, {})[index] = row["text"]
    return out


def discover_arms(out_dir: Path, benchmark: Path) -> Dict[str, Dict[Tuple[str, int], Dict[str, Any]]]:
    """The arms to score, in :data:`ARM_ORDER` (then any other arm file, by name).

    :param out_dir: The arms' output directory.
    :type out_dir: Path
    :param benchmark: Benchmark directory (the saved trained-prompt reads).
    :type benchmark: Path
    :return: ``{arm: rows}``.
    :rtype: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]]
    """
    found = {p.stem: p for p in out_dir.glob("*.jsonl")
             if p.stem not in ("panel", "request_cache", "orientation_calibration") and not p.stem.startswith("smoke")}
    trained = benchmark / TRAINED_SOURCE
    if trained.exists():
        found["trained2500"] = trained
    names = [a for a in ARM_ORDER if a in found] + sorted(a for a in found if a not in ARM_ORDER)
    return {a: load_arm_rows(found[a]) for a in names}


def behaviour(rows: Mapping[Tuple[str, int], Mapping[str, Any]], keys: Sequence[Tuple[str, int]]) -> Dict[str, Any]:
    """What the reader did on the panel images.

    :param rows: An arm's rows.
    :type rows: Mapping[Tuple[str, int], Mapping[str, Any]]
    :param keys: Panel images.
    :type keys: Sequence[Tuple[str, int]]
    :return: Counts (``None`` where the arm's rows lack the field, e.g. the saved reads).
    :rtype: Dict[str, Any]
    """
    reads = [rows[k] for k in keys if k in rows]
    texts = [r.get("text") for r in reads if isinstance(r.get("text"), str)]
    has = lambda field: any(field in r for r in reads)
    count = lambda pred: sum(1 for r in reads if pred(r))
    crops = [r.get("crop") for r in reads if r.get("crop")]
    probes = sum(len((r.get("orientation") or {}).get("probes") or {}) for r in reads)
    return {"images": len(reads), "answered": len(texts),
            "loop_flag": sum(ar.repeat_share(t) > sab.LOOP_REPEAT_SHARE for t in texts),
            "floods": sum(ro.marker_count(t) >= FLOOD_MARKERS for t in texts),
            "first_length": count(lambda r: r.get("first_finish_reason") == "length") if has("first_finish_reason") else None,
            "final_length": count(lambda r: r.get("finish_reason") == "length") if has("finish_reason") else None,
            "retried": count(lambda r: r.get("retried")) if has("retried") else None,
            "requests": (sum(int(r.get("attempts") or 0) for r in reads) + probes) if has("attempts") else None,
            "blank_gated": count(lambda r: r.get("reason") == "blank") if has("reason") else None,
            "failed": count(lambda r: r.get("reason") == "failed") if has("reason") else None,
            "filtered": count(lambda r: (r.get("stripped_chars") or 0) >= ro.FILTER_RETRY_CHARS) if has("stripped_chars") else None,
            "stripped_chars": sum(int(r.get("stripped_chars") or 0) for r in reads) if has("stripped_chars") else None,
            "probed": count(lambda r: (r.get("orientation") or {}).get("probes")) if has("orientation") else None,
            "turned": count(lambda r: r.get("rotation")) if has("rotation") else None,
            "cropped": len(crops) if has("crop") else None,
            "crops_accepted": sum(bool(c.get("accepted")) for c in crops) if has("crop") else None,
            "tokens_mean": round(statistics.mean(int(r["completion_tokens"]) for r in reads if r.get("completion_tokens")), 1)
            if any(r.get("completion_tokens") for r in reads) else None}


def image_score(record: Mapping[str, Any], coverage: Mapping[int, Mapping[str, Any]], index: int, copy_of: Optional[int],
                text: Optional[str]) -> Dict[str, Any]:
    """One image scored alone against the sections it shows (a copy: the sections of the image it copies).

    :param record: Benchmark record.
    :type record: Mapping[str, Any]
    :param coverage: The document's coverage rows.
    :type coverage: Mapping[int, Mapping[str, Any]]
    :param index: Image index.
    :type index: int
    :param copy_of: Image whose sections apply (None: the image's own).
    :type copy_of: Optional[int]
    :param text: The arm's text for the image.
    :type text: Optional[str]
    :return: ``ngram_f1``, ``ler``, ``answer_letters``, ``gt_letters`` (None fields when unanswered).
    :rtype: Dict[str, Any]
    """
    sections = coverage[copy_of if copy_of is not None else index].get("sections") or []
    if text is None or not sections:
        return {"ngram_f1": None, "ler": None, "answer_letters": None, "gt_letters": None}
    reduced = sac.covered_record(record, [index], sections)
    row = sab.score_document(reduced, {0: text})
    return {k: row[k] for k in ("ngram_f1", "ler", "answer_letters", "gt_letters")}


def median_difference(new_rows: Sequence[Mapping[str, Any]], old_rows: Sequence[Mapping[str, Any]], metric: str,
                      resamples: int = sac.BOOTSTRAP_RESAMPLES, seed: int = sac.BOOTSTRAP_SEED) -> Dict[str, Any]:
    """Difference of the two arms' medians, ``median(new) - median(old)``, with a paired bootstrap interval.

    Documents are resampled together (both arms' values of a document stay paired), as in
    :func:`score_arabic_covered.bootstrap_ci`. Unlike the median of per-document deltas, this moves when an arm changes
    only a few documents that sit near the median.

    :param new_rows: Document rows of the arm.
    :type new_rows: Sequence[Mapping[str, Any]]
    :param old_rows: Document rows of arm 0.
    :type old_rows: Sequence[Mapping[str, Any]]
    :param metric: ``ngram_f1`` or ``ler``.
    :type metric: str
    :param resamples: Bootstrap resamples.
    :type resamples: int
    :param seed: Bootstrap seed.
    :type seed: int
    :return: ``documents``, ``difference`` and ``ci`` (95 %), or ``{"documents": 0}``.
    :rtype: Dict[str, Any]
    """
    import numpy as np

    old = {r["doc_id"]: r[metric] for r in old_rows if r[metric] is not None}
    pairs = np.array([(r[metric], old[r["doc_id"]]) for r in new_rows if r[metric] is not None and r["doc_id"] in old], dtype=float)
    if not len(pairs):
        return {"documents": 0}
    rng = np.random.default_rng(seed)
    draws = pairs[rng.integers(0, len(pairs), size=(resamples, len(pairs)))]
    diffs = np.median(draws[:, :, 0], axis=1) - np.median(draws[:, :, 1], axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"documents": len(pairs), "difference": round(float(np.median(pairs[:, 0]) - np.median(pairs[:, 1])), 4),
            "ci": [round(float(lo), 4), round(float(hi), 4)]}


def _d(cmp: Mapping[str, Any]) -> str:
    """``+0.012 [−0.004, +0.030]`` for :func:`median_difference`."""
    if not cmp.get("documents"):
        return "–"
    return f"{cmp['difference']:+.3f} [{cmp['ci'][0]:+.3f}, {cmp['ci'][1]:+.3f}]"


def _f(v: Any, places: int = 3) -> str:
    """A table cell."""
    if v is None:
        return "–"
    return f"{v:.{places}f}" if isinstance(v, float) else str(v)


def _ci(cmp: Mapping[str, Any], key: str = "median") -> str:
    """``+0.012 [−0.004, +0.030]``."""
    if not cmp.get("documents"):
        return "–"
    lo, hi = cmp[f"{key}_ci"]
    return f"{cmp[f'{key}_delta']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def score(out_dir: Path, benchmark: Path, coverage_path: Optional[Path] = None) -> Tuple[str, Dict[str, Any]]:
    """Score every arm and build the tables.

    :param out_dir: The arms' output directory.
    :type out_dir: Path
    :param benchmark: Benchmark directory.
    :type benchmark: Path
    :param coverage_path: Coverage map (default ``coverage_v2.jsonl``).
    :type coverage_path: Optional[Path]
    :return: ``(Markdown, summary)``.
    :rtype: Tuple[str, Dict[str, Any]]
    """
    bench = sab.load_benchmark(benchmark)
    coverage = sac.load_coverage(coverage_path or benchmark / sac.DEFAULT_COVERAGE)
    arms = discover_arms(out_dir, benchmark)
    panel = [json.loads(l) for l in (out_dir / "panel.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    keys = [(p["doc_id"], int(p["image_index"])) for p in panel]
    answers = {a: answers_of({k: r for k, r in rows.items() if k in set(keys)}) for a, rows in arms.items()}
    paired = sac.read_ids(benchmark / PAIRED_IDS)
    sets = {"paired": paired, "all": None}
    results = {name: sac.run_set(bench, coverage, answers, only) for name, only in sets.items()}
    (out_dir / "scores").mkdir(exist_ok=True)
    for name, result in results.items():
        suffix = "" if name == "all" else f".{name}"
        for arm in arms:
            sac.write_rows(out_dir / "scores" / f"{arm}{suffix}.csv", result["docs"][arm], sac.DOC_FIELDS)
            sac.write_rows(out_dir / "scores" / f"{arm}{suffix}.images.csv", result["images"][arm], sac.IMAGE_FIELDS)
    parts = ["# Decode arms: automatic tables", "",
             f"Arms: {', '.join(f'`{a}` = {ARM_LABELS.get(a, a)}' for a in arms)}.", ""]
    summary: Dict[str, Any] = {"arms": list(arms), "sets": {}}
    for name, result in results.items():
        docs = result["docs"]
        base = docs.get(BASELINE)
        lines = [f"## Set `{name}`: {len(next(iter(docs.values())))} documents", "",
                 "### Table A: recognition on covered images, paired against arm 0", "",
                 "| arm | rec. docs | median F1 | median LER | mean F1 | F1 >= 0.2 | loop status | Δ median F1 vs arm0 [95 % CI] "
                 "| Δ median LER [95 % CI] | median of ΔF1 [95 % CI] | mean ΔF1 [95 % CI] | W/L/T F1 | median of ΔLER [95 % CI] "
                 "| W/L/T LER |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        set_summary: Dict[str, Any] = {}
        for arm in arms:
            s = result["summary"][arm]
            row = (f"| {arm} | {s['recognition_documents']} | {_f(s['recognition_f1_median'])} | {_f(s['recognition_ler_median'])} | "
                   f"{_f(s['recognition_f1_mean'])} | {s['recognition_f1_at_least_0.2']} | {s['recognition_loop']} | ")
            cmp_f = cmp_l = med_f = med_l = {}
            if base is not None and arm != BASELINE:
                cmp_f = sac.paired_comparison(docs[arm], base, "ngram_f1", True)
                cmp_l = sac.paired_comparison(docs[arm], base, "ler", False)
                med_f, med_l = median_difference(docs[arm], base, "ngram_f1"), median_difference(docs[arm], base, "ler")
                row += (f"{_d(med_f)} | {_d(med_l)} | {_ci(cmp_f)} | {_ci(cmp_f, 'mean')} | "
                        f"{cmp_f.get('wins')}/{cmp_f.get('losses')}/{cmp_f.get('ties')} | {_ci(cmp_l)} | "
                        f"{cmp_l.get('wins')}/{cmp_l.get('losses')}/{cmp_l.get('ties')} |")
            else:
                row += "– | – | – | – | – | – | – |"
            lines.append(row)
            set_summary[arm] = {"summary": s, "vs_arm0_f1": cmp_f, "vs_arm0_ler": cmp_l, "median_diff_f1": med_f,
                                "median_diff_ler": med_l}
        lines += ["", "### Table B: over-reading on images without edited text (answered / >= 40 Arabic letters)", "",
                  "| arm | uncovered answered | Hebrew sides >= 40 Ar | Hebrew sides Hebrew output | Arabic sides >= 40 | mixed >= 40 "
                  "| blank / faint >= 40 | blank Arabic letters mean | over-read loops | ambiguous Arabic letters mean |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for arm in arms:
            s = result["summary"][arm]
            g = lambda grp, stat: s.get(f"overread_{grp}_{stat}")
            heb_n, heb_ans = g("hebrew", "images"), g("hebrew", "answered") or 0
            lines.append(
                f"| {arm} | {g('uncovered', 'answered')}/{g('uncovered', 'images')} | "
                f"{round((g('hebrew', 'ge40') or 0) * heb_ans)}/{heb_n} | {round((g('hebrew', 'hebrew_output') or 0) * heb_ans)}/{heb_n} | "
                f"{round((g('arabic', 'ge40') or 0) * (g('arabic', 'answered') or 0))}/{g('arabic', 'images')} | "
                f"{round((g('mixed', 'ge40') or 0) * (g('mixed', 'answered') or 0))}/{g('mixed', 'images')} | "
                f"{round((g('blank', 'ge40') or 0) * (g('blank', 'answered') or 0))}/{g('blank', 'images')} | "
                f"{_f(s.get('overread_arabic_letters_mean_blank'), 1)} | {_f(s.get('overread_loop_share'))} | "
                f"{_f(s.get('ambiguous_arabic_letters_mean'), 1)} |")
        summary["sets"][name] = set_summary
        parts += lines + [""]
    lines = ["## Reader behaviour on all 117 images", "",
             "| arm | answered | scorer loop flag | `[...]` floods (>= 20) | first request hit limit | final `length` | retried "
             "| requests (incl. probes) | blank-gated | failed | filtered (>= 3 stripped) | chars stripped | probed / turned | cropped / bands kept "
             "| tokens mean |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    summary["behaviour"] = {}
    for arm, rows in arms.items():
        b = behaviour(rows, keys)
        summary["behaviour"][arm] = b
        lines.append(f"| {arm} | {b['answered']} | {b['loop_flag']} | {b['floods']} | {_f(b['first_length'])} | {_f(b['final_length'])} | "
                     f"{_f(b['retried'])} | {_f(b['requests'])} | {_f(b['blank_gated'])} | {_f(b['failed'])} | {_f(b['filtered'])} | "
                     f"{_f(b['stripped_chars'])} | {_f(b['probed'])} / {_f(b['turned'])} | {_f(b['cropped'])} / {_f(b['crops_accepted'])} | "
                     f"{_f(b['tokens_mean'], 1)} |")
    parts += lines + [""]
    lines = ["## The 7 rotated images, each scored alone (F1 / LER / letters; turn chosen by the arm)", "",
             "| image | turn | role | " + " | ".join(arms) + " |", "|---|---|---|" + "---|" * len(arms)]
    summary["rotated"] = {}
    for doc, index, turn, copy_of in ROTATED:
        cells = []
        for arm, rows in arms.items():
            row = rows.get((doc, index)) or {}
            sc = image_score(bench[doc], coverage[doc], index, copy_of, row.get("text"))
            summary["rotated"].setdefault(f"{doc}#{index}", {})[arm] = dict(sc, rotation=row.get("rotation"))
            turned = f" ↻{row['rotation']}" if row.get("rotation") else ""
            cells.append(f"{_f(sc['ngram_f1'], 2)} / {_f(sc['ler'], 2)} / {_f(sc['answer_letters'])}{turned}")
        lines.append(f"| {doc.replace('Cambridge_CUL_', '').replace('Oxford_Bodleian_', '')}#{index} | {turn}° | "
                     f"{coverage[doc][index].get('role')} | " + " | ".join(cells) + " |")
    parts += lines + [""]
    lines = ["## The 4 under-read documents (paired set; covered-only F1 / LER / length ratio)", "",
             "| document | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    docs_paired = {arm: {r["doc_id"]: r for r in results["paired"]["docs"][arm]} for arm in arms}
    for doc in UNDER_READ:
        cells = []
        for arm in arms:
            r = docs_paired[arm].get(doc) or {}
            cells.append(f"{_f(r.get('ngram_f1'), 2)} / {_f(r.get('ler'), 2)} / {_f(r.get('length_ratio'), 2)}")
        lines.append(f"| {doc.replace('Cambridge_CUL_', '')} | " + " | ".join(cells) + " |")
    parts += lines + [""]
    if BASELINE in arms:
        lines = [f"## Documents whose covered-only F1 moved by >= {MOVE} against arm 0 (paired set)", ""]
        for arm in arms:
            if arm == BASELINE:
                continue
            moved = []
            for doc, r in docs_paired[arm].items():
                b = docs_paired[BASELINE].get(doc)
                if b and r.get("ngram_f1") is not None and b.get("ngram_f1") is not None and abs(r["ngram_f1"] - b["ngram_f1"]) >= MOVE:
                    moved.append((r["ngram_f1"] - b["ngram_f1"], doc, b["ngram_f1"], r["ngram_f1"], b["ler"], r["ler"]))
            moved.sort()
            text = "; ".join(f"{d.replace('Cambridge_CUL_', '')} {bf:.2f}→{nf:.2f} (LER {bl:.2f}→{nl:.2f})" for _, d, bf, nf, bl, nl in moved)
            lines.append(f"- **{arm}** ({sum(m[0] > 0 for m in moved)} up, {sum(m[0] < 0 for m in moved)} down): {text or 'none'}")
        parts += lines + [""]
    (out_dir / "scores" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return "\n".join(parts), summary


def main(argv: Optional[List[str]] = None) -> None:
    """CLI.

    :param argv: Arguments.
    :type argv: Optional[List[str]]
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", type=Path, default=sab.DEFAULT_BENCHMARK)
    parser.add_argument("--out-dir", type=Path, default=None, help="default <benchmark>/outputs_decode_1029")
    parser.add_argument("--coverage", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    out_dir = args.out_dir or args.benchmark / "outputs_decode_1029"
    report, _ = score(out_dir, args.benchmark, args.coverage)
    if args.report:
        args.report.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
