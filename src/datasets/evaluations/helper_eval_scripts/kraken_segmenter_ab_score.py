"""Score the Kraken segmenter A/B (``kraken_segmenter_ab.py`` outputs) on both benchmarks.

Uses exactly the harness metrics: ``cer_pair(normalize_ink_hypothesis(hyp),
genizah_visible_ink_gt(gt))`` (strict CER) and ``aligned_prf`` F1, pages with
>= 50 GT letters. Reports per tag: median / mean CER and median F1, slices
(religious: Talmud, multi-column, single-column), paired wins against the
cached kraken 4 rows and (religious) the MiDRASH Zenodo transcriptions, lines
emitted per GT line, and seconds per page.

Writes ``<out>/ab_scores.json`` and ``<out>/ab_tables.md``; nothing in the
benchmark directories is modified.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO))

from src.datasets.evaluations.metrics import (  # noqa: E402
    cer_pair,
    genizah_visible_ink_gt,
    normalize_ink_hypothesis,
)
from src.datasets.evaluations.helper_eval_scripts.audit_genizah_benchmark import letters_only  # noqa: E402
from src.datasets.evaluations.helper_eval_scripts.kraken_segmenter_ab import _BENCHES  # noqa: E402
from src.datasets.evaluations.helper_eval_scripts.score_genizah_offline import (  # noqa: E402
    aligned_prf,
    load_benchmark,
)

BASELINES = ("kraken_raw", "kraken_seg")


def load_docs(bench: str) -> list:
    """Benchmark docs with the GT form each harness scores against.

    :param bench: ``religious`` or ``pgp``.
    :type bench: str
    :return: Doc dicts (PGP GT duplication-repaired as in the offline scorer).
    :rtype: list
    """
    if bench == "pgp":
        return load_benchmark("verified")
    return json.load(open(_BENCHES[bench][0]))["docs"]


def discover_tags(out_root: Path, docs: list) -> list:
    """Kraken 7 tags present in the cache (from ``kraken_lines_<tag>.json`` files).

    :param out_root: Raw-output root of the benchmark.
    :type out_root: Path
    :param docs: Benchmark docs.
    :type docs: list
    :return: Sorted tag names.
    :rtype: list
    """
    tags = set()
    for d in docs:
        for f in (out_root / d["doc_id"]).glob("kraken_lines_*.json"):
            tags.add(f.stem.removeprefix("kraken_lines_"))
    return sorted(t for t in tags if t.startswith("k7_"))


def score_page(hyp_raw: str, gt: str) -> tuple:
    """Strict CER and aligned F1 of one hypothesis.

    :param hyp_raw: Raw engine output.
    :type hyp_raw: str
    :param gt: Benchmark ground truth.
    :type gt: str
    :return: (cer, f1).
    :rtype: tuple
    """
    hyp = normalize_ink_hypothesis(hyp_raw)
    gt_ink = genizah_visible_ink_gt(gt)
    _, _, f1 = aligned_prf(hyp, gt_ink)
    cer, _ = cer_pair(hyp, gt_ink)
    return cer, f1


def collect(bench: str) -> tuple:
    """Per-page scores for every baseline and k7 key.

    :param bench: ``religious`` or ``pgp``.
    :type bench: str
    :return: (rows keyed doc_id -> {key: {cer, f1}} plus page meta, list of keys).
    :rtype: tuple
    """
    docs = load_docs(bench)
    out_root = _BENCHES[bench][1]
    tags = discover_tags(out_root, docs)
    keys = list(BASELINES) + [f"{kind}_{t}" for t in tags for kind in BASELINES]
    rows = {}
    for d in docs:
        if len(letters_only(genizah_visible_ink_gt(d["gt"]))) < 50:
            continue
        outdir = out_root / d["doc_id"]
        gt_lines = len([l for l in d["gt"].splitlines() if l.strip()])
        r = {"is_talmud": str(d.get("is_talmud")) == "True",
             "multi": int(d.get("n_columns") or 1) >= 2, "gt_lines": gt_lines, "scores": {}}
        for k in keys:
            f = outdir / f"{k}.txt"
            if f.exists():
                cer, f1 = score_page(f.read_text(errors="replace"), d["gt"])
                r["scores"][k] = {"cer": cer, "f1": f1}
        for t in tags:
            lf = outdir / f"kraken_lines_{t}.json"
            if lf.exists():
                res = json.loads(lf.read_text())
                r[f"n_lines_{t}"] = len(res.get("lines", []))
                r[f"sec_{t}"] = res.get("seconds")
        rows[d["doc_id"]] = r
    return rows, keys, tags


def summarise(rows: dict, keys: list, tags: list, midrash: dict) -> dict:
    """Aggregate headline metrics, slices and paired wins.

    :param rows: Output of :func:`collect`.
    :type rows: dict
    :param keys: Score keys.
    :type keys: list
    :param tags: k7 tags.
    :type tags: list
    :param midrash: doc_id -> MiDRASH Zenodo CER (religious only; may be empty).
    :type midrash: dict
    :return: Summary dict.
    :rtype: dict
    """
    slices = {"ALL": lambda r: True, "talmud": lambda r: r["is_talmud"],
              "multi-col": lambda r: r["multi"], "single-col": lambda r: not r["multi"]}
    out = {"n_pages": len(rows), "keys": {}}
    for k in keys:
        base = "kraken_raw" if k.startswith("kraken_raw") else "kraken_seg"
        ks = {}
        for sname, pred in slices.items():
            sub = [r for r in rows.values() if pred(r) and k in r["scores"]]
            if not sub:
                continue
            cers = [r["scores"][k]["cer"] for r in sub]
            f1s = [r["scores"][k]["f1"] for r in sub]
            ks[sname] = {"n": len(sub), "cer_median": statistics.median(cers),
                         "cer_mean": statistics.mean(cers), "f1_median": statistics.median(f1s)}
        paired = [r for r in rows.values() if k in r["scores"] and base in r["scores"]]
        ks["wins_vs_k4"] = sum(r["scores"][k]["cer"] < r["scores"][base]["cer"] for r in paired)
        ks["losses_vs_k4"] = sum(r["scores"][k]["cer"] > r["scores"][base]["cer"] for r in paired)
        ks["paired_n"] = len(paired)
        if paired:
            ks["median_delta_vs_k4"] = statistics.median(
                r["scores"][k]["cer"] - r["scores"][base]["cer"] for r in paired)
        mid = [(d, r) for d, r in rows.items() if k in r["scores"] and d in midrash]
        if mid:
            ks["midrash_subset_n"] = len(mid)
            ks["cer_median_midrash_subset"] = statistics.median(r["scores"][k]["cer"] for _, r in mid)
            ks["wins_vs_midrash"] = sum(r["scores"][k]["cer"] < midrash[d] for d, r in mid)
        out["keys"][k] = ks
    for t in tags:
        ratios = [r[f"n_lines_{t}"] / r["gt_lines"] for r in rows.values()
                  if f"n_lines_{t}" in r and r["gt_lines"]]
        secs = [r[f"sec_{t}"] for r in rows.values() if r.get(f"sec_{t}") is not None]
        out.setdefault("lines", {})[t] = {
            "lines_per_gt_line_median": statistics.median(ratios) if ratios else None,
            "sec_median": statistics.median(secs) if secs else None,
            "sec_p90": sorted(secs)[int(0.9 * (len(secs) - 1))] if secs else None,
            "n": len(secs)}
    if midrash:
        vals = [midrash[d] for d in rows if d in midrash]
        out["midrash_cer_median"] = statistics.median(vals)
    return out


def markdown(bench: str, s: dict) -> str:
    """Render one benchmark's summary as markdown tables.

    :param bench: Benchmark name.
    :type bench: str
    :param s: Output of :func:`summarise`.
    :type s: dict
    :return: Markdown text.
    :rtype: str
    """
    lines = [f"### {bench} ({s['n_pages']} pages)", "",
             "| key | CER med | CER mean | F1 med | wins/losses vs k4 | Δ med vs k4 | CER med (MiDRASH subset) | wins vs MiDRASH |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, ks in s["keys"].items():
        a = ks.get("ALL")
        if not a:
            continue
        mid = (f"{ks['cer_median_midrash_subset']:.3f} (n={ks['midrash_subset_n']})"
               if "midrash_subset_n" in ks else "—")
        wm = f"{ks['wins_vs_midrash']}/{ks['midrash_subset_n']}" if "wins_vs_midrash" in ks else "—"
        delta = f"{ks['median_delta_vs_k4']:+.3f}" if "median_delta_vs_k4" in ks else "—"
        lines.append(f"| {k} | {a['cer_median']:.3f} | {a['cer_mean']:.3f} | {a['f1_median']:.3f} | "
                     f"{ks['wins_vs_k4']}/{ks['losses_vs_k4']} of {ks['paired_n']} | {delta} | {mid} | {wm} |")
    if "midrash_cer_median" in s:
        lines.append(f"\nMiDRASH Zenodo CER median on the matched subset: {s['midrash_cer_median']:.3f}")
    slice_names = [n for n in ("talmud", "multi-col", "single-col")
                   if any(n in ks for ks in s["keys"].values())]
    if slice_names:
        lines += ["", "| key | " + " | ".join(f"{n} CER med (n)" for n in slice_names) + " |",
                  "|---|" + "---:|" * len(slice_names)]
        for k, ks in s["keys"].items():
            cells = [f"{ks[n]['cer_median']:.3f} ({ks[n]['n']})" if n in ks else "—" for n in slice_names]
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
    if s.get("lines"):
        lines += ["", "| tag | lines / GT line (med) | s/page med | s/page p90 | pages |", "|---|---:|---:|---:|---:|"]
        for t, v in s["lines"].items():
            lpg = f"{v['lines_per_gt_line_median']:.2f}" if v["lines_per_gt_line_median"] is not None else "—"
            lines.append(f"| {t} | {lpg} | {v['sec_median']} | {v['sec_p90']} | {v['n']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Score both benchmarks and write JSON + markdown."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=_REPO / "docs/kraken7_segmenter_ab")
    p.add_argument("--midrash", type=Path,
                   default=_REPO / "docs/kraken7_segmenter_ab/zenodo_vs_kraken_religious.json",
                   help="per-page MiDRASH Zenodo CERs on the religious benchmark")
    args = p.parse_args()
    midrash = {}
    if args.midrash.exists():
        midrash = {r["doc_id"]: r["zenodo_cer"] for r in json.load(open(args.midrash))}
    args.out.mkdir(parents=True, exist_ok=True)
    result, md = {}, []
    for bench in ("religious", "pgp"):
        rows, keys, tags = collect(bench)
        s = summarise(rows, keys, tags, midrash if bench == "religious" else {})
        result[bench] = {"summary": s, "pages": rows}
        md.append(markdown(bench, s))
    (args.out / "ab_scores.json").write_text(json.dumps(result, indent=1))
    (args.out / "ab_tables.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
