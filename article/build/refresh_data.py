"""Refresh the result-driven parts of article/build/data.json from pipeline CSVs.

Updates (or adds) the ``genizah`` and ``genizah_script`` entries for the
requested models from the offline Genizah scorer's outputs, leaving every
other key of the dataset untouched. Before writing, it recomputes an existing
model's entries and checks they match the stored ones, so a schema drift in
the CSVs cannot silently corrupt the figures.

Usage: python3 article/build/refresh_data.py [model_key ...]
       (default: qwen3_vl_8b_heb_v20a_step1800)
"""
import csv
import json
import pathlib
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
PT = REPO / "src/datasets/evaluations/transcription_results/paper_table"
SUMMARY = PT / "genizah_offline_summary.csv"
LONG = PT / "genizah_offline_long.csv"
DATA = REPO / "article/build/data.json"
CHECK_MODEL = "qwen3_vl_8b_heb_v19a_step1300"
MODES = {"substantive": "substantive", "abstained": "abstained",
         "loop_collapse": "loop", "hallucinated": "hallucinated"}


def genizah_entry(row: dict) -> dict:
    """Map one summary-CSV row to the article's ``genizah`` schema.

    :param row: Row of genizah_offline_summary.csv (rates already 0-1).
    :return: Entry with the keys the article's charts read.
    """
    return {"model": row["model"], "n": int(row["n"]),
            "abstained": float(row["pct_abstained"]),
            "loop": float(row["pct_loop_collapse"]),
            "hallucinated": float(row["pct_hallucinated"]),
            "substantive": float(row["pct_substantive"]),
            "ngramP": float(row["ngram_precision_median"]),
            "f1": float(row["aligned_f1_median"]),
            "cer_sub": float(row["cer_median_substantive"]),
            "n_sub": int(row["n_substantive"])}


def script_entries(long_rows: list, model: str) -> list:
    """Per-script-bucket behaviour and accuracy for one model.

    :param long_rows: Rows of genizah_offline_long.csv.
    :param model: Model key to aggregate.
    :return: One entry per script bucket (rates = share of fragments in that
        failure mode; ngramP / f1 = medians over the bucket).
    """
    by = {}
    for r in long_rows:
        if r["model"] == model:
            by.setdefault(r["script_bucket"], []).append(r)
    out = []
    for bucket in sorted(by):
        rs = by[bucket]
        e = {"model": model, "bucket": bucket, "n": len(rs)}
        for mode, key in MODES.items():
            e[key] = round(sum(r["failure_mode"] == mode for r in rs) / len(rs), 3)
        e["ngramP"] = round(statistics.median(float(r["ngram_precision"]) for r in rs), 4)
        e["f1"] = round(statistics.median(float(r["aligned_f1"]) for r in rs), 4)
        out.append(e)
    return out


def main() -> None:
    """Refresh data.json for the requested models after a consistency check."""
    models = sys.argv[1:] or ["qwen3_vl_8b_heb_v20a_step1800"]
    summary = {r["model"]: r for r in csv.DictReader(open(SUMMARY))}
    long_rows = list(csv.DictReader(open(LONG)))
    data = json.load(open(DATA))

    stored = {(e["bucket"]): e for e in data["genizah_script"] if e["model"] == CHECK_MODEL}
    for e in script_entries(long_rows, CHECK_MODEL):
        s = stored[e["bucket"]]
        for k in ("n", "substantive", "hallucinated", "ngramP", "f1"):
            assert abs(float(s[k]) - float(e[k])) < 1e-3, (CHECK_MODEL, e["bucket"], k, s[k], e[k])
    g_stored = next(e for e in data["genizah"] if e["model"] == CHECK_MODEL)
    assert abs(g_stored["f1"] - genizah_entry(summary[CHECK_MODEL])["f1"]) < 1e-4
    print(f"consistency check vs stored {CHECK_MODEL}: OK")

    for m in models:
        assert m in summary, f"{m} not in {SUMMARY.name}"
        data["genizah"] = [e for e in data["genizah"] if e["model"] != m] + [genizah_entry(summary[m])]
        data["genizah_script"] = ([e for e in data["genizah_script"] if e["model"] != m]
                                  + script_entries(long_rows, m))
        g = genizah_entry(summary[m])
        print(f"  {m}: f1 {g['f1']:.4f}  ngramP {g['ngramP']:.4f}  substantive {g['substantive']:.3f}  "
              f"buckets {len(script_entries(long_rows, m))}")
    json.dump(data, open(DATA, "w"), ensure_ascii=False, indent=0)
    print(f"wrote {DATA.relative_to(REPO)}")


if __name__ == "__main__":
    main()
