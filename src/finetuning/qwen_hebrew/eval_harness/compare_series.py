"""Three-way comparison: <ver>-step<N> vs v19a-1300 (flagship) and v19b-1300.

Generalization of compare_v19c.py: prints (a) benchmark medians, (b) the
known v19b-vs-v19a flip pages with the candidate's score appended, (c) NEW
sub-0.5 pages the candidate introduces vs v19a, and logs full/* metrics to
the W&B run ``<ver>-hard-evals``. Per-page flip counts are decode-noise
class (see the series' methodology note) — medians are the verdict.
"""
import argparse
import csv
import statistics
from pathlib import Path

REPO = Path("/Users/isaac/Documents/GitHub/historical-document-analysis")

ap = argparse.ArgumentParser()
ap.add_argument("--ver", required=True, help="series tag, e.g. v19c or v20a")
ap.add_argument("--step", required=True)
ap.add_argument("--wandb", action="store_true",
                help="append full/* metrics to the <ver>-hard-evals W&B run")
args = ap.parse_args()
C = f"{args.ver}_step{args.step}"
WB = {}


def load(path: Path, key: str, model_map: dict) -> dict:
    """Group long-format score rows by document for the three compared models.

    :param path: Long-format CSV with one row per (model, document).
    :param key: Column naming the document id.
    :param model_map: Model column value -> short label ("a"/"b"/"c").
    :return: doc id -> {label: row} for docs scored by all three models.
    """
    rows = list(csv.DictReader(open(path)))
    by = {}
    for r in rows:
        label = model_map.get(r["model"])
        if label:
            by.setdefault(r[key], {})[label] = r
    return {d: v for d, v in by.items() if len(v) == 3}


def report(name: str, trip: dict, f1key: str) -> None:
    """Print medians, known flip pages and new breaks for one benchmark.

    :param name: Benchmark display name.
    :param trip: Output of :func:`load`.
    :param f1key: Column holding the aligned-F1 score.
    """
    f1 = lambda r: float(r[f1key])  # noqa: E731
    print(f"\n===== {name}: {len(trip)} triple-scored =====")
    for m in ("a", "b", "c"):
        med = statistics.median(f1(v[m]) for v in trip.values())
        label = {"a": "v19a-1300", "b": "v19b-1300", "c": C}[m]
        print(f"  {label:14s} median F1 {med:.3f}")
    flips = [(d, v) for d, v in trip.items() if abs(f1(v["b"]) - f1(v["a"])) > 0.05]
    kept_rescue = dropped_break = 0
    print(f"  --- {len(flips)} known v19b-vs-v19a flip pages, with {C}:")
    for d, v in sorted(flips, key=lambda x: f1(x[1]["b"]) - f1(x[1]["a"])):
        fa, fb, fc = f1(v["a"]), f1(v["b"]), f1(v["c"])
        kind = ("break " if fb < 0.5 <= fa else
                "rescue" if fa < 0.5 <= fb else "      ")
        verdict = ""
        if kind == "rescue" and fc >= 0.5:
            verdict = "KEPT-RESCUE"; kept_rescue += 1
        elif kind == "rescue" and fc < 0.5:
            verdict = "lost-rescue"
        elif kind == "break " and fc >= 0.5:
            verdict = "DROPPED-BREAK"; dropped_break += 1
        elif kind == "break ":
            verdict = "still-broken"
        print(f"    {kind} a={fa:.3f} b={fb:.3f} c={fc:.3f}  {verdict:13s} {d[-40:]}")
    fresh = [(d, f1(v["a"]), f1(v["c"])) for d, v in trip.items()
             if f1(v["a"]) >= 0.5 > f1(v["c"])]
    print(f"  verdict: kept rescues {kept_rescue} | dropped v19b-breaks {dropped_break} | "
          f"NEW {C} breaks vs v19a {len(fresh)} (decode-noise class; medians decide)")
    for d, fa, fc in sorted(fresh, key=lambda x: x[2])[:6]:
        print(f"    NEW-BREAK a={fa:.3f} c={fc:.3f} {d[-40:]}")
    tag = "religious" if "RELIG" in name else "pgp"
    WB[f"full/{tag}_F1_median"] = statistics.median(f1(v["c"]) for v in trip.values())
    WB[f"fullflips/{tag}_kept_rescues"] = kept_rescue
    WB[f"fullflips/{tag}_dropped_breaks"] = dropped_break
    WB[f"fullflips/{tag}_new_breaks"] = len(fresh)


rel = load(REPO / "src/datasets/raw_data/cairo_genizah/evaluations/genizah_religious_v1/religious_scores_long.csv",
           "doc_id", {"v19a_step1300": "a", "v19b_step1300": "b", C: "c"})
pgp = load(REPO / "src/datasets/evaluations/transcription_results/paper_table/genizah_offline_long.csv",
           "fragment_id", {"qwen3_vl_8b_heb_v19a_step1300": "a",
                           "qwen3_vl_8b_heb_v19b_step1300": "b",
                           f"qwen3_vl_8b_heb_{C}": "c"})
print(f"NOTE: {C} vs step-1300 v19 models — different schedule/data; "
      "medians are the comparison, per-page flips carry the decode-noise caveat.")
report("RELIGIOUS-140 (GT rev 1.1)", rel, "aligned_f1")
report("PGP-131 (frozen)", pgp, "aligned_f1")

if args.wandb and WB:
    import os
    import dotenv
    dotenv.load_dotenv(REPO / ".env")
    import wandb
    WB.update({"ref/v19a_religious_F1": 0.816, "ref/v19b_religious_F1": 0.810,
               "ref/v19a_pgp_F1": 0.862, "ref/v19b_pgp_F1": 0.868})
    wandb.init(project="qwen-hebrew-finetune", id=f"{args.ver}-hard-evals",
               name=f"{args.ver}_hard_evals", resume="allow",
               settings=wandb.Settings(silent=True))
    wandb.log(WB, step=int(args.step))
    wandb.finish()
    print(f"logged full metrics to W&B {args.ver}_hard_evals @ step {args.step}")
