"""Three-way comparison: v19c-step<N> vs v19a-1300 and v19b-1300.

Prints (a) benchmark medians, (b) the known flip pages (where v19b differed
from v19a) with v19c's score appended — the decisive view for the
keeps-rescues / drops-collapses question, (c) NEW flips v19c introduces vs
v19a. Caveat printed when comparing a step-700 model against step-1300 ones.
"""
import argparse
import csv
import statistics
from pathlib import Path

REPO = Path("/Users/isaac/Documents/GitHub/historical-document-analysis")

ap = argparse.ArgumentParser()
ap.add_argument("--step", required=True)
ap.add_argument("--wandb", action="store_true",
                help="append full/* metrics to the v19c_hard_evals W&B run")
args = ap.parse_args()
C = f"v19c_step{args.step}"
WB = {}


def load(path, key, model_map):
    rows = list(csv.DictReader(open(path)))
    by = {}
    for r in rows:
        label = model_map.get(r["model"])
        if label:
            by.setdefault(r[key], {})[label] = r
    return {d: v for d, v in by.items() if len(v) == 3}


def report(name, trip, f1key):
    f1 = lambda r: float(r[f1key])
    print(f"\n===== {name}: {len(trip)} triple-scored =====")
    for m in ("a", "b", "c"):
        med = statistics.median(f1(v[m]) for v in trip.values())
        label = {"a": "v19a-1300", "b": "v19b-1300", "c": C}[m]
        print(f"  {label:14s} median F1 {med:.3f}")
    flips = [(d, v) for d, v in trip.items() if abs(f1(v["b"]) - f1(v["a"])) > 0.05]
    kept_rescue = dropped_break = new_break = 0
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
    new_break = len(fresh)
    print(f"  verdict: kept rescues {kept_rescue} | dropped v19b-breaks {dropped_break} | "
          f"NEW v19c breaks vs v19a {new_break}")
    for d, fa, fc in sorted(fresh, key=lambda x: x[2])[:6]:
        print(f"    NEW-BREAK a={fa:.3f} c={fc:.3f} {d[-40:]}")
    tag = "religious" if "RELIG" in name else "pgp"
    WB[f"full/{tag}_F1_median"] = statistics.median(f1(v["c"]) for v in trip.values())
    WB[f"fullflips/{tag}_kept_rescues"] = kept_rescue
    WB[f"fullflips/{tag}_dropped_breaks"] = dropped_break
    WB[f"fullflips/{tag}_new_breaks"] = new_break


rel = load(REPO / "src/datasets/raw_data/cairo_genizah/evaluations/genizah_religious_v1/religious_scores_long.csv",
           "doc_id", {"v19a_step1300": "a", "v19b_step1300": "b", C: "c"})
pgp = load(REPO / "src/datasets/evaluations/transcription_results/paper_table/genizah_offline_long.csv",
           "fragment_id", {"qwen3_vl_8b_heb_v19a_step1300": "a",
                           "qwen3_vl_8b_heb_v19b_step1300": "b",
                           f"qwen3_vl_8b_heb_{C}": "c"})
if args.step != "1300":
    print(f"NOTE: {C} compared against step-1300 models — schedule point differs; "
          "flip-page directionality is the meaningful signal, medians carry a caveat.")
report("RELIGIOUS-140 (GT rev 1.1)", rel, "aligned_f1")
report("PGP-131 (frozen)", pgp, "aligned_f1")

if args.wandb and WB:
    import os
    import dotenv
    dotenv.load_dotenv(REPO / ".env")
    import wandb
    WB.update({"ref/v19a_religious_F1": 0.816, "ref/v19b_religious_F1": 0.810,
               "ref/v19a_pgp_F1": 0.862, "ref/v19b_pgp_F1": 0.868})
    wandb.init(project="qwen-hebrew-finetune", id="v19c-hard-evals",
               name="v19c_hard_evals", resume="allow",
               settings=wandb.Settings(silent=True))
    wandb.log(WB, step=int(args.step))
    wandb.finish()
    print(f"logged full metrics to W&B v19c_hard_evals @ step {args.step}")
