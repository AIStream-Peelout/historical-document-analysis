"""Merge per-run Genizah fragment metrics into one paper-ready benchmark table.

The 150-fragment ``genizah_test_v1`` benchmark was collected across several
W&B runs (the run interrupted by the 2026-07-28 host-permission outage, the
resumed run, and the earlier 2-fragment pilots).  Re-running paid models to
force a single run would waste API spend, so this script merges instead —
the fragment-eval sibling of ``build_talmud_paper_table.py``.

It pulls every ``batch_*_metrics`` table from each source run, dedups by
(model, fragment) with the LATEST source run winning, and produces:

* ``genizah_paper_long.csv``    — one row per (fragment, model) with all metrics
* ``genizah_paper_summary.csv`` — per model: n, coverage%, mean/median CER
  (visible-ink and raw), collapse rates
* a new W&B run logging both as tables plus a CSV artifact, so the paper
  links ONE run.

Usage (from src/datasets/evaluations):
    PYTHONPATH=<repo> python helper_eval_scripts/build_genizah_paper_table.py \\
        [--runs ID,ID,...] [--no-wandb] [--out-dir talmud_results/...]
"""

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import dotenv

dotenv.load_dotenv()

import wandb  # noqa: E402

PROJECT = "igodfried/cairo-genizah-transcription"

# Oldest → newest; later runs override earlier ones for the same
# (model, fragment).  Run IDs are used rather than display names because the
# fragment harness auto-names runs "batch-N-of-M-<timestamp>".
SOURCE_RUN_IDS = [
    "hn9vq3xp",   # 2026-07-28 first full attempt — died at doc 15 (host TCC outage)
    "gc6gr2dn",   # 2026-07-29 resumed full run (reuse-cached, no --strict)
]

BENCHMARK_SIZE = 150          # frozen genizah_test_v1 fragment count
COLLAPSE_MILD = 0.3           # CER above this = failed transcription
COLLAPSE_HARD = 1.0           # CER above this = decoder collapse / over-generation

# Metrics-table column names (see fragment_evals.log_to_wandb).
_NUMERIC_COLS = [
    "cer", "cer_lenient", "cer_ink", "wer", "wer_lenient",
    "char_count_ratio", "similarity", "char_count", "gt_char_count",
    "char_diff", "processing_time_sec",
]

# Display names logged by the harness → canonical model keys for the paper.
_MODEL_RENAMES = {
    "Gemini Flash": "gemini_flash",
    "Gemini Pro": "gemini_pro",
    "Vision OCR → LLM extraction": "vision_ocr_seg",
    "Kraken → LLM extraction": "kraken_seg",
    "Consensus": "consensus",
}


def canonical_model(name: str) -> str:
    """Map a logged model display name to a stable key.

    :param name: Model name as written into the metrics table.
    :type name: str
    :return: Canonical snake_case model key.
    :rtype: str
    """
    if name in _MODEL_RENAMES:
        return _MODEL_RENAMES[name]
    # "Claude (claude-opus-4-8)" / "ChatGPT (gpt-5.6-sol)" → inner model id
    if "(" in name and name.endswith(")"):
        name = name[name.index("(") + 1:-1]
    return name.strip().lower().replace("-", "_").replace(".", "_").replace("/", "_")


def extract_records(run, source: str) -> list:
    """Convert a run's per-document history into long-format records.

    The fragment harness logs the ``batch_*_metrics`` TABLE only after the
    whole batch finishes, so an interrupted run has no table — but it logs
    per-document scalars (``<model>/<metric>``) as each document completes.
    Reading history therefore recovers every document a run actually
    evaluated, crashed or not.

    :param run: W&B run object.
    :param source: Source run identifier for provenance.
    :type source: str
    :return: One dict per (fragment, model).
    :rtype: list
    """
    records = {}
    for row in run.scan_history():
        doc_id = row.get("doc_id")
        if not doc_id:
            continue
        for key, value in row.items():
            if "/" not in key or key.startswith("_") or value is None:
                continue
            model, _, metric = key.partition("/")
            if metric not in _NUMERIC_COLS and metric not in ("processing_time",):
                continue
            rec = records.setdefault(
                (model, doc_id),
                {"fragment_id": doc_id, "model": model, "source_run": source},
            )
            rec[metric] = value
    return list(records.values())


def summarize(long_rows: list) -> list:
    """Aggregate per-model statistics for the paper table.

    Reports visible-ink CER (the headline Genizah metric) alongside raw CER,
    plus coverage against the frozen benchmark size so models that failed or
    refused on some fragments cannot benefit from the omission.

    :param long_rows: Merged per-(fragment, model) records.
    :type long_rows: list
    :return: One summary dict per model.
    :rtype: list
    """
    groups = defaultdict(list)
    for r in long_rows:
        groups[r["model"]].append(r)

    summary = []
    for model, rs in sorted(groups.items()):
        ink = [r["cer_ink"] for r in rs if r.get("cer_ink") is not None]
        raw = [r["cer"] for r in rs if r.get("cer") is not None]
        base = ink or raw
        summary.append({
            "model": model,
            "n_fragments": len(rs),
            "coverage_pct": round(len(rs) / BENCHMARK_SIZE, 4),
            "cer_ink_mean": round(sum(ink) / len(ink), 4) if ink else None,
            "cer_ink_median": round(statistics.median(ink), 4) if ink else None,
            "cer_raw_mean": round(sum(raw) / len(raw), 4) if raw else None,
            "cer_raw_median": round(statistics.median(raw), 4) if raw else None,
            f"pct_gt_{COLLAPSE_MILD}": (
                round(sum(v > COLLAPSE_MILD for v in base) / len(base), 4) if base else None),
            f"pct_gt_{COLLAPSE_HARD}": (
                round(sum(v > COLLAPSE_HARD for v in base) / len(base), 4) if base else None),
            "source_runs": ";".join(sorted({r["source_run"] for r in rs})),
        })
    return summary


def main() -> None:
    """Merge every source run and publish the combined paper table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                        default=SOURCE_RUN_IDS,
                        help="W&B run IDs, oldest first (later wins on conflict)")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("transcription_results/paper_table"))
    parser.add_argument("--wandb-run-name", default="genizah-paper-full-v1")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    api = wandb.Api()
    merged = {}
    for run_id in args.runs:
        run = api.run(f"{PROJECT}/{run_id}")
        recs = extract_records(run, run_id)
        frags = {r["fragment_id"] for r in recs}
        for r in recs:
            merged[(r["model"], r["fragment_id"])] = r
        print(f"\n{run_id} ({run.name}, state={run.state}): "
              f"{len(recs)} records over {len(frags)} fragments")

    long_rows = sorted(merged.values(), key=lambda r: (r["model"], r["fragment_id"]))
    if not long_rows:
        print("No records found — are the source runs still uploading?")
        return
    summary_rows = summarize(long_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    long_csv = args.out_dir / "genizah_paper_long.csv"
    summary_csv = args.out_dir / "genizah_paper_summary.csv"
    fieldnames = sorted({k for r in long_rows for k in r})
    with open(long_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(long_rows)
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nWrote {long_csv} ({len(long_rows)} rows) and {summary_csv} "
          f"({len(summary_rows)} models)")

    print("\nPer-model coverage (dedup check):")
    for s in summary_rows:
        print(f"   {s['model']:34s} n={s['n_fragments']:3d} "
              f"({s['coverage_pct']:.0%})  ink_mean={s['cer_ink_mean']}  "
              f"src={s['source_runs']}")

    if args.no_wandb:
        return

    entity, project = PROJECT.split("/")
    run = wandb.init(
        project=project, entity=entity, name=args.wandb_run_name,
        job_type="paper-table",
        config={
            "source_runs": args.runs,
            "benchmark": "genizah_test_v1",
            "benchmark_size": BENCHMARK_SIZE,
            "dedup_rule": "latest source run wins per (model, fragment)",
            "headline_metric": "cer_ink (visible-ink CER)",
            "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        tags=["paper", "merged-benchmark", "genizah"],
    )
    wandb.log({
        "paper/genizah_long": wandb.Table(
            columns=fieldnames,
            data=[[r.get(k) for k in fieldnames] for r in long_rows],
        ),
        "paper/genizah_summary": wandb.Table(
            columns=list(summary_rows[0].keys()),
            data=[list(r.values()) for r in summary_rows],
        ),
    })
    art = wandb.Artifact("genizah_paper_table", type="benchmark-table")
    art.add_file(str(long_csv))
    art.add_file(str(summary_csv))
    run.log_artifact(art)
    print(f"W&B: {run.url}")
    wandb.finish()


if __name__ == "__main__":
    main()
