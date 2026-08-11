"""Merge per-run Talmud section tables into one paper-ready benchmark table.

The full Talmud benchmark is the union of several W&B runs (closed models
July 13, v1.5, v1.6 checkpoints, the GPT-5.2/v1.7/v1.6-step1100 fill run).
This script pulls the ``transcriptions/{gemara,rashi,tosafot}`` tables from
each source run, extracts every ``*_cer_strict``/``*_cer_lenient`` column,
and produces:

* ``talmud_paper_long.csv``  — one row per (doc, section, model): CERs + source run
* ``talmud_paper_summary.csv`` — per (model, section): n, mean, median, collapse rates
* a new W&B run logging both as tables, so the paper links ONE run.

Dedup rule: when a model appears in several source runs (e.g. base qwen),
the LATEST run in SOURCE_RUNS order wins — order them oldest → newest.

Usage (from src/datasets/evaluations):
    PYTHONPATH=<repo> python helper_eval_scripts/build_talmud_paper_table.py \
        [--no-wandb] [--out-dir talmud_results/paper_table]
"""

import argparse
import csv
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import dotenv

dotenv.load_dotenv()

import wandb  # noqa: E402

PROJECT = "igodfried/cairo-genizah-transcription"
SECTIONS = ("gemara", "rashi", "tosafot")

# Oldest → newest; later runs override earlier ones for the same (model, doc).
SOURCE_RUNS = [
    "talmud-20260713-133127",          # Gemini Flash/Pro, Claude Opus+Sonnet, base qwen, OCR pipelines
    "talmud-v15-before-after",         # v1.5 + base qwen
    "talmud-v16-step600",              # v1.6 candidate
    "talmud-v16-step1000",             # v1.6 candidate (official step-1000 numbers)
    "talmud-fill-gpt52-v17-v16step1100",  # GPT-5.2 + v1.7-step800 + v1.6-step1100
    "talmud-fill-claude-37pages",      # Opus+Sonnet on the 37 July-13-missing pages
    "talmud-fill-gemini-part1",        # Flash+Pro fill (1 page before host restart)
    "talmud-fill-gemini-remaining",    # Flash+Pro fill (rest; partial until it finishes)
    "talmud-sol-sample10",             # GPT-5.6 Sol pages 1-10 (cost-bounded sample)
    "talmud-sol-sample11-20",          # GPT-5.6 Sol pages 11-20
]

# Column-prefix → canonical model key
_PREFIX_RENAMES = {"pro": "gemini_pro", "flash": "gemini_flash"}

COLLAPSE_MILD = 0.3   # CER above this = failed page (loop/omission territory)
COLLAPSE_HARD = 1.0   # CER above this = decoder collapse

# Pages with non-empty GT per section, accumulated during extraction — the
# denominator for each model's coverage%. Persistent API failure is reported
# as (low) coverage, a benchmark finding, not silently dropped.
GT_PAGES = defaultdict(set)


def _resolve_runs(api: "wandb.Api") -> list:
    """Return finished W&B run objects for SOURCE_RUNS, keeping list order.

    When several runs share a display name (crashed retries), the latest
    FINISHED one wins.

    :param api: Authenticated W&B public API client.
    :type api: wandb.Api
    :return: List of (name, run) tuples in SOURCE_RUNS order.
    :rtype: list
    """
    # "crashed" runs are acceptable sources: tables are logged incrementally
    # per page, so a run killed after its last page still carries full data
    # (the 2026-07-27 host restart left several complete runs marked crashed).
    by_name = {}
    for run in api.runs(PROJECT, per_page=300):
        if run.name in SOURCE_RUNS and run.state in ("finished", "crashed"):
            prev = by_name.get(run.name)
            if prev is None or run.created_at > prev.created_at:
                by_name[run.name] = run
    resolved = []
    for name in SOURCE_RUNS:
        if name in by_name:
            resolved.append((name, by_name[name]))
        else:
            print(f"⚠️  source run not found/finished yet — skipping: {name}")
    return resolved


def _latest_section_artifacts(run) -> dict:
    """Map each section to the highest-version logged table artifact.

    Run-table artifact names carry a content hash
    (``run-<id>-transcriptions<section>-<hash>:vN``), so they must be
    resolved by scanning ``logged_artifacts()`` rather than by name.

    :param run: W&B run object.
    :return: Dict of section → artifact (latest version only).
    :rtype: dict
    """
    latest = {}
    for art in run.logged_artifacts():
        if art.type != "run_table":
            continue
        for section in SECTIONS:
            if f"transcriptions{section}-" in art.name:
                version = int(art.name.rsplit(":v", 1)[1])
                prev = latest.get(section)
                if prev is None or version > prev[0]:
                    latest[section] = (version, art)
    return {s: art for s, (_v, art) in latest.items()}


def _fetch_section_table(art, section: str):
    """Load (columns, rows) from a run-table artifact.

    :param art: W&B artifact holding the section table.
    :param section: Section key (gemara/rashi/tosafot).
    :type section: str
    :return: (columns, rows) or (None, None) on failure.
    :rtype: tuple
    """
    import json
    try:
        table = art.get(f"transcriptions/{section}")
        if table is not None:
            return list(table.columns), list(table.data)
        path = art.download()
        table_files = list(Path(path).rglob("*.table.json"))
        payload = json.load(open(table_files[0]))
        return payload["columns"], payload["data"]
    except Exception as exc:
        print(f"   ⚠️  {section}: {type(exc).__name__}: {str(exc)[:120]}")
        return None, None


def extract_cer_rows(columns: list, rows: list, section: str, source: str) -> list:
    """Convert one section table into long-format CER records.

    :param columns: Table column names.
    :type columns: list
    :param rows: Table data rows.
    :type rows: list
    :param section: Section key for the emitted records.
    :type section: str
    :param source: Source run name for provenance.
    :type source: str
    :return: Dicts with doc_id/section/model/cer_strict/cer_lenient/source.
    :rtype: list
    """
    idx = {c: i for i, c in enumerate(columns)}
    doc_i = idx.get("doc_id")
    gt_i = idx.get("gt_chars")
    gt_text_i = idx.get("ground_truth")
    out = []
    models = sorted(
        c[: -len("_cer_strict")] for c in columns if c.endswith("_cer_strict")
    )
    for row in rows:
        doc_id = row[doc_i]
        gt_present = bool((row[gt_text_i] or "").strip()) if gt_text_i is not None else True
        if gt_present:
            GT_PAGES[section].add(doc_id)
        for prefix in models:
            strict = row[idx[f"{prefix}_cer_strict"]]
            lenient_col = idx.get(f"{prefix}_cer_lenient")
            lenient = row[lenient_col] if lenient_col is not None else None
            if strict is None:
                continue
            out.append({
                "doc_id": doc_id,
                "section": section,
                "model": _PREFIX_RENAMES.get(prefix, prefix),
                "cer_strict": strict,
                "cer_lenient": lenient,
                "gt_chars": row[gt_i] if gt_i is not None else None,
                "source_run": source,
            })
    return out


def main() -> None:
    """Build and (optionally) publish the merged Talmud paper table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("talmud_results/paper_table"))
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-run-name", default="talmud-paper-full-v1")
    args = parser.parse_args()

    api = wandb.Api()
    sources = _resolve_runs(api)
    print("Merging from:", [name for name, _ in sources])

    # later sources override earlier ones per (model, section, doc)
    merged = {}
    provenance_counts = defaultdict(int)
    for name, run in sources:
        section_arts = _latest_section_artifacts(run)
        for section in SECTIONS:
            art = section_arts.get(section)
            if art is None:
                print(f"   ⚠️  {name}/{section}: no table artifact")
                continue
            cols, rows = _fetch_section_table(art, section)
            if not cols:
                continue
            recs = extract_cer_rows(cols, rows, section, name)
            for r in recs:
                merged[(r["model"], r["section"], r["doc_id"])] = r
            print(f"   {name}/{section}: {len(recs)} records")

    long_rows = sorted(
        merged.values(), key=lambda r: (r["model"], r["section"], r["doc_id"])
    )
    for r in long_rows:
        provenance_counts[(r["model"], r["source_run"])] += 1

    # ── Summary per (model, section) ─────────────────────────────────────────
    groups = defaultdict(list)
    for r in long_rows:
        groups[(r["model"], r["section"])].append(r)
    summary_rows = []
    for (model, section), rs in sorted(groups.items()):
        strict = [r["cer_strict"] for r in rs]
        lenient = [r["cer_lenient"] for r in rs if r["cer_lenient"] is not None]
        denom = len(GT_PAGES.get(section) or ()) or None
        summary_rows.append({
            "model": model,
            "section": section,
            "n_pages": len(strict),
            "coverage_pct": round(len(strict) / denom, 4) if denom else None,
            "cer_strict_mean": round(sum(strict) / len(strict), 4),
            "cer_strict_median": round(statistics.median(strict), 4),
            "cer_lenient_mean": round(sum(lenient) / len(lenient), 4) if lenient else None,
            f"pct_gt_{COLLAPSE_MILD}": round(
                sum(s > COLLAPSE_MILD for s in strict) / len(strict), 4),
            f"pct_gt_{COLLAPSE_HARD}": round(
                sum(s > COLLAPSE_HARD for s in strict) / len(strict), 4),
            "source_runs": ";".join(sorted({r["source_run"] for r in rs})),
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    long_csv = args.out_dir / "talmud_paper_long.csv"
    summary_csv = args.out_dir / "talmud_paper_summary.csv"
    with open(long_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
        w.writeheader()
        w.writerows(long_rows)
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nWrote {long_csv} ({len(long_rows)} rows) and {summary_csv} "
          f"({len(summary_rows)} rows)")

    print("\nPer-model page counts by source (dedup check):")
    for (model, src), n in sorted(provenance_counts.items()):
        print(f"   {model:36s} {src:36s} {n}")

    if args.no_wandb:
        return

    run = wandb.init(
        project=PROJECT.split("/")[1],
        entity=PROJECT.split("/")[0],
        name=args.wandb_run_name,
        job_type="paper-table",
        config={
            "source_runs": [name for name, _ in sources],
            "dedup_rule": "latest source run wins per (model, section, doc)",
            "collapse_thresholds": [COLLAPSE_MILD, COLLAPSE_HARD],
            "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        tags=["paper", "merged-benchmark", "talmud"],
    )
    wandb.log({
        "paper/talmud_long": wandb.Table(
            columns=list(long_rows[0].keys()),
            data=[list(r.values()) for r in long_rows],
        ),
        "paper/talmud_summary": wandb.Table(
            columns=list(summary_rows[0].keys()),
            data=[list(r.values()) for r in summary_rows],
        ),
    })
    art = wandb.Artifact("talmud_paper_table", type="benchmark-table")
    art.add_file(str(long_csv))
    art.add_file(str(summary_csv))
    run.log_artifact(art)
    print(f"W&B: {run.url}")
    wandb.finish()


if __name__ == "__main__":
    main()
