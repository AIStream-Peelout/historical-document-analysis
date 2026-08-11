"""Offline LLM-judge error classification over saved benchmark outputs.

Runs the same error-type taxonomy as the in-run judge, but AFTER a benchmark
run, reading model outputs from ``transcription_raw_outputs/<doc>/<model>.txt``
and ground truth from ``<doc>/ground_truth.txt``.  Designed for a LOCAL judge
served by LM Studio so the judge costs no API tokens and adds no model-swap
churn to the paid benchmark run itself.

Usage (from src/datasets/evaluations):
    PYTHONPATH=<repo> python helper_eval_scripts/offline_error_analysis.py \
        --outputs-dir ./transcription_raw_outputs \
        --judge-model qwen/qwen3.6-35b-a3b \
        --judge-base-url http://localhost:1234/v1 \
        --wandb-run-name genizah-test-v1-error-analysis \
        [--models qwen3_vl_8b_heb_v17_step800,claude_opus_4_8,...] \
        [--limit-docs N] [--no-wandb]

Only one LM Studio request runs at a time (module lock in lms_transcriber),
so this is safe to run whenever the benchmark harness is idle.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import dotenv

dotenv.load_dotenv()

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO))

from src.datasets.evaluations.error_classifier import (  # noqa: E402
    TABLE_COLUMNS,
    aggregate_error_stats,
    classification_table_rows,
    classify_transcription_errors,
)

# Files in a doc dir that are never model outputs to judge.
_SKIP_STEMS = {"ground_truth", "consensus"}
_SKIP_SUFFIXES = ("_raw", "_flat")


def discover_model_outputs(doc_dir: Path, models: list) -> list:
    """List (model_key, path) pairs to judge in one document directory.

    :param doc_dir: Directory containing per-model ``.txt`` outputs.
    :type doc_dir: Path
    :param models: Explicit model keys to include (empty = all discovered).
    :type models: list
    :return: Sorted list of (model_key, Path) tuples.
    :rtype: list
    """
    out = []
    for f in sorted(doc_dir.glob("*.txt")):
        stem = f.stem
        if stem in _SKIP_STEMS or stem.endswith(_SKIP_SUFFIXES):
            continue
        if models and stem not in models:
            continue
        out.append((stem, f))
    return out


async def main() -> None:
    """Judge every saved model output against its ground truth and aggregate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=Path, default=Path("./transcription_raw_outputs"))
    parser.add_argument("--judge-model", required=True,
                        help="LM Studio model ID (local) or Gemini model name")
    parser.add_argument("--judge-base-url", default=None,
                        help="LM Studio base URL; leave unset to use Gemini")
    parser.add_argument("--models",
                        type=lambda s: [m.strip() for m in s.split(",") if m.strip()],
                        default=[],
                        help="Model keys to judge (default: every output file found)")
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--docs-from", type=Path, default=None,
                        help="Benchmark JSON; judge only its doc_ids (skips the "
                             "Talmud page directories that share this outputs dir)")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-project", default="cairo-genizah-transcription")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Also dump classifications to this JSON file")
    args = parser.parse_args()

    doc_dirs = sorted(
        d for d in args.outputs_dir.iterdir()
        if d.is_dir() and (d / "ground_truth.txt").exists()
    )
    if args.docs_from:
        wanted = {d["doc_id"] for d in json.load(open(args.docs_from))["docs"]}
        doc_dirs = [d for d in doc_dirs if d.name in wanted]
    if args.limit_docs:
        doc_dirs = doc_dirs[:args.limit_docs]
    print(f"Judging {len(doc_dirs)} docs with {args.judge_model}"
          + (f" @ {args.judge_base_url}" if args.judge_base_url else " (Gemini)"))

    classifications = []
    t0 = time.time()
    for i, doc_dir in enumerate(doc_dirs, 1):
        gt = (doc_dir / "ground_truth.txt").read_text(encoding="utf-8")
        targets = discover_model_outputs(doc_dir, args.models)
        print(f"[{i}/{len(doc_dirs)}] {doc_dir.name}: {len(targets)} outputs")
        for model_key, path in targets:
            hyp = path.read_text(encoding="utf-8").strip()
            if not hyp:
                continue
            result = await classify_transcription_errors(
                hyp, gt,
                judge_model=args.judge_model,
                judge_base_url=args.judge_base_url,
            )
            if result:
                classifications.append(
                    {"doc_id": doc_dir.name, "model": model_key, **result}
                )
                types = [e["type"] for e in result["errors"]] or ["none"]
                print(f"    [{model_key}] {result['overall_quality']}: {', '.join(types)}")
            else:
                print(f"    [{model_key}] ✗ judge returned nothing/unparseable")

    elapsed = time.time() - t0
    print(f"\n{len(classifications)} classifications in {elapsed/60:.1f} min")

    stats = aggregate_error_stats(classifications)
    for key in sorted(k for k in stats if k.endswith("/pct_docs")):
        print(f"  {key.removeprefix('error_types/').removesuffix('/pct_docs')}: {stats[key]:.0%}")

    if args.output_json:
        args.output_json.write_text(
            json.dumps(classifications, ensure_ascii=False, indent=1)
        )
        print(f"Wrote {args.output_json}")

    if not args.no_wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"error-analysis-{time.strftime('%Y%m%d-%H%M%S')}",
            config={
                "judge_model": args.judge_model,
                "judge_base_url": args.judge_base_url,
                "outputs_dir": str(args.outputs_dir),
                "num_docs": len(doc_dirs),
                "models_filter": args.models or "all",
            },
            tags=["error-analysis", "offline-judge"],
        )
        wandb.log({
            "error_classification": wandb.Table(
                columns=TABLE_COLUMNS,
                data=classification_table_rows(classifications),
            )
        })
        for key, value in stats.items():
            wandb.summary[key] = value
        print(f"W&B: {run.url}")
        wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())
