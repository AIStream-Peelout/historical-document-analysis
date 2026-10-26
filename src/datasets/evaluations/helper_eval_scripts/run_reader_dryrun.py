# File name: run_reader_dryrun.py
# Date: 10/25/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Dry run of the production page reader on the prompt-ablation panel, scored next to the ablation's arms.

Re-reads the 60-image panel of ``build_prompt_ablation_panel.py`` with :func:`page_reader.read_page` as
production calls it (script-neutral prompt, 1,200-token budget, ``finish_reason`` and token counts kept,
``length`` finishes cut by the loop guard and retried once at temperature 0, blank images gated by ink
density before the model) and scores the result the way ``score_prompt_ablation.py`` scores an arm:
covered documents with the covered-only scorer, the uncovered classes by behaviour (Arabic letters on
Hebrew-script sides, blank images, loops, finish reasons). The ablation's own arms of the same model are
scored on the same images next to it — ``trained`` at 2,500 tokens (the production setting before this
change), ``trained`` at 1,200 and ``noscript`` at 1,200 (the same prompt and budget without gate, guard
or retry) — with paired comparisons on the covered documents, so the effect of the change is read off
one report. A second, derived arm (``reader-raw``) scores the reader's rows with the un-cut text, which
isolates the loop guard.

Outputs (``--out-dir``, default ``<benchmark>/outputs_reader_dryrun``): ``<model>__reader__mt<N>.jsonl``
(one row per image in the ablation layout plus the reader's fields, :data:`READER_FIELDS`),
``<model>__reader-raw__mt<N>.jsonl``, ``report.md``, ``summary.json``, ``images.csv`` and
``documents.csv``. The run is resumable: an image with a stored read is skipped, a failed request is
retried on the next run.

Shared machine: one LM Studio consumer at a time. The run yields while any other loaded model is busy
(``lms ps``), waits while local free disk is under ``--min-free-gb`` and stops when LM Studio no longer
serves the model.

Usage::

    .venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.run_reader_dryrun \\
        --benchmark <benchmark> [--model qwen3-vl-8b-heb-v25a-step1200] [--panel <benchmark>/panels/prompt_ablation_v1.jsonl] \\
        [--limit N] [--no-retry] [--no-blank-gate] [--score-only]
"""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.datasets.evaluations.helper_eval_scripts import run_prompt_ablation as rpa
from src.datasets.evaluations.helper_eval_scripts import score_arabic_benchmark as sab
from src.datasets.evaluations.helper_eval_scripts import score_arabic_covered as sac
from src.datasets.evaluations.helper_eval_scripts import score_prompt_ablation as spa
from src.datasets.evaluations.helper_eval_scripts.build_prompt_ablation_panel import BLANK, CLASSES, COVERED, load_panel
from src.finetuning.qwen_hebrew.eval_harness.v22_task_eval import free_gb, model_served
from src.models.ocr import page_reader as pr

PROMPT_TAG = "reader"                 # the arm's prompt name: <model>__reader__mt<N>.jsonl
RAW_TAG = "reader-raw"                # the same rows with the un-cut text
DEFAULT_MODEL = "qwen3-vl-8b-heb-v25a-step1200"
DEFAULT_OUT_DIR = "outputs_reader_dryrun"
DEFAULT_PANEL = "panels/prompt_ablation_v1.jsonl"
REFERENCES = ((rpa.TRAINED, 2500), (rpa.TRAINED, 1200), (rpa.NO_SCRIPT, 1200))
READER_FIELDS = ("raw_text", "reason", "blank", "looped", "cut_at", "letters_removed", "repeat_share", "attempts", "retried",
                 "total_tokens", "ink")
DECISION_FIELDS = ("images", "answered", "blank_gated", "failed", "looped", "cut", "retried", "length", "tokens_mean", "secs_mean")


def row_for(panel_row: Mapping[str, Any], read: Mapping[str, Any], model: str) -> Dict[str, Any]:
    """One output row: the ablation's row layout plus the reader's fields.

    :param panel_row: Panel row (``doc_id``, ``image_index``, ``image_file``, ``image_class``).
    :type panel_row: Mapping[str, Any]
    :param read: :func:`page_reader.read_page` result.
    :type read: Mapping[str, Any]
    :param model: LM Studio model key.
    :type model: str
    :return: The row.
    :rtype: Dict[str, Any]
    """
    return {"doc_id": panel_row["doc_id"], "image_index": int(panel_row["image_index"]), "image_file": panel_row["image_file"],
            "image_class": panel_row.get("image_class"), "text": read["text"], "finish_reason": read["finish_reason"],
            "prompt_tokens": read["prompt_tokens"], "completion_tokens": read["completion_tokens"], "secs": read["secs"],
            "model": model, "prompt": PROMPT_TAG, "max_tokens": read["max_tokens"], "temperature": read["temperature"],
            **{k: read[k] for k in READER_FIELDS}}


def raw_rows(rows: Mapping[Tuple[str, int], Mapping[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """The reader's rows with the model's un-cut text (the guard undone), for the ``reader-raw`` arm.

    :param rows: Reader rows by image.
    :type rows: Mapping[Tuple[str, int], Mapping[str, Any]]
    :return: Rows with ``text`` replaced by ``raw_text`` wherever a cut was made, ``prompt`` :data:`RAW_TAG`.
    :rtype: Dict[Tuple[str, int], Dict[str, Any]]
    """
    return {k: dict(r, prompt=RAW_TAG, text=r["raw_text"] if r.get("raw_text") is not None else r.get("text")) for k, r in rows.items()}


async def run(panel: Sequence[Mapping[str, Any]], model: str, out_dir: Path, benchmark: Path,
              settings: pr.ReaderSettings = pr.DEFAULT_SETTINGS, limit: Optional[int] = None, min_free_gb: float = 10.0,
              reader: Optional[Callable[..., Awaitable[Dict[str, Any]]]] = None, served_check: Optional[Callable[[str], bool]] = None,
              free_check: Optional[Callable[[], float]] = None, status_check: Optional[Callable[[], List[Dict[str, Any]]]] = None,
              wait_s: float = 30.0) -> Dict[str, int]:
    """Read every panel image that has no stored read yet.

    :param panel: Panel rows.
    :type panel: Sequence[Mapping[str, Any]]
    :param model: LM Studio model key.
    :type model: str
    :param out_dir: Output directory.
    :type out_dir: Path
    :param benchmark: Benchmark directory (images under ``images/``).
    :type benchmark: Path
    :param settings: Reader settings.
    :type settings: pr.ReaderSettings
    :param limit: Only the first N panel images.
    :type limit: Optional[int]
    :param min_free_gb: Disk floor in GiB; the run waits below it.
    :type min_free_gb: float
    :param reader: ``(model, image_path, settings) -> read`` (default :func:`page_reader.read_page`; injectable for tests).
    :param served_check: ``model -> bool`` (default: LM Studio lists the model).
    :param free_check: ``() -> GiB free`` (default: the local data volume).
    :param status_check: ``() -> loaded models`` (default ``run_prompt_ablation.lms_models``).
    :param wait_s: Seconds between checks while waiting.
    :type wait_s: float
    :return: ``answered``, ``failed``, ``skipped`` (already stored), ``stopped`` (1 when the model vanished), ``yield_s``,
        ``blank`` (gated), ``looped``, ``retried``.
    :rtype: Dict[str, int]
    """
    reader = reader or pr.read_page
    served_check = served_check or model_served
    free_check = free_check or free_gb
    status_check = status_check or rpa.lms_models
    out_path = rpa.output_path(out_dir, model, PROMPT_TAG, settings.max_tokens)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jobs, skipped = rpa.pending_jobs(panel, out_path, limit)
    stats = {"answered": 0, "failed": 0, "skipped": skipped, "stopped": 0, "yield_s": 0, "blank": 0, "looped": 0, "retried": 0}
    print(f"{model} / {PROMPT_TAG} / max_tokens {settings.max_tokens}: {len(jobs)} images to read, {skipped} already stored", flush=True)
    started = time.time()
    with open(out_path, "a", encoding="utf-8") as fh:
        for n, row in enumerate(jobs, 1):
            while free_check() < min_free_gb:
                print(f"  disk floor: {free_check():.1f} GiB free < {min_free_gb} GiB; waiting", flush=True)
                await asyncio.sleep(wait_s)
            busy = rpa.busy_others(status_check(), model)
            while busy:
                print(f"  yielding: other model(s) busy {busy}; waiting {wait_s:.0f}s", flush=True)
                stats["yield_s"] += int(wait_s)
                await asyncio.sleep(wait_s)
                busy = rpa.busy_others(status_check(), model)
            read = await reader(model, str(benchmark / "images" / row["image_file"]), settings)
            if read["reason"] == pr.REASON_FAILED and not served_check(model):
                print(f"  {model} is no longer served by LM Studio; stopping after {n - 1} images (resumable)", flush=True)
                stats["stopped"] = 1
                break
            fh.write(json.dumps(row_for(row, read, model), ensure_ascii=False) + "\n")
            fh.flush()
            stats["answered" if isinstance(read["text"], str) else "failed"] += 1
            stats["blank"] += bool(read["blank"])
            stats["looped"] += bool(read["looped"])
            stats["retried"] += bool(read["retried"])
            what = ("BLANK (gated)" if read["blank"] else "FAILED" if read["text"] is None else
                    f"{len(read['text'])} chars, finish={read['finish_reason']}, tokens={read['completion_tokens']}"
                    + (", LOOPED" if read["looped"] else "") + (f", cut {read['letters_removed']} letters" if read["cut_at"] is not None else "")
                    + (", retried" if read["retried"] else ""))
            print(f"  {n:4d}/{len(jobs)} {row['doc_id']}#{row['image_index']} [{row.get('image_class')}]: {what} in {read['secs']}s", flush=True)
    print(f"done: {stats} in {(time.time() - started) / 60:.1f} min -> {out_path}", flush=True)
    return stats


# --- scoring ----------------------------------------------------------------------------------------------------------

def load_arms(benchmark: Path, out_dir: Path, model: str, max_tokens: int, panel_keys: Sequence[Tuple[str, int]],
              references: Sequence[Tuple[str, int]] = REFERENCES) -> List[spa.Arm]:
    """The arms of the report: the ablation's reference arms that exist, the reader, and the reader with the un-cut text.

    :param benchmark: Benchmark directory (the ablation's arms under ``outputs_prompt_ablation``).
    :type benchmark: Path
    :param out_dir: The dry run's output directory.
    :type out_dir: Path
    :param model: Model key.
    :type model: str
    :param max_tokens: The reader's budget.
    :type max_tokens: int
    :param panel_keys: Panel images.
    :type panel_keys: Sequence[Tuple[str, int]]
    :param references: ``(prompt, max_tokens)`` of the ablation arms to score next to the reader.
    :type references: Sequence[Tuple[str, int]]
    :return: Arms in report order (references, reader, reader-raw).
    :rtype: List[spa.Arm]
    :raises FileNotFoundError: When the reader's rows are missing.
    """
    arms = []
    for prompt, budget in references:
        path = rpa.output_path(benchmark / rpa.DEFAULT_OUT_DIR, model, prompt, budget)
        if path.exists():
            arms.append(spa.Arm(path.stem, model, prompt, budget, spa.load_rows(path, panel_keys)))
    path = rpa.output_path(out_dir, model, PROMPT_TAG, max_tokens)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the reader first")
    rows = spa.load_rows(path, panel_keys)
    arms.append(spa.Arm(path.stem, model, PROMPT_TAG, max_tokens, rows))
    raw = raw_rows(rows)
    raw_path = rpa.output_path(out_dir, model, RAW_TAG, max_tokens)
    raw_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in raw.values()), encoding="utf-8")
    arms.append(spa.Arm(raw_path.stem, model, RAW_TAG, max_tokens, raw))
    return arms


def _mean(values: Sequence[Any]) -> Optional[float]:
    """Mean of the non-None values to one place, None when there are none."""
    vals = [v for v in values if v is not None]
    return round(statistics.mean(vals), 1) if vals else None


def decisions(panel: Sequence[Mapping[str, Any]], rows: Mapping[Tuple[str, int], Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """What the reader did per image class: gated, failed, looped, cut, retried, length finishes, tokens, seconds.

    :param panel: Panel rows.
    :type panel: Sequence[Mapping[str, Any]]
    :param rows: Reader rows by image.
    :type rows: Mapping[Tuple[str, int], Mapping[str, Any]]
    :return: ``{class or "all": {DECISION_FIELDS}}``.
    :rtype: Dict[str, Dict[str, Any]]
    """
    out: Dict[str, Dict[str, Any]] = {}
    for cls in list(CLASSES) + ["all"]:
        members = [p for p in panel if cls == "all" or p["image_class"] == cls]
        reads = [rows[(p["doc_id"], int(p["image_index"]))] for p in members if (p["doc_id"], int(p["image_index"])) in rows]
        out[cls] = {"images": len(members), "answered": sum(isinstance(r.get("text"), str) for r in reads),
                    "blank_gated": sum(r.get("reason") == pr.REASON_BLANK for r in reads),
                    "failed": sum(r.get("reason") == pr.REASON_FAILED for r in reads),
                    "looped": sum(bool(r.get("looped")) for r in reads), "cut": sum(r.get("cut_at") is not None for r in reads),
                    "retried": sum(bool(r.get("retried")) for r in reads),
                    "length": sum(r.get("finish_reason") == pr.LENGTH for r in reads),
                    "tokens_mean": _mean([r.get("completion_tokens") for r in reads]), "secs_mean": _mean([r.get("secs") for r in reads])}
    return out


def decisions_table(summary: Mapping[str, Mapping[str, Any]]) -> str:
    """Markdown table of :func:`decisions`.

    :param summary: :func:`decisions` result.
    :type summary: Mapping[str, Mapping[str, Any]]
    :return: Table.
    :rtype: str
    """
    lines = ["| class | images | answered | blank-gated | failed | looped | cut | retried | length finishes | tokens mean | secs mean |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cls, d in summary.items():
        lines.append(f"| {cls} | " + " | ".join(spa._f(d[k], 1) for k in DECISION_FIELDS) + " |")
    return "\n".join(lines)


def gate_check(summary: Mapping[str, Mapping[str, Any]]) -> str:
    """One line on whether the blank gate fired where it should (blank images) and nowhere it must not (covered images).

    :param summary: :func:`decisions` result.
    :type summary: Mapping[str, Mapping[str, Any]]
    :return: The line.
    :rtype: str
    """
    blank, covered = summary[BLANK], summary[COVERED]
    others = sum(d["blank_gated"] for c, d in summary.items() if c not in (BLANK, COVERED, "all"))
    verdict = "OK" if covered["blank_gated"] == 0 else "NOT OK: covered images were gated"
    return (f"Blank gate: {blank['blank_gated']} of {blank['images']} blank / faint images gated; {covered['blank_gated']} of "
            f"{covered['images']} covered images gated ({verdict}); {others} other uncovered images gated.")


def comparison_table(arms: Sequence[spa.Arm], scored: Mapping[str, Mapping[str, Any]], reader_tag: str) -> str:
    """The reader against every other arm on the covered documents (paired, bootstrap intervals).

    :param arms: Arms.
    :type arms: Sequence[spa.Arm]
    :param scored: ``{tag: score_arm(...)}``.
    :type scored: Mapping[str, Mapping[str, Any]]
    :param reader_tag: The reader arm's tag.
    :type reader_tag: str
    :return: Markdown table.
    :rtype: str
    """
    reader = next(a for a in arms if a.tag == reader_tag)
    lines = ["| reader vs | docs | F1 reader / other | median ΔF1 [95 % CI] | W / L | LER reader / other | median ΔLER [95 % CI] | W / L |",
             "|---|---|---|---|---|---|---|---|"]
    for other in arms:
        if other.tag == reader_tag:
            continue
        f = sac.paired_comparison(scored[reader.tag]["documents"], scored[other.tag]["documents"], "ngram_f1", True)
        l = sac.paired_comparison(scored[reader.tag]["documents"], scored[other.tag]["documents"], "ler", False)
        if not f.get("documents"):
            lines.append(f"| {spa.arm_label(other)} | 0 | – | – | – | – | – | – |")
            continue
        lines.append(f"| {spa.arm_label(other)} | {f['documents']} | {spa._f(f['new_median'])} / {spa._f(f['old_median'])} "
                     f"| {f['median_delta']:+.3f} [{f['median_ci'][0]:+.3f}, {f['median_ci'][1]:+.3f}] | {f['wins']} / {f['losses']} "
                     f"| {spa._f(l['new_median'])} / {spa._f(l['old_median'])} | {l['median_delta']:+.3f} [{l['median_ci'][0]:+.3f}, "
                     f"{l['median_ci'][1]:+.3f}] | {l['wins']} / {l['losses']} |")
    return "\n".join(lines)


def score(panel_path: Path, benchmark: Path, out_dir: Path, model: str, max_tokens: int,
          coverage_path: Optional[Path] = None) -> Tuple[str, Dict[str, Any]]:
    """Score the dry run next to the ablation's arms and write the report files.

    :param panel_path: Panel file.
    :type panel_path: Path
    :param benchmark: Benchmark directory.
    :type benchmark: Path
    :param out_dir: The dry run's output directory (report files go there too).
    :type out_dir: Path
    :param model: Model key.
    :type model: str
    :param max_tokens: The reader's budget.
    :type max_tokens: int
    :param coverage_path: Coverage map (default ``<benchmark>/coverage_v2.jsonl``).
    :type coverage_path: Optional[Path]
    :return: ``(report text, summary)``; the summary holds every arm's class summaries and the reader's decisions.
    :rtype: Tuple[str, Dict[str, Any]]
    """
    panel = load_panel(panel_path)
    keys = [(p["doc_id"], int(p["image_index"])) for p in panel]
    bench = sab.load_benchmark(benchmark)
    coverage = sac.load_coverage(coverage_path or benchmark / sac.DEFAULT_COVERAGE)
    arms = load_arms(benchmark, out_dir, model, max_tokens, keys)
    reader_tag = rpa.output_path(out_dir, model, PROMPT_TAG, max_tokens).stem
    scored = {a.tag: spa.score_arm(a, panel, bench, coverage) for a in arms}
    reader_rows = next(a for a in arms if a.tag == reader_tag).rows
    decided = decisions(panel, reader_rows)
    report = "\n".join([
        "# Production page reader: dry run on the prompt-ablation panel", "",
        f"Model `{model}`; reader = script-neutral prompt, max_tokens {max_tokens}, loop guard on `length` finishes, one retry at "
        "temperature 0, ink-density blank gate. Reference arms are the ablation's own reads of the same images (`trained` / 2500 = the "
        "production setting before this change; `noscript` / 1200 = the same prompt and budget without gate, guard or retry); "
        "`reader-raw` = the reader's rows with the un-cut text.", "",
        "## Reader decisions", "", gate_check(decided), "", decisions_table(decided), "",
        "## Reader vs every other arm on the covered documents (paired)", "", comparison_table(arms, scored, reader_tag), "",
        spa.build_report(arms, scored, panel, panel_path)])
    out_dir.mkdir(parents=True, exist_ok=True)
    spa.write_csv(out_dir / "images.csv", [r for a in arms for r in scored[a.tag]["images"]], spa.IMAGE_FIELDS)
    spa.write_csv(out_dir / "documents.csv", [r for a in arms for r in scored[a.tag]["documents"]], spa.DOC_FIELDS)
    summary = {"model": model, "max_tokens": max_tokens, "reader_tag": reader_tag, "decisions": decided,
               "arms": {a.tag: {"model": a.model, "prompt": a.prompt, "max_tokens": a.max_tokens, **scored[a.tag]["by_class"]} for a in arms}}
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    return report, summary


def main(argv: Optional[List[str]] = None) -> None:
    """CLI.

    :param argv: Arguments.
    :type argv: Optional[List[str]]
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", type=Path, default=sab.DEFAULT_BENCHMARK)
    parser.add_argument("--panel", type=Path, default=None, help=f"default: <benchmark>/{DEFAULT_PANEL}")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LM Studio model key")
    parser.add_argument("--out-dir", type=Path, default=None, help=f"default: <benchmark>/{DEFAULT_OUT_DIR}")
    parser.add_argument("--max-tokens", type=int, default=pr.READ_MAX_TOKENS)
    parser.add_argument("--no-retry", action="store_true", help="no retry at temperature 0 after a length finish")
    parser.add_argument("--no-blank-gate", action="store_true", help="send every image to the model")
    parser.add_argument("--limit", type=int, default=None, help="only the first N panel images")
    parser.add_argument("--min-free-gb", type=float, default=10.0)
    parser.add_argument("--score-only", action="store_true", help="skip the reads, score what is stored")
    args = parser.parse_args(argv)
    panel_path = args.panel or args.benchmark / DEFAULT_PANEL
    out_dir = args.out_dir or args.benchmark / DEFAULT_OUT_DIR
    settings = pr.ReaderSettings(max_tokens=args.max_tokens, retry_on_length=not args.no_retry, blank_gate=not args.no_blank_gate)
    if not args.score_only:
        stats = asyncio.run(run(load_panel(panel_path), args.model, out_dir, args.benchmark, settings, args.limit, args.min_free_gb))
        info = rpa.model_info(rpa.lms_models(), args.model)
        print(json.dumps({"stats": stats, "loaded": info}), flush=True)
        if stats["stopped"]:
            return
    report, _ = score(panel_path, args.benchmark, out_dir, args.model, args.max_tokens)
    print(report)


if __name__ == "__main__":
    main()
