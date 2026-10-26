# File name: run_prompt_ablation.py
# Date: 10/24/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Decode a fixed image panel under one prompt and one output budget (the Arabic-script prompt ablation).

The v2.5a reader writes Arabic on sides that carry none and sometimes loops. Two reviews point at the
prompt it was trained and benchmarked with (``build_pgp_arabic_editions.ARABIC_FRAGMENT_PROMPT``): it
asserts "handwritten Arabic script" for every image and defines no output for an image with nothing to
read. This runner decodes the panel of ``build_prompt_ablation_panel.py`` with the same LM Studio call,
temperature and image handling as ``run_arabic_benchmark.py`` and keeps, per image, what that runner
discards: the generation's ``finish_reason`` (``stop`` = the model emitted its end token, ``length`` =
the budget cut it) and the server's token counts.

Prompt variants:

* ``trained``: the training rows' own prompt, unchanged;
* ``noscript``: the same prompt with the script assertion replaced by :data:`NO_SCRIPT_CLAUSE` (nothing
  else changes) — since 2026-10-25 the production reader's prompt, ``prompts.SCRIPT_NEUTRAL_PROMPT``;
* ``stopping``: :data:`STOPPING_PROMPT`, which names an abstention output (:data:`ABSTAIN_MARKER`) and
  tells the model to stop after the last visible text.

Output: ``<out-dir>/<model>__<variant>__mt<max_tokens>.jsonl``, one row per image (``doc_id``,
``image_index``, ``image_file``, ``image_class``, ``text``, ``finish_reason``, ``prompt_tokens``,
``completion_tokens``, ``secs``, ``model``, ``prompt``, ``max_tokens``, ``temperature``); the run is
resumable (an image with an answer is skipped, a failed one retried on the next run). Next to it
``<model>.loaded.json`` records how LM Studio holds the model (context length, status) after the run.

Shared machine: one LM Studio consumer at a time. Before each request the runner reads ``lms ps`` and
waits while any *other* loaded model is not idle (a production chat is in flight), waits while local
free disk is under ``--min-free-gb`` and stops when LM Studio no longer serves the model.

Usage::

    .venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.run_prompt_ablation \\
        --panel <benchmark>/panels/prompt_ablation_v1.jsonl --model qwen3-vl-8b-heb-v25a-step1200 \\
        --prompt trained|noscript|stopping --max-tokens 2500 --out-dir <benchmark>/outputs_prompt_ablation
"""
import argparse
import asyncio
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.datasets.evaluations.helper_eval_scripts.build_prompt_ablation_panel import load_panel
from src.datasets.evaluations.helper_eval_scripts.score_arabic_benchmark import DEFAULT_BENCHMARK
from src.finetuning.qwen_hebrew.eval_harness.v22_task_eval import free_gb, model_served
from src.finetuning.qwen_hebrew.prompts import ARABIC_FRAGMENT_PROMPT, SCRIPT_NEUTRAL_CLAUSE, SCRIPT_NEUTRAL_PROMPT
from src.models.ocr.lms_transcriber import DEFAULT_CONFIG, transcribe_with_lm_studio_meta

TRAINED, NO_SCRIPT, STOPPING = "trained", "noscript", "stopping"
VARIANTS = (TRAINED, NO_SCRIPT, STOPPING)
NO_SCRIPT_CLAUSE = SCRIPT_NEUTRAL_CLAUSE          # the production reader's clause (``prompts.SCRIPT_NEUTRAL_PROMPT``)
STOPPING_PROMPT = ("Transcribe only text visible in this image, using its original script and reading order. "
                   "Mark unclear characters with [?] and damaged spans with [...]. "
                   "Do not restore or complete text from memory. Stop after the last visible text. "
                   "If no text is readable, return [NO_READABLE_TEXT]. Return only the transcription or that marker.")
ABSTAIN_MARKER = "[NO_READABLE_TEXT]"
DEFAULT_MAX_TOKENS = 2500                     # run_arabic_benchmark.MAX_TOKENS, the budget of every earlier Arabic run
TEMPERATURE = DEFAULT_CONFIG.temperature      # the benchmarks' sampling temperature, unchanged
DEFAULT_OUT_DIR = "outputs_prompt_ablation"
LMS = Path.home() / ".lmstudio" / "bin" / "lms"
IDLE = "idle"


def build_prompt(variant: str) -> str:
    """The prompt of a variant.

    :param variant: One of :data:`VARIANTS`.
    :type variant: str
    :return: Prompt text.
    :rtype: str
    :raises ValueError: On an unknown variant.
    """
    if variant == STOPPING:
        return STOPPING_PROMPT
    if variant == TRAINED:
        return ARABIC_FRAGMENT_PROMPT
    if variant == NO_SCRIPT:
        return SCRIPT_NEUTRAL_PROMPT
    raise ValueError(f"unknown prompt variant {variant!r}; expected one of {VARIANTS}")


def output_path(out_dir: Path, model: str, variant: str, max_tokens: int) -> Path:
    """Where one arm's answers are stored.

    :param out_dir: Output directory.
    :type out_dir: Path
    :param model: LM Studio model key (``/`` replaced so it can be a file name).
    :type model: str
    :param variant: Prompt variant.
    :type variant: str
    :param max_tokens: Output budget.
    :type max_tokens: int
    :return: ``<out_dir>/<model>__<variant>__mt<max_tokens>.jsonl``.
    :rtype: Path
    """
    return out_dir / f"{model.replace('/', '_')}__{variant}__mt{max_tokens}.jsonl"


def pending_jobs(panel: Sequence[Mapping[str, Any]], out_path: Path, limit: Optional[int] = None) -> Tuple[List[Mapping[str, Any]], int]:
    """Panel images that still need an answer.

    :param panel: Panel rows.
    :type panel: Sequence[Mapping[str, Any]]
    :param out_path: The arm's answer file (may not exist yet).
    :type out_path: Path
    :param limit: Only the first N panel images.
    :type limit: Optional[int]
    :return: ``(rows to decode, number already answered)``.
    :rtype: Tuple[List[Mapping[str, Any]], int]
    """
    answered = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row.get("text"), str):
                        answered.add((row["doc_id"], int(row["image_index"])))
    rows = list(panel)[:limit] if limit else list(panel)
    return [r for r in rows if (r["doc_id"], int(r["image_index"])) not in answered], \
        sum((r["doc_id"], int(r["image_index"])) in answered for r in rows)


def lms_models(lms: Path = LMS) -> List[Dict[str, Any]]:
    """The models LM Studio holds in memory, as ``lms ps --json`` reports them.

    :param lms: The ``lms`` executable.
    :type lms: Path
    :return: One dict per loaded model (``identifier``, ``status``, ``contextLength``, ...); empty when the
        command fails or prints no JSON list.
    :rtype: List[Dict[str, Any]]
    """
    if not lms.exists():
        return []
    proc = subprocess.run([str(lms), "ps", "--json"], capture_output=True, text=True, timeout=30)
    out = proc.stdout.strip()
    if proc.returncode != 0 or not out.startswith("["):
        return []
    try:
        models = json.loads(out)
    except json.JSONDecodeError:      # a bracketed log line such as "[warn] ..." is not the list
        return []
    return models if isinstance(models, list) else []


def busy_others(models: Sequence[Mapping[str, Any]], model: str) -> List[str]:
    """Loaded models other than ``model`` that are not idle (a production request is in flight).

    :param models: :func:`lms_models` result.
    :type models: Sequence[Mapping[str, Any]]
    :param model: This run's model key.
    :type model: str
    :return: Their identifiers.
    :rtype: List[str]
    """
    return [m.get("identifier", "?") for m in models
            if m.get("identifier") != model and str(m.get("status", IDLE)).lower() != IDLE]


def model_info(models: Sequence[Mapping[str, Any]], model: str) -> Optional[Dict[str, Any]]:
    """How LM Studio holds ``model``: the fields worth recording next to a run.

    :param models: :func:`lms_models` result.
    :type models: Sequence[Mapping[str, Any]]
    :param model: Model key.
    :type model: str
    :return: ``identifier``, ``contextLength``, ``maxContextLength``, ``status``, ``parallel``, ``quantization``;
        None when the model is not loaded.
    :rtype: Optional[Dict[str, Any]]
    """
    for m in models:
        if m.get("identifier") == model:
            return {k: m.get(k) for k in ("identifier", "contextLength", "maxContextLength", "status", "parallel", "quantization")}
    return None


async def run(panel: Sequence[Mapping[str, Any]], model: str, variant: str, max_tokens: int, out_dir: Path, benchmark: Path,
              limit: Optional[int] = None, min_free_gb: float = 10.0, temperature: float = TEMPERATURE,
              transcribe: Optional[Callable[..., Awaitable[Optional[Dict[str, Any]]]]] = None,
              served_check: Optional[Callable[[str], bool]] = None, free_check: Optional[Callable[[], float]] = None,
              status_check: Optional[Callable[[], List[Dict[str, Any]]]] = None, wait_s: float = 30.0,
              prompt_text: Optional[str] = None) -> Dict[str, int]:
    """Decode every panel image of one arm that has no answer yet.

    :param panel: Panel rows.
    :type panel: Sequence[Mapping[str, Any]]
    :param model: LM Studio model key.
    :type model: str
    :param variant: Prompt variant.
    :type variant: str
    :param max_tokens: Output token budget.
    :type max_tokens: int
    :param out_dir: Output directory.
    :type out_dir: Path
    :param benchmark: Benchmark directory (images under ``images/``).
    :type benchmark: Path
    :param limit: Only the first N panel images.
    :type limit: Optional[int]
    :param min_free_gb: Disk floor in GiB; the run waits below it.
    :type min_free_gb: float
    :param temperature: Sampling temperature.
    :type temperature: float
    :param transcribe: ``(model, image_path, prompt, temperature=..., max_tokens=...) -> meta dict or None``
        (injectable for tests; default :func:`transcribe_with_lm_studio_meta`).
    :param served_check: ``model -> bool`` (default: LM Studio lists the model).
    :param free_check: ``() -> GiB free`` (default: the local data volume).
    :param status_check: ``() -> loaded models`` (default :func:`lms_models`).
    :param wait_s: Seconds between checks while waiting.
    :type wait_s: float
    :param prompt_text: The prompt to send (default :func:`build_prompt` of the variant).
    :type prompt_text: Optional[str]
    :return: ``{"answered": n, "failed": n, "skipped": already answered, "stopped": 1 if the model vanished, "yield_s": seconds waited for other models}``.
    :rtype: Dict[str, int]
    """
    transcribe = transcribe or transcribe_with_lm_studio_meta
    served_check = served_check or model_served
    free_check = free_check or free_gb
    status_check = status_check or lms_models
    prompt = prompt_text if prompt_text is not None else build_prompt(variant)
    out_path = output_path(out_dir, model, variant, max_tokens)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jobs, skipped = pending_jobs(panel, out_path, limit)
    stats = {"answered": 0, "failed": 0, "skipped": skipped, "stopped": 0, "yield_s": 0}
    print(f"{model} / {variant} / max_tokens {max_tokens}: {len(jobs)} images to decode, {skipped} already answered", flush=True)
    started = time.time()
    with open(out_path, "a", encoding="utf-8") as fh:
        for n, row in enumerate(jobs, 1):
            while free_check() < min_free_gb:
                print(f"  disk floor: {free_check():.1f} GiB free < {min_free_gb} GiB; waiting", flush=True)
                await asyncio.sleep(wait_s)
            busy = busy_others(status_check(), model)
            while busy:
                print(f"  yielding: other model(s) busy {busy}; waiting {wait_s:.0f}s", flush=True)
                stats["yield_s"] += int(wait_s)
                await asyncio.sleep(wait_s)
                busy = busy_others(status_check(), model)
            t0 = time.time()
            meta = await transcribe(model, str(benchmark / "images" / row["image_file"]), prompt,
                                    temperature=temperature, max_tokens=max_tokens)
            if meta is None and not served_check(model):
                print(f"  {model} is no longer served by LM Studio; stopping after {n - 1} images (resumable)", flush=True)
                stats["stopped"] = 1
                break
            meta = meta or {"text": None, "finish_reason": None, "prompt_tokens": None, "completion_tokens": None}
            text = meta["text"]
            fh.write(json.dumps({"doc_id": row["doc_id"], "image_index": int(row["image_index"]), "image_file": row["image_file"],
                                 "image_class": row.get("image_class"), "text": text, "finish_reason": meta.get("finish_reason"),
                                 "prompt_tokens": meta.get("prompt_tokens"), "completion_tokens": meta.get("completion_tokens"),
                                 "secs": round(time.time() - t0, 1), "model": model, "prompt": variant, "max_tokens": max_tokens,
                                 "temperature": temperature}, ensure_ascii=False) + "\n")
            fh.flush()
            stats["answered" if isinstance(text, str) else "failed"] += 1
            print(f"  {n:4d}/{len(jobs)} {row['doc_id']}#{row['image_index']} [{row.get('image_class')}]: "
                  f"{len(text) if text else 'FAILED'} chars, finish={meta.get('finish_reason')}, "
                  f"tokens={meta.get('completion_tokens')} in {time.time() - t0:.1f}s", flush=True)
    print(f"done: {stats} in {(time.time() - started) / 60:.1f} min -> {out_path}", flush=True)
    return stats


def main(argv: Optional[List[str]] = None) -> None:
    """CLI.

    :param argv: Arguments.
    :type argv: Optional[List[str]]
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", type=Path, required=True, help="panel .jsonl of build_prompt_ablation_panel")
    parser.add_argument("--model", required=True, help="LM Studio model key")
    parser.add_argument("--prompt", choices=VARIANTS, required=True)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--out-dir", type=Path, default=None, help=f"default: <benchmark>/{DEFAULT_OUT_DIR}")
    parser.add_argument("--limit", type=int, default=None, help="only the first N panel images")
    parser.add_argument("--min-free-gb", type=float, default=10.0)
    args = parser.parse_args(argv)
    out_dir = args.out_dir or args.benchmark / DEFAULT_OUT_DIR
    stats = asyncio.run(run(load_panel(args.panel), args.model, args.prompt, args.max_tokens, out_dir, args.benchmark,
                            args.limit, args.min_free_gb))
    info = model_info(lms_models(), args.model)
    if info is not None:
        with open(out_dir / f"{args.model.replace('/', '_')}.loaded.json", "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=1)
    print(json.dumps({"stats": stats, "loaded": info}), flush=True)


if __name__ == "__main__":
    main()
