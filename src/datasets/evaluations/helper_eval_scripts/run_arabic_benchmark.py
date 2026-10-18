# File name: run_arabic_benchmark.py
# Date: 10/2/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Decode the Arabic-script benchmark with a local LM Studio vision model.

One request per image, sequential (one LM Studio consumer at a time), through the same call
the other benchmarks use (``lms_transcriber.transcribe_with_lm_studio``, 2,500 output tokens as
in the religious benchmark). Answers are appended to
``<benchmark>/outputs/<model>__<prompt>.jsonl`` and the run is resumable: an image that already
has an answer is skipped, a failed request is retried on the next run.

Two prompt variants, because the prompt the fine-tunes were benchmarked with says the script is
Hebrew:

* ``arabic``: the same instructions, stating that the page is in Arabic script. This measures
  whether the model can read Arabic script at all (the primary condition).
* ``standard``: ``consensus_gate.build_fragment_prompt`` unchanged. This measures what happens to
  an Arabic page in today's pipelines, which do not know the script in advance.

Safeguards for the shared machine, as in ``v22_task_eval``: the run waits while local free disk
is under ``--min-free-gb`` and stops as soon as LM Studio no longer serves the model.

Usage::

    .venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.run_arabic_benchmark \\
        --model qwen/qwen3-vl-8b [--prompt arabic|standard] [--benchmark DIR] [--limit N]
"""
import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from src.datasets.consensus.consensus_gate import _series_from_doc_id, build_fragment_prompt
from src.datasets.evaluations.helper_eval_scripts.score_arabic_benchmark import DEFAULT_BENCHMARK, load_benchmark
from src.finetuning.qwen_hebrew.eval_harness.v22_task_eval import free_gb, model_served
from src.models.ocr.lms_transcriber import transcribe_with_lm_studio

MAX_TOKENS = 2500
PROMPT_VARIANTS = ("arabic", "standard", "trained")


def build_prompt(doc_id: str, variant: str) -> str:
    """The transcription prompt for one benchmark document.

    :param doc_id: Canonical document id.
    :type doc_id: str
    :param variant: ``"standard"`` (the benchmarked fragment prompt, which describes Hebrew script),
        ``"arabic"`` (the same instructions for a page in Arabic script) or ``"trained"`` (the exact prompt
        of the Arabic-script training rows, ``build_pgp_arabic_editions.ARABIC_FRAGMENT_PROMPT``: a checkpoint
        that learned Arabic script under that wording is asked the way it was taught).
    :type variant: str
    :return: Prompt text.
    :rtype: str
    :raises ValueError: On an unknown variant.
    """
    if variant == "standard":
        return build_fragment_prompt(doc_id)
    if variant == "trained":
        from src.finetuning.qwen_hebrew.build_pgp_arabic_editions import ARABIC_FRAGMENT_PROMPT  # heavy import: only here
        return ARABIC_FRAGMENT_PROMPT
    if variant != "arabic":
        raise ValueError(f"unknown prompt variant {variant!r}; expected one of {PROMPT_VARIANTS}")
    return f"""This is a manuscript from the Cairo Genizah ({_series_from_doc_id(doc_id)} collection).
The text is written in Arabic script.

Transcribe the text in this image exactly as written, in Arabic script. Do not normalize or correct the text.
Preserve the line structure.
Mark damaged or unclear characters with [?].

Return ONLY the transcription with no commentary."""


def output_path(benchmark: Path, model: str, variant: str) -> Path:
    """Where a model's answers for a prompt variant are stored.

    :param benchmark: Benchmark directory.
    :type benchmark: Path
    :param model: LM Studio model key (``/`` is replaced so it can be a file name).
    :type model: str
    :param variant: Prompt variant.
    :type variant: str
    :return: ``<benchmark>/outputs/<model>__<variant>.jsonl``.
    :rtype: Path
    """
    return benchmark / "outputs" / f"{model.replace('/', '_')}__{variant}.jsonl"


def pending_jobs(records: Dict[str, Dict[str, Any]], out_path: Path, limit: Optional[int] = None) -> Tuple[List[Tuple[str, int, str]], int]:
    """Images that still need an answer.

    :param records: Benchmark records by document id.
    :type records: Dict[str, Dict[str, Any]]
    :param out_path: The model's answer file (may not exist yet).
    :type out_path: Path
    :param limit: Only the first N benchmark documents.
    :type limit: Optional[int]
    :return: ``([(doc_id, image_index, file name)], number already answered)``.
    :rtype: Tuple[List[Tuple[str, int, str]], int]
    """
    answered = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row.get("text"), str):
                        answered.add((row["doc_id"], int(row["image_index"])))
    docs = list(records.values())[:limit] if limit else list(records.values())
    jobs = [(d["id"], int(i["image_index"]), i["file"]) for d in docs for i in d["images"]]
    return [j for j in jobs if (j[0], j[1]) not in answered], sum((j[0], j[1]) in answered for j in jobs)


async def run(benchmark: Path, model: str, variant: str, limit: Optional[int] = None, min_free_gb: float = 10.0,
              max_tokens: int = MAX_TOKENS, transcribe: Optional[Callable[..., Awaitable[Optional[str]]]] = None,
              served_check: Optional[Callable[[str], bool]] = None, free_check: Optional[Callable[[], float]] = None,
              wait_s: float = 60.0) -> Dict[str, int]:
    """Decode every image of the benchmark that has no answer yet.

    :param benchmark: Benchmark directory.
    :type benchmark: Path
    :param model: LM Studio model key.
    :type model: str
    :param variant: Prompt variant.
    :type variant: str
    :param limit: Only the first N documents.
    :type limit: Optional[int]
    :param min_free_gb: Disk floor in GiB; the run waits below it.
    :type min_free_gb: float
    :param max_tokens: Output token budget per image.
    :type max_tokens: int
    :param transcribe: ``(model, image_path, prompt, max_tokens=...) -> text or None`` (injectable for tests).
    :param served_check: ``model -> bool`` (default: LM Studio lists the model).
    :param free_check: ``() -> GiB free`` (default: the local data volume).
    :param wait_s: Seconds between disk checks while under the floor.
    :type wait_s: float
    :return: ``{"answered": n, "failed": n, "skipped": already answered, "stopped": 1 if the model vanished}``.
    :rtype: Dict[str, int]
    """
    transcribe = transcribe or transcribe_with_lm_studio
    served_check = served_check or model_served
    free_check = free_check or free_gb
    records = load_benchmark(benchmark)
    out_path = output_path(benchmark, model, variant)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jobs, skipped = pending_jobs(records, out_path, limit)
    stats = {"answered": 0, "failed": 0, "skipped": skipped, "stopped": 0}
    print(f"{model} / {variant}: {len(jobs)} images to decode, {skipped} already answered", flush=True)
    started = time.time()
    with open(out_path, "a", encoding="utf-8") as fh:
        for n, (doc_id, image_index, file_name) in enumerate(jobs, 1):
            while free_check() < min_free_gb:
                print(f"  disk floor: {free_check():.1f} GiB free < {min_free_gb} GiB; waiting", flush=True)
                await asyncio.sleep(wait_s)
            t0 = time.time()
            text = await transcribe(model, str(benchmark / "images" / file_name), build_prompt(doc_id, variant), max_tokens=max_tokens)
            if text is None and not served_check(model):
                print(f"  {model} is no longer served by LM Studio; stopping after {n - 1} images (resumable)", flush=True)
                stats["stopped"] = 1
                break
            fh.write(json.dumps({"doc_id": doc_id, "image_index": image_index, "text": text, "secs": round(time.time() - t0, 1),
                                 "model": model, "prompt": variant}, ensure_ascii=False) + "\n")
            fh.flush()
            stats["answered" if isinstance(text, str) else "failed"] += 1
            print(f"  {n:4d}/{len(jobs)} {doc_id}#{image_index}: {len(text) if text else 'FAILED'} chars in {time.time() - t0:.1f}s", flush=True)
    print(f"done: {stats} in {(time.time() - started) / 60:.1f} min -> {out_path}", flush=True)
    return stats


def main(argv: Optional[List[str]] = None) -> None:
    """CLI.

    :param argv: Arguments.
    :type argv: Optional[List[str]]
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="LM Studio model key")
    parser.add_argument("--prompt", choices=PROMPT_VARIANTS, default="arabic")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--limit", type=int, default=None, help="only the first N documents")
    parser.add_argument("--min-free-gb", type=float, default=10.0)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args(argv)
    asyncio.run(run(args.benchmark, args.model, args.prompt, args.limit, args.min_free_gb, args.max_tokens))


if __name__ == "__main__":
    main()
