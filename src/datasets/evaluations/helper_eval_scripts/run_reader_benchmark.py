# File name: run_reader_benchmark.py
# Date: 11/08/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Read a page benchmark with the production page reader, so a checkpoint is scored the way production reads.

The legacy benchmark runners (``run_pgp131_v19b.py``, ``run_religious_benchmark.py``) send the per-fragment benchmark
prompt with an 8,192 / 2,500-token budget and keep whatever comes back. Production reads go through
:func:`src.models.ocr.page_reader.read_page` instead: the script-neutral prompt, 1,200 tokens, the letter loop guard with
one retry at temperature 0 after a ``length`` finish, and the ink gate that answers a blank image without a request.
This runner reads every page of a benchmark spec that way.

Specs: the PGP-131 format (``{"docs": [{"doc_id": ...}]}``, images ``<images-dir>/<doc_id>.jpg``) or the religious-140
format (each doc carries an absolute ``image``). Outputs, per page: ``<outputs-dir>/<doc_id>/<key>.txt`` (the stored
read, which ``score_genizah_offline.py`` and ``run_religious_benchmark.py --score-only`` pick up as model ``<key>``) and
``<key>.read.json`` beside it (the reader's other fields: finish, loop flags, attempts, token counts, gate reason).
``<key>`` is the LM Studio id with ``-`` replaced by ``_`` plus ``--key-suffix`` (default ``_prod``). Resumable: a page
whose text file exists is skipped; a failed request stores nothing, so the next run asks again.

Shared machine: runs alongside other jobs (user 2026-11-08: run in parallel, no pausing; export
``LMS_YIELD_MODE=dryrun`` so the transcriber's yield gate only logs). The run waits while free disk is under
``--min-free-gb`` and unloads its own model (never a production one) when free disk falls under ``--unload-below-gb``,
because the vision worker's temp file grows with each image request and is released only when the worker exits; it
then loads the model again explicitly. Load the model explicitly before the run too: LM Studio's on-demand (JIT) load
evicts the previous on-demand model, so two runs on JIT-loaded models unload each other mid-request.

Usage::

    .venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.run_reader_benchmark MODEL \\
        [--spec SPEC.json] [--images-dir DIR] [--outputs-dir DIR] [--key-suffix _prod] [--limit N]
"""
import argparse
import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.models.ocr import page_reader as pr
from src.models.ocr.lms_transcriber import check_lm_studio_health

REPO = Path(__file__).resolve().parents[4]
PGP131_BENCH = REPO / "src/datasets/raw_data/cairo_genizah/evaluations/genizah_test_v1"
DEFAULT_SPEC = PGP131_BENCH / "genizah_test_v1_verified.json"
DEFAULT_IMAGES = PGP131_BENCH / "images"
DEFAULT_OUTPUTS = REPO / "src/datasets/evaluations/transcription_raw_outputs"
KEY_SUFFIX = "_prod"
LMS = Path.home() / ".lmstudio" / "bin" / "lms"
DISK_PATH = "/System/Volumes/Data"
#: Models this runner never unloads (the production LM Studio models).
PROTECTED_MODELS = ("qwen/qwen3-4b-2507", "qwen/qwen3.6-35b-a3b", "qwen/qwen3.8-27b", "qwen3-vl-8b-heb-v21b-step1200")
#: Explicit-load settings: the context and parallel slots LM Studio's JIT load gives these checkpoints (2026-11-08).
LOAD_CONTEXT = 8192
LOAD_PARALLEL = 4
#: The reader's fields stored beside the text.
SIDE_FIELDS = tuple(f for f in pr.READ_FIELDS if f != "text")
MAX_CONSECUTIVE_FAILURES = 5

Page = Tuple[str, Path]
Reader = Callable[[str, str, pr.ReaderSettings], Awaitable[Dict[str, Any]]]


def output_key(model: str, suffix: str = KEY_SUFFIX) -> str:
    """The scorer's model key for a production read of ``model``.

    :param model: LM Studio model id.
    :type model: str
    :param suffix: Appended to the key, so production reads sit beside the legacy reads of the same model.
    :type suffix: str
    :return: The id with ``-`` replaced by ``_``, plus ``suffix``.
    :rtype: str
    """
    return model.replace("-", "_") + suffix


def load_pages(spec: Path, images_dir: Optional[Path] = None) -> List[Page]:
    """The ``(doc_id, image)`` pairs of a benchmark spec, in spec order.

    :param spec: Benchmark JSON (``{"docs": [...]}`` or a list of docs).
    :type spec: Path
    :param images_dir: Folder of ``<doc_id>.jpg`` for docs without their own ``image`` (the PGP-131 format).
    :type images_dir: Optional[Path]
    :return: One pair per doc; a doc's own ``image`` wins over ``images_dir``.
    :rtype: List[Page]
    :raises ValueError: When a doc has no ``image`` and no ``images_dir`` is given.
    """
    data = json.loads(Path(spec).read_text(encoding="utf-8"))
    docs = data["docs"] if isinstance(data, dict) else data
    pages: List[Page] = []
    for doc in docs:
        if doc.get("image"):
            pages.append((doc["doc_id"], Path(doc["image"])))
        elif images_dir is not None:
            pages.append((doc["doc_id"], Path(images_dir) / f"{doc['doc_id']}.jpg"))
        else:
            raise ValueError(f"{doc['doc_id']}: no image in the spec and no images_dir")
    return pages


def pending(pages: Sequence[Page], out_dir: Path, key: str) -> Tuple[List[Page], int, List[str]]:
    """Split the pages into those still to read, those already read and those without an image file.

    :param pages: ``(doc_id, image)`` pairs.
    :type pages: Sequence[Page]
    :param out_dir: Output root (``<out_dir>/<doc_id>/<key>.txt``).
    :type out_dir: Path
    :param key: Output key.
    :type key: str
    :return: The pages to read, the number already read and the doc ids whose image is missing.
    :rtype: Tuple[List[Page], int, List[str]]
    """
    todo: List[Page] = []
    done, missing = 0, []
    for doc_id, image in pages:
        if (Path(out_dir) / doc_id / f"{key}.txt").exists():
            done += 1
        elif not Path(image).exists():
            missing.append(doc_id)
        else:
            todo.append((doc_id, Path(image)))
    return todo, done, missing


def side_record(result: Mapping[str, Any], doc_id: str) -> Dict[str, Any]:
    """The reader's fields kept beside the stored text.

    :param result: A :func:`page_reader.read_page` result.
    :type result: Mapping[str, Any]
    :param doc_id: The page's doc id.
    :type doc_id: str
    :return: ``doc_id`` plus every :data:`SIDE_FIELDS` key present in the result.
    :rtype: Dict[str, Any]
    """
    record: Dict[str, Any] = {"doc_id": doc_id}
    record.update({k: result[k] for k in SIDE_FIELDS if k in result})
    return record


def store(out_dir: Path, doc_id: str, key: str, result: Mapping[str, Any]) -> bool:
    """Write one read: ``<key>.read.json`` (its fields), then ``<key>.txt`` (the text, the resume marker).

    :param out_dir: Output root.
    :type out_dir: Path
    :param doc_id: The page's doc id.
    :type doc_id: str
    :param key: Output key.
    :type key: str
    :param result: A :func:`page_reader.read_page` result.
    :type result: Mapping[str, Any]
    :return: True when stored; False for a failed request (``text`` None), which writes nothing.
    :rtype: bool
    """
    text = result.get("text")
    if text is None:
        return False
    page_dir = Path(out_dir) / doc_id
    page_dir.mkdir(parents=True, exist_ok=True)
    side = json.dumps(side_record(result, doc_id), ensure_ascii=False, default=str)
    (page_dir / f"{key}.read.json").write_text(side + "\n", encoding="utf-8")
    (page_dir / f"{key}.txt").write_text(text, encoding="utf-8")
    return True


def free_disk_gb(path: str = DISK_PATH) -> float:
    """Free space on the data volume.

    :param path: A path on the volume.
    :type path: str
    :return: Free GiB.
    :rtype: float
    """
    return shutil.disk_usage(path).free / 2 ** 30


def unload_own_model(model: str, lms: Path = LMS, reload: bool = True) -> str:
    """Unload this run's model from LM Studio to release its worker's temp file, then load it again explicitly; never
    a production model.

    The explicit load keeps the model out of LM Studio's JIT auto-evict: a model loaded on demand is unloaded by the
    next on-demand load of another model, mid-request ("Model unloaded", 2026-11-08, two runs in parallel).

    :param model: LM Studio model id.
    :type model: str
    :param lms: The ``lms`` executable.
    :type lms: Path
    :param reload: Load the model again explicitly (:data:`LOAD_CONTEXT`, :data:`LOAD_PARALLEL`, full GPU offload).
    :type reload: bool
    :return: The last command's last output line.
    :rtype: str
    :raises ValueError: For a model in :data:`PROTECTED_MODELS`.
    """
    if model in PROTECTED_MODELS:
        raise ValueError(f"refusing to unload the production model {model}")
    proc = subprocess.run([str(lms), "unload", model], capture_output=True, text=True, timeout=120)
    if reload:
        proc = subprocess.run([str(lms), "load", model, "--identifier", model, "-c", str(LOAD_CONTEXT),
                               "--parallel", str(LOAD_PARALLEL), "--gpu", "max", "-y"],
                              capture_output=True, text=True, timeout=600)
    lines = (proc.stdout + proc.stderr).strip().splitlines()
    return lines[-1] if lines else ""


async def run(todo: Sequence[Page], model: str, out_dir: Path, key: str,
              settings: pr.ReaderSettings = pr.DEFAULT_SETTINGS, read: Optional[Reader] = None,
              min_free_gb: float = 15.0, unload_below_gb: float = 0.0, wait_s: float = 60.0,
              free_check: Optional[Callable[[], float]] = None, unload: Optional[Callable[[str], str]] = None,
              sleep: Optional[Callable[[float], Awaitable[Any]]] = None,
              log: Callable[[str], None] = print) -> Dict[str, int]:
    """Read the pages one at a time with the production reader and store each read.

    :param todo: ``(doc_id, image)`` pairs to read (see :func:`pending`).
    :type todo: Sequence[Page]
    :param model: LM Studio model id.
    :type model: str
    :param out_dir: Output root.
    :type out_dir: Path
    :param key: Output key.
    :type key: str
    :param settings: Reader settings (default: production).
    :type settings: pr.ReaderSettings
    :param read: ``async (model, image, settings) -> result`` (default :func:`page_reader.read_page`).
    :type read: Optional[Reader]
    :param min_free_gb: Disk floor; the run waits below it.
    :type min_free_gb: float
    :param unload_below_gb: Unload the model when free disk falls under this (0: never).
    :type unload_below_gb: float
    :param wait_s: Seconds between disk checks while waiting.
    :type wait_s: float
    :param free_check: ``() -> free GiB`` (default :func:`free_disk_gb`).
    :type free_check: Optional[Callable[[], float]]
    :param unload: ``model -> output`` (default :func:`unload_own_model`).
    :type unload: Optional[Callable[[str], str]]
    :param sleep: ``async (seconds)`` (default :func:`asyncio.sleep`).
    :type sleep: Optional[Callable[[float], Awaitable[Any]]]
    :param log: Progress printer.
    :type log: Callable[[str], None]
    :return: Counts: ``stored``, ``failed``, ``blank``, ``length`` (final finish hit the budget), ``retried``,
        ``looped``, ``unloads``.
    :rtype: Dict[str, int]
    :raises RuntimeError: After :data:`MAX_CONSECUTIVE_FAILURES` failed requests in a row.
    """
    read = read or pr.read_page
    free_check = free_check or free_disk_gb
    unload = unload or unload_own_model
    sleep = sleep or asyncio.sleep
    stats = {"stored": 0, "failed": 0, "blank": 0, "length": 0, "retried": 0, "looped": 0, "unloads": 0}
    streak = 0
    for i, (doc_id, image) in enumerate(todo, 1):
        if unload_below_gb and free_check() < unload_below_gb:
            log(f"  {free_check():.1f} GiB free < {unload_below_gb}: unloading {model}: {unload(model)}")
            stats["unloads"] += 1
        while free_check() < min_free_gb:
            log(f"  disk floor: {free_check():.1f} GiB free < {min_free_gb}; waiting")
            await sleep(wait_s)
        result = await read(model, str(image), settings)
        if not store(out_dir, doc_id, key, result):
            stats["failed"] += 1
            streak += 1
            log(f"  FAILED {doc_id} (attempts {result.get('attempts')})")
            if streak >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(f"{streak} failed requests in a row; is {model} served?")
            continue
        streak = 0
        stats["stored"] += 1
        stats["blank"] += bool(result.get("blank"))
        stats["length"] += result.get("finish_reason") == pr.LENGTH
        stats["retried"] += bool(result.get("retried"))
        stats["looped"] += bool(result.get("looped"))
        if i % 10 == 0 or i == len(todo):
            log(f"  {i}/{len(todo)} read; " + ", ".join(f"{k} {v}" for k, v in stats.items()))
    return stats


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command line.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :type argv: Optional[List[str]]
    :return: The parsed arguments.
    :rtype: argparse.Namespace
    """
    ap = argparse.ArgumentParser(description="Read a page benchmark with the production page reader.")
    ap.add_argument("model", help="LM Studio model id")
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC, help="benchmark JSON (default: PGP-131)")
    ap.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES,
                    help="folder of <doc_id>.jpg for docs without their own image")
    ap.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS, help="output root (<dir>/<doc_id>/<key>.txt)")
    ap.add_argument("--key-suffix", default=KEY_SUFFIX, help="appended to the model key")
    ap.add_argument("--limit", type=int, default=0, help="only the first N pages")
    ap.add_argument("--min-free-gb", type=float, default=15.0)
    ap.add_argument("--unload-below-gb", type=float, default=0.0)
    return ap.parse_args(argv)


async def amain(args: argparse.Namespace) -> Dict[str, int]:
    """Read the spec's pending pages and print the counts.

    :param args: :func:`parse_args` result.
    :type args: argparse.Namespace
    :return: The counts of :func:`run` plus ``pages``, ``skipped`` and ``missing_image``.
    :rtype: Dict[str, int]
    :raises RuntimeError: When LM Studio does not serve the model.
    """
    pages = load_pages(args.spec, args.images_dir)
    if args.limit:
        pages = pages[:args.limit]
    key = output_key(args.model, args.key_suffix)
    todo, done, missing = pending(pages, args.outputs_dir, key)
    for doc_id in missing:
        print(f"  MISSING IMAGE {doc_id}", flush=True)
    print(f"{len(pages)} pages, {done} already read, {len(todo)} to read with {args.model} as {key}", flush=True)
    stats: Dict[str, int] = {}
    if todo:
        served = await check_lm_studio_health()
        if args.model not in served:
            raise RuntimeError(f"{args.model} not served by LM Studio: {served}")
        stats = await run(todo, args.model, args.outputs_dir, key, min_free_gb=args.min_free_gb,
                          unload_below_gb=args.unload_below_gb, log=lambda m: print(m, flush=True))
    stats.update(pages=len(pages), skipped=done, missing_image=len(missing))
    print("stats " + json.dumps(stats), flush=True)
    return stats


if __name__ == "__main__":
    asyncio.run(amain(parse_args()))
