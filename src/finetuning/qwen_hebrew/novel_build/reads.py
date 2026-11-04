"""Stage 2 (reads) of the novel-rows build: OCR reads of the candidate photographs.

A read is evidence for placing a target side on a photograph (stage 2 sides); it never becomes a target. Every read
is one :class:`~.schema.ReadRecord` line of ``<work dir>/reads/reads.jsonl``, keyed by the photograph's sha256 and a
reader id:

* ``kraken:<model file stem>``: the line-recognition service (:func:`kraken_read`), cheap and good on Hebrew script;
* ``vlm:<model key>``: the production page reader (:func:`vlm_read`), for Arabic script and for photographs whose
  other reads are unusable;
* reads of the earlier legacy-store build (``kraken:...`` and ``gemini:<model>``) are brought in by
  :func:`import_legacy_reads`, so nothing already paid for is read twice.

:func:`run_reads` reads every photograph of the stage 1 file that this reader has not read yet, one request at a time,
and appends each record to the file the moment it exists (a flush per record). It is resumable: a re-run skips every
(sha256, reader) pair already in the file. A failed request is not a read: it goes to ``reads/reads_failures.jsonl``
with its error and the image stays unread for the next run. The run stops cleanly when the stop file exists (checked
before every image) and after five failures in a row.

A photograph that is a near-duplicate of an earlier photograph of the same candidate (64-bit perceptual hashes at most
:data:`NEAR_DUPLICATE_BITS` bits apart, ``--skip-near-duplicates``) is not read once that earlier one has a usable
read from any reader: the same leaf stored twice needs one read, not two. The rule is applied when the queue is made
and again before each read, so of two twins that are both unread at launch the first is read and the second skipped.
The number of photographs skipped either way is the summary key ``skipped_near_duplicate``.

The only try/except blocks of the module wrap the service request (:func:`kraken_read`) and the page-reader call
(:func:`vlm_read`): there a failure must be recorded as a status (``finish="error"``), not raised.

Usage::

    python -m src.finetuning.qwen_hebrew.novel_build.reads --work-dir WORK --import-legacy KRAKEN_CACHE \\
        [--gemini-index INDEX]
    python -m src.finetuning.qwen_hebrew.novel_build.reads --work-dir WORK --reader kraken --script hebrew
    python -m src.finetuning.qwen_hebrew.novel_build.reads --work-dir WORK --reader vlm --model KEY \\
        --script arabic --only-missing-usable [--skip-near-duplicates N]
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import functools
import hashlib
import json
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Awaitable, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import requests
from PIL import Image

from src.finetuning.qwen_hebrew.novel_build import paths, schema
from src.finetuning.qwen_hebrew.novel_build.schema import Candidate, ImageRef, ReadRecord
from src.models.ocr import page_reader

#: Line-transcription endpoint of the Kraken service (override with ``NOVEL_BUILD_KRAKEN_URL``).
KRAKEN_URL = os.environ.get("NOVEL_BUILD_KRAKEN_URL", "http://localhost:8002/transcribe_lines")
#: Recognition model as the service names it (override with ``NOVEL_BUILD_KRAKEN_MODEL``).
KRAKEN_MODEL = os.environ.get("NOVEL_BUILD_KRAKEN_MODEL", "/app/models/MiDRASH_Gen_01.mlmodel")
KRAKEN_TIMEOUT_S = 600.0
LONG_SIDE = 2000                    # long side of the copy sent to Kraken, in pixels
JPEG_QUALITY = 90
MIN_SCRIPT_LETTERS = 20             # a read with fewer Hebrew plus Arabic letters is unusable
MAX_CONSECUTIVE_FAILURES = 5
#: Two photographs of one candidate whose 64-bit perceptual hashes differ in at most this many bits are near-duplicates.
NEAR_DUPLICATE_BITS = 6
PROGRESS_EVERY = 25
READS_DIR = "reads"
READS_NAME = "reads.jsonl"
FAILURES_NAME = "reads_failures.jsonl"
STOP_NAME = "STOP"
READABLE_STATUSES = ("present", "fetched")      # ``too_small`` and every other status is never read
SCRIPT_CHOICES = ("all", "hebrew", "arabic")
READER_CHOICES = ("kraken", "vlm")
UNUSABLE_FINISH = frozenset({"looping", "length", "error", "blank"})
NO_TEXT = re.compile(r"^\W*NO\s+TEXT\W*$", re.IGNORECASE)      # a Gemini read that says the page has no text
GEMINI_FINISH = {"STOP": "ok", "MAX_TOKENS": "length"}          # every other Gemini finish reason is an error
#: Fields of a ``page_reader.read_page`` result kept in ``ReadRecord.meta``.
VLM_META_FIELDS = ("reason", "finish_reason", "first_finish_reason", "looped", "cut_at", "letters_removed",
                   "repeat_share", "attempts", "retried", "prompt", "max_tokens", "temperature", "model", "ink",
                   "prompt_tokens", "completion_tokens", "total_tokens", "rotation", "options")
#: Failures of the Kraken request or of preparing its upload that become an ``error`` record.
KRAKEN_ERRORS = (requests.RequestException, OSError, ValueError, Image.DecompressionBombError)

_HEBREW_LETTER = re.compile("[\u05d0-\u05ea]")
_ARABIC_LETTER = re.compile("[\u0621-\u064a\u0671-\u06d3]")

ReadFn = Callable[[Path], ReadRecord]
PageReader = Callable[..., Awaitable[Dict[str, Any]]]


# --- small helpers -----------------------------------------------------------------------------------------------------

def _count_letters(text: str) -> Tuple[int, int]:
    """Count Hebrew and Arabic letters (the same ranges as ``targets.count_letters``).

    :param text: Any text.
    :return: ``(hebrew letters U+05D0-05EA, arabic letters U+0621-064A and U+0671-06D3)``.
    """
    return len(_HEBREW_LETTER.findall(text)), len(_ARABIC_LETTER.findall(text))


def now_utc() -> str:
    """Current UTC time stamp.

    :return: ``YYYY-MM-DDTHH:MM:SSZ``.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path: Path) -> str:
    """Hex sha256 of a file's bytes (the photograph's identity everywhere in the build).

    :param path: The file.
    :return: Hex digest.
    :raises OSError: When the file cannot be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    """Round-trip a value through JSON so that numpy scalars and other odd types become plain data.

    :param value: A dict, list or scalar.
    :return: The same data as JSON-serialisable Python objects (unknown types become strings).
    """
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def make_reader_id(reader: str, model: str) -> str:
    """The reader id stored in ``ReadRecord.reader``.

    :param reader: ``kraken`` or ``vlm``.
    :param model: Kraken: the recognition model path (empty: the default model); vlm: the model key.
    :return: ``kraken:<model file stem>`` or ``vlm:<model key>``.
    :raises ValueError: For any other reader.
    """
    if reader == "kraken":
        return f"kraken:{PurePosixPath(model or KRAKEN_MODEL).stem}"
    if reader == "vlm":
        return f"vlm:{model}"
    raise ValueError(f"unknown reader {reader!r}; expected one of {READER_CHOICES}")


def reads_path(work_dir: Path) -> Path:
    """Location of the reads file.

    :param work_dir: The build's work dir.
    :return: ``<work dir>/reads/reads.jsonl`` (its parent may not exist yet).
    """
    return work_dir / READS_DIR / READS_NAME


def failures_path(work_dir: Path) -> Path:
    """Location of the failures file (requests that gave no read).

    :param work_dir: The build's work dir.
    :return: ``<work dir>/reads/reads_failures.jsonl`` (its parent may not exist yet).
    """
    return work_dir / READS_DIR / FAILURES_NAME


def default_stop_file(work_dir: Path) -> Path:
    """Default stop file: creating it ends a run cleanly before the next image.

    :param work_dir: The build's work dir.
    :return: ``<work dir>/reads/STOP``.
    """
    return work_dir / READS_DIR / STOP_NAME


def append_jsonl(path: Path, record: Any) -> None:
    """Append one record (dataclass or dict) as a JSON line and flush it to disk.

    :param path: The JSON-lines file (created with its parent dirs when absent).
    :param record: A :mod:`.schema` dataclass or a plain dict.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = schema.to_dict(record) if dataclasses.is_dataclass(record) else record
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        fh.flush()


def repair_tail(path: Path) -> int:
    """Drop an unterminated last line of a JSON-lines file.

    Records are written with their newline in one call, so a last line without one is a write that was cut off
    (a crash, a dropped network volume). Removing it keeps the file readable and the image unread for the next run.

    :param path: The JSON-lines file (nothing happens when it does not exist).
    :return: Number of bytes removed (0 for an intact file).
    """
    if not path.exists():
        return 0
    block = 1 << 16
    with path.open("rb+") as fh:
        size = fh.seek(0, os.SEEK_END)
        if size == 0:
            return 0
        fh.seek(size - 1)
        if fh.read(1) == b"\n":
            return 0
        end = size
        while end > 0:
            start = max(0, end - block)
            fh.seek(start)
            newline = fh.read(end - start).rfind(b"\n")
            if newline != -1:
                keep = start + newline + 1
                fh.truncate(keep)
                return size - keep
            end = start
        fh.truncate(0)
        return size


def load_reads(work_dir: Path) -> List[ReadRecord]:
    """All reads recorded so far.

    :param work_dir: The build's work dir.
    :return: The records of ``reads/reads.jsonl`` in file order (empty when the file does not exist).
    """
    path = reads_path(work_dir)
    return list(schema.read_jsonl(path, ReadRecord)) if path.exists() else []


def make_record(sha256: str, reader: str, text: str, *, finish: str, seconds: float, lines: Optional[Sequence[str]] = None,
                read_at: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> ReadRecord:
    """Build a :class:`~.schema.ReadRecord` and count its letters.

    :param sha256: The photograph's sha256.
    :param reader: Reader id.
    :param text: Read text (``""`` for none).
    :param finish: ``ok`` | ``length`` | ``blank`` | ``error`` | ``looping``.
    :param seconds: Wall-clock seconds of the request.
    :param lines: Read lines; None: the non-empty lines of ``text``.
    :param read_at: ISO timestamp; None: now.
    :param meta: Reader-specific extras.
    :return: The record (``n_lines`` is ``len(lines)``).
    """
    text = text or ""
    line_list = list(lines) if lines is not None else [line for line in text.splitlines() if line.strip()]
    hebrew, arabic = _count_letters(text)
    return ReadRecord(sha256=sha256, reader=reader, text=text, lines=line_list, n_lines=len(line_list),
                      letters_hebrew=hebrew, letters_arabic=arabic, finish=finish, seconds=seconds,
                      read_at=read_at or now_utc(), meta=dict(meta or {}))


def error_record(sha256: str, reader: str, error: BaseException, seconds: float) -> ReadRecord:
    """The record of a request that failed: no text, ``finish="error"``, the error in ``meta``.

    :param sha256: The photograph's sha256.
    :param reader: Reader id.
    :param error: What went wrong.
    :param seconds: Seconds spent before the failure.
    :return: A record whose ``meta["error"]`` is ``"<ExceptionType>: <message, 300 characters>"``.
    """
    return make_record(sha256, reader, "", finish="error", seconds=round(seconds, 1), lines=[],
                       meta={"error": f"{type(error).__name__}: {str(error)[:300]}"})


def is_usable(read: ReadRecord) -> bool:
    """Whether a read can serve as evidence for a side.

    :param read: A read.
    :return: False for a read with no lines, fewer than :data:`MIN_SCRIPT_LETTERS` Hebrew plus Arabic letters, or a
        ``finish`` of looping, length, blank or error.
    """
    letters = read.letters_hebrew + read.letters_arabic
    return read.n_lines > 0 and letters >= MIN_SCRIPT_LETTERS and read.finish not in UNUSABLE_FINISH


# --- legacy import -----------------------------------------------------------------------------------------------------

def _legacy_kraken_record(row: Mapping[str, Any]) -> ReadRecord:
    """A Kraken read of the earlier legacy-store build as a :class:`~.schema.ReadRecord`.

    :param row: A row of its Kraken read cache (``sha256``, ``lines`` of ``{text, bbox, confidence}``, ``seconds``,
        ``read_at``, ``model_path``, ``view``, ``prepared``, ``polygon_failures``, ...).
    :return: The record, ``reader`` = ``kraken:<model file stem>``, ``finish`` ``ok`` or ``blank`` (no text).
    """
    lines = row.get("lines") or []
    texts = [str(line.get("text") or "") for line in lines]
    text = "\n".join(texts)
    prepared = row.get("prepared") or {}
    size = [prepared.get("width"), prepared.get("height")] if prepared.get("width") else list(row.get("image_size") or [])
    model_path = row.get("model_path") or KRAKEN_MODEL
    meta = {"source": "legacy_import", "confidences": [line.get("confidence") for line in lines],
            "bboxes": [line.get("bbox") for line in lines], "prepared_size": size, "scale": prepared.get("scale"),
            "polygon_failures": row.get("polygon_failures"), "used_binarization": row.get("used_binarization"),
            "view": row.get("view"), "model_path": model_path, "canonical_id": row.get("canonical_id"),
            "file_name": row.get("file_name")}
    return make_record(row["sha256"], make_reader_id("kraken", model_path), text, lines=texts,
                       finish="ok" if text.strip() else "blank", seconds=float(row.get("seconds") or 0.0),
                       read_at=row.get("read_at"), meta=_jsonable(meta))


def _legacy_gemini_record(entry: Mapping[str, Any], body: Mapping[str, Any]) -> ReadRecord:
    """A Gemini read of the earlier legacy-store build as a :class:`~.schema.ReadRecord`.

    The marker ``NO TEXT`` (the model's statement that the page holds no text) becomes an empty ``blank`` read.

    :param entry: Its index row (``sha256``, ``canonical_id``, ``file_name``).
    :param body: Its per-key JSON (``text``, ``model``, ``finish``, ``secs``, ``read_at``, ``usage``, ...).
    :return: The record, ``reader`` = ``gemini:<model>``.
    """
    raw = str(body.get("text") or "")
    no_text = not raw.strip() or bool(NO_TEXT.match(raw.strip()))
    raw_finish = str(body.get("finish") or "")
    finish = GEMINI_FINISH.get(raw_finish.split(".")[-1].upper(), "error")
    if finish == "ok" and no_text:
        finish = "blank"
    meta = {"source": "legacy_import", "model": body.get("model"), "finish_reason": raw_finish, "usage": body.get("usage"),
            "cost_usd": body.get("cost_usd"), "settings": body.get("settings"), "prompt_sha1": body.get("prompt_sha1"),
            "no_text_marker": bool(raw.strip()) and no_text, "canonical_id": entry.get("canonical_id"),
            "file_name": entry.get("file_name")}
    return make_record(entry["sha256"], f"gemini:{body.get('model') or 'unknown'}", "" if no_text else raw, finish=finish,
                       seconds=float(body.get("secs") or 0.0), read_at=body.get("read_at"), meta=_jsonable(meta))


def _legacy_records(kraken_cache: Path, gemini_index: Optional[Path]) -> Iterator[ReadRecord]:
    """The records of both legacy caches: Kraken first, then Gemini (index rows without a key file are skipped).

    :param kraken_cache: The Kraken read cache (JSON lines).
    :param gemini_index: The Gemini ``index.jsonl`` (one ``<key>.json`` per row sits beside it), or None.
    :return: Iterator of records.
    """
    for row in schema.read_jsonl(kraken_cache):
        yield _legacy_kraken_record(row)
    if gemini_index is None:
        return
    skipped = 0
    for entry in schema.read_jsonl(gemini_index):
        key_file = gemini_index.parent / f"{entry['key']}.json"
        if not key_file.exists():
            skipped += 1
            continue
        yield _legacy_gemini_record(entry, json.loads(key_file.read_text(encoding="utf-8")))
    if skipped:
        print(f"{now_utc()} skipped {skipped} index rows without a key file", flush=True)


def import_legacy_reads(work_dir: Path, kraken_cache: Path, gemini_index: Optional[Path] = None) -> int:
    """Bring the reads of the earlier legacy-store build into this build's reads file.

    Kraken reads become ``kraken:<model file stem>`` records, Gemini reads ``gemini:<model>`` records; a record is
    appended unless its (sha256, reader) pair is already in ``reads/reads.jsonl``, so the import can be repeated.

    :param work_dir: The build's work dir.
    :param kraken_cache: The Kraken read cache of the earlier build (JSON lines: ``sha256``, ``lines`` of
        ``{text, bbox, confidence}``, ``n_lines``, ``seconds``, ``read_at``, ``model_path``, ``view``).
    :param gemini_index: Its Gemini ``index.jsonl`` (the ``<key>.json`` files sit beside it), or None.
    :return: Number of records appended.
    """
    target = reads_path(work_dir)
    repair_tail(target)
    seen = {(read.sha256, read.reader) for read in load_reads(work_dir)}
    imported = 0
    for record in _legacy_records(kraken_cache, gemini_index):
        key = (record.sha256, record.reader)
        if key in seen:
            continue
        seen.add(key)
        append_jsonl(target, record)
        imported += 1
    return imported


# --- the two readers ---------------------------------------------------------------------------------------------------

def prepare_copy(src: Path, dst: Path, long_side: int = LONG_SIDE) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Write the copy sent to Kraken: an RGB JPEG whose long side is at most ``long_side`` pixels.

    Same algorithm as ``side_match.prepare_image`` (RGB, Lanczos, quality 90), kept local so that reading images does
    not import the Anthropic client and the benchmark builder; the legacy Kraken reads were made from such copies.

    :param src: The photograph.
    :param dst: Where the copy goes (parent dirs are created).
    :param long_side: Long-side cap in pixels; smaller images are not enlarged.
    :return: ``((original width, height), (copy width, height))``.
    """
    with Image.open(src) as original:
        image = original.convert("RGB")
    original_size = image.size
    if max(image.size) > long_side:
        ratio = long_side / max(image.size)
        image = image.resize((max(1, round(image.width * ratio)), max(1, round(image.height * ratio))),
                             Image.Resampling.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    image.save(dst, format="JPEG", quality=JPEG_QUALITY)
    return original_size, image.size


def _kraken_lines(body: Any) -> List[Mapping[str, Any]]:
    """The line dicts of a ``transcribe_lines`` response body.

    :param body: The decoded JSON body.
    :return: ``body["lines"]`` (empty when absent).
    :raises ValueError: When the body or a line is not the expected shape.
    """
    if not isinstance(body, dict):
        raise ValueError(f"unexpected response body of type {type(body).__name__}")
    lines = body.get("lines") or []
    if not isinstance(lines, list) or not all(isinstance(line, dict) for line in lines):
        raise ValueError("response 'lines' is not a list of objects")
    return lines


def kraken_read(path: Path, *, url: Optional[str] = None, model_path: Optional[str] = None, long_side: int = LONG_SIDE,
                post: Optional[Callable[..., Any]] = None, timeout: float = KRAKEN_TIMEOUT_S) -> ReadRecord:
    """Read one photograph with the Kraken line-transcription service.

    A copy of the photograph is downscaled to ``long_side`` (see :func:`prepare_copy`) and posted as the multipart
    field ``image`` with the form field ``model_path``; the response's lines become the read, in service order. The
    request and the preparation of the upload are the only code wrapped in try/except: a failure there (connection,
    timeout, HTTP error status, unreadable image, malformed body) is recorded as ``finish="error"`` with the error in
    ``meta["error"]``, never raised, so the caller can log it and count it.

    :param path: The photograph (hashed before anything else; an unreadable file raises ``OSError``).
    :param url: Service endpoint; None: :data:`KRAKEN_URL` (env ``NOVEL_BUILD_KRAKEN_URL``).
    :param model_path: Recognition model as the service names it; None: :data:`KRAKEN_MODEL` (env
        ``NOVEL_BUILD_KRAKEN_MODEL``).
    :param long_side: Long-side cap of the copy sent, in pixels.
    :param post: ``post(url, data=..., files=..., timeout=...)`` returning a response; None: ``requests.post``.
    :param timeout: Request timeout in seconds.
    :return: The record, ``reader`` = ``kraken:<model file stem>``. ``finish`` is ``ok``, ``blank`` (no line, or only
        empty lines) or ``error``. ``meta`` holds ``confidences``, ``bboxes`` (in the copy's pixel frame),
        ``prepared_size`` (copy ``[width, height]``), ``scale`` (copy width / original width),
        ``polygon_failures``, ``used_binarization``, ``view`` and ``model_path``.
    """
    url = url or KRAKEN_URL
    model_path = model_path or KRAKEN_MODEL
    post = post or requests.post
    reader = make_reader_id("kraken", model_path)
    sha256 = file_sha256(path)
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / f"{path.stem}_{long_side}.jpg"
            original_size, prepared_size = prepare_copy(path, copy, long_side)
            with copy.open("rb") as fh:
                response = post(url, data={"model_path": model_path}, files={"image": fh}, timeout=timeout)
            response.raise_for_status()
            body = response.json()
            lines = _kraken_lines(body)
    except KRAKEN_ERRORS as error:
        return error_record(sha256, reader, error, time.perf_counter() - started)
    texts = [str(line.get("text") or "") for line in lines]
    text = "\n".join(texts)
    meta = {"confidences": [line.get("confidence") for line in lines], "bboxes": [line.get("bbox") for line in lines],
            "prepared_size": list(prepared_size), "scale": round(prepared_size[0] / original_size[0], 6),
            "polygon_failures": body.get("polygon_failures"), "used_binarization": body.get("used_binarization"),
            "view": body.get("view"), "model_path": model_path}
    return make_record(sha256, reader, text, lines=texts, finish="ok" if text.strip() else "blank",
                       seconds=round(time.perf_counter() - started, 1), meta=_jsonable(meta))


def vlm_finish(result: Mapping[str, Any]) -> str:
    """The ``ReadRecord.finish`` of a ``page_reader.read_page`` result.

    :param result: The result dict (``text``, ``reason``, ``blank``, ``finish_reason``, ``looped``, ``retried``).
    :return: ``error`` when the request failed (``reason`` ``failed`` or no text at all); ``blank`` for a page the ink
        gate skipped or a generation with nothing in it (``reason`` ``blank`` or ``empty``); ``looping`` for a
        generation the token budget cut after the length retry (``finish_reason`` ``length`` and ``retried``) and for
        any read the loop guard flagged (``looped``); ``length`` for a cut generation that was not retried; else ``ok``.
    """
    reason = result.get("reason")
    if reason == page_reader.REASON_FAILED or result.get("text") is None:
        return "error"
    if reason in (page_reader.REASON_BLANK, page_reader.REASON_EMPTY) or result.get("blank"):
        return "blank"
    if result.get("finish_reason") == page_reader.LENGTH:
        return "looping" if result.get("retried") else "length"
    return "looping" if result.get("looped") else "ok"


def vlm_read(path: Path, *, model_key: str, read_page: Optional[PageReader] = None) -> ReadRecord:
    """Read one photograph with the production page reader (``src.models.ocr.page_reader.read_page``).

    The reader already applies the script-neutral prompt, the 1,200-token budget, the retry after a length cut, the
    ink gate for blank pages and the model server's yield gate; this function runs it with ``asyncio.run`` and maps
    its result dict to a record (see :func:`vlm_finish`). The call is the only code wrapped in try/except: any
    exception of the reader (an unreadable image, a broken response) is recorded as ``finish="error"`` with the error
    in ``meta["error"]``, never raised; ``KeyboardInterrupt`` still stops the run.

    :param path: The photograph (hashed before anything else; an unreadable file raises ``OSError``).
    :param model_key: Model key of the vision-language model.
    :param read_page: ``async read_page(model_key, image_path)`` returning the result dict; None:
        ``page_reader.read_page``.
    :return: The record, ``reader`` = ``vlm:<model key>``, ``text`` the stored read (``""`` for a blank or empty page),
        ``lines`` its non-empty lines, ``meta`` the fields of :data:`VLM_META_FIELDS` the result has.
    """
    read_page = read_page or page_reader.read_page
    reader = make_reader_id("vlm", model_key)
    sha256 = file_sha256(path)
    started = time.perf_counter()
    try:
        result = asyncio.run(read_page(model_key, str(path)))
    except Exception as error:      # any failure of the reader is a status of this image, not a crash of the batch
        return error_record(sha256, reader, error, time.perf_counter() - started)
    meta = _jsonable({key: result[key] for key in VLM_META_FIELDS if key in result})
    return make_record(sha256, reader, result.get("text") or "", finish=vlm_finish(result),
                       seconds=round(time.perf_counter() - started, 1), meta=meta)


# --- selection and the run ---------------------------------------------------------------------------------------------

def _script_selected(candidate_script: str, wanted: str) -> bool:
    """Whether a candidate's ledger script is read in a run for ``wanted``.

    :param candidate_script: Ledger script (``hebrew`` | ``arabic`` | ``mixed`` | ``none``).
    :param wanted: ``all`` | ``hebrew`` | ``arabic``.
    :return: True for ``all``; ``hebrew`` and ``arabic`` also take ``mixed``.
    """
    if wanted == "all":
        return True
    return (candidate_script or "").lower() in (wanted, "mixed")


def _phash_distance(a: str, b: str) -> int:
    """Hamming distance, in bits, of two perceptual hashes.

    The same measure as ``images.phash_distance``, kept local so that reading images does not import the stage 1
    builder (and, through it, the merging and shelf-mark modules).

    :param a: 64-bit hash as hex.
    :param b: 64-bit hash as hex.
    :return: Number of bits in which they differ.
    :raises ValueError: When either is not hex.
    """
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _is_near_duplicate(phash: Optional[str], covered: Sequence[str], max_bits: Optional[int]) -> bool:
    """Whether a photograph's perceptual hash is within ``max_bits`` bits of one of the ``covered`` hashes.

    :param phash: The photograph's hash; None or empty: never a near-duplicate.
    :param covered: Hashes of the photographs that already have a usable read.
    :param max_bits: Largest distance in bits that still counts; None or 0: the rule is off.
    :return: True when any hash of ``covered`` is at most ``max_bits`` bits away from ``phash``.
    """
    if not phash or not max_bits:
        return False
    return any(_phash_distance(phash, other) <= max_bits for other in covered)


def _readable_photographs(cand: Candidate) -> Iterator[Tuple[ImageRef, str]]:
    """The photographs of a candidate that can be read, in candidate order, each with its sha256.

    :param cand: A candidate.
    :return: Iterator of ``(image, sha256)`` for every image whose status is ``present`` or ``fetched`` and that has a
        path; an image without a recorded sha256 is hashed from its file here (and left out when the file is missing).
    """
    for image in cand.images:
        if image.status not in READABLE_STATUSES or not image.path:
            continue
        sha256 = image.sha256
        if not sha256:
            if not Path(image.path).is_file():
                continue
            sha256 = file_sha256(Path(image.path))
        yield image, sha256


def select_images(cands: Iterable[Candidate], reads: Iterable[ReadRecord], *, script: str = "all",
                  only_missing_usable: bool = False, reader_id: Optional[str] = None,
                  skip_near_duplicates: Optional[int] = NEAR_DUPLICATE_BITS,
                  counts: Optional[Counter] = None) -> List[ImageRef]:
    """The photographs a reader still has to read, in candidate order.

    Images count when their status is ``present`` or ``fetched`` (never ``too_small``, ``listed``, ``missing``, ...)
    and they have a path; an image without a sha256 is hashed here (the returned copy carries it); an image that
    several candidates list is returned once, by its first occurrence.

    The near-duplicate rule is applied last, one candidate at a time and in candidate order: a photograph is dropped
    when an earlier photograph of the same candidate is within ``skip_near_duplicates`` bits of it and already has a
    usable read from any reader (:func:`is_usable`). The rule only fires once such a read exists, so the first
    photograph of a near-duplicate pair is always read normally, and it only drops what the other rules would have
    queued. A photograph dropped from one candidate stays eligible for any other candidate that lists it.

    :param cands: Candidates with completed ``images`` (the stage 1 file).
    :param reads: Reads of every reader so far.
    :param script: ``all`` | ``hebrew`` | ``arabic``: candidates whose ledger script matches (``mixed`` matches both
        ``hebrew`` and ``arabic``).
    :param only_missing_usable: Keep only images without a usable read from any reader (:func:`is_usable`): no read
        at all, or only reads with no lines, fewer than 20 script letters, or a finish of looping/length/error.
    :param reader_id: The reader about to run (:func:`make_reader_id`): its images are skipped once it has a read of
        them. None: no per-reader skip.
    :param skip_near_duplicates: Largest Hamming distance, in bits of the 64-bit hex ``ImageRef.phash``, at which two
        photographs of one candidate are near-duplicates. None or 0: the rule is off. A photograph without a
        ``phash`` is never dropped by it and never stands in for another.
    :param counts: Counter that receives ``skipped_near_duplicate``, the number of distinct photographs the rule kept
        out of the result (added to what the counter holds). None: the number is not reported.
    :return: The images to read.
    :raises ValueError: For an unknown ``script``, a negative ``skip_near_duplicates``, or a ``phash`` that is not hex
        (only compared when the rule is on).
    """
    if script not in SCRIPT_CHOICES:
        raise ValueError(f"unknown script {script!r}; expected one of {SCRIPT_CHOICES}")
    if skip_near_duplicates is not None and skip_near_duplicates < 0:
        raise ValueError(f"skip_near_duplicates must be None, 0 or a positive bit distance, not {skip_near_duplicates}")
    by_sha: Dict[str, List[ReadRecord]] = defaultdict(list)
    for read in reads:
        by_sha[read.sha256].append(read)
    selected: List[ImageRef] = []
    seen = set()
    near_duplicates = set()         # sha256 of the photographs the rule dropped and no other candidate then selected
    for cand in cands:
        if not _script_selected(cand.script, script):
            continue
        covered: List[str] = []     # hashes of this candidate's earlier photographs that already have a usable read
        for image, sha256 in _readable_photographs(cand):
            existing = by_sha.get(sha256, [])
            usable = any(is_usable(read) for read in existing)
            near_duplicate = _is_near_duplicate(image.phash, covered, skip_near_duplicates)
            if usable and image.phash:
                covered.append(image.phash)
            if sha256 in seen:
                continue
            seen.add(sha256)
            if reader_id is not None and any(read.reader == reader_id for read in existing):
                continue
            if only_missing_usable and usable:
                continue
            if near_duplicate:
                seen.discard(sha256)        # only this candidate has a read twin: another one may still need it read
                near_duplicates.add(sha256)
                continue
            near_duplicates.discard(sha256)
            selected.append(image if image.sha256 else dataclasses.replace(image, sha256=sha256))
    if counts is not None:
        counts["skipped_near_duplicate"] += len(near_duplicates)
    return selected


def _near_duplicate_peers(cands: Iterable[Candidate], script: str,
                          max_bits: Optional[int]) -> Dict[str, List[List[str]]]:
    """For every readable photograph, which earlier photographs of its candidates are near-duplicates of it.

    The input of the read-time form of the near-duplicate rule (see :func:`run_reads`). It scans the candidates that
    :func:`select_images` looks at for ``script``, in the same order, so both agree on what "earlier" means.

    :param cands: Candidates (the stage 1 file).
    :param script: ``all`` | ``hebrew`` | ``arabic`` (see :func:`select_images`).
    :param max_bits: Largest Hamming distance, in bits of the ``phash``, that counts; None or 0: nothing has a peer.
    :return: ``{sha256: [peers of its first listing, peers of its second listing, ...]}``: one list per candidate that
        lists the photograph, holding the sha256 of each earlier photograph of that candidate whose ``phash`` is within
        ``max_bits`` bits of its own (itself excluded). A listing of a photograph without a ``phash`` gets an empty
        list: nothing can cover it. Empty when ``max_bits`` is off.
    """
    peers: Dict[str, List[List[str]]] = defaultdict(list)
    if not max_bits:
        return peers
    for cand in cands:
        if not _script_selected(cand.script, script):
            continue
        earlier: List[Tuple[str, str]] = []         # (sha256, phash) of its earlier photographs that have a phash
        for image, sha256 in _readable_photographs(cand):
            if not image.phash:
                peers[sha256].append([])
                continue
            peers[sha256].append([other for other, phash in earlier
                                  if other != sha256 and _phash_distance(image.phash, phash) <= max_bits])
            earlier.append((sha256, image.phash))
    return peers


def _has_usable_peer(image: ImageRef, peers: Mapping[str, Sequence[Sequence[str]]], usable: Set[str]) -> bool:
    """Whether every candidate that lists a photograph has an earlier near-duplicate of it with a usable read.

    A photograph several candidates list is covered only when each of them has such a twin: one candidate without one
    still needs the photograph read.

    :param image: A queued image (``sha256`` set).
    :param peers: The result of :func:`_near_duplicate_peers`.
    :param usable: sha256 of every photograph with a usable read so far, from any reader, this run's reads included.
    :return: False for a photograph that ``peers`` does not know and for one with a listing that has no usable twin.
    """
    listings = peers.get(image.sha256 or "", ())
    return bool(listings) and all(any(peer in usable for peer in listing) for listing in listings)


def _attempt(image: ImageRef, read_fn: ReadFn) -> Tuple[Optional[ReadRecord], Optional[str]]:
    """Read one selected image and judge the outcome.

    :param image: A selected image (``path`` and ``sha256`` set).
    :param read_fn: ``read_fn(path) -> ReadRecord``.
    :return: ``(record, None)`` for a read; ``(None, error text)`` when the file is missing, the reader returned an
        ``error`` record, or the record's sha256 disagrees with the stage 1 file (a stale inventory).
    """
    path = Path(image.path or "")
    if not path.is_file():
        return None, "image_missing"
    record = read_fn(path)
    if record.finish == "error":
        return None, str(record.meta.get("error") or "error")
    if record.sha256 != image.sha256:
        return None, f"sha256_mismatch: stage 1 lists {image.sha256}, the file hashes to {record.sha256}"
    return record, None


def run_reads(work_dir: Path, *, reader: str, model: str, read_fn: ReadFn, script: str = "all",
              limit: Optional[int] = None, stop_file: Optional[Path] = None, only_missing_usable: bool = False,
              skip_near_duplicates: Optional[int] = NEAR_DUPLICATE_BITS) -> Dict[str, int]:
    """Read the photographs of the stage 1 file that this reader has not read yet.

    Records are appended to ``reads/reads.jsonl`` one at a time (flushed), so an interrupted run loses nothing. A
    failure (an ``error`` record, a missing file, a sha256 that disagrees with the stage 1 file) goes to
    ``reads/reads_failures.jsonl`` and leaves the image unread. The run ends after five failures in a row, when
    ``stop_file`` exists (checked before each image), or when the queue is empty. Progress is printed every 25 reads.

    The near-duplicate rule is applied twice. :func:`select_images` applies it when the queue is made, which removes
    the photographs whose twin already has a usable read. Most twins are still unread then, so it is applied again
    before each read: the photograph is skipped (not read, not a failure, counted) when, in every candidate that lists
    it, an earlier near-duplicate has a usable read by now, the reads of this run included. The queue keeps candidate
    order, so of two twins unread at launch the first is read and the second skipped, unless that first read is
    unusable or fails.

    :param work_dir: The build's work dir (stage 1 file in, reads out).
    :param reader: ``kraken`` or ``vlm``.
    :param model: Kraken: recognition model path (empty: default); vlm: model key. With ``reader`` it makes the reader
        id that decides which images count as read.
    :param read_fn: ``read_fn(path) -> ReadRecord``, e.g. a :func:`kraken_read` or :func:`vlm_read` partial.
    :param script: ``all`` | ``hebrew`` | ``arabic`` (see :func:`select_images`).
    :param limit: Attempt at most this many images in this run; None: all. The queue is cut to this length when it is
        made, so a photograph skipped at read time still uses up one place.
    :param stop_file: Stop before the next image when this file exists; None: no stop file.
    :param only_missing_usable: See :func:`select_images`.
    :param skip_near_duplicates: See :func:`select_images`; None or 0 turns both applications of the rule off.
    :return: Counts: ``queued``, ``skipped_near_duplicate`` (photographs the near-duplicate rule kept out of the queue
        plus those it skipped at read time), ``read``, ``failed``, ``finish_<finish>`` per finish seen,
        ``stopped_by_stop_file`` and ``stopped_by_failures`` (1 when that ended the run, else 0).
    :raises FileNotFoundError: When the stage 1 file does not exist.
    """
    stage1 = paths.stage_path(work_dir, "images")
    if not stage1.exists():
        raise FileNotFoundError(f"stage 1 file not found: {stage1} (run the images stage first)")
    reader_id = make_reader_id(reader, model)
    reads_file, failures_file = reads_path(work_dir), failures_path(work_dir)
    repair_tail(reads_file)
    repair_tail(failures_file)
    existing = load_reads(work_dir)
    skips: Counter = Counter()
    queue = select_images(schema.read_jsonl(stage1, Candidate), existing, script=script,
                          only_missing_usable=only_missing_usable, reader_id=reader_id,
                          skip_near_duplicates=skip_near_duplicates, counts=skips)
    if limit is not None:
        queue = queue[: max(0, limit)]
    peers = _near_duplicate_peers(schema.read_jsonl(stage1, Candidate), script, skip_near_duplicates)
    usable = {read.sha256 for read in existing if is_usable(read)}      # grows with every usable read of this run
    counts: Counter = Counter({"queued": len(queue), "skipped_near_duplicate": skips["skipped_near_duplicate"],
                               "read": 0, "failed": 0, "stopped_by_stop_file": 0, "stopped_by_failures": 0})
    print(f"{now_utc()} {reader_id}: {len(queue)} images queued ({script}, only_missing_usable={only_missing_usable}, "
          f"skip_near_duplicates={skip_near_duplicates}: {counts['skipped_near_duplicate']} skipped as "
          f"near-duplicates at launch)", flush=True)
    consecutive, seconds = 0, 0.0
    for image in queue:
        if stop_file is not None and stop_file.exists():
            counts["stopped_by_stop_file"] = 1
            print(f"{now_utc()} stopping: stop file {stop_file} exists", flush=True)
            break
        if _has_usable_peer(image, peers, usable):
            counts["skipped_near_duplicate"] += 1
            print(f"{now_utc()} skipped {image.path}: a near-duplicate of it now has a usable read", flush=True)
            continue
        record, error = _attempt(image, read_fn)
        if record is None:
            append_jsonl(failures_file, {"sha256": image.sha256, "reader": reader_id, "path": image.path, "error": error,
                                         "at": now_utc()})
            counts["failed"] += 1
            consecutive += 1
            print(f"{now_utc()} FAILED {image.path}: {error}", flush=True)
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                counts["stopped_by_failures"] = 1
                print(f"{now_utc()} stopping: {consecutive} consecutive failures", flush=True)
                break
            continue
        consecutive = 0
        append_jsonl(reads_file, record)
        if is_usable(record):
            usable.add(record.sha256)
        counts["read"] += 1
        counts[f"finish_{record.finish}"] += 1
        seconds += record.seconds
        if counts["read"] % PROGRESS_EVERY == 0:
            print(f"{now_utc()} {counts['read']} read this run of {len(queue)}, {counts['failed']} failed, mean "
                  f"{seconds / counts['read']:.1f} s/image, last {record.n_lines} lines ({record.finish})", flush=True)
    return dict(counts)


# --- command line ------------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """The command-line parser.

    :return: Parser with ``--work-dir --reader --model --script --limit --stop-file --only-missing-usable
        --skip-near-duplicates --import-legacy --gemini-index``.
    """
    parser = argparse.ArgumentParser(description="Stage 2 of the novel-rows build: OCR reads of the candidate photographs "
                                                 "(evidence for side placement only).")
    parser.add_argument("--work-dir", type=Path, default=paths.DEFAULT_WORK_DIR,
                        help="work dir of the build (default: the site work dir)")
    parser.add_argument("--reader", choices=READER_CHOICES, default=None,
                        help="kraken: line recognition service; vlm: production page reader")
    parser.add_argument("--model", default=None,
                        help="kraken: recognition model as the service names it (default: the site model); "
                             "vlm: model key (required)")
    parser.add_argument("--script", choices=SCRIPT_CHOICES, default="all",
                        help="read candidates of this ledger script (mixed counts for hebrew and arabic)")
    parser.add_argument("--limit", type=int, default=None, help="attempt at most N images in this run")
    parser.add_argument("--stop-file", type=Path, default=None,
                        help="stop cleanly before the next image when this file exists (default: <work dir>/reads/STOP)")
    parser.add_argument("--only-missing-usable", action="store_true",
                        help="read only images without a usable read from any reader")
    parser.add_argument("--skip-near-duplicates", type=int, default=NEAR_DUPLICATE_BITS, metavar="N",
                        help="do not read a photograph whose 64-bit phash is within N bits of an earlier photograph of "
                             "the same candidate that already has a usable read (default: %(default)s; 0 disables)")
    parser.add_argument("--import-legacy", type=Path, default=None, metavar="KRAKEN_CACHE",
                        help="import the Kraken read cache of the earlier legacy-store build (JSON lines) first")
    parser.add_argument("--gemini-index", type=Path, default=None,
                        help="with --import-legacy: also import its Gemini reads from this index.jsonl")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Command-line entry point: import legacy reads and/or run one reader over the stage 1 photographs.

    :param argv: Arguments; None: ``sys.argv``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.gemini_index is not None and args.import_legacy is None:
        parser.error("--gemini-index needs --import-legacy")
    if args.reader is None and args.import_legacy is None:
        parser.error("give --reader to read, or --import-legacy to import earlier reads")
    if args.reader == "vlm" and not args.model:
        parser.error("--model (the model key) is required with --reader vlm")
    if args.skip_near_duplicates < 0:
        parser.error("--skip-near-duplicates takes a number of bits: 0 (off) or more")
    if args.import_legacy is not None:
        imported = import_legacy_reads(args.work_dir, args.import_legacy, args.gemini_index)
        print(f"{now_utc()} imported {imported} legacy reads into {reads_path(args.work_dir)}", flush=True)
    if args.reader is None:
        return
    model = args.model or (KRAKEN_MODEL if args.reader == "kraken" else "")
    if args.reader == "kraken":
        read_fn: ReadFn = functools.partial(kraken_read, url=KRAKEN_URL, model_path=model)
    else:
        read_fn = functools.partial(vlm_read, model_key=model)
    counts = run_reads(args.work_dir, reader=args.reader, model=model, read_fn=read_fn, script=args.script,
                       limit=args.limit, stop_file=args.stop_file or default_stop_file(args.work_dir),
                       only_missing_usable=args.only_missing_usable, skip_near_duplicates=args.skip_near_duplicates)
    print(json.dumps(counts, indent=1), flush=True)


if __name__ == "__main__":
    main()
