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

Finish of a page-reader read (:func:`vlm_finish`): ``looping`` only when the reader's loop detector found a loop;
a generation the 1,200-token budget cut without one (a long page) is ``length``. A ``length`` read is usable evidence
(:func:`is_usable`): a truncated read still places a side. ``looping``, ``blank`` and ``error`` reads are not.

Fallback selections (a second reader for what the first left unplaced or doubtful): ``--only-missing-usable`` reads
photographs without a usable read from any reader; ``--only-unplaced SIDES`` reads the photographs of candidates with
at least one side that stage 2 sides did not accept, whatever reads they already have (a usable read that placed
nothing does not block a second read); ``--only-coverage-review QUALITY`` reads, for each side whose stage 3 quality
reasons ask for it (:data:`COVERAGE_REVIEW_REASONS`), the side's accepted photograph, or every present photograph of
its candidate when the sides file has none for it. Given together, the queue is the union of all of them.

Near-duplicates (opt-in, ``--skip-near-duplicates N``; :data:`NEAR_DUPLICATE_BITS` is 0, off): a photograph whose
64-bit perceptual hash is at most N bits from an earlier photograph of the same candidate is not read once that earlier
one has a usable read from any reader. It is off by default because two faces of one leaf, photographed alike, can be
perceptual-hash twins: skipping one would leave a side without its photograph's read. The rule is applied when the
queue is made and again before each read, so of two twins that are both unread at launch the first is read and the
second skipped. The number of photographs skipped either way is the summary key ``skipped_near_duplicate``.

Claims (optional, ``--claim-file PATH``): jobs that read into separate work dirs (one snapshot per job) share progress
through one append-only claim log (:class:`ClaimLog`): a job skips a photograph that another job claimed for the same
reader within the last 24 hours (``--ignore-claims`` reads it anyway), and claims each photograph before requesting it.

The only try/except blocks of the module wrap the service request (:func:`kraken_read`) and the page-reader call
(:func:`vlm_read`), where a failure must be recorded as a status (``finish="error"``), not raised, and the parse of one
claim line (:func:`_parse_claim`), where a line garbled by a concurrent writer is skipped and counted, not fatal.

Usage::

    python -m src.finetuning.qwen_hebrew.novel_build.reads --work-dir WORK --import-legacy KRAKEN_CACHE \\
        [--gemini-index INDEX]
    python -m src.finetuning.qwen_hebrew.novel_build.reads --work-dir WORK --reader kraken --script hebrew
    python -m src.finetuning.qwen_hebrew.novel_build.reads --work-dir WORK --reader vlm --model KEY \\
        --script arabic --only-missing-usable [--only-unplaced SIDES] [--skip-near-duplicates N] \\
        [--claim-file CLAIMS [--job-id NAME] [--ignore-claims]]
    python -m src.finetuning.qwen_hebrew.novel_build.reads --work-dir WORK --reader vlm --model KEY \\
        --only-coverage-review QUALITY (--only-unplaced SIDES | --sides SIDES)
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
#: 0 turns the rule off (the default: two faces of one leaf can be perceptual-hash twins); ``--skip-near-duplicates``.
NEAR_DUPLICATE_BITS = 0
PROGRESS_EVERY = 25
READS_DIR = "reads"
READS_NAME = "reads.jsonl"
FAILURES_NAME = "reads_failures.jsonl"
STOP_NAME = "STOP"
READABLE_STATUSES = ("present", "fetched")      # ``too_small`` and every other status is never read
SCRIPT_CHOICES = ("all", "hebrew", "arabic")
READER_CHOICES = ("kraken", "vlm")
#: Finishes that make a read unusable; ``length`` (a read the token budget cut, no loop found) is usable evidence.
UNUSABLE_FINISH = frozenset({"looping", "error", "blank"})
ACCEPTED = "accepted"                           # the ``SideDecision.outcome`` of a placed side
#: Stage 3 quality reasons (coverage gates and the flag) that ask for a second read of a side's photograph.
COVERAGE_REVIEW_REASONS = frozenset({"target_exceeds_read_2_5x", "partial_image_for_target", "read_exceeds_target_1_5x",
                                     "kraken_lines_over_1_5x", "coverage_second_read_wanted"})
NO_TEXT = re.compile(r"^\W*NO\s+TEXT\W*$", re.IGNORECASE)      # a Gemini read that says the page has no text
GEMINI_FINISH = {"STOP": "ok", "MAX_TOKENS": "length"}          # every other Gemini finish reason is an error
#: Fields of a ``page_reader.read_page`` result kept in ``ReadRecord.meta``.
VLM_META_FIELDS = ("reason", "finish_reason", "first_finish_reason", "looped", "cut_at", "letters_removed",
                   "repeat_share", "unit_repeat_share", "attempts", "retried", "prompt", "max_tokens", "temperature",
                   "model", "ink", "prompt_tokens", "completion_tokens", "total_tokens", "rotation", "options")
#: A claim older than this (seconds) no longer holds a photograph: the job that made it is presumed gone.
CLAIM_TTL_S = 24 * 3600.0
CLAIM_STAMP = "%Y-%m-%dT%H:%M:%SZ"              # ``claimed_at`` format (the format of :func:`now_utc`)
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


def loop_detected(fields: Mapping[str, Any]) -> bool:
    """Whether the page reader's loop detector found a loop in a read, as opposed to the token budget merely cutting it.

    The reader marks every generation the budget cut ``looped`` (``page_reader.settle_read``: a ``length`` finish is
    treated as a loop), so for such a read the flag alone says nothing about a loop. The detector's own verdict is
    then its evidence: the gateless loop guard found where a loop starts (``cut_at`` set), or the repeat share of the
    generation exceeds the production loop gate (``repeat_share``, or ``unit_repeat_share`` of the marker detector,
    above ``page_reader.DEFAULT_SETTINGS.loop_gate``). For any other finish the ``looped`` flag is the detector's.

    :param fields: A ``page_reader.read_page`` result, or the ``meta`` of a stored page-reader read.
    :return: False when ``looped`` is false; True for a looped read that the budget did not cut; for a cut read, True
        when ``cut_at`` is set or a repeat share exceeds the gate, and also when ``cut_at`` or ``repeat_share`` is not
        recorded at all (no evidence against the flag).
    """
    if not fields.get("looped"):
        return False
    if fields.get("finish_reason") != page_reader.LENGTH:
        return True
    if "cut_at" not in fields or "repeat_share" not in fields:
        return True
    gate = page_reader.DEFAULT_SETTINGS.loop_gate
    return (fields.get("cut_at") is not None or float(fields.get("repeat_share") or 0.0) > gate
            or float(fields.get("unit_repeat_share") or 0.0) > gate)


def effective_finish(read: ReadRecord) -> str:
    """The finish a read is judged by: its ``finish``, with page-reader ``looping`` records re-judged from ``meta``.

    Before the mapping of :func:`vlm_finish` was corrected, a page-reader read that the token budget cut twice (first
    request and retry) was stored as ``looping`` even when no loop was found. Such a record is re-judged from the
    reader's fields in its ``meta`` (``finish_reason`` and ``looped``; ``cut_at`` and ``repeat_share`` when ``looped``
    is set): a ``length`` finish without a detected loop (:func:`loop_detected`) counts as ``length``.

    :param read: A read.
    :return: ``length`` for a ``looping`` record whose ``meta`` shows a length cut and no loop; else ``read.finish``
        (a ``looping`` record without the reader's fields in ``meta`` stays ``looping``).
    """
    if read.finish != "looping" or "finish_reason" not in read.meta or "looped" not in read.meta:
        return read.finish
    if read.meta.get("finish_reason") == page_reader.LENGTH and not loop_detected(read.meta):
        return "length"
    return read.finish


def is_usable(read: ReadRecord) -> bool:
    """Whether a read can serve as evidence for a side.

    A ``length`` read (the token budget cut it, the loop detector found no loop) is usable: a truncated read is still
    valid placement evidence. A ``looping`` record already on disk is judged by :func:`effective_finish`: when its
    ``meta`` carries the page reader's fields and they show a length cut without a loop (``looped`` false, or set only
    by the cut: no ``cut_at`` and a repeat share within the gate), it is treated as ``length``, hence usable.

    :param read: A read.
    :return: False for a read with no lines, fewer than :data:`MIN_SCRIPT_LETTERS` Hebrew plus Arabic letters, or an
        effective finish of looping, blank or error; True otherwise (``ok`` and ``length``).
    """
    letters = read.letters_hebrew + read.letters_arabic
    return read.n_lines > 0 and letters >= MIN_SCRIPT_LETTERS and effective_finish(read) not in UNUSABLE_FINISH


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

    Whether the length retry ran does not matter: a long page whose first request and retry both reach the token
    budget is ``length``, not ``looping``.

    :param result: The result dict (``text``, ``reason``, ``blank``, ``finish_reason``, ``looped``, ``cut_at``,
        ``repeat_share``).
    :return: ``error`` when the request failed (``reason`` ``failed`` or no text at all); ``blank`` for a page the ink
        gate skipped or a generation with nothing in it (``reason`` ``blank`` or ``empty``); ``looping`` when the
        reader's loop detector found a loop (:func:`loop_detected`); ``length`` for a generation the token budget cut
        without one (``finish_reason`` ``length``); else ``ok``.
    """
    reason = result.get("reason")
    if reason == page_reader.REASON_FAILED or result.get("text") is None:
        return "error"
    if reason in (page_reader.REASON_BLANK, page_reader.REASON_EMPTY) or result.get("blank"):
        return "blank"
    if loop_detected(result):
        return "looping"
    return "length" if result.get("finish_reason") == page_reader.LENGTH else "ok"


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


def unplaced_candidates(sides_path: Path) -> Set[str]:
    """Candidates that stage 2 sides left with at least one side not placed.

    :param sides_path: A stage 2 sides file (:class:`~.schema.SideDecision` JSON lines).
    :return: ``row_id`` of every candidate with a side whose ``outcome`` is not ``accepted`` (``needs_review``,
        ``unassigned`` or ``dropped``). A candidate absent from the file (no side to place) is not included.
    :raises FileNotFoundError: When the file does not exist.
    """
    return {decision.row_id for decision in schema.read_jsonl(sides_path, schema.SideDecision)
            if decision.outcome != ACCEPTED}


def coverage_review_sides(quality_path: Path, sides_path: Path) -> Dict[str, List[Optional[str]]]:
    """Sides whose stage 3 quality reasons ask for a second read, with the photograph each was accepted on.

    :param quality_path: A stage 3 quality file (:class:`~.schema.QualityResult` JSON lines).
    :param sides_path: The stage 2 sides file the quality stage judged (:class:`~.schema.SideDecision` JSON lines).
    :return: ``{row_id: [accepted photograph of each such side]}``: one entry per side whose ``reasons`` hold any of
        :data:`COVERAGE_REVIEW_REASONS`, the sha256 of its ``accepted`` decision in ``sides_path``, or None when that
        file has no accepted decision with a photograph for the side.
    :raises FileNotFoundError: When either file does not exist.
    """
    accepted = {(decision.row_id, decision.side_key): decision.image_sha256
                for decision in schema.read_jsonl(sides_path, schema.SideDecision)
                if decision.outcome == ACCEPTED and decision.image_sha256}
    wanted: Dict[str, List[Optional[str]]] = defaultdict(list)
    for result in schema.read_jsonl(quality_path, schema.QualityResult):
        if COVERAGE_REVIEW_REASONS.intersection(result.reasons):
            wanted[result.row_id].append(accepted.get((result.row_id, result.side_key)))
    return dict(wanted)


def _coverage_photographs(sides: Sequence[Optional[str]], present: Set[str]) -> Set[str]:
    """The photographs of one candidate that its coverage-review sides ask to be read again.

    :param sides: The accepted photograph of each of its coverage-review sides, None when a side has none
        (:func:`coverage_review_sides`).
    :param present: sha256 of the candidate's readable photographs.
    :return: Each side's accepted photograph; every readable photograph for a side without one, or whose accepted
        photograph is not among the candidate's readable photographs.
    """
    wanted: Set[str] = set()
    for accepted in sides:
        wanted |= {accepted} if accepted in present else present
    return wanted


def select_images(cands: Iterable[Candidate], reads: Iterable[ReadRecord], *, script: str = "all",
                  only_missing_usable: bool = False, reader_id: Optional[str] = None,
                  skip_near_duplicates: Optional[int] = NEAR_DUPLICATE_BITS,
                  counts: Optional[Counter] = None, unplaced: Optional[Set[str]] = None,
                  coverage: Optional[Mapping[str, Sequence[Optional[str]]]] = None) -> List[ImageRef]:
    """The photographs a reader still has to read, in candidate order.

    Images count when their status is ``present`` or ``fetched`` (never ``too_small``, ``listed``, ``missing``, ...)
    and they have a path; an image without a sha256 is hashed here (the returned copy carries it); an image that
    several candidates list is returned once, by its first selected occurrence.

    Without a selection (``only_missing_usable`` false, ``unplaced`` and ``coverage`` None) every such image this reader
    has not read is returned. With any of them, an image is returned when it satisfies at least one (the union): it has
    no usable read, or a candidate in ``unplaced`` lists it, or a coverage-review side of the candidate listing it asks
    for it (:func:`_coverage_photographs`). The last two are decided per listing, so a photograph listed by a placed
    candidate and by an unplaced one is selected through the unplaced one.

    The near-duplicate rule is applied last, one candidate at a time and in candidate order: a photograph is dropped
    when an earlier photograph of the same candidate is within ``skip_near_duplicates`` bits of it and already has a
    usable read from any reader (:func:`is_usable`). The rule only fires once such a read exists, so the first
    photograph of a near-duplicate pair is always read normally, and it only drops what the other rules would have
    queued. A photograph dropped from one candidate stays eligible for any other candidate that lists it.

    :param cands: Candidates with completed ``images`` (the stage 1 file).
    :param reads: Reads of every reader so far.
    :param script: ``all`` | ``hebrew`` | ``arabic``: candidates whose ledger script matches (``mixed`` matches both
        ``hebrew`` and ``arabic``).
    :param only_missing_usable: Select images without a usable read from any reader (:func:`is_usable`): no read at
        all, or only reads with no lines, fewer than 20 script letters, or a finish of looping/blank/error.
    :param reader_id: The reader about to run (:func:`make_reader_id`): its images are skipped once it has a read of
        them. None: no per-reader skip.
    :param skip_near_duplicates: Largest Hamming distance, in bits of the 64-bit hex ``ImageRef.phash``, at which two
        photographs of one candidate are near-duplicates. None or 0 (the default): the rule is off. A photograph
        without a ``phash`` is never dropped by it and never stands in for another.
    :param counts: Counter that receives (added to what it holds) ``skipped_near_duplicate``, the number of distinct
        photographs the near-duplicate rule kept out of the result; with ``only_missing_usable``,
        ``selected_missing_usable``, the returned images without a usable read; with ``unplaced``,
        ``selected_unplaced``, the returned images that a candidate of ``unplaced`` lists; with ``coverage``,
        ``selected_coverage_review``, the returned images a coverage-review side asks for (an image satisfying several
        selections is counted in each). None: nothing is reported.
    :param unplaced: Row ids of candidates with a side not placed (:func:`unplaced_candidates`): select their
        photographs whatever reads they have. None: no such selection.
    :param coverage: ``{row_id: [accepted photograph or None per coverage-review side]}``
        (:func:`coverage_review_sides`): select those photographs whatever reads they have. None: no such selection.
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
    filtered = only_missing_usable or unplaced is not None or coverage is not None
    selected: List[ImageRef] = []
    chosen: Set[str] = set()                # sha256 of the images selected so far
    without_usable: Set[str] = set()        # sha256 of the photographs looked at that have no usable read
    of_unplaced: Set[str] = set()           # sha256 of the photographs that an unplaced candidate lists
    of_coverage: Set[str] = set()           # sha256 of the photographs a coverage-review side asks for
    near_duplicates: Set[str] = set()       # sha256 the near-duplicate rule dropped and no other listing selected
    for cand in cands:
        if not _script_selected(cand.script, script):
            continue
        photos = list(_readable_photographs(cand))
        cand_unplaced = unplaced is not None and cand.row_id in unplaced
        cand_coverage = (_coverage_photographs(coverage[cand.row_id], {sha256 for _, sha256 in photos})
                         if coverage is not None and cand.row_id in coverage else set())
        covered: List[str] = []     # hashes of this candidate's earlier photographs that already have a usable read
        for image, sha256 in photos:
            existing = by_sha.get(sha256, [])
            usable = any(is_usable(read) for read in existing)
            near_duplicate = _is_near_duplicate(image.phash, covered, skip_near_duplicates)
            if usable and image.phash:
                covered.append(image.phash)
            if not usable:
                without_usable.add(sha256)
            if cand_unplaced:
                of_unplaced.add(sha256)
            if sha256 in cand_coverage:
                of_coverage.add(sha256)
            if sha256 in chosen:
                continue
            if reader_id is not None and any(read.reader == reader_id for read in existing):
                continue
            if filtered and not ((only_missing_usable and not usable) or cand_unplaced or sha256 in cand_coverage):
                continue                    # another candidate that lists it may still select it
            if near_duplicate:
                near_duplicates.add(sha256)     # only this candidate has a read twin: another one may still need it
                continue
            near_duplicates.discard(sha256)
            chosen.add(sha256)
            selected.append(image if image.sha256 else dataclasses.replace(image, sha256=sha256))
    if counts is not None:
        counts["skipped_near_duplicate"] += len(near_duplicates)
        if only_missing_usable:
            counts["selected_missing_usable"] += len(chosen & without_usable)
        if unplaced is not None:
            counts["selected_unplaced"] += len(chosen & of_unplaced)
        if coverage is not None:
            counts["selected_coverage_review"] += len(chosen & of_coverage)
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


# --- claims shared by concurrent jobs ----------------------------------------------------------------------------------

def _parse_claim(raw: bytes) -> Optional[Tuple[str, str, str, float]]:
    """One line of a claim log.

    The parse is wrapped in try/except because the log is written by other processes, possibly from other machines
    over a network volume where appends are not guaranteed atomic: a line torn, garbled or edited by hand must not stop
    a reading job, so it is reported as None (the caller counts it) instead of raised.

    :param raw: The line's bytes, without its newline.
    :return: ``(sha256, reader, job, claimed_at as epoch seconds)``, or None for a line that is not a valid claim.
    """
    try:
        record = json.loads(raw)
        stamp = datetime.strptime(record["claimed_at"], CLAIM_STAMP).replace(tzinfo=timezone.utc).timestamp()
        return str(record["sha256"]), str(record["reader"]), str(record["job"]), stamp
    except (ValueError, KeyError, TypeError):       # invalid JSON or UTF-8, a field missing, not an object, bad time
        return None


def default_job_id(work_dir: Path) -> str:
    """A job's id in a claim log when none is given: its work dir as an absolute path.

    Under the per-job snapshot design each job reads into its own work dir, so concurrent jobs get different ids, and a
    job restarted on the same work dir gets its old id back (it holds its earlier claims and can read those images).

    :param work_dir: The job's work dir.
    :return: The resolved path as a string.
    """
    return str(work_dir.resolve())


class ClaimLog:
    """An append-only log through which jobs reading into separate work dirs avoid reading the same photograph twice.

    Each job keeps its own reads file (one snapshot work dir per job); the claim log is the only file they share.
    Before every read attempt a job appends one line ``{"sha256", "reader", "claimed_at", "job"}`` in a single
    ``O_APPEND`` write, then reads back what has been appended since it last looked. A (sha256, reader) pair is *held*
    by the job of its first claim, in file order, that is younger than ``ttl_s`` (24 hours): appends are ordered, so
    two jobs that claim one photograph at the same moment agree on which of them holds it. A job reads a photograph
    only when no other job holds it. Claims are never released: a failed or interrupted attempt keeps its claim, which
    expires after 24 hours; the same job (same id) holds its own claims and may read those photographs again.

    Never edit or truncate the log while jobs run. On a volume shared by two machines appends are not guaranteed atomic:
    a claim can be lost or a line torn (it is skipped and counted in ``malformed``); the worst outcome is one photograph
    read by both jobs, never a wrong read (the reads files are merged on (sha256, reader)).

    :ivar path: The log file (created, with its parent dirs, on the first claim).
    :ivar job: This job's id.
    :ivar ttl_s: Age in seconds after which a claim no longer holds its photograph.
    :ivar malformed: Complete lines read so far that are not valid claims (skipped).
    """

    def __init__(self, path: Path, job: str, ttl_s: float = CLAIM_TTL_S) -> None:
        """Open a claim log (nothing is read until :meth:`refresh`).

        :param path: The log file (it may not exist yet).
        :param job: This job's id (see :func:`default_job_id`).
        :param ttl_s: Age in seconds after which a claim expires.
        """
        self.path, self.job, self.ttl_s = path, job, ttl_s
        self.malformed = 0
        self._offset = 0                    # bytes of the log read so far (always the end of a complete line)
        self._claims: Dict[Tuple[str, str], List[Tuple[str, float]]] = defaultdict(list)

    def refresh(self) -> int:
        """Read the lines appended since the last call.

        A last line without its newline is still being written by another job: it is left for the next call.

        :return: Number of valid claims read.
        """
        if not self.path.exists():
            return 0
        with self.path.open("rb") as fh:
            fh.seek(self._offset)
            chunk = fh.read()
        end = chunk.rfind(b"\n") + 1
        added = 0
        for raw in chunk[:end].split(b"\n"):
            if not raw.strip():
                continue
            parsed = _parse_claim(raw)
            if parsed is None:
                self.malformed += 1
                continue
            sha256, reader, job, stamp = parsed
            self._claims[(sha256, reader)].append((job, stamp))
            added += 1
        self._offset += end
        return added

    def holder(self, sha256: str, reader: str, now: Optional[float] = None) -> Optional[str]:
        """The job holding a photograph for a reader, from the claims read so far.

        :param sha256: The photograph's sha256.
        :param reader: Reader id.
        :param now: Epoch seconds to judge the claims' age at; None: the current time.
        :return: The job of the first claim of the pair (file order) younger than ``ttl_s``; None when there is none.
        """
        now = time.time() if now is None else now
        for job, stamp in self._claims.get((sha256, reader), ()):
            if now - stamp < self.ttl_s:
                return job
        return None

    def held_by_other(self, sha256: str, reader: str) -> bool:
        """Whether another job holds a photograph for a reader (from the claims read so far; see :meth:`refresh`).

        :param sha256: The photograph's sha256.
        :param reader: Reader id.
        :return: True when :meth:`holder` is a job other than this one.
        """
        return self.holder(sha256, reader) not in (None, self.job)

    def claim(self, sha256: str, reader: str) -> bool:
        """Append this job's claim of a photograph, then read the log back and say whether this job holds it.

        :param sha256: The photograph's sha256.
        :param reader: Reader id.
        :return: False when another job's live claim of the pair comes first in the log (it holds the photograph).
        """
        line = json.dumps({"sha256": sha256, "reader": reader, "claimed_at": now_utc(), "job": self.job},
                          ensure_ascii=False) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))      # one write: on a local volume concurrent appends do not interleave
        finally:
            os.close(fd)
        self.refresh()
        return not self.held_by_other(sha256, reader)


# --- the run -----------------------------------------------------------------------------------------------------------

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


def _selection_note(counts: Mapping[str, int], only_missing_usable: bool, n_unplaced: Optional[int],
                    n_coverage_sides: Optional[int] = None) -> str:
    """The selection part of the launch line of :func:`run_reads`.

    :param counts: The counter :func:`select_images` filled.
    :param only_missing_usable: Whether that selection is on.
    :param n_unplaced: Number of candidates with an unplaced side; None when that selection is off.
    :param n_coverage_sides: Number of sides asking for a coverage review; None when that selection is off.
    :return: E.g. ``"selected: 12 without a usable read, 30 of 9 candidates with an unplaced side, 4 for coverage
        review of 3 sides"`` or ``"selected: every image not read by this reader"``.
    """
    parts = []
    if only_missing_usable:
        parts.append(f"{counts['selected_missing_usable']} without a usable read")
    if n_unplaced is not None:
        parts.append(f"{counts['selected_unplaced']} of {n_unplaced} candidates with an unplaced side")
    if n_coverage_sides is not None:
        parts.append(f"{counts['selected_coverage_review']} for coverage review of {n_coverage_sides} sides")
    return "selected: " + (", ".join(parts) if parts else "every image not read by this reader")


def run_reads(work_dir: Path, *, reader: str, model: str, read_fn: ReadFn, script: str = "all",
              limit: Optional[int] = None, stop_file: Optional[Path] = None, only_missing_usable: bool = False,
              skip_near_duplicates: Optional[int] = NEAR_DUPLICATE_BITS, only_unplaced: Optional[Path] = None,
              claim_file: Optional[Path] = None, ignore_claims: bool = False, job_id: Optional[str] = None,
              only_coverage_review: Optional[Path] = None, sides: Optional[Path] = None) -> Dict[str, int]:
    """Read the photographs of the stage 1 file that this reader has not read yet.

    Records are appended to ``reads/reads.jsonl`` one at a time (flushed), so an interrupted run loses nothing. A
    failure (an ``error`` record, a missing file, a sha256 that disagrees with the stage 1 file) goes to
    ``reads/reads_failures.jsonl`` and leaves the image unread. The run ends after five failures in a row, when
    ``stop_file`` exists (checked before each image), or when the queue is empty. Progress is printed every 25 reads.

    The near-duplicate rule (off unless ``skip_near_duplicates`` is set) is applied twice. :func:`select_images`
    applies it when the queue is made, which removes the photographs whose twin already has a usable read. Most twins
    are still unread then, so it is applied again before each read: the photograph is skipped (not read, not a failure,
    counted) when, in every candidate that lists it, an earlier near-duplicate has a usable read by now, the reads of
    this run included. The queue keeps candidate order, so of two twins unread at launch the first is read and the
    second skipped, unless that first read is unusable or fails.

    With a ``claim_file`` (:class:`ClaimLog`) the photographs another job holds for this reader are dropped when the
    queue is made (before ``limit`` cuts it), and the log is read again before each read, so jobs running at the same
    time share progress: a photograph another job has claimed by then is skipped; otherwise this job claims it (one
    line per attempt, written before the request) and reads it unless another job's claim landed first. With
    ``ignore_claims`` nothing is skipped for a claim, but every attempt is still claimed.

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
    :param only_unplaced: A stage 2 sides file: also select the photographs of the candidates it leaves with a side
        not accepted (:func:`unplaced_candidates`; the union with ``only_missing_usable`` when both are given). None:
        no such selection.
    :param claim_file: Claim log shared with other jobs; None: no claims.
    :param ignore_claims: Read photographs other jobs hold (claims are still written).
    :param job_id: This job's id in the claim log; None: :func:`default_job_id` of ``work_dir``.
    :param only_coverage_review: A stage 3 quality file: also select the photographs its coverage-review sides ask for
        (:func:`coverage_review_sides`; the union with the other selections). None: no such selection.
    :param sides: The stage 2 sides file that names each side's accepted photograph for ``only_coverage_review``, used
        when ``only_unplaced`` (which names it otherwise) is None.
    :return: Counts: ``queued``, ``skipped_near_duplicate`` (photographs the near-duplicate rule kept out of the queue
        plus those it skipped at read time), ``skipped_claimed`` (photographs another job held, at launch or at read
        time), ``claims_written``, ``read``, ``failed``, ``finish_<finish>`` per finish seen, ``stopped_by_stop_file``
        and ``stopped_by_failures`` (1 when that ended the run, else 0); with ``only_missing_usable``,
        ``selected_missing_usable``; with ``only_unplaced``, ``candidates_unplaced`` and ``selected_unplaced``; with
        ``only_coverage_review``, ``coverage_review_sides`` and ``selected_coverage_review`` (see
        :func:`select_images`; counted before claims and ``limit``).
    :raises FileNotFoundError: When the stage 1 file or a given sides or quality file does not exist.
    :raises ValueError: When ``only_coverage_review`` is given without a sides file (``sides`` or ``only_unplaced``).
    """
    sides_file = only_unplaced if only_unplaced is not None else sides
    if only_coverage_review is not None and sides_file is None:
        raise ValueError("only_coverage_review needs the stage 2 sides file (sides or only_unplaced)")
    stage1 = paths.stage_path(work_dir, "images")
    if not stage1.exists():
        raise FileNotFoundError(f"stage 1 file not found: {stage1} (run the images stage first)")
    reader_id = make_reader_id(reader, model)
    reads_file, failures_file = reads_path(work_dir), failures_path(work_dir)
    repair_tail(reads_file)
    repair_tail(failures_file)
    existing = load_reads(work_dir)
    unplaced = unplaced_candidates(only_unplaced) if only_unplaced is not None else None
    coverage = (coverage_review_sides(only_coverage_review, sides_file)
                if only_coverage_review is not None and sides_file is not None else None)
    selection: Counter = Counter()
    queue = select_images(schema.read_jsonl(stage1, Candidate), existing, script=script,
                          only_missing_usable=only_missing_usable, reader_id=reader_id,
                          skip_near_duplicates=skip_near_duplicates, counts=selection, unplaced=unplaced,
                          coverage=coverage)
    counts: Counter = Counter({"queued": 0, "skipped_near_duplicate": selection["skipped_near_duplicate"],
                               "skipped_claimed": 0, "claims_written": 0, "read": 0, "failed": 0,
                               "stopped_by_stop_file": 0, "stopped_by_failures": 0})
    if only_missing_usable:
        counts["selected_missing_usable"] = selection["selected_missing_usable"]
    if unplaced is not None:
        counts["candidates_unplaced"] = len(unplaced)
        counts["selected_unplaced"] = selection["selected_unplaced"]
    n_coverage_sides = None if coverage is None else sum(len(entries) for entries in coverage.values())
    if coverage is not None:
        counts["coverage_review_sides"] = n_coverage_sides or 0
        counts["selected_coverage_review"] = selection["selected_coverage_review"]
    claims = ClaimLog(claim_file, job_id or default_job_id(work_dir)) if claim_file is not None else None
    if claims is not None:
        claims.refresh()
        if not ignore_claims:
            free = [image for image in queue if not claims.held_by_other(image.sha256 or "", reader_id)]
            counts["skipped_claimed"] = len(queue) - len(free)
            queue = free
    if limit is not None:
        queue = queue[: max(0, limit)]
    counts["queued"] = len(queue)
    peers = _near_duplicate_peers(schema.read_jsonl(stage1, Candidate), script, skip_near_duplicates)
    usable = {read.sha256 for read in existing if is_usable(read)}      # grows with every usable read of this run
    claim_note = "no claim file" if claims is None else (
        f"claim file {claims.path} as job {claims.job}: "
        + ("claims ignored" if ignore_claims else f"{counts['skipped_claimed']} held by other jobs"))
    note = _selection_note(selection, only_missing_usable, None if unplaced is None else len(unplaced), n_coverage_sides)
    print(f"{now_utc()} {reader_id}: {len(queue)} images queued ({script}; {note}; "
          f"skip_near_duplicates={skip_near_duplicates}: {counts['skipped_near_duplicate']} skipped as "
          f"near-duplicates at launch; {claim_note})", flush=True)
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
        if claims is not None:
            sha256 = image.sha256 or ""             # always set on a selected image
            claims.refresh()
            if not ignore_claims and claims.held_by_other(sha256, reader_id):
                counts["skipped_claimed"] += 1
                print(f"{now_utc()} skipped {image.path}: claimed by job {claims.holder(sha256, reader_id)}", flush=True)
                continue
            held = claims.claim(sha256, reader_id)
            counts["claims_written"] += 1
            if not held and not ignore_claims:
                counts["skipped_claimed"] += 1
                print(f"{now_utc()} skipped {image.path}: job {claims.holder(sha256, reader_id)} claimed it first",
                      flush=True)
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
        --only-unplaced --only-coverage-review --sides --skip-near-duplicates --claim-file --job-id --ignore-claims
        --import-legacy --gemini-index``.
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
                        help="read only images without a usable read from any reader (with --only-unplaced: the union)")
    parser.add_argument("--only-unplaced", type=Path, default=None, metavar="SIDES_FILE",
                        help="read the photographs of candidates that this stage 2 sides file leaves with a side not "
                             "accepted, whatever reads they have (with the other selections: the union)")
    parser.add_argument("--only-coverage-review", type=Path, default=None, metavar="QUALITY_FILE",
                        help="read the accepted photograph of each side whose stage 3 quality reasons ask for a second "
                             "read (every present photograph of its candidate when it has none); needs the sides file "
                             "from --only-unplaced or --sides (with the other selections: the union)")
    parser.add_argument("--sides", type=Path, default=None, metavar="SIDES_FILE",
                        help="with --only-coverage-review and without --only-unplaced: the stage 2 sides file")
    parser.add_argument("--skip-near-duplicates", type=int, default=NEAR_DUPLICATE_BITS, metavar="N",
                        help="opt-in: do not read a photograph whose 64-bit phash is within N bits of an earlier "
                             "photograph of the same candidate that already has a usable read (default: %(default)s, "
                             "off; two faces of one leaf can be phash twins)")
    parser.add_argument("--claim-file", type=Path, default=None,
                        help="append-only claim log shared with jobs reading into other work dirs: skip photographs "
                             "another job claimed for this reader in the last 24 h, claim each one before reading it")
    parser.add_argument("--job-id", default=None,
                        help="with --claim-file: this job's id in the log (default: the work dir's absolute path)")
    parser.add_argument("--ignore-claims", action="store_true",
                        help="with --claim-file: read photographs other jobs claimed (this job still writes its claims)")
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
    if args.claim_file is None and (args.ignore_claims or args.job_id is not None):
        parser.error("--ignore-claims and --job-id need --claim-file")
    if args.sides is not None and args.only_coverage_review is None:
        parser.error("--sides is read only by --only-coverage-review")
    if args.only_coverage_review is not None and args.only_unplaced is None and args.sides is None:
        parser.error("--only-coverage-review needs the stage 2 sides file: give --only-unplaced or --sides")
    if (args.sides is not None and args.only_unplaced is not None
            and args.sides.resolve() != args.only_unplaced.resolve()):
        parser.error("--sides and --only-unplaced name different sides files; give the sides file once")
    for flag, given in (("--only-unplaced", args.only_unplaced), ("--only-coverage-review", args.only_coverage_review),
                        ("--sides", args.sides)):
        if args.reader is not None and given is not None and not given.is_file():
            parser.error(f"{flag}: no file at {given}")
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
                       only_missing_usable=args.only_missing_usable, skip_near_duplicates=args.skip_near_duplicates,
                       only_unplaced=args.only_unplaced, claim_file=args.claim_file, ignore_claims=args.ignore_claims,
                       job_id=args.job_id, only_coverage_review=args.only_coverage_review, sides=args.sides)
    print(json.dumps(counts, indent=1), flush=True)


if __name__ == "__main__":
    main()
