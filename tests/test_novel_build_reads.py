"""Tests of stage 2 (reads) of the novel-rows build: legacy import, the two readers, selection and the resumable run.

Every I/O boundary is stubbed: ``post`` for the Kraken service, ``read_page`` for the page reader, ``read_fn`` for the
run. An autouse fixture makes any accidental real request fail loudly (it raises a ``BaseException``, which the
readers' error handling does not swallow). Images are tiny JPEGs made with PIL; texts are synthetic alphabets.
"""
import functools
import hashlib
import importlib
import io
import json
import random
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import requests
from PIL import Image

from src.finetuning.qwen_hebrew.novel_build import paths, reads, schema
from src.finetuning.qwen_hebrew.novel_build.schema import Candidate, ImageRef, ReadRecord
from src.models.ocr import page_reader

HEBREW = "אבגדהוזחטיכלמנסעפצקרשת"                  # 22 Hebrew letters (the alphabet; synthetic, not a text)
ARABIC = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"              # 28 Arabic letters (the alphabet; synthetic, not a text)
KRAKEN_MODEL_PATH = "/models/test_model.mlmodel"
KRAKEN_READER = "kraken:test_model"
TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
TWIN_BITS = 6                       # the near-duplicate distance the tests opt in to (the module's default is 0: off)
REAL_READ_PAGE = page_reader.read_page      # taken before the autouse guard replaces it; only run with a stub transcribe


class ForbiddenIO(BaseException):
    """Raised by the autouse guard when a test reaches the real service or model."""


@pytest.fixture(autouse=True)
def no_real_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``requests.post`` and ``page_reader.read_page`` fail loudly when a test forgets to stub them.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise ForbiddenIO("a test reached the real service or model")

    monkeypatch.setattr(requests, "post", refuse)
    monkeypatch.setattr(page_reader, "read_page", refuse)


# --- fixtures and builders ----------------------------------------------------------------------------------------------

def make_jpeg(path: Path, size: Sequence[int] = (120, 80), color: Sequence[int] = (200, 30, 30)) -> Path:
    """Write a solid-colour JPEG (different colours give different sha256 values).

    :param path: Destination (parent dirs are created).
    :param size: ``(width, height)``.
    :param color: RGB.
    :return: ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", tuple(size), tuple(color)).save(path, format="JPEG")
    return path


def sha_of(path: Path) -> str:
    """sha256 of a file's bytes.

    :param path: File.
    :return: Hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_ref(image_id: str, path: Optional[Path], status: str = "present", sha256: Optional[str] = "auto",
              origin: str = "iiif_pgp", phash: Optional[str] = None) -> ImageRef:
    """An :class:`ImageRef` for a test file.

    :param image_id: Image id.
    :param path: Local file (None: no path).
    :param status: Image status.
    :param sha256: ``"auto"``: the file's sha256; None: unknown; else the given digest.
    :param origin: Image origin.
    :param phash: 64-bit perceptual hash as hex (None: the image has none).
    :return: The reference.
    """
    digest = sha_of(path) if (sha256 == "auto" and path is not None) else (None if sha256 == "auto" else sha256)
    return ImageRef(image_id=image_id, origin=origin, path=str(path) if path else None, sha256=digest, status=status,
                    phash=phash)


def candidate(row_id: str, script: str, images: List[ImageRef]) -> Candidate:
    """A synthetic candidate.

    :param row_id: Row id.
    :param script: Ledger script.
    :param images: Its images.
    :return: The candidate.
    """
    return Candidate(row_id=row_id, source="pgp", document_group=f"group-{row_id}", canonical_ids=[f"Testlib_{row_id}"],
                     pgpids=[], ktiv_id=None, shelfmark=f"Test MS {row_id}", script=script, languages_stated="synthetic",
                     verdict="novel", witnesses=[], unit="document", n_letters=100, images=images)


def write_stage1(work: Path, cands: List[Candidate]) -> Path:
    """Write the stage 1 file of a work dir.

    :param work: Work dir.
    :param cands: Candidates.
    :return: The stage 1 path.
    """
    path = paths.stage_path(work, "images")
    schema.write_jsonl(path, cands)
    return path


@pytest.fixture
def stage1(tmp_path: Path) -> SimpleNamespace:
    """A stage 1 file with 3 candidates and 4 images: one ``too_small``, one sha256 shared by two candidates.

    :param tmp_path: Pytest temp dir.
    :return: Namespace with ``work``, ``cands``, the image files (``he``, ``small``, ``ar``) and their ``sha`` values.
    """
    work = tmp_path / "work"
    he = make_jpeg(work / "images" / "he.jpg", color=(200, 10, 10))
    small = make_jpeg(work / "images" / "small.jpg", size=(20, 10), color=(10, 200, 10))
    ar = make_jpeg(work / "images" / "ar.jpg", color=(10, 10, 200))
    cands = [
        candidate("r_he", "hebrew", [image_ref("iiif_pgp:0", he), image_ref("iiif_pgp:1", small, status="too_small")]),
        candidate("r_ar", "arabic", [image_ref("iiif_pgp:0", ar, status="fetched")]),
        candidate("r_mixed", "mixed", [image_ref("cudl_local:0", ar, origin="cudl_local")]),
    ]
    write_stage1(work, cands)
    return SimpleNamespace(work=work, cands=cands, he=he, small=small, ar=ar,
                           sha_he=sha_of(he), sha_small=sha_of(small), sha_ar=sha_of(ar))


def read_record(sha256: str, reader: str = KRAKEN_READER, text: str = HEBREW, finish: str = "ok") -> ReadRecord:
    """A synthetic read.

    :param sha256: Photograph digest.
    :param reader: Reader id.
    :param text: Read text.
    :param finish: Finish.
    :return: The record.
    """
    return reads.make_record(sha256, reader, text, finish=finish, seconds=1.0)


def jsonl_dicts(path: Path) -> List[Dict[str, Any]]:
    """Plain dicts of a JSON-lines file.

    :param path: File.
    :return: Records.
    """
    return list(schema.read_jsonl(path))


def kraken_body(lines: Sequence[str] = (HEBREW, HEBREW[:10]), **extra: Any) -> Dict[str, Any]:
    """A ``transcribe_lines`` response body.

    :param lines: Line texts.
    :param extra: Further body fields.
    :return: The body.
    """
    body: Dict[str, Any] = {
        "lines": [{"text": text, "bbox": [10 * i, 20 * i, 10 * i + 90, 20 * i + 15], "confidence": round(0.9 - 0.1 * i, 2)}
                  for i, text in enumerate(lines)],
        "polygon_failures": 2, "used_binarization": True, "model_path": KRAKEN_MODEL_PATH, "image_size": [100, 50],
        "view": {"rotation": 0, "scale": 1.0, "scores": {"0/1.0": 700.0}}}
    body.update(extra)
    return body


class FakeResponse:
    """Minimal ``requests`` response."""

    def __init__(self, body: Any, status: int = 200) -> None:
        """Keep the body and status.

        :param body: What ``json()`` returns (an exception instance is raised instead).
        :param status: HTTP status.
        """
        self.body, self.status = body, status

    def raise_for_status(self) -> None:
        """Raise ``requests.HTTPError`` for an error status."""
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status} error")

    def json(self) -> Any:
        """Return the body (or raise it when it is an exception).

        :return: The decoded body.
        """
        if isinstance(self.body, BaseException):
            raise self.body
        return self.body


class FakePost:
    """Stub for ``requests.post`` that records the call and reads the uploaded file while it is open."""

    def __init__(self, body: Any = None, status: int = 200, error: Optional[BaseException] = None) -> None:
        """Configure the answer.

        :param body: Response body (default: :func:`kraken_body`).
        :param status: HTTP status.
        :param error: Exception to raise instead of answering.
        """
        self.body = kraken_body() if body is None else body
        self.status, self.error, self.calls = status, error, []

    def __call__(self, url: str, data: Optional[Dict[str, str]] = None, files: Optional[Dict[str, Any]] = None,
                 timeout: Optional[float] = None) -> FakeResponse:
        """Record the request.

        :param url: Endpoint.
        :param data: Form fields.
        :param files: Multipart files (the ``image`` handle is read here).
        :param timeout: Timeout.
        :return: The configured response.
        """
        handle = files["image"]
        self.calls.append({"url": url, "data": data, "timeout": timeout, "filename": Path(handle.name).name,
                           "payload": handle.read()})
        if self.error is not None:
            raise self.error
        return FakeResponse(self.body, self.status)


def vlm_result(**overrides: Any) -> Dict[str, Any]:
    """A ``page_reader.read_page`` result dict (the :data:`page_reader.READ_FIELDS` that matter).

    :param overrides: Fields to change.
    :return: The result.
    """
    result: Dict[str, Any] = {
        "text": f"{ARABIC}\n\n{ARABIC[:12]}", "raw_text": None, "reason": None, "blank": False, "looped": False,
        "cut_at": None, "letters_removed": 0, "repeat_share": 0.0, "finish_reason": "stop", "prompt_tokens": 6500,
        "completion_tokens": 40, "total_tokens": 6540, "attempts": 1, "retried": False, "prompt": "script_neutral",
        "max_tokens": 1200, "temperature": 0.1, "model": "model-x", "ink": {"class": "text", "ink_permille": 12.5},
        "secs": 3.2, "first_finish_reason": "stop", "rotation": 0, "options": ""}
    result.update(overrides)
    return result


def fake_read_page(result: Optional[Dict[str, Any]] = None, error: Optional[BaseException] = None) -> Any:
    """An async stand-in for ``page_reader.read_page`` that records its calls.

    :param result: The result dict to return (default :func:`vlm_result`).
    :param error: Exception to raise instead.
    :return: The coroutine function; its ``calls`` attribute lists ``(model_key, image_path)``.
    """
    calls: List[Any] = []

    async def read_page(model_id: str, image_path: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Stub reader.

        :param model_id: Model key.
        :param image_path: Image file.
        :return: The configured result.
        """
        calls.append((model_id, image_path))
        if error is not None:
            raise error
        return dict(result if result is not None else vlm_result())

    read_page.calls = calls  # type: ignore[attr-defined]
    return read_page


def write_legacy_caches(root: Path) -> SimpleNamespace:
    """Fake legacy caches in both shapes: a Kraken cache and a Gemini index with per-key files.

    :param root: Directory to write into.
    :return: Namespace with ``kraken``, ``gemini_index`` and the sha256 values used.
    """
    sha = {name: hashlib.sha256(name.encode()).hexdigest() for name in ("k1", "k2", "g_text", "g_none", "g_max", "g_safety")}
    row = {"sha256": sha["k1"], "canonical_id": "Testlib_A", "file_name": "a_1r.jpg", "side_token": "r",
           "main_language": "hebrew", "prepared": {"path": "x_2000.jpg", "width": 100, "height": 150, "scale": 0.5},
           "model_path": "/app/models/MiDRASH_Gen_01.mlmodel", "n_lines": 2,
           "lines": [{"text": HEBREW, "bbox": [1, 2, 3, 4], "confidence": 0.95},
                     {"text": HEBREW[:10], "bbox": [5, 6, 7, 8], "confidence": 0.5}],
           "view": {"rotation": 0, "scale": 1.0, "scores": {}}, "image_size": [100, 150], "polygon_failures": 1,
           "used_binarization": True, "seconds": 15.7, "read_at": "2026-10-30T18:10:06Z"}
    zero = dict(row, sha256=sha["k2"], n_lines=0, lines=[], seconds=3.0, read_at="2026-10-30T18:11:00Z")
    kraken = root / "kraken_reads.jsonl"
    schema.write_jsonl(kraken, [row, zero])

    gemini = root / "gemini_reads_arabic"
    gemini.mkdir(parents=True)
    bodies = {
        "key_text": ("g_text", ARABIC, "FinishReason.STOP"),
        "key_none": ("g_none", "NO TEXT", "FinishReason.STOP"),
        "key_max": ("g_max", ARABIC[:15], "FinishReason.MAX_TOKENS"),
        "key_safety": ("g_safety", "", "FinishReason.SAFETY"),
    }
    index = []
    for key, (name, text, finish) in bodies.items():
        (gemini / f"{key}.json").write_text(json.dumps({
            "image": f"{key}.jpg", "model": "gemini-test", "prompt_sha1": "abc123", "settings": {"thinking_level": "low"},
            "text": text, "usage": {"input_tokens": 10, "output_tokens": 5}, "cost_usd": 0.01, "finish": finish,
            "secs": 9.9, "read_at": "2026-10-30T18:06:36Z"}, ensure_ascii=False), encoding="utf-8")
        index.append({"key": key, "canonical_id": "Testlib_B", "sha256": sha[name], "file_name": f"{key}.jpg"})
    index.append({"key": "key_absent", "canonical_id": "Testlib_B", "sha256": "f" * 64, "file_name": "absent.jpg"})
    schema.write_jsonl(gemini / "index.jsonl", index)
    return SimpleNamespace(kraken=kraken, gemini_index=gemini / "index.jsonl", sha=sha)


# --- small helpers -----------------------------------------------------------------------------------------------------

def test_make_record_counts_letters_and_lines() -> None:
    """Letters per script and the non-empty lines are derived from the text."""
    record = reads.make_record("a" * 64, "vlm:m", f"{HEBREW}\n\n{ARABIC}\n", finish="ok", seconds=1.5)
    assert (record.letters_hebrew, record.letters_arabic) == (22, 28)
    assert record.lines == [HEBREW, ARABIC] and record.n_lines == 2
    assert TIMESTAMP.match(record.read_at)


def test_letter_ranges_skip_marks_and_punctuation() -> None:
    """Harakat, the Arabic full stop and Latin letters are not letters; alef wasla (U+0671) and U+06D3 are."""
    assert reads._count_letters("\u0628\u064e") == (0, 1)                      # beh + fatha
    assert reads._count_letters("\u0671\u06d3\u06d4abc.") == (0, 2)
    assert reads._count_letters("\u05d0\u05b0\u05d1\u05be") == (2, 0)           # Hebrew points and maqaf are not letters
    assert reads._count_letters("") == (0, 0)


@pytest.mark.parametrize("kwargs, usable", [
    ({}, True),
    ({"n_lines": 0}, False),
    ({"letters_hebrew": 19, "letters_arabic": 0}, False),
    ({"letters_hebrew": 10, "letters_arabic": 10}, True),
    ({"finish": "looping"}, False),
    ({"finish": "length"}, True),                       # a truncated read is still placement evidence
    ({"finish": "error"}, False),
    ({"finish": "blank"}, False),
    ({"finish": "length", "letters_hebrew": 19}, False),
])
def test_is_usable(kwargs: Dict[str, Any], usable: bool) -> None:
    """A read needs lines, 20 script letters and a finish of ok or length (looping, blank and error are unusable)."""
    base = dict(sha256="a" * 64, reader="kraken:m", text="t", lines=["t"], n_lines=1, letters_hebrew=30, letters_arabic=0,
                finish="ok", seconds=1.0, read_at="2026-11-03T00:00:00Z")
    base.update(kwargs)
    assert reads.is_usable(ReadRecord(**base)) is usable


@pytest.mark.parametrize("meta, finish, usable", [
    ({"finish_reason": "length", "looped": False, "retried": True}, "length", True),
    ({"finish_reason": "length", "looped": True, "cut_at": None, "repeat_share": 0.03, "retried": True}, "length", True),
    ({"finish_reason": "length", "looped": True, "cut_at": 120, "repeat_share": 0.7, "retried": True}, "looping", False),
    ({"finish_reason": "length", "looped": True, "cut_at": None, "repeat_share": 0.6}, "looping", False),
    ({"finish_reason": "length", "looped": True}, "looping", False),
    ({"finish_reason": "stop", "looped": True, "cut_at": 80, "repeat_share": 0.8}, "looping", False),
    ({}, "looping", False),
    ({"source": "legacy_import"}, "looping", False),
], ids=["length_not_looped", "length_flagged_by_the_cut_only", "length_loop_cut", "length_repeat_over_gate",
        "length_without_evidence", "stop_loop", "no_meta", "not_a_page_reader_read"])
def test_a_looping_record_on_disk_is_rejudged_from_its_meta(meta: Dict[str, Any], finish: str, usable: bool) -> None:
    """Records stored as looping by the old mapping count as length when the reader's fields show a cut, no loop."""
    record = reads.make_record("a" * 64, "vlm:model-x", f"{ARABIC}\n{ARABIC}", finish="looping", seconds=1.0, meta=meta)
    assert record.finish == "looping"                       # the stored record itself is never rewritten
    assert reads.effective_finish(record) == finish and reads.is_usable(record) is usable


def test_effective_finish_leaves_other_finishes_alone() -> None:
    """Only a looping record is re-judged: an ok read with looped meta stays ok, a blank one stays blank."""
    meta = {"finish_reason": "length", "looped": True, "cut_at": 30, "repeat_share": 0.9}
    for finish in ("ok", "length", "blank", "error"):
        assert reads.effective_finish(reads.make_record("a" * 64, "vlm:m", HEBREW, finish=finish, seconds=1.0,
                                                        meta=meta)) == finish


def test_reader_ids() -> None:
    """Kraken ids use the model file stem (default model when empty), vlm ids the model key."""
    assert reads.make_reader_id("kraken", "/app/models/MiDRASH_Gen_01.mlmodel") == "kraken:MiDRASH_Gen_01"
    assert reads.make_reader_id("kraken", "") == f"kraken:{Path(reads.KRAKEN_MODEL).stem}"
    assert reads.make_reader_id("vlm", "org/model-8b") == "vlm:org/model-8b"
    with pytest.raises(ValueError):
        reads.make_reader_id("claude", "x")


def test_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    """The streamed digest equals the digest of the bytes."""
    path = make_jpeg(tmp_path / "a.jpg")
    assert reads.file_sha256(path) == sha_of(path)


def test_append_jsonl_writes_one_utf8_line_per_record(tmp_path: Path) -> None:
    """Dataclasses and dicts both append as one line; Hebrew stays unescaped; the parent dir is created."""
    path = tmp_path / "deep" / "dir" / "reads.jsonl"
    reads.append_jsonl(path, read_record("a" * 64))
    reads.append_jsonl(path, {"note": HEBREW[:3]})
    raw = path.read_text(encoding="utf-8")
    assert raw.count("\n") == 2 and HEBREW[:3] in raw
    first, second = jsonl_dicts(path)
    assert first["sha256"] == "a" * 64 and first["reader"] == KRAKEN_READER and second == {"note": HEBREW[:3]}


@pytest.mark.parametrize("content, kept", [
    ('{"a": 1}\n{"b": 2}\n{"c"', '{"a": 1}\n{"b": 2}\n'),
    ('{"c"', ""),
    ('{"a": 1}\n', '{"a": 1}\n'),
    ("", ""),
])
def test_repair_tail_drops_only_an_unterminated_line(tmp_path: Path, content: str, kept: str) -> None:
    """A cut-off last line is removed; an intact or empty file is left alone."""
    path = tmp_path / "f.jsonl"
    path.write_text(content, encoding="utf-8")
    removed = reads.repair_tail(path)
    assert path.read_text(encoding="utf-8") == kept
    assert removed == len(content.encode()) - len(kept.encode())


def test_repair_tail_handles_a_tail_longer_than_one_block(tmp_path: Path) -> None:
    """The backwards scan crosses block boundaries; a missing file is a no-op."""
    path = tmp_path / "f.jsonl"
    path.write_bytes(b'{"a": 1}\n' + b"x" * 200_000)
    assert reads.repair_tail(path) == 200_000
    assert path.read_bytes() == b'{"a": 1}\n'
    assert reads.repair_tail(tmp_path / "missing.jsonl") == 0


def test_prepare_copy_downscales_without_enlarging(tmp_path: Path) -> None:
    """The long side is capped, the aspect ratio kept, smaller images stay, any mode becomes an RGB JPEG."""
    big = make_jpeg(tmp_path / "big.jpg", size=(300, 150))
    assert reads.prepare_copy(big, tmp_path / "out" / "big_100.jpg", 100) == ((300, 150), (100, 50))
    with Image.open(tmp_path / "out" / "big_100.jpg") as copy:
        assert copy.size == (100, 50) and copy.format == "JPEG" and copy.mode == "RGB"
    small = make_jpeg(tmp_path / "small.jpg", size=(60, 40))
    assert reads.prepare_copy(small, tmp_path / "out" / "small.jpg", 100) == ((60, 40), (60, 40))
    grey = tmp_path / "grey.png"
    Image.new("L", (40, 30), 128).save(grey)
    reads.prepare_copy(grey, tmp_path / "out" / "grey.jpg", 100)
    with Image.open(tmp_path / "out" / "grey.jpg") as copy:
        assert copy.mode == "RGB"


def test_error_record_truncates_the_message_and_rounds_the_seconds() -> None:
    """The stored error is the exception type plus at most 300 characters of its message."""
    record = reads.error_record("a" * 64, "vlm:m", RuntimeError("x" * 1000), 1.234)
    assert record.finish == "error" and record.seconds == 1.2 and record.text == "" and record.n_lines == 0
    assert record.meta["error"] == "RuntimeError: " + "x" * 300


def test_load_reads_of_a_fresh_work_dir_is_empty(tmp_path: Path) -> None:
    """No reads file yet means no reads, not an error."""
    assert reads.load_reads(tmp_path / "nowhere") == []


# --- legacy import -----------------------------------------------------------------------------------------------------

def test_import_legacy_reads_maps_both_cache_shapes(tmp_path: Path) -> None:
    """Kraken and Gemini cache rows become records with the right reader ids, finishes, letters and meta."""
    legacy = write_legacy_caches(tmp_path / "legacy")
    work = tmp_path / "work"
    assert reads.import_legacy_reads(work, legacy.kraken, legacy.gemini_index) == 6      # the absent key file is skipped
    records = {(r.sha256): r for r in reads.load_reads(work)}
    assert len(records) == 6

    kraken = records[legacy.sha["k1"]]
    assert kraken.reader == "kraken:MiDRASH_Gen_01" and kraken.finish == "ok"
    assert kraken.lines == [HEBREW, HEBREW[:10]] and kraken.text == f"{HEBREW}\n{HEBREW[:10]}" and kraken.n_lines == 2
    assert (kraken.letters_hebrew, kraken.letters_arabic) == (32, 0)
    assert (kraken.seconds, kraken.read_at) == (15.7, "2026-10-30T18:10:06Z")
    assert kraken.meta["confidences"] == [0.95, 0.5] and kraken.meta["bboxes"] == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert kraken.meta["prepared_size"] == [100, 150] and kraken.meta["polygon_failures"] == 1
    assert kraken.meta["source"] == "legacy_import" and kraken.meta["model_path"].endswith("MiDRASH_Gen_01.mlmodel")

    zero = records[legacy.sha["k2"]]
    assert zero.finish == "blank" and zero.n_lines == 0 and zero.text == "" and not reads.is_usable(zero)

    gemini = records[legacy.sha["g_text"]]
    assert gemini.reader == "gemini:gemini-test" and gemini.finish == "ok" and gemini.text == ARABIC
    assert gemini.letters_arabic == 28 and gemini.n_lines == 1 and gemini.seconds == 9.9
    assert gemini.read_at == "2026-10-30T18:06:36Z" and gemini.meta["cost_usd"] == 0.01
    assert reads.is_usable(gemini)

    none = records[legacy.sha["g_none"]]
    assert none.finish == "blank" and none.text == "" and none.lines == [] and none.meta["no_text_marker"] is True
    g_max = records[legacy.sha["g_max"]]
    assert g_max.finish == "length" and g_max.letters_arabic == 15
    assert not reads.is_usable(g_max)                       # a length finish is usable; 15 letters are not
    assert records[legacy.sha["g_safety"]].finish == "error"


def test_import_legacy_reads_without_a_gemini_index(tmp_path: Path) -> None:
    """Only the Kraken cache is read when no Gemini index is given."""
    legacy = write_legacy_caches(tmp_path / "legacy")
    assert reads.import_legacy_reads(tmp_path / "work", legacy.kraken) == 2
    assert {r.reader for r in reads.load_reads(tmp_path / "work")} == {"kraken:MiDRASH_Gen_01"}


def test_import_legacy_reads_dedups_on_sha_and_reader(tmp_path: Path) -> None:
    """A repeat import adds nothing; an existing (sha256, reader) pair is skipped, a new reader for it is added."""
    legacy = write_legacy_caches(tmp_path / "legacy")
    work = tmp_path / "work"
    reads.append_jsonl(reads.reads_path(work), read_record(legacy.sha["k1"], "kraken:MiDRASH_Gen_01"))
    reads.append_jsonl(reads.reads_path(work), read_record(legacy.sha["g_text"], "vlm:model-x"))
    assert reads.import_legacy_reads(work, legacy.kraken, legacy.gemini_index) == 5       # k1 with that reader exists
    assert reads.import_legacy_reads(work, legacy.kraken, legacy.gemini_index) == 0
    loaded = reads.load_reads(work)
    assert len(loaded) == 7
    assert sum(1 for r in loaded if (r.sha256, r.reader) == (legacy.sha["k1"], "kraken:MiDRASH_Gen_01")) == 1
    assert {r.reader for r in loaded if r.sha256 == legacy.sha["g_text"]} == {"vlm:model-x", "gemini:gemini-test"}


def test_import_legacy_reads_repairs_a_cut_off_reads_file(tmp_path: Path) -> None:
    """A last line cut off by an earlier crash is dropped, so the import neither fails nor glues two records."""
    legacy = write_legacy_caches(tmp_path / "legacy")
    work = tmp_path / "work"
    reads.append_jsonl(reads.reads_path(work), read_record("b" * 64))
    with reads.reads_path(work).open("a", encoding="utf-8") as fh:
        fh.write('{"sha256": "trunc')
    assert reads.import_legacy_reads(work, legacy.kraken) == 2
    loaded = reads.load_reads(work)
    assert [r.sha256 for r in loaded][0] == "b" * 64 and len(loaded) == 3


# --- kraken_read -------------------------------------------------------------------------------------------------------

def test_kraken_read_maps_the_response(tmp_path: Path) -> None:
    """The upload is a downscaled copy, the form carries the model, and the lines become the record."""
    photo = make_jpeg(tmp_path / "photo.jpg", size=(300, 150))
    post = FakePost()
    record = reads.kraken_read(photo, url="http://kraken.example.invalid/transcribe_lines", model_path=KRAKEN_MODEL_PATH,
                               long_side=100, post=post)

    (call,) = post.calls
    assert call["url"] == "http://kraken.example.invalid/transcribe_lines"
    assert call["data"] == {"model_path": KRAKEN_MODEL_PATH} and call["timeout"] == 600.0
    assert call["filename"] == "photo_100.jpg"
    assert Image.open(io.BytesIO(call["payload"])).size == (100, 50)

    assert record.sha256 == sha_of(photo) and record.reader == KRAKEN_READER and record.finish == "ok"
    assert record.lines == [HEBREW, HEBREW[:10]] and record.text == f"{HEBREW}\n{HEBREW[:10]}" and record.n_lines == 2
    assert (record.letters_hebrew, record.letters_arabic) == (32, 0)
    assert record.seconds >= 0 and TIMESTAMP.match(record.read_at)
    assert record.meta["confidences"] == [0.9, 0.8]
    assert record.meta["bboxes"] == [[0, 0, 90, 15], [10, 20, 100, 35]]
    assert record.meta["prepared_size"] == [100, 50] and record.meta["scale"] == round(100 / 300, 6)
    assert record.meta["polygon_failures"] == 2 and record.meta["used_binarization"] is True
    assert record.meta["view"]["rotation"] == 0 and record.meta["model_path"] == KRAKEN_MODEL_PATH
    json.dumps(schema.to_dict(record))


def test_kraken_read_does_not_enlarge_a_small_image(tmp_path: Path) -> None:
    """An image under the long side is sent at its own size with scale 1."""
    photo = make_jpeg(tmp_path / "photo.jpg", size=(60, 40))
    post = FakePost()
    record = reads.kraken_read(photo, url="http://kraken.example.invalid/x", model_path=KRAKEN_MODEL_PATH, post=post)
    assert Image.open(io.BytesIO(post.calls[0]["payload"])).size == (60, 40)
    assert record.meta["prepared_size"] == [60, 40] and record.meta["scale"] == 1.0


def test_kraken_read_zero_lines_is_blank(tmp_path: Path) -> None:
    """A response with no lines is a blank read, not an error."""
    photo = make_jpeg(tmp_path / "photo.jpg")
    record = reads.kraken_read(photo, url="http://kraken.example.invalid/x", model_path=KRAKEN_MODEL_PATH,
                               post=FakePost(body=kraken_body(lines=[])))
    assert record.finish == "blank" and record.n_lines == 0 and record.text == "" and record.lines == []
    assert record.meta["confidences"] == [] and not reads.is_usable(record)
    missing = reads.kraken_read(photo, url="http://kraken.example.invalid/x", model_path=KRAKEN_MODEL_PATH,
                                post=FakePost(body={"polygon_failures": 0}))
    assert missing.finish == "blank"
    empty_lines = reads.kraken_read(photo, url="http://kraken.example.invalid/x", model_path=KRAKEN_MODEL_PATH,
                                    post=FakePost(body=kraken_body(lines=["", "  "])))
    assert empty_lines.finish == "blank" and not reads.is_usable(empty_lines)         # lines found, nothing recognised


@pytest.mark.parametrize("post", [
    FakePost(error=requests.ConnectionError("refused")),
    FakePost(error=requests.Timeout("slow")),
    FakePost(body=kraken_body(), status=500),
    FakePost(body=ValueError("not json")),
    FakePost(body=["not", "a", "dict"]),
    FakePost(body={"lines": ["not a dict"]}),
], ids=["connection", "timeout", "http_500", "bad_json", "bad_body", "bad_line"])
def test_kraken_failures_become_error_records(tmp_path: Path, post: FakePost) -> None:
    """Request and response failures are recorded as ``finish="error"`` with the error type, never raised."""
    photo = make_jpeg(tmp_path / "photo.jpg")
    record = reads.kraken_read(photo, url="http://kraken.example.invalid/x", model_path=KRAKEN_MODEL_PATH, post=post)
    assert record.finish == "error" and record.text == "" and record.lines == [] and record.n_lines == 0
    assert record.sha256 == sha_of(photo) and record.reader == KRAKEN_READER
    assert re.match(r"^\w+: ", record.meta["error"]) and len(record.meta["error"]) <= 320
    assert not reads.is_usable(record)


def test_kraken_read_of_an_unreadable_image_is_an_error_record(tmp_path: Path) -> None:
    """A file PIL cannot decode fails before the request and is recorded, without calling the service."""
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    post = FakePost()
    record = reads.kraken_read(broken, url="http://kraken.example.invalid/x", model_path=KRAKEN_MODEL_PATH, post=post)
    assert record.finish == "error" and "UnidentifiedImageError" in record.meta["error"] and post.calls == []


def test_kraken_read_defaults_come_from_the_module_constants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without arguments the call uses ``KRAKEN_URL``, ``KRAKEN_MODEL`` and ``requests.post``."""
    photo = make_jpeg(tmp_path / "photo.jpg")
    post = FakePost()
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(reads, "KRAKEN_URL", "http://default.example.invalid/transcribe_lines")
    monkeypatch.setattr(reads, "KRAKEN_MODEL", "/models/default_model.mlmodel")
    record = reads.kraken_read(photo)
    assert post.calls[0]["url"] == "http://default.example.invalid/transcribe_lines"
    assert post.calls[0]["data"] == {"model_path": "/models/default_model.mlmodel"}
    assert record.reader == "kraken:default_model"


def test_endpoint_constants_honour_the_environment() -> None:
    """``NOVEL_BUILD_KRAKEN_URL`` and ``NOVEL_BUILD_KRAKEN_MODEL`` override the defaults at import."""
    defaults = (reads.KRAKEN_URL, reads.KRAKEN_MODEL)
    assert defaults[0].endswith("/transcribe_lines") and defaults[1].endswith(".mlmodel")
    try:
        with pytest.MonkeyPatch.context() as env:           # its own patcher: the autouse guard stays in place
            env.setenv("NOVEL_BUILD_KRAKEN_URL", "http://env.example.invalid/transcribe_lines")
            env.setenv("NOVEL_BUILD_KRAKEN_MODEL", "/models/env_model.mlmodel")
            module = importlib.reload(reads)
            assert module.KRAKEN_URL == "http://env.example.invalid/transcribe_lines"
            assert module.KRAKEN_MODEL == "/models/env_model.mlmodel"
    finally:
        importlib.reload(reads)                              # back to the environment of the test session
    assert (reads.KRAKEN_URL, reads.KRAKEN_MODEL) == defaults


# --- vlm_read ----------------------------------------------------------------------------------------------------------

CAPPED_TWICE = {"finish_reason": "length", "first_finish_reason": "length", "retried": True, "attempts": 2,
                "completion_tokens": 1200}


@pytest.mark.parametrize("overrides, finish", [
    ({}, "ok"),
    (dict(CAPPED_TWICE, looped=False), "length"),
    (dict(CAPPED_TWICE, looped=True, cut_at=None, repeat_share=0.02), "length"),
    ({"finish_reason": "length", "looped": True, "retried": False}, "length"),
    (dict(CAPPED_TWICE, looped=True, cut_at=40, repeat_share=0.7), "looping"),
    (dict(CAPPED_TWICE, looped=True, cut_at=None, repeat_share=0.8), "looping"),
    (dict(CAPPED_TWICE, looped=True, cut_at=None, unit_repeat_share=0.7), "looping"),
    ({"looped": True, "repeat_share": 0.8, "cut_at": 40}, "looping"),
    ({"looped": True, "repeat_share": 0.6, "retried": False}, "looping"),
    ({"text": "", "reason": "blank", "blank": True, "finish_reason": None}, "blank"),
    ({"text": "", "reason": "empty"}, "blank"),
    ({"text": None, "reason": "failed", "finish_reason": None}, "error"),
], ids=["ok", "capped_twice_no_loop_flag", "capped_twice_flagged_by_the_cut_only", "capped_no_retry",
        "capped_twice_loop_cut", "capped_twice_repeat_over_gate", "capped_twice_unit_repeat_over_gate",
        "stop_loop_cut", "stop_loop_flag", "blank_gate", "empty", "failed"])
def test_vlm_finish_mapping(overrides: Dict[str, Any], finish: str) -> None:
    """``looping`` only when the loop detector found a loop; a read the budget cut without one is ``length``.

    The page reader marks every budget cut ``looped``; for a cut read the detector's own evidence (a loop cut,
    ``cut_at``, or a repeat share over the gate) decides. For a read that stopped, the flag is the detector's.
    """
    assert reads.vlm_finish(vlm_result(**overrides)) == finish


def test_vlm_finish_keeps_the_loop_flag_of_a_cut_read_without_evidence_fields() -> None:
    """A cut, flagged read whose ``cut_at`` and ``repeat_share`` are absent cannot be cleared: it stays looping."""
    result = vlm_result(**CAPPED_TWICE, looped=True)
    del result["cut_at"], result["repeat_share"]
    assert reads.vlm_finish(result) == "looping"


def test_loop_detected_uses_the_production_gate_strictly() -> None:
    """A repeat share at the gate is not a loop (the reader's own ``>``), just above it is."""
    gate = page_reader.DEFAULT_SETTINGS.loop_gate
    assert gate == 0.5
    capped = dict(CAPPED_TWICE, looped=True, cut_at=None)
    assert reads.loop_detected(dict(capped, repeat_share=gate)) is False
    assert reads.loop_detected(dict(capped, repeat_share=gate + 0.01)) is True
    assert reads.loop_detected({"looped": False, "finish_reason": "stop", "repeat_share": 0.9}) is False


def test_vlm_read_maps_the_result_dict(tmp_path: Path) -> None:
    """A clean read keeps its text, non-empty lines, letters and the reader's generation metadata."""
    photo = make_jpeg(tmp_path / "photo.jpg")
    read_page = fake_read_page(vlm_result(ink={"class": "text", "count": np.int64(3)}))
    record = reads.vlm_read(photo, model_key="model-x", read_page=read_page)

    assert read_page.calls == [("model-x", str(photo))]
    assert record.sha256 == sha_of(photo) and record.reader == "vlm:model-x" and record.finish == "ok"
    assert record.text == f"{ARABIC}\n\n{ARABIC[:12]}" and record.lines == [ARABIC, ARABIC[:12]] and record.n_lines == 2
    assert (record.letters_hebrew, record.letters_arabic) == (0, 40)
    assert record.seconds >= 0 and TIMESTAMP.match(record.read_at)
    for key in ("reason", "finish_reason", "looped", "attempts", "retried", "prompt", "max_tokens", "completion_tokens"):
        assert key in record.meta
    assert record.meta["finish_reason"] == "stop" and record.meta["ink"]["class"] == "text"
    assert "text" not in record.meta and "raw_text" not in record.meta
    json.dumps(schema.to_dict(record))                  # numpy scalars in the reader's ink measures are made plain


@pytest.mark.parametrize("overrides, finish, text, usable", [
    (dict(CAPPED_TWICE, looped=True, cut_at=12, repeat_share=0.9, text=f"{ARABIC}\n{ARABIC[:12]}"), "looping",
     f"{ARABIC}\n{ARABIC[:12]}", False),
    (dict(CAPPED_TWICE, looped=False), "length", f"{ARABIC}\n\n{ARABIC[:12]}", True),
    (dict(CAPPED_TWICE, looped=True, cut_at=None, repeat_share=0.01), "length", f"{ARABIC}\n\n{ARABIC[:12]}", True),
    ({"finish_reason": "length", "looped": True, "retried": False}, "length", f"{ARABIC}\n\n{ARABIC[:12]}", True),
    ({"text": "", "reason": "blank", "blank": True, "finish_reason": None}, "blank", "", False),
    ({"text": "", "reason": "empty"}, "blank", "", False),
], ids=["capped_twice_loop", "capped_twice_no_loop_flag", "capped_twice_flagged_by_the_cut_only", "capped_no_retry",
        "blank_gate", "empty"])
def test_vlm_read_records_looping_length_and_blank(tmp_path: Path, overrides: Dict[str, Any], finish: str, text: str,
                                                   usable: bool) -> None:
    """A loop is looping (unusable); a page capped twice without a loop is length (usable); gated and empty pages are
    blank reads with no lines."""
    photo = make_jpeg(tmp_path / "photo.jpg")
    record = reads.vlm_read(photo, model_key="model-x", read_page=fake_read_page(vlm_result(**overrides)))
    assert record.finish == finish and record.text == text
    assert record.n_lines == (0 if not text else len([ln for ln in text.splitlines() if ln.strip()]))
    assert record.meta["reason"] == overrides.get("reason")
    assert reads.is_usable(record) is usable


def synthetic_page(n_lines: int, seed: int) -> str:
    """Pseudo-random Hebrew-letter lines without repeats (a long page that is not a loop; not a real text).

    :param n_lines: Number of lines.
    :param seed: Random seed.
    :return: Lines of eight words of two to six letters.
    """
    rng = random.Random(seed)
    return "\n".join(" ".join("".join(rng.choice(HEBREW) for _ in range(rng.randint(2, 6))) for _ in range(8))
                     for _ in range(n_lines))


def stub_transcribe(text: str, finish_reason: str) -> Any:
    """An async stand-in for the LM Studio call of ``page_reader.read_page``: every request returns ``text``.

    :param text: The generation.
    :param finish_reason: Its finish reason.
    :return: The coroutine function; its ``calls`` attribute counts the requests.
    """
    calls: List[float] = []

    async def transcribe(model_id: str, image_path: str, prompt: str, temperature: float = 0.1,
                         max_tokens: int = 1200, **kwargs: Any) -> Dict[str, Any]:
        """Stub request.

        :param model_id: Model key.
        :param image_path: Image file.
        :param prompt: Prompt.
        :param temperature: Temperature.
        :param max_tokens: Budget.
        :param kwargs: Ignored.
        :return: A completion's metadata.
        """
        calls.append(temperature)
        return {"text": text, "finish_reason": finish_reason, "prompt_tokens": 6500, "completion_tokens": max_tokens,
                "total_tokens": 6500 + max_tokens}

    transcribe.calls = calls  # type: ignore[attr-defined]
    return transcribe


@pytest.mark.parametrize("kind", ["long_page", "loop", "stop"])
def test_vlm_read_with_the_page_readers_own_loop_handling(tmp_path: Path, kind: str) -> None:
    """Through the real ``read_page`` logic (LM Studio call stubbed, no ink gate): a long page cut by the budget on both
    requests is flagged ``looped`` by the reader yet stored as a usable ``length`` read; a loop is ``looping``."""
    page = synthetic_page(40, seed=7)
    text, finish_reason = {"long_page": (page, "length"),
                           "loop": (page[:200] + "\n" + "אבגד הוזח טיכל מנסע פצקר " * 40, "length"),
                           "stop": (page[:400], "stop")}[kind]
    transcribe = stub_transcribe(text, finish_reason)
    read_page = functools.partial(REAL_READ_PAGE, settings=page_reader.ReaderSettings(blank_gate=False),
                                  transcribe=transcribe)
    record = reads.vlm_read(make_jpeg(tmp_path / "photo.jpg"), model_key="model-x", read_page=read_page)
    if kind == "long_page":
        assert len(transcribe.calls) == 2 and record.meta["retried"] is True       # capped, retried, capped again
        assert record.meta["looped"] is True and record.meta["cut_at"] is None       # flagged by the cut, no loop found
        assert record.finish == "length" and record.text == page and reads.is_usable(record)
    elif kind == "loop":
        assert record.meta["looped"] is True and record.meta["cut_at"] is not None
        assert record.finish == "looping" and not reads.is_usable(record)
    else:
        assert len(transcribe.calls) == 1 and record.finish == "ok" and reads.is_usable(record)


def test_vlm_read_failed_request_is_an_error_record(tmp_path: Path) -> None:
    """The reader's own ``failed`` result and an exception from the call both become error records."""
    photo = make_jpeg(tmp_path / "photo.jpg")
    failed = reads.vlm_read(photo, model_key="model-x",
                            read_page=fake_read_page(vlm_result(text=None, reason="failed", finish_reason=None)))
    assert failed.finish == "error" and failed.text == "" and failed.meta["reason"] == "failed"

    boom = reads.vlm_read(photo, model_key="model-x", read_page=fake_read_page(error=RuntimeError("server went away")))
    assert boom.finish == "error" and boom.meta["error"] == "RuntimeError: server went away"
    assert boom.sha256 == sha_of(photo) and boom.reader == "vlm:model-x"


def test_vlm_read_lets_keyboard_interrupt_through(tmp_path: Path) -> None:
    """Ctrl-C is not swallowed as a failed read."""
    photo = make_jpeg(tmp_path / "photo.jpg")
    with pytest.raises(KeyboardInterrupt):
        reads.vlm_read(photo, model_key="model-x", read_page=fake_read_page(error=KeyboardInterrupt()))


def test_vlm_read_defaults_to_the_production_page_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``read_page`` the call goes to ``page_reader.read_page``, run to completion with ``asyncio.run``."""
    photo = make_jpeg(tmp_path / "photo.jpg")
    stub = fake_read_page()
    monkeypatch.setattr(page_reader, "read_page", stub)
    record = reads.vlm_read(photo, model_key="model-y")
    assert stub.calls == [("model-y", str(photo))] and record.reader == "vlm:model-y" and record.finish == "ok"


# --- select_images -----------------------------------------------------------------------------------------------------

def ids(images: List[ImageRef]) -> List[str]:
    """Readable keys of selected images.

    :param images: Selected images.
    :return: ``["<file name>"]`` per image.
    """
    return [Path(image.path).name for image in images]


def test_select_images_by_script(stage1: SimpleNamespace) -> None:
    """``all`` reads every candidate, ``hebrew``/``arabic`` include mixed ones; too_small and shared images never repeat."""
    assert ids(reads.select_images(stage1.cands, [], script="all")) == ["he.jpg", "ar.jpg"]
    hebrew = reads.select_images(stage1.cands, [], script="hebrew")
    assert [(Path(i.path).name, i.image_id) for i in hebrew] == [("he.jpg", "iiif_pgp:0"), ("ar.jpg", "cudl_local:0")]
    arabic = reads.select_images(stage1.cands, [], script="arabic")
    assert [(Path(i.path).name, i.image_id) for i in arabic] == [("ar.jpg", "iiif_pgp:0")]
    with pytest.raises(ValueError):
        reads.select_images(stage1.cands, [], script="latin")


def test_select_images_never_returns_too_small_or_unfetched(stage1: SimpleNamespace, tmp_path: Path) -> None:
    """Only ``present`` and ``fetched`` images with a path are read."""
    spare = make_jpeg(tmp_path / "spare.jpg", color=(1, 2, 3))
    statuses = ["listed", "missing", "head_404", "duplicate", "too_small"]
    cands = [candidate("r_s", "hebrew", [image_ref(f"iiif_pgp:{i}", spare, status=s) for i, s in enumerate(statuses)]
                       + [image_ref("iiif_pgp:9", None, status="present", sha256="d" * 64)])]
    assert reads.select_images(cands, [], script="all") == []
    cands[0].images.append(image_ref("iiif_pgp:10", spare, status="fetched"))
    assert ids(reads.select_images(cands, [], script="all")) == ["spare.jpg"]


def test_select_images_skips_what_this_reader_has_read(stage1: SimpleNamespace) -> None:
    """An image read by the given reader is not selected again; another reader's read does not hide it."""
    existing = [read_record(stage1.sha_he, "kraken:test_model")]
    assert ids(reads.select_images(stage1.cands, existing, reader_id="kraken:test_model")) == ["ar.jpg"]
    assert ids(reads.select_images(stage1.cands, existing, reader_id="vlm:other")) == ["he.jpg", "ar.jpg"]
    assert ids(reads.select_images(stage1.cands, existing)) == ["he.jpg", "ar.jpg"]


def test_select_images_only_missing_usable(stage1: SimpleNamespace) -> None:
    """With ``only_missing_usable`` an image is kept only when none of its reads (any reader) is usable."""
    unusable = [read_record(stage1.sha_ar, "kraken:test_model", text="א", finish="ok"),                 # under 20 letters
                read_record(stage1.sha_ar, "vlm:old", text=HEBREW, finish="looping")]
    usable_he = read_record(stage1.sha_he, "kraken:test_model")
    both = [usable_he] + unusable
    assert ids(reads.select_images(stage1.cands, both, only_missing_usable=True, reader_id="vlm:model-x")) == ["ar.jpg"]
    # a usable read from any reader, even a different one, removes the image
    rescued = both + [read_record(stage1.sha_ar, "gemini:g", text=ARABIC)]
    assert reads.select_images(stage1.cands, rescued, only_missing_usable=True, reader_id="vlm:model-x") == []
    # no read at all counts as missing
    assert ids(reads.select_images(stage1.cands, [usable_he], only_missing_usable=True)) == ["ar.jpg"]
    # the per-reader skip still applies on top
    assert reads.select_images(stage1.cands, both, only_missing_usable=True, reader_id="vlm:old") == []


@pytest.mark.parametrize("finish, text, lines", [("blank", "", 0), ("error", "", 0), ("looping", HEBREW, 1),
                                                 ("ok", HEBREW[:19], 1), ("length", HEBREW[:19], 1)])
def test_unusable_reads_do_not_hide_an_image_from_only_missing_usable(stage1: SimpleNamespace, finish: str, text: str,
                                                                       lines: int) -> None:
    """Each unusable kind (no lines, error, looping, under 20 letters) leaves the image selectable."""
    record = reads.make_record(stage1.sha_he, "kraken:test_model", text, finish=finish, seconds=1.0,
                               lines=[text] * lines)
    selected = reads.select_images(stage1.cands, [record], script="hebrew", only_missing_usable=True)
    assert "he.jpg" in ids(selected)


def test_a_length_read_is_a_usable_read_for_only_missing_usable(stage1: SimpleNamespace) -> None:
    """A read the token budget cut (no loop) counts as usable: the image is not read again for lack of one."""
    capped = read_record(stage1.sha_he, "vlm:model-x", finish="length")
    assert "he.jpg" not in ids(reads.select_images(stage1.cands, [capped], script="hebrew", only_missing_usable=True))


def test_only_missing_usable_reads_a_looping_record_on_disk_as_its_meta_says(stage1: SimpleNamespace) -> None:
    """A record stored as looping whose meta shows a budget cut without a loop is usable and hides the image; one whose
    meta shows a loop does not."""
    no_loop = reads.make_record(stage1.sha_he, "vlm:old", HEBREW, finish="looping", seconds=1.0,
                                meta={"finish_reason": "length", "looped": True, "cut_at": None, "repeat_share": 0.0,
                                      "retried": True})
    loop = reads.make_record(stage1.sha_he, "vlm:old", HEBREW, finish="looping", seconds=1.0,
                             meta={"finish_reason": "length", "looped": True, "cut_at": 30, "repeat_share": 0.9})
    assert "he.jpg" not in ids(reads.select_images(stage1.cands, [no_loop], script="hebrew", only_missing_usable=True))
    assert "he.jpg" in ids(reads.select_images(stage1.cands, [loop], script="hebrew", only_missing_usable=True))


def test_select_images_hashes_an_image_without_sha256(stage1: SimpleNamespace, tmp_path: Path) -> None:
    """A missing digest is computed from the file and returned on a copy; an unhashable image is skipped."""
    new = make_jpeg(tmp_path / "new.jpg", color=(77, 77, 77))
    unhashed = image_ref("iiif_pgp:0", new, sha256=None)
    ghost = image_ref("iiif_pgp:1", tmp_path / "ghost.jpg", sha256=None)
    cands = [candidate("r_new", "hebrew", [unhashed, ghost])]
    (selected,) = reads.select_images(cands, [])
    assert selected.sha256 == sha_of(new) and unhashed.sha256 is None


def test_select_images_follows_candidate_order_and_reads_each_image_once(stage1: SimpleNamespace) -> None:
    """A photograph listed by two candidates is selected once, at its first occurrence."""
    shuffled = [stage1.cands[2], stage1.cands[0], stage1.cands[1]]
    selected = reads.select_images(shuffled, [], script="all")
    assert [(Path(i.path).name, i.image_id) for i in selected] == [("ar.jpg", "cudl_local:0"), ("he.jpg", "iiif_pgp:0")]


# --- select_images: near-duplicate photographs -----------------------------------------------------------------------

PHASH = 0x9C3A5E7102B4D6F8              # an arbitrary 64-bit perceptual hash


def phash_apart(bits: int) -> str:
    """A 64-bit hex perceptual hash that differs from :data:`PHASH` in exactly ``bits`` bits.

    :param bits: How many of the lowest bits to flip (0 to 64).
    :return: 16 hex characters.
    """
    return f"{PHASH ^ ((1 << bits) - 1):016x}"


def photographs(tmp_path: Path, phashes: Sequence[Optional[str]], tag: str = "photo",
                shade: int = 0) -> SimpleNamespace:
    """One photograph per phash, each its own file (so its own sha256), listed by one hebrew candidate in that order.

    :param tmp_path: Pytest temp dir.
    :param phashes: Phash of each photograph in candidate order (None: the photograph has no phash).
    :param tag: File name prefix (``<tag><index>.jpg``) and candidate row id (``r_<tag>``).
    :param shade: Added to the green channel, so that two calls with different shades give different files.
    :return: Namespace with ``cands`` (the one candidate), its ``refs`` and the files' ``sha`` values.
    """
    files = [make_jpeg(tmp_path / "photos" / f"{tag}{i}.jpg", color=(30 * i + 20, 90 + shade, 160))
             for i in range(len(phashes))]
    refs = [image_ref(f"iiif_pgp:{i}", f, phash=p) for i, (f, p) in enumerate(zip(files, phashes))]
    return SimpleNamespace(cands=[candidate(f"r_{tag}", "hebrew", refs)], refs=refs, sha=[sha_of(f) for f in files])


def select_counted(cands: List[Candidate], existing: Sequence[ReadRecord], **kwargs: Any) -> Tuple[List[str], int]:
    """:func:`reads.select_images` together with the ``skipped_near_duplicate`` count it reports.

    :param cands: Candidates.
    :param existing: Reads so far.
    :param kwargs: Further :func:`reads.select_images` arguments.
    :return: ``(file names of the selected images, number skipped as near-duplicates)``.
    """
    counts: Counter = Counter()
    selected = reads.select_images(cands, existing, counts=counts, **kwargs)
    return ids(selected), counts["skipped_near_duplicate"]


def test_phash_distance_counts_differing_bits() -> None:
    """The distance is the number of differing bits of two 64-bit hashes; the rule is opt-in (default 0: off)."""
    assert reads._phash_distance(phash_apart(0), phash_apart(0)) == 0
    assert reads._phash_distance(phash_apart(0), phash_apart(2)) == 2
    assert reads._phash_distance(phash_apart(3), phash_apart(10)) == 7
    assert reads._phash_distance("0" * 16, "f" * 16) == 64
    assert reads.NEAR_DUPLICATE_BITS == 0


def test_the_near_duplicate_rule_is_off_by_default(tmp_path: Path) -> None:
    """Two faces of one leaf can be phash twins: without opting in, a twin of a read photograph is still read."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(0)])
    assert select_counted(photos.cands, [read_record(photos.sha[0])]) == (["photo0.jpg", "photo1.jpg"], 0)
    assert select_counted(photos.cands, [read_record(photos.sha[0])], skip_near_duplicates=TWIN_BITS) == (
        ["photo0.jpg"], 1)


def test_select_images_skips_the_near_duplicate_of_a_read_photograph(tmp_path: Path) -> None:
    """A photograph 2 bits from an earlier one with a usable read is skipped and counted; the first is read normally."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(2)])
    first_read = [read_record(photos.sha[0], "kraken:test_model")]
    assert select_counted(photos.cands, first_read, reader_id="vlm:model-x", skip_near_duplicates=TWIN_BITS) == (
        ["photo0.jpg"], 1)
    # with only_missing_usable the read photograph leaves too; its near-duplicate is still what the rule removed
    assert select_counted(photos.cands, first_read, only_missing_usable=True, skip_near_duplicates=TWIN_BITS) == ([], 1)
    # nothing is read yet: the rule does not fire, so a pair fetched together is read together
    assert select_counted(photos.cands, [], skip_near_duplicates=TWIN_BITS) == (["photo0.jpg", "photo1.jpg"], 0)
    # the count is added to the counter the caller passes in
    counts = Counter({"skipped_near_duplicate": 2})
    reads.select_images(photos.cands, first_read, counts=counts, skip_near_duplicates=TWIN_BITS)
    assert counts["skipped_near_duplicate"] == 3


@pytest.mark.parametrize("finish, text, lines", [("blank", "", 0), ("error", "", 0), ("looping", HEBREW, 1),
                                                 ("ok", HEBREW[:19], 1)])
def test_an_unusable_read_does_not_make_its_twin_a_near_duplicate(tmp_path: Path, finish: str, text: str,
                                                                   lines: int) -> None:
    """Without a usable read on the first photograph nothing covers the second: both are read, nothing is counted."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(2)])
    record = reads.make_record(photos.sha[0], "kraken:test_model", text, finish=finish, seconds=1.0,
                               lines=[text] * lines)
    assert select_counted(photos.cands, [record], skip_near_duplicates=TWIN_BITS) == (["photo0.jpg", "photo1.jpg"], 0)
    assert select_counted(photos.cands, [record], only_missing_usable=True, skip_near_duplicates=TWIN_BITS) == (
        ["photo0.jpg", "photo1.jpg"], 0)


def test_a_length_read_covers_its_twin(tmp_path: Path) -> None:
    """A read the budget cut without a loop is usable, so it covers a near-duplicate like any usable read."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(2)])
    capped = read_record(photos.sha[0], "vlm:model-x", finish="length")
    assert select_counted(photos.cands, [capped], skip_near_duplicates=TWIN_BITS) == (["photo0.jpg"], 1)


@pytest.mark.parametrize("bits, skipped", [(0, True), (2, True), (6, True), (7, False), (10, False), (64, False)])
def test_near_duplicate_distance_is_inclusive(tmp_path: Path, bits: int, skipped: bool) -> None:
    """Hashes up to the threshold apart are near-duplicates (equal hashes too); 10 bits apart are two photographs."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(bits)])
    expected = ["photo0.jpg"] if skipped else ["photo0.jpg", "photo1.jpg"]
    assert select_counted(photos.cands, [read_record(photos.sha[0])], skip_near_duplicates=TWIN_BITS) == (
        expected, int(skipped))


def test_near_duplicate_threshold_is_the_given_number_of_bits(tmp_path: Path) -> None:
    """A wider threshold catches a pair 10 bits apart; a narrower one lets a pair 2 bits apart through."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(10)])
    first_read = [read_record(photos.sha[0])]
    assert select_counted(photos.cands, first_read, skip_near_duplicates=10) == (["photo0.jpg"], 1)
    close = photographs(tmp_path, [phash_apart(0), phash_apart(2)], tag="close", shade=50)
    assert select_counted(close.cands, [read_record(close.sha[0])], skip_near_duplicates=1) == (
        ["close0.jpg", "close1.jpg"], 0)


@pytest.mark.parametrize("off", [0, None])
def test_zero_or_none_turns_the_near_duplicate_rule_off(tmp_path: Path, off: Optional[int]) -> None:
    """``0`` (the CLI's way to disable it) and None read a photograph whose twin already has a usable read."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(2)])
    assert select_counted(photos.cands, [read_record(photos.sha[0])], skip_near_duplicates=off) == (
        ["photo0.jpg", "photo1.jpg"], 0)


def test_a_photograph_without_phash_is_never_skipped_and_covers_nothing(tmp_path: Path) -> None:
    """The rule needs both hashes: an image without one is read, and its own usable read skips no one."""
    no_second = photographs(tmp_path, [phash_apart(0), None])
    assert select_counted(no_second.cands, [read_record(no_second.sha[0])], skip_near_duplicates=TWIN_BITS) == (
        ["photo0.jpg", "photo1.jpg"], 0)
    no_first = photographs(tmp_path, [None, phash_apart(0)], tag="bare", shade=50)
    assert select_counted(no_first.cands, [read_record(no_first.sha[0])], skip_near_duplicates=TWIN_BITS) == (
        ["bare0.jpg", "bare1.jpg"], 0)
    neither = photographs(tmp_path, [None, None], tag="blank", shade=100)
    assert select_counted(neither.cands, [read_record(neither.sha[0])], skip_near_duplicates=TWIN_BITS) == (
        ["blank0.jpg", "blank1.jpg"], 0)


def test_the_first_photograph_of_a_pair_is_never_skipped(tmp_path: Path) -> None:
    """Candidate order decides: only an earlier photograph covers a later one, so a read on the second skips nothing."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(2)])
    assert select_counted(photos.cands, [read_record(photos.sha[1])], only_missing_usable=True,
                          skip_near_duplicates=TWIN_BITS) == (["photo0.jpg"], 0)
    # both read by other readers, neither by this one: the first is read, the second is the near-duplicate
    both = [read_record(photos.sha[0]), read_record(photos.sha[1], "gemini:g", text=ARABIC)]
    assert select_counted(photos.cands, both, reader_id="vlm:model-x", skip_near_duplicates=TWIN_BITS) == (
        ["photo0.jpg"], 1)
    # the same photographs in the other order: now the other one is first
    swapped = [candidate("r_swapped", "hebrew", list(reversed(photos.refs)))]
    assert select_counted(swapped, both, reader_id="vlm:model-x", skip_near_duplicates=TWIN_BITS) == (
        ["photo1.jpg"], 1)


@pytest.mark.parametrize("reader", ["kraken:other", "gemini:g", "vlm:model-x"])
def test_a_usable_read_from_any_reader_covers_the_near_duplicate(tmp_path: Path, reader: str) -> None:
    """The covering read may come from the reader about to run or from any other."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(2)])
    assert select_counted(photos.cands, [read_record(photos.sha[0], reader)], only_missing_usable=True,
                          skip_near_duplicates=TWIN_BITS) == ([], 1)


def test_near_duplicate_rule_leaves_the_distinct_photographs_of_a_candidate(tmp_path: Path) -> None:
    """Of three photographs the near-duplicate goes, the distant one stays."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(2), phash_apart(30)])
    assert select_counted(photos.cands, [read_record(photos.sha[0])], only_missing_usable=True,
                          skip_near_duplicates=TWIN_BITS) == (["photo2.jpg"], 1)


def test_near_duplicates_are_looked_for_inside_one_candidate_only(tmp_path: Path) -> None:
    """A read photograph of one candidate covers nothing in another candidate, whatever its hash."""
    mine = photographs(tmp_path, [phash_apart(0)])
    other = photographs(tmp_path, [phash_apart(2)], tag="other", shade=60)
    assert select_counted(mine.cands + other.cands, [read_record(mine.sha[0])], only_missing_usable=True,
                          skip_near_duplicates=TWIN_BITS) == (["other0.jpg"], 0)


def test_a_dropped_near_duplicate_is_still_read_for_another_candidate(tmp_path: Path) -> None:
    """The skip is per candidate: a photograph another candidate lists without a read twin is selected there."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(2)])
    existing = [read_record(photos.sha[0])]
    lone = candidate("r_lone", "hebrew", [photos.refs[1]])
    assert select_counted(photos.cands + [lone], existing, only_missing_usable=True,
                          skip_near_duplicates=TWIN_BITS) == (["photo1.jpg"], 0)
    # skipped under two candidates, it is still one photograph in the count
    twin = candidate("r_twin", "hebrew", list(photos.refs))
    assert select_counted(photos.cands + [twin], existing, only_missing_usable=True,
                          skip_near_duplicates=TWIN_BITS) == ([], 1)


def test_only_what_the_rule_removes_from_the_queue_is_counted(tmp_path: Path) -> None:
    """A photograph the other rules already dropped is not a near-duplicate skip, even when it has a read twin."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(2)])
    both = [read_record(photos.sha[0]), read_record(photos.sha[1])]
    assert select_counted(photos.cands, both, only_missing_usable=True, skip_near_duplicates=TWIN_BITS) == (
        [], 0)                                                                            # both have a usable read
    done = [read_record(photos.sha[0], "vlm:model-x"), read_record(photos.sha[1], "vlm:model-x")]
    assert select_counted(photos.cands, done, reader_id="vlm:model-x", skip_near_duplicates=TWIN_BITS) == (
        [], 0)                                                                            # this reader read both


def test_select_images_rejects_a_negative_near_duplicate_distance() -> None:
    """A negative distance would silently never match: it is refused."""
    with pytest.raises(ValueError):
        reads.select_images([], [], skip_near_duplicates=-1)


def test_near_duplicate_peers_lists_the_earlier_twins_of_each_listing(tmp_path: Path) -> None:
    """One list per candidate that lists a photograph: its earlier photographs within the distance, never itself."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(2), phash_apart(30), None])
    a, b, c, d = photos.sha
    arabic = candidate("r_arabic", "arabic", [photos.refs[1]])             # lists the second photograph again, alone
    cands = photos.cands + [arabic]
    assert reads._near_duplicate_peers(cands, "all", 6) == {a: [[]], b: [[a], []], c: [[]], d: [[]]}
    assert reads._near_duplicate_peers(cands, "hebrew", 6) == {a: [[]], b: [[a]], c: [[]], d: [[]]}    # no arabic
    assert reads._near_duplicate_peers(cands, "all", 1) == {a: [[]], b: [[], []], c: [[]], d: [[]]}    # 2 bits > 1
    assert reads._near_duplicate_peers(cands, "all", 0) == {} and reads._near_duplicate_peers(cands, "all", None) == {}
    twice = candidate("r_twice", "hebrew", [photos.refs[0], photos.refs[0]])      # a photograph is not its own twin
    assert reads._near_duplicate_peers([twice], "all", 6) == {a: [[], []]}


def test_has_usable_peer_needs_every_listing_covered() -> None:
    """A photograph is covered only when each of its listings has a twin with a usable read."""
    image = ImageRef(image_id="x", origin="iiif_pgp", sha256="s")
    assert reads._has_usable_peer(image, {"s": [["a", "b"]]}, {"b"}) is True
    assert reads._has_usable_peer(image, {"s": [["a"], ["b"]]}, {"b"}) is False        # first listing: no twin read
    assert reads._has_usable_peer(image, {"s": [[]]}, {"b"}) is False                  # a listing without any twin
    assert reads._has_usable_peer(image, {}, {"b"}) is False                           # unknown to peers: not covered
    assert reads._has_usable_peer(image, {"s": [["a"]]}, set()) is False


# --- select_images: candidates with an unplaced side ------------------------------------------------------------------

def side_decision(row_id: str, side_key: str, outcome: str, image_sha256: Optional[str] = None) -> schema.SideDecision:
    """A synthetic stage 2 side decision.

    :param row_id: Candidate id.
    :param side_key: Side.
    :param outcome: One of ``schema.OUTCOMES``.
    :param image_sha256: The photograph it was accepted on, if any.
    :return: The decision (one synthetic reason unless accepted).
    """
    return schema.SideDecision(row_id=row_id, side_key=side_key, outcome=outcome, image_sha256=image_sha256,
                               reasons=[] if outcome == "accepted" else ["synthetic_reason"])


@pytest.fixture
def unplaced_stage(tmp_path: Path) -> SimpleNamespace:
    """A work dir with candidates in four placement states, their reads, and a stage 2 sides file.

    ``r_done``: its one side accepted, its photograph read (usable). ``r_review``: recto accepted, verso needs_review;
    ``rev0`` has a usable Kraken read that placed nothing, ``rev1`` was read by ``vlm:model-x`` (the reader about to
    run). ``r_none``: its side unassigned, its photograph unread. ``r_fresh``: its side accepted, its photograph unread.

    :param tmp_path: Pytest temp dir.
    :return: Namespace with ``work``, ``cands``, ``refs`` and ``sha`` by name, the ``existing`` reads and ``sides``.
    """
    work = tmp_path / "work_unplaced"
    names = ["done", "rev0", "rev1", "none", "fresh"]
    files = {name: make_jpeg(work / "images" / f"{name}.jpg", color=(40 * i + 10, 60, 200 - 30 * i))
             for i, name in enumerate(names)}
    refs = {name: image_ref(f"iiif_pgp:{name}", path) for name, path in files.items()}
    cands = [candidate("r_done", "hebrew", [refs["done"]]),
             candidate("r_review", "hebrew", [refs["rev0"], refs["rev1"]]),
             candidate("r_none", "hebrew", [refs["none"]]),
             candidate("r_fresh", "hebrew", [refs["fresh"]])]
    write_stage1(work, cands)
    sha = {name: sha_of(path) for name, path in files.items()}
    existing = [read_record(sha["done"]), read_record(sha["rev0"]), read_record(sha["rev1"], "vlm:model-x")]
    for record in existing:
        reads.append_jsonl(reads.reads_path(work), record)
    sides = paths.stage_path(work, "sides")
    schema.write_jsonl(sides, [side_decision("r_done", "recto", "accepted"),
                               side_decision("r_review", "recto", "accepted"),
                               side_decision("r_review", "verso", "needs_review"),
                               side_decision("r_none", "", "unassigned"),
                               side_decision("r_fresh", "page", "accepted")])
    return SimpleNamespace(work=work, cands=cands, refs=refs, sha=sha, existing=existing, sides=sides)


def test_unplaced_candidates_are_those_with_a_side_not_accepted(unplaced_stage: SimpleNamespace) -> None:
    """One side needs_review, unassigned or dropped puts a candidate in; all sides accepted keeps it out."""
    assert reads.unplaced_candidates(unplaced_stage.sides) == {"r_review", "r_none"}
    dropped = unplaced_stage.work / "dropped.jsonl"
    schema.write_jsonl(dropped, [side_decision("r_a", "recto", "dropped"), side_decision("r_b", "recto", "accepted")])
    assert reads.unplaced_candidates(dropped) == {"r_a"}
    with pytest.raises(FileNotFoundError):
        reads.unplaced_candidates(unplaced_stage.work / "missing.jsonl")


def test_only_unplaced_selects_the_photographs_this_reader_has_not_read(unplaced_stage: SimpleNamespace) -> None:
    """The accepted-only candidate is skipped; the needs_review candidate's photograph is selected although another
    reader's usable read exists, its photograph read by this reader is not; the unassigned one's is selected."""
    unplaced = reads.unplaced_candidates(unplaced_stage.sides)
    counts: Counter = Counter()
    selected = reads.select_images(unplaced_stage.cands, unplaced_stage.existing, reader_id="vlm:model-x",
                                   unplaced=unplaced, counts=counts)
    assert ids(selected) == ["rev0.jpg", "none.jpg"]
    assert counts["selected_unplaced"] == 2 and "selected_missing_usable" not in counts
    # the usable read of rev0 is what keeps it out of --only-missing-usable
    assert ids(reads.select_images(unplaced_stage.cands, unplaced_stage.existing, reader_id="vlm:model-x",
                                   only_missing_usable=True)) == ["none.jpg", "fresh.jpg"]
    # the script filter still applies, and an empty set selects nothing
    assert reads.select_images(unplaced_stage.cands, unplaced_stage.existing, script="arabic", unplaced=unplaced) == []
    assert reads.select_images(unplaced_stage.cands, unplaced_stage.existing, unplaced=set()) == []


def test_only_unplaced_and_only_missing_usable_select_the_union(unplaced_stage: SimpleNamespace) -> None:
    """Together they select what either selects, in candidate order; each selection is counted (overlap in both)."""
    counts: Counter = Counter()
    selected = reads.select_images(unplaced_stage.cands, unplaced_stage.existing, reader_id="vlm:model-x",
                                   only_missing_usable=True, unplaced=reads.unplaced_candidates(unplaced_stage.sides),
                                   counts=counts)
    assert ids(selected) == ["rev0.jpg", "none.jpg", "fresh.jpg"]
    assert counts["selected_missing_usable"] == 2 and counts["selected_unplaced"] == 2         # none.jpg is in both


def test_a_photograph_a_placed_candidate_lists_first_is_selected_for_an_unplaced_one(
        unplaced_stage: SimpleNamespace) -> None:
    """Selection by placement is per listing: the placed candidate listing the photograph first does not hide it."""
    sharing = candidate("r_sharing", "hebrew", [unplaced_stage.refs["done"]])
    selected = reads.select_images(unplaced_stage.cands + [sharing], unplaced_stage.existing, reader_id="vlm:model-x",
                                   unplaced={"r_sharing"})
    assert [(Path(i.path).name, i.image_id) for i in selected] == [("done.jpg", "iiif_pgp:done")]


# --- select_images: sides whose quality asks for a coverage review ----------------------------------------------------

def quality_result(row_id: str, side_key: str, reasons: Sequence[str]) -> schema.QualityResult:
    """A synthetic stage 3 quality result.

    :param row_id: Candidate id.
    :param side_key: Side.
    :param reasons: Its reason codes.
    :return: The result (``ok`` when there is no reason).
    """
    return schema.QualityResult(row_id=row_id, side_key=side_key, ok=not reasons, reasons=list(reasons), script="hebrew",
                                letters_hebrew=60, letters_arabic=0, tokens=None, text=HEBREW)


@pytest.fixture
def coverage_stage(tmp_path: Path) -> SimpleNamespace:
    """A work dir whose quality file flags sides for a coverage review, with its sides file and reads.

    ``r_cov``: recto accepted on ``cov1`` and flagged ``target_exceeds_read_2_5x``, verso accepted on ``cov0`` and
    failing only ``under_40_letters``; ``cov2`` is too small. ``r_noacc``: its side flagged
    ``coverage_second_read_wanted`` has no accepted photograph (needs_review); ``na2`` is too small. ``r_ok``: accepted,
    no reason. ``r_read``: flagged ``partial_image_for_target`` on ``rd0``, which ``vlm:model-x`` has read.
    ``cov0``, ``cov1`` and ``na0`` have usable Kraken reads.

    :param tmp_path: Pytest temp dir.
    :return: Namespace with ``work``, ``cands``, ``sha`` by name, the ``existing`` reads, ``sides`` and ``quality``.
    """
    work = tmp_path / "work_coverage"
    names = ["cov0", "cov1", "cov2", "na0", "na1", "na2", "ok0", "rd0"]
    files = {name: make_jpeg(work / "images" / f"{name}.jpg", color=(25 * i + 5, 120, 230 - 25 * i))
             for i, name in enumerate(names)}
    refs = {name: image_ref(f"iiif_pgp:{name}", path, status="too_small" if name in ("cov2", "na2") else "present")
            for name, path in files.items()}
    cands = [candidate("r_cov", "hebrew", [refs["cov0"], refs["cov1"], refs["cov2"]]),
             candidate("r_noacc", "hebrew", [refs["na0"], refs["na1"], refs["na2"]]),
             candidate("r_ok", "hebrew", [refs["ok0"]]),
             candidate("r_read", "hebrew", [refs["rd0"]])]
    write_stage1(work, cands)
    sha = {name: sha_of(path) for name, path in files.items()}
    existing = [read_record(sha["cov0"]), read_record(sha["cov1"]), read_record(sha["na0"]),
                read_record(sha["rd0"], "vlm:model-x")]
    for record in existing:
        reads.append_jsonl(reads.reads_path(work), record)
    sides = paths.stage_path(work, "sides")
    schema.write_jsonl(sides, [side_decision("r_cov", "recto", "accepted", sha["cov1"]),
                               side_decision("r_cov", "verso", "accepted", sha["cov0"]),
                               side_decision("r_noacc", "page", "needs_review"),
                               side_decision("r_ok", "page", "accepted", sha["ok0"]),
                               side_decision("r_read", "recto", "accepted", sha["rd0"])])
    quality = paths.stage_path(work, "quality")
    schema.write_jsonl(quality, [quality_result("r_cov", "recto", ["target_exceeds_read_2_5x"]),
                                 quality_result("r_cov", "verso", ["under_40_letters"]),
                                 quality_result("r_noacc", "page", ["coverage_second_read_wanted"]),
                                 quality_result("r_ok", "page", []),
                                 quality_result("r_read", "recto", ["partial_image_for_target"])])
    return SimpleNamespace(work=work, cands=cands, sha=sha, existing=existing, sides=sides, quality=quality)


def test_coverage_review_sides_maps_each_flagged_side_to_its_accepted_photograph(
        coverage_stage: SimpleNamespace) -> None:
    """Only sides with a coverage reason are listed, each with its accepted photograph or None."""
    assert reads.coverage_review_sides(coverage_stage.quality, coverage_stage.sides) == {
        "r_cov": [coverage_stage.sha["cov1"]], "r_noacc": [None], "r_read": [coverage_stage.sha["rd0"]]}


@pytest.mark.parametrize("reason", sorted(reads.COVERAGE_REVIEW_REASONS))
def test_every_coverage_reason_asks_for_a_second_read(tmp_path: Path, reason: str) -> None:
    """Each of the four coverage gates and the flag selects the side, next to any other reason."""
    quality, sides = tmp_path / "quality.jsonl", tmp_path / "sides.jsonl"
    schema.write_jsonl(quality, [quality_result("r_x", "recto", ["under_40_letters", reason])])
    schema.write_jsonl(sides, [side_decision("r_x", "recto", "accepted", "a" * 64)])
    assert reads.coverage_review_sides(quality, sides) == {"r_x": ["a" * 64]}
    assert reads.COVERAGE_REVIEW_REASONS == {"target_exceeds_read_2_5x", "partial_image_for_target",
                                             "read_exceeds_target_1_5x", "kraken_lines_over_1_5x",
                                             "coverage_second_read_wanted"}


def test_coverage_photographs_per_side() -> None:
    """A side asks for its accepted photograph; a side without one (or with one not present) for every photograph."""
    present = {"a", "b", "c"}
    assert reads._coverage_photographs(["b"], present) == {"b"}
    assert reads._coverage_photographs(["a", "c"], present) == {"a", "c"}
    assert reads._coverage_photographs(["b", None], present) == present
    assert reads._coverage_photographs(["z"], present) == present
    assert reads._coverage_photographs([], present) == set()


def test_a_coverage_review_side_selects_its_accepted_photograph_only(coverage_stage: SimpleNamespace) -> None:
    """The accepted photograph is selected although it has a usable read; the candidate's other photograph is not."""
    counts: Counter = Counter()
    coverage = reads.coverage_review_sides(coverage_stage.quality, coverage_stage.sides)
    selected = reads.select_images(coverage_stage.cands[:1], coverage_stage.existing, reader_id="vlm:model-x",
                                   coverage=coverage, counts=counts)
    assert ids(selected) == ["cov1.jpg"] and counts["selected_coverage_review"] == 1


def test_a_side_without_an_accepted_photograph_selects_every_present_photograph(coverage_stage: SimpleNamespace) -> None:
    """No accepted photograph: every readable photograph of the candidate (never a too-small one); photographs this
    reader read and candidates without a coverage reason are left out."""
    coverage = reads.coverage_review_sides(coverage_stage.quality, coverage_stage.sides)
    selected = reads.select_images(coverage_stage.cands, coverage_stage.existing, reader_id="vlm:model-x",
                                   coverage=coverage)
    assert ids(selected) == ["cov1.jpg", "na0.jpg", "na1.jpg"]
    # an accepted photograph the candidate does not list as readable falls back the same way
    stale = {"r_cov": ["f" * 64]}
    assert ids(reads.select_images(coverage_stage.cands, coverage_stage.existing, coverage=stale)) == [
        "cov0.jpg", "cov1.jpg"]


def test_coverage_review_unions_with_the_other_selections_and_counts_each(coverage_stage: SimpleNamespace) -> None:
    """All three selections together: the union, each selection counted (overlaps in each)."""
    counts: Counter = Counter()
    selected = reads.select_images(coverage_stage.cands, coverage_stage.existing, reader_id="vlm:model-x",
                                   only_missing_usable=True, unplaced=reads.unplaced_candidates(coverage_stage.sides),
                                   coverage=reads.coverage_review_sides(coverage_stage.quality, coverage_stage.sides),
                                   counts=counts)
    assert ids(selected) == ["cov1.jpg", "na0.jpg", "na1.jpg", "ok0.jpg"]
    assert counts["selected_coverage_review"] == 3                          # cov1, na0, na1
    assert counts["selected_unplaced"] == 2 and counts["selected_missing_usable"] == 2      # na0, na1; na1, ok0


# --- run_reads ---------------------------------------------------------------------------------------------------------

class ScriptedReader:
    """A ``read_fn`` stub: records its calls, fails for chosen call numbers or files."""

    def __init__(self, fail_calls: Sequence[int] = (), reader: str = KRAKEN_READER, text: str = HEBREW,
                 hook: Optional[Any] = None, finish: str = "ok") -> None:
        """Configure the stub.

        :param fail_calls: 1-based call numbers that return an ``error`` record.
        :param reader: Reader id stamped on the records.
        :param text: Text of a successful read.
        :param hook: Called as ``hook(call number, path)`` before the record is made.
        :param finish: ``finish`` of the records that are not failures.
        """
        self.fail_calls, self.reader, self.text, self.hook, self.calls = set(fail_calls), reader, text, hook, []
        self.finish = finish

    def __call__(self, path: Path) -> ReadRecord:
        """Read one file.

        :param path: Image file.
        :return: A record (``error`` for the scripted failures).
        """
        self.calls.append(path)
        if self.hook is not None:
            self.hook(len(self.calls), path)
        sha = sha_of(path)
        if len(self.calls) in self.fail_calls:
            return reads.error_record(sha, self.reader, RuntimeError("service down"), 0.2)
        return reads.make_record(sha, self.reader, self.text, finish=self.finish, seconds=0.5)


def many_images(tmp_path: Path, n: int) -> SimpleNamespace:
    """A stage 1 file with ``n`` hebrew candidates of one distinct image each.

    :param tmp_path: Pytest temp dir.
    :param n: Number of images.
    :return: Namespace with ``work`` and the image ``files``.
    """
    work = tmp_path / "work_many"
    files = [make_jpeg(work / "images" / f"img{i}.jpg", color=(i * 20 % 256, i * 7 % 256, 100 + i)) for i in range(n)]
    write_stage1(work, [candidate(f"r{i}", "hebrew", [image_ref("iiif_pgp:0", f)]) for i, f in enumerate(files)])
    return SimpleNamespace(work=work, files=files)


def run(work: Path, read_fn: Any, **kwargs: Any) -> Dict[str, int]:
    """Run :func:`reads.run_reads` with the test reader id.

    :param work: Work dir.
    :param read_fn: Reader stub.
    :param kwargs: Further :func:`reads.run_reads` arguments.
    :return: The counts.
    """
    options: Dict[str, Any] = {"reader": "kraken", "model": KRAKEN_MODEL_PATH}
    options.update(kwargs)
    return reads.run_reads(work, read_fn=read_fn, **options)


def test_run_reads_reads_every_image_once_and_appends_records(stage1: SimpleNamespace) -> None:
    """Both distinct photographs are read; records land in ``reads/reads.jsonl`` and nothing in the failures file."""
    reader = ScriptedReader()
    counts = run(stage1.work, reader)
    assert counts["queued"] == 2 and counts["read"] == 2 and counts["failed"] == 0 and counts["finish_ok"] == 2
    assert counts["stopped_by_stop_file"] == 0 and counts["stopped_by_failures"] == 0
    assert [p.name for p in reader.calls] == ["he.jpg", "ar.jpg"]
    stored = reads.load_reads(stage1.work)
    assert [r.sha256 for r in stored] == [stage1.sha_he, stage1.sha_ar] and reads.reads_path(stage1.work).parent.name == "reads"
    assert not reads.failures_path(stage1.work).exists()


def test_run_reads_is_resumable(stage1: SimpleNamespace) -> None:
    """A re-run reads only what this reader has not read; a finished run reads nothing."""
    first = run(stage1.work, ScriptedReader(), limit=1)
    assert first["queued"] == 1 and first["read"] == 1
    second_reader = ScriptedReader()
    second = run(stage1.work, second_reader)
    assert second["queued"] == 1 and [p.name for p in second_reader.calls] == ["ar.jpg"]
    third_reader = ScriptedReader()
    assert run(stage1.work, third_reader)["queued"] == 0 and third_reader.calls == []
    assert [r.sha256 for r in reads.load_reads(stage1.work)] == [stage1.sha_he, stage1.sha_ar]
    # another reader still has everything to read
    other = ScriptedReader(reader="vlm:model-x")
    assert run(stage1.work, other, reader="vlm", model="model-x")["queued"] == 2


def test_run_reads_flushes_each_record_before_the_next_read(stage1: SimpleNamespace) -> None:
    """When the second image is being read, the first record is already in the file."""
    seen: List[int] = []

    def hook(call: int, path: Path) -> None:
        """Count the records on disk at the start of each read."""
        reads_file = reads.reads_path(stage1.work)
        seen.append(len(jsonl_dicts(reads_file)) if reads_file.exists() else 0)

    run(stage1.work, ScriptedReader(hook=hook))
    assert seen == [0, 1]


def test_run_reads_limit_counts_attempts(tmp_path: Path) -> None:
    """``limit`` caps the images attempted in a run (``0`` attempts none)."""
    stage = many_images(tmp_path, 4)
    assert run(stage.work, ScriptedReader(), limit=0)["queued"] == 0
    counts = run(stage.work, ScriptedReader(fail_calls=[1]), limit=2)
    assert counts["queued"] == 2 and counts["read"] + counts["failed"] == 2


def test_run_reads_stops_when_the_stop_file_appears(tmp_path: Path) -> None:
    """The stop file is checked before every image: a run that creates it mid-way ends after that image."""
    stage = many_images(tmp_path, 4)
    stop = stage.work / "reads" / "STOP"

    def create_stop_file_during_second_read(call: int, path: Path) -> None:
        """Someone touches the stop file while the second image is being read."""
        if call == 2:
            stop.parent.mkdir(parents=True, exist_ok=True)
            stop.write_text("")

    reader = ScriptedReader(hook=create_stop_file_during_second_read)
    counts = run(stage.work, reader, stop_file=stop)
    assert counts["read"] == 2 and counts["stopped_by_stop_file"] == 1 and len(reader.calls) == 2
    assert len(reads.load_reads(stage.work)) == 2
    # a stop file that exists from the start prevents the first read; removing it resumes the work
    untouched = ScriptedReader()
    again = run(stage.work, untouched, stop_file=stop)
    assert again["read"] == 0 and again["stopped_by_stop_file"] == 1 and untouched.calls == []
    stop.unlink()
    assert run(stage.work, ScriptedReader(), stop_file=stop)["read"] == 2


def test_run_reads_stops_after_five_consecutive_failures(tmp_path: Path) -> None:
    """Five failures in a row end the run; the sixth image is not attempted and nothing reaches the reads file."""
    stage = many_images(tmp_path, 8)
    reader = ScriptedReader(fail_calls=[1, 2, 3, 4, 5])
    counts = run(stage.work, reader)
    assert counts["failed"] == 5 and counts["read"] == 0 and counts["stopped_by_failures"] == 1 and len(reader.calls) == 5
    assert not reads.reads_path(stage.work).exists()
    failures = jsonl_dicts(reads.failures_path(stage.work))
    assert len(failures) == 5
    assert failures[0]["sha256"] == sha_of(stage.files[0]) and failures[0]["reader"] == KRAKEN_READER
    assert failures[0]["path"] == str(stage.files[0]) and failures[0]["error"] == "RuntimeError: service down"
    assert TIMESTAMP.match(failures[0]["at"])
    # the failed images were not marked as read: the next run attempts all eight again
    rerun = ScriptedReader()
    assert run(stage.work, rerun)["read"] == 8 and len(rerun.calls) == 8


def test_run_reads_a_success_resets_the_failure_streak(tmp_path: Path) -> None:
    """Failures separated by successes never reach five in a row."""
    stage = many_images(tmp_path, 9)
    reader = ScriptedReader(fail_calls=[1, 2, 3, 4, 6, 7, 8])
    counts = run(stage.work, reader)
    assert len(reader.calls) == 9 and counts["failed"] == 7 and counts["read"] == 2 and counts["stopped_by_failures"] == 0
    assert len(jsonl_dicts(reads.failures_path(stage.work))) == 7 and len(reads.load_reads(stage.work)) == 2


def test_run_reads_records_a_missing_file_and_a_stale_hash_as_failures(tmp_path: Path) -> None:
    """A file gone since stage 1 is not read; a record whose digest disagrees with stage 1 is not stored."""
    stage = many_images(tmp_path, 3)
    stage.files[0].unlink()
    changed = make_jpeg(stage.files[1], color=(5, 5, 5))               # same path, new bytes: stage 1 is stale
    reader = ScriptedReader()
    counts = run(stage.work, reader)
    assert counts["failed"] == 2 and counts["read"] == 1 and [p.name for p in reader.calls] == ["img1.jpg", "img2.jpg"]
    failures = jsonl_dicts(reads.failures_path(stage.work))
    assert failures[0]["error"] == "image_missing"
    assert failures[1]["error"].startswith("sha256_mismatch") and sha_of(changed) in failures[1]["error"]
    assert [r.sha256 for r in reads.load_reads(stage.work)] == [sha_of(stage.files[2])]


def test_run_reads_stores_blank_and_looping_reads_as_reads(tmp_path: Path) -> None:
    """A blank or looping read is a read (kept, counted, not retried by the same reader), not a failure."""
    stage = many_images(tmp_path, 3)
    counts = run(stage.work, ScriptedReader(text="", finish="blank"))
    assert counts["read"] == 3 and counts["failed"] == 0 and counts["finish_blank"] == 3
    assert not reads.failures_path(stage.work).exists()
    assert run(stage.work, ScriptedReader())["queued"] == 0
    looping = run(stage.work, ScriptedReader(reader="vlm:model-x", finish="looping"), reader="vlm", model="model-x",
                  only_missing_usable=True)
    assert looping["queued"] == 3 and looping["finish_looping"] == 3             # a blank read is not a usable read


def test_run_reads_prints_progress_every_25_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture) -> None:
    """The progress line appears at every ``PROGRESS_EVERY``-th read (25 by default)."""
    assert reads.PROGRESS_EVERY == 25 and reads.MAX_CONSECUTIVE_FAILURES == 5
    monkeypatch.setattr(reads, "PROGRESS_EVERY", 2)
    stage = many_images(tmp_path, 5)
    run(stage.work, ScriptedReader())
    out = capsys.readouterr().out
    assert out.count(" read this run of 5") == 2 and "2 read this run of 5" in out and "4 read this run of 5" in out


def test_run_reads_only_missing_usable_after_a_legacy_import(stage1: SimpleNamespace, tmp_path: Path) -> None:
    """The import makes earlier reads count: the page reader is run only on photographs left without a usable read."""
    kraken = tmp_path / "kraken_reads.jsonl"
    good = {"sha256": stage1.sha_he, "lines": [{"text": HEBREW, "bbox": [0, 0, 1, 1], "confidence": 0.9}], "seconds": 1.0,
            "read_at": "2026-10-30T18:10:06Z", "model_path": "/app/models/MiDRASH_Gen_01.mlmodel"}
    bad = dict(good, sha256=stage1.sha_ar, lines=[])
    schema.write_jsonl(kraken, [good, bad])
    assert reads.import_legacy_reads(stage1.work, kraken) == 2
    reader = ScriptedReader(reader="vlm:model-x", text=ARABIC)
    counts = run(stage1.work, reader, reader="vlm", model="model-x", only_missing_usable=True)
    assert counts["queued"] == 1 and [p.name for p in reader.calls] == ["ar.jpg"]
    # the Kraken pass with the legacy model finds its photographs already read
    kraken_again = ScriptedReader(reader="kraken:MiDRASH_Gen_01")
    assert run(stage1.work, kraken_again, model="/app/models/MiDRASH_Gen_01.mlmodel")["queued"] == 0


def test_run_reads_skips_near_duplicates_and_reports_the_count(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """The photograph whose twin has a usable read is not requested and is counted; ``0`` queues it again."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(2)])
    work = tmp_path / "work_twins"
    write_stage1(work, photos.cands)
    reads.append_jsonl(reads.reads_path(work), read_record(photos.sha[0], "kraken:test_model"))

    reader = ScriptedReader(reader="vlm:model-x")
    counts = run(work, reader, reader="vlm", model="model-x", skip_near_duplicates=TWIN_BITS)
    assert counts["queued"] == 1 and counts["skipped_near_duplicate"] == 1 and counts["read"] == 1
    assert [p.name for p in reader.calls] == ["photo0.jpg"]
    assert "1 skipped as near-duplicates" in capsys.readouterr().out

    off = ScriptedReader(reader="vlm:model-y")
    counts = run(work, off, reader="vlm", model="model-y", skip_near_duplicates=0)
    assert counts["queued"] == 2 and counts["skipped_near_duplicate"] == 0
    assert [p.name for p in off.calls] == ["photo0.jpg", "photo1.jpg"]


def test_run_reads_skips_a_twin_at_read_time_once_the_first_read_is_usable(tmp_path: Path,
                                                                           capsys: pytest.CaptureFixture) -> None:
    """Both twins are unread at launch, so both are queued: the first is read and its usable read covers the second."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(2)])
    work = tmp_path / "work_twins"
    write_stage1(work, photos.cands)
    reader = ScriptedReader()
    counts = run(work, reader, skip_near_duplicates=TWIN_BITS)
    assert counts["queued"] == 2 and counts["read"] == 1 and counts["skipped_near_duplicate"] == 1
    assert counts["failed"] == 0 and [p.name for p in reader.calls] == ["photo0.jpg"]
    assert [r.sha256 for r in reads.load_reads(work)] == [photos.sha[0]]
    assert not reads.failures_path(work).exists()                                      # a skip is not a failure
    out = capsys.readouterr().out
    assert "photo1.jpg: a near-duplicate of it now has a usable read" in out
    assert "0 skipped as near-duplicates at launch" in out
    # the next run reaches the same verdict before it starts
    again = ScriptedReader()
    counts = run(work, again, skip_near_duplicates=TWIN_BITS)
    assert counts["queued"] == 0 and counts["skipped_near_duplicate"] == 1 and again.calls == []


def test_run_reads_reads_both_twins_when_the_rule_is_off(tmp_path: Path) -> None:
    """``skip_near_duplicates=0`` turns the read-time check off too."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(2)])
    write_stage1(tmp_path / "work_off", photos.cands)
    reader = ScriptedReader()
    counts = run(tmp_path / "work_off", reader, skip_near_duplicates=0)
    assert counts["read"] == 2 and counts["skipped_near_duplicate"] == 0
    assert [p.name for p in reader.calls] == ["photo0.jpg", "photo1.jpg"]


@pytest.mark.parametrize("scripted", [{"text": "", "finish": "blank"}, {"text": HEBREW, "finish": "looping"},
                                      {"fail_calls": [1]}], ids=["blank", "looping", "failed"])
def test_run_reads_reads_the_twin_when_the_first_read_is_not_usable(tmp_path: Path, scripted: Dict[str, Any]) -> None:
    """A blank, looping or failed first read covers nothing, so the second twin is read as well."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(2)])
    write_stage1(tmp_path / "work_twins", photos.cands)
    reader = ScriptedReader(**scripted)
    counts = run(tmp_path / "work_twins", reader, skip_near_duplicates=TWIN_BITS)
    assert counts["queued"] == 2 and counts["read"] + counts["failed"] == 2 and counts["skipped_near_duplicate"] == 0
    assert [p.name for p in reader.calls] == ["photo0.jpg", "photo1.jpg"]


def test_run_reads_reads_a_twin_that_another_candidate_lists_without_its_twin(tmp_path: Path) -> None:
    """Skipping at read time needs every candidate that lists the photograph to have a twin of it read before it."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(2)])
    lone = candidate("r_lone", "hebrew", [photos.refs[1]])                 # lists the second photograph on its own
    write_stage1(tmp_path / "work_lone", photos.cands + [lone])
    reader = ScriptedReader()
    counts = run(tmp_path / "work_lone", reader, skip_near_duplicates=TWIN_BITS)
    assert [p.name for p in reader.calls] == ["photo0.jpg", "photo1.jpg"] and counts["skipped_near_duplicate"] == 0
    # when the other candidate lists the twin too, nothing needs the second photograph
    both = candidate("r_both", "hebrew", list(photos.refs))
    write_stage1(tmp_path / "work_both", photos.cands + [both])
    reader = ScriptedReader()
    counts = run(tmp_path / "work_both", reader, skip_near_duplicates=TWIN_BITS)
    assert [p.name for p in reader.calls] == ["photo0.jpg"] and counts["skipped_near_duplicate"] == 1


def test_run_reads_counts_reads_from_before_the_run_and_from_the_run_alike(tmp_path: Path) -> None:
    """Each listing of a photograph may be covered by a different read: one from before the run, one from this run."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(1), phash_apart(2)])        # all within 6 bits
    old, new, shared = photos.refs
    write_stage1(tmp_path / "work_mixed", [candidate("r_x", "hebrew", [old, shared]),
                                           candidate("r_y", "hebrew", [new, shared])])
    reads.append_jsonl(reads.reads_path(tmp_path / "work_mixed"), read_record(photos.sha[0], "kraken:other"))
    reader = ScriptedReader()
    counts = run(tmp_path / "work_mixed", reader, only_missing_usable=True, skip_near_duplicates=TWIN_BITS)
    # at launch only candidate r_x covers the shared photograph; once photo1 is read, r_y covers it as well
    assert counts["queued"] == 2 and counts["skipped_near_duplicate"] == 1
    assert [p.name for p in reader.calls] == ["photo1.jpg"]


def test_run_reads_reads_near_duplicates_that_sit_in_different_candidates(tmp_path: Path) -> None:
    """A read covers only photographs of its own candidate, at read time as at launch."""
    mine = photographs(tmp_path, [phash_apart(0)])
    other = photographs(tmp_path, [phash_apart(2)], tag="other", shade=60)
    write_stage1(tmp_path / "work_two", mine.cands + other.cands)
    reader = ScriptedReader()
    counts = run(tmp_path / "work_two", reader, skip_near_duplicates=TWIN_BITS)
    assert counts["read"] == 2 and counts["skipped_near_duplicate"] == 0


def test_run_reads_summary_always_has_the_near_duplicate_count(stage1: SimpleNamespace) -> None:
    """Images without a phash are never skipped: the key is there, and it is 0."""
    counts = run(stage1.work, ScriptedReader())
    assert counts["skipped_near_duplicate"] == 0 and counts["queued"] == 2


def test_run_reads_script_filter_and_missing_stage1(stage1: SimpleNamespace, tmp_path: Path) -> None:
    """``script`` limits the candidates; a work dir without a stage 1 file is an error."""
    reader = ScriptedReader()
    assert run(stage1.work, reader, script="arabic")["queued"] == 1 and [p.name for p in reader.calls] == ["ar.jpg"]
    with pytest.raises(FileNotFoundError):
        run(tmp_path / "empty_work", ScriptedReader())


def test_run_reads_repairs_a_cut_off_tail_before_appending(stage1: SimpleNamespace) -> None:
    """A partial last line left by a crash is dropped; new records start on their own line."""
    reads.append_jsonl(reads.reads_path(stage1.work), read_record(stage1.sha_he, "kraken:test_model"))
    with reads.reads_path(stage1.work).open("a", encoding="utf-8") as fh:
        fh.write('{"sha256": "cut off')
    assert run(stage1.work, ScriptedReader())["queued"] == 1
    assert [r.sha256 for r in reads.load_reads(stage1.work)] == [stage1.sha_he, stage1.sha_ar]


def test_run_reads_only_unplaced_reads_and_counts_the_selection(unplaced_stage: SimpleNamespace,
                                                                capsys: pytest.CaptureFixture) -> None:
    """The sides file selects the unread photographs of unplaced candidates; the counts are printed and returned."""
    reader = ScriptedReader(reader="vlm:model-x", text=ARABIC)
    counts = run(unplaced_stage.work, reader, reader="vlm", model="model-x", only_unplaced=unplaced_stage.sides)
    assert [p.name for p in reader.calls] == ["rev0.jpg", "none.jpg"]
    assert counts["queued"] == 2 and counts["candidates_unplaced"] == 2 and counts["selected_unplaced"] == 2
    assert "selected_missing_usable" not in counts
    assert "selected: 2 of 2 candidates with an unplaced side" in capsys.readouterr().out
    again = run(unplaced_stage.work, ScriptedReader(reader="vlm:model-x"), reader="vlm", model="model-x",
                only_unplaced=unplaced_stage.sides)
    assert again["queued"] == 0                                                 # both now read by this reader
    with pytest.raises(FileNotFoundError):
        run(unplaced_stage.work, ScriptedReader(), only_unplaced=unplaced_stage.work / "missing.jsonl")


def test_run_reads_with_both_selections_reads_the_union(unplaced_stage: SimpleNamespace,
                                                        capsys: pytest.CaptureFixture) -> None:
    """``only_missing_usable`` and ``only_unplaced`` together: the union is read, each selection counted."""
    reader = ScriptedReader(reader="vlm:model-x", text=ARABIC)
    counts = run(unplaced_stage.work, reader, reader="vlm", model="model-x", only_unplaced=unplaced_stage.sides,
                 only_missing_usable=True)
    assert [p.name for p in reader.calls] == ["rev0.jpg", "none.jpg", "fresh.jpg"]
    assert counts["queued"] == 3 and counts["selected_missing_usable"] == 2 and counts["selected_unplaced"] == 2
    assert "selected: 2 without a usable read, 2 of 2 candidates with an unplaced side" in capsys.readouterr().out


def test_run_reads_only_missing_usable_honours_a_looping_record_on_disk(stage1: SimpleNamespace) -> None:
    """A record on disk stored as looping whose meta shows a budget cut without a loop counts as a usable read."""
    reads.append_jsonl(reads.reads_path(stage1.work), reads.make_record(
        stage1.sha_ar, "vlm:old", ARABIC, finish="looping", seconds=1.0,
        meta={"finish_reason": "length", "looped": True, "cut_at": None, "repeat_share": 0.01, "retried": True}))
    reader = ScriptedReader(reader="vlm:model-x")
    counts = run(stage1.work, reader, reader="vlm", model="model-x", only_missing_usable=True)
    assert [p.name for p in reader.calls] == ["he.jpg"] and counts["selected_missing_usable"] == 1


def test_run_reads_only_coverage_review_takes_the_sides_file_from_either_option(coverage_stage: SimpleNamespace,
                                                                                capsys: pytest.CaptureFixture) -> None:
    """``sides`` names the sides file, or ``only_unplaced`` does (its selection then joins the union); without either
    the run refuses to start."""
    reader = ScriptedReader(reader="vlm:model-x")
    counts = run(coverage_stage.work, reader, reader="vlm", model="model-x",
                 only_coverage_review=coverage_stage.quality, sides=coverage_stage.sides)
    assert [p.name for p in reader.calls] == ["cov1.jpg", "na0.jpg", "na1.jpg"]
    assert counts["queued"] == 3 and counts["coverage_review_sides"] == 3 and counts["selected_coverage_review"] == 3
    assert "selected: 3 for coverage review of 3 sides" in capsys.readouterr().out

    other = ScriptedReader(reader="vlm:model-y")                # rd0 is unread for this reader
    counts = run(coverage_stage.work, other, reader="vlm", model="model-y",
                 only_coverage_review=coverage_stage.quality, only_unplaced=coverage_stage.sides)
    assert [p.name for p in other.calls] == ["cov1.jpg", "na0.jpg", "na1.jpg", "rd0.jpg"]
    assert counts["selected_coverage_review"] == 4 and counts["selected_unplaced"] == 2
    assert "2 of 1 candidates with an unplaced side, 4 for coverage review of 3 sides" in capsys.readouterr().out
    with pytest.raises(ValueError):
        run(coverage_stage.work, ScriptedReader(), only_coverage_review=coverage_stage.quality)


# --- claims shared by concurrent jobs ----------------------------------------------------------------------------------

def two_jobs(tmp_path: Path, n: int) -> SimpleNamespace:
    """Two snapshot work dirs whose stage 1 files list the same ``n`` photographs, and the claim log they share.

    :param tmp_path: Pytest temp dir.
    :param n: Number of photographs.
    :return: Namespace with the work dirs ``a`` and ``b``, the photograph ``files``, their ``sha`` and ``claims``.
    """
    stage = many_images(tmp_path, n)
    other = tmp_path / "work_other"
    write_stage1(other, list(schema.read_jsonl(paths.stage_path(stage.work, "images"), Candidate)))
    return SimpleNamespace(a=stage.work, b=other, files=stage.files, sha=[sha_of(f) for f in stage.files],
                           claims=tmp_path / "shared" / "claims.jsonl")


def claim_line(sha256: str, reader: str = KRAKEN_READER, job: str = "other-job",
               hours_ago: float = 0.0) -> Dict[str, str]:
    """A claim log line as another job would write it.

    :param sha256: Photograph.
    :param reader: Reader id.
    :param job: Job id.
    :param hours_ago: Age of the claim.
    :return: The line's fields.
    """
    stamp = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"sha256": sha256, "reader": reader, "claimed_at": stamp, "job": job}


def test_a_second_job_skips_what_the_first_claimed(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """One claim line per attempt with the four fields; a job on another work dir skips every claimed photograph."""
    jobs = two_jobs(tmp_path, 3)
    counts = run(jobs.a, ScriptedReader(), claim_file=jobs.claims)
    assert counts["read"] == 3 and counts["claims_written"] == 3 and counts["skipped_claimed"] == 0
    lines = jsonl_dicts(jobs.claims)
    assert [set(line) for line in lines] == [{"sha256", "reader", "claimed_at", "job"}] * 3
    assert [line["sha256"] for line in lines] == jobs.sha and {line["reader"] for line in lines} == {KRAKEN_READER}
    assert {line["job"] for line in lines} == {str(jobs.a.resolve())} and all(TIMESTAMP.match(line["claimed_at"])
                                                                              for line in lines)
    capsys.readouterr()
    second = ScriptedReader()
    counts = run(jobs.b, second, claim_file=jobs.claims)
    assert second.calls == [] and counts["queued"] == 0 and counts["skipped_claimed"] == 3 and counts["read"] == 0
    assert "3 held by other jobs" in capsys.readouterr().out and not reads.reads_path(jobs.b).exists()


def test_ignore_claims_reads_what_another_job_claimed(tmp_path: Path) -> None:
    """``ignore_claims`` reads every photograph and still writes its own claims."""
    jobs = two_jobs(tmp_path, 2)
    run(jobs.a, ScriptedReader(), claim_file=jobs.claims)
    second = ScriptedReader()
    counts = run(jobs.b, second, claim_file=jobs.claims, ignore_claims=True)
    assert len(second.calls) == 2 and counts["read"] == 2 and counts["skipped_claimed"] == 0
    assert counts["claims_written"] == 2 and len(jsonl_dicts(jobs.claims)) == 4


def test_each_attempt_is_claimed_before_its_request(tmp_path: Path) -> None:
    """The claim is in the log when the reader is called; a failed attempt keeps its claim."""
    jobs = two_jobs(tmp_path, 2)
    claimed_first: List[bool] = []

    def check_the_log(call: int, path: Path) -> None:
        """Look at the last claim when the request starts."""
        claimed_first.append(jsonl_dicts(jobs.claims)[-1]["sha256"] == sha_of(path))

    counts = run(jobs.a, ScriptedReader(hook=check_the_log, fail_calls=[2]), claim_file=jobs.claims)
    assert claimed_first == [True, True] and counts["failed"] == 1 and len(jsonl_dicts(jobs.claims)) == 2


def test_only_a_live_claim_of_this_reader_by_another_job_blocks(tmp_path: Path) -> None:
    """Another reader's claim, a claim older than 24 hours and the job's own claim do not block; the rest does."""
    jobs = two_jobs(tmp_path, 4)
    schema.write_jsonl(jobs.claims, [claim_line(jobs.sha[0], reader="vlm:model-x"),     # another reader
                                     claim_line(jobs.sha[1], hours_ago=25),             # expired
                                     claim_line(jobs.sha[2], hours_ago=23),             # live: blocks
                                     claim_line(jobs.sha[3], job="gpu")])               # this job's own
    reader = ScriptedReader()
    counts = run(jobs.a, reader, claim_file=jobs.claims, job_id="gpu")
    assert [p.name for p in reader.calls] == ["img0.jpg", "img1.jpg", "img3.jpg"]
    assert counts["skipped_claimed"] == 1 and counts["claims_written"] == 3
    # under another id the job's earlier claim of img3 is someone else's
    other = ScriptedReader()
    assert run(jobs.b, other, claim_file=jobs.claims, job_id="cpu")["skipped_claimed"] == 4 and other.calls == []


def test_a_restarted_job_reads_what_it_claimed_but_did_not_read(tmp_path: Path) -> None:
    """Same work dir, same default id: the photograph whose read failed is read on restart; another job skips it."""
    jobs = two_jobs(tmp_path, 3)
    first = run(jobs.a, ScriptedReader(fail_calls=[2]), claim_file=jobs.claims)
    assert first["read"] == 2 and first["failed"] == 1
    other = ScriptedReader()
    assert run(jobs.b, other, claim_file=jobs.claims)["skipped_claimed"] == 3 and other.calls == []
    again = ScriptedReader()
    counts = run(jobs.a, again, claim_file=jobs.claims)
    assert [p.name for p in again.calls] == ["img1.jpg"] and counts["skipped_claimed"] == 0


def test_claimed_photographs_do_not_use_up_the_limit(tmp_path: Path) -> None:
    """Claims are applied when the queue is made, before ``limit`` cuts it."""
    jobs = two_jobs(tmp_path, 3)
    run(jobs.a, ScriptedReader(), claim_file=jobs.claims, limit=1)
    second = ScriptedReader()
    counts = run(jobs.b, second, claim_file=jobs.claims, limit=1)
    assert [p.name for p in second.calls] == ["img1.jpg"] and counts["skipped_claimed"] == 1 and counts["queued"] == 1


def test_jobs_running_at_the_same_time_share_progress(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """The log is read again before each read: a photograph another job claims during this run is skipped."""
    jobs = two_jobs(tmp_path, 3)
    other = reads.ClaimLog(jobs.claims, "other-job")

    def other_job_claims_the_third(call: int, path: Path) -> None:
        """While this job reads its first photograph, the other job claims the third."""
        if call == 1:
            assert other.claim(jobs.sha[2], KRAKEN_READER)

    reader = ScriptedReader(hook=other_job_claims_the_third)
    counts = run(jobs.a, reader, claim_file=jobs.claims)
    assert [p.name for p in reader.calls] == ["img0.jpg", "img1.jpg"]
    assert counts["queued"] == 3 and counts["skipped_claimed"] == 1 and counts["read"] == 2
    assert "img2.jpg: claimed by job other-job" in capsys.readouterr().out


def test_when_two_jobs_claim_at_once_the_first_line_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture) -> None:
    """A claim that lands between this job's check and its own line holds the photograph: this job skips it."""
    jobs = two_jobs(tmp_path, 2)
    claim = reads.ClaimLog.claim

    def racing_claim(self: reads.ClaimLog, sha256: str, reader: str) -> bool:
        """The other job appends its line for the first photograph just before this job does."""
        if sha256 == jobs.sha[0] and self.job != "other-job":
            claim(reads.ClaimLog(self.path, "other-job"), sha256, reader)
        return claim(self, sha256, reader)

    monkeypatch.setattr(reads.ClaimLog, "claim", racing_claim)
    reader = ScriptedReader()
    counts = run(jobs.a, reader, claim_file=jobs.claims)
    assert [p.name for p in reader.calls] == ["img1.jpg"]
    assert counts["skipped_claimed"] == 1 and counts["claims_written"] == 2
    assert "img0.jpg: job other-job claimed it first" in capsys.readouterr().out


def test_claim_log_first_live_claim_holds(tmp_path: Path) -> None:
    """Both jobs agree on the holder; pairs are per reader; an expired first claim yields to the next live one."""
    path = tmp_path / "logs" / "claims.jsonl"
    a, b = reads.ClaimLog(path, "a"), reads.ClaimLog(path, "b")
    assert a.refresh() == 0                                           # no log yet
    assert a.claim("s1", "r") is True and b.claim("s1", "r") is False
    a.refresh()
    assert a.holder("s1", "r") == b.holder("s1", "r") == "a"
    assert b.held_by_other("s1", "r") and not a.held_by_other("s1", "r")
    assert b.claim("s1", "other-reader") is True
    assert a.holder("s1", "r", now=reads.time.time() + reads.CLAIM_TTL_S + 60) is None       # every claim expired
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(claim_line("s2", "r", job="a", hours_ago=30)) + "\n")
        fh.write(json.dumps(claim_line("s2", "r", job="b", hours_ago=1)) + "\n")
    a.refresh()
    assert a.holder("s2", "r") == "b"


def test_claim_log_skips_garbled_lines_and_waits_for_a_torn_one(tmp_path: Path) -> None:
    """Complete lines that are not claims are counted and skipped; an unterminated last line is read once complete."""
    path = tmp_path / "claims.jsonl"
    torn = json.dumps(claim_line("s2", job="b"))
    garbled = ['{"sha256": "s3", "reader"', '{"sha256": "s4"}', "[1, 2]", json.dumps(dict(claim_line("s5"),
                                                                                         claimed_at="yesterday"))]
    path.write_text("\n".join([json.dumps(claim_line("s1", job="a"))] + garbled) + "\n" + torn[:20], encoding="utf-8")
    log = reads.ClaimLog(path, "me")
    assert log.refresh() == 1 and log.malformed == 4
    assert log.holder("s1", KRAKEN_READER) == "a" and log.holder("s2", KRAKEN_READER) is None
    with path.open("a", encoding="utf-8") as fh:
        fh.write(torn[20:] + "\n")
    assert log.refresh() == 1 and log.holder("s2", KRAKEN_READER) == "b" and log.malformed == 4
    assert log.refresh() == 0


def test_default_job_id_is_the_absolute_work_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A relative work dir becomes its absolute path, so a restart from anywhere keeps the id."""
    monkeypatch.chdir(tmp_path)
    assert reads.default_job_id(Path("snap")) == str((tmp_path / "snap").resolve())


def test_run_reads_without_a_claim_file_writes_no_claims(stage1: SimpleNamespace, capsys: pytest.CaptureFixture) -> None:
    """The claim counts are in every summary (0 without a log), and the launch line says there is none."""
    counts = run(stage1.work, ScriptedReader())
    assert counts["skipped_claimed"] == 0 and counts["claims_written"] == 0
    assert "no claim file" in capsys.readouterr().out


# --- main --------------------------------------------------------------------------------------------------------------

def test_main_imports_legacy_reads_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                        capsys: pytest.CaptureFixture) -> None:
    """``--import-legacy`` alone imports and exits without starting a reader."""
    legacy = write_legacy_caches(tmp_path / "legacy")
    monkeypatch.setattr(reads, "run_reads", lambda *a, **k: pytest.fail("no reader run expected"))
    reads.main(["--work-dir", str(tmp_path / "work"), "--import-legacy", str(legacy.kraken),
                "--gemini-index", str(legacy.gemini_index)])
    assert len(reads.load_reads(tmp_path / "work")) == 6 and "imported 6 legacy reads" in capsys.readouterr().out


def test_main_runs_the_kraken_reader_with_the_default_model(stage1: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
                                                            capsys: pytest.CaptureFixture) -> None:
    """``--reader kraken`` builds ``read_fn`` from ``kraken_read`` with the module's URL and model and prints the counts."""
    seen: List[Dict[str, Any]] = []

    def fake_kraken_read(path: Path, **kwargs: Any) -> ReadRecord:
        """Stand-in for :func:`reads.kraken_read`."""
        seen.append(kwargs)
        return read_record(sha_of(path), reads.make_reader_id("kraken", kwargs["model_path"]))

    monkeypatch.setattr(reads, "kraken_read", fake_kraken_read)
    monkeypatch.setattr(reads, "KRAKEN_URL", "http://default.example.invalid/transcribe_lines")
    reads.main(["--work-dir", str(stage1.work), "--reader", "kraken", "--script", "hebrew"])
    assert seen == [{"url": "http://default.example.invalid/transcribe_lines", "model_path": reads.KRAKEN_MODEL}] * 2
    assert {r.reader for r in reads.load_reads(stage1.work)} == {f"kraken:{Path(reads.KRAKEN_MODEL).stem}"}
    out = capsys.readouterr().out
    assert json.loads(out[out.rindex("{\n"):])["read"] == 2            # the run ends by printing its counts as JSON


def test_main_passes_the_options_to_run_reads(stage1: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """The vlm options reach ``run_reads``; the default stop file sits in the reads folder."""
    captured: Dict[str, Any] = {}

    def fake_run_reads(work_dir: Path, **kwargs: Any) -> Dict[str, int]:
        """Capture the call instead of reading."""
        captured.update(kwargs, work_dir=work_dir)
        return {"read": 0}

    monkeypatch.setattr(reads, "run_reads", fake_run_reads)
    reads.main(["--work-dir", str(stage1.work), "--reader", "vlm", "--model", "model-x", "--script", "arabic",
                "--limit", "7", "--only-missing-usable"])
    assert captured["work_dir"] == stage1.work and captured["reader"] == "vlm" and captured["model"] == "model-x"
    assert captured["script"] == "arabic" and captured["limit"] == 7 and captured["only_missing_usable"] is True
    assert captured["skip_near_duplicates"] == reads.NEAR_DUPLICATE_BITS == 0        # off unless asked for
    assert captured["only_unplaced"] is None and captured["claim_file"] is None
    assert captured["ignore_claims"] is False and captured["job_id"] is None
    assert captured["stop_file"] == stage1.work / "reads" / "STOP"
    assert isinstance(captured["read_fn"], reads.functools.partial) and captured["read_fn"].func is reads.vlm_read
    assert captured["read_fn"].keywords == {"model_key": "model-x"}

    stop = stage1.work / "custom.stop"
    reads.main(["--work-dir", str(stage1.work), "--reader", "kraken", "--model", "/m/other.mlmodel", "--stop-file", str(stop)])
    assert captured["stop_file"] == stop and captured["model"] == "/m/other.mlmodel"
    assert captured["read_fn"].keywords["model_path"] == "/m/other.mlmodel" and captured["limit"] is None


@pytest.mark.parametrize("flag, expected", [([], 0), (["--skip-near-duplicates", "0"], 0),
                                            (["--skip-near-duplicates", "3"], 3), (["--skip-near-duplicates", "6"], 6)])
def test_main_passes_the_near_duplicate_distance(stage1: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
                                                 flag: List[str], expected: int) -> None:
    """``--skip-near-duplicates N`` reaches ``run_reads`` as ``skip_near_duplicates``; default 0 (off), opt-in."""
    captured: Dict[str, Any] = {}

    def fake_run_reads(work_dir: Path, **kwargs: Any) -> Dict[str, int]:
        """Capture the call instead of reading."""
        captured.update(kwargs)
        return {"read": 0}

    monkeypatch.setattr(reads, "run_reads", fake_run_reads)
    reads.main(["--work-dir", str(stage1.work), "--reader", "kraken"] + flag)
    assert captured["skip_near_duplicates"] == expected


def test_main_prints_the_near_duplicate_count_in_the_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                             capsys: pytest.CaptureFixture) -> None:
    """The final JSON summary carries ``skipped_near_duplicate``; a photograph with a read twin is not requested."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(2)])
    work = tmp_path / "work_twins"
    write_stage1(work, photos.cands)
    reads.append_jsonl(reads.reads_path(work), read_record(photos.sha[0], "kraken:test_model"))
    requested: List[str] = []

    def fake_kraken_read(path: Path, **kwargs: Any) -> ReadRecord:
        """Stand-in for :func:`reads.kraken_read` that notes which file was requested."""
        requested.append(path.name)
        return read_record(sha_of(path), reads.make_reader_id("kraken", kwargs["model_path"]))

    monkeypatch.setattr(reads, "kraken_read", fake_kraken_read)
    reads.main(["--work-dir", str(work), "--reader", "kraken", "--skip-near-duplicates", str(TWIN_BITS)])
    out = capsys.readouterr().out
    summary = json.loads(out[out.rindex("{\n"):])
    assert summary["skipped_near_duplicate"] == 1 and summary["queued"] == 1 and summary["read"] == 1
    assert requested == ["photo0.jpg"]


def test_main_imports_then_reads(stage1: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With both options the import runs first, so the imported reads are already skipped."""
    kraken = tmp_path / "kraken_reads.jsonl"
    schema.write_jsonl(kraken, [{"sha256": stage1.sha_he, "lines": [{"text": HEBREW}], "seconds": 1.0,
                                 "read_at": "2026-10-30T18:10:06Z", "model_path": "/app/models/MiDRASH_Gen_01.mlmodel"}])
    monkeypatch.setattr(reads, "kraken_read", lambda path, **kw: read_record(sha_of(path), "kraken:MiDRASH_Gen_01"))
    reads.main(["--work-dir", str(stage1.work), "--import-legacy", str(kraken), "--reader", "kraken",
                "--model", "/app/models/MiDRASH_Gen_01.mlmodel"])
    stored = reads.load_reads(stage1.work)
    assert [r.sha256 for r in stored] == [stage1.sha_he, stage1.sha_ar]
    assert stored[0].meta["source"] == "legacy_import" and "source" not in stored[1].meta


def test_main_passes_the_selection_and_claim_options(stage1: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--only-unplaced``, ``--claim-file``, ``--job-id`` and ``--ignore-claims`` reach ``run_reads``."""
    captured: Dict[str, Any] = {}

    def fake_run_reads(work_dir: Path, **kwargs: Any) -> Dict[str, int]:
        """Capture the call instead of reading."""
        captured.update(kwargs)
        return {"read": 0}

    monkeypatch.setattr(reads, "run_reads", fake_run_reads)
    sides = paths.stage_path(stage1.work, "sides")
    schema.write_jsonl(sides, [side_decision("r_he", "", "needs_review")])
    claims = stage1.work / "claims.jsonl"
    reads.main(["--work-dir", str(stage1.work), "--reader", "vlm", "--model", "model-x", "--only-unplaced", str(sides),
                "--only-missing-usable", "--claim-file", str(claims), "--job-id", "gpu", "--ignore-claims"])
    assert captured["only_unplaced"] == sides and captured["only_missing_usable"] is True
    assert captured["claim_file"] == claims and captured["job_id"] == "gpu" and captured["ignore_claims"] is True


def test_main_passes_the_coverage_review_options(coverage_stage: SimpleNamespace,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """``--only-coverage-review`` and ``--sides`` reach ``run_reads``; the same file under ``--sides`` and
    ``--only-unplaced`` is accepted."""
    captured: Dict[str, Any] = {}

    def fake_run_reads(work_dir: Path, **kwargs: Any) -> Dict[str, int]:
        """Capture the call instead of reading."""
        captured.update(kwargs)
        return {"read": 0}

    monkeypatch.setattr(reads, "run_reads", fake_run_reads)
    base = ["--work-dir", str(coverage_stage.work), "--reader", "vlm", "--model", "model-x",
            "--only-coverage-review", str(coverage_stage.quality), "--sides", str(coverage_stage.sides)]
    reads.main(base)
    assert captured["only_coverage_review"] == coverage_stage.quality and captured["sides"] == coverage_stage.sides
    assert captured["only_unplaced"] is None
    reads.main(base + ["--only-unplaced", str(coverage_stage.sides)])
    assert captured["only_unplaced"] == coverage_stage.sides


def test_main_reads_with_a_claim_file_end_to_end(stage1: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture) -> None:
    """Through the command line the claims are written under the job id, and the summary carries their count."""
    monkeypatch.setattr(reads, "kraken_read", lambda path, **kw: read_record(sha_of(path), "kraken:MiDRASH_Gen_01"))
    claims = stage1.work / "claims.jsonl"
    reads.main(["--work-dir", str(stage1.work), "--reader", "kraken", "--claim-file", str(claims), "--job-id", "cpu"])
    out = capsys.readouterr().out
    assert json.loads(out[out.rindex("{\n"):])["claims_written"] == 2
    assert {line["job"] for line in jsonl_dicts(claims)} == {"cpu"}


@pytest.mark.parametrize("argv", [
    [],
    ["--reader", "vlm"],
    ["--gemini-index", "index.jsonl", "--reader", "kraken"],
    ["--reader", "claude"],
    ["--reader", "kraken", "--script", "latin"],
    ["--reader", "kraken", "--skip-near-duplicates", "-1"],
    ["--reader", "kraken", "--skip-near-duplicates", "many"],
    ["--reader", "kraken", "--ignore-claims"],
    ["--reader", "kraken", "--job-id", "gpu"],
    ["--reader", "kraken", "--only-unplaced", "no_such_dir/stage2_sides.jsonl"],
    ["--reader", "kraken", "--only-coverage-review", "no_such_dir/stage3_quality.jsonl"],
    ["--reader", "kraken", "--sides", "no_such_dir/stage2_sides.jsonl"],
    ["--reader", "kraken", "--only-coverage-review", "no_such_dir/q.jsonl", "--sides", "no_such_dir/a.jsonl",
     "--only-unplaced", "no_such_dir/b.jsonl"],
    ["--reader", "kraken", "--only-coverage-review", "no_such_dir/q.jsonl", "--sides", "no_such_dir/a.jsonl"],
])
def test_main_rejects_bad_arguments(tmp_path: Path, argv: List[str]) -> None:
    """Missing reader, vlm without a model, Gemini index without import, unknown choices, bad bit distance, claim
    options without a claim file, coverage review without a sides file, ``--sides`` alone, two different sides files,
    a missing input file: exit 2."""
    with pytest.raises(SystemExit) as raised:
        reads.main(["--work-dir", str(tmp_path / "work")] + argv)
    assert raised.value.code == 2
