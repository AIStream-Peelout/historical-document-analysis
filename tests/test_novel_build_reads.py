"""Tests of stage 2 (reads) of the novel-rows build: legacy import, the two readers, selection and the resumable run.

Every I/O boundary is stubbed: ``post`` for the Kraken service, ``read_page`` for the page reader, ``read_fn`` for the
run. An autouse fixture makes any accidental real request fail loudly (it raises a ``BaseException``, which the
readers' error handling does not swallow). Images are tiny JPEGs made with PIL; texts are synthetic alphabets.
"""
import hashlib
import importlib
import io
import json
import re
from collections import Counter
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
    ({"finish": "length"}, False),
    ({"finish": "error"}, False),
    ({"finish": "blank"}, False),
])
def test_is_usable(kwargs: Dict[str, Any], usable: bool) -> None:
    """A read needs lines, 20 script letters and a clean finish."""
    base = dict(sha256="a" * 64, reader="kraken:m", text="t", lines=["t"], n_lines=1, letters_hebrew=30, letters_arabic=0,
                finish="ok", seconds=1.0, read_at="2026-11-03T00:00:00Z")
    base.update(kwargs)
    assert reads.is_usable(ReadRecord(**base)) is usable


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
    assert records[legacy.sha["g_max"]].finish == "length" and not reads.is_usable(records[legacy.sha["g_max"]])
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

@pytest.mark.parametrize("overrides, finish", [
    ({}, "ok"),
    ({"finish_reason": "length", "looped": True, "retried": True, "attempts": 2}, "looping"),
    ({"finish_reason": "length", "looped": True, "retried": False}, "length"),
    ({"looped": True, "repeat_share": 0.8, "cut_at": 40}, "looping"),
    ({"text": "", "reason": "blank", "blank": True, "finish_reason": None}, "blank"),
    ({"text": "", "reason": "empty"}, "blank"),
    ({"text": None, "reason": "failed", "finish_reason": None}, "error"),
], ids=["ok", "length_after_retry", "length_no_retry", "loop_flag", "blank_gate", "empty", "failed"])
def test_vlm_finish_mapping(overrides: Dict[str, Any], finish: str) -> None:
    """The page reader's reason/finish fields map to ok | looping | length | blank | error."""
    assert reads.vlm_finish(vlm_result(**overrides)) == finish


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


@pytest.mark.parametrize("overrides, finish, text", [
    ({"finish_reason": "length", "looped": True, "retried": True, "attempts": 2, "cut_at": 12, "text": ARABIC[:12]},
     "looping", ARABIC[:12]),
    ({"finish_reason": "length", "looped": True, "retried": False}, "length", f"{ARABIC}\n\n{ARABIC[:12]}"),
    ({"text": "", "reason": "blank", "blank": True, "finish_reason": None}, "blank", ""),
    ({"text": "", "reason": "empty"}, "blank", ""),
])
def test_vlm_read_records_looping_length_and_blank(tmp_path: Path, overrides: Dict[str, Any], finish: str, text: str) -> None:
    """Length with a retry is looping, without a retry length; gated and empty pages are blank reads with no lines."""
    photo = make_jpeg(tmp_path / "photo.jpg")
    record = reads.vlm_read(photo, model_key="model-x", read_page=fake_read_page(vlm_result(**overrides)))
    assert record.finish == finish and record.text == text
    assert record.n_lines == (0 if not text else len([ln for ln in text.splitlines() if ln.strip()]))
    assert record.meta["reason"] == overrides.get("reason")
    assert not reads.is_usable(record)


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


@pytest.mark.parametrize("finish, text, lines", [("blank", "", 0), ("error", "", 0), ("length", HEBREW, 1),
                                                 ("looping", HEBREW, 1), ("ok", HEBREW[:19], 1)])
def test_unusable_reads_do_not_hide_an_image_from_only_missing_usable(stage1: SimpleNamespace, finish: str, text: str,
                                                                       lines: int) -> None:
    """Each unusable kind (no lines, error, length, looping, under 20 letters) leaves the image selectable."""
    record = reads.make_record(stage1.sha_he, "kraken:test_model", text, finish=finish, seconds=1.0,
                               lines=[text] * lines)
    selected = reads.select_images(stage1.cands, [record], script="hebrew", only_missing_usable=True)
    assert "he.jpg" in ids(selected)


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
    """The distance is the number of differing bits of two 64-bit hashes; the default threshold is 6."""
    assert reads._phash_distance(phash_apart(0), phash_apart(0)) == 0
    assert reads._phash_distance(phash_apart(0), phash_apart(2)) == 2
    assert reads._phash_distance(phash_apart(3), phash_apart(10)) == 7
    assert reads._phash_distance("0" * 16, "f" * 16) == 64
    assert reads.NEAR_DUPLICATE_BITS == 6


def test_select_images_skips_the_near_duplicate_of_a_read_photograph(tmp_path: Path) -> None:
    """A photograph 2 bits from an earlier one with a usable read is skipped and counted; the first is read normally."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(2)])
    first_read = [read_record(photos.sha[0], "kraken:test_model")]
    assert select_counted(photos.cands, first_read, reader_id="vlm:model-x") == (["photo0.jpg"], 1)
    # with only_missing_usable the read photograph leaves too; its near-duplicate is still what the rule removed
    assert select_counted(photos.cands, first_read, only_missing_usable=True) == ([], 1)
    # nothing is read yet: the rule does not fire, so a pair fetched together is read together
    assert select_counted(photos.cands, []) == (["photo0.jpg", "photo1.jpg"], 0)
    # the count is added to the counter the caller passes in
    counts = Counter({"skipped_near_duplicate": 2})
    reads.select_images(photos.cands, first_read, counts=counts)
    assert counts["skipped_near_duplicate"] == 3


@pytest.mark.parametrize("finish, text, lines", [("blank", "", 0), ("error", "", 0), ("length", HEBREW, 1),
                                                 ("looping", HEBREW, 1), ("ok", HEBREW[:19], 1)])
def test_an_unusable_read_does_not_make_its_twin_a_near_duplicate(tmp_path: Path, finish: str, text: str,
                                                                   lines: int) -> None:
    """Without a usable read on the first photograph nothing covers the second: both are read, nothing is counted."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(2)])
    record = reads.make_record(photos.sha[0], "kraken:test_model", text, finish=finish, seconds=1.0,
                               lines=[text] * lines)
    assert select_counted(photos.cands, [record]) == (["photo0.jpg", "photo1.jpg"], 0)
    assert select_counted(photos.cands, [record], only_missing_usable=True) == (["photo0.jpg", "photo1.jpg"], 0)


@pytest.mark.parametrize("bits, skipped", [(0, True), (2, True), (6, True), (7, False), (10, False), (64, False)])
def test_near_duplicate_distance_is_inclusive_and_defaults_to_six(tmp_path: Path, bits: int, skipped: bool) -> None:
    """Hashes up to the threshold apart are near-duplicates (equal hashes too); 10 bits apart are two photographs."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(bits)])
    expected = ["photo0.jpg"] if skipped else ["photo0.jpg", "photo1.jpg"]
    assert select_counted(photos.cands, [read_record(photos.sha[0])]) == (expected, int(skipped))


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
    assert select_counted(no_second.cands, [read_record(no_second.sha[0])]) == (["photo0.jpg", "photo1.jpg"], 0)
    no_first = photographs(tmp_path, [None, phash_apart(0)], tag="bare", shade=50)
    assert select_counted(no_first.cands, [read_record(no_first.sha[0])]) == (["bare0.jpg", "bare1.jpg"], 0)
    neither = photographs(tmp_path, [None, None], tag="blank", shade=100)
    assert select_counted(neither.cands, [read_record(neither.sha[0])]) == (["blank0.jpg", "blank1.jpg"], 0)


def test_the_first_photograph_of_a_pair_is_never_skipped(tmp_path: Path) -> None:
    """Candidate order decides: only an earlier photograph covers a later one, so a read on the second skips nothing."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(2)])
    assert select_counted(photos.cands, [read_record(photos.sha[1])], only_missing_usable=True) == (["photo0.jpg"], 0)
    # both read by other readers, neither by this one: the first is read, the second is the near-duplicate
    both = [read_record(photos.sha[0]), read_record(photos.sha[1], "gemini:g", text=ARABIC)]
    assert select_counted(photos.cands, both, reader_id="vlm:model-x") == (["photo0.jpg"], 1)
    # the same photographs in the other order: now the other one is first
    swapped = [candidate("r_swapped", "hebrew", list(reversed(photos.refs)))]
    assert select_counted(swapped, both, reader_id="vlm:model-x") == (["photo1.jpg"], 1)


@pytest.mark.parametrize("reader", ["kraken:other", "gemini:g", "vlm:model-x"])
def test_a_usable_read_from_any_reader_covers_the_near_duplicate(tmp_path: Path, reader: str) -> None:
    """The covering read may come from the reader about to run or from any other."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(2)])
    assert select_counted(photos.cands, [read_record(photos.sha[0], reader)], only_missing_usable=True) == ([], 1)


def test_near_duplicate_rule_leaves_the_distinct_photographs_of_a_candidate(tmp_path: Path) -> None:
    """Of three photographs the near-duplicate goes, the distant one stays."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(2), phash_apart(30)])
    assert select_counted(photos.cands, [read_record(photos.sha[0])], only_missing_usable=True) == (["photo2.jpg"], 1)


def test_near_duplicates_are_looked_for_inside_one_candidate_only(tmp_path: Path) -> None:
    """A read photograph of one candidate covers nothing in another candidate, whatever its hash."""
    mine = photographs(tmp_path, [phash_apart(0)])
    other = photographs(tmp_path, [phash_apart(2)], tag="other", shade=60)
    assert select_counted(mine.cands + other.cands, [read_record(mine.sha[0])], only_missing_usable=True) == (
        ["other0.jpg"], 0)


def test_a_dropped_near_duplicate_is_still_read_for_another_candidate(tmp_path: Path) -> None:
    """The skip is per candidate: a photograph another candidate lists without a read twin is selected there."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(2)])
    existing = [read_record(photos.sha[0])]
    lone = candidate("r_lone", "hebrew", [photos.refs[1]])
    assert select_counted(photos.cands + [lone], existing, only_missing_usable=True) == (["photo1.jpg"], 0)
    # skipped under two candidates, it is still one photograph in the count
    twin = candidate("r_twin", "hebrew", list(photos.refs))
    assert select_counted(photos.cands + [twin], existing, only_missing_usable=True) == ([], 1)


def test_only_what_the_rule_removes_from_the_queue_is_counted(tmp_path: Path) -> None:
    """A photograph the other rules already dropped is not a near-duplicate skip, even when it has a read twin."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(2)])
    both = [read_record(photos.sha[0]), read_record(photos.sha[1])]
    assert select_counted(photos.cands, both, only_missing_usable=True) == ([], 0)       # both have a usable read
    done = [read_record(photos.sha[0], "vlm:model-x"), read_record(photos.sha[1], "vlm:model-x")]
    assert select_counted(photos.cands, done, reader_id="vlm:model-x") == ([], 0)         # this reader read both


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
    counts = run(work, reader, reader="vlm", model="model-x")
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
    counts = run(work, reader)
    assert counts["queued"] == 2 and counts["read"] == 1 and counts["skipped_near_duplicate"] == 1
    assert counts["failed"] == 0 and [p.name for p in reader.calls] == ["photo0.jpg"]
    assert [r.sha256 for r in reads.load_reads(work)] == [photos.sha[0]]
    assert not reads.failures_path(work).exists()                                      # a skip is not a failure
    out = capsys.readouterr().out
    assert "photo1.jpg: a near-duplicate of it now has a usable read" in out
    assert "0 skipped as near-duplicates at launch" in out
    # the next run reaches the same verdict before it starts
    again = ScriptedReader()
    counts = run(work, again)
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
    counts = run(tmp_path / "work_twins", reader)
    assert counts["queued"] == 2 and counts["read"] + counts["failed"] == 2 and counts["skipped_near_duplicate"] == 0
    assert [p.name for p in reader.calls] == ["photo0.jpg", "photo1.jpg"]


def test_run_reads_reads_a_twin_that_another_candidate_lists_without_its_twin(tmp_path: Path) -> None:
    """Skipping at read time needs every candidate that lists the photograph to have a twin of it read before it."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(2)])
    lone = candidate("r_lone", "hebrew", [photos.refs[1]])                 # lists the second photograph on its own
    write_stage1(tmp_path / "work_lone", photos.cands + [lone])
    reader = ScriptedReader()
    counts = run(tmp_path / "work_lone", reader)
    assert [p.name for p in reader.calls] == ["photo0.jpg", "photo1.jpg"] and counts["skipped_near_duplicate"] == 0
    # when the other candidate lists the twin too, nothing needs the second photograph
    both = candidate("r_both", "hebrew", list(photos.refs))
    write_stage1(tmp_path / "work_both", photos.cands + [both])
    reader = ScriptedReader()
    counts = run(tmp_path / "work_both", reader)
    assert [p.name for p in reader.calls] == ["photo0.jpg"] and counts["skipped_near_duplicate"] == 1


def test_run_reads_counts_reads_from_before_the_run_and_from_the_run_alike(tmp_path: Path) -> None:
    """Each listing of a photograph may be covered by a different read: one from before the run, one from this run."""
    photos = photographs(tmp_path, [phash_apart(0), phash_apart(1), phash_apart(2)])        # all within 6 bits
    old, new, shared = photos.refs
    write_stage1(tmp_path / "work_mixed", [candidate("r_x", "hebrew", [old, shared]),
                                           candidate("r_y", "hebrew", [new, shared])])
    reads.append_jsonl(reads.reads_path(tmp_path / "work_mixed"), read_record(photos.sha[0], "kraken:other"))
    reader = ScriptedReader()
    counts = run(tmp_path / "work_mixed", reader, only_missing_usable=True)
    # at launch only candidate r_x covers the shared photograph; once photo1 is read, r_y covers it as well
    assert counts["queued"] == 2 and counts["skipped_near_duplicate"] == 1
    assert [p.name for p in reader.calls] == ["photo1.jpg"]


def test_run_reads_reads_near_duplicates_that_sit_in_different_candidates(tmp_path: Path) -> None:
    """A read covers only photographs of its own candidate, at read time as at launch."""
    mine = photographs(tmp_path, [phash_apart(0)])
    other = photographs(tmp_path, [phash_apart(2)], tag="other", shade=60)
    write_stage1(tmp_path / "work_two", mine.cands + other.cands)
    reader = ScriptedReader()
    counts = run(tmp_path / "work_two", reader)
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
    assert captured["skip_near_duplicates"] == reads.NEAR_DUPLICATE_BITS == 6        # on by default
    assert captured["stop_file"] == stage1.work / "reads" / "STOP"
    assert isinstance(captured["read_fn"], reads.functools.partial) and captured["read_fn"].func is reads.vlm_read
    assert captured["read_fn"].keywords == {"model_key": "model-x"}

    stop = stage1.work / "custom.stop"
    reads.main(["--work-dir", str(stage1.work), "--reader", "kraken", "--model", "/m/other.mlmodel", "--stop-file", str(stop)])
    assert captured["stop_file"] == stop and captured["model"] == "/m/other.mlmodel"
    assert captured["read_fn"].keywords["model_path"] == "/m/other.mlmodel" and captured["limit"] is None


@pytest.mark.parametrize("flag, expected", [([], 6), (["--skip-near-duplicates", "0"], 0),
                                            (["--skip-near-duplicates", "3"], 3)])
def test_main_passes_the_near_duplicate_distance(stage1: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
                                                 flag: List[str], expected: int) -> None:
    """``--skip-near-duplicates N`` reaches ``run_reads`` as ``skip_near_duplicates``; default 6, ``0`` is off."""
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
    reads.main(["--work-dir", str(work), "--reader", "kraken"])
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


@pytest.mark.parametrize("argv", [
    [],
    ["--reader", "vlm"],
    ["--gemini-index", "index.jsonl", "--reader", "kraken"],
    ["--reader", "claude"],
    ["--reader", "kraken", "--script", "latin"],
    ["--reader", "kraken", "--skip-near-duplicates", "-1"],
    ["--reader", "kraken", "--skip-near-duplicates", "many"],
])
def test_main_rejects_bad_arguments(tmp_path: Path, argv: List[str]) -> None:
    """Missing reader, vlm without a model, Gemini index without import, unknown choices, bad bit distance: exit 2."""
    with pytest.raises(SystemExit) as raised:
        reads.main(["--work-dir", str(tmp_path / "work")] + argv)
    assert raised.value.code == 2
