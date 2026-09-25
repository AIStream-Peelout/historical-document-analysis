"""Tests for the Kraken-only re-run of the two-reader line pipeline (``--rekraken``).

Kraken and downloads are mocked (an autouse fixture fails any call a test did
not mock), so nothing here touches the Kraken service, LM Studio or ES; the CLI
test runs the module on an empty scratch set only.
"""
import argparse
import asyncio
import hashlib
import io
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
from PIL import Image

from src.datasets.consensus import two_reader_lines as trl

NEW = "MiDRASH_Gen_02"
NEW_MODEL = "/models/MiDRASH_Gen_02.mlmodel"
W, H = 100, 200
DOC = "Cambridge_T-S_1"
VLM_LINES = [{"text": "אבגדה הוזחט", "box": [100.0, 100.0, 900.0, 200.0]},
             {"text": "יכלמנ סעפצק", "box": [100.0, 300.0, 900.0, 400.0]}]
OLD_FRAGS = [{"text": "שששש", "conf": 0.5, "box": [150.0, 110.0, 850.0, 190.0]}]   # disagrees with line 0
KRAKEN_LINES = [{"text": "אבגדה הוזחט", "bbox": [10, 20, 90, 40], "confidence": 0.9},
                {"text": "  ", "bbox": [0, 0, 1, 1], "confidence": 0.1},          # blank: dropped
                {"text": "יכלמנ סעפצק", "bbox": [10, 60, 90, 80], "confidence": 0.8}]
NEW_FRAGS = [{"text": "אבגדה הוזחט", "conf": 0.9, "box": [100.0, 100.0, 900.0, 200.0]},
             {"text": "יכלמנ סעפצק", "conf": 0.8, "box": [100.0, 300.0, 900.0, 400.0]}]
ENVELOPE = ("doc_id", "source_index", "image_index", "image_url", "image_width", "image_height", "image_sha256")
V20A = "qwen3-vl-8b-heb-v20a-step1800"


def _jpeg(color: Tuple[int, int, int] = (255, 255, 255)) -> bytes:
    """A ``W`` x ``H`` single-colour JPEG (different colours give different bytes).

    :param color: RGB fill.
    :returns: JPEG bytes.
    """
    buf = io.BytesIO()
    Image.new("RGB", (W, H), color).save(buf, format="JPEG")
    return buf.getvalue()


class FakeKraken:
    """Stand-in for ``transcribe_with_kraken_lines`` that records its calls."""

    def __init__(self, lines: Optional[List[Dict[str, Any]]] = KRAKEN_LINES) -> None:
        """:param lines: Lines to answer with; None answers like a failed service call."""
        self.lines = lines
        self.calls: List[Tuple[str, str]] = []

    async def __call__(self, model_path: str, image_path: str,
                       timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Record ``(model_path, image file name)``; the image must exist while it is read.

        :param model_path: Kraken model sent to the service.
        :param image_path: Image sent to the service.
        :param timeout: Request timeout (ignored).
        :returns: A ``/transcribe_lines`` response, or None.
        """
        assert Path(image_path).exists()
        self.calls.append((model_path, Path(image_path).name))
        return None if self.lines is None else {"lines": self.lines, "polygon_failures": 0, "image_size": [W, H]}


@pytest.fixture(autouse=True)
def no_services(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any Kraken call or download a test did not mock.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    async def kraken(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("unexpected Kraken call")

    def download(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("unexpected download")

    monkeypatch.setattr(trl, "transcribe_with_kraken_lines", kraken)
    monkeypatch.setattr(trl, "download", download)


def _make_page(tmp_path: Path, doc_id: str = DOC, vlm_model: str = trl.VLM_MODEL, data: Optional[bytes] = None,
               with_record: bool = True) -> Dict[str, Any]:
    """Write one already-read page: a raw-cache entry with only legacy ``frags`` (+ its record in ``out``).

    :param tmp_path: Test directory (``raw/``, ``images/`` and ``ai_reads.jsonl`` live there).
    :param doc_id: Document id.
    :param vlm_model: VLM checkpoint of the entry.
    :param data: Image bytes (default a white JPEG).
    :param with_record: Also append the page's record to ``out``.
    :returns: ``raw``, ``work``, ``out``, ``cache`` paths, image ``data`` and the ``record``.
    """
    raw, work = tmp_path / "raw", tmp_path / "images"
    raw.mkdir(exist_ok=True)
    work.mkdir(exist_ok=True)
    data = data or _jpeg()
    sha = hashlib.sha256(data).hexdigest()
    job = {"doc_id": doc_id, "image_index": 0, "image_url": f"https://example.org/{doc_id}.jpg"}
    cache = raw / f"{trl.raw_stem(job, vlm_model)}.json"
    cache.write_text(json.dumps(dict(**job, width=W, height=H, sha256=sha, vlm_model=vlm_model, vlm_raw="[...]",
                                     vlm_lines=VLM_LINES, parsed=True, frags=OLD_FRAGS), ensure_ascii=False))
    out = tmp_path / "ai_reads.jsonl"
    record = {"doc_id": doc_id, "source_index": "genizah_merged_v6", "image_index": 0, "image_url": job["image_url"],
              "image_width": W, "image_height": H, "image_sha256": sha,
              "ai_read": trl.sidecar(VLM_LINES, OLD_FRAGS, True, vlm_model, "rev-old")}
    if with_record:
        with out.open("a") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"raw": raw, "work": work, "out": out, "cache": cache, "data": data, "record": record}


@pytest.fixture
def page(tmp_path: Path) -> Dict[str, Any]:
    """One already-read page (see :func:`_make_page`).

    :param tmp_path: Pytest temp dir.
    :returns: The page.
    """
    return _make_page(tmp_path)


def _rekraken(pg: Dict[str, Any], cache_key: str = NEW, **kwargs: Any) -> "Counter[str]":
    """Run :func:`trl.rekraken` with the new model over a test page's ``out`` and cache.

    :param pg: Page from :func:`_make_page`.
    :param cache_key: ``frags_by_htr`` key.
    :param kwargs: Extra keyword arguments for ``rekraken``.
    :returns: Its counts.
    """
    kwargs.setdefault("min_free_gb", 0.0)
    return asyncio.run(trl.rekraken(pg["out"], pg["work"], pg["raw"], NEW_MODEL, NEW, cache_key, **kwargs))


def _serve(monkeypatch: pytest.MonkeyPatch, by_url: Dict[str, bytes]) -> List[str]:
    """Mock ``download`` with fixed bytes per URL.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param by_url: Bytes to serve per URL.
    :returns: The list the downloaded URLs are appended to.
    """
    fetched: List[str] = []

    def download(url: str) -> bytes:
        fetched.append(url)
        return by_url[url]

    monkeypatch.setattr(trl, "download", download)
    return fetched


def _strip(ai_read: Dict[str, Any]) -> Dict[str, Any]:
    """``ai_read`` without its ``decoded_at`` stamp.

    :param ai_read: Sidecar block.
    :returns: Copy without the time stamp.
    """
    return {k: v for k, v in ai_read.items() if k != "decoded_at"}


def test_legacy_frags_read_as_default_model(page: Dict[str, Any]) -> None:
    """(a) An entry with only ``frags`` answers for the default model and for no other."""
    legacy = {"frags": OLD_FRAGS}
    assert trl.cached_frags(legacy, trl.HTR_MODEL_NAME) == OLD_FRAGS
    assert trl.cached_frags(legacy, NEW) is None
    both = {"frags": OLD_FRAGS, "frags_by_htr": {NEW: NEW_FRAGS}}
    assert trl.cached_frags(both, NEW) == NEW_FRAGS
    assert trl.cached_frags(both, trl.HTR_MODEL_NAME) == OLD_FRAGS
    # a --force re-read with the default model is kept under its name and wins over the legacy list
    forced = {"frags": OLD_FRAGS, "frags_by_htr": {trl.HTR_MODEL_NAME: NEW_FRAGS}}
    assert trl.cached_frags(forced, trl.HTR_MODEL_NAME) == NEW_FRAGS
    # an entry that names its reader answers for that reader only
    stamped = {"frags": OLD_FRAGS, "htr_model": NEW}
    assert trl.cached_frags(stamped, NEW) == OLD_FRAGS
    assert trl.cached_frags(stamped, trl.HTR_MODEL_NAME) is None
    # a real legacy entry rebuilt for the default model reproduces the original record
    entry = json.loads(page["cache"].read_text())
    rebuilt = trl.rebuild_record(page["record"], entry, trl.HTR_MODEL_NAME, trl.HTR_MODEL_NAME)
    assert _strip(rebuilt["ai_read"]) == _strip(page["record"]["ai_read"])
    assert trl.rebuild_record(page["record"], entry, NEW, NEW) is None


def test_rekraken_stores_new_key_and_leaves_frags(page: Dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """(b) New fragments land in ``frags_by_htr[key]``; ``frags`` and the VLM read are untouched."""
    kraken = FakeKraken()
    monkeypatch.setattr(trl, "transcribe_with_kraken_lines", kraken)
    _serve(monkeypatch, {page["record"]["image_url"]: page["data"]})
    before = json.loads(page["cache"].read_text())
    counts = _rekraken(page)
    after = json.loads(page["cache"].read_text())
    assert after["frags_by_htr"] == {NEW: NEW_FRAGS}
    assert {k: v for k, v in after.items() if k != "frags_by_htr"} == before
    assert kraken.calls == [(NEW_MODEL, f"{DOC}__0.jpg")]
    assert (counts["krakened"], counts["entries"]) == (1, 1)
    assert list(page["work"].iterdir()) == []                 # the download is not kept
    assert list(page["raw"].iterdir()) == [page["cache"]]     # no temp file left behind


def test_rebuilt_record_uses_new_frags_and_stamps_htr(page: Dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """(c) The rebuilt record matches against the new fragments and names the new model; envelope/VLM kept."""
    monkeypatch.setattr(trl, "transcribe_with_kraken_lines", FakeKraken())
    _serve(monkeypatch, {page["record"]["image_url"]: page["data"]})
    counts = _rekraken(page)
    old, new = page["record"], json.loads(page["out"].read_text())
    ar = new["ai_read"]
    assert ar["htr_model"] == NEW and "htr_cache_key" not in ar
    assert [ln["htr_text"] for ln in ar["lines"]] == [f["text"] for f in NEW_FRAGS]
    assert (old["ai_read"]["n_agreed"], ar["n_agreed"]) == (0, 2)
    assert (ar["vlm_model"], ar["vlm_revision"], ar["rule_version"]) == (trl.VLM_MODEL, "rev-old", trl.RULE_VERSION)
    assert [ln["text"] for ln in ar["lines"]] == [ln["text"] for ln in VLM_LINES]
    assert {k: new[k] for k in ENVELOPE} == {k: old[k] for k in ENVELOPE}
    assert (counts["rebuilt"], counts["kept"]) == (1, 0)


def test_skip_resume_and_force(page: Dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """(d) Pages holding the key are not re-read (but their records are rebuilt); ``force`` re-reads."""
    kraken = FakeKraken()
    monkeypatch.setattr(trl, "transcribe_with_kraken_lines", kraken)
    fetched = _serve(monkeypatch, {page["record"]["image_url"]: page["data"]})
    _rekraken(page)
    assert len(kraken.calls) == len(fetched) == 1
    # an interrupted pass: fragments cached but --out never rewritten -> the next pass reads nothing, rebuilds
    page["out"].write_text(json.dumps(page["record"], ensure_ascii=False) + "\n")
    counts = _rekraken(page)
    assert len(kraken.calls) == len(fetched) == 1
    assert (counts["skipped"], counts["krakened"], counts["rebuilt"]) == (1, 0, 1)
    assert json.loads(page["out"].read_text())["ai_read"]["htr_model"] == NEW
    counts = _rekraken(page, force=True)
    assert len(kraken.calls) == 2 and counts["krakened"] == 1
    # the legacy frags count as the default model's: nothing to read for it without --force
    counts = asyncio.run(trl.rekraken(page["out"], page["work"], page["raw"], trl.KRAKEN_MODEL, trl.HTR_MODEL_NAME,
                                      trl.HTR_MODEL_NAME, min_free_gb=0.0))
    assert counts["skipped"] == 1 and len(kraken.calls) == 2
    ar = json.loads(page["out"].read_text())["ai_read"]
    assert (ar["htr_model"], ar["n_agreed"]) == (trl.HTR_MODEL_NAME, 0)


def test_sha256_mismatch_skips_page(page: Dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """(e) A served image whose bytes changed is not re-read: cache and record stay as they were."""
    kraken = FakeKraken()
    monkeypatch.setattr(trl, "transcribe_with_kraken_lines", kraken)
    _serve(monkeypatch, {page["record"]["image_url"]: _jpeg((0, 0, 0))})
    out_before, cache_before = page["out"].read_text(), page["cache"].read_text()
    counts = _rekraken(page)
    assert (counts["sha256"], counts["krakened"], counts["kept"]) == (1, 0, 1)
    assert kraken.calls == []
    assert page["cache"].read_text() == cache_before
    assert page["out"].read_text() == out_before
    assert list(page["work"].iterdir()) == []


def test_kept_image_used_without_download(page: Dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``--work-dir`` image whose bytes hash to the cached sha256 is read in place and left there."""
    kraken = FakeKraken()
    monkeypatch.setattr(trl, "transcribe_with_kraken_lines", kraken)   # download stays the failing stub
    kept = page["work"] / f"{DOC}__0.jpg"
    kept.write_bytes(page["data"])
    counts = _rekraken(page)
    assert counts["krakened"] == 1 and kraken.calls == [(NEW_MODEL, kept.name)]
    assert kept.exists()


def test_all_cache_reads_each_image_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``all_cache``: one read per image hash for every VLM's entry; later siblings are filled; ``limit`` holds."""
    a21 = _make_page(tmp_path)                                               # record in out
    a20 = _make_page(tmp_path, vlm_model=V20A, with_record=False)           # same image, other VLM
    b20 = _make_page(tmp_path, doc_id="Cambridge_T-S_2", vlm_model=V20A, data=_jpeg((90, 90, 90)), with_record=False)
    kraken = FakeKraken()
    monkeypatch.setattr(trl, "transcribe_with_kraken_lines", kraken)
    _serve(monkeypatch, {a21["record"]["image_url"]: a21["data"], b20["record"]["image_url"]: b20["data"]})
    counts = _rekraken(a21, all_cache=True, limit=1)
    assert len(kraken.calls) == 1 and counts["entries"] == 2
    assert all(json.loads(pg["cache"].read_text())["frags_by_htr"] == {NEW: NEW_FRAGS} for pg in (a21, a20))
    assert "frags_by_htr" not in json.loads(b20["cache"].read_text())       # beyond the limit
    counts = _rekraken(a21, all_cache=True)
    assert len(kraken.calls) == 2 and (counts["skipped"], counts["krakened"]) == (1, 1)
    a06 = _make_page(tmp_path, vlm_model="qwen3-vl-8b-heb-v21b-step600", with_record=False)
    counts = _rekraken(a21, all_cache=True)
    assert len(kraken.calls) == 2 and counts["filled"] == 1
    assert json.loads(a06["cache"].read_text())["frags_by_htr"] == {NEW: NEW_FRAGS}
    assert json.loads(a21["out"].read_text())["ai_read"]["htr_model"] == NEW


def test_kraken_failures_keep_records_and_stop_the_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Failed reads leave cache and records alone; a run of Kraken failures stops the pass."""
    pages = [_make_page(tmp_path, doc_id=f"Cambridge_T-S_{i}", data=_jpeg((80 * i,) * 3)) for i in range(3)]
    kraken = FakeKraken(lines=None)
    monkeypatch.setattr(trl, "transcribe_with_kraken_lines", kraken)
    _serve(monkeypatch, {pg["record"]["image_url"]: pg["data"] for pg in pages})
    monkeypatch.setattr(trl, "REKRAKEN_MAX_KRAKEN_FAILURES", 2)
    before = pages[0]["out"].read_text()
    counts = _rekraken(pages[0])
    assert len(kraken.calls) == 2 and (counts["kraken"], counts["kept"]) == (2, 3)
    assert pages[0]["out"].read_text() == before
    assert all("frags_by_htr" not in json.loads(pg["cache"].read_text()) for pg in pages)
    assert list(pages[0]["work"].iterdir()) == []


def test_rematch_keeps_rekrakened_fragments(page: Dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """A later ``--rematch`` re-matches each record against the fragments it was built from."""
    monkeypatch.setattr(trl, "transcribe_with_kraken_lines", FakeKraken())
    _serve(monkeypatch, {page["record"]["image_url"]: page["data"]})
    key = NEW + "@seg2"                                    # a cache key other than the model name is recorded
    _rekraken(page, cache_key=key)
    ar = json.loads(page["out"].read_text())["ai_read"]
    assert (ar["htr_model"], ar["htr_cache_key"], ar["n_agreed"]) == (NEW, key, 2)
    monkeypatch.setattr(trl, "RAW_DIR", page["raw"])
    asyncio.run(trl.rematch(page["out"], page["work"], "genizah_merged_v6", "rev-old"))
    ar = json.loads(page["out"].read_text())["ai_read"]
    assert (ar["htr_model"], ar["htr_cache_key"], ar["n_agreed"]) == (NEW, key, 2)
    page["out"].write_text(json.dumps(page["record"], ensure_ascii=False) + "\n")   # a legacy record
    asyncio.run(trl.rematch(page["out"], page["work"], "genizah_merged_v6", "rev-old"))
    ar = json.loads(page["out"].read_text())["ai_read"]
    assert (ar["htr_model"], ar["n_agreed"]) == (trl.HTR_MODEL_NAME, 0) and "htr_cache_key" not in ar


def test_stage1_reads_with_the_default_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``stage1`` (main loop) still reads with ``KRAKEN_MODEL`` and a 300 s timeout through :func:`run_kraken`."""
    calls: List[Tuple[str, Optional[float]]] = []

    async def kraken(model_path: str, image_path: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        calls.append((model_path, timeout))
        return {"lines": KRAKEN_LINES}

    monkeypatch.setattr(trl, "transcribe_with_kraken_lines", kraken)
    data = _jpeg()
    _serve(monkeypatch, {"https://example.org/x.jpg": data})
    info = asyncio.run(trl.stage1({"doc_id": DOC, "image_index": 0, "image_url": "https://example.org/x.jpg"},
                                  tmp_path))
    assert calls == [(trl.KRAKEN_MODEL, 300.0)]
    assert info["frags"] == NEW_FRAGS and (info["width"], info["height"]) == (W, H)
    assert info["sha256"] == hashlib.sha256(data).hexdigest() and "failure" not in info


def test_rekraken_names() -> None:
    """Defaults: the model file's stem names a new model; the cache key follows the name."""
    assert trl.rekraken_names(None, None, None) == (trl.KRAKEN_MODEL, trl.HTR_MODEL_NAME, trl.HTR_MODEL_NAME)
    assert trl.rekraken_names(NEW_MODEL, None, None) == (NEW_MODEL, NEW, NEW)
    assert trl.rekraken_names(NEW_MODEL, "Gen2", "Gen2@seg") == (NEW_MODEL, "Gen2", "Gen2@seg")


def test_replace_records_carries_appends_and_refuses_shrink(tmp_path: Path) -> None:
    """Records appended during a pass are carried over; a half-written one is not; a shrunk file is left alone."""
    out = tmp_path / "out.jsonl"
    out.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n{"a": 4')   # 2 read; then 1 appended, 1 still being written
    assert trl.replace_records(out, ['{"b": 1}', '{"b": 2}'], 2) == 1
    assert out.read_text() == '{"b": 1}\n{"b": 2}\n{"a": 3}\n'
    with pytest.raises(RuntimeError):
        trl.replace_records(out, ["{}"] * 4, 4)
    assert out.read_text() == '{"b": 1}\n{"b": 2}\n{"a": 3}\n'


SERVICE_KEY = "MiDRASH_Gen_01@k7.0.3-blla2026"


def _cli_rekraken(pg: Dict[str, Any], **kw: Any) -> argparse.Namespace:
    """The CLI's arguments for a ``--rekraken`` run over a test page (argparse's defaults; ``kw`` overrides).

    :param pg: Page from :func:`_make_page`.
    :param kw: Overrides (``legacy_key``, ``kraken_cache_suffix``, ``kraken_model``, ``rekraken`` ...).
    :returns: The arguments.
    """
    return argparse.Namespace(**{
        "rekraken": True, "out": str(pg["out"]), "work_dir": str(pg["work"]), "raw_dir": str(pg["raw"]),
        "kraken_model": None, "htr_model_name": None, "kraken_cache_suffix": None, "legacy_key": False,
        "all_cache": False, "force": False, "limit": 0, "min_free_gb": 0.0, "keep_images": False, **kw})


def test_bare_rekraken_needs_legacy_key(page: Dict[str, Any]) -> None:
    """A ``--rekraken`` key must be explicit: a suffix, or the bare model name on purpose with ``--legacy-key``."""
    msg = trl.rekraken_key_error(_cli_rekraken(page))
    assert msg.startswith("--rekraken needs --kraken-cache-suffix KEY naming the Kraken service")
    assert f"the bare key {trl.HTR_MODEL_NAME} names none" in msg and msg.endswith("pass --legacy-key to use the "
                                                                                   "bare key on purpose")
    assert f"the bare key {NEW} names none" in trl.rekraken_key_error(_cli_rekraken(page, kraken_model=NEW_MODEL))
    assert trl.rekraken_key_error(_cli_rekraken(page, legacy_key=True)) is None
    assert trl.rekraken_key_error(_cli_rekraken(page, kraken_cache_suffix=SERVICE_KEY)) is None
    assert trl.rekraken_key_error(_cli_rekraken(page, legacy_key=True, kraken_cache_suffix=SERVICE_KEY)) == (
        "--legacy-key and --kraken-cache-suffix are exclusive: the bare key, or KEY")
    assert trl.rekraken_key_error(_cli_rekraken(page, rekraken=False)) is None      # normal runs: not concerned


def test_legacy_key_is_the_bare_key_pass(page: Dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """``--legacy-key`` runs the bare-key pass unchanged: legacy pages hold the key, service-stamped ones are relabelled."""
    kraken = FakeKraken()
    monkeypatch.setattr(trl, "transcribe_with_kraken_lines", kraken)
    _serve(monkeypatch, {page["record"]["image_url"]: page["data"]})
    args = _cli_rekraken(page, legacy_key=True)
    assert trl.rekraken_key_error(args) is None
    cache_before = page["cache"].read_text()
    asyncio.run(trl.main_async(args))                  # the legacy frags answer for the bare key: nothing to read
    assert kraken.calls == [] and page["cache"].read_text() == cache_before
    assert _strip(json.loads(page["out"].read_text())["ai_read"]) == _strip(page["record"]["ai_read"])
    # a page stamped with a service key (--htr-cache-key, stamp_htr_cache_key.py) is re-read and relabelled
    entry = json.loads(cache_before)
    entry["htr_model"] = SERVICE_KEY
    page["cache"].write_text(json.dumps(entry, ensure_ascii=False))
    record = json.loads(page["out"].read_text())
    record["ai_read"]["htr_cache_key"] = SERVICE_KEY
    page["out"].write_text(json.dumps(record, ensure_ascii=False) + "\n")
    asyncio.run(trl.main_async(args))
    assert kraken.calls == [(trl.KRAKEN_MODEL, f"{DOC}__0.jpg")]
    assert json.loads(page["cache"].read_text())["frags_by_htr"] == {trl.HTR_MODEL_NAME: NEW_FRAGS}
    ar = json.loads(page["out"].read_text())["ai_read"]
    assert (ar["htr_model"], ar["n_agreed"]) == (trl.HTR_MODEL_NAME, 2) and "htr_cache_key" not in ar


def test_suffix_rekraken_is_unchanged(page: Dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """``--kraken-cache-suffix KEY`` runs as before: re-read under KEY, record stamped with it, ``frags`` kept."""
    kraken = FakeKraken()
    monkeypatch.setattr(trl, "transcribe_with_kraken_lines", kraken)
    _serve(monkeypatch, {page["record"]["image_url"]: page["data"]})
    args = _cli_rekraken(page, kraken_cache_suffix=SERVICE_KEY)
    assert trl.rekraken_key_error(args) is None
    asyncio.run(trl.main_async(args))
    assert kraken.calls == [(trl.KRAKEN_MODEL, f"{DOC}__0.jpg")]
    entry = json.loads(page["cache"].read_text())
    assert (entry["frags_by_htr"], entry["frags"]) == ({SERVICE_KEY: NEW_FRAGS}, OLD_FRAGS)
    ar = json.loads(page["out"].read_text())["ai_read"]
    assert (ar["htr_model"], ar["htr_cache_key"], ar["n_agreed"]) == (trl.HTR_MODEL_NAME, SERVICE_KEY, 2)


def test_cli_refuses_a_bare_rekraken(tmp_path: Path) -> None:
    """Through the CLI: a bare ``--rekraken`` is an argument error; ``--legacy-key`` or a suffix runs (empty set)."""
    repo = Path(trl.__file__).resolve().parents[3]
    out, raw = tmp_path / "out.jsonl", tmp_path / "raw"
    out.write_text("")
    raw.mkdir()
    scratch = ["--out", str(out), "--raw-dir", str(raw), "--work-dir", str(tmp_path / "images")]

    def cli(*argv: str) -> "subprocess.CompletedProcess[str]":
        """Run the module's CLI from the repo root.

        :param argv: Arguments.
        :returns: The finished process.
        """
        return subprocess.run([sys.executable, "-m", "src.datasets.consensus.two_reader_lines", *argv], cwd=repo,
                              capture_output=True, text=True, timeout=120)

    bare = cli("--rekraken", *scratch)
    assert bare.returncode == 2 and "error: --rekraken needs --kraken-cache-suffix KEY" in bare.stderr
    assert not (tmp_path / "images").exists()                          # refused before anything ran
    both = cli("--rekraken", "--legacy-key", "--kraken-cache-suffix", SERVICE_KEY, *scratch)
    assert both.returncode == 2 and "--legacy-key and --kraken-cache-suffix are exclusive" in both.stderr
    stray = cli("--ids", str(out), "--legacy-key", "--out", str(tmp_path / "normal.jsonl"), "--work-dir",
                str(tmp_path / "images"), "--source-index", "genizah_merged_v6", "--site-env", str(tmp_path / "x.env"))
    assert stray.returncode == 2 and "error: --legacy-key: --rekraken options only" in stray.stderr
    legacy = cli("--rekraken", "--legacy-key", *scratch)
    assert legacy.returncode == 0, legacy.stderr
    assert f"rekraken {trl.HTR_MODEL_NAME} [key {trl.HTR_MODEL_NAME}]: 0 pages re-krakened" in legacy.stdout
    keyed = cli("--rekraken", "--kraken-cache-suffix", SERVICE_KEY, *scratch)
    assert keyed.returncode == 0, keyed.stderr
    assert f"rekraken {trl.HTR_MODEL_NAME} [key {SERVICE_KEY}]: 0 pages re-krakened" in keyed.stdout
