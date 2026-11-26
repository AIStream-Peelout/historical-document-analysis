"""Tests for the production-reader benchmark runner (no model: the reader is a fake)."""
import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.datasets.evaluations.helper_eval_scripts import run_reader_benchmark as rrb
from src.models.ocr import page_reader as pr


def _spec(tmp_path: Path, docs: List[Dict[str, Any]]) -> Path:
    """Write a benchmark spec.

    :param tmp_path: Test folder.
    :param docs: The spec's docs.
    :return: The spec path.
    """
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"docs": docs}), encoding="utf-8")
    return path


def _image(path: Path) -> Path:
    """Create an image placeholder file.

    :param path: File to create.
    :return: The path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"jpg")
    return path


def test_output_key_matches_the_legacy_naming_plus_suffix() -> None:
    """The key is the legacy runners' key (``-`` to ``_``) with the production suffix."""
    assert rrb.output_key("qwen3-vl-8b-heb-v25a-step1200") == "qwen3_vl_8b_heb_v25a_step1200_prod"
    assert rrb.output_key("m-1", "_x") == "m_1_x"


def test_load_pages_reads_both_spec_formats(tmp_path: Path) -> None:
    """PGP-131 docs take ``<images_dir>/<doc_id>.jpg``; religious docs keep their own image."""
    spec = _spec(tmp_path, [{"doc_id": "A"}, {"doc_id": "B", "image": "/abs/b.jpg"}])
    assert rrb.load_pages(spec, tmp_path / "img") == [("A", tmp_path / "img" / "A.jpg"), ("B", Path("/abs/b.jpg"))]
    with pytest.raises(ValueError):
        rrb.load_pages(spec, None)


def test_pending_skips_read_pages_and_reports_missing_images(tmp_path: Path) -> None:
    """A stored text marks a page done; a page without its image file is reported, not read."""
    out = tmp_path / "out"
    (out / "A").mkdir(parents=True)
    (out / "A" / "k.txt").write_text("x", encoding="utf-8")
    pages = [("A", _image(tmp_path / "A.jpg")), ("B", _image(tmp_path / "B.jpg")), ("C", tmp_path / "C.jpg")]
    todo, done, missing = rrb.pending(pages, out, "k")
    assert todo == [("B", tmp_path / "B.jpg")] and done == 1 and missing == ["C"]


def test_store_writes_text_and_fields_but_not_a_failed_read(tmp_path: Path) -> None:
    """A read stores its text and the reader's other fields; a failed request stores nothing."""
    result = {"text": "שלום", "raw_text": "שלום", "finish_reason": "stop", "attempts": 1, "looped": False}
    assert rrb.store(tmp_path, "A", "k", result)
    assert (tmp_path / "A" / "k.txt").read_text(encoding="utf-8") == "שלום"
    side = json.loads((tmp_path / "A" / "k.read.json").read_text(encoding="utf-8"))
    assert side["doc_id"] == "A" and side["finish_reason"] == "stop" and "text" not in side
    assert not rrb.store(tmp_path, "B", "k", {"text": None, "reason": pr.REASON_FAILED})
    assert not (tmp_path / "B").exists()


def test_store_keeps_a_blank_gated_page_as_an_empty_read(tmp_path: Path) -> None:
    """The ink gate's empty answer is a stored read (production abstains on it), not a failure."""
    assert rrb.store(tmp_path, "A", "k", {"text": "", "reason": pr.REASON_BLANK, "blank": True})
    assert (tmp_path / "A" / "k.txt").read_text(encoding="utf-8") == ""


def test_run_reads_with_production_settings_and_counts_reader_behaviour(tmp_path: Path) -> None:
    """Every page is read with the given settings; blank, length, retried and looped reads are counted."""
    seen: List[Any] = []
    results = {"A": {"text": "א", "finish_reason": "stop"},
               "B": {"text": "", "blank": True, "reason": pr.REASON_BLANK},
               "C": {"text": "ב", "finish_reason": "length", "retried": True, "looped": True}}

    async def fake_read(model: str, image: str, settings: pr.ReaderSettings) -> Dict[str, Any]:
        seen.append((model, Path(image).stem, settings))
        return results[Path(image).stem]

    todo = [(d, tmp_path / f"{d}.jpg") for d in "ABC"]
    stats = asyncio.run(rrb.run(todo, "m", tmp_path / "out", "k", read=fake_read, free_check=lambda: 100.0,
                                log=lambda m: None))
    assert [s[1] for s in seen] == ["A", "B", "C"] and all(s[2] == pr.DEFAULT_SETTINGS for s in seen)
    assert stats == {"stored": 3, "failed": 0, "blank": 1, "length": 1, "retried": 1, "looped": 1, "unloads": 0}


def test_run_stops_after_consecutive_failures(tmp_path: Path) -> None:
    """A model that answers nothing stops the run instead of failing every page."""
    async def failing(model: str, image: str, settings: pr.ReaderSettings) -> Dict[str, Any]:
        return {"text": None, "reason": pr.REASON_FAILED, "attempts": 2}

    todo = [(f"D{i}", tmp_path / f"D{i}.jpg") for i in range(rrb.MAX_CONSECUTIVE_FAILURES + 2)]
    with pytest.raises(RuntimeError):
        asyncio.run(rrb.run(todo, "m", tmp_path / "out", "k", read=failing, free_check=lambda: 100.0,
                            log=lambda m: None))


def test_run_waits_at_the_disk_floor_and_unloads_its_own_model(tmp_path: Path) -> None:
    """Under the unload threshold the run unloads its model; under the floor it waits until space returns."""
    free = iter([10.0, 10.0, 5.0, 5.0, 30.0])
    unloaded, slept = [], []

    async def fake_read(model: str, image: str, settings: pr.ReaderSettings) -> Dict[str, Any]:
        return {"text": "א", "finish_reason": "stop"}

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    stats = asyncio.run(rrb.run([("A", tmp_path / "A.jpg")], "m", tmp_path / "out", "k", read=fake_read,
                                min_free_gb=8.0, unload_below_gb=20.0, wait_s=1.0, free_check=lambda: next(free),
                                unload=lambda m: unloaded.append(m) or "ok", sleep=fake_sleep, log=lambda m: None))
    assert unloaded == ["m"] and slept == [1.0] and stats["unloads"] == 1 and stats["stored"] == 1


def test_unload_loads_the_model_again_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    """After the unload the model is loaded again explicitly, so another model's on-demand load cannot evict it."""
    calls: List[List[str]] = []

    def fake_run(cmd: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
        calls.append(cmd[1:])
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(rrb.subprocess, "run", fake_run)
    assert rrb.unload_own_model("m", lms=Path("/x/lms")) == "ok"
    assert calls == [["unload", "m"], ["load", "m", "--identifier", "m", "-c", "8192", "--parallel", "4", "--gpu", "max", "-y"]]
    calls.clear()
    rrb.unload_own_model("m", lms=Path("/x/lms"), reload=False)
    assert calls == [["unload", "m"]]


def test_unload_refuses_production_models() -> None:
    """The production LM Studio models are never unloaded."""
    for model in rrb.PROTECTED_MODELS:
        with pytest.raises(ValueError):
            rrb.unload_own_model(model)
