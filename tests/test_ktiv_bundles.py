"""Unit tests for duplicate-aware KTIV bundle selection (ktiv_bundles)."""

import json
import os
from pathlib import Path

import pytest

from src.finetuning.qwen_hebrew.ktiv_bundles import (
    bundle_shape,
    bundle_richness,
    pick_best_bundle,
    select_bundles,
)

SYS = "990099999990205171"


def _api_doc(n_items: int = 5) -> dict:
    """API-shape bundle with ``n_items`` word annotations on one page."""
    items = [{"body": {"value": "אבג"}, "target": {}} for _ in range(n_items)]
    return {"source": "nli_ktiv_viewer", "doc_id": f"PNX_MANUSCRIPTS{SYS}-1",
            "pages": [{"fl": "FL1", "annotation_page": {"items": items}}]}


def _dom_doc(n_chars: int = 500) -> dict:
    """DOM-shape bundle with ``n_chars`` of flat text on one page."""
    return {"source": "nli_ktiv_viewer_dom", "doc_id": f"PNX_MANUSCRIPTS{SYS}-1",
            "pages": [{"fl": "FL1", "text": "א" * n_chars}]}


def _write(tmp_path: Path, name: str, doc: dict, mtime: float) -> Path:
    """Write a bundle file and pin its mtime."""
    p = tmp_path / name
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    os.utime(p, (mtime, mtime))
    return p


def test_shape_classification():
    """source field decides; annotation items are the fallback."""
    assert bundle_shape(_api_doc()) == "api"
    assert bundle_shape(_dom_doc()) == "dom"
    unlabeled = {**_api_doc(), "source": None}
    assert bundle_shape(unlabeled) == "api"


def test_newer_api_rescrape_beats_older_dom(tmp_path):
    """The re-scrape case: (1)-suffixed API bundle supersedes the DOM base."""
    _write(tmp_path, f"ktiv_PNX_MANUSCRIPTS{SYS}-1_transcription.json",
           _dom_doc(5000), mtime=1_000)
    new = _write(tmp_path, f"ktiv_PNX_MANUSCRIPTS{SYS}-1_transcription(1).json",
                 _api_doc(3), mtime=2_000)
    path, doc = pick_best_bundle(tmp_path.glob("*_transcription*.json"))
    assert path == new and bundle_shape(doc) == "api"


def test_api_beats_even_newer_dom(tmp_path):
    """Shape outranks recency: an old word-box bundle beats a fresh flat one."""
    old_api = _write(tmp_path, f"ktiv_PNX_MANUSCRIPTS{SYS}-1_transcription.json",
                     _api_doc(3), mtime=1_000)
    _write(tmp_path, f"ktiv_PNX_MANUSCRIPTS{SYS}-1_transcription(1).json",
           _dom_doc(5000), mtime=2_000)
    path, _ = pick_best_bundle(tmp_path.glob("*_transcription*.json"))
    assert path == old_api


def test_newer_api_beats_older_richer_api(tmp_path):
    """Recency outranks richness within a shape (a re-scrape supersedes)."""
    _write(tmp_path, f"ktiv_PNX_MANUSCRIPTS{SYS}-1_transcription.json",
           _api_doc(100), mtime=1_000)
    new = _write(tmp_path, f"ktiv_PNX_MANUSCRIPTS{SYS}-1_transcription(1).json",
                 _api_doc(10), mtime=2_000)
    path, _ = pick_best_bundle(tmp_path.glob("*_transcription*.json"))
    assert path == new


def test_richness_breaks_equal_mtime(tmp_path):
    """Identical vintage: the richer bundle wins."""
    _write(tmp_path, f"ktiv_PNX_MANUSCRIPTS{SYS}-1_transcription.json",
           _api_doc(2), mtime=1_000)
    rich = _write(tmp_path, f"ktiv_PNX_MANUSCRIPTS{SYS}-1_transcription(1).json",
                  _api_doc(50), mtime=1_000)
    path, _ = pick_best_bundle(tmp_path.glob("*_transcription*.json"))
    assert path == rich
    assert bundle_richness(_api_doc(50)) > bundle_richness(_api_doc(2))


def test_select_bundles_sees_suffixed_duplicates(tmp_path):
    """Regression: the (1)-suffix files a plain glob misses must be found."""
    sys2 = "990088888880205171"
    _write(tmp_path, f"ktiv_PNX_MANUSCRIPTS{SYS}-1_transcription(1).json",
           _api_doc(3), mtime=2_000)   # ONLY a suffixed file for this ms
    _write(tmp_path, f"ktiv_PNX_MANUSCRIPTS{sys2}-1_transcription.json",
           _dom_doc(100), mtime=1_000)
    picked = select_bundles(tmp_path)
    assert set(picked) == {SYS, sys2}
    assert bundle_shape(picked[SYS][1]) == "api"
    assert bundle_shape(picked[sys2][1]) == "dom"


def test_unparseable_and_empty(tmp_path):
    """Corrupt files are skipped; nothing parseable yields None."""
    bad = tmp_path / f"ktiv_PNX_MANUSCRIPTS{SYS}-1_transcription.json"
    bad.write_text("{not json", encoding="utf-8")
    assert pick_best_bundle([bad]) is None
    assert pick_best_bundle([]) is None
    good = _write(tmp_path, f"ktiv_PNX_MANUSCRIPTS{SYS}-1_transcription(1).json",
                  _api_doc(1), mtime=1_000)
    path, _ = pick_best_bundle(tmp_path.glob("*_transcription*.json"))
    assert path == good


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
