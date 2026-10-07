"""Unit tests for the Kraken export manifest filter."""
import json
from pathlib import Path

from PIL import Image

from src.finetuning.kraken.filter_manifests import filter_manifest


def _line(root: Path, rel: str, w: int, h: int, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (w, h), 200).save(p, "JPEG")
    p.with_suffix("").with_suffix(".gt.txt").write_text(text, encoding="utf-8")


def test_filter_manifest_drops_wide_sparse_and_tiny_lines(tmp_path: Path) -> None:
    _line(tmp_path, "lines/a/ok.jpg", 1200, 120, "אמר רבי יוחנן משום רבי שמעון בן יוחאי")   # aspect 10, 30 letters
    _line(tmp_path, "lines/a/wide.jpg", 4000, 120, "אמר רבי יוחנן משום רבי שמעון בן יוחאי")  # aspect 33
    _line(tmp_path, "lines/a/sparse.jpg", 2400, 120, "אמר רבי")                                # aspect 20, 6 letters
    _line(tmp_path, "lines/a/tiny.jpg", 300, 10, "אמר רבי יוחנן")                               # 10 px high
    (tmp_path / "train.txt").write_text("lines/a/ok.jpg\nlines/a/wide.jpg\nlines/a/sparse.jpg\nlines/a/tiny.jpg\n", encoding="utf-8")
    counts = filter_manifest(tmp_path, "train.txt", max_aspect=25.0, min_height=20)
    assert dict(counts) == {"kept": 1, "wide": 1, "sparse": 1, "tiny": 1}
    assert (tmp_path / "train.txt").read_text(encoding="utf-8").split() == ["lines/a/ok.jpg"]
    # The unfiltered list survives and a second run is idempotent.
    assert len((tmp_path / "train.all.txt").read_text(encoding="utf-8").split()) == 4
    assert dict(filter_manifest(tmp_path, "train.txt", 25.0, 20)) == {"kept": 1, "wide": 1, "sparse": 1, "tiny": 1}


def test_filter_manifest_prefers_index_records(tmp_path: Path) -> None:
    _line(tmp_path, "lines/a/x.jpg", 1200, 120, "אמר רבי יוחנן")
    (tmp_path / "train.txt").write_text("lines/a/x.jpg\n", encoding="utf-8")
    # The index says the crop is far wider than the file on disk -> the index wins.
    (tmp_path / "index.jsonl").write_text(json.dumps({"path": "lines/a/x.jpg", "w": 9000, "h": 120, "letters": 10}) + "\n")
    assert dict(filter_manifest(tmp_path, "train.txt", 25.0, 20)) == {"wide": 1}
