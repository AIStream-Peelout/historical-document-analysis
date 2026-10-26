# File name: test_gate_lines.py
# Date: 10/19/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Unit tests for the compiled-arrow line gate (src/finetuning/kraken/gate_lines.py).

Everything runs on synthetic arrow files and fake kraken sources under ``tmp_path``; nothing reads the NAS.
"""
import io
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import pyarrow as pa
import pytest
from PIL import Image

from src.finetuning.kraken import gate_lines as gl

LINE_TYPE = pa.struct([("text", pa.string()), ("im", pa.binary()), ("language", pa.list_(pa.string()))])


def _encoded(width: int, height: int, fmt: str = "PNG") -> bytes:
    """Encoded grey test image.

    :param width: Width in pixels.
    :param height: Height in pixels.
    :param fmt: Pillow format name.
    :return: Image bytes.
    """
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (200, 190, 170)).save(buf, format=fmt)
    return buf.getvalue()


def _kraken_arrow(path: Path, batches: Sequence[Sequence[Tuple[str, bytes]]], kind: str = "kraken_recognition_baseline") -> Path:
    """Write an arrow file in the layout ``ketos compile`` (kraken 7.1.1) produces.

    :param path: Output file.
    :param batches: Record batches of ``(text, image bytes)`` rows.
    :param kind: Dataset type stored in the metadata.
    :return: ``path``.
    """
    alphabet: dict = {}
    for batch in batches:
        for text, _ in batch:
            for ch in text:
                alphabet[ch] = alphabet.get(ch, 0) + 1
    total = sum(len(b) for b in batches)
    meta = {"type": kind, "alphabet": alphabet, "counts": {"all": total, "train": 0, "validation": 0, "test": 0}}
    schema = pa.schema(
        [("lines", LINE_TYPE), ("train", pa.bool_()), ("validation", pa.bool_()), ("test", pa.bool_())],
        metadata={"lines": json.dumps(meta)},
    )
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(sink, schema) as writer:
        for batch in batches:
            lines = pa.array([{"text": t, "im": im, "language": None} for t, im in batch], type=LINE_TYPE)
            flags = pa.array([False] * len(batch), type=pa.bool_())
            writer.write_batch(pa.record_batch([lines, flags, flags, flags], schema=schema))
    return path


def _rows(n: int, start: int = 0) -> List[Tuple[str, bytes]]:
    """``n`` rows whose image width encodes the row index (width = 100 + index).

    :param n: Row count.
    :param start: Index of the first row.
    :return: ``(text, png)`` rows.
    """
    return [(f"سطر رقم {i}", _encoded(100 + i, 32)) for i in range(start, start + n)]


def test_find_layout_kraken_struct_top_level_and_mismatch() -> None:
    kraken = pa.schema([("lines", LINE_TYPE), ("train", pa.bool_())])
    assert gl.find_layout(kraken) == gl.LineLayout("lines", "im", "text")
    flat = pa.schema([("image", pa.binary()), ("transcription", pa.string())])
    assert gl.find_layout(flat) == gl.LineLayout("", "image", "transcription")
    assert gl.find_layout(pa.schema([("pixels", pa.binary()), ("label", pa.string())])) is None


def test_sample_indices_seeded_sorted_and_bounded() -> None:
    first = gl.sample_indices(1000, 30, seed=7)
    assert first == gl.sample_indices(1000, 30, seed=7)
    assert first == sorted(set(first)) and len(first) == 30 and all(0 <= i < 1000 for i in first)
    assert first != gl.sample_indices(1000, 30, seed=8)
    assert gl.sample_indices(5, 30, seed=0) == [0, 1, 2, 3, 4]
    assert gl.sample_indices(0, 30, seed=0) == []


def test_locate_rows_handles_empty_batches_and_rejects_out_of_range() -> None:
    sizes = [3, 0, 2, 0]
    assert gl.locate_rows(sizes, [0, 2, 3, 4]) == [(0, 0, 0), (2, 0, 2), (3, 2, 0), (4, 2, 1)]
    with pytest.raises(IndexError):
        gl.locate_rows(sizes, [5])


def test_tsv_field_keeps_one_line_one_cell() -> None:
    assert gl.tsv_field("a\tb\nc\\d\r") == "a\\tb\\nc\\\\d\\r"


def test_alphabet_outliers_list_allowed_non_arabic_symbols_only() -> None:
    alphabet = {"ب": 50, " ": 40, "\u0654": 3, "1": 7, ".": 9, "A": 2, "[": 1, "\n": 9}
    out = dict(gl.alphabet_outliers(alphabet))
    assert len(out) == 2  # digit and full stop; U+0654 (NFD hamza) is Arabic, A [ and newline are violations
    labels = " ".join(out)
    assert "U+0031" in labels and "U+002E" in labels
    assert next(iter(out)).startswith("U+002E")  # most frequent first


def test_text_rule_violations_catch_marks_bidi_latin_brackets_and_whitespace() -> None:
    forbidden = ["\u0640", "\u064b", "\u0650", "\u0652", "\u061c", "\ufeff", "\u200f", "\u202b", "\u2067",
                 "A", "z", "\u00e9", "\u1e0d", "[", "]", "<", ">", "{", "}", "\u3008", "\u27e9", "\n", "\t", "\u00a0"]
    allowed = ["ب", "\u0654", "\u0653", "\u0621", "\u06a9", "\ufefb", "1", "\u0663", ".", "\u060c", " ", "(", "\u00d7"]
    for char in forbidden:
        assert gl.is_forbidden(char), f"U+{ord(char):04X}"
    for char in allowed:
        assert not gl.is_forbidden(char), f"U+{ord(char):04X}"
    assert gl.is_forbidden("b\u0640")  # a multi-character key is forbidden when any character is
    alphabet = {**{c: 1 for c in forbidden}, **{c: 5 for c in allowed}, "\u0640": 4}
    labels = [label for label, _ in gl.text_rule_violations(alphabet)]
    assert len(labels) == len(forbidden)
    assert labels[0].startswith("U+0640 ARABIC TATWEEL")  # most frequent first
    assert not set(labels) & {label for label, _ in gl.alphabet_outliers(alphabet)}


def test_cmd_sample_writes_pngs_tsv_and_html(tmp_path: Path) -> None:
    rows = _rows(4) + [("سطر جيبيج", _encoded(150, 40, "JPEG"))]
    arrow = _kraken_arrow(tmp_path / "agapet_train.arrow", [rows[:3], [], rows[3:]])
    out_root = tmp_path / "gate"
    assert gl.cmd_sample([arrow], out_root, n=10, seed=0) == 0
    sample = out_root / "agapet_train"
    tsv = (sample / "index.tsv").read_text(encoding="utf-8").splitlines()
    assert tsv[0] == "row\ttext\twidth\theight\tfile" and len(tsv) == 6
    for line in tsv[1:]:
        row, text, width, height, name = line.split("\t")
        assert text == rows[int(row)][0]
        with Image.open(sample / name) as im:
            assert im.format == "PNG" and im.size == (int(width), int(height))
    assert tsv[5].split("\t")[2:4] == ["150", "40"]  # the JPEG row was re-encoded, size kept
    assert tsv[2].split("\t")[2] == "101"  # row 1 is 101 px wide
    page = (sample / "index.html").read_text(encoding="utf-8")
    assert 'dir="rtl"' in page and "row_0000004.png" in page
    # A smaller re-run replaces the earlier dump instead of leaving stale PNGs behind.
    assert gl.cmd_sample([arrow], out_root, n=2, seed=0) == 0
    assert len(list(sample.glob("row_*.png"))) == 2
    assert len((sample / "index.tsv").read_text(encoding="utf-8").splitlines()) == 3


def test_cmd_sample_prints_schema_on_unknown_columns(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    schema = pa.schema([("pixels", pa.binary()), ("label", pa.string())])
    path = tmp_path / "odd.arrow"
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(sink, schema) as writer:
        writer.write_batch(pa.record_batch([pa.array([b"x"]), pa.array(["y"])], schema=schema))
    assert gl.cmd_sample([path], tmp_path / "gate", n=3, seed=0) == gl.EXIT_SCHEMA
    assert "pixels: binary" in capsys.readouterr().err
    assert not (tmp_path / "gate").exists()


def test_cmd_count_rows_outliers_and_mixed_types(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    a = _kraken_arrow(tmp_path / "rasam_train.arrow", [_rows(3), _rows(2, 3)])
    b = _kraken_arrow(tmp_path / "baybars_train.arrow", [[("سطر، ١٢ - 3.", _encoded(90, 30))]])
    assert gl.count_arrow(a).rows == 5 and gl.count_arrow(a).meta_all == 5
    assert gl.main(["count", str(a), str(b)]) == 0
    captured = capsys.readouterr()
    assert "total\t-\t6" in captured.out and "non-Arabic symbol U+002D HYPHEN-MINUS" in captured.out
    assert "TEXT RULE" not in captured.out + captured.err
    c = _kraken_arrow(tmp_path / "crops_train.arrow", [_rows(1)], kind="kraken_recognition_bbox")
    assert gl.main(["count", str(a), str(c)]) == gl.EXIT_MIXED


def test_cmd_count_fails_on_text_rule_violations(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    clean = _kraken_arrow(tmp_path / "rasam_train.arrow", [_rows(2)])
    leaky = _kraken_arrow(tmp_path / "agapet_val.arrow", [[("كتـاب [x] بِسم\u200f", _encoded(90, 30))]])
    assert gl.main(["count", str(clean), str(leaky)]) == gl.EXIT_TEXT_RULE
    captured = capsys.readouterr()
    for code in ("U+0640", "U+0650", "U+200F", "U+005B", "U+005D", "U+0078"):
        assert f"TEXT RULE violation {code}" in captured.out
    assert "agapet_val.arrow" in captured.err and "rasam_train.arrow" not in captured.err
    bbox = _kraken_arrow(tmp_path / "crops_train.arrow", [_rows(1)], kind="kraken_recognition_bbox")
    assert gl.main(["count", str(leaky), str(bbox)]) == gl.EXIT_TEXT_RULE  # both reported, violation wins
    assert "mixed dataset types" in capsys.readouterr().err


def _fake_kraken(root: Path, ppocr: str) -> Path:
    """Minimal kraken package tree holding the two files ``early-check`` scans.

    :param root: Directory to create the tree in.
    :param ppocr: Source text of ``train/ppocr.py``.
    :return: The fake package directory.
    """
    pkg = root / "kraken"
    (pkg / "ketos").mkdir(parents=True)
    (pkg / "train").mkdir()
    (pkg / "ketos" / "recognition.py").write_text(
        "trainer = KrakenTrainer(\n    max_epochs=params['epochs'] if params['quit'] == 'fixed' else -1,\n)\n",
        encoding="utf-8",
    )
    (pkg / "train" / "ppocr.py").write_text(ppocr, encoding="utf-8")
    return pkg


@pytest.mark.parametrize(
    "ppocr, status",
    [
        ("from lightning.pytorch.callbacks import EarlyStopping\nclass M:\n    pass\n", gl.EXIT_NO_EARLY_STOPPING),
        (
            "def configure_callbacks(self):\n    return [EarlyStopping(monitor='val_cer')]\n"
            "total = self.trainer.estimated_stepping_batches\n",
            gl.EXIT_INFINITE_STEPS,
        ),
        ("def configure_callbacks(self):\n    return [EarlyStopping(monitor='val_cer')]\n", 0),
    ],
)
def test_early_check_judges_ppocr_stop_hooks(tmp_path: Path, capsys: pytest.CaptureFixture, ppocr: str, status: int) -> None:
    pkg = _fake_kraken(tmp_path, ppocr)
    assert gl.main(["early-check", "--kraken-root", str(pkg)]) == status
    out = capsys.readouterr().out
    assert "ketos/recognition.py:2:max_epochs=params['epochs']" in out
    assert "train/ppocr.py:" in out


def test_trainer_hooks_and_status_from_lines() -> None:
    hooks = [("ketos/recognition.py", 9, "max_epochs=-1"), ("train/vgsl.py", 3, "EarlyStopping(monitor='x')")]
    assert gl.early_stop_status(hooks) == gl.EXIT_NO_EARLY_STOPPING  # EarlyStopping elsewhere does not count
    assert gl.early_stop_status(hooks + [(gl.PPOCR_FILE, 4, "cbs.append(EarlyStopping(monitor='val_cer'))")]) == 0
