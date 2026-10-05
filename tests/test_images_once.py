# File name: test_images_once.py
# Date: 9/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Round-trip tests for the images-once export (images stored once by content hash).

A small DatasetDict in the KTIV row schema (3 page images + 1 crop, 8 rows,
one page shared across splits, two pages sharing a file name) is saved with
``save_to_disk``, exported, and read back through ``ImagesOnceDataset``:
every row must equal the original and every stored image must hold the
original JPEG bytes verbatim. PNG sources export only with ``reencode_jpeg``,
as upright RGB JPEGs named by the SHA-1 of the bytes written.
"""
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset, DatasetDict, load_from_disk
from datasets import Image as ImageFeature
from PIL import Image

from src.finetuning.qwen_hebrew import images_once
from src.finetuning.qwen_hebrew.build_ktiv_dataset import FEATURES


def _jpeg(path: Path, color: Tuple[int, int, int], size: Tuple[int, int] = (64, 48)) -> Path:
    """Write a solid-colour JPEG.

    :param path: Destination.
    :param color: RGB fill.
    :param size: (width, height).
    :return: The path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, "JPEG", quality=90)
    return path


def _row(image: Path, task: str, stem: str, answer: str) -> Dict:
    """One row in the KTIV feature schema.

    :param image: Image file.
    :param task: Task family.
    :param stem: Row id.
    :param answer: Target text.
    :return: Row dict.
    """
    with Image.open(image) as im:
        w, h = im.size
    return {"image": str(image), "question": f"prompt for {task}", "answer": answer,
            "task": task, "section": "page", "stem": stem, "label_source": "ktiv_nli",
            "target_chars": len(answer), "target_tokens": 0,
            "image_width": w, "image_height": h}


@pytest.fixture
def saved(tmp_path: Path) -> Tuple[Path, Dict[str, List[Path]]]:
    """Saved DatasetDict plus each split's source image path per row."""
    p1 = _jpeg(tmp_path / "src/111/FL1.jpg", (200, 10, 10))
    p2 = _jpeg(tmp_path / "src/222/FL1.jpg", (10, 200, 10))   # same name, other bytes
    p3 = _jpeg(tmp_path / "src/333/FL3.jpg", (10, 10, 200))
    crop = _jpeg(tmp_path / "src/111/FL1_line0.jpg", (90, 90, 90), size=(64, 16))
    train = [(p1, "fragment_transcribe", "ktiv_111_FL1", "שורה ראשונה\nשורה שנייה"),
             (p1, "locate", "ktiv_111_FL1_loc0", '{"bbox_2d": [1, 2, 30, 40]}'),
             (p1, "layout_qa", "ktiv_111_FL1_qa", "2"),
             (crop, "line_transcribe", "ktiv_111_FL1_line0", "שורה ראשונה"),
             (p2, "fragment_transcribe", "ktiv_222_FL1", "טקסט אחר"),
             (p2, "page_short", "ktiv_222_FL1b", "קטע [קצ]ר")]
    val = [(p3, "fragment_transcribe", "ktiv_333_FL3", "עמוד שלישי"),
           (p1, "region_transcribe", "ktiv_111_FL1_first_line", "שורה ראשונה")]
    splits = {"train": train, "val": val}
    dsd = DatasetDict({name: Dataset.from_list([_row(*r) for r in rows], features=FEATURES)
                       for name, rows in splits.items()})
    dsd.save_to_disk(str(tmp_path / "ds"))
    return tmp_path / "ds", {name: [r[0] for r in rows] for name, rows in splits.items()}


def test_export_round_trip_and_dedupe(saved, tmp_path):
    ds_dir, sources = saved
    out = tmp_path / "once"
    manifest = images_once.export_images_once(ds_dir, out)
    assert manifest["n_rows"] == 8
    assert manifest["n_images"] == 4                      # 3 pages + 1 crop, stored once
    assert manifest["splits"]["train"]["unique_images"] == 3
    assert manifest["splits"]["val"]["unique_images"] == 2
    assert sorted(p.name for p in (out / "images").iterdir()) == \
        sorted(f"{images_once.image_sha1(p.read_bytes())}.jpg"
               for p in {*sources["train"], *sources["val"]})
    total = sum(p.stat().st_size for rows in sources.values() for p in rows)
    assert manifest["row_image_bytes"] == total
    src = load_from_disk(str(ds_dir))
    for split in ("train", "val"):
        table = pq.read_table(out / "rows" / f"{split}.parquet")
        assert "image" not in table.column_names and images_once.SHA_COLUMN in table.column_names
        raw = src[split].cast_column("image", ImageFeature(decode=False))
        ds = images_once.ImagesOnceDataset(out / "rows" / f"{split}.parquet", out / "images")
        assert len(ds) == len(src[split])
        for i in range(len(ds)):
            got, want = ds[i], src[split][i]
            assert list(got) == list(want)                  # same keys, same order
            assert {k: v for k, v in got.items() if k != "image"} == \
                {k: v for k, v in want.items() if k != "image"}
            assert (got["image"].size, got["image"].mode) == (want["image"].size, want["image"].mode)
            assert got["image"].tobytes() == want["image"].tobytes()
            stored = ds.image_path(table[images_once.SHA_COLUMN][i].as_py()).read_bytes()
            assert stored == sources[split][i].read_bytes()   # verbatim JPEG bytes
            assert raw[i]["image"]["bytes"] in (None, stored)


def test_export_is_resumable(saved, tmp_path):
    ds_dir, _ = saved
    first = images_once.export_images_once(ds_dir, tmp_path / "once")
    stamps = {p.name: p.stat().st_mtime_ns for p in (tmp_path / "once" / "images").iterdir()}
    again = images_once.export_images_once(ds_dir, tmp_path / "once")
    assert first == again
    assert {p.name: p.stat().st_mtime_ns
            for p in (tmp_path / "once" / "images").iterdir()} == stamps   # nothing rewritten
    assert len(stamps) == 4


def test_task_filter_gives_one_family(saved, tmp_path):
    ds_dir, _ = saved
    out = tmp_path / "once"
    images_once.export_images_once(ds_dir, out)
    ds = images_once.ImagesOnceDataset(out / "rows" / "train.parquet", out / "images",
                                       tasks=["page_short"])
    assert len(ds) == 1
    want = next(r for r in load_from_disk(str(ds_dir))["train"] if r["task"] == "page_short")
    assert ds[0]["stem"] == want["stem"] and ds[0]["image"].tobytes() == want["image"].tobytes()


def test_missing_image_fails_at_construction(saved, tmp_path):
    ds_dir, _ = saved
    out = tmp_path / "once"
    images_once.export_images_once(ds_dir, out)
    next((out / "images").iterdir()).unlink()
    with pytest.raises(FileNotFoundError):
        images_once.ImagesOnceDataset(out / "rows" / "train.parquet", out / "images")


def test_resplit_moves_rows_by_key_and_leaves_images_alone(saved, tmp_path):
    ds_dir, _ = saved
    out = tmp_path / "once"
    images_once.export_images_once(ds_dir, out)
    stamps = {p.name: p.stat().st_mtime_ns for p in (out / "images").iterdir()}
    src = load_from_disk(str(ds_dir))
    by_stem = {r["stem"]: r for split in ("train", "val") for r in src[split]}

    def ms(stem: str) -> str:
        """Manuscript key of a fixture stem."""
        return stem.split("_")[1]

    new = images_once.resplit_rows(out, {"222"}, ms)
    # 222 moves train -> val; everything in old val (333, one 111 row) moves to
    # train, appended after the old train rows that stay
    assert [ms(s) for s in new["val"]["stem"].to_pylist()] == ["222", "222"]
    assert new["train"]["stem"].to_pylist() == [
        "ktiv_111_FL1", "ktiv_111_FL1_loc0", "ktiv_111_FL1_qa", "ktiv_111_FL1_line0",
        "ktiv_333_FL3", "ktiv_111_FL1_first_line"]
    for split in ("train", "val"):
        ds = images_once.ImagesOnceDataset(out / "rows" / f"{split}.parquet", out / "images")
        for i in range(len(ds)):
            got, want = ds[i], by_stem[ds[i]["stem"]]
            assert list(got) == list(want)
            assert {k: v for k, v in got.items() if k != "image"} == \
                {k: v for k, v in want.items() if k != "image"}
            assert got["image"].tobytes() == want["image"].tobytes()
    assert {p.name: p.stat().st_mtime_ns for p in (out / "images").iterdir()} == stamps
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["splits"]["val"]["rows"] == 2 and manifest["splits"]["train"]["rows"] == 6
    assert manifest["splits"]["val"]["unique_images"] == 1


def test_rows_table_does_not_pin_image_bytes(tmp_path):
    # Batches read from an IPC stream share one body buffer across columns; the
    # rows table kept per batch must not hold the batch's image bytes alive.
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    images = pa.array([{"bytes": bytes(1_000_000), "path": f"{i}.jpg"} for i in range(8)],
                      image_type)
    table = pa.table({"stem": [f"ktiv_1_FL{i}" for i in range(8)], "image": images,
                      "task": ["page_short"] * 8})
    path = tmp_path / "shard.arrow"
    with pa.OSFile(str(path), "wb") as fh, pa.ipc.new_stream(fh, table.schema) as writer:
        writer.write_table(table)
    del images, table
    base = pa.total_allocated_bytes()
    with pa.OSFile(str(path)) as fh:
        batch = pa.ipc.open_stream(fh).read_next_batch()
    rows = images_once.rows_table(batch, [f"sha{i}" for i in range(8)])
    del batch
    assert rows.column_names == ["stem", images_once.SHA_COLUMN, "task"]
    assert rows[images_once.SHA_COLUMN].to_pylist() == [f"sha{i}" for i in range(8)]
    assert pa.total_allocated_bytes() - base < 100_000     # vs ~8 MB of image bytes


def test_non_jpeg_image_is_rejected(tmp_path):
    png = tmp_path / "page.png"
    Image.new("RGB", (8, 8), "white").save(png, "PNG")
    dsd = DatasetDict({"train": Dataset.from_list([{"image": str(png)}]).cast_column(
        "image", ImageFeature())})
    dsd.save_to_disk(str(tmp_path / "ds"))
    with pytest.raises(ValueError, match="not a JPEG"):
        images_once.export_images_once(tmp_path / "ds", tmp_path / "once")


def _encoded(im: Image.Image, fmt: str, orientation: Optional[int] = None) -> bytes:
    """Encode an image in memory.

    :param im: Image.
    :param fmt: PIL format name (``PNG`` / ``JPEG``).
    :param orientation: EXIF Orientation tag to embed, or None.
    :return: Encoded bytes.
    """
    buf = io.BytesIO()
    extra = {}
    if orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = orientation
        extra["exif"] = exif
    im.save(buf, fmt, **extra)
    return buf.getvalue()


def _two_tone(size: Tuple[int, int], mode: str = "RGB") -> Image.Image:
    """Left half red-ish, right half blue-ish, so orientation is visible.

    :param size: (width, height).
    :param mode: PIL mode of the result.
    :return: Image.
    """
    im = Image.new("RGB", size, (220, 40, 40))
    im.paste((30, 60, 210), (size[0] // 2, 0, size[0], size[1]))
    return im.convert(mode)


def _bytes_row(data: bytes, task: str, section: str, stem: str) -> Dict:
    """One KTIV-schema row embedding raw image bytes.

    :param data: Encoded image bytes.
    :param task: Task family.
    :param section: Section name.
    :param stem: Row id.
    :return: Row dict.
    """
    with Image.open(io.BytesIO(data)) as im:
        w, h = im.size
    return {"image": {"bytes": data, "path": None}, "question": f"prompt for {task}",
            "answer": f"text of {stem}", "task": task, "section": section, "stem": stem,
            "label_source": "talmud", "target_chars": 10, "target_tokens": 0,
            "image_width": w, "image_height": h}


@pytest.fixture
def saved_png(tmp_path: Path) -> Tuple[Path, Dict[str, bytes]]:
    """Saved DatasetDict of PNG rows (one page shared by three rows across splits) + one JPEG row."""
    images = {"page": _encoded(_two_tone((64, 48)), "PNG"),
              "crop": _encoded(_two_tone((64, 16), "L"), "PNG"),
              "photo": _encoded(_two_tone((32, 32)), "JPEG")}
    train = [_bytes_row(images["page"], "page_extract", "rashi", "01_2"),
             _bytes_row(images["page"], "page_extract", "tosafot", "01_2"),
             _bytes_row(images["crop"], "crop_transcribe", "gemara", "01_2_gemara"),
             _bytes_row(images["photo"], "page_extract", "rashi", "01_3")]
    val = [_bytes_row(images["page"], "page_extract", "gemara", "01_2")]
    dsd = DatasetDict({"train": Dataset.from_list(train, features=FEATURES),
                       "val": Dataset.from_list(val, features=FEATURES)})
    dsd.save_to_disk(str(tmp_path / "ds"))
    return tmp_path / "ds", images


def test_png_dataset_needs_reencode_flag(saved_png, tmp_path):
    ds_dir, _ = saved_png
    with pytest.raises(ValueError, match="not a JPEG"):
        images_once.export_images_once(ds_dir, tmp_path / "plain")


def test_reencode_jpeg_exports_png_dataset(saved_png, tmp_path):
    ds_dir, images = saved_png
    out = tmp_path / "once"
    manifest = images_once.export_images_once(ds_dir, out, reencode_jpeg=True)
    assert manifest["reencode_jpeg"] is True and manifest["jpeg_quality"] == 95
    assert manifest["reencoded_images"] == 2                 # page + crop, each decoded once
    assert manifest["n_images"] == 3                         # + the JPEG row, verbatim
    stored = {p.name: p.read_bytes() for p in (out / "images").iterdir()}
    for name, data in stored.items():
        assert name == f"{images_once.image_sha1(data)}.jpg"   # hash of the bytes written
        assert data.startswith(images_once.JPEG_MAGIC)
    assert stored[f"{images_once.image_sha1(images['photo'])}.jpg"] == images["photo"]
    page_sha = images_once.image_sha1(images_once.as_jpeg(images["page"]))
    assert manifest["splits"]["val"]["unique_images"] == 1
    assert manifest["row_image_bytes"] == sum(
        len(stored[f"{s}.jpg"]) for split in ("train", "val")
        for s in pq.read_table(out / "rows" / f"{split}.parquet")[images_once.SHA_COLUMN].to_pylist())

    src = load_from_disk(str(ds_dir))
    for split in ("train", "val"):
        table = pq.read_table(out / "rows" / f"{split}.parquet")
        ds = images_once.ImagesOnceDataset(out / "rows" / f"{split}.parquet", out / "images")
        for i in range(len(ds)):
            got, want = ds[i], src[split][i]
            assert list(got) == list(want)
            assert {k: v for k, v in got.items() if k != "image"} == \
                {k: v for k, v in want.items() if k != "image"}   # widths/heights unchanged
            assert got["image"].mode == "RGB"
            assert got["image"].size == (want["image_width"], want["image_height"]) == want["image"].size
            diff = np.abs(np.asarray(got["image"], np.int16) -
                          np.asarray(want["image"].convert("RGB"), np.int16))
            assert diff.mean() < 3
        if split == "val":
            assert table[images_once.SHA_COLUMN].to_pylist() == [page_sha]   # train's page, reused


def test_reencode_keeps_exif_orientation_semantics(tmp_path):
    rotated = _encoded(_two_tone((64, 32)), "PNG", orientation=6)   # displayed as 32x64
    want = ImageFeature().decode_example({"path": None, "bytes": rotated})
    jpeg = images_once.as_jpeg(rotated)
    with Image.open(io.BytesIO(jpeg)) as im:
        assert im.format == "JPEG" and im.getexif().get(0x0112) is None
    got = ImageFeature().decode_example({"path": None, "bytes": jpeg})
    assert got.size == want.size == (32, 64)
    diff = np.abs(np.asarray(got, np.int16) - np.asarray(want.convert("RGB"), np.int16))
    assert diff.mean() < 3


@pytest.mark.parametrize("mode", ["1", "L", "LA", "P", "RGBA", "I;16"])
def test_as_jpeg_gives_rgb_for_any_mode(mode):
    data = _encoded(_two_tone((16, 8), mode if mode != "I;16" else "L").convert(mode), "PNG")
    jpeg = images_once.as_jpeg(data)
    assert jpeg.startswith(images_once.JPEG_MAGIC)
    assert jpeg == images_once.as_jpeg(data)                 # deterministic -> dedupes
    with Image.open(io.BytesIO(jpeg)) as im:
        assert (im.mode, im.size) == ("RGB", (16, 8))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_subset_export_keeps_chosen_splits_and_tasks(saved, tmp_path):
    ds_dir, sources = saved
    full = tmp_path / "once"
    images_once.export_images_once(ds_dir, full)
    out = tmp_path / "pages_only"
    manifest = images_once.subset_export(full, out, {"train": "train", "val": "val"}, tasks={"fragment_transcribe"})
    assert manifest["splits"] == {"train": {"rows": 2, "unique_images": 2, "from_split": "train"},
                                  "val": {"rows": 1, "unique_images": 1, "from_split": "val"}}
    assert manifest["n_rows"] == 3 and manifest["n_images"] == 3 and manifest["tasks"] == ["fragment_transcribe"]
    assert manifest["subset_of"] == str(full) and (out / "manifest.json").exists()
    assert len(list((out / "images").iterdir())) == 3                 # the line crop is not referenced any more
    source_table = pq.read_table(full / "rows" / "train.parquet")
    table = pq.read_table(out / "rows" / "train.parquet")
    assert table.schema.metadata == source_table.schema.metadata and table.column_names == source_table.column_names
    ds = images_once.ImagesOnceDataset(out / "rows" / "train.parquet", out / "images")
    assert [ds[i]["stem"] for i in range(len(ds))] == ["ktiv_111_FL1", "ktiv_222_FL1"]
    assert images_once.subset_export(full, out, {"train": "train", "val": "val"}, tasks={"fragment_transcribe"}) == manifest


def test_subset_export_can_rename_a_split_and_refuses_bad_requests(saved, tmp_path):
    ds_dir, _ = saved
    full = tmp_path / "once"
    images_once.export_images_once(ds_dir, full)
    manifest = images_once.subset_export(full, tmp_path / "renamed", {"val": "train"})
    assert manifest["splits"] == {"train": {"rows": 2, "unique_images": 2, "from_split": "val"}} and manifest["tasks"] is None
    with pytest.raises(ValueError, match="one new split"):
        images_once.subset_export(full, tmp_path / "clash", {"train": "train", "val": "train"})
    with pytest.raises(ValueError, match="no rows left"):
        images_once.subset_export(full, tmp_path / "empty", {"val": "val"}, tasks={"locate"})
