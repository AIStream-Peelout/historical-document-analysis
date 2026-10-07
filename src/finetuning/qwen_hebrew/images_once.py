# File name: images_once.py
# Date: 9/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Images-once export of a saved DatasetDict, plus its map-style loader.

``save_to_disk`` embeds every row's image, and a KTIV page image is shared by
the ~15 rows asked about that page (page, region, grounding, QA families), so
the v3 arrow is ~77 GB of mostly duplicated JPEG bytes. The export stores each
distinct image once, named by the SHA-1 of its bytes (crop rows carry their
own crops, so identity is by content, never by file name), and one parquet of
every other column per split with an ``image_sha1`` pointer::

    <out_dir>/images/<sha1>.jpg       original JPEG bytes, verbatim (or a re-encode, below)
    <out_dir>/rows/<split>.parquet    all non-image columns + image_sha1
    <out_dir>/manifest.json           counts, bytes, dedupe ratio, features

:class:`ImagesOnceDataset` reads one split back as a map-style dataset whose
items equal the original rows: same keys in the same order, the image
decoded by ``datasets.Image`` exactly as ``load_from_disk`` would decode it.

Stored names promise JPEG, so non-JPEG image bytes are rejected unless
``--reencode-jpeg`` is given: then each distinct non-JPEG image (e.g. the
Talmud replay's PNG pages) is decoded once, turned upright with
``ImageOps.exif_transpose`` (what ``datasets.Image`` does when decoding),
converted to RGB and re-encoded as a quality-95 JPEG, stored under the SHA-1
of the JPEG bytes written. JPEG inputs stay verbatim either way; row columns
(``image_width`` / ``image_height`` included) are never changed, and a
re-encoded image decodes to RGB whatever the source mode was.

The export streams the saved arrow shards with plain sequential reads, one
record batch at a time, instead of ``load_from_disk``: a memory-mapped load
over the NAS (SMB) mount paged in ~30 GB of the v3 arrow before touching a
row (2026-09-22), starving concurrent NAS jobs; sequential reads keep memory
at one batch (~100 rows).

Usage (repo root):
    .venv/bin/python -m src.finetuning.qwen_hebrew.images_once \\
        --src /Volumes/home/studio_offload/datasets/genizah_ktiv_v4 \\
        --out /Volumes/home/studio_offload/datasets/genizah_ktiv_v4_images_once
    # PNG sources (Talmud replay): add --reencode-jpeg
"""
import argparse
import hashlib
import io
import json
import logging
import os
import shutil
import time
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import Features, Image
from PIL import Image as PILImage
from PIL import ImageOps
from torch.utils.data import Dataset as TorchDataset

logger = logging.getLogger(__name__)

IMAGE_COLUMN = "image"
SHA_COLUMN = "image_sha1"
META_KEY = b"images_once"
JPEG_MAGIC = b"\xff\xd8\xff"
JPEG_QUALITY = 95
REENCODE_WORKERS = 4


def image_sha1(data: bytes) -> str:
    """Content hash that names an exported image.

    :param data: Encoded image bytes.
    :type data: bytes
    :return: Hex SHA-1 digest.
    :rtype: str
    """
    return hashlib.sha1(data).hexdigest()


def cell_bytes(cell: Dict) -> bytes:
    """Encoded bytes of one undecoded ``datasets.Image`` cell.

    :param cell: ``{"bytes": ..., "path": ...}`` storage struct.
    :type cell: Dict
    :return: The embedded bytes, else the bytes of the file at ``path``.
    :rtype: bytes
    """
    if cell["bytes"] is not None:
        return cell["bytes"]
    return Path(cell["path"]).read_bytes()


def write_image_once(data: bytes, images_dir: Path, seen: Dict[str, int],
                     on_disk: set) -> str:
    """Store image bytes under their hash unless already stored.

    The write goes through a temporary name, so an interrupted export never
    leaves a truncated ``<sha1>.jpg`` that a re-run would trust.

    :param data: Encoded JPEG bytes.
    :type data: bytes
    :param images_dir: ``<out_dir>/images``.
    :type images_dir: Path
    :param seen: sha1 -> byte size of images handled this run (updated).
    :type seen: Dict[str, int]
    :param on_disk: sha1s already stored before this run (one directory
        listing instead of a stat per image on the NAS).
    :type on_disk: set
    :return: The image's sha1.
    :rtype: str
    :raises ValueError: For non-JPEG bytes (stored names promise ``.jpg``).
    """
    sha = image_sha1(data)
    if sha in seen:
        return sha
    if not data.startswith(JPEG_MAGIC):
        raise ValueError(f"image {sha} is not a JPEG; images-once stores JPEG bytes verbatim "
                         "(export with --reencode-jpeg to convert)")
    if sha not in on_disk:
        tmp = images_dir / f"{sha}.jpg.tmp"
        tmp.write_bytes(data)
        os.replace(tmp, images_dir / f"{sha}.jpg")
    seen[sha] = len(data)
    return sha


def as_jpeg(data: bytes, quality: int = JPEG_QUALITY) -> bytes:
    """Re-encode image bytes as an upright RGB JPEG.

    ``ImageOps.exif_transpose`` runs before saving, so the stored pixels are
    already in the orientation ``datasets.Image`` shows after decoding the
    original bytes; the JPEG carries no EXIF, so nothing rotates it twice.
    Encoding is deterministic, so identical inputs give identical bytes.

    :param data: Encoded image bytes in any format PIL reads.
    :type data: bytes
    :param quality: JPEG quality.
    :type quality: int
    :return: JPEG bytes.
    :rtype: bytes
    """
    with PILImage.open(io.BytesIO(data)) as im:
        upright = ImageOps.exif_transpose(im).convert("RGB")
    buf = io.BytesIO()
    upright.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def store_images(datas: Sequence[bytes], images_dir: Path, seen: Dict[str, int], on_disk: set,
                 reencoded: Optional[Dict[str, str]] = None,
                 pool: Optional[Executor] = None) -> List[str]:
    """Store one batch's images once each; return the stored sha1 of every row's image.

    Without ``reencoded`` every image must be a JPEG and is stored verbatim.
    With it (``--reencode-jpeg``) JPEG bytes are still stored verbatim, while
    each distinct non-JPEG input is re-encoded once (:func:`as_jpeg`, on
    ``pool`` when given) and stored under the SHA-1 of the JPEG bytes.

    :param datas: Encoded image bytes, one per row.
    :type datas: Sequence[bytes]
    :param images_dir: ``<out_dir>/images``.
    :type images_dir: Path
    :param seen: sha1 -> byte size of images handled this run (updated).
    :type seen: Dict[str, int]
    :param on_disk: sha1s already stored before this run.
    :type on_disk: set
    :param reencoded: sha1 of a non-JPEG input -> sha1 of its stored JPEG
        (updated), so an image shared by several rows is decoded once; None
        disables re-encoding.
    :type reencoded: Optional[Dict[str, str]]
    :param pool: Executor for the re-encodes; None encodes in this thread.
    :type pool: Optional[Executor]
    :return: Stored sha1 per row, in row order.
    :rtype: List[str]
    :raises ValueError: For non-JPEG bytes when re-encoding is disabled.
    """
    if reencoded is None:
        return [write_image_once(data, images_dir, seen, on_disk) for data in datas]
    keys = [None if data.startswith(JPEG_MAGIC) else image_sha1(data) for data in datas]
    todo = {key: data for key, data in zip(keys, datas) if key is not None and key not in reencoded}
    encoded = (pool.map if pool is not None else map)(as_jpeg, todo.values())
    for key, jpeg in zip(todo, encoded):
        reencoded[key] = write_image_once(jpeg, images_dir, seen, on_disk)
    return [write_image_once(data, images_dir, seen, on_disk) if key is None else reencoded[key]
            for key, data in zip(keys, datas)]


def saved_splits(dataset_dir: Path) -> List[str]:
    """Split names of a ``DatasetDict.save_to_disk`` directory.

    :param dataset_dir: Saved DatasetDict directory.
    :type dataset_dir: Path
    :return: Split names in saved order.
    :rtype: List[str]
    :raises TypeError: When the directory holds no ``dataset_dict.json``.
    """
    marker = dataset_dir / "dataset_dict.json"
    if not marker.exists():
        raise TypeError(f"{dataset_dir} is not a saved DatasetDict (no dataset_dict.json)")
    return list(json.loads(marker.read_text())["splits"])


def split_features(split_dir: Path) -> Features:
    """Features of one saved split, from its ``dataset_info.json``.

    :param split_dir: ``<dataset_dir>/<split>``.
    :type split_dir: Path
    :return: The split's features.
    :rtype: Features
    """
    return Features.from_dict(json.loads((split_dir / "dataset_info.json").read_text())["features"])


def iter_saved_batches(split_dir: Path) -> Iterator[pa.RecordBatch]:
    """Stream a saved split's record batches in row order with sequential reads.

    :param split_dir: ``<dataset_dir>/<split>``.
    :type split_dir: Path
    :return: Iterator over the stored record batches (raw storage: an Image
        column is its ``{"bytes", "path"}`` struct).
    :rtype: Iterator[pa.RecordBatch]
    :raises ValueError: For a split saved with an indices mapping.
    """
    state = json.loads((split_dir / "state.json").read_text())
    if state.get("_indices_data_files"):
        raise ValueError(f"{split_dir} has an indices mapping; save a flattened copy first")
    for entry in state["_data_files"]:
        with pa.OSFile(str(split_dir / entry["filename"])) as fh:
            yield from pa.ipc.open_stream(fh)


def rows_table(batch: pa.RecordBatch, shas: List[str]) -> pa.Table:
    """Non-image columns of one stored batch plus the ``image_sha1`` pointer.

    Columns of a batch read from an IPC stream are zero-copy slices of the
    batch's whole message body, images included, so the kept columns are
    copied out (``take``): a retained rows table must never pin ~100 rows of
    image bytes, or a split's export would hold the whole arrow in memory.

    :param batch: One stored record batch (raw storage).
    :type batch: pa.RecordBatch
    :param shas: sha1 of each row's image, in row order.
    :type shas: List[str]
    :return: Table with :data:`SHA_COLUMN` in the image column's position,
        schema metadata dropped.
    :rtype: pa.Table
    """
    table = pa.Table.from_batches([batch])
    idx = table.column_names.index(IMAGE_COLUMN)
    table = table.remove_column(idx).add_column(idx, SHA_COLUMN, pa.array(shas, pa.string()))
    return table.take(pa.array(range(table.num_rows))).replace_schema_metadata(None)


def export_images_once(dataset_dir: Path, out_dir: Path, reencode_jpeg: bool = False,
                       workers: int = REENCODE_WORKERS) -> Dict:
    """Convert a saved DatasetDict with embedded images into the images-once layout.

    :param dataset_dir: ``save_to_disk`` directory of a DatasetDict whose
        :data:`IMAGE_COLUMN` is a ``datasets.Image`` feature.
    :type dataset_dir: Path
    :param out_dir: Destination (created); re-running resumes image writes.
    :type out_dir: Path
    :param reencode_jpeg: Re-encode non-JPEG images as quality-95 RGB JPEGs
        (:func:`as_jpeg`) instead of rejecting them.
    :type reencode_jpeg: bool
    :param workers: Threads re-encoding a batch's images.
    :type workers: int
    :return: The manifest (also written to ``out_dir/manifest.json``).
    :rtype: Dict
    :raises ValueError: When a split lacks the image column, or for a
        non-JPEG image without ``reencode_jpeg``.
    """
    splits = saved_splits(dataset_dir)
    images_dir, rows_dir = out_dir / "images", out_dir / "rows"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows_dir.mkdir(parents=True, exist_ok=True)
    on_disk = {p.name[:-len(".jpg")] for p in images_dir.iterdir() if p.name.endswith(".jpg")}
    seen: Dict[str, int] = {}
    reencoded: Optional[Dict[str, str]] = {} if reencode_jpeg else None
    manifest = {"source": str(dataset_dir), "image_column": IMAGE_COLUMN,
                "sha_column": SHA_COLUMN, "reencode_jpeg": reencode_jpeg, "splits": {}}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for split in splits:
            features = split_features(dataset_dir / split)
            feature = features.get(IMAGE_COLUMN)
            if not isinstance(feature, Image):
                raise ValueError(f"split {split!r} has no datasets.Image column {IMAGE_COLUMN!r}")
            meta = {"columns": list(features), "image_column": IMAGE_COLUMN,
                    "image_mode": feature.mode, "features": features.to_dict()}
            tables: List[pa.Table] = []
            split_shas = set()
            row_image_bytes = 0
            for batch in iter_saved_batches(dataset_dir / split):
                datas = [cell_bytes(cell) for cell in batch.column(IMAGE_COLUMN).to_pylist()]
                shas = store_images(datas, images_dir, seen, on_disk, reencoded, pool)
                row_image_bytes += sum(seen[s] for s in shas)
                tables.append(rows_table(batch, shas))
                split_shas.update(shas)
                if len(tables) % 50 == 0:
                    logger.info("%s: %d rows, %d unique images so far, %.0fs", split,
                                sum(t.num_rows for t in tables), len(seen), time.time() - t0)
            rows = pa.concat_tables(tables).replace_schema_metadata(
                {META_KEY: json.dumps(meta, ensure_ascii=False).encode()})
            tmp = rows_dir / f"{split}.parquet.tmp"
            pq.write_table(rows, tmp)
            os.replace(tmp, rows_dir / f"{split}.parquet")
            manifest["splits"][split] = {
                "rows": rows.num_rows, "unique_images": len(split_shas),
                "row_image_bytes": row_image_bytes,
                "image_bytes": sum(seen[s] for s in split_shas)}
            logger.info("%s: %d rows -> %d unique images", split, rows.num_rows, len(split_shas))
    if reencode_jpeg:
        manifest["jpeg_quality"] = JPEG_QUALITY
        manifest["reencoded_images"] = len(reencoded)
    manifest["features"] = split_features(dataset_dir / splits[0]).to_dict()
    manifest["n_rows"] = sum(s["rows"] for s in manifest["splits"].values())
    manifest["n_images"] = len(seen)
    manifest["image_bytes"] = sum(seen.values())
    manifest["row_image_bytes"] = sum(s["row_image_bytes"] for s in manifest["splits"].values())
    manifest["dedupe_ratio"] = round(manifest["row_image_bytes"] / max(1, manifest["image_bytes"]), 3)
    if (dataset_dir / "stats.json").exists():
        shutil.copy2(dataset_dir / "stats.json", out_dir / "stats.json")
    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    return manifest


def subset_export(export_dir: Path, out_dir: Path, splits: Mapping[str, str],
                  tasks: Optional[Iterable[str]] = None, task_column: str = "task") -> Dict:
    """Write a new images-once export that holds some splits (and some tasks) of an existing one.

    Rows keep their layout and parquet metadata; only the images those rows reference are copied.
    Example: the full documentary pages of the edition set as their own source
    (``{"train_page": "train", "val": "val"}``, ``tasks={"fragment_transcribe"}``).

    :param export_dir: Existing images-once export.
    :type export_dir: Path
    :param out_dir: Destination (created); images already there are not copied again.
    :type out_dir: Path
    :param splits: Source split name -> split name in the new export.
    :type splits: Mapping[str, str]
    :param tasks: Keep only rows whose ``task_column`` is one of these; None keeps every row.
    :type tasks: Optional[Iterable[str]]
    :param task_column: Name of the task column.
    :type task_column: str
    :return: The manifest (also written to ``out_dir/manifest.json``).
    :rtype: Dict
    :raises ValueError: When two source splits map to the same new split, or a kept split is empty.
    """
    if len(set(splits.values())) != len(splits):
        raise ValueError(f"two source splits map to one new split: {dict(splits)}")
    keep = None if tasks is None else pa.array(sorted(set(tasks)), pa.string())
    images_dir, rows_dir = out_dir / "images", out_dir / "rows"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows_dir.mkdir(parents=True, exist_ok=True)
    present = {p.name for p in images_dir.iterdir()}
    with open(export_dir / "manifest.json") as fh:
        source_manifest = json.load(fh)
    manifest = {"source": str(export_dir), "subset_of": str(export_dir), "image_column": IMAGE_COLUMN,
                "sha_column": SHA_COLUMN, "reencode_jpeg": source_manifest.get("reencode_jpeg", False),
                "tasks": None if tasks is None else sorted(set(tasks)), "splits": {}}
    all_shas: set = set()
    image_bytes = 0
    for src, dst in splits.items():
        table = pq.read_table(export_dir / "rows" / f"{src}.parquet")
        if keep is not None:
            table = table.filter(pc.is_in(table[task_column], value_set=keep))
        if table.num_rows == 0:
            raise ValueError(f"split {src!r} has no rows left for tasks {manifest['tasks']}")
        shas = set(table[SHA_COLUMN].to_pylist())
        for sha in sorted(shas - all_shas):
            name = f"{sha}.jpg"
            if name not in present:
                shutil.copyfile(export_dir / "images" / name, images_dir / name)
                present.add(name)
            image_bytes += (images_dir / name).stat().st_size
        all_shas |= shas
        tmp = rows_dir / f"{dst}.parquet.tmp"
        pq.write_table(table, tmp)
        os.replace(tmp, rows_dir / f"{dst}.parquet")
        manifest["splits"][dst] = {"rows": table.num_rows, "unique_images": len(shas), "from_split": src}
    manifest["features"] = source_manifest.get("features")
    manifest["n_rows"] = sum(s["rows"] for s in manifest["splits"].values())
    manifest["n_images"] = len(all_shas)
    manifest["image_bytes"] = image_bytes
    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    return manifest


def resplit_rows(out_dir: Path, val_keys: set, key_of: Callable[[str], str],
                 key_column: str = "stem") -> Dict[str, pa.Table]:
    """Re-partition an export's train/val rows by a row key; images are untouched.

    Each new split lists the rows it takes from the old train file first, then
    those from the old val file, both in their original order. The parquet
    metadata (column order, image mode) is kept and each file is replaced
    atomically; ``manifest.json``'s per-split entries are recomputed.

    :param out_dir: Images-once export directory.
    :type out_dir: Path
    :param val_keys: Keys (e.g. manuscript sys_nums) whose rows go to val.
    :type val_keys: set
    :param key_of: Maps a ``key_column`` value to its key.
    :type key_of: Callable[[str], str]
    :param key_column: Column the key is derived from.
    :type key_column: str
    :return: The new ``{"train": table, "val": table}``.
    :rtype: Dict[str, pa.Table]
    """
    rows_dir = out_dir / "rows"
    old = {s: pq.read_table(rows_dir / f"{s}.parquet") for s in ("train", "val")}
    meta = old["train"].schema.metadata
    in_val = {s: pa.array([key_of(v) in val_keys for v in t[key_column].to_pylist()], pa.bool_())
              for s, t in old.items()}
    new = {}
    for split in ("train", "val"):
        parts = [t.filter(in_val[s] if split == "val" else pc.invert(in_val[s]))
                 for s, t in old.items()]
        new[split] = pa.concat_tables(parts).replace_schema_metadata(meta)
    for split, table in new.items():
        tmp = rows_dir / f"{split}.parquet.tmp"
        pq.write_table(table, tmp)
        os.replace(tmp, rows_dir / f"{split}.parquet")
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        sizes: Dict[str, int] = {}
        for split, table in new.items():
            shas = table[SHA_COLUMN].to_pylist()
            for sha in set(shas) - set(sizes):
                sizes[sha] = (out_dir / "images" / f"{sha}.jpg").stat().st_size
            manifest["splits"][split] = {
                "rows": table.num_rows, "unique_images": len(set(shas)),
                "row_image_bytes": sum(sizes[s] for s in shas),
                "image_bytes": sum(sizes[s] for s in set(shas))}
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return new


class ImagesOnceDataset(TorchDataset):
    """Map-style dataset over one exported split; items equal the original rows.

    Rows stay in one Arrow table (not a list of Python dicts), so forked
    DataLoader workers share them without copy-on-write growth.

    :param rows_parquet: ``<out_dir>/rows/<split>.parquet``.
    :type rows_parquet: Path
    :param images_dir: ``<out_dir>/images``.
    :type images_dir: Path
    :param tasks: Keep only rows whose ``task`` is listed (one family per
        dataset lets the notebook weight families separately); None = all.
    :type tasks: Optional[Sequence[str]]
    :param check_files: Fail at construction if any referenced image is missing.
    :type check_files: bool
    :raises FileNotFoundError: When ``check_files`` finds missing images.
    """

    def __init__(self, rows_parquet: Path, images_dir: Path,
                 tasks: Optional[Sequence[str]] = None, check_files: bool = True) -> None:
        table = pq.read_table(rows_parquet)
        meta = json.loads(table.schema.metadata[META_KEY])
        if tasks is not None:
            table = table.filter(pc.is_in(table["task"], value_set=pa.array(list(tasks))))
        self.images_dir = Path(images_dir)
        self.columns: List[str] = meta["columns"]
        self.image_column: str = meta["image_column"]
        self._decoder = Image(mode=meta["image_mode"])
        self._table: pa.Table = table.combine_chunks()
        if check_files:
            missing = [s for s in pc.unique(table[SHA_COLUMN]).to_pylist()
                       if not self.image_path(s).exists()]
            if missing:
                raise FileNotFoundError(f"{len(missing)} images missing under {images_dir}, "
                                        f"e.g. {missing[0]}.jpg")

    def __len__(self) -> int:
        """Number of rows.

        :return: Row count.
        :rtype: int
        """
        return self._table.num_rows

    def image_path(self, sha1: str) -> Path:
        """Path of a stored image.

        :param sha1: Image content hash.
        :type sha1: str
        :return: ``images_dir/<sha1>.jpg``.
        :rtype: Path
        """
        return self.images_dir / f"{sha1}.jpg"

    def load_image(self, sha1: str) -> PILImage.Image:
        """Decode a stored image the way ``datasets.Image`` decodes embedded bytes.

        :param sha1: Image content hash.
        :type sha1: str
        :return: Loaded PIL image.
        :rtype: PILImage.Image
        """
        return self._decoder.decode_example(
            {"path": None, "bytes": self.image_path(sha1).read_bytes()})

    def __getitem__(self, idx: int) -> Dict:
        """One row, with the image decoded, keys in the original column order.

        :param idx: Row index.
        :type idx: int
        :return: Row dict identical to the source DatasetDict row.
        :rtype: Dict
        :raises IndexError: For an index outside ``[-len, len)``.
        """
        n = len(self)
        if not -n <= idx < n:
            raise IndexError(f"row {idx} out of range for {n} rows")
        row = self._table.slice(idx % n, 1).to_pylist()[0]
        return {c: self.load_image(row[SHA_COLUMN]) if c == self.image_column else row[c]
                for c in self.columns}


def main() -> None:
    """CLI: export a saved DatasetDict and print the manifest summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="save_to_disk DatasetDict directory")
    ap.add_argument("--out", type=Path, required=True, help="images-once output directory")
    ap.add_argument("--reencode-jpeg", action="store_true",
                    help=f"re-encode non-JPEG images as quality-{JPEG_QUALITY} RGB JPEGs "
                         "(default: reject them)")
    ap.add_argument("--workers", type=int, default=REENCODE_WORKERS,
                    help="threads re-encoding images (with --reencode-jpeg)")
    a = ap.parse_args()
    manifest = export_images_once(a.src, a.out, reencode_jpeg=a.reencode_jpeg, workers=a.workers)
    summary = {k: v for k, v in manifest.items() if k != "features"}
    logger.info("manifest: %s", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
