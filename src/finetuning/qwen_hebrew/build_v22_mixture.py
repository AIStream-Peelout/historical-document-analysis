# File name: build_v22_mixture.py
# Date: 9/24/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Build the v22 pilot training mixture as one images-once dataset directory.

Four sources, each in (or first exported to) the images-once layout of
:mod:`src.finetuning.qwen_hebrew.images_once`, are sampled to fixed shares of
a train budget and materialised as one directory that
:class:`~src.finetuning.qwen_hebrew.images_once.ImagesOnceDataset` loads::

    <out>/rows/train.parquet   sampled rows of every source, shuffled, + "source"
    <out>/rows/val.parquet     fixed-size draws from each source's val split
    <out>/images/<sha1>.jpg    only the images those rows reference
    <out>/manifest.json        rows / unique images / bytes per split
    <out>/mixture.json         config, per-component pool/quota/taken/passes, shares
    <out>/stats.json           counts by component, bucket and task
    <out>/README.md            dataset card

Mixture components (the ``source`` column) and their buckets:

- ``ktiv_transcription`` / ``ktiv_grounding``: KTIV v4 rows split by task
  family (:data:`KTIV_TASK_BUCKETS`; a grounding row names a ``bbox_2d`` in
  its question or answer, which the build checks on every row),
- ``pgp_editions`` (transcription), ``documentary_grounding`` (grounding),
  ``pgp_qa`` (qa): every ``train`` / ``train_*`` split of the dataset.

A component's quota is ``round(share * train_rows)``. Rows are drawn without
replacement while the pool lasts; a quota larger than its pool takes whole
passes over the pool plus one partial pass, so every row appears
``floor(q/n)`` or ``ceil(q/n)`` times. The union is shuffled with the seed,
so any window of the map-style dataset carries the mixture proportions.

Usage (repo root):
    .venv/bin/python -m src.finetuning.qwen_hebrew.build_v22_mixture \\
        --out /Volumes/home/studio_offload/datasets/genizah_v22_pilot
"""
import argparse
import json
import logging
import os
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.finetuning.qwen_hebrew.images_once import (
    META_KEY, SHA_COLUMN, ImagesOnceDataset, export_images_once, saved_splits)

logger = logging.getLogger(__name__)

DATASETS_ROOT = Path("/Volumes/home/studio_offload/datasets")
DEFAULT_OUT = DATASETS_ROOT / "genizah_v22_pilot"
DEFAULT_KTIV_DIR = DATASETS_ROOT / "genizah_ktiv_v4_images_once"
DEFAULT_EDITIONS_DIR = DATASETS_ROOT / "pgp_editions_v1"
DEFAULT_GROUNDING_DIR = DATASETS_ROOT / "documentary_grounding_v1"
DEFAULT_QA_DIR = DATASETS_ROOT / "pgp_qa_v1"
DEFAULT_TRAIN_ROWS = 8000
DEFAULT_SEED = 3407
DEFAULT_SHARES: Dict[str, float] = {
    "ktiv_transcription": 0.30, "pgp_editions": 0.25, "ktiv_grounding": 0.15,
    "documentary_grounding": 0.10, "pgp_qa": 0.20}
DEFAULT_VAL_ROWS: Dict[str, int] = {
    "ktiv": 60, "pgp_editions": 60, "documentary_grounding": 40, "pgp_qa": 40}

SOURCE_COLUMN = "source"
VAL_SPLIT = "val"
BOX_MARKER = "bbox_2d"
FORBIDDEN_CARD_STRINGS = ("friedberg", "fjms", "fjp")

# mixture component -> (dataset, bucket); a KTIV component keeps its bucket's tasks only
COMPONENTS: Dict[str, Tuple[str, str]] = {
    "ktiv_transcription": ("ktiv", "transcription"),
    "ktiv_grounding": ("ktiv", "grounding"),
    "pgp_editions": ("pgp_editions", "transcription"),
    "documentary_grounding": ("documentary_grounding", "grounding"),
    "pgp_qa": ("pgp_qa", "qa"),
}

# every KTIV v4 task family; grounding = the answer is bbox JSON or the
# question asks about a given box (read_box*)
KTIV_TASK_BUCKETS: Dict[str, str] = {
    "fragment_transcribe": "transcription",
    "page_short": "transcription",
    "region_transcribe": "transcription",
    "section_transcribe": "transcription",
    "line_transcribe": "transcription",
    "layout_qa": "transcription",       # text answers (line text, column count)
    "grounded_crop": "grounding",
    "grounded_detect": "grounding",
    "grounded_page": "grounding",
    "line_index": "grounding",
    "line_of_phrase": "grounding",
    "locate": "grounding",
    "locate_word": "grounding",
    "read_box": "grounding",
    "read_box_word": "grounding",
}


@dataclass
class SourceRows:
    """Rows of one images-once source dataset.

    :param train: Concatenated rows of the ``train`` / ``train_*`` splits.
    :type train: pa.Table
    :param val: Rows of the val split.
    :type val: pa.Table
    :param meta: The source's images-once parquet metadata.
    :type meta: Dict
    :param train_splits: Names of the concatenated train splits.
    :type train_splits: List[str]
    :param images_dir: The source's ``images`` directory.
    :type images_dir: Path
    """
    train: pa.Table
    val: pa.Table
    meta: Dict
    train_splits: List[str]
    images_dir: Path


def export_dir_for(dataset_dir: Path) -> Path:
    """Images-once export directory of a saved DatasetDict (a sibling).

    :param dataset_dir: ``save_to_disk`` directory.
    :type dataset_dir: Path
    :return: ``<parent>/<name>_images_once``.
    :rtype: Path
    """
    return dataset_dir.parent / f"{dataset_dir.name}_images_once"


def export_is_current(dataset_dir: Path, export_dir: Path) -> bool:
    """Whether ``export_dir`` holds a finished export of the dataset's current splits.

    The manifest is written last, so its presence marks a finished export; it
    must name this source and exactly its splits, be newer than every saved
    split, and every split's rows parquet must hold the manifest's row count.

    :param dataset_dir: Saved DatasetDict.
    :type dataset_dir: Path
    :param export_dir: Candidate images-once export.
    :type export_dir: Path
    :return: True when the export can be reused as is.
    :rtype: bool
    """
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    splits = saved_splits(dataset_dir)
    if Path(manifest["source"]).resolve() != dataset_dir.resolve() or list(manifest["splits"]) != splits:
        return False
    newest = max((dataset_dir / s / "state.json").stat().st_mtime for s in splits)
    if manifest_path.stat().st_mtime < newest:
        return False
    for split in splits:
        rows = export_dir / "rows" / f"{split}.parquet"
        if not rows.exists() or pq.read_metadata(rows).num_rows != manifest["splits"][split]["rows"]:
            return False
    return True


def ensure_export(dataset_dir: Path, export_dir: Path) -> Path:
    """Export a saved DatasetDict to images-once unless a current export exists.

    :param dataset_dir: Saved DatasetDict.
    :type dataset_dir: Path
    :param export_dir: Images-once destination.
    :type export_dir: Path
    :return: ``export_dir``.
    :rtype: Path
    """
    if export_is_current(dataset_dir, export_dir):
        logger.info("%s: current images-once export found, skipping", export_dir)
    else:
        logger.info("exporting %s -> %s", dataset_dir, export_dir)
        export_images_once(dataset_dir, export_dir)
    return export_dir


def export_splits(export_dir: Path) -> List[str]:
    """Train split names of an export: ``train`` and every ``train_*``, manifest order.

    :param export_dir: Images-once export.
    :type export_dir: Path
    :return: Train split names.
    :rtype: List[str]
    :raises ValueError: When the export has no train split or no val split.
    """
    names = list(json.loads((export_dir / "manifest.json").read_text())["splits"])
    train = [s for s in names if s == "train" or s.startswith("train_")]
    if not train or VAL_SPLIT not in names:
        raise ValueError(f"{export_dir} needs train* and {VAL_SPLIT!r} splits, has {names}")
    return train


def check_layouts(metas: Sequence[Dict]) -> None:
    """Require one row layout (columns, image column, image mode) across metadata.

    :param metas: images-once parquet metadata dicts.
    :type metas: Sequence[Dict]
    :raises ValueError: On the first disagreement.
    """
    for meta in metas[1:]:
        for key in ("columns", "image_column", "image_mode"):
            if meta[key] != metas[0][key]:
                raise ValueError(f"row layouts differ on {key}: {meta[key]!r} vs {metas[0][key]!r}")


def read_rows(export_dir: Path, splits: Sequence[str]) -> Tuple[pa.Table, Dict]:
    """Concatenate the rows parquets of some splits of one export.

    :param export_dir: Images-once export.
    :type export_dir: Path
    :param splits: Split names, concatenated in this order.
    :type splits: Sequence[str]
    :return: (rows without schema metadata, the first split's images-once metadata).
    :rtype: Tuple[pa.Table, Dict]
    """
    tables, metas = [], []
    for split in splits:
        table = pq.read_table(export_dir / "rows" / f"{split}.parquet")
        metas.append(json.loads(table.schema.metadata[META_KEY]))
        tables.append(table.replace_schema_metadata(None))
    check_layouts(metas)
    return pa.concat_tables(tables), metas[0]


def load_source(export_dir: Path) -> SourceRows:
    """Train and val rows of one images-once export.

    :param export_dir: Images-once export.
    :type export_dir: Path
    :return: The source's rows.
    :rtype: SourceRows
    """
    train_splits = export_splits(export_dir)
    train, meta = read_rows(export_dir, train_splits)
    val, val_meta = read_rows(export_dir, [VAL_SPLIT])
    check_layouts([meta, val_meta])
    return SourceRows(train, val, meta, train_splits, export_dir / "images")


def classify_task(task: str, buckets: Mapping[str, str] = KTIV_TASK_BUCKETS) -> str:
    """Bucket of a KTIV task family.

    :param task: Row ``task`` value.
    :type task: str
    :param buckets: task -> bucket table.
    :type buckets: Mapping[str, str]
    :return: ``"transcription"`` or ``"grounding"``.
    :rtype: str
    :raises ValueError: For a task the table does not list.
    """
    if task not in buckets:
        raise ValueError(f"unclassified KTIV task {task!r}: add it to KTIV_TASK_BUCKETS")
    return buckets[task]


def ktiv_task_report(table: pa.Table, buckets: Mapping[str, str] = KTIV_TASK_BUCKETS) -> Dict[str, Dict]:
    """Classify every KTIV task in a table and check the rule on every row.

    A grounding row names a box (``bbox_2d``) in its question or answer; a
    transcription row never does.

    :param table: KTIV rows (``task``, ``section``, ``question``, ``answer``).
    :type table: pa.Table
    :param buckets: task -> bucket table.
    :type buckets: Mapping[str, str]
    :return: task -> {"bucket", "rows", "box_rows", "sections"}, sorted by task.
    :rtype: Dict[str, Dict]
    :raises ValueError: For an unclassified task, or a task whose rows
        contradict its bucket.
    """
    boxed = pc.or_(pc.match_substring(table["question"], BOX_MARKER),
                   pc.match_substring(table["answer"], BOX_MARKER)).to_pylist()
    report: Dict[str, Dict] = {}
    for task, section, box in zip(table["task"].to_pylist(), table["section"].to_pylist(), boxed):
        entry = report.setdefault(task, {"bucket": classify_task(task, buckets), "rows": 0,
                                         "box_rows": 0, "sections": set()})
        entry["rows"] += 1
        entry["box_rows"] += int(box)
        entry["sections"].add(section)
    for task, entry in report.items():
        if entry["box_rows"] != (entry["rows"] if entry["bucket"] == "grounding" else 0):
            raise ValueError(f"KTIV task {task!r} is {entry['bucket']} but {entry['box_rows']}"
                             f"/{entry['rows']} rows name a {BOX_MARKER}")
        entry["sections"] = sorted(entry["sections"])
    return dict(sorted(report.items()))


def row_components(dataset: str, table: pa.Table) -> List[str]:
    """Mixture component of every row of one dataset (KTIV rows by task bucket).

    :param dataset: Dataset key (``ktiv``, ``pgp_editions``, ...).
    :type dataset: str
    :param table: Rows of that dataset.
    :type table: pa.Table
    :return: Component name per row.
    :rtype: List[str]
    """
    by_bucket = {bucket: comp for comp, (d, bucket) in COMPONENTS.items() if d == dataset}
    if dataset != "ktiv":
        (comp,) = by_bucket.values()
        return [comp] * table.num_rows
    return [by_bucket[classify_task(task)] for task in table["task"].to_pylist()]


def component_pools(train_tables: Mapping[str, pa.Table], components: Iterable[str]) -> Dict[str, pa.Table]:
    """Train pool of each component: its dataset's rows, KTIV cut to the bucket.

    :param train_tables: dataset -> train rows.
    :type train_tables: Mapping[str, pa.Table]
    :param components: Components to build pools for.
    :type components: Iterable[str]
    :return: component -> pool rows.
    :rtype: Dict[str, pa.Table]
    """
    pools = {}
    for comp in components:
        dataset = COMPONENTS[comp][0]
        table = train_tables[dataset]
        keep = pa.array([c == comp for c in row_components(dataset, table)], pa.bool_())
        pools[comp] = table.filter(keep)
    return pools


def plan_quotas(shares: Mapping[str, float], train_rows: int) -> Dict[str, int]:
    """Per-component train quota ``round(share * train_rows)``.

    :param shares: component -> share of the train rows.
    :type shares: Mapping[str, float]
    :param train_rows: Train budget.
    :type train_rows: int
    :return: component -> quota, in ``shares`` order.
    :rtype: Dict[str, int]
    :raises ValueError: For an unknown component, a negative share, or
        shares that do not sum to 1.
    """
    unknown = sorted(set(shares) - set(COMPONENTS))
    if unknown:
        raise ValueError(f"unknown mixture components {unknown}; known: {sorted(COMPONENTS)}")
    if any(share < 0 for share in shares.values()):
        raise ValueError(f"negative share in {dict(shares)}")
    if abs(sum(shares.values()) - 1.0) > 1e-6:
        raise ValueError(f"shares sum to {sum(shares.values())}, not 1")
    return {comp: int(round(share * train_rows)) for comp, share in shares.items()}


def component_rng(seed: int, name: str) -> np.random.Generator:
    """Reproducible generator for one named draw, independent of the other draws.

    :param seed: Build seed.
    :type seed: int
    :param name: Draw name (component, ``val/<dataset>``, ...).
    :type name: str
    :return: Generator seeded by (seed, crc32(name)).
    :rtype: np.random.Generator
    """
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


def sample_source(pool_rows: int, quota: int, rng: np.random.Generator) -> Tuple[np.ndarray, Dict]:
    """Row indices filling a quota: whole passes over the pool, then one partial pass.

    The partial pass draws without replacement, so each row is taken
    ``floor(quota/pool_rows)`` or ``ceil(quota/pool_rows)`` times.

    :param pool_rows: Pool size.
    :type pool_rows: int
    :param quota: Rows wanted.
    :type quota: int
    :param rng: Generator for the partial pass.
    :type rng: np.random.Generator
    :return: (int64 row indices, {"pool", "quota", "taken", "unique_rows",
        "full_passes", "partial_rows", "passes"}).
    :rtype: Tuple[np.ndarray, Dict]
    :raises ValueError: When a positive quota meets an empty pool.
    """
    if quota > 0 and pool_rows == 0:
        raise ValueError(f"cannot take {quota} rows from an empty pool")
    full, part = divmod(quota, pool_rows) if pool_rows else (0, 0)
    idx = np.concatenate([np.tile(np.arange(pool_rows, dtype=np.int64), full),
                          rng.choice(pool_rows, size=part, replace=False).astype(np.int64)])
    info = {"pool": pool_rows, "quota": quota, "taken": int(len(idx)),
            "unique_rows": min(quota, pool_rows), "full_passes": full, "partial_rows": part,
            "passes": round(quota / pool_rows, 4) if pool_rows else 0.0}
    return idx, info


def with_source(table: pa.Table, sources: Sequence[str]) -> pa.Table:
    """Append the ``source`` column.

    :param table: Rows.
    :type table: pa.Table
    :param sources: Component name per row.
    :type sources: Sequence[str]
    :return: The rows with :data:`SOURCE_COLUMN` last.
    :rtype: pa.Table
    """
    return table.append_column(SOURCE_COLUMN, pa.array(list(sources), pa.string()))


def materialize(pools: Mapping[str, pa.Table], quotas: Mapping[str, int],
                seed: int) -> Tuple[pa.Table, Dict[str, Dict]]:
    """Sample every component to its quota, tag the rows, shuffle the union with the seed.

    :param pools: component -> pool rows (one shared schema).
    :type pools: Mapping[str, pa.Table]
    :param quotas: component -> quota.
    :type quotas: Mapping[str, int]
    :param seed: Build seed.
    :type seed: int
    :return: (shuffled train rows with ``source``, component -> sampling info).
    :rtype: Tuple[pa.Table, Dict[str, Dict]]
    """
    parts, infos = [], {}
    for comp, quota in quotas.items():
        idx, infos[comp] = sample_source(pools[comp].num_rows, quota, component_rng(seed, comp))
        parts.append(with_source(pools[comp].take(pa.array(idx)), [comp] * len(idx)))
    table = pa.concat_tables(parts)
    return table.take(pa.array(np.random.default_rng(seed).permutation(table.num_rows))), infos


def sample_val(val_tables: Mapping[str, pa.Table], val_rows: Mapping[str, int],
               seed: int) -> Tuple[pa.Table, Dict[str, Dict]]:
    """Draw up to the requested rows (no repeats) from each dataset's val split.

    :param val_tables: dataset -> val rows (one shared schema).
    :type val_tables: Mapping[str, pa.Table]
    :param val_rows: dataset -> rows wanted.
    :type val_rows: Mapping[str, int]
    :param seed: Build seed.
    :type seed: int
    :return: (shuffled val rows with ``source``, dataset -> {"pool", "requested", "taken"}).
    :rtype: Tuple[pa.Table, Dict[str, Dict]]
    :raises ValueError: For a dataset without val rows loaded.
    """
    unknown = sorted(set(val_rows) - set(val_tables))
    if unknown:
        raise ValueError(f"val rows asked of unknown datasets {unknown}; known: {sorted(val_tables)}")
    parts, infos = [], {}
    for dataset, wanted in val_rows.items():
        table = val_tables[dataset]
        taken = min(wanted, table.num_rows)
        idx = np.sort(component_rng(seed, f"val/{dataset}").choice(table.num_rows, size=taken,
                                                                   replace=False))
        part = table.take(pa.array(idx, pa.int64()))
        parts.append(with_source(part, row_components(dataset, part)))
        infos[dataset] = {"pool": table.num_rows, "requested": wanted, "taken": taken}
    table = pa.concat_tables(parts)
    order = component_rng(seed, "val/shuffle").permutation(table.num_rows)
    return table.take(pa.array(order)), infos


def mixture_meta(metas: Sequence[Dict]) -> Dict:
    """images-once parquet metadata of the mixture: the sources' layout plus ``source``.

    :param metas: Every source's images-once metadata.
    :type metas: Sequence[Dict]
    :return: Metadata whose ``columns`` end with :data:`SOURCE_COLUMN`, so
        :class:`ImagesOnceDataset` items carry it.
    :rtype: Dict
    """
    check_layouts(metas)
    features = {**metas[0]["features"], SOURCE_COLUMN: {"dtype": "string", "_type": "Value"}}
    return {**metas[0], "columns": metas[0]["columns"] + [SOURCE_COLUMN], "features": features}


def write_rows(table: pa.Table, meta: Dict, path: Path) -> None:
    """Write a rows parquet with its images-once metadata, atomically.

    :param table: Rows (``image_sha1`` + the other columns + ``source``).
    :type table: pa.Table
    :param meta: images-once metadata.
    :type meta: Dict
    :param path: ``<out>/rows/<split>.parquet``.
    :type path: Path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table.replace_schema_metadata(
        {META_KEY: json.dumps(meta, ensure_ascii=False).encode()}), tmp)
    os.replace(tmp, path)


def referenced_images(tables: Iterable[pa.Table], image_dirs: Mapping[str, Path]) -> Dict[str, Path]:
    """Source file of every image the rows reference.

    :param tables: Rows with ``image_sha1`` and ``source``.
    :type tables: Iterable[pa.Table]
    :param image_dirs: component -> the source export's ``images`` directory.
    :type image_dirs: Mapping[str, Path]
    :return: sha1 -> ``<images dir>/<sha1>.jpg`` (same sha1 = same bytes, so
        the first source wins).
    :rtype: Dict[str, Path]
    """
    wanted: Dict[str, Path] = {}
    for table in tables:
        for sha, comp in zip(table[SHA_COLUMN].to_pylist(), table[SOURCE_COLUMN].to_pylist()):
            wanted.setdefault(sha, image_dirs[comp] / f"{sha}.jpg")
    return wanted


def copy_one(item: Tuple[str, Path], images_dir: Path, present: set) -> Tuple[str, int]:
    """Copy one image into ``images_dir/<sha1>.jpg`` unless a complete copy is there.

    Written in place, not through a temporary name: renames raced on the SMB
    mount under parallel copies (the fresh temporary file was reported
    missing, 2026-09-24). A copy cut short by an interrupted run has the
    wrong size, so an already-present file is kept only when its size
    matches the source's.

    :param item: (sha1, source file).
    :type item: Tuple[str, Path]
    :param images_dir: ``<out>/images``.
    :type images_dir: Path
    :param present: sha1s listed in ``images_dir`` before copying.
    :type present: set
    :return: (sha1, byte size).
    :rtype: Tuple[str, int]
    """
    sha, src = item
    dst = images_dir / f"{sha}.jpg"
    if sha in present:
        size = src.stat().st_size
        if dst.stat().st_size == size:
            return sha, size
    data = src.read_bytes()
    dst.write_bytes(data)
    return sha, len(data)


def copy_images(wanted: Mapping[str, Path], images_dir: Path, workers: int = 16) -> Dict[str, int]:
    """Copy the referenced images (only those) into the mixture's flat image directory.

    :param wanted: sha1 -> source file.
    :type wanted: Mapping[str, Path]
    :param images_dir: ``<out>/images`` (created).
    :type images_dir: Path
    :param workers: Parallel copies (NAS latency bound).
    :type workers: int
    :return: sha1 -> byte size of every referenced image.
    :rtype: Dict[str, int]
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    present = {p.name[:-len(".jpg")] for p in images_dir.iterdir() if p.name.endswith(".jpg")}
    stray = present - set(wanted)
    if stray:
        logger.warning("%d images in %s are not referenced by this mixture", len(stray), images_dir)
    logger.info("copying %d images (%d already present) with %d workers",
                len(wanted), len(present & set(wanted)), workers)
    with ThreadPoolExecutor(workers) as pool:
        return dict(pool.map(partial(copy_one, images_dir=images_dir, present=present),
                             sorted(wanted.items())))


def verify_images(tables: Iterable[pa.Table], images_dir: Path) -> int:
    """Check that every image the rows reference is stored.

    :param tables: Rows with ``image_sha1``.
    :type tables: Iterable[pa.Table]
    :param images_dir: ``<out>/images``.
    :type images_dir: Path
    :return: Number of distinct referenced images.
    :rtype: int
    :raises FileNotFoundError: When any is missing.
    """
    on_disk = {p.name[:-len(".jpg")] for p in images_dir.iterdir() if p.name.endswith(".jpg")}
    wanted = set()
    for table in tables:
        wanted.update(table[SHA_COLUMN].to_pylist())
    missing = sorted(wanted - on_disk)
    if missing:
        raise FileNotFoundError(f"{len(missing)} referenced images missing under {images_dir}, "
                                f"e.g. {missing[0]}.jpg")
    return len(wanted)


def split_manifest(table: pa.Table, sizes: Mapping[str, int]) -> Dict[str, int]:
    """Manifest entry of one split (same keys as the images-once export's).

    :param table: Split rows.
    :type table: pa.Table
    :param sizes: sha1 -> byte size.
    :type sizes: Mapping[str, int]
    :return: {"rows", "unique_images", "row_image_bytes", "image_bytes"}.
    :rtype: Dict[str, int]
    """
    shas = table[SHA_COLUMN].to_pylist()
    return {"rows": table.num_rows, "unique_images": len(set(shas)),
            "row_image_bytes": sum(sizes[s] for s in shas),
            "image_bytes": sum(sizes[s] for s in set(shas))}


def mixture_stats(tables: Mapping[str, pa.Table]) -> Dict[str, Dict]:
    """Row counts by component, bucket and task, and mean target length, per split.

    :param tables: split -> rows with ``source``.
    :type tables: Mapping[str, pa.Table]
    :return: split -> statistics.
    :rtype: Dict[str, Dict]
    """
    stats = {}
    for split, table in tables.items():
        sources = table[SOURCE_COLUMN].to_pylist()
        tasks = table["task"].to_pylist()
        chars = table["target_chars"].to_pylist()
        by_source = Counter(sources)
        char_sums = Counter()
        for source, n_chars in zip(sources, chars):
            char_sums[source] += n_chars
        stats[split] = {
            "rows": table.num_rows,
            "by_source": dict(sorted(by_source.items())),
            "by_bucket": dict(sorted(Counter(COMPONENTS[s][1] for s in sources).items())),
            "by_source_task": dict(sorted(Counter(f"{s}/{t}" for s, t in zip(sources, tasks)).items())),
            "mean_target_chars": {s: round(char_sums[s] / n, 1) for s, n in sorted(by_source.items())},
        }
    return stats


def card_text(mixture: Dict, manifest: Dict) -> str:
    """Markdown dataset card of a built mixture.

    :param mixture: ``mixture.json`` content.
    :type mixture: Dict
    :param manifest: ``manifest.json`` content.
    :type manifest: Dict
    :return: README.md text.
    :rtype: str
    """
    cfg, comps = mixture["config"], mixture["components"]
    train, val = manifest["splits"]["train"], manifest["splits"]["val"]
    comp_rows = "\n".join(
        f"| `{c}` | {i['bucket']} | {i['dataset']} | {i['pool']:,} | {i['share']:.2f} | "
        f"{i['taken']:,} | {mixture['realised_shares']['components'][c]:.3f} | {i['passes']} |"
        for c, i in comps.items())
    val_rows = "\n".join(f"| {d} | {i['pool']:,} | {i['taken']} |" for d, i in mixture["val"].items())
    bucket_rows = "\n".join(f"| {b} | {n:,} | {mixture['realised_shares']['buckets'][b]:.3f} |"
                            for b, n in mixture["buckets"]["train"].items())
    ktiv_rows = "\n".join(f"| `{t}` | {e['bucket']} | {', '.join(e['sections'])} |"
                          for t, e in mixture["ktiv_task_buckets"].items())
    source_dirs = "\n".join(f"- {d}: `{p}`" for d, p in cfg["dirs"].items())
    return f"""# Genizah v22 pilot mixture (images-once)

Training mixture for the v22 pilot fine-tune of the Hebrew-manuscript VLM:
{train['rows']:,} train rows and {val['rows']:,} val rows sampled from four sources at
fixed shares (seed {cfg['seed']}), stored in the images-once layout: every distinct
image once, named by the SHA-1 of its JPEG bytes.

## Structure

```
rows/train.parquet   {train['rows']:,} rows, shuffled; {train['unique_images']:,} distinct images
rows/val.parquet     {val['rows']:,} rows; {val['unique_images']:,} distinct images
images/<sha1>.jpg    {manifest['n_images']:,} images, {manifest['image_bytes'] / 1e9:.2f} GB (only images the rows reference)
manifest.json        rows / unique images / bytes per split
mixture.json         config, per-component pool / quota / taken / passes, realised shares
stats.json           counts by component, bucket and task
```

Load a split as a map-style dataset (items are dicts with the image decoded):

```python
from src.finetuning.qwen_hebrew.images_once import ImagesOnceDataset
ds = ImagesOnceDataset(root / "rows/train.parquet", root / "images")
```

## Columns

| column | meaning |
|---|---|
| `image` | page, crop or line image (stored as `image_sha1` in the parquet) |
| `question` | prompt |
| `answer` | target (text, or JSON for grounding / QA rows) |
| `task`, `section` | task family and variant |
| `stem` | row id in its source dataset |
| `label_source` | where the target comes from |
| `target_chars`, `target_tokens` | target length |
| `image_width`, `image_height` | image size in pixels |
| `source` | mixture component (below) |

## Components and shares

Quota = round(share x {cfg['train_rows']:,}). Rows are drawn without replacement while
the pool lasts; a quota above its pool takes whole passes plus one partial pass
(each row appears floor or ceil of quota/pool times).

| component | bucket | dataset | pool | share | taken | realised | passes |
|---|---|---|---|---|---|---|---|
{comp_rows}

| bucket | train rows | realised share |
|---|---|---|
{bucket_rows}

Val rows (drawn without replacement from each dataset's val split):

| dataset | val pool | taken |
|---|---|---|
{val_rows}

KTIV task families by bucket (grounding = the answer is bbox JSON or the
question reads a given box):

| task | bucket | sections |
|---|---|---|
{ktiv_rows}

Source datasets:

{source_dirs}

## Credits

KTIV manuscript images and transcriptions: the National Library of Israel's
KTIV project (NLI-KTIV). Editions, document metadata and the image links behind
the edition, documentary-grounding and QA rows: the Princeton Geniza Project.
Documentary-grounding boxes are model-derived (lines where two independent
readers agreed). Images remain under the terms of their holding institutions;
this directory is internal training data, not for redistribution.

Built {mixture['built_at']} by `src/finetuning/qwen_hebrew/build_v22_mixture.py`.
"""


def write_card(out_dir: Path, mixture: Dict, manifest: Dict) -> Path:
    """Write ``README.md``, refusing sources that must not be credited.

    :param out_dir: Mixture directory.
    :type out_dir: Path
    :param mixture: ``mixture.json`` content.
    :type mixture: Dict
    :param manifest: ``manifest.json`` content.
    :type manifest: Dict
    :return: The card's path.
    :rtype: Path
    :raises ValueError: When the card would name a forbidden source.
    """
    text = card_text(mixture, manifest)
    hits = [word for word in FORBIDDEN_CARD_STRINGS if word in text.lower()]
    if hits:
        raise ValueError(f"dataset card would name {hits}; those sources must not be credited")
    path = out_dir / "README.md"
    path.write_text(text)
    return path


def write_json(path: Path, data: Dict) -> None:
    """Write indented UTF-8 JSON.

    :param path: Destination.
    :type path: Path
    :param data: JSON-serialisable content.
    :type data: Dict
    """
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def build_mixture(out_dir: Path, export_dirs: Mapping[str, Path], train_rows: int = DEFAULT_TRAIN_ROWS,
                  seed: int = DEFAULT_SEED, shares: Mapping[str, float] = DEFAULT_SHARES,
                  val_rows: Mapping[str, int] = DEFAULT_VAL_ROWS, workers: int = 16) -> Dict[str, Dict]:
    """Sample, materialise and describe the mixture.

    Images are copied and verified before the rows parquets are written, so a
    fresh build's rows never point at images that are not there.

    :param out_dir: Destination directory.
    :type out_dir: Path
    :param export_dirs: dataset (``ktiv``, ``pgp_editions``,
        ``documentary_grounding``, ``pgp_qa``) -> its images-once export.
    :type export_dirs: Mapping[str, Path]
    :param train_rows: Train budget.
    :type train_rows: int
    :param seed: Build seed.
    :type seed: int
    :param shares: component -> share of the train budget.
    :type shares: Mapping[str, float]
    :param val_rows: dataset -> val rows.
    :type val_rows: Mapping[str, int]
    :param workers: Parallel image copies.
    :type workers: int
    :return: {"manifest", "mixture", "stats"} as written.
    :rtype: Dict[str, Dict]
    """
    t0 = time.time()
    quotas = plan_quotas(shares, train_rows)
    sources = {dataset: load_source(path) for dataset, path in export_dirs.items()}
    meta = mixture_meta([s.meta for s in sources.values()])
    schema = next(iter(sources.values())).train.schema
    ktiv_report = ktiv_task_report(sources["ktiv"].train)
    ktiv_task_report(sources["ktiv"].val)
    for task, entry in ktiv_report.items():
        logger.info("KTIV task %-20s -> %-13s (%d rows; %s)", task, entry["bucket"], entry["rows"],
                    ", ".join(entry["sections"]))
    pools = component_pools({d: s.train.cast(schema) for d, s in sources.items()}, quotas)
    train, infos = materialize(pools, quotas, seed)
    val, val_infos = sample_val({d: s.val.cast(schema) for d, s in sources.items()}, val_rows, seed)
    image_dirs = {comp: sources[d].images_dir for comp, (d, _) in COMPONENTS.items() if d in sources}
    sizes = copy_images(referenced_images([train, val], image_dirs), out_dir / "images", workers)
    verify_images([train, val], out_dir / "images")
    write_rows(train, meta, out_dir / "rows" / "train.parquet")
    write_rows(val, meta, out_dir / "rows" / "val.parquet")

    splits = {"train": split_manifest(train, sizes), "val": split_manifest(val, sizes)}
    manifest = {"source": {d: str(p) for d, p in export_dirs.items()}, "image_column": meta["image_column"],
                "sha_column": SHA_COLUMN, "splits": splits, "features": meta["features"],
                "n_rows": train.num_rows + val.num_rows, "n_images": len(sizes),
                "image_bytes": sum(sizes.values()),
                "row_image_bytes": sum(s["row_image_bytes"] for s in splits.values())}
    manifest["dedupe_ratio"] = round(manifest["row_image_bytes"] / max(1, manifest["image_bytes"]), 3)
    stats = mixture_stats({"train": train, "val": val})
    total = max(1, train.num_rows)
    mixture = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {"train_rows": train_rows, "seed": seed, "shares": dict(shares),
                   "val_rows": dict(val_rows), "dirs": {d: str(p) for d, p in export_dirs.items()}},
        "components": {c: {"dataset": COMPONENTS[c][0], "bucket": COMPONENTS[c][1],
                           "share": shares[c], **info} for c, info in infos.items()},
        "train_splits": {d: s.train_splits for d, s in sources.items()},
        "val": val_infos,
        "buckets": {"train": stats["train"]["by_bucket"], "val": stats["val"]["by_bucket"]},
        "realised_shares": {
            "components": {c: round(i["taken"] / total, 4) for c, i in infos.items()},
            "buckets": {b: round(n / total, 4) for b, n in stats["train"]["by_bucket"].items()}},
        "ktiv_task_buckets": ktiv_report,
    }
    write_card(out_dir, mixture, manifest)
    mixture["elapsed_s"] = round(time.time() - t0, 1)
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "mixture.json", mixture)
    write_json(out_dir / "stats.json", stats)
    return {"manifest": manifest, "mixture": mixture, "stats": stats}


def self_check(out_dir: Path) -> Dict[str, Dict]:
    """Load both splits through :class:`ImagesOnceDataset` (files checked) and decode one item.

    :param out_dir: Mixture directory.
    :type out_dir: Path
    :return: split -> {"rows", "keys", "image_size", "source"} of item 0.
    :rtype: Dict[str, Dict]
    """
    report = {}
    for split in ("train", "val"):
        ds = ImagesOnceDataset(out_dir / "rows" / f"{split}.parquet", out_dir / "images",
                               check_files=True)
        item = ds[0]
        report[split] = {"rows": len(ds), "keys": list(item), "image_size": list(item["image"].size),
                         "source": item[SOURCE_COLUMN]}
    return report


def main() -> None:
    """CLI: export the DatasetDict sources if needed, build the mixture, self-check it."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--train-rows", type=int, default=DEFAULT_TRAIN_ROWS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--shares", type=json.loads, default=DEFAULT_SHARES,
                    help="JSON component -> share (must sum to 1)")
    ap.add_argument("--val-rows", type=json.loads, default=DEFAULT_VAL_ROWS,
                    help="JSON dataset -> val rows")
    ap.add_argument("--ktiv-dir", type=Path, default=DEFAULT_KTIV_DIR, help="KTIV images-once export")
    ap.add_argument("--editions-dir", type=Path, default=DEFAULT_EDITIONS_DIR,
                    help="PGP editions DatasetDict (exported to <dir>_images_once)")
    ap.add_argument("--grounding-dir", type=Path, default=DEFAULT_GROUNDING_DIR,
                    help="documentary grounding DatasetDict (exported to <dir>_images_once)")
    ap.add_argument("--qa-dir", type=Path, default=DEFAULT_QA_DIR,
                    help="PGP QA DatasetDict (exported to <dir>_images_once)")
    ap.add_argument("--workers", type=int, default=16, help="parallel image copies")
    a = ap.parse_args()
    export_dirs = {"ktiv": a.ktiv_dir}
    for dataset, src in (("pgp_editions", a.editions_dir), ("documentary_grounding", a.grounding_dir),
                         ("pgp_qa", a.qa_dir)):
        export_dirs[dataset] = ensure_export(src, export_dir_for(src))
    result = build_mixture(a.out, export_dirs, a.train_rows, a.seed, a.shares, a.val_rows, a.workers)
    mixture, manifest = result["mixture"], result["manifest"]
    print("KTIV task -> bucket:")
    for task, entry in mixture["ktiv_task_buckets"].items():
        print(f"  {task:20s} {entry['bucket']:13s} {entry['rows']:6d} rows  {', '.join(entry['sections'])}")
    print("components:")
    for comp, info in mixture["components"].items():
        print(f"  {comp:22s} pool {info['pool']:6d}  quota {info['quota']:5d}  taken {info['taken']:5d}"
              f"  full_passes {info['full_passes']}  partial {info['partial_rows']:5d}"
              f"  passes {info['passes']}")
    print("val:", json.dumps(mixture["val"]))
    print("buckets:", json.dumps(mixture["buckets"]), "realised:", json.dumps(mixture["realised_shares"]))
    print("manifest:", json.dumps({k: v for k, v in manifest.items() if k != "features"}, indent=1))
    print(f"elapsed {mixture['elapsed_s']}s")
    for split, check in self_check(a.out).items():
        print(f"self-check {split}: {check}")


if __name__ == "__main__":
    main()
