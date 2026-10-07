# File name: build_v22_mixture.py
# Date: 9/24/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Build a v22 training mixture (pilot or full run) as one images-once dataset directory.

Source datasets, each in (or first exported to) the images-once layout of
:mod:`src.finetuning.qwen_hebrew.images_once`, are sampled to fixed shares of
a train budget and materialised as one directory that
:class:`~src.finetuning.qwen_hebrew.images_once.ImagesOnceDataset` loads::

    <out>/rows/train.parquet   sampled rows of every source, shuffled, + "source"
    <out>/rows/val.parquet     fixed-size draws from each source's val split,
                               never on an image a train row uses
    <out>/images/<sha1>.jpg    only the images those rows reference
    <out>/manifest.json        rows / unique images / bytes per split
    <out>/mixture.json         config, per-component pool/quota/taken/passes, shares
    <out>/stats.json           rows and unique images by component, bucket and task
    <out>/README.md            dataset card (credits and rights of every source used)

The four pilot datasets have their own flags: ``--ktiv-dir`` (an images-once
export, used as is) and ``--editions-dir`` / ``--grounding-dir`` /
``--qa-dir`` (saved DatasetDicts, exported to ``<dir>_images_once`` unless a
current export exists). Any other dataset of :data:`COMPONENTS` comes in with
``--source NAME=DIR`` (repeatable), an images-once export used as is.

Mixture components (the ``source`` column) and their buckets:

- ``ktiv_transcription`` / ``ktiv_grounding``: KTIV v4 rows split by task
  family (:data:`KTIV_TASK_BUCKETS`; a grounding row names a ``bbox_2d`` in
  its question or answer, which the build checks on every row),
- ``pgp_editions`` (transcription), ``documentary_grounding`` (grounding),
  ``pgp_qa`` (qa), and from extra sources ``talmud_replay`` (dataset
  ``talmud``) and ``synthetic`` (dataset ``synthetic``), both transcription:
  every ``train`` / ``train_*`` split of the dataset.

Without ``--shares`` the pilot shares (:data:`DEFAULT_SHARES`) apply; each
supplied extra dataset adds its components at :data:`EXTRA_DEFAULT_SHARES`
and the pilot shares scale down in proportion (:func:`default_shares`).
Explicit ``--shares`` must sum to 1 and may name any component whose dataset
is supplied. Default val rows (:data:`DEFAULT_VAL_ROWS`) apply to the
supplied datasets only.

A component's quota is ``round(share * train_rows)``. Rows are drawn without
replacement while the pool lasts; a quota larger than its pool takes whole
passes over the pool plus one partial pass, so every row appears
``floor(q/n)`` or ``ceil(q/n)`` times. The union is shuffled with the seed,
so any window of the map-style dataset carries the mixture proportions.

Val never shares a page image with train: the sources split their pages
independently (a PGP page can be val for the editions and train for the
documentary grounding), so a drawn val row whose ``image_sha1`` (the page
identity) backs any train row is dropped and replaced from the same source's
remaining val rows on images not in train; when those run out the source
keeps a smaller val count. ``manifest.json`` records the drops under
``val_dedupe``.

Usage (repo root):
    # pilot: the four pilot sources at the default shares
    .venv/bin/python -m src.finetuning.qwen_hebrew.build_v22_mixture \\
        --out /Volumes/home/studio_offload/datasets/genizah_v22_pilot
    # full run: + Talmud replay and synthetic renders, explicit shares
    .venv/bin/python -m src.finetuning.qwen_hebrew.build_v22_mixture \\
        --out /Volumes/home/studio_offload/datasets/genizah_v22_full --train-rows 20000 \\
        --source talmud=/Volumes/home/studio_offload/datasets/talmud_finetune_v2_images_once \\
        --source synthetic=/Volumes/home/studio_offload/datasets/synthetic_hebrew_v3_images_once \\
        --shares '{"ktiv_transcription": 0.28, "pgp_editions": 0.22, "ktiv_grounding": 0.13,
                   "documentary_grounding": 0.10, "pgp_qa": 0.12, "talmud_replay": 0.08,
                   "synthetic": 0.07}'
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
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
# the pilot's shares: the default when no extra source is supplied
DEFAULT_SHARES: Dict[str, float] = {
    "ktiv_transcription": 0.30, "pgp_editions": 0.25, "ktiv_grounding": 0.15,
    "documentary_grounding": 0.10, "pgp_qa": 0.20}
# default shares of extra-source components, added only when their dataset is supplied
EXTRA_DEFAULT_SHARES: Dict[str, float] = {"talmud_replay": 0.08, "synthetic": 0.07}
# dataset -> default val rows, applied to the supplied datasets only
DEFAULT_VAL_ROWS: Dict[str, int] = {
    "ktiv": 60, "pgp_editions": 60, "documentary_grounding": 40, "pgp_qa": 40,
    "talmud": 30, "synthetic": 20, "pgp_edition_pages": 60,
    "arabic_editions": 10, "agapet": 20, "muharaf": 20, "baybars": 15, "iskandar": 10, "vqa": 150}

SOURCE_COLUMN = "source"
VAL_SPLIT = "val"
BOX_MARKER = "bbox_2d"
VAL_DEDUPE_RULE = ("val rows whose image_sha1 (page identity) also backs a train row are dropped "
                   "and refilled from the same source's remaining val rows on images not in "
                   "train; a source keeps a smaller count when those run out")
FORBIDDEN_CARD_STRINGS = ("friedberg", "fjms", "fjp")
# characters the training notebooks' label-hygiene gate rejects in a transcription answer (the internal gap token,
# then the "mojibake" set): in an edition they are editor's marks ($...$, &, ___ line fill), not text on the page
UNCLEAN_TRANSCRIPTION_CHARS = "␣&#$_{}<>\\"
PARSE_TASKS = ("parse_lines", "parse_lines_boxes")     # JSON page parses (build_vqa_parse.py): [{"n", "text"[, "bbox_2d"]}, ...]

# mixture component -> (dataset, bucket); a KTIV component keeps its bucket's tasks only, a `vqa` component the rows
# of its task family (TASK_SPLIT_DATASETS); every other dataset feeds exactly one component
COMPONENTS: Dict[str, Tuple[str, str]] = {
    "ktiv_transcription": ("ktiv", "transcription"),
    "ktiv_grounding": ("ktiv", "grounding"),
    "pgp_editions": ("pgp_editions", "transcription"),
    "documentary_grounding": ("documentary_grounding", "grounding"),
    "pgp_qa": ("pgp_qa", "qa"),
    "talmud_replay": ("talmud", "transcription"),
    "synthetic": ("synthetic", "transcription"),
    # full documentary pages only (the `train_page` split of pgp_editions as its own export): Hebrew-script replay
    "pgp_edition_pages": ("pgp_edition_pages", "transcription"),
    # Arabic-script page transcription: Geniza editions, then outside handwriting sets (build_arabic_external.py)
    "arabic_editions": ("arabic_editions", "transcription"),
    "arabic_agapet": ("agapet", "transcription"),
    "arabic_muharaf": ("muharaf", "transcription"),
    "arabic_baybars": ("baybars", "transcription"),
    "arabic_iskandar": ("iskandar", "transcription"),
    # page parse and questions answered from the image plus a parse (build_vqa_parse.py): one export, one component
    # per task family (the bucket is the row's task)
    "vqa_parse_lines": ("vqa", "parse_lines"),
    "vqa_parse_boxes": ("vqa", "parse_lines_boxes"),
    "vqa_fields": ("vqa", "fields_from_parse"),
    "vqa_question": ("vqa", "question_from_parse"),
    "vqa_lookup": ("vqa", "lookup_from_parse"),
}
# datasets with a dedicated CLI flag; every other dataset of COMPONENTS comes in with --source
PILOT_FLAGS: Dict[str, str] = {"ktiv": "--ktiv-dir", "pgp_editions": "--editions-dir",
                               "documentary_grounding": "--grounding-dir", "pgp_qa": "--qa-dir"}
EXTRA_DATASETS: Tuple[str, ...] = tuple(dict.fromkeys(
    dataset for dataset, _ in COMPONENTS.values() if dataset not in PILOT_FLAGS))
# card notes of extra datasets, printed when the dataset contributes rows
RIGHTS_NOTES: Dict[str, str] = {
    "talmud": ("Talmud replay rows (`talmud_replay`): Vilna-layout Talmud page scans (whole pages or "
               "crops) from HebrewBooks.org, © Moznaim Publishers; the scanned pages carry the notice "
               "\"No commercial use allowed\". These scans are not public domain, which is why this dataset "
               "repository stays private: no public release, no redistribution, no commercial use."),
    "synthetic": ("Synthetic rows (`synthetic`): generated renders of printed Hebrew (scrambled words, "
                  "random strings and confusable-letter drills from the project's own renderer, "
                  "`synthetic_hebrew_v3.py`); no scanned or third-party images; CC-BY compatible."),
    "arabic_editions": ("Arabic-script Geniza rows (`arabic_editions`): page images from the holding libraries with "
                        "edition text from the Princeton Geniza Project (CC BY-NC 4.0)."),
    "agapet": ("Agapet rows (`arabic_agapet`): Christian Arabic manuscript pages (Sinai Arabic 418, Sinai Arabic "
               "423, BnF Arabe 76) with expert transcription, from Ibrahim and DiRusso, \"Agapet (Christian Arabic "
               "HTR Model) Training Datasets\", Zenodo 10.5281/zenodo.15473122, CC BY 4.0."),
    "muharaf": ("Muharaf rows (`arabic_muharaf`): Lebanese archival manuscripts, 19th to 21st century, from Saeed et "
                "al., \"Muharaf-public\", Zenodo 10.5281/zenodo.11492215, CC BY-NC-SA: non-commercial, and a "
                "published derivative must carry the same licence. This repository stays private."),
    "baybars": ("BAYBARS rows (`arabic_baybars`): lines of Sirat Baybars manuscripts stacked back into page images, "
                "from Calfa, dataset `calfa-ai/baybars`, Etalab Open Licence 2.0."),
    "iskandar": ("ISKANDAR rows (`arabic_iskandar`): lines of Sirat al-Iskandar manuscripts stacked back into page "
                 "images, from Calfa, dataset `calfa-ai/iskandar`, Etalab Open Licence 2.0."),
    "vqa": ("Page-parse rows (`vqa_*`): page images from the holding libraries with edition text and document metadata "
            "from the Princeton Geniza Project (CC BY-NC 4.0). Every target text is an edition line; a prompt may quote "
            "a machine reading of the page as context; line boxes come from automatic line segmentation."),
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


def source_flag(dataset: str) -> str:
    """The CLI argument that supplies a dataset.

    :param dataset: Dataset key.
    :type dataset: str
    :return: Its dedicated flag, else ``--source <dataset>=DIR``.
    :rtype: str
    """
    return PILOT_FLAGS.get(dataset, f"--source {dataset}=DIR")


def parse_sources(specs: Sequence[str]) -> Dict[str, Path]:
    """Extra images-once exports named by ``--source NAME=DIR`` arguments.

    :param specs: ``NAME=DIR`` strings; NAME is a dataset of :data:`COMPONENTS`
        that has no dedicated flag (:data:`EXTRA_DATASETS`).
    :type specs: Sequence[str]
    :return: dataset -> export directory, in argument order.
    :rtype: Dict[str, Path]
    :raises ValueError: For a malformed spec, a dataset that is unknown or has
        its own flag, or a dataset given twice.
    """
    sources: Dict[str, Path] = {}
    for spec in specs:
        name, sep, path = (part.strip() for part in spec.partition("="))
        if not (sep and name and path):
            raise ValueError(f"--source takes NAME=DIR, got {spec!r}")
        if name in PILOT_FLAGS:
            raise ValueError(f"--source {spec!r}: {name} has its own flag, {PILOT_FLAGS[name]}")
        if name not in EXTRA_DATASETS:
            known = ", ".join(f"{d} (component {', '.join(c for c, (cd, _) in COMPONENTS.items() if cd == d)})"
                              for d in EXTRA_DATASETS)
            raise ValueError(f"--source {spec!r}: unknown dataset {name!r}; --source datasets: {known}")
        if name in sources:
            raise ValueError(f"--source {name} given twice")
        sources[name] = Path(path).expanduser()
    return sources


def default_shares(datasets: Iterable[str]) -> Dict[str, float]:
    """Default component shares for a set of supplied datasets.

    With no extra dataset this is :data:`DEFAULT_SHARES` exactly (the pilot).
    Each supplied extra dataset adds its components at
    :data:`EXTRA_DEFAULT_SHARES`, and the pilot shares scale down in
    proportion so the total stays 1.

    :param datasets: Supplied dataset keys.
    :type datasets: Iterable[str]
    :return: component -> share, pilot components first.
    :rtype: Dict[str, float]
    """
    present = set(datasets)
    extra = {comp: share for comp, share in EXTRA_DEFAULT_SHARES.items() if COMPONENTS[comp][0] in present}
    if not extra:
        return dict(DEFAULT_SHARES)
    scale = 1.0 - sum(extra.values())
    return {**{comp: round(share * scale, 6) for comp, share in DEFAULT_SHARES.items()}, **extra}


def default_val_rows(datasets: Iterable[str]) -> Dict[str, int]:
    """Default val rows of the supplied datasets (:data:`DEFAULT_VAL_ROWS` order).

    :param datasets: Supplied dataset keys.
    :type datasets: Iterable[str]
    :return: dataset -> val rows.
    :rtype: Dict[str, int]
    """
    present = set(datasets)
    return {dataset: n for dataset, n in DEFAULT_VAL_ROWS.items() if dataset in present}


def val_dataset(key: str) -> str:
    """Dataset behind a val-rows key.

    :param key: A dataset key, or a component name (its val rows are that component's rows of its dataset).
    :type key: str
    :return: The dataset key.
    :rtype: str
    """
    return COMPONENTS[key][0] if key in COMPONENTS and key not in {d for d, _ in COMPONENTS.values()} else key


def check_sources(shares: Mapping[str, float], val_rows: Mapping[str, int], datasets: Iterable[str]) -> None:
    """Require a supplied source for every dataset the shares or val rows draw on.

    :param shares: component -> share (components already known).
    :type shares: Mapping[str, float]
    :param val_rows: dataset -> val rows.
    :type val_rows: Mapping[str, int]
    :param datasets: Supplied dataset keys.
    :type datasets: Iterable[str]
    :raises ValueError: Naming each component or val dataset without a
        source and the flag that would supply it.
    """
    present = set(datasets)
    missing = [f"share for {comp} needs dataset {COMPONENTS[comp][0]!r} ({source_flag(COMPONENTS[comp][0])})"
               for comp in shares if COMPONENTS[comp][0] not in present]
    missing += [f"val rows ask for dataset {val_dataset(key)!r} ({source_flag(val_dataset(key))})"
                for key in val_rows if val_dataset(key) not in present]
    if missing:
        raise ValueError(f"no source supplied: {'; '.join(missing)}; supplied: {sorted(present)}")


def resolve_plan(datasets: Sequence[str], train_rows: int, shares: Optional[Mapping[str, float]] = None,
                 val_rows: Optional[Mapping[str, int]] = None) -> Tuple[Dict[str, float], Dict[str, int],
                                                                        Dict[str, int]]:
    """Shares, val rows and quotas of a build, defaults filled in for the supplied datasets, all checked.

    Runs before any export or read, so a bad request fails in seconds.

    :param datasets: Supplied dataset keys.
    :type datasets: Sequence[str]
    :param train_rows: Train budget.
    :type train_rows: int
    :param shares: component -> share; None = :func:`default_shares`.
    :type shares: Optional[Mapping[str, float]]
    :param val_rows: dataset -> val rows; None = :func:`default_val_rows`.
    :type val_rows: Optional[Mapping[str, int]]
    :return: (shares, val rows, component -> quota).
    :rtype: Tuple[Dict[str, float], Dict[str, int], Dict[str, int]]
    :raises ValueError: From :func:`plan_quotas` or :func:`check_sources`.
    """
    shares = default_shares(datasets) if shares is None else dict(shares)
    val_rows = default_val_rows(datasets) if val_rows is None else dict(val_rows)
    quotas = plan_quotas(shares, train_rows)
    check_sources(shares, val_rows, datasets)
    return shares, val_rows, quotas


def plan_datasets(shares: Mapping[str, float], val_rows: Mapping[str, int]) -> set:
    """Datasets a plan draws rows from.

    :param shares: component -> share of the train rows.
    :type shares: Mapping[str, float]
    :param val_rows: dataset -> val rows.
    :type val_rows: Mapping[str, int]
    :return: Datasets with a positive share (through their components) or a positive val count.
    :rtype: set
    """
    return ({COMPONENTS[comp][0] for comp, share in shares.items() if share > 0}
            | {val_dataset(key) for key, rows in val_rows.items() if rows > 0})


def check_export(dataset: str, export_dir: Path) -> None:
    """Require a finished images-once export (its manifest is written last).

    :param dataset: Dataset key (for the message).
    :type dataset: str
    :param export_dir: Export directory.
    :type export_dir: Path
    :raises FileNotFoundError: When ``manifest.json`` is missing.
    """
    if not (export_dir / "manifest.json").is_file():
        raise FileNotFoundError(f"{dataset}: {export_dir} is not a finished images-once export "
                                "(no manifest.json, which the export writes last)")


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


def check_layouts(metas: Sequence[Dict], names: Optional[Sequence[str]] = None) -> None:
    """Require one row layout across metadata: column set, image column, image mode.

    Column order may differ between sources (an extra source may write its
    columns in another order); :func:`conform` aligns the rows to the first
    layout, whose order the mixture keeps.

    :param metas: images-once parquet metadata dicts.
    :type metas: Sequence[Dict]
    :param names: Label of each metadata dict for the message (dataset or
        split); None = positions.
    :type names: Optional[Sequence[str]]
    :raises ValueError: On the first disagreement, naming the layout that
        differs from the first.
    """
    names = list(names) if names is not None else [f"layout {i}" for i in range(len(metas))]
    ref, ref_name = metas[0], names[0]
    for meta, name in zip(metas[1:], names[1:]):
        if set(meta["columns"]) != set(ref["columns"]):
            lacks = sorted(set(ref["columns"]) - set(meta["columns"]))
            adds = sorted(set(meta["columns"]) - set(ref["columns"]))
            raise ValueError(f"row layouts differ on columns: {name} lacks {lacks} and adds {adds} "
                             f"compared with {ref_name}")
        for key in ("image_column", "image_mode"):
            if meta[key] != ref[key]:
                raise ValueError(f"row layouts differ on {key}: {name} has {meta[key]!r}, "
                                 f"{ref_name} has {ref[key]!r}")


def conform(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Rows in a reference layout: its column order and column types.

    :param table: Rows with the schema's column set (any order).
    :type table: pa.Table
    :param schema: Reference schema.
    :type schema: pa.Schema
    :return: The rows with ``schema``'s column order and types.
    :rtype: pa.Table
    """
    return table.select(schema.names).cast(schema)


def read_rows(export_dir: Path, splits: Sequence[str]) -> Tuple[pa.Table, Dict]:
    """Concatenate the rows parquets of some splits of one export.

    :param export_dir: Images-once export.
    :type export_dir: Path
    :param splits: Split names, concatenated in this order.
    :type splits: Sequence[str]
    :return: (rows without schema metadata in the first split's layout, the
        first split's images-once metadata).
    :rtype: Tuple[pa.Table, Dict]
    """
    tables, metas = [], []
    for split in splits:
        table = pq.read_table(export_dir / "rows" / f"{split}.parquet")
        metas.append(json.loads(table.schema.metadata[META_KEY]))
        tables.append(table.replace_schema_metadata(None))
    check_layouts(metas, [f"{export_dir.name}/{split}" for split in splits])
    return pa.concat_tables([conform(t, tables[0].schema) for t in tables]), metas[0]


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
    check_layouts([meta, val_meta], [f"{export_dir.name}/train", f"{export_dir.name}/{VAL_SPLIT}"])
    return SourceRows(train, val, meta, train_splits, export_dir / "images")


def unclean_answer(task: str, answer: str, chars: str = UNCLEAN_TRANSCRIPTION_CHARS) -> bool:
    """Whether a row's target holds editor's marks where page text is expected.

    :param task: Task family of the row.
    :type task: str
    :param answer: The row's target.
    :type answer: str
    :param chars: Characters page text may not hold.
    :type chars: str
    :return: True for a transcription answer (task ending in ``transcribe``) with one of ``chars``, and for a JSON
        page parse (:data:`PARSE_TASKS`) with one of them inside a line's text. Other tasks are never unclean.
    :rtype: bool
    """
    if task.endswith("transcribe"):
        return any(ch in answer for ch in chars)
    if task in PARSE_TASKS:
        return any(ch in line["text"] for line in json.loads(answer) for ch in chars)
    return False


def drop_unclean_transcriptions(table: pa.Table, chars: str = UNCLEAN_TRANSCRIPTION_CHARS) -> Tuple[pa.Table, List[str]]:
    """Drop the rows whose target text the notebooks' label-hygiene gate would reject.

    The gate looks at every row whose task ends in ``transcribe``; one such row with one of ``chars`` in
    its answer stops a launch at the data cell. A JSON page parse with such a character in a line goes
    too, together with every other row of the table on the same image: its questions quote that page.
    Other tasks (boxes, older question rows) are left alone.

    :param table: Rows of one dataset with ``task``, ``answer``, ``stem`` and the image hash.
    :type table: pa.Table
    :param chars: Characters page text may not hold.
    :type chars: str
    :return: (the rows kept, stems of the rows dropped).
    :rtype: Tuple[pa.Table, List[str]]
    """
    tasks, answers = table["task"].to_pylist(), table["answer"].to_pylist()
    unclean = [unclean_answer(task, answer, chars) for task, answer in zip(tasks, answers)]
    if not any(unclean):
        return table, []
    images = table[SHA_COLUMN].to_pylist() if SHA_COLUMN in table.column_names else [None] * table.num_rows
    parse_images = {image for image, task, bad in zip(images, tasks, unclean) if bad and task in PARSE_TASKS and image}
    keep = [not bad and image not in parse_images for bad, image in zip(unclean, images)]
    dropped = [stem for stem, kept in zip(table["stem"].to_pylist(), keep) if not kept]
    return table.filter(pa.array(keep, pa.bool_())), dropped


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


# datasets whose rows are split over several components: dataset -> (task -> bucket of COMPONENTS)
TASK_SPLIT_DATASETS: Dict[str, Callable[[str], str]] = {"ktiv": classify_task, "vqa": str}


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

    :param dataset: Dataset key (``ktiv``, ``pgp_editions``, ``talmud``, ...).
    :type dataset: str
    :param table: Rows of that dataset.
    :type table: pa.Table
    :return: Component name per row.
    :rtype: List[str]
    :raises ValueError: For a dataset without a component, or a non-KTIV
        dataset that :data:`COMPONENTS` maps to more than one component.
    """
    comps = [comp for comp, (d, _) in COMPONENTS.items() if d == dataset]
    if not comps:
        raise ValueError(f"unknown dataset {dataset!r}; known: {sorted({d for d, _ in COMPONENTS.values()})}")
    if dataset in TASK_SPLIT_DATASETS:
        by_bucket = {COMPONENTS[comp][1]: comp for comp in comps}
        buckets = [TASK_SPLIT_DATASETS[dataset](task) for task in table["task"].to_pylist()]
        unknown = sorted(set(buckets) - set(by_bucket))
        if unknown:
            raise ValueError(f"dataset {dataset!r} has rows of task buckets {unknown} without a component in COMPONENTS")
        return [by_bucket[bucket] for bucket in buckets]
    if len(comps) != 1:
        raise ValueError(f"dataset {dataset!r} maps to components {comps}; only {sorted(TASK_SPLIT_DATASETS)} split by "
                         "task, every other dataset needs exactly one entry in COMPONENTS")
    return comps * table.num_rows


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


def trained_component_rows(dataset: str, table: pa.Table, trained: Iterable[str]) -> pa.Table:
    """Rows of the components a plan trains: what a dataset's val rows are drawn from.

    A plan that trains KTIV transcription only is not validated on KTIV box rows. A dataset none of whose
    components is trained (supplied for val only) keeps all its rows.

    :param dataset: Dataset key.
    :type dataset: str
    :param table: Rows of that dataset.
    :type table: pa.Table
    :param trained: Components with a positive train quota.
    :type trained: Iterable[str]
    :return: The rows whose component is trained, or ``table`` itself when none is.
    :rtype: pa.Table
    """
    trained = set(trained)
    comps = row_components(dataset, table)
    if not trained & set(comps) or trained >= set(comps):
        return table
    return table.filter(pa.array([comp in trained for comp in comps], pa.bool_()))


def plan_quotas(shares: Mapping[str, float], train_rows: int) -> Dict[str, int]:
    """Per-component train quota ``round(share * train_rows)``.

    Any component of :data:`COMPONENTS` may be named; whether its dataset is
    supplied is :func:`check_sources`' job.

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


def refill_val_draw(drawn: np.ndarray, blocked: np.ndarray, wanted: int,
                    rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Drop drawn val rows on train images and refill from the pool's other allowed rows.

    The first draw's allowed rows plus a uniform refill from the allowed rows
    it missed form a uniform draw of the allowed rows; a draw that hits no
    blocked row stands unchanged.

    :param drawn: Pool row indices of the first draw (no repeats).
    :type drawn: np.ndarray
    :param blocked: Per pool row, True when its image also backs a train row.
    :type blocked: np.ndarray
    :param wanted: Rows requested.
    :type wanted: int
    :param rng: Generator for the refill.
    :type rng: np.random.Generator
    :return: (kept + refill indices, dropped indices), both sorted; fewer
        than ``wanted`` rows only when the allowed rows run out.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    kept, dropped = drawn[~blocked[drawn]], drawn[blocked[drawn]]
    spare = np.setdiff1d(np.flatnonzero(~blocked), drawn)
    refill = rng.choice(spare, size=min(wanted - len(kept), len(spare)), replace=False)
    return np.sort(np.concatenate([kept, refill]).astype(np.int64)), np.sort(dropped)


def sample_val(val_tables: Mapping[str, pa.Table], val_rows: Mapping[str, int], seed: int,
               train_images: frozenset = frozenset()) -> Tuple[pa.Table, Dict[str, Dict], Dict]:
    """Draw up to the requested rows (no repeats) from each dataset's val split, off train images.

    :param val_tables: dataset -> val rows (one shared schema).
    :type val_tables: Mapping[str, pa.Table]
    :param val_rows: dataset -> rows wanted.
    :type val_rows: Mapping[str, int]
    :param seed: Build seed.
    :type seed: int
    :param train_images: ``image_sha1`` of every train row; val rows on
        them are dropped and refilled (:func:`refill_val_draw`).
    :type train_images: frozenset
    :return: (shuffled val rows with ``source``, dataset -> {"pool",
        "eligible", "requested", "taken", "dropped", "refilled"},
        {"rule", "dropped_rows", "dropped_stems", "refilled_rows"}).
    :rtype: Tuple[pa.Table, Dict[str, Dict], Dict]
    :raises ValueError: For a dataset without val rows loaded.
    """
    unknown = sorted(key for key in val_rows if val_dataset(key) not in val_tables)
    if unknown:
        raise ValueError(f"val rows asked of unknown datasets {unknown}; known: {sorted(val_tables)}")
    parts, infos, dropped_stems = [], {}, []
    for key, wanted in val_rows.items():
        dataset = val_dataset(key)
        table = val_tables[dataset]
        if key != dataset:                                   # a component name: that component's val rows only
            table = table.filter(pa.array([comp == key for comp in row_components(dataset, table)], pa.bool_()))
        rng = component_rng(seed, f"val/{key}")
        drawn = rng.choice(table.num_rows, size=min(wanted, table.num_rows), replace=False)
        blocked = np.array([sha in train_images for sha in table[SHA_COLUMN].to_pylist()], dtype=bool)
        idx, dropped = refill_val_draw(drawn, blocked, wanted, rng)
        stems = table["stem"].take(pa.array(dropped, pa.int64())).to_pylist()
        if stems:
            logger.info("val %s: dropped %d drawn rows on train images: %s", key, len(stems), stems)
        dropped_stems.extend(stems)
        part = table.take(pa.array(idx, pa.int64()))
        parts.append(with_source(part, row_components(dataset, part)))
        infos[key] = {"pool": table.num_rows, "eligible": int((~blocked).sum()), "requested": wanted,
                          "taken": len(idx), "dropped": len(dropped),
                          "refilled": len(idx) - (len(drawn) - len(dropped))}
    table = pa.concat_tables(parts)
    order = component_rng(seed, "val/shuffle").permutation(table.num_rows)
    dedupe = {"rule": VAL_DEDUPE_RULE, "dropped_rows": len(dropped_stems), "dropped_stems": dropped_stems,
              "refilled_rows": sum(i["refilled"] for i in infos.values())}
    return table.take(pa.array(order)), infos, dedupe


def mixture_meta(metas: Sequence[Dict], names: Optional[Sequence[str]] = None) -> Dict:
    """images-once parquet metadata of the mixture: the first source's layout plus ``source``.

    :param metas: Every source's images-once metadata.
    :type metas: Sequence[Dict]
    :param names: Dataset of each metadata dict (for error messages).
    :type names: Optional[Sequence[str]]
    :return: Metadata whose ``columns`` end with :data:`SOURCE_COLUMN`, so
        :class:`ImagesOnceDataset` items carry it.
    :rtype: Dict
    """
    check_layouts(metas, names)
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
    """Rows and unique images by component, rows by bucket and task, mean target length, per split.

    :param tables: split -> rows with ``source``.
    :type tables: Mapping[str, pa.Table]
    :return: split -> statistics (``mean_target_chars`` skips rows without
        a ``target_chars`` value; None when a component has none).
    :rtype: Dict[str, Dict]
    """
    stats = {}
    for split, table in tables.items():
        sources = table[SOURCE_COLUMN].to_pylist()
        tasks = table["task"].to_pylist()
        chars = table["target_chars"].to_pylist()
        shas = table[SHA_COLUMN].to_pylist()
        by_source = Counter(sources)
        char_sums, char_rows = Counter(), Counter()
        images: Dict[str, set] = {}
        for source, n_chars, sha in zip(sources, chars, shas):
            images.setdefault(source, set()).add(sha)
            if n_chars is not None:
                char_sums[source] += n_chars
                char_rows[source] += 1
        stats[split] = {
            "rows": table.num_rows,
            "unique_images": len(set(shas)),
            "by_source": dict(sorted(by_source.items())),
            "unique_images_by_source": {s: len(images[s]) for s in sorted(images)},
            "by_bucket": dict(sorted(Counter(COMPONENTS[s][1] for s in sources).items())),
            "by_source_task": dict(sorted(Counter(f"{s}/{t}" for s, t in zip(sources, tasks)).items())),
            "mean_target_chars": {s: round(char_sums[s] / char_rows[s], 1) if char_rows[s] else None
                                  for s in sorted(by_source)},
        }
    return stats


def used_datasets(mixture: Dict) -> List[str]:
    """Datasets that contribute train or val rows to a built mixture, in config order.

    :param mixture: ``mixture.json`` content.
    :type mixture: Dict
    :return: Dataset keys.
    :rtype: List[str]
    """
    used = {info["dataset"] for info in mixture["components"].values() if info["taken"]}
    used |= {dataset for dataset, info in mixture["val"].items() if info["taken"]}
    return [dataset for dataset in mixture["config"]["dirs"] if dataset in used]


def card_text(mixture: Dict, manifest: Dict) -> str:
    """Markdown dataset card of a built mixture.

    Lists every component that contributes train rows (share, rows, unique
    images) and adds the rights note (:data:`RIGHTS_NOTES`) of every extra
    dataset that contributes train or val rows.

    :param mixture: ``mixture.json`` content.
    :type mixture: Dict
    :param manifest: ``manifest.json`` content.
    :type manifest: Dict
    :return: README.md text.
    :rtype: str
    """
    cfg, comps = mixture["config"], mixture["components"]
    train, val = manifest["splits"]["train"], manifest["splits"]["val"]
    used = used_datasets(mixture)
    comp_rows = "\n".join(
        f"| `{c}` | {i['bucket']} | {i['dataset']} | {i['pool']:,} | {i['share']:.4g} | "
        f"{i['taken']:,} | {mixture['realised_shares']['components'][c]:.3f} | {i['passes']} | "
        f"{i['unique_images']:,} |"
        for c, i in comps.items() if i["taken"])
    notes = [RIGHTS_NOTES[d] for d in used if d in RIGHTS_NOTES]
    rights = "\n\n" + "\n".join(f"- {note}" for note in notes) if notes else ""
    val_rows = "\n".join(f"| {d} | {i['pool']:,} | {i['eligible']:,} | {i['requested']} | {i['taken']} | "
                         f"{i['dropped']} | {i['refilled']} |" for d, i in mixture["val"].items())
    dedupe = manifest["val_dedupe"]
    bucket_rows = "\n".join(f"| {b} | {n:,} | {mixture['realised_shares']['buckets'][b]:.3f} |"
                            for b, n in mixture["buckets"]["train"].items())
    ktiv_rows = "\n".join(f"| `{t}` | {e['bucket']} | {', '.join(e['sections'])} |"
                          for t, e in mixture["ktiv_task_buckets"].items())
    source_dirs = "\n".join(f"- {d}: `{p}`" for d, p in cfg["dirs"].items())
    variants = "".join(
        f"{v['file']}   {v['rows']:,} rows: rows/train.parquet without {', '.join(v['dropped_components'])} "
        f"(same order; {v['unique_images']:,} distinct images)\n"
        for v in mixture.get("train_variants", {}).values())
    boxes = ("\nDocumentary-grounding boxes are model-derived (lines where two independent\nreaders agreed)."
             if "documentary_grounding" in used else "")
    return f"""# Genizah v22 training mixture: {cfg['name']} (images-once)

Training mixture for a v22 fine-tune of the Hebrew-manuscript VLM:
{train['rows']:,} train rows and {val['rows']:,} val rows sampled at fixed shares (seed {cfg['seed']})
from {len(used)} source datasets ({', '.join(used)}),
stored in the images-once layout: every distinct image once, named by the SHA-1
of its JPEG bytes.

## Structure

```
rows/train.parquet   {train['rows']:,} rows, shuffled; {train['unique_images']:,} distinct images
rows/val.parquet     {val['rows']:,} rows; {val['unique_images']:,} distinct images
{variants}images/<sha1>.jpg    {manifest['n_images']:,} images, {manifest['image_bytes'] / 1e9:.2f} GB (only images the rows reference)
manifest.json        rows / unique images / bytes per split
mixture.json         config, per-component pool / quota / taken / passes, realised shares
stats.json           rows and unique images by component, bucket and task
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
(each row appears floor or ceil of quota/pool times). Every component with train
rows is listed; unique images = distinct page/crop images behind its train rows.

| component | bucket | dataset | pool | share | taken | realised | passes | unique images |
|---|---|---|---|---|---|---|---|---|
{comp_rows}

| bucket | train rows | realised share |
|---|---|---|
{bucket_rows}

Val rows are drawn without replacement from each dataset's val split and never
share a page image with train: the sources split pages independently, so a drawn
val row whose image also backs a train row is dropped and refilled from the same
dataset's remaining val rows on images not in train (eligible = val rows off
train images). This build dropped {dedupe['dropped_rows']} and refilled {dedupe['refilled_rows']}.

| dataset | val pool | eligible | requested | taken | dropped | refilled |
|---|---|---|---|---|---|---|
{val_rows}

KTIV task families by bucket (grounding = the answer is bbox JSON or the
question reads a given box):

| task | bucket | sections |
|---|---|---|
{ktiv_rows}

Source datasets:

{source_dirs}

## Credits and rights

KTIV manuscript images and transcriptions: the National Library of Israel's
KTIV project (NLI-KTIV). Editions, document metadata and the image links behind
the edition, documentary-grounding and QA rows: the Princeton Geniza Project.{boxes}
Images remain under the terms of their holding institutions;
this directory is internal training data, not for redistribution.{rights}

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
                  seed: int = DEFAULT_SEED, shares: Optional[Mapping[str, float]] = None,
                  val_rows: Optional[Mapping[str, int]] = None, workers: int = 16) -> Dict[str, Dict]:
    """Sample, materialise and describe the mixture.

    The plan (shares, val rows, sources) is checked before any row is read.
    Images are copied and verified before the rows parquets are written, so a
    fresh build's rows never point at images that are not there.

    :param out_dir: Destination directory.
    :type out_dir: Path
    :param export_dirs: dataset (``ktiv`` first, then ``pgp_editions``,
        ``documentary_grounding``, ``pgp_qa`` and any extra dataset such as
        ``talmud`` or ``synthetic``) -> its images-once export. The first
        dataset's row layout is the mixture's.
    :type export_dirs: Mapping[str, Path]
    :param train_rows: Train budget.
    :type train_rows: int
    :param seed: Build seed.
    :type seed: int
    :param shares: component -> share of the train budget; None =
        :func:`default_shares` of the supplied datasets.
    :type shares: Optional[Mapping[str, float]]
    :param val_rows: dataset -> val rows; None = :func:`default_val_rows`
        of the supplied datasets.
    :type val_rows: Optional[Mapping[str, int]]
    :param workers: Parallel image copies.
    :type workers: int
    :return: {"manifest", "mixture", "stats"} as written.
    :rtype: Dict[str, Dict]
    :raises ValueError: For a bad plan (:func:`resolve_plan`) or row layouts
        that differ (:func:`check_layouts`).
    :raises FileNotFoundError: For an export without a manifest.
    """
    t0 = time.time()
    shares, val_rows, quotas = resolve_plan(list(export_dirs), train_rows, shares, val_rows)
    for dataset, path in export_dirs.items():
        check_export(dataset, path)
    sources = {dataset: load_source(path) for dataset, path in export_dirs.items()}
    meta = mixture_meta([s.meta for s in sources.values()], list(sources))
    schema = next(iter(sources.values())).train.schema
    ktiv_report = ktiv_task_report(sources["ktiv"].train)
    ktiv_task_report(sources["ktiv"].val)
    for task, entry in ktiv_report.items():
        logger.info("KTIV task %-20s -> %-13s (%d rows; %s)", task, entry["bucket"], entry["rows"],
                    ", ".join(entry["sections"]))
    train_tables, val_tables, unclean = {}, {}, {}
    for dataset, source in sources.items():
        train_tables[dataset], dropped_train = drop_unclean_transcriptions(conform(source.train, schema))
        val_tables[dataset], dropped_val = drop_unclean_transcriptions(conform(source.val, schema))
        if dropped_train or dropped_val:
            unclean[dataset] = {"train": dropped_train, "val": dropped_val}
            logger.info("%s: %d train and %d val transcription rows dropped for editor's marks in the answer",
                        dataset, len(dropped_train), len(dropped_val))
    pools = component_pools(train_tables, quotas)
    train, infos = materialize(pools, quotas, seed)
    trained = {comp for comp, quota in quotas.items() if quota > 0}
    val_tables = {d: trained_component_rows(d, table, trained) for d, table in val_tables.items()}
    val, val_infos, val_dedupe = sample_val(val_tables, val_rows, seed, frozenset(train[SHA_COLUMN].to_pylist()))
    stats = mixture_stats({"train": train, "val": val})
    val_missing = [c for c, info in infos.items() if info["taken"] and c not in stats["val"]["by_source"]]
    if val_missing:
        logger.warning("val has no rows of train components %s (the v22 notebook requires val to "
                       "cover every train source)", val_missing)
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
    manifest["val_dedupe"] = val_dedupe
    total = max(1, train.num_rows)
    train_images = stats["train"]["unique_images_by_source"]
    mixture = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {"name": out_dir.name, "train_rows": train_rows, "seed": seed, "shares": dict(shares),
                   "val_rows": dict(val_rows), "dirs": {d: str(p) for d, p in export_dirs.items()}},
        "components": {c: {"dataset": COMPONENTS[c][0], "bucket": COMPONENTS[c][1], "share": shares[c],
                           **info, "unique_images": train_images.get(c, 0)} for c, info in infos.items()},
        "train_splits": {d: s.train_splits for d, s in sources.items()},
        "val": val_infos,
        "val_missing_components": val_missing,
        "buckets": {"train": stats["train"]["by_bucket"], "val": stats["val"]["by_bucket"]},
        "realised_shares": {
            "components": {c: round(i["taken"] / total, 4) for c, i in infos.items()},
            "buckets": {b: round(n / total, 4) for b, n in stats["train"]["by_bucket"].items()}},
        "ktiv_task_buckets": ktiv_report,
        "unclean_transcriptions_dropped": {"chars": UNCLEAN_TRANSCRIPTION_CHARS, "stems": unclean},
    }
    write_card(out_dir, mixture, manifest)
    mixture["elapsed_s"] = round(time.time() - t0, 1)
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "mixture.json", mixture)
    write_json(out_dir / "stats.json", stats)
    return {"manifest": manifest, "mixture": mixture, "stats": stats}


def write_train_variant(out_dir: Path, name: str, drop_components: Sequence[str]) -> Dict:
    """Write ``rows/train_<name>.parquet``: the train rows without some components, in the same order.

    A control run trains on this file from the same directory and images: every row of it is a row of
    ``rows/train.parquet``, so the two runs differ by the dropped components only. The variant is recorded
    in ``mixture.json`` (``train_variants``) and in the card.

    :param out_dir: A built mixture directory.
    :type out_dir: Path
    :param name: Variant name (file ``rows/train_<name>.parquet``).
    :type name: str
    :param drop_components: Components whose rows are left out.
    :type drop_components: Sequence[str]
    :return: {"file", "dropped_components", "rows", "by_source", "unique_images"}.
    :rtype: Dict
    :raises ValueError: When a named component has no train rows, or no row is left.
    """
    table = pq.read_table(out_dir / "rows" / "train.parquet")
    meta = json.loads(table.schema.metadata[META_KEY])
    sources = table[SOURCE_COLUMN].to_pylist()
    drop = set(drop_components)
    absent = sorted(drop - set(sources))
    if absent:
        raise ValueError(f"components {absent} have no train rows in {out_dir}; present: {sorted(set(sources))}")
    kept = table.filter(pa.array([source not in drop for source in sources], pa.bool_()))
    if not kept.num_rows:
        raise ValueError(f"variant {name!r} would drop every train row")
    write_rows(kept, meta, out_dir / "rows" / f"train_{name}.parquet")
    info = {"file": f"rows/train_{name}.parquet", "dropped_components": sorted(drop), "rows": kept.num_rows,
            "by_source": dict(Counter(kept[SOURCE_COLUMN].to_pylist())),
            "unique_images": len(set(kept[SHA_COLUMN].to_pylist()))}
    mixture = json.loads((out_dir / "mixture.json").read_text())
    mixture.setdefault("train_variants", {})[name] = info
    write_card(out_dir, mixture, json.loads((out_dir / "manifest.json").read_text()))
    write_json(out_dir / "mixture.json", mixture)
    return info


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


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the CLI.

    :return: The parser; ``--source`` collects raw ``NAME=DIR`` strings
        (:func:`parse_sources` checks them), ``--shares`` / ``--val-rows``
        default to None (:func:`resolve_plan` fills them in).
    :rtype: argparse.ArgumentParser
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--train-rows", type=int, default=DEFAULT_TRAIN_ROWS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--shares", type=json.loads, default=None,
                    help="JSON component -> share (must sum to 1; any component whose dataset is supplied). "
                         "Default: the pilot shares, plus " + ", ".join(
                             f"{c} {s}" for c, s in EXTRA_DEFAULT_SHARES.items())
                         + " when those sources are supplied (pilot shares scaled to fill the rest)")
    ap.add_argument("--val-rows", type=json.loads, default=None,
                    help=f"JSON dataset -> val rows. Default: {json.dumps(DEFAULT_VAL_ROWS)} for the "
                         "supplied datasets")
    ap.add_argument("--ktiv-dir", type=Path, default=DEFAULT_KTIV_DIR, help="KTIV images-once export")
    ap.add_argument("--editions-dir", type=Path, default=DEFAULT_EDITIONS_DIR,
                    help="PGP editions DatasetDict (exported to <dir>_images_once)")
    ap.add_argument("--grounding-dir", type=Path, default=DEFAULT_GROUNDING_DIR,
                    help="documentary grounding DatasetDict (exported to <dir>_images_once)")
    ap.add_argument("--qa-dir", type=Path, default=DEFAULT_QA_DIR,
                    help="PGP QA DatasetDict (exported to <dir>_images_once)")
    ap.add_argument("--source", action="append", default=[], metavar="NAME=DIR",
                    help="extra images-once export, used as is (repeatable); NAME is one of "
                         + ", ".join(EXTRA_DATASETS))
    ap.add_argument("--workers", type=int, default=16, help="parallel image copies")
    ap.add_argument("--train-variant", action="append", default=[], metavar="NAME=COMPONENT[,COMPONENT...]",
                    help="also write rows/train_NAME.parquet: the train rows without these components, same order "
                         "(a control run on the same images; repeatable)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI: check the plan, export the DatasetDict sources if needed, build the mixture, self-check it.

    :param argv: Command-line arguments; None = ``sys.argv[1:]``.
    :type argv: Optional[Sequence[str]]
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = build_parser().parse_args(argv)
    extra = parse_sources(a.source)
    shares, val_rows, _ = resolve_plan([*PILOT_FLAGS, *extra], a.train_rows, a.shares, a.val_rows)
    for dataset, path in {"ktiv": a.ktiv_dir, **extra}.items():     # used as is: check before exporting
        check_export(dataset, path)
    export_dirs = {"ktiv": a.ktiv_dir}
    used = plan_datasets(shares, val_rows)
    for dataset, src in (("pgp_editions", a.editions_dir), ("documentary_grounding", a.grounding_dir),
                         ("pgp_qa", a.qa_dir)):
        if dataset in used:                                          # a plan without it neither exports nor loads it
            export_dirs[dataset] = ensure_export(src, export_dir_for(src))
    export_dirs.update(extra)
    result = build_mixture(a.out, export_dirs, a.train_rows, a.seed, shares, val_rows, a.workers)
    mixture, manifest = result["mixture"], result["manifest"]
    print("KTIV task -> bucket:")
    for task, entry in mixture["ktiv_task_buckets"].items():
        print(f"  {task:20s} {entry['bucket']:13s} {entry['rows']:6d} rows  {', '.join(entry['sections'])}")
    print("components:")
    for comp, info in mixture["components"].items():
        print(f"  {comp:22s} pool {info['pool']:6d}  quota {info['quota']:5d}  taken {info['taken']:5d}"
              f"  full_passes {info['full_passes']}  partial {info['partial_rows']:5d}"
              f"  passes {info['passes']}  unique_images {info['unique_images']:5d}")
    if mixture["val_missing_components"]:
        print("WARNING val has no rows of:", mixture["val_missing_components"])
    print("val:", json.dumps(mixture["val"]))
    print("buckets:", json.dumps(mixture["buckets"]), "realised:", json.dumps(mixture["realised_shares"]))
    print("manifest:", json.dumps({k: v for k, v in manifest.items() if k != "features"}, indent=1))
    print(f"elapsed {mixture['elapsed_s']}s")
    for split, check in self_check(a.out).items():
        print(f"self-check {split}: {check}")
    for spec in a.train_variant:
        name, _, comps = spec.partition("=")
        print("train variant:", json.dumps(write_train_variant(a.out, name.strip(), [c.strip() for c in comps.split(",") if c.strip()])))


if __name__ == "__main__":
    main()
