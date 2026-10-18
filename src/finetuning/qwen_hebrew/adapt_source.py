# File name: adapt_source.py
# Date: 9/28/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Adapt an older training source to the KTIV row schema, ready for the images-once export.

The v22 mixture builder concatenates images-once rows of several sources, so
every source must carry exactly the KTIV columns (``build_ktiv_dataset.FEATURES``)
in the KTIV order with the KTIV types, and ``train`` / ``val`` splits. This CLI
reads a ``save_to_disk`` DatasetDict (``--src``) or a directory of
``<split>-XXXXX-of-XXXXX.parquet`` files, e.g. a Hub snapshot's ``data/``
(``--parquet-dir``; only the kept splits' files are loaded), and writes a new
``save_to_disk`` DatasetDict:

- ``--rename-split OLD=NEW`` renames splits (``eval=val``), then
  ``--keep-splits`` keeps only the listed ones, in that order;
- ``--drop-columns`` removes source-only columns (``font style mode``);
- ``--label-source`` overwrites the ``label_source`` column;
- missing ``target_chars`` (``len(answer)``) and ``target_tokens`` (0) are
  added as int32, as the KTIV builder writes them;
- ``--filter <json>`` keeps the rows matching any rule of
  ``{"any": [{<column>: value | [values], <column>_regex: re,
  <column>_not_regex: re, ...}, ...]}`` in every split (a rule holds when all
  its conditions do; regexes use ``re.search``), e.g.
  ``filters/talmud_v21b.json``;
- columns are reordered to the KTIV order and their types checked; the
  ``image`` column (a ``datasets.Image`` feature) is never touched.

No split is loaded into memory whole: the saved or converted arrow stays
memory mapped, the filter keeps only an indices mapping (in memory, so nothing
is written next to the source), columns are added before the filter so no
step rewrites the images, and ``save_to_disk`` streams the kept rows in
batches. ``<out>/stats.json`` records the source, every step and the row
counts; the images-once export copies it next to its manifest.

Usage (repo root):
    .venv/bin/python -m src.finetuning.qwen_hebrew.adapt_source \\
        --parquet-dir /Users/isaac/hub_stage_talmud/talmud_finetune_v2/data \\
        --keep-splits train val \\
        --filter src/finetuning/qwen_hebrew/filters/talmud_v21b.json \\
        --cache-dir /Users/isaac/hub_stage_talmud/hf_cache \\
        --out /Users/isaac/hub_stage_talmud/talmud_v21b_filtered
    .venv/bin/python -m src.finetuning.qwen_hebrew.adapt_source \\
        --src src/datasets/processed/synthetic_hebrew_v3/dataset \\
        --rename-split eval=val --drop-columns font style mode \\
        --out /Users/isaac/hub_stage_synth/synthetic_v3_adapted
"""
import argparse
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from datasets import Dataset, DatasetDict, Features, Image, Value, load_dataset, load_from_disk

from src.finetuning.qwen_hebrew.build_ktiv_dataset import FEATURES
from src.finetuning.qwen_hebrew.images_once import IMAGE_COLUMN, saved_splits

logger = logging.getLogger(__name__)

REGEX_SUFFIX = "_regex"
NOT_REGEX_SUFFIX = "_not_regex"
FILTER_KEYS = ("any", "description")
PARQUET_NAME = re.compile(r"(?P<split>.+?)(?:-\d+-of-\d+)?\.parquet")
TARGET_COLUMNS = ("target_chars", "target_tokens")


@dataclass(frozen=True)
class Condition:
    """One column test of a filter rule.

    :param column: Column tested.
    :type column: str
    :param allowed: Values the column must take; None for a regex test.
    :type allowed: Optional[frozenset]
    :param pattern: Regex searched in the value (``re.search``); None for a
        membership test.
    :type pattern: Optional[str]
    :param negate: With ``pattern``: the value must NOT match.
    :type negate: bool
    """
    column: str
    allowed: Optional[frozenset] = None
    pattern: Optional[str] = None
    negate: bool = False

    def holds(self, value: Any) -> bool:
        """Whether a row's value passes this condition.

        :param value: The row's value in :attr:`column`.
        :type value: Any
        :return: True when it passes.
        :rtype: bool
        """
        if self.allowed is not None:
            return value in self.allowed
        return (re.search(self.pattern, value) is not None) != self.negate


@dataclass(frozen=True)
class RowFilter:
    """Compiled ``{"any": [rule, ...]}`` spec: a row passes when any rule holds.

    :param columns: Columns the rules read, sorted (the ``input_columns`` of
        the ``datasets`` filter).
    :type columns: Tuple[str, ...]
    :param rules: Conditions per rule; a rule holds when all of them do.
    :type rules: Tuple[Tuple[Condition, ...], ...]
    """
    columns: Tuple[str, ...]
    rules: Tuple[Tuple[Condition, ...], ...]

    def keeps(self, row: Mapping[str, Any]) -> bool:
        """Whether a row passes the filter.

        :param row: Column -> value (at least :attr:`columns`).
        :type row: Mapping[str, Any]
        :return: True when any rule holds.
        :rtype: bool
        """
        return any(all(c.holds(row[c.column]) for c in rule) for rule in self.rules)

    def batch_mask(self, *batch: List[Any]) -> List[bool]:
        """Batched ``datasets`` filter function.

        :param batch: One value list per column of :attr:`columns`, in order.
        :type batch: List[Any]
        :return: Keep flag per row.
        :rtype: List[bool]
        """
        return [self.keeps(dict(zip(self.columns, values))) for values in zip(*batch)]


def compile_condition(key: str, value: Any) -> Condition:
    """Parse one ``key: value`` entry of a filter rule.

    :param key: ``<column>`` (membership), ``<column>_regex`` (must match) or
        ``<column>_not_regex`` (must not match).
    :type key: str
    :param value: A value or non-empty list of values; a regex string for the
        regex keys.
    :type value: Any
    :return: The condition.
    :rtype: Condition
    :raises ValueError: For an empty value list or a non-string regex.
    :raises re.error: For a regex that does not compile.
    """
    for suffix, negate in ((NOT_REGEX_SUFFIX, True), (REGEX_SUFFIX, False)):
        if key.endswith(suffix):
            if not isinstance(value, str):
                raise ValueError(f"filter key {key!r} needs a regex string, got {value!r}")
            re.compile(value)
            return Condition(key[:-len(suffix)], pattern=value, negate=negate)
    values = value if isinstance(value, list) else [value]
    if not values:
        raise ValueError(f"filter key {key!r} lists no values")
    return Condition(key, allowed=frozenset(values))


def compile_filter(spec: Mapping[str, Any]) -> RowFilter:
    """Compile a filter spec.

    :param spec: ``{"any": [rule, ...], "description": "..."}``; each rule is a
        non-empty mapping of :func:`compile_condition` entries.
    :type spec: Mapping[str, Any]
    :return: The compiled filter.
    :rtype: RowFilter
    :raises ValueError: For unknown top-level keys, no rules, or an empty rule.
    """
    unknown = sorted(set(spec) - set(FILTER_KEYS))
    if unknown:
        raise ValueError(f"unknown filter keys {unknown}; allowed: {list(FILTER_KEYS)}")
    rules = spec.get("any")
    if not isinstance(rules, list) or not rules:
        raise ValueError('filter needs a non-empty "any" list of rules')
    compiled = []
    for rule in rules:
        if not isinstance(rule, dict) or not rule:
            raise ValueError(f"filter rule must be a non-empty mapping, got {rule!r}")
        compiled.append(tuple(compile_condition(k, v) for k, v in rule.items()))
    columns = tuple(sorted({c.column for rule in compiled for c in rule}))
    return RowFilter(columns, tuple(compiled))


def parse_pairs(pairs: Sequence[str], what: str) -> Dict[str, str]:
    """Parse ``KEY=VALUE`` arguments.

    :param pairs: ``KEY=VALUE`` strings.
    :type pairs: Sequence[str]
    :param what: Option name for error messages.
    :type what: str
    :return: key -> value, in argument order.
    :rtype: Dict[str, str]
    :raises ValueError: For a malformed pair or a repeated key.
    """
    parsed: Dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key or not value:
            raise ValueError(f"{what} expects KEY=VALUE, got {pair!r}")
        if key in parsed:
            raise ValueError(f"{what} repeats {key!r}")
        parsed[key] = value
    return parsed


def parquet_split_files(parquet_dir: Path) -> Dict[str, List[str]]:
    """Parquet files of a directory grouped by split (``<split>[-XXXXX-of-XXXXX].parquet``).

    :param parquet_dir: Directory of parquet files (e.g. a Hub snapshot's ``data``).
    :type parquet_dir: Path
    :return: split -> sorted file paths, splits in order of their first file name.
    :rtype: Dict[str, List[str]]
    :raises FileNotFoundError: When the directory holds no parquet file.
    """
    files: Dict[str, List[str]] = {}
    for path in sorted(parquet_dir.glob("*.parquet")):
        files.setdefault(PARQUET_NAME.fullmatch(path.name).group("split"), []).append(str(path))
    if not files:
        raise FileNotFoundError(f"no *.parquet files in {parquet_dir}")
    return files


def plan_splits(available: Sequence[str], renames: Mapping[str, str],
                keep: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """Output split -> source split, after renaming and keeping.

    :param available: Source split names.
    :type available: Sequence[str]
    :param renames: Source name -> new name.
    :type renames: Mapping[str, str]
    :param keep: Output splits to keep, in this order; None keeps all.
    :type keep: Optional[Sequence[str]]
    :return: Output split -> source split.
    :rtype: Dict[str, str]
    :raises ValueError: For renaming a missing split, colliding names, or
        keeping a split that does not exist after renaming.
    """
    missing = sorted(set(renames) - set(available))
    if missing:
        raise ValueError(f"cannot rename missing splits {missing}; source has {list(available)}")
    renamed = {renames.get(s, s): s for s in available}
    if len(renamed) != len(available):
        raise ValueError(f"renames {dict(renames)} make split names collide in {list(available)}")
    if keep is None:
        return renamed
    absent = [s for s in keep if s not in renamed]
    if absent:
        raise ValueError(f"cannot keep splits {absent}; after renaming there are {list(renamed)}")
    return {s: renamed[s] for s in keep}


def available_splits(src: Optional[Path] = None, parquet_dir: Optional[Path] = None) -> List[str]:
    """Split names of the source.

    :param src: ``save_to_disk`` DatasetDict directory.
    :type src: Optional[Path]
    :param parquet_dir: Directory of split parquet files.
    :type parquet_dir: Optional[Path]
    :return: Split names.
    :rtype: List[str]
    :raises ValueError: Unless exactly one source is given.
    """
    if (src is None) == (parquet_dir is None):
        raise ValueError("give exactly one of src / parquet_dir")
    return saved_splits(src) if src is not None else list(parquet_split_files(parquet_dir))


def load_splits(plan: Mapping[str, str], src: Optional[Path] = None,
                parquet_dir: Optional[Path] = None, cache_dir: Optional[Path] = None,
                num_proc: Optional[int] = None) -> DatasetDict:
    """Load the planned source splits under their output names (memory mapped).

    :param plan: Output split -> source split (:func:`plan_splits`).
    :type plan: Mapping[str, str]
    :param src: ``save_to_disk`` DatasetDict directory.
    :type src: Optional[Path]
    :param parquet_dir: Directory of split parquet files; only the planned
        splits' files are converted to arrow.
    :type parquet_dir: Optional[Path]
    :param cache_dir: ``datasets`` cache for the parquet -> arrow conversion.
    :type cache_dir: Optional[Path]
    :param num_proc: Conversion processes.
    :type num_proc: Optional[int]
    :return: The planned splits, in plan order.
    :rtype: DatasetDict
    :raises ValueError: Unless exactly one source is given.
    """
    if (src is None) == (parquet_dir is None):
        raise ValueError("give exactly one of src / parquet_dir")
    if src is not None:
        saved = load_from_disk(str(src))
        return DatasetDict({out: saved[source] for out, source in plan.items()})
    files = parquet_split_files(parquet_dir)
    loaded = load_dataset("parquet", data_files={out: files[source] for out, source in plan.items()},
                          cache_dir=str(cache_dir) if cache_dir is not None else None,
                          num_proc=num_proc)
    return DatasetDict({out: loaded[out] for out in plan})


def column_values(ds: Dataset, column: str) -> List[Any]:
    """One column's values in row order, reading no other column.

    :param ds: Dataset (an indices mapping is respected).
    :type ds: Dataset
    :param column: Column name.
    :type column: str
    :return: Values.
    :rtype: List[Any]
    """
    return ds.select_columns([column]).with_format("arrow")[:][column].to_pylist()


def add_target_columns(ds: Dataset) -> Tuple[Dataset, List[str]]:
    """Add the KTIV length columns a source lacks, as the KTIV builder fills them.

    ``target_chars`` = ``len(answer)``, ``target_tokens`` = 0, both int32.
    Run before any filter: ``add_column`` rewrites a dataset that carries an
    indices mapping.

    :param ds: Dataset with an ``answer`` column.
    :type ds: Dataset
    :return: (dataset, names of the added columns).
    :rtype: Tuple[Dataset, List[str]]
    """
    added = []
    if "target_chars" not in ds.column_names:
        chars = np.array([len(a) for a in column_values(ds, "answer")], dtype=np.int32)
        ds = ds.add_column("target_chars", chars, feature=Value("int32"))
        added.append("target_chars")
    if "target_tokens" not in ds.column_names:
        ds = ds.add_column("target_tokens", np.zeros(len(ds), dtype=np.int32), feature=Value("int32"))
        added.append("target_tokens")
    return ds, added


def set_label_source(ds: Dataset, label_source: str) -> Dataset:
    """Overwrite (or add) the ``label_source`` column with one value.

    :param ds: Dataset.
    :type ds: Dataset
    :param label_source: Value for every row.
    :type label_source: str
    :return: The dataset with the new column (last; :func:`to_ktiv_layout` reorders).
    :rtype: Dataset
    """
    if "label_source" in ds.column_names:
        ds = ds.remove_columns("label_source")
    return ds.add_column("label_source", [label_source] * len(ds), feature=Value("string"))


def filter_rows(ds: Dataset, row_filter: RowFilter, num_proc: Optional[int] = None) -> Dataset:
    """Keep the rows passing a filter (indices mapping only; nothing written next to the source).

    :param ds: Dataset.
    :type ds: Dataset
    :param row_filter: Compiled filter.
    :type row_filter: RowFilter
    :param num_proc: Filter processes.
    :type num_proc: Optional[int]
    :return: The filtered dataset.
    :rtype: Dataset
    :raises ValueError: When the filter reads a column the dataset lacks.
    """
    missing = sorted(set(row_filter.columns) - set(ds.column_names))
    if missing:
        raise ValueError(f"filter reads missing columns {missing}; dataset has {ds.column_names}")
    return ds.filter(row_filter.batch_mask, input_columns=list(row_filter.columns), batched=True,
                     keep_in_memory=True, num_proc=num_proc)


def to_ktiv_layout(ds: Dataset, features: Features = FEATURES) -> Dataset:
    """Reorder a dataset's columns to the KTIV order and check their types.

    :param ds: Dataset holding exactly the KTIV columns.
    :type ds: Dataset
    :param features: Target schema (image column: any ``datasets.Image``).
    :type features: Features
    :return: The reordered dataset (a view; nothing rewritten).
    :rtype: Dataset
    :raises ValueError: For extra or missing columns, a non-Image image
        column, or another column whose type differs.
    """
    extra = [c for c in ds.column_names if c not in features]
    missing = [c for c in features if c not in ds.column_names]
    if extra or missing:
        raise ValueError(f"columns differ from the KTIV schema: extra {extra} (drop them with "
                         f"--drop-columns), missing {missing}")
    ds = ds.select_columns(list(features))
    if not isinstance(ds.features[IMAGE_COLUMN], Image):
        raise ValueError(f"column {IMAGE_COLUMN!r} is {ds.features[IMAGE_COLUMN]}, not a datasets.Image")
    wrong = {c: f"{ds.features[c]} != {f}" for c, f in features.items()
             if c != IMAGE_COLUMN and ds.features[c] != f}
    if wrong:
        raise ValueError(f"column types differ from the KTIV schema: {wrong}")
    return ds


def adapt_split(ds: Dataset, drop_columns: Sequence[str] = (), label_source: Optional[str] = None,
                row_filter: Optional[RowFilter] = None,
                num_proc: Optional[int] = None) -> Tuple[Dataset, List[str]]:
    """Adapt one split: drop columns, set label_source, add length columns, filter, reorder.

    :param ds: Source split.
    :type ds: Dataset
    :param drop_columns: Columns to remove (each must exist).
    :type drop_columns: Sequence[str]
    :param label_source: New ``label_source`` for every row, or None to keep.
    :type label_source: Optional[str]
    :param row_filter: Filter, or None to keep every row.
    :type row_filter: Optional[RowFilter]
    :param num_proc: Filter processes.
    :type num_proc: Optional[int]
    :return: (split in the KTIV layout, names of the added length columns).
    :rtype: Tuple[Dataset, List[str]]
    :raises ValueError: For a column to drop that the split lacks.
    """
    absent = sorted(set(drop_columns) - set(ds.column_names))
    if absent:
        raise ValueError(f"cannot drop missing columns {absent}; split has {ds.column_names}")
    if drop_columns:
        ds = ds.remove_columns(list(drop_columns))
    if label_source is not None:
        ds = set_label_source(ds, label_source)
    ds, added = add_target_columns(ds)
    if row_filter is not None:
        ds = filter_rows(ds, row_filter, num_proc)
    return to_ktiv_layout(ds), added


def adapt_dataset(dsd: DatasetDict, drop_columns: Sequence[str] = (), label_source: Optional[str] = None,
                  row_filter: Optional[RowFilter] = None,
                  num_proc: Optional[int] = None) -> Tuple[DatasetDict, Dict[str, Dict]]:
    """Adapt every split of a DatasetDict.

    :param dsd: Source splits under their output names.
    :type dsd: DatasetDict
    :param drop_columns: Columns to remove.
    :type drop_columns: Sequence[str]
    :param label_source: New ``label_source``, or None to keep.
    :type label_source: Optional[str]
    :param row_filter: Filter applied to every split, or None.
    :type row_filter: Optional[RowFilter]
    :param num_proc: Filter processes.
    :type num_proc: Optional[int]
    :return: (adapted splits, split -> {"rows_in", "rows_out", "added_columns",
        "tasks", "sections"}).
    :rtype: Tuple[DatasetDict, Dict[str, Dict]]
    :raises ValueError: When the filter leaves a split empty.
    """
    out, counts = {}, {}
    for split, ds in dsd.items():
        adapted, added = adapt_split(ds, drop_columns, label_source, row_filter, num_proc)
        if len(adapted) == 0:
            raise ValueError(f"the filter kept no rows of split {split!r} ({len(ds)} rows)")
        out[split] = adapted
        counts[split] = {"rows_in": len(ds), "rows_out": len(adapted), "added_columns": added,
                         "tasks": dict(sorted(Counter(column_values(adapted, "task")).items())),
                         "sections": dict(sorted(Counter(column_values(adapted, "section")).items()))}
        logger.info("%s: %d -> %d rows, tasks %s", split, len(ds), len(adapted), counts[split]["tasks"])
    return DatasetDict(out), counts


def write_adapted(dsd: DatasetDict, out_dir: Path, stats: Dict) -> None:
    """Save the adapted DatasetDict and its ``stats.json``.

    :param dsd: Adapted splits.
    :type dsd: DatasetDict
    :param out_dir: New directory (must not exist or be empty).
    :type out_dir: Path
    :param stats: Provenance and counts, written to ``out_dir/stats.json``.
    :type stats: Dict
    :raises FileExistsError: For a non-empty ``out_dir``.
    """
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty")
    dsd.save_to_disk(str(out_dir))
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI: adapt a DatasetDict (saved or parquet) to the KTIV schema and save it.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :type argv: Optional[Sequence[str]]
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--src", type=Path, help="save_to_disk DatasetDict directory")
    source.add_argument("--parquet-dir", type=Path,
                        help="directory of <split>-XXXXX-of-XXXXX.parquet files")
    ap.add_argument("--out", type=Path, required=True, help="new save_to_disk directory")
    ap.add_argument("--rename-split", nargs="+", action="extend", default=[], metavar="OLD=NEW")
    ap.add_argument("--keep-splits", nargs="+", action="extend", default=None, metavar="SPLIT")
    ap.add_argument("--drop-columns", nargs="+", action="extend", default=[], metavar="COLUMN")
    ap.add_argument("--filter", type=Path, default=None, help='JSON {"any": [rule, ...]} row filter')
    ap.add_argument("--label-source", default=None, help="overwrite label_source on every row")
    ap.add_argument("--meta", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                    help="provenance recorded in stats.json (e.g. hub_repo=..., revision=...)")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="datasets cache for the parquet -> arrow conversion")
    ap.add_argument("--num-proc", type=int, default=None, help="conversion / filter processes")
    a = ap.parse_args(argv)

    renames = parse_pairs(a.rename_split, "--rename-split")
    meta = parse_pairs(a.meta, "--meta")
    spec = json.loads(a.filter.read_text()) if a.filter is not None else None
    row_filter = compile_filter(spec) if spec is not None else None
    available = available_splits(a.src, a.parquet_dir)
    plan = plan_splits(available, renames, a.keep_splits)
    dsd = load_splits(plan, a.src, a.parquet_dir, a.cache_dir, a.num_proc)
    adapted, counts = adapt_dataset(dsd, a.drop_columns, a.label_source, row_filter, a.num_proc)
    stats = {"adapter": "src/finetuning/qwen_hebrew/adapt_source.py",
             "source": str(a.src if a.src is not None else a.parquet_dir),
             "source_format": "save_to_disk" if a.src is not None else "parquet",
             "meta": meta, "source_splits": available,
             "split_sources": plan, "renamed_splits": renames,
             "dropped_splits": [s for s in available if s not in plan.values()],
             "dropped_columns": list(a.drop_columns), "label_source": a.label_source,
             "filter_file": str(a.filter) if a.filter is not None else None, "filter": spec,
             "columns": list(FEATURES), "splits": counts}
    write_adapted(adapted, a.out, stats)
    logger.info("saved %s: %s", a.out, {s: c["rows_out"] for s, c in counts.items()})


if __name__ == "__main__":
    main()
