# File name: subset_mixture.py
# Date: 10/12/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Derive a standalone ablation mixture from a built one by leaving out whole components.

``build_v22_mixture.write_train_variant`` writes a variant rows file *inside* a built mixture (a control run that
trains from the same Hub revision). An ablation that gets its own dataset repository needs a complete
images-once directory instead; this module writes one whose rows are the source mixture's rows, in the same
order, minus the dropped components, with only the images those rows reference. The source directory is only
read.

Written to ``<out>``::

    rows/train.parquet   source train rows without the dropped components: same order, same schema metadata
    rows/val.parquet     source val rows without them (val rows come only from trained components)
    images/<sha1>.jpg    only the images the kept rows reference (copied from the source mixture)
    manifest.json        recomputed, same keys as a build's
    mixture.json         the source's record of the kept components, recomputed shares, ``derived_from``
    stats.json           recomputed
    derivation.json      per-component rows before / after, passes, steps at the notebook's batch geometry
    README.md            the builder's card for the kept components + a "Derived from" section

Every kept row appears exactly as often as in the source train file, so one pass over the new file gives each
kept row the exposure it had in the source run: the step count is ``rows // (batch x accumulation)``.

Usage (repo root)::

    .venv/bin/python -m src.finetuning.qwen_hebrew.subset_mixture \\
        --src /Volumes/home/studio_offload/datasets/genizah_v23a_arabic \\
        --out /Volumes/home/studio_offload/datasets/genizah_v23h_hebonly --drop-prefix arabic_ \\
        --source-repo isaacmg/genizah_v23a_arabic --source-revision 3b762524067c77141c121ce84a3a724991618757
"""
import argparse
import hashlib
import json
import logging
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from src.finetuning.qwen_hebrew.build_v22_mixture import (
    FORBIDDEN_CARD_STRINGS, SOURCE_COLUMN, card_text, copy_images, mixture_stats, self_check, split_manifest,
    verify_images, write_json, write_rows)
from src.finetuning.qwen_hebrew.images_once import META_KEY, SHA_COLUMN

logger = logging.getLogger(__name__)

DEFAULT_PER_DEVICE_BATCH = 1
DEFAULT_GRAD_ACCUM = 8
DEFAULT_WARMUP_RATIO = 0.05
ARABIC_LETTERS = re.compile(r"[ؠ-ي]")
HEBREW_LETTERS = re.compile(r"[א-ת]")


def resolve_drop(sources: Iterable[str], drop: Sequence[str] = (), drop_prefix: Optional[str] = None) -> List[str]:
    """Components to leave out: the named ones plus every component whose name starts with the prefix.

    :param sources: Components present in the source train rows.
    :type sources: Iterable[str]
    :param drop: Components named explicitly; each must be present.
    :type drop: Sequence[str]
    :param drop_prefix: Leave out every present component starting with this (e.g. ``arabic_``); None = none.
    :type drop_prefix: Optional[str]
    :return: Sorted component names.
    :rtype: List[str]
    :raises ValueError: When a named component is absent, nothing is selected, or nothing would be left.
    """
    present = set(sources)
    absent = sorted(set(drop) - present)
    if absent:
        raise ValueError(f"components {absent} have no train rows; present: {sorted(present)}")
    chosen = set(drop) | ({s for s in present if s.startswith(drop_prefix)} if drop_prefix else set())
    if not chosen:
        raise ValueError(f"nothing to drop (prefix {drop_prefix!r} matches none of {sorted(present)})")
    if chosen == present:
        raise ValueError("every component would be dropped")
    return sorted(chosen)


def keep_rows(table: pa.Table, drop: Iterable[str]) -> pa.Table:
    """The rows whose component is not dropped, in their stored order.

    :param table: Rows with a ``source`` column.
    :type table: pa.Table
    :param drop: Components to leave out.
    :type drop: Iterable[str]
    :return: Filtered table (schema and metadata unchanged).
    :rtype: pa.Table
    """
    dropped = set(drop)
    return table.filter(pa.array([s not in dropped for s in table[SOURCE_COLUMN].to_pylist()], pa.bool_()))


def steps_for(rows: int, per_device_batch: int = DEFAULT_PER_DEVICE_BATCH, grad_accum: int = DEFAULT_GRAD_ACCUM) -> int:
    """Optimizer steps of one pass over ``rows`` (what the notebooks set as ``max_steps``).

    :param rows: Train rows.
    :type rows: int
    :param per_device_batch: ``per_device_train_batch_size``.
    :type per_device_batch: int
    :param grad_accum: ``gradient_accumulation_steps``.
    :type grad_accum: int
    :return: ``rows // (per_device_batch * grad_accum)``.
    :rtype: int
    """
    return rows // (per_device_batch * grad_accum)


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, streamed.

    :param path: File.
    :type path: Path
    :return: Hex digest.
    :rtype: str
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arabic_script_audit(table: pa.Table) -> Dict[str, object]:
    """How much Arabic script the kept answers still carry (mixed-script documentary pages).

    :param table: Kept train rows (``answer``, ``stem``, ``source``).
    :type table: pa.Table
    :return: Rows with any Arabic letter, Arabic and Hebrew letter totals, and those rows' stems by component.
    :rtype: Dict[str, object]
    """
    answers, stems, sources = (table[c].to_pylist() for c in ("answer", "stem", SOURCE_COLUMN))
    hits = [(s, st, len(ARABIC_LETTERS.findall(a))) for a, st, s in zip(answers, stems, sources) if ARABIC_LETTERS.search(a)]
    by_component: Dict[str, List[str]] = {}
    for source, stem, _ in hits:
        by_component.setdefault(source, []).append(stem)
    return {"rows_with_arabic_letters": len(hits),
            "arabic_letters": sum(n for *_, n in hits),
            "hebrew_letters": sum(len(HEBREW_LETTERS.findall(a)) for a in answers),
            "stems_by_component": {k: sorted(set(v)) for k, v in sorted(by_component.items())}}


def derivation_record(src_dir: Path, src_mixture: Mapping, before: Mapping[str, pa.Table], after: Mapping[str, pa.Table],
                      drop: Sequence[str], provenance: Mapping[str, str], per_device_batch: int = DEFAULT_PER_DEVICE_BATCH,
                      grad_accum: int = DEFAULT_GRAD_ACCUM, warmup_ratio: float = DEFAULT_WARMUP_RATIO) -> Dict[str, object]:
    """What was left out, what stayed, and the step arithmetic of one pass over what stayed.

    :param src_dir: Source mixture directory.
    :type src_dir: Path
    :param src_mixture: Source ``mixture.json``.
    :type src_mixture: Mapping
    :param before: split -> source rows.
    :type before: Mapping[str, pa.Table]
    :param after: split -> kept rows.
    :type after: Mapping[str, pa.Table]
    :param drop: Dropped components.
    :type drop: Sequence[str]
    :param provenance: Recorded as given (e.g. ``hub_repo``, ``revision`` of the source).
    :type provenance: Mapping[str, str]
    :param per_device_batch: Notebook batch size.
    :type per_device_batch: int
    :param grad_accum: Notebook gradient accumulation.
    :type grad_accum: int
    :param warmup_ratio: Notebook warm-up ratio (the trainer rounds the steps up).
    :type warmup_ratio: float
    :return: JSON-serialisable record (``derivation.json``).
    :rtype: Dict[str, object]
    """
    comps = src_mixture["components"]
    counts = {split: (Counter(before[split][SOURCE_COLUMN].to_pylist()), Counter(after[split][SOURCE_COLUMN].to_pylist()))
              for split in before}
    rows_before, rows_after = before["train"].num_rows, after["train"].num_rows
    steps_before = steps_for(rows_before, per_device_batch, grad_accum)
    steps_after = steps_for(rows_after, per_device_batch, grad_accum)
    per_component = {}
    for comp in sorted(set(counts["train"][0]) | set(counts["val"][0])):
        info = comps.get(comp, {})
        kept = comp not in drop
        per_component[comp] = {
            "kept": kept, "pool": info.get("pool"), "passes": info.get("passes") if kept else 0,
            "train_rows_before": counts["train"][0][comp], "train_rows_after": counts["train"][1][comp],
            "val_rows_before": counts["val"][0][comp], "val_rows_after": counts["val"][1][comp],
            "share_before": round(counts["train"][0][comp] / max(1, rows_before), 4),
            "share_after": round(counts["train"][1][comp] / max(1, rows_after), 4)}
    batch = per_device_batch * grad_accum
    return {
        "source": {"dir": str(src_dir), "name": src_mixture["config"]["name"], **dict(provenance),
                   "train_parquet_sha256": file_sha256(src_dir / "rows" / "train.parquet"),
                   "val_parquet_sha256": file_sha256(src_dir / "rows" / "val.parquet")},
        "dropped_components": list(drop),
        "rule": ("rows of the source mixture in their stored order, minus every row of the dropped components; each kept "
                 "row appears exactly as often as in the source train file, so one pass here gives every kept row the "
                 "exposure it had in the source run (passes per kept component unchanged)"),
        "seed": src_mixture["config"]["seed"],
        "train": {"rows_before": rows_before, "rows_after": rows_after,
                  "unique_images_after": len(set(after["train"][SHA_COLUMN].to_pylist()))},
        "val": {"rows_before": before["val"].num_rows, "rows_after": after["val"].num_rows,
                "unique_images_after": len(set(after["val"][SHA_COLUMN].to_pylist()))},
        "per_component": per_component,
        "schedule": {
            "per_device_batch": per_device_batch, "grad_accum": grad_accum,
            "steps_before": steps_before, "steps_after": steps_after, "warmup_ratio": warmup_ratio,
            "warmup_steps_before": math.ceil(steps_before * warmup_ratio),
            "warmup_steps_after": math.ceil(steps_after * warmup_ratio),
            "rows_left_over": rows_after - steps_after * batch,
            "arithmetic": (f"{rows_after:,} rows / ({per_device_batch} x {grad_accum}) = {steps_after:,} steps "
                           f"(source: {rows_before:,} rows / {batch} = {steps_before:,} steps); one pass = the source "
                           f"run's exposure of every kept row")},
        "arabic_script_audit": arabic_script_audit(after["train"]),
    }


def subset_mixture_json(src_mixture: Mapping, drop: Sequence[str], stats: Mapping[str, Dict], name: str,
                        derivation: Mapping[str, object]) -> Dict[str, object]:
    """``mixture.json`` of the subset: the source's record of the kept components, recomputed shares.

    ``config.shares`` and each component's ``share`` stay the planned shares of the source build (they say how the
    rows were sampled); ``realised_shares`` are this directory's.

    :param src_mixture: Source ``mixture.json``.
    :type src_mixture: Mapping
    :param drop: Dropped components.
    :type drop: Sequence[str]
    :param stats: :func:`mixture_stats` of the kept rows.
    :type stats: Mapping[str, Dict]
    :param name: Directory name of the subset.
    :type name: str
    :param derivation: :func:`derivation_record` output.
    :type derivation: Mapping[str, object]
    :return: The new ``mixture.json`` content.
    :rtype: Dict[str, object]
    """
    kept = {c: dict(i) for c, i in src_mixture["components"].items() if c not in drop}
    datasets = {i["dataset"] for i in kept.values()}
    cfg = dict(src_mixture["config"])
    cfg.update(name=name, train_rows=stats["train"]["rows"],
               shares={c: s for c, s in cfg["shares"].items() if c in kept},
               val_rows={d: n for d, n in cfg["val_rows"].items() if d in datasets},
               dirs={d: p for d, p in cfg["dirs"].items() if d in datasets})
    total = max(1, stats["train"]["rows"])
    unclean = dict(src_mixture.get("unclean_transcriptions_dropped", {}))
    if "stems" in unclean:
        unclean["stems"] = {d: v for d, v in unclean["stems"].items() if d in datasets}
    return {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg,
        "components": kept,
        "train_splits": {d: s for d, s in src_mixture.get("train_splits", {}).items() if d in datasets},
        "val": {d: i for d, i in src_mixture["val"].items() if d in datasets},
        "val_missing_components": [c for c in kept if c not in stats["val"]["by_source"]],
        "buckets": {"train": stats["train"]["by_bucket"], "val": stats["val"]["by_bucket"]},
        "realised_shares": {
            "components": {c: round(stats["train"]["by_source"].get(c, 0) / total, 4) for c in kept},
            "buckets": {b: round(n / total, 4) for b, n in stats["train"]["by_bucket"].items()}},
        "ktiv_task_buckets": src_mixture.get("ktiv_task_buckets", {}),
        "unclean_transcriptions_dropped": unclean,
        "derived_from": dict(derivation),
    }


def derived_card_section(derivation: Mapping[str, object]) -> str:
    """The card's "Derived from" section.

    :param derivation: :func:`derivation_record` output.
    :type derivation: Mapping[str, object]
    :return: Markdown.
    :rtype: str
    """
    src, train, val, sched = derivation["source"], derivation["train"], derivation["val"], derivation["schedule"]
    where = f"`{src['hub_repo']}` (revision `{src.get('revision', '?')}`)" if src.get("hub_repo") else f"`{src['name']}`"
    kept = ", ".join(f"`{c}` {i['train_rows_after']:,}" for c, i in derivation["per_component"].items() if i["kept"])
    dropped = ", ".join(f"`{c}`" for c in derivation["dropped_components"])
    return (f"\n## Derived from {src['name']}\n\n"
            f"The rows are the rows of {where} in their stored order, without the components {dropped}:\n"
            f"{train['rows_after']:,} of {train['rows_before']:,} train rows ({kept}) and "
            f"{val['rows_after']:,} of {val['rows_before']:,} val rows.\n"
            f"Every kept row appears as often as in the source train file, so one pass here "
            f"({sched['steps_after']:,} steps at batch {sched['per_device_batch']} x {sched['grad_accum']})\n"
            f"gives each kept row the exposure it had in the source's {sched['steps_before']:,}-step run. "
            f"Component shares above are the source build's planned shares; `derivation.json` has the per-component counts.\n")


def adapt_card(card: str, derivation: Mapping[str, object]) -> str:
    """Turn the builder's card into the subset's: quota base, file list, provenance line, "Derived from" section.

    :param card: :func:`card_text` of the subset's mixture and manifest.
    :type card: str
    :param derivation: :func:`derivation_record` output.
    :type derivation: Mapping[str, object]
    :return: The subset's card.
    :rtype: str
    :raises ValueError: When a phrase of the builder's card is not found exactly once.
    """
    train = derivation["train"]
    edits = (
        (f"Quota = round(share x {train['rows_after']:,})", f"Quota = round(share x {train['rows_before']:,}) in the source build"),
        ("stats.json           rows and unique images by component, bucket and task\n",
         "stats.json           rows and unique images by component, bucket and task\n"
         "derivation.json      rows before / after per component and the step arithmetic (see the end)\n"),
        ("by `src/finetuning/qwen_hebrew/build_v22_mixture.py`.",
         "by `src/finetuning/qwen_hebrew/subset_mixture.py` from a `build_v22_mixture.py` build."),
    )
    for old, new in edits:
        if card.count(old) != 1:
            raise ValueError(f"card phrase not found exactly once: {old!r}")
        card = card.replace(old, new)
    return card + derived_card_section(derivation)


def write_subset_mixture(src_dir: Path, out_dir: Path, drop: Sequence[str] = (), drop_prefix: Optional[str] = None,
                         provenance: Optional[Mapping[str, str]] = None, workers: int = 16,
                         per_device_batch: int = DEFAULT_PER_DEVICE_BATCH, grad_accum: int = DEFAULT_GRAD_ACCUM,
                         warmup_ratio: float = DEFAULT_WARMUP_RATIO) -> Dict[str, object]:
    """Write the subset directory (images first, then rows, then the JSON files and the card).

    :param src_dir: A built mixture (``rows/``, ``images/``, ``manifest.json``, ``mixture.json``); only read.
    :type src_dir: Path
    :param out_dir: Destination (created; must not be the source or inside it).
    :type out_dir: Path
    :param drop: Components to leave out by name.
    :type drop: Sequence[str]
    :param drop_prefix: Also leave out every component starting with this.
    :type drop_prefix: Optional[str]
    :param provenance: Recorded in ``derivation.json`` (``hub_repo``, ``revision`` of the source).
    :type provenance: Optional[Mapping[str, str]]
    :param workers: Parallel image copies.
    :type workers: int
    :param per_device_batch: Notebook batch size (step arithmetic only).
    :type per_device_batch: int
    :param grad_accum: Notebook gradient accumulation (step arithmetic only).
    :type grad_accum: int
    :param warmup_ratio: Notebook warm-up ratio (step arithmetic only).
    :type warmup_ratio: float
    :return: {"derivation", "manifest", "mixture", "stats"} as written.
    :rtype: Dict[str, object]
    :raises ValueError: For a bad drop list or an output inside the source.
    """
    src_dir, out_dir = Path(src_dir).resolve(), Path(out_dir).resolve()
    if out_dir == src_dir or src_dir in out_dir.parents:
        raise ValueError(f"output {out_dir} must not be the source mixture or lie inside it")
    before = {split: pq.read_table(src_dir / "rows" / f"{split}.parquet") for split in ("train", "val")}
    meta = json.loads(before["train"].schema.metadata[META_KEY])
    dropped = resolve_drop(before["train"][SOURCE_COLUMN].to_pylist(), drop, drop_prefix)
    after = {split: keep_rows(table, dropped) for split, table in before.items()}
    src_mixture = json.loads((src_dir / "mixture.json").read_text())
    src_manifest = json.loads((src_dir / "manifest.json").read_text())
    wanted = {sha: src_dir / "images" / f"{sha}.jpg" for table in after.values() for sha in table[SHA_COLUMN].to_pylist()}
    sizes = copy_images(wanted, out_dir / "images", workers)
    verify_images(after.values(), out_dir / "images")
    for split, table in after.items():
        write_rows(table, meta, out_dir / "rows" / f"{split}.parquet")
    stats = mixture_stats(after)
    derivation = derivation_record(src_dir, src_mixture, before, after, dropped, provenance or {},
                                   per_device_batch, grad_accum, warmup_ratio)
    mixture = subset_mixture_json(src_mixture, dropped, stats, out_dir.name, derivation)
    datasets = {i["dataset"] for i in mixture["components"].values()}
    splits = {split: split_manifest(table, sizes) for split, table in after.items()}
    manifest = {"source": {d: p for d, p in src_manifest["source"].items() if d in datasets},
                "image_column": src_manifest["image_column"], "sha_column": SHA_COLUMN, "splits": splits,
                "features": src_manifest["features"], "n_rows": sum(t.num_rows for t in after.values()),
                "n_images": len(sizes), "image_bytes": sum(sizes.values()),
                "row_image_bytes": sum(s["row_image_bytes"] for s in splits.values()),
                "val_dedupe": src_manifest["val_dedupe"], "derived_from": str(src_dir)}
    manifest["dedupe_ratio"] = round(manifest["row_image_bytes"] / max(1, manifest["image_bytes"]), 3)
    card = adapt_card(card_text(mixture, manifest), derivation)
    hits = [word for word in FORBIDDEN_CARD_STRINGS if word in card.lower()]
    if hits:
        raise ValueError(f"dataset card would name {hits}; those sources must not be credited")
    (out_dir / "README.md").write_text(card)
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "mixture.json", mixture)
    write_json(out_dir / "stats.json", stats)
    write_json(out_dir / "derivation.json", derivation)
    return {"derivation": derivation, "manifest": manifest, "mixture": mixture, "stats": stats}


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the CLI.

    :return: The parser.
    :rtype: argparse.ArgumentParser
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, required=True, help="built mixture directory (read only)")
    ap.add_argument("--out", type=Path, required=True, help="destination directory")
    ap.add_argument("--drop", action="append", default=[], metavar="COMPONENT[,COMPONENT...]",
                    help="component(s) to leave out (repeatable)")
    ap.add_argument("--drop-prefix", default=None, help="also leave out every component starting with this, e.g. arabic_")
    ap.add_argument("--source-repo", default=None, help="Hub repo of the source mixture (recorded only)")
    ap.add_argument("--source-revision", default=None, help="Hub revision of the source mixture (recorded only)")
    ap.add_argument("--per-device-batch", type=int, default=DEFAULT_PER_DEVICE_BATCH)
    ap.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    ap.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    ap.add_argument("--workers", type=int, default=16, help="parallel image copies")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI: write the subset, print its derivation record and load both splits back.

    :param argv: Command-line arguments; None = ``sys.argv[1:]``.
    :type argv: Optional[Sequence[str]]
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = build_parser().parse_args(argv)
    drop = [c.strip() for spec in a.drop for c in spec.split(",") if c.strip()]
    provenance = {k: v for k, v in (("hub_repo", a.source_repo), ("revision", a.source_revision)) if v}
    result = write_subset_mixture(a.src, a.out, drop, a.drop_prefix, provenance, a.workers,
                                  a.per_device_batch, a.grad_accum, a.warmup_ratio)
    print(json.dumps(result["derivation"], indent=1, ensure_ascii=False))
    print("manifest:", json.dumps({k: v for k, v in result["manifest"].items() if k != "features"}, indent=1))
    for split, check in self_check(a.out).items():
        print(f"self-check {split}: {check}")


if __name__ == "__main__":
    main()
