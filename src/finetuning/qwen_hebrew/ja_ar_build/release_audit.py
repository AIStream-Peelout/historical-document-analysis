# File name: release_audit.py
# Date: 11/22/26
# Author: Isaac Godfried. Coded by Claude.
"""The release audit of a training export (round ja_ar_v1): an independent sample of the final rows, judged blind.

Why. The integrity audit of 2026-11-22 (required change 5) found a dataset released although many of its rows left
out text that the photograph shows. The target "at least 95 % complete pairs" stood only in a docstring, the samples
that were judged had been used to tune the assembly, and nothing judged the rows as they were finally exported. This
module makes the target a release condition:

1. :func:`draw` takes a stratified sample from the EXPORT (the rows as they will be trained on: the exported answer,
   the exported image), joined to the build manifest for the strata;
2. :func:`write_audit` writes the judges' items (contract ``v2`` fields, :mod:`.audit_contract`) and the hidden key,
   in the formats ``audit/make_prompts.py``, ``audit/score.py`` and :func:`adjudicate.load_audit` already read;
3. :func:`verify` scores the judges' records against the fixed target (:data:`THRESHOLD`) and a minimum sample size
   fixed at draw time. Production requires at least 100 judgeable distinct images and two reviews per item.
4. :func:`marker` / :func:`write_marker` record the verdict in ``RELEASE_AUDIT.json`` in the build's report folder,
   bound by SHA-256 to the manifest, the split files of one export and every referenced JPEG. Small draws write
   ``DIAGNOSTIC_AUDIT.json``. :func:`require_release` refuses (``SystemExit``)
   a push or a training step on any export the marker does not cover.

Strata. Every row has one PRIMARY stratum, ``<approval kind>:<remedy class>|<script class>`` (``score.py`` reads the
part before ``|`` as the remedy class column and the part after it as the script column), and a list of FACETS:
``split:``, ``approval:``, ``remedy:``, ``script:``, ``source:`` and one facet per destructive or risky step the row
went through (:data:`FLAG_FACETS` from the gate flags, plus ``open_lines_settled``, ``scrambled_lines`` and
``text_cleaned``). The draw takes ``per_stratum`` rows of every primary stratum, then tops up every facet that has
fewer than ``per_facet`` rows in the sample, then tops up the whole sample to ``min_items``. Inside each step it takes
at most one row per document group while rows of unseen groups remain. Every choice is made with a random generator
seeded from the seed and the stratum or facet name, so a seed gives the same sample on every machine.

The sample must not be used to tune anything. The threshold is a constant, ``min_items`` is fixed in the draw
record, and a marker, once written, is never overwritten (a failed export cannot be re-drawn until it passes).
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import random
import tarfile
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from PIL import Image

from src.finetuning.qwen_hebrew.ja_ar_build import adjudicate, audit_contract, textnorm

#: The release target: share of judged items complete without any concern. Not a parameter of the command line.
THRESHOLD = 0.95
#: Fewest items that must be judged for a release audit to pass.
DEFAULT_MIN_ITEMS = 100
DEFAULT_PER_STRATUM = 4
DEFAULT_PER_FACET = 3
DEFAULT_SEED = 20261122
#: Long side of a judge's copy of a photograph (as ``audit/make_items.py`` renders it).
LONG_SIDE = 2400
#: One-sided 95 % normal quantile for the Wilson lower bound.
Z_ONE_SIDED_95 = 1.6448536269514722
#: Name of the marker in the build's report folder.
MARKER_NAME = "RELEASE_AUDIT.json"
MARKER_VERSION = 2
#: Gate flags that name a destructive or risky step; each gets a facet of its own.
FLAG_FACETS = ("unread_lines_kept", "several_runs", "two_pages", "judge_assigned", "doubled_lines_dropped",
               "lines_without_room_dropped", "audit_lines_dropped", "audit_lines_restored", "audit_marginalia")
#: Gate flags that mean the judges changed the text.
EDIT_FLAGS = ("audit_lines_dropped", "audit_lines_restored")
#: Split names of exports and manifests -> the audit's names.
SPLITS = {"train": "train", "train_page": "train", "val": "val"}
DEFAULT_SHA_COLUMN = "image_sha1"


# ----------------------------------------------------------------------------- files


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes.

    :param path: File path.
    :return: Hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def text_sha256(text: str) -> str:
    """SHA-256 of a text (UTF-8).

    :param text: Text.
    :return: Hex digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file.

    :param path: File path.
    :return: The records.
    """
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def export_row_files(export_dir: Path) -> List[Path]:
    """The row files of an export (``rows/*.parquet``), sorted.

    :param export_dir: Export folder (an images-once export or a training export).
    :return: Paths.
    :raises FileNotFoundError: When the export has no row file.
    """
    files = sorted((export_dir / "rows").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"{export_dir}: no rows/*.parquet")
    return files


def dedupe_export(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """One record per stem of an export's rows (a training export repeats a page once per pass).

    :param rows: Export rows with ``stem``, ``answer``, ``split``, ``image_sha1`` and ``image_path``.
    :return: The first row of each stem, in stem order.
    :raises ValueError: When two rows of one stem differ in answer, image or split.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        seen = out.get(row["stem"])
        if seen is None:
            out[row["stem"]] = dict(row)
        elif (seen["answer"], seen["image_sha1"], seen["split"]) != (row["answer"], row["image_sha1"], row["split"]):
            raise ValueError(f"export rows of stem {row['stem']} differ in answer, image or split")
    return [out[s] for s in sorted(out)]


def load_export(export_dir: Path) -> List[Dict[str, Any]]:
    """The distinct rows of an export, as they will be trained on.

    :param export_dir: Export folder: ``rows/<split>.parquet`` (``train``, ``train_page`` or ``val``) and
        ``images/<image_sha1>.jpg`` (images stored once).
    :return: ``[{stem, answer, split, image_sha1, image_path}]`` (:func:`dedupe_export`).
    :raises ValueError: On an unknown split file name.
    """
    import pyarrow.parquet as pq

    manifest = export_dir / "manifest.json"
    sha_column = DEFAULT_SHA_COLUMN
    if manifest.is_file():
        sha_column = json.loads(manifest.read_text(encoding="utf-8")).get("sha_column") or DEFAULT_SHA_COLUMN
    rows: List[Dict[str, Any]] = []
    for path in export_row_files(export_dir):
        if path.stem not in SPLITS:
            raise ValueError(f"{path}: unknown split {path.stem!r} (known: {sorted(SPLITS)})")
        for r in pq.read_table(path, columns=["stem", "answer", sha_column]).to_pylist():
            rows.append({"stem": r["stem"], "answer": r["answer"], "split": SPLITS[path.stem],
                         "image_sha1": r[sha_column], "image_path": str(export_dir / "images" / f"{r[sha_column]}.jpg")})
    return dedupe_export(rows)


def answer_lines(answer: str) -> List[str]:
    """The lines a judge is shown: the exported answer's non-empty lines, unchanged.

    :param answer: Exported answer.
    :return: Lines.
    """
    return [line for line in answer.split("\n") if line.strip()]


def image_inventory(export_dir: Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """Hash every referenced JPEG and check the directory against the row identifiers.

    :param export_dir: Export containing the images directory.
    :param rows: Distinct rows loaded from this export.
    :return: Exact filename-to-SHA-256 mapping.
    :raises ValueError: If a JPEG is missing, extra, linked or stored under an incorrect content ID.
    """
    folder = export_dir / "images"
    expected = {f"{row['image_sha1']}.jpg" for row in rows}
    if not folder.is_dir() or {p.name for p in folder.iterdir()} != expected:
        raise ValueError("the image directory has missing or extra files")
    inventory = {}
    for name in sorted(expected):
        path = folder / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"expected a regular image file: {name}")
        data = path.read_bytes()
        if hashlib.sha1(data).hexdigest() != path.stem:
            raise ValueError(f"image content differs from its row identifier: {name}")
        inventory[name] = hashlib.sha256(data).hexdigest()
    return inventory


# ----------------------------------------------------------------------------- strata


def approval_kind(record: Mapping[str, Any]) -> str:
    """Who approved a manifest row.

    :param record: Manifest record (``approval``, ``gate_flags``).
    :return: ``rule``, ``source_review``, ``judges``, ``judges_edit`` (the judges removed or put back lines) or ``unapproved`` (the
        record carries no approval: a build older than approval binding).
    """
    by = (record.get("approval") or {}).get("by")
    if by in ("rule", "source_review"):
        return by
    if by == "judges":
        return "judges_edit" if set(record.get("gate_flags") or []) & set(EDIT_FLAGS) else "judges"
    return "unapproved"


def script_class(record: Mapping[str, Any], answer: str, cand: Optional[Mapping[str, Any]] = None) -> str:
    """Reporting class of a row, as ``audit/make_items.script_class`` decides it for a pair.

    :param record: Manifest record (``script``, ``ja_signal``).
    :param answer: Exported answer.
    :param cand: The row's candidate record, when known (its screen class ``notes.class``).
    :return: ``arabic``, ``ja`` or ``heb``.
    """
    if record.get("script") == "arabic":
        return "arabic"
    cls = ((cand or {}).get("notes") or {}).get("class") or ""
    if cls.startswith("judaeo") or record.get("ja_signal") or textnorm.ja_signal(answer):
        return "ja"
    return "heb"


def facets_of(row: Mapping[str, Any]) -> List[str]:
    """Every facet of a sample row.

    :param row: Joined row (:func:`join`).
    :return: Facet names, sorted.
    """
    m = row["manifest"]
    out = {f"split:{row['split']}", f"approval:{row['approval_kind']}", f"remedy:{row['remedy_class']}",
           f"script:{row['script_class']}", f"source:{m.get('source')}"}
    out |= {f"flag:{f}" for f in m.get("gate_flags") or [] if f in FLAG_FACETS}
    if (m.get("open_lines_settled") or 0) > 0:
        out.add("flag:open_lines_settled")
    if (m.get("scrambled_lines") or 0) > 0:
        out.add("flag:scrambled_lines")
    if (m.get("clean_changes") or 0) > 0:
        out.add("flag:text_cleaned")
    return sorted(out)


def join(manifest: Sequence[Mapping[str, Any]], export: Sequence[Mapping[str, Any]],
         cands: Optional[Mapping[str, Mapping[str, Any]]] = None) -> List[Dict[str, Any]]:
    """The export's rows with their manifest records and strata.

    :param manifest: Build manifest records.
    :param export: Distinct export rows (:func:`load_export`).
    :param cands: Candidate records by row id (optional; only for :func:`script_class`).
    :return: One record per export row: the export fields, ``manifest`` (the record), ``approval_kind``,
        ``remedy_class``, ``script_class``, ``source``, ``document_group``, ``stratum``, ``facets``.
    :raises ValueError: When an export stem is not in the manifest, or its split differs from the manifest's.
    """
    by_stem = {m["stem"]: m for m in manifest}
    out: List[Dict[str, Any]] = []
    for row in export:
        m = by_stem.get(row["stem"])
        if m is None:
            raise ValueError(f"export stem {row['stem']} is not in the manifest: the export is not from this build")
        if SPLITS.get(m.get("split"), m.get("split")) != row["split"]:
            raise ValueError(f"stem {row['stem']}: export split {row['split']}, manifest split {m.get('split')}")
        joined = {**row, "manifest": dict(m), "approval_kind": approval_kind(m),
                  "remedy_class": m.get("remedy_class") or "unknown",
                  "script_class": script_class(m, row["answer"], (cands or {}).get(m.get("row_id"))),
                  "source": m.get("source"), "document_group": m.get("document_group") or row["stem"]}
        joined["stratum"] = f"{joined['approval_kind']}:{joined['remedy_class']}|{joined['script_class']}"
        joined["facets"] = facets_of(joined)
        out.append(joined)
    return out


def _ordered(rows: Sequence[Dict[str, Any]], seed: int, label: str) -> List[Dict[str, Any]]:
    """Rows in a seeded random order (stem order first, so the input order does not matter).

    :param rows: Rows.
    :param seed: Audit seed.
    :param label: Stratum or facet name (each gets its own generator).
    :return: The rows, shuffled.
    """
    out = sorted(rows, key=lambda r: r["stem"])
    random.Random(f"{seed}|{label}").shuffle(out)
    return out


def _take(pool: Sequence[Dict[str, Any]], n: int, avoid_groups: Set[str]) -> List[Dict[str, Any]]:
    """Up to ``n`` rows of a pool, rows of document groups not yet taken first.

    :param pool: Rows in draw order.
    :param n: Rows wanted.
    :param avoid_groups: Document groups already represented (not changed).
    :return: The rows taken: first one per new group, then (only when the new groups run out) the others.
    """
    take: List[Dict[str, Any]] = []
    groups = set(avoid_groups)
    for row in pool:
        if len(take) >= n:
            return take
        if row["document_group"] not in groups:
            take.append(row)
            groups.add(row["document_group"])
    stems = {row["stem"] for row in take}
    for row in pool:
        if len(take) >= n:
            break
        if row["stem"] not in stems:
            take.append(row)
            stems.add(row["stem"])
    return take


def draw(manifest: Sequence[Mapping[str, Any]], export: Sequence[Mapping[str, Any]],
         per_stratum: int = DEFAULT_PER_STRATUM, seed: int = DEFAULT_SEED, min_items: int = DEFAULT_MIN_ITEMS,
         per_facet: int = DEFAULT_PER_FACET, cands: Optional[Mapping[str, Mapping[str, Any]]] = None
         ) -> List[Dict[str, Any]]:
    """The release sample of an export (see the module docstring for the design).

    :param manifest: Build manifest records.
    :param export: Distinct export rows (:func:`load_export`): the sample's lines and images are the export's.
    :param per_stratum: Rows per primary stratum.
    :param seed: Seed; the same inputs and seed give the same sample.
    :param min_items: The sample is topped up to at least this many rows (fewer only when the export is smaller).
    :param per_facet: Fewest rows of each facet in the sample (fewer only when the export has fewer).
    :param cands: Candidate records by row id (optional, for the script class).
    :return: Sample rows (:func:`join` records plus ``drawn_for``: ``stratum``, ``facet:<name>`` or ``topup``), in
        the seeded audit order.
    """
    rows = join(manifest, export, cands)
    chosen: List[Dict[str, Any]] = []
    taken: Set[str] = set()

    def add(picked: Sequence[Dict[str, Any]], why: str) -> None:
        """Put rows into the sample.

        :param picked: Rows taken.
        :param why: What they were drawn for.
        """
        for row in picked:
            chosen.append({**row, "drawn_for": why})
            taken.add(row["stem"])

    strata: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[row["stratum"]].append(row)
    for name in sorted(strata):
        add(_take(_ordered(strata[name], seed, f"stratum:{name}"), per_stratum, set()), "stratum")
    facets = sorted({f for row in rows for f in row["facets"]})
    for facet in facets:
        have = [r for r in chosen if facet in r["facets"]]
        if len(have) >= per_facet:
            continue
        pool = [r for r in rows if facet in r["facets"] and r["stem"] not in taken]
        add(_take(_ordered(pool, seed, f"facet:{facet}"), per_facet - len(have),
                  {r["document_group"] for r in have}), f"facet:{facet}")
    if len(chosen) < min_items:
        pool = [r for r in rows if r["stem"] not in taken]
        add(_take(_ordered(pool, seed, "topup"), min_items - len(chosen),
                  {r["document_group"] for r in chosen}), "topup")
    return _ordered(chosen, seed, "order")


# ----------------------------------------------------------------------------- judge items and key


def render(src: Path, out: Path, long_side: int = LONG_SIDE) -> None:
    """Save the judge's copy of an exported image (long side capped, as ``audit/make_items.render``).

    The exported image already has its crop and rotation applied, so nothing else is done to it.

    :param src: Exported image.
    :param out: JPEG path.
    :param long_side: Longest side of the copy.
    """
    with Image.open(src) as im:
        im = im.convert("RGB")
        if max(im.size) > long_side:
            r = long_side / max(im.size)
            im = im.resize((round(im.width * r), round(im.height * r)), Image.Resampling.LANCZOS)
        im.save(out, quality=90)


def item_and_key(row: Mapping[str, Any], item_id: str, image_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The judge item and the hidden key line of one sample row.

    :param row: Sample row (:func:`draw`).
    :param item_id: Item id.
    :param image_path: The judge's copy of the image.
    :return: ``(item, key)``. The item has exactly the contract ``v2`` fields (``open_lines`` empty: an exported row
        has no unresolved lines beside it). The key carries what ``score.py`` and :func:`adjudicate.score_item` read
        and the strata.
    """
    m = row["manifest"]
    item = {"item_id": item_id, "image_path": str(image_path), "lines": answer_lines(row["answer"]), "open_lines": []}
    key = {"item_id": item_id, "pair_id": m.get("pair_id"), "stem": row["stem"], "row_id": m.get("row_id"),
           "stratum": row["stratum"], "decision": "accept", "letters": textnorm.letters(row["answer"]),
           "contract": audit_contract.CONTRACT, "fingerprint": (m.get("approval") or {}).get("fingerprint"),
           "split": row["split"], "approval_kind": row["approval_kind"], "remedy_class": row["remedy_class"],
           "script_class": row["script_class"], "source": row["source"], "document_group": row["document_group"],
           "facets": list(row["facets"]), "drawn_for": row["drawn_for"], "image_sha1": row["image_sha1"],
           "answer_sha256": text_sha256(row["answer"]), "export_image_sha256": file_sha256(Path(row["image_path"]))}
    return item, key


def write_audit(sample: Sequence[Mapping[str, Any]], name: str, items_path: Path, key_path: Path,
                image_dir: Path) -> List[Dict[str, Any]]:
    """Write the judges' items, their image copies and the hidden key of a release audit.

    :param sample: Sample rows (:func:`draw`), in audit order.
    :param name: Audit name (item id prefix).
    :param items_path: ``judge_items_<name>.jsonl`` (beside the judges' files).
    :param key_path: ``<name>_key.jsonl`` (out of the judges' reach).
    :param image_dir: Folder for the judges' image copies (must be absent or empty).
    :return: The key lines.
    :raises FileExistsError: When the items, the key or images of an audit of this name exist.
    """
    if items_path.exists() or key_path.exists() or (image_dir.exists() and any(image_dir.iterdir())):
        raise FileExistsError(f"audit {name} exists: its items, key and images are the record of what was judged")
    image_dir.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    items_path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[Dict[str, Any]] = []
    with open(items_path, "w", encoding="utf-8") as items, open(key_path, "w", encoding="utf-8") as key_fh:
        for k, row in enumerate(sample, 1):
            item_id = f"{name}_{k:03d}"
            out = image_dir / f"{item_id}.jpg"
            render(Path(row["image_path"]), out)
            item, key = item_and_key(row, item_id, out)
            items.write(json.dumps(item, ensure_ascii=False) + "\n")
            key_fh.write(json.dumps(key, ensure_ascii=False) + "\n")
            keys.append(key)
    return keys


def draw_record(name: str, manifest_path: Path, export_dir: Path, keys: Sequence[Mapping[str, Any]],
                per_stratum: int, per_facet: int, seed: int, min_items: int) -> Dict[str, Any]:
    """What a release sample was drawn from and how (written at draw time, checked at verify time).

    :param name: Audit name.
    :param manifest_path: The build manifest.
    :param export_dir: The audited export.
    :param keys: The key lines (:func:`write_audit`).
    :param per_stratum: Draw parameter.
    :param per_facet: Draw parameter.
    :param seed: Draw parameter.
    :param min_items: Draw parameter (also the pass rule's minimum judged items).
    :return: The record.
    """
    return {"name": name, "manifest_path": str(manifest_path), "manifest_sha256": file_sha256(manifest_path),
            "export_dir": str(export_dir), "export_files": file_digests(export_row_files(export_dir)),
            "image_inventory": image_inventory(export_dir, load_export(export_dir)),
            "per_stratum": per_stratum, "per_facet": per_facet, "seed": seed, "min_items": min_items,
            "threshold": THRESHOLD, "stems": [k["stem"] for k in keys], "item_ids": [k["item_id"] for k in keys],
            "drawn": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def file_digests(paths: Sequence[Path]) -> List[Dict[str, str]]:
    """Name, path and SHA-256 of files.

    :param paths: Files.
    :return: ``[{"name", "path", "sha256"}]``.
    """
    return [{"name": p.name, "path": str(p.resolve()), "sha256": file_sha256(p)} for p in paths]


def check_draw(record: Mapping[str, Any], manifest: Sequence[Mapping[str, Any]], export: Sequence[Mapping[str, Any]],
               manifest_path: Path, export_dir: Path, keys: Sequence[Mapping[str, Any]],
               cands: Optional[Mapping[str, Mapping[str, Any]]] = None) -> List[str]:
    """Why a judged sample is not the release sample of this export (empty: it is).

    The manifest and export files must be unchanged since the draw, the key must list the drawn stems, and drawing
    again with the recorded parameters must give the same stems (no hand-picked sample).

    :param record: The draw record (:func:`draw_record`).
    :param manifest: Manifest records.
    :param export: Distinct export rows.
    :param manifest_path: The manifest file.
    :param export_dir: The export folder.
    :param keys: The key lines.
    :param cands: Candidates by row id, as given to the draw.
    :return: Problems.
    """
    problems: List[str] = []
    if file_sha256(manifest_path) != record["manifest_sha256"]:
        problems.append("the manifest changed since the draw")
    now = {d["name"]: d["sha256"] for d in file_digests(export_row_files(export_dir))}
    then = {d["name"]: d["sha256"] for d in record["export_files"]}
    if now != then:
        problems.append("the export's row files changed since the draw")
    if image_inventory(export_dir, export) != record.get("image_inventory"):
        problems.append("the full image inventory differs from the draw")
    if [k["item_id"] for k in keys] != record["item_ids"] or [k["stem"] for k in keys] != record["stems"]:
        problems.append("the key does not list the drawn items and stems")
    again = draw(manifest, export, record["per_stratum"], record["seed"], record["min_items"], record["per_facet"],
                 cands)
    if [r["stem"] for r in again] != list(record["stems"]):
        problems.append("drawing again with the recorded parameters gives another sample")
    for row, key in zip(again, keys):
        image_sha = row["image_sha1"]
        if (key.get("answer_sha256") != text_sha256(row["answer"]) or key.get("image_sha1") != image_sha
                or key.get("export_image_sha256") != record.get("image_inventory", {}).get(f"{image_sha}.jpg")):
            problems.append(f"the key answer or photo differs for {row['stem']}")
    return problems


# ----------------------------------------------------------------------------- verdict


def wilson_lower(successes: int, n: int, z: float = Z_ONE_SIDED_95) -> float:
    """One-sided Wilson lower confidence bound of a proportion.

    :param successes: Successes.
    :param n: Trials.
    :param z: Normal quantile (default: one-sided 95 %).
    :return: The bound (0.0 when ``n`` is 0).
    """
    if n <= 0:
        return 0.0
    p = successes / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / (1 + z * z / n))


def _tally(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Counts and rates of a group of results.

    :param results: Result records (:func:`adjudicate.score_item`).
    :return: ``{judged, cannot_judge, complete, rate, wilson_lower_95}``; ``rate`` = complete / (judged +
        cannot_judge).
    """
    judged, complete = adjudicate.strict_rate(results)
    with_record = sum(1 for r in results if adjudicate.judge_records(r))
    unable = with_record - judged
    denominator = judged + unable
    return {"judged": judged, "cannot_judge": unable, "complete": complete,
            "rate": complete / denominator if denominator else 0.0,
            "wilson_lower_95": wilson_lower(complete, denominator)}


def verify(results: Sequence[Mapping[str, Any]], key: Sequence[Mapping[str, Any]], threshold: float = THRESHOLD,
           min_items: int = DEFAULT_MIN_ITEMS, min_judges: Optional[int] = None) -> Dict[str, Any]:
    """The verdict of a release audit.

    Counting. An item is complete only under :func:`adjudicate.strict_rate`: every judge who saw it says
    ``complete_pair`` and none reports a concern. An item no judge could judge (``cannot_judge`` from every judge) is
    reported and counts as NOT complete: ``rate`` = complete / (judged + cannot_judge). ``judged`` counts the items a
    judge could judge (:func:`adjudicate.strict_rate`).

    Pass rule: every key item has a judge record, every judged item was seen by at least ``min_judges`` judges (as
    many as an approval needs, :data:`adjudicate.JUDGES_FOR_APPROVAL`: one judge's "complete" was shared by a second
    judge in half of the cases of the 2026-11-19 audits, so one judge's word releases nothing),
    ``judged >= min_items`` and the point estimate ``rate >= threshold``. The one-sided 95 % Wilson lower bound is
    reported (overall and per stratum and facet) but is not part of the pass rule.

    :param results: Result records of the judged items (:func:`adjudicate.score_item`).
    :param key: The audit's key lines (every drawn item).
    :param threshold: The target (:data:`THRESHOLD`).
    :param min_items: Fewest judged items.
    :param min_judges: Fewest judges per item (None: :data:`adjudicate.JUDGES_FOR_APPROVAL`).
    :return: ``{pass, reasons, items, unjudged, under_judged, cannot_judge_items, judged, complete, rate,
        wilson_lower_95, threshold, min_items, min_judges, concerns, by_stratum, by_facet, by_split,
        by_approval_kind, by_remedy_class, by_script_class, by_source}``.
    """
    need = adjudicate.JUDGES_FOR_APPROVAL if min_judges is None else min_judges
    by_id = {r["item_id"]: r for r in results}
    keyed = [k for k in key if k["item_id"] in by_id and adjudicate.judge_records(by_id[k["item_id"]])]
    keyed_ids = {k["item_id"] for k in keyed}
    unjudged = sorted(k["item_id"] for k in key if k["item_id"] not in keyed_ids)
    scored = [by_id[k["item_id"]] for k in keyed]
    total = _tally(scored)
    unable = sorted(r["item_id"] for r in scored
                    if {rec["verdict"] for rec in adjudicate.judge_records(r).values()} == {"cannot_judge"})
    concern_counts: Counter = Counter(                      # by the item's own contract, as the pass rule reads it
        c for r in scored for rec in adjudicate.judge_records(r).values()
        for c in adjudicate.concerns(rec, "v2" if r.get("contract") == "v2" else "v1"))
    groups: Dict[str, Dict[str, List[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for k in keyed:
        r = by_id[k["item_id"]]
        groups["by_stratum"][k["stratum"]].append(r)
        for facet in k.get("facets") or []:
            groups["by_facet"][facet].append(r)
        for field in ("split", "approval_kind", "remedy_class", "script_class", "source"):
            groups[f"by_{field}"][str(k.get(field))].append(r)
    under = sorted(r["item_id"] for r in scored if len(adjudicate.judge_records(r)) < need)
    reasons: List[str] = []
    if unjudged:
        reasons.append(f"{len(unjudged)} drawn items have no judge record")
    if under:
        reasons.append(f"{len(under)} items were judged by fewer than {need} judges")
    if total["judged"] < min_items:
        reasons.append(f"{total['judged']} items judged, fewer than the required {min_items}")
    if total["rate"] < threshold:
        reasons.append(f"{100 * total['rate']:.1f} % complete without any concern, under the required "
                       f"{100 * threshold:.0f} %")
    out: Dict[str, Any] = {"pass": not reasons, "reasons": reasons, "items": len(key), "unjudged": unjudged,
                           "under_judged": under, "cannot_judge_items": unable, **total, "threshold": threshold,
                           "min_items": min_items, "min_judges": need,
                           "concerns": dict(sorted(concern_counts.items()))}
    for group in ("by_stratum", "by_facet", "by_split", "by_approval_kind", "by_remedy_class", "by_script_class",
                  "by_source"):
        out[group] = {name: _tally(members) for name, members in sorted(groups[group].items())}
    return out


# ----------------------------------------------------------------------------- marker


def marker(name: str, verdict: Mapping[str, Any], key: Sequence[Mapping[str, Any]], manifest_path: Path,
           export_files: Sequence[Path], record: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Build a release or diagnostic report with its manifest, rows and full image inventory.

    :param name: Audit name.
    :param verdict: :func:`verify` result.
    :param key: The key lines.
    :param manifest_path: The build manifest.
    :param export_files: The audited export's row files.
    :param record: The draw record (its parameters are copied), or None.
    :return: The marker.
    """
    scope = "release" if (verdict["min_items"] >= DEFAULT_MIN_ITEMS and verdict["threshold"] >= THRESHOLD
                          and verdict["min_judges"] >= adjudicate.JUDGES_FOR_APPROVAL) else "diagnostic"
    export_dir = export_files[0].parent.parent
    images = record["image_inventory"] if record else image_inventory(export_dir, load_export(export_dir))
    return {"version": MARKER_VERSION, "scope": scope, "audit": name, "pass": bool(verdict["pass"]),
            "verdict": dict(verdict), "image_inventory": images,
            "item_ids": [k["item_id"] for k in key], "stems": [k["stem"] for k in key],
            "items": [{"item_id": k["item_id"], "stem": k["stem"], "answer_sha256": k.get("answer_sha256"),
                       "image_sha1": k.get("image_sha1"), "export_image_sha256": k.get("export_image_sha256")}
                      for k in key],
            "manifest_path": str(manifest_path), "manifest_sha256": file_sha256(manifest_path),
            "export_files": file_digests(export_files), "bound_exports": [],
            "draw": {k: (record or {}).get(k) for k in ("seed", "per_stratum", "per_facet", "min_items", "export_dir")},
            "code_sha256": file_sha256(Path(__file__)),
            "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def require_production_verdict(content: Mapping[str, Any]) -> None:
    """Require a passing production audit with distinct photographed items and the standing quality floor.

    :param content: The stored marker or a proposed passing marker.
    :raises SystemExit: If the audit, counts, sample identities or image evidence are insufficient.
    """
    verdict = content.get("verdict") or {}
    if not content.get("pass") or not verdict.get("pass"):
        reasons = "; ".join(verdict.get("reasons") or []) or "no pass recorded"
        raise SystemExit(f"NOT RELEASED: release audit {content.get('audit')} failed: {reasons}")
    if content.get("version") != MARKER_VERSION or content.get("scope") != "release":
        raise SystemExit("NOT RELEASED: a current production release audit is required")
    judged, complete = verdict.get("judged", 0), verdict.get("complete", 0)
    items = content.get("items") or []
    n = len(items)
    if (verdict.get("min_items", 0) < DEFAULT_MIN_ITEMS or judged < verdict["min_items"]
            or verdict.get("threshold", 0) < THRESHOLD or verdict.get("min_judges", 0) < adjudicate.JUDGES_FOR_APPROVAL
            or verdict.get("unjudged") != [] or verdict.get("under_judged") != [] or verdict.get("items") != n
            or not 0 <= complete <= judged <= n or judged + verdict.get("cannot_judge", 0) != n
            or complete / n < verdict["threshold"]):
        raise SystemExit("NOT RELEASED: the production audit size, judge coverage or completeness floor was not met")
    images = content.get("image_inventory") or {}
    for field in ("item_id", "stem", "image_sha1"):
        values = {item.get(field) for item in items}
        if None in values or len(values) != n:
            raise SystemExit(f"NOT RELEASED: repeated or absent audit {field}")
    if any(not item.get("export_image_sha256") or
           images.get(f"{item['image_sha1']}.jpg") != item["export_image_sha256"] for item in items):
        raise SystemExit("NOT RELEASED: an audited item has no matching photo in the image inventory")


def write_marker(build_dir: Path, content: Mapping[str, Any]) -> Path:
    """Write the production or diagnostic marker into a build's report folder.

    :param build_dir: The build's report folder (``<dataset>_build``, beside ``manifest.jsonl``).
    :param content: :func:`marker`.
    :return: The marker path.
    :raises FileExistsError: When a marker exists (a failed export is not audited again until it passes).
    """
    if content.get("scope") == "diagnostic":
        path = build_dir / "DIAGNOSTIC_AUDIT.json"
    else:
        path = build_dir / MARKER_NAME
        if content.get("pass"):
            require_production_verdict(content)
    with path.open("x", encoding="utf-8") as fh:
        fh.write(json.dumps(content, indent=1, ensure_ascii=False) + "\n")
    return path


def same_content(audited: Sequence[Mapping[str, Any]], other: Sequence[Mapping[str, Any]]) -> List[str]:
    """Why another export of the build does not hold exactly the audited export's rows (empty: it does).

    :param audited: Distinct rows of the audited export.
    :param other: Distinct rows of the other export (for example the images-once source set of a training export).
    :return: Problems: stems present in only one, or a stem whose answer, image or split differ.
    """
    a = {r["stem"]: r for r in audited}
    b = {r["stem"]: r for r in other}
    problems = []
    if set(a) != set(b):
        problems.append(f"{len(set(a) - set(b))} audited stems missing and {len(set(b) - set(a))} stems added")
    differ = sorted(s for s in set(a) & set(b)
                    if (a[s]["answer"], a[s]["image_sha1"], a[s]["split"]) != (b[s]["answer"], b[s]["image_sha1"],
                                                                                 b[s]["split"]))
    if differ:
        problems.append(f"{len(differ)} stems differ in answer, image or split, e.g. {differ[:3]}")
    return problems


def complete_row_files(files: Sequence[Path]) -> Path:
    """Require one complete export directory and return its root.

    :param files: Paths supplied by the caller.
    :return: The export directory.
    :raises SystemExit: If any split is absent, extra, repeated or supplied from another export.
    """
    if not files:
        raise SystemExit("NOT RELEASED: no export file given to check against the release audit")
    paths = [p.resolve() for p in files]
    root = paths[0].parent.parent
    actual = sorted((root / "rows").glob("*.parquet"))
    if sorted(paths) != actual or not all(p.is_file() for p in paths):
        raise SystemExit("NOT RELEASED: supply all splits of one export together")
    return root


def check_bound_splits(content: Mapping[str, Any], paths: Sequence[Path]) -> None:
    """Compare complete row-file signatures against one bound layout.

    :param content: Release marker.
    :param paths: Current split files.
    :raises SystemExit: If these splits do not match one entire binding.
    """
    current = {p.name: file_sha256(p) for p in paths}
    bindings = [content["export_files"]] + [b["files"] for b in content.get("bound_exports", [])]
    if not any(current == {d["name"]: d["sha256"] for d in files} for files in bindings):
        raise SystemExit("NOT RELEASED: changed splits or a mixture of exports")


def bind_export(build_dir: Path, content: Mapping[str, Any], audited: Sequence[Mapping[str, Any]],
                other: Sequence[Mapping[str, Any]], other_files: Sequence[Path]) -> Dict[str, Any]:
    """Bind a further export of identical content to a passing marker (rewrites the marker with the binding added).

    Every image in the other export must match the recorded inventory.

    :param build_dir: The build's report folder.
    :param content: The current marker.
    :param audited: Distinct rows of the audited export.
    :param other: Distinct rows of the other export.
    :param other_files: The other export's row files.
    :return: The new marker.
    :raises SystemExit: When the marker did not pass or the other export holds other rows or images.
    """
    source = [Path(d["path"]) for d in content["export_files"]]
    current = require_release(build_dir, source)
    if current["audit"] != content["audit"]:
        raise SystemExit("REFUSED: the release marker changed")
    root = complete_row_files(other_files)
    problems = same_content(audited, other)
    other_rows = load_export(root)
    problems += same_content(load_export(source[0].parent.parent), other_rows)
    if image_inventory(root, other_rows) != current["image_inventory"]:
        problems.append("the full image inventory differs from the audited one")
    if problems:
        raise SystemExit("REFUSED: the export does not hold exactly the audited rows: " + "; ".join(problems))
    binding = file_digests(other_files)
    path = build_dir / MARKER_NAME
    with (build_dir / ".release_audit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        new = json.loads(path.read_text(encoding="utf-8"))
        if any(b["files"] == binding for b in new["bound_exports"]):
            return new
        new["bound_exports"].append({"files": binding, "bound": datetime.now(timezone.utc).isoformat()})
        with tempfile.NamedTemporaryFile(mode="w", dir=build_dir, encoding="utf-8", delete=False) as tmp:
            tmp.write(json.dumps(new, indent=1, ensure_ascii=False) + "\n")
        Path(tmp.name).replace(path)
        return new


def require_release(build_dir: Path, export_files: Sequence[Path]) -> Dict[str, Any]:
    """Refuse to go on unless a passing release audit covers exactly these export files and the build's manifest.

    :param build_dir: The build's report folder (``RELEASE_AUDIT.json`` beside ``manifest.jsonl``).
    :param export_files: The row files about to be pushed or trained on.
    :return: The marker.
    :raises SystemExit: When the production verdict, manifest, complete set of splits or image inventory differs.
    """
    path = Path(build_dir) / MARKER_NAME
    if not path.is_file():
        raise SystemExit(f"NOT RELEASED: no release audit at {path}. Draw and judge one "
                         "(audit/release_audit.py draw / verify) before any push or training.")
    content = json.loads(path.read_text(encoding="utf-8"))
    require_production_verdict(content)
    manifest = Path(build_dir) / "manifest.jsonl"
    if not manifest.is_file() or file_sha256(manifest) != content.get("manifest_sha256"):
        raise SystemExit(f"NOT RELEASED: {manifest} is not the manifest release audit {content.get('audit')} judged")
    root = complete_row_files(export_files)
    check_bound_splits(content, export_files)
    try:
        inventory = image_inventory(root, load_export(root))
    except ValueError as exc:
        raise SystemExit(f"NOT RELEASED: {exc}") from exc
    if inventory != content["image_inventory"]:
        raise SystemExit("NOT RELEASED: the image bytes differ from the release audit")
    return content


def verify_staged_export(stage: Path, archives: Sequence[str], content: Mapping[str, Any]) -> Dict[str, str]:
    """Compare staged rows and archived JPEG bytes with the bound export.

    :param stage: Staging root holding the row files and image archives.
    :param archives: Archive filenames to be uploaded.
    :param content: Passing marker returned by :func:`require_release`.
    :return: SHA-256 digests of the verified archives.
    :raises ValueError: On any missing, extra, duplicated or changed archived image.
    """
    check_bound_splits(content, export_row_files(stage))
    if sorted(archives) != sorted(p.name for p in stage.glob("*.tar")):
        raise ValueError("staged image archives differ from the packing list")
    expected = {f"images/{name}": digest for name, digest in content["image_inventory"].items()}
    seen: Set[str] = set()
    receipts = {}
    for name in archives:
        path = stage / name
        with tarfile.open(path, "r:") as archive:
            for entry in archive:
                if not entry.isfile() or entry.name not in expected or entry.name in seen:
                    raise ValueError(f"unexpected archived image: {entry.name}")
                with archive.extractfile(entry) as fh:
                    digest = hashlib.sha256()
                    for block in iter(lambda: fh.read(1 << 20), b""):
                        digest.update(block)
                if digest.hexdigest() != expected[entry.name]:
                    raise ValueError(f"changed archived image: {entry.name}")
                seen.add(entry.name)
        receipts[name] = file_sha256(path)
    if seen != set(expected):
        raise ValueError("an audited image is missing from the archives")
    return receipts
