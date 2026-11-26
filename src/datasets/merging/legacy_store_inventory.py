#!/usr/bin/env python3
"""Inventory of the legacy store's transcribed shelf marks, joined against what we hold.

The legacy store (a third-party manuscript catalogue; never cited or linked on
a public surface) lists, per collection, which items carry a
transcription. This module turns listing captures saved by a browser session
into one inventory JSONL, joins every shelf mark against the merged corpus
(``merged/merged_shelfmarks.jsonl``) and reports, per collection, what is on
the site but in neither our KTIV transcription scrape nor the PGP editions.

Inventory only: no transcription text is read or stored here.

Inputs
------
Either a hand-made list (the normal route: the site is never crawled) or
browser listing captures.

* A list as CSV or JSONL, one row per item, with the column ``shelf_mark``
  (as printed on the site) and any of ``collection``, ``language``,
  ``has_transcription`` (yes/no, true/false, 1/0; absent = yes),
  ``transcription_length``, ``item_url``, ``item_id``, ``listing_url``,
  ``captured_at``. Unknown columns are ignored.
* ``captures/*.json`` under :data:`INVENTORY_DIR`, one file per listing page::

    {"listing_url": "...", "collection": "...", "captured_at": "...",
     "items": [{"shelf_mark": "T-S 8J5.1", "item_url": "...",
                "has_transcription": true, "transcription_length": 1234,
                "language": "Judaeo-Arabic", "item_id": "..."}]}

Outputs (all under :data:`INVENTORY_DIR`)
-----------------------------------------
* ``inventory.jsonl``: one line per catalogue item (:data:`INVENTORY_FIELDS`).
* ``legacy_store_join.csv``: every inventory item with its match against the corpus.
* ``legacy_store_only_transcribed.csv``: transcribed on the site, in neither our KTIV
  transcription set nor a PGP edition.
* ``join_summary.json``: the per-collection counts.

Run::

    python -m src.datasets.merging.legacy_store_inventory join --input my_list.csv
    python -m src.datasets.merging.legacy_store_inventory build   # captures only
    python -m src.datasets.merging.legacy_store_inventory join    # inventory.jsonl
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re
import sys
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.datasets import shelfmarks  # noqa: E402
from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer  # noqa: E402
from src.datasets.merging.institution_tokens import resolve_token  # noqa: E402
from src.datasets.merging.merge_shelfmarks import (  # noqa: E402
    DEFAULT_OUT_DIR,
    MERGED_JSONL,
    RAW_DIR,
)

INVENTORY_DIR = os.path.join(RAW_DIR, "legacy_store_inventory_2026-10")
CAPTURES_GLOB = os.path.join(INVENTORY_DIR, "captures", "*.json")
INVENTORY_FILE = "inventory.jsonl"
JOIN_CSV = "legacy_store_join.csv"
NOT_HELD_CSV = "legacy_store_only_transcribed.csv"
SUMMARY_FILE = "join_summary.json"
MERGED_PATH = os.path.join(DEFAULT_OUT_DIR, MERGED_JSONL)
# The merge's key for the legacy store under ``sources`` / ``sources_present``.
LEGACY_SOURCE_KEY = "fjp"

INVENTORY_FIELDS: Tuple[str, ...] = (
    "shelf_mark",
    "collection",
    "language",
    "has_transcription",
    "transcription_length",
    "item_url",
    "item_id",
    "listing_url",
    "captured_at",
)

# Match statuses, from "we already hold a transcription" to "never seen".
STATUS_BOTH = "ktiv_and_pgp"
STATUS_KTIV = "ktiv_transcription"
STATUS_PGP = "pgp_edition"
STATUS_LEGACY_ONLY = "legacy_scrape_only"       # our own legacy-store scrape has text, no KTIV/PGP
STATUS_CORPUS = "corpus_no_transcription"  # known fragment, no transcription anywhere
STATUS_UNMATCHED = "unmatched"

# Canonical-id tokens that name the holder rather than the item, or spell one
# series two ways across sources (KTIV ``EVR ARAB II``, PGP/site ``Yevr.-Arab. II``).
_DROP_TOKENS: Set[str] = {"rnl", "nlr", "ms", "mss", "nli", "firkovich", "firkovitch", "firk"}
_TOKEN_SYNONYMS: Dict[str, str] = {"yevr": "evr", "ebr": "evr", "hebr": "heb"}
_WS_RE = re.compile(r"\s+")


# ───────────────────────────── join keys ─────────────────────────────

def clean_mark(mark: Optional[str]) -> str:
    """Collapse whitespace in a shelf mark as printed.

    :param mark: Raw shelf mark text (may be ``None``).
    :returns: The mark with runs of whitespace collapsed and ends trimmed.
    """
    return _WS_RE.sub(" ", mark or "").strip()


def join_key(mark: str) -> str:
    """Return the separator-, case- and holder-insensitive join key of a shelf mark.

    The mark goes through :meth:`ShelfmarkNormalizer.to_canonical_id` (which
    drops library long forms and ``City:`` prefixes), holder tokens are removed
    and series synonyms unified (:data:`_DROP_TOKENS`, :data:`_TOKEN_SYNONYMS`),
    then :meth:`ShelfmarkNormalizer.loose_key` keeps digit boundaries so
    ``AS 14.46`` and ``AS 144.6`` stay distinct.

    :param mark: A shelf mark in any source spelling.
    :returns: The key, or ``""`` when the mark normalises to nothing.
    """
    core = ShelfmarkNormalizer.to_canonical_id(clean_mark(mark))
    if not core:
        return ""
    tokens = []
    for tok in core.split("_"):
        low = tok.lower()
        if low in _DROP_TOKENS:
            continue
        tokens.append(_TOKEN_SYNONYMS.get(low, low))
    return ShelfmarkNormalizer.loose_key("_".join(tokens))


def record_marks(record: dict) -> Iterator[str]:
    """Yield every shelf-mark spelling a merged record is known by.

    Since PR-6 of the shelf-mark migration this is the shelf-mark facade's
    :func:`src.datasets.shelfmarks.record_marks` (the logic was lifted from
    here and made source-generic; :class:`src.datasets.shelfmarks.ShelfmarkIndex`
    indexes the same spellings).

    :param record: One ``merged_shelfmarks.jsonl`` line.
    :returns: Canonical id, display form, PGP primary and historic marks (also
        of extra fragment rows), every other source's single-mark strings (a
        multi-mark join string is skipped: its partners have records of their
        own), and the KTIV catalogue's mark.
    """
    yield from shelfmarks.record_marks(record)


def record_holdings(record: dict) -> dict:
    """Summarise what the corpus holds for one merged record.

    :param record: One ``merged_shelfmarks.jsonl`` line.
    :returns: ``canonical_id``, ``shelfmark_display``, ``sources_present``,
        ``pgp_edition`` (any attached PGP document has a transcription),
        ``pgp_languages``, ``ktiv_transcription`` (our viewer scrape holds
        text), ``ktiv_language`` and ``legacy_transcription`` (our legacy-store scrape
        holds text).
    """
    sources = record.get("sources") or {}
    pgp_docs = (sources.get("pgp") or {}).get("documents") or []
    languages: List[str] = []
    for doc in pgp_docs:
        for lang in (doc.get("languages_primary") or "").split(";"):
            if lang.strip() and lang.strip() not in languages:
                languages.append(lang.strip())
    ktiv = sources.get("ktiv") or {}
    return {
        "canonical_id": record.get("canonical_id") or "",
        "shelfmark_display": record.get("shelfmark_display") or "",
        "sources_present": list(record.get("sources_present") or []),
        "pgp_edition": any((doc.get("has_transcription") or "").upper() == "Y" for doc in pgp_docs),
        "pgp_languages": languages,
        "ktiv_transcription": bool(sources.get("ktiv_transcription")),
        "ktiv_language": (ktiv.get("basic_catalog") or {}).get("language"),
        "legacy_transcription": any(rec.get("transcriptions") for rec in sources.get(LEGACY_SOURCE_KEY) or []),
    }


def build_reference_index(merged_path: str = MERGED_PATH) -> Dict[str, List[dict]]:
    """Index the merged corpus by every join key each record answers to.

    :param merged_path: Path of ``merged_shelfmarks.jsonl``.
    :returns: ``join_key -> [holdings, ...]`` (:func:`record_holdings`); a key
        maps to several records only when distinct fragments share a spelling.
    """
    index: Dict[str, List[dict]] = collections.defaultdict(list)
    with open(merged_path, encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            holdings = record_holdings(record)
            keys = {join_key(mark) for mark in record_marks(record) if mark}
            keys.discard("")
            for key in keys:
                if all(h["canonical_id"] != holdings["canonical_id"] for h in index[key]):
                    index[key].append(holdings)
    return dict(index)


# ───────────────────────────── inventory ─────────────────────────────

_TRUE_WORDS = {"1", "y", "yes", "true", "t", "x"}
_FALSE_WORDS = {"0", "n", "no", "false", "f"}


def parse_flag(value: object, default: bool) -> bool:
    """Read a yes/no cell from a CSV or JSON value.

    :param value: ``True``/``False``, a number, or a word such as ``yes``,
        ``no``, ``1``, ``0``; ``None`` or a blank string means the cell was
        absent.
    :param default: Value for an absent or blank cell.
    :returns: The flag.
    :raises ValueError: For a word that is neither a yes nor a no.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    word = str(value).strip().lower()
    if not word:
        return default
    if word in _TRUE_WORDS:
        return True
    if word in _FALSE_WORDS:
        return False
    raise ValueError(f"not a yes/no value: {value!r}")


def parse_length(value: object) -> Optional[int]:
    """Read an optional transcription length.

    :param value: A number, a numeric string, or blank/``None``.
    :returns: The integer, or ``None`` when blank.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    word = str(value).strip().replace(",", "")
    return int(word) if word else None


def normalise_item(item: dict, defaults: Optional[dict] = None, absent_means_transcribed: bool = False) -> Optional[dict]:
    """Turn one listed item (capture entry or list row) into an inventory row.

    :param item: The item's fields; unknown keys are ignored.
    :param defaults: Capture-level values (``collection``, ``listing_url``,
        ``captured_at``) used when the item lacks them.
    :param absent_means_transcribed: Whether a missing ``has_transcription``
        counts as yes (true for a hand-made download list).
    :returns: The row (:data:`INVENTORY_FIELDS`), or ``None`` for a blank mark.
    """
    defaults = defaults or {}
    mark = clean_mark(item.get("shelf_mark"))
    if not mark:
        return None
    return {
        "shelf_mark": mark,
        "collection": clean_mark(item.get("collection") or defaults.get("collection") or ""),
        "language": clean_mark(item.get("language")) or None,
        "has_transcription": parse_flag(item.get("has_transcription"), absent_means_transcribed),
        "transcription_length": parse_length(item.get("transcription_length")),
        "item_url": clean_mark(item.get("item_url")) or None,
        "item_id": clean_mark(str(item.get("item_id") or "")) or None,
        "listing_url": item.get("listing_url") or defaults.get("listing_url") or None,
        "captured_at": item.get("captured_at") or defaults.get("captured_at") or None,
    }


def _dedupe_key(row: dict) -> str:
    """Identity of an item across captures: its URL, else collection + mark.

    :param row: An inventory row.
    :returns: The key.
    """
    return row["item_url"] or f"{row['collection']}::{row['shelf_mark']}"


def load_captures(pattern: str = CAPTURES_GLOB) -> List[dict]:
    """Flatten listing captures into inventory rows (:data:`INVENTORY_FIELDS`).

    Items are de-duplicated on ``item_url`` (or on collection + shelf mark when
    the capture has no URL); a later capture of the same item overrides the
    fields it sets and keeps the earlier capture's values for the rest.

    :param pattern: Glob of capture JSON files.
    :returns: Inventory rows in first-seen order.
    """
    rows: Dict[str, dict] = {}
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as fh:
            capture = json.load(fh)
        for item in capture.get("items") or []:
            row = normalise_item(item, capture)
            if row is None:
                continue
            rows[_dedupe_key(row)] = _merge_rows(rows.get(_dedupe_key(row)), row, "has_transcription" in item)
    return list(rows.values())


def load_list(path: str) -> List[dict]:
    """Read a hand-made item list (CSV or JSONL) into inventory rows.

    A row without a ``has_transcription`` column counts as transcribed, since
    a download list names what was fetched. Duplicates (same URL, else same
    collection + shelf mark) keep the last row.

    :param path: ``.csv`` (header row required) or ``.jsonl`` / ``.json``
        lines of objects.
    :returns: Inventory rows in first-seen order.
    :raises ValueError: For an unsupported extension.
    """
    ext = os.path.splitext(path)[1].lower()
    items: List[dict]
    if ext == ".csv":
        with open(path, encoding="utf-8-sig", newline="") as fh:
            items = list(csv.DictReader(fh))
    elif ext in (".jsonl", ".json"):
        with open(path, encoding="utf-8") as fh:
            items = [json.loads(line) for line in fh if line.strip()]
    else:
        raise ValueError(f"list must be .csv or .jsonl, got {path!r}")
    rows: Dict[str, dict] = {}
    for item in items:
        row = normalise_item({k: v for k, v in item.items() if k}, absent_means_transcribed=True)
        if row is None:
            continue
        rows[_dedupe_key(row)] = _merge_rows(rows.get(_dedupe_key(row)), row, "has_transcription" in item)
    return list(rows.values())


def _merge_rows(earlier: Optional[dict], later: dict, later_sets_flag: bool) -> dict:
    """Overlay a later capture of an item on an earlier one.

    :param earlier: The row seen first, or ``None``.
    :param later: The row seen later.
    :param later_sets_flag: Whether the later capture stated ``has_transcription``.
    :returns: The merged row: later non-empty values win; the transcription
        flag follows the later capture only when it stated one.
    """
    if earlier is None:
        return later
    merged = dict(earlier)
    for key, value in later.items():
        if key == "has_transcription":
            if later_sets_flag:
                merged[key] = value
        elif value not in (None, ""):
            merged[key] = value
    return merged


def write_inventory(rows: Iterable[dict], path: str) -> int:
    """Write inventory rows as JSONL.

    :param rows: Inventory rows.
    :param path: Output path.
    :returns: Number of rows written.
    """
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({k: row.get(k) for k in INVENTORY_FIELDS}, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_inventory(path: str) -> List[dict]:
    """Read an inventory JSONL.

    :param path: Path of ``inventory.jsonl``.
    :returns: The rows.
    """
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ─────────────────────────────── join ───────────────────────────────

def _pick_candidate(candidates: Sequence[dict], collection: str, mark: str) -> Tuple[Optional[dict], bool]:
    """Choose among several corpus records that answer to one join key.

    :param candidates: Holdings sharing the key.
    :param collection: The site's collection label for the item.
    :param mark: The item's shelf mark as printed.
    :returns: ``(chosen, ambiguous)``; the record whose institution token
        matches the item's collection wins, else the first, flagged ambiguous.
    """
    if len(candidates) == 1:
        return candidates[0], False
    wanted = resolve_token(f"{collection} {mark}")
    if wanted:
        same = [c for c in candidates if c["canonical_id"].startswith(wanted)]
        if len(same) == 1:
            return same[0], False
    return candidates[0], True


def match_status(holdings: Optional[dict]) -> str:
    """Name what the corpus holds for a matched record.

    :param holdings: :func:`record_holdings` output, or ``None`` for no match.
    :returns: One of the ``STATUS_*`` constants.
    """
    if holdings is None:
        return STATUS_UNMATCHED
    if holdings["ktiv_transcription"] and holdings["pgp_edition"]:
        return STATUS_BOTH
    if holdings["ktiv_transcription"]:
        return STATUS_KTIV
    if holdings["pgp_edition"]:
        return STATUS_PGP
    if holdings["legacy_transcription"]:
        return STATUS_LEGACY_ONLY
    return STATUS_CORPUS


def join_inventory(rows: Iterable[dict], index: Dict[str, List[dict]]) -> List[dict]:
    """Join inventory rows against the corpus index.

    :param rows: Inventory rows.
    :param index: :func:`build_reference_index` output.
    :returns: One dict per row: the row's fields plus ``join_key``,
        ``match_status``, ``matched_canonical_id``, ``matched_display``,
        ``ambiguous``, ``in_ktiv_transcription``, ``in_pgp_edition``,
        ``in_legacy_scrape`` and ``corpus_languages``.
    """
    joined: List[dict] = []
    for row in rows:
        key = join_key(row["shelf_mark"])
        candidates = index.get(key, []) if key else []
        chosen, ambiguous = _pick_candidate(candidates, row.get("collection") or "", row["shelf_mark"]) if candidates else (None, False)
        languages: List[str] = []
        if chosen:
            languages = list(chosen["pgp_languages"])
            if chosen["ktiv_language"] and chosen["ktiv_language"] not in languages:
                languages.append(chosen["ktiv_language"])
        joined.append({
            **row,
            "join_key": key,
            "match_status": match_status(chosen),
            "matched_canonical_id": chosen["canonical_id"] if chosen else "",
            "matched_display": chosen["shelfmark_display"] if chosen else "",
            "ambiguous": ambiguous,
            "candidate_count": len(candidates),
            "in_ktiv_transcription": bool(chosen and chosen["ktiv_transcription"]),
            "in_pgp_edition": bool(chosen and chosen["pgp_edition"]),
            "in_legacy_scrape": bool(chosen and chosen["legacy_transcription"]),
            "corpus_languages": "; ".join(languages),
        })
    return joined


def summarise(joined: Iterable[dict]) -> Dict[str, Dict[str, int]]:
    """Count, per collection, the transcribed items by match status.

    :param joined: :func:`join_inventory` output.
    :returns: ``collection -> {listed, transcribed, <status>: n, ...,
        not_held: n}`` where ``not_held`` sums the transcribed items that are
        in neither our KTIV transcription set nor a PGP edition; the key
        ``"ALL"`` totals every collection.
    """
    summary: Dict[str, Dict[str, int]] = collections.defaultdict(collections.Counter)
    for row in joined:
        for bucket in (row.get("collection") or "", "ALL"):
            counts = summary[bucket]
            counts["listed"] += 1
            if not row["has_transcription"]:
                continue
            counts["transcribed"] += 1
            counts[row["match_status"]] += 1
            if row["match_status"] not in (STATUS_BOTH, STATUS_KTIV, STATUS_PGP):
                counts["not_held"] += 1
            if row["ambiguous"]:
                counts["ambiguous"] += 1
    return {k: dict(v) for k, v in summary.items()}


def write_csv(rows: Sequence[dict], path: str, fields: Sequence[str]) -> int:
    """Write dict rows as CSV.

    :param rows: Rows to write.
    :param path: Output path.
    :param fields: Column order.
    :returns: Number of rows written.
    """
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


JOIN_FIELDS: Tuple[str, ...] = INVENTORY_FIELDS + (
    "join_key", "match_status", "matched_canonical_id", "matched_display", "ambiguous",
    "candidate_count", "in_ktiv_transcription", "in_pgp_edition", "in_legacy_scrape", "corpus_languages",
)


def run_build(out_dir: str = INVENTORY_DIR, captures: str = CAPTURES_GLOB, input_path: Optional[str] = None) -> int:
    """Build ``inventory.jsonl`` from a hand-made list or from the listing captures.

    :param out_dir: Inventory folder.
    :param captures: Glob of capture files (used when *input_path* is ``None``).
    :param input_path: A CSV / JSONL list (:func:`load_list`).
    :returns: Rows written.
    """
    rows = load_list(input_path) if input_path else load_captures(captures)
    os.makedirs(out_dir, exist_ok=True)
    return write_inventory(rows, os.path.join(out_dir, INVENTORY_FILE))


def run_join(out_dir: str = INVENTORY_DIR, merged_path: str = MERGED_PATH,
             input_path: Optional[str] = None) -> Dict[str, Dict[str, int]]:
    """Join the inventory against the corpus and write the CSVs and summary.

    :param out_dir: Inventory folder holding (or receiving) ``inventory.jsonl``.
    :param merged_path: Path of ``merged_shelfmarks.jsonl``.
    :param input_path: A CSV / JSONL list to build the inventory from first.
    :returns: The per-collection summary.
    """
    if input_path:
        run_build(out_dir, input_path=input_path)
    rows = read_inventory(os.path.join(out_dir, INVENTORY_FILE))
    index = build_reference_index(merged_path)
    joined = join_inventory(rows, index)
    write_csv(joined, os.path.join(out_dir, JOIN_CSV), JOIN_FIELDS)
    not_held = [r for r in joined if r["has_transcription"]
                and r["match_status"] not in (STATUS_BOTH, STATUS_KTIV, STATUS_PGP)]
    write_csv(not_held, os.path.join(out_dir, NOT_HELD_CSV), JOIN_FIELDS)
    summary = summarise(joined)
    with open(os.path.join(out_dir, SUMMARY_FILE), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, sort_keys=True)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point.

    :param argv: Arguments (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="list (--input) or captures/*.json -> inventory.jsonl")
    build.add_argument("--out-dir", default=INVENTORY_DIR)
    build.add_argument("--captures", default=CAPTURES_GLOB)
    build.add_argument("--input", default=None, help="hand-made CSV / JSONL list of items")
    join = sub.add_parser("join", help="inventory.jsonl (or --input list) -> join CSVs + summary")
    join.add_argument("--out-dir", default=INVENTORY_DIR)
    join.add_argument("--merged", default=MERGED_PATH)
    join.add_argument("--input", default=None, help="hand-made CSV / JSONL list; builds inventory.jsonl first")
    args = parser.parse_args(argv)
    if args.command == "build":
        print(f"inventory rows: {run_build(args.out_dir, args.captures, args.input)}")
        return
    summary = run_join(args.out_dir, args.merged, args.input)
    print(json.dumps(summary.get("ALL", {}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
