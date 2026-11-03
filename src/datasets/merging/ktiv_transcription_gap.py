#!/usr/bin/env python3
"""Which KTIV-transcribed manuscripts we still lack, and a scraper queue to fetch them.

KTIV's full-text search is backed by its Transcription index, so only transcribed
manuscripts appear in it. The KTIV scraper extension (``~/Documents/GitHub/ktiv-scraper``,
``fts.js`` + the ``enumerateTranscribed`` background action, how-to in its
``docs/fts_enumeration.md``) pages through that search for a list of ubiquitous Hebrew,
Judaeo-Arabic and Arabic-script words and saves every manuscript it meets as
``ktiv_fts_transcribed_<YYYYMMDD>[_partial].json``. The user moves those files into
``src/datasets/raw_data/cairo_genizah/ktiv/`` by hand; this module joins them against
what we already hold and writes the gap report plus a queue CSV for the extension.

Inputs (read-only; nothing under ``raw_data`` is written, moved or renamed)
-------------------------------------------------------------------------
* Every ``ktiv_fts_transcribed_*.json`` in the KTIV folder (or ``--fts`` files), unioned
  by system number: ``{generated_at, tokens, token_groups, total_requests, manuscripts:
  [{sysnum, shelfmark, ie, team, thumbnail, pages, snippets, tokens, groups, fields}]}``.
* Local transcription bundles, by file name:
  ``ktiv_PNX_MANUSCRIPTS<sysnum>-<n>_transcription[(k)].json`` (each read once to tell a
  bundle with text from an empty / zero-byte one).
* Item-page scrapes (every other ``*.json`` in the folder carrying ``sys_num`` /
  ``pnx_id``): Genizah flag from ``full_catalog.subjects`` ("Cairo Genizah fragments"),
  the page title / catalogue title ("קטע גניזה") or the MARC record ("Genizah");
  ``basic_catalog.language`` (e.g. "Hebrew Judeo-Arabic") and ``basic_catalog.script``;
  and whether ``full_text.text`` holds any line, i.e. the item page itself already gave
  us the transcription text (no word boxes).
* The merged corpus ``merged/merged_shelfmarks.jsonl``: ``canonical_id`` of every record
  whose ``sources.ktiv`` is present, keyed by its ``sys_num``.
* Local image ZIPs (``*.zip`` in the KTIV folder; ``--all-zip-dirs`` adds the NAS store),
  only to flag rows whose images are already on disk.

Outputs (under ``logs/ktiv_transcription_gap/``)
------------------------------------------------
* ``ktiv_transcription_gap_report.json`` / ``.md``: transcribed manuscripts enumerated;
  of which with a local bundle (with text), with transcription text on disk from the item
  page but no bundle, with an empty bundle only, with an item scrape but no bundle, in the
  merged corpus only, unknown to us; Genizah vs non-Genizah where decidable; counts by
  library and by hit-token group (Hebrew / Judaeo-Arabic / Arabic-script), each with its
  item-page-text count; how many of our own bundles the enumeration found (its recall
  check); how many item scrapes hold text with and without a bundle; and the search words
  the enumeration did not page to their TotalHits.
* ``ktiv_transcription_gap_queue.csv``: every enumerated manuscript without a local
  bundle that has text, in the extension's queue columns ``rank, tier, canonical_id,
  shelf_mark, ktiv_manifest_url`` (plus informational columns the extension ignores).
  ``canonical_id`` is the merged canonical id when known (the system number otherwise or
  when two rows would share one), ``shelf_mark`` is the search hit's ``lds06`` (else the
  item scrape's, else the system number), ``ktiv_manifest_url`` is built from the system
  number by :func:`merge_shelfmarks._ktiv_manifest_url`. Tiers: 1 = Judaeo-Arabic or
  Arabic-script hits (or an item-scrape language naming either) unless the manuscript is
  known not to be Genizah, 2 = Genizah Hebrew/Aramaic, 3 = non-Genizah or undecided. Rank
  = tier, then hit pages descending. ``text_on_disk`` is ``item_page`` when the item
  scrape already holds the transcription text (tier unchanged; skip those rows when plain
  text is enough), else ``no``.

Fetching the queue with the extension
-------------------------------------
On a KTIV search-results page click **📂 Load Queue CSV**, pick the queue CSV, answer the
tier prompt (e.g. ``1`` first, blank = all) and the start-rank prompt (blank, or the
``next_rank`` from ``Downloads/ktiv_queue_progress.csv`` after a restart). Every row
carries ``ktiv_manifest_url``, so each is a direct item-page row: the driver opens the
item page every ``TAB_DELAY_MS`` = 8 s (``content.js``), the item tab saves the catalogue
JSON (+ MARC and manifest), opens the viewer, and the viewer saves the images ZIP (per-
image IIIF fallback) and then the word-level transcription bundle
``ktiv_PNX_MANUSCRIPTS<sysnum>-1_transcription.json``.

For this queue tick **Transcription only (no images)** next to the queue loader before
loading the CSV (opt-in; off by default): each row then saves the catalogue JSON and the
transcription bundle but no image ZIP and no IIIF images, and a row without a
transcription still completes, with its reason in the ``transcription`` / ``reason``
columns of ``ktiv_queue_progress.csv``. Without it a queue run also downloads the ZIP
(about 2.7 MB per page at the MAX resolution the polygons need, per
``docs/full_ktiv_scrape_plan.md``); the ``zip_on_disk`` column marks rows whose ZIP is
already held. Expected run time: rows x 8 s, e.g. 3,000 rows take about 6.7 h of driver
time (``run_estimate`` in the report gives the exact figure, with and without the
``text_on_disk`` rows).

Run::

    PYTHONPATH=. .venv/bin/python -m src.datasets.merging.ktiv_transcription_gap
    PYTHONPATH=. .venv/bin/python -m src.datasets.merging.ktiv_transcription_gap --fts some.json --out-dir /tmp/x
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
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.datasets.merging.merge_shelfmarks import (  # noqa: E402
    DEFAULT_OUT_DIR,
    KTIV_DIR,
    MERGED_JSONL,
    _ktiv_manifest_url,
    index_ktiv_zips,
)
from src.datasets.shelfmarks import registry_token  # noqa: E402

FTS_PATTERN = "ktiv_fts_transcribed_*.json"
MERGED_PATH = os.path.join(DEFAULT_OUT_DIR, MERGED_JSONL)
OUT_DIR = os.path.join(_REPO_ROOT, "logs", "ktiv_transcription_gap")
REPORT_JSON = "ktiv_transcription_gap_report.json"
REPORT_MD = "ktiv_transcription_gap_report.md"
QUEUE_CSV = "ktiv_transcription_gap_queue.csv"
# content.js TAB_DELAY_MS: one item page opened every 8 s by the queue driver.
SECONDS_PER_ITEM = 8
# docs/full_ktiv_scrape_plan.md: unique ZIP bytes per page at MAX resolution.
ZIP_MB_PER_PAGE = 2.7

BUNDLE_RE = re.compile(r"^ktiv_PNX_MANUSCRIPTS(\d{15,})-\d+_transcription(?:\(\d+\))?\.json$")
SYSNUM_RE = re.compile(r"\d{15,}")
_KTIV_BLOCK_RE = re.compile(r'"ktiv":\s*\{')
ARABIC_SCRIPT_RE = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
GENIZAH_TEXT_RE = re.compile(r"geni[sz]a|גניזה", re.IGNORECASE)
# Shelf-mark heads of collections that are (almost) wholly Genizah material. Only ever
# used to say "yes"; a miss leaves the manuscript undecided, never "no".
GENIZAH_SHELFMARK_RE = re.compile(
    r"\bT-S\b|Taylor-Schechter|\bENA\b|Mosseri|Lewis-Gibson|\bL-G\b|Halper|"
    r"Yevr\.?-Arab|EVR[ _-]ARAB|Antonin|Westminster College|\bAIU\b|Or\.\s*108[01]\b",
    re.IGNORECASE,
)

GROUP_HEBREW = "hebrew"
GROUP_JA = "judaeo_arabic"
GROUP_ARABIC = "arabic_script"
GROUPS: Tuple[str, ...] = (GROUP_ARABIC, GROUP_JA, GROUP_HEBREW)
# Judaeo-Arabic tokens that also occur in Hebrew / Aramaic ("עד אן"; הדא is Palestinian
# Aramaic too, as in the Yerushalmi and midrash): alone they do not make a manuscript
# Judaeo-Arabic; any other JA token, or two JA tokens, do.
WEAK_JA_TOKENS: Set[str] = {"אן", "ואן", "הדא"}

STATUS_BUNDLE = "local_bundle"
# Transcription text on disk (item page): the item scrape's full_text holds the text, but
# there is no bundle with word boxes.
STATUS_ITEM_TEXT = "item_page_text_no_bundle"
STATUS_BUNDLE_EMPTY = "local_bundle_empty"
STATUS_ITEM_ONLY = "item_scrape_no_bundle"
STATUS_MERGED_ONLY = "merged_corpus_no_bundle"
STATUS_UNKNOWN = "unknown_to_us"
STATUSES: Tuple[str, ...] = (STATUS_BUNDLE, STATUS_ITEM_TEXT, STATUS_BUNDLE_EMPTY,
                             STATUS_ITEM_ONLY, STATUS_MERGED_ONLY, STATUS_UNKNOWN)
TEXT_ON_DISK_ITEM_PAGE = "item_page"
TEXT_ON_DISK_NONE = "no"

QUEUE_FIELDS: Tuple[str, ...] = (
    "rank", "tier", "canonical_id", "shelf_mark", "ktiv_manifest_url",
    "sysnum", "status", "genizah", "primary_group", "hit_pages", "library", "zip_on_disk",
    "text_on_disk",
)


# ───────────────────────────── enumeration files ─────────────────────────────

def token_group(token: str, token_groups: Dict[str, str]) -> str:
    """Group of a search token: the file's own mapping, else inferred from its script.

    :param token: The search token.
    :param token_groups: ``token_groups`` from the enumeration file (may be empty).
    :returns: ``arabic_script``, ``judaeo_arabic`` or ``hebrew``.
    """
    group = token_groups.get(token)
    if group in GROUPS:
        return group
    return GROUP_ARABIC if ARABIC_SCRIPT_RE.search(token or "") else GROUP_HEBREW


def load_enumerations(paths: Sequence[str]) -> Tuple[Dict[str, dict], dict]:
    """Union the enumeration files by system number.

    :param paths: ``ktiv_fts_transcribed_*.json`` files.
    :returns: ``(by_sysnum, meta)``: one merged record per system number (``tokens``,
        ``ja_tokens``, ``groups``, ``pages`` as sets; first non-empty ``shelfmark``, ``ie``,
        ``team``; ``fields`` merged) and per-file metadata.
    """
    by_sysnum: Dict[str, dict] = {}
    files: List[dict] = []
    for path in sorted(paths):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        token_groups = data.get("token_groups") or {}
        manuscripts = data.get("manuscripts") or []
        files.append({
            "file": os.path.basename(path),
            "generated_at": data.get("generated_at"),
            "complete": data.get("complete"),
            "stopped_reason": data.get("stopped_reason"),
            "total_requests": data.get("total_requests"),
            "bulk_size": data.get("bulk_size") or 50,
            "facet_filtered": bool(data.get("multi_facet_query") or data.get("single_facet_query")),
            "manuscripts": len(manuscripts),
            "token_stats": data.get("token_stats") or [],
        })
        for item in manuscripts:
            sysnum = str(item.get("sysnum") or "")
            match = SYSNUM_RE.search(sysnum)
            if not match:
                continue
            sysnum = match.group(0)
            record = by_sysnum.setdefault(sysnum, {
                "sysnum": sysnum, "shelfmark": None, "ie": None, "team": None,
                "tokens": set(), "ja_tokens": set(), "groups": set(), "pages": set(), "fields": {},
            })
            for key in ("shelfmark", "ie", "team"):
                if not record[key] and item.get(key):
                    record[key] = item[key]
            for token in item.get("tokens") or []:
                record["tokens"].add(token)
                group = token_group(token, token_groups)
                if group == GROUP_JA:
                    record["ja_tokens"].add(token)
                else:
                    record["groups"].add(group)
            for group in item.get("groups") or []:
                if group in GROUPS and group != GROUP_JA:
                    record["groups"].add(group)
            record["pages"].update(p for p in item.get("pages") or [] if isinstance(p, int))
            for key, value in (item.get("fields") or {}).items():
                record["fields"].setdefault(key, value)
    for record in by_sysnum.values():
        if ja_hit(record["ja_tokens"]):
            record["groups"].add(GROUP_JA)
    return by_sysnum, {"files": files}


def ja_hit(ja_tokens: Iterable[str]) -> bool:
    """Whether Judaeo-Arabic token hits are strong enough to call a text Judaeo-Arabic.

    :param ja_tokens: The Judaeo-Arabic tokens that hit the manuscript.
    :returns: True for any distinctive token or for two or more tokens.
    """
    tokens = set(ja_tokens)
    return bool(tokens - WEAK_JA_TOKENS) or len(tokens) >= 2


# ───────────────────────────── what we hold ─────────────────────────────

def bundle_has_text(path: str) -> bool:
    """Whether a transcription bundle holds any text.

    :param path: A ``ktiv_PNX_MANUSCRIPTS…_transcription.json`` file.
    :returns: True when some page has a non-empty line (DOM capture) or annotation items
        (API capture); False for empty, zero-byte or unparsable files.
    """
    if os.path.getsize(path) == 0:
        return False
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(data, dict):
        return False
    for page in data.get("pages") or []:
        if not isinstance(page, dict):
            continue
        if any(str(line).strip() for line in page.get("lines") or []):
            return True
        annotation = page.get("annotation_page")
        if isinstance(annotation, dict) and annotation.get("items"):
            return True
    return False


def scan_bundles(ktiv_dir: str) -> Dict[str, dict]:
    """Index local transcription bundles by system number.

    :param ktiv_dir: The KTIV scrape folder.
    :returns: ``sysnum -> {"files": n, "with_text": bool}``.
    """
    bundles: Dict[str, dict] = {}
    for name in sorted(os.listdir(ktiv_dir)):
        match = BUNDLE_RE.match(name)
        if not match:
            continue
        entry = bundles.setdefault(match.group(1), {"files": 0, "with_text": False})
        entry["files"] += 1
        if not entry["with_text"] and bundle_has_text(os.path.join(ktiv_dir, name)):
            entry["with_text"] = True
    return bundles


def item_genizah_flag(doc: dict) -> Optional[bool]:
    """Genizah flag of an item-page scrape.

    :param doc: Parsed item-page JSON.
    :returns: True when the subjects, titles or MARC record name the Genizah; False when the
        record has catalogue data that does not; None for a record without catalogue data.
    """
    full = doc.get("full_catalog") if isinstance(doc.get("full_catalog"), dict) else {}
    basic = doc.get("basic_catalog") if isinstance(doc.get("basic_catalog"), dict) else {}
    subjects = full.get("subjects") or []
    texts = [s for s in subjects if isinstance(s, str)]
    texts += [doc.get("page_title") or "", basic.get("title") or "", doc.get("marc_xml") or ""]
    if any(GENIZAH_TEXT_RE.search(text) for text in texts if isinstance(text, str)):
        return True
    if full or basic or doc.get("marc_xml"):
        return False
    return None


def item_full_text(doc: dict) -> bool:
    """Whether an item-page scrape already holds transcription text.

    :param doc: Parsed item-page JSON.
    :returns: True when ``full_text.text`` is a non-blank string or a list with a non-blank line.
    """
    full_text = doc.get("full_text")
    if not isinstance(full_text, dict):
        return False
    text = full_text.get("text")
    if isinstance(text, str):
        return bool(text.strip())
    if isinstance(text, list):
        return any(isinstance(line, str) and line.strip() for line in text)
    return False


def item_sysnum(doc: dict) -> Optional[str]:
    """System number of an item-page scrape (``sys_num``, else the digits of ``pnx_id``).

    :param doc: Parsed item-page JSON.
    :returns: The system number, or None.
    """
    for key in ("sys_num", "pnx_id"):
        match = SYSNUM_RE.search(str(doc.get(key) or ""))
        if match:
            return match.group(0)
    return None


def scan_item_scrapes(ktiv_dir: str) -> Dict[str, dict]:
    """Index item-page scrapes by system number (``(n)`` copies merged).

    :param ktiv_dir: The KTIV scrape folder.
    :returns: ``sysnum -> {"shelf_mark", "genizah", "language", "script", "full_text", "files"}``;
        ``genizah`` is True if any copy says so, else False if any copy has catalogue data;
        ``full_text`` is True if any copy holds transcription text (:func:`item_full_text`).
    """
    items: Dict[str, dict] = {}
    for name in sorted(os.listdir(ktiv_dir)):
        if not name.endswith(".json") or BUNDLE_RE.match(name) or "transcription" in name:
            continue
        if name.startswith("ktiv_fts_transcribed_"):
            continue
        path = os.path.join(ktiv_dir, name)
        if os.path.getsize(path) == 0:
            continue
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        try:
            doc = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(doc, dict):
            continue
        sysnum = item_sysnum(doc)
        if not sysnum:
            continue
        basic = doc.get("basic_catalog") if isinstance(doc.get("basic_catalog"), dict) else {}
        entry = items.setdefault(sysnum, {"shelf_mark": None, "genizah": None, "language": None,
                                          "script": None, "full_text": False, "files": 0})
        entry["files"] += 1
        if not entry["full_text"] and item_full_text(doc):
            entry["full_text"] = True
        if not entry["shelf_mark"] and doc.get("shelf_mark"):
            entry["shelf_mark"] = doc["shelf_mark"]
        flag = item_genizah_flag(doc)
        if flag is True or (flag is False and entry["genizah"] is None):
            entry["genizah"] = flag
        if not entry["language"] and isinstance(basic.get("language"), str):
            entry["language"] = basic["language"]
        if not entry["script"] and basic.get("script"):
            script = basic["script"]
            entry["script"] = "; ".join(script) if isinstance(script, list) else str(script)
    return items


def load_merged_ktiv(merged_path: str) -> Dict[str, List[str]]:
    """Canonical ids of merged-corpus records that carry a KTIV source, by system number.

    :param merged_path: ``merged_shelfmarks.jsonl``.
    :returns: ``sysnum -> [canonical_id, …]`` in file order (join pages map to several).
    """
    by_sysnum: Dict[str, List[str]] = collections.defaultdict(list)
    with open(merged_path, encoding="utf-8") as fh:
        for line in fh:
            # Cheap pre-filter: most of the corpus has no KTIV block at all.
            if not _KTIV_BLOCK_RE.search(line):
                continue
            record = json.loads(line)
            ktiv = (record.get("sources") or {}).get("ktiv")
            if not isinstance(ktiv, dict):
                continue
            sysnum = item_sysnum(ktiv)
            cid = record.get("canonical_id")
            if sysnum and cid and cid not in by_sysnum[sysnum]:
                by_sysnum[sysnum].append(cid)
    return dict(by_sysnum)


# ───────────────────────────── classification ─────────────────────────────

def language_groups(language: Optional[str]) -> Set[str]:
    """Token groups implied by an item scrape's catalogue language.

    :param language: ``basic_catalog.language`` (e.g. "Hebrew Judeo-Arabic").
    :returns: ``{"judaeo_arabic"}`` and/or ``{"arabic_script"}``, or an empty set.
    """
    text = (language or "").lower()
    groups: Set[str] = set()
    if re.search(r"judaeo-?arabic|judeo-?arabic", text):
        groups.add(GROUP_JA)
        text = re.sub(r"judaeo-?arabic|judeo-?arabic", " ", text)
    if re.search(r"\barabic\b", text):
        groups.add(GROUP_ARABIC)
    return groups


def primary_group(groups: Iterable[str]) -> str:
    """The group a manuscript is tiered by: Arabic script, then Judaeo-Arabic, then Hebrew.

    :param groups: Groups that hit the manuscript.
    :returns: One group name, or ``unknown`` for none.
    """
    present = set(groups)
    for group in GROUPS:
        if group in present:
            return group
    return "unknown"


def decide_genizah(item: Optional[dict], fields: dict, shelfmark: Optional[str]) -> Tuple[Optional[bool], str]:
    """Genizah or not, from the best available evidence.

    :param item: The item-scrape entry (authoritative when it has catalogue data).
    :param fields: The search hit's kept fields (lds*/local*/collection…).
    :param shelfmark: The shelf mark (collection-head fallback, positive only).
    :returns: ``(flag, source)``: flag True / False / None (undecided) and where it came from.
    """
    if item and item.get("genizah") is not None:
        return item["genizah"], "item_scrape"
    values: List[str] = []
    for value in (fields or {}).values():
        values.extend(value if isinstance(value, list) else [value])
    if any(isinstance(v, str) and GENIZAH_TEXT_RE.search(v) for v in values):
        return True, "search_fields"
    if shelfmark and GENIZAH_SHELFMARK_RE.search(shelfmark):
        return True, "shelfmark"
    return None, "undecided"


def assign_tier(groups: Iterable[str], genizah: Optional[bool]) -> int:
    """Queue tier.

    :param groups: Token / language groups of the manuscript.
    :param genizah: Genizah flag (None = undecided).
    :returns: 1 for Judaeo-Arabic or Arabic-script unless known not to be Genizah, 2 for
        Genizah Hebrew/Aramaic, 3 for non-Genizah (any script) or undecided Hebrew/Aramaic.
    """
    present = set(groups)
    if (GROUP_JA in present or GROUP_ARABIC in present) and genizah is not False:
        return 1
    return 2 if genizah is True else 3


def library_of(shelfmark: Optional[str]) -> str:
    """Holding library of a KTIV shelf mark.

    :param shelfmark: e.g. "Cambridge University Library, Cambridge, England Ms. T-S NS 25.288".
    :returns: The institution token (``Cambridge_CUL``), else the text before the first
        comma, else ``unknown``.
    """
    if not shelfmark:
        return "unknown"
    token = registry_token(shelfmark)
    if token:
        return token
    head = shelfmark.split(",")[0].strip()
    return head or "unknown"


def classify(enumerated: Dict[str, dict], bundles: Dict[str, dict], items: Dict[str, dict],
             merged: Dict[str, List[str]], zip_sysnums: Set[str]) -> List[dict]:
    """One row per enumerated manuscript with its holding status, groups, Genizah flag and tier.

    :param enumerated: :func:`load_enumerations` records.
    :param bundles: :func:`scan_bundles` index.
    :param items: :func:`scan_item_scrapes` index.
    :param merged: :func:`load_merged_ktiv` index.
    :param zip_sysnums: System numbers with an image ZIP on disk.
    :returns: Rows sorted by system number.
    """
    rows: List[dict] = []
    for sysnum in sorted(enumerated):
        record = enumerated[sysnum]
        bundle = bundles.get(sysnum)
        item = items.get(sysnum)
        item_text = bool(item and item.get("full_text"))
        if bundle and bundle["with_text"]:
            status = STATUS_BUNDLE
        elif item_text:
            status = STATUS_ITEM_TEXT
        elif bundle:
            status = STATUS_BUNDLE_EMPTY
        elif item:
            status = STATUS_ITEM_ONLY
        elif sysnum in merged:
            status = STATUS_MERGED_ONLY
        else:
            status = STATUS_UNKNOWN
        shelfmark = record["shelfmark"] or (item or {}).get("shelf_mark")
        groups = set(record["groups"]) | language_groups((item or {}).get("language"))
        genizah, genizah_source = decide_genizah(item, record["fields"], shelfmark)
        rows.append({
            "sysnum": sysnum,
            "shelf_mark": shelfmark,
            "canonical_id": (merged.get(sysnum) or [None])[0],
            "status": status,
            "groups": sorted(groups),
            "primary_group": primary_group(groups),
            "genizah": genizah,
            "genizah_source": genizah_source,
            "tier": assign_tier(groups, genizah),
            "hit_pages": len(record["pages"]),
            "tokens": len(record["tokens"]),
            "library": library_of(shelfmark),
            "ie": record["ie"],
            "team": record["team"],
            "zip_on_disk": sysnum in zip_sysnums,
            "text_on_disk": TEXT_ON_DISK_ITEM_PAGE if item_text else TEXT_ON_DISK_NONE,
            "ktiv_manifest_url": _ktiv_manifest_url({"sys_num": sysnum}),
        })
    return rows


def build_queue(rows: Sequence[dict]) -> List[dict]:
    """Queue rows for everything without a local bundle that has text.

    Rows whose item scrape already holds the text stay in the queue at their tier (the
    bundle adds word boxes) and carry ``text_on_disk=item_page`` so they can be skipped.

    :param rows: :func:`classify` rows.
    :returns: Queue rows in :data:`QUEUE_FIELDS` order, ranked 1..N by tier then hit pages
        (descending), then shelf mark and system number for a stable order.
    """
    todo = [row for row in rows if row["status"] != STATUS_BUNDLE]
    todo.sort(key=lambda row: (row["tier"], -row["hit_pages"], row["shelf_mark"] or "", row["sysnum"]))
    cid_counts = collections.Counter(row["canonical_id"] for row in todo if row["canonical_id"])
    queue: List[dict] = []
    for rank, row in enumerate(todo, start=1):
        cid = row["canonical_id"]
        queue.append({
            "rank": rank,
            "tier": row["tier"],
            # The extension keys results by canonical_id, so it must be unique per row.
            "canonical_id": cid if cid and cid_counts[cid] == 1 else row["sysnum"],
            "shelf_mark": row["shelf_mark"] or row["sysnum"],
            "ktiv_manifest_url": row["ktiv_manifest_url"],
            "sysnum": row["sysnum"],
            "status": row["status"],
            "genizah": genizah_label(row["genizah"]),
            "primary_group": row["primary_group"],
            "hit_pages": row["hit_pages"],
            "library": row["library"],
            "zip_on_disk": "yes" if row["zip_on_disk"] else "no",
            "text_on_disk": row["text_on_disk"],
        })
    return queue


def genizah_label(flag: Optional[bool]) -> str:
    """Report label of a Genizah flag.

    :param flag: True / False / None.
    :returns: ``genizah``, ``non_genizah`` or ``undecided``.
    """
    return "genizah" if flag is True else "non_genizah" if flag is False else "undecided"


# ───────────────────────────── report ─────────────────────────────

def _count(rows: Iterable[dict], key: str) -> Dict[str, int]:
    """Count rows by one field, most frequent first.

    :param rows: Rows.
    :param key: Field to count by.
    :returns: Value -> count.
    """
    counter = collections.Counter(str(row[key]) for row in rows)
    return dict(counter.most_common())


def words_not_fully_paged(token_stats: Sequence[dict], bulk_size: int) -> List[dict]:
    """Search words the enumeration stopped before their TotalHits.

    A word can end early by design (the saturation rule, a skipped group, a skip by hand)
    or because the run is unfinished; either way its later hits were never listed.

    :param token_stats: ``token_stats`` of an enumeration file.
    :param bulk_size: Results per request in that file.
    :returns: One entry per word with ``pages_fetched * bulk_size < total_hits``:
        ``token``, ``group``, ``total_hits``, ``pages_fetched``, ``hits_not_listed``,
        ``done``, ``end_reason``.
    """
    words: List[dict] = []
    for stats in token_stats:
        total = stats.get("total_hits")
        if not isinstance(total, (int, float)) or isinstance(total, bool):
            continue
        pages = int(stats.get("pages_fetched") or 0)
        if pages * bulk_size >= total:
            continue
        words.append({
            "token": stats.get("token"),
            "group": stats.get("group"),
            "total_hits": total,
            "pages_fetched": pages,
            "hits_not_listed": int(total - pages * bulk_size),
            "done": bool(stats.get("done")),
            "end_reason": stats.get("end_reason"),
        })
    return words


def summarise(rows: Sequence[dict], queue: Sequence[dict], meta: dict,
              bundles: Dict[str, dict], items: Dict[str, dict], merged: Dict[str, List[str]]) -> dict:
    """All report numbers, each with its denominator.

    :param rows: :func:`classify` rows.
    :param queue: :func:`build_queue` rows.
    :param meta: :func:`load_enumerations` metadata.
    :param bundles: Bundle index.
    :param items: Item-scrape index.
    :param merged: Merged-corpus index.
    :returns: The report dictionary.
    """
    total = len(rows)
    enumerated = {row["sysnum"] for row in rows}
    with_text = {s for s, b in bundles.items() if b["with_text"]}
    item_text = {s for s, item in items.items() if item.get("full_text")}
    missing = [row for row in rows if row["status"] != STATUS_BUNDLE]
    by_status = {status: sum(1 for row in rows if row["status"] == status) for status in STATUSES}
    genizah = {label: sum(1 for row in rows if genizah_label(row["genizah"]) == label)
               for label in ("genizah", "non_genizah", "undecided")}
    genizah_by_status = {
        status: {label: sum(1 for row in rows if row["status"] == status
                            and genizah_label(row["genizah"]) == label)
                 for label in ("genizah", "non_genizah", "undecided")}
        for status in STATUSES
    }
    group_membership = {group: {"enumerated": sum(1 for row in rows if group in row["groups"]),
                                "missing": sum(1 for row in missing if group in row["groups"]),
                                "item_page_text": sum(1 for row in rows if group in row["groups"]
                                                      and row["status"] == STATUS_ITEM_TEXT)}
                        for group in GROUPS}
    libraries = collections.Counter(row["library"] for row in rows)
    libraries_missing = collections.Counter(row["library"] for row in missing)
    libraries_item_text = collections.Counter(row["library"] for row in rows
                                              if row["status"] == STATUS_ITEM_TEXT)
    tiers = {str(tier): sum(1 for row in queue if row["tier"] == tier) for tier in (1, 2, 3)}
    pages_lower_bound = sum(int(row["hit_pages"]) for row in queue)
    queue_item_text = sum(1 for row in queue if row["text_on_disk"] == TEXT_ON_DISK_ITEM_PAGE)
    seconds = len(queue) * SECONDS_PER_ITEM
    files = meta.get("files") or []
    # The newest complete file speaks for the run; otherwise the newest partial one.
    best = max(files, key=lambda f: (bool(f.get("complete")), f.get("generated_at") or ""), default=None)
    token_stats = best.get("token_stats") if best else []
    return {
        "denominator_note": "every count is out of 'enumerated_manuscripts' unless named otherwise",
        "enumeration_files": [{k: v for k, v in f.items() if k != "token_stats"} for f in files],
        "enumeration_complete": any(f.get("complete") for f in files),
        "enumerated_manuscripts": total,
        "by_status": by_status,
        "missing_transcriptions": len(missing),
        "genizah": genizah,
        "genizah_by_status": genizah_by_status,
        "genizah_source": _count(rows, "genizah_source"),
        "primary_group": _count(rows, "primary_group"),
        "group_membership": group_membership,
        "by_library": {lib: {"enumerated": n, "missing": libraries_missing.get(lib, 0),
                             "item_page_text": libraries_item_text.get(lib, 0)}
                       for lib, n in libraries.most_common()},
        "recall_check": {
            "local_bundles_with_text": len(with_text),
            "found_by_enumeration": len(with_text & enumerated),
            "not_found_by_enumeration": len(with_text - enumerated),
        },
        "local_holdings": {
            "bundle_sysnums": len(bundles),
            "bundle_sysnums_with_text": len(with_text),
            "item_scrape_sysnums": len(items),
            "merged_ktiv_sysnums": len(merged),
        },
        "text_on_disk": {
            "note": "item scrapes whose full_text.text holds the transcription (no word boxes)",
            "item_page_text_sysnums": len(item_text),
            "item_page_text_with_bundle": len(item_text & with_text),
            "item_page_text_without_bundle": len(item_text - with_text),
            "enumerated_item_page_text_no_bundle": by_status[STATUS_ITEM_TEXT],
            "queue_rows_item_page_text": queue_item_text,
        },
        "queue": {
            "rows": len(queue),
            "by_tier": tiers,
            "zip_on_disk": sum(1 for row in queue if row["zip_on_disk"] == "yes"),
            "text_on_disk_item_page": queue_item_text,
            "hit_pages_lower_bound": pages_lower_bound,
        },
        "run_estimate": {
            "seconds_per_item": SECONDS_PER_ITEM,
            "hours": round(seconds / 3600, 1),
            "hours_without_text_on_disk": round((len(queue) - queue_item_text) * SECONDS_PER_ITEM / 3600, 1),
            "zip_gb_lower_bound": round(pages_lower_bound * ZIP_MB_PER_PAGE / 1000, 1),
            "note": "hit pages are pages with a search hit, a lower bound on manuscript pages",
        },
        "token_stats": token_stats,
        "words_not_fully_paged": words_not_fully_paged(token_stats, int(best.get("bulk_size") or 50) if best else 50),
    }


def render_markdown(summary: dict) -> str:
    """Human-readable version of the report.

    :param summary: :func:`summarise` output.
    :returns: Markdown text.
    """
    total = summary["enumerated_manuscripts"]

    def pct(n: int, d: int) -> str:
        """Format ``n / d (p%)``.

        :param n: Numerator.
        :param d: Denominator.
        :returns: The formatted ratio.
        """
        return f"{n:,} / {d:,} ({(100.0 * n / d) if d else 0:.1f}%)"

    lines = ["# KTIV transcription gap", ""]
    for f in summary["enumeration_files"]:
        lines.append(f"* `{f['file']}`: {f['manuscripts']:,} manuscripts, {f['total_requests']} requests, "
                     f"complete={f['complete']}, stopped_reason={f['stopped_reason']}")
    lines += ["", f"Transcribed manuscripts enumerated: **{total:,}**", "",
              "| status | manuscripts |", "| --- | --- |"]
    for status, n in summary["by_status"].items():
        lines.append(f"| {status} | {pct(n, total)} |")
    lines += ["", "| Genizah | manuscripts |", "| --- | --- |"]
    for label, n in summary["genizah"].items():
        lines.append(f"| {label} | {pct(n, total)} |")
    missing = summary["missing_transcriptions"]
    lines += ["", "| group | enumerated | missing | item-page text, no bundle |", "| --- | --- | --- | --- |"]
    for group, counts in summary["group_membership"].items():
        lines.append(f"| {group} | {pct(counts['enumerated'], total)} | {pct(counts['missing'], missing)} "
                     f"| {counts['item_page_text']:,} |")
    lines += ["", "| library | enumerated | missing | item-page text, no bundle |", "| --- | --- | --- | --- |"]
    for lib, counts in list(summary["by_library"].items())[:30]:
        lines.append(f"| {lib} | {counts['enumerated']:,} | {counts['missing']:,} | {counts['item_page_text']:,} |")
    recall = summary["recall_check"]
    text = summary["text_on_disk"]
    queue = summary["queue"]
    estimate = summary["run_estimate"]
    lines += [
        "",
        f"Recall check: the enumeration found {pct(recall['found_by_enumeration'], recall['local_bundles_with_text'])} "
        "of our own transcription bundles with text.",
        "",
        f"Transcription text on disk (item page): {text['item_page_text_sysnums']:,} item scrapes hold "
        f"full_text ({text['item_page_text_with_bundle']:,} also have a bundle, "
        f"{text['item_page_text_without_bundle']:,} do not); {text['enumerated_item_page_text_no_bundle']:,} "
        f"enumerated manuscripts have it but no bundle, and {text['queue_rows_item_page_text']:,} queue rows "
        "are flagged text_on_disk=item_page (skip them when plain text is enough).",
        "",
        f"Queue: {queue['rows']:,} rows (tier 1 {queue['by_tier']['1']:,}, tier 2 {queue['by_tier']['2']:,}, "
        f"tier 3 {queue['by_tier']['3']:,}); ZIP already on disk for {pct(queue['zip_on_disk'], queue['rows'])}; "
        f"text on disk (item page) for {pct(queue['text_on_disk_item_page'], queue['rows'])}.",
        f"At {estimate['seconds_per_item']} s per item: about {estimate['hours']} h of driver time "
        f"({estimate['hours_without_text_on_disk']} h without the text_on_disk rows); "
        f"ZIPs at least {estimate['zip_gb_lower_bound']} GB ({queue['hit_pages_lower_bound']:,} hit pages, a lower bound) "
        "unless the queue runs transcription only.",
        "",
    ]
    unpaged = summary["words_not_fully_paged"]
    if unpaged:
        lines += [f"Words not paged to their TotalHits ({len(unpaged)}): their later hits were never listed.", "",
                  "| word | group | total hits | pages fetched | hits not listed | end reason |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for word in unpaged:
            lines.append(f"| {word['token']} | {word['group']} | {word['total_hits']:,} | {word['pages_fetched']:,} "
                         f"| {word['hits_not_listed']:,} | {word['end_reason'] or ('not done' if not word['done'] else '')} |")
        lines.append("")
    return "\n".join(lines)


def write_csv(rows: Sequence[dict], path: str, fields: Sequence[str]) -> int:
    """Write rows as CSV.

    :param rows: Rows (dicts).
    :param path: Output path.
    :param fields: Column order.
    :returns: Rows written.
    """
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def run(ktiv_dir: str = KTIV_DIR, merged_path: str = MERGED_PATH, out_dir: str = OUT_DIR,
        fts_paths: Optional[Sequence[str]] = None, zip_glob: Optional[str] = None) -> dict:
    """Build the report and the queue.

    :param ktiv_dir: KTIV scrape folder (read-only).
    :param merged_path: Merged corpus JSONL (read-only).
    :param out_dir: Output folder (created if needed).
    :param fts_paths: Enumeration files; None = every ``ktiv_fts_transcribed_*.json`` in ``ktiv_dir``.
    :param zip_glob: Glob of image ZIPs; None = ``*.zip`` in ``ktiv_dir``; ``"ALL"`` = the
        local folder plus the NAS store (:func:`merge_shelfmarks.ktiv_zip_paths`).
    :returns: The report dictionary.
    :raises FileNotFoundError: When no enumeration file is found.
    """
    paths = list(fts_paths) if fts_paths else sorted(glob.glob(os.path.join(ktiv_dir, FTS_PATTERN)))
    if not paths:
        raise FileNotFoundError(
            f"no {FTS_PATTERN} in {ktiv_dir}: run the extension's transcribed-manuscript "
            "enumerator and move its file there first")
    enumerated, meta = load_enumerations(paths)
    bundles = scan_bundles(ktiv_dir)
    items = scan_item_scrapes(ktiv_dir)
    merged = load_merged_ktiv(merged_path) if os.path.exists(merged_path) else {}
    if zip_glob == "ALL":
        zip_sysnums = set(index_ktiv_zips())
    else:
        zip_sysnums = set(index_ktiv_zips(zip_glob or os.path.join(ktiv_dir, "*.zip")))
    rows = classify(enumerated, bundles, items, merged, zip_sysnums)
    queue = build_queue(rows)
    summary = summarise(rows, queue, meta, bundles, items, merged)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, REPORT_JSON), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, REPORT_MD), "w", encoding="utf-8") as fh:
        fh.write(render_markdown(summary))
    write_csv(queue, os.path.join(out_dir, QUEUE_CSV), QUEUE_FIELDS)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point.

    :param argv: Arguments (None = ``sys.argv[1:]``).
    :returns: None.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--ktiv-dir", default=KTIV_DIR, help="KTIV scrape folder (read-only)")
    parser.add_argument("--merged", default=MERGED_PATH, help="merged_shelfmarks.jsonl")
    parser.add_argument("--out-dir", default=OUT_DIR, help="report / queue folder")
    parser.add_argument("--fts", nargs="*", default=None,
                        help=f"enumeration files (default: every {FTS_PATTERN} in --ktiv-dir)")
    parser.add_argument("--all-zip-dirs", action="store_true",
                        help="also look for image ZIPs in the NAS store (zip_on_disk column)")
    args = parser.parse_args(argv)
    summary = run(args.ktiv_dir, args.merged, args.out_dir, args.fts,
                  "ALL" if args.all_zip_dirs else None)
    print(json.dumps({k: summary[k] for k in ("enumerated_manuscripts", "by_status", "genizah",
                                               "recall_check", "queue", "run_estimate")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
