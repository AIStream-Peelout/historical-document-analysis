#!/usr/bin/env python3
"""Merge PGP, FJP and KTIV shelfmark data into one consolidated JSONL.

Each output line is one physical fragment, keyed by the canonical shelfmark id
produced by :class:`ShelfmarkNormalizer`. The three sources are unioned: a
fragment present in any source gets a record. PGP is authoritative on field
conflicts (precedence ``PGP > KTIV > FJP``); every source's raw payload is
retained under ``sources.*`` so nothing is lost.

This pass is **metadata only** — image references are recorded as pointers
(FJP GCP filenames, KTIV PNX/zip ids) with a ``preferred_source`` flag, but no
archives are unzipped and no files are moved.

Run::

    python -m src.datasets.merging.merge_shelfmarks
    python -m src.datasets.merging.merge_shelfmarks --out /tmp/merged.jsonl

Outputs ``merged_shelfmarks.jsonl`` and ``merge_report.json`` under
``src/datasets/raw_data/cairo_genizah/merged/`` by default.
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
from typing import Dict, Iterable, List, Optional, Tuple

# Anchor everything to the repo root (three levels up from this file) so the
# script works regardless of the current working directory, and add it to
# sys.path so ``from src...`` resolves even when run by file path (e.g. the
# PyCharm debugger) rather than as ``python -m``.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer  # noqa: E402
from src.datasets.merging.institution_tokens import (  # noqa: E402
    combine,
    institution_token,
    resolve_token,
)
from src.datasets.merging.ktiv_images import (  # noqa: E402
    gcs_url,
    image_folders,
    ktiv_image_sources,
)

RAW_DIR = os.path.join(_REPO_ROOT, "src", "datasets", "raw_data", "cairo_genizah")
PGP_FRAGMENTS = os.path.join(RAW_DIR, "pgp_raw", "data", "fragments.csv")
PGP_DOCUMENTS = os.path.join(RAW_DIR, "pgp_raw", "data", "documents.csv")
FJP_FILE = os.path.join(
    RAW_DIR, "fjp_all", "merged_princeton_friedberger_all_documents_final.json"
)
KTIV_DIR = os.path.join(RAW_DIR, "ktiv")
KTIV_GLOB = os.path.join(KTIV_DIR, "*.json")
KTIV_ZIP_GLOB = os.path.join(KTIV_DIR, "*.zip")
DEFAULT_OUT_DIR = os.path.join(RAW_DIR, "merged")

MERGED_JSONL = "merged_shelfmarks.jsonl"
REPORT_FILE = "merge_report.json"
STATE_FILE = "merge_state.json"          # snapshot of id-sets for run-over-run diff
DIFF_FILE = "merge_diff.json"            # what changed since the previous run
IMAGE_GAP_FILE = "image_gap_targets.jsonl"  # worklist of records lacking images
IMAGE_GAP_CSV = "image_gap_targets.csv"     # same worklist, collection-sorted CSV
KTIV_IMAGES_MISSING_FILE = "ktiv_images_missing.jsonl"  # scraped but imageless
KTIV_IMAGES_MISSING_CSV = "ktiv_images_missing.csv"

# Collection anchors used to detect and split FJP multi-shelfmark strings.
_ANCHOR_RE = re.compile(
    r"(?=\b(?:T-S|ENA|MS\s+heb|Or\.|Moss\.?|Mosseri|Yevr\.?|EVR|Halper|CAJS|"
    r"Gaster|L-G|AIU|Kaufmann|DK)\b)",
    flags=re.IGNORECASE,
)
# An institution context prefix like "Cambridge CUL:" or "Paris AIU:".
_INST_PREFIX_RE = re.compile(r"\b([A-Z][A-Za-z.]*(?:\s+[A-Z][A-Za-z.()]*){0,3}):\s*")


# ─────────────────────────── shelfmark splitting ───────────────────────────

def looks_multi(shelf_mark: str) -> bool:
    """Heuristically decide whether *shelf_mark* lists more than one shelfmark.

    :param shelf_mark: Raw FJP ``shelf_mark`` string.
    :returns: True if it appears to enumerate multiple fragments (a join string).
    """
    if " - " in shelf_mark:
        return True
    return len(_ANCHOR_RE.findall(shelf_mark)) > 1 or shelf_mark.count(":") > 1


def split_shelfmarks(shelf_mark: str) -> List[str]:
    """Split a (possibly multi-fragment) FJP ``shelf_mark`` into individual marks.

    The institution context (text before a ``:``) is carried forward to segments
    that lack their own prefix, so ``"Cambridge CUL: T-S A.1 T-S B.2"`` yields two
    Cambridge shelfmarks. Range endpoints (``A - B``) are kept as separate marks,
    not expanded.

    :param shelf_mark: Raw FJP ``shelf_mark`` string.
    :returns: List of individual shelfmark strings (at least one).
    """
    # Drop parentheticals first ("(Alt: 1)", "(2nd Series: P 764)") so their
    # inner ":" is never mistaken for an institution prefix boundary.
    shelf_mark = re.sub(r"\([^)]*\)", " ", shelf_mark).strip()

    if not looks_multi(shelf_mark):
        return [shelf_mark]

    marks: List[str] = []
    current_inst = ""
    for inst, body in _split_by_institution(shelf_mark):
        if inst:
            current_inst = inst
        for part in _split_body(body):
            part = part.strip(" -,")
            if not part:
                continue
            marks.append(f"{current_inst}: {part}" if current_inst else part)
    return marks or [shelf_mark]


def _split_by_institution(shelf_mark: str) -> List[Tuple[str, str]]:
    """Split *shelf_mark* into ``(institution, body)`` chunks on ``Prefix:`` marks.

    :param shelf_mark: Raw shelfmark string.
    :returns: Ordered list of ``(institution_or_empty, body)`` tuples.
    """
    matches = list(_INST_PREFIX_RE.finditer(shelf_mark))
    if not matches:
        return [("", shelf_mark)]
    chunks: List[Tuple[str, str]] = []
    if matches[0].start() > 0:
        chunks.append(("", shelf_mark[: matches[0].start()]))
    for i, m in enumerate(matches):
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(shelf_mark)
        chunks.append((m.group(1), shelf_mark[body_start:body_end]))
    return chunks


def _split_body(body: str) -> List[str]:
    """Split an institution body into individual shelfmarks on anchors / ranges.

    :param body: Shelfmark text for a single institution context.
    :returns: List of individual shelfmark strings.
    """
    # Treat a range ("A - B") as two separate marks by dropping the dash.
    body = body.replace(" - ", " ")
    # The anchor regex is a zero-width lookahead, so splitting on it inserts a
    # boundary before each collection prefix without consuming any characters.
    parts = [p.strip() for p in _ANCHOR_RE.split(body) if p.strip()]
    return parts or [body.strip()]


# ─────────────────────────────── PGP source ────────────────────────────────

def load_pgp() -> Tuple[Dict[str, dict], Dict[str, str]]:
    """Load PGP fragments + documents into canonical-keyed records and an alias map.

    :returns: ``(records, alias)`` where ``records`` maps canonical id -> a PGP
        block (fragment row + attached documents), and ``alias`` maps every
        canonical-id variant (primary + historic) to its primary canonical id.
    """
    # Index documents by pgpid.
    docs_by_pgpid: Dict[str, dict] = {}
    with open(PGP_DOCUMENTS, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            pgpid = (row.get("pgpid") or "").strip()
            if pgpid:
                docs_by_pgpid[pgpid] = row

    records: Dict[str, dict] = {}
    alias: Dict[str, str] = {}
    with open(PGP_FRAGMENTS, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            # Institution token from PGP's clean library_abbrev (+ full name as a
            # fallback signal). The same token is applied to every historic
            # variant of this fragment so they all resolve to one canonical id.
            token = institution_token(
                f"{row.get('library_abbrev') or ''} {row.get('library') or ''}"
            )
            core = ShelfmarkNormalizer.to_canonical_id(row["shelfmark"])
            if not core:
                continue
            primary = combine(token, core)
            alias[primary] = primary
            for variant in _split_semicolons(row.get("shelfmarks_historic")):
                vcore = ShelfmarkNormalizer.to_canonical_id(variant)
                if vcore:
                    alias.setdefault(combine(token, vcore), primary)

            pgpids = [p.strip() for p in (row.get("pgpids") or "").split(",") if p.strip()]
            documents = [docs_by_pgpid[p] for p in pgpids if p in docs_by_pgpid]
            block = {"fragment": row, "documents": documents}
            # Merge if two fragment rows collapse to the same canonical id.
            if primary in records:
                records[primary].setdefault("_extra_fragments", []).append(row)
            else:
                records[primary] = block
    return records, alias


def _split_semicolons(value: Optional[str]) -> List[str]:
    """Split a ``;``-separated PGP field into trimmed non-empty parts.

    :param value: Raw field value or ``None``.
    :returns: List of trimmed parts.
    """
    return [v.strip() for v in (value or "").split(";") if v.strip()]


# ─────────────────────────────── FJP source ────────────────────────────────

def load_fjp(alias: Dict[str, str]) -> Tuple[Dict[str, List[Tuple[str, dict]]], dict]:
    """Group FJP records by resolved canonical id, exploding multi-shelfmark rows.

    :param alias: PGP alias map (variant cid -> primary cid).
    :returns: ``(by_cid, stats)`` where ``by_cid`` maps canonical id -> list of
        ``(matched_shelfmark, raw_record)`` tuples. The matched shelfmark is the
        single constituent this id came from (the whole join string for a
        multi-shelfmark record would be misleading as a per-fragment display).
    """
    with open(FJP_FILE, encoding="utf-8") as fh:
        fjp = json.load(fh)

    by_cid: Dict[str, List[Tuple[str, dict]]] = collections.defaultdict(list)
    stats = {"records": len(fjp), "exploded": 0, "segments": 0, "empty_cid": 0}
    for rec in fjp.values():
        sm = rec.get("shelf_mark") or ""
        marks = split_shelfmarks(sm)
        if len(marks) > 1:
            stats["exploded"] += 1
        for mark in marks:
            core = ShelfmarkNormalizer.to_canonical_id(mark)
            if not core:
                stats["empty_cid"] += 1
                continue
            # Token from the segment's own prefix first (a multi-shelfmark join
            # can span institutions); fall back to the record's collection field.
            token = resolve_token(mark) or institution_token(
                f"{rec.get('collection') or ''} {rec.get('institution') or ''} {mark}"
            )
            cid = combine(token, core)
            stats["segments"] += 1
            key = alias.get(cid, cid)
            by_cid[key].append((mark, rec))
    return by_cid, stats


# ────────────────────────────── KTIV source ────────────────────────────────

# KTIV image archives are named e.g.
# ``ktiv_PNX_MANUSCRIPTS990051236050205171-1_images.zip`` (with ``(N)`` repeats
# for re-downloads). The 15+-digit run is the manuscript ``sys_num``.
_KTIV_ZIP_SYSNUM_RE = re.compile(r"(\d{15,})")


def index_ktiv_zips(pattern: str = KTIV_ZIP_GLOB) -> Dict[str, List[str]]:
    """Index KTIV image-zip filenames by manuscript ``sys_num``.

    :param pattern: Glob for KTIV ``*.zip`` archives.
    :returns: Mapping ``sys_num -> sorted list of zip basenames`` (multiple when
        the same manuscript was downloaded more than once, e.g. ``...(1).zip``).
    """
    by_sysnum: Dict[str, List[str]] = collections.defaultdict(list)
    for zpath in glob.glob(pattern):
        name = os.path.basename(zpath)
        m = _KTIV_ZIP_SYSNUM_RE.search(name)
        if m:
            by_sysnum[m.group(1)].append(name)
    return {k: sorted(v) for k, v in by_sysnum.items()}


def _is_blocked_scrape(doc: dict) -> bool:
    """Return True for a scrape saved off a bot-challenge / unrendered page.

    Auto-scrape runs occasionally hit the site's interstitial challenge page
    ("Just a moment...") and save a record with no content. Such files carry no
    identifying fields at all, so they are noise — but their ``page_url`` still
    names the manuscript, which the merge report surfaces for re-scraping.

    :param doc: Parsed KTIV manuscript JSON.
    :returns: True when the scrape captured a challenge page / nothing at all.
    """
    if (doc.get("page_title") or "").startswith("Just a moment"):
        return True
    return not any((
        doc.get("sys_num"),
        doc.get("shelf_mark"),
        doc.get("shelfmarks"),
        doc.get("scholarly_entries"),
    ))


def _ktiv_fallback_shelfmarks(doc: dict) -> List[str]:
    """Derive shelfmark strings for a KTIV record whose ``shelf_mark`` is null.

    Join pages (צירוף) render no shelfmark element, so the scraper stores
    ``null`` — but the page title still enumerates every constituent, e.g.
    ``"Ms. T-S 10 J 5.6, Ms. T-S 20.113, Cambridge University Library, צירוף
    מכתב | Ktiv | Item page"``. Each ``Ms.`` segment is combined with the
    institution segment into a full shelfmark string. The ``shelfmarks`` block's
    own entry is preferred when present (single-shelfmark pages).

    :param doc: Parsed KTIV manuscript JSON.
    :returns: Full shelfmark strings (one per constituent), possibly empty.
    """
    block = ((doc.get("shelfmarks") or {}).get("shelfmark") or {})
    for field in ("canonical", "value"):
        if isinstance(block, dict) and block.get(field):
            return [block[field]]

    head = (doc.get("page_title") or "").split("|")[0]
    segments = [s.strip() for s in head.split(",") if s.strip()]
    marks = [s for s in segments if s.startswith("Ms.")]
    institution = next(
        (s for s in segments
         if not s.startswith("Ms.") and re.search(r"[A-Za-z]", s)),
        "",
    )
    if not marks or not institution:
        return []
    # Comma-join so the normalizer's KTIV "<Institution>, ... Ms. <core>" split
    # fires (it requires a comma before the " Ms. " delimiter).
    return [f"{institution}, {mark}" for mark in marks]


def load_ktiv(alias: Dict[str, str]) -> Tuple[Dict[str, dict], dict]:
    """Load KTIV manuscripts, de-duplicating ``(1)`` copies (keep richest).

    Records without a ``shelf_mark`` are not dropped wholesale: bot-challenge
    junk is counted (and listed) separately, and join pages are recovered from
    their page title — the record is registered under EVERY constituent
    shelfmark, mirroring how multi-shelfmark FJP strings are exploded.

    :param alias: PGP alias map.
    :returns: ``(by_cid, stats)`` mapping canonical id -> the chosen KTIV record.
    """
    best: Dict[str, dict] = {}
    files = glob.glob(KTIV_GLOB)
    bad: List[str] = []
    blocked: List[str] = []
    empty_cid = 0
    recovered = 0
    join_constituents = 0
    for fpath in files:
        # Transcription bundles saved by the viewer flow live in the same
        # directory but are page-annotation payloads, not manuscript records —
        # they are loaded separately by :func:`load_ktiv_transcriptions`.
        if os.path.basename(fpath).endswith("_transcription.json"):
            continue
        # KTIV is scraped incrementally over months; a single malformed scrape
        # must not abort a merge of the whole (eventually 200k+) corpus.
        try:
            with open(fpath, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (json.JSONDecodeError, OSError):
            bad.append(os.path.basename(fpath))
            continue
        sm = doc.get("shelf_mark") or ""
        if sm:
            marks = [sm]
        elif _is_blocked_scrape(doc):
            blocked.append(os.path.basename(fpath))
            continue
        else:
            marks = _ktiv_fallback_shelfmarks(doc)
            if not marks:
                empty_cid += 1
                continue
            recovered += 1
            if len(marks) > 1:
                join_constituents += len(marks)
        # JTS dual-shelfmark bridge: KTIV records a new shelfmark (Lutzki / MS /
        # Rabbinica) for items that also bear an old ENA number, carried in
        # shelfmarks.additional as "Adler, Elkan Nathan Ms. <ENA-number>". ENA is
        # the identifier PGP/FJP use, so re-key under it to make them join.
        # (Only meaningful for single-shelfmark records; a join spans fragments.)
        ena_cid = _ktiv_ena_alias(doc) if len(marks) == 1 else None
        for mark in marks:
            core = ShelfmarkNormalizer.to_canonical_id(mark)
            if not core:
                empty_cid += 1
                continue
            # The KTIV shelf_mark head names the holding library; resolve from it.
            cid = ena_cid or combine(
                resolve_token(mark) or institution_token(mark), core
            )
            key = alias.get(cid, cid)
            incumbent = best.get(key)
            if incumbent is None or _ktiv_richness(doc) > _ktiv_richness(incumbent):
                best[key] = doc
    return best, {
        "files": len(files),
        "distinct": len(best),
        "empty_shelfmark": empty_cid,
        "shelfmark_recovered": recovered,
        "join_constituents": join_constituents,
        "blocked_scrapes": len(blocked),
        "blocked_files": sorted(blocked),
        "unparseable_files": bad,
    }


def _transcription_page_lines(page: dict) -> List[str]:
    """Extract the text lines of one transcription-bundle page.

    Handles both bundle shapes the scraper produces: the background-API shape
    (a W3C AnnotationPage with one word Annotation per item, where items whose
    id names a ``BreackLine`` [sic — NLI's spelling] mark line ends) and the
    viewer-panel DOM shape (pre-built ``lines``).

    :param page: One entry of a bundle's ``pages`` list.
    :returns: The page's text lines (words joined by spaces).
    """
    if page.get("lines"):
        return [line for line in page["lines"] if line]
    items = ((page.get("annotation_page") or {}).get("items")) or []
    lines: List[str] = []
    current: List[str] = []
    for item in items:
        item_id = str(item.get("id") or "").lower()
        if "breakline" in item_id or "breackline" in item_id:
            if current:
                lines.append(" ".join(current))
                current = []
            continue
        word = ((item.get("body") or {}).get("value") or "").strip()
        if word:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def load_ktiv_transcriptions(
    pattern: Optional[str] = None,
) -> Dict[str, dict]:
    """Load viewer-scraped transcription bundles, keyed by manuscript sys_num.

    The scraper's viewer flow saves ``ktiv_<docid>_transcription.json`` files:
    per-page W3C AnnotationPages with one word-level Annotation each (text,
    editorial sigla, SVG polygon on the image). The full payload is large
    (~100KB/page), so the merge keeps a compact summary — the flattened plain
    text per page (words joined in served reading order) plus the source
    filename as a pointer back to the word-level data. Duplicate downloads of
    one manuscript keep the copy with the most words.

    :param pattern: Glob for transcription bundles (defaults to the KTIV dir).
    :returns: ``sys_num -> {file, ie, page_count, pages:[{fl, image_name,
        text, lines, words}]}`` where ``text`` joins lines with newlines.
    """
    pattern = pattern or os.path.join(KTIV_DIR, "*_transcription.json")
    best: Dict[str, dict] = {}
    for fpath in glob.glob(pattern):
        try:
            with open(fpath, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        m = _KTIV_ZIP_SYSNUM_RE.search(doc.get("doc_id") or os.path.basename(fpath))
        if not m:
            continue
        pages = []
        for page in doc.get("pages") or []:
            lines = _transcription_page_lines(page)
            pages.append({
                "fl": page.get("fl"),
                "image_name": page.get("image_name"),
                "text": "\n".join(lines),
                "lines": len(lines),
                "words": sum(len(line.split()) for line in lines),
            })
        summary = {
            "file": os.path.basename(fpath),
            "ie": doc.get("ie"),
            "page_count": len(pages),
            "pages": pages,
        }
        incumbent = best.get(m.group(1))
        if incumbent is None or (
            sum(p["words"] for p in pages)
            > sum(p["words"] for p in incumbent["pages"])
        ):
            best[m.group(1)] = summary
    return best


_KTIV_ADLER_RE = re.compile(r"(?:Adler|Elkan\s+Nathan).*?Ms\.?\s*(.+)$", re.IGNORECASE)


def _ktiv_ena_alias(doc: dict) -> Optional[str]:
    """Return the ENA canonical id for a KTIV JTS record, or ``None``.

    JTS items moved into the general manuscript series (Lutzki / MS / Rabbinica /
    Scroll) keep their old ENA number in ``shelfmarks.additional`` as
    ``"Adler, Elkan Nathan Ms. <number>"`` ("Adler" = Elkan Nathan Adler = ENA).
    PGP and FJP key these fragments on ENA, so we re-key the KTIV record on the
    ENA form to make the three sources join.

    :param doc: Parsed KTIV manuscript JSON.
    :returns: Canonical id like ``New_York_JTS_ENA_1205_55``, or ``None`` when no
        Adler/ENA alternate is present.
    """
    additional = (doc.get("shelfmarks") or {}).get("additional") or ""
    m = _KTIV_ADLER_RE.search(additional)
    if not m:
        return None
    ena_core = ShelfmarkNormalizer.to_canonical_id("ENA " + m.group(1).strip())
    return combine(institution_token("New York JTS"), ena_core) if ena_core else None


def _ktiv_richness(doc: dict) -> int:
    """Score a KTIV record so the most complete duplicate wins.

    :param doc: Parsed KTIV manuscript JSON.
    :returns: Integer richness score.
    """
    return (
        int(doc.get("scholarly_entry_count") or 0) * 10
        + len(json.dumps(doc, ensure_ascii=False))
    )


# ─────────────────────────────── merging ───────────────────────────────────

def _first(*values: Optional[str]) -> Optional[str]:
    """Return the first truthy value (PGP > KTIV > FJP precedence at call site).

    :param values: Candidate values in precedence order.
    :returns: First non-empty value, or ``None``.
    """
    for v in values:
        if v:
            return v
    return None


def build_merged_record(
    cid: str,
    pgp: Optional[dict],
    fjp: List[Tuple[str, dict]],
    ktiv: Optional[dict],
    ktiv_zips: Optional[Dict[str, List[str]]] = None,
    ktiv_images: Optional[Dict[str, List[str]]] = None,
    ktiv_transcriptions: Optional[Dict[str, dict]] = None,
) -> dict:
    """Assemble one merged record from the per-source blocks for *cid*.

    Top-level scalar fields follow ``PGP > KTIV > FJP`` precedence; every raw
    source block is retained under ``sources``.

    :param cid: Canonical shelfmark id (the record key).
    :param pgp: PGP block from :func:`load_pgp`, or ``None``.
    :param fjp: List of ``(matched_shelfmark, raw_record)`` tuples for this id.
    :param ktiv: Chosen KTIV record, or ``None``.
    :param ktiv_zips: ``sys_num -> [zip basenames]`` index from
        :func:`index_ktiv_zips`, used to point at the KTIV image archives.
    :param ktiv_transcriptions: ``sys_num -> transcription summary`` from
        :func:`load_ktiv_transcriptions` (flattened per-page full text).
    :returns: The merged record dict.
    """
    pgp_frag = (pgp or {}).get("fragment") or {}
    pgp_docs = (pgp or {}).get("documents") or []
    fjp_marks = [mark for mark, _ in fjp]
    fjp_recs = [rec for _, rec in fjp]
    fjp0 = fjp_recs[0] if fjp_recs else {}
    ktiv = ktiv or {}

    sources_present = [
        name for name, present in
        (("pgp", pgp), ("fjp", fjp), ("ktiv", ktiv)) if present
    ]

    display = _first(
        pgp_frag.get("shelfmark"),
        ktiv.get("shelf_mark"),
        fjp_marks[0] if fjp_marks else None,  # constituent mark, not the join string
    )
    # Feed the prefix-stripped core so institution lookup sees "T-S AS 62.645"
    # rather than "Cambridge CUL: T-S AS 62.645".
    inst_info = ShelfmarkNormalizer.get_institution_info(
        ShelfmarkNormalizer._strip_institution(display or "")
    )

    description = _first(
        next((d.get("description") for d in pgp_docs if d.get("description")), None),
        (ktiv.get("basic_catalog") or {}).get("title"),
        fjp0.get("description"),
    )

    # Image pointers. FJP images live in the shared GCS bucket (web app prepends
    # the bucket URL). KTIV images are extracted from the downloaded zips and
    # uploaded under KTIV/<sys_num>/...; we record the resulting bucket-relative
    # object paths + full URLs so the web app can route to KTIV when populated.
    fjp_images = sorted({img for rec in fjp_recs for img in (rec.get("images") or [])})
    ktiv_pnx = ktiv.get("pnx_id")
    ktiv_sysnum = ktiv.get("sys_num")
    ktiv_zip_files = (ktiv_zips or {}).get(ktiv_sysnum or "", []) if ktiv else []
    ktiv_image_paths = (ktiv_images or {}).get(ktiv_sysnum or "", []) if ktiv else []
    ktiv_trans = (ktiv_transcriptions or {}).get(ktiv_sysnum or "") if ktiv else None
    images = {
        "fjp": fjp_images,
        "ktiv": {
            "pnx_id": ktiv_pnx,
            "sys_num": ktiv_sysnum,
            "iiif_manifest_url": ktiv.get("iiif_manifest_url"),
            "friedberg_image_no": (ktiv.get("shelfmarks") or {}).get(
                "friedberg_geniza_project_images_no"
            ),
            "zip_files": ktiv_zip_files,
            "images": ktiv_image_paths,
            "image_urls": [gcs_url(p) for p in ktiv_image_paths],
            "image_count": len(ktiv_image_paths),
            "populated": bool(ktiv_image_paths),
        } if ktiv else None,
        # Route to KTIV imagery (higher quality) once its images are populated;
        # otherwise fall back to FJP's existing GCS images.
        "preferred_source": (
            "ktiv" if ktiv_image_paths else ("fjp" if fjp_images else None)
        ),
    }

    # Institution: prefer the normalizer's prefix mapping, else the source's own
    # institution label (e.g. FJP "Baltimore", KTIV holding library — the
    # shelf_mark head "<Institution>, <City>, ... Ms. <core>" names it).
    ktiv_inst_head = (
        (ktiv.get("shelf_mark") or "").split(" Ms.")[0].split(",")[0].strip()
    )
    institution = _first(
        inst_info.get("institution"),
        pgp_frag.get("library"),
        ktiv_inst_head,
        fjp0.get("institution") if fjp0.get("institution") != "Unknown" else None,
    )

    return {
        "canonical_id": cid,
        "shelfmark_display": display,
        "institution": institution,
        "collection": _first(inst_info.get("collection"), fjp0.get("collection")),
        "subcollection": inst_info.get("subcollection"),
        "sources_present": sources_present,
        "description": description,
        "pgpids": [d.get("pgpid") for d in pgp_docs if d.get("pgpid")],
        "images": images,
        "sources": {
            "pgp": pgp,
            "fjp": fjp_recs,
            "ktiv": ktiv or None,
            "ktiv_transcription": ktiv_trans,
        },
    }


def record_has_image(record: dict) -> bool:
    """Return True if a merged record points to any image (FJP files or KTIV scan).

    :param record: A merged record from :func:`build_merged_record`.
    :returns: True when at least one image pointer exists.
    """
    images = record.get("images") or {}
    if images.get("fjp"):
        return True
    ktiv = images.get("ktiv") or {}
    return bool(ktiv.get("pnx_id") or ktiv.get("friedberg_image_no"))


def record_is_rich(record: dict) -> bool:
    """Return True if a record carries substantive metadata worth imaging.

    Used to prioritise the image-gap worklist: a fragment we already have a
    description / transcription / scholarly catalog for, but no image, is a
    high-value KTIV scrape target.

    :param record: A merged record.
    :returns: True when description, transcription or scholarship is present.
    """
    if record.get("description"):
        return True
    sources = record.get("sources") or {}
    if sources.get("ktiv_transcription"):
        return True
    ktiv = sources.get("ktiv") or {}
    if ktiv.get("scholarly_entries") or ktiv.get("full_catalog"):
        return True
    for fjp in sources.get("fjp") or []:
        if fjp.get("transcriptions") or fjp.get("description"):
            return True
    return False


def compute_coverage(
    pgp_records: Dict[str, dict],
    fjp_by_cid: Dict[str, List[Tuple[str, dict]]],
    ktiv_by_cid: Dict[str, dict],
) -> dict:
    """Compute cross-source coverage stats for the report.

    Tracks two things the project cares about as KTIV grows: how much of PGP is
    already mirrored in FJP/KTIV (the shrinking PGP-only gap), and how the
    growing KTIV set breaks down against the frozen PGP/FJP sets.

    :param pgp_records: PGP canonical-id -> block.
    :param fjp_by_cid: FJP canonical-id -> records.
    :param ktiv_by_cid: KTIV canonical-id -> record.
    :returns: A coverage dict embedded in ``merge_report.json``.
    """
    pgp_ids = set(pgp_records)
    fjp_ids = set(fjp_by_cid)
    ktiv_ids = set(ktiv_by_cid)

    pgp_in_fjp = len(pgp_ids & fjp_ids)
    pgp_in_ktiv = len(pgp_ids & ktiv_ids)
    pgp_in_either = len(pgp_ids & (fjp_ids | ktiv_ids))

    ktiv_inst: collections.Counter = collections.Counter()
    for doc in ktiv_by_cid.values():
        head = (doc.get("shelf_mark") or "").split(" Ms.")[0].split(",")[0].strip()
        ktiv_inst[head or "?"] += 1

    return {
        "pgp": {
            "total": len(pgp_ids),
            "covered_by_fjp": pgp_in_fjp,
            "covered_by_ktiv": pgp_in_ktiv,
            "covered_by_either": pgp_in_either,
            "pgp_only_gap": len(pgp_ids) - pgp_in_either,
        },
        "ktiv": {
            "distinct": len(ktiv_ids),
            "matching_pgp": pgp_in_ktiv,
            "matching_fjp": len(ktiv_ids & fjp_ids),
            "matching_pgp_or_fjp": len(ktiv_ids & (pgp_ids | fjp_ids)),
            "ktiv_only_new": len(ktiv_ids - pgp_ids - fjp_ids),
            "by_institution": dict(ktiv_inst.most_common()),
        },
    }


def diff_against_snapshot(state: dict, prev: Optional[dict]) -> dict:
    """Compute what changed between this run's *state* and the *prev* snapshot.

    Surfaces the deltas the project tracks as KTIV grows: newly scraped
    KTIV-only fragments, PGP fragments that just gained FJP/KTIV coverage, and
    records that just gained an image. The first run (no ``prev``) reports the
    current sets as the baseline with empty deltas.

    :param state: Current snapshot (id lists) produced this run.
    :param prev: Previous ``merge_state.json`` contents, or ``None``.
    :returns: A diff dict written to ``merge_diff.json``.
    """
    if prev is None:
        return {"baseline": True, "note": "no previous merge_state.json; baseline run"}

    cur_all = set(state["all_ids"])
    prev_all = set(prev.get("all_ids", []))
    cur_ktiv_only = set(state["ktiv_only_ids"])
    prev_ktiv_only = set(prev.get("ktiv_only_ids", []))
    cur_pgp_cov = set(state["pgp_covered_ids"])
    prev_pgp_cov = set(prev.get("pgp_covered_ids", []))
    cur_no_img = set(state["no_image_ids"])
    prev_no_img = set(prev.get("no_image_ids", []))

    newly_imaged = sorted((prev_no_img - cur_no_img) & cur_all)
    return {
        "baseline": False,
        "previous_total": len(prev_all),
        "current_total": len(cur_all),
        "new_records": sorted(cur_all - prev_all),
        "removed_records": sorted(prev_all - cur_all),
        "new_ktiv_only": sorted(cur_ktiv_only - prev_ktiv_only),
        "newly_pgp_covered": sorted(cur_pgp_cov - prev_pgp_cov),
        "newly_imaged": newly_imaged,
        "counts": {
            "new_records": len(cur_all - prev_all),
            "removed_records": len(prev_all - cur_all),
            "new_ktiv_only": len(cur_ktiv_only - prev_ktiv_only),
            "newly_pgp_covered": len(cur_pgp_cov - prev_pgp_cov),
            "newly_imaged": len(newly_imaged),
        },
    }


def merge(out_dir: str = DEFAULT_OUT_DIR) -> dict:
    """Run the full merge and write the JSONL, report, image-gap worklist and diff.

    Outputs in *out_dir*: ``merged_shelfmarks.jsonl`` (the corpus),
    ``image_gap_targets.jsonl`` (records lacking images — KTIV scrape worklist),
    ``ktiv_images_missing.jsonl`` (KTIV-scraped records with no local image
    source — the image-acquisition backlog), ``merge_report.json`` (stats +
    coverage + image gap + backlog + run-over-run diff summary),
    ``merge_diff.json`` (full per-id deltas vs the previous run), and
    ``merge_state.json`` (id-set snapshot used for the next run's diff).

    :param out_dir: Output directory.
    :returns: The report dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    pgp_records, alias = load_pgp()
    fjp_by_cid, fjp_stats = load_fjp(alias)
    ktiv_by_cid, ktiv_stats = load_ktiv(alias)
    ktiv_zips = index_ktiv_zips()
    # Image pointers come from the zip archives AND the loose-image folders the
    # scraper's IIIF fallback leaves behind (zip wins when both exist).
    ktiv_sources = ktiv_image_sources(
        glob.glob(KTIV_ZIP_GLOB), image_folders(KTIV_DIR)
    )
    ktiv_images = {
        sys_num: [object_path for object_path, _ in source.entries]
        for sys_num, source in ktiv_sources.items()
    }
    ktiv_stats["zip_archives"] = sum(len(v) for v in ktiv_zips.values())
    ktiv_stats["zip_manuscripts"] = len(ktiv_zips)
    ktiv_stats["image_manuscripts"] = len(ktiv_images)
    ktiv_stats["image_objects"] = sum(len(v) for v in ktiv_images.values())
    ktiv_stats["image_source_breakdown"] = dict(collections.Counter(
        source.kind for source in ktiv_sources.values()
    ))
    ktiv_transcriptions = load_ktiv_transcriptions()
    ktiv_stats["transcribed_manuscripts"] = len(ktiv_transcriptions)
    ktiv_stats["transcribed_pages"] = sum(
        t["page_count"] for t in ktiv_transcriptions.values()
    )

    all_ids = set(pgp_records) | set(fjp_by_cid) | set(ktiv_by_cid)
    counts: collections.Counter = collections.Counter()
    no_image_ids: List[str] = []
    pgp_covered_ids: List[str] = []
    ktiv_only_ids: List[str] = []
    gap_rows: List[dict] = []
    backlog_rows: List[dict] = []

    out_path = os.path.join(out_dir, MERGED_JSONL)
    with open(out_path, "w", encoding="utf-8") as out:
        for cid in sorted(all_ids):
            pgp = pgp_records.get(cid)
            fjp = fjp_by_cid.get(cid, [])
            ktiv = ktiv_by_cid.get(cid)
            record = build_merged_record(
                cid, pgp, fjp, ktiv, ktiv_zips, ktiv_images, ktiv_transcriptions
            )
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[",".join(record["sources_present"])] += 1

            if pgp and (fjp or ktiv):
                pgp_covered_ids.append(cid)
            if record["sources_present"] == ["ktiv"]:
                ktiv_only_ids.append(cid)

            if not record_has_image(record):
                no_image_ids.append(cid)
                gap_rows.append({
                    "institution": record["institution"] or "(unknown)",
                    "collection": record["collection"] or "(unknown)",
                    "canonical_id": cid,
                    "shelfmark_display": record["shelfmark_display"],
                    "sources_present": "+".join(record["sources_present"]),
                    "has_rich_metadata": record_is_rich(record),
                })

            # KTIV-scraped but no local images: a different backlog than the
            # scrape worklist above — these need their IMAGES fetched, not a
            # (re-)scrape of the catalog record.
            if ktiv and not (record["images"]["ktiv"] or {}).get("populated"):
                backlog_rows.append({
                    "institution": record["institution"] or "(unknown)",
                    "canonical_id": cid,
                    "shelfmark_display": record["shelfmark_display"],
                    "sys_num": ktiv.get("sys_num"),
                    "pnx_id": ktiv.get("pnx_id"),
                    "iiif_manifest_url": _ktiv_manifest_url(ktiv),
                    "page_url": ktiv.get("page_url"),
                })

    image_gap = _write_image_gap(out_dir, gap_rows)
    ktiv_image_backlog = _write_ktiv_backlog(out_dir, backlog_rows)

    state = {
        "all_ids": sorted(all_ids),
        "pgp_covered_ids": sorted(pgp_covered_ids),
        "ktiv_only_ids": sorted(ktiv_only_ids),
        "no_image_ids": sorted(no_image_ids),
    }
    prev = _load_json(os.path.join(out_dir, STATE_FILE))
    diff = diff_against_snapshot(state, prev)
    _write_json(os.path.join(out_dir, DIFF_FILE), diff)
    _write_json(os.path.join(out_dir, STATE_FILE), state)

    report = {
        "total_records": len(all_ids),
        "pgp_fragments": len(pgp_records),
        "fjp": fjp_stats,
        "ktiv": ktiv_stats,
        "alias_index_size": len(alias),
        "fjp_matched_to_pgp": sum(1 for c in fjp_by_cid if c in pgp_records),
        "fjp_union_only": sum(1 for c in fjp_by_cid if c not in pgp_records),
        "ktiv_matched_to_pgp": sum(1 for c in ktiv_by_cid if c in pgp_records),
        "source_combinations": dict(counts),
        "coverage": compute_coverage(pgp_records, fjp_by_cid, ktiv_by_cid),
        "image_gap": image_gap,
        "ktiv_image_backlog": ktiv_image_backlog,
        "diff_summary": diff.get("counts", {"baseline": True}),
        "output": out_path,
    }
    _write_json(os.path.join(out_dir, REPORT_FILE), report)
    return report


def _ktiv_manifest_url(ktiv: dict) -> Optional[str]:
    """Return a usable IIIF manifest URL for a KTIV record.

    Prefers the URL scraped off the item page; falls back to the NLI's
    deterministic manifest location derived from the system number, so even
    records scraped before the URL was captured get a fetchable pointer.

    :param ktiv: Raw KTIV record.
    :returns: An ``https`` manifest URL, or ``None`` when underivable.
    """
    url = ktiv.get("iiif_manifest_url") or ""
    if url.startswith("http"):
        return url
    sys_num = ktiv.get("sys_num")
    if sys_num:
        return (
            "https://iiif.nli.org.il/IIIFv21/DOCID/"
            f"PNX_MANUSCRIPTS{sys_num}-1/manifest"
        )
    return None


def _write_ktiv_backlog(out_dir: str, backlog_rows: List[dict]) -> dict:
    """Write the KTIV images-missing worklist (JSONL + CSV) and roll it up.

    These are manuscripts whose catalog record IS scraped but for which no
    local image source (zip or folder) exists — the image-acquisition backlog,
    as opposed to :func:`_write_image_gap`'s not-yet-scraped worklist.

    :param out_dir: Output directory.
    :param backlog_rows: One dict per imageless KTIV record (see :func:`merge`).
    :returns: The ``ktiv_image_backlog`` report section.
    """
    rows = sorted(
        backlog_rows, key=lambda r: (r["institution"], r["canonical_id"])
    )
    fields = ["institution", "canonical_id", "shelfmark_display", "sys_num",
              "pnx_id", "iiif_manifest_url", "page_url"]

    jsonl_path = os.path.join(out_dir, KTIV_IMAGES_MISSING_FILE)
    csv_path = os.path.join(out_dir, KTIV_IMAGES_MISSING_CSV)
    with open(jsonl_path, "w", encoding="utf-8") as jl:
        for r in rows:
            jl.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(csv_path, "w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_institution = collections.Counter(r["institution"] for r in rows)
    return {
        "scraped_without_images": len(rows),
        "by_institution": dict(by_institution.most_common()),
        "worklist_jsonl": jsonl_path,
        "worklist_csv": csv_path,
    }


def _write_image_gap(out_dir: str, gap_rows: List[dict]) -> dict:
    """Write the image-gap worklist (JSONL + collection-sorted CSV) and roll it up.

    Rows are sorted to group by collection and surface the high-value targets
    first: by institution, then collection, then rich-metadata records ahead of
    bare ones. The rollup ranks collections by their rich-but-imageless count —
    the best candidates for a KTIV scrape run.

    :param out_dir: Output directory.
    :param gap_rows: One dict per imageless record (see :func:`merge`).
    :returns: The ``image_gap`` report section.
    """
    rows = sorted(
        gap_rows,
        key=lambda r: (r["institution"], r["collection"],
                       not r["has_rich_metadata"], r["canonical_id"]),
    )
    fields = ["institution", "collection", "canonical_id", "shelfmark_display",
              "sources_present", "has_rich_metadata"]

    with open(os.path.join(out_dir, IMAGE_GAP_FILE), "w", encoding="utf-8") as jl:
        for r in rows:
            jl.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, IMAGE_GAP_CSV), "w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Rank collections by rich-but-imageless count (the scrape-priority metric).
    by_collection: Dict[Tuple[str, str], Dict[str, int]] = {}
    for r in rows:
        key = (r["institution"], r["collection"])
        agg = by_collection.setdefault(key, {"total": 0, "rich": 0})
        agg["total"] += 1
        agg["rich"] += int(r["has_rich_metadata"])
    top = sorted(by_collection.items(), key=lambda kv: kv[1]["rich"], reverse=True)
    top_collections = [
        {"institution": inst, "collection": coll, "rich": agg["rich"],
         "total": agg["total"]}
        for (inst, coll), agg in top
    ]

    return {
        "records_without_image": len(rows),
        "without_image_but_rich": sum(1 for r in rows if r["has_rich_metadata"]),
        "top_collections": top_collections,
        "worklist_jsonl": os.path.join(out_dir, IMAGE_GAP_FILE),
        "worklist_csv": os.path.join(out_dir, IMAGE_GAP_CSV),
    }


def _load_json(path: str) -> Optional[dict]:
    """Load a JSON file if it exists, else return ``None``.

    :param path: File path.
    :returns: Parsed dict or ``None``.
    """
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: str, data: dict) -> None:
    """Write *data* to *path* as indented UTF-8 JSON.

    :param path: Destination path.
    :param data: JSON-serialisable dict.
    """
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def main() -> None:
    """CLI entry point: run the merge and print the report (or a top-collections rollup)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        metavar="N",
        help="Print the top N collections by rich-but-imageless count and exit.",
    )
    args = parser.parse_args()
    report = merge(args.out_dir)

    if args.top:
        ig = report["image_gap"]
        print(f"Records without images: {ig['records_without_image']} "
              f"({ig['without_image_but_rich']} with rich metadata)\n")
        print(f"{'rich':>7} {'total':>7}  collection")
        for c in ig["top_collections"][:args.top]:
            label = f"{c['institution']} / {c['collection']}"
            print(f"{c['rich']:>7} {c['total']:>7}  {label}")
        return

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
