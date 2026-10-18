#!/usr/bin/env python3
"""Bodleian direct-scrape records -> GCS object pointers.

``ktiv-scraper/bodleian/scrape_bodleian.py`` (sibling repo) writes one JSON
record per priority-queue row into ``raw_data/cairo_genizah/bodleian/records/``
and the full-resolution IIIF masters it downloaded into
``bodleian/images/<canonical_id>/<stem>.jpg``. Each record carries the canonical
id it was queued (and its images uploaded) under, the parsed TEI catalogue entry
(``tei``) and the list of downloaded ``images`` (stem / folio / side / public url
/ ``local_path``); the merge re-keys it onto the leaf's one PGP-style id
(``merge_shelfmarks.bodleian_canonical_id``) but its object paths keep the
stored id.

This module maps those local files, per canonical id, to the ordered GCS object
paths they live at once uploaded:

    BODLEIAN/<canonical_id>/<stem>.jpg

paths are relative to the bucket base (``cairo-genizah-es-json``), matching the
KTIV layout (``KTIV/<sys_num>/<member>``). A pointer is only emitted for an
image whose file actually exists on disk, so the merge never records an object
the uploader cannot produce.

Used by:
* the merge, to populate ``images.bodleian`` with pointers + a ``populated`` flag;
* :mod:`upload_bodleian_images`, which streams the same files to GCS.
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer
from src.datasets.merging.ktiv_images import GCS_BUCKET, gcs_url  # noqa: F401

BODLEIAN_PREFIX = "BODLEIAN"

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_BODLEIAN_DIR = os.path.join(
    _REPO_ROOT, "src", "datasets", "raw_data", "cairo_genizah", "bodleian"
)
DEFAULT_RECORDS_GLOB = os.path.join(DEFAULT_BODLEIAN_DIR, "records", "*.json")


class BodleianImages(NamedTuple):
    """The local image files chosen for one Bodleian fragment.

    :ivar canonical_id: Merge canonical id (also the record's file stem).
    :ivar entries: Ordered ``(object_path, local_file)`` pairs; ``local_file``
        is an absolute path to the downloaded JPEG.
    """

    canonical_id: str
    entries: List[Tuple[str, str]]


def gcs_object_path(canonical_id: str, stem: str) -> str:
    """Return the bucket-relative GCS object path for a Bodleian image.

    :param canonical_id: The id the record was scraped (and uploaded) under —
        the ``canonical_id`` stored in the record file (e.g.
        ``Oxford_Bodleian_Bodl_MS_heb_d_66_137``), which the merge may since
        have re-keyed.
    :param stem: Image stem from the record (``MS_HEB_d_66_137a``).
    :returns: ``BODLEIAN/<canonical_id>/<stem>.jpg``.
    """
    return f"{BODLEIAN_PREFIX}/{canonical_id}/{os.path.basename(stem)}.jpg"


def _richness(record: dict) -> Tuple[int, int, str]:
    """Score a scraped record so the most complete duplicate wins.

    :param record: One parsed ``records/*.json`` dict.
    :returns: Sort key: image count, TEI presence, then scrape timestamp.
    """
    return (
        int(record.get("image_count") or len(record.get("images") or [])),
        int(bool(record.get("tei"))),
        str(record.get("scraped_at") or ""),
    )


def _folio_number(value: Optional[str]) -> Optional[int]:
    """Return the leading folio number of a folio label (``"11a"`` -> 11).

    :param value: Folio / locus / leaf label, or ``None``.
    :returns: The integer, or ``None`` when the label does not start with digits.
    """
    m = re.match(r"^\s*(\d+)", str(value or ""))
    return int(m.group(1)) if m else None


def leaf_folios(record: dict) -> Set[int]:
    """Return the folio numbers of the leaf a scrape record was queued for.

    PGP, FJP and KTIV number an Oxford fragment by folio, so the leaf spec of
    the record's own queue shelfmark names its folios: ``d 44/28-29`` ->
    {28, 29}, ``c 13/6-8`` -> {6, 7, 8}, ``b 12/13a`` -> {13}. Falls back to
    the scraper's ``parsed.number`` when the shelfmark does not parse.

    :param record: One Bodleian scrape record.
    :returns: The folio numbers (empty when none can be derived).
    """
    parsed = ShelfmarkNormalizer.parse_oxford(record.get("shelf_mark") or "")
    folios = parsed.folios if parsed else set()
    if folios:
        return folios
    n = _folio_number((record.get("parsed") or {}).get("number"))
    return {n} if n is not None else set()


def _locus_ranges(tei: dict) -> List[Tuple[int, int]]:
    """Return the folio ranges covered by a TEI part's ``locus`` entries.

    :param tei: A record's ``tei`` block.
    :returns: Inclusive ``(from, to)`` folio ranges (``11a``–``18b`` -> (11, 18)).
    """
    ranges: List[Tuple[int, int]] = []
    for locus in tei.get("locus") or []:
        lo = _folio_number(locus.get("from"))
        if lo is None:
            continue
        hi = _folio_number(locus.get("to"))
        ranges.append((lo, hi if hi is not None and hi >= lo else lo))
    return ranges


def bodleian_folio_verified(record: Optional[dict]) -> bool:
    """Return True when a record's TEI entry is known to describe its own leaf.

    The scraper looked the queue number up as a TEI *part* first although every
    source numbers by *folio*, so a ``match == "part"`` record may carry another
    leaf's catalogue entry. A record is folio-verified when it matched by folio,
    or when the attached part's locus range contains the leaf's (first) folio.

    :param record: A Bodleian scrape record, or ``None``.
    :returns: True when the TEI description / date may be used for the leaf.
    """
    if not record:
        return False
    if record.get("match") == "folio":
        return True
    if record.get("match") != "part":
        return False
    folios = leaf_folios(record)
    if not folios:
        return False
    first = min(folios)
    return any(lo <= first <= hi for lo, hi in _locus_ranges(record.get("tei") or {}))


def bodleian_images_verified(record: Optional[dict]) -> bool:
    """Return True when a record's downloaded images can be trusted to show its leaf.

    Folio-verified records qualify (their images are the verified part's). So
    does a record whose every image is labelled with one of the leaf's folios —
    the case for volumes whose TEI has no numbered parts (``match == "none"``),
    where the scraper fetched folio N's facsimiles directly.

    :param record: A Bodleian scrape record, or ``None``.
    :returns: True when the Bodleian images may be preferred for the leaf.
    """
    if not record:
        return False
    if bodleian_folio_verified(record):
        return True
    folios = leaf_folios(record)
    image_folios = [_folio_number(img.get("folio")) for img in record.get("images") or []]
    return bool(folios) and bool(image_folios) and all(f in folios for f in image_folios)


def bodleian_preference(record: dict) -> Tuple[int, int, int, int, str]:
    """Rank duplicate scrape records of one leaf (higher wins).

    :param record: One Bodleian scrape record.
    :returns: Sort key: folio-verified, images-verified, then :func:`_richness`.
    """
    return (
        int(bodleian_folio_verified(record)),
        int(bodleian_images_verified(record)),
        *_richness(record),
    )


def load_bodleian_records(records_glob: str = DEFAULT_RECORDS_GLOB) -> Dict[str, dict]:
    """Load the scraper's per-fragment records keyed by canonical id.

    The ``canonical_id`` inside each file is authoritative (it was copied from
    the merge's own priority queue, so no re-normalisation happens here). If two
    files claim the same id, the richer one (more images, then TEI, then the
    later scrape) is kept. A missing directory yields an empty dict.

    :param records_glob: Glob matching the ``records/*.json`` files.
    :returns: ``canonical_id -> record``.
    """
    best: Dict[str, dict] = {}
    for path in sorted(glob.glob(records_glob)):
        with open(path, encoding="utf-8") as fh:
            record = json.load(fh)
        cid = record.get("canonical_id") or os.path.splitext(os.path.basename(path))[0]
        incumbent = best.get(cid)
        if incumbent is None or _richness(record) > _richness(incumbent):
            best[cid] = record
    return best


def bodleian_image_sources(
    records: Dict[str, dict],
    bodleian_dir: str = DEFAULT_BODLEIAN_DIR,
) -> Dict[str, BodleianImages]:
    """Resolve the on-disk image files for every record that has any.

    Only images whose ``local_path`` exists under *bodleian_dir* are kept, so
    the merge (pointer writing) and the uploader (object writing) stay in
    lockstep: a pointer the merge writes is always a file this resolves.

    The object path always uses the id the record was scraped under (its stored
    ``canonical_id``), which is where the uploader put the file, even when the
    merge has re-keyed the record onto another canonical id.

    :param records: ``canonical_id -> record`` from :func:`load_bodleian_records`
        (or the merge's re-keyed map).
    :param bodleian_dir: Root of the scrape tree (``local_path`` is relative to it).
    :returns: ``canonical_id -> BodleianImages`` (fragments with no file on disk
        are omitted).
    """
    sources: Dict[str, BodleianImages] = {}
    for cid, record in records.items():
        stored_cid = record.get("canonical_id") or cid
        entries: List[Tuple[str, str]] = []
        for image in record.get("images") or []:
            local = image.get("local_path")
            stem = image.get("stem")
            if not local or not stem:
                continue
            local_abs = os.path.join(bodleian_dir, local)
            if os.path.isfile(local_abs):
                entries.append((gcs_object_path(stored_cid, stem), local_abs))
        if entries:
            sources[cid] = BodleianImages(cid, entries)
    return sources


def bodleian_image_manifest(
    records: Dict[str, dict],
    bodleian_dir: str = DEFAULT_BODLEIAN_DIR,
) -> Dict[str, List[str]]:
    """Map each canonical id to its ordered GCS image object paths.

    Thin wrapper over :func:`bodleian_image_sources` for callers (the merge)
    that only need the object paths.

    :param records: ``canonical_id -> record``.
    :param bodleian_dir: Root of the scrape tree.
    :returns: ``canonical_id -> [bucket-relative object paths]``.
    """
    return {
        cid: [object_path for object_path, _ in source.entries]
        for cid, source in bodleian_image_sources(records, bodleian_dir).items()
    }


def tei_description(tei: Optional[dict]) -> Optional[str]:
    """Build a one-line description from a record's parsed TEI catalogue entry.

    Joins the msItem titles (``"Contract; Letter"``); when the part has no
    titles (temporary records), falls back to the first item note that is not
    the bare ``"Temporary record."`` placeholder.

    :param tei: The record's ``tei`` block, or ``None``.
    :returns: A description string, or ``None`` when the TEI has nothing usable.
    """
    if not tei:
        return None
    titles = [t.strip() for t in (tei.get("titles") or []) if t and t.strip()]
    if titles:
        return "; ".join(dict.fromkeys(titles))
    for item in tei.get("items") or []:
        for note in item.get("notes") or []:
            note = (note or "").strip()
            if note and note.lower() != "temporary record.":
                return note
    return None


def tei_date(tei: Optional[dict]) -> Optional[str]:
    """Return the catalogue's ``origDate`` text for a record, if any.

    :param tei: The record's ``tei`` block, or ``None``.
    :returns: The date string (``"1154"``, ``"12xx"``), or ``None``.
    """
    if not tei:
        return None
    orig = tei.get("orig_date") or {}
    return orig.get("text") or orig.get("not_before") or None
