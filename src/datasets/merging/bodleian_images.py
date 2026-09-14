#!/usr/bin/env python3
"""Bodleian direct-scrape records -> GCS object pointers.

``ktiv-scraper/bodleian/scrape_bodleian.py`` (sibling repo) writes one JSON
record per priority-queue row into ``raw_data/cairo_genizah/bodleian/records/``
and the full-resolution IIIF masters it downloaded into
``bodleian/images/<canonical_id>/<stem>.jpg``. Each record already carries the
merge's canonical id, the parsed TEI catalogue entry (``tei``) and the list of
downloaded ``images`` (stem / folio / side / public url / ``local_path``).

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
from typing import Dict, List, NamedTuple, Optional, Tuple

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

    :param canonical_id: Merge canonical id (e.g. ``Oxford_Bodleian_Bodl_MS_heb_d_66_137``).
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

    :param records: ``canonical_id -> record`` from :func:`load_bodleian_records`.
    :param bodleian_dir: Root of the scrape tree (``local_path`` is relative to it).
    :returns: ``canonical_id -> BodleianImages`` (fragments with no file on disk
        are omitted).
    """
    sources: Dict[str, BodleianImages] = {}
    for cid, record in records.items():
        entries: List[Tuple[str, str]] = []
        for image in record.get("images") or []:
            local = image.get("local_path")
            stem = image.get("stem")
            if not local or not stem:
                continue
            local_abs = os.path.join(bodleian_dir, local)
            if os.path.isfile(local_abs):
                entries.append((gcs_object_path(cid, stem), local_abs))
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
