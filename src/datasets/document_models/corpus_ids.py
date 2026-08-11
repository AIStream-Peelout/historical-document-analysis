#!/usr/bin/env python3
"""
Deterministic, cross-store identifiers shared by Elasticsearch and the KG.

Both the Elasticsearch full-text index and the Neo4j knowledge graph need to
agree on *which book* and *which page* a record came from, so the web app can
jump from a KG triplet to the page's full text / summary in ES (and back).

Identity model
--------------
* **book key** — a DOI when the book's metadata has one (the best, globally
  stable identifier), else the book's directory stem (``pdf_name`` /
  ``source_book``, which both pipelines already key on).
* **book_uuid** — ``uuid5(GENIZAH_NS, book_key)`` — a uniform UUID regardless
  of whether the underlying key was a DOI or a stem.
* **page_uuid** — ``uuid5(GENIZAH_NS, "<book_key>#p<seq>")`` where ``seq`` is
  the **filename sequence index** (``page_001`` → 1).  Both stores MUST use
  the sequence index, never the printed/`extracted_page_number`, because the
  printed page is unreliable (missing, duplicated, roman) — see the pipeline
  notes.  The raw DOI is also surfaced as its own field for external linking.

Usage
-----
::

    key = book_key(metadata, stem="Schirmann-DiepoetischenGenizafragmente-1932")
    book_uuid(key)            # -> "…" (stable per book)
    page_uuid(key, 14)        # -> "…" (stable per page)
    extract_doi(metadata)     # -> "10.17863/CAM.74938" or None
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, Optional

# Fixed namespace so ids are reproducible across machines and runs. Do not
# change — it would invalidate every previously written book_uuid / page_uuid.
GENIZAH_NS = uuid.UUID("a3f6e2c1-9b4d-5e7a-8c2f-1d6b0a9e4f73")


def normalize_book_key(key: str) -> str:
    """Normalise a book key (DOI or stem) for stable hashing.

    Lowercases, strips a ``doi:``/``https://doi.org/`` prefix, and collapses
    whitespace so trivially-different spellings hash to the same id.

    :param key: Raw book key (DOI or directory stem).
    :returns: Normalised key string.
    """
    k = (key or "").strip().lower()
    k = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", k)
    return re.sub(r"\s+", " ", k)


def extract_doi(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return the DOI from a book metadata dict, or None.

    Looks at a top-level ``doi`` field and the structured ``identifiers.doi``
    block used across the corpus.

    :param metadata: Book metadata dict (may be ``None``).
    :returns: The DOI string (normalised, no prefix), or ``None``.
    """
    if not metadata:
        return None
    raw = metadata.get("doi")
    if not raw:
        identifiers = metadata.get("identifiers")
        if isinstance(identifiers, dict):
            raw = identifiers.get("doi")
    if not raw:
        return None
    doi = normalize_book_key(str(raw))
    # A bare DOI always starts with the "10." registrant prefix.
    return doi if doi.startswith("10.") else None


def display_doi(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return a DOI suitable for *showing and linking*, never for keying.

    :func:`extract_doi` deliberately reads only the canonical positions, because
    whatever it returns becomes the book key and so changes ``book_uuid`` and
    every ``page_uuid``. Enrichment therefore writes looked-up DOIs to
    ``external_ids.doi`` instead, where they cannot re-key anything. This reader
    sees both: the canonical value first, then the enrichment block.

    ``external_ids.doi_candidate`` is *not* consulted. A candidate matched on
    title but not on work type, which is the signature of a review carrying the
    reviewed book's title, and a reader must never be linked to a review
    believing it is the work.

    :param metadata: Book metadata dict (may be ``None``).
    :returns: The DOI string (normalised, no prefix), or ``None``.
    """
    canonical = extract_doi(metadata)
    if canonical:
        return canonical
    if not metadata:
        return None
    external = metadata.get("external_ids")
    if not isinstance(external, dict):
        return None
    raw = external.get("doi")
    if not raw:
        return None
    doi = normalize_book_key(str(raw))
    return doi if doi.startswith("10.") else None


def external_link(metadata: Optional[Dict[str, Any]], name: str) -> Optional[str]:
    """Read a precomputed link out of the ``external_ids`` enrichment block.

    :param metadata: Book metadata dict (may be ``None``).
    :param name: Key within ``external_ids`` (``worldcat_url``, ``oclc``, …).
    :returns: The value, or ``None``.
    """
    if not metadata:
        return None
    external = metadata.get("external_ids")
    if not isinstance(external, dict):
        return None
    value = external.get(name)
    return str(value) if value else None


def book_key(metadata: Optional[Dict[str, Any]], stem: str) -> str:
    """Resolve the canonical book key: DOI if available, else directory stem.

    :param metadata: Book metadata dict (may be ``None``).
    :param stem: Book directory stem / ``pdf_name`` / ``source_book`` fallback.
    :returns: Normalised book key.
    """
    doi = extract_doi(metadata)
    return normalize_book_key(doi if doi else stem)


def book_uuid(key: str) -> str:
    """Return the deterministic book UUID for a (DOI-or-stem) book key.

    :param key: Book key from :func:`book_key` (or a raw stem/DOI).
    :returns: UUID string.
    """
    return str(uuid.uuid5(GENIZAH_NS, normalize_book_key(key)))


def page_uuid(key: str, page_seq: int) -> str:
    """Return the deterministic page UUID for a book key + sequence index.

    :param key: Book key from :func:`book_key`.
    :param page_seq: Filename sequence index (``page_001`` → 1), NOT the
        printed page number.
    :returns: UUID string.
    """
    return str(uuid.uuid5(GENIZAH_NS, f"{normalize_book_key(key)}#p{int(page_seq)}"))


_PAGE_SEQ_RE = re.compile(r"page[_-](\d+)", re.IGNORECASE)


def page_seq_from_filename(filename: str) -> Optional[int]:
    """Extract the sequence index from a ``page_NNN_structured.json`` filename.

    :param filename: Structured-page filename (or full path basename).
    :returns: Integer sequence index, or ``None`` if absent.
    """
    m = _PAGE_SEQ_RE.search(filename)
    return int(m.group(1)) if m else None
