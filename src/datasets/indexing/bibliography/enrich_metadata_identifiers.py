#!/usr/bin/env python3
"""Add DOI / OCLC / ISBN identifiers to the academic-literature metadata files.

The web front end wants to link a book straight out to WorldCat and to its
publisher. That needs identifiers the metadata files mostly do not carry: only
three of them record a DOI today. This module looks the rest up and writes them
back.

Sources, both public and unauthenticated:

* **Crossref** — DOI, by bibliographic query. Accurate when a DOI exists, but it
  always returns *something*, so a hit is only accepted when the returned title
  passes the same strict prefix match used by
  :mod:`src.datasets.indexing.bibliography.availability_survey`. Hebrew
  monographs and dissertations mostly have no DOI and are correctly left blank.
* **Brandeis Alma SRU** — OCLC number (MARC ``035``) and ISBN (MARC ``020``).
  These, not the DOI, are what actually drive a WorldCat link.

Identifiers are written under an ``external_ids`` block, which is deliberately
*not* where :func:`src.datasets.document_models.corpus_ids.extract_doi` looks.

    Why that matters: ``book_key()`` returns the DOI when the metadata has one
    at ``doi`` or ``identifiers.doi``, and ``book_uuid``/``page_uuid`` hash that
    key. Writing a DOI into either of those positions therefore silently
    re-keys the book — Elasticsearch ``_id``s change (so a re-index duplicates
    rather than updates every page) and the ``book_uuid``/``page_uuid`` stamped
    on existing Neo4j relations stop matching. Three books were keyed by DOI
    from the start and are unaffected; the rest were keyed by directory stem.

``--promote-doi`` opts into the canonical position anyway, for when the
re-index and KG re-import are being done deliberately. It is off by default.

Usage::

    python -m src.datasets.indexing.bibliography.enrich_metadata_identifiers
    python -m src.datasets.indexing.bibliography.enrich_metadata_identifiers --apply
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.datasets.indexing.bibliography.availability_survey import (
    BRANDEIS_SRU,
    MARC_NS,
    HttpClient,
    brandeis_queries,
    titles_match,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROOT = _REPO_ROOT / "src/datasets/raw_data/cairo_genizah/academic_literature"
DEFAULT_CACHE = _REPO_ROOT / "artifacts/identifiers/cache"

CROSSREF_API = "https://api.crossref.org/works"
DOI_RESOLVER = "https://doi.org/"
WORLDCAT_OCLC = "https://search.worldcat.org/oclc/"
WORLDCAT_ISBN = "https://search.worldcat.org/search?q=bn:"

#: The template is not a book.
TEMPLATE_METADATA = "example_book_metadata.json"

#: A bare DOI always starts with the "10." registrant prefix.
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")

#: Crossref work types acceptable for each kind of work we hold. Without this
#: gate a *book review* is matched instead of the book: reviews carry the
#: reviewed work's exact title, so title matching alone cannot tell them apart.
#: ``10.2307/601931`` is a review of Friedman's *Jewish Marriage in Palestine*
#: in JAOS, not the book.
_ACCEPTABLE_TYPES: Dict[str, Tuple[str, ...]] = {
    "book": ("book", "monograph", "edited-book", "reference-book", "book-set"),
    "chapter": ("book-chapter", "book-part", "book-section"),
    "article": ("journal-article",),
    "dissertation": ("dissertation",),
}

#: Metadata ``publication.type`` spellings seen in the corpus -> our kind.
_TYPE_ALIASES: Dict[str, str] = {
    "print book": "book",
    "book": "book",
    "edited volume": "book",
    "monograph": "book",
    "journal article": "article",
    "article": "article",
    "academic_article": "article",
    "book chapter": "chapter",
    "book section": "chapter",
    "chapter": "chapter",
    "doctoral dissertation": "dissertation",
    "dissertation": "dissertation",
    "thesis": "dissertation",
    "phd thesis": "dissertation",
}


@dataclass
class Identifiers:
    """Identifiers resolved for one metadata file.

    :param doi: Crossref DOI, when one was confidently matched.
    :param oclc: OCLC number from the Brandeis MARC record.
    :param isbn: ISBN from the Brandeis MARC record.
    :param matched_title: The catalogue/Crossref title that matched — audit it.
    :param sources: Which provider supplied each identifier.
    :param notes: Anything a human should know about this row.
    :param needs_review: The DOI matched on title but not on work type, so it
        may be a review of the work rather than the work itself.
    """

    doi: str = ""
    oclc: str = ""
    isbn: str = ""
    matched_title: str = ""
    sources: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    needs_review: bool = False

    @property
    def found(self) -> bool:
        """:returns: True when at least one identifier was resolved."""
        return bool(self.doi or self.oclc or self.isbn)

    def to_block(self) -> Dict[str, Any]:
        """Render the ``external_ids`` block for a metadata file.

        :returns: The block, with resolver URLs precomputed for the front end.
        """
        block: Dict[str, Any] = {}
        if self.doi and self.needs_review:
            # Deliberately NOT under "doi": an unverified match may be a review
            # of the work, and the front end must not link a reader to a review
            # believing it is the book. A human promotes it after checking.
            block["doi_candidate"] = self.doi
            block["doi_candidate_url"] = DOI_RESOLVER + self.doi
        elif self.doi:
            block["doi"] = self.doi
            block["doi_url"] = DOI_RESOLVER + self.doi
        if self.oclc:
            block["oclc"] = self.oclc
            block["worldcat_url"] = WORLDCAT_OCLC + self.oclc
        if self.isbn:
            block["isbn"] = self.isbn
            if not self.oclc:
                block["worldcat_url"] = WORLDCAT_ISBN + self.isbn
        if self.matched_title:
            block["matched_title"] = self.matched_title
        if self.sources:
            block["sources"] = self.sources
        if self.notes:
            block["notes"] = self.notes
        if self.needs_review:
            block["needs_review"] = True
        return block


def read_metadata(path: Path) -> Optional[Dict[str, Any]]:
    """Read one metadata file, tolerating the broken ones in the corpus.

    :param path: Metadata file path.
    :returns: The parsed dict, or ``None`` when unusable.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        logger.warning("Unreadable metadata %s: %s", path, exc)
        return None
    return payload if isinstance(payload, dict) else None


def metadata_files(root: Path) -> List[Path]:
    """List every real book metadata file under the literature root.

    :param root: academic_literature root.
    :returns: Sorted metadata paths, excluding the template.
    """
    return sorted(p for p in root.rglob("*_metadata.json") if p.name != TEMPLATE_METADATA)


def primary_author(metadata: Dict[str, Any]) -> str:
    """Pick the author a catalogue or Crossref would file the work under.

    :param metadata: Parsed metadata.
    :returns: A single author string, or ``""``.
    """
    authors = metadata.get("authors") or metadata.get("author") or []
    if isinstance(authors, str):
        return authors.strip()
    if isinstance(authors, dict):          # the one ad-hoc `author: {name: …}` file
        return str(authors.get("name") or "").strip()
    if isinstance(authors, list) and authors:
        first = authors[0]
        if isinstance(first, dict):
            return str(first.get("name") or "").strip()
        return str(first).strip()
    return ""


def existing_doi(metadata: Dict[str, Any]) -> str:
    """Return a DOI already recorded in the canonical positions.

    :param metadata: Parsed metadata.
    :returns: The DOI, or ``""``.
    """
    raw = metadata.get("doi")
    if not raw:
        identifiers = metadata.get("identifiers")
        if isinstance(identifiers, dict):
            raw = identifiers.get("doi")
    if not raw:
        publication = metadata.get("publication")
        if isinstance(publication, dict):
            raw = publication.get("doi")
    return str(raw).strip() if raw else ""


def expected_kind(metadata: Dict[str, Any]) -> str:
    """Infer what kind of publication a metadata file describes.

    :param metadata: Parsed metadata.
    :returns: ``book``, ``chapter``, ``article``, ``dissertation`` or ``""``.
    """
    publication = metadata.get("publication")
    stated = ""
    if isinstance(publication, dict):
        stated = str(publication.get("type") or "").strip().lower()
    stated = stated or str(metadata.get("type") or "").strip().lower()
    if stated in _TYPE_ALIASES:
        return _TYPE_ALIASES[stated]
    # Fall back on the shape of the record.
    if isinstance(publication, dict):
        if publication.get("container_title"):
            return "chapter"
        if publication.get("journal"):
            return "article"
        if publication.get("publisher") or publication.get("isbn"):
            return "book"
    return ""


def query_crossref(
    client: HttpClient,
    title: str,
    author: str = "",
    kind: str = "",
) -> Tuple[str, str, List[str]]:
    """Look a work up in Crossref and return its DOI when confidently matched.

    Two filters, both necessary. Crossref always returns a best-effort result,
    so the returned title must pass the strict prefix match. And the returned
    *work type* must be compatible with what we hold, because a book review
    carries the reviewed book's exact title — matching on title alone attaches
    a review's DOI to the book, which is wrong and would send the front end to
    the wrong page.

    :param client: HTTP client.
    :param title: Work title.
    :param author: Primary author, added to the bibliographic query.
    :param kind: Expected kind from :func:`expected_kind`; ``""`` disables the
        type gate but records a note.
    :returns: ``(doi, matched_title, notes)``.
    """
    notes: List[str] = []
    query = urllib.parse.urlencode({
        "query.bibliographic": f"{title} {author}".strip(),
        "rows": 5,
        "select": "DOI,title,issued,type",
    })
    body = client.get(f"{CROSSREF_API}?{query}")
    if not body:
        return "", "", notes
    try:
        items = json.loads(body)["message"]["items"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return "", "", notes

    candidates: List[Tuple[str, str, str]] = []
    for item in items:
        candidate_title = (item.get("title") or [""])[0]
        if not titles_match(title, candidate_title)[0]:
            continue
        doi = str(item.get("DOI") or "").strip().lower()
        if _DOI_RE.match(doi):
            candidates.append((doi, candidate_title, str(item.get("type") or "")))
    if not candidates:
        return "", "", notes

    acceptable = _ACCEPTABLE_TYPES.get(kind)
    if acceptable:
        for doi, candidate_title, item_type in candidates:
            if item_type in acceptable:
                return doi, candidate_title, notes
    # Nothing of the expected type. Take the best title match but say so: for a
    # work recorded as a book, a same-titled journal-article is the signature of
    # a *review*, and the corpus's own publication types are not reliable enough
    # to decide which this is. See the JAOS review of Friedman's Jewish Marriage
    # in Palestine, which carries the book's exact title.
    doi, candidate_title, item_type = candidates[0]
    if acceptable:
        notes.append(
            f"crossref returned type '{item_type}' for a work recorded as a "
            f"{kind}; if this is a book, that is the signature of a review — verify"
        )
    else:
        notes.append(
            f"metadata states no publication type; crossref type '{item_type}' unverified"
        )
    return doi, candidate_title, notes


def query_brandeis_ids(client: HttpClient, title: str, author: str = "") -> Tuple[str, str, str]:
    """Fetch OCLC and ISBN for a work from the Brandeis Alma SRU.

    :param client: HTTP client.
    :param title: Work title.
    :param author: Primary author, for the creator fallback query.
    :returns: ``(oclc, isbn, matched_title)``; empty strings when unmatched.
    """
    for query, _is_title_phrase in brandeis_queries(title, author):
        url = (
            f"{BRANDEIS_SRU}?version=1.2&operation=searchRetrieve&recordSchema=marcxml"
            f"&maximumRecords=5&query={urllib.parse.quote(query)}"
        )
        body = client.get(url)
        if not body:
            continue
        try:
            root = ET.fromstring(body)
        except ET.ParseError:
            continue
        for record in root.findall(".//marc:record", MARC_NS):
            record_title = _marc_join(record, "245")
            original = _marc_join(record, "880")
            matched = (
                titles_match(title, record_title)[0]
                or titles_match(title, original)[0]
            )
            if not matched:
                continue
            return (
                _marc_oclc(record),
                _marc_isbn(record),
                (original or record_title)[:200],
            )
    return "", "", ""


def _marc_join(record: ET.Element, tag: str) -> str:
    """Join a MARC field's ``a``/``b`` subfields.

    :param record: A ``marc:record`` element.
    :param tag: Datafield tag.
    :returns: The joined text.
    """
    return " ".join(
        sub.text or ""
        for f in record.findall(f"marc:datafield[@tag='{tag}']", MARC_NS)
        for sub in f.findall("marc:subfield", MARC_NS)
        if sub.get("code") in ("a", "b")
    ).strip(" /:,")


def _marc_oclc(record: ET.Element) -> str:
    """Extract the OCLC number from MARC ``035``.

    :param record: A ``marc:record`` element.
    :returns: The bare number, or ``""``.
    """
    for f in record.findall("marc:datafield[@tag='035']", MARC_NS):
        for sub in f.findall("marc:subfield", MARC_NS):
            if "OCoLC" in (sub.text or ""):
                digits = re.sub(r"\D", "", sub.text or "")
                if digits:
                    return digits
    return ""


def _marc_isbn(record: ET.Element) -> str:
    """Extract an ISBN from MARC ``020``.

    :param record: A ``marc:record`` element.
    :returns: The ISBN digits (with a trailing X preserved), or ``""``.
    """
    for f in record.findall("marc:datafield[@tag='020']", MARC_NS):
        for sub in f.findall("marc:subfield", MARC_NS):
            if sub.get("code") != "a":
                continue
            match = re.search(r"(97[89][\d-]{10,17}|[\dX-]{10,17})", sub.text or "")
            if match:
                cleaned = match.group(1).replace("-", "")
                if len(cleaned) in (10, 13):
                    return cleaned
    return ""


def resolve(client: HttpClient, metadata: Dict[str, Any]) -> Identifiers:
    """Resolve every identifier available for one metadata file.

    :param client: HTTP client.
    :param metadata: Parsed metadata.
    :returns: The resolved identifiers.
    """
    result = Identifiers()
    title = str(metadata.get("title") or "").strip()
    if not title:
        result.notes.append("no title in metadata; nothing to look up")
        return result
    author = primary_author(metadata)

    known = existing_doi(metadata)
    if known:
        result.doi = known
        result.sources["doi"] = "already in metadata"
    else:
        doi, matched, notes = query_crossref(client, title, author, expected_kind(metadata))
        result.notes.extend(notes)
        result.needs_review = any("verify" in n for n in notes)
        if doi:
            result.doi = doi
            result.matched_title = matched
            result.sources["doi"] = "crossref"

    oclc, isbn, matched_title = query_brandeis_ids(client, title, author)
    if oclc:
        result.oclc = oclc
        result.sources["oclc"] = "brandeis-alma-sru"
    if isbn:
        result.isbn = isbn
        result.sources["isbn"] = "brandeis-alma-sru"
    if matched_title and not result.matched_title:
        result.matched_title = matched_title
    return result


def apply_identifiers(
    metadata: Dict[str, Any],
    identifiers: Identifiers,
    promote_doi: bool = False,
) -> bool:
    """Merge resolved identifiers into a metadata dict.

    :param metadata: Parsed metadata, mutated in place.
    :param identifiers: Resolved identifiers.
    :param promote_doi: Also write the DOI to ``identifiers.doi``, the position
        that re-keys ``book_uuid``/``page_uuid``. Off by default.
    :returns: True when the metadata changed.
    """
    block = identifiers.to_block()
    if not block:
        return False
    changed = metadata.get("external_ids") != block
    metadata["external_ids"] = block
    if promote_doi and identifiers.doi and not existing_doi(metadata):
        canonical = metadata.setdefault("identifiers", {})
        if isinstance(canonical, dict):
            canonical["doi"] = identifiers.doi
            changed = True
    return changed


def run(
    root: Path,
    cache_dir: Path,
    apply_changes: bool = False,
    promote_doi: bool = False,
    interval: float = 1.0,
) -> List[Tuple[Path, Identifiers, bool]]:
    """Resolve and optionally write identifiers for every metadata file.

    :param root: academic_literature root.
    :param cache_dir: HTTP cache directory.
    :param apply_changes: Write the files (otherwise this is a dry run).
    :param promote_doi: Also write the DOI to the canonical, re-keying position.
    :param interval: Minimum seconds between requests to one host.
    :returns: One ``(path, identifiers, changed)`` per metadata file.
    """
    client = HttpClient(cache_dir, min_interval=interval)
    results: List[Tuple[Path, Identifiers, bool]] = []
    for path in metadata_files(root):
        metadata = read_metadata(path)
        if metadata is None:
            results.append((path, Identifiers(notes=["unreadable JSON — skipped"]), False))
            continue
        identifiers = resolve(client, metadata)
        changed = apply_identifiers(metadata, identifiers, promote_doi)
        if changed and apply_changes:
            path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        results.append((path, identifiers, changed))
    logger.info("HTTP: %s", client.stats)
    return results


def print_report(results: Sequence[Tuple[Path, Identifiers, bool]], root: Path) -> None:
    """Print what was resolved for each metadata file.

    :param results: Rows from :func:`run`.
    :param root: academic_literature root, for relative display paths.
    """
    header = f"{'metadata file':44s} {'doi':30s} {'oclc':9s} {'isbn':14s} {'?':3s}"
    print(header)
    print("-" * len(header))
    for path, identifiers, _ in results:
        try:
            label = str(path.relative_to(root))
        except ValueError:
            label = path.name
        print(f"{label[:44]:44s} {(identifiers.doi or '-')[:30]:30s} "
              f"{(identifiers.oclc or '-'):9s} {(identifiers.isbn or '-'):14s} "
              f"{'!' if identifiers.needs_review else '':3s}")
    total = len(results)
    print(
        f"\n{total} metadata files: "
        f"{sum(1 for _, i, _ in results if i.doi)} with a DOI, "
        f"{sum(1 for _, i, _ in results if i.oclc)} with an OCLC number, "
        f"{sum(1 for _, i, _ in results if i.isbn)} with an ISBN, "
        f"{sum(1 for _, i, _ in results if not i.found)} with none.\n"
        f"{sum(1 for _, i, _ in results if i.needs_review)} marked '!' — the DOI matched "
        f"on title but not on work type, so it may be a review. Verify those before use."
    )


def main() -> None:
    """CLI entry point: enrich metadata files with DOI / OCLC / ISBN."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="academic_literature root")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE, help="HTTP cache")
    parser.add_argument("--apply", action="store_true",
                        help="Write the files (default is a dry run)")
    parser.add_argument(
        "--promote-doi", action="store_true",
        help="ALSO write the DOI to identifiers.doi. This re-keys book_uuid and "
             "page_uuid for every affected book, so ES must be re-indexed and the "
             "KG re-imported afterwards. Off by default.",
    )
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Minimum seconds between requests to one host")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    results = run(args.root, args.cache_dir, args.apply, args.promote_doi, args.interval)
    print_report(results, args.root)
    if args.promote_doi and args.apply:
        print("\n!! DOIs were promoted to identifiers.doi — book_uuid/page_uuid have "
              "changed for the affected books. Re-index Elasticsearch and re-run the "
              "KG import, or the ES<->KG join will break.")
    elif not args.apply:
        print("\nDry run — nothing written. Re-run with --apply to save.")


if __name__ == "__main__":
    main()
