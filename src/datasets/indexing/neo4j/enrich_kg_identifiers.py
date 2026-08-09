#!/usr/bin/env python3
"""Backfill direct-link identifiers on ``(:BookArticle)`` nodes.

Implements ``docs/BOOK_LINK_INDEXING_CONTRACT.md``. The genizah_search backend
reads these properties in ``neo4j_service.find_book_article()`` and serves them
from ``GET /book-info`` to build the "open this work" link in the chat UI. The
serving side does not guess, so a wrongly-shaped value produces a broken link
rather than an error.

Properties written, exactly as the contract specifies:

============  ==========================================================
``doi``       Bare DOI, stored verbatim from Crossref ``message.DOI``.
              Never a URL — the backend builds ``https://doi.org/{doi}``.
``isbn``      Digits only, hyphens stripped, 10 or 13 chars. Books only.
``url``       Absolute landing-page URL, no trailing period.
``journal``   Set only for works Crossref types as ``journal-article``;
              its *presence* is how the UI picks the article layout.
``es_book_id`` / ``es_title``
              The exact join keys from the contract's "Title join"
              section, so ``/book-info`` stops relying on a fuzzy
              leading-word probe.
============  ==========================================================

Resolution follows the contract's priority order: DOI, then ISBN for books,
then a landing-page URL. Only works that would otherwise fall through to a
WorldCat/Scholar *search* — the experience the contract is trying to eliminate
— are worth an OpenLibrary round trip, so the ISBN step runs only when no DOI
was found.

Nothing here can move ``book_uuid``/``page_uuid``: those hash ``book_key``,
which reads a *metadata file's* canonical DOI position, not a graph property.

Two matching hazards, both guarded:

* Crossref always returns a fuzzy fallback, so a hit must pass the strict
  prefix match from :mod:`…availability_survey` before it is believed.
* A book review carries the reviewed book's exact title. Where a node states a
  ``source_type`` the returned work type is gated against it; otherwise the DOI
  is recorded with ``doi_unverified`` so the serving side can prefer confirmed
  links.

Usage::

    python -m src.datasets.indexing.neo4j.enrich_kg_identifiers --limit 50
    python -m src.datasets.indexing.neo4j.enrich_kg_identifiers --apply
    python -m src.datasets.indexing.neo4j.enrich_kg_identifiers --join-only --apply
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import dotenv

from src.datasets.indexing.bibliography.availability_survey import HttpClient, titles_match
from src.datasets.indexing.bibliography.enrich_metadata_identifiers import (
    _ACCEPTABLE_TYPES,
    CROSSREF_API,
)
from src.datasets.indexing.neo4j.citation_priority_report import (
    is_placeholder,
    normalize_title,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CACHE = _REPO_ROOT / "artifacts/identifiers/kg_cache"
DEFAULT_OUT = _REPO_ROOT / "artifacts/identifiers/kg_identifiers.json"

OPENLIBRARY_SEARCH = "https://openlibrary.org/search.json"
OPENLIBRARY_BASE = "https://openlibrary.org"

#: A node whose ``url`` is already a DOI resolver link carries its DOI for free.
#: A JSTOR ``stable/<id>`` link does *not*: the ``10.2307/<id>`` mapping was
#: tested against Crossref for this corpus's ids and does not hold.
_DOI_IN_URL = re.compile(r"doi\.org/(10\.\d{4,9}/\S+?)/?$", re.IGNORECASE)

#: Contract: a DOI starts "10.". Anything else belongs in ``url``.
_BARE_DOI = re.compile(r"^10\.\d{4,9}/\S+$")

#: Contract: digits only, 10 or 13 chars, trailing X allowed on ISBN-10.
_ISBN_SHAPE = re.compile(r"^(?:\d{9}[\dX]|\d{13})$", re.IGNORECASE)

#: A work merging this many nodes written by this many different people is a
#: journal used as a work title, not a publication. Journals have an ISSN, not
#: a DOI. Same rule as :func:`…genizah_focus_report.mark_serials`, which
#: separates ``סיני`` (79 nodes, 31 authors) from a multi-volume monograph.
SERIAL_MIN_NODES = 10
SERIAL_MIN_AUTHORS = 8

#: KG ``source_type`` values mapped onto the Crossref type gate's vocabulary.
_KG_TYPE_TO_KIND: Dict[str, str] = {
    "article": "article",
    "book": "book",
    "book section": "chapter",
    "dissertation": "dissertation",
}

#: Kinds for which an ISBN is meaningful.
_BOOKISH = {"book", "chapter", "dissertation", ""}


def bare_doi(value: Optional[str]) -> str:
    """Reduce a DOI to the bare form the contract requires.

    Strips resolver prefixes and trailing punctuation. Case is preserved: the
    contract says store ``message.DOI`` verbatim, and although DOIs are
    case-insensitive for resolution, altering the string is a needless
    deviation from the source record.

    :param value: A DOI, possibly expressed as a URL.
    :returns: The bare DOI, or ``""`` when the value is not one.
    """
    text = str(value or "").strip()
    text = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", text, flags=re.IGNORECASE)
    text = text.rstrip(" .,;")
    return text if _BARE_DOI.match(text) else ""


def clean_isbn(value: Optional[str]) -> str:
    """Reduce an ISBN to the digits-only form the contract requires.

    :param value: A raw ISBN, possibly hyphenated or annotated.
    :returns: The cleaned ISBN, or ``""`` when it is not a valid shape.
    """
    text = re.sub(r"[^0-9Xx]", "", str(value or ""))
    return text.upper() if _ISBN_SHAPE.match(text) else ""


def clean_url(value: Optional[str]) -> str:
    """Normalise a landing-page URL per the contract.

    :param value: A raw URL.
    :returns: An absolute URL with no trailing period, or ``""``.
    """
    text = str(value or "").strip().rstrip(".")
    return text if text.startswith(("http://", "https://")) else ""


def doi_from_url(url: Optional[str]) -> str:
    """Extract a bare DOI from a node's URL when it is a resolver link.

    :param url: The node's ``url`` property.
    :returns: The bare DOI, or ``""``.
    """
    match = _DOI_IN_URL.search(str(url or ""))
    return bare_doi(match.group(1)) if match else ""


@dataclass
class WorkIdentifiers:
    """Contract fields resolved for one distinct cited work.

    :param title: Display title.
    :param article_ids: Every KG node id sharing this work.
    :param doi: Bare DOI.
    :param isbn: Digits-only ISBN.
    :param url: Absolute landing-page URL.
    :param journal: Journal name, set only for journal articles.
    :param doi_unverified: The DOI matched on title but not on work type.
    :param matched_title: The Crossref/OpenLibrary title that matched.
    :param note: Why a match was flagged or skipped.
    """

    title: str
    article_ids: List[str] = field(default_factory=list)
    doi: str = ""
    isbn: str = ""
    url: str = ""
    journal: str = ""
    doi_unverified: bool = False
    matched_title: str = ""
    note: str = ""

    @property
    def tier(self) -> str:
        """:returns: Which contract link tier this work reaches."""
        if self.doi:
            return "doi"
        if self.isbn:
            return "isbn"
        if self.url:
            return "url"
        return "search-fallback"


def fetch_book_articles(session: Any) -> List[Dict[str, Any]]:
    """Read every BookArticle node with the fields needed to identify it.

    :param session: An open Neo4j session.
    :returns: One dict per node.
    """
    query = """
    MATCH (b:BookArticle)
    OPTIONAL MATCH (a)-[:WROTE]->(b)
    WITH b, [n IN collect(DISTINCT a.name) WHERE n IS NOT NULL] AS authors
    RETURN b.article_id AS article_id, b.title AS title, b.citation AS citation,
           b.year AS year, b.source_type AS source_type, b.journal AS journal,
           b.url AS url, b.doi AS doi, b.isbn AS isbn, authors AS authors
    """
    return [dict(record) for record in session.run(query)]


def group_works(
    rows: Sequence[Dict[str, Any]],
    skip_serials: bool = True,
) -> List[WorkIdentifiers]:
    """Collapse nodes into distinct works worth looking up.

    Duplicate nodes for one work are merged so a provider is asked once and the
    answer written to every node. Unpublished-edition placeholders and journal
    titles are dropped — neither has a DOI or an ISBN.

    :param rows: Rows from :func:`fetch_book_articles`.
    :param skip_serials: Drop works whose title names a journal.
    :returns: Distinct works, largest node-group first.
    """
    grouped: Dict[Tuple[str, Optional[int]], WorkIdentifiers] = {}
    meta: Dict[Tuple[str, Optional[int]], Dict[str, Any]] = {}
    for row in rows:
        title = (row.get("title") or row.get("citation") or "").strip()
        key_title = normalize_title(title)
        if not key_title or not row.get("article_id"):
            continue
        if is_placeholder(title, row.get("source_type")):
            continue
        year = row.get("year")
        try:
            year_key: Optional[int] = int(float(year)) if year is not None else None
        except (TypeError, ValueError):
            year_key = None
        key = (key_title, year_key)
        work = grouped.get(key)
        if work is None:
            work = WorkIdentifiers(title=title)
            grouped[key] = work
            meta[key] = {
                "authors": set(row.get("authors") or []),
                "source_type": row.get("source_type"),
                "year": year_key,
                "doi": row.get("doi"),
                "isbn": row.get("isbn"),
                "url": row.get("url"),
                "journal": row.get("journal"),
            }
        else:
            info = meta[key]
            info["authors"].update(row.get("authors") or [])
            for name in ("doi", "isbn", "url", "journal", "source_type"):
                info[name] = info.get(name) or row.get(name)
        work.article_ids.append(row["article_id"])
        if len(title) > len(work.title):
            work.title = title

    works: List[WorkIdentifiers] = []
    for key, work in grouped.items():
        info = meta[key]
        if skip_serials and (
            len(work.article_ids) >= SERIAL_MIN_NODES
            and len(info["authors"]) >= SERIAL_MIN_AUTHORS
        ):
            continue
        info["authors"] = sorted(info["authors"])
        setattr(work, "_meta", info)
        works.append(work)
    return sorted(works, key=lambda w: -len(w.article_ids))


def resolve_doi(
    client: HttpClient,
    title: str,
    author: str,
    year: Optional[int],
    kind: str,
) -> Tuple[str, str, str, bool, str]:
    """Resolve a DOI from Crossref, per the contract's step 1.

    Queried with title + first author + year, as the contract specifies. The
    returned DOI is stored verbatim; only resolver prefixes are stripped, and
    a value that does not start with ``10.`` is rejected outright rather than
    written into ``doi``.

    :param client: HTTP client.
    :param title: Work title.
    :param author: First author.
    :param year: Publication year, when the node records one.
    :param kind: Expected work kind, or ``""`` when the node states none.
    :returns: ``(doi, journal, matched_title, unverified, note)``.
    """
    bibliographic = " ".join(
        part for part in (title, author, str(year) if year else "") if part
    ).strip()
    query = urllib.parse.urlencode({
        "query.bibliographic": bibliographic,
        "rows": 5,
        "select": "DOI,title,issued,type,container-title",
    })
    body = client.get(f"{CROSSREF_API}?{query}")
    if not body:
        return "", "", "", False, "crossref unreachable"
    try:
        items = json.loads(body)["message"]["items"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return "", "", "", False, "unparseable crossref response"

    candidates: List[Tuple[str, str, str, str]] = []
    for item in items:
        candidate_title = (item.get("title") or [""])[0]
        if not titles_match(title, candidate_title)[0]:
            continue
        doi = bare_doi(item.get("DOI"))
        if doi:
            container = (item.get("container-title") or [""])[0]
            candidates.append((doi, candidate_title, str(item.get("type") or ""), container))
    if not candidates:
        return "", "", "", False, ""

    acceptable = _ACCEPTABLE_TYPES.get(kind)
    if acceptable:
        for doi, candidate_title, item_type, container in candidates:
            if item_type in acceptable:
                journal = container if item_type == "journal-article" else ""
                return doi, journal, candidate_title, False, ""
    doi, candidate_title, item_type, container = candidates[0]
    journal = container if item_type == "journal-article" else ""
    note = (
        f"crossref type '{item_type}' vs node type '{kind}'; may be a review"
        if acceptable
        else f"node states no source_type; crossref type '{item_type}' unverified"
    )
    return doi, journal, candidate_title, True, note


def resolve_isbn(
    client: HttpClient,
    title: str,
    author: str,
) -> Tuple[str, str, str]:
    """Resolve an ISBN (and landing page) from OpenLibrary, per step 2.

    :param client: HTTP client.
    :param title: Work title.
    :param author: First author.
    :returns: ``(isbn, landing_url, matched_title)``.
    """
    params = {"title": title[:150], "fields": "title,isbn,key,author_name", "limit": 5}
    if author:
        params["author"] = author
    body = client.get(f"{OPENLIBRARY_SEARCH}?{urllib.parse.urlencode(params)}")
    if not body:
        return "", "", ""
    try:
        docs = json.loads(body).get("docs", [])
    except (json.JSONDecodeError, AttributeError):
        return "", "", ""
    for doc in docs:
        candidate_title = doc.get("title") or ""
        if not titles_match(title, candidate_title)[0]:
            continue
        isbn = next((clean_isbn(i) for i in (doc.get("isbn") or []) if clean_isbn(i)), "")
        key = doc.get("key") or ""
        landing = f"{OPENLIBRARY_BASE}{key}" if key else ""
        if isbn or landing:
            return isbn, landing, candidate_title
    return "", "", ""


def resolve_all(
    client: HttpClient,
    works: Sequence[WorkIdentifiers],
    progress_every: int = 100,
) -> None:
    """Resolve contract fields for every work, in place.

    :param client: HTTP client.
    :param works: Works from :func:`group_works`.
    :param progress_every: Emit a progress line this often.
    """
    for index, work in enumerate(works, start=1):
        info = getattr(work, "_meta", {})
        authors = info.get("authors") or []
        author = str(authors[0]) if authors else ""
        work.url = clean_url(info.get("url"))
        work.journal = str(info.get("journal") or "")
        work.isbn = clean_isbn(info.get("isbn"))
        kind = _KG_TYPE_TO_KIND.get(str(info.get("source_type") or "").lower(), "")

        # Tier 1: DOI.
        work.doi = bare_doi(info.get("doi")) or doi_from_url(info.get("url"))
        if work.doi:
            work.note = "already on the node"
        else:
            doi, journal, matched, unverified, note = resolve_doi(
                client, work.title, author, info.get("year"), kind
            )
            work.doi, work.matched_title, work.note = doi, matched, note
            work.doi_unverified = unverified and bool(doi)
            if journal and not work.journal:
                work.journal = journal

        # Tier 2: ISBN, only when a DOI would not already win the link.
        if not work.doi and not work.isbn and kind in _BOOKISH:
            isbn, landing, matched = resolve_isbn(client, work.title, author)
            work.isbn = isbn
            if matched and not work.matched_title:
                work.matched_title = matched
            # Tier 3: an OpenLibrary work page is a stable landing page, which
            # beats dropping through to a bare search.
            if not work.url and landing:
                work.url = landing

        if index % progress_every == 0:
            print(f"  … {index}/{len(works)} works resolved", flush=True)


def write_identifiers(session: Any, works: Sequence[WorkIdentifiers]) -> int:
    """Write contract properties onto their BookArticle nodes.

    Only non-empty values are set, so a re-run never blanks a field another
    importer populated.

    :param session: An open Neo4j session.
    :param works: Resolved works.
    :returns: Number of nodes updated.
    """
    query = """
    UNWIND $rows AS row
    MATCH (b:BookArticle {article_id: row.article_id})
    SET b.doi  = CASE WHEN row.doi  <> '' THEN row.doi  ELSE b.doi  END,
        b.isbn = CASE WHEN row.isbn <> '' THEN row.isbn ELSE b.isbn END,
        b.url  = CASE WHEN row.url  <> '' THEN row.url  ELSE b.url  END,
        b.journal = CASE WHEN row.journal <> '' THEN row.journal ELSE b.journal END,
        b.doi_unverified = CASE WHEN row.doi <> '' THEN row.doi_unverified
                                ELSE b.doi_unverified END
    RETURN count(b) AS updated
    """
    rows = [
        {
            "article_id": article_id,
            "doi": work.doi,
            "isbn": work.isbn,
            "url": work.url,
            "journal": work.journal,
            "doi_unverified": work.doi_unverified,
        }
        for work in works
        for article_id in work.article_ids
    ]
    updated = 0
    for start in range(0, len(rows), 1000):
        record = session.run(query, rows=rows[start:start + 1000]).single()
        updated += record["updated"] if record else 0
    return updated


# ---------------------------------------------------------------------------
# Title join (contract: "Title join")
# ---------------------------------------------------------------------------


def es_books(index: str = "bibliography_text_only_0.7") -> List[Dict[str, str]]:
    """List the bibliography index's books with their exact titles and stems.

    ``es_book_id`` is the stem of the ES ``doc_id`` — ``malk-ish-1-426_p012``
    yields ``malk-ish-1-426`` — which is also the KG ``source_book``.

    :param index: Bibliography index name.
    :returns: ``[{es_book_id, es_title}]``.
    """
    from elasticsearch import Elasticsearch

    from src.datasets.indexing.elastic_index_genizah import es_config_from_env

    es = Elasticsearch(**es_config_from_env(), request_timeout=60, max_retries=1)
    result = es.search(index=index, size=0, aggs={
        "books": {"terms": {"field": "book_uuid", "size": 200}, "aggs": {
            "title": {"terms": {"field": "title.keyword", "size": 1}},
            "doc": {"terms": {"field": "doc_id", "size": 1}},
        }},
    })
    books = []
    for bucket in result["aggregations"]["books"]["buckets"]:
        titles = bucket["title"]["buckets"]
        docs = bucket["doc"]["buckets"]
        if not titles or not docs:
            continue
        books.append({
            "es_book_id": re.sub(r"_p\[?[^_]*\]?$", "", docs[0]["key"]),
            "es_title": titles[0]["key"],
        })
    return books


def match_join_keys(
    books: Sequence[Dict[str, str]],
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Pair each ES book with the BookArticle node describing the same work.

    Exact normalised-title equality first, then the strict prefix match. This is
    what replaces the serving side's leading-word probe, which misses bilingual
    titles such as ``Ginzei Kedem / גנזי קדם``.

    :param books: Rows from :func:`es_books`.
    :param rows: Rows from :func:`fetch_book_articles`.
    :returns: ``[{article_id, es_book_id, es_title}]``.
    """
    by_norm: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = normalize_title(row.get("title") or row.get("citation") or "")
        if key and row.get("article_id"):
            by_norm.setdefault(key, []).append(row)

    pairs: List[Dict[str, str]] = []
    for book in books:
        es_title = book["es_title"]
        # A bilingual ES title carries both forms; try each side too.
        variants = [es_title] + [p.strip() for p in re.split(r"\s+/\s+", es_title) if p.strip()]
        matched: List[Dict[str, Any]] = []
        for variant in variants:
            key = normalize_title(variant)
            if key in by_norm:
                matched = by_norm[key]
                break
        if not matched:
            for key, candidates in by_norm.items():
                if any(titles_match(variant, key)[0] for variant in variants):
                    matched = candidates
                    break
        for row in matched:
            pairs.append({
                "article_id": row["article_id"],
                "es_book_id": book["es_book_id"],
                "es_title": es_title,
            })
    return collapse_pairs(pairs)


def collapse_pairs(pairs: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Collect every ES book that maps to a node into one row per node.

    A multi-volume work is several ES books but one ``BookArticle`` node —
    *India traders of the middle ages* is scanned as ``india_trader_1_50``,
    ``india_trader_350_424`` and ``india_trader_426_500``. Writing one row per
    pair would let the last volume overwrite the others, so each node keeps the
    full list and a scalar primary for the single-book case.

    :param pairs: One row per (ES book, node) match.
    :returns: One row per node, carrying ``es_book_ids`` / ``es_titles`` lists.
    """
    grouped: Dict[str, Dict[str, Any]] = {}
    for pair in pairs:
        entry = grouped.setdefault(
            pair["article_id"],
            {"article_id": pair["article_id"], "es_book_ids": [], "es_titles": []},
        )
        if pair["es_book_id"] not in entry["es_book_ids"]:
            entry["es_book_ids"].append(pair["es_book_id"])
        if pair["es_title"] not in entry["es_titles"]:
            entry["es_titles"].append(pair["es_title"])
    for entry in grouped.values():
        entry["es_book_ids"].sort()
        entry["es_titles"].sort()
        # Scalars stay populated so a consumer expecting a single value still
        # works; the lists are authoritative when a work spans volumes.
        entry["es_book_id"] = entry["es_book_ids"][0]
        entry["es_title"] = entry["es_titles"][0]
        entry["ambiguous"] = len(entry["es_book_ids"]) > 1
    return sorted(grouped.values(), key=lambda e: e["article_id"])


def write_join_keys(session: Any, pairs: Sequence[Dict[str, str]]) -> int:
    """Write ``es_book_id`` / ``es_title`` onto the matched nodes.

    :param session: An open Neo4j session.
    :param pairs: Rows from :func:`match_join_keys`.
    :returns: Number of nodes updated.
    """
    query = """
    UNWIND $rows AS row
    MATCH (b:BookArticle {article_id: row.article_id})
    SET b.es_book_id = row.es_book_id,
        b.es_title = row.es_title,
        b.es_book_ids = row.es_book_ids,
        b.es_titles = row.es_titles
    RETURN count(b) AS updated
    """
    record = session.run(query, rows=[dict(p) for p in pairs]).single()
    return record["updated"] if record else 0


def write_report(works: Sequence[WorkIdentifiers], pairs: Sequence[Dict[str, str]],
                 path: Path) -> Dict[str, Any]:
    """Write the machine-readable result.

    :param works: Resolved works.
    :param pairs: Join-key pairs.
    :param path: Destination JSON path.
    :returns: The report dict.
    """
    tiers: Dict[str, int] = {}
    for work in works:
        tiers[work.tier] = tiers.get(work.tier, 0) + 1
    report = {
        "works": len(works),
        "nodes": sum(len(w.article_ids) for w in works),
        "tiers": tiers,
        "doi_unverified": sum(1 for w in works if w.doi_unverified),
        "join_pairs": len(pairs),
        "entries": [
            {"title": w.title, "doi": w.doi, "isbn": w.isbn, "url": w.url,
             "journal": w.journal, "tier": w.tier, "nodes": len(w.article_ids),
             "doi_unverified": w.doi_unverified, "matched_title": w.matched_title,
             "note": w.note}
            for w in works if w.tier != "search-fallback"
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    """CLI entry point: backfill direct-link identifiers per the contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE, help="HTTP cache")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Report path")
    parser.add_argument("--apply", action="store_true",
                        help="Write to Neo4j (default is a dry run)")
    parser.add_argument("--limit", type=int, default=0, help="Resolve only N works (0 = all)")
    parser.add_argument("--interval", type=float, default=0.6,
                        help="Minimum seconds between requests to one host")
    parser.add_argument("--join-only", action="store_true",
                        help="Only backfill es_book_id/es_title; skip all lookups")
    parser.add_argument("--include-serials", action="store_true",
                        help="Do not skip journal-title works (they have no DOI/ISBN)")
    parser.add_argument("--es-index", default="bibliography_text_only_0.7",
                        help="Bibliography index supplying the join keys")
    parser.add_argument("--es-host", default="localhost", help="ES host (default: local Docker)")
    parser.add_argument("--es-port", default="9200", help="ES port")
    parser.add_argument("--es-scheme", default="http", help="ES scheme")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    dotenv.load_dotenv(_REPO_ROOT / ".env")
    # Override, not setdefault: .env has already put the Cloudflare-forwarded
    # host into the environment, and that endpoint stalls on connect whenever
    # the tunnel is down.
    os.environ["ELASTIC_SEARCH_HOST"] = args.es_host
    os.environ["ELASTIC_SEARCH_PORT"] = args.es_port
    os.environ["ELASTIC_SEARCH_SCHEME"] = args.es_scheme
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
        notifications_min_severity="OFF",
    )
    database = os.environ.get("NEO4J_DATABASE", "neo4j")
    client = HttpClient(args.cache_dir, min_interval=args.interval)
    try:
        with driver.session(database=database) as session:
            rows = fetch_book_articles(session)

        pairs = match_join_keys(es_books(args.es_index), rows)
        works: List[WorkIdentifiers] = []
        if not args.join_only:
            works = group_works(rows, skip_serials=not args.include_serials)
            if args.limit:
                works = works[:args.limit]
            print(f"{len(rows)} BookArticle nodes -> {len(works)} distinct works to resolve")
            resolve_all(client, works)

        report = write_report(works, pairs, args.out)

        if args.apply:
            with driver.session(database=database) as session:
                updated = write_identifiers(session, works) if works else 0
                joined = write_join_keys(session, pairs)
            print(f"\nWrote identifiers to {updated} nodes; join keys to {joined} nodes")
        else:
            print("\nDry run — Neo4j not modified. Re-run with --apply to write.")
    finally:
        driver.close()

    print(
        f"{report['works']} works across {report['nodes']} nodes.\n"
        f"  link tiers: {report['tiers']}\n"
        f"  doi flagged unverified: {report['doi_unverified']}\n"
        f"  es_book_id/es_title join pairs: {report['join_pairs']}"
    )
    print(f"HTTP: {client.stats}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
