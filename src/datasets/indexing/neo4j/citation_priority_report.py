#!/usr/bin/env python3
"""Rank the secondary sources the KG cites most but the corpus has not scanned.

Every ``BookArticle`` node the graph holds is a work that some fragment's
bibliography points at. A handful of those works we have photographed and
indexed; the rest exist only as citations. The ones cited across many *distinct
fragments* are the highest-value scanning targets — each is a single trip to a
library that would attach text to hundreds of fragments at once.

Three things make the raw node list misleading, and this module corrects all
three:

* **One work, many nodes.** ``biblio`` and ``enriched`` imports create separate
  nodes for the same book, so citation counts are split across duplicates.
  Works are keyed on a normalised title plus year, which keeps the volumes of a
  multi-volume set apart (Goitein's *Mediterranean Society* volumes share a
  title and differ only by year) while merging true duplicates.
* **Unscannable placeholders.** PGP records scholars' unpublished transcriptions
  as works titled ``unpublished editions`` / ``Digital Editions``. These carry
  large citation counts but there is no book to scan, so they are separated out
  rather than topping the list.
* **"Do we have it?" is not one field.** ``source_books`` is only set on the
  ``enriched`` nodes built from our own scans, so a work we *have* scanned still
  shows up as an unscanned ``biblio`` node. Local availability is therefore
  resolved by title against the scanned corpus as well.

Usage::

    python -m src.datasets.indexing.neo4j.citation_priority_report --top 40
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import dotenv

logger = __import__("logging").getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LITERATURE_ROOT = _REPO_ROOT / "src/datasets/raw_data/cairo_genizah/academic_literature"
DEFAULT_OUT_DIR = _REPO_ROOT / "artifacts/citation_priority"

#: Works whose "title" is a stand-in for an unpublished scholarly transcription.
#: They are real citations but nothing exists to photograph.
_PLACEHOLDER_TITLES = {
    "unpublished editions",
    "digital editions",
    "unpublished edition",
    "digital edition",
    "unpublished",
    "personal communication",
}

#: Leading articles dropped when normalising a title for matching.
_LEADING_ARTICLES = ("a ", "an ", "the ")

#: Minimum normalised-title length before one title may be matched to another by
#: containment. Short titles are substrings of too many longer ones.
_MIN_CONTAINMENT_LEN = 20


@dataclass
class CitedWork:
    """One secondary source aggregated across its duplicate KG nodes.

    :param key: Normalised identity (title key plus year).
    :param title: Best display title seen across the merged nodes.
    :param year: Publication year, when recorded.
    :param authors: Scholars the graph links to the work via ``WROTE``.
    :param fragments: Distinct fragments citing the work — the priority metric.
    :param citations: Total citation edges (fragments may cite twice).
    :param node_count: How many KG nodes were merged into this work.
    :param source_books: Local scan directories, when the work is already ours.
    :param data_sources: Which importers contributed (biblio/pgp/enriched).
    :param source_type: PGP's classification, e.g. ``Unpublished``.
    :param publisher: Publisher, when recorded.
    :param journal: Journal, when recorded.
    :param placeholder: The "work" is an unpublished-edition stand-in.
    :param serial: The title names a journal, not a single publication.
    :param local_scan: Name of the scanned book this matched, if any.
    """

    key: Tuple[str, Optional[int]]
    title: str
    year: Optional[int] = None
    authors: Set[str] = field(default_factory=set)
    fragments: int = 0
    citations: int = 0
    node_count: int = 0
    source_books: Set[str] = field(default_factory=set)
    data_sources: Set[str] = field(default_factory=set)
    source_type: Optional[str] = None
    publisher: Optional[str] = None
    journal: Optional[str] = None
    placeholder: bool = False
    serial: bool = False
    local_scan: Optional[str] = None

    @property
    def have_locally(self) -> bool:
        """:returns: True when a scan of this work is already in the corpus."""
        return bool(self.source_books or self.local_scan)

    @property
    def kind(self) -> str:
        """:returns: ``placeholder``, ``serial`` or ``monograph``."""
        if self.placeholder:
            return "placeholder"
        return "serial" if self.serial else "monograph"


def normalize_title(title: Optional[str]) -> str:
    """Reduce a title to a comparable key.

    Case-folds, strips accents and punctuation, drops a leading article, and
    collapses whitespace, so ``A Mediterranean society: the Jewish…`` and
    ``A Mediterranean society; the Jewish…`` land on the same key.

    :param title: Raw title.
    :returns: The normalised key, or ``""`` when the title is empty.
    """
    if not title:
        return ""
    text = unicodedata.normalize("NFKD", title).casefold()
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for article in _LEADING_ARTICLES:
        if text.startswith(article):
            text = text[len(article):]
            break
    return text


def is_placeholder(title: Optional[str], source_type: Optional[str]) -> bool:
    """Return whether a "work" stands in for an unpublished transcription.

    PGP records a scholar's own unpublished reading of a fragment as a source.
    Some carry the title ``unpublished editions``; others have no title at all,
    so the citation — a bare personal name like ``Avraham David.`` — becomes the
    display title. Both are caught, because neither can be photographed.

    :param title: The work's title.
    :param source_type: PGP's ``source_type`` classification.
    :returns: True when there is no physical publication to scan.
    """
    if (source_type or "").strip().lower() == "unpublished":
        return True
    return normalize_title(title) in _PLACEHOLDER_TITLES


def scanned_corpus_titles(root: Path) -> Dict[str, str]:
    """Map normalised titles of already-scanned books to their book directory.

    Reads every ``*_metadata.json`` under the academic-literature root, which is
    the same set the bibliography indexer works from.

    :param root: The academic_literature root.
    :returns: Map of normalised title to a representative directory name.
    """
    titles: Dict[str, str] = {}
    for path in root.rglob("*_metadata.json"):
        if path.name == "example_book_metadata.json":
            continue
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("Unreadable metadata: %s", path)
            continue
        if not isinstance(metadata, dict):
            continue
        key = normalize_title(metadata.get("title"))
        if key:
            titles.setdefault(key, path.parent.name)
        series = metadata.get("series")
        if isinstance(series, dict):
            series_key = normalize_title(series.get("name"))
            if series_key:
                titles.setdefault(series_key, path.parent.name)
    return titles


def match_local(work: CitedWork, scanned: Dict[str, str]) -> Optional[str]:
    """Find the scanned book corresponding to a cited work, if we hold one.

    Matches on the normalised title, then on containment either way, which
    catches subtitle differences between a citation and our own metadata
    (``India traders of the middle ages`` vs the fuller catalogue title).

    :param work: The aggregated cited work.
    :param scanned: Map from :func:`scanned_corpus_titles`.
    :returns: The matching book directory name, or ``None``.
    """
    key = work.key[0]
    if not key:
        return None
    if key in scanned:
        return scanned[key]
    # Containment is only meaningful when *both* titles are long. A short key
    # such as "geniza" is a substring of half the corpus and must not match.
    if len(key) < _MIN_CONTAINMENT_LEN:
        return None
    for scanned_key, book in scanned.items():
        if len(scanned_key) < _MIN_CONTAINMENT_LEN:
            continue
        if scanned_key in key or key in scanned_key:
            return book
    return None


def fetch_cited_works(session: Any) -> List[Dict[str, Any]]:
    """Read every BookArticle with its citation counts from Neo4j.

    Counts both directions of the citation model: ``BookArticle-[:REFERENCES]->
    Fragment`` (from the bibliography import) and ``Fragment-[:CITED_IN]->
    BookArticle`` (from the academic-literature import).

    :param session: An open Neo4j session.
    :returns: One dict per BookArticle node.
    """
    # Fragment element ids are returned rather than counted in Cypher: two
    # duplicate nodes for one work may cite the same fragment, and that must
    # count once after the merge in :func:`aggregate_works`.
    query = """
    MATCH (b:BookArticle)
    OPTIONAL MATCH (b)-[:REFERENCES]->(f1:Fragment)
    WITH b, collect(DISTINCT elementId(f1)) AS refs
    OPTIONAL MATCH (f2:Fragment)-[:CITED_IN]->(b)
    WITH b, refs, collect(DISTINCT elementId(f2)) AS cited
    OPTIONAL MATCH (a)-[:WROTE]->(b)
    WITH b, refs, cited, [n IN collect(DISTINCT a.name) WHERE n IS NOT NULL] AS authors
    WITH b, authors,
         [x IN refs + cited WHERE x IS NOT NULL] AS all_frags,
         [x IN refs WHERE x IS NOT NULL] AS refs,
         [x IN cited WHERE x IS NOT NULL] AS cited
    RETURN b.title AS title, b.citation AS citation, b.year AS year,
           b.volume AS volume, b.publisher AS publisher, b.journal AS journal,
           b.source_type AS source_type, b.source_books AS source_books,
           b.data_sources AS data_sources,
           size(refs) + size(cited) AS edge_total,
           all_frags AS fragment_ids,
           authors AS authors
    """
    return [dict(record) for record in session.run(query)]


def _as_int(value: Any) -> Optional[int]:
    """Coerce a Neo4j numeric property to ``int``.

    :param value: Raw property value (often a float).
    :returns: The integer value, or ``None``.
    """
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def find_serial_titles(
    rows: Sequence[Dict[str, Any]],
    min_nodes: int = 20,
    min_years: int = 10,
) -> Set[str]:
    """Identify titles that name a journal rather than a single publication.

    A journal accumulates one KG node per cited article, so its title appears on
    scores of nodes spread across decades — ``Jewish Quarterly Review`` and
    ``Revue des études juives`` show up this way. A multi-volume monograph looks
    different: Goitein's *Mediterranean Society* is one node per volume.

    This catches serials whose nodes carry years. :func:`mark_serials` catches
    the rest, where the year is missing but many authors are not.

    :param rows: Rows from :func:`fetch_cited_works`.
    :param min_nodes: Node count at or above which a title reads as a serial.
    :param min_years: Distinct-year count that likewise marks a serial.
    :returns: The set of normalised titles judged to be serials.
    """
    nodes: Dict[str, int] = defaultdict(int)
    years: Dict[str, Set[int]] = defaultdict(set)
    for row in rows:
        key = normalize_title(row.get("title") or row.get("citation"))
        if not key:
            continue
        nodes[key] += 1
        year = _as_int(row.get("year"))
        if year is not None:
            years[key].add(year)
    return {
        key
        for key, count in nodes.items()
        if count >= min_nodes and len(years[key]) >= min_years
    }


def mark_serials(
    works: Sequence[CitedWork],
    min_nodes: int = 10,
    min_authors: int = 8,
) -> None:
    """Flag aggregated works whose shape is a journal run, not a book.

    Most Hebrew-language journals in this graph carry no year at all, so their
    nodes all merge into one work and :func:`find_serial_titles` cannot see the
    spread of issues. What still separates them cleanly is authorship: ``סיני``
    merges 79 nodes written by 31 different people, while every volume of
    *Mediterranean Society* is one node by Goitein alone. A work that many
    people wrote is a journal, not something to fetch from a shelf.

    :param works: Aggregated works, mutated in place.
    :param min_nodes: Merged-node count at or above which the shape reads serial.
    :param min_authors: Distinct-author count that must accompany it.
    """
    for work in works:
        if work.node_count >= min_nodes and len(work.authors) >= min_authors:
            work.serial = True


def aggregate_works(
    rows: Iterable[Dict[str, Any]],
    serial_titles: Optional[Set[str]] = None,
) -> List[CitedWork]:
    """Merge duplicate BookArticle nodes into one record per work.

    :param rows: Rows from :func:`fetch_cited_works`.
    :param serial_titles: Normalised titles to mark as serials.
    :returns: Aggregated works, most-cited first.
    """
    serial_titles = serial_titles or set()
    works: Dict[Tuple[str, Optional[int]], CitedWork] = {}
    fragment_sets: Dict[Tuple[str, Optional[int]], Set[str]] = defaultdict(set)
    for row in rows:
        title = row.get("title") or row.get("citation") or ""
        title_key = normalize_title(title)
        if not title_key:
            continue
        year = _as_int(row.get("year"))
        # Year separates the volumes of a set that share one title; without a
        # year the work merges on title alone.
        key = (title_key, year)
        work = works.get(key)
        if work is None:
            work = CitedWork(
                key=key,
                title=title.strip(),
                year=year,
                serial=title_key in serial_titles,
            )
            works[key] = work
        work.node_count += 1
        work.citations += row.get("edge_total") or 0
        fragment_sets[key].update(row.get("fragment_ids") or [])
        work.source_books.update(row.get("source_books") or [])
        work.data_sources.update(row.get("data_sources") or [])
        work.authors.update(row.get("authors") or [])
        work.source_type = work.source_type or row.get("source_type")
        work.publisher = work.publisher or row.get("publisher")
        work.journal = work.journal or row.get("journal")
        if len(title.strip()) > len(work.title):
            work.title = title.strip()
        work.placeholder = work.placeholder or is_placeholder(title, row.get("source_type"))
    for key, work in works.items():
        work.fragments = len(fragment_sets[key])
    return sorted(works.values(), key=lambda w: (-w.fragments, w.title))


def resolve_availability(works: Sequence[CitedWork], scanned: Dict[str, str]) -> None:
    """Stamp each work with the scanned book it corresponds to, if any.

    :param works: Aggregated works (mutated in place).
    :param scanned: Map from :func:`scanned_corpus_titles`.
    """
    for work in works:
        work.local_scan = match_local(work, scanned)


def write_csv(works: Sequence[CitedWork], path: Path) -> None:
    """Write the ranked priority list.

    :param works: Aggregated works, already ordered.
    :param path: Destination CSV path.
    """
    columns = [
        "rank", "fragments", "citations", "title", "year", "authors",
        "publisher", "journal", "have_locally", "local_scan", "source_books",
        "kind", "data_sources", "kg_nodes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for rank, work in enumerate(works, start=1):
            writer.writerow({
                "rank": rank,
                "fragments": work.fragments,
                "citations": work.citations,
                "title": work.title,
                "year": work.year or "",
                "authors": "; ".join(sorted(work.authors)),
                "publisher": work.publisher or "",
                "journal": work.journal or "",
                "have_locally": work.have_locally,
                "local_scan": work.local_scan or "",
                "source_books": "; ".join(sorted(work.source_books)),
                "kind": work.kind,
                "data_sources": "; ".join(sorted(work.data_sources)),
                "kg_nodes": work.node_count,
            })


def write_summary(works: Sequence[CitedWork], path: Path) -> Dict[str, Any]:
    """Write the machine-readable rollup.

    :param works: All aggregated works.
    :param path: Destination JSON path.
    :returns: The report dict that was written.
    """
    outstanding = [w for w in works if not w.placeholder and not w.have_locally]
    targets = [w for w in outstanding if not w.serial]
    serials = [w for w in outstanding if w.serial]

    def _rows(items: Sequence[CitedWork], limit: int) -> List[Dict[str, Any]]:
        """Render works as report rows.

        :param items: Works to render, already ordered.
        :param limit: Maximum rows.
        :returns: Serialisable row dicts.
        """
        return [
            {
                "rank": rank,
                "title": w.title,
                "year": w.year,
                "fragments": w.fragments,
                "citations": w.citations,
                "authors": sorted(w.authors),
                "publisher": w.publisher,
                "kind": w.kind,
                "kg_nodes": w.node_count,
            }
            for rank, w in enumerate(items[:limit], start=1)
        ]

    report = {
        "works_total": len(works),
        "works_with_local_scan": sum(1 for w in works if w.have_locally),
        "placeholder_works": sum(1 for w in works if w.placeholder),
        "scan_targets": len(targets),
        "serial_targets": len(serials),
        "fragments_behind_targets": sum(w.fragments for w in targets),
        "fragments_behind_serials": sum(w.fragments for w in serials),
        "top_targets": _rows(targets, 100),
        "top_serials": _rows(serials, 25),
        "placeholders": [
            {"title": w.title, "year": w.year, "fragments": w.fragments}
            for w in sorted(
                (w for w in works if w.placeholder),
                key=lambda w: -w.fragments,
            )[:25]
        ],
        "already_local": [
            {"title": w.title, "fragments": w.fragments, "local_scan": w.local_scan,
             "source_books": sorted(w.source_books)}
            for w in sorted(
                (w for w in works if w.have_locally), key=lambda w: -w.fragments
            )[:50]
        ],
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def print_report(works: Sequence[CitedWork], report: Dict[str, Any], top: int) -> None:
    """Print the ranked scanning priority list.

    :param works: All aggregated works.
    :param report: The rollup from :func:`write_summary`.
    :param top: How many targets to print.
    """
    print(
        f"{report['works_total']} distinct works cited in the KG: "
        f"{report['works_with_local_scan']} already scanned, "
        f"{report['placeholder_works']} unpublished-edition placeholders (nothing to scan).\n"
        f"Outstanding: {report['scan_targets']} monographs covering "
        f"{report['fragments_behind_targets']} fragment citations, plus "
        f"{report['serial_targets']} journal runs covering "
        f"{report['fragments_behind_serials']}.\n"
    )
    header = f"{'#':>3s} {'frags':>6s} {'yr':>6s}  {'title':66s} authors"
    print("MONOGRAPHS — one trip each, highest citation weight first")
    print(header)
    print("-" * 108)
    for row in report["top_targets"][:top]:
        authors = ", ".join(row["authors"][:2]) or "-"
        print(
            f"{row['rank']:3d} {row['fragments']:6d} {str(row['year'] or '-'):>6s}  "
            f"{row['title'][:66]:66s} {authors[:28]}"
        )
    print("\nJOURNAL RUNS — many issues each, treat as a serial acquisition")
    for row in report["top_serials"][:10]:
        print(
            f"{row['rank']:3d} {row['fragments']:6d} {str(row['year'] or '-'):>6s}  "
            f"{row['title'][:66]:66s} ({row['kg_nodes']} cited articles)"
        )
    print("\nAlready held locally (no trip needed), by citation weight:")
    for row in report["already_local"][:10]:
        print(f"  {row['fragments']:6d}  {row['title'][:72]:72s} -> {row['local_scan'] or row['source_books']}")
    print("\nUnpublished-edition placeholders (excluded — no physical source):")
    for row in report["placeholders"][:5]:
        print(f"  {row['fragments']:6d}  {row['title']} ({row['year'] or 'n.d.'})")


def main() -> None:
    """CLI entry point: rank uncollected sources by citation weight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument(
        "--literature-root", type=Path, default=DEFAULT_LITERATURE_ROOT,
        help="academic_literature root, used to detect works we already hold",
    )
    parser.add_argument("--top", type=int, default=30, help="Targets to print")
    parser.add_argument(
        "--min-fragments", type=int, default=1,
        help="Drop works cited by fewer than this many fragments",
    )
    args = parser.parse_args()

    dotenv.load_dotenv(_REPO_ROOT / ".env")
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
        notifications_min_severity="OFF",
    )
    try:
        with driver.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as session:
            rows = fetch_cited_works(session)
    finally:
        driver.close()

    serial_titles = find_serial_titles(rows)
    works = [
        w for w in aggregate_works(rows, serial_titles) if w.fragments >= args.min_fragments
    ]
    mark_serials(works)
    resolve_availability(works, scanned_corpus_titles(args.literature_root))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "citation_priority.csv"
    json_path = args.out_dir / "citation_priority_summary.json"
    write_csv(works, csv_path)
    report = write_summary(works, json_path)
    print_report(works, report, args.top)
    print(f"\nWrote {csv_path}\nWrote {json_path}")


if __name__ == "__main__":
    main()
