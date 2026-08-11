#!/usr/bin/env python3
"""Rank sources by how *concentrated* they are on the Genizah, not how often cited.

:mod:`src.datasets.indexing.neo4j.citation_priority_report` ranks by raw citation
count, which systematically favours large reference works. Kittel's *Biblia
Hebraica* is cited by 52 fragments not because it is about the Cairo Genizah but
because Genizah biblical fragments are collated against it — scanning its ~1,400
pages would add almost nothing to the corpus. The same logic inflates
dictionaries, grammars and concordances.

What is actually wanted is a work that is *entirely* Genizah material and small
enough to scan and index as one unit, with no splicing of relevant pages out of
an irrelevant volume. This module ranks on that basis, combining:

* **Topicality** — does the title or journal say this is Genizah work? Reference
  works that merely serve as a collation base are demoted explicitly, so it is
  visible *why* a heavily-cited title dropped out.
* **Extent** — the printed page range, parsed from the KG's ``pages`` field.
  Articles and chapters carry one; monographs do not.
* **Density** — distinct citing fragments per page. A 38-page article cited by
  19 fragments is an order of magnitude denser than a 1,400-page Bible edition
  cited by 52.
* **Splice-free** — whether the whole unit can be scanned and indexed as-is.

Output is grouped by acquisition shape rather than one flat ranking, because the
work involved differs: a journal article is a PDF request, a focused monograph is
a whole-book scan, and a reference work means extracting pages by hand.

Usage::

    python -m src.datasets.indexing.neo4j.genizah_focus_report --top 40
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import dotenv

from src.datasets.indexing.neo4j.citation_priority_report import (
    DEFAULT_LITERATURE_ROOT,
    normalize_title,
    scanned_corpus_titles,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT_DIR = _REPO_ROOT / "artifacts/genizah_focus"

#: Title/journal substrings that mark a work as Genizah scholarship. Hebrew and
#: transliterated spellings both appear in this corpus.
_TOPICAL_MARKERS: Tuple[str, ...] = (
    "geniza", "genizah", "gueniza", "guenizah", "genisa",
    "גניז",                       # covers גניזה / גניזת / מגניזת
    "taylor-schechter", "t-s ", "t.-s.",
    "fustat", "fustāt", "fusṭāṭ", "פסטאט",
    "ben ezra synagogue",
    "elkan nathan adler", "ena ",
    "westminster college",
)

#: Weaker signals — Genizah-adjacent subject matter that usually travels with it.
_ADJACENT_MARKERS: Tuple[str, ...] = (
    "judaeo-arabic", "judeo-arabic", "ערבית-יהודית",
    "cairo", "קהיר", "כאהיר",
    "fatimid", "פאטימי",
    "nagid", "נגיד",
    "gaon", "geonic", "גאון",
)

#: Works cited as a *collation base* or lookup tool rather than for their Genizah
#: content. High citation counts here are an artefact of how editors cite.
_REFERENCE_MARKERS: Tuple[str, ...] = (
    "biblia hebraica", "biblia sacra",
    "dictionary", "מילון", "lexicon", "concordance", "קונקורדנצ",
    "grammar", "דקדוק", "thesaurus", "אוצר המילים",
    "encyclopaedia", "encyclopedia", "אנציקלופדי",
    "bibliography", "ביבליוגרפי",
    "catalogue of the", "catalog of the",
)

#: Above this many pages a unit cannot be scanned casually as one piece.
SPLICE_FREE_MAX_PAGES = 150

#: Page-range formats seen in the corpus: ``67-104``, ``281–345``, ``247–92``
#: (abbreviated end), ``pp. 141-142``.
_PAGE_RANGE = re.compile(r"(\d{1,4})\s*[-–—]\s*(\d{1,4})")
_SINGLE_PAGE = re.compile(r"(\d{1,4})")

#: URL host -> how hard the full text is to reach.
_ACCESS_BY_HOST: Dict[str, str] = {
    "www.jstor.org": "institutional (JSTOR)",
    "jstor.org": "institutional (JSTOR)",
    "brill.com": "institutional (Brill)",
    "muse.jhu.edu": "institutional (Project MUSE)",
    "www.academia.edu": "likely free (Academia.edu)",
    "academia.edu": "likely free (Academia.edu)",
    "www.lib.cam.ac.uk": "open (Cambridge Genizah Unit)",
    "www.persee.fr": "open (Persée)",
    "persee.fr": "open (Persée)",
    "www.nli.org.il": "open (NLI)",
    "bookreader.nli.org.il": "open (NLI reader)",
    "www.ybz.org.il": "open (Ben-Zvi Institute)",
    "sefarad.revistas.csic.es": "open (CSIC)",
    "doi.org": "resolve DOI",
    "catalog.princeton.edu": "catalogue record only",
    "drive.google.com": "shared file",
}


@dataclass
class FocusedWork:
    """One cited work scored for Genizah focus and scanning effort.

    :param title: Work title.
    :param source_type: KG classification (Article, Book, Book Section …).
    :param journal: Journal or containing volume.
    :param year: Publication year.
    :param pages: Raw page range string from the KG.
    :param url: Publisher/repository link, when recorded.
    :param authors: Authors linked by ``WROTE``.
    :param fragments: Distinct fragments citing the work.
    :param extent: Parsed page count, or ``None``.
    :param topicality: ``core`` / ``adjacent`` / ``reference`` / ``unclear``.
    :param matched_markers: Which markers fired, for auditing.
    """

    title: str
    source_type: str = ""
    journal: str = ""
    year: str = ""
    pages: str = ""
    url: str = ""
    authors: List[str] = field(default_factory=list)
    fragments: int = 0
    extent: Optional[int] = None
    topicality: str = "unclear"
    matched_markers: List[str] = field(default_factory=list)

    @property
    def density(self) -> Optional[float]:
        """Distinct citing fragments per printed page.

        :returns: The density, or ``None`` when the extent is unknown.
        """
        if not self.extent:
            return None
        return round(self.fragments / self.extent, 3)

    @property
    def impact(self) -> float:
        """Rank score for self-contained units: citation weight times density.

        Density alone is the wrong objective — it puts a two-page note cited
        three times above a 38-page article cited nineteen times. Multiplying by
        the citation count restores the balance, so the score rewards a work
        that is both concentrated *and* actually load-bearing for the corpus.

        :returns: ``fragments² / extent``, or 0 when the extent is unknown.
        """
        if not self.extent:
            return 0.0
        return round(self.fragments ** 2 / self.extent, 2)

    @property
    def unit(self) -> str:
        """Classify the acquisition shape.

        :returns: ``article``, ``chapter``, ``dissertation``, ``book`` or
            ``unknown``.
        """
        return {
            "Article": "article",
            "Book Section": "chapter",
            "Dissertation": "dissertation",
            "Book": "book",
        }.get(self.source_type, "unknown")

    @property
    def splice_free(self) -> bool:
        """Whether the whole unit can be scanned and indexed without extraction.

        An article or chapter of known, modest extent is self-contained. A
        monograph qualifies only when its title marks the whole book as Genizah
        scholarship — otherwise the relevant pages must be found and spliced.

        :returns: True when the unit needs no page-level extraction.
        """
        if self.topicality == "reference":
            return False
        if self.unit in ("article", "chapter"):
            return bool(self.extent and self.extent <= SPLICE_FREE_MAX_PAGES)
        return self.topicality == "core"

    @property
    def access_hint(self) -> str:
        """Describe how reachable the full text is, from the recorded URL.

        :returns: A short human-readable hint, or ``""`` when no URL exists.
        """
        if not self.url:
            return ""
        host = urlparse(self.url).netloc.lower()
        return _ACCESS_BY_HOST.get(host, host or "")


def _fold(text: Optional[str]) -> str:
    """Case-fold and strip accents for marker matching.

    :param text: Raw text.
    :returns: The folded string.
    """
    if not text:
        return ""
    folded = unicodedata.normalize("NFKD", text).casefold()
    return "".join(ch for ch in folded if not unicodedata.combining(ch))


def classify_topicality(title: str, journal: str = "") -> Tuple[str, List[str]]:
    """Judge whether a work is Genizah scholarship or merely cited by it.

    Reference works are tested first and win outright: a title like *A
    Dictionary of Judaeo-Arabic* contains an adjacent marker but is a lookup
    tool, and the corpus does not need its pages.

    :param title: Work title.
    :param journal: Journal or containing volume title.
    :returns: ``(topicality, matched_markers)`` where topicality is ``core``,
        ``adjacent``, ``reference`` or ``unclear``.
    """
    haystack = f"{_fold(title)} {_fold(journal)}"
    reference = [m for m in _REFERENCE_MARKERS if _fold(m) in haystack]
    core = [m for m in _TOPICAL_MARKERS if _fold(m) in haystack]
    # A Genizah catalogue *is* Genizah scholarship even though "catalogue of the"
    # is a reference marker, so an explicit core marker overrides.
    if core:
        return "core", core
    if reference:
        return "reference", reference
    adjacent = [m for m in _ADJACENT_MARKERS if _fold(m) in haystack]
    if adjacent:
        return "adjacent", adjacent
    return "unclear", []


def parse_extent(pages: Optional[str]) -> Optional[int]:
    """Parse a printed page count from the KG's ``pages`` string.

    Handles ``67-104``, en/em-dashed ranges, and the abbreviated end pages
    scholarly citation uses (``247–92`` means 247–292).

    :param pages: Raw ``pages`` value.
    :returns: The page count, or ``None`` when unparseable.
    """
    if not pages:
        return None
    text = str(pages)
    match = _PAGE_RANGE.search(text)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        if end < start:
            # Abbreviated end page: 247–92 -> 247–292; 1198–204 -> 1198–1204.
            magnitude = 10 ** len(match.group(2))
            end = (start // magnitude) * magnitude + end
            if end < start:
                end += magnitude
        extent = end - start + 1
        return extent if 0 < extent <= 2000 else None
    single = _SINGLE_PAGE.search(text)
    return 1 if single else None


def fetch_works(session: Any) -> List[Dict[str, Any]]:
    """Read every cited work with the metadata this ranking needs.

    :param session: An open Neo4j session.
    :returns: One dict per BookArticle node.
    """
    query = """
    MATCH (b:BookArticle)
    OPTIONAL MATCH (b)-[:REFERENCES]->(f1:Fragment)
    WITH b, collect(DISTINCT elementId(f1)) AS refs
    OPTIONAL MATCH (f2:Fragment)-[:CITED_IN]->(b)
    WITH b, refs, collect(DISTINCT elementId(f2)) AS cited
    OPTIONAL MATCH (a)-[:WROTE]->(b)
    WITH b, refs, cited, [n IN collect(DISTINCT a.name) WHERE n IS NOT NULL] AS authors
    RETURN b.title AS title, b.citation AS citation, b.source_type AS source_type,
           b.journal AS journal, b.year AS year, b.pages AS pages, b.url AS url,
           b.source_books AS source_books,
           size([x IN refs + cited WHERE x IS NOT NULL]) AS fragments,
           authors AS authors
    """
    return [dict(record) for record in session.run(query)]


def build_works(
    rows: Sequence[Dict[str, Any]],
    scanned: Optional[Dict[str, str]] = None,
) -> List[FocusedWork]:
    """Score every row for topicality, extent and density.

    Works already scanned locally are dropped — they are not acquisition
    targets. ``source_books`` alone is not enough to detect them: the biblio and
    enriched importers create separate nodes for one work and only the enriched
    one carries that field, so the title is checked against the scanned corpus
    too.

    :param rows: Rows from :func:`fetch_works`.
    :param scanned: Normalised titles of already-scanned books, from
        :func:`~src.datasets.indexing.neo4j.citation_priority_report.scanned_corpus_titles`.
    :returns: Scored works, most-cited first.
    """
    scanned = scanned or {}
    works: List[FocusedWork] = []
    for row in rows:
        if row.get("source_books"):
            continue
        if _already_scanned(row.get("title") or row.get("citation") or "", scanned):
            continue
        title = (row.get("title") or row.get("citation") or "").strip()
        if not title:
            continue
        journal = (row.get("journal") or "").strip()
        topicality, markers = classify_topicality(title, journal)
        year = row.get("year")
        works.append(FocusedWork(
            title=title,
            source_type=(row.get("source_type") or "").strip(),
            journal=journal,
            year=str(int(float(year))) if isinstance(year, (int, float)) else "",
            pages=(row.get("pages") or "").strip(),
            url=(row.get("url") or "").strip(),
            authors=sorted(row.get("authors") or []),
            fragments=int(row.get("fragments") or 0),
            extent=parse_extent(row.get("pages")),
            topicality=topicality,
            matched_markers=markers,
        ))
    return sorted(works, key=lambda w: (-w.fragments, w.title))


def _already_scanned(title: str, scanned: Dict[str, str]) -> bool:
    """Whether a title corresponds to a book the corpus already holds.

    :param title: Work title.
    :param scanned: Map of normalised title to book directory.
    :returns: True when a scan of this work already exists.
    """
    key = normalize_title(title)
    if not key:
        return False
    if key in scanned:
        return True
    # Containment only between titles long enough for it to mean something.
    return len(key) >= 20 and any(
        len(other) >= 20 and (other in key or key in other) for other in scanned
    )


def group_works(works: Sequence[FocusedWork]) -> Dict[str, List[FocusedWork]]:
    """Split works by the shape of the acquisition job.

    :param works: Scored works.
    :returns: Map of group name to works, each sorted by citation weight.
    """
    groups: Dict[str, List[FocusedWork]] = {
        "self_contained_articles": [],
        "focused_monographs": [],
        "probable_monographs": [],
        "splice_required": [],
        "off_topic": [],
    }
    for work in works:
        if work.fragments <= 0:
            continue
        if work.topicality == "reference":
            groups["splice_required"].append(work)
        elif work.unit in ("article", "chapter") and work.splice_free:
            groups["self_contained_articles"].append(work)
        elif work.topicality == "core":
            groups["focused_monographs"].append(work)
        elif work.topicality == "adjacent":
            # Everything in this graph is cited by Genizah fragments, so an
            # adjacent marker (nagid, gaon, Fatimid, Judaeo-Arabic) is a
            # positive signal here rather than a weak one — it is just not
            # self-evident from the title, so it wants a human glance.
            groups["probable_monographs"].append(work)
        else:
            groups["off_topic"].append(work)
    for name, items in groups.items():
        # Articles rank on density (concentration per page); everything else on
        # raw citation weight, since no extent is known for whole books.
        if name == "self_contained_articles":
            items.sort(key=lambda w: (-w.impact, -w.fragments))
        else:
            items.sort(key=lambda w: -w.fragments)
    return groups


def write_csv(works: Sequence[FocusedWork], groups: Dict[str, List[FocusedWork]],
              path: Path) -> None:
    """Write the per-work table.

    :param works: All scored works.
    :param groups: Grouping from :func:`group_works`.
    :param path: Destination CSV path.
    """
    group_of = {id(w): name for name, items in groups.items() for w in items}
    columns = [
        "group", "fragments", "impact", "density", "extent_pages", "topicality", "unit",
        "splice_free", "title", "journal", "year", "pages", "authors",
        "access_hint", "url", "matched_markers",
    ]
    ordered = [w for name in groups for w in groups[name]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for work in ordered:
            writer.writerow({
                "group": group_of.get(id(work), ""),
                "fragments": work.fragments,
                "impact": work.impact or "",
                "density": work.density if work.density is not None else "",
                "extent_pages": work.extent or "",
                "topicality": work.topicality,
                "unit": work.unit,
                "splice_free": work.splice_free,
                "title": work.title,
                "journal": work.journal,
                "year": work.year,
                "pages": work.pages,
                "authors": "; ".join(work.authors),
                "access_hint": work.access_hint,
                "url": work.url,
                "matched_markers": "; ".join(work.matched_markers),
            })


def write_summary(groups: Dict[str, List[FocusedWork]], path: Path) -> Dict[str, Any]:
    """Write the machine-readable rollup.

    :param groups: Grouping from :func:`group_works`.
    :param path: Destination JSON path.
    :returns: The report dict that was written.
    """
    report = {
        "groups": {
            name: {
                "works": len(items),
                "fragments": sum(w.fragments for w in items),
                "entries": [
                    {
                        "title": w.title, "journal": w.journal, "year": w.year,
                        "fragments": w.fragments, "extent_pages": w.extent,
                        "density": w.density, "impact": w.impact, "unit": w.unit,
                        "topicality": w.topicality, "authors": w.authors,
                        "access_hint": w.access_hint, "url": w.url,
                    }
                    for w in items[:120]
                ],
            }
            for name, items in groups.items()
        },
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


GROUP_LABELS = {
    "self_contained_articles": (
        "SELF-CONTAINED ARTICLES & CHAPTERS — scan whole, no splicing"
    ),
    "focused_monographs": "GENIZAH BOOKS — title states it, whole volume on topic",
    "probable_monographs": "PROBABLY ON TOPIC — Genizah-adjacent subject, verify",
    "splice_required": "SPLICE REQUIRED — a lookup tool, not a Genizah work",
    "off_topic": "NO TOPICAL SIGNAL — title says nothing about the Genizah",
}


def print_report(groups: Dict[str, List[FocusedWork]], top: int) -> None:
    """Print the grouped report.

    :param groups: Grouping from :func:`group_works`.
    :param top: Entries to show per group.
    """
    for name in ("self_contained_articles", "focused_monographs",
                 "probable_monographs", "splice_required", "off_topic"):
        items = groups[name]
        print(f"{GROUP_LABELS[name]:62s} {len(items):5d} works, "
              f"{sum(w.fragments for w in items):6d} fragments")

    print(f"\n=== {GROUP_LABELS['self_contained_articles']} ===")
    print(f"{'frg':>4s} {'pp':>4s} {'dens':>5s} {'impact':>6s}  {'title':46s} {'access':26s}")
    print("-" * 100)
    for work in groups["self_contained_articles"][:top]:
        print(f"{work.fragments:4d} {work.extent or 0:4d} {work.density or 0:5.2f} "
              f"{work.impact:6.1f}  {work.title[:46]:46s} "
              f"{(work.access_hint or work.journal)[:26]:26s}")

    print(f"\n=== {GROUP_LABELS['focused_monographs']} ===")
    for work in groups["focused_monographs"][:top]:
        print(f"{work.fragments:5d}  {work.title[:66]:66s} {work.year:>5s}")

    print(f"\n=== {GROUP_LABELS['probable_monographs']} ===")
    for work in groups["probable_monographs"][:12]:
        why = ", ".join(work.matched_markers[:2])
        print(f"{work.fragments:5d}  {work.title[:56]:56s} [{why[:28]}]")

    print(f"\n=== {GROUP_LABELS['splice_required']} (why they were demoted) ===")
    for work in groups["splice_required"][:12]:
        why = ", ".join(work.matched_markers[:2]) or work.topicality
        print(f"{work.fragments:5d}  {work.title[:56]:56s} [{why[:30]}]")


def main() -> None:
    """CLI entry point: rank sources by Genizah focus rather than citation count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--top", type=int, default=30, help="Entries to print per group")
    parser.add_argument("--min-fragments", type=int, default=1,
                        help="Drop works cited by fewer fragments than this")
    parser.add_argument("--literature-root", type=Path, default=DEFAULT_LITERATURE_ROOT,
                        help="academic_literature root, to skip works already scanned")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    dotenv.load_dotenv(_REPO_ROOT / ".env")
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
        notifications_min_severity="OFF",
    )
    try:
        with driver.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as session:
            rows = fetch_works(session)
    finally:
        driver.close()

    scanned = scanned_corpus_titles(args.literature_root)
    works = [w for w in build_works(rows, scanned) if w.fragments >= args.min_fragments]
    groups = group_works(works)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "genizah_focus.csv"
    json_path = args.out_dir / "genizah_focus_summary.json"
    write_csv(works, groups, csv_path)
    write_summary(groups, json_path)
    print_report(groups, args.top)
    print(f"\nWrote {csv_path}\nWrote {json_path}")


if __name__ == "__main__":
    main()
