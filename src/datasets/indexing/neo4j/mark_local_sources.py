#!/usr/bin/env python3
"""
Mark locally-held academic sources in the Neo4j knowledge graph.

Scans the academic_literature directory, reads every *_metadata.json,
computes processing status, then either:
  - Updates an existing BookArticle node with local-copy metadata, or
  - Creates a new BookArticle node marked as locally held.

Processing levels (ordered):
  pdf_only   — PDF(s) present, nothing else
  ocr        — OCR results JSON present
  structured — per-page structured JSON files present
  enhanced   — enhanced analysis JSON present (highest)

After running this script you can use check_source() (or the Cypher snippets
at the bottom of this file) to query whether a title is already in the index.

Environment variables (.env):
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your_password
    NEO4J_DATABASE=genizah-prod
    ACADEMIC_LIT_DIR=/path/to/academic_literature   (optional override)
"""

import os
import sys
import json
import hashlib
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from neo4j import GraphDatabase
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mark_local_sources.log'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

DEFAULT_LIT_DIR = (
    Path(__file__).resolve().parents[3]
    / 'datasets' / 'raw_data' / 'cairo_genizah' / 'academic_literature'
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_article_id(title: str, author: str, year: str) -> str:
    key = f"{title}|{author}|{year}".lower().strip()
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _normalise_title(title: str) -> str:
    """Lowercase, strip punctuation — used for fuzzy title matching."""
    return re.sub(r'[^\w\s]', '', title.lower()).strip()


def _split_authors(raw: Any) -> List[str]:
    """Return a flat list of individual author names.

    Handles both a list of strings and a single semicolon-delimited string,
    e.g. "French, Mary ;Goldie, Rebecca ;Nichols, Emma".
    """
    if isinstance(raw, str):
        raw = [raw]
    authors: List[str] = []
    for item in raw:
        for part in str(item).split(';'):
            name = part.strip()
            if name:
                authors.append(name)
    return authors


def _processing_level(source_dir: Path) -> str:
    """Return the highest processing level achieved for a source directory."""
    if list(source_dir.rglob('*_enhanced.json')):
        return 'enhanced'
    if list(source_dir.rglob('page_*_structured.json')):
        return 'structured'
    if list(source_dir.rglob('*_ocr_results.json')):
        return 'ocr'
    if list(source_dir.rglob('*.pdf')):
        return 'pdf_only'
    return 'metadata_only'


def _source_stats(source_dir: Path) -> Dict[str, Any]:
    """Compute processing statistics for a source directory."""
    pdfs            = list(source_dir.rglob('*.pdf'))
    ocr_files       = list(source_dir.rglob('*_ocr_results.json'))
    enhanced_files  = list(source_dir.rglob('*_enhanced.json'))
    structured_pages = list(source_dir.rglob('page_*_structured.json'))

    # Total page count from OCR results (most reliable)
    total_pages = 0
    for ocr_file in ocr_files:
        try:
            with open(ocr_file) as f:
                data = json.load(f)
            total_pages += data.get('total_pages', 0)
        except Exception:
            pass

    return {
        'pdf_count':        len(pdfs),
        'has_ocr':          len(ocr_files) > 0,
        'has_enhanced':     len(enhanced_files) > 0,
        'structured_pages': len(structured_pages),
        'total_pages':      total_pages or len(structured_pages),
        'processing_level': _processing_level(source_dir),
        'local_path':       str(source_dir.relative_to(source_dir.parents[3])),
        'folder_name':      source_dir.name,
        'pdf_filenames':    ';'.join(p.name for p in sorted(pdfs)),
    }


def _load_metadata(metadata_file: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(metadata_file, encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not read {metadata_file}: {e}")
        return None


def _extract_fields(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten metadata into simple fields."""
    pub  = meta.get('publication', {})
    ids  = meta.get('identifiers', {})
    authors = _split_authors(meta.get('authors', []))

    return {
        'title':       meta.get('title', ''),
        'authors':     authors,
        'author_str':  '; '.join(authors),
        'first_author': authors[0] if authors else '',
        'year':        str(pub.get('year', '')),
        'publisher':   pub.get('publisher', ''),
        'journal':     pub.get('journal', pub.get('book_title', '')),
        'pub_type':    pub.get('type', ''),
        'language':    pub.get('language', ''),
        'pages':       pub.get('pages', ''),
        'doi':         ids.get('doi', ''),
        'isbn':        (ids.get('isbn', [''])[0] if isinstance(ids.get('isbn'), list)
                        else ids.get('isbn', '')),
        'url':         meta.get('url', ''),
        'summary':     meta.get('summary', ''),
        'subjects':    meta.get('subjects', []),
    }


# ---------------------------------------------------------------------------
# Importer
# ---------------------------------------------------------------------------

class LocalSourceMarker:
    """Scan academic_literature and stamp BookArticle nodes in Neo4j."""

    def __init__(self, uri: str, user: str, password: str, database: str = 'neo4j'):
        self.driver   = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        logger.info(f"Connected to Neo4j at {uri} / db={database}")

    def close(self):
        self.driver.close()

    def _session(self):
        return self.driver.session(database=self.database)

    # ------------------------------------------------------------------
    # Match an existing BookArticle by DOI → ISBN → normalised title
    # ------------------------------------------------------------------
    def _find_existing(self, fields: Dict[str, Any]) -> Optional[str]:
        """Return article_id of an existing BookArticle match, or None."""
        with self._session() as s:
            # 1. DOI match
            if fields['doi']:
                r = s.run(
                    "MATCH (b:BookArticle) WHERE b.doi = $doi RETURN b.article_id LIMIT 1",
                    doi=fields['doi']
                ).single()
                if r:
                    return r['b.article_id']

            # 2. ISBN match
            if fields['isbn']:
                r = s.run(
                    "MATCH (b:BookArticle) WHERE b.isbn = $isbn RETURN b.article_id LIMIT 1",
                    isbn=fields['isbn']
                ).single()
                if r:
                    return r['b.article_id']

            # 3. Normalised title match
            norm = _normalise_title(fields['title'])
            r = s.run(
                "MATCH (b:BookArticle) WHERE toLower(b.title) CONTAINS $fragment "
                "RETURN b.article_id, b.title LIMIT 5",
                fragment=norm[:60]   # use first 60 chars to keep query selective
            ).data()
            for row in r:
                if _normalise_title(row['b.title']) == norm:
                    return row['b.article_id']

        return None

    # ------------------------------------------------------------------
    # Stamp or create a node
    # ------------------------------------------------------------------
    def _upsert_node(
        self,
        article_id: Optional[str],
        fields: Dict[str, Any],
        stats: Dict[str, Any],
    ) -> Tuple[str, bool]:
        """Update existing node or create new one. Returns (article_id, created)."""
        if not article_id:
            article_id = _make_article_id(
                fields['title'], fields['first_author'], fields['year']
            )
            created = True
        else:
            created = False

        local_props = {
            'has_local_copy':    True,
            'local_path':        stats['local_path'],
            'folder_name':       stats['folder_name'],
            'pdf_filenames':     stats['pdf_filenames'],
            'pdf_count':         stats['pdf_count'],
            'has_ocr':           stats['has_ocr'],
            'has_enhanced':      stats['has_enhanced'],
            'structured_pages':  stats['structured_pages'],
            'total_pages':       stats['total_pages'],
            'processing_level':  stats['processing_level'],
        }

        with self._session() as s:
            s.run("""
                MERGE (b:BookArticle {article_id: $article_id})
                ON CREATE SET
                    b.title       = $title,
                    b.year        = $year,
                    b.publisher   = $publisher,
                    b.journal     = $journal,
                    b.language    = $language,
                    b.pub_type    = $pub_type,
                    b.url         = $url,
                    b.doi         = $doi,
                    b.isbn        = $isbn,
                    b.summary     = $summary
                SET
                    b.has_local_copy   = $has_local_copy,
                    b.local_path       = $local_path,
                    b.folder_name      = $folder_name,
                    b.pdf_filenames    = $pdf_filenames,
                    b.pdf_count        = $pdf_count,
                    b.has_ocr          = $has_ocr,
                    b.has_enhanced     = $has_enhanced,
                    b.structured_pages = $structured_pages,
                    b.total_pages      = $total_pages,
                    b.processing_level = $processing_level
            """, article_id=article_id, **fields, **local_props)

            # Ensure Scholar → WROTE → BookArticle edges
            for author in fields['authors']:
                author = author.strip()
                if author:
                    s.run("""
                        MERGE (sc:Scholar {name: $author})
                        MERGE (b:BookArticle {article_id: $article_id})
                        MERGE (sc)-[:WROTE]->(b)
                    """, author=author, article_id=article_id)

        return article_id, created

    # ------------------------------------------------------------------
    # Main scan
    # ------------------------------------------------------------------
    def mark_all(self, lit_dir: Path):
        metadata_files = sorted(lit_dir.rglob('*_metadata.json'))
        metadata_files = [f for f in metadata_files if 'example_book' not in f.name]

        logger.info(f"Found {len(metadata_files)} source metadata files in {lit_dir}")

        results = {'matched': 0, 'created': 0, 'skipped': 0}

        for meta_file in metadata_files:
            meta = _load_metadata(meta_file)
            if not meta or not meta.get('title'):
                logger.warning(f"  Skipping (no title): {meta_file}")
                results['skipped'] += 1
                continue

            fields = _extract_fields(meta)
            stats  = _source_stats(meta_file.parent)

            existing_id = self._find_existing(fields)
            article_id, created = self._upsert_node(existing_id, fields, stats)

            status = 'CREATED' if created else 'UPDATED'
            results['created' if created else 'matched'] += 1
            logger.info(
                f"  [{status}] {fields['title'][:70]} "
                f"| level={stats['processing_level']} "
                f"| pages={stats['structured_pages']}"
            )

        logger.info(
            f"\n✓ Done — matched={results['matched']}  "
            f"created={results['created']}  skipped={results['skipped']}"
        )
        self._print_summary()

    def _print_summary(self):
        with self._session() as s:
            rows = s.run("""
                MATCH (b:BookArticle)
                WHERE b.has_local_copy = true
                RETURN b.processing_level AS level, count(*) AS cnt
                ORDER BY cnt DESC
            """).data()
            print("\nLocally-held sources by processing level:")
            for r in rows:
                print(f"  {r['level']:<15} {r['cnt']}")

            total = s.run(
                "MATCH (b:BookArticle) WHERE b.has_local_copy = true RETURN count(b) AS n"
            ).single()['n']
            print(f"\n  Total with local copy: {total}")


# ---------------------------------------------------------------------------
# Query helpers  (also useful to paste directly into Neo4j Browser)
# ---------------------------------------------------------------------------

CHECK_SOURCE_CYPHER = """
// Check if a source is already in the index (paste into Neo4j Browser)
// Replace $search_term with the title fragment or author name you want.

MATCH (b:BookArticle)
WHERE toLower(b.title)     CONTAINS toLower($search_term)
   OR toLower(b.journal)   CONTAINS toLower($search_term)
   OR b.doi                = $search_term
   OR b.isbn               = $search_term
OPTIONAL MATCH (s:Scholar)-[:WROTE]->(b)
RETURN
    b.title            AS title,
    collect(s.name)    AS authors,
    b.year             AS year,
    b.has_local_copy   AS have_it,
    b.processing_level AS processing_level,
    b.structured_pages AS pages_processed,
    b.local_path       AS local_path,
    b.doi              AS doi,
    b.url              AS url
ORDER BY b.has_local_copy DESC, b.year DESC
"""

LOCAL_COVERAGE_CYPHER = """
// Overview of all locally-held sources
MATCH (b:BookArticle)
WHERE b.has_local_copy = true
OPTIONAL MATCH (s:Scholar)-[:WROTE]->(b)
OPTIONAL MATCH (b)-[:REFERENCES]->(f:Fragment)
RETURN
    b.title            AS title,
    collect(DISTINCT s.name) AS authors,
    b.year             AS year,
    b.processing_level AS level,
    b.structured_pages AS pages_processed,
    b.total_pages      AS total_pages,
    count(DISTINCT f)  AS fragments_referenced
ORDER BY fragments_referenced DESC, b.year
"""

def check_source(driver, database: str, search_term: str):
    """Programmatic version of the CHECK_SOURCE_CYPHER query."""
    with driver.session(database=database) as s:
        rows = s.run("""
            MATCH (b:BookArticle)
            WHERE toLower(b.title)   CONTAINS toLower($q)
               OR toLower(b.journal) CONTAINS toLower($q)
               OR b.doi = $q OR b.isbn = $q
            OPTIONAL MATCH (sc:Scholar)-[:WROTE]->(b)
            OPTIONAL MATCH (b)-[:REFERENCES]->(f:Fragment)
            RETURN b.title            AS title,
                   collect(DISTINCT sc.name) AS authors,
                   b.year             AS year,
                   b.has_local_copy   AS have_it,
                   b.processing_level AS processing_level,
                   b.structured_pages AS pages_processed,
                   b.total_pages      AS total_pages,
                   count(DISTINCT f)  AS fragments_referenced,
                   b.local_path       AS local_path,
                   b.doi              AS doi
            ORDER BY b.has_local_copy DESC, b.year DESC
        """, q=search_term).data()
    if not rows:
        print(f"No results for '{search_term}'")
        return
    for r in rows:
        have = "✓ HAVE IT" if r['have_it'] else "✗ not held"
        print(f"\n{have}  [{r['processing_level'] or 'in graph only'}]")
        print(f"  Title:     {r['title']}")
        print(f"  Authors:   {', '.join(r['authors'])}")
        print(f"  Year:      {r['year']}")
        print(f"  Pages:     {r['pages_processed']}/{r['total_pages']} processed")
        print(f"  Fragments: {r['fragments_referenced']} referenced in graph")
        if r['local_path']:
            print(f"  Path:      {r['local_path']}")
        if r['doi']:
            print(f"  DOI:       {r['doi']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    load_dotenv()
    uri      = os.getenv('NEO4J_URI',      'bolt://localhost:7687')
    user     = os.getenv('NEO4J_USER',     'neo4j')
    password = os.getenv('NEO4J_PASSWORD')
    database = os.getenv('NEO4J_DATABASE', 'genizah-prod')
    lit_dir  = Path(os.getenv('ACADEMIC_LIT_DIR', str(DEFAULT_LIT_DIR)))

    if not password:
        logger.error("NEO4J_PASSWORD not set")
        sys.exit(1)
    if not lit_dir.exists():
        logger.error(f"Literature directory not found: {lit_dir}")
        sys.exit(1)

    marker = LocalSourceMarker(uri, user, password, database)
    try:
        marker.mark_all(lit_dir)

        # Demo: interactive check if a second argument is provided
        if len(sys.argv) > 1:
            print(f"\n--- Checking: '{sys.argv[1]}' ---")
            check_source(marker.driver, database, sys.argv[1])
    finally:
        marker.close()


if __name__ == '__main__':
    main()
