#!/usr/bin/env python3
"""
Cairo Genizah Bibliography Import - Neo4j

Reads biblio.json (keyed by canonical shelfmark) and creates:
  - Fragment nodes (merged by canonical_shelfmark)
  - BookArticle nodes (merged by deterministic article_id)
  - Scholar nodes (merged by author name)
  - Scholar  -[:WROTE]->       BookArticle
  - BookArticle -[:REFERENCES {pages, mention_type, has_transcription,
                               has_translation, has_discussion}]-> Fragment

Shelfmarks in biblio.json are already in canonical underscore form
(e.g. "L-G_Ar_I_130").  ShelfmarkNormalizer.get_institution_info() is used
to derive institution metadata so that HELD_AT edges are created in the
same normalised way as the Princeton CSV import.

Requirements:
    pip install neo4j python-dotenv tqdm

Environment variables (.env):
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your_password
    BIBLIO_JSON=/path/to/biblio.json
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from neo4j import GraphDatabase
from dotenv import load_dotenv
from tqdm import tqdm

from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('biblio_import.log'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_article_id(title: str, author: str, year: str) -> str:
    """Deterministic 16-char hex ID for a BookArticle node."""
    key = f"{title}|{author}|{year}".lower().strip()
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _canonical_to_display(canonical: str) -> str:
    """Best-effort conversion of underscore canonical ID back to a display
    shelfmark (e.g. 'T_S_AS_18_170' → 'T-S AS 18.170').

    Only used as a human-readable label on Fragment nodes created from
    biblio.json keys that have no separate shelfmark source."""
    # Replace underscores between letters and digits or vice versa with space,
    # and between digits with a dot.
    parts = canonical.split('_')
    if not parts:
        return canonical
    result_parts = [parts[0]]
    for prev, curr in zip(parts, parts[1:]):
        if prev[-1].isdigit() and curr[0].isdigit():
            result_parts.append('.' + curr)
        elif prev[-1].isalpha() and curr[0].isdigit():
            result_parts.append(' ' + curr)
        elif prev[-1].isdigit() and curr[0].isalpha():
            result_parts.append(' ' + curr)
        else:
            result_parts.append(' ' + curr)
    return ''.join(result_parts)


def _parse_mention_type(mention_type: Dict[str, Any]) -> Dict[str, bool]:
    """Normalise the mention_type dict: values are "True"/"False"/str."""
    return {
        'has_discussion':   str(mention_type.get('discussion',   'False')).lower() == 'true',
        'has_transcription': mention_type.get('transcription') is not None
                             and mention_type.get('transcription') != 'False',
        'has_translation':  mention_type.get('translation') is not None
                            and mention_type.get('translation') != 'False',
        'transcription_extent': mention_type.get('transcription'),  # "Full" / "Partial" / None
        'translation_extent':   mention_type.get('translation'),
    }


# ---------------------------------------------------------------------------
# Importer
# ---------------------------------------------------------------------------

class GenizahBiblioImporter:
    """Import biblio.json bibliography data into the Neo4j knowledge graph."""

    BATCH_SIZE = 200

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        logger.info(f"Connected to Neo4j at {uri}")
        logger.info(f"Using database: {self.database}")

    def close(self):
        self.driver.close()
        logger.info("Closed Neo4j connection")

    # ------------------------------------------------------------------
    # Database creation
    # ------------------------------------------------------------------
    def ensure_database_exists(self):
        """Create the target database if it doesn't already exist.

        Runs against the Neo4j system database so it's safe to call on a
        fresh instance or an existing one — CREATE DATABASE is idempotent
        when IF NOT EXISTS is used.
        """
        if self.database == "neo4j":
            return  # default db always exists
        logger.info(f"Ensuring database '{self.database}' exists...")
        with self.driver.session(database="system") as session:
            session.run(f"CREATE DATABASE `{self.database}` IF NOT EXISTS")
        logger.info(f"✓ Database '{self.database}' ready")

    # ------------------------------------------------------------------
    # Constraints (idempotent – safe to run after Princeton CSV import)
    # ------------------------------------------------------------------
    def ensure_constraints(self):
        logger.info("Ensuring constraints...")
        stmts = [
            "CREATE CONSTRAINT fragment_canonical   IF NOT EXISTS FOR (f:Fragment)    REQUIRE f.canonical_shelfmark IS UNIQUE",
            "CREATE CONSTRAINT scholar_name         IF NOT EXISTS FOR (s:Scholar)     REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT book_article_id      IF NOT EXISTS FOR (b:BookArticle) REQUIRE b.article_id IS UNIQUE",
            "CREATE CONSTRAINT institution_name     IF NOT EXISTS FOR (i:Institution) REQUIRE i.name IS UNIQUE",
            "CREATE INDEX fragment_shelfmark        IF NOT EXISTS FOR (f:Fragment)    ON (f.shelfmark)",
            "CREATE INDEX book_year                 IF NOT EXISTS FOR (b:BookArticle) ON (b.year)",
        ]
        with self.driver.session(database=self.database) as session:
            for stmt in stmts:
                try:
                    session.run(stmt)
                except Exception as e:
                    logger.debug(f"Constraint note: {e}")
        logger.info("✓ Constraints ready")

    # ------------------------------------------------------------------
    # Core import
    # ------------------------------------------------------------------
    def import_biblio(self, biblio_path: str):
        path = Path(biblio_path)
        if not path.exists():
            raise FileNotFoundError(f"biblio.json not found: {path}")

        logger.info(f"Loading {path} ...")
        with open(path, encoding='utf-8') as f:
            data: Dict[str, Any] = json.load(f)

        total_shelfmarks = len(data)
        total_citations  = sum(v.get('total_citations', 0) for v in data.values())
        logger.info(f"  {total_shelfmarks:,} shelfmarks  /  {total_citations:,} citations")

        # Build flat list of (canonical_shelfmark, citation) pairs for batching
        rows: List[Dict[str, Any]] = []
        for canonical_shelfmark, entry in data.items():
            # The keys in biblio.json use underscores but may not follow the
            # exact canonical form produced by to_canonical_id.  Re-normalise
            # by converting underscores back to spaces for the display form,
            # then re-canonicalising.
            display = _canonical_to_display(canonical_shelfmark)
            canonical = ShelfmarkNormalizer.to_canonical_id(display) or canonical_shelfmark
            inst_info = ShelfmarkNormalizer.get_institution_info(display)

            for citation in entry.get('citations', []):
                rows.append({
                    'canonical_shelfmark': canonical,
                    'display_shelfmark':   display,
                    'institution':         inst_info.get('institution'),
                    'collection':          inst_info.get('collection'),
                    'subcollection':       inst_info.get('subcollection'),
                    'citation':            citation,
                })

        logger.info(f"Processing {len(rows):,} citation rows in batches of {self.BATCH_SIZE}...")

        with self.driver.session(database=self.database) as session:
            for i in tqdm(range(0, len(rows), self.BATCH_SIZE), desc="Importing bibliography"):
                batch = rows[i:i + self.BATCH_SIZE]
                session.execute_write(self._write_batch, batch)

        logger.info("✓ Bibliography import complete")
        self._print_stats()

    # ------------------------------------------------------------------
    # Batch writer (runs inside a single transaction)
    # ------------------------------------------------------------------
    @staticmethod
    def _write_batch(tx, batch: List[Dict[str, Any]]):
        for row in batch:
            citation    = row['citation']
            title       = citation.get('title', '').strip()
            author      = citation.get('author', '').strip()
            year        = str(citation.get('year', '')).strip()
            language    = citation.get('language', '').strip()
            pages       = citation.get('citedonpages', '').strip()
            mention     = _parse_mention_type(citation.get('mention_type', {}))
            article_id  = _make_article_id(title, author, year)

            # 1. Ensure Fragment node exists
            tx.run("""
                MERGE (f:Fragment {canonical_shelfmark: $canonical_shelfmark})
                ON CREATE SET f.shelfmark = $display_shelfmark
            """, {
                'canonical_shelfmark': row['canonical_shelfmark'],
                'display_shelfmark':   row['display_shelfmark'],
            })

            # 2. Link Fragment to Institution
            if row.get('institution'):
                tx.run("""
                    MERGE (f:Fragment {canonical_shelfmark: $canonical_shelfmark})
                    MERGE (i:Institution {name: $institution})
                    ON CREATE SET i.collection = $collection
                    MERGE (f)-[r:HELD_AT]->(i)
                    SET r.sub_collection = $subcollection
                """, {
                    'canonical_shelfmark': row['canonical_shelfmark'],
                    'institution':         row['institution'],
                    'collection':          row['collection'],
                    'subcollection':       row['subcollection'],
                })

            # 3. Ensure BookArticle node exists
            if title:
                tx.run("""
                    MERGE (b:BookArticle {article_id: $article_id})
                    ON CREATE SET b.title    = $title,
                                  b.year     = $year,
                                  b.language = $language
                    SET b.title    = $title,
                        b.year     = $year,
                        b.language = $language
                """, {
                    'article_id': article_id,
                    'title':      title,
                    'year':       year or None,
                    'language':   language or None,
                })

            # 4. Ensure Scholar node and WROTE relationship
            if author and title:
                tx.run("""
                    MERGE (s:Scholar {name: $author})
                    MERGE (b:BookArticle {article_id: $article_id})
                    MERGE (s)-[:WROTE]->(b)
                """, {
                    'author':     author,
                    'article_id': article_id,
                })

            # 5. BookArticle -[:REFERENCES]-> Fragment
            if title:
                tx.run("""
                    MATCH (b:BookArticle {article_id: $article_id})
                    MATCH (f:Fragment    {canonical_shelfmark: $canonical_shelfmark})
                    MERGE (b)-[r:REFERENCES]->(f)
                    SET r.pages                = $pages,
                        r.has_discussion       = $has_discussion,
                        r.has_transcription    = $has_transcription,
                        r.has_translation      = $has_translation,
                        r.transcription_extent = $transcription_extent,
                        r.translation_extent   = $translation_extent
                """, {
                    'article_id':            article_id,
                    'canonical_shelfmark':   row['canonical_shelfmark'],
                    'pages':                 pages or None,
                    **mention,
                })

    # ------------------------------------------------------------------
    def _print_stats(self):
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] AS label, count(*) AS count
                ORDER BY count DESC
            """)
            print("\nNode counts after biblio import:")
            for r in result:
                print(f"  {r['label']}: {r['count']:,}")

            result = session.run("""
                MATCH ()-[r:REFERENCES]->()
                RETURN count(r) AS refs
            """)
            rec = result.single()
            if rec:
                print(f"\n  REFERENCES relationships: {rec['refs']:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    load_dotenv()
    neo4j_uri      = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user     = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    neo4j_database = os.getenv('NEO4J_DATABASE', 'genizah-prod')
    biblio_path    = os.getenv(
        'BIBLIO_JSON',
        '/Users/isaac/Documents/GitHub/historical-document-analysis'
        '/src/datasets/raw_data/cairo_genizah/retained_json/biblio.json'
    )

    if not neo4j_password:
        logger.error("NEO4J_PASSWORD environment variable not set")
        sys.exit(1)

    importer = GenizahBiblioImporter(neo4j_uri, neo4j_user, neo4j_password, database=neo4j_database)
    try:
        importer.ensure_database_exists()
        importer.ensure_constraints()
        importer.import_biblio(biblio_path)
    finally:
        importer.close()


if __name__ == "__main__":
    main()
