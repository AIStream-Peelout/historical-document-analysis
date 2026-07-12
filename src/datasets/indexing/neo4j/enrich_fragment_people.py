#!/usr/bin/env python3
"""
Fragment–Person enrichment pass
================================

Reads every *_enhanced.json and creates
    Fragment -[:MENTIONS_PERSON]-> Person
edges by mining the page-level ``shelf_marks_mentioned`` data that the
main enhanced_kg_import pipeline never touches.

Two matching strategies (both write to the same relationship type):

  1. **Name-in-description** (certainty: definite)
     The shelf-mark description literally names the person:
       "Mentioned in connection with Rachel the Byzantine…"
     Any person name (or known variant, > 4 chars) found verbatim in the
     description text is a direct, high-confidence link.

  2. **Page co-occurrence** (certainty: possible)
     No name in the description, but exactly one person is listed as
     appearing on that page (via ``pages_mentioned``).  Used only when
     it's unambiguous (single person on that page) to avoid spurious
     many-to-many noise.

Shelf mark normalisation
------------------------
Academic citations vary: "Cambridge, T-S 12.732", "T-S 12.575",
"T-S N.S. 325.7".  We strip common prefixes ("Cambridge, ",
"Cambridge ", "ENA ") and try to match against Fragment.shelfmark in
the DB.  If no match is found we still MERGE a Fragment node so the
edge isn't lost — it will auto-merge with the Princeton node when that
fragment is later imported.

Usage
-----
python enrich_fragment_people.py               # run all
python enrich_fragment_people.py --dry-run     # count only
python enrich_fragment_people.py --dir malkhhut_ish
"""

import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dotenv
from neo4j import GraphDatabase
from tqdm import tqdm

# ---------------------------------------------------------------------------
project_root = Path(__file__).parent.parent.parent.parent.parent
data_root    = project_root / "src" / "datasets" / "raw_data" / "cairo_genizah" / "academic_literature"
sys.path.append(str(project_root))
dotenv.load_dotenv(project_root / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("enrich_fragment_people.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shelf-mark normalisation
# ---------------------------------------------------------------------------

_STRIP_PREFIXES = re.compile(
    r'^(Cambridge,?\s+|Cambridge University Library,?\s+|'
    r'Bodleian\s+|Oxford,?\s+|JTS,?\s+|Jewish Theological Seminary,?\s+)',
    re.IGNORECASE,
)

def _normalise_shelfmark(raw: str) -> str:
    """Strip institution prefixes; return cleaned shelf mark string."""
    cleaned = _STRIP_PREFIXES.sub("", raw.strip())
    return cleaned.strip().strip(".")


# ---------------------------------------------------------------------------
# Person name lookup builder
# ---------------------------------------------------------------------------

def _build_name_map(data: Dict) -> Dict[str, Dict]:
    """Merge people from both JSON people lists into a name→record map."""
    ca_people = data.get("context_analysis", {}).get("people_mentioned", []) or []
    pl_people  = data.get("people_locations",  {}).get("people", []) or []
    name_map: Dict[str, Dict] = {}
    for p in ca_people + pl_people:
        if not isinstance(p, dict):
            continue
        n = (p.get("name") or "").strip()
        if n and n not in name_map:
            name_map[n] = p
        for v in (p.get("name_variants") or []):
            v = (v or "").strip()
            if v and v not in name_map:
                name_map[v] = p
    return name_map


# ---------------------------------------------------------------------------
# Per-page edge extraction
# ---------------------------------------------------------------------------

def _extract_edges(data: Dict, source_book: str) -> List[Dict]:
    """Return list of {shelfmark, person_name, certainty, evidence, source_book}."""
    name_map = _build_name_map(data)
    all_people = list({
        id(p): p
        for p in (
            (data.get("context_analysis", {}).get("people_mentioned") or [])
            + (data.get("people_locations", {}).get("people") or [])
        )
        if isinstance(p, dict)
    }.values())

    edges: List[Dict] = []
    seen: set = set()  # deduplicate (shelfmark, person) pairs per file

    for page in (data.get("original_pages") or []):
        sm_raw = page.get("shelf_marks_mentioned", {})

        # Normalise shelf_marks_mentioned to {sm: description}
        if isinstance(sm_raw, list):
            sm_dict: Dict[str, str] = {s: "" for s in sm_raw if isinstance(s, str)}
        elif isinstance(sm_raw, dict):
            sm_dict = {k: (v or "") for k, v in sm_raw.items()}
        else:
            continue

        if not sm_dict:
            continue

        # Page number for co-occurrence fallback
        page_num = (
            (page.get("metadata") or {}).get("page_number")
            or ((page.get("extracted_page_number") or [None])[0])
        )

        # People on this specific page (for co-occurrence)
        people_on_page = [
            p for p in all_people
            if page_num is not None and page_num in (p.get("pages_mentioned") or [])
        ]

        for sm_raw_key, desc in sm_dict.items():
            sm = _normalise_shelfmark(sm_raw_key)
            if not sm:
                continue

            # --- Strategy 1: name found in description (definite) ---
            matched: List[str] = []
            for name in name_map:
                if len(name) > 4 and name in desc:
                    matched.append(name)

            for name in matched:
                key = (sm, name)
                if key not in seen:
                    seen.add(key)
                    edges.append({
                        "shelfmark":   sm,
                        "person_name": name,
                        "certainty":   "definite",
                        "evidence":    desc[:500],
                        "source_book": source_book,
                    })

            # --- Strategy 2: single person co-occurrence (possible) ---
            if not matched and len(people_on_page) == 1:
                name = (people_on_page[0].get("name") or "").strip()
                if name:
                    key = (sm, name)
                    if key not in seen:
                        seen.add(key)
                        edges.append({
                            "shelfmark":   sm,
                            "person_name": name,
                            "certainty":   "possible",
                            "evidence":    desc[:500] if desc else f"Co-occurs on page {page_num}",
                            "source_book": source_book,
                        })

    return edges


# ---------------------------------------------------------------------------
# Neo4j write
# ---------------------------------------------------------------------------

def _write_edge(tx, edge: Dict) -> None:
    """MERGE Fragment and Person nodes; MERGE MENTIONS_PERSON relationship."""
    tx.run("""
        MERGE (f:Fragment {shelfmark: $shelfmark})
        ON CREATE SET
            f.data_sources = ['extracted'],
            f.source_books  = [$book]
        ON MATCH SET
            f.data_sources = CASE
                WHEN 'extracted' IN coalesce(f.data_sources, []) THEN coalesce(f.data_sources, [])
                ELSE coalesce(f.data_sources, []) + ['extracted']
            END,
            f.source_books = CASE
                WHEN $book IN coalesce(f.source_books, []) THEN coalesce(f.source_books, [])
                ELSE coalesce(f.source_books, []) + [$book]
            END

        MERGE (p:Person {name: $person_name})
        ON CREATE SET
            p.data_sources = ['extracted'],
            p.source_books  = [$book]
        ON MATCH SET
            p.data_sources = CASE
                WHEN 'extracted' IN coalesce(p.data_sources, []) THEN coalesce(p.data_sources, [])
                ELSE coalesce(p.data_sources, []) + ['extracted']
            END,
            p.source_books = CASE
                WHEN $book IN coalesce(p.source_books, []) THEN coalesce(p.source_books, [])
                ELSE coalesce(p.source_books, []) + [$book]
            END

        MERGE (f)-[r:MENTIONS_PERSON]->(p)
        ON CREATE SET
            r.certainty    = $certainty,
            r.evidence     = $evidence,
            r.source       = $book,
            r.data_sources = ['extracted'],
            r.source_books  = [$book]
        ON MATCH SET
            r.source_books = CASE
                WHEN $book IN coalesce(r.source_books, []) THEN coalesce(r.source_books, [])
                ELSE coalesce(r.source_books, []) + [$book]
            END
    """,
        shelfmark=edge["shelfmark"],
        person_name=edge["person_name"],
        certainty=edge["certainty"],
        evidence=edge["evidence"],
        book=edge["source_book"],
    )


# ---------------------------------------------------------------------------
# Main enricher
# ---------------------------------------------------------------------------

class FragmentPeopleEnricher:

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver   = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        logger.info(f"Connected to Neo4j at {uri}, database={database}")

    def close(self):
        self.driver.close()

    def enrich_file(self, path: Path, dry_run: bool = False) -> Dict[str, int]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        source_book = path.stem.replace("_enhanced", "")
        edges = _extract_edges(data, source_book)

        definite = sum(1 for e in edges if e["certainty"] == "definite")
        possible = sum(1 for e in edges if e["certainty"] == "possible")

        if not edges:
            return {"edges_definite": 0, "edges_possible": 0}

        if dry_run:
            if edges:
                logger.debug(f"  Would write {definite} definite + {possible} possible edges")
                for e in edges[:3]:
                    logger.debug(f"    {e['shelfmark']} → {e['person_name']} [{e['certainty']}]")
            return {"edges_definite": definite, "edges_possible": possible}

        with self.driver.session(database=self.database) as session:
            for edge in edges:
                try:
                    session.execute_write(_write_edge, edge)
                except Exception as exc:
                    logger.warning(f"  Write failed {edge['shelfmark']} → {edge['person_name']}: {exc}")

        return {"edges_definite": definite, "edges_possible": possible}

    def enrich_all(self, root: Path, dry_run: bool = False):
        # Exclude enhanced JSONs inside *_structured*/ subdirs — those are
        # intermediate pipeline artifacts; only the copy in the book's own
        # directory is canonical.
        files = sorted(
            p for p in root.rglob("*_enhanced.json")
            if "_structured" not in p.parent.name
        )
        logger.info(f"Found {len(files)} enhanced JSON files")

        totals: Dict[str, int] = defaultdict(int)
        for path in tqdm(files, desc="Enriching fragment–person links"):
            rel = path.relative_to(root)
            try:
                counts = self.enrich_file(path, dry_run=dry_run)
                for k, v in counts.items():
                    totals[k] += v
                if any(v > 0 for v in counts.values()):
                    logger.info(f"  {rel}: {counts}")
            except Exception as exc:
                logger.error(f"  Failed {rel}: {exc}")

        tag = "[DRY RUN] " if dry_run else ""
        print(f"\n{tag}Enrichment summary")
        print("=" * 50)
        print(f"  Definite edges (name in description):  {totals['edges_definite']:,}")
        print(f"  Possible edges (page co-occurrence):   {totals['edges_possible']:,}")
        print(f"  Total MENTIONS_PERSON edges written:   {totals['edges_definite'] + totals['edges_possible']:,}")
        print("=" * 50)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", "-n", action="store_true")
    parser.add_argument("--dir", "-d", metavar="SUBDIR", default=None,
                        help="Limit to a subdirectory under academic_literature/")
    args = parser.parse_args()

    uri      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    user     = os.getenv("NEO4J_USER",     "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "genizah-prod")

    if not password:
        logger.error("NEO4J_PASSWORD not set")
        sys.exit(1)

    root = data_root
    if args.dir:
        root = data_root / args.dir
        if not root.exists():
            logger.error(f"Directory not found: {root}")
            sys.exit(1)

    enricher = FragmentPeopleEnricher(uri, user, password, database=database)
    try:
        enricher.enrich_all(root, dry_run=args.dry_run)
    finally:
        enricher.close()


if __name__ == "__main__":
    main()
