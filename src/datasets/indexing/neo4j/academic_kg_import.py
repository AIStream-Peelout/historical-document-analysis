#!/usr/bin/env python3
"""
Academic KG Import — Cairo Genizah

Single import step for all academic-literature derived data.  Replaces
enhanced_kg_import.py and the old --import-dir mode of enrich_node_relations.py.

Two data sources are merged in one run:

  1. *_enhanced.json  (Pass 2 — secondary_llm_processing.py output)
       → Person / Scholar / Place nodes
       → kg_triplets (general LLM relationships)

  2. enriched_relations/**/*.json  (Pass 3 — enrich_node_relations.py output)
       → entity-anchored triplets with richer, focused relationships

Node identity matches the Princeton CSV import (knowlege_graph_poc.py) and
bibliography import (biblio_import.py) exactly — no duplicates or floating nodes.

Merge rules
-----------
* Fragment   → MERGE on canonical_shelfmark (via ShelfmarkNormalizer).
               Tries to match existing PGP nodes first; only creates new ones
               when no match exists.
* Person     → MERGE on name  (unique constraint)
* Scholar    → MERGE on name  (unique constraint)
* Place      → MERGE on name  (unique constraint); resolves Princeton name
               variants for known aliases.
* Institution→ MERGE on name  (unique constraint)
* BookArticle→ MATCH on title first (case-insensitive) to reuse existing
               biblio nodes; falls back to MERGE with sha1(title) article_id.

Pipeline position
-----------------
knowlege_graph_poc.py  →  biblio_import.py  →  academic_kg_import.py  →  geocode_places.py

Usage
-----
# Import everything (enhanced JSONs + enriched_relations/)
python academic_kg_import.py

# Dry-run: count without writing
python academic_kg_import.py --dry-run

# Enhanced JSONs only (skip enriched triplets)
python academic_kg_import.py --enhanced-only

# Enriched triplets only (skip enhanced JSONs)
python academic_kg_import.py --enriched-only

# Limit enhanced JSONs to a subdirectory
python academic_kg_import.py --dir cambridge_articles

Environment variables (.env)
-----------------------------
NEO4J_URI      bolt://localhost:7687
NEO4J_USER     neo4j
NEO4J_PASSWORD your_password
NEO4J_DATABASE genizah-prod   (default)
"""

import hashlib
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import dotenv
from neo4j import GraphDatabase
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))
dotenv.load_dotenv(project_root / ".env")

from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_data_root      = project_root / "src" / "datasets" / "raw_data" / "cairo_genizah" / "academic_literature"
_enriched_root  = project_root / "src" / "datasets" / "raw_data" / "cairo_genizah" / "enriched_relations"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("academic_kg_import.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCHOLAR_ROLES = {"author", "scholar", "editor", "translator", "co-author"}

_INSTITUTION_KEYWORDS = {
    "university", "college", "library", "seminary", "institute", "museum",
    "academy", "archive", "school", "foundation", "society", "centre", "center",
}

_VALID_LABELS = {"Person", "Scholar", "Place", "Institution", "Fragment", "BookArticle", "Entity"}

_NON_GEOGRAPHIC = {
    "cairo genizah", "genizah", "jewish theological seminary",
    "cambridge university library", "bodleian library", "british library",
    "taylor-schechter", "t-s collection", "dropsie college",
}

_SHELFMARK_RE = re.compile(r'^[A-Z][\w./-]+ \d+[\w./-]*$')

_LABEL_MAP = {
    "Person": "Person", "Scholar": "Scholar", "Place": "Place",
    "Fragment": "Fragment", "BookArticle": "BookArticle",
    "Institution": "Institution", "Entity": "Entity",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_article_id(title: str, author: str = "", year: str = "") -> str:
    key = f"{title}|{author}|{year}".lower().strip()
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _is_scholar_role(role: str) -> bool:
    return role.strip().lower() in SCHOLAR_ROLES


def _is_institution_name(name: str) -> bool:
    lower = name.lower()
    return any(kw in lower for kw in _INSTITUTION_KEYWORDS)


def _src_books(alias: str) -> str:
    return (
        f"{alias}.source_books = CASE "
        f"WHEN $book IN coalesce({alias}.source_books, []) "
        f"THEN coalesce({alias}.source_books, []) "
        f"ELSE coalesce({alias}.source_books, []) + [$book] END"
    )


def _add_source(alias: str, source_tag: str) -> str:
    return (
        f"{alias}.data_sources = CASE "
        f"WHEN '{source_tag}' IN coalesce({alias}.data_sources, []) "
        f"THEN coalesce({alias}.data_sources, []) "
        f"ELSE coalesce({alias}.data_sources, []) + ['{source_tag}'] END"
    )


# ---------------------------------------------------------------------------
# Importer
# ---------------------------------------------------------------------------

class AcademicKGImporter:

    BATCH_SIZE = 100

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver   = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        logger.info(f"Connected to Neo4j at {uri}, database={database}")

    def close(self):
        self.driver.close()

    def ensure_constraints(self):
        stmts = [
            "CREATE CONSTRAINT fragment_canonical   IF NOT EXISTS FOR (f:Fragment)    REQUIRE f.canonical_shelfmark IS UNIQUE",
            "CREATE CONSTRAINT scholar_name         IF NOT EXISTS FOR (s:Scholar)     REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT person_name          IF NOT EXISTS FOR (p:Person)      REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT place_name           IF NOT EXISTS FOR (pl:Place)      REQUIRE pl.name IS UNIQUE",
            "CREATE CONSTRAINT book_article_id      IF NOT EXISTS FOR (b:BookArticle) REQUIRE b.article_id IS UNIQUE",
            "CREATE CONSTRAINT institution_name     IF NOT EXISTS FOR (i:Institution) REQUIRE i.name IS UNIQUE",
        ]
        with self.driver.session(database=self.database) as session:
            for stmt in stmts:
                try:
                    session.run(stmt)
                except Exception as e:
                    logger.debug(f"Constraint note: {e}")
        logger.info("✓ Constraints ready")

    # ===================================================================
    # Source 1: *_enhanced.json files
    # ===================================================================

    @staticmethod
    def find_enhanced_files(root: Path) -> List[Path]:
        return sorted(root.rglob("*_enhanced.json"))

    def import_enhanced_file(self, path: Path, dry_run: bool = False) -> Dict[str, int]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        people_locs      = data.get("people_locations", {})
        context_analysis = data.get("context_analysis", {})
        source_book      = path.stem.replace("_enhanced", "")

        # Merge both people lists (scholars + historical figures)
        people_by_name: Dict[str, Dict] = {}
        for person in (people_locs.get("people", []) + context_analysis.get("people_mentioned", [])):
            name = (person.get("name") or "").strip()
            if name and name not in people_by_name:
                people_by_name[name] = person
                for v in person.get("name_variants", []):
                    if v and v.strip() not in people_by_name:
                        people_by_name[v.strip()] = person

        # Merge both location lists, skip institutions
        places_by_name: Dict[str, Dict] = {}
        for loc in (people_locs.get("locations", []) + context_analysis.get("locations_mentioned", [])):
            if (loc.get("type") or "").lower() in ("institution", "library", "university", "synagogue"):
                continue
            name = (loc.get("name") or "").strip()
            if name and name not in places_by_name:
                places_by_name[name] = loc
                for v in loc.get("name_variants", []):
                    if v and v.strip() not in places_by_name:
                        places_by_name[v.strip()] = loc

        counts: Dict[str, int] = defaultdict(int)

        if not dry_run and (people_by_name or places_by_name):
            with self.driver.session(database=self.database) as session:
                session.execute_write(
                    self._write_people_and_places,
                    people_by_name, places_by_name, source_book
                )
            counts["people_upserted"] += len(set(p.get("name","") for p in people_by_name.values()))
            counts["places_upserted"] += len(set(p.get("name","") for p in places_by_name.values()))

        # kg_triplets from Pass 2 are no longer used — Pass 3 (enrich_node_relations.py)
        # handles all relationship extraction. We still enrich place metadata if present.
        if not dry_run:
            with self.driver.session(database=self.database) as session:
                for loc_meta in places_by_name.values():
                    if any(loc_meta.get(k) for k in ("city", "country", "region")):
                        try:
                            session.execute_write(self._enrich_place_node, loc_meta)
                        except Exception as e:
                            logger.debug(f"  Place enrich failed for {loc_meta.get('name')}: {e}")

        return dict(counts)

    def _normalise_enhanced_triplet(
        self,
        triplet: Dict,
        people_by_name: Dict,
        places_by_name: Dict,
        source_book: str,
    ) -> Optional[Dict]:
        subj      = (triplet.get("subject")      or "").strip()
        subj_type = (triplet.get("subject_type") or "").strip()
        relation  = (triplet.get("relation")     or "").strip().upper()
        obj       = (triplet.get("object")       or "").strip()
        obj_type  = (triplet.get("object_type")  or "").strip()
        evidence  = (triplet.get("evidence")     or "").strip()
        confidence = (triplet.get("confidence")  or "medium").strip()

        if not (subj and relation and obj):
            return None

        # Drop non-geographic place names
        for name, ntype in ((obj, obj_type), (subj, subj_type)):
            if ntype.lower() == "place" and name.lower() in _NON_GEOGRAPHIC:
                return None
            if ntype.lower() == "place" and "-" in name:
                parts = [p.strip() for p in name.split("-")]
                if len(parts) == 2 and all(len(p) > 3 for p in parts):
                    return None

        # Drop trivial BookArticle references
        for title, ttype in ((obj, obj_type), (subj, subj_type)):
            if ttype.lower() == "bookarticle":
                if not title or len(title.split()) < 2 or _SHELFMARK_RE.match(title):
                    return None

        def _resolve_label(name: str, declared: str) -> str:
            if declared.lower() == "person":
                role = (people_by_name.get(name, {}).get("role") or "").lower()
                if _is_scholar_role(role):
                    return "Scholar"
            label = _LABEL_MAP.get(declared, declared)
            if label == "Place" and _is_institution_name(name):
                label = "Institution"
            return label

        subj_label = _resolve_label(subj, subj_type)
        obj_label  = _resolve_label(obj, obj_type)

        def _place_meta(name: str) -> Dict:
            m = places_by_name.get(name, {})
            return {"city": m.get("city","") or "", "country": m.get("country","") or "",
                    "region": m.get("region","") or ""}

        return {
            "subject":       subj,
            "subject_label": subj_label,
            "subject_place": _place_meta(subj) if subj_label == "Place" else {},
            "relation":      relation,
            "object":        obj,
            "object_label":  obj_label,
            "object_place":  _place_meta(obj) if obj_label == "Place" else {},
            "evidence":      evidence[:500],
            "confidence":    confidence,
            "source_book":   source_book,
            "data_tag":      "extracted",
        }

    # ===================================================================
    # Source 2: enriched_relations/**/*.json files
    # ===================================================================

    @staticmethod
    def find_enriched_files(root: Path) -> List[Path]:
        return sorted(root.rglob("*.json"))

    def import_enriched_file(self, path: Path, dry_run: bool = False) -> Dict[str, int]:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)

        if rec.get("status") != "ok":
            return {"skipped": 1}

        triplets     = rec.get("triplets") or []
        entity_name  = rec.get("entity_name", "")
        entity_label = rec.get("entity_label", "Person")
        books        = rec.get("contributing_books") or rec.get("source_books") or []

        if not triplets:
            return {"no_triplets": 1}

        counts: Dict[str, int] = defaultdict(int)
        counts["triplets_total"] += len(triplets)

        if dry_run:
            counts["triplets_valid"] += len(triplets)
            return dict(counts)

        written = 0
        with self.driver.session(database=self.database) as s:
            for t in triplets:
                for book in books:
                    try:
                        s.execute_write(self._write_enriched_triplet, t, book)
                        written += 1
                    except Exception as e:
                        logger.warning(f"  Enriched write failed {t}: {e}")

            # Mark entity as enriched so re-extraction can be skipped
            try:
                s.execute_write(self._mark_enriched, entity_name, entity_label, books)
            except Exception as e:
                logger.debug(f"  Could not mark enriched for '{entity_name}': {e}")

        counts["triplets_written"] += written
        logger.info(f"  {path.name}: {written} edge(s)")
        return dict(counts)

    # ===================================================================
    # Write helpers — enhanced triplets
    # ===================================================================

    def _write_enhanced_triplet(self, tx, row: Dict) -> None:
        """Route to the right write method based on labels."""
        sl  = row["subject_label"]
        ol  = row["object_label"]
        rel = row["relation"]

        if sl in ("Person", "Scholar") and rel == "WROTE" and ol == "BookArticle":
            self._write_person_wrote_article(tx, row)
        elif sl == "BookArticle" and rel == "CITES" and ol == "BookArticle":
            self._write_article_cites_article(tx, row)
        elif sl in ("Person", "Scholar") and rel == "LIVED_IN" and ol == "Place":
            self._write_person_lived_in(tx, row)
        elif sl in ("Person", "Scholar") and rel == "AFFILIATED_WITH" and ol in ("Institution", "Place"):
            self._write_person_affiliated_with(tx, row)
        elif sl in ("Person", "Scholar") and rel == "TRAVELED_TO" and ol in ("Place", "Institution"):
            self._write_person_traveled_to(tx, row)
        elif sl == "Fragment" or ol == "Fragment":
            # Covers Fragment→Person, Fragment→Place, Fragment→BookArticle,
            # Person→Fragment (WROTE/SENT_TO/MENTIONED_IN), etc.
            self._write_fragment_rel(tx, row)
        else:
            self._write_generic_enhanced(tx, row)

    @staticmethod
    def _write_person_wrote_article(tx, row: Dict) -> None:
        label      = row["subject_label"]
        author     = row["subject"]
        title      = row["object"]
        article_id = _find_or_make_article_id(tx, title)
        book       = row["source_book"]
        tx.run(f"""
            MERGE (a:{label} {{name: $name}})
            SET {_add_source('a', 'extracted')}, {_src_books('a')}
            MERGE (b:BookArticle {{article_id: $article_id}})
              ON CREATE SET b.title = $title
            SET {_add_source('b', 'extracted')}, {_src_books('b')}
            MERGE (a)-[r:WROTE]->(b)
              ON CREATE SET r.source=$book, r.confidence=$conf, r.evidence=$ev
            SET {_add_source('r', 'extracted')}, {_src_books('r')}
        """, name=author, article_id=article_id, title=title,
             book=book, conf=row["confidence"], ev=row["evidence"])

    @staticmethod
    def _write_article_cites_article(tx, row: Dict) -> None:
        src_id = _find_or_make_article_id(tx, row["subject"])
        tgt_id = _find_or_make_article_id(tx, row["object"])
        book   = row["source_book"]
        tx.run("""
            MERGE (a:BookArticle {article_id: $src_id})
              ON CREATE SET a.title = $src_title
            SET a.data_sources = CASE WHEN 'extracted' IN coalesce(a.data_sources,[])
                THEN coalesce(a.data_sources,[]) ELSE coalesce(a.data_sources,[]) + ['extracted'] END
            MERGE (b:BookArticle {article_id: $tgt_id})
              ON CREATE SET b.title = $tgt_title
            SET b.data_sources = CASE WHEN 'extracted' IN coalesce(b.data_sources,[])
                THEN coalesce(b.data_sources,[]) ELSE coalesce(b.data_sources,[]) + ['extracted'] END
            MERGE (a)-[r:CITES]->(b)
              ON CREATE SET r.source=$book, r.confidence=$conf, r.evidence=$ev
            SET r.data_sources = CASE WHEN 'extracted' IN coalesce(r.data_sources,[])
                THEN coalesce(r.data_sources,[]) ELSE coalesce(r.data_sources,[]) + ['extracted'] END
        """, src_id=src_id, src_title=row["subject"],
             tgt_id=tgt_id, tgt_title=row["object"],
             book=book, conf=row["confidence"], ev=row["evidence"])

    @staticmethod
    def _write_person_lived_in(tx, row: Dict) -> None:
        if _is_institution_name(row["object"]):
            AcademicKGImporter._write_person_affiliated_with(tx, row)
            return
        label = row["subject_label"]
        place = _resolve_place_name(tx, row["object"])
        pm    = row.get("object_place", {})
        book  = row["source_book"]
        tx.run(f"""
            MERGE (p:{label} {{name: $name}})
            SET {_add_source('p', 'extracted')}, {_src_books('p')}
            MERGE (pl:Place {{name: $place}})
            SET pl.place_type='historical',
                {_add_source('pl', 'extracted')}, {_src_books('pl')},
                pl.city    = CASE WHEN $city    <> '' AND pl.city    IS NULL THEN $city    ELSE pl.city    END,
                pl.country = CASE WHEN $country <> '' AND pl.country IS NULL THEN $country ELSE pl.country END,
                pl.region  = CASE WHEN $region  <> '' AND pl.region  IS NULL THEN $region  ELSE pl.region  END
            MERGE (p)-[r:LIVED_IN]->(pl)
              ON CREATE SET r.source=$book, r.confidence=$conf, r.evidence=$ev
            SET {_add_source('r', 'extracted')}, {_src_books('r')}
        """, name=row["subject"], place=place,
             city=pm.get("city",""), country=pm.get("country",""), region=pm.get("region",""),
             book=book, conf=row["confidence"], ev=row["evidence"])

    @staticmethod
    def _write_person_affiliated_with(tx, row: Dict) -> None:
        label = row["subject_label"]
        book  = row["source_book"]
        tx.run(f"""
            MERGE (p:{label} {{name: $name}})
            SET {_add_source('p', 'extracted')}, {_src_books('p')}
            MERGE (i:Institution {{name: $inst}})
            SET {_add_source('i', 'extracted')}, {_src_books('i')}
            MERGE (p)-[r:AFFILIATED_WITH]->(i)
              ON CREATE SET r.source=$book, r.confidence=$conf, r.evidence=$ev
            SET {_add_source('r', 'extracted')}, {_src_books('r')}
        """, name=row["subject"], inst=row["object"],
             book=book, conf=row["confidence"], ev=row["evidence"])

    @staticmethod
    def _write_person_traveled_to(tx, row: Dict) -> None:
        label = row["subject_label"]
        place = _resolve_place_name(tx, row["object"])
        pm    = row.get("object_place", {})
        book  = row["source_book"]
        tx.run(f"""
            MERGE (p:{label} {{name: $name}})
            SET {_add_source('p', 'extracted')}, {_src_books('p')}
            MERGE (pl:Place {{name: $place}})
            SET pl.place_type='historical',
                {_add_source('pl', 'extracted')}, {_src_books('pl')},
                pl.city    = CASE WHEN $city    <> '' AND pl.city    IS NULL THEN $city    ELSE pl.city    END,
                pl.country = CASE WHEN $country <> '' AND pl.country IS NULL THEN $country ELSE pl.country END,
                pl.region  = CASE WHEN $region  <> '' AND pl.region  IS NULL THEN $region  ELSE pl.region  END
            MERGE (p)-[r:TRAVELED_TO]->(pl)
              ON CREATE SET r.source=$book, r.confidence=$conf, r.evidence=$ev
            SET {_add_source('r', 'extracted')}, {_src_books('r')}
        """, name=row["subject"], place=place,
             city=pm.get("city",""), country=pm.get("country",""), region=pm.get("region",""),
             book=book, conf=row["confidence"], ev=row["evidence"])

    @staticmethod
    def _write_fragment_rel(tx, row: Dict) -> None:
        """Any relationship where subject or object is a Fragment.

        Always keys Fragment nodes on canonical_shelfmark (via ShelfmarkNormalizer)
        so they merge correctly with PGP nodes.  Handles both directions:
          Fragment → Person / Place / BookArticle / Institution
          Person / Scholar → Fragment  (WROTE, SENT_TO, MENTIONED_IN, etc.)
        """
        sl  = row["subject_label"]
        ol  = row["object_label"]
        rel = re.sub(r'[^A-Z_]', '', (row.get("relation") or "").upper())
        book = row["source_book"]

        def _merge_fragment(tx, display: str, tag: str) -> str:
            canonical = ShelfmarkNormalizer.to_canonical_id(display) or display
            tx.run(f"""
                MERGE (f:Fragment {{canonical_shelfmark: $canonical}})
                  ON CREATE SET f.shelfmark = $display
                SET {_add_source('f', tag)}, {_src_books('f')}
            """, canonical=canonical, display=display, book=book)
            return canonical

        def _merge_other(tx, label: str, name: str, tag: str) -> str:
            if label == "BookArticle":
                aid = _find_or_make_article_id(tx, name)
                tx.run(f"""
                    MERGE (n:BookArticle {{article_id: $aid}})
                      ON CREATE SET n.title = $name
                    SET {_add_source('n', tag)}, {_src_books('n')}
                """, aid=aid, name=name, book=book)
                return f"BookArticle {{article_id: '{aid}'}}"
            elif label == "Place":
                resolved = _resolve_place_name(tx, name)
                tx.run(f"""
                    MERGE (n:Place {{name: $name}})
                    SET n.place_type = 'historical',
                        {_add_source('n', tag)}, {_src_books('n')}
                """, name=resolved, book=book)
                return f"Place {{name: '{resolved}'}}"
            else:
                tx.run(f"""
                    MERGE (n:{label} {{name: $name}})
                    SET {_add_source('n', tag)}, {_src_books('n')}
                """, name=name, book=book)
                return f"{label} {{name: '{name}'}}"

        if sl == "Fragment":
            canonical_s = _merge_fragment(tx, row["subject"], "extracted")
            other_ref   = _merge_other(tx, ol, row["object"], "extracted")
            tx.run(f"""
                MATCH (f:Fragment {{canonical_shelfmark: $canonical}})
                MATCH (o:{other_ref})
                MERGE (f)-[r:{rel}]->(o)
                  ON CREATE SET r.source=$book, r.confidence=$conf, r.evidence=$ev
                SET {_add_source('r', 'extracted')}, {_src_books('r')}
            """, canonical=canonical_s,
                 book=book, conf=row["confidence"], ev=row["evidence"])
        else:
            # subject is Person/Scholar/etc, object is Fragment
            canonical_o = _merge_fragment(tx, row["object"], "extracted")
            other_ref   = _merge_other(tx, sl, row["subject"], "extracted")
            tx.run(f"""
                MATCH (s:{other_ref})
                MATCH (f:Fragment {{canonical_shelfmark: $canonical}})
                MERGE (s)-[r:{rel}]->(f)
                  ON CREATE SET r.source=$book, r.confidence=$conf, r.evidence=$ev
                SET {_add_source('r', 'extracted')}, {_src_books('r')}
            """, canonical=canonical_o,
                 book=book, conf=row["confidence"], ev=row["evidence"])

    @staticmethod
    def _write_generic_enhanced(tx, row: Dict) -> None:
        """Fallback for relationships not covered by specific handlers.
        Fragment nodes always use canonical_shelfmark — never name."""
        sl  = row["subject_label"] or "Entity"
        ol  = row["object_label"]  or "Entity"
        rel = row["relation"].replace(" ", "_")
        book = row["source_book"]

        # Build subject
        if sl == "Fragment":
            canonical_s = ShelfmarkNormalizer.to_canonical_id(row["subject"]) or row["subject"]
            tx.run(f"""
                MERGE (a:Fragment {{canonical_shelfmark: $canonical}})
                  ON CREATE SET a.shelfmark = $display
                SET {_add_source('a', 'extracted')}, {_src_books('a')}
            """, canonical=canonical_s, display=row["subject"], book=book)
            subj_clause = f"Fragment {{canonical_shelfmark: '{canonical_s}'}}"
        elif sl == "Place":
            subj = _resolve_place_name(tx, row["subject"])
            tx.run(f"MERGE (a:Place {{name: $name}}) SET {_add_source('a','extracted')}, {_src_books('a')}",
                   name=subj, book=book)
            subj_clause = f"Place {{name: '{subj}'}}"
        else:
            tx.run(f"MERGE (a:{sl} {{name: $name}}) SET {_add_source('a','extracted')}, {_src_books('a')}",
                   name=row["subject"], book=book)
            subj_clause = f"{sl} {{name: '{row['subject']}'}}"

        # Build object
        if ol == "Fragment":
            canonical_o = ShelfmarkNormalizer.to_canonical_id(row["object"]) or row["object"]
            tx.run(f"""
                MERGE (b:Fragment {{canonical_shelfmark: $canonical}})
                  ON CREATE SET b.shelfmark = $display
                SET {_add_source('b', 'extracted')}, {_src_books('b')}
            """, canonical=canonical_o, display=row["object"], book=book)
            obj_clause = f"Fragment {{canonical_shelfmark: '{canonical_o}'}}"
        elif ol == "BookArticle":
            aid = _find_or_make_article_id(tx, row["object"])
            tx.run(f"""
                MERGE (b:BookArticle {{article_id: $aid}})
                  ON CREATE SET b.title = $title
                SET {_add_source('b','extracted')}, {_src_books('b')}
            """, aid=aid, title=row["object"], book=book)
            obj_clause = f"BookArticle {{article_id: '{aid}'}}"
        elif ol == "Place":
            obj = _resolve_place_name(tx, row["object"])
            tx.run(f"MERGE (b:Place {{name: $name}}) SET {_add_source('b','extracted')}, {_src_books('b')}",
                   name=obj, book=book)
            obj_clause = f"Place {{name: '{obj}'}}"
        else:
            tx.run(f"MERGE (b:{ol} {{name: $name}}) SET {_add_source('b','extracted')}, {_src_books('b')}",
                   name=row["object"], book=book)
            obj_clause = f"{ol} {{name: '{row['object']}'}}"

        tx.run(f"""
            MATCH (a:{subj_clause})
            MATCH (b:{obj_clause})
            MERGE (a)-[r:{rel}]->(b)
              ON CREATE SET r.source=$book, r.confidence=$conf, r.evidence=$ev
            SET {_add_source('r', 'extracted')}, {_src_books('r')}
        """, book=book, conf=row["confidence"], ev=row["evidence"])

    # ===================================================================
    # Write helpers — enriched triplets (Pass 3)
    # ===================================================================

    @staticmethod
    def _write_enriched_triplet(tx, t: Dict, book: str) -> None:
        """Write one enriched triplet with correct merge keys per label type."""
        sl  = t["subject_type"] if t["subject_type"] in _VALID_LABELS else "Entity"
        ol  = t["object_type"]  if t["object_type"]  in _VALID_LABELS else "Entity"
        rel = t["relation"].replace(" ", "_")

        subj = t["subject"]
        obj  = t["object"]

        # Build subject node
        if sl == "Fragment":
            canonical_s = ShelfmarkNormalizer.to_canonical_id(subj) or subj
            tx.run(f"""
                MERGE (a:Fragment {{canonical_shelfmark: $canonical}})
                  ON CREATE SET a.shelfmark = $display
                SET {_add_source('a', 'enriched')},
                    a.source_books = CASE WHEN $book IN coalesce(a.source_books,[])
                        THEN coalesce(a.source_books,[]) ELSE coalesce(a.source_books,[]) + [$book] END
            """, canonical=canonical_s, display=subj, book=book)
            subj_ref = f"Fragment {{canonical_shelfmark: '{canonical_s}'}}"
        elif sl == "BookArticle":
            aid = _find_or_make_article_id(tx, subj)
            tx.run(f"""
                MERGE (a:BookArticle {{article_id: $aid}})
                  ON CREATE SET a.title = $title
                SET {_add_source('a', 'enriched')},
                    a.source_books = CASE WHEN $book IN coalesce(a.source_books,[])
                        THEN coalesce(a.source_books,[]) ELSE coalesce(a.source_books,[]) + [$book] END
            """, aid=aid, title=subj, book=book)
            subj_ref = f"BookArticle {{article_id: '{aid}'}}"
        else:
            tx.run(f"""
                MERGE (a:{sl} {{name: $name}})
                SET {_add_source('a', 'enriched')},
                    a.source_books = CASE WHEN $book IN coalesce(a.source_books,[])
                        THEN coalesce(a.source_books,[]) ELSE coalesce(a.source_books,[]) + [$book] END
            """, name=subj, book=book)
            subj_ref = f"{sl} {{name: '{subj}'}}"

        # Build object node
        if ol == "Fragment":
            canonical_o = ShelfmarkNormalizer.to_canonical_id(obj) or obj
            tx.run(f"""
                MERGE (b:Fragment {{canonical_shelfmark: $canonical}})
                  ON CREATE SET b.shelfmark = $display
                SET {_add_source('b', 'enriched')},
                    b.source_books = CASE WHEN $book IN coalesce(b.source_books,[])
                        THEN coalesce(b.source_books,[]) ELSE coalesce(b.source_books,[]) + [$book] END
            """, canonical=canonical_o, display=obj, book=book)
            obj_ref = f"Fragment {{canonical_shelfmark: '{canonical_o}'}}"
        elif ol == "BookArticle":
            aid = _find_or_make_article_id(tx, obj)
            tx.run(f"""
                MERGE (b:BookArticle {{article_id: $aid}})
                  ON CREATE SET b.title = $title
                SET {_add_source('b', 'enriched')},
                    b.source_books = CASE WHEN $book IN coalesce(b.source_books,[])
                        THEN coalesce(b.source_books,[]) ELSE coalesce(b.source_books,[]) + [$book] END
            """, aid=aid, title=obj, book=book)
            obj_ref = f"BookArticle {{article_id: '{aid}'}}"
        else:
            resolved_obj = _resolve_place_name(tx, obj) if ol == "Place" else obj
            tx.run(f"""
                MERGE (b:{ol} {{name: $name}})
                SET {_add_source('b', 'enriched')},
                    b.source_books = CASE WHEN $book IN coalesce(b.source_books,[])
                        THEN coalesce(b.source_books,[]) ELSE coalesce(b.source_books,[]) + [$book] END
            """, name=resolved_obj, book=book)
            obj_ref = f"{ol} {{name: '{resolved_obj}'}}"

        # Create the relationship
        tx.run(f"""
            MATCH (a:{subj_ref})
            MATCH (b:{obj_ref})
            MERGE (a)-[r:{rel}]->(b)
              ON CREATE SET r.source     = $book,
                            r.evidence   = $evidence,
                            r.confidence = $confidence,
                            r.data_sources = ['enriched'],
                            r.source_books = [$book]
              ON MATCH SET r.source_books = CASE WHEN $book IN coalesce(r.source_books,[])
                  THEN coalesce(r.source_books,[]) ELSE coalesce(r.source_books,[]) + [$book] END
        """, book=book, evidence=t.get("evidence",""), confidence=t.get("confidence","medium"))

    @staticmethod
    def _mark_enriched(tx, name: str, label: str, books: List[str]) -> None:
        tx.run(f"""
            MATCH (n:{label} {{name: $name}})
            SET n.relations_enriched  = true,
                n.enriched_from_books = $books
        """, name=name, books=books)

    # ===================================================================
    # Shared write helpers
    # ===================================================================

    @staticmethod
    def _write_people_and_places(
        tx,
        people_by_name: Dict[str, Dict],
        places_by_name: Dict[str, Dict],
        source_book: str,
    ) -> None:
        seen_people: set = set()
        for person in people_by_name.values():
            canonical = (person.get("name") or "").strip()
            if not canonical or canonical in seen_people:
                continue
            seen_people.add(canonical)
            role        = (person.get("role") or "").lower().strip()
            label       = "Scholar" if _is_scholar_role(role) else "Person"
            description = (person.get("description") or "").strip()[:1000]
            tx.run(f"""
                MERGE (p:{label} {{name: $name}})
                ON CREATE SET
                    p.role        = CASE WHEN $role        <> '' THEN $role        ELSE null END,
                    p.description = CASE WHEN $description <> '' THEN $description ELSE null END
                SET {_add_source('p', 'extracted')},
                    p.source_books = CASE WHEN $book IN coalesce(p.source_books,[])
                        THEN coalesce(p.source_books,[]) ELSE coalesce(p.source_books,[]) + [$book] END
            """, name=canonical, role=role, description=description, book=source_book)

        seen_places: set = set()
        for place in places_by_name.values():
            canonical = (place.get("name") or "").strip()
            if not canonical or canonical in seen_places:
                continue
            seen_places.add(canonical)
            city    = (place.get("city",    "") or "").strip()
            country = (place.get("country", "") or "").strip()
            region  = (place.get("region",  "") or "").strip()
            tx.run(f"""
                MERGE (pl:Place {{name: $name}})
                SET pl.place_type = 'historical',
                    {_add_source('pl', 'extracted')},
                    pl.source_books = CASE WHEN $book IN coalesce(pl.source_books,[])
                        THEN coalesce(pl.source_books,[]) ELSE coalesce(pl.source_books,[]) + [$book] END,
                    pl.city    = CASE WHEN $city    <> '' AND pl.city    IS NULL THEN $city    ELSE pl.city    END,
                    pl.country = CASE WHEN $country <> '' AND pl.country IS NULL THEN $country ELSE pl.country END,
                    pl.region  = CASE WHEN $region  <> '' AND pl.region  IS NULL THEN $region  ELSE pl.region  END
            """, name=canonical, city=city, country=country, region=region, book=source_book)

    @staticmethod
    def _enrich_place_node(tx, loc_meta: Dict) -> None:
        name    = (loc_meta.get("name") or "").strip()
        city    = (loc_meta.get("city",    "") or "").strip()
        country = (loc_meta.get("country", "") or "").strip()
        region  = (loc_meta.get("region",  "") or "").strip()
        if not name:
            return
        tx.run("""
            MATCH (pl:Place {name: $name})
            SET pl.city    = CASE WHEN $city    <> '' AND pl.city    IS NULL THEN $city    ELSE pl.city    END,
                pl.country = CASE WHEN $country <> '' AND pl.country IS NULL THEN $country ELSE pl.country END,
                pl.region  = CASE WHEN $region  <> '' AND pl.region  IS NULL THEN $region  ELSE pl.region  END
        """, name=name, city=city, country=country, region=region)

    # ===================================================================
    # Bulk runners
    # ===================================================================

    def import_all_enhanced(self, root: Path, dry_run: bool = False) -> Dict[str, int]:
        files = self.find_enhanced_files(root)
        logger.info(f"Found {len(files)} enhanced JSON files under {root}")
        totals: Dict[str, int] = defaultdict(int)
        failed = []
        for path in tqdm(files, desc="Enhanced JSONs"):
            rel = path.relative_to(root)
            try:
                counts = self.import_enhanced_file(path, dry_run=dry_run)
                for k, v in counts.items():
                    totals[k] += v
            except Exception as e:
                logger.error(f"  Failed {rel}: {e}")
                failed.append(str(rel))
        if failed:
            logger.warning(f"Failed enhanced files: {failed}")
        return dict(totals)

    def import_all_enriched(self, root: Path, dry_run: bool = False) -> Dict[str, int]:
        files = self.find_enriched_files(root)
        logger.info(f"Found {len(files)} enriched JSON files under {root}")
        totals: Dict[str, int] = defaultdict(int)
        for path in tqdm(files, desc="Enriched triplets"):
            try:
                counts = self.import_enriched_file(path, dry_run=dry_run)
                for k, v in counts.items():
                    totals[k] += v
            except Exception as e:
                logger.error(f"  Failed {path.name}: {e}")
                totals["import_errors"] += 1
        return dict(totals)

    def import_all(
        self,
        enhanced_root: Path,
        enriched_root: Path,
        dry_run: bool = False,
        enhanced_only: bool = False,
        enriched_only: bool = False,
    ):
        if not dry_run:
            self.ensure_constraints()

        totals: Dict[str, int] = defaultdict(int)

        if not enriched_only:
            logger.info("━━━ Pass 2: enhanced JSON files ━━━")
            counts = self.import_all_enhanced(enhanced_root, dry_run=dry_run)
            for k, v in counts.items():
                totals[f"enhanced_{k}"] += v

        if not enhanced_only and enriched_root.exists():
            logger.info("━━━ Pass 3: enriched relation files ━━━")
            counts = self.import_all_enriched(enriched_root, dry_run=dry_run)
            for k, v in counts.items():
                totals[f"enriched_{k}"] += v

        self._print_summary(totals, dry_run)

    def _print_summary(self, counts: Dict[str, int], dry_run: bool):
        tag = "[DRY RUN] " if dry_run else ""
        print(f"\n{tag}Import summary\n{'='*55}")
        for k, v in sorted(counts.items()):
            print(f"  {k}: {v:,}")
        print("=" * 55)
        if not dry_run:
            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run(
                        "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY count DESC"
                    )
                    print("\nNode counts:")
                    for r in result:
                        print(f"  {r['label']}: {r['count']:,}")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Module-level helpers (called inside transactions)
# ---------------------------------------------------------------------------

def _resolve_place_name(tx, raw_name: str) -> str:
    """Return canonical Place.name, checking Princeton name_variants."""
    result = tx.run("""
        OPTIONAL MATCH (pl:Place)
        WHERE toLower(pl.name) = toLower($name)
           OR ANY(v IN split(coalesce(pl.name_variants, ''), ',')
                  WHERE toLower(trim(v)) = toLower($name))
        RETURN pl.name AS canonical_name
        LIMIT 1
    """, name=raw_name)
    record = result.single()
    if record and record["canonical_name"]:
        canonical = record["canonical_name"]
        if canonical != raw_name:
            logger.debug(f"  Place '{raw_name}' → '{canonical}'")
        return canonical
    return raw_name


def _find_or_make_article_id(tx, title: str) -> str:
    """Return existing article_id for this title, or generate a new one.

    Tries a case-insensitive title match against existing BookArticle nodes
    so that enriched triplets reuse biblio.json nodes rather than creating
    floating duplicates.
    """
    result = tx.run(
        "OPTIONAL MATCH (b:BookArticle) WHERE toLower(b.title) = toLower($title) "
        "RETURN b.article_id AS aid LIMIT 1",
        title=title,
    )
    record = result.single()
    if record and record["aid"]:
        return record["aid"]
    return _make_article_id(title)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Import academic KG data (enhanced JSONs + enriched triplets) into Neo4j.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Count what would be imported without writing.")
    parser.add_argument("--enhanced-only", action="store_true",
                        help="Import enhanced JSONs only (skip enriched triplets).")
    parser.add_argument("--enriched-only", action="store_true",
                        help="Import enriched triplets only (skip enhanced JSONs).")
    parser.add_argument("--dir", "-d", metavar="SUBDIR", default=None,
                        help="Limit enhanced JSONs to a subdirectory under academic_literature/.")
    parser.add_argument("--enriched-dir", metavar="DIR", default=None,
                        help="Path to enriched_relations/ directory (default: auto-detected).")
    args = parser.parse_args()

    uri      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    user     = os.getenv("NEO4J_USER",     "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "genizah-prod")

    if not password:
        logger.error("NEO4J_PASSWORD not set")
        sys.exit(1)

    enhanced_root = _data_root
    if args.dir:
        enhanced_root = _data_root / args.dir
        if not enhanced_root.exists():
            logger.error(f"Directory not found: {enhanced_root}")
            sys.exit(1)

    enriched_root = Path(args.enriched_dir) if args.enriched_dir else _enriched_root

    importer = AcademicKGImporter(uri, user, password, database=database)
    try:
        importer.import_all(
            enhanced_root=enhanced_root,
            enriched_root=enriched_root,
            dry_run=args.dry_run,
            enhanced_only=args.enhanced_only,
            enriched_only=args.enriched_only,
        )
    finally:
        importer.close()


if __name__ == "__main__":
    main()
