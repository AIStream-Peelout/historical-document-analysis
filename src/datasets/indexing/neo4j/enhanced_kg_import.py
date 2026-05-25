#!/usr/bin/env python3
"""
Enhanced KG Import — Cairo Genizah Academic Literature

Reads every *_enhanced.json produced by the secondary LLM pipeline and
pushes the KG triplets into Neo4j, reusing the same node identity scheme as
the Princeton CSV import (knowlege_graph_poc.py) and the bibliography import
(biblio_import.py) so nothing is duplicated.

Key rules
---------
* Person whose role is 'author', 'scholar', 'editor', or 'translator'
  → MERGEd as a :Scholar node (same as the Princeton / biblio imports).
  All other persons → :Person node.

* BookArticle identity: sha1(title|author|year)[:16]  — identical hash to
  biblio_import.py so the same publication is never stored twice.
  When author/year are unknown, they are left as empty strings in the hash.

* All writes use MERGE so re-running the script is always safe.

* Relationships get `source`, `confidence`, and `evidence` properties so
  you can trace which triplet each edge came from.

Usage
-----
# Import all enhanced JSONs under academic_literature/
python enhanced_kg_import.py

# Dry-run: count what would be imported without touching Neo4j
python enhanced_kg_import.py --dry-run

# Import a specific directory only
python enhanced_kg_import.py --dir kettubah_palestine

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
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import dotenv
from neo4j import GraphDatabase
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
project_root = Path(__file__).parent.parent.parent.parent.parent
data_root    = project_root / "src" / "datasets" / "raw_data" / "cairo_genizah" / "academic_literature"
sys.path.append(str(project_root))
dotenv.load_dotenv(project_root / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("enhanced_kg_import.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers — keep identical to biblio_import.py
# ---------------------------------------------------------------------------

SCHOLAR_ROLES = {"author", "scholar", "editor", "translator", "co-author"}


def _make_article_id(title: str, author: str = "", year: str = "") -> str:
    """Deterministic 16-char hex ID for a BookArticle node.
    Identical algorithm to biblio_import.py — same publication → same ID."""
    key = f"{title}|{author}|{year}".lower().strip()
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _is_scholar_role(role: str) -> bool:
    return role.strip().lower() in SCHOLAR_ROLES


# ---------------------------------------------------------------------------
# Importer
# ---------------------------------------------------------------------------

class EnhancedKGImporter:
    """Import KG triplets from *_enhanced.json files into Neo4j."""

    BATCH_SIZE = 100

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver   = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        logger.info(f"Connected to Neo4j at {uri}, database={database}")

    def close(self):
        self.driver.close()

    def ensure_constraints(self):
        """Ensure the schema constraints exist (idempotent)."""
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

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    @staticmethod
    def find_enhanced_files(root: Path) -> List[Path]:
        return sorted(root.rglob("*_enhanced.json"))

    # ------------------------------------------------------------------
    # Per-file import
    # ------------------------------------------------------------------
    def import_file(self, path: Path, dry_run: bool = False) -> Dict[str, int]:
        """Import one enhanced JSON.  Returns counts of imported entities."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Build a quick lookup of person role by name so triplet processing
        # can decide Scholar vs Person without re-scanning each time.
        people_by_name: Dict[str, Dict] = {}
        for person in data.get("people_locations", {}).get("people", []):
            name = person.get("name", "").strip()
            if name:
                people_by_name[name] = person
                # Index name variants too
                for variant in person.get("name_variants", []):
                    if variant:
                        people_by_name[variant.strip()] = person

        triplets: List[Dict[str, Any]] = data.get("kg_triplets", [])
        if not triplets:
            logger.info(f"  No triplets in {path.name}")
            return {}

        # Derive the source book identifier from the file path
        # e.g. "Ashtor_1_1963_enhanced.json" → "Ashtor_1_1963"
        source_book = path.stem.replace("_enhanced", "")

        # Build batches of normalised triplet dicts
        rows = []
        for t in triplets:
            row = self._normalise_triplet(t, people_by_name, source_book)
            if row:
                rows.append(row)

        counts = defaultdict(int)
        counts["triplets_total"] = len(triplets)
        counts["triplets_valid"] = len(rows)

        if dry_run:
            # Just count without writing
            for row in rows:
                counts[f"would_create_{row['write_fn']}"] += 1
            return dict(counts)

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(rows), self.BATCH_SIZE):
                batch = rows[i : i + self.BATCH_SIZE]
                for row in batch:
                    fn = row["write_fn"]
                    try:
                        session.execute_write(getattr(self, fn), row)
                        counts[fn] += 1
                    except Exception as e:
                        logger.warning(f"  Failed {fn} for {row}: {e}")

        return dict(counts)

    # ------------------------------------------------------------------
    # Triplet normalisation
    # ------------------------------------------------------------------
    def _normalise_triplet(
        self,
        triplet: Dict[str, Any],
        people_by_name: Dict[str, Dict],
        source_book: str,
    ) -> Optional[Dict[str, Any]]:
        """Convert a raw triplet dict into a canonical write-instruction dict."""
        subj      = (triplet.get("subject") or "").strip()
        subj_type = (triplet.get("subject_type") or "").strip()
        relation  = (triplet.get("relation") or "").strip().upper()
        obj       = (triplet.get("object") or "").strip()
        obj_type  = (triplet.get("object_type") or "").strip()
        evidence  = (triplet.get("evidence") or "").strip()
        confidence = (triplet.get("confidence") or "medium").strip()

        if not (subj and relation and obj):
            return None

        # ------------------------------------------------------------------
        # Resolve actual Neo4j labels, upgrading Person→Scholar where needed
        # ------------------------------------------------------------------
        def _resolve_label(name: str, declared_type: str) -> str:
            """Return Scholar instead of Person when role warrants it."""
            if declared_type.lower() == "person":
                person_rec = people_by_name.get(name, {})
                role = person_rec.get("role", "").lower()
                if _is_scholar_role(role):
                    return "Scholar"
            return _LABEL_MAP.get(declared_type, declared_type)

        subj_label = _resolve_label(subj, subj_type)
        obj_label  = _resolve_label(obj, obj_type)

        base = {
            "subject":      subj,
            "subject_label": subj_label,
            "relation":     relation,
            "object":       obj,
            "object_label": obj_label,
            "evidence":     evidence[:500],   # cap length
            "confidence":   confidence,
            "source_book":  source_book,
        }

        # Map to a write handler based on the (subject_label, relation, object_label) triple
        key = (subj_label, relation, obj_label)

        if key in (("Scholar", "WROTE", "BookArticle"), ("Person", "WROTE", "BookArticle")):
            return {**base, "write_fn": "_write_person_wrote_article"}

        if key == ("BookArticle", "CITES", "BookArticle"):
            return {**base, "write_fn": "_write_article_cites_article"}

        if key in (("Person", "LIVED_IN", "Place"), ("Scholar", "LIVED_IN", "Place")):
            return {**base, "write_fn": "_write_person_lived_in"}

        if key in (("Person", "TRAVELED_TO", "Place"), ("Scholar", "TRAVELED_TO", "Place")):
            return {**base, "write_fn": "_write_person_traveled_to"}

        if key == ("Place", "MENTIONED_IN", "BookArticle"):
            return {**base, "write_fn": "_write_place_mentioned_in"}

        if key in (("BookArticle", "ORIGINATED_FROM", "Place"),
                   ("BookArticle", "MENTIONS_PLACE", "Place")):
            return {**base, "write_fn": "_write_article_place_rel"}

        if key in (("Person", "MENTIONS", "Person"), ("Scholar", "MENTIONS", "Person")):
            return {**base, "write_fn": "_write_person_mentions_person"}

        if key in (("Fragment", "MENTIONS", "Person"), ("Fragment", "MENTIONS_PLACE", "Place")):
            return {**base, "write_fn": "_write_fragment_mentions"}

        # Generic fallback: store as a labelled relationship with metadata
        return {**base, "write_fn": "_write_generic"}

    # ------------------------------------------------------------------
    # Place-name resolver
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_place_name(tx, raw_name: str) -> str:
        """Return the canonical Place.name for *raw_name*.

        Princeton stores diacriticized/formal names (e.g. "Fusṭāṭ") and keeps
        plain-ASCII variants in the comma-delimited name_variants string
        (e.g. "Fustat, Old Cairo").  The LLM typically extracts the simplified
        ASCII form, so we need to check both exact name and each variant token
        (case-insensitive) before falling back to creating a new node.

        Returns raw_name unchanged when no existing Place matches.
        """
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
                logger.debug(f"  Place '{raw_name}' → resolved to canonical '{canonical}'")
            return canonical
        return raw_name

    # ------------------------------------------------------------------
    # Write handlers (all called inside execute_write)
    # ------------------------------------------------------------------

    @staticmethod
    def _write_person_wrote_article(tx, row: Dict):
        """Scholar/Person -[:WROTE]-> BookArticle"""
        label      = row["subject_label"]   # Scholar or Person
        author_name = row["subject"]
        title       = row["object"]
        article_id  = _make_article_id(title, author_name)

        tx.run(f"""
            MERGE (a:{label} {{name: $name}})
            SET a.data_sources = CASE
                WHEN 'extracted' IN coalesce(a.data_sources, []) THEN coalesce(a.data_sources, [])
                ELSE coalesce(a.data_sources, []) + ['extracted']
            END
            MERGE (b:BookArticle {{article_id: $article_id}})
              ON CREATE SET b.title = $title
            SET b.data_sources = CASE
                WHEN 'extracted' IN coalesce(b.data_sources, []) THEN coalesce(b.data_sources, [])
                ELSE coalesce(b.data_sources, []) + ['extracted']
            END
            MERGE (a)-[r:WROTE]->(b)
              ON CREATE SET r.source     = $source,
                            r.confidence = $confidence,
                            r.evidence   = $evidence
            SET r.data_sources = CASE
                WHEN 'extracted' IN coalesce(r.data_sources, []) THEN coalesce(r.data_sources, [])
                ELSE coalesce(r.data_sources, []) + ['extracted']
            END
        """, name=author_name, article_id=article_id, title=title,
             source=row["source_book"], confidence=row["confidence"],
             evidence=row["evidence"])

    @staticmethod
    def _write_article_cites_article(tx, row: Dict):
        """BookArticle -[:CITES]-> BookArticle"""
        src_id  = _make_article_id(row["subject"])
        tgt_id  = _make_article_id(row["object"])

        tx.run("""
            MERGE (a:BookArticle {article_id: $src_id})
              ON CREATE SET a.title = $src_title
            SET a.data_sources = CASE
                WHEN 'extracted' IN coalesce(a.data_sources, []) THEN coalesce(a.data_sources, [])
                ELSE coalesce(a.data_sources, []) + ['extracted']
            END
            MERGE (b:BookArticle {article_id: $tgt_id})
              ON CREATE SET b.title = $tgt_title
            SET b.data_sources = CASE
                WHEN 'extracted' IN coalesce(b.data_sources, []) THEN coalesce(b.data_sources, [])
                ELSE coalesce(b.data_sources, []) + ['extracted']
            END
            MERGE (a)-[r:CITES]->(b)
              ON CREATE SET r.source     = $source,
                            r.confidence = $confidence,
                            r.evidence   = $evidence
            SET r.data_sources = CASE
                WHEN 'extracted' IN coalesce(r.data_sources, []) THEN coalesce(r.data_sources, [])
                ELSE coalesce(r.data_sources, []) + ['extracted']
            END
        """, src_id=src_id, src_title=row["subject"],
             tgt_id=tgt_id, tgt_title=row["object"],
             source=row["source_book"], confidence=row["confidence"],
             evidence=row["evidence"])

    @staticmethod
    def _write_person_lived_in(tx, row: Dict):
        """Person/Scholar -[:LIVED_IN]-> Place"""
        label = row["subject_label"]
        place = EnhancedKGImporter._resolve_place_name(tx, row["object"])
        tx.run(f"""
            MERGE (p:{label} {{name: $name}})
            SET p.data_sources = CASE
                WHEN 'extracted' IN coalesce(p.data_sources, []) THEN coalesce(p.data_sources, [])
                ELSE coalesce(p.data_sources, []) + ['extracted']
            END
            MERGE (pl:Place {{name: $place}})
            SET pl.data_sources = CASE
                WHEN 'extracted' IN coalesce(pl.data_sources, []) THEN coalesce(pl.data_sources, [])
                ELSE coalesce(pl.data_sources, []) + ['extracted']
            END
            MERGE (p)-[r:LIVED_IN]->(pl)
              ON CREATE SET r.source     = $source,
                            r.confidence = $confidence,
                            r.evidence   = $evidence
            SET r.data_sources = CASE
                WHEN 'extracted' IN coalesce(r.data_sources, []) THEN coalesce(r.data_sources, [])
                ELSE coalesce(r.data_sources, []) + ['extracted']
            END
        """, name=row["subject"], place=place,
             source=row["source_book"], confidence=row["confidence"],
             evidence=row["evidence"])

    @staticmethod
    def _write_person_traveled_to(tx, row: Dict):
        """Person/Scholar -[:TRAVELED_TO]-> Place"""
        label = row["subject_label"]
        place = EnhancedKGImporter._resolve_place_name(tx, row["object"])
        tx.run(f"""
            MERGE (p:{label} {{name: $name}})
            SET p.data_sources = CASE
                WHEN 'extracted' IN coalesce(p.data_sources, []) THEN coalesce(p.data_sources, [])
                ELSE coalesce(p.data_sources, []) + ['extracted']
            END
            MERGE (pl:Place {{name: $place}})
            SET pl.data_sources = CASE
                WHEN 'extracted' IN coalesce(pl.data_sources, []) THEN coalesce(pl.data_sources, [])
                ELSE coalesce(pl.data_sources, []) + ['extracted']
            END
            MERGE (p)-[r:TRAVELED_TO]->(pl)
              ON CREATE SET r.source     = $source,
                            r.confidence = $confidence,
                            r.evidence   = $evidence
            SET r.data_sources = CASE
                WHEN 'extracted' IN coalesce(r.data_sources, []) THEN coalesce(r.data_sources, [])
                ELSE coalesce(r.data_sources, []) + ['extracted']
            END
        """, name=row["subject"], place=place,
             source=row["source_book"], confidence=row["confidence"],
             evidence=row["evidence"])

    @staticmethod
    def _write_place_mentioned_in(tx, row: Dict):
        """Place -[:MENTIONED_IN]-> BookArticle"""
        article_id = _make_article_id(row["object"])
        place = EnhancedKGImporter._resolve_place_name(tx, row["subject"])
        tx.run("""
            MERGE (pl:Place {name: $place})
            SET pl.data_sources = CASE
                WHEN 'extracted' IN coalesce(pl.data_sources, []) THEN coalesce(pl.data_sources, [])
                ELSE coalesce(pl.data_sources, []) + ['extracted']
            END
            MERGE (b:BookArticle {article_id: $article_id})
              ON CREATE SET b.title = $title
            SET b.data_sources = CASE
                WHEN 'extracted' IN coalesce(b.data_sources, []) THEN coalesce(b.data_sources, [])
                ELSE coalesce(b.data_sources, []) + ['extracted']
            END
            MERGE (pl)-[r:MENTIONED_IN]->(b)
              ON CREATE SET r.source     = $source,
                            r.confidence = $confidence,
                            r.evidence   = $evidence
            SET r.data_sources = CASE
                WHEN 'extracted' IN coalesce(r.data_sources, []) THEN coalesce(r.data_sources, [])
                ELSE coalesce(r.data_sources, []) + ['extracted']
            END
        """, place=place, article_id=article_id, title=row["object"],
             source=row["source_book"], confidence=row["confidence"],
             evidence=row["evidence"])

    @staticmethod
    def _write_article_place_rel(tx, row: Dict):
        """BookArticle -[:ORIGINATED_FROM / :MENTIONS_PLACE]-> Place"""
        rel        = row["relation"]
        article_id = _make_article_id(row["subject"])
        place      = EnhancedKGImporter._resolve_place_name(tx, row["object"])
        tx.run(f"""
            MERGE (b:BookArticle {{article_id: $article_id}})
              ON CREATE SET b.title = $title
            SET b.data_sources = CASE
                WHEN 'extracted' IN coalesce(b.data_sources, []) THEN coalesce(b.data_sources, [])
                ELSE coalesce(b.data_sources, []) + ['extracted']
            END
            MERGE (pl:Place {{name: $place}})
            SET pl.data_sources = CASE
                WHEN 'extracted' IN coalesce(pl.data_sources, []) THEN coalesce(pl.data_sources, [])
                ELSE coalesce(pl.data_sources, []) + ['extracted']
            END
            MERGE (b)-[r:{rel}]->(pl)
              ON CREATE SET r.source     = $source,
                            r.confidence = $confidence,
                            r.evidence   = $evidence
            SET r.data_sources = CASE
                WHEN 'extracted' IN coalesce(r.data_sources, []) THEN coalesce(r.data_sources, [])
                ELSE coalesce(r.data_sources, []) + ['extracted']
            END
        """, article_id=article_id, title=row["subject"], place=place,
             source=row["source_book"], confidence=row["confidence"],
             evidence=row["evidence"])

    @staticmethod
    def _write_person_mentions_person(tx, row: Dict):
        """Scholar/Person mentions another Person (e.g. cited in text)"""
        src_label = row["subject_label"]
        tgt_label = row["object_label"]
        tx.run(f"""
            MERGE (a:{src_label} {{name: $src_name}})
            SET a.data_sources = CASE
                WHEN 'extracted' IN coalesce(a.data_sources, []) THEN coalesce(a.data_sources, [])
                ELSE coalesce(a.data_sources, []) + ['extracted']
            END
            MERGE (b:{tgt_label} {{name: $tgt_name}})
            SET b.data_sources = CASE
                WHEN 'extracted' IN coalesce(b.data_sources, []) THEN coalesce(b.data_sources, [])
                ELSE coalesce(b.data_sources, []) + ['extracted']
            END
            MERGE (a)-[r:MENTIONS]->(b)
              ON CREATE SET r.source     = $source,
                            r.confidence = $confidence,
                            r.evidence   = $evidence
            SET r.data_sources = CASE
                WHEN 'extracted' IN coalesce(r.data_sources, []) THEN coalesce(r.data_sources, [])
                ELSE coalesce(r.data_sources, []) + ['extracted']
            END
        """, src_name=row["subject"], tgt_name=row["object"],
             source=row["source_book"], confidence=row["confidence"],
             evidence=row["evidence"])

    @staticmethod
    def _write_fragment_mentions(tx, row: Dict):
        """Fragment -[:MENTIONS / :MENTIONS_PLACE]-> Person/Place"""
        rel        = row["relation"]
        tgt_label  = row["object_label"]
        # Resolve place names against Princeton variants; leave Person names as-is.
        obj_name = (EnhancedKGImporter._resolve_place_name(tx, row["object"])
                    if tgt_label == "Place" else row["object"])
        # Fragment nodes are keyed by canonical_shelfmark; if we only have a
        # display form here, just store it — it may already exist from Princeton.
        tx.run(f"""
            MERGE (f:Fragment {{shelfmark: $shelfmark}})
            SET f.data_sources = CASE
                WHEN 'extracted' IN coalesce(f.data_sources, []) THEN coalesce(f.data_sources, [])
                ELSE coalesce(f.data_sources, []) + ['extracted']
            END
            MERGE (t:{tgt_label} {{name: $name}})
            SET t.data_sources = CASE
                WHEN 'extracted' IN coalesce(t.data_sources, []) THEN coalesce(t.data_sources, [])
                ELSE coalesce(t.data_sources, []) + ['extracted']
            END
            MERGE (f)-[r:{rel}]->(t)
              ON CREATE SET r.source     = $source,
                            r.confidence = $confidence,
                            r.evidence   = $evidence
            SET r.data_sources = CASE
                WHEN 'extracted' IN coalesce(r.data_sources, []) THEN coalesce(r.data_sources, [])
                ELSE coalesce(r.data_sources, []) + ['extracted']
            END
        """, shelfmark=row["subject"], name=obj_name,
             source=row["source_book"], confidence=row["confidence"],
             evidence=row["evidence"])

    @staticmethod
    def _write_generic(tx, row: Dict):
        """Fallback: store as generic labelled nodes + relationship."""
        subj_label = row["subject_label"] or "Entity"
        obj_label  = row["object_label"]  or "Entity"
        rel        = row["relation"].replace(" ", "_")
        # Resolve place names for subject/object when applicable
        subj = (EnhancedKGImporter._resolve_place_name(tx, row["subject"])
                if subj_label == "Place" else row["subject"])
        obj  = (EnhancedKGImporter._resolve_place_name(tx, row["object"])
                if obj_label == "Place" else row["object"])
        tx.run(f"""
            MERGE (a:{subj_label} {{name: $subj}})
            SET a.data_sources = CASE
                WHEN 'extracted' IN coalesce(a.data_sources, []) THEN coalesce(a.data_sources, [])
                ELSE coalesce(a.data_sources, []) + ['extracted']
            END
            MERGE (b:{obj_label}  {{name: $obj}})
            SET b.data_sources = CASE
                WHEN 'extracted' IN coalesce(b.data_sources, []) THEN coalesce(b.data_sources, [])
                ELSE coalesce(b.data_sources, []) + ['extracted']
            END
            MERGE (a)-[r:{rel}]->(b)
              ON CREATE SET r.source     = $source,
                            r.confidence = $confidence,
                            r.evidence   = $evidence
            SET r.data_sources = CASE
                WHEN 'extracted' IN coalesce(r.data_sources, []) THEN coalesce(r.data_sources, [])
                ELSE coalesce(r.data_sources, []) + ['extracted']
            END
        """, subj=subj, obj=obj,
             source=row["source_book"], confidence=row["confidence"],
             evidence=row["evidence"])

    # ------------------------------------------------------------------
    # Bulk import
    # ------------------------------------------------------------------
    def import_all(self, root: Path, dry_run: bool = False):
        files = self.find_enhanced_files(root)
        logger.info(f"Found {len(files)} enhanced JSON files under {root}")

        if not files:
            logger.warning("No *_enhanced.json files found — nothing to import")
            return

        if not dry_run:
            self.ensure_constraints()

        total_counts: Dict[str, int] = defaultdict(int)
        failed = []

        for path in tqdm(files, desc="Importing enhanced KG"):
            rel = path.relative_to(root)
            logger.info(f"Processing {rel}")
            try:
                counts = self.import_file(path, dry_run=dry_run)
                for k, v in counts.items():
                    total_counts[k] += v
                logger.info(f"  {counts}")
            except Exception as e:
                logger.error(f"  Failed: {e}")
                failed.append(str(rel))

        self._print_summary(total_counts, failed, dry_run)

    # ------------------------------------------------------------------
    def _print_summary(self, counts: Dict[str, int], failed: List[str], dry_run: bool):
        tag = "[DRY RUN] " if dry_run else ""
        print(f"\n{tag}Import summary")
        print("=" * 50)
        for k, v in sorted(counts.items()):
            print(f"  {k}: {v:,}")
        if failed:
            print(f"\n  ❌ Failed files ({len(failed)}):")
            for f in failed:
                print(f"    • {f}")
        print("=" * 50)

        if not dry_run:
            # Post-import graph stats
            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run("""
                        MATCH (n)
                        RETURN labels(n)[0] AS label, count(*) AS count
                        ORDER BY count DESC
                    """)
                    print("\nNode counts:")
                    for r in result:
                        print(f"  {r['label']}: {r['count']:,}")
            except Exception as e:
                logger.warning(f"Could not fetch node counts: {e}")


# ---------------------------------------------------------------------------
# Label normalisation map
# ---------------------------------------------------------------------------
_LABEL_MAP = {
    "Person":       "Person",
    "Scholar":      "Scholar",
    "Place":        "Place",
    "Fragment":     "Fragment",
    "BookArticle":  "BookArticle",
    "Institution":  "Institution",
    "Entity":       "Entity",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Import KG triplets from enhanced JSON files into Neo4j.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Count what would be imported without writing to Neo4j.",
    )
    parser.add_argument(
        "--dir", "-d",
        metavar="SUBDIR",
        default=None,
        help="Limit to a subdirectory under academic_literature/ "
             "(e.g. french or kettubah_palestine).",
    )
    args = parser.parse_args()

    neo4j_uri      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    neo4j_user     = os.getenv("NEO4J_USER",     "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_database = os.getenv("NEO4J_DATABASE", "genizah-prod")

    if not neo4j_password:
        logger.error("NEO4J_PASSWORD not set")
        sys.exit(1)

    search_root = data_root
    if args.dir:
        search_root = data_root / args.dir
        if not search_root.exists():
            logger.error(f"Directory not found: {search_root}")
            sys.exit(1)

    importer = EnhancedKGImporter(neo4j_uri, neo4j_user, neo4j_password, database=neo4j_database)
    try:
        importer.import_all(search_root, dry_run=args.dry_run)
    finally:
        importer.close()


if __name__ == "__main__":
    main()
