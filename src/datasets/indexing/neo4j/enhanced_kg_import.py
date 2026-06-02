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

# Keywords that indicate a name is an institution, not a geographic place.
_INSTITUTION_KEYWORDS = {
    "university", "college", "library", "seminary", "institute", "museum",
    "academy", "archive", "school", "foundation", "society", "centre", "center",
}

def _is_institution_name(name: str) -> bool:
    """Return True if the name looks like a modern institution rather than a place."""
    lower = name.lower()
    return any(kw in lower for kw in _INSTITUTION_KEYWORDS)


def _src_books(alias: str) -> str:
    """Cypher snippet: append $book to <alias>.source_books if not already present.

    Usage in a SET clause:
        SET n.source_books = _src_books('n')
    Requires $book parameter to be passed to tx.run().
    """
    return (
        f"{alias}.source_books = CASE "
        f"WHEN $book IN coalesce({alias}.source_books, []) "
        f"THEN coalesce({alias}.source_books, []) "
        f"ELSE coalesce({alias}.source_books, []) + [$book] END"
    )


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

        people_locs    = data.get("people_locations", {})
        context_analysis = data.get("context_analysis", {})

        # Build a quick lookup of person role by name so triplet processing
        # can decide Scholar vs Person without re-scanning each time.
        #
        # The JSON has TWO separate people lists that must both be imported:
        #   people_locations.people      — modern scholars (authors, editors …)
        #   context_analysis.people_mentioned — historical figures (Sherirah Gaon,
        #                                       Rachel the Byzantine, Obadiah …)
        # Without merging both, all historical persons are silently dropped.
        people_by_name: Dict[str, Dict] = {}
        _all_people = (
            people_locs.get("people", [])
            + context_analysis.get("people_mentioned", [])
        )
        for person in _all_people:
            name = person.get("name", "").strip()
            if name and name not in people_by_name:   # people_locations wins on conflict
                people_by_name[name] = person
                for variant in person.get("name_variants", []):
                    if variant and variant.strip() not in people_by_name:
                        people_by_name[variant.strip()] = person

        # Build a lookup of geographic place metadata (city/country/region)
        # extracted by the LLM so we can enrich Place nodes on creation.
        # Same two-list pattern applies: merge people_locations.locations and
        # context_analysis.locations_mentioned.
        places_by_name: Dict[str, Dict] = {}
        _all_locs = (
            people_locs.get("locations", [])
            + context_analysis.get("locations_mentioned", [])
        )
        for loc in _all_locs:
            loc_type = loc.get("type", "").lower()
            # Skip institutions — they are not geographic places
            if loc_type in ("institution", "library", "university", "synagogue"):
                continue
            name = loc.get("name", "").strip()
            if name and name not in places_by_name:
                places_by_name[name] = loc
                for variant in loc.get("name_variants", []):
                    if variant and variant.strip() not in places_by_name:
                        places_by_name[variant.strip()] = loc

        # Derive the source book identifier from the file path
        # e.g. "Ashtor_1_1963_enhanced.json" → "Ashtor_1_1963"
        source_book = path.stem.replace("_enhanced", "")

        triplets: List[Dict[str, Any]] = data.get("kg_triplets", [])

        counts: Dict[str, int] = defaultdict(int)

        # ── Always import people and places from people_locations ──────────
        # People/places are extracted even when no triplets were generated.
        # Without this, anyone mentioned in the text but not in a triplet
        # (e.g. Sherirah Gaon, Rachel the Byzantine) is silently dropped.
        if not dry_run and (people_by_name or places_by_name):
            with self.driver.session(database=self.database) as session:
                session.execute_write(
                    self._write_people_locations,
                    people_by_name, places_by_name, source_book
                )
            counts["people_upserted"] = len(people_by_name)
            counts["places_upserted"] = len(places_by_name)

        if not triplets:
            logger.info(f"  No triplets in {path.name}")
            return dict(counts)

        # Build batches of normalised triplet dicts
        rows = []
        for t in triplets:
            row = self._normalise_triplet(t, people_by_name, places_by_name, source_book)
            if row:
                rows.append(row)

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

            # Enrich Place nodes with city/country/region from locations data
            for loc_meta in places_by_name.values():
                if any(loc_meta.get(k) for k in ("city", "country", "region")):
                    try:
                        session.execute_write(self._enrich_place_node, loc_meta)
                    except Exception as e:
                        logger.debug(f"  Place enrich failed for {loc_meta.get('name')}: {e}")

        return dict(counts)

    # ------------------------------------------------------------------
    # Triplet normalisation
    # ------------------------------------------------------------------
    def _normalise_triplet(
        self,
        triplet: Dict[str, Any],
        people_by_name: Dict[str, Dict],
        places_by_name: Dict[str, Dict],
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
        # Data-quality filters
        # ------------------------------------------------------------------
        # Filter out pseudo-place names that are archives/collections/concepts
        _NON_GEOGRAPHIC = {
            "cairo genizah", "genizah", "jewish theological seminary",
            "cambridge university library", "bodleian library", "british library",
            "taylor-schechter", "t-s collection", "dropsie college",
        }
        for place_field, place_type_field in ((obj, obj_type), (subj, subj_type)):
            if place_type_field.lower() == "place":
                if place_field.lower() in _NON_GEOGRAPHIC:
                    logger.debug(f"  Skipping non-geographic place '{place_field}'")
                    return None
                # Reject vague compound constructs like "Egypt-Palestine"
                if "-" in place_field and place_type_field.lower() == "place":
                    parts = [p.strip() for p in place_field.split("-")]
                    if len(parts) == 2 and all(len(p) > 3 for p in parts):
                        logger.debug(f"  Skipping compound place '{place_field}'")
                        return None

        # Filter out BookArticle nodes with missing/trivial titles (single word,
        # shelf-mark pattern, or matches a known non-title pattern).
        import re as _re
        _SHELFMARK_RE = _re.compile(r'^[A-Z][\w./-]+ \d+[\w./-]*$')
        for title_field, type_field in ((obj, obj_type), (subj, subj_type)):
            if type_field.lower() == "bookarticle":
                if not title_field:
                    return None
                if len(title_field.split()) < 2:
                    logger.debug(f"  Skipping single-word BookArticle '{title_field}'")
                    return None
                if _SHELFMARK_RE.match(title_field):
                    logger.debug(f"  Skipping shelf-mark as BookArticle '{title_field}'")
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

        # Reclassify institution names that the LLM incorrectly tagged as Place
        if subj_label == "Place" and _is_institution_name(subj):
            subj_label = "Institution"
        if obj_label == "Place" and _is_institution_name(obj):
            obj_label = "Institution"

        # Attach place metadata (city/country/region) so write methods can store it
        def _place_meta(name: str) -> Dict[str, str]:
            meta = places_by_name.get(name, {})
            return {
                "city":    meta.get("city",    "") or "",
                "country": meta.get("country", "") or "",
                "region":  meta.get("region",  "") or "",
            }

        base = {
            "subject":        subj,
            "subject_label":  subj_label,
            "subject_place":  _place_meta(subj) if subj_label == "Place" else {},
            "relation":       relation,
            "object":         obj,
            "object_label":   obj_label,
            "object_place":   _place_meta(obj) if obj_label == "Place" else {},
            "evidence":       evidence[:500],   # cap length
            "confidence":     confidence,
            "source_book":    source_book,
        }

        # Map to a write handler based on the (subject_label, relation, object_label) triple
        key = (subj_label, relation, obj_label)

        if key in (("Scholar", "WROTE", "BookArticle"), ("Person", "WROTE", "BookArticle")):
            return {**base, "write_fn": "_write_person_wrote_article"}

        if key == ("BookArticle", "CITES", "BookArticle"):
            return {**base, "write_fn": "_write_article_cites_article"}

        if key in (("Person", "LIVED_IN", "Place"), ("Scholar", "LIVED_IN", "Place")):
            return {**base, "write_fn": "_write_person_lived_in"}

        if key in (("Person", "AFFILIATED_WITH", "Institution"), ("Scholar", "AFFILIATED_WITH", "Institution"),
                   ("Person", "AFFILIATED_WITH", "Place"),        ("Scholar", "AFFILIATED_WITH", "Place")):
            return {**base, "write_fn": "_write_person_affiliated_with"}

        if key in (("Person", "TRAVELED_TO", "Place"), ("Scholar", "TRAVELED_TO", "Place"),
                   ("Person", "TRAVELED_TO", "Institution"), ("Scholar", "TRAVELED_TO", "Institution")):
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
        """Person/Scholar -[:LIVED_IN]-> Place  (historical residence only)

        If the object resolves to an Institution name, silently redirects to
        AFFILIATED_WITH so that academic affiliations don't pollute LIVED_IN.
        """
        label = row["subject_label"]
        if _is_institution_name(row["object"]):
            # Reroute: person "lived in" a university → affiliation, not residence
            return EnhancedKGImporter._write_person_affiliated_with(tx, row)
        place = EnhancedKGImporter._resolve_place_name(tx, row["object"])
        pm    = row.get("object_place", {})
        tx.run(f"""
            MERGE (p:{label} {{name: $name}})
            SET p.data_sources = CASE
                WHEN 'extracted' IN coalesce(p.data_sources, []) THEN coalesce(p.data_sources, [])
                ELSE coalesce(p.data_sources, []) + ['extracted']
            END,
                {_src_books('p')}
            MERGE (pl:Place {{name: $place}})
            SET pl.place_type  = 'historical',
                pl.data_sources = CASE
                WHEN 'extracted' IN coalesce(pl.data_sources, []) THEN coalesce(pl.data_sources, [])
                ELSE coalesce(pl.data_sources, []) + ['extracted']
            END,
                {_src_books('pl')},
                pl.city    = CASE WHEN $city    <> '' AND pl.city    IS NULL THEN $city    ELSE pl.city    END,
                pl.country = CASE WHEN $country <> '' AND pl.country IS NULL THEN $country ELSE pl.country END,
                pl.region  = CASE WHEN $region  <> '' AND pl.region  IS NULL THEN $region  ELSE pl.region  END
            MERGE (p)-[r:LIVED_IN]->(pl)
              ON CREATE SET r.source     = $source,
                            r.confidence = $confidence,
                            r.evidence   = $evidence
            SET r.data_sources = CASE
                WHEN 'extracted' IN coalesce(r.data_sources, []) THEN coalesce(r.data_sources, [])
                ELSE coalesce(r.data_sources, []) + ['extracted']
            END,
                {_src_books('r')}
        """, name=row["subject"], place=place,
             city=pm.get("city",""), country=pm.get("country",""), region=pm.get("region",""),
             source=row["source_book"], confidence=row["confidence"],
             evidence=row["evidence"], book=row["source_book"])

    @staticmethod
    def _write_person_affiliated_with(tx, row: Dict):
        """Person/Scholar -[:AFFILIATED_WITH]-> Institution

        Used for modern scholars whose affiliation with a university/library/institute
        is mentioned in a bibliography or author bio. Distinct from LIVED_IN which is
        reserved for historical figures residing in a geographic place.
        """
        label = row["subject_label"]
        institution = row["object"]
        tx.run(f"""
            MERGE (p:{label} {{name: $name}})
            SET p.data_sources = CASE
                WHEN 'extracted' IN coalesce(p.data_sources, []) THEN coalesce(p.data_sources, [])
                ELSE coalesce(p.data_sources, []) + ['extracted']
            END,
                {_src_books('p')}
            MERGE (i:Institution {{name: $institution}})
            SET i.data_sources = CASE
                WHEN 'extracted' IN coalesce(i.data_sources, []) THEN coalesce(i.data_sources, [])
                ELSE coalesce(i.data_sources, []) + ['extracted']
            END,
                {_src_books('i')}
            MERGE (p)-[r:AFFILIATED_WITH]->(i)
              ON CREATE SET r.source     = $source,
                            r.confidence = $confidence,
                            r.evidence   = $evidence
            SET r.data_sources = CASE
                WHEN 'extracted' IN coalesce(r.data_sources, []) THEN coalesce(r.data_sources, [])
                ELSE coalesce(r.data_sources, []) + ['extracted']
            END,
                {_src_books('r')}
        """, name=row["subject"], institution=institution,
             source=row["source_book"], confidence=row["confidence"],
             evidence=row["evidence"], book=row["source_book"])

    @staticmethod
    def _write_person_traveled_to(tx, row: Dict):
        """Person/Scholar -[:TRAVELED_TO]-> Place"""
        label = row["subject_label"]
        place = EnhancedKGImporter._resolve_place_name(tx, row["object"])
        pm    = row.get("object_place", {})
        tx.run(f"""
            MERGE (p:{label} {{name: $name}})
            SET p.data_sources = CASE
                WHEN 'extracted' IN coalesce(p.data_sources, []) THEN coalesce(p.data_sources, [])
                ELSE coalesce(p.data_sources, []) + ['extracted']
            END,
                {_src_books('p')}
            MERGE (pl:Place {{name: $place}})
            SET pl.place_type  = 'historical',
                pl.data_sources = CASE
                WHEN 'extracted' IN coalesce(pl.data_sources, []) THEN coalesce(pl.data_sources, [])
                ELSE coalesce(pl.data_sources, []) + ['extracted']
            END,
                {_src_books('pl')},
                pl.city    = CASE WHEN $city    <> '' AND pl.city    IS NULL THEN $city    ELSE pl.city    END,
                pl.country = CASE WHEN $country <> '' AND pl.country IS NULL THEN $country ELSE pl.country END,
                pl.region  = CASE WHEN $region  <> '' AND pl.region  IS NULL THEN $region  ELSE pl.region  END
            MERGE (p)-[r:TRAVELED_TO]->(pl)
              ON CREATE SET r.source     = $source,
                            r.confidence = $confidence,
                            r.evidence   = $evidence
            SET r.data_sources = CASE
                WHEN 'extracted' IN coalesce(r.data_sources, []) THEN coalesce(r.data_sources, [])
                ELSE coalesce(r.data_sources, []) + ['extracted']
            END,
                {_src_books('r')}
        """, name=row["subject"], place=place,
             city=pm.get("city",""), country=pm.get("country",""), region=pm.get("region",""),
             source=row["source_book"], confidence=row["confidence"],
             evidence=row["evidence"], book=row["source_book"])

    @staticmethod
    def _write_place_mentioned_in(tx, row: Dict):
        """Place -[:MENTIONED_IN]-> BookArticle"""
        article_id = _make_article_id(row["object"])
        place = EnhancedKGImporter._resolve_place_name(tx, row["subject"])
        pm    = row.get("subject_place", {})
        tx.run(f"""
            MERGE (pl:Place {{name: $place}})
            SET pl.place_type  = 'historical',
                pl.data_sources = CASE
                WHEN 'extracted' IN coalesce(pl.data_sources, []) THEN coalesce(pl.data_sources, [])
                ELSE coalesce(pl.data_sources, []) + ['extracted']
            END,
                {_src_books('pl')},
                pl.city    = CASE WHEN $city    <> '' AND pl.city    IS NULL THEN $city    ELSE pl.city    END,
                pl.country = CASE WHEN $country <> '' AND pl.country IS NULL THEN $country ELSE pl.country END,
                pl.region  = CASE WHEN $region  <> '' AND pl.region  IS NULL THEN $region  ELSE pl.region  END
            MERGE (b:BookArticle {{article_id: $article_id}})
              ON CREATE SET b.title = $title
            SET b.data_sources = CASE
                WHEN 'extracted' IN coalesce(b.data_sources, []) THEN coalesce(b.data_sources, [])
                ELSE coalesce(b.data_sources, []) + ['extracted']
            END,
                {_src_books('b')}
            MERGE (pl)-[r:MENTIONED_IN]->(b)
              ON CREATE SET r.source     = $source,
                            r.confidence = $confidence,
                            r.evidence   = $evidence
            SET r.data_sources = CASE
                WHEN 'extracted' IN coalesce(r.data_sources, []) THEN coalesce(r.data_sources, [])
                ELSE coalesce(r.data_sources, []) + ['extracted']
            END,
                {_src_books('r')}
        """, place=place, article_id=article_id, title=row["object"],
             city=pm.get("city",""), country=pm.get("country",""), region=pm.get("region",""),
             source=row["source_book"], confidence=row["confidence"],
             evidence=row["evidence"], book=row["source_book"])

    @staticmethod
    def _write_article_place_rel(tx, row: Dict):
        """BookArticle -[:ORIGINATED_FROM / :MENTIONS_PLACE]-> Place"""
        rel        = row["relation"]
        article_id = _make_article_id(row["subject"])
        place      = EnhancedKGImporter._resolve_place_name(tx, row["object"])
        pm         = row.get("object_place", {})
        tx.run(f"""
            MERGE (b:BookArticle {{article_id: $article_id}})
              ON CREATE SET b.title = $title
            SET b.data_sources = CASE
                WHEN 'extracted' IN coalesce(b.data_sources, []) THEN coalesce(b.data_sources, [])
                ELSE coalesce(b.data_sources, []) + ['extracted']
            END,
                {_src_books('b')}
            MERGE (pl:Place {{name: $place}})
            SET pl.place_type  = 'historical',
                pl.data_sources = CASE
                WHEN 'extracted' IN coalesce(pl.data_sources, []) THEN coalesce(pl.data_sources, [])
                ELSE coalesce(pl.data_sources, []) + ['extracted']
            END,
                {_src_books('pl')},
                pl.city    = CASE WHEN $city    <> '' AND pl.city    IS NULL THEN $city    ELSE pl.city    END,
                pl.country = CASE WHEN $country <> '' AND pl.country IS NULL THEN $country ELSE pl.country END,
                pl.region  = CASE WHEN $region  <> '' AND pl.region  IS NULL THEN $region  ELSE pl.region  END
            MERGE (b)-[r:{rel}]->(pl)
              ON CREATE SET r.source     = $source,
                            r.confidence = $confidence,
                            r.evidence   = $evidence
            SET r.data_sources = CASE
                WHEN 'extracted' IN coalesce(r.data_sources, []) THEN coalesce(r.data_sources, [])
                ELSE coalesce(r.data_sources, []) + ['extracted']
            END,
                {_src_books('r')}
        """, article_id=article_id, title=row["subject"], place=place,
             city=pm.get("city",""), country=pm.get("country",""), region=pm.get("region",""),
             source=row["source_book"], confidence=row["confidence"],
             evidence=row["evidence"], book=row["source_book"])

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
            END,
                {_src_books('a')}
            MERGE (b:{tgt_label} {{name: $tgt_name}})
            SET b.data_sources = CASE
                WHEN 'extracted' IN coalesce(b.data_sources, []) THEN coalesce(b.data_sources, [])
                ELSE coalesce(b.data_sources, []) + ['extracted']
            END,
                {_src_books('b')}
            MERGE (a)-[r:MENTIONS]->(b)
              ON CREATE SET r.source     = $source,
                            r.confidence = $confidence,
                            r.evidence   = $evidence
            SET r.data_sources = CASE
                WHEN 'extracted' IN coalesce(r.data_sources, []) THEN coalesce(r.data_sources, [])
                ELSE coalesce(r.data_sources, []) + ['extracted']
            END,
                {_src_books('r')}
        """, src_name=row["subject"], tgt_name=row["object"],
             source=row["source_book"], confidence=row["confidence"],
             evidence=row["evidence"], book=row["source_book"])

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
            END,
                {_src_books('f')}
            MERGE (t:{tgt_label} {{name: $name}})
            SET t.data_sources = CASE
                WHEN 'extracted' IN coalesce(t.data_sources, []) THEN coalesce(t.data_sources, [])
                ELSE coalesce(t.data_sources, []) + ['extracted']
            END,
                {_src_books('t')}
            MERGE (f)-[r:{rel}]->(t)
              ON CREATE SET r.source     = $source,
                            r.confidence = $confidence,
                            r.evidence   = $evidence
            SET r.data_sources = CASE
                WHEN 'extracted' IN coalesce(r.data_sources, []) THEN coalesce(r.data_sources, [])
                ELSE coalesce(r.data_sources, []) + ['extracted']
            END,
                {_src_books('r')}
        """, shelfmark=row["subject"], name=obj_name,
             source=row["source_book"], confidence=row["confidence"],
             evidence=row["evidence"], book=row["source_book"])

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
            END,
                {_src_books('a')}
            MERGE (b:{obj_label}  {{name: $obj}})
            SET b.data_sources = CASE
                WHEN 'extracted' IN coalesce(b.data_sources, []) THEN coalesce(b.data_sources, [])
                ELSE coalesce(b.data_sources, []) + ['extracted']
            END,
                {_src_books('b')}
            MERGE (a)-[r:{rel}]->(b)
              ON CREATE SET r.source     = $source,
                            r.confidence = $confidence,
                            r.evidence   = $evidence
            SET r.data_sources = CASE
                WHEN 'extracted' IN coalesce(r.data_sources, []) THEN coalesce(r.data_sources, [])
                ELSE coalesce(r.data_sources, []) + ['extracted']
            END,
                {_src_books('r')}
        """, subj=subj, obj=obj,
             source=row["source_book"], confidence=row["confidence"],
             evidence=row["evidence"], book=row["source_book"])

    @staticmethod
    def _enrich_place_node(tx, loc_meta: Dict):
        """Stamp city/country/region onto an existing Place node (idempotent)."""
        name    = loc_meta.get("name", "").strip()
        city    = loc_meta.get("city",    "") or ""
        country = loc_meta.get("country", "") or ""
        region  = loc_meta.get("region",  "") or ""
        if not name:
            return
        tx.run("""
            MATCH (pl:Place {name: $name})
            SET pl.city    = CASE WHEN $city    <> '' AND pl.city    IS NULL THEN $city    ELSE pl.city    END,
                pl.country = CASE WHEN $country <> '' AND pl.country IS NULL THEN $country ELSE pl.country END,
                pl.region  = CASE WHEN $region  <> '' AND pl.region  IS NULL THEN $region  ELSE pl.region  END
        """, name=name, city=city, country=country, region=region)

    @staticmethod
    def _write_people_locations(
        tx,
        people_by_name: Dict[str, Dict],
        places_by_name: Dict[str, Dict],
        source_book: str,
    ) -> None:
        """MERGE Person/Scholar and Place nodes from people_locations data.

        Called for every enhanced JSON even when there are no KG triplets, so
        that people and places extracted by the LLM are never silently dropped
        (e.g. Sherirah Gaon who appears in people_locations but not in triplets).

        people_by_name includes name-variant keys that point to the same person
        dict, so we deduplicate on the canonical ``person['name']`` field.
        """
        # ── People ─────────────────────────────────────────────────────────
        seen_people: set = set()
        for person in people_by_name.values():
            canonical = (person.get("name") or "").strip()
            if not canonical or canonical in seen_people:
                continue
            seen_people.add(canonical)

            role = (person.get("role") or "").lower().strip()
            label = "Scholar" if _is_scholar_role(role) else "Person"

            description  = (person.get("description") or "").strip()[:1000]
            gender       = (person.get("gender")      or "").strip()
            social_roles = person.get("social_roles", [])
            if not isinstance(social_roles, list):
                social_roles = []
            social_roles = [str(s) for s in social_roles if s]

            tx.run(f"""
                MERGE (p:{label} {{name: $name}})
                ON CREATE SET
                    p.role         = CASE WHEN $role        <> '' THEN $role        ELSE null END,
                    p.description  = CASE WHEN $description <> '' THEN $description ELSE null END,
                    p.gender       = CASE WHEN $gender      <> '' THEN $gender      ELSE null END,
                    p.social_roles = CASE WHEN size($social_roles) > 0 THEN $social_roles ELSE null END
                SET p.data_sources = CASE
                        WHEN 'extracted' IN coalesce(p.data_sources, []) THEN coalesce(p.data_sources, [])
                        ELSE coalesce(p.data_sources, []) + ['extracted']
                    END,
                    {_src_books('p')}
            """, name=canonical, role=role, description=description,
                 gender=gender, social_roles=social_roles, book=source_book)

        # ── Places ─────────────────────────────────────────────────────────
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
                SET pl.place_type   = 'historical',
                    pl.data_sources = CASE
                        WHEN 'extracted' IN coalesce(pl.data_sources, []) THEN coalesce(pl.data_sources, [])
                        ELSE coalesce(pl.data_sources, []) + ['extracted']
                    END,
                    {_src_books('pl')},
                    pl.city    = CASE WHEN $city    <> '' AND pl.city    IS NULL THEN $city    ELSE pl.city    END,
                    pl.country = CASE WHEN $country <> '' AND pl.country IS NULL THEN $country ELSE pl.country END,
                    pl.region  = CASE WHEN $region  <> '' AND pl.region  IS NULL THEN $region  ELSE pl.region  END
            """, name=canonical, city=city, country=country, region=region, book=source_book)

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
