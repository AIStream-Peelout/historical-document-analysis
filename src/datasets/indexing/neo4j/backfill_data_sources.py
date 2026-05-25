#!/usr/bin/env python3
"""
Backfill data_sources on pre-existing Neo4j nodes.

All nodes imported before the data_sources tagging was added have
data_sources = null.  This script stamps the correct tag on each node
type based on which importer created it:

  Fragment, Person, Place, Institution,
  Language, Tag                         → ["pgp"]   (Princeton CSV)
  BookArticle, Scholar                  → ["pgp"]   (Princeton sources.csv;
                                                     biblio nodes get "biblio"
                                                     appended when biblio_import
                                                     is re-run)

After this runs, re-running enhanced_kg_import will correctly produce
["pgp", "extracted"] on shared nodes instead of just ["extracted"].

Usage
-----
python backfill_data_sources.py              # live run
python backfill_data_sources.py --dry-run    # just count, no writes
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import dotenv
from neo4j import GraphDatabase

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))
dotenv.load_dotenv(project_root / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Each entry: (cypher_label, source_tag, description)
BACKFILL_TASKS = [
    ("Fragment",    "pgp",   "Fragment nodes from Princeton CSV"),
    ("Person",      "pgp",   "Person nodes from Princeton people.csv"),
    ("Place",       "pgp",   "Place nodes from Princeton places.csv"),
    ("Institution", "pgp",   "Institution nodes from Princeton / biblio"),
    ("Scholar",     "pgp",   "Scholar nodes from Princeton sources.csv"),
    ("BookArticle", "pgp",   "BookArticle nodes from Princeton sources.csv"),
    ("Language",    "pgp",   "Language nodes (Princeton)"),
    ("Tag",         "pgp",   "Tag nodes (Princeton)"),
]

# Relationship types that Princeton created without tags
RELATIONSHIP_BACKFILL = [
    ("HELD_AT",        "pgp"),
    ("MENTIONS_PLACE", "pgp"),
    ("SENT_TO",        "pgp"),
    ("WRITTEN_AT",     "pgp"),
    ("ORIGINATED_FROM","pgp"),
    ("LIVED_IN",       "pgp"),
    ("TRAVELED_TO",    "pgp"),
    ("WROTE",          "pgp"),
    ("REFERENCES",     "pgp"),
    ("HAS_TAG",        "pgp"),
    ("WRITTEN_IN",     "pgp"),
    ("JOINED_WITH",    "pgp"),
]


def count_untagged(session, label: str, tag: str = "pgp") -> int:
    """Count nodes that would be updated by backfill_nodes.

    For structural Princeton labels (_ALWAYS_PGP_LABELS) we count nodes that
    don't already carry the tag (regardless of whether data_sources is null or
    set to something else like ['extracted']).  For other labels we conservatively
    count only nodes with no data_sources at all.
    """
    if label in _ALWAYS_PGP_LABELS:
        condition = f"NOT '{tag}' IN coalesce(n.data_sources, [])"
    else:
        condition = "n.data_sources IS NULL"
    result = session.run(
        f"MATCH (n:{label}) WHERE {condition} RETURN count(n) AS cnt"
    )
    return result.single()["cnt"]


def count_untagged_rels(session, rel_type: str, tag: str = "pgp") -> int:
    result = session.run(
        f"MATCH ()-[r:{rel_type}]->() WHERE NOT '{tag}' IN coalesce(r.data_sources, []) RETURN count(r) AS cnt"
    )
    return result.single()["cnt"]


# These node types are definitively Princeton-origin — tag them as 'pgp' even if
# the enhanced importer already set data_sources = ['extracted'] (because the
# enhanced importer ran before the backfill and stomped on null with 'extracted').
# Scholar and BookArticle are deliberately excluded: many of those nodes are
# genuinely new from the LLM extraction and should not be mislabelled as 'pgp'.
_ALWAYS_PGP_LABELS = {"Fragment", "Person", "Place", "Institution", "Language", "Tag"}


def backfill_nodes(session, label: str, tag: str, batch_size: int = 5000) -> int:
    """Stamp *tag* onto all nodes of *label* that don't yet carry it.

    For structural Princeton node types (Fragment, Person, Place, etc.) we tag
    even nodes that already have data_sources set (e.g. ['extracted']) because
    the enhanced importer may have run before the backfill and written
    'extracted' onto what is actually a Princeton node.

    For Scholar and BookArticle we are conservative and only tag nodes that
    have no data_sources at all — those could legitimately be LLM-only.
    """
    if label in _ALWAYS_PGP_LABELS:
        # Tag every node that doesn't already carry this tag
        condition = f"NOT '{tag}' IN coalesce(n.data_sources, [])"
    else:
        # Only tag nodes with no provenance info yet
        condition = "n.data_sources IS NULL"

    total = 0
    while True:
        result = session.run(f"""
            MATCH (n:{label}) WHERE {condition}
            WITH n LIMIT {batch_size}
            SET n.data_sources = CASE
                WHEN '{tag}' IN coalesce(n.data_sources, []) THEN coalesce(n.data_sources, [])
                ELSE coalesce(n.data_sources, []) + ['{tag}']
            END
            RETURN count(n) AS updated
        """)
        updated = result.single()["updated"]
        total += updated
        if updated == 0:
            break
    return total


def backfill_relationships(session, rel_type: str, tag: str, batch_size: int = 5000) -> int:
    """Stamp *tag* onto all relationships of *rel_type* that don't yet carry it."""
    total = 0
    while True:
        result = session.run(f"""
            MATCH ()-[r:{rel_type}]->() WHERE NOT '{tag}' IN coalesce(r.data_sources, [])
            WITH r LIMIT {batch_size}
            SET r.data_sources = CASE
                WHEN '{tag}' IN coalesce(r.data_sources, []) THEN coalesce(r.data_sources, [])
                ELSE coalesce(r.data_sources, []) + ['{tag}']
            END
            RETURN count(r) AS updated
        """)
        updated = result.single()["updated"]
        total += updated
        if updated == 0:
            break
    return total


def main():
    parser = argparse.ArgumentParser(description="Backfill data_sources on pre-existing KG nodes.")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Count untagged nodes/rels without writing.")
    args = parser.parse_args()

    neo4j_uri      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    neo4j_user     = os.getenv("NEO4J_USER",     "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_database = os.getenv("NEO4J_DATABASE", "genizah-prod")

    if not neo4j_password:
        logger.error("NEO4J_PASSWORD not set")
        sys.exit(1)

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    try:
        with driver.session(database=neo4j_database) as session:

            # ----------------------------------------------------------------
            # Nodes
            # ----------------------------------------------------------------
            print("\n── Node backfill ──────────────────────────────────────")
            total_nodes = 0
            for label, tag, desc in BACKFILL_TASKS:
                untagged = count_untagged(session, label)
                if args.dry_run:
                    print(f"  {label:<15}  {untagged:>7,} untagged  (would tag as '{tag}')")
                else:
                    if untagged == 0:
                        print(f"  {label:<15}  ✅ already fully tagged")
                        continue
                    updated = backfill_nodes(session, label, tag)
                    total_nodes += updated
                    print(f"  {label:<15}  ✅ tagged {updated:>7,} nodes as '{tag}'")

            # ----------------------------------------------------------------
            # Relationships
            # ----------------------------------------------------------------
            print("\n── Relationship backfill ──────────────────────────────")
            total_rels = 0
            for rel_type, tag in RELATIONSHIP_BACKFILL:
                untagged = count_untagged_rels(session, rel_type)
                if args.dry_run:
                    if untagged:
                        print(f"  {rel_type:<20}  {untagged:>7,} untagged  (would tag as '{tag}')")
                else:
                    if untagged == 0:
                        continue
                    updated = backfill_relationships(session, rel_type, tag)
                    total_rels += updated
                    if updated:
                        print(f"  {rel_type:<20}  ✅ tagged {updated:>7,} rels as '{tag}'")

            if not args.dry_run:
                print(f"\n✅ Backfill complete — {total_nodes:,} nodes, {total_rels:,} relationships tagged")
                print("\nNext steps:")
                print("  1. Re-run biblio_import.py  → will append 'biblio' to its nodes")
                print("  2. Re-run enhanced_kg_import.py → shared nodes will now show ['pgp','extracted']")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
