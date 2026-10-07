#!/usr/bin/env python3
"""
PGP website person–document relation import
============================================

Loads the person→document relationships scraped from the Princeton Geniza
Project website (``ktiv-scraper/princeton/pgp_person_document_relations.csv``)
into the knowledge graph. These are the role relationships (Scribe, Recipient,
Mentioned, …) that are absent from the pgp-metadata GitHub export the graph
was originally built from.

Graph model
-----------
There are no Document nodes in this graph: ``Fragment`` nodes *are* PGP
documents (keyed by unique ``pgpid``, with the joined shelfmark string on
multi-fragment documents). Edges therefore go directly

    (:Person)-[:<ROLE_TYPE> {pgpid, relation, uncertain, deceased,
                             source: "pgp-website",
                             data_sources: ["pgp_website"]}]->(:Fragment)

following the house style of per-type relationship names (MENTIONS_PERSON,
TRANSCRIBED, …) rather than a generic type property — ``RELATED_TO`` already
carries interpersonal semantics in this graph. The scraped relation string is
stored verbatim in ``relation``; trailing "(uncertain)" / "(deceased)"
qualifiers are additionally parsed into boolean edge properties.

Matching
--------
* Person: by ``url`` (verbatim people.csv value) first, then exact ``name``.
  Rows whose person cannot be matched (or whose name is ambiguous across
  several Person nodes with no url to disambiguate) are reported, not dropped
  silently.
* Fragment: by ``pgpid`` first (unique + indexed), then by the full joined
  ``shelfmark`` string, then by each individual shelfmark in
  ``fragment_shelfmarks`` (one edge per matched node).

Idempotency
-----------
Edges are MERGEd on ``(person, fragment, rel type, pgpid, relation)``, so
reruns create nothing new. All written edges carry
``source = "pgp-website"`` for verification and rollback.

Usage
-----
python pgp_person_relations_import.py --dry-run   # match + report only
python pgp_person_relations_import.py             # load
"""

import argparse
import csv
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dotenv
from neo4j import GraphDatabase, Driver
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[4]
dotenv.load_dotenv(project_root / ".env")

import os  # noqa: E402  (after dotenv, matching sibling import scripts)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pgp_person_relations_import")

DEFAULT_CSV = Path(
    "/Users/isaac/Documents/GitHub/ktiv-scraper/princeton/pgp_person_document_relations.csv"
)
UNMATCHED_REPORT = project_root / "artifacts" / "pgp_person_relations_unmatched.csv"

SOURCE_TAG = "pgp-website"
DATA_SOURCE_TOKEN = "pgp_website"

# Base PGP role -> relationship type. Qualifiers are parsed off before lookup;
# the verbatim string (qualifiers included) is always kept on the edge.
ROLE_RELTYPES: Dict[str, str] = {
    "Scribe": "SCRIBE_OF",
    "Mentioned": "MENTIONED_IN",
    "Recipient": "RECIPIENT_OF",
    "Sender": "SENDER_OF",
    "Witness": "WITNESS_OF",
    "Party": "PARTY_TO",
    "Authority (including Reshut)": "AUTHORITY_OF",
    "Validating judge": "VALIDATING_JUDGE_OF",
    "Legal and state personnel": "LEGAL_PERSONNEL_IN",
    "Author": "AUTHOR_OF",
    "Reuser": "REUSER_OF",
    "Bearer": "BEARER_OF",
    "Payer": "PAYER_IN",
    "Marrying/Divorcing Party": "MARRIAGE_PARTY_TO",
    "Petitioner": "PETITIONER_OF",
}

QUALIFIER_FLAGS = {"(uncertain)": "uncertain", "(deceased)": "deceased"}


def parse_relation(relation: str) -> Tuple[str, bool, bool]:
    """Split a scraped relation string into base role and qualifier flags.

    Only known *trailing* qualifiers are stripped, so parenthesized text that
    is part of the role itself ("Authority (including Reshut)") survives.

    :param relation: Verbatim relation string, e.g. "Scribe (uncertain)".
    :returns: (base role, uncertain flag, deceased flag).
    :rtype: Tuple[str, bool, bool]
    """
    base = relation.strip()
    flags = {"uncertain": False, "deceased": False}
    stripped = True
    while stripped:
        stripped = False
        for suffix, flag in QUALIFIER_FLAGS.items():
            if base.endswith(suffix):
                base = base[: -len(suffix)].strip()
                flags[flag] = True
                stripped = True
    return base, flags["uncertain"], flags["deceased"]


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    """Read the scraped relations CSV.

    :param csv_path: Path to pgp_person_document_relations.csv.
    :returns: All rows as dicts keyed by the CSV header.
    :rtype: List[Dict[str, str]]
    """
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    logger.info("Read %d rows from %s", len(rows), csv_path)
    return rows


def prefetch_person_maps(
    driver: Driver, database: str
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Load Person lookup maps in one pass.

    :param driver: Open Neo4j driver.
    :param database: Target database name.
    :returns: (url -> elementId, name -> [elementId, ...]).
    :rtype: Tuple[Dict[str, str], Dict[str, List[str]]]
    """
    url_map: Dict[str, str] = {}
    name_map: Dict[str, List[str]] = defaultdict(list)
    records, _, _ = driver.execute_query(
        "MATCH (p:Person) RETURN elementId(p) AS eid, p.url AS url, p.name AS name",
        database_=database,
    )
    for record in records:
        if record["url"]:
            url_map[record["url"]] = record["eid"]
        if record["name"]:
            name_map[record["name"]].append(record["eid"])
    logger.info(
        "Prefetched %d Person nodes (%d with url)", len(records), len(url_map)
    )
    return url_map, name_map


def prefetch_fragment_maps(
    driver: Driver, database: str
) -> Tuple[Dict[int, str], Dict[str, List[str]]]:
    """Load Fragment lookup maps in one pass.

    :param driver: Open Neo4j driver.
    :param database: Target database name.
    :returns: (pgpid -> elementId, shelfmark -> [elementId, ...]).
    :rtype: Tuple[Dict[int, str], Dict[str, List[str]]]
    """
    pgpid_map: Dict[int, str] = {}
    shelfmark_map: Dict[str, List[str]] = defaultdict(list)
    records, _, _ = driver.execute_query(
        "MATCH (f:Fragment) RETURN elementId(f) AS eid, f.pgpid AS pgpid, "
        "f.shelfmark AS shelfmark",
        database_=database,
    )
    for record in records:
        if record["pgpid"] is not None:
            pgpid_map[int(record["pgpid"])] = record["eid"]
        if record["shelfmark"]:
            shelfmark_map[record["shelfmark"]].append(record["eid"])
    logger.info(
        "Prefetched %d Fragment nodes (%d with pgpid)", len(records), len(pgpid_map)
    )
    return pgpid_map, shelfmark_map


def match_person(
    row: Dict[str, str], url_map: Dict[str, str], name_map: Dict[str, List[str]]
) -> Tuple[Optional[str], str]:
    """Resolve a CSV row to an existing Person elementId.

    :param row: CSV row.
    :param url_map: Person url -> elementId.
    :param name_map: Person name -> [elementId, ...].
    :returns: (elementId or None, match path label for reporting).
    :rtype: Tuple[Optional[str], str]
    """
    eid = url_map.get(row["person_url"])
    if eid:
        return eid, "url"
    candidates = name_map.get(row["person_name"], [])
    if len(candidates) == 1:
        return candidates[0], "name"
    if len(candidates) > 1:
        return None, "ambiguous_name"
    return None, "no_person"


def match_fragments(
    row: Dict[str, str],
    pgpid_map: Dict[int, str],
    shelfmark_map: Dict[str, List[str]],
) -> Tuple[List[str], str]:
    """Resolve a CSV row to one or more Fragment elementIds.

    :param row: CSV row.
    :param pgpid_map: Fragment pgpid -> elementId.
    :param shelfmark_map: Fragment shelfmark -> [elementId, ...].
    :returns: (elementIds, match path label for reporting).
    :rtype: Tuple[List[str], str]
    """
    eid = pgpid_map.get(int(row["pgpid"]))
    if eid:
        return [eid], "pgpid"
    full = shelfmark_map.get(row["shelfmark"].strip(), [])
    if full:
        return full, "shelfmark"
    pieces: List[str] = []
    for mark in row["fragment_shelfmarks"].split(";"):
        pieces.extend(shelfmark_map.get(mark.strip(), []))
    if pieces:
        return sorted(set(pieces)), "fragment_shelfmarks"
    return [], "no_fragment"


def build_edges(
    rows: List[Dict[str, str]],
    url_map: Dict[str, str],
    name_map: Dict[str, List[str]],
    pgpid_map: Dict[int, str],
    shelfmark_map: Dict[str, List[str]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, str]], Counter]:
    """Match every CSV row and assemble edge payloads grouped by rel type.

    :param rows: CSV rows.
    :param url_map: Person url -> elementId.
    :param name_map: Person name -> [elementId, ...].
    :param pgpid_map: Fragment pgpid -> elementId.
    :param shelfmark_map: Fragment shelfmark -> [elementId, ...].
    :returns: (rel type -> edge rows, unmatched rows with reasons, stats).
    :rtype: Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, str]], Counter]
    """
    edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    unmatched: List[Dict[str, str]] = []
    stats: Counter = Counter()
    seen: set = set()
    matched_people: set = set()

    for row in rows:
        person_eid, person_path = match_person(row, url_map, name_map)
        fragment_eids, fragment_path = match_fragments(row, pgpid_map, shelfmark_map)
        stats[f"person:{person_path}"] += 1
        stats[f"fragment:{fragment_path}"] += 1

        if person_eid is None or not fragment_eids:
            unmatched.append(
                {**row, "person_match": person_path, "fragment_match": fragment_path}
            )
            continue

        matched_people.add(person_eid)
        base, uncertain, deceased = parse_relation(row["relation"])
        rel_type = ROLE_RELTYPES.get(base)
        if rel_type is None:
            logger.warning("Unmapped relation %r; skipping row", row["relation"])
            unmatched.append(
                {**row, "person_match": "unmapped_relation", "fragment_match": fragment_path}
            )
            continue

        for fragment_eid in fragment_eids:
            key = (person_eid, fragment_eid, rel_type, row["relation"])
            if key in seen:
                continue
            seen.add(key)
            edges[rel_type].append(
                {
                    "pid": person_eid,
                    "fid": fragment_eid,
                    "pgpid": int(row["pgpid"]),
                    "relation": row["relation"],
                    "uncertain": uncertain,
                    "deceased": deceased,
                }
            )

    stats["distinct_people_matched"] = len(matched_people)
    return edges, unmatched, stats


def write_edges(
    driver: Driver,
    database: str,
    edges: Dict[str, List[Dict[str, Any]]],
    batch_size: int,
) -> Tuple[int, int]:
    """MERGE all edge payloads into the graph.

    :param driver: Open Neo4j driver.
    :param database: Target database name.
    :param edges: Rel type -> edge rows from :func:`build_edges`.
    :param batch_size: Rows per UNWIND transaction.
    :returns: (relationships created, edge rows processed).
    :rtype: Tuple[int, int]
    """
    created = 0
    processed = 0
    for rel_type, rel_rows in edges.items():
        query = (
            "UNWIND $rows AS row "
            "MATCH (p) WHERE elementId(p) = row.pid "
            "MATCH (f) WHERE elementId(f) = row.fid "
            f"MERGE (p)-[r:{rel_type} {{pgpid: row.pgpid, relation: row.relation}}]->(f) "
            "ON CREATE SET r.uncertain = row.uncertain, "
            "              r.deceased = row.deceased, "
            f"              r.source = '{SOURCE_TAG}', "
            f"              r.data_sources = ['{DATA_SOURCE_TOKEN}']"
        )
        for start in tqdm(
            range(0, len(rel_rows), batch_size), desc=rel_type, unit="batch"
        ):
            batch = rel_rows[start : start + batch_size]
            _, summary, _ = driver.execute_query(
                query, rows=batch, database_=database
            )
            created += summary.counters.relationships_created
            processed += len(batch)
    return created, processed


def count_people_without_fragment(driver: Driver, database: str) -> int:
    """Count Person nodes with no direct edge to any Fragment.

    :param driver: Open Neo4j driver.
    :param database: Target database name.
    :returns: Number of unlinked Person nodes.
    :rtype: int
    """
    records, _, _ = driver.execute_query(
        "MATCH (p:Person) WHERE NOT (p)--(:Fragment) RETURN count(p) AS n",
        database_=database,
    )
    return records[0]["n"]


def count_import_edges(driver: Driver, database: str) -> List[Tuple[str, int]]:
    """Count edges written by this import, per relationship type.

    :param driver: Open Neo4j driver.
    :param database: Target database name.
    :returns: (rel type, count) pairs, descending.
    :rtype: List[Tuple[str, int]]
    """
    records, _, _ = driver.execute_query(
        "MATCH (:Person)-[r]->(:Fragment) WHERE r.source = $source "
        "RETURN type(r) AS t, count(r) AS n ORDER BY n DESC",
        source=SOURCE_TAG,
        database_=database,
    )
    return [(record["t"], record["n"]) for record in records]


def main() -> None:
    """Run the import end to end and print the verification report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--dry-run", action="store_true", help="Match and report only")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    uri = os.environ["NEO4J_URI"]
    database = os.environ.get("NEO4J_DATABASE", "neo4j")
    auth = (os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])

    rows = read_rows(args.csv)
    driver = GraphDatabase.driver(uri, auth=auth)
    try:
        driver.verify_connectivity()
        before_unlinked = count_people_without_fragment(driver, database)
        logger.info("Person nodes with no Fragment edge (before): %d", before_unlinked)

        url_map, name_map = prefetch_person_maps(driver, database)
        pgpid_map, shelfmark_map = prefetch_fragment_maps(driver, database)
        edges, unmatched, stats = build_edges(
            rows, url_map, name_map, pgpid_map, shelfmark_map
        )

        total_edge_rows = sum(len(v) for v in edges.values())
        print("\n=== Matching report ===")
        for key in sorted(stats):
            print(f"  {key}: {stats[key]}")
        print(f"  edge rows to write: {total_edge_rows}")
        print(f"  unmatched rows: {len(unmatched)}")

        if unmatched:
            UNMATCHED_REPORT.parent.mkdir(parents=True, exist_ok=True)
            with open(UNMATCHED_REPORT, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(unmatched[0].keys()))
                writer.writeheader()
                writer.writerows(unmatched)
            print(f"  unmatched rows written to {UNMATCHED_REPORT}")

        if args.dry_run:
            print("\nDry run: nothing written.")
            return

        created, processed = write_edges(driver, database, edges, args.batch_size)
        after_unlinked = count_people_without_fragment(driver, database)

        print("\n=== Load report ===")
        print(f"  edge rows processed: {processed}")
        print(f"  relationships created this run: {created}")
        print("  edges in graph from this import, by type:")
        for rel_type, count in count_import_edges(driver, database):
            print(f"    {rel_type}: {count}")
        print(f"  Person nodes with no Fragment edge: {before_unlinked} -> {after_unlinked}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
