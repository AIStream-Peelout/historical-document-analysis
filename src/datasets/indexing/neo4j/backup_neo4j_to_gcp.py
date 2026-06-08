#!/usr/bin/env python3
"""
Neo4j → Google Cloud Storage backup script.

Performs a logical backup of the Neo4j knowledge graph by streaming all nodes
and relationships out through the bolt driver, writing them as compressed JSONL
files, and uploading the resulting archive to a GCS bucket.

Because Neo4j is accessed via Docker port (not the host CLI), this approach
avoids any dependency on ``neo4j-admin dump`` and works purely over bolt.

Backup layout inside the archive
---------------------------------
backup_<database>_<timestamp>/
    meta.json                   — database stats and backup metadata
    nodes/
        <Label>.jsonl           — one JSON object per line, one file per label
    relationships/
        <TYPE>.jsonl            — one JSON object per line, one file per rel-type

Each JSONL record for a node contains::

    {"_id": <neo4j_internal_id>, "_labels": [...], <property_key>: <value>, ...}

Each JSONL record for a relationship contains::

    {"_id": ..., "_type": "...", "_start_id": ..., "_end_id": ..., <props>}

This format is self-contained and sufficient to fully rebuild the graph with
the import scripts (or a custom restore script).

Usage
-----
python backup_neo4j_to_gcp.py                   # backup current NEO4J_DATABASE
python backup_neo4j_to_gcp.py --dry-run         # export locally, skip GCS upload
python backup_neo4j_to_gcp.py --out-dir /tmp    # write archive to a custom local dir

Environment variables (.env)
-----------------------------
NEO4J_URI               bolt://127.0.0.1:7681
NEO4J_USER              neo4j
NEO4J_PASSWORD          <password>
NEO4J_DATABASE          genizah-prod
GOOGLE_APPLICATION_CREDENTIALS   /path/to/service_account.json
GOOGLE_BACKUP_BUCKET1    my-gcs-bucket-name

Requirements
------------
pip install neo4j google-cloud-storage python-dotenv
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dotenv

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))
dotenv.load_dotenv(project_root / ".env")

from neo4j import GraphDatabase  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _record_to_dict(record_value: Any) -> Any:
    """Recursively convert neo4j types to plain Python types for JSON serialisation.

    :param record_value: A value returned from a Neo4j record field.
    :returns: JSON-serialisable Python object.
    """
    # neo4j Node / Relationship objects expose .items(), element_id, labels, type
    if hasattr(record_value, "items"):
        return {k: _record_to_dict(v) for k, v in record_value.items()}
    if isinstance(record_value, list):
        return [_record_to_dict(v) for v in record_value]
    if isinstance(record_value, dict):
        return {k: _record_to_dict(v) for k, v in record_value.items()}
    # Temporal / spatial types → string representation
    if hasattr(record_value, "iso_format"):
        return record_value.iso_format()
    if hasattr(record_value, "__str__") and type(record_value).__module__.startswith("neo4j"):
        return str(record_value)
    return record_value


def export_nodes(session, label: str, out_path: Path) -> int:
    """Stream all nodes with *label* to a gzip-compressed JSONL file.

    :param session: Active Neo4j session.
    :param label: Node label to export (e.g. ``"Fragment"``).
    :param out_path: Destination ``.jsonl.gz`` file path.
    :returns: Number of nodes written.
    """
    count = 0
    with gzip.open(out_path, "wt", encoding="utf-8") as fh:
        result = session.run(
            f"MATCH (n:`{label}`) "
            "RETURN id(n) AS _id, labels(n) AS _labels, properties(n) AS props"
        )
        for record in result:
            row: dict[str, Any] = {
                "_id":     record["_id"],
                "_labels": record["_labels"],
            }
            row.update(_record_to_dict(record["props"]))
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def export_relationships(session, rel_type: str, out_path: Path) -> int:
    """Stream all relationships of *rel_type* to a gzip-compressed JSONL file.

    :param session: Active Neo4j session.
    :param rel_type: Relationship type to export (e.g. ``"HELD_AT"``).
    :param out_path: Destination ``.jsonl.gz`` file path.
    :returns: Number of relationships written.
    """
    count = 0
    with gzip.open(out_path, "wt", encoding="utf-8") as fh:
        result = session.run(
            f"MATCH (a)-[r:`{rel_type}`]->(b) "
            "RETURN id(r) AS _id, type(r) AS _type, "
            "       id(a) AS _start_id, id(b) AS _end_id, "
            "       properties(r) AS props"
        )
        for record in result:
            row: dict[str, Any] = {
                "_id":       record["_id"],
                "_type":     record["_type"],
                "_start_id": record["_start_id"],
                "_end_id":   record["_end_id"],
            }
            row.update(_record_to_dict(record["props"]))
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def get_schema(session) -> tuple[list[str], list[str]]:
    """Return (node_labels, relationship_types) present in the database.

    :param session: Active Neo4j session.
    :returns: Tuple of (sorted list of labels, sorted list of rel-types).
    """
    labels = sorted(
        r["label"] for r in session.run("CALL db.labels() YIELD label")
    )
    rel_types = sorted(
        r["relationshipType"]
        for r in session.run("CALL db.relationshipTypes() YIELD relationshipType")
    )
    return labels, rel_types


def collect_stats(session) -> dict[str, Any]:
    """Return node and relationship counts per type for the metadata file.

    :param session: Active Neo4j session.
    :returns: Dict with ``nodes`` and ``relationships`` count dicts.
    """
    node_counts = {
        r["label"]: r["count"]
        for r in session.run(
            "CALL db.labels() YIELD label "
            "CALL apoc.cypher.run('MATCH (n:`' + label + '`) RETURN count(n) AS count', {}) "
            "YIELD value RETURN label, value.count AS count"
        )
    } if False else {}  # APOC may not be available; fall back below

    # Safe fallback without APOC
    labels, rel_types = get_schema(session)
    for label in labels:
        result = session.run(f"MATCH (n:`{label}`) RETURN count(n) AS c")
        node_counts[label] = result.single()["c"]

    rel_counts: dict[str, int] = {}
    for rtype in rel_types:
        result = session.run(f"MATCH ()-[r:`{rtype}`]->() RETURN count(r) AS c")
        rel_counts[rtype] = result.single()["c"]

    return {"nodes": node_counts, "relationships": rel_counts}


# ---------------------------------------------------------------------------
# Archive + upload
# ---------------------------------------------------------------------------

def build_archive(export_dir: Path, archive_path: Path) -> None:
    """Create a .tar.gz archive from *export_dir* at *archive_path*.

    :param export_dir: Directory containing the exported files.
    :param archive_path: Destination archive file path.
    """
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(export_dir, arcname=export_dir.name)
    size_mb = archive_path.stat().st_size / 1_048_576
    logger.info(f"Archive created: {archive_path.name} ({size_mb:.1f} MB)")


def upload_to_gcs(archive_path: Path, bucket_name: str, database: str) -> str:
    """Upload *archive_path* to GCS under a dated prefix.

    :param archive_path: Local path to the archive file.
    :param bucket_name: GCS bucket name (without ``gs://``).
    :param database: Neo4j database name, used to organise backup prefixes.
    :returns: Full GCS URI of the uploaded object.
    """
    from google.cloud import storage  # import here so --dry-run doesn't need it

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    gcs_key = f"neo4j-backups/{database}/{archive_path.name}"
    blob = bucket.blob(gcs_key)

    logger.info(f"Uploading to gs://{bucket_name}/{gcs_key} ...")
    blob.upload_from_filename(str(archive_path))

    uri = f"gs://{bucket_name}/{gcs_key}"
    logger.info(f"Upload complete → {uri}")
    return uri


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Logical Neo4j backup to Google Cloud Storage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Export locally and skip GCS upload.",
    )
    parser.add_argument(
        "--out-dir", metavar="DIR", default=None,
        help="Write the archive to this local directory (default: system temp).",
    )
    args = parser.parse_args()

    # ── Config from environment ───────────────────────────────────────────────
    neo4j_uri      = os.getenv("NEO4J_URI",      "bolt://127.0.0.1:7687")
    neo4j_user     = os.getenv("NEO4J_USER",     "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
    bucket_name    = os.getenv("GOOGLE_BACKUP_BUCKET1")
    gcp_creds      = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not neo4j_password:
        logger.error("NEO4J_PASSWORD is not set")
        sys.exit(1)
    if not args.dry_run and not bucket_name:
        logger.error("GOOGLE_BACKUP_BUCKET1 is not set (use --dry-run to export locally)")
        sys.exit(1)
    if not args.dry_run and not gcp_creds:
        logger.error("GOOGLE_APPLICATION_CREDENTIALS is not set")
        sys.exit(1)

    timestamp  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_name = f"backup_{neo4j_database}_{timestamp}"

    # Work inside a temp dir; if --out-dir is specified use that instead.
    out_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp())
    out_dir.mkdir(parents=True, exist_ok=True)

    export_dir = out_dir / backup_name
    nodes_dir  = export_dir / "nodes"
    rels_dir   = export_dir / "relationships"
    nodes_dir.mkdir(parents=True)
    rels_dir.mkdir(parents=True)

    logger.info(f"Starting backup of '{neo4j_database}' → {backup_name}")

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    try:
        with driver.session(database=neo4j_database) as session:

            # ── Schema discovery ──────────────────────────────────────────────
            labels, rel_types = get_schema(session)
            logger.info(f"Found {len(labels)} node labels, {len(rel_types)} relationship types")

            # ── Stats for metadata ────────────────────────────────────────────
            logger.info("Collecting database statistics ...")
            stats = collect_stats(session)
            total_nodes = sum(stats["nodes"].values())
            total_rels  = sum(stats["relationships"].values())
            logger.info(f"  Nodes: {total_nodes:,}  |  Relationships: {total_rels:,}")

            # ── Export nodes ──────────────────────────────────────────────────
            logger.info("Exporting nodes ...")
            node_summary: dict[str, int] = {}
            for label in labels:
                out_path = nodes_dir / f"{label}.jsonl.gz"
                count = export_nodes(session, label, out_path)
                node_summary[label] = count
                logger.info(f"  {label}: {count:,} nodes")

            # ── Export relationships ──────────────────────────────────────────
            logger.info("Exporting relationships ...")
            rel_summary: dict[str, int] = {}
            for rtype in rel_types:
                out_path = rels_dir / f"{rtype}.jsonl.gz"
                count = export_relationships(session, rtype, out_path)
                rel_summary[rtype] = count
                logger.info(f"  {rtype}: {count:,} edges")

        # ── Write metadata ────────────────────────────────────────────────────
        meta = {
            "backup_name":  backup_name,
            "database":     neo4j_database,
            "neo4j_uri":    neo4j_uri,
            "timestamp":    timestamp,
            "node_labels":  labels,
            "rel_types":    rel_types,
            "counts": {
                "nodes":         node_summary,
                "relationships": rel_summary,
                "total_nodes":   sum(node_summary.values()),
                "total_rels":    sum(rel_summary.values()),
            },
        }
        meta_path = export_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        logger.info(f"Metadata written to {meta_path}")

        # ── Build archive ─────────────────────────────────────────────────────
        archive_path = out_dir / f"{backup_name}.tar.gz"
        build_archive(export_dir, archive_path)

        # ── Upload ────────────────────────────────────────────────────────────
        if args.dry_run:
            logger.info(f"[dry-run] Skipping GCS upload. Archive at: {archive_path}")
        else:
            uri = upload_to_gcs(archive_path, bucket_name, neo4j_database)
            logger.info(f"Backup complete → {uri}")

        # ── Summary ───────────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"Backup: {backup_name}")
        print(f"  Total nodes:         {sum(node_summary.values()):,}")
        print(f"  Total relationships: {sum(rel_summary.values()):,}")
        print(f"  Archive:             {archive_path}")
        if not args.dry_run:
            print(f"  GCS:                 gs://{bucket_name}/neo4j-backups/{neo4j_database}/{archive_path.name}")
        print(f"{'='*60}\n")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
