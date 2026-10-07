#!/usr/bin/env python3
"""
Backfill ``Fragment.es_doc_id`` — the Neo4j → Elasticsearch join key
=====================================================================

The serving backend (``genizah_search/src/backend/neo4j_service.py``) reads
``f.es_doc_id`` to jump from a KG Fragment to its ES document (map view,
scholarship-on-a-fragment, RAG linking). The Studio's production graph
carries it on ~44.9k of 49k Fragments, but the pass that wrote it was a
one-off session in July 2026 and was never committed — so a rebuilt graph
(the MBP v3 import) has none until this script runs.

How the id is derived
---------------------
``merged_shelfmarks.jsonl`` is the source the ES ``genizah_merged_*``
indexes are built from, and each record's ``canonical_id`` **is** the ES
document ``_id``. A Fragment resolves to a merged record by, in order:

1. the canonical shelfmark of the **first component** of a join
   (``T_S_20_169_+_T_S_10J8_9`` → ``T_S_20_169``) — this is what the July
   pass did for multi-fragment joins (reproduces 97.7 % of its values);
2. ``Fragment.pgpid`` → a merged record listing that pgpid;
3. the full canonical shelfmark.

Every candidate is then checked against the **served** ES index (an ``ids``
query, batched), and the first one that exists wins. The merged JSONL is
rebuilt over time and its id shapes drift (``Oxford_Bodleian_MS_heb_…`` vs
``Oxford_Bodleian_Bodl_MS_heb_…``), so an unverified id could point at a
document the site cannot open; ``--no-verify`` skips the check for offline
runs.

Idempotent: only Fragments with ``es_doc_id IS NULL`` are written unless
``--overwrite`` is given. Also creates the ``fragment_es_doc_id`` index the
backend's fragment-context lookup needs (KG_ENHANCEMENTS_TODO #11).

Usage
-----
::

    PYTHONPATH=. .venv/bin/python -m src.datasets.indexing.neo4j.backfill_es_doc_id --dry-run
    PYTHONPATH=. .venv/bin/python -m src.datasets.indexing.neo4j.backfill_es_doc_id
    # against a specific ES / index
    ... --es-host localhost --es-port 9200 --es-scheme http --es-index genizah_merged_v5

Reads ``NEO4J_URI`` / ``NEO4J_USER`` / ``NEO4J_PASSWORD`` / ``NEO4J_DATABASE``
and ``ELASTIC_USER`` / ``ELASTIC_PASSWORD`` from the environment (``.env``
fallback; an exported variable wins).
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import dotenv
from neo4j import GraphDatabase
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[4]
sys.path.append(str(_REPO))
dotenv.load_dotenv(_REPO / ".env")

from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer  # noqa: E402

logger = logging.getLogger(__name__)

_MERGED = _REPO / "src" / "datasets" / "raw_data" / "cairo_genizah" / "merged" / "merged_shelfmarks.jsonl"
DEFAULT_ES_INDEX = "genizah_merged_v5"
JOIN_SEPARATOR = "_+_"
WRITE_BATCH = 1000
ES_BATCH = 500


# ---------------------------------------------------------------------------
# Merged-record lookup tables
# ---------------------------------------------------------------------------

def load_merged_maps(path: Path = _MERGED) -> Tuple[Dict[int, str], Dict[str, str]]:
    """Build ``pgpid → canonical_id`` and ``canonical_shelfmark → canonical_id``.

    First occurrence wins on both keys; the JSONL is deterministic, so the
    result is stable between runs.

    :param path: ``merged_shelfmarks.jsonl``.
    :returns: ``(by_pgpid, by_canonical_shelfmark)``.
    """
    by_pgpid: Dict[int, str] = {}
    by_canon: Dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in tqdm(fh, desc="merged_shelfmarks", unit=" rec"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = rec.get("canonical_id")
            if not cid:
                continue
            for p in rec.get("pgpids") or []:
                try:
                    by_pgpid.setdefault(int(p), cid)
                except (TypeError, ValueError):
                    continue
            display = (rec.get("shelfmark_display") or "").strip()
            canon = ShelfmarkNormalizer.to_canonical_id(display) if display else None
            if canon:
                by_canon.setdefault(canon, cid)
    return by_pgpid, by_canon


def candidate_ids(canonical_shelfmark: Optional[str], pgpid: Optional[int],
                  by_pgpid: Dict[int, str], by_canon: Dict[str, str]) -> List[str]:
    """Ordered, de-duplicated ES ids a Fragment might resolve to.

    :param canonical_shelfmark: ``Fragment.canonical_shelfmark``.
    :param pgpid: ``Fragment.pgpid`` (may be ``None``).
    :param by_pgpid: From :func:`load_merged_maps`.
    :param by_canon: From :func:`load_merged_maps`.
    :returns: Candidates, best first (first join component, pgpid, full mark).
    """
    canon = canonical_shelfmark or ""
    first = canon.split(JOIN_SEPARATOR)[0] if canon else ""
    raw = [
        by_canon.get(first) if first else None,
        by_pgpid.get(int(pgpid)) if pgpid is not None else None,
        by_canon.get(canon) if canon else None,
    ]
    out: List[str] = []
    for c in raw:
        if c and c not in out:
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# ES verification
# ---------------------------------------------------------------------------

def es_existing_ids(es: Any, index: str, ids: Iterable[str], batch: int = ES_BATCH) -> Set[str]:
    """Return the subset of *ids* that exist as ``_id`` in *index*.

    :param es: An :class:`elasticsearch.Elasticsearch` client.
    :param index: Index (or alias) name.
    :param ids: Candidate document ids.
    :param batch: Ids per ``ids`` query.
    :returns: Ids present in the index.
    """
    wanted = sorted(set(ids))
    found: Set[str] = set()
    for i in tqdm(range(0, len(wanted), batch), desc="verify ES ids", unit=" batch"):
        chunk = wanted[i:i + batch]
        resp = es.search(index=index, size=len(chunk), _source=False,
                         query={"ids": {"values": chunk}})
        found.update(h["_id"] for h in resp["hits"]["hits"])
    return found


def resolve(fragments: Sequence[Dict[str, Any]], by_pgpid: Dict[int, str],
            by_canon: Dict[str, str], existing: Optional[Set[str]],
            overwrite: bool) -> Tuple[List[Dict[str, str]], Counter]:
    """Decide the ``es_doc_id`` to write for each Fragment.

    :param fragments: Rows with ``c`` (canonical_shelfmark), ``p`` (pgpid),
        ``e`` (current es_doc_id).
    :param by_pgpid: From :func:`load_merged_maps`.
    :param by_canon: From :func:`load_merged_maps`.
    :param existing: Ids known to exist in ES, or ``None`` to skip verification.
    :param overwrite: Replace an existing ``es_doc_id`` when it differs.
    :returns: ``(rows_to_write, stats)`` where rows are ``{"c": …, "e": …}``.
    """
    rows: List[Dict[str, str]] = []
    stats: Counter = Counter()
    for f in fragments:
        cands = candidate_ids(f.get("c"), f.get("p"), by_pgpid, by_canon)
        if existing is not None:
            dropped = [c for c in cands if c not in existing]
            if dropped:
                stats["candidates_not_in_es"] += len(dropped)
            cands = [c for c in cands if c in existing]
        chosen = cands[0] if cands else None
        current = f.get("e")
        if chosen is None:
            stats["unresolved_has_value" if current else "unresolved"] += 1
        elif current == chosen:
            stats["already_set_same"] += 1
        elif current and not overwrite:
            stats["already_set_different_kept"] += 1
        else:
            stats["overwritten" if current else "set"] += 1
            rows.append({"c": f["c"], "e": chosen})
    return rows, stats


# ---------------------------------------------------------------------------
# Neo4j I/O
# ---------------------------------------------------------------------------

def fetch_fragments(driver: Any, database: str) -> List[Dict[str, Any]]:
    """Read every Fragment's join keys.

    :param driver: Neo4j driver.
    :param database: Database name.
    :returns: Rows with ``c``, ``p``, ``e``.
    """
    with driver.session(database=database) as s:
        return s.run(
            "MATCH (f:Fragment) RETURN f.canonical_shelfmark AS c, f.pgpid AS p, f.es_doc_id AS e"
        ).data()


def _write_rows(tx: Any, rows: List[Dict[str, str]]) -> None:
    tx.run(
        "UNWIND $rows AS r MATCH (f:Fragment {canonical_shelfmark: r.c}) SET f.es_doc_id = r.e",
        rows=rows,
    )


def write_es_doc_ids(driver: Any, database: str, rows: List[Dict[str, str]],
                     batch: int = WRITE_BATCH) -> None:
    """Write ``es_doc_id`` in batches and ensure the lookup index exists.

    :param driver: Neo4j driver.
    :param database: Database name.
    :param rows: ``{"c": canonical_shelfmark, "e": es_doc_id}`` rows.
    :param batch: Rows per transaction.
    """
    with driver.session(database=database) as s:
        s.run("CREATE INDEX fragment_es_doc_id IF NOT EXISTS FOR (f:Fragment) ON (f.es_doc_id)")
        for i in tqdm(range(0, len(rows), batch), desc="write es_doc_id", unit=" batch"):
            s.execute_write(_write_rows, rows[i:i + batch])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", "-n", action="store_true", help="Resolve and report, write nothing.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace es_doc_id values that differ from the resolved id.")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the ES existence check (offline).")
    parser.add_argument("--merged", type=Path, default=_MERGED, help="merged_shelfmarks.jsonl path.")
    parser.add_argument("--es-index", default=os.environ.get("ELASTICSEARCH_INDEX", DEFAULT_ES_INDEX),
                        help=f"Served index to verify ids against (default {DEFAULT_ES_INDEX}).")
    parser.add_argument("--es-host", default="localhost", help="ES host (default: local Docker)")
    parser.add_argument("--es-port", default="9200", help="ES port")
    parser.add_argument("--es-scheme", default="http", help="ES scheme")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger("elastic_transport").setLevel(logging.WARNING)  # one line per ids query otherwise

    uri = os.environ["NEO4J_URI"]
    database = os.environ.get("NEO4J_DATABASE", "neo4j")
    driver = GraphDatabase.driver(uri, auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))
    logger.info("Neo4j %s db=%s", uri, database)

    by_pgpid, by_canon = load_merged_maps(args.merged)
    logger.info("merged maps: %d pgpids, %d canonical shelfmarks", len(by_pgpid), len(by_canon))

    fragments = fetch_fragments(driver, database)
    logger.info("Fragments in graph: %d (with es_doc_id: %d)",
                len(fragments), sum(1 for f in fragments if f.get("e")))

    existing: Optional[Set[str]] = None
    if not args.no_verify:
        os.environ["ELASTIC_SEARCH_HOST"] = args.es_host
        os.environ["ELASTIC_SEARCH_PORT"] = args.es_port
        os.environ["ELASTIC_SEARCH_SCHEME"] = args.es_scheme
        from elasticsearch import Elasticsearch
        from src.datasets.indexing.elastic_index_genizah import es_config_from_env
        es = Elasticsearch(**es_config_from_env(), request_timeout=60, max_retries=1)
        all_cands = {c for f in fragments
                     for c in candidate_ids(f.get("c"), f.get("p"), by_pgpid, by_canon)}
        existing = es_existing_ids(es, args.es_index, all_cands)
        logger.info("ES %s: %d of %d candidate ids exist", args.es_index, len(existing), len(all_cands))

    rows, stats = resolve(fragments, by_pgpid, by_canon, existing, args.overwrite)
    tag = "[DRY RUN] " if args.dry_run else ""
    print(f"\n{tag}es_doc_id backfill\n{'=' * 50}")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]:,}")
    print(f"  rows to write: {len(rows):,}\n{'=' * 50}")

    if not args.dry_run and rows:
        write_es_doc_ids(driver, database, rows)
        after = sum(1 for f in fetch_fragments(driver, database) if f.get("e"))
        print(f"  Fragments with es_doc_id after write: {after:,}")
    driver.close()


if __name__ == "__main__":
    main()
