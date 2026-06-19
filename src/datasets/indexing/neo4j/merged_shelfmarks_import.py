#!/usr/bin/env python3
"""
Merged-shelfmarks KG import — enrich the graph from merged_shelfmarks.jsonl.

The merged corpus (PGP+FJP+KTIV, ~67k fragments) carries far richer per-fragment
catalog data than biblio.json. This importer surfaces the *relational* and
*descriptive* parts into Neo4j for the rich subset only (not the ~30k bare
fragments), keyed the same way as the rest of the KG so it merges with existing
Fragment nodes rather than duplicating them.

Imported (per the chosen scope):
  * FJP ``related_people``  → Person + ``Fragment -MENTIONS_PERSON-> Person``
                              (role + uncertainty carried on the edge)
  * FJP ``related_places``  → Place  + ``Fragment -MENTIONS_PLACE->  Place``
  * date / language / description → Fragment properties (fill-if-empty)
  * KTIV scholarly catalog  → Fragment properties (genres, catalog title,
                              scholarly_entry_count)
  * Institution / collection → ``Fragment -HELD_AT-> Institution``

Coverage gate: a record is imported only if it is *rich* (has related people/
places, KTIV scholarly entries, or a real description) OR its Fragment already
exists in the KG (so we enrich it).

Fragment key: ``ShelfmarkNormalizer.to_canonical_id(shelfmark_display)`` — the
same core form the PGP / biblio / academic imports use, so nodes merge.

Run::

    python -m src.datasets.indexing.neo4j.merged_shelfmarks_import
    python -m src.datasets.indexing.neo4j.merged_shelfmarks_import --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import dotenv
from neo4j import GraphDatabase
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO))
dotenv.load_dotenv(_REPO / ".env")

from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer  # noqa: E402
from src.datasets.document_models.person_normalizer import PersonNormalizer  # noqa: E402
from src.datasets.document_models.place_normalizer import PlaceNormalizer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_MERGED = _REPO / "src" / "datasets" / "raw_data" / "cairo_genizah" / "merged" / "merged_shelfmarks.jsonl"

# A description shorter than this (or these placeholders) is not "rich".
_MIN_DESC = 8
_EMPTY_DESCR = {"", "unknown", "piyyut"}  # 'Piyyut' alone is a bare genre label


# ---------------------------------------------------------------------------
# Pure extraction (no Neo4j) — testable offline
# ---------------------------------------------------------------------------

def _fjp_records(record: Dict) -> List[Dict]:
    return (record.get("sources") or {}).get("fjp") or []


def _ktiv_record(record: Dict) -> Optional[Dict]:
    return (record.get("sources") or {}).get("ktiv")


def is_rich(record: Dict) -> bool:
    """Return True if a merged record carries KG-worthy enrichment.

    :param record: One merged_shelfmarks.jsonl record.
    :returns: True if it has related people/places, KTIV scholarly entries, or
        a substantive description.
    """
    for fr in _fjp_records(record):
        if fr.get("related_people") or fr.get("related_places"):
            return True
    ktiv = _ktiv_record(record)
    if ktiv and ktiv.get("scholarly_entries"):
        return True
    desc = (record.get("description") or "").strip().lower()
    return len(desc) >= _MIN_DESC and desc not in _EMPTY_DESCR


def extract_record(record: Dict) -> Optional[Dict]:
    """Extract the KG-relevant payload from a merged record.

    :param record: One merged_shelfmarks.jsonl record.
    :returns: A dict with canonical_shelfmark, display, institution metadata,
        properties (date/language/description/genres/catalog), and deduped
        people/places — or ``None`` if it has no usable shelfmark.
    """
    display = (record.get("shelfmark_display") or "").strip()
    canonical = ShelfmarkNormalizer.to_canonical_id(display) if display else ""
    if not canonical:
        return None

    inst = ShelfmarkNormalizer.get_institution_info(display)
    fjps = _fjp_records(record)
    ktiv = _ktiv_record(record)

    # date / language / description (FJP preferred, KTIV catalog fallback)
    date = next((fr.get("date", {}).get("standard_date") or fr.get("date", {}).get("hebrew_date")
                 for fr in fjps if (fr.get("date") or {})), None)
    language = next((fr.get("language") for fr in fjps
                     if fr.get("language") and fr.get("language") != "Unknown"), None)
    description = (record.get("description") or "").strip() or None

    # people / places, deduped by normalized name
    people: Dict[str, Dict] = {}
    places: Dict[str, Dict] = {}
    for fr in fjps:
        for p in (fr.get("related_people") or []):
            raw = (p.get("name") or "").strip()
            if not raw:
                continue
            name = PersonNormalizer.normalize(raw)
            people.setdefault(name, {"name": name,
                                     "role": (p.get("role") or "").strip(),
                                     "uncertain": bool(p.get("uncertain"))})
        for pl in (fr.get("related_places") or []):
            raw = (pl.get("name") or "").strip()
            if not raw:
                continue
            name = PlaceNormalizer.normalize(raw)
            places.setdefault(name, {"name": name})

    # KTIV scholarly catalog metadata
    genres: List[str] = []
    catalog_title = None
    scholarly_count = 0
    if ktiv:
        scholarly_count = int(ktiv.get("scholarly_entry_count") or 0)
        catalog_title = (ktiv.get("basic_catalog") or {}).get("title")
        for se in (ktiv.get("scholarly_entries") or []):
            for sub in (se.get("subsections") or {}).values():
                dom = (sub.get("domain") or "").strip()
                if dom and dom not in genres:
                    genres.append(dom)

    return {
        "canonical_shelfmark": canonical,
        "display_shelfmark": display,
        "institution": inst.get("institution"),
        "collection": inst.get("collection"),
        "subcollection": inst.get("subcollection"),
        "date": date,
        "language": language,
        "description": description,
        "genres": genres,
        "catalog_title": catalog_title,
        "scholarly_entry_count": scholarly_count,
        "pgpids": record.get("pgpids") or [],
        "people": list(people.values()),
        "places": list(places.values()),
    }


# ---------------------------------------------------------------------------
# Neo4j importer
# ---------------------------------------------------------------------------

class MergedShelfmarkImporter:
    """Enrich the KG from merged_shelfmarks.jsonl (rich + already-present subset)."""

    BATCH = 200

    def __init__(self, uri: str, user: str, password: str, database: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self) -> None:
        self.driver.close()

    def _existing_fragment_keys(self) -> Set[str]:
        """Return the set of canonical_shelfmark values already in the KG.

        :returns: Set of existing Fragment.canonical_shelfmark strings.
        """
        with self.driver.session(database=self.database) as s:
            res = s.run("MATCH (f:Fragment) RETURN f.canonical_shelfmark AS c")
            return {r["c"] for r in res if r["c"]}

    def import_file(self, path: Path, dry_run: bool = False) -> Dict[str, int]:
        """Import the rich/known subset of *path* into Neo4j.

        :param path: merged_shelfmarks.jsonl path.
        :param dry_run: Extract + gate but do not write.
        :returns: Counts dict.
        """
        existing = set() if dry_run else self._existing_fragment_keys()
        if not dry_run:
            logger.info(f"  {len(existing):,} existing Fragment nodes in KG")

        counts = {"scanned": 0, "imported": 0, "people_edges": 0, "place_edges": 0,
                  "skipped_bare": 0}
        batch: List[Dict] = []
        with open(path, encoding="utf-8") as fh:
            for line in tqdm(fh, desc="merged_shelfmarks"):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                counts["scanned"] += 1
                payload = extract_record(record)
                if payload is None:
                    continue
                gate = is_rich(record) or (payload["canonical_shelfmark"] in existing)
                if not gate:
                    counts["skipped_bare"] += 1
                    continue
                counts["imported"] += 1
                counts["people_edges"] += len(payload["people"])
                counts["place_edges"] += len(payload["places"])
                if dry_run:
                    continue
                batch.append(payload)
                if len(batch) >= self.BATCH:
                    self._write_batch(batch)
                    batch = []
        if batch and not dry_run:
            self._write_batch(batch)
        return counts

    def _write_batch(self, batch: List[Dict]) -> None:
        with self.driver.session(database=self.database) as s:
            s.execute_write(self._write, batch)

    @staticmethod
    def _write(tx, batch: List[Dict]) -> None:
        for r in batch:
            # Fragment (merge with existing; fill-if-empty for catalog props).
            tx.run(
                """
                MERGE (f:Fragment {canonical_shelfmark: $canonical})
                  ON CREATE SET f.shelfmark = $display
                SET f.data_sources = CASE WHEN 'merged' IN coalesce(f.data_sources,[])
                        THEN coalesce(f.data_sources,[]) ELSE coalesce(f.data_sources,[]) + ['merged'] END,
                    f.date        = coalesce(f.date, $date),
                    f.language    = coalesce(f.language, $language),
                    f.description = coalesce(f.description, $description),
                    f.catalog_title = coalesce(f.catalog_title, $catalog_title),
                    f.scholarly_entry_count = CASE WHEN $sec > 0 THEN $sec ELSE f.scholarly_entry_count END,
                    f.genres = CASE WHEN size($genres) > 0 THEN $genres ELSE f.genres END,
                    f.pgpids = CASE WHEN size($pgpids) > 0 THEN $pgpids ELSE f.pgpids END
                """,
                canonical=r["canonical_shelfmark"], display=r["display_shelfmark"],
                date=r["date"], language=r["language"], description=r["description"],
                catalog_title=r["catalog_title"], sec=r["scholarly_entry_count"],
                genres=r["genres"], pgpids=[str(p) for p in r["pgpids"]],
            )
            if r["institution"]:
                tx.run(
                    """
                    MATCH (f:Fragment {canonical_shelfmark: $canonical})
                    MERGE (i:Institution {name: $institution})
                      ON CREATE SET i.collection = $collection
                    MERGE (f)-[h:HELD_AT]->(i)
                    SET h.sub_collection = coalesce(h.sub_collection, $subcollection),
                        h.data_sources = CASE WHEN 'merged' IN coalesce(h.data_sources,[])
                            THEN coalesce(h.data_sources,[]) ELSE coalesce(h.data_sources,[]) + ['merged'] END
                    """,
                    canonical=r["canonical_shelfmark"], institution=r["institution"],
                    collection=r["collection"], subcollection=r["subcollection"],
                )
            for p in r["people"]:
                tx.run(
                    """
                    MATCH (f:Fragment {canonical_shelfmark: $canonical})
                    MERGE (p:Person {name: $name})
                    SET p.data_sources = CASE WHEN 'merged' IN coalesce(p.data_sources,[])
                            THEN coalesce(p.data_sources,[]) ELSE coalesce(p.data_sources,[]) + ['merged'] END
                    MERGE (f)-[m:MENTIONS_PERSON]->(p)
                    SET m.role = coalesce(m.role, $role),
                        m.uncertain = $uncertain,
                        m.data_sources = CASE WHEN 'merged' IN coalesce(m.data_sources,[])
                            THEN coalesce(m.data_sources,[]) ELSE coalesce(m.data_sources,[]) + ['merged'] END
                    """,
                    canonical=r["canonical_shelfmark"], name=p["name"],
                    role=p["role"] or None, uncertain=p["uncertain"],
                )
            for pl in r["places"]:
                tx.run(
                    """
                    MATCH (f:Fragment {canonical_shelfmark: $canonical})
                    MERGE (pl:Place {name: $name})
                    SET pl.data_sources = CASE WHEN 'merged' IN coalesce(pl.data_sources,[])
                            THEN coalesce(pl.data_sources,[]) ELSE coalesce(pl.data_sources,[]) + ['merged'] END
                    MERGE (f)-[m:MENTIONS_PLACE]->(pl)
                    SET m.data_sources = CASE WHEN 'merged' IN coalesce(m.data_sources,[])
                            THEN coalesce(m.data_sources,[]) ELSE coalesce(m.data_sources,[]) + ['merged'] END
                    """,
                    canonical=r["canonical_shelfmark"], name=pl["name"],
                )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Count without writing.")
    ap.add_argument("--file", default=str(_MERGED))
    args = ap.parse_args()

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pw = os.getenv("NEO4J_PASSWORD")
    db = os.getenv("NEO4J_DATABASE", "genizah-prod")
    if not pw and not args.dry_run:
        logger.error("NEO4J_PASSWORD not set")
        sys.exit(1)

    importer = MergedShelfmarkImporter(uri, user, pw or "", db)
    try:
        counts = importer.import_file(Path(args.file), dry_run=args.dry_run)
        print(json.dumps(counts, indent=2))
    finally:
        importer.close()


if __name__ == "__main__":
    main()
