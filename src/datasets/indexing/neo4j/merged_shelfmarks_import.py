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
from src.datasets.document_models.institution_normalizer import InstitutionNormalizer  # noqa: E402
from src.datasets.document_models.person_normalizer import PersonNormalizer  # noqa: E402
from src.datasets.document_models.place_normalizer import PlaceNormalizer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_MERGED = _REPO / "src" / "datasets" / "raw_data" / "cairo_genizah" / "merged" / "merged_shelfmarks.jsonl"



# ---------------------------------------------------------------------------
# Pure extraction (no Neo4j) — testable offline
# ---------------------------------------------------------------------------

def _as_text(value: Any) -> Optional[str]:
    """Coerce a KTIV catalog field (str or list of str) to a clean string.

    :param value: Raw field value (str, list, or None).
    :returns: Trimmed string (lists joined with "; "), or None if empty.
    """
    if isinstance(value, list):
        value = "; ".join(str(v).strip() for v in value if str(v).strip())
    s = (str(value).strip() if value is not None else "")
    return s or None


def _fjp_records(record: Dict) -> List[Dict]:
    return (record.get("sources") or {}).get("fjp") or []


def _ktiv_record(record: Dict) -> Optional[Dict]:
    return (record.get("sources") or {}).get("ktiv")


def is_rich(record: Dict) -> bool:
    """Return True if a merged record carries KG-worthy *relational* enrichment.

    Rich = related people/places or KTIV scholarly catalog entries. A bare
    description (often just a genre label like "Piyyut") is NOT rich on its
    own — those fragments are only imported if already present in the KG
    (where the description is attached as a property during enrichment).

    :param record: One merged_shelfmarks.jsonl record.
    :returns: True if the record warrants adding/relating in the KG.
    """
    for fr in _fjp_records(record):
        if fr.get("related_people") or fr.get("related_places"):
            return True
    ktiv = _ktiv_record(record)
    if not ktiv:
        return False
    fc = ktiv.get("full_catalog") or {}
    return bool(ktiv.get("scholarly_entries") or fc.get("subjects") or fc.get("notes"))


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
    subjects: List[str] = []
    notes = catalog_author = catalog_persons = paleographic = None
    if ktiv:
        scholarly_count = int(ktiv.get("scholarly_entry_count") or 0)
        bc = ktiv.get("basic_catalog") or {}
        fc = ktiv.get("full_catalog") or {}
        catalog_title = _as_text(bc.get("title"))
        catalog_author = _as_text(bc.get("author"))
        # subjects: drop the bare numeric NLI record-ids mixed into the list
        subjects = [s.strip() for s in (fc.get("subjects") or [])
                    if isinstance(s, str) and s.strip() and not s.strip().isdigit()]
        notes = _as_text(fc.get("notes"))
        paleographic = _as_text(fc.get("paleographic_note"))
        # additional_persons is a single undelimited string — keep as a
        # searchable property, do NOT split into (garbage) Person nodes.
        ap = _as_text(fc.get("additional_persons")) or ""
        psub = _as_text(fc.get("persons_as_subject")) or ""
        catalog_persons = "; ".join(p for p in (ap, psub) if p) or None
        for se in (ktiv.get("scholarly_entries") or []):
            for sub in (se.get("subsections") or {}).values():
                if not isinstance(sub, dict):
                    continue
                dom = sub.get("domain")
                # domain may be a string or a list of strings
                for d in (dom if isinstance(dom, list) else [dom]):
                    d = (d or "").strip() if isinstance(d, str) else ""
                    if d and d not in genres:
                        genres.append(d)

    # ShelfmarkNormalizer.INSTITUTION_MAPPING and InstitutionNormalizer's
    # canonical dict are separately maintained; normalising here is a safety
    # net so this pipeline's Institution nodes always converge with the ones
    # the LLM/enriched pipeline creates via InstitutionNormalizer, even if the
    # two tables ever drift apart.
    institution = inst.get("institution")
    if institution:
        institution = InstitutionNormalizer.normalize(institution)

    return {
        "canonical_shelfmark": canonical,
        "display_shelfmark": display,
        "institution": institution,
        "collection": inst.get("collection"),
        "subcollection": inst.get("subcollection"),
        "date": date,
        "language": language,
        "description": description,
        "genres": genres,
        "catalog_title": catalog_title,
        "scholarly_entry_count": scholarly_count,
        "subjects": subjects,
        "notes": notes,
        "paleographic": paleographic,
        "catalog_author": catalog_author,
        "catalog_persons": catalog_persons,
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
                    f.catalog_author = coalesce(f.catalog_author, $catalog_author),
                    f.catalog_persons = coalesce(f.catalog_persons, $catalog_persons),
                    f.notes = coalesce(f.notes, $notes),
                    f.paleographic_note = coalesce(f.paleographic_note, $paleographic),
                    f.scholarly_entry_count = CASE WHEN $sec > 0 THEN $sec ELSE f.scholarly_entry_count END,
                    f.genres = CASE WHEN size($genres) > 0 THEN $genres ELSE f.genres END,
                    f.subjects = CASE WHEN size($subjects) > 0 THEN $subjects ELSE f.subjects END,
                    f.pgpids = CASE WHEN size($pgpids) > 0 THEN $pgpids ELSE f.pgpids END
                """,
                canonical=r["canonical_shelfmark"], display=r["display_shelfmark"],
                date=r["date"], language=r["language"], description=r["description"],
                catalog_title=r["catalog_title"], sec=r["scholarly_entry_count"],
                genres=r["genres"], subjects=r["subjects"], notes=r["notes"],
                paleographic=r["paleographic"], catalog_author=r["catalog_author"],
                catalog_persons=r["catalog_persons"], pgpids=[str(p) for p in r["pgpids"]],
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
