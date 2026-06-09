#!/usr/bin/env python3
"""
Node Relation Enrichment — Pass 3 extraction CLI

Queries Neo4j for Person/Place/Institution nodes that have source_books set,
assembles focused page-text context from the academic literature, calls the LLM,
and writes one reviewable JSON file per entity under:

    enriched_relations/<Label>/<EntityName>.json

Nothing is written to Neo4j here.  Review / edit the JSON files, then import
them with:

    python academic_kg_import.py --enriched-dir enriched_relations/

Usage
-----
# Extract all Person nodes
python enrich_node_relations.py --label Person

# Single entity
python enrich_node_relations.py --label Person --name "Rachel the Byzantine"

# Place nodes via Gemini
python enrich_node_relations.py --label Place --backend gemini

# Dry-run (shows context, skips LLM)
python enrich_node_relations.py --label Person --dry-run

# Re-extract entities already processed
python enrich_node_relations.py --label Person --force

Environment
-----------
NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD / NEO4J_DATABASE
LM_STUDIO_URL   http://localhost:1234/v1   (default)
LM_STUDIO_MODEL qwen3:8b                   (default)
GEMINI_API_KEY  ...                        (required when --backend gemini)
"""

import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import dotenv
from neo4j import GraphDatabase
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))
dotenv.load_dotenv(project_root / ".env")

from src.models.llm.academic.llm_client import (  # noqa: E402
    BACKEND_GEMINI,
    BACKEND_LMS,
    LLMClient,
    build_context_for_entity,
    build_prompt,
    output_path,
    output_root,
    parse_triplets,
    write_enrichment_file,
    _SYSTEM_PROMPT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("enrich_node_relations.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class NodeRelationEnricher:
    """Fetches candidate nodes from Neo4j and runs Pass 3 LLM extraction."""

    def __init__(
        self,
        neo4j_uri:        str,
        neo4j_user:       str,
        neo4j_password:   str,
        neo4j_database:   str = "neo4j",
        backend:          str = BACKEND_LMS,
        lms_url:          str = "http://localhost:1234/v1",
        lms_model:        str = "qwen3:8b",
        gemini_api_key:   Optional[str] = None,
        gemini_model:     str = "gemini-3.5-flash",
        min_source_books: int = 1,
        out_dir:          Optional[Path] = None,
    ):
        self.driver   = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.database = neo4j_database
        self.min_source_books = min_source_books
        self.out_dir  = out_dir or output_root

        self.llm = LLMClient(
            backend=backend, lms_url=lms_url, lms_model=lms_model,
            gemini_api_key=gemini_api_key, gemini_model=gemini_model,
        )
        self._model_id = (
            f"{backend}:{lms_model if backend == BACKEND_LMS else gemini_model}"
        )
        logger.info(f"NodeRelationEnricher ready (backend={backend}, db={neo4j_database})")

    def close(self):
        self.driver.close()

    def _fetch_candidates(
        self,
        label: str,
        name_filter: Optional[str] = None,
        force: bool = False,
    ) -> List[Dict]:
        where = [
            "n.source_books IS NOT NULL",
            f"size(n.source_books) >= {self.min_source_books}",
        ]
        if not force:
            where.append("(n.relations_enriched IS NULL OR n.relations_enriched = false)")
        if name_filter:
            where.append("n.name = $name")

        q = f"""
            MATCH (n:{label})
            WHERE {' AND '.join(where)}
            RETURN n.name         AS name,
                   n.source_books AS source_books,
                   coalesce(n.context, n.description, '') AS context
            ORDER BY size(n.source_books) DESC
        """
        params = {"name": name_filter} if name_filter else {}
        with self.driver.session(database=self.database) as s:
            return [r.data() for r in s.run(q, **params)]

    def extract_entity(
        self,
        name: str,
        label: str,
        source_books: List[str],
        entity_context: str = "",
        dry_run: bool = False,
        force: bool = False,
    ) -> Dict[str, int]:
        out = output_path(label, name, self.out_dir)

        if out.exists() and not force:
            logger.info(f"  Already extracted: {out.name} — skipping")
            return {"already_exists": 1}

        context_text, contributing = build_context_for_entity(name, label, source_books)

        if not context_text.strip():
            logger.info(f"  No context found for '{name}'")
            if not dry_run:
                write_enrichment_file(out, name, label, source_books, [],
                                      [], self._model_id, 0, status="no_context")
            return {"no_context": 1}

        logger.info(f"  '{name}': {len(context_text):,} chars from {len(contributing)} book(s)")

        if dry_run:
            print(f"\n{'='*60}\nDRY RUN [{label}] {name}")
            print(f"Books: {contributing}")
            print(f"Context ({len(context_text)} chars):\n{context_text[:500]}...")
            return {"dry_run": 1}

        prompt = build_prompt(name, label, entity_context, context_text)
        try:
            raw = self.llm.complete(_SYSTEM_PROMPT, prompt,
                                    max_context_chars=len(context_text))
        except Exception as e:
            logger.error(f"  LLM error for '{name}': {e}")
            write_enrichment_file(out, name, label, source_books, contributing,
                                  [], self._model_id, len(context_text), status="llm_error")
            return {"llm_error": 1}

        triplets = parse_triplets(raw, name, label)
        status   = "ok" if triplets else "no_triplets"
        write_enrichment_file(out, name, label, source_books, contributing,
                              triplets, self._model_id, len(context_text), status=status)

        logger.info(f"  '{name}': {len(triplets)} triplet(s) → {out.name}")
        return {"triplets_extracted": len(triplets), "files_written": 1}

    def extract_all(
        self,
        label: str,
        name_filter: Optional[str] = None,
        force: bool = False,
        dry_run: bool = False,
        delay: float = 0.5,
    ) -> Dict[str, int]:
        candidates = self._fetch_candidates(label, name_filter, force)
        logger.info(f"Found {len(candidates)} {label} node(s) to extract")

        if not candidates:
            logger.info("Nothing to do.")
            return {}

        totals: Dict[str, int] = defaultdict(int)
        for node in tqdm(candidates, desc=f"Extracting {label}"):
            counts = self.extract_entity(
                name=node["name"],
                label=label,
                source_books=node.get("source_books") or [],
                entity_context=node.get("context") or "",
                dry_run=dry_run,
                force=force,
            )
            for k, v in counts.items():
                totals[k] += v
            if not dry_run and delay > 0:
                time.sleep(delay)

        _print_summary(f"Extraction summary — {label}", totals, dry_run)
        return dict(totals)


def _print_summary(title: str, totals: Dict, dry_run: bool = False):
    tag = "[DRY RUN] " if dry_run else ""
    print(f"\n{tag}{title}\n{'='*50}")
    for k, v in sorted(totals.items()):
        print(f"  {k}: {v:,}")
    print("=" * 50)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label",     "-l", default="Person",
                        help="Node label to extract (Person, Place, Institution, ...)")
    parser.add_argument("--name",      "-n", default=None,
                        help="Extract only this specific entity name")
    parser.add_argument("--backend",   choices=[BACKEND_LMS, BACKEND_GEMINI],
                        default=BACKEND_LMS)
    parser.add_argument("--model",     default=None, help="Model name override")
    parser.add_argument("--lms-url",   default="http://localhost:1234/v1")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write JSON files (default: enriched_relations/)")
    parser.add_argument("--dry-run",   "-d", action="store_true",
                        help="Show context and skip LLM call")
    parser.add_argument("--force",     "-f", action="store_true",
                        help="Re-extract entities already processed")
    parser.add_argument("--min-books", type=int, default=1,
                        help="Minimum source_books count to qualify")
    parser.add_argument("--delay",     type=float, default=0.5,
                        help="Seconds between LLM calls (default: 0.5)")
    args = parser.parse_args()

    uri      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    user     = os.getenv("NEO4J_USER",     "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "genizah-prod")

    if not password:
        logger.error("NEO4J_PASSWORD not set")
        sys.exit(1)

    model   = args.model or os.getenv("LM_STUDIO_MODEL", "qwen3:8b")
    out_dir = Path(args.output_dir) if args.output_dir else None

    enricher = NodeRelationEnricher(
        neo4j_uri=uri, neo4j_user=user, neo4j_password=password,
        neo4j_database=database,
        backend=args.backend, lms_url=args.lms_url, lms_model=model,
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        min_source_books=args.min_books,
        out_dir=out_dir,
    )
    try:
        enricher.extract_all(
            label=args.label,
            name_filter=args.name,
            force=args.force,
            dry_run=args.dry_run,
            delay=args.delay,
        )
    finally:
        enricher.close()


if __name__ == "__main__":
    main()
