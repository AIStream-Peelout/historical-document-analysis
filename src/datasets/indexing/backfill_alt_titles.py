#!/usr/bin/env python3
"""Backfill ``alt_titles`` into an already-built ``genizah_merged_*`` index.

The KTIV "Varying form of title" (``basic_catalog.varying_form_of_title``,
e.g. ``"Talmud Bavli: Megillah 2 a – b"``) was scraped and merged but never
lifted into the Elasticsearch document, so tractate / folio searches missed
KTIV-only fragments. :meth:`GenizahDocument.from_merged_format` now surfaces
it as ``alt_titles`` and folds it into ``full_text_content``.

This script patches an index that was pushed before that change, without a
re-index and without touching ``embedding_vector``:

1. adds the ``alt_titles`` mapping to the index (idempotent), then
2. streams ``merged_shelfmarks.jsonl`` and bulk-``update``s every document
   that has alternate titles with ``alt_titles`` + a recomputed
   ``full_text_content``, keyed on the same ``_id`` the indexer used
   (``canonical_id``).

Documents missing from the index (a partial run) are counted and skipped.
Embeddings are NOT refreshed here — only a full re-index re-embeds — so the
semantic path picks the titles up on the next rebuild.

Run (ES creds via ``.env``)::

    python -m src.datasets.indexing.backfill_alt_titles --index genizah_merged_v6 --dry-run
    python -m src.datasets.indexing.backfill_alt_titles --index genizah_merged_v6
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, Iterable, Iterator, Optional

import dotenv

from src.datasets.document_models.genizah_document import GenizahDocument
from src.datasets.indexing.elastic_index_genizah import (
    ALT_TITLES_MAPPING,
    es_config_from_env,
)
from src.datasets.indexing.index_merged_genizah import (
    DEFAULT_MERGED,
    _REPO_ROOT,
    iter_merged_documents,
)

logger = logging.getLogger(__name__)


def alt_title_update_actions(
    documents: Iterable[GenizahDocument],
    index_name: str,
) -> Iterator[Dict[str, Any]]:
    """Yield bulk partial-update actions for documents that carry alternate titles.

    :param documents: GenizahDocuments built from the merged JSONL.
    :param index_name: Target index.
    :yields: ``elasticsearch.helpers`` ``update`` actions keyed on ``doc_id``
        (the indexer's ``_id`` for merged records).
    """
    for doc in documents:
        if not doc.alt_titles:
            continue
        yield {
            "_op_type": "update",
            "_index": index_name,
            "_id": doc.doc_id,
            "doc": {
                "alt_titles": doc.alt_titles,
                "full_text_content": doc.create_full_text_content(),
            },
        }


def ensure_alt_titles_mapping(es_client: Any, index_name: str) -> None:
    """Add the ``alt_titles`` field to an existing index's mapping.

    Adding a new field is always allowed; re-putting an identical mapping is a
    no-op, so this is safe to call on every run.

    :param es_client: Connected ``Elasticsearch`` client.
    :param index_name: Index to update.
    """
    es_client.indices.put_mapping(index=index_name, properties=ALT_TITLES_MAPPING)
    logger.info("alt_titles mapping present on %s", index_name)


def backfill(
    index_name: str,
    merged_path: str = DEFAULT_MERGED,
    limit: Optional[int] = None,
    batch_size: int = 500,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Patch ``alt_titles`` / ``full_text_content`` into *index_name*.

    :param index_name: Existing ``genizah_merged_*`` index.
    :param merged_path: Path to ``merged_shelfmarks.jsonl``.
    :param limit: Cap on merged records read (smoke test).
    :param batch_size: Update actions per bulk request.
    :param dry_run: Count and log what would change without writing.
    :returns: ``{"updated", "missing", "failed", "candidates"}`` counts.
    """
    dotenv.load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    from elasticsearch import Elasticsearch, helpers

    actions = alt_title_update_actions(
        iter_merged_documents(merged_path, limit=limit), index_name
    )
    counts = {"updated": 0, "missing": 0, "failed": 0, "candidates": 0}

    if dry_run:
        for action in actions:
            counts["candidates"] += 1
            if counts["candidates"] <= 5:
                logger.info("would update %s -> %s", action["_id"], action["doc"]["alt_titles"])
        logger.info("dry run: %d documents would be updated in %s", counts["candidates"], index_name)
        return counts

    es_client = Elasticsearch(**es_config_from_env())
    ensure_alt_titles_mapping(es_client, index_name)

    for ok, item in helpers.streaming_bulk(
        es_client, actions, chunk_size=batch_size, raise_on_error=False
    ):
        counts["candidates"] += 1
        if ok:
            counts["updated"] += 1
        elif item["update"].get("error", {}).get("type") == "document_missing_exception":
            counts["missing"] += 1
        else:
            counts["failed"] += 1
            logger.error("update failed: %s", item)
        if counts["candidates"] % 1000 == 0:
            logger.info("progress: %s", counts)

    logger.info("done: %s", counts)
    if counts["failed"]:
        raise RuntimeError(f"{counts['failed']} updates failed; see log")
    return counts


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, help="Existing index to patch.")
    parser.add_argument("--merged", default=DEFAULT_MERGED)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    backfill(
        index_name=args.index,
        merged_path=args.merged,
        limit=args.limit,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
