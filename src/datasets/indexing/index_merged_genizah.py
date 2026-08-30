#!/usr/bin/env python3
"""Index the merged Cairo Genizah corpus into Elasticsearch.

Streams ``merged_shelfmarks.jsonl`` (PGP + FJP + KTIV unioned by canonical id),
turns each line into a :class:`GenizahDocument` via ``from_merged_format``, and
indexes it with :class:`ElasticsearchGenizahProcessor`. Every merged record is
indexed (all shelfmarks), keyed on the institution-qualified ``canonical_id``.

Embeddings default to text-only (no image fetching); image URLs (FJP + KTIV GCS)
are still stored for display and surfaced provenance fields
(``canonical_id``, ``sources_present``, ``image_preferred_source``,
``has_ktiv_images``) make the index navigable by source.

Run (needs ES creds in env: ELASTIC_SEARCH_HOST / ELASTIC_USER / ELASTIC_PASSWORD)::

    python -m src.datasets.indexing.index_merged_genizah --index genizah_merged_v4
    python -m src.datasets.indexing.index_merged_genizah --limit 200   # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Iterator, List, Optional

import dotenv

from src.datasets.document_models.genizah_document import GenizahDocument
from src.datasets.indexing.elastic_index_genizah import (
    ElasticsearchGenizahProcessor,
    es_config_from_env,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_MERGED = os.path.join(
    _REPO_ROOT, "src", "datasets", "raw_data", "cairo_genizah",
    "merged", "merged_shelfmarks.jsonl",
)


def iter_merged_documents(
    path: str,
    limit: Optional[int] = None,
    skip: int = 0,
) -> Iterator[GenizahDocument]:
    """Yield GenizahDocuments built from each line of the merged JSONL.

    :param path: Path to ``merged_shelfmarks.jsonl``.
    :param limit: Optional cap on the number of yielded documents (for smoke tests).
    :param skip: Number of leading JSONL lines to skip (resume support; lines
        are processed in file order and indexing is idempotent by ``_id``).
    :yields: GenizahDocument instances.
    """
    yielded = 0
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i < skip:
                continue
            if limit is not None and yielded >= limit:
                return
            line = line.strip()
            if line:
                yielded += 1
                yield GenizahDocument.from_merged_format(json.loads(line))


def index_merged(
    merged_path: str = DEFAULT_MERGED,
    index_name: str = "genizah_merged_v4",
    text_only: bool = True,
    batch_size: int = 100,
    limit: Optional[int] = None,
    skip: int = 0,
) -> None:
    """Build and index all merged documents into Elasticsearch.

    :param merged_path: Path to the merged JSONL.
    :param index_name: Destination Elasticsearch index.
    :param text_only: Use text-only embeddings (no image fetching).
    :param batch_size: Documents per indexing batch.
    :param limit: Optional cap for smoke testing.
    :param skip: Skip the first N JSONL lines (resume an interrupted run).
    """
    dotenv.load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    from src.embeddings.qwen_text_embedding import (
        EMBEDDING_DIMS,
        QwenTextEmbedding,
        build_index_meta,
    )

    if not text_only:
        raise ValueError(
            "Multimodal indexing is no longer supported: embeddings are "
            "text-only (Qwen3-Embedding-0.6B)."
        )
    embedding_model = QwenTextEmbedding()
    es_config = es_config_from_env()
    processor = ElasticsearchGenizahProcessor(
        embedding_model,
        elasticsearch_config=es_config,
        index_name=index_name,
        embedding_dims=EMBEDDING_DIMS,
        index_meta=build_index_meta(embedding_model),
    )

    batch: List[GenizahDocument] = []
    total = 0
    if skip:
        logger.info("Resuming: skipping first %d merged records", skip)
    for doc in iter_merged_documents(merged_path, limit=limit, skip=skip):
        batch.append(doc)
        if len(batch) >= batch_size:
            processor.process_documents(batch)
            total += len(batch)
            logger.info("Indexed %d documents so far", total)
            batch = []
    if batch:
        processor.process_documents(batch)
        total += len(batch)
    logger.info("Done. Indexed %d documents into %s", total, index_name)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", default=DEFAULT_MERGED)
    parser.add_argument("--index", default="genizah_merged_v5")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip the first N merged records (resume an interrupted run).")
    parser.add_argument("--multimodal", action="store_true",
                        help="Use multimodal embeddings (fetch images) instead of text-only.")
    args = parser.parse_args()
    index_merged(
        merged_path=args.merged,
        index_name=args.index,
        text_only=not args.multimodal,
        batch_size=args.batch_size,
        limit=args.limit,
        skip=args.skip,
    )


if __name__ == "__main__":
    main()
