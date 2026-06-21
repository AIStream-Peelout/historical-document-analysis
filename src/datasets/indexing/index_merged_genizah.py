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

    python -m src.datasets.indexing.index_merged_genizah --index genizah_merged_v1
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
from src.datasets.indexing.elastic_index_genizah import ElasticsearchGenizahProcessor

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_MERGED = os.path.join(
    _REPO_ROOT, "src", "datasets", "raw_data", "cairo_genizah",
    "merged", "merged_shelfmarks.jsonl",
)


def iter_merged_documents(path: str, limit: Optional[int] = None) -> Iterator[GenizahDocument]:
    """Yield GenizahDocuments built from each line of the merged JSONL.

    :param path: Path to ``merged_shelfmarks.jsonl``.
    :param limit: Optional cap on the number of documents (for smoke tests).
    :yields: GenizahDocument instances.
    """
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                return
            line = line.strip()
            if line:
                yield GenizahDocument.from_merged_format(json.loads(line))


def index_merged(
    merged_path: str = DEFAULT_MERGED,
    index_name: str = "genizah_merged_v1",
    text_only: bool = True,
    batch_size: int = 100,
    limit: Optional[int] = None,
) -> None:
    """Build and index all merged documents into Elasticsearch.

    :param merged_path: Path to the merged JSONL.
    :param index_name: Destination Elasticsearch index.
    :param text_only: Use text-only embeddings (no image fetching).
    :param batch_size: Documents per indexing batch.
    :param limit: Optional cap for smoke testing.
    """
    dotenv.load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    from src.embeddings.embedding_models import NomicsEmbedding

    embedding_model = NomicsEmbedding(text_only=text_only)
    es_config = {
        "hosts": [{"host": os.environ["ELASTIC_SEARCH_HOST"], "port": 443, "scheme": "https"}],
        "basic_auth": (os.environ["ELASTIC_USER"], os.environ["ELASTIC_PASSWORD"]),
    }
    processor = ElasticsearchGenizahProcessor(
        embedding_model, elasticsearch_config=es_config, index_name=index_name
    )

    batch: List[GenizahDocument] = []
    total = 0
    for doc in iter_merged_documents(merged_path, limit=limit):
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
    parser.add_argument("--index", default="genizah_merged_v1")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--multimodal", action="store_true",
                        help="Use multimodal embeddings (fetch images) instead of text-only.")
    args = parser.parse_args()
    index_merged(
        merged_path=args.merged,
        index_name=args.index,
        text_only=not args.multimodal,
        batch_size=args.batch_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
