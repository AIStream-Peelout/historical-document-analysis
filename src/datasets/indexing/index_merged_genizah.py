#!/usr/bin/env python3
"""Index the merged Cairo Genizah corpus into Elasticsearch.

Streams ``merged_shelfmarks.jsonl`` (PGP + KTIV + Bodleian and the other catalogue sources, unioned by canonical id),
turns each line into a :class:`GenizahDocument` via ``from_merged_format``, and
indexes it with :class:`ElasticsearchGenizahProcessor`. Every merged record is
indexed (all shelfmarks), keyed on the institution-qualified ``canonical_id``.

Embeddings are text-only: the fine-tuned Qwen3 embedder over the versioned
document text (``--text-version``, default ``DEFAULT_TEXT_VERSION``; see
``src.embeddings.text_representation``), rendered with or without the KTIV
scholarly frames (:data:`INDEX_SCHOLARLY_FRAMES`, default off: the text the
pinned embedder was trained on; ``--scholarly-frames`` is the on-arm). Records
whose text is not eligible (empty, too short, or a boilerplate label plus
metadata only) are indexed with ``semantic_eligible: false`` and no vector, so
shelf-mark and BM25 lookup still find them. Every record carries
``embedded_text_sha256`` (the sha256 of its versioned text, ineligible ones
included), and each distinct text is embedded once. Image URLs are still stored
for display and surfaced provenance fields (``canonical_id``,
``sources_present``, ``image_preferred_source``, ``has_ktiv_images``,
``has_bodleian_images``) make the index navigable by source.

The index ``_meta`` records the model, its revision, the query prompt, the
text version, the text-hash rule, the text function (version, render flag,
golden-fixture sha, commit), the merged file (path, records, sha256) and three
canaries (``build_index_meta``); a successful build ends by adding ``counts``
(documents, eligible, distinct text hashes). Writing into an existing index
whose ``_meta`` differs on model, revision, text version or render flag is
refused, and so are the live / previous index names unless
``--i-know-this-is-live`` is passed. Never upsert into the live index: build a
new one and swap.

Run (needs ES creds in env: ELASTIC_SEARCH_HOST / ELASTIC_USER / ELASTIC_PASSWORD,
and HF1_TOKEN for the private model)::

    python -m src.datasets.indexing.index_merged_genizah --index genizah_merged_v10 --text-version semantic_v4
    python -m src.datasets.indexing.index_merged_genizah --index genizah_merged_v10_smoke --text-version semantic_v4 \\
        --limit 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from typing import Any, Dict, Iterator, List, Optional

import dotenv

from src.datasets.document_models.genizah_document import GenizahDocument
from src.datasets.indexing.elastic_index_genizah import (
    ElasticsearchGenizahProcessor,
    es_config_from_env,
    pick_document_canary,
    write_index_counts,
)
from src.embeddings.text_representation import (
    DEFAULT_TEXT_VERSION,
    TEXT_VERSIONS,
    DocumentCanary,
    check_text_version,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_MERGED = os.path.join(
    _REPO_ROOT, "src", "datasets", "raw_data", "cairo_genizah",
    "merged", "merged_shelfmarks.jsonl",
)
# The live index and its predecessors: never written to by accident.
LIVE_INDEX_NAMES = frozenset({"genizah_merged_v6", "genizah_merged_v7", "genizah_merged_v8"})
#: Destination of a build when ``--index`` is not given: bump it here for each new index (the live names
#: above are refused unless ``--i-know-this-is-live`` is passed, so a stale default cannot hit production).
DEFAULT_INDEX_NAME = "genizah_merged_v10"
#: Whether the merged index embeds the fragment text WITH the KTIV scholarly frames. It must match the text the pinned
#: embedder was trained on (v10: ``semantic_v4`` without frames), so it flips only together with ``FINETUNED_MODEL_*``
#: (``src.embeddings.qwen_text_embedding``) and ``DEFAULT_TEXT_VERSION`` (``src.embeddings.text_representation``).
#: Every default here reads it; ``ElasticsearchGenizahProcessor``, ``pick_document_canary`` and
#: ``text_function_block`` default to the same value (tests/test_v10_index_build.py pins them to it).
INDEX_SCHOLARLY_FRAMES = False


def check_index_name(index_name: str, allow_live: bool = False) -> str:
    """Refuse the live / previous production index names.

    :param index_name: Destination index.
    :param allow_live: ``--i-know-this-is-live`` was passed.
    :returns: The same name.
    :rtype: str
    :raises ValueError: For a name in :data:`LIVE_INDEX_NAMES` without ``allow_live``.
    """
    if index_name in LIVE_INDEX_NAMES and not allow_live:
        raise ValueError(
            f"{index_name!r} is the live index or a previous one; build a new index (e.g. {DEFAULT_INDEX_NAME}) "
            "and swap, or pass --i-know-this-is-live")
    return index_name


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


def merged_source(merged_path: str) -> Dict[str, Any]:
    """``_meta.source``: the merged file a build reads, in one streamed pass.

    :param merged_path: Path to the merged JSONL.
    :returns: ``merged_path`` (relative to the repository root when inside it), ``merged_records`` (non-blank lines,
        the records :func:`iter_merged_documents` yields) and ``merged_sha256`` (of the file's bytes).
    :rtype: Dict[str, Any]
    """
    digest = hashlib.sha256()
    records = 0
    with open(merged_path, "rb") as fh:
        for line in fh:
            digest.update(line)
            records += bool(line.strip())
    absolute = os.path.abspath(merged_path)
    inside = absolute.startswith(_REPO_ROOT + os.sep)
    return {"merged_path": os.path.relpath(absolute, _REPO_ROOT) if inside else absolute,
            "merged_records": records, "merged_sha256": digest.hexdigest()}


def find_document_canary(merged_path: str, text_version: str,
                         include_scholarly_frames: bool = INDEX_SCHOLARLY_FRAMES) -> DocumentCanary:
    """The first eligible, indexable record of the merged file (independent of ``--skip`` / ``--limit``).

    :param merged_path: Path to the merged JSONL.
    :param text_version: Text representation version.
    :param include_scholarly_frames: The build's render flag, so the canary text is the text the build embeds.
    :returns: The record's id and versioned text.
    :rtype: DocumentCanary
    :raises ValueError: When no record is eligible.
    """
    canary = pick_document_canary(iter_merged_documents(merged_path), text_version, include_scholarly_frames)
    if canary is None:
        raise ValueError(f"no eligible record in {merged_path} to serve as the document canary")
    return canary


def index_merged(
    *,
    index_name: str,
    merged_path: str = DEFAULT_MERGED,
    text_version: str = DEFAULT_TEXT_VERSION,
    include_scholarly_frames: bool = INDEX_SCHOLARLY_FRAMES,
    allow_live: bool = False,
    text_only: bool = True,
    batch_size: int = 100,
    limit: Optional[int] = None,
    skip: int = 0,
    embedding_model: Optional[Any] = None,
    elasticsearch_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    """Build and index all merged documents into Elasticsearch.

    The index ``_meta`` records the text function (with the render flag used) and the merged file
    (``build_index_meta``); after the last batch, ``_meta.counts`` is written from the index itself
    (``write_index_counts``: documents, eligible, distinct text hashes).

    :param index_name: Destination Elasticsearch index (a new name; the CLI default is :data:`DEFAULT_INDEX_NAME`, see :data:`LIVE_INDEX_NAMES`).
    :param merged_path: Path to the merged JSONL.
    :param text_version: Text representation version (``src.embeddings.text_representation``).
    :param include_scholarly_frames: Render the fragment text with the KTIV scholarly frames (default
        :data:`INDEX_SCHOLARLY_FRAMES`: off, the pinned embedder's training text); recorded in
        ``_meta.text_function.scholarly_frames``.
    :param allow_live: Allow a name in :data:`LIVE_INDEX_NAMES` (the ``_meta`` writer guard still applies).
    :param text_only: Use text-only embeddings (no image fetching).
    :param batch_size: Documents per indexing batch.
    :param limit: Optional cap for smoke testing.
    :param skip: Skip the first N JSONL lines (resume an interrupted run).
    :param embedding_model: Embedder to use (default: ``QwenTextEmbedding()``, the pinned fine-tune).
    :param elasticsearch_config: Elasticsearch client kwargs (default: from the environment / ``.env``).
    :returns: ``{"indexed": n, "without_vector": m}`` for this run (``_meta.counts`` covers the whole index).
    :rtype: Dict[str, int]
    """
    check_text_version(text_version)
    check_index_name(index_name, allow_live)
    if not text_only:
        raise ValueError(
            "Multimodal indexing is no longer supported: embeddings are "
            "text-only (Qwen3-Embedding-0.6B)."
        )
    from src.embeddings.qwen_text_embedding import (
        EMBEDDING_DIMS,
        QwenTextEmbedding,
        build_index_meta,
        text_function_block,
    )

    if embedding_model is None or elasticsearch_config is None:
        dotenv.load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    if embedding_model is None:
        embedding_model = QwenTextEmbedding()
    if elasticsearch_config is None:
        elasticsearch_config = es_config_from_env()

    source = merged_source(merged_path)
    logger.info("Merged file: %s (%d records, sha256 %s)", source["merged_path"], source["merged_records"],
                source["merged_sha256"])
    canary = find_document_canary(merged_path, text_version, include_scholarly_frames)
    logger.info("Document canary: %s (%d chars of %s text, scholarly frames %s)", canary.doc_id, len(canary.text),
                text_version, "on" if include_scholarly_frames else "off")
    index_meta = build_index_meta(embedding_model, text_version, canary,
                                  text_function=text_function_block(text_version, include_scholarly_frames),
                                  source=source)
    processor = ElasticsearchGenizahProcessor(
        embedding_model,
        elasticsearch_config=elasticsearch_config,
        index_name=index_name,
        embedding_dims=EMBEDDING_DIMS,
        index_meta=index_meta,
        text_version=text_version,
        include_scholarly_frames=include_scholarly_frames,
    )

    batch: List[GenizahDocument] = []
    total = without_vector = 0
    if skip:
        logger.info("Resuming: skipping first %d merged records", skip)
    for doc in iter_merged_documents(merged_path, limit=limit, skip=skip):
        batch.append(doc)
        if len(batch) >= batch_size:
            result = processor.process_documents(batch)
            total += result["processed_count"]
            without_vector += result["ineligible_count"]
            logger.info("Indexed %d documents so far (%d without a vector)", total, without_vector)
            batch = []
    if batch:
        result = processor.process_documents(batch)
        total += result["processed_count"]
        without_vector += result["ineligible_count"]
    counts = write_index_counts(processor.es_client, index_name)
    logger.info("Done. Indexed %d documents into %s (%d without a vector); %d distinct texts embedded, %d vectors "
                "reused; _meta.counts %s", total, index_name, without_vector, processor.stats["texts_embedded"],
                processor.stats["text_cache_hits"], counts)
    return {"indexed": total, "without_vector": without_vector}


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser; the defaults are the code-level constants (``DEFAULT_INDEX_NAME`` etc.).

    :return: The parser, so tests can check the defaults without running a build.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merged", default=DEFAULT_MERGED)
    parser.add_argument("--index", default=DEFAULT_INDEX_NAME,
                        help="Destination index (default: %(default)s, set in the code; the live names are refused).")
    parser.add_argument("--text-version", default=DEFAULT_TEXT_VERSION, choices=TEXT_VERSIONS,
                        help="Document text to embed (default: %(default)s, what the fine-tuned model was trained on).")
    default_frames = (f"default: {'with' if INDEX_SCHOLARLY_FRAMES else 'without'} frames (INDEX_SCHOLARLY_FRAMES), "
                      "which must match the pinned embedder's training text (v10: no frames)")
    frames = parser.add_mutually_exclusive_group()
    frames.add_argument("--scholarly-frames", dest="include_scholarly_frames", action="store_true",
                        default=INDEX_SCHOLARLY_FRAMES,
                        help=f"Embed the fragment text WITH the KTIV scholarly frames (the on-arm / frames ablation; "
                             f"{default_frames}).")
    frames.add_argument("--no-scholarly-frames", dest="include_scholarly_frames", action="store_false",
                        default=INDEX_SCHOLARLY_FRAMES,
                        help=f"Embed the fragment text without the KTIV scholarly frames ({default_frames}).")
    parser.add_argument("--i-know-this-is-live", dest="allow_live", action="store_true",
                        help=f"Allow writing to {', '.join(sorted(LIVE_INDEX_NAMES))}.")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip the first N merged records (resume an interrupted run).")
    parser.add_argument("--multimodal", action="store_true",
                        help="Use multimodal embeddings (fetch images) instead of text-only.")
    return parser


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args()
    index_merged(
        index_name=args.index,
        merged_path=args.merged,
        text_version=args.text_version,
        include_scholarly_frames=args.include_scholarly_frames,
        allow_live=args.allow_live,
        text_only=not args.multimodal,
        batch_size=args.batch_size,
        limit=args.limit,
        skip=args.skip,
    )


if __name__ == "__main__":
    main()
