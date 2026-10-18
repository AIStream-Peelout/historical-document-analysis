# file_name
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Any, Tuple
import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from src.embeddings.text_representation import (
    FRAGMENT,
    DocumentCanary,
    EmbeddingText,
    check_text_version,
    embedding_text,
)

logger = logging.getLogger(__name__)
logging.getLogger('elasticsearch').setLevel(logging.DEBUG)


# Alternate / varying titles (KTIV "Varying form of title"). Kept as its own
# field, separate from ``description``, so the web app can weight it
# independently. ``keyword`` sub-field for exact-title aggregation / filtering.
ALT_TITLES_MAPPING: Dict[str, Any] = {
    "alt_titles": {
        "type": "text",
        "analyzer": "multilingual",
        "fields": {"keyword": {"type": "keyword", "ignore_above": 512}},
    },
}

# KTIV scholarly identifications (each scholarly entry's frame and specific work
# title): their own field, the same shape as ``alt_titles``, so the web app can
# weight them separately and the embedding text can leave them out.
SCHOLARLY_FRAMES_MAPPING: Dict[str, Any] = {
    "scholarly_frames": {
        "type": "text",
        "analyzer": "multilingual",
        "fields": {"keyword": {"type": "keyword", "ignore_above": 512}},
    },
}

# Subject keywords (KTIV catalogue subjects, PGP tags): the ``Subjects:`` line of the
# embedding text, kept as its own field so the web app can facet on and boost it.
SUBJECTS_MAPPING: Dict[str, Any] = {
    "subjects": {
        "type": "text",
        "analyzer": "multilingual",
        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
    },
}
# Genre taxonomy terms (KTIV scholarly domain levels): facets and lexical search only;
# never part of the embedding text.
GENRES_MAPPING: Dict[str, Any] = {
    "genres": {
        "type": "text",
        "analyzer": "multilingual",
        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
    },
}

# sha256 of the UTF-8 bytes of a document's versioned text (``EmbeddingText.text_sha256``, index v10 on). Written on
# every document of a versioned write, ineligible ones included, so an updater can tell a changed text without
# re-embedding and the site can collapse documents that share one text.
TEXT_HASH_FIELD = "embedded_text_sha256"

# Versioned-text fields (index v9 on; the text hash from v10). A ``semantic_eligible: false`` document
# has no ``embedding_vector`` at all: a dense_vector field may be absent, kNN
# never returns such documents, and they stay reachable by shelf mark / BM25.
TEXT_VERSION_MAPPING: Dict[str, Any] = {
    "semantic_eligible": {"type": "boolean"},
    "text_representation_version": {"type": "keyword"},
    TEXT_HASH_FIELD: {"type": "keyword"},
}

# ``_meta`` keys an existing index must match before a versioned writer adds to it (plus the
# ``text_function.scholarly_frames`` render flag when both ``_meta`` blocks carry a ``text_function``).
GUARDED_META_KEYS = ("model_name", "model_revision", "text_representation_version")

# Embed once per text: one processor keeps at most this many distinct float32 vectors (4 KiB each at 1024 dims, so
# about 400 MiB at the cap; the merged corpus has about 56k distinct eligible texts). Past the cap new texts are still
# embedded once per batch, they are just not remembered for later batches.
TEXT_VECTOR_CACHE_MAX_ENTRIES = 100_000


class IndexMetaMismatchError(RuntimeError):
    """An existing index was built with another model, revision, text version or render flag than this run."""


def read_index_meta(es_client: Any, index_name: str) -> Dict[str, Any]:
    """Read an index's mapping ``_meta`` (empty when it has none).

    :param es_client: Elasticsearch client.
    :param index_name: Index (or alias) name.
    :return: The ``_meta`` dict.
    :rtype: Dict[str, Any]
    """
    mapping = es_client.indices.get_mapping(index=index_name)
    return (next(iter(mapping.values())).get("mappings") or {}).get("_meta") or {}


def index_scholarly_frames(meta: Mapping[str, Any]) -> bool:
    """Whether an index's texts were rendered with the KTIV scholarly frames (``_meta.text_function.scholarly_frames``).

    :param meta: Index ``_meta`` (or the ``_meta`` a run would write).
    :return: The flag; ``True`` when ``_meta`` has no ``text_function`` (an index built before the flag existed was
        rendered with ``create_text_representation()``'s default, which lists the frames).
    :rtype: bool
    """
    return bool((meta.get("text_function") or {}).get("scholarly_frames", True))


def document_scholarly_frames(doc: Any, include_scholarly_frames: bool) -> bool:
    """The render flag to hand ``embedding_text`` for one document.

    Only fragment documents (``embedding_text_kind == "fragment"``) carry KTIV scholarly frames, and only their
    ``create_text_representation`` takes the keyword; any other document (a bibliography page) has a single rendering,
    asked for with ``True`` (``text_representation._production_text`` then passes no keyword).

    :param doc: Document.
    :param include_scholarly_frames: The build's render flag.
    :return: The flag for this document.
    :rtype: bool
    """
    return include_scholarly_frames or getattr(doc, "embedding_text_kind", None) != FRAGMENT


def index_meta_mismatches(existing: Mapping[str, Any], expected: Mapping[str, Any]) -> List[str]:
    """Describe every :data:`GUARDED_META_KEYS` value that differs between two ``_meta`` blocks.

    A key missing from ``existing`` (a pre-v9 index) counts as a mismatch. When both blocks carry a ``text_function``
    (index v10 on, and the merged indexer's run), their ``scholarly_frames`` render flags must match too: resuming a
    build with the other flag would mix two texts in one index. An index that predates ``text_function`` is not
    compared on the flag.

    :param existing: ``_meta`` of the index on the cluster.
    :param expected: ``_meta`` this run would write.
    :return: One message per differing key (empty when the run may write).
    :rtype: List[str]
    """
    problems = [f"{key}: index has {existing.get(key)!r}, this run would write {expected.get(key)!r}"
                for key in GUARDED_META_KEYS if existing.get(key) != expected.get(key)]
    have, want = existing.get("text_function"), expected.get("text_function")
    if have is not None and want is not None and have.get("scholarly_frames") != want.get("scholarly_frames"):
        problems.append(f"text_function.scholarly_frames: index has {have.get('scholarly_frames')!r}, "
                        f"this run would write {want.get('scholarly_frames')!r}")
    return problems


def assert_index_writable(es_client: Any, index_name: str, expected_meta: Mapping[str, Any]) -> None:
    """Writer guard: refuse to add to an existing index built with another model or text version.

    :param es_client: Elasticsearch client.
    :param index_name: Target index.
    :param expected_meta: ``_meta`` this run would write (``build_index_meta``).
    :raises IndexMetaMismatchError: When the index exists and its ``_meta`` differs on a guarded key.
    """
    if not es_client.indices.exists(index=index_name):
        return
    existing = read_index_meta(es_client, index_name)
    problems = index_meta_mismatches(existing, expected_meta)
    if problems:
        legacy = (f" (pre-v9 _meta: embedding_model={existing.get('embedding_model')!r}"
                  f"@{str(existing.get('embedding_model_revision'))[:12]})" if "model_name" not in existing else "")
        raise IndexMetaMismatchError(
            f"Refusing to write to existing index {index_name!r}: its _meta does not match this run "
            f"({'; '.join(problems)}){legacy}. Build a new index name instead.")


def es_document_id(doc: Any) -> Optional[str]:
    """The ``_id`` a document is indexed under (page uuid when it has one, else ``doc_id``).

    :param doc: Document.
    :return: The id, or ``None`` when the document has neither.
    :rtype: Optional[str]
    """
    return getattr(doc, "page_uuid", None) or doc.doc_id


def pick_document_canary(docs: Iterable[Any], version: str,
                         include_scholarly_frames: bool = False) -> Optional[DocumentCanary]:
    """The first document that is indexed with a vector, as the index's ``_meta`` document canary.

    :param docs: Documents in indexing order.
    :param version: Text representation version.
    :param include_scholarly_frames: The build's render flag (as :class:`ElasticsearchGenizahProcessor`, default off:
        ``index_merged_genizah.INDEX_SCHOLARLY_FRAMES``), so the canary text is the text the build embeds.
    :return: Its id and versioned text, or ``None`` when no document is eligible.
    :rtype: Optional[DocumentCanary]
    """
    for doc in docs:
        doc_id = es_document_id(doc)
        if not doc_id or not doc.shelf_mark or not doc.shelf_mark.strip():
            continue  # never indexed (see process_documents)
        item = embedding_text(doc, version,
                              include_scholarly_frames=document_scholarly_frames(doc, include_scholarly_frames))
        if item.eligible:
            return DocumentCanary(doc_id=doc_id, text=item.text)
    return None


def distinct_values(es_client: Any, index_name: str, field: str, query: Optional[Mapping[str, Any]] = None,
                    page_size: int = 10_000) -> int:
    """Exact number of distinct values of a keyword field, by paging a composite aggregation.

    (A ``cardinality`` aggregation is approximate above 40,000 values; the merged index holds about 60k texts.)

    :param es_client: Elasticsearch client.
    :param index_name: Index.
    :param field: Keyword field.
    :param query: Documents to count over (default: every document).
    :param page_size: Buckets per request.
    :return: Distinct values; documents without the field are not counted.
    :rtype: int
    """
    total = 0
    after: Optional[Dict[str, Any]] = None
    while True:
        composite: Dict[str, Any] = {"size": page_size, "sources": [{"value": {"terms": {"field": field}}}]}
        if after is not None:
            composite["after"] = after
        filters = {"query": dict(query)} if query is not None else {}
        response = es_client.search(index=index_name, size=0, aggs={"distinct": {"composite": composite}}, **filters)
        page = response["aggregations"]["distinct"]
        total += len(page["buckets"])
        after = page.get("after_key")
        if len(page["buckets"]) < page_size or after is None:
            return total


def index_counts(es_client: Any, index_name: str) -> Dict[str, int]:
    """Document, eligible and distinct-text counts of an index, read from the index itself (so a build resumed with
    ``--skip`` counts every document, not only the last run's).

    :param es_client: Elasticsearch client.
    :param index_name: Index.
    :return: ``documents``, ``eligible`` (documents with ``semantic_eligible: true``), ``distinct_text_sha256``
        (distinct :data:`TEXT_HASH_FIELD` values over every document) and ``distinct_eligible_text_sha256`` (over the
        eligible ones: the number of distinct vectors).
    :rtype: Dict[str, int]
    """
    eligible = {"term": {"semantic_eligible": True}}
    return {
        "documents": es_client.count(index=index_name)["count"],
        "eligible": es_client.count(index=index_name, query=eligible)["count"],
        "distinct_text_sha256": distinct_values(es_client, index_name, TEXT_HASH_FIELD),
        "distinct_eligible_text_sha256": distinct_values(es_client, index_name, TEXT_HASH_FIELD, eligible),
    }


def write_index_counts(es_client: Any, index_name: str) -> Dict[str, int]:
    """Record :func:`index_counts` as ``_meta.counts`` at the end of a build.

    ``put_mapping`` replaces the whole ``_meta``, so every other key is read back and kept. The writer guard
    (:func:`index_meta_mismatches`) never looks at ``counts``.

    :param es_client: Elasticsearch client.
    :param index_name: Index.
    :return: The counts written.
    :rtype: Dict[str, int]
    """
    counts = index_counts(es_client, index_name)
    meta = read_index_meta(es_client, index_name)
    es_client.indices.put_mapping(index=index_name, meta={**meta, "counts": counts})
    return counts


def es_config_from_env() -> Dict[str, Any]:
    """Build the Elasticsearch client config from environment variables.

    Uses ``ELASTIC_SEARCH_HOST`` (required) plus optional
    ``ELASTIC_SEARCH_PORT`` (default 443) and ``ELASTIC_SEARCH_SCHEME``
    (default https). Indexing runs on the ES host machine can set
    ``ELASTIC_SEARCH_HOST=localhost ELASTIC_SEARCH_PORT=9200
    ELASTIC_SEARCH_SCHEME=http`` to talk to the Docker ES directly instead of
    going through the Cloudflare-forwarded endpoint.

    :return: Kwargs for the :class:`Elasticsearch` constructor.
    :rtype: Dict[str, Any]
    """
    return {
        "hosts": [{
            "host": os.environ["ELASTIC_SEARCH_HOST"],
            "port": int(os.environ.get("ELASTIC_SEARCH_PORT", "443")),
            "scheme": os.environ.get("ELASTIC_SEARCH_SCHEME", "https"),
        }],
        "basic_auth": (os.environ["ELASTIC_USER"], os.environ["ELASTIC_PASSWORD"]),
    }


class ElasticsearchGenizahProcessor:
    """Simple processor for indexing GenizahDocument objects to Elasticsearch.

    This processor handles embedding generation, caching, and Elasticsearch indexing
    for Pydantic GenizahDocument objects.

    Example:
        config = {
        ...     'hosts': ['localhost:9200'],
        ...     'index_name': 'genizah-documents'
        ... }
        processor = ElasticsearchGenizahProcessor(
        ...     embedding_model=my_model,
        ...     elasticsearch_config=config
        ... )
        results = processor.process_documents(documents=my_docs)
    """

    def __init__(self,
                 embedding_model: Any,
                 elasticsearch_config: Dict[str, Any],
                 index_name: str = "genizah_documents_text1",
                 cache_dir: str = "embedding_cache",
                 use_cache: bool = False,
                 embedding_dims: int = 128,
                 index_meta: Optional[Dict[str, Any]] = None,
                 text_version: Optional[str] = None,
                 include_scholarly_frames: bool = False):
        """Initialize the Elasticsearch processor.

        :param embedding_model: Model for generating document embeddings
        :type embedding_model: Any
        :param elasticsearch_config: The dictionary configuration of instantiating the Elasticsearch client. Should contain
        hosts, basic_auth, etc...
        :type elasticsearch_config: Dict[str, Any]
        :param cache_dir: Directory for caching the generated embedding model embeddings.
        :type cache_dir: str
        :param embedding_dims: Dimensionality of the dense_vector mapping.
        :type embedding_dims: int
        :param index_meta: Optional ``_meta`` block written into the index
            mapping at creation time (embedding contract, canary, etc.).
        :type index_meta: Optional[Dict[str, Any]]
        :param text_version: Versioned embedding text (``src.embeddings.text_representation``). When set, documents
            are embedded from ``embedding_text(doc, text_version)``, ineligible ones are indexed without a vector,
            every document records ``semantic_eligible``, ``text_representation_version`` and
            ``embedded_text_sha256``, each distinct text is embedded once (:meth:`_get_versioned_embeddings`), and an
            existing index is written to only when its ``_meta`` matches ``index_meta`` (:func:`assert_index_writable`).
            ``None`` keeps the legacy path (``doc.create_text_representation()``, every document embedded).
        :type text_version: Optional[str]
        :param include_scholarly_frames: Render flag of the versioned fragment text (``embedding_text``): False (the
            default, ``index_merged_genizah.INDEX_SCHOLARLY_FRAMES``: the v10 embedder was trained on the text without
            them) leaves the KTIV scholarly frames out, True lists them (the on-arm). Documents of another kind
            (bibliography pages) have no frames and ignore it (:func:`document_scholarly_frames`). Must agree with
            ``index_meta["text_function"]`` when that block is present.
        :type include_scholarly_frames: bool
        :raises ValueError: When ``text_version`` is set without an ``index_meta`` built for it (text version or
            render flag).
        :raises TypeError: When ``text_version`` is set and the model has no batched text path.
        :raises IndexMetaMismatchError: When the index exists with another model, revision, text version or render
            flag.
        """
        if text_version is not None:
            check_text_version(text_version)
            if not index_meta or index_meta.get("text_representation_version") != text_version:
                raise ValueError(f"text_version={text_version!r} needs an index_meta built for that text version")
            if (index_meta.get("text_function") is not None
                    and index_scholarly_frames(index_meta) != include_scholarly_frames):
                raise ValueError(f"include_scholarly_frames={include_scholarly_frames} but index_meta was built with "
                                 f"text_function.scholarly_frames={index_scholarly_frames(index_meta)}")
            if not hasattr(embedding_model, "get_text_embeddings_batch"):
                raise TypeError("a versioned write needs a batched text embedder (get_text_embeddings_batch)")
        self.embedding_model = embedding_model
        self.es_client = Elasticsearch(**elasticsearch_config)
        self.index_name = index_name
        self.embedding_dims = embedding_dims
        self.index_meta = index_meta
        self.text_version = text_version
        self.include_scholarly_frames = include_scholarly_frames
        # Embed once per text: embedded_text_sha256 -> float32 vector, shared by every batch of this processor.
        self.text_vectors: Dict[str, np.ndarray] = {}
        self.text_vectors_max = TEXT_VECTOR_CACHE_MAX_ENTRIES

        # Setup caching
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.use_cache = False
        if use_cache:
            self.embedding_cache = self._load_embedding_cache()
            self.use_cache = True


        # Statistics. texts_embedded / text_cache_hits count the versioned path's embed-once-per-text: distinct texts
        # sent to the model, and eligible documents that reused a vector (same text earlier in the batch or run).
        self.stats = {
            'processed': 0,
            'cached_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'ineligible': 0,
            'texts_embedded': 0,
            'text_cache_hits': 0,
        }

        # Create index if it doesn't exist
        self._create_index_if_needed()

    def process_documents(self,
                          documents: List,
                          batch_size: int = 100) -> Dict[str, Any]:
        """Process and index documents in batches - fail fast if anything goes wrong.

        A versioned write (``text_version`` set) gives every document ``semantic_eligible``,
        ``text_representation_version`` and ``embedded_text_sha256`` (the hash of its versioned text, ineligible
        documents included), and a vector only when it is eligible; each distinct text is embedded once
        (:meth:`_get_versioned_embeddings`).

        :param documents: Documents to index (``GenizahDocument`` / ``BibliographyDocument``).
        :type documents: List
        :param batch_size: Documents per embedding and bulk batch.
        :type batch_size: int
        :return: ``processed_count``, ``ineligible_count``, ``total_time`` and ``documents_per_second``.
        :rtype: Dict[str, Any]
        """

        start_time = time.time()
        logger.info(f"Processing {len(documents)} documents in batches of {batch_size}")

        total_processed = 0
        total_ineligible = 0

        # Process in batches
        for batch_start in range(0, len(documents), batch_size):
            batch_end = min(batch_start + batch_size, len(documents))
            batch = documents[batch_start:batch_end]

            logger.info(f"Processing batch {batch_start // batch_size + 1}: documents {batch_start + 1}-{batch_end}")

            # Batched embedding path for text-only models that support it.
            # Versioned writes embed only eligible texts (items[i].eligible False -> no vector).
            items: Optional[List[EmbeddingText]] = None
            if self.text_version is not None:
                items, batch_embeddings = self._get_versioned_embeddings(batch)
            else:
                batch_embeddings = self._get_embeddings_for_batch(batch)

            # Process this batch
            es_documents = []
            for i, doc in enumerate(batch):
                actual_index = batch_start + i
                logger.info(f"Processing document {actual_index + 1}/{len(documents)}: {doc.doc_id}")

                # Generate embedding - fail if this doesn't work
                if batch_embeddings is not None:
                    embedding = batch_embeddings[i]
                else:
                    embedding = self._get_embedding_for_document(document=doc)
                if embedding is None and items is None:
                    raise Exception(f"Failed to generate embedding for document {doc.doc_id}")

                # Validate document has required fields
                if not doc.shelf_mark or doc.shelf_mark.strip() == "":
                    logger.error(f"Document {doc.doc_id} has blank or missing shelf_mark - skipping")
                    continue

                # Convert to ES format
                es_doc = doc.to_elasticsearch_document(embedding=embedding)
                if items is not None:
                    # Ineligible documents have embedding None, so no vector field: shelf-mark / BM25 lookup only.
                    # Their text hash is still written, so a changed text is detectable without a vector.
                    es_doc["semantic_eligible"] = items[i].eligible
                    es_doc["text_representation_version"] = self.text_version
                    es_doc[TEXT_HASH_FIELD] = items[i].text_sha256
                    total_ineligible += not items[i].eligible
                # Prefer the cross-store page_uuid as the ES _id so the doc
                # joins cleanly to KG triplets and re-indexing is idempotent.
                # (Also avoids the legacy doc_id collision when two pages share
                # the same printed page number.) Falls back to doc_id for
                # document types without a page_uuid.
                es_documents.append({
                    '_index': self.index_name,
                    '_id': es_document_id(doc) or f"doc_{actual_index}",
                    '_source': es_doc
                })

            # Index this batch immediately - fail if this doesn't work
            if es_documents:
                logger.info(f"Indexing batch of {len(es_documents)} documents")
                self._bulk_index_documents_fail_fast(es_documents)
                total_processed += len(es_documents)
                logger.info(f"Successfully indexed batch. Total so far: {total_processed}/{len(documents)}")

        # Save cache
        if self.use_cache:
            self._save_embedding_cache()

        total_time = time.time() - start_time
        self.stats['ineligible'] += total_ineligible
        logger.info(f"Processing complete: {total_processed}/{len(documents)} documents in {total_time:.1f}s"
                    + (f" ({total_ineligible} without a vector; so far {self.stats['texts_embedded']} texts "
                       f"embedded, {self.stats['text_cache_hits']} vectors reused)"
                       if self.text_version is not None else ""))

        return {
            'processed_count': total_processed,
            'ineligible_count': total_ineligible,
            'total_time': total_time,
            'documents_per_second': len(documents) / max(total_time, 0.001)
        }

    def _bulk_index_documents_fail_fast(self, documents: List[Dict[str, Any]]) -> None:
        """Bulk index documents - fail immediately if anything goes wrong."""

        # Preprocess dates
        for doc in documents:
            if '_source' in doc:
                doc['_source'] = self._preprocess_document_dates(doc['_source'])

        # Do the bulk index
        try:
            success, failed = bulk(
                client=self.es_client,
                actions=documents,
                refresh='wait_for'
            )
            if failed:
                logger.warning(f"Failed to index {len(failed)} documents")
                for i, fail in enumerate(failed):
                    try:
                        idx_err = fail.get('index', {})
                        err = idx_err.get('error', {})
                        reason = err.get('reason', '')
                        etype = err.get('type', '')
                        caused_by = err.get('caused_by', {})
                        cid = idx_err.get('_id', 'unknown_id')
                        status = idx_err.get('status', 'unknown_status')
                        logger.error(
                            f"Bulk error #{i+1} id={cid} status={status} type={etype} reason={reason} caused_by={caused_by}"
                        )
                    except Exception:
                        logger.error(f"Bulk error (unparsable): {fail}")

            logger.info(f"Successfully bulk indexed {success} documents")
        except Exception as e:
            # Try to surface detailed bulk errors if available
            try:
                from elasticsearch.helpers import BulkIndexError
                if isinstance(e, BulkIndexError):
                    logger.error(f"BulkIndexError during indexing: {e}")
                    errors = getattr(e, 'errors', [])
                    logger.error(f"{len(errors)} document(s) failed to index.")
                    for i, fail in enumerate(errors):
                        try:
                            idx_err = fail.get('index', {})
                            err = idx_err.get('error', {})
                            reason = err.get('reason', '')
                            etype = err.get('type', '')
                            caused_by = err.get('caused_by', {})
                            cid = idx_err.get('_id', 'unknown_id')
                            status = idx_err.get('status', 'unknown_status')
                            logger.error(
                                f"Bulk error #{i+1} id={cid} status={status} type={etype} reason={reason} caused_by={caused_by}"
                            )
                        except Exception:
                            logger.error(f"Bulk error (unparsable): {fail}")
                else:
                    logger.error(f"Failed to index documents: {e}")
            except Exception:
                logger.error(f"Failed to index documents: {e}")



    def search_documents(self,
                         query_text: str,
                         query_embedding: Optional[np.ndarray] = None,
                         filters: Optional[Dict[str, Any]] = None,
                         size: int = 10) -> Dict[str, Any]:
        """Search documents using text and/or semantic similarity.

        :param query_text: Text query for lexical search
        :type query_text: str
        :param query_embedding: Embedding vector for semantic search
        :type query_embedding: Optional[np.ndarray]
        :param filters: Additional filters to apply
        :type filters: Optional[Dict[str, Any]]
        :param size: Number of results to return
        :type size: int
        :return: Search results
        :rtype: Dict[str, Any]

        Example:
             results = processor.search_documents(
            ...     query_text="Talmud fragment",
            ...     size=5
            ... )
            results['total'] >= 0
            True
        """
        search_body = {
            "size": size,
            "query": {
                "bool": {
                    "must": [],
                    "filter": []
                }
            }
        }

        # Add text search
        if query_text:
            text_query = {
                "multi_match": {
                    "query": query_text,
                    "fields": [
                        "description^2",
                        "alt_titles^2",
                        "scholarly_frames^2",
                        "subjects^1.5",
                        "genres",
                        "full_text_content",
                        "transcriptions.text",
                        "translations.text"
                    ],
                    "type": "best_fields"
                }
            }
            search_body["query"]["bool"]["must"].append(text_query)

        # Add semantic search
        if query_embedding is not None:
            semantic_query = {
                "script_score": {
                    # semantic_eligible=false documents have no vector; cosineSimilarity fails on them
                    "query": {"exists": {"field": "embedding_vector"}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding_vector') + 1.0",
                        "params": {"query_vector": query_embedding.tolist()}
                    }
                }
            }

            if query_text:
                # Combine text and semantic search
                search_body["query"] = {
                    "bool": {
                        "should": [
                            {"bool": search_body["query"]["bool"]},
                            semantic_query
                        ]
                    }
                }
            else:
                search_body["query"] = semantic_query

        # Add filters
        if filters:
            filter_queries = self._build_filter_queries(filters=filters)
            search_body["query"]["bool"]["filter"].extend(filter_queries)

        # Execute search
        response = self.es_client.search(index=self.index_name, body=search_body)

        return {
            'hits': [
                {
                    'id': hit['_id'],
                    'score': hit['_score'],
                    'source': hit['_source']
                }
                for hit in response['hits']['hits']
            ],
            'total': response['hits']['total']['value'],
            'max_score': response['hits']['max_score']
        }

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by its ID.

        :param doc_id: Document ID to retrieve
        :type doc_id: str
        :return: Document data or None if not found
        :rtype: Optional[Dict[str, Any]]

        Example:
            doc = processor.get_document_by_id(doc_id="MS-TS-F-00001")
            doc is not None
            True
        """
        response = self.es_client.get(index=self.index_name, id=doc_id, ignore=[404])

        if response['found']:
            return response['_source']
        return None

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Elasticsearch index.

        :return: Index statistics
        :rtype: Dict[str, Any]

        Example:
                stats = processor.get_index_stats()
                'document_count' in stats
                 True
        """
        count_response = self.es_client.count(index=self.index_name)
        stats_response = self.es_client.indices.stats(index=self.index_name)

        index_stats = stats_response['indices'][self.index_name]['total']

        return {
            'document_count': count_response['count'],
            'index_size_bytes': index_stats['store']['size_in_bytes'],
            'index_size_mb': index_stats['store']['size_in_bytes'] / (1024 * 1024)
        }

    def filter_existing_ids(self, ids: List[str]) -> set:
        """Return the subset of the given document ids already in the index.

        Lets callers resume an interrupted run without re-embedding documents
        that were already indexed (ids are stable across runs).

        :param ids: Candidate document ids.
        :type ids: List[str]
        :return: Ids present in the index.
        :rtype: set
        """
        if not ids or not self.es_client.indices.exists(index=self.index_name):
            return set()
        resp = self.es_client.mget(index=self.index_name, ids=list(ids), source=False)
        return {d["_id"] for d in resp["docs"] if d.get("found")}

    # ===== PRIVATE METHODS =====
    def _get_embeddings_for_batch(self, batch: List) -> Optional[np.ndarray]:
        """Embed a whole batch at once when the model supports it.

        Uses the embedding model's ``get_text_embeddings_batch`` (text-only
        models such as ``QwenTextEmbedding``) so encoding is batched instead of
        one forward pass per document. Returns ``None`` when the model has no
        batched path (or is image-based), in which case the caller falls back
        to the per-document ``_get_embedding_for_document``.

        :param batch: List of document objects with ``create_text_representation``.
        :type batch: List
        :return: Array of shape ``(len(batch), dims)`` or ``None``.
        :rtype: Optional[np.ndarray]
        """
        if not hasattr(self.embedding_model, 'get_text_embeddings_batch'):
            return None
        if getattr(self.embedding_model, 'image_only', False):
            return None
        texts = [doc.create_text_representation() for doc in batch]
        return self.embedding_model.get_text_embeddings_batch(texts)

    def _get_versioned_embeddings(self, batch: List) -> Tuple[List[EmbeddingText], List[Optional[np.ndarray]]]:
        """Embed a batch from its versioned text: only eligible texts, and each distinct text once.

        An empty or boilerplate-only text is never embedded; its slot stays ``None``. Texts are keyed on their
        ``text_sha256``: a text already embedded by this processor (earlier in the batch, or in an earlier batch while
        :attr:`text_vectors` holds fewer than :attr:`text_vectors_max` entries) reuses that float32 vector, so
        documents sharing one catalogue text get one vector and the model sees the text once.

        :param batch: Documents (``GenizahDocument`` / ``BibliographyDocument``).
        :type batch: List
        :return: (one ``EmbeddingText`` per document: text, eligibility, hash; one vector or ``None`` per document).
        :rtype: Tuple[List[EmbeddingText], List[Optional[np.ndarray]]]
        """
        items = [embedding_text(doc, self.text_version,
                                include_scholarly_frames=document_scholarly_frames(doc, self.include_scholarly_frames))
                 for doc in batch]
        batch_vectors: Dict[str, np.ndarray] = {}
        pending: Dict[str, str] = {}    # text_sha256 -> text, first occurrence order
        for item in items:
            if not item.eligible or item.text_sha256 in batch_vectors or item.text_sha256 in pending:
                continue
            if item.text_sha256 in self.text_vectors:
                batch_vectors[item.text_sha256] = self.text_vectors[item.text_sha256]
            else:
                pending[item.text_sha256] = item.text
        if pending:
            embedded = self.embedding_model.get_text_embeddings_batch(list(pending.values()))
            for text_hash, vector in zip(pending, embedded):
                vector = np.array(vector, dtype=np.float32)    # a copy: the cache never pins a whole batch array
                batch_vectors[text_hash] = vector
                if len(self.text_vectors) < self.text_vectors_max:
                    self.text_vectors[text_hash] = vector
        eligible = sum(item.eligible for item in items)
        self.stats['texts_embedded'] += len(pending)
        self.stats['text_cache_hits'] += eligible - len(pending)
        return items, [batch_vectors[item.text_sha256] if item.eligible else None for item in items]

    def _preprocess_document_dates(self, es_doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess document dates by normalizing all dates as strings.
        Much simpler approach - all dates stored as keywords for consistent handling.

        :param es_doc: Elasticsearch document dictionary
        :type es_doc: Dict[str, Any]
        :return: Processed document with normalized string dates.
        :rtype: Dict[str, Any]
        """
        # Process all date fields as strings
        for date_field in ['date_start', 'date_end', 'date_display']:
            if date_field in es_doc and es_doc[date_field]:
                original_date = es_doc[date_field]

                if date_field == 'date_display' and '/' in str(original_date):
                    # Handle ranges like "0932-09-03/0933-09-22" or "888/987"
                    parts = str(original_date).split('/')
                    cleaned_parts = [self._normalize_date_string(part) for part in parts]
                    es_doc[date_field] = '/'.join(cleaned_parts)
                else:
                    # Single date - just normalize
                    es_doc[date_field] = self._normalize_date_string(original_date)

                logger.debug(f"Normalized {date_field}: {original_date} -> {es_doc[date_field]}")

        return es_doc

    @staticmethod
    def _normalize_date_string(date_str) -> str:
        """Normalize date string by removing leading zeros.
        :param date_str: Date string to normalize (e.g., 0932-01-01)
        :type date_str: str
        :return: Normalized date string (e.g., 932-01-01)
        :rtype: str
        """
        if not date_str:
            return date_str

        try:
            date_str = str(date_str)
            # Handle leading zeros: 0932-01-01 -> 932-01-01
            if date_str.startswith('0') and len(date_str) > 4:
                normalized = date_str.lstrip('0')
                if normalized.startswith('-'):
                    normalized = '0' + normalized
                return normalized
            return date_str
        except (ValueError, AttributeError):
            return date_str

    def _create_index_if_needed(self) -> None:
        """Create Elasticsearch index with string-based date fields for historical documents.

        For a versioned write into an existing index, first applies the writer guard
        (:func:`assert_index_writable`).

        :raises IndexMetaMismatchError: When an existing index's ``_meta`` does not match this run.
        """
        if self.es_client.indices.exists(index=self.index_name):
            if self.text_version is not None:
                assert_index_writable(self.es_client, self.index_name, self.index_meta)
            logger.info(f"Index {self.index_name} already exists")
            return

        mapping: Dict[str, Any] = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "multilingual": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "asciifolding"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    # Core fields
                    "doc_id": {"type": "keyword"},
                    # Multi-source merge provenance (from merged_shelfmarks.jsonl).
                    "canonical_id": {"type": "keyword"},
                    "sources_present": {"type": "keyword"},
                    "image_preferred_source": {"type": "keyword"},
                    "has_ktiv_images": {"type": "boolean"},
                    "has_ktiv_transcription": {"type": "boolean"},
                    "ktiv_iiif_manifest_url": {"type": "keyword"},
                    # Bodleian direct scrape (TEI catalogue + IIIF masters).
                    "has_bodleian_images": {"type": "boolean"},
                    "bodleian_catalogue_url": {"type": "keyword"},
                    # Cross-store join keys (shared with the Neo4j KG).
                    "book_uuid": {"type": "keyword"},
                    "page_uuid": {"type": "keyword"},
                    "page_seq": {"type": "integer"},
                    "doi": {"type": "keyword"},
                    # Outbound links for the web app. Populated from the
                    # metadata's external_ids block, which never feeds book_key.
                    "oclc": {"type": "keyword"},
                    "worldcat_url": {"type": "keyword"},
                    "shelf_mark": {"type": "keyword"},
                    "description": {"type": "text", "analyzer": "multilingual"},
                    **ALT_TITLES_MAPPING,
                    **SCHOLARLY_FRAMES_MAPPING,
                    **SUBJECTS_MAPPING,
                    **GENRES_MAPPING,
                    "full_text_content": {"type": "text", "analyzer": "multilingual"},

                    # All dates as strings - handles any historical date
                    "date_start": {
                        "type": "keyword",  # String dates work for any year
                        "fields": {
                            "text": {"type": "text"}
                        }
                    },
                    "date_end": {
                        "type": "keyword",  # String dates work for any year
                        "fields": {
                            "text": {"type": "text"}
                        }
                    },
                    "date_display": {
                        "type": "keyword",  # Can handle ranges and any format
                        "fields": {
                            "text": {
                                "type": "text",
                                "analyzer": "multilingual"
                            }
                        }
                    },
                    "date_certainty": {"type": "keyword"},
                    "century": {"type": "keyword"},

                    # Classification fields
                    "document_type": {"type": "keyword"},
                    "main_language": {"type": "keyword"},
                    "other_languages": {"type": "keyword"},
                    "script_type": {"type": "keyword"},
                    "content_quality": {"type": "keyword"},
                    "source_collection": {"type": "keyword"},
                    "language": {"type": "keyword"},

                    # Bibliography-specific fields
                    "author": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "authors": {"type": "keyword"},
                    "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "isbn": {"type": "keyword"},
                    "subject_keywords": {"type": "keyword"},
                    "page_number": {"type": "integer"},
                    "extracted_page_number": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            # Roman-numeral pages ("vii") must not fail the doc.
                            "numeric": {"type": "long", "ignore_malformed": True}
                        }
                    },
                    "shelf_marks_mentioned": {
                        "type": "keyword"  # Store as keyword array to avoid dynamic field creation
                    },

                    # Named entities
                    "named_entities": {
                        "type": "object",
                        "properties": {
                            "persons": {"type": "keyword"},
                            "places": {"type": "keyword"},
                            "organizations": {"type": "keyword"},
                            "dates": {"type": "keyword"}
                        }
                    },

                    # Nested structures
                    "transcriptions": {
                        "type": "nested",
                        "properties": {
                            "text": {"type": "text", "analyzer": "multilingual"},
                            "editor": {"type": "keyword"},
                            "line_count": {"type": "integer"},
                            "language": {"type": "keyword"}
                        }
                    },
                    "translations": {
                        "type": "nested",
                        "properties": {
                            "text": {"type": "text", "analyzer": "multilingual"}
                        }
                    },
                    "bibliography": {
                        "type": "nested",
                        "properties": {
                            "citation": {"type": "text"},
                            "location": {"type": "text"},
                            "relations": {"type": "keyword"},
                            "url": {"type": "keyword"},
                            "source": {"type": "keyword"},
                            "title": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 512}}},
                            "authors": {"type": "keyword"},
                            "year": {"type": "keyword"}
                        }
                    },

                    # Images
                    "image_urls": {"type": "keyword"},
                    "primary_image_index": {"type": "integer"},
                    "has_images": {"type": "boolean"},
                    "actual_image_url": {"type": "keyword"},

                    # Embedding vector. Optional per document: semantic_eligible=false
                    # documents are written without it (see TEXT_VERSION_MAPPING).
                    "embedding_vector": {
                        "type": "dense_vector",
                        "dims": self.embedding_dims,
                        "index": True,
                        "similarity": "cosine"
                    },
                    **TEXT_VERSION_MAPPING,

                    # Other fields
                    "completeness_score": {"type": "float"},
                    "indexed_at": {"type": "date"},  # Keep this as date since it's modern
                    
                    # Joins data
                    "joins_data": {
                        "type": "object",
                        "properties": {
                            "joinedManuscripts": {
                                "type": "nested",
                                "properties": {
                                    "shelfmark": {"type": "keyword"},
                                    "index": {"type": "integer"},
                                    "source": {"type": "keyword"}
                                }
                            },
                            "mainShelfmark": {"type": "keyword"},
                            "source": {"type": "keyword"},
                            "metadata": {
                                "type": "object",
                                "properties": {
                                    "pageUrl": {"type": "keyword"},
                                    "extractedAt": {"type": "date"},
                                    "extractionMethod": {"type": "keyword"}
                                }
                            }
                        }
                    }
                }
            }
        }
        if self.index_meta:
            mapping["mappings"]["_meta"] = self.index_meta

        self.es_client.indices.create(index=self.index_name, body=mapping)
        logger.info(f"Created index {self.index_name} with string-based historical dates")

    def _get_embedding_for_document(self, document) -> Optional[np.ndarray]:
        """Generate or retrieve cached embedding for a document.

        :param document: GenizahDocument to get embedding for
        :type document: GenizahDocument
        :return: Embedding vector or None if generation fails
        :rtype: Optional[np.ndarray]
        """
        # Check cache first
        if self.use_cache:
            cache_key = document.get_embedding_cache_key()
            if cache_key in self.embedding_cache:
                self.stats['cached_hits'] += 1
                return self.embedding_cache[cache_key]

        # Check if using image-only mode - skip text representation
        is_image_only = hasattr(self.embedding_model, 'image_only') and self.embedding_model.image_only
        
        if is_image_only:
            # For image-only mode, pass doc_id as text for better caching (embedding model will ignore it for processing)
            text_representation = document.doc_id or ""
        else:
            # Generate new embedding using the document's text representation method
            text_representation = document.create_text_representation()
        
        embedding = self.embedding_model.get_embeddings(
            image=document.image,
            text=text_representation,
            use_cache=False
        )

        if embedding is not None and self.use_cache:
            # Cache the new embedding
            cache_key = document.get_embedding_cache_key()
            self.embedding_cache[cache_key] = embedding
            self.stats['cache_misses'] += 1

        return embedding

    def _build_filter_queries(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build Elasticsearch filter queries from filter dictionary.

        :param filters: Filter criteria
        :type filters: Dict[str, Any]
        :return: List of Elasticsearch filter queries
        :rtype: List[Dict[str, Any]]
        """
        filter_queries = []

        if 'document_type' in filters:
            filter_queries.append({"term": {"document_type": filters['document_type']}})

        if 'language' in filters:
            filter_queries.append({"term": {"main_language": filters['language']}})

        if 'source_collection' in filters:
            filter_queries.append({"term": {"source_collection": filters['source_collection']}})

        if 'date_start' in filters or 'date_end' in filters:
            date_range = {}
            if 'date_start' in filters:
                date_range['gte'] = filters['date_start']
            if 'date_end' in filters:
                date_range['lte'] = filters['date_end']
            filter_queries.append({"range": {"date_start": date_range}})

        if 'has_transcriptions' in filters:
            if filters['has_transcriptions']:
                filter_queries.append({"exists": {"field": "transcriptions"}})
            else:
                filter_queries.append({"bool": {"must_not": [{"exists": {"field": "transcriptions"}}]}})

        if 'min_completeness' in filters:
            filter_queries.append({"range": {"completeness_score": {"gte": filters['min_completeness']}}})

        return filter_queries

    def _load_embedding_cache(self) -> Dict[str, np.ndarray]:
        """Load embedding cache from disk.

        :return: Dictionary of cached embeddings
        :rtype: Dict[str, np.ndarray]
        """
        cache_file = self.cache_dir / "embeddings.pkl"

        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            logger.info(f"Loaded {len(cache)} cached embeddings")
            return cache

        return {}

    def _save_embedding_cache(self) -> None:
        """Save embedding cache to disk."""
        cache_file = self.cache_dir / "embeddings.pkl"

        with open(cache_file, 'wb') as f:
            pickle.dump(self.embedding_cache, f)

        logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate.

        :return: Cache hit rate as float between 0.0 and 1.0
        :rtype: float
        """
        total_requests = self.stats['cached_hits'] + self.stats['cache_misses']
        if total_requests == 0:
            return 0.0
        return self.stats['cached_hits'] / total_requests