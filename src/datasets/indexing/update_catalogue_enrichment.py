#!/usr/bin/env python3
# File name: update_catalogue_enrichment.py
# Date: 10/9/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Bring an existing fragment index in line with the current merged file and document model, in place.

On 2026-10-09 :meth:`GenizahDocument.from_merged_format` started filling a ``scholarly_frames`` field (KTIV scholarly
identifications: each entry's frame and specific work title, spellings that differ only in brackets, quote marks,
spacing or case kept once), which the embedding text lists after the alternate titles on its ``Alternate titles:``
line (unless rendered with ``include_scholarly_frames=False``), a ``subjects`` field (KTIV catalogue subjects and PGP
tags), which also reaches the embedding text as a ``Subjects:`` line, and a ``genres`` field (KTIV scholarly domain
levels and generic work titles), which reaches ``full_text_content`` and the index but never the embedding text; a
re-merge can also change a record's ``description`` or ``alt_titles`` (e.g. which catalogue label the merge picks). An
index built before that (``genizah_merged_v9``) holds the old values and vectors embedded from the old text. This
updater streams the merged file (``--merged``) through the current document model and compares every record already in
the index (``--index``, default :data:`DEFAULT_INDEX_NAME`) on the catalogue fields :data:`CATALOGUE_FIELDS`
(``description``, ``alt_titles``, ``scholarly_frames``, ``subjects``, ``genres``), and on its embedding text
(``embedding_text(doc, version)``) when the document carries ``embedded_text_sha256``:

* nothing differs: ``unchanged``, nothing is written;
* the text changed and is eligible under the index's text version: ``update/reembed`` -- the catalogue fields,
  ``full_text_content``, ``embedding_vector`` (the embedding of that text), ``semantic_eligible: true``,
  ``text_representation_version``, ``embedded_text_sha256`` and ``catalogue_fields_updated_at``. A document the index
  held without a vector gains one: an *eligibility flip*, the headline count;
* otherwise, when a catalogue field differs: ``update/fields_only`` -- the catalogue fields, ``full_text_content`` and
  ``catalogue_fields_updated_at``, no vector. A changed document whose text is not eligible but which holds a vector is
  refused instead (``refused:would_lose_vector``; enrichment only adds text, so this should not happen);
* a document whose vector comes from a machine-generated description (``description_source:
  generated_from_consensus``, :mod:`update_generated_descriptions`) gets the catalogue fields and
  ``full_text_content`` only (``generated``), never a new vector or eligibility flag;
* a record that is not in the index is counted (``missing``) and never created.

Without ``embedded_text_sha256`` (every document an indexer wrote, so all of v9 today) the text counts as changed
exactly when a catalogue field differs, ``genres`` included; once a document carries the hash, it is re-embedded exactly
when the hash of its new text differs, so later runs also catch text changes outside the catalogue fields, and a
``genres``-only change is ``update/fields_only``. Whether the new text gets a vector is the rule of the index's text
version (``embedding_text(doc, version).eligible``). The frozen ``semantic_v3`` rule counts a ``Subjects:`` line like
any other line, as content and towards the 40-character floor, so new subjects alone can flip a record there;
``semantic_v4`` counts it as neither (``semantic_text.semantic_v4_ineligible_reason``), so a record flips only through
new catalogue content such as a scholarly identification.

Guards, all before the merged file is streamed:

* the live index names are refused unless ``--i-know-this-is-live`` (``index_merged_genizah.check_index_name``);
* the index must exist and its ``_meta`` must match the embedder and the text version on ``GUARDED_META_KEYS``
  (``IndexMetaMismatchError``, the writer guard of ``elastic_index_genizah``);
* a real run also checks that the loaded embedder is the one the guard checked and re-embeds the ``_meta`` string
  canary (refused at cosine <= :data:`CANARY_COSINE_FLOOR`); every vector's size is checked against ``_meta.dims``
  before it is written.

Mapping fields the index lacks (``alt_titles``, ``scholarly_frames``, ``subjects``, ``genres``,
``embedded_text_sha256``, ``catalogue_fields_updated_at``) are added with ``put_mapping`` (properties only) before the
first write; a dry run only reports them. ``_meta`` is written only by ``--refresh-document-canary`` (below).

Idempotent and resumable: a document this updater wrote is ``unchanged`` on the next run, so an interrupted run is
simply started again; ``--skip N`` skips the first N merged lines (the progress log names a safe value).

Generated descriptions: a document holding a generated-description vector keeps it here even when its enriched
catalogue text has become eligible, while :mod:`update_generated_descriptions` refuses a document that holds a catalogue
vector; run this updater first and the catalogue text wins wherever it is now eligible.

The verifier's self-consistency check (``verify_reembedded_indexes``) re-embeds ``create_text_representation(doc,
version)`` of the merged file, which is the text embedded here, so it needs no change. Its document-canary check compares
``_meta.document_canary.text`` with that record's current text, which a document-model change breaks until the canary
is renewed: the summary says whether the current model changed that text (``document_canary``), and
``--refresh-document-canary`` re-embeds the same record's current text after the writes and stores it in
``_meta.document_canary`` (:func:`refresh_document_canary`, the only ``_meta`` write here; every other key is kept).

Usage (repo root; Elasticsearch credentials in ``.env``, and ``HF1_TOKEN`` for a real run)::

    python -m src.datasets.indexing.update_catalogue_enrichment --dry-run [--report enrichment_dry_run.json]
    python -m src.datasets.indexing.update_catalogue_enrichment [--index genizah_merged_v9] [--skip N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from elasticsearch.helpers import bulk

from src.datasets.consensus.describe_from_reads import DESCRIPTION_SOURCE
from src.datasets.document_models.genizah_document import GenizahDocument
from src.datasets.indexing.elastic_index_genizah import (
    ALT_TITLES_MAPPING,
    GENRES_MAPPING,
    SCHOLARLY_FRAMES_MAPPING,
    SUBJECTS_MAPPING,
    IndexMetaMismatchError,
    es_document_id,
    index_scholarly_frames,
    read_index_meta,
)
from src.datasets.indexing.index_merged_genizah import (
    DEFAULT_INDEX_NAME,
    DEFAULT_MERGED,
    LIVE_INDEX_NAMES,
    iter_merged_documents,
)
from src.datasets.indexing.update_generated_descriptions import (
    CANARY_COSINE_FLOOR,
    DEFAULT_BATCH_SIZE,
    MGET_CHUNK,
    VECTOR_FIELD,
    canary_cosine,
    check_index,
    default_es_client,
    expected_meta,
)
from src.embeddings.text_representation import (
    DEFAULT_TEXT_VERSION,
    TEXT_VERSIONS,
    check_text_version,
    embedding_text,
)

logger = logging.getLogger(__name__)

TEXT_HASH_FIELD = "embedded_text_sha256"
UPDATED_AT_FIELD = "catalogue_fields_updated_at"
#: Compared with the index and written on every update (with ``full_text_content``, which is built from them).
CATALOGUE_FIELDS = ("description", "alt_titles", "scholarly_frames", "subjects", "genres")
#: Fields this updater makes sure the mapping has (``description`` is in every merged-index mapping).
ENRICHMENT_MAPPING: Dict[str, Any] = {
    **ALT_TITLES_MAPPING,
    **SCHOLARLY_FRAMES_MAPPING,
    **SUBJECTS_MAPPING,
    **GENRES_MAPPING,
    TEXT_HASH_FIELD: {"type": "keyword"},
    UPDATED_AT_FIELD: {"type": "date"},
}
#: ``_source`` fields read per document.
STATE_FIELDS = [*CATALOGUE_FIELDS, "semantic_eligible", "description_source", "text_representation_version",
                TEXT_HASH_FIELD]

MISSING = "missing"
GENERATED = "generated"
UNCHANGED = "unchanged"
REEMBED = "update/reembed"
FIELDS_ONLY = "update/fields_only"
REFUSED_LOSE_VECTOR = "refused:would_lose_vector"
DECISIONS = (MISSING, GENERATED, UNCHANGED, REEMBED, FIELDS_ONLY, REFUSED_LOSE_VECTOR)
#: Decisions that write to the index.
WRITES = (REEMBED, FIELDS_ONLY, GENERATED)
FLIPS = "flips"
ELIGIBILITY_DRIFT = "eligibility_drift"
#: Why a document is updated or refused: a catalogue field differs, or (stored hash only) just its text.
CHANGE_REASONS = (*CATALOGUE_FIELDS, "text_only")
EXAMPLES_PER_CATEGORY = 5
PROGRESS_EVERY = 2000


@dataclass
class EnrichedRecord:
    """One merged record under the current document model: what the index should hold for it.

    :param doc_id: Index ``_id`` (:func:`es_document_id`, the record's ``canonical_id``).
    :param description: ``description``.
    :param alt_titles: ``alt_titles``.
    :param scholarly_frames: ``scholarly_frames`` (listed after the alternate titles on the ``Alternate titles:`` line
        of the embedding text).
    :param subjects: ``subjects`` (also the ``Subjects:`` line of the embedding text).
    :param genres: ``genres`` (never in the embedding text).
    :param full_text_content: ``create_full_text_content()``.
    :param text: Embedding text under the run's text version.
    :param eligible: Whether that text gets a vector.
    """

    doc_id: Optional[str]
    description: str
    alt_titles: List[str]
    scholarly_frames: List[str]
    subjects: List[str]
    genres: List[str]
    full_text_content: str
    text: str
    eligible: bool

    @property
    def text_sha256(self) -> str:
        """SHA-256 of the embedding text (what ``embedded_text_sha256`` stores after a re-embed).

        :return: Hex digest.
        :rtype: str
        """
        return text_sha256(self.text)


def text_sha256(text: str) -> str:
    """SHA-256 hex digest of a text (UTF-8).

    :param text: Text.
    :return: Hex digest.
    :rtype: str
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def enriched_record(doc: GenizahDocument, text_version: str, include_scholarly_frames: bool = True) -> EnrichedRecord:
    """The catalogue fields, full text and embedding text of one document under the current document model.

    :param doc: Document built from a merged record.
    :param text_version: The index's text version.
    :param include_scholarly_frames: Render the KTIV scholarly frames into the text (the index's own setting,
        ``index_scholarly_frames(_meta)``; a pre-v10 index was rendered with them).
    :return: The record's target state.
    :rtype: EnrichedRecord
    """
    item = embedding_text(doc, text_version, include_scholarly_frames=include_scholarly_frames)
    return EnrichedRecord(doc_id=es_document_id(doc), description=doc.description or "",
                          alt_titles=list(doc.alt_titles), scholarly_frames=list(doc.scholarly_frames),
                          subjects=list(doc.subjects), genres=list(doc.genres),
                          full_text_content=doc.create_full_text_content(), text=item.text, eligible=item.eligible)


def as_list(value: Any) -> List[Any]:
    """An indexed array field as a list (absent = empty, a scalar = one item).

    :param value: ``_source`` value.
    :return: List.
    :rtype: List[Any]
    """
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def indexed_catalogue_fields(state: Mapping[str, Any]) -> Tuple[str, List[Any], List[Any], List[Any], List[Any]]:
    """``(description, alt_titles, scholarly_frames, subjects, genres)`` as the index holds them (a missing field
    counts as empty).

    :param state: Document ``_source`` subset (:func:`index_states`).
    :return: The five fields, in :data:`CATALOGUE_FIELDS` order.
    :rtype: Tuple[str, List[Any], List[Any], List[Any], List[Any]]
    """
    return (state.get("description") or "", as_list(state.get("alt_titles")), as_list(state.get("scholarly_frames")),
            as_list(state.get("subjects")), as_list(state.get("genres")))


def catalogue_fields(record: EnrichedRecord) -> Tuple[str, List[str], List[str], List[str], List[str]]:
    """``(description, alt_titles, scholarly_frames, subjects, genres)`` under the current document model.

    :param record: Target state.
    :return: The five fields, in :data:`CATALOGUE_FIELDS` order.
    :rtype: Tuple[str, List[str], List[str], List[str], List[str]]
    """
    return (record.description, list(record.alt_titles), list(record.scholarly_frames), list(record.subjects),
            list(record.genres))


def decide(state: Optional[Mapping[str, Any]], record: EnrichedRecord) -> str:
    """What to do with one document: one of :data:`DECISIONS`.

    The text counts as changed when the stored ``embedded_text_sha256`` differs from the new text's hash, or, for a
    document without the hash, when a catalogue field differs (``genres`` included, although it is never embedded).
    With the hash, a change that leaves the text alone (e.g. ``genres`` only) is a field update.

    :param state: The document's fields in the index (:func:`index_states`), None when it is not indexed.
    :param record: Its target state.
    :return: ``missing``, ``generated``, ``unchanged``, ``update/reembed``, ``update/fields_only`` or
        ``refused:would_lose_vector``.
    :rtype: str
    """
    if state is None:
        return MISSING
    fields_differ = indexed_catalogue_fields(state) != catalogue_fields(record)
    if state.get("description_source") == DESCRIPTION_SOURCE:
        return GENERATED if fields_differ else UNCHANGED
    stored_hash = state.get(TEXT_HASH_FIELD)
    text_changed = stored_hash != record.text_sha256 if stored_hash else fields_differ
    if record.eligible and text_changed:
        return REEMBED
    if not (fields_differ or text_changed):
        return UNCHANGED
    if not record.eligible and state.get("semantic_eligible") is True:
        return REFUSED_LOSE_VECTOR
    return FIELDS_ONLY if fields_differ else UNCHANGED


def change_reasons(state: Mapping[str, Any], record: EnrichedRecord) -> List[str]:
    """The catalogue fields whose indexed value differs from the current one (``["text_only"]`` when none does).

    :param state: The document's fields in the index.
    :param record: Its target state.
    :return: Names from :data:`CHANGE_REASONS`.
    :rtype: List[str]
    """
    pairs = zip(CATALOGUE_FIELDS, indexed_catalogue_fields(state), catalogue_fields(record))
    return [name for name, indexed, current in pairs if indexed != current] or ["text_only"]


def is_flip(state: Optional[Mapping[str, Any]], decision: str) -> bool:
    """Whether a decision gives a vector to a document the index holds without one.

    :param state: The document's fields in the index.
    :param decision: :func:`decide` output.
    :return: True for an eligibility flip.
    :rtype: bool
    """
    return decision == REEMBED and state is not None and state.get("semantic_eligible") is not True


def eligibility_drift(state: Optional[Mapping[str, Any]], record: EnrichedRecord, decision: str) -> bool:
    """An unchanged document whose indexed eligibility differs from the current rule (expected never).

    :param state: The document's fields in the index.
    :param record: Its target state.
    :param decision: :func:`decide` output.
    :return: True when the index flag and the current eligibility disagree.
    :rtype: bool
    """
    return (decision == UNCHANGED and state is not None and state.get("description_source") != DESCRIPTION_SOURCE
            and (state.get("semantic_eligible") is True) != record.eligible)


@dataclass
class EnrichmentSummary:
    """Counts and examples of one run.

    :param index: Target index.
    :param text_version: Text version of the run (the index's, by the writer guard).
    :param merged_path: Merged JSONL.
    :param dry_run: Whether nothing is embedded or written.
    :param skip: Merged lines skipped.
    :param limit: Cap on merged records scanned.
    :param scanned: Merged records scanned.
    :param counts: Documents per decision.
    :param flips: Re-embedded documents the index held without a vector.
    :param changed: Updated or refused documents per :data:`CHANGE_REASONS` entry (a document can count for several).
    :param eligibility_drift: Unchanged documents whose indexed eligibility differs from the current rule.
    :param other_text_version: Documents stamped with another ``text_representation_version`` than the run's.
    :param mapping_missing: Enrichment fields the mapping lacked at the start.
    :param document_canary: ``{"doc_id", "text": "unchanged" | "changed" | "not scanned"}`` for ``_meta``'s
        document canary (empty when ``_meta`` has none).
    :param examples: Up to :data:`EXAMPLES_PER_CATEGORY` ids per decision, ``flips`` and ``eligibility_drift``.
    :param updated: Documents written per decision.
    :param canary_cosine: String-canary cosine of a real run.
    """

    index: str
    text_version: str
    merged_path: str
    dry_run: bool
    skip: int = 0
    limit: Optional[int] = None
    scanned: int = 0
    counts: Dict[str, int] = field(default_factory=lambda: dict.fromkeys(DECISIONS, 0))
    flips: int = 0
    changed: Dict[str, int] = field(default_factory=lambda: dict.fromkeys(CHANGE_REASONS, 0))
    eligibility_drift: int = 0
    other_text_version: int = 0
    mapping_missing: List[str] = field(default_factory=list)
    document_canary: Dict[str, Any] = field(default_factory=dict)
    examples: Dict[str, List[str]] = field(default_factory=dict)
    updated: Dict[str, int] = field(default_factory=lambda: dict.fromkeys(WRITES, 0))
    canary_cosine: Optional[float] = None

    def add(self, state: Optional[Mapping[str, Any]], record: EnrichedRecord, decision: str) -> None:
        """Count one scanned document.

        :param state: Its fields in the index.
        :param record: Its target state.
        :param decision: :func:`decide` output.
        """
        self.scanned += 1
        self.counts[decision] += 1
        self._example(decision, record.doc_id)
        if state is not None and decision in (*WRITES, REFUSED_LOSE_VECTOR):
            for reason in change_reasons(state, record):
                self.changed[reason] += 1
        if is_flip(state, decision):
            self.flips += 1
            self._example(FLIPS, record.doc_id)
        if eligibility_drift(state, record, decision):
            self.eligibility_drift += 1
            self._example(ELIGIBILITY_DRIFT, record.doc_id)
        if state is not None and state.get("text_representation_version") not in (None, self.text_version):
            self.other_text_version += 1
        canary = self.document_canary
        if canary and record.doc_id == canary["doc_id"]:
            canary["text"] = "unchanged" if record.text == canary.get("expected_text") else "changed"

    def _example(self, category: str, doc_id: Optional[str]) -> None:
        """Keep the first few ids of a category.

        :param category: Decision, ``flips`` or ``eligibility_drift``.
        :param doc_id: Document id.
        """
        ids = self.examples.setdefault(category, [])
        if len(ids) < EXAMPLES_PER_CATEGORY:
            ids.append(str(doc_id))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready summary.

        :return: Summary (``refused`` totals the ``refused:*`` decisions).
        :rtype: Dict[str, Any]
        """
        canary = {k: v for k, v in self.document_canary.items() if k != "expected_text"}
        return {
            "index": self.index, "text_version": self.text_version, "merged": self.merged_path,
            "dry_run": self.dry_run, "skip": self.skip, "limit": self.limit, "scanned": self.scanned,
            "counts": dict(self.counts),
            "refused": sum(n for name, n in self.counts.items() if name.startswith("refused")),
            "flips": self.flips, "changed": dict(self.changed), "eligibility_drift": self.eligibility_drift,
            "other_text_version": self.other_text_version, "mapping_missing": list(self.mapping_missing),
            "document_canary": canary, "examples": {k: list(v) for k, v in self.examples.items()},
            "updated": dict(self.updated), "canary_cosine": self.canary_cosine,
        }

    def describe(self) -> str:
        """Human-readable summary.

        :return: Multi-line text.
        :rtype: str
        """
        data = self.to_dict()
        lines = [f"index {self.index} (text {self.text_version}), merged {self.merged_path} "
                 f"(skip {self.skip}, limit {self.limit}): scanned {self.scanned}",
                 "  " + " | ".join(f"{name} {n}" for name, n in self.counts.items()),
                 f"  eligibility flips (indexed without a vector, eligible now): {self.flips}",
                 "  updated or refused because of: " + ", ".join(f"{k} {v}" for k, v in self.changed.items()),
                 f"  unchanged documents whose indexed eligibility differs from the current rule: {self.eligibility_drift}",
                 f"  documents stamped with a text version other than {self.text_version}: {self.other_text_version}",
                 "  mapping: " + (f"lacks {', '.join(self.mapping_missing)}"
                                  + (" (a real run adds them)" if self.dry_run else " (added)")
                                  if self.mapping_missing else "complete")]
        canary = data["document_canary"]
        lines.append(f"  document canary {canary['doc_id']}: text {canary['text']}"
                     + (f", {canary['refresh']}" if canary.get("refresh") else "") if canary
                     else "  document canary: none in _meta")
        lines.append("  examples:")
        lines += [f"    {category}: {', '.join(ids)}" for category, ids in self.examples.items()]
        if self.dry_run:
            lines.append("dry run: nothing embedded or written")
        else:
            lines.append("updated: " + ", ".join(f"{name} {n}" for name, n in self.updated.items())
                         + f" (canary cosine {self.canary_cosine:.5f})")
        return "\n".join(lines)


def index_states(es: Any, index: str, ids: Sequence[str]) -> Dict[str, Optional[Dict[str, Any]]]:
    """The indexed fields that decide each document's update (``None`` = not in the index).

    :param es: Elasticsearch client.
    :param index: Index.
    :param ids: Document ids.
    :return: ``{doc_id: _source subset or None}``.
    :rtype: Dict[str, Optional[Dict[str, Any]]]
    :raises RuntimeError: When the cluster reports an error for a document.
    """
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for start in range(0, len(ids), MGET_CHUNK):
        resp = es.mget(index=index, ids=list(ids[start:start + MGET_CHUNK]), source_includes=STATE_FIELDS)
        for doc in resp["docs"]:
            if "error" in doc:
                raise RuntimeError(f"mget {doc['_id']} in {index}: {doc['error']}")
            out[doc["_id"]] = (doc.get("_source") or {}) if doc.get("found") else None
    return out


def missing_mapping_fields(es: Any, index: str) -> List[str]:
    """Enrichment fields (:data:`ENRICHMENT_MAPPING`) the index mapping does not have yet.

    :param es: Elasticsearch client.
    :param index: Index.
    :return: Field names.
    :rtype: List[str]
    """
    mapping = es.indices.get_mapping(index=index)
    properties = (next(iter(mapping.values())).get("mappings") or {}).get("properties") or {}
    return [name for name in ENRICHMENT_MAPPING if name not in properties]


def utc_now() -> str:
    """Current UTC time, ISO 8601 to the second.

    :return: Timestamp.
    :rtype: str
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def update_fields(record: EnrichedRecord, updated_at: str, vector: Optional[Any] = None,
                  text_version: Optional[str] = None, dims: Optional[int] = None) -> Dict[str, Any]:
    """The partial document written for one update.

    :param record: Target state.
    :param updated_at: ``catalogue_fields_updated_at`` value.
    :param vector: Embedding of ``record.text`` (``update/reembed`` only).
    :param text_version: The index's text version (with ``vector``).
    :param dims: Vector size the index expects (checked when given).
    :return: Fields to merge into the document.
    :rtype: Dict[str, Any]
    :raises ValueError: When the vector has the wrong size.
    """
    fields: Dict[str, Any] = {"description": record.description, "alt_titles": list(record.alt_titles),
                              "scholarly_frames": list(record.scholarly_frames), "subjects": list(record.subjects),
                              "genres": list(record.genres), "full_text_content": record.full_text_content,
                              UPDATED_AT_FIELD: updated_at}
    if vector is None:
        return fields
    values = [float(v) for v in np.asarray(vector).reshape(-1)]
    if dims is not None and len(values) != dims:
        raise ValueError(f"{record.doc_id}: vector has {len(values)} dims, index has {dims}")
    fields.update({VECTOR_FIELD: values, "semantic_eligible": True, "text_representation_version": text_version,
                   TEXT_HASH_FIELD: record.text_sha256})
    return fields


def update_action(index: str, doc_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    """A bulk partial-update action.

    :param index: Index.
    :param doc_id: Document ``_id``.
    :param fields: Partial document.
    :return: Action for :func:`elasticsearch.helpers.bulk`.
    :rtype: Dict[str, Any]
    """
    return {"_op_type": "update", "_index": index, "_id": doc_id, "doc": fields}


def write_updates(es: Any, index: str, planned: Sequence[Tuple[EnrichedRecord, str]], embedder: Any,
                  text_version: str, batch_size: int = DEFAULT_BATCH_SIZE, dims: Optional[int] = None
                  ) -> Dict[str, int]:
    """Embed the ``update/reembed`` texts in batches and write every planned update with bulk ``update`` actions.

    Each embedding batch is one bulk call (vector sizes checked before it is sent); the field-only updates
    (``update/fields_only``, ``generated``) follow in one bulk call without a vector.

    :param es: Elasticsearch client.
    :param index: Index.
    :param planned: ``(record, decision)`` pairs, decisions in :data:`WRITES`.
    :param embedder: Embedder (``get_text_embeddings_batch``).
    :param text_version: The index's text version.
    :param batch_size: Documents per embedding / bulk batch.
    :param dims: Vector size the index expects.
    :return: Documents written per decision.
    :rtype: Dict[str, int]
    :raises ValueError: When a vector has the wrong size.
    """
    written = dict.fromkeys(WRITES, 0)
    to_embed = [record for record, decision in planned if decision == REEMBED]
    for start in range(0, len(to_embed), batch_size):
        batch = to_embed[start:start + batch_size]
        vectors = embedder.get_text_embeddings_batch([record.text for record in batch])
        updated_at = utc_now()
        actions = [update_action(index, record.doc_id, update_fields(record, updated_at, vector, text_version, dims))
                   for record, vector in zip(batch, vectors, strict=True)]
        bulk(client=es, actions=actions, refresh="wait_for")
        written[REEMBED] += len(actions)
    plain = [(record, decision) for record, decision in planned if decision != REEMBED]
    if plain:
        updated_at = utc_now()
        bulk(client=es, actions=[update_action(index, record.doc_id, update_fields(record, updated_at))
                                 for record, _ in plain], refresh="wait_for")
        for _, decision in plain:
            written[decision] += 1
    return written


def find_merged_document(merged_path: Any, doc_id: str) -> Optional[GenizahDocument]:
    """The merged record with canonical id ``doc_id`` as a document, or ``None`` when the file has none.

    Streams the file and parses only the lines that mention the id, so the lookup is one pass without building every
    document.

    :param merged_path: Merged JSONL.
    :param doc_id: Canonical id.
    :return: The document, or ``None``.
    :rtype: Optional[GenizahDocument]
    """
    with open(merged_path, encoding="utf-8") as fh:
        for line in fh:
            if doc_id in line:
                record = json.loads(line)
                if record.get("canonical_id") == doc_id:
                    return GenizahDocument.from_merged_format(record)
    return None


def refresh_document_canary(es: Any, index: str, merged_path: Any, embedder: Any, text_version: str) -> Dict[str, Any]:
    """Re-embed the ``_meta`` document canary from its record's current text and store it.

    The verifier compares ``_meta.document_canary.text`` with the record's text under the current document model, so
    after a model change the canary has to be renewed deliberately. The record stays the same and every other
    ``_meta`` key is kept; the write is refused when that record is gone or its text is no longer eligible.

    :param es: Elasticsearch client.
    :param index: Index.
    :param merged_path: Merged JSONL.
    :param embedder: Embedder (the one the writer guard checked).
    :param text_version: The index's text version.
    :return: ``{"doc_id", "text_changed", "vector_changed"}`` against the stored canary.
    :rtype: Dict[str, Any]
    :raises ValueError: When ``_meta`` has no document canary, its record is not in the merged file, or its current
        text is not eligible.
    """
    from src.embeddings.qwen_text_embedding import _vector_fields

    meta = read_index_meta(es, index)
    stored = meta.get("document_canary") or {}
    doc_id = stored.get("doc_id")
    if not doc_id:
        raise ValueError(f"{index}: _meta has no document canary to refresh")
    doc = find_merged_document(merged_path, doc_id)
    if doc is None:
        raise ValueError(f"{index}: document canary {doc_id} is not in {merged_path}")
    item = embedding_text(doc, text_version, include_scholarly_frames=index_scholarly_frames(meta))
    if not item.eligible:
        raise ValueError(f"{index}: document canary {doc_id} is no longer eligible under {text_version}; "
                         "only a new index build picks another canary")
    text = item.text
    block = {"doc_id": doc_id, "text": text, "text_sha256": item.text_sha256,
             **_vector_fields(embedder.get_text_embeddings_batch([text])[0])}
    es.indices.put_mapping(index=index, meta={**meta, "document_canary": block})
    return {"doc_id": doc_id, "text_changed": text != stored.get("text"),
            "vector_changed": block["vector"] != stored.get("vector")}


def chunked(items: Iterable[Any], size: int) -> Iterator[List[Any]]:
    """Consecutive lists of at most ``size`` items.

    :param items: Iterable (consumed lazily).
    :param size: Chunk size.
    :return: Iterator over chunks.
    :rtype: Iterator[List[Any]]
    """
    iterator = iter(items)
    while chunk := list(islice(iterator, size)):
        yield chunk


def checked_embedder(meta: Mapping[str, Any], index_name: str, name: str, revision: str,
                     embedding_model: Optional[Any] = None) -> Tuple[Any, float]:
    """Load the embedder for a real run and prove it is the index's: same model and revision, canary cosine.

    :param meta: Index ``_meta``.
    :param index_name: Index.
    :param name: Model name the ``_meta`` guard checked.
    :param revision: Revision the ``_meta`` guard checked.
    :param embedding_model: Embedder (default: ``QwenTextEmbedding()``, the pinned fine-tune).
    :return: ``(embedder, canary cosine)``.
    :rtype: Tuple[Any, float]
    :raises IndexMetaMismatchError: When the embedder is not the checked one or misses the canary.
    """
    from src.embeddings import qwen_text_embedding as qte

    embedder = embedding_model if embedding_model is not None else qte.QwenTextEmbedding()
    if (embedder.model_name, embedder.revision) != (name, revision):
        raise IndexMetaMismatchError(f"embedder {embedder.model_name}@{embedder.revision} is not the one the guard "
                                     f"checked ({name}@{revision})")
    cosine = canary_cosine(meta, embedder)
    if cosine <= CANARY_COSINE_FLOOR:
        raise IndexMetaMismatchError(f"the local embedder does not reproduce {index_name}'s canary "
                                     f"(cosine {cosine:.5f} <= {CANARY_COSINE_FLOOR})")
    return embedder, cosine


def update_catalogue_enrichment(
    merged_path: Any = DEFAULT_MERGED,
    *,
    index_name: str = DEFAULT_INDEX_NAME,
    text_version: str = DEFAULT_TEXT_VERSION,
    dry_run: bool = False,
    limit: Optional[int] = None,
    skip: int = 0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    allow_live: bool = False,
    report_path: Optional[Any] = None,
    refresh_canary: bool = False,
    es_client: Optional[Any] = None,
    embedding_model: Optional[Any] = None,
) -> Dict[str, Any]:
    """Guard, then stream the merged file in chunks, decide per document and (unless ``dry_run``) embed and update.

    A dry run reads the index (guards, mapping, document fields) but never loads the embedder or writes.

    :param merged_path: Merged JSONL (default: the indexer's :data:`DEFAULT_MERGED`).
    :param index_name: Target index (default :data:`DEFAULT_INDEX_NAME`, the code constant of the indexer).
    :param text_version: Text version of the run (must match the index ``_meta``).
    :param dry_run: Report only.
    :param limit: Cap on merged records scanned.
    :param skip: Merged lines to skip (resume).
    :param batch_size: Documents per embedding / bulk batch.
    :param allow_live: Allow a live index name.
    :param report_path: Where to write the summary as JSON (optional).
    :param refresh_canary: After the writes of a real run, renew ``_meta.document_canary`` from its record's current
        text (:func:`refresh_document_canary`); a dry run only notes that it would.
    :param es_client: Elasticsearch client (default: from the environment).
    :param embedding_model: Embedder (default: ``QwenTextEmbedding()``, the pinned fine-tune; never built in a dry
        run).
    :return: The summary (:meth:`EnrichmentSummary.to_dict`).
    :rtype: Dict[str, Any]
    :raises IndexMetaMismatchError: When the index ``_meta`` or the canary does not match the embedder.
    :raises ValueError: For a live name, a missing index or a wrong vector size.
    """
    from src.embeddings import qwen_text_embedding as qte

    check_text_version(text_version)
    es = es_client if es_client is not None else default_es_client()
    if embedding_model is not None:
        name, revision = embedding_model.model_name, embedding_model.revision
    else:
        name, revision = qte.FINETUNED_MODEL_NAME, qte.FINETUNED_MODEL_REVISION    # QwenTextEmbedding()'s defaults
    meta = check_index(es, index_name, expected_meta(name, revision, text_version), allow_live)
    frames = index_scholarly_frames(meta)       # render exactly as the index was built (frames on/off)
    summary = EnrichmentSummary(index=index_name, text_version=text_version, merged_path=str(merged_path),
                                dry_run=dry_run, skip=skip, limit=limit,
                                mapping_missing=missing_mapping_fields(es, index_name))
    canary = meta.get("document_canary") or {}
    if canary.get("doc_id"):
        summary.document_canary = {"doc_id": canary["doc_id"], "text": "not scanned",
                                   "expected_text": canary.get("text")}
    embedder = None
    if not dry_run:
        embedder, summary.canary_cosine = checked_embedder(meta, index_name, name, revision, embedding_model)
        if summary.mapping_missing:
            es.indices.put_mapping(index=index_name,
                                   properties={prop: ENRICHMENT_MAPPING[prop] for prop in summary.mapping_missing})
    next_log = PROGRESS_EVERY
    for docs in chunked(iter_merged_documents(str(merged_path), limit=limit, skip=skip), MGET_CHUNK):
        records = [enriched_record(doc, text_version, frames) for doc in docs]
        states = index_states(es, index_name, [record.doc_id for record in records if record.doc_id])
        planned: List[Tuple[EnrichedRecord, str]] = []
        for record in records:
            state = states.get(record.doc_id)
            decision = decide(state, record)
            summary.add(state, record, decision)
            if decision in WRITES:
                planned.append((record, decision))
        if embedder is not None and planned:
            written = write_updates(es, index_name, planned, embedder, text_version, batch_size, meta.get("dims"))
            for decision, n in written.items():
                summary.updated[decision] += n
        if summary.scanned >= next_log:
            logger.info("scanned %d merged records (safe to resume with --skip %d): %s", summary.scanned,
                        skip + summary.scanned, ", ".join(f"{k} {v}" for k, v in summary.counts.items() if v))
            next_log += PROGRESS_EVERY
    if refresh_canary and summary.document_canary:
        if embedder is None:
            summary.document_canary["refresh"] = "skipped (dry run)"
        else:
            refreshed = refresh_document_canary(es, index_name, merged_path, embedder, text_version)
            summary.document_canary.update(refresh="refreshed", refresh_text_changed=refreshed["text_changed"])
    elif refresh_canary and embedder is not None:
        raise ValueError(f"{index_name}: _meta has no document canary to refresh")
    print(summary.describe(), flush=True)
    result = summary.to_dict()
    if report_path is not None:
        Path(report_path).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser; the index and merged-file defaults are the indexer's code constants.

    :return: Parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merged", default=DEFAULT_MERGED, help="Merged JSONL (default: DEFAULT_MERGED).")
    parser.add_argument("--index", default=DEFAULT_INDEX_NAME,
                        help="Target index (default: %(default)s, DEFAULT_INDEX_NAME in index_merged_genizah).")
    parser.add_argument("--text-version", default=DEFAULT_TEXT_VERSION, choices=TEXT_VERSIONS)
    parser.add_argument("--dry-run", action="store_true",
                        help="Read the index and report per-document decisions; embed and write nothing.")
    parser.add_argument("--limit", type=int, default=None, help="Scan at most N merged records.")
    parser.add_argument("--skip", type=int, default=0, help="Skip the first N merged lines (resume).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--report", default=None, help="Also write the summary as JSON to this path.")
    parser.add_argument("--refresh-document-canary", dest="refresh_canary", action="store_true",
                        help="After the writes, re-embed the _meta document canary from its record's current text "
                             "(the verifier's text check needs this once the document model changed).")
    parser.add_argument("--i-know-this-is-live", dest="allow_live", action="store_true",
                        help=f"Allow writing to {', '.join(sorted(LIVE_INDEX_NAMES))}.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point.

    :param argv: Arguments (None = ``sys.argv``).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    for noisy in ("elastic_transport", "elasticsearch"):     # one line per request, naming the cluster host
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    update_catalogue_enrichment(args.merged, index_name=args.index, text_version=args.text_version,
                                dry_run=args.dry_run, limit=args.limit, skip=args.skip, batch_size=args.batch_size,
                                allow_live=args.allow_live, report_path=args.report,
                                refresh_canary=args.refresh_canary)


if __name__ == "__main__":
    main()
