#!/usr/bin/env python3
# File name: update_generated_descriptions.py
# Date: 10/8/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Embed machine-generated catalogue descriptions and write them into the merged index, in place.

Reads the rows of :mod:`src.datasets.consensus.describe_from_reads` (``--generated``), embeds each row's
``description_text`` with the pinned text embedder through the indexer's own text path
(``embed_text_for_version(text, version)``: the versioned text of a production-shaped text and its eligibility, with
the index's text version, ``semantic_v3`` or ``semantic_v4``) and updates each document:

* ``embedding_vector`` (the field the index's kNN uses), ``semantic_eligible: true``, ``text_representation_version``,
  ``embedded_text_sha256`` (the hash of the embedded text, as the indexer writes it);
* ``description_source: "generated_from_consensus"``;
* ``generated_description``: the labelled text that was embedded from (shown on the site as machine-derived);
* ``generated_description_meta``: generator model, prompt version, reasoning mode, agreed-lines share, pages / lines,
  readers, reads files, generation and embedding times, confidence, embedder, grounding summary.

Guards, all before anything is embedded or written:

* the live index names are refused unless ``--i-know-this-is-live`` (``index_merged_genizah.check_index_name``);
* the index must exist, and its ``_meta`` must match the embedder and the text version on ``GUARDED_META_KEYS``
  (``IndexMetaMismatchError``, the writer guard of ``elastic_index_genizah``); a real run also re-embeds the
  ``_meta`` string canary and refuses below cosine :data:`CANARY_COSINE_FLOOR`;
* rows with ``embed_eligible: false``, another ``description_source``, a confidence under ``--min-confidence`` or a
  text the index's version does not find eligible are refused;
* documents missing from the index, or holding a catalogue vector (``semantic_eligible: true`` without
  ``description_source: generated_from_consensus``), are refused: a generated description never replaces a
  catalogue text; documents already carrying the row's generation (same ``generated_at`` and ``prompt_version``)
  are skipped unless ``--force``.

The new fields are added to the mapping (``put_mapping`` with properties only, so ``_meta`` is untouched).
Rebuilding the index from the merged file drops these vectors (the indexer embeds merged-file text only): re-run this
updater after a rebuild (it is idempotent). The verifier's self-consistency check recomputes each sampled document's
text from the merged file, and for ``description_source: generated_from_consensus`` documents with
:func:`embedded_text_from_source` instead.

Usage (repo root; Elasticsearch credentials and ``HF1_TOKEN`` in the environment / ``.env``)::

    python -m src.datasets.indexing.update_generated_descriptions \\
        --generated logs/next_round/consensus_pool/generated_descriptions.jsonl --dry-run
    python -m src.datasets.indexing.update_generated_descriptions --generated ... [--index genizah_merged_v10] \\
        [--text-version semantic_v4]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import dotenv
import numpy as np
from elasticsearch.helpers import bulk

from src.datasets.consensus.describe_from_reads import DESCRIPTION_SOURCE, complete_lines, summarise_grounding
from src.datasets.indexing.elastic_index_genizah import (
    TEXT_HASH_FIELD,
    TEXT_VERSION_MAPPING,
    IndexMetaMismatchError,
    assert_index_writable,
    es_config_from_env,
    read_index_meta,
)
from src.datasets.indexing.index_merged_genizah import LIVE_INDEX_NAME, LIVE_INDEX_NAMES, check_index_name
from src.embeddings.semantic_text import text_sha256
from src.embeddings.text_representation import (
    DEFAULT_TEXT_VERSION,
    TEXT_VERSIONS,
    check_text_version,
    embed_text_for_version,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
VECTOR_FIELD = "embedding_vector"
CANARY_COSINE_FLOOR = 0.99      # the verifier's floor
DEFAULT_BATCH_SIZE = 32
MGET_CHUNK = 200
STATE_FIELDS = ["semantic_eligible", "description_source", "text_representation_version",
                "generated_description_meta.generated_at", "generated_description_meta.prompt_version"]
GENERATED_FIELDS_MAPPING: Dict[str, Any] = {
    "description_source": {"type": "keyword"},
    "generated_description": {"type": "text", "analyzer": "multilingual"},
    "generated_description_meta": {"properties": {
        "model": {"type": "keyword"},
        "prompt_version": {"type": "keyword"},
        "reasoning": {"type": "keyword"},
        "generated_at": {"type": "date"},
        "embedded_at": {"type": "date"},
        "agreed_lines_pct": {"type": "float"},
        "n_pages": {"type": "integer"},
        "n_lines": {"type": "integer"},
        "n_agreed": {"type": "integer"},
        "readers": {"properties": {"vlm": {"type": "keyword"}, "htr": {"type": "keyword"}}},
        "source_files": {"type": "keyword"},
        "confidence": {"type": "float"},
        "machine_derived": {"type": "boolean"},
        "embedder_model": {"type": "keyword"},
        "embedder_revision": {"type": "keyword"},
        "grounding": {"properties": {
            "entities_kept": {"type": "integer"},
            "entities_dropped": {"type": "integer"},
            "summary_spans_absent": {"type": "integer"},
            "confidence_capped": {"type": "boolean"},
            "identified_work": {"type": "keyword"},
            "normalisation": {"type": "keyword"},
        }},
    }},
    # every v10 mapping has it already (the indexer's TEXT_VERSION_MAPPING); an older index gets it as a keyword
    TEXT_HASH_FIELD: TEXT_VERSION_MAPPING[TEXT_HASH_FIELD],
}


@dataclass
class PlannedUpdate:
    """One document to update.

    :param doc_id: Index ``_id``.
    :param row: Its generator row.
    :param text: The versioned text that will be embedded.
    """

    doc_id: str
    row: Dict[str, Any]
    text: str


@dataclass
class UpdatePlan:
    """What a run would do.

    :param index: Target index.
    :param text_version: Text version of the run (the index's, by the writer guard).
    :param rows_read: Rows in the generator file.
    :param updates: Documents to update.
    :param refused: ``(doc_id, reason)`` refused documents.
    :param skipped: ``(doc_id, reason)`` documents left alone (already applied).
    :param mapping_missing: Generated-description fields the mapping still lacks.
    """

    index: str
    text_version: str
    rows_read: int
    updates: List[PlannedUpdate] = field(default_factory=list)
    refused: List[Tuple[str, str]] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)
    mapping_missing: List[str] = field(default_factory=list)

    def counts(self) -> Dict[str, Any]:
        """Counts per outcome, refusals grouped by reason.

        :return: ``rows_read``, ``to_update``, ``skipped``, ``refused`` and ``refused_by_reason``.
        :rtype: Dict[str, Any]
        """
        by_reason: Dict[str, int] = {}
        for _, reason in self.refused:
            by_reason[reason] = by_reason.get(reason, 0) + 1
        return {"rows_read": self.rows_read, "to_update": len(self.updates), "skipped": len(self.skipped),
                "refused": len(self.refused), "refused_by_reason": by_reason}

    def describe(self) -> str:
        """Human-readable plan.

        :return: Multi-line text.
        :rtype: str
        """
        counts = self.counts()
        lines = [f"index {self.index} (text {self.text_version}): {counts['rows_read']} generator rows; "
                 f"to update {counts['to_update']}, already applied {counts['skipped']}, refused {counts['refused']}"]
        lines += [f"  refused: {reason} x{n}" for reason, n in sorted(counts["refused_by_reason"].items())]
        if self.mapping_missing:
            lines.append(f"  mapping: would add {', '.join(self.mapping_missing)}")
        lines += [f"  update {u.doc_id}: {u.text.splitlines()[0][:110]}" for u in self.updates[:5]]
        return "\n".join(lines)


def read_generated(path: Any) -> Tuple[int, Dict[str, Dict[str, Any]]]:
    """Generator rows by document; a later row for the same document replaces an earlier one.

    :param path: Generator output JSONL.
    :return: ``(rows read, {doc_id: row})``.
    :rtype: Tuple[int, Dict[str, Dict[str, Any]]]
    """
    rows: Dict[str, Dict[str, Any]] = {}
    n = 0
    for line in complete_lines(Path(path)):
        row = json.loads(line)
        rows[row["doc_id"]] = row
        n += 1
    return n, rows


def row_refusal(row: Mapping[str, Any], text_version: str, min_confidence: float = 0.0
                ) -> Tuple[Optional[str], Optional[str]]:
    """Why a generator row may not be embedded, or the text it will be embedded from.

    :param row: Generator row.
    :param text_version: The index's text version.
    :param min_confidence: Lowest grounded confidence accepted.
    :return: ``(reason, None)`` when refused, else ``(None, versioned text)``.
    :rtype: Tuple[Optional[str], Optional[str]]
    """
    if row.get("description_source") != DESCRIPTION_SOURCE:
        return f"description_source is {row.get('description_source')!r}", None
    if row.get("embed_eligible") is not True:
        return f"embed_eligible false ({row.get('embed_reason')})", None
    confidence = (row.get("json") or {}).get("confidence")
    if confidence is None or confidence < min_confidence:
        return f"confidence {confidence} < {min_confidence}", None
    item = embed_text_for_version(str(row.get("description_text") or ""), text_version)
    if not item.eligible:
        return f"text not eligible under {text_version}", None
    return None, item.text


def index_states(es: Any, index: str, ids: Sequence[str]) -> Dict[str, Optional[Dict[str, Any]]]:
    """The flags of each document that decide whether it may be updated (``None`` = not in the index).

    :param es: Elasticsearch client.
    :param index: Index.
    :param ids: Document ids.
    :return: ``{doc_id: _source subset or None}``.
    :rtype: Dict[str, Optional[Dict[str, Any]]]
    """
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for start in range(0, len(ids), MGET_CHUNK):
        resp = es.mget(index=index, ids=list(ids[start:start + MGET_CHUNK]), source_includes=STATE_FIELDS)
        for doc in resp["docs"]:
            out[doc["_id"]] = (doc.get("_source") or {}) if doc.get("found") else None
    return out


def document_decision(state: Optional[Mapping[str, Any]], row: Mapping[str, Any], force: bool = False) -> str:
    """``update``, ``skip:<why>`` or ``refuse:<why>`` for one document.

    :param state: Its flags in the index (:func:`index_states`), None when absent.
    :param row: Its generator row.
    :param force: Re-apply a generation the document already carries.
    :return: Decision.
    :rtype: str
    """
    if state is None:
        return "refuse:not in the index"
    generated = state.get("description_source") == DESCRIPTION_SOURCE
    if state.get("semantic_eligible") is True and not generated:
        return "refuse:has a catalogue vector"
    meta = state.get("generated_description_meta") or {}
    same = (generated and meta.get("generated_at") == row.get("generated_at")
            and meta.get("prompt_version") == row.get("prompt_version") and state.get("semantic_eligible") is True)
    if same and not force:
        return "skip:already applied"
    return "update"


def missing_mapping_fields(es: Any, index: str) -> List[str]:
    """Generated-description fields the index mapping does not have yet.

    :param es: Elasticsearch client.
    :param index: Index.
    :return: Field names (top level).
    :rtype: List[str]
    """
    mapping = es.indices.get_mapping(index=index)
    properties = (next(iter(mapping.values())).get("mappings") or {}).get("properties") or {}
    return [name for name in GENERATED_FIELDS_MAPPING if name not in properties]


def plan_updates(rows: Mapping[str, Mapping[str, Any]], rows_read: int, es: Any, index: str, text_version: str,
                 min_confidence: float = 0.0, force: bool = False, limit: Optional[int] = None) -> UpdatePlan:
    """Decide, per generator row, whether its document is updated, skipped or refused (reads the index only).

    :param rows: Latest generator row per document.
    :param rows_read: Rows in the generator file.
    :param es: Elasticsearch client.
    :param index: Index.
    :param text_version: The index's text version.
    :param min_confidence: Lowest grounded confidence accepted.
    :param force: Re-apply generations already applied.
    :param limit: Cap on documents to update.
    :return: The plan.
    :rtype: UpdatePlan
    """
    plan = UpdatePlan(index=index, text_version=text_version, rows_read=rows_read)
    texts: Dict[str, str] = {}
    for doc_id, row in rows.items():
        reason, text = row_refusal(row, text_version, min_confidence)
        if reason:
            plan.refused.append((doc_id, reason))
        else:
            texts[doc_id] = text
    states = index_states(es, index, sorted(texts))
    for doc_id in sorted(texts):
        decision = document_decision(states.get(doc_id), rows[doc_id], force)
        kind, _, why = decision.partition(":")
        if kind == "refuse":
            plan.refused.append((doc_id, why))
        elif kind == "skip":
            plan.skipped.append((doc_id, why))
        elif limit is None or len(plan.updates) < limit:
            plan.updates.append(PlannedUpdate(doc_id=doc_id, row=dict(rows[doc_id]), text=texts[doc_id]))
    plan.mapping_missing = missing_mapping_fields(es, index)
    return plan


def expected_meta(model_name: str, model_revision: str, text_version: str) -> Dict[str, str]:
    """The guarded ``_meta`` values this run would write with.

    :param model_name: Embedder model name.
    :param model_revision: Embedder revision.
    :param text_version: Text version.
    :return: ``{model_name, model_revision, text_representation_version}``.
    :rtype: Dict[str, str]
    """
    return {"model_name": model_name, "model_revision": model_revision, "text_representation_version": text_version}


def check_index(es: Any, index: str, expected: Mapping[str, Any], allow_live: bool = False) -> Dict[str, Any]:
    """Name guard, existence, and the ``_meta`` writer guard.

    :param es: Elasticsearch client.
    :param index: Target index.
    :param expected: :func:`expected_meta`.
    :param allow_live: ``--i-know-this-is-live``.
    :return: The index ``_meta``.
    :rtype: Dict[str, Any]
    :raises ValueError: For a live name without ``allow_live``, or a missing index.
    :raises IndexMetaMismatchError: When the index ``_meta`` differs on a guarded key.
    """
    check_index_name(index, allow_live)
    if not es.indices.exists(index=index):
        raise ValueError(f"index {index!r} does not exist: generated descriptions are written into an existing index")
    assert_index_writable(es, index, expected)
    return read_index_meta(es, index)


def canary_cosine(meta: Mapping[str, Any], embedder: Any) -> float:
    """Cosine between the ``_meta`` string canary and the local embedder's vector for the same string.

    :param meta: Index ``_meta``.
    :param embedder: Text embedder (``get_text_embeddings_batch``).
    :return: Cosine.
    :rtype: float
    :raises IndexMetaMismatchError: When the index has no canary.
    """
    canary = meta.get("canary") or {}
    if not canary.get("vector") or not canary.get("string"):
        raise IndexMetaMismatchError("index _meta has no string canary to check the embedder against")
    stored = np.asarray(canary["vector"], dtype=np.float32).reshape(-1)
    fresh = np.asarray(embedder.get_text_embeddings_batch([canary["string"]])[0], dtype=np.float32).reshape(-1)
    return float(stored @ fresh / (np.linalg.norm(stored) * np.linalg.norm(fresh)))


def generated_meta(row: Mapping[str, Any], embedder: Any, embedded_at: str) -> Dict[str, Any]:
    """``generated_description_meta`` of one document.

    :param row: Generator row.
    :param embedder: Embedder (``model_name``, ``revision``).
    :param embedded_at: UTC timestamp.
    :return: Meta block.
    :rtype: Dict[str, Any]
    """
    return {
        "model": row.get("model"), "prompt_version": row.get("prompt_version"), "reasoning": row.get("reasoning"),
        "generated_at": row.get("generated_at"), "embedded_at": embedded_at,
        "agreed_lines_pct": row.get("agreed_lines_pct"), "n_pages": row.get("n_pages"),
        "n_lines": row.get("n_lines"), "n_agreed": row.get("n_agreed"), "readers": row.get("readers"),
        "source_files": row.get("source_files"), "confidence": (row.get("json") or {}).get("confidence"),
        "machine_derived": True, "embedder_model": embedder.model_name, "embedder_revision": embedder.revision,
        "grounding": summarise_grounding(row.get("grounding") or {}),
    }


def update_fields(planned: PlannedUpdate, vector: Any, text_version: str, embedder: Any,
                  embedded_at: str) -> Dict[str, Any]:
    """The partial document written for one update.

    :param planned: The planned update.
    :param vector: Its embedding.
    :param text_version: The index's text version.
    :param embedder: Embedder.
    :param embedded_at: UTC timestamp.
    :return: Fields to merge into the document.
    :rtype: Dict[str, Any]
    """
    return {
        VECTOR_FIELD: [float(v) for v in np.asarray(vector).reshape(-1)],
        "semantic_eligible": True,
        "text_representation_version": text_version,
        TEXT_HASH_FIELD: text_sha256(planned.text),
        "description_source": DESCRIPTION_SOURCE,
        "generated_description": planned.row["description_text"],
        "generated_description_meta": generated_meta(planned.row, embedder, embedded_at),
    }


def apply_plan(plan: UpdatePlan, es: Any, embedder: Any, batch_size: int = DEFAULT_BATCH_SIZE,
               dims: Optional[int] = None) -> int:
    """Embed the planned texts in batches and write each batch with a bulk ``update``.

    :param plan: The plan.
    :param es: Elasticsearch client.
    :param embedder: Embedder (``get_text_embeddings_batch``, ``model_name``, ``revision``).
    :param batch_size: Documents per embedding / bulk batch.
    :param dims: Vector size the index expects (checked before each write when given).
    :return: Documents updated.
    :rtype: int
    :raises ValueError: When a vector has the wrong size.
    """
    if plan.mapping_missing:
        es.indices.put_mapping(index=plan.index,
                               properties={name: GENERATED_FIELDS_MAPPING[name] for name in plan.mapping_missing})
    done = 0
    for start in range(0, len(plan.updates), batch_size):
        batch = plan.updates[start:start + batch_size]
        vectors = embedder.get_text_embeddings_batch([u.text for u in batch])
        embedded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        actions = []
        for planned, vector in zip(batch, vectors):
            fields = update_fields(planned, vector, plan.text_version, embedder, embedded_at)
            if dims is not None and len(fields[VECTOR_FIELD]) != dims:
                raise ValueError(f"{planned.doc_id}: vector has {len(fields[VECTOR_FIELD])} dims, index has {dims}")
            actions.append({"_op_type": "update", "_index": plan.index, "_id": planned.doc_id, "doc": fields})
        bulk(client=es, actions=actions, refresh="wait_for")
        done += len(actions)
        logger.info("updated %d / %d documents", done, len(plan.updates))
    return done


def embedded_text_from_source(source: Mapping[str, Any], text_version: str) -> Optional[str]:
    """The text a generated-description document was embedded from (for the verifier's self-consistency check).

    :param source: Index ``_source`` (needs ``description_source`` and ``generated_description``).
    :param text_version: The index's text version.
    :return: The versioned text, or None for a document embedded from its catalogue text.
    :rtype: Optional[str]
    """
    if source.get("description_source") != DESCRIPTION_SOURCE:
        return None
    return embed_text_for_version(str(source.get("generated_description") or ""), text_version).text


def default_es_client() -> Any:
    """Elasticsearch client from the repo's environment (``.env`` loaded, never printed).

    :return: Client.
    """
    from elasticsearch import Elasticsearch

    dotenv.load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    return Elasticsearch(**es_config_from_env(), request_timeout=60)


def update_generated_descriptions(
    generated_path: Any,
    *,
    index_name: str = LIVE_INDEX_NAME,
    text_version: str = DEFAULT_TEXT_VERSION,
    dry_run: bool = False,
    limit: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    min_confidence: float = 0.0,
    force: bool = False,
    allow_live: bool = False,
    es_client: Optional[Any] = None,
    embedding_model: Optional[Any] = None,
) -> Dict[str, Any]:
    """Guard, plan, then (unless ``dry_run``) embed and update.

    A dry run reads the index (guards, document flags, mapping) but never loads the embedder or writes.

    :param generated_path: Generator output JSONL.
    :param index_name: Target index (default :data:`LIVE_INDEX_NAME`, the served index; needs ``allow_live``).
    :param text_version: Text version of the run (must match the index ``_meta``).
    :param dry_run: Print the plan only.
    :param limit: Cap on documents to update.
    :param batch_size: Documents per embedding / bulk batch.
    :param min_confidence: Lowest grounded confidence accepted.
    :param force: Re-apply generations already applied.
    :param allow_live: Allow a live index name.
    :param es_client: Elasticsearch client (default: from the environment).
    :param embedding_model: Embedder (default: ``QwenTextEmbedding()``, the pinned fine-tune; never built in a dry
        run).
    :return: Plan counts plus ``updated`` and ``dry_run``.
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
    rows_read, rows = read_generated(generated_path)
    plan = plan_updates(rows, rows_read, es, index_name, text_version, min_confidence, force, limit)
    print(plan.describe(), flush=True)
    result = {**plan.counts(), "updated": 0, "dry_run": dry_run}
    if dry_run or not plan.updates:
        print("dry run: nothing embedded or written" if dry_run else "nothing to update", flush=True)
        return result
    embedder = embedding_model if embedding_model is not None else qte.QwenTextEmbedding()
    if (embedder.model_name, embedder.revision) != (name, revision):
        raise IndexMetaMismatchError(f"embedder {embedder.model_name}@{embedder.revision} is not the one the guard "
                                     f"checked ({name}@{revision})")
    cosine = canary_cosine(meta, embedder)
    if cosine <= CANARY_COSINE_FLOOR:
        raise IndexMetaMismatchError(f"the local embedder does not reproduce {index_name}'s canary "
                                     f"(cosine {cosine:.5f} <= {CANARY_COSINE_FLOOR})")
    result["updated"] = apply_plan(plan, es, embedder, batch_size, dims=meta.get("dims"))
    print(f"updated {result['updated']} documents in {index_name} (canary cosine {cosine:.5f})", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser; the index default is the indexer's code constant.

    :return: Parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generated", required=True, help="describe_from_reads output JSONL")
    parser.add_argument("--index", default=LIVE_INDEX_NAME,
                        help="Target index (default: %(default)s, LIVE_INDEX_NAME in index_merged_genizah, the served "
                             "index: pass --i-know-this-is-live to write to it).")
    parser.add_argument("--text-version", default=DEFAULT_TEXT_VERSION, choices=TEXT_VERSIONS)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan (guards, per-document decisions); embed and write nothing.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--force", action="store_true", help="Re-apply generations a document already carries.")
    parser.add_argument("--i-know-this-is-live", dest="allow_live", action="store_true",
                        help=f"Allow writing to {', '.join(sorted(LIVE_INDEX_NAMES))}.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point.

    :param argv: Arguments (None = ``sys.argv``).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args(argv)
    update_generated_descriptions(args.generated, index_name=args.index, text_version=args.text_version,
                                  dry_run=args.dry_run, limit=args.limit, batch_size=args.batch_size,
                                  min_confidence=args.min_confidence, force=args.force, allow_live=args.allow_live)


if __name__ == "__main__":
    main()
