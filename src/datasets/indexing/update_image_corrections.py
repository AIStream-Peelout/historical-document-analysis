#!/usr/bin/env python3
# File name: update_image_corrections.py
# Date: 10/31/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Bring the served index's image (and transcription) fields in line with corrected legacy-store records, in place.

Some legacy-store photographs (``images.fjp`` in the merged file) were saved under a neighbouring fragment's name.
The merge applies durable corrections (``image_corrections.jsonl``: each row moves an image from
``from_canonical_id`` to ``to_canonical_id``, or removes it when there is no target), and a corrected record can
also carry ``images.fjp_corrections`` / ``images.fjp_removed``. The site serves an index built from an earlier
merged file, so it still shows the misfiled photographs until the image fields of the affected documents are
rewritten. The same file can also hold transcription corrections (``kind: "transcription"``: legacy-store
transcription sections moved to the record they belong to, or detached), which the merge marks with the record-level
``transcriptions_corrections`` / ``transcriptions_removed`` lists. This updater rewrites exactly those fields and
nothing else.

Affected documents: every ``from_canonical_id`` and ``to_canonical_id`` in the corrections file, plus every merged
record that carries a marker (``images.fjp_corrections``, ``images.fjp_removed``, ``transcriptions_corrections``,
``transcriptions_removed``). Each one is built with
``GenizahDocument.from_merged_format(record).to_elasticsearch_document()`` and only its image fields are kept
(:data:`IMAGE_FIELDS`):

* top level: ``image_urls``, ``primary_image_index``, ``has_images``, ``actual_image_url`` (the first image, which
  the site shows first), ``image_preferred_source``, ``has_ktiv_images``, ``ktiv_iiif_manifest_url``,
  ``has_bodleian_images``, ``bodleian_catalogue_url``;
* the image provenance the document model keeps in ``full_metadata``, which the index stores as ``tei_metadata``:
  ``image_preferred_source``, ``ktiv_images``, ``ktiv_iiif_manifest_url``, ``bodleian_images``,
  ``bodleian_catalogue_url``, ``bodleian_tei_part_id``, ``bodleian_match``.

A *transcription-affected* document (named by a transcription correction, or carrying a transcription marker) also
has :data:`TRANSCRIPTION_FIELDS` compared and written: ``has_transcriptions``, the nested ``transcriptions`` and
``full_text_content`` (which carries the transcription text for lexical search). These are large, so they are read
and stored in the plan for those documents only. The embedding vector is never recomputed here: a document whose
transcriptions change keeps a vector embedded from its old text (the report counts them).

The document ``_id`` is ``es_document_id(doc)`` (the canonical id for a merged record); the served
``canonical_id`` is read too, and a document whose ``canonical_id`` differs is never written (``id_mismatch``).

Modes (the default is the dry run; nothing is written without ``--apply``, and nothing is written to the served
index -- any name in ``index_merged_genizah.LIVE_INDEX_NAMES``, the default target -- without
``--i-know-this-is-live`` as well):

* **dry run**: read the served documents (``mget``) and the index mapping, diff the image fields, and write the plan
  (:data:`DEFAULT_PLAN`: a header row, then one row per affected document with its before and after values) and a
  Markdown report (:data:`DEFAULT_REPORT`: counts, images added and removed per document, a sample of diffs that
  shows image FILE NAMES only, never full URLs);
* **apply** (``--apply``): apply that plan as bulk partial ``update`` actions (only the changed fields of each
  document, never a reindex), with ``refresh="wait_for"``, read every written document back to verify it, and write
  a result file (:data:`DEFAULT_RESULT`). The plan is first copied next to itself as ``*.applied-<UTC time>.jsonl``,
  the copy a rollback uses. Needs ``--i-know-this-is-live`` too for the served index;
* **rollback** (``--rollback PLAN``): check that the plan's documents hold its after values and, with ``--apply``
  (and ``--i-know-this-is-live`` for the served index), write the before values back (a field the index did not
  hold is written as ``null``).

Apply refuses (:class:`ApplyRefused`, nothing written) when the target is the served index and
``--i-know-this-is-live`` was not given, when the plan is older than the merged file or the merged
file changed since the plan, when the plan was built from another merged or corrections file or against another
index, when more than ``--max-docs`` documents would change, when a write would add a field the mapping does not
have, or when any document no longer holds the plan's before values (re-run the dry run). Rollback applies the
same index, ``--max-docs`` and state checks. Re-running an apply is a no-op for documents already written.

The Elasticsearch client is built from the repository ``.env`` inside
:func:`update_generated_descriptions.default_es_client`; this module never prints the client configuration, the
cluster host or the index name (the plan stores a fingerprint of the index name, not the name).

Usage (repo root)::

    PYTHONPATH=. .venv/bin/python -m src.datasets.indexing.update_image_corrections [--merged M] [--corrections C]
    PYTHONPATH=. .venv/bin/python -m src.datasets.indexing.update_image_corrections [--merged M] [--corrections C] \\
        --apply --i-know-this-is-live
    PYTHONPATH=. .venv/bin/python -m src.datasets.indexing.update_image_corrections --rollback PLAN \\
        [--apply --i-know-this-is-live]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import re
import shutil
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

from elasticsearch.helpers import bulk

from src.datasets.document_models.genizah_document import GenizahDocument
from src.datasets.indexing.elastic_index_genizah import es_document_id
from src.datasets.indexing.index_merged_genizah import DEFAULT_MERGED, LIVE_INDEX_NAME, LIVE_INDEX_NAMES
from src.datasets.indexing.update_generated_descriptions import MGET_CHUNK, default_es_client

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
#: The merge's durable image corrections, next to the merged file.
DEFAULT_CORRECTIONS = os.path.join(os.path.dirname(DEFAULT_MERGED), "image_corrections.jsonl")
DEFAULT_OUT_DIR = os.path.join(_REPO_ROOT, "logs", "fgp_native_1030")
DEFAULT_PLAN = os.path.join(DEFAULT_OUT_DIR, "site_image_update_plan.jsonl")
DEFAULT_REPORT = os.path.join(DEFAULT_OUT_DIR, "site_image_update_report.md")
DEFAULT_RESULT = os.path.join(DEFAULT_OUT_DIR, "site_image_update_result.jsonl")
DEFAULT_ROLLBACK_RESULT = os.path.join(DEFAULT_OUT_DIR, "site_image_rollback_result.jsonl")
DEFAULT_MAX_DOCS = 2000
PLAN_VERSION = 2
SAMPLE_DIFFS = 10
WRITE_CHUNK = 200

#: Top-level image fields of ``to_elasticsearch_document`` (``image_urls`` and the fields derived from it, and the
#: per-source image provenance it lifts out of ``full_metadata``).
TOP_LEVEL_IMAGE_FIELDS = ("image_urls", "primary_image_index", "has_images", "actual_image_url",
                          "image_preferred_source", "has_ktiv_images", "ktiv_iiif_manifest_url",
                          "has_bodleian_images", "bodleian_catalogue_url")
#: Where the index stores the document model's ``full_metadata``.
METADATA_FIELD = "tei_metadata"
#: The image-provenance keys ``from_merged_format`` writes into ``full_metadata`` (from the merged ``images`` block).
METADATA_IMAGE_KEYS = ("image_preferred_source", "ktiv_images", "ktiv_iiif_manifest_url", "bodleian_images",
                       "bodleian_catalogue_url", "bodleian_tei_part_id", "bodleian_match")
#: The image fields this updater compares and may write, nested ones as dotted paths.
IMAGE_FIELDS = TOP_LEVEL_IMAGE_FIELDS + tuple(f"{METADATA_FIELD}.{key}" for key in METADATA_IMAGE_KEYS)
#: The transcription fields of ``to_elasticsearch_document``, compared and written for transcription-affected
#: documents only (``transcriptions`` is the nested per-section list, ``full_text_content`` holds the text too).
TRANSCRIPTION_FIELDS = ("has_transcriptions", "transcriptions", "full_text_content")
#: Every field this updater may compare, in report order.
ALL_FIELDS = IMAGE_FIELDS + TRANSCRIPTION_FIELDS
#: ``_source`` paths read per served document (transcription-affected documents also read
#: :data:`TRANSCRIPTION_FIELDS`).
READ_FIELDS = ["canonical_id", *IMAGE_FIELDS]
#: Lists in :data:`IMAGE_FIELDS` that hold image URLs (shown as file names in the report).
IMAGE_LIST_FIELDS = ("image_urls", f"{METADATA_FIELD}.ktiv_images", f"{METADATA_FIELD}.bodleian_images")
#: Keys of a merged record's ``images`` block that mark it as corrected (data field names of the merge).
CORRECTION_MARKERS = ("fjp_corrections", "fjp_removed")
#: Top-level merged-record keys that mark a transcription correction (``merge_shelfmarks``).
TRANSCRIPTION_MARKERS = ("transcriptions_corrections", "transcriptions_removed")
#: The corrections-file ``kind`` of a transcription correction (``image_corrections.KIND_TRANSCRIPTION``).
TRANSCRIPTION_KIND = "transcription"

CHANGED = "changed"
UNCHANGED = "unchanged"
MISSING_IN_INDEX = "missing_in_index"
NOT_IN_MERGED = "not_in_merged"
ID_MISMATCH = "id_mismatch"
STATUSES = (CHANGED, UNCHANGED, MISSING_IN_INDEX, NOT_IN_MERGED, ID_MISMATCH)

#: Per-document states before an apply or rollback write.
TO_WRITE = "to_write"
DONE = "done"
DRIFTED = "drifted"

_CANONICAL_ID_PREFIX_RE = re.compile(r'^\{"canonical_id":\s*("(?:[^"\\]|\\.)*")')
_IIIF_QUALITY_RE = re.compile(r"^(default|color|colour|gray|grey|bitonal|native)\.\w+$", re.IGNORECASE)


class ApplyRefused(ValueError):
    """An apply or rollback that must not write (stale plan, too many documents, unexpected index state)."""


# ----------------------------------------------------------------------------------------------- inputs


def read_corrections(path: Any) -> List[Dict[str, Any]]:
    """Rows of the corrections file (JSONL; blank lines skipped).

    Only ``from_canonical_id`` (required) and ``to_canonical_id`` (optional: no target = a removal) are used here;
    other keys are kept as they are.

    :param path: Corrections JSONL.
    :return: Rows in file order.
    :rtype: List[Dict[str, Any]]
    :raises ValueError: When a row is not an object or has no ``from_canonical_id``.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not str(row.get("from_canonical_id") or "").strip():
                raise ValueError(f"corrections line {number}: no from_canonical_id")
            rows.append(row)
    return rows


def correction_ids(corrections: Iterable[Mapping[str, Any]]) -> Set[str]:
    """Every canonical id a correction names, as source or target.

    :param corrections: Rows of :func:`read_corrections`.
    :return: Canonical ids.
    :rtype: Set[str]
    """
    ids: Set[str] = set()
    for row in corrections:
        for key in ("from_canonical_id", "to_canonical_id"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                ids.add(value.strip())
    return ids


def transcription_correction_ids(corrections: Iterable[Mapping[str, Any]]) -> Set[str]:
    """The canonical ids a transcription correction names, as source or target.

    :param corrections: Rows of :func:`read_corrections`.
    :return: Canonical ids.
    :rtype: Set[str]
    """
    return correction_ids(row for row in corrections if row.get("kind") == TRANSCRIPTION_KIND)


def file_sha256(path: Any) -> str:
    """SHA-256 of a file's bytes.

    :param path: File.
    :return: Hex digest.
    :rtype: str
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def target_fingerprint(index_name: str) -> str:
    """A short fingerprint of the target index name, stored in the plan instead of the name.

    :param index_name: Index.
    :return: First 16 hex digits of the name's SHA-256.
    :rtype: str
    """
    return hashlib.sha256(index_name.encode("utf-8")).hexdigest()[:16]


def line_canonical_id(line: str) -> Optional[str]:
    """The ``canonical_id`` of one merged JSONL line, read from the line's prefix when it leads (as the merge
    writes it), else by parsing the whole line.

    :param line: Merged JSONL line.
    :return: Canonical id.
    :rtype: Optional[str]
    """
    match = _CANONICAL_ID_PREFIX_RE.match(line)
    if match:
        return json.loads(match.group(1))
    return json.loads(line).get("canonical_id")


def has_transcription_marker(record: Mapping[str, Any]) -> bool:
    """Whether a merged record says the merge corrected its transcriptions.

    :param record: Merged record.
    :return: True when ``transcriptions_corrections`` or ``transcriptions_removed`` is non-empty.
    :rtype: bool
    """
    return any(record.get(key) for key in TRANSCRIPTION_MARKERS)


def has_correction_marker(record: Mapping[str, Any]) -> bool:
    """Whether a merged record says the merge corrected its images or its transcriptions.

    :param record: Merged record.
    :return: True when ``images.fjp_corrections``, ``images.fjp_removed`` or a transcription marker is non-empty.
    :rtype: bool
    """
    images = record.get("images") or {}
    return any(images.get(key) for key in CORRECTION_MARKERS) or has_transcription_marker(record)


def scan_merged(merged_path: Any, ids: Iterable[str]) -> Tuple[Dict[str, Dict[str, Any]], Set[str]]:
    """One pass over the merged file: the records named by ``ids`` and every record carrying a correction marker.

    Only lines whose canonical id is wanted, or which mention a marker key, are parsed in full.

    :param merged_path: Merged JSONL.
    :param ids: Canonical ids to collect.
    :return: ``({canonical_id: record}, canonical ids found through a marker)``.
    :rtype: Tuple[Dict[str, Dict[str, Any]], Set[str]]
    """
    wanted = set(ids)
    markers = tuple(f'"{key}"' for key in CORRECTION_MARKERS + TRANSCRIPTION_MARKERS)
    records: Dict[str, Dict[str, Any]] = {}
    marked: Set[str] = set()
    with open(merged_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            canonical_id = line_canonical_id(line)
            mentions_marker = any(marker in line for marker in markers)
            if canonical_id not in wanted and not mentions_marker:
                continue
            record = json.loads(line)
            if has_correction_marker(record):
                marked.add(record["canonical_id"])
            elif canonical_id not in wanted:
                continue
            records[record["canonical_id"]] = record
    return records, marked


# ----------------------------------------------------------------------------------------------- image fields


def normalise(value: Any) -> Any:
    """A value as it reads back from JSON (tuples become lists), so served and built values compare equal.

    :param value: Value.
    :return: JSON round-tripped value.
    :rtype: Any
    """
    return json.loads(json.dumps(value, ensure_ascii=False))


def document_fields(source: Mapping[str, Any], names: Iterable[str]) -> Dict[str, Any]:
    """Fields of an index document (built or served) by dotted path; an absent field reads as ``None``.

    :param source: ``to_elasticsearch_document()`` output or a served ``_source``.
    :param names: Field paths (top level, or ``tei_metadata.<key>``).
    :return: ``{field: value}`` for every name, in the given order.
    :rtype: Dict[str, Any]
    """
    fields: Dict[str, Any] = {}
    for name in names:
        head, dot, tail = name.partition(".")
        value = (source.get(head) or {}).get(tail) if dot else source.get(name)
        fields[name] = normalise(value)
    return fields


def image_fields(source: Mapping[str, Any]) -> Dict[str, Any]:
    """The image fields of an index document (built or served), dotted paths for the metadata keys; an absent field
    reads as ``None``.

    :param source: ``to_elasticsearch_document()`` output or a served ``_source``.
    :return: ``{field: value}`` for every :data:`IMAGE_FIELDS` entry.
    :rtype: Dict[str, Any]
    """
    return document_fields(source, IMAGE_FIELDS)


def compared_names(with_transcriptions: bool) -> Tuple[str, ...]:
    """The fields compared for a document.

    :param with_transcriptions: Whether the document is transcription-affected.
    :return: :data:`IMAGE_FIELDS`, plus :data:`TRANSCRIPTION_FIELDS` when *with_transcriptions*.
    :rtype: Tuple[str, ...]
    """
    return ALL_FIELDS if with_transcriptions else IMAGE_FIELDS


def target_fields(record: Mapping[str, Any], with_transcriptions: bool = False) -> Tuple[str, Dict[str, Any]]:
    """The document id and compared fields a merged record gets under the current document model.

    :param record: Merged record.
    :param with_transcriptions: Also return :data:`TRANSCRIPTION_FIELDS`.
    :return: ``(es document id, fields)``.
    :rtype: Tuple[str, Dict[str, Any]]
    """
    doc = GenizahDocument.from_merged_format(dict(record))
    es_doc = doc.to_elasticsearch_document()
    return es_document_id(doc) or record["canonical_id"], document_fields(es_doc, compared_names(with_transcriptions))


def target_image_fields(record: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """The document id and image fields a merged record gets under the current document model.

    :param record: Merged record.
    :return: ``(es document id, image fields)``.
    :rtype: Tuple[str, Dict[str, Any]]
    """
    return target_fields(record, with_transcriptions=False)


def changed_fields(before: Mapping[str, Any], after: Mapping[str, Any]) -> List[str]:
    """The compared fields whose value differs (the fields *after* holds).

    :param before: Served fields.
    :param after: Target fields.
    :return: Field names in :data:`ALL_FIELDS` order.
    :rtype: List[str]
    """
    return [name for name in ALL_FIELDS if name in after and before.get(name) != after.get(name)]


def image_file_name(url: Any) -> str:
    """The file name of an image URL or path, never its host or bucket.

    An IIIF image URL (``.../<identifier>/<region>/<size>/<rotation>/default.jpg``) gives its identifier; a manifest
    URL gives ``<identifier>/manifest``.

    :param url: URL, bucket path or bare file name.
    :return: File name (``"(none)"`` when there is no path).
    :rtype: str
    """
    segments = [part for part in urlparse(str(url or "")).path.split("/") if part]
    if not segments:
        return "(none)"
    last = segments[-1]
    if _IIIF_QUALITY_RE.match(last) and len(segments) >= 5:
        return segments[-5]
    if last.lower().startswith("manifest") and len(segments) >= 2:
        return f"{segments[-2]}/{last}"
    return last


def images_diff(before: Optional[Sequence[str]], after: Optional[Sequence[str]]) -> Tuple[List[str], List[str]]:
    """Image file names added and removed between two ``image_urls`` lists.

    :param before: Served ``image_urls``.
    :param after: Target ``image_urls``.
    :return: ``(added names, removed names)``.
    :rtype: Tuple[List[str], List[str]]
    """
    old, new = list(before or []), list(after or [])
    old_set, new_set = set(old), set(new)
    return ([image_file_name(url) for url in new if url not in old_set],
            [image_file_name(url) for url in old if url not in new_set])


def partial_doc(fields: Mapping[str, Any]) -> Dict[str, Any]:
    """A partial update document from dotted field paths (Elasticsearch merges nested objects key by key).

    :param fields: ``{field path: value}``.
    :return: Nested partial document.
    :rtype: Dict[str, Any]
    """
    doc: Dict[str, Any] = {}
    for name, value in fields.items():
        head, dot, tail = name.partition(".")
        if dot:
            doc.setdefault(head, {})[tail] = copy.deepcopy(value)
        else:
            doc[name] = copy.deepcopy(value)
    return doc


# ----------------------------------------------------------------------------------------------- index reads


def check_index_exists(es: Any, index_name: str) -> None:
    """Refuse a target index that does not exist (documents are never created here).

    :param es: Elasticsearch client.
    :param index_name: Index.
    :raises ValueError: When it does not exist (the message does not name it).
    """
    if not es.indices.exists(index=index_name):
        raise ValueError("the target index does not exist; this updater only rewrites fields of existing documents")


def fetch_served(es: Any, index_name: str, ids: Sequence[str], with_transcriptions: bool = False
                 ) -> Dict[str, Optional[Dict[str, Any]]]:
    """The served ``canonical_id`` and compared fields of each document (``None`` = not in the index).

    :param es: Elasticsearch client.
    :param index_name: Index.
    :param ids: Document ids.
    :param with_transcriptions: Also read :data:`TRANSCRIPTION_FIELDS` (large; transcription-affected documents only).
    :return: ``{doc_id: _source subset or None}``.
    :rtype: Dict[str, Optional[Dict[str, Any]]]
    :raises RuntimeError: When the cluster reports an error for a document.
    """
    ids = list(ids)
    includes = READ_FIELDS + list(TRANSCRIPTION_FIELDS) if with_transcriptions else READ_FIELDS
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for start in range(0, len(ids), MGET_CHUNK):
        resp = es.mget(index=index_name, ids=ids[start:start + MGET_CHUNK], source_includes=includes)
        for doc in resp["docs"]:
            if "error" in doc:
                error = doc["error"]
                kind = error.get("type") if isinstance(error, Mapping) else "error"
                raise RuntimeError(f"reading document {doc.get('_id')} failed ({kind})")
            out[doc["_id"]] = (doc.get("_source") or {}) if doc.get("found") else None
    return out


def row_compares_transcriptions(row: Mapping[str, Any]) -> bool:
    """Whether a plan row compares the transcription fields.

    :param row: Plan document row.
    :return: True for a transcription-affected document.
    :rtype: bool
    """
    return bool(row.get("compares_transcriptions"))


def fetch_rows(es: Any, index_name: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[Dict[str, Any]]]:
    """The served sources of plan rows, reading the transcription fields only for the rows that compare them.

    :param es: Elasticsearch client.
    :param index_name: Index.
    :param rows: Plan document rows.
    :return: ``{doc_id: _source subset or None}``.
    :rtype: Dict[str, Optional[Dict[str, Any]]]
    """
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for with_transcriptions in (False, True):
        ids = [row["doc_id"] for row in rows if row_compares_transcriptions(row) == with_transcriptions]
        if ids:
            out.update(fetch_served(es, index_name, ids, with_transcriptions))
    return out


def mapped_paths(mapping_response: Mapping[str, Any]) -> Set[str]:
    """Every field path the index mapping defines (objects as dotted paths).

    :param mapping_response: ``indices.get_mapping`` response (``{index: {"mappings": ...}}``).
    :return: Dotted paths.
    :rtype: Set[str]
    """
    paths: Set[str] = set()
    for body in mapping_response.values():
        stack: List[Tuple[str, Mapping[str, Any]]] = [("", ((body or {}).get("mappings") or {}).get("properties") or {})]
        while stack:
            prefix, properties = stack.pop()
            for name, spec in properties.items():
                path = f"{prefix}{name}"
                paths.add(path)
                if isinstance(spec, Mapping) and spec.get("properties"):
                    stack.append((f"{path}.", spec["properties"]))
    return paths


def unmapped_writes(rows: Iterable[Mapping[str, Any]], mapped: Set[str], values_key: str = "after") -> Dict[str, int]:
    """Changed fields that would be written with a value the mapping has no field for (a write would add one).

    ``None`` and empty lists never add a mapping field and are not counted.

    :param rows: Plan document rows.
    :param mapped: :func:`mapped_paths` of the index.
    :param values_key: ``after`` (apply) or ``before`` (rollback).
    :return: ``{field: documents}``.
    :rtype: Dict[str, int]
    """
    counts: Dict[str, int] = {}
    for row in rows:
        if row["status"] != CHANGED:
            continue
        for name in row["changed_fields"]:
            value = row[values_key].get(name)
            if value is not None and value != [] and name not in mapped:
                counts[name] = counts.get(name, 0) + 1
    return counts


# ----------------------------------------------------------------------------------------------- plan


def plan_row(canonical_id: str, doc_id: Optional[str], source: Optional[Mapping[str, Any]],
             after: Optional[Dict[str, Any]], with_transcriptions: bool = False) -> Dict[str, Any]:
    """One plan row: the document's served and target fields and what would change.

    :param canonical_id: Canonical id.
    :param doc_id: Index ``_id`` (``None`` when the merged file has no record).
    :param source: Served ``_source`` subset (``None`` = not in the index).
    :param after: Target fields (``None`` = not in the merged file).
    :param with_transcriptions: Whether the transcription fields are compared (transcription-affected document).
    :return: Plan row (``kind: document``).
    :rtype: Dict[str, Any]
    """
    before = document_fields(source, compared_names(with_transcriptions)) if source is not None else None
    if after is None:
        status = NOT_IN_MERGED
    elif before is None:
        status = MISSING_IN_INDEX
    elif source.get("canonical_id") not in (None, canonical_id):
        status = ID_MISMATCH
    else:
        status = CHANGED if changed_fields(before, after) else UNCHANGED
    changed = changed_fields(before, after) if status == CHANGED else []
    added, removed = (images_diff(before["image_urls"], after["image_urls"]) if status == CHANGED else ([], []))
    return {"kind": "document", "doc_id": doc_id, "canonical_id": canonical_id, "status": status,
            "compares_transcriptions": with_transcriptions, "changed_fields": changed,
            "before_image_urls": before["image_urls"] if before is not None else None,
            "after_image_urls": after["image_urls"] if after is not None else None,
            "images_added": added, "images_removed": removed, "before": before, "after": after}


def build_plan_rows(affected: Sequence[str], records: Mapping[str, Mapping[str, Any]], es: Any,
                    index_name: str, transcription_affected: Iterable[str] = ()) -> List[Dict[str, Any]]:
    """Plan rows for every affected canonical id (reads the index only).

    :param affected: Canonical ids.
    :param records: Merged records by canonical id (:func:`scan_merged`).
    :param es: Elasticsearch client.
    :param index_name: Index.
    :param transcription_affected: The affected ids whose transcription fields are compared too.
    :return: Rows sorted by canonical id.
    :rtype: List[Dict[str, Any]]
    """
    with_text = set(transcription_affected)
    targets = {cid: target_fields(records[cid], cid in with_text) for cid in affected if cid in records}
    served: Dict[str, Optional[Dict[str, Any]]] = {}
    for flag in (False, True):
        ids = sorted({doc_id for cid, (doc_id, _) in targets.items() if (cid in with_text) == flag})
        if ids:
            served.update(fetch_served(es, index_name, ids, flag))
    rows = []
    for cid in sorted(affected):
        if cid not in targets:
            rows.append(plan_row(cid, None, None, None, cid in with_text))
            continue
        doc_id, after = targets[cid]
        rows.append(plan_row(cid, doc_id, served.get(doc_id), after, cid in with_text))
    return rows


def plan_header(merged_path: Any, merged_stat: os.stat_result, corrections_path: Any, corrections_sha256: str,
                corrections_rows: int, index_name: str, from_corrections: int, from_markers: int,
                created: datetime, unmapped: Mapping[str, int], max_docs: int,
                transcription_documents: int = 0) -> Dict[str, Any]:
    """The plan's header row: what it was built from (an apply checks every item).

    :param merged_path: Merged JSONL.
    :param merged_stat: Its ``stat`` taken before the scan.
    :param corrections_path: Corrections JSONL.
    :param corrections_sha256: Its hash.
    :param corrections_rows: Rows in it.
    :param index_name: Target index (stored as :func:`target_fingerprint` only).
    :param from_corrections: Canonical ids named by the corrections.
    :param from_markers: Canonical ids found through a merged-record marker.
    :param created: Plan time (UTC).
    :param unmapped: :func:`unmapped_writes` at plan time.
    :param max_docs: ``--max-docs`` of the dry run (for the report; the apply takes its own).
    :param transcription_documents: Affected canonical ids whose transcription fields are compared.
    :return: Header row (``kind: header``).
    :rtype: Dict[str, Any]
    """
    return {"kind": "header", "plan_version": PLAN_VERSION, "created_at": created.isoformat(timespec="seconds"),
            "created_at_epoch": created.timestamp(),
            "merged": {"path": str(Path(merged_path).resolve()), "size": merged_stat.st_size,
                       "mtime": merged_stat.st_mtime},
            "corrections": {"path": str(Path(corrections_path).resolve()), "sha256": corrections_sha256,
                            "rows": corrections_rows},
            "target": target_fingerprint(index_name), "ids_from_corrections": from_corrections,
            "ids_from_markers": from_markers, "transcription_documents": transcription_documents,
            "unmapped_writes": dict(unmapped), "max_docs": max_docs}


def write_jsonl(path: Any, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write rows as JSONL (parent directory created).

    :param path: Output file.
    :param rows: Rows.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_plan(path: Any) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """A plan file's header and document rows.

    :param path: Plan JSONL.
    :return: ``(header, rows)``.
    :rtype: Tuple[Dict[str, Any], List[Dict[str, Any]]]
    :raises ValueError: When the file does not start with a plan header of this version.
    """
    with open(path, encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    if not lines or lines[0].get("kind") != "header" or lines[0].get("plan_version") != PLAN_VERSION:
        raise ValueError(f"{path} is not a version-{PLAN_VERSION} image update plan: re-run the dry run")
    return lines[0], [row for row in lines[1:] if row.get("kind") == "document"]


def plan_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Counts of a plan.

    :param rows: Plan document rows.
    :return: ``documents_affected``, one count per :data:`STATUSES` entry, ``changed_fields``, ``images_added``,
        ``images_removed``, ``transcription_documents`` (rows comparing the transcription fields) and
        ``transcriptions_changed`` (changed rows whose ``transcriptions`` differ: their vectors go stale).
    :rtype: Dict[str, Any]
    """
    counts: Dict[str, Any] = {"documents_affected": len(rows), **dict.fromkeys(STATUSES, 0)}
    fields: Dict[str, int] = {}
    added = removed = 0
    for row in rows:
        counts[row["status"]] += 1
        for name in row["changed_fields"]:
            fields[name] = fields.get(name, 0) + 1
        added += len(row["images_added"])
        removed += len(row["images_removed"])
    counts.update(changed_fields=fields, images_added=added, images_removed=removed,
                  transcription_documents=sum(1 for row in rows if row_compares_transcriptions(row)),
                  transcriptions_changed=sum(1 for row in rows if "transcriptions" in row["changed_fields"]))
    return counts


# ----------------------------------------------------------------------------------------------- report


def repo_relative(path: Any) -> str:
    """A path relative to the repository when it lies inside it.

    :param path: Path.
    :return: Display path.
    :rtype: str
    """
    absolute = os.path.abspath(str(path))
    return os.path.relpath(absolute, _REPO_ROOT) if absolute.startswith(_REPO_ROOT + os.sep) else absolute


def display_value(name: str, value: Any) -> str:
    """A field value for the report: image URLs and URL lists as file names, transcription sections as editor and
    length, ``full_text_content`` as its length (never the text), other values as JSON.

    :param name: Field path.
    :param value: Value.
    :return: Markdown-safe text.
    :rtype: str
    """
    if name == "transcriptions" and isinstance(value, list):
        return "[" + ", ".join(f"`{(section or {}).get('editor')}` ({len((section or {}).get('text') or ''):,} chars)"
                               for section in value) + "]"
    if name == "full_text_content" and isinstance(value, str):
        return f"{len(value):,} chars"
    if isinstance(value, list) and (name in IMAGE_LIST_FIELDS or all(isinstance(v, str) for v in value)):
        return "[" + ", ".join(f"`{image_file_name(v)}`" for v in value) + "]"
    if isinstance(value, str) and "://" in value:
        return f"`{image_file_name(value)}`"
    return f"`{json.dumps(value, ensure_ascii=False)}`"


def render_report(header: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], plan_path: Any, max_docs: int,
                  note: Optional[str] = None) -> str:
    """The dry-run report (Markdown): counts, per-document image changes and a sample of diffs, file names only.

    :param header: Plan header.
    :param rows: Plan document rows.
    :param plan_path: Where the plan was written.
    :param max_docs: ``--max-docs``.
    :param note: Free text shown under the title (e.g. that the corrections are synthetic).
    :return: Report text.
    :rtype: str
    """
    counts = plan_counts(rows)
    changed = [row for row in rows if row["status"] == CHANGED]
    merged = header["merged"]
    modified = datetime.fromtimestamp(merged["mtime"], timezone.utc).isoformat(timespec="seconds")
    gate = (f"{counts[CHANGED]} documents would change; `--max-docs` is {max_docs}: "
            + ("within the limit." if counts[CHANGED] <= max_docs else "an apply would be REFUSED."))
    lines = ["# Site image update: dry run", "",
             f"Plan built {header['created_at']} by `src/datasets/indexing/update_image_corrections.py`. "
             "Dry run: nothing was written to the index.", ""]
    if note:
        lines += [f"> {note}", ""]
    lines += ["## Inputs", "",
              f"- Merged file: `{repo_relative(merged['path'])}` ({merged['size']:,} bytes, modified {modified})",
              f"- Corrections: `{repo_relative(header['corrections']['path'])}` ({header['corrections']['rows']} "
              f"rows, sha256 `{header['corrections']['sha256'][:12]}`)",
              "- Target: the served index (`LIVE_INDEX_NAME` unless `--index` was given; the plan stores a "
              "fingerprint, not the name)",
              f"- Plan: `{repo_relative(plan_path)}`", "",
              "## Counts", "", "| | documents |", "|---|---:|",
              f"| affected (canonical ids) | {counts['documents_affected']} |",
              f"| named by a correction | {header['ids_from_corrections']} |",
              f"| found through a merged-record marker (`images.fjp_corrections`, `images.fjp_removed`, "
              f"`transcriptions_corrections`, `transcriptions_removed`) | {header['ids_from_markers']} |",
              f"| transcription fields compared (named by a transcription correction or marked) "
              f"| {counts['transcription_documents']} |",
              f"| would change | {counts[CHANGED]} |",
              f"| unchanged | {counts[UNCHANGED]} |",
              f"| missing in the index (never created) | {counts[MISSING_IN_INDEX]} |",
              f"| not in the merged file | {counts[NOT_IN_MERGED]} |",
              f"| served `canonical_id` differs (never written) | {counts[ID_MISMATCH]} |", "",
              f"Images added: {counts['images_added']}; images removed: {counts['images_removed']} "
              "(over all changed documents, by `image_urls`).", "",
              f"Transcriptions change on {counts['transcriptions_changed']} documents; their embedding vectors are "
              "NOT recomputed by this updater (they keep the vector of the old text).", "", gate, ""]
    unmapped = header.get("unmapped_writes") or {}
    lines += ["Mapping: " + (", ".join(f"`{name}` ({n})" for name, n in sorted(unmapped.items()))
                             + " would be added by a write; an apply would be REFUSED."
                             if unmapped else "every field a write would set is already mapped."), ""]
    lines += ["## Changed fields", "", "| field | documents |", "|---|---:|"]
    lines += [f"| `{name}` | {n} |" for name, n in sorted(counts["changed_fields"].items())] or ["| (none) | 0 |"]
    lines += ["", "## Images added and removed per document", ""]
    if changed:
        lines += ["| canonical id | images before | images after | added | removed |", "|---|---:|---:|---|---|"]
        for row in changed:
            lines.append(f"| {row['canonical_id']} | {len(row['before_image_urls'] or [])} | "
                         f"{len(row['after_image_urls'] or [])} | "
                         f"{', '.join(f'`{n}`' for n in row['images_added']) or '-'} | "
                         f"{', '.join(f'`{n}`' for n in row['images_removed']) or '-'} |")
    else:
        lines.append("No document would change.")
    text_rows = [row for row in changed if set(row["changed_fields"]) & set(TRANSCRIPTION_FIELDS)]
    if text_rows:
        lines += ["", "## Transcription changes per document", "",
                  "| canonical id | sections before | sections after | characters before | characters after |",
                  "|---|---:|---:|---:|---:|"]
        for row in text_rows:
            before_sections = row["before"].get("transcriptions") or []
            after_sections = row["after"].get("transcriptions") or []
            lines.append(f"| {row['canonical_id']} | {len(before_sections)} | {len(after_sections)} | "
                         f"{sum(len((s or {}).get('text') or '') for s in before_sections):,} | "
                         f"{sum(len((s or {}).get('text') or '') for s in after_sections):,} |")
    lines += ["", f"## Sample diffs (first {SAMPLE_DIFFS} changed documents; image file names only, no text)", ""]
    for row in changed[:SAMPLE_DIFFS]:
        lines += [f"### {row['canonical_id']}", ""]
        lines += [f"- `{name}`: {display_value(name, row['before'][name])} -> {display_value(name, row['after'][name])}"
                  for name in row["changed_fields"]]
        lines.append("")
    if not changed:
        lines += ["(none)", ""]
    for status, title in ((MISSING_IN_INDEX, "Missing in the index"), (NOT_IN_MERGED, "Not in the merged file"),
                          (ID_MISMATCH, "Served canonical id differs")):
        ids = [row["canonical_id"] for row in rows if row["status"] == status]
        if ids:
            lines += [f"## {title}", "", ", ".join(ids), ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------- dry run


def dry_run(merged_path: Any = DEFAULT_MERGED, corrections_path: Any = DEFAULT_CORRECTIONS, *,
            index_name: str = LIVE_INDEX_NAME, plan_path: Any = DEFAULT_PLAN, report_path: Any = DEFAULT_REPORT,
            max_docs: int = DEFAULT_MAX_DOCS, note: Optional[str] = None, es_client: Optional[Any] = None
            ) -> Dict[str, Any]:
    """Diff the served fields of every affected document with the merged file; write the plan and report.

    Reads only (``indices.exists``, ``mget``, ``indices.get_mapping``); never writes to the index.

    :param merged_path: Merged JSONL (the real file, or a dry-merge output).
    :param corrections_path: Corrections JSONL.
    :param index_name: Target index (default: the served index).
    :param plan_path: Plan output.
    :param report_path: Report output.
    :param max_docs: Apply limit to report against.
    :param note: Line shown under the report title.
    :param es_client: Elasticsearch client (default: from the repository ``.env``).
    :return: :func:`plan_counts` plus ``would_refuse`` reasons.
    :rtype: Dict[str, Any]
    """
    merged_stat = Path(merged_path).stat()
    corrections = read_corrections(corrections_path)
    corrections_sha256 = file_sha256(corrections_path)
    ids = correction_ids(corrections)
    records, marked = scan_merged(merged_path, ids)
    if Path(merged_path).stat().st_mtime != merged_stat.st_mtime:
        raise RuntimeError("the merged file changed while it was being read; run the dry run again")
    affected = sorted(ids | marked)
    text_ids = transcription_correction_ids(corrections) | {
        cid for cid, record in records.items() if has_transcription_marker(record)}
    es = es_client if es_client is not None else default_es_client()
    check_index_exists(es, index_name)
    rows = build_plan_rows(affected, records, es, index_name, text_ids)
    unmapped = unmapped_writes(rows, mapped_paths(es.indices.get_mapping(index=index_name)))
    header = plan_header(merged_path, merged_stat, corrections_path, corrections_sha256, len(corrections),
                         index_name, len(ids), len(marked - ids), datetime.now(timezone.utc), unmapped, max_docs,
                         transcription_documents=len(text_ids & set(affected)))
    write_jsonl(plan_path, [header, *rows])
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(render_report(header, rows, plan_path, max_docs, note) + "\n", encoding="utf-8")
    counts = plan_counts(rows)
    refuse = ([f"{counts[CHANGED]} > --max-docs {max_docs}"] if counts[CHANGED] > max_docs else [])
    refuse += [f"unmapped field {name}" for name in sorted(unmapped)]
    summary = {**counts, "would_refuse": refuse, "plan": repo_relative(plan_path),
               "report": repo_relative(report_path)}
    print(f"dry run: {counts['documents_affected']} affected, {counts[CHANGED]} would change, "
          f"{counts[UNCHANGED]} unchanged, {counts[MISSING_IN_INDEX]} missing in the index, "
          f"{counts[NOT_IN_MERGED]} not in the merged file, {counts[ID_MISMATCH]} id mismatch; images +"
          f"{counts['images_added']} / -{counts['images_removed']}; transcriptions change on "
          f"{counts['transcriptions_changed']} (vectors not recomputed)"
          + (f"; an apply would be refused ({'; '.join(refuse)})" if refuse else ""), flush=True)
    print(f"plan {summary['plan']}, report {summary['report']}; nothing was written to the index", flush=True)
    return summary


# ----------------------------------------------------------------------------------------------- apply / rollback


def check_plan_current(header: Mapping[str, Any], plan_path: Any, merged_path: Any, corrections_path: Any) -> None:
    """Refuse a plan that no longer describes the merged and corrections files.

    :param header: Plan header.
    :param plan_path: Plan file.
    :param merged_path: Merged JSONL the apply is run with.
    :param corrections_path: Corrections JSONL the apply is run with.
    :raises ApplyRefused: When the plan was built from other files, is older than the merged file, or the merged or
        corrections file changed since.
    """
    merged = Path(merged_path).resolve()
    recorded = header["merged"]
    if str(merged) != recorded["path"]:
        raise ApplyRefused(f"the plan was built from {recorded['path']}, not {merged}: re-run the dry run")
    stat = merged.stat()
    if stat.st_mtime > Path(plan_path).stat().st_mtime or stat.st_mtime > header["created_at_epoch"]:
        raise ApplyRefused("the plan is older than the merged file: re-run the dry run")
    if (stat.st_size, stat.st_mtime) != (recorded["size"], recorded["mtime"]):
        raise ApplyRefused("the merged file changed since the plan was built: re-run the dry run")
    corrections = Path(corrections_path).resolve()
    if str(corrections) != header["corrections"]["path"]:
        raise ApplyRefused(f"the plan was built from {header['corrections']['path']}, not {corrections}: "
                           "re-run the dry run")
    if file_sha256(corrections) != header["corrections"]["sha256"]:
        raise ApplyRefused("the corrections file changed since the plan was built: re-run the dry run")


def check_live_gate(index_name: str, allow_live: bool) -> None:
    """Refuse a write to the served index (or a previous one) without ``--i-know-this-is-live``.

    :param index_name: Target index of this run.
    :param allow_live: ``--i-know-this-is-live`` was passed.
    :raises ApplyRefused: For a name in ``LIVE_INDEX_NAMES`` without *allow_live* (the message does not name it).
    """
    if index_name in LIVE_INDEX_NAMES and not allow_live:
        raise ApplyRefused("the target is the served index: add --i-know-this-is-live to --apply")


def check_target_and_size(header: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], index_name: str,
                          max_docs: int) -> List[Dict[str, Any]]:
    """Refuse another index or too many documents; return the rows to act on.

    :param header: Plan header.
    :param rows: Plan document rows.
    :param index_name: Target index of this run.
    :param max_docs: Most documents one run may change.
    :return: The ``changed`` rows.
    :rtype: List[Dict[str, Any]]
    :raises ApplyRefused: For another index or more than ``max_docs`` changed rows.
    """
    if header["target"] != target_fingerprint(index_name):
        raise ApplyRefused("the plan was built against another index: re-run the dry run against this one")
    changed = [dict(row) for row in rows if row["status"] == CHANGED]
    if len(changed) > max_docs:
        raise ApplyRefused(f"{len(changed)} documents would change, more than --max-docs {max_docs}")
    return changed


def document_state(row: Mapping[str, Any], source: Optional[Mapping[str, Any]], rollback: bool) -> str:
    """Where a served document stands relative to the plan: still to write, already written, or neither.

    :param row: Plan row (``changed``).
    :param source: Served ``_source`` subset (``None`` = not in the index).
    :param rollback: Compare for a rollback (start = after, goal = before) instead of an apply.
    :return: :data:`TO_WRITE`, :data:`DONE`, :data:`DRIFTED`, :data:`MISSING_IN_INDEX` or :data:`ID_MISMATCH`.
    :rtype: str
    """
    if source is None:
        return MISSING_IN_INDEX
    if source.get("canonical_id") not in (None, row["canonical_id"]):
        return ID_MISMATCH
    current = document_fields(source, row["after"])
    start, goal = (row["after"], row["before"]) if rollback else (row["before"], row["after"])
    if all(current[name] == goal[name] for name in row["changed_fields"]):
        return DONE
    if all(current[name] == start[name] for name in row["changed_fields"]):
        return TO_WRITE
    return DRIFTED


def chunked(items: Sequence[Any], size: int) -> Iterator[List[Any]]:
    """Consecutive lists of at most ``size`` items.

    :param items: Items.
    :param size: Chunk size.
    :return: Iterator over chunks.
    :rtype: Iterator[List[Any]]
    """
    iterator = iter(items)
    while chunk := list(islice(iterator, size)):
        yield chunk


def write_partial_updates(es: Any, index_name: str, updates: Sequence[Tuple[str, Dict[str, Any]]]
                          ) -> Dict[str, str]:
    """One bulk call of partial ``update`` actions (never a reindex), waiting for the refresh.

    :param es: Elasticsearch client.
    :param index_name: Index.
    :param updates: ``(doc_id, partial document)`` pairs.
    :return: ``{doc_id: error type}`` for the actions the cluster rejected.
    :rtype: Dict[str, str]
    """
    actions = [{"_op_type": "update", "_index": index_name, "_id": doc_id, "doc": doc} for doc_id, doc in updates]
    _, errors = bulk(client=es, actions=actions, refresh="wait_for", raise_on_error=False)
    failures: Dict[str, str] = {}
    for item in errors:
        info = next(iter(item.values()))
        error = info.get("error")
        failures[str(info.get("_id"))] = error.get("type") if isinstance(error, Mapping) else str(info.get("status"))
    return failures


def served_states(es: Any, index_name: str, changed: Sequence[Mapping[str, Any]], rollback: bool) -> Dict[str, str]:
    """Read every planned document and check it holds the values the run starts from (or already its goal).

    :param es: Elasticsearch client.
    :param index_name: Index.
    :param changed: Plan rows with status ``changed``.
    :param rollback: Check for a rollback (start = after values) instead of an apply (start = before values).
    :return: ``{doc_id: TO_WRITE or DONE}``.
    :rtype: Dict[str, str]
    :raises ApplyRefused: When any document is missing, has another canonical id or holds neither the start nor the
        goal values.
    """
    served = fetch_rows(es, index_name, changed)
    states = {row["doc_id"]: document_state(row, served.get(row["doc_id"]), rollback) for row in changed}
    blocked = sorted(doc_id for doc_id, state in states.items() if state not in (TO_WRITE, DONE))
    if blocked:
        expected = "after" if rollback else "before"
        raise ApplyRefused(f"{len(blocked)} documents do not hold the plan's {expected} values "
                           f"(e.g. {', '.join(blocked[:5])}): re-run the dry run")
    return states


def execute_rows(es: Any, index_name: str, changed: Sequence[Mapping[str, Any]], states: Mapping[str, str], *,
                 rollback: bool, execute: bool, result_path: Any, run_header: Mapping[str, Any]) -> Dict[str, Any]:
    """Record every document's state and (``execute``) write, read back and record each one still to write.

    :param es: Elasticsearch client.
    :param index_name: Index.
    :param changed: Plan rows with status ``changed``.
    :param states: :func:`served_states` output.
    :param rollback: Write the before values instead of the after values.
    :param execute: Write (``--apply``); otherwise only record the states.
    :param result_path: Result JSONL (header row, one row per document, summary row).
    :param run_header: Fields for the result header row.
    :return: Summary counts.
    :rtype: Dict[str, Any]
    """
    goal_key = "before" if rollback else "after"
    to_write = [row for row in changed if states[row["doc_id"]] == TO_WRITE]
    summary: Dict[str, Any] = {"documents": len(changed), "to_write": len(to_write),
                               "already_done": len(changed) - len(to_write), "written": 0, "verified": 0,
                               "failed": 0, "unverified": 0, "dry_run": not execute}
    rows_out: List[Dict[str, Any]] = [{"kind": "header", **run_header, "dry_run": not execute}]
    rows_out += [{"kind": "document", "doc_id": row["doc_id"], "canonical_id": row["canonical_id"],
                  "action": "already_done"} for row in changed if states[row["doc_id"]] == DONE]
    if not execute:
        rows_out += [{"kind": "document", "doc_id": row["doc_id"], "canonical_id": row["canonical_id"],
                      "action": "would_write", "fields": row["changed_fields"]} for row in to_write]
        write_jsonl(result_path, [*rows_out, {"kind": "summary", **summary}])
        return summary
    Path(result_path).parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as out:
        for row in rows_out:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        for chunk in chunked(to_write, WRITE_CHUNK):
            failures = write_partial_updates(es, index_name, [
                (row["doc_id"], partial_doc({name: row[goal_key][name] for name in row["changed_fields"]}))
                for row in chunk])
            readback = fetch_rows(es, index_name, chunk)
            for row in chunk:
                source = readback.get(row["doc_id"])
                current = document_fields(source, row["after"]) if source is not None else None
                mismatched = (list(row["changed_fields"]) if current is None else
                              [name for name in row["changed_fields"] if current[name] != row[goal_key][name]])
                error = failures.get(row["doc_id"])
                summary["failed"] += error is not None
                summary["written"] += error is None
                summary["verified"] += not mismatched
                summary["unverified"] += bool(mismatched)
                out.write(json.dumps({"kind": "document", "doc_id": row["doc_id"],
                                      "canonical_id": row["canonical_id"], "action": "written",
                                      "fields": row["changed_fields"], "error": error, "verified": not mismatched,
                                      "mismatched_fields": mismatched}, ensure_ascii=False) + "\n")
            out.flush()
            logger.info("written %d / %d documents", summary["written"] + summary["failed"], len(to_write))
        out.write(json.dumps({"kind": "summary", **summary,
                              "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")
    return summary


def archive_plan(plan_path: Any, stamp: str) -> Path:
    """Copy the plan next to itself before an apply; the copy holds the before values a rollback restores.

    :param plan_path: Plan file.
    :param stamp: UTC time stamp for the name.
    :return: The copy's path.
    :rtype: Path
    """
    plan = Path(plan_path)
    archive = plan.with_name(f"{plan.stem}.applied-{stamp}{plan.suffix}")
    shutil.copy2(plan, archive)
    return archive


def apply_plan(plan_path: Any = DEFAULT_PLAN, *, merged_path: Any = DEFAULT_MERGED,
               corrections_path: Any = DEFAULT_CORRECTIONS, index_name: str = LIVE_INDEX_NAME,
               max_docs: int = DEFAULT_MAX_DOCS, result_path: Any = DEFAULT_RESULT,
               es_client: Optional[Any] = None, allow_live: bool = False) -> Dict[str, Any]:
    """Apply a dry-run plan to the index (the ``--apply`` path): checks, archive copy, writes, read-back.

    :param plan_path: Plan written by :func:`dry_run`.
    :param merged_path: Merged JSONL (must be the plan's, unchanged since).
    :param corrections_path: Corrections JSONL (must be the plan's, unchanged since).
    :param index_name: Target index (must be the plan's).
    :param max_docs: Most documents this run may change.
    :param result_path: Result JSONL.
    :param es_client: Elasticsearch client (default: from the repository ``.env``).
    :param allow_live: ``--i-know-this-is-live``: required when *index_name* is the served index.
    :return: Summary counts plus ``archived_plan``.
    :rtype: Dict[str, Any]
    :raises ApplyRefused: When a refusal rule applies (nothing is written).
    """
    check_live_gate(index_name, allow_live)
    header, rows = read_plan(plan_path)
    check_plan_current(header, plan_path, merged_path, corrections_path)
    changed = check_target_and_size(header, rows, index_name, max_docs)
    if not changed:
        print("apply: the plan changes no document; nothing written", flush=True)
        return {"documents": 0, "written": 0, "dry_run": False}
    es = es_client if es_client is not None else default_es_client()
    check_index_exists(es, index_name)
    unmapped = unmapped_writes(changed, mapped_paths(es.indices.get_mapping(index=index_name)), "after")
    if unmapped:
        raise ApplyRefused(f"a write would add unmapped fields: {', '.join(sorted(unmapped))}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    states = served_states(es, index_name, changed, rollback=False)
    archive = archive_plan(plan_path, stamp)
    summary = execute_rows(es, index_name, changed, states, rollback=False, execute=True, result_path=result_path,
                           run_header={"mode": "apply", "started_at": stamp, "plan": repo_relative(archive)})
    summary["archived_plan"] = repo_relative(archive)
    print(f"apply: {summary['written']} written ({summary['verified']} verified, {summary['unverified']} not "
          f"verified, {summary['failed']} failed), {summary['already_done']} already applied; result "
          f"{repo_relative(result_path)}; roll back with --rollback {summary['archived_plan']}", flush=True)
    return summary


def rollback_plan(plan_path: Any, *, execute: bool = False, index_name: str = LIVE_INDEX_NAME,
                  max_docs: int = DEFAULT_MAX_DOCS, result_path: Any = DEFAULT_ROLLBACK_RESULT,
                  es_client: Optional[Any] = None, allow_live: bool = False) -> Dict[str, Any]:
    """Restore a plan's before values (``--rollback PLAN``); writes only with ``execute`` (``--apply``).

    The merged-file freshness checks do not apply (a rollback restores what the index held), the index, size and
    per-document state checks do.

    :param plan_path: The plan an apply used (its ``*.applied-*`` copy).
    :param execute: Write (``--apply``); otherwise check and record only.
    :param index_name: Target index (must be the plan's).
    :param max_docs: Most documents this run may change.
    :param result_path: Result JSONL.
    :param es_client: Elasticsearch client (default: from the repository ``.env``).
    :param allow_live: ``--i-know-this-is-live``: required with *execute* when *index_name* is the served index.
    :return: Summary counts.
    :rtype: Dict[str, Any]
    :raises ApplyRefused: When a refusal rule applies (nothing is written).
    """
    if execute:
        check_live_gate(index_name, allow_live)
    header, rows = read_plan(plan_path)
    changed = check_target_and_size(header, rows, index_name, max_docs)
    if not changed:
        print("rollback: the plan changes no document; nothing to restore", flush=True)
        return {"documents": 0, "written": 0, "dry_run": not execute}
    es = es_client if es_client is not None else default_es_client()
    check_index_exists(es, index_name)
    unmapped = unmapped_writes(changed, mapped_paths(es.indices.get_mapping(index=index_name)), "before")
    if unmapped:
        raise ApplyRefused(f"a restore would add unmapped fields: {', '.join(sorted(unmapped))}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    states = served_states(es, index_name, changed, rollback=True)
    summary = execute_rows(es, index_name, changed, states, rollback=True, execute=execute, result_path=result_path,
                           run_header={"mode": "rollback", "started_at": stamp, "plan": repo_relative(plan_path)})
    if execute:
        print(f"rollback: {summary['written']} restored ({summary['verified']} verified, {summary['unverified']} "
              f"not verified, {summary['failed']} failed), {summary['already_done']} already restored; result "
              f"{repo_relative(result_path)}", flush=True)
    else:
        print(f"rollback check: {summary['to_write']} documents would be restored, {summary['already_done']} "
              f"already hold the before values; nothing written (add --apply to restore)", flush=True)
    return summary


# ----------------------------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser (defaults are code constants; help never prints the index name).

    :return: Parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merged", default=DEFAULT_MERGED, help="Merged JSONL (default: the merged corpus file).")
    parser.add_argument("--corrections", default=DEFAULT_CORRECTIONS,
                        help="Image corrections JSONL (default: image_corrections.jsonl next to the merged file).")
    parser.add_argument("--index", default=LIVE_INDEX_NAME,
                        help="Target index (default: the served index, LIVE_INDEX_NAME in index_merged_genizah).")
    parser.add_argument("--plan", default=DEFAULT_PLAN, help="Plan file the dry run writes and --apply reads.")
    parser.add_argument("--report", default=DEFAULT_REPORT, help="Dry-run report (Markdown).")
    parser.add_argument("--result", default=None,
                        help="Result file of --apply / --rollback (default: site_image_update_result.jsonl or "
                             "site_image_rollback_result.jsonl in logs/fgp_native_1030).")
    parser.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS,
                        help="Refuse to write when more documents would change (default: %(default)s).")
    parser.add_argument("--note", default=None, help="Line shown under the dry-run report title.")
    parser.add_argument("--rollback", default=None, metavar="PLAN",
                        help="Restore the before values of PLAN (an applied plan copy); writes only with --apply.")
    parser.add_argument("--apply", action="store_true",
                        help="Write to the index (apply --plan, or with --rollback restore). Without it nothing is "
                             "written.")
    parser.add_argument("--i-know-this-is-live", dest="allow_live", action="store_true",
                        help="Second gate: needed together with --apply (and with --rollback --apply) when the "
                             "target is the served index or a previous one (LIVE_INDEX_NAMES).")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    :param argv: Arguments (None = ``sys.argv``).
    :return: Exit status (0 ok, 1 a write failed or did not verify, 2 refused).
    :rtype: int
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    for noisy in ("elastic_transport", "elasticsearch"):    # their records name the cluster host
        logging.getLogger(noisy).setLevel(logging.CRITICAL)
    args = build_parser().parse_args(argv)
    try:
        if args.rollback:
            summary = rollback_plan(args.rollback, execute=args.apply, index_name=args.index, max_docs=args.max_docs,
                                    result_path=args.result or DEFAULT_ROLLBACK_RESULT, allow_live=args.allow_live)
        elif args.apply:
            summary = apply_plan(args.plan, merged_path=args.merged, corrections_path=args.corrections,
                                 index_name=args.index, max_docs=args.max_docs,
                                 result_path=args.result or DEFAULT_RESULT, allow_live=args.allow_live)
        else:
            dry_run(args.merged, args.corrections, index_name=args.index, plan_path=args.plan,
                    report_path=args.report, max_docs=args.max_docs, note=args.note)
            return 0
    except ApplyRefused as refusal:
        logger.error("refused, nothing written: %s", refusal)
        return 2
    return 1 if summary.get("failed") or summary.get("unverified") else 0


if __name__ == "__main__":
    raise SystemExit(main())
