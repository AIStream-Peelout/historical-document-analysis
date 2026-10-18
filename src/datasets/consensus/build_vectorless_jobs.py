# File name: build_vectorless_jobs.py
# Date: 10/8/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Consensus (two-reader) job pool for the merged records that index v9 stores without a vector.

A record gets no vector in ``genizah_merged_v9`` when its ``semantic_v3`` text is empty, under 40 characters, or a
boilerplate label plus ``Language`` / ``Document Type`` / ``Date`` lines (``src.embeddings.semantic_text``).  Nothing
finds those records by meaning until something describes them; reading their images with the consensus pipeline
(``src.datasets.consensus.two_reader_lines``) is the first step towards a generated description.

Job format: the one ``src/datasets/consensus/build_ai_read_jobs.py`` and ``logs/routed/build_routed_jobs.py`` write,
one JSON object per image, ``{doc_id, image_index, image_url, document_type, institution, queue, reread_class}``, where
``doc_id`` is the Elasticsearch ``_id`` and ``image_index`` the position in the index's ``image_urls``.  Those builders
read both from Elasticsearch (``_source`` of the newest merged index); this one rebuilds them from the merged export
with the code the v9 indexer runs (``GenizahDocument.from_merged_format`` for ``image_urls``, ``es_document_id`` for
the ``_id``, ``to_elasticsearch_document`` for ``document_type`` / ``institution``), so the rows equal what
``genizah_merged_v9`` serves when it is built from the same merged file.  The pipeline's ``--ids`` loader keeps only
``doc_id`` / ``image_index`` / ``image_url`` (``two_reader_lines.build_jobs``); every other field is informational.

Tranches (``queue``): ``V1_pgp_documentary`` (the record has a PGP source), ``V2_ktiv_literary`` (KTIV source, no
PGP), ``V3_legacy_scraps`` (neither: legacy image store and a few Bodleian images, no PGP/KTIV catalogue text).
``reread_class`` is ``already_read`` when an earlier consensus run (``ai_reads_*.jsonl``, failure logs excluded) holds
a read of that exact image URL, else ``never_read``.  Extra fields: ``pool``, ``vectorless_reason``,
``image_source``, ``doc_n_images``; ``read_by`` on already-read images; ``pending_in`` on never-read images that a
not-yet-run routed job list already holds; ``also_in`` / ``also_in_fragment`` on images that other merged records
list too (box-level records repeat their fragments' images; duplicate records share one fragment's images).

Usage (repo root)::

    python -m src.datasets.consensus.build_vectorless_jobs

Writes ``jobs_vectorless_pool.jsonl``, ``jobs_vectorless_pilot200.jsonl``, ``iiif_only_docs.txt`` and
``vectorless_pool_summary.md`` next to this script and prints a one-line JSON tally.  Read-only everywhere else.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import logging
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import AbstractSet, Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))  # ``src`` imports when run by path
OUT_DIR = Path(__file__).resolve().parent
MERGED = REPO / "src/datasets/raw_data/cairo_genizah/merged/merged_shelfmarks.jsonl"
AI_READS = REPO / "src/datasets/raw_data/cairo_genizah/ai_reads"
ROUTED = REPO / "logs/routed"
#: Job lists of earlier runs whose rows carry a HEAD 404 for the image URL.
UNREACHABLE_LISTS = (AI_READS / "jobs_documentary_v5_unreachable_0916.jsonl",
                     AI_READS / "jobs_documentary_v6_unreachable_0922.jsonl")
#: Routed job lists whose pass has not (fully) run yet: a never-read pool image listed there will be read anyway.
PENDING_LISTS = (ROUTED / "jobs_routed_literary.jsonl", ROUTED / "jobs_routed_documentary.jsonl")
#: v8-served ``(doc_id, image_index, image_url)`` triples, to measure drift between v8 and the rebuilt lists.
V8_REFERENCE_JOBS = ROUTED / "jobs_literary_v8_0928.jsonl"
V8_REFERENCE_READS = AI_READS / "ai_reads_qwen3-vl-8b-heb-v22i-a50.jsonl"

POOL_NAME = "vectorless_v9"
TEXT_VERSION = "semantic_v3"
V1, V2, V3 = "V1_pgp_documentary", "V2_ktiv_literary", "V3_legacy_scraps"
TRANCHES = (V1, V2, V3)
PILOT_QUOTAS: Dict[str, int] = {V1: 70, V2: 70, V3: 60}
NEVER_READ, ALREADY_READ, PARTIALLY_READ, UNRESOLVED = "never_read", "already_read", "partially_read", "unresolved"
DOC_CLASSES = (NEVER_READ, PARTIALLY_READ, ALREADY_READ, UNRESOLVED)
#: ``semantic_ineligible_reason`` value -> ``vectorless_reason`` label.
REASONS: Dict[str, str] = {"empty": "empty", "short": "short", "boilerplate_only": "boilerplate"}
REASON_LABELS = ("empty", "short", "boilerplate")
LEGACY_IMAGE_KEY = "fjp"  # pre-existing merged-record key of the legacy image store (read only)
#: Base URL the index builder puts in front of bare legacy-store file names (``genizah_document``).
LEGACY_IMAGE_BASE = "https://storage.googleapis.com/cairo-genizah-es-json/images"
#: Index order after the merge's preferred source (``ktiv`` > ``bodleian`` > legacy store).
SOURCE_ORDER = ("ktiv", "bodleian", "legacy")
#: Records with at least this many images are box- or volume-level containers rather than one fragment.
CONTAINER_MIN_IMAGES = 20
#: ``also_in`` lists at most this many other records per image.
MAX_CO_OWNERS = 5
#: Studio throughput of the consensus pipeline (build_ai_read_jobs.py): ~55 s/image with prefetch, 90 s pessimistic.
SECONDS_PER_IMAGE = (55, 90)
#: The brief's availability tally (one class per record: legacy > KTIV local > KTIV IIIF-only > Bodleian > none).
EXPECTED = {"vectorless": 14461, "legacy_store": 9504, "ktiv_local": 2083, "ktiv_iiif_only": 481, "bodleian": 40,
            "none": 2353}
VLM_DOCUMENTARY = ("qwen3-vl-8b-heb-v22i-a50", "interp-v21b1200-v22a1300")
VLM_LITERARY = ("qwen3-vl-8b-heb-v21b-step1200", "6724c32cd0f6296223978ae69635fc782d2dbb03")
HTR_KEY = "MiDRASH_Gen_01@k7.0.3-ktivseg-e10-mv"


# ---------------------------------------------------------------------------
# Records and eligibility
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield the objects of a JSONL file, skipping blank lines and a torn line.

    A records file held by a paused writer can end in a half-written line; it is skipped, not fatal.

    :param path: JSONL file.
    :type path: Path
    :return: Iterator of parsed objects.
    :rtype: Iterator[Dict[str, Any]]
    """
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def iter_merged_records(path: Path) -> Iterator[Dict[str, Any]]:
    """Raw merged records in exactly the order ``iter_merged_documents`` yields documents (non-blank lines).

    :param path: ``merged_shelfmarks.jsonl``.
    :type path: Path
    :return: Iterator of merged records.
    :rtype: Iterator[Dict[str, Any]]
    """
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def vectorless_reason(ineligible_reason: str) -> Optional[str]:
    """Job label for a ``semantic_ineligible_reason`` value.

    :param ineligible_reason: ``empty``, ``short``, ``boilerplate_only`` or ``eligible``.
    :type ineligible_reason: str
    :return: ``empty`` / ``short`` / ``boilerplate``, or ``None`` when the record gets a vector.
    :rtype: Optional[str]
    :raises ValueError: For a value the eligibility module does not produce.
    """
    if ineligible_reason == "eligible":
        return None
    if ineligible_reason not in REASONS:
        raise ValueError(f"unknown ineligibility reason {ineligible_reason!r}")
    return REASONS[ineligible_reason]


def sources_of(record: Mapping[str, Any]) -> Set[str]:
    """Catalogue sources of a merged record: ``sources_present`` plus every non-empty ``sources`` block.

    :param record: Merged record.
    :type record: Mapping[str, Any]
    :return: Source keys.
    :rtype: Set[str]
    """
    present = {str(s) for s in record.get("sources_present") or []}
    present.update(k for k, v in (record.get("sources") or {}).items() if v)
    return present


def assign_tranche(record: Mapping[str, Any]) -> str:
    """Tranche of a vector-less record.

    :param record: Merged record.
    :type record: Mapping[str, Any]
    :return: :data:`V1` with a PGP source, else :data:`V2` with a KTIV source, else :data:`V3`.
    :rtype: str
    """
    present = sources_of(record)
    if "pgp" in present:
        return V1
    if "ktiv" in present:
        return V2
    return V3


# ---------------------------------------------------------------------------
# Image resolution (the index's image_urls, attributed per source)
# ---------------------------------------------------------------------------

def source_name(key: Optional[str]) -> Optional[str]:
    """Report name of a merged image-source key (the legacy store's key becomes ``legacy``).

    :param key: ``ktiv``, ``bodleian``, the legacy-store key, or ``None``.
    :type key: Optional[str]
    :return: Name used in the outputs.
    :rtype: Optional[str]
    """
    return "legacy" if key == LEGACY_IMAGE_KEY else key


def per_source_image_urls(images: Mapping[str, Any]) -> Dict[str, List[str]]:
    """Absolute image URLs of each source, as the index builder forms them.

    Legacy-store entries are bare file names under :data:`LEGACY_IMAGE_BASE` or already-absolute URLs; KTIV and
    Bodleian blocks carry ``image_urls`` (filled once their images are on the bucket).

    :param images: The merged record's ``images`` block.
    :type images: Mapping[str, Any]
    :return: ``{"ktiv": [...], "bodleian": [...], "legacy": [...]}``.
    :rtype: Dict[str, List[str]]
    """
    legacy = [img if str(img).startswith("http") else f"{LEGACY_IMAGE_BASE}/{img}"
              for img in (images.get(LEGACY_IMAGE_KEY) or [])]
    return {"ktiv": list((images.get("ktiv") or {}).get("image_urls") or []),
            "bodleian": list((images.get("bodleian") or {}).get("image_urls") or []),
            "legacy": legacy}


def ordered_image_urls(images: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """``(source, url)`` pairs in the index's ``image_urls`` order: preferred source first, then :data:`SOURCE_ORDER`.

    :param images: The merged record's ``images`` block.
    :type images: Mapping[str, Any]
    :return: Ordered pairs; position = ``image_index``.
    :rtype: List[Tuple[str, str]]
    """
    by_source = per_source_image_urls(images)
    order = list(SOURCE_ORDER)
    preferred = source_name(images.get("preferred_source"))
    if preferred in order:
        order.remove(preferred)
        order.insert(0, preferred)
    return [(source, url) for source in order for url in by_source[source]]


def ktiv_image_status(images: Mapping[str, Any]) -> str:
    """State of a record's KTIV images.

    :param images: The merged record's ``images`` block.
    :type images: Mapping[str, Any]
    :return: ``gcs`` (URLs listed), ``zip_unpopulated`` (local zips, no URLs), ``iiif_only`` (manifest only), ``none``.
    :rtype: str
    """
    ktiv = images.get("ktiv") or {}
    if ktiv.get("image_urls"):
        return "gcs"
    if ktiv.get("zip_files"):
        return "zip_unpopulated"
    if ktiv.get("iiif_manifest_url"):
        return "iiif_only"
    return "none"


def bodleian_image_status(images: Mapping[str, Any]) -> str:
    """State of a record's Bodleian images.

    :param images: The merged record's ``images`` block.
    :type images: Mapping[str, Any]
    :return: ``gcs`` (URLs listed), ``unpopulated`` (block without URLs) or ``none``.
    :rtype: str
    """
    bodleian = images.get("bodleian") or {}
    if bodleian.get("image_urls"):
        return "gcs"
    return "unpopulated" if bodleian else "none"


def availability_class(images: Mapping[str, Any]) -> str:
    """One availability class per record, with the brief's precedence.

    :param images: The merged record's ``images`` block.
    :type images: Mapping[str, Any]
    :return: ``legacy_store`` > ``ktiv_local`` (URLs or local zips) > ``ktiv_iiif_only`` > ``bodleian`` > ``none``.
    :rtype: str
    """
    if images.get(LEGACY_IMAGE_KEY):
        return "legacy_store"
    ktiv = ktiv_image_status(images)
    if ktiv in ("gcs", "zip_unpopulated"):
        return "ktiv_local"
    if ktiv == "iiif_only":
        return "ktiv_iiif_only"
    if bodleian_image_status(images) != "none":
        return "bodleian"
    return "none"


def unresolved_reason(images: Mapping[str, Any]) -> str:
    """Why a record has no image the index serves (call only when :func:`ordered_image_urls` is empty).

    :param images: The merged record's ``images`` block.
    :type images: Mapping[str, Any]
    :return: ``ktiv_iiif_only``, ``ktiv_zip_not_populated``, ``bodleian_not_populated`` or ``no_images``.
    :rtype: str
    """
    ktiv = ktiv_image_status(images)
    if ktiv == "iiif_only":
        return "ktiv_iiif_only"
    if ktiv == "zip_unpopulated":
        return "ktiv_zip_not_populated"
    if bodleian_image_status(images) == "unpopulated":
        return "bodleian_not_populated"
    return "no_images"


@dataclass
class Candidate:
    """One vector-less record with everything the job rows and the summary need."""

    doc_id: str
    tranche: str
    reason: str
    document_type: Optional[str]
    institution: Optional[str]
    images: List[Tuple[str, str]]
    availability: str
    ktiv_status: str
    bodleian_status: str
    sys_num: Optional[str] = None
    iiif_manifest_url: Optional[str] = None
    unresolved: Optional[str] = None


def make_candidate(record: Mapping[str, Any], doc_id: str, reason: str, document_type: Optional[str],
                   institution: Optional[str]) -> Candidate:
    """Build a :class:`Candidate` from a merged record and its index fields.

    :param record: Merged record.
    :type record: Mapping[str, Any]
    :param doc_id: Elasticsearch ``_id``.
    :type doc_id: str
    :param reason: ``vectorless_reason`` label.
    :type reason: str
    :param document_type: Index ``document_type`` (may be ``None``).
    :type document_type: Optional[str]
    :param institution: Index ``institution`` (may be ``None``).
    :type institution: Optional[str]
    :return: The candidate.
    :rtype: Candidate
    """
    images = record.get("images") or {}
    ktiv = images.get("ktiv") or {}
    pairs = ordered_image_urls(images)
    return Candidate(doc_id=doc_id, tranche=assign_tranche(record), reason=reason, document_type=document_type,
                     institution=institution, images=pairs, availability=availability_class(images),
                     ktiv_status=ktiv_image_status(images), bodleian_status=bodleian_image_status(images),
                     sys_num=ktiv.get("sys_num"), iiif_manifest_url=ktiv.get("iiif_manifest_url"),
                     unresolved=None if pairs else unresolved_reason(images))


# ---------------------------------------------------------------------------
# Earlier reads, known-bad URLs, pending queues
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReadRef:
    """One earlier consensus read of an image."""

    vlm_model: str
    file: str
    doc_id: str
    n_lines: int


def read_files(read_dir: Path) -> List[Path]:
    """Consensus records files of earlier runs (``ai_reads_*.jsonl``; ``*.failures.jsonl`` are not reads).

    :param read_dir: The ``ai_reads`` directory.
    :type read_dir: Path
    :return: Sorted paths.
    :rtype: List[Path]
    """
    return sorted(p for p in read_dir.glob("ai_reads_*.jsonl") if not p.name.endswith(".failures.jsonl"))


def index_reads(rows_by_file: Iterable[Tuple[str, Iterable[Mapping[str, Any]]]]) -> Dict[str, List[ReadRef]]:
    """Earlier reads keyed by image URL (the URL is stable across index versions; ``image_index`` is not).

    :param rows_by_file: ``(file name, records)`` pairs; records carry ``image_url`` and ``ai_read``.
    :type rows_by_file: Iterable[Tuple[str, Iterable[Mapping[str, Any]]]]
    :return: ``{image_url: [ReadRef, ...]}``.
    :rtype: Dict[str, List[ReadRef]]
    """
    out: Dict[str, List[ReadRef]] = defaultdict(list)
    for name, rows in rows_by_file:
        for row in rows:
            read = row.get("ai_read")
            url = row.get("image_url")
            if not url or not isinstance(read, Mapping):
                continue
            out[url].append(ReadRef(str(read.get("vlm_model") or "unknown"), name, str(row.get("doc_id")),
                                    int(read.get("n_lines") or 0)))
    return dict(out)


def urls_in(rows: Iterable[Mapping[str, Any]]) -> Set[str]:
    """Image URLs named by job rows.

    :param rows: Job rows.
    :type rows: Iterable[Mapping[str, Any]]
    :return: Set of URLs.
    :rtype: Set[str]
    """
    return {str(r["image_url"]) for r in rows if r.get("image_url")}


def image_read_class(refs: Sequence[ReadRef]) -> str:
    """Read class of one image.

    :param refs: Earlier reads of its URL.
    :type refs: Sequence[ReadRef]
    :return: :data:`ALREADY_READ` or :data:`NEVER_READ`.
    :rtype: str
    """
    return ALREADY_READ if refs else NEVER_READ


def doc_read_class(image_classes: Sequence[str]) -> str:
    """Read class of a document from its images' classes.

    :param image_classes: One class per resolvable image.
    :type image_classes: Sequence[str]
    :return: :data:`UNRESOLVED` (no image), :data:`ALREADY_READ` (all read), :data:`PARTIALLY_READ` or
        :data:`NEVER_READ`.
    :rtype: str
    """
    if not image_classes:
        return UNRESOLVED
    n_read = sum(c == ALREADY_READ for c in image_classes)
    if n_read == len(image_classes):
        return ALREADY_READ
    return PARTIALLY_READ if n_read else NEVER_READ


def co_owners(url: str, doc_id: str, owners: Mapping[str, Sequence[str]]) -> List[str]:
    """Other merged records whose image list holds the same URL.

    :param url: Image URL.
    :type url: str
    :param doc_id: The record the row belongs to.
    :type doc_id: str
    :param owners: ``{image_url: [doc_id, ...]}`` over every merged record.
    :type owners: Mapping[str, Sequence[str]]
    :return: Sorted distinct other doc ids.
    :rtype: List[str]
    """
    return sorted({d for d in owners.get(url, ()) if d != doc_id})


def fragment_overlap(others: Iterable[str], n_images: Mapping[str, int], vector_ids: AbstractSet[str],
                     container_min: int = CONTAINER_MIN_IMAGES) -> Optional[str]:
    """Whether an image is also listed under a fragment-level record (fewer than ``container_min`` images).

    A box-level record that lists every image of its fragments does not count: the fragment is the unit to read.

    :param others: Other records listing the image (:func:`co_owners`).
    :type others: Iterable[str]
    :param n_images: Image count of every merged record.
    :type n_images: Mapping[str, int]
    :param vector_ids: Records that get a vector.
    :type vector_ids: AbstractSet[str]
    :param container_min: Image count from which a record is a container.
    :type container_min: int
    :return: ``vector`` (one such record gets a vector), ``vectorless`` (only vector-less ones) or ``None``.
    :rtype: Optional[str]
    """
    fragments = [d for d in others if n_images.get(d, 0) < container_min]
    if not fragments:
        return None
    return "vector" if any(d in vector_ids for d in fragments) else "vectorless"


# ---------------------------------------------------------------------------
# Job emission, ordering, pilot
# ---------------------------------------------------------------------------

def job_rows(candidate: Candidate, reads: Mapping[str, Sequence[ReadRef]],
             pending: Optional[Mapping[str, Sequence[str]]] = None,
             owners: Optional[Mapping[str, Sequence[str]]] = None,
             vector_ids: AbstractSet[str] = frozenset(),
             n_images: Optional[Mapping[str, int]] = None) -> List[Dict[str, Any]]:
    """Job rows (one per image) for a candidate, in the pipeline's format plus informational fields.

    :param candidate: Vector-less record.
    :type candidate: Candidate
    :param reads: Earlier reads keyed by URL (:func:`index_reads`).
    :type reads: Mapping[str, Sequence[ReadRef]]
    :param pending: ``{image_url: [job-list name, ...]}`` of routed lists not yet run.
    :type pending: Optional[Mapping[str, Sequence[str]]]
    :param owners: ``{image_url: [doc_id, ...]}`` over every merged record (for ``also_in``).
    :type owners: Optional[Mapping[str, Sequence[str]]]
    :param vector_ids: Records that get a vector (for ``also_in_fragment``).
    :type vector_ids: AbstractSet[str]
    :param n_images: Image count of every merged record (for ``also_in_fragment``).
    :type n_images: Optional[Mapping[str, int]]
    :return: Rows in ``image_index`` order (empty when no image resolves).
    :rtype: List[Dict[str, Any]]
    """
    rows: List[Dict[str, Any]] = []
    for index, (source, url) in enumerate(candidate.images):
        refs = reads.get(url) or ()
        row: Dict[str, Any] = {
            "doc_id": candidate.doc_id, "image_index": index, "image_url": url,
            "document_type": candidate.document_type, "institution": candidate.institution,
            "queue": candidate.tranche, "reread_class": image_read_class(refs), "pool": POOL_NAME,
            "vectorless_reason": candidate.reason, "image_source": source, "doc_n_images": len(candidate.images)}
        if refs:
            row["read_by"] = [{"vlm_model": r.vlm_model, "file": r.file, "n_lines": r.n_lines} for r in refs]
        elif pending and pending.get(url):
            row["pending_in"] = list(pending[url])
        others = co_owners(url, candidate.doc_id, owners) if owners else []
        if others:
            row["also_in"] = others[:MAX_CO_OWNERS]
            overlap = fragment_overlap(others, n_images or {}, vector_ids)
            if overlap:
                row["also_in_fragment"] = overlap
        rows.append(row)
    return rows


def sort_rows(rows: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """Order rows by tranche (:data:`TRANCHES`), then ``doc_id``, then ``image_index``.

    :param rows: Job rows.
    :type rows: Iterable[Mapping[str, Any]]
    :return: Sorted rows.
    :rtype: List[Mapping[str, Any]]
    """
    rank = {t: i for i, t in enumerate(TRANCHES)}
    return sorted(rows, key=lambda r: (rank[r["queue"]], r["doc_id"], r["image_index"]))


def sha1_key(doc_id: str) -> str:
    """Stable pseudo-random rank of a document.

    :param doc_id: Document id.
    :type doc_id: str
    :return: Hex SHA-1 of the UTF-8 id.
    :rtype: str
    """
    return hashlib.sha1(doc_id.encode("utf-8")).hexdigest()


def select_pilot(candidates: Sequence[Candidate], doc_classes: Mapping[str, str],
                 quotas: Mapping[str, int] = PILOT_QUOTAS,
                 blocked_urls: AbstractSet[str] = frozenset()) -> Dict[str, List[str]]:
    """Deterministic pilot: per tranche, the never-read documents with the lowest SHA-1 of ``doc_id``.

    :param candidates: Vector-less records.
    :type candidates: Sequence[Candidate]
    :param doc_classes: ``{doc_id: doc read class}``.
    :type doc_classes: Mapping[str, str]
    :param quotas: Documents per tranche.
    :type quotas: Mapping[str, int]
    :param blocked_urls: URLs known not to download (a document needs one image outside them).
    :type blocked_urls: AbstractSet[str]
    :return: ``{tranche: [doc_id, ...]}`` in SHA-1 order (shorter than the quota when the tranche runs out).
    :rtype: Dict[str, List[str]]
    """
    eligible: Dict[str, List[str]] = defaultdict(list)
    for c in candidates:
        if doc_classes.get(c.doc_id) == NEVER_READ and any(url not in blocked_urls for _, url in c.images):
            eligible[c.tranche].append(c.doc_id)
    return {t: sorted(eligible.get(t, []), key=sha1_key)[:n] for t, n in quotas.items()}


def compare_positions(images_by_doc: Mapping[str, Sequence[str]],
                      reference: Iterable[Tuple[str, int, str]]) -> Counter:
    """Compare an older index's ``(doc_id, image_index) -> image_url`` with the rebuilt image lists.

    :param images_by_doc: Rebuilt ``image_urls`` per document (documents outside it are not compared).
    :type images_by_doc: Mapping[str, Sequence[str]]
    :param reference: ``(doc_id, image_index, image_url)`` triples served by the older index.
    :type reference: Iterable[Tuple[str, int, str]]
    :return: Counter of ``same`` (same URL at the same position), ``moved`` (listed elsewhere), ``gone``.
    :rtype: Counter
    """
    out: Counter = Counter()
    for doc_id, index, url in reference:
        urls = images_by_doc.get(doc_id)
        if urls is None:
            continue
        if index < len(urls) and urls[index] == url:
            out["same"] += 1
        elif url in urls:
            out["moved"] += 1
        else:
            out["gone"] += 1
    return out


# ---------------------------------------------------------------------------
# Statistics and summary
# ---------------------------------------------------------------------------

def percentile(sorted_values: Sequence[int], q: float) -> int:
    """Nearest-rank percentile of an ascending sequence.

    :param sorted_values: Ascending values (non-empty).
    :type sorted_values: Sequence[int]
    :param q: Quantile in ``[0, 1]``.
    :type q: float
    :return: The value at rank ``ceil(q * n)``.
    :rtype: int
    """
    rank = max(1, math.ceil(q * len(sorted_values)))
    return sorted_values[min(rank, len(sorted_values)) - 1]


def image_stats(counts: Sequence[int]) -> Dict[str, Any]:
    """Images-per-document statistics.

    :param counts: Images per resolvable document.
    :type counts: Sequence[int]
    :return: ``docs``, ``images``, ``mean``, ``median``, ``p90``, ``p99``, ``max`` (zeros when empty).
    :rtype: Dict[str, Any]
    """
    if not counts:
        return {"docs": 0, "images": 0, "mean": 0.0, "median": 0, "p90": 0, "p99": 0, "max": 0}
    ordered = sorted(counts)
    median = statistics.median(ordered)
    return {"docs": len(ordered), "images": sum(ordered), "mean": round(statistics.mean(ordered), 2),
            "median": int(median) if float(median).is_integer() else median, "p90": percentile(ordered, 0.9),
            "p99": percentile(ordered, 0.99), "max": ordered[-1]}


def hours(n_images: int) -> str:
    """Pipeline wall time for ``n_images`` at :data:`SECONDS_PER_IMAGE`.

    :param n_images: Images.
    :type n_images: int
    :return: ``"a–b h (c–d days)"``.
    :rtype: str
    """
    lo, hi = (n_images * s / 3600 for s in SECONDS_PER_IMAGE)
    return f"{lo:,.0f}–{hi:,.0f} h ({lo / 24:,.1f}–{hi / 24:,.1f} days)"


def md_table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Render a Markdown table.

    :param header: Column names.
    :type header: Sequence[str]
    :param rows: Rows of cells.
    :type rows: Iterable[Sequence[Any]]
    :return: Markdown text (ends with a newline).
    :rtype: str
    """
    def cell(v: Any) -> str:
        return f"{v:,}" if isinstance(v, int) and not isinstance(v, bool) else str(v)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def tranche_table(candidates: Sequence[Candidate], doc_classes: Mapping[str, str]) -> str:
    """Documents per tranche x reason x read class, with resolvable images.

    :param candidates: Vector-less records.
    :type candidates: Sequence[Candidate]
    :param doc_classes: ``{doc_id: doc read class}``.
    :type doc_classes: Mapping[str, str]
    :return: Markdown table.
    :rtype: str
    """
    cells: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for c in candidates:
        key = (c.tranche, c.reason)
        cells[key][doc_classes[c.doc_id]] += 1
        cells[key]["images"] += len(c.images)
        cells[key]["docs"] += 1
    rows: List[List[Any]] = []
    for tranche in TRANCHES + ("total",):
        for reason in REASON_LABELS + ("all",):
            got: Counter = Counter()
            for (t, r), counter in cells.items():
                if tranche in (t, "total") and reason in (r, "all"):
                    got.update(counter)
            if tranche == "total" and reason != "all":
                continue
            rows.append([f"**{tranche}**" if reason == "all" else tranche, reason, got["docs"]]
                        + [got[k] for k in DOC_CLASSES] + [got["images"]])
    return md_table(["tranche", "reason", "docs"] + list(DOC_CLASSES) + ["images (resolvable)"], rows)


def render_summary(s: Mapping[str, Any]) -> str:
    """The Markdown summary.

    :param s: Statistics collected by :func:`build` (keys documented there).
    :type s: Mapping[str, Any]
    :return: Markdown text.
    :rtype: str
    """
    st_all = s["image_stats"]
    pilot_mix = ", ".join(f"{t.split('_')[0]} {n}" for t, n in s["pilot_by_tranche"].items())
    out: List[str] = []
    out.append("# Vector-less records: consensus job pool (v9)\n")
    out.append(f"Built {s['built_at']} by `src/datasets/consensus/build_vectorless_jobs.py` from "
               f"`{s['merged_rel']}` ({s['merged_records']:,} records, file time {s['merged_mtime']}). "
               f"Eligibility = the v9 indexer's rule (`text_representation.create_text_representation(doc, "
               f"\"{TEXT_VERSION}\")` then `semantic_text.is_semantic_eligible`): **{s['vectorless']:,} records get "
               f"no vector** (brief: {EXPECTED['vectorless']:,}). Nothing was launched, fetched or uploaded.\n")
    out.append("## Files\n")
    out.append(md_table(["file", "rows", "what"], [
        ["`jobs_vectorless_pool.jsonl`", s["pool_rows"], "one row per image of every vector-less record with a "
         "resolvable image, ordered by tranche, doc_id, image_index"],
        ["`jobs_vectorless_pilot200.jsonl`", s["pilot_rows"], f"{s['pilot_docs']} never-read documents "
         f"({pilot_mix}), all their images"],
        ["`iiif_only_docs.txt`", s["iiif_docs"], "vector-less records whose KTIV images exist only as an IIIF "
         "manifest (no local zips, no bucket URLs): for a later fetch"],
    ]))
    out.append("\nRow format = the routed job lists (`doc_id`, `image_index`, `image_url`, `document_type`, "
               "`institution`, `queue`, `reread_class`) plus `pool: \"vectorless_v9\"`, `vectorless_reason`, "
               "`image_source` (ktiv / bodleian / legacy), `doc_n_images`; `read_by` (model, file, n_lines) on "
               "already-read images; `pending_in` on never-read images a routed list still holds; `also_in` / "
               "`also_in_fragment` on images other records list too. "
               "`two_reader_lines.build_jobs` keeps only `doc_id` / `image_index` / `image_url` from each `--ids` "
               "row, so the extra keys are inert and no sidecar file is needed.\n")

    pc = s["pilot_containers"]
    out.append(f"\nPilot: the brief's rule (per tranche the never-read documents with the lowest SHA-1 of doc_id, at "
               f"least one image not on a HEAD-404 list), {s['pilot_rows']:,} images, {hours(s['pilot_rows'])}. "
               + (f"{len(pc)} of its documents have ≥ {CONTAINER_MIN_IMAGES} images and hold "
                  f"{sum(n for _, n in pc):,} of them (" + ", ".join(f"`{d}` {n}" for d, n in pc) + "); without "
                  f"them the pilot is {s['pilot_rows'] - sum(n for _, n in pc):,} images "
                  f"({hours(s['pilot_rows'] - sum(n for _, n in pc))}).\n" if pc else "\n"))
    out.append("## Counts per tranche x reason x read class (documents)\n")
    out.append(f"Tranches: `{V1}` = record has a PGP source; `{V2}` = KTIV source, no PGP; `{V3}` = neither "
               "(legacy image store, plus Bodleian images on a few; no PGP or KTIV catalogue text). "
               "Read class per document: `already_read` = every resolvable image has an earlier consensus read "
               "(matched on the exact image URL), `partially_read` = some, `never_read` = none, `unresolved` = no "
               "image the index serves. Per-image `reread_class` in the rows is `already_read` / `never_read`.\n")
    out.append(tranche_table(s["candidates"], s["doc_classes"]))
    out.append(f"\nImages: **{s['pool_rows']:,}** rows, of which {s['images_by_class'].get(NEVER_READ, 0):,} "
               f"never read and {s['images_by_class'].get(ALREADY_READ, 0):,} already read. "
               f"Pipeline time for the never-read images at 55–90 s/image: "
               f"{hours(s['images_by_class'].get(NEVER_READ, 0))}.\n")

    out.append("## Images per document\n")
    rows = []
    for name, st in [("all resolvable", st_all)] + [(t, s["image_stats_by_tranche"][t]) for t in TRANCHES]:
        rows.append([name, st["docs"], st["images"], st["mean"], st["median"], st["p90"], st["p99"], st["max"]])
    out.append(md_table(["documents", "docs", "images", "mean", "median", "p90", "p99", "max"], rows))
    big = s["containers"]
    out.append(f"\n**Container records.** {len(big):,} records have ≥ {CONTAINER_MIN_IMAGES} images and hold "
               f"{sum(n for _, n, _ in big):,} of the {s['pool_rows']:,} images "
               f"({100 * sum(n for _, n, _ in big) / max(1, s['pool_rows']):.0f} %): box- or volume-level records "
               f"(e.g. a whole T-S AS box) rather than one fragment. Without them the pool is "
               f"{s['pool_rows'] - sum(n for _, n, _ in big):,} images "
               f"({hours(s['pool_rows'] - sum(n for _, n, _ in big))} for everything). Largest: "
               + ", ".join(f"`{d}` {n} ({t.split('_')[0]})" for d, n, t in big[:10])
               + f". Filter with `doc_n_images < {CONTAINER_MIN_IMAGES}`.\n")

    out.append("## Image source mix\n")
    out.append(md_table(["image source", "images (all)"] + [t.split("_")[0] for t in TRANCHES],
                        [[src, s["images_by_source"].get(src, 0)]
                         + [s["images_by_source_tranche"].get((src, t), 0) for t in TRANCHES]
                         for src in SOURCE_ORDER]))
    out.append("\nOne availability class per record (the brief's precedence: legacy store > KTIV local > "
               "KTIV IIIF-only > Bodleian > none):\n\n")
    out.append(md_table(["class", "records", "brief", "resolvable", "note"], [
        [k, s["availability"].get(k, 0), EXPECTED[k], s["availability_resolvable"].get(k, 0), note]
        for k, note in [("legacy_store", "legacy-store file names / bucket URLs"),
                        ("ktiv_local", "KTIV images on the bucket, or local zips not yet populated"),
                        ("ktiv_iiif_only", "manifest only"),
                        ("bodleian", "Bodleian block (bucket URLs when populated)"),
                        ("none", "no image pointer at all")]]))
    combos = ", ".join(f"{k} {v:,}" for k, v in s["source_combos"].most_common())
    out.append(f"\nResolvable records by the set of sources their images come from: {combos}.\n")
    out.append(f"\nURL hygiene: {s['space_urls']:,} pool URLs hold a literal space (macOS `name 2.jpg` copies; "
               f"`two_reader_lines.download` percent-encodes them); {s['known_404']:,} pool URLs appear in the "
               "earlier HEAD-404 lists (`jobs_documentary_v5_unreachable_0916.jsonl`, "
               "`jobs_documentary_v6_unreachable_0922.jsonl`); duplicate URLs within a record: "
               f"{s['dup_url_docs']:,}.\n")

    out.append("## Images that other records list too\n")
    cc, fr = s["covered_containers"], s["frag_rows"]
    out.append(f"{s['shared_rows']:,} pool images are also listed under at least one other merged record (`also_in`, "
               f"up to {MAX_CO_OWNERS} ids). `also_in_fragment` says whether one of those is a fragment-level record "
               f"(< {CONTAINER_MIN_IMAGES} images): `vector` on {fr.get('vector', 0):,} images (that record gets a "
               f"vector; {s['frag_vector_never']:,} never read), `vectorless` on {fr.get('vectorless', 0):,} (the "
               "other record is in this pool too, so the image is listed twice here); the remaining "
               f"{s['container_only_rows']:,} shared images are fragment images whose other owner is a box-level "
               "record" + (f" (e.g. `{s['box_example'][0]}` inside `{s['box_example'][1]}`)" if s["box_example"] else "")
               + ": the fragment is the unit to read, so those carry no `also_in_fragment`.\n")
    out.append(f"\n{s['covered_docs']:,} records have every image listed under a fragment-level record "
               f"({s['covered_images']:,} images; "
               + ", ".join(f"{t.split('_')[0]} {n:,}" for t, n in s["covered_by_tranche"].most_common())
               + "), chiefly box-level records repeating their fragments' images: "
               + ", ".join(f"`{d}` {n}" for d, n in cc[:6])
               + f" ({len(cc)} such containers in all), plus duplicate records of one fragment. "
               f"{s['containers_unique']:,} containers hold only images no other record lists (volumes and "
               "codices: " + ", ".join(f"`{d}` {n}" for d, n in s["unique_container_examples"]) + "). "
               "Never-read images left after dropping "
               f"`also_in_fragment: vector` rows and records with ≥ {CONTAINER_MIN_IMAGES} images: "
               f"**{s['never_lean']:,}** ({hours(s['never_lean'])}). The pilot keeps the brief's rule; "
               f"{len(s['pilot_covered'])} of its documents have every image listed under another fragment-level "
               "record" + (": " + ", ".join(f"`{d}`" for d in s["pilot_covered"]) if s["pilot_covered"] else "")
               + ".\n")

    out.append("## IIIF-only KTIV images\n")
    out.append(f"{s['iiif_docs']:,} vector-less records have a KTIV IIIF manifest but neither local zips nor bucket "
               f"URLs; {s['iiif_no_other']:,} of them have no other image (unresolvable until fetched) and "
               f"{s['iiif_docs'] - s['iiif_no_other']:,} are resolvable through legacy-store or Bodleian images "
               "(their KTIV images would be added in front once fetched, which shifts `image_index`). Listed in "
               "`iiif_only_docs.txt` (tab-separated: doc_id, tranche, reason, sys_num, manifest URL, other "
               "resolvable images). Not fetched.\n")

    out.append("## Records that could not be resolved\n")
    out.append(md_table(["reason", "records"] + [t.split("_")[0] for t in TRANCHES],
                        [[r, sum(s["unresolved"][(r, t)] for t in TRANCHES)] + [s["unresolved"][(r, t)]
                                                                                 for t in TRANCHES]
                         for r in ("no_images", "ktiv_iiif_only", "ktiv_zip_not_populated",
                                   "bodleian_not_populated")]))
    out.append("\n`ktiv_zip_not_populated` (local KTIV zips, but the merge lists no images or URLs, "
               f"`populated: false`): {', '.join(f'`{d}`' for d in s['zip_unpopulated_ids']) or 'none'}. "
               "`bodleian_not_populated` (Bodleian block with no images): "
               f"{', '.join(f'`{d}`' for d in s['bodleian_unpopulated_ids']) or 'none'}. "
               f"Records the indexer would skip for a blank shelf mark: {s['blank_shelfmark']:,}.\n")

    out.append("## Already-read images\n")
    by_model = ", ".join(f"`{m}` {n:,}" for m, n in s["read_models"].most_common()) or "none"
    by_file = ", ".join(f"`{f}` {n:,}" for f, n in s["read_files_hit"].most_common()) or "none"
    out.append(f"Reads indexed: {s['reads_indexed']:,} records from {len(s['read_files'])} files "
               f"(`ai_reads_*.jsonl` without the failure logs) covering {s['read_urls']:,} distinct image URLs. "
               f"Pool images with a read, by model (an image can have several): {by_model}. By file: {by_file}. "
               f"Already-read images whose every read has 0 lines: {s['empty_reads']:,}. "
               f"Reads made under another doc_id (same image, different record): {s['read_doc_mismatch']:,}.\n")
    pend = ", ".join(f"`{k}` {v:,}" for k, v in s["pending"].most_common()) or "none"
    out.append(f"\nNever-read pool images that a routed list not yet (fully) run also holds (`pending_in`): {pend}. "
               "The 10-05 relaunch runs the routed literary list after the paused documentary pass, so those "
               "images will be read there unless the lists are deduplicated.\n")

    out.append("## How the rows were resolved (and how that differs from the ES-based builders)\n")
    a50, lit = s["v8_check_reads"], s["v8_check_jobs"]
    out.append("`build_ai_read_jobs.py` (and through it the routed lists) takes `doc_id` = ES `_id`, `image_urls`, "
               "`document_type` and `institution` from the `_source` of the newest merged index. Elasticsearch is "
               "not reachable from this session (HTTP 403), so this builder runs the v9 indexer's own code on the "
               "merged export: `iter_merged_documents` (zipped by position with the raw records, `canonical_id` "
               "checked on every pair) -> `GenizahDocument.from_merged_format` -> `image_urls` (KTIV bucket URLs, "
               "Bodleian bucket URLs, legacy-store names under `.../cairo-genizah-es-json/images/`, preferred "
               "source first), `es_document_id(doc)` for `doc_id`, and `to_elasticsearch_document()` for "
               "`document_type` / `institution`; the per-source attribution was asserted equal to `doc.image_urls` "
               "for every record. The rows therefore match `genizah_merged_v9` exactly only if v9 is built from "
               f"this merged file ({s['merged_records']:,} records, {s['merged_mtime']}).\n")
    out.append(f"\nDrift against the live v8 index, measured on v8-served triples of pool documents: blend records "
               f"(`{V8_REFERENCE_READS.name}`, v8-stamped) same {a50.get('same', 0):,} / moved "
               f"{a50.get('moved', 0):,} / gone {a50.get('gone', 0):,}; v8 literary job list "
               f"(`{V8_REFERENCE_JOBS.name}`) same {lit.get('same', 0):,} / moved {lit.get('moved', 0):,} / gone "
               f"{lit.get('gone', 0):,}. Over all records: blend {s['v8_all_reads'].get('same', 0):,} / "
               f"{s['v8_all_reads'].get('moved', 0):,} / {s['v8_all_reads'].get('gone', 0):,}, literary "
               f"{s['v8_all_jobs'].get('same', 0):,} / {s['v8_all_jobs'].get('moved', 0):,} / "
               f"{s['v8_all_jobs'].get('gone', 0):,}. `build_jobs` trusts `image_index` whenever it is inside the "
               "served list (it reads `urls[image_index]`, not the row's URL), so against v8 a moved row reads a "
               "different image of the same record and a record new since v8 is dropped. Against v9 built from "
               "this file both are exact.\n")

    out.append("## Launch commands (not run)\n")
    out.append("Flags copied from `logs/routed/pipeline_relaunch_1005.sh` (`run_pass`): `--source-index`, "
               "`--vlm-model`/`--vlm-revision`, `--out ai_reads_<KEY>...`, `--work-dir images_routed_<NAME>`, "
               f"`--htr-cache-key {HTR_KEY}`, `--min-free-gb 10`; routing as on 10-05 (documentary -> the blend "
               f"`{VLM_DOCUMENTARY[0]}`, literary -> `{VLM_LITERARY[0]}`), so V1 goes to the blend and V2 + V3 to "
               "v21b. Preconditions: `genizah_merged_v9` exists and was built from this merged file (else use "
               "`--source-index genizah_merged_v8` and accept the drift above); the paused documentary pass "
               "(pid 10575, writing the blend's records file) has finished or been stopped — one LM Studio consumer "
               "at a time; Kraken :8002 healthy; the VLM staged as `stage()` does; free disk ≥ 20 GiB (the "
               "relaunch gate; the pipeline itself stops under 10 GiB). Split the files by tranche first "
               "(`grep -F` on the queue value):\n")
    pool, pilot = "$P/jobs_vectorless_pool.jsonl", "$P/jobs_vectorless_pilot200.jsonl"
    cmd = ("nohup .venv/bin/python -m src.datasets.consensus.two_reader_lines --ids {ids} "
           "--source-index genizah_merged_v9 \\\n    --vlm-model {key} --vlm-revision {rev} --out $A/ai_reads_{key}"
           "{suffix}.jsonl \\\n    --work-dir $A/images_routed_{name} --htr-cache-key " + HTR_KEY
           + " --min-free-gb 10 >> logs/two_reader_routed_{name}.log 2>&1 &")
    block = ["```zsh", "cd /Users/isaac/Documents/GitHub/historical-document-analysis",
             "P=logs/next_round/consensus_pool; A=src/datasets/raw_data/cairo_genizah/ai_reads", "",
             "# pilot (200 documents): V1 with the blend, then V2+V3 with v21b; separate --out files keep the pilot "
             "scoreable on its own",
             f"grep -F '\"queue\": \"{V1}\"' {pilot} > $P/jobs_vectorless_pilot200_V1.jsonl",
             f"grep -vF '\"queue\": \"{V1}\"' {pilot} > $P/jobs_vectorless_pilot200_V23.jsonl",
             cmd.format(ids="$P/jobs_vectorless_pilot200_V1.jsonl", key=VLM_DOCUMENTARY[0], rev=VLM_DOCUMENTARY[1],
                        suffix="_vectorless_pilot200", name="vectorless_pilot"),
             "# ...when it has printed 'done', and with the blend unloaded:",
             cmd.format(ids="$P/jobs_vectorless_pilot200_V23.jsonl", key=VLM_LITERARY[0], rev=VLM_LITERARY[1],
                        suffix="_vectorless_pilot200", name="vectorless_pilot"),
             "",
             "# full pool (never-read images only; drop the container records with a doc_n_images filter if wanted)",
             f"grep -F '\"reread_class\": \"never_read\"' {pool} | grep -F '\"queue\": \"{V1}\"' "
             "> $P/jobs_vectorless_pool_V1.jsonl",
             f"grep -F '\"reread_class\": \"never_read\"' {pool} | grep -vF '\"queue\": \"{V1}\"' "
             "> $P/jobs_vectorless_pool_V23.jsonl",
             cmd.format(ids="$P/jobs_vectorless_pool_V1.jsonl", key=VLM_DOCUMENTARY[0], rev=VLM_DOCUMENTARY[1],
                        suffix="", name="vectorless"),
             cmd.format(ids="$P/jobs_vectorless_pool_V23.jsonl", key=VLM_LITERARY[0], rev=VLM_LITERARY[1],
                        suffix="", name="vectorless"),
             "```"]
    out.append("\n".join(block) + "\n")
    out.append("\nThe pool runs write into the per-model records files (`ai_reads_<KEY>.jsonl`) as the routed passes "
               "do, so the resume skip and the site loader (`logs/routed/loader_loop.sh <out> <pidfile> <log>`) "
               "work unchanged; the resume key is `(doc_id, image_index)`, so a v8-stamped record of a pool "
               "document whose list has since moved could hide one pool image (only possible for already- or "
               "partially-read documents). Single-model alternative: pass the unsplit file to one run. "
               "Run one pass at a time (one LM Studio consumer).\n")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def v8_reference(jobs_path: Path, reads_path: Path) -> Tuple[List[Tuple[str, int, str]], List[Tuple[str, int, str]]]:
    """v8-served ``(doc_id, image_index, image_url)`` triples from the literary job list and the blend's records.

    :param jobs_path: ``jobs_literary_v8_0928.jsonl`` (built from v8's ``_source``).
    :type jobs_path: Path
    :param reads_path: The blend's records file (rows stamped ``genizah_merged_v8`` are used).
    :type reads_path: Path
    :return: ``(from records, from job list)``.
    :rtype: Tuple[List[Tuple[str, int, str]], List[Tuple[str, int, str]]]
    """
    reads = [(r["doc_id"], int(r["image_index"]), r["image_url"]) for r in iter_jsonl(reads_path)
             if r.get("source_index") == "genizah_merged_v8" and r.get("image_url")] if reads_path.exists() else []
    jobs = [(r["doc_id"], int(r["image_index"]), r["image_url"]) for r in iter_jsonl(jobs_path)
            if r.get("image_url")] if jobs_path.exists() else []
    return reads, jobs


def eligibility_pass(merged: Path, version: str,
                     reference_docs: AbstractSet[str]) -> Tuple[List[Candidate], Dict[str, Any]]:
    """Run the v9 eligibility rule over the merged export and collect the vector-less records.

    :param merged: ``merged_shelfmarks.jsonl``.
    :type merged: Path
    :param version: Text representation version (``semantic_v3``).
    :type version: str
    :param reference_docs: Document ids whose rebuilt ``image_urls`` are kept for the v8 drift check.
    :type reference_docs: AbstractSet[str]
    :return: Candidates in file order, and pass statistics (``records``, ``blank_shelfmark``, ``image_urls_by_doc``,
        ``owners`` = every record's doc id per image URL, ``vector_ids`` = records that get a vector,
        ``n_images`` = image count per record).
    :rtype: Tuple[List[Candidate], Dict[str, Any]]
    :raises RuntimeError: When the raw records and the documents fall out of step, the per-source attribution
        differs from the index's ``image_urls``, or the two eligibility functions disagree.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        from src.datasets.indexing.elastic_index_genizah import es_document_id
        from src.datasets.indexing.index_merged_genizah import iter_merged_documents
        from src.embeddings import semantic_text as st
        from src.embeddings import text_representation as tr
    candidates: List[Candidate] = []
    urls_by_doc: Dict[str, List[str]] = {}
    owners: Dict[str, List[str]] = defaultdict(list)
    vector_ids: Set[str] = set()
    n_images: Dict[str, int] = {}
    n_records, blank_shelfmark = 0, 0
    logging.disable(logging.WARNING)
    try:
        pairs = zip(iter_merged_records(merged), iter_merged_documents(str(merged)), strict=True)
        for n_records, (record, doc) in enumerate(pairs, 1):
            if doc.doc_id != record.get("canonical_id"):
                raise RuntimeError(f"line {n_records}: document {doc.doc_id!r} vs record {record.get('canonical_id')!r}")
            if doc.doc_id in reference_docs:
                urls_by_doc[doc.doc_id] = list(doc.image_urls)
            n_images[doc.doc_id] = len(doc.image_urls)
            for url in doc.image_urls:
                owners[url].append(doc.doc_id)
            text = tr.create_text_representation(doc, version=version)
            reason = vectorless_reason(st.semantic_ineligible_reason(text))
            if (reason is None) != st.is_semantic_eligible(text):
                raise RuntimeError(f"{doc.doc_id}: eligibility functions disagree")
            if reason is None:
                vector_ids.add(doc.doc_id)
                continue
            if not (doc.shelf_mark or "").strip():
                blank_shelfmark += 1
            pairs_by_source = ordered_image_urls(record.get("images") or {})
            if [u for _, u in pairs_by_source] != list(doc.image_urls):
                raise RuntimeError(f"{doc.doc_id}: per-source image attribution differs from the index image_urls")
            es_doc = doc.to_elasticsearch_document()   # after the text, as the indexer does (it maps the shelf mark)
            candidates.append(make_candidate(record, es_document_id(doc), reason, es_doc.get("document_type"),
                                             es_doc.get("institution")))
    finally:
        logging.disable(logging.NOTSET)
    return candidates, {"records": n_records, "blank_shelfmark": blank_shelfmark, "image_urls_by_doc": urls_by_doc,
                        "owners": dict(owners), "vector_ids": vector_ids, "n_images": n_images}


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Write rows as JSONL (UTF-8, one object per line).

    :param path: Destination.
    :type path: Path
    :param rows: Rows.
    :type rows: Iterable[Mapping[str, Any]]
    :return: Rows written.
    :rtype: int
    """
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def iiif_lines(candidates: Sequence[Candidate]) -> List[str]:
    """Tab-separated lines for ``iiif_only_docs.txt`` (with a ``#`` header).

    :param candidates: Vector-less records.
    :type candidates: Sequence[Candidate]
    :return: Lines, header first, documents in tranche then doc_id order.
    :rtype: List[str]
    """
    rank = {t: i for i, t in enumerate(TRANCHES)}
    chosen = sorted((c for c in candidates if c.ktiv_status == "iiif_only"), key=lambda c: (rank[c.tranche], c.doc_id))
    lines = ["# doc_id\ttranche\tvectorless_reason\tsys_num\tiiif_manifest_url\tother_resolvable_images"]
    lines += [f"{c.doc_id}\t{c.tranche}\t{c.reason}\t{c.sys_num or ''}\t{c.iiif_manifest_url or ''}\t{len(c.images)}"
              for c in chosen]
    return lines


def build(merged: Path, out_dir: Path, version: str = TEXT_VERSION) -> Dict[str, Any]:
    """Build every output and return the tally.

    :param merged: ``merged_shelfmarks.jsonl``.
    :type merged: Path
    :param out_dir: Output directory.
    :type out_dir: Path
    :param version: Text representation version.
    :type version: str
    :return: Compact tally (also printed by :func:`main`).
    :rtype: Dict[str, Any]
    """
    files = read_files(AI_READS)
    reads = index_reads((p.name, iter_jsonl(p)) for p in files)
    blocked: Set[str] = set()
    for path in UNREACHABLE_LISTS:
        if path.exists():
            blocked |= urls_in(iter_jsonl(path))
    pending: Dict[str, List[str]] = defaultdict(list)
    for path in PENDING_LISTS:
        if path.exists():
            for url in urls_in(iter_jsonl(path)):
                pending[url].append(path.stem)
    ref_reads, ref_jobs = v8_reference(V8_REFERENCE_JOBS, V8_REFERENCE_READS)
    reference_docs = {d for d, _, _ in ref_reads} | {d for d, _, _ in ref_jobs}

    candidates, pass_stats = eligibility_pass(merged, version, reference_docs)

    rows: List[Dict[str, Any]] = []
    doc_classes: Dict[str, str] = {}
    for c in candidates:
        doc_rows = job_rows(c, reads, pending, pass_stats["owners"], pass_stats["vector_ids"], pass_stats["n_images"])
        doc_classes[c.doc_id] = doc_read_class([r["reread_class"] for r in doc_rows])
        rows.extend(doc_rows)
    pool = sort_rows(rows)
    pilot_ids = select_pilot(candidates, doc_classes, PILOT_QUOTAS, blocked)
    wanted = {d for ids in pilot_ids.values() for d in ids}
    pilot = [r for r in pool if r["doc_id"] in wanted]

    out_dir.mkdir(parents=True, exist_ok=True)
    n_pool = write_jsonl(out_dir / "jobs_vectorless_pool.jsonl", pool)
    n_pilot = write_jsonl(out_dir / "jobs_vectorless_pilot200.jsonl", pilot)
    iiif = iiif_lines(candidates)
    (out_dir / "iiif_only_docs.txt").write_text("\n".join(iiif) + "\n", encoding="utf-8")

    pool_doc_ids = {c.doc_id for c in candidates}
    vectorless_urls = {d: u for d, u in pass_stats["image_urls_by_doc"].items() if d in pool_doc_ids}
    read_models: Counter = Counter()
    read_files_hit: Counter = Counter()
    empty_reads = mismatch = 0
    for r in pool:
        if r.get("read_by"):
            read_models.update({x["vlm_model"] for x in r["read_by"]})
            read_files_hit.update({x["file"] for x in r["read_by"]})
            empty_reads += all(x["n_lines"] == 0 for x in r["read_by"])
            mismatch += any(reads_doc != r["doc_id"] for reads_doc in
                            (ref.doc_id for ref in reads.get(r["image_url"], ())))
    rows_by_doc: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for r in pool:
        rows_by_doc[r["doc_id"]].append(r)
    covered = {d for d, rs in rows_by_doc.items() if all(r.get("also_in_fragment") for r in rs)}
    by_tranche_counts: Dict[str, List[int]] = defaultdict(list)
    for c in candidates:
        if c.images:
            by_tranche_counts[c.tranche].append(len(c.images))
    stats: Dict[str, Any] = {
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "merged_rel": os.path.relpath(merged, REPO),
        "merged_records": pass_stats["records"],
        "merged_mtime": datetime.fromtimestamp(merged.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        "vectorless": len(candidates),
        "candidates": candidates, "doc_classes": doc_classes,
        "pool_rows": n_pool, "pilot_rows": n_pilot, "pilot_docs": len(wanted),
        "pilot_by_tranche": {t: len(v) for t, v in pilot_ids.items()},
        "iiif_docs": len(iiif) - 1,
        "iiif_no_other": sum(1 for c in candidates if c.ktiv_status == "iiif_only" and not c.images),
        "images_by_class": Counter(r["reread_class"] for r in pool),
        "image_stats": image_stats([len(c.images) for c in candidates if c.images]),
        "image_stats_by_tranche": {t: image_stats(by_tranche_counts.get(t, [])) for t in TRANCHES},
        "containers": sorted(((c.doc_id, len(c.images), c.tranche) for c in candidates
                              if len(c.images) >= CONTAINER_MIN_IMAGES), key=lambda x: (-x[1], x[0])),
        "images_by_source": Counter(r["image_source"] for r in pool),
        "images_by_source_tranche": Counter((r["image_source"], r["queue"]) for r in pool),
        "availability": Counter(c.availability for c in candidates),
        "availability_resolvable": Counter(c.availability for c in candidates if c.images),
        "source_combos": Counter("+".join(sorted({s for s, _ in c.images})) for c in candidates if c.images),
        "space_urls": sum(" " in r["image_url"] for r in pool),
        "known_404": sum(r["image_url"] in blocked for r in pool),
        "dup_url_docs": sum(1 for c in candidates if len({u for _, u in c.images}) != len(c.images)),
        "unresolved": Counter((c.unresolved, c.tranche) for c in candidates if c.unresolved),
        "zip_unpopulated_ids": sorted(c.doc_id for c in candidates if c.unresolved == "ktiv_zip_not_populated"),
        "bodleian_unpopulated_ids": sorted(c.doc_id for c in candidates if c.unresolved == "bodleian_not_populated"),
        "blank_shelfmark": pass_stats["blank_shelfmark"],
        "reads_indexed": sum(len(v) for v in reads.values()), "read_files": [p.name for p in files],
        "read_urls": len(reads), "read_models": read_models, "read_files_hit": read_files_hit,
        "empty_reads": empty_reads, "read_doc_mismatch": mismatch,
        "pending": Counter(name for r in pool for name in r.get("pending_in", ())),
        "shared_rows": sum(1 for r in pool if r.get("also_in")),
        "frag_rows": Counter(r["also_in_fragment"] for r in pool if r.get("also_in_fragment")),
        "frag_vector_never": sum(1 for r in pool if r.get("also_in_fragment") == "vector"
                                 and r["reread_class"] == NEVER_READ),
        "container_only_rows": sum(1 for r in pool if r.get("also_in") and not r.get("also_in_fragment")),
        "covered_docs": len(covered), "covered_images": sum(len(rows_by_doc[d]) for d in covered),
        "containers_unique": sum(1 for d, rs in rows_by_doc.items()
                                 if len(rs) >= CONTAINER_MIN_IMAGES and not any(r.get("also_in") for r in rs)),
        "unique_container_examples": sorted(((d, len(rs)) for d, rs in rows_by_doc.items()
                                             if len(rs) >= CONTAINER_MIN_IMAGES
                                             and not any(r.get("also_in") for r in rs)), key=lambda x: (-x[1], x[0]))[:4],
        "box_example": next(((r["doc_id"], r["also_in"][0]) for r in pool
                             if r.get("also_in") and not r.get("also_in_fragment")
                             and r["doc_n_images"] < CONTAINER_MIN_IMAGES), None),
        "covered_by_tranche": Counter(rows_by_doc[d][0]["queue"] for d in covered),
        "covered_containers": sorted(((d, len(rows_by_doc[d])) for d in covered
                                      if len(rows_by_doc[d]) >= CONTAINER_MIN_IMAGES), key=lambda x: (-x[1], x[0])),
        "pilot_covered": sorted(d for d in wanted if d in covered),
        "pilot_containers": sorted(((d, len(rows_by_doc[d])) for d in wanted
                                    if len(rows_by_doc[d]) >= CONTAINER_MIN_IMAGES), key=lambda x: (-x[1], x[0])),
        "never_lean": sum(1 for r in pool if r["reread_class"] == NEVER_READ and r.get("also_in_fragment") != "vector"
                          and r["doc_n_images"] < CONTAINER_MIN_IMAGES),
        "v8_check_reads": compare_positions(vectorless_urls, ref_reads),
        "v8_check_jobs": compare_positions(vectorless_urls, ref_jobs),
        "v8_all_reads": compare_positions(pass_stats["image_urls_by_doc"], ref_reads),
        "v8_all_jobs": compare_positions(pass_stats["image_urls_by_doc"], ref_jobs),
    }
    (out_dir / "vectorless_pool_summary.md").write_text(render_summary(stats), encoding="utf-8")
    doc_tally = Counter((c.tranche, doc_classes[c.doc_id]) for c in candidates)
    return {"records": stats["merged_records"], "vectorless": len(candidates), "pool_rows": n_pool,
            "pilot_docs": len(wanted), "pilot_rows": n_pilot, "iiif_only_docs": stats["iiif_docs"],
            "docs": {f"{t}|{k}": v for (t, k), v in sorted(doc_tally.items())},
            "images_by_class": dict(stats["images_by_class"])}


def main(argv: Optional[List[str]] = None) -> None:
    """CLI.

    :param argv: Arguments (default ``sys.argv[1:]``).
    :type argv: Optional[List[str]]
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", type=Path, default=MERGED)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--text-version", default=TEXT_VERSION)
    a = ap.parse_args(argv)
    print(json.dumps(build(a.merged, a.out_dir, a.text_version), ensure_ascii=False))


if __name__ == "__main__":
    main()
