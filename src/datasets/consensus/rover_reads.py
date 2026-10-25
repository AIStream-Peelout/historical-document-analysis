# File name: rover_reads.py
# Date: 10/8/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Character-level ROVER over consensus-pipeline page reads, written back in the pipeline's record schema.

``rover_consensus.py`` measured offline (PGP-131, religious-140) that voting the characters of several readers beats
any single reader; the best configuration was char level, unweighted, unfiltered, with Kraken's text as one more
char-level voter. This module applies that vote to pipeline reads: ``two_reader_lines`` records (one per page and
VLM checkpoint) are grouped by page, each page's VLM reads plus Kraken's lines are aligned and voted with
``rover_consensus``'s own functions (``similarity_matrix`` / ``choose_backbone`` / ``merge_order`` /
``build_network`` / ``vote_slot``), and one record per page is written in the pipeline's schema, so
``describe_from_reads.load_document_reads`` reads it unchanged.

Inputs, and the design choices the record schema forces:

* **Pages.** A record is a page by its image bytes: ``image_sha256``, else the hash another record gives the same
  URL, else ``image_url``, else ``doc_id#image_index`` (the identity ``describe_from_reads`` groups by). With a job
  list (``doc_id``, ``image_index``, ``image_url`` per page) only job pages are kept, matched by exact URL as
  ``describe_from_reads`` matches its image map. One read per (page, reader): parsed first, then the newest.
* **Kraken.** A record keeps only the fragments assigned to its own lines, as one joined ``htr_text`` plus their
  boxes per VLM line; unassigned fragments and the individual fragment texts are not in it. The full fragment list
  (text and box, in Kraken's order) is in the pipeline's raw cache (``ai_reads/raw/<doc>__<image>__<vlm>.json``,
  ``frags`` / ``frags_by_htr[key]`` under the record's ``htr_cache_key``), which is used when its URL, VLM and hash
  match the record. Without it each record line's ``htr_text`` with the union of its ``htr_fragments`` boxes stands
  in as one pseudo-fragment (``kraken_source = record_htr_text``).
* **Geometry.** A record's ``bbox`` is the evidence box (VLM box joined with the assigned fragments); the pipeline
  matched with the VLM's own boxes, which only the raw cache keeps. They are read from the raw cache after the
  pipeline's loop-repeat dedupe when the texts line up with the record's lines, else the record's ``bbox`` is used.
* **Gap marks.** A bracket mark without letters (``[...]``, ``[?]``; at most 4 characters inside) is voted as ONE
  token and written back as it was, because ``describe_from_reads`` relies on ``[...]``: no grounded span may cross
  a gap. The benchmark vote stripped such apparatus with ``normalize_ink_hypothesis`` (the scorer strips it on both
  sides); here each line is only NFC-normalised and whitespace-collapsed.

Vote. Hypotheses are each VLM read's lines joined by newlines, plus (with ``kraken_voter``) Kraken's fragments in
Kraken's order, the way the benchmark's ``kraken_seg`` voter was a page of Kraken lines. A read with no line is left
out (as an empty voter it would delete every character Kraken did not also read). Backbone (highest mean
similarity), merge order, gap cost and the per-slot vote (NULL is a candidate, ties go to the backbone) are
``rover_consensus``'s, unweighted and unfiltered. A page with one VLM read is passed through unchanged
(``voted: false``): with two voters every disagreement is a tie the backbone wins, so a VLM + Kraken vote would only
copy one of the two. Two VLMs without Kraken do vote, which returns the backbone's text (the first read, on a tie).

Agreement. A voted line has no box of its own, and the pipeline's matching is geometric, so the box decides which
fragments a line is compared with (on real pages a reader with drifting boxes paired 0 of 30 lines where another
paired 11). Each page therefore has a geometry reader: the VLM read whose OWN lines pair best with the page's
fragments (most agreed lines under the current rule; independent of the vote; ties: characters backed, read order).
A voted line takes that reader's source line when it backs at least half the line's characters, else the VLM source
line backing most of them (Kraken's fragment boxes only when no VLM wrote anything there), and is then matched to
Kraken's fragments exactly as ``two_reader_lines.sidecar`` matches a VLM read: loop repeats dropped,
``line_rule.assign_fragments`` (geometric), ``similarity`` (letters only, ``1 - lev / max``), ``line_status``
(``agreed`` at >= 0.8), the clamped evidence box. ``two_reader_lines`` loads ``.env`` at import, so its few
non-``line_rule`` helpers (``sidecar``'s line loop, ``drop_loop_duplicates``, ``_clamp_box``, ``raw_stem``,
``cached_frags``) are re-implemented here; a test pins them to the originals by compiling those functions from the
pipeline's source without importing it. With Kraken as a voter the vote leans towards Kraken where the VLMs split, so
the agreement is not independent of the vote there: ``rover.lines[i].kraken_only`` counts the characters only Kraken
backed.

Output record: the pipeline's envelope and ``ai_read`` keys, with ``vlm_model = rover<N>`` (N voters, Kraken
included), ``vlm_revision = "<reader>=<revision>; ...; kraken=<htr key>"``, ``rule_version`` = the current line rule,
``htr_model`` / ``htr_cache_key`` carried over from the read whose fragments were used, and a top-level ``rover``
block: voters, unit, backbone, the single-reader records it was built from (``<file>:<line>``), per-line provenance
(``attested_by`` = characters each reader backed, ``full`` = readers who wrote the whole line, ``hybrid`` = nobody
did, ``non_input_words`` = words no reader wrote anywhere on the page, ``kraken_only``, ``source_lines``,
``box_from``) and page counts.

Usage (repo root; ``--reads`` order is the read order used for ties)::

    A=src/datasets/raw_data/cairo_genizah/ai_reads; P=logs/next_round/consensus_pool
    .venv/bin/python -m src.datasets.consensus.rover_reads \\
        --reads $A/ai_reads_qwen3-vl-8b-heb-v21b-step1200_vectorless_pilot200.jsonl \\
                $A/ai_reads_qwen3-vl-8b-heb-v22i-a50_vectorless_pilot200.jsonl \\
                $A/ai_reads_qwen3-vl-8b-heb-v23a-step1500_vectorless_pilot200.jsonl \\
        --jobs $P/jobs_vectorless_pilot200_cap4_all.jsonl \\
        --out $P/ai_reads_rover_vectorless_pilot200.jsonl [--dry-run] [--unit word] [--no-kraken-voter]
"""
import argparse
import json
import os
import re
import statistics
import unicodedata
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from src.datasets.consensus.line_rule import (
    RULE_VERSION, assign_fragments, evidence_box, letters, line_status, similarity)


def _import_rover_consensus() -> ModuleType:
    """Import ``rover_consensus`` (alignment and voting) without the ``.env`` load its scorer import makes.

    ``rover_consensus`` imports ``score_genizah_offline``, which calls ``dotenv.load_dotenv()`` at import time.
    Voting needs no credentials, so that call is a no-op while the module is imported (if the scorer was imported
    earlier in the process, the load already happened there).

    :return: The ``rover_consensus`` module.
    :rtype: ModuleType
    """
    import dotenv

    load_dotenv = dotenv.load_dotenv
    dotenv.load_dotenv = lambda *args, **kwargs: False
    try:
        from src.datasets.evaluations.helper_eval_scripts import rover_consensus
    finally:
        dotenv.load_dotenv = load_dotenv
    return rover_consensus


rc = _import_rover_consensus()

_REPO = Path(__file__).resolve().parents[3]
RAW_DIR = _REPO / "src/datasets/raw_data/cairo_genizah/ai_reads/raw"   # two_reader_lines.RAW_DIR
LEGACY_HTR_MODEL = "MiDRASH_Gen_01"   # two_reader_lines.LEGACY_HTR_MODEL: reader of raw-cache frags without htr_model
LOOP_MIN_LETTERS = 8                  # two_reader_lines.drop_loop_duplicates default
KRAKEN = "kraken"                     # voter name of Kraken's hypothesis
ROVER_VERSION = "rover-reads-v1-20261008"
MARK_RE = re.compile(r"\[[^\]\n\w]{0,4}\]")   # [...], [..], [?], [ ]: bracket marks without letters, one token each
ENVELOPE = ("doc_id", "source_index", "image_index", "image_url", "image_width", "image_height", "image_sha256")
_PUA_FIRST = 0xE000                   # private-use code points stand in for gap marks while voting
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")
Box = Tuple[float, float, float, float]


# ----------------------------------------------------------------------------- pipeline mirrors


def raw_stem(job: Mapping[str, Any], vlm_model: str) -> str:
    """File stem of a page's raw-cache entry (mirrors ``two_reader_lines.raw_stem``).

    :param job: ``{doc_id, image_index}`` (a record works).
    :param vlm_model: VLM checkpoint.
    :return: Stem without ``.json``.
    :rtype: str
    """
    return _UNSAFE.sub("_", job["doc_id"]) + f"__{job['image_index']}__{vlm_model}"


def cached_frags(entry: Mapping[str, Any], key: str) -> Optional[List[Dict[str, Any]]]:
    """Kraken fragments a raw-cache entry holds for one HTR cache key (mirrors ``two_reader_lines.cached_frags``).

    :param entry: Parsed raw-cache JSON.
    :param key: HTR cache key (``htr_cache_key``, else the HTR model name).
    :return: The fragments, or None when the entry has none for ``key``.
    :rtype: Optional[List[Dict[str, Any]]]
    """
    by_htr = entry.get("frags_by_htr") or {}
    if key in by_htr:
        return by_htr[key]
    if key == entry.get("htr_model", LEGACY_HTR_MODEL):
        return entry.get("frags")
    return None


def clamp_box(box: Sequence[float]) -> List[int]:
    """Clamp to 0-1000 ints and order the corners (mirrors ``two_reader_lines._clamp_box``).

    :param box: ``[x1, y1, x2, y2]``.
    :return: Clamped box.
    :rtype: List[int]
    """
    x1, y1, x2, y2 = (int(round(max(0.0, min(1000.0, float(v))))) for v in box)
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def loop_repeat_keep(texts: Sequence[str], min_letters: int = LOOP_MIN_LETTERS) -> List[int]:
    """Indices kept by the pipeline's loop-repeat rule: later exact (letters-only) repeats of long lines go.

    :param texts: Line texts in reading order.
    :param min_letters: Minimum letters for a line to be subject to the rule.
    :return: Indices of the kept lines, in order.
    :rtype: List[int]
    """
    seen, keep = set(), []
    for i, text in enumerate(texts):
        key = letters(text)
        if len(key) >= min_letters and key in seen:
            continue
        seen.add(key)
        keep.append(i)
    return keep


def drop_loop_duplicates(vlm_lines: Sequence[Dict[str, Any]],
                         min_letters: int = LOOP_MIN_LETTERS) -> List[Dict[str, Any]]:
    """Drop later exact repeats of a line (mirrors ``two_reader_lines.drop_loop_duplicates``).

    :param vlm_lines: ``{text, box}`` lines in reading order.
    :param min_letters: Minimum letters for a line to be subject to the rule.
    :return: Lines with loop repeats removed.
    :rtype: List[Dict[str, Any]]
    """
    return [vlm_lines[i] for i in loop_repeat_keep([ln["text"] for ln in vlm_lines], min_letters)]


def rederive_lines(lines: Sequence[Mapping[str, Any]],
                   frags: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Line records for ``{text, box}`` lines matched to Kraken fragments, as ``two_reader_lines.sidecar`` builds them.

    Loop repeats are dropped, fragments are assigned by ``line_rule.assign_fragments``, a line's Kraken text is its
    fragments joined right to left, ``agreement`` is the letters-only similarity (4 dp), ``status`` comes from
    ``line_status`` (``agreed`` at >= 0.8 with Kraken text), and ``bbox`` is the clamped evidence box.

    :param lines: Lines with ``text`` and ``box`` (0-1000).
    :param frags: Kraken fragments with ``text`` and ``box`` (0-1000).
    :return: ``(line records, indices of the input lines kept)``.
    :rtype: Tuple[List[Dict[str, Any]], List[int]]
    """
    keep = loop_repeat_keep([str(ln["text"]) for ln in lines])
    kept = [lines[i] for i in keep]
    groups = assign_fragments(kept, frags)
    out = []
    for i, ln in enumerate(kept):
        htr_text = " ".join(frags[j]["text"] for j in groups[i]).strip() or None
        agreement = round(similarity(ln["text"], htr_text), 4) if htr_text else None
        out.append({
            "index": i, "text": ln["text"],
            "bbox": clamp_box(evidence_box(ln["box"], [frags[j]["box"] for j in groups[i]])),
            "agreement": agreement, "status": line_status(agreement, htr_text),
            "htr_text": htr_text, "htr_fragments": [clamp_box(frags[j]["box"]) for j in groups[i]],
        })
    return out, keep


def fragment_dicts(lines: Sequence["SourceLine"]) -> List[Dict[str, Any]]:
    """``{text, box}`` dicts of source lines, the shape :func:`rederive_lines` takes for lines and fragments.

    :param lines: Lines.
    :return: Dicts.
    :rtype: List[Dict[str, Any]]
    """
    return [{"text": line.text, "box": list(line.box)} for line in lines]


def record_line_agreed(line: Mapping[str, Any]) -> bool:
    """The pipeline's agreement flag of a record line (``status``, else re-derived as ``describe_from_reads`` does).

    :param line: One ``ai_read.lines`` entry.
    :return: True when the line is ``agreed``.
    :rtype: bool
    """
    if "status" in line:
        return line["status"] == "agreed"
    return line_status(line.get("agreement"), line.get("htr_text")) == "agreed"


# ----------------------------------------------------------------------------- page reads


@dataclass(frozen=True)
class SourceLine:
    """One line of one reader.

    :param text: The reader's text as recorded.
    :param box: ``[x1, y1, x2, y2]`` in the 0-1000 frame of the oriented image both readers saw.
    """

    text: str
    box: Box


@dataclass
class PageRead:
    """One reader's read of one page, as the vote needs it.

    :param page_key: Page identity (see :func:`page_key`).
    :param record: The ``two_reader_lines`` record as read.
    :param record_id: ``<reads file name>:<line number>`` of the record.
    :param vlm_lines: The record's lines in order, each with the VLM's own box (raw cache) or the record's ``bbox``.
    :param box_source: ``raw_cache`` or ``record_bbox``.
    :param kraken_lines: Kraken's fragments in Kraken's order (raw cache), else one pseudo-fragment per record line
        that has ``htr_text``.
    :param kraken_source: ``raw_cache``, ``record_htr_text`` or ``none``.
    :param job: The job row the page matched, when a job list was given.
    """

    page_key: str
    record: Dict[str, Any]
    record_id: str
    vlm_lines: List[SourceLine]
    box_source: str
    kraken_lines: List[SourceLine]
    kraken_source: str
    job: Optional[Dict[str, Any]] = None

    @property
    def ai_read(self) -> Dict[str, Any]:
        """The record's ``ai_read`` block.

        :return: Block (empty when missing).
        :rtype: Dict[str, Any]
        """
        return self.record.get("ai_read") or {}

    @property
    def reader(self) -> str:
        """VLM checkpoint that made the read.

        :return: ``ai_read.vlm_model``.
        :rtype: str
        """
        return str(self.ai_read.get("vlm_model") or "unknown")

    @property
    def revision(self) -> Optional[str]:
        """Revision of the VLM checkpoint.

        :return: ``ai_read.vlm_revision``.
        :rtype: Optional[str]
        """
        return self.ai_read.get("vlm_revision")

    @property
    def htr_key(self) -> str:
        """HTR cache key of the read's Kraken fragments.

        :return: ``htr_cache_key``, else ``htr_model``, else the legacy model name.
        :rtype: str
        """
        return str(self.ai_read.get("htr_cache_key") or self.ai_read.get("htr_model") or LEGACY_HTR_MODEL)

    @property
    def parsed(self) -> bool:
        """Whether the VLM reply parsed as a JSON array.

        :return: ``ai_read.parsed``.
        :rtype: bool
        """
        return bool(self.ai_read.get("parsed", True))

    @property
    def decoded_at(self) -> str:
        """When the read was made.

        :return: ISO time ('' when unknown).
        :rtype: str
        """
        return str(self.ai_read.get("decoded_at") or "")

    @property
    def agreed(self) -> List[bool]:
        """The record's per-line agreement flags.

        :return: One flag per record line.
        :rtype: List[bool]
        """
        return [record_line_agreed(ln) for ln in self.ai_read.get("lines") or []]


def read_records(paths: Sequence[Path]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Records of reads JSONL files with their ids (a record still being appended is skipped).

    :param paths: Files, in read order.
    :return: ``(<file name>:<line number>, record)`` pairs.
    :rtype: Iterator[Tuple[str, Dict[str, Any]]]
    """
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                if line.endswith("\n") and line.strip():
                    yield f"{path.name}:{n}", json.loads(line)


def url_hashes(records: Iterable[Mapping[str, Any]]) -> Dict[str, str]:
    """Image URL to image hash, for URLs recorded with exactly one hash.

    :param records: Records.
    :return: ``{image_url: image_sha256}``.
    :rtype: Dict[str, str]
    """
    seen: Dict[str, set] = {}
    for record in records:
        if record.get("image_url") and record.get("image_sha256"):
            seen.setdefault(record["image_url"], set()).add(record["image_sha256"])
    return {url: next(iter(hashes)) for url, hashes in seen.items() if len(hashes) == 1}


def page_key(record: Mapping[str, Any], url_sha: Optional[Mapping[str, str]] = None) -> str:
    """Page identity of a record: its image bytes where known.

    :param record: A ``two_reader_lines`` record.
    :param url_sha: URL to hash from other records (see :func:`url_hashes`).
    :return: ``image_sha256``; else the hash recorded elsewhere for its URL; else ``image_url``; else
        ``doc_id#image_index``.
    :rtype: str
    """
    if record.get("image_sha256"):
        return str(record["image_sha256"])
    url = record.get("image_url")
    if url and url_sha and url in url_sha:
        return url_sha[url]
    if url:
        return str(url)
    return f"{record.get('doc_id')}#{record.get('image_index')}"


def load_jobs(path: Any) -> List[Dict[str, Any]]:
    """Job rows (``doc_id``, ``image_index``, ``image_url``, ...) of a job JSONL.

    :param path: File.
    :return: Rows in file order.
    :rtype: List[Dict[str, Any]]
    """
    return [record for _, record in read_records([Path(path)])]


def raw_entry(record: Mapping[str, Any], raw_dir: Path) -> Optional[Dict[str, Any]]:
    """The pipeline's raw-cache entry of a record, when it belongs to the same read.

    Paths follow ``two_reader_lines.raw_cache_path`` (per-model name, then the model-less name of the first pilot
    runs). An entry is used only when its VLM, URL and hash match the record: the cache file of a (page, checkpoint)
    is shared by every run of that checkpoint, so a later run may have replaced it.

    :param record: A ``two_reader_lines`` record.
    :param raw_dir: Raw-cache directory.
    :return: The parsed entry, or None.
    :rtype: Optional[Dict[str, Any]]
    """
    vlm = (record.get("ai_read") or {}).get("vlm_model")
    if vlm is None or "doc_id" not in record or "image_index" not in record:
        return None
    candidates = (raw_dir / f"{raw_stem(record, vlm)}.json",
                  raw_dir / f"{_UNSAFE.sub('_', record['doc_id'])}__{record['image_index']}.json")
    for path in candidates:
        if not path.exists():
            continue
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:          # a cache file cut short (disk full) is a missing one
            continue
        same_hash = not record.get("image_sha256") or not entry.get("sha256") \
            or entry["sha256"] == record["image_sha256"]
        if entry.get("vlm_model") == vlm and entry.get("image_url") == record.get("image_url") and same_hash:
            return entry
    return None


def raw_vlm_boxes(entry: Mapping[str, Any], texts: Sequence[str]) -> Optional[List[Box]]:
    """The VLM's own line boxes from a raw-cache entry, aligned with a record's lines.

    :param entry: Raw-cache entry.
    :param texts: The record's line texts.
    :return: One box per record line, or None when the entry's lines (after the loop-repeat dedupe the record went
        through) do not match the record's texts.
    :rtype: Optional[List[Box]]
    """
    lines = drop_loop_duplicates(list(entry.get("vlm_lines") or []))
    if [ln["text"] for ln in lines] != list(texts):
        return None
    return [tuple(float(v) for v in ln["box"]) for ln in lines]


def record_kraken_lines(ai_lines: Sequence[Mapping[str, Any]]) -> List[SourceLine]:
    """Kraken's text as a record keeps it: one pseudo-fragment per line with ``htr_text``.

    :param ai_lines: ``ai_read.lines`` of a record.
    :return: ``htr_text`` with the union of the line's ``htr_fragments`` boxes, in the record's line order.
    :rtype: List[SourceLine]
    """
    out = []
    for ln in ai_lines:
        text, boxes = str(ln.get("htr_text") or "").strip(), ln.get("htr_fragments") or []
        if text and boxes:
            out.append(SourceLine(text, tuple(float(v) for v in evidence_box(boxes[0], boxes[1:]))))
    return out


def page_read(record: Dict[str, Any], record_id: str, key: str, raw_dir: Optional[Path],
              job: Optional[Dict[str, Any]] = None) -> PageRead:
    """Build a :class:`PageRead` from a record and, when it matches, its raw-cache entry.

    :param record: A ``two_reader_lines`` record.
    :param record_id: ``<file>:<line>`` of the record.
    :param key: Page identity.
    :param raw_dir: Raw-cache directory (None: records only).
    :param job: Matching job row.
    :return: The read.
    :rtype: PageRead
    """
    ai = record.get("ai_read") or {}
    ai_lines = ai.get("lines") or []
    texts = [str(ln.get("text") or "") for ln in ai_lines]
    entry = raw_entry(record, raw_dir) if raw_dir is not None else None
    boxes = raw_vlm_boxes(entry, texts) if entry is not None else None
    box_source = "raw_cache"
    if boxes is None:
        box_source = "record_bbox"
        boxes = [tuple(float(v) for v in (ln.get("bbox") or (0, 0, 1000, 1000))) for ln in ai_lines]
    htr_key = str(ai.get("htr_cache_key") or ai.get("htr_model") or LEGACY_HTR_MODEL)
    frags = cached_frags(entry, htr_key) if entry is not None else None
    if frags is not None:
        kraken = [SourceLine(str(f.get("text") or ""), tuple(float(v) for v in f["box"])) for f in frags]
        kraken_source = "raw_cache"
    else:
        kraken = record_kraken_lines(ai_lines)
        kraken_source = "record_htr_text" if kraken else "none"
    return PageRead(page_key=key, record=record, record_id=record_id,
                    vlm_lines=[SourceLine(t, b) for t, b in zip(texts, boxes)], box_source=box_source,
                    kraken_lines=kraken, kraken_source=kraken_source, job=job)


def read_rank(record: Mapping[str, Any], position: int) -> Tuple[bool, str, int]:
    """Sort key among reads of one page by one reader: the highest is used.

    :param record: A record.
    :param position: Its position in the input (later = appended later).
    :return: ``(parsed, decoded_at, position)``.
    :rtype: Tuple[bool, str, int]
    """
    ai = record.get("ai_read") or {}
    return bool(ai.get("parsed", True)), str(ai.get("decoded_at") or ""), position


def collect_page_reads(paths: Sequence[Any], job_rows: Optional[Sequence[Mapping[str, Any]]] = None,
                       raw_dir: Optional[Path] = RAW_DIR,
                       stats: Optional[Counter] = None) -> Dict[str, List[PageRead]]:
    """Group the records of several reads files by page: one :class:`PageRead` per (reader, page).

    :param paths: ``two_reader_lines`` JSONL files (files still being appended are fine); the order in which readers
        first appear (file order) is the read order, which decides ties in the vote.
    :param job_rows: Pages to keep (``image_url``, else ``doc_id`` + ``image_index`` for records without a URL); None
        keeps every page.
    :param raw_dir: The pipeline's raw-cache directory (None: use the records only).
    :param stats: Counter filled with ``records``, ``outside_jobs`` and ``duplicate_reads``.
    :return: ``{page_key: reads in read order}``, pages in job order (else first appearance).
    :rtype: Dict[str, List[PageRead]]
    """
    stats = Counter() if stats is None else stats
    rows = list(read_records([Path(p) for p in paths]))
    url_sha = url_hashes(record for _, record in rows)
    by_url: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    by_pair: Dict[Tuple[str, int], Tuple[int, Dict[str, Any]]] = {}
    for n, job in enumerate(job_rows or ()):
        if job.get("image_url"):
            by_url.setdefault(str(job["image_url"]), (n, dict(job)))
        by_pair.setdefault((str(job.get("doc_id")), int(job.get("image_index", -1))), (n, dict(job)))
    # page -> reader -> (first position of the reader on the page, position of the kept read, record, id, job)
    chosen: Dict[str, Dict[str, Tuple[int, int, Dict[str, Any], str, Optional[Dict[str, Any]]]]] = {}
    page_order: Dict[str, Tuple[float, int]] = {}
    for position, (record_id, record) in enumerate(rows):
        stats["records"] += 1
        job_position, job = float("inf"), None
        if job_rows is not None:
            url = record.get("image_url")
            hit = by_url.get(url) if url else by_pair.get((str(record.get("doc_id")),
                                                            int(record.get("image_index", -1))))
            if hit is None:
                stats["outside_jobs"] += 1
                continue
            job_position, job = hit
        key = page_key(record, url_sha)
        page_order.setdefault(key, (job_position, position))
        reader = str((record.get("ai_read") or {}).get("vlm_model") or "unknown")
        per_reader = chosen.setdefault(key, {})
        first = position
        if reader in per_reader:
            stats["duplicate_reads"] += 1
            first, kept_position, kept = per_reader[reader][:3]
            if read_rank(record, position) <= read_rank(kept, kept_position):
                continue
        per_reader[reader] = (first, position, record, record_id, job)
    pages: Dict[str, List[PageRead]] = {}
    for key in sorted(chosen, key=lambda k: page_order[k]):
        reads = sorted(chosen[key].values(), key=lambda item: item[0])
        pages[key] = [page_read(record, record_id, key, raw_dir, job) for _, _, record, record_id, job in reads]
    return pages


def page_kraken(reads: Sequence[PageRead]) -> Tuple[Optional[PageRead], List[SourceLine]]:
    """The read whose Kraken lines a page uses.

    :param reads: The page's reads in read order.
    :return: ``(read, its Kraken lines)``: raw-cache fragments first, then the HTR key most reads share, then read
        order; ``(None, [])`` without reads.
    :rtype: Tuple[Optional[PageRead], List[SourceLine]]
    """
    if not reads:
        return None, []
    keys = Counter(r.htr_key for r in reads)
    best = max(reads, key=lambda r: (r.kraken_source == "raw_cache", r.kraken_source != "none", keys[r.htr_key]))
    return best, best.kraken_lines


# ----------------------------------------------------------------------------- hypotheses and vote


@dataclass(frozen=True)
class Hypothesis:
    """One voter's page as the vote sees it.

    :param name: Reader name (a VLM checkpoint or ``kraken``).
    :param lines: Prepared lines (NFC, gap marks encoded, whitespace collapsed; empty lines dropped).
    :param sources: Index of each prepared line in the reader's source lines.
    """

    name: str
    lines: Tuple[str, ...]
    sources: Tuple[int, ...]


def mark_codebook(texts: Sequence[str]) -> Dict[str, str]:
    """A private-use stand-in for every distinct gap mark in ``texts`` (never a character the texts contain).

    :param texts: Line texts of all voters (NFC).
    :return: ``{mark: one character}``.
    :rtype: Dict[str, str]
    """
    marks = sorted({mark for text in texts for mark in MARK_RE.findall(text)})
    used = {ch for text in texts for ch in text}
    book: Dict[str, str] = {}
    code = _PUA_FIRST
    for mark in marks:
        while chr(code) in used:
            code += 1
        book[mark] = chr(code)
        code += 1
    return book


def encode_marks(text: str, book: Mapping[str, str]) -> str:
    """Replace gap marks with their stand-ins.

    :param text: NFC text.
    :param book: From :func:`mark_codebook`.
    :return: Encoded text.
    :rtype: str
    """
    return MARK_RE.sub(lambda m: book[m.group(0)], text) if book else text


def decode_marks(text: str, book: Mapping[str, str]) -> str:
    """Put the gap marks back.

    :param text: Encoded text.
    :param book: From :func:`mark_codebook`.
    :return: Text with the original marks.
    :rtype: str
    """
    if not book:
        return text
    inverse = {v: k for k, v in book.items()}
    return "".join(inverse.get(ch, ch) for ch in text)


def prepare_hypotheses(names: Sequence[str], sources: Sequence[Sequence[SourceLine]]
                       ) -> Tuple[List[Hypothesis], Dict[str, str]]:
    """Voter pages for the vote: each line NFC-normalised, gap marks encoded, whitespace collapsed.

    :param names: Voter names.
    :param sources: Each voter's source lines, in reading order.
    :return: ``(hypotheses, gap-mark codebook)``.
    :rtype: Tuple[List[Hypothesis], Dict[str, str]]
    """
    nfc = [[unicodedata.normalize("NFC", line.text) for line in lines] for lines in sources]
    book = mark_codebook([text for texts in nfc for text in texts])
    hyps = []
    for name, texts in zip(names, nfc):
        prepared = [(i, " ".join(encode_marks(text, book).split())) for i, text in enumerate(texts)]
        prepared = [(i, text) for i, text in prepared if text]
        hyps.append(Hypothesis(name, tuple(t for _, t in prepared), tuple(i for i, _ in prepared)))
    return hyps, book


def token_sources(hyp: Hypothesis, unit: str) -> List[int]:
    """Source line of every token of ``rover_consensus.tokenize(hyp.lines, unit)`` (-1 for newline tokens).

    :param hyp: A hypothesis.
    :param unit: ``char`` or ``word``.
    :return: One source line index per token.
    :rtype: List[int]
    """
    out: List[int] = []
    for k, (line, source) in enumerate(zip(hyp.lines, hyp.sources)):
        if k:
            out.append(-1)
        out.extend([source] * (len(line) if unit == "char" else len(line.split())))
    return out


@dataclass(frozen=True)
class SlotVote:
    """A voted network.

    :param network: ``rover_consensus.Network`` (slots x merged hypotheses).
    :param winners: Winning token per slot (None = NULL won).
    :param backbone: Hypothesis index of the backbone.
    :param token_at: Per network column, per slot: index of that column's token in its hypothesis (None = NULL).
    """

    network: Any
    winners: Tuple[Optional[str], ...]
    backbone: int
    token_at: Tuple[Tuple[Optional[int], ...], ...]


def vote_hypotheses(hyps: Sequence[Hypothesis], unit: str) -> SlotVote:
    """Align and vote hypotheses exactly as ``rover_consensus.rover`` does, unweighted, keeping every slot.

    :param hyps: Prepared hypotheses (at least one).
    :param unit: ``char`` or ``word``.
    :return: The voted network.
    :rtype: SlotVote
    """
    tokens = [rc.tokenize(h.lines, unit) for h in hyps]
    weights = [1.0] * len(hyps)
    sim = rc.similarity_matrix([" ".join(h.lines) for h in hyps])
    backbone = rc.choose_backbone(sim, weights)
    gap = int(round(rc.DEFAULT_GAP[unit] * rc.COST_SCALE))
    network = rc.build_network(tokens, rc.merge_order(sim, backbone), gap)
    winners = tuple(rc.vote_slot(slot, network.members, weights, backbone) for slot in network.slots)
    token_at = []
    for c in range(len(network.members)):
        k, column = 0, []
        for slot in network.slots:
            column.append(k if slot[c] is not None else None)
            k += slot[c] is not None
        token_at.append(tuple(column))
    return SlotVote(network, winners, backbone, tuple(token_at))


def line_regions(winners: Sequence[Optional[str]]) -> List[List[int]]:
    """Slots between consecutive voted newlines (the newline slots excluded); one region per output row.

    :param winners: Winning token per slot.
    :return: Slot indices per region, in order (regions may be empty).
    :rtype: List[List[int]]
    """
    regions: List[List[int]] = [[]]
    for s, winner in enumerate(winners):
        if winner == rc.NEWLINE:
            regions.append([])
        else:
            regions[-1].append(s)
    return regions


def join_tokens(tokens: Sequence[str], unit: str) -> str:
    """One row of text from its tokens, as ``rover_consensus.detokenize`` writes a row.

    :param tokens: Tokens (no newline tokens).
    :param unit: ``char`` or ``word``.
    :return: Text with whitespace collapsed.
    :rtype: str
    """
    if unit == "char":
        return " ".join("".join(tokens).split())
    return " ".join(tokens)


def is_unit(token: Optional[str]) -> bool:
    """Whether a token counts as a character / word (not NULL, whitespace or a newline).

    :param token: Token.
    :return: True for a countable token.
    :rtype: bool
    """
    return token is not None and token.strip() != ""


@dataclass
class LineEvidence:
    """What the voters contributed to one voted line.

    :param text: Voted text (gap marks encoded).
    :param support: Hypothesis index -> voted tokens it wrote at their slot.
    :param source_support: ``(hypothesis, source line)`` -> voted tokens it backed.
    :param source_tokens: ``(hypothesis, source line)`` -> tokens it wrote in the line's slots.
    :param n_units: Voted characters (words at word level), spaces excluded.
    :param kraken_only: Voted tokens backed by Kraken alone.
    :param full: Hypotheses whose text in the line's slots is exactly the voted line.
    """

    text: str
    support: Counter = field(default_factory=Counter)
    source_support: Counter = field(default_factory=Counter)
    source_tokens: Counter = field(default_factory=Counter)
    n_units: int = 0
    kraken_only: int = 0
    full: List[int] = field(default_factory=list)


def line_evidence(region: Sequence[int], vote: SlotVote, sources: Sequence[Sequence[int]], unit: str,
                  kraken: Optional[int]) -> Optional[LineEvidence]:
    """Text and provenance of the line voted in one region.

    :param region: Slot indices (see :func:`line_regions`).
    :param vote: The voted network.
    :param sources: Per hypothesis, the source line of each token (see :func:`token_sources`).
    :param unit: ``char`` or ``word``.
    :param kraken: Hypothesis index of Kraken (None when Kraken does not vote).
    :return: The evidence, or None when the region votes no text.
    :rtype: Optional[LineEvidence]
    """
    slots, members = vote.network.slots, vote.network.members
    text = join_tokens([vote.winners[s] for s in region if vote.winners[s] is not None], unit)
    if not text:
        return None
    ev = LineEvidence(text)
    written: Dict[int, List[str]] = {h: [] for h in members}
    for s in region:
        winner, backers = vote.winners[s], []
        for c, entry in enumerate(slots[s]):
            if entry is None:
                continue
            h = members[c]
            if entry == rc.NEWLINE:
                if unit == "char":
                    written[h].append(" ")
                continue
            written[h].append(entry)
            source = sources[h][vote.token_at[c][s]]
            ev.source_tokens[(h, source)] += 1
            if is_unit(winner) and entry == winner:
                backers.append(h)
                ev.support[h] += 1
                ev.source_support[(h, source)] += 1
        if is_unit(winner):
            ev.n_units += 1
            if kraken is not None and backers == [kraken]:
                ev.kraken_only += 1
    ev.full = sorted(h for h in written if join_tokens(written[h], unit) == text)
    return ev


def geometry_scores(reads: Sequence[PageRead], kraken_lines: Sequence[SourceLine]) -> List[int]:
    """How well each read's own boxes pair its lines with Kraken: agreed lines under the current rule.

    Measured on the read's own text against the fragments the page's vote uses, so it does not depend on the vote.

    :param reads: VLM reads of the page.
    :param kraken_lines: The page's Kraken fragments.
    :return: Agreed lines per read.
    :rtype: List[int]
    """
    frags = fragment_dicts(kraken_lines)
    return [sum(ln["status"] == "agreed" for ln in rederive_lines(fragment_dicts(read.vlm_lines), frags)[0])
            for read in reads]


def choose_box(ev: LineEvidence, sources: Sequence[Sequence[SourceLine]], names: Sequence[str], n_vlm: int,
               anchor: int, kraken: Optional[int]) -> Tuple[Box, str]:
    """Box for a voted line: the geometry reader's line when it backs at least half the line, else the VLM source
    line backing most of it.

    :param ev: The line's evidence.
    :param sources: Each hypothesis's source lines (VLM readers first).
    :param names: Hypothesis names.
    :param n_vlm: Number of VLM hypotheses (indices below it).
    :param anchor: The page's geometry reader (see :func:`geometry_scores`).
    :param kraken: Kraken's hypothesis index, used only when no VLM wrote anything in the line.
    :return: ``(box, "<reader>#<source line>")``.
    :rtype: Tuple[Box, str]
    """
    vlm = [k for k in ev.source_tokens if k[0] < n_vlm]
    if vlm:
        h, i = max(vlm, key=lambda k: (k[0] == anchor and 2 * ev.source_support[k] >= ev.n_units,
                                       ev.source_support[k], ev.source_tokens[k], k[0] == anchor, -k[0], -k[1]))
        return sources[h][i].box, f"{names[h]}#{i}"
    kraken_lines = sorted(i for h, i in ev.source_tokens if h == kraken)
    boxes = [sources[kraken][i].box for i in kraken_lines]
    return tuple(evidence_box(boxes[0], boxes[1:])), f"{KRAKEN}#{','.join(map(str, kraken_lines))}"


@dataclass
class VotedPage:
    """Outcome of :func:`vote_page` for one page.

    :param page_key: Page identity.
    :param reads: All reads of the page, in read order.
    :param voting: The VLM reads that voted (the single read of a pass-through).
    :param voters: Voter names in hypothesis order (VLM readers, then ``kraken``).
    :param voted: False for a pass-through.
    :param reason: Why no vote happened (``single_vlm_read``, ``no_vlm_lines``), else None.
    :param unit: ``char`` or ``word``.
    :param lines: Output lines in the pipeline's line schema.
    :param provenance: One provenance dict per output line.
    :param kraken_read: Read whose Kraken lines were used.
    :param kraken_lines: Those lines (Kraken's order).
    :param skipped: VLM reads left out of the vote, with the reason.
    :param backbone: Backbone voter (None for a pass-through).
    :param geometry_reader: VLM reader whose boxes the voted lines take first.
    :param geometry_scores: Per VLM reader, its own agreed lines against the page's fragments (the choice above).
    :param n_slots: Network slots (0 for a pass-through).
    :param n_loop_duplicates: Voted lines dropped by the pipeline's loop-repeat rule.
    """

    page_key: str
    reads: List[PageRead]
    voting: List[PageRead]
    voters: List[str]
    voted: bool
    reason: Optional[str]
    unit: str
    lines: List[Dict[str, Any]]
    provenance: List[Dict[str, Any]]
    kraken_read: Optional[PageRead]
    kraken_lines: List[SourceLine]
    skipped: List[Dict[str, str]] = field(default_factory=list)
    backbone: Optional[str] = None
    geometry_reader: Optional[str] = None
    geometry_scores: Dict[str, int] = field(default_factory=dict)
    n_slots: int = 0
    n_loop_duplicates: int = 0

    @property
    def model(self) -> str:
        """Model name of the output record.

        :return: ``rover<number of voters>``.
        :rtype: str
        """
        return f"rover{len(self.voters)}"


def count_units(text: str, unit: str) -> int:
    """Characters (spaces excluded) or words of a line.

    :param text: Line text.
    :param unit: ``char`` or ``word``.
    :return: Count.
    :rtype: int
    """
    return len("".join(text.split())) if unit == "char" else len(text.split())


def passthrough_page(base: PageRead, reads: Sequence[PageRead], voters: List[str], reason: str, unit: str,
                     skipped: List[Dict[str, str]]) -> VotedPage:
    """A page whose single VLM read is kept as recorded (lines and statuses unchanged).

    :param base: The read passed through.
    :param reads: All reads of the page.
    :param voters: Voter names (the reader, plus ``kraken`` when it would have voted).
    :param reason: ``single_vlm_read`` or ``no_vlm_lines``.
    :param unit: ``char`` or ``word``.
    :param skipped: Other reads left out.
    :return: The page.
    :rtype: VotedPage
    """
    lines = deepcopy(base.ai_read.get("lines") or [])
    provenance = []
    for i, ln in enumerate(lines):
        n = count_units(str(ln.get("text") or ""), unit)
        provenance.append({"attested_by": {base.reader: n}, "n_tokens": n, "kraken_only": 0, "full": [base.reader],
                           "hybrid": False, "non_input_words": [], "source_lines": {base.reader: [i]},
                           "box_from": f"{base.reader}#{i}"})
    return VotedPage(page_key=base.page_key, reads=list(reads), voting=[base], voters=voters, voted=False,
                     reason=reason, unit=unit, lines=lines, provenance=provenance, kraken_read=base,
                     kraken_lines=base.kraken_lines, skipped=skipped)


def vote_page(reads: Sequence[PageRead], kraken_voter: bool = True, unit: str = "char") -> VotedPage:
    """ROVER-vote one page's reads and re-derive the voted lines' agreement with Kraken.

    :param reads: The page's reads (one per reader, read order).
    :param kraken_voter: Add Kraken's lines as one more hypothesis (when the page has Kraken text).
    :param unit: ``char`` (the measured best) or ``word``.
    :return: The voted page (a pass-through when fewer than two VLM reads have lines).
    :rtype: VotedPage
    :raises ValueError: For an unknown unit or no reads.
    """
    if unit not in rc.UNITS:
        raise ValueError(f"unknown unit {unit!r}; expected one of {rc.UNITS}")
    if not reads:
        raise ValueError("vote_page needs at least one read")
    reads = list(reads)
    active = [r for r in reads if any(line.text.strip() for line in r.vlm_lines)]
    active_ids = {id(r) for r in active}
    skipped = [{"reader": r.reader, "record": r.record_id, "reason": "no_lines"} for r in reads
               if id(r) not in active_ids]
    if len(active) < 2:
        base = active[0] if active else reads[0]
        has_kraken = kraken_voter and any(line.text.strip() for line in base.kraken_lines)
        return passthrough_page(base, reads, [base.reader] + ([KRAKEN] if has_kraken else []),
                                "single_vlm_read" if active else "no_vlm_lines", unit,
                                [s for s in skipped if s["record"] != base.record_id])

    kraken_read, kraken_lines = page_kraken(reads)
    use_kraken = kraken_voter and any(line.text.strip() for line in kraken_lines)
    names = [r.reader for r in active] + ([KRAKEN] if use_kraken else [])
    sources: List[Sequence[SourceLine]] = [r.vlm_lines for r in active] + ([kraken_lines] if use_kraken else [])
    kraken = len(active) if use_kraken else None
    hyps, book = prepare_hypotheses(names, sources)
    vote = vote_hypotheses(hyps, unit)
    token_src = [token_sources(h, unit) for h in hyps]
    evidence = [ev for ev in (line_evidence(region, vote, token_src, unit, kraken)
                              for region in line_regions(vote.winners)) if ev is not None]
    totals: Counter = Counter()
    for ev in evidence:
        totals.update(ev.support)
    scores = geometry_scores(active, kraken_lines)
    anchor = max(range(len(active)), key=lambda h: (scores[h], totals[h], -h))
    inputs = {word for h in hyps for line in h.lines for word in line.split()}

    candidates, provenance = [], []
    for ev in evidence:
        box, box_from = choose_box(ev, sources, names, len(active), anchor, kraken)
        candidates.append({"text": decode_marks(ev.text, book), "box": list(box)})
        provenance.append({
            "attested_by": {names[h]: ev.support[h] for h in sorted(ev.support)},
            "n_tokens": ev.n_units, "kraken_only": ev.kraken_only,
            "full": [names[h] for h in ev.full], "hybrid": not ev.full,
            "non_input_words": [decode_marks(w, book) for w in ev.text.split() if w not in inputs],
            "source_lines": {names[h]: sorted({i for hh, i in ev.source_tokens if hh == h})
                             for h in sorted({hh for hh, _ in ev.source_tokens})},
            "box_from": box_from,
        })
    lines, keep = rederive_lines(candidates, fragment_dicts(kraken_lines))
    return VotedPage(page_key=active[0].page_key, reads=reads, voting=active, voters=names, voted=True, reason=None,
                     unit=unit, lines=lines, provenance=[provenance[i] for i in keep], kraken_read=kraken_read,
                     kraken_lines=kraken_lines, skipped=skipped, backbone=names[vote.backbone],
                     geometry_reader=names[anchor], geometry_scores=dict(zip(names, scores)),
                     n_slots=len(vote.network.slots), n_loop_duplicates=len(candidates) - len(keep))


# ----------------------------------------------------------------------------- output records


def utc_now() -> str:
    """Current UTC time as the pipeline stamps ``decoded_at``.

    :return: ISO time to the second.
    :rtype: str
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def voter_revisions(page: VotedPage) -> str:
    """``vlm_revision`` of an output record: every voter with its revision (Kraken: its HTR cache key).

    :param page: The voted page.
    :return: ``"<reader>=<revision>; ...; kraken=<htr key>"``.
    :rtype: str
    """
    parts = [f"{r.reader}={r.revision}" for r in page.voting]
    if KRAKEN in page.voters and page.kraken_read is not None:
        parts.append(f"{KRAKEN}={page.kraken_read.htr_key}")
    return "; ".join(parts)


def rover_block(page: VotedPage) -> Dict[str, Any]:
    """The ``rover`` provenance block of an output record.

    :param page: The voted page.
    :return: Block.
    :rtype: Dict[str, Any]
    """
    kr = page.kraken_read
    job = page.voting[0].job if page.voting else None
    return {
        "version": ROVER_VERSION, "voted": page.voted, "reason": page.reason, "unit": page.unit,
        "voters": page.voters, "backbone": page.backbone, "geometry_reader": page.geometry_reader,
        "geometry_scores": page.geometry_scores,
        "sources": [{"reader": r.reader, "revision": r.revision, "record": r.record_id, "decoded_at": r.decoded_at,
                     "parsed": r.parsed, "n_lines": len(r.vlm_lines), "n_agreed": sum(r.agreed),
                     "box_source": r.box_source} for r in page.voting],
        "skipped": page.skipped,
        "kraken": None if kr is None else {"record": kr.record_id, "source": kr.kraken_source,
                                           "htr_key": kr.htr_key, "n_fragments": len(page.kraken_lines)},
        "job": None if job is None else {k: job.get(k) for k in ("doc_id", "image_index", "queue")},
        "n_slots": page.n_slots, "n_lines": len(page.lines),
        "n_hybrid_lines": sum(p["hybrid"] for p in page.provenance),
        "n_words": sum(len(str(ln.get("text") or "").split()) for ln in page.lines),
        "n_non_input_words": sum(len(p["non_input_words"]) for p in page.provenance),
        "n_kraken_only": sum(p["kraken_only"] for p in page.provenance),
        "n_loop_duplicates": page.n_loop_duplicates,
        "lines": page.provenance,
    }


def build_record(page: VotedPage, decoded_at: Optional[str] = None) -> Dict[str, Any]:
    """The output record of a page, in the pipeline's schema plus the ``rover`` block.

    A pass-through keeps the read's ``ai_read`` (lines, statuses, rule version, decode time) and changes only
    ``vlm_model`` and ``vlm_revision``.

    :param page: The voted page.
    :param decoded_at: Stamp for a voted record (default now, UTC).
    :return: Record (envelope keys in the pipeline's order, so ``describe_from_reads.record_head`` reads it).
    :rtype: Dict[str, Any]
    """
    base = page.voting[0].record
    record: Dict[str, Any] = {k: base[k] for k in ENVELOPE if k in base}
    if page.voted:
        kr = page.kraken_read
        htr_model = str((kr.ai_read.get("htr_model") if kr is not None else None) or LEGACY_HTR_MODEL)
        htr_key = kr.htr_key if kr is not None else htr_model
        ai: Dict[str, Any] = {
            "vlm_model": page.model, "vlm_revision": voter_revisions(page), "htr_model": htr_model,
            "rule_version": RULE_VERSION, "decoded_at": decoded_at or utc_now(),
            "parsed": all(r.parsed for r in page.voting), "n_lines": len(page.lines),
            "n_agreed": sum(ln["status"] == "agreed" for ln in page.lines), "lines": page.lines,
        }
        if htr_key != htr_model:
            ai["htr_cache_key"] = htr_key
    else:
        ai = deepcopy(base.get("ai_read") or {})
        ai["vlm_model"] = page.model
        ai["vlm_revision"] = voter_revisions(page)
    record["ai_read"] = ai
    record["rover"] = rover_block(page)
    return record


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Replace ``path`` with ``rows`` as JSONL via a temp file and a rename.

    :param path: Destination.
    :param rows: Rows.
    :return: Rows written.
    :rtype: int
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


# ----------------------------------------------------------------------------- report


def rederive_check(read: PageRead) -> Optional[Counter]:
    """Re-derive a single read's line records from its own boxes and fragments and compare them with the record.

    Only reads whose boxes and fragments come from the raw cache and whose record is under the current line rule
    can match exactly.

    :param read: A read.
    :return: Counts (``lines`` and the lines whose ``status`` / ``agreement`` / ``htr_text`` / ``bbox`` match), or
        None when the read cannot be checked.
    :rtype: Optional[Counter]
    """
    if read.box_source != "raw_cache" or read.kraken_source != "raw_cache" \
            or read.ai_read.get("rule_version") != RULE_VERSION:
        return None
    lines, _ = rederive_lines(fragment_dicts(read.vlm_lines), fragment_dicts(read.kraken_lines))
    recorded = read.ai_read.get("lines") or []
    counts = Counter(lines=len(recorded))
    if len(lines) != len(recorded):
        counts["length_mismatch"] += 1
        return counts
    for new, old in zip(lines, recorded):
        for key in ("status", "agreement", "htr_text", "bbox"):
            counts[key] += new[key] == old.get(key)
    return counts


def agreed_on_voted_pages(voted: Sequence[VotedPage]) -> Dict[str, Tuple[int, int]]:
    """Agreed lines on voted pages, every voting read and the output measured alike.

    Each voting read is re-derived under the current line rule against the fragments its page's vote used, so
    reads recorded under an older rule or another Kraken key compare fairly with the output. With Kraken as a voter
    the output is pulled towards Kraken, so its count is biased upwards.

    :param voted: Voted pages (pass-throughs are ignored).
    :return: ``{reader or "output": (agreed lines, lines)}``.
    :rtype: Dict[str, Tuple[int, int]]
    """
    agreed: Counter = Counter()
    total: Counter = Counter()
    for page in voted:
        if not page.voted:
            continue
        frags = fragment_dicts(page.kraken_lines)
        for read in page.voting:
            lines, _ = rederive_lines(fragment_dicts(read.vlm_lines), frags)
            agreed[read.reader] += sum(ln["status"] == "agreed" for ln in lines)
            total[read.reader] += len(lines)
        agreed["output"] += sum(ln["status"] == "agreed" for ln in page.lines)
        total["output"] += len(page.lines)
    return {name: (agreed[name], total[name]) for name in total}


def _stats(values: Sequence[float]) -> str:
    """``mean / median (min-max)`` of a list, or ``-``.

    :param values: Numbers.
    :return: Formatted summary.
    :rtype: str
    """
    if not values:
        return "-"
    return (f"mean {statistics.fmean(values):.1f}, median {statistics.median(values):g} "
            f"({min(values):g}-{max(values):g})")


def summary_lines(pages: Mapping[str, Sequence[PageRead]], voted: Sequence[VotedPage],
                  job_rows: Optional[Sequence[Mapping[str, Any]]], stats: Mapping[str, int]) -> List[str]:
    """Printable dry-run / run report.

    :param pages: Output of :func:`collect_page_reads`.
    :param voted: One voted page per page.
    :param job_rows: The job list (None without one).
    :param stats: Counts from :func:`collect_page_reads`.
    :return: Report lines.
    :rtype: List[str]
    """
    reads = [r for rs in pages.values() for r in rs]
    out = [f"records {stats.get('records', 0)}; outside the job list {stats.get('outside_jobs', 0)}; "
           f"duplicate reads (same reader, same page) {stats.get('duplicate_reads', 0)}"]
    jobs = f" of {len(job_rows)} job pages ({len(job_rows) - len(pages)} without a read yet)" if job_rows else ""
    out.append(f"pages {len(pages)}{jobs}; readers per page {dict(sorted(Counter(map(len, pages.values())).items()))}")
    out.append(f"reads by reader {dict(Counter(r.reader for r in reads))}")
    combos = Counter((p.model, " + ".join(p.voters), "voted" if p.voted else f"pass-through: {p.reason}")
                     for p in voted)
    out.append("voter combinations (pages):")
    out.extend(f"  {n:5d}  {model:7s} {voters}  [{how}]" for (model, voters, how), n in combos.most_common())
    with_kraken = sum(any(ln.text.strip() for ln in p.kraken_lines) for p in voted)
    share = f"{100.0 * with_kraken / len(voted):.1f} %" if voted else "-"
    out.append(f"pages with Kraken fragments {with_kraken}/{len(voted)} ({share}); Kraken source of the reads "
               f"{dict(Counter(r.kraken_source for r in reads))}; "
               f"VLM boxes {dict(Counter(r.box_source for r in reads))}")
    out.append(f"VLM lines per read: {_stats([len(r.vlm_lines) for r in reads])}")
    out.append(f"output lines per page: {_stats([len(p.lines) for p in voted])}")
    in_agreed = sum(sum(r.agreed) for r in reads)
    in_lines = sum(len(r.vlm_lines) for r in reads)
    out_agreed = sum(ln["status"] == "agreed" for p in voted for ln in p.lines)
    out_lines = sum(len(p.lines) for p in voted)
    out.append(f"agreed lines as recorded: input reads {in_agreed}/{in_lines}; output {out_agreed}/{out_lines}")
    n_voted = [p for p in voted if p.voted]
    if n_voted:
        same = agreed_on_voted_pages(n_voted)
        out.append(f"voted pages, agreed lines under {RULE_VERSION} against the fragments each vote used"
                   + (" (Kraken voted: the output count leans towards Kraken)" if any(KRAKEN in p.voters
                                                                                      for p in n_voted) else "")
                   + ": " + "; ".join(f"{name.split('heb-')[-1]} {a}/{n} ({100.0 * a / n:.1f} %)"
                                      for name, (a, n) in same.items() if n))
    out.append(f"voted pages {len(n_voted)}, pass-through {len(voted) - len(n_voted)}; hybrid lines "
               f"{sum(p['hybrid'] for v in n_voted for p in v.provenance)}; non-input words "
               f"{sum(len(p['non_input_words']) for v in n_voted for p in v.provenance)} of "
               f"{sum(len(ln['text'].split()) for v in n_voted for ln in v.lines)}; Kraken-only tokens "
               f"{sum(p['kraken_only'] for v in n_voted for p in v.provenance)}; loop repeats dropped "
               f"{sum(v.n_loop_duplicates for v in n_voted)}; skipped empty reads "
               f"{sum(len(v.skipped) for v in voted)}")
    checks = Counter()
    checked = 0
    for read in reads:
        counts = rederive_check(read)
        if counts is not None:
            checked += 1
            checks.update(counts)
    out.append(f"re-derivation check ({checked} reads with raw-cache boxes and fragments under {RULE_VERSION}): "
               + ", ".join(f"{k} {checks[k]}/{checks['lines']}" for k in ("status", "agreement", "htr_text", "bbox"))
               + (f"; line-count mismatches {checks['length_mismatch']}" if checks["length_mismatch"] else ""))
    return out


def _short(text: str, width: int) -> str:
    """Text cut to ``width`` characters.

    :param text: Text.
    :param width: Maximum length.
    :return: Shortened text.
    :rtype: str
    """
    return text if len(text) <= width else text[:width - 1] + "…"


def example_lines(page: VotedPage, n_lines: int = 3, width: int = 64) -> List[str]:
    """The first lines of a voted page next to the source lines they came from.

    :param page: A voted page.
    :param n_lines: Lines to show.
    :param width: Text width.
    :return: Report lines.
    :rtype: List[str]
    """
    base = page.voting[0].record
    how = "voted" if page.voted else f"pass-through ({page.reason})"
    out = [f"{base.get('doc_id')}#{base.get('image_index')}  {page.model} {how}; voters {' + '.join(page.voters)}"
           + (f"; backbone {page.backbone}" if page.backbone else "")]
    by_reader = {r.reader: r for r in page.voting}
    for line, prov in zip(page.lines[:n_lines], page.provenance[:n_lines]):
        agreement = "-" if line.get("agreement") is None else f"{line['agreement']:.2f}"
        out.append(f"  {'output':>16s} [{line['status']:>11s} {agreement:>4s}] {_short(line['text'], width)}")
        for reader, indices in prov["source_lines"].items():
            if reader == KRAKEN:
                texts, flags = [page.kraken_lines[i].text for i in indices], ""
            else:
                read = by_reader[reader]
                texts = [read.vlm_lines[i].text for i in indices]
                flags = ",".join(str((read.ai_read.get("lines") or [])[i].get("status")) for i in indices)
            out.append(f"  {_short(reader.split('heb-')[-1], 16):>16s} [{_short(flags, 16):>16s}] "
                       f"{_short(' / '.join(texts), width)}")
    return out


# ----------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser.

    :return: Parser.
    :rtype: argparse.ArgumentParser
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reads", nargs="+", required=True,
                    help="two_reader_lines JSONL files; their order is the read order (ties, geometry)")
    ap.add_argument("--jobs", default=None,
                    help="job JSONL (doc_id, image_index, image_url per page): only these pages, in this order")
    ap.add_argument("--out", default=None, help="output JSONL, rewritten atomically (required unless --dry-run)")
    ap.add_argument("--unit", choices=rc.UNITS, default="char")
    ap.add_argument("--no-kraken-voter", action="store_true", help="vote the VLM reads only")
    ap.add_argument("--dry-run", action="store_true", help="report page counts per voter combination; write nothing")
    ap.add_argument("--raw-dir", default=str(RAW_DIR), help="the pipeline's raw-cache directory")
    ap.add_argument("--examples", type=int, default=2, help="example pages printed (voted pages first)")
    return ap


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Collect, vote, report and (unless ``--dry-run``) write.

    :param args: Parsed CLI arguments.
    :return: ``{"pages", "voted", "written"}`` counts.
    :rtype: Dict[str, Any]
    """
    job_rows = load_jobs(args.jobs) if args.jobs else None
    stats: Counter = Counter()
    pages = collect_page_reads(args.reads, job_rows, raw_dir=Path(args.raw_dir), stats=stats)
    voted = [vote_page(reads, kraken_voter=not args.no_kraken_voter, unit=args.unit) for reads in pages.values()]
    print(f"rover_reads unit={args.unit} kraken_voter={not args.no_kraken_voter} rule={RULE_VERSION} "
          f"reads={', '.join(Path(p).name for p in args.reads)}", flush=True)
    print("\n".join(summary_lines(pages, voted, job_rows, stats)), flush=True)
    shown = sorted((p for p in voted if len(p.lines) >= 3), key=lambda p: not p.voted)[:max(args.examples, 0)]
    for k, page in enumerate(shown, 1):
        print(f"\nexample {k}: " + "\n".join(example_lines(page)), flush=True)
    written = 0
    if args.dry_run:
        print("\ndry run: nothing written", flush=True)
    else:
        stamp = utc_now()
        written = write_jsonl_atomic(Path(args.out), (build_record(p, stamp) for p in voted))
        print(f"\nwrote {written} records -> {args.out}", flush=True)
    return {"pages": len(pages), "voted": sum(p.voted for p in voted), "written": written}


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point.

    :param argv: Arguments (None = ``sys.argv``).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run and not args.out:
        parser.error("--out is required unless --dry-run")
    missing = [p for p in args.reads + ([args.jobs] if args.jobs else []) if not Path(p).exists()]
    if missing:
        parser.error(f"not found: {', '.join(missing)}")
    run(args)


if __name__ == "__main__":
    main()
