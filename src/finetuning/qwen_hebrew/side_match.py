# File name: side_match.py
# Date: 10/15/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Match PGP edition sides to library images by reading the images (v2.4a Arabic data).

``build_pgp_arabic_editions --emit-unassigned`` lists the documents it could not pair with images
for a *side* reason: an edition without side labels and several images, image labels that do not
name the edition's sides (figgy canvases labelled ``00000001.tif``), duplicated labels, ... This
script reads every canvas of those documents with Claude, compares each read with the edition on
Arabic letter 5-grams, and writes the side assignments the builder applies with
``--side-assignments`` (pages built that way carry ``side_source = "side_match:<reader>"``).

Files (the v2.4a spec; the builder implements the other end)::

    unassigned.jsonl        in: {pgpid, shelfmark, canonical_id, library, edition_fingerprint,
                            sections: {"recto"|"verso"|"": [lines]}, canvases: [{index, label, image_url}], reason}
    side_assignments.jsonl  out: {pgpid, edition_fingerprint, decision: "accept"|"drop", reader,
                            pages: [{canvas_index, image_url, assign: "all"|"recto"|"verso"}], evidence}
    <out>_review.csv        accepted pages with their scores and the local image, for a spot check

The edition fingerprint is passed through unchanged: the builder skips an assignment whose
fingerprint no longer matches its edition. ``--out`` is merged, not replaced: the builder re-emits
only documents without an assignment or with a stale one, so the rows of earlier runs are kept and
this run's decisions replace or add rows by ``pgpid``.

Reads: :data:`READER_PROMPT` (all Arabic-script text on the page, line by line, no commentary; worded
like the ``arabic`` prompt of ``run_arabic_benchmark``) on each canvas, downscaled to
:data:`MAX_SIDE` pixels on the long side. Every read is appended to a resumable JSONL cache keyed by
``(image_url, model, prompt hash)``; a re-run only reads what is missing (changing ``--effort``
does not invalidate cached reads). Transient API errors are retried with exponential backoff; a
document whose reads or image downloads failed stays pending (not written) and is picked up by the
next run. ``--max-usd`` is a hard cap: no call starts unless spent + the worst case of every call in
flight and of this one (largest image, :data:`MAX_TOKENS` of output) stays within it. Besides, no new
document starts once spent + reserved + its estimate (mean cost of the reads so far, or a prior
before the first read) would pass the budget.

Rules (on :func:`arabic_script.arabic_letters` of the read and of each edition section):

* more than :data:`MAX_CANVASES` canvases: drop (a whole-volume manifest); no text, no canvas: drop
  without reading;
* text outside the side labels next to a labelled side: left out (not read, no row), since the builder
  places ``all`` only on an edition without side labels and ``recto`` / ``verso`` only on an edition
  whose text is all under side labels; a final ``drop`` would keep it out for good;
* edition without side labels: canvas *i* gets ``all`` iff its 5-gram recall of the edition is at
  least :data:`MIN_RECALL` and :data:`RECALL_RATIO` times the best other canvas, no other canvas
  reads edition text (5-gram precision above :data:`OTHER_PRECISION_FLOOR`) and the read of *i* is
  not looping (``repeat_share`` below :data:`MAX_REPEAT_SHARE`). Otherwise drop; never split;
* edition with recto / verso sections: the assignment of sections to distinct canvases with the
  largest summed recall wins if it beats the second-best by :data:`MIN_MARGIN`, each section reaches
  :data:`MIN_RECALL` and no canvas outside the assignment reads edition text (precision above
  :data:`OTHER_PRECISION_FLOOR`); else drop. A canvas whose label names a side may only take that side.

``--calibrate`` runs the same rules on the frozen Arabic-script benchmark, on documents whose
images carry CUDL ``1r`` / ``1v`` labels and whose edition is labelled by side: the image labels
are hidden (and, for an edition of one side, the edition label too, which tests the unlabelled
rule) and the decision is compared with the labels. ``--calibrate-reads`` scores reads already on
disk (a benchmark outputs JSONL of ``{doc_id, image_index, text}``) instead of calling the API.

Usage::

    .venv/bin/python -m src.finetuning.qwen_hebrew.side_match --unassigned PATH --dry-run
    .venv/bin/python -m src.finetuning.qwen_hebrew.side_match --unassigned PATH [--out PATH] [--max-usd 20]
    .venv/bin/python -m src.finetuning.qwen_hebrew.side_match --calibrate [--limit 20] [--calibrate-reads FILE]
"""
import argparse
import asyncio
import base64
import csv
import hashlib
import http.client
import io
import json
import os
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

import anthropic
from PIL import Image as PILImage

from src.datasets.evaluations import arabic_script as ar
from src.datasets.evaluations.helper_eval_scripts import build_arabic_benchmark as bab
from src.datasets.evaluations.helper_eval_scripts.score_arabic_benchmark import DEFAULT_BENCHMARK, LOOP_REPEAT_SHARE
from src.models.llm.transcription.claude_transcriber import ClaudeConfig

REPO = Path(__file__).resolve().parents[3]
NAS_MOUNT = Path("/Volumes/home")
ROUND_DIR = REPO / "logs" / "next_round" / "arabic_v24a"
DEFAULT_OUT = ROUND_DIR / "side_match" / "side_assignments.jsonl"          # next to the builder's unassigned.jsonl (README)
DEFAULT_CALIBRATION_OUT = ROUND_DIR / "side_match" / "side_match_calibration.json"
DEFAULT_IMAGE_CACHE = bab.NAS_DATASETS / "pgp_arabic_editions_v2_images" / "side_match"
DEFAULT_READS_CACHE = DEFAULT_IMAGE_CACHE / "reads.jsonl"
CALIBRATION_LIMIT = 20

# ---- reader -------------------------------------------------------------------------------------------------
DEFAULT_MODEL = "claude-opus-5-5"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
DEFAULT_EFFORT = "low"          # a transcription is perception, not reasoning; thinking tokens bill as output
# Output budget per read, thinking included (streamed, so no HTTP timeout). One page's transcription at low effort is
# ~2-3k tokens; a read that runs to the cap is a decoder loop. The budget reserves this much output for every call in
# flight, so the cap also bounds the worst case of a call (~$0.18 at the Opus 5.5 prices) under --max-usd.
MAX_TOKENS = 8_000
MAX_SIDE = 2000                 # long side of the image sent to the reader, in pixels
DEFAULT_CONCURRENCY = 3
DEFAULT_MAX_USD = 5.0
MAX_ATTEMPTS = 5                # per read, for rate limits, overload, 5xx and connection errors
FATAL_API_ERRORS = (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError)   # halt the run
API_BASE_URL = "https://api.anthropic.com"   # pinned: the key from .env never goes to an inherited ANTHROPIC_BASE_URL
# A failed download or reader copy fails only its document: urllib errors and timeouts (OSError), a body cut short
# (http.client.IncompleteRead), a malformed URL (ValueError), an unreadable or oversized image (PIL).
FETCH_ERRORS = (OSError, ValueError, http.client.HTTPException, PILImage.DecompressionBombError)
BACKOFF_BASE_S = 2.0
BACKOFF_MAX_S = 60.0
DOWNLOAD_DELAY_S = 0.5          # pause after each library download (downloads run one at a time)
READER_PROMPT = """This is a manuscript page from the Cairo Genizah. It may contain Arabic script, Hebrew script, or both.

Transcribe all of the Arabic-script text visible in this image exactly as written, in Arabic script, including text in the margins.
Do not normalize or correct the text.
Preserve the line structure: one manuscript line per output line.
Mark damaged or unclear characters with [?].
If no Arabic-script text is visible, return only the word NONE.

Return ONLY the transcription with no commentary."""

# ---- prices: USD per million tokens. VERIFY against https://platform.claude.com/docs/en/about-claude/pricing
# before a large run (copied 2026-10-15 from the claude-api skill's model table, cached 2026-10-06; cache_write
# assumes the 5-minute rate of 1.25x input). Thinking tokens are billed as output tokens.
PRICES_PER_MTOK: Dict[str, Dict[str, float]] = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_write": 5.00, "cache_read": 0.20},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.20},
}
USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
PIXELS_PER_IMAGE_TOKEN = 750    # image tokens ~ width * height / 750 ...
MAX_IMAGE_TOKENS = 4784         # ... capped at the high-resolution limit
PRIOR_IMAGE_SIZE = (2000, 1500)  # cost prior before any read (and in --dry-run without cached reads)
PROMPT_TOKENS_PRIOR = 250
OUTPUT_TOKENS_PRIOR = 2500      # transcription plus thinking, per page

# ---- rules (validated on cached reads before this script; see the module docstring) --------------------------
NGRAM = 5                       # letter n-gram length of arabic_script.ngram_overlap
MAX_CANVASES = 8                # more canvases: a whole-volume manifest, the folio cannot be identified
MIN_RECALL = 0.05               # least share of a section's 5-grams found in the read of its canvas
RECALL_RATIO = 2.0              # unlabelled: the chosen canvas recalls at least this times the best other one
OTHER_PRECISION_FLOOR = 0.05    # unlabelled: another canvas whose read is this much edition text vetoes
MAX_REPEAT_SHARE = LOOP_REPEAT_SHARE   # a read at or above this repeat_share is a decoder loop (0.5)
MIN_MARGIN = 0.05               # labelled: best summed recall minus the second-best assignment's
THRESHOLDS = {"ngram": NGRAM, "max_canvases": MAX_CANVASES, "min_recall": MIN_RECALL, "recall_ratio": RECALL_RATIO,
              "other_precision_floor": OTHER_PRECISION_FLOOR, "max_repeat_share": MAX_REPEAT_SHARE, "min_margin": MIN_MARGIN}
SIDES = ("recto", "verso")
REVIEW_FIELDS = ["pgpid", "shelfmark", "canonical_id", "canvas_index", "canvas_label", "assign", "recall", "precision",
                 "repeat_share", "best_other_recall", "margin", "image_path", "image_url", "read_excerpt", "edition_excerpt"]

Reader = Callable[[Path], Awaitable[Dict[str, Any]]]
Download = Callable[[str, Path], Any]


def _log(message: str) -> None:
    """Print a progress line immediately.

    :param message: Text to print.
    :type message: str
    """
    print(message, flush=True)


def prompt_hash(prompt: str) -> str:
    """Short, stable hash of a reader prompt (part of the read cache key).

    :param prompt: Prompt text.
    :type prompt: str
    :return: First 12 hex digits of the prompt's SHA-1.
    :rtype: str
    """
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]


def text_fingerprint(text: str) -> str:
    """SHA-1 of a text (used for calibration records only; real records carry the builder's fingerprint).

    :param text: Any text.
    :type text: str
    :return: Hex digest.
    :rtype: str
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def prices_for(model: str) -> Dict[str, float]:
    """Per-million-token prices of a model.

    :param model: Model id.
    :type model: str
    :return: ``{input, output, cache_write, cache_read}`` in USD per million tokens.
    :rtype: Dict[str, float]
    :raises ValueError: When the model has no prices in :data:`PRICES_PER_MTOK` (the budget would be blind).
    """
    if model not in PRICES_PER_MTOK:
        raise ValueError(f"no prices for {model!r}; add them to PRICES_PER_MTOK")
    return PRICES_PER_MTOK[model]


def call_cost(usage: Dict[str, int], prices: Dict[str, float]) -> float:
    """Cost of one API call from its usage tokens.

    :param usage: Token counts (:data:`USAGE_FIELDS`; missing fields count as 0).
    :type usage: Dict[str, int]
    :param prices: :func:`prices_for` result.
    :type prices: Dict[str, float]
    :return: USD.
    :rtype: float
    """
    return (usage.get("input_tokens", 0) * prices["input"] + usage.get("output_tokens", 0) * prices["output"]
            + usage.get("cache_creation_input_tokens", 0) * prices["cache_write"]
            + usage.get("cache_read_input_tokens", 0) * prices["cache_read"]) / 1e6


def prior_image_cost(prices: Dict[str, float], size: Tuple[int, int] = PRIOR_IMAGE_SIZE) -> float:
    """Expected cost of reading one page before any read has been paid for.

    :param prices: :func:`prices_for` result.
    :type prices: Dict[str, float]
    :param size: Image size in pixels.
    :type size: Tuple[int, int]
    :return: USD.
    :rtype: float
    """
    image_tokens = min(MAX_IMAGE_TOKENS, size[0] * size[1] / PIXELS_PER_IMAGE_TOKEN)
    return call_cost({"input_tokens": round(image_tokens) + PROMPT_TOKENS_PRIOR, "output_tokens": OUTPUT_TOKENS_PRIOR}, prices)


def worst_read_cost(prices: Dict[str, float], max_tokens: int = MAX_TOKENS) -> float:
    """Most one read can cost: the largest image, the prompt and ``max_tokens`` of output (thinking included).

    :param prices: :func:`prices_for` result.
    :type prices: Dict[str, float]
    :param max_tokens: Output cap of the reader.
    :type max_tokens: int
    :return: USD.
    :rtype: float
    """
    return call_cost({"input_tokens": MAX_IMAGE_TOKENS + PROMPT_TOKENS_PRIOR, "output_tokens": max_tokens}, prices)


# =============================================================================================================
# Scoring and rules
# =============================================================================================================

def _labels(record: Dict[str, Any]) -> Set[str]:
    """Section labels of a record's edition that hold text.

    :param record: One ``unassigned.jsonl`` record.
    :type record: Dict[str, Any]
    :return: Labels (``"recto"``, ``"verso"``, ``""`` or anything else the record carries).
    :rtype: Set[str]
    """
    return {label for label, lines in record["sections"].items() if lines}


def precheck(record: Dict[str, Any]) -> str:
    """Why a document is dropped before any image is read (a final ``drop`` row).

    :param record: One ``unassigned.jsonl`` record.
    :type record: Dict[str, Any]
    :return: The reason, or ``""`` when the rules can decide the document.
    :rtype: str
    """
    canvases, labels = record["canvases"], _labels(record)
    if len(canvases) > MAX_CANVASES:
        return f"whole-volume manifest ({len(canvases)} canvases)"
    if not canvases:
        return "no canvas"
    if not labels:
        return "no edition text"
    if not labels <= {"", *SIDES}:
        return f"unknown section label {sorted(labels - {'', *SIDES})}"
    return ""


def unsupported(record: Dict[str, Any]) -> str:
    """Why the matcher leaves a document out of ``side_assignments.jsonl`` (neither read nor decided).

    An edition with text outside its side labels next to a labelled side cannot be placed by any assignment:
    the builder accepts ``all`` only on an edition without side labels (unlabelled text next to a labelled
    side is an implicit other side) and ``recto`` / ``verso`` only on an edition whose text is all under side
    labels. A ``drop`` row would keep the document out for good (the builder re-emits only missing or stale
    assignments); left out, it stays open for a later matcher or builder, and the run summary counts it.

    :param record: One ``unassigned.jsonl`` record that passed :func:`precheck`.
    :type record: Dict[str, Any]
    :return: The reason, or ``""`` when the matcher decides the document.
    :rtype: str
    """
    labels = _labels(record)
    if "" in labels and len(labels) > 1:
        return "edition has text outside its side labels (no assignment the builder places)"
    return ""


def needs_reading(record: Dict[str, Any]) -> bool:
    """Whether the canvases of a document are read (neither dropped by :func:`precheck` nor :func:`unsupported`).

    :param record: One ``unassigned.jsonl`` record.
    :type record: Dict[str, Any]
    :return: True when the rules need its reads.
    :rtype: bool
    """
    return not precheck(record) and not unsupported(record)


def score_canvases(sections: Dict[str, List[str]], canvas_indices: Sequence[int], reads: Sequence[str],
                   n: int = NGRAM) -> List[Dict[str, Any]]:
    """Compare each canvas's read with each edition section.

    :param sections: Edition lines by section label (``"recto"``, ``"verso"`` or ``""``).
    :type sections: Dict[str, List[str]]
    :param canvas_indices: Canvas ``index`` of each read.
    :type canvas_indices: Sequence[int]
    :param reads: Read text of each canvas.
    :type reads: Sequence[str]
    :param n: Letter n-gram length.
    :type n: int
    :return: Per canvas: ``canvas_index``, ``recall`` (by section label: share of the section's n-grams in
        the read), ``precision`` (share of the read's n-grams in the whole edition), ``repeat_share`` and
        ``read_letters``.
    :rtype: List[Dict[str, Any]]
    """
    section_letters = {label: ar.arabic_letters("\n".join(lines)) for label, lines in sections.items() if lines}
    edition = ar.arabic_letters("\n".join(line for lines in sections.values() for line in lines))
    scores = []
    for index, text in zip(canvas_indices, reads):
        letters = ar.arabic_letters(text)
        scores.append({"canvas_index": index,
                       "recall": {label: ar.ngram_overlap(letters, ref, n)["recall"] for label, ref in section_letters.items()},
                       "precision": ar.ngram_overlap(letters, edition, n)["precision"],
                       "repeat_share": ar.repeat_share(text), "read_letters": len(letters)})
    return scores


def decide_unlabelled(scores: List[Dict[str, Any]]) -> Tuple[List[Tuple[int, str]], str, Dict[str, Any]]:
    """Put a whole edition without side labels on one canvas, or on none.

    :param scores: :func:`score_canvases` result for an edition whose only section is ``""``.
    :type scores: List[Dict[str, Any]]
    :return: ``([(canvas_index, "all")] or [], drop reason or "", evidence)``.
    :rtype: Tuple[List[Tuple[int, str]], str, Dict[str, Any]]
    """
    ranked = sorted(scores, key=lambda score: -score["recall"][""])
    best = ranked[0]
    best_other = ranked[1]["recall"][""] if len(ranked) > 1 else 0.0
    evidence: Dict[str, Any] = {"best_canvas": best["canvas_index"], "best_recall": best["recall"][""], "best_other_recall": best_other}
    if best["recall"][""] < MIN_RECALL:
        return [], f"no canvas recalls {MIN_RECALL:.0%} of the edition", evidence
    if best["recall"][""] < RECALL_RATIO * best_other:
        return [], "ambiguous: another canvas recalls nearly as much of the edition", evidence
    spill = [score["canvas_index"] for score in scores if score is not best and score["precision"] > OTHER_PRECISION_FLOOR]
    if spill:
        evidence["other_canvases_with_edition_text"] = spill
        return [], "edition text also read on another canvas", evidence
    if best["repeat_share"] >= MAX_REPEAT_SHARE:
        return [], "the read of the best canvas is looping", evidence
    return [(best["canvas_index"], "all")], "", evidence


def decide_labelled(scores: List[Dict[str, Any]], sides: Sequence[str],
                    label_sides: Sequence[str]) -> Tuple[List[Tuple[int, str]], str, Dict[str, Any]]:
    """Assign recto / verso sections to distinct canvases jointly, or drop.

    Besides the margin and the recall floor, a canvas left out of the best assignment must not read edition
    text (precision above :data:`OTHER_PRECISION_FLOOR`): a side spread over two images would otherwise be put
    whole on the larger part, giving a page whose text is partly not in its image.

    :param scores: :func:`score_canvases` result.
    :type scores: List[Dict[str, Any]]
    :param sides: The edition's labelled sections, e.g. ``("recto", "verso")``.
    :type sides: Sequence[str]
    :param label_sides: Side named by each canvas's label (``""`` when it names none), aligned with ``scores``;
        a canvas whose label names a side may only take that side.
    :type label_sides: Sequence[str]
    :return: ``([(canvas_index, side)] or [], drop reason or "", evidence)``.
    :rtype: Tuple[List[Tuple[int, str]], str, Dict[str, Any]]
    """
    options = []
    for perm in permutations(range(len(scores)), len(sides)):
        if all(label_sides[c] in ("", side) for c, side in zip(perm, sides)):
            options.append((sum(scores[c]["recall"][side] for c, side in zip(perm, sides)), perm))
    if not options:
        return [], "fewer usable canvases than edition sides", {}
    options.sort(key=lambda option: -option[0])
    best_total, best = options[0]
    second_total = options[1][0] if len(options) > 1 else 0.0
    evidence = {"assignment": {side: scores[c]["canvas_index"] for c, side in zip(best, sides)}, "best_total": best_total,
                "second_total": second_total, "margin": best_total - second_total}
    if best_total - second_total < MIN_MARGIN:
        return [], "ambiguous: the best and second-best side assignments are within the margin", evidence
    weak = [side for c, side in zip(best, sides) if scores[c]["recall"][side] < MIN_RECALL]
    if weak:
        return [], f"{' and '.join(weak)} text not found on its canvas", evidence
    spill = [score["canvas_index"] for c, score in enumerate(scores) if c not in best and score["precision"] > OTHER_PRECISION_FLOOR]
    if spill:
        evidence["other_canvases_with_edition_text"] = spill
        return [], "edition text also read on a canvas outside the assignment", evidence
    return [(scores[c]["canvas_index"], side) for c, side in zip(best, sides)], "", evidence


def _rounded(value: Any) -> Any:
    """Round every float in a JSON-like value to 4 places.

    :param value: Number, string, list or dict.
    :type value: Any
    :return: The same structure with rounded floats.
    :rtype: Any
    """
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(item) for item in value]
    return value


def assignment_row(record: Dict[str, Any], reader: str, pages: List[Tuple[int, str]], evidence: Dict[str, Any]) -> Dict[str, Any]:
    """One ``side_assignments.jsonl`` object, exactly in the spec's shape.

    :param record: The ``unassigned.jsonl`` record.
    :type record: Dict[str, Any]
    :param reader: Reader id (the model id), recorded by the builder as ``side_match:<reader>``.
    :type reader: str
    :param pages: ``[(canvas_index, assign)]``; empty means ``drop``.
    :type pages: List[Tuple[int, str]]
    :param evidence: Scores and the reason.
    :type evidence: Dict[str, Any]
    :return: ``{pgpid, edition_fingerprint, decision, reader, pages, evidence}``.
    :rtype: Dict[str, Any]
    """
    url_of = {canvas["index"]: canvas["image_url"] for canvas in record["canvases"]}
    return {"pgpid": str(record["pgpid"]), "edition_fingerprint": record["edition_fingerprint"],
            "decision": "accept" if pages else "drop", "reader": reader,
            "pages": [{"canvas_index": index, "image_url": url_of[index], "assign": assign} for index, assign in pages],
            "evidence": _rounded(evidence)}


def decide_record(record: Dict[str, Any], read_texts: Sequence[Optional[str]], reader: str) -> Dict[str, Any]:
    """The side assignment of one document.

    :param record: One ``unassigned.jsonl`` record.
    :type record: Dict[str, Any]
    :param read_texts: Read of each canvas, aligned with ``record["canvases"]`` (``None``: the reader refused);
        ignored when :func:`precheck` drops the document.
    :type read_texts: Sequence[Optional[str]]
    :param reader: Reader id.
    :type reader: str
    :return: :func:`assignment_row` result.
    :rtype: Dict[str, Any]
    :raises ValueError: For a document :func:`unsupported` leaves out (it gets no row).
    """
    problem = precheck(record)
    if problem:
        return assignment_row(record, reader, [], {"rule": "precheck", "reason": problem})
    if unsupported(record):
        raise ValueError(f"{record['pgpid']}: {unsupported(record)}; the matcher writes no row for it")
    canvases = record["canvases"]
    refused = [canvas["index"] for canvas, text in zip(canvases, read_texts) if text is None]
    if refused:
        return assignment_row(record, reader, [], {"rule": "precheck", "reason": "a canvas read was refused", "refused": refused})
    sections = {label: lines for label, lines in record["sections"].items() if lines}
    scores = score_canvases(sections, [canvas["index"] for canvas in canvases], [str(text) for text in read_texts])
    sides = [side for side in SIDES if side in sections]
    if sides:
        rule = "labelled"
        pages, reason, evidence = decide_labelled(scores, sides, [bab.side_of_label(canvas["label"]) for canvas in canvases])
    else:
        rule = "unlabelled"
        pages, reason, evidence = decide_unlabelled(scores)
    return assignment_row(record, reader, pages, {"rule": rule, "reason": reason, **evidence, "canvases": scores,
                                                  "thresholds": THRESHOLDS})


# =============================================================================================================
# Reads: cache, budget, images, API
# =============================================================================================================

def ends_with_newline(path: Path) -> bool:
    """Whether a file is empty or its last byte is a newline (appending to it starts a new line).

    :param path: Existing file.
    :type path: Path
    :return: True when the next append starts on a line of its own.
    :rtype: bool
    """
    if path.stat().st_size == 0:
        return True
    with open(path, "rb") as fh:
        fh.seek(-1, os.SEEK_END)
        return fh.read(1) == b"\n"


class ReadCache:
    """Append-only JSONL of reads keyed by ``(image_url, model, prompt_hash)``; the last record of a key wins."""

    def __init__(self, path: Path) -> None:
        """Load the cache file if it exists (nothing is created until the first read).

        :param path: Cache file.
        :type path: Path
        """
        self.path = path
        self.reads: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._newline_first = False     # the file ends inside a line (a killed run): start the next read on a new line
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:          # a line cut short by a killed run; that read is redone
                        continue
                    self.reads[(row["image_url"], row["model"], row["prompt_hash"])] = row
            self._newline_first = not ends_with_newline(path)

    def get(self, image_url: str, model: str, prompt_hash_: str) -> Optional[Dict[str, Any]]:
        """A cached read.

        :param image_url: Canvas image URL.
        :type image_url: str
        :param model: Model id.
        :type model: str
        :param prompt_hash_: :func:`prompt_hash` of the prompt.
        :type prompt_hash_: str
        :return: The read record, or None.
        :rtype: Optional[Dict[str, Any]]
        """
        return self.reads.get((image_url, model, prompt_hash_))

    def add(self, row: Dict[str, Any]) -> None:
        """Store a read and append it to the file at once.

        :param row: Read record with ``image_url``, ``model`` and ``prompt_hash``.
        :type row: Dict[str, Any]
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(("\n" if self._newline_first else "") + json.dumps(row, ensure_ascii=False) + "\n")
        self._newline_first = False
        self.reads[(row["image_url"], row["model"], row["prompt_hash"])] = row

    def costs(self, model: str) -> List[float]:
        """Costs of the cached reads of a model.

        :param model: Model id.
        :type model: str
        :return: USD per read.
        :rtype: List[float]
        """
        return [row["cost_usd"] for (_, row_model, _), row in self.reads.items() if row_model == model and "cost_usd" in row]


@dataclass
class Budget:
    """Running cost of the reads against ``max_usd``.

    Two gates. A document reserves its estimated cost before its first call (:meth:`reserve`), so no document
    starts that the budget clearly cannot finish. Every call then holds the worst case of one read from start
    to finish (:meth:`start_call` / :meth:`end_call`), and a call starts only while spent + the worst case of
    every call in flight and of itself stays within ``max_usd``: spending never passes the cap, whatever a
    read actually costs up to :data:`MAX_TOKENS` of output.
    """

    max_usd: float
    prices: Dict[str, float]
    max_tokens: int = MAX_TOKENS
    spent: float = 0.0
    reserved: float = 0.0
    calls: int = 0
    in_flight: int = 0
    stopped: bool = False
    tokens: Counter = field(default_factory=Counter)

    def worst_read(self) -> float:
        """Most one call can cost.

        :return: :func:`worst_read_cost` at this budget's prices and output cap.
        :rtype: float
        """
        return worst_read_cost(self.prices, self.max_tokens)

    def start_call(self) -> bool:
        """Claim the worst case of one call, or refuse it (and stop the run from starting new documents).

        :return: True when the call may start; it must be followed by :meth:`end_call`.
        :rtype: bool
        """
        if self.spent + (self.in_flight + 1) * self.worst_read() > self.max_usd:
            self.stopped = True
            return False
        self.in_flight += 1
        return True

    def end_call(self) -> None:
        """Release the claim of a call that finished or failed (:meth:`record` has billed a finished one)."""
        self.in_flight = max(0, self.in_flight - 1)

    def per_image(self) -> float:
        """Estimated cost of the next read.

        :return: Mean cost of the reads so far, or :func:`prior_image_cost` before the first.
        :rtype: float
        """
        return self.spent / self.calls if self.calls else prior_image_cost(self.prices)

    def reserve(self, images: int) -> Optional[float]:
        """Reserve the estimate of a document's reads, or stop the run.

        :param images: Reads the document needs.
        :type images: int
        :return: The amount reserved, or None when it would pass the budget (no new document starts after that).
        :rtype: Optional[float]
        """
        amount = images * self.per_image()
        if self.stopped or self.spent + self.reserved + amount > self.max_usd:
            self.stopped = True
            return None
        self.reserved += amount
        return amount

    def release(self, amount: float) -> None:
        """Return a reservation once its document is done.

        :param amount: :meth:`reserve` result.
        :type amount: float
        """
        self.reserved = max(0.0, self.reserved - amount)

    def record(self, usage: Dict[str, int]) -> float:
        """Account for one finished call.

        :param usage: Its token counts.
        :type usage: Dict[str, int]
        :return: Its cost in USD.
        :rtype: float
        """
        cost = call_cost(usage, self.prices)
        self.spent += cost
        self.calls += 1
        self.tokens.update({key: int(usage.get(key, 0)) for key in USAGE_FIELDS})
        return cost


def prepare_image(src: Path, dst: Path, max_side: int = MAX_SIDE, max_bytes: int = ClaudeConfig.MAX_IMAGE_BYTES) -> Tuple[int, int]:
    """Write the reader's copy of an image: RGB JPEG, at most ``max_side`` pixels on the long side.

    :param src: Downloaded image.
    :type src: Path
    :param dst: Destination (kept when it already exists).
    :type dst: Path
    :param max_side: Long-side cap in pixels.
    :type max_side: int
    :param max_bytes: Size cap of the encoded image (JPEG quality is lowered to fit).
    :type max_bytes: int
    :return: ``(width, height)`` of the copy.
    :rtype: Tuple[int, int]
    """
    if dst.exists() and dst.stat().st_size > 0:
        with PILImage.open(dst) as done:
            return done.size
    with PILImage.open(src) as original:
        image = original.convert("RGB")
    if max(image.size) > max_side:
        ratio = max_side / max(image.size)
        image = image.resize((max(1, round(image.width * ratio)), max(1, round(image.height * ratio))), PILImage.Resampling.LANCZOS)
    for quality in (90, 80, 70, 60):
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        if buf.tell() <= max_bytes:
            break
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    tmp.write_bytes(buf.getvalue())
    os.replace(tmp, dst)
    return image.size


class ImageUnavailable(OSError):
    """A canvas could not be downloaded or prepared; only its document fails (it stays pending)."""


class ImageFetcher:
    """Downloads canvases one at a time (polite to library servers) and prepares the reader's copies."""

    def __init__(self, cache_dir: Path, max_side: int = MAX_SIDE, delay_s: float = DOWNLOAD_DELAY_S,
                 download: Download = bab.save_image) -> None:
        """Configure the fetcher.

        :param cache_dir: Where downloads (``<url sha1>.jpg``) and reader copies (``<url sha1>_<max_side>.jpg``) go.
        :type cache_dir: Path
        :param max_side: Long-side cap of the reader copies.
        :type max_side: int
        :param delay_s: Pause after each download.
        :type delay_s: float
        :param download: ``(url, destination) -> Any`` (default: the benchmark builder's ``save_image``).
        :type download: Download
        """
        self.cache_dir, self.max_side, self.delay_s, self.download = cache_dir, max_side, delay_s, download
        self._lock: Optional[asyncio.Lock] = None

    def stem(self, canvas: Dict[str, Any]) -> str:
        """File stem of a canvas in the cache.

        :param canvas: Canvas with ``image_url``.
        :type canvas: Dict[str, Any]
        :return: First 20 hex digits of the URL's SHA-1.
        :rtype: str
        """
        return hashlib.sha1(canvas["image_url"].encode("utf-8")).hexdigest()[:20]

    def prepared_path(self, canvas: Dict[str, Any]) -> Path:
        """Path of the reader copy of a canvas.

        :param canvas: Canvas with ``image_url``.
        :type canvas: Dict[str, Any]
        :return: ``<cache_dir>/<stem>_<max_side>.jpg``.
        :rtype: Path
        """
        return self.cache_dir / f"{self.stem(canvas)}_{self.max_side}.jpg"

    async def fetch(self, canvas: Dict[str, Any]) -> Tuple[Path, Tuple[int, int]]:
        """Download a canvas (unless cached, or given as an existing ``local_path``) and prepare its reader copy.

        ``local_path`` is only read, never written: when that file is missing (e.g. a frozen benchmark's
        ``images/``), the image is downloaded into the cache instead.

        :param canvas: Canvas with ``image_url`` and optionally ``local_path`` (an image already on disk).
        :type canvas: Dict[str, Any]
        :return: ``(reader copy, (width, height))``.
        :rtype: Tuple[Path, Tuple[int, int]]
        :raises ImageUnavailable: When the download or the reader copy fails (any of :data:`FETCH_ERRORS`).
        """
        try:
            return await self._fetch(canvas)
        except FETCH_ERRORS as error:
            raise ImageUnavailable(f"{type(error).__name__}: {error}") from error

    async def _fetch(self, canvas: Dict[str, Any]) -> Tuple[Path, Tuple[int, int]]:
        """:meth:`fetch` without the error translation.

        :param canvas: Canvas with ``image_url`` and optionally ``local_path``.
        :type canvas: Dict[str, Any]
        :return: ``(reader copy, (width, height))``.
        :rtype: Tuple[Path, Tuple[int, int]]
        """
        prepared = self.prepared_path(canvas)
        if not (prepared.exists() and prepared.stat().st_size > 0):
            local = Path(canvas["local_path"]) if canvas.get("local_path") else None
            raw = local if local is not None and local.exists() else self.cache_dir / f"{self.stem(canvas)}.jpg"
            if not raw.exists():
                if self._lock is None:
                    self._lock = asyncio.Lock()
                async with self._lock:
                    raw.parent.mkdir(parents=True, exist_ok=True)
                    await asyncio.to_thread(self.download, canvas["image_url"], raw)
                    await asyncio.sleep(self.delay_s)
            return prepared, await asyncio.to_thread(prepare_image, raw, prepared, self.max_side)
        return prepared, await asyncio.to_thread(prepare_image, prepared, prepared, self.max_side)


def make_client(env_file: Optional[Path] = REPO / ".env", log: Callable[[str], None] = _log) -> anthropic.AsyncAnthropic:
    """Async Anthropic client with the repo's ``.env`` loaded into the environment (the key is never printed).

    The endpoint is pinned to :data:`API_BASE_URL`: an ``ANTHROPIC_BASE_URL`` inherited from the launching
    process (a gateway or proxy) would otherwise receive the key. Such a variable is ignored, and its host
    (never the key) is logged.

    :param env_file: ``.env`` to load without overriding the environment (None: load nothing; tests).
    :type env_file: Optional[Path]
    :param log: Progress printer.
    :type log: Callable[[str], None]
    :return: Client with SDK retries off (:func:`read_with_retries` owns retrying).
    :rtype: anthropic.AsyncAnthropic
    """
    if env_file is not None:
        from dotenv import load_dotenv      # only for real runs: tests never touch .env

        load_dotenv(env_file)
    inherited = urlparse(os.environ.get("ANTHROPIC_BASE_URL") or "").hostname
    if inherited and inherited != urlparse(API_BASE_URL).hostname:
        log(f"ignoring ANTHROPIC_BASE_URL (host {inherited}); requests go to {API_BASE_URL}")
    return anthropic.AsyncAnthropic(base_url=API_BASE_URL, timeout=ClaudeConfig.TIMEOUT_S, max_retries=0)


def claude_reader(client: anthropic.AsyncAnthropic, model: str, prompt: str, effort: str, max_tokens: int = MAX_TOKENS) -> Reader:
    """A reader that sends one image and the prompt to Claude.

    Thinking is left at the model default (adaptive on Claude Opus 5.5, which cannot disable it); ``effort`` is
    sent through ``extra_body`` because the installed SDK predates the ``output_config`` argument. No
    server-side fallback model is configured, so every read is by ``model`` (the reader recorded in
    ``side_source``); a refusal is cached as a read without text and drops its document.

    :param client: :func:`make_client` result.
    :type client: anthropic.AsyncAnthropic
    :param model: Model id.
    :type model: str
    :param prompt: Instruction sent after the image.
    :type prompt: str
    :param effort: ``output_config.effort`` level.
    :type effort: str
    :param max_tokens: Output budget, thinking included.
    :type max_tokens: int
    :return: ``async (image_path) -> {text, stop_reason, usage, served_model}``; ``text`` is None on a refusal.
    :rtype: Reader
    """
    async def read(image_path: Path) -> Dict[str, Any]:
        """Read one image.

        :param image_path: JPEG to send.
        :type image_path: Path
        :return: ``{text, stop_reason, usage, served_model}``.
        :rtype: Dict[str, Any]
        """
        data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        async with client.messages.stream(
                model=model, max_tokens=max_tokens, extra_body={"output_config": {"effort": effort}},
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
                    {"type": "text", "text": prompt}]}]) as stream:
            response = await stream.get_final_message()
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        usage = {key: int(getattr(response.usage, key, 0) or 0) for key in USAGE_FIELDS}
        return {"text": None if response.stop_reason == "refusal" else text, "stop_reason": response.stop_reason,
                "usage": usage, "served_model": response.model}

    return read


def is_retryable(error: BaseException) -> bool:
    """Whether an API error is worth another attempt.

    :param error: The exception.
    :type error: BaseException
    :return: True for rate limits, connection errors and timeouts, overload and other 5xx answers.
    :rtype: bool
    """
    if isinstance(error, (anthropic.RateLimitError, anthropic.APIConnectionError)):
        return True
    return isinstance(error, anthropic.APIStatusError) and error.status_code >= 500


async def read_with_retries(read: Reader, image_path: Path, attempts: int = MAX_ATTEMPTS, base_delay_s: float = BACKOFF_BASE_S,
                            retryable: Callable[[BaseException], bool] = is_retryable,
                            sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
                            log: Callable[[str], None] = _log) -> Dict[str, Any]:
    """Call a reader, retrying transient failures with exponential backoff and jitter.

    :param read: The reader.
    :type read: Reader
    :param image_path: Image to read.
    :type image_path: Path
    :param attempts: Attempts in all.
    :type attempts: int
    :param base_delay_s: Delay before the second attempt; doubles each time (capped at :data:`BACKOFF_MAX_S`).
    :type base_delay_s: float
    :param retryable: Which errors are retried.
    :type retryable: Callable[[BaseException], bool]
    :param sleep: Awaitable sleep (injectable for tests).
    :type sleep: Callable[[float], Awaitable[Any]]
    :param log: Progress printer.
    :type log: Callable[[str], None]
    :return: The reader's result.
    :rtype: Dict[str, Any]
    :raises Exception: The last error when it is not retryable or the attempts run out.
    """
    for attempt in range(1, attempts + 1):
        try:
            return await read(image_path)
        except Exception as error:      # re-raised unless it is a transient API error
            if attempt == attempts or not retryable(error):
                raise
            delay = min(BACKOFF_MAX_S, base_delay_s * 2 ** (attempt - 1)) * (1 + random.random() / 2)
            log(f"    retry {attempt}/{attempts - 1} in {delay:.1f}s after {type(error).__name__}")
            await sleep(delay)
    raise RuntimeError("unreachable")


async def read_documents(records: Sequence[Dict[str, Any]], cache: ReadCache, fetcher: ImageFetcher, read: Reader, model: str,
                         prompt: str, effort: str, budget: Budget, concurrency: int = DEFAULT_CONCURRENCY,
                         log: Callable[[str], None] = _log,
                         sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep) -> Counter:
    """Read every canvas of the documents that still lacks a cached read, document by document.

    ``concurrency`` workers each take the next document, reserve its estimated cost and read its canvases;
    a document that would pass the budget stops the run from starting new documents. Every call is also
    gated by :meth:`Budget.start_call` (the worst case of the calls in flight and of this one must fit), so
    a document already started stops at the first call that could pass ``max_usd`` and is left pending (its
    finished reads stay cached). A failed download or image (:class:`ImageUnavailable`) or API error fails
    only its document (left pending); an authentication, permission or unknown-model error
    (:data:`FATAL_API_ERRORS`) halts the run, since every further call would fail the same way.

    :param records: ``unassigned.jsonl`` records (only those :func:`needs_reading` are read).
    :type records: Sequence[Dict[str, Any]]
    :param cache: Read cache.
    :type cache: ReadCache
    :param fetcher: Image fetcher.
    :type fetcher: ImageFetcher
    :param read: The reader.
    :type read: Reader
    :param model: Model id (cache key).
    :type model: str
    :param prompt: Reader prompt (its hash is part of the cache key).
    :type prompt: str
    :param effort: Effort level (stored with each read).
    :type effort: str
    :param budget: Budget.
    :type budget: Budget
    :param concurrency: Documents read at once.
    :type concurrency: int
    :param log: Progress printer.
    :type log: Callable[[str], None]
    :param sleep: Awaitable sleep for the backoff (injectable for tests).
    :type sleep: Callable[[float], Awaitable[Any]]
    :return: Counts: ``documents_read``, ``documents_failed``, ``documents_left_for_budget``, ``reads``, ``halted``.
    :rtype: Counter
    """
    hashed = prompt_hash(prompt)
    stats: Counter = Counter()
    halted: List[str] = []
    key_locks: Dict[str, asyncio.Lock] = {}
    todo = iter([record for record in records if needs_reading(record)])

    async def read_canvas(record: Dict[str, Any], canvas: Dict[str, Any]) -> bool:
        """Read one canvas unless another document already did, if the budget allows the call.

        :param record: Its document.
        :type record: Dict[str, Any]
        :param canvas: The canvas.
        :type canvas: Dict[str, Any]
        :return: False when the budget refused the call (nothing was downloaded or read), True otherwise.
        :rtype: bool
        """
        async with key_locks.setdefault(canvas["image_url"], asyncio.Lock()):
            if cache.get(canvas["image_url"], model, hashed) is not None:
                return True
            if not budget.start_call():
                return False
            try:
                path, size = await fetcher.fetch(canvas)
                started = time.time()
                result = await read_with_retries(read, path, sleep=sleep, log=log)
                cost = budget.record(result["usage"])
            finally:
                budget.end_call()
            cache.add({"image_url": canvas["image_url"], "model": model, "prompt_hash": hashed, "effort": effort,
                       "text": result["text"], "stop_reason": result["stop_reason"], "usage": result["usage"],
                       "cost_usd": round(cost, 6), "served_model": result.get("served_model", model), "image_size": list(size),
                       "secs": round(time.time() - started, 1), "read_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            stats["reads"] += 1
            flag = "" if result["stop_reason"] == "end_turn" else f" [{result['stop_reason']}]"
            log(f"  {record['pgpid']} canvas {canvas['index']}: {result['usage']['input_tokens']} in / "
                f"{result['usage']['output_tokens']} out tokens, ${cost:.4f}{flag}; spent ${budget.spent:.2f} of ${budget.max_usd:.2f}")
            return True

    async def worker() -> None:
        """Take documents until none is left, the budget stops the run or a fatal API error halts it."""
        for record in todo:
            if halted:
                return
            missing = [canvas for canvas in record["canvases"] if cache.get(canvas["image_url"], model, hashed) is None]
            if not missing:
                continue
            amount = budget.reserve(len(missing))
            if amount is None:
                stats["documents_left_for_budget"] += 1
                continue
            try:
                for canvas in missing:
                    if not await read_canvas(record, canvas):
                        stats["documents_left_for_budget"] += 1
                        log(f"  {record['pgpid']}: the next call could pass ${budget.max_usd:.2f} "
                            f"(spent ${budget.spent:.2f}, worst case ${budget.worst_read():.2f} per call); left pending")
                        break
                else:
                    stats["documents_read"] += 1
            except FATAL_API_ERRORS as error:                 # bad key, no access, unknown model: no call can succeed
                halted.append(type(error).__name__)
                stats["documents_failed"] += 1
                stats["halted"] = 1
                log(f"  {record['pgpid']}: {type(error).__name__} ({str(error)[:160]}); halting the run "
                    "(check ANTHROPIC_API_KEY in .env and the model id)")
            except (OSError, anthropic.APIError) as error:     # download refused, bad image (ImageUnavailable), API error: stays pending
                stats["documents_failed"] += 1
                log(f"  {record['pgpid']}: read failed ({type(error).__name__}: {str(error)[:160]}); left pending")
            finally:
                budget.release(amount)

    await asyncio.gather(*(worker() for _ in range(max(1, concurrency))))
    return stats


# =============================================================================================================
# Jobs
# =============================================================================================================

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Objects of a JSONL file (blank lines skipped).

    :param path: File.
    :type path: Path
    :return: Objects in file order.
    :rtype: List[Dict[str, Any]]
    """
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_unassigned(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """The builder's ``unassigned.jsonl``.

    :param path: File.
    :type path: Path
    :param limit: Only the first N documents.
    :type limit: Optional[int]
    :return: Records in file order.
    :rtype: List[Dict[str, Any]]
    """
    records = read_jsonl(path)
    return records[:limit] if limit else records


def merge_assignments(previous: Sequence[Dict[str, Any]], decided: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """An earlier ``side_assignments.jsonl`` with this run's decisions replacing or added to its rows.

    The builder re-emits only documents without an assignment or with a stale one, so a run sees only part of
    the documents; rewriting ``--out`` from this run alone would erase every earlier decision. Rows whose
    fingerprint went stale are kept: the builder ignores them (and re-emits their documents).

    :param previous: Rows of the existing output (empty when there is none).
    :type previous: Sequence[Dict[str, Any]]
    :param decided: Rows decided in this run.
    :type decided: Sequence[Dict[str, Any]]
    :return: One row per ``pgpid``: earlier rows in their order (replaced in place when decided again), then new ones.
    :rtype: List[Dict[str, Any]]
    """
    merged = {str(row["pgpid"]): row for row in previous}
    for row in decided:
        merged[str(row["pgpid"])] = row
    return list(merged.values())


def read_texts_of(record: Dict[str, Any], cache: ReadCache, model: str, prompt: str) -> Optional[List[Optional[str]]]:
    """Cached reads of every canvas of a document.

    :param record: ``unassigned.jsonl`` record.
    :type record: Dict[str, Any]
    :param cache: Read cache.
    :type cache: ReadCache
    :param model: Model id.
    :type model: str
    :param prompt: Reader prompt.
    :type prompt: str
    :return: Texts aligned with the canvases (None for a refused read), or None when a read is missing.
    :rtype: Optional[List[Optional[str]]]
    """
    rows = [cache.get(canvas["image_url"], model, prompt_hash(prompt)) for canvas in record["canvases"]]
    return None if any(row is None for row in rows) else [row["text"] for row in rows]


def plan(records: Sequence[Dict[str, Any]], cache: ReadCache, model: str, prompt: str) -> Dict[str, Any]:
    """What a run would read and cost (``--dry-run``: no download, no API call).

    :param records: ``unassigned.jsonl`` records.
    :type records: Sequence[Dict[str, Any]]
    :param cache: Read cache.
    :type cache: ReadCache
    :param model: Model id.
    :type model: str
    :param prompt: Reader prompt.
    :type prompt: str
    :return: Counts and the cost estimate (per image: mean of the model's cached reads, else the prior).
    :rtype: Dict[str, Any]
    """
    dropped = Counter(precheck(record) for record in records if precheck(record))
    left_out = Counter(unsupported(record) for record in records if not precheck(record) and unsupported(record))
    readable = [record for record in records if needs_reading(record)]
    urls = {canvas["image_url"] for record in readable for canvas in record["canvases"]}
    cached = {url for url in urls if cache.get(url, model, prompt_hash(prompt)) is not None}
    costs = cache.costs(model)
    per_image = sum(costs) / len(costs) if costs else prior_image_cost(prices_for(model))
    return {"documents": len(records), "dropped_before_reading": dict(dropped), "left_out_unsupported": dict(left_out),
            "documents_to_read": len(readable),
            "images": len(urls), "images_cached": len(cached), "images_to_read": len(urls - cached),
            "estimated_usd_per_image": round(per_image, 4), "estimated_usd": round(per_image * len(urls - cached), 2),
            "estimate_basis": f"mean of {len(costs)} cached reads" if costs else "prior (no cached read of this model)"}


def review_lines(record: Dict[str, Any], row: Dict[str, Any], read_texts: Sequence[Optional[str]],
                 image_paths: Dict[int, str]) -> List[Dict[str, Any]]:
    """Review-CSV lines of an accepted document, one per assigned page.

    :param record: ``unassigned.jsonl`` record.
    :type record: Dict[str, Any]
    :param row: Its :func:`decide_record` result.
    :type row: Dict[str, Any]
    :param read_texts: Reads aligned with the canvases.
    :type read_texts: Sequence[Optional[str]]
    :param image_paths: Local image of each canvas index.
    :type image_paths: Dict[int, str]
    :return: Dicts with :data:`REVIEW_FIELDS`.
    :rtype: List[Dict[str, Any]]
    """
    scores = {score["canvas_index"]: score for score in row["evidence"].get("canvases", [])}
    canvases = {canvas["index"]: canvas for canvas in record["canvases"]}
    texts = {canvas["index"]: text or "" for canvas, text in zip(record["canvases"], read_texts)}
    lines = []
    for page in row["pages"]:
        index, section = page["canvas_index"], "" if page["assign"] == "all" else page["assign"]
        lines.append({"pgpid": record["pgpid"], "shelfmark": record.get("shelfmark", ""), "canonical_id": record.get("canonical_id", ""),
                      "canvas_index": index, "canvas_label": canvases[index]["label"], "assign": page["assign"],
                      "recall": scores[index]["recall"][section], "precision": scores[index]["precision"],
                      "repeat_share": scores[index]["repeat_share"], "best_other_recall": row["evidence"].get("best_other_recall", ""),
                      "margin": row["evidence"].get("margin", ""), "image_path": image_paths.get(index, ""), "image_url": page["image_url"],
                      "read_excerpt": " / ".join(texts[index].splitlines())[:160],
                      "edition_excerpt": " / ".join(record["sections"][section])[:160]})
    return lines


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write a JSONL file atomically.

    :param path: Destination.
    :type path: Path
    :param rows: Objects.
    :type rows: Iterable[Dict[str, Any]]
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    """Write a CSV file atomically.

    :param path: Destination.
    :type path: Path
    :param rows: Dicts.
    :type rows: Sequence[Dict[str, Any]]
    :param fields: Column order.
    :type fields: Sequence[str]
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def review_path(out: Path) -> Path:
    """Default review-CSV path next to an output file.

    :param out: Output file.
    :type out: Path
    :return: ``<out stem>_review.csv``.
    :rtype: Path
    """
    return out.with_name(out.stem + "_review.csv")


def _image_paths(record: Dict[str, Any], fetcher: ImageFetcher) -> Dict[int, str]:
    """Local image of each canvas (the reader copy when it exists, else ``local_path``).

    :param record: Record.
    :type record: Dict[str, Any]
    :param fetcher: Fetcher (for the cache layout).
    :type fetcher: ImageFetcher
    :return: ``{canvas index: path}``.
    :rtype: Dict[int, str]
    """
    paths = {}
    for canvas in record["canvases"]:
        prepared = fetcher.prepared_path(canvas)
        paths[canvas["index"]] = str(prepared) if prepared.exists() else canvas.get("local_path", "")
    return paths


def run_side_match(unassigned: Path, out: Path, reads_cache: Path = DEFAULT_READS_CACHE, image_cache: Path = DEFAULT_IMAGE_CACHE,
                   model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT, concurrency: int = DEFAULT_CONCURRENCY,
                   max_usd: float = DEFAULT_MAX_USD, limit: Optional[int] = None, dry_run: bool = False, max_side: int = MAX_SIDE,
                   review_csv: Optional[Path] = None, read: Optional[Reader] = None, download: Download = bab.save_image,
                   delay_s: float = DOWNLOAD_DELAY_S, log: Callable[[str], None] = _log,
                   sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep) -> Dict[str, Any]:
    """Read the unassigned documents and write their side assignments.

    :param unassigned: The builder's ``unassigned.jsonl``.
    :type unassigned: Path
    :param out: ``side_assignments.jsonl`` to write, merged with what it already holds (:func:`merge_assignments`):
        this run's decisions replace or add rows by ``pgpid``; pending and unsupported documents get no new row.
    :type out: Path
    :param reads_cache: Read cache file.
    :type reads_cache: Path
    :param image_cache: Image cache directory.
    :type image_cache: Path
    :param model: Model id (also the ``reader`` of the assignments).
    :type model: str
    :param effort: Effort level.
    :type effort: str
    :param concurrency: Documents read at once.
    :type concurrency: int
    :param max_usd: Budget for this run's API calls.
    :type max_usd: float
    :param limit: Only the first N documents.
    :type limit: Optional[int]
    :param dry_run: Only report what would be read and its estimated cost.
    :type dry_run: bool
    :param max_side: Long-side cap of the images sent.
    :type max_side: int
    :param review_csv: Review CSV of accepted pages (default: next to ``out``).
    :type review_csv: Optional[Path]
    :param read: Reader (default: Claude through :func:`claude_reader`; injectable for tests).
    :type read: Optional[Reader]
    :param download: Image downloader (injectable for tests).
    :type download: Download
    :param delay_s: Pause after each download.
    :type delay_s: float
    :param log: Progress printer.
    :type log: Callable[[str], None]
    :param sleep: Awaitable sleep for the backoff.
    :type sleep: Callable[[float], Awaitable[Any]]
    :return: Run summary.
    :rtype: Dict[str, Any]
    """
    records = load_unassigned(unassigned, limit)
    prices = prices_for(model)
    cache = ReadCache(reads_cache)
    if dry_run:
        return {"dry_run": True, "model": model, **plan(records, cache, model, READER_PROMPT)}
    fetcher = ImageFetcher(image_cache, max_side, delay_s, download)
    budget = Budget(max_usd, prices, MAX_TOKENS)
    read = read or claude_reader(make_client(log=log), model, READER_PROMPT, effort, MAX_TOKENS)
    started = time.time()
    stats = asyncio.run(read_documents(records, cache, fetcher, read, model, READER_PROMPT, effort, budget, concurrency, log, sleep))
    rows, review, pending, left_out = [], [], [], Counter()
    for record in records:
        if not precheck(record) and unsupported(record):
            left_out[unsupported(record)] += 1
            continue
        texts = [] if precheck(record) else read_texts_of(record, cache, model, READER_PROMPT)
        if texts is None:
            pending.append(str(record["pgpid"]))
            continue
        row = decide_record(record, texts, model)
        rows.append(row)
        if row["decision"] == "accept":
            review += review_lines(record, row, texts, _image_paths(record, fetcher))
    previous = read_jsonl(out) if out.exists() else []
    merged = merge_assignments(previous, rows)
    write_jsonl(out, merged)
    write_csv(review_csv or review_path(out), review, REVIEW_FIELDS)
    return {"model": model, "effort": effort, "documents": len(records), "decided": len(rows), "pending": len(pending),
            "left_out_unsupported": dict(left_out),
            "accepted": sum(row["decision"] == "accept" for row in rows), "pages": sum(len(row["pages"]) for row in rows),
            "assign": dict(Counter(page["assign"] for row in rows for page in row["pages"])),
            "dropped": dict(Counter(row["evidence"]["reason"] for row in rows if row["decision"] == "drop")),
            "rows_in_out": len(merged), "rows_kept_from_earlier_runs": len(merged) - len(rows),
            "reads": dict(stats), "tokens": dict(budget.tokens), "spent_usd": round(budget.spent, 4), "max_usd": max_usd,
            "worst_case_usd_per_call": round(budget.worst_read(), 4),
            "budget_stopped": budget.stopped, "halted": bool(stats["halted"]), "minutes": round((time.time() - started) / 60, 1), "out": str(out),
            "review_csv": str(review_csv or review_path(out)), "pending_pgpids": pending[:50]}


# =============================================================================================================
# Calibration on the Arabic-script benchmark
# =============================================================================================================

def calibration_cases(benchmark: Path, limit: int = CALIBRATION_LIMIT) -> List[Dict[str, Any]]:
    """Benchmark documents with known sides, turned into unassigned-style records with the labels hidden.

    Eligible: 2 to :data:`MAX_CANVASES` images, each labelled with a distinct side (CUDL ``1r`` / ``1v``), and an
    edition whose text is all inside side labels, on sides the images have. Each gives a ``labelled`` case
    (canvas labels hidden); an edition of one side also gives an ``unlabelled`` case (its label hidden too).

    :param benchmark: Benchmark directory (``benchmark.jsonl``, ``images_manifest.json``, ``images/``).
    :type benchmark: Path
    :param limit: Documents to take, in benchmark order.
    :type limit: int
    :return: ``[{mode, doc_id, record, truth: [(canvas_index, assign)], true_side: {canvas_index: side}}]``.
    :rtype: List[Dict[str, Any]]
    """
    with open(benchmark / "images_manifest.json", encoding="utf-8") as fh:
        url_of = {(entry["doc_id"], int(entry["image_index"])): entry.get("image_url", "") for entry in json.load(fh)["images"]}
    cases: List[Dict[str, Any]] = []
    docs = 0
    with open(benchmark / "benchmark.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if docs >= limit:
                break
            doc = json.loads(line)
            sides = [image["side"] for image in doc["images"]]
            sections: Dict[str, List[str]] = {}
            for section in doc["gt"]["sections"]:
                sections.setdefault(section["side"], []).extend(section["lines"])
            if (not 2 <= len(sides) <= MAX_CANVASES or "" in sides or len(set(sides)) != len(sides)
                    or not sections or "" in sections or not set(sections) <= set(sides)):
                continue
            docs += 1
            canvases = [{"index": int(image["image_index"]), "label": "",
                         "image_url": url_of.get((doc["id"], int(image["image_index"]))) or f"file://{benchmark / 'images' / image['file']}",
                         "local_path": str(benchmark / "images" / image["file"])} for image in doc["images"]]
            true_side = {int(image["image_index"]): image["side"] for image in doc["images"]}
            base = {"pgpid": str(doc["pgpids"][0]), "shelfmark": doc["shelfmark"], "canonical_id": doc["id"], "library": doc["library"],
                    "canvases": canvases, "reason": "calibration: canvas labels hidden"}
            cases.append({"mode": "labelled", "doc_id": doc["id"], "true_side": true_side,
                          "record": {**base, "sections": sections, "edition_fingerprint": text_fingerprint(json.dumps(sections, ensure_ascii=False))},
                          "truth": sorted((index, side) for index, side in true_side.items() if side in sections)})
            if len(sections) == 1:
                (side, lines), = sections.items()
                cases.append({"mode": "unlabelled", "doc_id": doc["id"], "true_side": true_side,
                              "record": {**base, "sections": {"": lines}, "edition_fingerprint": text_fingerprint("\n".join(lines)),
                                         "reason": "calibration: canvas and edition labels hidden"},
                              "truth": sorted((index, "all") for index, s in true_side.items() if s == side)})
    return cases


def load_benchmark_reads(path: Path) -> Dict[Tuple[str, int], Optional[str]]:
    """Reads already on disk, from a benchmark outputs JSONL.

    :param path: File of ``{doc_id, image_index, text}`` objects.
    :type path: Path
    :return: ``{(doc_id, image_index): text}``.
    :rtype: Dict[Tuple[str, int], Optional[str]]
    """
    texts: Dict[Tuple[str, int], Optional[str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                texts[(row["doc_id"], int(row["image_index"]))] = row.get("text")
    return texts


def score_case(case: Dict[str, Any], row: Dict[str, Any]) -> str:
    """Outcome of one calibration case.

    :param case: :func:`calibration_cases` entry.
    :type case: Dict[str, Any]
    :param row: :func:`decide_record` result for its record.
    :type row: Dict[str, Any]
    :return: ``"dropped"``, ``"correct"`` (the pages are exactly the true sides) or ``"incorrect"``.
    :rtype: str
    """
    if row["decision"] == "drop":
        return "dropped"
    pages = sorted((page["canvas_index"], page["assign"]) for page in row["pages"])
    return "correct" if pages == [tuple(pair) for pair in case["truth"]] else "incorrect"


def run_calibration(benchmark: Path, out: Path, limit: int = CALIBRATION_LIMIT, calibrate_reads: Optional[Path] = None,
                    reads_cache: Path = DEFAULT_READS_CACHE, image_cache: Path = DEFAULT_IMAGE_CACHE, model: str = DEFAULT_MODEL,
                    effort: str = DEFAULT_EFFORT, concurrency: int = DEFAULT_CONCURRENCY, max_usd: float = DEFAULT_MAX_USD,
                    dry_run: bool = False, max_side: int = MAX_SIDE, review_csv: Optional[Path] = None,
                    read: Optional[Reader] = None, log: Callable[[str], None] = _log,
                    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep) -> Dict[str, Any]:
    """Run the rules on benchmark documents whose sides are known, with the labels hidden.

    :param benchmark: Benchmark directory.
    :type benchmark: Path
    :param out: Calibration report (JSON) to write.
    :type out: Path
    :param limit: Benchmark documents to use.
    :type limit: int
    :param calibrate_reads: Score these reads (benchmark outputs JSONL) instead of reading with the API.
    :type calibrate_reads: Optional[Path]
    :param reads_cache: Read cache file (API reads).
    :type reads_cache: Path
    :param image_cache: Directory for the reader copies of the benchmark images.
    :type image_cache: Path
    :param model: Model id.
    :type model: str
    :param effort: Effort level.
    :type effort: str
    :param concurrency: Documents read at once.
    :type concurrency: int
    :param max_usd: Budget for this run's API calls.
    :type max_usd: float
    :param dry_run: Only report what would be read and its estimated cost.
    :type dry_run: bool
    :param max_side: Long-side cap of the images sent.
    :type max_side: int
    :param review_csv: Review CSV of accepted pages (default: next to ``out``).
    :type review_csv: Optional[Path]
    :param read: Reader (default: Claude; injectable for tests).
    :type read: Optional[Reader]
    :param log: Progress printer.
    :type log: Callable[[str], None]
    :param sleep: Awaitable sleep for the backoff.
    :type sleep: Callable[[float], Awaitable[Any]]
    :return: The report (also written to ``out``).
    :rtype: Dict[str, Any]
    """
    cases = calibration_cases(benchmark, limit)
    documents = list({case["doc_id"]: case["record"] for case in cases if case["mode"] == "labelled"}.values())
    fetcher = ImageFetcher(image_cache, max_side)
    budget = Budget(max_usd, prices_for(model), MAX_TOKENS)
    stats: Counter = Counter()
    if calibrate_reads is not None:
        reader = f"file:{calibrate_reads.stem}"
        on_disk = load_benchmark_reads(calibrate_reads)

        def texts_for(record: Dict[str, Any]) -> Optional[List[Optional[str]]]:
            """Reads of a record from the file.

            :param record: Calibration record.
            :type record: Dict[str, Any]
            :return: Texts aligned with the canvases, or None when one is missing.
            :rtype: Optional[List[Optional[str]]]
            """
            keys = [(record["canonical_id"], canvas["index"]) for canvas in record["canvases"]]
            return [on_disk[key] for key in keys] if all(key in on_disk for key in keys) else None
    else:
        reader = model
        cache = ReadCache(reads_cache)
        if dry_run:
            return {"dry_run": True, "calibration_documents": len(documents), "cases": len(cases), "model": model,
                    **plan(documents, cache, model, READER_PROMPT)}
        read = read or claude_reader(make_client(log=log), model, READER_PROMPT, effort, MAX_TOKENS)
        stats = asyncio.run(read_documents(documents, cache, fetcher, read, model, READER_PROMPT, effort, budget, concurrency, log, sleep))

        def texts_for(record: Dict[str, Any]) -> Optional[List[Optional[str]]]:
            """Reads of a record from the read cache.

            :param record: Calibration record.
            :type record: Dict[str, Any]
            :return: Texts aligned with the canvases, or None when one is missing.
            :rtype: Optional[List[Optional[str]]]
            """
            return read_texts_of(record, cache, model, READER_PROMPT)
    results, review = [], []
    for case in cases:
        texts = texts_for(case["record"])
        if texts is None:
            results.append({"mode": case["mode"], "doc_id": case["doc_id"], "outcome": "no read"})
            continue
        row = decide_record(case["record"], texts, reader)
        outcome = score_case(case, row)
        results.append({"mode": case["mode"], "doc_id": case["doc_id"], "outcome": outcome, "reason": row["evidence"]["reason"],
                        "truth": case["truth"], "pages": [(page["canvas_index"], page["assign"]) for page in row["pages"]],
                        "evidence": {key: value for key, value in row["evidence"].items() if key not in ("canvases", "thresholds")},
                        "canvases": row["evidence"].get("canvases", [])})
        if row["decision"] == "accept":
            for line in review_lines(case["record"], row, texts, _image_paths(case["record"], fetcher)):
                review.append({**line, "mode": case["mode"], "true_side": case["true_side"][line["canvas_index"]],
                               "correct": outcome == "correct"})
    summary = {}
    for mode in ("labelled", "unlabelled"):
        outcomes = Counter(result["outcome"] for result in results if result["mode"] == mode)
        accepted = outcomes["correct"] + outcomes["incorrect"]
        summary[mode] = {"cases": sum(outcomes.values()), "accepted": accepted, "dropped": outcomes["dropped"],
                         "correct": outcomes["correct"], "incorrect": outcomes["incorrect"], "no_read": outcomes["no read"],
                         "precision": round(outcomes["correct"] / accepted, 4) if accepted else None,
                         "drop_reasons": dict(Counter(result["reason"] for result in results
                                                      if result["mode"] == mode and result["outcome"] == "dropped"))}
    report = {"reader": reader, "benchmark": str(benchmark), "documents": len(documents), "summary": summary,
              "thresholds": THRESHOLDS, "reads": dict(stats), "tokens": dict(budget.tokens), "spent_usd": round(budget.spent, 4),
              "results": _rounded(results)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    write_csv(review_csv or review_path(out), review, REVIEW_FIELDS + ["mode", "true_side", "correct"])
    return {key: value for key, value in report.items() if key != "results"}


# =============================================================================================================
# CLI
# =============================================================================================================

def _on_unmounted_nas(path: Path) -> bool:
    """Whether a path is on the NAS while the NAS is not mounted (writing there would fill the boot disk).

    :param path: Path to check.
    :type path: Path
    :return: True when the path is under ``/Volumes/home`` and that is not a mount point.
    :rtype: bool
    """
    return str(path).startswith(str(NAS_MOUNT) + "/") and not os.path.ismount(NAS_MOUNT)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI.

    :param argv: Arguments.
    :type argv: Optional[List[str]]
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unassigned", type=Path, default=None, help="the builder's unassigned.jsonl (required unless --calibrate)")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"side_assignments.jsonl, merged with the rows it already holds (default {DEFAULT_OUT}); "
                             f"with --calibrate the report (default {DEFAULT_CALIBRATION_OUT})")
    parser.add_argument("--reads-cache", type=Path, default=DEFAULT_READS_CACHE)
    parser.add_argument("--image-cache", type=Path, default=DEFAULT_IMAGE_CACHE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", choices=EFFORTS, default=DEFAULT_EFFORT)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD,
                        help="hard cap on this run's API spend (each call in flight holds its worst case, ~$0.18 on Opus 5.5)")
    parser.add_argument("--limit", type=int, default=None, help=f"only the first N documents (--calibrate: default {CALIBRATION_LIMIT})")
    parser.add_argument("--max-side", type=int, default=MAX_SIDE)
    parser.add_argument("--review-csv", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="no API call: report the images to read and the cost estimate")
    parser.add_argument("--calibrate", action="store_true", help="run the rules on the Arabic benchmark with the labels hidden")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--calibrate-reads", type=Path, default=None,
                        help="with --calibrate: score these reads (benchmark outputs JSONL) instead of calling the API")
    args = parser.parse_args(argv)
    for path in (args.reads_cache, args.image_cache, args.out, args.unassigned):
        if path is not None and _on_unmounted_nas(path):
            parser.error(f"{path} is on the NAS, which is not mounted")
    if args.calibrate:
        summary = run_calibration(args.benchmark, args.out or DEFAULT_CALIBRATION_OUT, args.limit or CALIBRATION_LIMIT,
                                  args.calibrate_reads, args.reads_cache, args.image_cache, args.model, args.effort,
                                  args.concurrency, args.max_usd, args.dry_run, args.max_side, args.review_csv)
    else:
        if args.unassigned is None:
            parser.error("--unassigned is required unless --calibrate")
        summary = run_side_match(args.unassigned, args.out or DEFAULT_OUT, args.reads_cache, args.image_cache, args.model,
                                 args.effort, args.concurrency, args.max_usd, args.limit, args.dry_run, args.max_side,
                                 args.review_csv)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
