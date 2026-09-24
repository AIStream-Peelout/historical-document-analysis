# File name: build_pgp_editions.py
# Date: 9/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Build ``pgp_editions_v1``: line-broken documentary page transcriptions from PGP editions.

Source text: PGP digital editions (``pgp_raw/data/footnotes.csv``). An edition covers every
side of a document while a training image shows one side, so a page image is used ONLY where
reader evidence shows that the edition text is on it (design: ``docs/v22_dataset.md`` §3.2):

1. **Evidence** — the two-reader pipeline's raw output for the image
   (``ai_reads/raw/<canonical_id>__<image_index>__<MODEL>.json``: VLM lines + Kraken
   fragments; fragments are clustered into rows by vertical overlap). The evidence file must
   describe the served image (same URL; the downloaded bytes must hash to the evidence
   ``sha256``), otherwise the page is skipped.
2. **Block presence** (rule of the ``reader_vs_gt`` analysis) — an edition block (delimited by
   blank lines or label lines) is kept when the share of its Hebrew letter 4-grams found in the
   union of the page's VLM lines and Kraken rows exceeds the maximum share measured on
   :data:`NULL_K` other pages (a per-page null drawn from a frozen pool of documentary pages of
   documents WITHOUT an edition), with >= :data:`MIN_HITS` matched n-grams, AND one of its lines
   (>= :data:`ANCHOR_MIN_LETTERS` letters) is read at >= :data:`ANCHOR_SIM` similarity
   (:func:`src.datasets.consensus.line_rule.similarity`) by a VLM line or a Kraken row.
3. **Side rule** — labelled editions (Recto/Verso, ``ע"א``/``ע"ב``): all kept blocks must lie
   on one side and cover >= :data:`SIDE_COVERAGE` of that side's Hebrew letters (and of its
   Hebrew+Arabic letters, so an unverifiable Arabic-script block cannot be silently omitted);
   unlabelled editions (single-sided): the whole edition must be >= :data:`SIDE_COVERAGE`
   present. Both sides kept, an incomplete side and paging/repeated-side labels drop the page.
4. **Document gates** — joins/multifragment records, > :data:`MAX_IMAGES` served images,
   fragments shared with another PGP document, flattened editions (inline ``(n)`` markers),
   benchmark ids, join partners, loose shelfmark keys and >= 2 shared 25-letter shingles with the
   benchmark GT (:class:`DecontamGate`).
5. **Cleaning** — PGP notation is normalised (sic/uncertainty marks, ``=`` glosses, mixed
   interlinear markers, ``/x/`` insertions, ``<x>`` supplied letters, deletions) before
   :func:`clean_diplomatic` (restorations become ``[...]``); label lines and editor commentary
   are removed. A page whose answer would still carry an unresolved editorial notation (Hebrew
   in parentheses, ``|`` separators, ``*``, glued footnote digits, years, ...) is dropped
   rather than guessed.

Rows per included page (KTIV :data:`FEATURES`, ``label_source="pgp_edition"``): ``page``
(the KTIV page prompt; answer = kept blocks with line breaks, margins/address blocks after the
main text), ``line_by_number`` (<= 2) and ``line_of_phrase_text`` (1). The line-structure rows
only address the side's first main block, and only lines above any line the edition does not
render (a lost ``[...]`` line), so "line N" is the N-th line of the image's main text.

Re-runnable: pages without an evidence file are skipped, so each run picks up what the
pipeline has read since. The split is decided per document on the fixed usable set, and the
null pool is frozen on first use, so earlier decisions do not move as evidence arrives.
Validation documents are drawn only from documents never trained on (``genizah_clean_v1``/
``v2``), so the val loss is genuinely held out. ``coverage.json`` reports how much of the
usable set has evidence.

Usage (repo root)::

    nice -n 10 .venv/bin/python -m src.finetuning.qwen_hebrew.build_pgp_editions
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import logging
import random
import re
import shutil
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

import requests
from datasets import Dataset, DatasetDict
from PIL import Image as PILImage
from PIL import ImageOps

from src.datasets.cleaning.clean_genizah_transcriptions import GAP, clean_diplomatic
from src.datasets.consensus.line_rule import letters, similarity
from src.datasets.evaluations.helper_eval_scripts.decontam_gate import DecontamGate
from src.finetuning.qwen_hebrew.build_ktiv_dataset import DECONTAM_SHINGLE, FEATURES
from src.finetuning.qwen_hebrew.ktiv_layout import GAP_TOKEN
from src.finetuning.qwen_hebrew.prompts import FRAGMENT_TRANSCRIBE_PROMPT

PILImage.MAX_IMAGE_PIXELS = None
csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)

MODEL = "qwen3-vl-8b-heb-v21b-step1200"
_REPO = Path(__file__).resolve().parents[3]
CG = _REPO / "src/datasets/raw_data/cairo_genizah"
FOOTNOTES_CSV = CG / "pgp_raw/data/footnotes.csv"
DOCUMENTS_CSV = CG / "pgp_raw/data/documents.csv"
FRAGMENTS_CSV = CG / "pgp_raw/data/fragments.csv"
MERGED_JSONL = CG / "merged/merged_shelfmarks.jsonl"
RAW_DIR = CG / "ai_reads/raw"
BENCH_IDS_JSON = CG / "decontam/benchmark_ids.json"
BENCH_GT_JSON = CG / "evaluations/genizah_test_v1/genizah_test_v1_verified.json"
CLEAN_V2_IDS_JSON = CG / "decontam/clean_v2_ids.json"
CLEAN_V1_DIR = _REPO / "src/datasets/processed/qwen_hebrew_genizah"
NAS_DATASETS = Path("/Volumes/home/studio_offload/datasets")
DEFAULT_OUT = NAS_DATASETS / "pgp_editions_v1"
DEFAULT_IMAGES = NAS_DATASETS / "pgp_editions_v1_images"
DEFAULT_INPUTS = NAS_DATASETS / "pgp_editions_v1_inputs"
DEFAULT_SERVED = DEFAULT_INPUTS / "v6_imaged_docs.jsonl"

LABEL_SOURCE = "pgp_edition"
MIN_EDITION_LETTERS = 100      # footnotes rows below this are not editions of a text page
MIN_LINE_LETTERS = 3           # a rendered line needs this many Hebrew/Arabic letters
NGRAM = 4
NULL_K = 50
NULL_POOL_SIZE = 2000
NULL_POOL_MIN_LETTERS = 200    # pool pages must carry real reader text (a strong null)
MIN_HITS = 3
ANCHOR_SIM = 0.5
ANCHOR_MIN_LETTERS = 8
SIDE_COVERAGE = 0.90
SHINGLE_CONTAINMENT = 0.5      # a benchmark text duplicate shares >= this share of either side's shingles
MAX_IMAGES = 3
MIN_ANSWER_LETTERS = 100
MIN_ANSWER_LINES = 3
MAX_LINE_CHARS = 200
MIN_IMAGE_SIDE_PX = 400
WINDOW_LINES = 6               # kept blocks longer than this are re-tested in sliding windows of this many lines
WINDOW_MIN_SHARE = 0.30        # a window fails when its share is <= its null max AND below this
EXTRA_NGRAM_SHARE = 0.30       # a reader line is explained by the answer at this 4-gram share ...
EXTRA_SIM = 0.40               # ... or this similarity to some answer line
EXTRA_CONFIRM_SIM = 0.50       # an unexplained VLM line confirmed by an unexplained Kraken row at this
LINE_ROWS_PER_PAGE = 2
LINE_MIN_LETTERS = 8
PHRASE_WORDS = (2, 3)
PHRASE_MIN_LETTERS = 8
VAL_FRACTION = 0.05
SPLIT_SEED = 20260922
DOWNLOAD_WORKERS = 4

LINE_BY_NUMBER_PROMPT = ("What is the text of line {n}, counting from the first line at the top of "
                         "the main text? Quote it exactly.")
LINE_OF_PHRASE_PROMPT = ('Which line contains the phrase «{phrase}»? Answer as JSON '
                         '{{"line": N, "text": "<the whole line>"}}.')
FAMILIES = ("page", "line_by_number", "line_of_phrase_text")
TASK_OF_FAMILY = {"page": "fragment_transcribe", "line_by_number": "line_by_number",
                  "line_of_phrase_text": "line_of_phrase_text"}

_HEB = re.compile(r"[א-ת]")
_ARABIC = re.compile(r"[ء-يٱ-ۓ]")

# ----------------------------------------------------------------------------- text helpers


def n_hebrew(text: str) -> int:
    """Number of Hebrew letters (the scorer's letters-only convention).

    :param text: Any text.
    :returns: Count of letters א-ת.
    """
    return len(letters(text))


def n_semitic(text: str) -> int:
    """Number of Hebrew plus Arabic letters.

    :param text: Any text.
    :returns: Count of Hebrew and Arabic letters.
    """
    return len(_HEB.findall(text)) + len(_ARABIC.findall(text))


def stable_hash(text: str) -> int:
    """Deterministic 64-bit hash (Python's ``hash`` is salted per process).

    :param text: Key.
    :returns: Integer hash.
    """
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16)


def page_key(canonical_id: str, image_index: int) -> str:
    """Stable identifier of one page image.

    :param canonical_id: Merged-index canonical id.
    :param image_index: Index into the served ``image_urls``.
    :returns: ``<canonical_id>__<image_index>``.
    """
    return f"{canonical_id}__{image_index}"


# ----------------------------------------------------------------------------- labels


@dataclass(frozen=True)
class Label:
    """A parsed edition label line ("Recto", "Verso - address", "ע"ב", ...).

    :param side: ``"recto"``, ``"verso"`` or None when the label names no side.
    :param kind: ``"main"``, ``"margin"``, ``"address"``, ``"other"`` or ``"stop"`` (the rest of
        the edition is apparatus: footnotes, remarks).
    :param paging: The label numbers pages/folios/sections -> non-standard side structure.
    :param bare: The label is only a side word ("Recto", "(verso)", "ע"א").
    """

    side: Optional[str]
    kind: str
    paging: bool = False
    bare: bool = False


_SIDE_WORD_RE = re.compile(r"\b(recto|verso)\b", re.I)
_REGION_WORD_RE = re.compile(r"margin|address|upside|perpendicular|diagonal|degrees|inverted|column|sideways", re.I)
_MARGINAL_RE = re.compile(r"margin|upside|perpendicular|diagonal|degrees|inverted|sideways", re.I)
_PAGING_RE = re.compile(r"\b(page|pages|fol|fols|folio|folios|leaf|leaves|sheet|bifolium|fragment|fragments|"
                        r"frag|section|part|quire)\b", re.I)
_PAGING_ONLY_RE = re.compile(r"\W*(?:[a-f]|[ivx]{1,4}|\d{1,3}\s*[rvab]|p\.?\s*\d{1,3})\W*", re.I)
_STOP_RE = re.compile(r"\W*(?:foot\s*notes?|notes?|remarks|commentary|bibliography|apparatus|הערות)"
                      r"(?:\W+(?:foot\s*notes?|notes?|remarks|הערות))?\W*", re.I)
_HEB_SIDE_LABEL_RE = re.compile(r"[(\[]?\s*ע\s*(?:[\"״]|[׳']{1,2})\s*([אב])\s*[)\]]?\s*[:.]?")
_HEB_MARGIN_LABEL_RE = re.compile(r"[(\[]?\s*ב?שוליים(?:[\s,]+[א-ת]+){0,3}\s*[)\]]?\s*[:.]?")
_HEB_ADDRESS_LABEL_RE = re.compile(r"[(\[]?\s*כתובת\s*[)\]]?\s*[:.]?")
_LABEL_VOCAB = frozenset(
    "recto verso right left top bottom upper lower margin margins address addresses column columns main "
    "text side sides upside down perpendicular diagonal diagonally line lines straight written at to "
    "degrees in the same direction as parallel other a b of and continued cont signatures signature below "
    "above inverted sideways vertical vertically horizontal horizontally outer inner middle center centre "
    "edge corner corners arabic hebrew script on with hand different second first end beginning also addendum "
    "postscript continuation".split())


_LABEL_FUNCTION_WORDS = frozenset("a of the and at to in as on with same also first second end beginning different hand "
                                  "line lines straight written direction parallel below above".split())


def _latin_label(text: str) -> Optional[Label]:
    """Interpret a Latin-script line as an edition label, or None if it reads as prose.

    :param text: Line without Hebrew letters (parenthesised Hebrew already removed).
    :returns: The label, or None.
    """
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return None
    side_m = _SIDE_WORD_RE.search(text)
    region = bool(_REGION_WORD_RE.search(text))
    unknown = [w for w in words if w not in _LABEL_VOCAB]
    paging = bool(_PAGING_RE.search(text)) or bool(_PAGING_ONLY_RE.fullmatch(text))
    if side_m or region:
        if len(words) > 20:
            return None
    elif paging:
        if len(words) > 4 or len(unknown) > 2:
            return None
    elif unknown or len(words) > 6:
        return None
    if not (side_m or region or paging) and set(words) <= _LABEL_FUNCTION_WORDS:
        return None             # "in the same direction", "on the other hand": prose, not a label
    kind = ("address" if re.search(r"\baddress", text, re.I) else
            "margin" if _MARGINAL_RE.search(text) else
            "other" if re.search(r"\b(?:other|addendum|postscript|continuation)\b", text, re.I) else "main")
    bare = bool(side_m) and set(words) <= {"recto", "verso"}
    return Label(side_m.group(1).lower() if side_m else None, kind, paging, bare)


def parse_label(line: str) -> Optional[Label]:
    """Recognise an edition label line.

    Latin labels ("Recto", "Verso - address", "Right margin, perpendicular lines.", "page b"),
    Hebrew labels (``ע"א``/``ע"ב``, ``שוליים ...``, ``כתובת``) and apparatus headers
    ("Footnotes", "הערות", which stop the edition). A Latin label carrying a parenthesised
    Hebrew annotation ("verso (השלישי)") numbers sections and is marked ``paging``.

    :param line: One raw edition line.
    :returns: The label, or None for a content/commentary line.
    """
    s = unicodedata.normalize("NFC", line).replace("\xa0", " ").strip()
    if not s or len(s) > 200:
        return None
    if _STOP_RE.fullmatch(s):
        return Label(None, "stop")
    if _ARABIC.search(s):
        return None
    if _HEB.search(s):
        m = _HEB_SIDE_LABEL_RE.fullmatch(s)
        if m:
            return Label("recto" if m.group(1) == "א" else "verso", "main", False, True)
        if _HEB_MARGIN_LABEL_RE.fullmatch(s):
            return Label(None, "margin")
        if _HEB_ADDRESS_LABEL_RE.fullmatch(s):
            return Label(None, "address")
        outside = re.sub(r"\([^()]*\)", " ", s)
        if _HEB.search(outside):
            return None
        lab = _latin_label(outside)
        return Label(lab.side, lab.kind, True, False) if lab else None
    return _latin_label(s)


# ----------------------------------------------------------------------------- line cleaning

_ZERO_WIDTH_RE = re.compile(r"[​-‏‪-‮⁦-⁩﻿]")
_SIC_RE = re.compile(r"[(\[{]\s*!\s*[)\]}]|!")
_UNCERTAIN_MARK_RE = re.compile(r"[(\[{]\s*\?+\s*[)\]}]|\(\?=[^()\n]*\)")
_EQ_GLOSS_RE = re.compile(r"[\[(]\s*=[^\[\]()\n]*[\])]")
_INTERLINEAR_MIXED_RE = re.compile(r"(?://|\\\\)([^/\\\n]{1,80})(?://|\\\\)")
_INTERLINEAR_SINGLE_RE = re.compile(r"/([^/\\\n]{1,80})\\|\\([^/\\\n]{1,80})/")
_SLASH_LETTERS_RE = re.compile(r"/([א-ת]{1,3})/")
_DELETION_RE = re.compile(r"⟦([^⟦⟧\n]*)⟧|<<([^<>\n]*)>>")
_SUPPLIED_ANGLE_RE = re.compile(r"<([^<>\n]{1,20})>")
_LATIN_PAREN_RE = re.compile(r"\(([^()א-ת]*[A-Za-z][^()א-ת]*)\)")
_LATIN_NOTE_RE = re.compile(r"omit|illegib|missing|eras|lost|torn|damag|blank|space|word|line|letter|column|"
                            r"margin|hand|continu|unclear|faded|name", re.I)
_LEADING_NUMBER_RE = re.compile(r"^\s*\(?(\d{1,3})[.):]?\)?\s+(?=\S)")
_INLINE_PAREN_NUMBER_RE = re.compile(r"\(\d{1,3}(?:\s*[–-]\s*\d{1,3})?\)")
_INLINE_DOT_NUMBER_RE = re.compile(r"(?:^|\s)\d{1,3}\.\s+[א-ת]")
_RULE_RE = re.compile(r"[_\-–—=]{5,}")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]{2,}")
# edition/page references; ``כרך`` and ``עמ'`` must stand alone before a number (inside words
# they are ordinary Hebrew: ושכרך, זוכרך; ``עמ'`` is also a scribal abbreviation)
_EDITOR_REF_RE = re.compile(r"מהד\s*['׳]|(?:^|[\s,(])עמ\s*['׳]?\s*\d|(?:^|[\s,(\[])כרך\s+(?:\d+|[א-ת]['׳\"]?)"
                            r"(?=[\s,.;:)]|$)|תרגום\s+ד\"?ר|השלמתי")
_LATIN_LOSS_RE = re.compile(r"torn|missing|illegib|no reading|lost|erased|damaged|blank|lacuna|gap|cut off", re.I)
_SCHOLAR_RE = re.compile(r"גויטיין|גוייטין|ברוקלמן|סזגין|בן-ששון|אשתור|בלאו|דוזי|פרידמן|סטילמן|שטילמן|פליישר")
_SEE_RE = re.compile(r"(?:^|[\s(\[])רא(?:ו|ה)\s*(?::|גם(?=\s)|למשל|לעיל|להלן|שם(?=[\s,.]|$)|על\s+כך)")
_PAGE_REF_RE = re.compile(r"עמ\s*['׳]\s*\d|שורות?\s+\d{1,3}(?!\d)|(?:^|\s)ש['׳]\s*\d")
_MODERN_RE = re.compile(r"(?:^|\s)(?:הכוונה|במובן|מילולית|לדעתי|כנראה|השוו|פירושו|מובנו|במקור|תרגמו)"
                        r"(?=[\s,.:;)]|$)")
_YEAR_RE = re.compile(r"(?<![\d/.,])(?:1[0-9]{3}|20[0-2][0-9])(?![\d/.,])")
_DIGIT_RANGE_RE = re.compile(r"\d+\s*[-–]\s*\d+")
_GLUED_DIGIT_RE = re.compile(r"(?<=[א-ת])\d{1,3}(?![\d/])|(?<![\d/])\d{1,3}(?=[א-ת])")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class CleanLine:
    """Outcome of cleaning one raw edition line.

    :param kind: ``"content"`` (rendered), ``"dropped"`` (ink the edition does not render as a
        line: a lost ``[...]`` line, numerals only, a stray fragment) or ``"commentary"``
        (editor text, not ink).
    :param text: Rendered text for content lines (gaps as ``[...]``), else "".
    :param flags: Unresolved editorial notations; a page whose answer carries any is dropped.
    """

    kind: str
    text: str = ""
    flags: FrozenSet[str] = frozenset()


def hebrew_commentary(text: str) -> bool:
    """Heuristic: True for a modern-Hebrew editor's note (not ink).

    Edition/page references (``עמ' 12``, ``מהד'``, ``כרך ב``), scholars' names, "see ..."
    cross references, and modern gloss vocabulary next to digits or a colon.

    :param text: Line after notation normalisation.
    :returns: Whether the line is an editor's note.
    """
    if _EDITOR_REF_RE.search(text) or _SCHOLAR_RE.search(text) or _SEE_RE.search(text):
        return True
    if _PAGE_REF_RE.search(text) and len(_HEB.findall(text)) >= 6:
        return True
    return bool(_MODERN_RE.search(text) and (re.search(r"\d", text) or ":" in text))


def is_commentary(text: str) -> bool:
    """Heuristic: True for an editor's note rather than a transcribed line.

    Latin prose (>= 2 Latin words once parenthesised glosses are removed) or a modern-Hebrew
    note (:func:`hebrew_commentary`).

    :param text: Line after notation normalisation.
    :returns: Whether the line is commentary.
    """
    return len(_LATIN_WORD_RE.findall(text)) >= 2 or hebrew_commentary(text)


def normalise_notation(text: str, numbered: bool) -> Tuple[str, Set[str]]:
    """Resolve PGP transcription notation that :func:`clean_diplomatic` does not handle.

    Removes sic marks (``(!)``, ``!``), uncertainty marks (``(?)``, ``[?]``), ``=`` glosses and
    parenthesised Latin glosses; unwraps interlinear insertions in any slash/backslash
    combination and ``/x/`` letter insertions; turns deletions (``⟦x⟧``, ``<<x>>``) into
    bracketed spans (-> gaps); drops ``<x>`` letters supplied by the editor; strips leading line
    numbers of numbered editions.

    :param text: Raw edition line.
    :param numbered: The edition prefixes its lines with line numbers.
    :returns: ``(normalised text, flags)``; ``latin_note`` flags a parenthesised note about
        omitted/illegible text (the omission cannot be rendered faithfully).
    """
    flags: Set[str] = set()
    s = unicodedata.normalize("NFC", text).replace("\xa0", " ")
    s = _ZERO_WIDTH_RE.sub("", s)
    if numbered:
        s = _LEADING_NUMBER_RE.sub("", s, count=1)

    def _latin(m: "re.Match[str]") -> str:
        """Drop a parenthesised Latin gloss, flagging notes about omitted/illegible text.

        :param m: Match of :data:`_LATIN_PAREN_RE`.
        :returns: A space.
        """
        if _LATIN_NOTE_RE.search(m.group(1)):
            flags.add("latin_note")
        return " "

    s = _LATIN_PAREN_RE.sub(_latin, s)
    s = _UNCERTAIN_MARK_RE.sub("", s)
    s = _SIC_RE.sub("", s)
    s = _EQ_GLOSS_RE.sub(" ", s)
    s = _INTERLINEAR_MIXED_RE.sub(lambda m: m.group(1), s)
    s = _INTERLINEAR_SINGLE_RE.sub(lambda m: m.group(1) or m.group(2), s)
    s = _SLASH_LETTERS_RE.sub(lambda m: m.group(1), s)
    s = _DELETION_RE.sub(lambda m: f" [{m.group(1) or m.group(2) or ''}] ", s)
    s = _SUPPLIED_ANGLE_RE.sub("", s)
    s = s.replace("^", "").replace("…", " ... ")    # a lone ellipsis is a lacuna like "..."
    return s, flags


def notation_flags(text: str) -> Set[str]:
    """Editorial notation left in a rendered line that cannot be turned into visible ink.

    :param text: Rendered line (after :func:`clean_diplomatic`).
    :returns: Flag names (empty when the line is clean).
    """
    flags: Set[str] = set()
    if "(" in text or ")" in text:
        flags.add("parentheses")
    if "|" in text:
        flags.add("pipe")
    if "*" in text:
        flags.add("asterisk")
    if "=" in text or "\\" in text or "{" in text or "}" in text:
        flags.add("markup")
    if re.search(r"(?<!\d)/|/(?!\d)", text):          # fractions (3/4) are the only slashes left as ink
        flags.add("slash")
    if "<" in text or ">" in text:
        flags.add("angle_brackets")
    if "?" in text:
        flags.add("question_mark")
    if _GLUED_DIGIT_RE.search(text):
        flags.add("glued_digits")
    if _YEAR_RE.search(text):
        flags.add("year")
    if _DIGIT_RANGE_RE.search(text):
        flags.add("digit_range")
    if re.search(r"[A-Za-z]", text):
        flags.add("latin")
    if len(text) > MAX_LINE_CHARS:
        flags.add("overlong")
    return flags


def clean_line(raw: str, numbered: bool = False) -> CleanLine:
    """Clean one raw edition line into visible-ink ground truth.

    :param raw: Raw edition line (not a label).
    :param numbered: Strip a leading line number (numbered editions).
    :returns: The cleaned line.
    """
    s, flags = normalise_notation(raw, numbered)
    if hebrew_commentary(s):
        return CleanLine("commentary")
    if len(_LATIN_WORD_RE.findall(s)) >= 2:
        latin_free = _LATIN_WORD_RE.sub(" ", s)
        if n_hebrew(latin_free) >= ANCHOR_MIN_LETTERS:
            # ink with an inline English gloss the notation rules could not remove: keep the line
            # (so it is not silently lost) but flag it; a page answering with it is dropped
            flags.add("latin_gloss")
        elif not _HEB.search(s) and _LATIN_LOSS_RE.search(s):
            return CleanLine("dropped")   # "torn off", "(no reading)": ink lines are missing here
        else:
            return CleanLine("commentary")
    cleaned, _, _ = clean_diplomatic(s)
    cleaned = unicodedata.normalize("NFC", cleaned.replace(GAP, GAP_TOKEN))
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    if n_semitic(cleaned) < MIN_LINE_LETTERS:
        return CleanLine("dropped")
    return CleanLine("content", cleaned, frozenset(flags | notation_flags(cleaned)))


def detect_line_numbering(lines: Sequence[str]) -> bool:
    """True when the edition prefixes its lines with (mostly increasing) line numbers.

    :param lines: Raw edition lines.
    :returns: Whether leading numbers are line numbers to strip.
    """
    heb = [ln for ln in lines if _HEB.search(ln)]
    nums = [int(m.group(1)) for ln in heb for m in [_LEADING_NUMBER_RE.match(ln)] if m]
    if len(heb) < 3 or len(nums) < 0.5 * len(heb):
        return False
    rising = sum(b >= a for a, b in zip(nums, nums[1:]))
    return rising >= 0.8 * max(1, len(nums) - 1)


def is_flattened(lines: Sequence[str]) -> bool:
    """True when line breaks were flattened into inline ``(1) ... (2) ...`` markers.

    :param lines: Raw edition lines.
    :returns: Whether any line carries two or more inline line markers.
    """
    return any(len(_INLINE_PAREN_NUMBER_RE.findall(ln)) >= 2 or len(_INLINE_DOT_NUMBER_RE.findall(ln)) >= 2
               for ln in lines)


# ----------------------------------------------------------------------------- edition parsing


@dataclass
class Block:
    """One edition block: lines between blank lines / label lines, with side and region.

    :param label: The raw label line that opened the block ("" when unlabelled).
    :param side: ``"recto"``/``"verso"`` (labelled editions) or None (unlabelled edition).
    :param kind: ``"main"``, ``"margin"``, ``"address"`` or ``"other"``.
    :param explicit_side: The side came from this block's own label.
    :param lines: Rendered content lines.
    :param flags: Per line, unresolved notation flags.
    :param countable: Per line, no rendered-less ink line was dropped above it in this block.
    :param n_dropped: Ink lines of the block that are not rendered.
    """

    label: str
    side: Optional[str]
    kind: str
    explicit_side: bool = False
    lines: List[str] = field(default_factory=list)
    flags: List[FrozenSet[str]] = field(default_factory=list)
    countable: List[bool] = field(default_factory=list)
    n_dropped: int = 0

    @property
    def hebrew(self) -> int:
        """Hebrew letters of the block.

        :returns: Letter count.
        """
        return sum(n_hebrew(ln) for ln in self.lines)

    @property
    def semitic(self) -> int:
        """Hebrew plus Arabic letters of the block.

        :returns: Letter count.
        """
        return sum(n_semitic(ln) for ln in self.lines)


@dataclass
class ParsedEdition:
    """An edition split into blocks, with its side structure.

    :param blocks: Blocks in edition order that hold lines or lost (unrendered) ink lines.
    :param labelled: The edition names sides (recto/verso).
    :param complex_sides: Paging labels or a side opened twice by a bare label.
    :param flattened: Inline line markers (line breaks lost).
    :param numbered: Leading line numbers were stripped.
    :param n_commentary: Commentary lines removed.
    :param n_tail: Lines ignored after an apparatus header.
    """

    blocks: List[Block]
    labelled: bool
    complex_sides: bool
    flattened: bool
    numbered: bool
    n_commentary: int = 0
    n_tail: int = 0


def parse_edition(content: str) -> ParsedEdition:
    """Split an edition into labelled blocks of cleaned lines.

    A block ends at a blank line or a label line; a label opens a new block with its side and
    region, an unlabelled block inherits both from the previous block. A drawn rule line
    (``_____``, ``-----``) ends a block (accounts and lists use rules between sections). Blocks
    before the first side label of a labelled edition are assigned to the recto. Everything
    after an apparatus header ("Footnotes", "Notes", "הערות") is ignored.

    :param content: ``footnotes.csv`` content cell.
    :returns: The parsed edition.
    """
    text = unicodedata.normalize("NFC", content).replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    raw_lines = text.split("\n")
    numbered = detect_line_numbering(raw_lines)
    blocks: List[Block] = []
    labels: List[Label] = []
    cur: Optional[Block] = None
    side: Optional[str] = None
    kind = "main"
    stop = False
    n_comment = n_tail = 0
    dropped_since_start = False
    for raw in raw_lines:
        if stop:
            n_tail += bool(raw.strip())
            continue
        if not raw.strip() or _RULE_RE.fullmatch(raw.strip()):
            cur = None          # blank line or drawn rule: the block ends
            continue
        lab = parse_label(raw)
        if lab is not None:
            if lab.kind == "stop":
                stop = True
                continue
            labels.append(lab)
            side = lab.side or side
            kind = lab.kind
            cur = Block(label=raw.strip(), side=side, kind=kind, explicit_side=lab.side is not None)
            blocks.append(cur)
            dropped_since_start = False
            continue
        cl = clean_line(raw, numbered)
        if cl.kind == "commentary":
            n_comment += 1
            continue
        if cur is None:
            cur = Block(label="", side=side, kind=kind)
            blocks.append(cur)
            dropped_since_start = False
        if cl.kind == "dropped":
            cur.n_dropped += 1
            dropped_since_start = True
            continue
        cur.lines.append(cl.text)
        cur.flags.append(cl.flags)
        cur.countable.append(not dropped_since_start)
    labelled = any(lab.side for lab in labels)
    if labelled:
        for b in blocks:
            if b.side is None:
                b.side = "recto"
    bare = collections.Counter(lab.side for lab in labels if lab.bare)
    complex_sides = any(lab.paging for lab in labels) or any(v > 1 for v in bare.values())
    return ParsedEdition(blocks=[b for b in blocks if b.lines or b.n_dropped], labelled=labelled,
                         complex_sides=complex_sides,
                         flattened=is_flattened(raw_lines), numbered=numbered, n_commentary=n_comment,
                         n_tail=n_tail)


# ----------------------------------------------------------------------------- reader evidence


@dataclass
class Reader:
    """Both readers' output on one page image.

    :param vlm: VLM line texts.
    :param rows: Kraken rows (fragments clustered by vertical overlap, right-to-left).
    :param grams: Letter n-grams of ``vlm + rows``.
    """

    vlm: List[str]
    rows: List[str]
    grams: Set[str]

    @property
    def pool(self) -> List[str]:
        """Reader lines used for the line anchor.

        :returns: VLM lines followed by Kraken rows.
        """
        return self.vlm + self.rows

    @property
    def n_letters(self) -> int:
        """Hebrew letters read on the page (both readers).

        :returns: Letter count.
        """
        return sum(n_hebrew(t) for t in self.pool)


def kraken_rows(frags: Sequence[Dict[str, Any]]) -> List[str]:
    """Cluster Kraken fragments into text rows (top to bottom, each row right-to-left).

    Fragments are sorted by vertical centre; a fragment joins an existing row when its centre
    lies inside the row's ``[y1, y2]`` span (the union of its members), else starts a new row.
    Row text = the row's fragments sorted by x descending, joined with "".

    :param frags: Fragments with ``text`` and 0-1000 ``box``.
    :returns: Row texts.
    """
    rows: List[Dict[str, Any]] = []
    for f in sorted(frags, key=lambda f: (f["box"][1] + f["box"][3]) / 2):
        cy = (f["box"][1] + f["box"][3]) / 2
        row = next((r for r in reversed(rows) if r["y1"] <= cy <= r["y2"]), None)
        if row is None:
            rows.append({"y1": f["box"][1], "y2": f["box"][3], "frags": [f]})
        else:
            row["frags"].append(f)
            row["y1"], row["y2"] = min(row["y1"], f["box"][1]), max(row["y2"], f["box"][3])
    rows.sort(key=lambda r: (r["y1"] + r["y2"]) / 2)
    return ["".join(g["text"] for g in sorted(r["frags"], key=lambda g: -(g["box"][0] + g["box"][2])))
            for r in rows]


def grams(texts: Iterable[str], n: int = NGRAM) -> Set[str]:
    """Letter n-grams of each text separately (never across text boundaries).

    :param texts: Lines.
    :param n: n-gram size.
    :returns: Set of n-grams.
    """
    out: Set[str] = set()
    for t in texts:
        s = letters(t)
        out.update(s[i:i + n] for i in range(len(s) - n + 1))
    return out


_VLM_RAW_TEXT_RE = re.compile(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _unescape(escaped: str) -> str:
    """Decode a JSON string body captured by regex from unparsed VLM output.

    :param escaped: String body without the quotes.
    :returns: Decoded text (the body itself when it holds an invalid escape).
    """
    try:
        return json.loads(f'"{escaped}"')
    except json.JSONDecodeError:  # truncated / invalid escape in a failed generation
        return escaped


def reader_from_raw(raw: Dict[str, Any]) -> Reader:
    """Build the page's reader view from a raw evidence record.

    VLM lines come from ``vlm_lines``; when the output did not parse, the ``text`` fields are
    recovered from ``vlm_raw``.

    :param raw: Raw evidence JSON.
    :returns: The reader view.
    """
    vlm = [ln["text"] for ln in raw.get("vlm_lines") or [] if (ln.get("text") or "").strip()]
    if not vlm and raw.get("vlm_raw"):
        vlm = [_unescape(m) for m in _VLM_RAW_TEXT_RE.findall(raw["vlm_raw"])]
    frags = [f for f in raw.get("frags") or [] if (f.get("text") or "").strip() and f.get("box")]
    rows = kraken_rows(frags)
    return Reader(vlm=vlm, rows=rows, grams=grams(vlm + rows))


def evidence_path(raw_dir: Path, canonical_id: str, image_index: int) -> Path:
    """Raw evidence file of one page.

    :param raw_dir: ``ai_reads/raw``.
    :param canonical_id: Canonical id.
    :param image_index: Image index.
    :returns: ``<canonical_id>__<image_index>__<MODEL>.json``.
    """
    return raw_dir / f"{canonical_id}__{image_index}__{MODEL}.json"


# ----------------------------------------------------------------------------- block presence


@dataclass(frozen=True)
class BlockTest:
    """Presence test of one block on one page.

    :param share: Share of the block's n-grams found in the page's readers.
    :param hits: Matched n-grams.
    :param null_max: Largest share on the null pages.
    :param ngram_ok: ``share > null_max`` and ``hits >= MIN_HITS``.
    :param anchored: A block line is read at >= :data:`ANCHOR_SIM`.
    :param kept: ``ngram_ok and anchored``.
    :param held_out_kept: The same test passes on a held-out unrelated page (false keep).
    """

    share: float
    hits: int
    null_max: float
    ngram_ok: bool
    anchored: bool
    kept: bool
    held_out_kept: bool = False


def best_similarity(line: str, pool: Sequence[str]) -> float:
    """Best letters-only similarity of a line against any reader line.

    :param line: Edition line.
    :param pool: Reader lines.
    :returns: Similarity in [0, 1] (0 for an empty pool).
    """
    return max((similarity(line, p) for p in pool), default=0.0)


def anchored(lines: Sequence[str], pool: Sequence[str]) -> bool:
    """Line anchor: some line (>= :data:`ANCHOR_MIN_LETTERS` letters) is read at >= :data:`ANCHOR_SIM`.

    Guards against blocks from the other side of the same document, which share names and
    formulae (n-gram hits) with this side but no whole line.

    :param lines: Block lines.
    :param pool: Reader lines on the page.
    :returns: True when anchored.
    """
    return any(best_similarity(ln, pool) >= ANCHOR_SIM for ln in lines if n_hebrew(ln) >= ANCHOR_MIN_LETTERS)


def test_block(lines: Sequence[str], reader: Reader, nulls: Sequence[Reader],
               held_out: Optional[Reader] = None) -> BlockTest:
    """Run the presence test of one block against its page and the page's null sample.

    :param lines: Block lines.
    :param reader: The page's readers.
    :param nulls: Null pages (unrelated documentary pages).
    :param held_out: One more unrelated page, tested like the real page (false-keep estimate).
    :returns: The test outcome (a block without n-grams is never kept).
    """
    g = grams(lines)
    if not g:
        return BlockTest(0.0, 0, 0.0, False, False, False)
    hits = len(g & reader.grams)
    share = hits / len(g)
    null_max = max((len(g & n.grams) / len(g) for n in nulls), default=0.0)
    ngram_ok = share > null_max and hits >= MIN_HITS
    anch = ngram_ok and anchored(lines, reader.pool)
    h_kept = False
    if held_out is not None:
        h_hits = len(g & held_out.grams)
        h_kept = (h_hits / len(g) > null_max and h_hits >= MIN_HITS and anchored(lines, held_out.pool))
    return BlockTest(share, hits, null_max, ngram_ok, anch, ngram_ok and anch, h_kept)


def null_sample(key: str, pool_keys: Sequence[str], k: int = NULL_K) -> Tuple[List[str], Optional[str]]:
    """Deterministic per-page null sample (independent of pool order and of other pages).

    :param key: Page key.
    :param pool_keys: Frozen null pool.
    :param k: Sample size.
    :returns: ``(k null keys, one held-out key)``.
    """
    ranked = sorted(pool_keys, key=lambda p: stable_hash(f"{key}|{p}"))
    return ranked[:k], (ranked[k] if len(ranked) > k else None)


# ----------------------------------------------------------------------------- side rule / answer


@dataclass
class PageDecision:
    """Inclusion decision for one page image.

    :param included: Whether the page passes the side rule.
    :param reason: Drop reason ("" when included).
    :param side: The kept side (None for an unlabelled edition).
    :param coverage: Kept share of the side's Hebrew letters.
    :param coverage_semitic: Kept share of the side's Hebrew + Arabic letters.
    """

    included: bool
    reason: str = ""
    side: Optional[str] = None
    coverage: float = 0.0
    coverage_semitic: float = 0.0


def decide_page(ed: ParsedEdition, kept: Sequence[bool], threshold: float = SIDE_COVERAGE) -> PageDecision:
    """Apply the strict side rule to one page's kept blocks.

    :param ed: Parsed edition.
    :param kept: Per block, whether the presence test kept it on this page.
    :param threshold: Required kept share of the side's letters.
    :returns: The decision.
    """
    if not any(kept):
        return PageDecision(False, "no_block_kept")
    if ed.complex_sides:
        return PageDecision(False, "complex_side_labels")
    side: Optional[str] = None
    scope = list(range(len(ed.blocks)))
    if ed.labelled:
        sides = {ed.blocks[i].side for i, k in enumerate(kept) if k}
        if len(sides) > 1:
            return PageDecision(False, "both_sides_kept")
        side = sides.pop()
        scope = [i for i, b in enumerate(ed.blocks) if b.side == side]
    heb_total = sum(ed.blocks[i].hebrew for i in scope)
    sem_total = sum(ed.blocks[i].semitic for i in scope)
    cov = sum(ed.blocks[i].hebrew for i in scope if kept[i]) / heb_total if heb_total else 0.0
    cov_sem = sum(ed.blocks[i].semitic for i in scope if kept[i]) / sem_total if sem_total else 0.0
    if cov < threshold:
        return PageDecision(False, "side_incomplete" if ed.labelled else "unlabelled_incomplete", side, cov, cov_sem)
    if cov_sem < threshold:
        return PageDecision(False, "side_arabic_unverified", side, cov, cov_sem)
    return PageDecision(True, "", side, cov, cov_sem)


@dataclass
class Answer:
    """The page transcription assembled from kept blocks.

    :param lines: Lines in answer order (main text, then margins/address/other).
    :param regions: Per line, the block kind.
    :param countable: Per line, the line may be addressed by its number from the top.
    :param flags: Per line, unresolved notation flags.
    :param blocks: Block indices in answer order.
    """

    lines: List[str]
    regions: List[str]
    countable: List[bool]
    flags: List[FrozenSet[str]]
    blocks: List[int]

    @property
    def text(self) -> str:
        """The answer text.

        :returns: Lines joined by newlines.
        """
        return "\n".join(self.lines)


def assemble_answer(ed: ParsedEdition, kept: Sequence[bool], side: Optional[str]) -> Answer:
    """Kept blocks of the page's side in edition order, main text first.

    Only lines of the side's FIRST main block can be addressed by number, and only above the
    first ink line the edition does not render; when that block was not kept, none can.

    :param ed: Parsed edition.
    :param kept: Per block, kept on this page.
    :param side: The page's side (None for unlabelled editions).
    :returns: The assembled answer.
    """
    in_side = [i for i, b in enumerate(ed.blocks) if side is None or b.side == side]
    scope = [i for i in in_side if kept[i]]
    order = [i for i in scope if ed.blocks[i].kind == "main"] + [i for i in scope if ed.blocks[i].kind != "main"]
    mains = [i for i in in_side if ed.blocks[i].kind == "main"]
    first_main = mains[0] if mains else None
    ans = Answer([], [], [], [], order)
    for i in order:
        b = ed.blocks[i]
        for j, ln in enumerate(b.lines):
            ans.lines.append(ln)
            ans.regions.append(b.kind)
            ans.countable.append(i == first_main and b.countable[j])
            ans.flags.append(b.flags[j])
    return ans


def answer_gate(ans: Answer) -> Optional[str]:
    """Reject an answer that is too short or carries unresolved notation.

    :param ans: Assembled answer.
    :returns: Drop reason, or None when the answer is usable.
    """
    flags = sorted({f for fl in ans.flags for f in fl})
    if flags:
        return f"notation_{flags[0]}"
    if n_hebrew(ans.text) < MIN_ANSWER_LETTERS:
        return "answer_too_few_letters"
    if len(ans.lines) < MIN_ANSWER_LINES:
        return "answer_too_few_lines"
    return None


# ----------------------------------------------------------------------------- rows


def row(image_path: Path, question: str, answer: str, family: str, section: str, stem: str,
        width: int, height: int) -> Dict[str, Any]:
    """One dataset row in the KTIV :data:`FEATURES` schema.

    :param image_path: Page image on disk.
    :param question: Prompt.
    :param answer: Target.
    :param family: Row family (see :data:`FAMILIES`).
    :param section: Section label.
    :param stem: Row id.
    :param width: Image width.
    :param height: Image height.
    :returns: Feature dict.
    """
    answer = unicodedata.normalize("NFC", answer).strip()
    return {"image": str(image_path), "question": question, "answer": answer,
            "task": TASK_OF_FAMILY[family], "section": section, "stem": stem,
            "label_source": LABEL_SOURCE, "target_chars": len(answer), "target_tokens": 0,
            "image_width": width, "image_height": height}


def numbered_candidates(ans: Answer) -> List[int]:
    """Answer lines that can be asked for by number (countable, >= 8 letters, no gap).

    :param ans: Assembled answer.
    :returns: Line indices (0-based).
    """
    return [i for i, ln in enumerate(ans.lines)
            if ans.countable[i] and n_hebrew(ln) >= LINE_MIN_LETTERS and GAP_TOKEN not in ln]


def line_by_number_items(ans: Answer, rng: random.Random, k: int = LINE_ROWS_PER_PAGE) -> List[Tuple[str, str, int]]:
    """``line_by_number`` question/answer pairs.

    :param ans: Assembled answer.
    :param rng: Seeded RNG.
    :param k: Maximum pairs.
    :returns: ``(question, answer, one-based line number)`` tuples.
    """
    cands = numbered_candidates(ans)
    picks = sorted(rng.sample(cands, min(k, len(cands))))
    return [(LINE_BY_NUMBER_PROMPT.format(n=i + 1), ans.lines[i], i + 1) for i in picks]


_CLEAN_WORD_RE = re.compile(r"[א-ת]+(?:[׳״'\"][א-ת]*)?")


def unique_phrases(ans: Answer, i: int) -> List[str]:
    """Phrases of 2-3 consecutive clean words of line ``i`` that occur once on the page.

    :param ans: Assembled answer.
    :param i: Line index.
    :returns: Candidate phrases (exact substrings of the line).
    """
    toks = ans.lines[i].split(" ")
    page_letters = [letters(ln) for ln in ans.lines]
    joined = "".join(page_letters)
    out = []
    for n in range(PHRASE_WORDS[0], PHRASE_WORDS[1] + 1):
        for s in range(len(toks) - n + 1):
            words = toks[s:s + n]
            if not all(_CLEAN_WORD_RE.fullmatch(w) for w in words):
                continue
            phrase = " ".join(words)
            pl = letters(phrase)
            if len(pl) < PHRASE_MIN_LETTERS:
                continue
            if sum(p.count(pl) for p in page_letters) != 1 or joined.count(pl) != 1:
                continue
            if ans.text.count(phrase) != 1:
                continue
            out.append(phrase)
    return out


def line_of_phrase_item(ans: Answer, rng: random.Random) -> Optional[Tuple[str, str, int]]:
    """One ``line_of_phrase_text`` question/answer pair.

    :param ans: Assembled answer.
    :param rng: Seeded RNG.
    :returns: ``(question, JSON answer, one-based line number)`` or None.
    """
    hosts = numbered_candidates(ans)
    rng.shuffle(hosts)
    for i in hosts:
        phrases = unique_phrases(ans, i)
        if phrases:
            phrase = rng.choice(phrases)
            answer = json.dumps({"line": i + 1, "text": ans.lines[i]}, ensure_ascii=False)
            return LINE_OF_PHRASE_PROMPT.format(phrase=phrase), answer, i + 1
    return None


# ----------------------------------------------------------------------------- inputs


def load_editions(path: Path = FOOTNOTES_CSV, min_letters: int = MIN_EDITION_LETTERS) -> Dict[str, Dict[str, str]]:
    """Longest edition per PGP document.

    :param path: ``footnotes.csv`` (utf-8-sig).
    :param min_letters: Minimum Hebrew letters of the content cell.
    :returns: ``pgpid -> footnotes row`` for rows whose ``doc_relation`` contains "edition".
    """
    eds: Dict[str, Dict[str, str]] = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if "edition" not in (r["doc_relation"] or "").lower():
                continue
            content = r["content"] or ""
            if n_hebrew(content) < min_letters:
                continue
            pid = r["document_id"].strip()
            if pid not in eds or len(content) > len(eds[pid]["content"]):
                eds[pid] = r
    return eds


@dataclass
class MergedIndex:
    """What the builders need from ``merged_shelfmarks.jsonl``.

    :param first_canonical: ``pgpid -> first canonical id`` it appears under.
    :param canonicals_of: ``pgpid -> every canonical id`` it appears under.
    :param pgpids_of: ``canonical id -> pgpids``.
    :param ktiv_sys: ``canonical id -> KTIV sys_num`` (edition documents only).
    """

    first_canonical: Dict[str, str]
    canonicals_of: Dict[str, List[str]]
    pgpids_of: Dict[str, List[str]]
    ktiv_sys: Dict[str, str]


def load_merged(path: Path, wanted_pgpids: Set[str]) -> MergedIndex:
    """Scan the merged index once (regex pre-parse; full JSON only for edition documents).

    :param path: ``merged_shelfmarks.jsonl``.
    :param wanted_pgpids: PGP ids whose records are parsed fully (KTIV sys_num).
    :returns: The index.
    """
    rx_id = re.compile(r'^\{"canonical_id": "((?:[^"\\]|\\.)*)"')
    rx_p = re.compile(r'"pgpids": \[([^\]]*)\]')
    first: Dict[str, str] = {}
    canon_of: Dict[str, List[str]] = collections.defaultdict(list)
    pgpids_of: Dict[str, List[str]] = {}
    ktiv: Dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = rx_id.match(line)
            if not m:
                continue
            cid = json.loads(f'"{m.group(1)}"')
            mp = rx_p.search(line)
            pids = [x.strip().strip('"') for x in mp.group(1).split(",") if x.strip()] if mp else []
            pgpids_of[cid] = pids
            for p in pids:
                first.setdefault(p, cid)
                canon_of[p].append(cid)
            if any(p in wanted_pgpids for p in pids):
                rec = json.loads(line)
                k = ((rec.get("images") or {}).get("ktiv") or {}).get("sys_num") \
                    or ((rec.get("sources") or {}).get("ktiv") or {}).get("sys_num")
                if k:
                    ktiv[cid] = str(k)
    return MergedIndex(first, dict(canon_of), pgpids_of, ktiv)


def load_served(path: Path) -> Dict[str, List[str]]:
    """Served image lists of the web index.

    :param path: JSONL with ``_id`` (canonical id) and ordered ``image_urls``.
    :returns: ``canonical id -> image urls``.
    """
    out: Dict[str, List[str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[r["_id"]] = list(r.get("image_urls") or [])
    return out


def load_csv_by(path: Path, key: str) -> Dict[str, Dict[str, str]]:
    """Read a utf-8-sig CSV keyed by one column.

    :param path: CSV path.
    :param key: Key column.
    :returns: ``key value -> row``.
    """
    with open(path, encoding="utf-8-sig") as fh:
        return {r[key].strip(): r for r in csv.DictReader(fh)}


def load_fragment_pgpids(path: Path = FRAGMENTS_CSV) -> Tuple[Dict[str, List[str]], Dict[str, Set[str]]]:
    """PGP fragments and the documents written on them.

    :param path: ``fragments.csv``.
    :returns: ``(fragment shelfmark -> pgpids, pgpid -> fragment shelfmarks)``.
    """
    f2p: Dict[str, List[str]] = {}
    p2f: Dict[str, Set[str]] = collections.defaultdict(set)
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            pids = [x.strip() for x in (r["pgpids"] or "").split(";") if x.strip()]
            f2p[r["shelfmark"].strip()] = pids
            for p in pids:
                p2f[p].add(r["shelfmark"].strip())
    return f2p, dict(p2f)


def load_trained_ids(clean_v2_ids: Path = CLEAN_V2_IDS_JSON, clean_v1_dir: Path = CLEAN_V1_DIR) -> Set[str]:
    """Canonical ids of every documentary page trained on so far (clean_v1 and clean_v2, all splits).

    :param clean_v2_ids: ``decontam/clean_v2_ids.json``.
    :param clean_v1_dir: Local ``qwen_hebrew_genizah`` (clean_v1) dataset; skipped if absent.
    :returns: Canonical ids.
    """
    ids = set(json.loads(clean_v2_ids.read_text(encoding="utf-8")))
    if clean_v1_dir.exists():
        from datasets import load_from_disk
        d = load_from_disk(str(clean_v1_dir))
        for split in d:
            ids.update(d[split].select_columns(["stem"])["stem"])
    else:
        logger.warning("clean_v1 dataset not found at %s; val eligibility uses clean_v2 ids only", clean_v1_dir)
    return ids


def strip_suffix(cid: str) -> str:
    """Canonical id without one trailing ``_<n>`` part number.

    :param cid: Canonical id.
    :returns: The parent id ("Cambridge_CUL_T_S_13J4_15_1" -> "Cambridge_CUL_T_S_13J4_15").
    """
    return re.sub(r"_\d+$", "", cid)


def subpart_related(a: str, b: str, pgpids_of: Dict[str, List[str]]) -> bool:
    """Two canonical ids are parts of one shelfmark (``X`` vs ``X_<n>``, or siblings ``X_<n>``/``X_<m>``
    of a parent ``X`` that is itself a PGP document record).

    A bare common prefix is NOT enough: ``T_S_10J12_4`` and ``T_S_10J12_22`` are different
    fragments of box 10J12 (stripping the fragment number would join whole boxes).

    :param a: Canonical id.
    :param b: Canonical id.
    :param pgpids_of: ``canonical id -> pgpids`` (the parent must carry one).
    :returns: Whether the two ids are sub-parts of one shelfmark.
    """
    if a == b:
        return True
    if strip_suffix(a) == b or strip_suffix(b) == a:
        return True
    pa, pb = strip_suffix(a), strip_suffix(b)
    return pa == pb and pa != a and pb != b and bool(pgpids_of.get(pa))


# ----------------------------------------------------------------------------- decontamination

_INSTITUTION_PREFIXES = ("oxfordbodleian", "oxford", "bodleian", "bodl", "cambridgecul", "cambridge", "cul", "ulc",
                         "newyorkjts", "newyork", "jts", "manchesterjrl", "manchester", "jrl", "budapestmta",
                         "budapest", "mta", "paris", "philadelphia", "penn", "london", "washington")


def fragment_key(shelfmark_or_id: str, gate: DecontamGate) -> Optional[str]:
    """Institution-free fragment key, so one fragment matches across naming schemes.

    The Bodleian source added canonical ids ``Oxford_Bodleian_Bodl_MS_heb_d_79_36`` for
    fragments the benchmark knows as ``Oxford_Bodleian_MS_heb_d_79_36`` and PGP as
    ``Bodl. MS heb. d 79/36``; :meth:`DecontamGate.loose_key` keeps the institution tokens, so
    they are stripped here (all three -> ``mshebd79|36``).

    :param shelfmark_or_id: Shelfmark or canonical id.
    :param gate: Decontamination gate (loose-key normaliser).
    :returns: The key, or None when unkeyable.
    """
    key = gate.loose_key(shelfmark_or_id.replace("_", " "))
    if not key:
        return None
    changed = True
    while changed:
        changed = False
        for p in _INSTITUTION_PREFIXES:
            if key.startswith(p) and len(key) > len(p) + 2:
                key, changed = key[len(p):], True
    return key


def shingle_set(text: str, n: int = DECONTAM_SHINGLE) -> Set[str]:
    """Letter shingles of a text.

    :param text: Any text.
    :param n: Shingle length in Hebrew letters.
    :returns: Set of shingles.
    """
    s = letters(text)
    return {s[i:i + n] for i in range(len(s) - n + 1)}


@dataclass
class Benchmark:
    """The frozen benchmark in the forms the document gate compares against.

    :param ids: Benchmark canonical ids.
    :param keys: Fragment keys of the benchmark ids, shelfmarks and gate inventory.
    :param shingles: Per benchmark document, its GT shingle set.
    :param all_shingles: Union of the GT shingles.
    """

    ids: Set[str]
    keys: Set[str]
    shingles: Dict[str, Set[str]]
    all_shingles: Set[str]


def load_benchmark(gate: DecontamGate, ids_path: Path = BENCH_IDS_JSON, gt_path: Path = BENCH_GT_JSON) -> Benchmark:
    """Load benchmark ids, fragment keys and GT shingles.

    :param gate: Decontamination gate.
    :param ids_path: ``decontam/benchmark_ids.json``.
    :param gt_path: Verified benchmark JSON (``docs[*].gt``).
    :returns: The benchmark.
    """
    ids = set(json.loads(ids_path.read_text(encoding="utf-8")))
    docs = json.loads(gt_path.read_text(encoding="utf-8"))["docs"]
    keys = {k for k in (fragment_key(x, gate) for x in list(ids) + [d.get("shelf_mark", "") for d in docs]) if k}
    keys |= {k for k in (fragment_key(k0, gate) for k0 in gate.bench_keys) if k}
    sh = {d["doc_id"]: shingle_set(d.get("gt") or "") for d in docs}
    return Benchmark(ids, keys, sh, set().union(*sh.values()) if sh else set())


def benchmark_overlap(text: str, bench: Benchmark) -> Tuple[int, float, Optional[str]]:
    """Largest shingle overlap between a text and any single benchmark document.

    :param text: Candidate edition text.
    :param bench: Benchmark.
    :returns: ``(shingles shared with the whole benchmark, max pair containment, that benchmark doc)``;
        containment = shared / min(|candidate|, |benchmark doc|) shingles.
    """
    s = shingle_set(text)
    total = len(s & bench.all_shingles)
    if not total:
        return 0, 0.0, None
    best, best_id = 0.0, None
    for bid, b in bench.shingles.items():
        shared = len(s & b)
        if shared >= 2:
            c = shared / max(1, min(len(s), len(b)))
            if c > best:
                best, best_id = c, bid
    return total, best, best_id


# ----------------------------------------------------------------------------- document gates


@dataclass
class DocInfo:
    """One usable edition document and its document-level gate.

    :param pgpid: PGP document id.
    :param canonical_id: First canonical id of the pgpid in the merged index.
    :param urls: Served image urls.
    :param reason: Document-level drop reason ("" when eligible).
    :param trained: Seen in genizah_clean_v1/v2 (never a val document).
    """

    pgpid: str
    canonical_id: str
    urls: List[str]
    reason: str = ""
    trained: bool = False


def usable_documents(editions: Dict[str, Dict[str, str]], merged: MergedIndex, served: Dict[str, List[str]],
                     bench_ids: Set[str]) -> Tuple[Dict[str, DocInfo], collections.Counter]:
    """Editions whose document is in the served index and outside the benchmark (the usable set).

    :param editions: ``pgpid -> edition row``.
    :param merged: Merged index.
    :param served: Served image lists.
    :param bench_ids: Benchmark canonical ids.
    :returns: ``(pgpid -> DocInfo, counts of editions not usable by reason)``.
    """
    out: Dict[str, DocInfo] = {}
    lost: collections.Counter = collections.Counter()
    for pid in sorted(editions, key=lambda p: (len(p), p)):
        cid = merged.first_canonical.get(pid)
        if cid is None:
            lost["not_in_merged_index"] += 1
        elif cid not in served or not served[cid]:
            lost["not_served"] += 1
        elif cid in bench_ids:
            lost["benchmark_id"] += 1
        else:
            out[pid] = DocInfo(pid, cid, served[cid])
    return out, lost


def gate_documents(docs: Dict[str, DocInfo], editions: Dict[str, Dict[str, str]], parsed: Dict[str, ParsedEdition],
                   merged: MergedIndex, pgp_docs: Dict[str, Dict[str, str]], fragments: Tuple[Dict, Dict],
                   bench: Benchmark, gate: DecontamGate, trained: Set[str],
                   containment: float = SHINGLE_CONTAINMENT) -> collections.Counter:
    """Set each usable document's document-level drop reason and trained flag (in place).

    Benchmark layers, each sufficient: a join partner (shares a PGP id or is a sub-part of a
    benchmark id), the same fragment under another name (:func:`fragment_key` of any canonical
    id or shelfmark part), and a text duplicate (>= ``containment`` of the smaller side's 25-letter
    shingles shared with one benchmark document). Sharing a few shingles is NOT exclusion:
    documentary formulae (deeds in one scribe's hand, ketubba clauses) put >= 2 shared shingles on
    a quarter of all editions; that count is reported instead.

    :param docs: Usable documents.
    :param editions: Edition rows.
    :param parsed: Parsed editions.
    :param merged: Merged index.
    :param pgp_docs: ``documents.csv`` by pgpid.
    :param fragments: ``(fragment -> pgpids, pgpid -> fragments)``.
    :param bench: Benchmark.
    :param gate: Decontamination gate (loose keys, KTIV training sys_nums).
    :param trained: Canonical ids trained on before.
    :param containment: Pair containment that marks a text duplicate.
    :returns: Informational counters (formula overlaps kept).
    """
    f2p, p2f = fragments
    info: collections.Counter = collections.Counter()
    bench_pgpids = {p for b in bench.ids for p in merged.pgpids_of.get(b, [])}
    bench_partners = {c for p in bench_pgpids for c in merged.canonicals_of.get(p, [])} - bench.ids
    for pid, d in docs.items():
        meta = pgp_docs.get(pid, {})
        shelf = meta.get("shelfmark", "")
        frags = p2f.get(pid, set())
        others = {q for f in frags for q in f2p.get(f, [])} - {pid}
        d.trained = d.canonical_id in trained or bool(
            merged.ktiv_sys.get(d.canonical_id) and merged.ktiv_sys[d.canonical_id] in gate.ktiv_train_sys)
        names = set(merged.canonicals_of.get(pid, [])) | {d.canonical_id} | {x.strip() for x in shelf.split("+")}
        total, cont, best_id = benchmark_overlap(editions[pid]["content"], bench)
        if total >= 2 and cont < containment:
            info["formula_overlap_ge2_shingles_kept"] += 1
        if d.canonical_id in bench_partners or pid in bench_pgpids or any(
                subpart_related(d.canonical_id, b, merged.pgpids_of) for b in bench.ids):
            d.reason = "benchmark_join_partner"
        elif any(fragment_key(x, gate) in bench.keys for x in names if x):
            d.reason = "benchmark_fragment_key"
        elif best_id is not None and cont >= containment:
            d.reason = "benchmark_text_duplicate"
        elif (meta.get("multifragment") or "").strip() or "+" in shelf or len(frags) > 1 \
                or len(merged.canonicals_of.get(pid, [])) > 1:
            d.reason = "join_or_multifragment"
        elif others:
            d.reason = "fragment_shared_with_other_document"
        elif len(d.urls) > MAX_IMAGES:
            d.reason = "too_many_images"
        elif parsed[pid].flattened:
            d.reason = "edition_flattened"
        elif not any(b.lines for b in parsed[pid].blocks):
            d.reason = "edition_empty_after_cleaning"
    return info


# ----------------------------------------------------------------------------- split


def document_groups(docs: Dict[str, DocInfo], merged: MergedIndex, fragments: Tuple[Dict, Dict]) -> Dict[str, str]:
    """Group documents that must share a split (shared canonical ids, fragments or sub-parts).

    :param docs: Documents to group.
    :param merged: Merged index.
    :param fragments: ``(fragment -> pgpids, pgpid -> fragments)``.
    :returns: ``pgpid -> group key`` (the smallest canonical id of the group).
    """
    parent: Dict[str, str] = {p: p for p in docs}

    def find(x: str) -> str:
        """Union-find root with path halving.

        :param x: Document id.
        :returns: Root id.
        """
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        """Merge the groups of two documents (the smaller id becomes the root).

        :param a: Document id.
        :param b: Document id.
        """
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    _, p2f = fragments
    by_canon: Dict[str, List[str]] = collections.defaultdict(list)
    by_frag: Dict[str, List[str]] = collections.defaultdict(list)
    by_base: Dict[str, List[str]] = collections.defaultdict(list)
    for p, d in docs.items():
        for c in set(merged.canonicals_of.get(p, [])) | {d.canonical_id}:
            by_canon[c].append(p)
        for f in p2f.get(p, set()):
            by_frag[f].append(p)
        by_base[strip_suffix(d.canonical_id)].append(p)
        by_base[d.canonical_id].append(p)
    for groups in (by_canon, by_frag):
        for members in groups.values():
            for m in members[1:]:
                union(members[0], m)
    for members in by_base.values():
        for a in members:
            for b in members:
                if a < b and subpart_related(docs[a].canonical_id, docs[b].canonical_id, merged.pgpids_of):
                    union(a, b)
    comp: Dict[str, List[str]] = collections.defaultdict(list)
    for p in docs:
        comp[find(p)].append(p)
    out = {}
    for members in comp.values():
        key = min(docs[m].canonical_id for m in members)
        for m in members:
            out[m] = key
    return out


def assign_splits(docs: Dict[str, DocInfo], groups: Dict[str, str], val_fraction: float = VAL_FRACTION,
                  seed: int = SPLIT_SEED) -> Dict[str, str]:
    """Deterministic document-level split with groups kept together.

    A group goes to ``val`` when none of its documents was trained on before and its hash
    falls under a threshold scaled so that ``val_fraction`` of all groups end up in val. The
    assignment depends only on the (fixed) document set, never on which pages have evidence.

    :param docs: Eligible documents.
    :param groups: ``pgpid -> group key``.
    :param val_fraction: Target share of groups in val.
    :param seed: Hash seed.
    :returns: ``pgpid -> "train" | "val"``.
    """
    members: Dict[str, List[str]] = collections.defaultdict(list)
    for p, g in groups.items():
        members[g].append(p)
    eligible = [g for g, ms in members.items() if not any(docs[m].trained for m in ms)]
    threshold = min(1.0, val_fraction * len(members) / max(1, len(eligible)))
    val_groups = {g for g in eligible if stable_hash(f"{seed}:{g}") / 2 ** 64 < threshold}
    return {p: ("val" if groups[p] in val_groups else "train") for p in docs}


# ----------------------------------------------------------------------------- null pool


def load_or_create_null_pool(path: Path, raw_dir: Path, merged: MergedIndex, edition_pgpids: Set[str],
                             size: int = NULL_POOL_SIZE) -> List[str]:
    """Frozen pool of null pages: documentary pages of documents WITHOUT an edition.

    Created once (sha1-ordered, pages with >= :data:`NULL_POOL_MIN_LETTERS` reader letters)
    and reused on every run so decisions are stable as evidence arrives.

    :param path: Pool JSON (created when missing).
    :param raw_dir: Raw evidence directory.
    :param merged: Merged index (to exclude any canonical id holding an edition document).
    :param edition_pgpids: PGP ids with an edition.
    :param size: Pool size.
    :returns: Evidence file names in the pool.
    """
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["pages"]
    suffix = f"__{MODEL}.json"
    cands = []
    for name in sorted((p.name for p in raw_dir.iterdir() if p.name.endswith(suffix)), key=stable_hash):
        cid = name.split("__")[0]
        if any(p in edition_pgpids for p in merged.pgpids_of.get(cid, [])):
            continue
        rec = json.loads((raw_dir / name).read_text(encoding="utf-8"))
        if reader_from_raw(rec).n_letters < NULL_POOL_MIN_LETTERS:
            continue
        cands.append(name)
        if len(cands) >= size:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": MODEL, "created": datetime.now(timezone.utc).isoformat(),
                                "rule": f"no edition document, >= {NULL_POOL_MIN_LETTERS} reader letters, "
                                        "sha1(file name) order", "pages": cands}, indent=1), encoding="utf-8")
    return cands


# ----------------------------------------------------------------------------- images


def image_path_for(images_dir: Path, canonical_id: str, image_index: int) -> Path:
    """Where a page image is stored.

    :param images_dir: NAS images directory.
    :param canonical_id: Canonical id.
    :param image_index: Image index.
    :returns: ``<images_dir>/<canonical_id>/<image_index>.jpg``.
    """
    return images_dir / canonical_id / f"{image_index}.jpg"


def fetch_image(url: str, sha256: str, dest: Path, session: requests.Session, retries: int = 3) -> Dict[str, Any]:
    """Download a served page image, verify it is the image the readers saw, store it.

    The original bytes are kept when no EXIF rotation is needed; otherwise the image is
    rotated upright (``ImageOps.exif_transpose``) and saved as JPEG at full resolution. A JSON
    sidecar records the source hash so re-runs reuse the file.

    :param url: Served image URL.
    :param sha256: Evidence ``sha256`` of the image bytes.
    :param dest: Target JPEG path.
    :param session: HTTP session.
    :param retries: Download attempts.
    :returns: ``{"ok", "reason", "width", "height", ...}``.
    """
    meta_path = dest.with_suffix(".json")
    if dest.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("source_sha256") == sha256:
            return {"ok": True, "reason": "", **meta}
    data = b""
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=120)
            resp.raise_for_status()
            data = resp.content
            break
        except requests.RequestException as exc:  # network hiccups on a multi-day incremental build
            logger.warning("download failed (%d/%d) %s: %s", attempt + 1, retries, url, exc)
            time.sleep(2 * (attempt + 1))
    if not data:
        return {"ok": False, "reason": "image_download_failed"}
    if hashlib.sha256(data).hexdigest() != sha256:
        return {"ok": False, "reason": "image_changed_since_read"}
    im = PILImage.open(io.BytesIO(data))
    orientation = im.getexif().get(0x0112, 1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if im.format == "JPEG" and orientation in (0, 1) and im.mode == "RGB":
        dest.write_bytes(data)
        width, height, reencoded = im.width, im.height, False
    else:
        up = ImageOps.exif_transpose(im).convert("RGB")
        up.save(dest, "JPEG", quality=95)
        width, height, reencoded = up.width, up.height, True
    meta = {"source_url": url, "source_sha256": sha256, "exif_orientation": int(orientation),
            "width": width, "height": height, "reencoded": reencoded}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return {"ok": True, "reason": "", **meta}


# ----------------------------------------------------------------------------- build


@dataclass
class PageResult:
    """Evaluation of one evidence page (every evaluated page is recorded for audit).

    :param pgpid: PGP id.
    :param canonical_id: Canonical id.
    :param image_index: Image index.
    :param image_url: Served URL.
    :param sha256: Evidence image hash.
    :param split: Document split.
    :param decision: Side-rule decision.
    :param tests: Per block presence test.
    :param answer: Assembled answer (included pages).
    :param reason: Final drop reason ("" when included).
    """

    pgpid: str
    canonical_id: str
    image_index: int
    image_url: str
    sha256: str
    split: str
    decision: PageDecision
    tests: List[BlockTest]
    answer: Optional[Answer] = None
    reason: str = ""


def unsupported_windows(ed: ParsedEdition, kept: Sequence[bool], reader: Reader, nulls: Sequence[Reader],
                        size: int = WINDOW_LINES) -> List[Tuple[int, int]]:
    """Windows of kept blocks that the readers do not support.

    A block is kept on the evidence of part of it; an unlabelled edition that runs from the
    recto into the verso without a blank line, or a rotated margin whose words also occur in
    the main text, passes the block test while half its lines are not read on this image. Each
    kept block longer than ``size`` lines is therefore re-tested in sliding windows of ``size``
    lines: a window fails when its 4-gram share is no higher than on the page's null pages AND
    below :data:`WINDOW_MIN_SHARE`. Six lines is long enough that the 2-4 line witness
    signatures in other hands, which the readers often miss although they are on the image, do
    not fail a window on their own (calibrated on the first 129 included pages).

    :param ed: Parsed edition.
    :param kept: Per block, kept on this page.
    :param reader: The page's readers.
    :param nulls: The page's null sample.
    :param size: Window length in lines.
    :returns: ``(block index, first line of the window)`` of every failing window.
    """
    out = []
    for i, b in enumerate(ed.blocks):
        if not kept[i] or len(b.lines) <= size:
            continue
        for st in range(len(b.lines) - size + 1):
            g = grams(b.lines[st:st + size])
            if not g:
                continue
            share = len(g & reader.grams) / len(g)
            null_max = max((len(g & n.grams) / len(g) for n in nulls), default=0.0)
            if share <= null_max and share < WINDOW_MIN_SHARE:
                out.append((i, st))
    return out


def missing_reader_lines(answer_lines: Sequence[str], reader: Reader) -> List[str]:
    """Lines both readers see on the image that the answer does not contain.

    A VLM line (>= :data:`LINE_MIN_LETTERS` letters) counts when the answer does not explain it
    (4-gram share < :data:`EXTRA_NGRAM_SHARE` and similarity < :data:`EXTRA_SIM` to every answer
    line) and an equally unexplained Kraken row reads the same text (similarity >=
    :data:`EXTRA_CONFIRM_SIM`): two independent readers agree on ink the edition does not
    render (a second document on the leaf, lines the editor skipped, an unkept margin).

    :param answer_lines: Answer lines.
    :param reader: The page's readers.
    :returns: The confirmed VLM lines.
    """
    ag = grams(answer_lines)

    def explained(t: str) -> bool:
        """The answer accounts for a reader line (n-gram share or line similarity).

        :param t: Reader line.
        :returns: Whether it is explained.
        """
        g = grams([t])
        if g and len(g & ag) / len(g) >= EXTRA_NGRAM_SHARE:
            return True
        return best_similarity(t, answer_lines) >= EXTRA_SIM

    rows = [k for k in reader.rows if n_hebrew(k) >= LINE_MIN_LETTERS and not explained(k)]
    return [v for v in reader.vlm if n_hebrew(v) >= LINE_MIN_LETTERS and not explained(v)
            and best_similarity(v, rows) >= EXTRA_CONFIRM_SIM]


def evaluate_page(ed: ParsedEdition, reader: Reader, nulls: Sequence[Reader],
                  held_out: Optional[Reader]) -> Tuple[List[BlockTest], PageDecision, Optional[Answer], str]:
    """Presence test, side rule, answer gates and the two reader cross-checks for one page.

    :param ed: Parsed edition.
    :param reader: The page's readers.
    :param nulls: Null sample.
    :param held_out: Held-out unrelated page.
    :returns: ``(block tests, decision, answer or None, drop reason or "")``.
    """
    tests = [test_block(b.lines, reader, nulls, held_out) for b in ed.blocks]
    kept = [t.kept for t in tests]
    dec = decide_page(ed, kept)
    if not dec.included:
        return tests, dec, None, dec.reason
    ans = assemble_answer(ed, kept, dec.side)
    reason = answer_gate(ans)
    if not reason and unsupported_windows(ed, kept, reader, nulls):
        reason = "kept_block_partly_unread"
    if not reason and missing_reader_lines(ans.lines, reader):
        reason = "reader_line_not_in_answer"
    return tests, dec, (None if reason else ans), reason or ""


def drop_duplicate_side_photos(results: Sequence[PageResult]) -> int:
    """Keep one page per photographed side of a document.

    Some served image lists hold two photographs of the same side (a library image and a
    PGP/FJP copy), which pass the side rule with identical answers. A later page (by image index)
    is dropped with reason ``duplicate_side_photo`` when its image bytes hash equals, or its kept
    lines are identical to, those of an earlier included page of the SAME document.

    :param results: Evaluated pages; included ones (empty ``reason``) are updated in place.
    :returns: Number of pages dropped.
    """
    seen_sha: Dict[str, Set[str]] = collections.defaultdict(set)
    seen_lines: Dict[str, Set[Tuple[str, ...]]] = collections.defaultdict(set)
    dropped = 0
    for r in sorted((r for r in results if not r.reason), key=lambda r: (r.pgpid, r.image_index)):
        lines = tuple(r.answer.lines)
        if (r.sha256 and r.sha256 in seen_sha[r.pgpid]) or lines in seen_lines[r.pgpid]:
            r.reason = "duplicate_side_photo"
            dropped += 1
            continue
        if r.sha256:
            seen_sha[r.pgpid].add(r.sha256)
        seen_lines[r.pgpid].add(lines)
    return dropped


def rows_for_page(res: PageResult, img: Dict[str, Any], image_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """All rows of one included page.

    :param res: Included page.
    :param img: Image metadata (width/height).
    :param image_path: Stored image.
    :returns: ``(family, row)`` pairs, page row first.
    """
    ans = res.answer
    stem = f"pgp_{res.pgpid}_{res.image_index}"
    w, h = img["width"], img["height"]
    rng = random.Random(stable_hash(f"{SPLIT_SEED}:{stem}"))
    out = [("page", row(image_path, FRAGMENT_TRANSCRIBE_PROMPT, ans.text, "page", "pgp_page", f"{stem}_page", w, h))]
    for j, (q, a, _) in enumerate(line_by_number_items(ans, rng)):
        out.append(("line_by_number", row(image_path, q, a, "line_by_number", "line", f"{stem}_lbn{j}", w, h)))
    lop = line_of_phrase_item(ans, rng)
    if lop:
        out.append(("line_of_phrase_text",
                    row(image_path, lop[0], lop[1], "line_of_phrase_text", "line", f"{stem}_lop", w, h)))
    return out


def save_dataset(splits: Dict[str, List[Dict[str, Any]]], out_dir: Path, sidecars: Dict[str, str]) -> None:
    """Write the DatasetDict plus sidecar files, replacing the previous build atomically.

    Splits without rows are not written (``stats.json`` lists every split's row count).

    :param splits: ``split name -> rows``.
    :param out_dir: Destination.
    :param sidecars: ``file name -> text`` written next to the splits.
    """
    tmp = out_dir.with_name(out_dir.name + ".building")
    if tmp.exists():
        shutil.rmtree(tmp)
    # empty splits are left out: datasets 4.4 cannot save an empty split with an Image column
    dsd = DatasetDict({name: Dataset.from_list(rows, features=FEATURES) for name, rows in splits.items() if rows})
    dsd.save_to_disk(str(tmp))
    for name, text in sidecars.items():
        (tmp / name).write_text(text, encoding="utf-8")
    prev = out_dir.with_name(out_dir.name + ".previous")
    if out_dir.exists():
        if prev.exists():
            shutil.rmtree(prev)
        out_dir.rename(prev)
    tmp.rename(out_dir)
    if prev.exists():
        shutil.rmtree(prev)


@dataclass
class Context:
    """Everything the page evaluation needs, loaded once per run.

    :param editions: ``pgpid -> edition row``.
    :param merged: Merged index.
    :param docs: Usable documents (with their document-level gate).
    :param not_usable: Editions outside the usable set, by reason.
    :param parsed: Parsed editions of the usable documents.
    :param eligible: Usable documents that pass the document gate.
    :param splits: ``pgpid -> "train" | "val"``.
    :param pool_names: Frozen null pool (evidence file names).
    :param pool: Null pool readers.
    :param gate_info: Informational document-gate counters.
    :param shingle_containment: Benchmark text-duplicate threshold used.
    """

    editions: Dict[str, Dict[str, str]]
    merged: MergedIndex
    docs: Dict[str, DocInfo]
    not_usable: collections.Counter
    parsed: Dict[str, ParsedEdition]
    eligible: Dict[str, DocInfo]
    splits: Dict[str, str]
    pool_names: List[str]
    pool: Dict[str, Reader]
    gate_info: collections.Counter
    shingle_containment: float


def prepare(inputs_dir: Path, served_path: Path, raw_dir: Path = RAW_DIR,
            shingle_containment: float = SHINGLE_CONTAINMENT) -> Context:
    """Load inputs, apply the document gates, split, and load the null pool.

    :param inputs_dir: Frozen build inputs (null pool).
    :param served_path: Served image list snapshot.
    :param raw_dir: Raw evidence directory.
    :param shingle_containment: Pair containment marking a benchmark text duplicate.
    :returns: The context.
    """
    t0 = time.time()
    editions = load_editions()
    merged = load_merged(MERGED_JSONL, set(editions))
    served = load_served(served_path)
    bench_ids = set(json.loads(BENCH_IDS_JSON.read_text(encoding="utf-8")))
    docs, not_usable = usable_documents(editions, merged, served, bench_ids)
    parsed = {p: parse_edition(editions[p]["content"]) for p in docs}
    gate = DecontamGate()
    bench = load_benchmark(gate)
    trained = load_trained_ids()
    fragments = load_fragment_pgpids()
    gate_info = gate_documents(docs, editions, parsed, merged, load_csv_by(DOCUMENTS_CSV, "pgpid"), fragments,
                               bench, gate, trained, shingle_containment)
    eligible = {p: d for p, d in docs.items() if not d.reason}
    splits = assign_splits(eligible, document_groups(eligible, merged, fragments))
    pool_names = load_or_create_null_pool(inputs_dir / "null_pool.json", raw_dir, merged, set(editions))
    pool = {n: reader_from_raw(json.loads((raw_dir / n).read_text(encoding="utf-8"))) for n in pool_names}
    logger.info("editions %d, usable %d, eligible %d, null pool %d (%.0fs)", len(editions), len(docs),
                len(eligible), len(pool), time.time() - t0)
    return Context(editions, merged, docs, not_usable, parsed, eligible, splits, pool_names, pool, gate_info,
                   shingle_containment)


def evaluate_pages(ctx: Context, raw_dir: Path = RAW_DIR,
                   limit_pages: int = 0) -> Tuple[List[PageResult], collections.Counter, int]:
    """Evaluate every eligible page that has an evidence file.

    :param ctx: Prepared context.
    :param raw_dir: Raw evidence directory.
    :param limit_pages: Stop after this many evidence pages (0 = all).
    :returns: ``(page results, outcome counts, evidence pages seen)``.
    """
    reasons: collections.Counter = collections.Counter()
    results: List[PageResult] = []
    n_evidence = 0
    for pid in sorted(ctx.eligible, key=lambda p: (len(p), p)):
        d = ctx.eligible[pid]
        for idx, url in enumerate(d.urls):
            path = evidence_path(raw_dir, d.canonical_id, idx)
            if not path.exists() or (limit_pages and n_evidence >= limit_pages):
                continue
            n_evidence += 1
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:  # the pipeline may be writing this file right now
                reasons["evidence_unreadable"] += 1
                continue
            if raw.get("image_url") != url:
                reasons["evidence_url_mismatch"] += 1
                continue
            null_keys, held = null_sample(page_key(d.canonical_id, idx), ctx.pool_names)
            tests, dec, ans, reason = evaluate_page(ctx.parsed[pid], reader_from_raw(raw),
                                                    [ctx.pool[k] for k in null_keys],
                                                    ctx.pool[held] if held else None)
            results.append(PageResult(pid, d.canonical_id, idx, url, raw.get("sha256", ""), ctx.splits[pid], dec,
                                      tests, ans, reason))
            reasons[reason or "included"] += 1
    return results, reasons, n_evidence


def build(out_dir: Path, images_dir: Path, inputs_dir: Path, served_path: Path, raw_dir: Path = RAW_DIR,
          limit_pages: int = 0, download: bool = True,
          shingle_containment: float = SHINGLE_CONTAINMENT) -> Dict[str, Any]:
    """Build (or rebuild) ``pgp_editions_v1`` from the evidence available now.

    :param out_dir: Dataset destination (NAS).
    :param images_dir: Page image store (NAS).
    :param inputs_dir: Frozen build inputs (null pool).
    :param served_path: Served image list snapshot.
    :param raw_dir: Raw evidence directory.
    :param limit_pages: Stop after this many evidence pages (0 = all; smoke tests).
    :param download: Download images (False = dry run, no dataset written).
    :param shingle_containment: Pair containment marking a benchmark text duplicate
        (0 = exclude any document sharing >= 2 shingles with one benchmark document).
    :returns: The stats dict.
    """
    t0 = time.time()
    ctx = prepare(inputs_dir, served_path, raw_dir, shingle_containment)
    editions, docs, eligible, parsed, splits = ctx.editions, ctx.docs, ctx.eligible, ctx.parsed, ctx.splits
    results, reasons, n_evidence = evaluate_pages(ctx, raw_dir, limit_pages)

    included = [r for r in results if not r.reason]
    img_meta: Dict[str, Dict[str, Any]] = {}
    if download and included:
        session = requests.Session()
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
            futs = {page_key(r.canonical_id, r.image_index): ex.submit(
                fetch_image, r.image_url, r.sha256, image_path_for(images_dir, r.canonical_id, r.image_index), session)
                for r in included}
            img_meta = {k: f.result() for k, f in futs.items()}
        for r in included:
            m = img_meta[page_key(r.canonical_id, r.image_index)]
            if not m["ok"]:
                r.reason = m["reason"]
            elif min(m["width"], m["height"]) < MIN_IMAGE_SIDE_PX:
                r.reason = "image_too_small"
            if r.reason:
                reasons["included"] -= 1
                reasons[r.reason] += 1
    n_duplicate = drop_duplicate_side_photos(results)
    if n_duplicate:
        reasons["included"] -= n_duplicate
        reasons["duplicate_side_photo"] += n_duplicate
    included = [r for r in results if not r.reason]

    split_rows: Dict[str, List[Dict[str, Any]]] = {f"train_{f}": [] for f in FAMILIES}
    split_rows["val"] = []
    manifest: List[Dict[str, Any]] = []
    fam_counts: collections.Counter = collections.Counter()
    for r in included:
        key = page_key(r.canonical_id, r.image_index)
        ipath = image_path_for(images_dir, r.canonical_id, r.image_index)
        meta = img_meta.get(key, {"width": 0, "height": 0})
        rows = rows_for_page(r, meta, ipath) if download else []
        for fam, x in rows:
            split_rows["val" if r.split == "val" else f"train_{fam}"].append(x)
            fam_counts[fam] += 1
        ed = parsed[r.pgpid]
        manifest.append({
            "pgpid": r.pgpid, "canonical_id": r.canonical_id, "image_index": r.image_index,
            "image_url": r.image_url, "image_path": str(ipath), "image_sha256": r.sha256,
            "image_width": meta.get("width"), "image_height": meta.get("height"), "split": r.split,
            "side": r.decision.side, "labelled": ed.labelled, "coverage": round(r.decision.coverage, 4),
            "coverage_semitic": round(r.decision.coverage_semitic, 4),
            "blocks_total": len(ed.blocks), "blocks_kept": r.answer.blocks,
            "lines": r.answer.lines, "regions": r.answer.regions, "countable": r.answer.countable,
            "edition_source": editions[r.pgpid]["source"], "rows": [x["stem"] for _, x in rows]})

    tests_all = [t for r in results for t in r.tests]
    doc_reasons = collections.Counter(d.reason or "eligible" for d in docs.values())
    usable_pages = sum(len(d.urls) for d in docs.values())
    eligible_pages = sum(len(d.urls) for d in eligible.values())
    ev_usable = sum(evidence_path(raw_dir, d.canonical_id, i).exists() for d in docs.values() for i in range(len(d.urls)))
    coverage = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "model": MODEL,
        "usable_documents": len(docs), "usable_pages": usable_pages,
        "usable_pages_with_evidence": ev_usable,
        "usable_evidence_share": round(ev_usable / max(1, usable_pages), 4),
        "eligible_documents": len(eligible), "eligible_pages": eligible_pages,
        "eligible_pages_with_evidence": n_evidence,
        "eligible_evidence_share": round(n_evidence / max(1, eligible_pages), 4),
        "pages_included": len(included), "documents_included": len({r.pgpid for r in included}),
    }
    stats = {
        "coverage": coverage,
        "editions_ge_100_letters": len(editions), "editions_not_usable": dict(ctx.not_usable),
        "document_gate": dict(doc_reasons), "document_gate_info": dict(ctx.gate_info),
        "benchmark_rule": f"join partners, fragment keys, text duplicate = >= {shingle_containment} pair "
                          f"containment of {DECONTAM_SHINGLE}-letter shingles",
        "documents_trained_before": sum(d.trained for d in eligible.values()),
        "val_documents": sum(1 for p in eligible if splits[p] == "val"),
        "pages_evaluated": len(results), "page_outcomes": dict(reasons),
        "pages_dropped_duplicate_side_photo": n_duplicate,
        "pages_included_by_split": dict(collections.Counter(r.split for r in included)),
        "pages_included_labelled": sum(parsed[r.pgpid].labelled for r in included),
        "rows_by_family": dict(fam_counts), "rows_by_split": {k: len(v) for k, v in split_rows.items()},
        "page_letters": sum(n_hebrew(r.answer.text) for r in included),
        "page_lines": sum(len(r.answer.lines) for r in included),
        "block_presence": {
            "rule": f"keep block if {NGRAM}-gram presence in union(VLM lines, Kraken rows) > max over {NULL_K} "
                    f"null pages, >= {MIN_HITS} hits, and a line (>= {ANCHOR_MIN_LETTERS} letters) read at >= "
                    f"{ANCHOR_SIM}; side coverage >= {SIDE_COVERAGE}",
            "blocks_tested": len(tests_all), "blocks_kept": sum(t.kept for t in tests_all),
            "held_out_false_keep_rate": round(sum(t.held_out_kept for t in tests_all) / max(1, len(tests_all)), 5)},
        "null_pool_pages": len(ctx.pool), "served_snapshot": str(served_path),
        "served_snapshot_sha256": hashlib.sha256(served_path.read_bytes()).hexdigest(),
        "build_seconds": round(time.time() - t0, 1),
    }
    evaluated = [{"pgpid": r.pgpid, "canonical_id": r.canonical_id, "image_index": r.image_index, "split": r.split,
                  "reason": r.reason or "included", "side": r.decision.side,
                  "coverage": round(r.decision.coverage, 4), "coverage_semitic": round(r.decision.coverage_semitic, 4),
                  "kept": [t.kept for t in r.tests], "presence": [round(t.share, 3) for t in r.tests],
                  "null_max": [round(t.null_max, 3) for t in r.tests],
                  "block_sides": [b.side for b in parsed[r.pgpid].blocks],
                  "block_kinds": [b.kind for b in parsed[r.pgpid].blocks]} for r in results]
    if download:
        save_dataset(split_rows, out_dir, {
            "stats.json": json.dumps(stats, ensure_ascii=False, indent=1),
            "coverage.json": json.dumps(coverage, indent=1),
            "manifest.jsonl": "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest),
            "pages_evaluated.jsonl": "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in evaluated)})
    logger.info("stats: %s", json.dumps(stats, ensure_ascii=False, indent=1))
    return stats


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    ap.add_argument("--inputs-dir", type=Path, default=DEFAULT_INPUTS)
    ap.add_argument("--served", type=Path, default=DEFAULT_SERVED)
    ap.add_argument("--limit-pages", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="evaluate pages only; no downloads, no dataset")
    ap.add_argument("--shingle-containment", type=float, default=SHINGLE_CONTAINMENT,
                    help="pair containment marking a benchmark text duplicate (0 = any 2 shared shingles)")
    a = ap.parse_args()
    build(a.output_dir, a.images_dir, a.inputs_dir, a.served, limit_pages=a.limit_pages, download=not a.dry_run,
          shingle_containment=a.shingle_containment)


if __name__ == "__main__":
    main()
