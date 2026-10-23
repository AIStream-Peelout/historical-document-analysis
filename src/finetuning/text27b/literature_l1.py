# File name: literature_l1.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""L1 ``domain_reading`` rows from the academic literature (text27b, plan §9.2 / build brief D).

Each row is a completion task on one or two consecutive pages of one work: the user turn holds a
source stub (author, *title*, year, printed page, from the local literature manifest) and the
opening ≈25 % of the text, the assistant turn the remainder. The text is the OCR file's text
(:attr:`pages.PageRec.ocr_text`), never the structuring model's ``full_main_text`` or any other
model-written field (rule 4). ``meta.page_keys`` (also in ``meta.provenance``) lists the chunk's
page keys, which is where the contamination re-scan looks for excluded pages; ``meta.lang_class`` is
the language of the text (EN / FR / DE / HE / AR / OTHER), not a document language.

Pipeline (:func:`build`), per page then per chunk, with the first failing stage as the drop reason:

1. ``no_ocr`` -- no OCR page joined.
2. ``quality:<rule>`` -- the ported ``code/quality.py`` rules on the furniture-stripped OCR text
   (:func:`quality_reasons`: short page, low letter share, fragmented lines, reversed Hebrew,
   Arabic presentation forms, bibliography-like, index-like, plate).
3. ``low_containment`` / ``containment_none`` -- provenance gate: share of ``full_main_text``
   5-word shingles found in the OCR text below ``min_containment`` (0.8), or unmeasurable.
4. ``duplicate`` -- near-duplicate of an earlier page (the same article stored twice).
5. ``excluded`` / ``excluded_via_duplicate`` -- the page key (or a duplicate's) is listed in
   ``excluded_pages.json`` (contamination module; the ±1 window is applied there).
6. ``empty_after_strip`` -- nothing usable left after the project-name strip.
7. chunks of 1-2 consecutive kept pages of one part, ≤ ``max_tokens`` page-text tokens; about
   ``probe_frac`` of each work's pages go to ``split=probe`` (in-run loss probe, never trained);
8. ``work_cap`` -- train rows beyond the per-work token cap (rendered tokens, seeded choice).

Page furniture (running heads found per part, bare page numbers, download stamps) is peeled off
the top and bottom of every page before the quality rules run (:func:`strip_furniture`).

The literature manifest (``logs/text_finetune_27b/literature_manifest.json``, local, git-ignored)
has one entry per work directory with ``work_dir``, ``type``, ``author``, ``title``, ``year``,
``licence``, ``licence_class`` and, for directories whose parts carry different metadata
(collections of articles), a ``parts`` map with the same fields per part
(:func:`build_literature_manifest`). Hand edits survive a rebuild (:func:`merge_manifests`).

Public API
    Quality: :data:`QUALITY_RULES`, :func:`page_features`, :func:`quality_reasons`,
    :func:`training_text`, :func:`page_quality`, :func:`quality_ok` (used by ``rag_l3``).
    Furniture: :func:`norm_head`, :func:`is_page_number_line`, :func:`is_boilerplate_line`,
    :func:`running_heads`, :func:`running_heads_by_part`, :func:`strip_furniture`.
    Manifest: :func:`licence_class`, :func:`format_authors`, :func:`parse_metadata`,
    :func:`find_metadata`, :func:`part_location`, :func:`match_metadata`,
    :func:`build_literature_manifest`, :func:`merge_manifests`, :func:`load_literature_manifest`,
    :func:`source_info`, :func:`default_manifest_path`.
    Rows: :func:`lang_class_of`, :func:`format_pages`, :func:`source_stub`, :func:`split_opening`,
    :class:`PageCand`, :class:`Chunk`, :func:`make_chunks`, :func:`split_long_text`,
    :func:`find_duplicates`, :func:`assign_probe`, :func:`apply_cap`, :func:`load_excluded_pages`,
    :func:`make_row`, :func:`build`.
    CLI: :func:`main` (``manifest`` and ``build`` sub-commands).

CLI::

    .venv/bin/python -m src.finetuning.text27b.literature_l1 manifest [--force]
    .venv/bin/python -m src.finetuning.text27b.literature_l1 build [--excluded-pages F] [--dry-run]
        [--token-cap 300000] [--max-tokens 2048] [--probe-frac 0.03] [--works W ...]
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import json
import logging
import os
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from src.finetuning.text27b import pages as _pages
from src.finetuning.text27b import paths as _paths
from src.finetuning.text27b import rows as _rows
from src.finetuning.text27b import textnorm as _tn

LOG = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------ constants
FAMILY = "domain_reading"
CONDITION = "literature"
SOURCE = "academic_literature_ocr"
DEFAULT_TOKEN_CAP = 300_000
DEFAULT_MAX_TOKENS = 2048
DEFAULT_PROBE_FRAC = 0.03
DEFAULT_OPEN_FRAC = 0.25
DEFAULT_MIN_CONTAINMENT = 0.8
DEFAULT_SEED = 3407
#: Works with fewer kept pages than this get a probe only when ``round(frac * pages) >= 1``.
MIN_PROBE_PAGES = 10
#: Lines peeled from the top and bottom of a page when looking for furniture.
EDGE_LINES = 3
#: A normalised edge line is a running head when it occurs on at least this many pages of a part.
MIN_HEAD_PAGES = 3
#: Shortest normalised running head (letters and spaces) that counts.
MIN_HEAD_CHARS = 4
#: Normalised heads shorter than this must recur on at least :data:`SHORT_HEAD_SHARE` of a part's pages.
SHORT_HEAD_CHARS = 10
SHORT_HEAD_SHARE = 0.25
#: A chunk shorter than this after the project-name strip is dropped.
MIN_CHUNK_CHARS = 200
#: Instruction line of the user turn.
L1_INSTRUCTION = "Continue this passage from a work of scholarship on the Cairo Genizah."
#: Licence classes from least to most restrictive (a collection gets its most restrictive part's class).
LICENCE_ORDER: Tuple[str, ...] = ("pd", "cc-by", "cc-by-nc", "unknown", "copyright")
#: Manifest fields that describe a work or part.
INFO_FIELDS: Tuple[str, ...] = ("type", "author", "title", "year", "licence", "licence_class")
MANIFEST_VERSION = 1

# ------------------------------------------------------------------------------------ quality rules
#: Thresholds of the ported ``code/quality.py`` rules (names = drop reasons).
QUALITY_RULES: Dict[str, float] = {
    "short_page_chars": 400,          # short_page: fewer characters
    "min_letter_share": 0.5,          # low_letters: letters / characters below this
    "fragmented_share": 0.5,          # fragmented_lines: share of lines with < 3 words above this
    "reversed_min_words": 30,         # reversed_hebrew: at least this many Hebrew words ...
    "reversed_final_share": 0.05,     # ... and this share of them starting with a final letter
    "presentation_min_arabic": 50,    # arabic_presentation_forms: at least this many Arabic letters ...
    "presentation_share": 0.2,        # ... and this share of them presentation forms
    "bibliography_year_share": 0.45,  # bibliography_like: share of lines with a 1500-2029 year
    "index_endnum_share": 0.6,        # index_like: share of lines ending in a number
}
QUALITY_REASON_ORDER: Tuple[str, ...] = ("short_page", "low_letters", "fragmented_lines", "reversed_hebrew",
                                         "arabic_presentation_forms", "bibliography_like", "index_like", "plate")
#: ``classification`` value of plate / figure pages.
PLATE_CLASS = "diagram/map"

_HEB = re.compile("[\u0590-\u05ff\ufb1d-\ufb4f]")
_ARA = re.compile("[\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufeff]")
_PRES = re.compile("[\ufb50-\ufdff\ufe70-\ufeff]")
_LAT = re.compile("[A-Za-z\u00c0-\u024f\u1e00-\u1eff]")
_HEBWORD = re.compile("[\u05d0-\u05ea]{2,}")
_FINAL_START = re.compile("^[\u05da\u05dd\u05df\u05e3\u05e5]")
_YEAR = re.compile(r"\b(1[5-9]\d\d|20[0-2]\d)\b")
_ENDNUM = re.compile(r"\d[\d,\u2013\-\s]*$")
_EN_SW = frozenset("the of and to in is that was for with as by on which from his this be are at it were an or not "
                   "their he they had has have but its these also".split())
_FR_SW = frozenset("le la les des et du un une est que qui dans pour sur pas au aux sont par".split())
_DE_SW = frozenset("der die das und ist nicht mit von den dem zu ein eine auch sich im auf".split())
_LATIN_WORD = re.compile(r"[A-Za-z\u00c0-\u024f]+")


def page_features(text: str) -> Dict[str, int]:
    """Counts the quality rules use (port of the ``code/lit_survey.py`` row fields).

    :param text: Page text.
    :returns: ``chars``, ``lat``, ``heb``, ``ara``, ``pres``, ``lines`` (non-empty), ``short_lines``
        (fewer than 3 words), ``heb_words``, ``heb_final_start``, ``year_lines``, ``endnum_lines``.
    """
    text = text or ""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    hw = _HEBWORD.findall(text)
    return {
        "chars": len(text), "lat": len(_LAT.findall(text)), "heb": len(_HEB.findall(text)),
        "ara": len(_ARA.findall(text)), "pres": len(_PRES.findall(text)), "lines": len(lines),
        "short_lines": sum(1 for ln in lines if len(ln.split()) < 3),
        "heb_words": len(hw), "heb_final_start": sum(1 for w in hw if _FINAL_START.match(w)),
        "year_lines": sum(1 for ln in lines if _YEAR.search(ln)),
        "endnum_lines": sum(1 for ln in lines if _ENDNUM.search(ln.strip())),
    }


def quality_reasons(text: str, page_type: Optional[str] = None,
                    rules: Optional[Mapping[str, float]] = None) -> List[str]:
    """Reasons a page fails the L1 quality filter (port of ``code/quality.py``; empty list = keep).

    :param text: Page text (the furniture-stripped OCR text in the L1 pipeline).
    :param page_type: The page's ``classification`` (``diagram/map`` → ``plate``).
    :param rules: Threshold overrides (keys of :data:`QUALITY_RULES`).
    :returns: Failing rules in :data:`QUALITY_REASON_ORDER`.
    """
    r = {**QUALITY_RULES, **(rules or {})}
    f = page_features(text)
    n = max(1, f["lines"])
    letters = f["lat"] + f["heb"] + f["ara"]
    out: List[str] = []
    if f["chars"] < r["short_page_chars"]:
        out.append("short_page")
    if letters < r["min_letter_share"] * max(1, f["chars"]):
        out.append("low_letters")
    if f["short_lines"] / n > r["fragmented_share"]:
        out.append("fragmented_lines")
    if f["heb_words"] >= r["reversed_min_words"] and f["heb_final_start"] / f["heb_words"] > r["reversed_final_share"]:
        out.append("reversed_hebrew")
    if f["ara"] >= r["presentation_min_arabic"] and f["pres"] / f["ara"] > r["presentation_share"]:
        out.append("arabic_presentation_forms")
    if f["year_lines"] / n > r["bibliography_year_share"] and f["chars"] > 0:
        out.append("bibliography_like")
    if f["endnum_lines"] / n > r["index_endnum_share"]:
        out.append("index_like")
    if str(page_type) == PLATE_CLASS:
        out.append("plate")
    return out


# ------------------------------------------------------------------------------------ furniture
_PAGE_NUM_LINE = re.compile(r"^[\s\[\](){}\-\u2010-\u2015\u2212\u2022\u00b7.|*]*(?:\d{1,4}|[ivxlcdm]{1,8})[\s\[\](){}\-\u2010-\u2015\u2212\u2022\u00b7.|*]*$",
                            re.I)
_BOILERPLATE = re.compile(r"(?i)^\s*(?:this content downloaded from|all use subject to|downloaded from\b|"
                          r"via free access\b|brought to you by\b|terms (?:and conditions )?of use\b|"
                          r"electronic copy available|for personal use only|provided by\b.{0,40}\b(?:repository|library)\b)")
_HEAD_DROP = re.compile(r"[^\w\s]|\d|_")
_SPACES = re.compile(r"\s+")
_MANY_BLANKS = re.compile(r"\n{3,}")


def norm_head(line: str) -> str:
    """Normalise a line for running-head matching (lower case, no digits or punctuation, single spaces).

    :param line: A text line.
    :returns: The normalised form ('' when nothing is left).
    """
    return _SPACES.sub(" ", _HEAD_DROP.sub(" ", (line or "").lower())).strip()


def is_page_number_line(line: str) -> bool:
    """Whether a line is only a page number (Arabic or Roman digits, optional brackets / dashes).

    :param line: A text line.
    :returns: True for lines such as ``12``, ``— 12 —``, ``[xiv]``.
    """
    return bool(line.strip()) and bool(_PAGE_NUM_LINE.match(line))


def is_boilerplate_line(line: str) -> bool:
    """Whether a line is a download / licence stamp printed on every page of a PDF.

    :param line: A text line.
    :returns: True for stamps such as ``Downloaded from ...`` or ``via free access``.
    """
    return bool(_BOILERPLATE.match(line or ""))


def _outer_lines(text: str, edge: int) -> List[str]:
    """The outermost line on each side of a page once page numbers and download stamps are peeled off.

    :param text: Page text.
    :param edge: Most lines peeled per side.
    :returns: Up to two lines (top, then bottom; one line when they coincide).
    """
    lines = [ln for ln in text.split("\n") if ln.strip()]
    out: List[int] = []
    for seq in (range(len(lines)), range(len(lines) - 1, -1, -1)):
        peeled = 0
        for i in seq:
            if peeled < edge and (is_page_number_line(lines[i]) or is_boilerplate_line(lines[i])):
                peeled += 1
                continue
            out.append(i)
            break
    return [lines[i] for i in sorted(set(out))]


def running_heads(texts: Iterable[str], min_pages: int = MIN_HEAD_PAGES, edge: int = EDGE_LINES,
                  min_chars: int = MIN_HEAD_CHARS, short_chars: int = SHORT_HEAD_CHARS,
                  short_share: float = SHORT_HEAD_SHARE) -> FrozenSet[str]:
    """Running heads of one part: normalised outermost lines repeated on at least ``min_pages`` pages.

    Only the outermost line of each side counts (after peeling page numbers and download stamps);
    lines naming a shelfmark are content and never count.
    A short head (fewer than ``short_chars`` normalised characters) must also recur on at least
    ``short_share`` of the part's pages, so edition labels (``Verso``, ``Margin``), ``Ibid.`` lines
    and shelfmark headings that collapse once digits are removed are not taken for heads.

    :param texts: The OCR texts of one part's pages.
    :param min_pages: Pages a line must occur on.
    :param edge: Most lines peeled per side before the outermost line.
    :param min_chars: Shortest normalised head.
    :param short_chars: Heads shorter than this need ``short_share``.
    :param short_share: Share of the part's pages a short head must occur on.
    :returns: Normalised heads (see :func:`norm_head`).
    """
    counts: collections.Counter = collections.Counter()
    n = 0
    for t in texts:
        n += 1
        counts.update({h for h in (norm_head(ln) for ln in _outer_lines(t or "", edge) if not _tn.find_shelfmarks(ln))
                       if len(h) >= min_chars})
    return frozenset(h for h, c in counts.items()
                     if c >= min_pages and (len(h) >= short_chars or c >= short_share * n))


def running_heads_by_part(pages: Iterable[_pages.PageRec], min_pages: int = MIN_HEAD_PAGES,
                          edge: int = EDGE_LINES) -> Dict[Tuple[str, str], FrozenSet[str]]:
    """Running heads per ``(work, part)``, from the OCR text (else ``full_main_text``).

    :param pages: Literature pages.
    :param min_pages: Pages a line must occur on.
    :param edge: Lines per side considered.
    :returns: ``{(work, part): heads}``.
    """
    by_part: Dict[Tuple[str, str], List[str]] = collections.defaultdict(list)
    for p in pages:
        by_part[(p.work, p.part)].append(p.ocr_text or p.full_main_text or "")
    return {k: running_heads(v, min_pages, edge) for k, v in by_part.items()}


def strip_furniture(text: str, heads: Iterable[str] = (), edge: int = EDGE_LINES) -> Tuple[str, List[str]]:
    """Peel page furniture off the top and bottom of a page.

    From each side, while fewer than ``edge`` lines were removed, the outermost non-empty line is
    removed when it is a download stamp (``boilerplate``), a bare page number (``page_number``) or a
    running head of the part (``running_head``; a head with its page number, ``22 Title``, matches
    too; a line naming a shelfmark never does). Lines are right-stripped and runs of blank lines
    collapsed.

    :param text: Page text.
    :param heads: Normalised running heads of the part (:func:`running_heads`).
    :param edge: Most lines removed per side.
    :returns: ``(text, ops)`` with one op name per removed line.
    """
    head_set = set(heads)
    lines = [ln.rstrip() for ln in (text or "").split("\n")]
    ops: List[str] = []

    def classify(line: str) -> Optional[str]:
        """Furniture kind of a line, or None.

        :param line: A text line.
        :returns: Op name or None.
        """
        if is_boilerplate_line(line):
            return "boilerplate"
        if is_page_number_line(line):
            return "page_number"
        if head_set and norm_head(line) in head_set and not _tn.find_shelfmarks(line):
            return "running_head"
        return None

    for from_top in (True, False):
        removed = 0
        while removed < edge:
            idx = [i for i, ln in enumerate(lines) if ln.strip()]
            if not idx:
                break
            i = idx[0] if from_top else idx[-1]
            kind = classify(lines[i])
            if kind is None:
                break
            ops.append(kind)
            del lines[i]
            removed += 1
    out = _MANY_BLANKS.sub("\n\n", "\n".join(lines)).strip()
    return out, ops


def training_text(page: _pages.PageRec, heads: Iterable[str] = (), use_ocr: bool = True) -> Tuple[str, List[str]]:
    """The text a page contributes: its OCR text (or ``full_main_text``) with the furniture stripped.

    :param page: A literature page.
    :param heads: Running heads of the page's part.
    :param use_ocr: Use the OCR text (the L1 rule); False uses ``full_main_text``.
    :returns: ``(text, furniture ops)``; ``('', [])`` when the chosen text is missing.
    """
    raw = page.ocr_text if use_ocr else page.full_main_text
    if not raw:
        return "", []
    return strip_furniture(raw, heads)


def page_quality(page: _pages.PageRec, heads: Optional[Iterable[str]] = None, use_ocr: bool = True,
                 rules: Optional[Mapping[str, float]] = None) -> List[str]:
    """Quality reasons of a page (``no_ocr`` when the OCR text is wanted but missing).

    :param page: A literature page.
    :param heads: Running heads of the page's part (None = page-local furniture only).
    :param use_ocr: Judge the OCR text (L1 targets) rather than ``full_main_text``.
    :param rules: Threshold overrides.
    :returns: Failing reasons (empty = keep).
    """
    text, _ = training_text(page, heads or (), use_ocr)
    if use_ocr and not text:
        return ["no_ocr"]
    return quality_reasons(text, page.page_type, rules)


def quality_ok(page: _pages.PageRec, heads: Optional[Iterable[str]] = None, use_ocr: bool = True) -> bool:
    """Whether a page passes the L1 quality filter (the rule ``rag_l3`` applies to context pages).

    :param page: A literature page.
    :param heads: Running heads of the page's part (:func:`running_heads_by_part`); None strips only
        page-local furniture (page numbers, download stamps).
    :param use_ocr: Judge the OCR text (default; a page without OCR fails) or ``full_main_text``.
    :returns: True when no rule fails.
    """
    return not page_quality(page, heads, use_ocr)


# ------------------------------------------------------------------------------------ manifest
_CC = re.compile(r"(?i)\bCC[\s-]*BY\b|creative\s+commons|\battribution\b")
_NC = re.compile(r"(?i)\bNC\b|non[\s-]?commercial")
_PD = re.compile(r"(?i)public\s+domain|\bCC0\b|\bPDM\b")
_COPY = re.compile(r"(?i)\u00a9|\bcopyright\b|all rights reserved|\brights\b")
_YEAR4 = re.compile(r"(1[4-9]\d\d|20\d\d)")


def licence_class(licence: Optional[str]) -> str:
    """Map a licence statement to a class in :data:`rows.LICENCE_CLASSES`.

    ``public domain`` / CC0 → ``pd``; Creative Commons with NC → ``cc-by-nc``; other Creative Commons
    (BY, BY-SA, BY-ND) → ``cc-by``; a statement naming rights or copyright → ``copyright``;
    anything else (access notes such as "free on <site>", None) → ``unknown``.

    :param licence: Raw licence text.
    :returns: The class.
    """
    if not licence:
        return "unknown"
    s = str(licence)
    if _PD.search(s):
        return "pd"
    if _CC.search(s):
        return "cc-by-nc" if _NC.search(s) else "cc-by"
    if _COPY.search(s):
        return "copyright"
    return "unknown"


def format_authors(names: Sequence[str]) -> Optional[str]:
    """Author string for a source stub: ``A``, ``A and B``, ``A, B and C``, ``A et al.`` (four or more).

    :param names: Author names.
    :returns: The string, or None for no names.
    """
    names = [str(n).strip() for n in names if str(n or "").strip()]
    if not names:
        return None
    if len(names) == 1:
        return names[0]
    if len(names) <= 3:
        return ", ".join(names[:-1]) + " and " + names[-1]
    return names[0] + " et al."


def _names(value: Any) -> List[str]:
    """Names from an ``authors`` / ``author`` / ``editors`` value (string, dict with ``name``, or list).

    :param value: Raw value.
    :returns: Names.
    """
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [str(value["name"])] if value.get("name") else []
    if isinstance(value, list):
        return [n for v in value for n in _names(v)]
    return []


def _year_of(*values: Any) -> Optional[int]:
    """First plausible four-digit year among values.

    :param values: Candidates (ints or strings).
    :returns: The year or None.
    """
    for v in values:
        if v is None or isinstance(v, bool):
            continue
        m = _YEAR4.search(str(v))
        if m:
            return int(m.group(1))
    return None


def _work_type(raw: Optional[str]) -> Optional[str]:
    """Normalise a publication type (``Journal Article`` → ``article``, ``PhD Dissertation`` → ``thesis``).

    :param raw: Raw type.
    :returns: ``book`` / ``chapter`` / ``article`` / ``thesis`` / the lower-cased raw value / None.
    """
    if not raw:
        return None
    s = str(raw).lower().replace("_", " ")
    for key, name in (("dissert", "thesis"), ("thesis", "thesis"), ("chapter", "chapter"), ("article", "article"),
                      ("lecture", "article"), ("book", "book"), ("volume", "book")):
        if key in s:
            return name
    return s


def _licence_text(d: Mapping[str, Any]) -> Optional[str]:
    """The licence statement of a metadata record (``license``, ``source.licence``, restriction notes).

    :param d: Metadata JSON.
    :returns: The statement, or None when the record gives none.
    """
    src = d.get("source") if isinstance(d.get("source"), dict) else {}
    for v in (d.get("license"), d.get("licence"), src.get("licence"), src.get("license")):
        if isinstance(v, str) and v.strip():
            return v.strip()
    for k in ("access_restrictions", "sharing_restrictions"):
        v = d.get(k)
        if isinstance(v, str) and (_CC.search(v) or _PD.search(v) or _COPY.search(v)):
            return v.strip()
    return None


def parse_metadata(d: Mapping[str, Any]) -> Dict[str, Any]:
    """Source fields of a ``*_metadata.json`` record.

    :param d: Metadata JSON.
    :returns: ``{type, author, title, year, licence, licence_class}`` (None where absent).
    """
    pub = d.get("publication") if isinstance(d.get("publication"), dict) else {}
    authors = _names(d.get("authors")) or _names(d.get("author"))
    author = format_authors(authors)
    if author is None:
        eds = _names(d.get("editors"))
        author = (format_authors(eds) + " (ed.)") if eds else None
    title = d.get("title") if isinstance(d.get("title"), str) and d.get("title").strip() else None
    lic = _licence_text(d)
    return {"type": _work_type(pub.get("type") or d.get("type")), "author": author,
            "title": title.strip() if title else None,
            "year": _year_of(pub.get("year"), pub.get("date"), d.get("publication_date"), d.get("year")),
            "licence": lic, "licence_class": licence_class(lic)}


@dataclass(frozen=True)
class MetaFile:
    """One ``*_metadata.json`` of a work directory.

    ``rel_dir`` is its directory relative to the work directory ('' = the work root), ``stem`` the
    file name without ``_metadata.json``.
    """

    rel_dir: str
    stem: str
    path: Path
    info: Dict[str, Any] = field(hash=False, compare=False)


def find_metadata(work_dir: Path) -> List[MetaFile]:
    """All readable ``*_metadata.json`` files of a work (structuring / image directories skipped).

    :param work_dir: Work directory.
    :returns: Parsed metadata files sorted by path (unreadable or non-object files are logged and skipped).
    """
    out: List[MetaFile] = []
    for dirpath, dirs, files in os.walk(work_dir):
        dirs[:] = sorted(d for d in dirs if "_structured" not in d and not d.endswith("_images"))
        for f in sorted(files):
            if not f.endswith("_metadata.json"):
                continue
            path = Path(dirpath) / f
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, ValueError) as exc:
                LOG.warning("unreadable metadata %s: %s", path.name, exc)
                continue
            if not isinstance(data, dict):
                continue
            rel = Path(dirpath).relative_to(work_dir).as_posix()
            out.append(MetaFile("" if rel == "." else rel, f[: -len("_metadata.json")], path, parse_metadata(data)))
    return out


def part_location(page: _pages.PageRec, root: Path) -> Tuple[str, str]:
    """Directory of a page's part (relative to its work) and the part's file stem.

    :param page: A literature page.
    :param root: Literature root.
    :returns: ``(rel_dir, stem)``; ``rel_dir`` is '' when the structuring directory sits in the work root.
    """
    comps = page.structuring_dir.split("/")
    loc = Path(page.path).parents[len(comps)]
    rel = loc.relative_to(Path(root) / page.work).as_posix()
    return ("" if rel == "." else rel), re.sub(r"_structured.*$", "", comps[0])


def _norm_stem(s: str) -> str:
    """Stem form used for metadata matching (lower case, ``-`` → ``_``).

    :param s: File stem.
    :returns: Normalised stem.
    """
    return s.lower().replace("-", "_")


def _is_ancestor(meta_dir: str, loc: str) -> bool:
    """Whether ``meta_dir`` is ``loc`` or one of its ancestors (both relative to the work).

    :param meta_dir: Metadata directory.
    :param loc: Part directory.
    :returns: True when the metadata directory contains the part.
    """
    return meta_dir == "" or meta_dir == loc or loc.startswith(meta_dir + "/")


def match_metadata(loc: str, stem: str, metas: Sequence[MetaFile]) -> Optional[MetaFile]:
    """The metadata file that specifically describes a part.

    A file in the part's directory or an ancestor matches when its stem equals the part's stem or
    one is a prefix of the other (≥ 4 characters); a file in the part's own sub-directory matches
    regardless of its stem. Best = exact stem, then prefix, then own directory; deeper wins ties.

    :param loc: Part directory (:func:`part_location`).
    :param stem: Part stem.
    :param metas: The work's metadata files.
    :returns: The matching file or None.
    """
    best: Optional[Tuple[int, int, str]] = None
    chosen: Optional[MetaFile] = None
    b = _norm_stem(stem)
    for m in metas:
        if not _is_ancestor(m.rel_dir, loc):
            continue
        a = _norm_stem(m.stem)
        if a == b:
            score = 3
        elif min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)):
            score = 2
        elif m.rel_dir and m.rel_dir == loc:
            score = 1
        else:
            continue
        key = (score, m.rel_dir.count("/") + (1 if m.rel_dir else 0), str(m.path))
        if best is None or key > best:
            best, chosen = key, m
    return chosen


def _most_restrictive(classes: Iterable[str]) -> str:
    """Most restrictive licence class (see :data:`LICENCE_ORDER`).

    :param classes: Classes.
    :returns: The class (``unknown`` for none).
    """
    cs = [c for c in classes if c in LICENCE_ORDER]
    return max(cs, key=LICENCE_ORDER.index) if cs else "unknown"


def _empty_info() -> Dict[str, Any]:
    """An info block with every field unknown.

    :returns: ``{type: None, ..., licence_class: 'unknown'}``.
    """
    return {"type": None, "author": None, "title": None, "year": None, "licence": None, "licence_class": "unknown"}


def build_literature_manifest(root: Path, pages: Optional[Sequence[_pages.PageRec]] = None) -> Dict[str, Any]:
    """Build the literature manifest from the work directories' metadata files.

    Each part gets the metadata file that specifically matches it (:func:`match_metadata`); a part
    without one falls back to the work's only metadata file, else to the single unclaimed file at
    the deepest directory containing it. A work whose parts share one record carries that record;
    one whose parts differ is ``type = collection`` with a ``parts`` map and the most restrictive
    licence class. Directories without structured pages (derived or metadata folders) are
    ``type = no_pages``.

    :param root: Literature root.
    :param pages: Pages (default: :func:`pages.iter_pages` without OCR).
    :returns: ``{"version", "generated", "works": {work_dir: entry}}`` with paths relative to ``root``.
    """
    root = Path(root)
    pages = list(pages) if pages is not None else _pages.iter_pages(root, with_ocr=False)
    parts: Dict[str, Dict[str, Tuple[str, str]]] = collections.defaultdict(dict)
    n_pages: collections.Counter = collections.Counter()
    for p in pages:
        n_pages[(p.work, p.part)] += 1
        if p.part not in parts[p.work]:
            parts[p.work][p.part] = part_location(p, root)
    works: Dict[str, Any] = {}
    for wd in sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")):
        metas = find_metadata(root / wd)
        rel = {m.path: m.path.relative_to(root).as_posix() for m in metas}
        wparts = parts.get(wd, {})
        if not wparts:
            only = metas[0].info if len(metas) == 1 else _empty_info()
            works[wd] = {"work_dir": wd, **only, "type": "no_pages", "n_parts": 0, "n_pages": 0,
                         "metadata_files": sorted(rel.values())}
            continue
        chosen: Dict[str, Optional[MetaFile]] = {pt: match_metadata(loc, stem, metas) for pt, (loc, stem) in wparts.items()}
        claimed = {m.path for m in chosen.values() if m is not None}
        for pt, (loc, _stem) in wparts.items():
            if chosen[pt] is not None:
                continue
            if len(metas) == 1:
                chosen[pt] = metas[0]
                continue
            anc = [m for m in metas if m.path not in claimed and _is_ancestor(m.rel_dir, loc)]
            if anc:
                depth = max(m.rel_dir.count("/") + (1 if m.rel_dir else 0) for m in anc)
                at = [m for m in anc if m.rel_dir.count("/") + (1 if m.rel_dir else 0) == depth]
                chosen[pt] = at[0] if len(at) == 1 else None
        used = {m.path for m in chosen.values() if m is not None}
        entry: Dict[str, Any] = {"work_dir": wd}
        if len(used) == 1 and all(m is not None for m in chosen.values()):
            m0 = next(m for m in chosen.values() if m is not None)
            entry.update(m0.info)
            entry["metadata_file"] = rel[m0.path]
        else:
            entry.update(_empty_info())
            entry["type"] = "collection"
            entry["parts"] = {}
            for pt in sorted(wparts):
                m = chosen[pt]
                pinfo = dict(m.info) if m is not None else _empty_info()
                pinfo["metadata_file"] = rel[m.path] if m is not None else None
                entry["parts"][pt] = pinfo
            entry["licence_class"] = _most_restrictive(v["licence_class"] for v in entry["parts"].values())
        entry["n_parts"] = len(wparts)
        entry["n_pages"] = sum(n for (w, _), n in n_pages.items() if w == wd)
        entry["metadata_files"] = sorted(rel.values())
        works[wd] = entry
    return {"version": MANIFEST_VERSION, "generated": _dt.date.today().isoformat(), "works": works}


def _known(value: Any) -> bool:
    """Whether a manifest value carries information (not None / '' / ``unknown``).

    :param value: Field value.
    :returns: True when set.
    """
    return value not in (None, "", "unknown")


def merge_manifests(new: Mapping[str, Any], old: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Merge a rebuilt manifest onto an existing one, keeping hand-set values.

    For every work (and part) present in both, an informative old value of an :data:`INFO_FIELDS`
    field wins over the rebuilt one; works / parts only in ``old`` are kept.

    :param new: Freshly built manifest.
    :param old: Existing manifest (None = nothing to keep).
    :returns: The merged manifest.
    """
    out = json.loads(json.dumps(new))
    if not old:
        return out
    owork = old.get("works") or {}
    for wd, oe in owork.items():
        ne = out["works"].get(wd)
        if ne is None:
            out["works"][wd] = oe
            continue
        for k in INFO_FIELDS:
            if _known(oe.get(k)):
                ne[k] = oe[k]
        oparts = oe.get("parts") or {}
        if oparts:
            nparts = ne.setdefault("parts", {})
            for pt, op in oparts.items():
                np_ = nparts.setdefault(pt, dict(op))
                for k in INFO_FIELDS:
                    if _known(op.get(k)):
                        np_[k] = op[k]
    return out


def default_manifest_path(paths: Optional[_paths.Paths] = None) -> Path:
    """Location of the literature manifest: ``paths.literature_manifest`` or ``<out_dir>/../literature_manifest.json``.

    :param paths: Build configuration (default :func:`paths.load_paths`).
    :returns: The path.
    """
    p = paths or _paths.load_paths()
    explicit = p.get("literature_manifest")
    return Path(explicit) if explicit else p.out_dir.parent / "literature_manifest.json"


def load_literature_manifest(path: Optional[Union[str, Path]]) -> Dict[str, Any]:
    """Read the literature manifest (missing file → no works, logged).

    :param path: Manifest path.
    :returns: The manifest dict.
    """
    if path is None or not Path(path).exists():
        LOG.warning("literature manifest %s missing: source stubs will be empty", path)
        return {"works": {}}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def source_info(manifest: Mapping[str, Any], work: str, part: str) -> Dict[str, Any]:
    """Source fields of a page's work / part.

    :param manifest: Literature manifest.
    :param work: Work directory.
    :param part: Part path.
    :returns: Info block (``parts[part]`` when present, else the work entry; unknowns when absent).
    """
    entry = (manifest.get("works") or {}).get(work) or {}
    pinfo = (entry.get("parts") or {}).get(part)
    src = pinfo if pinfo else entry
    info = _empty_info()
    for k in INFO_FIELDS:
        if src.get(k) is not None:
            info[k] = src[k]
    if info["licence_class"] not in _rows.LICENCE_CLASSES:
        info["licence_class"] = licence_class(info.get("licence"))
    return info


# ------------------------------------------------------------------------------------ rows
def lang_class_of(text: str) -> str:
    """Language class of a literature text: ``HE`` / ``AR`` (dominant script), else ``EN`` / ``FR`` / ``DE`` / ``OTHER``.

    :param text: Text.
    :returns: The class.
    """
    heb, ara, lat = len(_HEB.findall(text)), len(_ARA.findall(text)), len(_LAT.findall(text))
    if heb > lat and heb >= ara:
        return "HE"
    if ara > lat:
        return "AR"
    words = [w.lower() for w in _LATIN_WORD.findall(text)]
    scores = {"EN": sum(w in _EN_SW for w in words), "FR": sum(w in _FR_SW for w in words),
              "DE": sum(w in _DE_SW for w in words)}
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "OTHER"


def format_pages(printed: Sequence[Optional[int]], page_nos: Sequence[int]) -> Optional[str]:
    """Printed-page label of a chunk (``p. 12`` / ``pp. 12–13``), filling one unknown side of a pair.

    :param printed: Printed page per chunk page (None = unknown).
    :param page_nos: PDF page index per chunk page (consecutive pages fill each other).
    :returns: The label or None.
    """
    vals = list(printed)
    if len(vals) == 2 and page_nos[1] == page_nos[0] + 1:
        if vals[0] is None and vals[1] is not None:
            vals[0] = vals[1] - 1
        elif vals[1] is None and vals[0] is not None:
            vals[1] = vals[0] + 1
    known = [v for v in vals if v is not None and v > 0]
    if not known:
        return None
    if len(known) == 1 or known[0] == known[-1]:
        return f"p. {known[0]}"
    return f"pp. {known[0]}\u2013{known[-1]}"


def source_stub(info: Mapping[str, Any], pages_label: Optional[str]) -> str:
    """Source line in the site's citation shape: ``{author}, *{title}*, {year}, p. {page}`` (unknown parts left out).

    :param info: Source info (:func:`source_info`).
    :param pages_label: ``p. N`` / ``pp. N–M`` or None.
    :returns: The stub ('' when nothing is known).
    """
    bits: List[str] = []
    if info.get("author"):
        bits.append(str(info["author"]))
    if info.get("title"):
        bits.append(f"*{info['title']}*")
    if info.get("year"):
        bits.append(str(info["year"]))
    if bits and pages_label:
        bits.append(pages_label)
    return ", ".join(bits)


_SENT_END = re.compile(r"[.!?;:][\"\u201d\u2019\u00bb)\]]*(?=\s)")


def split_opening(text: str, frac: float = DEFAULT_OPEN_FRAC, window: float = 0.10) -> Tuple[str, str]:
    """Split a chunk into the opening (user turn) and the remainder (assistant turn).

    The cut lies within ``frac ± window`` of the characters, preferring a sentence end, then a
    line break, then a space, each nearest to ``frac``; else a hard cut at ``frac``.

    :param text: Chunk text.
    :param frac: Target share of the opening.
    :param window: Allowed deviation (share of characters).
    :returns: ``(opening, rest)``, both stripped.
    """
    n = len(text)
    target = int(n * frac)
    lo, hi = int(n * max(0.0, frac - window)), int(n * min(1.0, frac + window))

    def nearest(cands: List[int]) -> Optional[int]:
        """Candidate closest to the target.

        :param cands: Cut positions.
        :returns: The closest position or None.
        """
        inside = [c for c in cands if lo <= c <= hi and 0 < c < n]
        return min(inside, key=lambda c: (abs(c - target), c)) if inside else None

    cut = nearest([m.end() for m in _SENT_END.finditer(text)])
    if cut is None:
        cut = nearest([m.end() for m in re.finditer(r"\n", text)])
    if cut is None:
        cut = nearest([m.start() for m in re.finditer(r"\s", text)])
    if cut is None:
        cut = max(1, target)
    return text[:cut].strip(), text[cut:].strip()


@dataclass
class PageCand:
    """A page that passed the page-level gates, with its training text."""

    key: str
    work: str
    part: str
    page_no: int
    text: str
    tokens: int
    printed: Optional[int]
    ocr_source: Optional[str]
    containment: Optional[float]
    ops: List[str] = field(default_factory=list)


@dataclass
class Chunk:
    """One to two consecutive pages of a part (or a piece of one long page)."""

    pages: List[PageCand]
    text: str
    tokens: int
    piece: Optional[int] = None

    @property
    def key(self) -> str:
        """Stable chunk key: page keys joined by ``+`` (``#n`` for a piece of a long page).

        :returns: The key.
        """
        k = "+".join(p.key for p in self.pages)
        return k if self.piece is None else f"{k}#{self.piece}"

    @property
    def work(self) -> str:
        """Work directory of the chunk.

        :returns: The work.
        """
        return self.pages[0].work


def split_long_text(text: str, max_tokens: int, count_batch: Callable[[Sequence[str]], List[int]]) -> List[str]:
    """Split a text that exceeds ``max_tokens`` into consecutive line groups that fit.

    :param text: Text.
    :param max_tokens: Token budget per piece.
    :param count_batch: Token counter for a list of texts.
    :returns: Pieces (a single line over budget becomes its own piece).
    """
    lines = text.split("\n")
    counts = count_batch(lines)
    pieces: List[str] = []
    cur: List[str] = []
    cur_t = 0
    for ln, t in zip(lines, counts):
        if cur and cur_t + t + 1 > max_tokens:
            pieces.append("\n".join(cur).strip())
            cur, cur_t = [], 0
        cur.append(ln)
        cur_t += t + 1
    if cur:
        pieces.append("\n".join(cur).strip())
    return [p for p in pieces if p]


def make_chunks(cands: Sequence[PageCand], max_tokens: int,
                count_batch: Callable[[Sequence[str]], List[int]]) -> List[Chunk]:
    """Group kept pages into chunks: a page joins the next one of the same part when they are
    consecutive and together fit ``max_tokens``; a page alone over budget is split by lines.

    :param cands: Kept pages sorted by work, part, page number.
    :param max_tokens: Page-text token budget per chunk.
    :param count_batch: Token counter for a list of texts.
    :returns: Chunks in page order.
    """
    out: List[Chunk] = []
    i = 0
    while i < len(cands):
        a = cands[i]
        if a.tokens > max_tokens:
            for j, piece in enumerate(split_long_text(a.text, max_tokens, count_batch)):
                out.append(Chunk([a], piece, count_batch([piece])[0], piece=j))
            i += 1
            continue
        if i + 1 < len(cands):
            b = cands[i + 1]
            if ((b.work, b.part) == (a.work, a.part) and b.page_no == a.page_no + 1
                    and a.tokens + b.tokens + 2 <= max_tokens):
                out.append(Chunk([a, b], a.text + "\n\n" + b.text, a.tokens + b.tokens + 2))
                i += 2
                continue
        out.append(Chunk([a], a.text, a.tokens))
        i += 1
    return out


def _shingle_hashes(text: str, n: int) -> Set[int]:
    """CRC32 hashes of lower-cased word ``n``-grams.

    :param text: Text.
    :param n: Words per shingle.
    :returns: Hash set.
    """
    words = re.findall(r"\w+", (text or "").lower())
    return {zlib.crc32(" ".join(words[i:i + n]).encode("utf-8")) for i in range(len(words) - n + 1)}


def find_duplicates(texts: Mapping[str, str], n: int = 8, sample_mod: int = 8, min_shared: int = 3,
                    threshold: float = 0.8) -> Dict[str, str]:
    """Near-duplicate pages: a page whose ``n``-word shingles lie ≥ ``threshold`` inside an earlier page.

    Candidates come from sampled shingles (hash mod ``sample_mod`` == 0) shared at least
    ``min_shared`` times; containment is then checked on the full shingle sets.

    :param texts: ``{page_key: text}`` in priority order (earlier pages are kept).
    :param n: Words per shingle.
    :param sample_mod: Sampling modulus for the candidate index.
    :param min_shared: Sampled shingles a candidate must share.
    :param threshold: Containment that makes a duplicate.
    :returns: ``{duplicate_key: kept_key}``.
    """
    sh = {k: _shingle_hashes(t, n) for k, t in texts.items()}
    inverted: Dict[int, List[str]] = collections.defaultdict(list)
    dup: Dict[str, str] = {}
    for k, s in sh.items():
        if not s:
            continue
        sampled = [h for h in s if h % sample_mod == 0]
        cand: collections.Counter = collections.Counter()
        for h in sampled:
            cand.update(inverted.get(h, ()))
        for other, c in cand.most_common():
            if c < min_shared:
                break
            if len(s & sh[other]) / len(s) >= threshold:
                dup[k] = other
                break
        if k not in dup:
            for h in sampled:
                inverted[h].append(k)
    return dup


def _hash_order(key: str, seed: int, salt: str) -> str:
    """Deterministic pseudo-random sort key.

    :param key: Item key.
    :param seed: Seed.
    :param salt: Purpose tag (probe / cap).
    :returns: Hex digest.
    """
    return hashlib.sha1(f"{seed}|{salt}|{key}".encode("utf-8")).hexdigest()


def assign_probe(chunks: Sequence[Chunk], frac: float = DEFAULT_PROBE_FRAC, seed: int = DEFAULT_SEED,
                 min_pages: int = MIN_PROBE_PAGES) -> Set[str]:
    """Choose each work's probe chunks: page groups in seeded order until ``round(frac × pages)`` pages.

    The pieces of one long page form one group, so a page is never split between probe and train.
    ``frac <= 0`` disables the probe. A work with fewer than ``min_pages`` kept pages gets a probe only when the rounded count is ≥ 1;
    a work with at least ``min_pages`` gets at least one probe page.

    :param chunks: All kept chunks.
    :param frac: Share of each work's pages.
    :param seed: Seed.
    :param min_pages: Work size from which one probe page is guaranteed.
    :returns: Keys of probe chunks.
    """
    by_work: Dict[str, Dict[str, List[Chunk]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for c in chunks:
        by_work[c.work]["+".join(p.key for p in c.pages)].append(c)
    probe: Set[str] = set()
    if frac <= 0:
        return probe
    for work, groups in by_work.items():
        n_pages = len({p.key for cs in groups.values() for c in cs for p in c.pages})
        want = round(frac * n_pages)
        if n_pages >= min_pages:
            want = max(1, want)
        got = 0
        for base in sorted(groups, key=lambda b: _hash_order(b, seed, "probe")):
            if got >= want:
                break
            probe.update(c.key for c in groups[base])
            got += len(groups[base][0].pages)
    return probe


def apply_cap(rows: Sequence[Dict[str, Any]], cap: Optional[int], seed: int = DEFAULT_SEED,
              group_key: str = "work") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Per-work token cap on train rows: rows in seeded order are kept while the work's total stays ≤ ``cap``.

    :param rows: Rows (only ``split == train`` rows are capped; others pass through).
    :param cap: Token cap per work (``meta.tokens``); None or ≤ 0 disables the cap.
    :param seed: Seed.
    :param group_key: Meta field that names the work.
    :returns: ``(kept rows in input order, dropped rows)``.
    """
    if not cap or cap <= 0:
        return list(rows), []
    order = sorted((r for r in rows if r["meta"]["split"] == "train"),
                   key=lambda r: _hash_order(r["meta"]["row_id"], seed, "cap"))
    used: collections.Counter = collections.Counter()
    drop: Set[str] = set()
    for r in order:
        w, t = r["meta"][group_key], int(r["meta"]["tokens"])
        if used[w] + t <= cap:
            used[w] += t
        else:
            drop.add(r["meta"]["row_id"])
    kept = [r for r in rows if r["meta"]["row_id"] not in drop]
    dropped = [r for r in rows if r["meta"]["row_id"] in drop]
    return kept, dropped


def load_excluded_pages(path: Optional[Union[str, Path]]) -> Dict[str, List[str]]:
    """Read ``excluded_pages.json`` (``{page_key: [reasons]}``; a ``pages`` / ``excluded`` wrapper or a key list also work).

    :param path: File, or None.
    :returns: ``{page_key: reasons}`` (empty when no file is given).
    :raises FileNotFoundError: When a path is given but does not exist.
    """
    if path is None:
        return {}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        for k in ("pages", "excluded"):
            if isinstance(data.get(k), (dict, list)):
                data = data[k]
                break
    if isinstance(data, list):
        return {str(k): ["listed"] for k in data}
    return {str(k): (list(v) if isinstance(v, (list, tuple)) else [str(v)]) for k, v in data.items()}


def make_row(chunk: Chunk, info: Mapping[str, Any], split: str, open_frac: float = DEFAULT_OPEN_FRAC,
             source_scrub_hits: int = 0) -> Dict[str, Any]:
    """Build one ``domain_reading`` row (token fields left at 0; :func:`build` fills them).

    :param chunk: The chunk (text already furniture- and project-name-stripped).
    :param info: Source info of the chunk's work / part.
    :param split: ``train`` or ``probe``.
    :param open_frac: Opening share placed in the user turn.
    :param source_scrub_hits: project-name strings removed from the chunk.
    :returns: The row.
    """
    opening, rest = split_opening(chunk.text, open_frac)
    label = format_pages([p.printed for p in chunk.pages], [p.page_no for p in chunk.pages])
    stub = source_stub(info, label)
    user = L1_INSTRUCTION + (f"\n\nSource: {stub}" if stub else "") + "\n\n" + opening
    ops: List[str] = []
    for p in chunk.pages:
        ops.extend(o for o in p.ops if o not in ops)
    if source_scrub_hits:
        ops.append(_tn.SCRUB_OP)
    first, last = chunk.pages[0], chunk.pages[-1]
    meta = _rows.new_meta(
        FAMILY, CONDITION, split, chunk.key,
        lang_class=lang_class_of(chunk.text), pgpid=None, shelfmark=None, shelfmark_group=None,
        held_out_vision_benchmark=False, target_chars=len(rest), target_tokens=0, tokens=0, input_cer=None,
        input_source="ocr:" + "+".join(sorted({p.ocr_source or "unknown" for p in chunk.pages})),
        input_truncated=False, licence=info.get("licence"), licence_class=info.get("licence_class") or "unknown",
        page_keys=[p.key for p in chunk.pages],
        provenance={"page_keys": [p.key for p in chunk.pages],
                    "ocr_containment": min((p.containment for p in chunk.pages if p.containment is not None),
                                           default=None)},
        source=SOURCE, work=chunk.work, part=first.part,
        page_range=f"{first.page_no}" if first is last else f"{first.page_no}-{last.page_no}",
        printed_pages=label, piece=chunk.piece, strip_ops=ops, source_scrub_hits=source_scrub_hits, target_raw=None,
        reader_exposed=None, exposure_level=None)
    return {"messages": _rows.make_messages(user, rest), "meta": meta}


# ------------------------------------------------------------------------------------ build
def _tally() -> Dict[str, Any]:
    """Empty per-work report block.

    :returns: Counters for pages, tokens, drops and rows.
    """
    return {"pages_total": 0, "page_tokens_total": 0, "dropped": collections.defaultdict(lambda: {"pages": 0, "tokens": 0}),
            "quality_multilabel": collections.Counter(), "train": {"pages": 0, "page_tokens": 0, "rows": 0, "row_tokens": 0, "target_tokens": 0},
            "probe": {"pages": 0, "page_tokens": 0, "rows": 0, "row_tokens": 0, "target_tokens": 0}}


def _drop(t: Dict[str, Any], reason: str, tokens: int, pages: int = 1) -> None:
    """Record dropped pages in a report block.

    :param t: Work report block.
    :param reason: Drop reason.
    :param tokens: Page-text tokens dropped.
    :param pages: Pages dropped.
    :returns: None.
    """
    t["dropped"][reason]["pages"] += pages
    t["dropped"][reason]["tokens"] += tokens


def _default_count_batch(tokenizer_path: Optional[Union[str, Path]]) -> Callable[[Sequence[str]], List[int]]:
    """Token counter bound to a tokenizer file.

    :param tokenizer_path: ``tokenizer.json`` (None = ``paths.tokenizer_json``).
    :returns: A function counting tokens of a list of texts.
    """
    def count(texts: Sequence[str]) -> List[int]:
        """Count tokens of texts.

        :param texts: Texts.
        :returns: Counts.
        """
        return _rows.count_tokens_batch(list(texts), tokenizer_path)

    return count


def build(pages: Sequence[_pages.PageRec], lit_manifest: Mapping[str, Any],
          excluded: Optional[Mapping[str, Sequence[str]]] = None, token_cap: Optional[int] = DEFAULT_TOKEN_CAP,
          max_tokens: int = DEFAULT_MAX_TOKENS, probe_frac: float = DEFAULT_PROBE_FRAC,
          open_frac: float = DEFAULT_OPEN_FRAC, min_containment: float = DEFAULT_MIN_CONTAINMENT,
          seed: int = DEFAULT_SEED, tokenizer_path: Optional[Union[str, Path]] = None,
          template_path: Optional[Union[str, Path]] = None, dedupe: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run the L1 pipeline (module docstring) on literature pages.

    :param pages: Pages from :func:`pages.iter_pages` (with OCR).
    :param lit_manifest: Literature manifest (source stubs, licence classes).
    :param excluded: ``{page_key: reasons}`` from ``excluded_pages.json`` (None = no exclusions).
    :param token_cap: Per-work cap on train row tokens (None = off).
    :param max_tokens: Page-text token budget per chunk.
    :param probe_frac: Share of each work's pages in ``split=probe``.
    :param open_frac: Opening share in the user turn.
    :param min_containment: Provenance gate on ``ocr_containment``.
    :param seed: Seed for probe and cap choices.
    :param tokenizer_path: Explicit ``tokenizer.json`` (default ``paths.tokenizer_json``).
    :param template_path: Explicit ``chat_template.jinja`` (default ``paths.chat_template``).
    :param dedupe: Drop near-duplicate pages.
    :returns: ``(rows, report)``; rows are train then probe rows in page order, each rendered and
        counted; the report carries per-work pages / tokens kept and dropped by reason.
    """
    excluded = excluded or {}
    count_batch = _default_count_batch(tokenizer_path)
    pages = sorted(pages, key=lambda p: (p.work, p.part, p.page_no))
    heads = running_heads_by_part(pages)
    report: Dict[str, Dict[str, Any]] = collections.defaultdict(_tally)

    texts, opses = [], []
    for p in pages:
        raw = p.ocr_text or ""
        t, ops = strip_furniture(raw, heads.get((p.work, p.part), ())) if raw else (p.full_main_text or "", [])
        texts.append(t)
        opses.append(ops)
    toks = count_batch(texts)
    furniture: collections.Counter = collections.Counter()

    stage1: List[PageCand] = []
    for p, t, ops, n_tok in zip(pages, texts, opses, toks):
        w = report[p.work]
        w["pages_total"] += 1
        w["page_tokens_total"] += n_tok
        if not p.ocr_text:
            _drop(w, "no_ocr", n_tok)
            continue
        q = quality_reasons(t, p.page_type)
        w["quality_multilabel"].update(q)
        if q:
            _drop(w, f"quality:{q[0]}", n_tok)
            continue
        if p.ocr_containment is None:
            _drop(w, "containment_none", n_tok)
            continue
        if p.ocr_containment < min_containment:
            _drop(w, "low_containment", n_tok)
            continue
        furniture.update(ops)
        printed = p.printed_page_num if p.printed_page_source in ("extracted", "inferred") else None
        stage1.append(PageCand(p.page_key, p.work, p.part, p.page_no, t, n_tok, printed, p.ocr_source,
                               p.ocr_containment, sorted(set(ops))))

    dups = find_duplicates({c.key: c.text for c in stage1}) if dedupe else {}
    groups: Dict[str, List[str]] = collections.defaultdict(list)
    for k, rep in dups.items():
        groups[rep].append(k)
    excluded_keys = set(excluded)
    tainted = {rep for rep, ks in groups.items() if rep in excluded_keys or any(k in excluded_keys for k in ks)}
    tainted |= {k for rep in tainted for k in groups[rep]}
    scrub_total = 0
    kept: List[PageCand] = []
    scrub_by_key: Dict[str, int] = {}
    for c in stage1:
        w = report[c.work]
        if c.key in excluded_keys:
            _drop(w, "excluded", c.tokens)
            continue
        if c.key in tainted:
            _drop(w, "excluded_via_duplicate", c.tokens)
            continue
        if c.key in dups:
            _drop(w, "duplicate", c.tokens)
            continue
        stripped, hits = _tn.strip_source_names(c.text)
        if hits:
            scrub_total += hits
            scrub_by_key[c.key] = hits
            c.text = stripped
            c.tokens = count_batch([stripped])[0]
        if len(c.text) < MIN_CHUNK_CHARS:
            _drop(w, "empty_after_strip", c.tokens)
            continue
        kept.append(c)

    chunks = make_chunks(kept, max_tokens, count_batch)
    probe_keys = assign_probe(chunks, probe_frac, seed)
    rows: List[Dict[str, Any]] = []
    for ch in chunks:
        info = source_info(lit_manifest, ch.work, ch.pages[0].part)
        hits = sum(scrub_by_key.get(p.key, 0) for p in ch.pages) if ch.piece in (None, 0) else 0
        rows.append(make_row(ch, info, "probe" if ch.key in probe_keys else "train", open_frac, hits))
    rendered = [_rows.render_chat(r["messages"], template_path=template_path) for r in rows]
    n_render = count_batch(rendered)
    n_target = count_batch([r["messages"][2]["content"] for r in rows])
    page_tok = {c.key: c.tokens for c in kept}
    chunk_pages = {ch.key: ch for ch in chunks}
    for r, nr, nt in zip(rows, n_render, n_target):
        r["meta"]["tokens"], r["meta"]["target_tokens"] = nr, nt
    rows, capped = apply_cap(rows, token_cap, seed)
    for r in capped:
        ch = chunk_pages[_chunk_key_of(r)]
        _drop(report[ch.work], "work_cap", ch.tokens if ch.piece is not None else sum(page_tok[p.key] for p in ch.pages),
              pages=0 if ch.piece not in (None, 0) else len(ch.pages))
    for r in rows:
        ch = chunk_pages[_chunk_key_of(r)]
        blk = report[ch.work][r["meta"]["split"]]
        blk["rows"] += 1
        blk["row_tokens"] += r["meta"]["tokens"]
        blk["target_tokens"] += r["meta"]["target_tokens"]
        if ch.piece in (None, 0):
            blk["pages"] += len(ch.pages)
        blk["page_tokens"] += ch.tokens if ch.piece is not None else sum(page_tok[p.key] for p in ch.pages)
    rows.sort(key=lambda r: (r["meta"]["split"] != "train", _chunk_key_of(r)))
    summary = _summarise(report, rows, lit_manifest)
    summary.update({"source_scrub_hits_removed": scrub_total, "rows_with_source_names_after": sum(1 for r in rows if any(
        _tn.has_source_names(m["content"]) for m in r["messages"])), "furniture_lines_removed": dict(furniture),
        "duplicate_pages": len(dups), "chunks": len(chunks), "capped_rows": len(capped)})
    return rows, summary


def _chunk_key_of(row: Mapping[str, Any]) -> str:
    """Chunk key of a built row (page keys joined by ``+``, ``#piece`` for split pages).

    :param row: A ``domain_reading`` row.
    :returns: The key.
    """
    meta = row["meta"]
    k = "+".join(meta["provenance"]["page_keys"])
    return k if meta.get("piece") is None else f"{k}#{meta['piece']}"


def _summarise(report: Mapping[str, Dict[str, Any]], rows: Sequence[Dict[str, Any]],
               lit_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Turn per-work tallies and rows into the manifest's ``works`` / ``counts`` / ``totals`` blocks.

    :param report: Per-work tallies from :func:`build`.
    :param rows: Kept rows.
    :param lit_manifest: Literature manifest (licence class per work).
    :returns: Summary dict.
    """
    works: Dict[str, Any] = {}
    tot_drop: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: {"pages": 0, "tokens": 0})
    qml: collections.Counter = collections.Counter()
    tot = collections.Counter()
    for w in sorted(report):
        t = report[w]
        entry = (lit_manifest.get("works") or {}).get(w) or {}
        works[w] = {"licence_class": entry.get("licence_class", "unknown"), "pages_total": t["pages_total"],
                    "page_tokens_total": t["page_tokens_total"], "train": dict(t["train"]), "probe": dict(t["probe"]),
                    "dropped": {k: dict(v) for k, v in sorted(t["dropped"].items())},
                    "quality_multilabel": dict(t["quality_multilabel"])}
        for k, v in t["dropped"].items():
            tot_drop[k]["pages"] += v["pages"]
            tot_drop[k]["tokens"] += v["tokens"]
        qml.update(t["quality_multilabel"])
        tot["pages_total"] += t["pages_total"]
        tot["page_tokens_total"] += t["page_tokens_total"]
        for s in ("train", "probe"):
            for k2, v2 in t[s].items():
                tot[f"{s}_{k2}"] += v2
    rows_c: Dict[str, Any] = {}
    tok_c: Dict[str, Any] = {}
    for r in rows:
        m = r["meta"]
        path = (m["family"], m["condition"], m["lang_class"], m["split"])
        for tree, val in ((rows_c, 1), (tok_c, m["tokens"])):
            node = tree
            for k in path[:-1]:
                node = node.setdefault(k, {})
            node[path[-1]] = node.get(path[-1], 0) + val
    by_class: collections.Counter = collections.Counter()
    for r in rows:
        if r["meta"]["split"] == "train":
            by_class[r["meta"]["licence_class"]] += r["meta"]["tokens"]
    return {"works": works, "dropped_by_reason": {k: dict(v) for k, v in sorted(tot_drop.items())},
            "quality_multilabel": dict(qml), "totals": dict(tot), "counts": {"rows": rows_c, "tokens": tok_c},
            "train_tokens_by_licence_class": dict(by_class)}


# ------------------------------------------------------------------------------------ CLI
def _sha256(path: Optional[Path]) -> Optional[str]:
    """SHA-256 of a file.

    :param path: File or None.
    :returns: Hex digest or None.
    """
    if path is None or not Path(path).exists():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rel(path: Optional[Path], repo: Path) -> Optional[str]:
    """Path relative to the repository when possible (manifests avoid absolute machine paths).

    :param path: Path or None.
    :param repo: Repository root.
    :returns: Relative (or the file name for outside paths) string, or None.
    """
    if path is None:
        return None
    try:
        return Path(path).resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return Path(path).name


def cmd_manifest(args: argparse.Namespace) -> int:
    """``manifest`` sub-command: (re)build the local literature manifest.

    :param args: Parsed arguments (``root``, ``out``, ``force``).
    :returns: Exit code.
    """
    p = _paths.load_paths()
    root = Path(args.root) if args.root else p.academic_literature
    out = Path(args.out) if args.out else default_manifest_path(p)
    new = build_literature_manifest(root)
    old = None if args.force or not out.exists() else load_literature_manifest(out)
    man = merge_manifests(new, old)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, out)
    cls = collections.Counter(e.get("licence_class") for e in man["works"].values())
    typ = collections.Counter(e.get("type") for e in man["works"].values())
    print(f"literature manifest: {len(man['works'])} work dirs -> {_rel(out, p.repo)}")
    print(f"  licence_class: {dict(cls)}")
    print(f"  type: {dict(typ)}")
    print(f"  with author: {sum(1 for e in man['works'].values() if e.get('author') or e.get('parts'))}, "
          f"with year: {sum(1 for e in man['works'].values() if e.get('year') or e.get('parts'))}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """``build`` sub-command: write ``families/domain_reading.jsonl`` + manifest, or the dry report.

    :param args: Parsed arguments.
    :returns: Exit code.
    """
    p = _paths.load_paths()
    root = Path(args.root) if args.root else p.academic_literature
    man_path = Path(args.literature_manifest) if args.literature_manifest else default_manifest_path(p)
    lit = load_literature_manifest(man_path)
    excl_path = Path(args.excluded_pages) if args.excluded_pages else None
    excluded = load_excluded_pages(excl_path)
    if excl_path is None:
        LOG.warning("no --excluded-pages: building WITHOUT contamination exclusions")
    pages = _pages.iter_pages(root, works=args.works or None)
    if args.max_pages_per_work:
        seen: collections.Counter = collections.Counter()
        sample = []
        for pg in pages:
            if seen[pg.work] < args.max_pages_per_work:
                seen[pg.work] += 1
                sample.append(pg)
        pages = sample
    rows, summary = build(pages, lit, excluded, token_cap=args.token_cap or None, max_tokens=args.max_tokens,
                          probe_frac=args.probe_frac, open_frac=args.open_frac, min_containment=args.min_containment,
                          seed=args.seed, dedupe=not args.no_dedupe)
    build_args = {k: v for k, v in vars(args).items() if k not in ("func",)}
    for k in ("root", "literature_manifest", "excluded_pages", "out", "report"):
        if build_args.get(k):
            build_args[k] = _rel(Path(build_args[k]), p.repo)
    manifest = {"family": FAMILY, "build": _rows.BUILD, "dry_run": bool(args.dry_run),
                "generated": _dt.datetime.now().isoformat(timespec="seconds"), "build_args": build_args,
                "inputs": {"excluded_pages": _rel(excl_path, p.repo), "excluded_pages_sha256": _sha256(excl_path),
                           "excluded_page_keys": len(excluded), "literature_manifest": _rel(man_path, p.repo),
                           "literature_manifest_sha256": _sha256(man_path), "pages_read": len(pages)},
                **summary}
    if not args.dry_run:
        fam = p.out("families", f"{FAMILY}.jsonl", mkdir=True)
        n = _rows.write_jsonl(fam, rows)
        manifest["output"] = _rel(fam, p.repo)
        manifest["fingerprint"] = _rows.fingerprint(rows)
        out = Path(args.report) if args.report else p.out("families", f"{FAMILY}_manifest.json", mkdir=True)
        print(f"wrote {n} rows -> {_rel(fam, p.repo)}")
    else:
        out = Path(args.report) if args.report else p.out("literature_l1_dry_report.json", mkdir=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    t = summary["totals"]
    print(f"pages {t.get('pages_total', 0)} ({t.get('page_tokens_total', 0):,} page tokens); "
          f"train rows {t.get('train_rows', 0)} / {t.get('train_row_tokens', 0):,} tokens on {t.get('train_pages', 0)} pages; "
          f"probe rows {t.get('probe_rows', 0)} / {t.get('probe_row_tokens', 0):,} tokens")
    for k, v in summary["dropped_by_reason"].items():
        print(f"  dropped {k:32} pages {v['pages']:5}  tokens {v['tokens']:10,}")
    print(f"  project-name hits removed {summary['source_scrub_hits_removed']}, rows with project names after {summary['rows_with_source_names_after']}, "
          f"duplicates {summary['duplicate_pages']}, capped rows {summary['capped_rows']}")
    print(f"report -> {_rel(out, p.repo)}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point.

    :param argv: Arguments (default ``sys.argv[1:]``).
    :returns: Exit code.
    """
    ap = argparse.ArgumentParser(description="text27b L1 domain_reading rows from the academic literature")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest", help="(re)build the local literature manifest")
    m.add_argument("--root", help="literature root (default paths.academic_literature)")
    m.add_argument("--out", help="manifest path (default logs/text_finetune_27b/literature_manifest.json)")
    m.add_argument("--force", action="store_true", help="overwrite hand-set values")
    m.set_defaults(func=cmd_manifest)
    b = sub.add_parser("build", help="build families/domain_reading.jsonl (or a dry sizing report)")
    b.add_argument("--root", help="literature root (default paths.academic_literature)")
    b.add_argument("--literature-manifest", help="literature manifest (default next to out_dir)")
    b.add_argument("--excluded-pages", help="excluded_pages.json from the contamination module (optional)")
    b.add_argument("--token-cap", type=int, default=DEFAULT_TOKEN_CAP, help="per-work train token cap (0 = off)")
    b.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="page-text tokens per chunk")
    b.add_argument("--probe-frac", type=float, default=DEFAULT_PROBE_FRAC, help="share of each work's pages in probe")
    b.add_argument("--open-frac", type=float, default=DEFAULT_OPEN_FRAC, help="opening share in the user turn")
    b.add_argument("--min-containment", type=float, default=DEFAULT_MIN_CONTAINMENT, help="provenance gate")
    b.add_argument("--seed", type=int, default=DEFAULT_SEED)
    b.add_argument("--works", nargs="*", help="restrict to these work directories (sampling)")
    b.add_argument("--max-pages-per-work", type=int, default=0, help="sample the first N pages per work (0 = all)")
    b.add_argument("--no-dedupe", action="store_true", help="keep near-duplicate pages")
    b.add_argument("--dry-run", action="store_true", help="write only literature_l1_dry_report.json, no rows")
    b.add_argument("--report", help="report path override")
    b.set_defaults(func=cmd_build)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
