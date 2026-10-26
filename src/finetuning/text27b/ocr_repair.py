# File name: ocr_repair.py
# Date: 10/25/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Repair bidi-fragmented OCR text layers of literature works (text27b books-mining pass).

Some works' ``*_ocr_results.json`` hold a PDF text layer that a word processor exported run by run: every change
of direction (a bracket, a digit, a vowel point, a line number) starts a new row, so a printed right-to-left line
becomes five to twenty rows of one or two words, with mirrored brackets (``]...[``, ``)1``). Such pages fail the
L1 quality filter (``fragmented_lines``) and no translation span can be cut from them.

This module rebuilds proper lines from the glyph geometry and writes the repaired page texts next to the build
outputs (never into the literature tree):

* **Geometry** (``method = pdf_geometry``): the OCR file's own PDF (found by file name inside the work directory)
  is opened with PyMuPDF and its text layer is checked to be *identical* to the OCR JSON text of the page; only
  then are its glyph boxes used (the OCR JSON itself stores no boxes). Glyphs are clustered into printed lines by
  baseline (raised superscripts join the line below them), combining marks are attached to the base letter under
  them, a two-column gutter is detected and each column read in turn, and each line is converted from visual to
  logical order (:func:`visual_to_logical`: right-to-left base, left-to-right runs re-reversed, mirrored brackets
  swapped). Numbered edition lines that wrap are joined to their line, words broken across two lines are joined,
  and paragraph gaps become blank lines.
* **Heuristic** (``method = json_heuristic``, no PDF or a text-layer mismatch): rows are concatenated until a
  whitespace-only row or a bare line number, and mirrored brackets are swapped in right-to-left rows.

Every repaired page keeps exactly the characters of the OCR page (whitespace and line breaks aside; a mirrored
bracket pair counts as the same pair): :func:`chars_preserved` is checked per page and recorded. The text is a
re-ordering of the OCR text, so spans cut from it remain verbatim OCR text in the sense of decision D15.

Outputs under ``paths.out_dir`` / :data:`REPAIR_DIRNAME` / ``<work>/``: ``page_NNN.txt`` (one per repaired page,
``NNN`` = the OCR ``page_number``), ``manifest.json`` (OCR file, PDF file, per-page sha1 of the original OCR text
and method -- :func:`pages.iter_pages` applies a repair only when the sha1 still matches) and ``report.json``
(lines before / after, share of lines with ≥ 3 words, the L1 quality verdicts before / after, 10 before / after
samples).

CLI::

    PYTHONPATH=. .venv/bin/python -m src.finetuning.text27b.ocr_repair --work WORK            # repair + report
    PYTHONPATH=. .venv/bin/python -m src.finetuning.text27b.ocr_repair --work WORK --show 241  # print one page
    PYTHONPATH=. .venv/bin/python -m src.finetuning.text27b.ocr_repair --work WORK --dry-run   # report only

Scope: the bracket mirroring and run rules follow a word processor's right-to-left export, where the text layer
stores the *displayed* (mirrored) bracket glyph and punctuation typed between two left-to-right items belongs to
the right-to-left text. A typesetter's layer that stores logical brackets would be damaged by the mirroring, so a
work is repaired only on request (``--work``) after its pages have been checked by eye (``--show N``).

Public API: :data:`REPAIR_DIRNAME`, :class:`Glyph`, :class:`Cluster`, :class:`Line`, :class:`TextLine`,
:func:`bidi_class`, :func:`mirror`, :func:`glyphs_from_rawdict`, :func:`supplement_glyphs`, :func:`load_pdf_page`,
:func:`find_pdf`, :func:`cluster_lines`, :func:`attach_marks`, :func:`detect_gutter`, :func:`split_columns`,
:func:`visual_to_logical`, :func:`line_rtl`, :func:`mark_superscripts`, :func:`line_text`, :func:`join_continuations`,
:func:`join_broken_words`, :func:`mark_paragraphs`, :func:`assemble_page`, :func:`repair_geometry`,
:func:`heuristic_rejoin`, :func:`chars_preserved`, :func:`fragment_stats`, :class:`PageRepair`, :func:`repair_work`,
:func:`report`, :func:`write_repairs`, :func:`default_repair_root`, :func:`load_repairs`, :func:`main`.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import re
import statistics
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.finetuning.text27b import paths as _paths

LOG = logging.getLogger(__name__)
#: Directory (under ``paths.out_dir``) holding the repaired page texts, one sub-directory per work.
REPAIR_DIRNAME = "ocr_repair"
MANIFEST_NAME = "manifest.json"
REPORT_NAME = "report.json"
VERSION = 1
#: Bracket glyphs a right-to-left run displays mirrored.
_MIRROR: Dict[str, str] = {"(": ")", ")": "(", "[": "]", "]": "[", "{": "}", "}": "{", "<": ">", ">": "<",
                           "«": "»", "»": "«"}
#: Hebrew letters that have a final form: a word cannot end in the medial form.
_MEDIAL_WITH_FINAL = set("כמנפצ")
_HE_LETTER = re.compile(r"[א-ת]")
_WORD = re.compile(r"[א-ת؀-ۿA-Za-z]+")
#: Line clustering: glyphs whose baselines differ by at most this share of the font size share a line.
BASELINE_TOL = 0.3
#: A tentative line set in a font smaller than this share of the line below it is a superscript row.
SUPERSCRIPT_SIZE = 0.85
#: ... and it joins that line when its baseline is within this share of the line's size above it.
SUPERSCRIPT_RISE = 0.75
#: A horizontal gap wider than this share of the font size between two inked glyphs is a word space. Space glyphs
#: of the text layer are not trusted: measured on a repaired work, gaps inside words are ≤ 0.1 em and word gaps
#: ≥ 0.2 em, while 1 in 10 space glyphs sits between two touching letters (after a vowel point or a bracket).
SPACE_GAP = 0.16
#: Gutter: minimum gap width (points) and the share of the page's long lines that must show it.
GUTTER_MIN_GAP = 9.0
GUTTER_LINE_SHARE = 0.6
GUTTER_MIN_LINES = 6
#: Each column must hold at least this many characters on a line to count as prose.
GUTTER_MIN_SIDE_CHARS = 12
#: A vertical gap wider than this multiple of the median line pitch starts a new paragraph.
PARAGRAPH_GAP = 1.55


# ============================================================================== geometry records
@dataclass
class Glyph:
    """One glyph of a PDF text layer.

    ``x0`` / ``x1`` / ``y0`` / ``y1`` are the glyph box, ``ox`` / ``oy`` its origin (``oy`` = baseline),
    ``size`` the span's font size, ``seq`` the glyph's position in the content stream and ``span`` the index of
    its text-layer span (-1 when unknown).
    """

    c: str
    x0: float
    x1: float
    y0: float
    y1: float
    ox: float
    oy: float
    size: float
    seq: int
    span: int = -1

    @property
    def xc(self) -> float:
        """Horizontal centre of the glyph (the origin for a zero-width glyph).

        :returns: x coordinate.
        """
        return (self.x0 + self.x1) / 2 if self.x1 > self.x0 else self.ox


@dataclass
class Cluster:
    """A base glyph with the combining marks drawn on it (or a synthetic word space, ``base.c == ' '``)."""

    base: Glyph
    marks: List[Glyph] = field(default_factory=list)
    sup: bool = False

    @property
    def text(self) -> str:
        """Base character followed by its marks in stream order.

        :returns: Text.
        """
        return self.base.c + "".join(m.c for m in sorted(self.marks, key=lambda g: g.seq))


@dataclass
class Line:
    """One printed line: its glyphs, baseline and box."""

    glyphs: List[Glyph]
    baseline: float
    size: float

    @property
    def x0(self) -> float:
        """Left edge of the non-space glyphs.

        :returns: x.
        """
        xs = [g.x0 for g in self.glyphs if not g.c.isspace()]
        return min(xs) if xs else 0.0

    @property
    def x1(self) -> float:
        """Right edge of the non-space glyphs.

        :returns: x.
        """
        xs = [g.x1 for g in self.glyphs if not g.c.isspace()]
        return max(xs) if xs else 0.0

    @property
    def top(self) -> float:
        """Top of the line box.

        :returns: y.
        """
        return min(g.y0 for g in self.glyphs)

    @property
    def bottom(self) -> float:
        """Bottom of the line box.

        :returns: y.
        """
        return max(g.y1 for g in self.glyphs)


# ============================================================================== characters
def bidi_class(ch: str) -> str:
    """Simplified bidi class of a character.

    :param ch: One character.
    :returns: ``R`` (Hebrew / Arabic letters), ``L`` (left-to-right letters), ``N`` (digits, European or
        Arabic-Indic), ``S`` (number separators ``. , : / -`` and ``+``), ``M`` (combining mark), ``W``
        (whitespace) or ``O`` (other neutrals: brackets, quotes, other punctuation).
    """
    b = unicodedata.bidirectional(ch)
    if b in ("R", "AL"):
        return "R"
    if b == "L":
        return "L"
    if b in ("EN", "AN"):
        return "N"
    if b in ("ES", "CS", "ET"):
        return "S"
    if b == "NSM":
        return "M"
    if b in ("WS", "B", "S") or ch.isspace():
        return "W"
    return "O"


def mirror(ch: str) -> str:
    """The mirrored form of a bracket glyph (other characters unchanged).

    :param ch: Character.
    :returns: Mirrored character.
    """
    return _MIRROR.get(ch, ch)


# ============================================================================== PDF access
def glyphs_from_rawdict(raw: Mapping[str, Any]) -> List[Glyph]:
    """Glyphs of a PyMuPDF ``rawdict`` page dictionary, in content-stream order.

    :param raw: ``page.get_text("rawdict")``.
    :returns: Glyph list.
    """
    out: List[Glyph] = []
    seq = 0
    span = 0
    for b in raw.get("blocks") or []:
        for ln in b.get("lines") or []:
            for sp in ln.get("spans") or []:
                size = float(sp.get("size") or 0.0)
                for c in sp.get("chars") or []:
                    x0, y0, x1, y1 = (float(v) for v in c["bbox"])
                    ox, oy = (float(v) for v in c["origin"])
                    out.append(Glyph(c=c["c"], x0=x0, x1=x1, y0=y0, y1=y1, ox=ox, oy=oy, size=size, seq=seq,
                                     span=span))
                    seq += 1
                span += 1
    return out


def find_pdf(ocr_file: Path, work_dir: Path) -> Optional[Path]:
    """The PDF an OCR results file was made from, searched by file name inside the work directory.

    :param ocr_file: ``<stem>_ocr_results.json``.
    :param work_dir: Top-level work directory (the search never leaves it).
    :returns: PDF path or None.
    """
    names: List[str] = []
    try:
        meta = json.loads(ocr_file.read_text(encoding="utf-8"))
        if meta.get("pdf_path"):
            names.append(Path(str(meta["pdf_path"])).name)
    except (OSError, ValueError):
        pass
    stem = ocr_file.name[: -len("_ocr_results.json")] if ocr_file.name.endswith("_ocr_results.json") else ocr_file.stem
    names.append(f"{stem}.pdf")
    for name in dict.fromkeys(names):
        hits = sorted(work_dir.rglob(name))
        if hits:
            return hits[0]
    return None


def supplement_glyphs(glyphs: List[Glyph], trace: Sequence[Mapping[str, Any]], text: str) -> int:
    """Add the glyphs the ``rawdict`` view drops (zero-width combining dots) from the page's text trace.

    The plain-text view (= the OCR JSON text) keeps zero-width marks that ``rawdict`` omits. Characters still
    missing from the glyph list (by multiset against ``text``) are taken from the text trace -- deduplicated by
    character and origin (a bold face draws every glyph twice) and skipped when a glyph with the same character
    already sits at that origin -- each as a span of its own.

    :param glyphs: Glyphs from :func:`glyphs_from_rawdict` (extended in place).
    :param trace: ``page.get_texttrace()``.
    :param text: The page's plain text.
    :returns: Number of glyphs added.
    """
    deficit = collections.Counter(ch for ch in text if not ch.isspace())
    deficit.subtract(g.c for g in glyphs if not g.c.isspace())
    if not any(v > 0 for v in deficit.values()):
        return 0
    have = {(g.c, round(g.ox, 1), round(g.oy, 1)) for g in glyphs}
    seen = set()
    seq = max((g.seq for g in glyphs), default=-1) + 1
    span = max((g.span for g in glyphs), default=-1) + 1
    added = 0
    for sp in trace:
        size = float(sp.get("size") or 0.0)
        for ch in sp.get("chars") or ():
            code, _, origin, bbox = ch[0], ch[1], ch[2], ch[3]
            if code <= 0:
                continue
            c = chr(code)
            key = (c, round(float(origin[0]), 1), round(float(origin[1]), 1))
            if key in seen or deficit.get(c, 0) <= 0:
                continue
            seen.add(key)
            if key in have:
                continue
            x0, y0, x1, y1 = (float(v) for v in bbox)
            glyphs.append(Glyph(c=c, x0=x0, x1=x1, y0=y0, y1=y1, ox=float(origin[0]), oy=float(origin[1]),
                                size=size, seq=seq, span=span))
            seq += 1
            span += 1
            deficit[c] -= 1
            added += 1
    return added


def load_pdf_page(pdf: Any, page_number: int) -> Tuple[List[Glyph], str, Tuple[float, float]]:
    """Glyphs and plain text of one PDF page (``rawdict`` glyphs, supplemented from the text trace).

    :param pdf: An open PyMuPDF document.
    :param page_number: 1-based page number (the OCR ``page_number``).
    :returns: ``(glyphs, page.get_text() text, (width, height))``.
    """
    page = pdf[page_number - 1]
    text = page.get_text()
    glyphs = glyphs_from_rawdict(page.get_text("rawdict"))
    supplement_glyphs(glyphs, page.get_texttrace(), text)
    return glyphs, text, (float(page.rect.width), float(page.rect.height))


# ============================================================================== lines
def cluster_lines(glyphs: Sequence[Glyph]) -> List[Line]:
    """Cluster glyphs into printed lines by baseline, top to bottom.

    Glyphs whose baselines lie within :data:`BASELINE_TOL` × font size of a line's first baseline share it.
    A tentative line set in a smaller font (< :data:`SUPERSCRIPT_SIZE` of the line below) whose baseline is at
    most :data:`SUPERSCRIPT_RISE` × that line's size above it is a row of superscripts (footnote references)
    and joins the line below.

    :param glyphs: Page glyphs.
    :returns: Lines sorted by baseline.
    """
    gl = sorted((g for g in glyphs if g.c), key=lambda g: (round(g.oy, 1), -g.xc))
    tentative: List[Line] = []
    for g in gl:
        if tentative:
            cur = tentative[-1]
            tol = BASELINE_TOL * max(cur.size, g.size, 1.0)
            if abs(g.oy - cur.baseline) <= tol:
                cur.glyphs.append(g)
                if not g.c.isspace():
                    cur.size = max(cur.size, g.size)
                continue
        tentative.append(Line(glyphs=[g], baseline=g.oy, size=g.size))
    merged: List[Line] = []
    i = 0
    while i < len(tentative):
        ln = tentative[i]
        nxt = tentative[i + 1] if i + 1 < len(tentative) else None
        if nxt is not None and _median_size(ln) < SUPERSCRIPT_SIZE * _median_size(nxt) and \
                0 < nxt.baseline - ln.baseline <= SUPERSCRIPT_RISE * _median_size(nxt) and \
                ln.x0 >= nxt.x0 - 2 * nxt.size and ln.x1 <= nxt.x1 + 2 * nxt.size:
            nxt.glyphs = ln.glyphs + nxt.glyphs
            i += 1
            continue
        merged.append(ln)
        i += 1
    return [ln for ln in merged if any(not g.c.isspace() for g in ln.glyphs)]


def _median_size(line: Line) -> float:
    """Median font size of a line's non-space glyphs.

    :param line: Line.
    :returns: Size (the line size when it holds only spaces).
    """
    s = [g.size for g in line.glyphs if not g.c.isspace()]
    return statistics.median(s) if s else line.size


def attach_marks(glyphs: Sequence[Glyph]) -> List[Cluster]:
    """Attach combining marks to their base glyph.

    A mark that shares its text-layer span with letters goes to the last letter before it in that span (the
    exporter writes a base and its marks together). A mark in a span of its own is placed by geometry: the
    letter whose left edge is at the mark's origin (marks drawn from the base's left edge), else the letter
    whose right edge is there, else the letter whose box contains the mark's centre, else the nearest letter.

    :param glyphs: Glyphs of one line.
    :returns: Clusters (base glyphs with their marks; spaces as their own clusters).
    """
    bases = [Cluster(g) for g in glyphs if bidi_class(g.c) != "M"]
    letters = [c for c in bases if not c.base.c.isspace()]
    for m in sorted((g for g in glyphs if bidi_class(g.c) == "M"), key=lambda g: g.seq):
        if not letters:
            bases.append(Cluster(m))
            continue
        prev = [c for c in letters if m.span >= 0 and c.base.span == m.span and c.base.seq < m.seq]
        if prev:
            max(prev, key=lambda c: c.base.seq).marks.append(m)
            continue
        best = None
        for test in (lambda c: abs(c.base.x0 - m.ox) <= 0.6, lambda c: abs(c.base.x1 - m.ox) <= 0.6,
                     lambda c: c.base.x0 - 0.5 <= m.xc <= c.base.x1 + 0.5):
            pool = [c for c in letters if test(c)]
            if pool:
                best = min(pool, key=lambda c: (abs(c.base.xc - m.xc), abs(c.base.seq - m.seq)))
                break
        if best is None:
            best = min(letters, key=lambda c: (abs(c.base.xc - m.xc), abs(c.base.seq - m.seq)))
        best.marks.append(m)
    return bases


def detect_gutter(lines: Sequence[Line]) -> Optional[float]:
    """x coordinate of a two-column gutter, or None for a single-column page.

    A gutter is an empty vertical band at least :data:`GUTTER_MIN_GAP` wide in the middle half of the text
    block that splits at least :data:`GUTTER_LINE_SHARE` of the lines with text on both sides of it, each
    side holding ≥ :data:`GUTTER_MIN_SIDE_CHARS` characters, on at least :data:`GUTTER_MIN_LINES` lines.

    :param lines: Page lines.
    :returns: Gutter centre or None.
    """
    body = [ln for ln in lines if sum(1 for g in ln.glyphs if not g.c.isspace()) >= 2 * GUTTER_MIN_SIDE_CHARS]
    if len(body) < GUTTER_MIN_LINES:
        return None
    left = min(ln.x0 for ln in body)
    right = max(ln.x1 for ln in body)
    width = right - left
    if width <= 0:
        return None
    votes: collections.Counter = collections.Counter()
    for ln in body:
        boxes = sorted((g.x0, g.x1) for g in ln.glyphs if not g.c.isspace())
        cover_end = boxes[0][1]
        for x0, x1 in boxes[1:]:
            if x0 - cover_end >= GUTTER_MIN_GAP:
                mid = (x0 + cover_end) / 2
                if left + 0.25 * width <= mid <= left + 0.75 * width:
                    for x in range(int(cover_end) + 1, int(x0)):
                        votes[x] += 1
            cover_end = max(cover_end, x1)
    if not votes:
        return None
    x, n = max(votes.items(), key=lambda kv: (kv[1], -abs(kv[0] - (left + width / 2))))
    if n < max(GUTTER_MIN_LINES, GUTTER_LINE_SHARE * len(body)):
        return None
    both = 0
    for ln in body:
        r = sum(1 for g in ln.glyphs if not g.c.isspace() and g.x0 > x)
        lft = sum(1 for g in ln.glyphs if not g.c.isspace() and g.x1 < x)
        if r >= GUTTER_MIN_SIDE_CHARS and lft >= GUTTER_MIN_SIDE_CHARS:
            both += 1
    return float(x) if both >= max(GUTTER_MIN_LINES, GUTTER_LINE_SHARE * len(body)) else None


def split_columns(lines: Sequence[Line], gutter: float, rtl: bool) -> List[Line]:
    """Reading order of a two-column page: regions of split lines read column by column.

    Lines with glyphs on both sides of the gutter but none straddling it are split; a line that straddles the
    gutter (a full-width heading) closes the current two-column region. Within a region the right column is
    read first on a right-to-left page.

    :param lines: Page lines (top to bottom).
    :param gutter: Gutter x.
    :param rtl: Right-to-left page.
    :returns: Lines in reading order.
    """
    out: List[Line] = []
    first: List[Line] = []
    second: List[Line] = []

    def flush() -> None:
        """Emit the current region.

        :returns: None.
        """
        out.extend(first)
        out.extend(second)
        first.clear()
        second.clear()

    for ln in lines:
        straddle = any(g.x0 < gutter < g.x1 for g in ln.glyphs if not g.c.isspace())
        if straddle:
            flush()
            out.append(ln)
            continue
        right = [g for g in ln.glyphs if g.xc > gutter]
        left = [g for g in ln.glyphs if g.xc <= gutter]
        parts = [(right, first), (left, second)] if rtl else [(left, first), (right, second)]
        for gs, bucket in parts:
            if any(not g.c.isspace() for g in gs):
                bucket.append(Line(glyphs=gs, baseline=ln.baseline, size=ln.size))
    flush()
    return out


# ============================================================================== visual → logical
def _with_spaces(clusters: List[Cluster]) -> List[Cluster]:
    """Replace the text layer's space glyphs by spaces wherever two visually adjacent glyphs are a word gap apart.

    :param clusters: Clusters sorted left to right.
    :returns: Inked clusters with synthetic spaces between words (left to right).
    """
    out: List[Cluster] = []
    for c in (x for x in clusters if not x.base.c.isspace()):
        if out:
            p = out[-1]
            gap = c.base.x0 - p.base.x1
            size = max(c.base.size, p.base.size, 1.0)
            digit_sup = (c.sup != p.sup) and (c.base.c.isdigit() or c.base.c in ".,/") and \
                (p.base.c.isdigit() or p.base.c in ".,/")
            if (gap > SPACE_GAP * size or digit_sup) and not p.base.c.isspace():
                out.append(Cluster(Glyph(" ", p.base.x1, c.base.x0, p.base.y0, p.base.y1, p.base.x1, p.base.oy,
                                         size, p.base.seq)))
        out.append(c)
    return out


def _runs(classes: Sequence[str], strong: str) -> List[Tuple[int, int]]:
    """Index ranges ``[a, b]`` of embedded runs in a visual sequence.

    For a right-to-left line (``strong = 'L'``): stretches that start and end with an L letter or digit, hold at
    least one L letter, no R letter and no bracket, semicolon, dash or other neutral except spaces, number
    separators and word-internal punctuation (a character between two L letters or digits, as in ``Smith’s`` or
    ``Ar. 18(1).136``); plus number runs (digits
    joined by single separators) outside them. The word processor that wrote these layers sets punctuation
    typed between two left-to-right items in the right-to-left direction, so ``TS 13 J 5.4; TS NS 184.51`` is two
    runs and ``68–72)`` two numbers. For a left-to-right line (``strong = 'R'``): stretches that start and end
    with an R letter and hold no L letter (digits and neutrals inside them belong to the run).

    :param classes: :func:`bidi_class` of each position.
    :param strong: The embedded direction's strong class.
    :returns: Ranges (inclusive).
    """
    out: List[Tuple[int, int]] = []
    n = len(classes)
    if strong == "L" and "R" not in classes:
        # a line without right-to-left letters (a shelfmark heading in a right-to-left paragraph) is one run
        idx = [k for k, c in enumerate(classes) if c in ("L", "N")]
        return [(idx[0], idx[-1])] if idx else []
    if strong == "L":
        i = 0
        while i < n:
            if classes[i] == "n":
                k = i
                while k + 1 < n and classes[k + 1] == "n":
                    k += 1
                out.append((i, k))
                i = k + 1
                continue
            if classes[i] in ("L", "N"):
                j, last, has_l = i, i, False
                while j < n and classes[j] != "R":
                    if classes[j] == "n" or (classes[j] == "O" and not (0 < j < n - 1 and classes[j - 1] in ("L", "N")
                                                                        and classes[j + 1] in ("L", "N"))):
                        break
                    if classes[j] in ("L", "N"):
                        last = j
                        has_l = has_l or classes[j] == "L"
                    j += 1
                if has_l:
                    out.append((i, last))
                    i = last + 1
                    continue
                # number run: digits joined by single separators
                k = i
                while k + 1 < n and (classes[k + 1] == "N" or (classes[k + 1] == "S" and k + 2 < n and
                                                               classes[k + 2] == "N")):
                    k += 2 if classes[k + 1] == "S" else 1
                out.append((i, k))
                i = k + 1
                continue
            i += 1
        return out
    i = 0
    while i < n:
        if classes[i] == "R":
            j, last = i, i
            while j < n and classes[j] != "L":
                if classes[j] == "R":
                    last = j
                j += 1
            out.append((i, last))
            i = last + 1
            continue
        i += 1
    return out


def visual_to_logical(clusters: Sequence[Cluster], rtl: bool) -> str:
    """Logical text of a line's clusters (simplified Unicode bidi, visual order in, logical order out).

    Right-to-left line: clusters are read right to left; embedded left-to-right runs (Latin words with their
    numbers, or numbers) are re-reversed; bracket glyphs outside those runs are mirrored. Left-to-right line:
    clusters are read left to right; right-to-left runs are reversed (numbers inside them kept left to right)
    and the brackets inside them mirrored.

    :param clusters: Clusters of one line (any order; sorted here by horizontal centre).
    :param rtl: Right-to-left base direction.
    :returns: Logical text (spaces collapsed, stripped).
    """
    vis = _with_spaces(sorted(clusters, key=lambda c: (c.base.xc, c.base.seq)))
    if rtl:
        vis = vis[::-1]
    # superscripts (footnote references) never join a run: digits among them form runs of their own ("n")
    classes = [("n" if c.base.c.isdigit() else "O") if c.sup else bidi_class(c.base.c) for c in vis]
    texts = [c.text for c in vis]
    runs = _runs(classes, "L" if rtl else "R")
    inside = [False] * len(vis)
    for a, b in runs:
        for k in range(a, b + 1):
            inside[k] = True
    order = list(range(len(vis)))
    for a, b in runs:
        seg = order[a:b + 1][::-1]
        if not rtl:
            # keep digit groups left to right inside a reversed right-to-left run
            fixed: List[int] = []
            k = 0
            while k < len(seg):
                if classes[seg[k]] in ("N", "n"):
                    m = k
                    while m + 1 < len(seg) and classes[seg[m + 1]] in ("N", "n", "S") and \
                            (classes[seg[m + 1]] in ("N", "n") or
                             (m + 2 < len(seg) and classes[seg[m + 2]] in ("N", "n"))):
                        m += 1
                    fixed.extend(seg[k:m + 1][::-1])
                    k = m + 1
                else:
                    fixed.append(seg[k])
                    k += 1
            seg = fixed
        order[a:b + 1] = seg
    out: List[str] = []
    prev: Optional[int] = None
    for k in order:
        t = texts[k]
        flip = (rtl and not inside[k]) or (not rtl and inside[k])
        if flip and t[:1] in _MIRROR:
            t = mirror(t[0]) + t[1:]
        if prev is not None and vis[prev].sup != vis[k].sup and (
                (vis[prev].base.c.isdigit() and vis[k].base.c.isdigit()) or
                bidi_class(vis[prev].base.c) == "L" or bidi_class(vis[k].base.c) == "L"):
            out.append(" ")  # a footnote reference never merges with a number or a Latin word (shelfmarks)
        out.append(t)
        prev = k
    return re.sub(r"[ \t ]{2,}", " ", "".join(out)).strip()


def line_rtl(line: Line, default: bool = True) -> bool:
    """Whether a line has a right-to-left base direction (at least as many R as L letters).

    :param line: Line.
    :param default: Direction of a line without letters (digits and punctuation only): the page's direction.
    :returns: True for right-to-left.
    """
    c = collections.Counter(bidi_class(g.c) for g in line.glyphs)
    if not c["R"] and not c["L"]:
        return default
    return c["R"] >= c["L"]


def mark_superscripts(line: Line, clusters: Sequence[Cluster]) -> int:
    """Flag the clusters of a line set as raised superscripts (footnote references).

    A cluster is a superscript when its glyph is smaller than :data:`SUPERSCRIPT_SIZE` × the line's median size
    and its baseline sits at least 0.15 × that size above the line's baseline.

    :param line: Line.
    :param clusters: Its clusters (flags set in place).
    :returns: Number of superscript clusters.
    """
    med = _median_size(line)
    normal = [g.oy for g in line.glyphs if not g.c.isspace() and g.size >= SUPERSCRIPT_SIZE * med]
    base = statistics.median(normal) if normal else line.baseline
    n = 0
    for c in clusters:
        g = c.base
        if not g.c.isspace() and g.size < SUPERSCRIPT_SIZE * med and g.oy < base - 0.15 * med:
            c.sup = True
            n += 1
    return n


def line_text(line: Line, rtl: Optional[bool] = None) -> str:
    """Logical text of one printed line.

    :param line: Line.
    :param rtl: Base direction (default: :func:`line_rtl`).
    :returns: Text.
    """
    clusters = attach_marks(line.glyphs)
    mark_superscripts(line, clusters)
    return visual_to_logical(clusters, line_rtl(line) if rtl is None else rtl)


# ============================================================================== line joins
_LEAD_NUM = re.compile(r"^(\d{1,3})\s+\S")


@dataclass
class TextLine:
    """A logical line with the geometry needed for joins."""

    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    baseline: float
    size: float
    rtl: bool
    numbered: bool = False
    text_edge: Optional[float] = None
    para_break: bool = False
    med_size: float = 0.0


def _aligned_rtl(line: Line, lines: Sequence[Line]) -> Optional[bool]:
    """Direction of a line without right-to-left letters on a right-to-left page, from its alignment.

    Such a line that ends at the text block's right margin but does not reach its left margin (a shelfmark
    heading) belongs to a right-to-left paragraph; otherwise the line's own letters decide (None).

    :param line: Line.
    :param lines: All lines of the page (for the margins).
    :returns: True, or None to let :func:`line_rtl` decide.
    """
    if any(bidi_class(g.c) == "R" for g in line.glyphs):
        return None
    xs0 = sorted(ln.x0 for ln in lines)
    xs1 = sorted(ln.x1 for ln in lines)
    if len(xs1) < 3:
        return None
    left, right = xs0[len(xs0) // 10], xs1[(9 * len(xs1)) // 10]
    size = max(line.size, 1.0)
    if abs(line.x1 - right) <= 2 * size and line.x0 - left > 4 * size:
        return True
    return None


def _text_line(line: Line, default_rtl: bool = True, force_rtl: Optional[bool] = None) -> TextLine:
    """Logical line with its geometry and its edition line-number status.

    A line is ``numbered`` when its first logical token is a 1–3 digit number set apart from the text by a gap
    of at least one em; ``text_edge`` is then the start edge of the text after the number.

    :param line: Printed line.
    :param default_rtl: The page's direction (for lines without letters).
    :param force_rtl: Direction decided by alignment (:func:`_aligned_rtl`), or None.
    :returns: Text line.
    """
    rtl = force_rtl if force_rtl is not None else line_rtl(line, default_rtl)
    text = line_text(line, rtl)
    tl = TextLine(text=text, x0=line.x0, x1=line.x1, top=line.top, bottom=line.bottom, baseline=line.baseline,
                  size=line.size, rtl=rtl, med_size=_median_size(line))
    if _LEAD_NUM.match(text):
        ink = sorted((g for g in line.glyphs if not g.c.isspace()), key=lambda g: -g.xc if rtl else g.xc)
        digits = [g for g in ink[:3] if g.c.isdigit()]
        rest = ink[len(digits):]
        if digits and rest:
            gap = (min(g.x0 for g in digits) - max(g.x1 for g in rest)) if rtl else \
                (min(g.x0 for g in rest) - max(g.x1 for g in digits))
            if gap >= 0.9 * line.size:
                tl.numbered = True
                tl.text_edge = max(g.x1 for g in rest) if rtl else min(g.x0 for g in rest)
    return tl


def join_continuations(lines: Sequence[TextLine], tol: float = 4.0) -> Tuple[List[TextLine], int]:
    """Join the wrapped continuation lines of numbered edition lines to their line.

    A line without a number that directly follows a numbered line (or its continuation), starts at the numbered
    line's text edge (± ``tol`` points) and sits within 1.5 line pitches below it continues that line.

    :param lines: Lines in reading order.
    :param tol: Edge tolerance in points.
    :returns: ``(lines, number of joins)``.
    """
    out: List[TextLine] = []
    joins = 0
    anchor: Optional[TextLine] = None
    for ln in lines:
        if anchor is not None and out and not ln.numbered and ln.rtl == anchor.rtl and anchor.text_edge is not None:
            prev = out[-1]
            edge = ln.x1 if ln.rtl else ln.x0
            close = 0 < ln.baseline - prev.baseline <= 1.9 * max(prev.size, ln.size)
            if close and abs(edge - anchor.text_edge) <= tol:
                prev.text = prev.text.rstrip() + " " + ln.text.lstrip()
                prev.baseline, prev.bottom = ln.baseline, ln.bottom
                joins += 1
                continue
        out.append(ln)
        anchor = ln if ln.numbered else None
    return out, joins


def join_broken_words(lines: Sequence[TextLine], vocab: Mapping[str, int]) -> Tuple[List[TextLine], int]:
    """Join a Hebrew word broken across two lines.

    Line *i* must end in a Hebrew letter and reach the text block's end margin (a full line), line *i+1* must
    start with a Hebrew letter, hold more than one word and not open a numbered edition line or a paragraph;
    both halves must have at least two letters. The halves are joined when the first ends in the medial form
    of a letter that has a final form (``כ מ נ פ צ``), or when the joined word occurs in the work's vocabulary
    more often than either half.

    :param lines: Lines in reading order.
    :param vocab: Word counts of the work (repaired text).
    :returns: ``(lines, number of joins)``.
    """
    out: List[TextLine] = []
    joins = 0
    left_margin = min((t.x0 for t in lines if t.rtl), default=0.0)
    for ln in lines:
        if out and not ln.numbered and not ln.para_break and len(ln.text.split()) > 1:
            prev = out[-1]
            a = re.search(r"([א-ת]{2,})$", prev.text)
            b = re.match(r"^([א-ת]{2,})", ln.text)
            full = prev.x0 - left_margin <= 2 * max(prev.size, 1.0)
            if a and b and prev.rtl and ln.rtl and full:
                wa, wb = a.group(1), b.group(1)
                w = wa + wb
                medial = wa[-1] in _MEDIAL_WITH_FINAL
                common = vocab.get(w, 0) > max(vocab.get(wa, 0), vocab.get(wb, 0))
                if medial or common:
                    prev.text = prev.text + ln.text
                    prev.baseline, prev.bottom = ln.baseline, ln.bottom
                    joins += 1
                    continue
        out.append(ln)
    return out, joins


def mark_paragraphs(lines: Sequence[TextLine]) -> None:
    """Flag lines that open a paragraph (vertical gap above them > :data:`PARAGRAPH_GAP` × median pitch).

    :param lines: Lines in reading order (flags set in place).
    :returns: None.
    """
    pitches = [b.baseline - a.baseline for a, b in zip(lines, lines[1:]) if 0 < b.baseline - a.baseline < 40]
    med = statistics.median(pitches) if pitches else 14.0
    for a, b in zip(lines, lines[1:]):
        gap = b.baseline - a.baseline
        b.para_break = gap > PARAGRAPH_GAP * med or gap < 0


def assemble_page(lines: Sequence[Line], vocab: Optional[Mapping[str, int]] = None) -> Tuple[str, Dict[str, Any]]:
    """Page text from printed lines: column order, logical lines, joins, paragraph breaks.

    :param lines: Printed lines (top to bottom) from :func:`cluster_lines`.
    :param vocab: Word counts of the work for :func:`join_broken_words` (None: medial-form rule only).
    :returns: ``(text, info)`` with ``gutter``, ``continuation_joins``, ``word_joins``, ``lines`` and ``line_sizes``
        (median font size of each output line, None for a paragraph-break blank line).
    """
    rtl_page = sum(1 for ln in lines if line_rtl(ln)) >= len(lines) / 2 if lines else True
    gutter = detect_gutter(lines)
    ordered = split_columns(lines, gutter, rtl_page) if gutter is not None else list(lines)
    tls = [_text_line(ln, rtl_page, _aligned_rtl(ln, lines) if rtl_page else None) for ln in ordered]
    tls = [t for t in tls if t.text]
    mark_paragraphs(tls)
    tls, cj = join_continuations(tls)
    tls, wj = join_broken_words(tls, vocab or {})
    parts: List[str] = []
    sizes: List[Optional[float]] = []
    for t in tls:
        if t.para_break and parts:
            parts.append("")
            sizes.append(None)
        parts.append(t.text)
        sizes.append(round(t.med_size, 1))
    return "\n".join(parts), {"gutter": gutter, "continuation_joins": cj, "word_joins": wj, "lines": len(tls),
                               "line_sizes": sizes}


# ============================================================================== heuristic fallback
def heuristic_rejoin(text: str) -> str:
    """Re-join a fragmented text layer without geometry.

    Rows are concatenated (as they are: the fragments carry their own spaces) until a whitespace-only row (one
    right after a line number does not count) or a row that is a bare 1–3 digit line number, which opens the
    next line; mirrored brackets are swapped in lines
    with more right-to-left than left-to-right letters, outside left-to-right runs.

    :param text: OCR page text.
    :returns: Re-joined text.
    """
    groups: List[List[str]] = []
    cur: List[str] = []
    for row in (text or "").split("\n"):
        if not row.strip():
            if cur and not (len(cur) == 1 and re.fullmatch(r"\d{1,3} ", cur[0])):
                groups.append(cur)
                cur = []
            continue
        if re.fullmatch(r"\s*\d{1,3}\s*", row):
            if cur:
                groups.append(cur)
            cur = [row.strip() + " "]
            continue
        cur.append(row)
    if cur:
        groups.append(cur)
    out: List[str] = []
    for g in groups:
        s = "".join(g)
        cls = [bidi_class(ch) for ch in s]
        if cls.count("R") >= cls.count("L") and cls.count("R"):
            runs = _runs(cls, "L")
            inside = [False] * len(s)
            for a, b in runs:
                for k in range(a, b + 1):
                    inside[k] = True
            s = "".join(mirror(ch) if not inside[k] else ch for k, ch in enumerate(s))
        out.append(re.sub(r"[ \t]{2,}", " ", s).strip())
    return "\n".join(x for x in out if x)


# ============================================================================== checks
def _char_multiset(text: str) -> collections.Counter:
    """Non-whitespace characters of a text with each bracket mapped to its pair's opening form.

    :param text: Text.
    :returns: Counter.
    """
    canon = {")": "(", "]": "[", "}": "{", ">": "<", "»": "«"}
    return collections.Counter(canon.get(ch, ch) for ch in text if not ch.isspace())


def chars_preserved(original: str, repaired: str) -> bool:
    """Whether a repair kept exactly the original characters (whitespace aside, bracket pairs merged).

    :param original: OCR page text.
    :param repaired: Repaired text.
    :returns: True when the multisets match.
    """
    return _char_multiset(original) == _char_multiset(repaired)


def fragment_stats(text: str) -> Dict[str, Any]:
    """Line statistics of a page text.

    :param text: Text.
    :returns: ``lines`` (non-empty), ``ge3`` (lines with ≥ 3 words) and ``share_ge3``.
    """
    lines = [ln for ln in (text or "").split("\n") if ln.strip()]
    ge3 = sum(1 for ln in lines if len(ln.split()) >= 3)
    return {"lines": len(lines), "ge3": ge3, "share_ge3": round(ge3 / len(lines), 4) if lines else 0.0}


def _sha1(text: str) -> str:
    """sha1 hex digest of a text.

    :param text: Text.
    :returns: Digest.
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ============================================================================== per page / per work
@dataclass
class PageRepair:
    """The repair of one OCR page."""

    page_number: int
    original: str
    text: str
    method: str
    preserved: bool
    info: Dict[str, Any] = field(default_factory=dict)


def repair_geometry(glyphs: Sequence[Glyph], vocab: Optional[Mapping[str, int]] = None) -> Tuple[str, Dict[str, Any]]:
    """Repaired text of one page from its glyphs.

    :param glyphs: Page glyphs (:func:`glyphs_from_rawdict`).
    :param vocab: Word counts of the work.
    :returns: ``(text, info)``.
    """
    return assemble_page(cluster_lines(glyphs), vocab)


def _vocab(texts: Iterable[str]) -> Dict[str, int]:
    """Word counts over texts.

    :param texts: Texts.
    :returns: Counts.
    """
    c: collections.Counter = collections.Counter()
    for t in texts:
        c.update(_WORD.findall(t or ""))
    return dict(c)


def repair_work(ocr_file: Path, work_dir: Path, pdf_path: Optional[Path] = None,
                pages: Optional[Iterable[int]] = None) -> Tuple[List[PageRepair], Dict[str, Any]]:
    """Repair every page of one OCR results file.

    Geometry is used for a page only when the PDF's text layer of that page equals the OCR text exactly; other
    pages fall back to :func:`heuristic_rejoin`. A first geometry pass builds the work vocabulary for the
    broken-word join of the second pass.

    :param ocr_file: ``<stem>_ocr_results.json``.
    :param work_dir: Work directory (where the PDF is searched).
    :param pdf_path: Explicit PDF (default :func:`find_pdf`).
    :param pages: Restrict to these page numbers.
    :returns: ``(repairs, info)``.
    """
    data = json.loads(ocr_file.read_text(encoding="utf-8"))
    ocr: Dict[int, str] = {}
    for rec in data.get("pages") or []:
        try:
            n = int(rec.get("page_number"))
        except (TypeError, ValueError):
            continue
        r = rec.get("ocr_result") or {}
        ocr[n] = str(r.get("full_text") or r.get("page_text") or "")
    wanted = sorted(set(pages) & set(ocr)) if pages is not None else sorted(ocr)
    pdf_path = pdf_path or find_pdf(ocr_file, work_dir)
    pdf = None
    if pdf_path is not None:
        try:
            import fitz  # PyMuPDF: optional; without it every page uses the heuristic
            pdf = fitz.open(str(pdf_path))
        except ImportError:
            LOG.warning("PyMuPDF not installed: heuristic repair only")
    geo: Dict[int, List[Glyph]] = {}
    mismatch: List[int] = []
    if pdf is not None:
        for n in wanted:
            if n > pdf.page_count:
                mismatch.append(n)
                continue
            glyphs, text, _ = load_pdf_page(pdf, n)
            if text == ocr[n]:
                geo[n] = glyphs
            else:
                mismatch.append(n)
    first = {n: repair_geometry(g)[0] for n, g in geo.items()}
    vocab = _vocab(first.values())
    out: List[PageRepair] = []
    for n in wanted:
        if not ocr[n].strip():
            continue
        if n in geo:
            text, info = repair_geometry(geo[n], vocab)
            method = "pdf_geometry"
        else:
            text, info = heuristic_rejoin(ocr[n]), {}
            method = "json_heuristic"
        out.append(PageRepair(page_number=n, original=ocr[n], text=text, method=method,
                              preserved=chars_preserved(ocr[n], text), info=info))
    info = {"ocr_pages": len(ocr), "pages": len(out), "pdf": pdf_path.name if pdf_path else None,
            "text_layer_mismatch": mismatch, "methods": dict(collections.Counter(r.method for r in out))}
    return out, info


def _quality(text: str) -> List[str]:
    """L1 quality reasons of a page text (furniture stripped as L1 does, page-local only).

    :param text: Page text.
    :returns: Failing reasons.
    """
    from src.finetuning.text27b import literature_l1 as _l1  # L1's own rules, read-only use
    t, _ = _l1.strip_furniture(text)
    return _l1.quality_reasons(t)


def report(repairs: Sequence[PageRepair], info: Mapping[str, Any], samples: int = 10, seed: int = 3407
           ) -> Dict[str, Any]:
    """Before / after statistics of a work's repair.

    :param repairs: Page repairs.
    :param info: :func:`repair_work` info.
    :param samples: Number of before / after samples.
    :param seed: Sample seed.
    :returns: Report dict.
    """
    import random
    before = [fragment_stats(r.original) for r in repairs]
    after = [fragment_stats(r.text) for r in repairs]
    lb, la = sum(s["lines"] for s in before), sum(s["lines"] for s in after)
    gb, ga = sum(s["ge3"] for s in before), sum(s["ge3"] for s in after)
    qb = collections.Counter()
    qa = collections.Counter()
    pass_b = pass_a = 0
    for r in repairs:
        rb, ra = _quality(r.original), _quality(r.text)
        pass_b += not rb
        pass_a += not ra
        qb.update(rb)
        qa.update(ra)
    rng = random.Random(seed)
    pick = sorted(rng.sample(range(len(repairs)), min(samples, len(repairs))))
    return {
        **info, "version": VERSION,
        "lines_before": lb, "lines_after": la,
        "share_ge3_before": round(gb / lb, 4) if lb else 0.0, "share_ge3_after": round(ga / la, 4) if la else 0.0,
        "share_lt3_before": round(1 - gb / lb, 4) if lb else 0.0, "share_lt3_after": round(1 - ga / la, 4) if la else 0.0,
        "chars_preserved": sum(1 for r in repairs if r.preserved), "chars_not_preserved": [r.page_number for r in repairs
                                                                                            if not r.preserved],
        "l1_quality_pass_before": pass_b, "l1_quality_pass_after": pass_a,
        "l1_reasons_before": dict(qb), "l1_reasons_after": dict(qa),
        "gutter_pages": [r.page_number for r in repairs if (r.info or {}).get("gutter") is not None],
        "continuation_joins": sum((r.info or {}).get("continuation_joins", 0) for r in repairs),
        "word_joins": sum((r.info or {}).get("word_joins", 0) for r in repairs),
        "samples": [{"page": repairs[i].page_number, "before": repairs[i].original[:600],
                     "after": repairs[i].text[:600]} for i in pick],
    }


def write_repairs(work: str, repairs: Sequence[PageRepair], ocr_file: Path, root: Path, out_root: Path,
                  rep: Mapping[str, Any], pdf_path: Optional[Path] = None) -> Path:
    """Write page texts, manifest and report of one work.

    Pages whose characters were not preserved are not written (and not listed in the manifest).

    :param work: Work directory name.
    :param repairs: Page repairs.
    :param ocr_file: OCR results file (recorded relative to the literature root).
    :param root: Literature root.
    :param out_root: Repair root (``out_dir/ocr_repair``).
    :param rep: :func:`report` output.
    :param pdf_path: PDF used (recorded relative to the literature root).
    :returns: The work's output directory.
    """
    d = out_root / work
    d.mkdir(parents=True, exist_ok=True)
    pages: Dict[str, Any] = {}
    for r in repairs:
        if not r.preserved:
            continue
        name = f"page_{r.page_number:03d}.txt"
        (d / name).write_text(r.text, encoding="utf-8")
        pages[str(r.page_number)] = {"file": name, "ocr_sha1": _sha1(r.original), "method": r.method,
                                     "lines_before": fragment_stats(r.original)["lines"],
                                     "lines_after": fragment_stats(r.text)["lines"]}

    def rel(x: Optional[Path]) -> Optional[str]:
        """Path relative to the literature root.

        :param x: Path or None.
        :returns: Relative string or the name.
        """
        if x is None:
            return None
        return str(x.relative_to(root)) if x.is_relative_to(root) else x.name

    manifest = {"version": VERSION, "work": work, "ocr_file": rel(ocr_file), "ocr_file_name": ocr_file.name,
                "pdf_file": rel(pdf_path), "created": time.strftime("%Y-%m-%d %H:%M"), "pages": pages}
    (d / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (d / REPORT_NAME).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


def default_repair_root(p: Optional[_paths.Paths] = None) -> Path:
    """The repair root of the build (``paths.out_dir / ocr_repair``).

    :param p: Build paths.
    :returns: Directory (may not exist).
    """
    p = p or _paths.load_paths()
    return p.out(REPAIR_DIRNAME)


def load_repairs(repair_root: Path, work: str) -> Tuple[Optional[str], Dict[int, Tuple[str, str, str]]]:
    """Repaired pages of one work.

    :param repair_root: Repair root.
    :param work: Work directory name.
    :returns: ``(OCR file name or None, {page_number: (repaired text, original sha1, method)})``.
    """
    d = Path(repair_root) / work
    mpath = d / MANIFEST_NAME
    if not mpath.exists():
        return None, {}
    man = json.loads(mpath.read_text(encoding="utf-8"))
    out: Dict[int, Tuple[str, str, str]] = {}
    for n, rec in (man.get("pages") or {}).items():
        f = d / rec["file"]
        if f.exists():
            out[int(n)] = (f.read_text(encoding="utf-8"), rec["ocr_sha1"], rec.get("method", "repair"))
    return man.get("ocr_file_name"), out


def _ocr_files(work_dir: Path) -> List[Path]:
    """OCR results files of a work.

    :param work_dir: Work directory.
    :returns: Sorted paths.
    """
    return sorted(work_dir.rglob("*_ocr_results.json"))


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI: repair one work's OCR text layer, or print one repaired page.

    :param argv: Arguments (default ``sys.argv[1:]``).
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", required=True, help="top-level literature work directory")
    ap.add_argument("--show", type=int, default=None, help="print this page before / after and exit")
    ap.add_argument("--pages", default=None, help="restrict to page numbers a-b")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = _paths.load_paths()
    work_dir = p.academic_literature / args.work
    files = _ocr_files(work_dir)
    if len(files) != 1:
        raise SystemExit(f"expected one OCR results file in the work, found {len(files)}")
    pages = None
    if args.pages:
        a, b = (int(x) for x in args.pages.split("-"))
        pages = range(a, b + 1)
    if args.show is not None:
        pages = [args.show]
    t0 = time.time()
    repairs, info = repair_work(files[0], work_dir, pages=pages)
    if args.show is not None:
        for r in repairs:
            print(f"=== page {r.page_number} ({r.method}, preserved={r.preserved}) BEFORE ===\n{r.original}\n"
                  f"=== AFTER ===\n{r.text}")
        return
    rep = report(repairs, info)
    LOG.info("[%5.1fs] %d pages repaired", time.time() - t0, len(repairs))
    if not args.dry_run:
        d = write_repairs(args.work, repairs, files[0], p.academic_literature, default_repair_root(p), rep,
                          find_pdf(files[0], work_dir))
        LOG.info("wrote %s", d.relative_to(p.repo) if d.is_relative_to(p.repo) else d.name)
    print(json.dumps({k: v for k, v in rep.items() if k != "samples"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
