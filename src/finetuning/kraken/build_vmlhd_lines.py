# File name: build_vmlhd_lines.py
# Date: 10/28/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Kraken baseline lines (PAGE XML) from the VML-HD sub-word annotations.

VML-HD (Kassis et al., ASAR 2017; research use only, CC BY-NC-ND 4.0) is 668 pages of five
handwritten Arabic books (1088 to 1451) with a box and a transcript for every part of a word (PAW,
one connected letter group). It has no lines, no word boundaries and no reading order: the element
IDs follow the annotators' passes, not the text, and a third of the ``OriginX/OriginY`` points are
``(0, 0)``. This builder derives all three from the box geometry:

* **Lines** (:func:`group_lines`): the boxes no taller than the page's median PAW height ("core"
  boxes: body-zone letters, no tall alif) give line centres as density peaks of their vertical
  centres; each line gets a linear fit ``y = a + b * x`` (skewed lines), refined three times. Every
  box then joins the line whose body band (``centre +/- core height / 2``) it overlaps most, so a
  tall alif reaching into the line above stays on its own line. A line is split where two
  neighbouring PAWs are more than :data:`SPLIT_GAP` core heights apart (margin notes, columns).
* **Reading order** (:func:`reading_order`): right edge, right to left.
* **Spaces** (:func:`join_paws`): after a PAW that ends in a letter joining to the left (or in ة / ى,
  which only end words) a space is certain, because inside a word that letter would connect to the
  next one. After a non-joining letter (ا د ذ ر ز و ء and hamza forms) the word may go on; those
  boundaries are decided per run by a unigram word segmentation over a lexicon of the public
  Arabic HTR sets already on the NAS (RASAM, Agapet, Iskandar; :func:`build_lexicon`) plus a weak
  prior on the gap width. The Arabic benchmark scores letters only, so spaces do not move its
  numbers; they matter for any later VLM rows.

Geometry (:func:`line_polygon`, :func:`line_baseline`): the polygon is the upper and lower envelope
of the padded PAW boxes (each widened by half a core height, so gaps leave no notches), never
narrower than the body band; the baseline is the line fit moved down
to the median bottom of the line's isolated alifs (the page median where a line has none), drawn left
to right like the other Arabic sets, straight.

A line is dropped when a PAW has an empty transcript or a character that is not an Arabic letter
(the source has a few ``\\``, ``q``, ``k``, ``1``, ``.``), when its body band holds black ink that no
box covers (:func:`unannotated_ink`: on some pages the annotators skipped whole PAWs, so the text
would miss letters the crop shows), or when :func:`build_arabic_lines.clean_target` rejects the text. Splits are by page, stratified by book (:func:`build_arabic_lines.assign_splits`).

Output (``--out-root``) as in :mod:`src.finetuning.kraken.build_arabic_lines`::

    page_xml/vmlhd/<stem>.xml        PAGE 2019 with TextLine Coords + Baseline + Unicode, absolute imageFilename
    lists/vmlhd_train.lst, vmlhd_val.lst
    stats.json                       entry "vmlhd" merged in
    lexicon.json                     the word counts used for the spaces

Usage (repo root)::

    .venv/bin/python -m src.finetuning.kraken.build_vmlhd_lines --out-root DIR [--limit N] [--workers 4]
"""
import argparse
import glob
import json
import logging
import math
import re
import os
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from lxml import etree
from PIL import Image
from scipy import ndimage

from src.finetuning.kraken import build_arabic_lines as bal

logger = logging.getLogger(__name__)



def _env_dir(name: str) -> Optional[Path]:
    """A directory named by an environment variable, or ``None`` when the variable is unset or empty.

    :param name: Variable name.
    :type name: str
    :return: The path, or ``None``.
    :rtype: Optional[Path]
    """
    value = os.environ.get(name, "").strip()
    return Path(value) if value else None


# Storage roots come from the environment so that no storage layout is written into the source:
#   VMLHD_ROOT             the VML-HD folder (``xml/<book>/<page>.xml`` and ``pages/<book>/<page>.png``); ``--root``
#                          overrides it
#   ARABIC_LINES_PAGE_XML  the ``page_xml`` folder of the Arabic line set whose ``rasam`` / ``agapet`` / ``iskandar``
#                          sub-folders supply the word lexicon
VMLHD_DIR = _env_dir("VMLHD_ROOT")
_PAGE_XML = _env_dir("ARABIC_LINES_PAGE_XML")
LEXICON_DIRS: Tuple[Path, ...] = tuple(_PAGE_XML / s for s in ("rasam", "agapet", "iskandar")) if _PAGE_XML else ()
SET_NAME = "vmlhd"
LICENCE = "research use only (CC BY-NC-ND 4.0 badge); internal training, never redistributed"
PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"

SPLIT_GAP = 4.0            # core heights between neighbouring PAWs that split a line (verse markers leave ~2)
PEAK_MIN_MASS = 1.5        # kernel mass a line-centre peak needs (about two PAWs)
PEAK_MIN_SEP = 0.7         # core heights between two line centres
FIT_MIN_PAWS = 5           # core PAWs a line needs for a sloped fit
MAX_SLOPE = 0.03         # ~1.7 degrees: a steeper fit is two lines mixed, not skew
MAX_RECOVERY_ROUNDS = 3    # passes that add lines the peaks missed (:func:`_missed_lines`)
UNEXPLAINED_OVERLAP = 0.25 # a PAW overlapping every band by less than this share of its height is loose
MIN_MISSED_PAWS = 3        # loose PAWs a recovered line needs
PAD = 3                    # px around every PAW box in the polygon
ENVELOPE_REACH = 0.5       # core heights each box is widened by on both sides for the outline (smoother, like a
                           # segmenter polygon: the black mask kraken paints outside it then shows no notches)
FRAME_SLACK = 4            # px a box may exceed the image before the page counts as another frame

JOINING_LEFT = frozenset("بتثجحخسشصضطظعغفقكلمنهيئ")   # connect to the next letter inside a word
WORD_FINAL_ONLY = frozenset("ةى")
NON_JOINING = frozenset("اأإآدذرزوؤء")
ARABIC_LETTERS = JOINING_LEFT | WORD_FINAL_ONLY | NON_JOINING

LEX_ALPHA = 0.1
SINGLE_LETTER_LOGP = -25.0  # a lone letter is never a word in Arabic orthography (clitics attach)
GAP_MU = 0.08               # gap / core height at which the gap prior is neutral
GAP_SCALE = 0.1
GAP_WEIGHT = 1.0
WORD_BONUS = 6.0           # per word (>0 splits runs); 2026-10-28: boundary P 0.90 / R 0.90 on 17 known Quran lines (0 → 0.95 / 0.70)

MIN_BLOB = 0.04            # core heights²: smaller unannotated blobs are dots / short vowels
MAX_INK_SATURATION = 25    # median max-min channel spread of black ink (verse rosettes: 30 to 70)
MAX_MISSING_BLOB = 0.2     # core heights²: a larger unannotated black blob is a skipped letter group
MAX_MISSING_TOTAL = 0.35   # core heights²: total unannotated black ink a kept line may have

DROP_EMPTY_PAW = "empty paw transcript"
DROP_SYMBOL = "non-arabic symbol in a paw"
DROP_UNANNOTATED = "unannotated ink in the line"

_HARAKAT = re.compile("[ً-ْٰـ]")
_NON_LETTER = re.compile("[^ء-يً-ْٰ]")


# ----------------------------------------------------------------------------- records

@dataclass(frozen=True)
class Paw:
    """One annotated part of a word.

    :param x: Left edge (px).
    :param y: Top edge (px).
    :param w: Width (px).
    :param h: Height (px).
    :param text: Transcript, stripped (may be empty in the source).
    """
    x: int
    y: int
    w: int
    h: int
    text: str

    @property
    def right(self) -> int:
        """Right edge (px).

        :return: ``x + w``.
        :rtype: int
        """
        return self.x + self.w

    @property
    def bottom(self) -> int:
        """Bottom edge (px).

        :return: ``y + h``.
        :rtype: int
        """
        return self.y + self.h

    @property
    def cx(self) -> float:
        """Horizontal centre (px).

        :return: ``x + w / 2``.
        :rtype: float
        """
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        """Vertical centre (px).

        :return: ``y + h / 2``.
        :rtype: float
        """
        return self.y + self.h / 2


@dataclass
class Line:
    """PAWs of one text line in reading order, with the line's centre fit.

    :param paws: PAWs right to left.
    :param a: Fitted centre at ``x = xm``.
    :param b: Fitted slope (px per px).
    :param xm: Reference x of the fit.
    :param core: The page's core PAW height.
    """
    paws: List[Paw]
    a: float
    b: float
    xm: float
    core: float

    def centre(self, x: float) -> float:
        """Fitted line centre at ``x``.

        :param x: Horizontal position (px).
        :type x: float
        :return: Centre y (px).
        :rtype: float
        """
        return self.a + self.b * (x - self.xm)


# ----------------------------------------------------------------------------- source

def load_paws(xml_path: Path) -> List[Paw]:
    """PAW boxes of one VML-HD page XML (``ArrayOfDocumentElement``).

    :param xml_path: Page XML.
    :type xml_path: Path
    :return: Every ``PartOfWord`` element (other element types are ignored).
    :rtype: List[Paw]
    """
    parser = etree.XMLParser(resolve_entities=False, no_network=True)   # downloaded data: no entity expansion
    root = etree.parse(str(xml_path), parser).getroot()
    out = []
    for el in root:
        if el.findtext("ElementType") != "PartOfWord":
            continue
        out.append(Paw(int(el.findtext("X")), int(el.findtext("Y")), int(el.findtext("Width")),
                       int(el.findtext("Height")), (el.findtext("Transcript") or "").strip()))
    return out


def page_sources(root: Optional[Path] = None) -> List[Tuple[str, Path, Optional[Path]]]:
    """Every annotated page with its image.

    :param root: VML-HD folder with ``xml/<book>/<page>.xml`` and ``pages/<book>/<page>.png``; ``None`` = ``$VMLHD_ROOT``.
    :type root: Path
    :return: ``[(book, xml path, image path or None), ...]`` sorted by book and page.
    :rtype: List[Tuple[str, Path, Optional[Path]]]
    """
    root = root if root is not None else VMLHD_DIR
    if root is None:
        raise SystemExit("VML-HD root unknown: pass --root or set VMLHD_ROOT")
    out = []
    for xml in sorted(Path(p) for p in glob.glob(str(root / "xml" / "*" / "*.xml"))):
        image = root / "pages" / xml.parent.name / f"{xml.stem}.png"
        out.append((xml.parent.name, xml, image if image.exists() else None))
    return out


# ----------------------------------------------------------------------------- lines

def density_peaks(values: Sequence[float], bandwidth: float, min_sep: float, min_mass: float = PEAK_MIN_MASS) -> List[float]:
    """Peaks of a Gaussian kernel density, strongest first, at least ``min_sep`` apart.

    :param values: Sample positions.
    :type values: Sequence[float]
    :param bandwidth: Kernel sigma.
    :type bandwidth: float
    :param min_sep: Minimum distance between two kept peaks.
    :type min_sep: float
    :param min_mass: Minimum density at a kept peak.
    :type min_mass: float
    :return: Peak positions, ascending.
    :rtype: List[float]
    """
    v = np.asarray(values, dtype=float)
    grid = np.arange(v.min() - 3 * bandwidth, v.max() + 3 * bandwidth, 1.0)
    dens = np.exp(-0.5 * ((grid[:, None] - v[None, :]) / bandwidth) ** 2).sum(axis=1)
    local = [i for i in range(1, len(grid) - 1) if dens[i] >= dens[i - 1] and dens[i] > dens[i + 1]]
    keep: List[int] = []
    for i in sorted(local, key=lambda i: -dens[i]):
        if dens[i] >= min_mass and all(abs(grid[i] - grid[j]) >= min_sep for j in keep):
            keep.append(i)
    return sorted(float(grid[i]) for i in keep)


def band_overlap(paw: Paw, centre: float, half: float) -> float:
    """Vertical overlap of a PAW box with the band ``centre +/- half``.

    :param paw: The PAW.
    :type paw: Paw
    :param centre: Band centre (px).
    :type centre: float
    :param half: Half band height (px).
    :type half: float
    :return: Overlap in px (0 when disjoint).
    :rtype: float
    """
    return max(0.0, min(paw.bottom, centre + half) - max(paw.y, centre - half))


def _assign(paws: Sequence[Paw], fits: Sequence[Tuple[float, float]], xm: float, half: float) -> List[List[Paw]]:
    """Each PAW to the line whose band it overlaps most (ties: nearest centre).

    :param paws: PAWs to assign.
    :type paws: Sequence[Paw]
    :param fits: ``(a, b)`` per line.
    :type fits: Sequence[Tuple[float, float]]
    :param xm: Reference x of the fits.
    :type xm: float
    :param half: Half band height.
    :type half: float
    :return: PAWs per line, in ``fits`` order.
    :rtype: List[List[Paw]]
    """
    lines: List[List[Paw]] = [[] for _ in fits]
    for p in paws:
        centres = [a + b * (p.cx - xm) for a, b in fits]
        best = max(range(len(fits)), key=lambda i: (band_overlap(p, centres[i], half), -abs(p.cy - centres[i])))
        lines[best].append(p)
    return lines


def _fit(paws: Sequence[Paw], xm: float) -> Tuple[float, float]:
    """Linear centre fit of a line's PAWs (flat when there are too few).

    :param paws: PAWs of the line.
    :type paws: Sequence[Paw]
    :param xm: Reference x.
    :type xm: float
    :return: ``(a, b)``, slope clipped to :data:`MAX_SLOPE`.
    :rtype: Tuple[float, float]
    """
    if len(paws) < FIT_MIN_PAWS or len({p.cx for p in paws}) < 2:
        return statistics.median(p.cy for p in paws), 0.0
    xs = np.array([(p.cx - xm) / 1000.0 for p in paws])
    b, a = np.polyfit(xs, np.array([p.cy for p in paws]), 1)
    return float(a), float(np.clip(b / 1000.0, -MAX_SLOPE, MAX_SLOPE))


def _refine(paws: Sequence[Paw], fits: Sequence[Tuple[float, float]], xm: float, core: float,
            rounds: int = 3) -> List[Tuple[float, float]]:
    """Re-assign every PAW and re-fit each line (on its core PAWs when it has enough, else on all).

    :param paws: All PAWs of the page.
    :type paws: Sequence[Paw]
    :param fits: Current ``(a, b)`` per line.
    :type fits: Sequence[Tuple[float, float]]
    :param xm: Reference x.
    :type xm: float
    :param core: Core PAW height.
    :type core: float
    :param rounds: Assign / fit rounds.
    :type rounds: int
    :return: Fits of the lines that kept at least one PAW.
    :rtype: List[Tuple[float, float]]
    """
    fits = list(fits)
    for _ in range(rounds):
        new = []
        for group in _assign(paws, fits, xm, core / 2):
            if group:
                short = [p for p in group if p.h <= core]
                new.append(_fit(short if len(short) >= FIT_MIN_PAWS else group, xm))
        fits = new
    return fits


def _missed_lines(paws: Sequence[Paw], fits: Sequence[Tuple[float, float]], xm: float, core: float) -> List[float]:
    """Centres of lines the peaks missed: clusters of PAWs that overlap no line's body band.

    A line written mostly in tall letters has few core PAWs and can lose its peak; its PAWs then
    overlap no band (or only a neighbour's edge).

    :param paws: All PAWs of the page.
    :type paws: Sequence[Paw]
    :param fits: Current ``(a, b)`` per line.
    :type fits: Sequence[Tuple[float, float]]
    :param xm: Reference x.
    :type xm: float
    :param core: Core PAW height.
    :type core: float
    :return: New centres (at ``xm``) at least :data:`PEAK_MIN_SEP` core heights from every line.
    :rtype: List[float]
    """
    half = core / 2
    loose = []
    for p in paws:
        best = max((band_overlap(p, a + b * (p.cx - xm), half) for a, b in fits), default=0.0)
        if best < UNEXPLAINED_OVERLAP * min(p.h, core):
            loose.append(p)
    if len(loose) < MIN_MISSED_PAWS:
        return []
    found = density_peaks([p.cy for p in loose], core / 4, PEAK_MIN_SEP * core, min_mass=MIN_MISSED_PAWS - 0.5)
    return [c for c in found if all(abs(c - a) >= PEAK_MIN_SEP * core for a, _ in fits)]


def reading_order(paws: Sequence[Paw]) -> List[Paw]:
    """PAWs of one line right to left (by right edge, then by centre).

    :param paws: PAWs of one line.
    :type paws: Sequence[Paw]
    :return: Sorted copy.
    :rtype: List[Paw]
    """
    return sorted(paws, key=lambda p: (-p.right, -p.cx))


def split_at_gaps(paws: Sequence[Paw], core: float, split_gap: float = SPLIT_GAP) -> List[List[Paw]]:
    """Split a reading-ordered line where two neighbouring PAWs are far apart.

    :param paws: PAWs right to left.
    :type paws: Sequence[Paw]
    :param core: Core PAW height.
    :type core: float
    :param split_gap: Gap (in core heights) that splits.
    :type split_gap: float
    :return: Segments, right to left.
    :rtype: List[List[Paw]]
    """
    segments: List[List[Paw]] = []
    for p in paws:
        if segments and min(q.x for q in segments[-1]) - p.right > split_gap * core:
            segments.append([p])
        elif segments:
            segments[-1].append(p)
        else:
            segments.append([p])
    return segments


def group_lines(paws: Sequence[Paw]) -> List[Line]:
    """Text lines of one page, top to bottom (see the module docstring).

    :param paws: All PAWs of the page.
    :type paws: Sequence[Paw]
    :return: Lines with PAWs right to left; empty for a page without PAWs.
    :rtype: List[Line]
    """
    if not paws:
        return []
    core = float(sorted(p.h for p in paws)[len(paws) // 2])
    core_paws = [p for p in paws if p.h <= core]
    xm = statistics.median(p.cx for p in paws)
    half = core / 2
    fits = [(c, 0.0) for c in density_peaks([p.cy for p in core_paws], core / 4, PEAK_MIN_SEP * core)]
    for _ in range(MAX_RECOVERY_ROUNDS + 1):
        fits = _refine(paws, fits, xm, core)
        found = _missed_lines(paws, fits, xm, core)
        if not found:
            break
        fits = sorted(fits + [(c, 0.0) for c in found])
    lines = []
    for group in _assign(paws, fits, xm, half):
        if not group:
            continue
        for seg in split_at_gaps(reading_order(group), core):
            seg_core = [p for p in seg if p.h <= core] or seg
            a, b = _fit(seg_core, xm)
            lines.append(Line(seg, a, b, xm, core))
    lines.sort(key=lambda ln: (ln.centre(xm), -max(p.right for p in ln.paws)))
    return lines


# ----------------------------------------------------------------------------- spaces

def fold_word(word: str) -> str:
    """Lexicon key of a word: no harakat or tatweel, hamza seats and final forms folded.

    :param word: Arabic word.
    :type word: str
    :return: Folded word (أ إ آ ٱ → ا, ى → ي, ة → ه, ؤ → و, ئ → ي).
    :rtype: str
    """
    w = re.sub("[أإآٱ]", "ا", _HARAKAT.sub("", word))
    return w.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")


def build_lexicon(dirs: Optional[Sequence[Path]] = None) -> Counter:
    """Folded word counts (two letters or more) from the ``Unicode`` of PAGE XML files.

    :param dirs: Folders of PAGE XML; ``None`` = the ``$ARABIC_LINES_PAGE_XML`` sub-folders.
    :type dirs: Sequence[Path]
    :return: ``Counter`` of folded words.
    :rtype: Counter
    """
    dirs = tuple(dirs) if dirs is not None else LEXICON_DIRS
    if not dirs:
        raise SystemExit("lexicon PAGE-XML folders unknown: set ARABIC_LINES_PAGE_XML")
    counts: Counter = Counter()
    for d in dirs:
        for f in sorted(glob.glob(str(d / "*.xml"))):
            for el in etree.parse(f).iter("{*}Unicode"):
                for w in (el.text or "").split():
                    w = fold_word(_NON_LETTER.sub("", w))
                    if len(w) >= 2:
                        counts[w] += 1
    return counts


class WordModel:
    """Unigram log-probabilities over a folded lexicon.

    :param counts: Folded word counts.
    :type counts: Counter
    :param alpha: Additive smoothing.
    :type alpha: float
    """

    def __init__(self, counts: Counter, alpha: float = LEX_ALPHA) -> None:
        self.counts = counts
        self.alpha = alpha
        self.denom = sum(counts.values()) + alpha * max(len(counts), 1)

    def logp(self, word: str) -> float:
        """Log-probability of a word (a lone letter is :data:`SINGLE_LETTER_LOGP`).

        :param word: Unfolded word.
        :type word: str
        :return: Log-probability (unknown words get the smoothing mass).
        :rtype: float
        """
        w = fold_word(word)
        if len(w) <= 1:
            return SINGLE_LETTER_LOGP
        return math.log((self.counts.get(w, 0) + self.alpha) / self.denom)


def gap_prior(gap: float, core: float) -> float:
    """Score added when a space is put into a gap (positive favours a space).

    :param gap: Horizontal gap between two PAWs (px; negative when they overlap).
    :type gap: float
    :param core: Core PAW height.
    :type core: float
    :return: ``GAP_WEIGHT * clip((gap / core - GAP_MU) / GAP_SCALE, -2, 2)``.
    :rtype: float
    """
    return GAP_WEIGHT * max(-2.0, min(2.0, (gap / core - GAP_MU) / GAP_SCALE))


def segment_run(pieces: Sequence[str], gaps: Sequence[float], core: float, model: WordModel) -> List[str]:
    """Best split of a run of PAWs whose inner boundaries may or may not be spaces.

    Exact dynamic programme over word ends: a word is any span of consecutive pieces, scored by
    :meth:`WordModel.logp`, plus :func:`gap_prior` for every boundary that becomes a space.

    :param pieces: PAW transcripts in reading order.
    :type pieces: Sequence[str]
    :param gaps: Gap after each piece but the last (``len(pieces) - 1`` values).
    :type gaps: Sequence[float]
    :param core: Core PAW height.
    :type core: float
    :param model: Word model.
    :type model: WordModel
    :return: Words.
    :rtype: List[str]
    """
    n = len(pieces)
    best = [-math.inf] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0.0
    for end in range(1, n + 1):
        for start in range(end):
            space = gap_prior(gaps[start - 1], core) if start > 0 else 0.0
            score = best[start] + model.logp("".join(pieces[start:end])) + WORD_BONUS + space
            if score > best[end]:
                best[end], back[end] = score, start
    words, end = [], n
    while end > 0:
        words.append("".join(pieces[back[end]:end]))
        end = back[end]
    return words[::-1]


def join_paws(paws: Sequence[Paw], core: float, model: WordModel) -> str:
    """Line text from reading-ordered PAWs (certain spaces + :func:`segment_run` for the rest).

    :param paws: PAWs right to left, every transcript non-empty.
    :type paws: Sequence[Paw]
    :param core: Core PAW height.
    :type core: float
    :param model: Word model.
    :type model: WordModel
    :return: Space-separated line text.
    :rtype: str
    """
    words: List[str] = []
    run: List[str] = []
    gaps: List[float] = []
    for i, p in enumerate(paws):
        run.append(p.text)
        last = i == len(paws) - 1
        if last or p.text[-1] in JOINING_LEFT or p.text[-1] in WORD_FINAL_ONLY:
            words.extend(segment_run(run, gaps, core, model))
            run, gaps = [], []
        else:
            gaps.append(float(p.x - paws[i + 1].right))
    return " ".join(words)


# ----------------------------------------------------------------------------- geometry

def line_polygon(line: Line, width: int, height: int, pad: int = PAD) -> List[Tuple[int, int]]:
    """Envelope polygon of a line's padded PAW boxes, never narrower than the body band.

    Each box is widened by :data:`ENVELOPE_REACH` core heights on both sides (within the line's own x
    range) before the envelope is taken, so short gaps between PAWs do not cut notches into it.

    :param line: The line.
    :type line: Line
    :param width: Image width (clip).
    :type width: int
    :param height: Image height (clip).
    :type height: int
    :param pad: Padding around each box (px).
    :type pad: int
    :return: Clockwise points: the top edge left to right, then the bottom edge right to left.
    :rtype: List[Tuple[int, int]]
    """
    half = line.core / 2
    left = max(0, min(p.x for p in line.paws) - pad)
    right = min(width - 1, max(p.right for p in line.paws) + pad)
    reach = int(round(ENVELOPE_REACH * line.core))   # widen each box so narrow gaps do not notch the outline
    boxes = [(max(left, p.x - pad - reach), min(right, p.right + pad + reach), p.y - pad, p.bottom + pad) for p in line.paws]
    xs = sorted({b[0] for b in boxes} | {b[1] for b in boxes})
    top, bottom = [], []
    for x0, x1 in zip(xs, xs[1:]):
        mid = (x0 + x1) / 2
        cover = [b for b in boxes if b[0] <= mid <= b[1]]
        c = line.centre(mid)
        t = min([b[2] for b in cover] + [c - half])
        u = max([b[3] for b in cover] + [c + half])
        t, u = int(round(max(0, t))), int(round(min(height - 1, u)))
        top += [(x0, t), (x1, t)]
        bottom += [(x0, u), (x1, u)]
    pts = top + bottom[::-1]
    out: List[Tuple[int, int]] = []
    for pt in pts:   # drop repeated points and middle points of straight runs
        if out and pt == out[-1]:
            continue
        if len(out) >= 2 and (out[-2][1] == out[-1][1] == pt[1] or out[-2][0] == out[-1][0] == pt[0]):
            out[-1] = pt
            continue
        out.append(pt)
    return out


def alif_offset(lines: Sequence[Line]) -> Optional[float]:
    """Median distance from the line centre down to the bottom of isolated alifs, over lines.

    :param lines: Lines of a page.
    :type lines: Sequence[Line]
    :return: Offset in px, or None when the page has no isolated alif.
    :rtype: Optional[float]
    """
    offsets = [p.bottom - ln.centre(p.cx) for ln in lines for p in ln.paws if p.text in ("ا", "أ", "إ", "آ")]
    return statistics.median(offsets) if offsets else None


def line_baseline(line: Line, page_offset: float, width: int, height: int) -> List[Tuple[int, int]]:
    """Straight baseline from the left end to the right end of a line.

    :param line: The line.
    :type line: Line
    :param page_offset: Offset below the centre used when the line has no isolated alif.
    :type page_offset: float
    :param width: Image width (clip).
    :type width: int
    :param height: Image height (clip).
    :type height: int
    :return: ``[(x_left, y), (x_right, y)]``, inside the body band.
    :rtype: List[Tuple[int, int]]
    """
    own = alif_offset([line])
    off = min(line.core / 2 - 1, max(0.0, own if own is not None else page_offset))
    x0 = max(0, min(p.x for p in line.paws))
    x1 = min(width - 1, max(p.right for p in line.paws))
    clip_y = lambda y: int(round(min(height - 1, max(0, y))))
    return [(x0, clip_y(line.centre(x0) + off)), (x1, clip_y(line.centre(x1) + off))]


# ----------------------------------------------------------------------------- annotation coverage

def ink_mask(image: Image.Image) -> np.ndarray:
    """Black-ink pixels of a page: dark in every channel (red / gold marks are not ink here).

    :param image: Page image.
    :type image: Image.Image
    :return: Boolean array (height, width), True where the brightest channel is below the Otsu
        threshold of the brightest-channel image.
    :rtype: np.ndarray
    """
    bright = np.asarray(image.convert("RGB"), dtype=np.uint8).max(axis=2)
    hist = np.bincount(bright.ravel(), minlength=256).astype(float)
    total, cum, cum_mean = hist.sum(), np.cumsum(hist), np.cumsum(hist * np.arange(256))
    w0, w1 = cum, total - cum
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (cum_mean[-1] * w0 / total - cum_mean) ** 2 / (w0 * w1 / total)
    threshold = int(np.nanargmax(np.where((w0 > 0) & (w1 > 0), between, np.nan)))
    return bright <= threshold


def box_mask(paws: Sequence[Paw], shape: Tuple[int, int], pad: int = PAD) -> np.ndarray:
    """Pixels inside any padded PAW box of the page.

    :param paws: All PAWs of the page.
    :type paws: Sequence[Paw]
    :param shape: ``(height, width)``.
    :type shape: Tuple[int, int]
    :param pad: Padding (px).
    :type pad: int
    :return: Boolean array.
    :rtype: np.ndarray
    """
    mask = np.zeros(shape, dtype=bool)
    for p in paws:
        mask[max(0, p.y - pad):max(0, p.bottom + pad), max(0, p.x - pad):max(0, p.right + pad)] = True
    return mask


def band_owner(lines: Sequence[Line], shape: Tuple[int, int]) -> np.ndarray:
    """Which line's body band (over its x range) each pixel lies in.

    :param lines: Lines of the page.
    :type lines: Sequence[Line]
    :param shape: ``(height, width)``.
    :type shape: Tuple[int, int]
    :return: int32 array: line index + 1, 0 outside every band (a later line wins an overlap).
    :rtype: np.ndarray
    """
    height, width = shape
    owner = np.zeros(shape, dtype=np.int32)
    for i, line in enumerate(lines):
        half = line.core / 2
        for x in range(max(0, min(p.x for p in line.paws)), min(width, max(p.right for p in line.paws))):
            c = line.centre(x)
            owner[max(0, int(c - half)):max(0, min(height, int(c + half) + 1)), x] = i + 1
    return owner


def unannotated_ink(lines: Sequence[Line], image: Image.Image, paws: Sequence[Paw]) -> List[Tuple[float, float]]:
    """Black ink in each line's body band that no PAW box covers, as connected blobs.

    Annotators skipped whole PAWs on some pages; the line's text then misses letters its crop shows.
    Blobs smaller than :data:`MIN_BLOB` core heights squared (dots, short vowels) and coloured blobs
    (median channel spread at or above :data:`MAX_INK_SATURATION`: gilded / blue verse rosettes,
    red marks) are ignored; boxes of any line count as cover (descenders of the line above).

    :param lines: Lines of the page.
    :type lines: Sequence[Line]
    :param image: Page image.
    :type image: Image.Image
    :param paws: All PAWs of the page.
    :type paws: Sequence[Paw]
    :return: Per line ``(largest blob, total)`` in core heights squared.
    :rtype: List[Tuple[float, float]]
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    ink = ink_mask(image)
    owner = band_owner(lines, ink.shape)
    labels, _ = ndimage.label(ink & ~box_mask(paws, ink.shape) & (owner > 0))
    out = [(0.0, 0.0) for _ in lines]
    for i, sl in enumerate(ndimage.find_objects(labels), 1):
        if sl is None:
            continue
        blob = labels[sl] == i
        idx = int(np.bincount(owner[sl][blob]).argmax()) - 1
        core2 = lines[idx].core ** 2
        area = float(blob.sum()) / core2
        if area < MIN_BLOB or float(np.median(saturation[sl][blob])) >= MAX_INK_SATURATION:
            continue
        largest, total = out[idx]
        out[idx] = (max(largest, area), total + area)
    return out


# ----------------------------------------------------------------------------- page

def line_reason(line: Line) -> str:
    """Why a line cannot be a target ("" when it can).

    :param line: The line.
    :type line: Line
    :return: :data:`DROP_EMPTY_PAW`, :data:`DROP_SYMBOL` or ``""``.
    :rtype: str
    """
    if any(not p.text for p in line.paws):
        return DROP_EMPTY_PAW
    if any(ch not in ARABIC_LETTERS for p in line.paws for ch in p.text):
        return DROP_SYMBOL
    return ""


def frame_ok(paws: Sequence[Paw], width: int, height: int, slack: int = FRAME_SLACK) -> bool:
    """Whether the boxes fit the image (same coordinate frame).

    :param paws: PAWs of the page.
    :type paws: Sequence[Paw]
    :param width: Image width.
    :type width: int
    :param height: Image height.
    :type height: int
    :param slack: Tolerance (px).
    :type slack: int
    :return: True when every box lies within the image plus ``slack``.
    :rtype: bool
    """
    return all(p.x >= -slack and p.y >= -slack and p.right <= width + slack and p.bottom <= height + slack for p in paws)


def page_xml(lines: Sequence[Tuple[Line, str]], image_path: Path, width: int, height: int,
             page_offset: float) -> bytes:
    """PAGE 2019 XML with one region and the kept lines.

    :param lines: ``(line, text)`` of the kept lines, top to bottom.
    :type lines: Sequence[Tuple[Line, str]]
    :param image_path: Absolute image path (``imageFilename``).
    :type image_path: Path
    :param width: Image width.
    :type width: int
    :param height: Image height.
    :type height: int
    :param page_offset: Baseline offset for lines without an isolated alif.
    :type page_offset: float
    :return: Serialised XML.
    :rtype: bytes
    """
    ns = "{%s}" % PAGE_NS
    pcgts = etree.Element(ns + "PcGts", nsmap={None: PAGE_NS})
    meta = etree.SubElement(pcgts, ns + "Metadata")
    etree.SubElement(meta, ns + "Creator").text = "build_vmlhd_lines"
    etree.SubElement(meta, ns + "Created").text = "2026-10-28T00:00:00"
    etree.SubElement(meta, ns + "LastChange").text = "2026-10-28T00:00:00"
    page = etree.SubElement(pcgts, ns + "Page", imageFilename=str(image_path), imageWidth=str(width), imageHeight=str(height))
    region = etree.SubElement(page, ns + "TextRegion", id="r0", type="paragraph")
    etree.SubElement(region, ns + "Coords", points=f"0,0 {width - 1},0 {width - 1},{height - 1} 0,{height - 1}")
    for i, (line, text) in enumerate(lines):
        tl = etree.SubElement(region, ns + "TextLine", id=f"l{i}")
        etree.SubElement(tl, ns + "Coords", points=" ".join(f"{x},{y}" for x, y in line_polygon(line, width, height)))
        etree.SubElement(tl, ns + "Baseline", points=" ".join(f"{x},{y}" for x, y in line_baseline(line, page_offset, width, height)))
        te = etree.SubElement(tl, ns + "TextEquiv")
        etree.SubElement(te, ns + "Unicode").text = text
    return etree.tostring(pcgts, xml_declaration=True, encoding="UTF-8", pretty_print=True)


def process_page(book: str, xml_path: Path, image_path: Optional[Path], out_dir: Path, model: WordModel) -> bal.PageResult:
    """Group, clean and write one page.

    :param book: Source book (split stratum).
    :type book: str
    :param xml_path: VML-HD page XML.
    :type xml_path: Path
    :param image_path: Page image, or None.
    :type image_path: Optional[Path]
    :param out_dir: ``page_xml/vmlhd`` directory.
    :type out_dir: Path
    :param model: Word model for the spaces.
    :type model: WordModel
    :return: Page result (counts per line outcome, letters, written path).
    :rtype: bal.PageResult
    """
    key = f"{SET_NAME}/{book}/{xml_path.stem}"
    result = bal.PageResult(key=key, manuscript=book)
    if image_path is None:
        result.skip = "no image"
        return result
    with Image.open(image_path) as im:
        width, height = im.size
    paws = load_paws(xml_path)
    if not paws:
        result.skip = "no paws"
        return result
    if not frame_ok(paws, width, height):
        result.skip = "boxes outside the image"
        return result
    lines = group_lines(paws)
    page_offset = alif_offset(lines)
    page_offset = page_offset if page_offset is not None else lines[0].core * 0.3
    with Image.open(image_path) as im:
        missing = unannotated_ink(lines, im, paws)
    kept: List[Tuple[Line, str]] = []
    for line, (largest, total) in zip(lines, missing):
        reason = line_reason(line)
        if not reason and (largest >= MAX_MISSING_BLOB or total >= MAX_MISSING_TOTAL):
            reason = DROP_UNANNOTATED
        text = ""
        if not reason:
            text, reason = bal.clean_target(join_paws(line.paws, line.core, model))
        if reason:
            result.counts[reason] += 1
            continue
        result.counts[bal.KEPT] += 1
        result.letters += bal.arabic_letter_count(text)
        kept.append((line, text))
    if not kept:
        result.skip = "no line kept"
        return result
    out = out_dir / f"{SET_NAME}_{bal.slug(book)}_{bal.slug(xml_path.stem)}.xml"
    out.write_bytes(page_xml(kept, image_path.absolute(), width, height, page_offset))
    result.paths.append(out)
    return result


def build(root: Path, out_root: Path, limit: Optional[int] = None, workers: int = bal.WORKERS,
          lexicon_dirs: Optional[Sequence[Path]] = None) -> Dict[str, object]:
    """Build the VML-HD PAGE set, its lists and its ``stats.json`` entry.

    :param root: VML-HD folder (``xml/`` and ``pages/``).
    :type root: Path
    :param out_root: Output root.
    :type out_root: Path
    :param limit: Pages to process at most (after sorting).
    :type limit: Optional[int]
    :param workers: Threads.
    :type workers: int
    :param lexicon_dirs: PAGE XML folders for the lexicon; ``None`` = the ``$ARABIC_LINES_PAGE_XML`` sub-folders.
    :type lexicon_dirs: Optional[Sequence[Path]]
    :return: The set's summary as written to ``stats.json``.
    :rtype: Dict[str, object]
    """
    out_root = out_root.absolute()
    out_dir = out_root / "page_xml" / SET_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = build_lexicon(lexicon_dirs)
    (out_root / "lexicon.json").write_text(json.dumps(counts, ensure_ascii=False), encoding="utf-8")
    logger.info("lexicon: %d words, %d tokens", len(counts), sum(counts.values()))
    model = WordModel(counts)
    sources = page_sources(root)
    groups: Dict[str, List[str]] = {}
    for book, xml, _ in sources:
        groups.setdefault(book, []).append(f"{SET_NAME}/{book}/{xml.stem}")
    splits = bal.assign_splits(groups, bal.VAL_SHARE, bal.MIN_VAL_PAGES)
    todo = sources[:limit] if limit else sources
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda s: process_page(s[0], s[1], s[2], out_dir, model), todo))
    for key in bal.split_results(results, splits, bal.MIN_VAL_PAGES):
        logger.info("  moved %s to val (book had no written val page)", key)
    bal.write_lists(SET_NAME, results, out_root)
    spec = bal.SetSpec("page_xml", LICENCE, str(root))
    summary = bal.set_stats(results, spec, limit)
    bal.update_stats(out_root / "stats.json", {SET_NAME: summary})
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    """Command-line entry point.

    :param argv: Arguments (default ``sys.argv[1:]``).
    :type argv: Optional[List[str]]
    :return: None.
    :rtype: None
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=VMLHD_DIR, required=VMLHD_DIR is None,
                        help="VML-HD folder (default: $VMLHD_ROOT)")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=bal.WORKERS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(build(args.root, args.out_root, args.limit, args.workers), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
