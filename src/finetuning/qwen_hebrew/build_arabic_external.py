# File name: build_arabic_external.py
# Date: 10/5/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Arabic-script page-transcription training sets from four public handwriting datasets.

The fine-tuned model reads whole manuscript pages (one image in, one line of text per manuscript
line out) and cannot read Arabic script yet. This builder turns four public Arabic HTR datasets,
downloaded to ``arabic_external/`` on the NAS, into page rows of the same task the PGP Arabic
editions use (:data:`~src.finetuning.qwen_hebrew.build_pgp_arabic_editions.ARABIC_FRAGMENT_PROMPT`,
task ``fragment_transcribe``, KTIV ``FEATURES`` schema), one saved DatasetDict and one images-once
export per dataset:

* **agapet** (CC BY 4.0): real page images with PAGE XML -- SA-418 (342 pages), Sin423 (239
  images, each two book pages) and BnF Arabe 76 (11 TIFF pages). One row per image; the answer is
  the transcribed lines in the XML reading order. Sin423 images whose XML lists the left page
  before the right page are reordered right page first (:func:`right_page_first`); the other
  collections keep the XML order. Square-bracket restorations become the gap token ``[...]``;
  a page with an editorial note in brackets (``[فوق السطر: ...]``, ``[اقرأ: ...]``) is skipped.
  Val: in each manuscript the :data:`VAL_SHARE` of its source pages with the lowest SHA-1 of their
  row id (:func:`ranked_val_pages`), so every manuscript is in val.
* **muharaf** (CC BY-NC-SA 4.0): the full-page release (``public_data_files.zip``, read in place):
  one row per image from its PAGE XML (reading-order records, nested text regions of stamps and
  letterheads, lines sorted by their ``index``; square brackets kept as written). The official split of the line release (HF
  ``muharaf-public``) is page-level and is recovered by matching line texts
  (:func:`official_page_splits`): train to train, validation and test to val. A page whose XML
  export dropped transcribed lines of its annotation JSON is skipped.
* **baybars** / **iskandar** (Etalab 2.0): line crops only. One page-like image per source page
  (``manuscript_name`` + ``image_name``): the main-text crops stacked top to bottom
  (:func:`stack_lines`: right-aligned, a margin, the paper colour of the crops as background, the
  black fill outside each line polygon replaced by paper, crops at their own resolution, JPEG
  quality 95). Page numbers, catchwords, titles and marginalia are dropped (their rows sit per
  region, not where they are read). The dataset's split is kept: train to train, validation and
  test to val. BAYBARS rows are contiguous per page and in reading order; ISKANDAR rows are
  shuffled and split at line level, so its pages hold the lines of one source page and split in
  file order (the answer follows the stacked image, but the text does not run on).

Every line: Unicode NFC, tatweel and U+064B to U+0652 (short vowels, tanwin, shadda, sukun)
removed, invisible bidi controls removed, whitespace runs collapsed, empty lines dropped; letters,
hamza forms and dots are kept as transcribed. A page with fewer than :data:`MIN_LINES` lines or
fewer than ``MIN_PAGE_LETTERS`` Arabic letters is skipped, as is a page with an untranscribed line
at least half as wide as its median transcribed line (a text line missing from the answer). A real
page photograph that holds more than ``MAX_LETTERS_PER_MEGAPIXEL`` Arabic letters per megapixel (the
PGP builders' and the Arabic benchmark's limit) is skipped (:func:`too_dense`): the Sin423 images
are 1132 x 800 thumbnails of two book pages, about 2,500 letters per megapixel, and a target the
image cannot show teaches the model to invent text. Real page images above ``MAX_PAGE_PIXELS`` (the
KTIV page budget, the training ``min_pixels``) are downscaled; upright RGB JPEGs within it are
stored verbatim.

Output (per dataset ``<name>``)::

    <out-root>/arabic_external_<name>_v1/              saved DatasetDict (train, val) + manifest.jsonl + stats.json
    <out-root>/arabic_external_<name>_v1_images_once/  images-once export for the mixture builder

Usage (repo root)::

    .venv/bin/python -m src.finetuning.qwen_hebrew.build_arabic_external [--datasets agapet muharaf baybars iskandar]
        [--out-root DIR] [--limit N] [--no-export]
"""
import argparse
import hashlib
import io
import json
import logging
import os
import re
import shutil
import statistics
import unicodedata
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, DatasetInfo
from datasets.table import InMemoryTable
from lxml import etree
from PIL import Image as PILImage
from PIL import ImageOps
from scipy import ndimage

from src.datasets.evaluations import arabic_script as ar
from src.datasets.evaluations.helper_eval_scripts.build_arabic_benchmark import MAX_LETTERS_PER_MEGAPIXEL
from src.finetuning.qwen_hebrew.build_ktiv_dataset import FEATURES, MAX_PAGE_PIXELS, scaled_copy
from src.finetuning.qwen_hebrew.build_pgp_arabic_editions import ARABIC_FRAGMENT_PROMPT, GAP, MIN_PAGE_LETTERS, TASK, VAL_SHARE
from src.finetuning.qwen_hebrew.images_once import export_images_once

logger = logging.getLogger(__name__)

DATASETS_DIR = Path("/Volumes/home/studio_offload/datasets")
EXTERNAL_DIR = DATASETS_DIR / "arabic_external"
DATASET_NAMES = ("agapet", "muharaf", "baybars", "iskandar")
VERSION = "v1"
MIN_LINES = 3
JPEG_QUALITY = 95
WORKERS = 4
BATCH_ROWS = 32                # rows per Arrow record batch while a split is collected
WIDE_LINE_SHARE = 0.5          # an untranscribed line this wide (share of the median transcribed line) is a missed text line
MAIN_REGION_LINES = 5          # a region with this many transcribed lines is a page's text block
SIDE_OVERLAP = 0.1             # two text blocks overlapping by at most this share of the narrower one sit side by side
SIMILAR_WIDTH = 0.5            # ... and are two pages when the narrower is at least this share of the wider
AGAPET_COLLECTIONS: Dict[str, str] = {"sa418": "SA-418 (13th cent).zip", "sin423": "Sin423 (17th cent).zip",
                                      "bnf76": "BnF Arabe 76 pp 68-78.zip"}
DOUBLE_PAGE_COLLECTIONS = frozenset({"sin423"})
MUHARAF_ZIP = EXTERNAL_DIR / "muharaf_pages" / "public_data_files.zip"
MUHARAF_LINES = EXTERNAL_DIR / "muharaf_public" / "data"
MAIN_TEXT_REGIONS: Dict[str, frozenset] = {"baybars": frozenset({"MainText", "MainText_Right", "MainText_Left"}),
                                           "iskandar": frozenset({"MainText"})}
LINE_SPLITS: Dict[str, Tuple[str, ...]] = {"train": ("train",), "val": ("validation", "test")}
LICENCES: Dict[str, str] = {"agapet": "CC BY 4.0", "muharaf": "CC BY-NC-SA 4.0",
                            "baybars": "Etalab Open Licence 2.0", "iskandar": "Etalab Open Licence 2.0"}
SECTIONS: Dict[str, str] = {"agapet": "arabic_page", "muharaf": "arabic_page", "baybars": "arabic_stacked_lines",
                            "iskandar": "arabic_stacked_lines_unordered"}
STACK_MARGIN = 0.6             # page margin, in median line heights
STACK_MIN_MARGIN = 24          # pixels
STACK_GAP = 0.1                # space between stacked lines, in median line heights
STACK_MIN_GAP = 4              # pixels: keeps a feathered crop edge off the line above
EDGE_SOFTNESS = 1.0            # Gaussian sigma (pixels) of a pasted crop's edge
PAPER_QUANTILE = 0.4           # pixels darker than this luminance quantile are ink, not paper

_STRIP = re.compile(r"[\u0640\u064B-\u0652]")                                # tatweel, short vowels, tanwin, shadda, sukun
_BIDI = re.compile(r"[\u061C\u200E\u200F\u202A-\u202E\u2066-\u2069\uFEFF]")  # invisible direction controls
_SPACES = re.compile(r"\s+")
_BRACKET = re.compile(r"\[([^\[\]]*)\]")
_NOTE = re.compile(r":|^\s*(فوق|خارج|اقرأ|إقرأ|مكرر)|[0-9\u0660-\u0669]")    # editor's words, not the scribe's
_RO_INDEX = re.compile(r"readingOrder\s*\{\s*index:\s*(\d+)")
_EXIF_ORIENTATION = 0x0112


# ----------------------------------------------------------------------------- text

def clean_line(text: str) -> str:
    """Normalise one transcribed line.

    :param text: Raw line text.
    :type text: str
    :return: NFC text without tatweel, short vowels, tanwin, shadda, sukun and bidi controls, whitespace
        runs collapsed to one space, stripped; letters, hamza forms and dots unchanged.
    :rtype: str
    """
    text = _BIDI.sub("", _STRIP.sub("", unicodedata.normalize("NFC", text)))
    return _SPACES.sub(" ", unicodedata.normalize("NFC", text)).strip()


def resolve_brackets(line: str, gap: str = GAP) -> Tuple[str, int, int]:
    """Turn editorial square brackets into what is on the page.

    A restoration (``[على الكنـ]يسه``) or an empty bracket marks text the editor could not see on the
    page: it becomes ``gap`` set off by spaces, neighbouring gaps merged (as
    ``arabic_script.clean_edition_line`` writes the PGP targets). A bracket holding the editor's own
    words (a colon, ``فوق`` / ``خارج`` / ``اقرأ`` / ``مكرر``, a folio number) is a note; it is
    counted and left in place, and so is an unpaired bracket.

    :param line: One line of text.
    :type line: str
    :param gap: Token for lost text.
    :type gap: str
    :return: ``(line with restorations replaced, restorations, notes)``.
    :rtype: Tuple[str, int, int]
    """
    counts: Counter = Counter()

    def replace(match: re.Match) -> str:
        kind = "note" if _NOTE.search(match.group(1)) else "restoration"
        counts[kind] += 1
        return match.group(0) if kind == "note" else f" {gap} "

    out = _BRACKET.sub(replace, line)
    if counts["restoration"]:
        out = re.sub(r"(?:" + re.escape(gap) + r"\s*)+", gap + " ", out)
    rest = out.replace(gap, "")
    unpaired = rest.count("[") + rest.count("]") - 2 * counts["note"]
    return out, counts["restoration"], counts["note"] + max(0, unpaired)


def clean_page_lines(raw_lines: Sequence[str], gap: str = GAP, editorial: bool = True) -> Tuple[List[str], Counter]:
    """Clean every line of a page, keeping positions.

    :param raw_lines: Raw line texts in reading order.
    :type raw_lines: Sequence[str]
    :param gap: Token for lost text.
    :type gap: str
    :param editorial: Square brackets are the editor's (:func:`resolve_brackets`); False keeps them as
        written (the line releases transcribe brackets the scribe drew, e.g. BAYBARS ``[صلعم]``).
    :type editorial: bool
    :return: ``(cleaned line per input line, "" for a line that is dropped; counts of restorations,
        notes and dropped empty lines)``.
    :rtype: Tuple[List[str], Counter]
    """
    counts: Counter = Counter()
    out = []
    for raw in raw_lines:
        line = unicodedata.normalize("NFC", raw)
        line, restorations, notes = resolve_brackets(line, gap) if editorial else (line, 0, 0)
        counts["restorations"] += restorations
        counts["editorial notes"] += notes
        line = clean_line(line)
        counts["empty lines dropped"] += not line
        out.append(line)
    return out, counts


def arabic_letter_count(text: str) -> int:
    """Number of Arabic letters, counted as the PGP builder and the Arabic benchmark count them.

    :param text: Any text.
    :type text: str
    :return: Letters U+0621 to U+064A after Unicode folding.
    :rtype: int
    """
    return len(ar.arabic_letters(text))


def page_gate(lines: Sequence[str], min_lines: int = MIN_LINES, min_letters: int = MIN_PAGE_LETTERS) -> str:
    """Why a page is too small to train on.

    :param lines: The page's cleaned, non-empty lines.
    :type lines: Sequence[str]
    :param min_lines: Fewest lines a page may have.
    :type min_lines: int
    :param min_letters: Fewest Arabic letters a page may have.
    :type min_letters: int
    :return: The reason, or ``""`` when the page passes.
    :rtype: str
    """
    if len(lines) < min_lines:
        return f"fewer than {min_lines} lines"
    if arabic_letter_count("\n".join(lines)) < min_letters:
        return f"fewer than {min_letters} Arabic letters"
    return ""


def too_dense(lines: Sequence[str], width: int, height: int, limit: float = MAX_LETTERS_PER_MEGAPIXEL) -> bool:
    """Whether a page photograph holds more text than its pixels can show.

    At 1,000 letters per megapixel a letter has about one 32-pixel patch of the vision tower, margins included.

    :param lines: The page's cleaned lines.
    :type lines: Sequence[str]
    :param width: Image width in pixels, as stored in the row.
    :type width: int
    :param height: Image height in pixels, as stored in the row.
    :type height: int
    :param limit: Most Arabic letters per megapixel a page may hold.
    :type limit: float
    :return: ``True`` when the page is over the limit.
    :rtype: bool
    """
    return arabic_letter_count("\n".join(lines)) / (width * height / 1e6) > limit


# ----------------------------------------------------------------------------- PAGE XML

@dataclass
class PageLine:
    """One ``TextLine`` of a PAGE file.

    :param text: Raw ``TextEquiv/Unicode`` text (``""`` when the line has none).
    :type text: str
    :param bbox: ``(x0, y0, x1, y1)`` of the line polygon.
    :type bbox: Tuple[int, int, int, int]
    """
    text: str
    bbox: Tuple[int, int, int, int]


@dataclass
class PageRegion:
    """A region's lines, in their reading order.

    :param region_id: Region id.
    :type region_id: str
    :param bbox: ``(x0, y0, x1, y1)`` of the region polygon (of its lines when it has none).
    :type bbox: Tuple[int, int, int, int]
    :param lines: Lines of the region.
    :type lines: List[PageLine]
    """
    region_id: str
    bbox: Tuple[int, int, int, int]
    lines: List[PageLine]


@dataclass
class PageXml:
    """What the builders use of a PAGE file.

    :param image_filename: ``Page/@imageFilename``.
    :type image_filename: str
    :param width: ``Page/@imageWidth``.
    :type width: int
    :param height: ``Page/@imageHeight``.
    :type height: int
    :param regions: Regions with lines, in reading order.
    :type regions: List[PageRegion]
    :param unordered_regions: Regions with lines that the reading order does not reach (appended in
        document order).
    :type unordered_regions: int
    """
    image_filename: str
    width: int
    height: int
    regions: List[PageRegion]
    unordered_regions: int

    def lines(self) -> List[PageLine]:
        """All lines in reading order.

        :return: Lines of every region, region by region.
        :rtype: List[PageLine]
        """
        return [line for region in self.regions for line in region.lines]


def _local(element: Any) -> str:
    """Local tag name of an element (``""`` for comments and processing instructions).

    :param element: An lxml node.
    :type element: Any
    :return: Tag name without namespace.
    :rtype: str
    """
    return etree.QName(element).localname if isinstance(element.tag, str) else ""


def _kids(element: Any, name: str) -> List[Any]:
    """Child elements with a given local name.

    :param element: Parent element.
    :type element: Any
    :param name: Local tag name.
    :type name: str
    :return: Matching children in document order.
    :rtype: List[Any]
    """
    return [child for child in element if _local(child) == name]


def _is_region(element: Any) -> bool:
    """Whether an element is a PAGE region (``TextRegion``, ``GraphicRegion``, ...).

    :param element: Any element.
    :type element: Any
    :return: True for an element named ``*Region`` with an id.
    :rtype: bool
    """
    return _local(element).endswith("Region") and element.get("id") is not None


def _bbox(element: Any) -> Tuple[int, int, int, int]:
    """Bounding box of an element's ``Coords/@points``.

    :param element: Region or line element.
    :type element: Any
    :return: ``(x0, y0, x1, y1)``; all zero when the element has no points.
    :rtype: Tuple[int, int, int, int]
    """
    coords = _kids(element, "Coords")
    points = [p.split(",") for p in (coords[0].get("points", "") if coords else "").split() if "," in p]
    if not points:
        return 0, 0, 0, 0
    xs = [int(float(x)) for x, _ in points]
    ys = [int(float(y)) for _, y in points]
    return min(xs), min(ys), max(xs), max(ys)


def _line_text(line: Any) -> str:
    """Text of a ``TextLine`` (its own ``TextEquiv``, not its words').

    :param line: ``TextLine`` element.
    :type line: Any
    :return: The first ``TextEquiv/Unicode`` text, ``""`` when there is none.
    :rtype: str
    """
    for equiv in _kids(line, "TextEquiv"):
        for unicode in _kids(equiv, "Unicode"):
            return unicode.text or ""
    return ""


def ordered_lines(region: Any) -> List[Any]:
    """A region's ``TextLine`` elements in reading order.

    :param region: Region element.
    :type region: Any
    :return: Sorted by ``@index`` (PAGE 2019, Muharaf) when every line has one, else by the
        Transkribus ``custom="readingOrder {index:N;}"`` when every line has one, else document order.
    :rtype: List[Any]
    """
    lines = _kids(region, "TextLine")
    if lines and all(line.get("index") is not None for line in lines):
        keys = [int(line.get("index")) for line in lines]
    else:
        found = [_RO_INDEX.search(line.get("custom") or "") for line in lines]
        if not lines or not all(found):
            return lines
        keys = [int(match.group(1)) for match in found]
    return [lines[i] for i in sorted(range(len(lines)), key=lambda i: keys[i])]


def reading_order_ids(page: Any) -> List[str]:
    """Region ids in the order of the page's ``ReadingOrder`` (nested groups flattened).

    :param page: ``Page`` element.
    :type page: Any
    :return: Ids, first occurrence only; empty when the page has no reading order.
    :rtype: List[str]
    """
    ids: List[str] = []

    def walk(group: Any) -> None:
        children = [child for child in group if _local(child)]
        if _local(group).startswith("OrderedGroup"):
            children = sorted(children, key=lambda child: int(child.get("index", "0")))
        for child in children:
            name = _local(child)
            if name.startswith("RegionRef"):
                ids.append(child.get("regionRef"))
            elif "Group" in name:
                if child.get("regionRef"):
                    ids.append(child.get("regionRef"))
                walk(child)

    for order in _kids(page, "ReadingOrder"):
        walk(order)
    return list(dict.fromkeys(ids))


def parse_page_xml(data: bytes) -> PageXml:
    """Read the lines of a PAGE XML file (2013 or 2019 schema) in reading order.

    Regions follow the ``ReadingOrder``; a region's lines follow :func:`ordered_lines`; regions
    nested in a region (Muharaf keeps the text of stamps and letterheads in a ``TextRegion`` inside
    a ``GraphicRegion``) follow their parent unless the reading order names them. Regions with
    lines that the reading order does not reach come last, in document order; without a reading
    order every region is in document order.

    :param data: XML bytes.
    :type data: bytes
    :return: The parsed page.
    :rtype: PageXml
    """
    page = _kids(etree.fromstring(data), "Page")[0]
    elements = {el.get("id"): el for el in page.iter() if _is_region(el)}
    order = [rid for rid in reading_order_ids(page) if rid in elements]
    named = set(order)
    regions: List[PageRegion] = []
    seen: Set[str] = set()

    def emit(element: Any) -> None:
        if element.get("id") in seen:
            return
        seen.add(element.get("id"))
        lines = [PageLine(_line_text(line), _bbox(line)) for line in ordered_lines(element)]
        if lines:
            box = _bbox(element)
            if box == (0, 0, 0, 0):
                box = (min(l.bbox[0] for l in lines), min(l.bbox[1] for l in lines),
                       max(l.bbox[2] for l in lines), max(l.bbox[3] for l in lines))
            regions.append(PageRegion(element.get("id"), box, lines))
        for child in element:
            if _is_region(child) and child.get("id") not in named:
                emit(child)

    for rid in order:
        emit(elements[rid])
    unordered = 0
    for rid, element in elements.items():
        if rid not in seen and _kids(element, "TextLine"):
            unordered += 1
            emit(element)
    return PageXml(page.get("imageFilename", ""), int(page.get("imageWidth", 0)), int(page.get("imageHeight", 0)),
                   regions, unordered if order else 0)


def _center_x(region: PageRegion) -> float:
    """Horizontal centre of a region.

    :param region: A region.
    :type region: PageRegion
    :return: ``(x0 + x1) / 2``.
    :rtype: float
    """
    return (region.bbox[0] + region.bbox[2]) / 2


def right_page_first(regions: Sequence[PageRegion], min_lines: int = MAIN_REGION_LINES,
                     max_overlap: float = SIDE_OVERLAP, min_width_ratio: float = SIMILAR_WIDTH) -> Tuple[List[PageRegion], bool]:
    """Put the right-hand page of a two-page image first, as an Arabic book is read.

    Applies only when the image has exactly two text blocks (regions with ``min_lines`` transcribed
    lines) of similar width side by side (two pages, not a page and a margin column) and the
    reading order lists the left one first; then every region right of the midpoint between the
    blocks comes first, each side in its listed order.

    :param regions: Regions in the XML reading order.
    :type regions: Sequence[PageRegion]
    :param min_lines: Transcribed lines that make a region a text block.
    :type min_lines: int
    :param max_overlap: Largest horizontal overlap of the blocks, as a share of the narrower one.
    :type max_overlap: float
    :param min_width_ratio: Smallest width of the narrower block, as a share of the wider one.
    :type min_width_ratio: float
    :return: ``(regions in reading order, whether they were reordered)``.
    :rtype: Tuple[List[PageRegion], bool]
    """
    blocks = [r for r in regions if sum(1 for line in r.lines if clean_line(line.text)) >= min_lines]
    if len(blocks) != 2:
        return list(regions), False
    first, second = blocks
    widths = sorted((first.bbox[2] - first.bbox[0], second.bbox[2] - second.bbox[0]))
    two_pages = first.bbox[2] - second.bbox[0] <= max_overlap * widths[0] and widths[0] >= min_width_ratio * widths[1]
    if _center_x(first) >= _center_x(second) or not two_pages:
        return list(regions), False
    middle = (_center_x(first) + _center_x(second)) / 2
    return [r for r in regions if _center_x(r) >= middle] + [r for r in regions if _center_x(r) < middle], True


def missed_text_line(lines: Sequence[PageLine], share: float = WIDE_LINE_SHARE) -> bool:
    """Whether a page has an untranscribed line as wide as a text line.

    Small untranscribed lines are marks, monograms, catchwords, page numbers or a marginal word; one
    at least ``share`` of the median transcribed line width is a line of text the answer would miss.

    :param lines: The page's lines.
    :type lines: Sequence[PageLine]
    :param share: Width threshold as a share of the median transcribed-line width.
    :type share: float
    :return: True when such a line exists.
    :rtype: bool
    """
    widths = [line.bbox[2] - line.bbox[0] for line in lines if clean_line(line.text)]
    if not widths:
        return False
    floor = share * statistics.median(widths)
    return any(not clean_line(line.text) and line.bbox[2] - line.bbox[0] >= floor for line in lines)


def xml_page_lines(page: PageXml, double_page: bool = False, editorial: bool = True) -> Tuple[List[str], Counter, str]:
    """The answer lines of a real page and why the page cannot be used.

    :param page: Parsed PAGE file.
    :type page: PageXml
    :param double_page: The image shows two book pages (:func:`right_page_first` applies).
    :type double_page: bool
    :param editorial: Square brackets are an editor's (Agapet); False keeps them as written
        (Muharaf transcribes brackets the writer drew, and its ``[...]`` already is the gap token).
    :type editorial: bool
    :return: ``(cleaned lines in reading order, counters, skip reason or "")``.
    :rtype: Tuple[List[str], Counter, str]
    """
    regions, moved = right_page_first(page.regions) if double_page else (page.regions, False)
    counts: Counter = Counter({"left page listed first: reordered": int(moved),
                               "regions outside the reading order": page.unordered_regions})
    lines = [line for region in regions for line in region.lines]
    if missed_text_line(lines):
        return [], counts, "an untranscribed line as wide as a text line"
    cleaned, line_counts = clean_page_lines([line.text for line in lines], editorial=editorial)
    counts.update(line_counts)
    if line_counts["editorial notes"]:
        return [], counts, "editorial note in brackets"
    kept = [line for line in cleaned if line]
    return kept, counts, page_gate(kept)


# ----------------------------------------------------------------------------- images

def encode_jpeg(image: PILImage.Image, quality: int = JPEG_QUALITY) -> bytes:
    """Encode an image as an RGB JPEG.

    :param image: Any PIL image.
    :type image: PILImage.Image
    :param quality: JPEG quality.
    :type quality: int
    :return: JPEG bytes.
    :rtype: bytes
    """
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def page_image(data: bytes, max_pixels: int = MAX_PAGE_PIXELS, quality: int = JPEG_QUALITY) -> Tuple[bytes, int, int, str]:
    """A page image as the training rows store it.

    An upright RGB JPEG within the pixel budget is kept byte for byte. Anything else is turned
    upright (EXIF orientation, as ``datasets.Image`` shows it), converted to RGB, downscaled to the
    budget when above it, and encoded as a JPEG.

    :param data: Encoded image (JPEG, TIFF, PNG, ...).
    :type data: bytes
    :param max_pixels: Pixel budget.
    :type max_pixels: int
    :param quality: JPEG quality of a re-encode.
    :type quality: int
    :return: ``(JPEG bytes, width, height, "verbatim" | "re-encoded" | "downscaled")``.
    :rtype: Tuple[bytes, int, int, str]
    """
    with PILImage.open(io.BytesIO(data)) as im:
        orientation = im.getexif().get(_EXIF_ORIENTATION, 1)
        if im.format == "JPEG" and im.mode == "RGB" and orientation == 1 and im.width * im.height <= max_pixels:
            return data, im.width, im.height, "verbatim"
        upright = ImageOps.exif_transpose(im).convert("RGB")
    scaled, scale = scaled_copy(upright, max_pixels)
    return encode_jpeg(scaled, quality), scaled.width, scaled.height, "downscaled" if scale < 1 else "re-encoded"


def page_image_file(path: Path) -> Tuple[bytes, int, int, str]:
    """:func:`page_image` of an image file.

    :param path: Image file.
    :type path: Path
    :return: ``(JPEG bytes, width, height, how it was stored)``.
    :rtype: Tuple[bytes, int, int, str]
    """
    return page_image(path.read_bytes())


def fill_mask(rgb: np.ndarray) -> np.ndarray:
    """Pixels outside a line polygon: the exact black fill connected to the crop border.

    :param rgb: ``(h, w, 3)`` uint8 crop.
    :type rgb: np.ndarray
    :return: ``(h, w)`` bool mask (all False for a crop without fill).
    :rtype: np.ndarray
    """
    black = ~rgb.any(axis=2)
    if not black.any():
        return black
    labels, _ = ndimage.label(black)
    edge = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    return np.isin(labels, np.unique(edge[edge > 0]))


def paper_color(rgb: np.ndarray, mask: np.ndarray, ink_quantile: float = PAPER_QUANTILE) -> np.ndarray:
    """Colour of the paper in a crop.

    :param rgb: ``(h, w, 3)`` uint8 crop.
    :type rgb: np.ndarray
    :param mask: Fill pixels to ignore (:func:`fill_mask`).
    :type mask: np.ndarray
    :param ink_quantile: Pixels darker than this luminance quantile count as ink.
    :type ink_quantile: float
    :return: Per-channel median of the lighter pixels, float ``(3,)``; white for a crop that is all fill.
    :rtype: np.ndarray
    """
    pixels = rgb[~mask].reshape(-1, 3)
    if not len(pixels):
        return np.full(3, 255.0)
    channels = pixels.astype(np.float32)
    luminance = 0.299 * channels[:, 0] + 0.587 * channels[:, 1] + 0.114 * channels[:, 2]   # element-wise: no BLAS call
    return np.median(pixels[luminance >= np.quantile(luminance, ink_quantile)], axis=0)


def band_profile(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """First and last row of the line polygon in every column of a crop.

    :param mask: Fill mask of the crop (:func:`fill_mask`).
    :type mask: np.ndarray
    :return: ``(top, bottom)`` float arrays of the crop width; NaN in a column without polygon pixels.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    inside = ~mask
    present = inside.any(axis=0)
    top = np.where(present, inside.argmax(axis=0), np.nan)
    bottom = np.where(present, inside.shape[0] - 1 - inside[::-1].argmax(axis=0), np.nan)
    return top, bottom


def stack_lines(crops: Sequence[np.ndarray], margin: Optional[int] = None, gap: Optional[int] = None,
                softness: float = EDGE_SOFTNESS) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """Stack line crops into a plain page: top to bottom in the given order, right-aligned.

    Each crop goes as high as it can while its line polygon stays ``gap`` rows below the polygons
    already placed, column by column, so slanted lines nest as on the page instead of leaving the
    empty corners of their bounding boxes (rectangular crops simply stack). The background is the
    median paper colour of the crops; each crop keeps its pixels and resolution (whole-pixel
    offsets only), its fill outside the polygon becomes its own paper colour, and its polygon edge
    is feathered into the background.

    :param crops: ``(h, w, 3)`` uint8 line crops in reading order.
    :type crops: Sequence[np.ndarray]
    :param margin: Page margin in pixels (default :data:`STACK_MARGIN` median line heights, at least
        :data:`STACK_MIN_MARGIN`).
    :type margin: Optional[int]
    :param gap: Empty rows between line polygons (default :data:`STACK_GAP` median line heights, at
        least :data:`STACK_MIN_GAP`).
    :type gap: Optional[int]
    :param softness: Gaussian sigma of the feathered crop edge (0 pastes hard edges).
    :type softness: float
    :return: ``(page, (x0, y0, x1, y1) of each crop on the page)``.
    :rtype: Tuple[np.ndarray, List[Tuple[int, int, int, int]]]
    """
    masks = [fill_mask(crop) for crop in crops]
    papers = [paper_color(crop, mask) for crop, mask in zip(crops, masks)]
    profiles = [band_profile(mask) for mask in masks]
    background = np.median(np.stack(papers), axis=0).round().astype(np.uint8)
    line_height = float(np.median([np.nanmedian(bottom - top + 1) for top, bottom in profiles]))
    margin = max(STACK_MIN_MARGIN, round(STACK_MARGIN * line_height)) if margin is None else margin
    gap = max(STACK_MIN_GAP, round(STACK_GAP * line_height)) if gap is None else gap
    width = max(crop.shape[1] for crop in crops) + 2 * margin
    skyline = np.full(width, -np.inf)                   # lowest polygon row placed so far, per page column
    boxes: List[Tuple[int, int, int, int]] = []
    y = margin
    for crop, (top, bottom) in zip(crops, profiles):
        h, w = crop.shape[:2]
        x = width - margin - w
        below = skyline[x:x + w]
        shared = ~np.isnan(top) & np.isfinite(below)
        if shared.any():
            y = max(y, int(np.ceil(np.max(below[shared] + 1 + gap - top[shared]))))
        filled = ~np.isnan(bottom)
        below[filled] = np.maximum(below[filled], y + bottom[filled])
        boxes.append((x, y, x + w, y + h))
    page = np.empty((max(box[3] for box in boxes) + margin, width, 3), np.uint8)
    page[:] = background
    for crop, mask, paper, (x0, y0, x1, y1) in zip(crops, masks, papers, boxes):
        filled_crop = crop.astype(np.float32)
        filled_crop[mask] = paper
        alpha = (~mask).astype(np.float32)
        if softness:
            alpha = ndimage.gaussian_filter(alpha, softness, mode="constant", cval=0.0)
        target = page[y0:y1, x0:x1].astype(np.float32)
        blended = filled_crop * alpha[..., None] + target * (1 - alpha[..., None])
        page[y0:y1, x0:x1] = np.clip(blended, 0, 255).round().astype(np.uint8)
    return page, boxes


def stack_page(crop_bytes: Sequence[bytes], quality: int = JPEG_QUALITY) -> Tuple[bytes, int, int]:
    """Decode line crops, stack them (:func:`stack_lines`) and encode the page.

    :param crop_bytes: Encoded crops in reading order.
    :type crop_bytes: Sequence[bytes]
    :param quality: JPEG quality.
    :type quality: int
    :return: ``(JPEG bytes, width, height)``.
    :rtype: Tuple[bytes, int, int]
    """
    crops = []
    for data in crop_bytes:
        with PILImage.open(io.BytesIO(data)) as im:
            crops.append(np.asarray(im.convert("RGB")))
    page, _ = stack_lines(crops)
    return encode_jpeg(PILImage.fromarray(page), quality), page.shape[1], page.shape[0]


# ----------------------------------------------------------------------------- rows

def page_row(image: bytes, text: str, stem: str, width: int, height: int, label_source: str, section: str) -> Dict[str, Any]:
    """One page-transcription row in the KTIV ``FEATURES`` schema, with the PGP Arabic page task.

    :param image: Encoded JPEG of the page.
    :type image: bytes
    :param text: Target: the page's lines joined by newlines.
    :type text: str
    :param stem: Row id.
    :type stem: str
    :param width: Image width in pixels.
    :type width: int
    :param height: Image height in pixels.
    :type height: int
    :param label_source: Dataset name.
    :type label_source: str
    :param section: Kind of page image.
    :type section: str
    :return: Feature dict.
    :rtype: Dict[str, Any]
    """
    return {"image": image, "question": ARABIC_FRAGMENT_PROMPT, "answer": text, "task": TASK, "section": section,
            "stem": stem, "label_source": label_source, "target_chars": len(text), "target_tokens": 0,
            "image_width": width, "image_height": height}


def slug(text: str) -> str:
    """Lower-case id fragment of a file or page name.

    :param text: Any name.
    :type text: str
    :return: Runs of anything but ASCII letters and digits as ``_``.
    :rtype: str
    """
    return re.sub(r"[^0-9a-z]+", "_", text.lower()).strip("_")


def row_stems(prefix: str, names: Sequence[str]) -> Dict[str, str]:
    """Unique row ids for source names: ``<prefix>_<slug>``, and a short hash where slugs collide.

    Muharaf has ``BEK1A_27_01r`` and ``BEK1A_27_01r-``, two different pages with one slug. Computed
    over every source name, so a page's id does not depend on which pages are later skipped.

    :param prefix: Dataset prefix.
    :type prefix: str
    :param names: Source names (file stems).
    :type names: Sequence[str]
    :return: ``{name: row id}``.
    :rtype: Dict[str, str]
    """
    slugs = Counter(slug(name) for name in names)
    return {name: f"{prefix}_{slug(name)}" if slugs[slug(name)] == 1
            else f"{prefix}_{slug(name)}_{hashlib.sha1(name.encode('utf-8')).hexdigest()[:6]}" for name in names}


def hash_split(key: str, val_share: float = VAL_SHARE) -> str:
    """Deterministic split of a page.

    :param key: Stable page id.
    :type key: str
    :param val_share: Share of pages that go to validation.
    :type val_share: float
    :return: ``"val"`` or ``"train"``.
    :rtype: str
    """
    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % 10_000
    return "val" if bucket < round(val_share * 10_000) else "train"


def ranked_val_pages(keys_by_group: Dict[str, Sequence[str]], val_share: float = VAL_SHARE) -> Set[str]:
    """An exact, stratified validation draw: in every group the pages with the lowest SHA-1 of their key.

    :param keys_by_group: ``{group (e.g. collection): stable page keys}``; rank over every source page,
        before any page is skipped, so a page's split depends only on the source files.
    :type keys_by_group: Dict[str, Sequence[str]]
    :param val_share: Share of each group's pages held out (at least one page of a non-empty group
        when the share is positive).
    :type val_share: float
    :return: Keys of the validation pages.
    :rtype: Set[str]
    """
    chosen: Set[str] = set()
    for keys in keys_by_group.values():
        n = max(1, round(val_share * len(keys))) if keys and val_share > 0 else 0
        chosen.update(sorted(keys, key=lambda k: hashlib.sha1(k.encode("utf-8")).hexdigest())[:n])
    return chosen


def map_in_chunks(fn: Callable[[Any], Any], items: Iterable[Any], workers: int = WORKERS, chunk: int = 16) -> Iterator[Any]:
    """Map a function over items on a thread pool, a chunk at a time (bounded memory, input order).

    :param fn: Function of one item.
    :type fn: Callable[[Any], Any]
    :param items: Items; drawn lazily, ``chunk`` at a time.
    :type items: Iterable[Any]
    :param workers: Threads.
    :type workers: int
    :param chunk: Items in flight.
    :type chunk: int
    :return: Results in input order.
    :rtype: Iterator[Any]
    """
    with ThreadPoolExecutor(max_workers=workers) as pool:
        batch: List[Any] = []
        for item in items:
            batch.append(item)
            if len(batch) == chunk:
                yield from pool.map(fn, batch)
                batch = []
        if batch:
            yield from pool.map(fn, batch)


def split_summary(manifest: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per-split counts of a built dataset.

    :param manifest: One entry per row with ``split``, ``lines``, ``arabic_letters``, ``image_width``,
        ``image_height``.
    :type manifest: Sequence[Dict[str, Any]]
    :return: ``{split: {pages, lines, arabic_letters, median_lines_per_page, image_width [min, max],
        image_height [min, max]}}``.
    :rtype: Dict[str, Dict[str, Any]]
    """
    out: Dict[str, Dict[str, Any]] = {}
    for split in dict.fromkeys(entry["split"] for entry in manifest):
        rows = [entry for entry in manifest if entry["split"] == split]
        out[split] = {"pages": len(rows), "lines": sum(r["lines"] for r in rows),
                      "arabic_letters": sum(r["arabic_letters"] for r in rows),
                      "median_lines_per_page": statistics.median(r["lines"] for r in rows),
                      "image_width": [min(r["image_width"] for r in rows), max(r["image_width"] for r in rows)],
                      "image_height": [min(r["image_height"] for r in rows), max(r["image_height"] for r in rows)]}
    return out


def record_batch(rows: Sequence[Dict[str, Any]]) -> pa.RecordBatch:
    """Rows as one Arrow record batch with the storage schema of the KTIV ``FEATURES``.

    :param rows: :func:`page_row` dicts (``image`` holds the encoded bytes).
    :type rows: Sequence[Dict[str, Any]]
    :return: The batch; the image column is the ``{"bytes", "path"}`` struct ``datasets.Image`` stores.
    :rtype: pa.RecordBatch
    """
    schema = FEATURES.arrow_schema
    arrays = []
    for name in FEATURES:
        values = [row[name] for row in rows]
        if name == "image":
            values = [{"bytes": value, "path": None} for value in values]
        arrays.append(pa.array(values, type=schema.field(name).type))
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


class RowBatches:
    """The rows of one split, kept as Arrow record batches as they arrive.

    A row's image bytes live in Python only until its batch is full; after that they are held once,
    in Arrow buffers, and no binary array comes near Arrow's 2 GB limit.

    :param batch_rows: Rows per record batch.
    :type batch_rows: int
    """

    def __init__(self, batch_rows: int = BATCH_ROWS) -> None:
        self.batch_rows = batch_rows
        self.batches: List[pa.RecordBatch] = []
        self.pending: List[Dict[str, Any]] = []
        self.count = 0

    def append(self, row: Dict[str, Any]) -> None:
        """Add one row; a full batch is converted at once.

        :param row: A :func:`page_row` dict.
        :type row: Dict[str, Any]
        """
        self.pending.append(row)
        self.count += 1
        if len(self.pending) >= self.batch_rows:
            self.flush()

    def flush(self) -> None:
        """Convert the pending rows to a record batch."""
        if self.pending:
            self.batches.append(record_batch(self.pending))
            self.pending = []

    def __len__(self) -> int:
        """Number of rows.

        :return: Rows appended so far.
        :rtype: int
        """
        return self.count

    def table(self) -> pa.Table:
        """All rows as one table (no copy of the batches).

        :return: Table with the ``FEATURES`` storage schema.
        :rtype: pa.Table
        """
        self.flush()
        return pa.Table.from_batches(self.batches, schema=FEATURES.arrow_schema)


def save_rows(splits: Dict[str, RowBatches], out: Path, sidecars: Dict[str, str]) -> None:
    """Write the DatasetDict plus sidecar files, replacing the previous build atomically.

    Same layout and swap as ``build_pgp_editions.save_dataset`` (``<out>.building``, then rename;
    empty splits are not written), but each split goes from its Arrow batches straight into a
    ``Dataset`` with a fingerprint taken from its row stems: ``Dataset.from_list`` and an unnamed
    ``Dataset`` both fingerprint by hashing the whole table, which held about eight times the image
    bytes in memory (8.7 GB for 744 MB of Agapet images).

    :param splits: ``{split name: rows}``.
    :type splits: Dict[str, RowBatches]
    :param out: Destination directory.
    :type out: Path
    :param sidecars: ``{file name: text}`` written next to the splits.
    :type sidecars: Dict[str, str]
    """
    tmp = out.with_name(out.name + ".building")
    if tmp.exists():
        shutil.rmtree(tmp)
    parts = {}
    for name, rows in splits.items():
        if not len(rows):
            continue
        table = rows.table()
        fingerprint = hashlib.sha1("\n".join([name, *table.column("stem").to_pylist()]).encode("utf-8")).hexdigest()
        parts[name] = Dataset(InMemoryTable(table), info=DatasetInfo(features=FEATURES), fingerprint=fingerprint)
    DatasetDict(parts).save_to_disk(str(tmp))
    for name, text in sidecars.items():
        (tmp / name).write_text(text, encoding="utf-8")
    previous = out.with_name(out.name + ".previous")
    if out.exists():
        if previous.exists():
            shutil.rmtree(previous)
        out.rename(previous)
    tmp.rename(out)
    if previous.exists():
        shutil.rmtree(previous)


def write_dataset(name: str, out: Path, rows: Dict[str, RowBatches], manifest: List[Dict[str, Any]],
                  stats: Dict[str, Any]) -> Dict[str, Any]:
    """Save the DatasetDict with its manifest and statistics.

    :param name: Dataset name.
    :type name: str
    :param out: Destination directory (replaced atomically).
    :type out: Path
    :param rows: ``{"train": rows, "val": rows}``.
    :type rows: Dict[str, RowBatches]
    :param manifest: One entry per row.
    :type manifest: List[Dict[str, Any]]
    :param stats: Dataset-specific statistics; splits, licence, task and prompt are added.
    :type stats: Dict[str, Any]
    :return: The statistics written to ``stats.json``.
    :rtype: Dict[str, Any]
    """
    stats = {"dataset": name, "licence": LICENCES[name], "task": TASK, "section": SECTIONS[name],
             "splits": split_summary(manifest), **stats, "prompt": ARABIC_FRAGMENT_PROMPT}
    save_rows(rows, out, {"manifest.jsonl": "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest),
                          "stats.json": json.dumps(stats, ensure_ascii=False, indent=1)})
    return stats


def manifest_entry(stem: str, split: str, lines: Sequence[str], width: int, height: int, **extra: Any) -> Dict[str, Any]:
    """Manifest line of one row.

    :param stem: Row id.
    :type stem: str
    :param split: ``train`` or ``val``.
    :type split: str
    :param lines: The answer lines.
    :type lines: Sequence[str]
    :param width: Image width.
    :type width: int
    :param height: Image height.
    :type height: int
    :param extra: Source-specific fields.
    :type extra: Any
    :return: Entry with line, letter and density counts.
    :rtype: Dict[str, Any]
    """
    letters = arabic_letter_count("\n".join(lines))
    return {"stem": stem, "split": split, "lines": len(lines), "arabic_letters": letters, "image_width": width,
            "image_height": height, "letters_per_megapixel": round(letters / (width * height / 1e6), 1), **extra}


# ----------------------------------------------------------------------------- agapet

def extract_zip(zip_path: Path, dest: Path) -> int:
    """Unzip into a working folder, skipping members already there with the right size.

    :param zip_path: Archive.
    :type zip_path: Path
    :param dest: Destination folder.
    :type dest: Path
    :return: Members written.
    :rtype: int
    """
    present: Dict[str, int] = {}
    for root, _, files in os.walk(dest):
        for name in files:
            path = Path(root) / name
            present[path.relative_to(dest).as_posix()] = path.stat().st_size
    with zipfile.ZipFile(zip_path) as archive:
        todo = [m for m in archive.infolist() if not m.is_dir() and present.get(m.filename) != m.file_size]
        for member in todo:
            archive.extract(member, dest)
    return len(todo)


def agapet_stem(collection: str, xml: Path) -> str:
    """Row id of an Agapet page.

    :param collection: Collection key (``sa418``, ``sin423``, ``bnf76``).
    :type collection: str
    :param xml: The page's PAGE file.
    :type xml: Path
    :return: ``agapet_<collection>_<file name slug>``.
    :rtype: str
    """
    return f"agapet_{collection}_{slug(xml.stem)}"


def agapet_pages(work_dir: Path, collections: Dict[str, str] = AGAPET_COLLECTIONS) -> List[Tuple[str, Path]]:
    """PAGE files of the unzipped Agapet collections.

    :param work_dir: Folder holding one extracted folder per zip (named after the zip).
    :type work_dir: Path
    :param collections: ``{collection key: zip file name}``.
    :type collections: Dict[str, str]
    :return: ``(collection, xml path)`` sorted by collection order, then path; METS files excluded.
    :rtype: List[Tuple[str, Path]]
    """
    pages = []
    for collection, zip_name in collections.items():
        folder = work_dir / Path(zip_name).stem
        pages += [(collection, xml) for xml in sorted(folder.rglob("*.xml")) if xml.name != "METS.xml"]
    return pages


def build_agapet(out: Path, source_dir: Path = EXTERNAL_DIR / "agapet", limit: Optional[int] = None,
                 val_share: float = VAL_SHARE, workers: int = WORKERS, collections: Dict[str, str] = AGAPET_COLLECTIONS,
                 double_page: frozenset = DOUBLE_PAGE_COLLECTIONS,
                 max_density: float = MAX_LETTERS_PER_MEGAPIXEL) -> Dict[str, Any]:
    """Build the Agapet page set.

    :param out: Saved-DatasetDict directory.
    :type out: Path
    :param source_dir: Folder with the zips; they are unzipped to ``<source_dir>/extracted``.
    :type source_dir: Path
    :param limit: Only the first N pages of each collection (dry run).
    :type limit: Optional[int]
    :param val_share: Share of each collection's source pages held out for validation (:func:`ranked_val_pages`).
    :type val_share: float
    :param workers: Threads encoding images.
    :type workers: int
    :param collections: ``{collection key: zip file name}``.
    :type collections: Dict[str, str]
    :param double_page: Collections whose images show two book pages.
    :type double_page: frozenset
    :param max_density: Most Arabic letters per megapixel a page may hold (:func:`too_dense`).
    :type max_density: float
    :return: The statistics.
    :rtype: Dict[str, Any]
    """
    work = source_dir / "extracted"
    for zip_name in collections.values():
        written = extract_zip(source_dir / zip_name, work / Path(zip_name).stem)
        logger.info("%s: %d members extracted", zip_name, written)
    pages = agapet_pages(work, collections)
    if limit:
        pages = [p for c in collections for p in [q for q in pages if q[0] == c][:limit]]
    val_stems = ranked_val_pages({c: [agapet_stem(c, xml) for cc, xml in pages if cc == c] for c in collections}, val_share)
    planned, skipped, counts, reordered = [], Counter(), Counter(), []
    for collection, xml in pages:
        page = parse_page_xml(xml.read_bytes())
        lines, page_counts, reason = xml_page_lines(page, collection in double_page)
        counts.update(page_counts)
        counts["untranscribed small lines omitted"] += sum(1 for line in page.lines() if not clean_line(line.text)) if not reason else 0
        if page_counts["left page listed first: reordered"]:
            reordered.append(xml.stem)
        if reason:
            skipped[f"{collection}: {reason}"] += 1
            continue
        planned.append((collection, xml, xml.parent / page.image_filename, lines))
    rows: Dict[str, RowBatches] = {"train": RowBatches(), "val": RowBatches()}
    manifest = []
    images = map_in_chunks(page_image_file, (image_path for _, _, image_path, _ in planned), workers)
    for (collection, xml, image_path, lines), (data, width, height, how) in zip(planned, images):
        if too_dense(lines, width, height, max_density):
            skipped[f"{collection}: text too dense for the image"] += 1
            continue
        stem = agapet_stem(collection, xml)
        split = "val" if stem in val_stems else "train"
        rows[split].append(page_row(data, "\n".join(lines), stem, width, height, "agapet", SECTIONS["agapet"]))
        manifest.append(manifest_entry(stem, split, lines, width, height, collection=collection,
                                       source=str(image_path.relative_to(source_dir)), image=how))
    stats = {"source": str(source_dir), "pages_seen": len(pages), "skipped_pages": dict(skipped), "counts": dict(counts),
             "by_collection": dict(Counter(f"{m['collection']}/{m['split']}" for m in manifest)),
             "images": dict(Counter(m["image"] for m in manifest)), "double_pages_reordered_right_page_first": reordered}
    return write_dataset("agapet", out, rows, manifest, stats)


# ----------------------------------------------------------------------------- muharaf

def _norm_space(text: str) -> str:
    """NFC with whitespace runs collapsed: the key that matches one line across releases.

    :param text: Line text.
    :type text: str
    :return: Normalised text.
    :rtype: str
    """
    return _SPACES.sub(" ", unicodedata.normalize("NFC", text or "")).strip()


def official_page_splits(page_lines: Dict[str, Sequence[str]], split_texts: Dict[str, Sequence[str]]) -> Dict[str, str]:
    """Recover a page-level split from a line release that has no page ids.

    Each line text of the line release that occurs on exactly one page votes for that page's split;
    a page gets a split when all its votes agree.

    :param page_lines: ``{page: raw line texts}`` of the page release.
    :type page_lines: Dict[str, Sequence[str]]
    :param split_texts: ``{"train" | "validation" | "test": line texts}`` of the line release.
    :type split_texts: Dict[str, Sequence[str]]
    :return: ``{page: "train" | "val"}`` for the pages with unanimous votes.
    :rtype: Dict[str, str]
    """
    pages_of: Dict[str, Set[str]] = {}
    for page, lines in page_lines.items():
        for text in lines:
            key = _norm_space(text)
            if key:
                pages_of.setdefault(key, set()).add(page)
    votes: Dict[str, Set[str]] = {}
    for split, texts in split_texts.items():
        target = next(name for name, sources in LINE_SPLITS.items() if split in sources)
        for text in texts:
            pages = pages_of.get(_norm_space(text), ())
            if len(pages) == 1:
                votes.setdefault(next(iter(pages)), set()).add(target)
    return {page: next(iter(splits)) for page, splits in votes.items() if len(splits) == 1}


def line_release_texts(line_dir: Path, column: str = "text") -> Dict[str, List[str]]:
    """Line texts of a HuggingFace line release, by split (the text column only).

    :param line_dir: Folder of ``<split>-*.parquet`` files.
    :type line_dir: Path
    :param column: Text column.
    :type column: str
    :return: ``{"train" | "validation" | "test": texts}``.
    :rtype: Dict[str, List[str]]
    """
    return {split: [t for f in sorted(line_dir.glob(f"{split}-*.parquet")) for t in pq.read_table(f, columns=[column]).column(column).to_pylist()]
            for split in ("train", "validation", "test")}


def json_only_lines(annotation: Dict[str, Any], xml_texts: Sequence[str]) -> List[str]:
    """Transcribed lines of a Muharaf annotation JSON that its PAGE XML lacks.

    :param annotation: The page's main JSON (``region_dict`` lists each region's lines, ``json`` holds
        each line's ``text``).
    :type annotation: Dict[str, Any]
    :param xml_texts: Line texts of the PAGE XML.
    :type xml_texts: Sequence[str]
    :return: Texts with at least one letter that are in a JSON region but not in the XML.
    :rtype: List[str]
    """
    remaining = Counter(_norm_space(t) for t in xml_texts)
    missing = []
    for region in annotation.get("region_dict", {}).values():
        for line_id in region.get("lines", []):
            text = _norm_space(annotation.get("json", {}).get(line_id, {}).get("text", ""))
            if not any(ch.isalpha() for ch in text):
                continue
            if remaining[text]:
                remaining[text] -= 1
            else:
                missing.append(text)
    return missing


def build_muharaf(out: Path, zip_path: Path = MUHARAF_ZIP, line_dir: Path = MUHARAF_LINES, limit: Optional[int] = None,
                  val_share: float = VAL_SHARE, workers: int = WORKERS,
                  max_density: float = MAX_LETTERS_PER_MEGAPIXEL) -> Dict[str, Any]:
    """Build the Muharaf page set from the full-page release.

    :param out: Saved-DatasetDict directory.
    :type out: Path
    :param zip_path: ``public_data_files.zip`` (read in place).
    :type zip_path: Path
    :param line_dir: The line release's parquet folder (texts give the official split).
    :type line_dir: Path
    :param limit: Only the first N pages (dry run).
    :type limit: Optional[int]
    :param val_share: Validation share for a page the official split does not place.
    :type val_share: float
    :param workers: Threads encoding images.
    :type workers: int
    :param max_density: Most Arabic letters per megapixel a page may hold (:func:`too_dense`).
    :type max_density: float
    :return: The statistics.
    :rtype: Dict[str, Any]
    """
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        xmls = sorted(n for n in names if n.endswith(".xml"))
        parsed = {xml[:-4]: parse_page_xml(archive.read(xml)) for xml in xmls}
        official = official_page_splits({stem: [l.text for l in page.lines()] for stem, page in parsed.items()},
                                        line_release_texts(line_dir))
        stems = sorted(parsed)[:limit] if limit else sorted(parsed)
        row_ids = row_stems("muharaf", [Path(stem).name for stem in parsed])
        planned, skipped, counts = [], Counter(), Counter()
        for stem in stems:
            page = parsed[stem]
            lines, page_counts, reason = xml_page_lines(page, editorial=False)
            counts.update(page_counts)
            image_member = str(Path(stem).parent / page.image_filename)
            if not reason and image_member not in names:
                reason = "image missing from the zip"
            if not reason and json_only_lines(json.loads(archive.read(stem + ".json")), [l.text for l in page.lines()]):
                reason = "the XML export lost transcribed lines of the annotation JSON"
            if reason:
                skipped[reason] += 1
                continue
            counts["untranscribed small lines omitted"] += sum(1 for line in page.lines() if not clean_line(line.text))
            planned.append((stem, image_member, lines))
        rows: Dict[str, RowBatches] = {"train": RowBatches(), "val": RowBatches()}
        manifest = []
        images = map_in_chunks(page_image, (archive.read(member) for _, member, _ in planned), workers)
        for (stem, member, lines), (data, width, height, how) in zip(planned, images):
            if too_dense(lines, width, height, max_density):
                skipped["text too dense for the image"] += 1
                continue
            row_stem = row_ids[Path(stem).name]
            split = official.get(stem) or hash_split(row_stem, val_share)
            rows[split].append(page_row(data, "\n".join(lines), row_stem, width, height, "muharaf", SECTIONS["muharaf"]))
            manifest.append(manifest_entry(row_stem, split, lines, width, height, source=member, image=how,
                                           split_from="official" if stem in official else "hash"))
    stems_out = [m["stem"] for m in manifest]
    assert len(stems_out) == len(set(stems_out)), "two Muharaf pages share a row stem"
    stats = {"source": str(zip_path), "pages_seen": len(stems), "skipped_pages": dict(skipped), "counts": dict(counts),
             "official_split_pages": dict(Counter(official.values())),
             "split_from": dict(Counter(m["split_from"] for m in manifest)), "images": dict(Counter(m["image"] for m in manifest))}
    return write_dataset("muharaf", out, rows, manifest, stats)


# ----------------------------------------------------------------------------- baybars / iskandar

def group_pages(manuscripts: Sequence[str], images: Sequence[str]) -> Dict[Tuple[str, str], List[int]]:
    """Rows of each source page, pages in order of first appearance, rows in row order.

    :param manuscripts: ``manuscript_name`` per row.
    :type manuscripts: Sequence[str]
    :param images: ``image_name`` (the source page) per row.
    :type images: Sequence[str]
    :return: ``{(manuscript, image): row indices}``.
    :rtype: Dict[Tuple[str, str], List[int]]
    """
    pages: Dict[Tuple[str, str], List[int]] = {}
    for i, key in enumerate(zip(manuscripts, images)):
        pages.setdefault(key, []).append(i)
    return pages


def read_line_split(data_dir: Path, splits: Sequence[str]) -> pa.Table:
    """The rows of one or more splits of a line release, in file order.

    :param data_dir: Folder of ``<split>-*.parquet`` files.
    :type data_dir: Path
    :param splits: Split names, read in this order.
    :type splits: Sequence[str]
    :return: Concatenated table.
    :rtype: pa.Table
    """
    files = [f for split in splits for f in sorted(data_dir.glob(f"{split}-*.parquet"))]
    return pa.concat_tables([pq.read_table(f) for f in files])


def page_crops(table: pa.Table, indices: Sequence[int], main_regions: frozenset) -> Tuple[List[str], List[bytes], Counter]:
    """The main-text lines of one page and their crops, aligned.

    :param table: Line rows (``image``, ``transcription``, ``region_type``).
    :type table: pa.Table
    :param indices: The page's row indices, in reading order.
    :type indices: Sequence[int]
    :param main_regions: Region types that are main text.
    :type main_regions: frozenset
    :return: ``(cleaned lines, crop bytes of those lines, counts of dropped lines by reason)``.
    :rtype: Tuple[List[str], List[bytes], Counter]
    """
    picked = table.take(pa.array(indices, pa.int64()))
    types = picked.column("region_type").to_pylist()
    texts = picked.column("transcription").to_pylist()
    counts = Counter(f"dropped region {t}" for t in types if t not in main_regions)
    main = [i for i, t in enumerate(types) if t in main_regions]
    cleaned, line_counts = clean_page_lines([texts[i] or "" for i in main], editorial=False)
    counts.update(line_counts)
    crops = picked.column("image").combine_chunks().field("bytes")
    keep = [(line, crops[main[i]].as_py()) for i, line in enumerate(cleaned) if line]
    return [line for line, _ in keep], [crop for _, crop in keep], counts


LinePage = Tuple[Tuple[str, str], List[str], List[bytes], int]


def usable_line_pages(table: pa.Table, pages: Dict[Tuple[str, str], List[int]], main_regions: frozenset,
                      counts: Counter, skipped: Counter, split: str) -> Iterator[LinePage]:
    """The pages of a line split that pass the gates, read one at a time.

    :param table: Line rows of the split.
    :type table: pa.Table
    :param pages: ``{(manuscript, image): row indices}`` (:func:`group_pages`).
    :type pages: Dict[Tuple[str, str], List[int]]
    :param main_regions: Region types that are main text.
    :type main_regions: frozenset
    :param counts: Line counters (updated).
    :type counts: Counter
    :param skipped: Skipped pages by reason (updated).
    :type skipped: Counter
    :param split: Split name, for the skip reasons.
    :type split: str
    :return: ``((manuscript, image), lines, crop bytes, source rows)`` per usable page.
    :rtype: Iterator[LinePage]
    """
    for key, indices in pages.items():
        lines, crops, page_counts = page_crops(table, indices, main_regions)
        counts.update(page_counts)
        reason = "editorial note in brackets" if page_counts["editorial notes"] else page_gate(lines)
        if reason:
            skipped[f"{split}: {reason}"] += 1
            continue
        yield key, lines, crops, len(indices)


def stack_line_page(page: LinePage) -> Tuple[Tuple[str, str], List[str], int, Tuple[bytes, int, int]]:
    """Stack one page's crops, keeping its metadata and letting its crops go.

    :param page: Output of :func:`usable_line_pages`.
    :type page: LinePage
    :return: ``(key, lines, source rows, (JPEG bytes, width, height))``.
    :rtype: Tuple[Tuple[str, str], List[str], int, Tuple[bytes, int, int]]
    """
    key, lines, crops, n_rows = page
    return key, lines, n_rows, stack_page(crops)


def build_line_dataset(name: str, out: Path, data_dir: Optional[Path] = None, limit: Optional[int] = None,
                       workers: int = WORKERS) -> Dict[str, Any]:
    """Build BAYBARS or ISKANDAR: one stacked page per source page and split.

    :param name: ``"baybars"`` or ``"iskandar"``.
    :type name: str
    :param out: Saved-DatasetDict directory.
    :type out: Path
    :param data_dir: Parquet folder (default ``arabic_external/<name>/data``).
    :type data_dir: Optional[Path]
    :param limit: Only the first N pages of each split (dry run).
    :type limit: Optional[int]
    :param workers: Threads stacking pages.
    :type workers: int
    :return: The statistics.
    :rtype: Dict[str, Any]
    """
    data_dir = data_dir or EXTERNAL_DIR / name / "data"
    rows: Dict[str, RowBatches] = {"train": RowBatches(), "val": RowBatches()}
    manifest, skipped, counts = [], Counter(), Counter()
    for split, sources in LINE_SPLITS.items():
        table = read_line_split(data_dir, sources)
        pages = group_pages(table.column("manuscript_name").to_pylist(), table.column("image_name").to_pylist())
        if limit:
            pages = dict(list(pages.items())[:limit])
        usable = usable_line_pages(table, pages, MAIN_TEXT_REGIONS[name], counts, skipped, split)
        for key, lines, n_rows, (data, width, height) in map_in_chunks(stack_line_page, usable, workers):
            stem = f"{name}_{slug(key[0])}_{slug(key[1])}_{split}"
            rows[split].append(page_row(data, "\n".join(lines), stem, width, height, name, SECTIONS[name]))
            manifest.append(manifest_entry(stem, split, lines, width, height, manuscript=key[0], image_name=key[1],
                                           source_rows=n_rows))
        logger.info("%s %s: %d pages", name, split, len(rows[split]))
        del table
    stats = {"source": str(data_dir), "skipped_pages": dict(skipped), "counts": dict(counts),
             "line_order": "parquet row order" + (" (shuffled at line level in the source: not reading order)" if name == "iskandar" else "")}
    return write_dataset(name, out, rows, manifest, stats)


# ----------------------------------------------------------------------------- CLI

def build(name: str, out_root: Path = DATASETS_DIR, limit: Optional[int] = None, export: bool = True,
          workers: int = WORKERS) -> Dict[str, Any]:
    """Build one dataset and its images-once export.

    :param name: One of :data:`DATASET_NAMES`.
    :type name: str
    :param out_root: Parent of ``arabic_external_<name>_v1`` and its ``_images_once`` export.
    :type out_root: Path
    :param limit: Dry-run page limit (see the builders).
    :type limit: Optional[int]
    :param export: Also write the images-once export.
    :type export: bool
    :param workers: Threads for image work.
    :type workers: int
    :return: The statistics, with the export manifest summary under ``images_once``.
    :rtype: Dict[str, Any]
    """
    out = out_root / f"arabic_external_{name}_{VERSION}"
    if name == "agapet":
        stats = build_agapet(out, limit=limit, workers=workers)
    elif name == "muharaf":
        stats = build_muharaf(out, limit=limit, workers=workers)
    else:
        stats = build_line_dataset(name, out, limit=limit, workers=workers)
    if export:
        manifest = export_images_once(out, out.with_name(out.name + "_images_once"))
        stats["images_once"] = {k: v for k, v in manifest.items() if k != "features"}
    return stats


def main(argv: Optional[List[str]] = None) -> None:
    """CLI.

    :param argv: Arguments.
    :type argv: Optional[List[str]]
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_NAMES, default=list(DATASET_NAMES))
    parser.add_argument("--out-root", type=Path, default=DATASETS_DIR)
    parser.add_argument("--limit", type=int, default=None, help="dry run: first N pages (per collection / split)")
    parser.add_argument("--no-export", action="store_true", help="skip the images-once export")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args(argv)
    for name in args.datasets:
        stats = build(name, args.out_root, args.limit, not args.no_export, args.workers)
        print(json.dumps({k: v for k, v in stats.items() if k not in ("prompt", "double_pages_reordered_right_page_first")},
                         ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
