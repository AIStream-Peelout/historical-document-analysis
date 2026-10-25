# File name: build_arabic_lines.py
# Date: 10/19/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Kraken 7.1.1 recognition-training layout for Arabic handwriting from four public sets.

The Arabic recogniser is a PP-OCRv6 fine-tune trained on baseline lines (``ketos compile -f page``
for PAGE XML, ``ketos compile -f path --force-type baseline`` for pre-cropped lines). This builder
turns the public Arabic HTR sets on the NAS into that layout:

* **agapet** (CC BY 4.0): SA-418 (342 pages) and BnF Arabe 76 (11 TIFF pages), PAGE XML next to
  the images. Sin423 is excluded (its images are 38-px-line thumbnails).
* **rasam** (Apache-2.0): RASAM 1 + 2 PAGE XML; images fetched over IIIF by
  :mod:`src.finetuning.kraken.fetch_page_images` into ``rasam_pages/images``. The served image is
  smaller than the XML frame (same aspect), so every ``Coords`` and ``Baseline`` point is rescaled
  per axis to the real image size.
* **iskandar** (Etalab 2.0): PAGE XML; only the Gotha pages have fetched images (the rest are
  skipped). About half of its lines are layout-only (no text) and are dropped.
* **baybars** (Etalab 2.0): no page images exist, so the line-crop release (parquet) is used. The
  black fill outside each line polygon becomes the crop's paper colour
  (:func:`~src.finetuning.qwen_hebrew.build_arabic_external.fill_mask` /
  :func:`~src.finetuning.qwen_hebrew.build_arabic_external.paper_color`); main-text regions only.

Every target line follows the TEXT RULES (:func:`clean_target`): NFC; tatweel and U+064B to U+0652
removed; bidi controls removed; whitespace collapsed. A line with square / angle brackets or braces
(editorial marks), a Latin letter (shelfmarks, notes) or fewer than :data:`MIN_ARABIC_LETTERS` Arabic
letters is dropped: in PAGE XML its ``Unicode`` is emptied so ``ketos compile`` skips it, a crop is
not written. Lines without a usable ``<Baseline>`` or ``<Coords>`` are emptied too (ketos would drop
them anyway). A page with no line left is not written.

Splits: PAGE sets are split by page, stratified by source manuscript (:func:`assign_splits`), over
the set's SOURCE pages (:func:`source_splits`), so a page's split does not depend on which other pages
have an image or survive cleaning: in a manuscript with at least :data:`MIN_VAL_PAGES` source pages
the :data:`VAL_SHARE` of its pages with the lowest SHA-1 of their page key go to val (at least one);
pages of smaller manuscripts go to val when their SHA-1 bucket falls under the share. A manuscript
with at least :data:`MIN_VAL_PAGES` kept pages but no kept val page (its val pages were skipped) sends
its kept page with the lowest SHA-1 to val (:func:`split_results`, logged), so every such manuscript
is in val. Baybars keeps the official split (train to train, validation + test to val).

Output (``--out-root``)::

    page_xml/<set>/<stem>.xml          cleaned PAGE copies (absolute imageFilename, real image size)
    crops/baybars/<stem>.png           paper-filled line crops + <stem>.gt.txt (stems without dots)
    lists/<set>_train.lst, <set>_val.lst   one absolute path per line (the lists are authoritative)
    stats.json                         per set: pages / lines seen, kept, dropped by reason, letters, splits

The image paths are absolute host paths, so a training container must mount the NAS at the same
path. New sets (e.g. Muharaf, CC BY-NC-SA, as a later NC arm) are added as a :class:`SetSpec` in
:data:`SETS`.

Usage (repo root)::

    .venv/bin/python -m src.finetuning.kraken.build_arabic_lines [--sets agapet rasam iskandar baybars]
        [--out-root DIR] [--limit N] [--workers 4]
"""
import argparse
import csv
import hashlib
import io
import json
import logging
import re
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
from lxml import etree
from PIL import Image

logger = logging.getLogger(__name__)

EXTERNAL_DIR = Path("/Volumes/home/studio_offload/datasets/arabic_external")
OUT_ROOT = Path("/Volumes/home/studio_offload/datasets/kraken_arabic_lines/v1")
VAL_SHARE = 0.08
MIN_VAL_PAGES = 10             # a manuscript with this many kept pages always has a val page
MIN_ARABIC_LETTERS = 2
ASPECT_TOLERANCE = 0.02        # per-axis scales differing by more than this: a different image, not a resize
WORKERS = 4
CROP_BATCH = 64                # Baybars rows per parquet batch / thread-pool flush
PAGE_SET_DIRS: Dict[str, str] = {"agapet": "agapet/extracted", "rasam": "rasam_pages", "iskandar": "iskandar_pages"}
AGAPET_COLLECTIONS: Dict[str, str] = {"sa418": "SA-418 (13th cent)/SA-418", "bnf76": "BnF Arabe 76 pp 68-78"}
EXCLUDED_AGAPET = {"sin423": "Sin423 (17th cent)"}   # 38-px line thumbnails: unreadable at kraken's line height
NON_PAGE_FILES = frozenset({"METS.xml", "mets.xml"})
IMAGE_SUFFIXES = (".jpeg", ".jpg", ".tif", ".tiff", ".png")
BAYBARS_DIR = "baybars/data"
BAYBARS_SPLITS: Dict[str, Tuple[str, ...]] = {"train": ("train",), "val": ("validation", "test")}
BAYBARS_COLUMNS = ["image", "transcription", "region_type", "manuscript_name", "image_name"]
MAIN_TEXT_REGIONS = frozenset({"MainText", "MainText_Right", "MainText_Left"})
TEXTLINE_SCHEMA_BEFORE_TEXT = frozenset({"AlternativeImage", "Coords", "Baseline", "Word"})

_STRIP = re.compile(r"[ـً-ْ]")                                # tatweel, short vowels, tanwin, shadda, sukun
_BIDI = re.compile(r"[؜‎‏‪-‮⁦-⁩﻿]")  # invisible direction controls
_SPACES = re.compile(r"\s+")
_EDITORIAL = re.compile(r"[\[\]<>{}〈〉⟨⟩〈〉]")   # square, angle (incl. math/CJK forms), braces
_LATIN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ]")
_ARABIC_LETTER = re.compile(r"[ؠ-يٮ-ۓ]")
_POINT = re.compile(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)")

DROP_EMPTY = "empty"
DROP_EDITORIAL = "editorial brackets"
DROP_LATIN = "latin letter"
DROP_FEW_LETTERS = f"fewer than {MIN_ARABIC_LETTERS} arabic letters"
DROP_NO_BASELINE = "no baseline"
DROP_NO_COORDS = "no coords"
KEPT = "kept"


# ----------------------------------------------------------------------------- records

@dataclass(frozen=True)
class PageSource:
    """One source page of a PAGE XML set.

    :param key: Stable page key for the split (``<set>/<source stem>``).
    :param stem: Output file stem (ASCII letters, digits and ``_`` only).
    :param manuscript: Source manuscript (the split stratum).
    :param xml_path: Source PAGE XML.
    :param image_path: The page image; None when no image is available (the page is skipped).
    :param served: ``(width, height)`` the image manifest recorded, checked against the file; None when
        the set has no manifest.
    """
    key: str
    stem: str
    manuscript: str
    xml_path: Path
    image_path: Optional[Path]
    served: Optional[Tuple[int, int]] = None


@dataclass
class PageResult:
    """What became of one source page.

    :param key: Stable page key.
    :param manuscript: Source manuscript.
    :param skip: Why the page was not written; ``""`` when it was.
    :param counts: Line outcomes: :data:`KEPT` and drop reasons.
    :param letters: Arabic letters in the kept lines.
    :param paths: Written training files (one PAGE XML, or the PNG crops of a line set).
    :param split: ``"train"`` or ``"val"`` once assigned.
    :param edits: Page edits worth counting (rescaled coordinates, removed alternative transcriptions).
    """
    key: str
    manuscript: str
    skip: str = ""
    counts: Counter = field(default_factory=Counter)
    letters: int = 0
    paths: List[Path] = field(default_factory=list)
    split: str = ""
    edits: Counter = field(default_factory=Counter)


# ----------------------------------------------------------------------------- text

def arabic_letter_count(text: str) -> int:
    """Number of Arabic letters (U+0620 to U+064A and U+066E to U+06D3).

    :param text: Any text.
    :type text: str
    :return: Letter count (tatweel counts; :func:`clean_target` removes it first).
    :rtype: int
    """
    return len(_ARABIC_LETTER.findall(text))


def normalize_line(text: str) -> str:
    """Normalise one transcribed line without judging it.

    :param text: Raw line text.
    :type text: str
    :return: NFC text without tatweel, U+064B to U+0652 and bidi controls, whitespace runs collapsed to
        one space, stripped.
    :rtype: str
    """
    text = _BIDI.sub("", _STRIP.sub("", unicodedata.normalize("NFC", text)))
    return _SPACES.sub(" ", unicodedata.normalize("NFC", text)).strip()


def clean_target(text: str) -> Tuple[str, str]:
    """Apply the TEXT RULES to one line.

    :param text: Raw line text (None-safe callers pass ``""``).
    :type text: str
    :return: ``(normalised text, "")`` for a kept line, ``("", reason)`` for a dropped one; reasons are
        :data:`DROP_EMPTY`, :data:`DROP_EDITORIAL`, :data:`DROP_LATIN`, :data:`DROP_FEW_LETTERS`.
    :rtype: Tuple[str, str]
    """
    line = normalize_line(text)
    if not line:
        return "", DROP_EMPTY
    if _EDITORIAL.search(line):
        return "", DROP_EDITORIAL
    if _LATIN.search(line):
        return "", DROP_LATIN
    if arabic_letter_count(line) < MIN_ARABIC_LETTERS:
        return "", DROP_FEW_LETTERS
    return line, ""


def slug(text: str) -> str:
    """Lower-case file-name fragment.

    :param text: Any name.
    :type text: str
    :return: Runs of anything but ASCII letters and digits as ``_`` (so never a dot).
    :rtype: str
    """
    return re.sub(r"[^0-9a-z]+", "_", text.lower()).strip("_")


def unique_stems(prefix: str, names: Sequence[str]) -> Dict[str, str]:
    """Output stems ``<prefix>_<slug>``, with a short hash where two names share a slug.

    :param prefix: Set prefix.
    :type prefix: str
    :param names: Source names.
    :type names: Sequence[str]
    :return: ``{name: stem}``.
    :rtype: Dict[str, str]
    """
    seen = Counter(slug(name) for name in names)
    return {name: f"{prefix}_{slug(name)}" if seen[slug(name)] == 1
            else f"{prefix}_{slug(name)}_{hashlib.sha1(name.encode('utf-8')).hexdigest()[:6]}" for name in names}


# ----------------------------------------------------------------------------- geometry

def parse_points(points: str) -> List[Tuple[float, float]]:
    """Points of a PAGE ``points`` attribute.

    :param points: ``"x1,y1 x2,y2 ..."`` (integers or decimals).
    :type points: str
    :return: ``[(x, y), ...]``.
    :rtype: List[Tuple[float, float]]
    """
    return [(float(x), float(y)) for x, y in _POINT.findall(points or "")]


def rescale_points(points: str, sx: float, sy: float, width: int, height: int) -> str:
    """Scale a PAGE ``points`` string per axis, round, and clamp it into the image.

    :param points: ``"x1,y1 x2,y2 ..."``.
    :type points: str
    :param sx: Horizontal scale (real image width / XML frame width).
    :type sx: float
    :param sy: Vertical scale.
    :type sy: float
    :param width: Real image width (x is clamped to ``0 .. width - 1``).
    :type width: int
    :param height: Real image height.
    :type height: int
    :return: Integer points joined by single spaces.
    :rtype: str
    """
    out = []
    for x, y in parse_points(points):
        px = min(max(int(round(x * sx)), 0), width - 1)
        py = min(max(int(round(y * sy)), 0), height - 1)
        out.append(f"{px},{py}")
    return " ".join(out)


def frame_scale(xml_size: Tuple[int, int], image_size: Tuple[int, int]) -> Tuple[float, float]:
    """Per-axis scale from the XML frame to the real image.

    :param xml_size: ``(imageWidth, imageHeight)`` the XML declares; a zero means undeclared.
    :type xml_size: Tuple[int, int]
    :param image_size: ``(width, height)`` of the image file.
    :type image_size: Tuple[int, int]
    :return: ``(sx, sy)``; 1.0 on an undeclared axis.
    :rtype: Tuple[float, float]
    """
    sx = image_size[0] / xml_size[0] if xml_size[0] > 0 else 1.0
    sy = image_size[1] / xml_size[1] if xml_size[1] > 0 else 1.0
    return sx, sy


def aspect_mismatch(sx: float, sy: float, tolerance: float = ASPECT_TOLERANCE) -> bool:
    """Whether a frame-to-image scale is not a plain resize (rotated, cropped or another image).

    :param sx: Horizontal scale.
    :type sx: float
    :param sy: Vertical scale.
    :type sy: float
    :param tolerance: Allowed relative difference of the two scales.
    :type tolerance: float
    :return: True when the scales differ by more than ``tolerance``.
    :rtype: bool
    """
    return abs(sx - sy) > tolerance * max(sx, sy)


# ----------------------------------------------------------------------------- PAGE XML

def _local(element: Any) -> str:
    """Local name of an element (no namespace).

    :param element: lxml element.
    :type element: Any
    :return: Tag without namespace.
    :rtype: str
    """
    return etree.QName(element).localname


def page_element(root: Any) -> Any:
    """The ``<Page>`` element of a PAGE document.

    :param root: ``PcGts`` root.
    :type root: Any
    :return: The first ``Page`` element.
    :rtype: Any
    :raises ValueError: When the document has no ``Page``.
    """
    page = root.find("{*}Page")
    if page is None:
        raise ValueError("not a PAGE document: no <Page> element")
    return page


def declared_size(page: Any) -> Tuple[int, int]:
    """``imageWidth`` / ``imageHeight`` of a ``<Page>``.

    :param page: ``Page`` element.
    :type page: Any
    :return: ``(width, height)``; 0 where undeclared.
    :rtype: Tuple[int, int]
    """
    return int(float(page.get("imageWidth") or 0)), int(float(page.get("imageHeight") or 0))


def line_text(line: Any) -> str:
    """The transcription of a ``TextLine``, read the way kraken reads it.

    :param line: ``TextLine`` element.
    :type line: Any
    :return: The ``Unicode`` text of the line's first ``TextEquiv``; without one, the words' first
        transcriptions joined by spaces (kraken would glue them without spaces).
    :rtype: str
    """
    equiv = line.find("{*}TextEquiv")
    if equiv is not None:
        return "".join(u.text or "" for u in equiv.findall(".//{*}Unicode"))
    words = []
    for word in line.findall("{*}Word"):
        word_equiv = word.find("{*}TextEquiv")
        if word_equiv is not None:
            words.append("".join(u.text or "" for u in word_equiv.findall(".//{*}Unicode")))
    return " ".join(w for w in words if w)


def set_line_text(line: Any, text: str) -> int:
    """Write a line's transcription into its first ``TextEquiv/Unicode``.

    Further line-level ``TextEquiv`` elements (alternative readings) are removed, so whichever one a
    reader picks it reads ``text``. A line without a ``TextEquiv`` but with word transcriptions gets one
    (placed where the PAGE schema puts it, after the words); a line without any text is left alone.

    :param line: ``TextLine`` element (modified in place).
    :type line: Any
    :param text: Cleaned text; ``""`` empties the line.
    :type text: str
    :return: Number of alternative ``TextEquiv`` elements removed.
    :rtype: int
    """
    ns = etree.QName(line).namespace
    equivs = line.findall("{*}TextEquiv")
    if not equivs:
        if not line.findall(".//{*}Unicode") and not text:
            return 0
        children = list(line)
        before = [i for i, child in enumerate(children)
                  if isinstance(child.tag, str) and _local(child) in TEXTLINE_SCHEMA_BEFORE_TEXT]
        equiv = etree.Element(f"{{{ns}}}TextEquiv")
        line.insert(before[-1] + 1 if before else len(children), equiv)
        equivs = [equiv]
    first = equivs[0]
    for extra in equivs[1:]:
        line.remove(extra)
    unicodes = first.findall("{*}Unicode")
    if not unicodes:
        unicodes = [etree.SubElement(first, f"{{{ns}}}Unicode")]
    for extra in unicodes[1:]:
        first.remove(extra)
    unicodes[0].text = text
    return len(equivs) - 1


def geometry_reason(line: Any) -> str:
    """Why a line cannot be compiled as a baseline line.

    :param line: ``TextLine`` element.
    :type line: Any
    :return: :data:`DROP_NO_BASELINE` (missing or under two points), :data:`DROP_NO_COORDS` (missing or
        under three points), or ``""``.
    :rtype: str
    """
    baseline = line.find("{*}Baseline")
    if baseline is None or len(set(parse_points(baseline.get("points", "")))) < 2:
        return DROP_NO_BASELINE
    coords = line.find("{*}Coords")
    if coords is None or len(set(parse_points(coords.get("points", "")))) < 3:
        return DROP_NO_COORDS
    return ""


def parse_xml(data: bytes) -> Any:
    """Parse a PAGE document (no entity expansion, large files allowed).

    :param data: XML bytes.
    :type data: bytes
    :return: Root element.
    :rtype: Any
    """
    return etree.fromstring(data, parser=etree.XMLParser(huge_tree=True, resolve_entities=False, no_network=True))


def rewrite_tree(root: Any, image_path: str, width: int, height: int) -> Tuple[Counter, int, Counter]:
    """Clean a parsed PAGE document in place for ``ketos compile -f page``.

    ``imageFilename`` becomes ``image_path`` and ``imageWidth`` / ``imageHeight`` the real size; every
    ``Coords`` and ``Baseline`` ``points`` (regions, lines, words) is scaled from the declared frame to
    the real size and clamped into it; every ``TextLine`` text is cleaned by :func:`clean_target`, and
    a dropped line (text rules or geometry) gets an empty ``Unicode``. Namespace, schema location,
    element order and ``<Baseline>`` elements are kept.

    :param root: ``PcGts`` root (modified in place).
    :type root: Any
    :param image_path: Absolute image path to write into ``imageFilename``.
    :type image_path: str
    :param width: Real image width.
    :type width: int
    :param height: Real image height.
    :type height: int
    :return: ``(line outcomes (KEPT + drop reasons), Arabic letters kept, edits)``.
    :rtype: Tuple[Counter, int, Counter]
    """
    page = page_element(root)
    sx, sy = frame_scale(declared_size(page), (width, height))
    edits: Counter = Counter()
    edits["rescaled pages"] += (sx, sy) != (1.0, 1.0)
    page.set("imageFilename", image_path)
    page.set("imageWidth", str(width))
    page.set("imageHeight", str(height))
    for element in root.iter(etree.Element):
        if _local(element) in ("Coords", "Baseline") and element.get("points") is not None:
            element.set("points", rescale_points(element.get("points"), sx, sy, width, height))
    counts: Counter = Counter()
    letters = 0
    for line in root.iter(etree.Element):
        if _local(line) != "TextLine":
            continue
        text, reason = clean_target(line_text(line))
        reason = reason or geometry_reason(line)
        if reason:
            text = ""
            counts[reason] += 1
        else:
            counts[KEPT] += 1
            letters += arabic_letter_count(text)
        edits["alternative transcriptions removed"] += set_line_text(line, text)
    return counts, letters, Counter({k: v for k, v in edits.items() if v})


def serialize(root: Any) -> bytes:
    """Serialise a PAGE document.

    :param root: Root element.
    :type root: Any
    :return: UTF-8 XML with a declaration.
    :rtype: bytes
    """
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8")


def rewrite_page(data: bytes, image_path: str, width: int, height: int) -> Tuple[bytes, Counter, int, Counter]:
    """Clean a PAGE document for ``ketos compile -f page`` (bytes in, bytes out; see :func:`rewrite_tree`).

    :param data: Source PAGE XML.
    :type data: bytes
    :param image_path: Absolute image path to write into ``imageFilename``.
    :type image_path: str
    :param width: Real image width.
    :type width: int
    :param height: Real image height.
    :type height: int
    :return: ``(new XML bytes, line outcomes (KEPT + drop reasons), Arabic letters kept, edits)``.
    :rtype: Tuple[bytes, Counter, int, Counter]
    """
    root = parse_xml(data)
    counts, letters, edits = rewrite_tree(root, image_path, width, height)
    return serialize(root), counts, letters, edits


# ----------------------------------------------------------------------------- page sets: sources

def read_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    """Rows of a fetched-image ``manifest.jsonl``, by stem.

    Only lines ending with a newline are read: a fetcher may still be appending, so a last line
    without one may be half-written (its page counts as not fetched until the next build).

    :param path: ``images/manifest.jsonl`` (may not exist yet).
    :type path: Path
    :return: ``{stem: row}``; empty when the file is missing.
    :rtype: Dict[str, Dict[str, Any]]
    """
    if not path.exists():
        return {}
    complete = path.read_text(encoding="utf-8").split("\n")[:-1]   # the last piece is "" or an unfinished line
    rows = [json.loads(line) for line in complete if line.strip()]
    return {row["stem"]: row for row in rows}


def manifest_sources(set_name: str, xml_paths: Sequence[Path], manuscripts: Dict[str, str],
                     image_dir: Path) -> List[PageSource]:
    """Page sources of a set whose images were fetched into ``image_dir`` with a manifest.

    :param set_name: Set name (key and stem prefix).
    :type set_name: str
    :param xml_paths: Source PAGE files, in build order.
    :type xml_paths: Sequence[Path]
    :param manuscripts: ``{xml stem: manuscript}``; a stem missing from it is manuscript ``"unknown"``.
    :type manuscripts: Dict[str, str]
    :param image_dir: Folder of ``<stem>.jpg`` + ``manifest.jsonl``.
    :type image_dir: Path
    :return: Sources; a page without a manifest row has ``image_path=None``.
    :rtype: List[PageSource]
    """
    manifest = read_manifest(image_dir / "manifest.jsonl")
    stems = unique_stems(set_name, [p.stem for p in xml_paths])
    sources = []
    for xml in xml_paths:
        row = manifest.get(xml.stem)
        sources.append(PageSource(
            key=f"{set_name}/{xml.stem}", stem=stems[xml.stem], manuscript=manuscripts.get(xml.stem, "unknown"),
            xml_path=xml, image_path=image_dir / f"{xml.stem}.jpg" if row else None,
            served=(int(row["served_width"]), int(row["served_height"])) if row else None))
    return sources


def agapet_sources(root: Path) -> List[PageSource]:
    """Agapet pages of the kept collections (images next to the XML, same stem).

    :param root: ``agapet/extracted``.
    :type root: Path
    :return: Sources in collection then file order; a page without an image file has ``image_path=None``.
    :rtype: List[PageSource]
    """
    sources = []
    for collection, sub in AGAPET_COLLECTIONS.items():
        folder = root / sub
        names = sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []
        images = {Path(n).stem: folder / n for n in names if Path(n).suffix.lower() in IMAGE_SUFFIXES}
        xmls = [folder / n for n in names if n.endswith(".xml") and n not in NON_PAGE_FILES]
        stems = unique_stems(f"agapet_{collection}", [p.stem for p in xmls])
        sources.extend(PageSource(key=f"agapet/{collection}/{xml.stem}", stem=stems[xml.stem], manuscript=collection,
                                  xml_path=xml, image_path=images.get(xml.stem)) for xml in xmls)
    return sources


def rasam_sources(root: Path) -> List[PageSource]:
    """RASAM 1 + 2 pages; manuscript from ``list-images.tsv``, images from ``images/manifest.jsonl``.

    :param root: ``rasam_pages``.
    :type root: Path
    :return: Sources in file order.
    :rtype: List[PageSource]
    """
    manuscripts: Dict[str, str] = {}
    listing = root / "list-images.tsv"
    if listing.exists():
        with open(listing, encoding="utf-8", newline="") as fh:
            manuscripts = {row["FileName"]: row["Manuscript"] for row in csv.DictReader(fh, delimiter="\t")}
    xmls = sorted((root / "page").glob("rasam*/*.xml"))
    return manifest_sources("rasam", xmls, manuscripts, root / "images")


def iskandar_sources(root: Path) -> List[PageSource]:
    """Iskandar pages; manuscript from ``iskandar-images-list.csv``, images from ``images/manifest.jsonl``.

    :param root: ``iskandar_pages``.
    :type root: Path
    :return: Sources in file order (only fetched pages have an image).
    :rtype: List[PageSource]
    """
    manuscripts: Dict[str, str] = {}
    listing = root / "iskandar-images-list.csv"
    if listing.exists():
        with open(listing, encoding="utf-8", newline="") as fh:
            manuscripts = {Path(row["XML_File_Name"]).stem: row["Manuscript"] for row in csv.DictReader(fh)}
    xmls = sorted((root / "XML-pages").glob("*.xml"))
    return manifest_sources("iskandar", xmls, manuscripts, root / "images")


# ----------------------------------------------------------------------------- page sets: build

def image_size(path: Path) -> Tuple[int, int]:
    """Pixel size of an image file (header only).

    :param path: Image file.
    :type path: Path
    :return: ``(width, height)``.
    :rtype: Tuple[int, int]
    """
    with Image.open(path) as image:
        return image.size


def process_page(source: PageSource, out_dir: Path) -> PageResult:
    """Clean one source page and write it to ``out_dir/<stem>.xml``.

    :param source: The page.
    :type source: PageSource
    :param out_dir: ``page_xml/<set>``.
    :type out_dir: Path
    :return: The outcome; skip reasons are ``no image``, ``image size differs from manifest``,
        ``not a PAGE document``, ``image aspect differs from XML frame`` and ``no kept lines``.
    :rtype: PageResult
    """
    result = PageResult(key=source.key, manuscript=source.manuscript)
    if source.image_path is None or not source.image_path.exists():
        result.skip = "no image"
        return result
    width, height = image_size(source.image_path)
    if source.served is not None and source.served != (width, height):
        result.skip = "image size differs from manifest"
        return result
    root = parse_xml(source.xml_path.read_bytes())
    if root.find("{*}Page") is None:
        result.skip = "not a PAGE document"
        return result
    if aspect_mismatch(*frame_scale(declared_size(page_element(root)), (width, height))):
        result.skip = "image aspect differs from XML frame"
        return result
    result.counts, result.letters, result.edits = rewrite_tree(root, str(source.image_path.absolute()), width, height)
    if not result.counts[KEPT]:
        result.skip = "no kept lines"
        return result
    out_dir.mkdir(parents=True, exist_ok=True)
    out = (out_dir / f"{source.stem}.xml").absolute()
    out.write_bytes(serialize(root))
    result.paths = [out]
    return result


def process_pages(sources: Sequence[PageSource], out_dir: Path, limit: Optional[int] = None,
                  workers: int = WORKERS) -> List[PageResult]:
    """Process sources in order on a thread pool, stopping once ``limit`` pages are written.

    With a limit, a chunk never holds more pages than are still needed, so no page beyond the limit is
    written.

    :param sources: Pages in build order.
    :type sources: Sequence[PageSource]
    :param out_dir: ``page_xml/<set>``.
    :type out_dir: Path
    :param limit: Written pages to stop at; None for all.
    :type limit: Optional[int]
    :param workers: Threads.
    :type workers: int
    :return: Results of the pages looked at, in source order.
    :rtype: List[PageResult]
    """
    results: List[PageResult] = []
    kept, i, next_log = 0, 0, 100
    work = partial(process_page, out_dir=out_dir)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while i < len(sources) and (limit is None or kept < limit):
            size = workers * 4 if limit is None else min(workers, limit - kept)
            for result in pool.map(work, sources[i:i + size]):
                results.append(result)
                kept += not result.skip
            i += size
            if len(results) >= next_log:
                logger.info("  %d / %d pages looked at, %d written", len(results), len(sources), kept)
                next_log += 100
    return results


def sha1_hex(key: str) -> str:
    """SHA-1 hex digest of a key.

    :param key: Any string.
    :type key: str
    :return: 40 hex characters.
    :rtype: str
    """
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def hash_split(key: str, val_share: float = VAL_SHARE) -> str:
    """Per-page split from the SHA-1 bucket of its key.

    :param key: Stable page key.
    :type key: str
    :param val_share: Share of pages that go to val.
    :type val_share: float
    :return: ``"val"`` or ``"train"``.
    :rtype: str
    """
    return "val" if int(sha1_hex(key), 16) % 10_000 < round(val_share * 10_000) else "train"


def assign_splits(keys_by_group: Dict[str, Sequence[str]], val_share: float = VAL_SHARE,
                  min_pages: int = MIN_VAL_PAGES) -> Dict[str, str]:
    """Deterministic page split, stratified by manuscript.

    A group with at least ``min_pages`` pages sends its ``max(1, round(val_share * n))`` pages with the
    lowest SHA-1 of their key to val; a smaller group splits page by page (:func:`hash_split`).

    :param keys_by_group: ``{manuscript: page keys}``.
    :type keys_by_group: Dict[str, Sequence[str]]
    :param val_share: Val share.
    :type val_share: float
    :param min_pages: Group size from which val is guaranteed.
    :type min_pages: int
    :return: ``{key: "train" | "val"}``.
    :rtype: Dict[str, str]
    """
    splits: Dict[str, str] = {}
    for keys in keys_by_group.values():
        if len(keys) >= min_pages:
            n_val = max(1, round(val_share * len(keys)))
            val = set(sorted(keys, key=sha1_hex)[:n_val])
            splits.update({k: "val" if k in val else "train" for k in keys})
        else:
            splits.update({k: hash_split(k, val_share) for k in keys})
    return splits


def source_splits(sources: Sequence[PageSource], val_share: float = VAL_SHARE,
                  min_pages: int = MIN_VAL_PAGES) -> Dict[str, str]:
    """Train / val of every source page of a PAGE set, stratified by manuscript (:func:`assign_splits`).

    The split depends only on the source listing, not on which pages have an image or survive cleaning,
    so a page keeps its split when another page is skipped or comes back.

    :param sources: All the set's source pages.
    :type sources: Sequence[PageSource]
    :param val_share: Val share.
    :type val_share: float
    :param min_pages: Manuscript size (in source pages) from which val is guaranteed.
    :type min_pages: int
    :return: ``{page key: "train" | "val"}``.
    :rtype: Dict[str, str]
    """
    groups: Dict[str, List[str]] = {}
    for source in sources:
        groups.setdefault(source.manuscript, []).append(source.key)
    return assign_splits(groups, val_share, min_pages)


def split_results(results: Sequence[PageResult], splits: Dict[str, str], min_pages: int = MIN_VAL_PAGES) -> List[str]:
    """Copy the source splits onto the written pages of a PAGE set (in place), keeping the val guarantee.

    A manuscript with at least ``min_pages`` written pages but no written val page (its val pages were
    skipped) sends its written page with the lowest SHA-1 of its key to val.

    :param results: Page results; only pages without a skip reason are assigned.
    :type results: Sequence[PageResult]
    :param splits: ``{page key: split}`` of the source pages (:func:`source_splits`).
    :type splits: Dict[str, str]
    :param min_pages: Written pages per manuscript from which val is guaranteed.
    :type min_pages: int
    :return: Keys of the pages moved to val by the guarantee.
    :rtype: List[str]
    """
    by_manuscript: Dict[str, List[PageResult]] = {}
    for r in results:
        if not r.skip:
            r.split = splits[r.key]
            by_manuscript.setdefault(r.manuscript, []).append(r)
    moved = []
    for pages in by_manuscript.values():
        if len(pages) >= min_pages and not any(r.split == "val" for r in pages):
            page = min(pages, key=lambda r: sha1_hex(r.key))
            page.split = "val"
            moved.append(page.key)
    return moved


def build_page_set(name: str, sources: Sequence[PageSource], out_root: Path, limit: Optional[int] = None,
                   workers: int = WORKERS) -> List[PageResult]:
    """Write the cleaned PAGE copies of one set and assign their splits (decided on the source pages).

    :param name: Set name.
    :type name: str
    :param sources: The set's pages.
    :type sources: Sequence[PageSource]
    :param out_root: Output root.
    :type out_root: Path
    :param limit: Pages to write at most.
    :type limit: Optional[int]
    :param workers: Threads.
    :type workers: int
    :return: Page results with splits.
    :rtype: List[PageResult]
    """
    splits = source_splits(sources, VAL_SHARE, MIN_VAL_PAGES)
    results = process_pages(sources, out_root / "page_xml" / name, limit, workers)
    for key in split_results(results, splits, MIN_VAL_PAGES):
        logger.info("  %s: the manuscript of %s had no written val page, moved it to val", name, key)
    return results


# ----------------------------------------------------------------------------- baybars crops

def paper_fill(rgb: np.ndarray) -> np.ndarray:
    """Replace the black fill outside a line polygon by the crop's paper colour.

    :param rgb: ``(h, w, 3)`` uint8 crop.
    :type rgb: np.ndarray
    :return: A filled copy (the input itself when the crop has no fill).
    :rtype: np.ndarray
    """
    # Imported here: the page-set helpers pull in `datasets`, which the kraken image lacks.
    from src.finetuning.qwen_hebrew.build_arabic_external import fill_mask, paper_color
    mask = fill_mask(rgb)
    if not mask.any():
        return rgb
    out = rgb.copy()
    out[mask] = np.clip(np.round(paper_color(rgb, mask)), 0, 255).astype(np.uint8)
    return out


def write_crop(job: Tuple[str, bytes, str], out_dir: Path) -> Path:
    """Write one paper-filled line crop and its ground truth.

    :param job: ``(stem, encoded crop, cleaned text)``.
    :type job: Tuple[str, bytes, str]
    :param out_dir: ``crops/<set>``.
    :type out_dir: Path
    :return: Absolute path of the PNG.
    :rtype: Path
    """
    stem, data, text = job
    with Image.open(io.BytesIO(data)) as image:
        rgb = np.asarray(image.convert("RGB"))
    png = (out_dir / f"{stem}.png").absolute()
    Image.fromarray(paper_fill(rgb)).save(png, format="PNG")
    (out_dir / f"{stem}.gt.txt").write_text(unicodedata.normalize("NFC", text), encoding="utf-8")
    return png


def crop_stem(official_split: str, manuscript: str, image_name: str, row: int) -> str:
    """Dot-free stem of a Baybars crop.

    :param official_split: ``train`` / ``validation`` / ``test``.
    :type official_split: str
    :param manuscript: ``manuscript_name``.
    :type manuscript: str
    :param image_name: ``image_name`` (its extension is dropped).
    :type image_name: str
    :param row: Row index within the official split.
    :type row: int
    :return: ``baybars_<split>_<manuscript>_<image>_<row>``.
    :rtype: str
    """
    return f"baybars_{official_split}_{slug(manuscript)}_{slug(Path(image_name).stem)}_{row:05d}"


def parquet_rows(files: Sequence[Path], batch_size: int = CROP_BATCH) -> Iterator[Dict[str, Any]]:
    """Rows of parquet files, streamed a batch at a time.

    :param files: Parquet files, read in order.
    :type files: Sequence[Path]
    :param batch_size: Rows per batch.
    :type batch_size: int
    :return: Row dicts with :data:`BAYBARS_COLUMNS`.
    :rtype: Iterator[Dict[str, Any]]
    """
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=BAYBARS_COLUMNS):
            yield from batch.to_pylist()


def crop_split(data_dir: Path, split: str, official: Sequence[str], out_dir: Path, limit: Optional[int] = None,
               workers: int = WORKERS, main_regions: frozenset = MAIN_TEXT_REGIONS) -> List[PageResult]:
    """Write the kept crops of one output split of a line release.

    :param data_dir: Folder of ``<official split>-*.parquet``.
    :type data_dir: Path
    :param split: Output split (``train`` / ``val``).
    :type split: str
    :param official: Official splits feeding it, read in this order.
    :type official: Sequence[str]
    :param out_dir: ``crops/<set>``.
    :type out_dir: Path
    :param limit: Source pages (``manuscript_name`` + ``image_name``) to read at most.
    :type limit: Optional[int]
    :param workers: Threads for decoding and writing.
    :type workers: int
    :param main_regions: Region types kept.
    :type main_regions: frozenset
    :return: One result per source page, in order of first appearance.
    :rtype: List[PageResult]
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pages: Dict[Tuple[str, str], PageResult] = {}
    jobs: List[Tuple[Tuple[str, str], Tuple[str, bytes, str]]] = []
    work = partial(write_crop, out_dir=out_dir)

    def flush(pool: ThreadPoolExecutor) -> None:
        """Write the pending crops and record their paths.

        :param pool: Thread pool.
        :type pool: ThreadPoolExecutor
        :return: None.
        :rtype: None
        """
        for (page_key, _), png in zip(jobs, pool.map(work, [job for _, job in jobs])):
            pages[page_key].paths.append(png)
        jobs.clear()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name in official:
            files = sorted(data_dir.glob(f"{name}-*.parquet"))
            for row_index, row in enumerate(parquet_rows(files)):
                manuscript, image_name = str(row["manuscript_name"]), str(row["image_name"])
                page_key = (manuscript, image_name)
                if page_key not in pages:
                    if limit is not None and len(pages) >= limit:
                        break
                    pages[page_key] = PageResult(key=f"baybars/{manuscript}/{image_name}", manuscript=manuscript,
                                                 split=split)
                page = pages[page_key]
                if row["region_type"] not in main_regions:
                    page.counts[f"region {row['region_type']}"] += 1
                    continue
                text, reason = clean_target(row["transcription"] or "")
                if reason:
                    page.counts[reason] += 1
                    continue
                page.counts[KEPT] += 1
                page.letters += arabic_letter_count(text)
                jobs.append((page_key, (crop_stem(name, manuscript, image_name, row_index), row["image"]["bytes"], text)))
                if len(jobs) >= CROP_BATCH:
                    flush(pool)
            flush(pool)
            if limit is not None and len(pages) >= limit:
                break
    for page in pages.values():
        page.skip = "" if page.counts[KEPT] else "no kept lines"
    return list(pages.values())


def build_crop_set(name: str, data_dir: Path, out_root: Path, limit: Optional[int] = None,
                   workers: int = WORKERS) -> List[PageResult]:
    """Write a line-crop set with its official split (``limit`` applies per output split).

    :param name: Set name (output folder).
    :type name: str
    :param data_dir: Folder of the release's parquet files.
    :type data_dir: Path
    :param out_root: Output root.
    :type out_root: Path
    :param limit: Source pages per output split at most.
    :type limit: Optional[int]
    :param workers: Threads.
    :type workers: int
    :return: Page results with splits.
    :rtype: List[PageResult]
    """
    out_dir = out_root / "crops" / name
    results: List[PageResult] = []
    for split, official in BAYBARS_SPLITS.items():
        results.extend(crop_split(data_dir, split, official, out_dir, limit, workers))
        logger.info("  %s %s: %d pages, %d crops", name, split, sum(1 for r in results if r.split == split),
                    sum(len(r.paths) for r in results if r.split == split))
    return results


# ----------------------------------------------------------------------------- registry, lists, stats

@dataclass(frozen=True)
class SetSpec:
    """A training set: how it is built and under which licence.

    :param kind: ``"page_xml"`` (PAGE copies) or ``"crops"`` (line images + ``.gt.txt``).
    :param licence: Licence of the source data.
    :param source_dir: Source folder relative to the source root.
    :param sources: For a PAGE set, lists its pages from the source folder.
    """
    kind: str
    licence: str
    source_dir: str
    sources: Optional[Callable[[Path], List[PageSource]]] = None


SETS: Dict[str, SetSpec] = {
    "agapet": SetSpec("page_xml", "CC BY 4.0", PAGE_SET_DIRS["agapet"], agapet_sources),
    "rasam": SetSpec("page_xml", "Apache-2.0", PAGE_SET_DIRS["rasam"], rasam_sources),
    "iskandar": SetSpec("page_xml", "Etalab Open Licence 2.0", PAGE_SET_DIRS["iskandar"], iskandar_sources),
    "baybars": SetSpec("crops", "Etalab Open Licence 2.0", BAYBARS_DIR),
}


def build_set(name: str, source_root: Path = EXTERNAL_DIR, out_root: Path = OUT_ROOT, limit: Optional[int] = None,
              workers: int = WORKERS, sets: Dict[str, SetSpec] = SETS) -> List[PageResult]:
    """Build one registered set.

    :param name: Set name in ``sets``.
    :type name: str
    :param source_root: Root of the source folders.
    :type source_root: Path
    :param out_root: Output root.
    :type out_root: Path
    :param limit: Pages per set (per output split for a crop set).
    :type limit: Optional[int]
    :param workers: Threads.
    :type workers: int
    :param sets: Registry.
    :type sets: Dict[str, SetSpec]
    :return: Page results with splits.
    :rtype: List[PageResult]
    """
    spec = sets[name]
    if spec.kind == "crops":
        return build_crop_set(name, source_root / spec.source_dir, out_root, limit, workers)
    return build_page_set(name, spec.sources(source_root / spec.source_dir), out_root, limit, workers)


def write_lists(name: str, results: Sequence[PageResult], out_root: Path) -> Dict[str, Path]:
    """Write ``lists/<set>_train.lst`` and ``<set>_val.lst``.

    :param name: Set name.
    :type name: str
    :param results: Page results with splits.
    :type results: Sequence[PageResult]
    :param out_root: Output root.
    :type out_root: Path
    :return: ``{split: list path}``.
    :rtype: Dict[str, Path]
    """
    lists_dir = out_root / "lists"
    lists_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in ("train", "val"):
        paths = [str(p) for r in results if not r.skip and r.split == split for p in r.paths]
        out[split] = lists_dir / f"{name}_{split}.lst"
        out[split].write_text("".join(f"{p}\n" for p in paths), encoding="utf-8")
    return out


def set_stats(results: Sequence[PageResult], spec: SetSpec, limit: Optional[int] = None) -> Dict[str, Any]:
    """Summary of one built set.

    :param results: Page results with splits.
    :type results: Sequence[PageResult]
    :param spec: The set's registry entry.
    :type spec: SetSpec
    :param limit: The ``--limit`` used (None for a full build).
    :type limit: Optional[int]
    :return: Pages seen / kept / skipped by reason, lines seen / kept / dropped by reason, letters kept,
        per-split pages / lines / letters, val shares, per-manuscript kept pages by split, edits.
    :rtype: Dict[str, Any]
    """
    lines: Counter = Counter()
    edits: Counter = Counter()
    for r in results:
        lines.update(r.counts)
        edits.update(r.edits)
    kept = [r for r in results if not r.skip]
    splits = {s: {"pages": sum(1 for r in kept if r.split == s),
                  "lines": sum(r.counts[KEPT] for r in kept if r.split == s),
                  "arabic_letters": sum(r.letters for r in kept if r.split == s)} for s in ("train", "val")}
    manuscripts: Dict[str, Dict[str, int]] = {}
    for r in kept:
        manuscripts.setdefault(r.manuscript, {"train": 0, "val": 0})[r.split] += 1
    return {
        "kind": spec.kind, "licence": spec.licence, "limit": limit,
        "pages_seen": len(results), "pages_kept": len(kept),
        "pages_skipped": dict(Counter(r.skip for r in results if r.skip).most_common()),
        "lines_seen": sum(lines.values()), "lines_kept": lines[KEPT],
        "lines_dropped": dict(Counter({k: v for k, v in lines.items() if k != KEPT}).most_common()),
        "arabic_letters_kept": sum(r.letters for r in kept),
        "splits": splits,
        "val_share_pages": round(splits["val"]["pages"] / len(kept), 4) if kept else 0.0,
        "val_share_lines": round(splits["val"]["lines"] / lines[KEPT], 4) if lines[KEPT] else 0.0,
        "manuscripts": dict(sorted(manuscripts.items())),
        "edits": dict(edits),
    }


def update_stats(path: Path, new: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Merge set summaries into ``stats.json`` (sets built earlier keep their entries).

    :param path: ``stats.json``.
    :type path: Path
    :param new: ``{set: summary}`` of this run.
    :type new: Dict[str, Dict[str, Any]]
    :return: The merged statistics as written.
    :rtype: Dict[str, Any]
    """
    stats = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    stats.update(new)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def build(names: Sequence[str], source_root: Path = EXTERNAL_DIR, out_root: Path = OUT_ROOT,
          limit: Optional[int] = None, workers: int = WORKERS) -> Dict[str, Any]:
    """Build sets, write their lists and merge their statistics into ``stats.json``.

    ``stats.json`` is updated right after each set's lists are written, so a crash in a later set
    leaves the lists and the statistics of the finished sets in agreement.

    :param names: Set names.
    :type names: Sequence[str]
    :param source_root: Root of the source folders.
    :type source_root: Path
    :param out_root: Output root.
    :type out_root: Path
    :param limit: Pages per set (per output split for a crop set).
    :type limit: Optional[int]
    :param workers: Threads.
    :type workers: int
    :return: The merged statistics as last written.
    :rtype: Dict[str, Any]
    """
    out_root = out_root.absolute()
    merged: Dict[str, Any] = {}
    for name in names:
        logger.info("building %s", name)
        results = build_set(name, source_root, out_root, limit, workers)
        write_lists(name, results, out_root)
        summary = set_stats(results, SETS[name], limit)
        merged = update_stats(out_root / "stats.json", {name: summary})
        logger.info("  %s: %d / %d pages, %d lines kept (val pages %s)", name, summary["pages_kept"],
                    summary["pages_seen"], summary["lines_kept"], summary["splits"]["val"]["pages"])
    return merged


def main(argv: Optional[List[str]] = None) -> None:
    """Command-line entry point.

    :param argv: Arguments (default ``sys.argv[1:]``).
    :type argv: Optional[List[str]]
    :return: None.
    :rtype: None
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sets", nargs="+", choices=list(SETS), default=list(SETS))
    parser.add_argument("--source-root", type=Path, default=EXTERNAL_DIR)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--limit", type=int, default=None, help="pages per set (per output split for baybars)")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stats = build(args.sets, args.source_root, args.out_root, args.limit, args.workers)
    print(json.dumps({name: stats[name] for name in args.sets}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
