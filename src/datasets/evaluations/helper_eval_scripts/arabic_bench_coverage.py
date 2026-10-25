# File name: arabic_bench_coverage.py
# Date: 10/24/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Which edition section(s) each Arabic-script benchmark image carries (offline, no model needed).

The benchmark scorer (``score_arabic_benchmark.py``) concatenates the answers to every image of a
document and compares them with the whole edition. An edition and the images do not always cover the
same sides: a second image may show a Hebrew text, a blank side or Arabic text nobody edited, an image
may be a second copy of another one, and an edition may hold a side no image shows. The scorer then
penalises a reader for reading text the edition does not cover and cannot separate recognition errors
from over-reading. This module writes a coverage map: for every image, the edition section(s) it
carries, or ``uncovered`` (no edited text on it), ``blank_or_faint`` (the readers found almost nothing)
or ``ambiguous``.

Edition sections are grouped into *side units*: all sections with the same side tag (``recto``,
``verso``, or ``""`` for text before any side label) form one unit, since ``split_sections`` opens a new
section at every "Recto ..." label (margins, witness clauses) and those parts are on the same side.

Images that show the same photograph (identical bytes, or a difference hash within
``max_duplicate_bits`` bits: a library image and a smaller copy of it) form one *image group*; a later
copy is recorded with ``duplicate_of`` and inherits its group's decision. Decisions are made per group.

Methods, in order of trust (the first that applies decides; recorded per image as ``method``):

1. ``side_label``: every group has one side label (an IIIF canvas label such as ``1r`` / ``1v``, or an
   image-store file name ending in ``_1r`` / ``_1v`` / ``_23a`` / ``_23b``; an unlabelled copy takes its
   group's label), no side is labelled twice, every edition unit is tagged and every unit's side is
   among the labels (one unit per labelled side). A group carries the unit tagged with its side, or
   nothing (``uncovered``) when the edition has no unit for that side.
   ``pgp_side_label``: the same for an untagged single-unit edition, with PGP's record-level ``side``
   field (``recto`` or ``verso``) naming the side the text is on.
   Labels are not used when two copies of one photograph carry different labels, or when the reads
   place a unit on a group the labels exclude (PGP's recto is not always the library's ``1r``); the
   document then falls through to the next method and its rows say ``label_conflict``.
2. ``single_image``: one image group and an edition of one side (one unit) -> covered.
3. ``reads``: the cached reads of several models of every image of the group, pooled (default: six
   fine-tuned checkpoints; never one model alone). Per group and unit, the clipped recall of the unit's
   rasm-skeleton 4-grams in the pooled read (per-gram maximum over the reads). Pooled reads are long
   (six readers, decoding loops), so the raw recall of an *unrelated* document's text is far from zero
   (median about 0.2 on the first 60 documents). Each group's recall is therefore chance-corrected
   against the group's own floor ``f``, the ``floor_quantile`` quantile of the recall of every other
   document's units in the same pooled read: ``c = max(0, (raw - f) / (1 - f))``. On ``c``:

   * fewer than ``min_pooled_letters`` script letters across all reads -> ``blank_or_faint``;
   * best unit ``c >= min_recall`` and ``>= recall_ratio`` x the runner-up -> covered by it;
   * otherwise, when exactly one unit reaches ``min_recall`` on its *own* n-grams (those of no other unit
     of the document: the recto and verso of one document share names and formulae) and every other
     unit stays under ``uncovered_max_recall`` on its own n-grams -> covered by that unit;
   * several units each ``c >= min_recall`` -> all assigned when each also holds on its own n-grams and
     is ``>= recall_ratio`` x its ``c`` on every other group: a photograph of two sides;
   * pooled read mostly Hebrew letters (``arabic_script.script_share`` < ``arabic_majority``) ->
     ``uncovered`` (a Hebrew-script side);
   * best ``c < uncovered_max_recall`` with a mostly Arabic read -> ``uncovered``, but only when every
     placeable unit of the edition was placed on another group (otherwise a failed reading and an
     unedited side cannot be told apart: ``ambiguous``);
   * otherwise ``ambiguous``.

   Units with fewer than ``min_unit_letters`` rasm letters (an address, a one-line note) are too short to
   place by recall; they are reported (``short_units``) and ignored by the decisions.

For groups decided by a label or the single-image rule, the reads decision is computed as well when
reads exist and recorded as ``reads_check`` (``agree`` / ``contradict`` / ``inconclusive`` / ``no_reads``).
Note that a reads decision depends on the readers reading the image: ``ambiguous`` collects both
unreadable covered images and uncovered images whose edition was not found elsewhere.

**Map v2** (``coverage_v2.jsonl``, :func:`decide_document_v2`, the CLI default) changes v1 as follows:

* copies are grouped whatever the rotation: difference hashes of the whole image and of the fragment crop
  (backing replaced by the paper grey, so different backings and framings compare) under 0 / 90 / 180 / 270
  degrees; a best match that is mirrored means the other side of the leaf and is never a copy. A copy
  (``role`` ``copy_of``) inherits its group's coverage and is left out of scoring; the group's reads
  evidence comes from its best-read member (a rotated copy is read as noise);
* ``blank_or_faint`` comes from the image itself (:func:`ink_measures`, :func:`ink_class`): ink density
  inside the fragment mask (backing, stitching, rulers and stickers removed) after a black-hat transform
  with a threshold relative to the paper grey, calibrated on by-eye blank and written images
  (:data:`BLANK_CALIBRATION`, :data:`TEXT_CALIBRATION`); a one-unit edition with exactly one non-blank
  image group is placed on it (``single_text_image``);
* every image gets a ``script`` that does not come from the scored models: a by-eye override, else the
  edition's script on a covered image, ``none`` on a blank one, else ``unknown``;
* reads place only units of at least ``min_place_letters`` rasm letters; while a shorter unit is unplaced a
  group the reads find empty stays ``ambiguous``;
* by-eye overrides (``logs/arabic_bench_coverage/manual_overrides_v2.jsonl``, :func:`load_overrides`) take
  precedence over every rule (method ``manual``, ``override_reason``);
* a unit on several image groups is a joined section (``joined_units``, per image ``joined_with``): all those
  images are covered by it, and the scorer compares their answers, joined in image order, with it once.

Output (``coverage_v1.jsonl`` / ``coverage_v2.jsonl`` in the benchmark directory): a ``settings`` row, then per document one
``image`` row per image followed by one ``document`` row (counts, duplicates, placed / unplaced units,
whether the coverage says the images and the edition cover the same sides, and whether that agrees with
the scorer's ``sides_match`` heuristic). Image difference hashes are cached in the report directory.

Usage::

    .venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.arabic_bench_coverage \\
        [--map-version v1|v2] [--benchmark DIR] [--tags T1 T2 ...] [--out FILE] [--report-dir DIR] \\
        [--overrides JSONL] [--compare-to V1_MAP] [--paired-ids CSV]
"""
import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.datasets.evaluations import arabic_script as ar
from src.datasets.evaluations.helper_eval_scripts import score_arabic_benchmark as sab

DEFAULT_BENCHMARK = Path("/Volumes/home/studio_offload/datasets/arabic_script_benchmark_v0")
POOL_TAGS = ("qwen3-vl-8b-heb-v23a-step1500__trained", "qwen3-vl-8b-heb-v24a-step1000__trained",
             "qwen3-vl-8b-heb-v24a-step2000__trained", "qwen3-vl-8b-heb-v25a-step600__trained",
             "qwen3-vl-8b-heb-v25a-step900__trained", "qwen3-vl-8b-heb-v25a-step1200__trained")
COVERED, UNCOVERED, AMBIGUOUS, BLANK = "covered", "uncovered", "ambiguous", "blank_or_faint"
SIDE_LABEL, PGP_SIDE_LABEL, SINGLE_IMAGE, READS, NO_READS = "side_label", "pgp_side_label", "single_image", "reads", "no_reads"

# rasm classes: every dotted letter folds to its undotted shape; hamza seats, alef forms and Persian letters fold too
_RASM = str.maketrans({
    "ا": "ا", "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ء": "", "ٲ": "ا", "ٳ": "ا",
    "ب": "ٮ", "ت": "ٮ", "ث": "ٮ", "ن": "ٮ", "ي": "ٮ", "ى": "ٮ", "ئ": "ٮ", "ی": "ٮ", "پ": "ٮ", "ے": "ٮ", "ٹ": "ٮ",
    "ج": "ح", "ح": "ح", "خ": "ح", "چ": "ح",
    "د": "د", "ذ": "د", "ر": "ر", "ز": "ر", "ژ": "ر", "س": "س", "ش": "س", "ص": "ص", "ض": "ص",
    "ط": "ط", "ظ": "ط", "ع": "ع", "غ": "ع", "ف": "ف", "ق": "ف", "ڤ": "ف", "ك": "ك", "ک": "ك", "گ": "ك",
    "ل": "ل", "م": "م", "ه": "ه", "ة": "ه", "ۀ": "ه", "ہ": "ه", "و": "و", "ؤ": "و",
})
_RASM_LETTER = re.compile("[اٮحدرسصطعفكلمهو]")
_FILE_SIDE = re.compile(r"_\d+([rvab])\.[A-Za-z]+$")
_FILE_SIDE_NAMES = {"r": "recto", "v": "verso", "a": "recto", "b": "verso"}     # Bodleian folio a / b = recto / verso


@dataclass(frozen=True)
class Settings:
    """Thresholds of the coverage decisions.

    :param ngram: n-gram length on the rasm skeleton.
    :param min_recall: Smallest corrected recall that places a unit on an image group.
    :param recall_ratio: The placed unit's corrected recall must be this many times the runner-up's.
    :param uncovered_max_recall: A group whose best corrected recall is below this (and whose read is
        mostly Arabic) carries no edited text, provided the edition is placed elsewhere.
    :param min_pooled_letters: Fewer script letters than this across all reads -> blank or faint.
    :param arabic_majority: A pooled read with a smaller Arabic share of its script letters is Hebrew.
    :param floor_quantile: Quantile of the wrong-document recalls that is a group's chance floor.
    :param min_unit_letters: Units with fewer rasm letters are too short to place by recall.
    :param min_readers: Fewer distinct readers with a read of a group -> no reads-based decision.
    :param max_duplicate_bits: Images of one document whose difference hashes differ in at most this
        many bits are copies of one photograph.
    :param hash_size: Side of the difference-hash grid (``hash_size ** 2`` bits).
    :param min_place_letters: (v2) Units with fewer rasm letters are never placed by reads (only by side
        labels, the single-image rules or a manual override); ``min_unit_letters`` still selects the
        chance floor's reference units.
    :param mirror_margin: (v2) A copy's rotation distance must be at least this many bits below its best
        mirrored distance (a mirrored match is the other side of the leaf).
    """
    ngram: int = 4
    min_recall: float = 0.08
    recall_ratio: float = 2.0
    uncovered_max_recall: float = 0.03
    min_pooled_letters: int = 40
    arabic_majority: float = 0.5
    floor_quantile: float = 0.95
    min_unit_letters: int = 40
    min_readers: int = 2
    max_duplicate_bits: int = 24
    hash_size: int = 16
    min_place_letters: int = 100
    mirror_margin: int = 10


# --- text and n-grams ------------------------------------------------------------------------------------------------

def rasm_skeleton(text: str) -> str:
    """The rasm of a text: its Arabic letters folded to their undotted shapes, everything else dropped.

    :param text: Any text (an edition section or a model read).
    :type text: str
    :return: Rasm letters in order (no spaces).
    :rtype: str
    """
    folded = ar.normalize_arabic(unicodedata.normalize("NFKC", text)).translate(_RASM)
    return "".join(_RASM_LETTER.findall(folded))


def ngram_counts(letters: str, n: int) -> Counter:
    """n-grams of a letter string.

    :param letters: Letters (no spaces).
    :type letters: str
    :param n: n-gram length.
    :type n: int
    :return: Counter of n-grams (empty for a string shorter than ``n``).
    :rtype: Counter
    """
    return Counter(letters[i:i + n] for i in range(len(letters) - n + 1))


def pooled_ngrams(texts: Iterable[str], n: int) -> Counter:
    """Rasm n-grams of several reads of one image, pooled as a multiset union (per-gram maximum).

    A gram counts as often as the read that has it most often, so six readers that each find a formula
    once do not count it six times.

    :param texts: Reads of the image (or of the copies of one photograph).
    :type texts: Iterable[str]
    :param n: n-gram length.
    :type n: int
    :return: Pooled n-gram counts.
    :rtype: Counter
    """
    pooled: Counter = Counter()
    for text in texts:
        for gram, count in ngram_counts(rasm_skeleton(text), n).items():
            if count > pooled[gram]:
                pooled[gram] = count
    return pooled


def clipped_recall(reference: Mapping[str, int], hypothesis: Mapping[str, int]) -> float:
    """Share of the reference's n-grams found in the hypothesis (counts clipped).

    :param reference: Reference n-gram counts (an edition unit).
    :type reference: Mapping[str, int]
    :param hypothesis: Hypothesis n-gram counts (a pooled read).
    :type hypothesis: Mapping[str, int]
    :return: Recall in [0, 1]; 0.0 for an empty reference.
    :rtype: float
    """
    total = sum(reference.values())
    if not total:
        return 0.0
    return sum(min(count, hypothesis.get(gram, 0)) for gram, count in reference.items()) / total


def exclusive_ngrams(units: Sequence[Mapping[str, int]], index: int) -> Counter:
    """The n-grams of one unit that no other unit of the document has.

    :param units: n-gram counts of every unit of a document.
    :type units: Sequence[Mapping[str, int]]
    :param index: The unit.
    :type index: int
    :return: Counts of the unit's own n-grams.
    :rtype: Counter
    """
    shared = set().union(*(set(u) for k, u in enumerate(units) if k != index))
    return Counter({gram: count for gram, count in units[index].items() if gram not in shared})


def quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile.

    :param values: Values (any order).
    :type values: Sequence[float]
    :param q: Quantile in [0, 1].
    :type q: float
    :return: The quantile; 0.0 for no values.
    :rtype: float
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def chance_corrected(raw: float, floor: float) -> float:
    """Recall above a chance floor, rescaled to [0, 1].

    :param raw: Raw clipped recall.
    :type raw: float
    :param floor: The chance floor (recall of unrelated documents' text in the same read).
    :type floor: float
    :return: ``max(0, (raw - floor) / (1 - floor))``; 0.0 when the floor is 1.
    :rtype: float
    """
    if floor >= 1.0:
        return 0.0
    return max(0.0, (raw - floor) / (1.0 - floor))


# --- images -----------------------------------------------------------------------------------------------------------

def side_from_filename(name: str) -> str:
    """Side named by an image-store file name (``..._1r.jpg``, ``..._23b.jpg``).

    :param name: File name or URL.
    :type name: str
    :return: ``"recto"``, ``"verso"`` or ``""``.
    :rtype: str
    """
    match = _FILE_SIDE.search(name.rsplit("/", 1)[-1])
    return _FILE_SIDE_NAMES[match.group(1)] if match else ""


def image_sides(record: Mapping[str, Any], manifest_urls: Mapping[str, str]) -> List[Tuple[str, str]]:
    """Side label of every image of a document and where it comes from.

    :param record: Benchmark record.
    :type record: Mapping[str, Any]
    :param manifest_urls: ``{image file: source URL}`` from ``images_manifest.json``.
    :type manifest_urls: Mapping[str, str]
    :return: ``(side, source)`` per image in image order; source ``canvas_label``, ``filename`` or ``""``.
    :rtype: List[Tuple[str, str]]
    """
    out = []
    for image in record["images"]:
        if image.get("side"):
            out.append((image["side"], "canvas_label"))
            continue
        side = side_from_filename(manifest_urls.get(image["file"], ""))
        out.append((side, "filename" if side else ""))
    return out


def dhash(path: Path, size: int = 16) -> int:
    """Difference hash of an image (grey levels of neighbouring cells compared, row by row).

    :param path: Image file.
    :type path: Path
    :param size: Grid side; the hash has ``size ** 2`` bits.
    :type size: int
    :return: The hash as an integer.
    :rtype: int
    """
    from PIL import Image

    with Image.open(path) as image:
        image.draft("L", (size * 16, size * 16))       # JPEG: decode at a reduced scale
        grey = image.convert("L").resize((size + 1, size), Image.LANCZOS)
    pixels = list(grey.getdata())
    bits = 0
    for row in range(size):
        for col in range(size):
            left, right = pixels[row * (size + 1) + col], pixels[row * (size + 1) + col + 1]
            bits = (bits << 1) | int(left > right)
    return bits


# --- image analysis (v2): fragment mask, ink, rotation / mirror hashes ---------------------------------------------------

TRANSFORMS = ("0", "90", "180", "270", "m0", "m90", "m180", "m270")    # "m" = mirrored left-right, then rotated
ROTATIONS = TRANSFORMS[:4]


@dataclass(frozen=True)
class ImageSettings:
    """Thresholds of the image analysis (pixel sizes refer to the image scaled to ``long_side``).

    :param long_side: Longest side of the analysed copy of an image.
    :param border_band: Share of the shorter side sampled as the image border (the backing's colours).
    :param border_clusters: k-means clusters of the border colours.
    :param min_cluster_share: Smallest share of the border a backing colour must hold.
    :param warm_b: Border clusters with an OpenCV Lab ``b`` at or above this are paper, not backing.
    :param backing_distance: Lab distance (lightness weighted 1/3) under which a pixel has a backing colour.
    :param label_b: Bright pixels at least this yellow (Lab ``b``) are a shelf-mark sticker.
    :param open_px: Opening diameter of the fragment mask (removes stitching, grid lines).
    :param close_px: Closing diameter of the fragment mask (absorbs ink strokes).
    :param min_piece_share: Mask components smaller than this share of the largest are dropped (labels, rulers).
    :param erode_px: The mask is eroded by this diameter before ink is counted (torn edges, shadows).
    :param min_inner_share: An eroded mask smaller than this share of the image: no ink measure (mask failed).
    :param blur_sigma: Gaussian blur before the black-hat (pixel noise).
    :param blackhat_px: Black-hat diameter (strokes thinner than this respond).
    :param ink_contrast: Ink = black-hat response above this share of the paper's grey level.
    :param min_ink_component: Ink components smaller than this (pixels) are specks.
    :param line_kernel: Width of the horizontal (or vertical) closing that joins letters into a line.
    :param min_line_px: A joined component at least this long is a line of writing (recorded, not used by the rule).
    :param dark_backing_l: The loose mask drops pixels darker than this (OpenCV Lab ``L``: black cloth, shadows).
    :param blue_backing_b: The loose mask drops pixels bluer than this (Lab ``b``: Cambridge blue card).
    :param blank_max_permille: An image with less ink (per mille of the eroded fragment mask) ...
    :param blank_max_loose_permille: ... and less ink than this inside the loose mask (everything that is not
        dark or blue backing: catches a grey page the fragment mask took for backing) is ``blank_or_faint``.
    :param hash_size: Side of the difference-hash grid.
    """
    long_side: int = 1000
    border_band: float = 0.03
    border_clusters: int = 5
    min_cluster_share: float = 0.05
    warm_b: float = 134.0
    backing_distance: float = 14.0
    label_b: float = 163.0
    open_px: int = 11
    close_px: int = 25
    min_piece_share: float = 0.3
    erode_px: int = 21
    min_inner_share: float = 0.05
    blur_sigma: float = 1.0
    blackhat_px: int = 15
    ink_contrast: float = 0.22
    min_ink_component: int = 15
    line_kernel: int = 31
    min_line_px: int = 110
    dark_backing_l: float = 50.0
    blue_backing_b: float = 118.0
    blank_max_permille: float = 8.0
    blank_max_loose_permille: float = 16.0
    hash_size: int = 16


def _disk(diameter: int) -> Any:
    """Elliptical structuring element.

    :param diameter: Diameter in pixels.
    :type diameter: int
    :return: OpenCV kernel.
    :rtype: numpy.ndarray
    """
    import cv2

    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))


def load_rgb(path: Path, long_side: int) -> Any:
    """An image as an RGB array scaled so that its longer side is at most ``long_side``.

    :param path: Image file.
    :type path: Path
    :param long_side: Longest side after scaling.
    :type long_side: int
    :return: ``uint8`` array ``(h, w, 3)``.
    :rtype: numpy.ndarray
    """
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        image.draft("RGB", (long_side * 2, long_side * 2))     # JPEG: decode at a reduced scale
        rgb = image.convert("RGB")
    rgb.thumbnail((long_side, long_side), Image.LANCZOS)
    return np.asarray(rgb)


def backing_colours(lab: Any, settings: ImageSettings) -> List[Any]:
    """Colours of the backing / mount, from the image border.

    The border is clustered with k-means; clusters holding at least ``min_cluster_share`` of the border
    and not warm (paper is yellow-brown; blue card, grey or white board and black cloth are not) are the
    backing. A fragment that touches the border therefore does not turn into backing.

    :param lab: OpenCV Lab image (``float32``).
    :type lab: numpy.ndarray
    :param settings: Image thresholds.
    :type settings: ImageSettings
    :return: Lab centres of the backing colours (may be empty).
    :rtype: List[numpy.ndarray]
    """
    import cv2
    import numpy as np

    h, w = lab.shape[:2]
    band = max(2, int(round(min(h, w) * settings.border_band)))
    border = np.concatenate([lab[:band].reshape(-1, 3), lab[-band:].reshape(-1, 3),
                             lab[:, :band].reshape(-1, 3), lab[:, -band:].reshape(-1, 3)]).astype(np.float32)
    k = min(settings.border_clusters, len(border))
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centres = cv2.kmeans(border, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    shares = np.bincount(labels.ravel(), minlength=k) / len(labels)
    return [centres[i] for i in range(k) if shares[i] >= settings.min_cluster_share and centres[i][2] < settings.warm_b]


def fragment_mask(rgb: Any, settings: ImageSettings) -> Any:
    """Pixels of the fragment (paper or parchment), without backing, stitching, rulers and labels.

    :param rgb: RGB image.
    :type rgb: numpy.ndarray
    :param settings: Image thresholds.
    :type settings: ImageSettings
    :return: Boolean mask.
    :rtype: numpy.ndarray
    """
    import cv2
    import numpy as np

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    backing = backing_colours(lab, settings)
    if backing:
        weight = np.array([1 / 3, 1.0, 1.0], np.float32)
        distance = np.min(np.stack([np.sqrt((((lab - c) * weight) ** 2).sum(axis=2)) for c in backing]), axis=0)
        candidate = distance > settings.backing_distance
    else:
        candidate = np.ones(lab.shape[:2], bool)
    light, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    sticker = ((b >= settings.label_b) & (light >= 140)) | ((light >= 200) & (np.abs(b - 128) <= 8) & (np.abs(a - 128) <= 6))
    candidate = (candidate & ~sticker).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, _disk(settings.open_px))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, _disk(settings.close_px))
    count, components, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    if count <= 1:
        return candidate.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, area in enumerate(areas) if area >= settings.min_piece_share * areas.max()]
    return np.isin(components, keep)


def loose_mask(rgb: Any, settings: ImageSettings) -> Any:
    """Everything that is not dark or blue backing nor a yellow sticker (a looser fragment mask).

    :param rgb: RGB image.
    :type rgb: numpy.ndarray
    :param settings: Image thresholds.
    :type settings: ImageSettings
    :return: Boolean mask.
    :rtype: numpy.ndarray
    """
    import cv2
    import numpy as np

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    light, b = lab[..., 0], lab[..., 2]
    keep = (light >= settings.dark_backing_l) & (b >= settings.blue_backing_b) & ~((b >= settings.label_b) & (light >= 140))
    keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_OPEN, _disk(settings.open_px))
    return cv2.morphologyEx(keep, cv2.MORPH_CLOSE, _disk(settings.close_px)).astype(bool)


def line_components(ink: Any, settings: ImageSettings) -> int:
    """Line-like groups of ink: letters joined by a closing along the rows (or the columns).

    :param ink: Boolean ink mask.
    :type ink: numpy.ndarray
    :param settings: Image thresholds.
    :type settings: ImageSettings
    :return: The larger of the horizontal and the vertical count of joined components at least
        ``min_line_px`` long and at least 6 pixels thick (a fold or a hair is thinner).
    :rtype: int
    """
    import cv2
    import numpy as np

    counts = []
    for kernel in ((settings.line_kernel, 5), (5, settings.line_kernel)):
        joined = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, kernel))
        _, _, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
        along, across = (cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT) if kernel[0] > kernel[1] else (cv2.CC_STAT_HEIGHT, cv2.CC_STAT_WIDTH)
        counts.append(int(((stats[1:, along] >= settings.min_line_px) & (stats[1:, across] >= 6)).sum()))
    return max(counts)


def ink_measures(rgb: Any, mask: Any, settings: ImageSettings) -> Tuple[Dict[str, Any], Optional[Any]]:
    """How much writing an image shows inside its fragment mask.

    The mask is eroded (torn edges and shadows are dark), the grey image is lightly blurred, the area
    outside the mask is filled with the paper's median grey, and a black-hat transform picks out strokes
    thinner than ``blackhat_px``; a pixel is ink when that response exceeds ``ink_contrast`` times the
    paper grey (a threshold relative to the paper, so dark and light papers are treated alike).

    :param rgb: RGB image.
    :type rgb: numpy.ndarray
    :param mask: :func:`fragment_mask`.
    :type mask: numpy.ndarray
    :param settings: Image thresholds.
    :type settings: ImageSettings
    :return: ``({mask_ok, inner_share, paper_grey, ink_permille, ink_components, line_components}, ink mask)``;
        with ``mask_ok`` False (eroded mask under ``min_inner_share`` of the image) the ink fields are None
        and the ink mask is None.
    :rtype: Tuple[Dict[str, Any], Optional[numpy.ndarray]]
    """
    import cv2
    import numpy as np

    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    inner = cv2.erode(mask.astype(np.uint8), _disk(settings.erode_px)).astype(bool)
    inner_share = float(inner.mean())
    if inner_share < settings.min_inner_share:
        return {"mask_ok": False, "inner_share": round(inner_share, 4), "paper_grey": None, "ink_permille": None,
                "ink_components": None, "line_components": None}, None
    paper = float(np.median(grey[inner]))
    filled = np.where(mask, grey, paper).astype(np.uint8)
    if settings.blur_sigma:
        filled = cv2.GaussianBlur(filled, (0, 0), settings.blur_sigma)
    response = cv2.morphologyEx(filled, cv2.MORPH_BLACKHAT, _disk(settings.blackhat_px)).astype(np.float32)
    ink = (response > settings.ink_contrast * max(paper, 1.0)) & inner
    count, components, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    speck = np.zeros(count, bool)
    speck[1:] = areas < settings.min_ink_component
    ink &= ~speck[components]
    return {"mask_ok": True, "inner_share": round(inner_share, 4), "paper_grey": paper,
            "ink_permille": round(1000.0 * float(ink.sum()) / float(inner.sum()), 3),
            "ink_components": int((areas >= settings.min_ink_component).sum()),
            "line_components": line_components(ink, settings)}, ink


def transform_hashes(grey: Any, size: int) -> Dict[str, int]:
    """Difference hash of a grey image under the eight rotations / mirrorings.

    :param grey: PIL image in mode ``L``.
    :type grey: PIL.Image.Image
    :param size: Hash grid side (``size ** 2`` bits).
    :type size: int
    :return: ``{transform: hash}`` for :data:`TRANSFORMS`; a hash compares the image under that transform.
    :rtype: Dict[str, int]
    """
    import numpy as np
    from PIL import Image

    turns = {"0": None, "90": Image.Transpose.ROTATE_90, "180": Image.Transpose.ROTATE_180, "270": Image.Transpose.ROTATE_270}
    out = {}
    for name in TRANSFORMS:
        image = grey.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if name.startswith("m") else grey
        turn = turns[name.lstrip("m")]
        if turn is not None:
            image = image.transpose(turn)
        pixels = np.asarray(image.resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
        bits = 0
        for bit in (pixels[:, :-1] > pixels[:, 1:]).ravel():
            bits = (bits << 1) | int(bit)
        out[name] = bits
    return out


def analyse_image(path: Path, settings: ImageSettings) -> Dict[str, Any]:
    """Everything the coverage needs from one image's pixels.

    :param path: Image file.
    :type path: Path
    :param settings: Image thresholds.
    :type settings: ImageSettings
    :return: ``hashes`` (whole image, :func:`transform_hashes`, hex strings), ``crop_hashes`` (the same on
        the fragment's bounding box with the backing replaced by the paper grey, so two photographs on
        different backings or with different framing compare; None when the mask failed) and ``ink``
        (:func:`ink_measures` plus ``loose_ink_permille``, the ink inside :func:`loose_mask`).
    :rtype: Dict[str, Any]
    """
    import cv2
    import numpy as np
    from PIL import Image

    rgb = load_rgb(path, settings.long_side)
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = fragment_mask(rgb, settings)
    ink, _ = ink_measures(rgb, mask, settings)
    loose, _ = ink_measures(rgb, loose_mask(rgb, settings), settings)
    ink["loose_ink_permille"] = loose["ink_permille"]
    crop = None
    if ink["mask_ok"]:
        rows, cols = np.nonzero(mask)
        paper = np.median(grey[mask])
        filled = np.where(mask, grey, paper).astype(np.uint8)[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
        crop = {k: format(v, "x") for k, v in transform_hashes(Image.fromarray(filled), settings.hash_size).items()}
    whole = {k: format(v, "x") for k, v in transform_hashes(Image.fromarray(grey), settings.hash_size).items()}
    return {"hashes": whole, "crop_hashes": crop, "ink": ink}


def image_analyses(benchmark: Path, files: Iterable[str], cache: Optional[Path], settings: ImageSettings) -> Dict[str, Dict[str, Any]]:
    """:func:`analyse_image` of every image, read from or added to a JSON cache keyed by the settings.

    :param benchmark: Benchmark directory (images under ``images/``).
    :type benchmark: Path
    :param files: Image file names.
    :type files: Iterable[str]
    :param cache: Cache file (None: no cache).
    :type cache: Optional[Path]
    :param settings: Image thresholds (a cache made with other settings is ignored).
    :type settings: ImageSettings
    :return: ``{file: analysis}``.
    :rtype: Dict[str, Dict[str, Any]]
    """
    key = asdict(settings)
    known: Dict[str, Dict[str, Any]] = {}
    if cache is not None and cache.exists():
        stored = json.loads(cache.read_text(encoding="utf-8"))
        if stored.get("settings") == key:
            known = stored["images"]
    missing = [f for f in files if f not in known]
    for name in missing:
        known[name] = analyse_image(benchmark / "images" / name, settings)
    if cache is not None and missing:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"settings": key, "images": dict(sorted(known.items()))}), encoding="utf-8")
    return known


def hash_distances(a: Mapping[str, str], b: Mapping[str, str]) -> Tuple[int, int, str]:
    """Distances between image ``a`` as it is and image ``b`` under every transform.

    :param a: :func:`analyse_image` hashes of the first image (hex strings).
    :type a: Mapping[str, str]
    :param b: Same for the second image.
    :type b: Mapping[str, str]
    :return: ``(best rotation distance, best mirrored distance, best transform)``.
    :rtype: Tuple[int, int, str]
    """
    base = int(a["0"], 16)
    distance = {t: bin(base ^ int(b[t], 16)).count("1") for t in TRANSFORMS}
    return (min(distance[t] for t in ROTATIONS), min(distance[t] for t in TRANSFORMS if t.startswith("m")),
            min(distance, key=distance.get))


def is_copy(a: Mapping[str, Any], b: Mapping[str, Any], max_bits: int, mirror_margin: int) -> Tuple[bool, Dict[str, Any]]:
    """Whether two images of a document show the same side (rotation-invariant), with the evidence.

    Two hashes are compared: the whole image and the fragment crop. A pair is a copy when one of them is
    within ``max_bits`` under a pure rotation (0 / 90 / 180 / 270) and that rotation distance is at least
    ``mirror_margin`` bits below the best mirrored distance of the same hash. A pair whose best match is
    mirrored is the other side of the same leaf (its outline is the mirror image), never a copy.

    :param a: :func:`analyse_image` of the first image.
    :type a: Mapping[str, Any]
    :param b: :func:`analyse_image` of the second image.
    :type b: Mapping[str, Any]
    :param max_bits: Largest rotation distance of a copy.
    :type max_bits: int
    :param mirror_margin: Required gap between the mirrored and the rotation distance.
    :type mirror_margin: int
    :return: ``(copy, {kind: [rotation, mirrored, transform]})`` for ``whole`` and (when both have one) ``crop``.
    :rtype: Tuple[bool, Dict[str, Any]]
    """
    evidence: Dict[str, Any] = {}
    copy = False
    for kind, field in (("whole", "hashes"), ("crop", "crop_hashes")):
        if not a.get(field) or not b.get(field):
            continue
        rotation, mirrored, transform = hash_distances(a[field], b[field])
        evidence[kind] = [rotation, mirrored, transform]
        copy = copy or (rotation <= max_bits and rotation + mirror_margin <= mirrored)
    return copy, evidence


def ink_class(ink: Mapping[str, Any], settings: ImageSettings) -> str:
    """Whether an image shows writing, from :func:`ink_measures`.

    :param ink: Ink measures.
    :type ink: Mapping[str, Any]
    :param settings: Image thresholds.
    :type settings: ImageSettings
    :return: ``"blank_or_faint"`` (ink under ``blank_max_permille`` in the fragment mask and under
        ``blank_max_loose_permille`` in the loose mask), ``"text"`` or ``"unknown"`` (the fragment mask
        failed, its paper is implausibly dark, or the two masks disagree: little ink on the fragment, much
        ink elsewhere).
    :rtype: str
    """
    if not ink.get("mask_ok") or ink["paper_grey"] < settings.dark_backing_l + 10:
        return "unknown"
    if ink["ink_permille"] >= settings.blank_max_permille:
        return "text"
    loose = ink.get("loose_ink_permille")
    if loose is not None and loose >= settings.blank_max_loose_permille:
        return "unknown"
    return BLANK


def duplicate_groups(hashes: Sequence[Optional[int]], digests: Sequence[Optional[str]], max_bits: int) -> List[List[int]]:
    """Group a document's images that show the same photograph.

    :param hashes: Difference hash per image (None: unknown).
    :type hashes: Sequence[Optional[int]]
    :param digests: Content digest per image (None: unknown); equal digests are always one group.
    :type digests: Sequence[Optional[str]]
    :param max_bits: Largest hash distance between copies.
    :type max_bits: int
    :return: Groups of image positions in image order; the first member is the group's representative.
    :rtype: List[List[int]]
    """
    groups: List[List[int]] = []
    for j in range(len(hashes)):
        home = None
        for group in groups:
            for i in group:
                same_bytes = digests[i] is not None and digests[i] == digests[j]
                close = hashes[i] is not None and hashes[j] is not None and bin(hashes[i] ^ hashes[j]).count("1") <= max_bits
                if same_bytes or close:
                    home = group
                    break
            if home is not None:
                break
        if home is None:
            groups.append([j])
        else:
            home.append(j)
    return groups


def group_sides(groups: Sequence[Sequence[int]], sides: Sequence[str]) -> Tuple[List[str], bool]:
    """Side label of every image group.

    :param groups: :func:`duplicate_groups`.
    :type groups: Sequence[Sequence[int]]
    :param sides: Side label per image (``""`` when unlabelled).
    :type sides: Sequence[str]
    :return: ``(label per group, conflict)``; a group's label is the one label its members carry
        (``""`` if none); ``conflict`` is True when copies of one photograph carry different labels.
    :rtype: Tuple[List[str], bool]
    """
    labels, conflict = [], False
    for group in groups:
        found = {sides[i] for i in group if sides[i]}
        conflict = conflict or len(found) > 1
        labels.append(found.pop() if len(found) == 1 else "")
    return labels, conflict


# --- edition ----------------------------------------------------------------------------------------------------------

def side_units(sections: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Group edition sections by side tag.

    :param sections: ``gt.sections`` of a record (``[{side, lines}]``).
    :type sections: Sequence[Mapping[str, Any]]
    :return: ``[{side, sections: [index, ...], text}]`` in order of first appearance.
    :rtype: List[Dict[str, Any]]
    """
    units: Dict[str, Dict[str, Any]] = {}
    for index, section in enumerate(sections):
        unit = units.setdefault(section["side"], {"side": section["side"], "sections": [], "lines": []})
        unit["sections"].append(index)
        unit["lines"].extend(section["lines"])
    return [{"side": u["side"], "sections": u["sections"], "text": "\n".join(u["lines"])} for u in units.values()]


def label_assignment(units: Sequence[Mapping[str, Any]], sides: Sequence[str],
                     pgp_side: str = "") -> Optional[Tuple[str, List[List[int]]]]:
    """Units per image group from side labels (method 1), or None when the labels do not decide.

    :param units: :func:`side_units` of the edition.
    :type units: Sequence[Mapping[str, Any]]
    :param sides: Side label per image group (``""`` when unlabelled).
    :type sides: Sequence[str]
    :param pgp_side: PGP's record-level side field (used for an untagged single-unit edition).
    :type pgp_side: str
    :return: ``(method, [unit indices per group])`` with method ``side_label`` or ``pgp_side_label``;
        a group whose side has no unit gets an empty list. None when a group is unlabelled, a side is
        labelled twice, or a unit's side is not among the labels.
    :rtype: Optional[Tuple[str, List[List[int]]]]
    """
    if not sides or any(s not in ("recto", "verso") for s in sides) or len(set(sides)) != len(sides):
        return None
    tags = [u["side"] for u in units]
    if tags and all(tags) and set(tags) <= set(sides):
        return SIDE_LABEL, [[k for k, tag in enumerate(tags) if tag == side] for side in sides]
    if tags == [""] and pgp_side in sides:
        return PGP_SIDE_LABEL, [[0] if side == pgp_side else [] for side in sides]
    return None


# --- reads ------------------------------------------------------------------------------------------------------------

def read_stats(texts: Sequence[str]) -> Dict[str, Any]:
    """Letters and script of a group's pooled reads.

    :param texts: The reads.
    :type texts: Sequence[str]
    :return: ``read_letters`` (Arabic plus Hebrew letters summed over the reads), ``arabic_share``
        (``arabic_script.script_share`` of the reads together) and ``hebrew_reads`` (reads with at
        least 20 script letters, most of them Hebrew).
    :rtype: Dict[str, Any]
    """
    arabic, hebrew, share = ar.script_share("\n".join(texts))
    hebrew_reads = 0
    for text in texts:
        a, h, s = ar.script_share(text)
        hebrew_reads += int(a + h >= 20 and s < 0.5)
    return {"read_letters": arabic + hebrew, "arabic_share": round(share, 4), "hebrew_reads": hebrew_reads}


def reads_decisions(corrected: Sequence[Sequence[float]], exclusive: Sequence[Sequence[float]],
                    stats: Sequence[Mapping[str, Any]], placeable: Sequence[bool], settings: Settings,
                    short_units_block_uncovered: bool = False) -> List[Dict[str, Any]]:
    """Reads-based decision for every image group of a document (method 3).

    :param corrected: ``corrected[g][u]``: chance-corrected recall of unit ``u`` on group ``g``.
    :type corrected: Sequence[Sequence[float]]
    :param exclusive: Same on each unit's own n-grams (:func:`exclusive_ngrams`).
    :type exclusive: Sequence[Sequence[float]]
    :param stats: :func:`read_stats` per group.
    :type stats: Sequence[Mapping[str, Any]]
    :param placeable: Per unit, whether it is long enough to place by recall.
    :type placeable: Sequence[bool]
    :param settings: Thresholds.
    :type settings: Settings
    :param short_units_block_uncovered: (v2) A group below the floor is ``ambiguous``, not ``uncovered``,
        while a unit too short to place by reads exists (it may be on that group).
    :type short_units_block_uncovered: bool
    :return: Per group ``{status, units, reason}``.
    :rtype: List[Dict[str, Any]]
    """
    n_groups = len(corrected)
    candidates = [u for u, ok in enumerate(placeable) if ok]
    decisions: List[Dict[str, Any]] = []
    for g in range(n_groups):
        if stats[g]["read_letters"] < settings.min_pooled_letters:
            decisions.append({"status": BLANK, "units": [], "reason": "pooled reads under the letter minimum"})
            continue
        ranked = sorted(candidates, key=lambda u: -corrected[g][u])
        best = corrected[g][ranked[0]] if ranked else 0.0
        runner_up = corrected[g][ranked[1]] if len(ranked) > 1 else 0.0
        if ranked and best >= settings.min_recall and best >= settings.recall_ratio * runner_up:
            decisions.append({"status": COVERED, "units": [ranked[0]], "reason": "one unit above the floor"})
            continue
        strong = [u for u in ranked if corrected[g][u] >= settings.min_recall]
        evidenced = [u for u in strong if exclusive[g][u] >= settings.min_recall]
        if len(evidenced) == 1 and all(exclusive[g][u] < settings.uncovered_max_recall for u in candidates if u != evidenced[0]):
            decisions.append({"status": COVERED, "units": evidenced, "reason": "one unit above the floor on its own n-grams"})
            continue
        if len(strong) > 1 and all(exclusive[g][u] >= settings.min_recall
                                   and all(corrected[g][u] >= settings.recall_ratio * corrected[h][u] for h in range(n_groups) if h != g)
                                   for u in strong):
            decisions.append({"status": COVERED, "units": sorted(strong), "reason": "several units, each on its own n-grams"})
            continue
        if stats[g]["arabic_share"] < settings.arabic_majority:
            decisions.append({"status": UNCOVERED, "units": [], "reason": "hebrew-script read"})
            continue
        if best < settings.uncovered_max_recall:
            decisions.append({"status": UNCOVERED, "units": [], "reason": "below floor", "_needs_guard": True})
            continue
        decisions.append({"status": AMBIGUOUS, "units": [],
                          "reason": "between thresholds" if best < settings.min_recall else "no unit clears the ratio"})
    placed = {u for d in decisions if d["status"] == COVERED for u in d["units"]}
    short = [u for u, ok in enumerate(placeable) if not ok]
    for d in decisions:
        if d.pop("_needs_guard", False):
            if not candidates or not set(candidates) <= placed:
                d.update(status=AMBIGUOUS, reason="below floor, but the edition is not placed on another image (unread or uncovered)")
            elif short and short_units_block_uncovered:
                d.update(status=AMBIGUOUS, reason=f"below floor, but units {short} are too short to place by reads and may be on this image")
            elif short:
                d["reason"] = f"below floor (units {short} are too short to place and may be on this image)"
    return decisions


def reads_check(status: str, units: Sequence[int], reads: Optional[Mapping[str, Any]]) -> str:
    """Whether the reads confirm a label or single-image decision.

    :param status: Status from the label or single-image rule.
    :type status: str
    :param units: Units from that rule.
    :type units: Sequence[int]
    :param reads: The reads decision for the group (None: no reads).
    :type reads: Optional[Mapping[str, Any]]
    :return: ``no_reads``, ``inconclusive`` (reads ambiguous or blank), ``agree`` (same status, and the
        reads' units among the rule's) or ``contradict``.
    :rtype: str
    """
    if reads is None:
        return "no_reads"
    if reads["status"] in (AMBIGUOUS, BLANK):
        return "inconclusive"
    if reads["status"] == status and set(reads["units"]) <= set(units):
        return "agree"
    return "contradict"


def labels_contradicted(assigned: Sequence[Sequence[int]], reads: Optional[Sequence[Mapping[str, Any]]]) -> bool:
    """Whether the reads place a unit on a group the labels exclude it from.

    :param assigned: Units per group from :func:`label_assignment`.
    :type assigned: Sequence[Sequence[int]]
    :param reads: Reads decision per group (None: no reads).
    :type reads: Optional[Sequence[Mapping[str, Any]]]
    :return: True when some group is covered by the reads with a unit its label does not give it.
    :rtype: bool
    """
    if reads is None:
        return False
    return any(d["status"] == COVERED and not set(d["units"]) <= set(units) for d, units in zip(reads, assigned))


def coverage_match(statuses: Sequence[str], placed: Iterable[int], placeable: Sequence[bool]) -> Optional[bool]:
    """Whether the coverage says the image groups and the edition cover the same sides.

    :param statuses: Status per image group.
    :type statuses: Sequence[str]
    :param placed: Units placed on some group.
    :type placed: Iterable[int]
    :param placeable: Per unit, whether it is long enough to count.
    :type placeable: Sequence[bool]
    :return: False when a group carries no edited text or (with every group decided) a placeable unit
        is on no group; True when every group is covered and every placeable unit is on a group; None
        (undetermined) when a group is ambiguous and nothing already says False.
    :rtype: Optional[bool]
    """
    if any(s in (UNCOVERED, BLANK) for s in statuses):
        return False
    if any(s == AMBIGUOUS for s in statuses):
        return None
    placed_set = set(placed)
    return all(u in placed_set for u, ok in enumerate(placeable) if ok)


# --- one document -----------------------------------------------------------------------------------------------------

def group_evidence(texts: Sequence[str], unit_grams: Sequence[Mapping[str, int]], own_grams: Sequence[Mapping[str, int]],
                   section_grams: Sequence[Mapping[str, int]], null_units: Sequence[Mapping[str, int]],
                   settings: Settings) -> Dict[str, Any]:
    """Recall evidence of one image group's pooled reads.

    :param texts: All reads of the group's images.
    :type texts: Sequence[str]
    :param unit_grams: n-grams per edition unit.
    :type unit_grams: Sequence[Mapping[str, int]]
    :param own_grams: Each unit's own n-grams (:func:`exclusive_ngrams`).
    :type own_grams: Sequence[Mapping[str, int]]
    :param section_grams: n-grams per edition section.
    :type section_grams: Sequence[Mapping[str, int]]
    :param null_units: n-grams of other documents' units (the chance floor).
    :type null_units: Sequence[Mapping[str, int]]
    :param settings: Thresholds.
    :type settings: Settings
    :return: :func:`read_stats` plus ``floor``, ``recall_units``, ``corrected_units``,
        ``corrected_own_units`` and ``recall_sections``.
    :rtype: Dict[str, Any]
    """
    pooled = pooled_ngrams(texts, settings.ngram)
    floor = quantile([clipped_recall(unit, pooled) for unit in null_units], settings.floor_quantile)
    raw = [clipped_recall(g, pooled) for g in unit_grams]
    return {**read_stats(texts), "floor": round(floor, 4), "recall_units": [round(r, 4) for r in raw],
            "corrected_units": [round(chance_corrected(r, floor), 4) for r in raw],
            "corrected_own_units": [round(chance_corrected(clipped_recall(g, pooled), floor), 4) for g in own_grams],
            "recall_sections": [round(clipped_recall(g, pooled), 4) for g in section_grams]}


def decide_document(record: Mapping[str, Any], sides: Sequence[Tuple[str, str]], reads: Mapping[int, Mapping[str, str]],
                    null_units: Sequence[Mapping[str, int]], settings: Settings,
                    hashes: Optional[Sequence[Optional[int]]] = None,
                    digests: Optional[Sequence[Optional[str]]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Coverage of one document's images.

    :param record: Benchmark record.
    :type record: Mapping[str, Any]
    :param sides: :func:`image_sides` of the record.
    :type sides: Sequence[Tuple[str, str]]
    :param reads: ``{image_index: {tag: text}}`` of the pooled readers (may be empty).
    :type reads: Mapping[int, Mapping[str, str]]
    :param null_units: n-gram counts of other documents' units (the chance floor).
    :type null_units: Sequence[Mapping[str, int]]
    :param settings: Thresholds.
    :type settings: Settings
    :param hashes: Difference hash per image (None: no duplicate detection by appearance).
    :type hashes: Optional[Sequence[Optional[int]]]
    :param digests: Content digest per image (None: no duplicate detection by bytes).
    :type digests: Optional[Sequence[Optional[str]]]
    :return: ``(image rows, document row)``.
    :rtype: Tuple[List[Dict[str, Any]], Dict[str, Any]]
    """
    images = record["images"]
    n = len(images)
    groups = duplicate_groups(hashes or [None] * n, digests or [None] * n, settings.max_duplicate_bits)
    labels, label_clash = group_sides(groups, [s for s, _ in sides])
    units = side_units(record["gt"]["sections"])
    unit_grams = [ngram_counts(rasm_skeleton(u["text"]), settings.ngram) for u in units]
    own_grams = [exclusive_ngrams(unit_grams, k) for k in range(len(units))]
    section_grams = [ngram_counts(rasm_skeleton("\n".join(s["lines"])), settings.ngram) for s in record["gt"]["sections"]]
    placeable = [len(rasm_skeleton(u["text"])) >= settings.min_unit_letters for u in units]

    group_texts = [[text for i in group for text in reads.get(images[i]["image_index"], {}).values()] for group in groups]
    group_readers = [len({tag for i in group for tag in reads.get(images[i]["image_index"], {})}) for group in groups]
    evidence: List[Dict[str, Any]] = []
    by_reads: Optional[List[Dict[str, Any]]] = None
    if all(count >= settings.min_readers for count in group_readers):
        evidence = [{"readers": count, **group_evidence(texts, unit_grams, own_grams, section_grams, null_units, settings)}
                    for texts, count in zip(group_texts, group_readers)]
        by_reads = reads_decisions([e["corrected_units"] for e in evidence], [e["corrected_own_units"] for e in evidence],
                                   evidence, placeable, settings)

    labelled = None if label_clash else label_assignment(units, labels, record.get("pgp_side", ""))
    conflict = "copies of one photograph carry different side labels" if label_clash else ""
    if labelled is not None and labels_contradicted(labelled[1], by_reads):
        labelled, conflict = None, "the reads place a unit on an image its side label excludes"
    decisions = []
    for g, group in enumerate(groups):
        reads_g = by_reads[g] if by_reads else None
        if labelled is not None:
            assigned = labelled[1][g]
            decision = {"method": labelled[0], "status": COVERED if assigned else UNCOVERED, "units": assigned,
                        "reason": "side label" if assigned else "no edition unit for this side"}
        elif len(groups) == 1 and len(units) == 1:
            decision = {"method": SINGLE_IMAGE, "status": COVERED, "units": [0], "reason": "one image, one-side edition"}
        elif reads_g is not None:
            decision = {"method": READS, **reads_g}
        else:
            decision = {"method": NO_READS, "status": AMBIGUOUS, "units": [], "reason": "no pooled reads"}
        checked = decision["method"] in (SIDE_LABEL, PGP_SIDE_LABEL, SINGLE_IMAGE)
        decision["reads_check"] = reads_check(decision["status"], decision["units"], reads_g) if checked else None
        decisions.append(decision)

    group_of = {i: g for g, group in enumerate(groups) for i in group}
    rows = []
    for position, image in enumerate(images):
        g = group_of[position]
        d, reads_g = decisions[g], by_reads[g] if by_reads else None
        rows.append({"kind": "image", "doc_id": record["id"], "image_index": image["image_index"], "image_file": image["file"],
                     "side_label": sides[position][0], "side_source": sides[position][1],
                     "duplicate_of": images[groups[g][0]]["image_index"] if groups[g][0] != position else None,
                     "method": d["method"], "status": d["status"], "reason": d["reason"], "units": sorted(d["units"]),
                     "sections": sorted(s for u in d["units"] for s in units[u]["sections"]), "label_conflict": conflict or None,
                     "reads_status": reads_g["status"] if reads_g else None, "reads_units": reads_g["units"] if reads_g else None,
                     "reads_check": d["reads_check"], **({"pooled_images": len(groups[g]), **evidence[g]} if evidence else {})})

    statuses = [d["status"] for d in decisions]
    placed = sorted({u for d in decisions for u in d["units"]})
    match = coverage_match(statuses, placed, placeable)
    sides_match = sab.images_match_edition(record)
    doc = {"kind": "document", "doc_id": record["id"], "images": n, "distinct_images": len(groups),
           "duplicates": [[images[group[0]]["image_index"], images[i]["image_index"]] for group in groups for i in group[1:]],
           "covered": sum(r["status"] == COVERED for r in rows), "uncovered": sum(r["status"] == UNCOVERED for r in rows),
           "blank_or_faint": sum(r["status"] == BLANK for r in rows), "ambiguous": sum(r["status"] == AMBIGUOUS for r in rows),
           "methods": dict(Counter(r["method"] for r in rows)), "label_conflict": conflict or None,
           "units": [{"side": u["side"], "sections": u["sections"], "rasm_letters": len(rasm_skeleton(u["text"])), "placeable": ok,
                      "images": [r["image_index"] for r in rows if k in r["units"] and r["duplicate_of"] is None]}
                     for k, (u, ok) in enumerate(zip(units, placeable))],
           "units_unplaced": [k for k, ok in enumerate(placeable) if ok and k not in placed],
           "units_on_several_images": [k for k in placed if sum(k in d["units"] for d in decisions) > 1],
           "short_units": [k for k, ok in enumerate(placeable) if not ok],
           "coverage_match": match, "sides_match": sides_match,
           "consistent_with_sides_match": None if match is None else match == sides_match}
    return rows, doc


# --- one document, v2 ---------------------------------------------------------------------------------------------------

COPY_OF, MANUAL, INK, SINGLE_TEXT_IMAGE = "copy_of", "manual", "ink", "single_text_image"
ROLES = (COVERED, UNCOVERED, BLANK, AMBIGUOUS, COPY_OF)
SCRIPTS = ("arabic", "hebrew", "mixed", "none", "unknown")


def load_overrides(path: Optional[Path]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """By-eye overrides of the coverage map (one JSON object per line).

    Each row names ``doc_id`` and ``image_index`` and gives ``script`` (one of :data:`SCRIPTS`),
    ``role`` (one of :data:`ROLES`), ``units`` (unit indices, required for ``covered``), ``copy_of``
    (the image this one copies, required for ``copy_of``), ``edited_text_visible`` and ``reason``.

    :param path: Overrides file (None or missing: no overrides).
    :type path: Optional[Path]
    :return: ``{doc_id: {image_index: row}}``.
    :rtype: Dict[str, Dict[int, Dict[str, Any]]]
    :raises ValueError: A row with an unknown role or script, a ``covered`` row without units, a
        ``copy_of`` row without its target, or two rows for one image.
    """
    out: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    if path is None or not path.exists():
        return out
    with open(path, encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            where = f"{path.name}:{number}"
            if row.get("role") not in ROLES or row.get("script") not in SCRIPTS:
                raise ValueError(f"{where}: role {row.get('role')!r} / script {row.get('script')!r} not allowed")
            if row["role"] == COVERED and not row.get("units"):
                raise ValueError(f"{where}: a covered override needs its units")
            if row["role"] == COPY_OF and row.get("copy_of") is None:
                raise ValueError(f"{where}: a copy_of override needs the image it copies")
            index = int(row["image_index"])
            if index in out[row["doc_id"]]:
                raise ValueError(f"{where}: second override for {row['doc_id']} image {index}")
            out[row["doc_id"]][index] = row
    return out


def copy_groups(analyses: Sequence[Optional[Mapping[str, Any]]], digests: Sequence[Optional[str]],
                forced: Mapping[int, int], settings: Settings) -> Tuple[List[List[int]], Dict[int, Dict[str, Any]], List[List[int]]]:
    """Group a document's images that show the same side, whatever the rotation (v2).

    Two images join one group when their bytes are identical, when :func:`is_copy` holds (whole image or
    fragment crop, any of the four rotations, not a mirrored match) or when an override says so
    (``forced``). Groups are closed transitively. The representative of a group is the image an override
    names as the original, else its first image.

    :param analyses: :func:`analyse_image` per image position (None: not analysed).
    :type analyses: Sequence[Optional[Mapping[str, Any]]]
    :param digests: Content digest per image position (None: unknown).
    :type digests: Sequence[Optional[str]]
    :param forced: ``{copy position: original position}`` from overrides.
    :type forced: Mapping[int, int]
    :param settings: Thresholds.
    :type settings: Settings
    :return: ``(groups, evidence, mirrored)``: groups as position lists with the representative first and
        the rest in image order; per copy position its evidence against the representative
        (``{"by": [...], "whole": [...], "crop": [...]}``); and the position pairs whose best match is a
        mirrored one within ``max_duplicate_bits`` (likely the two sides of one leaf, kept apart).
    :rtype: Tuple[List[List[int]], Dict[int, Dict[str, Any]], List[List[int]]]
    """
    n = len(analyses)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    reasons: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    mirrored: List[List[int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if digests[i] is not None and digests[i] == digests[j]:
                reasons[(i, j)].append("bytes")
            if analyses[i] is not None and analyses[j] is not None:
                copy, evidence = is_copy(analyses[i], analyses[j], settings.max_duplicate_bits, settings.mirror_margin)
                if copy:
                    reasons[(i, j)].append("hash")
                elif any(e[1] <= settings.max_duplicate_bits and e[1] < e[0] for e in evidence.values()):
                    mirrored.append([i, j])
    for copy, original in forced.items():
        reasons[tuple(sorted((copy, original)))].append("override")
    for i, j in reasons:
        parent[find(j)] = find(i)
    members: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        members[find(i)].append(i)
    groups = []
    for group in sorted(members.values(), key=min):
        originals = [forced[i] for i in group if i in forced and forced[i] in group and forced[i] not in forced]
        representative = min(originals) if originals else group[0]
        groups.append([representative] + [i for i in group if i != representative])
    evidence_out: Dict[int, Dict[str, Any]] = {}
    for group in groups:
        rep = group[0]
        for i in group[1:]:
            entry: Dict[str, Any] = {"by": sorted(set(reasons.get(tuple(sorted((rep, i))), []))) or ["transitive"]}
            if analyses[rep] is not None and analyses[i] is not None:
                entry.update(is_copy(analyses[rep], analyses[i], settings.max_duplicate_bits, settings.mirror_margin)[1])
            evidence_out[i] = entry
    return groups, evidence_out, mirrored


def edition_script(text: str) -> str:
    """Script of an edition text.

    :param text: Edition text (one or several units).
    :type text: str
    :return: ``"arabic"``, ``"hebrew"``, ``"mixed"`` (the minority script holds at least a quarter of the
        letters) or ``"none"`` (no letters).
    :rtype: str
    """
    arabic, hebrew, _ = ar.script_share(text)
    if not arabic + hebrew:
        return "none"
    if min(arabic, hebrew) >= 0.25 * (arabic + hebrew):
        return "mixed"
    return "arabic" if arabic >= hebrew else "hebrew"


def group_ink(classes: Sequence[str]) -> str:
    """Ink class of an image group from its members' :func:`ink_class`.

    :param classes: Class per member.
    :type classes: Sequence[str]
    :return: ``"text"`` when any member shows writing, ``blank_or_faint`` when every member is blank,
        otherwise ``"unknown"``.
    :rtype: str
    """
    if "text" in classes:
        return "text"
    if classes and all(c == BLANK for c in classes):
        return BLANK
    return "unknown"


def member_evidence(member_reads: Sequence[Mapping[str, str]], unit_grams: Sequence[Mapping[str, int]],
                    own_grams: Sequence[Mapping[str, int]], section_grams: Sequence[Mapping[str, int]],
                    null_units: Sequence[Mapping[str, int]], settings: Settings) -> Optional[Dict[str, Any]]:
    """Reads evidence of an image group taken from its best-read member (v2).

    v1 pooled the reads of every copy of a photograph; a copy that is rotated (or upside down) is read as
    noise, which raises the group's chance floor and can push the right unit under the threshold. v2 scores
    every member with at least ``min_readers`` readers on its own and keeps the member whose best unit has
    the highest corrected recall (the first member on ties). Without such a member the reads of all members
    are pooled as in v1.

    :param member_reads: ``{tag: text}`` per member of the group, representative first.
    :type member_reads: Sequence[Mapping[str, str]]
    :param unit_grams: n-grams per edition unit.
    :type unit_grams: Sequence[Mapping[str, int]]
    :param own_grams: Each unit's own n-grams.
    :type own_grams: Sequence[Mapping[str, int]]
    :param section_grams: n-grams per edition section.
    :type section_grams: Sequence[Mapping[str, int]]
    :param null_units: n-gram counts of other documents' units.
    :type null_units: Sequence[Mapping[str, int]]
    :param settings: Thresholds.
    :type settings: Settings
    :return: :func:`group_evidence` plus ``readers`` and ``member`` (position of the member within the
        group; 0 when pooled), or None when the group has fewer than ``min_readers`` readers in all.
    :rtype: Optional[Dict[str, Any]]
    """
    best: Optional[Dict[str, Any]] = None
    for k, by_tag in enumerate(member_reads):
        if len(by_tag) < settings.min_readers:
            continue
        found = {"readers": len(by_tag), "member": k,
                 **group_evidence(list(by_tag.values()), unit_grams, own_grams, section_grams, null_units, settings)}
        if best is None or max(found["corrected_units"], default=0.0) > max(best["corrected_units"], default=0.0):
            best = found
    if best is not None:
        return best
    tags = {tag for by_tag in member_reads for tag in by_tag}
    if len(tags) < settings.min_readers:
        return None
    texts = [text for by_tag in member_reads for text in by_tag.values()]
    return {"readers": len(tags), "member": 0, **group_evidence(texts, unit_grams, own_grams, section_grams, null_units, settings)}


def decide_document_v2(record: Mapping[str, Any], sides: Sequence[Tuple[str, str]], reads: Mapping[int, Mapping[str, str]],
                       null_units: Sequence[Mapping[str, int]], settings: Settings, image_settings: ImageSettings,
                       analyses: Optional[Sequence[Optional[Mapping[str, Any]]]] = None,
                       digests: Optional[Sequence[Optional[str]]] = None,
                       overrides: Optional[Mapping[int, Mapping[str, Any]]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Coverage of one document's images, v2.

    Differences from :func:`decide_document`: copies are found whatever the rotation (:func:`copy_groups`);
    the image's own ink decides ``blank_or_faint`` (:func:`ink_class`); reads place only units of at least
    ``min_place_letters`` rasm letters, and a group the reads find empty stays ``ambiguous`` while a shorter
    unit is unplaced; a one-unit edition with exactly one non-blank image group is placed on it
    (``single_text_image``); by-eye overrides take precedence over every rule (method ``manual``); a unit
    placed on several groups is a joined section (``joined_units``); every image gets a ``role`` and a
    ``script`` that does not come from the scored models' reads.

    Decision order per image group: override, side labels, single image, single text image, reads, no reads;
    then an ``uncovered`` or ``ambiguous`` group whose images are all blank becomes ``blank_or_faint``
    (method ``ink``), and a covered group with long units but almost no ink is flagged ``ink_conflict``.

    :param record: Benchmark record.
    :type record: Mapping[str, Any]
    :param sides: :func:`image_sides` of the record.
    :type sides: Sequence[Tuple[str, str]]
    :param reads: ``{image_index: {tag: text}}`` of the pooled readers (may be empty).
    :type reads: Mapping[int, Mapping[str, str]]
    :param null_units: n-gram counts of other documents' units (the chance floor).
    :type null_units: Sequence[Mapping[str, int]]
    :param settings: Coverage thresholds.
    :type settings: Settings
    :param image_settings: Image thresholds (ink classes).
    :type image_settings: ImageSettings
    :param analyses: :func:`analyse_image` per image position (None: no image evidence).
    :type analyses: Optional[Sequence[Optional[Mapping[str, Any]]]]
    :param digests: Content digest per image position.
    :type digests: Optional[Sequence[Optional[str]]]
    :param overrides: ``{image_index: override row}`` (:func:`load_overrides`).
    :type overrides: Optional[Mapping[int, Mapping[str, Any]]]
    :return: ``(image rows, document row)``.
    :rtype: Tuple[List[Dict[str, Any]], Dict[str, Any]]
    :raises ValueError: An override names an image the record does not have, or two members of one image
        group carry different non-copy overrides.
    """
    images = record["images"]
    n = len(images)
    position = {im["image_index"]: k for k, im in enumerate(images)}
    analyses = list(analyses) if analyses is not None else [None] * n
    overrides = dict(overrides or {})
    unknown = [i for i in overrides if i not in position]
    if unknown:
        raise ValueError(f"{record['id']}: overrides for missing images {unknown}")
    forced = {position[i]: position[int(o["copy_of"])] for i, o in overrides.items() if o["role"] == COPY_OF}
    groups, copy_evidence, mirrored = copy_groups(analyses, list(digests) if digests is not None else [None] * n, forced, settings)
    labels, label_clash = group_sides(groups, [s for s, _ in sides])
    ink_classes = [ink_class(a["ink"], image_settings) if a is not None else "unknown" for a in analyses]
    inks = [group_ink([ink_classes[i] for i in group]) for group in groups]

    units = side_units(record["gt"]["sections"])
    letters = [len(rasm_skeleton(u["text"])) for u in units]
    unit_grams = [ngram_counts(rasm_skeleton(u["text"]), settings.ngram) for u in units]
    own_grams = [exclusive_ngrams(unit_grams, k) for k in range(len(units))]
    section_grams = [ngram_counts(rasm_skeleton("\n".join(s["lines"])), settings.ngram) for s in record["gt"]["sections"]]
    placeable = [count >= settings.min_place_letters for count in letters]

    evidence: List[Dict[str, Any]] = []
    by_reads: Optional[List[Dict[str, Any]]] = None
    group_evidences = [member_evidence([reads.get(images[i]["image_index"], {}) for i in group], unit_grams, own_grams, section_grams,
                                       null_units, settings) for group in groups]
    if all(e is not None for e in group_evidences):
        evidence = [{**e, "evidence_image": images[groups[g][e.pop("member")]]["image_index"]} for g, e in enumerate(group_evidences)]
        by_reads = reads_decisions([e["corrected_units"] for e in evidence], [e["corrected_own_units"] for e in evidence],
                                   evidence, placeable, settings, short_units_block_uncovered=True)

    labelled = None if label_clash else label_assignment(units, labels, record.get("pgp_side", ""))
    conflict = "copies of one photograph carry different side labels" if label_clash else ""
    if labelled is not None and labels_contradicted(labelled[1], by_reads):
        labelled, conflict = None, "the reads place a unit on an image its side label excludes"
    text_groups = [g for g, ink in enumerate(inks) if ink != BLANK]

    group_override: Dict[int, Mapping[str, Any]] = {}
    for g, group in enumerate(groups):
        found = {images[i]["image_index"]: overrides[images[i]["image_index"]] for i in group
                 if images[i]["image_index"] in overrides and overrides[images[i]["image_index"]]["role"] != COPY_OF}
        decided = {(o["role"], tuple(o.get("units") or [])) for o in found.values()}
        if len(decided) > 1:
            raise ValueError(f"{record['id']}: copies {sorted(found)} carry different overrides")
        if found:
            group_override[g] = found[min(found)]

    decisions = []
    for g, group in enumerate(groups):
        reads_g = by_reads[g] if by_reads else None
        if g in group_override:
            o = group_override[g]
            decision = {"method": MANUAL, "status": o["role"], "units": sorted(o.get("units") or []), "reason": o.get("reason", "")}
        elif labelled is not None:
            assigned = labelled[1][g]
            decision = {"method": labelled[0], "status": COVERED if assigned else UNCOVERED, "units": assigned,
                        "reason": "side label" if assigned else "no edition unit for this side"}
        elif len(groups) == 1 and len(units) == 1:
            decision = {"method": SINGLE_IMAGE, "status": COVERED, "units": [0], "reason": "one image, one-side edition"}
        elif len(units) == 1 and len(text_groups) == 1 and len(groups) > 1:
            decision = ({"method": SINGLE_TEXT_IMAGE, "status": COVERED, "units": [0], "reason": "the only image with writing, one-side edition"}
                        if g == text_groups[0] else
                        {"method": SINGLE_TEXT_IMAGE, "status": BLANK, "units": [], "reason": "blank beside the only image with writing"})
        elif reads_g is not None:
            decision = {"method": READS, **reads_g}
        else:
            decision = {"method": NO_READS, "status": AMBIGUOUS, "units": [], "reason": "no pooled reads"}
        checked = decision["method"] in (SIDE_LABEL, PGP_SIDE_LABEL, SINGLE_IMAGE, SINGLE_TEXT_IMAGE, MANUAL)
        decision["reads_check"] = reads_check(decision["status"], decision["units"], reads_g) if checked else None
        decision["ink_conflict"] = False
        if decision["method"] != MANUAL and inks[g] == BLANK:
            if decision["status"] in (UNCOVERED, AMBIGUOUS):
                decision.update(method=INK, status=BLANK, reason=f"almost no ink on the image ({decision['reason']})")
            elif decision["status"] == COVERED and sum(letters[u] for u in decision["units"]) >= settings.min_place_letters:
                decision["ink_conflict"] = True
        decisions.append(decision)

    carried: Dict[int, List[int]] = defaultdict(list)
    for g, d in enumerate(decisions):
        if d["status"] == COVERED:
            for u in d["units"]:
                carried[u].append(g)
    joined = {u: gs for u, gs in carried.items() if len(gs) > 1}

    group_of = {i: g for g, group in enumerate(groups) for i in group}
    rows = []
    for pos, image in enumerate(images):
        g = group_of[pos]
        d, reads_g = decisions[g], by_reads[g] if by_reads else None
        index = image["image_index"]
        own = overrides.get(index)
        rep_override = group_override.get(g)
        is_copy_row = groups[g][0] != pos
        if own is not None:
            script, script_source = own["script"], "manual"
        elif rep_override is not None:
            script, script_source = rep_override["script"], "manual"
        elif d["status"] == COVERED:
            script, script_source = edition_script("\n".join(units[u]["text"] for u in d["units"])), "edition"
        elif d["status"] == BLANK:
            script, script_source = "none", "ink"
        else:
            script, script_source = "unknown", ""
        analysis = analyses[pos]
        joined_with = sorted({images[groups[h][0]]["image_index"] for u in d["units"] if u in joined for h in joined[u] if h != g})
        rows.append({"kind": "image", "doc_id": record["id"], "image_index": index, "image_file": image["file"],
                     "side_label": sides[pos][0], "side_source": sides[pos][1],
                     "role": COPY_OF if is_copy_row else d["status"], "status": d["status"],
                     "copy_of": images[groups[g][0]]["image_index"] if is_copy_row else None,
                     "duplicate_of": images[groups[g][0]]["image_index"] if is_copy_row else None,
                     "copy_evidence": copy_evidence.get(pos), "method": d["method"], "reason": d["reason"],
                     "override_reason": (own or rep_override or {}).get("reason") if (own or rep_override) else None,
                     "override_edited_text_visible": (own or rep_override or {}).get("edited_text_visible") if (own or rep_override) else None,
                     "units": sorted(d["units"]), "sections": sorted(s for u in d["units"] for s in units[u]["sections"]),
                     "joined_with": joined_with, "script": script, "script_source": script_source,
                     "ink_class": ink_classes[pos], "group_ink": inks[g], "ink_conflict": d["ink_conflict"],
                     "ink": analysis["ink"] if analysis is not None else None,
                     "label_conflict": conflict or None,
                     "reads_status": reads_g["status"] if reads_g else None, "reads_units": reads_g["units"] if reads_g else None,
                     "reads_check": d["reads_check"], **({"pooled_images": len(groups[g]), **evidence[g]} if evidence else {})})

    statuses = [d["status"] for d in decisions]
    placed = sorted({u for d in decisions for u in d["units"]})
    match = coverage_match(statuses, placed, placeable)
    sides_match = sab.images_match_edition(record)
    roles = Counter(r["role"] for r in rows)
    doc = {"kind": "document", "doc_id": record["id"], "images": n, "distinct_images": len(groups),
           "copies": [[images[group[0]]["image_index"], images[i]["image_index"]] for group in groups for i in group[1:]],
           "mirrored_pairs": [[images[i]["image_index"], images[j]["image_index"]] for i, j in mirrored],
           **{role: roles.get(role, 0) for role in ROLES},
           "methods": dict(Counter(r["method"] for r in rows)), "overrides": len(overrides), "label_conflict": conflict or None,
           "units": [{"side": u["side"], "sections": u["sections"], "rasm_letters": count, "placeable_by_reads": ok,
                      "images": [images[groups[g][0]]["image_index"] for g in carried.get(k, [])]}
                     for k, (u, count, ok) in enumerate(zip(units, letters, placeable))],
           "units_unplaced": [k for k in range(len(units)) if k not in carried],
           "joined_units": {str(u): [images[groups[g][0]]["image_index"] for g in gs] for u, gs in sorted(joined.items())},
           "short_units": [k for k, ok in enumerate(placeable) if not ok],
           "ink_conflicts": [r["image_index"] for r in rows if r["ink_conflict"] and r["role"] != COPY_OF],
           "coverage_match": match, "sides_match": sides_match,
           "consistent_with_sides_match": None if match is None else match == sides_match}
    return rows, doc


# --- whole benchmark --------------------------------------------------------------------------------------------------

def load_reads(outputs: Path, tags: Sequence[str]) -> Dict[str, Dict[int, Dict[str, str]]]:
    """Cached reads of the pooled readers.

    :param outputs: ``<benchmark>/outputs``.
    :type outputs: Path
    :param tags: Answer-file stems.
    :type tags: Sequence[str]
    :return: ``{doc_id: {image_index: {tag: text}}}``.
    :rtype: Dict[str, Dict[int, Dict[str, str]]]
    """
    reads: Dict[str, Dict[int, Dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    for tag in tags:
        for doc_id, pages in sab.load_outputs(outputs / f"{tag}.jsonl").items():
            for index, text in pages.items():
                reads[doc_id][index][tag] = text
    return reads


def load_manifest(benchmark: Path) -> Dict[str, Dict[str, Any]]:
    """Manifest entry of every benchmark image file.

    :param benchmark: Benchmark directory.
    :type benchmark: Path
    :return: ``{file: entry}`` (``image_url``, ``sha256``, ...).
    :rtype: Dict[str, Dict[str, Any]]
    """
    with open(benchmark / "images_manifest.json", encoding="utf-8") as fh:
        manifest = json.load(fh)
    return {e["file"]: e for e in manifest["images"] if e.get("file")}


def image_hashes(benchmark: Path, files: Iterable[str], cache: Optional[Path], size: int) -> Dict[str, int]:
    """Difference hash of every image, read from or added to a JSON cache.

    :param benchmark: Benchmark directory (images under ``images/``).
    :type benchmark: Path
    :param files: Image file names.
    :type files: Iterable[str]
    :param cache: Cache file (None: no cache).
    :type cache: Optional[Path]
    :param size: Hash grid side.
    :type size: int
    :return: ``{file: hash}``.
    :rtype: Dict[str, int]
    """
    known: Dict[str, int] = {}
    if cache is not None and cache.exists():
        stored = json.loads(cache.read_text(encoding="utf-8"))
        if stored.get("size") == size:
            known = {k: int(v, 16) for k, v in stored["hashes"].items()}
    missing = [f for f in files if f not in known]
    for name in missing:
        known[name] = dhash(benchmark / "images" / name, size)
    if cache is not None and missing:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"size": size, "hashes": {k: format(v, "x") for k, v in sorted(known.items())}}), encoding="utf-8")
    return known


def null_unit_grams(benchmark: Mapping[str, Mapping[str, Any]], settings: Settings) -> Dict[str, List[Counter]]:
    """n-gram counts of every placeable unit, by document (the chance floor's reference texts).

    :param benchmark: Records by id.
    :type benchmark: Mapping[str, Mapping[str, Any]]
    :param settings: Thresholds.
    :type settings: Settings
    :return: ``{doc_id: [Counter, ...]}``.
    :rtype: Dict[str, List[Counter]]
    """
    out = {}
    for doc_id, record in benchmark.items():
        skeletons = [rasm_skeleton(u["text"]) for u in side_units(record["gt"]["sections"])]
        out[doc_id] = [ngram_counts(s, settings.ngram) for s in skeletons if len(s) >= settings.min_unit_letters]
    return out


def build_coverage(benchmark: Mapping[str, Mapping[str, Any]], manifest: Mapping[str, Mapping[str, Any]],
                   reads: Mapping[str, Mapping[int, Mapping[str, str]]], settings: Settings,
                   hashes: Optional[Mapping[str, int]] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Coverage rows of every document.

    :param benchmark: Records by id (file order).
    :type benchmark: Mapping[str, Mapping[str, Any]]
    :param manifest: :func:`load_manifest`.
    :type manifest: Mapping[str, Mapping[str, Any]]
    :param reads: :func:`load_reads`.
    :type reads: Mapping[str, Mapping[int, Mapping[str, str]]]
    :param settings: Thresholds.
    :type settings: Settings
    :param hashes: Difference hash per image file (None: duplicates by content digest only).
    :type hashes: Optional[Mapping[str, int]]
    :return: ``(image rows, document rows)``.
    :rtype: Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
    """
    nulls = null_unit_grams(benchmark, settings)
    urls = {f: e.get("image_url", "") for f, e in manifest.items()}
    image_rows, doc_rows = [], []
    for doc_id, record in benchmark.items():
        others = [g for other, grams in nulls.items() if other != doc_id for g in grams]
        files = [im["file"] for im in record["images"]]
        rows, doc = decide_document(record, image_sides(record, urls), reads.get(doc_id, {}), others, settings,
                                    [hashes.get(f) for f in files] if hashes else None,
                                    [manifest.get(f, {}).get("sha256") for f in files])
        image_rows.extend(rows)
        doc_rows.append(doc)
    return image_rows, doc_rows


def summarise(image_rows: Sequence[Mapping[str, Any]], doc_rows: Sequence[Mapping[str, Any]],
              only: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Counts by method and status, and agreement with the scorer's ``sides_match``.

    :param image_rows: Image rows.
    :type image_rows: Sequence[Mapping[str, Any]]
    :param doc_rows: Document rows.
    :type doc_rows: Sequence[Mapping[str, Any]]
    :param only: Restrict to these document ids.
    :type only: Optional[Iterable[str]]
    :return: Summary dict.
    :rtype: Dict[str, Any]
    """
    keep = set(only) if only is not None else None
    images = [r for r in image_rows if keep is None or r["doc_id"] in keep]
    docs = [d for d in doc_rows if keep is None or d["doc_id"] in keep]
    by_method: Dict[str, Counter] = defaultdict(Counter)
    for r in images:
        by_method[r["method"]][r["status"]] += 1
    agreement = Counter((d["sides_match"], d["coverage_match"]) for d in docs)
    return {"documents": len(docs), "images": len(images),
            "duplicate_images": sum(r["duplicate_of"] is not None for r in images),
            "status": dict(Counter(r["status"] for r in images)),
            "method": {m: dict(c) for m, c in sorted(by_method.items())},
            "reads_check": dict(Counter(r["reads_check"] for r in images if r["reads_check"])),
            "label_conflict_documents": sum(bool(d["label_conflict"]) for d in docs),
            "documents_without_ambiguous_images": sum(d["ambiguous"] == 0 for d in docs),
            "sides_match_vs_coverage": {f"sides_match={k[0]} coverage_match={k[1]}": v for k, v in sorted(agreement.items(), key=str)},
            "consistent": sum(d["consistent_with_sides_match"] is True for d in docs),
            "inconsistent": sum(d["consistent_with_sides_match"] is False for d in docs),
            "undetermined": sum(d["consistent_with_sides_match"] is None for d in docs)}


# blank_or_faint calibration (v2): images checked by eye on 2026-10-24
BLANK_CALIBRATION = ("Cambridge_CUL_T_S_Ar_40_196__1.jpg", "Cambridge_CUL_T_S_Ar_41_2__1.jpg", "Cambridge_CUL_T_S_AS_177_159__1.jpg",
                     "Cambridge_CUL_T_S_NS_327_2c__1.jpg", "New_York_JTS_ENA_NS_71_13__1.jpg", "Cambridge_CUL_T_S_Ar_39_217__1.jpg",
                     "Cambridge_CUL_T_S_Ar_39_370__1.jpg", "Cambridge_CUL_T_S_AS_214_237__1.jpg", "Cambridge_CUL_T_S_Ar_42_80__1.jpg",
                     "Cambridge_CUL_T_S_Ar_51_104__1.jpg")
TEXT_CALIBRATION = ("Cambridge_CUL_T_S_AS_176_166__1.jpg", "Cambridge_CUL_T_S_Ar_39_399__1.jpg", "Cambridge_CUL_T_S_Ar_42_165__1.jpg",
                    "Cambridge_CUL_T_S_Ar_39_245__1.jpg", "Cambridge_Mosseri_X_36__1.jpg", "Cambridge_CUL_T_S_Ar_39_69__1.jpg",
                    "Cambridge_CUL_T_S_Ar_39_341__1.jpg", "Cambridge_CUL_T_S_NS_305_97__1.jpg", "Cambridge_CUL_T_S_Ar_53_45__0.jpg",
                    "New_York_JTS_ENA_3958_10__0.jpg")
# hard cases seen by eye, reported with the calibration but not used to set the threshold
HARD_CASES = {"Cambridge_CUL_T_S_Ar_42_164__1.jpg": "one line of edited text at the top",
              "Princeton_PUL_Garrett_Additional_20_12__1.jpg": "faint lines of writing",
              "Manchester_JRL_B_2751__1.jpg": "a two-line note and show-through",
              "Cambridge_CUL_T_S_AS_178_98__1.jpg": "blank with creases and a stain",
              "Cambridge_CUL_T_S_Ar_39_372__1.jpg": "blank with stains and holes",
              "Cambridge_CUL_T_S_Ar_38_6__1.jpg": "blank, dark stained paper",
              "Oxford_Bodleian_Bodl_MS_Arab_c_56_5__0.jpg": "a grey text page beside blank volume pages"}


def blank_calibration(analyses: Mapping[str, Mapping[str, Any]], settings: ImageSettings,
                      blank: Sequence[str] = BLANK_CALIBRATION, text: Sequence[str] = TEXT_CALIBRATION,
                      hard: Mapping[str, str] = HARD_CASES) -> Dict[str, Any]:
    """How the ink rule classifies the by-eye calibration images.

    :param analyses: :func:`image_analyses`.
    :type analyses: Mapping[str, Mapping[str, Any]]
    :param settings: Image thresholds.
    :type settings: ImageSettings
    :param blank: Images known to be blank.
    :type blank: Sequence[str]
    :param text: Images known to show writing.
    :type text: Sequence[str]
    :param hard: Further images with a note (reported, not scored).
    :type hard: Mapping[str, str]
    :return: Threshold, per image ``[file, truth, class, ink per mille, loose ink per mille]`` and the
        number of calibration images classed as expected.
    :rtype: Dict[str, Any]
    """
    rows = []
    for truth, files in (("blank", blank), ("text", text), ("hard", list(hard))):
        for f in files:
            ink = analyses[f]["ink"]
            rows.append({"file": f, "truth": truth, "class": ink_class(ink, settings), "ink_permille": ink["ink_permille"],
                         "loose_ink_permille": ink.get("loose_ink_permille"), "note": hard.get(f, "")})
    expected = {"blank": BLANK, "text": "text"}
    return {"blank_max_permille": settings.blank_max_permille, "blank_max_loose_permille": settings.blank_max_loose_permille,
            "blank_ok": sum(r["class"] == BLANK for r in rows if r["truth"] == "blank"), "blank_n": len(blank),
            "text_ok": sum(r["class"] == "text" for r in rows if r["truth"] == "text"), "text_n": len(text),
            "max_blank_permille": max(r["ink_permille"] for r in rows if r["truth"] == "blank"),
            "min_text_permille": min(r["ink_permille"] for r in rows if r["truth"] == "text"),
            "images": rows, "expected": expected}


def build_coverage_v2(benchmark: Mapping[str, Mapping[str, Any]], manifest: Mapping[str, Mapping[str, Any]],
                      reads: Mapping[str, Mapping[int, Mapping[str, str]]], settings: Settings, image_settings: ImageSettings,
                      analyses: Mapping[str, Mapping[str, Any]],
                      overrides: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Coverage rows of every document, v2 (:func:`decide_document_v2`).

    :param benchmark: Records by id (file order).
    :type benchmark: Mapping[str, Mapping[str, Any]]
    :param manifest: :func:`load_manifest`.
    :type manifest: Mapping[str, Mapping[str, Any]]
    :param reads: :func:`load_reads`.
    :type reads: Mapping[str, Mapping[int, Mapping[str, str]]]
    :param settings: Coverage thresholds.
    :type settings: Settings
    :param image_settings: Image thresholds.
    :type image_settings: ImageSettings
    :param analyses: :func:`image_analyses` by image file.
    :type analyses: Mapping[str, Mapping[str, Any]]
    :param overrides: :func:`load_overrides`.
    :type overrides: Mapping[str, Mapping[int, Mapping[str, Any]]]
    :return: ``(image rows, document rows)``.
    :rtype: Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
    :raises ValueError: Overrides for a document the benchmark does not have.
    """
    stray = sorted(set(overrides) - set(benchmark))
    if stray:
        raise ValueError(f"overrides for unknown documents: {stray}")
    nulls = null_unit_grams(benchmark, settings)
    urls = {f: e.get("image_url", "") for f, e in manifest.items()}
    image_rows, doc_rows = [], []
    for doc_id, record in benchmark.items():
        others = [g for other, grams in nulls.items() if other != doc_id for g in grams]
        files = [im["file"] for im in record["images"]]
        rows, doc = decide_document_v2(record, image_sides(record, urls), reads.get(doc_id, {}), others, settings, image_settings,
                                       [analyses.get(f) for f in files], [manifest.get(f, {}).get("sha256") for f in files],
                                       overrides.get(doc_id, {}))
        image_rows.extend(rows)
        doc_rows.append(doc)
    return image_rows, doc_rows


def summarise_v2(image_rows: Sequence[Mapping[str, Any]], doc_rows: Sequence[Mapping[str, Any]],
                 only: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Counts of a v2 map: roles, methods, scripts, overrides, copies, joined units, ink.

    :param image_rows: v2 image rows.
    :type image_rows: Sequence[Mapping[str, Any]]
    :param doc_rows: v2 document rows.
    :type doc_rows: Sequence[Mapping[str, Any]]
    :param only: Restrict to these document ids.
    :type only: Optional[Iterable[str]]
    :return: Summary dict.
    :rtype: Dict[str, Any]
    """
    keep = set(only) if only is not None else None
    images = [r for r in image_rows if keep is None or r["doc_id"] in keep]
    docs = [d for d in doc_rows if keep is None or d["doc_id"] in keep]
    by_method: Dict[str, Counter] = defaultdict(Counter)
    by_script: Dict[str, Counter] = defaultdict(Counter)
    for r in images:
        by_method[r["method"]][r["role"]] += 1
        by_script[r["role"]][f"{r['script']} ({r['script_source'] or 'none'})"] += 1
    agreement = Counter((d["sides_match"], d["coverage_match"]) for d in docs)
    return {"documents": len(docs), "images": len(images), "role": dict(Counter(r["role"] for r in images)),
            "method": {m: dict(c) for m, c in sorted(by_method.items())},
            "script_by_role": {role: dict(c) for role, c in sorted(by_script.items())},
            "overridden_images": sum(r["override_reason"] is not None for r in images),
            "ink_class": dict(Counter(r["ink_class"] for r in images)),
            "ink_conflicts": sum(r["ink_conflict"] and r["role"] != COPY_OF for r in images),
            "copies": sum(r["role"] == COPY_OF for r in images),
            "copies_by": dict(Counter("+".join(r["copy_evidence"]["by"]) for r in images if r["role"] == COPY_OF and r["copy_evidence"])),
            "mirrored_pairs": sum(len(d["mirrored_pairs"]) for d in docs),
            "documents_with_joined_units": sum(bool(d["joined_units"]) for d in docs),
            "reads_check": dict(Counter(r["reads_check"] for r in images if r["reads_check"])),
            "label_conflict_documents": sum(bool(d["label_conflict"]) for d in docs),
            "documents_without_ambiguous_images": sum(d[AMBIGUOUS] == 0 for d in docs),
            "documents_with_a_covered_image": sum(d[COVERED] > 0 for d in docs),
            "sides_match_vs_coverage": {f"sides_match={k[0]} coverage_match={k[1]}": v for k, v in sorted(agreement.items(), key=str)}}


def v1_role(row: Mapping[str, Any]) -> str:
    """The role a v1 image row would have under the v2 vocabulary.

    :param row: v1 image row.
    :type row: Mapping[str, Any]
    :return: ``copy_of`` for a copy, else the v1 status.
    :rtype: str
    """
    return COPY_OF if row.get("duplicate_of") is not None else row["status"]


def compare_maps(old_rows: Sequence[Mapping[str, Any]], new_rows: Sequence[Mapping[str, Any]],
                 only: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    """Images whose role or units changed between a v1 and a v2 map.

    :param old_rows: v1 image rows.
    :type old_rows: Sequence[Mapping[str, Any]]
    :param new_rows: v2 image rows.
    :type new_rows: Sequence[Mapping[str, Any]]
    :param only: Restrict to these document ids.
    :type only: Optional[Iterable[str]]
    :return: One dict per changed image (``doc_id``, ``image_index``, old / new role, units, method, and
        the v2 reason or override reason).
    :rtype: List[Dict[str, Any]]
    """
    keep = set(only) if only is not None else None
    old = {(r["doc_id"], r["image_index"]): r for r in old_rows}
    changes = []
    for r in new_rows:
        if keep is not None and r["doc_id"] not in keep:
            continue
        o = old.get((r["doc_id"], r["image_index"]))
        if o is None:
            continue
        old_role = v1_role(o)
        if old_role != r["role"] or sorted(o["units"]) != sorted(r["units"]):
            changes.append({"doc_id": r["doc_id"], "image_index": r["image_index"], "v1_role": old_role, "v2_role": r["role"],
                            "v1_units": o["units"], "v2_units": r["units"], "v1_method": o["method"], "v2_method": r["method"],
                            "copy_of": r["copy_of"], "script": r["script"],
                            "why": r["override_reason"] or r["reason"]})
    return changes


def load_image_rows(path: Path) -> List[Dict[str, Any]]:
    """Image rows of a coverage map file.

    :param path: ``coverage_v*.jsonl``.
    :type path: Path
    :return: Image rows in file order.
    :rtype: List[Dict[str, Any]]
    """
    with open(path, encoding="utf-8") as fh:
        return [row for row in map(json.loads, (line for line in fh if line.strip())) if row.get("kind") == "image"]


def v2_report(report: Mapping[str, Any]) -> str:
    """Markdown report of a v2 build: counts, the paired-set changes against v1, the calibration.

    :param report: The dict written to ``summary_v2.json``.
    :type report: Mapping[str, Any]
    :return: Report text.
    :rtype: str
    """
    def counts(title: str, s: Mapping[str, Any]) -> List[str]:
        roles = list(ROLES)
        lines = [f"### {title}: {s['documents']} documents, {s['images']} images", "",
                 "| method | " + " | ".join(roles) + " | total |", "|---|" + "---|" * (len(roles) + 1)]
        for method, c in s["method"].items():
            lines.append(f"| {method} | " + " | ".join(str(c.get(r, 0)) for r in roles) + f" | {sum(c.values())} |")
        lines.append("| **all** | " + " | ".join(str(s["role"].get(r, 0)) for r in roles) + f" | {s['images']} |")
        lines += ["", "Script by role (source in brackets): "
                  + "; ".join(f"**{role}** " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())) for role, c in s["script_by_role"].items()),
                  "", f"Copies {s['copies']} ({s['copies_by']}); documents with joined units {s['documents_with_joined_units']}; "
                  f"overridden images {s['overridden_images']}; ink classes {s['ink_class']}; covered images with almost no ink "
                  f"{s['ink_conflicts']}; mirrored pairs kept apart {s['mirrored_pairs']}.", ""]
        return lines

    parts = ["# Coverage map v2 (`coverage_v2.jsonl`)", ""]
    parts += counts("All documents", report["all"])
    if "paired" in report:
        parts += counts("Paired set", report["paired"])
    changes = report.get("changes_paired", [])
    if changes:
        parts += [f"### Paired set: {len(changes)} images whose role or units changed from v1", "",
                  "| doc | img | v1 | v2 | v1 units | v2 units | v2 method | script | why |", "|---|---|---|---|---|---|---|---|---|"]
        for c in changes:
            role = c["v2_role"] + (f" {c['copy_of']}" if c["copy_of"] is not None else "")
            parts.append(f"| {c['doc_id']} | {c['image_index']} | {c['v1_role']} | {role} | {c['v1_units']} | {c['v2_units']} | "
                         f"{c['v2_method']} | {c['script']} | {c['why'].replace('|', '/')} |")
        parts.append("")
    cal = report.get("blank_calibration")
    if cal:
        parts += [f"### blank_or_faint calibration: ink < {cal['blank_max_permille']} per mille of the fragment mask and "
                  f"< {cal['blank_max_loose_permille']} in the loose mask", "",
                  f"Known blank classed blank: {cal['blank_ok']}/{cal['blank_n']} (max {cal['max_blank_permille']}); "
                  f"known text classed text: {cal['text_ok']}/{cal['text_n']} (min {cal['min_text_permille']}).", "",
                  "| image | truth | class | ink ‰ | loose ink ‰ | note |", "|---|---|---|---|---|---|"]
        for r in cal["images"]:
            parts.append(f"| {r['file']} | {r['truth']} | {r['class']} | {r['ink_permille']} | {r['loose_ink_permille']} | {r['note']} |")
        parts.append("")
    return "\n".join(parts)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI: build a coverage map (v2 by default; ``--map-version v1`` rebuilds the v1 logic).

    :param argv: Arguments.
    :type argv: Optional[List[str]]
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--tags", nargs="+", default=list(POOL_TAGS), help="answer files (outputs/<tag>.jsonl) whose reads are pooled")
    parser.add_argument("--map-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--out", type=Path, default=None, help="default: <benchmark>/coverage_<version>.jsonl (never overwritten)")
    parser.add_argument("--report-dir", type=Path, default=Path("logs/arabic_bench_coverage"))
    parser.add_argument("--overrides", type=Path, default=None, help="v2: default <report-dir>/manual_overrides_v2.jsonl")
    parser.add_argument("--compare-to", type=Path, default=None, help="v2: v1 map to compare with (default <benchmark>/coverage_v1.jsonl)")
    parser.add_argument("--paired-ids", type=Path, default=None, help="per-document scores CSV whose doc_ids form the paired set")
    args = parser.parse_args(argv)
    settings = Settings()
    if len(args.tags) < settings.min_readers:
        parser.error(f"pool at least {settings.min_readers} readers")
    out = args.out or args.benchmark / f"coverage_{args.map_version}.jsonl"
    if out.exists():
        parser.error(f"{out} exists; this tool never overwrites a coverage file")
    benchmark = sab.load_benchmark(args.benchmark)
    manifest = load_manifest(args.benchmark)
    files = [im["file"] for r in benchmark.values() for im in r["images"]]
    reads = load_reads(args.benchmark / "outputs", args.tags)
    image_settings = ImageSettings()
    if args.map_version == "v1":
        hashes = image_hashes(args.benchmark, files, args.report_dir / "image_dhash.json", settings.hash_size)
        image_rows, doc_rows = build_coverage(benchmark, manifest, reads, settings, hashes)
        summary = summarise
    else:
        analyses = image_analyses(args.benchmark, files, args.report_dir / "image_analysis_v2.json", image_settings)
        overrides = load_overrides(args.overrides or args.report_dir / "manual_overrides_v2.jsonl")
        image_rows, doc_rows = build_coverage_v2(benchmark, manifest, reads, settings, image_settings, analyses, overrides)
        summary = summarise_v2
    rows_by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in image_rows:
        rows_by_doc[row["doc_id"]].append(row)
    head = {"kind": "settings", "map_version": args.map_version, "tags": list(args.tags), **asdict(settings), "skeleton": "rasm (undotted)",
            "pooling": "per-gram maximum over the reads of a group",
            "floor": "per image group: floor_quantile of the recalls of other documents' units"}
    if args.map_version == "v2":
        head.update(image_settings=asdict(image_settings), overrides=str(args.overrides or args.report_dir / "manual_overrides_v2.jsonl"),
                    overridden_documents=len(overrides), overridden_images=sum(len(v) for v in overrides.values()))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(head, ensure_ascii=False) + "\n")
        for doc in doc_rows:
            for row in rows_by_doc[doc["doc_id"]]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
    report: Dict[str, Any] = {"map": str(out), "all": summary(image_rows, doc_rows)}
    paired = None
    if args.paired_ids:
        with open(args.paired_ids, encoding="utf-8") as fh:
            paired = [row["doc_id"] for row in csv.DictReader(fh)]
        report["paired"] = {"ids_from": str(args.paired_ids), **summary(image_rows, doc_rows, paired)}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    name = "summary.json" if args.map_version == "v1" else "summary_v2.json"
    if args.map_version == "v2":
        report["override_rows"] = sum(len(v) for v in overrides.values())
        report["blank_calibration"] = blank_calibration(analyses, image_settings)
        old = args.compare_to or args.benchmark / "coverage_v1.jsonl"
        if old.exists():
            old_rows = load_image_rows(old)
            report["compared_to"] = str(old)
            report["changes_all"] = compare_maps(old_rows, image_rows)
            if paired is not None:
                report["changes_paired"] = compare_maps(old_rows, image_rows, paired)
                report["paired_role_transitions"] = dict(Counter(f"{c['v1_role']} -> {c['v2_role']}" for c in report["changes_paired"]))
        (args.report_dir / "coverage_v2_report.md").write_text(v2_report(report), encoding="utf-8")
    with open(args.report_dir / name, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("changes")}, ensure_ascii=False, indent=1)[:6000])
    print(f"wrote {out} ({len(image_rows)} images, {len(doc_rows)} documents)")


if __name__ == "__main__":
    main()
