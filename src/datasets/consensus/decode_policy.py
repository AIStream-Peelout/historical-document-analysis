# File name: decode_policy.py
# Date: 10/12/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Decode policy for the transcription benchmarks: keep the whole-page read, read column crops, or rescue a failed read.

The whole-page VLM read fails in two known ways on the benchmarks. On two-column manuscripts it under-reads:
the model reads one column, or runs across the gutter. On some single-column pages the read fails outright:
it loops, hits the token cap, comes back near-empty or skips text (:mod:`src.datasets.consensus.read_health`).
The policy decides what to do with each page from what is already on disk: the cached Kraken lines and the
existing whole-page read.

``columns``  Kraken's line geometry splits the page into two or more columns, and by default the image also
             shows a vertical ink gap in its central band. Each column crop is read on its own, rightmost first.
``rescue``   the page is a single column and the whole-page read fails health (loop, capped, near-empty,
             skipped text). The column's top and bottom halves are read and joined.
``keep``     the page is a single column and the whole-page read is healthy, so the existing read stands and
             no model is called.

:func:`assemble` turns the unit reads into the page text. It judges every read (:func:`judge_read`) and falls
back one unit at a time: the crop read, then bands within the crop, then the whole-page read's share of the
unit. If no candidate resolves a unit, the page goes back to the whole-page read when that read is healthy,
and so does an assembly holding under 0.8x the letters of a healthy whole-page read. The over-generation test
budgets a read by the Kraken letters of its unit, unless Kraken under-segmented the page (under half the letters
of a healthy whole-page read): then only a runaway past 1.5x the whole-page read fails (:func:`weak_kraken_limit`).
Every decision is recorded in a JSON-ready trace.

The module is pure logic with no file or network I/O. The only image code (:func:`central_ink_gap`,
:func:`view_image`, :func:`crop_unit`) works on PIL images the caller has already opened. The driver is
``src/datasets/evaluations/helper_eval_scripts/run_decode_policy.py``.

Coordinates are in Kraken's *view* frame. For the few pages Kraken read in a rotated view (text running
vertically on the scan), the lines carry ``view_bbox``. The policy then plans in that frame, and the caller
crops from :func:`view_image` of the page, so the VLM sees upright text.
"""
import dataclasses
import math
import statistics
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from src.datasets.consensus.read_health import (
    DEFAULT_THRESHOLDS, REREAD_LABELS, SKIPPED_RATIO, ReadFacts, ReadHealth, Thresholds, classify, letter_count,
    read_facts)
from src.datasets.consensus.reread import OVERLAP_SHARE, half_boxes, join_halves, line_match
from src.finetuning.qwen_hebrew.ktiv_layout import (
    OCR_SUBSTANTIAL_LETTERS, _DOTS_RE, _split_columns_weighted, hebrew_letters)

Box = Tuple[int, int, int, int]
FBox = Tuple[float, float, float, float]

ACTION_COLUMNS = "columns"
ACTION_RESCUE = "rescue"
ACTION_KEEP = "keep"
ROLE_COLUMN = "column"
ROLE_HALF = "half"
SOURCE_CROP = "crop"
SOURCE_BANDS = "bands"
SOURCE_SHARE = "share"
FINAL_ASSEMBLED = "assembled"            # every unit resolved by a healthy read
FINAL_PARTIAL = "assembled_partial"      # some unit unresolved and no healthy whole-page read to fall back to
FINAL_WHOLE_PAGE = "whole_page"          # the existing whole-page read, verbatim
FALLBACK_SHORTER = "assembled_shorter_than_whole_page"   # page_fallback: the assembly lost text a healthy read holds
LABEL_OVERGEN = "overgen"

# ----------------------------------------------------------------------------- thresholds (one reason each)

PLAIN_CHARS_PER_TOKEN = 2.0      # plain-text replies run 2.0-2.6 characters per Qwen3-VL token (p5-max of 402 cached
                                 # benchmark reads, median 2.2), so len/2.0 bounds the tokens of 95 % of replies from
                                 # above; on those 402 reads it gives the same health labels as an exact re-tokenisation
CROP_MARGIN_LINE_HEIGHTS = 0.5   # half a line around the Kraken boxes keeps nikud, ascenders and descenders in the crop
MIN_CROP_MARGIN_PX = 25          # linewise._ink_envelope's pad: a crop never sits closer than this to the segmented ink
OVERGEN_RATIO = 3.0              # linewise._read_band's loop suspicion: more than 3x the letters Kraken saw in the region ...
OVERGEN_MIN_LETTERS = 400        # ... and more than 400 letters is invented text, not a read of the crop
WEAK_KRAKEN_RATIO = 0.5          # Kraken holding under half the letters of a healthy whole-page read under-segmented
                                 # the page (religious-140 v21b: 178 of 1,144 and 231 of 1,091 letters), so its count is
                                 # no budget: correct column reads of 506-587 letters failed the 3x rule there ...
RUNAWAY_RATIO = 1.5              # ... and on such a page a crop or band read over-generates only past 1.5x the
                                 # whole-page read's letters: one column cannot hold half again the whole page's text
ASSEMBLED_MIN_RATIO = SKIPPED_RATIO   # read_health's skipped-text ratio (0.8), measured against a healthy whole-page
                                      # read: an assembly holding fewer letters left text out, so that read stands
SHARE_MIN_MATCH = 0.55           # a whole-page line goes to the unit holding its best-matching Kraken line when
                                 # reread.line_match >= 0.55: 82 % of 1,494 whole-page lines (60 pages) reach it against
                                 # their own page's Kraken lines (median 0.84), 1.2 % against another page's (p95 0.50)
NO_CAP = 10 ** 9                 # token cap for joined texts (bands, shares): no single request produced them

INK_DOWNSAMPLE = 8               # split_x's working resolution: 1/8 of the page (at least INK_MIN_SIDE px a side)
INK_MIN_SIDE = 200
INK_TRIM_Y = 0.08                # the top and bottom 8 % (margins, rulers, catch-words) do not vote on the gutter
INK_BAND = (0.35, 0.65)          # the gutter of a two-column page lies in the central 30 % of the width
INK_LOW_PCT, INK_HIGH_PCT = 20, 80
INK_THRESH_SHARE = 0.15          # a column is "empty" below p20 + 15 % of the p20-p80 ink range
INK_MIN_RUN_DIV = 40             # the empty run must span >= 1/40 of the central band (and >= INK_MIN_RUN columns)
INK_MIN_RUN = 3

_TRANSPOSE = {90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_270}


@dataclass(frozen=True)
class PolicyThresholds:
    """Every knob of the decode policy.

    :param page_token_cap: ``max_tokens`` the existing whole-page read was made with (8192 PGP-131, 2500 religious-140).
    :param crop_token_cap: ``max_tokens`` of a crop read (the benchmark's own value).
    :param health: :mod:`~src.datasets.consensus.read_health` thresholds.
    :param chars_per_token: Characters per token for estimating a plain-text reply's token count.
    :param rescue_labels: Whole-page health labels that send a single-column page to ``rescue``.
    :param require_ink_gap: A Kraken column split counts only when the image has a central vertical ink gap.
    :param crop_margin_line_heights: Margin around a crop's Kraken boxes, in line heights.
    :param half_overlap: Share of the column height both halves of a rescue contain.
    :param overgen_ratio: A read holding more than this many times the unit's Kraken letters ...
    :param overgen_min_letters: ... and more than this many letters fails as ``overgen``.
    :param share_min_match: Minimum ``line_match`` for attributing a whole-page line to a unit.
    :param weak_kraken_ratio: Kraken is weak on a page when its letters fall below this share of a healthy
        whole-page read's letters. The over-generation test then uses ``runaway_ratio`` instead of Kraken's
        count (:func:`weak_kraken_limit`). 0 turns the relaxation off.
    :param runaway_ratio: On a weak-Kraken page, a read holding more than this many times the whole-page read's
        letters (and more than ``overgen_min_letters``) fails as ``overgen``.
    :param assembled_min_ratio: An assembly holding fewer letters than this share of a healthy whole-page read's
        letters gives way to the whole-page read (:func:`assemble`). 0 turns the check off.
    """

    page_token_cap: int = 8192
    crop_token_cap: int = 8192
    health: Thresholds = DEFAULT_THRESHOLDS
    chars_per_token: float = PLAIN_CHARS_PER_TOKEN
    rescue_labels: Tuple[str, ...] = REREAD_LABELS
    require_ink_gap: bool = True
    crop_margin_line_heights: float = CROP_MARGIN_LINE_HEIGHTS
    half_overlap: float = OVERLAP_SHARE
    overgen_ratio: float = OVERGEN_RATIO
    overgen_min_letters: int = OVERGEN_MIN_LETTERS
    share_min_match: float = SHARE_MIN_MATCH
    weak_kraken_ratio: float = WEAK_KRAKEN_RATIO
    runaway_ratio: float = RUNAWAY_RATIO
    assembled_min_ratio: float = ASSEMBLED_MIN_RATIO

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready form.

        :return: Every field (the health thresholds as a nested dict).
        :rtype: Dict[str, Any]
        """
        out = dataclasses.asdict(self)
        out["rescue_labels"] = list(self.rescue_labels)
        return out


def thresholds_for(token_cap: int, require_ink_gap: bool = True) -> PolicyThresholds:
    """Policy thresholds for a benchmark whose reads use one ``max_tokens`` for pages and crops alike.

    :param token_cap: The benchmark's ``max_tokens`` (8192 PGP-131, 2500 religious-140).
    :type token_cap: int
    :param require_ink_gap: Gate Kraken column splits on a central ink gap.
    :type require_ink_gap: bool
    :return: Thresholds.
    :rtype: PolicyThresholds
    """
    return PolicyThresholds(page_token_cap=token_cap, crop_token_cap=token_cap, require_ink_gap=require_ink_gap)


# ----------------------------------------------------------------------------- plan records


@dataclass(frozen=True)
class ColumnPlan:
    """One column of a page, as Kraken's lines lay it out.

    :param index: Reading-order position (0 = rightmost).
    :param crop: Pixel crop box ``(x0, y0, x1, y1)`` in the view frame, with margin, clipped to the page and
        to the gutters on either side.
    :param line_indices: Indices into the Kraken JSON's ``lines`` of the column's lines, top to bottom.
    :param boxes: The same lines' boxes (view frame).
    :param letters: Hebrew letters Kraken read in the column.
    :param core: x-extent ``(min x0, max x1)`` of the column's substantial lines (>= ``OCR_SUBSTANTIAL_LETTERS``).
    """

    index: int
    crop: Box
    line_indices: Tuple[int, ...]
    boxes: Tuple[FBox, ...]
    letters: int
    core: Tuple[float, float]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready form (boxes left out: the line indices point at them).

        :return: Index, crop, line indices, line count, letters and core extent.
        :rtype: Dict[str, Any]
        """
        return {"index": self.index, "crop": list(self.crop), "line_indices": list(self.line_indices),
                "n_lines": len(self.line_indices), "letters": self.letters,
                "core": [round(self.core[0], 1), round(self.core[1], 1)]}


@dataclass(frozen=True)
class ReadUnit:
    """One crop the VLM reads: a column, or one half of a rescued column.

    :param name: ``col0``, ``col1``, ... (reading order) or ``top`` / ``bottom``.
    :param role: :data:`ROLE_COLUMN` or :data:`ROLE_HALF`.
    :param crop: Pixel crop box in the view frame.
    :param seg_lines: The Kraken lines inside the crop, top to bottom, as ``{"idx", "bbox", "text"}`` (the
        record shape :func:`~src.datasets.consensus.linewise._read_band` and ``build_bands`` take).
    """

    name: str
    role: str
    crop: Box
    seg_lines: Tuple[Dict[str, Any], ...]

    @property
    def kraken_texts(self) -> List[str]:
        """Texts of the unit's Kraken lines.

        :return: Line texts, top to bottom.
        :rtype: List[str]
        """
        return [ln["text"] for ln in self.seg_lines]

    @property
    def kraken_letters(self) -> int:
        """Hebrew and Arabic letters Kraken read in the unit (the health detector's count).

        :return: Letter count.
        :rtype: int
        """
        return sum(letter_count(t) for t in self.kraken_texts)

    @property
    def boxes(self) -> List[List[int]]:
        """Boxes of the unit's Kraken lines.

        :return: ``[x0, y0, x1, y1]`` per line.
        :rtype: List[List[int]]
        """
        return [list(ln["bbox"]) for ln in self.seg_lines]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready form.

        :return: Name, role, crop, Kraken line indices and letters.
        :rtype: Dict[str, Any]
        """
        return {"name": self.name, "role": self.role, "crop": list(self.crop),
                "line_indices": [ln["idx"] for ln in self.seg_lines], "kraken_letters": self.kraken_letters}


@dataclass(frozen=True)
class PagePlan:
    """What the policy does with one page, and why.

    :param action: :data:`ACTION_COLUMNS`, :data:`ACTION_RESCUE` or :data:`ACTION_KEEP`.
    :param reason: Short reason (``2_columns``, ``whole_page_loop``, ``whole_page_ok``, ``whole_page_missing``).
    :param rotation: Kraken view rotation the coordinates are in (crop from :func:`view_image`).
    :param image_size: View-frame page size ``(width, height)``.
    :param line_h: Median Kraken line height (px).
    :param columns: The columns the policy acts on (one for ``rescue`` / ``keep``).
    :param units: Crops to read, in reading order (none for ``keep``).
    :param whole_page: Health of the existing whole-page read (None when there is none).
    :param kraken_columns: Columns Kraken's split found before the ink-gap gate.
    :param ink_gap_x: x of the image's central vertical ink gap in the view frame (None: no gap, or not measured).
    :param ink_gap_in_gutter: Whether that gap lies within a line height of a Kraken gutter (diagnostic only).
    :param kraken_letters: Hebrew and Arabic letters on all Kraken lines.
    :param n_kraken_lines: Kraken lines with text.
    """

    action: str
    reason: str
    rotation: int
    image_size: Tuple[int, int]
    line_h: float
    columns: Tuple[ColumnPlan, ...]
    units: Tuple[ReadUnit, ...]
    whole_page: Optional[ReadHealth]
    kraken_columns: int
    ink_gap_x: Optional[int]
    ink_gap_in_gutter: Optional[bool]
    kraken_letters: int
    n_kraken_lines: int

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready form.

        :return: Every field; the whole-page health as its dict (None when missing).
        :rtype: Dict[str, Any]
        """
        return {"action": self.action, "reason": self.reason, "rotation": self.rotation,
                "image_size": list(self.image_size), "line_h": round(self.line_h, 1),
                "kraken_columns": self.kraken_columns, "n_columns": len(self.columns),
                "ink_gap_x": self.ink_gap_x, "ink_gap_in_gutter": self.ink_gap_in_gutter,
                "kraken_letters": self.kraken_letters, "n_kraken_lines": self.n_kraken_lines,
                "whole_page": self.whole_page.to_dict() if self.whole_page else None,
                "columns": [c.to_dict() for c in self.columns], "units": [u.to_dict() for u in self.units]}


@dataclass
class UnitReads:
    """What the reader returned for one unit (filled by the driver, judged by :func:`assemble`).

    :param crop: Reply to the unit's crop (None: not read).
    :param bands: Joined band reads within the crop (None: not read; the driver reads bands only when the crop
        read fails :func:`judge_read`).
    :param band_details: Per-band records for the trace (box, letters, ``_read_band`` flags).
    """

    crop: Optional[str] = None
    bands: Optional[str] = None
    band_details: List[Dict[str, Any]] = field(default_factory=list)


# ----------------------------------------------------------------------------- Kraken geometry


def kraken_rotation(kraken_json: Mapping[str, Any]) -> int:
    """Rotation of the Kraken view the policy plans in.

    :param kraken_json: Cached ``/transcribe_lines`` response (``lines``, ``image_size``, ``view``).
    :type kraken_json: Mapping[str, Any]
    :return: 0, 90, 180 or 270. A rotated view counts only when every line carries its ``view_bbox``; otherwise
        the original frame (0) is used.
    :rtype: int
    :raises ValueError: For a rotation other than a multiple of 90 degrees.
    """
    rotation = int((kraken_json.get("view") or {}).get("rotation") or 0) % 360
    if rotation not in (0, 90, 180, 270):
        raise ValueError(f"unsupported Kraken view rotation {rotation}")
    if rotation and not all("view_bbox" in ln for ln in kraken_json.get("lines") or []):
        return 0
    return rotation


def kraken_units(kraken_json: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], float, Tuple[int, int]]:
    """Kraken lines as layout units, built the way ``ktiv_layout.reorder_ocr_lines`` builds them.

    Text is NFC-normalised and stripped, empty lines are dropped, and ``letters`` counts Hebrew letters.
    ``line_h`` is the median height of units with at least two letters, or of all units when none has two.

    :param kraken_json: Cached ``/transcribe_lines`` response.
    :type kraken_json: Mapping[str, Any]
    :return: ``(units, line_h, image_size)``. Each unit is ``{"idx", "text", "box", "gap", "letters"}``, where
        ``idx`` indexes ``kraken_json["lines"]`` and ``box`` is in the view frame. ``image_size`` is the
        view-frame ``(width, height)``.
    :rtype: Tuple[List[Dict[str, Any]], float, Tuple[int, int]]
    """
    rotation = kraken_rotation(kraken_json)
    width, height = (int(v) for v in kraken_json["image_size"])
    size = (height, width) if rotation in (90, 270) else (width, height)
    units: List[Dict[str, Any]] = []
    for idx, line in enumerate(kraken_json.get("lines") or []):
        text = unicodedata.normalize("NFC", (line.get("text") or "").strip())
        if not text:
            continue
        x0, y0, x1, y1 = (float(v) for v in (line["view_bbox"] if rotation else line["bbox"]))
        units.append({"idx": idx, "text": text, "box": (x0, y0, x1, y1),
                      "gap": bool(_DOTS_RE.match(text)), "letters": hebrew_letters(text)})
    heights = ([u["box"][3] - u["box"][1] for u in units if u["letters"] >= 2]
               or [u["box"][3] - u["box"][1] for u in units])
    line_h = max(statistics.median(heights), 1.0) if heights else 1.0
    return units, line_h, size


def _envelope(boxes: Iterable[Sequence[float]]) -> FBox:
    """Bounding box of some boxes.

    :param boxes: ``(x0, y0, x1, y1)`` boxes (at least one).
    :type boxes: Iterable[Sequence[float]]
    :return: ``(min x0, min y0, max x1, max y1)``.
    :rtype: FBox
    """
    bs = list(boxes)
    return (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))


def _core_extent(band: Sequence[Mapping[str, Any]]) -> Tuple[float, float]:
    """x-extent of a column's substantial units, the ones that define gutters in ``_split_columns_weighted``.

    :param band: Units of one column.
    :type band: Sequence[Mapping[str, Any]]
    :return: ``(min x0, max x1)`` over units with >= ``OCR_SUBSTANTIAL_LETTERS`` letters (all units when none).
    :rtype: Tuple[float, float]
    """
    core = [u for u in band if u["letters"] >= OCR_SUBSTANTIAL_LETTERS] or list(band)
    return min(u["box"][0] for u in core), max(u["box"][2] for u in core)


def _gutter_x(right: Sequence[Mapping[str, Any]], left: Sequence[Mapping[str, Any]]) -> float:
    """x of the gutter between two neighbouring columns.

    :param right: Units of the right-hand column.
    :type right: Sequence[Mapping[str, Any]]
    :param left: Units of the left-hand column.
    :type left: Sequence[Mapping[str, Any]]
    :return: Midpoint of the gap between their substantial extents. If those extents overlap, the midpoint
        between the left column's rightmost unit centre and the right column's leftmost unit centre.
    :rtype: float
    """
    right_x0, left_x1 = _core_extent(right)[0], _core_extent(left)[1]
    if right_x0 > left_x1:
        return (right_x0 + left_x1) / 2

    def centre(u: Mapping[str, Any]) -> float:
        """Horizontal centre of a unit.

        :param u: Unit.
        :type u: Mapping[str, Any]
        :return: Centre x.
        :rtype: float
        """
        return (u["box"][0] + u["box"][2]) / 2

    return (min(centre(u) for u in right) + max(centre(u) for u in left)) / 2


def crop_margin(line_h: float, line_heights: float = CROP_MARGIN_LINE_HEIGHTS) -> int:
    """Margin around a crop's Kraken boxes.

    :param line_h: Median line height (px).
    :type line_h: float
    :param line_heights: Margin in line heights.
    :type line_heights: float
    :return: ``max(MIN_CROP_MARGIN_PX, round(line_heights * line_h))`` pixels.
    :rtype: int
    """
    return max(MIN_CROP_MARGIN_PX, int(round(line_heights * line_h)))


def detect_columns(units: Sequence[Dict[str, Any]], line_h: float, image_size: Tuple[int, int],
                   require_ink_gap: bool = True, ink_gap_x: Optional[int] = None,
                   margin_line_heights: float = CROP_MARGIN_LINE_HEIGHTS) -> List[ColumnPlan]:
    """Columns of a page from its Kraken units, rightmost first.

    The split is ``ktiv_layout._split_columns_weighted``: gutters of at least 1.5 line heights between
    substantial lines, with bands holding under 12 % of the letters merged into a neighbour. When
    ``require_ink_gap`` is set, a split also needs a central vertical ink gap in the image (:func:`central_ink_gap`),
    anywhere in the central band. On religious-140, Kraken alone finds 25 true and 1 false two-column pages;
    with the gap it finds 24 and 0.

    :param units: Units from :func:`kraken_units`.
    :type units: Sequence[Dict[str, Any]]
    :param line_h: Median line height (px).
    :type line_h: float
    :param image_size: View-frame page size ``(width, height)``.
    :type image_size: Tuple[int, int]
    :param require_ink_gap: Accept a multi-column split only when ``ink_gap_x`` is not None.
    :type require_ink_gap: bool
    :param ink_gap_x: The image's central ink gap (:func:`central_ink_gap` of the view image), or None.
    :type ink_gap_x: Optional[int]
    :param margin_line_heights: Crop margin in line heights.
    :type margin_line_heights: float
    :return: One :class:`ColumnPlan` per column, rightmost first. A single column gets one plan covering the ink
        envelope, or the whole page when Kraken found no text.
    :rtype: List[ColumnPlan]
    """
    width, height = image_size
    if not units:
        return [ColumnPlan(0, (0, 0, width, height), (), (), 0, (0.0, float(width)))]
    bands = _split_columns_weighted(list(units), line_h)
    if len(bands) >= 2 and require_ink_gap and ink_gap_x is None:
        bands = [list(units)]
    gutters = [_gutter_x(right, left) for right, left in zip(bands, bands[1:])]
    margin = crop_margin(line_h, margin_line_heights)
    plans = []
    for k, band in enumerate(bands):
        x0, y0, x1, y1 = _envelope(u["box"] for u in band)
        left_limit = gutters[k] if k < len(gutters) else 0.0             # gutter with the column to the left
        right_limit = gutters[k - 1] if k > 0 else float(width)          # gutter with the column to the right
        cx0 = max(0, math.floor(max(x0 - margin, left_limit)))
        cx1 = min(width, math.ceil(min(x1 + margin, right_limit)))
        if cx1 <= cx0:                                                   # degenerate clip: keep the plain envelope
            cx0, cx1 = max(0, math.floor(x0 - margin)), min(width, math.ceil(x1 + margin))
        crop = (cx0, max(0, math.floor(y0 - margin)), cx1, min(height, math.ceil(y1 + margin)))
        ordered = sorted(band, key=lambda u: ((u["box"][1] + u["box"][3]) / 2, -u["box"][2]))
        plans.append(ColumnPlan(index=k, crop=crop, line_indices=tuple(u["idx"] for u in ordered),
                                boxes=tuple(tuple(u["box"]) for u in ordered),
                                letters=sum(u["letters"] for u in band), core=_core_extent(band)))
    return plans


def ink_gap_in_gutter(columns: Sequence[ColumnPlan], gap_x: Optional[int], line_h: float) -> Optional[bool]:
    """Whether an ink gap lies in one of the Kraken gutters (within a line height). Used only as a trace diagnostic.

    The gate in :func:`detect_columns` does not use the gap's position. On religious-140, 3 of the 24 correct
    two-column calls have their widest ink gap outside the Kraken gutter.

    :param columns: Kraken columns, rightmost first.
    :type columns: Sequence[ColumnPlan]
    :param gap_x: Ink gap x, or None.
    :type gap_x: Optional[int]
    :param line_h: Median line height (px).
    :type line_h: float
    :return: None when there is no gap or no gutter; else whether the gap sits in a gutter.
    :rtype: Optional[bool]
    """
    if gap_x is None or len(columns) < 2:
        return None
    return any(left.core[1] - line_h <= gap_x <= right.core[0] + line_h
               for right, left in zip(columns, columns[1:]))


# ----------------------------------------------------------------------------- images (PIL in, PIL out)


def view_image(image: Image.Image, rotation: int) -> Image.Image:
    """The page in Kraken's view frame (the transpose of ``kraken_microservice/k7/views.rotate``).

    :param image: Page as stored.
    :type image: Image.Image
    :param rotation: :func:`kraken_rotation` of the page's Kraken JSON.
    :type rotation: int
    :return: The same image for 0, else the lossless transpose (90 = counter-clockwise).
    :rtype: Image.Image
    """
    return image if rotation == 0 else image.transpose(_TRANSPOSE[rotation])


def crop_unit(image: Image.Image, unit: ReadUnit) -> Image.Image:
    """Crop one unit out of the view-frame page.

    :param image: :func:`view_image` of the page.
    :type image: Image.Image
    :param unit: Unit to crop.
    :type unit: ReadUnit
    :return: The crop.
    :rtype: Image.Image
    """
    return image.crop(unit.crop)


def central_ink_gap(image: Image.Image) -> Optional[int]:
    """The widest vertical ink gap in the central band of a page (``probe_column_transcription.split_x`` on a PIL image).

    The page is shrunk to 1/8, inverted so ink is bright, and trimmed of its top and bottom 8 %. The ink profile
    over x then finds the longest run of near-empty columns inside the central 35-65 % of the width.

    :param image: Page (any mode; converted to greyscale).
    :type image: Image.Image
    :return: Gap x in the image's pixels, or None when no clear gap exists.
    :rtype: Optional[int]
    """
    grey = image.convert("L")
    w, h = grey.size
    small = grey.resize((max(INK_MIN_SIDE, w // INK_DOWNSAMPLE), max(INK_MIN_SIDE, h // INK_DOWNSAMPLE)))
    ink = 255 - np.asarray(small, dtype=np.float32)
    ink = ink[int(ink.shape[0] * INK_TRIM_Y): int(ink.shape[0] * (1 - INK_TRIM_Y))]
    profile = ink.mean(axis=0)
    lo, hi = int(profile.size * INK_BAND[0]), int(profile.size * INK_BAND[1])
    band = profile[lo:hi]
    p_low, p_high = np.percentile(profile, INK_LOW_PCT), np.percentile(profile, INK_HIGH_PCT)
    low = band <= p_low + INK_THRESH_SHARE * (p_high - p_low)
    best, start, best_span = None, None, 0
    for i, flag in enumerate(list(low) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start > best_span:
                best_span, best = i - start, (start + i) // 2
            start = None
    if best is None or best_span < max(INK_MIN_RUN, band.size // INK_MIN_RUN_DIV):
        return None
    return int((lo + best) * w / profile.size)


# ----------------------------------------------------------------------------- health


def estimate_tokens(text: str, chars_per_token: float = PLAIN_CHARS_PER_TOKEN) -> int:
    """Token count of a plain-text reply estimated from its length (an upper bound for most replies).

    :param text: Reply text.
    :type text: str
    :param chars_per_token: Characters per token.
    :type chars_per_token: float
    :return: ``ceil(len(text) / chars_per_token)``.
    :rtype: int
    """
    return math.ceil(len(text or "") / chars_per_token)


def health_of(text: Optional[str], kraken_texts: Sequence[str], cap: Optional[int], reply_tokens: Optional[int] = None,
              th: Thresholds = DEFAULT_THRESHOLDS, chars_per_token: float = PLAIN_CHARS_PER_TOKEN) -> ReadHealth:
    """Health of a plain-text read: :func:`~src.datasets.consensus.read_health.read_health` minus its JSON assumptions.

    Without a token count, ``read_health`` calls a reply capped when it is not a closed JSON array and is at
    least ``MIN_CHARS_PER_TOKEN`` (1.0) characters per allowed token long. Plain text is never a closed array,
    so every long healthy read misfires: 5 of the 140 religious v21b reads. Here a plain reply has no array to
    close (``closed`` is False, even when the text happens to end in ``]``). Its token count comes from the
    caller or from :func:`estimate_tokens`, so ``capped`` means "used (nearly) the whole cap".

    :param text: Reply text (None or empty for no reply).
    :type text: Optional[str]
    :param kraken_texts: Kraken line texts of the same region.
    :type kraken_texts: Sequence[str]
    :param cap: ``max_tokens`` of the request. Pass None for joined texts (bands, shares), which no single
        request produced, so the cap check never fires.
    :type cap: Optional[int]
    :param reply_tokens: Tokens the reply used, when known.
    :type reply_tokens: Optional[int]
    :param th: Detector thresholds.
    :type th: Thresholds
    :param chars_per_token: Characters per token for the estimate.
    :type chars_per_token: float
    :return: Label, flags and facts. The facts carry the token count the verdict used.
    :rtype: ReadHealth
    """
    text = text or ""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    tokens = estimate_tokens(text, chars_per_token) if reply_tokens is None else reply_tokens
    facts = dataclasses.replace(read_facts(text, lines, list(kraken_texts), tokens, th.tail_letters), closed=False)
    label, flags = classify(facts, NO_CAP if cap is None else cap, th)
    return ReadHealth(label, flags, facts)


def weak_kraken_limit(plan: PagePlan, whole_page_text: Optional[str], th: PolicyThresholds) -> Optional[float]:
    """The over-generation limit of a page where Kraken under-segmented the text, or None for Kraken's rule.

    Kraken sometimes finds only a fraction of a page's lines (religious-140 v21b: 178 letters on a page whose
    healthy whole-page read holds 1,144). Three times a unit's Kraken letters is then no budget for a crop read,
    and correct column reads fail as ``overgen``. When the whole-page read is healthy and the page's Kraken letters
    fall below ``th.weak_kraken_ratio`` of its letters, a read over-generates only when it holds more than
    ``th.runaway_ratio`` times the whole-page read's letters. The limit never drops below
    ``th.overgen_min_letters``, and with the default ratios (0.5 x 3 = 1.5) ``3 x`` any unit's Kraken letters
    stays under ``1.5 x`` the whole-page letters on such a page, so the relaxed test never fails a read that
    Kraken's rule passes.

    :param plan: The page's plan (its ``kraken_letters`` and whole-page health).
    :type plan: PagePlan
    :param whole_page_text: The existing whole-page read (None when there is none).
    :type whole_page_text: Optional[str]
    :param th: Policy thresholds.
    :type th: PolicyThresholds
    :return: ``max(runaway_ratio * whole-page letters, overgen_min_letters)`` when the relaxation applies; None
        when the whole-page read is missing or unhealthy, or Kraken read enough letters.
    :rtype: Optional[float]
    """
    if whole_page_text is None or plan.whole_page is None or plan.whole_page.failed:
        return None
    whole_letters = letter_count(whole_page_text)
    if plan.kraken_letters >= th.weak_kraken_ratio * whole_letters:
        return None
    return float(max(th.runaway_ratio * whole_letters, th.overgen_min_letters))


def is_overgenerated(facts: ReadFacts, th: PolicyThresholds, runaway_letters: Optional[float] = None) -> bool:
    """Whether a read holds far more letters than its region can (the ``_read_band`` suspicion rule).

    :param facts: Measurements of the read.
    :type facts: ReadFacts
    :param th: Policy thresholds.
    :type th: PolicyThresholds
    :param runaway_letters: The page's :func:`weak_kraken_limit`, or None for Kraken's rule.
    :type runaway_letters: Optional[float]
    :return: With a limit, True when ``reply_letters > runaway_letters``. Without one, True when
        ``reply_letters > max(overgen_ratio * kraken_letters, overgen_min_letters)``.
    :rtype: bool
    """
    if runaway_letters is not None:
        return facts.reply_letters > runaway_letters
    return facts.reply_letters > max(th.overgen_ratio * facts.kraken_letters, th.overgen_min_letters)


def judge_read(text: Optional[str], kraken_texts: Sequence[str], cap: Optional[int], th: PolicyThresholds,
               runaway_letters: Optional[float] = None) -> ReadHealth:
    """Verdict on a unit read: :func:`health_of` plus the over-generation guard.

    The same function decides whether the driver reads bands (crop failed) and which candidate
    :func:`assemble` keeps.

    :param text: The read.
    :type text: Optional[str]
    :param kraken_texts: Kraken line texts of the unit.
    :type kraken_texts: Sequence[str]
    :param cap: ``max_tokens`` of the request (None for joined texts).
    :type cap: Optional[int]
    :param th: Policy thresholds.
    :type th: PolicyThresholds
    :param runaway_letters: The page's :func:`weak_kraken_limit` (None: Kraken's rule).
    :type runaway_letters: Optional[float]
    :return: The health verdict. When the read over-generates, ``overgen`` is added to its flags, and it
        becomes the label if no other check failed.
    :rtype: ReadHealth
    """
    health = health_of(text, kraken_texts, cap, th=th.health, chars_per_token=th.chars_per_token)
    if not is_overgenerated(health.facts, th, runaway_letters):
        return health
    return ReadHealth(health.label if health.failed else LABEL_OVERGEN, health.flags + (LABEL_OVERGEN,), health.facts)


# ----------------------------------------------------------------------------- planning


def _seg_line(unit: Mapping[str, Any]) -> Dict[str, Any]:
    """A layout unit as a Kraken line record (the shape ``_read_band`` / ``build_bands`` consume).

    :param unit: Unit from :func:`kraken_units`.
    :type unit: Mapping[str, Any]
    :return: ``{"idx", "bbox", "text"}`` with an integer view-frame box.
    :rtype: Dict[str, Any]
    """
    return {"idx": unit["idx"], "bbox": [int(round(v)) for v in unit["box"]], "text": unit["text"]}


def _centre_inside(box: Sequence[float], crop: Sequence[int]) -> bool:
    """Whether a box's centre lies inside a crop.

    :param box: ``(x0, y0, x1, y1)``.
    :type box: Sequence[float]
    :param crop: ``(x0, y0, x1, y1)``.
    :type crop: Sequence[int]
    :return: True when the centre is inside (edges included).
    :rtype: bool
    """
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return crop[0] <= cx <= crop[2] and crop[1] <= cy <= crop[3]


def column_units(columns: Sequence[ColumnPlan], units: Sequence[Dict[str, Any]]) -> List[ReadUnit]:
    """One read unit per column, in reading order.

    :param columns: Columns, rightmost first.
    :type columns: Sequence[ColumnPlan]
    :param units: Units from :func:`kraken_units`.
    :type units: Sequence[Dict[str, Any]]
    :return: Units ``col0``, ``col1``, ... with each column's lines.
    :rtype: List[ReadUnit]
    """
    by_idx = {u["idx"]: u for u in units}
    return [ReadUnit(f"col{c.index}", ROLE_COLUMN, c.crop, tuple(_seg_line(by_idx[i]) for i in c.line_indices))
            for c in columns]


def half_units(column: ColumnPlan, units: Sequence[Dict[str, Any]], overlap: float = OVERLAP_SHARE) -> List[ReadUnit]:
    """The top and bottom half of a column crop (:func:`~src.datasets.consensus.reread.half_boxes` within the crop).

    :param column: The page's single column.
    :type column: ColumnPlan
    :param units: Units from :func:`kraken_units`.
    :type units: Sequence[Dict[str, Any]]
    :param overlap: Share of the crop height both halves contain.
    :type overlap: float
    :return: Units ``top`` and ``bottom``. Each holds the column lines whose centre falls inside it, and lines
        in the overlap belong to both.
    :rtype: List[ReadUnit]
    """
    x0, y0, x1, y1 = column.crop
    wanted = set(column.line_indices)
    members = [u for u in units if u["idx"] in wanted]
    members.sort(key=lambda u: ((u["box"][1] + u["box"][3]) / 2, -u["box"][2]))
    out = []
    for name, (hx0, hy0, hx1, hy1) in zip(("top", "bottom"), half_boxes(x1 - x0, y1 - y0, overlap)):
        crop = (x0 + hx0, y0 + hy0, x0 + hx1, y0 + hy1)
        out.append(ReadUnit(name, ROLE_HALF, crop,
                            tuple(_seg_line(u) for u in members if _centre_inside(u["box"], crop))))
    return out


def plan_page(kraken_json: Mapping[str, Any], whole_page_text: Optional[str], thresholds: PolicyThresholds,
              ink_gap_x: Optional[int] = None) -> PagePlan:
    """Decide what to do with one page from its cached Kraken lines and its existing whole-page read.

    * ``columns``: Kraken finds two or more columns and, when ``thresholds.require_ink_gap`` is set, the
      image has a central ink gap. Each column is read, whatever the whole-page read's health.
    * ``rescue``: a single column whose whole-page read is missing or fails health with a label in
      ``thresholds.rescue_labels`` (loop, capped, near-empty, skipped text). The column's halves are read. When
      Kraken read fewer than ``thresholds.health.min_kraken_letters`` letters, its ink envelope is no guide to
      where the text is, so the halves cover the whole page instead (reason suffix ``_kraken_blank``).
    * ``keep``: otherwise; the whole-page read stands.

    The policy never looks at ground truth. The religious spec's ``n_columns`` is for the dry-run report only.

    :param kraken_json: Cached ``/transcribe_lines`` response of the page.
    :type kraken_json: Mapping[str, Any]
    :param whole_page_text: The existing whole-page read (None when there is none).
    :type whole_page_text: Optional[str]
    :param thresholds: Policy thresholds.
    :type thresholds: PolicyThresholds
    :param ink_gap_x: :func:`central_ink_gap` of the view-frame image (None: no gap, or not measured).
    :type ink_gap_x: Optional[int]
    :return: The plan.
    :rtype: PagePlan
    """
    units, line_h, size = kraken_units(kraken_json)
    margin = thresholds.crop_margin_line_heights
    kraken_split = detect_columns(units, line_h, size, False, None, margin)
    columns = detect_columns(units, line_h, size, thresholds.require_ink_gap, ink_gap_x, margin)
    kraken_texts = [u["text"] for u in units]
    kraken_letters = sum(letter_count(t) for t in kraken_texts)
    whole = (None if whole_page_text is None
             else health_of(whole_page_text, kraken_texts, thresholds.page_token_cap, th=thresholds.health,
                            chars_per_token=thresholds.chars_per_token))
    if len(columns) >= 2:
        action, reason, read_units = ACTION_COLUMNS, f"{len(columns)}_columns", column_units(columns, units)
    elif whole is None or whole.label in thresholds.rescue_labels:
        action = ACTION_RESCUE
        reason = "whole_page_missing" if whole is None else f"whole_page_{whole.label}"
        if kraken_letters < thresholds.health.min_kraken_letters:
            # Kraken read next to nothing, so its ink envelope is no guide to where the text is: halve the page.
            columns = [dataclasses.replace(columns[0], crop=(0, 0, size[0], size[1]))]
            reason += "_kraken_blank"
        read_units = half_units(columns[0], units, thresholds.half_overlap)
    else:
        action, reason, read_units = ACTION_KEEP, f"whole_page_{whole.label}", []
    return PagePlan(action=action, reason=reason, rotation=kraken_rotation(kraken_json), image_size=size,
                    line_h=line_h, columns=tuple(columns), units=tuple(read_units), whole_page=whole,
                    kraken_columns=len(kraken_split), ink_gap_x=ink_gap_x,
                    ink_gap_in_gutter=ink_gap_in_gutter(kraken_split, ink_gap_x, line_h),
                    kraken_letters=kraken_letters, n_kraken_lines=len(units))


# ----------------------------------------------------------------------------- assembly


def whole_page_shares(whole_page_text: str, units: Sequence[ReadUnit],
                      min_match: float = SHARE_MIN_MATCH) -> Dict[str, str]:
    """Split the whole-page read among the units by matching its lines to the units' Kraken lines.

    Each whole-page line goes to the unit holding its best-matching Kraken line (``reread.line_match``, with
    the first unit winning a tie), provided the match reaches ``min_match``. Lines that match nothing are
    left out.

    :param whole_page_text: The whole-page read.
    :type whole_page_text: str
    :param units: The plan's units.
    :type units: Sequence[ReadUnit]
    :param min_match: Minimum line match.
    :type min_match: float
    :return: ``unit name -> its lines`` (newline-joined, in whole-page order; empty when nothing matched).
    :rtype: Dict[str, str]
    """
    kraken = [(u.name, t) for u in units for t in u.kraken_texts]
    shares: Dict[str, List[str]] = {u.name: [] for u in units}
    for line in (ln.strip() for ln in (whole_page_text or "").splitlines()):
        best, owner = 0.0, None
        for name, ktext in kraken:
            score = line_match(line, ktext)
            if score > best:
                best, owner = score, name
        if owner is not None and best >= min_match:
            shares[owner].append(line)
    return {name: "\n".join(lines) for name, lines in shares.items()}


def join_columns(texts: Sequence[str]) -> str:
    """Column reads joined in reading order.

    :param texts: Column texts, rightmost column first.
    :type texts: Sequence[str]
    :return: Non-empty texts joined with a newline.
    :rtype: str
    """
    return "\n".join(t.strip() for t in texts if t and t.strip())


def join_half_texts(top: str, bottom: str) -> Tuple[str, int]:
    """Two half reads joined, the lines read by both halves kept once (:func:`~src.datasets.consensus.reread.join_halves`).

    :param top: Top half's read.
    :type top: str
    :param bottom: Bottom half's read.
    :type bottom: str
    :return: ``(joined text, bottom lines dropped as overlap)``.
    :rtype: Tuple[str, int]
    """
    def lines_of(text: str) -> List[Dict[str, str]]:
        """Non-empty lines as ``{"text": line}`` records.

        :param text: A read.
        :type text: str
        :return: Line records.
        :rtype: List[Dict[str, str]]
        """
        return [{"text": ln.strip()} for ln in (text or "").splitlines() if ln.strip()]

    joined, dropped = join_halves(lines_of(top), lines_of(bottom))
    return "\n".join(ln["text"] for ln in joined), dropped


def _attempt(source: str, text: str, health: ReadHealth) -> Dict[str, Any]:
    """Trace record of one candidate read of a unit.

    :param source: :data:`SOURCE_CROP`, :data:`SOURCE_BANDS` or :data:`SOURCE_SHARE`.
    :type source: str
    :param text: The candidate text.
    :type text: str
    :param health: Its verdict.
    :type health: ReadHealth
    :return: Source, letters, characters and the verdict.
    :rtype: Dict[str, Any]
    """
    return {"source": source, "letters": letter_count(text), "chars": len(text), "health": health.to_dict()}


def resolve_unit(unit: ReadUnit, reads: UnitReads, share: Callable[[], str],
                 th: PolicyThresholds, runaway_letters: Optional[float] = None) -> Tuple[str, Dict[str, Any]]:
    """Pick one unit's text: the first healthy candidate of crop -> bands -> whole-page share.

    :param unit: The unit.
    :type unit: ReadUnit
    :param reads: What the reader returned for it.
    :type reads: UnitReads
    :param share: Returns the whole-page read's share of the unit (called only when crop and bands fail).
    :type share: Callable[[], str]
    :param th: Policy thresholds.
    :type th: PolicyThresholds
    :param runaway_letters: The page's :func:`weak_kraken_limit` for the over-generation test (None: Kraken's rule).
    :type runaway_letters: Optional[float]
    :return: ``(text, trace)``. The trace's ``used`` names the source; it is ``unresolved`` when no candidate
        was healthy, and then the text is the best effort: bands (``_read_band`` already bisected or truncated
        suspect bands), else crop, else share.
    :rtype: Tuple[str, Dict[str, Any]]
    """
    texts = unit.kraken_texts
    candidates: List[Tuple[str, str]] = []
    attempts: List[Dict[str, Any]] = []
    used: Optional[str] = None
    if reads.crop is not None:
        candidates.append((SOURCE_CROP, reads.crop))
    if reads.bands is not None:
        candidates.append((SOURCE_BANDS, reads.bands))
    for source, text in candidates:
        health = judge_read(text, texts, th.crop_token_cap if source == SOURCE_CROP else None, th, runaway_letters)
        attempts.append(_attempt(source, text, health))
        if not health.failed:
            used = source
            break
    if used is None:
        share_text = share()
        if share_text:
            health = judge_read(share_text, texts, None, th, runaway_letters)
            attempts.append(_attempt(SOURCE_SHARE, share_text, health))
            candidates.append((SOURCE_SHARE, share_text))
            if not health.failed:
                used = SOURCE_SHARE
    by_source = dict(candidates)
    if used is not None:
        text = by_source[used]
    else:
        text = next((by_source[s] for s in (SOURCE_BANDS, SOURCE_CROP, SOURCE_SHARE) if by_source.get(s)), "")
    trace = {**unit.to_dict(), "attempts": attempts, "band_details": list(reads.band_details),
             "used": used or "unresolved", "resolved": used is not None, "letters": letter_count(text)}
    return text, trace


def assemble(plan: PagePlan, reads: Mapping[str, UnitReads], whole_page_text: Optional[str],
             th: PolicyThresholds) -> Tuple[str, Dict[str, Any]]:
    """The page text from the unit reads, with a JSON-ready trace of every decision.

    * ``keep``: the whole-page read, verbatim.
    * Every unit resolved (:func:`resolve_unit`): the columns rightmost first, or the halves joined with
      their overlap kept once.
    * Some unit unresolved and the whole-page read healthy, or every unit unresolved: the whole-page read
      entirely (``page_fallback`` says why).
    * Some unit unresolved and the whole-page read unhealthy or missing: the assembly with each unresolved
      unit's best effort.
    * An assembly holding fewer than ``th.assembled_min_ratio`` x a healthy whole-page read's letters: the
      whole-page read (``page_fallback`` :data:`FALLBACK_SHORTER`; the units' trace is kept).

    Units are judged with :func:`weak_kraken_limit` when Kraken under-segmented the page. The trace records
    the page's Kraken letters, whether that relaxation applied (``weak_kraken_relaxation``) and its limit
    (``overgen_runaway_letters``), next to ``whole_page_letters``, ``assembled_letters`` (None when no
    assembly was built) and ``final_letters``.

    :param plan: The page's plan.
    :type plan: PagePlan
    :param reads: ``unit name -> UnitReads`` (missing units count as not read).
    :type reads: Mapping[str, UnitReads]
    :param whole_page_text: The existing whole-page read (None when there is none).
    :type whole_page_text: Optional[str]
    :param th: Policy thresholds.
    :type th: PolicyThresholds
    :return: ``(page text, trace)``.
    :rtype: Tuple[str, Dict[str, Any]]
    """
    whole_letters = None if whole_page_text is None else letter_count(whole_page_text)
    runaway = weak_kraken_limit(plan, whole_page_text, th)
    trace: Dict[str, Any] = {"action": plan.action, "reason": plan.reason, "units": [], "page_fallback": None,
                             "whole_page_letters": whole_letters, "kraken_letters": plan.kraken_letters,
                             "weak_kraken_relaxation": runaway is not None, "overgen_runaway_letters": runaway,
                             "assembled_letters": None}
    if plan.action == ACTION_KEEP:
        text = whole_page_text or ""
        trace.update(final_source=FINAL_WHOLE_PAGE, final_letters=letter_count(text))
        return text, trace
    cache: Dict[str, Dict[str, str]] = {}

    def share_of(name: str) -> str:
        """The whole-page read's share of one unit (computed once per page, on first use).

        :param name: Unit name.
        :type name: str
        :return: The share ("" without a whole-page read).
        :rtype: str
        """
        if not whole_page_text:
            return ""
        if "shares" not in cache:
            cache["shares"] = whole_page_shares(whole_page_text, plan.units, th.share_min_match)
        return cache["shares"].get(name, "")

    texts = []
    for unit in plan.units:
        text, unit_trace = resolve_unit(unit, reads.get(unit.name) or UnitReads(), lambda n=unit.name: share_of(n), th,
                                        runaway)
        texts.append(text)
        trace["units"].append(unit_trace)
    resolved = [u["resolved"] for u in trace["units"]]
    whole_ok = whole_page_text is not None and plan.whole_page is not None and not plan.whole_page.failed
    if not all(resolved) and whole_page_text is not None and (whole_ok or not any(resolved)):
        trace.update(page_fallback="unit_unresolved" if whole_ok else "all_units_unresolved",
                     final_source=FINAL_WHOLE_PAGE, final_letters=whole_letters)
        return whole_page_text, trace
    if plan.action == ACTION_RESCUE and len(texts) == 2:
        final, dropped = join_half_texts(texts[0], texts[1])
        trace["overlap_lines_dropped"] = dropped
    else:
        final = join_columns(texts)
    trace["assembled_letters"] = letter_count(final)
    if whole_ok and trace["assembled_letters"] < th.assembled_min_ratio * whole_letters:
        # The units resolved, but together they hold far less text than a healthy whole-page read: some unit's
        # source (typically a whole-page share standing in for a rejected crop) lost text.
        trace.update(page_fallback=FALLBACK_SHORTER, final_source=FINAL_WHOLE_PAGE, final_letters=whole_letters)
        return whole_page_text, trace
    trace.update(final_source=FINAL_ASSEMBLED if all(resolved) else FINAL_PARTIAL,
                 final_letters=trace["assembled_letters"])
    return final, trace
