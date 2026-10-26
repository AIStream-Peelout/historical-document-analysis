# File name: line_extent.py
# Date: 10/20/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Ink extent of a text line beyond its segmenter polygon (image helpers for ``align_edition_lines``).

The segmenter's polygon often stops before the end of a handwritten Arabic line: the line-final (left) word
is written raised or compressed, or segmented as a separate small box, and the first (right) word can be
cut too; on tall hands its top edge also cuts ascenders and dots along the whole line. A training line
whose polygon misses part of its edition text teaches the recogniser text that is not in its image, and one
that takes ink of a neighbouring line or a margin note teaches it strokes that are not in its text. This
module rebuilds the polygon of a placed line from the page's ink, text-free:

1. **Ink** (:func:`ink_mask`, :func:`page_ink`). The mount (the blue or black backing board around and behind
   the fragment) is masked by a saturation rule (blue hue, high saturation) and a darkness rule (very dark
   pixels connected to the image border) (:func:`mount_mask`) and painted with the median paper gray. The
   paper background is a grayscale morphological closing wider than a pen stroke (:data:`BACKGROUND_SHARE`
   of a line height), so stains and lighting gradients cancel out. A pixel is ink when its darkness below
   that background passes the page's Otsu threshold (never below :data:`MIN_CONTRAST`), or, for faint grey
   ink, when it is locally dark (darkness above the mean plus :data:`ADAPTIVE_K` standard deviations over a
   :data:`ADAPTIVE_WINDOW`-pixel window, :func:`adaptive_ink`), reaches :data:`ADAPTIVE_OTSU_SHARE` of the
   Otsu threshold and its piece touches a letter-sized Otsu component (at least :data:`ADAPTIVE_SEED` line
   heights squared, grown by :data:`ADAPTIVE_JUMP` pixels; :func:`hysteresis`): a faint tail of a stroke or a
   faded word next to darker ink counts, parchment texture and stains alone do not. Specks are dropped, and
   so are components lying mostly in the zone around the mount (the
   dark rim of a hole or of the fragment's edge, :data:`RIM_SHARE`). The ink is kept as labelled connected
   components (:class:`PageInk`) with their stroke width (:func:`stroke_widths`) and median darkness
   (:func:`component_darkness`).
2. **Scene** (:func:`line_scene`). A window of the page around the line holds the ink labels, the line's
   sloped baseline per column, the baselines of the nearest other lines above and below it per column
   (:func:`neighbour_rows`: lines whose x-range overlaps the line's, carried :data:`NEIGHBOUR_REACH` line
   heights beyond their ends, with the baseline at least :data:`NEIGHBOUR_MIN_GAP` line heights away) and
   the polygons of every other segmenter line (grown by :data:`OTHER_DILATE` pixels and subtracted from
   everything the line gains). Lines the segmenter never found are **detected** as peaks of the ink row
   profile along the baseline (:func:`detect_rows`) and join the neighbours. Every component in the window is
   classified: **foreign** when its centroid lies nearer another (segmented, carried :data:`FOREIGN_REACH` line
   heights beyond its ends, or detected) baseline than this line's (:func:`foreign_components`); **connected**
   when it holds the line's own segment ink (non-foreign ink inside its polygons) or lies within
   :data:`OWN_TOUCH` pixels of it; **pencil** when, not connected, its
   median darkness or stroke width is below :data:`PENCIL_SHARE` of the own ink's (:func:`pencil_components`);
   **beyond** when, not connected, its centroid lies past the walk's bounds (side material: a seal, a margin
   column). A component **counts** as the line's ink when it is at least :data:`WALK_MIN_AREA` (squared, in
   line heights) large, at most :data:`MAX_INSIDE_OTHER` of its pixels lie inside another line's polygon, its
   centroid lies in the line's own band (inside the band, :data:`BAND_ABOVE` line heights above the baseline
   to :data:`BAND_BELOW` below, or between :data:`CENTROID_ABOVE` above and :data:`CENTROID_BELOW` below while
   the component reaches no higher than :data:`REACH_ABOVE` nor lower than :data:`REACH_BELOW`), between the
   previous line's baseline plus :data:`NEIGHBOUR_CLAMP` line heights and the next line's baseline minus it,
   and it is neither foreign, pencil nor beyond.
3. **Walk** (:func:`extend_end`). From each polygon end outward, a column holds ink when at least
   :data:`MIN_COLUMN_INK` of a line height (and :data:`MIN_COLUMN_PIXELS`) of its band pixels (band clamped
   to the neighbours) are counted or connected ink outside the other lines' polygons. The end moves to the
   last ink column reachable without a blank run of :data:`EXTEND_GAP` line heights, at most :data:`EXTEND_MAX`
   line heights per end. Up to the text block bound (:data:`BLOCK_SLACK` line heights beyond the column
   block's margin, :func:`text_block`) and to the hard limits (margin material, gutters; no slack) any counted
   ink moves the end; past them only ink connected to the line's own letters does (a swash, a kashida or a
   final letter running past the margin is never cut). Same-row segmenter boxes (centre inside the band) whose
   near edge the walk reaches are absorbed whole (unless that would pass the bound, or the box is not
   absorbable); a box flagged ``steal`` (it holds another edition line's text) is reported instead of
   absorbed. Ink still continuing beyond the final end (same gap rule, inside the bound) is measured for the
   geometry gate.
4. **Polygon** (:func:`line_shape`), rasterised per column. The member polygons and the absorbed boxes are
   kept. Columns beyond the ends and in the seams between merged members get the line's own vertical extent
   at that end (the median top and bottom of the polygon over its last :data:`END_WINDOW` line heights,
   relative to the baseline, carried along the sloped baseline; interpolated across a seam). With
   ``raise_top`` every column's top is raised to ``min(BAND_ABOVE lh, SPACING_ABOVE x spacing)`` above the
   baseline and its bottom lowered to ``min(BAND_BELOW lh, SPACING_BELOW x spacing)`` below it
   (:func:`raise_offsets`: the page's measured baseline spacing, not the inflated polygon height, limits the
   raise), clamped to the neighbours; foreign ink hanging into that strip pulls it back
   (:func:`trim_foreign`). Component pixels (grown by :data:`COMPONENT_PAD` line heights) are added only for
   admitted components: connected ones in any column (up to :data:`UP_CAP` above and :data:`OWN_DOWN_CAP`
   below the baseline, :data:`NEXT_TOP_CLEAR` above the next line's ink top), the walk's own components
   beyond the ends, and counted components lying inside the original x-range (:data:`UP_CAP` /
   :data:`DOWN_CAP` and the clamps). Everything added loses the other lines' grown polygons and the foreign
   components (grown; never the strip's spine, :data:`SPINE_ABOVE` / :data:`SPINE_BELOW`); protrusions beyond
   the strip narrower than :data:`STUB_WIDTH` line heights without own-connected ink are removed
   (:func:`remove_stubs`); slivers narrower than :data:`SLIVER` line heights are opened away; the outline of
   the piece holding the members is traced (:func:`mask_outline`). A single segment whose polygon would change
   by less than :data:`SLIVER` line heights in every column keeps its polygon exactly.
5. **Measures.** The share of each other line's (non-speck) polygon area newly covered (:func:`new_overlap`),
   ``own_ink_outside`` (:func:`own_ink_share`: the share of the counted or connected ink in the line's raise
   band, over its final x-range plus :data:`OWN_INK_REACH` line heights inside its column block and outside
   the other lines' polygons, that lies outside the final polygon) and the **end shift** of each end
   (:func:`end_shift`: how far detached ink an end took sits off the line's own ink, for the L-shape gate).
   Overlaps between the final polygons of a page are resolved by the aligner
   (:func:`~src.finetuning.kraken.page_layout.resolve_overlaps`).

Everything here works on numpy arrays and point lists; nothing reads files.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

Point = Tuple[float, float]

BAND_ABOVE = 0.75           # band: line heights above the baseline ...
BAND_BELOW = 0.3            # ... and below it
EXTEND_GAP = 0.4            # a blank run of this many line heights ends the walk
EXTEND_MAX = 2.0            # at most this many line heights of extension per end
MIN_COLUMN_INK = 0.03       # a band column holds ink with at least this share of a line height of ink pixels ...
MIN_COLUMN_PIXELS = 2       # ... and at least this many
WALK_MIN_AREA = 0.12        # the walk counts components of at least (this x line height)^2 pixels ...
CENTROID_ABOVE = 1.1        # ... whose centroid lies at most this many line heights above the baseline ...
CENTROID_BELOW = 0.5        # ... and at most this many below it (a raised final word counts, a neighbour line not) ...
REACH_ABOVE = 1.15          # ... and that reaches no higher than this many line heights above the baseline ...
REACH_BELOW = 0.8           # ... and no lower than this many below it (a neighbour line's tail reaching into the band)
MAX_INSIDE_OTHER = 0.2      # ... and with at most this share of its pixels inside another line's polygon
NEIGHBOUR_CLAMP = 0.15      # own band / raise: no higher than the previous baseline + this, no lower than the next - this
NEIGHBOUR_MIN_GAP = 0.5     # a line whose baseline lies closer than this (line heights) is not a neighbour above/below
NEIGHBOUR_REACH = 2.5       # a neighbour's baseline is carried this many line heights beyond its ends (its text goes on)
OTHER_DILATE = 2            # other lines' polygons grow by this many pixels before they are subtracted
BLOCK_SLACK = 0.3           # an end never moves more than this many line heights beyond the text block
BLOCK_LOW = 5.0             # text block: left margin = this percentile of the segmenter's line starts ...
BLOCK_HIGH = 95.0           # ... right margin = this percentile of their ends
BLOCK_MIN_LINES = 4         # a vertical third with fewer lines uses the page's block
END_WINDOW = 0.5            # the vertical extent at an end is the polygon's median over this many line heights
COMPONENT_PAD = 0.05        # counted components' pixels grow by this many line heights in the polygon
SLIVER = 0.1                # added parts narrower than this many line heights are removed
SPECK_SHARE = 0.05          # ink components smaller than (this x line height)^2 pixels are specks
MIN_SPECK_AREA = 6          # ... never below this many pixels
UP_CAP = 1.0                # counted components enter the polygon at most this many line heights above the baseline
DOWN_CAP = 0.5              # ... and below it
STRIP_PAD = 2               # extended ranges run this many pixels past the last ink column
MOUNT_HUE = (120, 200)      # blue mount: PIL HSV hue (0-255) range, about 170-280 degrees ...
MOUNT_MIN_SATURATION = 90   # ... saturation at least this (0-255) ...
MOUNT_MIN_VALUE = 25        # ... and value at least this (darker pixels fall to the darkness rule)
DARK_MAX_VALUE = 45         # black mount: value at most this, in a component touching the image border
MOUNT_DILATE = 3            # mount mask grown by this many pixels (edge shadow, JPEG fringes)
RIM_SHARE = 0.1             # rim zone: the mount grown by this share of a line height ...
RIM_MAX_SHARE = 0.5         # ... an ink component with at least this share of its pixels in it is a rim
BACKGROUND_SHARE = 0.6      # paper background: gray closing with a square of this share of a line height ...
MIN_BACKGROUND_KERNEL = 15  # ... at least this many pixels wide
MIN_CONTRAST = 12           # ink is at least this many gray levels darker than its background
ADAPTIVE_WINDOW = 25        # faint ink: local mean / std of the darkness over this many pixels square ...
ADAPTIVE_K = 1.0            # ... a pixel darker than mean + this x std ...
ADAPTIVE_OTSU_SHARE = 0.4   # ... and at least this share of the page's Otsu threshold (and MIN_CONTRAST) ...
ADAPTIVE_SEED = 0.2         # ... is ink when its piece touches an Otsu component of at least (this x lh)^2 pixels ...
ADAPTIVE_JUMP = 12          # ... grown by this many pixels (the local rule is blind right next to a dark stroke)
SPACING_ABOVE = 0.55        # raise: the top goes at most this share of the baseline spacing above the baseline ...
SPACING_BELOW = 0.45        # ... and the bottom at most this share below it
OWN_TOUCH = 2               # a component within this many pixels (8-connectivity) of the own segment ink joins it
PENCIL_SHARE = 0.5          # a non-own component with a median darkness or stroke width below this share of the
                            # line's own (pencil shelfmarks, faint annotations) is dropped
STUB_WIDTH = 0.3            # protrusions beyond the band narrower than this many line heights are removed unless own
OWN_DOWN_CAP = 0.9          # own-connected components enter down to this many line heights below the baseline ...
NEXT_TOP_CLEAR = 0.1        # ... staying this many line heights above the next line's detected ink top
DETECT_MIN = 0.6            # an unsegmented line's row-profile peak lies this many spacings from a baseline ...
DETECT_REACH = 1.6          # ... and at most this many above the first / below the last line
DETECT_PEAK = 0.3           # ... its peak reaches this share of the line's own body peak ...
DETECT_VALLEY = 0.5         # ... with valleys below this share of it on both sides
OWN_INK_REACH = 0.5         # own-ink measure: the line's final x-range plus this many line heights on each side
END_WINDOW_SHIFT = 0.25     # end shift: an end extended by at least this many line heights is measured
FOREIGN_REACH = 6.0         # the foreign test carries neighbours' baselines this many lh beyond their ends (a
                            # neighbour whose segment missed its first words still owns them)
OWN_EVIDENCE = 0.25         # own-ink measure: own ink below this share of the ink inside the polygon is no evidence
SEAM_BOX = 1.0              # a same-row box narrower than this (and lower than 1.2 x this) many lh inside the
                            # line's span (a speck boxed in a seam) is merged
SPINE_ABOVE = 0.15          # foreign ink never cuts the band strip between this many line heights above the
SPINE_BELOW = 0.05          # ... baseline and this many below it (merged pieces stay joined)


@dataclass(frozen=True)
class ExtentConfig:
    """Thresholds of the band walk and the polygon rebuild, in line heights unless stated.

    :param above: :data:`BAND_ABOVE`.
    :param below: :data:`BAND_BELOW`.
    :param gap: :data:`EXTEND_GAP`.
    :param max_extend: :data:`EXTEND_MAX` (0 measures the ink beyond the original ends without extending).
    :param min_column_ink: :data:`MIN_COLUMN_INK`.
    :param min_column_pixels: :data:`MIN_COLUMN_PIXELS` (pixels).
    :param walk_min_area: :data:`WALK_MIN_AREA`.
    :param centroid_above: :data:`CENTROID_ABOVE`.
    :param centroid_below: :data:`CENTROID_BELOW`.
    :param reach_above: :data:`REACH_ABOVE`.
    :param reach_below: :data:`REACH_BELOW`.
    :param up_cap: :data:`UP_CAP`.
    :param down_cap: :data:`DOWN_CAP`.
    :param strip_pad: :data:`STRIP_PAD` (pixels).
    :param raise_top: Rebuild every column of the line to cover the band (the whole-line vertical rebuild);
        off by default here, the aligner turns it on with its extension.
    :param clamp: :data:`NEIGHBOUR_CLAMP`.
    :param min_neighbour_gap: :data:`NEIGHBOUR_MIN_GAP`.
    :param neighbour_reach: :data:`NEIGHBOUR_REACH`.
    :param block_slack: :data:`BLOCK_SLACK` (``inf``: no block bound).
    :param other_dilate: :data:`OTHER_DILATE` (pixels).
    :param max_inside_other: :data:`MAX_INSIDE_OTHER`.
    :param sliver: :data:`SLIVER`.
    :param end_window: :data:`END_WINDOW`.
    :param component_pad: :data:`COMPONENT_PAD`.
    :param spacing_above: :data:`SPACING_ABOVE` (share of the baseline spacing).
    :param spacing_below: :data:`SPACING_BELOW` (share of the baseline spacing).
    :param own_touch: :data:`OWN_TOUCH` (pixels).
    :param pencil_share: :data:`PENCIL_SHARE` (0 switches the pencil filter off).
    :param stub_width: :data:`STUB_WIDTH` (0 switches stub removal off).
    :param own_down_cap: :data:`OWN_DOWN_CAP`.
    :param next_top_clear: :data:`NEXT_TOP_CLEAR`.
    :param detect: Detect unsegmented neighbour lines from the ink row profile (:func:`detect_rows`).
    :param foreign: Components whose centroid is nearer another (segmented or detected) baseline are never
        this line's ink (:func:`foreign_components`).
    :param own_ink_reach: :data:`OWN_INK_REACH` (``inf``: the whole text block, the round-2 measure).
    :param foreign_reach: :data:`FOREIGN_REACH`.
    """
    above: float = BAND_ABOVE
    below: float = BAND_BELOW
    gap: float = EXTEND_GAP
    max_extend: float = EXTEND_MAX
    min_column_ink: float = MIN_COLUMN_INK
    min_column_pixels: int = MIN_COLUMN_PIXELS
    walk_min_area: float = WALK_MIN_AREA
    centroid_above: float = CENTROID_ABOVE
    centroid_below: float = CENTROID_BELOW
    reach_above: float = REACH_ABOVE
    reach_below: float = REACH_BELOW
    up_cap: float = UP_CAP
    down_cap: float = DOWN_CAP
    strip_pad: int = STRIP_PAD
    raise_top: bool = False
    clamp: float = NEIGHBOUR_CLAMP
    min_neighbour_gap: float = NEIGHBOUR_MIN_GAP
    neighbour_reach: float = NEIGHBOUR_REACH
    block_slack: float = BLOCK_SLACK
    other_dilate: int = OTHER_DILATE
    max_inside_other: float = MAX_INSIDE_OTHER
    sliver: float = SLIVER
    end_window: float = END_WINDOW
    component_pad: float = COMPONENT_PAD
    spacing_above: float = SPACING_ABOVE
    spacing_below: float = SPACING_BELOW
    own_touch: int = OWN_TOUCH
    pencil_share: float = PENCIL_SHARE
    stub_width: float = STUB_WIDTH
    own_down_cap: float = OWN_DOWN_CAP
    next_top_clear: float = NEXT_TOP_CLEAR
    detect: bool = True
    foreign: bool = True
    own_ink_reach: float = OWN_INK_REACH
    foreign_reach: float = FOREIGN_REACH


@dataclass(frozen=True)
class PageInk:
    """Labelled ink components of a page.

    :param labels: ``(H, W)`` int32, 0 = no ink, ``k`` = component ``k``.
    :param area: Pixels per label (index = label; ``area[0]`` is unused).
    :param centroid_x: Centroid column per label.
    :param centroid_y: Centroid row per label.
    :param bbox: ``(x0, y0, x1, y1)`` per label, ends exclusive (index = label; row 0 unused).
    :param stroke: Stroke width estimate per label in pixels (:func:`stroke_widths`; None: not measured).
    :param darkness: Median darkness below the paper background per label (None: no image, e.g. a mask).
    """
    labels: np.ndarray
    area: np.ndarray
    centroid_x: np.ndarray
    centroid_y: np.ndarray
    bbox: np.ndarray
    stroke: Optional[np.ndarray] = None
    darkness: Optional[np.ndarray] = None

    @property
    def shape(self) -> Tuple[int, int]:
        """Image shape.

        :return: ``(height, width)``.
        :rtype: Tuple[int, int]
        """
        return self.labels.shape

    @property
    def mask(self) -> np.ndarray:
        """Boolean ink mask.

        :return: ``labels > 0``.
        :rtype: np.ndarray
        """
        return self.labels > 0


@dataclass(frozen=True)
class Box:
    """Another segmenter line on the page, as the walk sees it.

    :param key: Segmenter index.
    :param polygon: Boundary polygon.
    :param steal: Absorbing it would take another edition line's text (it is placed on another edition line,
        or its read matches one).
    :param baseline: Its baseline; empty when the line must not clamp this line's band (noise, rotated lines).
    :param absorbable: The walk may absorb it (False: its read has no counterpart at the edition line's end, or
        the end must not take a detached glyph).
    """
    key: int
    polygon: Tuple[Point, ...]
    steal: bool = False
    baseline: Tuple[Point, ...] = ()
    absorbable: bool = True

    def bbox(self) -> Tuple[float, float, float, float]:
        """Bounding box.

        :return: ``(x_min, y_min, x_max, y_max)``.
        :rtype: Tuple[float, float, float, float]
        """
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return min(xs), min(ys), max(xs), max(ys)

    def centre(self) -> Point:
        """Bounding box centre.

        :return: ``(x, y)``.
        :rtype: Point
        """
        x0, y0, x1, y1 = self.bbox()
        return (x0 + x1) / 2, (y0 + y1) / 2


@dataclass(frozen=True)
class EndExtent:
    """What the walk found beyond one end of a line.

    :param extended_px: Pixels the end moved outward (0 when nothing was found or extension is off).
    :param ink_beyond_px: Pixels of ink still continuing beyond the final end (same gap rule, inside the text
        block when one bounds the walk).
    :param absorbed: Keys of the boxes absorbed.
    :param stolen: Keys of reachable boxes flagged ``steal`` (not absorbed).
    :param capped: The walk stopped at the extension cap.
    :param components: Ink component labels the walk counted inside the extended range.
    :param limited: The text block stopped the walk while ink continued beyond it.
    :param past_block_px: Pixels of that ink beyond the block bound (not gated: it is not this line's ink).
    :param crossed_px: Pixels the end moved beyond the block bound along ink connected to the line's own letters.
    """
    extended_px: int = 0
    ink_beyond_px: int = 0
    absorbed: Tuple[int, ...] = ()
    stolen: Tuple[int, ...] = ()
    capped: bool = False
    components: Tuple[int, ...] = ()
    limited: bool = False
    past_block_px: int = 0
    crossed_px: int = 0


@dataclass(frozen=True)
class LineExtent:
    """A line's geometry after the walk at both ends.

    :param baseline: Baseline (end points moved with the polygon ends).
    :param boundary: Polygon (the original one when nothing changed).
    :param left: Left end.
    :param right: Right end.
    :param rebuilt: The polygon was rebuilt (an end moved, a box was absorbed, the line has several members or
        the vertical rebuild changed a column by at least :data:`SLIVER` line heights).
    :param hull_fallback: The rebuilt shape fell apart into several pieces and their convex hull was used.
    :param raised: The vertical rebuild (``raise_top``) ran and the polygon was rebuilt.
    :param overlap: Largest share of another line's polygon area that the final polygon newly covers (beyond
        what the members already covered).
    :param overlap_key: That line's key (None without overlap).
    :param own_ink_outside: Share of the ink in the line's band (over its final x-range plus
        :data:`OWN_INK_REACH` line heights, inside its column block, outside the other lines' polygons, own and
        neighbouring components only) that lies outside the final polygon (:func:`own_ink_share`).
    :param block: ``(left, right)`` text block bound used for this line (None: unbounded).
    :param overlap_final: Largest share of the smaller polygon shared with another written line's final polygon
        (set by the page-level pass, :func:`~src.finetuning.kraken.page_layout.resolve_overlaps`).
    :param overlap_final_key: That line's key.
    :param trimmed: The page-level pass trimmed a shared region from this polygon.
    :param trim_cut_own: Trimming the shared region would cut this line's own ink (the line is rejected).
    :param end_shift: ``(left, right)``: how far the detached ink an end took (walk components not connected to
        the line's own letters, absorbed boxes) sits below (+) or above (-) the line's own ink at that end, in
        line heights relative to the sloped baseline (0.0 when the end took none).
    :param spacing: Baseline spacing used for the raise (pixels).
    :param detected: ``(above, below)``: an unsegmented neighbour line was detected from the row profile.
    :param foreign_dropped: Components in the window nearer another baseline (never this line's ink).
    :param pencil_dropped: Non-own components dropped as pencil / annotation (darkness or stroke width).
    :param stubs_removed: Narrow protrusions removed from the polygon.
    :param forced: Boxes merged unconditionally (a supralinear insertion's segment, or a box narrower than
        :data:`SEAM_BOX` line heights and lower than 1.2 of them on the line's row inside its span, e.g. a speck
        boxed in a merged line's seam).
    :param hard: ``(left, right)`` hard limits (margin material, gutters) used for this line.
    """
    baseline: Tuple[Point, ...]
    boundary: Tuple[Point, ...]
    left: EndExtent = field(default_factory=EndExtent)
    right: EndExtent = field(default_factory=EndExtent)
    rebuilt: bool = False
    hull_fallback: bool = False
    raised: bool = False
    overlap: float = 0.0
    overlap_key: Optional[int] = None
    own_ink_outside: float = 0.0
    block: Optional[Tuple[float, float]] = None
    overlap_final: float = 0.0
    overlap_final_key: Optional[int] = None
    trimmed: bool = False
    trim_cut_own: bool = False
    end_shift: Tuple[float, float] = (0.0, 0.0)
    spacing: float = 0.0
    detected: Tuple[bool, bool] = (False, False)
    foreign_dropped: int = 0
    pencil_dropped: int = 0
    stubs_removed: int = 0
    forced: Tuple[int, ...] = ()
    hard: Optional[Tuple[float, float]] = None

    @property
    def absorbed(self) -> Tuple[int, ...]:
        """Boxes absorbed at either end.

        :return: Keys, left end first.
        :rtype: Tuple[int, ...]
        """
        return self.left.absorbed + self.right.absorbed

    @property
    def stolen(self) -> Tuple[int, ...]:
        """Reachable boxes of other edition lines at either end.

        :return: Keys, left end first.
        :rtype: Tuple[int, ...]
        """
        return self.left.stolen + self.right.stolen


# ----------------------------------------------------------------------------- binarisation

def otsu_threshold(hist: np.ndarray) -> int:
    """Otsu threshold of a gray-level histogram.

    :param hist: Counts per level (index = gray level).
    :type hist: np.ndarray
    :return: Level ``t`` maximising the between-class variance of ``<= t`` vs ``> t``; ``-1`` when the
        histogram holds fewer than two distinct levels (nothing to separate).
    :rtype: int
    """
    hist = np.asarray(hist, dtype=np.float64)
    total = hist.sum()
    levels = np.arange(hist.size, dtype=np.float64)
    w0 = np.cumsum(hist)
    w1 = total - w0
    m0 = np.cumsum(hist * levels)
    valid = (w0 > 0) & (w1 > 0)
    if not valid.any():
        return -1
    mu0 = np.divide(m0, w0, out=np.zeros_like(m0), where=w0 > 0)
    mu1 = np.divide(m0[-1] - m0, w1, out=np.zeros_like(m0), where=w1 > 0)
    between = np.where(valid, w0 * w1 * (mu0 - mu1) ** 2, -1.0)
    return int(np.argmax(between))


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Grow a boolean mask by a square of ``2 * radius + 1`` pixels.

    :param mask: Boolean image.
    :type mask: np.ndarray
    :param radius: Pixels (0 or less: a copy).
    :type radius: int
    :return: Dilated mask.
    :rtype: np.ndarray
    """
    if radius <= 0 or not mask.any():
        return mask.copy()
    element = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), element) > 0


def mount_mask(hsv: np.ndarray, grow: int = MOUNT_DILATE) -> np.ndarray:
    """Pixels of the mount: the blue or black board the fragment is mounted on.

    :param hsv: ``(H, W, 3)`` uint8 array in PIL's HSV (hue, saturation, value all 0-255).
    :type hsv: np.ndarray
    :param grow: Grow the mask by this many pixels.
    :type grow: int
    :return: Boolean ``(H, W)``: blue (hue in :data:`MOUNT_HUE`, saturation >= :data:`MOUNT_MIN_SATURATION`,
        value >= :data:`MOUNT_MIN_VALUE`) or dark (value <= :data:`DARK_MAX_VALUE`) in a dark component that
        touches the image border.
    :rtype: np.ndarray
    """
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    blue = (hue >= MOUNT_HUE[0]) & (hue <= MOUNT_HUE[1]) & (sat >= MOUNT_MIN_SATURATION) & (val >= MOUNT_MIN_VALUE)
    dark = val <= DARK_MAX_VALUE
    labels, n = ndimage.label(dark, structure=np.ones((3, 3), bool))
    if n:
        border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        dark = np.isin(labels, border[border > 0])
    return dilate(blue | dark, grow)


def remove_specks(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop connected components (8-connectivity) smaller than ``min_area`` pixels.

    :param mask: Boolean image.
    :type mask: np.ndarray
    :param min_area: Smallest component kept.
    :type min_area: int
    :return: The cleaned mask (a copy).
    :rtype: np.ndarray
    """
    if min_area <= 1 or not mask.any():
        return mask.copy()
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    keep = sizes >= min_area
    keep[0] = False
    return keep[labels]


def remove_rims(mask: np.ndarray, mount: np.ndarray, radius: int, max_share: float = RIM_MAX_SHARE) -> np.ndarray:
    """Drop ink components that lie mostly next to the mount (the dark rim of a hole or of the fragment edge).

    :param mask: Boolean ink.
    :type mask: np.ndarray
    :param mount: Boolean mount.
    :type mount: np.ndarray
    :param radius: Width of the zone around the mount in pixels.
    :type radius: int
    :param max_share: A component with at least this share of its pixels in the zone is dropped.
    :type max_share: float
    :return: The cleaned mask (a copy).
    :rtype: np.ndarray
    """
    if radius <= 0 or not mask.any() or not mount.any():
        return mask.copy()
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    zone = dilate(mount, radius)
    area = np.bincount(labels.ravel(), minlength=n + 1)
    near = np.bincount(labels[zone], minlength=n + 1)
    keep = near < max_share * np.maximum(area, 1)
    keep[0] = False
    return keep[labels]


def speck_area(line_height: float) -> int:
    """Smallest ink component kept for a page's line height.

    :param line_height: Median line height in pixels.
    :type line_height: float
    :return: ``max(MIN_SPECK_AREA, round((SPECK_SHARE * line_height) ** 2))``.
    :rtype: int
    """
    return max(MIN_SPECK_AREA, int(round((SPECK_SHARE * line_height) ** 2)))


def background_darkness(gray: np.ndarray, mount: np.ndarray, kernel: int) -> np.ndarray:
    """How much darker each pixel is than the paper around it.

    :param gray: ``(H, W)`` uint8 grayscale.
    :type gray: np.ndarray
    :param mount: Boolean mount mask (painted with the median paper gray first, so the dark board does not
        pull the background down).
    :type mount: np.ndarray
    :param kernel: Side of the square closing element in pixels (wider than a pen stroke).
    :type kernel: int
    :return: ``(H, W)`` uint8: closing(gray) - gray, clipped at 0.
    :rtype: np.ndarray
    """
    gray = gray.copy()
    if mount.any() and not mount.all():
        gray[mount] = np.uint8(np.median(gray[~mount]))
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, element)
    return (background.astype(np.int16) - gray).clip(0, 255).astype(np.uint8)


def adaptive_ink(dark: np.ndarray, mount: np.ndarray, floor: float, window: int = ADAPTIVE_WINDOW,
                 k: float = ADAPTIVE_K) -> np.ndarray:
    """Locally dark pixels: the mean - k x std rule on the background-normalised gray, for faint ink.

    :param dark: ``(H, W)`` darkness below the paper background (:func:`background_darkness`).
    :type dark: np.ndarray
    :param mount: Boolean mount mask (never ink; its darkness is zeroed before the local statistics).
    :type mount: np.ndarray
    :param floor: Smallest darkness that can be ink (keeps paper grain out where the window is flat).
    :type floor: float
    :param window: Side of the square window in pixels.
    :type window: int
    :param k: Standard deviations above the local mean.
    :type k: float
    :return: Boolean ``(H, W)``: ``dark > mean + k * std`` over the window and ``dark >= floor``, off the mount.
    :rtype: np.ndarray
    """
    values = dark.astype(np.float32)
    values[mount] = 0.0
    size = (int(window), int(window))
    mean = cv2.boxFilter(values, -1, size, normalize=True, borderType=cv2.BORDER_REFLECT)
    square = cv2.boxFilter(values * values, -1, size, normalize=True, borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(square - mean * mean, 0.0))
    return (values > mean + k * std) & (values >= floor) & ~mount


def hysteresis(strong: np.ndarray, weak: np.ndarray, seed_area: int, jump: int = 0) -> np.ndarray:
    """Strong pixels plus the pieces of weak pixels that touch a large enough strong component.

    :param strong: Boolean confident ink.
    :type strong: np.ndarray
    :param weak: Boolean candidate ink (faint).
    :type weak: np.ndarray
    :param seed_area: Smallest strong component (pixels) that can recruit weak pieces.
    :type seed_area: int
    :param jump: The seeds are grown by this many pixels for the touch test, so a faint tail that the local
        rule cannot see right next to its dark stroke still joins it.
    :type jump: int
    :return: ``strong`` plus every 8-connected piece of ``weak`` (outside ``strong``) that meets a grown seed.
    :rtype: np.ndarray
    """
    seeds = remove_specks(strong, seed_area)
    weak = weak & ~strong
    if not seeds.any() or not weak.any():
        return strong.copy()
    labels, n = ndimage.label(weak, structure=np.ones((3, 3), bool))
    hit = np.bincount(labels[dilate(seeds, jump)], minlength=n + 1) > 0
    hit[0] = False
    return strong | hit[labels]


def ink_mask(image: Image.Image, line_height: float, mask_mount: bool = True, adaptive: bool = True) -> np.ndarray:
    """Binarised ink of a page: Otsu (plus a local adaptive rule for faint ink) on the background-normalised
    grayscale, mount, rims and specks removed (:func:`ink_layers` without the darkness image).

    :param image: Page image (any mode).
    :type image: Image.Image
    :param line_height: Median line height of the page in pixels.
    :type line_height: float
    :param mask_mount: Mask the mount and drop the rims next to it.
    :type mask_mount: bool
    :param adaptive: Add faint ink connected to letter-sized Otsu ink.
    :type adaptive: bool
    :return: Boolean ``(H, W)``, True for ink.
    :rtype: np.ndarray
    """
    return ink_layers(image, line_height, mask_mount, adaptive)[0]


def ink_layers(image: Image.Image, line_height: float, mask_mount: bool = True, adaptive: bool = True
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Binarised ink of a page and the darkness image it was thresholded on.

    :param image: Page image (any mode).
    :type image: Image.Image
    :param line_height: Median line height of the page in pixels (sets the background kernel, the rim zone
        and the speck size, :func:`speck_area`).
    :type line_height: float
    :param mask_mount: Mask the mount (:func:`mount_mask`) and drop the rims next to it (:func:`remove_rims`).
    :type mask_mount: bool
    :param adaptive: Add the locally dark pixels of :func:`adaptive_ink` (floor: :data:`ADAPTIVE_OTSU_SHARE`
        of the Otsu threshold, at least :data:`MIN_CONTRAST`) whose pieces touch letter-sized Otsu ink
        (:func:`hysteresis`: seeds of at least ``(ADAPTIVE_SEED x line height)^2`` pixels, grown by
        :data:`ADAPTIVE_JUMP`).
    :type adaptive: bool
    :return: ``(ink, dark)``: boolean ``(H, W)``, True for ink: darkness below the paper background above the
        Otsu threshold of the non-mount pixels' darkness (at least :data:`MIN_CONTRAST`), or locally dark and
        connected to such ink, outside the mount; and the uint8 darkness image (:func:`background_darkness`).
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    rgb = image.convert("RGB")
    gray = np.asarray(rgb.convert("L"))
    mount = mount_mask(np.asarray(rgb.convert("HSV"))) if mask_mount else np.zeros(gray.shape, bool)
    kernel = max(MIN_BACKGROUND_KERNEL, int(round(BACKGROUND_SHARE * line_height)))
    dark = background_darkness(gray, mount, kernel)
    paper = dark[~mount]
    threshold = otsu_threshold(np.bincount(paper, minlength=256)) if paper.size else -1
    ink = (dark > max(threshold, MIN_CONTRAST)) & ~mount
    if adaptive and threshold >= 0:
        weak = adaptive_ink(dark, mount, max(MIN_CONTRAST, ADAPTIVE_OTSU_SHARE * threshold))
        seed = max(speck_area(line_height), int(round((ADAPTIVE_SEED * line_height) ** 2)))
        ink = hysteresis(ink, weak, seed, ADAPTIVE_JUMP)
    ink = remove_specks(ink, speck_area(line_height))
    return remove_rims(ink, mount, max(MOUNT_DILATE, int(round(RIM_SHARE * line_height)))), dark


def stroke_widths(labels: np.ndarray, n: int) -> np.ndarray:
    """Stroke width of every component: four times its pixels' mean distance to the background.

    The mean distance-to-background over a stroke of width ``w`` is about ``w / 4`` (``w / 6`` for a round
    dot), so the estimate compares strokes of one pen consistently; a pencil line or a fine annotation hand
    scores far below a reed pen.

    :param labels: ``(H, W)`` component labels (0 = background).
    :type labels: np.ndarray
    :param n: Number of labels.
    :type n: int
    :return: Width estimate per label in pixels (index = label; 0.0 for label 0).
    :rtype: np.ndarray
    """
    if n == 0:
        return np.zeros(1)
    dist = cv2.distanceTransform((labels > 0).astype(np.uint8), cv2.DIST_L2, 3)
    flat = labels.ravel()
    area = np.maximum(np.bincount(flat, minlength=n + 1), 1)
    out = 4.0 * np.bincount(flat, weights=dist.ravel(), minlength=n + 1) / area
    out[0] = 0.0
    return out


def component_darkness(labels: np.ndarray, n: int, dark: np.ndarray) -> np.ndarray:
    """Median darkness (below the paper background) of every component.

    :param labels: ``(H, W)`` component labels.
    :type labels: np.ndarray
    :param n: Number of labels.
    :type n: int
    :param dark: ``(H, W)`` darkness image (:func:`background_darkness`).
    :type dark: np.ndarray
    :return: Median darkness per label (index = label; 0.0 for label 0).
    :rtype: np.ndarray
    """
    out = np.zeros(n + 1)
    if n == 0:
        return out
    sel = labels > 0
    lab = labels[sel]
    val = dark[sel].astype(np.int64)
    order = np.lexsort((val, lab))
    lab, val = lab[order], val[order]
    starts = np.searchsorted(lab, np.arange(1, n + 1), side="left")
    ends = np.searchsorted(lab, np.arange(1, n + 1), side="right")
    present = ends > starts
    mid = (starts + ends - 1) // 2
    out[1:][present] = val[mid[present]]
    return out


def label_ink(mask: np.ndarray, dark: Optional[np.ndarray] = None) -> PageInk:
    """Label the connected components (8-connectivity) of an ink mask.

    :param mask: Boolean ink.
    :type mask: np.ndarray
    :param dark: Darkness image of the page (None: component darkness is not measured).
    :type dark: Optional[np.ndarray]
    :return: Labels, areas, centroids, bounding boxes, stroke widths (and darkness).
    :rtype: PageInk
    """
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    labels = labels.astype(np.int32)
    flat = labels.ravel()
    area = np.bincount(flat, minlength=n + 1).astype(np.float64)
    rows, cols = np.indices(labels.shape)
    safe = np.maximum(area, 1.0)
    cy = np.bincount(flat, weights=rows.ravel(), minlength=n + 1) / safe
    cx = np.bincount(flat, weights=cols.ravel(), minlength=n + 1) / safe
    bbox = np.zeros((n + 1, 4), dtype=np.int64)
    for k, sl in enumerate(ndimage.find_objects(labels), 1):
        if sl is not None:
            bbox[k] = (sl[1].start, sl[0].start, sl[1].stop, sl[0].stop)
    darkness = component_darkness(labels, n, dark) if dark is not None else None
    return PageInk(labels, area, cx, cy, bbox, stroke_widths(labels, n), darkness)


def page_ink(image: Image.Image, line_height: float, mask_mount: bool = True) -> PageInk:
    """Labelled ink of a page (:func:`ink_layers` then :func:`label_ink`, with component darkness).

    :param image: Page image.
    :type image: Image.Image
    :param line_height: Median line height in pixels.
    :type line_height: float
    :param mask_mount: Mask the mount.
    :type mask_mount: bool
    :return: The page's ink components.
    :rtype: PageInk
    """
    ink, dark = ink_layers(image, line_height, mask_mount)
    return label_ink(ink, dark)


# ----------------------------------------------------------------------------- text block

@dataclass(frozen=True)
class TextBlock:
    """Left and right margins of a page's text column, from the segmenter's own line ends.

    :param left: Page-wide left margin (x).
    :param right: Page-wide right margin (x).
    :param thirds: ``(y_from, y_to, left, right)`` per vertical third of the lines (empty: page-wide only).
    """
    left: float
    right: float
    thirds: Tuple[Tuple[float, float, float, float], ...] = ()

    def bounds(self, y: float) -> Tuple[float, float]:
        """Margins at a height.

        :param y: Row (a line's mean baseline y).
        :type y: float
        :return: ``(left, right)`` of the third holding ``y`` (the first or last third beyond the lines'
            range), else the page-wide margins.
        :rtype: Tuple[float, float]
        """
        for y_from, y_to, left, right in self.thirds:
            if y <= y_to:
                return left, right
        if self.thirds:
            return self.thirds[-1][2], self.thirds[-1][3]
        return self.left, self.right


def text_block(spans: Sequence[Tuple[float, float, float]], thirds: bool = False, low: float = BLOCK_LOW,
               high: float = BLOCK_HIGH, min_lines: int = BLOCK_MIN_LINES) -> Optional[TextBlock]:
    """Text column margins from the segmenter's line ends (robust percentiles).

    Lines that do not overlap the page's core column (from the median line start to the median line end) are
    left out first: a margin note, a seal or a margin column beside the text does not widen the block.

    :param spans: ``(x_start, x_end, y)`` per segmenter line (polygon x-range, mean baseline y).
    :type spans: Sequence[Tuple[float, float, float]]
    :param thirds: Also estimate the margins per vertical third of the lines (follows a slanted block).
    :type thirds: bool
    :param low: Percentile of the line starts that is the left margin.
    :type low: float
    :param high: Percentile of the line ends that is the right margin.
    :type high: float
    :param min_lines: A third with fewer lines uses the page-wide margins.
    :type min_lines: int
    :return: The block, or None without lines.
    :rtype: Optional[TextBlock]
    """
    if not spans:
        return None
    arr = np.asarray(spans, dtype=np.float64)
    starts, ends, ys = arr[:, 0], arr[:, 1], arr[:, 2]
    core_lo, core_hi = float(np.median(starts)), float(np.median(ends))
    keep = (ends > core_lo) & (starts < core_hi) if core_lo < core_hi else np.ones(len(arr), bool)
    if not keep.any():
        keep = np.ones(len(arr), bool)
    left, right = float(np.percentile(starts[keep], low)), float(np.percentile(ends[keep], high))
    bands: List[Tuple[float, float, float, float]] = []
    if thirds:
        edges = np.linspace(ys[keep].min(), ys[keep].max(), 4)
        for i in range(3):
            upper = (ys <= edges[i + 1]) if i == 2 else (ys < edges[i + 1])
            sel = keep & (ys >= edges[i]) & upper
            if sel.sum() >= min_lines:
                bands.append((float(edges[i]), float(edges[i + 1]), float(np.percentile(starts[sel], low)),
                              float(np.percentile(ends[sel], high))))
            else:
                bands.append((float(edges[i]), float(edges[i + 1]), left, right))
    return TextBlock(left, right, tuple(bands))


# ----------------------------------------------------------------------------- band

def baseline_y(baseline: Sequence[Point], xs: np.ndarray) -> np.ndarray:
    """Baseline height at the given columns.

    :param baseline: Baseline points (any x order; the points are sorted by x).
    :type baseline: Sequence[Point]
    :param xs: Columns.
    :type xs: np.ndarray
    :return: Linear interpolation between the points; beyond the ends, extrapolation along the end-to-end
        chord (flat for a one-point or vertical baseline).
    :rtype: np.ndarray
    """
    pts = sorted(baseline)
    px = np.array([p[0] for p in pts], dtype=np.float64)
    py = np.array([p[1] for p in pts], dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.interp(xs, px, py)
    dx = px[-1] - px[0]
    slope = (py[-1] - py[0]) / dx if dx > 0 else 0.0
    ys = np.where(xs < px[0], py[0] + slope * (xs - px[0]), ys)
    return np.where(xs > px[-1], py[-1] + slope * (xs - px[-1]), ys)


def band_limits(baseline: Sequence[Point], xs: np.ndarray, line_height: float, above: float = BAND_ABOVE,
                below: float = BAND_BELOW) -> Tuple[np.ndarray, np.ndarray]:
    """Top and bottom rows of the line band at the given columns.

    :param baseline: Baseline points.
    :type baseline: Sequence[Point]
    :param xs: Columns.
    :type xs: np.ndarray
    :param line_height: Line height in pixels.
    :type line_height: float
    :param above: Line heights above the baseline.
    :type above: float
    :param below: Line heights below the baseline.
    :type below: float
    :return: ``(top, bottom)`` float arrays.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    ys = baseline_y(baseline, xs)
    return ys - above * line_height, ys + below * line_height


def in_band(baseline: Sequence[Point], point: Point, line_height: float, above: float = BAND_ABOVE,
            below: float = BAND_BELOW) -> bool:
    """Whether a point lies inside the line band.

    :param baseline: Baseline points.
    :type baseline: Sequence[Point]
    :param point: ``(x, y)``.
    :type point: Point
    :param line_height: Line height in pixels.
    :type line_height: float
    :param above: Line heights above the baseline.
    :type above: float
    :param below: Line heights below the baseline.
    :type below: float
    :return: True when ``top <= y <= bottom`` at ``x``.
    :rtype: bool
    """
    top, bottom = band_limits(baseline, np.array([point[0]]), line_height, above, below)
    return bool(top[0] <= point[1] <= bottom[0])


def polygons_window(polygons: Sequence[Sequence[Point]], x0: int, y0: int, width: int, height: int) -> np.ndarray:
    """Rasterise polygons into a window of the page.

    :param polygons: Polygons in page coordinates.
    :type polygons: Sequence[Sequence[Point]]
    :param x0: Window left column.
    :type x0: int
    :param y0: Window top row.
    :type y0: int
    :param width: Window width.
    :type width: int
    :param height: Window height.
    :type height: int
    :return: Boolean ``(height, width)``, True inside any polygon.
    :rtype: np.ndarray
    """
    canvas = Image.new("1", (max(1, width), max(1, height)), 0)
    draw = ImageDraw.Draw(canvas)
    for poly in polygons:
        if len(poly) >= 3:
            draw.polygon([(x - x0, y - y0) for x, y in poly], fill=1)
    return np.asarray(canvas, dtype=bool)[:height, :width]


def polygon_area(polygon: Sequence[Point]) -> float:
    """Area of a simple polygon (shoelace formula).

    :param polygon: Points.
    :type polygon: Sequence[Point]
    :return: Absolute area in square pixels (0.0 for fewer than three points).
    :rtype: float
    """
    if len(polygon) < 3:
        return 0.0
    xs = np.array([p[0] for p in polygon], dtype=np.float64)
    ys = np.array([p[1] for p in polygon], dtype=np.float64)
    return float(abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))) / 2)


def scan_run(inked: Sequence[bool], gap: int, limit: int) -> int:
    """Length of the ink run that starts at index 0, bridging blank runs shorter than ``gap``.

    :param inked: Whether each column outward from an end holds ink (index 0 = first column outside).
    :type inked: Sequence[bool]
    :param gap: A blank run of this many columns ends the walk (the blank run before the first ink counts).
    :type gap: int
    :param limit: Columns the walk may cover at most.
    :type limit: int
    :return: Index of the last ink column reached plus one (0 when none is reached).
    :rtype: int
    """
    last, blank = -1, 0
    for i in range(min(limit, len(inked))):
        if inked[i]:
            last, blank = i, 0
        else:
            blank += 1
            if blank >= gap:
                break
    return last + 1


# ----------------------------------------------------------------------------- scene

@dataclass(frozen=True, eq=False)
class LineScene:
    """Raster context of one line: a window of the page (coordinates relative to ``x0``, ``y0``).

    :param x0: Window left column on the page.
    :param y0: Window top row on the page.
    :param labels: ``(h, w)`` ink labels of the window.
    :param base: ``(w,)`` the line's baseline row per window column (page rows).
    :param prev: ``(w,)`` baseline row of the nearest line above per column (page rows; NaN: none).
    :param next: ``(w,)`` baseline row of the nearest line below per column (page rows; NaN: none).
    :param counted: Per page label: the component counts as this line's ink.
    :param boxes: ``{key: (row, col, mask)}``: each other line's polygon rasterised in its bounding box (window
        coordinates of the box's top-left corner).
    :param others: ``(h, w)`` union of the other lines' polygons.
    :param blocked: ``others`` grown by ``other_dilate`` pixels.
    :param own: ``(h, w)`` counted or own-connected ink outside ``blocked`` (what the walk sees).
    :param line_height: Line height in pixels.
    :param spacing: Baseline spacing in pixels (the raise's unit).
    :param foreign: Per page label: the component's centroid is nearer another (segmented or detected)
        baseline than this line's (:func:`foreign_components`).
    :param connected: Per page label: the component holds the line's own segment ink or touches it within
        ``own_touch`` pixels, and is not foreign.
    :param pencil: Per page label: a component (not connected) whose darkness or stroke width is below
        ``pencil_share`` of the line's own median (:func:`pencil_components`).
    :param next_top: ``(w,)`` page row of the next line's ink top per column (``inf``: none in the window).
    :param members: ``(h, w)`` the line's own polygons (all False when the scene was built without them).
    :param detected: ``(above, below)``: an unsegmented neighbour line was detected.
    :param beyond: Per page label: a counted component left out because its centroid lies beyond the walk's
        bounds (measured as ink past the block only).
    """
    x0: int
    y0: int
    labels: np.ndarray
    base: np.ndarray
    prev: np.ndarray
    next: np.ndarray
    counted: np.ndarray
    boxes: Dict[int, Tuple[int, int, np.ndarray]]
    others: np.ndarray
    blocked: np.ndarray
    own: np.ndarray
    line_height: float
    spacing: float = 0.0
    foreign: Optional[np.ndarray] = None
    connected: Optional[np.ndarray] = None
    pencil: Optional[np.ndarray] = None
    next_top: Optional[np.ndarray] = None
    members: Optional[np.ndarray] = None
    detected: Tuple[bool, bool] = (False, False)
    beyond: Optional[np.ndarray] = None

    def label_mask(self, per_label: Optional[np.ndarray]) -> np.ndarray:
        """Pixels of the window whose component is flagged.

        :param per_label: Boolean per page label (None: nothing flagged).
        :type per_label: Optional[np.ndarray]
        :return: Boolean ``(h, w)``.
        :rtype: np.ndarray
        """
        if per_label is None:
            return np.zeros(self.shape, bool)
        return per_label[self.labels]

    @property
    def shape(self) -> Tuple[int, int]:
        """Window shape.

        :return: ``(h, w)``.
        :rtype: Tuple[int, int]
        """
        return self.labels.shape

    def clamps(self, clamp: float) -> Tuple[np.ndarray, np.ndarray]:
        """Highest and lowest rows the line may take per column (the neighbour clamps).

        :param clamp: Line heights inside the neighbours' baselines.
        :type clamp: float
        :return: ``(top, bottom)`` page rows per window column (``-inf`` / ``inf`` without a neighbour).
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        off = clamp * self.line_height
        top = np.where(np.isnan(self.prev), -np.inf, self.prev + off)
        bottom = np.where(np.isnan(self.next), np.inf, self.next - off)
        return top, bottom

    def others_except(self, keys: Set[int], grow: int) -> np.ndarray:
        """The other lines' polygons without some of them, grown.

        :param keys: Lines left out (absorbed boxes).
        :type keys: Set[int]
        :param grow: Pixels of dilation.
        :type grow: int
        :return: Boolean ``(h, w)``.
        :rtype: np.ndarray
        """
        if not keys:
            return self.blocked
        out = np.zeros(self.shape, bool)
        for key, (r, c, mask) in self.boxes.items():
            if key not in keys:
                out[r:r + mask.shape[0], c:c + mask.shape[1]] |= mask
        return dilate(out, grow)

    def box_mask(self, key: int) -> np.ndarray:
        """One other line's polygon on the whole window.

        :param key: Line key.
        :type key: int
        :return: Boolean ``(h, w)`` (all False when the line is not in the window).
        :rtype: np.ndarray
        """
        out = np.zeros(self.shape, bool)
        if key in self.boxes:
            r, c, mask = self.boxes[key]
            out[r:r + mask.shape[0], c:c + mask.shape[1]] = mask
        return out


def neighbour_rows(baseline: Sequence[Point], cols: np.ndarray, others: Sequence[Box], line_height: float,
                   min_gap: float = NEIGHBOUR_MIN_GAP, reach: float = NEIGHBOUR_REACH,
                   x_range: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Baselines of the nearest lines above and below a line, per column.

    A line is a neighbour when it has a baseline and its x-range (polygon or baseline) overlaps this line's;
    it is one at a column when that x-range, carried ``reach`` line heights beyond each end along its chord
    (a neighbour's final word is often outside its segment too), covers the column and its baseline there
    lies at least ``min_gap`` line heights above (or below) this line's.

    :param baseline: The line's baseline.
    :type baseline: Sequence[Point]
    :param cols: Page columns.
    :type cols: np.ndarray
    :param others: Other lines (those with an empty baseline are ignored).
    :type others: Sequence[Box]
    :param line_height: Line height in pixels.
    :type line_height: float
    :param min_gap: Line heights.
    :type min_gap: float
    :param reach: Line heights beyond a neighbour's ends.
    :type reach: float
    :param x_range: This line's own x-range (default: its baseline's).
    :type x_range: Optional[Tuple[float, float]]
    :return: ``(prev, next)``: page rows per column, NaN where there is none.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    cols = np.asarray(cols, dtype=np.float64)
    own = baseline_y(baseline, cols)
    prev = np.full(cols.shape, np.nan)
    nxt = np.full(cols.shape, np.nan)
    gap = min_gap * line_height
    extra = reach * line_height
    if x_range is None:
        x_range = (min(p[0] for p in baseline), max(p[0] for p in baseline))
    for box in others:
        if len(box.baseline) < 2:
            continue
        xs = [p[0] for p in box.baseline] + [p[0] for p in box.polygon]
        if min(xs) >= x_range[1] or max(xs) <= x_range[0]:
            continue
        cover = (cols >= min(xs) - extra) & (cols <= max(xs) + extra)
        if not cover.any():
            continue
        ys = np.full(cols.shape, np.nan)
        ys[cover] = baseline_y(box.baseline, cols[cover])
        above = cover & (ys <= own - gap)
        below = cover & (ys >= own + gap)
        prev = np.where(above, np.fmax(prev, ys), prev)
        nxt = np.where(below, np.fmin(nxt, ys), nxt)
    return prev, nxt


def counted_components(ink: PageInk, ids: np.ndarray, baseline: Sequence[Point], line_height: float,
                       config: ExtentConfig, inside: Optional[np.ndarray] = None,
                       clamp_top: Optional[np.ndarray] = None, clamp_bottom: Optional[np.ndarray] = None
                       ) -> np.ndarray:
    """Which components count as a line's ink.

    :param ink: Page ink components.
    :type ink: PageInk
    :param ids: Component labels to test.
    :type ids: np.ndarray
    :param baseline: The line's baseline.
    :type baseline: Sequence[Point]
    :param line_height: Line height in pixels.
    :type line_height: float
    :param config: Thresholds.
    :type config: ExtentConfig
    :param inside: Per id, the number of its pixels inside other lines' polygons (None: not tested).
    :type inside: Optional[np.ndarray]
    :param clamp_top: Per id, the highest row its centroid may take (the previous baseline plus the clamp).
    :type clamp_top: Optional[np.ndarray]
    :param clamp_bottom: Per id, the lowest row its centroid may take.
    :type clamp_bottom: Optional[np.ndarray]
    :return: Boolean per id: large enough (``walk_min_area``), centroid in the band or (in the wider centroid
        band while reaching no further than ``reach_above`` / ``reach_below``), centroid between the clamps,
        and at most ``max_inside_other`` of its pixels inside other polygons.
    :rtype: np.ndarray
    """
    if ids.size == 0:
        return np.zeros(0, bool)
    lh = line_height
    big = ink.area[ids] >= (config.walk_min_area * lh) ** 2
    cx, cy = ink.centroid_x[ids], ink.centroid_y[ids]
    base = baseline_y(baseline, cx)
    in_band_centre = (cy >= base - config.above * lh) & (cy <= base + config.below * lh)
    contained = ((ink.bbox[ids, 1] >= base - config.reach_above * lh)
                 & (ink.bbox[ids, 3] - 1 <= base + config.reach_below * lh))
    wide = (cy >= base - config.centroid_above * lh) & (cy <= base + config.centroid_below * lh)
    ok = big & (in_band_centre | (wide & contained))
    if clamp_top is not None:
        ok &= cy >= clamp_top
    if clamp_bottom is not None:
        ok &= cy <= clamp_bottom
    if inside is not None:
        ok &= inside <= config.max_inside_other * ink.area[ids]
    return ok


def row_profile(mask: np.ndarray, base_rows: np.ndarray, cols: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Ink pixels per row offset from a sloped baseline, summed over some columns.

    :param mask: ``(h, w)`` boolean window.
    :type mask: np.ndarray
    :param base_rows: ``(w,)`` baseline row per window column (window rows, float).
    :type base_rows: np.ndarray
    :param cols: Window columns summed.
    :type cols: np.ndarray
    :param offsets: Row offsets from the baseline (negative = above).
    :type offsets: np.ndarray
    :return: ``(len(offsets),)`` counts (rows outside the window count nothing).
    :rtype: np.ndarray
    """
    if cols.size == 0:
        return np.zeros(offsets.size)
    rows = np.round(base_rows[cols]).astype(np.int64)[None, :] + offsets.astype(np.int64)[:, None]
    valid = (rows >= 0) & (rows < mask.shape[0])
    cc = np.broadcast_to(cols[None, :], rows.shape)
    out = np.zeros(rows.shape, bool)
    out[valid] = mask[rows[valid], cc[valid]]
    return out.sum(axis=1).astype(np.float64)


def close_runs(flags: np.ndarray, gap: int, grow: int) -> np.ndarray:
    """Fill blank runs shorter than ``gap`` between flagged positions, then grow the flagged runs.

    :param flags: Boolean 1-D array.
    :type flags: np.ndarray
    :param gap: Blank runs shorter than this are filled.
    :type gap: int
    :param grow: Positions added on each side of every run.
    :type grow: int
    :return: Boolean 1-D array.
    :rtype: np.ndarray
    """
    out = flags.copy()
    idx = np.nonzero(flags)[0]
    for a, b in zip(idx, idx[1:]):
        if 1 < b - a <= gap:
            out[a:b] = True
    if grow > 0 and out.any():
        out = ndimage.binary_dilation(out, structure=np.ones(2 * grow + 1, bool))
    return out


def detect_rows(mask: np.ndarray, base: np.ndarray, cols: np.ndarray, prev: np.ndarray, nxt: np.ndarray, y0: int,
                spacing: float, line_height: float) -> Tuple[np.ndarray, np.ndarray, Tuple[bool, bool]]:
    """Unsegmented neighbour lines above and below a line, from the ink row profile along its baseline.

    The profile (ink per row offset from the sloped baseline over the line's own columns, smoothed over a tenth
    of the spacing) has the line's body peak just above its baseline. A second peak at least
    :data:`DETECT_MIN` spacings away from the line's baseline and from the next segmented baseline (or within
    :data:`DETECT_REACH` spacings when there is none), reaching :data:`DETECT_PEAK` of the body peak with
    valleys below :data:`DETECT_VALLEY` of it on both sides, is a line the segmenter never found (a title, a
    cancelled line, a line whose segment was dropped). Its baseline lies as far below its peak as the line's
    own baseline lies below its body peak; it becomes the neighbour over the columns where its band holds ink
    (gaps under a line height closed, grown by half a line height), wherever it is nearer than a segmented one.

    :param mask: ``(h, w)`` ink window without the other segmented lines' polygons.
    :type mask: np.ndarray
    :param base: ``(w,)`` the line's baseline row per window column (page rows).
    :type base: np.ndarray
    :param cols: Window columns of the line itself.
    :type cols: np.ndarray
    :param prev: ``(w,)`` segmented baseline above per column (page rows; NaN: none).
    :type prev: np.ndarray
    :param nxt: ``(w,)`` segmented baseline below per column (page rows; NaN: none).
    :type nxt: np.ndarray
    :param y0: Window top row on the page.
    :type y0: int
    :param spacing: Baseline spacing in pixels.
    :type spacing: float
    :param line_height: Line height in pixels.
    :type line_height: float
    :return: ``(prev, next, (above found, below found))`` with detected lines merged in.
    :rtype: Tuple[np.ndarray, np.ndarray, Tuple[bool, bool]]
    """
    s = max(1.0, spacing)
    reach = int(math.ceil(DETECT_REACH * s))
    offsets = np.arange(-reach, reach + 1)
    base_w = base - y0
    prof = row_profile(mask, base_w, cols, offsets)
    k = max(3, int(round(0.1 * s))) | 1
    prof = np.convolve(prof, np.ones(k) / k, mode="same")
    body = (offsets >= -0.5 * s) & (offsets <= 0.15 * s)
    if not body.any() or prof[body].max() <= 0:
        return prev, nxt, (False, False)
    own_peak = float(prof[body].max())
    d_own = int(offsets[body][int(np.argmax(prof[body]))])
    out = [prev.copy(), nxt.copy()]
    found = [False, False]
    h, w = mask.shape
    rr = np.arange(h)[:, None]
    for n, side in enumerate((-1, 1)):
        nb = out[n]
        rel = nb[cols] - base[cols]
        rel = rel[~np.isnan(rel)]
        far = float(np.median(rel)) if rel.size else side * reach
        if side < 0:
            lo, hi = max(-reach, far + DETECT_MIN * s), -DETECT_MIN * s
        else:
            lo, hi = DETECT_MIN * s, min(reach, far - DETECT_MIN * s)
        sel = (offsets >= lo) & (offsets <= hi)
        if sel.sum() < 2:
            continue
        j = int(np.argmax(prof[sel]))
        d_star, peak = int(offsets[sel][j]), float(prof[sel][j])
        if peak < DETECT_PEAK * own_peak or peak <= 0:
            continue
        if side < 0:
            between = (offsets > d_star) & (offsets < d_own)
            outer = (offsets < d_star) & (offsets >= max(-reach, far))
        else:
            between = (offsets < d_star) & (offsets > d_own)
            outer = (offsets > d_star) & (offsets <= min(reach, far))
        if not between.any() or not outer.any():
            continue
        if prof[between].min() > DETECT_VALLEY * peak or prof[outer].min() > DETECT_VALLEY * peak:
            continue
        d_base = d_star - d_own
        top = np.floor(base_w + d_base - 0.5 * s)[None, :]
        bottom = np.ceil(base_w + d_base + 0.2 * s)[None, :]
        inked = (mask & (rr >= top) & (rr <= bottom)).sum(axis=0) >= 2
        extent = close_runs(inked, int(round(line_height)), int(round(0.5 * line_height)))
        row = base + d_base
        nearer = np.isnan(nb) | ((row > nb) if side < 0 else (row < nb))
        take = extent & nearer
        if take.any():
            out[n] = np.where(take, row, nb)
            found[n] = True
    return out[0], out[1], (found[0], found[1])


def foreign_components(ink: PageInk, ids: np.ndarray, baseline: Sequence[Point], prev: np.ndarray,
                       nxt: np.ndarray, x0: int) -> np.ndarray:
    """Components whose centroid lies nearer another line's baseline than this line's.

    :param ink: Page ink components.
    :type ink: PageInk
    :param ids: Labels tested.
    :type ids: np.ndarray
    :param baseline: The line's baseline.
    :type baseline: Sequence[Point]
    :param prev: ``(w,)`` baseline above per window column (page rows; NaN: none).
    :type prev: np.ndarray
    :param nxt: ``(w,)`` baseline below per window column (page rows; NaN: none).
    :type nxt: np.ndarray
    :param x0: Window left column on the page.
    :type x0: int
    :return: Boolean per id: the vertical distance from its centroid to the neighbour baseline above or below
        (at the centroid's column) is smaller than to this line's sloped baseline.
    :rtype: np.ndarray
    """
    if ids.size == 0:
        return np.zeros(0, bool)
    cx, cy = ink.centroid_x[ids], ink.centroid_y[ids]
    at = np.clip(np.round(cx).astype(np.int64) - x0, 0, prev.size - 1)
    own = np.abs(cy - baseline_y(baseline, cx))
    up = np.abs(cy - prev[at])
    down = np.abs(cy - nxt[at])
    other = np.fmin(np.where(np.isnan(up), np.inf, up), np.where(np.isnan(down), np.inf, down))
    return other < own


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Median of values weighted by counts.

    :param values: Values.
    :type values: np.ndarray
    :param weights: Non-negative weights (same length).
    :type weights: np.ndarray
    :return: The smallest value at which the cumulative weight reaches half the total (0.0 when empty).
    :rtype: float
    """
    if values.size == 0 or weights.sum() <= 0:
        return 0.0
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][int(np.searchsorted(cum, cum[-1] / 2.0))])


def pencil_components(ink: PageInk, ids: np.ndarray, own_ids: np.ndarray, share: float) -> np.ndarray:
    """Components too light or too thin for the line's pen (pencil shelfmarks, faint annotations).

    :param ink: Page ink with :attr:`PageInk.stroke` (and :attr:`PageInk.darkness` when measured).
    :type ink: PageInk
    :param ids: Labels tested.
    :type ids: np.ndarray
    :param own_ids: Labels of the line's own segment ink (the reference).
    :type own_ids: np.ndarray
    :param share: Share of the line's own area-weighted median (0 or less: nothing is dropped).
    :type share: float
    :return: Boolean per id: median darkness or stroke width below ``share`` of the own median.
    :rtype: np.ndarray
    """
    out = np.zeros(ids.size, bool)
    if share <= 0 or own_ids.size == 0 or ids.size == 0:
        return out
    for values in (ink.stroke, ink.darkness):
        if values is None:
            continue
        ref = weighted_median(values[own_ids], ink.area[own_ids])
        if ref > 0:
            out |= values[ids] < share * ref
    return out


def line_scene(ink: PageInk, baseline: Sequence[Point], x_from: float, x_to: float, line_height: float,
               others: Sequence[Box] = (), config: ExtentConfig = ExtentConfig(),
               polygons: Sequence[Sequence[Point]] = (), spacing: Optional[float] = None,
               x_limits: Optional[Tuple[float, float]] = None) -> LineScene:
    """Build the raster context of a line over a range of columns.

    Besides the round-2 context (neighbour baselines, other lines' polygons, counted components), the scene
    classifies the window's components: unsegmented neighbour lines are detected and join the neighbours
    (:func:`detect_rows`); components nearer another baseline are foreign (:func:`foreign_components`); the
    line's own segment ink is the non-foreign components with pixels inside its polygons, and every
    non-foreign component within ``own_touch`` pixels of it is connected to it; non-connected components
    lighter or thinner than ``pencil_share`` of the own ink are pencil (:func:`pencil_components`). Foreign
    and pencil components never count. ``next_top`` is the top of the ink below the line that belongs to
    another line (foreign, centroid below the baseline), minimum-filtered over a fifth of a line height.

    :param ink: Page ink components.
    :type ink: PageInk
    :param baseline: The line's baseline.
    :type baseline: Sequence[Point]
    :param x_from: First page column of the window.
    :type x_from: float
    :param x_to: Last page column of the window.
    :type x_to: float
    :param line_height: Line height in pixels.
    :type line_height: float
    :param others: Every other segmenter line of the page.
    :type others: Sequence[Box]
    :param config: Thresholds.
    :type config: ExtentConfig
    :param polygons: The line's own polygons (the window's rows cover them).
    :type polygons: Sequence[Sequence[Point]]
    :param spacing: Baseline spacing of the page in pixels (None: the line height).
    :type spacing: Optional[float]
    :param x_limits: ``(left, right)`` page columns of the walk's bounds: a component not connected to the
        line's own letters whose centroid lies beyond them never counts (side material: a seal, a margin
        column); its ink is kept in :attr:`LineScene.beyond` for the past-the-block measure. None: no limit.
    :type x_limits: Optional[Tuple[float, float]]
    :return: The scene; rows reach half a line height beyond the farthest band, cap, reach limit or detection
        reach.
    :rtype: LineScene
    """
    height, width = ink.shape
    lh = max(1.0, line_height)
    s = max(1.0, spacing) if spacing else lh
    x0 = int(max(0, math.floor(x_from)))
    x1 = int(min(width, math.ceil(x_to) + 1))
    x1 = max(x1, x0 + 1)
    cols = np.arange(x0, x1)
    base = baseline_y(baseline, cols)
    up = max(config.above, config.reach_above, config.centroid_above, config.up_cap) + 0.5
    down = max(config.below, config.reach_below, config.centroid_below, config.down_cap, config.own_down_cap) + 0.5
    if config.detect:
        up = max(up, DETECT_REACH * s / lh + 0.3)
        down = max(down, DETECT_REACH * s / lh + 0.3)
    ys = [p[1] for poly in polygons for p in poly]
    y0 = int(max(0, math.floor(min([base.min() - up * lh] + ys))))
    y1 = int(min(height, math.ceil(max([base.max() + down * lh] + ys)) + 1))
    y1 = max(y1, y0 + 1)
    h, w = y1 - y0, x1 - x0
    labels = ink.labels[y0:y1, x0:x1]
    boxes: Dict[int, Tuple[int, int, np.ndarray]] = {}
    raw = np.zeros((h, w), bool)
    for box in others:
        if len(box.polygon) < 3:
            continue
        bx0, by0, bx1, by1 = box.bbox()
        c0, r0 = max(0, int(math.floor(bx0)) - x0), max(0, int(math.floor(by0)) - y0)
        c1, r1 = min(w, int(math.ceil(bx1)) + 1 - x0), min(h, int(math.ceil(by1)) + 1 - y0)
        if c1 <= c0 or r1 <= r0:
            continue
        mask = polygons_window([box.polygon], x0 + c0, y0 + r0, c1 - c0, r1 - r0)
        boxes[box.key] = (r0, c0, mask)
        raw[r0:r1, c0:c1] |= mask
    blocked = dilate(raw, config.other_dilate)
    own_xs = [p[0] for poly in polygons for p in poly] or [p[0] for p in baseline]
    prev, nxt = neighbour_rows(baseline, cols, others, lh, config.min_neighbour_gap, config.neighbour_reach,
                               (min(own_xs), max(own_xs)))
    detected = (False, False)
    if config.detect:
        line_cols = np.nonzero((cols >= min(own_xs)) & (cols <= max(own_xs)))[0]
        prev, nxt, detected = detect_rows((labels > 0) & ~raw, base, line_cols, prev, nxt, y0, s, lh)
    n_labels = ink.area.size
    counted = np.zeros(n_labels, bool)
    foreign = np.zeros(n_labels, bool)
    connected = np.zeros(n_labels, bool)
    pencil = np.zeros(n_labels, bool)
    beyond = np.zeros(n_labels, bool)
    members = polygons_window(polygons, x0, y0, w, h) if polygons else np.zeros((h, w), bool)
    ids = np.unique(labels)
    ids = ids[ids > 0]
    next_top = np.full(w, np.inf)
    if ids.size:
        if config.foreign:
            far_prev, far_next = neighbour_rows(baseline, cols, others, lh, config.min_neighbour_gap,
                                                max(config.neighbour_reach, config.foreign_reach),
                                                (min(own_xs), max(own_xs)))
            foreign[ids] = foreign_components(ink, ids, baseline, np.fmax(far_prev, prev), np.fmin(far_next, nxt),
                                              x0)
        in_members = np.bincount(labels[members], minlength=n_labels) > 0
        in_members[0] = False
        own_seg = in_members & ~foreign
        if own_seg.any():
            touch = dilate(own_seg[labels], config.own_touch) & (labels > 0)
            connected[np.unique(labels[touch])] = True
            connected[0] = False
            connected &= ~foreign
            connected |= own_seg
        candidates = ids[~connected[ids]]
        pencil[candidates] = pencil_components(ink, candidates, np.nonzero(own_seg)[0], config.pencil_share)
        inside = np.bincount(labels[raw], minlength=n_labels)[ids]
        at = np.clip(np.round(ink.centroid_x[ids]).astype(np.int64) - x0, 0, w - 1)
        off = config.clamp * lh
        top = np.where(np.isnan(prev[at]), -np.inf, prev[at] + off)
        bottom = np.where(np.isnan(nxt[at]), np.inf, nxt[at] - off)
        counted[ids] = counted_components(ink, ids, baseline, lh, config, inside, top, bottom)
        counted &= ~foreign & ~pencil
        if x_limits is not None:
            cx = ink.centroid_x[ids]
            out_ids = ids[(counted[ids] & ~connected[ids]) & ((cx < x_limits[0]) | (cx > x_limits[1]))]
            beyond[out_ids] = True
            counted[out_ids] = False
        below = np.zeros(n_labels, bool)
        below[ids] = foreign[ids] & (ink.centroid_y[ids] > baseline_y(baseline, ink.centroid_x[ids]))
        low = below[labels]
        if low.any():
            first = np.argmax(low, axis=0)
            next_top = np.where(low.any(axis=0), first + y0, np.inf).astype(np.float64)
            size = max(1, int(round(0.2 * lh))) | 1
            next_top = ndimage.minimum_filter1d(next_top, size, mode="nearest")
    own = (counted | connected)[labels] & ~blocked
    return LineScene(x0, y0, labels, base, prev, nxt, counted, boxes, raw, blocked, own, lh, s, foreign,
                     connected, pencil, next_top, members, detected, beyond)


def band_rows(scene: LineScene, idx: np.ndarray, above: float, below: float, clamp: float
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Window rows of the line's band at some window columns, clamped to the neighbours.

    :param scene: Line scene.
    :type scene: LineScene
    :param idx: Window columns.
    :type idx: np.ndarray
    :param above: Line heights above the baseline.
    :type above: float
    :param below: Line heights below the baseline.
    :type below: float
    :param clamp: Neighbour clamp in line heights.
    :type clamp: float
    :return: ``(top, stop)`` window rows per column (``stop`` exclusive, never below ``top``).
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    lh = scene.line_height
    h = scene.shape[0]
    c_top, c_bottom = scene.clamps(clamp)
    top = np.maximum(scene.base[idx] - above * lh, c_top[idx])
    bottom = np.minimum(scene.base[idx] + below * lh, c_bottom[idx])
    t = np.clip(np.floor(top) - scene.y0, 0, h).astype(np.int64)
    b = np.clip(np.ceil(bottom) + 1 - scene.y0, 0, h).astype(np.int64)
    return t, np.maximum(b, t)


def column_counts(scene: LineScene, idx: np.ndarray, config: ExtentConfig, mask: Optional[np.ndarray] = None
                  ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Counted ink pixels in the clamped band per window column.

    :param scene: Line scene.
    :type scene: LineScene
    :param idx: Window columns.
    :type idx: np.ndarray
    :param config: Thresholds (band and clamp).
    :type config: ExtentConfig
    :param mask: ``(h, w)`` ink counted (default: :attr:`LineScene.own`).
    :type mask: Optional[np.ndarray]
    :return: ``(counts, (top, stop))`` per column of ``idx``.
    :rtype: Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]
    """
    t, b = band_rows(scene, idx, config.above, config.below, config.clamp)
    own = (scene.own if mask is None else mask)[:, idx]
    cs = np.vstack([np.zeros((1, idx.size), np.int64), np.cumsum(own, axis=0, dtype=np.int64)])
    cols = np.arange(idx.size)
    return cs[b, cols] - cs[t, cols], (t, b)


def band_components(scene: LineScene, idx: np.ndarray, rows: Tuple[np.ndarray, np.ndarray]) -> Tuple[int, ...]:
    """Counted components present in the band over some window columns.

    :param scene: Line scene.
    :type scene: LineScene
    :param idx: Window columns.
    :type idx: np.ndarray
    :param rows: ``(top, stop)`` window rows per column (:func:`band_rows`).
    :type rows: Tuple[np.ndarray, np.ndarray]
    :return: Sorted labels.
    :rtype: Tuple[int, ...]
    """
    if idx.size == 0:
        return ()
    rr = np.arange(scene.shape[0])[:, None]
    sel = (rr >= rows[0][None, :]) & (rr < rows[1][None, :]) & scene.own[:, idx]
    return tuple(int(k) for k in np.unique(scene.labels[:, idx][sel]))


# ----------------------------------------------------------------------------- walk

def extend_end(ink: PageInk, baseline: Sequence[Point], end_x: int, direction: int, line_height: float,
               boxes: Sequence[Box] = (), blocked: Sequence[Sequence[Point]] = (),
               config: ExtentConfig = ExtentConfig(), limit: Optional[int] = None,
               scene: Optional[LineScene] = None, connected_only: bool = False) -> EndExtent:
    """Walk the band outward from one end of a line.

    Up to ``limit`` (the text block bound) a column holds ink when the counted or own-connected ink in its band
    reaches the column threshold; beyond it, only ink of components connected to the line's own letters counts
    (a swash, a kashida or a final letter running past the margin), still with the gap rule and the cap.

    :param ink: Page ink components.
    :type ink: PageInk
    :param baseline: The line's baseline.
    :type baseline: Sequence[Point]
    :param end_x: Last column of the line's polygon at this end.
    :type end_x: int
    :param direction: ``-1`` for the left end, ``+1`` for the right end.
    :type direction: int
    :param line_height: Line height in pixels.
    :type line_height: float
    :param boxes: Same-row segmenter boxes (absorption candidates; their pixels never count as ink).
    :type boxes: Sequence[Box]
    :param blocked: Polygons of lines on other rows (used only to build a scene when none is given).
    :type blocked: Sequence[Sequence[Point]]
    :param config: Thresholds.
    :type config: ExtentConfig
    :param limit: Pixels the end may move at most (the text block bound; None: unbounded). Ink beyond it is
        not this line's: the walk stops there and ink beyond is measured only up to it.
    :type limit: Optional[int]
    :param scene: Line scene covering the walk's columns (built from ``boxes`` and ``blocked`` when None).
    :type scene: Optional[LineScene]
    :param connected_only: Only ink connected to the line's own letters may move this end (the block bound is
        the end itself), e.g. after an edition line ending in ':' whose filler glyph must stay out.
    :type connected_only: bool
    :return: How far the end moved, the ink still beyond it, the absorbed and the stolen boxes, the counted
        components, whether the block stopped the walk and how far own-connected ink carried the end past it.
    :rtype: EndExtent
    """
    lh = max(1.0, line_height)
    gap = max(1, int(round(config.gap * lh)))
    cap = max(0, int(round(config.max_extend * lh)))
    span = cap + gap + max(cap, int(round(EXTEND_MAX * lh)))
    xs = end_x + direction * np.arange(1, span + 1)
    xs = xs[(xs >= 0) & (xs < ink.shape[1])]
    if xs.size == 0:
        return EndExtent()
    if scene is None:
        walls = [Box(-1 - n, tuple((float(x), float(y)) for x, y in poly)) for n, poly in enumerate(blocked)]
        scene = line_scene(ink, baseline, min(int(xs.min()), end_x), max(int(xs.max()), end_x), lh,
                           list(boxes) + walls, config)
    idx = xs - scene.x0
    need = max(config.min_column_pixels, config.min_column_ink * lh)
    counts, rows = column_counts(scene, idx, config)
    inked_all = counts >= need
    conn_mask = scene.label_mask(scene.connected) & ~scene.blocked
    inked_conn = column_counts(scene, idx, config, conn_mask)[0] >= config.min_column_pixels
    bounded = limit is not None or connected_only
    bound = 0 if connected_only else (cap if limit is None else max(0, min(cap, int(limit))))
    inked = np.where(np.arange(inked_all.size) < bound, inked_all, inked_conn)
    reach = scan_run(inked, gap, cap)
    pending = []
    for box in boxes:
        if not box.absorbable:
            continue
        x_min, _, x_max, _ = box.bbox()
        centre = (box.centre()[0] - end_x) * direction
        near = (x_min - end_x) if direction > 0 else (end_x - x_max)
        far = (x_max - end_x) if direction > 0 else (end_x - x_min)
        if 0 < centre <= bound:
            pending.append((near, far, box))
    absorbed: List[int] = []
    stolen: List[int] = []
    changed = True
    while changed and pending:
        changed = False
        for item in sorted(pending, key=lambda t: t[0]):
            near, far, box = item
            if near > reach + gap:
                continue
            if box.steal:
                stolen.append(box.key)
                pending.remove(item)
            elif far <= bound:
                absorbed.append(box.key)
                pending.remove(item)
                reach = max(reach, int(math.ceil(far)))
                changed = True
        if changed and reach < cap:
            reach += scan_run(inked[reach:], gap, cap - reach)
    diag = inked_all
    if scene.beyond is not None and scene.beyond.any():
        side = scene.own | (scene.label_mask(scene.beyond) & ~scene.blocked)
        diag = column_counts(scene, idx, config, side)[0] >= need
    full = scan_run(diag[reach:], gap, len(diag) - reach)
    room = max(0, (0 if connected_only else int(limit)) - reach) if bounded else len(diag)
    beyond = min(full, room)
    components = band_components(scene, idx[:reach], (rows[0][:reach], rows[1][:reach]))
    return EndExtent(extended_px=int(reach), ink_beyond_px=int(beyond), absorbed=tuple(absorbed),
                     stolen=tuple(stolen), capped=cap > 0 and reach >= cap, components=components,
                     limited=full > beyond, past_block_px=int(full - beyond),
                     crossed_px=int(max(0, reach - bound)) if bounded else 0)


# ----------------------------------------------------------------------------- polygon

def end_extent(base: np.ndarray, top: np.ndarray, bottom: np.ndarray, cols: np.ndarray) -> Tuple[float, float]:
    """Vertical extent of a polygon around its baseline over some columns.

    :param base: Baseline row per window column.
    :type base: np.ndarray
    :param top: Polygon top row per window column.
    :type top: np.ndarray
    :param bottom: Polygon bottom row per window column.
    :type bottom: np.ndarray
    :param cols: Window columns holding polygon pixels.
    :type cols: np.ndarray
    :return: ``(up, down)``: median pixels above and below the baseline (each at least 0).
    :rtype: Tuple[float, float]
    """
    if cols.size == 0:
        return 0.0, 0.0
    return (max(0.0, float(np.median(base[cols] - top[cols]))),
            max(0.0, float(np.median(bottom[cols] - base[cols]))))


def open_slivers(mask: np.ndarray, width: int) -> np.ndarray:
    """Remove the parts of a mask narrower than ``width`` pixels horizontally (morphological opening).

    :param mask: Boolean image.
    :type mask: np.ndarray
    :param width: Pixels (1 or less: a copy); an even width is rounded up to the next odd one, so the
        element is centred and the opening never shifts the mask.
    :type width: int
    :return: Opened mask (a subset of ``mask``).
    :rtype: np.ndarray
    """
    if width <= 1 or not mask.any():
        return mask.copy()
    width = int(width) | 1
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, width), np.uint8)) > 0


def raise_offsets(config: ExtentConfig, line_height: float, spacing: float) -> Tuple[float, float]:
    """How far the vertical rebuild reaches above and below the baseline.

    :param config: Thresholds.
    :type config: ExtentConfig
    :param line_height: Line height in pixels.
    :type line_height: float
    :param spacing: Baseline spacing in pixels (0 or less: the line height).
    :type spacing: float
    :return: ``(up, down)`` pixels: ``min(above x lh, spacing_above x spacing)`` and
        ``min(below x lh, spacing_below x spacing)``.
    :rtype: Tuple[float, float]
    """
    s = spacing if spacing > 0 else line_height
    return min(config.above * line_height, config.spacing_above * s), min(config.below * line_height,
                                                                         config.spacing_below * s)


def column_extremes(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First and last True row of every column.

    :param mask: Boolean ``(h, w)``.
    :type mask: np.ndarray
    :return: ``(any, first, last)`` per column (``first`` = h and ``last`` = -1 where the column is empty).
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    h = mask.shape[0]
    has = mask.any(axis=0)
    first = np.where(has, np.argmax(mask, axis=0), h)
    last = np.where(has, h - 1 - np.argmax(mask[::-1], axis=0), -1)
    return has, first, last


def trim_foreign(top: np.ndarray, bottom: np.ndarray, base: np.ndarray, foreign: np.ndarray, own: np.ndarray,
                 columns: np.ndarray, pad: int, spacing: float) -> Tuple[np.ndarray, np.ndarray]:
    """Pull the interval's top below foreign ink hanging into it from above (and its bottom above foreign ink
    reaching up from below), in every column where that leaves no own ink outside.

    :param top: ``(w,)`` interval top per column (window rows).
    :type top: np.ndarray
    :param bottom: ``(w,)`` interval bottom per column (window rows).
    :type bottom: np.ndarray
    :param base: ``(w,)`` baseline per column (window rows).
    :type base: np.ndarray
    :param foreign: ``(h, w)`` foreign ink.
    :type foreign: np.ndarray
    :param own: ``(h, w)`` own-connected ink.
    :type own: np.ndarray
    :param columns: ``(w,)`` columns rebuilt.
    :type columns: np.ndarray
    :param pad: Clearance in pixels.
    :type pad: int
    :param spacing: Baseline spacing in pixels (the top never goes below ``base - spacing / 4``, the bottom
        never above ``base + spacing / 10``: deeper foreign ink is only subtracted).
    :type spacing: float
    :return: ``(top, bottom)``.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    h = foreign.shape[0]
    rr = np.arange(h)[:, None]
    above = rr < base[None, :]
    hit, _, f_low = column_extremes(foreign & above & (rr >= np.floor(top)[None, :]))
    own_above = (own & (rr < f_low[None, :])).any(axis=0)
    cut = hit & ~own_above & columns
    top = np.where(cut, np.maximum(top, np.minimum(f_low + pad + 1, base - 0.25 * spacing)), top)
    hit, f_high, _ = column_extremes(foreign & ~above & (rr <= np.ceil(bottom)[None, :]))
    own_below = (own & (rr > f_high[None, :])).any(axis=0)
    cut = hit & ~own_below & columns
    bottom = np.where(cut, np.minimum(bottom, np.maximum(f_high - pad - 1, base + 0.1 * spacing)), bottom)
    return top, bottom


def remove_stubs(added: np.ndarray, interval: np.ndarray, kept: np.ndarray, own: np.ndarray, width: int
                 ) -> Tuple[np.ndarray, int]:
    """Drop narrow protrusions beyond the band strip that hold no own-connected ink.

    :param added: ``(h, w)`` everything the rebuild adds.
    :type added: np.ndarray
    :param interval: ``(h, w)`` the band strip.
    :type interval: np.ndarray
    :param kept: ``(h, w)`` members and absorbed boxes (never removed).
    :type kept: np.ndarray
    :param own: ``(h, w)`` own-connected ink.
    :type own: np.ndarray
    :param width: Pieces narrower than this many columns are stubs (1 or less: nothing is removed).
    :type width: int
    :return: ``(added without the stubs, number of stubs removed)``.
    :rtype: Tuple[np.ndarray, int]
    """
    if width <= 1:
        return added, 0
    out = added & ~interval & ~kept
    if not out.any():
        return added, 0
    labels, n = ndimage.label(out, structure=np.ones((3, 3), bool))
    owned = np.bincount(labels[own & out], minlength=n + 1) > 0
    drop = np.zeros(n + 1, bool)
    for k, sl in enumerate(ndimage.find_objects(labels), 1):
        if sl is not None and sl[1].stop - sl[1].start < width and not owned[k]:
            drop[k] = True
    return added & ~drop[labels], int(drop.sum())


def line_shape(scene: LineScene, members_mask: np.ndarray, absorbed_mask: np.ndarray, x_lo: int, x_hi: int,
               extended: Tuple[int, int], blocked: np.ndarray, config: ExtentConfig,
               walk_components: Sequence[int] = (), extra_components: Sequence[int] = ()
               ) -> Tuple[np.ndarray, int]:
    """The line's region in the scene window, rebuilt per column.

    The band strip runs over the extended columns (the whole line with ``raise_top``): the line's own extent
    at the ends, raised to :func:`raise_offsets` (spacing-limited) and clamped to the neighbours (segmented
    and detected). Foreign ink hanging into it pulls it back (:func:`trim_foreign`). Component pixels are
    added (grown by ``component_pad``) only for admitted components: those connected to the line's own
    segment ink (any column; up to ``up_cap`` above and ``own_down_cap`` below the baseline, clear of the next
    line's ink top by ``next_top_clear``), those the walk counted beyond the ends, the supralinear extras, and
    counted components lying inside the original x-range (extended columns only; ``up_cap`` / ``down_cap`` and
    the clamps); the columns of the supralinear extras are filled from their top down to the band so they join
    the line. Foreign components (grown) and the other lines' polygons are subtracted (foreign ink never cuts the
    strip's spine, :data:`SPINE_ABOVE` / :data:`SPINE_BELOW`, so merged pieces stay joined), narrow protrusions
    without own ink are dropped (:func:`remove_stubs`) and slivers are opened away.

    :param scene: Line scene.
    :type scene: LineScene
    :param members_mask: The member polygons rasterised in the window.
    :type members_mask: np.ndarray
    :param absorbed_mask: The absorbed boxes rasterised in the window.
    :type absorbed_mask: np.ndarray
    :param x_lo: Left end of the members (page column).
    :type x_lo: int
    :param x_hi: Right end of the members (page column).
    :type x_hi: int
    :param extended: ``(left, right)`` pixels the walk moved each end.
    :type extended: Tuple[int, int]
    :param blocked: Other lines' polygons, grown (absorbed boxes left out); nothing added may touch them.
    :type blocked: np.ndarray
    :param config: Thresholds.
    :type config: ExtentConfig
    :param walk_components: Components the walk counted beyond the ends.
    :type walk_components: Sequence[int]
    :param extra_components: Components added unconditionally (a supralinear insertion's ink).
    :type extra_components: Sequence[int]
    :return: ``(region, stubs removed)``: boolean ``(h, w)``: members, absorbed boxes, and the added columns
        and component pixels (module docstring, step 4).
    :rtype: Tuple[np.ndarray, int]
    """
    h, w = scene.shape
    lh = scene.line_height
    has = members_mask.any(axis=0)
    member_cols = np.nonzero(has)[0]
    if member_cols.size == 0:
        return members_mask | absorbed_mask, 0
    pad = config.strip_pad
    first, last = int(member_cols[0]), int(member_cols[-1])
    lo = x_lo - extended[0] - pad - scene.x0 if extended[0] else first
    hi = x_hi + extended[1] + pad - scene.x0 if extended[1] else last
    lo, hi = int(max(0, math.floor(lo))), int(min(w - 1, math.ceil(hi)))
    in_range = np.zeros(w, bool)
    in_range[lo:hi + 1] = True
    base = scene.base - scene.y0
    c_top, c_bottom = scene.clamps(config.clamp)
    c_top, c_bottom = c_top - scene.y0, c_bottom - scene.y0
    top_m = np.argmax(members_mask, axis=0).astype(np.float64)
    bottom_m = (h - 1 - np.argmax(members_mask[::-1], axis=0)).astype(np.float64)
    span = max(1, int(round(config.end_window * lh)))
    up = np.zeros(w)
    down = np.zeros(w)
    up[:first], down[:first] = end_extent(base, top_m, bottom_m, member_cols[member_cols < first + span])
    up[last + 1:], down[last + 1:] = end_extent(base, top_m, bottom_m, member_cols[member_cols > last - span])
    holes = np.nonzero(~has[first:last + 1])[0] + first
    if holes.size:
        runs = np.split(holes, np.nonzero(np.diff(holes) > 1)[0] + 1)
        for run in runs:
            g0, g1 = int(run[0]), int(run[-1])
            ua, da = end_extent(base, top_m, bottom_m, member_cols[(member_cols < g0) & (member_cols >= g0 - span)])
            ub, db = end_extent(base, top_m, bottom_m, member_cols[(member_cols > g1) & (member_cols <= g1 + span)])
            t = (np.arange(g0, g1 + 1) - (g0 - 1)) / float(g1 + 1 - (g0 - 1))
            up[g0:g1 + 1] = ua + t * (ub - ua)
            down[g0:g1 + 1] = da + t * (db - da)
    s = scene.spacing if scene.spacing > 0 else lh
    up_px, down_px = raise_offsets(config, lh, s)
    band_top, band_bottom = base - up_px, base + down_px
    raised_top = np.maximum(band_top, c_top)
    raised_bottom = np.minimum(band_bottom, c_bottom)
    top = base - up
    bottom = base + down
    if config.raise_top:
        top = np.minimum(top, band_top)
        bottom = np.maximum(bottom, band_bottom)
    top = np.maximum(top, c_top)
    bottom = np.minimum(bottom, c_bottom)
    if config.raise_top:
        top = np.where(has, np.minimum(top_m, raised_top), top)
        bottom = np.where(has, np.maximum(bottom_m, raised_bottom), bottom)
        columns = in_range
    else:
        columns = in_range & ~has
    pad = max(1, int(round(config.component_pad * lh)))
    foreign = scene.label_mask(scene.foreign)
    own = scene.label_mask(scene.connected)
    if foreign.any():
        top, bottom = trim_foreign(top, bottom, base, foreign, own, columns, pad, s)
    rr = np.arange(h)[:, None]
    interval = (rr >= np.floor(top)[None, :]) & (rr <= np.ceil(bottom)[None, :]) & columns[None, :]
    n_labels = scene.counted.size
    rows_l, cols_l = np.nonzero(scene.labels)
    labs = scene.labels[rows_l, cols_l]
    bbox_x0 = np.full(n_labels, np.inf)
    bbox_x1 = np.full(n_labels, -np.inf)
    np.minimum.at(bbox_x0, labs, cols_l + scene.x0)
    np.maximum.at(bbox_x1, labs, cols_l + scene.x0)
    counted_inside = scene.counted & (bbox_x0 >= x_lo) & (bbox_x1 <= x_hi)
    walk = np.zeros(n_labels, bool)
    walk[[k for k in walk_components if 0 < k < n_labels]] = True
    extra = np.zeros(n_labels, bool)
    extra[[k for k in extra_components if 0 < k < n_labels]] = True
    connected = scene.connected if scene.connected is not None else np.zeros(n_labels, bool)
    other = (counted_inside | walk) & ~connected & ~extra
    comp = np.zeros((h, w), bool)
    if other.any():
        part = dilate(other[scene.labels], pad)
        cap_top = np.maximum(base - config.up_cap * lh, c_top)
        cap_bottom = np.minimum(base + config.down_cap * lh, c_bottom)
        comp |= part & (rr >= np.floor(cap_top)[None, :]) & (rr <= np.ceil(cap_bottom)[None, :]) & columns[None, :]
    if (connected | extra).any():
        part = dilate((connected | extra)[scene.labels], pad)
        next_top = scene.next_top - scene.y0 if scene.next_top is not None else np.full(w, np.inf)
        cap_top = np.maximum(base - config.up_cap * lh, c_top)
        cap_top = np.where(extra[scene.labels].any(axis=0), np.maximum(base - (config.up_cap + 0.5) * lh, c_top),
                           cap_top)
        cap_bottom = np.minimum(base + config.own_down_cap * lh, next_top - config.next_top_clear * lh)
        idx = np.arange(w)
        outside = columns | (idx < first) | (idx > last)
        comp |= (part & (rr >= np.floor(cap_top)[None, :]) & (rr <= np.ceil(cap_bottom)[None, :])
                 & outside[None, :])
        if extra.any():
            lifted = dilate(extra[scene.labels], pad) & (rr >= np.floor(cap_top)[None, :])
            has_e, first_e, _ = column_extremes(lifted)
            comp |= (rr >= first_e[None, :]) & (rr <= np.ceil(bottom)[None, :]) & has_e[None, :]
    added = (interval | comp) & ~blocked
    if foreign.any():
        added &= ~dilate(foreign & ~extra[scene.labels], pad)
        spine = ((rr >= np.floor(base - SPINE_ABOVE * lh)[None, :]) & (rr <= np.ceil(base + SPINE_BELOW * lh)[None, :])
                 & columns[None, :])
        added |= spine & interval & ~blocked
    kept = members_mask | absorbed_mask
    added, stubs = remove_stubs(added, interval, kept, own | extra[scene.labels],
                                int(round(config.stub_width * lh)))
    union = kept | added
    return kept | open_slivers(union, max(2, int(round(config.sliver * lh)))), stubs


def mask_outline(region: np.ndarray, anchor: np.ndarray, x0: int, y0: int, epsilon: float = 1.0
                 ) -> Tuple[List[Point], bool]:
    """Outline of the part of a region that holds the anchor pixels.

    :param region: Boolean window.
    :type region: np.ndarray
    :param anchor: Boolean window of pixels that must stay (the members and absorbed boxes).
    :type anchor: np.ndarray
    :param x0: Window left column on the page.
    :type x0: int
    :param y0: Window top row on the page.
    :type y0: int
    :param epsilon: Douglas-Peucker tolerance in pixels.
    :type epsilon: float
    :return: ``(outline, fell_apart)`` in page coordinates: the simplified outer contour of the one piece
        holding the anchor (pieces without anchor pixels are dropped); when the anchor lies in several
        pieces, their convex hull and True.
    :rtype: Tuple[List[Point], bool]
    """
    n, labels = cv2.connectedComponents(region.astype(np.uint8), connectivity=8)
    keep = np.unique(labels[anchor & region])
    keep = keep[keep > 0]
    if keep.size == 0:
        keep = np.arange(1, n)
    piece = np.isin(labels, keep).astype(np.uint8)
    if keep.size > 1:
        hull = cv2.convexHull(cv2.findNonZero(piece))[:, 0]
        return [(float(x + x0), float(y + y0)) for x, y in hull], True
    contours, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(contour, epsilon, True)[:, 0]
    if len(approx) < 3:
        x, y, bw, bh = cv2.boundingRect(contour)
        approx = np.array([(x, y), (x + bw - 1, y), (x + bw - 1, y + bh - 1), (x, y + bh - 1)])
    return [(float(x + x0), float(y + y0)) for x, y in approx], False


def union_outline(polygons: Sequence[Sequence[Point]], width: int, height: int,
                  epsilon: float = 1.0) -> Tuple[List[Point], bool]:
    """Outer outline of the union of polygons.

    :param polygons: Polygons in page coordinates.
    :type polygons: Sequence[Sequence[Point]]
    :param width: Image width (the outline is clamped into the image).
    :type width: int
    :param height: Image height.
    :type height: int
    :param epsilon: Douglas-Peucker tolerance in pixels for simplifying the traced outline.
    :type epsilon: float
    :return: ``(outline, fell_apart)``: the outline of the single connected union; when the union has several
        pieces, the convex hull of all of them and True.
    :rtype: Tuple[List[Point], bool]
    """
    pts = np.array([p for poly in polygons for p in poly], dtype=np.float64)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    x0, y0 = int(np.floor(pts[:, 0].min())) - 1, int(np.floor(pts[:, 1].min())) - 1
    x1, y1 = int(np.ceil(pts[:, 0].max())) + 2, int(np.ceil(pts[:, 1].max())) + 2
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    for poly in polygons:
        arr = np.array(poly, dtype=np.float64)
        arr[:, 0] = np.clip(arr[:, 0], 0, width - 1) - x0
        arr[:, 1] = np.clip(arr[:, 1], 0, height - 1) - y0
        cv2.fillPoly(mask, [np.round(arr).astype(np.int32)], 1)
    n, _ = cv2.connectedComponents(mask, connectivity=8)
    if n - 1 > 1:
        hull = cv2.convexHull(np.round(pts).astype(np.int32))[:, 0]
        return [(float(x), float(y)) for x, y in hull], True
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(contour, epsilon, True)[:, 0]
    return [(float(x + x0), float(y + y0)) for x, y in approx], False


def moved_baseline(baseline: Sequence[Point], new_left: Optional[float], new_right: Optional[float]
                   ) -> List[Point]:
    """A baseline with its end points moved outward, its point order kept.

    :param baseline: Baseline points in their original order (left to right or right to left).
    :type baseline: Sequence[Point]
    :param new_left: New left end column (None: unchanged).
    :type new_left: Optional[float]
    :param new_right: New right end column (None: unchanged).
    :type new_right: Optional[float]
    :return: The baseline with an extrapolated point added at each moved end (the chord slope of the
        original baseline, :func:`baseline_y`).
    :rtype: List[Point]
    """
    pts = list(baseline)
    left_to_right = pts[0][0] <= pts[-1][0]
    if not left_to_right:
        pts.reverse()
    if new_left is not None and new_left < pts[0][0]:
        pts.insert(0, (float(new_left), float(baseline_y(baseline, np.array([new_left]))[0])))
    if new_right is not None and new_right > pts[-1][0]:
        pts.append((float(new_right), float(baseline_y(baseline, np.array([new_right]))[0])))
    return pts if left_to_right else pts[::-1]


# ----------------------------------------------------------------------------- measures

def block_columns(scene: LineScene, block: Optional[Tuple[float, float]], slack_px: float, x_from: float,
                  x_to: float) -> np.ndarray:
    """Window columns of the text block (with slack), or of a fallback range without a block.

    :param scene: Line scene.
    :type scene: LineScene
    :param block: ``(left, right)`` margins (None: use ``x_from`` .. ``x_to``).
    :type block: Optional[Tuple[float, float]]
    :param slack_px: Slack beyond the margins in pixels.
    :type slack_px: float
    :param x_from: Fallback first page column.
    :type x_from: float
    :param x_to: Fallback last page column.
    :type x_to: float
    :return: Boolean per window column.
    :rtype: np.ndarray
    """
    w = scene.shape[1]
    lo, hi = (block[0] - slack_px, block[1] + slack_px) if block is not None else (x_from, x_to)
    cols = np.arange(w) + scene.x0
    return (cols >= math.floor(lo)) & (cols <= math.ceil(hi))


def own_ink_share(scene: LineScene, polygon_mask: np.ndarray, columns: np.ndarray, blocked: np.ndarray,
                  config: ExtentConfig) -> float:
    """Share of the ink in the line's band that lies outside its polygon.

    :param scene: Line scene.
    :type scene: LineScene
    :param polygon_mask: The final polygon rasterised in the window.
    :type polygon_mask: np.ndarray
    :param columns: Window columns measured (the text block, :func:`block_columns`).
    :type columns: np.ndarray
    :param blocked: Other lines' grown polygons (their pixels are not measured).
    :type blocked: np.ndarray
    :param config: Thresholds (band, clamp).
    :type config: ExtentConfig
    :return: Outside / all ink pixels in the band (the raise band, :func:`raise_offsets`, clamped to the
        neighbours) over ``columns``, outside ``blocked``, of the components that can be the line's own (counted
        or connected to its letters: foreign, pencil, side material and stray page-sized blobs are left out);
        0.0 without such ink, or when it is less than :data:`OWN_EVIDENCE` of the ink inside the polygon's band
        (the line's letters are lost in a blob, e.g. a stained fragment or a rim joined to them: no evidence).
    :rtype: float
    """
    idx = np.nonzero(columns)[0]
    if idx.size == 0:
        return 0.0
    up_px, down_px = raise_offsets(config, scene.line_height, scene.spacing)
    t, b = band_rows(scene, idx, up_px / scene.line_height, down_px / scene.line_height, config.clamp)
    rr = np.arange(scene.shape[0])[:, None]
    band = (rr >= t[None, :]) & (rr < b[None, :])
    own = scene.counted | (scene.connected if scene.connected is not None else False)
    labels = scene.labels[:, idx]
    ink = band & own[labels] & ~blocked[:, idx]
    total = int(ink.sum())
    inside = int((band & (labels > 0) & polygon_mask[:, idx] & ~blocked[:, idx]).sum())
    if not total or total < OWN_EVIDENCE * inside:
        return 0.0
    return float((ink & ~polygon_mask[:, idx]).sum()) / total


def reach_columns(scene: LineScene, region: np.ndarray, reach: float, block_cols: Optional[np.ndarray]
                  ) -> np.ndarray:
    """Columns the own-ink measure looks at: the region's x-range plus ``reach`` pixels, inside the block.

    :param scene: Line scene.
    :type scene: LineScene
    :param region: ``(h, w)`` the line's final region (or its polygon).
    :type region: np.ndarray
    :param reach: Pixels on each side (``inf``: every column of ``block_cols``).
    :type reach: float
    :param block_cols: Boolean per window column of the line's column block (None: no block).
    :type block_cols: Optional[np.ndarray]
    :return: Boolean per window column.
    :rtype: np.ndarray
    """
    w = scene.shape[1]
    cols = np.nonzero(region.any(axis=0))[0]
    if not math.isfinite(reach):
        out = np.ones(w, bool)
    elif cols.size == 0:
        out = np.zeros(w, bool)
    else:
        idx = np.arange(w)
        out = (idx >= cols[0] - reach) & (idx <= cols[-1] + reach)
    return out & block_cols if block_cols is not None else out


def new_overlap(scene: LineScene, polygon_mask: np.ndarray, members_mask: np.ndarray, others: Sequence[Box],
                skip: Set[int]) -> Tuple[float, Optional[int]]:
    """Largest share of another line's polygon that the final polygon covers beyond what the members covered.

    The share is taken of the other polygon's rasterised area (so it never exceeds 1, whatever the outline's
    rounding), and lines whose rasterised polygon is smaller than a quarter line height squared (specks the
    segmenter boxed, e.g. a dot inside a merged line's seam) are not tested.

    :param scene: Line scene.
    :type scene: LineScene
    :param polygon_mask: Final polygon in the window.
    :type polygon_mask: np.ndarray
    :param members_mask: Member polygons in the window.
    :type members_mask: np.ndarray
    :param others: Other lines.
    :type others: Sequence[Box]
    :param skip: Keys not tested (absorbed boxes).
    :type skip: Set[int]
    :return: ``(share, key)``; ``(0.0, None)`` without new overlap.
    :rtype: Tuple[float, Optional[int]]
    """
    best, best_key = 0.0, None
    tiny = (0.25 * scene.line_height) ** 2
    for box in others:
        if box.key in skip or box.key not in scene.boxes:
            continue
        r, c, mask = scene.boxes[box.key]
        area = max(int(mask.sum()), int(round(polygon_area(box.polygon))))
        if area < tiny:
            continue
        sl = (slice(r, r + mask.shape[0]), slice(c, c + mask.shape[1]))
        gained = int((polygon_mask[sl] & mask).sum()) - int((members_mask[sl] & mask).sum())
        if gained <= 0:
            continue
        share = min(1.0, gained / max(1.0, float(mask.sum())))
        if share > best:
            best, best_key = share, box.key
    return best, best_key


def end_shift(scene: LineScene, members_mask: np.ndarray, end_x: int, direction: int, end: EndExtent,
              absorbed_mask: np.ndarray, config: ExtentConfig) -> float:
    """How far the detached ink an end took sits below (+) or above (-) the line's own ink at that end.

    Detached ink is the walk's counted components that are not connected to the line's own letters, in the
    columns beyond the end, plus the ink inside boxes absorbed at that end; the reference is the own segment
    ink within ``end_window`` line heights inside the end. Both are mean row offsets from the sloped baseline.

    :param scene: Line scene.
    :type scene: LineScene
    :param members_mask: Member polygons in the window.
    :type members_mask: np.ndarray
    :param end_x: The end's page column (members' extreme).
    :type end_x: int
    :param direction: ``-1`` left end, ``+1`` right end.
    :type direction: int
    :param end: What the walk found there.
    :type end: EndExtent
    :param absorbed_mask: ``(h, w)`` boxes absorbed at this end.
    :type absorbed_mask: np.ndarray
    :param config: Thresholds.
    :type config: ExtentConfig
    :return: Shift in line heights (0.0 when the end took less than :data:`END_WINDOW_SHIFT` line heights of
        detached ink or no own ink is found).
    :rtype: float
    """
    lh = scene.line_height
    h, w = scene.shape
    cols = np.arange(w) + scene.x0
    if direction < 0:
        beyond = (cols >= end_x - end.extended_px) & (cols < end_x)
        inner = (cols >= end_x) & (cols < end_x + config.end_window * lh)
    else:
        beyond = (cols > end_x) & (cols <= end_x + end.extended_px)
        inner = (cols <= end_x) & (cols > end_x - config.end_window * lh)
    n_labels = scene.counted.size
    detached = np.zeros(n_labels, bool)
    detached[[k for k in end.components if 0 < k < n_labels]] = True
    if scene.connected is not None:
        detached &= ~scene.connected
    ink = scene.labels > 0
    far = (detached[scene.labels] & beyond[None, :] & ~scene.blocked) | (absorbed_mask & ink)
    if far.sum(axis=0).astype(bool).sum() < END_WINDOW_SHIFT * lh:
        return 0.0
    own = members_mask & ink & inner[None, :] & scene.label_mask(scene.connected)
    if not own.any():
        return 0.0
    base = scene.base - scene.y0
    rows_f, cols_f = np.nonzero(far)
    rows_o, cols_o = np.nonzero(own)
    return float(np.mean(rows_f - base[cols_f]) - np.mean(rows_o - base[cols_o])) / lh


def measure_polygon(ink: PageInk, baseline: Sequence[Point], polygon: Sequence[Point], line_height: float,
                    others: Sequence[Box] = (), config: ExtentConfig = ExtentConfig(),
                    block: Optional[Tuple[float, float]] = None, spacing: Optional[float] = None) -> float:
    """:func:`own_ink_share` of a given polygon (e.g. the segmenter's, when lines are not extended).

    :param ink: Page ink components.
    :type ink: PageInk
    :param baseline: The line's baseline.
    :type baseline: Sequence[Point]
    :param polygon: The polygon measured.
    :type polygon: Sequence[Point]
    :param line_height: Line height in pixels.
    :type line_height: float
    :param others: Every other segmenter line of the page.
    :type others: Sequence[Box]
    :param config: Thresholds.
    :type config: ExtentConfig
    :param block: ``(left, right)`` text block margins (None: the polygon's range plus the walk's span).
    :type block: Optional[Tuple[float, float]]
    :param spacing: Baseline spacing in pixels (None: the line height).
    :type spacing: Optional[float]
    :return: Share of the band's ink outside the polygon (the polygon's x-range plus ``own_ink_reach`` line
        heights, inside the block).
    :rtype: float
    """
    lh = max(1.0, line_height)
    xs = [p[0] for p in polygon]
    span = int(round((config.max_extend + config.gap + EXTEND_MAX) * lh))
    slack = config.block_slack * lh if math.isfinite(config.block_slack) else 0.0
    x_from, x_to = min(xs) - span, max(xs) + span
    if block is not None:
        x_from, x_to = min(x_from, block[0] - slack), max(x_to, block[1] + slack)
    scene = line_scene(ink, baseline, x_from, x_to, lh, others, config, [polygon], spacing)
    h, w = scene.shape
    mask = polygons_window([polygon], scene.x0, scene.y0, w, h)
    columns = reach_columns(scene, mask, config.own_ink_reach * lh,
                            block_columns(scene, block, slack, min(xs) - span, max(xs) + span))
    return own_ink_share(scene, mask, columns, scene.blocked, config)


def extend_line(ink: PageInk, baseline: Sequence[Point], members: Sequence[Sequence[Point]],
                line_height: float, others: Sequence[Box] = (), config: ExtentConfig = ExtentConfig(),
                block: Optional[Tuple[float, float]] = None, spacing: Optional[float] = None,
                hard: Optional[Tuple[Optional[float], Optional[float]]] = None,
                connected_only: Tuple[bool, bool] = (False, False), forced: Sequence[int] = (),
                extra_components: Sequence[int] = ()) -> LineExtent:
    """Extend a (possibly merged) line to the ink beyond both of its polygon ends and rebuild its polygon.

    :param ink: Page ink components.
    :type ink: PageInk
    :param baseline: The line's baseline (fragments concatenated left to right for a merged line).
    :type baseline: Sequence[Point]
    :param members: Boundary polygon of each member segment (one for a single segment).
    :type members: Sequence[Sequence[Point]]
    :param line_height: Median line height of the page.
    :type line_height: float
    :param others: Every other segmenter line of the page: those whose centre lies in the band are
        absorption candidates; all of them are subtracted from what the line gains, and those with a baseline
        clamp its band.
    :type others: Sequence[Box]
    :param config: Thresholds.
    :type config: ExtentConfig
    :param block: ``(left, right)`` margins of the text block at this line (:class:`TextBlock`); no end moves
        more than ``block_slack`` line heights beyond them (None or an infinite slack: unbounded) except along
        ink connected to the line's own letters (:func:`extend_end`).
    :type block: Optional[Tuple[float, float]]
    :param spacing: Baseline spacing of the page in pixels (None: the line height); sets the raise
        (:func:`raise_offsets`) and the detection of unsegmented neighbours.
    :type spacing: Optional[float]
    :param hard: ``(left, right)`` hard limits without slack (margin material, a gutter; None entries: none):
        unconnected ink beyond them never counts.
    :type hard: Optional[Tuple[Optional[float], Optional[float]]]
    :param connected_only: ``(left, right)``: only own-connected ink may move that end.
    :type connected_only: Tuple[bool, bool]
    :param forced: Keys of boxes (among ``others``) merged into the line unconditionally (a supralinear
        insertion's segment).
    :type forced: Sequence[int]
    :param extra_components: Ink labels added unconditionally (a supralinear insertion's ink).
    :type extra_components: Sequence[int]
    :return: The new geometry, what each end found and the measures (overlap, own ink outside, end shifts). A
        single segment whose polygon would change by less than ``sliver`` line heights in every column keeps
        its polygon and baseline exactly; a merged line is always rebuilt (never their convex hull, unless its
        pieces cannot be joined).
    :rtype: LineExtent
    """
    lh = max(1.0, line_height)
    s = max(1.0, spacing) if spacing else lh
    forced_set = set(forced)
    same_row = [b for b in others if b.key not in forced_set
                and in_band(baseline, b.centre(), lh, config.above, config.below)]
    xs = [p[0] for poly in members for p in poly]
    x_lo, x_hi = int(math.floor(min(xs))), int(math.ceil(max(xs)))
    gap = max(1, int(round(config.gap * lh)))
    cap = max(0, int(round(config.max_extend * lh)))
    span = cap + gap + max(cap, int(round(EXTEND_MAX * lh)))
    bounded = block is not None and math.isfinite(config.block_slack)
    slack = config.block_slack * lh if bounded else 0.0
    left_limit = right_limit = None
    w_from, w_to = x_lo - span - 1, x_hi + span + 1
    if bounded:
        left_limit = int(math.floor(x_lo - (block[0] - slack)))
        right_limit = int(math.floor(block[1] + slack - x_hi))
        w_from, w_to = min(w_from, block[0] - slack), max(w_to, block[1] + slack)
    hard_left, hard_right = hard if hard is not None else (None, None)
    if hard_left is not None and math.isfinite(hard_left):
        lim = int(math.floor(x_lo - hard_left))
        left_limit = lim if left_limit is None else min(left_limit, lim)
    if hard_right is not None and math.isfinite(hard_right):
        lim = int(math.floor(hard_right - x_hi))
        right_limit = lim if right_limit is None else min(right_limit, lim)
    x_limits = (x_lo - (left_limit if left_limit is not None else math.inf),
                x_hi + (right_limit if right_limit is not None else math.inf))
    if connected_only[0]:
        x_limits = (float(x_lo), x_limits[1])
    if connected_only[1]:
        x_limits = (x_limits[0], float(x_hi))
    scene = line_scene(ink, baseline, w_from, w_to, lh, others, config, members, s, x_limits)
    left = extend_end(ink, baseline, x_lo, -1, lh, same_row, config=config, limit=left_limit, scene=scene,
                      connected_only=connected_only[0])
    right = extend_end(ink, baseline, x_hi, +1, lh, same_row, config=config, limit=right_limit, scene=scene,
                       connected_only=connected_only[1])
    seam = {b.key for b in same_row if not b.steal and b.absorbable and x_lo < b.centre()[0] < x_hi
            and b.bbox()[2] - b.bbox()[0] < SEAM_BOX * lh and b.bbox()[3] - b.bbox()[1] < 1.2 * SEAM_BOX * lh
            and b.key in scene.boxes}
    forced_set |= seam
    absorbed = set(left.absorbed + right.absorbed) | {k for k in forced_set if k in scene.boxes}
    h, w = scene.shape
    members_mask = polygons_window(members, scene.x0, scene.y0, w, h)
    used_block = (float(block[0]), float(block[1])) if bounded else None
    used_hard = (hard_left, hard_right) if hard is not None else None
    common = dict(block=used_block, spacing=float(s), detected=scene.detected, hard=used_hard,
                  foreign_dropped=int(np.count_nonzero(scene.foreign)) if scene.foreign is not None else 0,
                  pencil_dropped=int(np.count_nonzero(scene.pencil)) if scene.pencil is not None else 0)
    if not members_mask.any():
        return LineExtent(tuple(baseline), tuple(members[0]), left, right, **common)
    absorbed_mask = np.zeros((h, w), bool)
    side_masks = {-1: np.zeros((h, w), bool), 1: np.zeros((h, w), bool)}
    for key in absorbed:
        box_mask = scene.box_mask(key)
        absorbed_mask |= box_mask
        if key in left.absorbed:
            side_masks[-1] |= box_mask
        elif key in right.absorbed:
            side_masks[1] |= box_mask
    blocked = scene.others_except(absorbed, config.other_dilate)
    region, stubs = line_shape(scene, members_mask, absorbed_mask, x_lo, x_hi,
                               (left.extended_px, right.extended_px), blocked, config,
                               left.components + right.components, extra_components)
    gained = (region & ~members_mask).sum(axis=0)
    changed = (bool(left.extended_px or right.extended_px or absorbed or extra_components) or len(members) > 1
               or (gained.size and gained.max() >= config.sliver * lh))
    block_cols = block_columns(scene, used_block, slack, x_lo - span, x_hi + span) if bounded else None
    shifts = (end_shift(scene, members_mask, x_lo, -1, left, side_masks[-1], config),
              end_shift(scene, members_mask, x_hi, 1, right, side_masks[1], config))
    common.update(end_shift=shifts, forced=tuple(sorted(k for k in forced_set if k in absorbed)),
                  stubs_removed=stubs)
    if not changed:
        columns = reach_columns(scene, members_mask, config.own_ink_reach * lh, block_cols)
        share = own_ink_share(scene, members_mask, columns, blocked, config)
        return LineExtent(tuple(baseline), tuple(members[0]), left, right, own_ink_outside=share, **common)
    outline, fell_apart = mask_outline(region, members_mask | absorbed_mask, scene.x0, scene.y0)
    polygon_mask = polygons_window([outline], scene.x0, scene.y0, w, h)
    overlap, overlap_key = new_overlap(scene, polygon_mask, members_mask, others, absorbed)
    columns = reach_columns(scene, polygon_mask, config.own_ink_reach * lh, block_cols)
    share = own_ink_share(scene, polygon_mask, columns, blocked, config)
    pad = config.strip_pad
    new_left = max(0.0, x_lo - left.extended_px - pad) if left.extended_px else None
    new_right = min(ink.shape[1] - 1.0, x_hi + right.extended_px + pad) if right.extended_px else None
    moved = moved_baseline(baseline, None if new_left is None else new_left + 1,
                           None if new_right is None else new_right - 1)
    return LineExtent(tuple(moved), tuple(outline), left, right, rebuilt=True, hull_fallback=fell_apart,
                      raised=config.raise_top, overlap=overlap, overlap_key=overlap_key, own_ink_outside=share,
                      **common)
