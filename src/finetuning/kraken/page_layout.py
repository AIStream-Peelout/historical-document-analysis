# File name: page_layout.py
# Date: 10/20/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Page-level layout for the ink walk of ``align_edition_lines`` (round 3).

The per-line rebuild in :mod:`~src.finetuning.kraken.line_extent` needs three things measured on the whole page:

1. **Baseline spacing** (:func:`baseline_spacing`): the median vertical distance between horizontally
   overlapping adjacent segmenter baselines. The median polygon height over-estimates it on dense pages (the
   segmenter's polygons overlap), so the raise is limited to a share of the spacing.
2. **Column blocks** (:func:`page_blocks`). Segmenter lines are clustered into column groups by a vertical gap
   no long line crosses (:func:`column_groups`): a two-page spread or a two-column page gives one block per
   column. In each group, short (under :data:`MAIN_WIDTH_SHARE` of the block width) or steep (over
   :data:`MARGIN_STEEP` degrees) lines whose centre lies outside the long lines' block, and that are not a
   boxed-apart fragment of a long line on the same row, are **margin material** (:func:`margin_lines`): left
   out of the block's percentiles, their inner edge is a
   hard limit for the lines beside them. A **gutter** (:func:`find_gutter`) is a run of at least
   :data:`GUTTER_WIDTH` line heights where the group's band-restricted ink projection (:func:`band_projection`)
   falls to :data:`GUTTER_LOW` of the core's level, followed outward by side material; its middle is a hard
   limit too. Hard limits take no slack: the walk crosses them only along ink connected to the line's letters.
3. **Final overlaps** (:func:`resolve_overlaps`): after every line is rebuilt, a pair of written polygons
   sharing more than ``max_overlap`` of the smaller one is resolved by trimming the shared region from the
   line whose baseline is farther from the shared ink; when that region holds ink nearer the trimmed line's
   baseline (its own ink) the line is rejected instead.

Everything works on numpy arrays and point lists; nothing reads files.
"""
import math
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from src.finetuning.kraken import line_extent as le

Point = Tuple[float, float]

SPACING_MIN_LINES = 4       # fewer lines (with an overlapping neighbour below): the spacing is the line height
SPACING_MIN_GAP = 0.25      # baselines closer than this many line heights are one row (fragments)
COLUMN_MIN_LINES = 3        # a column group needs this many long lines
COLUMN_GUTTER = 0.5         # a vertical gap no long line crosses, at least this many line heights wide, splits groups
MAIN_WIDTH_SHARE = 0.35     # margin material: shorter than this share of the block width ...
MARGIN_STEEP = 20.0         # ... or steeper than this (degrees), with its centre outside the long lines' block
HARD_ROWS = 1.0             # a hard limit holds for lines whose baseline lies within its rows +- this many lh
FRAGMENT_REACH = 2.0        # a short line on a long line's row within this many lh of its end is a fragment of it
GUTTER_WIDTH = 0.3          # gutter: a low-ink run at least this many line heights wide ...
GUTTER_LOW = 0.15           # ... where the band projection is at most this share of the core's median ...
GUTTER_SIDE = 0.3           # ... with side material reaching this share of it ...
GUTTER_SEARCH = 3.0         # ... within this many line heights beyond the run
ROTATED = 45.0              # lines steeper than this (degrees) never shape a block (the aligner's STEEP_DEGREES)
START_PERCENTILE = 50.0     # a block's right edge (where right-to-left lines start): this percentile of the line ends


@dataclass(frozen=True)
class LayoutLine:
    """A segmenter line as the page layout sees it.

    :param key: Segmenter index.
    :param baseline: Baseline points.
    :param x_min: Polygon (else baseline) left edge.
    :param x_max: Polygon (else baseline) right edge.
    :param y_min: Polygon top.
    :param y_max: Polygon bottom.
    :param y_mean: Mean baseline row.
    :param angle: Baseline angle from the horizontal in degrees.
    """
    key: int
    baseline: Tuple[Point, ...]
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    y_mean: float
    angle: float

    @property
    def width(self) -> float:
        """Horizontal extent.

        :return: ``x_max - x_min``.
        :rtype: float
        """
        return self.x_max - self.x_min

    @property
    def x_center(self) -> float:
        """Middle of the x-range.

        :return: ``(x_min + x_max) / 2``.
        :rtype: float
        """
        return (self.x_min + self.x_max) / 2


def baseline_spacing(lines: Sequence[LayoutLine], line_height: float, min_lines: int = SPACING_MIN_LINES,
                     min_gap: float = SPACING_MIN_GAP) -> float:
    """Median vertical distance between horizontally overlapping adjacent baselines.

    For every line, the nearest line below it whose x-range overlaps its own is found; the distance is taken at
    the middle of the overlap along both sloped baselines. Pairs closer than ``min_gap`` line heights (fragments
    of one row) are ignored.

    :param lines: Segmenter lines (noise and rotated lines left out by the caller).
    :type lines: Sequence[LayoutLine]
    :param line_height: Median line height in pixels (the fallback).
    :type line_height: float
    :param min_lines: With fewer measured pairs the line height is returned.
    :type min_lines: int
    :param min_gap: Line heights.
    :type min_gap: float
    :return: Spacing in pixels.
    :rtype: float
    """
    gaps: List[float] = []
    for a in lines:
        if len(a.baseline) < 2:
            continue
        best: Optional[float] = None
        for b in lines:
            if b is a or len(b.baseline) < 2:
                continue
            lo, hi = max(a.x_min, b.x_min), min(a.x_max, b.x_max)
            if hi <= lo:
                continue
            xm = np.array([(lo + hi) / 2])
            dy = float(le.baseline_y(b.baseline, xm)[0] - le.baseline_y(a.baseline, xm)[0])
            if dy > min_gap * line_height and (best is None or dy < best):
                best = dy
        if best is not None:
            gaps.append(best)
    if len(gaps) < min_lines:
        return float(line_height)
    return float(statistics.median(gaps))


def column_groups(lines: Sequence[LayoutLine], line_height: float, min_lines: int = COLUMN_MIN_LINES,
                  gutter: float = COLUMN_GUTTER, main_share: float = MAIN_WIDTH_SHARE) -> List[Tuple[float, float]]:
    """Column groups of a page: x-ranges separated by a vertical gap that no long line crosses.

    :param lines: Segmenter lines (noise and rotated lines left out).
    :type lines: Sequence[LayoutLine]
    :param line_height: Line height in pixels.
    :type line_height: float
    :param min_lines: Long lines needed entirely on each side of a split.
    :type min_lines: int
    :param gutter: The gap is at least this many line heights wide.
    :type gutter: float
    :param main_share: Long lines are at least this share of the median line width.
    :type main_share: float
    :return: ``[(x_from, x_to)]`` left to right; the splits sit in the middle of the gaps (one group spanning
        everything when there is no split).
    :rtype: List[Tuple[float, float]]
    """
    if not lines:
        return [(-math.inf, math.inf)]
    median = statistics.median(l.width for l in lines)
    long = sorted((l for l in lines if l.width >= main_share * median), key=lambda l: l.x_min)
    splits: List[float] = []
    if long:
        reach = long[0].x_max
        for line in long[1:]:
            if line.x_min - reach >= gutter * line_height:
                left = sum(1 for l in long if l.x_max <= reach)
                right = sum(1 for l in long if l.x_min >= line.x_min)
                if left >= min_lines and right >= min_lines:
                    splits.append((reach + line.x_min) / 2)
            reach = max(reach, line.x_max)
    edges = [-math.inf] + splits + [math.inf]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def core_column(lines: Sequence[LayoutLine]) -> Tuple[float, float]:
    """The core of a column: median line start to median line end.

    :param lines: Lines of the column.
    :type lines: Sequence[LayoutLine]
    :return: ``(left, right)``.
    :rtype: Tuple[float, float]
    """
    return float(statistics.median(l.x_min for l in lines)), float(statistics.median(l.x_max for l in lines))


def margin_lines(lines: Sequence[LayoutLine], main: Sequence[LayoutLine], line_height: float,
                 main_share: float = MAIN_WIDTH_SHARE, steep: float = MARGIN_STEEP,
                 fragment_reach: float = FRAGMENT_REACH) -> List[LayoutLine]:
    """Margin material among a column's lines: short or steep lines whose centre lies outside the main block.

    The main block is the percentile block (:func:`~src.finetuning.kraken.line_extent.text_block`) of the long
    lines (at least ``main_share`` of their median width), so short margin lines never widen it. A short line
    on the row of a long line (baselines within half a line height) beyond its left end (where right-to-left
    lines end) by less than ``fragment_reach`` line heights, overlapping it by at most a quarter line height, is
    a fragment of that line (a final word boxed apart), not margin material.

    :param lines: Lines of the column (rotated ones included).
    :type lines: Sequence[LayoutLine]
    :param main: The column's long, non-rotated lines.
    :type main: Sequence[LayoutLine]
    :param line_height: Line height in pixels.
    :type line_height: float
    :param main_share: Shorter than this share of the main block's width is short.
    :type main_share: float
    :param steep: Steeper than this (degrees) is steep.
    :type steep: float
    :param fragment_reach: Line heights.
    :type fragment_reach: float
    :return: The margin lines.
    :rtype: List[LayoutLine]
    """
    if not main:
        return []
    block = le.text_block([(l.x_min, l.x_max, l.y_mean) for l in main])
    width = block.right - block.left
    main_keys = {l.key for l in main}
    out = []
    for l in lines:
        if l.key in main_keys or not (l.width < main_share * width or l.angle > steep):
            continue
        if block.left <= l.x_center <= block.right:
            continue
        fragment = l.angle <= steep and any(
            abs(m.y_mean - l.y_mean) < 0.5 * line_height
            and -0.25 * line_height <= m.x_min - l.x_max < fragment_reach * line_height for m in main)
        if not fragment:
            out.append(l)
    return out


def band_projection(mask: np.ndarray, lines: Sequence[LayoutLine], spacing: float, x_from: int, x_to: int
                    ) -> np.ndarray:
    """Ink per page column summed over the lines' bands (``0.55`` spacing above to ``0.45`` below each sloped
    baseline, carried beyond the line's ends).

    :param mask: ``(H, W)`` boolean ink.
    :type mask: np.ndarray
    :param lines: Lines whose bands are summed.
    :type lines: Sequence[LayoutLine]
    :param spacing: Baseline spacing in pixels.
    :type spacing: float
    :param x_from: First column.
    :type x_from: int
    :param x_to: Last column (exclusive).
    :type x_to: int
    :return: ``(x_to - x_from,)`` counts.
    :rtype: np.ndarray
    """
    h = mask.shape[0]
    x_from, x_to = max(0, x_from), min(mask.shape[1], x_to)
    out = np.zeros(max(0, x_to - x_from))
    if out.size == 0:
        return out
    sub = mask[:, x_from:x_to]
    cs = np.vstack([np.zeros((1, sub.shape[1]), np.int32), np.cumsum(sub, axis=0, dtype=np.int32)])
    cols = np.arange(x_from, x_to)
    idx = np.arange(cols.size)
    for line in lines:
        if len(line.baseline) < 2:
            continue
        base = le.baseline_y(line.baseline, cols)
        t = np.clip(np.floor(base - 0.55 * spacing), 0, h).astype(np.int64)
        b = np.clip(np.ceil(base + 0.45 * spacing) + 1, 0, h).astype(np.int64)
        out += cs[np.maximum(b, t), idx] - cs[t, idx]
    return out


def find_gutter(profile: np.ndarray, start: int, direction: int, level: float, line_height: float,
                width: float = GUTTER_WIDTH, low: float = GUTTER_LOW, side: float = GUTTER_SIDE,
                search: float = GUTTER_SEARCH) -> Optional[Tuple[int, int]]:
    """The first low-ink run outward from the core edge that has side material beyond it.

    :param profile: Ink per column (:func:`band_projection`).
    :type profile: np.ndarray
    :param start: Index of the core edge in ``profile``.
    :type start: int
    :param direction: ``+1`` searches to higher indices (the right margin), ``-1`` to lower ones.
    :type direction: int
    :param level: The core's median projection.
    :type level: float
    :param line_height: Line height in pixels.
    :type line_height: float
    :param width: Line heights the run needs.
    :type width: float
    :param low: The run's columns are at most this share of ``level``.
    :type low: float
    :param side: Side material reaches this share of ``level`` ...
    :type side: float
    :param search: ... within this many line heights beyond the run.
    :type search: float
    :return: ``(inner, outer)`` indices of the run (inner = nearer the core), or None.
    :rtype: Optional[Tuple[int, int]]
    """
    if level <= 0 or profile.size == 0:
        return None
    need = max(2, int(round(width * line_height)))
    reach = max(need, int(round(search * line_height)))
    n = profile.size
    i = start
    run_start: Optional[int] = None
    while 0 <= i < n:
        if profile[i] <= low * level:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and abs(i - run_start) >= need:
                beyond = profile[i:min(n, i + reach)] if direction > 0 else profile[max(0, i - reach + 1):i + 1]
                if beyond.size and beyond.max() >= side * level:
                    return run_start, i - direction
            run_start = None
        i += direction
    return None


@dataclass(frozen=True)
class ColumnBlock:
    """One column group's text block with its hard limits.

    :param x_from: Left edge of the group's x-range (a split between groups, or ``-inf``).
    :param x_to: Right edge of the group's x-range.
    :param block: Percentile margins of the group's main lines (:class:`~src.finetuning.kraken.line_extent.TextBlock`).
    :param core: ``(left, right)`` core column.
    :param hard: ``(y_from, y_to, x, side)`` hard limits from margin material (side ``-1`` left, ``+1`` right).
    :param gutters: ``(left, right)`` gutter middles (None: no gutter on that side).
    :param margin_keys: Segmenter indices of the margin material.
    """
    x_from: float
    x_to: float
    block: le.TextBlock
    core: Tuple[float, float]
    hard: Tuple[Tuple[float, float, float, int], ...] = ()
    gutters: Tuple[Optional[float], Optional[float]] = (None, None)
    margin_keys: Tuple[int, ...] = ()

    def hard_bounds(self, y: float) -> Tuple[Optional[float], Optional[float]]:
        """Hard limits at a height.

        :param y: A line's mean baseline row.
        :type y: float
        :return: ``(left, right)``: the innermost margin-material edge whose rows cover ``y`` and the gutter on
            each side (None where there is neither).
        :rtype: Tuple[Optional[float], Optional[float]]
        """
        left, right = self.gutters
        for y_from, y_to, x, side in self.hard:
            if not y_from <= y <= y_to:
                continue
            if side > 0:
                right = x if right is None else min(right, x)
            else:
                left = x if left is None else max(left, x)
        return left, right


@dataclass(frozen=True)
class PageBlocks:
    """The column blocks of a page.

    :param columns: Blocks left to right.
    """
    columns: Tuple[ColumnBlock, ...]

    def column_of(self, x: float) -> ColumnBlock:
        """The block whose x-range holds a column (the nearest one otherwise).

        :param x: Page column (a line's x-centre).
        :type x: float
        :return: The block.
        :rtype: ColumnBlock
        """
        for column in self.columns:
            if column.x_from <= x < column.x_to:
                return column
        return min(self.columns, key=lambda c: min(abs(x - c.x_from), abs(x - c.x_to)))


def page_blocks(lines: Sequence[LayoutLine], line_height: float, spacing: float, mask: Optional[np.ndarray] = None,
                thirds: bool = False, gutters: bool = True, steep: float = ROTATED,
                start_percentile: float = START_PERCENTILE) -> Optional[PageBlocks]:
    """Column blocks of a page with margin material and gutters.

    :param lines: Non-noise segmenter lines (rotated ones included: they can be margin material).
    :type lines: Sequence[LayoutLine]
    :param line_height: Line height in pixels.
    :type line_height: float
    :param spacing: Baseline spacing in pixels.
    :type spacing: float
    :param mask: ``(H, W)`` page ink (None: no gutter detection).
    :type mask: Optional[np.ndarray]
    :param thirds: Block margins per vertical third (:func:`~src.finetuning.kraken.line_extent.text_block`).
    :type thirds: bool
    :param gutters: Detect gutters (needs ``mask``).
    :type gutters: bool
    :param steep: Lines steeper than this (degrees) are rotated: never part of a block (they can be margin
        material).
    :type steep: float
    :param start_percentile: The block's right edge is this percentile of the main lines' right ends (their
        starts: the lines run right to left and start at an aligned margin, so the median is that margin and a
        few lines starting further right, a title or an address, do not widen it); the left edge stays the
        :data:`~src.finetuning.kraken.line_extent.BLOCK_LOW` percentile of the ragged line ends.
    :type start_percentile: float
    :return: The blocks, or None without lines.
    :rtype: Optional[PageBlocks]
    """
    flat = [l for l in lines if l.angle <= steep]
    if not flat:
        return None
    columns: List[ColumnBlock] = []
    for x_from, x_to in column_groups(flat, line_height):
        group = [l for l in lines if x_from <= l.x_center < x_to]
        flat_group = [l for l in group if l.angle <= steep]
        if not flat_group:
            continue
        median = statistics.median(l.width for l in flat_group)
        long = [l for l in flat_group if l.width >= MAIN_WIDTH_SHARE * median] or flat_group
        margins = margin_lines(group, long, line_height)
        margin_keys = {l.key for l in margins}
        main = [l for l in flat_group if l.key not in margin_keys] or flat_group
        core = core_column(main)
        block = le.text_block([(l.x_min, l.x_max, l.y_mean) for l in main], thirds, high=start_percentile)
        hard = []
        for l in margins:
            rows = (l.y_min - HARD_ROWS * line_height, l.y_max + HARD_ROWS * line_height)
            if l.x_center > (block.left + block.right) / 2:
                hard.append((rows[0], rows[1], l.x_min, 1))
            else:
                hard.append((rows[0], rows[1], l.x_max, -1))
        found: List[Optional[float]] = [None, None]
        if gutters and mask is not None:
            lo = int(max(0, math.floor(max(x_from, core[0] - (GUTTER_SEARCH + 4) * line_height))))
            hi = int(min(mask.shape[1], math.ceil(min(x_to, core[1] + (GUTTER_SEARCH + 4) * line_height))))
            if hi - lo > 2:
                profile = band_projection(mask, main, spacing, lo, hi)
                level = float(np.median(profile[int(core[0]) - lo:int(core[1]) - lo])) if core[1] > core[0] else 0.0
                right = find_gutter(profile, int(core[1]) - lo, 1, level, line_height)
                left = find_gutter(profile, int(core[0]) - lo, -1, level, line_height)
                if right is not None:
                    found[1] = lo + (right[0] + right[1]) / 2
                if left is not None:
                    found[0] = lo + (left[0] + left[1]) / 2
        columns.append(ColumnBlock(x_from, x_to, block, core, tuple(hard), (found[0], found[1]),
                                   tuple(sorted(margin_keys))))
    return PageBlocks(tuple(columns)) if columns else None


# ----------------------------------------------------------------------------- final overlaps

@dataclass(frozen=True)
class OverlapResult:
    """What the final-overlap pass did to one line.

    :param polygon: The (possibly trimmed) polygon.
    :param overlap: Largest share of the smaller polygon shared with another kept line afterwards.
    :param key: That line.
    :param trimmed: A shared region was trimmed from this polygon.
    :param cut_own: Trimming would cut this line's own ink (or split it): the line must be rejected.
    """
    polygon: Tuple[Point, ...]
    overlap: float = 0.0
    key: Optional[int] = None
    trimmed: bool = False
    cut_own: bool = False


def _raster(polygon: Sequence[Point], width: int, height: int) -> Tuple[int, int, np.ndarray]:
    """A polygon rasterised in its clamped bounding box.

    :param polygon: Points.
    :type polygon: Sequence[Point]
    :param width: Image width.
    :type width: int
    :param height: Image height.
    :type height: int
    :return: ``(x0, y0, mask)``.
    :rtype: Tuple[int, int, np.ndarray]
    """
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x0, y0 = max(0, int(math.floor(min(xs)))), max(0, int(math.floor(min(ys))))
    x1, y1 = min(width, int(math.ceil(max(xs))) + 1), min(height, int(math.ceil(max(ys))) + 1)
    return x0, y0, le.polygons_window([polygon], x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _shared(a: Tuple[int, int, np.ndarray], b: Tuple[int, int, np.ndarray]) -> Optional[Tuple[int, int, np.ndarray]]:
    """Intersection of two rasterised polygons.

    :param a: ``(x0, y0, mask)``.
    :type a: Tuple[int, int, np.ndarray]
    :param b: ``(x0, y0, mask)``.
    :type b: Tuple[int, int, np.ndarray]
    :return: ``(x0, y0, mask)`` of the shared pixels, or None when the boxes do not meet.
    :rtype: Optional[Tuple[int, int, np.ndarray]]
    """
    ax, ay, am = a
    bx, by, bm = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + am.shape[1], bx + bm.shape[1]), min(ay + am.shape[0], by + bm.shape[0])
    if x1 <= x0 or y1 <= y0:
        return None
    part = (am[y0 - ay:y1 - ay, x0 - ax:x1 - ax] & bm[y0 - by:y1 - by, x0 - bx:x1 - bx])
    return x0, y0, part


def _distance(baseline: Sequence[Point], x: float, y: float) -> float:
    """Vertical distance from a point to a sloped baseline.

    :param baseline: Baseline points.
    :type baseline: Sequence[Point]
    :param x: Column.
    :type x: float
    :param y: Row.
    :type y: float
    :return: ``|y - baseline(x)|``.
    :rtype: float
    """
    return abs(y - float(le.baseline_y(baseline, np.array([x]))[0]))


def resolve_overlaps(ink: le.PageInk, lines: Dict[int, Tuple[Sequence[Point], Sequence[Point]]],
                     max_overlap: float, line_height: float) -> Dict[int, OverlapResult]:
    """Trim the regions written lines share, or reject the line that would lose its own ink.

    Pairs are taken by decreasing shared area. A pair sharing more than ``max_overlap`` of the smaller polygon's
    area is resolved: the shared ink's centroid (the shared region's centroid when it holds no ink) decides
    which baseline is farther; that line is trimmed (shared region grown by one pixel removed, the piece
    holding its baseline kept) unless an ink component with at least a speck's worth of pixels in the shared
    region lies nearer its baseline than the other's (its own ink), or trimming splits its baseline across
    pieces: then it is rejected (``cut_own``) and leaves the pass. Afterwards every line's largest remaining
    share is recorded.

    :param ink: Page ink components.
    :type ink: le.PageInk
    :param lines: ``{key: (baseline, polygon)}`` of the written lines.
    :type lines: Dict[int, Tuple[Sequence[Point], Sequence[Point]]]
    :param max_overlap: Share of the smaller polygon tolerated (``inf``: nothing is trimmed).
    :type max_overlap: float
    :param line_height: Line height in pixels (sets the speck size).
    :type line_height: float
    :return: ``{key: result}`` for every line.
    :rtype: Dict[int, OverlapResult]
    """
    height, width = ink.shape
    rasters = {k: _raster(poly, width, height) for k, (_, poly) in lines.items() if len(poly) >= 3}
    polygons = {k: tuple(poly) for k, (_, poly) in lines.items()}
    trimmed: set = set()
    cut: set = set()
    speck = le.speck_area(line_height)
    keys = sorted(rasters)
    if math.isfinite(max_overlap):
        pairs = []
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                part = _shared(rasters[a], rasters[b])
                if part is not None and part[2].any():
                    pairs.append((int(part[2].sum()), a, b))
        for _, a, b in sorted(pairs, reverse=True):
            if a in cut or b in cut:
                continue
            part = _shared(rasters[a], rasters[b])
            if part is None:
                continue
            x0, y0, shared = part
            n = int(shared.sum())
            smaller = min(int(rasters[a][2].sum()), int(rasters[b][2].sum()))
            if n == 0 or n <= max_overlap * max(1, smaller):
                continue
            labels = ink.labels[y0:y0 + shared.shape[0], x0:x0 + shared.shape[1]]
            inked = shared & (labels > 0)
            rows, cols = np.nonzero(inked if inked.any() else shared)
            cy, cx = float(rows.mean()) + y0, float(cols.mean()) + x0
            base_a, base_b = lines[a][0], lines[b][0]
            loser, winner = (a, b) if _distance(base_a, cx, cy) > _distance(base_b, cx, cy) else (b, a)
            counts = np.bincount(labels[inked], minlength=ink.area.size) if inked.any() else np.zeros(1)
            own = False
            for comp in np.nonzero(counts >= speck)[0]:
                if comp == 0:
                    continue
                kx, ky = float(ink.centroid_x[comp]), float(ink.centroid_y[comp])
                if _distance(lines[loser][0], kx, ky) < _distance(lines[winner][0], kx, ky):
                    own = True
                    break
            if own:
                cut.add(loser)
                continue
            lx, ly, lm = rasters[loser]
            grown = np.zeros_like(lm)
            sy, sx = y0 - ly, x0 - lx
            grown[sy:sy + shared.shape[0], sx:sx + shared.shape[1]] = shared
            region = lm & ~le.dilate(grown, 1)
            anchor = np.zeros_like(lm)
            base_pts = [(x - lx, y - ly) for x, y in lines[loser][0]]
            if len(base_pts) >= 2:
                canvas = Image.new("1", (lm.shape[1], lm.shape[0]), 0)
                ImageDraw.Draw(canvas).line(base_pts, fill=1, width=3)
                anchor = np.asarray(canvas, dtype=bool) & region
            if not anchor.any():
                anchor = region
            if not region.any():
                cut.add(loser)
                continue
            outline, apart = le.mask_outline(region, anchor, lx, ly)
            if apart:
                cut.add(loser)
                continue
            polygons[loser] = tuple(outline)
            rasters[loser] = _raster(outline, width, height)
            trimmed.add(loser)
    out: Dict[int, OverlapResult] = {}
    for k in lines:
        best, best_key = 0.0, None
        if k in rasters and k not in cut:
            area = int(rasters[k][2].sum())
            for other in keys:
                if other == k or other in cut:
                    continue
                part = _shared(rasters[k], rasters[other])
                if part is None:
                    continue
                share = int(part[2].sum()) / max(1, min(area, int(rasters[other][2].sum())))
                if share > best:
                    best, best_key = share, other
        out[k] = OverlapResult(polygons[k], best, best_key, k in trimmed, k in cut)
    return out
