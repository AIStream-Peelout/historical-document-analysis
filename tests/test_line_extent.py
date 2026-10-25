# File name: test_line_extent.py
# Date: 10/20/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Unit tests for the line ink-extent helpers (src/finetuning/kraken/line_extent.py).

Everything runs on synthetic masks and images built in memory; nothing reads the NAS.
"""
import dataclasses
import math
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.finetuning.kraken import line_extent as le

LH = 60.0                      # line height of the synthetic pages
BASE_Y = 500                   # baseline of the line under test
PAPER = (220, 205, 180)
INK = (45, 30, 20)
BLUE = (30, 80, 170)


def box_poly(x0: float, y0: float, x1: float, y1: float) -> List[Tuple[float, float]]:
    """Axis-aligned rectangle as a polygon.

    :param x0: Left.
    :param y0: Top.
    :param x1: Right.
    :param y1: Bottom.
    :return: Four corners clockwise from the top left.
    """
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def line_poly(x0: float, x1: float, y: float = BASE_Y) -> List[Tuple[float, float]]:
    """Polygon of a segmenter line: 0.75 line heights above the baseline, 0.25 below.

    :param x0: Left end.
    :param x1: Right end.
    :param y: Baseline.
    :return: Rectangle.
    """
    return box_poly(x0, y - 0.75 * LH, x1, y + 0.25 * LH)


def baseline(x0: float, x1: float, y: float = BASE_Y) -> List[Tuple[float, float]]:
    """Straight baseline, left to right.

    :param x0: Left end.
    :param x1: Right end.
    :param y: Height.
    :return: Two points.
    """
    return [(x0, y), (x1, y)]


def words(mask: np.ndarray, x0: int, x1: int, top: int, bottom: int, word: int = 40, space: int = 10) -> None:
    """Paint word blocks (ink rectangles separated by small spaces) into a mask, in place.

    :param mask: Boolean page.
    :param x0: Left end of the run.
    :param x1: Right end of the run.
    :param top: Top row of the blocks.
    :param bottom: Bottom row of the blocks (exclusive).
    :param word: Block width.
    :param space: Space between blocks.
    """
    x = x0
    while x < x1:
        mask[top:bottom, x:min(x + word, x1)] = True
        x += word + space


def page(width: int = 1200, height: int = 900) -> np.ndarray:
    """Empty boolean page.

    :param width: Width.
    :param height: Height.
    :return: All False.
    """
    return np.zeros((height, width), dtype=bool)


def inside(polygon: Sequence[Tuple[float, float]], x: float, y: float) -> bool:
    """Whether a point is inside (or on) a polygon.

    :param polygon: Points.
    :param x: Column.
    :param y: Row.
    :return: True inside or on the edge.
    """
    contour = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0


# ----------------------------------------------------------------------------- binarisation

def test_otsu_threshold_splits_two_modes_and_needs_two_levels() -> None:
    """A bimodal histogram splits between its modes; a one-level histogram has nothing to split."""
    hist = np.zeros(256)
    hist[30:40] = 100
    hist[200:210] = 900
    t = le.otsu_threshold(hist)
    assert 39 <= t < 200
    single = np.zeros(256)
    single[128] = 50
    assert le.otsu_threshold(single) == -1 and le.otsu_threshold(np.zeros(256)) == -1


def mounted_page() -> Image.Image:
    """Parchment with a blue mount on the left, a black mount strip along the top, a blue hole ringed by a
    dark rim, a brown stroke, a black stroke inside the fragment and a stain.

    :return: RGB image 600 x 400.
    """
    im = Image.new("RGB", (600, 400), PAPER)
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, 79, 399), fill=BLUE)                          # blue mount
    draw.rectangle((80, 0, 599, 29), fill=(12, 12, 14))                 # black mount touching the top border
    draw.rectangle((200, 150, 400, 160), fill=INK)                      # brown ink stroke
    draw.rectangle((200, 250, 400, 258), fill=(10, 10, 10))            # black ink, not touching the border
    draw.ellipse((470, 200, 530, 260), fill=(90, 70, 50))               # dark rim ...
    draw.ellipse((474, 204, 526, 256), fill=BLUE)                       # ... around a hole showing the mount
    draw.rectangle((150, 320, 450, 380), fill=(195, 180, 155))          # a broad, faint stain
    return im


def test_mount_mask_finds_blue_and_border_black_only() -> None:
    """The blue board and the dark strip on the border are mount; black ink inside the fragment is not."""
    rgb = mounted_page()
    mount = le.mount_mask(np.asarray(rgb.convert("HSV")))
    assert mount[200, 40] and mount[10, 300] and mount[230, 500]
    assert not mount[254, 300] and not mount[155, 300] and not mount[100, 300]


def test_ink_mask_masks_the_mount_rims_and_stains() -> None:
    """Strokes are ink; the blue and black mount, the rim around the hole and a broad faint stain are not."""
    ink = le.ink_mask(mounted_page(), LH)
    assert ink[155, 300] and ink[254, 300]
    assert not ink[200, 40] and not ink[10, 300]
    assert not ink[230, 472] and not ink[202, 500]                      # the rim is gone
    assert not ink[350, 300]                                            # the stain is below the background contrast
    assert not ink[100, 300]


def test_ink_mask_on_a_blank_page_is_empty() -> None:
    """A uniform page holds no ink (Otsu has nothing to split, the contrast floor holds)."""
    assert not le.ink_mask(Image.new("RGB", (300, 200), PAPER), LH).any()


def test_remove_specks_and_label_ink() -> None:
    """Small components go; labels carry area, centroid and an exclusive bounding box."""
    mask = page(100, 100)
    mask[10:20, 10:40] = True
    mask[60, 60] = True
    clean = le.remove_specks(mask, 6)
    assert clean[15, 20] and not clean[60, 60]
    ink = le.label_ink(clean)
    k = ink.labels[15, 20]
    assert ink.area[k] == 300 and ink.centroid_x[k] == pytest.approx(24.5) and ink.centroid_y[k] == pytest.approx(14.5)
    assert tuple(ink.bbox[k]) == (10, 10, 40, 20)


# ----------------------------------------------------------------------------- band

def test_baseline_y_interpolates_and_extrapolates_along_the_chord() -> None:
    """Inside the baseline: interpolation; outside: the end-to-end slope (not flat)."""
    base = [(100, 500), (300, 520), (500, 520)]                          # chord slope 20 / 400
    ys = le.baseline_y(base, np.array([200, 400, 0, 600]))
    assert ys[0] == pytest.approx(510) and ys[1] == pytest.approx(520)
    assert ys[2] == pytest.approx(495) and ys[3] == pytest.approx(525)
    assert le.baseline_y([(50, 80)], np.array([0, 100])).tolist() == [80, 80]


def test_band_limits_and_in_band() -> None:
    """The band reaches 0.75 line heights above the baseline and 0.3 below."""
    top, bottom = le.band_limits(baseline(0, 100), np.array([50]), LH)
    assert top[0] == pytest.approx(BASE_Y - 45) and bottom[0] == pytest.approx(BASE_Y + 18)
    assert le.in_band(baseline(0, 100), (50, BASE_Y - 40), LH)
    assert not le.in_band(baseline(0, 100), (50, BASE_Y - 50), LH)


def test_scan_run_bridges_short_gaps_and_stops_at_long_ones() -> None:
    """Runs continue over blanks shorter than the gap; a blank run of the gap's length ends the walk; the
    blank before the first ink counts; the limit caps the walk."""
    inked = [True, True, False, False, True, False, False, False, True]
    assert le.scan_run(inked, 3, 100) == 5
    assert le.scan_run(inked, 4, 100) == 9
    assert le.scan_run([False] * 3 + [True], 3, 100) == 0
    assert le.scan_run([False] * 2 + [True], 3, 100) == 3
    assert le.scan_run([True] * 50, 3, 10) == 10
    assert le.scan_run([], 3, 10) == 0


# ----------------------------------------------------------------------------- walk

def clipped_line(mask: np.ndarray) -> None:
    """Paint a line from x = 300 to 900 on the baseline (polygon later starts at 300).

    :param mask: Page.
    """
    words(mask, 300, 900, BASE_Y - 28, BASE_Y + 4)


def test_detached_raised_final_word_is_absorbed() -> None:
    """The line-final (left) word, written raised and detached by a word space, lies outside the polygon:
    the walk reaches it, the polygon grows to cover it and the baseline's left end moves with it."""
    mask = page()
    clipped_line(mask)
    mask[BASE_Y - 55:BASE_Y - 22, 230:290] = True                       # raised word, 10 px left of the polygon
    ink = le.label_ink(mask)
    ext = le.extend_line(ink, baseline(300, 900), [line_poly(300, 900)], LH)
    assert ext.left.extended_px >= 70 and ext.right.extended_px == 0
    assert ext.rebuilt and not ext.hull_fallback
    for x, y in [(231, BASE_Y - 54), (289, BASE_Y - 23), (260, BASE_Y - 40), (600, BASE_Y - 10)]:
        assert inside(ext.boundary, x, y)
    assert ext.baseline[0][0] <= 232 and ext.baseline[-1] == (900, BASE_Y)
    assert ext.left.ink_beyond_px == 0


def test_ink_gap_stops_the_walk() -> None:
    """A word space under 0.4 line heights is bridged; a gap of 0.4 line heights or more ends the walk."""
    mask = page()
    clipped_line(mask)
    mask[BASE_Y - 28:BASE_Y + 4, 260:290] = True                        # 10 px gap: reached
    mask[BASE_Y - 28:BASE_Y + 4, 190:230] = True                        # 30 px gap (> 24): not reached
    ext = le.extend_line(le.label_ink(mask), baseline(300, 900), [line_poly(300, 900)], LH)
    assert ext.left.extended_px == 40
    assert inside(ext.boundary, 265, BASE_Y - 10) and not inside(ext.boundary, 210, BASE_Y - 10)


def test_nothing_found_keeps_the_polygon_exactly() -> None:
    """A line with no ink beyond its ends keeps its polygon and baseline untouched."""
    mask = page()
    clipped_line(mask)
    poly = line_poly(298, 902)
    ext = le.extend_line(le.label_ink(mask), baseline(300, 900), [poly], LH)
    assert not ext.rebuilt and list(ext.boundary) == poly and list(ext.baseline) == baseline(300, 900)


def test_long_run_is_capped_and_reported_as_ink_beyond() -> None:
    """Ink running on far past 2 line heights: the end stops at the cap and the rest is ink beyond."""
    mask = page()
    words(mask, 20, 900, BASE_Y - 28, BASE_Y + 4)
    end = le.extend_end(le.label_ink(mask), baseline(300, 900), 300, -1, LH)
    assert end.capped and end.extended_px == 120
    assert end.ink_beyond_px > 0.4 * LH
    off = le.extend_end(le.label_ink(mask), baseline(300, 900), 300, -1, LH, config=le.ExtentConfig(max_extend=0))
    assert off.extended_px == 0 and not off.capped and off.ink_beyond_px > 2 * LH


def test_dots_and_neighbour_tails_do_not_extend_a_line() -> None:
    """Letter dots / dirt (small components) and a letter of the line above whose tail hangs into the band are
    not this line's ink; a raised final word whose centroid sits above the band (but that reaches no higher
    than the line above's letters) still is."""
    mask = page()
    clipped_line(mask)
    for x in (285, 270, 255):
        mask[BASE_Y - 20:BASE_Y - 15, x:x + 5] = True                   # dots, 25 px each
    mask[BASE_Y - 140:BASE_Y - 100, 200:290] = True                     # body of a letter of the line above ...
    mask[BASE_Y - 100:BASE_Y - 20, 280:288] = True                      # ... whose tail hangs into the band
    assert le.extend_line(le.label_ink(mask), baseline(300, 900), [line_poly(300, 900)], LH).left.extended_px == 0
    mask[BASE_Y - 140:BASE_Y - 20, 200:290] = False
    mask[BASE_Y - 62:BASE_Y - 36, 258:292] = True                       # raised word: centroid 0.82 lh up
    ink = le.label_ink(mask)
    k = ink.labels[BASE_Y - 50, 270]
    assert BASE_Y - ink.centroid_y[k] > 0.75 * LH                       # outside the band ...
    ext = le.extend_line(ink, baseline(300, 900), [line_poly(300, 900)], LH)
    assert ext.left.extended_px == 42 and inside(ext.boundary, 270, BASE_Y - 60)  # ... but counted, and covered


def test_separate_small_box_is_absorbed_and_a_neighbours_box_is_stolen() -> None:
    """A noise box holding the final word is absorbed into the line (its whole polygon); a box flagged as
    another edition line's text is reported as stolen and not absorbed; a box on another row blocks."""
    mask = page()
    clipped_line(mask)
    mask[BASE_Y - 30:BASE_Y + 2, 250:292] = True                        # the final word, in its own box
    final = le.Box(9, tuple(box_poly(246, BASE_Y - 40, 294, BASE_Y + 6)))
    ext = le.extend_line(le.label_ink(mask), baseline(300, 900), [line_poly(300, 900)], LH, [final])
    assert ext.absorbed == (9,) and ext.stolen == () and ext.left.extended_px >= 54
    assert all(inside(ext.boundary, x, y) for x, y in final.polygon)
    neighbour = le.Box(9, final.polygon, steal=True)
    ext = le.extend_line(le.label_ink(mask), baseline(300, 900), [line_poly(300, 900)], LH, [neighbour])
    assert ext.stolen == (9,) and ext.absorbed == ()
    above = le.Box(4, tuple(box_poly(200, BASE_Y - 130, 296, BASE_Y + 6)))  # centre outside the band: blocks
    ext = le.extend_line(le.label_ink(mask), baseline(300, 900), [line_poly(300, 900)], LH, [above])
    assert ext.left.extended_px == 0 and ext.absorbed == ()


def test_merged_line_is_rebuilt_as_the_union_of_its_members() -> None:
    """Two fragments at different heights become one polygon that follows them (not their convex hull):
    the gap between them is bridged by the band, and a point the hull would include is left out."""
    mask = page()
    words(mask, 300, 560, BASE_Y - 28, BASE_Y + 4)
    words(mask, 600, 900, BASE_Y - 58, BASE_Y - 26)
    left, right = line_poly(300, 560), line_poly(600, 900, BASE_Y - 30)
    base = baseline(300, 560) + baseline(600, 900, BASE_Y - 30)
    ext = le.extend_line(le.label_ink(mask), base, [left, right], LH)
    assert ext.rebuilt and not ext.hull_fallback and ext.left.extended_px == ext.right.extended_px == 0
    assert inside(ext.boundary, 580, BASE_Y - 15) and inside(ext.boundary, 700, BASE_Y - 50)
    hull = cv2.convexHull(np.array(left + right, dtype=np.float32))
    assert cv2.pointPolygonTest(hull, (550.0, BASE_Y - 60.0), False) >= 0
    assert not inside(ext.boundary, 550, BASE_Y - 60)


def test_union_outline_falls_back_to_the_hull_when_apart() -> None:
    """Disjoint pieces give their convex hull and say so; overlapping pieces give one outline."""
    outline, apart = le.union_outline([box_poly(0, 0, 10, 10), box_poly(5, 5, 20, 20)], 100, 100)
    assert not apart and inside(outline, 2, 2) and inside(outline, 18, 18) and not inside(outline, 18, 2)
    outline, apart = le.union_outline([box_poly(0, 0, 10, 10), box_poly(50, 50, 60, 60)], 100, 100)
    assert apart and inside(outline, 30, 30)


def test_moved_baseline_keeps_point_order() -> None:
    """A right-to-left baseline stays right to left with both ends moved outward."""
    assert le.moved_baseline([(900, 500), (300, 500)], 250, 950) == [(950, 500), (900, 500), (300, 500), (250, 500)]
    assert le.moved_baseline([(300, 500), (900, 500)], None, None) == [(300, 500), (900, 500)]
    assert le.moved_baseline([(300, 500), (900, 500)], 320, None) == [(300, 500), (900, 500)]  # never inward


def test_walk_on_a_rendered_page_with_a_mount() -> None:
    """End to end on an image: the walk extends to a clipped final word on the parchment and stops at the
    blue mount next to it (the board is not ink)."""
    im = Image.new("RGB", (1200, 900), PAPER)
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, 200, 899), fill=BLUE)
    for x in range(300, 900, 50):
        draw.rectangle((x, BASE_Y - 28, x + 39, BASE_Y + 3), fill=INK)
    draw.rectangle((240, BASE_Y - 28, 290, BASE_Y + 3), fill=INK)        # final word outside the polygon
    ink = le.page_ink(im, LH)
    ext = le.extend_line(ink, baseline(300, 900), [line_poly(300, 900)], LH)
    assert 55 <= ext.left.extended_px <= 64 and ext.left.ink_beyond_px == 0
    assert inside(ext.boundary, 245, BASE_Y - 10) and not inside(ext.boundary, 195, BASE_Y - 10)


# ----------------------------------------------------------------------------- round 2: block, masks, raise, measures

def sloped(x0: float, x1: float, y_at_x0: float, slope: float) -> List[Tuple[float, float]]:
    """Straight sloped baseline, left to right.

    :param x0: Left end.
    :param x1: Right end.
    :param y_at_x0: Height at the left end.
    :param slope: Rows per column.
    :return: Two points.
    """
    return [(x0, y_at_x0), (x1, y_at_x0 + slope * (x1 - x0))]


def sloped_poly(x0: float, x1: float, y_at_x0: float, slope: float, up: float, down: float
                ) -> List[Tuple[float, float]]:
    """Parallelogram polygon around a sloped baseline.

    :param x0: Left end.
    :param x1: Right end.
    :param y_at_x0: Baseline height at the left end.
    :param slope: Rows per column.
    :param up: Pixels above the baseline.
    :param down: Pixels below the baseline.
    :return: Four corners.
    """
    y1 = y_at_x0 + slope * (x1 - x0)
    return [(x0, y_at_x0 - up), (x1, y1 - up), (x1, y1 + down), (x0, y_at_x0 + down)]


def sloped_words(mask: np.ndarray, x0: int, x1: int, y_at: float, x_ref: float, slope: float, up: int = 28,
                 down: int = 4) -> None:
    """Paint word blocks that follow a sloped baseline (one column at a time), in place.

    :param mask: Boolean page.
    :param x0: Left end of the run.
    :param x1: Right end of the run.
    :param y_at: Baseline height at ``x_ref``.
    :param x_ref: Reference column.
    :param slope: Rows per column.
    :param up: Pixels above the baseline.
    :param down: Pixels below the baseline.
    """
    for x in range(x0, x1):
        if (x - x0) % 50 < 40:
            y = int(round(y_at + slope * (x - x_ref)))
            mask[y - up:y + down, x] = True


def polygon_mask(polygon: Sequence[Tuple[float, float]], shape: Tuple[int, int]) -> np.ndarray:
    """Rasterise a polygon on a page.

    :param polygon: Points.
    :param shape: ``(height, width)``.
    :return: Boolean mask.
    """
    return le.polygons_window([polygon], 0, 0, shape[1], shape[0])


def test_text_block_ignores_margin_notes_and_follows_thirds() -> None:
    """Percentile margins of the segmenter's own line ends; short lines beside the core column (a margin
    note) do not widen the block; per third the margins follow a slanted block."""
    spans = [(300 + 4 * n, 900 + 4 * n, 100 + 60 * n) for n in range(15)]
    notes = [(60, 220, 160 + 60 * n) for n in range(4)]                      # a margin column of short lines
    block = le.text_block(spans + notes)
    assert 300 <= block.left <= 310 and 950 <= block.right <= 960
    assert block.bounds(500) == (block.left, block.right)
    slanted = le.text_block(spans, thirds=True)
    top_left, top_right = slanted.bounds(100)
    bottom_left, bottom_right = slanted.bounds(940)
    assert bottom_left - top_left >= 30 and bottom_right - top_right >= 30
    assert le.text_block([]) is None


def test_margin_note_beyond_the_block_is_not_reached() -> None:
    """A margin note 5 px left of the line's end (closer than the walk's gap): unbounded, the walk swallows
    it (the round-1 failure); with the text block's margin at the line's end, the end moves at most the slack
    (0.3 line heights), the note's far part stays outside, and the ink past the block is reported, not gated."""
    mask = page()
    clipped_line(mask)
    mask[BASE_Y - 70:BASE_Y + 30, 230:295] = True                       # margin note / seal next to the block
    ink = le.label_ink(mask)
    free = le.extend_line(ink, baseline(300, 900), [line_poly(300, 900)], LH)
    assert free.left.extended_px >= 60
    bounded = le.extend_line(ink, baseline(300, 900), [line_poly(300, 900)], LH, block=(300.0, 900.0))
    slack = int(le.BLOCK_SLACK * LH)
    assert bounded.left.extended_px <= slack and bounded.left.limited and bounded.left.past_block_px > 0
    assert bounded.left.ink_beyond_px <= slack and not inside(bounded.boundary, 250, BASE_Y - 10)
    assert bounded.block == (300.0, 900.0)
    box = le.Box(7, tuple(box_poly(232, BASE_Y - 40, 270, BASE_Y + 6)))     # an absorbable box past the block
    assert le.extend_line(ink, baseline(300, 900), [line_poly(300, 900)], LH, [box], block=(300.0, 900.0)).absorbed == ()


def dense_slanted_page() -> Tuple[np.ndarray, List[le.Box], List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Three sloped lines 60 px apart (1.0 line height): each line's final (left) word lies outside its
    polygon, beside the neighbours' unsegmented final words; the middle line is the one under test.

    :return: ``(mask, boxes of the lines above and below, middle baseline, middle polygon)``.
    """
    slope = 0.12
    mask = page(1400, 900)
    for k in range(3):
        y = 300 + 60 * k
        sloped_words(mask, 470, 1200, y, 600, slope)
    boxes = []
    for k in (0, 2):
        y = 300 + 60 * k
        boxes.append(le.Box(k, tuple(sloped_poly(600, 1200, y, slope, 26, 8)), False,
                            tuple(sloped(600, 1200, y, slope))))
    return mask, boxes, sloped(600, 1200, 360, slope), sloped_poly(600, 1200, 360, slope, 26, 8)


def test_neighbour_letters_are_not_absorbed_on_a_slanted_dense_page() -> None:
    """With an inflated line height (90 px for 60 px spacing), a fixed band reaches into the neighbours: the
    walk counts their unsegmented final words. Clamped to the neighbours' baselines (and subtracting their
    polygons), the end follows the line's own slope: its own final word is covered, the words of the lines
    above and below are not."""
    mask, boxes, base, poly = dense_slanted_page()
    ink = le.label_ink(mask)
    lh = 90.0
    ext = le.extend_line(ink, base, [poly], lh, boxes, le.ExtentConfig(raise_top=True))
    assert ext.left.extended_px >= 120
    region = polygon_mask(ext.boundary, mask.shape)
    for k, inside_expected in ((0, False), (1, True), (2, False)):
        y = int(round(300 + 60 * k + 0.12 * (500 - 600)))
        own = mask[y - 28:y + 4, 480:510] & region[y - 28:y + 4, 480:510]
        assert own.any() == inside_expected, k
    assert ext.overlap <= 0.05
    loose = [le.Box(b.key, b.polygon) for b in boxes]                          # no baselines: no clamps
    round2 = le.ExtentConfig(raise_top=True, detect=False, foreign=False, spacing_above=math.inf,
                            spacing_below=math.inf)
    taken = le.extend_line(ink, base, [poly], lh, loose, round2)
    region = polygon_mask(taken.boundary, mask.shape)
    y = int(round(300 + 0.12 * (500 - 600)))
    assert (mask[y - 28:y + 4, 480:510] & region[y - 28:y + 4, 480:510]).any()
    # round 3: the raise reaches at most 0.55 of the measured spacing, so even without the neighbours'
    # baselines their final words stay out
    spaced = le.extend_line(ink, base, [poly], lh, loose, le.ExtentConfig(raise_top=True), spacing=60.0)
    region = polygon_mask(spaced.boundary, mask.shape)
    assert not (mask[y - 28:y + 4, 480:510] & region[y - 28:y + 4, 480:510]).any()
    assert (mask[y + 32:y + 64, 480:510] & region[y + 32:y + 64, 480:510]).any()


def test_raise_covers_a_raised_word_and_ascenders_inside_the_line() -> None:
    """A short polygon (top 0.5 line heights above the baseline) cuts a raised word and an ascender inside the
    x-range: the whole-line raise covers both (the word as an admitted component, the ascender as the line's
    own letter), never above the previous line's baseline + 0.15 line heights, with the same top at an
    extended end as in the middle (no L-shape); without the raise the polygon is kept. (Round 3: the line above
    sits 1.5 line heights up; at 1.17 line heights the raised word's centroid is nearer its baseline and the
    word is foreign, see the last assertion.)"""
    mask = page()
    clipped_line(mask)
    mask[BASE_Y - 47:BASE_Y - 30, 320:380] = True                       # raised word inside the x-range
    mask[BASE_Y - 44:BASE_Y, 600:606] = True                            # ascender
    mask[BASE_Y - 28:BASE_Y + 4, 255:292] = True                        # final word beyond the polygon
    poly = box_poly(300, BASE_Y - 30, 900, BASE_Y + 15)
    above = le.Box(1, tuple(box_poly(250, BASE_Y - 125, 950, BASE_Y - 82)), False,
                   tuple(baseline(250, 950, BASE_Y - 90)))
    ink = le.label_ink(mask)
    kept = le.extend_line(ink, baseline(300, 900), [poly], LH, [above], le.ExtentConfig(max_extend=0))
    assert not kept.rebuilt and list(kept.boundary) == poly
    ext = le.extend_line(ink, baseline(300, 900), [poly], LH, [above], le.ExtentConfig(raise_top=True))
    assert ext.rebuilt and ext.raised and ext.left.extended_px >= 40
    for x, y in [(350, BASE_Y - 46), (603, BASE_Y - 43), (270, BASE_Y - 20), (700, BASE_Y + 17)]:
        assert inside(ext.boundary, x, y), (x, y)
    region = polygon_mask(ext.boundary, mask.shape)
    top = lambda x: int(np.argmax(region[:, x]))                             # noqa: E731 - first row inside
    assert top(500) >= BASE_Y - 90 + 0.15 * LH - 1
    assert abs(top(270) - top(500)) <= 2 and abs(top(296) - top(500)) <= 2
    close = le.Box(1, tuple(box_poly(250, BASE_Y - 105, 950, BASE_Y - 62)), False,
                   tuple(baseline(250, 950, BASE_Y - 70)))
    near = le.extend_line(ink, baseline(300, 900), [poly], LH, [close], le.ExtentConfig(raise_top=True))
    assert not inside(near.boundary, 350, BASE_Y - 46) and inside(near.boundary, 603, BASE_Y - 43)


def test_seam_between_merged_pieces_is_not_full_height() -> None:
    """Two pieces of one line under a close line above, with an inflated line height (90 px): a round-1 band
    strip over the seam (0.75 line heights up) would climb into the line above's body. The seam follows the
    pieces' own extent; with the raise it covers the band but never rises above the line above's baseline
    plus 0.15 line heights; the line above's descender hanging into that zone is not taken without the raise."""
    mask = page()
    words(mask, 300, 560, BASE_Y - 28, BASE_Y + 4)
    words(mask, 640, 900, BASE_Y - 28, BASE_Y + 4)
    words(mask, 300, 900, BASE_Y - 98, BASE_Y - 66)                     # body of the line above
    mask[BASE_Y - 66:BASE_Y - 50, 590:596] = True                       # its descender, hanging over the seam
    left, right = box_poly(300, BASE_Y - 32, 560, BASE_Y + 10), box_poly(640, BASE_Y - 32, 900, BASE_Y + 10)
    above = le.Box(3, tuple(box_poly(300, BASE_Y - 112, 900, BASE_Y - 62)), False,
                   tuple(baseline(300, 900, BASE_Y - 70)))
    base = baseline(300, 560) + baseline(640, 900)
    clamp_row = BASE_Y - 70 + le.NEIGHBOUR_CLAMP * 90.0
    assert BASE_Y - 0.75 * 90.0 < clamp_row                             # a round-1 strip would reach above it
    for config in (le.ExtentConfig(), le.ExtentConfig(raise_top=True)):
        ext = le.extend_line(le.label_ink(mask), base, [left, right], 90.0, [above], config)
        assert ext.rebuilt and not ext.hull_fallback
        assert inside(ext.boundary, 600, BASE_Y - 10) and inside(ext.boundary, 600, BASE_Y + 5)
        region = polygon_mask(ext.boundary, mask.shape)
        assert int(np.argmax(region[:, 600])) >= clamp_row - 1
        assert not inside(ext.boundary, 600, BASE_Y - 62) and not inside(ext.boundary, 450, BASE_Y - 70)
        if not config.raise_top:
            assert not inside(ext.boundary, 600, BASE_Y - 45) and not inside(ext.boundary, 593, BASE_Y - 52)


def faint_page() -> Image.Image:
    """Parchment with many dark strokes, one of them ending in a long faint grey tail, and an isolated faint
    smear of the same grey.

    :return: RGB image 600 x 400.
    """
    im = Image.new("RGB", (600, 400), PAPER)
    draw = ImageDraw.Draw(im)
    for y in range(40, 360, 40):
        draw.rectangle((40, y, 300, y + 10), fill=INK)
    draw.rectangle((301, 120, 470, 124), fill=(170, 155, 135))           # faint tail of the stroke at y = 120
    draw.rectangle((430, 250, 434, 330), fill=(170, 155, 135))           # isolated faint smear
    return im


def test_adaptive_threshold_finds_faint_ink() -> None:
    """Otsu alone misses the faint grey tail of a stroke on a page of dark ink; the local adaptive rule finds
    it through its dark start (hysteresis), while an isolated faint smear and the paper stay out."""
    im = faint_page()
    otsu = le.ink_mask(im, LH, adaptive=False)
    both = le.ink_mask(im, LH)
    assert otsu[45, 100] and both[45, 100]
    assert not otsu[122, 420] and both[122, 420] and both[122, 465]
    assert not both[290, 432] and not both[200, 450] and not both[380, 300]
    strong = np.zeros((10, 12), bool)
    strong[2:5, 2:5] = True
    weak = np.zeros((10, 12), bool)
    weak[3, 6:11] = True                                                 # a tail starting 1 px after the stroke
    weak[8, 0:3] = True                                                  # a piece far from it
    assert not le.hysteresis(strong, weak, 9)[3, 10] and le.hysteresis(strong, weak, 9, jump=2)[3, 10]
    grown = le.hysteresis(strong, weak, 9, jump=2)
    assert grown[3, 6] and not grown[3, 5] and not grown[8, 1]
    assert not le.hysteresis(strong, weak, 10, jump=2)[3, 10]                     # the stroke is too small a seed


def test_own_ink_outside_counts_band_ink_left_outside_in_the_block() -> None:
    """Words of the line beyond a gap the walk does not bridge, inside the text block: measured over the whole
    block (``own_ink_reach`` inf, the round-2 measure) their share of the band's ink is reported, and another
    line's polygon over part of them removes that part from the measure. Round 3 measures only the line's
    x-range plus half a line height (the words lie 1 line height away: nothing outside)."""
    mask = page()
    words(mask, 100, 240, BASE_Y - 28, BASE_Y + 4)                      # 1 line height before the polygon
    clipped_line(mask)
    ink = le.label_ink(mask)
    total = mask[:, 100:900].sum()
    outside = mask[:, 100:300].sum()
    whole = le.ExtentConfig(own_ink_reach=math.inf)
    ext = le.extend_line(ink, baseline(300, 900), [line_poly(300, 900)], LH, block=(100.0, 900.0), config=whole)
    assert ext.left.extended_px == 0 and not ext.rebuilt
    assert ext.own_ink_outside == pytest.approx(outside / total, abs=0.01)
    other = le.Box(5, tuple(box_poly(90, BASE_Y - 40, 245, BASE_Y + 10)))
    covered = le.extend_line(ink, baseline(300, 900), [line_poly(300, 900)], LH, [other], block=(100.0, 900.0),
                             config=whole)
    assert covered.own_ink_outside < 0.01
    assert le.measure_polygon(ink, baseline(300, 900), line_poly(300, 900), LH, block=(100.0, 900.0),
                              config=whole) == pytest.approx(ext.own_ink_outside, abs=1e-9)
    local = le.extend_line(ink, baseline(300, 900), [line_poly(300, 900)], LH, block=(100.0, 900.0))
    assert local.own_ink_outside == 0.0


def test_pieces_that_cannot_join_fall_back_to_the_hull_and_report_the_overlap() -> None:
    """Another line's box fills the seam of a merged line: the seam may not take it, the pieces stay apart,
    the convex hull covers the box and the new overlap is reported for the gate."""
    mask = page()
    words(mask, 300, 500, BASE_Y - 28, BASE_Y + 4)
    words(mask, 560, 900, BASE_Y - 28, BASE_Y + 4)
    blocker = le.Box(8, tuple(box_poly(503, BASE_Y - 70, 557, BASE_Y + 40)))
    ext = le.extend_line(le.label_ink(mask), baseline(300, 500) + baseline(560, 900),
                         [line_poly(300, 500), line_poly(560, 900)], LH, [blocker])
    assert ext.hull_fallback and ext.overlap_key == 8 and ext.overlap > 0.05


def test_neighbour_rows_overlap_reach_and_gap() -> None:
    """The nearest baselines above and below per column: carried 2.5 line heights past a neighbour's end,
    never from a line that does not overlap this one horizontally (the other column of a two-column page) or
    that sits on this line's own row."""
    base = baseline(300, 900)
    cols = np.array([150, 400, 950, 1300])
    above = le.Box(1, tuple(box_poly(400, BASE_Y - 100, 900, BASE_Y - 50)), False, tuple(baseline(400, 900, BASE_Y - 60)))
    below = le.Box(2, tuple(box_poly(300, BASE_Y + 20, 800, BASE_Y + 80)), False, tuple(baseline(300, 800, BASE_Y + 62)))
    same_row = le.Box(3, tuple(box_poly(100, BASE_Y - 40, 290, BASE_Y + 10)), False, tuple(baseline(100, 290, BASE_Y + 5)))
    other_column = le.Box(4, tuple(box_poly(1000, BASE_Y - 70, 1500, BASE_Y - 20)), False,
                          tuple(baseline(1000, 1500, BASE_Y - 30)))
    noise = le.Box(5, tuple(box_poly(350, BASE_Y - 40, 360, BASE_Y - 30)))
    prev, nxt = le.neighbour_rows(base, cols, [above, below, same_row, other_column, noise], LH)
    assert prev[1] == prev[2] == BASE_Y - 60                            # 950: 50 px past the end of line 1
    assert np.isnan(prev[0]) and np.isnan(prev[3])                      # 150: 4.2 line heights before it
    assert nxt[0] == nxt[1] == nxt[2] == BASE_Y + 62 and np.isnan(nxt[3])  # line 4 never counts (no overlap)
    alone = le.neighbour_rows(base, cols, [other_column], LH, x_range=(300, 1100))
    assert alone[0][3] == BASE_Y - 30                                   # ... unless it overlaps the line


# ----------------------------------------------------------------------------- round 3: spacing, admission, crossing

def column_top(region: np.ndarray, x: int) -> int:
    """First row of a region in a column.

    :param region: Boolean page mask.
    :param x: Column.
    :return: Row index (the region's height when the column is empty).
    """
    col = region[:, x]
    return int(np.argmax(col)) if col.any() else region.shape[0]


def test_raise_follows_the_measured_spacing_on_an_inflated_line_height_page() -> None:
    """Line height 90 px for baselines 60 px apart: the round-2 raise (0.75 line heights, clamped to the line
    above's baseline + 0.15 line heights) reaches into the line above's descender zone and takes its tail; the
    round-3 raise stops at 0.55 of the spacing below the previous baseline's ... and the tail, nearer the line
    above's baseline, is subtracted."""
    mask = page()
    words(mask, 300, 900, BASE_Y - 28, BASE_Y + 4)                      # the line under test
    words(mask, 300, 900, BASE_Y - 88, BASE_Y - 56)                     # the line above (baseline BASE_Y - 60)
    mask[BASE_Y - 56:BASE_Y - 40, 500:506] = True                       # its tail hanging down
    above = le.Box(1, tuple(box_poly(300, BASE_Y - 92, 900, BASE_Y - 52)), False,
                   tuple(baseline(300, 900, BASE_Y - 60)))
    poly = box_poly(300, BASE_Y - 30, 900, BASE_Y + 8)
    ink = le.label_ink(mask)
    round2 = le.ExtentConfig(raise_top=True, spacing_above=math.inf, detect=False, foreign=False)
    old = le.extend_line(ink, baseline(300, 900), [poly], 90.0, [above], round2)
    assert inside(old.boundary, 503, BASE_Y - 42)
    new = le.extend_line(ink, baseline(300, 900), [poly], 90.0, [above], le.ExtentConfig(raise_top=True), spacing=60.0)
    region = polygon_mask(new.boundary, mask.shape)
    assert column_top(region, 700) >= BASE_Y - 0.55 * 60 - 1 and column_top(region, 700) > BASE_Y - 60
    assert not inside(new.boundary, 503, BASE_Y - 42) and inside(new.boundary, 700, BASE_Y - 20)
    assert new.spacing == 60.0


def test_unsegmented_line_above_is_detected_and_clamps_the_raise() -> None:
    """A line the segmenter never found sits 60 px above the line (no box, no baseline): its row-profile peak
    makes it a neighbour, so its tail hanging into the raise band is foreign and left out; without detection
    the band takes it."""
    mask = page()
    words(mask, 300, 900, BASE_Y - 28, BASE_Y + 4)
    words(mask, 300, 900, BASE_Y - 88, BASE_Y - 56)                     # unsegmented line, baseline ~BASE_Y - 60
    mask[BASE_Y - 56:BASE_Y - 29, 640:646] = True                       # its tail, down into the band
    poly = box_poly(300, BASE_Y - 26, 900, BASE_Y + 8)
    ink = le.label_ink(mask)
    blind = le.extend_line(ink, baseline(300, 900), [poly], LH, [], le.ExtentConfig(raise_top=True, detect=False))
    assert blind.detected == (False, False) and inside(blind.boundary, 643, BASE_Y - 31)
    seen = le.extend_line(ink, baseline(300, 900), [poly], LH, [], le.ExtentConfig(raise_top=True))
    assert seen.detected[0] and not inside(seen.boundary, 643, BASE_Y - 31)
    assert inside(seen.boundary, 700, BASE_Y - 20)


def test_detect_rows_finds_a_peak_between_baselines_only_with_valleys() -> None:
    """The row profile finds an unsegmented line between the line and a segmented line two spacings up, and
    nothing when the ink above is a plateau (no valley)."""
    mask = np.zeros((400, 300), bool)
    base = np.full(300, 300.0)
    mask[272:304, 20:280:2] = True                                       # own body
    mask[212:244, 20:280:2] = True                                       # unsegmented line, baseline ~240
    prev = np.full(300, 180.0)
    p, n, found = le.detect_rows(mask, base, np.arange(20, 280), prev, np.full(300, np.nan), 0, 60.0, 60.0)
    assert found == (True, False) and 236 <= p[150] <= 244 and np.isnan(n[150])
    flat = np.zeros((400, 300), bool)
    flat[200:304, 20:280:2] = True                                       # ink everywhere above: no separate peak
    _, _, found = le.detect_rows(flat, base, np.arange(20, 280), np.full(300, np.nan), np.full(300, np.nan), 0,
                                 60.0, 60.0)
    assert found == (False, False)


def test_pencil_component_is_not_counted_by_the_walk() -> None:
    """A thin stroke (a pencil shelfmark) 10 px beyond the line's end is far thinner than the line's own pen:
    it is never counted, so the walk does not extend to it; with the filter off it does."""
    mask = page()
    clipped_line(mask)
    mask[BASE_Y - 30:BASE_Y + 2, 284:287] = True                        # 3 px wide stroke, 13 px before the polygon
    mask[BASE_Y - 30:BASE_Y - 27, 250:287] = True                       # ... with a thin cross-bar
    ink = le.label_ink(mask)
    k = ink.labels[BASE_Y - 10, 285]
    assert ink.stroke[k] < 0.5 * ink.stroke[ink.labels[BASE_Y - 10, 600]]
    ext = le.extend_line(ink, baseline(300, 900), [line_poly(300, 900)], LH)
    assert ext.left.extended_px == 0 and ext.pencil_dropped >= 1
    off = le.extend_line(ink, baseline(300, 900), [line_poly(300, 900)], LH, config=le.ExtentConfig(pencil_share=0))
    assert off.left.extended_px >= 14


def test_component_darkness_is_the_median_and_page_ink_measures_it() -> None:
    """Darkness per component is its pixels' median; page_ink fills it from the image."""
    labels = np.array([[1, 1, 1, 0, 2]], np.int32)
    dark = np.array([[10, 50, 30, 0, 7]], np.uint8)
    assert le.component_darkness(labels, 2, dark).tolist() == [0.0, 30.0, 7.0]
    im = Image.new("RGB", (300, 200), PAPER)
    ImageDraw.Draw(im).rectangle((50, 80, 250, 95), fill=INK)
    ink = le.page_ink(im, LH)
    assert ink.darkness is not None and ink.darkness[ink.labels[88, 150]] > 100


def test_narrow_stub_of_an_unconnected_component_is_removed_but_an_own_ascender_kept() -> None:
    """A narrow stroke inside the x-range, not touching the line's letters, sticks out above the raise band:
    its protrusion is a stub and is removed; an own ascender of the same width is kept (it belongs to a
    component connected to the line's ink); with stub removal off the stub stays."""
    mask = page()
    clipped_line(mask)
    mask[BASE_Y - 50:BASE_Y - 31, 594:600] = True                       # loose stroke (word space at 590-600)
    mask[BASE_Y - 50:BASE_Y, 700:706] = True                            # own ascender (word 700-740)
    poly = box_poly(300, BASE_Y - 30, 900, BASE_Y + 8)
    ink = le.label_ink(mask)
    config = le.ExtentConfig(raise_top=True, pencil_share=0)              # (rectangles are no pen: no pencil test)
    ext = le.extend_line(ink, baseline(300, 900), [poly], LH, [], config)
    assert not inside(ext.boundary, 597, BASE_Y - 46) and inside(ext.boundary, 703, BASE_Y - 46)
    assert ext.stubs_removed >= 1
    kept = le.extend_line(ink, baseline(300, 900), [poly], LH, [], dataclasses.replace(config, stub_width=0))
    assert inside(kept.boundary, 597, BASE_Y - 46)


def test_own_swash_crosses_the_block_bound_along_connected_ink() -> None:
    """The line's last letter runs into a swash 50 px past the block's margin (0.3 line heights of slack): the
    walk follows the connected ink to its end and never cuts it; a separate glyph past the margin is not
    reached, and the round-2 rule (bound only) would have stopped at the slack."""
    mask = page()
    clipped_line(mask)
    mask[BASE_Y - 4:BASE_Y + 2, 250:310] = True                         # swash of the first word (300-340)
    mask[BASE_Y - 28:BASE_Y + 4, 200:235] = True                        # separate glyph 15 px beyond the swash
    ink = le.label_ink(mask)
    ext = le.extend_line(ink, baseline(300, 900), [line_poly(300, 900)], LH, block=(300.0, 900.0))
    assert 48 <= ext.left.extended_px <= 52 and ext.left.crossed_px >= 30
    assert inside(ext.boundary, 253, BASE_Y - 1) and not inside(ext.boundary, 220, BASE_Y - 10)
    assert ext.left.limited
    free = le.extend_line(ink, baseline(300, 900), [line_poly(300, 900)], LH,
                          config=le.ExtentConfig(own_touch=0), block=(300.0, 900.0))
    assert free.left.extended_px >= 48                                   # own segment ink alone still crosses


def test_hard_limit_takes_no_slack() -> None:
    """Margin material's edge (a hard limit) stops unconnected ink at the edge itself, without the block slack."""
    mask = page()
    clipped_line(mask)
    words(mask, 250, 292, BASE_Y - 28, BASE_Y + 4)                      # a separate word 8 px before the polygon
    ink = le.label_ink(mask)
    free = le.extend_line(ink, baseline(300, 900), [line_poly(300, 900)], LH)
    assert free.left.extended_px >= 48
    hard = le.extend_line(ink, baseline(300, 900), [line_poly(300, 900)], LH, hard=(295.0, None))
    assert hard.left.extended_px <= 5 and hard.hard == (295.0, None)


def test_descender_of_an_own_letter_is_kept_clear_of_the_next_line() -> None:
    """An own letter's tail reaching 0.8 line heights below the baseline stays in the polygon (round 2 cut it at
    0.5); when the next line's ink starts just below it, the tail stops 0.1 line heights above that ink."""
    mask = page()
    clipped_line(mask)
    mask[BASE_Y:BASE_Y + 48, 450:456] = True                            # tail of the word at 450-490
    poly = box_poly(300, BASE_Y - 30, 900, BASE_Y + 8)
    ink = le.label_ink(mask)
    ext = le.extend_line(ink, baseline(300, 900), [poly], LH, [], le.ExtentConfig(raise_top=True))
    assert inside(ext.boundary, 453, BASE_Y + 45)
    old = le.extend_line(ink, baseline(300, 900), [poly], LH, [], le.ExtentConfig(raise_top=True, own_down_cap=0.5))
    assert not inside(old.boundary, 453, BASE_Y + 45)
    words(mask, 300, 900, BASE_Y + 40, BASE_Y + 72)                     # next line, ink top at BASE_Y + 40
    mask[BASE_Y + 36:BASE_Y + 40, :] = False
    below = le.Box(2, tuple(box_poly(300, BASE_Y + 38, 900, BASE_Y + 80)), False,
                   tuple(baseline(300, 900, BASE_Y + 68)))
    ink = le.label_ink(mask)
    near = le.extend_line(ink, baseline(300, 900), [poly], LH, [below], le.ExtentConfig(raise_top=True))
    region = polygon_mask(near.boundary, mask.shape)
    assert not region[BASE_Y + 35:, 453].any() and region[BASE_Y + 20, 453]


def test_own_ink_measure_looks_half_a_line_height_beyond_the_ends_only() -> None:
    """With the walk off, own ink 20 px beyond the polygon end counts as outside; a word 1 line height away does
    not."""
    mask = page()
    clipped_line(mask)
    mask[BASE_Y - 28:BASE_Y + 4, 275:295] = True                        # 5-25 px beyond the end
    mask[BASE_Y - 28:BASE_Y + 4, 200:240] = True                        # 60-100 px beyond the end
    ink = le.label_ink(mask)
    no_walk = le.ExtentConfig(max_extend=0)
    ext = le.extend_line(ink, baseline(300, 900), [line_poly(300, 900)], LH, config=no_walk)
    near = 20 * 32
    total = mask[:, 270:900].sum()
    assert ext.own_ink_outside == pytest.approx(near / total, abs=0.01)


def test_end_shift_measures_a_detached_word_below_the_line() -> None:
    """The walk takes a detached final word sitting 0.53 line heights below the line's own ink (the next
    line's word on a dense slanted page): the end shift reports it; a raised word inside the band does not."""
    mask = page()
    clipped_line(mask)
    mask[BASE_Y + 4:BASE_Y + 36, 250:292] = True                        # dropped word: mean offset +20
    ink = le.label_ink(mask)
    ext = le.extend_line(ink, baseline(300, 900), [line_poly(300, 900)], LH)
    assert ext.left.extended_px >= 40 and ext.end_shift[0] == pytest.approx((20 - (-12)) / LH, abs=0.03)
    mask[BASE_Y + 4:BASE_Y + 36, 250:292] = False
    mask[BASE_Y - 36:BASE_Y - 8, 250:292] = True                        # raised word: mean offset -22
    ext = le.extend_line(le.label_ink(mask), baseline(300, 900), [line_poly(300, 900)], LH)
    assert abs(ext.end_shift[0]) < 0.25


def test_connected_only_end_takes_no_detached_glyph() -> None:
    """An end walked along own-connected ink only (an edition line ending in ':') leaves a detached filler glyph
    beyond it out (and does not report it as ink beyond for the gate)."""
    mask = page()
    clipped_line(mask)
    mask[BASE_Y - 20:BASE_Y + 2, 270:292] = True                        # filler glyph 8 px beyond the end
    ink = le.label_ink(mask)
    assert le.extend_line(ink, baseline(300, 900), [line_poly(300, 900)], LH).left.extended_px >= 25
    ext = le.extend_line(ink, baseline(300, 900), [line_poly(300, 900)], LH, connected_only=(True, False))
    assert ext.left.extended_px == 0 and ext.left.ink_beyond_px == 0 and not inside(ext.boundary, 280, BASE_Y - 10)


def test_forced_box_and_extra_components_join_the_line() -> None:
    """A supralinear insertion: a box forced into the line is kept whole; ink components given as extras are
    covered (they sit above the band)."""
    mask = page()
    clipped_line(mask)
    mask[BASE_Y - 50:BASE_Y - 38, 500:560] = True                       # insertion above the line
    ink = le.label_ink(mask)
    box = le.Box(7, tuple(box_poly(495, BASE_Y - 55, 565, BASE_Y - 33)))
    poly = box_poly(300, BASE_Y - 30, 900, BASE_Y + 8)
    plain = le.extend_line(ink, baseline(300, 900), [poly], LH, [box], le.ExtentConfig(raise_top=True))
    assert not inside(plain.boundary, 530, BASE_Y - 52)
    forced = le.extend_line(ink, baseline(300, 900), [poly], LH, [box], le.ExtentConfig(raise_top=True), forced=[7])
    assert forced.forced == (7,) and inside(forced.boundary, 530, BASE_Y - 52)
    k = int(ink.labels[BASE_Y - 45, 530])
    extra = le.extend_line(ink, baseline(300, 900), [poly], LH, [], le.ExtentConfig(raise_top=True),
                           extra_components=[k])
    assert inside(extra.boundary, 530, BASE_Y - 48)


def test_new_overlap_share_never_exceeds_one_and_ignores_specks() -> None:
    """The overlap share is taken of the other polygon's rasterised area (at most 1), and a speck-sized box
    inside a merged line's seam is not tested."""
    mask = page()
    words(mask, 300, 500, BASE_Y - 28, BASE_Y + 4)
    words(mask, 560, 900, BASE_Y - 28, BASE_Y + 4)
    speck = le.Box(8, tuple(box_poly(528, BASE_Y - 10, 532, BASE_Y - 6)))
    ext = le.extend_line(le.label_ink(mask), baseline(300, 500) + baseline(560, 900),
                         [line_poly(300, 500), line_poly(560, 900)], LH, [speck])
    assert ext.overlap == 0.0


def test_speck_box_in_a_merged_lines_seam_is_merged_not_a_hull() -> None:
    """A small noise box sitting in the seam between a merged line's pieces is merged into the line (round 2:
    the seam could not cross it, the pieces fell back to their hull and the overlap gate fired falsely)."""
    mask = page()
    words(mask, 300, 500, BASE_Y - 28, BASE_Y + 4)
    words(mask, 560, 900, BASE_Y - 28, BASE_Y + 4)
    speck = le.Box(8, tuple(box_poly(515, BASE_Y - 45, 545, BASE_Y + 20)))
    ext = le.extend_line(le.label_ink(mask), baseline(300, 500) + baseline(560, 900),
                         [line_poly(300, 500), line_poly(560, 900)], LH, [speck])
    assert not ext.hull_fallback and ext.forced == (8,) and ext.overlap == 0.0
    assert inside(ext.boundary, 530, BASE_Y - 10)


def test_own_ink_measure_needs_evidence() -> None:
    """When the line's letters are lost in a blob whose centroid lies nearer the next line (a stain or rim joined
    to them), the few counted pieces left are no evidence: the measure is 0.0 instead of a large false share."""
    mask = page()
    clipped_line(mask)
    mask[BASE_Y + 4:BASE_Y + 6, 300:900] = True                         # joins every word into one blob ...
    mask[BASE_Y + 6:BASE_Y + 300, 600:900] = True                       # ... with a stain far below
    mask[BASE_Y - 20:BASE_Y - 10, 905:925] = True                       # a small counted piece beyond the end
    below = le.Box(3, tuple(box_poly(300, BASE_Y + 200, 900, BASE_Y + 260)), False,
                   tuple(baseline(300, 900, BASE_Y + 250)))
    ink = le.label_ink(mask)
    ext = le.extend_line(ink, baseline(300, 900), [line_poly(300, 900)], LH, [below],
                         config=le.ExtentConfig(max_extend=0))
    assert ext.foreign_dropped >= 1 and ext.own_ink_outside == 0.0


def test_foreign_test_carries_a_short_neighbour_far_beyond_its_segment() -> None:
    """The next line's segment starts 3.3 line heights right of its first glyph, which hangs just below this
    line's left end touching its letters: carried 6 line heights (the foreign reach) the neighbour's baseline
    claims the glyph, so it is not taken as an own descender; with the clamp reach (2.5) it would be."""
    mask = page()
    clipped_line(mask)
    mask[BASE_Y + 6:BASE_Y + 60, 305:330] = True                        # next line's glyph, 2 px below ours
    below = le.Box(3, tuple(box_poly(530, BASE_Y + 40, 900, BASE_Y + 80)), False,
                   tuple(baseline(530, 900, BASE_Y + 60)))
    ink = le.label_ink(mask)
    poly = box_poly(300, BASE_Y - 30, 900, BASE_Y + 8)
    ext = le.extend_line(ink, baseline(300, 900), [poly], LH, [below], le.ExtentConfig(raise_top=True))
    assert ext.foreign_dropped >= 1 and not inside(ext.boundary, 317, BASE_Y + 40)
    short = le.extend_line(ink, baseline(300, 900), [poly], LH, [below],
                           le.ExtentConfig(raise_top=True, foreign_reach=2.5))
    assert inside(short.boundary, 317, BASE_Y + 40)
