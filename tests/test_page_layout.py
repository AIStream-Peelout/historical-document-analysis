# File name: test_page_layout.py
# Date: 10/20/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Unit tests for the page-level layout of the ink walk (src/finetuning/kraken/page_layout.py).

Everything runs on synthetic lines and masks built in memory; nothing reads the NAS.
"""
import math
from typing import List, Tuple

import numpy as np
import pytest

from src.finetuning.kraken import line_extent as le
from src.finetuning.kraken import page_layout as pl

LH = 60.0


def lay(key: int, x0: float, x1: float, y: float, angle: float = 0.0) -> pl.LayoutLine:
    """A straight segmenter line (polygon 0.75 line heights above the baseline, 0.25 below).

    :param key: Index.
    :param x0: Left end.
    :param x1: Right end.
    :param y: Baseline.
    :param angle: Angle in degrees (only recorded).
    :return: The line.
    """
    return pl.LayoutLine(key, ((x0, y), (x1, y)), x0, x1, y - 0.75 * LH, y + 0.25 * LH, y, angle)


def column(x0: float, x1: float, first_key: int = 0, n: int = 8, y0: float = 200, step: float = 70
           ) -> List[pl.LayoutLine]:
    """A column of right-aligned lines with ragged left ends.

    :param x0: Leftmost start.
    :param x1: Right margin (every line starts there).
    :param first_key: Index of the first line.
    :param n: Lines.
    :param y0: First baseline.
    :param step: Baseline spacing.
    :return: Lines top to bottom.
    """
    return [lay(first_key + k, x0 + 15 * (k % 3), x1, y0 + step * k) for k in range(n)]


def poly(x0: float, y0: float, x1: float, y1: float) -> List[Tuple[float, float]]:
    """Rectangle polygon.

    :param x0: Left.
    :param y0: Top.
    :param x1: Right.
    :param y1: Bottom.
    :return: Corners.
    """
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def test_baseline_spacing_is_the_median_overlapping_neighbour_distance() -> None:
    """Spacing from horizontally overlapping adjacent baselines (70 px), not from polygon heights; a fragment on
    one row and the lines of another column (35 px lower, no x overlap) do not count; fewer than four pairs
    fall back to the line height."""
    lines = column(300, 900)
    lines.append(lay(50, 700, 760, 200 + 3 * 70 + 5))                   # fragment on row 3
    lines += column(1200, 1800, first_key=100, y0=235)
    assert pl.baseline_spacing(lines, 90.0) == pytest.approx(70.0)
    assert pl.baseline_spacing(column(300, 900, n=3), 90.0) == 90.0


def test_two_page_spread_gets_two_column_blocks() -> None:
    """A gap no long line crosses splits the page into two column groups; each gets its own block and core."""
    left = column(100, 800)
    right = column(1000, 1800, first_key=100)
    assert pl.column_groups(left + right, LH) == [(-math.inf, 900.0), (900.0, math.inf)]
    blocks = pl.page_blocks(left + right, LH, 70.0)
    assert len(blocks.columns) == 2
    a, b = blocks.columns
    assert a.block.left == pytest.approx(100, abs=2) and a.block.right == pytest.approx(800)
    assert b.block.left == pytest.approx(1000, abs=2) and b.block.right == pytest.approx(1800)
    assert blocks.column_of(400) is a and blocks.column_of(1500) is b
    assert pl.column_groups(column(100, 800), LH) == [(-math.inf, math.inf)]


def test_margin_column_is_a_hard_edge_and_leaves_the_block() -> None:
    """Short lines right of the main block (a margin column split into lines by the segmenter) are margin
    material: the block's right edge stays the main lines' margin, and their inner edge bounds the lines beside
    them (only on their rows); a short final word boxed apart on a main line's row is a fragment, not margin."""
    main = column(300, 900)
    margin = [lay(200 + k, 960, 1080, 270 + 70 * k) for k in range(4)]  # rows 270 .. 480
    fragment = lay(300, 200, 280, 200 + 70 * 6)                         # left of line 6 (starts at 300)
    blocks = pl.page_blocks(main + margin + [fragment], LH, 70.0)
    col = blocks.columns[0]
    assert set(col.margin_keys) == {200, 201, 202, 203}
    assert col.block.right == pytest.approx(900)
    assert col.hard_bounds(340) == (None, 960) and col.hard_bounds(200 + 70 * 7) == (None, None)
    rough = pl.page_blocks(main + margin, LH, 70.0, start_percentile=95.0)
    assert rough.columns[0].block.right == pytest.approx(900)


def test_steep_margin_line_is_margin_material() -> None:
    """A rotated (30 degree) line beside the block is margin material even when it is long."""
    main = column(300, 900)
    steep = pl.LayoutLine(9, ((950, 600), (1100, 300)), 950, 1100, 300, 600, 450, 30.0)
    col = pl.page_blocks(main + [steep], LH, 70.0).columns[0]
    assert col.margin_keys == (9,) and col.hard_bounds(450) == (None, 950)


def test_find_gutter_needs_a_wide_low_run_with_side_material_beyond() -> None:
    """A low run of at least 0.3 line heights followed by side material is a gutter; the page margin (nothing
    beyond) and a narrow dip are not."""
    profile = np.concatenate([np.full(100, 50.0), np.full(25, 2.0), np.full(40, 30.0), np.zeros(50)])
    assert pl.find_gutter(profile, 90, 1, 50.0, LH) == (100, 124)
    assert pl.find_gutter(np.concatenate([np.full(100, 50.0), np.zeros(80)]), 90, 1, 50.0, LH) is None
    narrow = np.concatenate([np.full(100, 50.0), np.full(10, 2.0), np.full(40, 30.0), np.zeros(50)])
    assert pl.find_gutter(narrow, 90, 1, 50.0, LH) is None
    mirrored = profile[::-1].copy()
    assert pl.find_gutter(mirrored, mirrored.size - 91, -1, 50.0, LH) == (mirrored.size - 101, mirrored.size - 125)


def test_page_blocks_find_a_gutter_before_unsegmented_side_material() -> None:
    """Ink of a margin column the segmenter never found, 30 px right of the block: the band projection shows a
    gutter, whose middle becomes the right hard limit."""
    lines = column(300, 900)
    mask = np.zeros((900, 1200), bool)
    for line in lines:
        y = int(line.y_mean)
        mask[y - 28:y + 4, line.x_min:900] = True
        mask[y - 28:y + 4, 930:1000] = True                              # side material
    col = pl.page_blocks(lines, LH, 70.0, mask).columns[0]
    assert col.gutters[1] is not None and 900 <= col.gutters[1] <= 930
    assert col.hard_bounds(400)[1] == col.gutters[1]


def ink_with(blocks: List[Tuple[int, int, int, int]], shape: Tuple[int, int] = (600, 800)) -> le.PageInk:
    """Labelled ink from filled rectangles.

    :param blocks: ``(x0, y0, x1, y1)`` rectangles (ends exclusive).
    :param shape: Page shape.
    :return: Page ink.
    """
    mask = np.zeros(shape, bool)
    for x0, y0, x1, y1 in blocks:
        mask[y0:y1, x0:x1] = True
    return le.label_ink(mask)


def test_resolve_overlaps_trims_the_line_farther_from_the_shared_ink() -> None:
    """Two written lines share a band holding a descender of the upper line: the lower line (farther from that
    ink) is trimmed and no longer shares more than 5 %; the upper one keeps its polygon."""
    ink = ink_with([(100, 172, 700, 204), (300, 204, 306, 236), (100, 252, 700, 284)])
    upper = ([(100, 200), (700, 200)], poly(100, 150, 700, 240))
    lower = ([(100, 280), (700, 280)], poly(100, 205, 700, 300))
    out = pl.resolve_overlaps(ink, {1: upper, 2: lower}, 0.05, LH)
    assert out[2].trimmed and not out[2].cut_own and not out[1].trimmed
    assert out[1].overlap <= 0.05 and out[2].overlap <= 0.05
    assert min(p[1] for p in out[2].polygon) >= 238 and list(out[1].polygon) == upper[1]


def test_resolve_overlaps_rejects_a_trim_that_would_cut_own_ink() -> None:
    """When the shared region also holds a raised word of the lower line (nearer its baseline), trimming it
    would cut its own ink: the lower line is rejected instead of trimmed; with ``max_overlap`` inf nothing is
    done and the overlap is only recorded."""
    ink = ink_with([(100, 172, 700, 204), (300, 204, 306, 236), (500, 238, 560, 248), (100, 252, 700, 284)])
    upper = ([(100, 200), (700, 200)], poly(100, 150, 700, 240))
    lower = ([(100, 280), (700, 280)], poly(100, 205, 700, 300))
    out = pl.resolve_overlaps(ink, {1: upper, 2: lower}, 0.05, LH)
    assert out[2].cut_own and not out[2].trimmed and not out[1].cut_own
    kept = pl.resolve_overlaps(ink, {1: upper, 2: lower}, math.inf, LH)
    assert not kept[2].trimmed and not kept[2].cut_own and kept[1].overlap > 0.3 and kept[1].key == 2
