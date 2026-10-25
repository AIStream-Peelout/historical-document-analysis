# File name: test_align_edition_lines.py
# Date: 10/19/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Unit tests for the edition-to-segmenter line alignment (src/finetuning/kraken/align_edition_lines.py).

Everything runs on synthetic pages, editions and images under ``tmp_path``; nothing reads the NAS.
"""
import dataclasses
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import datasets
import numpy as np
import pytest
from lxml import etree
from PIL import Image, ImageDraw

from src.finetuning.kraken import align_edition_lines as ael
from src.finetuning.kraken import line_extent as le

NS = {"p": ael.PAGE_NS}

# Edition lines of a synthetic page (distinct words so that wrong placements score low).
EDITION = [
    "بسم الله الرحمن الرحيم",
    "كتابي اطال الله بقا مولاي الشيخ الجليل",
    "وصلت رسالتك مع الحامل وسررت بها كثيرا",
    "واعلم ان القمح قد غلا في السوق هذه الايام",
    "وقد بعثت اليك عشرين دينارا مع صاحبنا يوسف",
    "فاكتب لي بما يتجدد عندك من الاخبار",
    "والسلام عليك ورحمة الله وبركاته",
]


def seg(index: int, x0: float, x1: float, y: float, text: str = "", height: float = 60.0,
        confidence: Optional[float] = 0.7, dy: float = 0.0) -> Dict[str, Any]:
    """One segmenter line as the JSON stores it (straight baseline, box boundary).

    :param index: Segmenter index.
    :param x0: Left end of the baseline.
    :param x1: Right end of the baseline.
    :param y: Baseline y at the left end.
    :param text: Zero-shot read.
    :param height: Boundary height (above the baseline 3/4, below 1/4).
    :param confidence: Mean read confidence.
    :param dy: Rise of the baseline from left to right end.
    :return: ``{"index", "baseline", "boundary", "text", "mean_confidence"}``.
    """
    top, bottom = y - 0.75 * height, y + 0.25 * height
    return {"index": index, "baseline": [[x0, y], [(x0 + x1) / 2, y + dy / 2], [x1, y + dy]],
            "boundary": [[x0, top], [x1, top + dy], [x1, bottom + dy], [x0, bottom]],
            "text": text, "mean_confidence": confidence if text else None}


def line(index: int, x0: float, x1: float, y: float, text: str = "", **kw: Any) -> ael.SegLine:
    """A :class:`SegLine` built through :func:`seg` and :func:`ael.make_seg_line`.

    :param index: Segmenter index.
    :param x0: Left end.
    :param x1: Right end.
    :param y: Baseline y.
    :param text: Read.
    :param kw: Extra :func:`seg` arguments.
    :return: The line.
    """
    return ael.make_seg_line(seg(index, x0, x1, y, text, **kw))


def width_for(text: str, cpp: float = 0.04) -> float:
    """Baseline length that holds an edition line at a given characters-per-pixel.

    :param text: Edition line.
    :param cpp: Characters per pixel.
    :return: Width in pixels.
    """
    return len(ael.edition_skeleton(text)) / cpp


def synthetic_page(edition: Sequence[str], reads: Sequence[str], split_line: Optional[int] = None,
                   junk: bool = False, stem: str = "pgpar_1_0_page", image_path: str = "/nonexistent.jpg"
                   ) -> Dict[str, Any]:
    """A one-column page: one segmenter line per edition line, right-aligned at x = 1900, 150 px apart.

    :param edition: Edition lines (fix the widths).
    :param reads: Zero-shot read of each line (``""`` = unreadable).
    :param split_line: Edition line whose segment is split into two same-height fragments.
    :param junk: Add an empty-read short line between lines 2 and 3 (a speck the segmenter found).
    :param stem: Page stem.
    :param image_path: Image path to record.
    :return: Segmenter JSON of the page.
    """
    lines: List[Dict[str, Any]] = []
    for n, (text, read) in enumerate(zip(edition, reads)):
        y = 200 + 150 * n
        x1 = 1900.0
        x0 = x1 - width_for(text)
        if n == split_line:
            mid = (x0 + x1) / 2
            words = read.split()
            right_read, left_read = " ".join(words[:len(words) // 2]), " ".join(words[len(words) // 2:])
            lines.append(seg(len(lines), mid + 20, x1, y, right_read))
            lines.append(seg(len(lines), x0, mid - 20, y + 8, left_read))
        else:
            lines.append(seg(len(lines), x0, x1, y, read))
        if junk and n == 2:
            lines.append(seg(len(lines), 900, 1300, y + 75, ""))
    return {"stem": stem, "pgpid": stem.split("_")[1], "canonical_id": "", "image_path": image_path,
            "image_size": [2000, 1400], "expected_lines": len(edition), "n_lines": len(lines), "n_dropped": 0,
            "secs": 1.0, "source": "test", "lines": lines}


# ----------------------------------------------------------------------------- skeleton and similarity

def test_skeleton_folds_dotted_letter_classes() -> None:
    """Letters that share a rasm map to one class; ن stays its own class."""
    assert ael.skeleton("بتثٮ") == "بببب"
    assert ael.skeleton("جحخ") == "ححح"
    assert ael.skeleton("دذ رز سش صض طظ عغ فقڡ") == "دد رر سس صص طط عع ففف"
    assert ael.skeleton("ن") == "ن" and ael.skeleton("ن") != ael.skeleton("ب")
    assert ael.skeleton("يىئ") == "ييي"


def test_skeleton_folds_hamza_ta_marbuta_and_marks() -> None:
    """Hamza carriers, ta marbuta, tashkeel and tatweel fold away; spaces are kept and collapsed."""
    assert ael.skeleton("أإآٱ ؤ ة") == "اااا و ه"
    assert ael.skeleton("مُحَمَّد  ـالـ") == "محمد ال"
    assert ael.skeleton("  ") == ""


def test_skeleton_drops_non_arabic_characters() -> None:
    """Hebrew letters, Latin, digits and punctuation do not enter the skeleton."""
    assert ael.skeleton("ב ע abc 12 ، الله.") == "الله"


def test_edition_skeleton_removes_gap_tokens() -> None:
    """``[...]`` and editorial brackets are not text on the page."""
    assert ael.edition_skeleton("حسب [...] مقصودا") == "حسب مفصودا"


@pytest.mark.parametrize("a,b,expected", [("kitten", "sitting", 3), ("", "abc", 3), ("abc", "", 3),
                                          ("abc", "abc", 0), ("ab", "ba", 2), ("flaw", "lawn", 2)])
def test_levenshtein(a: str, b: str, expected: int) -> None:
    """Classic edit distances, in both argument orders.

    :param a: First string.
    :param b: Second string.
    :param expected: Distance.
    """
    assert ael.levenshtein(a, b) == expected
    assert ael.levenshtein(b, a) == expected


def test_similarity_bounds_and_dot_blindness() -> None:
    """Identical skeletons score 1, empty strings 0, and dot-only differences vanish."""
    assert ael.similarity("abc", "abc") == 1.0
    assert ael.similarity("", "abc") == 0.0 and ael.similarity("", "") == 0.0
    assert ael.similarity("abcd", "abxd") == pytest.approx(0.75)
    assert ael.similarity(ael.skeleton("فقير"), ael.skeleton("ڡٯىر")) == 1.0


def test_prefix_distances_match_the_dp_on_random_strings() -> None:
    """The bit-parallel distances equal the DP's for every prefix, including patterns longer than 64."""
    rng = random.Random(5)
    for _ in range(400):
        a = "".join(rng.choice("abc d") for _ in range(rng.randint(0, 90)))
        b = "".join(rng.choice("abc d") for _ in range(rng.randint(0, 90)))
        assert ael.prefix_distances(a, b) == [ael.levenshtein(a, b[:j]) for j in range(len(b) + 1)]


def test_span_similarity_finds_the_covered_part_of_a_line() -> None:
    """A read of the first half of a line matches that span exactly and leaves the rest out; a read of the
    whole line keeps the whole line; the span sim is never below the whole-line sim."""
    full = ael.edition_skeleton(EDITION[3])
    first_half = ael.skeleton(" ".join(EDITION[3].split()[:5]))
    sim, lost = ael.span_similarity(first_half, full)
    assert sim == pytest.approx(1.0) and lost == pytest.approx(1 - len(first_half) / len(full))
    assert ael.similarity(first_half, full) < 0.6
    assert ael.span_similarity(full, full) == (pytest.approx(1.0), 0.0)
    short = ael.skeleton(EDITION[3].split()[0])                                     # one word of nine: below half
    assert ael.span_similarity(short, full)[0] < 0.5
    rng = random.Random(3)
    for _ in range(200):
        read = " ".join("".join(rng.choice("بحدرس") for _ in range(rng.randint(1, 5))) for _ in range(rng.randint(1, 5)))
        line = " ".join("".join(rng.choice("بحدرس") for _ in range(rng.randint(1, 5))) for _ in range(rng.randint(1, 8)))
        assert ael.span_similarity(read, line)[0] >= ael.similarity(read, line) - 1e-12
    assert ael.span_similarity("", full) == (0.0, 0.0)


def test_span_fields_own_other_and_neighbour() -> None:
    """Own span, best other line and best neighbour (e-1 / e+1) come from the per-line span sims."""
    spans = [(0.3, 0.0), (0.7, 0.4), (0.5, 0.0), (0.9, 0.1)]
    assert ael.span_fields(spans, 1) == (0.7, 0.4, 0.9, 0.5)
    assert ael.span_fields(spans, 0) == (0.3, 0.0, 0.9, 0.7)


# ----------------------------------------------------------------------------- geometry

def test_make_seg_line_geometry_and_read_statistics() -> None:
    """Derived x-range, mean y, width, height, angle and letter count."""
    l = line(3, 100, 500, 300, "بسم الله", height=80)
    assert (l.index, l.x_min, l.x_max, l.y_mean) == (3, 100, 500, 300)
    assert l.width == pytest.approx(400) and l.height == pytest.approx(80)
    assert l.angle == pytest.approx(0) and l.letters == 7 and l.x_center == 300
    steep = ael.make_seg_line({"index": 0, "baseline": [[10, 10], [12, 300]], "boundary": [[0, 0], [20, 0], [20, 300]],
                               "text": "", "mean_confidence": None})
    assert steep.angle > 80


def test_is_noise_needs_tiny_baseline_and_empty_read() -> None:
    """A speck with no read is noise; a short line with a read or a long empty line is not."""
    params = ael.Params()
    assert ael.is_noise(line(0, 100, 110, 50), 400, params)
    assert not ael.is_noise(line(1, 100, 110, 50, "الله"), 400, params)
    assert not ael.is_noise(line(2, 100, 500, 50), 400, params)
    one_point = ael.make_seg_line({"index": 3, "baseline": [[5, 5]], "boundary": [[0, 0], [9, 0], [9, 9]], "text": "الله"})
    assert ael.is_noise(one_point, 400, params)


def test_mergeable_requires_same_height_and_disjoint_x() -> None:
    """Fragments sit on one baseline height and barely overlap in x."""
    params = ael.Params()
    right, left = line(0, 1000, 1800, 500), line(1, 300, 980, 520)
    assert ael.mergeable(right, left, 100, params)
    assert not ael.mergeable(right, line(2, 300, 1200, 500), 100, params)        # overlaps 200/800 = 25 %
    assert ael.mergeable(right, line(3, 300, 1100, 500), 100, params)            # overlaps 100/800 = 12.5 %
    assert not ael.mergeable(right, line(4, 300, 980, 560), 100, params)         # 60 px >= 0.5 * 100
    assert ael.x_overlap(right, line(5, 1700, 2000, 500)) == pytest.approx(100 / 300)


def test_group_mergeable_checks_every_pair_and_bounds() -> None:
    """Three fragments merge only when all of them are pairwise mergeable."""
    params = ael.Params()
    order = [line(0, 1300, 1800, 500), line(1, 800, 1250, 505), line(2, 300, 750, 495), line(3, 300, 1800, 650)]
    assert ael.group_mergeable(order, 0, 3, 100, params)
    assert ael.group_mergeable(order, 0, 1, 100, params)
    assert not ael.group_mergeable(order, 1, 3, 100, params)
    assert not ael.group_mergeable(order, 3, 2, 100, params)                    # runs past the end


def test_merged_read_puts_rightmost_fragment_first() -> None:
    """Arabic reads right to left, so the fragment with the larger x comes first."""
    group = [line(0, 300, 900, 500, "الثاني"), line(1, 1000, 1800, 500, "الاول"), line(2, 100, 250, 500, "")]
    assert ael.merged_read(group) == "الاول الثاني"


def test_convex_hull_drops_interior_and_collinear_points() -> None:
    """Monotone chain returns the four corners of a square, counter-clockwise."""
    pts = [(0, 0), (2, 0), (1, 0), (2, 2), (0, 2), (1, 1), (0, 1)]
    hull = ael.convex_hull(pts)
    assert sorted(hull) == [(0, 0), (0, 2), (2, 0), (2, 2)]
    area = sum(hull[i][0] * hull[(i + 1) % 4][1] - hull[(i + 1) % 4][0] * hull[i][1] for i in range(4)) / 2
    assert area == pytest.approx(4)
    assert ael.convex_hull([(1, 1), (0, 0)]) == [(0, 0), (1, 1)]


def test_merged_geometry_concatenates_baselines_left_to_right() -> None:
    """A merged line's baseline runs left to right across its fragments; its polygon is their hull."""
    right = line(0, 1000, 1800, 500)
    left = ael.make_seg_line({"index": 1, "baseline": [[900, 510], [300, 505]],          # given right-to-left
                              "boundary": [[300, 450], [900, 450], [900, 520], [300, 520]], "text": ""})
    baseline, boundary = ael.merged_geometry([right, left])
    assert [p[0] for p in baseline] == sorted(p[0] for p in baseline)
    assert baseline[0] == (300, 505) and baseline[-1] == (1800, 500)
    xs = [p[0] for p in boundary]
    assert min(xs) == 300 and max(xs) == 1800 and len(boundary) >= 4
    single_baseline, single_boundary = ael.merged_geometry([right])
    assert single_baseline == list(right.baseline) and single_boundary == list(right.boundary)


# ----------------------------------------------------------------------------- order

def two_columns(gap_x: Tuple[float, float] = (900, 1100), y_offset: float = 40) -> List[ael.SegLine]:
    """Four lines in a right column and four in a left column on independent heights.

    :param gap_x: ``(left column right edge, right column left edge)``.
    :param y_offset: Vertical offset of the left column's lines.
    :return: Lines (right column indices 0-3, left 4-7).
    """
    right = [line(i, gap_x[1], 1900, 200 + 150 * i) for i in range(4)]
    left = [line(4 + i, 100, gap_x[0], 200 + y_offset + 150 * i) for i in range(4)]
    return right + left


def test_column_split_on_wide_centre_gap() -> None:
    """Centres 1500 vs 500 on a 2000 px page: gap 1000 > 25 %."""
    split = ael.column_split(two_columns(), 2000, ael.Params())
    assert split is not None and 500 < split < 1500


def test_column_split_none_for_one_column_or_too_few_lines() -> None:
    """A single column, or a "column" of fewer than three lines, gives no split."""
    one = [line(i, 100 + 10 * i, 1900, 200 + 150 * i) for i in range(8)]
    assert ael.column_split(one, 2000, ael.Params()) is None
    lopsided = [line(i, 1100, 1900, 200 + 150 * i) for i in range(6)] + [line(9, 100, 300, 300)]
    assert ael.column_split(lopsided, 2000, ael.Params()) is None


def test_gutter_split_finds_bifolium_fold_and_rejects_crossing_lines() -> None:
    """A narrow empty gutter (too narrow for the centre-gap rule) splits; one crossing line removes it."""
    lines = two_columns(gap_x=(1000, 1060))
    params = ael.Params(column_gap=0.9)
    assert ael.column_split(lines, 2000, params) is None
    split = ael.gutter_split(lines, 2000, params)
    assert split is not None and 1000 < split < 1060
    assert ael.gutter_split(lines + [line(20, 200, 1800, 900)], 2000, params) is None


def test_order_lines_columns_right_first_and_steep_last() -> None:
    """Right column top to bottom, then the left column; a rotated line goes last."""
    lines = two_columns()
    steep = ael.make_seg_line({"index": 30, "baseline": [[1950, 1000], [1950, 200]],
                               "boundary": [[1930, 200], [1970, 200], [1970, 1000]], "text": ""})
    params = ael.Params()
    order = ael.order_lines(lines + [steep], 1000, True, params)
    assert [l.index for l in order] == [0, 1, 2, 3, 4, 5, 6, 7, 30]
    assert [l.index for l in ael.order_lines(lines, 1000, False, params)] == [4, 5, 6, 7, 0, 1, 2, 3]
    assert [l.index for l in ael.order_lines(lines, None, True, params)] == [0, 4, 1, 5, 2, 6, 3, 7]


def test_candidate_orders_offers_both_column_orders_only_with_a_split() -> None:
    """One ordering for one column; three when a split exists, the two-column ones carrying the split x."""
    params = ael.Params()
    assert list(ael.candidate_orders([line(i, 100, 1900, 200 + 150 * i) for i in range(5)], 2000, params)) == ["one_column"]
    orders = ael.candidate_orders(two_columns(), 2000, params)
    assert set(orders) == {"one_column", "two_columns_right_first", "two_columns_left_first"}
    assert orders["one_column"][1] is None and 500 < orders["two_columns_right_first"][1] < 1500


def witness_page() -> List[ael.SegLine]:
    """Three full-width lines, then a right block of three short lines beside a left block of two.

    :return: Lines; right block 3-5, left block 6-7 on heights interleaved with the right block.
    """
    main = [line(i, 100, 1900, 200 + 150 * i) for i in range(3)]
    right = [line(3 + i, 1100, 1900, 700 + 100 * i) for i in range(3)]
    left = [line(6 + i, 100, 800, 750 + 100 * i) for i in range(2)]
    return main + right + left


def test_block_gutter_and_block_order_read_right_block_first() -> None:
    """Short lines leave a gutter; runs between wide lines are read right block first."""
    lines = witness_page()
    params = ael.Params()
    assert ael.column_split(lines, 2000, params) is None and ael.gutter_split(lines, 2000, params) is None
    gutter = ael.block_gutter(lines, 2000, params)
    assert gutter is not None and 800 < gutter < 1100
    assert [l.index for l in ael.order_lines(lines, None, True, params)] == [0, 1, 2, 3, 6, 4, 7, 5]
    assert [l.index for l in ael.block_order(lines, gutter, params)] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert "blocks" in ael.candidate_orders(lines, 2000, params)


def test_block_gutter_needs_two_short_lines_per_side() -> None:
    """A single short line beside a block is not a block layout."""
    lines = witness_page()[:-1]
    assert ael.block_gutter(lines, 2000, ael.Params()) is None


# ----------------------------------------------------------------------------- scores and DP

def test_length_score() -> None:
    """Exact fit 1, off by half 0.5, off by the whole length or more 0."""
    assert ael.length_score(1000, 0.04, 40) == pytest.approx(1.0)
    assert ael.length_score(500, 0.04, 40) == pytest.approx(0.5)
    assert ael.length_score(3000, 0.04, 40) == 0.0
    assert ael.length_score(10, 0.04, 0) == pytest.approx(0.6)


def test_pair_score_uses_text_only_for_readable_reads() -> None:
    """0.6 sim + 0.4 length for a read of four letters or more, else half the length score."""
    params = ael.Params()
    assert ael.pair_score(0.5, 1.0, 4, params) == pytest.approx(0.7)
    assert ael.pair_score(0.9, 1.0, 3, params) == pytest.approx(0.5)


def matrix_pair(scores: Dict[Tuple[int, Tuple[int, ...]], float], default: float = 0.0
                ) -> Callable[[int, int, int], float]:
    """``pair(j, i, k)`` reading a score table keyed by ``(j, segmenter positions)``.

    :param scores: Scores.
    :param default: Score of an unlisted placement.
    :return: The callable :func:`ael.align` expects.
    """
    return lambda j, i, k: scores.get((j, tuple(range(i, i + k))), default)


def test_align_identity() -> None:
    """Strong diagonal scores give the 1:1 path."""
    scores = {(j, (j,)): 0.9 for j in range(3)}
    score, steps = ael.align(3, 3, matrix_pair(scores, 0.1), [0.05] * 3, 0.3, lambda i, k: False, 3)
    assert [(s.kind, s.e, s.s) for s in steps] == [("match", 0, (0,)), ("match", 1, (1,)), ("match", 2, (2,))]
    assert score == pytest.approx(2.7)


def test_align_skips_junk_segmenter_line() -> None:
    """An extra empty line in the middle is skipped cheaply instead of shifting the matches."""
    scores = {(0, (0,)): 0.9, (1, (2,)): 0.9, (2, (3,)): 0.9}
    _, steps = ael.align(3, 4, matrix_pair(scores, 0.1), [0.05] * 4, 0.3, lambda i, k: False, 3)
    assert [(s.kind, s.e, s.s) for s in steps] == [("match", 0, (0,)), ("skip_s", None, (1,)),
                                                   ("match", 1, (2,)), ("match", 2, (3,))]


def test_align_merges_fragments_when_allowed() -> None:
    """Two fragments merged score better than one fragment plus a skip, but only when mergeable."""
    scores = {(0, (0,)): 0.9, (1, (1,)): 0.4, (1, (1, 2)): 0.9, (2, (3,)): 0.9}
    _, steps = ael.align(3, 4, matrix_pair(scores, 0.0), [0.05] * 4, 0.3, lambda i, k: i == 1 and k == 2, 3)
    assert ("match", 1, (1, 2)) in [(s.kind, s.e, s.s) for s in steps]
    _, steps = ael.align(3, 4, matrix_pair(scores, 0.0), [0.05] * 4, 0.3, lambda i, k: False, 3)
    assert ("match", 1, (1, 2)) not in [(s.kind, s.e, s.s) for s in steps]


def test_align_skips_missing_edition_line() -> None:
    """An edition line the segmenter missed is skipped rather than displacing a good match."""
    scores = {(0, (0,)): 0.9, (2, (1,)): 0.9}
    _, steps = ael.align(3, 2, matrix_pair(scores, 0.0), [0.05] * 2, 0.3, lambda i, k: False, 3)
    assert [(s.kind, s.e) for s in steps] == [("match", 0), ("skip_e", 1), ("match", 2)]


def make_pairs(steps: Sequence[ael.Step], sims: Sequence[float], letters: Sequence[int],
               off: Optional[Sequence[float]] = None, widths: Optional[Sequence[float]] = None) -> List[ael.Pair]:
    """Pairs for the match steps of a path (edition lines of 40 characters; span sims equal to the sims).

    :param steps: Path.
    :param sims: Sim of each match step in order.
    :param letters: Read letters of each match step.
    :param off: Off-diagonal sims (default 0.2 each).
    :param widths: Segment widths (default 1000 px each, i.e. 0.04 characters per pixel).
    :return: Pairs.
    """
    matches = [(n, s) for n, s in enumerate(steps) if s.kind == "match"]
    off = off if off is not None else [0.2] * len(matches)
    widths = widths if widths is not None else [1000.0] * len(matches)
    return [ael.Pair(n, s.e, s.s, sim, 0.9, let, o, width=w, length=40, cover_sim=sim)
            for (n, s), sim, let, o, w in zip(matches, sims, letters, off, widths)]


def test_assign_tiers_anchor_and_interpolated_between_anchors() -> None:
    """Unreadable lines between two anchors with equal counts are interpolated; the line after is not."""
    steps = [ael.Step("match", j, (j,)) for j in range(4)] + [ael.Step("skip_s", None, (4,)), ael.Step("match", 4, (5,))]
    pairs = make_pairs(steps, [0.8, 0.0, 0.0, 0.7, 0.0], [20, 0, 0, 15, 0])
    identity, page_sim, margin, readable = ael.assign_tiers(steps, pairs, ael.Params())
    assert [p.tier for p in pairs] == ["anchor", "interpolated", "interpolated", "anchor", "rejected"]
    assert pairs[-1].reason == ael.REJECT_OUTSIDE and pairs[1].reason == ""
    assert not identity and readable == 2 and page_sim == pytest.approx(0.75) and margin == pytest.approx(0.55)


def test_assign_tiers_no_interpolation_across_a_skip() -> None:
    """A skipped segmenter line between anchors means unequal counts: nothing is interpolated."""
    steps = [ael.Step("match", 0, (0,)), ael.Step("match", 1, (1,)), ael.Step("skip_s", None, (2,)),
             ael.Step("match", 2, (3,))]
    pairs = make_pairs(steps, [0.8, 0.1, 0.8], [20, 2, 20])
    ael.assign_tiers(steps, pairs, ael.Params())
    assert [p.tier for p in pairs] == ["anchor", "rejected", "anchor"]
    assert pairs[1].reason == ael.REJECT_UNEQUAL


def test_assign_tiers_anchor_needs_enough_letters() -> None:
    """A high sim on a five-letter read is not an anchor."""
    steps = [ael.Step("match", 0, (0,)), ael.Step("skip_s", None, (1,))]
    pairs = make_pairs(steps, [0.9], [5])
    ael.assign_tiers(steps, pairs, ael.Params())
    assert pairs[0].tier == "rejected"


def test_assign_tiers_count_match_with_guard() -> None:
    """An identity path passes when the readable pairs beat their off-diagonal control."""
    steps = [ael.Step("match", j, (j,)) for j in range(5)]
    pairs = make_pairs(steps, [0.35, 0.3, 0.0, 0.4, 0.3], [10, 10, 0, 10, 10], [0.2] * 5)
    identity, page_sim, margin, readable = ael.assign_tiers(steps, pairs, ael.Params())
    assert identity and readable == 4 and margin == pytest.approx(0.1375)
    assert [p.tier for p in pairs] == ["count_match"] * 5


def test_assign_tiers_wrong_side_guard_blocks_count_match() -> None:
    """Same counts but reads no closer to their own lines than to the others (a wrong side): rejected."""
    steps = [ael.Step("match", j, (j,)) for j in range(5)]
    pairs = make_pairs(steps, [0.3, 0.25, 0.0, 0.3, 0.28], [10, 10, 0, 10, 10], [0.28, 0.27, 0.0, 0.27, 0.29])
    identity, page_sim, margin, _ = ael.assign_tiers(steps, pairs, ael.Params())
    assert identity and page_sim >= ael.SIM_PAGE and margin < ael.SIM_MARGIN
    assert {p.tier for p in pairs} == {"rejected"} and {p.reason for p in pairs} == {ael.REJECT_GUARD}
    few = make_pairs(steps, [0.6, 0.0, 0.0, 0.0, 0.45], [10, 0, 0, 0, 10])
    ael.assign_tiers(steps, few, ael.Params())                                  # only two readable pairs
    assert [p.tier for p in few] == ["anchor"] + ["rejected"] * 4


def test_contradicted_read_vetoes_count_match_and_interpolation() -> None:
    """A readable read much closer to another edition line breaks the count match and its anchor stretch."""
    steps = [ael.Step("match", j, (j,)) for j in range(5)]
    pairs = make_pairs(steps, [0.35, 0.3, 0.3, 0.4, 0.3], [10, 10, 10, 10, 10], [0.2] * 5)
    pairs[2].best_other = 0.6
    identity, *_ = ael.assign_tiers(steps, pairs, ael.Params())
    assert identity and {p.tier for p in pairs} == {"rejected"}
    assert {p.reason for p in pairs} == {ael.REJECT_CONTRADICTED}
    pairs[2].best_other = 0.42                                                    # below the contradiction floor
    ael.assign_tiers(steps, pairs, ael.Params())
    assert {p.tier for p in pairs} == {"count_match"}
    bridge = make_pairs(steps, [0.8, 0.3, 0.3, 0.8, 0.0], [20, 10, 10, 20, 0], [0.2] * 5)
    bridge[1].best_other = 0.7
    ael.assign_tiers(steps, bridge, ael.Params(sim_margin=1.0))                   # no count match: stretch logic only
    assert [p.tier for p in bridge] == ["anchor", "rejected", "rejected", "anchor", "rejected"]
    assert bridge[2].reason == ael.REJECT_CONTRADICTED


def test_assign_tiers_local_length_guard_is_asymmetric_and_spares_anchors() -> None:
    """A segment too short for its text (0.75 of the local length) is rejected, one too long by 45 % is not;
    an anchor that is short is still an anchor; ``[0, inf]`` disables the guard."""
    steps = [ael.Step("match", 0, (0,)), ael.Step("match", 1, (1,)), ael.Step("match", 2, (2,))]
    pairs = make_pairs(steps, [0.8, 0.0, 0.8], [20, 0, 20], widths=[1000, 750, 1000])
    ael.assign_tiers(steps, pairs, ael.Params())
    assert [p.tier for p in pairs] == ["anchor", "rejected", "anchor"] and pairs[1].reason == ael.REJECT_LENGTH
    assert pairs[1].len_ratio == pytest.approx(0.75)
    ael.assign_tiers(steps, pairs, ael.Params(len_short_min=0.0, len_long_max=math.inf))
    assert [p.tier for p in pairs] == ["anchor", "interpolated", "anchor"]
    long_pairs = make_pairs(steps, [0.8, 0.0, 0.8], [20, 0, 20], widths=[1000, 1450, 1000])
    ael.assign_tiers(steps, long_pairs, ael.Params())
    assert [p.tier for p in long_pairs] == ["anchor", "interpolated", "anchor"]
    short_anchor = make_pairs(steps, [0.8, 0.0, 0.8], [20, 0, 20], widths=[1000, 1000, 400])
    ael.assign_tiers(steps, short_anchor, ael.Params())
    assert short_anchor[2].tier == "anchor"
    identity_pairs = make_pairs(steps, [0.3, 0.3, 0.3], [10, 10, 10], [0.1, 0.1, 0.1], widths=[1000, 1000, 700])
    ael.assign_tiers(steps, identity_pairs, ael.Params())
    assert [p.tier for p in identity_pairs] == ["count_match", "count_match", "rejected"]


def test_local_cpps_use_bracketing_covering_anchors() -> None:
    """A line's local cpp is the median of the anchors around it and the page value; partial anchors do not
    vote; without anchors the page value is the path median."""
    steps = [ael.Step("match", j, (j,)) for j in range(5)]
    pairs = make_pairs(steps, [0.8, 0.0, 0.8, 0.0, 0.8], [20, 0, 20, 0, 20], widths=[1000, 900, 500, 900, 800])
    for p in pairs:
        p.anchored = p.sim >= 0.5
    cpps = ael.local_cpps(pairs)
    page = sorted([0.04, 0.08, 0.05, 40 / 900])[1:3]                                # median of 3 anchors + path vote
    assert cpps[1] == pytest.approx(sorted([0.04, 0.08, sum(page) / 2])[1])
    pairs[2].partial = True                                                         # the 0.08 anchor stops short
    cpps = ael.local_cpps(pairs)
    assert cpps[1] == pytest.approx(40 / 900) and cpps[3] == pytest.approx(40 / 900)  # median(0.04, 0.05, page)
    for p in pairs:
        p.anchored = False
    assert ael.local_cpps(pairs)[1] == pytest.approx(40 / 900)                     # path median of 0.04 .. 0.08


def test_partial_segment_is_rejected_in_every_tier() -> None:
    """A read that matches a sub-span far better than the whole line (and leaves a quarter of it out) is
    rejected, anchors included; it still counts as a position anchor for bridging; thresholds override."""
    steps = [ael.Step("match", j, (j,)) for j in range(4)] + [ael.Step("skip_s", None, (4,))]   # not a count match
    pairs = make_pairs(steps, [0.8, 0.0, 0.62, 0.8], [20, 0, 20, 20])
    pairs[2].cover_sim, pairs[2].cover_lost = 0.85, 0.35                           # an anchor cut at the left end
    ael.assign_tiers(steps, pairs, ael.Params())
    assert [p.tier for p in pairs] == ["anchor", "interpolated", "rejected", "anchor"]
    assert pairs[2].reason == ael.REJECT_PARTIAL and pairs[2].anchored and pairs[2].partial
    ael.assign_tiers(steps, pairs, ael.Params(cover_lost=0.4, max_lost=1.0))       # lost guard off: old behaviour
    assert pairs[2].tier == "anchor"
    ael.assign_tiers(steps, pairs, ael.Params(cover_gain=0.3, max_lost=1.0))
    assert pairs[2].tier == "anchor"
    identity = steps[:4]
    count = make_pairs(identity, [0.35, 0.3, 0.3, 0.4], [10, 10, 10, 10], [0.1] * 4)
    count[1].cover_sim, count[1].cover_lost = 0.6, 0.5
    ael.assign_tiers(identity, count, ael.Params())
    assert [p.tier for p in count] == ["count_match", "rejected", "count_match", "count_match"]
    assert count[1].reason == ael.REJECT_PARTIAL
    short_read = make_pairs(identity, [0.35, 0.3, 0.3, 0.4], [10, 5, 10, 10], [0.1] * 4)
    short_read[1].cover_sim, short_read[1].cover_lost = 0.9, 0.5                   # 5 letters: no span evidence
    ael.assign_tiers(identity, short_read, ael.Params())
    assert short_read[1].tier == "count_match"


def test_span_vetoes_contradiction_and_neighbour_preference() -> None:
    """A read whose span sim to another line beats its own by 0.1 (and reaches 0.5) contradicts; a non-anchor
    read closer by 0.05 to e-1 / e+1 prefers its neighbour; an anchor's neighbour preference does not count."""
    steps = [ael.Step("match", j, (j,)) for j in range(4)]
    pairs = make_pairs(steps, [0.35, 0.3, 0.3, 0.7], [10, 10, 10, 10])
    for p in pairs:
        p.anchored = p.sim >= 0.5
    pairs[1].best_other = 0.55                                                     # 0.55 - 0.3 >= 0.1
    pairs[2].best_other, pairs[2].neighbor_sim = 0.4, 0.36                         # below 0.5, +0.06 to a neighbour
    pairs[3].neighbor_sim = 0.8                                                    # an anchor: ignored
    assert ael.span_vetoes(pairs, ael.Params()) == {1: ael.REJECT_CONTRADICTED, 2: ael.REJECT_NEIGHBOR}
    pairs[1].letters = 5                                                           # too short for span evidence
    assert ael.span_vetoes(pairs, ael.Params()) == {2: ael.REJECT_NEIGHBOR}
    assert ael.span_vetoes(pairs, ael.Params(neighbor_margin=0.1)) == {}


def test_neighbour_preference_cancels_a_count_match() -> None:
    """One mid-page read that prefers the next edition line cancels the whole page's count match."""
    steps = [ael.Step("match", j, (j,)) for j in range(5)]
    pairs = make_pairs(steps, [0.35, 0.3, 0.3, 0.4, 0.3], [10, 10, 10, 10, 10], [0.2] * 5)
    ael.assign_tiers(steps, pairs, ael.Params())
    assert {p.tier for p in pairs} == {"count_match"}
    pairs[2].neighbor_sim = 0.36
    ael.assign_tiers(steps, pairs, ael.Params())
    assert {p.tier for p in pairs} == {"rejected"} and {p.reason for p in pairs} == {ael.REJECT_NEIGHBOR}


def test_count_match_without_anchor_needs_margin_buffer_or_column_support() -> None:
    """With no anchor, a margin just over SIM_MARGIN is not enough; a wider margin or two reads per column
    that best match their own line are."""
    steps = [ael.Step("match", j, (j,)) for j in range(6)]
    sims = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    pairs = make_pairs(steps, sims, [10] * 6, [0.225] * 6)                         # margin 0.075
    for n, p in enumerate(pairs):
        p.column = 0 if n < 3 else 1
        p.best_other_line = 0.32                                                   # every read prefers another line
    ael.assign_tiers(steps, pairs, ael.Params())
    assert {p.tier for p in pairs} == {"rejected"} and {p.reason for p in pairs} == {ael.REJECT_NO_SUPPORT}
    ael.assign_tiers(steps, pairs, ael.Params(no_anchor_margin_buffer=0.0))
    assert {p.tier for p in pairs} == {"count_match"}
    for n in (0, 1, 3):
        pairs[n].best_other_line = 0.25                                            # two own-best reads in column 0 only
    ael.assign_tiers(steps, pairs, ael.Params())
    assert {p.tier for p in pairs} == {"rejected"}
    pairs[4].best_other_line = 0.25                                                # and two in column 1
    ael.assign_tiers(steps, pairs, ael.Params())
    assert {p.tier for p in pairs} == {"count_match"}
    wide = make_pairs(steps, sims, [10] * 6, [0.2] * 6)                            # margin 0.1 >= 0.07 + 0.02
    for p in wide:
        p.best_other_line = 0.32                                                   # no column support at all
    ael.assign_tiers(steps, wide, ael.Params())
    assert {p.tier for p in wide} == {"count_match"}


def test_visible_length_counts_digits_and_hebrew() -> None:
    """The length model counts what takes room on the page, which the skeleton drops."""
    line = ael.edition_lines("ربيع اول 1259 تسع [...]\nעדים כשרים فان", ael.Params())
    assert line[0].visible == "ربيع اول 1259 تسع" and line[0].length == len("ربيع اول 1259 تسع")
    assert line[1].length == len("עדים כשרים فان") and line[1].skeleton == "فان"


# ----------------------------------------------------------------------------- page

def test_edition_lines_placeholders_and_lacuna_only_lines() -> None:
    """A line with a gap token is a placeholder (empty target); a lone gap token is left out."""
    lines = ael.edition_lines("بسم الله\nحسب [...] مقصودا\n[...]\n\nابو الفرج", ael.Params())
    assert [e.raw for e in lines] == ["بسم الله", "حسب [...] مقصودا", "ابو الفرج"]
    assert [e.target for e in lines] == ["بسم الله", "", "ابو الفرج"]
    assert [e.index for e in lines] == [0, 1, 2]
    assert lines[1].skeleton == "حسب مفصودا" and lines[0].length == len("بسم الله")


def test_align_page_clear_page_with_fragment_and_junk() -> None:
    """Good reads anchor, unreadable lines between anchors interpolate, the split line is merged."""
    reads = [EDITION[0], EDITION[1], "", EDITION[3], EDITION[4], "", EDITION[6]]
    page = synthetic_page(EDITION, reads, split_line=4, junk=True)
    segments, noise, edition, alignment, skip = ael.align_page(page, "\n".join(EDITION), ael.Params())
    assert skip == "" and noise == [] and len(edition) == 7
    tiers = {p.e: p.tier for p in alignment.pairs}
    assert tiers[0] == tiers[1] == tiers[3] == tiers[6] == "anchor"
    assert tiers[5] == "interpolated"
    merged = [p for p in alignment.pairs if len(p.s) == 2]
    assert len(merged) == 1 and merged[0].e == 4 and tiers[4] == "anchor"
    assert tiers[2] == "rejected"                                                # the junk line sits between anchors 1 and 3
    assert alignment.ordering == "one_column" and alignment.cpp == pytest.approx(0.04, rel=0.05)


def test_align_page_count_match_without_anchors() -> None:
    """Equal counts and moderately readable lines: every line is a count match."""
    def blur(text: str) -> str:
        """Poorly legible read across the whole line: every third character misread as kaf.

        :param text: Edition line.
        :return: Read.
        """
        return "".join("ك" if n % 3 == 1 and ch != " " else ch for n, ch in enumerate(text))
    reads = [blur(e) for e in EDITION]
    page = synthetic_page(EDITION, reads)
    _, _, _, alignment, skip = ael.align_page(page, "\n".join(EDITION), ael.Params(sim_anchor=0.95))
    assert skip == "" and alignment.identity
    assert {p.tier for p in alignment.pairs} == {"count_match"}


def first_half(text: str) -> str:
    """The first half of a line's words (at least two), as a segment cut at the left end reads it.

    :param text: Edition line.
    :return: Its right-hand (first) half.
    """
    words = text.split()
    return " ".join(words[:max(2, len(words) // 2)])


def test_shifted_tail_under_a_passing_guard_loses_its_count_match() -> None:
    """The segment of edition line 4 is lost and an empty junk line sits at the bottom, so the counts still
    agree and the path is the identity: anchors on lines 0-3 carry the page-wide guard, and the shifted reads
    (half of the NEXT line each) stay under the whole-line contradiction floor. Span sims expose the shift:
    the page loses its count match and the shifted lines are not written."""
    reads = [EDITION[0], EDITION[1], EDITION[2], EDITION[3], first_half(EDITION[5]), first_half(EDITION[6]), ""]
    page = synthetic_page(EDITION, reads)
    _, _, edition, alignment, skip = ael.align_page(page, "\n".join(EDITION), ael.Params())
    assert skip == "" and alignment.identity
    assert alignment.page_sim >= ael.SIM_PAGE and alignment.sim_margin >= ael.SIM_MARGIN
    shifted = [p for p in alignment.pairs if p.e in (4, 5)]
    assert all(p.best_other_line < 0.45 for p in shifted)                          # whole-line sims miss it
    assert all(p.best_other >= ael.CONTRADICT_MIN_SIM for p in shifted)            # span sims do not
    tiers = {p.e: p.tier for p in alignment.pairs}
    assert [tiers[e] for e in range(4)] == ["anchor"] * 4
    assert tiers[4] == tiers[5] == tiers[6] == "rejected" and "count_match" not in tiers.values()
    assert {p.reason for p in alignment.pairs if p.e >= 4} == {ael.REJECT_CONTRADICTED}


def test_align_page_rejects_a_segment_cut_at_the_left_end() -> None:
    """A segment holding only the first two thirds of its line (its read matches them exactly) is not
    written, even though its whole-line sim makes it an anchor; its neighbours are unaffected."""
    reads = list(EDITION)
    reads[3] = " ".join(EDITION[3].split()[:6])                                    # 6 of 9 words
    page = synthetic_page(EDITION, reads)
    cut = page["lines"][3]
    x1 = cut["baseline"][-1][0]
    x0 = x1 - (x1 - cut["baseline"][0][0]) * 2 / 3
    page["lines"][3] = seg(3, x0, x1, cut["baseline"][0][1], reads[3])
    _, _, _, alignment, skip = ael.align_page(page, "\n".join(EDITION), ael.Params())
    pair = next(p for p in alignment.pairs if p.e == 3)
    assert skip == "" and pair.anchored and pair.partial
    assert pair.tier == "rejected" and pair.reason == ael.REJECT_PARTIAL
    assert {p.tier for p in alignment.pairs if p.e != 3} == {"anchor"}


def test_align_page_skips() -> None:
    """Too few targets, and a page whose reads never match the edition, contribute nothing."""
    page = synthetic_page(EDITION, EDITION)
    *_, skip = ael.align_page(page, "بسم الله\n[...] حسب", ael.Params())
    assert skip == ael.SKIP_FEW_TARGETS
    other = ["شهد عندي الشهود المسمون في هذا الكتاب", "قبض فلان من فلان ثمن الدار كاملا وافيا",
             "ثلثة اسهم من جملة ستة اسهم شائعة", "حدود الدار الاربعة الحد الاول ينتهي",
             "وكتب في شهر رمضان سنة اربعمائة", "والحمد لله وحده وصلواته على نبيه", "شهد بذلك وكتب بخطه"]
    page = synthetic_page(EDITION, other)
    *_, alignment, skip = ael.align_page(page, "\n".join(EDITION), ael.Params())
    assert skip == ael.SKIP_NO_ANCHORS and not alignment.tier_counts()["anchor"]


def test_align_page_picks_two_column_order_for_a_bifolium() -> None:
    """A right page and a left page on independent heights align right page first."""
    edition = EDITION[:6]
    lines = []
    for n, text in enumerate(edition[:3]):
        lines.append(seg(len(lines), 1900 - width_for(text) * 0.5, 1900, 200 + 150 * n, text))
    for n, text in enumerate(edition[3:]):
        lines.append(seg(len(lines), 900 - width_for(text) * 0.5, 900, 260 + 150 * n, text))
    page = {"stem": "pgpar_2_0_page", "pgpid": "2", "canonical_id": "", "image_path": "/x.jpg",
            "image_size": [2000, 1000], "lines": lines}
    *_, alignment, skip = ael.align_page(page, "\n".join(edition), ael.Params())
    assert skip == "" and alignment.ordering == "two_columns_right_first"
    assert [p.tier for p in alignment.pairs] == ["anchor"] * 6
    assert [p.column for p in alignment.pairs] == [0, 0, 0, 1, 1, 1]


def test_align_page_reads_witness_blocks_in_edition_order() -> None:
    """Under a full-width text, a right witness block then a left one: the block ordering wins."""
    edition = EDITION[:3] + ["المملوك يعلم صحة ذلك", "وكتب هبة الله بن", "فضائل في تاريخه",
                             "شهد على اقراره بذلك", "وكتب يوسف بن سالم"]
    lines = [dict(seg(l.index, l.x_min, l.x_max, l.y_mean, text)) for l, text in zip(witness_page(), edition)]
    page = {"stem": "pgpar_3_0_page", "pgpid": "3", "canonical_id": "", "image_path": "/x.jpg",
            "image_size": [2000, 1200], "lines": lines}
    *_, alignment, skip = ael.align_page(page, "\n".join(edition), ael.Params())
    assert skip == "" and alignment.ordering == "blocks"
    assert [(p.e, alignment.order[p.s[0]].index) for p in alignment.pairs] == [(j, j) for j in range(8)]
    assert {p.tier for p in alignment.pairs} == {"anchor"}


# ----------------------------------------------------------------------------- PAGE XML

def test_custom_attr_parses_like_kraken() -> None:
    """The audit attribute uses PAGE ``tag {key:value;}`` syntax that kraken's custom parser splits."""
    custom = ael.custom_attr("anchor", 0.623, 0.81, 3)
    assert custom == "alignment {tier:anchor; sim:0.62; len:0.81; edition:3;}"
    tag, vals = custom.rstrip("}").split("{")                                      # what kraken does per chunk
    parsed = dict(v.strip().split(":") for v in vals.split(";") if v.strip())
    assert tag.strip() == "alignment" and parsed == {"tier": "anchor", "sim": "0.62", "len": "0.81", "edition": "3"}


def test_page_xml_bytes_round_trip() -> None:
    """Schema namespace, absolute image path, clamped points, one line per record, text and custom kept."""
    lines = [ael.OutLine("l0", ((10.4, 50.0), (400.0, 52.0)), ((5, 20), (405, 20), (405, 60), (5, 60)), "بسم الله",
                         ael.custom_attr("anchor", 0.7, 0.9, 0)),
             ael.OutLine("l1_2", ((-5, 120), (2500, 120)), ((-5, 90), (2500, 90), (2500, 130)), "",
                         ael.custom_attr("rejected", 0.1, 0.5, 1))]
    data = ael.page_xml_bytes("/abs/page.jpg", 2000, 1500, lines, "2026-10-19T00:00:00+00:00")
    root = etree.fromstring(data)
    page = root.find("p:Page", NS)
    assert etree.QName(root).namespace == ael.PAGE_NS
    assert page.get("imageFilename") == "/abs/page.jpg" and page.get("imageWidth") == "2000"
    tl = page.findall("p:TextRegion/p:TextLine", NS)
    assert [t.get("id") for t in tl] == ["l0", "l1_2"]
    assert tl[0].find("p:Baseline", NS).get("points") == "10,50 400,52"
    assert tl[1].find("p:Baseline", NS).get("points") == "0,120 1999,120"
    assert tl[0].find("p:TextEquiv/p:Unicode", NS).text == "بسم الله"
    assert not tl[1].find("p:TextEquiv/p:Unicode", NS).text
    assert [etree.QName(c).localname for c in tl[0]] == ["Coords", "Baseline", "TextEquiv"]
    assert tl[1].get("custom").startswith("alignment {tier:rejected;")


def test_out_lines_and_written_text_follow_tiers(tmp_path: Path) -> None:
    """Accepted lines carry the cleaned edition text; rejected, unmatched and noise lines are empty.

    :param tmp_path: pytest temporary directory.
    """
    reads = [EDITION[0], EDITION[1], "", EDITION[3], EDITION[4], "", EDITION[6]]
    page = synthetic_page(EDITION, reads, split_line=4, junk=True)
    page["lines"].append(seg(len(page["lines"]), 50, 55, 1300, ""))                 # a speck
    result = ael.PageResult("pgpar_1_0_page", "1", "", "/abs/p.jpg", (2000, 1400))
    result.segments, result.noise, result.edition, result.alignment, result.skip = ael.align_page(
        page, "\n".join(EDITION), ael.Params())
    path = ael.write_page(result, tmp_path, "2026-10-19T00:00:00+00:00")
    root = etree.parse(str(path)).getroot()
    tl = root.findall(".//p:TextLine", NS)
    assert len(tl) == len(page["lines"]) - 1                                         # the split line is one TextLine
    by_id = {t.get("id"): t for t in tl}
    texts = {t.get("id"): (t.find("p:TextEquiv/p:Unicode", NS).text or "") for t in tl}
    assert texts["l0"] == EDITION[0]
    merged_id = [i for i in by_id if "_" in i]
    assert merged_id == ["l5_6"] and texts["l5_6"] == EDITION[4]
    assert "tier:noise" in by_id[f"l{len(page['lines']) - 1}"].get("custom")
    assert texts[f"l{len(page['lines']) - 1}"] == ""
    rejected = [i for i, t in by_id.items() if "tier:rejected" in t.get("custom") or "tier:unmatched" in t.get("custom")]
    assert rejected and all(texts[i] == "" for i in rejected)
    assert len(result.accepted()) == sum(1 for t in texts.values() if t)


def test_untargeted_lines_are_written_empty() -> None:
    """An accepted placement of a placeholder edition line keeps empty text and says so."""
    edition = EDITION[:3] + ["حسب الادلال عليه [...] مقصودا"] + EDITION[4:]
    page = synthetic_page(edition, edition)
    result = ael.PageResult("pgpar_1_0_page", "1", "", "/abs/p.jpg", (2000, 1400))
    result.segments, result.noise, result.edition, result.alignment, result.skip = ael.align_page(
        page, "\n".join(edition), ael.Params())
    lines = {l.line_id: l for l in ael.out_lines(result)}
    assert lines["l3"].text == "" and "untargeted" in lines["l3"].custom
    assert lines["l2"].text == EDITION[2]


# ----------------------------------------------------------------------------- run

def test_split_key_prefers_canonical_id() -> None:
    """Documents split by canonical id, falling back to the PGP id; the split is deterministic."""
    assert ael.split_key("Cambridge_T_S_1", "77") == "genizah/Cambridge_T_S_1"
    assert ael.split_key("", "77") == "genizah/pgpid:77"
    assert ael.hash_split(ael.split_key("", "77"), 0.08) == ael.hash_split(ael.split_key("", "77"), 0.08)


def test_split_by_page_keys_each_page_and_parts_one_document(tmp_path: Path) -> None:
    """``--split-by-page`` keys the page stem, so two pages of one document can land in different splits;
    without it both follow their document.

    :param tmp_path: pytest temporary directory.
    """
    assert ael.split_key("Cambridge_T_S_1", "77", "bk_T_S_1_0") == "genizah/page:bk_T_S_1_0"
    assert ael.split_key("Cambridge_T_S_1", "77", "") == "genizah/Cambridge_T_S_1"
    stems = [f"bk_doc_{n}_0" for n in range(50)]
    val = next(s for s in stems if ael.hash_split(ael.split_key("", "", s), 0.5) == "val")
    train = next(s for s in stems if ael.hash_split(ael.split_key("", "", s), 0.5) == "train")
    manifest = {}
    for stem in (val, train):
        page = synthetic_page(EDITION[:2], EDITION[:2], stem=stem)
        (tmp_path / f"{stem}.json").write_text(json.dumps(page), encoding="utf-8")
        manifest[stem] = {"stem": stem, "pgpid": "", "canonical_id": "Doc_shared"}
    doc_split = ael.hash_split(ael.split_key("Doc_shared", ""), 0.5)
    for by_page, expected in ((False, {val: doc_split, train: doc_split}), (True, {val: "val", train: "train"})):
        params = ael.Params(val_share=0.5, split_by_page=by_page)
        got = {s: ael.process_page(tmp_path / f"{s}.json", {}, manifest, (set(), set()), params).split
               for s in (val, train)}
        assert got == expected
    assert ael.build_parser().parse_args([]).split_by_page is False
    assert ael.build_parser().parse_args(["--split-by-page"]).split_by_page is True


def test_audit_sample_caps_lines_per_page_and_tier() -> None:
    """At most two lines per (page, tier) while other pages remain; the overflow is used when a tier runs
    short."""
    results = []
    for n, count in enumerate((12, 1, 1)):
        result = ael.PageResult(f"pgpar_{n}_0_page", str(n), "", "/x.jpg", (100, 100))
        result.edition = [ael.EditionLine(j, "بسم الله", "بسم الله", "بسم الله") for j in range(count)]
        steps = [ael.Step("match", j, (j,)) for j in range(count)]
        result.alignment = ael.Alignment("one_column", [], steps, [ael.Pair(j, j, (j,), 0.9, 1.0, 10, tier="anchor")
                                                                  for j in range(count)], 0.0, 0.04, True, 0.9)
        results.append(result)
    sample = ael.audit_sample(results, 4, 0, seed=1)
    assert [r.stem for r, _ in sample].count("pgpar_0_0_page") == 2 and len(sample) == 4
    assert {r.stem for r, _ in sample} == {"pgpar_0_0_page", "pgpar_1_0_page", "pgpar_2_0_page"}
    more = ael.audit_sample(results, 8, 0, seed=1)
    assert len(more) == 8 and [r.stem for r, _ in more].count("pgpar_0_0_page") == 6
    assert [r.stem for r, _ in ael.audit_sample(results, 8, 0, seed=1, page_cap=0)].count("pgpar_0_0_page") >= 6


def test_crop_box_pads_and_clamps() -> None:
    """Bounding box plus padding, inside the image."""
    assert ael.crop_box([(5, 10), (100, 40)], 90, 200, pad=10) == (0, 0, 90, 50)


def build_inputs(tmp_path: Path, pages: Dict[str, Tuple[Dict[str, Any], str]],
                 painted: Optional[Dict[str, Image.Image]] = None) -> Tuple[Path, Path]:
    """Write segmenter JSONs, page images and a tiny edition dataset (no image column) under ``tmp_path``.

    :param tmp_path: Test directory.
    :param pages: ``{stem: (segmenter JSON without image path, edition text)}``.
    :param painted: Page images by stem (saved as PNG); other pages get a blank parchment-coloured JPEG.
    :return: ``(pages dir, editions dir)``.
    """
    pages_dir = tmp_path / "pages"
    images = tmp_path / "images"
    pages_dir.mkdir()
    images.mkdir()
    rows = {"stem": [], "answer": []}
    manifest = []
    painted = painted or {}
    for stem, (page, answer) in pages.items():
        if stem in painted:
            image = images / f"{stem}.png"
            painted[stem].save(image)
        else:
            image = images / f"{stem}.jpg"
            Image.new("RGB", tuple(page["image_size"]), (220, 205, 180)).save(image)
        page = dict(page, stem=stem, image_path=str(image))
        (pages_dir / f"{stem}.json").write_text(json.dumps(page, ensure_ascii=False), encoding="utf-8")
        rows["stem"].append(stem)
        rows["answer"].append(answer)
        manifest.append({"stem": stem, "pgpid": page["pgpid"], "canonical_id": f"Doc_{page['pgpid']}"})
    (pages_dir / "summary.jsonl").write_text("{}\n", encoding="utf-8")
    editions = tmp_path / "editions"
    datasets.DatasetDict({"train_page": datasets.Dataset.from_dict(rows)}).save_to_disk(str(editions))
    (editions / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest), encoding="utf-8")
    return pages_dir, editions


def test_run_end_to_end(tmp_path: Path) -> None:
    """Pages are aligned, written, listed, counted and audited; registry pages are held out.

    :param tmp_path: pytest temporary directory.
    """
    good = synthetic_page(EDITION, [EDITION[0], EDITION[1], "", EDITION[3], EDITION[4], "", EDITION[6]],
                          junk=True, stem="pgpar_1_0_page")                       # line 2 rejected (junk beside it)
    held = synthetic_page(EDITION, EDITION, stem="pgpar_2_0_page")
    held["pgpid"] = "2"
    few = synthetic_page(EDITION[:1], EDITION[:1], stem="pgpar_3_0_page")
    pages_dir, editions = build_inputs(tmp_path, {"pgpar_1_0_page": (good, "\n".join(EDITION)),
                                                  "pgpar_2_0_page": (held, "\n".join(EDITION)),
                                                  "pgpar_3_0_page": (few, EDITION[0])})
    out = tmp_path / "out"
    stale = out / "page_xml" / "genizah" / "pgpar_9_0_page.xml"
    stale.parent.mkdir(parents=True)
    stale.write_text("<old/>", encoding="utf-8")
    stats = ael.run(pages_dir, editions, out, ael.Params(audit_n=4, audit_rejected=2),
                    held_out=({"Doc_2"}, set()))
    assert stats["pages_seen"] == 3 and stats["pages_aligned"] == 1
    assert stats["pages_skipped"] == {ael.SKIP_REGISTRY: 1, ael.SKIP_FEW_TARGETS: 1}
    assert stats["placed_lines"] == 6 and stats["edition_target_lines"] == 8
    assert stats["yield_pct_of_targets"] == 75.0
    xml = out / "page_xml" / "genizah" / "pgpar_1_0_page.xml"
    assert xml.exists() and not stale.exists()
    listed = [l for s in ("train", "val") for l in (out / "lists" / f"genizah_{s}.lst").read_text().split()]
    assert listed == [str(xml)]
    assert json.loads((out / "alignment_stats.json").read_text())["placed_lines"] == 6
    html_text = (out / "gate" / "genizah_alignment" / "index.html").read_text(encoding="utf-8")
    crops = sorted((out / "gate" / "genizah_alignment" / "crops").glob("*.png"))
    assert len(crops) == 5 and all(c.name in html_text for c in crops)               # 4 accepted + 1 rejected exists
    rows = [json.loads(l) for l in (out / "gate" / "genizah_alignment" / "pages.jsonl").read_text().splitlines()]
    assert {r["stem"]: r["skip"] for r in rows}["pgpar_1_0_page"] == ""


def test_cli_overrides_every_threshold() -> None:
    """Every :class:`ael.Params` field is a CLI option."""
    args = ael.build_parser().parse_args(["--sim-anchor", "0.6", "--sim-page", "0.2", "--val-share", "0.1",
                                          "--audit-n", "12", "--max-merge", "2", "--limit", "5"])
    assert (args.sim_anchor, args.sim_page, args.val_share, args.audit_n, args.max_merge, args.limit) == (
        0.6, 0.2, 0.1, 12, 2, 5)
    assert all(hasattr(args, f.name) for f in __import__("dataclasses").fields(ael.Params))


# ----------------------------------------------------------------------------- lost guard and count-match drop

def test_lost_guard_applies_to_every_tier_and_merged_segments() -> None:
    """A read that leaves part of its edition line out is rejected whatever its tier: an anchor, a merged
    anchor, an interpolated and a count-match line (the partial test alone let these through when the span
    barely beat the whole line). ``max_lost`` 0.25 keeps a 0.2 loss; 0.0 is strict; reads too short for span
    evidence are exempt; a lost-guarded anchor still bridges its stretch."""
    steps = [ael.Step("match", 0, (0,)), ael.Step("match", 1, (1, 2)), ael.Step("match", 2, (3,)),
             ael.Step("match", 3, (4,)), ael.Step("skip_s", None, (5,))]
    pairs = make_pairs(steps, [0.8, 0.8, 0.3, 0.8], [20, 20, 10, 20])
    pairs[1].cover_lost = 0.3                                                       # merged anchor, gain 0: not partial
    pairs[2].cover_lost = 0.2                                                       # interpolated
    ael.assign_tiers(steps, pairs, ael.Params())
    assert [p.tier for p in pairs] == ["anchor", "rejected", "interpolated", "anchor"]
    assert pairs[1].reason == ael.REJECT_LOST and pairs[1].anchored and not pairs[1].partial
    ael.assign_tiers(steps, pairs, ael.Params(max_lost=0.0))
    assert [p.tier for p in pairs] == ["anchor", "rejected", "rejected", "anchor"]
    assert pairs[2].reason == ael.REJECT_LOST
    ael.assign_tiers(steps, pairs, ael.Params(max_lost=1.0))                       # guard off: old behaviour
    assert [p.tier for p in pairs] == ["anchor", "anchor", "interpolated", "anchor"]
    pairs[0].cover_lost = 0.1                                                       # first anchor lost-guarded ...
    ael.assign_tiers(steps, pairs, ael.Params(max_lost=0.05))
    assert pairs[0].reason == ael.REJECT_LOST and pairs[2].tier == "rejected"      # ... (0.2 > 0.05 too)
    pairs[2].cover_lost = 0.0
    ael.assign_tiers(steps, pairs, ael.Params(max_lost=0.05))
    assert pairs[2].tier == "interpolated"                                          # ... and still bridges
    identity = [ael.Step("match", j, (j,)) for j in range(4)]
    count = make_pairs(identity, [0.35, 0.3, 0.3, 0.4], [10, 10, 10, 10], [0.1] * 4)
    count[2].cover_lost = 0.3
    ael.assign_tiers(identity, count, ael.Params())
    assert [p.tier for p in count] == ["count_match", "count_match", "rejected", "count_match"]
    assert count[2].reason == ael.REJECT_LOST
    count[2].letters = 5                                                            # no span evidence: exempt
    ael.assign_tiers(identity, count, ael.Params(max_lost=0.0))
    assert count[2].tier == "count_match"


def test_drop_count_match() -> None:
    """``drop_count_match``: a count-match line between two anchors falls back to interpolated, any other
    count-match line is rejected."""
    steps = [ael.Step("match", j, (j,)) for j in range(5)]
    pairs = make_pairs(steps, [0.8, 0.3, 0.3, 0.8, 0.3], [20, 10, 10, 20, 10], [0.1] * 5)
    ael.assign_tiers(steps, pairs, ael.Params())
    assert [p.tier for p in pairs] == ["anchor", "count_match", "count_match", "anchor", "count_match"]
    ael.assign_tiers(steps, pairs, ael.Params(drop_count_match=True))
    assert [p.tier for p in pairs] == ["anchor", "interpolated", "interpolated", "anchor", "rejected"]
    assert pairs[4].reason == ael.REJECT_COUNT_MATCH_DROPPED


# ----------------------------------------------------------------------------- ink walk and geometry gate

def line_ink(mask: np.ndarray, x0: int, x1: int, y: float) -> None:
    """Paint word blocks along a baseline (40 px words, 10 px spaces, 28 px above to 4 px below), in place.

    :param mask: Boolean page.
    :param x0: Left end.
    :param x1: Right end.
    :param y: Baseline.
    """
    x = x0
    while x < x1:
        mask[int(y) - 28:int(y) + 4, x:min(x + 40, x1)] = True
        x += 50


def inked_page(clip_left: Optional[Dict[int, int]] = None) -> Tuple[Dict[str, Any], np.ndarray]:
    """The seven-line synthetic page (every read exact, so every line is an anchor) and its ink: each line's
    words fill its segment, plus ``clip_left[n]`` pixels of words left of line ``n``'s polygon.

    :param clip_left: ``{edition line: pixels of ink beyond the left end}``.
    :return: ``(segmenter JSON, ink mask)``.
    """
    page = synthetic_page(EDITION, EDITION)
    mask = np.zeros((1400, 2000), dtype=bool)
    for n, raw in enumerate(page["lines"]):
        x0, x1, y = raw["baseline"][0][0], raw["baseline"][-1][0], raw["baseline"][0][1]
        line_ink(mask, int(math.ceil(x0)) - (clip_left or {}).get(n, 0), int(x1), y)
    return page, mask


def aligned(page: Dict[str, Any]) -> ael.PageResult:
    """Align a synthetic page into a :class:`ael.PageResult`.

    :param page: Segmenter JSON.
    :return: The result (no image).
    """
    result = ael.PageResult("pgpar_1_0_page", "1", "", "/abs/p.png", (2000, 1400))
    result.segments, result.noise, result.edition, result.alignment, result.skip = ael.align_page(
        page, "\n".join(EDITION), ael.Params())
    return result


def pair_of(result: ael.PageResult, e: int) -> ael.Pair:
    """The pair of edition line ``e``.

    :param result: Page result.
    :param e: Edition position.
    :return: Pair.
    """
    return next(p for p in result.alignment.pairs if p.e == e)


def covers(polygon: Sequence[Tuple[float, float]], x: float, y: float) -> bool:
    """Whether a point lies inside (or on) a polygon.

    :param polygon: Points.
    :param x: Column.
    :param y: Row.
    :return: True inside or on the edge.
    """
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.float32).reshape(-1, 1, 2), (float(x), float(y)), False) >= 0


def test_clipped_line_is_extended_and_written_with_its_ink() -> None:
    """Line 3's words run 70 px past its polygon: the walk extends it, the gate keeps it, and the written
    PAGE line's polygon and baseline cover the missing word; untouched lines keep their geometry. (Line 3 is
    the page's longest segment, so its own clipped end is the text block's left margin: the block bound,
    tested in ``test_block_bound_stops_a_walk_at_the_page_margin``, is off here.)"""
    page, mask = inked_page({3: 70})
    result = aligned(page)
    ael.extend_pairs(result, le.label_ink(mask), ael.Params(block_slack=math.inf))
    ael.gate_extents(result, ael.Params(block_slack=math.inf))
    assert result.line_height == pytest.approx(60)
    p3 = pair_of(result, 3)
    ext = result.geometry[p3.step]
    x0, y = page["lines"][3]["baseline"][0]
    assert p3.tier == "anchor" and 68 <= ext.left.extended_px <= 72 and ext.right.extended_px == 0
    assert covers(ext.boundary, x0 - 66, y - 10) and min(p[0] for p in ext.baseline) <= x0 - 66
    assert not result.geometry[pair_of(result, 2).step].rebuilt
    lines = {l.line_id: l for l in ael.out_lines(result)}
    assert lines["l3"].text == EDITION[3] and covers(lines["l3"].boundary, x0 - 66, y - 10)
    assert "extended:" in lines["l3"].custom and lines["l2"].boundary == tuple(map(tuple, page["lines"][2]["boundary"]))


def test_gate_rejects_ink_beyond_the_cap_and_can_be_disabled() -> None:
    """Ink running 400 px past line 5's polygon (beyond the 2-line-height cap) rejects it; with the gate off
    (``max_ink_beyond`` inf) the line is kept, extended to the cap; with extension off the gate still runs on
    the segmenter's ends and the polygon is left alone."""
    page, mask = inked_page({5: 400})
    ink = le.label_ink(mask)
    result = aligned(page)
    ael.extend_pairs(result, ink, ael.Params())
    ael.gate_extents(result, ael.Params())
    p5 = pair_of(result, 5)
    assert p5.tier == "rejected" and p5.reason == ael.REJECT_INK_BEYOND
    assert ael.ink_beyond(result.geometry[p5.step], 60) > 0.4 and result.geometry[p5.step].left.capped
    assert {p.tier for p in result.alignment.pairs if p.e != 5} == {"anchor"}
    result = aligned(page)
    ael.extend_pairs(result, ink, ael.Params())
    ael.gate_extents(result, ael.Params(max_ink_beyond=math.inf))
    assert pair_of(result, 5).tier == "anchor" and result.geometry[pair_of(result, 5).step].left.extended_px == 120
    result = aligned(page)
    off = ael.Params(extend_to_ink=False)
    ael.extend_pairs(result, ink, off)
    ael.gate_extents(result, off)
    p5 = pair_of(result, 5)
    assert p5.reason == ael.REJECT_INK_BEYOND and not result.geometry[p5.step].rebuilt
    assert result.geometry[p5.step].left.extended_px == 0
    assert not ael.needs_geometry(ael.Params(extend_to_ink=False, max_ink_beyond=math.inf))


def test_gate_rejects_a_line_whose_walk_reaches_another_edition_lines_segment() -> None:
    """An unplaced segment just left of line 2 whose read is edition line 5: absorbing it would steal that
    line's text, so line 2 is rejected; the same box with an empty read is absorbed instead and is not
    written as a TextLine of its own."""
    page, mask = inked_page({2: 70})
    x0, y = page["lines"][2]["baseline"][0]
    ink = le.label_ink(mask)
    for read, tier in ((EDITION[5], "rejected"), ("", "anchor")):
        result = aligned(page)
        box = ael.make_seg_line(seg(98, x0 - 64, x0 - 14, y, read))
        result.segments.append(box)
        if not read:
            result.noise.append(98)
        ael.extend_pairs(result, ink, ael.Params())
        ael.gate_extents(result, ael.Params())
        p2 = pair_of(result, 2)
        assert p2.tier == tier
        if read:
            assert p2.reason == ael.REJECT_STEAL and result.geometry[p2.step].stolen == (98,)
        else:
            assert result.geometry[p2.step].absorbed == (98,) and result.absorbed_segments() == {98}
            lines = {l.line_id: l for l in ael.out_lines(result)}
            assert "l98" not in lines and "absorbed:98" in lines["l2"].custom


def test_shared_absorption_rejects_both_lines() -> None:
    """A box absorbed by two accepted lines belongs to neither."""
    page, mask = inked_page()
    result = aligned(page)
    ink = le.label_ink(mask)
    ael.extend_pairs(result, ink, ael.Params())
    a, b = pair_of(result, 1), pair_of(result, 2)
    shared = le.EndExtent(extended_px=10, absorbed=(42,))
    for p in (a, b):
        g = result.geometry[p.step]
        result.geometry[p.step] = le.LineExtent(g.baseline, g.boundary, shared, g.right, True)
    ael.gate_extents(result, ael.Params())
    assert a.reason == b.reason == ael.REJECT_STEAL and pair_of(result, 3).tier == "anchor"


def test_run_writes_pairs_jsonl_extension_stats_and_wide_audit_crops(tmp_path: Path) -> None:
    """End to end on a painted page: line 3 is extended and written, line 5 is rejected by the gate, every
    pair has a pairs.jsonl row with tier, lost, extension and ink beyond, the statistics count both, and the
    audit crop of an extended line shows two panels. (The block bound is off: line 3 defines the block.)

    :param tmp_path: pytest temporary directory.
    """
    page, mask = inked_page({3: 70, 5: 400})
    image = Image.new("RGB", (2000, 1400), (220, 205, 180))
    pixels = np.asarray(image).copy()
    pixels[mask] = (45, 30, 20)
    pages_dir, editions = build_inputs(tmp_path, {"pgpar_1_0_page": (page, "\n".join(EDITION))},
                                       {"pgpar_1_0_page": Image.fromarray(pixels)})
    out = tmp_path / "out"
    stats = ael.run(pages_dir, editions, out, ael.Params(audit_n=7, audit_rejected=1, block_slack=math.inf),
                    held_out=(set(), set()))
    assert stats["placed_lines"] == 6
    assert stats["rejected_target_pairs_by_reason"] == {ael.REJECT_INK_BEYOND: 1}
    ext = stats["extension"]
    assert ext["extended_lines"] == 1 and ext["ink_beyond_rejections"] == 1 and ext["extended_px_left"] >= 68
    assert ext["extend_to_ink"] and ext["max_lost"] == ael.MAX_LOST
    rows = [json.loads(l) for l in (out / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 7 and {r["edition_index"] for r in rows} == set(range(7))
    by_e = {r["edition_index"]: r for r in rows}
    assert by_e[3]["written"] and 68 <= by_e[3]["extended_px_left"] <= 72 and by_e[3]["line_id"] == "l3"
    assert by_e[3]["lost"] == 0.0 and by_e[3]["text"] == EDITION[3] and by_e[3]["line_height"] == 60.0
    assert not by_e[5]["written"] and by_e[5]["reason"] == ael.REJECT_INK_BEYOND and by_e[5]["ink_beyond_left"] > 0.4
    assert by_e[0]["extended_px_left"] == 0 and by_e[0]["ink_beyond_right"] == 0.0
    root = etree.parse(str(out / "page_xml" / "genizah" / "pgpar_1_0_page.xml")).getroot()
    tl = {t.get("id"): t for t in root.findall(".//p:TextLine", NS)}
    pts = [tuple(map(float, p.split(","))) for p in tl["l3"].find("p:Coords", NS).get("points").split()]
    x0, y = page["lines"][3]["baseline"][0]
    assert covers(pts, x0 - 66, y - 10) and not (tl["l5"].find("p:TextEquiv/p:Unicode", NS).text or "")
    crop = next((out / "gate" / "genizah_alignment" / "crops").glob("*_l3.png"))
    with Image.open(crop) as im:
        assert im.height > 2 * 60 and im.width >= 2 * 1.5 * 60


def test_cli_flags_for_the_ink_walk() -> None:
    """New options parse, booleans come as --flag / --no-flag pairs, and the defaults are the documented ones."""
    args = ael.build_parser().parse_args([])
    assert (args.max_lost, args.drop_count_match, args.extend_to_ink, args.max_ink_beyond) == (0.25, False, True, 0.4)
    args = ael.build_parser().parse_args(["--max-lost", "0.0", "--drop-count-match", "--no-extend-to-ink",
                                          "--max-ink-beyond", "inf", "--extend-gap", "0.5", "--audit-margin", "2"])
    assert (args.max_lost, args.drop_count_match, args.extend_to_ink) == (0.0, True, False)
    assert math.isinf(args.max_ink_beyond) and args.extend_gap == 0.5 and args.audit_margin == 2.0
    assert ael.extent_config(ael.Params(**{f.name: getattr(args, f.name) for f in
                                           __import__("dataclasses").fields(ael.Params)})).max_extend == 0.0


def test_custom_attr_carries_extension_and_absorbed() -> None:
    """Extension pixels and absorbed segments ride along in the custom attribute, still parseable."""
    custom = ael.custom_attr("anchor", 0.7, 0.9, 2, extended=(70, 0), absorbed=(9, 12))
    assert custom.endswith("edition:2; extended:70/0; absorbed:9/12;}")
    vals = custom.rstrip("}").split("{")[1]
    parsed = dict(v.strip().split(":") for v in vals.split(";") if v.strip())
    assert parsed["extended"] == "70/0" and parsed["absorbed"] == "9/12"


def test_audit_box_reaches_a_margin_of_line_heights() -> None:
    """Audit crops reach 1.5 line heights beyond the polygons on each side (clamped into the image)."""
    assert ael.audit_box([[(500, 300), (900, 360)]], 2000, 1400, 60) == (410, 285, 990, 375)
    assert ael.audit_box([[(20, 300), (900, 360)], [(10, 290), (30, 310)]], 950, 1400, 60) == (0, 275, 950, 375)


# ----------------------------------------------------------------------------- round 2: block bound, gates, drop flags

def test_page_block_uses_the_segmenters_own_line_ends() -> None:
    """The page's text block comes from every non-noise, non-rotated segmenter line (polygon x-ranges); a
    line inside it is bounded by those margins."""
    page, _ = inked_page()
    result = aligned(page)
    block = ael.page_block(result, ael.Params())
    assert block.left == pytest.approx(875) and block.right == pytest.approx(1900)
    result.segments.append(ael.make_seg_line({"index": 40, "baseline": [[50, 100], [52, 1300]],
                                              "boundary": [[30, 100], [70, 100], [70, 1300], [30, 1300]],
                                              "text": "", "mean_confidence": None}))
    assert ael.page_block(result, ael.Params()).left == pytest.approx(875)        # a rotated margin line
    group = [result.alignment.order[i] for i in pair_of(result, 2).s]
    assert ael.line_block(block, group, ael.Params()) == (block.left, block.right)
    assert ael.line_block(block, group, ael.Params(block_slack=math.inf)) is None


def test_block_bound_stops_a_walk_at_the_page_margin() -> None:
    """Line 3 (the page's longest segment, its end is the block's left margin) has ink 70 px past its end:
    the walk crosses the bound (0.3 line heights past the margin) only along its own word that straddles the
    polygon end (round 3: ink connected to the line's own letters is never cut), never reaching the separate
    word beyond, which is reported as ink past the block; line 2 (inside the block) is extended fully.
    pairs.jsonl rows carry the block, the limit, own ink outside and overlap."""
    page, mask = inked_page({2: 70, 3: 70})
    result = aligned(page)
    ael.extend_pairs(result, le.label_ink(mask), ael.Params())
    ael.gate_extents(result, ael.Params())
    e3, e2 = result.geometry[pair_of(result, 3).step], result.geometry[pair_of(result, 2).step]
    assert round(le.BLOCK_SLACK * 60) < e3.left.extended_px <= 20 and e3.left.crossed_px >= 1
    assert e3.left.limited and e3.left.past_block_px >= 40
    assert 68 <= e2.left.extended_px <= 72 and not e2.left.limited
    assert pair_of(result, 2).tier == pair_of(result, 3).tier == "anchor"
    result.xml_path = Path("/x.xml")
    rows = {r["edition_index"]: r for r in ael.pair_records(result)}
    assert rows[3]["block"] == [875.0, 1900.0] and rows[3]["block_limited"] == [True, False]
    assert rows[3]["past_block_left"] >= 40 / 60 and rows[2]["own_ink_outside"] == 0.0
    assert rows[2]["overlap"] == 0.0 and rows[2]["raised"] is not None


def test_own_ink_gate_rejects_a_line_whose_words_stay_outside() -> None:
    """Line 6 has three words beyond a gap the walk does not bridge, inside the text block: measured over the
    whole block (``own_ink_reach`` inf, the round-2 measure) a fifth of its band's ink is outside its polygon,
    so the gate rejects it; ``inf`` keeps it. Round 3 measures the line's x-range plus half a line height
    only: the words 0.67 line heights away are not counted and the line is kept."""
    page, mask = inked_page()
    x0, y = page["lines"][6]["baseline"][0]
    line_ink(mask, int(x0) - 190, int(x0) - 50, y)                             # 40 px blank before the polygon
    ink = le.label_ink(mask)
    whole = ael.Params(own_ink_reach=math.inf)
    result = aligned(page)
    ael.extend_pairs(result, ink, whole)
    ael.gate_extents(result, whole)
    p6 = pair_of(result, 6)
    ext = result.geometry[p6.step]
    assert ext.left.extended_px == 0 and 0.15 < ext.own_ink_outside < 0.3
    assert p6.tier == "rejected" and p6.reason == ael.REJECT_OWN_INK
    assert {p.tier for p in result.alignment.pairs if p.e != 6} == {"anchor"}
    result = aligned(page)
    gate_off = ael.Params(own_ink_reach=math.inf, max_own_ink_outside=math.inf)
    ael.extend_pairs(result, ink, gate_off)
    ael.gate_extents(result, gate_off)
    assert pair_of(result, 6).tier == "anchor"
    off = ael.Params(extend_to_ink=False, own_ink_reach=math.inf)
    result = aligned(page)
    ael.extend_pairs(result, ink, off)
    assert result.geometry[pair_of(result, 6).step].own_ink_outside == pytest.approx(ext.own_ink_outside, abs=0.01)
    result = aligned(page)
    ael.extend_pairs(result, ink, ael.Params())
    ael.gate_extents(result, ael.Params())
    assert pair_of(result, 6).tier == "anchor" and result.geometry[pair_of(result, 6).step].own_ink_outside == 0.0


def test_overlap_gate_and_reason_order() -> None:
    """A polygon that newly covers more than 5 % of another line's polygon is rejected for the overlap (after
    steal, before ink beyond and own ink); ``inf`` disables it."""
    page, mask = inked_page()
    result = aligned(page)
    ael.extend_pairs(result, le.label_ink(mask), ael.Params())
    p1, p2 = pair_of(result, 1), pair_of(result, 2)
    g1, g2 = result.geometry[p1.step], result.geometry[p2.step]
    result.geometry[p1.step] = le.LineExtent(g1.baseline, g1.boundary, g1.left, g1.right, True, overlap=0.2,
                                             overlap_key=2, own_ink_outside=0.5)
    result.geometry[p2.step] = le.LineExtent(g2.baseline, g2.boundary, g2.left, g2.right, own_ink_outside=0.5)
    ael.gate_extents(result, ael.Params())
    assert p1.reason == ael.REJECT_OVERLAP and p2.reason == ael.REJECT_OWN_INK
    for p in (p1, p2):
        p.tier, p.reason = "anchor", ""
    ael.gate_extents(result, ael.Params(max_overlap=math.inf, max_own_ink_outside=math.inf))
    assert p1.tier == p2.tier == "anchor"


def test_drop_interpolated() -> None:
    """``drop_interpolated`` rejects the interpolated tier, also a count match that fell back to it."""
    steps = [ael.Step("match", j, (j,)) for j in range(4)] + [ael.Step("skip_s", None, (4,)), ael.Step("match", 4, (5,))]
    pairs = make_pairs(steps, [0.8, 0.0, 0.0, 0.7, 0.0], [20, 0, 0, 15, 0])
    ael.assign_tiers(steps, pairs, ael.Params(drop_interpolated=True))
    assert [p.tier for p in pairs] == ["anchor", "rejected", "rejected", "anchor", "rejected"]
    assert pairs[1].reason == pairs[2].reason == ael.REJECT_INTERPOLATED_DROPPED
    identity = [ael.Step("match", j, (j,)) for j in range(5)]
    count = make_pairs(identity, [0.8, 0.3, 0.3, 0.8, 0.3], [20, 10, 10, 20, 10], [0.1] * 5)
    ael.assign_tiers(identity, count, ael.Params(drop_count_match=True, drop_interpolated=True))
    assert [p.tier for p in count] == ["anchor", "rejected", "rejected", "anchor", "rejected"]
    assert count[1].reason == ael.REJECT_INTERPOLATED_DROPPED and count[4].reason == ael.REJECT_COUNT_MATCH_DROPPED


def test_duplicate_positions_compare_nfc_targets() -> None:
    """Targets equal after NFC (and whitespace collapsing) are duplicates; placeholders never are."""
    composed, decomposed = "قال آخر", "قال  آخر"
    lines = [ael.EditionLine(0, composed, composed, ""), ael.EditionLine(1, "x", "بسم الله", ""),
             ael.EditionLine(2, decomposed, decomposed, ""), ael.EditionLine(3, "[...]", "", ""),
             ael.EditionLine(4, "[...]", "", "")]
    assert ael.duplicate_positions(lines) == {0, 2}


def test_drop_duplicate_texts_rejects_every_copy() -> None:
    """A page whose edition repeats a line (a pen trial): with ``drop_duplicate_texts`` both copies are
    rejected (they still anchor the path), the other lines are kept; off by default."""
    edition = EDITION[:5] + [EDITION[1]] + EDITION[6:]
    page = synthetic_page(edition, edition)
    *_, alignment, skip = ael.align_page(page, "\n".join(edition), ael.Params())
    assert skip == "" and {p.tier for p in alignment.pairs} == {"anchor"}
    *_, alignment, skip = ael.align_page(page, "\n".join(edition), ael.Params(drop_duplicate_texts=True))
    tiers = {p.e: (p.tier, p.reason) for p in alignment.pairs}
    assert tiers[1] == tiers[5] == ("rejected", ael.REJECT_DUPLICATE) and skip == ""
    assert {tiers[e][0] for e in (0, 2, 3, 4, 6)} == {"anchor"}
    assert all(p.anchored for p in alignment.pairs if p.e in (1, 5))


def test_cli_flags_round_two() -> None:
    """Round-2 options parse with their documented defaults; the raise needs the extension."""
    args = ael.build_parser().parse_args([])
    assert (args.raise_top, args.block_slack, args.block_thirds, args.max_overlap, args.max_own_ink_outside,
            args.drop_interpolated, args.drop_duplicate_texts) == (True, 0.3, False, 0.05, 0.15, False, False)
    args = ael.build_parser().parse_args(["--no-raise-top", "--block-slack", "inf", "--block-thirds",
                                          "--max-own-ink-outside", "0.2", "--drop-interpolated",
                                          "--drop-duplicate-texts", "--neighbour-clamp", "0.2"])
    assert not args.raise_top and math.isinf(args.block_slack) and args.block_thirds
    assert args.max_own_ink_outside == 0.2 and args.drop_interpolated and args.drop_duplicate_texts
    assert ael.extent_config(ael.Params()).raise_top and not ael.extent_config(ael.Params(extend_to_ink=False)).raise_top
    assert ael.extent_config(ael.Params(neighbour_clamp=0.2)).clamp == 0.2


# ----------------------------------------------------------------------------- round 3: text guards, end checks, gates

def test_missing_words_finds_an_edition_word_the_read_skips_but_not_a_misread_one() -> None:
    """A word aligned to nothing in the read is missing (with its position along the line); a misread word
    (substitutions) and short words are not."""
    line = ael.skeleton("واعلم ان القمح قد غلا في السوق")
    skipped = ael.skeleton("واعلم ان قد غلا في السوق")
    missing = ael.missing_words(skipped, line)
    assert [k for k, _ in missing] == [2] and 0.2 < missing[0][1] < 0.5
    misread = ael.skeleton("واعلم ان الفمخ قد غلا في السوق")
    assert ael.missing_words(misread, line) == []
    assert ael.missing_words(ael.skeleton("واعلم القمح قد غلا في السوق"), line) == []    # "ان" is too short
    assert ael.missing_words("", line) == [] and ael.missing_words(line, "") == []


def test_long_read_guard_rejects_a_read_with_another_lines_words() -> None:
    """A read with three more words than its edition line (and 1.15 times its letters; a read that only splits
    words is not longer) or more words and a span sim of at most 0.5 is much longer: an accepted pair becomes
    REJECT_LONG_READ; ``long_read_words`` 0 switches the guard off."""
    edition = list(EDITION)
    reads = list(EDITION)
    reads[2] = EDITION[2] + " ثم قال لنا"
    page = synthetic_page(edition, reads)
    *_, alignment, _ = ael.align_page(page, "\n".join(edition), ael.Params())
    p2 = next(p for p in alignment.pairs if p.e == 2)
    assert p2.read_words - p2.edition_words == 3 and p2.tier == "rejected" and p2.reason == ael.REJECT_LONG_READ
    *_, alignment, _ = ael.align_page(page, "\n".join(edition), ael.Params(long_read_words=0))
    assert next(p for p in alignment.pairs if p.e == 2).tier == "anchor"
    pair = ael.Pair(0, 0, (0,), 0.5, 1.0, 20, read_words=5, edition_words=4, cover_sim=0.5)
    assert ael.long_read(pair, ael.Params())
    pair.cover_sim = pair.sim = 0.7
    assert not ael.long_read(pair, ael.Params())
    split = ael.Pair(0, 0, (0,), 0.77, 1.0, 20, read_words=9, edition_words=6, read_chars=24, edition_chars=23)
    assert not ael.long_read(split, ael.Params())                     # a read that only splits words
    split.read_chars = 30
    assert ael.long_read(split, ael.Params())


def painted(page: Dict[str, Any], clip_left: Optional[Dict[int, int]] = None) -> np.ndarray:
    """Ink of a synthetic page: each segment's words plus ``clip_left[n]`` pixels beyond line ``n``'s left end.

    :param page: Segmenter JSON.
    :param clip_left: ``{segment: pixels}``.
    :return: Ink mask.
    """
    mask = np.zeros((1400, 2000), dtype=bool)
    for n, raw in enumerate(page["lines"]):
        x0, x1, y = raw["baseline"][0][0], raw["baseline"][-1][0], raw["baseline"][0][1]
        line_ink(mask, int(math.ceil(x0)) - (clip_left or {}).get(n, 0), int(x1), y)
    return mask


def test_supralinear_insertion_is_merged_or_the_line_is_rejected() -> None:
    """Line 3's read skips "القمح": a short unplaced segment just above the line near that word's position is
    merged into the line (a supralinear insertion); with no segment and no ink there, the line is rejected
    with REJECT_MISSING_WORD."""
    reads = list(EDITION)
    reads[3] = "واعلم ان قد غلا في السوق هذه الايام"
    page = synthetic_page(EDITION, reads)
    mask = painted(page)
    result = ael.PageResult("pgpar_1_0_page", "1", "", "/abs/p.png", (2000, 1400))
    result.segments, result.noise, result.edition, result.alignment, result.skip = ael.align_page(
        page, "\n".join(EDITION), ael.Params())
    p3 = pair_of(result, 3)
    assert p3.tier == "anchor" and [k for k, _ in p3.missing] == [2]
    group = [result.alignment.order[i] for i in p3.s]
    x = ael.missing_x(p3, group)[0]
    y = page["lines"][3]["baseline"][0][1]
    mask[int(y) - 80:int(y) - 62, int(x) - 30:int(x) + 30] = True       # the inserted word, above the line
    box = ael.make_seg_line(seg(98, x - 35, x + 35, y - 62, "", height=24.0))
    result.segments.append(box)
    result.noise.append(98)
    ink = le.label_ink(mask)
    ael.extend_pairs(result, ink, ael.Params())
    ael.gate_extents(result, ael.Params())
    ext = result.geometry[p3.step]
    assert result.notes[p3.step]["supralinear"] == "box:98" and ext.forced == (98,) and p3.tier == "anchor"
    assert covers(ext.boundary, x, y - 70) and 98 in result.absorbed_segments()
    assert "l98" not in {l.line_id for l in ael.out_lines(result)}
    bare = ael.PageResult("pgpar_1_0_page", "1", "", "/abs/p.png", (2000, 1400))
    bare.segments, bare.noise, bare.edition, bare.alignment, bare.skip = ael.align_page(
        page, "\n".join(EDITION), ael.Params())
    ael.extend_pairs(bare, le.label_ink(painted(page)), ael.Params())
    ael.gate_extents(bare, ael.Params())
    assert bare.notes[pair_of(bare, 3).step]["supralinear"] == "missing"
    assert pair_of(bare, 3).reason == ael.REJECT_MISSING_WORD
    ink_only = ael.PageResult("pgpar_1_0_page", "1", "", "/abs/p.png", (2000, 1400))
    ink_only.segments, ink_only.noise, ink_only.edition, ink_only.alignment, ink_only.skip = ael.align_page(
        page, "\n".join(EDITION), ael.Params())
    ael.extend_pairs(ink_only, ink, ael.Params())
    ael.gate_extents(ink_only, ael.Params())
    q3 = pair_of(ink_only, 3)
    assert ink_only.notes[q3.step]["supralinear"].startswith("ink:") and q3.tier == "anchor"
    assert covers(ink_only.geometry[q3.step].boundary, x, y - 70)


def test_line_ending_in_a_colon_takes_no_detached_filler_glyph() -> None:
    """Edition line 2 ends in ':' and a detached glyph sits 8 px beyond its left end: the left end moves along
    connected ink only, so the filler stays out; without the colon the walk takes it; ``end_checks`` off
    restores round 2."""
    for ending, params, taken in (("", ael.Params(), True), (" :", ael.Params(), False),
                                  (" :", ael.Params(end_checks=False), True)):
        edition = list(EDITION)
        edition[2] = EDITION[2] + ending
        page = synthetic_page(edition, edition)
        mask = painted(page)
        x0, y = page["lines"][2]["baseline"][0]
        mask[int(y) - 20:int(y) + 2, int(x0) - 30:int(x0) - 8] = True     # filler glyph
        result = ael.PageResult("pgpar_1_0_page", "1", "", "/abs/p.png", (2000, 1400))
        result.segments, result.noise, result.edition, result.alignment, result.skip = ael.align_page(
            page, "\n".join(edition), params)
        ael.extend_pairs(result, le.label_ink(mask), params)
        p2 = pair_of(result, 2)
        ext = result.geometry[p2.step]
        assert (ext.left.extended_px >= 20) == taken, ending
        assert ("connected_only_end" in result.notes.get(p2.step, {})) == (not taken)


def test_end_box_whose_read_has_no_counterpart_is_not_absorbed() -> None:
    """A box just beyond line 2's left end is absorbed when its read is the line's last word; with a read
    that matches none of the line's end words (and no other line) it is not absorbable: the walk leaves it
    out and it is written as a TextLine of its own."""
    page, mask = inked_page({2: 70})
    x0, y = page["lines"][2]["baseline"][0]
    ink = le.label_ink(mask)
    for read, absorbed in (("كثيرا", True), ("ززززز", False)):
        result = aligned(page)
        result.segments.append(ael.make_seg_line(seg(98, x0 - 64, x0 - 14, y, read)))
        ael.extend_pairs(result, ink, ael.Params())
        ael.gate_extents(result, ael.Params())
        p2 = pair_of(result, 2)
        assert (98 in result.geometry[p2.step].absorbed) == absorbed, read
        assert ("not_absorbable" in result.notes.get(p2.step, {})) == (not absorbed)
        assert p2.tier == "anchor"
    assert ael.end_read_matches(ael.skeleton("كثيرا"), ael.skeleton(EDITION[2]), True)
    assert not ael.end_read_matches(ael.skeleton("كثيرا"), ael.skeleton(EDITION[2]), False)
    assert ael.end_read_matches("", ael.skeleton(EDITION[2]), True)


def test_end_shift_gate_rejects_an_l_shaped_end() -> None:
    """A line whose end took detached ink 0.6 line heights below its own ink is rejected (after steal, overlap and
    ink beyond, before the missing-word and own-ink gates); ``inf`` disables the gate."""
    page, mask = inked_page()
    result = aligned(page)
    ael.extend_pairs(result, le.label_ink(mask), ael.Params())
    p1, p2 = pair_of(result, 1), pair_of(result, 2)
    g1, g2 = result.geometry[p1.step], result.geometry[p2.step]
    result.geometry[p1.step] = dataclasses.replace(g1, end_shift=(0.6, 0.0), own_ink_outside=0.5)
    result.geometry[p2.step] = dataclasses.replace(g2, end_shift=(0.0, -0.3))
    ael.gate_extents(result, ael.Params())
    assert p1.reason == ael.REJECT_END_SHIFT and p2.tier == "anchor"
    p1.tier, p1.reason = "anchor", ""
    ael.gate_extents(result, ael.Params(max_end_shift=math.inf, max_own_ink_outside=math.inf))
    assert p1.tier == "anchor"


def test_final_overlaps_trim_the_farther_line_or_reject_it() -> None:
    """Line 1's final polygon reaches down over line 2's words: the shared region holds line 2's ink only, so
    line 1 is trimmed (and stays accepted, overlap_final recorded); when line 2's polygon also reaches up over
    a word of line 1 written low (nearer line 1's baseline), trimming would cut it: line 1 is rejected with
    REJECT_OVERLAP_TRIM."""
    page, mask = inked_page()
    y1 = page["lines"][1]["baseline"][0][1]
    y2 = page["lines"][2]["baseline"][0][1]
    for own_word, trimmed in ((False, True), (True, False)):
        m = mask.copy()
        if own_word:
            m[int(y1) + 35:int(y1) + 45, 1500:1560] = True                # a word of line 1 written low
        ink = le.label_ink(m)
        result = aligned(page)
        ael.extend_pairs(result, ink, ael.Params())
        p1, p2 = pair_of(result, 1), pair_of(result, 2)
        g1, g2 = result.geometry[p1.step], result.geometry[p2.step]
        xs = [p[0] for p in g1.boundary]
        tall = [(min(xs), y1 - 50), (max(xs), y1 - 50), (max(xs), y2 - 10), (min(xs), y2 - 10)]
        result.geometry[p1.step] = dataclasses.replace(g1, boundary=tuple(tall))
        if own_word:
            xs2 = [p[0] for p in g2.boundary]
            high = [(min(xs2), y1 + 30), (max(xs2), y1 + 30), (max(xs2), y2 + 20), (min(xs2), y2 + 20)]
            result.geometry[p2.step] = dataclasses.replace(g2, boundary=tuple(high))
        ael.final_overlaps(result, ink, ael.Params())
        ext = result.geometry[p1.step]
        if trimmed:
            assert p1.tier == "anchor" and ext.trimmed and ext.overlap_final <= 0.05
            x2 = (page["lines"][2]["baseline"][0][0] + 1900) / 2
            assert covers(tall, x2, y2 - 20) and not covers(ext.boundary, x2, y2 - 20)
        else:
            assert p1.tier == "rejected" and p1.reason == ael.REJECT_OVERLAP_TRIM and ext.trim_cut_own
        assert pair_of(result, 2).tier == "anchor"
        result.xml_path = Path("/x.xml")
        row = {r["edition_index"]: r for r in ael.pair_records(result)}[1]
        assert row["trimmed"] == trimmed and row["overlap_final"] is not None


def test_cli_flags_round_three() -> None:
    """Round-3 options parse with their documented defaults and reach the walk configuration."""
    args = ael.build_parser().parse_args([])
    assert (args.spacing_above, args.spacing_below, args.detect_unsegmented, args.foreign_components,
            args.column_blocks, args.gutters, args.max_end_shift, args.long_read_words, args.supralinear,
            args.end_checks, args.own_ink_reach, args.block_start_percentile) == \
        (0.55, 0.45, True, True, True, True, 0.5, 3, True, True, 0.5, 50.0)
    args = ael.build_parser().parse_args(["--no-detect-unsegmented", "--spacing-above", "inf", "--stub-width", "0",
                                          "--no-column-blocks", "--own-ink-reach", "inf", "--no-supralinear"])
    params = ael.Params(**{f.name: getattr(args, f.name) for f in dataclasses.fields(ael.Params)})
    config = ael.extent_config(params)
    assert not config.detect and math.isinf(config.spacing_above) and config.stub_width == 0
    assert math.isinf(config.own_ink_reach) and not params.column_blocks and not params.supralinear
