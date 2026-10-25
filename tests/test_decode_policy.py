"""Tests for the decode policy (src/datasets/consensus/decode_policy.py) and its benchmark driver.

The driver tests run on a temporary two-to-three page benchmark. The VLM call (``transcribe_with_lm_studio``) and
the LM Studio health check are monkeypatched everywhere they are bound, so nothing reaches LM Studio, Kraken or
the network.
"""
import dataclasses
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest
from PIL import Image, ImageDraw

from src.datasets.consensus import decode_policy as dp
from src.datasets.consensus import linewise
from src.datasets.consensus.read_health import read_health
from src.datasets.evaluations.helper_eval_scripts import run_decode_policy as driver
from src.models.ocr import lms_transcriber

HEBREW = "אבגדהוזחטיכלמנסעפצקרשת"
PAGE = (2000, 2400)
LINE_H = 60
PITCH = 80


def random_line(seed: int, n_letters: int = 30) -> str:
    """A line of pseudo-random Hebrew words (distinct per seed, so no 12-letter run repeats across lines).

    :param seed: Random seed.
    :type seed: int
    :param n_letters: Letters in the line.
    :type n_letters: int
    :return: Line text with exactly ``n_letters`` letters in words of 3-6 letters.
    :rtype: str
    """
    rng = random.Random(seed)
    letters = [rng.choice(HEBREW) for _ in range(n_letters)]
    words, i = [], 0
    while i < len(letters):
        k = rng.randint(3, 6)
        words.append("".join(letters[i:i + k]))
        i += k
    return " ".join(words)


def column(x0: int, x1: int, n: int, seed: int, top: int = 200, n_letters: int = 30) -> List[Tuple[str, List[int]]]:
    """Kraken lines of one column: ``n`` lines of ``n_letters`` letters, one line pitch apart.

    :param x0: Left edge of the column (px).
    :type x0: int
    :param x1: Right edge (px).
    :type x1: int
    :param n: Number of lines.
    :type n: int
    :param seed: Seed of the first line's text.
    :type seed: int
    :param top: y of the first line (px).
    :type top: int
    :param n_letters: Letters per line.
    :type n_letters: int
    :return: ``(text, bbox)`` per line, top to bottom.
    :rtype: List[Tuple[str, List[int]]]
    """
    return [(random_line(seed + i, n_letters), [x0, top + i * PITCH, x1, top + i * PITCH + LINE_H]) for i in range(n)]


def kraken_json(lines: Sequence[Tuple[str, List[int]]], size: Tuple[int, int] = PAGE, rotation: int = 0,
                view_bboxes: Optional[Sequence[List[int]]] = None) -> Dict[str, Any]:
    """A cached Kraken ``/transcribe_lines`` response.

    :param lines: ``(text, bbox)`` per line.
    :type lines: Sequence[Tuple[str, List[int]]]
    :param size: Original image size ``(width, height)``.
    :type size: Tuple[int, int]
    :param rotation: View rotation.
    :type rotation: int
    :param view_bboxes: View-frame boxes per line (rotated views).
    :type view_bboxes: Optional[Sequence[List[int]]]
    :return: The response dict.
    :rtype: Dict[str, Any]
    """
    out = []
    for i, (text, bbox) in enumerate(lines):
        rec: Dict[str, Any] = {"text": text, "bbox": list(bbox), "confidence": 0.9}
        if view_bboxes is not None:
            rec["view_bbox"] = list(view_bboxes[i])
        out.append(rec)
    return {"lines": out, "image_size": list(size), "view": {"rotation": rotation, "scale": 1.0},
            "polygon_failures": 0, "used_binarization": True}


def two_column_lines(n: int = 20) -> List[Tuple[str, List[int]]]:
    """Lines of a two-column page: right column x 1100-1900, left x 100-900, served left column first.

    :param n: Lines per column.
    :type n: int
    :return: ``(text, bbox)`` per line.
    :rtype: List[Tuple[str, List[int]]]
    """
    return column(100, 900, n, seed=500) + column(1100, 1900, n, seed=100)


def single_column_lines(n: int = 20, seed: int = 900) -> List[Tuple[str, List[int]]]:
    """Lines of a one-column page (x 300-1700).

    :param n: Number of lines.
    :type n: int
    :param seed: Seed of the first line.
    :type seed: int
    :return: ``(text, bbox)`` per line.
    :rtype: List[Tuple[str, List[int]]]
    """
    return column(300, 1700, n, seed=seed)


def texts_of(lines: Sequence[Tuple[str, List[int]]]) -> List[str]:
    """The texts of some Kraken lines.

    :param lines: ``(text, bbox)`` per line.
    :type lines: Sequence[Tuple[str, List[int]]]
    :return: Texts.
    :rtype: List[str]
    """
    return [t for t, _ in lines]


def loop_text(lines: Sequence[str]) -> str:
    """A looping read: three real lines, then one line over and over.

    :param lines: Real line texts.
    :type lines: Sequence[str]
    :return: Read text.
    :rtype: str
    """
    return "\n".join(list(lines[:3]) + [lines[3]] * 30)


TH = dp.thresholds_for(2500)


# ----------------------------------------------------------------------------- Kraken geometry


def test_kraken_units_mirror_reorder_ocr_lines() -> None:
    """Units drop empty lines, keep their Kraken index, count Hebrew letters and use the median line height."""
    lines = [("  שלום  ", [10, 10, 200, 70]), ("   ", [10, 90, 200, 150]), ("ab", [10, 170, 200, 200]),
             ("ברכה רבה", [10, 250, 300, 330])]
    units, line_h, size = dp.kraken_units(kraken_json(lines, size=(400, 500)))
    assert [u["idx"] for u in units] == [0, 2, 3]
    assert units[0]["text"] == "שלום" and units[0]["letters"] == 4 and units[1]["letters"] == 0
    assert line_h == 70.0                                   # median of the >= 2-letter units' heights (60, 80)
    assert size == (400, 500)


def test_kraken_units_use_the_view_frame_for_rotated_pages() -> None:
    """A rotated view plans on ``view_bbox`` with the width and height swapped; without view boxes it stays put."""
    lines = [("שלום עליכם", [2400, 100, 2500, 700]), ("ברכה והצלחה", [2280, 100, 2380, 700])]
    views = [[100, 0, 700, 100], [100, 120, 700, 220]]
    rotated = kraken_json(lines, size=(2600, 1000), rotation=90, view_bboxes=views)
    units, _, size = dp.kraken_units(rotated)
    assert dp.kraken_rotation(rotated) == 90 and size == (1000, 2600)
    assert [u["box"] for u in units] == [(100.0, 0.0, 700.0, 100.0), (100.0, 120.0, 700.0, 220.0)]
    no_views = kraken_json(lines, size=(2600, 1000), rotation=90)
    assert dp.kraken_rotation(no_views) == 0 and dp.kraken_units(no_views)[2] == (2600, 1000)


def test_view_image_matches_the_kraken_view_frame() -> None:
    """``view_image`` maps an original pixel (x, y) to (y, W-1-x) for 90 degrees, as k7/views does."""
    im = Image.new("L", (40, 30), 0)
    im.putpixel((5, 7), 255)
    view = dp.view_image(im, 90)
    assert view.size == (30, 40) and view.getpixel((7, 40 - 1 - 5)) == 255
    assert dp.view_image(im, 0) is im


# ----------------------------------------------------------------------------- columns


def test_two_column_page_splits_rightmost_first_with_gutter_clipped_crops() -> None:
    """Two columns come back rightmost first; crops keep a margin but never cross the gutter."""
    units, line_h, size = dp.kraken_units(kraken_json(two_column_lines()))
    cols = dp.detect_columns(units, line_h, size, require_ink_gap=True, ink_gap_x=1000)
    assert len(cols) == 2
    right, left = cols
    assert right.index == 0 and right.core == (1100.0, 1900.0) and left.core == (100.0, 900.0)
    assert right.line_indices == tuple(range(20, 40)) and left.line_indices == tuple(range(20))
    margin = dp.crop_margin(line_h)
    assert right.crop == (1100 - margin, 200 - margin, 1900 + margin, 200 + 19 * PITCH + LINE_H + margin)
    assert left.crop[2] <= 1000 <= right.crop[0]           # the gutter stays out of both crops' text
    assert right.letters == left.letters == 600


def test_ink_gap_gate() -> None:
    """With the gate on, a Kraken split without an ink gap is one column; with the gate off it stands."""
    units, line_h, size = dp.kraken_units(kraken_json(two_column_lines()))
    assert len(dp.detect_columns(units, line_h, size, require_ink_gap=True, ink_gap_x=None)) == 1
    assert len(dp.detect_columns(units, line_h, size, require_ink_gap=False)) == 2


def test_single_column_page_gets_one_plan_over_the_ink_envelope() -> None:
    """One column: one plan covering the ink envelope plus the margin."""
    units, line_h, size = dp.kraken_units(kraken_json(single_column_lines()))
    (col,) = dp.detect_columns(units, line_h, size, require_ink_gap=False)
    margin = dp.crop_margin(line_h)
    assert col.crop == (300 - margin, 200 - margin, 1700 + margin, 200 + 19 * PITCH + LINE_H + margin)
    assert col.line_indices == tuple(range(20))


def test_false_gutter_below_the_letter_share_stays_one_column() -> None:
    """A marginal band holding < 12 % of the letters merges into the main column, ink gap or not."""
    main = column(600, 1900, 20, seed=300)
    margin_notes = column(100, 300, 8, seed=700, n_letters=6)        # 48 of 648 letters = 7.4 %
    units, line_h, size = dp.kraken_units(kraken_json(main + margin_notes))
    assert len(dp.detect_columns(units, line_h, size, require_ink_gap=True, ink_gap_x=450)) == 1
    assert len(dp.detect_columns(units, line_h, size, require_ink_gap=False)) == 1


def test_no_kraken_lines_plan_covers_the_page() -> None:
    """Without Kraken text the single plan is the whole page."""
    (col,) = dp.detect_columns([], 1.0, PAGE)
    assert col.crop == (0, 0, PAGE[0], PAGE[1]) and col.line_indices == ()


def test_central_ink_gap_finds_the_gutter_and_matches_split_x(tmp_path: Path) -> None:
    """The PIL port of ``split_x`` finds a two-column gutter, finds none on a full-width block, and agrees with it."""
    from src.datasets.evaluations.helper_eval_scripts.probe_column_transcription import split_x

    two = Image.new("RGB", PAGE, (235, 225, 205))
    draw = ImageDraw.Draw(two)
    for _, box in two_column_lines():
        draw.rectangle(box, fill=(40, 30, 30))
    gap = dp.central_ink_gap(two)
    assert gap is not None and 900 <= gap <= 1100
    path = tmp_path / "two.jpg"
    two.save(path, quality=95)
    assert split_x(str(path))[0] == dp.central_ink_gap(Image.open(path))
    one = Image.new("RGB", PAGE, (235, 225, 205))
    draw = ImageDraw.Draw(one)
    for _, box in single_column_lines():
        draw.rectangle(box, fill=(40, 30, 30))
    assert dp.central_ink_gap(one) is None


# ----------------------------------------------------------------------------- health


def test_health_of_avoids_the_plain_text_capped_misfire() -> None:
    """A long healthy plain read is ``capped`` for read_health without a token count, ``ok`` for health_of."""
    lines = [random_line(2000 + i) for i in range(70)]
    text = "\n".join(lines)
    assert len(text) >= 2400
    assert read_health(text, lines, lines, 2500).label == "capped"     # the misfire
    health = dp.health_of(text, lines, 2500)
    assert health.label == "ok" and health.facts.reply_tokens == dp.estimate_tokens(text)
    assert dp.health_of(text, lines, 2500, reply_tokens=2450).label == "capped"   # a real count still decides


def test_health_of_detects_a_read_that_used_the_cap() -> None:
    """A plain read long enough to have used the cap is ``capped``, even when it ends in ``]``."""
    lines = [random_line(3000 + i) for i in range(20)]
    text = "\n".join(lines) + " [?]"
    assert dp.health_of(text, lines, 300).label == "capped"
    assert dp.health_of(text, lines, None).label == "ok"                # joined texts: no cap check


def test_judge_read_flags_overgeneration() -> None:
    """More than max(3x Kraken letters, 400) letters fails as ``overgen`` when nothing else failed."""
    kraken = [random_line(4000 + i, 25) for i in range(4)]             # 100 Kraken letters
    read = "\n".join(random_line(4100 + i, 30) for i in range(17))     # 510 letters, no loop
    verdict = dp.judge_read(read, kraken, 2500, TH)
    assert verdict.label == dp.LABEL_OVERGEN and dp.LABEL_OVERGEN in verdict.flags
    assert dp.judge_read("\n".join(kraken), kraken, 2500, TH).label == "ok"


# ----------------------------------------------------------------------------- planning


@pytest.mark.parametrize("make_text, reason", [
    (lambda ls: "\n".join(ls[:10]), "whole_page_skipped_text"),
    (loop_text, "whole_page_loop"),
    (lambda ls: "", "whole_page_near_empty"),
    (lambda ls: None, "whole_page_missing"),
])
def test_plan_page_rescues_failed_whole_page_reads(make_text: Any, reason: str) -> None:
    """Under-read, looping, near-empty and missing whole-page reads of a single column are rescued by halves."""
    lines = single_column_lines()
    plan = dp.plan_page(kraken_json(lines), make_text(texts_of(lines)), TH)
    assert plan.action == dp.ACTION_RESCUE and plan.reason == reason
    top, bottom = plan.units
    assert (top.name, bottom.name) == ("top", "bottom") and top.role == bottom.role == dp.ROLE_HALF
    col = plan.columns[0].crop
    assert top.crop[1] == col[1] and bottom.crop[3] == col[3] and bottom.crop[1] < top.crop[3]   # halves overlap
    shared = {ln["idx"] for ln in top.seg_lines} & {ln["idx"] for ln in bottom.seg_lines}
    assert {ln["idx"] for ln in top.seg_lines} | {ln["idx"] for ln in bottom.seg_lines} == set(range(20))
    assert len(shared) <= 3


def test_plan_page_rescues_a_capped_read() -> None:
    """A full read that used (nearly) the whole token cap is rescued."""
    lines = single_column_lines()
    th = dp.PolicyThresholds(page_token_cap=300, crop_token_cap=300)
    plan = dp.plan_page(kraken_json(lines), "\n".join(texts_of(lines)), th)
    assert plan.action == dp.ACTION_RESCUE and plan.reason == "whole_page_capped"


def test_plan_page_keeps_a_healthy_read() -> None:
    """A healthy single-column read is kept and nothing is read."""
    lines = single_column_lines()
    plan = dp.plan_page(kraken_json(lines), "\n".join(texts_of(lines)), TH)
    assert plan.action == dp.ACTION_KEEP and plan.reason == "whole_page_ok" and plan.units == ()


def test_plan_page_reads_columns_whatever_the_whole_page_health() -> None:
    """Two columns (with an ink gap) are read column by column even when the whole-page read is healthy."""
    lines = two_column_lines()
    plan = dp.plan_page(kraken_json(lines), "\n".join(texts_of(lines)), TH, ink_gap_x=1000)
    assert plan.action == dp.ACTION_COLUMNS and plan.reason == "2_columns" and not plan.whole_page.failed
    assert [u.name for u in plan.units] == ["col0", "col1"]
    assert plan.units[0].kraken_texts == texts_of(lines[20:])            # rightmost column first
    assert plan.kraken_columns == 2 and plan.ink_gap_in_gutter is True
    gated = dp.plan_page(kraken_json(lines), "\n".join(texts_of(lines)), TH, ink_gap_x=None)
    assert gated.action == dp.ACTION_KEEP and gated.kraken_columns == 2


def test_plan_page_halves_the_whole_page_when_kraken_read_almost_nothing() -> None:
    """Kraken under 100 letters: its envelope is no guide, so the rescue halves cover the page."""
    lines = column(800, 1200, 5, seed=50, n_letters=3)                  # 15 Kraken letters
    plan = dp.plan_page(kraken_json(lines), "[...]\n" * 10, TH)
    assert plan.action == dp.ACTION_RESCUE and plan.reason == "whole_page_loop_kraken_blank"
    top, bottom = plan.units
    assert top.crop[:3] == (0, 0, PAGE[0]) and bottom.crop[2:] == PAGE


# ----------------------------------------------------------------------------- assembly


def column_plan_and_texts() -> Tuple[dp.PagePlan, List[str], List[str]]:
    """A two-column plan with a healthy whole-page read, and the two columns' texts.

    :return: ``(plan, right column lines, left column lines)``.
    :rtype: Tuple[dp.PagePlan, List[str], List[str]]
    """
    lines = two_column_lines()
    plan = dp.plan_page(kraken_json(lines), "\n".join(texts_of(lines)), TH, ink_gap_x=1000)
    return plan, texts_of(lines[20:]), texts_of(lines[:20])


def used(trace: Dict[str, Any]) -> List[str]:
    """The source each unit's text came from.

    :param trace: Assembly trace.
    :type trace: Dict[str, Any]
    :return: ``used`` per unit.
    :rtype: List[str]
    """
    return [u["used"] for u in trace["units"]]


def test_assemble_joins_healthy_columns_rightmost_first() -> None:
    """Healthy crop reads are joined right column first."""
    plan, right, left = column_plan_and_texts()
    reads = {"col0": dp.UnitReads(crop="\n".join(right)), "col1": dp.UnitReads(crop="\n".join(left))}
    text, trace = dp.assemble(plan, reads, "\n".join(left + right), TH)
    assert text == "\n".join(right + left)
    assert used(trace) == ["crop", "crop"] and trace["final_source"] == dp.FINAL_ASSEMBLED
    json.dumps(trace)
    json.dumps(plan.to_dict())


def test_assemble_falls_back_to_bands_when_a_crop_read_fails() -> None:
    """A looping crop read gives way to the healthy band reads of the same crop."""
    plan, right, left = column_plan_and_texts()
    reads = {"col0": dp.UnitReads(crop="\n".join(right)),
             "col1": dp.UnitReads(crop=loop_text(left), bands="\n".join(left), band_details=[{"bbox": [0, 0, 1, 1]}])}
    text, trace = dp.assemble(plan, reads, None, TH)
    assert text == "\n".join(right + left) and used(trace) == ["crop", "bands"]
    assert trace["units"][1]["attempts"][0]["health"]["label"] == "loop"
    assert trace["units"][1]["band_details"] == [{"bbox": [0, 0, 1, 1]}]


def test_assemble_falls_back_to_the_whole_page_share() -> None:
    """Crop and bands both failing: the unit takes its share of the (healthy) whole-page read."""
    plan, right, left = column_plan_and_texts()
    reads = {"col0": dp.UnitReads(crop="\n".join(right)), "col1": dp.UnitReads(crop="", bands="")}
    whole = "\n".join(r + "\n" + l for r, l in zip(right, left))     # read across the gutter, lines interleaved
    text, trace = dp.assemble(plan, reads, whole, TH)
    assert used(trace) == ["crop", "share"] and trace["final_source"] == dp.FINAL_ASSEMBLED
    assert text == "\n".join(right + left)


def test_assemble_returns_the_whole_page_read_when_a_unit_stays_unresolved() -> None:
    """A unit nothing resolves sends the page back to the whole-page read when that read is healthy."""
    lines = column(1100, 1900, 17, seed=100) + column(100, 900, 3, seed=500)    # left column: 15 % of the letters
    right, left = texts_of(lines[:17]), texts_of(lines[17:])
    whole = "\n".join(right)                                                     # 85 % of Kraken's letters: healthy
    plan = dp.plan_page(kraken_json(lines), whole, TH, ink_gap_x=1000)
    assert plan.action == dp.ACTION_COLUMNS and not plan.whole_page.failed
    reads = {"col0": dp.UnitReads(crop="\n".join(right)),
             "col1": dp.UnitReads(crop=loop_text(left + right), bands=loop_text(left + right))}
    text, trace = dp.assemble(plan, reads, whole, TH)
    assert text == whole and trace["final_source"] == dp.FINAL_WHOLE_PAGE
    assert trace["page_fallback"] == "unit_unresolved" and used(trace) == ["crop", "unresolved"]


def test_assemble_partial_when_the_whole_page_read_is_unhealthy() -> None:
    """With an unhealthy whole-page read, the resolved units stand and an unresolved one gives its best effort."""
    lines = texts_of(single_column_lines())
    whole = loop_text(lines)
    plan = dp.plan_page(kraken_json(single_column_lines()), whole, TH)
    top, bottom = plan.units
    reads = {"top": dp.UnitReads(crop="\n".join(top.kraken_texts)),
             "bottom": dp.UnitReads(crop=loop_text(bottom.kraken_texts), bands=loop_text(bottom.kraken_texts[1:]))}
    text, trace = dp.assemble(plan, reads, whole, TH)
    assert used(trace) == ["crop", "unresolved"] and trace["final_source"] == dp.FINAL_PARTIAL
    assert trace["page_fallback"] is None and text.endswith(bottom.kraken_texts[4])   # the bands text, joined


def test_assemble_rescue_joins_halves_once() -> None:
    """Two healthy half reads are joined with the lines both halves read kept once."""
    lines = single_column_lines()
    plan = dp.plan_page(kraken_json(lines), "", TH)
    top, bottom = plan.units
    reads = {"top": dp.UnitReads(crop="\n".join(top.kraken_texts)),
             "bottom": dp.UnitReads(crop="\n".join(bottom.kraken_texts))}
    text, trace = dp.assemble(plan, reads, "", TH)
    assert text == "\n".join(texts_of(lines)) and trace["final_source"] == dp.FINAL_ASSEMBLED
    assert trace["overlap_lines_dropped"] == len(set(top.kraken_texts) & set(bottom.kraken_texts))


def test_assemble_keep_returns_the_whole_page_read() -> None:
    """``keep`` returns the whole-page read verbatim."""
    lines = texts_of(single_column_lines())
    whole = "\n".join(lines)
    plan = dp.plan_page(kraken_json(single_column_lines()), whole, TH)
    text, trace = dp.assemble(plan, {}, whole, TH)
    assert text == whole and trace["final_source"] == dp.FINAL_WHOLE_PAGE and trace["units"] == []


def test_whole_page_shares_attribute_lines_to_units() -> None:
    """Whole-page lines go to the unit holding the matching Kraken line; invented lines go nowhere."""
    plan, right, left = column_plan_and_texts()
    whole = "\n".join([left[0], right[0], random_line(99999), right[1]])
    shares = dp.whole_page_shares(whole, plan.units)
    assert shares == {"col0": "\n".join([right[0], right[1]]), "col1": left[0]}


# ----------------------------------------------------------------------------- weak Kraken (under-segmented pages)

# The two religious-140 pages where Kraken under-segmented the page (v21b, 2026-10-12): Kraken letters per column
# (right column first), the crop reads' letters, and the healthy whole-page read's letters.
WEAK_KRAKEN_PAGES = {
    "KTIV_990051100930205171_FL202443168": ((20, 20, 20, 19, 19, 19), (16, 15, 15, 15), (506, 557), 1144),
    "KTIV_990051727480205171_FL203739117": ((19,) * 5, (17,) * 8, (587, 378), 1091),
}


def column_with_letters(x0: int, x1: int, counts: Sequence[int], seed: int,
                        top: int = 200) -> List[Tuple[str, List[int]]]:
    """Kraken lines of one column with the given letter counts, one line pitch apart.

    :param x0: Left edge of the column (px).
    :type x0: int
    :param x1: Right edge (px).
    :type x1: int
    :param counts: Letters per line, top to bottom.
    :type counts: Sequence[int]
    :param seed: Seed of the first line's text.
    :type seed: int
    :param top: y of the first line (px).
    :type top: int
    :return: ``(text, bbox)`` per line.
    :rtype: List[Tuple[str, List[int]]]
    """
    return [(random_line(seed + i, n), [x0, top + i * PITCH, x1, top + i * PITCH + LINE_H])
            for i, n in enumerate(counts)]


def lines_with_letters(n_letters: int, seed: int, per_line: int = 30) -> List[str]:
    """Pseudo-random Hebrew lines holding exactly ``n_letters`` letters.

    :param n_letters: Letters in all.
    :type n_letters: int
    :param seed: Seed of the first line.
    :type seed: int
    :param per_line: Letters per full line.
    :type per_line: int
    :return: Lines.
    :rtype: List[str]
    """
    sizes = [per_line] * (n_letters // per_line) + ([n_letters % per_line] if n_letters % per_line else [])
    return [random_line(seed + i, n) for i, n in enumerate(sizes)]


def weak_kraken_page(doc_id: str) -> Tuple[List[Tuple[str, List[int]]], str, List[str]]:
    """One of :data:`WEAK_KRAKEN_PAGES` rebuilt with its letter counts.

    Kraken found a few lines per column (x 1100-1900 and 100-900). Each crop read holds its column's Kraken lines
    and the text Kraken missed; the whole-page read holds both crop reads plus a few more lines, so its share of
    each column is only the Kraken-matched lines.

    :param doc_id: Key of :data:`WEAK_KRAKEN_PAGES`.
    :type doc_id: str
    :return: ``(Kraken lines, whole-page read, crop reads rightmost first)``.
    :rtype: Tuple[List[Tuple[str, List[int]]], str, List[str]]
    """
    right_counts, left_counts, (right_letters, left_letters), whole_letters = WEAK_KRAKEN_PAGES[doc_id]
    right = column_with_letters(1100, 1900, right_counts, seed=7000)
    left = column_with_letters(100, 900, left_counts, seed=7100)
    crop0 = texts_of(right) + lines_with_letters(right_letters - sum(right_counts), seed=7200)
    crop1 = texts_of(left) + lines_with_letters(left_letters - sum(left_counts), seed=7400)
    extra = lines_with_letters(whole_letters - right_letters - left_letters, seed=7600)
    return left + right, "\n".join(crop0 + crop1 + extra), ["\n".join(crop0), "\n".join(crop1)]


def weak_kraken_assembly(doc_id: str, th: dp.PolicyThresholds) -> Tuple[str, Dict[str, Any], str, List[str]]:
    """Plan and assemble a weak-Kraken page from its two crop reads (no band reads).

    :param doc_id: Key of :data:`WEAK_KRAKEN_PAGES`.
    :type doc_id: str
    :param th: Policy thresholds.
    :type th: dp.PolicyThresholds
    :return: ``(page text, trace, whole-page read, crop reads)``.
    :rtype: Tuple[str, Dict[str, Any], str, List[str]]
    """
    lines, whole, crops = weak_kraken_page(doc_id)
    plan = dp.plan_page(kraken_json(lines), whole, th, ink_gap_x=1000)
    assert plan.action == dp.ACTION_COLUMNS and plan.whole_page.label == "ok"
    reads = {"col0": dp.UnitReads(crop=crops[0]), "col1": dp.UnitReads(crop=crops[1])}
    text, trace = dp.assemble(plan, reads, whole, th)
    json.dumps(trace)
    return text, trace, whole, crops


@pytest.mark.parametrize("doc_id", sorted(WEAK_KRAKEN_PAGES))
def test_weak_kraken_fixture_has_the_real_letter_counts(doc_id: str) -> None:
    """The rebuilt pages carry the real pages' Kraken, crop and whole-page letter counts."""
    right_counts, left_counts, crop_letters, whole_letters = WEAK_KRAKEN_PAGES[doc_id]
    lines, whole, crops = weak_kraken_page(doc_id)
    plan = dp.plan_page(kraken_json(lines), whole, TH, ink_gap_x=1000)
    assert plan.kraken_letters == sum(right_counts) + sum(left_counts)
    assert [u.kraken_letters for u in plan.units] == [sum(right_counts), sum(left_counts)]
    assert tuple(dp.letter_count(c) for c in crops) == crop_letters
    assert dp.letter_count(whole) == whole_letters


@pytest.mark.parametrize("doc_id", sorted(WEAK_KRAKEN_PAGES))
def test_without_either_rule_the_weak_kraken_page_regresses(doc_id: str) -> None:
    """The 2026-10-12 failure: the 3x Kraken rule rejects the correct crop reads and the shares lose the text."""
    th = dataclasses.replace(TH, weak_kraken_ratio=0.0, assembled_min_ratio=0.0)
    text, trace, whole, _ = weak_kraken_assembly(doc_id, th)
    assert trace["units"][0]["used"] == "share" and trace["units"][0]["attempts"][0]["health"]["label"] == "overgen"
    assert trace["final_source"] == dp.FINAL_ASSEMBLED and trace["page_fallback"] is None
    assert trace["final_letters"] < 0.5 * dp.letter_count(whole)


@pytest.mark.parametrize("doc_id, assembled", [("KTIV_990051100930205171_FL202443168", 117 + 61),
                                               ("KTIV_990051727480205171_FL203739117", 95 + 378)])
def test_assembly_shorter_than_a_healthy_whole_page_read_falls_back(doc_id: str, assembled: int) -> None:
    """Rule 1 alone (relaxation off): the share-built assembly holds < 0.8x the whole-page letters -> whole page."""
    th = dataclasses.replace(TH, weak_kraken_ratio=0.0)
    text, trace, whole, _ = weak_kraken_assembly(doc_id, th)
    assert text == whole and trace["final_source"] == dp.FINAL_WHOLE_PAGE
    assert trace["page_fallback"] == dp.FALLBACK_SHORTER == "assembled_shorter_than_whole_page"
    assert trace["assembled_letters"] == assembled and trace["whole_page_letters"] == WEAK_KRAKEN_PAGES[doc_id][3]
    assert trace["final_letters"] == trace["whole_page_letters"]
    assert trace["weak_kraken_relaxation"] is False and trace["overgen_runaway_letters"] is None
    assert len(trace["units"]) == 2 and trace["units"][0]["used"] == "share"      # the units' trace is kept


@pytest.mark.parametrize("doc_id", sorted(WEAK_KRAKEN_PAGES))
def test_weak_kraken_relaxation_keeps_the_crop_reads(doc_id: str) -> None:
    """Rule 2: Kraken under half the whole-page letters -> only a runaway past 1.5x the whole page over-generates."""
    right_counts, left_counts, crop_letters, whole_letters = WEAK_KRAKEN_PAGES[doc_id]
    text, trace, _, crops = weak_kraken_assembly(doc_id, TH)
    assert used(trace) == ["crop", "crop"] and text == "\n".join(crops)
    assert all(u["attempts"][0]["health"]["flags"] == [] for u in trace["units"])
    assert trace["final_source"] == dp.FINAL_ASSEMBLED and trace["page_fallback"] is None
    assert trace["final_letters"] == trace["assembled_letters"] == sum(crop_letters)    # 1,063 and 965 letters
    assert trace["final_letters"] >= TH.assembled_min_ratio * whole_letters             # rule 1 stays quiet
    assert trace["weak_kraken_relaxation"] is True and trace["kraken_letters"] == sum(right_counts + left_counts)
    assert trace["overgen_runaway_letters"] == 1.5 * whole_letters


def test_weak_kraken_limit_applies_only_to_a_healthy_whole_page_read() -> None:
    """No relaxation without a healthy whole-page read or when Kraken read enough; the limit keeps the 400 floor."""
    doc_id = "KTIV_990051100930205171_FL202443168"
    lines, whole, crops = weak_kraken_page(doc_id)
    kraken = kraken_json(lines)
    plan = dp.plan_page(kraken, whole, TH, ink_gap_x=1000)
    assert dp.weak_kraken_limit(plan, whole, TH) == 1716.0
    missing = dp.plan_page(kraken, None, TH, ink_gap_x=1000)
    assert dp.weak_kraken_limit(missing, None, TH) is None
    looping = loop_text(whole.splitlines())
    assert dp.weak_kraken_limit(dp.plan_page(kraken, looping, TH, ink_gap_x=1000), looping, TH) is None
    strong = "\n".join(texts_of(lines))                              # whole-page letters = Kraken letters
    assert dp.weak_kraken_limit(dp.plan_page(kraken, strong, TH, ink_gap_x=1000), strong, TH) is None
    small_lines = column_with_letters(300, 1700, (10, 10, 10, 10, 10), seed=8000)       # 50 Kraken letters
    small_whole = "\n".join(lines_with_letters(200, seed=8100))                         # 200 letters, healthy
    small = dp.plan_page(kraken_json(small_lines), small_whole, TH)
    assert dp.weak_kraken_limit(small, small_whole, TH) == 400.0                          # not 1.5 x 200
    texts = plan.units[0].kraken_texts
    assert dp.judge_read(crops[0], texts, 2500, TH).label == dp.LABEL_OVERGEN               # 506 > max(3 x 117, 400)
    assert dp.judge_read(crops[0], texts, 2500, TH, runaway_letters=1716.0).label == "ok"
    assert dp.judge_read(crops[0], texts, 2500, TH, runaway_letters=500.0).label == dp.LABEL_OVERGEN
    assert {"weak_kraken_ratio": 0.5, "runaway_ratio": 1.5, "assembled_min_ratio": 0.8}.items() <= TH.to_dict().items()


# ----------------------------------------------------------------------------- driver


def page_image(lines: Sequence[Tuple[str, List[int]]]) -> Image.Image:
    """A synthetic manuscript photo: dark bars where the Kraken lines are.

    :param lines: ``(text, bbox)`` per line.
    :type lines: Sequence[Tuple[str, List[int]]]
    :return: RGB image of size :data:`PAGE`.
    :rtype: Image.Image
    """
    im = Image.new("RGB", PAGE, (235, 225, 205))
    draw = ImageDraw.Draw(im)
    for _, box in lines:
        draw.rectangle(box, fill=(40, 30, 30))
    return im


def make_benchmark(root: Path, fallback: str, extra: Optional[Dict[str, Tuple[Any, ...]]] = None) -> Dict[str, Path]:
    """A three-page religious-format benchmark on disk: a two-column page, a looping page, a healthy page.

    :param root: Temporary folder.
    :type root: Path
    :param fallback: Stem of the whole-page reads.
    :type fallback: str
    :param extra: Further pages, ``doc_id -> (Kraken lines, n_columns, whole-page read from the line texts)``.
    :type extra: Optional[Dict[str, Tuple[Any, ...]]]
    :return: Paths ``spec``, ``outputs``, ``details``.
    :rtype: Dict[str, Path]
    """
    pages = {"DOC_TWO": (two_column_lines(), 2, None),
             "DOC_LOOP": (single_column_lines(seed=1500), 1, loop_text),
             "DOC_KEEP": (single_column_lines(seed=2500), 1, None), **(extra or {})}
    outputs, images = root / "outputs", root / "images"
    images.mkdir(parents=True)
    docs = []
    for doc_id, (lines, n_columns, make_whole) in pages.items():
        image = images / f"{doc_id}.jpg"
        page_image(lines).save(image, quality=92)
        out = outputs / doc_id
        out.mkdir(parents=True)
        (out / driver.KRAKEN_LINES_NAME).write_text(json.dumps(kraken_json(lines), ensure_ascii=False), encoding="utf-8")
        whole = make_whole(texts_of(lines)) if make_whole else "\n".join(texts_of(lines))
        (out / f"{fallback}.txt").write_text(whole, encoding="utf-8")
        docs.append({"doc_id": doc_id, "image": str(image), "gt": "", "n_columns": n_columns,
                     "is_talmud": True, "genre": "talmud/halakha"})
    spec = root / "spec.json"
    spec.write_text(json.dumps({"docs": docs}, ensure_ascii=False), encoding="utf-8")
    return {"spec": spec, "outputs": outputs, "details": root / "details"}


def patch_vlm(monkeypatch: pytest.MonkeyPatch, replies: List[str], calls: List[Dict[str, Any]]) -> None:
    """Replace every binding of the VLM call and the LM Studio health check.

    :param monkeypatch: Pytest monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    :param replies: Canned replies, consumed in call order (an empty list makes any call fail the test).
    :type replies: List[str]
    :param calls: Receives one record per call (``max_tokens`` and the crop's size).
    :type calls: List[Dict[str, Any]]
    """
    async def fake_transcribe(model_id: str, image_path: str, prompt: str, max_tokens: int = 8192,
                              **_: Any) -> Optional[str]:
        """Canned VLM reply.

        :param model_id: Model id.
        :type model_id: str
        :param image_path: Crop path.
        :type image_path: str
        :param prompt: Prompt.
        :type prompt: str
        :param max_tokens: Token cap.
        :type max_tokens: int
        :return: The next canned reply.
        :rtype: Optional[str]
        """
        assert replies, "unexpected VLM call"
        with Image.open(image_path) as im:
            calls.append({"max_tokens": max_tokens, "size": im.size, "prompt": prompt})
        return replies.pop(0)

    async def fake_health(*_: Any, **__: Any) -> List[str]:
        """Served models.

        :return: The model the test runs.
        :rtype: List[str]
        """
        return ["test-vlm"]

    for module in (driver, linewise, lms_transcriber):
        monkeypatch.setattr(module, "transcribe_with_lm_studio", fake_transcribe)
    monkeypatch.setattr(driver, "check_lm_studio_health", fake_health)
    monkeypatch.setattr(lms_transcriber, "check_lm_studio_health", fake_health)


def cli(paths: Dict[str, Path], *extra: str) -> List[str]:
    """Driver arguments for the temporary benchmark.

    :param paths: From :func:`make_benchmark`.
    :type paths: Dict[str, Path]
    :param extra: Further arguments.
    :type extra: str
    :return: argv.
    :rtype: List[str]
    """
    return ["--benchmark", "religious", "--vlm-model", "test-vlm", "--stem", "test_vlm_decode_policy",
            "--fallback-stem", "test_vlm_page", "--spec", str(paths["spec"]), "--outputs-dir", str(paths["outputs"]),
            "--details-dir", str(paths["details"]), *extra]


def test_driver_dry_run_plans_without_model_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The dry run plans every page, scores the detector against n_columns and never calls the VLM."""
    paths = make_benchmark(tmp_path, "test_vlm_page")
    calls: List[Dict[str, Any]] = []
    patch_vlm(monkeypatch, [], calls)
    report = driver.main(cli(paths, "--dry-run"))
    assert calls == []
    assert report["actions"] == {"columns": 1, "rescue": 1, "keep": 1}
    assert report["reasons"]["whole_page_loop"] == 1 and report["whole_page_labels"]["loop"] == 1
    assert report["detector"]["policy"] == {"tp": 1, "fp": 0, "fn": 0, "tn": 2, "precision": 1.0, "recall": 1.0,
                                            "false_positives": [], "false_negatives": []}
    assert report["reads"]["base"] == 4 and report["reads"]["worst_case_with_bands"] > 4
    saved = paths["details"] / "religious" / "test_vlm_decode_policy" / "dry_run__test_vlm_page.json"
    assert json.loads(saved.read_text(encoding="utf-8"))["n_pages"] == 3
    assert not list(paths["outputs"].glob("*/test_vlm_decode_policy.txt"))


def test_driver_real_run_reads_crops_falls_back_to_bands_and_resumes(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """Real mode reads the planned crops, re-reads a failed crop as bands, writes stems and traces, and resumes."""
    paths = make_benchmark(tmp_path, "test_vlm_page")
    cfg = driver.benchmark_config("religious", paths["spec"], paths["outputs"])
    docs = json.loads(paths["spec"].read_text(encoding="utf-8"))["docs"]
    plans = {}
    for doc in docs:
        page = driver.load_page(cfg, doc, "test_vlm_page")
        plans[doc["doc_id"]] = dp.plan_page(page.kraken, page.whole, TH, ink_gap_x=page.ink_gap)
    col0, col1 = plans["DOC_TWO"].units
    top, bottom = plans["DOC_LOOP"].units
    n_bands = driver.bands_per_unit(plans["DOC_TWO"])[1]
    size = -(-len(col1.kraken_texts) // n_bands)
    chunks = [col1.kraken_texts[i * size:(i + 1) * size] for i in range(n_bands)]
    replies = (["\n".join(col0.kraken_texts), ""] + ["\n".join(c) for c in chunks]
               + ["\n".join(top.kraken_texts), "\n".join(bottom.kraken_texts)])
    calls: List[Dict[str, Any]] = []
    patch_vlm(monkeypatch, replies, calls)
    summary = driver.main(cli(paths))
    assert replies == [] and summary["written"] == 3
    assert [c["max_tokens"] for c in calls] == [2500, 2500] + [linewise.BAND_MAX_TOKENS] * n_bands + [2500, 2500]
    assert calls[0]["size"] == (col0.crop[2] - col0.crop[0], col0.crop[3] - col0.crop[1])
    out = paths["outputs"]
    two = (out / "DOC_TWO" / "test_vlm_decode_policy.txt").read_text(encoding="utf-8")
    assert two == "\n".join(col0.kraken_texts + col1.kraken_texts)
    loop = (out / "DOC_LOOP" / "test_vlm_decode_policy.txt").read_text(encoding="utf-8")
    assert loop == "\n".join(texts_of(single_column_lines(seed=1500)))
    keep = (out / "DOC_KEEP" / "test_vlm_decode_policy.txt").read_text(encoding="utf-8")
    assert keep == (out / "DOC_KEEP" / "test_vlm_page.txt").read_text(encoding="utf-8")
    traces = paths["details"] / "religious" / "test_vlm_decode_policy"
    two_trace = json.loads((traces / "DOC_TWO.json").read_text(encoding="utf-8"))
    assert used(two_trace) == ["crop", "bands"] and two_trace["calls"]["calls"] == 2 + n_bands
    assert len(two_trace["units"][1]["band_details"]) == n_bands and two_trace["plan"]["action"] == "columns"
    keep_trace = json.loads((traces / "DOC_KEEP.json").read_text(encoding="utf-8"))
    assert keep_trace["calls"]["calls"] == 0 and keep_trace["final_source"] == dp.FINAL_WHOLE_PAGE
    again = driver.main(cli(paths))                                  # resumable: every page is done
    assert again["written"] == 0 and len(calls) == 4 + n_bands


def test_driver_force_redecodes_listed_pages_with_the_weak_kraken_limit(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """``--force`` re-decodes a page whose stem exists and overwrites output and trace; nothing else is touched.

    The page is a weak-Kraken page, so the driver's band decision uses the relaxed limit too: the two crop reads
    are accepted and no band is read.
    """
    lines, whole, crops = weak_kraken_page("KTIV_990051100930205171_FL202443168")
    paths = make_benchmark(tmp_path, "test_vlm_page", extra={"DOC_WEAK": (lines, 2, lambda _: whole)})
    out = paths["outputs"] / "DOC_WEAK" / "test_vlm_decode_policy.txt"
    traces = paths["details"] / "religious" / "test_vlm_decode_policy"
    traces.mkdir(parents=True)
    out.write_text("stale assembly", encoding="utf-8")                # an earlier policy's output and trace
    (traces / "DOC_WEAK.json").write_text("{}", encoding="utf-8")
    calls: List[Dict[str, Any]] = []
    patch_vlm(monkeypatch, [], calls)
    assert driver.main(cli(paths, "--doc-ids", "DOC_WEAK"))["written"] == 0 and calls == []    # resume skips it
    replies = list(crops)
    patch_vlm(monkeypatch, replies, calls)
    summary = driver.main(cli(paths, "--doc-ids", "DOC_WEAK", "--force"))
    assert summary["written"] == 1 and replies == [] and len(calls) == 2
    assert out.read_text(encoding="utf-8") == "\n".join(crops)
    trace = json.loads((traces / "DOC_WEAK.json").read_text(encoding="utf-8"))
    assert used(trace) == ["crop", "crop"] and trace["calls"]["calls"] == 2           # no band reads
    assert trace["weak_kraken_relaxation"] is True and trace["final_letters"] == 506 + 557
    assert not list(paths["outputs"].glob("DOC_[TLK]*/test_vlm_decode_policy.txt"))  # unlisted pages untouched
    with pytest.raises(SystemExit):
        driver.parse_args(cli(paths, "--force"))                                     # --force needs --doc-ids


def test_check_stems_and_select_docs() -> None:
    """Stems the scorers would skip or that equal the fallback are refused; doc selection keeps spec order."""
    for bad in ("kraken_x", "x_flat", "consensus", "a/b"):
        with pytest.raises(ValueError):
            driver.check_stems(bad, "fallback")
    with pytest.raises(ValueError):
        driver.check_stems("same", "same")
    driver.check_stems("qwen3_vl_8b_heb_v21b_decode_policy", "qwen3_vl_8b_heb_v21b_step1200")
    docs = [{"doc_id": "a"}, {"doc_id": "b"}, {"doc_id": "c"}]
    assert [d["doc_id"] for d in driver.select_docs(docs, "c,a")] == ["a", "c"]
    assert [d["doc_id"] for d in driver.select_docs(docs, limit=2)] == ["a", "b"]
    with pytest.raises(ValueError):
        driver.select_docs(docs, "zz")
