"""Tests for the decode policy in the two-reader pipeline (``pipeline_decode_policy`` and ``two_reader_lines``).

Pages are synthetic: Kraken lines of one or two columns of pseudo-random Hebrew, the grounded JSON replies the VLM
would give for the page or a crop (boxes in the crop's own 0-1000 frame), and drawn page images where an ink gap is
needed. An autouse fixture points the raw cache at a temp dir and fails any LM Studio, Kraken, download or
Elasticsearch call a test did not mock, so nothing here touches the services or the production records.
"""
import argparse
import asyncio
import dataclasses
import hashlib
import io
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest
from PIL import Image, ImageDraw

from src.datasets.consensus import decode_policy as dp
from src.datasets.consensus import pipeline_decode_policy as pdp
from src.datasets.consensus import two_reader_lines as trl

HEBREW = "אבגדהוזחטיכלמנסעפצקרשת"
W, H = 1000, 1200
LINE_H, PITCH, TOP = 30, 40, 100
KEY = "MiDRASH_Gen_01@k7.0.3-ktivseg-e10-mv"
MODEL = trl.VLM_MODEL
INDEX = "genizah_merged_v8"
REV = "rev-test"
TH = dp.thresholds_for(trl.VLM_MAX_TOKENS)
PxLine = Tuple[str, List[int]]


# ----------------------------------------------------------------------------- synthetic pages


def random_line(seed: int, n_letters: int = 30) -> str:
    """A line of pseudo-random Hebrew words (distinct per seed, so no 12-letter run repeats across lines).

    :param seed: Random seed.
    :param n_letters: Letters in the line.
    :returns: Line text with exactly ``n_letters`` letters.
    """
    rng = random.Random(seed)
    letters = [rng.choice(HEBREW) for _ in range(n_letters)]
    words, i = [], 0
    while i < len(letters):
        k = rng.randint(3, 6)
        words.append("".join(letters[i:i + k]))
        i += k
    return " ".join(words)


def column(x0: int, x1: int, n: int, seed: int, top: int = TOP) -> List[PxLine]:
    """Lines of one column, one pitch apart, with pixel boxes.

    :param x0: Left edge (px).
    :param x1: Right edge (px).
    :param n: Number of lines.
    :param seed: Seed of the first line's text.
    :param top: y of the first line (px).
    :returns: ``(text, [x0, y0, x1, y1])`` per line, top to bottom.
    """
    return [(random_line(seed + i), [x0, top + i * PITCH, x1, top + i * PITCH + LINE_H]) for i in range(n)]


RIGHT = column(550, 950, 20, seed=100)
LEFT = column(50, 450, 20, seed=500)
TWO_COLUMNS = LEFT + RIGHT          # Kraken serves the left column first
SINGLE = column(100, 900, 20, seed=900)
DENSE = [(random_line(3000 + i), [100, 20 + 11 * i, 900, 28 + 11 * i]) for i in range(100)]   # a long one-column page
BLANK = [(random_line(50 + i, 5), [100, TOP + i * PITCH, 900, TOP + i * PITCH + LINE_H]) for i in range(3)]  # 15 letters


def frags_of(lines: Sequence[PxLine], width: int = W, height: int = H) -> List[Dict[str, Any]]:
    """The pipeline's Kraken fragments of some lines (``run_kraken``'s 0-1000 scaling).

    :param lines: ``(text, pixel box)`` per line.
    :param width: Page width.
    :param height: Page height.
    :returns: ``{"text", "conf", "box"}`` per line.
    """
    return [{"text": t, "conf": 0.9, "box": [1000 * b[0] / width, 1000 * b[1] / height, 1000 * b[2] / width,
                                             1000 * b[3] / height]} for t, b in lines]


def reply_of(lines: Sequence[PxLine], crop: Optional[Sequence[int]] = None, close: bool = True) -> str:
    """A grounded JSON reply reading some lines, boxes in the crop's 0-1000 frame (the page's when ``crop`` is None).

    :param lines: ``(text, pixel box)`` per line.
    :param crop: Pixel crop the reply was read from.
    :param close: Close the array (False: cut inside the last entry, as the token cap does).
    :returns: Reply text.
    """
    x0, y0, x1, y1 = crop or (0, 0, W, H)
    entries = [{"text": t, "bbox_2d": [round(1000 * (b[0] - x0) / (x1 - x0)), round(1000 * (b[1] - y0) / (y1 - y0)),
                                      round(1000 * (b[2] - x0) / (x1 - x0)), round(1000 * (b[3] - y0) / (y1 - y0))]}
               for t, b in lines]
    text = json.dumps(entries, ensure_ascii=False)
    return text if close else text[:-25]


def loop_reply(lines: Sequence[PxLine]) -> str:
    """A looping whole-page reply: three real lines, then the fourth over and over.

    :param lines: The page's lines.
    :returns: Reply text.
    """
    return reply_of(list(lines[:3]) + [lines[3]] * 30)


def parsed_lines(reply: str) -> List[Dict[str, Any]]:
    """Lines the pipeline parses from a whole-page reply.

    :param reply: Reply text.
    :returns: ``{"text", "box"}`` per line.
    """
    return trl.parse_grounded(reply)[1]


def page_jpeg(lines: Sequence[PxLine]) -> bytes:
    """A drawn page: dark line boxes on a light background (a two-column page shows its gutter).

    :param lines: ``(text, pixel box)`` per line.
    :returns: JPEG bytes of a ``W`` x ``H`` page.
    """
    im = Image.new("RGB", (W, H), (235, 225, 205))
    draw = ImageDraw.Draw(im)
    for _, box in lines:
        draw.rectangle(box, fill=(40, 30, 30))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def unit_of(image_path: str) -> str:
    """The unit a request's image stands for: ``<doc>__<image>.jpg`` is the page, ``<doc>__<image>__<unit>.jpg`` a crop.

    :param image_path: Image sent to the model.
    :returns: Unit name (``page`` for the page itself).
    """
    parts = Path(image_path).stem.split("__")
    return parts[2] if len(parts) == 3 else pdp.PAGE_UNIT


class FakeVLM:
    """Stand-in for ``transcribe_with_lm_studio``: a canned reply per unit, read off the image file name."""

    def __init__(self, replies: Dict[str, Optional[str]]) -> None:
        """:param replies: ``unit -> reply`` (None answers like a failed request)."""
        self.replies = replies
        self.calls: List[str] = []
        self.dirs: List[Path] = []

    async def __call__(self, model: str, image_path: str, prompt: str, max_tokens: int = 0) -> Optional[str]:
        """Answer the unit's reply; the image must exist while it is read.

        :param model: LM Studio key.
        :param image_path: Image sent to the model.
        :param prompt: Must be the grounded prompt.
        :param max_tokens: Must be the pipeline's cap.
        :returns: The reply, or None.
        """
        assert prompt == trl.P_GROUNDED and max_tokens == trl.VLM_MAX_TOKENS and Path(image_path).exists()
        unit = unit_of(image_path)
        self.calls.append(unit)
        self.dirs.append(Path(image_path).parent)
        return self.replies[unit]


class FakeReader:
    """Stand-in for the pipeline's page reader in :func:`pdp.decode_page`: canned replies, parsed and mapped."""

    def __init__(self, replies: Dict[str, Optional[str]]) -> None:
        """:param replies: ``unit -> reply`` (None = failed request)."""
        self.replies = replies
        self.calls: List[str] = []

    async def read(self, unit: str, role: str, crop: Optional[Tuple[int, int, int, int]]) -> Optional[pdp.Reply]:
        """Answer like ``PolicyPageReader.read``.

        :param unit: Unit name.
        :param role: Unit role.
        :param crop: Pixel crop (None = the page).
        :returns: The reply, or None.
        """
        self.calls.append(unit)
        raw = self.replies[unit]
        if raw is None:
            return None
        parsed, lines = trl.parse_grounded(raw)
        if crop is not None:
            lines = pdp.crop_lines_to_page(lines, crop, W, H)
        return pdp.Reply(unit, role, crop, raw, parsed, lines)


def units_of(frags: List[Dict[str, Any]], whole: Optional[str] = None, gap: Optional[int] = None,
             th: dp.PolicyThresholds = TH) -> Dict[str, Tuple[int, int, int, int]]:
    """Crops of a page's plan by unit name, with the column halves of every column unit.

    :param frags: Fragments.
    :param whole: Whole-page reply (None: none yet).
    :param gap: Ink gap.
    :param th: Thresholds.
    :returns: ``unit -> pixel crop``.
    """
    plan = pdp.plan_from_frags(frags, W, H, whole, parsed_lines(whole) if whole else None, gap, th)
    units, _, _ = pdp.units_from_frags(frags, W, H)
    crops = {u.name: u.crop for u in plan.units}
    for u in plan.units:
        if u.role == dp.ROLE_COLUMN:
            crops.update({h.name: h.crop for h in pdp.column_halves(plan, u, units, th)})
    return crops


def lines_in(lines: Sequence[PxLine], crop: Sequence[int]) -> List[PxLine]:
    """The lines whose box centre lies inside a crop.

    :param lines: ``(text, pixel box)`` per line.
    :param crop: Pixel crop.
    :returns: Those lines.
    """
    return [(t, b) for t, b in lines if crop[0] <= (b[0] + b[2]) / 2 <= crop[2] and crop[1] <= (b[1] + b[3]) / 2 <= crop[3]]


async def _unexpected_async(*args: Any, **kwargs: Any) -> None:
    """Stand-in for a service a test did not mock.

    :param args: Ignored.
    :param kwargs: Ignored.
    :raises AssertionError: Always.
    """
    raise AssertionError("unexpected Kraken / LM Studio call")


def _unexpected(*args: Any, **kwargs: Any) -> None:
    """Stand-in for a download or Elasticsearch request a test did not mock.

    :param args: Ignored.
    :param kwargs: Ignored.
    :raises AssertionError: Always.
    """
    raise AssertionError("unexpected network call")


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the raw cache at a temp dir and fail every service call a test did not mock.

    :param tmp_path: Pytest temp dir.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setattr(trl, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(trl, "transcribe_with_kraken_lines", _unexpected_async)
    monkeypatch.setattr(trl, "transcribe_with_lm_studio", _unexpected_async)
    monkeypatch.setattr(trl, "download", _unexpected)
    monkeypatch.setattr(trl, "_es_request", _unexpected)


# ----------------------------------------------------------------------------- adapters and plans


def test_units_from_frags_are_the_benchmark_units_of_the_kraken_lines() -> None:
    """Fragments scaled back to pixels give the units ``kraken_units`` builds from the Kraken service's own lines."""
    kraken = {"lines": [{"text": t, "bbox": b, "confidence": 0.9} for t, b in TWO_COLUMNS],
              "image_size": [W, H], "view": {"rotation": 0}}
    units, line_h, size = pdp.units_from_frags(frags_of(TWO_COLUMNS), W, H)
    expected, expected_h, expected_size = dp.kraken_units(kraken)
    assert (line_h, size) == (expected_h, expected_size) == (float(LINE_H), (W, H))
    assert [(u["idx"], u["text"], u["letters"]) for u in units] == [(u["idx"], u["text"], u["letters"]) for u in expected]
    assert all(u["box"] == pytest.approx(e["box"], abs=0.01) for u, e in zip(units, expected))


def test_two_column_page_is_planned_as_column_crops_rightmost_first() -> None:
    """Two Kraken columns with an ink gap: one pixel crop per column, right column first, gutter never crossed."""
    frags = frags_of(TWO_COLUMNS)
    plan = pdp.plan_from_frags(frags, W, H, None, None, 500, TH)
    assert plan.action == dp.ACTION_COLUMNS and plan.reason == "2_columns" and plan.whole_page is None
    right, left = plan.units
    assert (right.name, left.name) == ("col0", "col1") and right.role == dp.ROLE_COLUMN
    assert right.kraken_texts == [t for t, _ in RIGHT] and left.kraken_texts == [t for t, _ in LEFT]
    assert left.crop[2] <= 500 <= right.crop[0] and right.crop == (525, 75, 975, 915)
    assert all(isinstance(v, int) for v in right.crop + left.crop)
    gated = pdp.plan_from_frags(frags, W, H, None, None, None, TH)                   # Kraken splits, no ink gap
    assert gated.action == dp.ACTION_RESCUE and gated.reason == "whole_page_missing" and gated.kraken_columns == 2
    kraken_only = pdp.plan_from_frags(frags, W, H, None, None, None, dataclasses.replace(TH, require_ink_gap=False))
    assert kraken_only.action == dp.ACTION_COLUMNS
    with_read = pdp.plan_from_frags(frags, W, H, reply_of(TWO_COLUMNS), parsed_lines(reply_of(TWO_COLUMNS)), 500, TH)
    assert with_read.action == dp.ACTION_COLUMNS and with_read.whole_page.label == "ok"   # columns whatever the health


def test_single_column_page_keeps_a_healthy_read_and_rescues_a_failed_one() -> None:
    """A healthy JSON read is kept; a looping, capped or empty one is rescued as two overlapping halves."""
    frags = frags_of(SINGLE)
    healthy = reply_of(SINGLE)
    keep = pdp.plan_from_frags(frags, W, H, healthy, parsed_lines(healthy), None, TH)
    assert keep.action == dp.ACTION_KEEP and keep.reason == "whole_page_ok" and keep.units == ()
    capped = reply_of(DENSE, close=False)
    assert len(capped) >= TH.page_token_cap and not capped.endswith(("]", "}"))
    for lines, raw, reason in ((SINGLE, loop_reply(SINGLE), "whole_page_loop"), (DENSE, capped, "whole_page_capped"),
                               (SINGLE, "[]", "whole_page_near_empty")):
        plan = pdp.plan_from_frags(frags_of(lines), W, H, raw, parsed_lines(raw), None, TH)
        assert (plan.action, plan.reason) == (dp.ACTION_RESCUE, reason)
        top, bottom = plan.units
        assert (top.name, bottom.name, top.role) == ("top", "bottom", dp.ROLE_HALF)
        column_crop = plan.columns[0].crop
        assert top.crop[1] == column_crop[1] and bottom.crop[3] == column_crop[3]
        overlap = top.crop[3] - bottom.crop[1]
        assert overlap == pytest.approx(0.12 * (column_crop[3] - column_crop[1]), abs=1)


def test_capped_is_judged_on_the_json_reply_not_on_the_plain_text() -> None:
    """A long closed JSON array is healthy (the plain-text check would call a reply this long capped)."""
    closed = reply_of(DENSE)
    assert len(closed) > 2.0 * TH.page_token_cap
    plan = pdp.plan_from_frags(frags_of(DENSE), W, H, closed, parsed_lines(closed), None, TH)
    assert plan.action == dp.ACTION_KEEP
    assert dp.health_of(closed, [t for t, _ in DENSE], TH.page_token_cap).label == "capped"


# ----------------------------------------------------------------------------- boxes and merging


def test_crop_boxes_map_back_to_the_page_frame() -> None:
    """A page box drawn on a crop (in the crop's 0-1000 frame) maps back to where it is on the page."""
    crop = (525, 75, 975, 915)
    for _, box in RIGHT[:5]:
        [line] = trl.parse_grounded(reply_of([("שלום עליכם", box)], crop))[1]
        [mapped] = pdp.crop_lines_to_page([line], crop, W, H)
        assert mapped["box"] == pytest.approx(frags_of([("x", box)])[0]["box"], abs=1.5)
    assert pdp.crop_box_to_page(crop, W, H) == pytest.approx([525.0, 62.5, 975.0, 762.5])


def test_merge_column_lines_puts_the_right_column_first() -> None:
    """Columns come back rightmost first whatever order they are given in; each column keeps its line order."""
    right = [{"text": t, "box": f["box"]} for (t, _), f in zip(RIGHT, frags_of(RIGHT))]
    left = [{"text": t, "box": f["box"]} for (t, _), f in zip(LEFT, frags_of(LEFT))]
    assert pdp.merge_column_lines([left, right]) == right + left
    assert pdp.merge_column_lines([right, [], left]) == right + left


def test_merge_half_lines_drops_overlap_duplicates_only() -> None:
    """Lines both halves read are kept once (the fuller reading wins); a repeated formula outside the band stays."""
    page = [{"text": t, "box": f["box"]} for (t, _), f in zip(SINGLE, frags_of(SINGLE))]
    cut = dict(page[10], text=page[10]["text"][:12])                     # the top crop's edge cut line 10
    formula = dict(page[19], text=page[2]["text"])                       # line 19 repeats line 2's words
    top, bottom = page[:10] + [cut], page[9:19] + [formula]
    band = pdp.half_overlap_box((75, 75, 925, 545), (75, 445, 925, 915), W, H)
    merged, dropped = pdp.merge_half_lines(top, bottom, band)
    assert dropped == 2
    assert [ln["text"] for ln in merged] == [ln["text"] for ln in page[:19]] + [page[2]["text"]]
    assert merged[10] == page[10]                                        # the bottom's full reading replaced the cut one
    assert pdp.merge_half_lines(top, [], band) == (top, 0)


# ----------------------------------------------------------------------------- decode_page


def run(coro: Any) -> Any:
    """Run a coroutine to completion.

    :param coro: Coroutine.
    :returns: Its result.
    """
    return asyncio.run(coro)


def two_column_replies(gap: int = 500) -> Dict[str, Optional[str]]:
    """Healthy crop replies of the two-column page, keyed by unit.

    :param gap: Ink gap the plan is made with.
    :returns: ``unit -> reply`` (column units and their halves).
    """
    crops = units_of(frags_of(TWO_COLUMNS), gap=gap)
    return {name: reply_of(lines_in(TWO_COLUMNS, crop), crop) for name, crop in crops.items()}


def test_live_column_page_reads_two_crops_and_never_the_whole_page() -> None:
    """Healthy crops: two requests, lines in the page frame, right column first, no whole-page read."""
    reader = FakeReader(two_column_replies())
    gaps: List[str] = []
    result = run(pdp.decode_page(frags_of(TWO_COLUMNS), W, H, TH, reader.read, lambda: gaps.append("x") or 500))
    assert reader.calls == ["col0", "col1"] and gaps == ["x"] and result.calls == 2
    assert [ln["text"] for ln in result.lines] == [t for t, _ in RIGHT + LEFT]
    assert all(ln["box"] == pytest.approx(f["box"], abs=1.5) for ln, f in zip(result.lines, frags_of(RIGHT + LEFT)))
    assert result.parsed and isinstance(result.vlm_raw, list) and len(result.vlm_raw) == 2
    assert result.trace["final_source"] == dp.FINAL_ASSEMBLED and result.trace["whole_page_letters"] is None
    assert [r["unit"] for r in result.trace["requests"]] == ["col0", "col1"]
    json.dumps(result.trace)


def test_failed_crop_is_reread_as_halves_then_takes_the_whole_page_share() -> None:
    """A looping crop is re-read as two halves; when they loop too, the whole page is read and its share is used."""
    replies = two_column_replies()
    left_crop = units_of(frags_of(TWO_COLUMNS), gap=500)["col1"]
    looping = reply_of(LEFT[:3] + [LEFT[3]] * 30, left_crop)
    healed = dict(replies, col1=looping)
    reader = FakeReader(healed)
    result = run(pdp.decode_page(frags_of(TWO_COLUMNS), W, H, TH, reader.read, lambda: 500))
    assert reader.calls == ["col0", "col1", "col1.top", "col1.bottom"]          # the halves heal the column
    assert [u["used"] for u in result.trace["units"]] == ["crop", "halves"]
    assert [ln["text"] for ln in result.lines] == [t for t, _ in RIGHT + LEFT]
    stuck = dict(healed, **{"col1.top": looping, "col1.bottom": looping, "page": reply_of(TWO_COLUMNS)})
    reader = FakeReader(stuck)
    result = run(pdp.decode_page(frags_of(TWO_COLUMNS), W, H, TH, reader.read, lambda: 500))
    assert reader.calls == ["col0", "col1", "col1.top", "col1.bottom", "page"] and result.calls == 5
    assert [u["used"] for u in result.trace["units"]] == ["crop", "share"]
    assert [ln["text"] for ln in result.lines] == [t for t, _ in RIGHT + LEFT]


def test_live_single_column_page_keep_and_rescue() -> None:
    """A healthy whole-page read is the only request; a looping one is rescued by two half reads joined once."""
    frags = frags_of(SINGLE)
    reader = FakeReader({"page": reply_of(SINGLE)})
    kept = run(pdp.decode_page(frags, W, H, TH, reader.read, _unexpected))
    assert reader.calls == ["page"] and kept.plan.action == dp.ACTION_KEEP and kept.vlm_raw == reply_of(SINGLE)
    assert kept.lines == parsed_lines(reply_of(SINGLE)) and kept.trace["final_source"] == dp.FINAL_WHOLE_PAGE
    crops = units_of(frags, loop_reply(SINGLE))
    halves = {name: reply_of(lines_in(SINGLE, crop), crop) for name, crop in crops.items()}
    reader = FakeReader({"page": loop_reply(SINGLE), **halves})
    rescued = run(pdp.decode_page(frags, W, H, TH, reader.read, _unexpected))
    assert reader.calls == ["page", "top", "bottom"] and rescued.plan.reason == "whole_page_loop"
    assert [ln["text"] for ln in rescued.lines] == [t for t, _ in SINGLE]          # overlap lines kept once
    assert rescued.trace["overlap_lines_dropped"] == 2 and rescued.vlm_raw[0] == loop_reply(SINGLE)


def test_a_failed_request_decides_nothing() -> None:
    """LM Studio failing on any request returns None (the page is retried later, nothing is recorded)."""
    reader = FakeReader(dict(two_column_replies(), col1=None))
    assert run(pdp.decode_page(frags_of(TWO_COLUMNS), W, H, TH, reader.read, lambda: 500)) is None


# ----------------------------------------------------------------------------- the safety rules


def test_assembly_shorter_than_a_healthy_whole_page_read_gives_way_to_it() -> None:
    """Kraken found only the top half of each column: healthy crops hold half the text -> the whole-page read stands."""
    found = column(550, 950, 10, seed=100) + column(50, 450, 10, seed=500)        # Kraken: top half of each column
    whole = reply_of(TWO_COLUMNS)
    crops = units_of(frags_of(found), whole, gap=500)
    reader = FakeReader({name: reply_of(lines_in(found, crop), crop) for name, crop in crops.items()})
    cached = pdp.Reply("page", "page", None, whole, True, parsed_lines(whole), cached=True)
    result = run(pdp.decode_page(frags_of(found), W, H, TH, reader.read, lambda: 500, whole=cached))
    assert reader.calls == ["col0", "col1"] and result.calls == 2
    assert result.trace["page_fallback"] == dp.FALLBACK_SHORTER and result.trace["assembled_letters"] == 600
    assert result.lines == parsed_lines(whole) and result.trace["final_letters"] == 1200
    assert result.vlm_raw[0] == whole and result.trace["requests"][0]["cached"] is True


def test_weak_kraken_relaxation_keeps_full_column_reads() -> None:
    """Kraken under half the healthy whole-page read: only a runaway past 1.5x that read over-generates."""
    found = column(550, 950, 4, seed=100) + column(50, 450, 4, seed=500)          # 240 Kraken letters, page 1,200
    whole = reply_of(TWO_COLUMNS)
    crops = units_of(frags_of(found), whole, gap=500)
    replies = {name: reply_of(lines_in(TWO_COLUMNS, crop), crop) for name, crop in crops.items()}
    for unit in ("col0", "col1"):                                  # the crops hold the whole column, not Kraken's part
        x0, _, x1, _ = crops[unit]
        replies[unit] = reply_of([ln for ln in TWO_COLUMNS if x0 <= ln[1][0] <= x1], crops[unit])
    cached = pdp.Reply("page", "page", None, whole, True, parsed_lines(whole), cached=True)
    result = run(pdp.decode_page(frags_of(found), W, H, TH, FakeReader(replies).read, lambda: 500, whole=cached))
    assert result.trace["weak_kraken_relaxation"] is True and result.trace["overgen_runaway_letters"] == 1800.0
    assert [u["used"] for u in result.trace["units"]] == ["crop", "crop"] and result.trace["page_fallback"] is None
    assert len(result.lines) == 40 and result.trace["final_letters"] == 1200
    strict = dataclasses.replace(TH, weak_kraken_ratio=0.0)
    result = run(pdp.decode_page(frags_of(found), W, H, strict, FakeReader(replies).read, lambda: 500, whole=cached))
    assert result.trace["units"][0]["attempts"][0]["health"]["label"] == dp.LABEL_OVERGEN
    assert result.trace["final_source"] == dp.FINAL_WHOLE_PAGE and result.lines == parsed_lines(whole)


# ----------------------------------------------------------------------------- the pipeline's read path


def _page_info(root: Path, lines: Sequence[PxLine], doc: str = "Cambridge_T-S_1") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """A job and the ``stage1`` output for a drawn page, as if downloaded and Kraken-read.

    :param root: Work dir root.
    :param lines: The page's lines.
    :param doc: Document id.
    :returns: ``(job, info)``.
    """
    (root / "images").mkdir(parents=True, exist_ok=True)
    data = page_jpeg(lines)
    path = root / "images" / f"{doc}__0.jpg"
    path.write_bytes(data)
    job = {"doc_id": doc, "image_index": 0, "image_url": f"https://example.org/{doc}.jpg"}
    info = {"doc_id": doc, "image_index": 0, "path": path, "width": W, "height": H,
            "sha256": hashlib.sha256(data).hexdigest(), "download_s": 0.1, "kraken_s": 0.1, "frags": frags_of(lines)}
    return job, info


LEGACY_AI_READ = ["vlm_model", "vlm_revision", "htr_model", "rule_version", "decoded_at", "parsed", "n_lines",
                  "n_agreed", "lines"]
ENVELOPE = ["doc_id", "source_index", "image_index", "image_url", "image_width", "image_height", "image_sha256",
            "ai_read"]


def test_stage2_reads_column_crops_and_keeps_the_record_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``stage2`` with the policy on a drawn two-column page: crop reads in the page frame, trace in the raw cache."""
    job, info = _page_info(tmp_path, TWO_COLUMNS)
    assert dp.central_ink_gap(Image.open(info["path"])) is not None
    vlm = FakeVLM(two_column_replies(dp.central_ink_gap(Image.open(info["path"]))))
    monkeypatch.setattr(trl, "transcribe_with_lm_studio", vlm)
    record = run(trl.stage2(job, info, INDEX, MODEL, REV, htr_cache_key=KEY, policy=TH))
    assert vlm.calls == ["col0", "col1"] and vlm.dirs == [info["path"].parent] * 2      # crops beside the page image
    assert sorted(p.name for p in info["path"].parent.iterdir()) == [info["path"].name]  # crops deleted after
    assert list(record) == ENVELOPE
    ar = record["ai_read"]
    assert list(ar) == LEGACY_AI_READ + ["htr_cache_key", "decode_policy"]
    assert ar["decode_policy"] == {"version": pdp.POLICY_VERSION, "action": "columns", "reason": "2_columns",
                                   "final_source": "assembled", "calls": 2}
    assert [ln["text"] for ln in ar["lines"]] == [t for t, _ in RIGHT + LEFT]        # right column first
    assert ar["lines"][0]["bbox"] == trl._clamp_box(frags_of(RIGHT)[0]["box"])        # page frame
    assert ar["lines"][20]["bbox"] == trl._clamp_box(frags_of(LEFT)[0]["box"])
    assert (ar["n_lines"], ar["n_agreed"], ar["parsed"]) == (40, 40, True)
    entry = json.loads((trl.RAW_DIR / f"{trl.raw_stem(job, MODEL)}.json").read_text())
    assert list(entry) == ["doc_id", "image_index", "image_url", "width", "height", "sha256", "vlm_model", "vlm_raw",
                           "vlm_lines", "parsed", "frags", "htr_model", "decode_policy"]
    assert entry["vlm_raw"] == [vlm.replies["col0"], vlm.replies["col1"]] and entry["htr_model"] == KEY
    trace = entry["decode_policy"]
    assert (trace["action"], trace["mode"], trace["calls"], trace["final_source"]) == ("columns", "live", 2, "assembled")
    assert [r["unit"] for r in trace["requests"]] == ["col0", "col1"] and trace["plan"]["ink_gap_x"] is not None
    assert info["policy"] == "columns" and info["calls"] == 2 and info["policy_fallback"] is None
    # the stamp survives a rematch and a rekraken of the record
    rebuilt = trl.rebuild_record(record, entry, KEY, trl.HTR_MODEL_NAME)
    assert rebuilt["ai_read"]["decode_policy"] == ar["decode_policy"]
    assert [ln["text"] for ln in rebuilt["ai_read"]["lines"]] == [ln["text"] for ln in ar["lines"]]


def test_stage2_without_the_policy_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``policy=None`` is the whole-page read: one request on the page, no trace, no stamp, a string ``vlm_raw``."""
    job, info = _page_info(tmp_path, TWO_COLUMNS)
    vlm = FakeVLM({"page": reply_of(TWO_COLUMNS)})
    monkeypatch.setattr(trl, "transcribe_with_lm_studio", vlm)
    record = run(trl.stage2(job, info, INDEX, MODEL, REV))
    assert vlm.calls == ["page"] and list(record["ai_read"]) == LEGACY_AI_READ and "policy" not in info
    entry = json.loads((trl.RAW_DIR / f"{trl.raw_stem(job, MODEL)}.json").read_text())
    assert "decode_policy" not in entry and entry["vlm_raw"] == reply_of(TWO_COLUMNS)


class FakeKraken:
    """Stand-in for ``transcribe_with_kraken_lines``: the drawn page's lines with pixel boxes."""

    def __init__(self, lines: Sequence[PxLine]) -> None:
        """:param lines: The page's lines."""
        self.lines = lines

    async def __call__(self, model_path: str, image_path: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Answer a ``/transcribe_lines`` response.

        :param model_path: Kraken model.
        :param image_path: Image sent to the service.
        :param timeout: Ignored.
        :returns: The response.
        """
        assert Path(image_path).exists()
        return {"lines": [{"text": t, "bbox": b, "confidence": 0.9} for t, b in self.lines], "image_size": [W, H]}


def _main_run(root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, lines: Sequence[PxLine],
              replies: Dict[str, Optional[str]], rescue_min_kraken_letters: int = 0) -> Tuple[str, FakeVLM]:
    """A normal ``--decode-policy`` run (``main_async``, ``--ids --no-resolve``) over one drawn page.

    :param root: Run directory.
    :param monkeypatch: Pytest monkeypatch fixture.
    :param capsys: Pytest capture fixture.
    :param lines: The page's lines (Kraken reads them all).
    :param replies: ``unit -> reply`` of the fake VLM.
    :param rescue_min_kraken_letters: ``--rescue-min-kraken-letters``.
    :returns: ``(log text, the fake VLM)``.
    """
    source = root / "source.jpg"
    source.write_bytes(page_jpeg(lines))
    jobs = root / "jobs.jsonl"
    jobs.write_text(json.dumps({"doc_id": "Cambridge_T-S_1", "image_index": 0, "image_url": "https://example.org/a.jpg",
                                "local_path": str(source)}) + "\n")
    vlm = FakeVLM(replies)
    monkeypatch.setattr(trl, "transcribe_with_kraken_lines", FakeKraken(lines))
    monkeypatch.setattr(trl, "transcribe_with_lm_studio", vlm)
    args = argparse.Namespace(
        from_consensus=None, ids=str(jobs), limit=0, out=str(root / "ai_reads.jsonl"),
        work_dir=str(root / "images"), source_index=INDEX, rematch=False, restamp=False, rekraken=False,
        kraken_model=None, htr_model_name=None, kraken_cache_suffix=None, all_cache=False, force=False,
        raw_dir=None, site_env=str(root / "missing.env"), vlm_model=MODEL, vlm_revision=REV,
        htr_cache_key=KEY, min_free_gb=0.0, no_resolve=True, keep_images=False, decode_policy=True,
        require_ink_gap=True, revisit=False, rescue_min_kraken_letters=rescue_min_kraken_letters)
    capsys.readouterr()
    run(trl.main_async(args))
    return capsys.readouterr().out, vlm


def test_main_loop_logs_the_action_and_call_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture) -> None:
    """A normal run with ``--decode-policy``: the header names the policy, the page line its action and calls."""
    gap = dp.central_ink_gap(Image.open(io.BytesIO(page_jpeg(TWO_COLUMNS))))
    log, vlm = _main_run(tmp_path, monkeypatch, capsys, TWO_COLUMNS, two_column_replies(gap))
    assert f"htr_cache_key={KEY} decode_policy={pdp.POLICY_VERSION}\n" in log    # no rescue_min note at 0
    assert re.search(r"1/1 Cambridge_T-S_1#0: lines 40 agreed 40 frags 40 parsed=True policy=columns calls=2  \[dl ",
                     log)
    assert vlm.calls == ["col0", "col1"] and list((tmp_path / "images").iterdir()) == []
    record = json.loads((tmp_path / "ai_reads.jsonl").read_text())
    assert record["ai_read"]["decode_policy"]["action"] == "columns"


def test_main_loop_logs_a_rescue_skipped_as_blank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture) -> None:
    """``--rescue-min-kraken-letters``: a looping read on a Kraken-blank page is kept, and the log line says why."""
    log, vlm = _main_run(tmp_path, monkeypatch, capsys, BLANK, {"page": reply_of([BLANK[0]] * 30)}, 100)
    assert "rescue_min_kraken_letters=100" in log and vlm.calls == ["page"]
    assert re.search(r"1/1 Cambridge_T-S_1#0: .* policy=keep\(blank_kraken_15\) calls=1  \[dl ", log)
    stamp = json.loads((tmp_path / "ai_reads.jsonl").read_text())["ai_read"]["decode_policy"]
    assert (stamp["action"], stamp["reason"], stamp["calls"]) == ("keep", "blank_kraken_15", 1)


# ----------------------------------------------------------------------------- --rescue-min-kraken-letters


def test_rescue_min_kraken_letters_keeps_kraken_blank_pages_only() -> None:
    """Under N Kraken letters a would-be rescue is kept (reason ``blank_kraken_<letters>``); nothing else changes."""
    looping = reply_of([BLANK[0]] * 30)

    def plan(lines: Sequence[PxLine], raw: Optional[str], n: int, gap: Optional[int] = None) -> dp.PagePlan:
        """Plan a page.

        :param lines: Its lines.
        :param raw: Its whole-page reply.
        :param n: ``rescue_min_kraken_letters``.
        :param gap: Ink gap.
        :returns: The plan.
        """
        return pdp.plan_from_frags(frags_of(lines), W, H, raw, parsed_lines(raw) if raw else None, gap, TH, n)

    validated = plan(BLANK, looping, 0)
    assert (validated.action, validated.reason) == (dp.ACTION_RESCUE, "whole_page_loop_kraken_blank")
    for n in (16, 100):
        skipped = plan(BLANK, looping, n)
        assert (skipped.action, skipped.reason, skipped.units) == (dp.ACTION_KEEP, "blank_kraken_15", ())
        assert pdp.is_blank_skip(skipped) and skipped.whole_page.label == "loop" and skipped.kraken_letters == 15
        assert pdp.estimated_calls(skipped, True) == (0, 0) and pdp.estimated_calls(skipped, False) == (1, 1)
    assert plan(BLANK, looping, 15).action == dp.ACTION_RESCUE                       # fewer than N letters only
    assert plan(BLANK, None, 100).reason == "blank_kraken_15"                       # not read yet: no rescue planned
    healthy = plan(BLANK, reply_of(BLANK), 100)
    assert (healthy.action, healthy.reason) == (dp.ACTION_KEEP, "whole_page_ok") and not pdp.is_blank_skip(healthy)
    assert plan(SINGLE, loop_reply(SINGLE), 100).action == dp.ACTION_RESCUE          # 600 Kraken letters: rescued
    assert plan(TWO_COLUMNS, reply_of(TWO_COLUMNS), 10_000, gap=500).action == dp.ACTION_COLUMNS   # columns unaffected


def test_live_blank_page_is_read_whole_once() -> None:
    """Live: the whole-page read is the page's only request when its rescue is skipped as blank."""
    looping = reply_of([BLANK[0]] * 30)
    reader = FakeReader({"page": looping, "top": "[]", "bottom": "[]"})
    validated = run(pdp.decode_page(frags_of(BLANK), W, H, TH, reader.read, _unexpected))
    assert reader.calls == ["page", "top", "bottom"] and validated.plan.action == dp.ACTION_RESCUE
    reader = FakeReader({"page": looping})
    kept = run(pdp.decode_page(frags_of(BLANK), W, H, TH, reader.read, _unexpected, rescue_min_kraken_letters=100))
    assert reader.calls == ["page"] and kept.lines == parsed_lines(looping) and kept.vlm_raw == looping
    assert (kept.trace["reason"], kept.trace["rescue_min_kraken_letters"]) == ("blank_kraken_15", 100)


# ----------------------------------------------------------------------------- --revisit


def _recorded(root: Path, doc: str, lines: Sequence[PxLine], whole: str) -> Tuple[str, bytes]:
    """A page as the pipeline recorded it before the policy: raw-cache entry and record (under ``KEY``).

    :param root: Run directory (``raw/`` is the patched raw cache).
    :param doc: Document id.
    :param lines: The page's lines (Kraken read them all).
    :param whole: The cached whole-page reply.
    :returns: ``(record line, the page's JPEG bytes)``.
    """
    data = page_jpeg(lines)
    job = {"doc_id": doc, "image_index": 0, "image_url": f"https://example.org/{doc}.jpg"}
    vlm_lines, frags = parsed_lines(whole), frags_of(lines)
    trl.RAW_DIR.mkdir(parents=True, exist_ok=True)
    (trl.RAW_DIR / f"{trl.raw_stem(job, MODEL)}.json").write_text(json.dumps(dict(
        doc_id=doc, image_index=0, image_url=job["image_url"], width=W, height=H,
        sha256=hashlib.sha256(data).hexdigest(), vlm_model=MODEL, vlm_raw=whole, vlm_lines=vlm_lines,
        parsed=True, frags=frags, htr_model=KEY), ensure_ascii=False))
    ai_read = trl.sidecar(vlm_lines, frags, True, MODEL, REV)
    ai_read["htr_cache_key"] = KEY
    record = {"doc_id": doc, "source_index": INDEX, "image_index": 0, "image_url": job["image_url"],
              "image_width": W, "image_height": H, "image_sha256": hashlib.sha256(data).hexdigest(), "ai_read": ai_read}
    return json.dumps(record, ensure_ascii=False), data


def _revisit_set(root: Path) -> Tuple[Path, Dict[str, bytes]]:
    """Three recorded pages: a two-column page, a looping single column and a healthy single column.

    :param root: Run directory.
    :returns: ``(records file, doc -> JPEG bytes)``.
    """
    pages = {"Cambridge_T-S_1": (TWO_COLUMNS, reply_of(TWO_COLUMNS)),
             "Cambridge_T-S_2": (SINGLE, loop_reply(SINGLE)),
             "Cambridge_T-S_3": (SINGLE, reply_of(SINGLE))}
    out, images, lines = root / "ai_reads.jsonl", {}, []
    for doc, (page, whole) in pages.items():
        line, images[doc] = _recorded(root, doc, page, whole)
        lines.append(line)
    out.write_text("".join(l + "\n" for l in lines))
    return out, images


def _snapshot(root: Path) -> Dict[str, bytes]:
    """Bytes of every file under a directory.

    :param root: Directory.
    :returns: ``relative path -> bytes``.
    """
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_revisit_dry_run_plans_without_a_request_or_a_write(tmp_path: Path) -> None:
    """The dry run plans every recorded page from the raw cache, writes the report and touches nothing else."""
    out, _ = _revisit_set(tmp_path)
    ids = tmp_path / "jobs.jsonl"
    ids.write_text("".join(json.dumps({"doc_id": d, "image_index": 0}) + "\n"
                           for d in ("Cambridge_T-S_1", "Cambridge_T-S_2", "Cambridge_T-S_9")))
    before = _snapshot(tmp_path)
    report_path = tmp_path / "reports" / "dry.json"
    rep = run(trl.revisit(out, tmp_path / "images", trl.RAW_DIR, MODEL, TH, ids=ids, dry_run=True, report=report_path))
    assert _snapshot(tmp_path) == {**before, "reports/dry.json": report_path.read_bytes()}
    assert not (tmp_path / "images").exists()
    assert rep["status"] == {"ok": 2, "not_selected": 1} and rep["planned"] == 2
    assert rep["actions"] == {"columns": 1, "rescue": 1, "skipped_blank": 0, "keep": 0}
    assert rep["kraken_columns"] == {1: 1, 2: 1}
    assert rep["rescue_labels"] == {"loop": 1} and rep["whole_page_labels_on_column_pages"] == {"ok": 1}
    assert rep["revisit_cost"]["downloads"] == 2 and rep["revisit_cost"]["calls_base"] == 4
    assert rep["revisit_cost"]["calls_worst"] == 8
    assert rep["jobs"]["pending"] == 1 and rep["jobs"]["recorded"] == 2
    assert [(p["doc_id"], p["action"]) for p in rep["pages"]] == [("Cambridge_T-S_1", "columns"),
                                                                   ("Cambridge_T-S_2", "rescue")]
    assert json.loads(report_path.read_text())["actions"] == rep["actions"]
    every = run(trl.revisit(out, tmp_path / "images", trl.RAW_DIR, MODEL, TH, dry_run=True,
                            report=tmp_path / "reports" / "all.json"))
    assert every["actions"] == {"columns": 1, "rescue": 1, "skipped_blank": 0, "keep": 1} and "jobs" not in every
    assert trl.revisit_report_path(out, ids).name == "revisit_ai_reads__jobs.json"


def test_revisit_rereads_only_the_pages_the_policy_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                           capsys: pytest.CaptureFixture) -> None:
    """A live revisit re-reads the column and rescue pages, rewrites their entries and records, keeps the rest."""
    out, images = _revisit_set(tmp_path)
    original = out.read_text().splitlines()
    gap = dp.central_ink_gap(Image.open(io.BytesIO(images["Cambridge_T-S_1"])))
    crops = units_of(frags_of(SINGLE), loop_reply(SINGLE))
    replies = {**two_column_replies(gap), **{n: reply_of(lines_in(SINGLE, c), c) for n, c in crops.items()}}
    vlm = FakeVLM(replies)
    downloads: List[str] = []

    def fake_download(url: str) -> bytes:
        """Serve the recorded page's bytes.

        :param url: Image URL.
        :returns: JPEG bytes.
        """
        downloads.append(url)
        return images[url.rsplit("/", 1)[1][:-4]]

    monkeypatch.setattr(trl, "transcribe_with_lm_studio", vlm)
    monkeypatch.setattr(trl, "download", fake_download)
    counts = run(trl.revisit(out, tmp_path / "images", trl.RAW_DIR, MODEL, TH, min_free_gb=0.0))
    assert vlm.calls == ["col0", "col1", "top", "bottom"] and len(downloads) == 2
    assert (counts["read_columns"], counts["read_rescue"], counts["calls"], counts["rebuilt"]) == (1, 1, 4, 2)
    assert list((tmp_path / "images").iterdir()) == []                      # downloads and crops deleted
    rows = out.read_text().splitlines()
    assert rows[2] == original[2]                                           # the healthy page: byte for byte
    two, rescued = json.loads(rows[0]), json.loads(rows[1])
    assert [ln["text"] for ln in two["ai_read"]["lines"]] == [t for t, _ in RIGHT + LEFT]
    assert [ln["text"] for ln in rescued["ai_read"]["lines"]] == [t for t, _ in SINGLE]
    assert rescued["ai_read"]["decode_policy"]["action"] == "rescue" and rescued["ai_read"]["htr_cache_key"] == KEY
    old = json.loads(original[1])
    assert {k: v for k, v in rescued.items() if k != "ai_read"} == {k: v for k, v in old.items() if k != "ai_read"}
    assert (rescued["ai_read"]["vlm_revision"], rescued["ai_read"]["n_agreed"]) == (REV, 20)
    entry = json.loads((trl.RAW_DIR / f"Cambridge_T-S_2__0__{MODEL}.json").read_text())
    assert entry["vlm_raw"] == [loop_reply(SINGLE), replies["top"], replies["bottom"]]
    assert entry["decode_policy"]["mode"] == "revisit" and entry["decode_policy"]["calls"] == 2
    assert entry["decode_policy"]["requests"][0] == {**entry["decode_policy"]["requests"][0], "unit": "page",
                                                     "cached": True}
    assert entry["decode_policy"]["cached_whole_page"]["vlm_lines"] == parsed_lines(loop_reply(SINGLE))
    log = capsys.readouterr().out
    assert "rescue/whole_page_loop calls=2 lines 4 -> 20" in log and "revisit done " in log
    # resumable: a second pass reads nothing and changes nothing ...
    monkeypatch.setattr(trl, "transcribe_with_lm_studio", _unexpected_async)
    monkeypatch.setattr(trl, "download", _unexpected)
    settled = out.read_text()
    counts = run(trl.revisit(out, tmp_path / "images", trl.RAW_DIR, MODEL, TH, min_free_gb=0.0))
    assert out.read_text() == settled and counts["done"] == 2 and counts["rebuilt"] == 0
    # ... and a pass that died after the raw cache but before --out gets its records rebuilt without a request
    out.write_text("\n".join(original) + "\n")
    counts = run(trl.revisit(out, tmp_path / "images", trl.RAW_DIR, MODEL, TH, min_free_gb=0.0))
    assert counts["resumed"] == 2
    resumed = out.read_text().splitlines()
    assert resumed[2] == original[2]
    assert json.loads(resumed[1])["ai_read"]["lines"] == rescued["ai_read"]["lines"]


def test_revisit_reads_a_page_recorded_twice_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two records of one page (the a50 file has two such pairs): one read, both records rebuilt from it."""
    line, data = _recorded(tmp_path, "Cambridge_T-S_2", SINGLE, loop_reply(SINGLE))
    out = tmp_path / "ai_reads.jsonl"
    out.write_text(line + "\n" + line + "\n")
    crops = units_of(frags_of(SINGLE), loop_reply(SINGLE))
    vlm = FakeVLM({n: reply_of(lines_in(SINGLE, c), c) for n, c in crops.items()})
    monkeypatch.setattr(trl, "transcribe_with_lm_studio", vlm)
    monkeypatch.setattr(trl, "download", lambda url: data)
    dry = run(trl.revisit(out, tmp_path / "images", trl.RAW_DIR, MODEL, TH, dry_run=True, report=tmp_path / "r.json"))
    assert dry["status"] == {"ok": 1, "duplicate": 1} and dry["revisit_cost"]["calls_base"] == 2
    counts = run(trl.revisit(out, tmp_path / "images", trl.RAW_DIR, MODEL, TH, min_free_gb=0.0))
    assert vlm.calls == ["top", "bottom"] and counts["rebuilt"] == 2
    first, second = (json.loads(l) for l in out.read_text().splitlines())
    assert first["ai_read"]["lines"] == second["ai_read"]["lines"] and len(first["ai_read"]["lines"]) == 20
    assert first["ai_read"]["decode_policy"] == second["ai_read"]["decode_policy"]


def test_revisit_counts_and_never_reads_rescues_skipped_as_blank(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``rescue_min_kraken_letters`` the report counts blank skips apart, and the live pass leaves them alone."""
    blank_line, _ = _recorded(tmp_path, "Cambridge_T-S_4", BLANK, reply_of([BLANK[0]] * 30))
    loop_line, data = _recorded(tmp_path, "Cambridge_T-S_2", SINGLE, loop_reply(SINGLE))
    out = tmp_path / "ai_reads.jsonl"
    out.write_text(blank_line + "\n" + loop_line + "\n")
    validated = run(trl.revisit(out, tmp_path / "images", trl.RAW_DIR, MODEL, TH, dry_run=True,
                                report=tmp_path / "v.json"))
    assert validated["actions"] == {"columns": 0, "rescue": 2, "skipped_blank": 0, "keep": 0}
    rep = run(trl.revisit(out, tmp_path / "images", trl.RAW_DIR, MODEL, TH, dry_run=True, report=tmp_path / "b.json",
                          rescue_min_kraken_letters=100))
    assert rep["actions"] == {"columns": 0, "rescue": 1, "skipped_blank": 1, "keep": 0}
    assert rep["skipped_blank_labels"] == {"loop": 1} and rep["rescue_min_kraken_letters"] == 100
    assert rep["revisit_cost"]["calls_base"] == 2 and rep["revisit_cost"]["downloads"] == 1
    assert trl.revisit_report_path(out, None, 100).name == "revisit_ai_reads__rescue_min100.json"
    crops = units_of(frags_of(SINGLE), loop_reply(SINGLE))
    vlm = FakeVLM({n: reply_of(lines_in(SINGLE, c), c) for n, c in crops.items()})
    downloads: List[str] = []
    monkeypatch.setattr(trl, "transcribe_with_lm_studio", vlm)
    monkeypatch.setattr(trl, "download", lambda url: downloads.append(url) or data)
    counts = run(trl.revisit(out, tmp_path / "images", trl.RAW_DIR, MODEL, TH, min_free_gb=0.0,
                             rescue_min_kraken_letters=100))
    assert downloads == ["https://example.org/Cambridge_T-S_2.jpg"] and vlm.calls == ["top", "bottom"]
    assert (counts["skipped_blank"], counts["read_rescue"]) == (1, 1)
    assert out.read_text().splitlines()[0] == blank_line                       # the blank page: byte for byte


def test_revisit_skips_a_page_whose_served_image_changed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Served bytes that do not hash to the cached ``sha256`` are never re-read: the record stays as it is."""
    out, _ = _revisit_set(tmp_path)
    before = out.read_text()
    monkeypatch.setattr(trl, "download", lambda url: page_jpeg(SINGLE[:3]))
    counts = run(trl.revisit(out, tmp_path / "images", trl.RAW_DIR, MODEL, TH, min_free_gb=0.0))
    assert counts["sha256"] == 2 and counts["rebuilt"] == 0 and out.read_text() == before


def test_policy_args_error() -> None:
    """The policy options go with reading modes only; ``--dry-run`` / ``--report`` belong to ``--revisit``."""
    def ns(**kw: Any) -> argparse.Namespace:
        """:param kw: Overrides.
        :returns: CLI arguments of a plain normal run unless overridden."""
        return argparse.Namespace(**{"revisit": False, "decode_policy": False, "rekraken": False, "rematch": False,
                                     "restamp": False, "from_consensus": None, "out": __file__, "dry_run": False,
                                     "report": None, **kw})

    assert trl.policy_args_error(ns()) is None
    assert trl.policy_args_error(ns(decode_policy=True, from_consensus="high")) is None
    assert trl.policy_args_error(ns(revisit=True, dry_run=True, report="r.json")) is None
    assert "not with --rematch" in trl.policy_args_error(ns(revisit=True, rematch=True))
    assert "not with --rekraken" in trl.policy_args_error(ns(decode_policy=True, rekraken=True))
    assert "existing --out" in trl.policy_args_error(ns(revisit=True, out="/nonexistent/ai_reads.jsonl"))
    assert trl.policy_args_error(ns(dry_run=True)) == "--dry-run plans a --revisit: give --revisit"
    assert "--revisit --dry-run" in trl.policy_args_error(ns(revisit=True, report="r.json"))
    assert trl.policy_args_error(ns(decode_policy=True, rescue_min_kraken_letters=100)) is None
    assert trl.policy_args_error(ns(revisit=True, rescue_min_kraken_letters=100)) is None
    assert "give --decode-policy or --revisit" in trl.policy_args_error(ns(rescue_min_kraken_letters=100))
    assert trl.policy_args_error(ns(decode_policy=True, rescue_min_kraken_letters=-1)) == (
        "--rescue-min-kraken-letters must be >= 0")
    assert "--revisit matches" in trl.htr_cache_key_error(
        argparse.Namespace(htr_cache_key=KEY, rekraken=False, rematch=False, restamp=False, revisit=True))
