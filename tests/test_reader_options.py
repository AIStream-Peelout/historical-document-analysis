# File name: test_reader_options.py
# Date: 10/29/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests of the page reader's decode options: marker-aware loops, the script filter, the orientation probe, band crops,
and ``read_page`` with each option on (fake transcriber, no model)."""
import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from PIL import Image, ImageDraw

from src.models.ocr import page_reader as pr
from src.models.ocr import reader_options as ro

LINES = ["بسم الله الرحمن الرحيم", "الى الشيخ ابو سعد اطال الله بقاه وادام عزه",
         "وصل كتاب مولاي الشيخ الجليل وفهمت ما ذكره من امر البضاعه", "ثم ان التاجر المذكور سافر الى الاسكندريه في المركب",
         "وقد قبضت منه عشرين دينارا ذهبا جيدا وزنا وعددا بحضرة الشهود"]
TEXT = "\n".join(LINES)
FLOOD = TEXT + "\n" + "[...] " * 120
INK = {"mask_ok": True, "inner_share": 0.5, "ink_permille": 100.0, "paper_grey": 180.0, "loose_ink_permille": 100.0, "class": "text"}


def _meta(text: Optional[str], finish: str = "stop", logprobs: Optional[List[float]] = None) -> Dict[str, Any]:
    """A ``completion_meta``-shaped result.

    :param text: Answer.
    :type text: Optional[str]
    :param finish: Finish reason.
    :type finish: str
    :param logprobs: Token log-probabilities (probes).
    :type logprobs: Optional[List[float]]
    :return: Meta dict.
    :rtype: Dict[str, Any]
    """
    meta = {"text": text, "finish_reason": finish, "prompt_tokens": 6500, "completion_tokens": 50, "total_tokens": 6550}
    if logprobs is not None:
        meta["token_logprobs"] = logprobs
    return meta


def _page(path: Path, size=(400, 600)) -> str:
    """Write a light page with dark strokes and return its path."""
    im = Image.new("RGB", size, (225, 205, 170))
    draw = ImageDraw.Draw(im)
    for y in range(60, size[1] - 60, 40):
        draw.line((40, y, size[0] - 40, y), fill=(40, 30, 20), width=3)
    im.save(path)
    return str(path)


# --- loop units ----------------------------------------------------------------------------------------------------------------

def test_loop_units_collapse_markers_and_keep_offsets() -> None:
    text = "ab [...] c […]d"
    units, offsets = ro.loop_units(text)
    assert units == "ab" + ro.MARKER_UNIT + "c" + ro.MARKER_UNIT + "d"
    assert [text[o] for o in offsets] == ["a", "b", "[", "c", "[", "d"]
    assert ro.marker_count("[...] [. . .] [....] […] [?]") == 4


def test_a_marker_flood_is_a_loop_only_for_the_marker_detector() -> None:
    letters = pr.settle_read(FLOOD, "stop")
    assert not letters["looped"] and letters["cut_at"] is None           # production guard: letters only, no loop
    markers = pr.settle_read_markers(FLOOD, "stop")
    assert markers["looped"] and markers["markers"] == 120 and markers["unit_repeat_share"] < 0.5   # the window rule, not the gate
    assert markers["text"].startswith(TEXT) and markers["text"].count("[...]") <= 1   # cut right after the flood starts
    assert markers["letters_removed"] == 0


def test_a_text_with_a_few_markers_is_not_a_loop() -> None:
    text = TEXT.replace("الشيخ", "[...]")
    read = pr.settle_read_markers(text, "stop")
    assert not read["looped"] and read["text"] == text and read["cut_at"] is None


def test_the_marker_detector_still_cuts_letter_loops() -> None:
    text = TEXT + "\n" + "وصلواته على سيدنا محمد النبي " * 25
    read = pr.settle_read_markers(text, "stop")
    assert read["looped"] and read["cut_at"] is not None and len(read["text"]) < len(text)


# --- script filter --------------------------------------------------------------------------------------------------------------

def test_filter_strips_latin_and_cjk_and_keeps_scripts_digits_punctuation() -> None:
    text = "بسم الله يtasayar الرحمن\nשלום 12 [...] [?] ، ٣٤\n满意 الكتاب"
    out, n, stripped = ro.filter_script(text)
    assert stripped == "tasayar满意" and n == 9
    assert out == "بسم الله ي الرحمن\nשלום 12 [...] [?] ، ٣٤\nالكتاب"


def test_filter_leaves_a_clean_text_unchanged() -> None:
    assert ro.filter_script(TEXT) == (TEXT, 0, "")


# --- orientation ----------------------------------------------------------------------------------------------------------------

def _probe(mean: float, share: float = 0.9, ok: bool = True) -> Dict[str, Any]:
    return {"ok": ok, "mean_logprob": mean, "first_logprob": mean, "script_share": share, "repeat_share": 0.0}


def test_choose_orientation_keeps_upright_when_it_wins_or_wins_within_the_margin() -> None:
    assert ro.choose_orientation({0: _probe(-0.2), 90: _probe(-1.0), 180: _probe(-0.9), 270: _probe(-1.2)}, 0.3) == (0, "upright")
    assert ro.choose_orientation({0: _probe(-0.5), 90: _probe(-1.0), 180: _probe(-0.3), 270: _probe(-1.2)}, 0.3) == (0, "margin")


def test_choose_orientation_turns_a_page_whose_turned_probe_wins_by_the_margin() -> None:
    assert ro.choose_orientation({0: _probe(-1.5), 90: _probe(-1.0), 180: _probe(-0.3), 270: _probe(-1.2)}, 0.3) == (180, "turned")


def test_a_probe_without_script_letters_cannot_win() -> None:
    probes = {0: _probe(-1.5), 90: _probe(-0.1, share=0.1), 180: _probe(-1.4), 270: _probe(-1.6)}
    assert ro.choose_orientation(probes, 0.3) == (0, "margin")          # 90 is out; 180 beats upright by 0.1 only
    assert ro.choose_orientation({r: _probe(-1.0, ok=False) for r in ro.ROTATIONS}, 0.3) == (0, "upright")


def test_probe_features() -> None:
    f = ro.probe_features(_meta("بسم الله abc", logprobs=[-0.1, -0.3, -0.2]), first_k=2)
    assert f["ok"] and f["mean_logprob"] == -0.2 and f["first_logprob"] == -0.2 and f["min_logprob"] == -0.3 and f["tokens"] == 3
    assert f["script_share"] == round(7 / 10, 4)
    assert not ro.probe_features(None)["ok"] and not ro.probe_features(_meta("x"))["ok"]


def test_rotated_view_turns_counter_clockwise_and_reuses_the_file(tmp_path: Path) -> None:
    src = tmp_path / "p.png"
    im = Image.new("RGB", (40, 20), "white")
    im.paste((255, 0, 0), (0, 0, 8, 8))                 # a red block in the top-left corner
    im.save(src)
    assert ro.rotated_view(str(src), 0, tmp_path / "v") == str(src)
    turned = ro.rotated_view(str(src), 90, tmp_path / "v")
    with Image.open(turned) as t:
        assert t.size == (20, 40)
        r, g, b = t.convert("RGB").getpixel((2, 37))     # top-left corner goes bottom-left on a counter-clockwise turn
        assert r > 150 and g < 120
    assert ro.rotated_view(str(src), 90, tmp_path / "v") == turned
    with pytest.raises(ValueError):
        ro.rotated_view(str(src), 45, tmp_path / "v")


# --- band crops -----------------------------------------------------------------------------------------------------------------

def test_band_boxes_cover_the_box_with_the_overlap() -> None:
    boxes = ro.band_boxes((10, 100, 410, 1100), 3, 0.12)
    assert boxes[0][1] == 100 and boxes[-1][3] == 1100 and all(b[0] == 10 and b[2] == 410 for b in boxes)
    for upper, lower in zip(boxes, boxes[1:]):
        assert abs((upper[3] - lower[1]) - 120) <= 1          # 12 % of the 1,000-pixel box in both bands
    assert ro.band_boxes((0, 0, 10, 10), 1) == [(0, 0, 10, 10)]
    with pytest.raises(ValueError):
        ro.band_boxes((0, 0, 0, 10), 2)


def test_band_count_grows_with_the_box_and_is_bounded() -> None:
    assert ro.band_count(1000, 1000) == 2
    assert ro.band_count(4000, 5000) == 3
    assert ro.band_count(8000, 9000) == ro.CROP_MAX_BANDS


def test_merge_bands_drops_the_repeated_overlap_lines() -> None:
    top = "\n".join(LINES[:3])
    bottom = "\n".join(LINES[2:])
    joined, dropped = ro.merge_bands([top, bottom])
    assert joined == TEXT and dropped == 1
    assert ro.merge_bands(["", top]) == (top, 0)


def test_crop_decision_modes() -> None:
    assert ro.crop_decision("off", 10**8, 10, INK, 30, 2) == (False, "off")
    assert ro.crop_decision("always", 10, 1000, INK, 30, 2) == (True, "always")
    assert ro.crop_decision("auto", 31 * 10**6, 1000, INK, 30, 2) == (True, "size")
    assert ro.crop_decision("auto", 10**6, 50, INK, 30, 2) == (True, "density")       # 50 / (100 x 0.5) = 1.0 < 2
    assert ro.crop_decision("auto", 10**6, 500, INK, 30, 2) == (False, "no")
    assert ro.crop_decision("auto", 10**6, 50, None, 30, 2) == (False, "no")
    with pytest.raises(ValueError):
        ro.crop_decision("sometimes", 1, 1, INK, 30, 2)


def test_accept_bands() -> None:
    whole = {"text": TEXT, "looped": False}
    letters = len(pr.script_letters(TEXT))
    assert ro.accept_bands(whole, letters, False, False) == (True, "bands")
    assert ro.accept_bands(whole, letters, False, True) == (False, "failed")
    assert ro.accept_bands(whole, letters, True, False) == (False, "looped")
    assert ro.accept_bands(dict(whole, looped=True), letters, True, False) == (True, "bands")
    assert ro.accept_bands(whole, int(letters * 0.5), False, False) == (False, "shorter")


# --- read_page with options -----------------------------------------------------------------------------------------------------

class FakeModel:
    """Answers by the view's file name (turn / band tags) and records every call."""

    def __init__(self, answers: Dict[str, Any]) -> None:
        self.answers, self.calls = answers, []

    async def __call__(self, model_id: str, image_path: str, prompt: str, temperature: float = 0.1, max_tokens: int = 1200,
                       **kwargs: Any) -> Optional[Dict[str, Any]]:
        self.calls.append({"path": image_path, "temperature": temperature, "max_tokens": max_tokens, "extra": kwargs.get("extra_payload")})
        for tag, answer in self.answers.items():
            if tag in Path(image_path).name:
                return answer(temperature, max_tokens) if callable(answer) else answer
        return self.answers["default"](temperature, max_tokens) if callable(self.answers["default"]) else self.answers["default"]


def _settings(tmp_path: Path, **over: Any) -> pr.ReaderSettings:
    return replace(pr.DEFAULT_SETTINGS, view_dir=str(tmp_path / "views"), **over)


def _ink(_path: str, _settings: pr.ReaderSettings) -> Dict[str, Any]:
    return dict(INK)


def test_marker_loops_are_retried_at_temperature_zero(tmp_path: Path) -> None:
    page = _page(tmp_path / "p.png")
    model = FakeModel({"default": lambda t, m: _meta(FLOOD) if t > 0 else _meta(TEXT)})
    read = asyncio.run(pr.read_page("m", page, _settings(tmp_path, loop_detector="markers"), transcribe=model, measure=_ink))
    assert read["retried"] and read["attempts"] == 2 and read["text"] == TEXT and not read["looped"]
    assert read["first_finish_reason"] == "stop" and read["options"] == "markers" and [c["temperature"] for c in model.calls] == [0.1, 0.0]
    production = asyncio.run(pr.read_page("m", page, _settings(tmp_path), transcribe=FakeModel({"default": _meta(FLOOD)}), measure=_ink))
    assert not production["retried"] and production["text"] == FLOOD.strip() and production["markers"] == 120


def test_the_script_filter_strips_and_retries(tmp_path: Path) -> None:
    page = _page(tmp_path / "p.png")
    leaky = TEXT.replace("الرحمن", "الرحمن 这笔交易的买方")
    model = FakeModel({"default": lambda t, m: _meta(leaky) if t > 0 else _meta(TEXT)})
    read = asyncio.run(pr.read_page("m", page, _settings(tmp_path, script_filter=True), transcribe=model, measure=_ink))
    assert read["retried"] and read["text"] == TEXT and read["stripped_chars"] == 0
    stubborn = FakeModel({"default": _meta(leaky)})
    read = asyncio.run(pr.read_page("m", page, _settings(tmp_path, script_filter=True), transcribe=stubborn, measure=_ink))
    assert read["retried"] and read["text"] == TEXT and read["stripped_chars"] == 7 and read["stripped"] == "这笔交易的买方"


def test_the_orientation_probe_turns_the_page_and_reads_it_turned(tmp_path: Path) -> None:
    page = _page(tmp_path / "p.png")
    upside = "الحمد لله وحده"
    model = FakeModel({"__r180": lambda t, m: _meta(TEXT[:40], logprobs=[-0.1] * 10) if m == 48 else _meta(TEXT),
                       "default": lambda t, m: _meta(upside, logprobs=[-1.5] * 10) if m == 48 else _meta(upside)})
    read = asyncio.run(pr.read_page("m", page, _settings(tmp_path, orientation="probe", orientation_margin=0.3), transcribe=model,
                                    measure=_ink))
    assert read["rotation"] == 180 and read["orientation"]["reason"] == "turned" and read["text"] == TEXT
    probes = [c for c in model.calls if c["max_tokens"] == 48]
    assert len(probes) == 4 and all(c["temperature"] == 0.0 and c["extra"] == {"logprobs": True} for c in probes)
    assert "__r180" in Path(model.calls[-1]["path"]).name and model.calls[-1]["max_tokens"] == 1200
    assert set(read["orientation"]["probes"]) == {"0", "90", "180", "270"}


def test_band_crops_replace_a_short_whole_page_read(tmp_path: Path) -> None:
    page = _page(tmp_path / "p.png", (400, 1200))
    model = FakeModel({"b1of2": _meta("\n".join(LINES[:3])), "b2of2": _meta("\n".join(LINES[2:])), "default": _meta(LINES[0])})
    read = asyncio.run(pr.read_page("m", page, _settings(tmp_path, crops="always"), transcribe=model, measure=_ink))
    crop = read["crop"]
    assert crop["accepted"] and crop["reason"] == "always" and len(crop["bands"]) == 2 and crop["dropped_lines"] == 1
    assert read["text"] == TEXT and read["raw_text"] == LINES[0] and read["attempts"] == 3
    longer = FakeModel({"b1of2": _meta(LINES[0]), "b2of2": _meta(""), "default": _meta(TEXT)})
    read = asyncio.run(pr.read_page("m", page, _settings(tmp_path, crops="always"), transcribe=longer, measure=_ink))
    assert not read["crop"]["accepted"] and read["crop"]["accept_reason"] == "shorter" and read["text"] == TEXT


def test_auto_crops_skip_a_well_read_small_page(tmp_path: Path) -> None:
    page = _page(tmp_path / "p.png")
    model = FakeModel({"default": _meta(TEXT)})
    read = asyncio.run(pr.read_page("m", page, _settings(tmp_path, crops="auto", crop_max_density=0.5), transcribe=model, measure=_ink))
    assert read["crop"] is None and len(model.calls) == 1


def test_default_settings_add_no_requests_and_tag_no_option(tmp_path: Path) -> None:
    page = _page(tmp_path / "p.png")
    model = FakeModel({"default": _meta(TEXT)})
    read = asyncio.run(pr.read_page("m", page, _settings(tmp_path), transcribe=model, measure=_ink))
    assert len(model.calls) == 1 and read["options"] == "" and read["rotation"] == 0 and read["crop"] is None
    assert read["orientation"] is None and read["unit_repeat_share"] is None and set(read) == set(pr.READ_FIELDS)


def test_suspect_read_flags_a_length_finish_a_loop_or_a_marker_flood() -> None:
    assert ro.suspect_read({"first_finish_reason": "length", "finish_reason": "stop", "text": TEXT}) == (True, "length")
    assert ro.suspect_read({"finish_reason": "stop", "looped": True, "text": TEXT}) == (True, "looped")
    assert ro.suspect_read({"finish_reason": "stop", "text": FLOOD}) == (True, "markers")
    assert ro.suspect_read({"finish_reason": "stop", "text": TEXT + " [...]"}) == (False, "fluent")
    assert ro.suspect_read({"finish_reason": "stop", "text": ""}) == (False, "fluent")


def test_gated_orientation_probes_only_a_suspect_read_and_reads_it_turned(tmp_path: Path) -> None:
    page = _page(tmp_path / "p.png")
    upside = "[...] " * 20 + "الحمد"
    model = FakeModel({"__r180": lambda t, m: _meta(TEXT[:40], logprobs=[-0.1] * 10) if m == 48 else _meta(TEXT),
                       "default": lambda t, m: _meta(upside, logprobs=[-1.5] * 10) if m == 48 else _meta(upside)})
    read = asyncio.run(pr.read_page("m", page, _settings(tmp_path, orientation="gated", orientation_margin=0.3), transcribe=model,
                                    measure=_ink))
    assert read["rotation"] == 180 and read["text"] == TEXT and read["options"] == "gated"
    assert read["orientation"]["gate"] == "markers" and read["orientation"]["reason"] == "turned"
    assert [c["max_tokens"] for c in model.calls] == [1200, 48, 48, 48, 48, 1200] and read["attempts"] == 2
    fluent = FakeModel({"default": lambda t, m: _meta(TEXT, logprobs=[-1.5] * 10)})
    read = asyncio.run(pr.read_page("m", page, _settings(tmp_path, orientation="gated"), transcribe=fluent, measure=_ink))
    assert read["rotation"] == 0 and read["orientation"] == {"gate": "fluent"} and len(fluent.calls) == 1


def test_gated_orientation_keeps_the_upright_read_when_the_turned_read_fails(tmp_path: Path) -> None:
    page = _page(tmp_path / "p.png")
    upside = "[...] " * 20 + "الحمد"
    model = FakeModel({"__r90": lambda t, m: _meta(TEXT[:40], logprobs=[-0.1] * 10) if m == 48 else None,
                       "default": lambda t, m: _meta(upside, logprobs=[-1.5] * 10) if m == 48 else _meta(upside)})
    read = asyncio.run(pr.read_page("m", page, _settings(tmp_path, orientation="gated"), transcribe=model, measure=_ink))
    assert read["rotation"] == 0 and read["orientation"]["turned_read_failed"] and read["text"] == upside.strip()
    assert read["attempts"] == 2 and read["reason"] is None


def test_gated_orientation_runs_the_band_crops_first_and_keeps_a_page_the_bands_read(tmp_path: Path) -> None:
    page = _page(tmp_path / "p.png", (400, 1200))
    looped = "\n".join([LINES[0]] + [LINES[1]] * 12)
    model = FakeModel({"__r90": lambda t, m: _meta(TEXT[:40], logprobs=[-0.1] * 10) if m == 48 else _meta(TEXT),
                       "b1of2": _meta("\n".join(LINES[:3])), "b2of2": _meta("\n".join(LINES[2:])),
                       "default": lambda t, m: _meta(looped, logprobs=[-1.5] * 10) if m == 48 else _meta(looped)})
    settings = _settings(tmp_path, orientation="gated", crops="always")
    read = asyncio.run(pr.read_page("m", page, settings, transcribe=model, measure=_ink))
    assert read["rotation"] == 0 and read["crop"]["accepted"] and read["text"] == TEXT
    assert read["orientation"] == {"gate": "fluent"} and not any(c["max_tokens"] == 48 for c in model.calls)


def test_gate_view_judges_the_joined_band_read_when_the_bands_replaced_the_whole_read() -> None:
    whole = {"text": TEXT, "looped": True, "finish_reason": "length", "first_finish_reason": "length"}
    crop = {"accepted": True, "looped": False, "bands": [{"first_finish_reason": "stop"}, {"first_finish_reason": "stop"}]}
    assert pr.gate_view(whole, crop) == {"text": TEXT, "looped": False, "first_finish_reason": "stop"}
    assert pr.gate_view(whole, dict(crop, bands=[{"first_finish_reason": "length"}]))["first_finish_reason"] == "length"
    assert pr.gate_view(whole, dict(crop, accepted=False))["looped"] and pr.gate_view(whole, None)["first_finish_reason"] == "length"
