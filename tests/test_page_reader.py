# File name: test_page_reader.py
# Date: 10/25/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Tests of the production page reader: prompt selection, loop handling, the blank gate, and ``read_page`` with fakes."""
import asyncio
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.datasets.evaluations import arabic_script as ar
from src.datasets.evaluations.helper_eval_scripts import loop_guard_outputs as lg
from src.finetuning.qwen_hebrew import prompts as P
from src.models.ocr import page_reader as pr

ARABIC = "\n".join([
    "بسم الله الرحمن الرحيم",
    "هذا ما استاجر افرايم بن عالي الاسرائيلي من ابي الفتوح",
    "الى الشيخ ابو سعد اطال الله بقاه وادام عزه",
    "وصل كتاب مولاي الشيخ الجليل وفهمت ما ذكره من امر البضاعه المحموله",
    "ثم ان التاجر المذكور سافر الى الاسكندريه في المركب وباع جميع ما معه من القماش والحرير والفلفل ورجع سالما",
    "وقد قبضت منه عشرين دينارا ذهبا جيدا وزنا وعددا بحضرة الشهود",
])
HEBREW = "שלום עליכם אחי היקר קיבלתי את מכתבך ושמחתי מאד לשמוע ממך ומשלומך ומשלום כל בני ביתך"
AR_PHRASE = "وصلواته على سيدنا محمد النبي "
HE_PHRASE = "ברוך אתה יי אלהינו מלך העולם "


def _meta(text: Optional[str], finish: Optional[str] = "stop", tokens: int = 40) -> Dict[str, Any]:
    """A ``completion_meta``-shaped result.

    :param text: Answer text.
    :type text: Optional[str]
    :param finish: Finish reason.
    :type finish: Optional[str]
    :param tokens: Completion tokens.
    :type tokens: int
    :return: Meta dict.
    :rtype: Dict[str, Any]
    """
    return {"text": text, "finish_reason": finish, "prompt_tokens": 6500, "completion_tokens": tokens, "total_tokens": 6500 + tokens}


def _loop(prefix: str, phrase: str, copies: int = 25) -> str:
    """A read that collapses into repeating ``phrase`` after ``prefix``."""
    return prefix + "\n" + phrase * copies


# --- prompt selection ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("hint", [None, "", "hebrew", "arabic", "Hebrew", "mixed", "unknown", "judaeo-arabic"])
def test_the_production_prompt_never_depends_on_the_script(hint: Optional[str]) -> None:
    assert pr.select_prompt(hint) == (pr.PROMPT_SCRIPT_NEUTRAL, P.SCRIPT_NEUTRAL_PROMPT)


def test_the_script_neutral_prompt_names_no_script_and_keeps_every_instruction() -> None:
    prompt = P.SCRIPT_NEUTRAL_PROMPT
    assert P.SCRIPT_NEUTRAL_CLAUSE in prompt and "Hebrew" not in prompt and "Arabic" not in prompt and "Aramaic" not in prompt
    for instruction in ("Transcribe the text exactly as written, in reading order", "Mark unclear characters with [?]",
                        "Where text is lost or illegible due to damage, write [...]", "Do NOT correct, restore, or complete from memory",
                        "Return ONLY the transcription"):
        assert instruction in prompt


def test_the_three_fragment_prompts_differ_in_the_script_clause_only() -> None:
    assert P.SCRIPT_NEUTRAL_PROMPT == P.FRAGMENT_TRANSCRIBE_PROMPT.replace(P.HEBREW_SCRIPT_CLAUSE, P.SCRIPT_NEUTRAL_CLAUSE)
    assert P.SCRIPT_NEUTRAL_PROMPT == P.ARABIC_FRAGMENT_PROMPT.replace(P.ARABIC_SCRIPT_CLAUSE, P.SCRIPT_NEUTRAL_CLAUSE)
    assert P.ARABIC_FRAGMENT_PROMPT == P.FRAGMENT_TRANSCRIBE_PROMPT.replace(P.HEBREW_SCRIPT_CLAUSE, P.ARABIC_SCRIPT_CLAUSE)
    assert len({P.SCRIPT_NEUTRAL_PROMPT, P.FRAGMENT_TRANSCRIBE_PROMPT, P.ARABIC_FRAGMENT_PROMPT}) == 3
    assert (P.SCRIPT_NEUTRAL_PROMPT.splitlines()[1:] == P.FRAGMENT_TRANSCRIBE_PROMPT.splitlines()[1:]
            == P.ARABIC_FRAGMENT_PROMPT.splitlines()[1:])
    assert P.LEGACY_SCRIPT_PROMPTS == {"hebrew": P.FRAGMENT_TRANSCRIBE_PROMPT, "arabic": P.ARABIC_FRAGMENT_PROMPT}


def test_the_legacy_prompts_come_only_on_request() -> None:
    assert pr.select_prompt("arabic", legacy=True) == ("legacy_arabic", P.ARABIC_FRAGMENT_PROMPT)
    assert pr.select_prompt("Hebrew", legacy=True) == ("legacy_hebrew", P.FRAGMENT_TRANSCRIBE_PROMPT)
    for hint in (None, "", "mixed", "unknown"):
        with pytest.raises(ValueError):
            pr.select_prompt(hint, legacy=True)


# --- loop handling ------------------------------------------------------------------------------------------------------

def test_script_letters_count_both_scripts_in_order() -> None:
    assert pr.script_letters("abc שלום 12 سلام! ") == "שלוםسلام"
    assert pr.script_letters(ARABIC) == "".join(ar.SCRIPT_LETTER.findall(ar.normalize_arabic(ARABIC)))
    assert pr.script_letters("[...] [?] 1 2 3") == ""


def test_a_finished_clean_read_is_stored_unchanged() -> None:
    for text in (ARABIC, HEBREW, "", "[...]"):
        read = pr.settle_read(text, "stop")
        assert read == {"text": text, "raw_text": None, "looped": False, "cut_at": None, "letters_removed": 0,
                        "repeat_share": round(ar.repeat_share(text), 4)}
    assert pr.settle_read(None, None) == {"text": None, "raw_text": None, "looped": False, "cut_at": None, "letters_removed": 0,
                                          "repeat_share": 0.0}


def test_a_length_finish_is_a_loop_and_is_cut_where_the_loop_starts() -> None:
    text = _loop(ARABIC, AR_PHRASE)
    read = pr.settle_read(text, "length")
    assert read["looped"] and read["raw_text"] == text and read["cut_at"] is not None
    assert read["text"] == text[:read["cut_at"]].rstrip()
    assert pr.script_letters(read["text"]) == pr.script_letters(ARABIC) + pr.script_letters(AR_PHRASE)   # one copy is real text
    assert read["letters_removed"] == 24 * len(pr.script_letters(AR_PHRASE)) and read["repeat_share"] > 0.5
    assert ar.repeat_share(read["text"]) < 0.1


def test_a_hebrew_loop_is_cut_too() -> None:
    text = _loop(HEBREW, HE_PHRASE)
    assert lg.guard_text(text, gate=None) == (text, None, False)              # the benchmark guard counts Arabic letters only
    read = pr.settle_read(text, "length")
    assert read["looped"] and read["cut_at"] is not None
    assert pr.script_letters(read["text"]) == pr.script_letters(HEBREW) + pr.script_letters(HE_PHRASE)
    assert read["letters_removed"] == 24 * len(pr.script_letters(HE_PHRASE))


def test_a_length_finish_without_a_loop_keeps_its_text_but_is_marked() -> None:
    read = pr.settle_read(ARABIC, "length")
    assert read["looped"] and read["text"] == ARABIC and read["raw_text"] is None and read["cut_at"] is None
    assert read["letters_removed"] == 0


def test_a_stop_finish_that_loops_is_cut_through_the_scorer_gate() -> None:
    text = _loop(ARABIC, AR_PHRASE)
    read = pr.settle_read(text, "stop")
    assert read["looped"] and read["cut_at"] is not None and read["raw_text"] == text
    assert pr.script_letters(read["text"]) == pr.script_letters(ARABIC) + pr.script_letters(AR_PHRASE)
    loose = pr.ReaderSettings(loop_gate=0.99)                                   # a gate the loop does not pass: left alone
    assert pr.settle_read(text, "stop", loose) == {"text": text, "raw_text": None, "looped": False, "cut_at": None,
                                                   "letters_removed": 0, "repeat_share": round(ar.repeat_share(text), 4)}
    assert pr.settle_read(text, "length", loose)["cut_at"] is not None          # a length finish is tried whatever the gate


def test_choose_read_prefers_a_finished_generation_then_more_letters() -> None:
    first = dict(pr.settle_read(_loop(ARABIC, AR_PHRASE), "length"), finish_reason="length")
    stopped = dict(pr.settle_read(HEBREW, "stop"), finish_reason="stop")
    assert pr.choose_read(first, stopped) == stopped                           # the retry finished
    assert pr.choose_read(stopped, first) == stopped                           # the first read finished
    shorter = dict(pr.settle_read(_loop(HEBREW, HE_PHRASE), "length"), finish_reason="length")
    assert pr.choose_read(first, shorter) == first                             # both cut: the one with more letters
    assert pr.choose_read(shorter, first) == first
    assert pr.choose_read(first, dict(first)) == first                         # a tie goes to the retry (equal here)
    assert pr.choose_read(stopped, dict(stopped, text=HEBREW + " עוד")) == dict(stopped, text=HEBREW + " עוד")
    for nothing in ("", "[...]", "[?] [...]", "שלום"):                           # a retry that stops with (almost) nothing
        empty = dict(pr.settle_read(nothing, "stop"), finish_reason="stop")
        assert pr.choose_read(first, empty) == first and pr.choose_read(empty, first) == first


# --- blank gate ---------------------------------------------------------------------------------------------------------

def test_gate_decision_skips_only_the_coverage_tools_blank_class() -> None:
    ok = {"mask_ok": True, "paper_grey": 150.0, "ink_permille": 2.0, "loose_ink_permille": 3.0}
    assert pr.gate_decision(ok) == (True, "blank_or_faint")
    assert pr.gate_decision(dict(ok, ink_permille=7.9)) == (True, "blank_or_faint")
    assert pr.gate_decision(dict(ok, ink_permille=8.0)) == (False, "text")
    assert pr.gate_decision(dict(ok, loose_ink_permille=16.0)) == (False, "unknown")      # ink outside the fragment: read it
    assert pr.gate_decision(dict(ok, paper_grey=30.0)) == (False, "unknown")              # dark "paper": read it
    assert pr.gate_decision({"mask_ok": False}) == (False, "unknown")                     # mask failed: read it
    strict = pr.ReaderSettings(image=pr.abc.ImageSettings(blank_max_permille=1.0))
    assert pr.gate_decision(ok, strict) == (False, "text")


def _fragment_image(path: Path, text: bool, backing=(30, 70, 160), paper=(200, 180, 140), seed: int = 0) -> None:
    """A synthetic photograph: a paper rectangle on a blue backing, with or without rows of dark strokes.

    :param path: Output file.
    :type path: Path
    :param text: Draw writing-like strokes.
    :type text: bool
    :param backing: Backing colour.
    :param paper: Paper colour.
    :param seed: Random seed.
    :type seed: int
    """
    rng = random.Random(seed)
    image = Image.new("RGB", (600, 800), backing)
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 100, 500, 700), fill=paper)
    if text:
        for row in range(150, 660, 40):
            x = 140
            while x < 460:
                width, height = rng.randint(6, 22), rng.randint(4, 9)
                draw.rectangle((x, row, x + width, row + height), fill=(60, 40, 30))
                x += width + rng.randint(4, 10)
    pixels = np.asarray(image).astype(np.int16)
    noise = np.random.default_rng(seed).integers(-4, 5, size=pixels.shape)
    Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8)).save(path)


def test_measure_ink_gates_a_blank_page_and_passes_a_written_one(tmp_path: Path) -> None:
    _fragment_image(tmp_path / "blank.png", text=False)
    _fragment_image(tmp_path / "text.png", text=True)
    blank, text = pr.measure_ink(str(tmp_path / "blank.png")), pr.measure_ink(str(tmp_path / "text.png"))
    assert blank["mask_ok"] and text["mask_ok"] and "loose_ink_permille" in blank
    assert blank["ink_permille"] < 1.0 < pr.DEFAULT_SETTINGS.image.blank_max_permille < text["ink_permille"]
    assert pr.gate_decision(blank) == (True, "blank_or_faint") and pr.gate_decision(text) == (False, "text")


# --- read_page ------------------------------------------------------------------------------------------------------------

class FakeModel:
    """A fake ``transcribe_with_lm_studio_meta`` answering from a queue and recording its calls."""

    def __init__(self, answers: Sequence[Optional[Dict[str, Any]]]) -> None:
        """
        :param answers: Meta (or None) per call, in order.
        :type answers: Sequence[Optional[Dict[str, Any]]]
        """
        self.answers = list(answers)
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, model: str, image_path: str, prompt: str, temperature: float, max_tokens: int,
                       **kwargs: Any) -> Optional[Dict[str, Any]]:
        """Answer one request.

        :return: The next queued meta.
        :rtype: Optional[Dict[str, Any]]
        """
        self.calls.append({"model": model, "image": image_path, "prompt": prompt, "temperature": temperature, "max_tokens": max_tokens,
                           **kwargs})
        return self.answers.pop(0)


INK_TEXT = {"mask_ok": True, "inner_share": 0.4, "paper_grey": 150.0, "ink_permille": 90.0, "ink_components": 300,
            "line_components": 12, "loose_ink_permille": 91.0}
INK_BLANK = dict(INK_TEXT, ink_permille=1.2, loose_ink_permille=2.0, ink_components=3, line_components=0)


def _read(model: FakeModel, ink: Dict[str, Any] = INK_TEXT, settings: pr.ReaderSettings = pr.DEFAULT_SETTINGS,
          **kwargs: Any) -> Dict[str, Any]:
    """``read_page`` with the fake model and a fixed ink measure."""
    return asyncio.run(pr.read_page("qwen/m", "/images/page.jpg", settings, transcribe=model, measure=lambda p, s: dict(ink), **kwargs))


def test_a_blank_page_is_not_sent_to_the_model() -> None:
    model = FakeModel([])
    read = _read(model, INK_BLANK)
    assert model.calls == []
    assert {k: read[k] for k in ("text", "reason", "blank", "looped", "attempts", "retried", "finish_reason", "prompt", "max_tokens",
                                 "model")} == {
        "text": "", "reason": "blank", "blank": True, "looped": False, "attempts": 0, "retried": False, "finish_reason": None,
        "prompt": "script_neutral", "max_tokens": 1200, "model": "qwen/m"}
    assert read["ink"] == dict(INK_BLANK, **{"class": "blank_or_faint"}) and isinstance(read["secs"], float)
    assert set(read) == set(pr.READ_FIELDS)


def test_a_clean_read_is_one_request_with_the_neutral_prompt_and_the_budget() -> None:
    model = FakeModel([_meta(ARABIC)])
    read = _read(model, base_url="http://fake/v1")
    assert len(model.calls) == 1
    assert model.calls[0] == {"model": "qwen/m", "image": "/images/page.jpg", "prompt": P.SCRIPT_NEUTRAL_PROMPT, "temperature": 0.1,
                              "max_tokens": 1200, "base_url": "http://fake/v1"}
    assert read["text"] == ARABIC and read["reason"] is None and not read["blank"] and not read["looped"]
    assert (read["finish_reason"], read["prompt_tokens"], read["completion_tokens"], read["total_tokens"]) == ("stop", 6500, 40, 6540)
    assert (read["attempts"], read["retried"], read["temperature"], read["ink"]["class"]) == (1, False, 0.1, "text")
    assert set(read) == set(pr.READ_FIELDS)


def test_a_length_finish_is_retried_at_temperature_zero_and_the_finished_retry_is_kept() -> None:
    loop = _loop(ARABIC, AR_PHRASE)
    model = FakeModel([_meta(loop, "length", 1200), _meta(HEBREW, "stop", 60)])
    read = _read(model)
    assert [(c["temperature"], c["max_tokens"]) for c in model.calls] == [(0.1, 1200), (0.0, 1200)]
    assert read["text"] == HEBREW and read["finish_reason"] == "stop" and not read["looped"]
    assert (read["attempts"], read["retried"], read["temperature"], read["completion_tokens"]) == (2, True, 0.0, 60)


def test_two_cut_reads_keep_the_one_with_more_letters_and_stay_marked() -> None:
    long_loop, short_loop = _loop(ARABIC, AR_PHRASE), _loop(HEBREW, HE_PHRASE)
    read = _read(FakeModel([_meta(long_loop, "length", 1200), _meta(short_loop, "length", 1200)]))
    assert read["looped"] and read["finish_reason"] == "length" and read["raw_text"] == long_loop.strip()   # stored as completion_meta strips it
    assert pr.script_letters(read["text"]) == pr.script_letters(ARABIC) + pr.script_letters(AR_PHRASE)
    assert (read["attempts"], read["retried"], read["temperature"]) == (2, True, 0.1)
    other = _read(FakeModel([_meta(short_loop, "length", 1200), _meta(long_loop, "length", 1200)]))
    assert other["raw_text"] == long_loop.strip() and other["temperature"] == 0.0


def test_a_failed_retry_keeps_the_guarded_first_read() -> None:
    loop = _loop(ARABIC, AR_PHRASE)
    read = _read(FakeModel([_meta(loop, "length", 1200), None]))
    assert read["looped"] and read["raw_text"] == loop.strip() and read["cut_at"] is not None and read["attempts"] == 2 and read["retried"]


def test_the_retry_can_be_switched_off() -> None:
    loop = _loop(ARABIC, AR_PHRASE)
    model = FakeModel([_meta(loop, "length", 1200)])
    read = _read(model, settings=pr.ReaderSettings(retry_on_length=False))
    assert len(model.calls) == 1 and read["looped"] and read["cut_at"] is not None and (read["attempts"], read["retried"]) == (1, False)


def test_an_empty_generation_is_an_empty_read_not_a_failure() -> None:
    for content in (None, "", "  \n"):
        read = _read(FakeModel([_meta(content, "stop", 1)]))
        assert (read["text"], read["reason"], read["blank"], read["attempts"], read["finish_reason"]) == ("", "empty", False, 1, "stop")
    read = _read(FakeModel([_meta(_loop(ARABIC, AR_PHRASE), "length", 1200), _meta(None, "stop", 1)]))
    assert read["looped"] and read["reason"] is None and pr.script_letters(read["text"]).startswith(pr.script_letters(ARABIC))


def test_a_failed_request_is_recorded_as_failed() -> None:
    read = _read(FakeModel([None]))
    assert (read["text"], read["reason"], read["attempts"], read["finish_reason"], read["looped"]) == (None, "failed", 1, None, False)
    assert read["ink"]["class"] == "text"


def test_the_gate_can_be_switched_off() -> None:
    model = FakeModel([_meta(HEBREW)])
    read = asyncio.run(pr.read_page("qwen/m", "/images/page.jpg", pr.ReaderSettings(blank_gate=False), transcribe=model,
                                    measure=lambda p, s: (_ for _ in ()).throw(AssertionError("measured"))))
    assert len(model.calls) == 1 and read["ink"] is None and read["text"] == HEBREW


def test_the_legacy_prompt_passes_through_on_request_only() -> None:
    model = FakeModel([_meta(ARABIC)])
    read = _read(model, script_hint="arabic", legacy=True)
    assert model.calls[0]["prompt"] == P.ARABIC_FRAGMENT_PROMPT and read["prompt"] == "legacy_arabic"
    model = FakeModel([_meta(ARABIC)])
    read = _read(model, script_hint="arabic")
    assert model.calls[0]["prompt"] == P.SCRIPT_NEUTRAL_PROMPT and read["prompt"] == "script_neutral"


def test_settings_defaults_are_the_ablations_recommendation() -> None:
    s = pr.DEFAULT_SETTINGS
    assert (s.max_tokens, s.temperature, s.retry_on_length, s.retry_temperature, s.blank_gate) == (1200, 0.1, True, 0.0, True)
    assert (s.image.blank_max_permille, s.image.blank_max_loose_permille) == (8.0, 16.0)
    assert (s.run_len, s.window, s.min_repeat, s.loop_gate) == (lg.RUN_LEN, lg.WINDOW, lg.MIN_REPEAT, 0.5)
