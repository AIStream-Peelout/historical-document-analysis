# File name: test_loop_guard_outputs.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for the loop guard of cached Arabic-script benchmark reads."""
import json

import pytest

from src.datasets.evaluations import arabic_script as ar
from src.datasets.evaluations.helper_eval_scripts import loop_guard_outputs as lg
from src.datasets.evaluations.helper_eval_scripts import score_arabic_benchmark as sab

REAL = "\n".join([
    "بسم الله الرحمن الرحيم",
    "هذا ما استاجر افرايم بن عالي الاسرائيلي من ابي الفتوح",
    "الى الشيخ ابو سعد اطال الله بقاه وادام عزه",
    "وصل كتاب مولاي الشيخ الجليل وفهمت ما ذكره من امر البضاعه المحموله",
    "ثم ان التاجر المذكور سافر الى الاسكندريه في المركب وباع جميع ما معه من القماش والحرير والفلفل ورجع سالما",
    "وقد قبضت منه عشرين دينارا ذهبا جيدا وزنا وعددا بحضرة الشهود",
    "وكتب في شهر رمضان سنة خمس واربعمايه بمدينة مصر المحروسه",
])
PHRASE = "وصلواته على سيدنا محمد النبي "


def _loop(prefix: str, phrase: str = PHRASE, copies: int = 25) -> str:
    """A read that collapses into repeating ``phrase``.

    :param prefix: Text read before the loop.
    :type prefix: str
    :param phrase: The repeated phrase.
    :type phrase: str
    :param copies: Number of copies.
    :type copies: int
    :return: The looping answer.
    :rtype: str
    """
    return prefix + "\n" + phrase * copies


def test_real_text_has_no_loop_even_without_the_gate():
    assert ar.repeat_share(REAL) < 0.1 and lg.loop_start(ar.arabic_letters(REAL)) is None
    for gate in (lg.LOOP_REPEAT_SHARE, None):
        assert lg.guard_text(REAL, gate=gate) == (REAL, None, False)


def test_a_loop_is_cut_after_its_first_copy():
    text = _loop("بسم الله الرحمن الرحيم\nهذا ما استاجر افرايم بن داود")
    assert ar.repeat_share(text) > sab.LOOP_REPEAT_SHARE
    guarded, cut_at, looped = lg.guard_text(text)
    assert looped and cut_at is not None and guarded == text[:cut_at].rstrip()
    head = ar.arabic_letters("بسم الله الرحمن الرحيم هذا ما استاجر افرايم بن داود")
    assert ar.arabic_letters(guarded) == head + ar.arabic_letters(PHRASE)            # the first copy is real text
    assert text[cut_at:].startswith(PHRASE.strip()[:3])                                   # cut where the second copy begins
    assert ar.repeat_share(guarded) < 0.1


def test_the_letters_parameter_lets_a_hebrew_loop_be_cut():
    hebrew = "שלום עליכם אחי היקר קיבלתי את מכתבך ושמחתי מאד לשמוע ממך ומשלומך ומשלום כל בני ביתך"
    phrase = "ברוך אתה יי אלהינו מלך העולם "
    text = _loop(hebrew, phrase)
    assert ar.repeat_share(text) > sab.LOOP_REPEAT_SHARE
    assert lg.guard_text(text) == (text, None, False)                                       # default: Arabic letters only
    letters = lambda t: "".join(ar.SCRIPT_LETTER.findall(ar.normalize_arabic(t)))          # noqa: E731
    guarded, cut_at, looped = lg.guard_text(text, letters=letters)
    assert looped and cut_at is not None and guarded == text[:cut_at].rstrip()
    assert letters(guarded) == letters(hebrew) + letters(phrase)
    assert lg.letter_char_offset(text, len(letters(hebrew)), letters) == len(hebrew) + 1       # the first letter of the phrase
    assert lg.letter_char_offset(text, len(letters(hebrew)) + 1, letters) == len(hebrew) + 2   # its second letter


def test_a_loop_from_the_first_letter_keeps_one_copy():
    guarded, cut_at, looped = lg.guard_text(PHRASE * 30)
    assert looped and ar.arabic_letters(guarded) == ar.arabic_letters(PHRASE)


@pytest.mark.parametrize("text", ["", "   \n", "بسم الله", PHRASE * 2, "ابجد" * 9])
def test_short_texts_are_never_cut(text):
    for gate in (lg.LOOP_REPEAT_SHARE, None):
        assert lg.guard_text(text, gate=gate) == (text, None, False)


def test_a_short_window_of_repeats_inside_real_text_is_left_alone():
    after = "ورجع الغلام من الريف ومعه الكتب والحساب الذي طلبته فاعلم ذلك وتقدم بحفظه الى ان اصل اليك"
    text = REAL + "\n" + PHRASE * 2 + "\n" + after                                          # a formula said twice, then new text
    assert lg.guard_text(text, gate=None) == (text, None, False)
    whole_again = REAL + "\n" + REAL.replace("الله", "اللّه")                               # the whole read twice (marks fold away)
    assert lg.guard_text(whole_again, gate=None)[2] is True


def test_the_gate_leaves_answers_the_scorer_does_not_flag():
    text = REAL + "\n" + REAL[::-1] + "\n" + PHRASE * 4                                  # a short loop at the end of a long read
    assert ar.repeat_share(text) <= sab.LOOP_REPEAT_SHARE
    assert lg.guard_text(text) == (text, None, False)
    guarded, cut_at, looped = lg.guard_text(text, gate=None)
    assert looped and len(ar.arabic_letters(guarded)) < len(ar.arabic_letters(text))


def test_marks_and_presentation_forms_map_the_cut_exactly():
    phrase = "ﻻ" + "مَوْلَايَ الشَّيْخُ الْأَجَلُّ " + "ﻷ" + "ﺑﻮ"                       # lam-alef ligatures, vowels, hamza
    text = _loop("وَصَلَ كِتَابُ مَوْلَايَ", phrase, 30)
    guarded, cut_at, looped = lg.guard_text(text)
    letters = ar.arabic_letters(text)
    k = lg.loop_start(letters)
    assert looped and ar.arabic_letters(guarded) == letters[:k]
    assert ar.arabic_letters(text[:cut_at]) == letters[:k] and ar.arabic_letters(text[cut_at:]) == letters[k:]


def test_a_hebrew_letter_loop_holds_no_scored_letter_and_is_left_alone():
    text = "שלום עליכם ושלום לכל ישראל " * 20
    assert ar.repeat_share(text) > sab.LOOP_REPEAT_SHARE
    assert lg.guard_text(text) == (text, None, False)


def test_letter_char_offset_bounds():
    assert lg.letter_char_offset("  ابت", 0) == 2
    assert lg.letter_char_offset("ا ب ت", 2) == 4
    assert lg.letter_char_offset("ا ب ت", 3) == len("ا ب ت") == lg.letter_char_offset("ا ب ت", 99)


def _rows():
    """Answer rows of a two-document model file, with a failed request and a re-answered image.

    :return: Rows in file order.
    :rtype: list
    """
    base = {"model": "m", "prompt": "trained", "secs": 1.0}
    return [dict(base, doc_id="a", image_index=0, text=_loop("بسم الله الرحمن الرحيم")),
            dict(base, doc_id="a", image_index=1, text=REAL),
            dict(base, doc_id="b", image_index=0, error="timeout"),
            dict(base, doc_id="b", image_index=0, text=PHRASE * 30),
            dict(base, doc_id="b", image_index=0, text=REAL)]                               # the last answer wins in the scorer


def test_guard_rows_keeps_order_keys_and_failed_rows():
    rows = _rows()
    guarded, summary = lg.guard_rows(rows)
    assert len(guarded) == len(rows)
    assert [(r["doc_id"], r["image_index"]) for r in guarded] == [(r["doc_id"], r["image_index"]) for r in rows]
    assert guarded[2] == rows[2] and "loop_guard" not in guarded[2]
    for raw, new in zip(rows, guarded):
        assert set(new) - set(raw) <= {"loop_guard"} and all(new[k] == raw[k] for k in raw if k != "text")
    assert guarded[0]["loop_guard"]["looped"] and guarded[0]["loop_guard"]["cut_at"] > 0
    assert guarded[1]["text"] == REAL and guarded[1]["loop_guard"] == {"cut_at": None, "looped": False,
                                                                        "letters_kept": len(ar.arabic_letters(REAL)),
                                                                        "letters_removed": 0, "repeat_share": round(ar.repeat_share(REAL), 4)}
    assert summary["images_answered"] == 3 and summary["images_cut"] == 1 and summary["docs_cut"] == 1   # b/0 counted once, last answer
    assert summary["inexact_cuts"] == 0 and summary["letters_removed"] == guarded[0]["loop_guard"]["letters_removed"] > 0


def test_guarded_answers_score_better_and_lose_the_loop_status():
    record = {"id": "a", "images": [{"file": "a__0.jpg"}], "gt": {"sections": [{"side": "", "lines": REAL.split("\n")}]}}
    raw = sab.score_document(record, {0: _loop(REAL)})
    guarded = sab.score_document(record, {0: lg.guard_text(_loop(REAL))[0]})
    assert raw["status"] == "loop" and guarded["status"] == "read"
    assert guarded["ngram_precision"] > raw["ngram_precision"] and guarded["ler"] < raw["ler"]


def _write(path, rows):
    """Write JSONL rows.

    :param path: Destination.
    :type path: pathlib.Path
    :param rows: Rows.
    :type rows: list
    """
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def test_guard_file_writes_a_new_guarded_copy_and_refuses_to_clobber(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    _write(out / "m__trained.jsonl", _rows())
    summary = lg.guard_file(out, "m__trained")
    dest = out / "m__trained__guarded.jsonl"
    assert summary["guarded_tag"] == "m__trained__guarded" and dest.exists()
    assert sab.load_outputs(dest)["a"][0] == lg.guard_text(_rows()[0]["text"])[0]
    assert sab.load_outputs(dest)["b"][0] == REAL
    assert [p.name for p in out.iterdir() if p.name.startswith(".")] == []             # no temporary file left
    with pytest.raises(ValueError, match="exists"):
        lg.guard_file(out, "m__trained")
    assert lg.guard_file(out, "m__trained", overwrite=True)["images_cut"] == 1
    with pytest.raises(ValueError, match="already a guarded copy"):
        lg.guard_file(out, "m__trained__guarded")


def test_guard_file_refuses_an_incomplete_read(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    _write(out / "m__trained.jsonl", _rows())
    with pytest.raises(ValueError, match="fewer than the required 117"):
        lg.guard_file(out, "m__trained", require_images=117)
    assert not (out / "m__trained__guarded.jsonl").exists()


def test_cli_prints_a_summary_per_tag_and_flags_refusals(tmp_path, capsys):
    out = tmp_path / "outputs"
    out.mkdir()
    _write(out / "m__trained.jsonl", _rows())
    assert lg.main(["--benchmark", str(tmp_path), "--tags", "m__trained"]) == 0
    first = json.loads(capsys.readouterr().out.strip())
    assert first["images_cut"] == 1 and first["params"]["gate"] == sab.LOOP_REPEAT_SHARE
    assert lg.main(["--benchmark", str(tmp_path), "--tags", "absent__trained", "m__trained", "--overwrite"]) == 2
    lines = [json.loads(l) for l in capsys.readouterr().out.strip().splitlines()]
    assert "not found" in lines[0]["refused"] and lines[1]["images_cut"] == 1                # a refusal does not stop the next tag
    assert lg.main(["--benchmark", str(tmp_path), "--tags", "m__trained"]) == 2              # exists, no --overwrite
    assert "refused" in capsys.readouterr().out
    assert lg.main(["--benchmark", str(tmp_path), "--tags", "m__trained", "--overwrite", "--no-gate"]) == 0
    assert json.loads(capsys.readouterr().out.strip())["params"]["gate"] is None
