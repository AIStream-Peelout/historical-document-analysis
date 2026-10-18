# File name: test_arabic_script.py
# Date: 10/2/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Tests for the Arabic-script ground-truth cleaning and letter metrics."""
import pytest

from src.datasets.evaluations import arabic_script as ar


def test_normalize_arabic_removes_marks_and_maps_variants_without_changing_the_reading():
    assert ar.normalize_arabic("بســــــــم اللّٰهِ") == "بسم الله"                     # tatweel, shadda, dagger alef, kasra
    assert ar.normalize_arabic("افرایم میمون کتاب") == "افرايم ميمون كتاب"            # Persian yeh / keheh as typed in editions
    assert ar.normalize_arabic("ﻻﻻجرة") == "لالاجرة"                        # lam-alef ligature (presentation form)


def test_arabic_letters_folds_what_editions_and_scribes_differ_on():
    assert ar.arabic_letters("أحمد إلى الآخرة") == ar.arabic_letters("احمد الي الاخره")
    assert ar.arabic_letters("مسؤول شيئ") == "مسوولشيي"
    assert ar.arabic_letters("أحمد", fold=False) == "أحمد"
    assert ar.arabic_letters("دينار 12 (؟) שלום abc، ٣") == "دينار"                  # digits, punctuation, other scripts go


def test_script_share_tells_an_arabic_answer_from_a_hebrew_one():
    assert ar.script_share("بسم الله") == (7, 0, 1.0)
    arabic, hebrew, share = ar.script_share("בסם אללה الرحمن")
    assert (arabic, hebrew) == (6, 7) and share == pytest.approx(6 / 13)
    assert ar.script_share("1234 ...") == (0, 0, 0.0)


@pytest.mark.parametrize("line, expected", [
    ("بمشـــــــــرفة [الشيـ]ـخ [بو الحسن علي بن احمد بن]", "بمشرفة خ"),                   # restorations out, visible letter stays
    ("المجاورة للفرن المعروف بفرن الجلا[ل] (؟) وحدها", "المجاورة للفرن المعروف بفرن الجلا وحدها"),
    ("واغتناما لاجرتهم (alt= لاخرتهم)", "واغتناما لاجرتهم"),
    ("اختاير (=اختيار) عمدة", "اختاير عمدة"),
    ("ما يبقا (!) على", "ما يبقا على"),
    ("وكان ذرعها {وكان ذرعها} من الجانب", "وكان ذرعها وكان ذرعها من الجانب"),         # the scribe wrote it twice
    ("\\\\في يده {ابتياعا}\\\\ ابتياعا صحيحا", "في يده ابتياعا ابتياعا صحيحا"),          # insertion above the line
    ("يصلح [[له]] \\\\للمداواة\\\\ وانه اهل", "يصلح للمداواة وانه اهل"),                  # struck word out, its replacement in
    ("⟦ناصر⟧ بن علي", "بن علي"),
    ("ولور<ثته> بن <ا>لشبخ المحدود<ة>", "ولور بن لشبخ المحدود"),                        # letters the scribe left out
    ("حتى يقوم به [vacat] ونحن نمثّل", "حتى يقوم به ونحن نمثل"),
    ("وهو ...... سلم الى معجنة", "وهو سلم الى معجنة"),
    ("3 عبدها يقبل يديها", "عبدها يقبل يديها"),
    ("يصلح //له// وانه", "يصلح له وانه"),
    ("الا((ـكرمين)) وابنائه", "الا وابنائه"),
])
def test_clean_edition_line_keeps_the_scribe_and_drops_the_editor(line, expected):
    assert ar.normalize_arabic(ar.clean_edition_line(line)) == ar.normalize_arabic(expected)


def test_clean_edition_line_handles_brackets_that_span_lines():
    assert ar.clean_edition_line("الشيخ ابو سعد باستشفاف حاله[") == "الشيخ ابو سعد باستشفاف حاله"
    assert ar.clean_edition_line("                    ]والحمد لله وحده و[صلواته على سيدنا محمد]") == "والحمد لله وحده و"
    assert ar.clean_edition_line("[هذا ما اشترى حسان واحـ]ــدا جميع الدار") == ar.clean_edition_line("ــدا جميع الدار")
    assert ar.clean_edition_line("[ ]ـي [      عشرة ] دنانير") == "ي دنانير"
    assert ar.clean_edition_line("[...] ... ..... [ ]") == ""


@pytest.mark.parametrize("line, expected", [
    ("بمشـــــــــرفة [الشيـ]ـخ [بو الحسن علي بن احمد بن]", "بمشرفة [...] خ [...]"),
    ("المجاورة للفرن المعروف بفرن الجلا[ل] (؟) وحدها", "المجاورة للفرن المعروف بفرن الجلا [...] وحدها"),
    ("[...] ... ..... [ ]", "[...]"),                                              # neighbouring gaps merge
    ("وهو ...... سلم الى معجنة", "وهو [...] سلم الى معجنة"),
    ("الشيخ ابو سعد باستشفاف حاله[", "الشيخ ابو سعد باستشفاف حاله [...]"),
    ("]والحمد لله وحده و[صلواته على سيدنا محمد]", "[...] والحمد لله وحده و [...]"),
    ("يصلح [[له]] \\\\للمداواة\\\\ وانه اهل", "يصلح للمداواة وانه اهل"),                # struck text is not a loss
    ("ولور<ثته> بن <ا>لشبخ", "ولور بن لشبخ"),                                     # nor are letters the scribe never wrote
    ("3 [بسم الله] الرحمن الرحيم", "[...] الرحمن الرحيم"),
    ("بسم الله الرحمن الرحيم", "بسم الله الرحمن الرحيم"),
])
def test_clean_edition_line_can_mark_losses_for_training_targets(line, expected):
    assert ar.normalize_arabic(ar.clean_edition_line(line, gap="[...]")) == ar.normalize_arabic(expected)
    assert ar.arabic_letters(ar.clean_edition_line(line, gap="[...]")) == ar.arabic_letters(ar.clean_edition_line(line))   # same letters either way


EDITION = """
Recto

بســـم الله الرحمن الرحيم
هذا ما استاجر افرایم بن عالي [الاسرائيلي]
Right margin
وكتب في التاريخ (؟)

Verso (address)

الى الشيخ ابو سعد
[. . .]
#
"""


def test_split_sections_follows_side_labels_and_ignores_other_labels():
    sections = ar.split_sections(EDITION)
    assert [side for side, _ in sections] == ["recto", "verso"]
    assert sections[0][1] == ["بسم الله الرحمن الرحيم", "هذا ما استاجر افرایم بن عالي", "وكتب في التاريخ"]   # the margin stays on its side
    assert sections[1][1] == ["الى الشيخ ابو سعد"]
    assert ar.split_sections("بسم الله\nVerso\nالحمد لله") == [("", ["بسم الله"]), ("verso", ["الحمد لله"])]
    assert ar.split_sections("Recto\n[...]\nVerso:\nلله") == [("verso", ["لله"])]                           # an empty side is dropped
    assert ar.edition_text(EDITION).splitlines()[-1] == "الى الشيخ ابو سعد"
    marked = ar.split_sections(EDITION, gap="[...]")
    assert marked[0][1][1] == "هذا ما استاجر افرایم بن عالي [...]" and marked[1][1] == ["الى الشيخ ابو سعد"]   # a line that is only a loss is dropped


def test_ngram_overlap_is_order_robust_and_clipped():
    recto, verso = ar.arabic_letters("بسم الله الرحمن الرحيم هذا ما استاجر"), ar.arabic_letters("الى الشيخ ابو سعد اطال الله بقاه")
    reference = recto + verso
    assert ar.ngram_overlap(reference, reference) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    swapped = ar.ngram_overlap(verso + recto, reference)                       # sides read in the other order
    assert swapped["precision"] > 0.9 and swapped["recall"] > 0.9
    one_side = ar.ngram_overlap(recto, reference)                              # only the recto was read
    assert one_side["precision"] == 1.0 and 0.4 < one_side["recall"] < 0.6
    looping = ar.ngram_overlap(recto * 5, reference)                           # a correct phrase repeated counts once
    assert looping["precision"] == pytest.approx(one_side["recall"] * len(reference) / (5 * len(recto)), abs=0.05)
    assert ar.ngram_overlap("", reference) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert ar.ngram_overlap("ابجد", reference)["f1"] == 0.0                    # shorter than one gram
    unrelated = ar.arabic_letters("وصل كتاب مولاي الشيخ الجليل وفهمت ما ذكره من امر البضاعه")
    assert ar.ngram_overlap(unrelated, reference)["f1"] < 0.15                 # formulae such as the honorific still overlap: the floor is not zero


def test_repeat_share_separates_a_collapsed_answer_from_a_text():
    basmala = "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ\n"
    assert ar.repeat_share(basmala * 90) > 0.6                                 # one phrase over and over
    assert ar.repeat_share("עסם " * 400) > 0.9                                 # also in Hebrew letters
    letter = "بسم الله الرحمن الرحيم والله لقد صدق حضرة مولاي الشيخ اطال الله بقاه وادام عزه وتاييده وعلوه وتمكينه"
    assert ar.repeat_share(letter) < 0.2
    assert ar.repeat_share(letter + basmala * 2) < 0.45                        # a formula written twice is not a collapse
    assert ar.repeat_share("بسم الله") == 0.0 and ar.repeat_share("") == 0.0


def test_letter_error_rate():
    ref = ar.arabic_letters("بسم الله الرحمن الرحيم")
    assert ar.letter_error_rate(ref, ref) == 0.0
    assert ar.letter_error_rate(ar.arabic_letters("بسم الله الرحمن الرحين"), ref) == pytest.approx(1 / len(ref))
    assert ar.letter_error_rate("", ref) == 1.0 and ar.letter_error_rate("", "") == 0.0 and ar.letter_error_rate("ا", "") == 1.0
    assert ar.letter_error_rate(ref + ref, ref) == pytest.approx(1.0)          # can reach and pass 1.0
