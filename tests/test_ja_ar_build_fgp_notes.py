"""Tests for the wording rules that find editor's lines in legacy-store transcription targets."""
from __future__ import annotations

from typing import Optional

import pytest

from src.finetuning.qwen_hebrew.ja_ar_build import fgp_notes as N


@pytest.mark.parametrize("line,kind", [
    ("דף :א", "label"),
    ("עמוד ב", "label"),
    ("א ע\"א", "label"),
    ("לפני הגהה אחרונה", "label"),
    ("ב׳ ע״ב", "label"),
    ("דף ס' ב'", "label"),
    ("ב", "label"),
    ("טקסט בשפה הערבית", "note"),
    ("ט\"ו ע\"ב", "label"),
    ("א\"ע א", "label"),
    ("תה' קיט", "note"),
    ("בבא מציעא צ ב", "note"),
    ("צ\"ל אלאבניה", "note"),
    ("על הגליון מלמטה למעלה כתוב", "note"),
    ("בכתב היד כתוב: וארתפע' ל \"צו וארתפאע'", "note"),
    ("הסופר חזר על אותה מילה", "note"),
    ("השלמנו לפי המקבילה להלן וכן הגירסה בדק ס\"וראה הערת המהדיר שם על אתר", "note"),
    ("המילה במלואה: ספרגל.", "note"),
    ("התוספת מתחת לשורה", "note"),
    (" 8 .יהודה.[ בבא מציעא נ';  ב ,  ח\"עיי רי\", במקומו שמביא דעות שונות", "note"),
    ("ההמש כבר פורסם בהפעמ ש\"' ה ואיל ולכן סיימתי כאן את התעקתי", "note"),
    ("רב חנניה [...] ל מארי בר חנניה עיין סוף העמוד שלפני זה", "note"),
])
def test_editor_lines_are_classified(line: str, kind: str) -> None:
    """Labels, the workflow header and notes in modern editorial Hebrew are recognised."""
    assert N.classify_line(line) == kind


@pytest.mark.parametrize("line", [
    "שרח אלוציה",
    "מצחף תורה מלוח ומצחפין מלוחה נבי' ארבעה אול",
    "ולחן שאו מרום עיניכם וראו מי ברא",                      # the verb ראו, not a reference
    "עינים להם ולא יראו אזנים להם ולא יאזינו אף",
    "33  . צריך ששי' ולא אמרינין קליפה אלא בבלתי מפעפע",       # a numbered manuscript line, no lemma bracket
    "11 ' הד ה אלי' אשיא ינבגי",
    "ולם תפרק אליהוד פי סאיר אעמאלהם עמי ועמך",               # עמ inside words
    "דף זה כתוב",                                              # a label needs the whole line
    "שבת שלום ומבורך לכל הקהל הקדוש",                         # a tractate's name inside a sentence
    "שמואל בן יהודה הכהן",                                      # a name, not a reference
])
def test_manuscript_lines_are_kept(line: str) -> None:
    """Manuscript text, numbered lines included, is not classified."""
    assert N.classify_line(line) is None


def test_soft_marker_reports_without_classifying() -> None:
    """A weak sign is reported apart and never makes a line a note by itself."""
    line = "כנראה שעינו של הסופר קפצה למופע הקודם של המילה"
    assert N.classify_line(line) is None
    assert N.soft_marker(line) == "כנראה"
    assert N.soft_marker("שרח אלוציה") is None


def test_split_at_labels_gives_one_part_per_face() -> None:
    """Lines after an interior page label become a part of their own; the labels are left out."""
    lines = ["עמוד א", "שרח אלוציה", "מצחף תורה", "עמוד ב", "דפתר פיה מגמוע", "לפני הגהה אחרונה"]
    assert N.split_at_labels(lines) == [("עמוד א", ["שרח אלוציה", "מצחף תורה"]),
                                        ("עמוד ב", ["דפתר פיה מגמוע", "לפני הגהה אחרונה"])]
    assert N.split_at_labels(["שרח אלוציה", "מצחף תורה"]) == [(None, ["שרח אלוציה", "מצחף תורה"])]
    assert N.split_at_labels(["עמוד א", "עמוד ב"]) == []


@pytest.mark.parametrize("line,fixed", [(":נפשי את יי הללויה", "נפשי את יי הללויה:"), ("נפשי: את יי", None)])
def test_move_leading_colon(line: str, fixed: Optional[str]) -> None:
    """Only a line-initial colon moves to the line's end."""
    out, changed = N.move_leading_colon(line)
    assert (out, changed) == ((fixed, True) if fixed else (line, False))


def test_clean_side_drops_and_reports() -> None:
    """Editor's lines are dropped with their index and kind; colons are repaired; the rest is untouched."""
    lines = ["לפני הגהה אחרונה", "שרח אלוציה", ":וראו כל העמים כבודו", "הסופר חזר על אותה מילה", "מצחף תורה"]
    kept, dropped, moved = N.clean_side(lines)
    assert kept == ["שרח אלוציה", "וראו כל העמים כבודו:", "מצחף תורה"]
    assert dropped == [(0, "label", "לפני הגהה אחרונה"), (3, "note", "הסופר חזר על אותה מילה")]
    assert moved == 1


def test_clean_side_drops_a_translation_to_the_end_of_the_side() -> None:
    """An editor's translation runs from its heading to the end of the side."""
    lines = ["שרח אלוציה", "מצחף תורה", "תרגום השורות הערביות", "פירוש הצוואה", "ספר תורה"]
    kept, dropped, _ = N.clean_side(lines)
    assert kept == ["שרח אלוציה", "מצחף תורה"]
    assert [kind for _, kind, _ in dropped] == ["translation"] * 3


@pytest.mark.parametrize("line,rest,label", [
    ("רבנן בהא מילתא ומיסתיעא לן מיהא דתנן היו כולן יוצאות דף א''", "רבנן בהא מילתא ומיסתיעא לן מיהא דתנן היו כולן יוצאות", "דף א''"),
    ("לא קאמ' אלא בעכו דמקרבא אבל בבל דמרחקא דף ד''", "לא קאמ' אלא בעכו דמקרבא אבל בבל דמרחקא", "דף ד''"),
    ("ו' דנא ורבע ותמן \"ע ב טור א", "ו' דנא ורבע ותמן", "\"ע ב טור א"),
    ("עליאלוגיה ופי אלסוכה י' ג' \"ע ב טור ב", "עליאלוגיה ופי אלסוכה", "י' ג' \"ע ב טור ב"),
])
def test_a_page_label_is_cut_off_the_end_of_a_text_line(line: str, rest: str, label: str) -> None:
    """The editions' folio marks at a line end are not on the manuscript.

    :param line: Target line.
    :param rest: The line without the label.
    :param label: The label cut off.
    """
    assert N.strip_tail_label(line) == (rest, label)


@pytest.mark.parametrize("line", [
    "אמ' ר' אסי אמ' ר' אלעזר אמ' ר' חנינא אמ' רבי בצד עמוד זה",     # "beside this pillar": text
    "ואם אין שם דף אחד כשר",                                           # the word alone, no folio number
    "שנאמר ויאמר לו טור א",                                            # a column word without the folio form
    "אמר דף ד''",                                                      # too little text left: not cut
])
def test_text_that_only_resembles_a_page_label_is_kept(line: str) -> None:
    """A label is cut only in the editions' own forms and only off a line that keeps its text.

    :param line: Target line.
    """
    assert N.strip_tail_label(line) == (line, None)


def test_folio_and_column_lines_are_labels_and_clean_side_reports_cut_tails() -> None:
    """A whole line of folio and column is a label; a cut tail is reported, its line kept."""
    assert N.classify_line("יב \"ע א טור ב") == "label" and N.classify_line("ע\"א טור א") == "label"
    assert N.classify_line("טור ב") == "label" and N.classify_line("ראה טור א") == "label"
    assert N.classify_line("ויעש טור אחד של אבנים") is None
    kept, dropped, _ = N.clean_side(["שורה ראשונה של הטקסט", "ושורה שניה נגמרת כאן דף ב''", "ע\"א טור א"])
    assert kept == ["שורה ראשונה של הטקסט", "ושורה שניה נגמרת כאן"]
    assert dropped == [(1, "label_tail", "דף ב''"), (2, "label", "ע\"א טור א")]
