# File name: fgp_notes.py
# Date: 11/19/26
# Author: Isaac Godfried. Coded by Claude.
"""Editor's lines in legacy-store transcription targets that the page layout does not mark (round ja_ar_v1).

The transcription PDFs carry lines that are not on the manuscript: page and folio labels (``דף :א``, ``עמוד ב``,
``א ע"א``), a workflow header (``לפני הגהה אחרונה``, "before final proofreading") and the editors' notes in modern
Hebrew (``בכתב היד כתוב``, ``השלמנו לפי``, ``המילה במלואה``). Footnotes set below a drawn rule or in a smaller font
are cut by the PDF parser (:func:`src.datasets.indexing.legacy_store.pdf_transcriptions.mark_layout_notes`); this
module finds the rest by their wording, measured on the 2026-11 batches (gate run ``novel1114_02``: 14 of 14 lines
with a strong marker were notes).

* :func:`classify_line` names a line ``label`` (page / folio label or the workflow header), ``note`` (an editor's
  note: one strong marker of modern editorial Hebrew, or a line-lemma apparatus entry ``8 .יהודה.[ ...``) or None.
* :func:`split_at_labels` cuts a side at page labels: the lines after an interior ``עמוד ב`` belong to the other
  face of the leaf, so each part becomes a side of its own and the gate places it on its own photograph.
* :func:`clean_side` drops label and note lines, drops an editor's translation (from its heading to the end of the
  side), cuts a page label off the END of a text line (:func:`strip_tail_label`) and moves a line-initial colon to
  the line's end (the text layer stores a right-to-left line's final colon first). Every dropped line or label is
  returned: nothing changes silently.

Page labels at a line end (2026-11-22). The editions of the responsa mark the folio where a new one begins at the
end of the manuscript line (``... היו כולן יוצאות דף א''``), and some give folio and column there
(``... ותמן "ע ב טור א``). Two judges comparing 120 texts with their photographs found these labels in the text of
three pages (T-S G1.82, G1.83, G1.67): they are not on the manuscript. 59 lines of 17 documents end in the first
form, 12 lines of 4 documents in the second.

A wording rule cannot be exact: words such as ``כנראה`` or ``הסופר`` also occur in manuscripts, so they are only
reported by :func:`soft_marker` for lines that no read supports, as candidates for a human decision.
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

_NB = r"(?<![א-ת])"  # no Hebrew letter before
_NA = r"(?![א-ת])"   # no Hebrew letter after
#: a page or folio label: ``דף :א``, ``עמוד ב``, ``דף ס' ב'``, ``א ע"א``, ``ב׳ ע״ב``, or a face letter alone (``א`` / ``ב``)
_FACE = (r"(?:(?:דף|עמוד|צד)\s*:?\s*[א-ת]{1,3}['׳]?(?:\s*(?:ע[\"״'׳]?)?[אב]['׳]?)?\s*:?|"
         r"[א-ת](?:[\"״]?[א-ת])?['׳]?\s*ע[\"״'׳]?[אב]|[א-ת][\"״]ע\s*[אב]|[אב]['׳]?)")
#: a whole line that is a page or folio label, or the workflow header
LABEL_RE = re.compile(r"^\s*(?:" + _FACE + r"|לפני הגהה(?: אחרונה)?)\s*$")
#: a page label that starts the text of another face (``לפני הגהה`` is a header, not a face)
FACE_LABEL_RE = re.compile(r"^\s*" + _FACE + r"\s*$")
#: the heading of an editor's translation: everything after it, to the end of the side, is the translation
TRANSLATION_RE = re.compile(r"תרגום (?:השורות|הקטע|הטקסט|לעברית|עברי)|^\s*תרגום\s*:?\s*$")
#: modern editorial Hebrew: one hit makes the line a note
STRONG_RE = re.compile("|".join([
    r"המהדיר", r"כתב היד", _NB + r"ו?ב?כת?[\"״]י" + _NA, _NB + r"ו?ב?ה?הער(?:ה|ת|ות)" + _NA, r"פורס[םמ]",
    r"השלמנו", r"ההשלמה", r"וההשלמה", r"הושלם (?:עפ|על פי)", r"הושלמו", r"משוער", r"קריאה מסופקת",
    r"הקריאה (?:מסופקת|משוערת|אינה|ודאית)", r"המובאה", r"סימן מחיקה", r"שהסופר", r"הסופר (?:חזר|כתב|דילג|טעה|השמיט)",
    r"נכפל", r"מפתח זה", r"מספרות הגאונים", r"שטמ[\"״]ק", r"ברשימה", r"(?:אפשר|יש) לשער",
    _NB + r"עמ['׳]" + _NA, r"(?:לעיל|להלן) הערה", r"ראה (?:לעיל|להלן)", r"עי?ין (?:לעיל|להלן|ה?הערה)",
    r"גויטיין", r"פרידמן", r"היימן", r"הנוסח שגוי", r"בהעתקתו", r"העתקתי", r"התעתקתי", r"סיימתי כאן", r"ההמש(?:ך)? כבר",
    r"המי?לה (?:במלואה|החסרה)", r"המי?לים (?:במלואן|החסרות)", r"התוספת (?:מעל|מתחת|בשוליים|בין)",
    r"נכתב למילוי", r"הכתיב הנכון", r"גי?רסת הקרי", r"מי?לה (?:עודפת|לא ברורה)", r"נראה להשלים", r"המחיקה בקו",
    r"עי?ין (?:סוף|ראש|תחי?לת) ה?עמוד", r"העמוד (?:שלפני|הקודם|הבא)", r"טקסט בשפה ה?ערבית",
    r"על הגליון", r"הגליון ה(?:ימני|שמאלי|עליון|תחתון)", r"בשולי(?:ים)? ה(?:ימני|שמאלי|עליון|תחתון|דף|עמוד)",
    _NB + r"צ[\"״]ל" + _NA,
]))
#: a whole line that is only a reference to a biblical book or a tractate (an apparatus entry): ``תה' קיט``,
#: ``שמ' ב``, ``תהלים קז``, ``בבא מציעא צ ב``
REFERENCE_RE = re.compile(
    r"^\s*(?:בר['׳]|שמ['׳]|וי['׳]|במ['׳]|דב['׳]|יהו['׳]|שופ['׳]|ש[\"״][אב]|מ[\"״][אב]|יש['׳]|יר['׳]|יח['׳]|תה['׳]|"
    r"מש['׳]|קה['׳]|דה[\"״][אב]|בראשית|שמות|ויקרא|במדבר|דברים|יהושע|שופטים|שמואל|מלכים|ישעיה|ירמיה|יחזקאל|"
    r"תהלים|משלי|איוב|קהלת|עזרא|נחמיה|דניאל|בבא (?:קמא|מציעא|בתרא)|ברכות|שבת|עירובין|פסחים|יומא|סוכה|ביצה|"
    r"תענית|מגילה|חגיגה|יבמות|כתובות|נדרים|גיטין|קידושין|סנהדרין|מכות|שבועות|חולין)"
    r"\s+[א-ת]{1,3}['׳\"״]?[א-ת]?(?:\s*[,:]?\s*(?:[א-ת]{1,3}['׳\"״]?[א-ת]?|\d{1,3}))?\s*[.:]?\s*$")
#: an entry of a line-lemma apparatus: line number, the lemma, ``.[`` and the editor's comment
LEMMA_RE = re.compile(r"^\s*\d{1,3}\s*\.\s*[^\s\[\]][^\[\]]{0,40}\.\s*\[")
#: weaker signs of an editor's note; they also occur in manuscript text (see :func:`soft_marker`)
SOFT_RE = re.compile(r"כנראה|הסופר|ה?מילה|ה?שורה|בשורה|ה?שורות|לאות|האות|מופע|התרגום|המקבילה|ה?גי?רסה|מילולית|"
                     r"הנוסח|השוו?ה|מהד|סה[\"״]ש|אברמסון|היובל|תשמ[\"״]|מחיקה|נמחק|קפצה|השמיט|דילג|כתוב:|לכאורה|"
                     r"י?יתכן|אולי|נראה ש|שיבוש|התוספת|נכתב ל|הביאור|נטבעה|צ['׳]\s?ל")
LEADING_COLON_RE = re.compile(r"^\s*:\s*(\S.*?)\s*$")
#: folio and column as the editions write it (``ע"א טור ב``; the text layer often displaces the quote: ``"ע א טור ב``)
_FOLIO_COLUMN = r"[\"״]?ע\s?[\"״]?\s?[אב]\s+טור\s+[א-ד]"
#: a whole line that is a folio-and-column label, with or without the folio number (``יב "ע א טור ב``), or a
#: column heading alone (``טור ב``, ``ראה טור א``)
COLUMN_LABEL_RE = re.compile(r"^\s*(?:(?:[א-ת]{1,2}['׳]?\s+){0,2}" + _FOLIO_COLUMN + r"|(?:ראה\s+)?טור\s+[א-ד]['׳]?)\s*$")
#: a page label at the END of a text line: ``דף ד''`` (folio number with its geresh marks), or folio and column,
#: with the folio number only when it carries a geresh (a short word before the label is text)
TAIL_LABEL_RE = re.compile(r"\s+(?:דף\s+[א-ת]{1,3}['׳]{1,2}|(?:[א-ת]{1,2}['׳]\s+){0,2}" + _FOLIO_COLUMN + r")\s*$")
#: fewest words a line must keep for a label to be cut off its end (a shorter line is a label line or left alone)
TAIL_LABEL_MIN_WORDS = 2


def classify_line(line: str) -> Optional[str]:
    """Whether a target line is an editor's line by its wording.

    :param line: Target line.
    :return: ``"label"``, ``"note"`` or None.
    """
    if LABEL_RE.match(line) or COLUMN_LABEL_RE.match(line):
        return "label"
    if STRONG_RE.search(line) or LEMMA_RE.match(line) or REFERENCE_RE.match(line):
        return "note"
    return None


def soft_marker(line: str) -> Optional[str]:
    """A weak sign of an editor's note in a line (a word manuscripts also use).

    :param line: Target line.
    :return: The matched word, or None.
    """
    m = SOFT_RE.search(line)
    return m.group(0) if m else None


def move_leading_colon(line: str) -> Tuple[str, bool]:
    """Move a line-initial colon to the end of the line.

    :param line: Target line.
    :return: ``(fixed line, whether it changed)``.
    """
    m = LEADING_COLON_RE.match(line)
    if not m:
        return line, False
    return m.group(1) + ":", True


def strip_tail_label(line: str, min_words: int = TAIL_LABEL_MIN_WORDS) -> Tuple[str, Optional[str]]:
    """Cut an editor's page label off the end of a text line.

    :param line: Target line.
    :param min_words: Fewest words the line must keep; with fewer the line comes back unchanged.
    :return: ``(line without the label, the label that was cut or None)``.
    """
    m = TAIL_LABEL_RE.search(line)
    if not m:
        return line, None
    rest = line[:m.start()].rstrip()
    if len(rest.split()) < min_words:
        return line, None
    return rest, m.group(0).strip()


def split_at_labels(lines: Sequence[str]) -> List[Tuple[Optional[str], List[str]]]:
    """Cut a side's lines at page labels.

    :param lines: The side's lines, labels included.
    :return: ``(label or None, lines)`` per part, in order; the label line itself is left out. A side without an
        interior label comes back as one part.
    """
    parts: List[Tuple[Optional[str], List[str]]] = [(None, [])]
    for line in lines:
        if FACE_LABEL_RE.match(line):
            if parts[-1][1] or parts[-1][0] is not None:
                parts.append((line.strip(), []))
            else:
                parts[-1] = (line.strip(), [])
        else:
            parts[-1][1].append(line)
    return [(label, part) for label, part in parts if part]


def clean_side(lines: Sequence[str]) -> Tuple[List[str], List[Tuple[int, str, str]], int]:
    """Drop the editor's lines of one side and repair line-initial colons.

    :param lines: The side's lines.
    :return: ``(kept lines, dropped as (line index, kind, text), colons moved)``; a page label cut off a line's end
        is in the dropped list with kind ``label_tail`` and the label as text (the line itself is kept).
    """
    kept: List[str] = []
    dropped: List[Tuple[int, str, str]] = []
    moved = 0
    translation = False
    for i, line in enumerate(lines):
        translation = translation or bool(TRANSLATION_RE.search(line))
        kind = "translation" if translation else classify_line(line)
        if kind:
            dropped.append((i, kind, line))
            continue
        line, label = strip_tail_label(line)
        if label:
            dropped.append((i, "label_tail", label))
        line, changed = move_leading_colon(line)
        moved += changed
        kept.append(line)
    return kept, dropped, moved
