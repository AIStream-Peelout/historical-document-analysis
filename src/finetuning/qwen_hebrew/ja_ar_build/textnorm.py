"""Letter skeletons for gate v2: one character per letter, so reads and targets compare letter by letter.

Hebrew letters keep their identity with final forms folded; Arabic letters are folded the way the Arabic letter-error
metric folds them (hamza carriers, alef maqsura, ta marbuta); every mark, digit, space, Latin letter and bracket is
dropped. A Hebrew-script read and an Arabic-script target never match each other, which is intended.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

#: Hebrew final forms folded to their medial letter.
HEBREW_FINALS = str.maketrans({"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"})
#: Arabic letter folding (alef forms, alef maqsura, ta marbuta, hamza carriers).
ARABIC_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})
#: Characters kept after folding: Hebrew letters and the Arabic letter blocks.
LETTER_RE = re.compile("[א-תء-يٱ-ۓ]")
HEBREW_RE = re.compile("[א-ת]")
ARABIC_RE = re.compile("[ء-يٱ-ۓ]")


def skeleton(text: str) -> str:
    """The letter skeleton of a text: letters only, finals and Arabic letter variants folded.

    :param text: Any text (target line, read line).
    :return: The kept letters in order, without spaces or marks.
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFD", text)
    no_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    folded = unicodedata.normalize("NFC", no_marks).translate(HEBREW_FINALS).translate(ARABIC_FOLD)
    return "".join(LETTER_RE.findall(folded.replace("\u0640", "")))


#: Arabic letters folded to their dotless form (rasm), for matching only: manuscripts often omit dots and readers
#: confuse them, while editions supply them.
RASM = str.maketrans({"ب": "ٮ", "ت": "ٮ", "ث": "ٮ", "ن": "ٮ", "ي": "ٮ", "ج": "ح", "خ": "ح", "ذ": "د", "ز": "ر",
                      "ش": "س", "ض": "ص", "ظ": "ط", "غ": "ع", "ف": "ڡ", "ق": "ڡ"})


def match_form(skel: str) -> str:
    """The form of a skeleton used for matching: Arabic letters reduced to their rasm, Hebrew letters unchanged.

    :param skel: A letter skeleton (:func:`skeleton`).
    :return: The matching form (same length as the input).
    """
    return skel.translate(RASM)


def line_skeletons(lines: Iterable[str]) -> List[str]:
    """Skeleton of every line, empty strings kept so indices stay aligned with the input.

    :param lines: Lines of a target or a read.
    :return: One skeleton per input line.
    """
    return [skeleton(line) for line in lines]


def letters(text: str) -> int:
    """Number of letters in a text (length of its skeleton).

    :param text: Any text.
    :return: Letter count.
    """
    return len(skeleton(text))


#: Judaeo-Arabic search markers (the word list of the 2026-11-11 gates survey, ``survey/gates_novel_v1/scripts/
#: common.py``): a lexical search signal on Hebrew-script text, never a language label.
JA_MARKERS = frozenset("""
אלדי אלתי אלדין הדה דלך כדלך ענד ענדי ענדה ענדך ענדנא כאנת ליס איצא אלאן אללה תעאלי אלשיך מולאי מולאנא
סידי סידנא אלסלאם כתאב כתאבי כתאבך אעלם קאל קאלת יקול אלמדכור אלמדכורה פאן גמיע מנהא פיהא אליהא עליהא
""".split())
#: Distinct markers a text needs to carry the JA signal.
JA_MIN_DISTINCT = 3
_ATTACHED_MARK = re.compile("(?<=[א-ת])['’`׳״]")
_HEBREW_RUN = re.compile("[א-ת]+")


def ja_marker_types(text: str) -> List[str]:
    """Distinct Judaeo-Arabic markers in a text (same tokenisation as the gates survey).

    Combining marks are dropped, a geresh or apostrophe attached to a Hebrew letter is dropped, every run of Hebrew
    letters is a token, and a token prefixed by ``ו`` or ``פ`` counts when the rest is a marker.

    :param text: Target text.
    :return: Sorted distinct markers.
    """
    plain = "".join(ch for ch in unicodedata.normalize("NFD", text or "") if not unicodedata.combining(ch))
    plain = _ATTACHED_MARK.sub("", plain)
    types = set()
    for word in _HEBREW_RUN.findall(plain):
        if word in JA_MARKERS:
            types.add(word)
        elif len(word) > 2 and word[0] in "ופ" and word[1:] in JA_MARKERS:
            types.add(word[1:])
    return sorted(types)


def ja_signal(text: str) -> bool:
    """Whether a Hebrew-script text carries the JA search signal (at least :data:`JA_MIN_DISTINCT` markers).

    :param text: Target text.
    :return: True when the signal is present.
    """
    return len(ja_marker_types(text)) >= JA_MIN_DISTINCT


def script_family(text: str) -> str:
    """Script family of a text by its letters: ``hebrew`` when Hebrew letters are at least as many as Arabic ones.

    :param text: Any text (empty text counts as hebrew).
    :return: ``hebrew`` or ``arabic``.
    """
    heb = len(HEBREW_RE.findall(text or ""))
    ara = len(ARABIC_RE.findall(text or ""))
    return "arabic" if ara > heb else "hebrew"
