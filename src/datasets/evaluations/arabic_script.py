# File name: arabic_script.py
# Date: 10/2/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Arabic-script ground truth and scoring helpers.

The Arabic-script benchmark takes its ground truth from Princeton Geniza Project editions
(``footnotes.csv`` rows whose relation contains "Edition"). Those are scholarly editions, not
diplomatic transcriptions, so two things happen here:

* **From edition to what is on the page.**  :func:`clean_edition_line` removes what the editor added
  (restorations in square brackets, supplied letters in angle brackets, every parenthesised
  note or marker, struck text in double brackets) and keeps what the scribe wrote (insertions
  between ``\\\\ \\\\`` or ``// //``, repeated or superfluous words in braces).
  :func:`split_sections` cuts an edition into its labelled parts (Recto, Verso, margins).
* **Scoring on Arabic letters.**  :func:`arabic_letters` reduces any text to a letters-only
  string after Unicode and orthographic folding (presentation forms, Persian letter variants,
  vowel marks, elongation strokes, hamza carriers, alef maqsura, ta marbuta), so an edition that
  writes a hamza the scribe did not is not counted as a reading error.  :func:`ngram_overlap` is
  the order-robust clipped n-gram precision / recall / F1 used when a document's sides may be read
  in any order, :func:`letter_error_rate` the edit-distance rate for a single aligned text, and
  :func:`script_share` reports how much of an answer is in Arabic letters at all (a model that
  answers an Arabic page in Hebrew letters has read nothing), and :func:`repeat_share` detects an
  answer that collapsed into repeating one phrase.
"""
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Tuple

from Levenshtein import distance as _edit_distance

ARABIC_LETTER = re.compile(r"[\u0621-\u064A]")
HEBREW_LETTER = re.compile(r"[\u05D0-\u05EA]")
SCRIPT_LETTER = re.compile(r"[\u0621-\u064A\u05D0-\u05EA]")
_MARKS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")   # vowel marks, Quranic signs, tatweel
_VARIANTS = str.maketrans({"\u06CC": "\u064A", "\u06A9": "\u0643", "\u06D2": "\u064A", "\u06C1": "\u0647", "\u06D5": "\u0647"})   # Persian yeh, keheh, yeh barree, heh goal, ae
_FOLD = str.maketrans({"\u0623": "\u0627", "\u0625": "\u0627", "\u0622": "\u0627", "\u0671": "\u0627",   # hamza / madda / wasla on alef
                       "\u0649": "\u064A", "\u0626": "\u064A", "\u0624": "\u0648", "\u0629": "\u0647"})   # alef maqsura, hamza carriers, ta marbuta
_SIDE = re.compile(r"^\s*(recto|verso)\b", re.IGNORECASE)
_LINE_NUMBER = re.compile(r"^\s*[0-9\u0660-\u0669]{1,3}[.)]?\s+(?=\S)")
_INSERTION_MARKS = re.compile(r"\\\\|//")
_STRUCK = re.compile(r"\[\[[^\[\]]*\]\]|\u27E6[^\u27E6\u27E7]*\u27E7")      # [[...]] and white square brackets


def normalize_arabic(text: str) -> str:
    """Unicode-level normalisation that changes no reading.

    :param text: Any text.
    :type text: str
    :return: NFKC text (presentation forms and ligatures become plain letters) with Persian letter
        variants mapped to their Arabic counterparts and vowel marks / elongation strokes removed.
    :rtype: str
    """
    return _MARKS.sub("", unicodedata.normalize("NFKC", text).translate(_VARIANTS))


def arabic_letters(text: str, fold: bool = True) -> str:
    """The Arabic letters of a text, as one string.

    :param text: Any text.
    :type text: str
    :param fold: Also fold orthographic variants an edition and a scribe routinely differ on
        (hamza carriers to the bare letter, alef maqsura to ya, ta marbuta to ha).
    :type fold: bool
    :return: Letters only, in order.
    :rtype: str
    """
    norm = normalize_arabic(text)
    if fold:
        norm = norm.translate(_FOLD)
    return "".join(ARABIC_LETTER.findall(norm))


def script_share(text: str) -> Tuple[int, int, float]:
    """How much of a text is written in Arabic letters.

    :param text: A model answer or a ground-truth text.
    :type text: str
    :return: ``(arabic letters, hebrew letters, arabic share of the two)``; the share is 0.0 for a
        text with neither.
    :rtype: Tuple[int, int, float]
    """
    norm = normalize_arabic(text)
    arabic, hebrew = len(ARABIC_LETTER.findall(norm)), len(HEBREW_LETTER.findall(norm))
    return arabic, hebrew, (arabic / (arabic + hebrew) if arabic + hebrew else 0.0)


def repeat_share(text: str, span: int = 12) -> float:
    """Share of an answer's letter runs that repeat an earlier run of the same answer.

    On Arabic and Hebrew letters. A decoder that collapsed into repeating a phrase scores close to
    1 whatever the length of the phrase; a real text, formulae included, stays far below one half.
    (The documentary scorer's ``loop_ratio`` measures only the single most frequent run, which
    misses a loop whose phrase is longer than about 26 letters.)

    :param text: A model answer.
    :type text: str
    :param span: Length of a run, in letters.
    :type span: int
    :return: ``1 - distinct runs / runs``, in [0, 1); 0.0 for answers shorter than three spans.
    :rtype: float
    """
    letters = "".join(SCRIPT_LETTER.findall(normalize_arabic(text)))
    if len(letters) < span * 3:
        return 0.0
    runs = [letters[i:i + span] for i in range(len(letters) - span + 1)]
    return 1.0 - len(set(runs)) / len(runs)


def _strip_brackets(line: str, open_ch: str, close_ch: str, fill: str = " ") -> str:
    """Remove bracketed spans, including one left open at the line end or closed from the line start.

    :param line: One line.
    :type line: str
    :param open_ch: Opening character.
    :type open_ch: str
    :param close_ch: Closing character.
    :type close_ch: str
    :param fill: What each span is replaced by.
    :type fill: str
    :return: The line without the spans.
    :rtype: str
    """
    pattern = re.compile(re.escape(open_ch) + r"[^" + re.escape(open_ch) + re.escape(close_ch) + r"]*" + re.escape(close_ch))
    line = line.replace(fill, "\x00") if fill.strip() else line      # a fill that contains the brackets must not be stripped again
    previous = None
    while previous != line:                       # innermost first, so nested spans go too
        previous, line = line, pattern.sub("\x00", line)
    if close_ch in line:                           # "... restored] visible": restored from the line start
        line = "\x00" + line[line.rindex(close_ch) + 1:]
    if open_ch in line:                            # "visible [restored ...": restored to the line end
        line = line[:line.index(open_ch)] + "\x00"
    return line.replace("\x00", fill)


def clean_edition_line(line: str, gap: Optional[str] = None) -> str:
    """One edition line reduced to what the scribe wrote on the page.

    Removed: restorations and lacuna markers ``[...]``, struck text ``[[...]]`` (also written with white square brackets),
    letters supplied by the editor ``<...>``, every parenthesised span (uncertainty and *sic*
    markers, alternative readings, notes, expansions), dot runs, elongation strokes, a leading
    line number.
    Kept: text inserted by the scribe (``\\\\...\\\\``, ``//...//``) and superfluous or repeated
    words the editor put in braces.

    :param line: One line of a PGP edition.
    :type line: str
    :param gap: When given (training targets use ``"[...]"``), every restoration, lacuna marker
        and dot run is replaced by this token instead of being dropped; neighbouring gaps merge.
    :type gap: Optional[str]
    :return: Cleaned line with single spaces (possibly empty).
    :rtype: str
    """
    fill = f" {gap} " if gap else " "
    line = unicodedata.normalize("NFKC", line).replace("\u00A0", " ")
    line = _strip_brackets(line, "(", ")")
    line = _STRUCK.sub(" ", line)
    line = _strip_brackets(line, "[", "]", fill)
    line = _strip_brackets(line, "<", ">")
    line = _INSERTION_MARKS.sub(" ", line).replace("{", " ").replace("}", " ")
    line = _LINE_NUMBER.sub("", line)
    line = re.sub(r"(?<![\[.])[.\u2026]{2,}(?!\])|\u2026", fill, line).replace("\u0640", "")   # dot runs; elongation strokes are calligraphy, not text
    line = re.sub(r"\s+", " ", line).strip()
    if gap:
        line = re.sub(r"(?:" + re.escape(gap) + r" ?)+", gap + " ", line).strip()
    return line


def split_sections(content: str, gap: Optional[str] = None) -> List[Tuple[str, List[str]]]:
    """Cut an edition into its labelled parts, cleaned line by line.

    A line without any Hebrew or Arabic letter is a label or a gap. A label starting with
    "Recto" or "Verso" opens a new side; other labels (margins, witness clauses, address)
    stay inside the current side, since they are on the same image.

    :param content: Edition text as stored in ``footnotes.csv``.
    :type content: str
    :param gap: Gap token for :func:`clean_edition_line` (None: restorations are dropped).
    :type gap: Optional[str]
    :return: ``(side, lines)`` in edition order, ``side`` in ``{"", "recto", "verso"}`` (``""``
        for text before any side label); sides without text are dropped.
    :rtype: List[Tuple[str, List[str]]]
    """
    sections: List[Tuple[str, List[str]]] = [("", [])]
    for raw in content.splitlines():
        if not SCRIPT_LETTER.search(unicodedata.normalize("NFKC", raw)):
            side = _SIDE.match(raw)
            if side:
                sections.append((side.group(1).lower(), []))
            continue
        cleaned = clean_edition_line(raw, gap)
        if SCRIPT_LETTER.search(cleaned):
            sections[-1][1].append(cleaned)
    return [(side, lines) for side, lines in sections if lines]


def edition_text(content: str) -> str:
    """The visible text of a whole edition, one cleaned line per line.

    :param content: Edition text as stored in ``footnotes.csv``.
    :type content: str
    :return: Cleaned lines of every section, joined by newlines.
    :rtype: str
    """
    return "\n".join(line for _, lines in split_sections(content) for line in lines)


def ngram_overlap(hypothesis: str, reference: str, n: int = 5) -> Dict[str, float]:
    """Clipped character n-gram overlap of two letters-only strings.

    Order-robust at the scale of lines and sides: a reading that has the right text with the
    verso before the recto still scores, and repeating a correct phrase earns it only once.

    :param hypothesis: Letters-only model text (see :func:`arabic_letters`).
    :type hypothesis: str
    :param reference: Letters-only ground truth.
    :type reference: str
    :param n: Gram length in letters.
    :type n: int
    :return: ``precision`` (share of the hypothesis' grams found in the reference), ``recall``,
        ``f1``; all 0.0 when either side has no gram.
    :rtype: Dict[str, float]
    """
    hyp = Counter(hypothesis[i:i + n] for i in range(len(hypothesis) - n + 1))
    ref = Counter(reference[i:i + n] for i in range(len(reference) - n + 1))
    if not hyp or not ref:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    matched = sum(min(count, ref[gram]) for gram, count in hyp.items())
    precision, recall = matched / sum(hyp.values()), matched / sum(ref.values())
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def letter_error_rate(hypothesis: str, reference: str) -> float:
    """Edit distance between two letters-only strings over the reference length.

    Meaningful only when both cover the same text in the same order (one side, one image).

    :param hypothesis: Letters-only model text.
    :type hypothesis: str
    :param reference: Letters-only ground truth.
    :type reference: str
    :return: The rate (can exceed 1.0); 0.0 when both are empty, 1.0 when only the reference is.
    :rtype: float
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _edit_distance(hypothesis, reference) / len(reference)
