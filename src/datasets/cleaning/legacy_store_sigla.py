# File name: legacy_store_sigla.py
# Date: 10/30/26
# Author: Isaac Godfried. Coded by Claude.
"""Clean legacy-store diplomatic transcription sections into visible-ink text.

The project's legacy image store holds anonymous transcription sections stored as FLAT
single-line strings (no line breaks). They follow five transcription conventions
(*families*): KTIV-style sigla (``ktiv_sigla``), plain documentary transcriptions
(``documentary_plain``), printed-edition scrapes with an apparatus (``edition_apparatus``),
liturgical layout markup (``liturgical_markup``) and near-empty sections (``empty``). A mark
can mean different things in different families, so every rule below is family-aware. The
contract (mark -> action, checked letter by letter against KTIV word annotations of the same
transcription teams) is the internal sigla specification of 2026-10-30.

Output conventions (the target of a script-neutral page reader):

* Lost or restored text becomes the gap token :data:`GAP` (``[...]``, as in the Arabic-script
  and book-edition builders); neighbouring gaps merge into one.
* Ink is kept in reading order without its delimiters: interlinear additions ``[[x]]`` and
  ``^x^``, margin text ``[x]`` (``ktiv_sigla`` only), deleted-but-visible text ``(x)``
  (``ktiv_sigla`` only), scribal corrections ``{x}``, line fillers ``<x>``, catchwords
  ``%x%``, not-body-text ink ``*x*`` and in-line corrections ``[|x|]``.
* Uncertain letters (``?x?``, edition ``x$``) are kept without their mark and counted.
* Not ink, dropped: paragraph marks (``¶``, ``\\``), note marks (``&``), erasures ``((x))``,
  editor's expansions and notes in parentheses (outside ``ktiv_sigla``), Latin text, digits,
  line numbers (an explicit per-line number becomes a line break), side and column headings.
* Restorations (``[x]`` outside ``ktiv_sigla``; ``(x)`` in Weiss-format sections) become gaps
  and their letters are counted as reconstruction.
* A bracket without a partner never erases text: the bracket is removed (an unpaired square
  bracket marks a damaged edge and becomes a gap), the text stays, and the side is flagged.
* Legacy-font mojibake finals are repaired with a per-section symbol -> letter mapping
  (:data:`FINALS_MAPPING`, keyed by :func:`section_digest`); local visual-order fragments
  (``םא`` for ``אם``) are re-ordered, and so are multi-letter uncertain wraps stored in visual
  order (``ש?ול?ם`` for ``ש?לו?ם``, ``ktiv_sigla`` only, decided per wrap by a lexicon).

Every letter in the output comes from the input: letters are only moved (fragment repair),
dropped (recorded in :attr:`CleanedSection.dropped_spans`) or, for mojibake finals, restored
from the mapping. :func:`letter_audit` checks this balance.
"""
import collections
import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Collection, Counter, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from src.datasets.cleaning.clean_genizah_transcriptions import (
    EMPTY_PARENS_RE,
    FOOTNOTE_RULE_RE,
    LOSS_NOTE_RE,
    MAX_EDITORIAL_SPAN,
    PIPE_RE,
    classify_section,
)

# ───────────────────────────── constants ─────────────────────────────
GAP = "[...]"
_GAP = "\ue000"          # internal gap placeholder (brackets of GAP must not meet bracket rules)
_UNRESOLVED = "\ue001"   # internal placeholder for a word-final mojibake symbol without a mapping
HEB = "א-ת"
FINALS = "ךםןףץ"
NONFINAL_LETTERS = "אבגדהוזחטיכלמנסעפצקרשת"
HEBREW_LETTER_RE = re.compile(f"[{HEB}]")
ARABIC_LETTER_RE = re.compile(r"[\u0620-\u064A\u066E-\u066F\u0671-\u06D3\u06D5\u06FA-\u06FC\u06FF]")
LETTER_RE = re.compile(r"[\u05D0-\u05EA\u0620-\u064A\u066E-\u066F\u0671-\u06D3\u06D5\u06FA-\u06FC\u06FF]")
HEB_RUN_RE = re.compile(f"[{HEB}]+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
LATIN_RE = re.compile(r"[A-Za-z\u00C0-\u024F]+")
FAMILIES = ("ktiv_sigla", "edition_apparatus", "liturgical_markup", "documentary_plain", "empty")
BRACKET_PAIRS: Dict[str, Tuple[str, str]] = {"[]": ("[", "]"), "()": ("(", ")"), "<>": ("<", ">"), "{}": ("{", "}")}

# Family features (identical to the survey that wrote the specification).
SIGLA_FEATURES: Dict[str, Tuple["re.Pattern[str]", int]] = {
    "uncertain_wrap": (re.compile(rf"\?[{HEB}]{{1,4}}\?"), 2),
    "pilcrow": (re.compile("¶"), 1),
    "middot": (re.compile("·"), 1),
    "column": (re.compile(rf"טור_[{HEB}]"), 1),
    "at_wrap": (re.compile(rf"@[{HEB}]{{1,6}}@"), 1),
    "dot_gap": (re.compile(rf"\.{{2,}} \.(?=[{HEB}?])"), 2),
    "line_filler": (re.compile(rf"<[{HEB}]{{1,6}}['׳]>"), 1),
}
EDITION_MARK_RE = re.compile(r"מקרא סימני|השלמת המהדיר|John Rylands|ק\"\s?ג |מהד\s*['׳]")
LITURGICAL_FEATURES: Tuple["re.Pattern[str]", ...] = (
    re.compile("§"), re.compile("##"), re.compile(r"\\\\"), re.compile(r"\*&"))
# Editor vocabulary that betrays apparatus or notes left inside a section (flag only).
EDITORIAL_TERMS_RE = re.compile(
    r"(?<![א-ת])(?:שורה\s*\d|מעל השורה|השלמתי|הערה\s*\d|עמ\s*['׳]?\s*\d|מהד\s*['׳]|קטלוג|ציטטות|תוספת)(?![א-ת])")

# Asymmetric frequent words: the letter reversal of each is not an ordinary word.
REVERSAL_PROBES = ("אשר", "כי", "של", "הוא", "היא", "אמר", "זה", "ישראל", "כל", "שלא", "הזה", "דבר",
                   "לו", "פי", "הדא", "אלדי", "קאל", "אללה", "עלי", "אלי", "את")

# ── structure ──
_SIDE_ALTERNATIVES: Tuple[Tuple[str, str], ...] = (
    ("daf", r"(?<![א-ת])דף\s*(?P<daf_leaf>[א-ת]{1,2}|\d{1,3})\s*['׳]?\s*צד\s*['׳]?\s*(?P<daf_side>[אב12])(?![א-ת\d])['׳]?"),
    # "לפני צד א" / "אחרי צד ב" (before / after side א) is an editor's note, not a heading
    ("tsad", r"(?<![א-ת])(?<!לפני )(?<!אחרי )(?<!של )צד\s*['׳]?\s*(?P<tsad_side>[אב12])(?![א-ת\d])['׳]?"),
    ("amud", r"(?<![א-ת\d.])(?<![A-Za-z] )(?:ד\s*)?(?P<amud_leaf>\d{1,2})\s+(?:ע[\"״](?P<amud_s1>[אב])|(?P<amud_s2>[אב])[\"״]\s?ע)(?![א-ת])"),
    # edition header: shelf mark (Latin, digits, '*') then "ע\"ב )"
    ("amudc", r"(?<=[A-Za-z\d*.)] )(?:ע[\"״](?P<amudc_s1>[אב])|(?P<amudc_s2>[אב])[\"״]\s?ע)(?![א-ת])(?=\s*\))"),
    ("amud0", r"^\s*(?:ע[\"״](?P<amud0_s1>[אב])|(?P<amud0_s2>[אב])[\"״]\s?ע)(?![א-ת])"),
    ("folio", r"(?<![^\s(])(?P<folio_leaf>\d{1,2})(?P<folio_side>[rv])(?![A-Za-z\d])"),
    ("colon", r"(?<![^\s])(?P<colon_leaf>\d{1,2})(?P<colon_side>[אב]):"),
    ("card", r"(?<![^\s])\d{3,4}\s+(?P<card_side>[אב])(?=\s)"),
    ("rv", r"\(?\s*\b(?P<rv_word>[Rr]ecto|[Vv]erso)\b\s*\)?"),
    ("pcode", r"[\[(]?\s*\bP\d\s*-?\s*(?P<p_side>[FB])\b\s*[\])]?"),
    ("half", r"(?<![א-ת])צד\s+(?:ימין|שמאל)(?![א-ת])"),
)
SIDE_MARKER_RE = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in _SIDE_ALTERNATIVES))
_STRONG_SIDE_KINDS = ("daf", "folio", "colon", "amud", "amud0", "amudc", "tsad", "card", "rv")   # label priority order
_MARKER_JOIN_RE = re.compile(r"^[\s()\[\]'׳\"״:.,\-]*$")   # text allowed between grouped markers
_MARKER_TAIL_RE = re.compile(r"^\s*:")
COLUMN_MARKER_RE = re.compile(rf"טור_([{HEB}])|_טור ([{HEB}])(?![{HEB}])")
_GEMATRIA = {"א": 1, "ב": 2, "ג": 3, "ד": 4, "ה": 5, "ו": 6, "ז": 7, "ח": 8, "ט": 9, "י": 10, "כ": 20, "ך": 20,
             "ל": 30, "מ": 40, "ם": 40, "נ": 50, "ן": 50, "ס": 60, "ע": 70, "פ": 80, "ף": 80, "צ": 90, "ץ": 90,
             "ק": 100, "ר": 200, "ש": 300, "ת": 400}
# Explicit per-line numbers (01 02 / 1. 2. / visual "28 .X") become line breaks; every-fifth
# numbers (5 10 15) do not number every line, so they are only dropped.
LINE_NUMBER_RE = re.compile(rf"(?<![^\s])(?:0\d(?=\s)|\d{{1,2}} ?\.(?= ?[{HEB}\[\].?(<{{^]))")
EVERY_FIFTH_RE = re.compile(r"(?<![^\s])(?:5|10|15|20|25|30)(?= [\[\]])")

# ── marks (applied to bracket-orientation-normalised text) ──
INLINE_CORRECTION_RE = re.compile(r"\[\|([^|\]]*)\|\]")
INTERLINEAR_DOUBLE_RE = re.compile(r"\[\[([^\[\]]*)\]\]")
BLANK_BRACKETS_RE = re.compile(r"\[[\s.…?·]*\]|\][\s.…?·]*\[")
# An erasure is followed by its correction {x} in the same word; the export puts a space between.
ERASURE_RE = re.compile(r"\(\(([^()\n]*)\)\)(?: (?=\{))?")
PAREN_RE = re.compile(r"\(([^()\n]*)\)")
SQUARE_RE = re.compile(r"\[([^\[\]\n]*)\]")
BRACE_RE = re.compile(r"\{([^{}]*)\}")
EMPTY_ANGLE_RE = re.compile(r"<\s*>")
ANGLE_RE = re.compile(r"<([^<>]*)>")
UNCERTAIN_WRAP_RE = re.compile(rf"\?([{HEB}]{{1,6}})\?")
MULTI_WRAP_RE = re.compile(rf"\?([{HEB}]{{2,4}})\?")      # wraps whose letter order can be visual
_WORD_TAIL_RE = re.compile(rf"[{HEB}?]*$")                 # word piece before a wrap (? marks inside)
_WORD_HEAD_RE = re.compile(rf"[{HEB}?]*")                  # word piece after a wrap
DOLLAR_UNCERTAIN_RE = re.compile(rf"(?<=[{HEB}])\$")
STAR_WRAP_RE = re.compile(r"\*([^*]{1,60})\*")
DOUBLET_RE = re.compile(rf"([{HEB}]+) ?/{{3}} ?([{HEB}]+)")
DOT_RUN_RE = re.compile(r"\.(?: ?\.)+")
UNDERSCORE_RUN_RE = re.compile(r"_{2,}")
# A geresh / gershayim standing alone after a word (export spacing: "אמ ' רב", "איצ ' א") belongs to it.
DETACHED_GERESH_RE = re.compile(rf"(?<=[{HEB}]) +(['\"׳״]+)(?= |\n|$)")
LEADING_GERESH_RE = re.compile(rf"(?<![^\s])'([{HEB}]{{1,4}})(?=[\s.:]|$)")
DIGITS_RE = re.compile(r"\d+")
_SPACE_MARKS = "¶\\#§=,;-–—\u05BE\u05C0"          # separators: replaced by a space
_JOIN_MARKS = "?@^*%&+/!~`\u02CB$\u2022\u00A9"   # delimiters inside or next to words: removed (| via PIPE_RE)
_KEEP_CHARS = set(" \n'\"׳״:." + _GAP)


# ───────────────────────────── data ─────────────────────────────
@dataclass
class CleanedSection:
    """One side (page face) of a cleaned legacy-store section.

    :param side: Side label: ``"r"`` / ``"v"``, with the leaf number when the heading gives it
        (``"1r"``, ``"2v"``); None when the section has no side heading.
    :param column: Column labels found on this side (``"א,ב"``), None without column headings.
    :param text: Cleaned text; columns are separated by a blank line, explicit line numbers became
        line breaks, :data:`GAP` marks lost or restored text.
    :param letters: Hebrew and Arabic letters in ``text``.
    :param gaps: Gap tokens in ``text``.
    :param recon_letters: Letters of editorial restorations replaced by gaps.
    :param uncertain_letters: Letters kept that the transcriber marked as uncertain.
    :param finals_repaired: Mojibake final symbols replaced by their final letter.
    :param reversed: Whether the whole section was stored in visual (reversed) order.
    :param dropped_spans: ``(kind, text)`` of every dropped span that held letters.
    :param flags: Review flags with counts (unmatched brackets, unresolved finals, ...).
    :param family: Convention family of the section.
    :param side_source: ``"marker"``, ``"inferred"`` (text before a verso heading) or None.
    :param fragments_repaired: Visual-order letter fragments re-ordered (final-first runs, lone-final
        joins and uncertain wraps; the wraps are also counted in the ``wrap_reversed`` flag).
    """

    side: Optional[str]
    column: Optional[str]
    text: str
    letters: int
    gaps: int
    recon_letters: int
    uncertain_letters: int
    finals_repaired: int
    reversed: bool
    dropped_spans: List[Tuple[str, str]] = field(default_factory=list)
    flags: Dict[str, int] = field(default_factory=dict)
    family: str = ""
    side_source: Optional[str] = None
    fragments_repaired: int = 0


@dataclass
class _Acc:
    """Running counts while one side is cleaned.

    :param recon_letters: Letters of restorations replaced by gaps.
    :param uncertain_letters: Letters marked uncertain.
    :param dropped: ``(kind, text)`` of dropped spans holding letters.
    :param flags: Flag counts.
    """

    recon_letters: int = 0
    uncertain_letters: int = 0
    dropped: List[Tuple[str, str]] = field(default_factory=list)
    flags: Counter = field(default_factory=collections.Counter)

    def drop(self, kind: str, span: str) -> None:
        """Record a dropped span when it holds letters.

        :param kind: Why it was dropped.
        :param span: The dropped text.
        """
        if LETTER_RE.search(span):
            self.dropped.append((kind, span.strip()))


@dataclass
class _SideSegment:
    """A stretch of a section between side headings.

    :param side: Side label (None before the first heading or without headings).
    :param marker: Heading text that opens the segment ("" for none).
    :param text: Segment text.
    :param source: ``"marker"`` or None.
    """

    side: Optional[str]
    marker: str
    text: str
    source: Optional[str] = None


# ───────────────────────────── text helpers ─────────────────────────────
def letters_of(text: str) -> str:
    """Hebrew and Arabic letters of a text, in order.

    :param text: Any text.
    :return: The letters.
    """
    return "".join(LETTER_RE.findall(text))


def section_digest(text: str) -> str:
    """Content key of a raw section (the key of :data:`FINALS_MAPPING`).

    :param text: Raw section text exactly as stored.
    :return: First 16 hex digits of its SHA-1.
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def normalise_text(text: str) -> Tuple[str, int]:
    """Unicode clean-up that keeps every letter and every mark the rules need.

    NFC; Hebrew presentation forms decomposed to their base letter; vowel points, cantillation
    and other combining marks removed (other training targets are unpointed); sof pasuq -> ``:``;
    maqaf and paseq -> space; Arabic presentation forms -> base letters; no-break space -> space;
    bidi controls removed; curly apostrophes -> ``'``; ellipsis -> ``...``. Control characters are
    left alone (they may be mojibake finals).

    :param text: Raw section text.
    :return: (normalised text, combining marks removed).
    """
    text = unicodedata.normalize("NFC", text)
    out: List[str] = []
    points = 0
    for ch in text:
        o = ord(ch)
        if 0xFB1D <= o <= 0xFB4F or 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF:
            for part in unicodedata.normalize("NFKD", ch):
                if unicodedata.category(part) == "Mn":
                    points += 1
                else:
                    out.append(part)
            continue
        if unicodedata.category(ch) == "Mn":
            points += 1
            continue
        if ch == "\u05C3":
            out.append(":")
        elif ch in "\u05BE\u05C0\u00A0":
            out.append(" ")
        elif ch in "\u200B\u200E\u200F\u202A\u202B\u202C\u202D\u202E\u2066\u2067\u2068\u2069\uFEFF\u05C6":
            continue
        elif ch in "\u2018\u2019":
            out.append("'")
        elif ch == "…":
            out.append("...")
        else:
            out.append(ch)
    return "".join(out), points


def strip_controls(text: str) -> str:
    """Replace control characters, tabs and carriage returns by spaces.

    Run :func:`repair_finals` first: in mojibake sections some control characters are finals.

    :param text: Any text.
    :return: Text without C0 control characters (newlines kept).
    """
    return re.sub(r"[\x00-\x08\x0b-\x1f\x7f\t\r]", " ", text)


def _collapse(text: str) -> str:
    """Single spaces inside lines, no empty lines, no gap-only lines, merged gaps.

    :param text: Text with :data:`_GAP` placeholders.
    :return: Tidied text.
    """
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line)
        line = re.sub(f"{_GAP}(?: ?{_GAP})+", _GAP, line).strip()
        if line and line != _GAP:
            lines.append(line)
    return "\n".join(lines)


# ───────────────────────────── families ─────────────────────────────
def classify_family(text: str) -> str:
    """Convention family of a raw section.

    Precedence: ``empty`` (fewer than 10 Semitic letters) > ``edition_apparatus`` (the July content
    classifier calls it an edition scrape, or two edition marks: legend, edition header,
    catalogue line) > ``ktiv_sigla`` (two KTIV-style sigla features) > ``liturgical_markup``
    (two liturgical layout marks) > ``documentary_plain``.

    :param text: Raw section text.
    :return: One of :data:`FAMILIES`.
    """
    if len(re.findall(r"[א-ת\u0600-\u06FF]", text)) < 10:
        return "empty"
    if classify_section("", text) == "edition_scrape" or len(EDITION_MARK_RE.findall(text)) >= 2:
        return "edition_apparatus"
    if sum(1 for pat, need in SIGLA_FEATURES.values() if len(pat.findall(text)) >= need) >= 2:
        return "ktiv_sigla"
    if sum(len(p.findall(text)) for p in LITURGICAL_FEATURES) >= 2:
        return "liturgical_markup"
    return "documentary_plain"


def is_finals_mojibake(text: str, min_words: int = 15, max_rate: float = 2.0) -> bool:
    """Whether a section lost its final letters to legacy-font mojibake.

    :param text: Raw section text.
    :param min_words: Shortest section judged (shorter ones can lack finals by chance).
    :param max_rate: Real final letters per 100 words below which the finals count as missing.
    :return: True when the section's finals are missing.
    """
    words = HEB_RUN_RE.findall(text)
    if len(words) < min_words:
        return False
    return 100 * sum(text.count(f) for f in FINALS) / len(words) < max_rate


# ───────────────────────────── reversal ─────────────────────────────
def unreverse_if_needed(text: str, min_words: int = 20) -> Tuple[str, bool]:
    """Undo whole-section visual-order (right-to-left) storage.

    A section is reversed when, with at least ``min_words`` letter runs, word-initial final
    letters outnumber word-final ones AND reversed probe words (:data:`REVERSAL_PROBES`)
    outnumber their forward forms. Mirrored brackets are left to :func:`normalise_brackets`.

    :param text: Section text.
    :param min_words: Fewest letter runs judged.
    :return: (text in logical order, whether it was reversed).
    """
    words = HEB_RUN_RE.findall(text)
    if len(words) < min_words:
        return text, False
    counts = collections.Counter(words)
    initial = sum(1 for w in words if len(w) > 1 and w[0] in FINALS and w[-1] not in FINALS)
    final = sum(1 for w in words if w[-1] in FINALS)
    forward = sum(counts[w] for w in REVERSAL_PROBES)
    backward = sum(counts[w[::-1]] for w in REVERSAL_PROBES)
    if initial > final and backward > forward:
        return text[::-1], True
    return text, False


def repair_reversed_fragments(text: str) -> Tuple[str, int]:
    """Re-order letter runs stored in visual order (``ךל`` -> ``לך``).

    A maximal Hebrew letter run that starts with a final letter and does not end with one cannot
    be a word in logical order; it is reversed. A lone final letter is left alone.

    :param text: Section text.
    :return: (repaired text, runs reversed).
    """
    pattern = re.compile(rf"(?<![{HEB}])[{FINALS}][{HEB}]*[{NONFINAL_LETTERS}](?![{HEB}])")
    return pattern.subn(lambda m: m.group(0)[::-1], text)


def repair_fragment_order(text: str, lexicon: Optional[Collection[str]]) -> Tuple[str, int]:
    """Join a lone final letter to the following short fragment when stored in reversed order.

    ``ן שאי`` -> ``שאין``: a final letter standing alone before a 2-4 letter run is joined after
    it when that join is a lexicon word and the final does not instead complete the preceding run
    (``חכ ם``, the line-break split). Without a lexicon nothing changes.

    :param text: Section text.
    :param lexicon: Known words, or None.
    :return: (repaired text, joins made).
    """
    if not lexicon:
        return text, 0
    pattern = re.compile(rf"(?<![{HEB}])([{FINALS}]) ([{HEB}]{{2,4}})(?![{HEB}])")
    prev_run = re.compile(rf"([{HEB}]+) $")
    made = 0

    def _join(m: "re.Match[str]") -> str:
        """Join one candidate when the lexicon supports only the reversed order.

        :param m: Candidate match (final letter, following run).
        :return: Replacement text.
        """
        nonlocal made
        fin, run = m.group(1), m.group(2)
        before = prev_run.search(text[max(0, m.start() - 40):m.start()])
        completes_previous = bool(before) and before.group(1) + fin in lexicon
        if run[-1] not in FINALS and run + fin in lexicon and not completes_previous:
            made += 1
            return run + fin
        return m.group(0)

    return pattern.sub(_join, text), made


def repair_uncertain_wraps(text: str, lexicon: Optional[Collection[str]], window: int = 60) -> Tuple[str, int]:
    """Re-order multi-letter uncertain wraps stored in visual order (``ש?ול?ם`` -> ``ש?לו?ם``).

    In ``ktiv_sigla`` sections a ``?xy?`` wrap of 2-4 letters is often stored reversed while the
    rest of its word is in logical order, and the direction varies inside one section, so every
    wrap is decided alone. The whole word is rebuilt from the letters next to the wrap (``?``
    marks ignored); the wrap content is reversed when the reversed join is a lexicon word and the
    forward join is not. Palindromes and wraps that start with a final letter (those are left to
    :func:`repair_reversed_fragments`) are skipped. All wraps are judged on the input text, so
    one repair never changes the context of the next. Without a lexicon nothing changes.

    :param text: Section text (``?`` marks still in place).
    :param lexicon: Known words, or None.
    :param window: Characters searched before a wrap for the rest of its word.
    :return: (repaired text, wraps reversed).
    """
    if not lexicon:
        return text, 0
    made = 0

    def _flip(m: "re.Match[str]") -> str:
        """Reverse one wrap when only the reversed word is known.

        :param m: Match of :data:`MULTI_WRAP_RE`.
        :return: Replacement text.
        """
        nonlocal made
        inner = m.group(1)
        if inner == inner[::-1] or inner[0] in FINALS:
            return m.group(0)
        before = _WORD_TAIL_RE.search(text[max(0, m.start() - window):m.start()]).group(0).replace("?", "")
        after = _WORD_HEAD_RE.match(text, m.end()).group(0).replace("?", "")
        if before + inner[::-1] + after in lexicon and before + inner + after not in lexicon:
            made += 1
            return f"?{inner[::-1]}?"
        return m.group(0)

    return MULTI_WRAP_RE.sub(_flip, text), made


# ───────────────────────────── finals ─────────────────────────────
FINAL_CANDIDATE_RE = re.compile(
    r"([א-ת]+)([^א-ת \n'\".,:;?!()\[\]{}<>·¶_\-–/|\\0-9A-Za-z\u0591-\u05C7\uFB1D-\uFB4F\u0600-\u06FF])(?![א-ת])")


def repair_finals(text: str, mapping: Optional[Mapping[str, str]]) -> Tuple[str, int]:
    """Replace word-final mojibake symbols by the final letter the section's mapping gives.

    Only a symbol right after a Hebrew letter and not followed by one is replaced; the same symbol
    elsewhere is left for the mark rules.

    :param text: Section text.
    :param mapping: Symbol -> final letter for this section (None or empty: nothing to do).
    :return: (repaired text, symbols replaced).
    """
    n = 0
    for sym, letter in (mapping or {}).items():
        if letter not in FINALS:
            raise ValueError(f"mapping target {letter!r} is not a final letter")
        text, k = re.subn(rf"(?<=[{HEB}]){re.escape(sym)}(?![{HEB}])", letter, text)
        n += k
    return text, n


def infer_finals_mapping(text: str, lexicon: Mapping[str, int], min_final_share: float = 0.8) -> Dict[str, str]:
    """Infer the mojibake-symbol -> final-letter mapping of one section from a lexicon.

    The specification's rule: every candidate symbol with >= 2 word-final uses is eligible when
    >= ``min_final_share`` of its letter-adjacent uses are word-final; each final letter is scored
    by the words that become lexicon words under that letter only; the best letter is accepted
    with >= 2 such words and >= 2x the runner-up; letters are assigned one-to-one in decreasing
    support (a symbol whose letter is taken stays unresolved). Run only on sections that
    :func:`is_finals_mojibake` flags.

    :param text: Raw section text.
    :param lexicon: Word -> count (any positive count means known).
    :param min_final_share: Minimum word-final share of a symbol's letter-adjacent uses.
    :return: Assigned symbols.
    """
    stems: Dict[str, List[str]] = collections.defaultdict(list)
    for m in FINAL_CANDIDATE_RE.finditer(text):
        stems[m.group(2)].append(m.group(1))
    scored = []
    for sym, st in stems.items():
        if len(st) < 2:
            continue
        final = len(re.findall(rf"(?<=[{HEB}]){re.escape(sym)}(?![{HEB}])", text))
        adjacent = len(re.findall(rf"(?<=[{HEB}]){re.escape(sym)}|{re.escape(sym)}(?=[{HEB}])", text))
        if not adjacent or final / adjacent < min_final_share:
            continue
        unique = {f: 0 for f in FINALS}
        hits = {f: 0 for f in FINALS}
        for s in st:
            ok = [f for f in FINALS if lexicon.get(s + f, 0) > 0]
            for f in ok:
                hits[f] += 1
            if len(ok) == 1:
                unique[ok[0]] += 1
        best = max(FINALS, key=lambda f: (unique[f], hits[f]))
        runner = sorted(unique.values(), reverse=True)[1]
        if unique[best] >= 2 and unique[best] >= 2 * runner:
            scored.append((unique[best], sym, best))
    mapping: Dict[str, str] = {}
    for _, sym, best in sorted(scored, key=lambda r: -r[0]):
        if best not in mapping.values():
            mapping[sym] = best
    return mapping


# ───────────────────────────── brackets ─────────────────────────────
def bracket_orientation(text: str, opener: str, closer: str) -> str:
    """Whether a bracket type is stored in logical or mirrored (visual) orientation.

    Each orientation is scored by the unbalanced brackets it leaves (closers at depth 0 plus
    openers never closed); the smaller count wins.

    :param text: Section text.
    :param opener: Logical opening character.
    :param closer: Logical closing character.
    :return: ``logical``, ``flipped``, ``tie`` or ``none`` (fewer than two brackets).
    """
    seq = [ch for ch in text if ch in (opener, closer)]
    if len(seq) < 2:
        return "none"

    def violations(o: str) -> int:
        """Unbalanced brackets when ``o`` opens.

        :param o: The opening character under this hypothesis.
        :return: Closers met at depth 0 plus openers never closed.
        """
        depth = bad = 0
        for ch in seq:
            if ch == o:
                depth += 1
            elif depth == 0:
                bad += 1
            else:
                depth -= 1
        return bad + depth

    a, b = violations(opener), violations(closer)
    return "logical" if a < b else "flipped" if b < a else "tie"


def normalise_brackets(text: str) -> Tuple[str, Dict[str, str]]:
    """Mirror back every bracket type whose orientation is ``flipped``.

    :param text: Section text.
    :return: (text, orientation per bracket type).
    """
    orient = {}
    for name, (o, c) in BRACKET_PAIRS.items():
        orient[name] = bracket_orientation(text, o, c)
        if orient[name] == "flipped":
            text = text.translate(str.maketrans({o: c, c: o}))
    return text, orient


# ───────────────────────────── structure ─────────────────────────────
def _leaf_number(token: Optional[str]) -> Optional[int]:
    """Leaf number of a heading token (digits or Hebrew numeral letters).

    :param token: ``"2"``, ``"ב"``, ``"יב"`` or None.
    :return: The number, or None.
    """
    if not token:
        return None
    if token.isdigit():
        return int(token)
    return sum(_GEMATRIA.get(ch, 0) for ch in token) or None


def _marker_label(m: "re.Match[str]") -> Tuple[Optional[int], Optional[str]]:
    """Leaf and side (``r``/``v``) a side-heading match states.

    :param m: Match of :data:`SIDE_MARKER_RE`.
    :return: (leaf or None, ``"r"``, ``"v"`` or None).
    """
    g = m.groupdict()
    kind = m.lastgroup   # the outer alternative closes last
    to_rv = {"א": "r", "1": "r", "ב": "v", "2": "v", "F": "r", "B": "v", "r": "r", "v": "v"}
    if kind == "daf":
        return _leaf_number(g["daf_leaf"]), to_rv[g["daf_side"]]
    if kind == "tsad":
        return None, to_rv[g["tsad_side"]]
    if kind == "amud":
        return int(g["amud_leaf"]), to_rv[g["amud_s1"] or g["amud_s2"]]
    if kind == "amud0":
        return None, to_rv[g["amud0_s1"] or g["amud0_s2"]]
    if kind == "amudc":
        return None, to_rv[g["amudc_s1"] or g["amudc_s2"]]
    if kind == "folio":
        return int(g["folio_leaf"]), g["folio_side"]
    if kind == "colon":
        return int(g["colon_leaf"]), to_rv[g["colon_side"]]
    if kind == "card":
        return None, to_rv[g["card_side"]]
    if kind == "rv":
        return None, "r" if g["rv_word"].lower() == "recto" else "v"
    if kind == "pcode":
        return None, to_rv[g["p_side"]]
    return None, None


def _side_boundaries(text: str) -> List[Tuple[int, int, Optional[str]]]:
    """Side headings of a section, grouped.

    Headings separated only by spaces, brackets or punctuation form one group (``דף א צד א
    (recto) (P2-F) צד שמאל``). A group opens a side when it holds a strong heading (``דף X צד Y``,
    ``צד א/ב``, ``ד1 ע"א`` / ``1 א" ע``, a leading ``ע"א``, an edition header's ``ע"ב )``,
    ``1r``, ``1א:``, ``213 א``, recto/verso); page codes and ``צד ימין/שמאל`` alone do not.

    :param text: Section text.
    :return: (start, end, label) per side-opening group; the label is ``"r"``/``"v"`` with the
        leaf number when known (``"2v"``).
    """
    matches = [m for m in SIDE_MARKER_RE.finditer(text)]
    groups: List[List["re.Match[str]"]] = []
    for m in matches:
        if groups and _MARKER_JOIN_RE.match(text[groups[-1][-1].end():m.start()]):
            groups[-1].append(m)
        else:
            groups.append([m])
    out = []
    for grp in groups:
        strong = [m for m in grp if m.lastgroup in _STRONG_SIDE_KINDS]
        if not strong:
            continue
        best = min(strong, key=lambda m: _STRONG_SIDE_KINDS.index(m.lastgroup))
        leaf, rv = _marker_label(best)
        if leaf is None:
            leaf = next((lf for lf, _ in (_marker_label(m) for m in strong) if lf is not None), None)
        end = grp[-1].end()
        tail = _MARKER_TAIL_RE.match(text[end:])
        if tail:
            end += tail.end()
        out.append((grp[0].start(), end, f"{leaf}{rv}" if leaf is not None else rv))
    return out


def _side_segments(text: str, heading_text: Optional[str] = None) -> List[_SideSegment]:
    """Cut a section at its side headings.

    :param text: Section text.
    :param heading_text: Text the headings are found in, character-aligned with ``text`` (the
        section before bracket mirroring, whose ``)`` after an edition header's ``ע"ב`` is still in
        place); None finds them in ``text``.
    :return: Segments in order; the first has side None and no marker when text precedes the first
        heading (or when there is no heading at all).
    :raises ValueError: When ``heading_text`` is not the same length as ``text``.
    """
    if heading_text is not None and len(heading_text) != len(text):
        raise ValueError("heading_text must be character-aligned with text")
    bounds = _side_boundaries(text if heading_text is None else heading_text)
    if not bounds:
        return [_SideSegment(None, "", text)]
    segs = [_SideSegment(None, "", text[:bounds[0][0]])]
    for i, (start, end, label) in enumerate(bounds):
        stop = bounds[i + 1][0] if i + 1 < len(bounds) else len(text)
        segs.append(_SideSegment(label, text[start:end], text[end:stop], "marker"))
    return segs


def split_sides(text: str) -> List[Tuple[Optional[str], str]]:
    """Split a flat section at its side headings.

    :param text: Section text (normalised).
    :return: ``(side, text)`` per segment, headings removed; text before the first heading comes
        first with side None (omitted when it is blank).
    """
    return [(s.side, s.text) for s in _side_segments(text) if s.side is not None or s.text.strip()]


def split_columns(text: str) -> List[Tuple[Optional[str], str, str]]:
    """Split one side at its column headings (``טור_א``, ``טור_ב``).

    :param text: Side text.
    :return: ``(column label, heading, text)`` per chunk; text before the first heading has label
        None and an empty heading.
    """
    out: List[Tuple[Optional[str], str, str]] = []
    pos, label, head = 0, None, ""
    for m in COLUMN_MARKER_RE.finditer(text):
        out.append((label, head, text[pos:m.start()]))
        label, head, pos = m.group(1) or m.group(2), m.group(0), m.end()
    out.append((label, head, text[pos:]))
    return [c for c in out if c[0] is not None or c[2].strip()]


# ───────────────────────────── marks ─────────────────────────────
def _paren_resolver(family: str, weiss: bool, acc: _Acc) -> Callable[["re.Match[str]"], str]:
    """Build the replacement function for single parentheses ``(x)``.

    ``ktiv_sigla``: deletion, the struck ink is kept. Weiss-format sections (lacunae written
    ``( )``): restorations become gaps, as in the July cleaner's ``_restoration``. Others: the July
    ``_expansion`` rule (expansions, glosses and supplied letters dropped; loss notes and dots ->
    gap); spans without Semitic letters (``(?)``, ``(!)``, numbers, Latin notes) are dropped; long
    or Latin-mixed spans keep their text and are flagged.

    :param family: Section family.
    :param weiss: Whether the section writes lacunae as empty parentheses.
    :param acc: Accumulator.
    :return: Function for ``PAREN_RE.sub``.
    """
    def resolve(m: "re.Match[str]") -> str:
        """Resolve one parenthesised span.

        :param m: Match of :data:`PAREN_RE`.
        :return: Replacement text.
        """
        inner = m.group(1)
        semitic = bool(LETTER_RE.search(inner))
        dots_only = bool(re.fullmatch(r"[\s.…·?]*\.[\s.…·?]*", inner))
        if family == "ktiv_sigla":
            acc.flags["deletion_kept"] += 1 if semitic else 0
            return inner
        if dots_only:
            return _GAP
        if weiss:
            if not inner.strip() or semitic:
                if len(inner) <= MAX_EDITORIAL_SPAN:
                    acc.recon_letters += len(letters_of(inner))
                    acc.drop("restoration", inner)
                    return _GAP
                acc.flags["long_paren_span"] += 1
                return f"{_GAP} {inner} {_GAP}"
            return " "
        if LOSS_NOTE_RE.search(inner):
            acc.drop("loss_note", inner)
            return f" {_GAP} "
        if not semitic:
            return " "
        if not LATIN_RE.search(inner) and len(inner) <= MAX_EDITORIAL_SPAN:
            acc.drop("expansion", inner)
            return ""
        acc.flags["long_paren_span"] += 1
        return f" {inner} "

    return resolve


def _square_resolver(family: str, acc: _Acc) -> Callable[["re.Match[str]"], str]:
    """Build the replacement function for single square brackets ``[x]``.

    ``ktiv_sigla``: margin text (KTIV 'j'), kept in reading order. Others: a restoration -> gap
    (letters counted); a span over :data:`MAX_EDITORIAL_SPAN` characters is two edge marks paired
    in flat text -> gap + text + gap.

    :param family: Section family.
    :param acc: Accumulator.
    :return: Function for ``SQUARE_RE.sub``.
    """
    def resolve(m: "re.Match[str]") -> str:
        """Resolve one bracketed span.

        :param m: Match of :data:`SQUARE_RE`.
        :return: Replacement text.
        """
        inner = m.group(1)
        if family == "ktiv_sigla":
            acc.flags["margin_kept"] += 1
            return f" {inner} "
        if len(inner) <= MAX_EDITORIAL_SPAN:
            acc.recon_letters += len(letters_of(inner))
            acc.drop("restoration", inner)
            return _GAP
        acc.flags["long_bracket_span"] += 1
        return f"{_GAP} {inner} {_GAP}"

    return resolve


def _sub_until_stable(pattern: "re.Pattern[str]", repl: Union[str, Callable[["re.Match[str]"], str]], text: str, rounds: int = 6) -> str:
    """Apply a substitution until nothing changes (nested pairs resolve innermost first).

    :param pattern: Pattern.
    :param repl: Replacement function or string.
    :param text: Text.
    :param rounds: Maximum passes.
    :return: Text.
    """
    for _ in range(rounds):
        new = pattern.sub(repl, text)
        if new == text:
            break
        text = new
    return text


def _drop_unpaired(text: str, chars: str, acc: _Acc) -> str:
    """Remove left-over bracket characters, flagging each.

    :param text: Text.
    :param chars: Bracket characters to remove.
    :param acc: Accumulator.
    :return: Text without them.
    """
    n = sum(text.count(c) for c in chars)
    if n:
        acc.flags["unmatched_bracket"] += n
        text = text.translate(str.maketrans({c: "" for c in chars}))
    return text


def _sequential_line_numbers(text: str, min_run: int = 3) -> Tuple[str, int]:
    """Turn bare per-line numbers (``1 ... 2 ... 3``) into line breaks.

    Standalone numbers are scanned in order for the sequence 1, 2, 3, ...; when at least
    ``min_run`` are found, each with letters (or a bracket) since the previous one, they are
    line numbers. Other numbers are left for the digit rule.

    :param text: Chunk text.
    :param min_run: Shortest sequence trusted.
    :return: (text, numbers turned into breaks).
    """
    hits = []
    expected, last_end = 1, 0
    for m in re.finditer(r"(?<![^\s\[\]])(\d{1,2})(?=[\s\[\]])", text):
        if int(m.group(1)) == expected and (expected == 1 or re.search(rf"[{HEB}\[\]]", text[last_end:m.start()])):
            hits.append(m)
            expected += 1
            last_end = m.end()
    if len(hits) < min_run:
        return text, 0
    for m in reversed(hits):
        text = text[:m.start()] + "\n" + text[m.end():]
    return text, len(hits)


def _dot_run(m: "re.Match[str]") -> str:
    """Gap for a dot run; a run ending in a displaced ``' .'`` keeps a word boundary after it.

    :param m: Match of :data:`DOT_RUN_RE`.
    :return: Gap placeholder.
    """
    return _GAP + " " if m.group(0).endswith(" .") else _GAP


def _doublet(m: "re.Match[str]") -> str:
    """Keep one copy of an ``A///B`` render doublet whose B is A reversed.

    B may already read as A when :func:`repair_reversed_fragments` re-ordered it.

    :param m: Match of :data:`DOUBLET_RE`.
    :return: Replacement text.
    """
    a, b = m.group(1), m.group(2)
    if b in (a, a[::-1]):
        return a + "\ue002" + b + "\ue002"
    return f"{a} {b}"


def clean_marks(text: str, family: str, weiss: bool, acc: _Acc) -> str:
    """Apply the mark table to one column chunk.

    :param text: Chunk text (orientation-normalised, finals repaired, controls stripped).
    :param family: Section family.
    :param weiss: Whether parentheses are restoration brackets (Weiss format).
    :param acc: Accumulator (counts, dropped spans, flags).
    :return: Cleaned text with :data:`_GAP` placeholders.
    """
    t = text
    n = t.count(_UNRESOLVED)
    if n:
        acc.flags["unresolved_finals"] += n
        t = t.replace(_UNRESOLVED, "")
    if family == "edition_apparatus":
        m = FOOTNOTE_RULE_RE.search(t)
        if m:
            acc.drop("apparatus", t[m.start():])
            acc.flags["apparatus_cut"] += 1
            t = t[:m.start()]
    acc.flags["editorial_terms"] += len(EDITORIAL_TERMS_RE.findall(t))
    # line numbers (structure)
    t, n = LINE_NUMBER_RE.subn("\n", t)
    if family != "edition_apparatus":
        t, k = _sequential_line_numbers(t)
        n += k
    acc.flags["explicit_lines"] += n
    t = EVERY_FIFTH_RE.sub(" ", t)
    # render doublets A///B (B = A reversed): the reversed copy is dropped
    t = DOUBLET_RE.sub(_doublet, t)
    for dup in re.findall("\ue002([^\ue002]*)\ue002", t):
        acc.drop("doublet", dup)
    t = re.sub("\ue002[^\ue002]*\ue002", "", t)
    # square-bracket family
    t = INLINE_CORRECTION_RE.sub(lambda m: m.group(1), t)
    t = INTERLINEAR_DOUBLE_RE.sub(lambda m: m.group(1), t)
    t = BLANK_BRACKETS_RE.sub(_GAP, t)
    # parentheses
    if family != "edition_apparatus":
        def _erasure(m: "re.Match[str]") -> str:
            """Drop one erasure (scraped letters) unless implausibly long.

            :param m: Match of :data:`ERASURE_RE`.
            :return: Replacement text.
            """
            if len(m.group(1)) <= MAX_EDITORIAL_SPAN:
                acc.drop("erasure", m.group(1))
                return ""
            acc.flags["long_erasure"] += 1
            return m.group(1)

        t = ERASURE_RE.sub(_erasure, t)
    t = _sub_until_stable(PAREN_RE, _paren_resolver(family, weiss, acc), t)
    t = _drop_unpaired(t, "()", acc)
    t = _sub_until_stable(SQUARE_RE, _square_resolver(family, acc), t)
    n = t.count("[") + t.count("]")
    if n:
        acc.flags["edge_bracket"] += n
        t = re.sub(r"[\[\]]", _GAP, t)
    t = _sub_until_stable(BRACE_RE, lambda m: m.group(1), t)
    t = _drop_unpaired(t, "{}", acc)
    t = EMPTY_ANGLE_RE.sub(" ", t)
    t = _sub_until_stable(ANGLE_RE, lambda m: m.group(1), t)
    t = _drop_unpaired(t, "<>", acc)
    # uncertainty and tag marks
    acc.uncertain_letters += sum(len(m.group(1)) for m in UNCERTAIN_WRAP_RE.finditer(t))
    acc.uncertain_letters += len(DOLLAR_UNCERTAIN_RE.findall(t))
    acc.flags["not_body_text"] += len(STAR_WRAP_RE.findall(t))
    # Latin, digits, dots, underscores
    n = sum(len(x) for x in LATIN_RE.findall(t))
    if n:
        acc.flags["latin_chars"] += n
        t = LATIN_RE.sub(" ", t)
    n = sum(len(x) for x in DIGITS_RE.findall(t))
    if n:
        acc.flags["digits"] += n
        t = DIGITS_RE.sub(" ", t)
    t = PIPE_RE.sub("", t)
    t = t.translate(str.maketrans({c: "" for c in _JOIN_MARKS}))
    t = t.translate(str.maketrans({c: " " for c in _SPACE_MARKS}))
    t = DOT_RUN_RE.sub(_dot_run, t)
    t = t.replace(".", " ")
    t = t.replace("·", ".")
    t = UNDERSCORE_RUN_RE.sub(_GAP, t).replace("_", " ")
    if family in ("ktiv_sigla", "edition_apparatus"):
        t = LEADING_GERESH_RE.sub(lambda m: m.group(1) + "'", t)
    t = re.sub(r"[ \t]+", " ", t)
    t, n = DETACHED_GERESH_RE.subn(lambda m: m.group(1), t)
    acc.flags["detached_geresh_joined"] += n
    other = [ch for ch in t if ch not in _KEEP_CHARS and not LETTER_RE.match(ch)]
    if other:
        acc.flags["other_symbol"] += len(other)
        t = "".join(ch if ch in _KEEP_CHARS or LETTER_RE.match(ch) else " " for ch in t)
    return _collapse(t)


# ───────────────────────────── section ─────────────────────────────
def _is_header(prefix: str, max_words: int = 15) -> bool:
    """Whether text before the first side heading is a catalogue or editor header.

    A long prefix is never a header, even with Latin in it: it is kept (its Latin is dropped by
    the mark rules) rather than risking a side of ink.

    :param prefix: The text.
    :param max_words: Most Hebrew words a catalogue header may hold.
    :return: True for Latin catalogue lines of at most ``max_words`` Hebrew words, numbered labels
        and one- or two-word labels.
    """
    words = HEB_RUN_RE.findall(prefix)
    latin = sum(len(x) for x in LATIN_RE.findall(prefix))
    return (len(words) <= 2 or (latin >= 3 and len(words) <= max_words)
            or (bool(DIGITS_RE.search(prefix)) and len(words) <= 3))


def clean_section(text: str, finals_mapping: Optional[Mapping[str, str]] = None, family: Optional[str] = None,
                  lexicon: Optional[Collection[str]] = None) -> List[CleanedSection]:
    """Clean one raw legacy-store section into one record per side.

    Order: Unicode clean-up; whole-section reversal; bracket orientation (per bracket type, on the
    whole section); side split (headings found in the text as it was before bracket mirroring, which
    is character-aligned with the mirrored text); then per side: finals repair (``finals_mapping``),
    unresolved finals noted, control characters removed, visual-order fragments re-ordered,
    multi-letter uncertain wraps re-ordered (``ktiv_sigla`` only, :func:`repair_uncertain_wraps`),
    column split, the mark table (:func:`clean_marks`) per column.

    Text before the first side heading is dropped as a header when it is a catalogue or editor
    label; otherwise it is kept, labelled as the recto of the leaf when the first heading opens
    that leaf's verso (``side_source="inferred"``), else with side None.

    :param text: Raw section text exactly as stored.
    :param finals_mapping: Symbol -> final letter for this section (see :data:`FINALS_MAPPING`).
    :param family: Convention family; computed by :func:`classify_family` when None.
    :param lexicon: Known words for :func:`repair_fragment_order` and :func:`repair_uncertain_wraps`
        (optional; without it neither changes anything).
    :return: Cleaned sides in section order (sides without letters are omitted; their dropped
        spans move to a neighbouring side).
    """
    family = family or classify_family(text)
    mojibake = is_finals_mojibake(text)
    t, points = normalise_text(text)
    t, rev = unreverse_if_needed(t)
    # Side headings are found before mirroring: in a flipped section the ")" that closes an edition
    # header's ע"ב would otherwise turn into "(" and hide the heading.
    heading_text = t
    t, orient = normalise_brackets(t)
    weiss = family != "ktiv_sigla" and bool(EMPTY_PARENS_RE.search(t))
    section_flags: Dict[str, int] = {}
    if orient["[]"] == "tie":
        section_flags["bracket_orientation_tie"] = 1
    if {"flipped", "logical"} <= set(orient.values()):
        section_flags["bracket_orientation_mixed"] = 1
    segs = _side_segments(t, heading_text)
    header_drop: List[Tuple[str, str]] = []
    if segs[0].side is None and len(segs) > 1:
        prefix = segs[0].text
        if not LETTER_RE.search(prefix):
            segs = segs[1:]
        elif _is_header(prefix):
            header_drop.append(("header", repair_finals(prefix, finals_mapping)[0].strip()))
            segs = segs[1:]
        elif segs[1].side and segs[1].side.endswith("v"):
            segs[0].side = segs[1].side[:-1] + "r"
            segs[0].source = "inferred"
    out: List[CleanedSection] = []
    pending: List[Tuple[str, str]] = []   # dropped spans of sides that produced no text
    for seg in segs:
        acc = _Acc()
        if header_drop:
            acc.dropped.extend(header_drop)
            acc.flags["header_cut"] += 1
            header_drop = []
        acc.drop("side_marker", seg.marker)
        body, n_fin = repair_finals(seg.text, finals_mapping)
        if mojibake:
            body = re.sub(rf"(?<=[{HEB}])[\x00-\x08\x0b-\x1f](?![{HEB}])", _UNRESOLVED, body)
        acc.flags["control_chars"] += len(CONTROL_RE.findall(body))
        body = strip_controls(body)
        body, n_frag = repair_reversed_fragments(body)
        body, n_join = repair_fragment_order(body, lexicon)
        n_wrap = 0
        if family == "ktiv_sigla":
            body, n_wrap = repair_uncertain_wraps(body, lexicon)
            acc.flags["wrap_reversed"] += n_wrap
        chunks = []
        labels = []
        for label, head, chunk in split_columns(body):
            acc.drop("column_marker", head)
            cleaned = clean_marks(chunk, family, weiss, acc)
            if label is not None:
                labels.append(label)
            if cleaned:
                chunks.append(cleaned)
        cleaned_text = "\n\n".join(chunks).replace(_GAP, GAP)
        if not LETTER_RE.search(cleaned_text):
            pending.extend(acc.dropped)
            continue
        acc.dropped[:0] = pending
        pending = []
        flags = {k: v for k, v in acc.flags.items() if v}
        flags.update(section_flags)
        if mojibake:
            flags["finals_mojibake"] = 1
        if points:
            flags["vocalised"] = points
        if weiss:
            flags["weiss_parens"] = 1
        if family in ("edition_apparatus", "empty"):
            flags[family] = 1
        if seg.source == "inferred":
            flags["side_inferred"] = 1
        elif seg.side is None and len(segs) > 1:
            flags["text_before_first_marker"] = 1
        out.append(CleanedSection(
            side=seg.side, column=",".join(labels) or None, text=cleaned_text,
            letters=len(letters_of(cleaned_text)), gaps=cleaned_text.count(GAP),
            recon_letters=acc.recon_letters, uncertain_letters=acc.uncertain_letters,
            finals_repaired=n_fin, reversed=rev, dropped_spans=acc.dropped, flags=flags, family=family,
            side_source=seg.source, fragments_repaired=n_frag + n_join + n_wrap))
    if pending and out:
        out[-1].dropped_spans.extend(pending)
    return out


def letter_audit(text: str, sides: Sequence[CleanedSection],
                 finals_mapping: Optional[Mapping[str, str]] = None) -> Tuple[Counter, Counter]:
    """Check that cleaning neither invented nor silently lost letters.

    Source letters = letters of the normalised raw text after finals repair. Every one must end up
    either in a side's text or in its ``dropped_spans``.

    :param text: Raw section text.
    :param sides: Output of :func:`clean_section` for it.
    :param finals_mapping: The mapping passed to :func:`clean_section`.
    :return: (invented letters, lost letters); both empty when the balance holds.
    """
    source, _ = normalise_text(text)
    source, _ = repair_finals(source, finals_mapping)
    have = collections.Counter(letters_of(source))
    got: Counter = collections.Counter()
    for s in sides:
        got.update(letters_of(s.text))
        for _, span in s.dropped_spans:
            got.update(letters_of(span))
    return got - have, have - got


# ───────────────────────────── finals table ─────────────────────────────
# Per-section mojibake finals from the 2026-10-30 survey (lexicon rule, simulated-mojibake
# validation 366/366 assignments correct): section_digest(raw text) -> {symbol: final letter}.
# Keys are content hashes, so the table names no document. The table was inferred with the
# survey's "plus" lexicon (the core clean-target + KTIV word list plus the words of legacy-store
# sections whose finals are intact); infer_finals_mapping run with the core lexicon alone differs
# on 6/95 mojibake sections (it leaves some pairs unassigned), so this table, not re-inference,
# is what the cleaner applies.
FINALS_MAPPING: Dict[str, Dict[str, str]] = {
    "0492d2765f23211a": {"\u001c": "ם", "\u001f": "ף"},
    "0547a8e97ada6c02": {"\u0006": "ך", "\u0019": "ם", "#": "ן", "$": "ץ"},
    "0741aa0a24e56cf5": {"\u0010": "ן", "\u001f": "ם"},
    "0d69f4b95356dbd3": {"\u0010": "ן", "%": "ם"},
    "19bcf68932f0f33e": {"\u0001": "ף", "\u001b": "ן"},
    "19da0d88da4bdb19": {"\u0014": "ף", "\u0015": "ן", "\u0016": "ם"},
    "1a04030ac2ed6ff1": {"\u000e": "ך", "\u0013": "ן"},
    "214d95b8c7a4e35c": {"\u0006": "ך", "\u0019": "ם", "#": "ן", "$": "ץ"},
    "2a1583488c1b1d62": {"\u001f": "ן"},
    "2a5212d966f526df": {"\t": "ן", "\u0019": "ם"},
    "2dd9df6b10be9740": {"\u0010": "ן", "\u001e": "ם"},
    "3124ea46ec594d3b": {"\u0001": "ף", "\u0012": "ן", "\u0016": "ם"},
    "34c7a79278824042": {"\u0011": "ן", "\u0019": "ך"},
    "34ef54268e6ebec3": {"\r": "ן", "\u0019": "ם"},
    "356bd446f2a3529b": {"\u000b": "ן", "\u0013": "ם", "\u001d": "ך"},
    "3b8c23ee5529a21d": {"\u000e": "ן", "\u001f": "ם", "&": "ך"},
    "4180e848dafa2156": {"\u001a": "ם"},
    "43c2905b7fd993d6": {"\u0010": "ן", "%": "ם", "+": "ך"},
    "4936d175d7ef408a": {"\t": "ן"},
    "4a3414ebe2545d2b": {"\u000e": "ך"},
    "4eeee519d81ee3a8": {"\u000f": "ם"},
    "535e694999bdb4bc": {"\u001a": "ן", "\u001c": "ך"},
    "53684bade4bacb26": {"\u0018": "ם", "\u001a": "ן", "\u001d": "ף"},
    "54c3f9e1cd0462fa": {"\u000e": "ן", "&": "ם"},
    "5e9c8c4101e81bbc": {"&": "ך"},
    "62b840c6e9ad3929": {"\u0011": "ן", "\u0019": "ם"},
    "6907a9d9fff8e6e6": {"\u000e": "ם", "\u0018": "ך", "\u001e": "ן", "%": "ף"},
    "6f0dbab0c25d10ce": {"\u0015": "ך", "\u0017": "ם", "\u001b": "ן", "\u001d": "ף", "\u001f": "ץ"},
    "6fc630e2017aa324": {"\u0010": "ן", "\u0012": "ם"},
    "71eca6ea23cff97a": {"\u000e": "ם", "\u0015": "ן"},
    "7b84932c2c67f023": {"\u0015": "ם", "\u0017": "ן", "\u001c": "ף", "\u001e": "ך"},
    "8394ba3fa0af698c": {"\u0010": "ך"},
    "85d328fc8b669f96": {"\u000b": "ן", "\u001a": "ם", "\u001d": "ף"},
    "89699e41fc128939": {"\u000e": "ן", "\u001f": "ם", "+": "ץ"},
    "9bd05423178f2ae0": {"\u000f": "ן"},
    "9ecbbdf06a7837f3": {"\u0018": "ן"},
    "a4937649b6d198cd": {"\u000b": "ם"},
    "ac3fb6b20cab3551": {"\u0019": "ם", "\u001e": "ן"},
    "ad320917da6f730d": {"\t": "ן", "\u001f": "ם"},
    "b0247327b272a4e0": {"\u0010": "ן", "\u001e": "ם", "#": "ך"},
    "b09024f4ba7750b9": {"\u001d": "ך", "\u001f": "ם"},
    "bda28e2a0150950b": {"\u0006": "ן"},
    "bece3ee98adad017": {"\u0018": "ם"},
    "cdc684e2931a576f": {"\u0001": "ף", "\u001a": "ן"},
    "ceb1e407ca514049": {"\u000f": "ן"},
    "d67aa30333d5b387": {"\u0001": "ף", "\u0013": "ם", "\u0016": "ן"},
    "d9bffbbd8914ea05": {"\u000e": "ם", "\u001c": "ן"},
    "ecddcfc5da25a43c": {"\u0014": "ם", "\u001e": "ן"},
    "fe95bad6996b5e6e": {"\u000f": "ן", "\u0016": "ם"},
    "fee591c99f3c8d21": {"\u0018": "ך"},
}


def finals_mapping_for(text: str) -> Dict[str, str]:
    """The mojibake-finals mapping of a raw section (empty when it has none).

    :param text: Raw section text exactly as stored.
    :return: Symbol -> final letter.
    """
    return dict(FINALS_MAPPING.get(section_digest(text), {}))
