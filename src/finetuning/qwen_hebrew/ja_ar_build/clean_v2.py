# File name: clean_v2.py
# Date: 11/12/26
# Author: Isaac Godfried. Coded by Claude.
"""Gate v2 target cleaning (round ja_ar_v1): the defects novel_build's per-source cleaners leave in a target.

Applied to the ``lines`` of every target side after :mod:`src.finetuning.qwen_hebrew.novel_build.targets` (spec
``logs/next_round/ja_ar_v1/build/GATE_V2_SPEC.md`` §6). Five rules, in this order; the first three change text, the last
two only flag. Every change is one ``clean_log`` entry ``{rule, sub, line_index, before, after, letters_removed}``
(``line_index`` = index of the line in the side as given; ``after`` is None for a dropped line); flags are
``{flag, line_index, out_index, ...}``. Nothing changes silently, and a rule that is unsure flags instead of deleting.

1. ``duplicate`` (:func:`drop_duplicates`): lines match when their letter skeletons are equal, or near-equal
   (:func:`rapidfuzz.fuzz.ratio` >= :data:`NEAR_RATIO`, both lines >= :data:`MIN_NEAR_LETTERS` letters). A run of >=
   :data:`MIN_BLOCK_LINES` consecutive lines (>= :data:`MIN_BLOCK_LETTERS` letters) repeating an earlier run line by
   line loses its later copy (``block``) when it has >= :data:`LONG_BLOCK_LINES` lines or when the side's repeated
   copies hold >= :data:`MIN_DUPLICATE_SHARE` of its letters (a text transcribed or exported twice); shorter blocks in
   a side under that share are only flagged ``repeated_block`` (court registers repeat a witness line and ``עד שני``,
   deeds a witness clause, papyri a basmala). A line of >= :data:`MIN_REPEAT_LETTERS` letters that repeats the line
   just above it is dropped (``doubled_line``) when the skeletons are equal, or near-equal in a side with
   :data:`SYSTEMATIC_DOUBLES` exact doubles (the legacy-store export that doubles lines); a lone near repeat (dated
   receipts, a call and its response, a work log's weekdays) and a repeat further down (formula lines of accounts) are
   only flagged ``repeated_line``.
2. ``editorial`` (:func:`drop_editorial`): English / Latin editorial notes, numeric glosses and labels. Sub-rules:
   ``footnote_line`` (a line opened by a side + line anchor ``r20`` / ``v11``: an editor's footnote), ``apparatus_tail``
   (``הערות :`` / ``הערות שורה :`` and the rest of the line: the legacy store's apparatus), ``bracket_note`` (a
   ``(...)`` or ``[...]`` span with Latin letters or digits and no Hebrew / Arabic letter: ``(margin)``, ``[sic]``,
   ``(+1,234)``, ``(1)``, ``(Fol. 2 recto.)``; ``(!)`` / ``(?)`` marks; folio spans ``(12ב)``; a span with a loss
   word, also reversed as RTL exports store it (``nrot`` = torn), or with Hebrew / Arabic letters and a conjecture
   word (``may be``), becomes ``[...]``), ``latin_line`` (a line with Latin letters or ASCII digits and no Hebrew / Arabic letter; a
   number-only line with a fraction is an amount, kept and flagged ``numeral_line``), ``rule_line`` (``__`` runs drawn
   for ruled lines), ``latin_run`` (inline Latin words: a run of >= 3, or one holding an editorial word or folio label
   such as ``nigram:`` or ``1r``), ``footnote_anchor`` (1-3 ASCII digits glued to the end of a word of >= 2 letters or
   to a closing bracket; not a fraction, not before a currency / weight word), ``line_number`` (bare integers that count
   lines: a chain of >= 3 consecutive integers in reading order, at most one line apart, or an every-fifth chain at line
   starts). Never dropped: Hebrew or Arabic letters outside the two Hebrew-note sub-rules (so an Arabic-script line in
   a Hebrew-script target, Hebrew-script lines in an Arabic one and Hebrew letter numerals stay), Arabic-Indic digits
   (numerals on the page). Flags: ``latin_inline`` (one or two unknown Latin words: a sign written ``X``, a Latin
   signature), ``latin_glued`` (Latin glued to Hebrew letters, ``אvב``), ``bracket_mixed_latin``, ``glued_digits``
   (digits inside a word, after one letter or before a unit word), ``hebrew_editorial_candidate`` (editor phrases such
   as ``השלמת המהדיר`` or ``, הכוונה``).
3. ``restoration`` (:func:`render_restorations`): restorations as ``[...]`` only where the source's convention is
   clear. Measured on the 5,270 stage-1 sides (2026-11-12): pgp, fjp, fjp_scrape, ktiv, apd_v3 and aodeh_v2_1 hold no
   square span but ``[...]`` (their cleaners already render restorations), so nothing is converted there and their
   leftover parentheses are flagged ``paren_unresolved``. literature and lit_run2 (scanned and web editions, no
   upstream cleaner) keep the editions' sigla. A side with an empty ``( )`` is Weiss format (``( )`` lacuna, ``(x)``
   restoration; the :mod:`src.datasets.cleaning.clean_genizah_transcriptions` rule): its paired parentheses of <=
   :data:`MAX_RESTORATION_SPAN` characters become ``[...]`` (``weiss_paren``), and so does the run from an unpaired
   parenthesis to the torn line edge when it has <= :data:`WEISS_EDGE_MAX` characters and crosses no lacuna
   (``weiss_edge``); its square brackets are left (they do not mark restorations there). Any other side is Leiden style: paired ``[x]`` with letters or only dots / ``?``
   (``square``), dot-only parentheses (``dots``) and a bracket at the very edge of a line (``edge``, read as
   ``clean_diplomatic`` reads it) become ``[...]``. Editions whose OCR mirrors brackets (:func:`mirrored_works`; there
   ``[x]`` may be ink between two lacunae) and sides with a mirrored pair inside a line are left whole (flag
   ``brackets_mirrored``). Left and flagged: ``[[x]]`` (deletion or interlinear), ``⟨x⟩`` / ``<x>`` / ``{x}`` (meaning
   differs by edition), non-Weiss ``(x)`` (expansion, gloss or restoration), unpaired brackets inside a line.
4. ``flattened`` (:func:`flag_flattened`): a line of more than :data:`FLAT_LINE_LETTERS` letters is flagged
   ``flattened`` (``side_flat`` when the side is one such line or its median line is that long); the gate re-lines.
5. ``scrambled`` (:func:`flag_scrambled`): per-letter log2-likelihood of a line under a character 3-gram model of its
   script (:class:`Char3Model`, Witten-Bell interpolation, trained on all targets after rules 1-3; numerals and
   abbreviations such as ``ק'ג'`` only split the text, since their letter order carries no signal) below the model's
   :data:`SCRAMBLED_PERCENTILE` threshold flags ``scrambled_candidate`` with the score (``basis: line``; ``reversed``
   when the line read backwards scores much better). A flattened line is also scored in word windows and flagged when
   its worst window is under the threshold and well under its own median window (``basis: window``, the token range
   given): scrambling at damaged line ends is local. The gate excludes such a line only when it also aligns worse than
   both neighbours.

Usage (dry run over the novel_v1 stage-1 sides; trains and saves the models, writes the two reports)::

    PYTHONPATH=. .venv/bin/python -m src.finetuning.qwen_hebrew.ja_ar_build.clean_v2 \\
        --stage1 /Volumes/home/studio_offload/datasets/genizah_novel_v1_work/stages/stage1_images.jsonl \\
        --models-dir /Volumes/home/studio_offload/datasets/genizah_ja_ar_v1_work/models \\
        --report-dir logs/next_round/ja_ar_v1/build \\
        --audit-key logs/next_round/ja_ar_v1/visual_audit/judge_key_REVEALED.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from rapidfuzz import fuzz

from src.finetuning.qwen_hebrew.ja_ar_build.textnorm import skeleton as _textnorm_skeleton

# ----------------------------------------------------------------------------------------------------------- constants

GAP = "[...]"
_NEW_GAP = "\ue000"   # a gap this module writes (merged with its neighbours at the end of a rule)
_OLD_GAP = "\ue001"   # a gap the input already had (protected from bracket rules)

RULE_DUPLICATE = "duplicate"
RULE_EDITORIAL = "editorial"
RULE_RESTORATION = "restoration"
RULE_FLATTENED = "flattened"
RULE_SCRAMBLED = "scrambled"
RULES = (RULE_DUPLICATE, RULE_EDITORIAL, RULE_RESTORATION, RULE_FLATTENED, RULE_SCRAMBLED)

#: duplicates: near-exact match of two line skeletons (rapidfuzz ratio, 0-100)
NEAR_RATIO = 90.0
#: duplicates: a repeated block has at least this many consecutive lines ...
MIN_BLOCK_LINES = 2
#: ... and its later copy at least this many letters
MIN_BLOCK_LETTERS = 20
#: duplicates: a block this long is dropped whatever the side's repeated share (an exported text repeated, pgp:7572)
LONG_BLOCK_LINES = 4
#: duplicates: shorter blocks are dropped only when the later copies hold this share of the side's letters (a text
#: transcribed twice); below it they are signature lines, witness clauses or formula pairs of a register: flagged
MIN_DUPLICATE_SHARE = 0.20
#: duplicates: exact adjacent repeats that make a side's doubling systematic (then near adjacent repeats drop too)
SYSTEMATIC_DOUBLES = 2
#: duplicates: a single repeated line needs this many letters
MIN_REPEAT_LETTERS = 20
#: duplicates: a line with fewer letters never matches
MIN_MATCH_LETTERS = 3
#: duplicates: a near (non-exact) match needs this many letters on both lines
MIN_NEAR_LETTERS = 8

#: restorations: longest parenthesised span read as one restoration (``clean_genizah_transcriptions``' limit)
MAX_RESTORATION_SPAN = 40
#: restorations: longest Weiss run from an unpaired parenthesis to the line edge (328 of 351 such runs on the data are
#: empty, 13 are <= 10 characters; the longer ones are real text with a stray parenthesis)
WEISS_EDGE_MAX = 12
#: sources whose targets keep an edition's own sigla (no upstream cleaner renders their restorations)
RESTORATION_SOURCES = frozenset({"literature", "lit_run2"})

#: flattened: a line with more letters than this is not a manuscript line
FLAT_LINE_LETTERS = 120

#: scrambled: lines with fewer scored letters get no score
MIN_SCORE_LETTERS = 12
#: scrambled: percentile of all scored lines of a script under which a line is flagged
SCRAMBLED_PERCENTILE = 1.0
#: scrambled: word windows of a flattened line (tokens per window, step)
WINDOW_WORDS = 8
WINDOW_STEP = 4
#: scrambled: a flattened line's worst window must lie this far (log2 per letter) under the line's median window
WINDOW_DROP = 1.0
#: scrambled: a flagged line read backwards scoring this much better is marked ``reversed`` (text stored in visual
#: order); a hint for the gate, nothing is reversed here
REVERSED_GAIN = 1.0
MODEL_FORMAT = "char3_witten_bell_v1"
LM_PAD = "#"
LM_GERESH = "'"

_LATIN_LETTERS = "A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u024F\u1E00-\u1EFF"
LATIN_RE = re.compile(f"[{_LATIN_LETTERS}]")
_LATIN_WORD_RE = re.compile(f"[{_LATIN_LETTERS}]+")
#: Hebrew and Arabic letters, presentation forms included (a line holding any of them is text, never a note)
_SEMITIC_CLASS = ("\u05D0-\u05EA\u05F0-\u05F2\uFB1D-\uFB4F\u0620-\u063F\u0641-\u064A\u066E\u066F\u0671-\u06D3"
                  "\u06D5\u06FA-\u06FC\u06FF\uFB50-\uFDFF\uFE70-\uFEFC")
SEMITIC_RE = re.compile(f"[{_SEMITIC_CLASS}]")
#: ASCII digits only: Arabic-Indic digits are numerals written on the page
ASCII_DIGIT_RE = re.compile("[0-9]")
_PRESENTATION_RE = re.compile("[\uFB1D-\uFB4F\uFB50-\uFDFF\uFE70-\uFEFC]")
_HEB_LETTER_RE = re.compile("[\u05D0-\u05EA]")
_WS_RE = re.compile(r"\s+")

#: Latin words that mark a span or token as an editor's (also matched reversed, as RTL exports store them)
EDITORIAL_WORDS = frozenset("""
margin margins marginal recto verso fol fols folio folios sic cf ibid idem col cols column columns line lines page pages
side top bottom upside sideways interlinear above below address seal signature signed hand handwriting writing written
read reads reading see note notes gloss lit viz scil continues continued continuation end text torn missing illegible
lost lacuna erased deleted crossed damaged blank hole faded unclear perhaps probably possibly may maybe
""".split())
#: Latin words that say text is lost: a bracket span holding one becomes ``[...]``
LOSS_WORDS = frozenset("""
torn tear missing illegible lost lacuna gap damaged damage erased erasure effaced blank hole broken cut faded rubbed
unclear unreadable obliterated destroyed wanting
""".split())
LOSS_PHRASES = ("no reading", "cut off")
#: Latin words that introduce an editor's conjecture: a span holding one AND Hebrew / Arabic letters (the conjectured
#: text) becomes ``[...]``; a Latin-only span with one is only dropped (the ink before it is there, just uncertain)
CONJECTURE_WORDS = frozenset("perhaps probably possibly maybe".split())
CONJECTURE_PHRASES = ("may be", "read as")

_PAREN_SPAN_RE = re.compile(r"\(([^()\n]*)\)")
_SQUARE_SPAN_RE = re.compile(r"\[([^\[\]\n]*)\]")
_MARKS_ONLY_RE = re.compile(r"[\s!?\u061F]*[!?\u061F][\s!?\u061F]*")
_FOLIO_SPAN_RE = re.compile(r"\s*(?:[0-9]{1,4}\s*(?:[אב]|ע[\"״']?[אב]|[abrv])|[rv]\s*[0-9]{1,4})\s*\.?\s*")
_FOLIO_TOKEN_RE = re.compile(r"[0-9]{1,3}[rvab]|[rv][0-9]{1,3}")
_FOOTNOTE_LINE_RE = re.compile(r"^\s*[rv][0-9]{1,3}\s+(?=[\u05D0-\u05EA])")
_APPARATUS_RE = re.compile(r"(?<![\u05D0-\u05EA])הערות\s*(?::|שורה\s*:)")
_RULE_LINE_RE = re.compile(r"_{2,}")
_HEB_EDITORIAL_RE = re.compile(r"השלמת המהדיר|הושלם עפ|יש להעיר|קריאה מסופקת|,\s*הכוונה(?![א-ת])|לכאורה חסרה|"
                               r"סימני תנועות|מעל השורה")
_WORD_MARKS = "\u0591-\u05C7\u064B-\u065F\u0670'׳\"״"
#: digits glued to the end of a word (a footnote anchor), not a fraction (``3/4``), a decimal or the inside of a word
_ANCHOR_RE = re.compile(f"(?P<word>[{_SEMITIC_CLASS}{_WORD_MARKS}]+)(?P<num>[0-9]{{1,3}})"
                        f"(?![0-9]|[/.,:][0-9]|[{_SEMITIC_CLASS}])")
#: currency, weight and count words (letter skeletons): digits before one of them are an amount, not a footnote anchor
UNIT_SKELETONS = frozenset(_textnorm_skeleton(w) for w in """
דינאר דינר דינ דנאניר דנא דינרין דינארא דרהם דראהם דרה דרהמא דרהמין רטל ארטאל רטלין קיראט קיראטין קירט קירטין
קרט מתקאל מתאקיל אוקיה אואק ויבה ארדב זוז זוזים זהובים גרושוש מאידיש פלס פלוס כסף זהב
دينار دنانير درهم دراهم رطل ارطال قيراط مثقال
""".split())
#: a numeral-only line holding a fraction is an amount (kept, flagged ``numeral_line``)
_FRACTION_RE = re.compile(r"[½¼¾⅛⅓⅔]|[0-9]\s*/\s*[0-9]")
#: digits glued to a closing bracket or parenthesis (``[שלום]14``), not to the gap token
_BRACKET_ANCHOR_RE = re.compile(f"(?<=[^.\\]]\\]|.\\))(?P<num>[0-9]{{1,3}})(?![0-9]|[/.,:][0-9]|[{_SEMITIC_CLASS}])")
_GLUED_INSIDE_RE = re.compile(f"[{_SEMITIC_CLASS}][0-9]{{1,3}}[{_SEMITIC_CLASS}]"
                              f"|(?<![0-9])[0-9]{{1,3}}[{_SEMITIC_CLASS}]")
_NUMBER_TOKEN_RE = re.compile(r"\(?0*([0-9]{1,3})[.):]?\)?")
_TOKEN_RE = re.compile(r"\S+")
_EMPTY_PAREN_RE = re.compile(r"\(\s*\)")
_GAP_RUN_RE = re.compile(f"(?:{_OLD_GAP}\\s*)*{_NEW_GAP}(?:\\s*[{_NEW_GAP}{_OLD_GAP}])*")
_DOTS_ONLY_RE = re.compile(r"[\s.…·?\-–_]*")
_GERESH_CHARS = frozenset("'׳״\"`’‘")
#: characters that separate words for the 3-gram model (maqaf, dashes, slashes, pipes)
_WORD_SEPARATORS = frozenset("־-–—/|\\")
_ARABIC_FOLD = str.maketrans({"\u0623": "\u0627", "\u0625": "\u0627", "\u0622": "\u0627", "\u0671": "\u0627",
                              "\u0649": "\u064A", "\u0629": "\u0647", "\u0624": "\u0648", "\u0626": "\u064A"})


# ------------------------------------------------------------------------------------------------------------- results

@dataclass
class RuleOutput:
    """What one rule did to a side's lines.

    :param lines: The lines after the rule.
    :param kept: For each output line, the index of the input line it comes from.
    :param log: Change entries, ``line_index`` = index in the rule's input.
    :param flags: Flag entries, ``line_index`` = index in the rule's input.
    """

    lines: List[str]
    kept: List[int]
    log: List[Dict[str, Any]] = field(default_factory=list)
    flags: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CleanResult:
    """A cleaned target side.

    :param lines: Cleaned lines.
    :param clean_log: One entry per change, ``line_index`` = index of the line in the side as given.
    :param flags: Flag entries, ``line_index`` = index in the side as given, ``out_index`` = index in ``lines`` (None
        when a later rule dropped the line).
    :param origin: For each cleaned line, the index of the input line it comes from.
    """

    lines: List[str]
    clean_log: List[Dict[str, Any]] = field(default_factory=list)
    flags: List[Dict[str, Any]] = field(default_factory=list)
    origin: List[int] = field(default_factory=list)

    @property
    def flag_codes(self) -> List[str]:
        """Distinct flag codes of the side.

        :return: Sorted codes.
        """
        return sorted({f["flag"] for f in self.flags})

    @property
    def letters_removed(self) -> int:
        """Letters removed by all changes.

        :return: Letter count.
        """
        return sum(int(e.get("letters_removed") or 0) for e in self.clean_log)


# ---------------------------------------------------------------------------------------------------------------- text

def unpresent(text: str) -> str:
    """Replace Hebrew and Arabic presentation forms by their NFKC letters (``ﬠ`` -> ``ע``, ``ﻻ`` -> ``لا``).

    :param text: Any text.
    :return: The text with presentation forms expanded, everything else unchanged.
    """
    return _PRESENTATION_RE.sub(lambda m: unicodedata.normalize("NFKC", m.group(0)), text)


def skel(text: str) -> str:
    """Letter skeleton of a text (:func:`textnorm.skeleton`, presentation forms expanded first).

    :param text: Any text.
    :return: One character per Hebrew or Arabic letter, finals and Arabic letter variants folded.
    """
    return _textnorm_skeleton(unpresent(text))


def letters(text: Optional[str]) -> int:
    """Letters of a text (length of :func:`skel`).

    :param text: Any text, or None.
    :return: Letter count (0 for None).
    """
    return len(skel(text)) if text else 0


def has_semitic(text: str) -> bool:
    """Whether a text holds a Hebrew or Arabic letter (presentation forms included).

    :param text: Any text.
    :return: True when a letter is present.
    """
    return SEMITIC_RE.search(text) is not None


def tidy(text: str) -> str:
    """Collapse whitespace runs to one space and strip (applied only to lines a rule edited).

    :param text: Edited line.
    :return: The line.
    """
    return _WS_RE.sub(" ", text).strip()


def change(rule: str, sub: str, index: int, before: str, after: Optional[str], **extra: Any) -> Dict[str, Any]:
    """One ``clean_log`` entry.

    :param rule: Rule name.
    :param sub: Sub-rule name.
    :param index: Index of the line in the rule's input.
    :param before: The line before the change.
    :param after: The line after it, or None when the line is dropped.
    :param extra: Further fields (``removed`` spans, the duplicated line, ...).
    :return: The entry.
    """
    entry: Dict[str, Any] = {"rule": rule, "sub": sub, "line_index": index, "before": before, "after": after,
                             "letters_removed": letters(before) - letters(after)}
    entry.update(extra)
    return entry


def flag(code: str, index: Optional[int], **extra: Any) -> Dict[str, Any]:
    """One flag entry.

    :param code: Flag code.
    :param index: Index of the line in the rule's input (None for a side-level flag).
    :param extra: Further fields (score, letters, ...).
    :return: The entry.
    """
    out: Dict[str, Any] = {"flag": code, "line_index": index}
    out.update(extra)
    return out


def _latin_words(text: str) -> List[str]:
    """Latin words of a text, lower-cased.

    :param text: Any text.
    :return: Words in order.
    """
    return [w.lower() for w in _LATIN_WORD_RE.findall(text)]


def has_loss_word(text: str, conjecture: bool = False) -> bool:
    """Whether a text's Latin words say that text is lost (also read reversed, as RTL exports store them: ``nrot``).

    :param text: Bracket content or token run.
    :param conjecture: Also count conjecture words and phrases (``perhaps``, ``may be`` / reversed ``:eb yam``).
    :return: True for a word of :data:`LOSS_WORDS` or a phrase of :data:`LOSS_PHRASES` (plus
        :data:`CONJECTURE_WORDS` / :data:`CONJECTURE_PHRASES` when ``conjecture``).
    """
    words = _latin_words(text)
    vocab = LOSS_WORDS | CONJECTURE_WORDS if conjecture else LOSS_WORDS
    if any(w in vocab or w[::-1] in vocab for w in words):
        return True
    phrases = LOSS_PHRASES + (CONJECTURE_PHRASES if conjecture else ())
    forward = " ".join(words)
    backward = " ".join(w[::-1] for w in reversed(words))
    return any(p in forward or p in backward for p in phrases)


def is_editorial_word(token: str) -> bool:
    """Whether a Latin token is an editor's word or a folio / side label (``margin``, ``nigram:``, ``1r``, ``v11``).

    :param token: One whitespace token.
    :return: True when it names an editorial category.
    """
    core = token.strip("()[]{}<>:;.,!?/\\'\"-–")
    if _FOLIO_TOKEN_RE.fullmatch(core):
        return True
    word = "".join(_LATIN_WORD_RE.findall(core)).lower()
    return bool(word) and (word in EDITORIAL_WORDS or word[::-1] in EDITORIAL_WORDS)


# -------------------------------------------------------------------------------------------------------- 1 duplicates

def lines_match(a: str, b: str, near: float = NEAR_RATIO) -> bool:
    """Whether two line skeletons are the same line (exact, or near-exact when both are long enough).

    :param a: Skeleton of one line.
    :param b: Skeleton of the other.
    :param near: Minimum :func:`rapidfuzz.fuzz.ratio` of a near match.
    :return: True for a match.
    """
    la, lb = len(a), len(b)
    if la < MIN_MATCH_LETTERS or lb < MIN_MATCH_LETTERS:
        return False
    if a == b:
        return True
    if min(la, lb) < MIN_NEAR_LETTERS or 200.0 * min(la, lb) / (la + lb) < near:
        return False
    return fuzz.ratio(a, b) >= near


def repeated_blocks(sk: Sequence[str], near: float = NEAR_RATIO, min_block_lines: int = MIN_BLOCK_LINES,
                    min_block_letters: int = MIN_BLOCK_LETTERS) -> List[Tuple[int, int, int]]:
    """Runs of lines that repeat an earlier run line by line.

    Scanning down, the longest run starting at line ``j`` that matches (:func:`lines_match`) a run of earlier lines
    ending before ``j`` is a block when it has ``min_block_lines`` lines and ``min_block_letters`` letters; the scan
    continues after it.

    :param sk: Line skeletons.
    :param near: Near-match ratio.
    :param min_block_lines: Shortest block.
    :param min_block_letters: Fewest letters of the later copy.
    :return: ``(start of the later copy, lines, start of the earlier copy)`` per block.
    """
    n = len(sk)
    blocks: List[Tuple[int, int, int]] = []
    j = 1
    while j < n:
        best_k, best_i = 0, -1
        for i in range(j):
            k = 0
            while j + k < n and i + k < j and lines_match(sk[i + k], sk[j + k], near):
                k += 1
            if k > best_k:
                best_k, best_i = k, i
        if best_k >= min_block_lines and sum(len(sk[j + t]) for t in range(best_k)) >= min_block_letters:
            blocks.append((j, best_k, best_i))
            j += best_k
            continue
        j += 1
    return blocks


def drop_duplicates(lines: Sequence[str], *, near: float = NEAR_RATIO, min_block_lines: int = MIN_BLOCK_LINES,
                    min_block_letters: int = MIN_BLOCK_LETTERS, long_block_lines: int = LONG_BLOCK_LINES,
                    min_share: float = MIN_DUPLICATE_SHARE, min_repeat_letters: int = MIN_REPEAT_LETTERS,
                    systematic: int = SYSTEMATIC_DOUBLES) -> RuleOutput:
    """Drop repeated blocks and doubled lines; flag repeats that may be real text.

    Blocks (:func:`repeated_blocks`) lose their later copy (``block``) when they have ``long_block_lines`` lines or
    when all later copies together hold ``min_share`` of the side's letters (a text transcribed or exported twice);
    otherwise every block is kept and flagged ``repeated_block`` (registers repeat signature lines and witness clauses).
    A line of ``min_repeat_letters`` letters outside a block that repeats the nearest kept line above it is dropped
    (``doubled_line``) when the skeletons are equal, or when they are near-equal and the side has ``systematic`` exact
    doubles; otherwise flagged ``repeated_line`` (``adjacent``). A line repeating any other earlier line is flagged
    ``repeated_line`` (accounts and registers repeat formula lines).

    :param lines: Lines of a side.
    :param near: Near-match ratio.
    :param min_block_lines: Shortest block.
    :param min_block_letters: Fewest letters of a block's later copy.
    :param long_block_lines: Block length dropped regardless of share.
    :param min_share: Repeated share of the side's letters above which every block is dropped.
    :param min_repeat_letters: Fewest letters of a single repeated line.
    :param systematic: Exact adjacent doubles that make near doubles droppable.
    :return: The kept lines; log entries carry ``copy_of`` (index of the earlier line).
    """
    sk = [skel(line) for line in lines]
    n = len(lines)
    blocks = repeated_blocks(sk, near, min_block_lines, min_block_letters)
    side_letters = sum(len(s) for s in sk) or 1
    share = sum(len(sk[j + t]) for j, k, _ in blocks for t in range(k)) / side_letters
    dropped: Dict[int, Tuple[str, int, Dict[str, Any]]] = {}
    flags: List[Dict[str, Any]] = []
    in_block: Set[int] = set()
    for j, k, i in blocks:
        in_block.update(range(j, j + k))
        info = {"block": [i, i + k - 1], "block_lines": k, "share": round(share, 3)}
        if k >= long_block_lines or share >= min_share:
            for t in range(k):
                dropped[j + t] = ("block", i + t, info)
        else:
            flags.append(flag("repeated_block", j, copy_of=i, block_lines=k, share=round(share, 3)))
    singles: List[Tuple[int, int, bool]] = []
    for j in range(1, n):
        if j in in_block or len(sk[j]) < min_repeat_letters:
            continue
        prev = j - 1
        while prev >= 0 and prev in dropped:
            prev -= 1
        if prev >= 0 and lines_match(sk[prev], sk[j], near):
            singles.append((j, prev, sk[prev] == sk[j]))
            continue
        earlier = next((i for i in range(j - 1) if lines_match(sk[i], sk[j], near)), None)
        if earlier is not None:
            flags.append(flag("repeated_line", j, copy_of=earlier, letters=len(sk[j]), adjacent=False))
    exact_doubles = sum(1 for _, _, exact in singles if exact)
    for j, prev, exact in singles:
        if exact or exact_doubles >= systematic:
            dropped[j] = ("doubled_line", prev, {"exact": exact})
        else:
            flags.append(flag("repeated_line", j, copy_of=prev, letters=len(sk[j]), adjacent=True))
    log = [change(RULE_DUPLICATE, sub, j, lines[j], None, copy_of=i, **extra)
           for j, (sub, i, extra) in sorted(dropped.items())]
    kept = [i for i in range(n) if i not in dropped]
    return RuleOutput([lines[i] for i in kept], kept, log, flags)


# --------------------------------------------------------------------------------------------------------- 2 editorial

def span_kind(inner: str, bracket: str) -> str:
    """What a bracket span's content is, for the editorial rule.

    :param inner: Text between the brackets.
    :param bracket: ``(`` or ``[``.
    :return: ``marks`` (only ``!`` / ``?``: a sic or uncertainty mark; for ``[?]`` only when it holds a ``!``),
        ``folio`` (``12ב``, ``2r``), ``loss`` (Latin or digits and a loss word), ``editorial`` (Latin letters or digits,
        no Hebrew / Arabic letter), ``mixed_loss`` / ``mixed`` (letters and Latin words, with or without a loss word),
        or ``text`` (anything else, left to the restoration rule).
    """
    sem, lat, dig = has_semitic(inner), LATIN_RE.search(inner) is not None, ASCII_DIGIT_RE.search(inner) is not None
    if not (sem or lat or dig):
        if _MARKS_ONLY_RE.fullmatch(inner) and (bracket == "(" or "!" in inner):
            return "marks"
        return "text"
    if _FOLIO_SPAN_RE.fullmatch(inner):
        return "folio"
    if not sem:
        return "loss" if has_loss_word(inner) else "editorial"
    if lat:
        return "mixed_loss" if has_loss_word(inner, conjecture=True) else "mixed"
    return "text"


def bracket_notes(line: str) -> Tuple[str, List[str], List[Tuple[int, int]], List[str]]:
    """Remove editorial bracket spans from one line (sub-rule ``bracket_note``).

    Parenthesised spans are resolved first, then square ones (so ``[x(?)]`` keeps its ``[x]`` for the restoration
    rule); the ``[...]`` gap token is never a span.

    :param line: One line.
    :return: ``(line, removed spans, protected character ranges of mixed spans in the returned line, flag codes)``.
    """
    removed: List[str] = []
    codes: List[str] = []
    text = line.replace(GAP, _OLD_GAP)
    for pattern, bracket in ((_PAREN_SPAN_RE, "("), (_SQUARE_SPAN_RE, "[")):
        def _resolve(match: "re.Match[str]") -> str:
            """Replacement of one span.

            :param match: Span match.
            :return: Replacement text.
            """
            kind = span_kind(match.group(1), bracket)
            if kind in ("marks", "folio", "editorial"):
                removed.append(match.group(0))
                return " " if kind != "marks" else ""
            if kind in ("loss", "mixed_loss"):
                removed.append(match.group(0))
                return f" {_NEW_GAP} "
            if kind == "mixed":
                codes.append("bracket_mixed_latin")
            return match.group(0)

        text = pattern.sub(_resolve, text)
    if removed:
        text = _GAP_RUN_RE.sub(GAP, text)
    out = text.replace(_OLD_GAP, GAP).replace(_NEW_GAP, GAP)
    out = tidy(out) if removed else out
    protected = [(m.start(), m.end()) for pattern in (_PAREN_SPAN_RE, _SQUARE_SPAN_RE) for m in pattern.finditer(out)
                 if has_semitic(m.group(1)) and LATIN_RE.search(m.group(1))]
    return out, removed, protected, codes


def is_latin_line(line: str) -> bool:
    """Whether a line is an editor's note or number: Latin letters or ASCII digits and no Hebrew / Arabic letter.

    :param line: One line.
    :return: True for a note line.
    """
    rest = line.replace(GAP, " ")
    return not has_semitic(rest) and (LATIN_RE.search(rest) is not None or ASCII_DIGIT_RE.search(rest) is not None)


def _token_kind(token: str) -> str:
    """Kind of a whitespace token for :func:`latin_runs`.

    :param token: Token.
    :return: ``mixed`` (Latin and Hebrew / Arabic letters), ``sem``, ``lat``, ``num`` (ASCII digits, no letters),
        ``gap`` or ``punct``.
    """
    sem, lat = has_semitic(token), LATIN_RE.search(token) is not None
    if sem and lat:
        return "mixed"
    if sem:
        return "sem"
    if lat:
        return "lat"
    if ASCII_DIGIT_RE.search(token):
        return "num"
    return "gap" if GAP in token else "punct"


def latin_runs(line: str, protected: Sequence[Tuple[int, int]] = ()) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    """Drop inline Latin word runs that are editorial (sub-rule ``latin_run``); flag the others.

    A run is a maximal sequence of tokens without Hebrew / Arabic letters that starts and ends with a Latin token. It
    is dropped when it holds at least three Latin tokens or one editorial word / folio label
    (:func:`is_editorial_word`); otherwise it is kept and flagged ``latin_inline``. A token mixing Latin and Hebrew /
    Arabic letters is kept and flagged ``latin_glued``. Tokens inside ``protected`` ranges (mixed bracket spans) are
    never dropped.

    :param line: One line (a line with Hebrew / Arabic letters).
    :param protected: Character ranges not to touch.
    :return: ``(line, dropped runs, flag entries without line index)``.
    """
    tokens = [(m.start(), m.end(), m.group(0)) for m in _TOKEN_RE.finditer(line)]
    kinds = [_token_kind(t) for _, _, t in tokens]
    cut: List[Tuple[int, int]] = []
    removed: List[str] = []
    flags_out: List[Dict[str, Any]] = []
    i = 0
    while i < len(tokens):
        if kinds[i] == "mixed":
            flags_out.append({"flag": "latin_glued", "token": tokens[i][2]})
        if kinds[i] not in ("lat", "num", "punct"):
            i += 1
            continue
        j = i
        while j < len(tokens) and kinds[j] in ("lat", "num", "punct"):
            j += 1
        run = [t for t in range(i, j) if kinds[t] == "lat"]
        if run:
            first, last = run[0], run[-1]
            start, end = tokens[first][0], tokens[last][1]
            text = line[start:end]
            inside = any(a <= start < b or a < end <= b for a, b in protected)
            editorial = len(run) >= 3 or any(is_editorial_word(tokens[t][2]) for t in run)
            if editorial and not inside:
                cut.append((start, end))
                removed.append(text)
            else:
                flags_out.append({"flag": "latin_inline", "token": text})
        i = j
    if not cut:
        return line, removed, flags_out
    out, pos = [], 0
    for start, end in cut:
        out.append(line[pos:start])
        pos = end
    out.append(line[pos:])
    return tidy(" ".join(out)), removed, flags_out


def _amount_follows(text: str, end: int) -> bool:
    """Whether the word after position ``end`` is a currency, weight or count word (:data:`UNIT_SKELETONS`).

    :param text: Line.
    :param end: Position after the digits.
    :return: True when a unit word follows.
    """
    nxt = text[end:].split(None, 1)
    return bool(nxt) and _textnorm_skeleton(unpresent(nxt[0])) in UNIT_SKELETONS


def footnote_anchors(line: str) -> Tuple[str, List[str], List[str]]:
    """Drop footnote anchors: 1-3 ASCII digits glued to the end of a word of at least two letters (``footnote_anchor``).

    Not a fraction (``קכ3/4``), a decimal, digits followed by a letter, or digits followed by a currency / weight word
    (``…5 דינאר``: an amount). Digits after a one-letter word (``ב2``), before a unit word or inside a word (``א5ב``)
    are kept and flagged ``glued_digits``.

    :param line: One line.
    :return: ``(line, dropped anchors, flag codes)``.
    """
    removed: List[str] = []
    codes: List[str] = []

    def _anchor(match: "re.Match[str]") -> str:
        """Replacement of one glued number.

        :param match: :data:`_ANCHOR_RE` match.
        :return: The word with or without its digits.
        """
        word = match.group("word")
        if len(SEMITIC_RE.findall(word)) >= 2 and not _amount_follows(line, match.end()):
            removed.append(match.group(0))
            return word
        codes.append("glued_digits")
        return match.group(0)

    def _bracket_anchor(match: "re.Match[str]") -> str:
        """Drop digits glued to a closing bracket.

        :param match: :data:`_BRACKET_ANCHOR_RE` match.
        :return: Empty string.
        """
        removed.append(match.group(0))
        return ""

    out = _BRACKET_ANCHOR_RE.sub(_bracket_anchor, _ANCHOR_RE.sub(_anchor, line))
    if _GLUED_INSIDE_RE.search(out):
        codes.append("glued_digits")
    return out, removed, codes


def line_number_tokens(lines: Sequence[str]) -> Set[Tuple[int, int, int]]:
    """Bare integers that number lines (sub-rule ``line_number``), as ``(line, start, end)`` character ranges.

    Candidates are tokens that are only an integer (``12``, ``12.``, ``(12)``, ``03``). In reading order, a chain of
    consecutive integers (each one more than the last, at most one line further down) of at least three members numbers
    lines, wherever the integers stand in their lines (line starts, inline markers of a flattened text, the column
    numbers of a merged two-column list). An every-fifth chain (5, 10, 15 ...; each four to six lines further down; at
    line starts; at least three members) does too. Amounts, dates and isolated numbers form no such chain.

    :param lines: Lines of a side.
    :return: The ranges to drop.
    """
    toks: List[Tuple[int, int, int, int]] = []
    for li, line in enumerate(lines):
        for m in _TOKEN_RE.finditer(line):
            num = _NUMBER_TOKEN_RE.fullmatch(m.group(0))
            if num:
                toks.append((li, m.start(), m.end(), int(num.group(1))))
    chosen: Set[Tuple[int, int, int]] = set()
    for step, min_gap, max_gap, starts_only in ((1, 0, 1, False), (5, 4, 6, True)):
        cand = [t for t in toks if not starts_only or not lines[t[0]][:t[1]].strip()]
        if step == 5:
            cand = [t for t in cand if t[3] % 5 == 0]
        best: List[int] = [1] * len(cand)
        prev: List[Optional[int]] = [None] * len(cand)
        for b in range(len(cand)):
            for a in range(b):
                gap = cand[b][0] - cand[a][0]
                if cand[a][3] + step == cand[b][3] and min_gap <= gap <= max_gap and best[a] + 1 > best[b]:
                    best[b], prev[b] = best[a] + 1, a
        used: Set[int] = set()
        for end in sorted(range(len(cand)), key=lambda x: -best[x]):
            if best[end] < 3 or end in used:
                continue
            chain, node = [], end
            while node is not None and node not in used:
                chain.append(node)
                node = prev[node]
            if len(chain) >= 3:
                used.update(chain)
                chosen.update((cand[c][0], cand[c][1], cand[c][2]) for c in chain)
    return chosen


def drop_editorial(lines: Sequence[str]) -> RuleOutput:
    """Drop English / Latin editorial notes, numeric glosses, labels, footnote matter and line numbers.

    Per line, in order: ``footnote_line``, ``apparatus_tail``, ``bracket_note``, ``latin_line``, ``rule_line``,
    ``latin_run``, ``footnote_anchor``; then over the side ``line_number`` (:func:`line_number_tokens`). A line an
    edit leaves empty is dropped. Lines without a change keep their spacing.

    :param lines: Lines of a side.
    :return: The kept lines, one log entry per change, flags (``latin_inline``, ``latin_glued``,
        ``bracket_mixed_latin``, ``glued_digits``, ``hebrew_editorial_candidate``).
    """
    log: List[Dict[str, Any]] = []
    flags: List[Dict[str, Any]] = []
    work: List[Tuple[int, str]] = []
    for i, line in enumerate(lines):
        cur = line
        if _FOOTNOTE_LINE_RE.match(cur):
            log.append(change(RULE_EDITORIAL, "footnote_line", i, cur, None))
            continue
        match = _APPARATUS_RE.search(cur)
        if match:
            new = tidy(cur[:match.start()])
            log.append(change(RULE_EDITORIAL, "apparatus_tail", i, cur, new or None, removed=[cur[match.start():]]))
            if not new:
                continue
            cur = new
        new, removed, protected, codes = bracket_notes(cur)
        flags += [flag(c, i) for c in sorted(set(codes))]
        if removed:
            log.append(change(RULE_EDITORIAL, "bracket_note", i, cur, new or None, removed=removed))
            if not new:
                continue
            cur = new
        if is_latin_line(cur):
            if not LATIN_RE.search(cur) and _FRACTION_RE.search(cur):
                flags.append(flag("numeral_line", i))
            else:
                log.append(change(RULE_EDITORIAL, "latin_line", i, cur, None))
                continue
        if _RULE_LINE_RE.search(cur):
            new = tidy(_RULE_LINE_RE.sub(" ", cur))
            log.append(change(RULE_EDITORIAL, "rule_line", i, cur, new or None, removed=_RULE_LINE_RE.findall(cur)))
            if not new:
                continue
            cur = new
            protected = [(m.start(), m.end()) for pattern in (_PAREN_SPAN_RE, _SQUARE_SPAN_RE)
                         for m in pattern.finditer(cur) if has_semitic(m.group(1)) and LATIN_RE.search(m.group(1))]
        new, removed, run_flags = latin_runs(cur, protected)
        flags += [flag(f["flag"], i, token=f["token"]) for f in run_flags]
        if removed:
            log.append(change(RULE_EDITORIAL, "latin_run", i, cur, new or None, removed=removed))
            if not new:
                continue
            cur = new
        new, removed, codes = footnote_anchors(cur)
        flags += [flag(c, i) for c in sorted(set(codes))]
        if removed:
            log.append(change(RULE_EDITORIAL, "footnote_anchor", i, cur, new, removed=removed))
            cur = new
        if _HEB_EDITORIAL_RE.search(cur):
            flags.append(flag("hebrew_editorial_candidate", i, match=_HEB_EDITORIAL_RE.search(cur).group(0)))
        work.append((i, cur))
    numbers = line_number_tokens([text for _, text in work])
    out_lines: List[str] = []
    kept: List[int] = []
    for w, (i, cur) in enumerate(work):
        spans = sorted((s, e) for li, s, e in numbers if li == w)
        if spans:
            pieces, pos = [], 0
            for s, e in spans:
                pieces.append(cur[pos:s])
                pos = e
            pieces.append(cur[pos:])
            new = tidy(" ".join(pieces))
            log.append(change(RULE_EDITORIAL, "line_number", i, cur, new or None,
                              removed=[cur[s:e] for s, e in spans]))
            if not new:
                continue
            cur = new
        out_lines.append(cur)
        kept.append(i)
    log.sort(key=lambda e: e["line_index"])
    return RuleOutput(out_lines, kept, log, flags)


# ------------------------------------------------------------------------------------------------------ 3 restorations

def mirrored_pairs(line: str, open_ch: str = "[", close_ch: str = "]") -> int:
    """Mirrored bracket pairs inside a line (``x]y[z``: neither bracket at the line's edge) once proper pairs are gone.

    :param line: One line.
    :param open_ch: Opening bracket.
    :param close_ch: Closing bracket.
    :return: Count of mirrored pairs.
    """
    o, c = re.escape(open_ch), re.escape(close_ch)
    proper = re.compile(f"{o}[^{o}{c}\\n]*{c}")
    text, prev = line.replace(GAP, " "), None
    while prev != text:
        prev, text = text, proper.sub(" ", text)
    text = text.strip()
    return sum(1 for m in re.finditer(f"{c}[^{o}{c}\\n]*{o}", text) if m.start() > 0 and m.end() < len(text))


def _blank(inner: str) -> bool:
    """Whether a bracket's content marks a lacuna only: nothing, spaces, dots, ``?``, dashes or gaps.

    :param inner: Text between the brackets (gaps as placeholders).
    :return: True for a lacuna mark.
    """
    return _DOTS_ONLY_RE.fullmatch(inner.replace(_NEW_GAP, "").replace(_OLD_GAP, "")) is not None


def _convert_spans(text: str, pattern: "re.Pattern[str]", removed: List[str], max_len: Optional[int]) -> str:
    """Paired spans holding letters, or only a lacuna mark (:func:`_blank`), become gaps (innermost first, repeated).

    :param text: Line with old gaps protected and double brackets masked.
    :param pattern: :data:`_SQUARE_SPAN_RE` or :data:`_PAREN_SPAN_RE`.
    :param removed: Collects the converted spans.
    :param max_len: Longest content converted (None: any length).
    :return: The line.
    """
    def _span(match: "re.Match[str]") -> str:
        """Replacement of one span.

        :param match: Span match.
        :return: A new gap, or the span unchanged.
        """
        inner = match.group(1)
        if (has_semitic(inner) or _blank(inner)) and (max_len is None or len(inner) <= max_len):
            removed.append(match.group(0))
            return _NEW_GAP
        return match.group(0)

    prev = None
    while prev != text:
        prev, text = text, pattern.sub(_span, text)
    return text


def _dot_parens(text: str, removed: List[str]) -> str:
    """Parentheses around dots only (``(...)``, ``(…)``) become gaps: a lacuna in every convention.

    :param text: Line with old gaps protected.
    :param removed: Collects the converted spans.
    :return: The line.
    """
    def _dots(match: "re.Match[str]") -> str:
        """Replacement of one parenthesised span.

        :param match: Span match.
        :return: A new gap, or the span unchanged.
        """
        if re.fullmatch(r"\s*(?:\.\s*){2,}|\s*…\s*", match.group(1)):
            removed.append(match.group(0))
            return _NEW_GAP
        return match.group(0)

    return _PAREN_SPAN_RE.sub(_dots, text)


def _edges(text: str, open_ch: str, close_ch: str, removed: List[str]) -> str:
    """A closing bracket opening a line, or an opening bracket ending it, becomes a gap (a damaged line edge).

    :param text: Line.
    :param open_ch: Opening bracket.
    :param close_ch: Closing bracket.
    :param removed: Collects the converted brackets.
    :return: The line.
    """
    stripped = text.strip()
    if stripped.startswith(close_ch):
        removed.append(close_ch)
        lead = len(text) - len(text.lstrip())
        text = text[:lead] + _NEW_GAP + text[lead + 1:]
    if text.rstrip().endswith(open_ch):
        removed.append(open_ch)
        tail = len(text.rstrip())
        text = text[:tail - 1] + _NEW_GAP + text[tail:]
    return text


def _weiss_tails(text: str, removed: List[str], max_len: int = WEISS_EDGE_MAX) -> str:
    """Weiss restorations that run to a torn line edge: the last unpaired ``(`` with at most ``max_len`` characters
    after it, and the first unpaired ``)`` with at most ``max_len`` characters before it, become gaps with that text
    (``… אבג (``, ``… אבג (ד``, ``)אבג …``). A longer run, or one crossing a lacuna mark, is left: on the data such
    runs are real text with a stray parenthesis (a title, a formula).

    :param text: Line after the paired spans were converted.
    :param removed: Collects the converted runs.
    :param max_len: Longest run converted.
    :return: The line.
    """
    stack: List[int] = []
    closes: List[int] = []
    for idx, ch in enumerate(text):
        if ch == "(":
            stack.append(idx)
        elif ch == ")":
            if stack:
                stack.pop()
            else:
                closes.append(idx)

    def short(run: str) -> bool:
        """Whether a run qualifies: short, no lacuna mark, no parenthesis.

        :param run: Text between the parenthesis and the line edge.
        :return: True when it is converted.
        """
        return len(run.strip()) <= max_len and not any(ch in run for ch in (_NEW_GAP, _OLD_GAP, "(", ")"))

    if stack and short(text[stack[-1] + 1:]):
        removed.append(text[stack[-1]:])
        text = text[:stack[-1]] + _NEW_GAP
    if closes and short(text[:closes[0]]):
        removed.append(text[:closes[0] + 1])
        text = _NEW_GAP + text[closes[0] + 1:]
    return text


def work_key(row_id: str, source: str) -> str:
    """The edition a literature row comes from (bracket behaviour is a property of a scanned book's OCR).

    :param row_id: Candidate row id (``lit:<collection>:<work>:p<page>:<key>``, ``lit:weiss_1970:<n>``, ``web:...``).
    :param source: Candidate source.
    :return: ``<source>:`` plus the first three ``:`` fields for literature, the first two for other sources.
    """
    parts = row_id.split(":")
    return f"{source}:" + ":".join(parts[:3] if source == "literature" else parts[:2])


def mirrored_works(sides: Iterable[Tuple[str, str, Sequence[str]]], *, min_pairs: int = 5,
                   min_share: float = 0.25) -> Set[str]:
    """Editions whose OCR mirrors square brackets (``אבג]x[ דה``), from all their sides together.

    :param sides: ``(row_id, source, lines)`` of every side.
    :param min_pairs: Fewest mirrored pairs inside lines (:func:`mirrored_pairs`) of a mirrored edition.
    :param min_share: Fewest mirrored pairs as a share of mirrored + proper pairs.
    :return: :func:`work_key` of the mirrored editions of :data:`RESTORATION_SOURCES`.
    """
    mid: Counter = Counter()
    proper: Counter = Counter()
    for row_id, source, lines in sides:
        if source not in RESTORATION_SOURCES:
            continue
        key = work_key(row_id, source)
        for line in lines:
            mid[key] += mirrored_pairs(line)
            proper[key] += len(_SQUARE_SPAN_RE.findall(line.replace(GAP, " ")))
    return {k for k, v in mid.items() if v >= min_pairs and v / (v + proper[k]) >= min_share}


def _residue_flags(text: str, weiss: bool, convert: bool) -> List[str]:
    """Notation left in a line after the restoration rule.

    :param text: Line (gaps as placeholders, double brackets masked).
    :param weiss: The side is Weiss format.
    :param convert: The rule converted this side's spans.
    :return: Flag codes.
    """
    codes: List[str] = []
    if "⟨" in text or "⟩" in text or re.search(r"<[^<>\n]+>", text):
        codes.append("angle_unresolved")
    if "{" in text or "}" in text:
        codes.append("brace_unresolved")
    if any(has_semitic(m.group(1)) for m in _PAREN_SPAN_RE.finditer(text)):
        codes.append("restoration_long_unresolved" if weiss and convert else "paren_unresolved")
    if weiss and any(has_semitic(m.group(1)) for m in _SQUARE_SPAN_RE.finditer(text)):
        codes.append("square_in_weiss_unresolved")
    rest = _SQUARE_SPAN_RE.sub(" ", _PAREN_SPAN_RE.sub(" ", text))
    if any(ch in rest for ch in "[]()"):
        codes.append("bracket_unpaired")
    return codes


def _mask_doubles(text: str) -> Tuple[str, List[str]]:
    """Mask double square brackets ``[[x]]`` (one private-use character each) so the span rules leave them.

    :param text: Line.
    :return: ``(masked line, the double spans in order)``.
    """
    doubles: List[str] = []

    def _mask(match: "re.Match[str]") -> str:
        """Placeholder of one double span.

        :param match: Span match.
        :return: Its placeholder.
        """
        doubles.append(match.group(0))
        return chr(0xF000 + len(doubles) - 1)

    return re.sub(r"\[\[[^\[\]\n]*\]\]", _mask, text), doubles


def render_restorations(lines: Sequence[str], *, source: str, mirrored: bool = False) -> RuleOutput:
    """Render restorations not visible on the page as ``[...]`` where the source's convention is clear.

    Only sources in :data:`RESTORATION_SOURCES` are converted (the others' cleaners already did it, see the module
    docstring); for every source, notation left behind is flagged. A Weiss-format side (an empty ``( )`` anywhere)
    converts its paired parentheses of at most :data:`MAX_RESTORATION_SPAN` characters (``weiss_paren``) and runs from
    an unpaired parenthesis to the torn line edge (``weiss_edge``, :func:`_weiss_tails`); any other side converts paired
    single square spans (``square``), dot-only parentheses (``dots``) and a bracket at the very edge of a line
    (``edge``). Double brackets ``[[x]]`` are masked and flagged. A side of a mirrored edition (``mirrored``,
    :func:`mirrored_works`) or with a mirrored pair inside a line (:func:`mirrored_pairs`) is not converted (flag
    ``brackets_mirrored``). New gaps merge with neighbouring gaps.

    :param lines: Lines of a side (after the editorial rule).
    :param source: Candidate source.
    :param mirrored: The side's edition mirrors brackets.
    :return: The lines (none dropped), one log entry per changed line with the converted spans.
    """
    weiss = any(_EMPTY_PAREN_RE.search(line) for line in lines)
    restorable = source in RESTORATION_SOURCES
    pair = ("(", ")") if weiss else ("[", "]")
    mirrored = restorable and (mirrored or any(mirrored_pairs(line, *pair) for line in lines))
    convert = restorable and not mirrored
    out_lines: List[str] = []
    log: List[Dict[str, Any]] = []
    flags: List[Dict[str, Any]] = [flag("brackets_mirrored", None)] if mirrored else []
    for i, line in enumerate(lines):
        text, doubles = _mask_doubles(line.replace(GAP, _OLD_GAP))
        removed: List[str] = []
        subs: List[str] = []
        steps: List[Tuple[str, Any]] = []
        if convert and weiss:
            steps = [("weiss_paren", lambda t: _convert_spans(t, _PAREN_SPAN_RE, removed, MAX_RESTORATION_SPAN)),
                     ("weiss_edge", lambda t: _weiss_tails(t, removed))]
        elif convert:
            steps = [("square", lambda t: _convert_spans(t, _SQUARE_SPAN_RE, removed, None)),
                     ("dots", lambda t: _dot_parens(t, removed)),
                     ("edge", lambda t: _edges(t, "[", "]", removed))]
        for name, step in steps:
            before = len(removed)
            text = step(text)
            if len(removed) > before:
                subs.append(name)
        codes = _residue_flags(text, weiss, convert)
        if doubles:
            codes.append("double_bracket_unresolved")
        flags += [flag(c, i) for c in codes]
        for k, d in enumerate(doubles):
            text = text.replace(chr(0xF000 + k), d)
        if removed:
            text = _GAP_RUN_RE.sub(GAP, text)
            new = tidy(text.replace(_OLD_GAP, GAP).replace(_NEW_GAP, GAP))
            log.append(change(RULE_RESTORATION, "+".join(subs), i, line, new, removed=removed))
            out_lines.append(new)
        else:
            out_lines.append(line)
    return RuleOutput(out_lines, list(range(len(lines))), log, flags)


# --------------------------------------------------------------------------------------------------------- 4 flattened

def flag_flattened(lines: Sequence[str], *, max_letters: int = FLAT_LINE_LETTERS) -> RuleOutput:
    """Flag lines longer than a manuscript line (``flattened``), for re-lining from the aligned read lines.

    :param lines: Lines of a side.
    :param max_letters: Longest line not flagged.
    :return: The lines unchanged; one flag per long line with its ``letters`` and ``side_flat`` (the side is a single
        long line, or its median line is long).
    """
    counts = [letters(line) for line in lines]
    side_flat = bool(counts) and (statistics.median(counts) > max_letters)
    flags = [flag("flattened", i, letters=n, side_flat=side_flat or len(counts) == 1)
             for i, n in enumerate(counts) if n > max_letters]
    return RuleOutput(list(lines), list(range(len(lines))), [], flags)


# --------------------------------------------------------------------------------------------------------- 5 scrambled

def line_script(line: str, default: str = "hebrew") -> str:
    """Script of a line by its letters.

    :param line: One line.
    :param default: Script on a tie (``mixed`` is read as ``hebrew``).
    :return: ``hebrew`` or ``arabic``.
    """
    sk = skel(line)
    heb = len(_HEB_LETTER_RE.findall(sk))
    ara = len(sk) - heb
    if heb == ara:
        return "arabic" if default == "arabic" else "hebrew"
    return "hebrew" if heb > ara else "arabic"


def lm_segments(line: str, script: str) -> List[str]:
    """A line as 3-gram model input: letter segments of one script, words separated by single spaces.

    Hebrew keeps its final forms and writes every geresh / gershayim / apostrophe after a letter as ``'``; Arabic is
    folded as the skeleton folds it. A gap token, a square bracket or a letter of the other script breaks the line, and
    the partial word on each side of a break is left out (fragments at damage are not scrambling). A maqaf, dash, slash
    or pipe separates words; digits, Latin letters, marks and other punctuation are dropped without a break. Numerals
    and abbreviations (:func:`skip_token`) split a segment.

    :param line: One line.
    :param script: ``hebrew`` or ``arabic``.
    :return: Segments (possibly none).
    """
    text = unpresent(unicodedata.normalize("NFC", line)).replace(GAP, "[")
    segments: List[Tuple[str, bool, bool]] = []
    cur: List[str] = []
    left_break = False

    def close(right_break: bool) -> None:
        """End the current segment.

        :param right_break: The segment ends at a break.
        """
        nonlocal cur, left_break
        seg = _WS_RE.sub(" ", "".join(cur)).strip()
        if seg:
            segments.append((seg, left_break, right_break))
        cur = []

    for ch in text:
        code = ord(ch)
        hebrew = 0x05D0 <= code <= 0x05EA
        arabic = (0x0621 <= code <= 0x064A and code != 0x0640) or 0x0671 <= code <= 0x06D3
        if (hebrew and script == "hebrew") or (arabic and script == "arabic"):
            cur.append(ch.translate(_ARABIC_FOLD) if arabic else ch)
        elif hebrew or arabic or ch in "[]":
            close(True)
            left_break = True
        elif ch.isspace() or ch in _WORD_SEPARATORS:
            cur.append(" ")
        elif script == "hebrew" and ch in _GERESH_CHARS and cur and cur[-1] not in (" ", LM_GERESH):
            cur.append(LM_GERESH)
    close(False)
    out: List[str] = []
    for seg, left, right in segments:
        words = seg.split(" ")
        if left:
            words = words[1:]
        if right:
            words = words[:-1]
        run: List[str] = []
        for word in words + [""]:
            if word and not skip_token(word):
                run.append(word)
                continue
            if run:
                out.append(" ".join(run))
            run = []
    return out


def skip_token(word: str) -> bool:
    """Whether a word is left out of the 3-gram model: a numeral or an abbreviation, not a word whose letters can be
    in a wrong order (one letter; up to three letters closed by a geresh, ``ק'``, ``דינ'``; up to four letters around a
    geresh or gershayim, ``ש'מ``, ``ק'ג'``).

    :param word: One word of :func:`lm_segments` (geresh as ``'``).
    :return: True when the word only splits the segment.
    """
    bare = word.replace(LM_GERESH, "")
    if len(bare) <= 1:
        return True
    return LM_GERESH in word and (len(bare) <= 3 if word.endswith(LM_GERESH) else len(bare) <= 4)


@dataclass
class Char3Model:
    """Character 3-gram model of one script, Witten-Bell interpolated down to an add-one unigram.

    :param script: ``hebrew`` or ``arabic``.
    :param c1: Unigram counts of predicted symbols.
    :param c2: Bigram counts ``{history char: {next: count}}``.
    :param c3: Trigram counts ``{history 2 chars: {next: count}}``.
    :param threshold: Score under which a line is flagged (the training lines' :data:`SCRAMBLED_PERCENTILE`), or None.
    :param meta: Training statistics (lines, quantiles, sources).
    """

    script: str
    c1: Dict[str, int] = field(default_factory=dict)
    c2: Dict[str, Dict[str, int]] = field(default_factory=dict)
    c3: Dict[str, Dict[str, int]] = field(default_factory=dict)
    threshold: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Derive totals and type counts from the counts."""
        self.refresh()

    def refresh(self) -> None:
        """Recompute the totals the probabilities use (after counts change)."""
        self._n1 = sum(self.c1.values())
        self._v = len(self.c1) + 1
        self._t2 = {h: (sum(row.values()), len(row)) for h, row in self.c2.items()}
        self._t3 = {h: (sum(row.values()), len(row)) for h, row in self.c3.items()}

    def add(self, segment: str) -> None:
        """Count one segment (padded ``##`` before, ``#`` after).

        :param segment: Output of :func:`lm_segments`.
        """
        seq = LM_PAD * 2 + segment + LM_PAD
        for k in range(2, len(seq)):
            nxt = seq[k]
            self.c1[nxt] = self.c1.get(nxt, 0) + 1
            row2 = self.c2.setdefault(seq[k - 1], {})
            row2[nxt] = row2.get(nxt, 0) + 1
            row3 = self.c3.setdefault(seq[k - 2:k], {})
            row3[nxt] = row3.get(nxt, 0) + 1

    def prob(self, nxt: str, history: str) -> float:
        """P(next symbol | two symbols of history), Witten-Bell interpolated.

        :param nxt: Next symbol.
        :param history: The two symbols before it.
        :return: Probability.
        """
        p = (self.c1.get(nxt, 0) + 1) / (self._n1 + self._v)
        for row, totals, key in ((self.c2, self._t2, history[-1:]), (self.c3, self._t3, history[-2:])):
            counts = row.get(key)
            if counts:
                total, types = totals[key]
                p = (counts.get(nxt, 0) + types * p) / (total + types)
        return p

    def segment_logprob(self, segment: str) -> Tuple[float, int]:
        """Log2-probability of a segment and its letter count.

        :param segment: Output of :func:`lm_segments`.
        :return: ``(log2 probability of every predicted symbol, letters)``.
        """
        seq = LM_PAD * 2 + segment + LM_PAD
        total = sum(math.log2(self.prob(seq[k], seq[k - 2:k])) for k in range(2, len(seq)))
        return total, sum(1 for ch in segment if ch not in (" ", LM_GERESH))

    def score(self, line: str) -> Optional[Tuple[float, int]]:
        """Per-letter log2-likelihood of a line's segments of this model's script.

        :param line: One line.
        :return: ``(score, letters scored)``, or None with fewer than :data:`MIN_SCORE_LETTERS` letters.
        """
        total, n = 0.0, 0
        for seg in lm_segments(line, self.script):
            lp, k = self.segment_logprob(seg)
            total, n = total + lp, n + k
        return (total / n, n) if n >= MIN_SCORE_LETTERS else None

    def to_json(self) -> Dict[str, Any]:
        """The model as JSON-ready data.

        :return: Dict with format, script, threshold, meta and counts.
        """
        return {"format": MODEL_FORMAT, "script": self.script, "threshold": self.threshold,
                "threshold_percentile": SCRAMBLED_PERCENTILE, "min_score_letters": MIN_SCORE_LETTERS,
                "meta": self.meta, "c1": self.c1, "c2": self.c2, "c3": self.c3}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Char3Model":
        """A model from :meth:`to_json` data.

        :param data: Loaded JSON.
        :return: The model.
        """
        if data.get("format") != MODEL_FORMAT:
            raise ValueError(f"not a {MODEL_FORMAT} model: {data.get('format')}")
        return cls(script=data["script"], c1=dict(data["c1"]), c2={k: dict(v) for k, v in data["c2"].items()},
                   c3={k: dict(v) for k, v in data["c3"].items()}, threshold=data.get("threshold"),
                   meta=dict(data.get("meta") or {}))


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile (numpy's default method).

    :param values: Values (non-empty).
    :param pct: Percentile, 0-100.
    :return: The percentile.
    """
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def train_models(lines: Iterable[Tuple[str, str]], *, pct: float = SCRAMBLED_PERCENTILE) -> Dict[str, Char3Model]:
    """Train one model per script and set each threshold at the ``pct`` percentile of its training lines' scores.

    :param lines: ``(line, script)`` pairs (script ``hebrew`` / ``arabic``: the line's own, :func:`line_script`).
    :param pct: Threshold percentile.
    :return: ``{script: model}`` for the scripts with scored lines.
    """
    by_script: Dict[str, List[str]] = defaultdict(list)
    for line, script in lines:
        by_script[script].append(line)
    models: Dict[str, Char3Model] = {}
    for script, items in sorted(by_script.items()):
        model = Char3Model(script)
        for line in items:
            for seg in lm_segments(line, script):
                model.add(seg)
        model.refresh()
        scores = [s for s in (model.score(line) for line in items) if s is not None]
        if not scores:
            continue
        values = [v for v, _ in scores]
        model.threshold = percentile(values, pct)
        model.meta = {"lines": len(items), "lines_scored": len(values), "letters_scored": sum(n for _, n in scores),
                      "quantiles": {str(q): round(percentile(values, q), 4) for q in (0.5, 1, 5, 25, 50, 75, 95)},
                      "trained": date.today().isoformat()}
        models[script] = model
    return models


def save_models(models: Mapping[str, Char3Model], models_dir: Path) -> Dict[str, Path]:
    """Write ``char3_<script>.json`` per model.

    :param models: Models by script.
    :param models_dir: Output directory (created).
    :return: Written paths by script.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    for script, model in models.items():
        path = models_dir / f"char3_{script}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(model.to_json(), fh, ensure_ascii=False)
        out[script] = path
    return out


def load_models(models_dir: Path, scripts: Sequence[str] = ("hebrew", "arabic")) -> Dict[str, Char3Model]:
    """Read the models written by :func:`save_models`.

    :param models_dir: Directory of ``char3_<script>.json``.
    :param scripts: Scripts to load (a missing file is skipped).
    :return: Models by script.
    """
    out: Dict[str, Char3Model] = {}
    for script in scripts:
        path = models_dir / f"char3_{script}.json"
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                out[script] = Char3Model.from_json(json.load(fh))
    return out


def window_scores(line: str, model: Char3Model, words: int = WINDOW_WORDS,
                  step: int = WINDOW_STEP) -> List[Tuple[int, int, float]]:
    """Scores of word windows of a line (a flattened text holds many manuscript lines; scrambling is local).

    :param line: One line.
    :param model: Model of the line's script.
    :param words: Window length in whitespace tokens.
    :param step: Window step.
    :return: ``(first token, end token, score)`` of every window with a score.
    """
    tokens = line.split()
    out: List[Tuple[int, int, float]] = []
    for start in range(0, max(1, len(tokens) - words + step), step):
        end = min(len(tokens), start + words)
        scored = model.score(" ".join(tokens[start:end]))
        if scored is not None:
            out.append((start, end, scored[0]))
        if end == len(tokens):
            break
    return out


def flag_scrambled(lines: Sequence[str], models: Mapping[str, Char3Model], *, script: str = "hebrew") -> RuleOutput:
    """Flag lines whose letters are unlikely under the 3-gram model of their script (``scrambled_candidate``).

    A line is flagged when its score is below its model's threshold (``basis: line``). A line longer than
    :data:`FLAT_LINE_LETTERS` letters (a flattened text) is also scored in windows of :data:`WINDOW_WORDS` tokens
    (:func:`window_scores`) and flagged when its worst window is below the threshold and at least
    :data:`WINDOW_DROP` under the line's median window (``basis: window``; a local anomaly, not a text that is
    unusual throughout); ``window`` gives the token range. After the gate re-lines a flattened text, this function
    can be run again on the new lines.

    :param lines: Lines of a side.
    :param models: Models by script, each with a threshold.
    :param script: The side's script (tie-break of :func:`line_script`).
    :return: The lines unchanged; one flag per low line with ``score``, ``threshold``, ``script``, ``basis`` and
        ``letters`` and ``reversed`` (the line read backwards scores :data:`REVERSED_GAIN` better), or ``window``.
    """
    flags: List[Dict[str, Any]] = []
    for i, line in enumerate(lines):
        ls = line_script(line, script)
        model = models.get(ls)
        if model is None or model.threshold is None:
            continue
        threshold = round(model.threshold, 4)
        scored = model.score(line)
        if scored is not None and scored[0] < model.threshold:
            backwards = model.score(line[::-1])
            flags.append(flag("scrambled_candidate", i, score=round(scored[0], 4), threshold=threshold, script=ls,
                              basis="line", letters=scored[1],
                              reversed=backwards is not None and backwards[0] >= scored[0] + REVERSED_GAIN))
            continue
        if letters(line) <= FLAT_LINE_LETTERS:
            continue
        windows = window_scores(line, model)
        if len(windows) < 3:
            continue
        worst = min(windows, key=lambda w: w[2])
        median = statistics.median(w[2] for w in windows)
        if worst[2] < model.threshold and worst[2] <= median - WINDOW_DROP:
            flags.append(flag("scrambled_candidate", i, score=round(worst[2], 4), threshold=threshold, script=ls,
                              basis="window", window=[worst[0], worst[1]], median_window=round(median, 4),
                              windows_below=sum(1 for w in windows if w[2] < model.threshold)))
    return RuleOutput(list(lines), list(range(len(lines))), [], flags)


# ---------------------------------------------------------------------------------------------------------------- side

def remap(entry: Mapping[str, Any], origin: Sequence[int]) -> Dict[str, Any]:
    """A log or flag entry with its line indices (``line_index``, ``copy_of``, ``block``) mapped through ``origin``.

    :param entry: Entry whose indices refer to a rule's input.
    :param origin: Input-side index of each line of that rule's input.
    :return: The mapped copy.
    """
    out = dict(entry)
    for key in ("line_index", "copy_of"):
        if out.get(key) is not None:
            out[key] = origin[out[key]]
    if out.get("block") is not None:
        out["block"] = [origin[k] for k in out["block"]]
    return out


def text_rules(lines: Sequence[str], *, source: str, brackets_mirrored: bool = False
               ) -> Tuple[List[str], List[int], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Rules 1-3 (duplicates, editorial, restorations) with indices mapped to the input side.

    :param lines: Lines of a side.
    :param source: Candidate source.
    :param brackets_mirrored: The side's edition mirrors brackets (:func:`mirrored_works`).
    :return: ``(lines, origin, log, flags)``; log and flag indices are input-side indices.
    """
    cur = list(lines)
    origin = list(range(len(lines)))
    log: List[Dict[str, Any]] = []
    flags: List[Dict[str, Any]] = []
    for rule in (drop_duplicates, drop_editorial,
                 lambda ls: render_restorations(ls, source=source, mirrored=brackets_mirrored)):
        res = rule(cur)
        log += [remap(e, origin) for e in res.log]
        flags += [remap(f, origin) for f in res.flags]
        origin = [origin[k] for k in res.kept]
        cur = res.lines
    return cur, origin, log, flags


def clean_side(lines: Sequence[str], *, source: str, script: str,
               model: Optional[Mapping[str, Char3Model]] = None, brackets_mirrored: bool = False) -> CleanResult:
    """Clean one target side: duplicates, editorial matter, restorations; flag flattened and scrambled lines.

    :param lines: The side's lines (novel_build ``TargetSide.lines``).
    :param source: Candidate source (``pgp``, ``fjp``, ``literature`` ...).
    :param script: The side's script (``hebrew`` / ``arabic`` / ``mixed``), the tie-break of a line's script.
    :param model: 3-gram models by script (:func:`load_models`); None skips the scrambled rule.
    :param brackets_mirrored: The side's edition mirrors brackets (``work_key(row_id, source) in
        mirrored_works(...)``): its restorations are left and flagged.
    :return: Cleaned lines, change log, flags (``out_index`` = index in the cleaned lines) and line origins.
    """
    cur, origin, log, flags = text_rules(lines, source=source, brackets_mirrored=brackets_mirrored)
    late = flag_flattened(cur).flags
    if model:
        late += flag_scrambled(cur, model, script="arabic" if script == "arabic" else "hebrew").flags
    flags += [remap(f, origin) for f in late]
    position = {o: k for k, o in enumerate(origin)}
    for f in flags:
        f["out_index"] = None if f["line_index"] is None else position.get(f["line_index"])
    return CleanResult(cur, log, flags, origin)


# ------------------------------------------------------------------------------------------------------------- dry run

@dataclass
class SideRecord:
    """One target side of the stage-1 file.

    :param row_id: Candidate row id.
    :param side_key: Side key.
    :param source: Candidate source.
    :param script: Candidate script.
    :param lines: Target lines.
    """

    row_id: str
    side_key: str
    source: str
    script: str
    lines: List[str]


def read_sides(path: Path) -> List[SideRecord]:
    """Every target side of a stage-1 candidates file.

    :param path: ``stage1_images.jsonl`` (``schema.Candidate`` records).
    :return: Sides in file order.
    """
    out: List[SideRecord] = []
    try:
        with open(path, encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc}")
    for rec in records:
        for target in rec.get("targets") or []:
            out.append(SideRecord(str(rec["row_id"]), str(target.get("side_key") or ""), str(rec.get("source")),
                                  str(rec.get("script") or ""), list(target.get("lines") or [])))
    return out


def summarize(sides: Sequence[SideRecord], results: Sequence[CleanResult]) -> Dict[str, Any]:
    """Counts per rule x source x script, per sub-rule x source and per flag x source.

    :param sides: Sides.
    :param results: Their results, same order.
    :return: Nested counters (JSON-ready).
    """
    rule_tab: Dict[Tuple[str, str, str], Counter] = defaultdict(Counter)
    sub_tab: Dict[Tuple[str, str, str], Counter] = defaultdict(Counter)
    flag_tab: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    totals: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for side, res in zip(sides, results):
        key = (side.source, side.script)
        totals[key]["sides"] += 1
        totals[key]["lines"] += len(side.lines)
        totals[key]["letters"] += sum(letters(x) for x in side.lines)
        touched: Set[str] = set()
        for e in res.clean_log:
            for tab, k in ((rule_tab, (e["rule"], side.source, side.script)),
                           (sub_tab, (e["rule"], e["sub"], side.source))):
                tab[k]["changes"] += 1
                tab[k]["lines_dropped"] += e["after"] is None
                tab[k]["letters_removed"] += e["letters_removed"]
            touched.add(e["rule"])
        for rule in touched:
            rule_tab[(rule, side.source, side.script)]["sides"] += 1
        flagged: Dict[str, Set[Optional[int]]] = defaultdict(set)
        for f in res.flags:
            flagged[f["flag"]].add(f["line_index"])
        for code, idx in flagged.items():
            flag_tab[(code, side.source)]["sides"] += 1
            flag_tab[(code, side.source)]["lines"] += len([i for i in idx if i is not None])
            if code in (RULE_FLATTENED, "scrambled_candidate"):
                rule = RULE_FLATTENED if code == RULE_FLATTENED else RULE_SCRAMBLED
                rule_tab[(rule, side.source, side.script)]["sides"] += 1
                rule_tab[(rule, side.source, side.script)]["lines_flagged"] += len(idx)
    return {"rules": rule_tab, "subs": sub_tab, "flags": flag_tab, "totals": totals}


def _md_table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> List[str]:
    """A Markdown table.

    :param header: Column names.
    :param rows: Rows.
    :return: Table lines.
    """
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" if k == 0 else "---:" for k in range(len(header)))
           + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


def convention_survey(sides: Sequence[SideRecord]) -> Dict[str, Counter]:
    """Bracket conventions in the input sides per source (what survives novel_build's cleaners).

    :param sides: Sides.
    :return: ``{source: Counter}`` of spans and marks.
    """
    out: Dict[str, Counter] = defaultdict(Counter)
    for side in sides:
        c = out[side.source]
        c["sides"] += 1
        c["sides_weiss"] += any(_EMPTY_PAREN_RE.search(x) for x in side.lines)
        for line in side.lines:
            text = line.replace(GAP, _OLD_GAP)
            c["gap_tokens"] += line.count(GAP)
            c["square_with_letters"] += sum(1 for m in _SQUARE_SPAN_RE.finditer(text) if has_semitic(m.group(1)))
            c["square_dots_or_empty"] += sum(1 for m in _SQUARE_SPAN_RE.finditer(text)
                                             if _DOTS_ONLY_RE.fullmatch(m.group(1)))
            c["double_square"] += len(re.findall(r"\[\[[^\[\]\n]*\]\]", text))
            c["paren_empty"] += len(_EMPTY_PAREN_RE.findall(text))
            c["paren_with_letters"] += sum(1 for m in _PAREN_SPAN_RE.finditer(text) if has_semitic(m.group(1)))
            c["paren_latin_or_digits_only"] += sum(
                1 for m in _PAREN_SPAN_RE.finditer(text) if not has_semitic(m.group(1))
                and (LATIN_RE.search(m.group(1)) or ASCII_DIGIT_RE.search(m.group(1))))
            c["angle_math"] += text.count("⟨")
            c["braces"] += text.count("{")
            rest = _SQUARE_SPAN_RE.sub(" ", _PAREN_SPAN_RE.sub(" ", text))
            c["unpaired_brackets"] += sum(rest.count(ch) for ch in "[]()")
            c["lines_mid_mirrored"] += mirrored_pairs(line) > 0
    return out


def render_dryrun(sides: Sequence[SideRecord], results: Sequence[CleanResult], models: Mapping[str, Char3Model],
                  model_paths: Mapping[str, Path], stage1: Path, audited: Sequence[Dict[str, Any]],
                  mirrored: Sequence[str] = ()) -> str:
    """The dry-run report (counts per rule x source x script, sub-rules, flags, conventions, models, audit items).

    :param sides: Sides.
    :param results: Their results.
    :param models: Trained models.
    :param model_paths: Where they were written.
    :param stage1: Input file.
    :param audited: Rows for the audited-defect table (from :func:`audited_rows`).
    :param mirrored: Editions treated as bracket-mirrored (:func:`mirrored_works`).
    :return: Markdown.
    """
    tabs = summarize(sides, results)
    out = ["# clean_v2 dry run (generated)", "",
           f"Input: `{stage1}` ({len(sides):,} target sides). Generated {date.today().isoformat()} by "
           "`python -m src.finetuning.qwen_hebrew.ja_ar_build.clean_v2`. INTERNAL: holds no edition text; examples are "
           "in `clean_v2_examples.md`.", ""]
    out += ["## Input per source x script", ""]
    out += _md_table(["source", "script", "sides", "lines", "letters"],
                     [(s, sc, c["sides"], c["lines"], c["letters"]) for (s, sc), c in sorted(tabs["totals"].items())])
    out += ["", "## Rules x source x script", "",
            "`sides` = sides the rule changed (rules 1-3) or flagged (4-5); `changes` = log entries; letters removed "
            "= Hebrew + Arabic letters (skeleton) taken out of the target.", ""]
    rows = []
    for (rule, s, sc), c in sorted(tabs["rules"].items(), key=lambda kv: (RULES.index(kv[0][0]), kv[0][1], kv[0][2])):
        rows.append((rule, s, sc, c["sides"], c["changes"], c["lines_dropped"], c["letters_removed"],
                     c["lines_flagged"]))
    out += _md_table(["rule", "source", "script", "sides", "changes", "lines dropped", "letters removed",
                      "lines flagged"], rows)
    out += ["", "## Sub-rules x source", ""]
    out += _md_table(["rule", "sub-rule", "source", "changes", "lines dropped", "letters removed"],
                     [(r, sub, s, c["changes"], c["lines_dropped"], c["letters_removed"])
                      for (r, sub, s), c in sorted(tabs["subs"].items())])
    out += ["", "## Flags x source (no text changed)", ""]
    out += _md_table(["flag", "source", "sides", "lines"],
                     [(code, s, c["sides"], c["lines"]) for (code, s), c in sorted(tabs["flags"].items())])
    survey = convention_survey(sides)
    keys = ["sides", "sides_weiss", "gap_tokens", "square_with_letters", "square_dots_or_empty", "double_square",
            "paren_empty", "paren_with_letters", "paren_latin_or_digits_only", "angle_math", "braces",
            "unpaired_brackets", "lines_mid_mirrored"]
    out += ["", "## Bracket conventions in the input (what survives novel_build's cleaners)", ""]
    out += _md_table(["source"] + keys, [[s] + [survey[s][k] for k in keys] for s in sorted(survey)])
    out += ["", f"Editions treated as bracket-mirrored (no restoration converted in any of their sides): "
            f"{', '.join(f'`{w}`' for w in mirrored) or 'none'}."]
    out += ["", "## 3-gram models", ""]
    out += _md_table(["script", "file", "lines", "lines scored", "threshold (p1)", "p0.5", "p5", "p50"],
                     [(sc, f"`{model_paths.get(sc, '')}`", m.meta.get("lines"), m.meta.get("lines_scored"),
                       round(m.threshold or 0.0, 4), m.meta["quantiles"].get("0.5"), m.meta["quantiles"].get("5"),
                       m.meta["quantiles"].get("50")) for sc, m in sorted(models.items())])
    basis: Counter = Counter()
    for side, res in zip(sides, results):
        for f in res.flags:
            if f["flag"] == "scrambled_candidate":
                basis[(f.get("basis", "") + (" (reversed)" if f.get("reversed") else ""), side.source)] += 1
    out += ["", "Scrambled-candidate flags by basis (`line`: the line's score; `window`: worst word window of a "
            "flattened line, a local drop):", ""]
    out += _md_table(["basis", "source", "lines"], [(b, s, n) for (b, s), n in sorted(basis.items())])
    if audited:
        out += ["", "## The audited pairs (blind visual audit) and what clean_v2 does to their target sides", ""]
        out += _md_table(["item", "stratum", "row_id", "side", "clean_v2"],
                         [(a["item"], a["stratum"], f"`{a['row_id']}`", a["side_key"] or "''", a["outcome"])
                          for a in audited])
    return "\n".join(out) + "\n"


def _short(text: Optional[str], limit: int = 300) -> str:
    """A line for a Markdown table cell.

    :param text: Line or None.
    :param limit: Characters kept.
    :return: Escaped, shortened text (``(dropped)`` for None).
    """
    if text is None:
        return "*(dropped)*"
    text = text.replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + " …"


def render_examples(sides: Sequence[SideRecord], results: Sequence[CleanResult], *, per_rule: int = 15,
                    per_sub: int = 8, seed: int = 20261112) -> str:
    """Random examples per rule (and per sub-rule / flag), before and after.

    :param sides: Sides.
    :param results: Their results.
    :param per_rule: Examples per rule.
    :param per_sub: Examples per sub-rule and per flag code.
    :param seed: Random seed.
    :return: Markdown.
    """
    rng = random.Random(seed)
    by_rule: Dict[str, List[Tuple[SideRecord, Dict[str, Any]]]] = defaultdict(list)
    by_sub: Dict[str, List[Tuple[SideRecord, Dict[str, Any]]]] = defaultdict(list)
    by_flag: Dict[str, List[Tuple[SideRecord, Dict[str, Any], str]]] = defaultdict(list)
    for side, res in zip(sides, results):
        for e in res.clean_log:
            by_rule[e["rule"]].append((side, e))
            by_sub[f"{e['rule']}:{e['sub']}"].append((side, e))
        for f in res.flags:
            line = side.lines[f["line_index"]] if f["line_index"] is not None else ""
            if f.get("window"):
                line = "… " + " ".join(line.split()[f["window"][0]:f["window"][1]]) + " …"
            by_flag[f["flag"]].append((side, f, line))
            if f["flag"] in (RULE_FLATTENED, "scrambled_candidate"):
                rule = RULE_FLATTENED if f["flag"] == RULE_FLATTENED else RULE_SCRAMBLED
                by_rule[rule].append((side, dict(f, line=line)))
    out = ["# clean_v2 examples (generated)", "",
           "INTERNAL / PRIVATE: holds edition text (git-ignored folder). Random draws, seed "
           f"{seed}. `before` / `after` are whole lines; `line` = index in the side as given.", ""]
    for rule in RULES:
        items = by_rule.get(rule, [])
        out += [f"## {rule} ({len(items):,} entries; {min(per_rule, len(items))} drawn)", ""]
        draw = rng.sample(items, min(per_rule, len(items)))
        if rule in (RULE_FLATTENED, RULE_SCRAMBLED):
            out += _md_table(["row_id", "side", "source", "line", "detail", "line text"],
                             [(f"`{s.row_id}`", s.side_key or "''", s.source, e["line_index"],
                               _detail(e), _short(e.get("line"))) for s, e in draw])
        else:
            out += _md_table(["row_id", "side", "source", "sub", "line", "before", "after"],
                             [(f"`{s.row_id}`", s.side_key or "''", s.source, e["sub"], e["line_index"],
                               _short(e["before"]), _short(e["after"])) for s, e in draw])
        out.append("")
    out += ["## Appendix A: per sub-rule", ""]
    for sub in sorted(by_sub):
        items = by_sub[sub]
        out += [f"### {sub} ({len(items):,})", ""]
        draw = rng.sample(items, min(per_sub, len(items)))
        out += _md_table(["row_id", "side", "source", "line", "removed", "before", "after"],
                         [(f"`{s.row_id}`", s.side_key or "''", s.source, e["line_index"],
                           _short("; ".join(e.get("removed") or []), 120), _short(e["before"]), _short(e["after"]))
                          for s, e in draw])
        out.append("")
    out += ["## Appendix B: per flag", ""]
    for code in sorted(by_flag):
        items = by_flag[code]
        out += [f"### {code} ({len(items):,})", ""]
        draw = rng.sample(items, min(per_sub, len(items)))
        out += _md_table(["row_id", "side", "source", "line", "detail", "line text"],
                         [(f"`{s.row_id}`", s.side_key or "''", s.source, f["line_index"], _detail(f), _short(line))
                          for s, f, line in draw])
        out.append("")
    return "\n".join(out) + "\n"


def _detail(entry: Mapping[str, Any]) -> str:
    """Flag details for a table cell.

    :param entry: Flag entry.
    :return: ``key=value`` pairs without the standard keys.
    """
    skip = {"flag", "line_index", "out_index", "line"}
    return _short(", ".join(f"{k}={v}" for k, v in entry.items() if k not in skip), 160)


def load_audit_key(path: Optional[Path]) -> List[Dict[str, str]]:
    """Audited pairs of a blind visual audit's judge key (JSON lines with ``item_id``, ``stratum``, ``row_id``,
    ``side_key``), to report what clean_v2 does to each audited target side.

    :param path: ``judge_key_REVEALED.jsonl``, or None.
    :return: ``{item, stratum, row_id, side_key}`` per audited pair, in item order (empty for None).
    """
    if path is None:
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc}")
    items = [{"item": str(r.get("item_id")), "stratum": str(r.get("stratum") or ""), "row_id": str(r.get("row_id")),
              "side_key": str(r.get("side_key") or "")} for r in records]
    return sorted(items, key=lambda x: x["item"])


def audited_rows(sides: Sequence[SideRecord], results: Sequence[CleanResult],
                 items: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    """What clean_v2 did on each audited target side.

    :param sides: Sides.
    :param results: Their results.
    :param items: :func:`load_audit_key` result.
    :return: The items with ``outcome`` (sub-rules and flags with counts, letters removed).
    """
    index = {(s.row_id, s.side_key): r for s, r in zip(sides, results)}
    rows = []
    for it in items:
        res = index.get((it["row_id"], it["side_key"]))
        if res is None:
            outcome = "side not in input"
        else:
            subs = Counter(f"{e['rule']}:{e['sub']}" for e in res.clean_log)
            codes = Counter(f["flag"] for f in res.flags)
            parts = ([f"{k} x{v}" for k, v in sorted(subs.items())]
                     + [f"flag {k} x{v}" for k, v in sorted(codes.items())])
            outcome = (", ".join(parts) or "no change") + f"; letters removed {res.letters_removed}"
        rows.append(dict(it, outcome=outcome))
    return rows


def build_parser() -> argparse.ArgumentParser:
    """The command line of the dry run.

    :return: Parser.
    """
    ap = argparse.ArgumentParser(description="clean_v2 dry run: train the 3-gram models, clean every side, report.")
    ap.add_argument("--stage1", type=Path, required=True, help="novel_build stage-1 candidates file")
    ap.add_argument("--models-dir", type=Path, required=True, help="where char3_<script>.json are written")
    ap.add_argument("--report-dir", type=Path, required=True, help="where the two Markdown reports are written")
    ap.add_argument("--audit-key", type=Path, default=None,
                    help="judge key of a blind visual audit (JSON lines with item_id, stratum, row_id, side_key): "
                         "report what clean_v2 does to each audited side")
    ap.add_argument("--seed", type=int, default=20261112, help="example draw seed")
    ap.add_argument("--examples", type=int, default=15, help="examples per rule")
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    """Dry run: rules 1-3 on every side, train and save the models on the result, clean every side, write reports.

    :param argv: Arguments (default ``sys.argv``).
    """
    args = build_parser().parse_args(argv)
    sides = read_sides(args.stage1)
    mirrored = mirrored_works((s.row_id, s.source, s.lines) for s in sides)
    flips = [work_key(s.row_id, s.source) in mirrored for s in sides]
    training: List[Tuple[str, str]] = []
    for side, flip in zip(sides, flips):
        cleaned, _, _, _ = text_rules(side.lines, source=side.source, brackets_mirrored=flip)
        training += [(line, line_script(line, side.script)) for line in cleaned]
    models = train_models(training)
    paths = save_models(models, args.models_dir)
    results = [clean_side(s.lines, source=s.source, script=s.script, model=models, brackets_mirrored=flip)
               for s, flip in zip(sides, flips)]
    args.report_dir.mkdir(parents=True, exist_ok=True)
    audited = audited_rows(sides, results, load_audit_key(args.audit_key))
    report = render_dryrun(sides, results, models, paths, args.stage1, audited, sorted(mirrored))
    (args.report_dir / "clean_v2_dryrun.md").write_text(report, encoding="utf-8")
    (args.report_dir / "clean_v2_examples.md").write_text(
        render_examples(sides, results, per_rule=args.examples, seed=args.seed), encoding="utf-8")
    changed = sum(1 for r in results if r.clean_log)
    print(json.dumps({"sides": len(sides), "sides_changed": changed,
                      "letters_removed": sum(r.letters_removed for r in results),
                      "models": {k: str(v) for k, v in paths.items()},
                      "thresholds": {k: m.threshold for k, m in models.items()}}, indent=1))


if __name__ == "__main__":
    main()
