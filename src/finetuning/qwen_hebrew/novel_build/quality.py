"""Stage 3 (quality) of the novel-rows build: repairs and gates of every accepted side's target text.

Input: the accepted :class:`~.schema.SideDecision` records of stage 2, the candidates (targets and photographs) and,
only for a decision written without its coverage signals, the reads. Output: one :class:`~.schema.QualityResult` per
accepted side; ``text`` is the final target.

Repairs (in this order):

* Hebrew presentation forms (U+FB1D-U+FB4F, legacy-font finals) are replaced by their NFKC form, one character at a
  time (``mojibake_final``: a flag when every one was repaired, a failure when one is left).
* Harakat are stripped as the v2.5a Arabic pool's label rule does (``derive_v25a.strip_harakat``, re-implemented):
  Arabic presentation forms to their NFKC letters, Persian yeh / keheh to the Arabic letters, tatweel, U+064B-U+0652
  and U+0670 removed, NFC, a hamza above / below that NFC could not join dropped (flag ``harakat_stripped`` when the
  text changed).
* Signs (:func:`normalise_signs`, each a flag when it changed the text): an ASCII backtick becomes ``'``
  (``backtick_to_geresh``); for PGP targets a right-to-left reversed supplied span ``>x<`` is dropped as the PGP cleaner
  drops ``<x>`` (letters supplied by the editor are not ink; ``reversed_supplied_dropped``); a lone ``<`` / ``>`` after
  a digit, at the end of a token after a Hebrew letter numeral (``ז>``), or at a line end (a merchant's half-fraction
  sign, a flourish) is dropped (``fraction_sign_dropped``).

Gates (reason codes; each fails the side unless it is listed in :data:`FLAG_ONLY`):

``under_40_letters``; ``under_3_lines`` (when the lines are known, :func:`lines_known`); ``hygiene_chars`` (any of
:data:`HYGIENE_CHARS`); ``non_text_char`` (a character outside Hebrew / Arabic letters and marks, Latin letters, digits,
common punctuation, ``[]()``, space and newline, combining marks U+0300-U+036F, the fractions of :data:`FRACTIONS`, and
for Arabic-script sides an asterisk, Greek letters and the modifier prime of papyrus numerals, after NFC);
``mojibake_final``; ``editorial_terms`` and ``edition_apparatus`` (the legacy-store cleaner's detectors,
``legacy_store_sigla.EDITORIAL_TERMS_RE`` and ``classify_family``, on the final text, plus the family or flags an
upstream cleaner recorded); the target extraction's flags (``Candidate.notes["target_flags"]``): legacy-store
``unresolved_finals`` (unmapped glyphs) and PGP ``complex_side_labels`` fail, ``not_body_text`` (spans the sigla keep
as ink), ``flattened_line_breaks``, ``notation_latin``, ``notation_parentheses`` and ``notation_glued_digits`` are
recorded as flags;
``gap_density_over_30pct`` (legacy-store sides with more than 30 gap tokens per 100 letters, the October build's
rule); for Arabic-script sides ``arabic_under_40`` and ``arabic_ratio_below_9x`` (Arabic letters at least 9 times the
Hebrew ones); ``arabic_too_dense`` (more than 1,000 Arabic letters per megapixel of the chosen photograph); coverage
signals of the decision: ``read_exceeds_target_1_5x`` (the read holds at least 1.5 times the target's letters: the
target is likely partial and would teach early stopping), ``target_exceeds_read_2_5x`` (the read holds under 40 % of
the target's letters: the target holds text the photograph does not show), ``partial_image_for_target`` (the first or
last third of the target, at least 100 letters, has no n-gram in the read; both are computed on the usable read of the
photograph that covers most of the target, and when that photograph has Kraken reads only the flag
``coverage_second_read_wanted`` asks for a page-reader read, which clears them on the next run if it covers more),
``kraken_lines_over_1_5x`` (the Kraken read has at least 1.5 times the target's lines, when the lines are known)
and ``kraken_lines_over_1_3x`` (flag, over 1.3 times and under 1.5 times); ``token_over_budget`` (question +
answer over :data:`TOKEN_BUDGET` tokens of the Qwen3-VL tokenizer; ``--no-tokens`` skips it with the flag
``tokens_unmeasured``). ``script`` is measured from the final text.

Usage::

    python -m src.finetuning.qwen_hebrew.novel_build.quality --work-dir DIR [--tokenizer NAME] [--no-tokens]
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from PIL import Image, UnidentifiedImageError

from src.datasets.cleaning.legacy_store_sigla import EDITORIAL_TERMS_RE, classify_family
from src.datasets.evaluations.helper_eval_scripts import build_arabic_benchmark as bab
from src.finetuning.qwen_hebrew import build_legacy_store_rows as lsr
from src.finetuning.qwen_hebrew.novel_build import paths
from src.finetuning.qwen_hebrew.novel_build import side_assign as sa
from src.finetuning.qwen_hebrew.novel_build.schema import (Candidate, QualityResult, ReadRecord, SideDecision,
                                                           TargetSide, read_jsonl, write_jsonl)
from src.finetuning.qwen_hebrew.prompts import SCRIPT_NEUTRAL_PROMPT

#: tokenizer of the trained model, read from the local cache only
TOKENIZER_NAME = os.environ.get("NOVEL_BUILD_TOKENIZER", "unsloth/Qwen3-VL-8B-Instruct")
MAX_SEQ = 12288
IMAGE_TOKENS = 7_000_000 // (32 * 32)          # 6,835 visual tokens of a page packed at 7 MP
PROMPT_MARGIN = 200
TOKEN_BUDGET = MAX_SEQ - IMAGE_TOKENS - PROMPT_MARGIN
MIN_LETTERS = lsr.MIN_TARGET_LETTERS           # 40
MIN_LINES = 3
MIN_ARABIC_LETTERS = 40
ARABIC_OVER_HEBREW = 9
MAX_ARABIC_PER_MEGAPIXEL = bab.MAX_LETTERS_PER_MEGAPIXEL
MAX_READ_OVER_TARGET = 1.5                     # read letters / target letters at or above: the target is partial
MIN_READ_OVER_TARGET = 0.40                    # under: the target holds text the photograph does not show
KRAKEN_LINES_REVIEW = 1.5                      # Kraken lines / target lines at or above: review
MAX_KRAKEN_LINES_OVER_TARGET = 1.3             # above (and under the review ratio): flag only
DEAD_THIRD_LETTERS = 100                       # a first or last third this long without one matched n-gram: partial
GAP = "[...]"                                  # the cleaners' gap token
MAX_GAP_DENSITY = lsr.MAX_GAP_DENSITY / 100.0  # gaps per letter (the legacy-store cleaner's 30 per 100 letters)
LEGACY_STORE_SOURCES = ("fjp", "fjp_scrape")
#: target flags (``Candidate.notes["target_flags"][side_key]``) that fail a side, with the sources they apply to
FAILING_TARGET_FLAGS: Dict[str, Tuple[str, ...]] = {"unresolved_finals": LEGACY_STORE_SOURCES,
                                                    "complex_side_labels": ("pgp",)}
#: target flags recorded on the result without failing it (``not_body_text``: the legacy store's sigla keep those
#: spans, a later hand or a pen trial, as ink on the page)
RECORDED_TARGET_FLAGS = ("not_body_text", "flattened_line_breaks", "notation_latin", "notation_parentheses",
                         "notation_glued_digits")
#: sources whose ``<x>`` marks letters supplied by the editor (the PGP cleaner drops such spans); a reversed ``>x<`` of
#: theirs is dropped the same way
SUPPLIED_ANGLE_SOURCES = ("pgp",)
HYGIENE_CHARS = "\u2423&#$_{}<>\\"
FRACTIONS = "\u00bd\u00bc\u00be\u215b\u2044"   # one half, one quarter, three quarters, one eighth, fraction slash
#: characters allowed only in Arabic-script sides (papyrus numerals): asterisk and modifier prime, with Greek letters
ARABIC_SIDE_EXTRA = "*\u02b9"
#: sources whose targets keep the manuscript's line breaks; others are flat unless the text has a newline
LINE_STRUCTURED_SOURCES = ("pgp", "ktiv", "literature", "lit_run2", "apd_v3", "aodeh_v2_1")
#: a single "line" longer than this many letters is a flat text, not a manuscript line
SINGLE_LINE_MAX_LETTERS = 80
EXCLUDED_FAMILIES = lsr.EXCLUDED_FAMILIES     # ("edition_apparatus",)
EXCLUDED_FLAGS = lsr.EXCLUDED_FLAGS           # ("editorial_terms",)
HEBREW_PRESENTATION = re.compile(r"[\ufb1d-\ufb4f]")
#: the v2.5a pool's label rule (``derive_v25a.strip_harakat``): Arabic presentation forms by NFKC, Persian yeh and
#: keheh to the Arabic letters; tatweel, tanwin, short vowels, shadda, sukun, superscript alef removed; NFC; a hamza
#: above / below that NFC could not join is dropped
ARABIC_PRESENTATION = re.compile(r"[\ufb50-\ufdff\ufe70-\ufefe]")
PERSIAN_LETTERS = str.maketrans({"\u06cc": "\u064a", "\u06a9": "\u0643"})
HARAKAT = re.compile(r"[\u0640\u064b-\u0652\u0670]")
STRAY_HAMZA = re.compile(r"[\u0654\u0655]")
COMMON_PUNCTUATION = set(".,;:!?'\"-–—/…·“”‘’«»׳״־׀׃،؛؟۔٪٫٬")
#: the inside of an angle pair: 1-20 characters, no angle, no line break, no space at either edge (a supplied span
#: hugs its letters; ``ז> ... ז<`` is two half-fraction signs, not a pair)
ANGLE_INSIDE = r"(?:[^<>\s]|[^<>\s][^<>\n]{0,18}[^<>\s])"
#: a reversed supplied span ``>x<`` (RTL text typed in visual order), not part of a doubled ``>>x<<``
REVERSED_SUPPLIED = re.compile(r"(?<![<>])>(" + ANGLE_INSIDE + r")<(?![<>])")
#: an angle pair on one line, either orientation (its signs are not lone)
ANGLE_PAIR = re.compile("<" + ANGLE_INSIDE + ">|>" + ANGLE_INSIDE + "<")

# reason codes
UNDER_LETTERS = f"under_{MIN_LETTERS}_letters"
UNDER_LINES = f"under_{MIN_LINES}_lines"
HYGIENE = "hygiene_chars"
NON_TEXT = "non_text_char"
MOJIBAKE = "mojibake_final"
EDITORIAL = EXCLUDED_FLAGS[0]
APPARATUS = EXCLUDED_FAMILIES[0]
ARABIC_UNDER = f"arabic_under_{MIN_ARABIC_LETTERS}"
ARABIC_RATIO = f"arabic_ratio_below_{ARABIC_OVER_HEBREW}x"
ARABIC_DENSE = "arabic_too_dense"
HARAKAT_STRIPPED = "harakat_stripped"
READ_OVER = "read_exceeds_target_1_5x"
READ_UNDER = "target_exceeds_read_2_5x"
PARTIAL_IMAGE = "partial_image_for_target"
KRAKEN_LINES_OVER = "kraken_lines_over_1_5x"
KRAKEN_LINES = "kraken_lines_over_1_3x"
GAP_DENSITY = "gap_density_over_30pct"
TOKENS_OVER = "token_over_budget"
TOKENS_UNMEASURED = "tokens_unmeasured"
SIZE_UNKNOWN = "image_size_unknown"
COVERAGE_UNKNOWN = "coverage_unknown"
TARGET_MISSING = "target_missing"
BACKTICK_GERESH = "backtick_to_geresh"
REVERSED_SUPPLIED_DROPPED = "reversed_supplied_dropped"
FRACTION_SIGN_DROPPED = "fraction_sign_dropped"
SECOND_READ_WANTED = "coverage_second_read_wanted"
#: codes that never fail a side (``mojibake_final`` fails only when a presentation form is left)
FLAG_ONLY = frozenset({HARAKAT_STRIPPED, KRAKEN_LINES, TOKENS_UNMEASURED, SIZE_UNKNOWN, COVERAGE_UNKNOWN,
                       BACKTICK_GERESH, REVERSED_SUPPLIED_DROPPED, FRACTION_SIGN_DROPPED, SECOND_READ_WANTED,
                       *RECORDED_TARGET_FLAGS})

_HEBREW_LETTER = re.compile("[א-ת]")
_ARABIC_LETTER = re.compile("[ء-يٱ-ۓ]")


def _count_letters(text: str) -> Tuple[int, int]:
    """Hebrew and Arabic letters (private copy of ``targets.count_letters``).

    :param text: Any text.
    :type text: str
    :return: ``(Hebrew letters U+05D0-05EA, Arabic letters U+0621-064A and U+0671-06D3)``.
    :rtype: Tuple[int, int]
    """
    return len(_HEBREW_LETTER.findall(text)), len(_ARABIC_LETTER.findall(text))


def _measure_script(text: str) -> str:
    """Script of a text (private copy of ``targets.measure_script``).

    :param text: Any text.
    :type text: str
    :return: ``mixed`` when both scripts have at least 20 letters and the minority at least 10 % of them, else
        ``hebrew`` / ``arabic`` by majority (``hebrew`` on a tie), ``none`` without letters.
    :rtype: str
    """
    hebrew, arabic = _count_letters(text)
    if not hebrew + arabic:
        return "none"
    minority = min(hebrew, arabic)
    if minority >= 20 and minority >= 0.10 * (hebrew + arabic):
        return "mixed"
    return "hebrew" if hebrew >= arabic else "arabic"


def _from_targets(name: str, fallback: Callable[..., Any]) -> Callable[..., Any]:
    """A function of the sibling ``targets`` module when that module exists, else the private copy.

    :param name: Function name.
    :type name: str
    :param fallback: Private copy.
    :type fallback: Callable[..., Any]
    :return: The function.
    :rtype: Callable[..., Any]
    """
    module_name = f"{__package__ or 'src.finetuning.qwen_hebrew.novel_build'}.targets"
    if importlib.util.find_spec(module_name) is None:
        return fallback
    return getattr(importlib.import_module(module_name), name, fallback)


measure_script: Callable[[str], str] = _from_targets("measure_script", _measure_script)
count_letters: Callable[[str], Tuple[int, int]] = _from_targets("count_letters", _count_letters)


# ===================================================================================================================
# Repairs
# ===================================================================================================================

def repair_presentation_forms(text: str) -> Tuple[str, int, int]:
    """Replace Hebrew presentation forms by their NFKC form, one character at a time.

    :param text: Target text.
    :type text: str
    :return: ``(text, characters repaired, presentation forms left)``.
    :rtype: Tuple[str, int, int]
    """
    found = len(HEBREW_PRESENTATION.findall(text))
    if not found:
        return text, 0, 0
    repaired = HEBREW_PRESENTATION.sub(lambda m: unicodedata.normalize("NFKC", m.group(0)), text)
    left = len(HEBREW_PRESENTATION.findall(repaired))
    return repaired, found - left, left


def strip_harakat(text: str) -> str:
    """The text in the v2.5a pool's Arabic spelling (``derive_v25a.strip_harakat``, re-implemented here).

    :param text: Text.
    :type text: str
    :return: Arabic presentation forms replaced by their NFKC form (one character at a time), Persian yeh / keheh
        folded, harakat, shadda, sukun, superscript alef and tatweel removed, NFC, unjoined hamza above / below dropped.
    :rtype: str
    """
    text = ARABIC_PRESENTATION.sub(lambda m: unicodedata.normalize("NFKC", m.group(0)), text).translate(PERSIAN_LETTERS)
    return STRAY_HAMZA.sub("", unicodedata.normalize("NFC", HARAKAT.sub("", unicodedata.normalize("NFC", text))))


def drop_fraction_signs(text: str) -> Tuple[str, int]:
    """Drop a lone ``<`` or ``>`` that follows a digit, ends a token right after a Hebrew letter (a letter numeral:
    ``ז>``, ``יז>``) or ends a line: a merchant's half-fraction sign or a flourish.

    A sign is lone when it is not part of an angle pair on its line (:data:`ANGLE_PAIR`).

    :param text: Text.
    :type text: str
    :return: ``(text without those signs, signs dropped)``.
    :rtype: Tuple[str, int]
    """
    dropped = 0
    lines = []
    for line in text.split("\n"):
        paired = {i for m in ANGLE_PAIR.finditer(line) for i in (m.start(), m.end() - 1)}
        kept = []
        for i, ch in enumerate(line):
            after_numeral = i > 0 and (line[i - 1].isdigit()
                                       or (_HEBREW_LETTER.match(line[i - 1]) is not None
                                           and (i + 1 == len(line) or line[i + 1].isspace())))
            if ch in "<>" and i not in paired and (after_numeral or not line[i + 1:].strip()):
                dropped += 1
                continue
            kept.append(ch)
        lines.append("".join(kept))
    return "\n".join(lines), dropped


def normalise_signs(text: str, source: str = "") -> Tuple[str, List[str]]:
    """Sign normalisations applied before the character gates, each recorded as a flag when it changed the text.

    An ASCII backtick becomes ``'`` (a geresh; ``backtick_to_geresh``); for a source whose ``<x>`` marks letters
    supplied by the editor (:data:`SUPPLIED_ANGLE_SOURCES`), a right-to-left reversed ``>x<`` is dropped as the PGP
    cleaner drops ``<x>`` (``reversed_supplied_dropped``); a lone ``<`` / ``>`` after a digit, at the end of a token
    after a Hebrew letter numeral, or at a line end is dropped (:func:`drop_fraction_signs`,
    ``fraction_sign_dropped``). Any other ``<`` / ``>`` stays and fails ``hygiene_chars``.

    :param text: Text after the repairs.
    :type text: str
    :param source: Candidate source.
    :type source: str
    :return: ``(text, flags)``.
    :rtype: Tuple[str, List[str]]
    """
    flags: List[str] = []
    if "`" in text:
        text = text.replace("`", "'")
        flags.append(BACKTICK_GERESH)
    if source in SUPPLIED_ANGLE_SOURCES:
        unreversed = REVERSED_SUPPLIED.sub("", text)
        if unreversed != text:
            text = unreversed
            flags.append(REVERSED_SUPPLIED_DROPPED)
    text, dropped = drop_fraction_signs(text)
    if dropped:
        flags.append(FRACTION_SIGN_DROPPED)
    return text, flags


def final_text(text: str, source: str = "") -> Tuple[str, List[str], List[str]]:
    """The target after the repairs and sign normalisations, with the flags and failures they raise.

    :param text: Target text.
    :type text: str
    :param source: Candidate source (for :func:`normalise_signs`).
    :type source: str
    :return: ``(final text, flags, failures)``.
    :rtype: Tuple[str, List[str], List[str]]
    """
    flags: List[str] = []
    fails: List[str] = []
    repaired, fixed, left = repair_presentation_forms(text)
    if left:
        fails.append(MOJIBAKE)
    elif fixed:
        flags.append(MOJIBAKE)
    stripped = strip_harakat(repaired)
    if stripped != unicodedata.normalize("NFC", repaired):
        flags.append(HARAKAT_STRIPPED)
    normalised, sign_flags = normalise_signs(stripped, source)
    return normalised, flags + sign_flags, fails


# ===================================================================================================================
# Gates
# ===================================================================================================================

def _allowed(ch: str, script: str = "") -> bool:
    """Whether a character may stand in a target.

    :param ch: One character.
    :type ch: str
    :param script: Measured script of the side; ``arabic`` also allows :data:`ARABIC_SIDE_EXTRA` and Greek letters
        (papyrus numerals).
    :type script: str
    :return: True for Hebrew letters and marks, Arabic letters and marks, Latin letters, digits, common punctuation,
        ``[]()``, space and newline, combining marks U+0300-U+036F (diacritical points such as a dotted gimel), and
        the fractions of :data:`FRACTIONS`.
    :rtype: bool
    """
    code = ord(ch)
    if ch in " \n[]()" or ch in COMMON_PUNCTUATION or ch in FRACTIONS or 0x0300 <= code <= 0x036F:
        return True
    if script == "arabic" and (ch in ARABIC_SIDE_EXTRA or (0x0370 <= code <= 0x03FF
                                                           and unicodedata.category(ch).startswith("L"))):
        return True
    if 0x0591 <= code <= 0x05C7 or 0x05D0 <= code <= 0x05EA or 0x05F0 <= code <= 0x05F2:
        return True
    if 0x0610 <= code <= 0x061A or 0x0620 <= code <= 0x065F or 0x066E <= code <= 0x06D3 or 0x06D5 <= code <= 0x06ED:
        return True
    if 0x06FA <= code <= 0x06FC or code == 0x06FF:
        return True
    if unicodedata.category(ch) == "Nd":
        return True
    return unicodedata.category(ch).startswith("L") and (ch.isascii() or 0x00C0 <= code <= 0x024F
                                                        or 0x1E00 <= code <= 0x1EFF)


def hygiene_hits(text: str) -> List[str]:
    """Hygiene characters in a text.

    :param text: Text.
    :type text: str
    :return: The distinct characters of :data:`HYGIENE_CHARS` found, sorted.
    :rtype: List[str]
    """
    return sorted({ch for ch in text if ch in HYGIENE_CHARS})


def non_text_chars(text: str, script: str = "") -> List[str]:
    """Characters a target may not hold, after NFC (hygiene characters are reported by :func:`hygiene_hits`).

    :param text: Text.
    :type text: str
    :param script: Measured script of the side (:func:`_allowed`).
    :type script: str
    :return: The distinct offending characters, sorted.
    :rtype: List[str]
    """
    return sorted({ch for ch in unicodedata.normalize("NFC", text)
                   if ch not in HYGIENE_CHARS and not _allowed(ch, script)})


def text_lines(text: str) -> List[str]:
    """Non-empty lines of a text.

    :param text: Text.
    :type text: str
    :return: Lines.
    :rtype: List[str]
    """
    return [line for line in text.splitlines() if line.strip()]


def lines_known(text: str, source: str) -> bool:
    """Whether a target's line breaks are the manuscript's.

    :param text: Final target text.
    :type text: str
    :param source: Candidate source.
    :type source: str
    :return: True for a line-structured source (:data:`LINE_STRUCTURED_SOURCES`) with lines of at most
        :data:`SINGLE_LINE_MAX_LETTERS` letters on average; for other sources only when the text has two lines or
        more AND they average at most that many letters (a two-line run of hundreds of letters is partial markup,
        not the manuscript's lines, so the line gates do not apply to it).
    :rtype: bool
    """
    lines = text_lines(text)
    if not lines:
        return False
    per_line = sum(count_letters(text)) / len(lines)
    if source in LINE_STRUCTURED_SOURCES:
        return per_line <= SINGLE_LINE_MAX_LETTERS
    return len(lines) >= 2 and per_line <= SINGLE_LINE_MAX_LETTERS


def upstream_marks(cand: Candidate, target: TargetSide) -> Tuple[bool, bool]:
    """Whether an upstream cleaner recorded the edition-apparatus family or the editorial-terms flag.

    :param cand: Candidate (``notes`` may carry ``family`` / ``flags``).
    :type cand: Candidate
    :param target: Target (``pointer`` may carry ``family`` / ``flags``).
    :type target: TargetSide
    :return: ``(edition apparatus, editorial terms)``.
    :rtype: Tuple[bool, bool]
    """
    holders = [target.pointer or {}, cand.notes or {}]
    apparatus = any(h.get("family") in EXCLUDED_FAMILIES for h in holders)
    editorial = any(isinstance(h.get("flags"), dict) and any(h["flags"].get(f) for f in EXCLUDED_FLAGS)
                    for h in holders)
    return apparatus, editorial


def gap_density(text: str) -> float:
    """Gap tokens per letter, as the legacy-store cleaner measures it (its ``gap_density`` is per 100 letters).

    :param text: Final target text.
    :type text: str
    :return: ``[...]`` tokens divided by Hebrew plus Arabic letters; 0.0 without letters.
    :rtype: float
    """
    letters = sum(count_letters(text))
    return text.count(GAP) / letters if letters else 0.0


def target_flag_names(cand: Candidate, side_key: str) -> List[str]:
    """The flags the target extraction recorded for one side (``Candidate.notes["target_flags"][side_key]``).

    :param cand: Candidate.
    :type cand: Candidate
    :param side_key: Side key.
    :type side_key: str
    :return: Flag names (a list as recorded, or the keys of a count mapping with a non-zero count).
    :rtype: List[str]
    """
    recorded = (cand.notes.get("target_flags") or {}).get(side_key) or []
    if isinstance(recorded, dict):
        return [name for name, count in recorded.items() if count]
    return [str(name) for name in recorded]


def target_flag_reasons(cand: Candidate, side_key: str, text: str) -> Tuple[List[str], List[str]]:
    """Failures and flags from the target extraction's flags and the legacy store's gap density.

    :param cand: Candidate.
    :type cand: Candidate
    :param side_key: Side key.
    :type side_key: str
    :param text: Final target text.
    :type text: str
    :return: ``(failures, flags)``: :data:`FAILING_TARGET_FLAGS` recorded for the side (each for its sources) and
        ``gap_density_over_30pct`` for a legacy-store side over :data:`MAX_GAP_DENSITY`; :data:`RECORDED_TARGET_FLAGS`
        recorded for the side as flags.
    :rtype: Tuple[List[str], List[str]]
    """
    names = set(target_flag_names(cand, side_key))
    fails = [name for name, sources in FAILING_TARGET_FLAGS.items() if name in names and cand.source in sources]
    if cand.source in LEGACY_STORE_SOURCES and gap_density(text) > MAX_GAP_DENSITY:
        fails.append(GAP_DENSITY)
    return fails, [name for name in RECORDED_TARGET_FLAGS if name in names]


def partial_third(thirds: Optional[Mapping[str, Any]]) -> bool:
    """Whether the first or the last third of a target is absent from the chosen photograph's read.

    :param thirds: ``coverage["thirds"]`` (``side_assign.third_matches``), or None.
    :type thirds: Optional[Mapping[str, Any]]
    :return: True when the first or the last third holds at least :data:`DEAD_THIRD_LETTERS` skeleton letters and
        none of its primary n-grams is found in the read.
    :rtype: bool
    """
    letters, matched = list((thirds or {}).get("letters") or []), list((thirds or {}).get("matched") or [])
    if len(letters) != 3 or len(matched) != 3:
        return False
    return any(letters[i] >= DEAD_THIRD_LETTERS and matched[i] == 0 for i in (0, 2))


def image_size(path: Optional[str]) -> Optional[Tuple[int, int]]:
    """Pixel size of a local photograph from its header.

    A missing or unreadable file is recorded as an unknown size (the gate then flags ``image_size_unknown``), not
    raised: this is the one file read of the stage whose failure is a status.

    :param path: Local path or None.
    :type path: Optional[str]
    :return: ``(width, height)`` or None.
    :rtype: Optional[Tuple[int, int]]
    """
    if not path or not Path(path).exists():
        return None
    try:
        with Image.open(path) as im:
            return im.size
    except (OSError, UnidentifiedImageError):
        return None


def load_tokenizer(name: str = TOKENIZER_NAME) -> Any:
    """The tokenizer, from the local Hugging Face cache only (never downloaded).

    :param name: Model id.
    :type name: str
    :return: A ``transformers`` tokenizer.
    :rtype: Any
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, local_files_only=True)


def count_tokens(tokenizer: Any, question: str, answer: str) -> int:
    """Tokens of a row's text: question plus answer, without special tokens (the mixture's length rule).

    :param tokenizer: Callable returning ``{"input_ids": [...]}``.
    :type tokenizer: Any
    :param question: Prompt.
    :type question: str
    :param answer: Target.
    :type answer: str
    :return: Token count.
    :rtype: int
    """
    return (len(tokenizer(answer, add_special_tokens=False)["input_ids"])
            + len(tokenizer(question, add_special_tokens=False)["input_ids"]))


def decision_coverage(decision: SideDecision) -> Optional[Dict[str, Any]]:
    """The coverage signals stage 2 wrote on the chosen photograph.

    :param decision: Accepted decision.
    :type decision: SideDecision
    :return: ``competitors[0]["coverage"]`` or None.
    :rtype: Optional[Dict[str, Any]]
    """
    if decision.competitors and isinstance(decision.competitors[0].get("coverage"), dict):
        return decision.competitors[0]["coverage"]
    return None


def coverage_from_reads(cand: Candidate, target: TargetSide, decision: SideDecision,
                        reads: Mapping[str, Sequence[ReadRecord]]) -> Optional[Dict[str, Any]]:
    """Coverage signals recomputed from the reads (for a decision written without them).

    :param cand: Candidate.
    :type cand: Candidate
    :param target: Target side.
    :type target: TargetSide
    :param decision: Accepted decision (``image_sha256``, ``reader``).
    :type decision: SideDecision
    :param reads: Reads per photograph.
    :type reads: Mapping[str, Sequence[ReadRecord]]
    :return: ``side_assign.coverage_of`` of the decision's read, or None without it.
    :rtype: Optional[Dict[str, Any]]
    """
    views = [sa.make_view(r) for r in reads.get(decision.image_sha256 or "", [])]
    used = [i for i, v in enumerate(views) if v.record.reader == decision.reader]
    if not used:
        return None
    side = sa.make_side(cand, cand.targets.index(target), target)
    return sa.best_coverage(side, views, used[0])


def second_read_wanted(cover: Mapping[str, Any]) -> bool:
    """Whether a low coverage rests on line-reader reads alone, so a page-reader read of the photograph may clear it.

    :param cover: Coverage signals of the decision.
    :type cover: Mapping[str, Any]
    :return: True when every usable read of the photograph is a Kraken read (``usable_readers``; for signals written
        without it, when the read they were computed on is a Kraken read).
    :rtype: bool
    """
    families = cover.get("usable_readers")
    if families is not None:
        return set(families) == {"kraken"}
    reader = cover.get("coverage_reader") or cover.get("reader") or ""
    return sa.family_of(reader) == "kraken"


def check_side(cand: Candidate, target: TargetSide, decision: SideDecision, *, tokenizer: Any = None,
               question: str = SCRIPT_NEUTRAL_PROMPT, coverage: Optional[Mapping[str, Any]] = None,
               size_of: Callable[[Optional[str]], Optional[Tuple[int, int]]] = image_size) -> QualityResult:
    """Repair and gate one accepted side.

    :param cand: Candidate.
    :type cand: Candidate
    :param target: The side's target.
    :type target: TargetSide
    :param decision: Its accepted decision.
    :type decision: SideDecision
    :param tokenizer: Tokenizer (:func:`load_tokenizer`), or None to skip the budget (flag ``tokens_unmeasured``).
    :type tokenizer: Any
    :param question: The row's question.
    :type question: str
    :param coverage: Coverage signals (default: the decision's, :func:`decision_coverage`).
    :type coverage: Optional[Mapping[str, Any]]
    :param size_of: Reads a photograph's size when the record lacks it.
    :type size_of: Callable[[Optional[str]], Optional[Tuple[int, int]]]
    :return: The result; ``ok`` is False when any reason outside :data:`FLAG_ONLY` (and outside a repaired
        ``mojibake_final``) is present.
    :rtype: QualityResult
    """
    text, flags, fails = final_text(target.text, cand.source)
    hebrew, arabic = count_letters(text)
    script = measure_script(text)
    reasons: List[str] = []
    lines = text_lines(text)
    known = lines_known(text, cand.source)
    if hebrew + arabic < MIN_LETTERS:
        reasons.append(UNDER_LETTERS)
    if known and len(lines) < MIN_LINES:
        reasons.append(UNDER_LINES)
    if hygiene_hits(text):
        reasons.append(HYGIENE)
    if non_text_chars(text, script):
        reasons.append(NON_TEXT)
    reasons.extend(fails)
    flag_fails, recorded = target_flag_reasons(cand, target.side_key, text)
    reasons.extend(flag_fails)
    flags.extend(recorded)
    apparatus, editorial = upstream_marks(cand, target)
    if editorial or EDITORIAL_TERMS_RE.search(text):
        reasons.append(EDITORIAL)
    if apparatus or classify_family(text) in EXCLUDED_FAMILIES:
        reasons.append(APPARATUS)
    if script == "arabic":
        if arabic < MIN_ARABIC_LETTERS:
            reasons.append(ARABIC_UNDER)
        if arabic < ARABIC_OVER_HEBREW * hebrew:
            reasons.append(ARABIC_RATIO)
    if script in ("arabic", "mixed"):
        image = next((im for im in cand.images if im.sha256 and im.sha256 == decision.image_sha256), None)
        size = (image.width, image.height) if image is not None and image.width and image.height else None
        if size is None and image is not None:
            size = size_of(image.path)
        if size is None:
            flags.append(SIZE_UNKNOWN)
        elif arabic / (size[0] * size[1] / 1e6) > MAX_ARABIC_PER_MEGAPIXEL:
            reasons.append(ARABIC_DENSE)
    cover = coverage if coverage is not None else decision_coverage(decision)
    if cover is None:
        flags.append(COVERAGE_UNKNOWN)
    else:
        ratio = cover.get("letters_ratio")
        if ratio is not None and ratio >= MAX_READ_OVER_TARGET:
            reasons.append(READ_OVER)
        low = ratio is not None and ratio < MIN_READ_OVER_TARGET
        if low:
            reasons.append(READ_UNDER)
        partial = partial_third(cover.get("thirds"))
        if partial:
            reasons.append(PARTIAL_IMAGE)
        if (low or partial) and second_read_wanted(cover):
            flags.append(SECOND_READ_WANTED)
        kraken_lines = cover.get("kraken_lines")
        if known and lines and kraken_lines is not None:
            if kraken_lines / len(lines) >= KRAKEN_LINES_REVIEW:
                reasons.append(KRAKEN_LINES_OVER)
            elif kraken_lines / len(lines) > MAX_KRAKEN_LINES_OVER_TARGET:
                flags.append(KRAKEN_LINES)
    tokens: Optional[int] = None
    if tokenizer is None:
        flags.append(TOKENS_UNMEASURED)
    else:
        tokens = count_tokens(tokenizer, question, text)
        if tokens > TOKEN_BUDGET:
            reasons.append(TOKENS_OVER)
    ok = not reasons
    return QualityResult(row_id=cand.row_id, side_key=target.side_key, ok=ok, reasons=reasons + flags, script=script,
                         letters_hebrew=hebrew, letters_arabic=arabic, tokens=tokens, text=text)


def find_target(cand: Candidate, side_key: str) -> Optional[TargetSide]:
    """The candidate's target of a side key.

    :param cand: Candidate.
    :type cand: Candidate
    :param side_key: Side key.
    :type side_key: str
    :return: The first target with that key, or None.
    :rtype: Optional[TargetSide]
    """
    return next((t for t in cand.targets if t.side_key == side_key), None)


def run_quality(cands: Mapping[str, Candidate], decisions: Iterable[SideDecision], *, tokenizer: Any = None,
                reads: Optional[Mapping[str, Sequence[ReadRecord]]] = None,
                size_of: Callable[[Optional[str]], Optional[Tuple[int, int]]] = image_size) -> List[QualityResult]:
    """Gate every accepted side.

    :param cands: Candidates by row id.
    :type cands: Mapping[str, Candidate]
    :param decisions: Side decisions (only ``accepted`` ones are gated).
    :type decisions: Iterable[SideDecision]
    :param tokenizer: Tokenizer, or None (flag ``tokens_unmeasured``).
    :type tokenizer: Any
    :param reads: Reads per photograph, used only for a decision without coverage signals.
    :type reads: Optional[Mapping[str, Sequence[ReadRecord]]]
    :param size_of: Photograph size reader.
    :type size_of: Callable[[Optional[str]], Optional[Tuple[int, int]]]
    :return: One result per accepted side, in decision order.
    :rtype: List[QualityResult]
    """
    out = []
    for decision in decisions:
        if decision.outcome != "accepted":
            continue
        cand = cands.get(decision.row_id)
        target = find_target(cand, decision.side_key) if cand is not None else None
        if cand is None or target is None:
            out.append(QualityResult(decision.row_id, decision.side_key, False, [TARGET_MISSING], "none", 0, 0, None,
                                     ""))
            continue
        coverage = decision_coverage(decision)
        if coverage is None and reads:
            coverage = coverage_from_reads(cand, target, decision, reads)
        out.append(check_side(cand, target, decision, tokenizer=tokenizer, coverage=coverage, size_of=size_of))
    return out


# ===================================================================================================================
# CLI
# ===================================================================================================================

def summarize(results: Sequence[QualityResult]) -> str:
    """Pass counts and reason counts.

    :param results: Results.
    :type results: Sequence[QualityResult]
    :return: Printable text.
    :rtype: str
    """
    reasons: Counter = Counter(r for res in results for r in res.reasons)
    scripts: Counter = Counter(res.script for res in results if res.ok)
    passed = sum(res.ok for res in results)
    return "\n".join([f"{passed:,}/{len(results):,} accepted sides pass; scripts of the passed: {dict(scripts)}",
                      "  reasons: " + ", ".join(f"{r} {n:,}" for r, n in reasons.most_common())])


def build_parser() -> argparse.ArgumentParser:
    """The command line.

    :return: The parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description="Stage 3 (quality): repairs and gates of the accepted sides.")
    parser.add_argument("--work-dir", type=Path, default=paths.DEFAULT_WORK_DIR, help="build work dir")
    parser.add_argument("--candidates", type=Path, default=None, help="candidates (default: the images stage file)")
    parser.add_argument("--sides", type=Path, default=None, help="side decisions (default: the sides stage file)")
    parser.add_argument("--reads", type=Path, default=None,
                        help="reads, for decisions without coverage signals (default: <work dir>/reads/reads.jsonl)")
    parser.add_argument("--out", type=Path, default=None, help="output (default: the quality stage file)")
    parser.add_argument("--tokenizer", default=TOKENIZER_NAME, help="tokenizer id, read from the local cache only")
    parser.add_argument("--no-tokens", action="store_true", help="skip the token budget (flag tokens_unmeasured)")
    parser.add_argument("--sources", default=None, help="comma-separated sources (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="gate at most this many accepted sides")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Gate the accepted sides of a work dir and write the quality stage file.

    The stage is recomputed from its inputs on every run (it takes minutes), so there is nothing to resume.

    :param argv: Command-line arguments (None: ``sys.argv``).
    :type argv: Optional[List[str]]
    """
    args = build_parser().parse_args(argv)
    cands = {c.row_id: c for c in read_jsonl(args.candidates or paths.stage_path(args.work_dir, "images"), Candidate)}
    sources = {s.strip() for s in args.sources.split(",") if s.strip()} if args.sources else None
    decisions = [d for d in read_jsonl(args.sides or paths.stage_path(args.work_dir, "sides"), SideDecision)
                 if d.outcome == "accepted"
                 and (sources is None or getattr(cands.get(d.row_id), "source", None) in sources)]
    if args.limit is not None:
        decisions = decisions[:args.limit]
    reads: Dict[str, List[ReadRecord]] = {}
    reads_path = args.reads or args.work_dir / sa.READS_FILE
    if any(decision_coverage(d) is None for d in decisions) and reads_path.exists():
        for record in read_jsonl(reads_path, ReadRecord):
            reads.setdefault(record.sha256, []).append(record)
    tokenizer = None if args.no_tokens else load_tokenizer(args.tokenizer)
    results = run_quality(cands, decisions, tokenizer=tokenizer, reads=reads)
    out = args.out or paths.stage_path(args.work_dir, "quality")
    write_jsonl(out, results)
    print(summarize(results))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
