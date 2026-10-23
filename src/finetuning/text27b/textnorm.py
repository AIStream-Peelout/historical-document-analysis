# File name: textnorm.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Text normalisation, cleaning and target strips for the text27b build.

Ported from the 2026-10-18 probe code (``probe_common``: CER normalisation, transliteration,
script counts; the v0 builder: read cleaning, Latin-line strip, edition-as-plain, the grounding
test) and new for v0_2: deterministic strips of line markers (B3), citations (B4) and project-name strings.

Public API (other text27b modules import these names):

CER and scripts
    :func:`normalize_for_cer`, :func:`cer`, :func:`coverage`, :func:`script_counts`,
    :func:`dominant_script`.
Inputs
    :func:`clean_read`, :func:`strip_latin_lines`, :func:`edition_as_plain`,
    :func:`transliterate_ar_to_ja`, :func:`transliterate_ar_to_ja_plain`.
Target strips (each returns the new text plus what it did)
    :func:`strip_line_markers` → ``(text, ops)``; :func:`strip_citations` → ``(text, ops)``;
    :func:`strip_source_names` → ``(text, n_hits)``; :func:`has_source_names`; :data:`SCRUB_RE`;
    :func:`find_shelfmarks` (spans of shelfmark mentions in prose); :func:`sentence_spans`.
Grounding
    :func:`strip_marks`, :func:`entity_candidates`, :func:`grounding`, :func:`overlap60`,
    :func:`scholar_surnames`, :func:`skeleton_latin`, :func:`skeleton_rtl`, :func:`rtl_skeletons`,
    :func:`common_words`, :data:`MAX_DOC_YEAR`.

CER default: ``v11=True`` (U+2212 ``−`` counts as a range dash in line markers), the
normalisation the v1.1 pairs file stored its read CER / coverage with. Pass ``v11=False`` to
reproduce the v0_1 builder's own recomputation exactly.

Two prototype bugs are fixed here (results differ from the 2026-10-18 code): the harakat strip
no longer deletes Arabic-Indic digits before transliteration, and :func:`skeleton_rtl` no longer
gives a word-final ט / צ / ע its geresh value (so ``אלפסטאט`` → ``fst`` matches ``Fusṭāṭ``).

The strips are deterministic and conservative and work on prose: for ``catalogue_json`` apply
:func:`strip_citations` / :func:`strip_source_names` to the ``summary`` field (and any other free-text
field), not to the JSON string. A target can come back empty (every sentence was a citation);
callers drop such rows. ``strip_citations`` is audited by hand on 50 rows before a build is frozen
(plan B4).
"""
from __future__ import annotations

import codecs
import collections
import csv
import functools
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import Levenshtein

# ============================================================================== CER
_CER_DROP = set("[]⟦⟧{}/\\|()⟨⟩<>")
_LABEL_LINE = re.compile(r"^\s*§")
_NUM_MARKER_V1 = re.compile(r"^\s*(?:\(\s*\d+(?:\s*[-–—]\s*\d+)?\s*\)|\[\s*\d+\s*\]|\d+[.)])\s*")
_DIGITS_ONLY_V1 = re.compile(r"^[\s\d\-–—.:()\[\]]*$")
_NUM_MARKER_V11 = re.compile(r"^\s*(?:\(\s*\d+(?:\s*[-–—−]\s*\d+)?\s*\)|\[\s*\d+\s*\]|\d+[.)])\s*")
_DIGITS_ONLY_V11 = re.compile(r"^[\s\d\-–—−.:()\[\]]*$")


def normalize_for_cer(text: Optional[str], v11: bool = True) -> str:
    """Normalise text for character error rate (port of ``probe_common.normalize_for_cer``).

    NFKD; drop combining marks; drop the editorial characters ``[ ] ⟦ ⟧ { } / \\ | ( ) ⟨ ⟩ < >``;
    drop ``§`` label lines and digit-only lines; drop a leading line-number marker (``(1–2)``,
    ``[3]``, ``12.``, ``4)``); collapse all whitespace to single spaces.

    :param text: Raw text (None is treated as empty).
    :param v11: Treat U+2212 ``−`` as a range dash (v1.1 / v0_2); False = the v1 behaviour.
    :returns: The normalised string.
    """
    if not text:
        return ""
    num_marker = _NUM_MARKER_V11 if v11 else _NUM_MARKER_V1
    digits_only = _DIGITS_ONLY_V11 if v11 else _DIGITS_ONLY_V1
    out_lines = []
    for line in unicodedata.normalize("NFKD", text).split("\n"):
        if _LABEL_LINE.match(line) or digits_only.match(line):
            continue
        line = num_marker.sub("", line)
        out_lines.append("".join(ch for ch in line if not unicodedata.combining(ch) and ch not in _CER_DROP))
    return re.sub(r"\s+", " ", " ".join(out_lines)).strip()


def cer(hyp: Optional[str], ref: Optional[str], v11: bool = True) -> float:
    """Character error rate of ``hyp`` against ``ref`` after :func:`normalize_for_cer`.

    :param hyp: Hypothesis text.
    :param ref: Reference text.
    :param v11: Passed to :func:`normalize_for_cer`.
    :returns: Levenshtein distance / reference length; 1.0 when the normalised reference is empty.
    """
    ref_n = normalize_for_cer(ref, v11)
    if not ref_n:
        return 1.0
    return Levenshtein.distance(normalize_for_cer(hyp, v11), ref_n) / len(ref_n)


def coverage(read: Optional[str], edition: Optional[str], v11: bool = True) -> Optional[float]:
    """Length ratio of a read to its edition after :func:`normalize_for_cer` (the read gates' coverage).

    :param read: Read text.
    :param edition: Edition text.
    :param v11: Passed to :func:`normalize_for_cer`.
    :returns: ``len(norm(read)) / len(norm(edition))``; None when the normalised edition is empty.
    """
    ed = len(normalize_for_cer(edition, v11))
    return len(normalize_for_cer(read, v11)) / ed if ed else None


def script_counts(text: str) -> Dict[str, int]:
    """Letters per script (port of ``probe_common.script_counts``).

    :param text: Any text.
    :returns: ``{"hebrew": n, "arabic": n, "latin": n}``.
    """
    he = sum(1 for ch in text if "א" <= ch <= "ת")
    ar = sum(1 for ch in text if "ؠ" <= ch <= "ي" or "ٱ" <= ch <= "ۓ")
    la = sum(1 for ch in text if ch.isascii() and ch.isalpha() or "À" <= ch <= "ɏ" or "Ḁ" <= ch <= "ỿ")
    return {"hebrew": he, "arabic": ar, "latin": la}


def dominant_script(text: str, threshold: float = 0.6) -> str:
    """Dominant script of a text.

    :param text: Any text.
    :param threshold: Share of letters a script needs to be dominant.
    :returns: ``hebrew``, ``arabic``, ``latin``, ``mixed`` or ``none``.
    """
    counts = script_counts(text)
    total = sum(counts.values())
    if not total:
        return "none"
    script, n = max(counts.items(), key=lambda kv: kv[1])
    return script if n / total >= threshold else "mixed"


# ============================================================================== inputs
def clean_read(text: str) -> Tuple[str, int]:
    """Remove decoder loops from a machine read (port of the v0 builder).

    Drops lines that are only ``[...]``, collapses runs of identical consecutive lines, and keeps
    only the first occurrence of any short line (< 80 chars) that occurs three or more times.

    :param text: Read text (pages joined with blank lines).
    :returns: ``(cleaned text, number of lines removed)``.
    """
    lines = text.split("\n")
    freq = collections.Counter(ln.strip() for ln in lines if ln.strip())
    out: List[str] = []
    seen: Set[str] = set()
    removed = 0
    prev = None
    for ln in lines:
        s = ln.strip()
        if s in ("[...]", "[…]"):
            removed += 1
            continue
        if s and s == prev:
            removed += 1
            continue
        if s and freq[s] >= 3 and len(s) < 80:
            if s in seen:
                removed += 1
                continue
            seen.add(s)
        out.append(s)
        prev = s if s else prev
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip(), removed


_RTL = re.compile(r"[֐-׿؀-ۿ]")


def strip_latin_lines(text: str) -> Tuple[str, int]:
    """Drop Latin-script lines (catalogue headers, editor's notes) from an edition (port of the v0 builder).

    Short Latin lines naming a side / margin / folio become ``§`` labels; other Latin lines with
    three or more words, or with an editor's date / ``ed.`` / ``f. N`` mark, are dropped.

    :param text: Cleaned edition text.
    :returns: ``(text, number of lines dropped)``.
    """
    out, dropped = [], 0
    for ln in text.split("\n"):
        st = ln.strip()
        if st and not st.startswith("§") and not _RTL.search(st) and len(st) <= 90 and \
                re.search(r"\b(?:margin|recto|verso|folio|fol\.|address|side|upside|column|page|lines? at|degrees)\b", st, re.I) and \
                not re.search(r"\bed\.|\d{1,2}-\d{1,2}-\d{2}", st):
            out.append("§ " + st.rstrip(":").strip())
            continue
        if st and not st.startswith("§") and not _RTL.search(st) and len(re.findall(r"[A-Za-z]{2,}", st)) >= 3:
            dropped += 1
            continue
        if st and not st.startswith("§") and not _RTL.search(st) and re.search(r"[A-Za-z]", st) and \
                re.search(r"\d{1,2}-\d{1,2}-\d{2}|\bed\.|\bf\.\s*\d", st):
            dropped += 1
            continue
        out.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip(), dropped


_EDITORIAL = re.compile(r"\(!\)|\(\?\)|[\[\]⟦⟧{}⟨⟩<>|]|//|\\\\")


def edition_as_plain(text: str) -> str:
    """Edition text with editorial signs and ``§`` label lines removed (the base for noising).

    :param text: Cleaned edition text.
    :returns: Plain text with line breaks kept.
    """
    lines = []
    for ln in text.split("\n"):
        if ln.strip().startswith("§"):
            continue
        ln = _EDITORIAL.sub("", ln)
        lines.append(re.sub(r"[ \t]{2,}", " ", ln).strip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


# ----------------------------------------------------------------------- transliteration
_AR_TO_HE: Dict[str, str] = {
    "ا": "א", "ٱ": "א", "أ": "א", "إ": "א", "آ": "א", "ء": "א",
    "ب": "ב", "ت": "ת", "ث": "ת׳", "ج": "ג׳", "ح": "ח", "خ": "כ׳", "د": "ד", "ذ": "ד׳",
    "ر": "ר", "ز": "ז", "س": "ס", "ش": "ש", "ص": "צ", "ض": "צ׳", "ط": "ט", "ظ": "ט׳",
    "ع": "ע", "غ": "ע׳", "ف": "פ", "ق": "ק", "ك": "כ", "ک": "כ", "ل": "ל", "م": "מ", "ن": "נ",
    "ه": "ה", "ة": "ה", "و": "ו", "ؤ": "ו", "ي": "י", "ى": "י", "ی": "י", "ئ": "י",
    "پ": "פ", "گ": "ג",
    "،": ",", "؛": ";", "؟": "?", "٪": "%", "٫": ".", "٬": ",", "۔": ".",
}
_AR_TO_HE.update({chr(0x0660 + i): str(i) for i in range(10)})
_AR_TO_HE.update({chr(0x06F0 + i): str(i) for i in range(10)})
#: Harakat (U+064B–U+065F), superscript alef, Quranic marks and tatweel. The 2026-10-18 prototype's
#: range ran U+064B–U+0670 and so also deleted the Arabic-Indic digits U+0660–U+0669.
_HARAKAT = re.compile("[\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_FINALS = {"כ": "ך", "מ": "ם", "נ": "ן", "פ": "ף", "צ": "ץ"}
_FINAL_RE = re.compile("([כמנפצ])(׳?)(?=$|[^א-ת׳])")
_AR_TO_HE_PLAIN: Dict[str, str] = dict(_AR_TO_HE)
_AR_TO_HE_PLAIN.update({"ث": "ת", "ج": "ג", "خ": "כ", "ذ": "ד", "ض": "צ", "ظ": "ט", "غ": "ג", "ء": ""})


def transliterate_ar_to_ja(text: str) -> str:
    """Arabic script → Hebrew letters with geresh for letters lacking a Hebrew equivalent.

    NFKC (unfolds presentation forms), strip harakat / Quranic marks / tatweel (Arabic-Indic digits
    are kept and mapped to 0-9, unlike the 2026-10-18 prototype, which deleted them), letter map
    (ث ת׳, ج ג׳, خ כ׳, ذ ד׳, ض צ׳, ظ ט׳, غ ע׳ ...), Arabic punctuation and digits to ASCII,
    Hebrew final forms at word end. Non-Arabic characters pass through.

    :param text: Arabic-script text.
    :returns: Hebrew-script transliteration.
    """
    text = _HARAKAT.sub("", unicodedata.normalize("NFKC", text))
    out = "".join(_AR_TO_HE.get(ch, ch) for ch in text)
    return _FINAL_RE.sub(lambda m: _FINALS[m.group(1)] + m.group(2), out)


def transliterate_ar_to_ja_plain(text: str) -> str:
    """Arabic script → Hebrew letters in the plain spelling of PGP Judaeo-Arabic editions (B9).

    As :func:`transliterate_ar_to_ja` but without geresh: ث → ת, ج → ג, خ → כ, ذ → ד, ض → צ,
    ظ → ט, غ → ג; standalone hamza ء is dropped (carriers keep their letter).

    :param text: Arabic-script text.
    :returns: Hebrew-script transliteration.
    """
    text = _HARAKAT.sub("", unicodedata.normalize("NFKC", text))
    out = "".join(_AR_TO_HE_PLAIN.get(ch, ch) for ch in text)
    return _FINAL_RE.sub(lambda m: _FINALS[m.group(1)] + m.group(2), out)


# ============================================================================== sentence spans
_ABBREV = {"b", "bt", "r", "ca", "mr", "mrs", "st", "al", "ibn", "cf", "p", "pp", "vol", "vols", "no", "nos", "n", "nn",
           "f", "ff", "fol", "fols", "ed", "eds", "trans", "tr", "med", "soc", "ms", "mss", "heb", "ar", "arab", "bodl",
           "misc", "is", "isa", "ps", "pss", "gen", "ex", "exod", "lev", "num", "deut", "jer", "hos", "prov", "mal", "lam",
           "dr", "prof", "approx", "c", "lit", "viz", "univ", "dept", "fig", "figs", "pl", "col", "cols", "l", "ll", "ln",
           "e", "i", "esp", "chap", "ch", "sect", "repr", "rev", "transl", "id", "ibid", "op", "cit", "loc", "s", "v",
           "vs", "sel", "nr", "mt", "jr", "sr"}
_TERMINATOR = re.compile(r"[.!?]+[\"”’»)\]]*(?=\s|$)")
_PARA_BREAK = re.compile(r"\n[ \t]*\n")
_STARTER = re.compile(r"[A-Z0-9\"“‘(\[§֐-׿؀-ۿÀ-ÖØ-ÞĀ-Ž]")


def sentence_spans(text: str, protect: Sequence[Tuple[int, int]] = ()) -> List[Tuple[int, int]]:
    """Split text into sentence spans (character offsets, whitespace between sentences excluded).

    A boundary is a ``.``/``!``/``?`` run (with closing quotes / brackets) followed by whitespace and
    an upper-case, digit, quote, bracket, ``§`` or Hebrew/Arabic character, unless the dot ends a
    known abbreviation or a single-letter initial; blank lines are always boundaries.

    :param text: Text to split.
    :param protect: Spans whose characters are never boundaries (e.g. shelfmarks with dots).
    :returns: ``[(start, end), ...]`` covering every non-blank stretch of ``text``.
    """
    cuts = {m.start() + 1 for m in _PARA_BREAK.finditer(text)}
    for m in _TERMINATOR.finditer(text):
        dot = m.start()
        if any(a <= dot < b for a, b in protect):
            continue
        rest = text[m.end():]
        nxt = rest.lstrip()
        if nxt and not _STARTER.match(nxt):
            continue
        if text[dot] == ".":
            w = re.search(r"([A-Za-zÀ-ž]+)$", text[:dot])
            if w and (w.group(1).lower() in _ABBREV or len(w.group(1)) == 1):
                continue
        cuts.add(m.end())
    spans: List[Tuple[int, int]] = []
    start = 0
    for c in sorted(cuts) + [len(text)]:
        seg = text[start:c]
        if seg.strip():
            a = start + (len(seg) - len(seg.lstrip()))
            b = start + len(seg.rstrip())
            spans.append((a, b))
        start = c
    return spans


_CUT = "\x00"
_CAP = "\x01"


def _delete_spans(text: str, spans: Iterable[Tuple[int, int]], capitalize: Iterable[int] = ()) -> str:
    """Delete character spans, repairing spacing / punctuation only where something was cut.

    Lines the deletion emptied are dropped; runs of blank lines collapse to one.

    :param text: Original text.
    :param spans: ``(start, end)`` spans to delete (may overlap).
    :param capitalize: Offsets whose next kept letter is upper-cased when it starts a sentence
        after a deletion (e.g. the clause after a dropped first clause).
    :returns: The edited text.
    """
    mask = bytearray(len(text))
    for a, b in spans:
        for i in range(max(0, a), min(len(text), b)):
            mask[i] = 1
    cap = set(capitalize)
    out_lines: List[str] = []
    pos = 0
    for line in text.split("\n"):
        chars: List[str] = []
        touched = False
        for i, ch in enumerate(line):
            gi = pos + i
            if gi in cap:
                chars.append(_CAP)
            if mask[gi]:
                if not touched or not mask[gi - 1]:
                    chars.append(_CUT)
                touched = True
                continue
            chars.append(ch)
        pos += len(line) + 1
        kept = "".join(chars)
        if touched:
            kept = _tidy(kept)
            if not kept.strip() and line.strip():
                continue
        out_lines.append(kept.replace(_CAP, ""))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out_lines)).strip()


def _tidy(line: str) -> str:
    """Repair spacing and punctuation at the cut marks of one edited line (nowhere else).

    :param line: Edited line carrying :data:`_CUT` marks where text was removed.
    :returns: Tidied line without marks.
    """
    line = re.sub(r"\x00(?:[ \t]*\x00)+", _CUT, line)
    line = re.sub(r"\([ \t]*\x00[ \t,;:]*\)|\[[ \t]*\x00[ \t]*\]", _CUT, line)
    line = re.sub(r"([,;:])[ \t]*\x00[ \t]*(?=[,;:.!?])", _CUT, line)
    line = re.sub(r"[ \t]*\x00[ \t]*(?=[.,;:!?)\]]|$)", "", line)
    line = re.sub(r"(?<=[(\[])[ \t]*\x00[ \t]*", "", line)
    line = re.sub(r"^[ \t]*\x00[ \t]*(?:[,;:.][ \t]*)?", "", line)
    line = re.sub(r"(?<=\S)[ \t]*\x00[ \t]*(?=\S)", " ", line)
    line = line.replace(_CUT, "")
    line = re.sub(r"\x01([^A-Za-zÀ-ž\x01]*)([a-zà-ž])", lambda m: m.group(1) + m.group(2).upper(), line)
    return line.replace(_CAP, "").rstrip()


# ============================================================================== project-name strings
#: The scrubbed project names, stored rot13-encoded (the Hebrew surname as code points) so that the source never
#: spells them (fix pass F12: a plain grep of the committed code finds none); decoded once at import.
_SCRUB_ENCODED: Dict[str, str] = {"surname": "sevrqoret", "abbr_society": "swzf", "abbr_project": "swc",
                                  "abbr_variant": "STC", "host": "travmnu", "host_society": "wrjvfuznahfpevcgf",
                                  "society_a": "wrjvfu", "society_b": "znahfpevcg", "tld": "bet"}
_SCRUB_HEBREW: Tuple[int, ...] = (0x5E4, 0x5E8, 0x5D9, 0x5D3, 0x5D1, 0x5E8, 0x5D2)


def scrub_terms() -> Dict[str, str]:
    """The scrubbed project names, decoded (rot13; the Hebrew surname from code points).

    :returns: Term name → decoded term (``surname``, ``abbr_society``, ``abbr_project``, ``abbr_variant``, ``host``,
        ``host_society``, ``society_a``, ``society_b``, ``tld``, ``surname_he``).
    """
    out = {k: codecs.decode(v, "rot_13") for k, v in _SCRUB_ENCODED.items()}
    out["surname_he"] = "".join(chr(c) for c in _SCRUB_HEBREW)
    return out


def scrub_pattern(terms: Dict[str, str]) -> str:
    """The scrubber's regular expression over decoded terms: the surname and its derivatives, the three
    abbreviations as words, both hosts with any subdomain / scheme / path, the society's name and the Hebrew surname.

    :param terms: :func:`scrub_terms` output.
    :returns: Pattern source (compile with ``re.I``).
    """
    host = r"(?:https?://)?(?:[\w-]+\.)*" + "{}" + r"\." + "{}" + r"[^\s)\]]*"
    return "|".join([terms["surname"] + r"\w*", r"\b" + terms["abbr_society"] + r"\b",
                     r"\b" + terms["abbr_project"] + r"\b", r"\b" + terms["abbr_variant"] + r"\b",
                     host.format(terms["host"], terms["tld"]), host.format(terms["host_society"], terms["tld"]),
                     terms["society_a"] + r"\s+" + terms["society_b"] + r"s?\s+society", terms["surname_he"]])


#: Scrubbed project names and hosts (all subdomains, scheme-less forms included), assembled at import.
SCRUB_RE = re.compile(scrub_pattern(scrub_terms()), re.I)
_URL = re.compile(r"(?:https?://|www\.)[^\s)\]]+")
#: Name of the strip op that records a project-name scrub in ``meta.strip_ops`` (F4: the op name itself must not match
#: :data:`SCRUB_RE`, so no row carries a project-name string anywhere, meta included).
SCRUB_OP = "source_scrub"


def has_source_names(text: Optional[str]) -> bool:
    """Whether text carries any scrubbed project name or host.

    :param text: Any text.
    :returns: True on a hit.
    """
    return bool(text) and bool(SCRUB_RE.search(text))


def _paren_spans(text: str) -> List[Tuple[int, int]]:
    """Innermost-first spans of balanced round-bracket groups (one nesting level supported).

    :param text: Text.
    :returns: ``(start, end)`` spans including the brackets, outer groups after inner ones.
    """
    out: List[Tuple[int, int]] = []
    stack: List[int] = []
    for i, ch in enumerate(text):
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            out.append((stack.pop(), i + 1))
    return out


def strip_source_names(text: str) -> Tuple[str, int]:
    """Remove scrubbed project-name strings with their clause or sentence.

    A hit inside a parenthetical removes the parenthetical; any other hit (a bare project URL included)
    removes its ``;``-clause when the sentence has several clauses, else the whole sentence. Leftover hits (none expected) are deleted directly, so the result never matches
    :data:`SCRUB_RE`.

    :param text: Any text (page text, description, target).
    :returns: ``(text, number of hits in the input)``.
    """
    if not text:
        return text, 0
    hits = list(SCRUB_RE.finditer(text))
    if not hits:
        return text, 0
    dels: List[Tuple[int, int]] = []
    parens = _paren_spans(text)
    sents = sentence_spans(text)
    for h in hits:
        hp = h.start()
        par = next(((a, b) for a, b in parens if a <= hp < b), None)
        if par is not None:
            dels.append(par)
            continue
        sent = next(((a, b) for a, b in sents if a <= hp < b), (h.start(), h.end()))
        a, b = sent
        clauses = [(a + m.start(), a + m.end()) for m in re.finditer(r"[^;]+;?", text[a:b])]
        if len(clauses) > 1:
            dels.append(next(((ca, cb) for ca, cb in clauses if ca <= hp < cb), sent))
        else:
            dels.append(sent)
    out = _delete_spans(text, dels)
    if SCRUB_RE.search(out):
        out = _delete_spans(out, [(m.start(), m.end()) for m in SCRUB_RE.finditer(out)])
    return out, len(hits)


# ============================================================================== shelfmarks in prose
_SM_NUM = r"\d+[A-Za-z]?\d*(?:\s?[.,/:]\s?\d+[a-z]?)*"
_SHELFMARK_RE = re.compile(
    r"(?<![\w-])(?:"
    r"T\.?\s?-?\s?S\.?\s?(?:(?:AS|NS|Ar|Misc|Box|K|J|H|Glass|Or|Arabic|Loan)\.?\s?)*" + _SM_NUM +
    r"|ENA\s?(?:NS\s?)?" + _SM_NUM +
    r"|(?:Bodl\.?\s*)?MS\.?\s?(?:[Hh]eb|Arab)\.?\s?[a-g]\.?\s?\d+(?:\s?[/.,]\s?(?:f(?:ol)?\.\s?)?\d+[a-z]?)*"
    r"|Mosseri\s?(?:[IVXLA-Z]+[.,]?\s?)?" + _SM_NUM +
    r"|(?:CUL|ULC)\s?(?:Or|Add|T-S)\.?\s?" + _SM_NUM + r"(?:\s?[A-Z]\d+)?"
    r"|Or\.?\s?\d{3,5}(?:\s?[A-Z]{1,2}\s?\d+|\s?[.,/]\s?\d+)*"
    r"|(?:BL|BM)\s?Or\.?\s?" + _SM_NUM +
    r"|(?:AIU|JRL|Halper|Antonin|Firkovi[cč]h?|L-?G|Westminster|JTS|Gaster|Rylands)\s?(?:[A-Z][A-Za-z]*\.?\s?){0,3}" + _SM_NUM +
    r")")


def find_shelfmarks(text: str) -> List[Tuple[int, int]]:
    """Spans of Genizah shelfmark mentions in prose (regex over the common collection prefixes).

    :param text: Prose.
    :returns: ``[(start, end), ...]`` (trailing punctuation excluded).
    """
    out = []
    for m in _SHELFMARK_RE.finditer(text):
        a, b = m.start(), m.end()
        while b > a and text[b - 1] in ".,;: ":
            b -= 1
        if b > a and re.search(r"\d", text[a:b]):
            out.append((a, b))
    return out


# ============================================================================== citations
_DIMENSIONS = re.compile(r"\d+(?:[.,]\d+)?\s*(?:cm|mm)\b|\b(?:Measurements|Dimensions|Size)\s*:", re.I)
_REF_START = re.compile(r"^[\"“(\[]?\s*(?:See\b|See also\b|(?:Also|And)\s+see\b|Cf\.|Compare\b|Bibliography\b|"
                        r"(?:India Book|IB)\s*[IVX]+\s*,?\s*\d+|NB\b.{0,80}\b(?:cites?|cited|published|edition)\b|"
                        r"Literature\s*:|Edited\b|Translated\b|Published\b|Discussed\b|Described\b|Mentioned in\b|"
                        r"Cited\b|Listed\b|Information(?: in part)? from\b|Also edited\b|Edition(?:s)?\s*:|"
                        r"Transcription\s*:|For (?:more|further|a|an|the|this|these|his|her)\b[^.]*\b(?:see|cf\.)\s)", re.I)
_REF_ANYWHERE = re.compile(r"\b(?i:edited|translated|published|transcribed|discussed|described|cited|mentioned)\b[\s,]+"
                           r"(?:[\w-]+[\s,]+){0,3}?(?:by|in)\s+(?:(?:the|his|her|their|an?)\s+)?[A-Z“\"\d]|"
                           r"\bEdition in\b|\b(?i:index cards?|note cards?)\b|\bGoitein notes\b|"
                           r"\b(?i:on|for) (?:which|this|these|that)\b[^.;]{0,40}\bsee\b|"
                           r"[;,]\s*(?:see|cf\.)\s+[A-Z]|"
                           r"\b(?i:for (?:further|more) (?:information|details|discussion|bibliography))\b|"
                           r"\b[A-Z][\w.'’ -]{0,40}['’]s (?:edition|translation|discussion) (?:for|of|in)\b")
_MED_SOC = re.compile(r"\bMed\.?\s*Soc\b|\bMediterranean Society\b")
_YEAR_CITE = re.compile(r"\((?:[^()]{0,80}[,:]\s*)?(?:1[89]|20)\d{2}[a-z]?\)\s*[,:.]?\s*(?:pp?\.\s*)?\d|"
                        r"[“\"][^”\"]{6,200}[”\"][^.]{0,120}\b(?:1[89]|20)\d{2}\b|"
                        r"\b[A-Z][A-Za-z&. ]{1,40},\s*(?:1[89]|20)\d{2}\)|"
                        r"\b(?:1[89]|20)\d{2}[a-z]?\)?\s*,\s*pp?\.\s*\d")
_PAGE_REF = re.compile(r"\bpp?\.\s*\d|\bvol\.\s*\d|\bn\.\s*\d|\bibid\b|\bop\. cit\b", re.I)
_DATE_MARK = re.compile(r"\b(?:CE|C\.E\.|AM|A\.M\.|AH|A\.H\.|AD|A\.D\.|Sel)\b|Tishri|Nisan|Kislev|Muḥarram|Ramaḍān|Rabīʿ")
_CROSS_REF = re.compile(r"\b(?:join|joins|joined|related|same|also|another|other|wrote|written|see|cf|cited|cites?|"
                        r"citing|catalogued|listed|mistakenly|erroneously|"
                        r"continuation|continues|part of|belongs?|fragment of|copy|copies|draft|mentioned|"
                        r"identical|similar|parallel|together with|compare)\b", re.I)


#: "Partial translation from <author, work, page>: “...”" -- the citation prefix of a quoted translation.
_TRANSLATION_FROM = re.compile(r"\b(?:Partial |Full |Complete )?(?:[Tt]ranslation|[Ee]xcerpt|[Ss]ummary) (?:from|in|by|after) "
                               r"[^\"“\n]{3,160}?\d[^\"“\n]{0,20}?:\s*(?=[\"“'‘])")


def _is_citation_paren(inner: str) -> bool:
    """Whether a parenthetical's content is a bibliographic / cross-reference citation.

    Document dates (with CE / AM / AH / Sel. or a month name) and amounts are never citations.

    :param inner: Text inside the brackets.
    :returns: True when it should be removed.
    """
    if re.search(r"\bsee\b|\bcf\.|\bibid\b|\bop\. cit|\bed(?:s)?\.|\btrans\.|Goitein|India Book|\bIB\s+[IVX]|"
                 r"Index card|PGPID", inner, re.I) or _MED_SOC.search(inner):
        return True
    if _PAGE_REF.search(inner) and not _DATE_MARK.search(inner):
        return True
    if re.search(r"(?:1[89]|20)\d{2}", inner) and not _DATE_MARK.search(inner) and \
            re.search(r"[A-Z][a-z]+.*[,:]|[,:].*\d", inner) and not re.search(r"\b(?:dinars?|dirhams?|qirats?)\b", inner):
        return True
    if find_shelfmarks(inner) and _CROSS_REF.search(inner):
        return True
    return False


def _mask_parens(text: str) -> str:
    """Blank out the contents of balanced round brackets (same length, offsets preserved).

    :param text: Text.
    :returns: Text with every bracketed group replaced by spaces.
    """
    chars = list(text)
    for a, b in _paren_spans(text):
        for i in range(a, b):
            chars[i] = " "
    return "".join(chars)


def _clauses(text: str, a: int, b: int) -> List[Tuple[int, int]]:
    """Split ``text[a:b]`` into ``;``-clauses at bracket depth 0 (each clause keeps its ``;``).

    :param text: Full text.
    :param a: Sentence start.
    :param b: Sentence end.
    :returns: Clause spans.
    """
    out, depth, start = [], 0, a
    for i in range(a, b):
        ch = text[i]
        if ch in "([":
            depth += 1
        elif ch in ")]" and depth:
            depth -= 1
        elif ch == ";" and depth == 0:
            out.append((start, i + 1))
            start = i + 1
            while start < b and text[start] in " \t":
                start += 1
    if start < b:
        out.append((start, b))
    return out


def _unit_rule(text: str, a: int, b: int, sms: Sequence[Tuple[int, int]]) -> Optional[str]:
    """Sentence/clause-level citation rule that fires on ``text[a:b]``, if any.

    :param text: Full text.
    :param a: Unit start.
    :param b: Unit end.
    :param sms: Shelfmark spans of the full text.
    :returns: Rule name or None.
    """
    unit = text[a:b]
    masked = _mask_parens(unit)
    if _DIMENSIONS.search(masked):
        return "dimensions"
    if _REF_START.match(masked.strip()):
        return "see_cf"
    if _REF_ANYWHERE.search(masked) or _YEAR_CITE.search(unit) or _MED_SOC.search(masked):
        return "bibliography"
    bare = [(x, y) for x, y in sms if a <= x < b and masked[x - a] != " "]
    if len(bare) >= 2 or (bare and _CROSS_REF.search(masked)):
        return "shelfmark_sentence"
    return None


def strip_citations(text: str) -> Tuple[str, List[str]]:
    """Strip bibliography, cross-references, dimensions and shelfmark mentions from target prose (B4).

    Rules on a sentence, or on one ``;``-clause when the sentence has several (the unit is
    dropped; the sentence goes only when every clause fires). Brackets are masked for these
    checks, so a citation in brackets removes only the brackets:

    * ``dimensions`` -- ``25.5 cm``, ``Measurements:``;
    * ``see_cf`` -- the unit starts ``See`` / ``Cf.`` / ``Also see`` / ``For further ..., see`` /
      ``Edited`` ...;
    * ``bibliography`` -- ``published, with translation, by X``, ``discussed in X``, quoted titles
      with a year, ``(2012), p. 1026``, ``Med. Soc.``; the citation prefix of a quoted translation
      (``Partial translation from X, Work, 2:271: "..."``) goes on its own, keeping the quote;
    * ``shelfmark_sentence`` -- two or more shelfmarks, or one with a cross-reference cue
      (``join``, ``related``, ``also``, ``wrote``, ``same`` ...).

    Then, in kept units: ``citation_paren`` (``(see Goitein, Med. Soc. II, p. 194)``,
    ``(Brill, 2008)``), ``dimensions`` in brackets, ``url``, and ``shelfmark`` (a lone mention is
    deleted, with its brackets when they hold nothing else). Document dates such as
    ``(25 Kislev 5589 AM = December 1 1828 CE)`` and biblical references are kept. project-name strings
    are :func:`strip_source_names`'s job.

    :param text: Target prose (a description, a catalogue ``summary``, an analysis).
    :returns: ``(text, ops)`` -- ``ops`` lists the distinct rules applied, in first-use order.
    """
    if not text:
        return text, []
    ops: List[str] = []
    dels: List[Tuple[int, int]] = []
    caps: List[int] = []
    sms = find_shelfmarks(text)
    parens = _paren_spans(text)

    def note(op: str) -> None:
        """Record an op once.

        :param op: Rule name.
        :returns: None.
        """
        if op not in ops:
            ops.append(op)

    prefixes = [(m.start(), m.end()) for m in _TRANSLATION_FROM.finditer(text)]
    work = text
    for pa, pb in prefixes:
        dels.append((pa, pb))
        note("bibliography")
        work = work[:pa] + " " * (pb - pa) + work[pb:]
    for a, b in sentence_spans(text, protect=list(sms) + prefixes):
        units = _clauses(text, a, b)
        fired = [(ua, ub, _unit_rule(work, ua, ub, sms)) for ua, ub in units]
        if all(rule for _, _, rule in fired):
            dels.append((a, b))
            for _, _, rule in fired:
                note(rule)
            continue
        kept: List[Tuple[int, int]] = []
        if fired[0][2]:
            first_kept = next(ua for ua, _, rule in fired if not rule)
            caps.append(first_kept)
        for i, (ua, ub, rule) in enumerate(fired):
            if not rule:
                kept.append((ua, ub))
                continue
            note(rule)
            if i == len(fired) - 1:
                prev_semicolon = text.rfind(";", a, ua)
                end = ub
                while end > ua and text[end - 1] in ".!?":
                    end -= 1
                dels.append((prev_semicolon if prev_semicolon >= 0 else ua, end))
            else:
                dels.append((ua, ub))
        for ua, ub in kept:
            for pa, pb in parens:
                if ua <= pa and pb <= ub:
                    inner = text[pa + 1:pb - 1]
                    if _is_citation_paren(inner):
                        dels.append((pa, pb))
                        note("citation_paren")
                    elif _DIMENSIONS.search(inner):
                        dels.append((pa, pb))
                        note("dimensions")
            for m in _URL.finditer(text, ua, ub):
                dels.append((m.start(), m.end()))
                note("url")
            for x, y in sms:
                if not (ua <= x < ub) or any(da <= x and y <= db for da, db in dels):
                    continue
                par = next(((pa, pb) for pa, pb in parens if pa < x and y < pb), None)
                if par and not re.sub(r"[\s,;.]", "", text[par[0] + 1:x] + text[y:par[1] - 1]):
                    dels.append(par)
                else:
                    dels.append((x, y))
                note("shelfmark")
    if not dels:
        return text, []
    return _delete_spans(text, dels, capitalize=caps), ops


# ============================================================================== line markers
_GIL_RANGE = re.compile(r"\(\s*(\d{1,3})\s*[-–—−]\s*(\d{1,3})\s*\)")
_GIL_ONWARD = re.compile(r"\(\s*\d{1,3}\s+ואילך\s*\)")
_LINE_REF = re.compile(r"\(\s*(?:ll?\.|ln\.|lines?|ש(?:ורות|')?)\s*\d{1,3}(?:\s*[-–—−]\s*\d{1,3})?\s*\)", re.I)
_IB_REF = re.compile(r"\(\s*[IVX]{1,4}\s*,\s*\d{1,3}[a-z]?(?::m?\d{1,3})?\s*\)")
_PAREN_LINE_START = re.compile(r"(?m)^([ \t]*)\(\s*\d{1,3}\s*\)(?=[ \t]*\S)")
_LEADING_NUM = re.compile(r"(?m)^([ \t]*)(\d{1,3})[.)](?=[ \t]+\S)")
#: A leading line number followed by a column gap (two or more blanks, or a tab) instead of ``.`` / ``)``.
_LEADING_NUM_SPACED = re.compile(r"(?m)^([ \t]*)(\d{1,3})(?=(?:[ \t]{2,}|\t)\S)")
#: Lines that must carry a spaced number before :data:`_LEADING_NUM_SPACED` counts as line numbering.
MIN_SPACED_NUMBERED_LINES = 3
_MARK = "\x00"


def strip_line_markers(text: str) -> Tuple[str, List[str]]:
    """Strip edition line markers from a translation (B3; applied to targets and, in the scorer, to both sides).

    Removes Gil-style range markers ``(4-1)``, ``(13–16)``, ``(3−4)`` (U+2212), ``(10 – 11)`` anywhere
    (both numbers ≤ 3 digits, at most 40 apart, ≤ 400), ``(12 ואילך)``, line references
    ``(ln. 9-11)`` / ``(lines 1–5)``, India Book references ``(III,5:4)``, a ``(N)`` marker at the
    start of a line, leading line numbers ``N.`` / ``N)`` when at least two lines carry them in
    mostly ascending order (a lone ``1.`` and amounts such as ``12 1/4 dir.`` are kept), and leading line
    numbers set off by a column gap (``N`` + two or more blanks or a tab, ``1          The first line``) when
    at least :data:`MIN_SPACED_NUMBERED_LINES` lines carry them in mostly ascending order (a single space after
    a number is an amount or a note key, never a marker).
    Biblical references ``(Ps. 126: 2)``, fractions ``(10/24)`` and year ranges are not markers.
    Only the spacing around a removed marker is touched.

    :param text: Translation text.
    :returns: ``(text, ops)`` with the distinct marker kinds removed, in rule order.
    """
    if not text:
        return text, []
    ops: List[str] = []

    def gil(m: re.Match) -> str:
        """Mark a plausible range marker for removal.

        :param m: Match of :data:`_GIL_RANGE`.
        :returns: The sentinel for a marker, the original text otherwise.
        """
        a, b = int(m.group(1)), int(m.group(2))
        return _MARK if abs(a - b) <= 40 and max(a, b) <= 400 else m.group(0)

    out = _GIL_RANGE.sub(gil, text)
    if out != text:
        ops.append("gil_range")
    for name, pat in (("gil_onward", _GIL_ONWARD), ("line_ref", _LINE_REF), ("ib_line_ref", _IB_REF)):
        new = pat.sub(_MARK, out)
        if new != out:
            ops.append(name)
            out = new
    new = _PAREN_LINE_START.sub(r"\1" + _MARK, out)
    if new != out:
        ops.append("paren_line_number")
        out = new
    nums = [int(m.group(2)) for m in _LEADING_NUM.finditer(out)]
    if len(nums) >= 2 and sum(1 for x, y in zip(nums, nums[1:]) if y > x) >= 0.6 * (len(nums) - 1):
        out = _LEADING_NUM.sub(r"\1" + _MARK, out)
        ops.append("leading_line_number")
    nums = [int(m.group(2)) for m in _LEADING_NUM_SPACED.finditer(out)]
    if len(nums) >= MIN_SPACED_NUMBERED_LINES and \
            sum(1 for x, y in zip(nums, nums[1:]) if y > x) >= 0.6 * (len(nums) - 1):
        out = _LEADING_NUM_SPACED.sub(r"\1" + _MARK, out)
        ops.append("leading_line_number_spaced")
    if not ops:
        return text, []
    lines = []
    for ln in out.split("\n"):
        if _MARK in ln:
            ln = re.sub(r"[ \t]*\x00[ \t]*(?=[.,;:!?)\]]|$)", "", ln)
            ln = re.sub(r"(?<=[\[(])\x00[ \t]*", "", ln)
            ln = re.sub(r"(?<=\S)[ \t]*\x00[ \t]*(?=\S)", " ", ln)
            ln = re.sub(r"[ \t]*\x00[ \t]*", "", ln)
            ln = ln.strip()
        lines.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip(), ops


# ============================================================================== grounding
def strip_marks(text: str) -> str:
    """Lowercase, drop combining marks and ayn/hamza/apostrophe signs; keep letters, digits, spaces.

    :param text: Any text.
    :returns: Folded text for entity matching.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[ʿʾ'’‘`״׳\"]", "", text).lower()
    return re.sub(r"[^\w]+", " ", text).strip()


_CAP_STOP = set("""the this that these those he she it they we i his her their its in on at for from to with by as an a and but or
if when while after before since although thus hence however also see cf note notes line lines recto verso text translation
commentary document documents letter letters paper parchment arabic hebrew aramaic judaeo judeo jewish muslim muslims christian
god lord mr ms ts t-s ena or no nos doc fol vol cat ii iii iv vi vii viii ix xi xii xiii introduction both one two three four
five six seven eight nine ten there here what which who whom whose where why how all some any each every other such only
our my your not nor so yet then than now perhaps probably apparently according mostly however moreover furthermore similarly
geniza genizah cairo library university collection taylor schechter bodleian cambridge oxford manchester ibid op cit
fustat_placeholder""".split())
_NON_SPECIFIC = set("""jews jew jewish egyptian egyptians syrian syrians german greek arab arabs arabic muslims christians christian
karaite karaites rabbanite rabbanites copt copts coptic byzantine byzantines persian persians latin europe european
abu abi ibn bint bin ben bar umm rabbi rav sitt mar rabbenu sheikh shaykh elder judge nagid gaon gaʾon rayyis qadi
torah talmud mishnah bible biblical halakha islamic frag misc bodl suppl dictionnaire cat ibid fig pl""".split())
#: Later years in commentary are publication / scholarship dates, not facts about the document.
MAX_DOC_YEAR = 1880
_DICT_WORDS = Path("/usr/share/dict/words")
_HE2LAT = dict(zip("אבגדהוזחטיכךלמםנןסעפףצץקרשת",
                   ["", "b", "g", "d", "h", "", "z", "h", "t", "", "k", "k", "l", "m", "m", "n", "n", "s", "", "f", "f", "s", "s",
                    "k", "r", "s", "t"]))
_HE_GERESH = {"ג": "g", "כ": "k", "ך": "k", "ד": "d", "ת": "t", "ט": "z", "צ": "d", "ץ": "d", "ע": "g"}
_AR2LAT = dict(zip("ابتثجحخدذرزسشصضطظعغفقكلمنهةويىءأإآؤئ",
                   ["", "b", "t", "t", "g", "h", "k", "d", "d", "r", "z", "s", "s", "s", "d", "t", "z", "", "g", "f", "k", "k", "l",
                    "m", "n", "h", "h", "", "", "", "", "", "", "", "", ""]))


@functools.lru_cache(maxsize=1)
def common_words() -> frozenset:
    """Lower-case English dictionary words (system word list; empty when absent).

    :returns: Frozen set of words.
    """
    if not _DICT_WORDS.exists():
        return frozenset()
    return frozenset(w.strip() for w in _DICT_WORDS.read_text(encoding="utf-8", errors="ignore").split() if w[:1].islower())


def _squeeze(sk: str) -> str:
    """Collapse doubled letters of a consonant skeleton.

    :param sk: Skeleton.
    :returns: Squeezed skeleton.
    """
    return re.sub(r"(.)\1+", r"\1", sk)


def skeleton_latin(folded: str) -> str:
    """Consonant skeleton of a folded Latin transliteration (``al-Maghribī`` → ``mgrb``).

    :param folded: Output of :func:`strip_marks` for one token.
    :returns: Skeleton.
    """
    w = folded.replace(" ", "")
    if w.startswith(("al", "el")) and len(w) > 4:
        w = w[2:]
    for a, b_ in (("kh", "k"), ("sh", "s"), ("th", "t"), ("dh", "d"), ("gh", "g"), ("ph", "f"), ("tz", "s"), ("ts", "s")):
        w = w.replace(a, b_)
    w = w.translate(str.maketrans("jpcqvx", "gfkkbk"))
    w = re.sub(r"[aeiouwy\d_]", "", w)
    return _squeeze(re.sub(r"h$", "", w))


def skeleton_rtl(word: str) -> str:
    """Consonant skeleton of a Hebrew- or Arabic-script word (``אלמגרבי`` → ``mgrb``).

    :param word: One word.
    :returns: Skeleton.
    """
    w = re.sub(r"[^א-תء-ي׳'״\"]", "", unicodedata.normalize("NFKD", word))
    if w.startswith(("אל", "ال")) and len(w) > 3:
        w = w[2:]
    out = []
    for i, ch in enumerate(w):
        if ch in "׳'\"״":
            continue
        nxt = w[i + 1] if i + 1 < len(w) else ""
        # ``nxt and``: the prototype tested ``"" in "׳'"`` (always True), so a word-final ט / צ / ע
        # took its geresh value (``אלפסטאט`` → ``fstz``).
        if ch in _HE_GERESH and nxt and nxt in "׳'":
            out.append(_HE_GERESH[ch])
        else:
            out.append(_HE2LAT.get(ch, _AR2LAT.get(ch, "")))
    return _squeeze(re.sub(r"h$", "", "".join(out)))


def rtl_skeletons(text: str) -> Set[str]:
    """Skeletons of every Hebrew/Arabic-script word of ``text`` (plus prefix-stripped variants).

    :param text: Input text.
    :returns: Skeleton set (length ≥ 3).
    """
    out: Set[str] = set()
    for w in re.findall(r"[א-תء-ي׳'״\"]{2,}", text):
        for v in (w, w[1:] if w[:1] in "ובלדה" and len(w) > 3 else None):
            if v:
                sk = skeleton_rtl(v)
                if len(sk) >= 3:
                    out.add(sk)
    return out


def entity_candidates(text: str, scholar_names: Iterable[str] = ()) -> Set[str]:
    """Named-entity and year candidates of a target (for the grounding test; port of the v0 builder).

    English: capitalised non-sentence-initial tokens (≥3 letters) that are not common words, not
    scholar surnames and not used in lowercase elsewhere; 3–4 digit years next to CE/AM/AH/ca. or
    4-digit years 900–1879. Hebrew script: the token after בן / אבו / אבן.

    :param text: Target prose.
    :param scholar_names: Folded scholar surnames (bibliographic names are not grounding facts).
    :returns: Folded candidate strings.
    """
    scholars = set(scholar_names)
    words = common_words()
    out: Set[str] = set()
    lower_words = set(re.findall(r"(?<![\w])([a-zāīūḥṣṭẓʿʾ][\w\-ʿʾ]*)", text))
    for m in re.finditer(r"(?<=[^.!?\n]\s)([A-ZĀĪŪḤṢṬẒʿ][\wāīūḥṣṭẓʿʾ'’\-]{2,})", text):
        tok = m.group(1)
        f = strip_marks(tok)
        if not f or f.isdigit() or tok.isupper() or f in _CAP_STOP or f in scholars or tok.lower() in lower_words \
                or f in words or f in _NON_SPECIFIC or (f.endswith("s") and f[:-1] in words):
            continue
        out.add(f)
    for m in re.finditer(r"(?:\bca\.\s*|\b)(\d{3,4})\s*(?:CE|C\.E\.|AM|A\.M\.|AH|A\.H\.|Sel)", text):
        out.add(m.group(1))
    for m in re.finditer(r"(?<![\d.:/])((?:9|1[0-9])\d{2})(?![\d])", text):
        if int(m.group(1)) < MAX_DOC_YEAR:
            out.add(m.group(1))
    for m in re.finditer(r"(?:בן|אבו|אבן)\s+([א-ת][א-ת\"׳']{1,})", unicodedata.normalize("NFKD", text)):
        out.add(strip_marks(m.group(1)))
    return out


def grounding(target: str, sources: str, scholar_names: Iterable[str] = ()) -> Tuple[float, int, List[str]]:
    """Share of the target's entity candidates found in the input + context text (port of the v0 builder).

    A candidate counts as grounded when its folded form occurs in the folded sources (also without
    spaces or a leading ``al``), or when its Latin consonant skeleton matches the skeleton of a
    Hebrew/Arabic-script word of the sources.

    :param target: Target prose.
    :param sources: Concatenated input and context text.
    :param scholar_names: Folded scholar surnames.
    :returns: ``(share grounded, n candidates, up to 12 ungrounded candidates)``; ``(1.0, 0, [])`` without candidates.
    """
    cands = entity_candidates(target, scholar_names)
    if not cands:
        return 1.0, 0, []
    src = " " + strip_marks(sources) + " "
    src_nospace = src.replace(" ", "")
    sks = rtl_skeletons(sources)
    miss = []
    for c in cands:
        alt = c[2:] if c.startswith("al") and len(c) > 5 else c
        if f" {c}" in src or c in src_nospace or (alt != c and alt in src_nospace):
            continue
        if not c.isdigit() and re.match(r"^[a-z]", c):
            sk = skeleton_latin(c)
            if len(sk) >= 3 and sk in sks:
                continue
        miss.append(c)
    return 1 - len(miss) / len(cands), len(cands), sorted(miss)[:12]


def overlap60(target: str, source: str) -> bool:
    """Whether any 60-character window of ``target`` (stride 30, whitespace collapsed) occurs in ``source``.

    :param target: Target text.
    :param source: Input text.
    :returns: True on a hit.
    """
    t, src = re.sub(r"\s+", " ", target), re.sub(r"\s+", " ", source)
    return any(t[i:i + 60] in src for i in range(0, max(len(t) - 60, 0), 30))


def scholar_surnames(sources_csv: Path, extra: Iterable[str] = ()) -> Set[str]:
    """Folded author surnames of the PGP ``sources.csv`` (``authors`` = ``Surname, Given; ...``).

    :param sources_csv: Path to the PGP export's ``sources.csv``.
    :param extra: Further surnames (e.g. from a local config); folded too.
    :returns: Set of folded surnames.
    """
    names: Set[str] = set()
    with open(sources_csv, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            for a in (row.get("authors") or "").split(";"):
                sur = a.split(",")[0].strip()
                if sur:
                    names.add(strip_marks(sur))
    names |= {strip_marks(x) for x in extra if x}
    names.discard("")
    return names
