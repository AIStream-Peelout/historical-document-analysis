# File name: lit_segment.py
# Date: 10/27/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Text segmenters for the run-2 literature mining pass (text27b, 2026-10-27).

Pure functions over page OCR text (no file or network access), used by :mod:`lit_run2` to turn newly ingested
scholarship into per-document sections. Every section records the page keys and inclusive 0-based line ranges
it was cut from, so a span can be re-cut verbatim (:func:`book_pairs.piece_from_lines`) and verified (D15).

Layouts handled (named generically; which work uses which layout lives in the local, git-ignored run-2 config):

* **doc-headed** (:func:`segment_doc_headed`) -- a typewritten catalogue / edition where every page carries a
  ``Doc. N`` (or ``Docs. N, M`` / ``Rec. Form. N``) header line and a shelfmark line, and a document runs over
  the following pages until the next header (``(Cont.)`` pages repeat or omit the header).
* **numbered anthology** (:func:`anthology_sections`) -- ``N. "Title"`` headings, an editor's introduction, a
  ``<shelfmark> (<language>)`` line and then the translation, with running heads and footnotes on every page.
* **numbered catalogue** (:func:`catalogue_entries`) -- entries opened by a line holding only the entry number.
* **edition runs** (:func:`rtl_runs`) -- runs of Hebrew / Arabic-script lines (edition text inside a prose page).

Helpers: :func:`is_stamp_line` / :func:`scrub_stamps` (download stamps that carry an IP address and the dissertation
reproduction notice), :func:`reversed_hebrew_share` (visual-order Hebrew from legacy PDF fonts),
:func:`legacy_variants` (old shelfmark spellings -> forms the corpus normaliser knows),
:func:`clean_edition_lines` (edition lines without line numbers, OCR debris and stamps),
:func:`parse_description` (document type / date / stated line count of a catalogue description),
:func:`numbered_share`, :func:`running_head_end`, :func:`footnote_line_start`.
"""
from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

# ------------------------------------------------------------------------------------ stamps
#: Lines of download stamps / reproduction notices that must never enter a training text.
_STAMP_RES: Tuple[re.Pattern, ...] = (
    re.compile(r"This content downloaded from", re.I),
    re.compile(r"All use subject to https?://about\.jstor\.org", re.I),
    re.compile(r"Reproduced with permission of the copyright owner", re.I),
    re.compile(r"Further reproduction prohibited without permission", re.I),
    re.compile(r"^\s*\d{1,3}(?:\.\d{1,3}){3}\s+on\s+\w{3},", re.I),
)
IPV4 = re.compile(r"(?<![\d.])\d{1,3}(?:\.\d{1,3}){3}(?![\d.])")


def is_stamp_line(line: str) -> bool:
    """Whether a line belongs to a download stamp or a reproduction notice.

    :param line: One line of page text.
    :returns: True for stamp lines.
    """
    return any(r.search(line) for r in _STAMP_RES)


def scrub_stamps(text: str) -> Tuple[str, int]:
    """Remove stamp lines from a text.

    :param text: Text (possibly several lines).
    :returns: ``(text without stamp lines, number of lines removed)``.
    """
    kept, n = [], 0
    for ln in text.split("\n"):
        if is_stamp_line(ln):
            n += 1
            continue
        kept.append(ln)
    return "\n".join(kept), n


def has_ip_address(text: str) -> bool:
    """Whether a text contains something shaped like an IPv4 address (a stamp leaked through).

    :param text: Text.
    :returns: True when an IPv4-shaped token occurs.
    """
    for m in IPV4.finditer(text or ""):
        parts = [int(x) for x in m.group(0).split(".")]
        if all(0 <= x <= 255 for x in parts) and parts[0] > 0:
            return True
    return False


# ------------------------------------------------------------------------------------ script checks
_HE = re.compile(r"[א-ת]")
_AR = re.compile(r"[؀-ۿ]")
_LAT = re.compile(r"[A-Za-zÀ-ɏḀ-ỿ]")
_HE_WORD = re.compile(r"[א-ת]{2,}")
_FINALS = "ךםןףץ"


def letters(text: str) -> Tuple[int, int, int]:
    """Hebrew, Arabic and Latin letter counts.

    :param text: Text.
    :returns: ``(hebrew, arabic, latin)``.
    """
    return len(_HE.findall(text)), len(_AR.findall(text)), len(_LAT.findall(text))


def reversed_hebrew_share(text: str, min_words: int = 10) -> Optional[float]:
    """Share of final-form letters that stand at a word START (visual-order / reversed Hebrew).

    In logical-order Hebrew the final forms ך ם ן ף ץ end words; a text layer that stores lines right-to-left in
    visual order puts them first. 0.0 is clean, values near 1.0 mean reversed lines.

    :param text: Hebrew-script text.
    :param min_words: Minimum number of words with a final-form letter at either end.
    :returns: The share, or None when too few words carry a final form.
    """
    start = end = 0
    for w in _HE_WORD.findall(text or ""):
        if w[0] in _FINALS:
            start += 1
        if w[-1] in _FINALS:
            end += 1
    if start + end < min_words:
        return None
    return round(start / (start + end), 3)


#: Fixed-order Hebrew word pairs of scholarly prose; a text layer stored in visual order shows them reversed.
ORDERED_PAIRS: Tuple[Tuple[str, str], ...] = (("על", "ידי"), ("בית", "דין"), ("בית", "הכנסת"), ("אף", "על"), ("על", "פי"),
                                             ("ראו", "גם"), ("לעיל", "הערה"), ("כמו", "כן"), ("בין", "היתר"),
                                             ("יחד", "עם"), ("ארץ", "ישראל"), ("ראש", "הישיבה"), ("בית", "המדרש"),
                                             ("לא", "היה"), ("את", "זה"), ("של", "הגניזה"), ("כתב", "יד"))


def word_order_reversal(text: str, min_hits: int = 5) -> Optional[float]:
    """Share of fixed-order word pairs found reversed (word order stored right-to-left, letters intact).

    :param text: Hebrew prose.
    :param min_hits: Minimum pairs found (either order).
    :returns: Reversed share, or None when too few pairs occur.
    """
    words = re.findall(r"[א-ת'\"]+", text or "")
    big = collections.Counter(zip(words, words[1:]))
    fwd = sum(big[(a, b)] for a, b in ORDERED_PAIRS)
    rev = sum(big[(b, a)] for a, b in ORDERED_PAIRS)
    if fwd + rev < min_hits:
        return None
    return round(rev / (fwd + rev), 3)


# ------------------------------------------------------------------------------------ shelfmarks
_LANG_PAREN = re.compile(r"\s*\((?:[^()]*\b(?:Arabic|Hebrew|Aramaic|Persian|Greek|Ladino|Cat\.|\*)[^()]*|\*)\)\s*")


_SIDE_WORDS = re.compile(r"\s*\((?:and\s+)?(?:recto|verso|not listed)[^()]*\)|\s+(?:recto|verso)\b\.?", re.I)


_NOS_LIST = re.compile(r"^(?P<base>.*?\d[a-z]?)\s*,?\s*nos\.?\s*(?P<list>\d+[a-z]?(?:\s*,\s*\d+[a-z]?)+)[\s,]*"
                       r"(?P<tail>\(\*\))?\s*$")
_LETTER_LIST = re.compile(r"^(?P<base>.*\d)\s*(?P<list>[a-z](?:\s*,\s*[a-z])+)[\s,]*(?P<tail>\(\*\))?\s*$")


def split_piece_lists(part: str) -> List[str]:
    """Split one shelfmark that lists several numbered or lettered pieces into one shelfmark per piece.

    ``TS NS J 185, nos. 5, 6, 2`` → ``TS NS J 185, no. 5`` / ``no. 6`` / ``no. 2``; ``TS NS J 67 a, b (*)`` →
    ``TS NS J 67a`` / ``TS NS J 67b``. A list of side letters only (``r, v``) is left alone.

    :param part: One shelfmark (one part of a join).
    :returns: The pieces (the part alone when it lists no pieces).
    """
    m = _NOS_LIST.match(part)
    if m:
        return [f"{m.group('base').rstrip(' ,')}, no. {n}" for n in re.findall(r"\d+[a-z]?", m.group("list"))]
    m = _LETTER_LIST.match(part)
    if m:
        lets = re.findall(r"[a-z]", m.group("list"))
        if not set(lets) <= {"r", "v"}:
            return [f"{m.group('base')}{x}" for x in lets]
    return [part]


def split_joins(raw: str) -> List[str]:
    """Split a printed join (``A and B``, ``A + B``, ``A; B``) into its shelfmarks; a bare second number inherits
    the first mark's collection prefix (``TS NS J 110 and 152`` → ``TS NS J 110``, ``TS NS J 152``; ``..., f. 141 and
    f. 10`` → ``..., f. 141``, ``..., f. 10``). A part that lists several pieces (``nos. 5, 6, 2``, ``67 a, b``) is
    split into one shelfmark per piece (:func:`split_piece_lists`).

    :param raw: Printed shelfmark.
    :returns: Parts (the raw string alone when it is not a join and lists no pieces).
    """
    s = re.sub(r"\s+", " ", (raw or "").strip())
    bits = [b.strip(" ,;") for b in re.split(r"\s+(?:and|\+|&)\s+|\s*;\s*", s) if b.strip(" ,;")]
    if len(bits) < 2:
        return split_piece_lists(s) if s else []
    out = [bits[0]]
    for b in bits[1:]:
        if re.match(r"^(?:ff?\.|fols?\.|nos?\.)\s*\d", b):
            base = re.sub(r",?\s*(?:ff?\.|fols?\.|nos?\.)\s*[\d\w ,]+$", "", bits[0]).strip(" ,")
            out.append(f"{base}, {b}")
        elif re.fullmatch(r"\d+[a-z]?", b):
            base = re.sub(r"\d+[a-z]?\s*$", "", bits[0]).rstrip()
            out.append(f"{base} {b}" if base else b)
        else:
            out.append(b)
    return [p for b in out for p in split_piece_lists(b)]


#: A numbered piece of a box / volume (``no. 7``, ``no 7``) at the end of a printed shelfmark.
PIECE_NO = re.compile(r"[\s,]*\bno\.?\s*(?P<no>\d+[a-z]?)\s*(?:\(\*\))?\s*$")


def piece_of(raw: str) -> Optional[Tuple[str, str]]:
    """The numbered piece a printed shelfmark ends with, and the shelfmark without it.

    :param raw: Printed shelfmark (one piece; see :func:`split_joins`).
    :returns: ``(base shelfmark, "no. N")`` or None when the mark names no numbered piece.
    """
    m = PIECE_NO.search(raw or "")
    if not m or not re.search(r"\d", raw[:m.start()]):
        return None
    return raw[:m.start()].rstrip(" ,"), f"no. {m.group('no')}"


def legacy_variants(raw: str, neubauer: Optional[Mapping[str, str]] = None) -> List[str]:
    """Rewrite an old-style shelfmark into the spellings the corpus normaliser matches.

    Covers the forms of 1920s–1970s scholarship: ``TS`` for ``T-S``, ``Box`` volumes, ``f.`` / ``fol.`` / ``no.`` /
    ``no`` folio markers (the first of a list of folios / numbers / part letters), ``TS 8 J 34`` for ``T-S 8J34``,
    ``TS Arabic (Box)`` for ``T-S Ar.``, a part letter in parentheses or after a space (``J 318(a)``, ``J 67 a``),
    a volume in parentheses (``Ar. Box 18 (1).33`` → ``Ar.18(1).33``), ``ULC Or Box 5, f. 13`` for
    ``CUL Or.1080 5.13``, ``Westminster College, Misc. 34`` / ``West. Coll., Frag. Cairens., f. 34`` for
    ``L-G Misc. 34`` (and ``Arabica I, f. N`` for ``L-G Ar. I.N``), ``Bodl. MS Heb. c 28, fol. 33`` /
    ``Bodleian MS Heb. d 66.5`` for ``Bodl. MS Heb. c 28/33`` / ``d 66/5``, Neubauer catalogue numbers
    (``Oxford (Neubauer) 2878/4``) when a crosswalk to MS Heb. volumes is supplied, side words (``Verso``,
    ``(and Verso)``) and a language / ``(Cat. N)`` / ``(*)`` parenthesis after the mark. Last-resort variants (marked
    by their position in the list): a footnote number glued to the folio (``37.1141`` → ``37.11``; ``15.60v74`` →
    ``15.60``), the folio without a part letter, a Dropsie / Penn ``E 16.510`` as ``UPenn E 16510`` and
    ``JTS Krengel 5.123`` as ``JTS Krengel 123``.

    :param raw: Shelfmark as printed (one part of a join; see :func:`split_joins`).
    :param neubauer: Optional Neubauer catalogue number -> ``MS Heb. <letter> <number>`` crosswalk.
    :returns: Distinct candidate spellings, the raw (stripped) form first.
    """
    s0 = re.sub(r"\s+", " ", (raw or "").strip())
    out: List[str] = [s0] if s0 else []
    s = _LANG_PAREN.sub(" ", s0)
    s = _SIDE_WORDS.sub(" ", s).strip(" ,;")
    s = re.sub(r"(?<=\d)\s*\(([a-z])\)", r"\1", s)
    s = re.sub(r"(?<=\d)\s*\((\d)\)(?=\s*[.,])", r"(\1)", s)
    s, n_or = re.subn(r"^(?:ULC|CUL|T\.?-?\s?S\.?)\s+Or\.?\s*(?:1080\s*)?Box\s+(\d+)", r"CUL Or.1080 \1", s)
    s = re.sub(r"^(?:Westminster\s+College|West\.?\s*Coll\.?)\s*(?:\(Westminster College\))?\s*,?\s*"
               r"(?:Misc\.?|Frag\.?\s*Cairens\.?)\s*,?\s*(?:(?:ff?\.|fols?\.)\s*)?(\d+)", r"L-G Misc. \1", s)
    s = re.sub(r"^(?:Westminster\s+College|West\.?\s*Coll\.?)\s*(?:\(Westminster College\))?\s*,?\s*"
               r"(?:Ar\.?|Arabica)\s*(?:\(Arabica\)\s*)?(I{1,2})\s*,?\s*(?:ff?\.|fols?\.)\s*(\d+)", r"L-G Ar. \1.\2", s)
    if not n_or:
        s = re.sub(r"^(?:ULC|CUL)\s+(?=Or)", "T-S ", s)
    s = re.sub(r"^T\.?\s?S\.?\b\.?", "T-S", s)
    s = re.sub(r"^T-S\s+Arabic\s*(?:Box\s*)?", "T-S Ar.", s)
    s = re.sub(r"^T-S\s+(NS\s+J|NS|Misc\.?|AS|K|Box\s+K)\s*Box\s+", lambda m: f"T-S {m.group(1)} ", s)
    s = re.sub(r"^T-S\s+Box\s+", "T-S ", s)
    s = re.sub(r"^T-S\s+Misc\w*\.?\s+(?:Box\s+)?", "T-S Misc.", s)
    s = re.sub(r"^T-S\s+(\d+)\s*J\s*(\d+)", r"T-S \1J\2", s)
    s = re.sub(r"^T-S\s+NS\s+J\s*(\d+)", r"T-S NS J\1", s)
    s = re.sub(r"^T-S\s+K\s+(\d+)", r"T-S K\1", s)
    s = re.sub(r"^Bodleian\b\.?", "Bodl.", s)
    s = re.sub(r"(MS\.?\s+Heb\.?\s+[a-f])\.?\s*(\d+)\s*[.,]\s*(?:ff?\.|fols?\.)?\s*(\d+[a-z]?)", r"\1 \2/\3", s)
    m = re.match(r"^(?:Oxford|Bodl\.?)\s*(?:\(Neubauer\)|Neubauer)?\s*(?:Cat\.?\s*)?(\d{4})\s*/\s*(\d+[a-z]?)", s)
    if m and neubauer and m.group(1) in neubauer:
        s = f"Bodl. {neubauer[m.group(1)]}/{m.group(2)}"
    s = re.sub(r",?\s*(?:ff?\.|fols?\.)\s*(\d+)\s*([a-d])(?:\s*,\s*[a-d])*\b", r".\1\2", s)
    s = re.sub(r",?\s*(?:ff?\.|fols?\.)\s*(\d+[a-zA-Z]?)(?:\s*[-–]\s*\d+)?", r".\1", s)
    s = re.sub(r",?\s*\bnos?(?:\.\s*|\s+)(\d+)(?:\s*,\s*\d+)*,?", r".\1", s)
    s = re.sub(r"^(ENA\s+\d+[A-Za-z]?)\s*\.\s*(\d+)([A-Z])\b", lambda mm: f"{mm.group(1)}.{mm.group(2)}{mm.group(3).lower()}", s)
    s = re.sub(r"(?<=\d)\s+([a-z])(?:\s*,\s*[a-z])*\s*$", r"\1", s)
    s = re.sub(r"\s+\.", ".", s).strip(" ,;.")
    if s and s not in out:
        out.append(s)
    # last resort: glued footnote numbers, side letter + note, part letter, Penn E numbers, Krengel box numbers
    g = re.match(r"^(.*\.\d{1,3})(\d{2})$", s)
    if g and len(re.sub(r"\D", "", s.split(".")[-1])) >= 4:
        out.append(g.group(1))
    g = re.match(r"^(.*\.\d{1,3})[rv]\d{0,3}$", s)
    if g:
        out.append(g.group(1))
    g = re.match(r"^(.*\.\d{1,3})[a-d]$", s)
    if g and g.group(1) not in out:
        out.append(g.group(1))
    g = re.match(r"^E\s?(\d{1,2})\.(\d{3})$", s)
    if g:
        out.append(f"UPenn E {g.group(1)}{g.group(2)}")
    g = re.match(r"^(JTS\s+Krengel)\s+\d{1,2}\.(\d{1,3})$", s)
    if g:
        out.append(f"{g.group(1)} {g.group(2)}")
    return out


# ------------------------------------------------------------------------------------ page layout
_RUNHEAD = re.compile(r"^(?:[A-Z][A-Z0-9 ,.'’&:()–-]{3,}\s+\d{1,3}|\d{1,3}\s+[A-Z][A-Z0-9 ,.'’&:()–-]{3,}|"
                      r"[A-Z][A-Z ,.'’&:()–-]{3,}|"
                      r"\d{1,4}|[ivxlc]{1,6}|CHAPTER\s+\w+|Chapter\s+\w+|\(Cont\.?\)|\(\s*Cont\.?\s*\))$")


def running_head_end(lines: Sequence[str], max_lines: int = 3) -> int:
    """Index of the first body line after a page's running head / page number lines.

    :param lines: Page lines.
    :param max_lines: Maximum number of leading head lines.
    :returns: Index of the first body line.
    """
    i, numeric = 0, 0
    while i < min(max_lines, len(lines)) and (not lines[i].strip() or _RUNHEAD.match(lines[i].strip())):
        if re.fullmatch(r"\d{1,4}", lines[i].strip()):
            if numeric:
                break
            numeric += 1
        i += 1
    return i


_FOOT_START = re.compile(r"^\s*(?:\d{1,3}|[¹²³⁴-⁹]+)\s*[A-Z“\"(\[‘'א-ת]|^\s*\d{1,3}\s*$|^\s*\d{1,3}\s*[Ii]bid")


def footnote_line_start(lines: Sequence[str], hint: Optional[int] = None, min_frac: float = 0.35) -> int:
    """Index of the first footnote line of a page (``len(lines)`` when none is found).

    Uses the caller's hint (e.g. :func:`pages.footnote_region_start` converted to a line index) when given;
    otherwise the earliest line in the lower part of the page that opens like a numbered note and is followed only
    by note-like or continuation lines.

    :param lines: Page lines.
    :param hint: Footnote start line found from the page JSON, or None.
    :param min_frac: Lines above this share of the page are never footnotes.
    :returns: Line index.
    """
    n = len(lines)
    body = [i for i, ln in enumerate(lines) if ln.strip() and not is_stamp_line(ln)]
    if hint is not None and 0 < hint <= n:
        return hint
    lo = int(n * min_frac)
    cand = n
    for i in range(n - 1, lo - 1, -1):
        if (i in body and _FOOT_START.match(lines[i]) and not re.match(r"^\s*\d{1,3}\.\s", lines[i])
                and not re.fullmatch(r"\d{1,3}\s+[A-Z][A-Z0-9 ,.'’&:()–-]{3,}", lines[i].strip())):
            cand = i
    return cand


def numbered_share(lines: Sequence[str]) -> float:
    """Share of lines that carry a leading or trailing line number (1–3 digits).

    :param lines: Lines.
    :returns: Share in [0, 1] (0.0 for no lines).
    """
    ls = [ln.strip() for ln in lines if ln.strip()]
    if not ls:
        return 0.0
    k = sum(1 for ln in ls if re.match(r"^\(?\d{1,3}\)?(?:[\s.]|$)", ln) or re.search(r"\s\(?\d{1,3}\)?$", ln))
    return round(k / len(ls), 3)


def _letters_only(s: str) -> str:
    """Lower-cased letters of a text (digits, spaces and punctuation removed).

    :param s: Text.
    :returns: Letters only.
    """
    return "".join(ch for ch in s.lower() if ch.isalpha())


def footnote_line_set(lines: Sequence[str], notes: Sequence[str], min_letters: int = 12) -> Set[int]:
    """Lines of a page whose letters occur inside the page's footnote texts (from the page JSON).

    Catches footnote lines anywhere on the page, including note text continued from the previous page that has no
    note number.

    :param lines: Page lines.
    :param notes: Footnote strings of the page.
    :param min_letters: Lines with fewer letters are not tested.
    :returns: Indices of footnote lines.
    """
    hay = "\n".join(_letters_only(n) for n in notes if n)
    if not hay:
        return set()
    out: Set[int] = set()
    for i, ln in enumerate(lines):
        k = _letters_only(ln)
        if len(k) >= min_letters and k in hay:
            out.add(i)
    return out


# ------------------------------------------------------------------------------------ edition lines
_LINE_NO_LEAD = re.compile(r"^\s*\(?\d{1,3}\)?[.)]?\s+(?=\S)")
_LINE_NO_TRAIL = re.compile(r"(?<=\S)\s+\(?\d{1,3}\)?\s*$")
_DEBRIS = re.compile(r"[✓☑✔•■□▪◆○●]")


def clean_edition_lines(lines: Sequence[str], min_letters: int = 2) -> Tuple[List[str], Dict[str, int]]:
    """Edition lines without stamps, line numbers, OCR debris and letter-less lines.

    :param lines: Raw lines of an edition section.
    :param min_letters: Hebrew / Arabic letters a kept line needs.
    :returns: ``(lines, stats)`` with ``raw``, ``kept``, ``stamp``, ``numeric``, ``debris``, ``short`` (kept lines
        of one or two words) and ``latin`` (dropped Latin-only lines).
    """
    st = {"raw": 0, "kept": 0, "stamp": 0, "numeric": 0, "debris": 0, "short": 0, "latin": 0}
    out: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        st["raw"] += 1
        if is_stamp_line(s):
            st["stamp"] += 1
            continue
        if re.fullmatch(r"[\d\s.,()\[\]{}:;|/-]+", s):
            st["numeric"] += 1
            continue
        s = _DEBRIS.sub("", s)
        s = _LINE_NO_LEAD.sub("", s)
        s = _LINE_NO_TRAIL.sub("", s).strip()
        he, ar, lat = letters(s)
        if he + ar < min_letters:
            st["latin" if lat >= 3 else "debris"] += 1
            continue
        if lat > he + ar:
            st["latin"] += 1
            continue
        if len(s.split()) <= 2:
            st["short"] += 1
        out.append(s)
        st["kept"] += 1
    return out, st


# ------------------------------------------------------------------------------------ data types
@dataclass
class PageInput:
    """One page handed to a segmenter.

    :ivar key: Page key (``<work>/<part>/<page file>``).
    :ivar page_no: PDF page number.
    :ivar text: OCR text.
    :ivar foot_line: Footnote start line from the page JSON (None when unknown).
    :ivar note_lines: Lines that hold footnote text anywhere on the page (:func:`footnote_line_set`).
    """

    key: str
    page_no: int
    text: str
    foot_line: Optional[int] = None
    note_lines: FrozenSet[int] = frozenset()


@dataclass
class Section:
    """A document section cut out of consecutive pages.

    :ivar label: Document label (``"16"``, ``"97,98"``, ``"RF258"``, anthology number).
    :ivar kind: ``doc``, ``docs``, ``rec_form`` or ``anthology``.
    :ivar shelfmark_raw: Shelfmark line as printed (None when absent).
    :ivar ranges: ``[(page key, first line, last line)]`` (inclusive, 0-based) of the body.
    :ivar text: The body text (ranges joined by newlines).
    :ivar extra: Other ranges by role (``intro``, ``translation`` ...) and notes.
    """

    label: str
    kind: str
    shelfmark_raw: Optional[str]
    ranges: List[Tuple[str, int, int]] = field(default_factory=list)
    text: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def pages(self) -> List[str]:
        """Page keys in order (no duplicates).

        :returns: Keys.
        """
        return list(dict.fromkeys(k for k, _, _ in self.ranges))


def _join_ranges(pages: Mapping[str, Sequence[str]], ranges: Sequence[Tuple[str, int, int]]) -> str:
    """Text of line ranges.

    :param pages: page key -> lines.
    :param ranges: ``[(key, a, b)]``.
    :returns: Joined text.
    """
    return "\n".join("\n".join(pages[k][a:b + 1]) for k, a, b in ranges)


def _trim_range(lines: Sequence[str], a: int, b: int) -> Optional[Tuple[int, int]]:
    """Shrink a line range to its first / last non-blank, non-stamp line.

    :param lines: Page lines.
    :param a: First line.
    :param b: Last line (inclusive).
    :returns: The trimmed range or None when nothing is left.
    """
    while a <= b and (not lines[a].strip() or is_stamp_line(lines[a])):
        a += 1
    while b >= a and (not lines[b].strip() or is_stamp_line(lines[b])):
        b -= 1
    return (a, b) if a <= b else None


def _split_on_stamps(lines: Sequence[str], a: int, b: int) -> List[Tuple[int, int]]:
    """Split a range around stamp lines (a piece never contains a stamp).

    :param lines: Page lines.
    :param a: First line.
    :param b: Last line.
    :returns: Sub-ranges without stamp lines.
    """
    out: List[Tuple[int, int]] = []
    start = a
    for i in range(a, b + 1):
        if is_stamp_line(lines[i]):
            if start <= i - 1:
                t = _trim_range(lines, start, i - 1)
                if t:
                    out.append(t)
            start = i + 1
    if start <= b:
        t = _trim_range(lines, start, b)
        if t:
            out.append(t)
    return out


# ------------------------------------------------------------------------------------ doc-headed layout
DOC_LINE = re.compile(r"^\(?\s*(?P<kind>Docs?|Rec\.\s*Form)\.?\s*(?P<nums>\d+[a-z]?(?:\s*(?:,|and|&)\s*\d+[a-z]?)*)\s*\)?\s*$")
_SM_LINE = re.compile(r"^(?:T\.?\s?-?\s?S\b|ENA\b|Bodl|Oxford|PER\b|E\s?\d|Or\.?\s|ULC|CUL|Mosseri|JTS|Antonin|AIU|"
                      r"Westminster|Halper|BM\b|BL\b|Dropsie|Gottheil|Firk|Cambridge)")


def doc_header(lines: Sequence[str], scan: int = 12) -> Optional[Tuple[str, str, int]]:
    """Find a page's ``Doc. N`` header line.

    :param lines: Page lines.
    :param scan: Number of leading lines to search.
    :returns: ``(label, kind, line index)`` or None.
    """
    for i, ln in enumerate(lines[:scan]):
        m = DOC_LINE.match(ln.strip())
        if m:
            nums = re.findall(r"\d+[a-z]?", m.group("nums"))
            kind = "rec_form" if m.group("kind").startswith("Rec") else ("docs" if len(nums) > 1 else "doc")
            label = ("RF" if kind == "rec_form" else "") + ",".join(nums)
            return label, kind, i
    return None


def shelfmark_header(lines: Sequence[str], scan: int = 8) -> Optional[Tuple[str, int]]:
    """Find a page's shelfmark header line (a short line opening with a collection prefix).

    :param lines: Page lines.
    :param scan: Number of leading lines to search.
    :returns: ``(line, index)`` or None.
    """
    for i, ln in enumerate(lines[:scan]):
        s = ln.strip()
        if 4 <= len(s) <= 80 and _SM_LINE.match(s) and re.search(r"\d", s):
            return s, i
    return None


def _ranges_without(lines: Sequence[str], a: int, b: int, skip: Sequence[int]) -> List[Tuple[int, int]]:
    """Sub-ranges of ``[a, b]`` that avoid the ``skip`` lines and stamp lines.

    :param lines: Page lines.
    :param a: First line.
    :param b: Last line (inclusive).
    :param skip: Line indices to leave out.
    :returns: Trimmed sub-ranges.
    """
    out: List[Tuple[int, int]] = []
    start = a
    for i in sorted(x for x in set(skip) if a <= x <= b) + [b + 1]:
        if start <= i - 1:
            out.extend(_split_on_stamps(lines, start, i - 1))
        start = i + 1
    return out


def segment_doc_headed(pages: Sequence[PageInput], header_scan: int = 12, whole_page: bool = False,
                       footnotes: bool = True) -> List[Section]:
    """Cut a doc-headed run of pages into document sections.

    A page whose header names a document opens (or continues) that document; a page without a header continues
    the current one. By default the body of a page runs from the line after its last header line (printed page
    number, shelfmark, ``Doc.`` and ``(Cont.)`` lines among the first ``header_scan`` lines) to the line before its
    footnotes. With ``whole_page`` (OCR that scatters the header lines into the text, as on typewritten edition
    pages read by a layout-blind OCR) the body is every line of the page except the header lines themselves.
    Stamp lines are never part of a body.

    :param pages: Pages in reading order.
    :param header_scan: Leading lines searched for header lines.
    :param whole_page: Keep the lines before the header lines (see above).
    :param footnotes: Cut each page at its footnotes (:func:`footnote_line_start`); False for edition pages whose
        bare line-number lines would look like notes.
    :returns: Sections in order of first appearance (a label seen again later is appended to its section).
    """
    by_label: Dict[str, Section] = {}
    order: List[str] = []
    page_lines: Dict[str, List[str]] = {}
    cur: Optional[str] = None
    for pg in pages:
        lines = pg.text.split("\n")
        page_lines[pg.key] = lines
        hd = doc_header(lines, header_scan)
        smh = shelfmark_header(lines, header_scan)
        head_idx = [i for i, ln in enumerate(lines[:header_scan])
                    if ln.strip() and (DOC_LINE.match(ln.strip()) or re.fullmatch(r"\(\s*Cont\.?\s*\)", ln.strip())
                                       or (not whole_page and _RUNHEAD.match(ln.strip())))]
        if hd:
            head_idx.append(hd[2])
        if smh:
            head_idx.append(smh[1])
        end = (footnote_line_start(lines, pg.foot_line) if footnotes else len(lines)) - 1
        if hd:
            cur = hd[0]
            if cur not in by_label:
                by_label[cur] = Section(label=cur, kind=hd[1], shelfmark_raw=smh[0] if smh else None)
                order.append(cur)
            elif by_label[cur].shelfmark_raw is None and smh:
                by_label[cur].shelfmark_raw = smh[0]
        if cur is None:
            continue
        if whole_page:
            pieces = _ranges_without(lines, 0, end, list(head_idx) + (list(pg.note_lines) if footnotes else []))
        else:
            start = (max(head_idx) + 1) if head_idx else 0
            pieces = _ranges_without(lines, start, end, pg.note_lines if footnotes else ()) if start <= end else []
        for a, b in pieces:
            by_label[cur].ranges.append((pg.key, a, b))
    for lab in order:
        sec = by_label[lab]
        sec.text = _join_ranges(page_lines, sec.ranges)
    return [by_label[lab] for lab in order]


# ------------------------------------------------------------------------------------ description fields
def parse_description(text: str) -> Dict[str, Any]:
    """Document type, date phrase, stated line count and state of a catalogue description.

    :param text: Description text (header lines removed).
    :returns: ``{"document_type", "date_text", "lines_stated", "state"}`` (None where not found).
    """
    body = "\n".join(ln for ln in (text or "").split("\n") if not re.fullmatch(r"\s*[\divxlc]{1,4}\s*", ln))
    flat = re.sub(r"-\n(?=[a-z])", "", body)
    flat = re.sub(r"\s+", " ", flat).strip()
    sents = [s.strip() for s in re.split(r"(?<=\.)\s+", flat) if s.strip()]
    doc_type = None
    for s in sents[:3]:
        words = s.rstrip(".").split()
        if (1 <= len(words) <= 8 and s.endswith(".") and not re.search(r"\d", s) and s[0].isupper()
                and not re.match(r"(?i)(fragment|finished|draft|copy|complete|incomplete|date|undated)\b", s)):
            doc_type = s.rstrip(".")
            break
    date = None
    m = re.search(r"(?:The )?[Dd]ate(?: preserved)?(?: is| was)?:?\s+([^.]{3,80})\.", flat)
    if m and not re.match(r"(?i)not preserved", m.group(1)):
        date = m.group(1).strip()
    elif re.search(r"(?i)date (?:is )?not preserved|undated", flat):
        date = "not preserved"
    else:
        m = re.search(r"\bDated?\s+([^.]{3,80})\.", flat)
        date = m.group(1).strip() if m else None
    m = re.search(r"\b(\d{1,3})\s+lines\b", flat)
    lines_stated = int(m.group(1)) if m else None
    state = None
    for s in ("Fragment", "Finished", "Draft", "Copy", "Complete", "Incomplete"):
        if re.search(rf"\b{s}\.", flat):
            state = (state + "; " if state else "") + s.lower()
    return {"document_type": doc_type, "date_text": date, "lines_stated": lines_stated, "state": state}


# ------------------------------------------------------------------------------------ numbered anthology
ANTH_HEAD = re.compile(r"^(?P<no>\d{1,3})\.\s+(?P<title>[\"“”'‘A-Z].{3,})$")
LANG_LINE = re.compile(r"^(?P<sm>[A-Z].{2,90}?)\s*\((?P<lang>(?:Judaeo-|Judeo-)?(?:Arabic|Hebrew|Aramaic|Persian)"
                       r"[^()]{0,40})\)\s*[\d¹²³⁴-⁹]{0,3}\s*$")
LIST_START = re.compile(r"^(?:Recto|Verso|Col(?:umn)?\.?\s*[IVX1-9]|Side\s+[A-Z1-9]|Page\s+\d|1\.\s|\[?Right|\[?Left)")


def _clean_sm(sm: str) -> str:
    """Shelfmark of an anthology line without a ``(formerly ...)`` / old-number parenthesis or glued note digits.

    :param sm: Text before the language parenthesis.
    :returns: Cleaned shelfmark.
    """
    sm = re.sub(r"(?<=\d)\s*\((\d)\)(?=\.\d)", r"(\1)", sm)
    sm = re.sub(r"\s*\((?!\d\)\.)(?:formerly|olim|old|=)?[^()]*\)\s*[\d¹²³⁴-⁹]*", "", sm).strip()
    sm = re.sub(r"[¹²³⁴-⁹]+$", "", sm)
    sm = re.sub(r"\s+\d{1,3}$", "", sm) if re.search(r"\d[\d.]*\s+\d{1,3}$", sm) else sm
    return sm.strip(" ·,;")


def alt_classmark(sm: str) -> Optional[str]:
    """A second classmark printed in parentheses after a shelfmark (``Mosseri L 129.1 (II 127.1)``,
    ``Mosseri L 291 (Ia, 21)``, ``X (formerly Y)``), completed with the mark's collection word when it starts with a
    volume number (``Mosseri II 127.1``).

    :param sm: Shelfmark text as printed (before :func:`_clean_sm`).
    :returns: The alternate classmark, or None.
    """
    for m in re.finditer(r"\(([^()]{1,40})\)", sm or ""):
        inner = m.group(1).strip()
        if re.fullmatch(r"\d|[a-z]|\*", inner) or not re.search(r"\d", inner):
            continue
        if re.search(r"Arabic|Hebrew|Aramaic|Persian|Greek|Ladino|Cat\.", inner):
            continue
        inner = re.sub(r"^(?:formerly|olim|old|now|=)\s*:?\s*", "", inner).strip(" ,;")
        if re.match(r"^(?:[IVX]+[a-z]?|\d)", inner):
            first = (sm.split() or [""])[0]
            inner = f"{first} {inner}" if first and not first[0].isdigit() else inner
        return inner
    return None


def _is_lang_line(line: str) -> Optional[Tuple[str, str, Optional[str]]]:
    """``<shelfmark> (<language>)`` line of an anthology entry.

    :param line: Line.
    :returns: ``(shelfmark, language, alternate classmark or None)`` or None.
    """
    m = LANG_LINE.match(line.strip())
    if not m or not _SM_LINE.match(m.group("sm")) or not re.search(r"\d", m.group("sm")):
        return None
    return _clean_sm(m.group("sm")), m.group("lang").strip(), alt_classmark(m.group("sm"))


def anthology_sections(pages: Sequence[PageInput], stop: Optional[re.Pattern] = None, max_skip: int = 2) -> List[Section]:
    """Cut a numbered anthology into entries: introduction ranges, shelfmark line(s) and translation ranges.

    The page body excludes the running head (:func:`running_head_end`) and the footnotes
    (:func:`footnote_line_start`). Within an entry, lines up to the first ``<shelfmark> (<language>)`` line are the
    introduction; lines after it are the translation; a further shelfmark line starts a further translated
    document of the same entry (``extra["docs"]`` holds one record per shelfmark line, each with its own
    translation ranges). Text after the last entry on a page that matches ``stop`` ends the run.

    :param pages: Pages in reading order.
    :param stop: Pattern of a line that ends the anthology part (e.g. a bibliography heading).
    :param max_skip: A heading must continue the entry sequence (number = last + 1 … last + ``max_skip``), which
        keeps numbered list items inside a translation from opening entries.
    :returns: Sections (``kind="anthology"``; ``ranges`` = all translation ranges).
    """
    out: List[Section] = []
    page_lines: Dict[str, List[str]] = {}
    notes_of: Dict[str, FrozenSet[int]] = {}
    sec: Optional[Section] = None
    role = "pre"
    stopped = False
    last_no = 0
    title_open, title_lines = False, 0

    def add(rk: str, key: str, a: int, b: int) -> None:
        """Append a range to the current section's role.

        :param rk: ``intro`` or ``translation``.
        :param key: Page key.
        :param a: First line.
        :param b: Last line.
        :returns: None.
        """
        if sec is None or a > b:
            return
        for x, y in _ranges_without(page_lines[key], a, b, notes_of.get(key, ())):
            if rk == "intro":
                sec.extra.setdefault("intro", []).append((key, x, y))
            else:
                sec.extra["docs"][-1]["translation"].append((key, x, y))

    for pg in pages:
        if stopped:
            break
        lines = pg.text.split("\n")
        page_lines[pg.key] = lines
        # running heads the OCR placed below the first text line (e.g. after a shelfmark line) are skipped too
        stray_heads = {i for i in range(min(4, len(lines))) if lines[i].strip() and not ANTH_HEAD.match(lines[i].strip())
                       and re.fullmatch(r"\d{1,3}\s+[A-Z][A-Z0-9 ,.'’&:()–-]{3,}|[A-Z][A-Z0-9 ,.'’&:()–-]{3,}\s+\d{1,3}",
                                        lines[i].strip())}
        notes_of[pg.key] = frozenset(pg.note_lines | stray_heads)
        a = running_head_end(lines)
        end = footnote_line_start(lines, pg.foot_line) - 1
        seg_start = a
        for i in range(a, end + 1):
            s = lines[i].strip()
            if stop is not None and stop.match(s):
                if role in ("intro", "translation"):
                    add(role, pg.key, seg_start, i - 1)
                stopped = True
                break
            head = ANTH_HEAD.match(s)
            if head and not (last_no < int(head.group("no")) <= last_no + max_skip):
                head = None
            lang = _is_lang_line(s)
            if head:
                if sec is not None and role in ("intro", "translation"):
                    add(role, pg.key, seg_start, i - 1)
                last_no = int(head.group("no"))
                title = head.group("title").strip().strip("\"“”")
                sec = Section(label=head.group("no"), kind="anthology", shelfmark_raw=None,
                              extra={"title": title, "heading": (pg.key, i), "docs": []})
                if _SM_LINE.match(title) and re.search(r"\d", title) and len(title) <= 60:
                    sec.shelfmark_raw = _clean_sm(title)
                    sec.extra["title_is_shelfmark"] = True
                    sec.extra["docs"].append({"shelfmark": sec.shelfmark_raw, "language": None,
                                              "alt": alt_classmark(title), "line": (pg.key, i), "translation": []})
                out.append(sec)
                role, seg_start = "intro", i + 1
                title_open = bool(re.match(r"^[\"“‘']", head.group("title").strip())) and not re.search(
                    r"[\"”’']\s*$", head.group("title").strip())
                title_lines = 0
            elif sec is not None and role == "intro" and title_open and title_lines < 3 and i == seg_start:
                # a heading title wrapped onto the next line(s): fold it into the title, not the introduction
                sec.extra["title"] = (sec.extra["title"] + " " + s).strip()
                title_lines += 1
                seg_start = i + 1
                if re.search(r"[\"”’']\s*$", s):
                    title_open = False
            elif (sec is not None and role == "intro" and sec.extra.get("title_is_shelfmark")
                  and LIST_START.match(s)):
                add("intro", pg.key, seg_start, i - 1)
                role, seg_start = "translation", i
            elif lang and sec is not None:
                if role in ("intro", "translation"):
                    add(role, pg.key, seg_start, i - 1)
                sec.extra["docs"].append({"shelfmark": lang[0], "language": lang[1], "alt": lang[2],
                                          "line": (pg.key, i), "translation": []})
                if sec.shelfmark_raw is None:
                    sec.shelfmark_raw = lang[0]
                role, seg_start = "translation", i + 1
        else:
            if sec is not None and role in ("intro", "translation"):
                add(role, pg.key, seg_start, end)
    for s in out:
        s.ranges = [r for d in s.extra["docs"] for r in d["translation"]]
        s.text = _join_ranges(page_lines, s.ranges)
        s.extra["intro_text"] = _join_ranges(page_lines, s.extra.get("intro", []))
        for d in s.extra["docs"]:
            d["text"] = _join_ranges(page_lines, d["translation"])
    return out


# ------------------------------------------------------------------------------------ numbered catalogue
def catalogue_entries(pages: Sequence[PageInput], min_no: int = 1, max_no: int = 2000, footnotes: bool = False,
                      max_jump: int = 8) -> List[Section]:
    """Cut a catalogue whose entries open with a line holding only the entry number (in ascending order).

    A number line is an entry start only when it continues the sequence (next number within ``max_jump`` of the
    last accepted one), which keeps page numbers and stray digits out.

    :param pages: Pages in reading order.
    :param min_no: First entry number expected.
    :param max_no: Largest entry number accepted.
    :param footnotes: Cut pages at their footnotes (off by default: bare entry-number lines look like notes).
    :param max_jump: Largest step between consecutive entry numbers (entries whose number line the OCR lost).
    :returns: Sections (``kind="catalogue"``, label = entry number).
    """
    out: List[Section] = []
    page_lines: Dict[str, List[str]] = {}
    last = min_no - 1
    cur: Optional[Section] = None
    for pg in pages:
        lines = pg.text.split("\n")
        page_lines[pg.key] = lines
        a = running_head_end(lines)
        end = (footnote_line_start(lines, pg.foot_line) if footnotes else len(lines)) - 1
        seg = a
        for i in range(a, end + 1):
            s = lines[i].strip()
            if re.fullmatch(r"\d{1,4}", s) and last < int(s) <= min(last + max_jump, max_no):
                if cur is not None and seg <= i - 1:
                    cur.ranges.extend((pg.key, x, y) for x, y in _split_on_stamps(lines, seg, i - 1))
                cur = Section(label=s, kind="catalogue", shelfmark_raw=None)
                out.append(cur)
                last = int(s)
                seg = i + 1
        if cur is not None and seg <= end:
            cur.ranges.extend((pg.key, x, y) for x, y in _split_on_stamps(lines, seg, end))
    for s in out:
        s.text = _join_ranges(page_lines, s.ranges)
    return out


# ------------------------------------------------------------------------------------ edition runs
def rtl_runs(lines: Sequence[str], min_lines: int = 4, min_letters: int = 40, gap: int = 1) -> List[Tuple[int, int]]:
    """Runs of Hebrew / Arabic-script lines (edition text) in a page.

    A run may bridge ``gap`` consecutive non-RTL lines (line numbers, sigla) between RTL lines.

    :param lines: Page lines.
    :param min_lines: Minimum RTL lines in a run.
    :param min_letters: Minimum Hebrew + Arabic letters in a run.
    :param gap: Non-RTL lines a run may bridge.
    :returns: ``[(first, last)]`` inclusive line ranges.
    """
    def rtl(s: str) -> bool:
        """Mostly Hebrew / Arabic-script line.

        :param s: Line.
        :returns: True when RTL letters outnumber Latin ones (≥ 3 of them).
        """
        he, ar, lat = letters(s)
        return he + ar >= 3 and he + ar > lat

    out: List[Tuple[int, int]] = []
    i, n = 0, len(lines)
    while i < n:
        if not rtl(lines[i]) or is_stamp_line(lines[i]):
            i += 1
            continue
        j, last_rtl, miss = i, i, 0
        while j + 1 < n:
            nxt = lines[j + 1]
            if rtl(nxt) and not is_stamp_line(nxt):
                last_rtl, miss = j + 1, 0
            elif miss < gap and not is_stamp_line(nxt):
                miss += 1
            else:
                break
            j += 1
        seg = lines[i:last_rtl + 1]
        k = sum(1 for s in seg if rtl(s))
        he, ar, _ = letters("\n".join(seg))
        if k >= min_lines and he + ar >= min_letters:
            out.append((i, last_rtl))
        i = last_rtl + 1
    return out
