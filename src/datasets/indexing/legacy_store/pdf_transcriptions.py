# File name: pdf_transcriptions.py
# Date: 10/30/26
# Author: Isaac Godfried. Coded by Claude.
"""Structured ingest of legacy-store transcription PDFs: one record per folio.

The project's legacy image store serves transcriptions as PDFs the user downloads by hand
(``transcription_manual_<shelf mark>.pdf``, ``transcription_<uuid>.pdf``,
``transcription_response_<uuid>.pdf``). Unlike the flat one-line section strings of the store's
pages, a PDF keeps the transcriber's line breaks and folio headings, so a folio's text arrives
with its side already known.

Steps (each a public function):

1. :func:`is_junk` spots the store's error page; :func:`content_digest` keys byte-level and
   text-level copies (the same document downloaded several times, uuid plain/response pairs);
   :func:`dedupe_pdfs` groups a folder's PDFs by it.
2. :func:`page_lines` rebuilds visual lines from the text layer: spans sharing a baseline form a
   line, ordered right to left when the line holds Hebrew or Arabic letters (a span's own text is
   already in logical order), left to right otherwise.
3. :func:`parse_header` reads the page-1 header (institution, shelf mark, title) when the PDF has
   one; :func:`series_key_for` maps a shelf mark to the image file-name series through the explicit
   table :data:`SERIES_TABLE`; :func:`shelfmark_from_filename` reads ``transcription_manual_`` names.
4. :func:`split_folios` cuts the line stream at folio headings (:func:`match_folio_marker`: ``1r``;
   ``דף א צד א``; the visual-order ``ד1 א" ע``; ``1: א``; ``3ב :`` before text), drops the editor's
   notes after ``הערות:``, an underscore rule or a sigla legend up to the next heading, and falls
   back to one folio per PDF page labelled ``p1``, ``p2`` ... when the document has no heading (flag
   ``no_folio_markers``). Documents whose first line is ``העתקות`` (copies) or ``אתחלתא``
   (incipit) are excerpt lists: their folios are passages, flagged ``excerpt_document``. A ``daf``
   heading followed by ``(גמרא)`` / ``(משנה)`` cites a printed edition page (kind ``edition_ref``,
   never a leaf); a heading naming another shelf mark (:func:`heading_shelfmark`, a joined
   fragment filed under one mark) flags its folio ``heading_other_shelfmark``.
5. :func:`font_check` reports which fonts carry the letters and every symbol in word-final
   position after a Hebrew letter (legacy fonts map final letters to ``$ # & % + ! ,`` or to control
   characters).
6. :func:`clean_folio` strips line numbers, re-joins lines the PDF wrapped, puts displaced gereshes
   back (:func:`reattach_geresh`), and applies
   :func:`src.datasets.cleaning.legacy_store_sigla.clean_section` ONCE PER FOLIO to the folio text
   with explicit line breaks (not once per line: the family, bracket-orientation, reversal and
   finals decisions need the whole side, and brackets may span lines; the cleaner keeps ``\\n``
   and drops only lines left without letters). Lines reading like the editor's notes are flagged
   ``editor_note_suspect``; :func:`flag_editor_note_document` flags a whole document where a third
   or more of the folios carry one.
7. :func:`pair_images` finds a CANDIDATE image for a folio by its file name
   (``<Series>_<number>_<folio>.jpg``, Firefox ``(n)`` copies deduplicated by SHA-256, the largest
   file kept); :func:`ingest` runs the steps for one PDF.
8. A second style (:func:`detect_style`): old downloads have no header and no folio headings and
   hold ONE SIDE per file (``transcription_manual_<shelf mark>[_<n>].pdf``, side ``n``, default 1;
   :func:`parse_side_file_name`), with inline ``טור_א`` / ``טור_ב`` column markers.
   :func:`split_side_columns` cuts a side into columns, :func:`ingest_side_files` cleans the sides
   of one shelf mark with one finals mapping, :func:`merge_side_redownloads` folds re-downloads
   saved under a higher suffix into the side they copy (:func:`side_content_key`),
   :func:`renumber_sides` numbers the distinct sides by suffix order and
   :func:`group_side_records` groups side records into documents.
   :func:`near_duplicate_pairs` finds texts saved twice with near-identical letters.
9. :func:`provenance_signals` summarises a transcription's format family and quality signals;
   :func:`page_row_exclusions` lists why a folio may not become a page row.

Image pairing is by file name only (``image_match = "file_name_token_unverified"``): the token of
the new downloads is a viewer sequence number, not a folio (one image checked, 1/1 mismatched;
0/388 tokens end in ``v``), so ``image_candidate_path`` is never a page row's image until an
image-to-folio check exists.
"""
import collections
import difflib
import hashlib
import os
import re
import statistics
from dataclasses import dataclass, field
from typing import Callable, Counter, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import fitz

from src.datasets.cleaning import legacy_store_sigla as L
from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer

# ───────────────────────────── constants ─────────────────────────────
JUNK_PHRASE = "A minor technical problem was encountered while displaying the requested image"
NBSP = chr(0xA0)
SPAN_NEWLINE = "\u2424"          # a newline INSIDE a span's text (a legacy-font final, never a line break)
FINAL_SYMBOLS = "$#&%+!,"
# Symbols the committed finals inference never proposes (its candidate class excludes them):
# word-final uses are routed through private-use proxies for the inference only.
_INFERENCE_PROXIES = {"!": "\ue0a1", ",": "\ue0a2"}
SYMBOL_CHAR_RE = re.compile(r"[$#&%+!,\x00-\x1f\x7f\u2424]")
HEB_LETTER_RE = L.HEBREW_LETTER_RE
ARABIC_LETTER_RE = L.ARABIC_LETTER_RE
LETTER_RE = L.LETTER_RE
RTL_CHAR_RE = re.compile(r"[\u0590-\u05FF\u0600-\u06FF\uFB1D-\uFDFF\uFE70-\uFEFF]")
CONTROL_OR_SPAN_NL_RE = re.compile(r"[\x00-\x1f\x7f\u2424]")
INSTITUTION_RE = re.compile(r"\b(?:Library|Museum|Institute|Collection|Archives?|University|Seminary)\b|Firkovi",
                            re.IGNORECASE)
# Notes heading (also in visual order, colon first), apparatus rule, sigla legend.
NOTES_START_RE = re.compile(r"^\s*(?::?\s*הערות\s*:?|_{10,})\s*$|^\s*מקרא\s+סימני")
# Only spaces and tabs around the number: Python's \s also matches control characters that may be finals.
LINE_NUMBER_RE = re.compile(r"^[ \t]*(\d{1,4})(?:[ \t]*\.)?[ \t]*(?=[^\d \t.:])")
# A geresh the text layer moved from the end of a word to the start of the next (``אמ 'לו``).
LEAD_GERESH_RE = re.compile(r"(?<=[א-ת]) (['׳])(?=[א-ת])")
TRAIL_GERESH_RE = re.compile(r"(?<=[א-ת])(['׳]) (?=[א-ת])")
WORD_FINAL_QUOTE_RE = re.compile(r'[א-ת]["״](?![א-ת])')
# The editor's in-line labels: "uncertain reading", and the catchword label ``שומר הדף:`` (the label
# goes; the catchword after it is real ink and stays). Spaces and tabs only around the label: a
# ``\s`` there would eat the line break before it and glue the catchword to the line above.
EDITOR_PHRASE_RE = re.compile(r"קריאה\s+מסופקת|:?[ \t]*שומר[ \t]+הדף[ \t]*:?")
# Words of the editor's comments on the margin, the photograph, proof marks and line counts, and on
# the physical page (diagonal lines, "above the line" ``מע"ה``, stains, holes, ink smears and ink
# squares, erasures, scraping, a line struck through): a cleaned line holding one is a suspected
# editor's note (the folio is kept but excluded from rows). The physical-page words are bounded so
# that the ordinary words ``בן קרחה`` and ``ולמחיקה`` do not match.
EDITOR_NOTE_RE = re.compile(r"בגליון|לגליון|הגליון|הגהה|בצילום|מהדף\s+הקוד|שורות\s+למטה|מס['׳]\s+שורות|כנראה\s+מד"
                            r"|אלכסוני|(?<![א-ת])מע[\"״]?\s*ה[\"״]|(?<![א-ת])מע\s+ה(?![א-ת])|(?<![א-ת])מע[\"״]ה(?![א-ת])"
                            r"|מריחת\s+דיו|ריבוע\s+דיו"
                            r"|חור\s+בקל|(?<![א-ת])כתם?\s+(?:על|מע)(?![א-ת])|קרחה\s+דרכה|גירוד|חוצה\s+למחיקה")
# A document whose folios carry ``editor_note_suspect`` this often (at least a third) mixes the
# editor's notes into the text throughout: every folio gets ``editor_note_document``.
EDITOR_NOTE_DOCUMENT_SHARE = 1 / 3
# Folio flags that keep a folio out of page rows (see :func:`page_row_exclusions`).
PAGE_ROW_EXCLUDING_FLAGS = ("excerpt_document", "word_quote_separators", "duplicate_folio_label", "unresolved_finals",
                            "editor_note_suspect", "editor_note_document", "side_sequence_gap", "line_count_outlier",
                            "side_index_conflict", "side_index_duplicate", "shelfmark_unresolved",
                            "shelfmark_name_conflict", "heading_other_shelfmark", "near_duplicate_document")
# Near-duplicate texts (same transcription saved twice with another text-layer order or a few edits):
# letter similarity at or above this, on texts of at least NEAR_DUPLICATE_MIN_LETTERS letters whose
# lengths are within NEAR_DUPLICATE_MIN_LENGTH_RATIO of each other.
NEAR_DUPLICATE_RATIO = 0.98
NEAR_DUPLICATE_MIN_LETTERS = 100
NEAR_DUPLICATE_MIN_LENGTH_RATIO = 0.9
LINE_COUNT_OUTLIER_FACTOR = 1.6
FOLIO_LABEL_RE = re.compile(r"\d{1,3}[rv]")
_COPY_NAME_RE = re.compile(r"^(?P<stem>.+?)(?:\((?P<n>\d+)\))?(?P<ext>\.(?:jpe?g|png|tiff?))$", re.IGNORECASE)
_GEMATRIA = {"א": 1, "ב": 2, "ג": 3, "ד": 4, "ה": 5, "ו": 6, "ז": 7, "ח": 8, "ט": 9, "י": 10, "כ": 20,
             "ל": 30, "מ": 40, "נ": 50, "ס": 60, "ע": 70, "פ": 80, "צ": 90, "ק": 100, "ר": 200, "ש": 300,
             "ת": 400}
_SIDE_OF = {"א": "r", "ב": "v", "r": "r", "v": "v"}

# Shelf mark -> image file-name series. Keys are matched against :func:`normalise_shelfmark`
# output; the first match wins, so the more specific series come first. Unknown series -> None.
SERIES_TABLE: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("Yevr_Arab_II", re.compile(r"^Y?EVR ARAB II (\d+)\b")),
    ("Yevr_Arab_I", re.compile(r"^Y?EVR ARAB I (\d+)\b")),
    ("Yevr_II_A", re.compile(r"^Y?EVR II A (\d+)\b")),
)

# Folio headings, matched against a line with control characters removed (see
# :func:`match_folio_marker`). ``rest`` is what follows the heading on its line.
_MARKER_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("latin", re.compile(r"^[\s'׳]*(?P<leaf>\d{1,3})(?P<side>[rv])[\s'׳]*$")),
    ("daf", re.compile(r"^[\s'׳]*דף\s*(?P<leaf>[א-ת]{1,3}|\d{1,3})\s*['׳]?\s*(?:צד|עמוד|ע[\"״])\s*['׳]?\s*"
                       r"(?P<side>[אב])(?![א-ת])(?P<rest>.*)$")),
    ("amud_visual", re.compile(r"^[\s'׳]*ד\s*(?P<leaf>\d{1,3})\s*(?P<side>[אב])[\"״]\s*ע(?![א-ת])(?P<rest>.*)$")),
    ("amud_visual", re.compile(r"^[\s'׳]*ד\s*(?P<leaf>\d{1,3})[\"״]\s*ע\s*(?P<side>[אב])(?![א-ת])(?P<rest>.*)$")),
    ("numeric", re.compile(r"^[\s:()\[\]]*(?P<leaf>\d{1,3})\s*:?\s*(?P<side>[אב])[\s:()\[\]]*$")),
    ("colon_prefix", re.compile(r"^\s*(?P<leaf>\d{1,3})(?P<side>[אב])\s*:\s*(?P<rest>\S.*)$")),
    ("numeric_colon", re.compile(r"^[\s:]*(?P<leaf>\d{1,3})\s*:\s*(?P<side>[אב])(?P<rest>.*\S.*)$")),
    # A bracketed number and side letter before text (``(1א)`` or the mirrored ``)1א(``), as in
    # the incipit lists.
    ("bracketed", re.compile(r"^[\s:]*[()\[\]]\s*(?P<leaf>\d{1,3})\s*:?\s*(?P<side>[אב])\s*[()\[\]]"
                             r"(?P<rest>.*\S.*)$")),
)
# Heading kinds whose trailing text is the folio's first line (not an edition reference).
TEXT_AFTER_HEADING_KINDS = ("colon_prefix", "numeric_colon", "bracketed")
# A ``daf`` heading followed by ``(גמרא)`` or ``(משנה)`` (brackets in either orientation) cites a page
# of the printed Talmud or Mishnah, not a leaf of the manuscript: kind ``edition_ref``, label
# ``ed<page><r|v>`` (never a folio label).
EDITION_REF_REST_RE = re.compile(r"^[\s()\[\]'׳]*(?:גמרא|משנה)[\s()\[\]'׳]*$")
# A Latin run with a digit in a heading's trailing text (``( 'A 444 1R) ...``): a shelf mark, with
# an optional final ``<leaf><R|V>`` token naming the side.
HEADING_LATIN_RUN_RE = re.compile(r"[A-Za-z](?:[A-Za-z0-9 .\-]|\(\d+\))*\d(?:[A-Za-z0-9.\-]|\(\d+\))*")
HEADING_SIDE_TOKEN_RE = re.compile(r"^(?P<mark>.+?)[ \t]+(?P<leaf>\d{1,3})[ \t]*(?P<side>[RrVv])$")
HOLDER_ID_PREFIX_RE = re.compile(r"^(?:JRL|RNL|NLR)_", re.IGNORECASE)
# A first line naming the document an excerpt list ("copies") or an incipit: its "folios" are
# passages quoted from those leaves, not full-page transcriptions.
EXCERPT_LABEL_RE = re.compile(r"^\s*(?:העתקות|אתחלתא)\s*:?")
# Heading kinds that never occur as text lines; the others (a number and a side letter) could.
STRONG_MARKER_KINDS = frozenset({"latin", "daf", "amud_visual", "edition_ref"})
# Old style: ``transcription_manual_<shelf mark>[_<side>].pdf``; ``(n)`` / ``-n`` are download copies.
SIDE_FILE_RE = re.compile(r"^transcription_manual_(?P<stem>.+?)(?P<copy>\(\d+\)|-\d+)?\.pdf$", re.IGNORECASE)
MAX_SIDE_SUFFIX = 9
# Inline column markers of the old style (``טור_א``, ``טור ב``), alone on their line or before text.
COLUMN_MARKER_RE = re.compile(r"^[\s'׳]*טור[_\s]*(?P<col>[א-ד])(?![א-ת])\s*:?\s*(?P<rest>.*)$")
# Transcription sigla (brackets, uncertainty, tags, scribal stops): counted per 100 letters.
SIGLA_CHAR_RE = re.compile(r"[\[\]{}()<>?@^*·¶|\\~&]")
# Institution heads of the store's ``Head: core`` shelf marks, as they open a file-name stem.
DEFAULT_INSTITUTION_HEADS = (
    "St. Petersburg, St. Peterburg", "Cambridge, CUL", "Cambridge CUL", "Cambridge Lewis-Gibson", "Manchester",
    "New York JTS", "Paris AIU", "Oxford", "London BL", "Budapest MTA", "Philadelphia Penn CAJS", "Jerusalem NLI",
    "Birmingham", "Strasbourg", "Geneva", "Frankfurt", "Vienna Wien N", "Washington D.C. Freer",
)


# ───────────────────────────── data ─────────────────────────────
@dataclass
class PdfLine:
    """One visual line of a PDF page.

    :param text: Span texts joined in reading order.
    :param page_index: 0-based PDF page.
    :param y: Baseline (points from the page top).
    :param spans: ``(font, text)`` per span, in reading order.
    :param rtl: Whether the line was ordered right to left.
    """

    text: str
    page_index: int
    y: float
    spans: List[Tuple[str, str]] = field(default_factory=list)
    rtl: bool = False


@dataclass
class FolioMarker:
    """A folio heading found on a line.

    :param label: ``"<leaf><r|v>"`` (``"12v"``).
    :param kind: Which heading form matched (a :data:`_MARKER_PATTERNS` name).
    :param rest: Text after the heading on the same line ("" when none).
    """

    label: str
    kind: str
    rest: str = ""


@dataclass
class _Span:
    """A text-layer span with its geometry.

    :param text: Span text (see :func:`normalise_span_text`).
    :param font: Font name.
    :param x0: Left edge.
    :param x1: Right edge.
    :param y: Baseline.
    :param size: Font size.
    """

    text: str
    font: str
    x0: float
    x1: float
    y: float
    size: float


# ───────────────────────────── text layer ─────────────────────────────
def normalise_span_text(text: str) -> str:
    """Make a span's text safe to join into lines.

    A newline inside a span is a legacy-font symbol (seen standing for a final letter), never a
    line break: it becomes :data:`SPAN_NEWLINE`. No-break spaces become spaces.

    :param text: Span text as extracted.
    :return: The text.
    """
    return text.replace("\n", SPAN_NEWLINE).replace(NBSP, " ")


def full_text(doc: "fitz.Document") -> str:
    """The whole text layer, page by page, as MuPDF extracts it.

    :param doc: Open PDF.
    :return: Concatenated page texts.
    """
    return "".join(page.get_text() for page in doc)


def is_junk(doc: "fitz.Document") -> bool:
    """Whether a PDF is the legacy store's error page instead of a transcription.

    :param doc: Open PDF.
    :return: True when the text layer holds :data:`JUNK_PHRASE` (whitespace-insensitive).
    """
    return JUNK_PHRASE in re.sub(r"\s+", " ", full_text(doc))


def content_digest(doc: "fitz.Document") -> str:
    """SHA-1 of the full text layer: equal for copies of one download whatever the file bytes.

    :param doc: Open PDF.
    :return: 40 hex digits.
    """
    return hashlib.sha1(full_text(doc).encode("utf-8")).hexdigest()


def dedupe_pdfs(paths: Iterable[str]) -> List[Dict[str, object]]:
    """Group PDFs by :func:`content_digest`.

    The canonical file of a group is the earliest download (modification time, then file name).

    :param paths: PDF paths.
    :return: One dict per group, in canonical-file order: ``digest``, ``canonical`` (path),
        ``copies`` (all paths, canonical first), ``junk`` (bool), ``pages``.
    """
    groups: Dict[str, Dict[str, object]] = {}
    for path in paths:
        with fitz.open(path) as doc:
            digest = content_digest(doc)
            junk = is_junk(doc)
            pages = len(doc)
        groups.setdefault(digest, {"digest": digest, "copies": [], "junk": junk, "pages": pages})
        groups[digest]["copies"].append(path)
    out = []
    for g in groups.values():
        g["copies"] = sorted(g["copies"], key=lambda p: (os.path.getmtime(p), os.path.basename(p)))
        g["canonical"] = g["copies"][0]
        out.append(g)
    return sorted(out, key=lambda g: (os.path.getmtime(g["canonical"]), os.path.basename(g["canonical"])))


def page_lines(page: "fitz.Page", page_index: int = 0, tolerance: float = 0.3) -> List[PdfLine]:
    """Visual lines of one page, rebuilt from the text layer's spans.

    Spans whose baselines differ by at most ``tolerance`` x font size form one line (MuPDF's own
    lines split a visual line wherever the font, a gap or the direction changes). A line with a
    Hebrew or Arabic letter is ordered right to left by span centre, any other line left to right;
    a space is inserted between spans separated by more than a quarter of the font size when
    neither side has one. Blank lines are omitted.

    :param page: PDF page.
    :param page_index: Its 0-based index (stored on the lines).
    :param tolerance: Baseline tolerance as a fraction of the font size.
    :return: Lines from top to bottom.
    """
    spans: List[_Span] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for s in line["spans"]:
                text = normalise_span_text(s["text"])
                if text:
                    spans.append(_Span(text, s["font"], s["bbox"][0], s["bbox"][2], s["origin"][1], s["size"]))
    spans.sort(key=lambda s: s.y)
    groups: List[List[_Span]] = []
    for sp in spans:
        if groups and abs(sp.y - groups[-1][0].y) <= tolerance * max(sp.size, groups[-1][0].size, 1.0):
            groups[-1].append(sp)
        else:
            groups.append([sp])
    out: List[PdfLine] = []
    for grp in groups:
        rtl = any(RTL_CHAR_RE.search(s.text) for s in grp)
        grp.sort(key=lambda s: -(s.x0 + s.x1) / 2 if rtl else (s.x0 + s.x1) / 2)
        text = grp[0].text
        for prev, cur in zip(grp, grp[1:]):
            gap = (prev.x0 - cur.x1) if rtl else (cur.x0 - prev.x1)
            if gap > 0.25 * max(cur.size, 1.0) and not text.endswith(" ") and not cur.text.startswith(" "):
                text += " "
            text += cur.text
        if text.strip(" \t" + NBSP):
            out.append(PdfLine(text=text, page_index=page_index, y=grp[0].y,
                               spans=[(s.font, s.text) for s in grp], rtl=rtl))
    return out


def document_lines(doc: "fitz.Document") -> List[PdfLine]:
    """Visual lines of every page, in page order.

    :param doc: Open PDF.
    :return: Lines.
    """
    out: List[PdfLine] = []
    for i, page in enumerate(doc):
        out.extend(page_lines(page, i))
    return out


# ───────────────────────────── shelf marks ─────────────────────────────
def normalise_shelfmark(text: str) -> str:
    """Comparable form of a shelf mark (header line, catalogue string or file-name stem).

    Drops a ``St. Petersburg, St. Peterburg:`` prefix (catalogue form) or
    ``St_Petersburg_St_Peterburg_`` (file-name form), turns ``_ . - ,`` into spaces, upper-cases,
    splits ``IIA`` into ``II A``, and writes ``YEVR`` as ``EVR``.

    :param text: Shelf mark in any of the forms seen.
    :return: E.g. ``"EVR II A 223 2"``, ``"EVR ARAB II 1201"``.
    """
    s = text.strip()
    if ":" in s:
        s = s.rsplit(":", 1)[1]
    s = re.sub(r"^St[_ .]*Petersburg[_ ,]*St[_ .]*Peterburg[_ ]*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[_.\-,]", " ", s).upper()
    s = re.sub(r"\bIIA\b", "II A", s)
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"^YEVR\b", "EVR", s)


def series_key_for(shelfmark: Optional[str]) -> Optional[Tuple[str, str]]:
    """Image file-name series of a shelf mark (``Evr. IIA 223.2`` -> ``("Yevr_II_A", "223")``).

    FROZEN 2026-11: key format of image file names on disk; new code must use src.datasets.shelfmarks

    The sub-number after the dot is not part of image names, so different sub-shelf-marks of one
    number share a series key.

    :param shelfmark: Shelf mark in any form :func:`normalise_shelfmark` reads, or None.
    :return: ``(series, number)`` or None when the series is not in :data:`SERIES_TABLE`.
    """
    if not shelfmark:
        return None
    norm = normalise_shelfmark(shelfmark)
    for series, pattern in SERIES_TABLE:
        m = pattern.match(norm)
        if m:
            return series, m.group(1)
    return None


def shelfmark_from_filename(name: str) -> Optional[str]:
    """Shelf mark written in a ``transcription_manual_<shelf mark>.pdf`` file name.

    FROZEN 2026-11: key format of image file names on disk; new code must use src.datasets.shelfmarks

    :param name: File name (or path).
    :return: The shelf mark with underscores as spaces and the city prefix removed
        (``"Yevr Arab II 1201"``, ``"Evr. IIA 223.2"``); None for uuid-named files.
    """
    base = os.path.basename(name)
    m = re.match(r"^transcription_manual_(.+?)\.pdf$", base, flags=re.IGNORECASE)
    if not m:
        return None
    stem = re.sub(r"^St_Petersburg_St_Peterburg_", "", m.group(1), flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", stem.replace("_", " ")).strip() or None


def head_to_stem_prefix(head: str) -> str:
    """File-name form of an institution head (``"Cambridge, CUL"`` -> ``"Cambridge_CUL_"``).

    :param head: Institution head as printed before the colon.
    :return: Its underscore form with a trailing underscore.
    """
    return re.sub(r"[^A-Za-z0-9]+", "_", head).strip("_") + "_"


def strip_institution(mark: str, heads: Sequence[str] = DEFAULT_INSTITUTION_HEADS) -> Tuple[str, Optional[str]]:
    """Split a shelf mark into (core, institution head).

    ``Head: core`` strings split at the last colon; underscore stems (file names) lose a leading
    head in its file-name form (``Cambridge_CUL_T_S_AS_97_1`` -> ``T_S_AS_97_1``; repeated
    underscores, as in ``Cambridge__CUL__``, are collapsed first).

    :param mark: Shelf mark in any spelling.
    :param heads: Known institution heads.
    :return: (core, head or None).
    """
    s = mark.strip()
    if ":" in s:
        head, core = s.rsplit(":", 1)
        return core.strip(), head.strip() or None
    collapsed = re.sub(r"_+", "_", s)
    for head in sorted(heads, key=len, reverse=True):
        prefix = head_to_stem_prefix(head)
        if collapsed.lower().startswith(prefix.lower()):
            return collapsed[len(prefix):], head
    return s, None


def parse_side_file_name(name: str, sibling_stems: Optional[Iterable[str]] = None,
                         is_known: Optional[Callable[[str], bool]] = None,
                         heads: Sequence[str] = DEFAULT_INSTITUTION_HEADS) -> Optional[Dict[str, object]]:
    """Shelf mark and side index of an old-style ``transcription_manual_<shelf mark>[_<n>].pdf``.

    The plain name is side 1; ``_2`` ... ``_9`` name sides 2-9. A final ``_<n>`` is ambiguous
    (``..._AS_97_2`` is T-S AS 97.2 side 1, or T-S AS 97 side 2), so it is read as a side only on
    evidence, in this order: a sibling file named by the shorter stem (``sibling_stems``) when the
    long stem is not a known shelf mark; ``is_known`` (a corpus or catalogue lookup) knowing one
    reading but not the other; otherwise the naming rule (suffix = side), flagged ambiguous. A
    download-copy suffix (``-1``, ``(1)``) is kept apart: such a copy's side is only trusted when
    its text equals a file named without one (the packager checks that).

    :param name: File name (or path).
    :param sibling_stems: Stems (``<shelf mark>`` part, institution included) of the other
        ``transcription_manual_`` files of the same folder, copy suffixes removed.
    :param is_known: Callable telling whether a core shelf mark (underscores) is a known item.
    :param heads: Institution heads for :func:`strip_institution`.
    :return: ``stem`` (shelf-mark part of the name, side suffix removed), ``core`` (institution
        removed), ``institution``, ``shelfmark`` (readable), ``canonical_id`` (merging normaliser),
        ``side_index``, ``side_resolution`` (``no_suffix`` / ``sibling_file`` / ``corpus`` /
        ``suffix_rule``), ``side_ambiguous``, ``copy_suffix``; None for another name shape.
    """
    m = SIDE_FILE_RE.match(os.path.basename(name))
    if not m:
        return None
    stem = m.group("stem")
    core, head = strip_institution(stem, heads)
    siblings = {re.sub(r"_+", "_", x) for x in (sibling_stems or ())}
    tail = re.search(r"_(\d+)$", core)
    side, resolution, ambiguous, use = 1, "no_suffix", False, stem
    if tail and 2 <= int(tail.group(1)) <= MAX_SIDE_SUFFIX:
        n = int(tail.group(1))
        short_core = core[:tail.start()]
        short_stem = re.sub(r"_+", "_", stem)[:-len(tail.group(0))]
        sibling = short_stem in siblings
        full_known = bool(is_known and is_known(core))
        short_known = bool(is_known and is_known(short_core))
        if sibling and not full_known:
            side, resolution, use = n, "sibling_file", short_stem
        elif short_known and not full_known:
            side, resolution, use = n, "corpus", short_stem
        elif full_known and not short_known and not sibling:
            resolution = "corpus"
        else:
            side, resolution, use, ambiguous = n, "suffix_rule", short_stem, True
    core, head = strip_institution(use, heads)
    display = re.sub(r"\s+", " ", core.replace("_", " ")).strip()
    canonical = ShelfmarkNormalizer.to_canonical_id(f"{head}: {display}" if head else display)
    return {"stem": use, "core": core, "institution": head, "shelfmark": display, "canonical_id": canonical,
            "side_index": side, "side_resolution": resolution, "side_ambiguous": ambiguous,
            "copy_suffix": m.group("copy")}


def _is_shelfmark_line(line: str) -> bool:
    """Whether a header line looks like a Latin shelf mark.

    :param line: Stripped line.
    :return: True for a short Latin line with a digit that names no institution.
    """
    return (bool(re.match(r"^[A-Za-z]", line)) and bool(re.search(r"\d", line)) and len(line) <= 40
            and not LETTER_RE.search(line) and not INSTITUTION_RE.search(line)
            and match_folio_marker(line) is None)


def parse_header(page1_lines: Sequence[str], max_lines: int = 6) -> Optional[Dict[str, object]]:
    """Read the header at the top of page 1.

    The header is an institution line (Latin, naming a library or collection), a shelf-mark line
    (``Evr. IIA 223.2``) and a title line (usually Hebrew), before the first folio heading. A shelf
    mark is accepted when an institution line precedes it or its series is in
    :data:`SERIES_TABLE`.

    :param page1_lines: Text of page 1's lines, top to bottom.
    :param max_lines: How many non-blank lines are searched.
    :return: ``institution``, ``shelfmark``, ``title``, ``series_key`` (tuple or None) and
        ``header_lines`` (non-blank lines the header occupies); None when there is no header.
    """
    lines = [ln.strip() for ln in page1_lines if ln.strip()][:max_lines]
    for i, line in enumerate(lines):
        if match_folio_marker(line) is not None:
            return None
        if not _is_shelfmark_line(line):
            continue
        after_institution = i > 0 and bool(INSTITUTION_RE.search(lines[i - 1]))
        key = series_key_for(line)
        if not after_institution and key is None:
            continue
        institution = lines[i - 1] if after_institution else None
        title = None
        if i + 1 < len(lines) and match_folio_marker(lines[i + 1]) is None and LETTER_RE.search(lines[i + 1]):
            title = lines[i + 1]
        return {"institution": institution, "shelfmark": line, "title": title, "series_key": key,
                "header_lines": i + (2 if title else 1)}
    return None


# ───────────────────────────── folios ─────────────────────────────
def hebrew_numeral(token: str) -> Optional[int]:
    """Value of a leaf number written in digits or Hebrew numeral letters.

    :param token: ``"12"``, ``"ה"``, ``"יב"``.
    :return: The number, or None for an empty token.
    """
    if token.isdigit():
        return int(token)
    return sum(_GEMATRIA.get(ch, 0) for ch in token) or None


def match_folio_marker(text: str) -> Optional[FolioMarker]:
    """Folio heading at the start of a line, if any.

    Control characters and in-span newlines (legacy-font symbols) are removed before matching.
    Forms (:data:`_MARKER_PATTERNS`): ``1r`` / ``12v`` alone; ``דף א צד א`` / ``דף ה עמוד ב`` /
    ``דף 3 ע"א``; the visual-order ``ד1 א" ע`` or ``ד2" עא`` (``דף 1 ע"א``); a number and a side
    letter alone (``1: א``, ``13 : ב``, ``5 א``); a number and side letter with a colon before text
    (``3ב :``, kind ``colon_prefix``); a number, a colon and a side letter before text (``6: א``
    followed by the text with no gap, kind ``numeric_colon``: in the excerpt lists the words of the
    heading are lost glyphs, so a line number written with a colon before a word starting with
    ``א``/``ב`` would be misread; no such line numbering was seen); a bracketed number and side
    letter before text (``(1א) <text>``, also mirrored ``)1א(``, kind ``bracketed``). Side ``א`` is
    recto, ``ב`` verso. A leading excerpt label (:data:`EXCERPT_LABEL_RE`: ``אתחלתא:``, ``העתקות:``) is
    skipped first, so ``אתחלתא: 1: א`` and ``אתחלתא: (1א)`` are headings of leaf 1. A ``daf`` heading
    whose trailing text is ``(גמרא)`` or ``(משנה)`` (:data:`EDITION_REF_REST_RE`) cites a page of the
    printed edition: kind ``edition_ref``, label ``ed<page><r|v>`` (not a leaf).

    :param text: Line text.
    :return: The marker, or None. ``rest`` is cut from the ORIGINAL line, so legacy-font symbols in
        the text after a heading survive.
    """
    keep = [i for i, ch in enumerate(text) if not CONTROL_OR_SPAN_NL_RE.match(ch)]
    probe = "".join(text[i] for i in keep).replace(NBSP, " ")
    label = EXCERPT_LABEL_RE.match(probe)
    offset = label.end() if label else 0
    body = probe[offset:]           # sliced, not ``match(probe, offset)``: ``^`` anchors at index 0 only
    for kind, pattern in _MARKER_PATTERNS:
        m = pattern.match(body)
        if not m:
            continue
        leaf = hebrew_numeral(m.group("leaf"))
        if leaf is None:
            continue
        rest = ""
        if m.groupdict().get("rest"):
            rest = text[keep[offset + m.start("rest")]:].strip(" \t" + NBSP)
        side = _SIDE_OF[m.group("side")]
        if kind == "daf" and EDITION_REF_REST_RE.match(m.groupdict().get("rest") or ""):
            return FolioMarker(label=f"ed{leaf}{side}", kind="edition_ref", rest=rest)
        return FolioMarker(label=f"{leaf}{side}", kind=kind, rest=rest)
    return None


def shelfmark_core_id(mark: Optional[str], heads: Sequence[str] = DEFAULT_INSTITUTION_HEADS) -> str:
    """Comparable id of a shelf mark: the merging normaliser's canonical id of its core.

    The institution head is split off first (:func:`strip_institution`) and a leading holder token
    (``JRL_``, ``RNL_``) is dropped from the id, so ``Manchester: A 444``, ``JRL A 444`` and
    ``A 444`` agree.

    :param mark: Shelf mark in any spelling, or None.
    :param heads: Known institution heads.
    :return: Lower-case id ("" for an empty mark).
    """
    if not mark:
        return ""
    core, _ = strip_institution(mark, heads)
    core = re.sub(r"\s+", " ", core.replace("_", " ")).strip()
    return HOLDER_ID_PREFIX_RE.sub("", ShelfmarkNormalizer.to_canonical_id(core)).lower()


def heading_shelfmark(rest: Optional[str]) -> Optional[Dict[str, Optional[str]]]:
    """Shelf mark (and side) named in a folio heading's trailing text.

    Joined fragments are sometimes transcribed in one file whose headings name the leaf actually
    transcribed (``דף א צד א ( 'A 444 1R) ...``). The first Latin run holding a digit
    (:data:`HEADING_LATIN_RUN_RE`) is read; a final ``<leaf><R|V>`` token is the side. The rest must
    be a shelf mark for the merging normaliser: its canonical id keeps a letter token (``fol. 3``
    gives ``3`` and is no shelf mark).

    :param rest: The heading's trailing text (``marker_rest``).
    :return: ``mark`` (as written), ``core_id`` (:func:`shelfmark_core_id`), ``folio`` (``1r`` or
        None); None when the text names no shelf mark.
    """
    probe = "".join(ch for ch in (rest or "") if not CONTROL_OR_SPAN_NL_RE.match(ch))
    run = HEADING_LATIN_RUN_RE.search(probe)
    if not run:
        return None
    text = run.group(0).strip(" .-")
    side = HEADING_SIDE_TOKEN_RE.match(text)
    mark = side.group("mark").strip(" .-") if side else text
    core_id = shelfmark_core_id(mark)
    tokens = core_id.split("_")
    if not any(t.isalpha() for t in tokens) or not any(ch.isdigit() for ch in core_id):
        return None
    folio = f"{int(side.group('leaf'))}{side.group('side').lower()}" if side else None
    return {"mark": mark, "core_id": core_id, "folio": folio}


def same_shelfmark(a: Optional[str], b: Optional[str]) -> bool:
    """Whether two spellings name one shelf mark (merging normaliser, or :func:`normalise_shelfmark`).

    :param a: One spelling.
    :param b: Another spelling.
    :return: True when either comparable form agrees (False when one is empty).
    """
    if not a or not b:
        return False
    if shelfmark_core_id(a) == shelfmark_core_id(b):
        return True
    core_a, core_b = strip_institution(a)[0], strip_institution(b)[0]
    return normalise_shelfmark(core_a) == normalise_shelfmark(core_b)


def _new_folio(label: Optional[str], kind: str, line: Optional[PdfLine], page_index: int) -> Dict[str, object]:
    """Empty folio record.

    :param label: Folio label (None for unlabelled text).
    :param kind: Marker kind, ``"page_fallback"`` or ``"preamble"``.
    :param line: The heading line (None for fallback and preamble folios).
    :param page_index: Page the folio starts on.
    :return: The record.
    """
    return {"folio": label, "page_index": page_index, "page_indices": [page_index], "marker_kind": kind,
            "marker_text": line.text if line else "", "lines": [], "spans": [], "spans_fonts": collections.Counter(),
            "notes_lines": 0, "flags": collections.Counter()}


def _add_line(folio: Dict[str, object], text: str, line: PdfLine) -> None:
    """Append a line (and its spans) to a folio.

    :param folio: Folio record.
    :param text: Line text to store (a heading's trailing text, or the whole line).
    :param line: The PDF line it came from (spans, page).
    """
    line_no = len(folio["lines"])
    folio["lines"].append(text)
    for font, span in line.spans:
        folio["spans"].append((font, span, line_no))
        folio["spans_fonts"][font] += len(LETTER_RE.findall(span))
    if line.page_index not in folio["page_indices"]:
        folio["page_indices"].append(line.page_index)


def side_index(label: Optional[str]) -> Optional[int]:
    """Position of a ``<leaf><r|v>`` label in the side sequence 1r, 1v, 2r, ... (``1r`` -> 2).

    :param label: Folio label.
    :return: ``2 * leaf + (1 for v)``, or None for another label (``p1``, None).
    """
    if not label or not FOLIO_LABEL_RE.fullmatch(label):
        return None
    return 2 * int(label[:-1]) + (label[-1] == "v")


def flag_sequence_and_outliers(folios: Sequence[Dict[str, object]],
                               factor: float = LINE_COUNT_OUTLIER_FACTOR, min_folios: int = 3) -> None:
    """Flag folios whose next heading skips a side, and folios with far more lines than usual.

    ``side_sequence_gap`` (value: sides skipped) goes on a folio when the next labelled folio is
    later than its successor side (``6r`` followed by ``7r``: no ``6v`` heading, so ``6r`` may hold
    both sides). ``line_count_outlier`` goes on a folio with more than ``factor`` times the
    document's median line count (at least ``min_folios`` folios with lines). Both are page-row
    exclusions (:data:`PAGE_ROW_EXCLUDING_FLAGS`). Flags are added in place.

    :param folios: Folio records in document order (``folio``, ``lines``, ``flags``).
    :param factor: Outlier factor over the median.
    :param min_folios: Fewest folios with lines for the outlier rule.
    """
    labelled = [f for f in folios if side_index(f["folio"]) is not None]
    for a, b in zip(labelled, labelled[1:]):
        skipped = side_index(b["folio"]) - side_index(a["folio"]) - 1
        if skipped > 0:
            a["flags"]["side_sequence_gap"] = skipped
    counts = [len(f["lines"]) for f in folios if f["lines"]]
    if len(counts) < min_folios:
        return
    median = statistics.median(counts)
    for f in folios:
        if median and len(f["lines"]) > factor * median:
            f["flags"]["line_count_outlier"] = 1


def split_document(doc: "fitz.Document", header_lines: int = 0,
                   min_preamble_text_lines: int = 3) -> Tuple[List[PdfLine], List[Dict[str, object]]]:
    """Cut a PDF's lines into folios at folio headings.

    With headings: lines before the first heading are the preamble (header, ``העתקות :`` label);
    when the preamble holds at least ``min_preamble_text_lines`` lines of 10+ letters it is kept
    as an unlabelled folio (flag ``text_before_first_marker``) rather than dropped. A heading's
    trailing text is kept as the folio's first line for :data:`TEXT_AFTER_HEADING_KINDS` headings and
    recorded as ``marker_rest`` otherwise (``edition_ref`` headings, a named leaf). An ``edition_ref`` heading
    (``דף פא עמוד ב (גמרא)``) starts a section labelled ``ed<page><r|v>``, outside the leaf checks
    (duplicates, order, sequence). From a ``הערות:`` line or an
    underscore rule or a sigla legend (``מקרא סימנים``) to the next heading every line is the editor's
    notes (counted in ``notes_lines``, not kept; lines so cut on a page with neither a heading nor a
    notes line are counted in the flag ``unmarked_page_cut``, a sign of a missed heading). Repeated
    labels get ``duplicate_folio_label``; a leaf number lower than the previous one gets
    ``folio_order_break``; :func:`flag_sequence_and_outliers` adds ``side_sequence_gap`` and
    ``line_count_outlier``.

    Without any heading: one folio per PDF page, labelled ``p<page>``, flag ``no_folio_markers``;
    the first ``header_lines`` lines of page 1 are the preamble; notes are cut per page.

    :param doc: Open PDF.
    :param header_lines: Lines of page 1 that :func:`parse_header` consumed.
    :param min_preamble_text_lines: Text lines that make a preamble a folio.
    :return: (preamble lines, folios). Each folio: ``folio``, ``page_index``, ``page_indices``,
        ``marker_kind``, ``marker_text``, ``lines`` (str), ``spans`` ((font, text, line number)),
        ``spans_fonts`` (Counter font -> letters), ``notes_lines``, ``flags`` (Counter).
    """
    lines = document_lines(doc)
    markers = [match_folio_marker(ln.text) for ln in lines]
    folios: List[Dict[str, object]] = []
    preamble: List[PdfLine] = []
    if not any(markers):
        preamble = lines[:header_lines]
        current: Optional[Dict[str, object]] = None
        notes = False
        for ln in lines[header_lines:]:
            if current is None or ln.page_index != current["page_index"]:
                current = _new_folio(f"p{ln.page_index + 1}", "page_fallback", None, ln.page_index)
                current["flags"]["no_folio_markers"] = 1
                folios.append(current)
                notes = False
            if NOTES_START_RE.match(ln.text):
                notes = True
            if notes:
                current["notes_lines"] += 1
                continue
            _add_line(current, ln.text, ln)
        return preamble, folios
    current = None
    notes = False
    seen: Dict[str, int] = {}
    last_leaf = 0
    marked_pages = {ln.page_index for ln, mk in zip(lines, markers) if mk is not None}
    marked_pages |= {ln.page_index for ln in lines if NOTES_START_RE.match(ln.text)}
    for ln, mk in zip(lines, markers):
        if mk is not None:
            current = _new_folio(mk.label, mk.kind, ln, ln.page_index)
            if mk.kind != "edition_ref":
                leaf = int(re.match(r"\d+", mk.label).group(0))
                if mk.label in seen:
                    current["flags"]["duplicate_folio_label"] = 1
                if leaf < last_leaf:
                    current["flags"]["folio_order_break"] = 1
                seen[mk.label] = seen.get(mk.label, 0) + 1
                last_leaf = leaf
            if mk.kind in TEXT_AFTER_HEADING_KINDS:
                _add_line(current, mk.rest, ln)
            elif mk.rest:
                current["marker_rest"] = mk.rest
            folios.append(current)
            notes = False
            continue
        if current is None:
            preamble.append(ln)
            continue
        if NOTES_START_RE.match(ln.text):
            notes = True
        if notes:
            current["notes_lines"] += 1
            if ln.page_index not in marked_pages:
                current["flags"]["unmarked_page_cut"] += 1
            continue
        _add_line(current, ln.text, ln)
    text_lines = [ln for ln in preamble if len(LETTER_RE.findall(ln.text)) >= 10]
    if len(text_lines) >= min_preamble_text_lines:
        pre = _new_folio(None, "preamble", None, preamble[0].page_index)
        pre["flags"]["text_before_first_marker"] = 1
        for ln in preamble:
            _add_line(pre, ln.text, ln)
        folios.insert(0, pre)
        preamble = []
    flag_sequence_and_outliers(folios)
    return preamble, folios


def split_folios(doc: "fitz.Document", header_lines: int = 0) -> List[Dict[str, object]]:
    """Folios of a PDF (see :func:`split_document`, whose preamble this drops).

    :param doc: Open PDF.
    :param header_lines: Lines of page 1 that :func:`parse_header` consumed.
    :return: Folio records.
    """
    return split_document(doc, header_lines)[1]


# ───────────────────────────── fonts and finals ─────────────────────────────
def visible_symbol(ch: str) -> str:
    """Printable name of a symbol (control characters as ``⟨0B⟩``, an in-span newline as ``⟨0A⟩``).

    :param ch: One character.
    :return: Its printable form.
    """
    if ch == SPAN_NEWLINE:
        return "⟨0A⟩"
    if ord(ch) < 0x20 or ord(ch) == 0x7F:
        return f"⟨{ord(ch):02X}⟩"
    return ch


def font_check(spans: Sequence[Tuple[str, str, int]]) -> Dict[str, object]:
    """Which fonts carry the letters, and which carry symbols standing for final letters.

    A symbol (:data:`FINAL_SYMBOLS`, a control character or :data:`SPAN_NEWLINE`) counts as a
    final-letter candidate when it stands right after a Hebrew letter and before a non-letter: the
    word-final slot. Neighbours are read across span boundaries within a line, so a symbol drawn
    in a second font is attributed to that font. ``$`` and ``&`` are also transcription marks
    (uncertainty, note mark): the candidates are only evidence of missing finals when the text
    also lacks real ones (``finals_missing``).

    :param spans: ``(font, text, line number)`` in reading order.
    :return: ``hebrew_by_font``, ``arabic_by_font``, ``real_finals_by_font``,
        ``symbol_finals_by_font`` (font -> symbol -> count), ``letter_fonts`` (fonts by letters,
        most first), ``has_symbol_finals``, ``hebrew_words``, ``real_finals``, ``finals_missing``.
    """
    by_line: Dict[int, List[Tuple[str, str]]] = collections.defaultdict(list)
    for font, text, line_no in spans:
        by_line[line_no].append((font, text))
    hebrew: Counter = collections.Counter()
    arabic: Counter = collections.Counter()
    finals: Counter = collections.Counter()
    symbols: Dict[str, Counter] = collections.defaultdict(collections.Counter)
    texts = []
    for line_no in sorted(by_line):
        chars: List[str] = []
        fonts: List[str] = []
        for font, text in by_line[line_no]:
            chars.extend(text)
            fonts.extend([font] * len(text))
        texts.append("".join(chars))
        for i, ch in enumerate(chars):
            font = fonts[i]
            if HEB_LETTER_RE.match(ch):
                hebrew[font] += 1
                if ch in L.FINALS:
                    finals[font] += 1
            elif ARABIC_LETTER_RE.match(ch):
                arabic[font] += 1
            elif SYMBOL_CHAR_RE.match(ch):
                prev = chars[i - 1] if i else ""
                nxt = chars[i + 1] if i + 1 < len(chars) else ""
                if HEB_LETTER_RE.match(prev) and not HEB_LETTER_RE.match(nxt):
                    symbols[font][visible_symbol(ch)] += 1
    letters = hebrew + arabic
    joined = "\n".join(texts)
    words = len(L.HEB_RUN_RE.findall(joined))
    return {
        "hebrew_by_font": dict(hebrew), "arabic_by_font": dict(arabic), "real_finals_by_font": dict(finals),
        "symbol_finals_by_font": {f: dict(c) for f, c in symbols.items()},
        "letter_fonts": [f for f, _ in letters.most_common()], "has_symbol_finals": bool(symbols),
        "hebrew_words": words, "real_finals": sum(finals.values()), "finals_missing": L.is_finals_mojibake(joined),
    }


def infer_document_finals(text: str, lexicon: Mapping[str, int]) -> Dict[str, str]:
    """Mojibake symbol -> final letter for a text whose final letters are missing.

    Runs :func:`legacy_store_sigla.infer_finals_mapping` only when
    :func:`legacy_store_sigla.is_finals_mojibake` says the finals are missing. Word-final ``!`` and
    ``,`` (seen as finals in legacy fonts, but outside the committed inference's candidate class)
    take part through private-use proxies and are mapped back.

    :param text: Document (or folio) text.
    :param lexicon: Word -> count.
    :return: Symbol -> final letter (empty when the finals are present or nothing is supported).
    """
    if not lexicon or not L.is_finals_mojibake(text):
        return {}
    probe = text
    for sym, proxy in _INFERENCE_PROXIES.items():
        probe = re.sub(rf"(?<=[{L.HEB}]){re.escape(sym)}(?![{L.HEB}])", proxy, probe)
    back = {proxy: sym for sym, proxy in _INFERENCE_PROXIES.items()}
    return {back.get(sym, sym): letter for sym, letter in L.infer_finals_mapping(probe, lexicon).items()}


# ───────────────────────────── cleaning ─────────────────────────────
def structure_lines(lines: Sequence[str], dense_share: float = 0.6, min_numbered: int = 3) -> Tuple[List[str], Counter]:
    """Strip per-line numbers and re-join lines the PDF wrapped.

    A leading number (``12``, ``12.``) before text is a line number and is removed. When the
    numbering is dense (at least ``min_numbered`` numbered lines, ``dense_share`` of consecutive
    numbered lines stepping by one), every numbered line starts a manuscript line, so an unnumbered
    line with letters after a numbered one is the PDF's wrap of it and is joined to it. Every-fifth
    numbering joins nothing. Lines without letters are kept but never joined.

    :param lines: Raw folio lines.
    :param dense_share: Share of +1 steps that makes numbering dense.
    :param min_numbered: Fewest numbered lines for the join rule.
    :return: (lines, flags ``line_numbers_stripped`` and ``wrapped_lines_joined``).
    """
    flags: Counter = collections.Counter()
    numbers: List[Optional[int]] = []
    stripped: List[str] = []
    for line in lines:
        m = LINE_NUMBER_RE.match(line)
        if m:
            numbers.append(int(m.group(1)))
            stripped.append(line[m.end():])
            flags["line_numbers_stripped"] += 1
        else:
            numbers.append(None)
            stripped.append(line)
    numbered = [n for n in numbers if n is not None]
    steps = sum(1 for a, b in zip(numbered, numbered[1:]) if b == a + 1)
    dense = len(numbered) >= min_numbered and steps >= dense_share * (len(numbered) - 1)
    if not dense:
        return stripped, flags
    out: List[str] = []
    open_numbered = False
    for line, n in zip(stripped, numbers):
        if n is not None:
            out.append(line)
            open_numbered = True
        elif open_numbered and LETTER_RE.search(line):
            # spaces only: str.strip() would also eat \x0b and \x1c-\x1f, which may be finals
            out[-1] = out[-1].rstrip(" \t") + " " + line.strip(" \t")
            flags["wrapped_lines_joined"] += 1
        else:
            out.append(line)
    return out, flags


def reattach_geresh(text: str, min_moves: int = 2) -> Tuple[str, int]:
    """Put a geresh the PDF text layer displaced back at the end of its word (``אמ 'לו`` -> ``אמ' לו``).

    In some downloads (all three header PDFs) the right-to-left export writes an abbreviation's
    geresh before the NEXT word; in others it stays in place. The text decides: the geresh is moved
    only when the displaced form (letter, space, geresh, letter) outnumbers the regular one
    (letter, geresh, space, letter) and occurs at least ``min_moves`` times.

    :param text: Folio text.
    :param min_moves: Fewest displaced gereshes that trigger the repair.
    :return: (text, gereshes moved).
    """
    lead = len(LEAD_GERESH_RE.findall(text))
    if lead < min_moves or lead <= len(TRAIL_GERESH_RE.findall(text)):
        return text, 0
    return LEAD_GERESH_RE.subn(lambda m: m.group(1) + " ", text)


def quote_separator_share(text: str) -> float:
    """Share of Hebrew words followed by a double quote that is not an inner gershayim.

    Abbreviations put gershayim INSIDE a word (``ב"ד``); a text layer whose words end in ``"`` uses
    the quote as a word separator, a sign of an export that scrambled word order (one document seen,
    40 % of its words, against at most 0.4 % elsewhere).

    :param text: Raw text.
    :return: Share in [0, 1] (0 for a text without Hebrew words).
    """
    words = len(L.HEB_RUN_RE.findall(text))
    return len(WORD_FINAL_QUOTE_RE.findall(text)) / words if words else 0.0


def clean_folio(lines: Sequence[str], lexicon: Optional[Mapping[str, int]],
                finals_mapping: Optional[Mapping[str, str]] = None) -> Dict[str, object]:
    """Clean one folio's lines into a visible-ink target, keeping the line structure.

    One :func:`legacy_store_sigla.clean_section` call on the folio text joined with ``\\n``
    (per-folio, not per-line: family, bracket orientation, whole-text reversal and finals detection
    are decided on the whole side, brackets may span lines, and the cleaner keeps line breaks,
    dropping only lines left without letters). Before it: :func:`structure_lines`, removal of the
    editor's in-line labels (:data:`EDITOR_PHRASE_RE`: ``קריאה מסופקת``, the catchword label
    ``שומר הדף:``) and :func:`reattach_geresh`. After it, cleaned lines that read like the editor's
    comments (:data:`EDITOR_NOTE_RE`: the margin, the photograph, proof marks, the physical page) are
    counted in the flag ``editor_note_suspect`` (kept, but a page-row exclusion; see also
    :func:`flag_editor_note_document`). When ``finals_mapping`` is None
    the mapping is inferred from the folio itself (:func:`infer_document_finals`); :func:`ingest`
    passes the document's.

    :param lines: Raw folio lines.
    :param lexicon: Word -> count for finals inference and visual-order repairs (None: no repairs).
    :param finals_mapping: Symbol -> final letter, or None to infer.
    :return: ``cleaned_lines``, ``letters``, ``hebrew_letters``, ``arabic_letters``, ``gaps``,
        ``finals_repaired``, ``uncertain_letters``, ``recon_letters``, ``fragments_repaired``,
        ``family``, ``finals_mapping`` (printable symbols), ``dropped_spans``, ``flags`` (dict).
    """
    work, flags = structure_lines(lines)
    joined = "\n".join(work)
    joined, n_phrase = EDITOR_PHRASE_RE.subn(" ", joined)
    if n_phrase:
        flags["editor_phrase_removed"] += n_phrase
    joined, n_geresh = reattach_geresh(joined)
    if n_geresh:
        flags["geresh_reattached"] += n_geresh
    mapping = dict(finals_mapping) if finals_mapping is not None else infer_document_finals(joined, lexicon or {})
    sides = L.clean_section(joined, mapping, lexicon=lexicon) if LETTER_RE.search(joined) else []
    if len(sides) > 1:
        flags["inner_side_split"] += len(sides)
    for s in sides:
        flags.update(s.flags)
    if SPAN_NEWLINE not in mapping:
        flags["span_newline_unmapped"] += len(re.findall(rf"(?<=[{L.HEB}]){SPAN_NEWLINE}(?![{L.HEB}])", joined))
    if sides:
        invented, lost = L.letter_audit(joined, sides, mapping)
        if invented:
            flags["audit_invented_letters"] += sum(invented.values())
        if lost:
            flags["audit_lost_letters"] += sum(lost.values())
    cleaned = "\n".join(s.text for s in sides)
    cleaned_lines = [ln.strip() for ln in cleaned.split("\n") if ln.strip()]
    notes = sum(1 for ln in cleaned_lines if EDITOR_NOTE_RE.search(ln))
    if notes:
        flags["editor_note_suspect"] += notes
    text = "\n".join(cleaned_lines)
    hebrew = len(HEB_LETTER_RE.findall(text))
    arabic = len(ARABIC_LETTER_RE.findall(text))
    return {
        "cleaned_lines": cleaned_lines, "letters": hebrew + arabic, "hebrew_letters": hebrew, "arabic_letters": arabic,
        "gaps": text.count(L.GAP), "finals_repaired": sum(s.finals_repaired for s in sides),
        "uncertain_letters": sum(s.uncertain_letters for s in sides),
        "recon_letters": sum(s.recon_letters for s in sides),
        "fragments_repaired": sum(s.fragments_repaired for s in sides),
        "family": sides[0].family if sides else "empty",
        "finals_mapping": {visible_symbol(k): v for k, v in mapping.items()},
        "dropped_spans": [list(d) for s in sides for d in s.dropped_spans],
        "flags": {k: v for k, v in flags.items() if v},
    }


def flag_editor_note_document(records: Sequence[Dict[str, object]],
                              share: float = EDITOR_NOTE_DOCUMENT_SHARE) -> bool:
    """Flag every folio of a document whose folios often carry the editor's notes.

    When at least ``share`` of the records carry ``editor_note_suspect``, the transcriber writes
    notes into the text throughout, so the folios without a matched note are suspect too: every
    record gets ``editor_note_document`` (a page-row exclusion). Flags are added in place.

    :param records: Folio or side records of ONE document (``flags``).
    :param share: Smallest share of flagged records that flags the document.
    :return: Whether the document was flagged.
    """
    if not records:
        return False
    flagged = sum(1 for r in records if r["flags"].get("editor_note_suspect"))
    if not flagged or flagged < share * len(records):
        return False
    for r in records:
        r["flags"]["editor_note_document"] = 1
    return True


# ───────────────────────────── duplicates ─────────────────────────────
def text_letters(records: Sequence[Mapping[str, object]]) -> str:
    """Letters of a transcription's cleaned lines, in record order (the near-duplicate input).

    :param records: Folio or side records of one text.
    :return: Hebrew and Arabic letters.
    """
    return "".join("".join(LETTER_RE.findall("\n".join(r["cleaned_lines"]))) for r in records)


def side_content_key(record: Mapping[str, object]) -> str:
    """Order-insensitive key of a side's raw transcription: equal for re-downloads of one side.

    A re-download keeps the transcriber's letters and line count, but the text layer may come out
    in another order, so the text digest differs. The key is the SHA-1 of the sorted Hebrew letters
    of the raw lines plus the raw line count.

    :param record: Side record (``raw_lines``).
    :return: 40 hex digits.
    """
    lines = list(record["raw_lines"])
    letters = "".join(sorted(HEB_LETTER_RE.findall("\n".join(lines))))
    return hashlib.sha1(f"{letters}|{len(lines)}".encode("utf-8")).hexdigest()


def near_duplicate_pairs(letter_texts: Sequence[str], ratio: float = NEAR_DUPLICATE_RATIO,
                         min_letters: int = NEAR_DUPLICATE_MIN_LETTERS,
                         min_length_ratio: float = NEAR_DUPLICATE_MIN_LENGTH_RATIO) -> List[Tuple[int, int, float]]:
    """Pairs of texts whose letters are near-identical (``difflib.SequenceMatcher`` ratio).

    Texts shorter than ``min_letters`` letters or whose lengths differ by more than
    ``min_length_ratio`` are not compared. ``quick_ratio`` (an upper bound of the ratio) screens
    the pairs first, so the result equals the full comparison.

    :param letter_texts: Letters of each text (:func:`text_letters`), in priority order (an
        earlier text is the original of a later one).
    :param ratio: Smallest similarity of a near-duplicate.
    :param min_letters: Fewest letters a compared text has.
    :param min_length_ratio: Smallest shorter/longer length ratio of a compared pair.
    :return: ``(earlier index, later index, similarity)`` in index order.
    """
    out: List[Tuple[int, int, float]] = []
    for i, a in enumerate(letter_texts):
        for j in range(i + 1, len(letter_texts)):
            b = letter_texts[j]
            short, long_ = sorted((len(a), len(b)))
            if short < min_letters or short / long_ < min_length_ratio:
                continue
            sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
            if sm.quick_ratio() < ratio:
                continue
            value = sm.ratio()
            if value >= ratio:
                out.append((i, j, value))
    return out


# ───────────────────────────── images ─────────────────────────────
def image_index(raw_dir: str) -> Dict[str, List[str]]:
    """Image files of a folder keyed by name stem without the Firefox ``(n)`` copy suffix.

    :param raw_dir: Folder.
    :return: Stem (``Yevr_II_A_223_1r``) -> paths, sorted by name.
    """
    out: Dict[str, List[str]] = collections.defaultdict(list)
    for name in sorted(os.listdir(raw_dir)):
        m = _COPY_NAME_RE.match(name)
        if m:
            out[m.group("stem")].append(os.path.join(raw_dir, name))
    return dict(out)


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, read in chunks.

    :param path: File.
    :param chunk: Read size.
    :return: 64 hex digits.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def pair_images(series_key: Optional[Tuple[str, str]], folio: Optional[str], raw_dir: str,
                index: Optional[Mapping[str, List[str]]] = None) -> Optional[Dict[str, object]]:
    """The image file(s) named for a folio: ``<series>_<number>_<folio>.jpg`` and its ``(n)`` copies.

    Copies are deduplicated by SHA-256; of the distinct files the largest is chosen (ties: the
    name without ``(n)``, then the shorter name). Only folio labels ``<leaf><r|v>`` are looked up
    (page-fallback labels never pair).

    :param series_key: ``(series, number)`` from :func:`series_key_for`, or None.
    :param folio: Folio label.
    :param raw_dir: Folder holding the images.
    :param index: :func:`image_index` of ``raw_dir`` (built when None).
    :return: ``image_candidate_path``, ``image_sha256``, ``image_size`` (bytes), ``image_files`` (all
        copies), ``distinct_images``, ``image_match`` (always ``"file_name_token_unverified"``: the
        token is not known to be a folio), ``image_verified`` (False); None when no file has that
        name.
    """
    if not series_key or not folio or not FOLIO_LABEL_RE.fullmatch(folio):
        return None
    files = (index if index is not None else image_index(raw_dir)).get(f"{series_key[0]}_{series_key[1]}_{folio}", [])
    if not files:
        return None
    variants: Dict[str, Tuple[int, str]] = {}
    for path in files:
        sha = file_sha256(path)
        size = os.path.getsize(path)
        is_copy = bool(_COPY_NAME_RE.match(os.path.basename(path)).group("n"))
        rank = (size, not is_copy, -len(os.path.basename(path)))
        if sha not in variants or rank > variants[sha][0]:
            variants[sha] = (rank, path)
    best_sha, (best_rank, best_path) = max(variants.items(), key=lambda kv: kv[1][0])
    return {"image_candidate_path": best_path, "image_sha256": best_sha, "image_size": best_rank[0],
            "image_files": list(files), "distinct_images": len(variants), "image_match": "file_name_token_unverified",
            "image_verified": False}


# ───────────────────────────── one PDF ─────────────────────────────
def ingest(pdf_path: str, raw_dir: Optional[str], lexicon: Optional[Mapping[str, int]],
           index: Optional[Mapping[str, List[str]]] = None, shelfmark_hint: Optional[str] = None,
           finals_mapping: Optional[Mapping[str, str]] = None) -> List[Dict[str, object]]:
    """Folio records of one transcription PDF.

    Shelf mark: the page-1 header when there is one, else ``shelfmark_hint`` or the
    ``transcription_manual_`` file name (a mismatch between header and file name is flagged).
    Finals: one mapping for the whole document (more evidence than a folio), inferred unless given.
    Document flags copied onto every folio: ``headerless``, ``excerpt_document``,
    ``shelfmark_unresolved``, ``series_unknown``, ``shelfmark_header_file_mismatch``,
    ``word_quote_separators`` (:func:`quote_separator_share` above 0.1: scrambled export). A heading
    naming another shelf mark than the document's (:func:`heading_shelfmark`, a joined fragment
    filed under one mark) gives that folio ``heading_other_shelfmark``; the named mark and side are
    kept in ``heading_shelfmark`` for re-attribution. :func:`flag_editor_note_document` runs over
    the document's folios.

    :param pdf_path: The PDF.
    :param raw_dir: Folder searched for candidate images (None: no image lookup).
    :param lexicon: Word -> count (see :func:`clean_folio`).
    :param index: :func:`image_index` of ``raw_dir`` (built once when None and ``raw_dir`` is set).
    :param shelfmark_hint: Shelf mark from another copy's file name (used when the header has none).
    :param finals_mapping: Document finals mapping, or None to infer.
    :return: One dict per folio (empty for a junk PDF): ``pdf_file``, ``shelfmark``,
        ``shelfmark_source``, ``series_key``, ``institution``, ``title``, ``folio``, ``page_index``,
        ``page_indices``, ``marker_kind``, ``marker_rest``, ``raw_lines``, ``notes_lines``,
        the :func:`clean_folio` fields, ``fonts`` (font -> letters), ``font_check``, ``flags``,
        ``image_candidate_path`` (unverified, never a row image), ``image_sha256``, ``image_size``,
        ``image_files``, ``distinct_images``, ``image_match``, ``image_verified``, ``style``,
        ``heading_shelfmark`` (:func:`heading_shelfmark` of ``marker_rest``, or None).
    """
    with fitz.open(pdf_path) as doc:
        if is_junk(doc):
            return []
        page1 = [ln.text for ln in page_lines(doc[0], 0)] if len(doc) else []
        header = parse_header(page1)
        preamble, folios = split_document(doc, header["header_lines"] if header else 0)
    doc_flags: Counter = collections.Counter()
    if any(EXCERPT_LABEL_RE.match(line) for line in page1[:3]):
        doc_flags["excerpt_document"] = 1
    file_mark = shelfmark_hint or shelfmark_from_filename(pdf_path)
    if header:
        shelfmark, source, key = header["shelfmark"], "header", header["series_key"]
        if file_mark and normalise_shelfmark(file_mark) != normalise_shelfmark(header["shelfmark"]):
            doc_flags["shelfmark_header_file_mismatch"] = 1
    else:
        doc_flags["headerless"] = 1
        shelfmark, source, key = file_mark, ("file_name" if file_mark else None), series_key_for(file_mark)
    if shelfmark is None:
        doc_flags["shelfmark_unresolved"] = 1
    elif key is None:
        doc_flags["series_unknown"] = 1
    doc_text = "\n".join(line for f in folios for line in f["lines"])
    if len(L.HEB_RUN_RE.findall(doc_text)) >= 20 and quote_separator_share(doc_text) > 0.1:
        doc_flags["word_quote_separators"] = 1
    mapping = dict(finals_mapping) if finals_mapping is not None else infer_document_finals(doc_text, lexicon or {})
    if index is None:
        index = image_index(raw_dir) if raw_dir else {}
    records = []
    for f in folios:
        cleaned = clean_folio(f["lines"], lexicon, mapping)
        fc = font_check(f["spans"])
        img = pair_images(key, f["folio"], raw_dir, index) or {}
        flags: Counter = collections.Counter(doc_flags)
        flags.update(f["flags"])
        flags.update(cleaned.pop("flags"))
        if fc["has_symbol_finals"]:
            flags["symbol_finals"] += 1
        if f["notes_lines"]:
            flags["notes_cut"] += 1
        named = heading_shelfmark(f.get("marker_rest"))
        if named and shelfmark and not same_shelfmark(named["mark"], str(shelfmark)):
            flags["heading_other_shelfmark"] = 1
        records.append({
            "pdf_file": os.path.basename(pdf_path), "shelfmark": shelfmark, "shelfmark_source": source,
            "series_key": list(key) if key else None, "institution": header["institution"] if header else None,
            "title": header["title"] if header else None, "folio": f["folio"], "page_index": f["page_index"],
            "page_indices": f["page_indices"], "marker_kind": f["marker_kind"], "marker_rest": f.get("marker_rest", ""),
            "raw_lines": list(f["lines"]), "notes_lines": f["notes_lines"], "preamble_lines": len(preamble),
            **cleaned, "fonts": dict(f["spans_fonts"]), "font_check": fc, "flags": dict(flags),
            "image_candidate_path": img.get("image_candidate_path"), "image_sha256": img.get("image_sha256"),
            "image_size": img.get("image_size"), "image_files": img.get("image_files", []),
            "distinct_images": img.get("distinct_images", 0), "image_match": img.get("image_match"),
            "image_verified": False, "style": "header_folio", "heading_shelfmark": named,
        })
    flag_editor_note_document(records)
    return records


# ───────────────────────────── styles ─────────────────────────────
def detect_style(doc: "fitz.Document", file_name: str, side_file_evidence: bool = False) -> str:
    """Which of the two PDF styles a transcription uses.

    ``header_folio`` (new downloads): a page-1 header, or folio headings: any strong heading
    (:data:`STRONG_MARKER_KINDS`), two or more number-and-side headings, or one in an excerpt list
    (first line ``העתקות`` / ``אתחלתא``). ``side_per_file`` (old downloads): letters but no such
    headings; a ``transcription_manual_`` name with a side suffix (``_2`` ... ``_9``) keeps a file
    with only weak headings in this style (one side per file), and with ``side_file_evidence`` (a
    sibling file named by the shorter stem) even a file with strong headings (edition page
    references such as ``דף פא עמוד ב (גמרא)`` are no leaves). ``unknown``: no letters in the text
    layer.

    :param doc: Open PDF.
    :param file_name: The PDF's file name.
    :param side_file_evidence: Whether the name's side suffix is confirmed by a sibling file
        (:func:`parse_side_file_name` resolution ``sibling_file``).
    :return: ``"header_folio"``, ``"side_per_file"`` or ``"unknown"``.
    """
    lines = document_lines(doc)
    if not any(LETTER_RE.search(ln.text) for ln in lines):
        return "unknown"
    page1 = [ln.text for ln in lines if ln.page_index == 0]
    if parse_header(page1):
        return "header_folio"
    side_named = bool(re.search(rf"_[2-{MAX_SIDE_SUFFIX}](?:\(\d+\)|-\d+)?\.pdf$", os.path.basename(file_name)))
    if side_named and side_file_evidence:
        return "side_per_file"
    kinds = [mk.kind for mk in (match_folio_marker(ln.text) for ln in lines) if mk]
    if any(k in STRONG_MARKER_KINDS for k in kinds):
        return "header_folio"
    excerpt = any(EXCERPT_LABEL_RE.match(t) for t in page1[:3])
    if kinds and not side_named and (len(kinds) >= 2 or excerpt):
        return "header_folio"
    return "side_per_file"


def split_side_columns(doc: "fitz.Document") -> Dict[str, object]:
    """Cut an old-style side file into its columns.

    Lines are rebuilt as for the new style (:func:`document_lines`, every page). A column marker
    (:data:`COLUMN_MARKER_RE`, ``טור_א`` / ``טור_ב``) opens a column of the SAME side (text after
    the marker on its line is kept); lines before the first marker form an unlabelled column (flag
    ``text_before_column_marker`` when such a marker follows). Editor's notes (:data:`NOTES_START_RE`)
    are cut up to the next column marker or edition page reference. Edition page references
    (``edition_ref`` headings, ``דף פא עמוד ב (גמרא)``) are the editor's, not the manuscript's: the
    line is dropped and counted in the flag ``edition_ref_heading``.

    :param doc: Open PDF.
    :return: ``columns`` (records like :func:`split_document` folios, ``folio`` = the column letter
        or None), ``column_markers`` (count), ``flags`` (Counter: ``no_column_markers``,
        ``text_before_column_marker``, ``duplicate_column_label``, ``edition_ref_heading``).
    """
    lines = document_lines(doc)
    columns: List[Dict[str, object]] = []
    flags: Counter = collections.Counter()
    current: Optional[Dict[str, object]] = None
    notes = False
    markers = 0
    for ln in lines:
        m = COLUMN_MARKER_RE.match("".join(ch for ch in ln.text if not CONTROL_OR_SPAN_NL_RE.match(ch)))
        if m:
            markers += 1
            if any(c["folio"] == m.group("col") for c in columns):
                flags["duplicate_column_label"] += 1
            current = _new_folio(m.group("col"), "column", ln, ln.page_index)
            columns.append(current)
            notes = False
            if m.group("rest").strip():
                _add_line(current, m.group("rest").strip(), ln)
            continue
        heading = match_folio_marker(ln.text)
        if heading is not None and heading.kind == "edition_ref":
            flags["edition_ref_heading"] += 1
            notes = False                      # a heading ends the notes, as in :func:`split_document`
            continue
        if current is None:
            current = _new_folio(None, "no_column", None, ln.page_index)
            columns.append(current)
        if NOTES_START_RE.match(ln.text):
            notes = True
        if notes:
            current["notes_lines"] += 1
            continue
        _add_line(current, ln.text, ln)
    if not markers:
        flags["no_column_markers"] = 1
    elif columns and columns[0]["folio"] is None:
        if any(LETTER_RE.search(t) for t in columns[0]["lines"]):
            flags["text_before_column_marker"] = 1
        else:
            columns.pop(0)
    return {"columns": columns, "column_markers": markers, "flags": flags}


def ingest_side_files(items: Sequence[Mapping[str, object]], lexicon: Optional[Mapping[str, int]],
                      finals_mapping: Optional[Mapping[str, str]] = None) -> List[Dict[str, object]]:
    """Side records of ONE shelf mark's old-style files, cleaned with one finals mapping.

    The finals mapping is inferred once from all sides together (more evidence than one side).
    Each column is cleaned on its own (:func:`clean_folio`; brackets do not span columns) and the
    side's ``cleaned_lines`` are the columns in order, with ``columns`` giving each column's
    ``[start, end)`` range in them. A side whose raw text uses the double quote as a word
    separator (:func:`quote_separator_share` above 0.1 on 20+ words: scrambled export) is flagged
    ``word_quote_separators``, as :func:`ingest` flags a document.

    :param items: One dict per side file: ``path``, ``shelfmark``, ``side_index``, and any extra
        keys (``side_resolution``, ``side_ambiguous``, ``copy_suffix``, ``canonical_id``, ...),
        which are copied onto the record.
    :param lexicon: Word -> count (see :func:`clean_folio`).
    :param finals_mapping: Shelf-mark finals mapping, or None to infer.
    :return: One record per non-junk side file, ordered by side index: ``pdf_file``, ``style``
        (``side_per_file``), ``shelfmark``, ``shelfmark_source`` (``file_name``), ``side_index``,
        ``folio`` (``s<side>``), ``columns`` (``column``, ``start``, ``end``, ``raw_lines``,
        ``letters``), ``column_markers``, ``raw_lines``, ``notes_lines``, the :func:`clean_folio`
        fields, ``fonts``, ``font_check``, ``flags``, the item's extra keys.
    """
    parsed = []
    for item in items:
        with fitz.open(str(item["path"])) as doc:
            if is_junk(doc):
                continue
            parsed.append((item, split_side_columns(doc)))
    doc_text = "\n".join(line for _, sp in parsed for col in sp["columns"] for line in col["lines"])
    mapping = dict(finals_mapping) if finals_mapping is not None else infer_document_finals(doc_text, lexicon or {})
    records = []
    for item, sp in parsed:
        flags: Counter = collections.Counter(sp["flags"])
        cleaned_lines: List[str] = []
        columns = []
        spans: List[Tuple[str, str, int]] = []
        fonts: Counter = collections.Counter()
        raw_lines: List[str] = []
        totals: Counter = collections.Counter()
        dropped: List[List[str]] = []
        families = []
        for col in sp["columns"]:
            cleaned = clean_folio(col["lines"], lexicon, mapping)
            flags.update(cleaned.pop("flags"))
            start = len(cleaned_lines)
            cleaned_lines.extend(cleaned["cleaned_lines"])
            columns.append({"column": col["folio"], "start": start, "end": len(cleaned_lines),
                            "raw_lines": list(col["lines"]), "letters": cleaned["letters"]})
            offset = len(raw_lines)
            spans.extend((font, text, offset + n) for font, text, n in col["spans"])
            raw_lines.extend(col["lines"])
            fonts.update(col["spans_fonts"])
            for key in ("letters", "hebrew_letters", "arabic_letters", "gaps", "finals_repaired", "uncertain_letters",
                        "recon_letters", "fragments_repaired"):
                totals[key] += cleaned[key]
            dropped.extend(cleaned["dropped_spans"])
            families.append(cleaned["family"])
            totals["notes_lines"] += col["notes_lines"]
        fc = font_check(spans)
        if fc["has_symbol_finals"]:
            flags["symbol_finals"] += 1
        if totals["notes_lines"]:
            flags["notes_cut"] += 1
        raw_text = "\n".join(raw_lines)
        if len(L.HEB_RUN_RE.findall(raw_text)) >= 20 and quote_separator_share(raw_text) > 0.1:
            flags["word_quote_separators"] = 1
        extra = {k: v for k, v in item.items() if k not in ("path", "shelfmark", "side_index")}
        records.append({
            "pdf_file": os.path.basename(str(item["path"])), "style": "side_per_file",
            "shelfmark": item["shelfmark"], "shelfmark_source": "file_name" if item["shelfmark"] else None,
            "side_index": item["side_index"], "folio": f"s{item['side_index']}", "columns": columns,
            "column_markers": sp["column_markers"], "raw_lines": raw_lines, "notes_lines": totals["notes_lines"],
            "cleaned_lines": cleaned_lines, **{k: totals[k] for k in ("letters", "hebrew_letters", "arabic_letters",
                                                                       "gaps", "finals_repaired", "uncertain_letters",
                                                                       "recon_letters", "fragments_repaired")},
            "family": collections.Counter(families).most_common(1)[0][0] if families else "empty",
            "finals_mapping": {visible_symbol(k): v for k, v in mapping.items()}, "dropped_spans": dropped,
            "fonts": dict(fonts), "font_check": fc, "flags": dict(flags), **extra,
        })
    return sorted(records, key=lambda r: (r["side_index"], r["pdf_file"]))


def merge_side_redownloads(records: Sequence[Dict[str, object]]
                           ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Fold re-downloads of a side into the side they copy (one shelf mark's side records).

    A file saved again later takes the next free ``_<n>`` suffix, so its suffix is no side. A record
    whose :func:`side_content_key` equals that of a record with a lower (or the same) side index,
    taken in ``(side index, file name)`` order, is a copy of it: it is no separate side or unit.
    The original keeps the copy in ``side_redownloads`` (``pdf_file``, ``digest``,
    ``suffix_side_index``, ``flag = "side_redownload"``) and counts it in the flag
    ``side_redownload_copies`` (not a page-row exclusion).

    :param records: :func:`ingest_side_files` records of ONE shelf mark.
    :return: (kept records in side order, copies: ``pdf_file``, ``digest``, ``suffix_side_index``,
        ``copy_of_pdf_file``, ``copy_of_digest``, ``copy_of_suffix_side_index``, ``flag``).
    """
    kept: List[Dict[str, object]] = []
    copies: List[Dict[str, object]] = []
    first: Dict[str, Dict[str, object]] = {}
    for r in sorted(records, key=lambda x: (int(x["side_index"]), str(x["pdf_file"]))):
        key = side_content_key(r)
        original = first.get(key)
        if original is None:
            first[key] = r
            kept.append(r)
            continue
        copy = {"pdf_file": r["pdf_file"], "digest": r.get("digest"), "suffix_side_index": int(r["side_index"]),
                "flag": "side_redownload"}
        original.setdefault("side_redownloads", []).append(copy)
        original["flags"]["side_redownload_copies"] = original["flags"].get("side_redownload_copies", 0) + 1
        copies.append(dict(copy, copy_of_pdf_file=original["pdf_file"], copy_of_digest=original.get("digest"),
                           copy_of_suffix_side_index=int(original["side_index"])))
    return kept, copies


def renumber_sides(records: Sequence[Dict[str, object]]) -> None:
    """Re-number one shelf mark's distinct sides by suffix order (1, 2, 3, ...).

    The ``_<n>`` suffix is download order, not the leaf order, and re-downloads and error pages
    leave holes in it. Sides are numbered by the dense rank of their suffix index (equal suffixes
    keep one number, so :func:`group_side_records` still sees the duplicate); the suffix index is
    kept in ``suffix_side_index``. Every side of a shelf mark with two or more sides, and a lone side
    whose number changed, gets ``side_index_from_download_order_unverified``. Records change in
    place (``side_index``, ``folio``, ``flags``).

    :param records: Side records of ONE shelf mark (after :func:`merge_side_redownloads`).
    """
    suffixes = sorted({int(r["side_index"]) for r in records})
    rank = {s: i + 1 for i, s in enumerate(suffixes)}
    for r in records:
        old = int(r["side_index"])
        r["suffix_side_index"] = old
        r["side_index"] = rank[old]
        r["folio"] = f"s{rank[old]}"
        if len(suffixes) > 1 or rank[old] != old:
            r["flags"]["side_index_from_download_order_unverified"] = 1


def group_side_records(records: Sequence[Dict[str, object]],
                       key: Callable[[Dict[str, object]], str] = lambda r: normalise_shelfmark(str(r["shelfmark"] or ""))
                       ) -> List[Dict[str, object]]:
    """Group side records into documents by shelf mark and check the side sequence.

    Within a document, sides are ordered by index. Two records with one side index get
    ``side_index_duplicate`` (both); a side followed by a later side than its successor gets
    ``side_sequence_gap`` (value: sides skipped). Flags are added to the records in place.

    :param records: :func:`ingest_side_files` records (any shelf marks).
    :param key: Shelf-mark key of a record.
    :return: One dict per document: ``key``, ``shelfmark``, ``sides`` (records), ``side_indices``.
    """
    groups: Dict[str, List[Dict[str, object]]] = collections.defaultdict(list)
    for r in records:
        groups[key(r)].append(r)
    out = []
    for k, recs in groups.items():
        recs = sorted(recs, key=lambda r: (int(r["side_index"]), r["pdf_file"]))
        by_index = collections.Counter(int(r["side_index"]) for r in recs)
        for r in recs:
            if by_index[int(r["side_index"])] > 1:
                r["flags"]["side_index_duplicate"] = 1
        ordered = sorted(by_index)
        for a, b in zip(ordered, ordered[1:]):
            if b - a > 1:
                for r in recs:
                    if int(r["side_index"]) == a:
                        r["flags"]["side_sequence_gap"] = b - a - 1
        out.append({"key": k, "shelfmark": recs[0]["shelfmark"], "sides": recs, "side_indices": ordered})
    return out


# ───────────────────────────── quality ─────────────────────────────
def line_structure(cleaned_lines: Sequence[str], numbered_lines: int = 0, min_lines: int = 3,
                   wide: int = 70, uniform_share: float = 0.6) -> str:
    """Whether a transcription keeps the manuscript's lines (a heuristic).

    ``numbered_lines``: half or more of the lines carried line numbers. ``reflowed_suspect``: the
    longest line has ``wide`` letters or more and ``uniform_share`` of the lines are within 15 %
    of it (text wrapped at the page width rather than at the manuscript's line ends).
    ``too_few_lines``: fewer than ``min_lines``. Otherwise ``line_breaks``.

    :param cleaned_lines: Cleaned lines.
    :param numbered_lines: Lines whose numbers were stripped.
    :param min_lines: Fewest lines for a verdict.
    :param wide: Letters that make a line page-wide.
    :param uniform_share: Share of near-maximal lines that signals reflowed text.
    :return: ``numbered_lines``, ``reflowed_suspect``, ``too_few_lines`` or ``line_breaks``.
    """
    if len(cleaned_lines) < min_lines:
        return "too_few_lines"
    if numbered_lines >= 0.5 * len(cleaned_lines):
        return "numbered_lines"
    counts = [len(LETTER_RE.findall(ln)) for ln in cleaned_lines]
    top = max(counts)
    if top >= wide and sum(1 for c in counts if c >= 0.85 * top) >= uniform_share * len(counts):
        return "reflowed_suspect"
    return "line_breaks"


def credit_kind(credit: Optional[str]) -> Optional[str]:
    """Kind of a transcription or catalogue credit (the credit's wording is never kept).

    :param credit: Credit string from the store's metadata, or None.
    :return: ``site_user`` (``Site User - ...``), ``project_team`` (names a team), ``other``, or None
        for an empty credit.
    """
    if not credit or not str(credit).strip():
        return None
    low = str(credit).strip().lower()
    if low.startswith("site user"):
        return "site_user"
    return "project_team" if "team" in low else "other"


def provenance_signals(records: Sequence[Mapping[str, object]], style: str,
                       credit: Optional[str] = None) -> Dict[str, object]:
    """Format family and quality signals of one transcription (all its folio or side records).

    :param records: Folio records (:func:`ingest`) or side records (:func:`ingest_side_files`) of
        one distinct PDF text.
    :param style: :func:`detect_style` result.
    :param credit: Credit KIND (:func:`credit_kind`: ``project_team``, ``site_user``, ``other``) of
        the store's transcription metadata, or None. A credit string is reduced to its kind; no
        wording is returned.
    :return: ``format_family``, ``layout`` (heading kinds / column markers),
        ``credit_kind`` (``project_team`` / ``site_user`` / ``other`` / None), ``letters``,
        ``sigla_per_100_letters``, ``line_structure`` (most common over the records),
        ``median_letters_per_line``, ``finals_mapping``, ``finals_repaired``, ``finals_unresolved``,
        ``excerpt``, ``units`` (folios or sides).
    """
    letters = sum(int(r["letters"]) for r in records)
    raw = "\n".join(line for r in records for line in r["raw_lines"])
    mapping_symbols = {k for r in records for k in r.get("finals_mapping", {})}
    sigla = sum(1 for ch in SIGLA_CHAR_RE.findall(raw) if ch not in mapping_symbols)
    raw_letters = len(LETTER_RE.findall(raw))
    structures = collections.Counter(line_structure(r["cleaned_lines"], int(r["flags"].get("line_numbers_stripped", 0)))
                                     for r in records)
    per_line = [len(LETTER_RE.findall(ln)) for r in records for ln in r["cleaned_lines"]]
    if style == "side_per_file":
        layout = "columns" if any(r.get("column_markers") for r in records) else "plain"
    else:
        kinds = collections.Counter(str(r.get("marker_kind")) for r in records)
        layout = "+".join(k for k, _ in kinds.most_common())
    kind = credit if credit in ("project_team", "site_user", "other") else credit_kind(credit)
    return {
        "format_family": style, "layout": layout, "credit_kind": kind,
        "letters": letters, "sigla_per_100_letters": round(100 * sigla / raw_letters, 2) if raw_letters else 0.0,
        "line_structure": structures.most_common(1)[0][0] if structures else "too_few_lines",
        "median_letters_per_line": statistics.median(per_line) if per_line else 0,
        "finals_mapping": records[0].get("finals_mapping", {}) if records else {},
        "finals_repaired": sum(int(r["finals_repaired"]) for r in records),
        "finals_unresolved": sum(int(r["flags"].get("unresolved_finals", 0)) for r in records),
        "excerpt": any(r["flags"].get("excerpt_document") for r in records), "units": len(records),
    }


def page_row_exclusions(record: Mapping[str, object]) -> List[str]:
    """Why a folio or side record may not become a page row (text side only).

    Reasons: every flag of :data:`PAGE_ROW_EXCLUDING_FLAGS` it carries, ``label_not_folio`` for a
    new-style record without a ``<leaf><r|v>`` label (page fallback, preamble), ``no_letters``.
    An image is never part of this check: no record has a verified image (image-to-folio pairing
    is a separate, still missing step).

    :param record: Folio or side record.
    :return: Reasons (empty when the text may be used).
    """
    reasons = [f for f in PAGE_ROW_EXCLUDING_FLAGS if record["flags"].get(f)]
    if record.get("style", "header_folio") == "header_folio" and side_index(record.get("folio")) is None:
        reasons.append("label_not_folio")
    if not record.get("letters"):
        reasons.append("no_letters")
    return reasons
