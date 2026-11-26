"""Letters for the glyphs of a PDF whose text layer has none, or a wrong one, for them.

Word-processor PDFs embed their fonts as TrueType subsets. For many of those subsets the text layer does not know
which letter a glyph is: every connected (initial, medial, final) form of an Arabic letter and the lam-alef ligature
come out as a control character or U+FFFD, some as a wrong letter (kaf as alef with madda), and legacy Hebrew fonts
lose their final letters the same way. Reading such a page through the text layer keeps a third of its Arabic
letters. The glyph itself is in the file, and a glyph is named by its outline:

1. :class:`ReferenceFont`: a font installed on this machine (:data:`REFERENCE_FONT_FILES`), indexed by outline. An
   embedded glyph with the same outline is the same glyph, and the reference font knows its letter: its character map
   (a presentation form folds to its letter, a ligature to its letters) or its glyph name (``uni0628.medi``,
   ``uni06440627.isol``, :data:`GLYPH_NAME_TEXT`). An embedded font uses a reference font only when at least
   :data:`MIN_REFERENCE_SHARE` of its outlines are found in it.
2. :data:`LEARNED_TABLE`: outline digest -> text for fonts that are not installed here, learned once from rendered
   pages (the round's ``learn_glyph_table.py``) and checked against them.
3. The font's own outlines (:func:`pointed_letters`): some pointed (vocalised) Hebrew editions are set in a font with
   a private encoding in which a letter carrying a dot (dagesh, shin/sin dot, holam or shuruk on vav) is a
   precomposed glyph that the text layer reports as an ASCII digit, capital or symbol. Such a glyph is the outline of
   one of the font's plain letters (the ones the text layer reports as Hebrew letters) plus one or two mark-sized
   contours, and takes that letter. A font with such glyphs in use is rebuilt glyph by glyph (its text layer also
   has spaces that MuPDF invents inside pointed words); its points that the layer has no character for are left out.

:class:`GlyphDecoder` holds a document's tables. :meth:`GlyphDecoder.page_spans` gives, for the fonts of a page whose
text layer is wrong about Arabic letters or reports precomposed pointed letters, one span per glyph (per run of digits
or Latin letters) with its position,
so the caller orders them by geometry: the text layer's own order is not usable there, because MuPDF reverses
right-to-left runs by the (wrong) characters. :meth:`GlyphDecoder.page_substitutions` gives the letters of the
remaining glyphs the text layer has no character for (legacy-font final letters), by position.

No reader and no OCR takes part: the result is the edition's own text.
"""
from __future__ import annotations

import collections
import functools
import hashlib
import io
import json
import logging
import re
import statistics
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Counter, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import fitz
from fontTools.ttLib import TTFont

# embedded subsets carry empty or odd creation dates; fontTools warns about each one when it reads the font header
logging.getLogger("fontTools.ttLib.tables._h_e_a_d").setLevel(logging.ERROR)

REFERENCE_FONT_DIRS: Tuple[Path, ...] = (Path("/System/Library/Fonts/Supplemental"), Path("/Library/Fonts"))
REFERENCE_FONT_FILES: Tuple[str, ...] = (
    "Times New Roman.ttf", "Times New Roman Bold.ttf", "Times New Roman Italic.ttf", "Times New Roman Bold Italic.ttf",
    "Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf", "Arial Bold Italic.ttf", "Tahoma.ttf", "Tahoma Bold.ttf",
    "Courier New.ttf", "Courier New Bold.ttf", "Microsoft Sans Serif.ttf")
LEARNED_TABLE = Path(__file__).with_name("pdf_glyph_table.json")
GLYPH_NAME_TEXT: Dict[str, str] = {"lillah": "لله"}   # the lam-lam-heh ligature of "Allah"
MIN_REFERENCE_SHARE = 0.5
MIN_REFERENCE_GLYPHS = 5
UNKNOWN = "�"
POSITION_DIGITS = 1                  # origins are matched to a tenth of a point
ARABIC_LETTER_RE = re.compile(r"[ء-يٮ-ۓ]")
HEBREW_LETTER_RE = re.compile(r"[א-ת]")
LTR_TEXT_RE = re.compile(r"^[0-9A-Za-zÀ-ɏ٠-٩۰-۹]+$")
UNI_NAME_RE = re.compile(r"^uni((?:[0-9A-Fa-f]{4})+)$")
PLAIN_HEBREW_RE = re.compile(r"^[א-ת]$")
# a precomposed pointed letter: a plain letter's contours, all moved by one offset of at most MAX_OFFSET_SHARE of the
# letter's larger side, plus 1..MAX_MARK_CONTOURS extra contours whose larger side is at most MARK_MAX_SHARE of it
# (measured in T-S Misc.35.54's font: dots 0.10-0.18 of the letter, the patah or qamats of a precomposed alef 0.35)
MARK_MAX_SHARE = 0.4
MAX_OFFSET_SHARE = 0.1
MAX_MARK_CONTOURS = 2
# font units a matched point may differ by: a precomposed alef with patah (T-S Ar.1a.90) repeats the alef's 52 points
# but one off-curve point at 683 instead of 684
POINT_TOLERANCE = 1

Point = Tuple[int, int, int]                     # x, y, on-curve bit
Contour = Tuple[Point, ...]


def is_unmapped(code_point: int) -> bool:
    """Whether a text-layer character stands for a glyph the layer has no letter for.

    :param code_point: The character's code point.
    :return: True for a control character, a private-use character and U+FFFD.
    """
    return code_point < 0x20 or 0x7F <= code_point < 0xA0 or 0xE000 <= code_point <= 0xF8FF or code_point == 0xFFFD


def outline_digest(font: TTFont, glyph_name: str) -> Optional[str]:
    """Digest of a glyph's outline (points, contour ends, on-curve bits; components flattened).

    :param font: A TrueType font.
    :param glyph_name: Name of one of its glyphs.
    :return: The digest, or None for a glyph without an outline (a space).
    """
    glyf = font["glyf"]
    glyph = glyf[glyph_name]
    if glyph.numberOfContours == 0:
        return None
    coordinates, ends, flags = glyph.getCoordinates(glyf)
    if not len(coordinates):
        return None
    payload = "|".join((",".join(f"{round(x)},{round(y)}" for x, y in coordinates), ",".join(str(e) for e in ends),
                        ",".join(str(f & 1) for f in flags)))
    return hashlib.sha1(payload.encode("ascii")).hexdigest()


def glyph_contours(font: TTFont, glyph_name: str) -> List[Contour]:
    """A glyph's contours (components flattened), as rounded points with their on-curve bits.

    :param font: A TrueType font.
    :param glyph_name: Name of one of its glyphs.
    :return: The contours in drawing order (empty for a glyph without an outline).
    """
    glyf = font["glyf"]
    glyph = glyf[glyph_name]
    if glyph.numberOfContours == 0:
        return []
    coordinates, ends, flags = glyph.getCoordinates(glyf)
    points = [(round(x), round(y), f & 1) for (x, y), f in zip(coordinates, flags)]
    out: List[Contour] = []
    start = 0
    for end in ends:
        out.append(tuple(points[start:end + 1]))
        start = end + 1
    return [c for c in out if c]


def contours_box(contours: Sequence[Contour]) -> Tuple[int, int, int, int]:
    """Bounding box of contours.

    :param contours: Non-empty contours.
    :return: ``(x_min, y_min, x_max, y_max)``.
    """
    xs = [p[0] for c in contours for p in c]
    ys = [p[1] for c in contours for p in c]
    return min(xs), min(ys), max(xs), max(ys)


def larger_side(contours: Sequence[Contour]) -> int:
    """The larger side of the bounding box of contours.

    :param contours: Non-empty contours.
    :return: Width or height, whichever is larger.
    """
    x0, y0, x1, y1 = contours_box(contours)
    return max(x1 - x0, y1 - y0)


def same_contour(a: Contour, b: Contour, dx: int, dy: int, tolerance: int = POINT_TOLERANCE) -> bool:
    """Whether contour ``b`` is contour ``a`` moved by ``(dx, dy)``, point for point.

    :param a: A contour.
    :param b: Another contour.
    :param dx: Horizontal offset.
    :param dy: Vertical offset.
    :param tolerance: Font units each coordinate may differ by.
    :return: True when the points (and their on-curve bits) agree.
    """
    return len(a) == len(b) and all(pa[2] == pb[2] and abs(pa[0] + dx - pb[0]) <= tolerance
                                    and abs(pa[1] + dy - pb[1]) <= tolerance for pa, pb in zip(a, b))


def letter_with_marks(candidate: Sequence[Contour], letter: Sequence[Contour], mark_max_share: float = MARK_MAX_SHARE,
                      max_offset_share: float = MAX_OFFSET_SHARE, max_marks: int = MAX_MARK_CONTOURS) -> bool:
    """Whether a glyph is a letter's outline plus a few mark-sized contours.

    Every contour of ``letter``, moved by one common offset (at most ``max_offset_share`` of the letter's larger
    side in each direction), must be a contour of ``candidate`` point for point (within :data:`POINT_TOLERANCE`
    font units); the 1..``max_marks`` contours left over must each have a larger side of at most
    ``mark_max_share`` of the letter's larger side.

    :param candidate: The glyph's contours.
    :param letter: The letter glyph's contours.
    :param mark_max_share: Largest mark, as a share of the letter's larger side.
    :param max_offset_share: Largest offset, as a share of the letter's larger side.
    :param max_marks: Most extra contours.
    :return: True when the glyph is the letter with marks.
    """
    if not letter or not 1 <= len(candidate) - len(letter) <= max_marks:
        return False
    size = larger_side(letter)
    first = letter[0]
    for anchor in candidate:
        if len(anchor) != len(first):
            continue
        dx, dy = anchor[0][0] - first[0][0], anchor[0][1] - first[0][1]
        if max(abs(dx), abs(dy)) > max_offset_share * size:
            continue
        left = list(candidate)
        matched = True
        for contour in letter:
            hit = next((c for c in left if same_contour(contour, c, dx, dy)), None)
            if hit is None:
                matched = False
                break
            left.remove(hit)
        if matched and all(larger_side([c]) <= mark_max_share * size for c in left):
            return True
    return False


def precomposed_shapes(font: TTFont) -> Dict[int, List[int]]:
    """Glyphs whose outline is another glyph's outline plus mark-sized contours (:func:`letter_with_marks`).

    :param font: A TrueType font.
    :return: Glyph id -> ids of the glyphs it is made of (outlines only; nothing is known yet of their letters).
    """
    contours = {gid: glyph_contours(font, name) for gid, name in enumerate(font.getGlyphOrder())}
    lengths = {gid: collections.Counter(len(c) for c in cs) for gid, cs in contours.items() if cs}
    out: Dict[int, List[int]] = {}
    for gid, cs in contours.items():
        if len(cs) < 2:
            continue
        bases = [base for base, base_lengths in lengths.items() if base != gid and len(contours[base]) < len(cs)
                 and not base_lengths - lengths[gid] and letter_with_marks(cs, contours[base])]
        if bases:
            out[gid] = bases
    return out


def is_hebrew_or_arabic(text: str) -> bool:
    """Whether a text holds a Hebrew or Arabic letter or mark (any code point of those scripts' blocks).

    :param text: A text-layer character (or characters).
    :return: True when one of its characters is named HEBREW or ARABIC by Unicode.
    """
    return any(unicodedata.name(c, "").startswith(("HEBREW", "ARABIC")) for c in text)


def is_hebrew_mark_text(text: str) -> bool:
    """Whether a text is made only of Hebrew points and accents (U+0591..U+05C7).

    :param text: A text.
    :return: True for a non-empty text of those marks only.
    """
    return bool(text) and all(0x0591 <= ord(c) <= 0x05C7 for c in text)


def pointed_letters(font: TTFont, layer: Mapping[int, str],
                    named: Mapping[int, str]) -> Tuple[Dict[int, str], Set[int]]:
    """Name a font's precomposed pointed letters from its own outlines, and find its unnamed points.

    The plain letters are the glyphs the text layer reports as a Hebrew letter (``layer``), or that ``named`` (the
    font's table) names one. A glyph the layer reports as anything but a Hebrew or Arabic letter or mark (an ASCII
    character, a symbol, an unmapped character) whose outline is exactly one plain letter's outline plus mark-sized
    contours (:func:`precomposed_shapes`) takes that letter: a shin with its dot is shin, a vav with holam or shuruk
    is vav, a final letter with dagesh the final letter. When it fits glyphs of two letters it stays unnamed. In a
    font with such letters, a glyph the layer has no character for and nothing names is a point, and carries no
    letter, when it is drawn wholly below the baseline, or wholly above the median top of the plain letters and no
    larger than a mark (:data:`MARK_MAX_SHARE` of their median height): the points the layer reports lie there too.

    :param font: The embedded TrueType font.
    :param layer: Glyph id -> the text layer's character for it (by majority over the document); glyphs not drawn
        are absent.
    :param named: Glyph id -> text the font's table already has.
    :return: ``(glyph id -> letter of the precomposed glyphs, glyph ids of the points)``.
    """
    shapes = precomposed_shapes(font)
    plain = {gid: ch for gid, ch in layer.items() if PLAIN_HEBREW_RE.match(ch)}
    plain.update({gid: t for gid, t in named.items() if PLAIN_HEBREW_RE.match(t)})
    letters: Dict[int, str] = {}
    for gid, bases in shapes.items():
        if gid in named or gid not in layer or is_hebrew_or_arabic(layer[gid]):
            continue
        fits = {plain[b] for b in bases if b in plain}
        if len(fits) == 1:
            letters[gid] = fits.pop()
    points: Set[int] = set()
    if letters:
        order = font.getGlyphOrder()
        boxes = [contours_box(cs) for cs in (glyph_contours(font, order[g]) for g in plain) if cs]
        top = statistics.median(b[3] for b in boxes)
        height = statistics.median(b[3] - b[1] for b in boxes)
        for gid, ch in layer.items():
            if gid in named or gid in letters or not is_unmapped(ord(ch)):
                continue
            contours = glyph_contours(font, order[gid])
            if not contours:
                continue
            _, y_min, _, y_max = contours_box(contours)
            if y_max <= 0 or (y_min > top and larger_side(contours) <= MARK_MAX_SHARE * height):
                points.add(gid)
    return letters, points


def fold_letters(text: str) -> str:
    """Fold presentation forms to their letters (compatibility composition) and leave everything else.

    :param text: Text.
    :return: The folded text.
    """
    return unicodedata.normalize("NFKC", text)


def glyph_text(code_points: Iterable[int], glyph_name: str) -> Optional[str]:
    """The text of a reference font's glyph, from the characters mapped to it and its name.

    The order of preference keeps Arabic text in its standard letters: an Arabic presentation form B (U+FE70..FEFC,
    folded to its letter or letters), then any character that is not a presentation form or private (standard Arabic
    letters U+0621..064A first), then a presentation form A (folded). A glyph no character maps to is named by
    :data:`GLYPH_NAME_TEXT` or a ``uniXXXX[YYYY...]`` name (anything after the first full stop is the form).

    :param code_points: Characters the font's character map sends to the glyph.
    :param glyph_name: The glyph's name.
    :return: The text, or None when neither names it.
    """
    points = sorted(set(code_points))
    forms_b = [c for c in points if 0xFE70 <= c <= 0xFEFC]
    if forms_b:
        return fold_letters(chr(forms_b[0]))
    plain = [c for c in points if not (0xFB50 <= c <= 0xFDFF or 0xE000 <= c <= 0xF8FF or c < 0x20)]
    if plain:
        standard = [c for c in plain if 0x0621 <= c <= 0x064A]
        return chr((standard or plain)[0])
    forms_a = [c for c in points if 0xFB50 <= c <= 0xFDFF]
    if forms_a:
        return fold_letters(chr(forms_a[0]))
    base = glyph_name.split(".")[0]
    if base in GLYPH_NAME_TEXT:
        return GLYPH_NAME_TEXT[base]
    named = UNI_NAME_RE.match(base)
    if named:
        digits = named.group(1)
        return fold_letters("".join(chr(int(digits[i:i + 4], 16)) for i in range(0, len(digits), 4)))
    return None


def text_rank(text: str) -> Tuple[int, str]:
    """Sort key choosing among glyphs of one outline: plain ASCII, then Hebrew, then standard Arabic, then the rest.

    :param text: A candidate text.
    :return: The key (lower is preferred).
    """
    if all(ord(c) < 0x80 for c in text):
        return 0, text
    if all(0x0590 <= ord(c) <= 0x05FF for c in text):
        return 1, text
    if all(0x0621 <= ord(c) <= 0x064A for c in text):
        return 2, text
    return 3, text


@dataclass
class ReferenceFont:
    """An installed font indexed by glyph outline.

    :param name: File name.
    :param texts: Outline digest -> text of the glyph (the preferred one when several glyphs share an outline).
    """

    name: str
    texts: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "ReferenceFont":
        """Index a font file.

        :param path: A TrueType font file.
        :return: The index.
        """
        font = TTFont(path, fontNumber=0, lazy=True)
        mapped: Dict[str, List[int]] = collections.defaultdict(list)
        for table in font["cmap"].tables:
            if table.isUnicode():
                for code_point, glyph_name in table.cmap.items():
                    mapped[glyph_name].append(code_point)
        candidates: Dict[str, List[str]] = collections.defaultdict(list)
        for glyph_name in font.getGlyphOrder():
            text = glyph_text(mapped.get(glyph_name, ()), glyph_name)
            if text is None:
                continue
            digest = outline_digest(font, glyph_name)
            if digest is not None:
                candidates[digest].append(text)
        return cls(path.name, {digest: min(texts, key=text_rank) for digest, texts in candidates.items()})


@functools.lru_cache(maxsize=4)
def load_references(directories: Tuple[Path, ...] = REFERENCE_FONT_DIRS,
                    files: Tuple[str, ...] = REFERENCE_FONT_FILES) -> Tuple[ReferenceFont, ...]:
    """The reference fonts found on this machine.

    :param directories: Folders searched, in order.
    :param files: Font file names.
    :return: One index per file found (the first folder holding it wins).
    """
    out: List[ReferenceFont] = []
    for name in files:
        path = next((d / name for d in directories if (d / name).exists()), None)
        if path is not None:
            out.append(ReferenceFont.load(path))
    return tuple(out)


@functools.lru_cache(maxsize=4)
def load_learned(path: Path = LEARNED_TABLE) -> Dict[str, str]:
    """The learned outline table.

    :param path: The JSON file (``{"glyphs": {digest: text}}``); a missing file is an empty table.
    :return: Outline digest -> text.
    """
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8"))["glyphs"])


@dataclass
class FontTable:
    """What is known of one embedded font's glyphs.

    :param texts: Glyph id -> text.
    :param digests: Glyph id -> outline digest (glyphs with an outline).
    :param reference: The reference font used, if one matched.
    :param n_reference: Glyphs named by the reference font.
    :param n_learned: Glyphs named by the learned table.
    :param pointed: Glyph id -> letter of the precomposed pointed letters (:func:`pointed_letters`; also in
        ``texts``). A font with any is rebuilt glyph by glyph.
    :param points: Glyph ids of its points the layer has no character for (text ``""`` in ``texts``).
    """

    texts: Dict[int, str] = field(default_factory=dict)
    digests: Dict[int, str] = field(default_factory=dict)
    reference: Optional[str] = None
    n_reference: int = 0
    n_learned: int = 0
    pointed: Dict[int, str] = field(default_factory=dict)
    points: Set[int] = field(default_factory=set)


def font_table(program: bytes, references: Sequence[ReferenceFont], learned: Mapping[str, str]) -> FontTable:
    """Name the glyphs of an embedded TrueType font.

    :param program: The embedded font file.
    :param references: Reference fonts.
    :param learned: Learned outline table.
    :return: The table (empty for a font without TrueType outlines).
    """
    font = TTFont(io.BytesIO(program), lazy=True)
    table = FontTable()
    if "glyf" not in font:
        return table
    for glyph_id, glyph_name in enumerate(font.getGlyphOrder()):
        digest = outline_digest(font, glyph_name)
        if digest is not None:
            table.digests[glyph_id] = digest
    if not table.digests:
        return table
    hits = [(sum(1 for d in table.digests.values() if d in ref.texts), ref) for ref in references]
    best_hits, best = max(hits, key=lambda h: h[0], default=(0, None))
    use_reference = (best is not None and best_hits >= MIN_REFERENCE_GLYPHS
                     and best_hits >= MIN_REFERENCE_SHARE * len(table.digests))
    for glyph_id, digest in table.digests.items():
        if use_reference and digest in best.texts:
            table.texts[glyph_id] = best.texts[digest]
            table.n_reference += 1
        elif digest in learned:
            table.texts[glyph_id] = learned[digest]
            table.n_learned += 1
    if use_reference:
        table.reference = best.name
    return table


@dataclass
class Glyph:
    """One drawn glyph of a page.

    :param font: Font name (without the subset tag).
    :param size: Font size.
    :param glyph_id: Glyph id in the embedded font.
    :param layer_char: The text layer's character for it.
    :param x0: Left edge.
    :param x1: Right edge.
    :param x: Origin x.
    :param y: Baseline.
    :param text: Its text from the font's table, or None.
    """

    font: str
    size: float
    glyph_id: int
    layer_char: str
    x0: float
    x1: float
    x: float
    y: float
    text: Optional[str] = None


@dataclass
class GlyphSpan:
    """Text placed by geometry: one glyph, or one left-to-right run of digits or Latin letters.

    :param text: The text.
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


def font_name(base_font: str) -> str:
    """A PDF font's name without its subset tag (``ABCDEF+Name`` -> ``Name``).

    :param base_font: The PDF's BaseFont.
    :return: The name the text layer reports.
    """
    return base_font.split("+", 1)[-1]


def position_key(font: str, x: float, y: float) -> Tuple[str, float, float]:
    """Key matching a text-layer character to its drawn glyph.

    :param font: Font name.
    :param x: Origin x.
    :param y: Origin y.
    :return: The key.
    """
    return font, round(x, POSITION_DIGITS), round(y, POSITION_DIGITS)


class GlyphDecoder:
    """A document's glyph tables, and the page texts they correct.

    :param doc: Open PDF.
    :param references: Reference fonts (default: :func:`load_references`).
    :param learned: Learned outline table (default: :func:`load_learned`).
    """

    def __init__(self, doc: "fitz.Document", references: Optional[Sequence[ReferenceFont]] = None,
                 learned: Optional[Mapping[str, str]] = None) -> None:
        self.doc = doc
        self.references = load_references() if references is None else references
        self.learned = load_learned() if learned is None else learned
        self.tables: Dict[int, FontTable] = {}
        self.stats: Counter[str] = collections.Counter()
        self._layer: Optional[Dict[int, Dict[int, str]]] = None

    def layer_characters(self, xref: int) -> Dict[int, str]:
        """The text layer's character for each drawn glyph of an embedded font, by majority over the document.

        Read from :meth:`fitz.Page.get_texttrace` (glyph ids with their characters). A page that uses two fonts of
        the same name does not vote for either.

        :param xref: The font's xref.
        :return: Glyph id -> character.
        """
        if self._layer is None:
            votes: Dict[int, Dict[int, Counter[str]]] = collections.defaultdict(
                lambda: collections.defaultdict(collections.Counter))
            for page in self.doc:
                xrefs: Dict[str, Set[int]] = collections.defaultdict(set)
                for entry in page.get_fonts(full=True):
                    if entry[1] == "ttf":
                        xrefs[font_name(entry[3])].add(entry[0])
                for span in page.get_texttrace():
                    named = xrefs.get(span["font"], set())
                    if len(named) != 1:
                        continue
                    font_votes = votes[next(iter(named))]
                    for code_point, glyph_id, _, _ in span["chars"]:
                        font_votes[glyph_id][chr(code_point)] += 1
            self._layer = {x: {g: c.most_common(1)[0][0] for g, c in by_glyph.items()} for x, by_glyph in votes.items()}
        return self._layer.get(xref, {})

    def add_pointed(self, xref: int, program: bytes, table: FontTable) -> None:
        """Add a font's precomposed pointed letters and its unnamed points to its table (:func:`pointed_letters`).

        The document's text layer is read only for a font that has glyphs made of another glyph plus marks.

        :param xref: The font's xref.
        :param program: The embedded font file.
        :param table: Its table (changed in place).
        """
        font = TTFont(io.BytesIO(program), lazy=True)
        if "glyf" not in font or not precomposed_shapes(font):
            return
        letters, points = pointed_letters(font, self.layer_characters(xref), table.texts)
        table.pointed = letters
        table.points = points
        table.texts.update(letters)
        table.texts.update({gid: "" for gid in points})

    def page_tables(self, page: "fitz.Page") -> Dict[str, FontTable]:
        """Tables of the embedded TrueType fonts a page uses, by font name.

        Two fonts of one name on a page keep only the glyph ids they agree on.

        :param page: PDF page.
        :return: Font name -> table.
        """
        out: Dict[str, FontTable] = {}
        for entry in page.get_fonts(full=True):
            xref, extension, base_font = entry[0], entry[1], entry[3]
            if extension != "ttf":
                continue
            if xref not in self.tables:
                program = self.doc.extract_font(xref)[3]
                self.tables[xref] = font_table(program, self.references, self.learned) if program else FontTable()
                if program:
                    self.add_pointed(xref, program, self.tables[xref])
            table, name = self.tables[xref], font_name(base_font)
            if name not in out:
                out[name] = table
            elif out[name] is not table:
                both = out[name]
                texts = {g: t for g, t in both.texts.items() if table.texts.get(g, t) == t and
                         (g not in table.digests or table.digests[g] == both.digests.get(g))}
                out[name] = FontTable(texts, dict(both.digests), both.reference, both.n_reference, both.n_learned,
                                      {g: t for g, t in both.pointed.items() if g in texts},
                                      {g for g in both.points if g in texts})
        return out

    def page_glyphs(self, page: "fitz.Page") -> List[Glyph]:
        """The drawn glyphs of the fonts that have a table, with their texts.

        :param page: PDF page.
        :return: Glyphs in drawing order.
        """
        tables = {name: t for name, t in self.page_tables(page).items() if t.texts}
        out: List[Glyph] = []
        if not tables:
            return out
        for span in page.get_texttrace():
            table = tables.get(span["font"])
            if table is None:
                continue
            for code_point, glyph_id, origin, bbox in span["chars"]:
                out.append(Glyph(span["font"], span["size"], glyph_id, chr(code_point), bbox[0], bbox[2], origin[0],
                                 origin[1], table.texts.get(glyph_id)))
        return out

    @staticmethod
    def rebuilt_fonts(glyphs: Sequence[Glyph], pointed: Iterable[str] = ()) -> Set[str]:
        """Fonts whose text layer is wrong about Arabic letters, or reports precomposed pointed letters, on a page.

        A font is rebuilt from its glyphs when its table reads an Arabic letter in at least one of them and the text
        layer disagrees with the table on at least one glyph (no letter for it, or another letter). A font of
        ``pointed`` (its table has precomposed pointed letters in use in the document) is rebuilt on every page
        that draws its glyphs.

        :param glyphs: The page's glyphs.
        :param pointed: Names of the page's fonts with precomposed pointed letters.
        :return: Font names.
        """
        arabic: Counter[str] = collections.Counter()
        wrong: Counter[str] = collections.Counter()
        for g in glyphs:
            if g.text is None:
                continue
            arabic[g.font] += bool(ARABIC_LETTER_RE.search(g.text))
            wrong[g.font] += fold_letters(g.layer_char) != g.text
        drawn = {g.font for g in glyphs}
        return {name for name in arabic if arabic[name] and wrong[name]} | (set(pointed) & drawn)

    def page_spans(self, page: "fitz.Page") -> Tuple[Set[str], List[GlyphSpan]]:
        """Glyph-level spans of the fonts rebuilt on a page.

        Every glyph is one span, in no particular order (the caller orders a line's spans by position); digits and
        Latin letters that follow each other on a baseline form one span read left to right. A glyph the table has
        no text for keeps the text layer's character when that is a real character, and is :data:`UNKNOWN`
        otherwise (counted in ``stats["glyphs_unknown"]``). In a font with precomposed pointed letters a point joins
        the letter it is drawn on (:meth:`attach_points`), and a point without a character (text ``""``) gives no
        span.

        :param page: PDF page.
        :return: ``(names of the rebuilt fonts, their spans)``.
        """
        glyphs = self.page_glyphs(page)
        pointed = {name: t for name, t in self.page_tables(page).items() if t.pointed}
        fonts = self.rebuilt_fonts(glyphs, pointed)
        spans: List[GlyphSpan] = []
        if not fonts:
            return fonts, spans
        rows: Dict[Tuple[str, float, float], List[Glyph]] = collections.defaultdict(list)
        for g in glyphs:
            if g.font in fonts:
                rows[(g.font, round(g.size, 2), round(g.y, 1))].append(g)
        for (name, _, _), row in rows.items():
            row.sort(key=lambda g: g.x0)
            texts = {id(g): self._text(g) for g in row}
            if name in pointed:
                row = self.attach_points(row, texts)
            run: List[Glyph] = []
            for g in row + [None]:
                text = None if g is None else texts[id(g)]
                if g is not None and LTR_TEXT_RE.match(text) and (not run or g.x0 - run[-1].x1 <= 0.25 * g.size):
                    run.append(g)
                    continue
                if run:
                    spans.append(GlyphSpan("".join(texts[id(r)] for r in run), name, run[0].x0, run[-1].x1,
                                           run[0].y, run[0].size))
                    run = []
                if g is None:
                    continue
                if LTR_TEXT_RE.match(text):
                    run = [g]
                else:
                    spans.append(GlyphSpan(text, name, g.x0, g.x1, g.y, g.size))
        self.stats["fonts_rebuilt"] += len(fonts)
        self.stats["glyphs_rebuilt"] += sum(1 for g in glyphs if g.font in fonts)
        self.stats["glyphs_unknown"] += sum(1 for g in glyphs if g.font in fonts and self._text(g) == UNKNOWN)
        self.stats["glyphs_pointed"] += sum(1 for g in glyphs
                                            if g.font in pointed and g.glyph_id in pointed[g.font].pointed)
        return fonts, spans

    @staticmethod
    def attach_points(row: Sequence[Glyph], texts: Dict[int, str]) -> List[Glyph]:
        """Join each point of a baseline to the letter it is drawn on.

        A point (a glyph whose text is combining marks only) whose horizontal centre lies within the box of a glyph
        of right-to-left text (neither blank, a point nor a digit or Latin letter) is appended to that glyph's text
        (the nearest by centre when two hold it) and leaves the row: as a span of its own, a point narrower than its
        letter would open a gap that reads as a space between letters of one word.

        :param row: The glyphs of one baseline of one font.
        :param texts: ``id(glyph)`` -> its text (changed in place for the letters that take points).
        :return: The row without the attached points.
        """
        def is_point(g: Glyph) -> bool:
            text = texts[id(g)]
            return bool(text) and all(unicodedata.category(c) == "Mn" for c in text)

        hosts = [g for g in row if texts[id(g)].strip() and not is_point(g) and not LTR_TEXT_RE.match(texts[id(g)])]
        attached: Set[int] = set()
        for g in row:
            if not is_point(g):
                continue
            centre = (g.x0 + g.x1) / 2
            holding = [h for h in hosts if h.x0 <= centre <= h.x1]
            if holding:
                host = min(holding, key=lambda h: abs((h.x0 + h.x1) / 2 - centre))
                texts[id(host)] += texts[id(g)]
                attached.add(id(g))
        return [g for g in row if id(g) not in attached]

    @staticmethod
    def _text(glyph: Glyph) -> str:
        """Text of a rebuilt glyph: the table's, else the layer's real character, else :data:`UNKNOWN`.

        :param glyph: The glyph.
        :return: Its text.
        """
        if glyph.text is not None:
            return glyph.text
        if not is_unmapped(ord(glyph.layer_char)):
            return fold_letters(glyph.layer_char)
        return UNKNOWN

    def page_substitutions(self, page: "fitz.Page", skip_fonts: Iterable[str] = ()) -> Dict[Tuple[str, float, float], str]:
        """Letters of the glyphs the text layer has no character for, and points it reports as other characters,
        outside the rebuilt fonts.

        A glyph the table names with a Hebrew or Arabic letter is substituted when the layer has no character for it.
        A glyph the table names with Hebrew points or accents only (:func:`is_hebrew_mark_text`) is substituted when
        the layer's character is none of those marks, no Hebrew or Arabic letter and no space (an ASCII letter, digit
        or symbol, or nothing): otherwise later cleaning, which drops Latin letters and digits, cuts the word there.
        A point that shares its position with another glyph (a point drawn at a letter's origin) is left alone: the
        caller replaces one character per position and could not tell which is the point's.

        :param page: PDF page.
        :param skip_fonts: Fonts handled by :meth:`page_spans`.
        :return: :func:`position_key` of the glyph -> its text (Hebrew or Arabic letters, or Hebrew marks).
        """
        skip = set(skip_fonts)
        out: Dict[Tuple[str, float, float], str] = {}
        marks = 0
        glyphs = self.page_glyphs(page)
        drawn = collections.Counter(position_key(g.font, g.x, g.y) for g in glyphs)
        for g in glyphs:
            if g.font in skip or g.text is None:
                continue
            key = position_key(g.font, g.x, g.y)
            if is_unmapped(ord(g.layer_char)) and (HEBREW_LETTER_RE.search(g.text) or ARABIC_LETTER_RE.search(g.text)):
                out[key] = g.text
            elif (is_hebrew_mark_text(g.text) and not is_hebrew_mark_text(g.layer_char) and not g.layer_char.isspace()
                  and not HEBREW_LETTER_RE.match(g.layer_char) and not ARABIC_LETTER_RE.match(g.layer_char)
                  and drawn[key] == 1):
                out[key] = g.text
                marks += 1
        self.stats["glyphs_substituted"] += len(out) - marks
        self.stats["points_substituted"] += marks
        return out
