"""Tests for naming PDF glyphs by their outlines (letters the text layer lacks)."""
from __future__ import annotations

import hashlib
import os
import unicodedata
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import fitz
import pytest
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

from src.datasets.indexing.legacy_store import pdf_glyphs as G
from src.datasets.indexing.legacy_store import pdf_transcriptions as P

Polygon = List[Tuple[int, int]]


def box(width: int, height: int, notch: int = 0) -> Polygon:
    """A rectangle (with an optional notch, so that shapes differ) as a glyph outline."""
    points = [(50, 0), (50 + width, 0), (50 + width, height), (50, height)]
    if notch:
        points.insert(2, (50 + width - notch, height // 2))
    return points


def build_font(path: Path, glyphs: Dict[str, Union[Polygon, List[Polygon], List[str]]], cmap: Dict[int, str],
               family: str = "TestGlyphs") -> Path:
    """Write a small TrueType font.

    :param path: Output file.
    :param glyphs: Glyph name -> outline: one polygon, a list of polygons (one contour each), or a list of glyph
        names (a composite of those glyphs, unmoved); an empty list is a glyph without an outline. Components must
        come before the composites that use them.
    :param cmap: Character -> glyph name.
    :param family: Family name.
    :return: The path.
    """
    builder = FontBuilder(1000, isTTF=True)
    builder.setupGlyphOrder([".notdef"] + list(glyphs))
    builder.setupCharacterMap(cmap)
    table = {}
    for name, outline in [(".notdef", [])] + list(glyphs.items()):
        pen = TTGlyphPen(table)                                      # the glyphs built so far (for components)
        if outline and isinstance(outline[0], str):                  # a composite: names of its components
            for component in outline:
                pen.addComponent(component, (1, 0, 0, 1, 0, 0))
            table[name] = pen.glyph()
            continue
        polygons = outline if outline and isinstance(outline[0], list) else ([outline] if outline else [])
        for polygon in polygons:
            pen.moveTo(polygon[0])
            for point in polygon[1:]:
                pen.lineTo(point)
            pen.closePath()
        table[name] = pen.glyph()
    builder.setupGlyf(table)
    builder.setupHorizontalMetrics({name: (600, 0) for name in table})
    builder.setupHorizontalHeader(ascent=800, descent=-200)
    builder.setupNameTable({"familyName": family, "styleName": "Regular"})
    builder.setupOS2()
    builder.setupPost()
    builder.save(str(path))
    return path


SHAPES: Dict[str, Polygon] = {
    "space": [], "theh_init": box(300, 500), "beh_medi": box(320, 420, 60), "teh_fina": box(280, 610, 90),
    "one": box(200, 700), "two": box(210, 690, 40), "alef": box(120, 650), "mem_final": box(400, 400, 120),
    "unknown": box(333, 333, 33)}
# the embedded font's character map sends every glyph to a private-use character: a text layer without letters
PRIVATE = {name: 0xE001 + k for k, name in enumerate(SHAPES)}
LETTERS = {"theh_init": "ث", "beh_medi": "ب", "teh_fina": "ت", "one": "1", "two": "2",
           "alef": "א", "mem_final": "ם"}


@pytest.fixture()
def fonts(tmp_path: Path) -> Tuple[Path, G.ReferenceFont]:
    """The embedded font's file and a reference index naming its outlines (all but the glyph ``unknown``)."""
    embedded = build_font(tmp_path / "embedded.ttf", SHAPES, {cp: name for name, cp in PRIVATE.items()})
    font = TTFont(embedded)
    texts = {G.outline_digest(font, name): text for name, text in LETTERS.items()}
    return embedded, G.ReferenceFont("reference.ttf", texts)


def make_pdf(font_file: Path, runs: Sequence[Tuple[float, float, str]], extra: Sequence[Tuple[float, float, str]] = ()
             ) -> "fitz.Document":
    """A one-page PDF drawing each run with the test font (and ``extra`` runs with a standard font).

    :param font_file: The embedded font.
    :param runs: ``(x, y, text)`` in the test font.
    :param extra: ``(x, y, text)`` in Helvetica.
    :return: The document, reopened from its bytes.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_font(fontname="tg", fontfile=str(font_file))
    for x, y, text in runs:
        page.insert_text((x, y), text, fontname="tg", fontsize=12)
    for x, y, text in extra:
        page.insert_text((x, y), text, fontname="helv", fontsize=12)
    return fitz.open("pdf", doc.tobytes())


def chars(*names: str) -> str:
    """The private-use characters of glyph names."""
    return "".join(chr(PRIVATE[name]) for name in names)


@pytest.mark.parametrize("points,name,text", [
    ([0x0627, 0xFE8D], "uni0627", "ا"),
    ([0xFBAD, 0xFEEC], "uni06BE.fina", "ه"),                 # the heh medial form, not heh doachashmee
    ([0xFBFF, 0xFEF4], "uni064A.medi", "ي"),
    ([0xFEFB], "uni06440627.isol", "لا"),
    ([], "uni06440627.fina", "لا"),
    ([], "uni0628.medi", "ب"),
    ([], "lillah", "لله"),
    ([0x05DD], "finalmem", "ם"),
    ([0x06CC, 0x064A], "yeh", "ي"),                          # a standard letter before an extended one
    ([], "glyph00123", None),
])
def test_glyph_text_prefers_standard_letters(points: List[int], name: str, text: str) -> None:
    """A reference glyph's text comes from its characters (presentation forms folded) or its name."""
    assert G.glyph_text(points, name) == text


def test_is_unmapped() -> None:
    """Control, private-use and replacement characters stand for glyphs without a letter."""
    assert all(G.is_unmapped(cp) for cp in (0x01, 0x1F, 0x7F, 0x85, 0xE001, 0xFFFD))
    assert not any(G.is_unmapped(cp) for cp in (0x20, 0x41, 0x05D0, 0x0627, 0xFE8D))


def test_outline_digest_follows_the_outline_only(tmp_path: Path) -> None:
    """Equal outlines under other names and glyph ids share a digest; an empty glyph has none."""
    a = TTFont(build_font(tmp_path / "a.ttf", {"x": box(300, 500), "gap": [], "y": box(300, 501)}, {0x41: "x"}))
    b = TTFont(build_font(tmp_path / "b.ttf", {"pad": box(10, 10), "other": box(300, 500)}, {0x42: "other"}))
    assert G.outline_digest(a, "x") == G.outline_digest(b, "other")
    assert G.outline_digest(a, "x") != G.outline_digest(a, "y")
    assert G.outline_digest(a, "gap") is None


def test_reference_font_load_indexes_by_outline(tmp_path: Path) -> None:
    """A font file is indexed by outline; of two glyphs with one outline the plainer text wins."""
    path = build_font(tmp_path / "ref.ttf", {"a": box(300, 500), "alpha": box(300, 500), "beh": box(200, 300)},
                      {0x41: "a", 0x0391: "alpha", 0xFE91: "beh"})
    ref = G.ReferenceFont.load(path)
    font = TTFont(path)
    assert ref.texts[G.outline_digest(font, "a")] == "A"
    assert ref.texts[G.outline_digest(font, "beh")] == "ب"


def test_font_table_uses_reference_then_learned(fonts: Tuple[Path, G.ReferenceFont]) -> None:
    """Glyphs are named by the reference font; one it lacks comes from the learned table."""
    embedded, reference = fonts
    font = TTFont(embedded)
    learned = {G.outline_digest(font, "unknown"): "ق"}
    table = G.font_table(embedded.read_bytes(), [reference], learned)
    by_name = {font.getGlyphName(gid): text for gid, text in table.texts.items()}
    assert by_name == {**LETTERS, "unknown": "ق"}
    assert (table.reference, table.n_reference, table.n_learned) == ("reference.ttf", len(LETTERS), 1)


def test_font_table_ignores_a_reference_that_is_not_the_font(fonts: Tuple[Path, G.ReferenceFont]) -> None:
    """A reference sharing only a few outlines names nothing; the learned table still applies."""
    embedded, reference = fonts
    few = G.ReferenceFont("other.ttf", dict(list(reference.texts.items())[:2]))
    table = G.font_table(embedded.read_bytes(), [few], {})
    assert table.texts == {} and table.reference is None


def test_arabic_line_is_rebuilt_right_to_left(fonts: Tuple[Path, G.ReferenceFont]) -> None:
    """Glyphs drawn left to right in visual order come out in reading order; a number keeps its own order."""
    embedded, reference = fonts
    doc = make_pdf(embedded, [(200, 100, chars("teh_fina", "beh_medi", "theh_init")), (150, 100, chars("one", "two"))])
    decoder = G.GlyphDecoder(doc, [reference], {})
    rebuilt, spans = decoder.page_spans(doc[0])
    assert len(rebuilt) == 1 and decoder.stats["glyphs_unknown"] == 0
    lines = P.page_lines(doc[0], 0, glyphs=decoder)
    assert [ln.text.strip() for ln in lines] == ["ثبت 12"]
    assert P.page_lines(doc[0], 0)[0].text != lines[0].text                 # the text layer alone has no letters


def test_unnamed_glyph_is_marked_unknown(fonts: Tuple[Path, G.ReferenceFont]) -> None:
    """A glyph neither source names is the replacement character and is counted."""
    embedded, reference = fonts
    doc = make_pdf(embedded, [(200, 100, chars("teh_fina", "unknown", "theh_init"))])
    decoder = G.GlyphDecoder(doc, [reference], {})
    line = P.page_lines(doc[0], 0, glyphs=decoder)[0].text.strip()
    assert line == "ث" + G.UNKNOWN + "ت" and decoder.stats["glyphs_unknown"] == 1


def test_spaces_of_another_font_over_the_glyphs_are_left_out(fonts: Tuple[Path, G.ReferenceFont]) -> None:
    """A run of spaces that another font draws over a rebuilt word does not split the word."""
    embedded, reference = fonts
    doc = make_pdf(embedded, [(200, 100, chars("teh_fina", "beh_medi", "theh_init"))], extra=[(204, 100, "  ")])
    decoder = G.GlyphDecoder(doc, [reference], {})
    assert P.page_lines(doc[0], 0, glyphs=decoder)[0].text.strip() == "ثبت"


def test_final_letter_is_substituted_in_place(fonts: Tuple[Path, G.ReferenceFont]) -> None:
    """Outside the rebuilt fonts an unmapped character takes its glyph's letter, only when asked to."""
    embedded, reference = fonts
    doc = make_pdf(embedded, [(200, 100, chars("alef", "mem_final"))])
    decoder = G.GlyphDecoder(doc, [reference], {})
    assert decoder.page_spans(doc[0])[0] == set()                           # no Arabic letter: nothing is rebuilt
    assert "ם" not in P.page_lines(doc[0], 0, glyphs=decoder)[0].text
    line = P.page_lines(doc[0], 0, glyphs=decoder, glyph_finals=True)[0].text
    assert "ם" in line and "א" in line and chr(PRIVATE["mem_final"]) not in line


def test_ingest_reads_glyph_text_and_flags_the_document(fonts: Tuple[Path, G.ReferenceFont], tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``glyph_text`` the folios hold the letters of the glyphs and the document is flagged."""
    embedded, reference = fonts
    doc = make_pdf(embedded, [(200, 100 + 20 * k, chars("teh_fina", "beh_medi", "theh_init")) for k in range(4)])
    path = tmp_path / "transcription_test.pdf"
    doc.save(str(path))
    monkeypatch.setattr(G, "load_references", lambda: (reference,))
    monkeypatch.setattr(G, "load_learned", lambda: {})
    folios = P.ingest(str(path), None, None, glyph_text=True)
    assert folios and all(f["flags"].get("glyph_text") for f in folios)
    assert not any(f["flags"].get("glyphs_unknown") for f in folios)
    assert [ln.strip() for ln in folios[0]["raw_lines"]] == ["ثبت"] * 4
    plain = P.ingest(str(path), None, None)
    assert not any("ث" in ln for f in plain for ln in f["raw_lines"])
    assert not any(f["flags"].get("glyph_text") for f in plain)


# ---- precomposed pointed letters (named from the font's own outlines) ----------------------------------------------

def dot(x: int, y: int, side: int = 60) -> Polygon:
    """A square contour (a dot) with its lower left corner at ``(x, y)``."""
    return [(x, y), (x + side, y), (x + side, y + side), (x, y + side)]


BET, KAF, SHIN, HE = box(400, 600, 80), box(380, 620, 50), box(500, 610, 100), box(420, 640, 70)
POINTED: Dict[str, Union[Polygon, List[Polygon], List[str]]] = {
    "space": [], "bet": BET, "kaf": KAF, "shin": SHIN, "he": HE, "het": HE,      # he and het: one outline (a fault)
    "dagesh": dot(220, 280), "bet_dagesh": [BET, dot(220, 280)], "shin_dots": [SHIN, dot(500, 640), dot(250, 280)],
    "kaf_dagesh_moved": [[(x + 6, y) for x, y in KAF], dot(230, 290)], "he_dagesh": [HE, dot(230, 300)],
    "bet_big": [BET, box(300, 300)], "bet_composite": ["bet", "dagesh"], "qubuts": [dot(100, -250), dot(250, -200)],
    "holam_alone": dot(300, 800), "bar_above": [(50, 700), (500, 700), (500, 950), (50, 950)],
    "bet_patah_off": [[(x, y + (1 if k == 2 else 0)) for k, (x, y) in enumerate(BET)], box(200, 60)]}
LAYER_CHARS = {"space": " ", "bet": "ב", "kaf": "כ", "shin": "ש", "he": "ה", "het": "ח", "bet_dagesh": "9",
               "shin_dots": "=", "kaf_dagesh_moved": "3", "he_dagesh": "K", "bet_big": "@", "bet_composite": "G",
               "qubuts": G.UNKNOWN, "holam_alone": G.UNKNOWN, "bar_above": G.UNKNOWN, "bet_patah_off": "1"}


@pytest.fixture()
def pointed_font(tmp_path: Path) -> TTFont:
    """A font with plain letters, precomposed pointed letters and a point below the baseline."""
    return TTFont(build_font(tmp_path / "pointed.ttf", POINTED, {0x41 + k: n for k, n in enumerate(POINTED)}))


def test_letter_with_marks_takes_dots_only() -> None:
    """A letter with one or two dots fits it; a large extra contour, or a letter moved too far, does not."""
    bet = [tuple((x, y, 1) for x, y in BET)]
    dots = [tuple((x, y, 1) for x, y in dot(220, 280)), tuple((x, y, 1) for x, y in dot(100, 700))]
    assert G.letter_with_marks(bet + dots[:1], bet)
    assert G.letter_with_marks(dots + bet, bet)
    assert not G.letter_with_marks(bet, bet)                                       # no mark: the letter itself
    assert not G.letter_with_marks(bet + [tuple((x, y, 1) for x, y in box(300, 300))], bet)
    assert not G.letter_with_marks(bet + dots + dots[:1], bet)                     # three extra contours
    far = [tuple((x + 200, y, 1) for x, y in BET)]
    assert not G.letter_with_marks(far + dots[:1], bet)


def test_letter_with_marks_tolerates_one_unit() -> None:
    """A point of the letter repeated one font unit off still fits; three units off does not."""
    bet = [tuple((x, y, 1) for x, y in BET)]
    dot_contour = tuple((x, y, 1) for x, y in dot(220, 280))
    for shift, fits in ((1, True), (3, False)):
        moved = [tuple((x, y + (shift if k == 2 else 0), 1) for k, (x, y) in enumerate(BET))]
        assert G.letter_with_marks(moved + [dot_contour], bet) is fits


def test_pointed_letters_names_letter_plus_dots(pointed_font: TTFont) -> None:
    """Letter + dot, letter + two dots, a slightly moved letter + dot and a composite take the plain letter."""
    gid = {name: k for k, name in enumerate(pointed_font.getGlyphOrder())}
    layer = {gid[n]: c for n, c in LAYER_CHARS.items()}
    letters, points = G.pointed_letters(pointed_font, layer, {})
    names = {pointed_font.getGlyphName(g): t for g, t in letters.items()}
    assert names == {"bet_dagesh": "ב", "shin_dots": "ש", "kaf_dagesh_moved": "כ", "bet_composite": "ב",
                     "bet_patah_off": "ב"}
    # a mark-sized glyph wholly above the letters is a point too; a large one there is not
    assert points == {gid["qubuts"], gid["holam_alone"]}


def test_pointed_letters_leaves_ambiguous_and_large_unnamed(pointed_font: TTFont) -> None:
    """A glyph fitting two letters (he/het share an outline) or carrying a large contour stays unnamed."""
    gid = {name: k for k, name in enumerate(pointed_font.getGlyphOrder())}
    layer = {gid[n]: c for n, c in LAYER_CHARS.items()}
    letters, _ = G.pointed_letters(pointed_font, layer, {})
    assert gid["he_dagesh"] not in letters and gid["bet_big"] not in letters
    # a glyph the layer already reports as a Hebrew character is the layer's, and nothing is named without letters
    layer[gid["bet_dagesh"]] = "בּ"                                           # HEBREW LETTER BET WITH DAGESH
    assert gid["bet_dagesh"] not in G.pointed_letters(pointed_font, layer, {})[0]
    no_letters = {g: c for g, c in layer.items() if not G.PLAIN_HEBREW_RE.match(c)}
    assert G.pointed_letters(pointed_font, no_letters, {}) == ({}, set())


def test_attach_points_joins_a_point_to_its_letter() -> None:
    """A point drawn on a letter joins it; one between letters stays apart; a Latin run takes none."""
    def glyph(x0: float, x1: float) -> G.Glyph:
        return G.Glyph("f", 12, 0, "?", x0, x1, x0, 100)
    letter, point, other, digit, loose = glyph(10, 17), glyph(12, 14), glyph(3, 10), glyph(30, 36), glyph(31, 32)
    gap_point = glyph(18, 19)
    texts = {id(letter): "ב", id(point): "ָ", id(other): "כ", id(digit): "2", id(loose): "ְ", id(gap_point): "ִ"}
    row = G.GlyphDecoder.attach_points([other, letter, point, gap_point, digit, loose], texts)
    assert row == [other, letter, gap_point, digit, loose]
    assert texts[id(letter)] == "בָ" and texts[id(digit)] == "2"


PLAIN_CMAP = {0x05D1: "bet", 0x05DB: "kaf", 0x20: "space"}


def test_pointed_font_is_rebuilt_from_its_glyphs(tmp_path: Path) -> None:
    """A precomposed letter that the layer reports as a digit reads as its letter, in reading order."""
    glyphs = {n: POINTED[n] for n in ("space", "bet", "kaf", "dagesh", "bet_dagesh")}
    font_file = build_font(tmp_path / "pointed.ttf", glyphs, {**PLAIN_CMAP, ord("9"): "bet_dagesh"})
    doc = make_pdf(font_file, [(200, 100, "בכ9")])                 # drawn left to right: reads bet-dagesh, kaf, bet
    decoder = G.GlyphDecoder(doc, [], {})
    rebuilt, _ = decoder.page_spans(doc[0])
    assert rebuilt and decoder.stats["glyphs_pointed"] == 1
    assert "9" in "".join(ln.text for ln in P.page_lines(doc[0], 0, glyph_finals=True))
    line = P.page_lines(doc[0], 0, glyphs=decoder, glyph_finals=True)[0].text.strip()
    assert line == "בכב"


def test_font_without_precomposed_letters_is_not_rebuilt(tmp_path: Path) -> None:
    """A Hebrew font whose dotted letter the layer reports as Hebrew (a presentation form) is left alone."""
    glyphs = {n: POINTED[n] for n in ("space", "bet", "kaf", "dagesh", "bet_dagesh")}
    font_file = build_font(tmp_path / "plain.ttf", glyphs, {**PLAIN_CMAP, 0xFB31: "bet_dagesh"})
    doc = make_pdf(font_file, [(200, 100, "בכבּ")])
    decoder = G.GlyphDecoder(doc, [], {})
    assert decoder.page_spans(doc[0])[0] == set() and decoder.page_substitutions(doc[0]) == {}
    assert all(not t.pointed and not t.points for t in decoder.tables.values())
    assert ([ln.text for ln in P.page_lines(doc[0], 0, glyphs=decoder, glyph_finals=True)]
            == [ln.text for ln in P.page_lines(doc[0], 0)])


# A locally held pointed edition PDF (T-S Misc.35.54); set POINTED_EDITION_PDF to its path to run the check.
_POINTED_ENV = os.environ.get("POINTED_EDITION_PDF")
POINTED_PDF = Path(_POINTED_ENV) if _POINTED_ENV else None
# sha256 of the poem's first line with points removed (the edition text itself is not kept in the repository)
POINTED_FIRST_LINE_SHA256 = "d6ebd89b44d4e5b84f2fab4f733f3dee1e2ffb084e8d7b79f7f859d68f8ddbb2"


@pytest.mark.skipif(POINTED_PDF is None or not POINTED_PDF.exists(), reason="POINTED_EDITION_PDF is not set or not mounted")
def test_pointed_edition_reads_whole_words() -> None:
    """T-S Misc.35.54 (pointed, precomposed dotted letters): the poem's first line reads as the manuscript does."""
    doc = fitz.open(str(POINTED_PDF))
    lines = P.page_lines(doc[3], 3, glyphs=G.GlyphDecoder(doc), glyph_finals=True)
    plain = [" ".join("".join(c for c in ln.text if unicodedata.category(c) != "Mn").split()) for ln in lines]
    assert POINTED_FIRST_LINE_SHA256 in {hashlib.sha256(p.encode("utf-8")).hexdigest() for p in plain}


def test_learned_space_separates_words(fonts: Tuple[Path, G.ReferenceFont]) -> None:
    """A learned entry whose text is one space (an editor's maqaf) is kept as a space between two words."""
    embedded, reference = fonts
    learned = {G.outline_digest(TTFont(embedded), "unknown"): " "}
    doc = make_pdf(embedded, [(200, 100, chars("teh_fina", "unknown", "theh_init"))])
    decoder = G.GlyphDecoder(doc, [reference], learned)
    assert G.font_table(embedded.read_bytes(), [reference], learned).texts[list(SHAPES).index("unknown") + 1] == " "
    rebuilt, spans = decoder.page_spans(doc[0])
    assert rebuilt and " " in [s.text for s in spans] and decoder.stats["glyphs_unknown"] == 0
    assert P.page_lines(doc[0], 0, glyphs=decoder)[0].text.strip() == "ث ت"


MARKED: Dict[str, Union[Polygon, List[Polygon], List[str]]] = {
    "space": [], "alef": box(120, 650), "bet": BET, "hiriq": dot(150, -200),
    "patah": [(60, -200), (400, -200), (400, -150), (60, -150)], "qamats": [(60, -250), (400, -250), (230, -150)]}
# one glyph per character: the layer reports a glyph by one character only
MARKED_CMAP = {0x20: "space", 0x05D0: "alef", 0x05D1: "bet", ord("Q"): "hiriq", ord("@"): "patah", 0x05D2: "qamats"}


@pytest.fixture()
def marked(tmp_path: Path) -> Tuple[Path, G.ReferenceFont]:
    """A font whose points the layer reports as ``Q`` and ``@``, and an outline table naming them as points."""
    path = build_font(tmp_path / "marked.ttf", MARKED, MARKED_CMAP)
    font = TTFont(path)
    names = {"alef": "א", "bet": "ב", "hiriq": "ִ", "patah": "ַ", "qamats": "ָ"}
    return path, G.ReferenceFont("reference.ttf", {G.outline_digest(font, n): t for n, t in names.items()})


def test_point_reported_as_ascii_is_substituted(marked: Tuple[Path, G.ReferenceFont]) -> None:
    """A point the layer reports as an ASCII character takes its mark; one the layer reports as a letter does not."""
    path, reference = marked
    doc = make_pdf(path, [(200, 100, "אQב@"), (200, 130, "אג")])
    decoder = G.GlyphDecoder(doc, [], reference.texts)
    marks = sorted(decoder.page_substitutions(doc[0]).values())
    assert marks == ["ִ", "ַ"] and decoder.stats["points_substituted"] == 2
    lines = [ln.text for ln in P.page_lines(doc[0], 0, glyphs=decoder, glyph_finals=True)]
    assert not any(c in "".join(lines) for c in "Q@") and "ִ" in lines[0] and "ג" in lines[1]
    assert "ָ" not in "".join(lines)                                       # the layer's letter is kept
    assert [ln.text for ln in P.page_lines(doc[0], 0, glyphs=G.GlyphDecoder(doc, [], reference.texts))] != lines


def test_point_sharing_a_letter_position_is_left_alone(marked: Tuple[Path, G.ReferenceFont]) -> None:
    """A point drawn at a letter's origin is not substituted (the caller could replace the letter instead)."""
    path, reference = marked
    doc = make_pdf(path, [(200, 100, "Q"), (200, 100, "ב")])
    decoder = G.GlyphDecoder(doc, [], reference.texts)
    assert decoder.page_substitutions(doc[0]) == {}
    assert "ב" in P.page_lines(doc[0], 0, glyphs=decoder, glyph_finals=True)[0].text
