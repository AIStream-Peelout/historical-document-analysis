# File name: test_legacy_store_pdf_transcriptions.py
# Date: 10/30/26
# Author: Isaac Godfried. Coded by Claude.
"""Unit tests for the legacy-store transcription PDF ingester (``pdf_transcriptions``).

Every PDF is generated here with PyMuPDF from synthetic text (common Hebrew words composed for
the test, never transcription text). Hebrew is drawn right to left with ``TextWriter`` so that, as
in the real downloads, each span's text extracts in logical order while its glyphs run leftwards.
Covers junk detection, copy grouping, header parsing, folio splitting (line order, notes cut,
page fallback), finals written as ``$`` in a second font (font check, inference, repair count),
line-number stripping and wrapped-line joins, and image pairing with Firefox ``(n)`` copies
(unverified candidates only). The verifier's fixes: editor's labels and notes, side-sequence gaps
and line-count outliers, excerpt labels before headings, page-row exclusions. The old style:
style detection, side-file names, column markers, side records cleaned with one finals mapping,
side grouping, provenance signals. The second verification pass: editor's notes on the physical
page and the document-level note rule, edition page references, headings naming another shelf
mark, side re-downloads and suffix-order numbering, near-duplicate texts, scrambled side files,
credit kinds.
"""
import collections
import os
import pathlib
import random
from typing import Dict, List, Optional, Sequence, Tuple

import fitz
import pytest

from src.datasets.indexing.legacy_store import pdf_transcriptions as P

HEBREW_FONT = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
SYMBOL_FONT = "/System/Library/Fonts/Supplemental/Courier New.ttf"
pytestmark = pytest.mark.skipif(not (os.path.exists(HEBREW_FONT) and os.path.exists(SYMBOL_FONT)),
                                reason="system TrueType fonts with Hebrew glyphs not found")
RIGHT = 540.0
SIZE = 12.0


@pytest.fixture(scope="module")
def fonts() -> Dict[str, "fitz.Font"]:
    """The two fonts: Hebrew-capable text font and a second font for symbols.

    :return: ``{"heb": Font, "sym": Font}``.
    """
    return {"heb": fitz.Font(fontfile=HEBREW_FONT), "sym": fitz.Font(fontfile=SYMBOL_FONT)}


def draw_line(page: "fitz.Page", y: float, parts: Sequence[Tuple[str, str]], fonts: Dict[str, "fitz.Font"]) -> None:
    """Draw one line right-aligned, its parts in reading order from right to left.

    :param page: Page.
    :param y: Baseline.
    :param parts: ``(text, font key)`` in logical order.
    :param fonts: Fonts by key.
    """
    tw = fitz.TextWriter(page.rect)
    x = RIGHT
    for text, key in parts:
        font = fonts[key]
        x -= font.text_length(text, fontsize=SIZE)
        tw.append((x, y), text, font=font, fontsize=SIZE, right_to_left=bool(P.RTL_CHAR_RE.search(text)))
    tw.write_text(page)


def draw_ltr(page: "fitz.Page", y: float, text: str, fonts: Dict[str, "fitz.Font"], x: float = 150.0) -> None:
    """Draw one left-to-right line.

    :param page: Page.
    :param y: Baseline.
    :param text: Latin text.
    :param fonts: Fonts by key.
    :param x: Left edge.
    """
    tw = fitz.TextWriter(page.rect)
    tw.append((x, y), text, font=fonts["heb"], fontsize=SIZE)
    tw.write_text(page)


def save(doc: "fitz.Document", path: str) -> str:
    """Save a document and return its path.

    :param doc: Document.
    :param path: Target path.
    :return: The path.
    """
    doc.save(path)
    return path


FOLIO_1R = ["שלום עולם גדול", "אדם טוב בבית", "היה איש אחד"]
FOLIO_1V = ["כל העם הלך", "ספר תורה קטן"]


def header_pdf(fonts: Dict[str, "fitz.Font"]) -> "fitz.Document":
    """Header (institution, shelf mark, title) and two folios on two pages, notes after ``הערות:``.

    :param fonts: Fonts.
    :return: The document.
    """
    doc = fitz.open()
    p1 = doc.new_page()
    draw_ltr(p1, 72, "St. Petersburg, Russian National Library, Firkovich", fonts)
    draw_ltr(p1, 86, "Evr. IIA 223.2", fonts, x=250)
    draw_line(p1, 100, [("מדרש לדוגמה", "heb")], fonts)
    draw_ltr(p1, 128, "1r", fonts, x=525)
    for i, text in enumerate(FOLIO_1R, 1):
        draw_line(p1, 128 + 16 * i, [(str(i), "heb"), (" " + text, "heb")], fonts)
    p2 = doc.new_page()
    draw_ltr(p2, 86, "1v", fonts, x=525)
    for i, text in enumerate(FOLIO_1V, 1):
        draw_line(p2, 86 + 16 * i, [(str(i), "heb"), (" " + text, "heb")], fonts)
    draw_line(p2, 200, [("הערות:", "heb")], fonts)
    draw_line(p2, 216, [("הוספה מעל השורה", "heb")], fonts)
    return doc


# ───────────────────────────── junk and copies ─────────────────────────────
def junk_pdf() -> "fitz.Document":
    """The store's error page.

    :return: The document.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "A minor technical problem was encountered while", fontname="helv", fontsize=11)
    page.insert_text((72, 86), "displaying the requested image.", fontname="helv", fontsize=11)
    return doc


def test_junk_detection(fonts: Dict[str, "fitz.Font"]) -> None:
    """The error page is junk (its phrase spans two lines); a transcription is not.

    :param fonts: Fonts fixture.
    """
    assert P.is_junk(junk_pdf())
    assert not P.is_junk(header_pdf(fonts))


def test_dedupe_copies_and_text_equal_resave(fonts: Dict[str, "fitz.Font"], tmp_path: pathlib.Path) -> None:
    """Byte-identical copies and a re-save with other bytes share one digest; the earliest is canonical.

    :param fonts: Fonts fixture.
    :param tmp_path: Temporary folder.
    """
    data = header_pdf(fonts).tobytes()
    a, b = tmp_path / "transcription_aaa.pdf", tmp_path / "transcription_response_aaa.pdf"
    a.write_bytes(data)
    b.write_bytes(data)
    resaved = fitz.open(stream=data, filetype="pdf")
    resaved.set_metadata({"title": "other bytes"})
    c = save(resaved, str(tmp_path / "transcription_manual_x.pdf"))
    j = save(junk_pdf(), str(tmp_path / "transcription_junk.pdf"))
    assert open(c, "rb").read() != data
    os.utime(b, (1000, 1000))
    os.utime(a, (2000, 2000))
    os.utime(c, (3000, 3000))
    os.utime(j, (4000, 4000))
    groups = P.dedupe_pdfs([str(a), str(b), c, j])
    assert len(groups) == 2
    real = next(g for g in groups if not g["junk"])
    assert real["canonical"] == str(b)
    assert real["copies"] == [str(b), str(a), c]
    with fitz.open(str(a)) as da, fitz.open(c) as dc:
        assert P.content_digest(da) == P.content_digest(dc)
    assert next(g for g in groups if g["junk"])["copies"] == [j]


# ───────────────────────────── header and folios ─────────────────────────────
def test_header_parse(fonts: Dict[str, "fitz.Font"]) -> None:
    """Institution, shelf mark, title and series key are read from page 1.

    :param fonts: Fonts fixture.
    """
    doc = header_pdf(fonts)
    lines = [ln.text for ln in P.page_lines(doc[0], 0)]
    h = P.parse_header(lines)
    assert h is not None
    assert h["institution"] == "St. Petersburg, Russian National Library, Firkovich"
    assert h["shelfmark"] == "Evr. IIA 223.2"
    assert h["title"] == "מדרש לדוגמה"
    assert h["series_key"] == ("Yevr_II_A", "223")
    assert h["header_lines"] == 3


def test_folio_split_keeps_line_order_and_cuts_notes(fonts: Dict[str, "fitz.Font"]) -> None:
    """Two marker folios; lines in reading order with their numbers; notes after הערות: dropped.

    :param fonts: Fonts fixture.
    """
    doc = header_pdf(fonts)
    preamble, folios = P.split_document(doc)
    assert [ln.text for ln in preamble][1:] == ["Evr. IIA 223.2", "מדרש לדוגמה"]
    assert [f["folio"] for f in folios] == ["1r", "1v"]
    assert [f["page_index"] for f in folios] == [0, 1]
    assert folios[0]["lines"] == [f"{i} {t}" for i, t in enumerate(FOLIO_1R, 1)]
    assert folios[1]["lines"] == [f"{i} {t}" for i, t in enumerate(FOLIO_1V, 1)]
    assert folios[1]["notes_lines"] == 2
    assert all(f["marker_kind"] == "latin" for f in folios)
    cleaned = P.clean_folio(folios[0]["lines"], None)
    assert cleaned["cleaned_lines"] == FOLIO_1R
    assert cleaned["letters"] == sum(len(t.replace(" ", "")) for t in FOLIO_1R)
    assert cleaned["flags"]["line_numbers_stripped"] == 3


def test_headerless_one_page_falls_back_to_page_folio(fonts: Dict[str, "fitz.Font"]) -> None:
    """No header, no heading: one folio for the page, flagged.

    :param fonts: Fonts fixture.
    """
    doc = fitz.open()
    page = doc.new_page()
    for i, text in enumerate(["ברוך אתה", "בכל יום ויום"]):
        draw_line(page, 80 + 16 * i, [(text, "heb")], fonts)
    assert P.parse_header([ln.text for ln in P.page_lines(page, 0)]) is None
    folios = P.split_folios(doc)
    assert len(folios) == 1
    assert folios[0]["folio"] == "p1"
    assert folios[0]["flags"]["no_folio_markers"] == 1
    assert folios[0]["lines"] == ["ברוך אתה", "בכל יום ויום"]


def test_hebrew_headings_and_rest_text(fonts: Dict[str, "fitz.Font"]) -> None:
    """Hebrew headings split folios; a ``3ב :`` heading keeps its trailing text as the first line.

    :param fonts: Fonts fixture.
    """
    doc = fitz.open()
    page = doc.new_page()
    draw_line(page, 72, [("דף א צד א", "heb")], fonts)
    draw_line(page, 88, [("שלום עולם גדול", "heb")], fonts)
    draw_line(page, 104, [("דף א צד ב", "heb")], fonts)
    draw_line(page, 120, [("אדם טוב", "heb")], fonts)
    draw_line(page, 136, [("3", "heb"), ("ב :ספר תורה", "heb")], fonts)
    folios = P.split_folios(doc)
    assert [f["folio"] for f in folios] == ["1r", "1v", "3v"]
    assert folios[2]["lines"][0].endswith("ספר תורה")
    assert [f["marker_kind"] for f in folios] == ["daf", "daf", "colon_prefix"]


# ───────────────────────────── finals in a second font ─────────────────────────────
FINAL_WORDS = ["שלום", "עולם", "חכם", "אדם", "יום", "מקום"]
PLAIN_WORDS = ["טוב", "היה", "כל", "על", "אשר", "גדול"]


def finals_pdf(fonts: Dict[str, "fitz.Font"]) -> "fitz.Document":
    """A page whose final mem is drawn as ``$`` in a second font (no real finals anywhere).

    :param fonts: Fonts.
    :return: The document.
    """
    doc = fitz.open()
    page = doc.new_page()
    for row in range(3):
        parts: List[Tuple[str, str]] = []
        for k, (fw, pw) in enumerate(zip(FINAL_WORDS, PLAIN_WORDS)):
            if (k + row) % 3 == 0:
                continue
            parts.append((("" if not parts else " ") + fw[:-1], "heb"))
            parts.append(("$", "sym"))
            parts.append((" " + pw, "heb"))
        draw_line(page, 80 + 16 * row, parts, fonts)
    return doc


def test_symbol_finals_font_check_and_repair(fonts: Dict[str, "fitz.Font"]) -> None:
    """``$`` in Courier stands after Hebrew letters; the lexicon maps it to ם; every one is repaired.

    :param fonts: Fonts fixture.
    """
    doc = finals_pdf(fonts)
    folios = P.split_folios(doc)
    assert len(folios) == 1
    fc = P.font_check(folios[0]["spans"])
    n_symbols = "".join(folios[0]["lines"]).count("$")
    assert n_symbols == 12
    assert fc["symbol_finals_by_font"] == {"CourierNewPSMT": {"$": n_symbols}}
    assert fc["letter_fonts"] == ["TimesNewRomanPSMT"]
    assert fc["real_finals"] == 0
    assert fc["finals_missing"]
    lexicon = collections.Counter({w: 3 for w in FINAL_WORDS + PLAIN_WORDS})
    text = "\n".join(folios[0]["lines"])
    assert P.infer_document_finals(text, lexicon) == {"$": "ם"}
    cleaned = P.clean_folio(folios[0]["lines"], lexicon)
    assert cleaned["finals_repaired"] == n_symbols
    assert cleaned["finals_mapping"] == {"$": "ם"}
    joined = " ".join(cleaned["cleaned_lines"])
    assert "$" not in joined
    assert sorted(set(joined.split())) == sorted(set(FINAL_WORDS + PLAIN_WORDS))


def test_finals_inference_off_when_finals_present() -> None:
    """With real finals present, ``$`` is a transcription mark: no mapping is inferred."""
    words = " ".join(["שלום", "עולם", "טו$ב", "חכם", "אדם"] * 5)
    lexicon = collections.Counter({"שלום": 1, "עולם": 1, "טוב": 1})
    assert P.infer_document_finals(words, lexicon) == {}


def test_inference_handles_bang_and_comma_symbols() -> None:
    """``!`` and ``,`` (outside the committed candidate class) are inferred through proxies."""
    words = ["תו!", "לכ!", "מל!", "חו,", "אר,", "קי,", "טוב", "היה", "אשר", "כל", "על", "גדול", "ספר", "איש", "בית",
             "בא", "יצא"]
    lexicon = collections.Counter({w: 2 for w in ["תוך", "לכך", "מלך", "חוץ", "ארץ", "קיץ"]})
    mapping = P.infer_document_finals(" ".join(words * 2), lexicon)
    assert mapping == {"!": "ך", ",": "ץ"}


# ───────────────────────────── structure ─────────────────────────────
def test_wrapped_lines_join_only_with_dense_numbering() -> None:
    """Dense 1, 2, 3 numbering joins unnumbered wraps; every-fifth numbering joins nothing."""
    dense = ["1.שלום עולם", "גדול", "2.אדם טוב", "3.היה איש", "אחד"]
    lines, flags = P.structure_lines(dense)
    assert lines == ["שלום עולם גדול", "אדם טוב", "היה איש אחד"]
    assert flags["wrapped_lines_joined"] == 2
    sparse = ["שלום עולם", "גדול", "5 אדם טוב", "היה איש", "10 אחד"]
    lines, flags = P.structure_lines(sparse)
    assert lines == ["שלום עולם", "גדול", "אדם טוב", "היה איש", "אחד"]
    assert "wrapped_lines_joined" not in flags


def test_reattach_geresh_only_when_displaced_form_dominates() -> None:
    """``אמ 'לו`` is repaired when that form dominates; a text using the regular form is untouched."""
    assert P.reattach_geresh("אמ 'לו כת 'שם ר 'יוסי") == ("אמ' לו כת' שם ר' יוסי", 3)
    regular = "אמ' לו כת' שם ר 'יוסי"
    assert P.reattach_geresh(regular) == (regular, 0)


def test_quote_separator_share() -> None:
    """Gershayim inside abbreviations do not count; quotes after words do."""
    assert P.quote_separator_share('ב"ד אמר ר"ל שלום') == 0.0
    assert P.quote_separator_share('שלום" עולם" גדול טוב') == 0.5


def test_join_keeps_control_character_finals() -> None:
    """A control-character final at the end of a wrapped line survives the join."""
    lines, _ = P.structure_lines(["1 שלו" + chr(0x1F), "עולם", "2 טוב", "3 היה"])
    assert lines[0] == "שלו" + chr(0x1F) + " עולם"


def test_notes_cut_by_rule_and_legend(fonts: Dict[str, "fitz.Font"]) -> None:
    """An underscore rule and a sigla legend open notes up to the next heading.

    :param fonts: Fonts fixture.
    """
    doc = fitz.open()
    page = doc.new_page()
    draw_ltr(page, 72, "1r", fonts, x=525)
    draw_line(page, 88, [("שלום עולם", "heb")], fonts)
    draw_ltr(page, 104, "_" * 30, fonts)
    draw_line(page, 120, [("הערה על השורה", "heb")], fonts)
    draw_ltr(page, 136, "1v", fonts, x=525)
    draw_line(page, 152, [("אדם טוב", "heb")], fonts)
    draw_line(page, 168, [("מקרא סימנים", "heb")], fonts)
    draw_line(page, 184, [("אות מסופקת", "heb")], fonts)
    folios = P.split_folios(doc)
    assert [f["lines"] for f in folios] == [["שלום עולם"], ["אדם טוב"]]
    assert [f["notes_lines"] for f in folios] == [2, 2]


# ───────────────────────────── shelf marks and markers ─────────────────────────────
@pytest.mark.parametrize("mark,key", [
    ("Evr. IIA 223.2", ("Yevr_II_A", "223")),
    ("Evr. II A 227.6", ("Yevr_II_A", "227")),
    ("Evr. Arab. II 1201", ("Yevr_Arab_II", "1201")),
    ("St. Petersburg, St. Peterburg: Yevr.-Arab. II 1252", ("Yevr_Arab_II", "1252")),
    ("Yevr Arab II 1201", ("Yevr_Arab_II", "1201")),
    ("Yevr Arab I 1", ("Yevr_Arab_I", "1")),
    ("Yevr II A 293 1 a", ("Yevr_II_A", "293")),
    ("T-S 8J5.1", None),
    ("Evr. I 12", None),
])
def test_series_key(mark: str, key: Optional[Tuple[str, str]]) -> None:
    """Shelf marks map to the image series through the explicit table, None when unknown.

    :param mark: Shelf mark.
    :param key: Expected series key.
    """
    assert P.series_key_for(mark) == key


def test_shelfmark_from_filename() -> None:
    """Manual file names carry the shelf mark; uuid names do not."""
    assert P.shelfmark_from_filename("transcription_manual_St_Petersburg_St_Peterburg_Yevr_Arab_II_1201.pdf") == \
        "Yevr Arab II 1201"
    assert P.shelfmark_from_filename("transcription_manual_Evr. IIA 223.2.pdf") == "Evr. IIA 223.2"
    assert P.shelfmark_from_filename("transcription_0573c6a5-23c7-4601-8090-b6e5ab3de69b.pdf") is None


@pytest.mark.parametrize("line,label,kind", [
    ("1r", "1r", "latin"),
    ("12v ", "12v", "latin"),
    ("דף א צד ב  '", "1v", "daf"),
    ("דף ה עמוד א", "5r", "daf"),
    (chr(5) + 'ד1 א" ע   (מהד', "1r", "amud_visual"),
    ('ד2" עא', "2r", "amud_visual"),
    ("5   א", "5r", "numeric"),
    (":  13 :  ב", "13v", "numeric"),
    ("3ב :שלום", "3v", "colon_prefix"),
    ("6:  אשלום", "6r", "numeric_colon"),
    ("1 שלום עולם", None, None),
    ("4שלום", None, None),
])
def test_match_folio_marker(line: str, label: Optional[str], kind: Optional[str]) -> None:
    """Heading forms seen in the downloads; numbered text lines are not headings.

    :param line: Synthetic line.
    :param label: Expected folio label.
    :param kind: Expected heading kind.
    """
    m = P.match_folio_marker(line)
    assert (m.label if m else None, m.kind if m else None) == (label, kind)


def test_marker_rest_keeps_control_characters() -> None:
    """Text after a heading is cut from the original line, legacy-font symbols included."""
    m = P.match_folio_marker("6:  אשלו" + chr(0x0B) + " טוב")
    assert m.rest == "שלו" + chr(0x0B) + " טוב"


# ───────────────────────────── images ─────────────────────────────
def test_pair_images_dedupes_firefox_copies(tmp_path: pathlib.Path) -> None:
    """Identical ``(n)`` copies collapse; of distinct files the largest wins; other labels miss.

    :param tmp_path: Temporary folder.
    """
    (tmp_path / "Yevr_II_A_223_1r.jpg").write_bytes(b"A" * 100)
    (tmp_path / "Yevr_II_A_223_1r(1).jpg").write_bytes(b"A" * 100)
    (tmp_path / "Yevr_II_A_223_1v.jpg").write_bytes(b"B" * 50)
    (tmp_path / "Yevr_II_A_223_1v(1).jpg").write_bytes(b"C" * 80)
    (tmp_path / "Yevr_II_A_2230_1r.jpg").write_bytes(b"D" * 10)
    key = ("Yevr_II_A", "223")
    r = P.pair_images(key, "1r", str(tmp_path))
    assert r["image_candidate_path"] == str(tmp_path / "Yevr_II_A_223_1r.jpg")
    assert r["image_match"] == "file_name_token_unverified" and r["image_verified"] is False
    assert "image_path" not in r
    assert r["distinct_images"] == 1 and len(r["image_files"]) == 2 and r["image_size"] == 100
    v = P.pair_images(key, "1v", str(tmp_path))
    assert v["image_candidate_path"] == str(tmp_path / "Yevr_II_A_223_1v(1).jpg")
    assert v["distinct_images"] == 2 and v["image_size"] == 80
    assert P.pair_images(key, "2r", str(tmp_path)) is None
    assert P.pair_images(key, "p1", str(tmp_path)) is None
    assert P.pair_images(None, "1r", str(tmp_path)) is None


def test_ingest_end_to_end(fonts: Dict[str, "fitz.Font"], tmp_path: pathlib.Path) -> None:
    """One PDF -> folio records with header fields, cleaned lines and the 1r image.

    :param fonts: Fonts fixture.
    :param tmp_path: Temporary folder.
    """
    pdf = save(header_pdf(fonts), str(tmp_path / "transcription_abc.pdf"))
    (tmp_path / "Yevr_II_A_223_1r.jpg").write_bytes(b"A" * 64)
    (tmp_path / "Yevr_II_A_223_1r(1).jpg").write_bytes(b"A" * 64)
    recs = P.ingest(pdf, str(tmp_path), None)
    assert [r["folio"] for r in recs] == ["1r", "1v"]
    assert recs[0]["shelfmark"] == "Evr. IIA 223.2" and recs[0]["shelfmark_source"] == "header"
    assert recs[0]["series_key"] == ["Yevr_II_A", "223"]
    assert recs[0]["cleaned_lines"] == FOLIO_1R and recs[1]["cleaned_lines"] == FOLIO_1V
    assert recs[0]["image_candidate_path"].endswith("Yevr_II_A_223_1r.jpg") and recs[1]["image_candidate_path"] is None
    assert recs[0]["image_match"] == "file_name_token_unverified" and recs[0]["image_verified"] is False
    assert all("image_path" not in r for r in recs)
    assert recs[0]["style"] == "header_folio"
    assert recs[1]["flags"]["notes_cut"] == 1
    assert P.ingest(save(junk_pdf(), str(tmp_path / "transcription_junk.pdf")), str(tmp_path), None) == []


# ───────────────────────────── verifier fixes ─────────────────────────────
def test_catchword_label_removed_catchword_kept() -> None:
    """The label ``שומר הדף:`` goes, the catchword after it stays (real ink)."""
    out = P.clean_folio(["שלום עולם גדול", "שומר הדף: אדם"], None)
    assert out["cleaned_lines"] == ["שלום עולם גדול", "אדם"]
    assert out["flags"]["editor_phrase_removed"] == 1


def test_editor_note_suspect_flag_and_exclusion() -> None:
    """A cleaned line about the photograph or the margin is counted and excludes the folio."""
    out = P.clean_folio(["שלום עולם גדול", "בצילום יש קפל", "הושלם בגליון", "אדם טוב"], None)
    assert out["flags"]["editor_note_suspect"] == 2
    rec = {"flags": out["flags"], "folio": "1r", "letters": out["letters"], "style": "header_folio"}
    assert "editor_note_suspect" in P.page_row_exclusions(rec)
    clean = P.clean_folio(["שלום עולם גדול", "אדם טוב"], None)
    assert "editor_note_suspect" not in clean["flags"]


def folio(label: Optional[str], n_lines: int) -> Dict[str, object]:
    """A bare folio record for the sequence checks.

    :param label: Folio label.
    :param n_lines: Number of lines.
    :return: The record.
    """
    return {"folio": label, "lines": ["שלום"] * n_lines, "flags": collections.Counter()}


def test_side_sequence_gap_and_line_count_outlier() -> None:
    """``6r`` followed by ``7r`` is a gap; a folio with twice the median lines is an outlier."""
    folios = [folio("5r", 22), folio("5v", 21), folio("6r", 45), folio("7r", 23), folio("7v", 24)]
    P.flag_sequence_and_outliers(folios)
    assert [f["folio"] for f in folios if f["flags"].get("side_sequence_gap")] == ["6r"]
    assert folios[2]["flags"]["side_sequence_gap"] == 1
    assert [f["folio"] for f in folios if f["flags"].get("line_count_outlier")] == ["6r"]
    jump = [folio("1v", 10), folio("3r", 10), folio("p2", 10)]
    P.flag_sequence_and_outliers(jump)
    assert jump[0]["flags"]["side_sequence_gap"] == 2 and not jump[1]["flags"]


def test_split_document_flags_merged_side(fonts: Dict[str, "fitz.Font"]) -> None:
    """In a generated PDF, the folio whose verso heading is missing gets both flags.

    :param fonts: Fonts fixture.
    """
    doc = fitz.open()
    page = doc.new_page(height=1400)
    y = 60.0
    for label, n in (("5r", 6), ("5v", 6), ("6r", 13), ("7r", 6)):
        draw_ltr(page, y, label, fonts, x=525)
        y += 16
        for _ in range(n):
            draw_line(page, y, [("שלום עולם גדול", "heb")], fonts)
            y += 16
    folios = P.split_folios(doc)
    assert [f["folio"] for f in folios] == ["5r", "5v", "6r", "7r"]
    assert folios[2]["flags"]["side_sequence_gap"] == 1 and folios[2]["flags"]["line_count_outlier"] == 1
    assert not folios[1]["flags"].get("side_sequence_gap")


@pytest.mark.parametrize("line,label,kind,rest", [
    ("אתחלתא: 1: א", "1r", "numeric", ""),
    ("אתחלתא: (1א)", "1r", "numeric", ""),
    ("העתקות : 3ב :שלום", "3v", "colon_prefix", "שלום"),
    ("אתחלתא:", None, None, ""),
    ("אתחלתא:  )1א(   שלום עולם", "1r", "bracketed", "שלום עולם"),
    ("(2ב) אדם טוב", "2v", "bracketed", "אדם טוב"),
])
def test_excerpt_label_stripped_before_heading(line: str, label: Optional[str], kind: Optional[str], rest: str) -> None:
    """An excerpt label before a heading no longer hides it.

    :param line: Synthetic line.
    :param label: Expected label.
    :param kind: Expected kind.
    :param rest: Expected trailing text.
    """
    m = P.match_folio_marker(line)
    assert (m.label if m else None, m.kind if m else None) == (label, kind)
    if m:
        assert m.rest == rest


def test_excerpt_heading_splits_document(fonts: Dict[str, "fitz.Font"]) -> None:
    """``אתחלתא: 1: א`` on the first line is the 1r heading, not a page-fallback document.

    :param fonts: Fonts fixture.
    """
    doc = fitz.open()
    page = doc.new_page()
    draw_line(page, 72, [("אתחלתא", "heb"), (": 1: ", "heb"), ("א", "heb")], fonts)
    draw_line(page, 88, [("שלום עולם גדול", "heb")], fonts)
    draw_line(page, 104, [("אדם טוב בבית", "heb")], fonts)
    folios = P.split_folios(doc)
    assert [f["folio"] for f in folios] == ["1r"]
    assert folios[0]["lines"] == ["שלום עולם גדול", "אדם טוב בבית"]
    assert "no_folio_markers" not in folios[0]["flags"]


def test_page_row_exclusions() -> None:
    """Flags, page-fallback labels and empty records are reasons; a clean folio has none."""
    ok = {"flags": {"line_numbers_stripped": 3}, "folio": "2v", "letters": 120, "style": "header_folio"}
    assert P.page_row_exclusions(ok) == []
    assert P.page_row_exclusions(dict(ok, folio="p1")) == ["label_not_folio"]
    assert P.page_row_exclusions(dict(ok, flags={"unresolved_finals": 2, "excerpt_document": 1})) == \
        ["excerpt_document", "unresolved_finals"]
    assert P.page_row_exclusions({"flags": {}, "folio": "s2", "letters": 50, "style": "side_per_file"}) == []
    assert P.page_row_exclusions(dict(ok, letters=0)) == ["no_letters"]


# ───────────────────────────── old style ─────────────────────────────
COLUMN_A = ["טוב היה כל", "אשר גדול ספר"]          # no final letters: the finals test needs that
COLUMN_B = ["איש בית בא", "על כל יצא", "היה טוב אשר"]


def side_pdf(fonts: Dict[str, "fitz.Font"], columns: Sequence[Tuple[str, Sequence[str]]],
             notes: Sequence[str] = ()) -> "fitz.Document":
    """An old-style side: no header, no folio heading, ``טור_`` column markers.

    :param fonts: Fonts.
    :param columns: ``(marker or "", lines)``; an empty marker writes the lines without one.
    :param notes: Lines written after a ``הערות:`` line.
    :return: The document.
    """
    doc = fitz.open()
    page = doc.new_page()
    y = 72.0
    for marker, lines in columns:
        if marker:
            draw_line(page, y, [(marker, "heb")], fonts)
            y += 24
        for text in lines:
            draw_line(page, y, [(text, "heb")], fonts)
            y += 14
        y += 16
    if notes:
        draw_line(page, y, [("הערות:", "heb")], fonts)
        for text in notes:
            y += 14
            draw_line(page, y, [(text, "heb")], fonts)
    return doc


def test_detect_style(fonts: Dict[str, "fitz.Font"]) -> None:
    """Header or headings -> header_folio; a side file -> side_per_file; no letters -> unknown.

    :param fonts: Fonts fixture.
    """
    assert P.detect_style(header_pdf(fonts), "transcription_abc.pdf") == "header_folio"
    side = side_pdf(fonts, [("טור_א", COLUMN_A), ("טור_ב", COLUMN_B)])
    assert P.detect_style(side, "transcription_manual_Manchester_B_7777.pdf") == "side_per_file"
    weak = fitz.open()
    page = weak.new_page()
    draw_line(page, 72, [("5", "heb"), (" א", "heb")], fonts)
    draw_line(page, 88, [("שלום עולם גדול", "heb")], fonts)
    assert P.detect_style(weak, "transcription_manual_Manchester_B_7777_2.pdf") == "side_per_file"
    assert P.detect_style(weak, "transcription_0573c6a5.pdf") == "side_per_file"
    draw_line(page, 104, [("5", "heb"), (" ב", "heb")], fonts)
    draw_line(page, 120, [("אדם טוב", "heb")], fonts)
    assert P.detect_style(weak, "transcription_0573c6a5.pdf") == "header_folio"
    assert P.detect_style(weak, "transcription_manual_Manchester_B_7777_2.pdf") == "side_per_file"
    empty = fitz.open()
    empty.new_page()
    assert P.detect_style(empty, "transcription_manual_X.pdf") == "unknown"


@pytest.mark.parametrize("name,siblings,known,stem,side,resolution,ambiguous", [
    ("transcription_manual_Cambridge_CUL_T_S_AS_97_1.pdf", [], None, "Cambridge_CUL_T_S_AS_97_1", 1, "no_suffix", False),
    ("transcription_manual_Cambridge_CUL_T_S_AS_97_1_2.pdf", ["Cambridge_CUL_T_S_AS_97_1"], None,
     "Cambridge_CUL_T_S_AS_97_1", 2, "sibling_file", False),
    ("transcription_manual_Cambridge_CUL_T_S_AS_97_2.pdf", [], {"T_S_AS_97_2"}, "Cambridge_CUL_T_S_AS_97_2", 1,
     "corpus", False),
    ("transcription_manual_Cambridge_CUL_T_S_AS_20_34_3.pdf", [], {"T_S_AS_20_34"}, "Cambridge_CUL_T_S_AS_20_34", 3,
     "corpus", False),
    ("transcription_manual_Manchester_B_7795_4.pdf", [], None, "Manchester_B_7795", 4, "suffix_rule", True),
    ("transcription_manual_Cambridge_CUL_T_S_AS_97_10.pdf", ["Cambridge_CUL_T_S_AS_97_1"], None,
     "Cambridge_CUL_T_S_AS_97_10", 1, "no_suffix", False),
    ("transcription_manual_St_Petersburg_St_Peterburg_Yevr_II_A_270_1.pdf", [], None,
     "St_Petersburg_St_Peterburg_Yevr_II_A_270_1", 1, "no_suffix", False),
])
def test_parse_side_file_name(name: str, siblings: List[str], known: Optional[set], stem: str, side: int,
                              resolution: str, ambiguous: bool) -> None:
    """Side suffixes are read on sibling or corpus evidence, else by the naming rule (ambiguous).

    :param name: File name.
    :param siblings: Sibling stems.
    :param known: Known core shelf marks (None: no lookup).
    :param stem: Expected stem.
    :param side: Expected side index.
    :param resolution: Expected resolution.
    :param ambiguous: Expected ambiguity flag.
    """
    out = P.parse_side_file_name(name, siblings, (lambda c: c in known) if known is not None else None)
    assert (out["stem"], out["side_index"], out["side_resolution"], out["side_ambiguous"]) == \
        (stem, side, resolution, ambiguous)


def test_parse_side_file_name_normalises() -> None:
    """Institution split off, merging-normaliser id, copy suffix kept apart, other names None."""
    out = P.parse_side_file_name("transcription_manual_Manchester_B_7777-1.pdf")
    assert out["institution"] == "Manchester" and out["shelfmark"] == "B 7777"
    assert out["canonical_id"] == "JRL_B_7777" and out["copy_suffix"] == "-1" and out["side_index"] == 1
    cul = P.parse_side_file_name("transcription_manual_Cambridge_CUL_T_S_AS_97_1.pdf")
    assert cul["canonical_id"] == "T_S_AS_97_1"
    assert P.parse_side_file_name("transcription_0573c6a5-23c7-4601-8090-b6e5ab3de69b.pdf") is None


def test_split_side_columns_keeps_two_columns_and_cuts_notes(fonts: Dict[str, "fitz.Font"]) -> None:
    """Two column markers -> two columns of one side, in order; notes dropped.

    :param fonts: Fonts fixture.
    """
    doc = side_pdf(fonts, [("טור_א", COLUMN_A), ("טור_ב", COLUMN_B)], notes=["הערה על השורה"])
    out = P.split_side_columns(doc)
    assert out["column_markers"] == 2 and not out["flags"]
    assert [c["folio"] for c in out["columns"]] == ["א", "ב"]
    assert [c["lines"] for c in out["columns"]] == [COLUMN_A, COLUMN_B]
    assert out["columns"][1]["notes_lines"] == 2
    plain = P.split_side_columns(side_pdf(fonts, [("", COLUMN_A)]))
    assert plain["flags"]["no_column_markers"] == 1 and plain["columns"][0]["lines"] == COLUMN_A


def test_ingest_side_files_shares_finals_and_keeps_column_bounds(fonts: Dict[str, "fitz.Font"],
                                                                 tmp_path: pathlib.Path) -> None:
    """Two side files of one shelf mark: one finals mapping, columns as ranges of cleaned lines.

    :param fonts: Fonts fixture.
    :param tmp_path: Temporary folder.
    """
    s1 = save(side_pdf(fonts, [("טור_א", COLUMN_A), ("טור_ב", COLUMN_B)]),
              str(tmp_path / "transcription_manual_Manchester_B_7777.pdf"))
    s2 = save(finals_pdf(fonts), str(tmp_path / "transcription_manual_Manchester_B_7777_2.pdf"))
    lexicon = collections.Counter({w: 3 for w in FINAL_WORDS + PLAIN_WORDS})
    items = [{"path": s2, "shelfmark": "B 7777", "side_index": 2, "side_resolution": "sibling_file"},
             {"path": s1, "shelfmark": "B 7777", "side_index": 1, "side_resolution": "no_suffix"}]
    recs = P.ingest_side_files(items, lexicon)
    assert [r["side_index"] for r in recs] == [1, 2] and [r["folio"] for r in recs] == ["s1", "s2"]
    assert recs[0]["style"] == "side_per_file" and recs[0]["side_resolution"] == "no_suffix"
    assert recs[0]["cleaned_lines"] == COLUMN_A + COLUMN_B
    assert [(c["column"], c["start"], c["end"]) for c in recs[0]["columns"]] == [("א", 0, 2), ("ב", 2, 5)]
    assert recs[1]["finals_mapping"] == {"$": "ם"} == recs[0]["finals_mapping"]
    assert recs[1]["finals_repaired"] == 12 and "$" not in " ".join(recs[1]["cleaned_lines"])
    junk = save(junk_pdf(), str(tmp_path / "transcription_manual_Manchester_B_7777_3.pdf"))
    assert P.ingest_side_files([{"path": junk, "shelfmark": "B 7777", "side_index": 3}], None) == []


def test_group_side_records_flags_gaps_and_duplicates() -> None:
    """Sides 1, 1, 3 of one shelf mark: duplicate side 1, gap after side 1; other marks apart."""
    def rec(mark: str, side: int, name: str) -> Dict[str, object]:
        """A bare side record.

        :param mark: Shelf mark.
        :param side: Side index.
        :param name: File name.
        :return: The record.
        """
        return {"shelfmark": mark, "side_index": side, "pdf_file": name, "flags": {}}
    docs = P.group_side_records([rec("B 7777", 3, "c"), rec("B 7777", 1, "a"), rec("B 7777", 1, "b"),
                                 rec("B 7778", 1, "d")])
    assert len(docs) == 2
    d = next(x for x in docs if x["shelfmark"] == "B 7777")
    assert d["side_indices"] == [1, 3] and [r["pdf_file"] for r in d["sides"]] == ["a", "b", "c"]
    assert all(r["flags"]["side_index_duplicate"] == 1 for r in d["sides"][:2])
    assert all(r["flags"]["side_sequence_gap"] == 1 for r in d["sides"][:2])
    assert not d["sides"][2]["flags"]


def test_provenance_signals_and_line_structure() -> None:
    """Family, layout, credit kind, sigla density, line structure, finals counts."""
    recs = [{"letters": 20, "raw_lines": ["שלו?ם [עו]לם", "אדם טוב"], "cleaned_lines": ["שלום [...]לם", "אדם טוב",
                                                                                     "היה איש"],
             "flags": {"unresolved_finals": 1}, "finals_repaired": 2, "finals_mapping": {}, "column_markers": 2}]
    sig = P.provenance_signals(recs, "side_per_file", "Example Transcriptions Team")
    assert sig["format_family"] == "side_per_file" and sig["layout"] == "columns"
    assert sig["credit_kind"] == "project_team" and sig["finals_unresolved"] == 1 and sig["finals_repaired"] == 2
    assert sig["sigla_per_100_letters"] == round(100 * 3 / 14, 2)
    assert sig["line_structure"] == "line_breaks"
    assert P.provenance_signals(recs, "side_per_file", "Site User - A")["credit_kind"] == "site_user"
    assert P.line_structure(["שלום"]) == "too_few_lines"
    assert P.line_structure(["שלום עולם"] * 4, numbered_lines=4) == "numbered_lines"
    wide = ["א" * 80] * 5
    assert P.line_structure(wide) == "reflowed_suspect"


# ───────────────────────────── second verification pass ─────────────────────────────
@pytest.mark.parametrize("line,note", [
    ("שני קווים אלכסוניים בשוליים", True),
    ("נוסף מע\"ה בכתב אחר", True),
    ("נוסף מע ה בכתב אחר", True),
    ("יש כתם על השורה", True),
    ("כאן מריחת דיו גדולה", True),
    ("ריבוע דיו קטן", True),
    ("יש חור בקלף", True),
    ("גירוד בשוליים", True),
    ("קו חוצה למחיקה", True),
    ("קרחה דרכה נראה", True),
    ("הלך בן קרחה אל העיר", False),
    ("ולמחיקה אין זכר", False),
    ("מעשה היה בעיר", False),
    ("שלום עולם גדול", False),
])
def test_editor_note_words_of_the_physical_page(line: str, note: bool) -> None:
    """Notes on stains, holes, smears, erasures and above-the-line additions are editor's notes.

    :param line: Synthetic cleaned line.
    :param note: Whether it reads as an editor's note.
    """
    assert bool(P.EDITOR_NOTE_RE.search(line)) is note


def test_editor_note_document_rule() -> None:
    """A third of the folios with notes flags every folio of the document; fewer flags none."""
    def recs(flagged: int, total: int) -> List[Dict[str, object]]:
        """Bare folio records.

        :param flagged: Records carrying ``editor_note_suspect``.
        :param total: Records.
        :return: The records.
        """
        return [{"flags": {"editor_note_suspect": 1} if i < flagged else {}, "folio": f"{i + 1}r", "letters": 50,
                 "style": "header_folio"} for i in range(total)]
    doc = recs(1, 3)
    assert P.flag_editor_note_document(doc)
    assert all(r["flags"]["editor_note_document"] == 1 for r in doc)
    assert "editor_note_document" in P.page_row_exclusions(doc[2])
    few = recs(1, 4)
    assert not P.flag_editor_note_document(few) and not any("editor_note_document" in r["flags"] for r in few)
    assert not P.flag_editor_note_document(recs(0, 3)) and not P.flag_editor_note_document([])


@pytest.mark.parametrize("line,label,kind", [
    ("דף פא עמוד ב (גמרא)", "ed81v", "edition_ref"),
    ("דף לא עמוד א )משנה   (", "ed31r", "edition_ref"),
    ("דף ה עמוד א", "5r", "daf"),
    ("דף ב עמוד א (גמרא) ועוד", "2r", "daf"),
])
def test_edition_reference_heading(line: str, label: str, kind: str) -> None:
    """A ``daf`` heading citing the printed Talmud or Mishnah is no leaf.

    :param line: Synthetic heading.
    :param label: Expected label.
    :param kind: Expected kind.
    """
    m = P.match_folio_marker(line)
    assert (m.label, m.kind) == (label, kind)
    assert P.side_index(m.label) is None if kind == "edition_ref" else P.side_index(m.label) is not None


def edition_pdf(fonts: Dict[str, "fitz.Font"]) -> "fitz.Document":
    """Two edition page references, each before two lines of text (no header).

    The reference is drawn as its own span with mirrored brackets (``)גמרא(``), as the text layer of
    the downloads has it.

    :param fonts: Fonts.
    :return: The document.
    """
    doc = fitz.open()
    page = doc.new_page()
    y = 72.0
    for heading, lines in (("דף פא עמוד ב", COLUMN_A), ("דף פב עמוד א", COLUMN_B[:2])):
        draw_line(page, y, [(heading, "heb"), (" )גמרא(", "heb")], fonts)
        y += 16
        for text in lines:
            draw_line(page, y, [(text, "heb")], fonts)
            y += 16
    return doc


def test_edition_reference_sections_and_side_style(fonts: Dict[str, "fitz.Font"]) -> None:
    """Edition headings cut sections that are no folios; a sibling-confirmed side file stays one side.

    :param fonts: Fonts fixture.
    """
    doc = edition_pdf(fonts)
    folios = P.split_folios(doc)
    assert [f["folio"] for f in folios] == ["ed81v", "ed82r"]
    assert all(not f["flags"].get("folio_order_break") and not f["flags"].get("side_sequence_gap") for f in folios)
    assert P.page_row_exclusions({"flags": {}, "folio": "ed81v", "letters": 30, "style": "header_folio"}) == \
        ["label_not_folio"]
    name = "transcription_manual_Cambridge_CUL_T_S_F1_1_34_3.pdf"
    assert P.detect_style(doc, name) == "header_folio"
    assert P.detect_style(doc, name, side_file_evidence=True) == "side_per_file"
    assert P.detect_style(doc, "transcription_manual_Cambridge_CUL_T_S_F1_1_34.pdf", side_file_evidence=True) == \
        "header_folio"
    side = P.split_side_columns(doc)
    assert side["flags"]["edition_ref_heading"] == 2
    assert [line for c in side["columns"] for line in c["lines"]] == COLUMN_A + COLUMN_B[:2]


@pytest.mark.parametrize("rest,mark,folio", [
    ("( 'A 11 1R) צד שמאל", "A 11", "1r"),
    ("(B 7777 2V)", "B 7777", "2v"),
    ("T-S F1(1).33 1v", "T-S F1(1).33", "1v"),
    ("Yevr. Arab. II 12", "Yevr. Arab. II 12", None),
    (")גמרא (", None, None),
    ("1/12 – 2/14", None, None),
    ("fol. 3", None, None),
    ("", None, None),
])
def test_heading_shelfmark(rest: str, mark: Optional[str], folio: Optional[str]) -> None:
    """A heading's trailing text may name a shelf mark and a side; other text names none.

    :param rest: Synthetic trailing text.
    :param mark: Expected mark.
    :param folio: Expected side label.
    """
    out = P.heading_shelfmark(rest)
    assert (out["mark"] if out else None, out["folio"] if out else None) == (mark, folio)


def test_same_shelfmark_across_spellings() -> None:
    """Institution heads, holder tokens and the store's file-name spellings agree; other marks differ."""
    assert P.same_shelfmark("A 10", "Manchester: A 10") and P.same_shelfmark("JRL A 10", "A 10")
    assert P.same_shelfmark("Yevr II A 223 2", "Evr. IIA 223.2")
    assert not P.same_shelfmark("A 11", "A 10") and not P.same_shelfmark("", "A 10")


def test_ingest_flags_heading_naming_another_shelfmark(fonts: Dict[str, "fitz.Font"], tmp_path: pathlib.Path) -> None:
    """A joined-fragment file: the heading naming another shelf mark excludes that folio and records it.

    :param fonts: Fonts fixture.
    :param tmp_path: Temporary folder.
    """
    doc = fitz.open()
    page = doc.new_page()
    y = 72.0
    for heading, rest, lines in (("דף א צד א", " (A 11 1R)", COLUMN_A), ("דף א צד ב", " (A 10 1V)", COLUMN_B)):
        draw_line(page, y, [(heading, "heb"), (rest, "heb")], fonts)
        y += 16
        for text in lines:
            draw_line(page, y, [(text, "heb")], fonts)
            y += 16
    path = save(doc, str(tmp_path / "transcription_manual_Manchester_A_10.pdf"))
    recs = P.ingest(path, None, None, index={}, shelfmark_hint="A 10")
    assert [r["folio"] for r in recs] == ["1r", "1v"]
    assert recs[0]["heading_shelfmark"] == {"mark": "A 11", "core_id": "a_11", "folio": "1r"}
    assert recs[0]["flags"]["heading_other_shelfmark"] == 1
    assert "heading_other_shelfmark" in P.page_row_exclusions(recs[0])
    assert recs[1]["heading_shelfmark"]["mark"] == "A 10" and "heading_other_shelfmark" not in recs[1]["flags"]


def side_rec(side: int, lines: Sequence[str], name: str) -> Dict[str, object]:
    """A bare side record for the re-download and numbering checks.

    :param side: Suffix side index.
    :param lines: Raw lines.
    :param name: File name.
    :return: The record.
    """
    return {"side_index": side, "folio": f"s{side}", "pdf_file": name, "digest": f"d{side}{name}",
            "shelfmark": "B 7777", "raw_lines": list(lines), "flags": {}}


def test_side_content_key_ignores_text_layer_order() -> None:
    """Same letters and line count in another order: one key; another line count: another key."""
    a = side_rec(1, ["טוב היה כל", "אשר גדול ספר"], "a")
    b = side_rec(3, ["ספר גדול אשר", "כל היה טוב"], "b")
    c = side_rec(2, ["טוב היה כל אשר גדול ספר"], "c")
    assert P.side_content_key(a) == P.side_content_key(b) != P.side_content_key(c)


def test_merge_side_redownloads_and_renumber() -> None:
    """Sides 1, 2, 3≡1, 5≡2, 6: two copies folded, the rest numbered 1-3 by suffix order."""
    recs = [side_rec(1, COLUMN_A, "x.pdf"), side_rec(2, COLUMN_B, "x_2.pdf"),
            side_rec(3, list(reversed(COLUMN_A)), "x_3.pdf"), side_rec(5, list(reversed(COLUMN_B)), "x_5.pdf"),
            side_rec(6, ["איש טוב", "בית גדול"], "x_6.pdf")]
    kept, copies = P.merge_side_redownloads(recs)
    assert [r["pdf_file"] for r in kept] == ["x.pdf", "x_2.pdf", "x_6.pdf"]
    assert [(c["pdf_file"], c["copy_of_pdf_file"], c["flag"]) for c in copies] == \
        [("x_3.pdf", "x.pdf", "side_redownload"), ("x_5.pdf", "x_2.pdf", "side_redownload")]
    assert kept[0]["flags"]["side_redownload_copies"] == 1 and kept[0]["side_redownloads"][0]["suffix_side_index"] == 3
    P.renumber_sides(kept)
    assert [(r["side_index"], r["suffix_side_index"], r["folio"]) for r in kept] == [(1, 1, "s1"), (2, 2, "s2"),
                                                                                     (3, 6, "s3")]
    assert all(r["flags"]["side_index_from_download_order_unverified"] == 1 for r in kept)
    docs = P.group_side_records(kept)
    assert docs[0]["side_indices"] == [1, 2, 3] and not any(r["flags"].get("side_sequence_gap") for r in kept)


def test_renumber_keeps_duplicates_and_lone_sides() -> None:
    """Equal suffixes keep one number (still a duplicate); a lone side 1 is not flagged, a lone side 3 is."""
    dup = [side_rec(1, COLUMN_A, "a.pdf"), side_rec(1, COLUMN_B, "b.pdf"), side_rec(4, ["איש טוב"], "c.pdf")]
    P.renumber_sides(dup)
    assert [r["side_index"] for r in dup] == [1, 1, 2]
    P.group_side_records(dup, key=lambda r: "k")
    assert dup[0]["flags"]["side_index_duplicate"] == 1 and not dup[2]["flags"].get("side_index_duplicate")
    lone = [side_rec(1, COLUMN_A, "a.pdf")]
    P.renumber_sides(lone)
    assert not lone[0]["flags"]
    late = [side_rec(3, COLUMN_A, "a_3.pdf")]
    P.renumber_sides(late)
    assert late[0]["side_index"] == 1 and late[0]["flags"]["side_index_from_download_order_unverified"] == 1


def test_near_duplicate_pairs() -> None:
    """A one-letter edit of a long text pairs with it; short and different texts do not."""
    rng = random.Random(7)
    base = "".join(rng.choice("אבגדהוזחטיכלמנסעפצקרשת") for _ in range(160))   # synthetic, not periodic
    edited = base[:40] + ("ת" if base[40] != "ת" else "ש") + base[41:]
    other = "".join(rng.choice("אבגדהוזחטיכלמנסעפצקרשת") for _ in range(160))
    pairs = P.near_duplicate_pairs([base, other, edited, "שלום"])
    assert [(a, b) for a, b, _ in pairs] == [(0, 2)] and pairs[0][2] >= P.NEAR_DUPLICATE_RATIO
    assert P.text_letters([{"cleaned_lines": ["שלום עולם", "אדם"]}, {"cleaned_lines": ["טוב"]}]) == "שלוםעולםאדםטוב"


def test_side_file_with_quote_separators_is_flagged(fonts: Dict[str, "fitz.Font"], tmp_path: pathlib.Path) -> None:
    """A side whose words end in ``"`` (scrambled export) is flagged as a document would be.

    :param fonts: Fonts fixture.
    :param tmp_path: Temporary folder.
    """
    words = ["שלום", "עולם", "גדול", "אדם", "טוב", "בבית"]
    lines = [" ".join(w + '"' for w in words) for _ in range(4)]
    path = save(side_pdf(fonts, [("", lines)]), str(tmp_path / "transcription_manual_Manchester_B_7777.pdf"))
    recs = P.ingest_side_files([{"path": path, "shelfmark": "B 7777", "side_index": 1}], None)
    assert recs[0]["flags"]["word_quote_separators"] == 1
    assert "word_quote_separators" in P.page_row_exclusions(recs[0])


def test_credit_kind_and_provenance_keep_no_wording() -> None:
    """A credit is reduced to its kind; provenance accepts a kind or a string and returns no wording."""
    assert P.credit_kind("Example Transcriptions Team") == "project_team"
    assert P.credit_kind("Site User - A") == "site_user" and P.credit_kind("Some Catalog") == "other"
    assert P.credit_kind(None) is None and P.credit_kind("  ") is None
    recs = [{"letters": 5, "raw_lines": ["שלום"], "cleaned_lines": ["שלום"], "flags": {}, "finals_repaired": 0}]
    sig = P.provenance_signals(recs, "side_per_file", "site_user")
    assert sig["credit_kind"] == "site_user" and "team_credit" not in sig
    assert P.provenance_signals(recs, "side_per_file", "Example Team")["credit_kind"] == "project_team"
