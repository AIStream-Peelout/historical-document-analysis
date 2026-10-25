# File name: test_build_arabic_lines.py
# Date: 10/19/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for the kraken Arabic line-set builder (text rules, coordinates, splits, PAGE rewrite, crops, CLI).

Everything runs on synthetic PAGE strings, images and parquet files under ``tmp_path``; nothing reads the NAS.
"""
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from lxml import etree
from PIL import Image

from src.finetuning.kraken import build_arabic_lines as bal

NS13 = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
NS19 = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
LONG = "وكتب الكتاب في المدينة"
PAPER = (200, 180, 150)


def line_xml(line_id: str, text: Optional[str] = None, baseline: Optional[str] = "10,50 400,50",
             coords: str = "10,20 400,20 400,60 10,60", words: Optional[Sequence[str]] = None,
             alternative: Optional[str] = None) -> str:
    """One synthetic ``TextLine``.

    :param line_id: Line id.
    :param text: Line-level transcription; None writes no line ``TextEquiv``.
    :param baseline: Baseline points; None writes no ``Baseline``.
    :param coords: Polygon points.
    :param words: Word transcriptions (each a ``Word`` with its own ``TextEquiv``).
    :param alternative: A second line-level ``TextEquiv``.
    :return: XML fragment.
    """
    parts = [f'<TextLine id="{line_id}">', f'<Coords points="{coords}"/>']
    if baseline is not None:
        parts.append(f'<Baseline points="{baseline}"/>')
    for i, word in enumerate(words or []):
        parts.append(f'<Word id="{line_id}_w{i}"><Coords points="{coords}"/><TextEquiv><Unicode>{word}</Unicode></TextEquiv></Word>')
    if text is not None:
        parts.append(f"<TextEquiv><Unicode>{text}</Unicode></TextEquiv>")
    if alternative is not None:
        parts.append(f'<TextEquiv index="2"><Unicode>{alternative}</Unicode></TextEquiv>')
    parts.append("<TextStyle fontSize=\"10\"/>" if words else "")
    parts.append("</TextLine>")
    return "".join(parts)


def page_xml(lines: Sequence[str], ns: str = NS13, width: int = 1000, height: int = 800, image: str = "page.jpg") -> bytes:
    """A synthetic PAGE document with one text region.

    :param lines: ``TextLine`` fragments in document order.
    :param ns: PAGE namespace.
    :param width: Declared ``imageWidth``.
    :param height: Declared ``imageHeight``.
    :param image: Declared ``imageFilename``.
    :return: XML bytes.
    """
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<PcGts xmlns="{ns}" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            f'xsi:schemaLocation="{ns} {ns}/pagecontent.xsd"><Metadata><Creator>test</Creator></Metadata>'
            f'<Page imageFilename="{image}" imageWidth="{width}" imageHeight="{height}">'
            f'<TextRegion id="r1"><Coords points="0,0 999,0 999,799 0,799"/>{"".join(lines)}'
            f'<TextEquiv><Unicode>region text</Unicode></TextEquiv></TextRegion></Page></PcGts>').encode("utf-8")


def unicode_of(root: Any, line_id: str) -> List[str]:
    """Texts of the line-level ``Unicode`` elements of a line.

    :param root: Parsed document.
    :param line_id: Line id.
    :return: Texts (``""`` for an empty element).
    """
    line = root.find(f".//{{*}}TextLine[@id='{line_id}']")
    return [u.text or "" for te in line.findall("{*}TextEquiv") for u in te.findall("{*}Unicode")]


def write_jpeg(path: Path, size: tuple) -> Path:
    """Write a plain JPEG.

    :param path: Target.
    :param size: ``(width, height)``.
    :return: The path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, PAPER).save(path, format="JPEG")
    return path


# ----------------------------------------------------------------------------- text rules

def test_clean_target_strips_marks_tatweel_bidi_and_spaces() -> None:
    """clean_target removes harakat, tatweel and bidi controls and collapses whitespace."""
    raw = "‏بِسْمِ   اللّـهِ‫ الرَّحْمَنِ﻿ "
    assert bal.clean_target(raw) == ("بسم الله الرحمن", "")


def test_clean_target_recomposes_nfc_after_removing_marks() -> None:
    """clean_target re-normalises to NFC after a mark between letter and hamza is removed."""
    text, reason = bal.clean_target("أّمر")       # alef + shadda + hamza above
    assert reason == "" and text == "أمر"


@pytest.mark.parametrize("raw, reason", [
    ("", bal.DROP_EMPTY),
    ("  َـ ", bal.DROP_EMPTY),
    ("وكتب [الكتاب]", bal.DROP_EDITORIAL),
    ("وكتب <الكتاب>", bal.DROP_EDITORIAL),
    ("وكتب {الكتاب}", bal.DROP_EDITORIAL),
    ("وكتب ⟨الكتاب⟩", bal.DROP_EDITORIAL),
    ("MS.ARA.609 وكتب", bal.DROP_LATIN),
    ("وكتب ḥadīth", bal.DROP_LATIN),
    ("١٢٣ ب", bal.DROP_FEW_LETTERS),
    ("٤٥", bal.DROP_FEW_LETTERS),
])
def test_clean_target_drop_reasons(raw: str, reason: str) -> None:
    """clean_target drops empty, bracketed, Latin and letter-poor lines with the right reason."""
    assert bal.clean_target(raw) == ("", reason)


def test_clean_target_keeps_digits_punctuation_and_parentheses() -> None:
    """clean_target keeps digits, Arabic punctuation and parentheses unchanged."""
    assert bal.clean_target("في سنة ٣٤٥، (قال) 12 × ؟") == ("في سنة ٣٤٥، (قال) 12 × ؟", "")


def test_arabic_letter_count_range() -> None:
    """arabic_letter_count counts only the two Arabic letter ranges."""
    assert bal.arabic_letter_count("ؠيٮۓ") == 4
    assert bal.arabic_letter_count("۔٠ً ab") == 0


def test_slug_and_unique_stems_have_no_dots() -> None:
    """unique_stems gives dot- and space-free stems and disambiguates names sharing a slug."""
    stems = bal.unique_stems("agapet_bnf76", ["BnF Paris Ara 76-041_2L", "a.b", "a b"])
    assert stems["BnF Paris Ara 76-041_2L"] == "agapet_bnf76_bnf_paris_ara_76_041_2l"
    assert stems["a.b"] != stems["a b"] and all("." not in s and " " not in s for s in stems.values())


# ----------------------------------------------------------------------------- geometry

def test_rescale_points_scales_rounds_and_clamps() -> None:
    """rescale_points scales per axis, rounds and clamps points into the image."""
    assert bal.rescale_points("100,200 40,80", 0.5, 0.25, 1000, 1000) == "50,50 20,20"
    assert bal.rescale_points(" 10.6,3.2   -5,7 2000,900 ", 1.0, 1.0, 1000, 800) == "11,3 0,7 999,799"


def test_frame_scale_and_aspect_mismatch() -> None:
    """frame_scale handles undeclared sizes and aspect_mismatch flags a rotated frame, not a resize."""
    sx, sy = bal.frame_scale((4264, 5786), (3832, 5200))
    assert sx == pytest.approx(0.8987, abs=1e-4) and not bal.aspect_mismatch(sx, sy)
    assert bal.frame_scale((0, 0), (10, 20)) == (1.0, 1.0)
    assert bal.aspect_mismatch(*bal.frame_scale((1000, 2000), (2000, 1000)))


# ----------------------------------------------------------------------------- splits

def test_assign_splits_stratified_exact_and_deterministic() -> None:
    """assign_splits sends the lowest-hash share of large manuscripts to val, hash-splits small ones, ignores order."""
    big = [f"rasam/a_{i:03d}" for i in range(100)]
    ten = [f"rasam/b_{i:03d}" for i in range(10)]
    small = [f"rasam/c_{i:03d}" for i in range(9)]
    splits = bal.assign_splits({"A": big, "B": ten, "C": small})
    assert sum(splits[k] == "val" for k in big) == 8
    assert sum(splits[k] == "val" for k in ten) == 1
    lowest = min(big, key=lambda k: hashlib.sha1(k.encode()).hexdigest())
    assert splits[lowest] == "val"
    assert all(splits[k] == bal.hash_split(k) for k in small)
    again = bal.assign_splits({"C": list(reversed(small)), "B": list(reversed(ten)), "A": list(reversed(big))})
    assert again == splits


def test_hash_split_share() -> None:
    """hash_split sends roughly VAL_SHARE of keys to val."""
    keys = [f"p{i}" for i in range(5000)]
    share = sum(bal.hash_split(k) == "val" for k in keys) / len(keys)
    assert 0.06 < share < 0.10


# ----------------------------------------------------------------------------- PAGE rewrite

def test_rewrite_page_rescales_rewrites_and_cleans() -> None:
    """rewrite_page rescales geometry, sets the image path and size, and cleans or empties every line."""
    lines = [line_xml("l1", "بِسْمِ اللّه الرَّحمن", baseline="2132,1000 4264,1000", coords="0,0 4264,0 4264,5786 0,5786"),
             line_xml("l2", ""),
             line_xml("l3", "وكتب [الكتاب]"),
             line_xml("l4", "f. 12r وكتب"),
             line_xml("l5", LONG)]
    data = page_xml(lines, NS13, width=4264, height=5786, image="BULAC_MS_ARA_609_00004.jpg")
    out, counts, letters, edits = bal.rewrite_page(data, "/abs/images/BULAC_MS_ARA_609_00004.jpg", 3832, 5200)
    root = etree.fromstring(out)
    assert etree.QName(root).namespace == NS13
    assert NS13 in root.get("{http://www.w3.org/2001/XMLSchema-instance}schemaLocation")
    page = root.find("{*}Page")
    assert (page.get("imageFilename"), page.get("imageWidth"), page.get("imageHeight")) == \
        ("/abs/images/BULAC_MS_ARA_609_00004.jpg", "3832", "5200")
    line1 = root.find(".//{*}TextLine[@id='l1']")
    assert line1.find("{*}Baseline").get("points") == "1916,899 3831,899"
    assert line1.find("{*}Coords").get("points") == "0,0 3831,0 3831,5199 0,5199"
    assert root.find("{*}Page/{*}TextRegion/{*}Coords").get("points") == "0,0 898,0 898,718 0,718"
    assert [el.get("id") for el in root.iter("{*}TextLine")] == ["l1", "l2", "l3", "l4", "l5"]
    assert len(root.findall(".//{*}Baseline")) == 5
    assert unicode_of(root, "l1") == ["بسم الله الرحمن"]
    assert unicode_of(root, "l2") == unicode_of(root, "l3") == unicode_of(root, "l4") == [""]
    assert unicode_of(root, "l5") == [LONG]
    assert counts == Counter({bal.KEPT: 2, bal.DROP_EMPTY: 1, bal.DROP_EDITORIAL: 1, bal.DROP_LATIN: 1})
    assert letters == bal.arabic_letter_count("بسم الله الرحمن" + LONG)
    assert edits == Counter({"rescaled pages": 1})
    assert root.find("{*}Page/{*}TextRegion/{*}TextEquiv/{*}Unicode").text == "region text"


def test_rewrite_page_same_size_keeps_points() -> None:
    """rewrite_page keeps points and namespace when the image matches the XML frame."""
    data = page_xml([line_xml("l1", LONG)], NS19)
    out, counts, _, edits = bal.rewrite_page(data, "/x/page.jpg", 1000, 800)
    root = etree.fromstring(out)
    assert etree.QName(root).namespace == NS19
    assert root.find(".//{*}TextLine/{*}Baseline").get("points") == "10,50 400,50"
    assert counts[bal.KEPT] == 1 and not edits


def test_rewrite_page_alternatives_words_and_geometry() -> None:
    """rewrite_page drops alternative readings, builds line text from words and empties lines without geometry."""
    lines = [line_xml("alt", "قال الشيخ", alternative="قال الشيخ الامام"),
             line_xml("words", None, words=["قال", "الشيخ"]),
             line_xml("words_dropped", None, words=["MS", "12"]),
             line_xml("nobase", LONG, baseline=None),
             line_xml("onepoint", LONG, baseline="10,50 10,50"),
             line_xml("nocoords", LONG, coords="10,20 400,20"),
             line_xml("textless", None)]
    out, counts, _, edits = bal.rewrite_page(page_xml(lines, NS19), "/x/page.tif", 1000, 800)
    root = etree.fromstring(out)
    assert unicode_of(root, "alt") == ["قال الشيخ"]
    assert edits["alternative transcriptions removed"] == 1
    words = root.find(".//{*}TextLine[@id='words']")
    assert unicode_of(root, "words") == ["قال الشيخ"]
    tags = [etree.QName(child).localname for child in words]
    assert tags.index("TextEquiv") == tags.index("Word") + 2 and tags[-1] == "TextStyle"
    assert etree.QName(words.find("{*}TextEquiv")).namespace == NS19
    assert unicode_of(root, "words_dropped") == [""]
    assert unicode_of(root, "nobase") == unicode_of(root, "onepoint") == unicode_of(root, "nocoords") == [""]
    assert root.find(".//{*}TextLine[@id='textless']").find("{*}TextEquiv") is None
    assert counts == Counter({bal.KEPT: 2, bal.DROP_LATIN: 1, bal.DROP_NO_BASELINE: 2, bal.DROP_NO_COORDS: 1,
                              bal.DROP_EMPTY: 1})


def test_line_text_reads_first_textequiv() -> None:
    """line_text reads the first line-level TextEquiv."""
    root = etree.fromstring(page_xml([line_xml("l", "أول", alternative="ثان")]))
    assert bal.line_text(root.find(".//{*}TextLine")) == "أول"


# ----------------------------------------------------------------------------- page sets

def _source(tmp_path: Path, xml: bytes, image_size: Optional[tuple] = (1000, 800),
            served: Optional[tuple] = None, stem: str = "rasam_p1") -> bal.PageSource:
    """A source page on disk.

    :param tmp_path: Folder.
    :param xml: PAGE bytes.
    :param image_size: Image to write (None: no image).
    :param served: Manifest size.
    :param stem: Output stem.
    :return: The source.
    """
    xml_path = tmp_path / "src" / f"{stem}.xml"
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_bytes(xml)
    image = write_jpeg(tmp_path / "img" / f"{stem}.jpg", image_size) if image_size else None
    return bal.PageSource(key=f"rasam/{stem}", stem=stem, manuscript="MS.1", xml_path=xml_path, image_path=image,
                          served=served)


def test_process_page_writes_cleaned_copy(tmp_path: Path) -> None:
    """process_page writes a rescaled, cleaned PAGE copy pointing at the absolute image path."""
    source = _source(tmp_path, page_xml([line_xml("l1", LONG)], width=2000, height=1600), (1000, 800), (1000, 800))
    result = bal.process_page(source, tmp_path / "out")
    assert result.skip == "" and result.counts[bal.KEPT] == 1
    assert result.paths == [(tmp_path / "out" / "rasam_p1.xml").absolute()]
    root = etree.fromstring(result.paths[0].read_bytes())
    assert root.find("{*}Page").get("imageFilename") == str(source.image_path.absolute())
    assert root.find(".//{*}Baseline").get("points") == "5,25 200,25"


@pytest.mark.parametrize("image_size, served, lines, skip", [
    (None, None, [line_xml("l", LONG)], "no image"),
    ((1000, 800), (999, 800), [line_xml("l", LONG)], "image size differs from manifest"),
    ((800, 1000), None, [line_xml("l", LONG)], "image aspect differs from XML frame"),
    ((1000, 800), None, [line_xml("l", ""), line_xml("m", "[x]")], "no kept lines"),
])
def test_process_page_skips(tmp_path: Path, image_size: Optional[tuple], served: Optional[tuple],
                            lines: List[str], skip: str) -> None:
    """process_page skips pages without an image, with a size or aspect mismatch, or with no kept line, writing nothing."""
    result = bal.process_page(_source(tmp_path, page_xml(lines), image_size, served), tmp_path / "out")
    assert result.skip == skip and not result.paths
    assert not (tmp_path / "out" / "rasam_p1.xml").exists()


def test_process_page_not_page(tmp_path: Path) -> None:
    """process_page skips an XML file that is not a PAGE document."""
    source = _source(tmp_path, b'<?xml version="1.0"?><mets xmlns="urn:x"/>')
    assert bal.process_page(source, tmp_path / "out").skip == "not a PAGE document"


def test_process_pages_limit_writes_no_extra(tmp_path: Path) -> None:
    """process_pages stops at the limit without writing any page beyond it."""
    sources = [_source(tmp_path, page_xml([line_xml("l", LONG)]), None if i in (0, 2) else (1000, 800), stem=f"rasam_p{i}")
               for i in range(8)]
    results = bal.process_pages(sources, tmp_path / "out", limit=3, workers=2)
    assert sum(not r.skip for r in results) == 3
    assert len(list((tmp_path / "out").glob("*.xml"))) == 3
    assert [r.skip for r in results[:2]] == ["no image", ""]


def twelve_page_sources(tmp_path: Path) -> List[bal.PageSource]:
    """Twelve imaged source pages of one manuscript (``MS.1``).

    :param tmp_path: Folder.
    :return: The sources, in stem order.
    """
    return [_source(tmp_path, page_xml([line_xml("l", LONG)]), stem=f"rasam_p{i:02d}") for i in range(12)]


@pytest.mark.parametrize("val_share", [bal.VAL_SHARE, 0.125])
def test_build_page_set_split_survives_a_lost_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                    val_share: float) -> None:
    """Losing one train page's image changes no other page's split (0.125: 2 val of 12 pages, 1 of 11)."""
    monkeypatch.setattr(bal, "VAL_SHARE", val_share)
    sources = twelve_page_sources(tmp_path)
    before = {r.key: r.split for r in bal.build_page_set("rasam", sources, tmp_path / "out1", workers=2)}
    assert sum(split == "val" for split in before.values()) == max(1, round(val_share * 12))
    gone = next(s for s in sources if before[s.key] == "train")
    gone.image_path.unlink()
    after = {r.key: r for r in bal.build_page_set("rasam", sources, tmp_path / "out2", workers=2)}
    assert after[gone.key].skip == "no image"
    assert {k: r.split for k, r in after.items() if k != gone.key} == {k: s for k, s in before.items() if k != gone.key}


def test_build_page_set_moves_a_page_to_val_when_val_pages_are_lost(tmp_path: Path) -> None:
    """A manuscript whose val pages all lost their images still gets one val page: its lowest-hash kept page."""
    sources = twelve_page_sources(tmp_path)
    splits = bal.source_splits(sources)
    for source in sources:
        if splits[source.key] == "val":
            source.image_path.unlink()
    kept = [r for r in bal.build_page_set("rasam", sources, tmp_path / "out", workers=2) if not r.skip]
    assert len(kept) == 11
    assert [r.key for r in kept if r.split == "val"] == [min((r.key for r in kept), key=bal.sha1_hex)]


def make_agapet(root: Path) -> None:
    """Synthetic Agapet tree: SA-418 (12 pages, one without image), BnF 76 (TIFF + METS.xml), Sin423 (excluded).

    :param root: ``agapet/extracted``.
    """
    sa = root / "SA-418 (13th cent)" / "SA-418"
    for i in range(12):
        stem = f"{i:04d}_SA-418-{23577 + i}"
        (sa / f"{stem}.xml").parent.mkdir(parents=True, exist_ok=True)
        (sa / f"{stem}.xml").write_bytes(page_xml([line_xml("l", LONG)], image=f"{stem}.jpeg"))
        if i != 11:
            write_jpeg(sa / f"{stem}.jpeg", (1000, 800))
    bnf = root / "BnF Arabe 76 pp 68-78"
    bnf.mkdir(parents=True)
    (bnf / "BnF Paris Ara 76-041_2L.xml").write_bytes(page_xml([line_xml("l", LONG)], NS19, image="BnF Paris Ara 76-041_2L.tif"))
    Image.new("L", (1000, 800), 200).save(bnf / "BnF Paris Ara 76-041_2L.tif", format="TIFF")
    (bnf / "METS.xml").write_bytes(b"<mets/>")
    sin = root / "Sin423 (17th cent)" / "Sin423"
    sin.mkdir(parents=True)
    (sin / "a.xml").write_bytes(page_xml([line_xml("l", LONG)]))


def test_agapet_sources(tmp_path: Path) -> None:
    """agapet_sources lists the kept collections, excludes Sin423 and METS.xml, and marks pages without an image."""
    make_agapet(tmp_path)
    sources = bal.agapet_sources(tmp_path)
    assert len(sources) == 13
    assert {s.manuscript for s in sources} == {"sa418", "bnf76"}
    bnf = [s for s in sources if s.manuscript == "bnf76"][0]
    assert bnf.stem == "agapet_bnf76_bnf_paris_ara_76_041_2l" and bnf.image_path.suffix == ".tif"
    assert bnf.key == "agapet/bnf76/BnF Paris Ara 76-041_2L"
    assert sum(s.image_path is None for s in sources) == 1


def make_rasam(root: Path, stems: Sequence[str], fetched: Sequence[str]) -> None:
    """Synthetic RASAM tree (XML frame 2000x1600, served 1000x800).

    :param root: ``rasam_pages``.
    :param stems: Page stems (``rasam1``).
    :param fetched: Stems with an image and a manifest row.
    """
    (root / "page" / "rasam1").mkdir(parents=True)
    (root / "images").mkdir(parents=True)
    with open(root / "list-images.tsv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["Library", "Dataset", "Manuscript", "FileName", "Bibliographical record", "IIIF manifest",
                         "IIIF image ID", "Width (px)", "Height (px)"])
        for stem in stems:
            writer.writerow(["BULAC", "RASAM1", "MS.ARA.609", stem, "", "", "1", "2000", "1600"])
            (root / "page" / "rasam1" / f"{stem}.xml").write_bytes(page_xml([line_xml("l", LONG)], width=2000, height=1600))
    with open(root / "images" / "manifest.jsonl", "w", encoding="utf-8") as fh:
        for stem in fetched:
            write_jpeg(root / "images" / f"{stem}.jpg", (1000, 800))
            fh.write(json.dumps({"stem": stem, "url": "u", "sha256": "s", "bytes": 1, "served_width": 1000,
                                 "served_height": 800, "xml_width": 2000, "xml_height": 1600}) + "\n")


def test_rasam_sources(tmp_path: Path) -> None:
    """rasam_sources takes manuscripts from the listing and images from the manifest."""
    make_rasam(tmp_path, ["BULAC_MS_ARA_609_00004", "BULAC_MS_ARA_609_00005"], ["BULAC_MS_ARA_609_00005"])
    sources = bal.rasam_sources(tmp_path)
    assert [s.stem for s in sources] == ["rasam_bulac_ms_ara_609_00004", "rasam_bulac_ms_ara_609_00005"]
    assert sources[0].image_path is None and sources[1].served == (1000, 800)
    assert {s.manuscript for s in sources} == {"MS.ARA.609"}


def test_read_manifest_skips_an_unfinished_last_line(tmp_path: Path) -> None:
    """A last manifest line without a newline (a fetcher still appending) is not read until it is finished."""
    path = tmp_path / "manifest.jsonl"
    rows = [{"stem": "a", "served_width": 1, "served_height": 2}, {"stem": "b", "served_width": 3, "served_height": 4}]
    path.write_text("".join(json.dumps(r) + "\n\n" for r in rows) + '{"stem": "c", "served_wi', encoding="utf-8")
    assert bal.read_manifest(path) == {"a": rows[0], "b": rows[1]}
    path.write_text(path.read_text(encoding="utf-8") + 'dth": 5, "served_height": 6}\n', encoding="utf-8")
    assert set(bal.read_manifest(path)) == {"a", "b", "c"}
    assert bal.read_manifest(tmp_path / "missing.jsonl") == {}


def make_iskandar(root: Path) -> None:
    """Synthetic Iskandar tree: one fetched Gotha page with text, one layout-only, one unfetched.

    :param root: ``iskandar_pages``.
    """
    (root / "XML-pages").mkdir(parents=True)
    (root / "images").mkdir(parents=True)
    rows = [("Forschungsbibliothek_Gotha", "Ms-orient-A-02385", "Ms_orient_A_02385_004.xml"),
            ("Forschungsbibliothek_Gotha", "Ms-orient-A-02390", "Ms_orient_A_02390_006.xml"),
            ("BnF", "AR.3684", "btv1b10031486f_1.xml")]
    with open(root / "iskandar-images-list.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Library", "Manuscript", "XML_File_Name", "Bibliographical_Record", "IIIF_manifest", "Image_id",
                         "Height (px)", "Width (px)"])
        for library, ms, name in rows:
            writer.writerow([library, ms, name, "", "", "1", "800", "1000"])
    (root / "XML-pages" / rows[0][2]).write_bytes(page_xml([line_xml("l", LONG), line_xml("m", "")]))
    (root / "XML-pages" / rows[1][2]).write_bytes(page_xml([line_xml("l", ""), line_xml("m", "")]))
    (root / "XML-pages" / rows[2][2]).write_bytes(page_xml([line_xml("l", LONG)]))
    with open(root / "images" / "manifest.jsonl", "w", encoding="utf-8") as fh:
        for _, _, name in rows[:2]:
            stem = Path(name).stem
            write_jpeg(root / "images" / f"{stem}.jpg", (1000, 800))
            fh.write(json.dumps({"stem": stem, "served_width": 1000, "served_height": 800,
                                 "xml_width": 1000, "xml_height": 800}) + "\n")


def test_iskandar_sources(tmp_path: Path) -> None:
    """iskandar_sources takes manuscripts from the CSV and leaves unfetched pages without an image."""
    make_iskandar(tmp_path)
    sources = {s.key: s for s in bal.iskandar_sources(tmp_path)}
    assert sources["iskandar/Ms_orient_A_02385_004"].manuscript == "Ms-orient-A-02385"
    assert sources["iskandar/btv1b10031486f_1"].image_path is None


# ----------------------------------------------------------------------------- baybars crops

def crop_png(width: int = 60, height: int = 20, ink: bool = True) -> bytes:
    """A polygon-style crop: black fill in the corners, paper inside, one black ink pixel inside.

    :param width: Width.
    :param height: Height.
    :param ink: Draw the interior ink pixel.
    :return: PNG bytes.
    """
    rgb = np.full((height, width, 3), PAPER, np.uint8)
    rgb[:4, :8] = 0
    rgb[-4:, -8:] = 0
    if ink:
        rgb[10, 30] = 0
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def test_paper_fill_replaces_border_fill_only() -> None:
    """paper_fill paints the black border fill in paper colour, keeps ink, and passes fill-free crops through."""
    rgb = np.asarray(Image.open(io.BytesIO(crop_png())).convert("RGB"))
    out = bal.paper_fill(rgb)
    assert tuple(out[0, 0]) == PAPER and tuple(out[-1, -1]) == PAPER
    assert tuple(out[10, 30]) == (0, 0, 0)
    plain = np.full((5, 5, 3), 100, np.uint8)
    assert bal.paper_fill(plain) is plain


def write_baybars(data_dir: Path, name: str, rows: Sequence[Dict[str, Any]]) -> None:
    """Write one synthetic Baybars parquet file.

    :param data_dir: ``baybars/data``.
    :param name: File name.
    :param rows: Rows (manuscript, image, region, text).
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "image": [{"bytes": crop_png(), "path": "x.png"} for _ in rows],
        "transcription": [r["text"] for r in rows],
        "support_type": ["paper" for _ in rows],
        "region_type": [r["region"] for r in rows],
        "manuscript_name": [r["ms"] for r in rows],
        "image_name": [r["image"] for r in rows],
        "script": ["naskh" for _ in rows],
    })
    pq.write_table(table, data_dir / name)


def make_baybars(data_dir: Path) -> None:
    """Train: two pages (one with a margin note and a Latin line); validation and test: one page each.

    :param data_dir: ``baybars/data``.
    """
    train = [{"ms": "BnF Arabe 3757", "image": "f12.jpg", "region": "MainText", "text": "بِسْمِ اللَّهِ"},
             {"ms": "BnF Arabe 3757", "image": "f12.jpg", "region": "MarginText", "text": LONG},
             {"ms": "BnF Arabe 3757", "image": "f12.jpg", "region": "MainText_Right", "text": "ABC وكتب"},
             {"ms": "BnF Arabe 3757", "image": "f13.jpg", "region": "MainText_Left", "text": LONG}]
    write_baybars(data_dir, "train-00000-of-00001.parquet", train)
    write_baybars(data_dir, "validation-00000-of-00001.parquet",
                  [{"ms": "Gotha A1", "image": "p.1.jpg", "region": "MainText", "text": LONG}])
    write_baybars(data_dir, "test-00000-of-00001.parquet",
                  [{"ms": "Gotha A1", "image": "p2.jpg", "region": "MainText", "text": "[صلعم] قال"}])


def test_build_crop_set(tmp_path: Path) -> None:
    """build_crop_set writes paper-filled main-text crops with cleaned GT under the official split, dot-free stems."""
    make_baybars(tmp_path / "data")
    results = bal.build_crop_set("baybars", tmp_path / "data", tmp_path / "out", workers=2)
    by_key = {r.key: r for r in results}
    assert set(by_key) == {"baybars/BnF Arabe 3757/f12.jpg", "baybars/BnF Arabe 3757/f13.jpg",
                           "baybars/Gotha A1/p.1.jpg", "baybars/Gotha A1/p2.jpg"}
    page = by_key["baybars/BnF Arabe 3757/f12.jpg"]
    assert page.split == "train" and page.counts == Counter({bal.KEPT: 1, "region MarginText": 1, bal.DROP_LATIN: 1})
    png = page.paths[0]
    assert png.name == "baybars_train_bnf_arabe_3757_f12_00000.png" and png.is_absolute()
    assert png.with_name(png.stem + ".gt.txt").read_text(encoding="utf-8") == "بسم الله"
    assert tuple(np.asarray(Image.open(png))[0, 0]) == PAPER
    val = by_key["baybars/Gotha A1/p.1.jpg"]
    assert val.split == "val" and val.paths[0].name == "baybars_validation_gotha_a1_p_1_00000.png"
    assert by_key["baybars/Gotha A1/p2.jpg"].skip == "no kept lines"
    assert all(p.name.count(".") == 1 for r in results for p in r.paths)


def test_build_crop_set_limit_per_split(tmp_path: Path) -> None:
    """build_crop_set applies the page limit per output split."""
    make_baybars(tmp_path / "data")
    results = bal.build_crop_set("baybars", tmp_path / "data", tmp_path / "out", limit=1, workers=1)
    assert [(r.split, r.key) for r in results] == [("train", "baybars/BnF Arabe 3757/f12.jpg"),
                                                   ("val", "baybars/Gotha A1/p.1.jpg")]
    assert len(list((tmp_path / "out" / "crops" / "baybars").glob("*.png"))) == 2


# ----------------------------------------------------------------------------- end to end

def make_source_root(root: Path) -> Path:
    """All four synthetic sets under one source root.

    :param root: Source root.
    :return: The root.
    """
    make_agapet(root / "agapet" / "extracted")
    make_rasam(root / "rasam_pages", [f"BULAC_MS_ARA_609_{i:05d}" for i in range(3)],
               [f"BULAC_MS_ARA_609_{i:05d}" for i in range(2)])
    make_iskandar(root / "iskandar_pages")
    make_baybars(root / "baybars" / "data")
    return root


def test_build_writes_lists_and_stats(tmp_path: Path) -> None:
    """build writes absolute-path lists and per-set statistics for all four sets."""
    source_root = make_source_root(tmp_path / "src")
    out = tmp_path / "out"
    stats = bal.build(["agapet", "rasam", "iskandar", "baybars"], source_root, out, workers=2)
    agapet = stats["agapet"]
    assert agapet["pages_seen"] == 13 and agapet["pages_kept"] == 12
    assert agapet["pages_skipped"] == {"no image": 1}
    assert agapet["splits"]["val"]["pages"] == 1 and agapet["manuscripts"]["sa418"]["val"] == 1
    assert agapet["arabic_letters_kept"] == 12 * bal.arabic_letter_count(LONG)
    train = (out / "lists" / "agapet_train.lst").read_text(encoding="utf-8").splitlines()
    val = (out / "lists" / "agapet_val.lst").read_text(encoding="utf-8").splitlines()
    assert len(train) + len(val) == 12 and all(Path(p).is_absolute() and Path(p).exists() for p in train + val)
    rasam = stats["rasam"]
    assert rasam["pages_kept"] == 2 and rasam["pages_skipped"] == {"no image": 1} and rasam["edits"]["rescaled pages"] == 2
    root = etree.fromstring(Path(((out / "lists" / "rasam_train.lst").read_text() +
                                  (out / "lists" / "rasam_val.lst").read_text()).split()[0]).read_bytes())
    assert root.find("{*}Page").get("imageWidth") == "1000"
    iskandar = stats["iskandar"]
    assert iskandar["pages_kept"] == 1 and iskandar["pages_skipped"] == {"no kept lines": 1, "no image": 1}
    assert iskandar["lines_dropped"] == {bal.DROP_EMPTY: 3}
    baybars = stats["baybars"]
    assert baybars["lines_seen"] == 6 and baybars["lines_kept"] == 3
    assert baybars["splits"]["val"] == {"pages": 1, "lines": 1, "arabic_letters": bal.arabic_letter_count(LONG)}
    crops = (out / "lists" / "baybars_val.lst").read_text(encoding="utf-8").split()
    assert len(crops) == 1 and crops[0].endswith(".png")
    on_disk = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    assert set(on_disk) == {"agapet", "rasam", "iskandar", "baybars"}


def test_build_merges_stats_across_runs(tmp_path: Path) -> None:
    """build merges a later run's set statistics into the existing stats.json."""
    source_root = make_source_root(tmp_path / "src")
    out = tmp_path / "out"
    bal.build(["rasam"], source_root, out, workers=1)
    stats = bal.build(["iskandar"], source_root, out, limit=1, workers=1)
    assert set(stats) == {"rasam", "iskandar"} and stats["iskandar"]["limit"] == 1


def test_build_writes_stats_after_each_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A crash while building a later set leaves stats.json holding the sets whose lists were written."""
    source_root = make_source_root(tmp_path / "src")
    out = tmp_path / "out"
    real_build_set = bal.build_set

    def build_or_crash(name: str, *args: Any) -> List[bal.PageResult]:
        """Build a set, but crash on Iskandar.

        :param name: Set name.
        :param args: The remaining :func:`bal.build_set` arguments.
        :return: Page results.
        """
        if name == "iskandar":
            raise RuntimeError("crashed mid-run")
        return real_build_set(name, *args)

    monkeypatch.setattr(bal, "build_set", build_or_crash)
    with pytest.raises(RuntimeError):
        bal.build(["rasam", "iskandar"], source_root, out, workers=1)
    assert set(json.loads((out / "stats.json").read_text(encoding="utf-8"))) == {"rasam"}
    assert (out / "lists" / "rasam_train.lst").exists() and not (out / "lists" / "iskandar_train.lst").exists()


def test_main_cli(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """The CLI builds the requested set under the limit and prints its statistics."""
    source_root = make_source_root(tmp_path / "src")
    bal.main(["--sets", "agapet", "--source-root", str(source_root), "--out-root", str(tmp_path / "out"),
              "--limit", "3", "--workers", "1"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["agapet"]["pages_kept"] == 3
    assert len(list((tmp_path / "out" / "page_xml" / "agapet").glob("*.xml"))) == 3
