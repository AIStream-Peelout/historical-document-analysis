# File name: test_build_arabic_external.py
# Date: 10/5/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for the external Arabic page-set builder (PAGE XML, text rules, line stacking, rows)."""
import io
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset, load_from_disk
from PIL import Image

from src.finetuning.qwen_hebrew import build_arabic_external as bae
from src.finetuning.qwen_hebrew.build_ktiv_dataset import FEATURES
from src.finetuning.qwen_hebrew.build_pgp_arabic_editions import ARABIC_FRAGMENT_PROMPT
from src.finetuning.qwen_hebrew.images_once import JPEG_MAGIC, export_images_once

NS13 = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
NS19 = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
PAPER = (200, 180, 150)
LONG = "وكتب الكتاب في المدينة"          # 18 Arabic letters


def _points(box: Tuple[int, int, int, int]) -> str:
    """PAGE ``points`` of a rectangle."""
    x0, y0, x1, y1 = box
    return f"{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1}"


def transkribus_xml(regions: Sequence[Tuple[str, Tuple[int, int, int, int], Sequence[Tuple[int, Optional[str], Tuple[int, int, int, int]]]]],
                    order: Sequence[str], image: str = "page.jpeg") -> bytes:
    """A Transkribus-style PAGE 2013 file.

    :param regions: ``(id, box, lines)`` in document order; a line is ``(readingOrder index, text or None, box)``.
    :param order: Region ids in reading order (written with the indexes in reverse document order).
    :param image: ``imageFilename``.
    :return: XML bytes.
    """
    refs = "".join(f'<RegionRefIndexed index="{i}" regionRef="{rid}"/>' for i, rid in reversed(list(enumerate(order))))
    body = []
    for rid, box, lines in regions:
        body.append(f'<TextRegion id="{rid}" custom="readingOrder {{index:0;}}"><Coords points="{_points(box)}"/>')
        for idx, text, lbox in lines:
            equiv = "" if text is None else f"<TextEquiv><Unicode>{text}</Unicode></TextEquiv>"
            body.append(f'<TextLine id="{rid}_{idx}" custom="readingOrder {{index:{idx};}}"><Coords points="{_points(lbox)}"/>{equiv}</TextLine>')
        body.append("<TextEquiv><Unicode></Unicode></TextEquiv></TextRegion>")
    return (f'<?xml version="1.0" encoding="UTF-8"?><PcGts xmlns="{NS13}"><Page imageFilename="{image}" imageWidth="1000" '
            f'imageHeight="800"><ReadingOrder><OrderedGroup id="ro">{refs}</OrderedGroup></ReadingOrder>{"".join(body)}'
            f"</Page></PcGts>").encode("utf-8")


def _line(i: int, text: Optional[str], x0: int = 100, x1: int = 400) -> Tuple[int, Optional[str], Tuple[int, int, int, int]]:
    """A Transkribus line ``i`` of a region, 30 pixels per line."""
    return i, text, (x0, 100 + 30 * i, x1, 125 + 30 * i)


MUHARAF_XML = f"""<?xml version="1.0" ?>
<PcGts xmlns="{NS19}"><Page imageFilename="p1.jpg" imageWidth="1000" imageHeight="1200">
<ReadingOrder><OrderedGroup id="g">
  <OrderedGroupIndexed id="rec1" index="1"><RegionRefIndexed regionRef="region_2" index="1"/><RegionRefIndexed regionRef="region_1" index="0"/></OrderedGroupIndexed>
  <OrderedGroupIndexed id="rec0" index="0"><RegionRefIndexed regionRef="region_0" index="0"/></OrderedGroupIndexed>
</OrderedGroup></ReadingOrder>
<TextRegion id="region_0" type="floating"><Coords points="{_points((800, 20, 950, 60))}"/>
  <TextLine id="line_0" index="0"><Coords points="{_points((800, 20, 950, 60))}"/><TextEquiv><Unicode>AF-291</Unicode></TextEquiv></TextLine>
</TextRegion>
<TextRegion id="region_1" type="paragraph"><Coords points="{_points((100, 100, 900, 600))}"/>
  <TextLine id="line_5" index="1"><Coords points="{_points((100, 200, 900, 260))}"/><TextEquiv><Unicode>السطر الثاني</Unicode></TextEquiv></TextLine>
  <TextLine id="line_4" index="0"><Coords points="{_points((100, 100, 900, 160))}"/><TextEquiv><Unicode>السطر الأول</Unicode></TextEquiv></TextLine>
</TextRegion>
<GraphicRegion id="region_2" type="stamp"><Coords points="{_points((100, 700, 400, 900))}"/>
  <TextRegion id="sub_region_2" type="paragraph"><Coords points="{_points((100, 700, 400, 900))}"/>
    <TextLine id="line_9" index="0"><Coords points="{_points((100, 700, 400, 760))}"/><TextEquiv><Unicode>ختم المحكمة</Unicode></TextEquiv></TextLine>
  </TextRegion>
</GraphicRegion>
<TextRegion id="region_3" type="signature-mark"><Coords points="{_points((500, 1000, 900, 1100))}"/>
  <TextLine id="line_7" index="0"><Coords points="{_points((500, 1000, 900, 1100))}"/><TextEquiv><Unicode>التوقيع</Unicode></TextEquiv></TextLine>
</TextRegion>
</Page></PcGts>""".encode("utf-8")


def _jpeg(size: Tuple[int, int] = (40, 30), color: Tuple[int, int, int] = PAPER, mode: str = "RGB", fmt: str = "JPEG", **save) -> bytes:
    """Encoded test image."""
    buf = io.BytesIO()
    Image.new(mode, size, color if mode == "RGB" else 128).save(buf, fmt, **save)
    return buf.getvalue()


def _crop(h: int, w: int, ink: Optional[Tuple[int, int, int]] = None, black_corners: bool = False) -> np.ndarray:
    """A line crop: paper, an ink block in the middle, optionally the black polygon fill in two corners."""
    a = np.empty((h, w, 3), np.uint8)
    a[:] = PAPER
    if ink:
        a[h // 3: 2 * h // 3, w // 4: 3 * w // 4] = ink
    if black_corners:
        a[:3, :3] = 0
        a[-3:, -3:] = 0
    return a


def _png(a: np.ndarray) -> bytes:
    """PNG bytes of an array."""
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "PNG")
    return buf.getvalue()


# ----------------------------------------------------------------------------- text

def test_clean_line_removes_marks_tatweel_and_extra_space_and_keeps_letters():
    assert bae.clean_line("اللـه") == "الله"
    assert bae.clean_line("مُحَمَّدٌ وَسَلَّمَ") == "محمد وسلم"
    assert bae.clean_line("ا\u0654مر") == "أمر"                          # combining hamza composes under NFC
    assert bae.clean_line(" \tكتب \u00A0 الله\n") == "كتب الله"
    assert bae.clean_line("\u200Fسلام\u200E") == "سلام"                  # invisible direction marks
    kept = "أ إ آ ؤ ئ ء ة ى شࢨ ه\u0654 ١٢ . ،"                          # hamza forms, dots, digits, punctuation
    assert bae.clean_line(kept) == kept


def test_resolve_brackets_turns_restorations_into_gaps_and_counts_notes():
    assert bae.resolve_brackets("[على الكنـ]يسه") == (" [...] يسه", 1, 0)
    line, restored, notes = bae.resolve_brackets("[ا] [ب]ـتاب و[]")
    assert bae.clean_line(line) == "[...] تاب و [...]" and restored == 3 and notes == 0
    assert bae.resolve_brackets("يعجب [فوق السطر: من] حكمته")[1:] == (0, 1)
    assert bae.resolve_brackets("[فوقها اسمعيل]")[2] == 1 and bae.resolve_brackets("[مكرره]")[2] == 1
    assert bae.resolve_brackets("[193ب]")[2] == 1
    assert bae.resolve_brackets("كلمة [ناقصة")[2] == 1                    # unpaired


def test_clean_page_lines_keeps_positions_and_counts():
    lines, counts = bae.clean_page_lines(["", "  ـ ", "نص [مفقود] هنا", "[اقرأ: ابى]"])
    assert lines == ["", "", "نص [...] هنا", "[اقرأ: ابى]"]
    assert counts["empty lines dropped"] == 2 and counts["restorations"] == 1 and counts["editorial notes"] == 1
    kept, counts = bae.clean_page_lines(["[صلعم]", "نص [مفقود] هنا"], editorial=False)   # brackets the scribe drew
    assert kept == ["[صلعم]", "نص [مفقود] هنا"] and counts["restorations"] == counts["editorial notes"] == 0


def test_page_gate():
    assert bae.page_gate([LONG, LONG]) == "fewer than 3 lines"
    assert bae.page_gate(["ابت", "ثجح", "خدذ"]) == "fewer than 40 Arabic letters"
    assert bae.page_gate([LONG, LONG, LONG]) == ""


# ----------------------------------------------------------------------------- PAGE XML

def test_parse_transkribus_follows_reading_order_not_document_order():
    xml = transkribus_xml([("r_margin", (20, 100, 80, 200), [_line(0, "حاشية", 20, 80)]),
                           ("r_main", (100, 90, 400, 300), [_line(2, "ثالث"), _line(0, "اول"), _line(1, "ثاني")])],
                          order=["r_main", "r_margin"])
    page = bae.parse_page_xml(xml)
    assert [r.region_id for r in page.regions] == ["r_main", "r_margin"]
    assert [line.text for line in page.lines()] == ["اول", "ثاني", "ثالث", "حاشية"]
    assert page.image_filename == "page.jpeg" and (page.width, page.height) == (1000, 800) and page.unordered_regions == 0


def test_parse_page_2019_records_nested_stamp_text_line_index_and_unordered_region():
    page = bae.parse_page_xml(MUHARAF_XML)
    assert [line.text for line in page.lines()] == ["AF-291", "السطر الأول", "السطر الثاني", "ختم المحكمة", "التوقيع"]
    assert [r.region_id for r in page.regions] == ["region_0", "region_1", "sub_region_2", "region_3"]
    assert page.unordered_regions == 1                                   # region_3 is not in the reading order


def test_parse_without_reading_order_keeps_document_order():
    xml = f"""<PcGts xmlns="{NS19}"><Page imageFilename="x.tif" imageWidth="10" imageHeight="10">
      <TextRegion id="a"><TextLine id="l1"><Coords points="0,0 5,0 5,2 0,2"/><TextEquiv><Unicode>اول</Unicode></TextEquiv></TextLine>
      <TextLine id="l2"><Coords points="0,3 5,3 5,5 0,5"/><TextEquiv><Unicode>ثاني</Unicode></TextEquiv></TextLine></TextRegion>
      <TextRegion id="b"><TextLine id="l3"><Coords points="0,6 5,6 5,8 0,8"/></TextLine></TextRegion></Page></PcGts>""".encode()
    page = bae.parse_page_xml(xml)
    assert [line.text for line in page.lines()] == ["اول", "ثاني", ""] and page.unordered_regions == 0
    assert page.regions[1].bbox == (0, 6, 5, 8)                           # a region without Coords takes its lines' box


def _double_page(order: Sequence[str]) -> bae.PageXml:
    """A two-page image: left page text block, right page text block, a page number above the left page."""
    left = ("left", (50, 80, 450, 700), [_line(i, f"يسار {LONG} {i}", 60, 440) for i in range(6)])
    right = ("right", (550, 80, 950, 700), [_line(i, f"يمين {LONG} {i}", 560, 940) for i in range(6)])
    number = ("number", (240, 20, 260, 50), [(0, "56", (240, 20, 260, 50))])
    return bae.parse_page_xml(transkribus_xml([left, right, number], order))


def test_right_page_first_moves_the_right_page_of_a_left_first_double_page():
    regions, moved = bae.right_page_first(_double_page(["number", "left", "right"]).regions)
    assert moved and [r.region_id for r in regions] == ["right", "number", "left"]
    lines, counts, reason = bae.xml_page_lines(_double_page(["number", "left", "right"]), double_page=True)
    assert reason == "" and lines[0].startswith("يمين") and lines[6] == "56" and lines[-1].startswith("يسار")
    assert counts["left page listed first: reordered"] == 1


def test_right_page_first_keeps_a_right_first_page_and_single_pages():
    regions, moved = bae.right_page_first(_double_page(["right", "number", "left"]).regions)
    assert not moved and [r.region_id for r in regions] == ["right", "number", "left"]
    margin_first = bae.parse_page_xml(transkribus_xml([("margin", (20, 80, 90, 700), [_line(i, "حاشية", 20, 90) for i in range(6)]),
                                                       ("main", (100, 80, 900, 700), [_line(i, LONG) for i in range(6)])],
                                                      ["margin", "main"]))
    assert bae.right_page_first(margin_first.regions) == (margin_first.regions, False)   # a margin column is not a page
    overlapping = bae.parse_page_xml(transkribus_xml([("a", (100, 80, 600, 400), [_line(i, LONG, 100, 600) for i in range(6)]),
                                                      ("b", (300, 420, 900, 700), [_line(i, LONG, 300, 900) for i in range(6)])],
                                                     ["a", "b"]))
    assert bae.right_page_first(overlapping.regions) == (overlapping.regions, False)     # one page, two blocks


def test_missed_text_line_flags_a_wide_untranscribed_line_only():
    wide = [bae.PageLine(LONG, (0, 0, 400, 20)), bae.PageLine(LONG, (0, 30, 380, 50)), bae.PageLine("", (0, 60, 300, 80))]
    small = wide[:2] + [bae.PageLine("", (0, 60, 60, 80)), bae.PageLine("\u064E", (0, 90, 50, 100))]
    assert bae.missed_text_line(wide) and not bae.missed_text_line(small)
    page = bae.parse_page_xml(transkribus_xml([("r", (100, 80, 400, 300), [_line(0, LONG), _line(1, None), _line(2, LONG)])], ["r"]))
    assert bae.xml_page_lines(page)[2] == "an untranscribed line as wide as a text line"


def test_xml_page_lines_cleans_restores_and_skips_notes():
    ok = bae.parse_page_xml(transkribus_xml([("r", (100, 80, 400, 300), [_line(0, f"{LONG} [مفقود]"), _line(1, LONG),
                                                                          _line(2, "اللـهُ " + LONG), _line(3, None, 100, 130)])], ["r"]))
    lines, counts, reason = bae.xml_page_lines(ok)
    assert reason == "" and lines == [f"{LONG} [...]", LONG, "الله " + LONG]
    noted = bae.parse_page_xml(transkribus_xml([("r", (100, 80, 400, 300), [_line(i, f"{LONG} [فوق السطر: من]") for i in range(3)])], ["r"]))
    assert bae.xml_page_lines(noted)[2] == "editorial note in brackets"
    drawn = bae.parse_page_xml(transkribus_xml([("r", (100, 80, 400, 300), [_line(0, f"[ {LONG}"), _line(1, f"{LONG} [...] هنا"),
                                                                             _line(2, LONG)])], ["r"]))
    assert bae.xml_page_lines(drawn)[2] == "editorial note in brackets"                 # an unpaired bracket, read as an editor's
    lines, counts, reason = bae.xml_page_lines(drawn, editorial=False)                  # Muharaf: the writer drew it
    assert reason == "" and lines == [f"[ {LONG}", f"{LONG} [...] هنا", LONG] and counts["editorial notes"] == 0


# ----------------------------------------------------------------------------- images

def test_page_image_keeps_an_upright_jpeg_verbatim():
    data = _jpeg((40, 30))
    assert bae.page_image(data, max_pixels=10_000) == (data, 40, 30, "verbatim")


def test_page_image_turns_exif_rotation_upright():
    exif = Image.Exif()
    exif[0x0112] = 6                                                      # rotate 90 degrees to display
    out, width, height, how = bae.page_image(_jpeg((40, 30), exif=exif), max_pixels=10_000)
    assert (width, height, how) == (30, 40, "re-encoded")
    with Image.open(io.BytesIO(out)) as im:
        assert im.size == (30, 40) and im.getexif().get(0x0112) is None


def test_page_image_downscales_to_the_budget_and_converts_tiff_to_rgb_jpeg():
    out, width, height, how = bae.page_image(_jpeg((400, 300)), max_pixels=30_000)
    assert how == "downscaled" and width * height <= 30_000 and abs(width / height - 4 / 3) < 0.02
    out, width, height, how = bae.page_image(_jpeg((50, 40), mode="L", fmt="TIFF"), max_pixels=10_000)
    assert out.startswith(JPEG_MAGIC) and how == "re-encoded"
    with Image.open(io.BytesIO(out)) as im:
        assert im.mode == "RGB" and im.size == (50, 40)


def test_fill_mask_takes_only_black_connected_to_the_border():
    a = _crop(20, 30)
    a[:4, :4] = 0
    a[10, 15] = 0                                                         # black ink inside the polygon
    mask = bae.fill_mask(a)
    assert mask[:4, :4].all() and not mask[10, 15] and mask.sum() == 16
    assert not bae.fill_mask(_crop(5, 5)).any()


def test_paper_color_ignores_ink_and_fill():
    a = _crop(30, 30, ink=(20, 20, 20), black_corners=True)
    assert np.allclose(bae.paper_color(a, bae.fill_mask(a)), PAPER)


def test_stack_lines_order_right_alignment_size_background_and_resolution():
    crops = [_crop(30, 100, ink=(255, 0, 0), black_corners=True), _crop(40, 60, ink=(0, 255, 0)), _crop(20, 80, ink=(0, 0, 255))]
    page, boxes = bae.stack_lines(crops, margin=10, gap=5, softness=0)
    assert page.shape == (30 + 40 + 20 + 2 * 5 + 2 * 10, 100 + 2 * 10, 3)
    assert boxes == [(10, 10, 110, 40), (50, 45, 110, 85), (30, 90, 110, 110)]   # top to bottom, right edges aligned
    assert tuple(page[0, 0]) == PAPER                                    # background from the crops, not white
    assert tuple(page[10, 10]) == PAPER                                  # the black polygon fill became paper
    for (x0, y0, x1, y1), crop, ink in zip(boxes, crops, [(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        assert tuple(page[(y0 + y1) // 2, (x0 + x1) // 2]) == ink
        assert np.array_equal(page[y0 + 3:y1 - 3, x0 + 3:x1 - 3], crop[3:-3, 3:-3])   # crop pixels at their own resolution
    soft, _ = bae.stack_lines(crops)
    assert soft.shape == (30 + 40 + 20 + 2 * 4 + 2 * 24, 100 + 2 * 24, 3)   # default margin and gap from the median line height
    assert tuple(soft[24 + 15, 148 - 24 - 50]) == (255, 0, 0)


def _slanted(h: int = 60, w: int = 200, thickness: int = 12) -> np.ndarray:
    """A crop of a line rising to the right: a polygon band, black fill above and below it, ink inside."""
    a = _crop(h, w)
    for c in range(w):
        top = round((h - thickness) * (1 - c / (w - 1)))
        a[top + 3:top + thickness - 3, c] = (40, 40, 40) if c % 7 < 4 else PAPER
        a[:top, c] = 0
        a[top + thickness:, c] = 0
    return a


def test_stack_lines_nests_slanted_lines_without_overlapping_them():
    crops = [_slanted(), _slanted()]
    page, boxes = bae.stack_lines(crops, margin=10, gap=4, softness=0)
    assert boxes == [(10, 10, 210, 70), (10, 26, 210, 86)]               # boxes would stack at y=74; the bands nest at 26
    assert page.shape == (86 + 10, 220, 3)
    occupied = np.zeros(page.shape[:2], int)
    for crop, (x0, y0, x1, y1) in zip(crops, boxes):
        inside = ~bae.fill_mask(crop)
        occupied[y0:y1, x0:x1] += inside
        assert np.array_equal(page[y0:y1, x0:x1][inside], crop[inside])   # every polygon pixel kept as is
    assert occupied.max() == 1                                            # no two lines share a pixel
    top, bottom = bae.band_profile(bae.fill_mask(crops[0]))
    assert top[0] == 48 and bottom[0] == 59 and top[-1] == 0 and bottom[-1] == 11


def test_stack_page_encodes_a_jpeg_of_the_reported_size():
    data, width, height = bae.stack_page([_png(_crop(30, 100, ink=(0, 0, 0))), _png(_crop(30, 80))])
    assert data.startswith(JPEG_MAGIC)
    with Image.open(io.BytesIO(data)) as im:
        assert im.size == (width, height) == (100 + 2 * 24, 60 + 4 + 2 * 24)


# ----------------------------------------------------------------------------- line datasets

def test_group_pages_by_manuscript_and_image_in_row_order():
    pages = bae.group_pages(["m1", "m1", "m2", "m1", "m2", "m3"], ["p1", "p2", "p1", "p1", "p1", "p1"])
    assert list(pages.items()) == [(("m1", "p1"), [0, 3]), (("m1", "p2"), [1]), (("m2", "p1"), [2, 4]), (("m3", "p1"), [5])]


IMAGE_TYPE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])


def _line_table(rows: Sequence[Tuple[bytes, str, str, str, str]]) -> pa.Table:
    """A line-release table: ``(crop, transcription, region_type, manuscript_name, image_name)`` per row."""
    return pa.table({"image": pa.array([{"bytes": r[0], "path": None} for r in rows], IMAGE_TYPE),
                     "transcription": [r[1] for r in rows], "support_type": ["manuscript"] * len(rows),
                     "region_type": [r[2] for r in rows], "manuscript_name": [r[3] for r in rows],
                     "image_name": [r[4] for r in rows], "script": ["oriental"] * len(rows)})


def test_page_crops_keeps_main_text_lines_and_their_crops_aligned():
    table = _line_table([(b"a", "سطر اول", "MainText", "m", "p"), (b"b", "كلمة", "Marginalia_Catchword", "m", "p"),
                         (b"c", "\u064E", "MainText", "m", "p"), (b"d", "سطر ثان", "MainText_Right", "m", "p"),
                         (b"e", "[صلعم]", "MainText", "m", "p")])
    lines, crops, counts = bae.page_crops(table, [3, 0, 1, 2, 4], bae.MAIN_TEXT_REGIONS["baybars"])
    assert lines == ["سطر ثان", "سطر اول", "[صلعم]"] and crops == [b"d", b"a", b"e"]   # scribe's brackets kept
    assert counts["dropped region Marginalia_Catchword"] == 1 and counts["empty lines dropped"] == 1


def test_build_line_dataset_keeps_the_split_and_stacks_each_page(tmp_path):
    crop = _png(_crop(30, 120, ink=(30, 30, 30), black_corners=True))
    data = tmp_path / "data"
    data.mkdir()
    pq.write_table(_line_table([(crop, f"{LONG} ا", "MainText", "ms", "p1"), (crop, "٣٢", "page_number", "ms", "p1"),
                                (crop, f"{LONG} ب", "MainText", "ms", "p2"), (crop, f"{LONG} ج", "MainText", "ms", "p1"),
                                (crop, f"{LONG} د", "MainText", "ms", "p1")]), data / "train-00000-of-00001.parquet")
    pq.write_table(_line_table([(crop, f"{LONG} ه", "MainText", "ms", "p1"), (crop, f"{LONG} و", "MainText", "ms", "p1")]),
                   data / "validation-00000-of-00001.parquet")
    pq.write_table(_line_table([(crop, f"{LONG} ز", "MainText", "ms", "p1")]), data / "test-00000-of-00001.parquet")
    out = tmp_path / "arabic_external_iskandar_v1"
    stats = bae.build_line_dataset("iskandar", out, data_dir=data, workers=2)
    dsd = load_from_disk(str(out))
    assert set(dsd) == {"train", "val"} and dsd["train"].features == FEATURES
    train, val = dsd["train"][0], dsd["val"][0]
    assert train["stem"] == "iskandar_ms_p1_train" and train["answer"].split("\n") == [f"{LONG} ا", f"{LONG} ج", f"{LONG} د"]
    assert val["stem"] == "iskandar_ms_p1_val" and val["answer"].split("\n") == [f"{LONG} ه", f"{LONG} و", f"{LONG} ز"]
    assert train["image"].size == (train["image_width"], train["image_height"]) == (120 + 2 * 24, 3 * 30 + 2 * 4 + 2 * 24)
    assert train["section"] == "arabic_stacked_lines_unordered" and train["label_source"] == "iskandar"
    assert stats["skipped_pages"] == {"train: fewer than 3 lines": 1} and stats["counts"]["dropped region page_number"] == 1
    assert stats["splits"]["train"]["pages"] == 1 and stats["splits"]["val"]["lines"] == 3
    manifest = export_images_once(out, tmp_path / "export")
    assert manifest["splits"]["train"]["rows"] == 1 and manifest["splits"]["val"]["rows"] == 1


# ----------------------------------------------------------------------------- rows

def test_page_row_matches_the_ktiv_schema_and_the_pgp_arabic_task():
    data = _jpeg((10, 20))
    row = bae.page_row(data, "سطر\nسطر", "agapet_x", 10, 20, "agapet", "arabic_page")
    assert list(row) == list(FEATURES)
    ds = Dataset.from_list([row], features=FEATURES)
    got = ds[0]
    assert got["image"].size == (10, 20) and got["question"] == ARABIC_FRAGMENT_PROMPT and got["task"] == "fragment_transcribe"
    assert got["target_chars"] == len("سطر\nسطر") and got["target_tokens"] == 0 and got["label_source"] == "agapet"
    assert ds.features["image_width"].dtype == "int32"


def test_save_rows_writes_arrow_batches_without_from_list_and_swaps_atomically(tmp_path, monkeypatch):
    def no_from_list(*args, **kwargs):
        raise AssertionError("Dataset.from_list fingerprints by hashing every image byte")
    monkeypatch.setattr(Dataset, "from_list", no_from_list)
    images = {i: _jpeg((10 + i, 20)) for i in range(5)}
    train = bae.RowBatches(batch_rows=2)
    for i in range(5):
        train.append(bae.page_row(images[i], f"سطر {i}\nسطر", f"s{i}", 10 + i, 20, "agapet", "arabic_page"))
    assert len(train) == 5 and len(train.batches) == 2 and len(train.pending) == 1   # converted as it fills
    out = tmp_path / "arabic_external_x_v1"
    bae.save_rows({"train": train, "val": bae.RowBatches()}, out, {"stats.json": "{}"})
    dsd = load_from_disk(str(out))
    assert list(dsd) == ["train"] and dsd["train"].features == FEATURES     # empty splits are not written
    assert [r["stem"] for r in dsd["train"]] == [f"s{i}" for i in range(5)]
    assert [r["image"].size for r in dsd["train"]] == [(10 + i, 20) for i in range(5)]
    assert dsd["train"][4]["answer"] == "سطر 4\nسطر" and (out / "stats.json").read_text() == "{}"
    again = bae.RowBatches()
    again.append(bae.page_row(images[0], "جديد", "new", 10, 20, "agapet", "arabic_page"))
    bae.save_rows({"train": again}, out, {})
    assert [r["stem"] for r in load_from_disk(str(out))["train"]] == ["new"] and not (out / "stats.json").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["arabic_external_x_v1"]   # no .building / .previous left
    manifest = export_images_once(out, tmp_path / "export")
    assert manifest["splits"] == {"train": {"rows": 1, "unique_images": 1, "row_image_bytes": len(images[0]), "image_bytes": len(images[0])}}


def test_ranked_val_pages_is_exact_per_group_and_depends_only_on_the_keys():
    groups = {"big": [f"b{i}" for i in range(340)], "mid": [f"m{i}" for i in range(239)], "small": [f"s{i}" for i in range(11)]}
    val = bae.ranked_val_pages(groups, 0.05)
    assert [sum(k in val for k in keys) for keys in groups.values()] == [17, 12, 1]     # every group, exact share
    assert bae.ranked_val_pages({"mid": list(reversed(groups["mid"]))}, 0.05) == {k for k in val if k.startswith("m")}
    assert bae.ranked_val_pages(groups, 0.0) == set() and bae.ranked_val_pages({"none": []}, 0.05) == set()


def test_row_stems_are_unique_when_slugs_collide():
    ids = bae.row_stems("muharaf", ["BEK1A_27_01r", "BEK1A_27_01r-", "AF_291r"])
    assert ids["AF_291r"] == "muharaf_af_291r" and len(set(ids.values())) == 3
    assert ids["BEK1A_27_01r"].startswith("muharaf_bek1a_27_01r_") and ids["BEK1A_27_01r-"].startswith("muharaf_bek1a_27_01r_")
    assert bae.row_stems("muharaf", ["BEK1A_27_01r-", "BEK1A_27_01r", "AF_291r"]) == ids        # independent of order


def test_hash_split_and_slug():
    assert bae.slug("0002_tn_Sinai  MF UCL Arabe 423 (2)") == "0002_tn_sinai_mf_ucl_arabe_423_2"
    assert bae.hash_split("x", 0.0) == "train" and bae.hash_split("x", 1.0) == "val"
    assert bae.hash_split("agapet_sa418_0005", 0.05) == bae.hash_split("agapet_sa418_0005", 0.05)


# ----------------------------------------------------------------------------- muharaf

def test_official_page_splits_follow_unambiguous_line_votes():
    pages = {"A": ["سطر واحد", "سطر اثنان", "مشترك"], "B": ["سطر ثلاثة", "مشترك"], "C": ["سطر اربعة", "سطر خمسة"]}
    splits = {"train": ["سطر  واحد ", "مشترك", "سطر خمسة"], "validation": ["سطر ثلاثة", "مشترك"], "test": ["سطر اربعة"]}
    assert bae.official_page_splits(pages, splits) == {"A": "train", "B": "val"}   # C has votes for both


def test_json_only_lines_reports_transcribed_lines_the_xml_lost():
    annotation = {"region_dict": {"r1": {"lines": ["line_1", "line_2", "line_3"]}},
                  "json": {"line_1": {"text": "نص اول"}, "line_2": {"text": "عنوان مفقود"}, "line_3": {"text": " . . . "}}}
    assert bae.json_only_lines(annotation, ["نص  اول"]) == ["عنوان مفقود"]


def _muharaf_page(name: str, lines: Sequence[str]) -> bytes:
    """A Muharaf-style PAGE 2019 file with one paragraph region."""
    body = "".join(f'<TextLine id="line_{i}" index="{i}"><Coords points="{_points((100, 100 + 50 * i, 900, 140 + 50 * i))}"/>'
                   f"<TextEquiv><Unicode>{t}</Unicode></TextEquiv></TextLine>" for i, t in enumerate(lines))
    return (f'<PcGts xmlns="{NS19}"><Page imageFilename="{name}.jpg" imageWidth="64" imageHeight="48"><ReadingOrder>'
            f'<OrderedGroup id="g"><OrderedGroupIndexed id="r" index="0"><RegionRefIndexed regionRef="region_0" index="0"/>'
            f'</OrderedGroupIndexed></OrderedGroup></ReadingOrder><TextRegion id="region_0" type="paragraph">{body}</TextRegion>'
            f"</Page></PcGts>").encode("utf-8")


def test_build_muharaf_reads_the_zip_in_place_and_uses_the_official_split(tmp_path):
    lines = {"p1": [f"{LONG} {i}" for i in range(3)], "p2": [f"{LONG} ثاني {i}" for i in range(3)],
             "p3": [f"{LONG} ثالث {i}" for i in range(3)]}
    zip_path = tmp_path / "public_data_files.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for name, texts in lines.items():
            archive.writestr(f"public/{name}.xml", _muharaf_page(name, texts))
            archive.writestr(f"public/{name}.jpg", _jpeg((64, 48)))
            extra = ["عنوان في فاصل"] if name == "p3" else []                # p3's XML export lost a line
            ids = [f"line_{i}" for i in range(len(texts) + len(extra))]
            archive.writestr(f"public/{name}.json", json.dumps({"region_dict": {"region_text": {"lines": ids}},
                                                                "json": {i: {"text": t} for i, t in zip(ids, texts + extra)}}))
    line_dir = tmp_path / "lines"
    line_dir.mkdir()
    pq.write_table(pa.table({"text": lines["p1"]}), line_dir / "train-00000-of-00001.parquet")
    pq.write_table(pa.table({"text": lines["p2"][:1]}), line_dir / "test-00000-of-00001.parquet")
    pq.write_table(pa.table({"text": pa.array([], pa.string())}), line_dir / "validation-00000-of-00001.parquet")
    out = tmp_path / "arabic_external_muharaf_v1"
    stats = bae.build_muharaf(out, zip_path=zip_path, line_dir=line_dir, workers=2, max_density=float("inf"))
    dsd = load_from_disk(str(out))
    assert [r["stem"] for r in dsd["train"]] == ["muharaf_p1"] and [r["stem"] for r in dsd["val"]] == ["muharaf_p2"]
    assert dsd["train"][0]["answer"] == "\n".join(lines["p1"]) and dsd["train"][0]["section"] == "arabic_page"
    assert stats["skipped_pages"] == {"the XML export lost transcribed lines of the annotation JSON": 1}
    assert stats["split_from"] == {"official": 2} and stats["images"] == {"verbatim": 2}
    dense = bae.build_muharaf(tmp_path / "dense", zip_path=zip_path, line_dir=line_dir, workers=2)   # 64 x 48 thumbnails
    assert dense["skipped_pages"]["text too dense for the image"] == 2 and dense["splits"] == {}


# ----------------------------------------------------------------------------- agapet

def test_build_agapet_unzips_reorders_double_pages_and_skips_editorial_notes(tmp_path):
    left = ("left", (50, 80, 450, 700), [_line(i, f"يسار {LONG} {i}", 60, 440) for i in range(6)])
    right = ("right", (550, 80, 950, 700), [_line(i, f"يمين {LONG} {i}", 560, 940) for i in range(6)])
    noted = ("r", (100, 80, 400, 300), [_line(i, f"{LONG} [فوق السطر: من]") for i in range(3)])
    with zipfile.ZipFile(tmp_path / "Col (test).zip", "w") as archive:
        archive.writestr("Col/0001_a.xml", transkribus_xml([left, right], ["left", "right"], image="0001_a.jpeg"))
        archive.writestr("Col/0001_a.jpeg", _jpeg((64, 48)))
        archive.writestr("Col/0002_b.xml", transkribus_xml([noted], ["r"], image="0002_b.jpeg"))
        archive.writestr("Col/0002_b.jpeg", _jpeg((64, 48)))
    out = tmp_path / "arabic_external_agapet_v1"
    stats = bae.build_agapet(out, source_dir=tmp_path, val_share=0.0, workers=2, collections={"col": "Col (test).zip"},
                             double_page=frozenset({"col"}), max_density=float("inf"))
    assert (tmp_path / "extracted" / "Col (test)" / "Col" / "0001_a.jpeg").exists()
    row = load_from_disk(str(out))["train"][0]
    assert row["stem"] == "agapet_col_0001_a" and row["answer"].split("\n")[0].startswith("يمين")
    assert row["answer"].split("\n")[-1].startswith("يسار") and row["label_source"] == "agapet"
    assert stats["skipped_pages"] == {"col: editorial note in brackets": 1}
    assert stats["double_pages_reordered_right_page_first"] == ["0001_a"]
    assert bae.extract_zip(tmp_path / "Col (test).zip", tmp_path / "extracted" / "Col (test)") == 0   # nothing re-written
    dense = bae.build_agapet(tmp_path / "dense", source_dir=tmp_path, val_share=0.0, workers=2,
                             collections={"col": "Col (test).zip"}, double_page=frozenset({"col"}))
    assert dense["skipped_pages"]["col: text too dense for the image"] == 1 and dense["splits"] == {}


def test_too_dense_is_letters_per_megapixel_of_the_stored_image():
    lines = ["كتاب" * 250, "كتاب" * 250]                                   # 2,000 Arabic letters
    assert bae.too_dense(lines, 1132, 800) and not bae.too_dense(lines, 2000, 1500)
    assert not bae.too_dense(lines, 1132, 800, limit=3000) and bae.too_dense(lines, 2000, 1000, limit=999)
