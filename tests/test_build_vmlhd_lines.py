# File name: test_build_vmlhd_lines.py
# Date: 10/28/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for :mod:`src.finetuning.kraken.build_vmlhd_lines` on synthetic VML-HD pages."""
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pytest
from lxml import etree
from PIL import Image, ImageDraw

from src.finetuning.kraken import build_vmlhd_lines as bv

CORE = 40


def paw(x: int, cy: int, w: int = 40, h: int = CORE, text: str = "من") -> bv.Paw:
    """A PAW box centred vertically on ``cy``.

    :param x: Left edge.
    :type x: int
    :param cy: Vertical centre.
    :type cy: int
    :param w: Width.
    :type w: int
    :param h: Height.
    :type h: int
    :param text: Transcript.
    :type text: str
    :return: The PAW.
    :rtype: bv.Paw
    """
    return bv.Paw(x, cy - h // 2, w, h, text)


def row(cy: int, texts: Sequence[str], right: int = 560, step: int = 60) -> List[bv.Paw]:
    """Core PAWs of one line, right to left.

    :param cy: Line centre.
    :type cy: int
    :param texts: Transcripts in reading order.
    :type texts: Sequence[str]
    :param right: Right edge of the first PAW.
    :type right: int
    :param step: Horizontal pitch.
    :type step: int
    :return: PAWs.
    :rtype: List[bv.Paw]
    """
    return [paw(right - 40 - i * step, cy, text=t) for i, t in enumerate(texts)]


def write_page(tmp_path: Path, paws: Sequence[bv.Paw], size: Tuple[int, int] = (600, 400)) -> Tuple[Path, Path]:
    """VML-HD style XML plus a white PNG with every box filled black.

    :param tmp_path: Directory.
    :type tmp_path: Path
    :param paws: PAWs to write.
    :type paws: Sequence[bv.Paw]
    :param size: Image size.
    :type size: Tuple[int, int]
    :return: ``(xml path, png path)``.
    :rtype: Tuple[Path, Path]
    """
    root = etree.Element("ArrayOfDocumentElement")
    for i, p in enumerate(paws):
        el = etree.SubElement(root, "DocumentElement")
        for tag, val in (("ID", i), ("ElementType", "PartOfWord"), ("X", p.x), ("Y", p.y), ("Width", p.w),
                         ("Height", p.h), ("Transcript", p.text), ("OriginX", 0), ("OriginY", 0)):
            etree.SubElement(el, tag).text = str(val)
    other = etree.SubElement(root, "DocumentElement")
    etree.SubElement(other, "ElementType").text = "Line"
    xml = tmp_path / "0001-1.xml"
    xml.write_bytes(etree.tostring(root))
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    for p in paws:
        d.rectangle([p.x + 2, p.y + 2, p.right - 2, p.bottom - 2], fill="black")
    png = tmp_path / "0001-1.png"
    im.save(png)
    return xml, png


# ----------------------------------------------------------------------------- source and lines

def test_load_paws_reads_part_of_word_only(tmp_path: Path) -> None:
    xml, _ = write_page(tmp_path, row(100, ["من", "قال"]))
    paws = bv.load_paws(xml)
    assert [p.text for p in paws] == ["من", "قال"]
    assert paws[0].right == paws[0].x + paws[0].w


def test_density_peaks_two_clusters() -> None:
    peaks = bv.density_peaks([100, 101, 99, 200, 202, 198], bandwidth=5, min_sep=20)
    assert len(peaks) == 2
    assert abs(peaks[0] - 100) < 3 and abs(peaks[1] - 200) < 3


def test_group_lines_orders_and_keeps_tall_alif_on_its_line() -> None:
    a = row(100, ["ا", "ب", "ج", "د", "ه"])
    b = row(200, ["و", "ز", "ح", "ط", "ي"])
    alif = bv.Paw(300, 110, 12, 110, "ا")        # reaches 10 px into line A's band, 40 px into B's
    lines = bv.group_lines(a + b + [alif])
    assert len(lines) == 2
    assert alif in lines[1].paws and alif not in lines[0].paws
    assert [p.text for p in lines[0].paws] == ["ا", "ب", "ج", "د", "ه"]   # right to left


def test_group_lines_recovers_a_line_of_tall_letters() -> None:
    a = row(100, ["ب"] * 5)
    b = row(200, ["ت"] * 5)
    tall = [paw(500 - i * 60, 320, h=80, text="ل") for i in range(4)]   # no core PAW on this line
    lines = bv.group_lines(a + b + tall)
    assert len(lines) == 3
    assert set(tall) == set(lines[2].paws)


def test_split_at_gaps() -> None:
    paws = [paw(500, 100), paw(440, 100), paw(100, 100)]   # 300 px gap = 7.5 core heights
    segs = bv.split_at_gaps(paws, CORE)
    assert [len(s) for s in segs] == [2, 1]
    assert len(bv.split_at_gaps(paws[:2], CORE)) == 1


# ----------------------------------------------------------------------------- spaces

@pytest.fixture
def model() -> bv.WordModel:
    """Small word model.

    :return: Model over three words.
    :rtype: bv.WordModel
    """
    return bv.WordModel(Counter({"بالله": 50, "من": 100, "قال": 80}))


def test_fold_word() -> None:
    assert bv.fold_word("أَإِلى") == "االي"
    assert bv.fold_word("رحمة") == "رحمه"


def test_join_paws_certain_space_after_joining_letter(model: bv.WordModel) -> None:
    paws = [paw(500, 100, text="من"), paw(400, 100, text="قال")]   # ن joins left: word end
    assert bv.join_paws(paws, CORE, model) == "من قال"


def test_join_paws_merges_known_word_across_alif(model: bv.WordModel) -> None:
    paws = [paw(500, 100, text="با"), paw(440, 100, text="لله")]
    assert bv.join_paws(paws, CORE, model) == "بالله"


def test_join_paws_never_leaves_a_lone_letter(model: bv.WordModel) -> None:
    paws = [paw(500, 100, text="و"), paw(400, 100, text="من")]
    assert bv.join_paws(paws, CORE, model) == "ومن"


def test_segment_run_splits_two_frequent_words(model: bv.WordModel) -> None:
    words = bv.segment_run(["من", "قال"], [20.0], CORE, model)   # boundary forced ambiguous here
    assert words == ["من", "قال"]


def test_gap_prior_sign() -> None:
    assert bv.gap_prior(0.0, CORE) < 0 < bv.gap_prior(CORE * 0.5, CORE)


# ----------------------------------------------------------------------------- geometry

def test_polygon_covers_boxes_and_band() -> None:
    line = bv.Line(row(100, ["ب", "ت", "ث"]), a=100.0, b=0.0, xm=300.0, core=CORE)
    pts = bv.line_polygon(line, 600, 400)
    xs, ys = [x for x, _ in pts], [y for _, y in pts]
    assert min(xs) <= min(p.x for p in line.paws) and max(xs) >= max(p.right for p in line.paws)
    assert min(ys) <= 80 - bv.PAD + 1 and max(ys) >= 120 + bv.PAD - 1
    assert pts[0][0] < pts[len(pts) // 2 - 1][0]   # top edge runs left to right first


def test_baseline_left_to_right_inside_band() -> None:
    line = bv.Line(row(100, ["ا", "ب", "ت"]), a=100.0, b=0.0, xm=300.0, core=CORE)
    (x0, y0), (x1, y1) = bv.line_baseline(line, page_offset=10.0, width=600, height=400)
    assert x0 < x1
    assert 100 <= y0 <= 100 + CORE / 2 and 100 <= y1 <= 100 + CORE / 2


# ----------------------------------------------------------------------------- gates

def test_line_reason() -> None:
    ok = bv.Line([paw(0, 100, text="من")], 100, 0, 0, CORE)
    empty = bv.Line([paw(0, 100, text="")], 100, 0, 0, CORE)
    symbol = bv.Line([paw(0, 100, text="م1")], 100, 0, 0, CORE)
    assert bv.line_reason(ok) == ""
    assert bv.line_reason(empty) == bv.DROP_EMPTY_PAW
    assert bv.line_reason(symbol) == bv.DROP_SYMBOL


def test_frame_ok() -> None:
    assert bv.frame_ok([paw(10, 100)], 600, 400)
    assert not bv.frame_ok([paw(590, 100)], 600, 400)


def test_unannotated_ink_flags_black_blob_only() -> None:
    paws = row(100, ["ب"] * 5, step=100) + row(250, ["ت"] * 5, step=100)   # boxes 520-560, 420-460, ...
    im = Image.new("RGB", (600, 400), "white")
    d = ImageDraw.Draw(im)
    for p in paws:
        d.rectangle([p.x + 2, p.y + 2, p.right - 2, p.bottom - 2], fill="black")
    d.rectangle([475, 88, 505, 112], fill="black")             # unannotated letter in line A's gap
    d.rectangle([475, 238, 505, 262], fill=(30, 40, 120))      # dark blue rosette in line B: not ink
    d.rectangle([388, 248, 391, 251], fill="black")            # a dot in line B: too small
    lines = bv.group_lines(paws)
    missing = bv.unannotated_ink(lines, im, paws)
    assert missing[0][0] >= bv.MAX_MISSING_BLOB
    assert missing[1] == (0.0, 0.0)


# ----------------------------------------------------------------------------- end to end

def test_process_page_writes_page_xml(tmp_path: Path, model: bv.WordModel) -> None:
    paws = row(100, ["من", "قال", "من", "قال", "من"]) + row(250, ["با", "لله", "من", "قال", "من"])
    xml, png = write_page(tmp_path, paws)
    out = tmp_path / "out"
    out.mkdir()
    result = bv.process_page("book1", xml, png, out, model)
    assert result.skip == "" and result.counts[bv.bal.KEPT] == 2
    root = etree.parse(str(result.paths[0])).getroot()
    texts = [u.text for u in root.iter("{*}Unicode")]
    assert texts == ["من قال من قال من", "بالله من قال من"]
    for tl in root.iter("{*}TextLine"):
        assert tl.find("{*}Baseline") is not None and len(tl.find("{*}Coords").get("points").split()) >= 4
    assert root.find("{*}Page").get("imageFilename") == str(png.absolute())


def test_process_page_without_image(tmp_path: Path, model: bv.WordModel) -> None:
    xml, _ = write_page(tmp_path, row(100, ["من"] * 3))
    assert bv.process_page("book1", xml, None, tmp_path, model).skip == "no image"
