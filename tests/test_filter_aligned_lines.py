# File name: test_filter_aligned_lines.py
# Date: 10/20/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Unit tests for the aligned-line filter (src/finetuning/kraken/filter_aligned_lines.py).

Everything runs on synthetic pairs rows, a tiny PAGE XML and in-memory images under ``tmp_path``; nothing
reads the NAS.
"""
import json
from pathlib import Path
from typing import List, Optional

import pytest
from PIL import Image, ImageDraw

from src.finetuning.kraken import filter_aligned_lines as fal

PAGE_XML = """<?xml version='1.0' encoding='UTF-8'?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15" \
xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" \
xsi:schemaLocation="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 x.xsd">
  <Metadata><Creator>test</Creator></Metadata>
  <Page imageFilename="{img}" imageWidth="400" imageHeight="300">
    <TextRegion id="r0">
      <Coords points="0,0 400,0 400,300 0,300"/>
      <TextLine id="l0" custom="alignment {{tier:anchor;}}">
        <Coords points="10,40 390,40 390,90 10,90"/>
        <Baseline points="10,80 390,80"/>
        <TextEquiv><Unicode>بسم الله</Unicode></TextEquiv>
      </TextLine>
      <TextLine id="l1" custom="alignment {{tier:rejected;}}">
        <Coords points="10,100 390,100 390,150 10,150"/>
        <Baseline points="10,140 390,140"/>
        <TextEquiv><Unicode></Unicode></TextEquiv>
      </TextLine>
      <TextLine id="l2_3" custom="alignment {{tier:anchor;}}">
        <Coords points="10,160 390,160 390,210 10,210"/>
        <Baseline points="10,200 390,200"/>
        <TextEquiv><Unicode>الحمد لله ٣</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
"""


def _row(stem: str = "p1", lid: str = "l0", written: bool = True, ext_r: float = 0.0,
         limited: Optional[List[bool]] = None, ooi: float = 0.0, lost: float = 0.0) -> dict:
    """Synthetic pairs row with the fields the filter reads.

    :param stem: page stem.
    :param lid: line id.
    :param written: placed flag.
    :param ext_r: ``extended_lh_right``.
    :param limited: ``block_limited`` [left, right].
    :param ooi: ``own_ink_outside``.
    :param lost: ``lost``.
    :return: row dict.
    """
    return {"stem": stem, "line_id": lid, "written": written, "tier": "anchor" if written else "rejected",
            "extended_lh_right": ext_r, "extended_lh_left": 0.0, "block_limited": limited or [False, False],
            "own_ink_outside": ooi, "lost": lost, "line_height": 50.0, "segments": [0], "text": "x"}


def _write_run(tmp_path: Path, rows: List[dict]) -> Path:
    """Tiny run folder: one page XML, pairs.jsonl and genizah lists.

    :param tmp_path: pytest temp dir.
    :param rows: pairs rows.
    :return: run folder.
    """
    run = tmp_path / "run"
    (run / "page_xml" / "genizah").mkdir(parents=True)
    (run / "lists").mkdir()
    xml = run / "page_xml" / "genizah" / "p1.xml"
    xml.write_text(PAGE_XML.format(img="/abs/img/p1.jpg"), encoding="utf-8")
    xml2 = run / "page_xml" / "genizah" / "p2.xml"
    xml2.write_text(PAGE_XML.format(img="/abs/img/p2.jpg"), encoding="utf-8")
    with open(run / "pairs.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (run / "lists" / "genizah_train.lst").write_text(str(xml) + "\n")
    (run / "lists" / "genizah_val.lst").write_text(str(xml2) + "\n")
    return run


# ------------------------------------------------------------------------------------------------ drop rules

@pytest.mark.parametrize("row,expected", [
    (_row(), []),
    (_row(ext_r=0.79), []),
    (_row(ext_r=0.8), ["ext_right"]),
    (_row(ext_r=0.3, limited=[False, True]), ["block_right"]),
    (_row(ext_r=0.29, limited=[False, True]), []),
    (_row(ext_r=0.5, limited=[True, False]), []),
    (_row(ooi=0.10), []),
    (_row(ooi=0.1001), ["own_ink_outside"]),
    (_row(lost=0.01), ["lost"]),
    (_row(ext_r=0.9, limited=[False, True], lost=0.2), ["lost", "ext_right", "block_right"]),
])
def test_drop_reasons_thresholds(row: dict, expected: List[str]) -> None:
    """Each threshold bites at the stated boundary and reasons come in REASON_ORDER."""
    rules = fal.FilterRules(max_lost=0.0, max_own_ink_outside=0.10, max_ext_right_lh=0.8, block_ext_right_lh=0.3)
    assert fal.drop_reasons(row, rules) == expected


def test_drop_reasons_page_named_and_neighbour() -> None:
    """Page and named drops come first; neighbour ink counts only when measured."""
    rules = fal.FilterRules(drop_pages={"p1"}, drop_lines={("p1", "l0")}, neighbour_ink_max=0.02)
    assert fal.drop_reasons(_row(), rules, neighbour_ink=0.02) == ["page", "named", "neighbour_ink"]
    assert fal.drop_reasons(_row(stem="p2"), rules, neighbour_ink=0.019) == []
    assert fal.drop_reasons(_row(stem="p2"), rules, neighbour_ink=None) == []


def test_rules_off_when_none() -> None:
    """A ``None`` threshold switches its rule off."""
    rules = fal.FilterRules(max_lost=None, max_own_ink_outside=None)
    assert fal.drop_reasons(_row(ext_r=5, limited=[True, True], ooi=1, lost=1), rules) == []


def test_decide_and_tally() -> None:
    """First-reason and any-reason tallies differ when a line trips two rules."""
    rows = [_row(lid="a", ext_r=0.9, lost=0.1), _row(lid="b"), _row(lid="c", ooi=0.5), _row(lid="d", written=False,
                                                                                               lost=0.9)]
    placed = fal.placed_rows(rows)
    assert list(placed) == [("p1", "a"), ("p1", "b"), ("p1", "c")]
    dec = fal.decide(placed, fal.FilterRules(max_ext_right_lh=0.8))
    t = fal.tally(dec)
    assert t["first"]["lost"] == 1 and t["first"]["ext_right"] == 0 and t["first"]["own_ink_outside"] == 1
    assert t["any"]["ext_right"] == 1 and t["any"]["lost"] == 1
    assert set(t["first"]) == set(fal.REASON_ORDER)


def test_parse_line_key() -> None:
    """Line specs split at the last colon."""
    assert fal.parse_line_key("pgpar_33840_1_page:l13_14") == ("pgpar_33840_1_page", "l13_14")
    with pytest.raises(ValueError):
        fal.parse_line_key("no_colon")


def test_arabic_letters() -> None:
    """Only U+0621..U+064A count: no digits, spaces, harakat or Latin."""
    assert fal.arabic_letters("بسم الله ٣ a" + "َ") == 7


# ------------------------------------------------------------------------------------------------ XML

def test_remove_text_lines_keeps_everything_else() -> None:
    """Dropped TextLines vanish; the empty rejected line, regions, metadata and imageFilename stay."""
    src = PAGE_XML.format(img="/abs/img/p1.jpg").encode("utf-8")
    out, removed = fal.remove_text_lines(src, {"l2_3", "zz"})
    assert removed == ["l2_3"]
    text = out.decode("utf-8")
    assert 'imageFilename="/abs/img/p1.jpg"' in text
    assert 'id="l1"' in text and 'id="l2_3"' not in text
    assert text.startswith("<?xml") and "<PcGts xmlns=\"http://schema.primaresearch.org" in text
    assert "xsi:schemaLocation" in text and "<Creator>test</Creator>" in text
    assert fal.text_lines(out) == [("l0", "بسم الله")]


def test_text_lines_and_polygons() -> None:
    """Only non-empty lines are reported, with integer polygons."""
    src = PAGE_XML.format(img="/i.jpg").encode("utf-8")
    assert [lid for lid, _ in fal.text_lines(src)] == ["l0", "l2_3"]
    img, polys = fal.line_polygons(src)
    assert img == "/i.jpg" and set(polys) == {"l0", "l2_3"} and polys["l0"][0] == (10, 40)


# ------------------------------------------------------------------------------------------------ run level

def test_filter_run_end_to_end(tmp_path: Path) -> None:
    """Dropped lines leave the XML, emptied pages are not written, lists keep the source split."""
    rows = [_row("p1", "l0"), _row("p1", "l1", written=False), _row("p1", "l2_3", ext_r=1.0),
            _row("p2", "l0", lost=0.2), _row("p2", "l2_3", ooi=0.3)]
    run = _write_run(tmp_path, rows)
    out = tmp_path / "out"
    rules = fal.FilterRules(drop_lines={("p9", "l1")}, drop_pages={"p7"}, max_ext_right_lh=0.8)
    rep = fal.filter_run(run, out, rules)
    assert rep["placed_lines_in"] == 4 and rep["kept_lines"] == 1 and rep["dropped_lines"] == 3
    assert rep["dropped_by_reason"]["first"]["ext_right"] == 1
    assert rep["pages_emptied"] == ["p2"]
    assert rep["drop_lines_not_placed"] == ["p9:l1"]
    assert rep["drop_pages_without_placed_lines"] == [{"stem": "p7", "in_pairs": False}]
    assert rep["letters_kept"] == {"train": 7, "val": 0}
    train = (out / "lists" / "train.lst").read_text().split()
    assert train == [str((out / "page_xml" / "genizah" / "p1.xml").resolve())]
    assert (out / "lists" / "val.lst").read_text() == ""
    kept = fal.text_lines(Path(train[0]).read_bytes())
    assert kept == [("l0", "بسم الله")]
    assert 'id="l1"' in Path(train[0]).read_text(encoding="utf-8")
    assert json.loads((out / "filter_report.json").read_text())["kept_lines"] == 1
    # the source run is untouched
    assert 'id="l2_3"' in (run / "page_xml" / "genizah" / "p1.xml").read_text(encoding="utf-8")


def test_filter_run_drop_page(tmp_path: Path) -> None:
    """A dropped page is neither written nor listed."""
    run = _write_run(tmp_path, [_row("p1", "l0"), _row("p2", "l0")])
    rep = fal.filter_run(run, tmp_path / "out", fal.FilterRules(drop_pages={"p1"}))
    assert rep["dropped_by_reason"]["first"]["page"] == 1
    assert not (tmp_path / "out" / "page_xml" / "genizah" / "p1.xml").exists()
    assert rep["pages_out"] == {"train": 0, "val": 1}


def test_filter_run_refuses_inside_source(tmp_path: Path) -> None:
    """The out-root may not lie inside the source run, and an existing non-empty output is refused."""
    run = _write_run(tmp_path, [_row("p1", "l0")])
    with pytest.raises(ValueError):
        fal.filter_run(run, run / "clean", fal.FilterRules())
    fal.filter_run(run, tmp_path / "out", fal.FilterRules())
    with pytest.raises(FileExistsError):
        fal.filter_run(run, tmp_path / "out", fal.FilterRules())


def test_combine_lists(tmp_path: Path) -> None:
    """Combined lists concatenate roots and totals count pages, lines and letters."""
    run = _write_run(tmp_path, [_row("p1", "l0"), _row("p1", "l2_3"), _row("p2", "l0")])
    a = tmp_path / "a"
    fal.filter_run(run, a, fal.FilterRules())
    tot = fal.combine_lists(tmp_path / "lists", {"only": [a]})
    assert tot["only"]["train"] == {"pages": 1, "lines": 2, "letters": 7 + 8}
    assert tot["only"]["val"]["pages"] == 1 and tot["only"]["val"]["lines"] == 2
    assert (tmp_path / "lists" / "only_train.lst").read_text().split() == (a / "lists" / "train.lst").read_text().split()
    with pytest.raises(ValueError):
        fal.combine_lists(tmp_path / "lists", {"dup": [a, a]})
    assert fal.parse_groups(["x=/a,/b"]) == {"x": [Path("/a"), Path("/b")]}


# ------------------------------------------------------------------------------------------------ neighbour ink

def _page(blobs: List[tuple]) -> Image.Image:
    """Page 400x300 with ink rectangles (dark core, mid-grey rim so Otsu's threshold lands above the core).

    :param blobs: rectangles (x0, y0, x1, y1).
    :return: RGB image.
    """
    im = Image.new("RGB", (400, 300), (225, 215, 195))
    d = ImageDraw.Draw(im)
    for b in blobs:
        d.rectangle(b, fill=(20, 20, 20), outline=(70, 70, 70))
    return im


def test_baseline_y_at() -> None:
    """Baseline is interpolated inside and carried flat up to the reach."""
    bl = [[100, 50], [0, 10]]
    assert fal.baseline_y_at(bl, 50, 0) == pytest.approx(30.0)
    assert fal.baseline_y_at(bl, 120, 30) == pytest.approx(50.0)
    assert fal.baseline_y_at(bl, 140, 30) is None


def test_neighbour_ink_area_counts_other_line_ink() -> None:
    """Ink sitting on the other baseline inside the polygon counts; ink on the own baseline does not."""
    lh = 40.0
    own = {"index": 0, "baseline": [[20, 150], [380, 150]]}
    other = {"index": 1, "baseline": [[20, 100], [380, 100]]}
    poly = [(20, 95), (380, 95), (380, 155), (20, 155)]
    clean = _page([(50, 130, 120, 150)])
    gray, blue = fal.page_arrays(clean)
    area, comps = fal.neighbour_ink_area(poly, lh, {0}, [own, other], gray, blue)
    assert area == 0.0 and comps == []
    dirty = _page([(50, 130, 120, 150), (200, 112, 240, 124)])  # centroid 18 px (0.45 lh) below the other baseline
    gray, blue = fal.page_arrays(dirty)
    area, comps = fal.neighbour_ink_area(poly, lh, {0}, [own, other], gray, blue)
    assert len(comps) == 1 and abs(comps[0][1] - 118) <= 2
    inside = 39 * 11  # dark core rows 113..123, columns 201..239, all inside the polygon
    assert area == pytest.approx(inside / lh ** 2, rel=0.15)
    # the audit's rule ignores other baselines within 0.4 lh of the centroid (ink sitting on the other baseline)
    on_line = _page([(50, 130, 120, 150), (200, 95, 240, 104)])
    gray, blue = fal.page_arrays(on_line)
    assert fal.neighbour_ink_area(poly, lh, {0}, [own, other], gray, blue)[0] == 0.0


def test_neighbour_ink_ignores_mount() -> None:
    """Blue mount pixels are never ink."""
    im = _page([(50, 130, 120, 150)])
    ImageDraw.Draw(im).rectangle((200, 112, 240, 124), fill=(20, 40, 160))
    gray, blue = fal.page_arrays(im)
    assert blue[118, 220] and not blue[140, 80]
    own = {"index": 0, "baseline": [[20, 150], [380, 150]]}
    other = {"index": 1, "baseline": [[20, 100], [380, 100]]}
    area, _ = fal.neighbour_ink_area([(20, 95), (380, 95), (380, 155), (20, 155)], 40.0, {0}, [own, other], gray, blue)
    assert area == 0.0


# ------------------------------------------------------------------------------------------------ custom-attribute runs

@pytest.mark.parametrize("lid,custom,expected", [
    ("l4", "alignment {tier:anchor; sim:0.74; len:0.98; edition:1; span:0.74; lost:0.00;}", []),
    ("l4_5", "alignment {tier:anchor; sim:0.74; lost:0.00;}", ["merged"]),
    ("l4", "alignment {tier:anchor; sim:0.74; lost:0.01;}", ["lost"]),
    ("l4", "alignment {tier:anchor; sim:0.74;}", ["lost"]),
    ("l4", "alignment {tier:interpolated; lost:0.00;}", ["tier"]),
    ("l4_6", "alignment {tier:count_match; ratio:1.2;}", ["tier", "lost", "merged"]),
    ("l4", None, ["tier", "lost"]),
])
def test_custom_drop_reasons(lid: str, custom: Optional[str], expected: List[str]) -> None:
    """Keep only anchor, lost == 0 and unmerged ids; a missing lost is dropped."""
    assert fal.custom_drop_reasons(lid, custom, {"anchor"}) == expected


def test_parse_custom() -> None:
    """Custom attribute pairs parse to raw strings."""
    kv = fal.parse_custom("alignment {tier:anchor; sim:0.85; extended:5/0;}")
    assert kv == {"tier": "anchor", "sim": "0.85", "extended": "5/0"}


def test_filter_custom_run(tmp_path: Path) -> None:
    """The custom-attribute mode removes the non-anchor/merged lines and keeps the source split."""
    run = _write_run(tmp_path, [])
    for name in ("p1.xml", "p2.xml"):
        f = run / "page_xml" / "genizah" / name
        txt = f.read_text(encoding="utf-8")
        txt = txt.replace('id="l0" custom="alignment {tier:anchor;}"', 'id="l0" custom="alignment {tier:anchor; lost:0.00;}"')
        txt = txt.replace('id="l2_3" custom="alignment {tier:anchor;}"', 'id="l2_3" custom="alignment {tier:anchor; lost:0.00;}"')
        f.write_text(txt, encoding="utf-8")
    rep = fal.filter_custom_run(run, tmp_path / "out", {"anchor"})
    assert rep["placed_lines_in"] == 4 and rep["kept_lines"] == 2
    assert rep["dropped_by_reason"]["first"]["merged"] == 2
    assert rep["lines_out"] == {"train": 1, "val": 1}
    val = (tmp_path / "out" / "lists" / "val.lst").read_text().split()
    assert fal.text_lines(Path(val[0]).read_bytes()) == [("l0", "بسم الله")]


def test_atomic_write_text(tmp_path: Path) -> None:
    """The file appears complete and no temp file is left behind."""
    f = tmp_path / "x.lst"
    fal.atomic_write_text(f, "a\nb\n")
    assert f.read_text() == "a\nb\n" and sorted(p.name for p in tmp_path.iterdir()) == ["x.lst"]
