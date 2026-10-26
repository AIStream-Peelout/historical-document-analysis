# File name: test_text27b_ocr_repair.py
# Date: 10/25/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for text27b ``ocr_repair`` (bidi-fragmented text layers) and the ``pages`` repair hook.

Glyph geometry is synthetic: lines are built from (character, x) lists drawn the way a right-to-left word
processor exports them (visual positions, mirrored bracket glyphs); no real work, PDF or page text is used.
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pytest

from src.finetuning.text27b import ocr_repair as R
from src.finetuning.text27b import pages as G

W = 5.0  # glyph advance used by the helpers


def run(chars: str, x_right: float, y: float = 100.0, size: float = 12.0, span: int = 0, seq0: int = 0,
        rtl: bool = True) -> List[R.Glyph]:
    """Glyphs of a run of characters drawn from ``x_right`` leftwards (RTL) or rightwards (LTR).

    A space character advances the pen without making a glyph (word gaps come from geometry).

    :param chars: Characters in drawing order.
    :param x_right: Start x (right edge for RTL, left edge for LTR).
    :param y: Baseline.
    :param size: Font size.
    :param span: Span id.
    :param seq0: First stream index.
    :param rtl: Draw right to left.
    :returns: Glyphs.
    """
    out: List[R.Glyph] = []
    x = x_right
    for i, c in enumerate(chars):
        if rtl:
            x0, x1 = x - W, x
            x -= W
        else:
            x0, x1 = x, x + W
            x += W
        if c == " ":
            continue
        out.append(R.Glyph(c=c, x0=x0, x1=x1, y0=y - size * 0.8, y1=y + size * 0.2, ox=x0, oy=y, size=size,
                           seq=seq0 + i, span=span))
    return out


def line_of(glyphs: Sequence[R.Glyph]) -> R.Line:
    """A line from glyphs.

    :param glyphs: Glyphs.
    :returns: Line.
    """
    return R.Line(glyphs=list(glyphs), baseline=glyphs[0].oy, size=max(g.size for g in glyphs))


# ------------------------------------------------------------------------------------------- characters
def test_bidi_class_and_mirror() -> None:
    assert R.bidi_class("ש") == "R" and R.bidi_class("ع") == "R" and R.bidi_class("A") == "L"
    assert R.bidi_class("7") == "N" and R.bidi_class(".") == "S" and R.bidi_class("(") == "O"
    assert R.bidi_class("ִ") == "M" and R.bidi_class(" ") == "W"
    assert R.mirror("(") == ")" and R.mirror("]") == "[" and R.mirror("א") == "א"


# ------------------------------------------------------------------------------------------- visual → logical
def test_rtl_line_numbers_and_brackets() -> None:
    # logical "(1) [שלום]" drawn RTL: glyph ')' at the right is the logical '('
    g = run(")1( ]שלום[", 200.0)
    assert R.line_text(line_of(g), rtl=True) == "(1) [שלום]"


def test_rtl_multi_digit_number_is_re_reversed() -> None:
    # "עמ 68–72" : the numbers are drawn left to right inside the RTL line, the dash between them
    # visual right to left: עמ | 68 | – | 72  (68 is to the right of 72; each number drawn left to right)
    glyphs = run("עמ", 200.0) + run("68", 175.0, rtl=False, seq0=10) + run("–", 175.0, seq0=20) + \
        run("72", 160.0, rtl=False, seq0=30)
    assert R.line_text(line_of(glyphs), rtl=True) == "עמ 68–72"


def test_rtl_latin_runs_split_at_semicolon() -> None:
    # logical "כי TS 1; ENA 2" : the two shelfmarks are separate LTR runs, ';' belongs to the RTL text
    glyphs = run("כי", 300.0) + run("TS 1", 260.0, rtl=False, seq0=10) + run(";", 260.0, seq0=20) + \
        run("ENA 2", 225.0, rtl=False, seq0=30)
    assert R.line_text(line_of(glyphs), rtl=True) == "כי TS 1; ENA 2"


def test_apostrophe_inside_latin_word_keeps_run() -> None:
    glyphs = run("ראה", 300.0) + run("Smith’s book", 200.0, rtl=False, seq0=10)
    assert R.line_text(line_of(glyphs), rtl=True) == "ראה Smith’s book"


def test_ltr_line_with_hebrew_run() -> None:
    glyphs = run("see", 50.0, rtl=False) + run("שלום", 90.0, seq0=10) + run("here", 100.0, rtl=False, seq0=20)
    assert R.line_text(line_of(glyphs), rtl=False) == "see שלום here"


def test_spaces_from_geometry_not_from_space_glyphs() -> None:
    g = run("אב", 200.0)
    # a spurious space glyph between two touching letters, and a word gap with no space glyph
    spur = R.Glyph(c=" ", x0=193.0, x1=197.0, y0=90, y1=102, ox=193.0, oy=100.0, size=12.0, seq=5, span=0)
    g2 = run("גד", 185.0, seq0=6)
    assert R.line_text(line_of(g + [spur] + g2), rtl=True) == "אב גד"
    touching = run("גד", 190.0, seq0=6)
    assert R.line_text(line_of(g + [spur] + touching), rtl=True) == "אבגד"


def test_marks_attach_to_their_base() -> None:
    g = run("כשרים", 200.0)
    hiriq = R.Glyph(c="ִ", x0=g[1].x0 + 2, x1=g[1].x0 + 3, y0=90, y1=102, ox=g[1].x0 + 2, oy=100.0, size=12.0,
                    seq=1, span=0)
    # stream order: ש, then its point, then the rest of the word (same span)
    for x in g[2:]:
        x.seq += 10
    hiriq.seq = 2
    text = R.line_text(line_of(g + [hiriq]), rtl=True)
    assert text == "כשִרים"


def test_lone_mark_placed_by_geometry() -> None:
    g = run("אנה", 200.0)
    shadda = R.Glyph(c="ّ", x0=g[1].x0, x1=g[1].x0 + 2.5, y0=90, y1=102, ox=g[1].x0, oy=100.0, size=12.0,
                     seq=50, span=9)
    assert R.line_text(line_of(g + [shadda]), rtl=True) == "אנّה"


def test_superscript_footnote_mark_never_merges_with_number() -> None:
    sm = run("TS 10", 400.0, rtl=False)
    sup = R.Glyph(c="1", x0=393.0, x1=397.0, y0=88, y1=96, ox=393.0, oy=95.0, size=8.0, seq=20, span=3)
    lines = R.cluster_lines(sm + [sup])
    assert len(lines) == 1
    assert R.line_text(lines[0], rtl=True) == "TS 10 1"
    he = run("הפטור", 300.0)
    sup2 = R.Glyph(c="3", x0=271.0, x1=274.5, y0=88, y1=96, ox=271.0, oy=95.0, size=8.0, seq=20, span=3)
    tail = run("שלה", 268.0, seq0=30)
    lines = R.cluster_lines(he + [sup2] + tail)
    assert R.line_text(lines[0], rtl=True) == "הפטור3 שלה"


# ------------------------------------------------------------------------------------------- lines and page
def test_cluster_lines_by_baseline() -> None:
    a = run("שורה", 200.0, y=100.0)
    b = run("אחרת", 200.0, y=114.0, seq0=10)
    lines = R.cluster_lines(b + a)
    assert [round(ln.baseline) for ln in lines] == [100, 114]


def test_numbered_edition_line_continuation_joined() -> None:
    num = run("6", 300.0)
    text1 = run("אבג דהו", 280.0, seq0=5)
    cont = run("זחט", 280.0, y=112.0, seq0=20)
    num7 = run("7", 300.0, y=126.0, seq0=30)
    text7 = run("יכל", 280.0, y=126.0, seq0=35)
    txt, info = R.assemble_page(R.cluster_lines(num + text1 + cont + num7 + text7))
    assert txt.split("\n") == ["6 אבג דהו זחט", "7 יכל"]
    assert info["continuation_joins"] == 1


def test_broken_word_joined_by_medial_form() -> None:
    tl = [R.TextLine(text="הוא כתב", x0=0, x1=100, top=0, bottom=10, baseline=10, size=12, rtl=True),
          R.TextLine(text="ב מכתב", x0=0, x1=100, top=12, bottom=22, baseline=24, size=12, rtl=True)]
    out, n = R.join_broken_words(tl, {})
    assert n == 0 and len(out) == 2
    tl[0].text = "שלח את המ"  # ends in a medial mem: the word cannot end here
    tl[1].text = "כתב היום"
    out, n = R.join_broken_words(tl, {})
    assert n == 1 and out[0].text == "שלח את המכתב היום"


def test_paragraph_gap_becomes_blank_line() -> None:
    a = run("ראשונה שורה", 200.0, y=100.0)
    b = run("שנייה שורה", 200.0, y=114.0, seq0=20)
    c = run("חדשה פסקה", 200.0, y=150.0, seq0=40)
    d = run("המשך פסקה", 200.0, y=164.0, seq0=60)
    txt, _ = R.assemble_page(R.cluster_lines(a + b + c + d))
    assert txt.split("\n")[2] == ""


def test_two_columns_read_right_first() -> None:
    glyphs: List[R.Glyph] = []
    for i in range(8):
        y = 100.0 + 14 * i
        glyphs += run("ימין" * 4, 400.0, y=y, seq0=100 * i)
        glyphs += run("שמאל" * 4, 250.0, y=y, seq0=100 * i + 50)
    lines = R.cluster_lines(glyphs)
    gutter = R.detect_gutter(lines)
    assert gutter is not None and 250 < gutter < 320
    txt, info = R.assemble_page(lines)
    rows = [r for r in txt.split("\n") if r]
    assert rows[:8] == ["ימין" * 4] * 8 and rows[8:] == ["שמאל" * 4] * 8
    assert info["gutter"] is not None


def test_single_column_has_no_gutter() -> None:
    glyphs: List[R.Glyph] = []
    for i in range(8):
        glyphs += run("מילים רבות בשורה אחת ארוכה", 400.0, y=100.0 + 14 * i, seq0=100 * i)
    assert R.detect_gutter(R.cluster_lines(glyphs)) is None


# ------------------------------------------------------------------------------------------- fallback, checks
def test_heuristic_rejoin() -> None:
    text = "6\n \n]אבג\n דהו[ \nזחט\n  \n7\nיכל"
    out = R.heuristic_rejoin(text)
    assert out.split("\n") == ["6 [אבג דהו] זחט", "7 יכל"]


def test_chars_preserved_and_fragment_stats() -> None:
    assert R.chars_preserved(")1( אב\nג", "(1) אבג")
    assert not R.chars_preserved("אב", "אבג")
    st = R.fragment_stats("one two three\nfour\n\nfive six seven eight")
    assert st == {"lines": 3, "ge3": 2, "share_ge3": round(2 / 3, 4)}


def test_supplement_glyphs_from_trace() -> None:
    glyphs = run("אב", 200.0)
    trace = [{"size": 12.0, "chars": [(ord("̇"), 1, (195.0, 100.0), (195.0, 90.0, 195.0, 102.0)),
                                      (ord("̇"), 1, (195.0, 100.0), (195.0, 90.0, 195.0, 102.0)),
                                      (ord("א"), 2, (195.0, 100.0), (195.0, 90.0, 200.0, 102.0))]}]
    n = R.supplement_glyphs(glyphs, trace, "א̇ב")
    assert n == 1 and sum(1 for g in glyphs if g.c == "̇") == 1


# ------------------------------------------------------------------------------------------- work + hook
def _write_lit(root: Path, work: str, texts: Dict[int, str]) -> Path:
    """A one-part literature work with structured pages and an OCR file (no PDF).

    :param root: Literature root.
    :param work: Work directory name.
    :param texts: ``{page_number: OCR text}``.
    :returns: The OCR file.
    """
    part = root / work / "stem"
    sdir = part / "stem_structured_m"
    sdir.mkdir(parents=True, exist_ok=True)
    for n, t in texts.items():
        (sdir / f"page_{n:03d}_structured.json").write_text(json.dumps(
            {"full_main_text": t, "metadata": {"page_number": n, "pdf_name": "stem"}, "classification": "x",
             "shelf_marks_mentioned": {}}, ensure_ascii=False), encoding="utf-8")
    ocr = part / "stem_ocr_results.json"
    ocr.write_text(json.dumps({"pages": [{"page_number": n, "ocr_result": {"full_text": t}} for n, t in texts.items()],
                               "processing_info": {"mode": "text_only"}}, ensure_ascii=False), encoding="utf-8")
    return ocr


def test_repair_work_writes_and_pages_hook_applies(tmp_path: Path) -> None:
    root = tmp_path / "lit"
    frag = "\n".join(["6", " ", "]אבג", " דהו[ ", "זחט", "  ", "7", "יכל מנס עפצ"] * 3)
    ocr = _write_lit(root, "wk", {1: frag, 2: "שורה אחת שלמה של טקסט רגיל"})
    repairs, info = R.repair_work(ocr, root / "wk")
    assert info["pdf"] is None and info["methods"] == {"json_heuristic": 2}
    assert all(r.preserved for r in repairs)
    rep = R.report(repairs, info, samples=2)
    assert rep["share_ge3_after"] > rep["share_ge3_before"]
    out_root = tmp_path / "rep"
    d = R.write_repairs("wk", repairs, ocr, root, out_root, rep)
    man = json.loads((d / R.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert man["ocr_file"] == "wk/stem/stem_ocr_results.json" and set(man["pages"]) == {"1", "2"}
    name, recs = R.load_repairs(out_root, "wk")
    assert name == "stem_ocr_results.json" and set(recs) == {1, 2}
    pg = G.iter_pages(root, repair_root=out_root)
    p1 = [p for p in pg if p.page_no == 1][0]
    assert p1.ocr_repair == "json_heuristic" and p1.ocr_text_raw == frag and p1.ocr_text.startswith("6 [אבג דהו] זחט")
    assert p1.ocr_source.endswith("+repair")
    assert p1.ocr_containment == 1.0 or p1.ocr_containment is None  # join check uses the raw OCR text
    raw = G.iter_pages(root, repair_root=out_root, use_repairs=False)
    assert [p.ocr_text for p in raw if p.page_no == 1] == [frag]


def test_hook_skips_stale_repair(tmp_path: Path) -> None:
    page = G.PageRec(work="wk", part="stem", page_file="page_001_structured.json", page_no=1, path="x",
                     structuring_dir="s", structurer_model=None, structurer_provider=None, printed_page=None,
                     full_main_text="", page_type=None, shelf_marks=[], ocr_file="/a/stem_ocr_results.json",
                     ocr_text="new text")
    old_sha = hashlib.sha1("old text".encode("utf-8")).hexdigest()
    assert not G.apply_ocr_repair(page, {1: ("repaired", old_sha, "pdf_geometry")}, "stem_ocr_results.json")
    assert page.ocr_text == "new text"
    sha = hashlib.sha1("new text".encode("utf-8")).hexdigest()
    assert not G.apply_ocr_repair(page, {1: ("repaired", sha, "pdf_geometry")}, "other_ocr_results.json")
    assert G.apply_ocr_repair(page, {1: ("repaired", sha, "pdf_geometry")}, "stem_ocr_results.json")
    assert page.ocr_text == "repaired" and page.ocr_text_raw == "new text" and page.ocr_repair == "pdf_geometry"


def test_write_repairs_skips_unpreserved(tmp_path: Path) -> None:
    root = tmp_path / "lit"
    ocr = _write_lit(root, "wk", {1: "אב"})
    bad = R.PageRepair(page_number=1, original="אב", text="אבג", method="pdf_geometry", preserved=False)
    d = R.write_repairs("wk", [bad], ocr, root, tmp_path / "rep", {"x": 1})
    assert json.loads((d / R.MANIFEST_NAME).read_text(encoding="utf-8"))["pages"] == {}
    assert not (d / "page_001.txt").exists()
