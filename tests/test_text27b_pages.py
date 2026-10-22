# File name: test_text27b_pages.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for text27b ``pages``: structuring choice, ``work`` / ``part`` derivation (nested parts and a
structuring directory directly under the work), the OCR join by file stem, shingle containment,
printed-page inference and the ported page-text helpers. A synthetic literature tree is written
to ``tmp_path``; no real work names are used.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.finetuning.text27b import pages as G

PROSE = ["the merchant wrote to his partner about the cargo of flax and pepper",
         "he asks for news of the ships that sailed from the western ports last year",
         "the judge confirmed the sale of the house in the presence of witnesses",
         "a note on the paper and the hand of the scribe follows in the commentary"]


def write_page(path: Path, text: str, page_number: Optional[int] = None, model: str = "m", pdf: Optional[str] = None,
               printed: Any = None, extra: Optional[Dict[str, Any]] = None) -> None:
    """Write one structured page JSON.

    :param path: Destination file.
    :param text: ``full_main_text``.
    :param page_number: ``metadata.page_number``.
    :param model: ``metadata.model_used``.
    :param pdf: ``metadata.pdf_name``.
    :param printed: ``extracted_page_number``.
    :param extra: Further top-level fields.
    :returns: None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    md: Dict[str, Any] = {"model_used": model, "provider": "p"}
    if page_number is not None:
        md["page_number"] = page_number
    if pdf:
        md["pdf_name"] = pdf
    data = {"full_main_text": text, "metadata": md, "classification": "general_academic",
            "extracted_page_number": printed, "shelf_marks_mentioned": {}, "footnotes": {}, "transcriptions": {}}
    data.update(extra or {})
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def write_ocr(path: Path, texts: Dict[int, str], mode: Optional[str] = None, sources: Optional[Dict[int, str]] = None) -> None:
    """Write an ``*_ocr_results.json`` file.

    :param path: Destination file.
    :param texts: ``{page_number: text}``.
    :param mode: ``processing_info.mode``.
    :param sources: Per-page ``ocr_source``.
    :returns: None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pages: List[Dict[str, Any]] = []
    for n, t in texts.items():
        p: Dict[str, Any] = {"page_number": n, "ocr_result": {"full_text": t, "page_text": t}}
        if sources and n in sources:
            p["ocr_source"] = sources[n]
        pages.append(p)
    info = {"mode": mode} if mode else {}
    path.write_text(json.dumps({"pages": pages, "processing_info": info}, ensure_ascii=False), encoding="utf-8")


@pytest.fixture()
def lib(tmp_path: Path) -> Path:
    """A synthetic literature tree with a nested part and a work-level structuring directory.

    :param tmp_path: Pytest temporary directory.
    :returns: Literature root.
    """
    root = tmp_path / "lit"
    # work_a: nested part vol_1/sub_x/stem_x, two structurings, hybrid OCR
    base = root / "work_a" / "vol_1" / "sub_x" / "stem_x"
    write_ocr(base / "stem_x_ocr_results.json", {1: PROSE[0], 2: PROSE[1], 3: PROSE[2]}, mode="hybrid",
              sources={1: "text_layer", 2: "vision", 3: "text_layer"})
    q = base / "stem_x_structured_qwen" / "qwen3_vl_8b"
    write_page(q / "page_001_structured.json", "worse " + PROSE[0], 1, model="qwen", pdf="stem_x", printed=[10])
    write_page(q / "page_002_structured.json", PROSE[1], 2, model="qwen", pdf="stem_x", printed=[119])
    write_page(q / "page_003_structured.json", PROSE[2] + " plus words the model added on its own here", 3,
               model="qwen", pdf="stem_x", printed=[12])
    g = base / "stem_x_structured_gemini_gemini_3.5_flash"
    write_page(g / "page_001_structured.json", PROSE[0], 1, model="gemini", pdf="stem_x", printed=[10])
    write_page(g / "page_002_structured.json", "", 2, model="gemini", pdf="stem_x")
    # work_b: structuring directory directly under the work, OCR in a sub directory; misnamed file
    write_ocr(root / "work_b" / "stem_b" / "stem_b_ocr_results.json", {1: PROSE[3], 30: PROSE[0]}, mode="text_only")
    s = root / "work_b" / "stem_b_structured"
    write_page(s / "page_001_structured.json", PROSE[3], 1, extra={"shelf_marks_mentioned": {"T-S 99J1.2": "n", "T-S A43.1": "x"}})
    write_page(s / "page_030_structured.json", "older", None, model="plain")
    gb = root / "work_b" / "stem_b" / "stem_b_structured_gemini_gemini_2.5_flash"
    write_page(gb / "page_0301_structured.json", PROSE[0], 30, model="gemini")
    # a non-page file and a non-structured page are ignored
    (root / "work_b" / "stem_b" / "notes.json").write_text("{}")
    write_page(root / "work_c" / "loose" / "page_001_structured.json", "no structuring dir")
    return root


def test_iter_pages_choice_and_parts(lib: Path) -> None:
    """One structuring per page number; nested parts and work-level structuring dirs get the right part.

    :param lib: Synthetic literature root.
    """
    pages = G.iter_pages(lib)
    keys = [(p.work, p.part, p.page_no) for p in pages]
    assert keys == [("work_a", "vol_1/sub_x/stem_x", 1), ("work_a", "vol_1/sub_x/stem_x", 2),
                    ("work_a", "vol_1/sub_x/stem_x", 3), ("work_b", "stem_b", 1), ("work_b", "stem_b", 30)]
    by = {(p.work, p.page_no): p for p in pages}
    assert by[("work_a", 1)].structurer_model == "gemini"
    assert by[("work_a", 2)].structurer_model == "qwen"
    assert by[("work_b", 30)].page_file == "page_0301_structured.json"
    assert by[("work_a", 1)].page_key == "work_a/vol_1/sub_x/stem_x/page_001_structured.json"
    assert by[("work_a", 1)].book == "work_a" and by[("work_a", 1)].idx == 1 and by[("work_a", 1)].text == PROSE[0]


def test_iter_pages_ocr_join_and_containment(lib: Path) -> None:
    """The OCR page is found by stem (nested part included) and containment measures model residue.

    :param lib: Synthetic literature root.
    """
    by = {(p.work, p.page_no): p for p in G.iter_pages(lib)}
    a1, a2, a3 = by[("work_a", 1)], by[("work_a", 2)], by[("work_a", 3)]
    assert a1.ocr_text == PROSE[0] and a1.ocr_file.endswith("stem_x_ocr_results.json")
    assert (a1.ocr_source, a2.ocr_source, a1.ocr_mode) == ("text_layer", "vision", "hybrid")
    assert a1.ocr_containment == 1.0
    assert 0 < a3.ocr_containment < 0.6
    b1, b30 = by[("work_b", 1)], by[("work_b", 30)]
    assert b1.ocr_text == PROSE[3] and b1.ocr_source == "text_layer" and b30.ocr_text == PROSE[0]
    assert all(p.ocr_text is None for p in G.iter_pages(lib, with_ocr=False))


def test_iter_pages_tie_break(tmp_path: Path) -> None:
    """At equal priority the file whose name carries the page number wins, deterministically.

    :param tmp_path: Pytest temporary directory.
    """
    s = tmp_path / "lit" / "w" / "s_structured"
    write_page(s / "page_0301_structured.json", PROSE[0], 30)
    write_page(s / "page_030_structured.json", PROSE[1], None)
    (page,) = G.iter_pages(tmp_path / "lit", with_ocr=False)
    assert page.page_file == "page_030_structured.json" and page.part == "s"


def test_iter_pages_filters(lib: Path) -> None:
    """``works`` restricts and ``skip`` excludes top-level directories.

    :param lib: Synthetic literature root.
    """
    assert {p.work for p in G.iter_pages(lib, works=["work_b"])} == {"work_b"}
    assert {p.work for p in G.iter_pages(lib, skip=["work_a"])} == {"work_b"}


def test_located_shelf_marks(lib: Path) -> None:
    """Only keys present in the page text survive (the prompt example is dropped).

    :param lib: Synthetic literature root.
    """
    b1 = {(p.work, p.page_no): p for p in G.iter_pages(lib)}[("work_b", 1)]
    assert b1.shelf_marks == ["T-S 99J1.2", "T-S A43.1"]
    assert b1.located_shelf_marks() == []
    b1.ocr_text = "see T-S 99J1.2 here"
    assert b1.located_shelf_marks() == ["T-S 99J1.2"]


def test_resolve_ocr_prefers_nearest_and_stays_in_work(tmp_path: Path) -> None:
    """With duplicate stems the copy nearest the page wins; other works' files are never used.

    :param tmp_path: Pytest temporary directory.
    """
    root = tmp_path / "lit"
    near = root / "w" / "a" / "s" / "s_ocr_results.json"
    far = root / "w" / "s_ocr_results.json"
    other = root / "v" / "t_ocr_results.json"
    for f in (near, far, other):
        write_ocr(f, {1: "x"})
    page = root / "w" / "a" / "s" / "s_structured" / "page_001_structured.json"
    idx = G.ocr_index(root)
    assert G.resolve_ocr(page, {"metadata": {"pdf_name": "s"}}, "s_structured", idx, root / "w") == near
    assert G.resolve_ocr(page, {}, "s_structured", idx, root / "w") == near
    assert G.resolve_ocr(page, {"metadata": {"pdf_name": "t"}}, "t_structured", idx, root / "w") is None


def test_containment_and_shingles() -> None:
    """Containment is the share of the text's shingles found in the reference; short text gives None."""
    assert G.containment("a b c d e f", "x a b c d e f y") == 1.0
    assert G.containment("a b c d e f", "a b c d e") == 0.5
    assert G.containment("too short", "anything") is None
    assert G.shingles("שלום עליך אדוני ורבי היקר", 5) == {"שלום עליך אדוני ורבי היקר"}


def test_infer_printed_pages() -> None:
    """Neighbour-consistent numbers are kept; the leaked ``119`` and stray numbers are replaced."""
    def rec(n: int, printed: Optional[str]) -> G.PageRec:
        """Make a minimal page record.

        :param n: Page index.
        :param printed: Extracted printed page.
        :returns: Record.
        """
        return G.PageRec(work="w", part="p", page_file=f"page_{n:03d}_structured.json", page_no=n, path="",
                         structuring_dir="s", structurer_model=None, structurer_provider=None, printed_page=printed,
                         full_main_text="", page_type=None, shelf_marks=[])

    pages = [rec(1, None), rec(2, "12"), rec(3, "119"), rec(4, "14"), rec(5, "15"), rec(6, "77"), rec(7, "17"),
             rec(8, "18"), rec(20, "300")]
    G.infer_printed_pages(pages)
    got = [(p.page_no, p.printed_page_num, p.printed_page_source) for p in pages]
    assert got == [(1, 11, "inferred"), (2, 12, "extracted"), (3, 13, "inferred"), (4, 14, "extracted"),
                   (5, 15, "extracted"), (6, 16, "inferred"), (7, 17, "extracted"), (8, 18, "extracted"),
                   (20, 300, "extracted_unverified")]
    lone = [rec(1, "119")]
    G.infer_printed_pages(lone)
    assert (lone[0].printed_page_num, lone[0].printed_page_source) == (None, None)


def test_page_key_helpers() -> None:
    """Page keys round-trip and page numbers parse from file names."""
    k = G.page_key("w", "a/b/c", "page_007_structured.json")
    assert G.parse_page_key(k) == ("w", "a/b/c", "page_007_structured.json")
    assert G.page_no_of("page_0301_structured.json") == 301
    with pytest.raises(ValueError):
        G.parse_page_key("w/x")
    with pytest.raises(ValueError):
        G.page_no_of("cover.json")
    assert G.model_priority("x_structured_gemini_gemini_3.5_flash") < G.model_priority("x_structured_qwen/qwen3_vl_8b") \
        < G.model_priority("x_structured_gemma3_27b") < G.model_priority("x_structured")


@pytest.mark.parametrize("value,expected", [([118], "118"), ([], None), (None, None), (12, "12"), ("[xii]", "xii"),
                                            (["", "5"], "5"), (True, None)])
def test_parse_printed_page(value: Any, expected: Optional[str]) -> None:
    """``extracted_page_number`` variants normalise to a label or None.

    :param value: Raw value.
    :param expected: Expected label.
    """
    assert G.parse_printed_page(value) == expected


# ----------------------------------------------------------------------------------- page JSON / text helpers
def test_page_json_field_helpers() -> None:
    """Shelfmark keys, notes, transcription keys / chars and footnotes from dict or list forms."""
    d = {"shelf_marks_mentioned": {"T-S 99J1.2": "note", "X 1": {"a": 1}}, "transcriptions": {"T-S 99J1.2": ["ab", "c"], "e": ""},
         "footnotes": {"1": "first", "2": ""}}
    assert G.sm_keys(d) == ["T-S 99J1.2", "X 1"] and G.sm_keys({"shelf_marks_mentioned": ["A 1", 3]}) == ["A 1"]
    assert G.sm_note(d, "T-S 99J1.2") == "note" and G.sm_note(d, "X 1") == '{"a": 1}' and G.sm_note(d, "none") == ""
    assert G.tr_keys(d) == ["T-S 99J1.2"] and G.tr_chars(d, "T-S 99J1.2") == 3
    assert G.footnote_texts(d) == ["first"] and G.footnote_texts({"footnotes": ["a", None]}) == ["a"]


def test_text_analysis_helpers() -> None:
    """Script / language classifiers, blocks, index and catalogue detectors behave as in the prototype."""
    assert G.letter_counts("אב سل ab") == (2, 2, 2)
    assert G.is_rtl_line("שלום עליך a") and not G.is_rtl_line("hello שלום")
    assert G.ja_or_he("פי הדא אלכתאב ען דלך") == "ja" and G.ja_or_he("זה הוא של אשר") == "he"
    assert G.lang_of("the letter of the merchant is in the archive") == "en" and G.lang_of("123") == "other"
    text = "CHAPTER ONE\n" + "\n".join(["a long line of ordinary prose text that runs across the page width."] * 3) + \
           "\n\nsecond block of prose that also runs fairly long across the page."
    blocks = G.segment_blocks(text)
    assert blocks[0].text == "CHAPTER ONE" and len(blocks) == 3 and blocks[1].start == text.index("a long")
    assert G.line_offsets("ab\ncd") == [0, 3]
    assert G.index_like("\n".join(["Aleppo 12, 14, 19, 22"] * 12)) and not G.index_like("short")
    assert G.catalogue_like(["Paper.", "Size: 20 x 14", "Script: square", "Contents: a letter"])
    assert G.rtl_run_chars(["שלום עליך אדוני", "ורבי היקר", "english line"]) == len("שלום עליך אדוני") + len("ורבי היקר")
    letter = " ".join(["I am writing to you, my lord, may God keep you; I received your letter and the dinars."] * 4)
    assert G.docness(letter) >= 1.0 and G.docness("short") == 0.0
    page = "Body text of the page that is long enough to matter here.\n" * 5 + "1 Goitein discusses this letter in detail elsewhere."
    d = {"footnotes": {"1": "Goitein discusses this letter in detail elsewhere."}}
    assert G.footnote_region_start(page, d) == page.index("1 Goitein")
    assert G.footnote_region_start(page, {}) is None
