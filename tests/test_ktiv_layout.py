"""Unit tests for KTIV layout reconstruction and the dataset builder's gates."""

import pytest

from src.finetuning.qwen_hebrew.ktiv_layout import (
    GAP_TOKEN,
    clean_word,
    extract_words,
    reconstruct_page,
    reorder_ocr_lines,
)
from src.finetuning.qwen_hebrew.build_ktiv_dataset import (
    page_gate,
    region_candidates,
    shingle_hits,
)


def _item(text: str, x0: float, y0: float, x1: float, y1: float, item_id: str = "1 oldVer",
          sigla: str = None) -> dict:
    """Build one AnnotationPage item with a rectangular SvgSelector."""
    path = f"M{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1} {x0},{y0}z"
    return {"id": item_id, "type": "Annotation",
            "body": {"type": "TextualBody", "value": text, "sigla": sigla},
            "target": {"id": "https://iiif.nli.org.il/IIIFv3/FL1",
                       "selector": {"type": "SvgSelector", "value": f"<svg><path d=\"{path}\"/></svg>"}}}


def _rtl_line(words, y0, y1, right=1000.0, width=120.0, space=40.0):
    """Lay words right-to-left on one line, returning items."""
    items = []
    x1 = right
    for w in words:
        items.append(_item(w, x1 - width, y0, x1, y1))
        x1 = x1 - width - space
    return items


def test_clean_word_policy():
    """Dots -> gap token; editorial symbols stripped; ink kept."""
    assert clean_word(".....") == (GAP_TOKEN, True)
    assert clean_word("⟡") == ("", False)
    assert clean_word("שלום~") == ("שלום", False)
    assert clean_word("אמ'") == ("אמ'", False)
    assert clean_word("קטרת²") == ("קטרת", False)


def test_extract_words_dedupes_and_drops_breaks():
    """Duplicate (text, box) items collapse; BreackLine markers are dropped."""
    items = [_item("אבג", 0, 0, 100, 50), _item("אבג", 0, 0, 100, 50, item_id="2 oldVer"),
             {"id": "/supplementing/t0-BreackLine", "body": {"value": ""}}]
    words = extract_words(items)
    assert len(words) == 1 and words[0]["text"] == "אבג"


def test_reading_order_and_word_merge():
    """Lines top-to-bottom, words right-to-left, touching fragments merged."""
    items = _rtl_line(["ראשון", "שני", "שלישי"], 0, 50)
    items += _rtl_line(["רביעי", "חמישי"], 80, 130)
    # A split word on line 3: two touching boxes 'שלו' + 'ם' (gap 2px on a 50px line).
    items += [_item("שלו", 800, 160, 1000, 210), _item("ם", 758, 160, 798, 210),
              _item("עליכם", 500, 160, 700, 210)]
    page = reconstruct_page(items)
    assert [l["text"] for l in page["lines"]] == ["ראשון שני שלישי", "רביעי חמישי", "שלום עליכם"]
    assert len(page["columns"]) == 1


def test_two_columns_split_and_ordered_right_first():
    """A wide empty gutter yields two columns; right column is read first."""
    items = []
    for i in range(4):
        items += _rtl_line([f"ימין{i}", "מלה", "עוד"], i * 80, i * 80 + 50, right=2000)
        items += _rtl_line([f"שמאל{i}", "מלה", "עוד"], i * 80, i * 80 + 50, right=800)
    page = reconstruct_page(items)
    assert len(page["columns"]) == 2
    assert page["lines"][0]["text"].startswith("ימין0")
    assert page["columns"][1]["lines"][0]["text"].startswith("שמאל0")
    cands = dict((c[0], c[2]) for c in region_candidates(page))
    assert cands["right_column"].startswith("ימין0") and "שמאל" not in cands["right_column"]
    assert cands["left_column"].startswith("שמאל0")


def test_gaps_collapse_and_gate():
    """Consecutive gap boxes collapse; the gate rejects gap-dominated pages."""
    items = [_item(".....", 800, 0, 1000, 50), _item("....", 700, 0, 790, 50),
             _item("אבגד", 500, 0, 690, 50)]
    page = reconstruct_page(items)
    assert page["lines"][0]["text"] == f"{GAP_TOKEN} אבגד"
    assert page_gate(page) == "too_few_letters"


def _ocr_line(text: str, x0: float, y0: float, x1: float, y1: float) -> dict:
    """Build one OCR line dict as the Kraken microservice returns it."""
    return {"text": text, "bbox": [x0, y0, x1, y1], "confidence": 0.9}


def test_reorder_ocr_lines_sorts_single_column():
    """Served order is scrambled; output follows y (top to bottom)."""
    lines = [_ocr_line("שורה שלישית כאן", 100, 200, 900, 250),
             _ocr_line("שורה ראשונה כאן", 100, 0, 900, 50),
             _ocr_line("שורה שניה כאן", 100, 100, 900, 150)]
    assert reorder_ocr_lines(lines).splitlines() == [
        "שורה ראשונה כאן", "שורה שניה כאן", "שורה שלישית כאן"]


def test_reorder_ocr_lines_columns_right_first_despite_gutter_stray():
    """A short stray word inside the gutter neither bridges the columns nor
    fabricates a third one; right column is read before the left."""
    lines = []
    for i in range(8):
        lines.append(_ocr_line(f"שמאל שורה מספר {i}", 100, i * 80, 1000, i * 80 + 50))
        lines.append(_ocr_line(f"ימין שורה מספר {i}", 1600, i * 80, 2500, i * 80 + 50))
    lines.append(_ocr_line("אלו", 1250, 300, 1350, 350))   # gutter stray (3 letters)
    out = reorder_ocr_lines(lines).splitlines()
    assert out[0].startswith("ימין שורה מספר 0")
    right = [ln for ln in out if ln.startswith("ימין")]
    left = [ln for ln in out if ln.startswith("שמאל")]
    assert len(right) == 8 and len(left) == 8
    assert out.index(right[-1]) < out.index(left[0])   # whole right column first


def test_reorder_ocr_lines_joins_same_row_fragments_rtl():
    """Two fragments of one visual line join right-to-left; touching
    fragments are merged into one word."""
    lines = [_ocr_line("חצי", 600, 0, 800, 50),          # right fragment
             _ocr_line("שמאלי", 100, 5, 400, 55),        # left fragment, same row
             _ocr_line("שלו", 800, 100, 950, 150),       # split word, touching
             _ocr_line("ם", 770, 100, 798, 150)]
    out = reorder_ocr_lines(lines).splitlines()
    assert out[0] == "חצי שמאלי"
    assert out[1] == "שלום"


def test_reorder_ocr_lines_empty_and_blank():
    """No lines or only blank text yields an empty string."""
    assert reorder_ocr_lines([]) == ""
    assert reorder_ocr_lines([_ocr_line("  ", 0, 0, 10, 10)]) == ""


def test_shingle_decontamination():
    """A page sharing long letter runs with a benchmark GT is flagged."""
    gt = "אבגדהוזחטיכלמנסעפצקרשת" * 3
    n = 25
    shingles = {gt[i:i + n] for i in range(len(gt) - n + 1)}
    assert shingle_hits(gt[:40], shingles) >= 2
    assert shingle_hits("שלום עליכם מורי ורבותי", shingles) == 0


# --- GT-mash fixes (2026-08 audit: over-tall boxes, dense pages, y-less pages) ---


def test_interleave_rescue_unchains_tall_box_lines():
    """Boxes 2.5x taller than the line pitch chain neighbouring lines into
    one cluster; the rescue splits them back apart instead of letting the
    x-sort interleave the words into an unspaced mash."""
    # Two physical lines, pitch 60px, but every box drawn 150px tall.
    items = []
    for li, texts in enumerate((["ראשונה", "שורה", "זוהי"], ["שניה", "שורה", "זוהי"])):
        y0 = li * 60
        x1 = 1000.0
        for t in reversed(texts):
            items.append(_item(t, x1 - 150, y0, x1, y0 + 150))
            x1 -= 190
    page = reconstruct_page(items)
    assert [l["text"] for l in page["lines"]] == ["זוהי שורה ראשונה", "זוהי שורה שניה"]
    from src.finetuning.qwen_hebrew.ktiv_layout import hebrew_letters
    assert max(len(t) for t in page["text"].split()) <= 6   # no chain-merged runs


def test_dense_page_adaptive_merge_cut():
    """Real spaces at ~0.08 line heights (dense hand) survive: the per-page
    cut adapts below them instead of mashing every line into one run."""
    items = []
    for li in range(12):   # enough gaps for the estimator to see the modes
        y0 = li * 70
        x1 = 4000.0
        for _ in range(4):
            items.append(_item("אבג", x1 - 100, y0, x1, y0 + 50))       # word
            items.append(_item("דה", x1 - 170, y0, x1 - 100, y0 + 50))  # touching fragment
            x1 -= 174             # next word 4px on -> real space 4px = 0.08 lh
    page = reconstruct_page(items)
    assert page["merge_cut"] < 0.08
    for line in page["lines"]:
        assert line["text"] == "אבגדה אבגדה אבגדה אבגדה"


def test_deep_overlap_insertion_not_merged():
    """An interlinear insertion whose box overlaps the word deeply stays a
    separate token (v1 glued it into a pseudo-word)."""
    items = [_item("ויעשה", 600, 0, 900, 60), _item("הד", 620, 5, 700, 40),
             _item("אחר", 300, 0, 500, 60)]
    page = reconstruct_page(items)
    assert "ויעשההד" not in page["text"] and "הדויעשה" not in page["text"]


def test_merged_word_letter_cap():
    """A chain of touching fragments can never merge past 18 letters."""
    items = []
    x1 = 5000.0
    for _ in range(10):
        items.append(_item("אבג", x1 - 60, 0, x1, 50))
        x1 -= 61   # 1px gaps: everything touching
    page = reconstruct_page(items)
    words = page["text"].split()
    assert all(len(w) <= 18 for w in words) and sum(len(w) for w in words) == 30


def test_degenerate_geometry_served_breakline_fallback():
    """Pages whose selectors are all y=0 (mirrored x) fall back to served
    order split at BreakLine markers, and touching mirrored fragments join."""
    def flat(text, x0, x1):
        return _item(text, x0, 0, x1, 0)
    br = {"id": "t1-BreakLine", "body": {"value": ""}}
    items = [flat("שלום", -900, -700), flat("עולם", -650, -450), dict(br),
             # split word on line 2: boxes touch in the mirrored frame
             flat("שלו", -900, -760), flat("ם", -758, -720), flat("טוב", -650, -500)]
    page = reconstruct_page(items)
    assert page["mode"] == "served_breaklines"
    assert page["text"].splitlines() == ["שלום עולם", "שלום טוב"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
