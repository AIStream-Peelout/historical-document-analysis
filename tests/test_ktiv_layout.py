"""Unit tests for KTIV layout reconstruction and the dataset builder's gates."""

import pytest

from src.finetuning.qwen_hebrew.ktiv_layout import (
    GAP_TOKEN,
    clean_word,
    extract_words,
    reconstruct_page,
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


def test_shingle_decontamination():
    """A page sharing long letter runs with a benchmark GT is flagged."""
    gt = "אבגדהוזחטיכלמנסעפצקרשת" * 3
    n = 25
    shingles = {gt[i:i + n] for i in range(len(gt) - n + 1)}
    assert shingle_hits(gt[:40], shingles) >= 2
    assert shingle_hits("שלום עליכם מורי ורבותי", shingles) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
