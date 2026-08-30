# File name: test_ktiv_grounding_rows.py
# Date: 8/30/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.
"""Unit tests for the v20 grounding/QA row emitters in build_ktiv_dataset.

Design: docs/v20_grounding_qa_design.md. All coordinates must come out as
0-1000 ints normalized to the ORIGINAL frame; geometry-free (served) pages
must emit nothing; phrases must be letter-unique on their page.
"""
import json
import random
from pathlib import Path

import pytest

from src.finetuning.qwen_hebrew import build_ktiv_dataset as bkd
from src.finetuning.qwen_hebrew.ktiv_layout import reconstruct_page


def _item(text, x0, y0, x1, y1, item_id="1 oldVer"):
    """One AnnotationPage item with a rectangular SvgSelector."""
    path = f"M{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1} {x0},{y0}z"
    return {"id": item_id, "type": "Annotation",
            "body": {"type": "TextualBody", "value": text},
            "target": {"selector": {"type": "SvgSelector",
                                    "value": f"<svg><path d=\"{path}\"/></svg>"}}}


def _page_items():
    """Three-line page, distinct words, generous spacing (no merges)."""
    words = [["ראשונה", "שורה", "זוהי"], ["שנייה", "מילים", "כאן"],
             ["שלישית", "פסקה", "עוד"], ["אחרונה", "כתובת", "סוף"]]
    items = []
    for li, ws in enumerate(words):
        x1 = 1900.0
        for w in reversed(ws):
            items.append(_item(w, x1 - 300, li * 100, x1, li * 100 + 60))
            x1 -= 340
    return items


def test_norm_box_rounds_to_0_1000():
    assert bkd.norm_box((0, 0, 500, 250), 1000, 500) == [0, 0, 500, 500]
    assert bkd.norm_box((999, 499, 1000, 500), 1000, 500) == [999, 998, 1000, 1000]


def test_locate_candidates_unique_and_boxed():
    items = _page_items()
    page = reconstruct_page(items)
    cands = bkd.locate_candidates(items, page["text"])
    assert cands, "expected unique phrases on a page of distinct words"
    for phrase, box in cands:
        letters = "".join(bkd._HEB_RE.findall(phrase))
        assert len(letters) >= bkd.LOCATE_MIN_PHRASE_LETTERS
        page_letters = "".join(bkd._HEB_RE.findall(page["text"]))
        assert page_letters.count(letters) == 1
        x0, y0, x1, y1 = box
        assert x0 < x1 and y0 < y1


def test_grounding_rows_geometry_page(monkeypatch):
    items = _page_items()
    page = reconstruct_page(items)
    assert page["mode"] == "geometry"
    monkeypatch.setattr(bkd, "GROUNDED_PAGE_FRACTION", 1.0)
    monkeypatch.setattr(bkd, "READBOX_MIN_LETTERS", 5)
    rows = bkd.grounding_rows(page, items, Path("x.jpg"), "ktiv_9_FL1",
                              2000, 400, 1000, 200, random.Random(7))
    tasks = [r["task"] for r in rows]
    assert tasks.count("locate") >= 1 and tasks.count("locate") <= bkd.LOCATE_ROWS_PER_PAGE
    assert "read_box" in tasks and "layout_qa" in tasks and "grounded_page" in tasks
    for r in rows:
        assert r["image_width"] == 1000 and r["image_height"] == 200
    loc = next(r for r in rows if r["task"] == "locate")
    box = json.loads(loc["answer"])["bbox_2d"]
    assert len(box) == 4 and all(0 <= v <= 1000 for v in box)
    assert loc["question"].count('"') >= 2 and "0-1000" in loc["question"]
    rb = next(r for r in rows if r["task"] == "read_box")
    assert "bbox_2d = [" in rb["question"] and "0-1000" in rb["question"]
    assert any(rb["answer"] == ln["text"] for ln in page["lines"])
    gp = next(r for r in rows if r["task"] == "grounded_page")
    payload = json.loads(gp["answer"])
    assert len(payload) == len(page["lines"])
    assert all(len(e["bbox_2d"]) == 4 and all(0 <= v <= 1000 for v in e["bbox_2d"])
               and e["text"] for e in payload)
    # normalization must use the ORIGINAL frame (2000x400), not the saved size
    x_norms = [e["bbox_2d"][2] for e in payload]
    assert max(x_norms) == round(1000 * 1900 / 2000)


def test_layout_qa_answers_derive_from_page():
    items = _page_items()
    page = reconstruct_page(items)
    rng = random.Random(3)
    seen = set()
    for _ in range(30):
        row = bkd._layout_qa_row(page, bkd.locate_candidates(items, page["text"]),
                                 Path("x.jpg"), "s", 10, 10, rng)
        if row is None:
            continue
        seen.add(row["section"])
        if row["section"] == "columns":
            assert row["answer"] == str(len(page["columns"]))
        elif row["section"] == "edge_line":
            assert row["answer"] in (page["lines"][0]["text"], page["lines"][-1]["text"])
        else:
            assert any(row["answer"] == ln["text"] for ln in page["lines"])
    assert {"columns", "edge_line"} <= seen


def test_served_page_emits_no_grounding_rows():
    def flat(text, x0, x1):
        return _item(text, x0, 0, x1, 0)
    br = {"id": "t1-BreakLine", "body": {"value": ""}}
    items = [flat("שלום", -900, -700), flat("עולם", -650, -450), dict(br),
             flat("טובה", -900, -700), flat("שנה", -650, -450), dict(br),
             flat("מזל", -900, -700), flat("טוב", -650, -450)]
    page = reconstruct_page(items)
    assert page["mode"] == "served_breaklines"
    rows = bkd.grounding_rows(page, items, Path("x.jpg"), "s",
                              2000, 400, 1000, 200, random.Random(0))
    assert rows == []


def test_exclude_benchmark_manuscripts(tmp_path):
    religious_sys = json.load(open(bkd.RELIGIOUS_BENCH_PATH))["docs"][0]["sys_num"]
    pgp_shelfmark = "T-S 8.4"   # frozen PGP-131 member
    meta = {"sys_num": "990000000000000001", "shelf_mark": pgp_shelfmark}
    (tmp_path / "ktiv_fake_meta.json").write_text(json.dumps(meta))
    bundles = [{"sys_num": religious_sys},
               {"sys_num": "990000000000000001"},
               {"sys_num": "990000000000000002"}]
    kept, excl = bkd.exclude_benchmark_manuscripts(bundles, tmp_path)
    assert [b["sys_num"] for b in kept] == ["990000000000000002"]
    assert excl == {"religious_benchmark_ms": 1, "pgp131_shelfmark": 1}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
