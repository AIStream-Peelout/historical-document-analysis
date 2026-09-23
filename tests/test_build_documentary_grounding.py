# File name: test_build_documentary_grounding.py
# Date: 9/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Unit tests for build_documentary_grounding (two-reader line-grounding rows).

Covers the line rule, the fragment-union box and its sanity checks, the
per-page caps, the decontamination filter, the KTIV prompt/answer format, the
document-level split and the image orientation / verification path.
"""
import hashlib
import io
import json
from pathlib import Path
from typing import Dict, List, Optional

import pytest
from PIL import Image

from src.finetuning.qwen_hebrew import build_documentary_grounding as bdg
from src.finetuning.qwen_hebrew import build_ktiv_dataset as bkd
from src.finetuning.qwen_hebrew.push_ktiv_dataset import _answer_ok

W, H = 2000, 3000
FRAG = [[100, 100, 600, 120]]


def _line(index: int, text: str, frags: List[List[int]], agreement: Optional[float] = 0.95,
          status: str = "agreed") -> Dict:
    """One ``ai_read.lines`` entry.

    :param index: Line index.
    :param text: VLM text.
    :param frags: Assigned Kraken fragment boxes (0-1000).
    :param agreement: Two-reader similarity.
    :param status: Line status.
    :return: Line dict.
    """
    return {"index": index, "text": text, "bbox": [0, 0, 1000, 1000], "agreement": agreement,
            "status": status, "htr_text": text, "htr_fragments": frags}


def _record(lines: List[Dict], doc_id: str = "Cambridge_CUL_T_S_99_1") -> Dict:
    """One batch record around ``lines``.

    :param lines: Line entries.
    :param doc_id: Canonical id.
    :return: Record dict.
    """
    return {"doc_id": doc_id, "image_index": 0, "image_width": W, "image_height": H,
            "image_url": f"https://storage.googleapis.com/cairo-genizah-es-json/images/{doc_id}.jpg",
            "image_sha256": "", "ai_read": {"vlm_model": "m", "lines": lines}}


def _usable(index: int, text: str, y: int = 100) -> bdg.UsableLine:
    """A usable line with a plausible box.

    :param index: Line index.
    :param text: Line text.
    :param y: Top of the box.
    :return: UsableLine.
    """
    return bdg.UsableLine(index, text, (100, y, 600, y + 20), bdg.hebrew_letters(text), 0.9)


def _jpeg(w: int, h: int, orientation: Optional[int] = None) -> bytes:
    """Encode a flat JPEG, optionally with an EXIF orientation tag.

    :param w: Width.
    :param h: Height.
    :param orientation: EXIF orientation value or None.
    :return: JPEG bytes.
    """
    buf = io.BytesIO()
    im = Image.new("RGB", (w, h), (200, 180, 150))
    if orientation:
        exif = Image.Exif()
        exif[0x0112] = orientation
        im.save(buf, "JPEG", exif=exif)
    else:
        im.save(buf, "JPEG")
    return buf.getvalue()


# --- prompts / schema ---------------------------------------------------------

def test_prompts_and_features_are_the_ktiv_objects():
    assert bdg._LOCATE_PROMPT is bkd._LOCATE_PROMPT
    assert bdg._READBOX_PROMPT is bkd._READBOX_PROMPT
    assert bdg.FEATURES is bkd.FEATURES
    phrase = "שלום רב לאוהבי תורתך"
    assert bdg.locate_question(phrase) == bkd._LOCATE_PROMPT.format(phrase=phrase)
    assert bdg.read_box_question([1, 2, 3, 4]) == bkd._READBOX_PROMPT.format(x0=1, y0=2, x1=3, y1=4)


# --- box union and sanity ---------------------------------------------------

def test_union_box_is_fragment_union_clamped_to_ints():
    assert bdg.union_box([[100, 200, 300, 240], [310, 205, 500, 238]]) == (100, 200, 500, 240)
    assert bdg.union_box([[-3.2, 10.4, 1004.0, 20.6]]) == (0, 10, 1000, 21)


def test_box_sanity_checks():
    med = 20.0
    assert bdg.box_rejection((100, 200, 500, 240), med, W) is None
    assert bdg.box_rejection((100, 200, 100, 240), med, W) == "box_not_positive"
    assert bdg.box_rejection((100, 240, 500, 240), med, W) == "box_not_positive"
    assert bdg.box_rejection((100, 200, 500, 260), med, W) is None             # exactly 3x median
    assert bdg.box_rejection((100, 200, 500, 261), med, W) == "box_too_tall"
    assert bdg.box_rejection((100, 200, 130, 240), med, W) is None             # 30/1000*2000 = 60 px
    assert bdg.box_rejection((100, 200, 129, 240), med, W) == "box_too_narrow"  # 58 px
    assert bdg.box_rejection((100, 200, 500, 240), 0.0, W) == "no_page_median"


def test_median_height_ignores_degenerate_boxes():
    assert bdg.median_height([[0, 0, 1, 10], [0, 0, 1, 20], [0, 5, 1, 5]]) == 15.0
    assert bdg.median_height([]) == 0.0


def test_page_fragment_boxes_prefer_raw_cache(tmp_path):
    rec = _record([_line(0, "ואמר לו רבי שמעון", FRAG)])
    boxes, src = bdg.page_fragment_boxes(rec, tmp_path)
    assert (boxes, src) == (FRAG, "record")
    raw = {"frags": [{"text": "x", "conf": 0.9, "box": [1.5, 2.0, 30.0, 12.5]}]}
    (tmp_path / "Cambridge_CUL_T_S_99_1__0__m.json").write_text(json.dumps(raw))
    boxes, src = bdg.page_fragment_boxes(rec, tmp_path)
    assert (boxes, src) == ([[1.5, 2.0, 30.0, 12.5]], "raw_cache")


# --- line rule -----------------------------------------------------------------

def test_line_rule_keeps_only_clean_agreed_lines():
    lines = [
        _line(0, "ואמר  לו רבי שמעון בן גמליאל", FRAG),                          # usable
        _line(1, "כתבנו ונתנו לו לראיה", FRAG, agreement=0.5, status="unconfirmed"),
        _line(2, "שלום רב לאוהבי תורתך", FRAG, agreement=0.79),
        _line(3, "אבג דה", FRAG),                                                # 5 letters
        _line(4, "ברוך אתה [...] אלהינו", FRAG),
        _line(5, "הוצרכנו להקור וurdאוי", FRAG),                                   # decoding glitch
        _line(6, "והיה אם שמוע תשמעו", []),
        _line(7, "אשר קדשנו במצותיו וצונו", [[100, 300, 600, 390]]),               # 90 > 3 x 20
        _line(8, "מודים אנחנו לך שאתה", [[100, 400, 110, 420]]),                  # 20 px wide
        _line(9, "לשנה הבאה בירושלים הבנויה", [[100, 500, 600, 520], [620, 505, 900, 525]]),
    ]
    usable, reasons = bdg.usable_lines(lines, 20.0, W)
    assert [u.index for u in usable] == [0, 9]
    assert usable[0].text == "ואמר לו רבי שמעון בן גמליאל"                     # whitespace collapsed
    assert usable[1].box == (100, 500, 900, 525)                               # union, not the evidence bbox
    assert usable[1].letters == bdg.hebrew_letters("לשנה הבאה בירושלים הבנויה")
    expected = {"agreed_seen": 9, "usable": 2, "agreement_below_min": 1, "too_few_letters": 1,
                "gap_marker": 1, "foreign_script": 1, "no_fragments": 1, "box_too_tall": 1,
                "box_too_narrow": 1}
    assert dict(reasons) == expected


def test_repeated_phrase_is_not_usable():
    lines = [_line(0, "קדשים נאכלין לפנים מן הקלעים", FRAG),
             _line(1, "קדשים נאכלין לפנים מן הקלעים קדשים קלים", [[100, 200, 600, 220]])]
    usable, reasons = bdg.usable_lines(lines, 20.0, W)
    assert [u.index for u in usable] == [1]
    assert reasons["not_unique_on_page"] == 1


def test_select_lines_takes_most_letters_with_caps():
    texts = ["אבגדהוזח", "אבגדהוזחט", "אבגדהוזחטי", "אבגדהוזחטיכ", "אבגדהוזחטיכל", "אבגדהוזחטיכלמ"]
    usable = [_usable(i, t, 100 + 30 * i) for i, t in enumerate(texts)]
    loc, rb = bdg.select_lines(usable)
    assert [u.index for u in loc] == [5, 4, 3, 2]
    assert [u.index for u in rb] == [5, 4]
    assert bdg.select_lines(usable[:1]) == ([], [])
    two = bdg.select_lines(usable[:2])
    assert [u.index for u in two[0]] == [1, 0] and [u.index for u in two[1]] == [1, 0]


def test_plan_page_skips_pages_with_fewer_than_two_usable_lines():
    one = _record([_line(0, "ואמר לו רבי שמעון", FRAG), _line(1, "אבג", [[100, 200, 600, 220]])])
    plan, reasons = bdg.plan_page(one, FRAG * 5)
    assert plan is None and reasons["usable"] == 1
    two = _record([_line(0, "ואמר לו רבי שמעון", FRAG),
                   _line(1, "כתבנו ונתנו לו לראיה", [[100, 200, 600, 220]])])
    plan, _ = bdg.plan_page(two, FRAG * 5)
    assert plan is not None and (plan.width, plan.height) == (W, H)
    assert len(plan.locate) == 2 and len(plan.read_box) == 2


# --- answer format ---------------------------------------------------------------

def test_rows_match_the_ktiv_locate_and_read_box_families(tmp_path):
    u0, u1 = _usable(3, "ואמר לו רבי שמעון בן גמליאל", 100), _usable(7, "כתבנו ונתנו לו לראיה", 300)
    url = "https://storage.googleapis.com/cairo-genizah-es-json/images/1_image.jpg"
    plan = bdg.PagePlan("Cambridge_CUL_T_S_99_1", 2, url, "", W, H, [u0, u1], [u0])
    pairs = bdg.page_rows(plan, tmp_path / "p.jpg")
    rows = [r for r, _ in pairs]
    assert [r["task"] for r in rows] == ["locate", "locate", "read_box"]
    assert [r["stem"] for r in rows] == ["dg_Cambridge_CUL_T_S_99_1__2_loc0",
                                         "dg_Cambridge_CUL_T_S_99_1__2_loc1",
                                         "dg_Cambridge_CUL_T_S_99_1__2_rb0"]
    for r in rows:
        assert set(r) == set(bkd.FEATURES)
        assert r["section"] == "line" and r["label_source"] == "two_reader_agreed"
        assert r["target_chars"] == len(r["answer"]) and r["target_tokens"] == 0
        assert (r["image_width"], r["image_height"]) == (W, H) and r["image"] == str(tmp_path / "p.jpg")
        assert _answer_ok(r["task"], r["answer"])                  # the hub-push validator
    loc = json.loads(rows[0]["answer"])
    assert list(loc) == ["bbox_2d"]
    assert loc["bbox_2d"] == list(u0.box)
    assert all(isinstance(v, int) and 0 <= v <= 1000 for v in loc["bbox_2d"])
    assert f'"{u0.text}"' in rows[0]["question"]
    rb = rows[2]
    assert "bbox_2d = [100, 100, 600, 120]" in rb["question"]
    assert rb["answer"] == u0.text
    meta = [m for _, m in pairs]
    assert meta[2] == {"doc_id": "Cambridge_CUL_T_S_99_1", "image_index": 2,
                       "image_url": url, "task": "read_box",
                       "stem": "dg_Cambridge_CUL_T_S_99_1__2_rb0", "box": [100, 100, 600, 120],
                       "text": u0.text, "agreement": 0.9, "line_index": 3, "url_prefix": "images"}


# --- decontamination ----------------------------------------------------------------

def test_decontam_filter_layers():
    rel_sys = "990052087050205171"
    idx = bdg.build_benchmark_index(
        ids=["Cambridge_CUL_T_S_10J12_4", "New_York_JTS_ENA_2557_1"],
        image_urls=["https://storage.googleapis.com/cairo-genizah-es-json/images/3661_image.jpg"],
        religious_docs=[{"sys_num": rel_sys, "shelf_mark":
                         "Cambridge University Library, Cambridge, England Ms. T-S AS 124.138"}])

    def why(doc_id: str, urls=(), refs=None, **kw) -> Optional[str]:
        return bdg.decontam_reason(doc_id, urls, refs or bdg.DocRefs(), idx, **kw)

    assert why("Cambridge_CUL_T_S_10J12_4") == "benchmark_id"
    assert why("Cambridge_CUL_T_S_10J12_4_2") == "benchmark_join_suffix"     # doc = benchmark + _n
    assert why("New_York_JTS_ENA_2557") == "benchmark_join_suffix"          # benchmark = doc + _n
    assert why("Cambridge_CUL_T_S_10_J_12_4") == "benchmark_loose_key"      # separator variant
    assert why("Cambridge_CUL_T_S_16_1",
               ["https://storage.googleapis.com/cairo-genizah-es-json/images/3661_image.jpg"]) \
        == "benchmark_image_url"
    assert why("Cambridge_CUL_T_S_NS_1_1", ["https://storage.googleapis.com/cairo-genizah-es-json/"
                                            f"KTIV/{rel_sys}/0001_FL1.jpg"]) == "religious_sys_num"
    assert why("Cambridge_CUL_T_S_NS_1_2", refs=bdg.DocRefs(sys_nums={rel_sys})) == "religious_sys_num"
    assert why("Cambridge_CUL_T_S_AS_124_138") == "religious_shelfmark"
    assert why("Cambridge_CUL_X_1", refs=bdg.DocRefs(shelfmarks={"T-S AS 124.138"})) \
        == "religious_shelfmark"
    # a different fragment of the same volume is kept (unless volume-level is asked for)
    assert why("Cambridge_CUL_T_S_10J12_5") is None
    assert why("Cambridge_CUL_T_S_10J12_45") is None                        # 10J12.45 is not 10J12.4
    assert why("Cambridge_CUL_T_S_10J12_5", volume_level=True) == "benchmark_volume"


def test_real_benchmark_index_catches_benchmark_docs():
    idx = bdg.load_benchmark_index()
    ids = json.loads(bdg.BENCH_IDS_PATH.read_text())
    assert set(ids) <= idx.ids and len(idx.image_urls) >= 100
    assert bdg.decontam_reason(ids[0], (), bdg.DocRefs(), idx) == "benchmark_id"
    rel = json.loads(bdg.RELIGIOUS_BENCH_PATH.read_text())["docs"][0]
    assert bdg.decontam_reason("Nowhere_1", (), bdg.DocRefs(sys_nums={rel["sys_num"]}), idx) \
        == "religious_sys_num"


def test_doc_refs_from_merged_record():
    rec = {"canonical_id": "A_1", "shelfmark_display": "T-S AS 1.2",
           "images": {"ktiv": {"sys_num": "990000000000000001"}},
           "sources": {"ktiv": [{"sys_num": "990000000000000002"}], "pgp": {"fragment": {"shelfmark": "T-S AS 1.2"}}}}
    refs = bdg.doc_refs_from_record(rec)
    assert refs.sys_nums == {"990000000000000001", "990000000000000002"}
    assert refs.shelfmarks == {"T-S AS 1.2"}
    assert bdg.doc_refs_from_record({"canonical_id": "B"}) == bdg.DocRefs()


# --- split / batch -------------------------------------------------------------------

def test_val_split_is_by_document_and_stable_as_the_batch_grows():
    docs = [f"Doc_{i}" for i in range(4000)]
    val = bdg.val_documents(docs)
    assert 0.035 < len(val) / len(docs) < 0.065
    assert val == bdg.val_documents(reversed(docs))
    assert val <= bdg.val_documents(docs + [f"New_{i}" for i in range(1000)])
    assert len(bdg.val_documents(["a", "b"])) == 1


def test_load_batch_dedupes_and_skips_partial_tail(tmp_path):
    a1 = {"doc_id": "A", "image_index": 0, "v": 1}
    a2 = {"doc_id": "A", "image_index": 0, "v": 2}
    b = {"doc_id": "B", "image_index": 1, "v": 1}
    p = tmp_path / "batch.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in (a1, b, a2)) + "\n" + '{"doc_id": "C", "ima')
    recs, counts = bdg.load_batch(p)
    assert recs == [a2, b]
    assert counts == {"lines": 3, "duplicates": 1, "partial_tail_skipped": 1}


# --- images ---------------------------------------------------------------------------

def test_orient_and_save_mirrors_prepare_image(tmp_path):
    upright = _jpeg(40, 20)
    assert bdg.orient_and_save(upright, tmp_path / "a.jpg") == (40, 20)
    assert (tmp_path / "a.jpg").read_bytes() == upright              # kept byte-for-byte
    rotated = _jpeg(40, 20, orientation=6)
    assert bdg.orient_and_save(rotated, tmp_path / "b.jpg") == (20, 40)
    with Image.open(tmp_path / "b.jpg") as im:
        assert im.size == (20, 40) and (im.getexif() or {}).get(0x0112, 1) == 1


def test_fetch_page_image_verifies_hash_and_dims(tmp_path, monkeypatch):
    data = _jpeg(40, 20)
    sha = hashlib.sha256(data).hexdigest()
    plan = bdg.PagePlan("D_1", 0, "https://x/images/1.jpg", sha, 40, 20, [], [])
    monkeypatch.setattr(bdg, "download_bytes", lambda url: data)
    path, why = bdg.fetch_page_image(plan, tmp_path)
    assert why is None and path == tmp_path / "D_1__0.jpg" and path.read_bytes() == data

    def no_download(url: str) -> bytes:
        raise AssertionError("a verified file must be reused")
    monkeypatch.setattr(bdg, "download_bytes", no_download)
    assert bdg.fetch_page_image(plan, tmp_path) == (path, None)

    monkeypatch.setattr(bdg, "download_bytes", lambda url: data)
    assert bdg.fetch_page_image(bdg.PagePlan("D_2", 0, "u", "0" * 64, 40, 20, [], []), tmp_path) \
        == (None, "sha256_mismatch")
    assert bdg.fetch_page_image(bdg.PagePlan("D_3", 0, "u", sha, 20, 40, [], []), tmp_path) \
        == (None, "dims_mismatch")
    monkeypatch.setattr(bdg, "download_bytes", lambda url: None)
    assert bdg.fetch_page_image(bdg.PagePlan("D_4", 0, "u", sha, 40, 20, [], []), tmp_path) \
        == (None, "download")
    assert not list(tmp_path.glob("*.part"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
