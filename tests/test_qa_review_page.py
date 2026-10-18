# File name: test_qa_review_page.py
# Date: 9/24/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Unit tests for the QA review page builder (``qa_review_page``).

Covers the Kraken row grouping, the box choice (Kraken row first, then VLM line, then none),
answer parsing, the row builder (one box per target line for list answers, dropped pages,
out-of-range lines), the memoised raw-cache reader, ``</script`` escaping in the rendered page,
the shipped template and the CLI end to end on tiny synthetic inputs.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from src.finetuning.qwen_hebrew import qa_review_page as R

LINE_A = "נתן הכהן בר שלמה"
LINE_B = "חלפון הלוי ביר מנשה"   # letters-only similarity to LINE_A is 0.31: below both bars
DATA_BLOCK = re.compile(r'<script id="data" type="application/json">(.*?)</script>', re.S)
TEMPLATE = ('<html><body><script id="data" type="application/json">__DATA__</script>'
            '<script>boot()</script></body></html>')


def frag(text: str, box: List[float]) -> Dict[str, Any]:
    """A Kraken fragment as stored in the raw cache.

    :param text: Fragment text.
    :param box: ``[x1, y1, x2, y2]`` in 0-1000 units.
    :returns: Fragment record.
    """
    return {"text": text, "box": box}


def evidence(kraken: Sequence[Tuple[str, List[float]]] = (),
             vlm: Sequence[Tuple[str, List[float]]] = ()) -> R.Evidence:
    """Page evidence from ``(text, box)`` readings.

    :param kraken: Kraken rows.
    :param vlm: VLM lines.
    :returns: ``(vlm_lines, kraken_rows)``.
    """
    return [{"text": t, "box": b} for t, b in vlm], list(kraken)


def page_record(lines: List[str], cid: str = "C", idx: int = 0) -> Dict[str, Any]:
    """A minimal edition-manifest page.

    :param lines: Page transcription lines.
    :param cid: Canonical id.
    :param idx: Image index.
    :returns: Record.
    """
    return {"canonical_id": cid, "image_index": idx, "lines": list(lines), "image_width": 1000,
            "image_height": 1500, "side": "recto", "edition_source": "Ed. Goitein"}


def qa_record(answer: Any, stem: str = "q", cid: str = "C", idx: int = 0,
              family: str = "qa_person") -> Dict[str, Any]:
    """A minimal QA-manifest record; the answer is stored as JSON text, as ``build_pgp_qa`` writes it.

    :param answer: Answer object (or a raw answer string).
    :param stem: Row id.
    :param cid: Canonical id.
    :param idx: Image index.
    :param family: QA family.
    :returns: Record.
    """
    text = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
    return {"stem": stem, "family": family, "section": "witness", "pgpid": "7", "canonical_id": cid,
            "image_index": idx, "image_url": f"https://example.org/{cid}/{idx}.jpg", "split": "train",
            "question": "Who is the witness?", "answer": text, "line": 1, "evidence": "{'relation': 'Witness'}"}


def no_reads(canonical_id: str, image_index: int) -> Optional[R.Evidence]:
    """Evidence lookup for a page set without any raw-cache file.

    :param canonical_id: Page document id.
    :param image_index: Page image index.
    :returns: None.
    """
    return None


# ----------------------------------------------------------------------------- Kraken rows

def test_rows_from_frags_groups_rows_and_reads_right_to_left() -> None:
    """Fragments sharing a vertical band form one row, joined right to left; the box is their union."""
    left = frag("ב", [100, 100, 300, 130])
    right = frag("א", [500, 105, 700, 128])
    below = frag("ג", [100, 200, 600, 230])
    assert R.rows_from_frags([below, left, right]) == [("אב", [100, 100, 700, 130]),
                                                       ("ג", [100, 200, 600, 230])]


def test_rows_from_frags_band_grows_with_each_fragment() -> None:
    """A later fragment joins through the grown band even when outside the first fragment's band."""
    a = frag("א", [0, 100, 100, 130])      # centre 115 opens [100, 130]
    b = frag("ב", [200, 105, 300, 145])    # centre 125 joins: band [100, 145]
    c = frag("ג", [400, 128, 500, 158])    # centre 143 is outside [100, 130], inside [100, 145]
    d = frag("ד", [0, 300, 100, 320])
    assert R.rows_from_frags([d, c, b, a]) == [("גבא", [0, 100, 500, 158]), ("ד", [0, 300, 100, 320])]
    assert R.rows_from_frags([]) == []


# ----------------------------------------------------------------------------- box choice

def test_pick_box_prefers_kraken_over_a_better_vlm_line() -> None:
    """A Kraken row above the bar wins; the box is rounded to ints and the similarity to 2 places."""
    ev = evidence(kraken=[("אבגד", [0, 0, 10, 10]), (LINE_A[:-1], [100.4, 200.6, 300.2, 250.7])],
                  vlm=[(LINE_A, [1, 2, 3, 4])])
    assert R.pick_box(LINE_A, ev) == {"src": "kraken", "box": [100, 201, 300, 251], "sim": 0.92}


def test_pick_box_kraken_bar_is_inclusive() -> None:
    """Kraken similarity of exactly 0.5 still gives the Kraken box."""
    ev = evidence(kraken=[("אבהו", [1, 2, 3, 4])], vlm=[("אבגד", [5, 6, 7, 8])])
    assert R.pick_box("אבגד", ev) == {"src": "kraken", "box": [1, 2, 3, 4], "sim": 0.5}


def test_pick_box_falls_back_to_the_best_vlm_line() -> None:
    """Below the Kraken bar, the most similar VLM line at >= 0.6 gives the box."""
    ev = evidence(kraken=[("אבגד", [0, 0, 10, 10])],
                  vlm=[("שלום", [9, 9, 9, 9]), (LINE_A, [10, 20, 30, 40])])
    assert R.pick_box(LINE_A, ev) == {"src": "vlm", "box": [10, 20, 30, 40], "sim": 1.0}


def test_pick_box_none_when_no_reader_is_close_enough() -> None:
    """No box when neither reader clears its bar, when the page has no reads or no readings."""
    assert R.pick_box(LINE_A, evidence(kraken=[("אבגד", [0, 0, 1, 1])], vlm=[("שלום", [0, 0, 1, 1])])) is None
    assert R.pick_box("אבגד", evidence(vlm=[("אבהו", [1, 2, 3, 4])])) is None   # 0.5 < the VLM bar
    assert R.pick_box(LINE_A, evidence()) is None
    assert R.pick_box(LINE_A, None) is None


# ----------------------------------------------------------------------------- answers and rows

@pytest.mark.parametrize("raw, parsed, lines", [
    ('{"line": 3, "text": "א"}', {"line": 3, "text": "א"}, [3]),
    ('[{"line": 2, "text": "א"}, {"text": "ב"}, "x"]', [{"line": 2, "text": "א"}, {"text": "ב"}, "x"], [2, None]),
    ('{"answer": "not stated"}', {"answer": "not stated"}, []),
    ("not json", {"answer": "not json"}, []),
])
def test_parse_answer_and_target_lines(raw: str, parsed: Any, lines: List[Optional[int]]) -> None:
    """Answers decode from JSON (else wrap as ``{"answer": ...}``); each quoted item names one line.

    :param raw: Manifest answer string.
    :param parsed: Expected parsed answer.
    :param lines: Expected target lines.
    """
    assert R.parse_answer(raw) == parsed
    assert R.target_lines(parsed) == lines


def test_build_rows_list_answer_gets_one_box_per_target_line() -> None:
    """A list answer gets one box per quoted line, each from the reader that matches that line."""
    pages = R.index_pages([page_record(["אבגד", LINE_A, LINE_B])])
    reads = {("C", 0): evidence(kraken=[(LINE_A, [400, 700, 900, 760])], vlm=[(LINE_B, [450, 760, 820, 800])])}
    answer = [{"line": 2, "text": LINE_A}, {"line": 3, "text": LINE_B}]
    rows, sources = R.build_rows([qa_record(answer, family="qa_witnesses_list")], pages,
                                 lambda cid, idx: reads.get((cid, idx)))
    assert rows[0]["boxes"] == [{"line": 2, "src": "kraken", "box": [400, 700, 900, 760], "sim": 1.0},
                                {"line": 3, "src": "vlm", "box": [450, 760, 820, 800], "sim": 1.0}]
    assert rows[0]["answer"] == answer
    assert sources == {"kraken": 1, "vlm": 1}


def test_build_rows_row_fields() -> None:
    """A row carries the QA fields and the page's lines, size, side and edition source, in order."""
    rows, _ = R.build_rows([qa_record({"line": 1, "text": LINE_A})], R.index_pages([page_record([LINE_A])]),
                           no_reads)
    row = rows[0]
    assert list(row) == ["id", "family", "section", "doc", "pgpid", "img", "w", "h", "split", "question",
                         "answer", "line", "lines", "evidence", "boxes", "side", "source"]
    assert (row["id"], row["doc"], row["img"], row["w"], row["h"]) == ("q", "C", "https://example.org/C/0.jpg",
                                                                       1000, 1500)
    assert (row["lines"], row["side"], row["source"]) == ([LINE_A], "recto", "Ed. Goitein")


def test_build_rows_drops_missing_pages_and_skips_out_of_range_lines() -> None:
    """Rows without an edition page are dropped; out-of-range lines are neither boxed nor counted."""
    pages = R.index_pages([page_record([LINE_A, LINE_B])])
    records = [qa_record({"line": 1, "text": LINE_A}, stem="no-reads"),
               qa_record({"line": 9, "text": "x"}, stem="out-of-range"),
               qa_record({"answer": "not stated"}, stem="abstain"),
               qa_record({"line": 1, "text": LINE_A}, stem="no-page", cid="MISSING")]
    rows, sources = R.build_rows(records, pages, no_reads)
    assert [r["id"] for r in rows] == ["no-reads", "out-of-range", "abstain"]
    assert all(r["boxes"] == [] for r in rows)
    assert sources == {"none": 1}


def test_evidence_loader_reads_each_cache_file_once(tmp_path: Path) -> None:
    """The raw cache gives ``(vlm_lines, kraken_rows)``; a missing page gives None; reads are memoised.

    :param tmp_path: pytest temporary directory.
    """
    vlm_lines = [{"text": LINE_A, "box": [1, 2, 3, 4]}]
    path = tmp_path / "C__0__m.json"
    path.write_text(json.dumps({"vlm_lines": vlm_lines, "frags": [frag("א", [0, 0, 10, 10])]}), encoding="utf-8")
    assert R.load_evidence(tmp_path, "C", 1, "m") is None
    lookup = R.evidence_loader(tmp_path, "m")
    assert lookup("C", 0) == (vlm_lines, [("א", [0, 0, 10, 10])])
    path.unlink()
    assert lookup("C", 0) == (vlm_lines, [("א", [0, 0, 10, 10])])


# ----------------------------------------------------------------------------- page

def test_render_page_escapes_script_close_tags() -> None:
    """``</script`` in the data (any case) is escaped, so the data block parses back to the rows."""
    rows = [{"id": "r1", "lines": ["</script><script>alert(1)</script>", "</SCRIPT>", "a/b"]}]
    page = R.render_page(rows, TEMPLATE)
    block = DATA_BLOCK.search(page).group(1)
    assert "</script" not in block.lower()
    assert "<\\/script>" in block and "<\\/SCRIPT>" in block and "a/b" in block
    assert json.loads(block) == rows
    assert page.lower().count("</script") == TEMPLATE.lower().count("</script")


def test_render_page_needs_exactly_one_placeholder() -> None:
    """A template without the placeholder, or with two, is refused."""
    with pytest.raises(ValueError):
        R.render_page([], "<html></html>")
    with pytest.raises(ValueError):
        R.render_page([], "__DATA__ __DATA__")


def test_shipped_template_holds_the_placeholder_in_the_data_script() -> None:
    """The template next to the module has one placeholder, inside the JSON data element."""
    assert R.TEMPLATE_PATH.parent == Path(R.__file__).resolve().parent
    template = R.load_template()
    assert template.count(R.DATA_PLACEHOLDER) == 1
    assert '<script id="data" type="application/json">__DATA__</script>' in template


def test_main_writes_the_page_and_the_rows(tmp_path: Path) -> None:
    """The CLI joins the manifests with the raw cache and writes the page and the rows JSON.

    :param tmp_path: pytest temporary directory.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "C__0__m.json").write_text(json.dumps({"vlm_lines": [], "frags": [frag(LINE_A, [100, 200, 900, 240])]}),
                                      encoding="utf-8")
    eds, qa = tmp_path / "eds.jsonl", tmp_path / "qa.jsonl"
    eds.write_text(json.dumps(page_record(["אבגד", LINE_A])) + "\n", encoding="utf-8")
    qa.write_text(json.dumps(qa_record({"line": 2, "text": LINE_A})) + "\n\n", encoding="utf-8")
    out, data = tmp_path / "page.html", tmp_path / "rows.json"
    R.main(["--qa-manifest", str(qa), "--editions-manifest", str(eds), "--raw-dir", str(raw),
            "--vlm-model", "m", "--out", str(out), "--data-json", str(data)])
    rows = json.loads(data.read_text(encoding="utf-8"))
    assert rows[0]["boxes"] == [{"line": 2, "src": "kraken", "box": [100, 200, 900, 240], "sim": 1.0}]
    page = out.read_text(encoding="utf-8")
    assert json.loads(DATA_BLOCK.search(page).group(1)) == rows
    assert "<title>Genizah QA Review</title>" in page
    with pytest.raises(SystemExit):
        R.main(["--qa-manifest", str(qa), "--editions-manifest", str(eds), "--raw-dir", str(tmp_path / "none"),
                "--out", str(out)])
