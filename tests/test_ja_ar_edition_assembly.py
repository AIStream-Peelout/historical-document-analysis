"""Regression cases for source-led assembly before the training and release gates."""
import hashlib
import json
from pathlib import Path

import fitz
import pytest

from src.finetuning.qwen_hebrew.ja_ar_build import assemble, edition_lines as el
from src.finetuning.qwen_hebrew.ja_ar_build.align import stream_from_read


def baseline(text: str, number: int, page: int = 1) -> dict:
    """Build a synthetic source baseline.

    :param text: Source text.
    :param number: Stable ordinal.
    :param page: Physical PDF page.
    :return: Baseline record.
    """
    return {"baseline_pt": number*15, "raw": text, "raw_sha256": el.digest(text), "text": text, "kind": "source_or_gap", "ordinal": number,
            "line_id": f"{page}.{number}", "pdf_page": page, "edition_folio": None, "flags": [], "placement": None}


def test_missing_lines_and_competing_faces() -> None:
    """A reader can locate source on the page while missing its central line and both edges."""
    first, last = "מכתב ראשון עבור הכותב", "וכאן הסתיים המכתב בשלום"
    lines = [baseline(t, i) for i, t in enumerate(["קצה", first, "קטע", last, "סוף"], 1)]
    stream = stream_from_read("a", "kraken:x", [first, "קריאה אחרת", last], {})
    el.place_lines(lines, [stream])
    assert [l["placement"] and l["placement"]["method"] for l in lines] == [
        None, "cached_line_match", "between_ordered_anchors", "cached_line_match", None]
    assert lines[2]["text"] == "קטע"
    ambiguous = [baseline(first, 1)]
    el.place_lines(ambiguous, [stream, stream_from_read("b", "kraken:x", [first], {})])
    assert ambiguous[0]["placement"] is None


def test_previous_pairs_recover_internal_omission_in_source_order() -> None:
    """An omitted edition line survives between prior anchors and repeated formulas remain unassigned."""
    lines = [baseline(t, i) for i, t in enumerate(["ראשונה", "חדש", "לפני סיום", "נוסח", "נוסח"], 1)]
    pair = {"decision": "review", "image_sha256": "a", "crop": None, "rotation": 180,
            "pair_id": "p", "lines": ["ראשונה", "לפני סיום", "נוסח"]}
    el.previous_locations(lines, [pair])
    assert lines[1]["placement"]["method"] == "between_previous_pair_lines"
    assert [l["text"] for l in lines] == ["ראשונה", "חדש", "לפני סיום", "נוסח", "נוסח"]
    assert lines[-1]["placement"] is None and lines[-2]["placement"] is None
    changed = [baseline("ראשונה", 1), baseline("חדש", 2), baseline("לפני סיום", 1, 2)]
    el.previous_locations(changed, [pair])
    assert changed[1]["placement"] is None


def test_inspection_wins_over_conflicting_prior_and_stale_binding_fails() -> None:
    """A stale binding or supplied foreign word fails and another prior photo cannot replace an inspection."""
    line = baseline("מסמך ישן ארוך", 1)
    binding = {"pdf_sha256": "pdf", "image_sha256": "a", "rotation": 0, "observation": "Exact source",
               "lines": [{"pdf_page": 1, "ordinal": 1, "raw_sha256": line["raw_sha256"], "text": "ישן ארוך"}]}
    el.apply_bindings([line], [binding], "pdf", {"a": {"sha256": "a", "width": 20, "height": 30}})
    el.previous_locations([line], [{"pair_id": "old", "image_sha256": "b", "rotation": 0,
                                   "crop": None, "decision": "accept", "lines": ["ישן ארוך"]}])
    assert line["placement"]["image_sha256"] == "a" and line["other_view"]["image_sha256"] == "b"
    with pytest.raises(ValueError, match="different edition"):
        el.apply_bindings([line], [binding], "stale", {})
    binding["lines"][0]["text"] = "זרה"
    with pytest.raises(ValueError, match="absent from the source"):
        el.apply_bindings([baseline("מסמך ישן ארוך", 1)], [binding], "pdf", {"a": {"sha256": "a"}})


def test_folio_labels_and_ambiguous_layout() -> None:
    """Only a unique explicit manuscript face can supply an independent label-based placement."""
    l1, l2 = baseline("שורה אחת", 1), baseline("שורה שניה", 2)
    l2["edition_folio"] = "1r"
    ed = {"sides": [{"pointer": {"flags": []}}]}
    el.place_labelled_folios([l1, l2], {"a": {"label": "1r"}, "b": {"label": "1r"}}, ed)
    assert l2["placement"] is None
    el.place_labelled_folios([l1, l2], {"a": {"label": "1r"}}, ed)
    assert l2["placement"] is None and l2["catalogue_face_hint"] == "a"
    anchor = baseline("עוד שורה", 3)
    anchor.update(edition_folio="1r", placement={"image_sha256": "a", "rotation": 180,
                                               "crop": None, "method": "source_inspection"})
    el.place_labelled_folios([l1, l2, anchor], {"a": {"label": "1r"}}, ed)
    assert l1["placement"] is None and l2["placement"]["rotation"] == 180


def test_normal_entry_point_keeps_pdfs_gaps_and_holds(tmp_path: Path) -> None:
    """Exercise assembly dispatch, original PDF bytes, two pages, overlap holds and atomic retry behavior.

    :param tmp_path: Isolated run directory.
    """
    path = tmp_path / "edition.pdf"
    with fitz.open() as doc:
        for t in ("..........", "........."):
            doc.new_page().insert_text((60, 70), t)
        doc.save(path)
    image = tmp_path / "photo"
    image.write_bytes(b"source-photo")
    sha = hashlib.sha256(image.read_bytes()).hexdigest()
    ed = {"row_id": "test", "sides": [{"lines": [], "pointer": {"file": path.name, "flags": []}}],
          "pdf_file": path.name, "rnl": True}
    doc = {"document_group": "one", "editions": [ed], "overlap_witnesses": [],
           "photos": [{"sha256": sha, "path": str(image), "label": "1r", "width": 20, "height": 30}]}
    inv = tmp_path / "inventory.jsonl"
    inv.write_text(json.dumps(doc) + "\n" + json.dumps(dict(doc, document_group="held", overlap_witnesses=["seen"])) + "\n")
    prior = tmp_path / "prior.jsonl"
    prior.touch()
    args = ["edition-lines", "--inventory", str(inv), "--pdf-dir", str(tmp_path), "--previous-pairs", str(prior),
            "--out", str(tmp_path / "result")]
    assemble.main(args)
    result = json.loads((tmp_path / "result/editions.jsonl").read_text())
    assert [l["pdf_page"] for l in result["lines"]] == [1, 2]
    assert all(l["raw"] and l["runs"] and "check_gap_or_mark" in l["flags"] for l in result["lines"])
    assert json.loads((tmp_path / "result/run.json").read_text())["overlap_holds"] == 1
    with pytest.raises(FileExistsError):
        assemble.main(args)


def test_edition_edges_recover_short_torn_line_without_crossing_notes_or_crops() -> None:
    """An edge can get a provisional location when two anchors support its contiguous printed block."""
    lines = [baseline(t, i) for i, t in enumerate(["פתיחה", "שורה", "שניה", "שריד", "סימן", "בלתי ידוע"], 1)]
    lines[4]["kind"] = "folio_heading"
    for index in (1, 2):
        lines[index]["placement"] = {"image_sha256": "s", "rotation": 0, "crop": None, "method": "cached_line_match"}
    el.place_block_edges(lines)
    assert lines[0]["placement"]["method"] == lines[3]["placement"]["method"] == "edition_block_edge"
    assert lines[5]["placement"] is None
    for i in (1, 2):
        lines[i]["placement"]["crop"] = [1, 2, 3, 4]
    lines[0]["placement"] = lines[3]["placement"] = None
    el.place_block_edges(lines)
    assert lines[0]["placement"] is None


def test_edge_alternative_preserves_only_edition_letters() -> None:
    """An unresolved edge row can retain a candidate without inserting its restoration into a draft."""
    assert el.edge_reading("[.....מלה אחת&]......") == "[...]מלה אחת[...]"
    assert el.edge_reading("[.....מ[ל]ה אחת]....") is None
    assert el.edge_reading("מלה אחת [...]") is None


def test_previous_fused_answer_locates_whole_printed_rows_only() -> None:
    """A fused answer supplies a location while the edition owns both lines and their wording."""
    lines = [baseline(t, i) for i, t in enumerate(["תחילת שורה", "סוף השורה", "נוסח נוסף"], 1)]
    pair = {"decision": "review", "image_sha256": "a", "crop": None, "rotation": 0,
            "pair_id": "p", "lines": ["תחילת שורה סוף השורה", "שורה בסוף"]}
    el.previous_locations(lines, [pair])
    assert [l["placement"] and l["placement"]["method"] for l in lines] == ["previous_merged_line"]*2 + [None]
    assert [l["text"] for l in lines] == ["תחילת שורה", "סוף השורה", "נוסח נוסף"]
    gap = [baseline("תחילת שורה", 1), baseline("[...]", 2), baseline("סוף השורה", 3)]
    el.previous_locations(gap, [pair])
    assert not any(l["placement"] for l in gap)
