# File name: test_v22_task_eval.py
# Date: 9/27/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Unit tests for the v22 task-level evaluation scorers (no LM Studio needed)."""
import json

import pytest

from src.finetuning.qwen_hebrew.eval_harness import v22_task_eval as te


# --- text helpers -----------------------------------------------------------
def test_normalise_folds_finals_nikud_quotes_and_whitespace() -> None:
    assert te.normalise_hebrew("שָׁלוֹם  עוֹלָם") == "שלומ עולמ"
    assert te.normalise_hebrew("ביר׳ יעקב") == "ביר' יעקב"
    assert te.normalise_hebrew(" בפסטאט\n") == "בפסטאט"


@pytest.mark.parametrize("raw,expected", [
    ('{"line": 4, "text": "בפסטאט"}', {"line": 4, "text": "בפסטאט"}),
    ('```json\n{"line": 4, "text": "בפסטאט"}\n```', {"line": 4, "text": "בפסטאט"}),
    ('The answer is {"line": 4, "text": "בפסטאט"} as written.', {"line": 4, "text": "בפסטאט"}),
    ('[{"line": 1, "text": "a"}, {"line": 2, "text": "b"}]', [{"line": 1, "text": "a"}, {"line": 2, "text": "b"}]),
    ("no json here", None),
    ("", None),
])
def test_extract_json(raw: str, expected) -> None:
    assert te.extract_json(raw) == expected


def test_parse_box_iou_and_hit() -> None:
    assert te.parse_box('{"bbox_2d": [73, 389, 740, 422]}') == [73, 389, 740, 422]
    assert te.parse_box("nothing") is None
    assert te.iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert te.iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    assert te.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert te.centre_hit([4, 4, 6, 6], [0, 0, 10, 10]) is True
    assert te.centre_hit([9, 9, 20, 20], [0, 0, 10, 10]) is False


# --- QA scoring -------------------------------------------------------------
GOLD_PLACE = json.dumps({"line": 4, "text": "בפסטאט"}, ensure_ascii=False)
GOLD_ABSTAIN = json.dumps({"answer": "not stated"})


def test_qa_exact_hit() -> None:
    s = te.score_qa('{"line": 4, "text": "בפסטאט"}', GOLD_PLACE, "qa_place")
    assert s["parsed"] and s["line_hit"] and s["exact"] and s["contains"]
    assert s["span_cer"] == pytest.approx(0.0)
    assert s["exact_family"] is True and s["pred_abstain"] is False and s["abstain_correct"] is True


def test_qa_wrong_line_partial_span() -> None:
    s = te.score_qa('{"line": 5, "text": "פסטאט"}', GOLD_PLACE, "qa_place")
    assert s["line_hit"] is False and s["exact"] is False
    assert 0 < s["span_cer"] < 0.5
    assert s["contains"] is False  # a sub-token is not a whole-token containment


def test_qa_longer_quote_counts_as_containment() -> None:
    s = te.score_qa('{"line": 4, "text": "נכתב בפסטאט העיר"}', GOLD_PLACE, "qa_place")
    assert s["line_hit"] and s["contains"] and not s["exact"]


def test_qa_abstain_both_branches() -> None:
    right = te.score_qa('{"answer": "not stated"}', GOLD_ABSTAIN, "qa_abstain")
    assert right["gold_abstain"] and right["pred_abstain"] and right["abstain_correct"]
    assert right["line_hit"] is None and right["span_cer"] is None
    wrong = te.score_qa('{"line": 2, "text": "בתמוז"}', GOLD_ABSTAIN, "qa_abstain")
    assert wrong["abstain_correct"] is False
    false_abstain = te.score_qa("not stated", GOLD_PLACE, "qa_place")
    assert false_abstain["pred_abstain"] and false_abstain["abstain_correct"] is False
    assert false_abstain["exact"] is False and false_abstain["span_cer"] == 1.0


def test_qa_unparseable_prediction() -> None:
    s = te.score_qa("I cannot read this page.", GOLD_PLACE, "qa_place")
    assert s["parsed"] is False and s["exact"] is False and s["span_cer"] == 1.0 and s["line_hit"] is False


def test_qa_list_answer_greedy_match() -> None:
    gold = json.dumps([{"line": 19, "text": "נתן ביר שמואל"}, {"line": 20, "text": "יוסף בן יעקב"}], ensure_ascii=False)
    pred = '[{"line": 20, "text": "יוסף בן יעקב"}, {"line": 19, "text": "נתן ביר שמואל"}]'
    s = te.score_qa(pred, gold, "qa_witness_list")
    assert s["exact"] and s["line_hit"] and s["span_cer"] == pytest.approx(0.0)


# --- grounding scoring ------------------------------------------------------
def test_box_scoring_with_and_without_text() -> None:
    gold = json.dumps({"bbox_2d": [100, 400, 700, 440]})
    good = te.score_box('{"bbox_2d": [110, 395, 690, 445]}', gold)
    assert good["parsed"] and good["hit"] and good["iou"] == pytest.approx(0.7785, abs=1e-3) and good["span_cer"] is None
    off = te.score_box('{"bbox_2d": [100, 600, 700, 640]}', gold)
    assert off["hit"] is False and off["iou"] == 0.0
    none = te.score_box("no box", gold)
    assert none["parsed"] is False and none["iou"] == 0.0 and none["hit"] is False
    gold_text = json.dumps({"bbox_2d": [100, 400, 700, 440], "text": "שלום עולם"}, ensure_ascii=False)
    with_text = te.score_box('{"bbox_2d": [100, 400, 700, 440], "text": "שלום עולם"}', gold_text)
    assert with_text["span_cer"] == pytest.approx(0.0)


def test_text_scoring() -> None:
    assert te.score_text("שביתה במקומן", "שביתה במקומן")["span_cer"] == pytest.approx(0.0)
    assert te.score_text("", "שביתה")["parsed"] is False


def test_score_row_dispatch_and_max_tokens() -> None:
    qa = {"source": "pgp_qa", "task": "qa_place", "answer": GOLD_PLACE}
    box = {"source": "documentary_grounding", "task": "locate", "answer": json.dumps({"bbox_2d": [1, 2, 3, 4]})}
    txt = {"source": "documentary_grounding", "task": "read_box", "answer": "abc"}
    other = {"source": "ktiv_grounding", "task": "grounded_page", "answer": "[]"}
    assert te.score_row(qa, '{"line": 4, "text": "בפסטאט"}')["kind"] == "qa"
    assert te.score_row(box, "[1, 2, 3, 4]")["kind"] == "box"
    assert te.score_row(txt, "abc")["kind"] == "text"
    assert te.score_row(other, "[]") == {"kind": "unsupported", "parsed": True}
    assert te.score_row(qa, None)["exact"] is False  # failed decode scores as a miss
    assert te.max_tokens_for(qa) == te.MAX_TOKENS["qa"] and te.max_tokens_for(box) == te.MAX_TOKENS["box"]


# --- aggregation ------------------------------------------------------------
def test_aggregate_rates_and_medians() -> None:
    qa_row = {"source": "pgp_qa", "task": "qa_place", "answer": GOLD_PLACE}
    ab_row = {"source": "pgp_qa", "task": "qa_abstain", "answer": GOLD_ABSTAIN}
    box_row = {"source": "documentary_grounding", "task": "locate", "answer": json.dumps({"bbox_2d": [0, 0, 10, 10]})}
    recs = [
        {**qa_row, "score": te.score_row(qa_row, '{"line": 4, "text": "בפסטאט"}')},
        {**qa_row, "score": te.score_row(qa_row, '{"line": 9, "text": "xx"}')},
        {**ab_row, "score": te.score_row(ab_row, '{"answer": "not stated"}')},
        {**box_row, "score": te.score_row(box_row, '{"bbox_2d": [0, 0, 10, 10]}')},
        {**box_row, "score": te.score_row(box_row, "none")},
    ]
    agg = te.aggregate(recs)
    assert agg["pgp_qa/qa_place"]["exact"] == 0.5 and agg["pgp_qa/qa_place"]["line_hit"] == 0.5
    assert agg["pgp_qa/qa_abstain"]["abstain_acc_on_abstain_rows"] == 1.0
    assert agg["pgp_qa/ALL"]["n"] == 3 and agg["pgp_qa/ALL"]["false_abstain_rate"] == 0.0
    assert agg["documentary_grounding/locate"]["hit"] == 0.5 and agg["documentary_grounding/locate"]["parse_rate"] == 0.5
    assert agg["documentary_grounding/ALL"]["iou_median"] == pytest.approx(0.5)
    text = te.format_summary(agg, "m")
    assert "pgp_qa/qa_place" in text and "documentary_grounding/ALL" in text


def test_load_done_skips_failed_decodes(tmp_path) -> None:
    p = tmp_path / "out.jsonl"
    p.write_text(json.dumps({"key": "a", "raw": None}) + "\n" + json.dumps({"key": "b", "raw": "x"}) + "\n")
    assert set(te.load_done(p)) == {"b"}


def test_load_rows_defaults_source_and_excludes_train_images(tmp_path) -> None:
    import pandas as pd
    rows_dir = tmp_path / "rows"; rows_dir.mkdir()
    pd.DataFrame({"image_sha1": ["a", "b", "c"], "task": ["locate", "read_box", "locate"],
                  "question": ["q1", "q2", "q3"], "answer": ["x", "y", "z"]}).to_parquet(rows_dir / "val.parquet")
    pd.DataFrame({"image_sha1": ["b"], "other": [1]}).to_parquet(tmp_path / "train.parquet")
    rows = te.load_rows(tmp_path, "val", ["documentary_grounding"], None, None, exclude_train=tmp_path / "train.parquet")
    assert [r["image_sha1"] for r in rows] == ["a", "c"]
    assert all(r["source"] == "documentary_grounding" for r in rows)
    assert rows[0]["image_path"].endswith("images/a.jpg") and len(rows[0]["key"]) == 16
    limited = te.load_rows(tmp_path, "val", ["documentary_grounding"], ["locate"], 1)
    assert len(limited) == 1 and limited[0]["task"] == "locate"


def test_decode_rows_stops_when_model_disappears(tmp_path, monkeypatch) -> None:
    import asyncio
    calls = []

    async def fake_transcribe(model, image_path, prompt, temperature, max_tokens):
        calls.append(prompt)
        return None  # every request fails

    monkeypatch.setattr(te, "transcribe_with_lm_studio", fake_transcribe)
    rows = [{"key": f"k{i}", "image_sha1": f"s{i}", "image_path": "x.jpg", "question": f"q{i}", "answer": GOLD_PLACE,
             "source": "pgp_qa", "task": "qa_place"} for i in range(3)]
    out = tmp_path / "o.jsonl"
    recs = asyncio.run(te.decode_rows(rows, "m", out, 0.1, {}, min_free_gb=0.0,
                                      served_check=lambda m: False, free_check=lambda: 100.0))
    assert len(calls) == 1 and len(recs) == 0  # the failed row is not written
    assert te.load_done(out) == {}  # the failed row is retried on the next run


def test_free_gb_and_model_served_are_safe() -> None:
    assert te.free_gb("/") > 0
    assert te.model_served("no-such-model", base_url="http://127.0.0.1:9/v1") is False
