# File name: test_run_vqa_parse_eval.py
# Date: 10/5/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Tests for the page-parse evaluation: reply decoding, answer matching, the two-step run and its scores."""
import asyncio
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.datasets.evaluations.helper_eval_scripts import run_vqa_parse_eval as ev
from src.finetuning.qwen_hebrew import build_vqa_parse as vqa
from src.finetuning.qwen_hebrew.images_once import SHA_COLUMN

GOLD_LINES = ["בשמך רחמנא", "כתאבי אליך [...] כד אלול", "שלום"]
QUESTION = "In which month was this document written? " + vqa.ANSWER_INSTRUCTION


def test_reply_json_takes_fenced_and_plain_json_and_rejects_the_rest():
    assert ev.reply_json('```json\n{"answer": null}\n```') == {"answer": None}
    assert ev.reply_json(' [{"n": 1, "text": "א"}] ') == [{"n": 1, "text": "א"}]
    assert ev.reply_json("אלול") is None and ev.reply_json("") is None and ev.reply_json(None) is None
    assert ev.reply_json('{"answer": "אלול", "line": 2') is None                      # cut off


def test_parse_lines_needs_a_list_of_text_objects():
    assert ev.parse_lines([{"n": 1, "text": "א"}, {"n": 2, "text": "ב", "bbox_2d": [1, 2, 3, 4]}]) == ["א", "ב"]
    for bad in (None, [], {"text": "א"}, ["א"], [{"n": 1}], [{"n": 1, "text": 5}]):
        assert ev.parse_lines(bad) is None


def test_request_of_returns_what_follows_the_parse():
    prompt = vqa.context_prompt(GOLD_LINES, QUESTION)
    assert ev.request_of(prompt) == QUESTION
    assert vqa.context_prompt(["שורה אחרת"], ev.request_of(prompt)).endswith(QUESTION)
    with pytest.raises(ValueError):
        ev.request_of(QUESTION)


@pytest.mark.parametrize("task, gold, reply, expected", [
    ("question_from_parse", '{"answer": "אלול", "line": 2}', {"answer": "אלול", "line": 5}, (1, 1)),     # text decides, not the number
    ("question_from_parse", '{"answer": "אלול", "line": 2}', {"answer": "תמוז", "line": 2}, (0, 1)),
    ("question_from_parse", '{"answer": "ליצירה שלום", "line": 2}', {"answer": "לִיצִירָה  שלום", "line": 2}, (1, 1)),   # nikud, spaces
    ("question_from_parse", '{"answer": "שלום", "line": 3}', {"answer": "שלומ", "line": 3}, (1, 1)),      # final letters fold
    ("question_from_parse", '{"answer": null}', {"answer": None}, (1, 1)),
    ("question_from_parse", '{"answer": null}', {"answer": "אלול", "line": 2}, (0, 1)),
    ("question_from_parse", '{"answer": "אלול", "line": 2}', {"answer": None}, (0, 1)),
    ("question_from_parse", '{"answer": "אלול", "line": 2}', None, (0, 1)),                                # not JSON
    ("question_from_parse", '{"answer": [{"text": "יעקב", "line": 1}, {"text": "יוסף", "line": 2}]}',
     {"answer": [{"text": "יוסף", "line": 9}, {"text": "יעקב", "line": 1}]}, (1, 1)),                       # a set of names
    ("question_from_parse", '{"answer": [{"text": "יעקב", "line": 1}, {"text": "יוסף", "line": 2}]}',
     {"answer": [{"text": "יעקב", "line": 1}]}, (0, 1)),
    ("lookup_from_parse", '{"line": 2, "text": "כתאבי אליך"}', {"line": 2, "text": "כתאבי אליך"}, (1, 1)),
    ("lookup_from_parse", '{"answer": "כד אלול", "line": 2}', {"answer": "כד תמוז", "line": 2}, (0, 1)),
    ("fields_from_parse", '{"month": {"text": "אלול", "line": 2}, "year": null, "place": {"text": "מצר", "line": 1}}',
     {"month": {"text": "אלול", "line": 2}, "year": None, "place": {"text": "אסכנדריה", "line": 1}}, (2, 3)),
    ("fields_from_parse", '{"month": {"text": "אלול", "line": 2}, "year": null}', {"month": "אלול"}, (0, 2)),   # wrong shape, key missing
    ("fields_from_parse", '{"month": {"text": "אלול", "line": 2}}', None, (0, 1)),
])
def test_answer_matches(task, gold, reply, expected):
    assert ev.answer_matches(task, gold, reply) == expected


def _export(tmp_path: Path) -> Path:
    """A tiny export: two held-out pages; page a has a question and a fields row, page b only its parse."""
    rows = [
        {"stem": "a_parse", "task": "parse_lines", "question": vqa.PARSE_PROMPT, "answer": vqa.parse_json(GOLD_LINES), SHA_COLUMN: "aaa"},
        {"stem": "a_q", "task": "question_from_parse", "question": vqa.context_prompt(GOLD_LINES, QUESTION),
         "answer": '{"answer": "אלול", "line": 2}', SHA_COLUMN: "aaa"},
        {"stem": "a_fields", "task": "fields_from_parse", "question": vqa.context_prompt(GOLD_LINES, "Fill in these fields from this page:\n- month\n- year"),
         "answer": '{"month": {"text": "אלול", "line": 2}, "year": null}', SHA_COLUMN: "aaa"},
        {"stem": "b_parse", "task": "parse_lines", "question": vqa.PARSE_PROMPT, "answer": vqa.parse_json(["שורה אחת"]), SHA_COLUMN: "bbb"},
    ]
    (tmp_path / "rows").mkdir()
    (tmp_path / "images").mkdir()
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "rows" / "val.parquet")
    return tmp_path


def _fake_model(calls):
    """A model that parses page a with one misread letter, fails to parse page b, and answers from what it is shown."""
    async def transcribe(model, image, prompt, max_tokens=0):
        calls.append((Path(image).name, prompt, max_tokens))
        if prompt == vqa.PARSE_PROMPT:
            return vqa.parse_json(["בשמך רחמנא", "כתאבי אליך כד אלול", "שלוס"]) if "aaa" in image else "I cannot read this page."
        if "Fill in these fields" in prompt:
            return '{"month": {"text": "אלול", "line": 2}, "year": {"text": "קצז", "line": 2}}'
        return '```json\n{"answer": "אלול", "line": 2}\n```' if "כתאבי אליך כד אלול" in prompt else '{"answer": "תמוז", "line": 2}'
    return transcribe


def test_run_asks_parse_then_each_row_with_both_parses_and_resumes(tmp_path):
    export, out, calls = _export(tmp_path), tmp_path / "out", []
    stats = asyncio.run(ev.run("m/x", export, out, transcribe=_fake_model(calls), served_check=lambda m: True, free_check=lambda: 99.0))
    assert stats == {"asked": 6, "failed": 0, "skipped": 0, "stopped": 0}            # 2 parses + 2 rows x 2 conditions
    prompts = [c[1] for c in calls]
    assert prompts[0] == vqa.PARSE_PROMPT and calls[0][2] == ev.PARSE_MAX_TOKENS and calls[0][0] == "aaa.jpg"
    own = [p for p in prompts if p.startswith(vqa.PARSE_LEAD) and '"text": "שלוס"' in p]
    gold = [p for p in prompts if p.startswith(vqa.PARSE_LEAD) and '"text": "שלום"' in p]
    assert len(own) == 2 and len(gold) == 2 and all(p.endswith(QUESTION) or "Fill in these fields" in p for p in own + gold)
    records = [json.loads(l) for l in open(out / "m_x.jsonl", encoding="utf-8")]
    assert {(r["stem"], r["condition"]) for r in records} == {("a_parse", "parse"), ("b_parse", "parse"), ("a_q", "own"), ("a_q", "gold"),
                                                              ("a_fields", "own"), ("a_fields", "gold")}
    again = asyncio.run(ev.run("m/x", export, out, transcribe=_fake_model(calls), served_check=lambda m: True, free_check=lambda: 99.0))
    assert again == {"asked": 0, "failed": 0, "skipped": 6, "stopped": 0} and len(calls) == 6


def test_run_stops_when_the_model_is_gone(tmp_path):
    async def dead(model, image, prompt, max_tokens=0):
        return None
    stats = asyncio.run(ev.run("m", _export(tmp_path), tmp_path / "out", transcribe=dead, served_check=lambda m: False, free_check=lambda: 99.0))
    assert stats["stopped"] == 1 and stats["failed"] == 1


def test_score_counts_json_parse_cer_and_answers_per_condition(tmp_path, capsys):
    export, out = _export(tmp_path), tmp_path / "out"
    asyncio.run(ev.run("m", export, out, transcribe=_fake_model([]), served_check=lambda m: True, free_check=lambda: 99.0))
    res = ev.score(out, export)["m"]
    assert res["parse"]["pages"] == 2 and res["parse"]["json"] == 1                   # page b's reply is not JSON
    assert 0.2 < res["parse"]["cer_pooled"] < 0.3 and res["parse"]["cer_median"] == 1.0   # page b holds no line: CER 1
    assert res["question_from_parse|own"] == {"right": 1, "asked": 1, "json": 1, "rows": 1, "missing": 0}
    assert res["question_from_parse|gold"]["right"] == 0                              # the fake answers תמוז when shown the gold lines
    assert res["fields_from_parse|own"] == {"right": 1, "asked": 2, "json": 1, "rows": 1, "missing": 0}   # month right, year invented
    table = capsys.readouterr().out
    assert "| m | 2 (1) |" in table and "1/1" in table and "1/2" in table
    assert ev.visible_text(["א  [...] ב", "ג [?]"]) == ev.visible_text(["א ב", "ג"])


def test_missing_replies_count_as_asked_and_wrong(tmp_path):
    export = _export(tmp_path)
    pages = ev.val_pages(export)
    assert [p["parse"]["stem"] for p in pages] == ["a_parse", "b_parse"] and len(pages[0]["context"]) == 2
    assert len(ev.val_pages(export, limit_pages=1)) == 1
    res = ev.score_model(pages, {})
    assert res["question_from_parse|own"] == {"right": 0, "asked": 1, "json": 0, "rows": 1, "missing": 1}
    assert res["parse"] == {"pages": 0, "json": 0, "cer_median": None, "cer_pooled": None}


def test_a_short_run_asks_only_the_chosen_families_and_conditions_on_pages_that_have_them(tmp_path):
    export, calls = _export(tmp_path), []
    stats = asyncio.run(ev.run("m", export, tmp_path / "out", transcribe=_fake_model(calls), served_check=lambda m: True,
                               free_check=lambda: 99.0, families=("question_from_parse",), conditions=("own",), only_with_rows=True))
    assert stats["asked"] == 2                                     # page a: its parse and its one question with the own parse
    assert [c[0] for c in calls] == ["aaa.jpg", "aaa.jpg"] and '"text": "שלוס"' in calls[1][1]
    assert [p["parse"]["stem"] for p in ev.val_pages(export, families=("lookup_from_parse",), only_with_rows=True)] == []


def test_reply_lines_reads_what_a_malformed_parse_still_holds():
    good = vqa.parse_json(["שורה אחת", "שורה שתיים"])
    assert ev.reply_lines(good) == ["שורה אחת", "שורה שתיים"]
    stray_quote = '[\n{"n": 1, "text": "הודה ס"ז לתת"},\n{"n": 2, "text": "שורה שתיים"}\n]'
    assert ev.reply_json(stray_quote) is None and ev.reply_lines(stray_quote) == ['הודה ס"ז לתת', "שורה שתיים"]
    cut_off = '[\n{"n": 1, "text": "שורה אחת"},\n{"n": 2, "text": "שורה שת'
    assert ev.reply_lines(cut_off) == ["שורה אחת"]
    single = '{"n": 1, "text": "תבואתך טובה ורחבה"}'                     # one element without the array (zero-shot v21b)
    assert ev.parse_lines(ev.reply_json(single)) is None and ev.reply_lines(single) == ["תבואתך טובה ורחבה"]
    boxed = '[{"n": 1, "text": "עם תיבה", "bbox_2d": [1, 2, 3, 4]}, {"n": 2, "text": "מילה \\"מצוטטת\\""}]'
    assert ev.reply_lines(boxed) == ["עם תיבה", 'מילה "מצוטטת"']
    for nothing in (None, "", "I cannot read this page.", '{"answer": null}'):
        assert ev.reply_lines(nothing) is None


def test_score_table_shows_a_dash_for_a_condition_that_was_never_asked(tmp_path, capsys):
    export, out = _export(tmp_path), tmp_path / "out"
    asyncio.run(ev.run("m", export, out, transcribe=_fake_model([]), served_check=lambda m: True, free_check=lambda: 99.0,
                       conditions=("own",)))
    ev.score(out, export)
    row = [l for l in capsys.readouterr().out.splitlines() if l.startswith("| m |")][0]
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[3:7] == ["1/2", "-", "1/1", "-"]                      # fields own, fields gold, question own, question gold
