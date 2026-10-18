# File name: test_colab_notebook_v23q.py
# Date: 10/5/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Tests for the v23q (page parse + questions) notebook derivation.

Every intended delta lands, the cells compile, and the page-parse data gate accepts rows of the agreed
shapes and stops on each kind of malformed target.
"""
import ast
import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from src.finetuning.qwen_hebrew import build_vqa_parse as vqa
from src.finetuning.qwen_hebrew.colab import derive_v23q as dq

SRC = Path("src/finetuning/qwen_hebrew/colab/genizah_v22b.ipynb")
SHA = "b" * 40
PARSE = '[\n{"n": 1, "text": "בשמך רחמנא"},\n{"n": 2, "text": "כתאבי אליך [...] כד אלול"}\n]'
CONTEXT = vqa.PARSE_LEAD + PARSE + "\n\n"
PAGE = "בשמך רחמנא\nכתאבי אליך [...] כד אלול"


@pytest.fixture(scope="module")
def derived():
    nb = json.load(open(SRC, encoding="utf-8"))
    return nb, dq.derive(nb, SHA)


def _src(nb, i):
    return "".join(nb["cells"][i]["source"])


def test_cells_compile_and_only_planned_cells_change(derived):
    a, b = derived
    assert len(b["cells"]) == len(a["cells"])
    for c in b["cells"]:
        if c["cell_type"] == "code":
            ast.parse("\n".join(l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith(("%", "!"))))
    for i in (1, 3, 5, 6):
        assert _src(a, i) == _src(b, i), i                                  # install, model + warm start, collator helpers
    assert a == json.load(open(SRC, encoding="utf-8")) and dq.derive(copy.deepcopy(a), SHA) == b


def test_data_cell_points_at_the_page_parse_mixture(derived):
    _, b = derived
    c2 = _src(b, 2)
    assert f'V22_REPO = "{dq.DATA_REPO}"' in c2 and f'V22_REVISION = "{SHA}"' in c2
    assert 'LOCAL = "/content/genizah_v23q_vqa"' in c2 and "genizah_v22_full" not in c2
    assert f"len(train_ds) >= {dq.MIN_TRAIN_ROWS} and len(eval_ds) >= {dq.MIN_VAL_ROWS}" in c2
    assert f"for _s, _lo in {dq.SHARE_FLOORS}:" in c2 and dq.NEW_HYGIENE in c2
    for gone in ('"pgp_qa", 0.10', "_loc = ", "no QA rows in the mixture", "bad documentary box", "read_box prompt drift"):
        assert gone not in c2, gone
    for kept in ('assert set(eval_ds.column("source")) == set(_src), "val must cover every train source"',
                 '"internal gap token leaked"', '"mojibake leaked"', '"expected damage-gap markers"'):
        assert kept in c2, kept


def test_share_floors_name_real_components_and_leave_room():
    from src.finetuning.qwen_hebrew.build_v22_mixture import COMPONENTS
    floors = dict(ast.literal_eval(dq.SHARE_FLOORS))
    assert set(floors) <= set(COMPONENTS) and 0.6 < sum(floors.values()) < 0.95
    assert {c for c in COMPONENTS if c.startswith("vqa_")} <= set(floors)
    assert dq.MAX_STEPS * 8 == dq.TRAIN_ROWS and dq.MIN_TRAIN_ROWS < dq.TRAIN_ROWS


def _run_gate(rows):
    """Execute the page-parse hygiene block on ``(answer, question, source, task)`` rows."""
    ns = {"_ans": [r[0] for r in rows], "_q": [r[1] for r in rows], "_srcs": [r[2] for r in rows], "_t": [r[3] for r in rows],
          "_transc": [r[0] for r in rows if r[3].endswith("transcribe")], "Counter": Counter, "json": json}
    exec(compile(dq.NEW_HYGIENE, "<page-parse hygiene>", "exec"), ns)


QUESTION = CONTEXT + "In which month was this document written? " + vqa.ANSWER_INSTRUCTION
FIELDS = CONTEXT + vqa.FIELDS_HEAD + "\n- month: In which month?\n" + vqa.FIELDS_TAIL.format(keys='"month"')
LOOKUP = CONTEXT + vqa.LOOKUP_LINE_PROMPT.format(phrase="כתאבי אליך")
BOXES = '[{"n": 1, "text": "בשמך רחמנא", "bbox_2d": [100, 50, 900, 120]}, {"n": 2, "text": "כתאבי אליך", "bbox_2d": [90, 130, 910, 200]}]'
GOOD = [
    ('[{"n": 1, "text": "בשמך רחמנא"}, {"n": 2, "text": "כתאבי אליך [...] כד אלול"}]', vqa.PARSE_PROMPT, "vqa_parse_lines", "parse_lines"),
    (BOXES, vqa.PARSE_BOXES_PROMPT, "vqa_parse_boxes", "parse_lines_boxes"),
    ('{"answer": "אלול", "line": 2}', QUESTION, "vqa_question", "question_from_parse"),
    ('{"answer": null}', QUESTION, "vqa_question", "question_from_parse"),
    ('{"answer": [{"text": "יעקב", "line": 1}, {"text": "יוסף", "line": 2}]}', QUESTION, "vqa_question", "question_from_parse"),
    ('{"month": {"text": "אלול", "line": 2}, "year": null}', FIELDS, "vqa_fields", "fields_from_parse"),
    ('{"witnesses": [{"text": "יעקב", "line": 1}]}', FIELDS, "vqa_fields", "fields_from_parse"),
    ('{"line": 2, "text": "כתאבי אליך [...] כד אלול"}', LOOKUP, "vqa_lookup", "lookup_from_parse"),
    ('{"answer": "כד אלול", "line": 2}', LOOKUP, "vqa_lookup", "lookup_from_parse"),
    (PAGE, "Transcribe the handwritten Hebrew script.", "pgp_edition_pages", "fragment_transcribe"),
    (PAGE, "Transcribe the handwritten Hebrew script.", "ktiv_transcription", "fragment_transcribe"),
]


def test_gate_accepts_every_agreed_shape(capsys):
    _run_gate(GOOD)
    assert "hygiene OK" in capsys.readouterr().out


@pytest.mark.parametrize("bad, message", [
    (('[{"n": 2, "text": "בשמך"}]', vqa.PARSE_PROMPT, "vqa_parse_lines", "parse_lines"), "parse numbering"),
    (('[{"n": 1, "text": " "}]', vqa.PARSE_PROMPT, "vqa_parse_lines", "parse_lines"), "empty parse line"),
    (('[{"n": 1, "text": "בשמך", "bbox_2d": [1, 2, 3, 4]}]', vqa.PARSE_PROMPT, "vqa_parse_lines", "parse_lines"), "parse keys"),
    (('[{"n": 1, "text": "בשמך", "bbox_2d": [900, 50, 100, 120]}]', vqa.PARSE_BOXES_PROMPT, "vqa_parse_boxes", "parse_lines_boxes"), "bad line box"),
    (('{"answer": "אלול", "line": 3}', QUESTION, "vqa_question", "question_from_parse"), "bad answer"),            # the reading has 2 lines
    (('{"answer": "אלול"}', QUESTION, "vqa_question", "question_from_parse"), "bad answer"),
    (('{"answer": null, "line": 2}', QUESTION, "vqa_question", "question_from_parse"), "bad answer"),
    (('{"month": {"text": "אלול", "line": 0}}', FIELDS, "vqa_fields", "fields_from_parse"), "bad field value"),
    (('{"month": "אלול"}', FIELDS, "vqa_fields", "fields_from_parse"), "bad field value"),
    (('{"line": 7, "text": "כתאבי"}', LOOKUP, "vqa_lookup", "lookup_from_parse"), "bad look-up answer"),
    (('{"answer": "אלול", "line": 2}', "In which month? Answer with JSON.", "vqa_question", "question_from_parse"), "does not open with the reading"),
    (('{"answer": "אלול", "line": 2}', QUESTION, "vqa_question", "essay"), "unknown page-parse task"),
    (('{"line": 1, "text": "תשרי"}', "When? JSON", "pgp_qa", "qa_date"), "older question or box formats"),
    (('{"bbox_2d": [1, 2, 3, 4]}', "Locate, 0-1000.", "ktiv_grounding", "locate"), "older question or box formats"),
])
def test_gate_stops_on_each_malformed_row(bad, message):
    with pytest.raises(AssertionError, match=message):
        _run_gate(GOOD + [bad])


def test_gate_needs_page_parse_rows_and_not_stated_rows():
    with pytest.raises(AssertionError, match="no page-parse rows"):
        _run_gate(GOOD[-2:])
    with pytest.raises(AssertionError, match="no 'not stated' rows"):
        _run_gate([r for r in GOOD if r[0] != '{"answer": null}'])


def test_collator_guardrail_uses_a_question_row(derived):
    _, b = derived
    c4 = _src(b, 4)
    assert 's == "vqa_question")' in c4 and '"pgp_qa"' not in c4 and "page + question with a reading" in c4


def test_training_cells_deltas(derived):
    _, b = derived
    c7, c8, c9 = _src(b, 7), _src(b, 8), _src(b, 9)
    assert f"\nMAX_STEPS = {dq.MAX_STEPS}\n" in c7 and "MAX_STEPS = 2500" not in c7
    for needle in (f'CKPT_REPO = "{dq.CKPT_REPO}"', 'OUT_DIR = "outputs_v23q"', 'RESUME_ROOT = "/content/v23q_resume"',
                   "max_steps=MAX_STEPS", f"learning_rate={dq.LR},", f"warmup_ratio={dq.WARMUP_RATIO},",
                   f'run_name="{dq.RUN_NAME}"', f'wandb.init(project="qwen-hebrew-finetune", name="{dq.RUN_NAME}", ',
                   "hub_always_push=True", 'hub_strategy="checkpoint", hub_private_repo=False,'):
        assert needle in c8, needle
    for gone in ("v22b-ckpt", "outputs_v22b", "v22b_resume", 'name="genizah_v22b"', "learning_rate=3e-5", "warmup_ratio=0.02"):
        assert gone not in c8, gone
    assert f'MERGED_REPO = "{dq.MERGED_REPO}"' in c9 and c9.count('"v23q-merged"') == 2 and "v22b-merged" not in c9


def test_intro_and_attribution(derived):
    _, b = derived
    intro = _src(b, 0)
    assert "v2.3q" in intro and dq.DATA_REPO in intro and dq.CKPT_REPO in intro and "run every cell again" in intro
    text = json.dumps(b, ensure_ascii=False).lower()
    assert not any(bad in text for bad in ("friedberg", "fjms", "fjp"))


def test_bad_revision_rejected(tmp_path):
    with pytest.raises(SystemExit):
        dq.main(["--revision", "not-a-sha", "--out", str(tmp_path / "x.ipynb")])
    dq.main(["--revision", "PIN-AFTER-PUSH", "--out", str(tmp_path / "y.ipynb")])
    assert json.load(open(tmp_path / "y.ipynb"))["cells"]
