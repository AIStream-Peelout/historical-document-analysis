# File name: test_colab_notebook_v24q.py
# Date: 10/18/26
# Author: Isaac Godfried. Coded originally by Claude Sonnet 5.5.
"""Tests for the v24q (page parse + questions, warm start v2.4a step 2000) notebook derivation.

v24q is v23q on another warm start: the data, helper, install and collator cells are v23q's byte for byte, every intended
delta lands (warm start v2.4a step 2000 and its step check, names, private checkpoints, 50-step saves, intro), the cells
compile, the page-parse data gate (executed from the derived notebook's own text) accepts rows of the agreed shapes and
stops on each kind of malformed target, and the Lambda edition keeps all of it.
"""
import ast
import copy
import difflib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pytest

from src.finetuning.qwen_hebrew import build_vqa_parse as vqa
from src.finetuning.qwen_hebrew.colab import derive_v22b as d22
from src.finetuning.qwen_hebrew.colab import derive_v23q as dq
from src.finetuning.qwen_hebrew.colab import derive_v24a as d4
from src.finetuning.qwen_hebrew.colab import derive_v24q as dv
from src.finetuning.qwen_hebrew.colab import host_lambda as hl

SRC = Path("src/finetuning/qwen_hebrew/colab/genizah_v22b.ipynb")
COLAB_DIR = Path("src/finetuning/qwen_hebrew/colab")
CHAIN_LOG = Path("logs/v24a_eval_chain.log")                   # git-ignored: the test that reads it skips without it
V23Q_REVISION = "a600dacd02365b1d16475529832e8c701d27b304"
SHA = "b" * 40
PARSE = '[\n{"n": 1, "text": "בשמך רחמנא"},\n{"n": 2, "text": "כתאבי אליך [...] כד אלול"}\n]'
CONTEXT = vqa.PARSE_LEAD + PARSE + "\n\n"
PAGE = "בשמך רחמנא\nכתאבי אליך [...] כד אלול"

Notebook = Dict
Editions = Tuple[Notebook, Notebook, Notebook]


@pytest.fixture(scope="module")
def editions() -> Editions:
    """(v23q Colab edition at its pinned revision, v24q Colab edition, v24q Lambda edition)."""
    nb = json.load(open(SRC, encoding="utf-8"))
    colab = dv.derive(nb)
    return dq.derive(nb, V23Q_REVISION), colab, dv.to_lambda(colab, "genizah_v24q_lambda.ipynb")


def _src(nb: Notebook, i: int) -> str:
    """Source of cell ``i``.

    :param nb: Notebook.
    :type nb: Notebook
    :param i: Cell index.
    :type i: int
    :return: The joined source.
    :rtype: str
    """
    return "".join(nb["cells"][i]["source"])


def _changed(a: Notebook, b: Notebook, i: int) -> Tuple[List[str], List[str]]:
    """(lines only in ``a``, lines only in ``b``) of cell ``i``.

    :param a: Notebook before.
    :type a: Notebook
    :param b: Notebook after.
    :type b: Notebook
    :param i: Cell index.
    :type i: int
    :return: The removed and the added lines.
    :rtype: Tuple[List[str], List[str]]
    """
    diff = [l for l in difflib.unified_diff(_src(a, i).splitlines(), _src(b, i).splitlines(), lineterm="", n=0)
            if not l.startswith(("---", "+++", "@@"))]
    return [l[1:] for l in diff if l.startswith("-")], [l[1:] for l in diff if l.startswith("+")]


def _parses(nb: Notebook) -> None:
    """Every code cell parses (IPython magics and shell lines left out).

    :param nb: Notebook.
    :type nb: Notebook
    """
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            ast.parse("\n".join(l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith(("%", "!"))))


# ----------------------------------------------------------------------------- structure


def test_cells_compile_and_only_planned_cells_change(editions: Editions) -> None:
    """Cells 0, 3, 8, 9 differ from v23q's; install, data + gates, collator, helpers and schedule are v23q's exactly."""
    v23q, colab, lam = editions
    assert len(colab["cells"]) == len(v23q["cells"]) == 10 and len(lam["cells"]) == 11
    assert {k: v for k, v in colab.items() if k != "cells"} == {k: v for k, v in v23q.items() if k != "cells"}     # metadata, nbformat
    _parses(colab)
    _parses(lam)
    for i in (1, 2, 4, 5, 6, 7):                                    # install, data + gates, collator, grad gate, bench, helpers + schedule
        assert _src(v23q, i) == _src(colab, i), i
    for i in (0, 3, 8, 9):                                          # intro, model + warm start, training names, merged names
        assert _src(v23q, i) != _src(colab, i), i
    for i in (3, 4, 5, 6, 9):                                       # the Lambda edition leaves these as the Colab edition has them
        assert _src(colab, i) == _src(lam, i), i


def test_source_notebook_is_not_modified_and_derivation_is_deterministic(editions: Editions) -> None:
    """Deriving twice gives the same notebook and never edits the v22b source in place."""
    nb = json.load(open(SRC, encoding="utf-8"))
    assert dv.derive(copy.deepcopy(nb)) == dv.derive(nb) == editions[1]
    assert nb == json.load(open(SRC, encoding="utf-8"))


def test_only_the_planned_lines_differ_from_v23q(editions: Editions) -> None:
    """Line by line, the changes are the warm start, the names, the save interval and the private flag, and nothing else."""
    v23q, colab, _ = editions
    out, new = _changed(v23q, colab, 3)
    assert len(out) == 6 and len(new) == 11                         # title, the 4-line pin -> 5 lines, the download + its 4-line step check
    assert out == [d4.OLD_MODEL_TITLE, *d4.OLD_WARM.splitlines(), d4.OLD_WARM_DOWNLOAD.splitlines()[1]]
    assert new == [dv.NEW_MODEL_TITLE, *dv.NEW_WARM.splitlines(), *dv.NEW_WARM_DOWNLOAD.splitlines()[1:]]
    out, new = _changed(v23q, colab, 8)
    assert len(out) == len(new) == 8                                # title, repo, out dir, resume root, save steps, private, run + W&B names
    assert all(re.search(r"v2\.?3q|save_steps=100|hub_private_repo=False", l) for l in out)
    assert all(re.search(r"v2\.?4q|save_steps=50|hub_private_repo=True", l) for l in new)
    for line in out:                                                # steps, learning rate, schedule, seed, gates are not among the changes
        assert not re.search(r"max_steps|learning_rate|warmup_ratio|seed|gate|dispatch|lock_session_plan|train_with_exact_resume", line), line
    out, new = _changed(v23q, colab, 9)
    assert len(out) == len(new) == 3                                # merged repo, merged folder twice
    assert all("v23q" in l for l in out) and all("v24q" in l for l in new)


def test_committed_baseline_and_notebooks_are_current() -> None:
    """genizah_v23q.ipynb is what derive_v23q writes (v24q builds on it); the committed v24q files equal a fresh derivation."""
    nb = json.load(open(SRC, encoding="utf-8"))
    assert dq.derive(nb, V23Q_REVISION) == json.load(open(COLAB_DIR / "genizah_v23q.ipynb", encoding="utf-8"))
    colab = json.load(open(COLAB_DIR / dv.NOTEBOOK, encoding="utf-8"))
    revision = re.search(r'^V22_REVISION = "([^"]+)"', _src(colab, 2), re.M).group(1)
    assert revision == dv.DATA_REVISION == V23Q_REVISION
    fresh = dv.derive(nb, revision)
    assert colab == fresh
    lam_name = hl.lambda_files(dv.NOTEBOOK)[0]
    assert json.load(open(COLAB_DIR / lam_name, encoding="utf-8")) == dv.to_lambda(fresh, lam_name)


# ----------------------------------------------------------------------------- data cell and its gates (v23q's, unchanged)


def test_data_cell_points_at_the_page_parse_mixture(editions: Editions) -> None:
    """Same repository, pinned revision, size gate, share floors and hygiene block as v23q (the cell is identical)."""
    v23q, colab, _ = editions
    c2 = _src(colab, 2)
    assert c2 == _src(v23q, 2)
    assert f'V22_REPO = "{dq.DATA_REPO}"' in c2 and f'V22_REVISION = "{V23Q_REVISION}"' in c2 and dv.DATA_REPO == dq.DATA_REPO
    assert 'LOCAL = "/content/genizah_v23q_vqa"' in c2 and "genizah_v22_full" not in c2       # the dataset's folder: it is the same dataset
    assert f"len(train_ds) >= {dq.MIN_TRAIN_ROWS} and len(eval_ds) >= {dq.MIN_VAL_ROWS}" in c2
    assert f"for _s, _lo in {dq.SHARE_FLOORS}:" in c2 and dq.NEW_HYGIENE in c2
    for gone in ('"pgp_qa", 0.10', "_loc = ", "no QA rows in the mixture", "bad documentary box", "read_box prompt drift"):
        assert gone not in c2, gone
    for kept in ('assert set(eval_ds.column("source")) == set(_src), "val must cover every train source"',
                 '"internal gap token leaked"', '"mojibake leaked"', '"expected damage-gap markers"'):
        assert kept in c2, kept


def test_share_floors_name_real_components_and_leave_room(editions: Editions) -> None:
    """The notebook's share floors name components the mixture builder has and cover every page-parse component."""
    from src.finetuning.qwen_hebrew.build_v22_mixture import COMPONENTS
    line = next(l for l in _src(editions[1], 2).splitlines() if l.startswith("for _s, _lo in "))
    floors = dict(ast.literal_eval(line[len("for _s, _lo in "):-1]))
    assert set(floors) <= set(COMPONENTS) and 0.6 < sum(floors.values()) < 0.95
    assert {c for c in COMPONENTS if c.startswith("vqa_")} <= set(floors)
    assert dq.MAX_STEPS * 8 == dq.TRAIN_ROWS and dq.MIN_TRAIN_ROWS < dq.TRAIN_ROWS


def _gate_text(nb: Notebook) -> str:
    """The page-parse hygiene block as the derived notebook's data cell has it.

    :param nb: Notebook.
    :type nb: Notebook
    :return: Source from the block's first comment to the end of its final ``print``.
    :rtype: str
    """
    c2 = _src(nb, 2)
    start = c2.index("# Page-parse hygiene:")
    return c2[start:c2.index("\n", c2.index('print(f"hygiene OK'))]


def _run_gate(nb: Notebook, rows: Sequence[Tuple[str, str, str, str]]) -> None:
    """Execute the notebook's page-parse hygiene block on ``(answer, question, source, task)`` rows.

    :param nb: Notebook whose data cell holds the block.
    :type nb: Notebook
    :param rows: The rows to check.
    :type rows: Sequence[Tuple[str, str, str, str]]
    """
    ns = {"_ans": [r[0] for r in rows], "_q": [r[1] for r in rows], "_srcs": [r[2] for r in rows], "_t": [r[3] for r in rows],
          "_transc": [r[0] for r in rows if r[3].endswith("transcribe")], "Counter": Counter, "json": json}
    exec(compile(_gate_text(nb), "<page-parse hygiene>", "exec"), ns)


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


def test_gate_text_is_v23qs_and_accepts_every_agreed_shape(editions: Editions, capsys: pytest.CaptureFixture) -> None:
    """The block in the notebook is v23q's, and it accepts a row of each agreed shape."""
    assert _gate_text(editions[1]) == dq.NEW_HYGIENE.rstrip("\n")
    _run_gate(editions[1], GOOD)
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
def test_gate_stops_on_each_malformed_row(editions: Editions, bad: Tuple[str, str, str, str], message: str) -> None:
    """Each kind of malformed page-parse target stops the notebook's data cell with its own message."""
    with pytest.raises(AssertionError, match=message):
        _run_gate(editions[1], GOOD + [bad])


def test_gate_needs_page_parse_rows_and_not_stated_rows(editions: Editions) -> None:
    """A mixture without page-parse rows, or without a 'not stated' answer, is refused."""
    with pytest.raises(AssertionError, match="no page-parse rows"):
        _run_gate(editions[1], GOOD[-2:])
    with pytest.raises(AssertionError, match="no 'not stated' rows"):
        _run_gate(editions[1], [r for r in GOOD if r[0] != '{"answer": null}'])


def test_collator_guardrail_uses_a_question_row(editions: Editions) -> None:
    """The collator cell's second guardrail row is a question with a reading in its prompt."""
    c4 = _src(editions[1], 4)
    assert 's == "vqa_question")' in c4 and '"pgp_qa"' not in c4 and "page + question with a reading" in c4


# ----------------------------------------------------------------------------- warm start


def test_warm_start_is_v24a_step_2000(editions: Editions) -> None:
    """The model cell loads the v2.4a step-2000 adapter from its checkpoint repository at the pinned commit, and checks the step."""
    v23q, colab, _ = editions
    c3 = _src(colab, 3)
    assert dv.WARM_CKPT_REPO == d4.ARM.ckpt_repo == "isaacmg/qwen3-vl-8b-hebrew-v24a-ckpt" != dv.CKPT_REPO
    assert dv.WARM_STEP == d4.ARM.max_steps == 2000                                           # the last step of the v2.4a run
    assert re.fullmatch(r"[0-9a-f]{40}", dv.WARM_REVISION) and dv.WARM_REVISION.startswith("cafd125e")
    assert dv.NEW_MODEL_TITLE in c3 and dv.NEW_WARM in c3 and dv.NEW_WARM_DOWNLOAD in c3
    assert f'WARM_REVISION = "{dv.WARM_REVISION}"' in c3 and "\nWARM_STEP = 2000\n" in c3
    for gone in ("v21b-ckpt", "6724c32c", "v2.1b step 1200", "flagship warm start", "never promoted"):
        assert gone not in c3, gone
    assert "# weights-only warm start (fresh optimizer + schedule)" in c3
    assert 'r=16, lora_alpha=16, lora_dropout=0.0, bias="none", random_state=3407,' in c3 and 'MERGER_MODE = "frozen"' in c3
    assert 'assert len(_wv) == 108, f"warm start vision adapter not loaded' in c3            # the load gates still follow
    assert 'assert len(WARM_REVISION) == 40, "invalid revision SHA"' in c3
    assert _src(v23q, 3).count("WARM_CKPT_REPO") < c3.count("WARM_CKPT_REPO")                 # the step check adds uses


@pytest.mark.skipif(not CHAIN_LOG.is_file(), reason="Studio log of the v2.4a eval chain not on this machine")
def test_warm_start_commit_is_the_one_the_eval_chain_staged() -> None:
    """The sha is the commit the chain staged for the step-2000 evaluation (the one the v2.4a scores belong to)."""
    log = CHAIN_LOG.read_text(encoding="utf-8", errors="replace")
    assert f"stage v24a step {dv.WARM_STEP} ({dv.WARM_REVISION})" in log


def test_warm_start_step_check_runs_as_written(tmp_path: Path) -> None:
    """The step check passes on a step-2000 trainer state and fails with a clear message on another step."""
    (tmp_path / "last-checkpoint").mkdir()
    check = dv.NEW_WARM_DOWNLOAD.split("\n", 2)[2]                                            # the lines after the download call
    ns = {"json": json, "local": str(tmp_path), "WARM_STEP": 2000, "WARM_CKPT_REPO": "r", "WARM_REVISION": "cafd125e60d0"}
    state = tmp_path / "last-checkpoint" / "trainer_state.json"
    state.write_text(json.dumps({"global_step": 2000}))
    exec(compile(check, "<warm step>", "exec"), dict(ns))
    state.write_text(json.dumps({"global_step": 1000}))
    with pytest.raises(AssertionError, match="is step 1000, expected 2000"):
        exec(compile(check, "<warm step>", "exec"), dict(ns))


# ----------------------------------------------------------------------------- schedule, names, gates


def test_schedule_steps_seed_and_gates_are_v23qs(editions: Editions) -> None:
    """Steps, learning rate, warm-up, cosine, seeds, batch and eval interval are v23q's in both editions."""
    v23q, colab, lam = editions
    c7, c8 = _src(colab, 7), _src(colab, 8)
    assert dq.MAX_STEPS == 1000 == dq.TRAIN_ROWS // 8 and "\nMAX_STEPS = 1000\n" in c7 and "\nORDER_SEED = 3407\n" in c7
    assert "\nMAX_INSESSION_RETRIES = 2\n" in c7 and 'if kind != "cosine":' in c7              # the resume gate only knows cosine
    for needle in ("max_steps=MAX_STEPS,               # 8,000 rows = one pass = 1,000 steps", "learning_rate=1e-4,",
                   'warmup_ratio=0.05, lr_scheduler_type="cosine", weight_decay=0.01,',
                   'optim="adamw_8bit", seed=3407, data_seed=3407, output_dir=OUT_DIR,',
                   "per_device_train_batch_size=1, gradient_accumulation_steps=8,", "save_steps=50, save_total_limit=2,",
                   'eval_strategy="steps", eval_steps=100,', 'accelerator_config={"dispatch_batches": False},',
                   'dataset_kwargs={"skip_prepare_dataset": True},'):
        assert needle in c8, needle
    for same in ("max_steps=MAX_STEPS,", "learning_rate=1e-4,", "warmup_ratio=0.05,", "seed=3407, data_seed=3407,", "eval_steps=100,"):
        assert (same in _src(v23q, 8)) and (same in _src(lam, 8)), same
    assert dv.warmup_steps() == 50 == math.ceil(1000 * 0.05)


def test_save_steps_are_50_and_checkpoints_private(editions: Editions) -> None:
    """Saves every 50 steps (v23q: 100), pushed to a private checkpoint repository (v23q: public)."""
    v23q, colab, _ = editions
    c8 = _src(colab, 8)
    assert "save_steps=50, save_total_limit=2,   # a termination costs at most 50 steps" in c8 and "save_steps=100" not in c8
    assert 'hub_strategy="checkpoint", hub_private_repo=True,' in c8 and "hub_private_repo=False" not in c8 and "Muharaf" in c8
    assert "hub_always_push=True" in c8 and "save_steps=100, save_total_limit=2," in _src(v23q, 8)


def test_names_repositories_and_gates(editions: Editions) -> None:
    """New run, checkpoint, output, resume and merged names; every gate and the exact-resume machinery are still in the cell."""
    v23q, colab, lam = editions
    c8, c9 = _src(colab, 8), _src(colab, 9)
    assert dv.CKPT_REPO == "isaacmg/qwen3-vl-8b-hebrew-v24q-ckpt" != dq.CKPT_REPO != dv.WARM_CKPT_REPO
    assert dv.MERGED_REPO == "isaacmg/qwen3-vl-8b-hebrew-v24q-merged" and dv.RUN_NAME == "genizah_v24q"
    for needle in (f'CKPT_REPO = "{dv.CKPT_REPO}"', 'OUT_DIR = "outputs_v24q"', 'RESUME_ROOT = "/content/v24q_resume"',
                   'run_name="genizah_v24q"', 'wandb.init(project="qwen-hebrew-finetune", name="genizah_v24q", ',
                   "# Cell 6 — v2.4q (page parse + questions, warm start v2.4a step 2000) training:",
                   "# ---- DATALOADER GATE", "prepared-dataloader loss", "resume_dir, _resume_step = discover_resume(CKPT_REPO, OUT_DIR, RESUME_ROOT)",
                   "_order = lock_session_plan(CKPT_REPO, _plan, _order, has_checkpoint=resume_dir is not None)",
                   'warm_start=f"{WARM_CKPT_REPO}@{WARM_REVISION}"',                         # the schedule lock records the warm start
                   "TimeBudgetTrainerCallback(budget_s=TIME_BUDGET_H * 3600)", "ensure_hub_checkpoint(CKPT_REPO, OUT_DIR, _final_step)"):
        assert needle in c8, needle
    for gone in ("v23q-ckpt", "outputs_v23q", "v23q_resume", 'name="genizah_v23q"', "v21b-ckpt"):
        assert gone not in c8, gone
    assert f'MERGED_REPO = "{dv.MERGED_REPO}"' in c9 and c9.count('"v24q-merged"') == 2 and "v23q" not in c9 and "private=True" in c9
    assert c8 == _src(lam, 8).replace('RESUME_ROOT = os.path.expanduser("~/genizah_work/v24q_resume")', 'RESUME_ROOT = "/content/v24q_resume"') \
        .replace('import os\nfrom getpass import getpass\nwandb.login(key=os.environ.get("WANDB_API_KEY") or getpass("W&B API key: "))',
                 'wandb.login(key=userdata.get("WANDB_API_KEY"))')


def test_nothing_names_the_v23q_run_or_the_v21b_warm_start(editions: Editions) -> None:
    """In code, 'v23q' survives only as the dataset's name (the same mixture); the older run's names and warm start are gone."""
    _, colab, lam = editions
    for nb in (colab, lam):
        code = "\n".join(hl.code_cells(nb))
        assert "v23q" not in code.replace("genizah_v23q_vqa", "").replace("# v23q page-parse mixture", "")
        for stale in ("v21b-ckpt", "6724c32c", "outputs_v23q", "v23q_resume", "v23q-merged", "hub_private_repo=False", "save_steps=100",
                      "flagship warm start", "v2.3q"):
            assert stale not in code, stale


def test_session_budgets(editions: Editions) -> None:
    """The Colab edition keeps v23q's budget; the Lambda edition's safety stop is 1.3 x one measured pass, rounded up."""
    _, colab, lam = editions
    assert "\nTIME_BUDGET_H = 21.5\n" in _src(colab, 7) and d22.TIME_BUDGET_H == 21.5            # the Colab edition keeps v23q's budget
    one_pass_h = dq.MAX_STEPS * d4.MEASURED_S_PER_STEP / 3600
    assert dv.lambda_budget_h() == 23.0 and 1.3 * one_pass_h <= dv.lambda_budget_h() < 1.3 * one_pass_h + 1
    assert "\nTIME_BUDGET_H = 23.0   # Lambda, one session: 1,000 steps x ~63.6 s" in _src(lam, 7)
    assert f"TIME_BUDGET_H = {hl.LAMBDA_TIME_BUDGET_H}   # Lambda: no session cap" not in _src(lam, 7)
    out, new = _changed(colab, lam, 7)
    assert len(out) == len(new) == 1 and out[0] == "TIME_BUDGET_H = 21.5" and new[0].startswith("TIME_BUDGET_H = 23.0")


# ----------------------------------------------------------------------------- Lambda edition


def test_lambda_edition_has_no_colab_lines_and_a_wrap_up(editions: Editions) -> None:
    """Paths under the work directory, secrets from the environment, kernel gate instead of %pip, a wrap-up cell, nothing Colab-only."""
    _, colab, lam = editions
    c2, c8, c10 = _src(lam, 2), _src(lam, 8), _src(lam, 10)
    assert 'LOCAL = os.path.expanduser("~/genizah_work/genizah_v23q_vqa")' in c2 and f'V22_REVISION = "{V23Q_REVISION}"' in c2
    assert 'RESUME_ROOT = os.path.expanduser("~/genizah_work/v24q_resume")' in c8
    assert 'wandb.alert(title="genizah_v24q: notebook finished"' in c10 and "TERMINATE THE INSTANCE" in c10
    text = json.dumps(lam, ensure_ascii=False).lower()
    assert not any(bad in text for bad in ("google.colab", "/content", "userdata"))
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", json.dumps(lam))
    assert "%pip install" in _src(colab, 1) and "%pip install" not in _src(lam, 1) and "LAMBDA_KERNEL_PINS" in _src(lam, 1)


# ----------------------------------------------------------------------------- text


def test_intro_describes_this_run(editions: Editions) -> None:
    """The intro names the warm start, the repositories, the schedule and the resume rule; the Lambda notes come first."""
    _, colab, lam = editions
    intro = _src(colab, 0)
    for needle in ("# Genizah v2.4q", dv.DATA_REPO, dv.CKPT_REPO, dv.MERGED_REPO, "8,000 / 1,000", "v2.4a step 2000", "cafd125e",
                   "50 steps", "private", "safety stop 23 h", "run every cell again", "genizah_v24q", "logs/next_round/vqa/baseline_v24a.sh"):
        assert needle in intro, needle
    lam_intro = _src(lam, 0)
    assert lam_intro.startswith("# Genizah v2.4q") and "**Lambda Cloud edition**" in lam_intro.splitlines()[0]
    assert "genizah_v24q_lambda.ipynb" in lam_intro and "bash lambda_setup.sh" in lam_intro and lam_intro.endswith(intro)


def test_no_restricted_attribution(editions: Editions) -> None:
    """No attribution or link the project never publishes appears in either edition."""
    for nb in editions[1:]:
        text = json.dumps(nb, ensure_ascii=False).lower()
        assert not any(bad in text for bad in ("friedberg", "fjms", "fjp", "genizah.org"))


# ----------------------------------------------------------------------------- derivation guards and the CLI


@pytest.mark.parametrize("revision", ["PIN-AFTER-PUSH", "main", "b" * 39, "B" * 40, "g" * 40, ""])
def test_derive_refuses_anything_but_a_40_hex_sha(revision: str) -> None:
    """The mixture is already pushed: a placeholder or a branch name must never reach the notebook."""
    with pytest.raises(ValueError, match="40-hex"):
        dv.derive(json.load(open(SRC, encoding="utf-8")), revision)


def test_derive_fails_loudly_when_an_anchor_of_the_source_is_missing() -> None:
    """Each v24q-specific edit is asserted: a source notebook without its anchor stops the derivation."""
    nb = json.load(open(SRC, encoding="utf-8"))
    for cell, old in ((3, d4.OLD_WARM), (8, dv.OLD_SAVE), (8, dv.OLD_PRIVATE), (9, '"v22b-merged"')):
        broken = copy.deepcopy(nb)
        text = "".join(broken["cells"][cell]["source"])
        assert old in text, (cell, old)
        broken["cells"][cell]["source"] = text.replace(old, "# removed").splitlines(keepends=True)
        with pytest.raises(ValueError):
            dv.derive(broken)


def test_cli_writes_both_editions(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Both editions and the setup script by default, one edition per --host, the pinned revision by default."""
    dv.main(["--revision", SHA, "--out-dir", str(tmp_path)])
    colab = json.loads((tmp_path / "genizah_v24q.ipynb").read_text(encoding="utf-8"))
    lam = json.loads((tmp_path / "genizah_v24q_lambda.ipynb").read_text(encoding="utf-8"))
    assert (tmp_path / "lambda_setup.sh").read_text() == hl.setup_script() and "wrote" in capsys.readouterr().out
    assert f'V22_REVISION = "{SHA}"' in _src(colab, 2) and "%pip install" in _src(colab, 1) and "TIME_BUDGET_H = 23.0" in _src(lam, 7)
    dv.main(["--out-dir", str(tmp_path)])                                                        # default revision = the one v23q pins
    assert f'V22_REVISION = "{V23Q_REVISION}"' in _src(json.loads((tmp_path / "genizah_v24q.ipynb").read_text(encoding="utf-8")), 2)
    assert "lambda_setup.sh" not in capsys.readouterr().out                                      # unchanged: not rewritten
    (tmp_path / "c").mkdir()
    dv.main(["--host", "colab", "--out-dir", str(tmp_path / "c")])
    assert (tmp_path / "c" / "genizah_v24q.ipynb").exists() and not (tmp_path / "c" / "genizah_v24q_lambda.ipynb").exists()
    (tmp_path / "l").mkdir()
    dv.main(["--host", "lambda", "--out-dir", str(tmp_path / "l")])
    assert (tmp_path / "l" / "genizah_v24q_lambda.ipynb").exists() and not (tmp_path / "l" / "genizah_v24q.ipynb").exists()
    assert (COLAB_DIR / "lambda_setup.sh").read_text(encoding="utf-8") == hl.setup_script()    # the committed script is the current one


@pytest.mark.parametrize("revision", ["not-a-sha", "PIN-AFTER-PUSH"])
def test_cli_refuses_a_bad_revision(tmp_path: Path, revision: str) -> None:
    """A bad revision exits before anything is written."""
    with pytest.raises(SystemExit):
        dv.main(["--revision", revision, "--out-dir", str(tmp_path)])
    assert not list(tmp_path.iterdir())
