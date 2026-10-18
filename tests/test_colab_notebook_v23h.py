# File name: test_colab_notebook_v23h.py
# Date: 10/12/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for the v23h (Hebrew-only ablation of v2.3a) notebook derivation.

The Lambda edition must differ from v2.3a's Lambda edition only where planned (data, gates, guardrail row, names,
steps, session budget), keep v2.3a's schedule, seed, saves and gates, and its Hebrew-only data gate must accept a
sound mixture and stop on each kind of bad row.
"""
import ast
import difflib
import json
from collections import Counter
from pathlib import Path

import pytest

from src.finetuning.qwen_hebrew import subset_mixture
from src.finetuning.qwen_hebrew.colab import derive_v23a as dv
from src.finetuning.qwen_hebrew.colab import derive_v23h as dh
from src.finetuning.qwen_hebrew.colab import host_lambda as hl

SRC = Path("src/finetuning/qwen_hebrew/colab/genizah_v22b.ipynb")
SHA = "b" * 40
HEB_Q = "Transcribe the handwritten Hebrew/Judaeo-Arabic text on this fragment."
HEB = "כתאבי אליך אטאל אללה בקאך ואדאם עזך\nשלום רב [...] לאהובי"
ARABIC = "كتابي اليك اطال الله بقاك وادام عزك وتاييدك وسعادتك\nوسلامتك من مصر لخمس خلون من شهر رمضان"


@pytest.fixture(scope="module")
def pair():
    nb = json.load(open(SRC, encoding="utf-8"))
    v23a = hl.to_lambda(dv.derive(nb, dh.SOURCE_REVISION, dv.MAIN), "genizah_v23a_lambda.ipynb")
    return v23a, dh.to_lambda(dh.derive(nb, SHA), "genizah_v23h.ipynb")


def _src(nb, i):
    return "".join(nb["cells"][i]["source"])


def _changed(a, b, i):
    """(lines only in v23a, lines only in v23h) of cell ``i``."""
    d = [l for l in difflib.unified_diff(_src(a, i).splitlines(), _src(b, i).splitlines(), lineterm="", n=0)
         if not l.startswith(("---", "+++", "@@"))]
    return [l[1:] for l in d if l.startswith("-")], [l[1:] for l in d if l.startswith("+")]


def test_committed_v23a_lambda_edition_is_the_baseline(pair):
    v23a, _ = pair
    assert v23a == json.load(open("src/finetuning/qwen_hebrew/colab/genizah_v23a_lambda.ipynb", encoding="utf-8"))


def test_only_the_planned_cells_change(pair):
    v23a, v23h = pair
    assert len(v23h["cells"]) == len(v23a["cells"]) == 11 and v23h["metadata"] == v23a["metadata"]
    for i in (1, 3, 5, 6):                                       # install gate, model + warm start, grad gate, bench
        assert _src(v23h, i) == _src(v23a, i), i
    for c in v23h["cells"]:
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))


def test_data_cell(pair):
    v23a, v23h = pair
    c2 = _src(v23h, 2)
    assert f'V22_REPO = "{dh.DATA_REPO}"' in c2 and f'V22_REVISION = "{SHA}"' in c2 and dv.DATA_REPO not in c2
    assert f'LOCAL = os.path.expanduser("~/genizah_work/{dh.LOCAL_LEAF}")' in c2
    assert f"len(train_ds) >= {dh.ARM.min_train_rows} and len(eval_ds) >= {dh.MIN_VAL_ROWS}," in c2 and dh.ARM.min_train_rows == 5358
    assert dh.MIN_VAL_ROWS <= dh.VAL_ROWS == 80 < dv.MIN_VAL_ROWS                  # v2.3a's gate of 100 would refuse the 80 Hebrew val rows
    assert f"for _s, _lo in {dh.SHARE_FLOORS}:" in c2 and dh.HEB_HYGIENE in c2 and dv.NEW_HYGIENE not in c2
    assert 'train_ds = ImagesOnceDataset(DATA_DIR / "rows/train.parquet", DATA_DIR / "images")' in c2
    for kept in ('"internal gap token leaked"', '"mojibake leaked"', '"expected damage-gap markers"',
                 'assert set(_src) <= set(eval_ds.column("source")), "val must cover every train source"',
                 'allow_patterns=["rows/*", "*.json", "README.md", "images*.tar"]'):
        assert kept in c2, kept
    out, new = _changed(v23a, v23h, 2)
    anchors = ("V22_REPO = ", "V22_REVISION = ", "LOCAL = ", "assert len(train_ds) >= ", "for _s, _lo in ")
    assert set(out) <= set(dv.NEW_HYGIENE.splitlines()) | {l for l in _src(v23a, 2).splitlines() if l.startswith(anchors)}
    assert set(new) <= set(dh.HEB_HYGIENE.splitlines()) | {l for l in c2.splitlines() if l.startswith(anchors)}
    assert len([l for l in new if l.startswith(anchors)]) == 5


def test_share_floors_fit_the_built_shares():
    floors = dict(ast.literal_eval(dh.SHARE_FLOORS))
    realised = {"pgp_edition_pages": 2280 / 5640, "ktiv_transcription": 3360 / 5640}
    assert set(floors) == set(realised) and all(floors[c] < realised[c] for c in floors) and 0.7 < sum(floors.values()) < 0.95
    assert dh.ARM.train_rows == 5640 and dh.ARM.max_steps == 705 == subset_mixture.steps_for(5640)
    assert dh.ARM.train_rows == round(dv.MAIN.train_rows * (0.19 + 0.28))     # SHARES in build_v23a.sh, Hebrew components


def _run_gate(rows):
    """Execute the Hebrew-only hygiene block on ``(answer, question, source, task)`` rows."""
    ns = {"_ans": [r[0] for r in rows], "_q": [r[1] for r in rows], "_srcs": [r[2] for r in rows], "_t": [r[3] for r in rows],
          "_transc": [r[0] for r in rows if r[3].endswith("transcribe")], "Counter": Counter}
    exec(compile(dh.HEB_HYGIENE, "<hebrew-only hygiene>", "exec"), ns)


GOOD = [(HEB, HEB_Q, "pgp_edition_pages", "fragment_transcribe"),
        (HEB + " كتابي", HEB_Q, "pgp_edition_pages", "fragment_transcribe"),        # a mixed-script page as in v2.3a's replay
        (HEB, HEB_Q, "ktiv_transcription", "line_transcribe")]


def test_hebrew_gate_accepts_a_sound_mixture(capsys):
    _run_gate(GOOD)
    assert "hygiene OK" in capsys.readouterr().out


@pytest.mark.parametrize("bad, message", [
    ((ARABIC, "Transcribe the handwritten Arabic script on this page.", "arabic_muharaf", "fragment_transcribe"), "Arabic-source rows"),
    ((HEB, "Transcribe the handwritten Arabic script on this page.", "pgp_edition_pages", "fragment_transcribe"), "names Arabic script"),
    ((ARABIC, HEB_Q, "pgp_edition_pages", "fragment_transcribe"), "mainly Arabic script"),
    (('{"bbox_2d": [1, 2, 3, 4]}', "Locate the phrase, 0-1000.", "ktiv_grounding", "locate"), "box or QA rows"),
    (('{"answer": "not stated"}', "When was it written? JSON", "pgp_qa", "qa_abstain"), "box or QA rows"),
])
def test_hebrew_gate_stops_on_each_bad_row(bad, message):
    with pytest.raises(AssertionError, match=message):
        _run_gate(GOOD + [bad])


def test_hebrew_gate_stops_when_the_rows_are_not_hebrew_overall():
    with pytest.raises(AssertionError, match="not Hebrew script"):
        _run_gate([("שלום " + "كتابي اليك " * 3, HEB_Q, "pgp_edition_pages", "fragment_transcribe")] * 3)


def test_guardrail_uses_a_documentary_page(pair):
    v23a, v23h = pair
    c4 = _src(v23h, 4)
    assert 'if s == "pgp_edition_pages")' in c4 and 's.startswith("arabic_")' not in c4 and "KTIV page + documentary page" in c4
    out, new = _changed(v23a, v23h, 4)
    assert len(out) == len(new) == 3


def test_steps_budget_and_training_settings(pair):
    v23a, v23h = pair
    c7, c8 = _src(v23h, 7), _src(v23h, 8)
    out, new = _changed(v23a, v23h, 7)
    assert sorted(new) == sorted([l for l in c7.splitlines() if l.startswith(("MAX_STEPS = ", "TIME_BUDGET_H = "))])
    assert "\nMAX_STEPS = 705\n" in c7 and f"\nTIME_BUDGET_H = {dh.LAMBDA_TIME_BUDGET_H}   #" in c7 and len(out) == 2
    assert 705 * dh.MEASURED_S_PER_STEP / 3600 < dh.LAMBDA_TIME_BUDGET_H < 2 * 705 * dh.MEASURED_S_PER_STEP / 3600   # one session, a stop not a cap
    assert "ORDER_SEED = 3407" in c7
    for needle in ('CKPT_REPO = "isaacmg/qwen3-vl-8b-hebrew-v23h-ckpt"', 'OUT_DIR = "outputs_v23h"',
                   'RESUME_ROOT = os.path.expanduser("~/genizah_work/v23h_resume")', 'run_name="genizah_v23h"',
                   'wandb.init(project="qwen-hebrew-finetune", name="genizah_v23h", ', f"learning_rate={dv.LR},",
                   f'warmup_ratio={dv.WARMUP_RATIO}, lr_scheduler_type="cosine", weight_decay=0.01,',
                   "per_device_train_batch_size=1, gradient_accumulation_steps=8,", "save_steps=50, save_total_limit=2,",
                   'eval_strategy="steps", eval_steps=100,', 'optim="adamw_8bit", seed=3407, data_seed=3407,',
                   'hub_strategy="checkpoint", hub_private_repo=True,', 'accelerator_config={"dispatch_batches": False},',
                   "# ---- DATALOADER GATE", "prepared-dataloader loss", "lock_session_plan(CKPT_REPO, _plan, _order",
                   "TimeBudgetTrainerCallback(budget_s=TIME_BUDGET_H * 3600)", "ensure_hub_checkpoint(CKPT_REPO, OUT_DIR, _final_step)"):
        assert needle in c8, needle
    for gone in ("v23a-ckpt", "outputs_v23a", "v23a_resume", 'name="genizah_v23a"', "Muharaf"):
        assert gone not in c8, gone
    out, new = _changed(v23a, v23h, 8)
    assert len(out) == len(new) == 9                             # names, title, max_steps comment, two comments


def test_merged_cell_and_wrapup(pair):
    _, v23h = pair
    c9, c10 = _src(v23h, 9), _src(v23h, 10)
    assert 'MERGED_REPO = "isaacmg/qwen3-vl-8b-hebrew-v23h-merged"' in c9 and c9.count('"v23h-merged"') == 2 and "private=True" in c9
    assert 'wandb.alert(title="genizah_v23h: notebook finished"' in c10 and "TERMINATE THE INSTANCE" in c10


def test_intro_and_launch_notes(pair):
    _, v23h = pair
    intro = _src(v23h, 0)
    assert intro.startswith("# Genizah v2.3h") and "**Lambda Cloud edition**" in intro.splitlines()[0]
    for needle in (dh.DATA_REPO, dh.ARM.ckpt_repo, "5,640 / 705", "0.153", "0.160", "genizah_v23h.ipynb", "bash lambda_setup.sh",
                   "300 / 550 / 705", "run every cell again"):
        assert needle in intro, needle


def test_no_restricted_attribution_and_no_colab_lines(pair):
    _, v23h = pair
    text = json.dumps(v23h, ensure_ascii=False).lower()
    assert not any(bad in text for bad in ("friedberg", "fjms", "fjp", "google.colab", "/content", "userdata"))


def test_cli(tmp_path):
    out = tmp_path / "genizah_v23h.ipynb"
    dh.main(["--revision", SHA, "--out", str(out)])
    nb = json.loads(out.read_text(encoding="utf-8"))
    assert (tmp_path / "lambda_setup.sh").read_text() == hl.setup_script() and f'V22_REVISION = "{SHA}"' in _src(nb, 2)
    dh.main(["--revision", "PIN-AFTER-PUSH", "--host", "colab", "--out", str(tmp_path / "colab.ipynb")])
    colab = json.loads((tmp_path / "colab.ipynb").read_text(encoding="utf-8"))
    assert "TIME_BUDGET_H = 7.5" in _src(colab, 7) and "%pip install" in _src(colab, 1)
    with pytest.raises(SystemExit):
        dh.main(["--revision", "not-a-sha", "--out", str(tmp_path / "x.ipynb")])
