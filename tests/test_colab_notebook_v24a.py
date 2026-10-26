# File name: test_colab_notebook_v24a.py
# Date: 10/15/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for the v24a (second Arabic-script run) notebook derivation and the round's build checks.

Both editions must differ from v2.3a's only where planned (data repository and pin guard, warm start v2.3a step 1500,
steps, warm-up, names, session budget), keep v2.3a's stack and gates, refuse an unpinned dataset, and plan the mixture
exactly as the build script does: Arabic editions at the quota, outside sets at v2.3a's share, replay the remainder.
The build checks refuse an Arabic editions export not built from the pinned PGP export and train rows of registered
benchmark documents; the round's runbook (``logs/next_round/arabic_v24a``, git-ignored: those tests skip without it)
must give commands the real command-line interfaces accept, with paths that agree from step to step.
"""
import ast
import copy
import difflib
import importlib.util
import json
import math
import re
import shlex
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.datasets.indexing import arabic_scrape_queue as asq
from src.finetuning.qwen_hebrew import build_pgp_arabic_editions as bpae
from src.finetuning.qwen_hebrew.build_v22_mixture import COMPONENTS, plan_quotas
from src.finetuning.qwen_hebrew.colab import derive_v23a as dv
from src.finetuning.qwen_hebrew.colab import derive_v24a as d4
from src.finetuning.qwen_hebrew.colab import host_lambda as hl
from src.finetuning.qwen_hebrew.train_session import cosine_warmup_lr, warmup_steps_for

SRC = Path("src/finetuning/qwen_hebrew/colab/genizah_v22b.ipynb")
COLAB_DIR = Path("src/finetuning/qwen_hebrew/colab")
ROUND_DIR = Path("logs/next_round/arabic_v24a")
README = ROUND_DIR / "README.md"
GATES = ROUND_DIR / "run_notebook_gates_v24a.py"
TEST_COMMIT = "b2343a17" + "0" * 32
SHA = "c" * 40
V23A_REVISION = "3b762524067c77141c121ce84a3a724991618757"          # the v2.3a dataset pin of the committed v2.3a editions
ARABIC_PROMPT = "Transcribe the handwritten Arabic script on this fragment."
ARABIC_PAGE = "كتابي اليك اطال الله بقاك وادام عزك وتاييدك وسعادتك\nوسلامتك من مصر لخمس خلون من شهر رمضان"
HEBREW_PAGE = "כתאבי אליך אטאל אללה בקאך ואדאם עזך\nשלום רב [...] לאהובי"


@pytest.fixture(scope="module")
def editions():
    """(v2.3a Lambda edition, v24a Colab edition, v24a Lambda edition)."""
    nb = json.load(open(SRC, encoding="utf-8"))
    v23a = hl.to_lambda(dv.derive(nb, V23A_REVISION, dv.MAIN), "genizah_v23a_lambda.ipynb")
    colab = d4.derive(nb, SHA)
    return v23a, colab, d4.to_lambda(colab, "genizah_v24a_lambda.ipynb")


def _src(nb: dict, i: int) -> str:
    """Source of cell ``i``.

    :param nb: Notebook.
    :param i: Cell index.
    :return: The joined source.
    """
    return "".join(nb["cells"][i]["source"])


def _changed(a, b, i):
    """(lines only in ``a``, lines only in ``b``) of cell ``i``."""
    diff = [l for l in difflib.unified_diff(_src(a, i).splitlines(), _src(b, i).splitlines(), lineterm="", n=0)
            if not l.startswith(("---", "+++", "@@"))]
    return [l[1:] for l in diff if l.startswith("-")], [l[1:] for l in diff if l.startswith("+")]


def _parses(nb: dict) -> None:
    """Every code cell parses (IPython magics and shell lines left out).

    :param nb: Notebook.
    """
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            ast.parse("\n".join(l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith(("%", "!"))))


# ----------------------------------------------------------------------------- structure


def test_committed_v23a_lambda_edition_is_the_baseline(editions):
    v23a, _, _ = editions
    assert v23a == json.load(open(COLAB_DIR / "genizah_v23a_lambda.ipynb", encoding="utf-8"))


def test_cells_compile_and_only_the_planned_cells_change(editions):
    v23a, colab, lam = editions
    assert len(colab["cells"]) == len(json.load(open(SRC, encoding="utf-8"))["cells"]) == 10
    assert len(lam["cells"]) == len(v23a["cells"]) == 11 and lam["metadata"] == v23a["metadata"]
    _parses(colab)
    _parses(lam)
    for i in (1, 4, 5, 6):                                       # install gate, collator + Arabic guardrail, grad gate, bench
        assert _src(lam, i) == _src(v23a, i), i


def test_source_notebook_is_not_modified_and_derivation_is_deterministic(editions):
    nb = json.load(open(SRC, encoding="utf-8"))
    assert d4.derive(copy.deepcopy(nb), SHA) == d4.derive(nb, SHA) == editions[1]
    assert nb == json.load(open(SRC, encoding="utf-8"))


# ----------------------------------------------------------------------------- data cell


def test_data_cell(editions):
    v23a, colab, lam = editions
    c2 = _src(lam, 2)
    assert f'V22_REPO = "{d4.DATA_REPO}"' in c2 and f'V22_REVISION = "{SHA}"' in c2 and dv.DATA_REPO not in c2
    assert f'LOCAL = os.path.expanduser("~/genizah_work/{d4.LOCAL_LEAF}")' in c2
    assert f'LOCAL = "/content/{d4.LOCAL_LEAF}"' in _src(colab, 2)
    assert f'len(train_ds) >= {d4.ARM.min_train_rows} and len(eval_ds) >= {dv.MIN_VAL_ROWS}, "Arabic mixture incomplete"' in c2
    assert d4.ARM.min_train_rows == 15200
    assert f"for _s, _lo in {d4.ARM.share_floors}:" in c2 and dv.SHARE_FLOORS not in c2
    assert dv.NEW_HYGIENE in c2                                  # v2.3a's Arabic-script / Hebrew-replay gates, unchanged
    assert 'train_ds = ImagesOnceDataset(DATA_DIR / "rows/train.parquet", DATA_DIR / "images")' in c2
    for kept in ('"internal gap token leaked"', '"mojibake leaked"', '"expected damage-gap markers"',
                 'assert set(_src) <= set(eval_ds.column("source")), "val must cover every train source"',
                 'allow_patterns=["rows/*", "*.json", "README.md", "images*.tar"]'):
        assert kept in c2, kept
    out, new = _changed(v23a, lam, 2)
    assert set(out) <= set(d4.OLD_UNPINNED.splitlines()) | {
        l for l in _src(v23a, 2).splitlines() if l.startswith(("# Cell 2", "V22_REPO = ", "V22_REVISION = ", "LOCAL = ", "assert len(train_ds) >= ", "for _s, _lo in "))}
    assert set(new) <= set(d4.PIN_GUARD.splitlines()) | {
        l for l in c2.splitlines() if l.startswith(("# Cell 2", "V22_REPO = ", "V22_REVISION = ", "LOCAL = ", "assert len(train_ds) >= ", "for _s, _lo in "))}


def _pin_block(nb):
    """The data cell's revision lines (assignment + guard), ready to exec."""
    c2 = _src(nb, 2)
    start = c2.index("V22_REVISION = ")
    return c2[start:c2.index("LOCAL = ")]


def test_notebook_refuses_the_placeholder_and_bad_pins(editions):
    _, _, lam = editions
    block = _pin_block(lam)
    assert "HfApi().dataset_info" not in _src(lam, 2) and d4.PIN_GUARD in block
    exec(compile(block, "<pin guard>", "exec"), {})                              # a 40-hex sha passes
    for bad in (d4.PLACEHOLDER, "c" * 39, "C" * 40, "main"):
        with pytest.raises(RuntimeError, match="not a pinned 40-hex sha"):
            exec(compile(block.replace(SHA, bad), "<pin guard>", "exec"), {})
    placeholder = d4.to_lambda(d4.derive(json.load(open(SRC, encoding="utf-8")), d4.PLACEHOLDER), "x.ipynb")
    with pytest.raises(RuntimeError, match="push_v24a.sh"):
        exec(compile(_pin_block(placeholder), "<pin guard>", "exec"), {})


def _run_gate(rows):
    """Execute the Arabic hygiene block of the data cell on ``(answer, question, source, task)`` rows."""
    ns = {"_ans": [r[0] for r in rows], "_q": [r[1] for r in rows], "_srcs": [r[2] for r in rows], "_t": [r[3] for r in rows],
          "_transc": [r[0] for r in rows if r[3].endswith("transcribe")], "Counter": Counter}
    exec(compile(dv.NEW_HYGIENE, "<arabic hygiene>", "exec"), ns)


def test_arabic_gate_still_accepts_a_sound_mixture_and_stops_on_a_bad_row(capsys):
    good = [(ARABIC_PAGE, ARABIC_PROMPT, "arabic_editions", "fragment_transcribe"),
            (ARABIC_PAGE, ARABIC_PROMPT, "arabic_muharaf", "fragment_transcribe"),
            (HEBREW_PAGE, "Transcribe the handwritten Hebrew script.", "pgp_edition_pages", "fragment_transcribe"),
            (HEBREW_PAGE, "Transcribe the handwritten Hebrew script.", "ktiv_transcription", "fragment_transcribe")]
    _run_gate(good)
    assert "hygiene OK" in capsys.readouterr().out
    with pytest.raises(AssertionError, match="not mainly Arabic script"):
        _run_gate(good + [(HEBREW_PAGE, ARABIC_PROMPT, "arabic_editions", "fragment_transcribe")])


# ----------------------------------------------------------------------------- the mixture plan


def test_plan_keeps_v23a_outside_shares_and_replay_ratio():
    shares, quotas = d4.planned_shares(), d4.planned_quotas()
    v23a_order = [c for c, _ in ast.literal_eval(dv.SHARE_FLOORS)]
    assert list(shares) == list(quotas) == list(d4.V23A_ROWS) == v23a_order       # v2.3a's key order
    assert set(shares) <= set(COMPONENTS) and abs(sum(shares.values()) - 1) < 1e-12
    assert d4.ARABIC_QUOTA == 2500 and quotas["arabic_editions"] == 2500 and sum(quotas.values()) == d4.TRAIN_ROWS == 16000
    assert {c: shares[c] for c in d4.OUTSIDE} == {"arabic_agapet": 960 / 12000, "arabic_muharaf": 2640 / 12000,
                                                   "arabic_baybars": 1080 / 12000, "arabic_iskandar": 360 / 12000}
    assert {c: quotas[c] for c in d4.OUTSIDE} == {"arabic_agapet": 1280, "arabic_muharaf": 3520, "arabic_baybars": 1440, "arabic_iskandar": 480}
    replay = quotas["pgp_edition_pages"] + quotas["ktiv_transcription"]
    assert replay == 16000 - 2500 - 6720 and abs(quotas["pgp_edition_pages"] - replay * 2280 / 5640) < 1
    assert shares["pgp_edition_pages"] / shares["ktiv_transcription"] == pytest.approx(2280 / 3360, rel=1e-12)
    assert plan_quotas(shares, d4.TRAIN_ROWS) == quotas                        # the builder rounds the same way


@pytest.mark.parametrize("quota", [1000, 1500, 2000, 2500, 3000, 4000, 6000])
def test_plan_holds_for_other_quotas(quota):
    quotas = d4.planned_quotas(quota)
    assert quotas["arabic_editions"] == quota and sum(quotas.values()) == 16000
    assert plan_quotas(d4.planned_shares(quota), 16000) == quotas
    assert d4.make_arm(quota).train_rows == 16000 and d4.make_arm(quota).max_steps == 2000


@pytest.mark.parametrize("quota", [0, -5, 9280, 12000])
def test_plan_rejects_a_quota_without_replay_rows(quota):
    with pytest.raises(ValueError):
        d4.planned_shares(quota)


def test_share_floors_sit_under_the_plan():
    shares = d4.planned_shares()
    floors = dict(ast.literal_eval(d4.ARM.share_floors))
    assert list(floors) == list(shares)
    assert all(0.65 * shares[c] <= floors[c] < shares[c] for c in floors), (floors, shares)
    assert 0.7 < sum(floors.values()) < 0.95
    assert d4.make_arm(4000).share_floors != d4.ARM.share_floors               # the floors follow the quota


def test_arabic_quota_is_read_back_from_a_build():
    mixture = json.loads(json.dumps({"config": {"train_rows": 16000, "shares": d4.planned_shares(3000)}}))
    assert d4.arabic_quota_of(mixture) == 3000
    with pytest.raises(ValueError, match="train rows"):
        d4.arabic_quota_of({"config": {"train_rows": 12000, "shares": d4.planned_shares(3000)}})
    v23a_shares = dict(zip(d4.V23A_ROWS, (r / 12000 for r in d4.V23A_ROWS.values())))
    assert d4.arabic_quota_of({"config": {"train_rows": 16000, "shares": v23a_shares}}) == 1760   # v2.3a's shares = the plan at 1,760
    moved = d4.planned_shares(3000)
    moved["arabic_muharaf"] -= 0.01
    moved["ktiv_transcription"] += 0.01                                       # an outside set off v2.3a's share
    with pytest.raises(ValueError, match="not the v24a plan"):
        d4.arabic_quota_of({"config": {"train_rows": 16000, "shares": moved}})
    reordered = dict(reversed(list(d4.planned_shares(3000).items())))
    with pytest.raises(ValueError, match="not the v24a plan"):
        d4.arabic_quota_of({"config": {"train_rows": 16000, "shares": reordered}})


# ----------------------------------------------------------------------------- warm start, schedule, names


def test_warm_start_is_v23a_step_1500(editions):
    v23a, _, lam = editions
    c3 = _src(lam, 3)
    assert f'WARM_CKPT_REPO = "{dv.MAIN.ckpt_repo}"' in c3 and dv.MAIN.ckpt_repo == "isaacmg/qwen3-vl-8b-hebrew-v23a-ckpt"
    assert 'WARM_REVISION = "b5503b186a61e640b95277c3886fd65e503d97d1"' in c3 and "\nWARM_STEP = 1500\n" in c3
    assert "v21b-ckpt" not in c3 and "6724c32c" not in c3 and "v2.1b step 1200" not in c3
    assert "# weights-only warm start (fresh optimizer + schedule)" in c3
    assert 'r=16, lora_alpha=16, lora_dropout=0.0, bias="none", random_state=3407,' in c3 and 'MERGER_MODE = "frozen"' in c3
    assert 'assert len(_wv) == 108, f"warm start vision adapter not loaded' in c3          # the load gates still follow
    assert d4.NEW_WARM in c3 and d4.NEW_WARM_DOWNLOAD in c3
    out, new = _changed(v23a, lam, 3)
    assert len(out) == 6 and len(new) == 11              # title, the 4-line pin -> 5 lines, the download + its 4-line step check


def test_warm_start_step_check_runs_as_written(tmp_path):
    (tmp_path / "last-checkpoint").mkdir()
    check = d4.NEW_WARM_DOWNLOAD.split("\n", 2)[2]                              # the lines after the download
    ns = {"json": json, "local": str(tmp_path), "WARM_STEP": 1500, "WARM_CKPT_REPO": "r", "WARM_REVISION": "b5503b18"}
    (tmp_path / "last-checkpoint" / "trainer_state.json").write_text(json.dumps({"global_step": 1500}))
    exec(compile(check, "<warm step>", "exec"), dict(ns))
    (tmp_path / "last-checkpoint" / "trainer_state.json").write_text(json.dumps({"global_step": 1400}))
    with pytest.raises(AssertionError, match="is step 1400, expected 1500"):
        exec(compile(check, "<warm step>", "exec"), dict(ns))


def test_schedule_steps_and_seed(editions):
    v23a, colab, lam = editions
    c7, c8 = _src(lam, 7), _src(lam, 8)
    assert d4.ARM.train_rows == 16000 and d4.ARM.max_steps == 2000 == d4.ARM.train_rows // 8
    assert "\nMAX_STEPS = 2000\n" in c7 and "\nORDER_SEED = 3407\n" in c7 and "MAX_STEPS = 1500" not in c7
    assert d4.warmup_steps() == 60 == warmup_steps_for(2000, float(d4.WARMUP_RATIO)) == math.ceil(2000 * 0.03)
    assert cosine_warmup_lr(60, 1e-4, 60, 2000) == pytest.approx(1e-4) and cosine_warmup_lr(0, 1e-4, 60, 2000) == 0.0
    for needle in ("max_steps=MAX_STEPS,               # 16,000 rows = one pass = 2,000 steps",
                   "learning_rate=1e-4,", 'warmup_ratio=0.03, lr_scheduler_type="cosine", weight_decay=0.01,   # 60 warm-up steps of 2,000',
                   'optim="adamw_8bit", seed=3407, data_seed=3407, output_dir=OUT_DIR,',
                   "per_device_train_batch_size=1, gradient_accumulation_steps=8,", "save_steps=50, save_total_limit=2,",
                   'eval_strategy="steps", eval_steps=100,', 'accelerator_config={"dispatch_batches": False},'):
        assert needle in c8, needle
    assert 'if kind != "cosine":' in c7                                         # the resume gate only knows cosine
    out, new = _changed(v23a, lam, 7)
    assert len(out) == len(new) == 2
    assert sorted(new) == sorted(l for l in c7.splitlines() if l.startswith(("MAX_STEPS = ", "TIME_BUDGET_H = ")))


def test_learning_rate_comes_from_lr(monkeypatch):
    """derive_v24a.LR reaches the training cell (and the intro); it is not inherited from derive_v23a."""
    nb = json.load(open(SRC, encoding="utf-8"))
    monkeypatch.setattr(d4, "LR", "5e-5")
    c8 = _src(d4.derive(nb, SHA), 8)
    assert len(re.findall(r"^\s*learning_rate=5e-5,", c8, re.M)) == 1 and "learning_rate=1e-4," not in c8
    assert "learning_rate=_a.learning_rate" in c8                               # the resume gate's line is left alone
    assert "| 5e-5, 3% warm-up (60 steps), fresh cosine |" in d4.intro_md()
    monkeypatch.setattr(d4, "LR", "fast")
    with pytest.raises(ValueError):
        d4.derive(nb, SHA)


def test_session_budgets(editions):
    _, colab, lam = editions
    assert f"\nTIME_BUDGET_H = {dv.TIME_BUDGET_H}   #" in _src(colab, 7) and dv.TIME_BUDGET_H == 7.5      # Colab's ~8 h lifetime
    one_pass_h = 2000 * d4.MEASURED_S_PER_STEP / 3600
    assert d4.lambda_budget_h() == 46.0 and 1.3 * one_pass_h <= d4.lambda_budget_h() < 1.3 * one_pass_h + 1
    assert "\nTIME_BUDGET_H = 46.0   # Lambda, one session: 2,000 steps x ~63.6 s" in _src(lam, 7)
    assert f"TIME_BUDGET_H = {hl.LAMBDA_TIME_BUDGET_H}   # Lambda: no session cap" not in _src(lam, 7)


def test_names_repositories_and_gates(editions):
    v23a, _, lam = editions
    c8, c9, c10 = _src(lam, 8), _src(lam, 9), _src(lam, 10)
    assert d4.ARM.ckpt_repo == "isaacmg/qwen3-vl-8b-hebrew-v24a-ckpt" != d4.WARM_CKPT_REPO and d4.ARM.ckpt_repo.endswith("-v24a-ckpt")
    for needle in ('CKPT_REPO = "isaacmg/qwen3-vl-8b-hebrew-v24a-ckpt"', 'OUT_DIR = "outputs_v24a"',
                   'RESUME_ROOT = os.path.expanduser("~/genizah_work/v24a_resume")', 'run_name="genizah_v24a"',
                   'wandb.init(project="qwen-hebrew-finetune", name="genizah_v24a", ',
                   'hub_strategy="checkpoint", hub_private_repo=True,', "hub_always_push=True",
                   "# ---- DATALOADER GATE", "prepared-dataloader loss", "resume_dir, _resume_step = discover_resume(CKPT_REPO, OUT_DIR, RESUME_ROOT)",
                   "_order = lock_session_plan(CKPT_REPO, _plan, _order, has_checkpoint=resume_dir is not None)",
                   "TimeBudgetTrainerCallback(budget_s=TIME_BUDGET_H * 3600)", "ensure_hub_checkpoint(CKPT_REPO, OUT_DIR, _final_step)",
                   "# Cell 6 — v2.4a (Arabic script, editions v2, warm start v2.3a step 1500) training:"):
        assert needle in c8, needle
    for gone in ("v23a-ckpt", "outputs_v23a", "v23a_resume", 'name="genizah_v23a"', "warmup_ratio=0.05", "v22b-ckpt"):
        assert gone not in c8, gone
    out, new = _changed(v23a, lam, 8)
    assert len(out) == len(new) == 9                     # title, repo, out dir, resume root, max_steps / lr / warm-up comments, run + W&B names
    assert 'MERGED_REPO = "isaacmg/qwen3-vl-8b-hebrew-v24a-merged"' in c9 and c9.count('"v24a-merged"') == 2 and "private=True" in c9
    assert 'wandb.alert(title="genizah_v24a: notebook finished"' in c10 and "TERMINATE THE INSTANCE" in c10


def test_collator_guardrail_uses_an_arabic_page(editions):
    _, _, lam = editions
    c4 = _src(lam, 4)
    assert 's.startswith("arabic_")' in c4 and "Hebrew page + Arabic page" in c4


# ----------------------------------------------------------------------------- text


def test_intro_describes_this_run(editions):
    _, colab, lam = editions
    intro = _src(colab, 0)
    for needle in ("# Genizah v2.4a", d4.DATA_REPO, d4.ARM.ckpt_repo, "16,000 / 2,000", "v2.3a step 1500", "b5503b18",
                   "2,500 rows", "60 steps", "safety stop 46 h", "refuses to run", "run every cell again", "genizah_v24a"):
        assert needle in intro, needle
    lam_intro = _src(lam, 0)
    assert lam_intro.startswith("# Genizah v2.4a") and "**Lambda Cloud edition**" in lam_intro.splitlines()[0]
    assert "genizah_v24a_lambda.ipynb" in lam_intro and "bash lambda_setup.sh" in lam_intro and lam_intro.endswith(intro)


def test_nothing_states_the_v23a_schedule(editions):
    """No line of either edition describes this run with v2.3a's rows, steps, session budget or data pin."""
    _, colab, lam = editions
    for nb in (colab, lam):
        text = json.dumps(nb, ensure_ascii=False)
        for stale in ("1,500 steps", "12,000 rows", "12,000 train rows", "12,000 / 1,500", "21.5 h", "MAX_STEPS = 1500",
                      "TIME_BUDGET_H = 21.5", "warmup_ratio=0.05", V23A_REVISION, "v2.3a (Arabic script) training"):
            assert stale not in text, stale
        code = "\n".join(hl.code_cells(nb))                       # the intro's comparison column may name v2.3a's data; code may not
        for stale in ("genizah_v23a_arabic", "v21b-ckpt", "outputs_v23a", "v23a_resume", "v23a-merged"):
            assert stale not in code, stale


def test_no_restricted_attribution_and_no_colab_lines(editions):
    _, colab, lam = editions
    assert not any(bad in json.dumps(colab, ensure_ascii=False).lower() for bad in ("friedberg", "fjms", "fjp", "genizah.org"))
    text = json.dumps(lam, ensure_ascii=False).lower()
    assert not any(bad in text for bad in ("friedberg", "fjms", "fjp", "genizah.org", "google.colab", "/content", "userdata"))
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", json.dumps(lam))


# ----------------------------------------------------------------------------- CLI and the committed files


def test_cli_writes_both_editions(tmp_path):
    d4.main(["--revision", SHA, "--out-dir", str(tmp_path)])
    colab = json.loads((tmp_path / "genizah_v24a.ipynb").read_text(encoding="utf-8"))
    lam = json.loads((tmp_path / "genizah_v24a_lambda.ipynb").read_text(encoding="utf-8"))
    assert (tmp_path / "lambda_setup.sh").read_text() == hl.setup_script()
    assert f'V22_REVISION = "{SHA}"' in _src(colab, 2) and "%pip install" in _src(colab, 1) and "TIME_BUDGET_H = 46.0" in _src(lam, 7)
    (tmp_path / "c").mkdir()
    d4.main(["--revision", d4.PLACEHOLDER, "--host", "colab", "--arabic-quota", "3000", "--out-dir", str(tmp_path / "c")])
    only = json.loads((tmp_path / "c" / "genizah_v24a.ipynb").read_text(encoding="utf-8"))
    assert not (tmp_path / "c" / "genizah_v24a_lambda.ipynb").exists()
    assert f"for _s, _lo in {d4.make_arm(3000).share_floors}:" in _src(only, 2) and "3,000 Geniza Arabic" in _src(only, 2)
    with pytest.raises(SystemExit):
        d4.main(["--revision", "not-a-sha", "--out-dir", str(tmp_path)])


def test_setup_script_is_rewritten_only_when_it_changes(tmp_path):
    path = tmp_path / "lambda_setup.sh"
    assert d4.write_if_changed(path, hl.setup_script()) and not d4.write_if_changed(path, hl.setup_script())
    assert (COLAB_DIR / "lambda_setup.sh").read_text(encoding="utf-8") == hl.setup_script()


@pytest.mark.parametrize("name, lambda_edition", [("genizah_v24a.ipynb", False), ("genizah_v24a_lambda.ipynb", True)])
def test_committed_notebooks_are_current(name, lambda_edition):
    """The generated files equal a fresh derivation at the revision and quota written into them."""
    committed = json.load(open(COLAB_DIR / name, encoding="utf-8"))
    m = re.search(r'^V22_REVISION = "([^"]+)".*\(16,000 train rows, ([\d,]+) Geniza Arabic\)$', _src(committed, 2), re.M)
    assert m, "revision line not found"
    revision, quota = m.group(1), int(m.group(2).replace(",", ""))
    fresh = d4.derive(json.load(open(SRC, encoding="utf-8")), revision, quota)
    if lambda_edition:
        fresh = d4.to_lambda(fresh, name, quota)
    assert committed == fresh


# ----------------------------------------------------------------------------- build checks: provenance


def _stats(edition_commit: str, queue_commit: str) -> dict:
    """An editions ``stats.json`` with the two export commits the builder records."""
    return {"inputs": {"queue_csv": "q.csv", "footnotes_csv": "f.csv", "pgp_export_commit": edition_commit,
                       "queue_pgp_export_commit": queue_commit, "queue_pgp_dir": "p"}}


def _pinned_export(tmp_path: Path, commit: str = TEST_COMMIT) -> Path:
    """A PGP export directory holding only its ``COMMIT`` file."""
    pgp = tmp_path / "pgp_export"
    pgp.mkdir()
    (pgp / "COMMIT").write_text(commit + "\n", encoding="utf-8")
    return pgp


def _editions(tmp_path: Path, stats: dict) -> Path:
    """An editions export directory holding only its ``stats.json``."""
    editions = tmp_path / "editions_images_once"
    editions.mkdir()
    (editions / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    return editions


def test_provenance_problems_name_each_unpinned_input():
    """An unpinned edition text or queue is a problem each, whatever the builder's own check said."""
    assert d4.editions_provenance_problems(_stats(TEST_COMMIT, TEST_COMMIT), TEST_COMMIT) == []
    raw_data = d4.editions_provenance_problems(_stats("", ""), TEST_COMMIT)      # unpinned export + old queue: '' == '' for the builder
    assert len(raw_data) == 2 and "edition text" in raw_data[0] and "Arabic queue" in raw_data[1]
    assert len(d4.editions_provenance_problems({"inputs": {"queue_csv": "q"}}, TEST_COMMIT)) == 2   # v1: no commit recorded
    assert len(d4.editions_provenance_problems({}, TEST_COMMIT)) == 2
    other = d4.editions_provenance_problems(_stats(TEST_COMMIT, "f" * 40), TEST_COMMIT)
    assert len(other) == 1 and "Arabic queue" in other[0] and "f" * 40 in other[0]


def test_check_editions_provenance(tmp_path):
    """The export passes only with the pinned commit for both inputs, and only against the spec's snapshot."""
    pgp = _pinned_export(tmp_path)
    assert d4.check_editions_provenance(_editions(tmp_path, _stats(TEST_COMMIT, TEST_COMMIT)), pgp) == TEST_COMMIT
    (tmp_path / "editions_images_once" / "stats.json").write_text(json.dumps(_stats("", "")), encoding="utf-8")
    with pytest.raises(ValueError, match="not built from the pinned PGP export.*--queue-csv"):
        d4.check_editions_provenance(tmp_path / "editions_images_once", pgp)
    (pgp / "COMMIT").write_text("a" * 40, encoding="utf-8")                     # another snapshot than the spec's
    with pytest.raises(ValueError, match="not the pinned PGP export b2343a17"):
        d4.check_editions_provenance(tmp_path / "editions_images_once", pgp)


@pytest.mark.skipif(not (d4.PINNED_PGP_DIR / "COMMIT").is_file(), reason="pinned PGP export not on this machine")
def test_pinned_export_is_the_spec_snapshot():
    """The snapshot on disk is b2343a17, read the same way the queue builder reads it."""
    commit = d4.pinned_pgp_commit()
    assert commit.startswith("b2343a17") and commit == asq.pgp_export_commit(d4.PINNED_PGP_DIR)


# ----------------------------------------------------------------------------- build checks: benchmark registries


ROWS = [{"stem": "pgpar_10_0_page", "source": "arabic_editions"}, {"stem": "pgp_20_1_page", "source": "pgp_edition_pages"},
        {"stem": "pgp_21_0_page", "source": "pgp_edition_pages"},
        {"stem": "ktiv_990001_FL202442468_left_column", "source": "ktiv_transcription"},
        {"stem": "muharaf_hm1b_04_08r", "source": "arabic_muharaf"}, {"stem": "agapet_sa418_0421", "source": "arabic_agapet"}]


def test_every_planned_component_is_checked_or_outside():
    """No planned component escapes the registry check unnoticed."""
    assert set(d4.GENIZA_STEMS) | set(d4.OUTSIDE) == set(d4.planned_shares()) and not set(d4.GENIZA_STEMS) & set(d4.OUTSIDE)


def test_registry_contamination_finds_documents_and_manuscripts():
    """Held-out PGP documents (both stem forms) and KTIV manuscripts are found; outside sets are skipped."""
    found = d4.registry_contamination(ROWS, {"10", "20", "99"}, {"990001"})
    assert found == {"arabic_editions": {"documents": 1, "held_out": ["10"]},
                     "pgp_edition_pages": {"documents": 2, "held_out": ["20"]},
                     "ktiv_transcription": {"documents": 1, "held_out": ["990001"]}}
    assert d4.check_train_registry(ROWS, set(), set()) == {"arabic_editions": 1, "pgp_edition_pages": 2, "ktiv_transcription": 1}
    with pytest.raises(ValueError, match="pgp_edition_pages 1, e.g. \\['21'\\]"):
        d4.check_train_registry(ROWS, {"21"}, set())
    with pytest.raises(ValueError, match="ktiv_transcription 1"):
        d4.check_train_registry(ROWS, set(), {"990001"})


def test_registry_check_refuses_rows_it_cannot_read():
    """An unplanned source or a stem off its pattern fails instead of passing unchecked."""
    with pytest.raises(ValueError, match="not a planned component"):
        d4.registry_contamination(ROWS + [{"stem": "pgp_5_0_qa", "source": "pgp_qa"}], set(), set())
    with pytest.raises(ValueError, match="does not match"):
        d4.registry_contamination([{"stem": "pgp_x_0_page", "source": "pgp_edition_pages"}], set(), set())
    with pytest.raises(ValueError, match="does not match"):
        d4.registry_contamination([{"stem": "pgp_7_0_page", "source": "arabic_editions"}], set(), set())   # pgpar_ expected


def test_registry_check_reads_every_registry_by_default(monkeypatch):
    """Without explicit sets the check loads the benchmark registries."""
    calls = []
    monkeypatch.setattr(d4, "registered_benchmark_documents", lambda: calls.append("docs") or (set(), {"20"}))
    monkeypatch.setattr(d4, "registered_benchmark_manuscripts", lambda: calls.append("mss") or set())
    with pytest.raises(ValueError, match="registered benchmark documents"):
        d4.check_train_registry(ROWS)
    assert calls == ["docs", "mss"]


# ----------------------------------------------------------------------------- the local gates runner (git-ignored)


def _load_gates():
    """``run_notebook_gates_v24a.py`` as a module (it lives under the git-ignored ``logs/``)."""
    spec = importlib.util.spec_from_file_location("run_notebook_gates_v24a", GATES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mixture_dir(tmp_path: Path, editions: Path) -> Path:
    """A built mixture with the v24a plan in ``mixture.json`` and a few rows (the gates' data cell is not run here)."""
    out = tmp_path / "mixture"
    (out / "rows").mkdir(parents=True)
    mixture = {"config": {"train_rows": d4.TRAIN_ROWS, "shares": d4.planned_shares(), "dirs": {"arabic_editions": str(editions)}},
               "components": {comp: {"taken": rows} for comp, rows in d4.planned_quotas().items()}}
    (out / "mixture.json").write_text(json.dumps(mixture), encoding="utf-8")
    table = pa.Table.from_pylist(ROWS)
    pq.write_table(table, out / "rows" / "train.parquet")
    pq.write_table(table.slice(0, 2), out / "rows" / "val.parquet")
    return out


@pytest.mark.skipif(not GATES.is_file(), reason="the round's gates runner is git-ignored and not on this machine")
def test_gates_runner_checks_provenance_and_registries(tmp_path, monkeypatch):
    """The gates runner repeats both build checks and records their results for the push."""
    gates = _load_gates()
    monkeypatch.setattr(d4, "PINNED_PGP_DIR", _pinned_export(tmp_path))
    monkeypatch.setattr(d4, "registered_benchmark_documents", lambda: (set(), {"99"}))
    monkeypatch.setattr(d4, "registered_benchmark_manuscripts", lambda: set())
    editions = _editions(tmp_path, _stats(TEST_COMMIT, TEST_COMMIT))
    data_dir = _mixture_dir(tmp_path, editions)
    plan = gates.check_plan(data_dir)
    assert plan == {"arabic_quota": d4.ARABIC_QUOTA, "pgp_export_commit": TEST_COMMIT,
                    "registry_documents_checked": {"arabic_editions": 1, "pgp_edition_pages": 2, "ktiv_transcription": 1}}
    monkeypatch.setattr(d4, "registered_benchmark_documents", lambda: (set(), {"21"}))
    with pytest.raises(ValueError, match="registered benchmark documents"):
        gates.check_plan(data_dir)
    (editions / "stats.json").write_text(json.dumps(_stats("", "")), encoding="utf-8")
    with pytest.raises(ValueError, match="not built from the pinned PGP export"):
        gates.check_plan(data_dir)


# ----------------------------------------------------------------------------- the round's runbook (git-ignored)


def _readme_commands() -> Tuple[List[Tuple[str, List[str]]], str]:
    """``python -m`` commands of the README's bash blocks, with the blocks' shell variables expanded.

    :return: ``([(module, argv)], README text)`` in README order.
    """
    text = README.read_text(encoding="utf-8")
    env: Dict[str, str] = {}

    def expand(s: str) -> str:
        """Expand the README's ``$NAME`` / ``${NAME}`` variables seen so far."""
        return re.sub(r"\$\{?([A-Z][A-Z0-9_]*)\}?", lambda m: env.get(m.group(1), m.group(0)), s)

    commands = []
    for block in re.findall(r"```bash\n(.*?)```", text, re.S):
        for line in block.replace("\\\n", " ").splitlines():
            assignment = re.match(r"^([A-Z][A-Z0-9_]*)=(\S+)", line)
            if assignment and " -m " not in line:
                env[assignment.group(1)] = expand(assignment.group(2))
                continue
            if " -m src." not in line:
                continue
            tokens = shlex.split(expand(line), comments=True)
            i = tokens.index("-m")
            commands.append((tokens[i + 1], tokens[i + 2:]))
    return commands, text


def _flags(argv: List[str]) -> Dict[str, object]:
    """``--flag value`` / ``--switch`` pairs of an argument list (switch -> True)."""
    out: Dict[str, object] = {}
    for i, token in enumerate(argv):
        if token.startswith("--"):
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            out[token] = True if nxt is None or nxt.startswith("--") else nxt
    return out


needs_readme = pytest.mark.skipif(not README.is_file(), reason="the round's README is git-ignored and not on this machine")


@needs_readme
def test_readme_builds_queue_and_editions_from_the_pinned_export():
    """Steps 0, 1 and 3 build from the pinned export into the round's queue, accepted by the real parser."""
    commands, text = _readme_commands()
    queue = [_flags(argv) for module, argv in commands if module == "src.datasets.indexing.arabic_scrape_queue"]
    builds = [bpae.build_parser().parse_args(argv) for module, argv in commands
              if module == "src.finetuning.qwen_hebrew.build_pgp_arabic_editions"]
    assert len(queue) == 1 and len(builds) == 2
    assert text.index("### 0.") < text.index("### 1.") < text.index("### 2.") < text.index("### 3.")
    pgp, out_dir = Path(queue[0]["--pgp-dir"]), Path(queue[0]["--out-dir"])
    assert pgp.resolve() == d4.PINNED_PGP_DIR.resolve() and queue[0].get("--verify-cudl") is True
    assert asq.out_dir_problem(pgp, out_dir) == "" and out_dir.resolve() == (ROUND_DIR / "queue").resolve()
    for args in builds:                                  # steps 1 and 3: the same export and queue, or every fingerprint goes stale
        assert args.pgp_dir == pgp and args.queue_csv == asq.output_paths(out_dir)[0]
        assert "pgp_arabic_editions_v2" in str(args.out) and "v1" not in str(args.out)
    assert builds[0].emit_unassigned is not None and builds[0].side_assignments is None
    assert builds[1].side_assignments is not None and builds[0].out == builds[1].out


@needs_readme
def test_readme_side_match_is_user_go_and_its_paths_meet_the_builder():
    """The paid side match is USER GO, estimates first, and reads / writes the builder's paths."""
    commands, text = _readme_commands()
    builds = [bpae.build_parser().parse_args(argv) for module, argv in commands
              if module == "src.finetuning.qwen_hebrew.build_pgp_arabic_editions"]
    matches = [_flags(argv) for module, argv in commands if module == "src.finetuning.qwen_hebrew.side_match"]
    step2 = text[text.index("### 2."):text.index("### 3.")]
    assert "USER GO" in step2.splitlines()[0]
    assert len(matches) == 2 and matches[0].get("--dry-run") is True and "--max-usd" in matches[1] and "--dry-run" not in matches[1]
    for flags in matches:
        assert Path(flags["--unassigned"]) == builds[0].emit_unassigned and Path(flags["--out"]) == builds[1].side_assignments


@needs_readme
def test_readme_removes_the_local_build_only_after_checking_the_nas_copy():
    """The local build is removed only behind both copy checks and the user's OK."""
    _, text = _readme_commands()
    step12 = text[text.index("### 12."):]
    rm_lines = [line for line in step12.splitlines() if "rm -rf" in line]
    assert len(rm_lines) == 1 and rm_lines[0].startswith('[ "$N" = 0 ] && [ "$OK" = True ] && rm -rf ')
    assert step12.index("rsync -a ") < step12.index("rsync -ain --checksum") < step12.index("rm -rf")
    assert "user's OK" in step12
