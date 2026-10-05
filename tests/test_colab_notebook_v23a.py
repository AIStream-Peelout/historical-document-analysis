# File name: test_colab_notebook_v23a.py
# Date: 10/5/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Tests for the v23a (Arabic script) notebook derivation.

Every intended delta lands, the cells compile, nothing of the v22b run is left where it would
matter (repositories, run name, gates for rows this mixture does not have), and the Arabic data
gate accepts a sound mixture and stops on each kind of bad row.
"""
import ast
import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from src.finetuning.qwen_hebrew.colab import derive_v23a as dv

SRC = Path("src/finetuning/qwen_hebrew/colab/genizah_v22b.ipynb")
SHA = "a" * 40
ARABIC_PROMPT = "Transcribe the handwritten Arabic script on this fragment."
ARABIC_PAGE = "كتابي اليك اطال الله بقاك وادام عزك وتاييدك وسعادتك\nوسلامتك من مصر لخمس خلون من شهر رمضان"
HEBREW_PAGE = "כתאבי אליך אטאל אללה בקאך ואדאם עזך\nשלום רב [...] לאהובי"


@pytest.fixture(scope="module")
def derived():
    nb = json.load(open(SRC, encoding="utf-8"))
    return nb, dv.derive(nb, SHA)


def _src(nb, i):
    return "".join(nb["cells"][i]["source"])


def test_cells_compile_and_count_is_unchanged(derived):
    a, b = derived
    assert len(b["cells"]) == len(a["cells"])
    for c in b["cells"]:
        if c["cell_type"] == "code":
            ast.parse("\n".join(l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith(("%", "!"))))


def test_data_cell_points_at_the_arabic_mixture(derived):
    _, b = derived
    c2 = _src(b, 2)
    assert f'V22_REPO = "{dv.DATA_REPO}"' in c2 and f'V22_REVISION = "{SHA}"' in c2
    assert 'LOCAL = "/content/genizah_v23a_arabic"' in c2 and "genizah_v22_full" not in c2
    assert f"len(train_ds) >= {dv.MIN_TRAIN_ROWS} and len(eval_ds) >= {dv.MIN_VAL_ROWS}" in c2
    assert f"for _s, _lo in {dv.SHARE_FLOORS}:" in c2
    for gone in ('"pgp_qa", 0.10', "_loc = ", "no QA rows in the mixture", "bad documentary box", "read_box prompt drift"):
        assert gone not in c2, gone
    for kept in ('assert set(_src) <= set(eval_ds.column("source")), "val must cover every train source"',
                 '"internal gap token leaked"', '"mojibake leaked"', '"expected damage-gap markers"'):
        assert kept in c2, kept
    assert dv.NEW_HYGIENE in c2


def test_share_floors_name_real_components_and_leave_room():
    from src.finetuning.qwen_hebrew.build_v22_mixture import COMPONENTS
    floors = dict(ast.literal_eval(dv.SHARE_FLOORS))
    assert set(floors) <= set(COMPONENTS) and 0.5 < sum(floors.values()) < 0.95
    assert dv.MAX_STEPS * 8 == dv.TRAIN_ROWS and dv.MIN_TRAIN_ROWS < dv.TRAIN_ROWS


def _run_gate(rows):
    """Execute the Arabic hygiene block on ``(answer, question, source, task)`` rows."""
    ns = {"_ans": [r[0] for r in rows], "_q": [r[1] for r in rows], "_srcs": [r[2] for r in rows], "_t": [r[3] for r in rows],
          "_transc": [r[0] for r in rows if r[3].endswith("transcribe")], "Counter": Counter}
    exec(compile(dv.NEW_HYGIENE, "<arabic hygiene>", "exec"), ns)


GOOD = [(ARABIC_PAGE, ARABIC_PROMPT, "arabic_editions", "fragment_transcribe"),
        (ARABIC_PAGE, ARABIC_PROMPT, "arabic_muharaf", "fragment_transcribe"),
        (HEBREW_PAGE, "Transcribe the handwritten Hebrew script.", "pgp_edition_pages", "fragment_transcribe"),
        (HEBREW_PAGE, "Transcribe the handwritten Hebrew script.", "ktiv_transcription", "fragment_transcribe")]


def test_arabic_gate_accepts_a_sound_mixture(capsys):
    _run_gate(GOOD)
    assert "hygiene OK" in capsys.readouterr().out


@pytest.mark.parametrize("bad, message", [
    ((HEBREW_PAGE, ARABIC_PROMPT, "arabic_agapet", "fragment_transcribe"), "not mainly Arabic script"),
    (("كتابي اليك", ARABIC_PROMPT, "arabic_agapet", "fragment_transcribe"), "not mainly Arabic script"),          # too short
    ((ARABIC_PAGE, "Transcribe the handwritten Hebrew script.", "arabic_baybars", "fragment_transcribe"), "does not name Arabic script"),
    ((ARABIC_PAGE.replace("كتابي", "كِتابي"), ARABIC_PROMPT, "arabic_muharaf", "fragment_transcribe"), "vowel marks or tatweel"),
    ((ARABIC_PAGE.replace("الله", "اللـه"), ARABIC_PROMPT, "arabic_iskandar", "fragment_transcribe"), "vowel marks or tatweel"),
    (('{"bbox_2d": [1, 2, 3, 4]}', "Locate the phrase, 0-1000.", "ktiv_grounding", "locate"), "box or QA rows"),
    (('{"answer": "not stated"}', "When was it written? JSON", "pgp_qa", "qa_abstain"), "box or QA rows"),
])
def test_arabic_gate_stops_on_each_bad_row(bad, message):
    with pytest.raises(AssertionError, match=message):
        _run_gate(GOOD + [bad])


def test_arabic_gate_allows_vowel_marks_in_geniza_editions_only():
    _run_gate(GOOD + [(ARABIC_PAGE.replace("كتابي", "كِتابي"), ARABIC_PROMPT, "arabic_editions", "fragment_transcribe")])


def test_arabic_gate_needs_arabic_rows_and_hebrew_replay():
    with pytest.raises(AssertionError, match="no Arabic-script rows"):
        _run_gate(GOOD[2:])
    with pytest.raises(AssertionError, match="replay rows are not Hebrew script"):
        _run_gate(GOOD[:2] + [(ARABIC_PAGE, "Transcribe.", "pgp_edition_pages", "fragment_transcribe")])


def test_collator_guardrail_uses_an_arabic_page(derived):
    _, b = derived
    c4 = _src(b, 4)
    assert 's.startswith("arabic_")' in c4 and '"pgp_qa"' not in c4 and "Hebrew page + Arabic page" in c4


def test_training_cells_deltas(derived):
    _, b = derived
    c7, c8, c9 = _src(b, 7), _src(b, 8), _src(b, 9)
    assert f"\nMAX_STEPS = {dv.MAX_STEPS}\n" in c7 and "MAX_STEPS = 2500" not in c7
    for needle in (f'CKPT_REPO = "{dv.CKPT_REPO}"', 'OUT_DIR = "outputs_v23a"', 'RESUME_ROOT = "/content/v23a_resume"',
                   "max_steps=MAX_STEPS", f"learning_rate={dv.LR},", f"warmup_ratio={dv.WARMUP_RATIO},",
                   f'run_name="{dv.RUN_NAME}"', f'wandb.init(project="qwen-hebrew-finetune", name="{dv.RUN_NAME}", ',
                   "hub_always_push=True", "resume_dir, _resume_step = discover_resume(CKPT_REPO, OUT_DIR, RESUME_ROOT)",
                   "_order = lock_session_plan(CKPT_REPO, _plan, _order, has_checkpoint=resume_dir is not None)",
                   "TimeBudgetTrainerCallback(budget_s=TIME_BUDGET_H * 3600)", "ensure_hub_checkpoint(CKPT_REPO, OUT_DIR, _final_step)"):
        assert needle in c8, needle
    for gone in ("v22b-ckpt", "outputs_v22b", "v22b_resume", 'name="genizah_v22b"', "learning_rate=3e-5", "warmup_ratio=0.02",
                 "hub_private_repo=False"):
        assert gone not in c8, gone
    assert 'hub_strategy="checkpoint", hub_private_repo=True,' in c8                 # NC-SA data in the mixture
    assert f'MERGED_REPO = "{dv.MERGED_REPO}"' in c9 and "v22a-merged" not in c9 and "v22b-merged" not in c9
    assert c9.count('"v23a-merged"') == 2


def test_warm_start_and_resolution_contract_are_untouched(derived):
    a, b = derived
    assert _src(a, 3) == _src(b, 3) and _src(a, 5) == _src(b, 5) and _src(a, 1) == _src(b, 1)


def test_intro_describes_this_run(derived):
    _, b = derived
    intro = _src(b, 0)
    assert "v2.3a" in intro and dv.DATA_REPO in intro and dv.CKPT_REPO in intro and "run every cell again" in intro


def test_no_restricted_attribution_anywhere_in_the_notebook(derived):
    _, b = derived
    text = json.dumps(b, ensure_ascii=False).lower()
    assert not any(bad in text for bad in ("friedberg", "fjms", "fjp"))


def test_source_notebook_is_not_modified(derived):
    a, _ = derived
    assert a == json.load(open(SRC, encoding="utf-8"))
    assert dv.derive(copy.deepcopy(a), SHA) == dv.derive(a, SHA)


def test_bad_revision_rejected(tmp_path):
    with pytest.raises(SystemExit):
        dv.main(["--revision", "not-a-sha", "--out", str(tmp_path / "x.ipynb")])
    dv.main(["--revision", "PIN-AFTER-PUSH", "--out", str(tmp_path / "y.ipynb")])
    assert json.load(open(tmp_path / "y.ipynb"))["cells"]


# ----------------------------------------------------------------------------- the control arm

@pytest.fixture(scope="module")
def control():
    nb = json.load(open(SRC, encoding="utf-8"))
    return dv.derive(nb, SHA, dv.CONTROL)


def test_control_reads_the_variant_rows_of_the_same_revision(control, derived):
    _, run = derived
    c2 = _src(control, 2)
    assert f'train_ds = ImagesOnceDataset(DATA_DIR / "rows/{dv.CONTROL_TRAIN_FILE}", DATA_DIR / "images")' in c2
    assert 'DATA_DIR / "rows/train.parquet"' not in c2
    assert 'eval_ds = ImagesOnceDataset(DATA_DIR / "rows/val.parquet", DATA_DIR / "images")' in c2     # the run's val rows
    assert f'V22_REPO = "{dv.DATA_REPO}"' in c2 and f'V22_REVISION = "{SHA}"' in c2
    assert f"len(train_ds) >= {dv.CONTROL.min_train_rows} and len(eval_ds) >= {dv.MIN_VAL_ROWS}" in c2
    assert f"for _s, _lo in {dv.CONTROL_SHARE_FLOORS}:" in c2 and dv.SHARE_FLOORS not in c2
    assert dv.NEW_HYGIENE + dv.CONTROL_GATE in c2 and dv.CONTROL_GATE not in _src(run, 2)
    assert '"rows/*"' in c2                                           # the download pattern covers the variant file


def test_control_differs_from_the_run_only_where_planned(control, derived):
    _, run = derived
    for i in (1, 3, 4, 5, 6):
        assert _src(control, i) == _src(run, i), i
    c7, c8, c9 = _src(control, 7), _src(control, 8), _src(control, 9)
    assert dv.CONTROL.max_steps == 870 and f"\nMAX_STEPS = {dv.CONTROL.max_steps}\n" in c7
    assert c7.replace("MAX_STEPS = 870", f"MAX_STEPS = {dv.MAX_STEPS}") == _src(run, 7)
    for needle in ('CKPT_REPO = "isaacmg/qwen3-vl-8b-hebrew-v23actl-ckpt"', 'OUT_DIR = "outputs_v23actl"',
                   'RESUME_ROOT = "/content/v23actl_resume"', 'run_name="genizah_v23a_ctl"',
                   'wandb.init(project="qwen-hebrew-finetune", name="genizah_v23a_ctl", ',
                   f"learning_rate={dv.LR},", f"warmup_ratio={dv.WARMUP_RATIO},"):
        assert needle in c8, needle
    for gone in ("v23a-ckpt", "outputs_v23a\"", "v23a_resume", 'name="genizah_v23a"', "v22b-ckpt", "outputs_v22b", "v22b_resume"):
        assert gone not in c8, gone
    assert "hub_private_repo=True," in c8 and "private=True" in c9
    assert 'MERGED_REPO = "isaacmg/qwen3-vl-8b-hebrew-v23actl-merged"' in c9 and c9.count('"v23actl-merged"') == 2
    assert '"v23a-merged"' not in c9
    for c in control["cells"]:
        if c["cell_type"] == "code":
            ast.parse("\n".join(l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith(("%", "!"))))
    intro = _src(control, 0)
    assert "control" in intro and dv.CONTROL_TRAIN_FILE in intro and dv.CONTROL.ckpt_repo in intro and "6,960 / 870" in intro


def test_control_rows_and_floors_follow_the_runs_plan():
    floors = dict(ast.literal_eval(dv.CONTROL_SHARE_FLOORS))
    assert set(floors) == {"arabic_editions", "pgp_edition_pages", "ktiv_transcription"} and 0.7 < sum(floors.values()) < 0.95
    assert not set(dv.OUTSIDE_COMPONENTS) & set(floors)
    assert set(dv.OUTSIDE_COMPONENTS) < set(dict(ast.literal_eval(dv.SHARE_FLOORS)))
    assert dv.CONTROL.train_rows == round(dv.MAIN.train_rows * (0.11 + 0.19 + 0.28))     # SHARES in build_v23a.sh
    assert dv.MAIN.ckpt_repo == dv.CKPT_REPO == "isaacmg/qwen3-vl-8b-hebrew-v23a-ckpt" and not dv.MAIN.is_control


def _run_control_gate(rows):
    ns = {"_ans": [r[0] for r in rows], "_q": [r[1] for r in rows], "_srcs": [r[2] for r in rows], "_t": [r[3] for r in rows],
          "_transc": [r[0] for r in rows if r[3].endswith("transcribe")], "Counter": Counter}
    exec(compile(dv.NEW_HYGIENE + dv.CONTROL_GATE, "<control hygiene>", "exec"), ns)


def test_control_gate_stops_on_an_outside_row():
    _run_control_gate([r for r in GOOD if r[2] not in dv.OUTSIDE_COMPONENTS])
    with pytest.raises(AssertionError, match="outside Arabic rows in the control run"):
        _run_control_gate(GOOD)


def test_cli_writes_each_arm_to_its_own_file(tmp_path, monkeypatch):
    monkeypatch.setattr(dv, "HERE", tmp_path)
    src = str(Path(SRC).resolve())
    dv.main(["--revision", SHA, "--src", src])
    dv.main(["--revision", SHA, "--src", src, "--control"])
    run, ctl = json.load(open(tmp_path / "genizah_v23a.ipynb")), json.load(open(tmp_path / "genizah_v23a_ctl.ipynb"))
    assert 'run_name="genizah_v23a"' in _src(run, 8) and 'run_name="genizah_v23a_ctl"' in _src(ctl, 8)


# ----------------------------------------------------------------------------- the rank-64 arm

@pytest.fixture(scope="module")
def r64():
    nb = json.load(open(SRC, encoding="utf-8"))
    return dv.derive(nb, SHA, dv.R64)


def test_r64_changes_the_adapter_and_the_names_only(r64, derived):
    _, run = derived
    for i in (1, 2, 4, 5, 6, 7):
        assert _src(r64, i) == _src(run, i), i                       # same data, gates, collator, steps
    c3, c8, c9 = _src(r64, 3), _src(r64, 8), _src(r64, 9)
    assert 'r=64, lora_alpha=64, lora_dropout=0.0, bias="none", random_state=3407,' in c3 and "r=16, lora_alpha=16" not in c3
    assert dv.rank_expansion_block(64) in c3 and c3.count("set_peft_model_state_dict(model, _sd)") == 1
    assert c3.index("_sd = expand_lora_state(_warm_sd, _fresh)") < c3.index("missing = set_peft_model_state_dict(model, _sd)")
    assert "def expand_lora_state(" in c3 and "rank expansion OK" in c3
    assert 'assert len(_wv) == 108, f"warm start vision adapter not loaded' in c3          # the old gates still follow
    assert "expand_lora_state" not in _src(run, 3) and 'r=16, lora_alpha=16' in _src(run, 3)
    for needle in ('CKPT_REPO = "isaacmg/qwen3-vl-8b-hebrew-v23ar64-ckpt"', 'OUT_DIR = "outputs_v23ar64"',
                   'RESUME_ROOT = "/content/v23ar64_resume"', 'run_name="genizah_v23a_r64"', "hub_private_repo=True,",
                   f"learning_rate={dv.LR},"):
        assert needle in c8, needle
    assert 'MERGED_REPO = "isaacmg/qwen3-vl-8b-hebrew-v23ar64-merged"' in c9 and c9.count('"v23ar64-merged"') == 2
    for c in r64["cells"]:
        if c["cell_type"] == "code":
            ast.parse("\n".join(l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith(("%", "!"))))
    intro = _src(r64, 0)
    assert "rank 64" in intro and dv.R64.ckpt_repo in intro and "12,000 / 1,500" in intro
    assert dv.R64.train_rows == dv.MAIN.train_rows and not dv.R64.is_control and dv.ARMS == {"run": dv.MAIN, "control": dv.CONTROL, "r64": dv.R64}


def _tiny_adapter(rank, seed):
    """A small real PEFT model: fixed base weights, a LoRA adapter of the given rank at scale 1."""
    import torch
    from peft import LoraConfig, get_peft_model
    from torch import nn
    torch.manual_seed(0)
    base = nn.Sequential(nn.Linear(16, 24), nn.Tanh(), nn.Linear(24, 5))
    torch.manual_seed(seed)
    return get_peft_model(base, LoraConfig(r=rank, lora_alpha=rank, target_modules=["0", "2"], lora_dropout=0.0))


def _warm_state():
    import torch
    from peft import get_peft_model_state_dict
    small = _tiny_adapter(dv.WARM_RANK, seed=1)
    torch.manual_seed(2)
    for name, p in small.named_parameters():
        if "lora_B" in name:
            p.data.normal_(0, 0.3)
    return {k: v.detach().clone() for k, v in get_peft_model_state_dict(small).items()}, small


def test_rank_expansion_block_runs_as_written_and_keeps_the_function(capsys):
    """The model-cell code is executed on a real adapter: rank 16 into rank 64, outputs unchanged."""
    import torch
    from peft import set_peft_model_state_dict
    warm, small = _warm_state()
    big = _tiny_adapter(64, seed=7)
    ns = {"model": big, "_sd": warm, "torch": torch, "set_peft_model_state_dict": set_peft_model_state_dict}
    exec(compile(dv.rank_expansion_block(64), "<rank expansion>", "exec"), ns)
    out = capsys.readouterr().out
    assert "rank expansion 16 -> 64: 4 LoRA tensors widened, 0 adapter tensors left at their fresh init" in out
    assert "rank expansion OK" in out
    x = torch.randn(5, 16)
    with torch.no_grad():
        assert torch.allclose(big(x), small(x), atol=1e-6)


def test_rank_expansion_block_stops_when_the_load_is_wrong():
    import torch
    from peft import set_peft_model_state_dict
    warm, _ = _warm_state()

    def noisy_new_directions(model, state):
        return set_peft_model_state_dict(model, {k: (v + 0.01 if "lora_B" in k else v) for k, v in state.items()})

    def scrambled_rows(model, state):
        return set_peft_model_state_dict(model, {k: (v.flip(0) if "lora_A" in k else v) for k, v in state.items()})

    for loader, message in ((scrambled_rows, "rank expansion changed the adapter"), (noisy_new_directions, "rank expansion changed the adapter|not zero")):
        ns = {"model": _tiny_adapter(64, seed=7), "_sd": warm, "torch": torch, "set_peft_model_state_dict": loader}
        with pytest.raises(AssertionError, match=message):
            exec(compile(dv.rank_expansion_block(64), "<rank expansion>", "exec"), ns)


def test_cli_writes_all_three_arms(tmp_path, monkeypatch):
    monkeypatch.setattr(dv, "HERE", tmp_path)
    dv.main(["--revision", SHA, "--src", str(Path(SRC).resolve()), "--arm", "all"])
    names = {a.notebook: a.run_name for a in dv.ARMS.values()}
    for file_name, run_name in names.items():
        assert f'run_name="{run_name}"' in _src(json.load(open(tmp_path / file_name)), 8)
    with pytest.raises(SystemExit):
        dv.main(["--revision", SHA, "--arm", "all", "--out", str(tmp_path / "x.ipynb")])
