# File name: test_colab_notebook_v22b.py
# Date: 9/28/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Tests for the v22b notebook derivation and its Hub-side resume glue.

The derivation tests check that every intended delta lands, nothing else moves and the cells
compile. The glue tests run the helper cell's Hub functions against a fake Hub (a temp folder):
where the resume checkpoint comes from, the stored session plan, and the end-of-session check
that the Hub holds the step training stopped on.
"""
import ast
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from src.finetuning.qwen_hebrew import train_session as ts
from src.finetuning.qwen_hebrew.colab import derive_v22b as dv

SRC = Path("src/finetuning/qwen_hebrew/colab/genizah_v22a.ipynb")
OUT = Path("src/finetuning/qwen_hebrew/colab/genizah_v22b.ipynb")
SHA = "f" * 40
REPO = "isaacmg/qwen3-vl-8b-hebrew-v22b-ckpt"


@pytest.fixture(scope="module")
def derived():
    nb = json.load(open(SRC, encoding="utf-8"))
    return nb, dv.derive(nb, SHA)


def _src(nb, i):
    return "".join(nb["cells"][i]["source"])


def test_cell_count_and_compilation(derived):
    a, b = derived
    assert len(b["cells"]) == len(a["cells"]) + 1
    for i, c in enumerate(b["cells"]):
        if c["cell_type"] == "code":
            code = "\n".join(l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith(("%", "!")))
            ast.parse(code)  # every code cell is valid Python once IPython magics are dropped


def test_data_cell_points_at_the_full_mixture(derived):
    _, b = derived
    c2 = _src(b, 2)
    assert 'V22_REPO = "isaacmg/genizah_v22_full"' in c2
    assert f'V22_REVISION = "{SHA}"' in c2
    assert "len(train_ds) >= 19000" in c2 and "full mixture incomplete" in c2
    assert dv.SHARE_FLOORS in c2 and "isaacmg/genizah_v22_pilot" not in c2


def test_training_cell_deltas(derived):
    _, b = derived
    c8 = _src(b, 8)
    for needle in ('CKPT_REPO = "isaacmg/qwen3-vl-8b-hebrew-v22b-ckpt"', 'OUT_DIR = "outputs_v22b"', "max_steps=MAX_STEPS",
                   "learning_rate=3e-5", 'run_name="genizah_v22b"',
                   'wandb.init(project="qwen-hebrew-finetune", name="genizah_v22b", config={"resume_step": _resume_step, ',
                   "hub_always_push=True",
                   'RESUME_ROOT = "/content/v22b_resume"',
                   "resume_dir, _resume_step = discover_resume(CKPT_REPO, OUT_DIR, RESUME_ROOT)",
                   "materialize_order(len(train_ds), epochs_for(MAX_STEPS, len(train_ds), _a.per_device_train_batch_size, _a.gradient_accumulation_steps), seed=ORDER_SEED)",
                   "_order = lock_session_plan(CKPT_REPO, _plan, _order, has_checkpoint=resume_dir is not None)",
                   "dataset_repo=V22_REPO, dataset_revision=V22_REVISION", 'rows_fingerprint([train_ds.column(c) for c in ("image_sha1", "source", "task", "question", "answer")])',
                   "trainer.add_callback(_gate)", "TimeBudgetTrainerCallback(budget_s=TIME_BUDGET_H * 3600)",
                   "_final_step = train_with_exact_resume(trainer, train_ds, _order, resume_dir, _gate, max_retries=MAX_INSESSION_RETRIES)",
                   "ensure_hub_checkpoint(CKPT_REPO, OUT_DIR, _final_step)",
                   "train_dataset=train_ds, eval_dataset=eval_ds"):
        assert needle in c8, needle
    assert "wandb.config.update" not in c8 and "v22a" not in c8.replace("v22a lesson", "")


def test_the_pilots_resume_hazards_are_gone(derived):
    a, b = derived
    old, new = _src(a, 7), _src(b, 8)
    assert "local_dir=OUT_DIR" in old and "local_dir=OUT_DIR" not in new          # stale folder inside output_dir: Hub rollback
    assert "except Exception as e:" in old and "except Exception" not in new      # a Hub error read as "fresh start"
    assert "data order restarted" in old and "data order restarted" not in new    # reshuffle on resume
    assert "trainer.train(" not in new                                           # training only through train_with_exact_resume
    assert not dv.RESUME_ROOT.startswith("outputs_v22b") and "/outputs_v22b" not in dv.RESUME_ROOT


def test_training_cell_order_of_operations(derived):
    _, b = derived
    c8 = _src(b, 8)
    marks = ["discover_resume(", "trainer = make_trainer(", "wandb.init(", "dataloader gate OK", "trainer.train_dataset = train_ds\n",
             "lock_session_plan(", "trainer.add_callback(_gate)", "train_with_exact_resume(", "ensure_hub_checkpoint("]
    positions = [c8.index(m) for m in marks]
    assert positions == sorted(positions), dict(zip(marks, positions))


def test_intro_cell_describes_this_run_and_how_to_resume(derived):
    _, b = derived
    intro = _src(b, 0)
    assert b["cells"][0]["cell_type"] == "markdown" and intro.startswith("# Genizah v2.2b")
    shares = [float(m) for m in re.findall(r"^\| \w+ \| (0\.\d+) \|", intro, flags=re.M)]
    assert len(shares) == 7 and sum(shares) == pytest.approx(1.0)
    for needle in ("run the whole notebook again", "session_plan.json", "same step, same learning rate", "2500 steps", "21.5 h"):
        assert needle in intro, needle
    assert "genizah_v22_pilot" not in intro and "8,000 train rows" not in intro


def test_no_restricted_attribution_anywhere_in_the_notebook(derived):
    _, b = derived
    text = json.dumps(b, ensure_ascii=False)
    assert not re.search(r"friedberg|fjms|fjp", text, flags=re.I)


def test_written_notebook_is_the_current_derivation():
    written = json.load(open(OUT, encoding="utf-8"))
    revision = re.search(r'V22_REVISION = "([0-9a-f]{40})"', "".join(written["cells"][2]["source"])).group(1)
    fresh = dv.derive(json.load(open(SRC, encoding="utf-8")), revision)
    assert written == fresh, "genizah_v22b.ipynb is stale: re-run derive_v22b"


def test_pilot_notebook_untouched(derived):
    a, b = derived
    fresh = json.load(open(SRC, encoding="utf-8"))
    assert fresh == a  # derive() works on a copy


def test_bad_revision_rejected(tmp_path):
    with pytest.raises(SystemExit):
        dv.main(["--revision", "not-a-sha", "--out", str(tmp_path / "x.ipynb")])


# ---- the helper cell, executed as the notebook executes it -----------------------------------------------------

@pytest.fixture(scope="module")
def ns() -> Dict[str, object]:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    space: Dict[str, object] = {}
    exec(compile(dv.session_helper_source(), "<v22b helper cell>", "exec"), space)
    return space


def test_helper_cell_matches_the_module(ns):
    assert ns["materialize_order"](12, 2, 5) == ts.materialize_order(12, 2, 5)
    assert ns["remaining_rows"]([1, 2, 3, 4, 5, 6, 7, 8, 9], 1, 1, 8) == [9]
    assert ns["epochs_for"](2500, 20000, 1, 8) == 1
    assert (ns["MAX_STEPS"], ns["TIME_BUDGET_H"], ns["MAX_INSESSION_RETRIES"], ns["ORDER_SEED"]) == (2500, 21.5, 2, 3407)
    assert ns["cosine_warmup_lr"](1300, 3e-5, 50, 2500) == ts.cosine_warmup_lr(1300, 3e-5, 50, 2500)
    assert ns["REQUIRED_CHECKPOINT_FILES"] == ts.REQUIRED_CHECKPOINT_FILES and ns["PLAN_FILE"] == "session_plan.json"
    for name in ("build_session_plan", "diff_session_plan", "checkpoint_step", "newest_complete_checkpoint", "resume_problems",
                 "may_retry", "rows_fingerprint", "discover_resume", "lock_session_plan", "ensure_hub_checkpoint",
                 "train_with_exact_resume", "order_gate", "ResumeGateTrainerCallback", "TimeBudgetTrainerCallback"):
        assert callable(ns[name]), name
    assert issubclass(ns["ResumeGateError"], AssertionError)       # never caught by the in-session retry
    view = ns["OrderedView"]([10, 11, 12], [2, 0])
    assert len(view) == 2 and view[0] == 12 and view[1] == 10


class FakeHub:
    """A checkpoint repository backed by a folder, with the five Hub calls the glue makes."""

    def __init__(self, root: Path) -> None:
        self.root, self.exists, self.commits, self.fail_with = root, False, [], None

    def _files(self) -> List[str]:
        return sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file())

    def list_repo_files(self, repo_id: str) -> List[str]:
        from huggingface_hub.utils import RepositoryNotFoundError
        if self.fail_with is not None:
            raise self.fail_with
        if not self.exists:
            raise RepositoryNotFoundError(f"404: {repo_id}")
        return self._files()

    def snapshot_download(self, repo_id: str, allow_patterns: str, local_dir: str) -> str:
        prefix = allow_patterns.rstrip("*")
        for rel in self._files():
            if rel.startswith(prefix):
                dst = Path(local_dir) / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(self.root / rel, dst)
        return local_dir

    def hf_hub_download(self, repo_id: str, filename: str, force_download: bool = False) -> str:
        from huggingface_hub.utils import EntryNotFoundError
        if not (self.root / filename).is_file():
            raise EntryNotFoundError(f"404: {filename}")
        return str(self.root / filename)

    def upload_folder(self, repo_id: str, folder_path: str, path_in_repo: str, commit_message: str) -> None:
        self.commits.append(commit_message)
        shutil.copytree(folder_path, self.root / path_in_repo, dirs_exist_ok=True)

    def upload_file(self, path_or_fileobj: bytes, path_in_repo: str, repo_id: str, commit_message: str) -> None:
        self.commits.append(commit_message)
        (self.root / path_in_repo).write_bytes(path_or_fileobj)


@pytest.fixture()
def hub(tmp_path, monkeypatch) -> FakeHub:
    import huggingface_hub
    fake = FakeHub(tmp_path / "hub")
    fake.root.mkdir()
    for name in ("list_repo_files", "snapshot_download", "hf_hub_download", "upload_folder"):
        monkeypatch.setattr(huggingface_hub, name, getattr(fake, name))
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: fake)
    return fake


def _checkpoint(root: Path, name: str, step: int, skip=()) -> str:
    d = root / name
    d.mkdir(parents=True)
    for fname, text in {"trainer_state.json": '{"global_step": %d}' % step, "optimizer.pt": "o", "scheduler.pt": "s",
                        "adapter_model.safetensors": "w"}.items():
        if fname not in skip:
            (d / fname).write_text(text, encoding="utf-8")
    return str(d)


def test_discover_resume_fresh_start_only_when_the_repo_does_not_exist(ns, hub, tmp_path):
    out, root = str(tmp_path / "outputs_v22b"), str(tmp_path / "v22b_resume")
    assert ns["discover_resume"](REPO, out, root) == (None, 0)             # no repository yet
    hub.exists = True
    assert ns["discover_resume"](REPO, out, root) == (None, 0)             # repository, no checkpoint (died before step 100)
    hub.fail_with = ConnectionError("Hub unreachable")
    with pytest.raises(ConnectionError):                                    # NOT a fresh start: the pilot swallowed this
        ns["discover_resume"](REPO, out, root)


def test_discover_resume_downloads_outside_output_dir_and_prefers_the_newest_complete(ns, hub, tmp_path):
    out, root = tmp_path / "outputs_v22b", tmp_path / "v22b_resume"
    hub.exists = True
    _checkpoint(hub.root, "last-checkpoint", 1300)
    path, step = ns["discover_resume"](REPO, str(out), str(root))
    assert (path, step) == (str(root / "last-checkpoint"), 1300)
    assert not (out / "last-checkpoint").exists()                           # nothing for the model push to re-upload
    newer = _checkpoint(out, "checkpoint-1340", 1340)                        # same machine, kernel restarted after a later save
    _checkpoint(out, "checkpoint-1400", 1400, skip=("scheduler.pt",))        # half-written when it died
    assert ns["discover_resume"](REPO, str(out), str(root)) == (newer, 1340)
    with pytest.raises(ns["ResumeGateError"], match="must not be inside output_dir"):
        ns["discover_resume"](REPO, str(out), str(out / "resume"))


def test_discover_resume_refuses_an_incomplete_hub_checkpoint(ns, hub, tmp_path):
    hub.exists = True
    _checkpoint(hub.root, "last-checkpoint", 1300, skip=("optimizer.pt",))
    with pytest.raises(ns["ResumeGateError"], match="no complete copy"):
        ns["discover_resume"](REPO, str(tmp_path / "outputs_v22b"), str(tmp_path / "v22b_resume"))


def _plan(order: List[int], **over) -> Dict[str, object]:
    base = dict(order_seed=3407, n_rows=20, max_steps=2500, per_device_batch=1, grad_accum=8, learning_rate=3e-5,
                warmup_ratio=0.02, lr_scheduler_type="cosine", weight_decay=0.01, optim="adamw_8bit",
                dataset_repo="isaacmg/genizah_v22_full", dataset_revision="9" * 40, warm_start="repo@" + "6" * 40, rows_sha256="r" * 64)
    base.update(over)
    return ts.build_session_plan(order, **base)


def test_session_plan_is_stored_once_and_must_match_afterwards(ns, hub, tmp_path):
    hub.exists = True
    order = ts.materialize_order(20, 1, seed=3407)
    assert ns["lock_session_plan"](REPO, _plan(order), order, has_checkpoint=False) == order
    stored = json.loads((hub.root / "session_plan.json").read_text())
    assert stored["order"] == order and stored["max_steps"] == 2500 and stored["learning_rate"] == 3e-5 and len(hub.commits) == 1
    assert ns["lock_session_plan"](REPO, _plan(order), order, has_checkpoint=True) == order      # next session: verified, not rewritten
    assert len(hub.commits) == 1
    for change, needle in ((dict(learning_rate=5e-5), "learning_rate: stored 3e-05, now 5e-05"), (dict(max_steps=2000), "max_steps"),
                           (dict(grad_accum=4), "grad_accum"), (dict(dataset_revision="8" * 40), "dataset_revision"),
                           (dict(rows_sha256="x" * 64), "rows_sha256"), (dict(warmup_ratio=0.05), "warmup_ratio")):
        with pytest.raises(ns["ResumeGateError"], match=needle):
            ns["lock_session_plan"](REPO, _plan(order, **change), order, has_checkpoint=True)


def test_session_plan_keeps_the_stored_order_and_refuses_a_foreign_checkpoint(ns, hub, tmp_path):
    hub.exists = True
    order = ts.materialize_order(20, 1, seed=3407)
    with pytest.raises(ns["ResumeGateError"], match="was not started by this notebook"):
        ns["lock_session_plan"](REPO, _plan(order), order, has_checkpoint=True)                  # checkpoint, but no plan
    ns["lock_session_plan"](REPO, _plan(order), order, has_checkpoint=False)
    drifted = order[::-1]                                                                        # a library that permutes differently
    assert ns["lock_session_plan"](REPO, _plan(drifted), drifted, has_checkpoint=True) == order  # the run keeps ITS order
    stored = json.loads((hub.root / "session_plan.json").read_text())
    stored["order"][0], stored["order"][1] = stored["order"][1], stored["order"][0]
    (hub.root / "session_plan.json").write_text(json.dumps(stored))
    with pytest.raises(ns["ResumeGateError"], match="does not match its own fingerprint"):
        ns["lock_session_plan"](REPO, _plan(order), order, has_checkpoint=True)


def test_session_ends_only_when_the_hub_holds_the_stopping_step(ns, hub, tmp_path):
    hub.exists = True
    out = tmp_path / "outputs_v22b"
    assert ns["hub_checkpoint_step"](REPO) is None
    _checkpoint(hub.root, "last-checkpoint", 1300)
    _checkpoint(out, "checkpoint-1300", 1300)
    ns["ensure_hub_checkpoint"](REPO, str(out), 1300)                       # already there: nothing pushed
    assert hub.commits == []
    _checkpoint(out, "checkpoint-1383", 1383)                               # the time-budget save whose push did not land
    ns["ensure_hub_checkpoint"](REPO, str(out), 1383)
    assert ns["hub_checkpoint_step"](REPO) == 1383 and hub.commits == ["Training in progress, step 1383, checkpoint (verified re-push)"]
    with pytest.raises(ns["ResumeGateError"], match="no complete local checkpoint of that step"):
        ns["ensure_hub_checkpoint"](REPO, str(out), 1400)
