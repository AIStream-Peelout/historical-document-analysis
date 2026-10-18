# File name: test_train_session_resume_integration.py
# Date: 10/2/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Crash, stop and retry on the real ``transformers.Trainer``: a resumed run must BE the uninterrupted run.

The notebook's helper cell is executed as the notebook would execute it, and its
``train_with_exact_resume`` drives a tiny CPU model through the same code path the Colab run uses
(fixed row order, sequential sampler, ``ignore_data_skip``, resume gate, time budget, in-session
retry). Every scenario is compared with one uninterrupted reference run: rows per step, logged
learning rate / loss / gradient norm per step, and the final weights.
"""
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import Trainer, TrainerCallback, TrainingArguments  # noqa: E402

from src.finetuning.qwen_hebrew import train_session as ts  # noqa: E402
from src.finetuning.qwen_hebrew.colab import derive_v22b as dv  # noqa: E402

N_ROWS, MAX_STEPS, ACCUM, SAVE_EVERY, BASE_LR, WARMUP_RATIO, SEED = 160, 20, 8, 5, 1e-2, 0.1, 3407


class ToyRows(torch.utils.data.Dataset):
    """A small regression set whose items carry their own row index."""

    def __init__(self, n: int) -> None:
        gen = torch.Generator().manual_seed(7)
        self.x = torch.randn(n, 4, generator=gen)
        self.y = self.x @ torch.tensor([1.0, -2.0, 0.5, 3.0]) + 0.1 * torch.randn(n, generator=gen)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> Dict[str, object]:
        return {"x": self.x[i], "labels": self.y[i], "row": int(i)}


class ToyModel(torch.nn.Module):
    """Two-layer regressor returning a loss, as the trainer expects."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))

    def forward(self, x: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Mean-squared-error loss on a batch.

        :param x: Features.
        :param labels: Targets.
        :return: ``loss`` and ``logits``.
        """
        pred = self.net(x).squeeze(-1)
        return {"loss": torch.nn.functional.mse_loss(pred, labels), "logits": pred}


class Collate:
    """Collator that records every row it is asked to collate, in order."""

    def __init__(self) -> None:
        self.rows: List[int] = []

    def __call__(self, features: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        self.rows.extend(int(f["row"]) for f in features)
        return {"x": torch.stack([f["x"] for f in features]), "labels": torch.stack([f["labels"] for f in features])}


class TrainStartMarker(TrainerCallback):
    """Marks where each ``train()`` call's own fetches begin in the collator log (after the order gate's)."""

    def __init__(self, collate: Collate) -> None:
        self.collate, self.starts = collate, []

    def on_train_begin(self, args, state, control, **kwargs):
        self.starts.append((int(state.global_step), len(self.collate.rows)))


class CrashAt(TrainerCallback):
    """Raises at the end of a given optimizer step, a limited number of times."""

    def __init__(self, step: int, times: int = 1) -> None:
        self.step, self.left = step, times

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.step and self.left > 0:
            self.left -= 1
            raise RuntimeError(f"simulated crash at step {self.step}")


class SpendBudgetAt(TrainerCallback):
    """Moves a fake clock past the budget while a given step is running."""

    def __init__(self, step: int, now: List[float]) -> None:
        self.step, self.now = step, now

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == self.step - 1:
            self.now[0] = 1e9


@pytest.fixture(scope="module")
def ns() -> Dict[str, object]:
    """The notebook's helper cell, executed as the notebook executes it.

    :return: The cell's namespace.
    """
    space: Dict[str, object] = {}
    exec(compile(dv.session_helper_source(), "<v22b helper cell>", "exec"), space)
    return space


@pytest.fixture(scope="module")
def rows() -> ToyRows:
    """The shared toy dataset."""
    return ToyRows(N_ROWS)


@pytest.fixture(scope="module")
def order() -> List[int]:
    """The fixed row order of the whole toy run."""
    return ts.materialize_order(N_ROWS, ts.epochs_for(MAX_STEPS, N_ROWS, 1, ACCUM), seed=SEED)


def run_session(ns: Dict[str, object], out_dir: Path, rows: ToyRows, order: Sequence[int], model_seed: int,
                extra: Sequence[TrainerCallback] = (), max_retries: int = 0, max_steps: int = MAX_STEPS,
                resume_dir: Optional[str] = "auto") -> Dict[str, object]:
    """One notebook session: new model object, new trainer, resume from the newest complete checkpoint.

    :param ns: Helper-cell namespace.
    :param out_dir: The run's output directory (shared by its sessions).
    :param rows: Train dataset.
    :param order: Fixed row order.
    :param model_seed: Initialisation seed (differs between sessions so restored weights are visible).
    :param extra: Scenario callbacks.
    :param max_retries: In-session retry budget.
    :param max_steps: Planned steps for this session's trainer.
    :param resume_dir: ``"auto"`` to discover, or an explicit checkpoint directory / ``None``.
    :return: Trainer, gate, collator, marker and the final step.
    """
    collate, gate = Collate(), ns["ResumeGateTrainerCallback"]()
    marker = TrainStartMarker(collate)
    args = TrainingArguments(
        output_dir=str(out_dir), max_steps=max_steps, per_device_train_batch_size=1, gradient_accumulation_steps=ACCUM,
        learning_rate=BASE_LR, warmup_ratio=WARMUP_RATIO, lr_scheduler_type="cosine", weight_decay=0.01, optim="adamw_torch",
        logging_steps=1, save_steps=SAVE_EVERY, save_total_limit=2, eval_strategy="no", report_to=[], use_cpu=True,
        dataloader_num_workers=0, seed=SEED, data_seed=SEED, remove_unused_columns=False, disable_tqdm=True)
    trainer = Trainer(model=ToyModel(model_seed), args=args, train_dataset=rows, data_collator=collate,
                      callbacks=[gate, marker, *extra])
    if resume_dir == "auto":
        resume_dir, _ = ts.newest_complete_checkpoint(ts.local_checkpoints(str(out_dir)))
    final = ns["train_with_exact_resume"](trainer, rows, order, resume_dir, gate, max_retries=max_retries)
    return {"trainer": trainer, "gate": gate, "collate": collate, "marker": marker, "final": final}


def trained_rows(session: Dict[str, object], attempt: int = -1) -> Dict[int, List[int]]:
    """Rows each optimizer step of one ``train()`` call consumed, from the collator log.

    :param session: Result of :func:`run_session`.
    :param attempt: Which ``train()`` call of the session.
    :return: ``{step: rows}`` for every step that call completed.
    """
    start_step, start = session["marker"].starts[attempt]
    seq = session["collate"].rows[start:]
    done = int(session["trainer"].state.global_step) - start_step
    return {start_step + i + 1: seq[i * ACCUM:(i + 1) * ACCUM] for i in range(done)}


def history(trainer: Trainer) -> Dict[int, Tuple[float, float, float]]:
    """Logged (learning rate, loss, gradient norm) per step.

    :param trainer: A trainer after ``train()``.
    :return: ``{step: (lr, loss, grad_norm)}``.
    """
    return {h["step"]: (h["learning_rate"], h["loss"], h["grad_norm"]) for h in trainer.state.log_history if "loss" in h}


def weights(trainer: Trainer) -> Dict[str, torch.Tensor]:
    """A detached copy of the model's parameters."""
    return {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}


def assert_same_run(session: Dict[str, object], reference: Dict[str, object]) -> None:
    """The session ended as the uninterrupted reference did: same per-step log, same weights.

    :param session: The (last) session of an interrupted run.
    :param reference: The uninterrupted run.
    """
    got, want = history(session["trainer"]), history(reference["trainer"])
    assert sorted(got) == sorted(want) == list(range(1, MAX_STEPS + 1))
    for step in want:
        assert got[step] == pytest.approx(want[step], rel=1e-9, abs=0), f"step {step}: {got[step]} != {want[step]}"
    a, b = weights(session["trainer"]), weights(reference["trainer"])
    assert all(torch.equal(a[k], b[k]) for k in b), "final weights differ from the uninterrupted run"


@pytest.fixture(scope="module")
def reference(ns, rows, order, tmp_path_factory) -> Dict[str, object]:
    """The uninterrupted run every scenario is compared with."""
    return run_session(ns, tmp_path_factory.mktemp("reference"), rows, order, model_seed=11)


def test_uninterrupted_run_follows_the_fixed_order_and_the_cosine_formula(reference, order):
    assert reference["final"] == MAX_STEPS and reference["gate"].passed == [0]
    assert trained_rows(reference) == {s: order[(s - 1) * ACCUM:s * ACCUM] for s in range(1, MAX_STEPS + 1)}
    warm = ts.warmup_steps_for(MAX_STEPS, WARMUP_RATIO)
    for step, (lr, _, _) in history(reference["trainer"]).items():  # the trainer logs the rate the step was taken with
        assert lr == pytest.approx(ts.cosine_warmup_lr(step - 1, BASE_LR, warm, MAX_STEPS), rel=1e-9, abs=1e-15)


def test_crash_between_saves_then_a_new_session_is_the_same_run(ns, rows, order, reference, tmp_path):
    with pytest.raises(RuntimeError, match="simulated crash at step 13"):
        run_session(ns, tmp_path, rows, order, model_seed=11, extra=[CrashAt(13)])
    assert ts.newest_complete_checkpoint(ts.local_checkpoints(str(tmp_path)))[1] == 10
    resumed = run_session(ns, tmp_path, rows, order, model_seed=99)  # a new process would have different initial weights
    assert resumed["final"] == MAX_STEPS and resumed["gate"].passed == [10]
    assert trained_rows(resumed) == {s: order[(s - 1) * ACCUM:s * ACCUM] for s in range(11, MAX_STEPS + 1)}
    assert_same_run(resumed, reference)


def test_time_budget_stop_saves_and_the_next_session_continues_exactly(ns, rows, order, reference, tmp_path):
    now = [0.0]
    budget = ns["TimeBudgetTrainerCallback"](budget_s=100.0)
    budget.core._clock = lambda: now[0]
    stopped = run_session(ns, tmp_path, rows, order, model_seed=11, extra=[SpendBudgetAt(7, now), budget])
    assert stopped["final"] == 7 and budget.core.state.stopped and budget.core.state.step == 7
    assert ts.checkpoint_step(str(tmp_path / "checkpoint-7")) == 7  # an off-cadence save, complete
    resumed = run_session(ns, tmp_path, rows, order, model_seed=99)
    assert resumed["final"] == MAX_STEPS and resumed["gate"].passed == [7]
    assert trained_rows(resumed) == {s: order[(s - 1) * ACCUM:s * ACCUM] for s in range(8, MAX_STEPS + 1)}
    assert_same_run(resumed, reference)


def test_a_failure_inside_a_session_is_retried_from_the_last_checkpoint(ns, rows, order, reference, tmp_path):
    session = run_session(ns, tmp_path, rows, order, model_seed=11, extra=[CrashAt(13)], max_retries=2)
    assert session["final"] == MAX_STEPS and session["gate"].passed == [0, 10]
    assert trained_rows(session, attempt=1) == {s: order[(s - 1) * ACCUM:s * ACCUM] for s in range(11, MAX_STEPS + 1)}
    assert_same_run(session, reference)


def test_the_time_budget_keeps_counting_across_an_in_session_retry(ns, rows, order, tmp_path):
    now = [0.0]
    budget = ns["TimeBudgetTrainerCallback"](budget_s=100.0)
    budget.core._clock = lambda: now[0]
    session = run_session(ns, tmp_path, rows, order, model_seed=11, max_retries=2,
                          extra=[CrashAt(13), SpendBudgetAt(17, now), budget])
    assert session["gate"].passed == [0, 10]          # crashed at 13, retried from 10 ...
    assert session["final"] == 17                     # ... and the budget set before the retry still ends the session
    assert ts.checkpoint_step(str(tmp_path / "checkpoint-17")) == 17


def test_a_failure_that_repeats_at_the_same_step_stops_the_session(ns, rows, order, tmp_path):
    crash = CrashAt(13, times=99)
    with pytest.raises(RuntimeError, match="simulated crash at step 13"):
        run_session(ns, tmp_path, rows, order, model_seed=11, extra=[crash], max_retries=5)
    assert crash.left == 97  # failed, retried once from step 10, failed at 13 again: no third attempt


def test_a_failure_before_the_first_checkpoint_is_not_retried(ns, rows, order, tmp_path):
    crash = CrashAt(3, times=1)
    with pytest.raises(RuntimeError, match="simulated crash at step 3"):
        run_session(ns, tmp_path, rows, order, model_seed=11, extra=[crash], max_retries=2)
    assert crash.left == 0 and ts.local_checkpoints(str(tmp_path)) == []  # the weights in memory were no longer step 0's


def test_late_resume_and_a_finished_run(ns, rows, order, reference, tmp_path):
    with pytest.raises(RuntimeError):
        run_session(ns, tmp_path, rows, order, model_seed=11, extra=[CrashAt(19)])
    resumed = run_session(ns, tmp_path, rows, order, model_seed=99)   # 15 of 20 steps done: 5 steps = one short "epoch"
    assert resumed["final"] == MAX_STEPS and resumed["gate"].passed == [15]
    assert_same_run(resumed, reference)
    again = run_session(ns, tmp_path, rows, order, model_seed=5)       # the run is complete: nothing to train
    assert again["final"] == MAX_STEPS and again["gate"].passed == [] and again["marker"].starts == []
    want = weights(reference["trainer"])
    assert all(torch.equal(v, want[k]) for k, v in weights(again["trainer"]).items())  # the final weights were loaded


def test_a_changed_schedule_is_refused_before_the_first_step(ns, rows, order, tmp_path):
    with pytest.raises(RuntimeError):
        run_session(ns, tmp_path, rows, order, model_seed=11, extra=[CrashAt(13)])
    longer = ts.materialize_order(N_ROWS, ts.epochs_for(30, N_ROWS, 1, ACCUM), seed=SEED)
    with pytest.raises(ns["ResumeGateError"], match="learning rate is"):
        run_session(ns, tmp_path, rows, longer, model_seed=99, max_steps=30, max_retries=2)   # not retried either


def test_a_checkpoint_without_optimizer_state_is_never_trained_from(ns, rows, order, tmp_path):
    with pytest.raises(RuntimeError):
        run_session(ns, tmp_path / "run", rows, order, model_seed=11, extra=[CrashAt(13)])
    good = tmp_path / "run" / "checkpoint-10"
    missing = tmp_path / "no_optimizer"
    shutil.copytree(good, missing)
    (missing / "optimizer.pt").unlink()
    assert ts.checkpoint_step(str(missing)) is None
    with pytest.raises(ns["ResumeGateError"], match="not a complete checkpoint"):
        run_session(ns, tmp_path / "other", rows, order, model_seed=99, resume_dir=str(missing))
    blank = tmp_path / "blank_optimizer"
    shutil.copytree(good, blank)
    emptied = torch.load(good / "optimizer.pt", weights_only=True)
    emptied["state"] = {}                                    # same parameter groups and rate, but no moments
    torch.save(emptied, blank / "optimizer.pt")
    with pytest.raises(ns["ResumeGateError"], match="optimizer has no state"):
        run_session(ns, tmp_path / "third", rows, order, model_seed=99, resume_dir=str(blank))


def test_the_order_gate_catches_a_shuffling_sampler(ns, rows, order, tmp_path):
    collate, gate = Collate(), ns["ResumeGateTrainerCallback"]()
    args = TrainingArguments(output_dir=str(tmp_path), max_steps=MAX_STEPS, per_device_train_batch_size=1,
                             gradient_accumulation_steps=ACCUM, report_to=[], use_cpu=True, seed=SEED, disable_tqdm=True,
                             remove_unused_columns=False)
    trainer = Trainer(model=ToyModel(1), args=args, train_dataset=rows, data_collator=collate, callbacks=[gate])
    view = ns["OrderedView"](rows, order)
    trainer.train_dataset = view            # the trainer's default sampler shuffles: the fixed order would be lost
    with pytest.raises(ns["ResumeGateError"], match="order gate FAILED"):
        ns["order_gate"](trainer, view)
    assert json.dumps(order[:3])            # (the order itself is JSON-serialisable for the session plan)


def test_what_the_earlier_notebooks_did_on_resume_was_not_the_same_run(rows, tmp_path):
    """v19c / v20a / v21b / the v22a pilot: restore optimizer + scheduler, set ``ignore_data_skip``, train.

    The learning-rate schedule did continue. The data did not: the trainer's default sampler is Accelerate's
    seedable one, whose permutation depends only on ``data_seed`` and the epoch number, and with
    ``ignore_data_skip`` a resume inside an epoch restarts that epoch. So every resumed step re-trained the
    rows of the run's FIRST steps, in the same order, and the rest of the epoch was never trained. This is the
    behaviour the fixed order replaces, kept here as a measured reference on the pinned transformers version.
    """
    def session(out: Path, crash_at: Optional[int], resume: Optional[str], seed: int) -> Dict[str, object]:
        collate = Collate()
        marker = TrainStartMarker(collate)
        args = TrainingArguments(
            output_dir=str(out), max_steps=MAX_STEPS, per_device_train_batch_size=1, gradient_accumulation_steps=ACCUM,
            learning_rate=BASE_LR, warmup_ratio=WARMUP_RATIO, lr_scheduler_type="cosine", optim="adamw_torch", logging_steps=1,
            save_steps=SAVE_EVERY, eval_strategy="no", report_to=[], use_cpu=True, seed=SEED, data_seed=SEED,
            remove_unused_columns=False, disable_tqdm=True, ignore_data_skip=resume is not None)
        trainer = Trainer(model=ToyModel(seed), args=args, train_dataset=rows, data_collator=collate,
                          callbacks=[marker] + ([CrashAt(crash_at)] if crash_at else []))
        if crash_at:
            with pytest.raises(RuntimeError):
                trainer.train()
        else:
            trainer.train(resume_from_checkpoint=resume)
        return {"trainer": trainer, "collate": collate, "marker": marker}

    whole = session(tmp_path / "whole", None, None, 11)
    planned = trained_rows(whole)                                         # one pass: every row exactly once
    assert sorted(r for step in planned.values() for r in step) == list(range(N_ROWS))
    session(tmp_path / "run", 13, None, 11)
    resumed = session(tmp_path / "run", None, str(tmp_path / "run" / "checkpoint-10"), 99)
    got = trained_rows(resumed)
    assert sorted(got) == list(range(11, MAX_STEPS + 1))
    lr_resumed, lr_whole = history(resumed["trainer"]), history(whole["trainer"])
    assert all(lr_resumed[s][0] == pytest.approx(lr_whole[s][0], rel=1e-9, abs=1e-15) for s in lr_whole)   # the schedule resumed
    seen_before = {r for s in range(1, 11) for r in planned[s]}
    after = [r for s in range(11, MAX_STEPS + 1) for r in got[s]]
    repeats = sum(r in seen_before for r in after)
    never_trained = N_ROWS - len(seen_before | set(after))
    assert [got[s] for s in range(11, MAX_STEPS + 1)] != [planned[s] for s in range(11, MAX_STEPS + 1)]     # ... the data did not:
    assert [got[s] for s in range(11, MAX_STEPS + 1)] == [planned[s - 10] for s in range(11, MAX_STEPS + 1)]  # steps 11-20 replayed steps 1-10
    assert repeats == len(after) == 80 and never_trained == 80, (repeats, never_trained)
    sampler = whole["trainer"].get_train_dataloader().batch_sampler.sampler
    assert type(sampler).__name__ == "SeedableRandomSampler"
    print(f"old resume at step 10 of 20: {repeats} of {len(after)} resumed rows were repeats, {never_trained} of {N_ROWS} rows never trained")
