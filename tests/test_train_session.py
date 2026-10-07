# File name: test_train_session.py
# Date: 9/28/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Unit tests for the session-safe training helpers (deterministic order, exact resume, time budget)."""
from types import SimpleNamespace

import pytest

from src.finetuning.qwen_hebrew import train_session as ts


def test_materialize_order_is_deterministic_and_covers_each_epoch():
    a = ts.materialize_order(10, 3, seed=7)
    b = ts.materialize_order(10, 3, seed=7)
    assert a == b and len(a) == 30
    for e in range(3):
        assert sorted(a[e * 10:(e + 1) * 10]) == list(range(10))  # each epoch is a full permutation
    assert a[:10] != a[10:20]  # epochs differ
    assert ts.materialize_order(10, 3, seed=8) != a
    with pytest.raises(ValueError):
        ts.materialize_order(0, 1, 1)


def test_remaining_rows_matches_trainer_arithmetic():
    order = ts.materialize_order(100, 2, seed=1)
    # batch 1 × accum 8: step 13 has consumed 104 rows
    assert ts.rows_consumed(13, 1, 8) == 104
    rem = ts.remaining_rows(order, 13, 1, 8)
    assert rem == order[104:] and len(rem) == 96
    assert ts.remaining_rows(order, 0, 1, 8) == order
    assert ts.remaining_rows(order, 25, 1, 8) == []  # exhausted


def test_epochs_for_covers_max_steps():
    # 20,000 rows, 2,500 steps × 8 = 20,000 rows → exactly 1 pass; 2,501 steps → 2 passes
    assert ts.epochs_for(2500, 20000, 1, 8) == 1
    assert ts.epochs_for(2501, 20000, 1, 8) == 2
    assert ts.epochs_for(2000, 8000, 1, 8) == 2  # the pilot: 16,000 rows over 8,000 = 2 passes


def test_time_budget_callback_stops_once_with_save():
    now = [0.0]
    cb = ts.TimeBudgetCallback(budget_s=100, clock=lambda: now[0])
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    cb.on_train_begin(control=control)
    now[0] = 50
    cb.on_step_end(state=SimpleNamespace(global_step=10), control=control)
    assert not control.should_training_stop and not cb.state.stopped
    now[0] = 100
    cb.on_step_end(state=SimpleNamespace(global_step=20), control=control)
    assert control.should_save and control.should_training_stop
    assert cb.state.stopped and cb.state.step == 20 and cb.state.elapsed_s == pytest.approx(100)
    # a later call does not re-trigger or change the recorded step
    control2 = SimpleNamespace(should_save=False, should_training_stop=False)
    now[0] = 200
    cb.on_step_end(state=SimpleNamespace(global_step=21), control=control2)
    assert cb.state.step == 20 and not control2.should_training_stop


def test_describe_session_mentions_remaining_steps():
    s = ts.describe_session(n_rows=8000, max_steps=2000, per_device_batch=1, grad_accum=8, global_step=1300, seed=3407)
    assert "8000 rows × 2 pass(es) = 16000 ordered rows" in s
    assert "5600 rows / 700 steps remain" in s


# ---- 2026-10-02: schedule lock, checkpoint selection, resume gate, retry rule --------------------------------

def test_time_budget_clock_starts_once_so_a_retry_does_not_extend_the_session():
    now = [0.0]
    cb = ts.TimeBudgetCallback(budget_s=100, clock=lambda: now[0])
    cb.on_train_begin()
    now[0] = 80
    cb.on_train_begin()                      # train() re-entered after an in-session failure
    now[0] = 100
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    cb.on_step_end(state=SimpleNamespace(global_step=5), control=control)
    assert control.should_training_stop and cb.state.elapsed_s == pytest.approx(100)


def test_cosine_warmup_lr_shape():
    assert ts.warmup_steps_for(2500, 0.02) == 50 and ts.warmup_steps_for(2000, 0.02) == 40
    assert ts.cosine_warmup_lr(0, 3e-5, 50, 2500) == 0.0
    assert ts.cosine_warmup_lr(25, 3e-5, 50, 2500) == pytest.approx(1.5e-5)
    assert ts.cosine_warmup_lr(50, 3e-5, 50, 2500) == pytest.approx(3e-5)
    assert ts.cosine_warmup_lr(1275, 3e-5, 50, 2500) == pytest.approx(1.5e-5)   # halfway down the cosine
    assert ts.cosine_warmup_lr(2500, 3e-5, 50, 2500) == pytest.approx(0.0, abs=1e-20)


def test_cosine_warmup_lr_is_the_trainers_own_schedule():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    args = transformers.TrainingArguments(output_dir="unused", max_steps=2500, warmup_ratio=0.02, learning_rate=3e-5,
                                          lr_scheduler_type="cosine", report_to=[])
    warm = args.get_warmup_steps(2500)
    assert warm == ts.warmup_steps_for(2500, 0.02)
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=3e-5)
    sched = transformers.get_cosine_schedule_with_warmup(opt, num_warmup_steps=warm, num_training_steps=2500)
    for step in range(2501):                 # every position a resume could land on
        assert opt.param_groups[0]["lr"] == pytest.approx(ts.cosine_warmup_lr(step, 3e-5, warm, 2500), rel=1e-9, abs=1e-18)
        opt.step()
        sched.step()


def test_session_plan_round_trip_and_diff():
    order = ts.materialize_order(20, 1, seed=3)
    plan = ts.build_session_plan(order, max_steps=2500, learning_rate=3e-5, warmup_ratio=0.02, dataset_revision="a" * 40)
    assert plan["order_len"] == 20 and plan["order_sha256"] == ts.order_fingerprint(order)
    import json
    stored = json.loads(json.dumps(dict(plan, order=order, created_at="2026-10-02T00:00:00Z")))
    assert ts.diff_session_plan(stored, plan) == []            # floats survive JSON; order / timestamp are not schedule keys
    changed = dict(plan, learning_rate=5e-5, max_steps=2000)
    diff = ts.diff_session_plan(stored, changed)
    assert len(diff) == 2 and diff[0].startswith("learning_rate: stored 3e-05, now 5e-05") and "max_steps" in diff[1]
    assert ts.diff_session_plan(stored, {k: v for k, v in plan.items() if k != "warmup_ratio"}) == ["warmup_ratio: stored 0.02, missing now"]
    assert ts.diff_session_plan(stored, dict(plan, optim="adamw_8bit")) == ["optim: not in the stored plan, now 'adamw_8bit'"]
    assert ts.order_fingerprint(order) != ts.order_fingerprint(order[::-1])


def test_rows_fingerprint_sees_content_and_position():
    a = ts.rows_fingerprint([["s1", "s2"], ["x", None]])
    assert a == ts.rows_fingerprint([["s1", "s2"], ["x", None]])
    assert a != ts.rows_fingerprint([["s2", "s1"], [None, "x"]])       # same rows, other order
    assert a != ts.rows_fingerprint([["s1", "s2"], ["x", "y"]])
    assert ts.rows_fingerprint([["ab"], ["c"]]) != ts.rows_fingerprint([["a"], ["bc"]])   # field boundaries count
    with pytest.raises(ValueError):
        ts.rows_fingerprint([["a"], ["b", "c"]])


def _checkpoint(root, name, step, skip=(), state_text=None):
    """Write a fake checkpoint folder.

    :param root: Parent directory (a ``pathlib.Path``).
    :param name: Folder name.
    :param step: ``global_step`` to record.
    :param skip: File names to leave out.
    :param state_text: Raw trainer-state text instead of valid JSON.
    :return: The folder path as a string.
    """
    d = root / name
    d.mkdir(parents=True)
    files = {"trainer_state.json": state_text if state_text is not None else '{"global_step": %d}' % step,
             "optimizer.pt": "o", "scheduler.pt": "s", "adapter_model.safetensors": "w"}
    for fname, text in files.items():
        if fname not in skip:
            (d / fname).write_text(text, encoding="utf-8")
    return str(d)


def test_checkpoint_step_requires_the_full_training_state(tmp_path):
    assert ts.checkpoint_step(_checkpoint(tmp_path, "checkpoint-300", 300)) == 300
    for i, missing in enumerate(("optimizer.pt", "scheduler.pt", "trainer_state.json", "adapter_model.safetensors")):
        assert ts.checkpoint_step(_checkpoint(tmp_path, f"partial-{i}", 300, skip=(missing,))) is None
    assert ts.checkpoint_step(_checkpoint(tmp_path, "truncated", 300, state_text='{"global_step": 3')) is None
    assert ts.checkpoint_step(_checkpoint(tmp_path, "no-step", 300, state_text='{"epoch": 1.0}')) is None


def test_newest_complete_checkpoint_prefers_the_highest_complete_step(tmp_path):
    out = tmp_path / "out"
    a = _checkpoint(out, "checkpoint-1200", 1200)
    b = _checkpoint(out, "checkpoint-1300", 1300)
    _checkpoint(out, "checkpoint-1400", 1400, skip=("optimizer.pt",))        # still being written when the session died
    hub = _checkpoint(tmp_path / "resume", "last-checkpoint", 1200)
    assert ts.local_checkpoints(str(out)) == [a, b, str(out / "checkpoint-1400")]
    assert ts.newest_complete_checkpoint(ts.local_checkpoints(str(out)) + [hub]) == (b, 1300)
    assert ts.newest_complete_checkpoint([hub, str(tmp_path / "missing")]) == (hub, 1200)
    assert ts.newest_complete_checkpoint([]) == (None, 0) and ts.local_checkpoints(str(tmp_path / "nowhere")) == []


def _state(**over):
    """Arguments of ``resume_problems`` for a correct resume at step 1300 of 2500, with overrides."""
    lr = ts.cosine_warmup_lr(1300, 3e-5, 50, 2500)
    base = dict(global_step=1300, expected_step=1300, scheduler_step=1300, lr=lr, expected_lr=lr, n_batches=9600,
                expected_batches=9600, state_max_steps=2500, max_steps=2500, optimizer_steps=[1300, 1300, 1300])
    base.update(over)
    return base


def test_resume_problems_accepts_the_planned_state_only():
    assert ts.resume_problems(**_state()) == []
    assert ts.resume_problems(**_state(optimizer_steps=None, scheduler_step=None)) == []       # not exposed: not a failure
    fresh = _state(global_step=0, expected_step=0, scheduler_step=0, lr=0.0, expected_lr=0.0, n_batches=20000,
                   expected_batches=20000, optimizer_steps=[])
    assert ts.resume_problems(**fresh) == []
    cases = {
        "global_step is 0": _state(global_step=0),
        "LR scheduler is at step 0": _state(scheduler_step=0),                                  # a schedule that restarted
        "learning rate is": _state(lr=3e-5),
        "20000 batches": _state(n_batches=20000),                                               # the whole set, not the unseen tail
        "will stop at step 2000": _state(state_max_steps=2000),
        "optimizer has no state": _state(optimizer_steps=[]),                                   # fresh moments at a mid-run rate
        "optimizer has taken 1200 steps": _state(optimizer_steps=[1200, 1200]),
        "optimizer has taken 40 steps": dict(fresh, optimizer_steps=[40]),
    }
    for needle, state in cases.items():
        problems = ts.resume_problems(**state)
        assert len(problems) == 1 and needle in problems[0], (needle, problems)


def test_may_retry_rule():
    assert ts.may_retry(attempt=0, max_retries=2, failed_step=1340, last_failed_step=None, checkpoint=1300)
    assert ts.may_retry(attempt=1, max_retries=2, failed_step=1420, last_failed_step=1340, checkpoint=1400)
    assert not ts.may_retry(attempt=2, max_retries=2, failed_step=1500, last_failed_step=1420, checkpoint=1400)   # budget spent
    assert not ts.may_retry(attempt=1, max_retries=2, failed_step=1340, last_failed_step=1340, checkpoint=1300)   # same step again
    assert not ts.may_retry(attempt=0, max_retries=2, failed_step=40, last_failed_step=None, checkpoint=None)     # nothing to restart from
    assert not ts.may_retry(attempt=0, max_retries=0, failed_step=1340, last_failed_step=None, checkpoint=1300)
