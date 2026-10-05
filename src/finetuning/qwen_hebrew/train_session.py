# File name: train_session.py
# Date: 9/28/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Session-safe training helpers for the Colab fine-tunes (v22b on).

Two problems from the v22a pilot are solved here:

* **Colab's 24-hour ceiling killed the run mid-step** (the session died at step ~1,340, the last
  checkpoint was 1,300).  :class:`TimeBudgetCallback` ends training cleanly, with a save and a
  Hub push, once a wall-clock budget is spent, so every session ends on a deliberate checkpoint.
* **Resume was not 1:1.** With ``ignore_data_skip`` the trainer restarted the epoch under a fresh
  shuffle, replaying rows already seen and skipping others.  :func:`materialize_order` fixes the
  complete multi-epoch row order up front from a seed (a concatenation of per-epoch permutations),
  and :func:`remaining_rows` returns the rows a resumed session still has to consume given the
  restored ``global_step``.  Training then runs over ``Subset(dataset, order)`` with a
  *sequential* sampler, so a resumed session sees exactly the rows the crashed one did not.

The trainer's own bookkeeping (optimizer, scheduler, ``global_step``) still comes from
``resume_from_checkpoint``; with the dataset pre-ordered, ``ignore_data_skip=True`` is correct
because the skip has already been done by construction.

Hardening added 2026-10-02 (what "resumes on the same training schedule" has to mean, checked
against the earlier runs' Hub history and saved trainer states):

* **The schedule is locked.**  :func:`build_session_plan` records everything that defines the
  schedule (steps, batch, accumulation, LR, warm-up, optimiser, dataset revision, row fingerprint,
  row order); the notebook stores it in the checkpoint repo on the first launch and
  :func:`diff_session_plan` refuses a later session whose settings differ.
* **The restored state is verified before the first step.**  :func:`resume_problems` compares the
  restored ``global_step``, scheduler position, learning rate (against
  :func:`cosine_warmup_lr`), optimiser step counters and remaining row count with the plan.
* **The resume point is chosen, not assumed.**  :func:`newest_complete_checkpoint` picks the
  highest-step checkpoint that has optimizer, scheduler, trainer state and weights; an
  incomplete one is never used.
* **A failure inside a session can be retried** from the newest complete checkpoint
  (:func:`may_retry`): bounded, never without a checkpoint, never twice at the same step.
"""
import glob
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---- inlined below this line into the Colab notebook by colab/derive_v22b.py (keep the marker) ----

REQUIRED_CHECKPOINT_FILES: Tuple[str, ...] = ("trainer_state.json", "optimizer.pt", "scheduler.pt")
WEIGHT_FILES: Tuple[str, ...] = ("adapter_model.safetensors", "adapter_model.bin", "model.safetensors", "pytorch_model.bin")
PLAN_IGNORED_KEYS: Tuple[str, ...] = ("order", "order_sha256", "created_at")


def materialize_order(n_rows: int, epochs: int, seed: int) -> List[int]:
    """Deterministic multi-epoch row order: one independent permutation per epoch, concatenated.

    :param n_rows: Dataset size.
    :type n_rows: int
    :param epochs: Number of passes to lay out (use the ceiling of ``max_steps × batch / n_rows``).
    :type epochs: int
    :param seed: Base seed; epoch ``e`` uses ``seed + e``.
    :type seed: int
    :return: ``n_rows × epochs`` dataset indices.
    :rtype: List[int]
    :raises ValueError: On non-positive sizes.
    """
    if n_rows <= 0 or epochs <= 0:
        raise ValueError("n_rows and epochs must be positive")
    order: List[int] = []
    for e in range(epochs):
        order.extend(np.random.default_rng(seed + e).permutation(n_rows).tolist())
    return order


def rows_consumed(global_step: int, per_device_batch: int, grad_accum: int, world_size: int = 1) -> int:
    """Rows the trainer has consumed after ``global_step`` optimizer steps.

    :param global_step: Restored ``trainer_state.global_step``.
    :type global_step: int
    :param per_device_batch: ``per_device_train_batch_size``.
    :type per_device_batch: int
    :param grad_accum: ``gradient_accumulation_steps``.
    :type grad_accum: int
    :param world_size: Number of processes (1 on Colab).
    :type world_size: int
    :return: Number of dataset rows already seen.
    :rtype: int
    """
    return global_step * per_device_batch * grad_accum * world_size


def remaining_rows(order: Sequence[int], global_step: int, per_device_batch: int, grad_accum: int,
                   world_size: int = 1) -> List[int]:
    """The tail of the materialised order a resumed session still has to train on.

    :param order: Output of :func:`materialize_order`.
    :type order: Sequence[int]
    :param global_step: Restored ``global_step`` (0 for a fresh run).
    :type global_step: int
    :param per_device_batch: ``per_device_train_batch_size``.
    :type per_device_batch: int
    :param grad_accum: ``gradient_accumulation_steps``.
    :type grad_accum: int
    :param world_size: Number of processes.
    :type world_size: int
    :return: Remaining indices (possibly empty when the order is exhausted).
    :rtype: List[int]
    """
    done = rows_consumed(global_step, per_device_batch, grad_accum, world_size)
    return list(order[done:])


def epochs_for(max_steps: int, n_rows: int, per_device_batch: int, grad_accum: int, world_size: int = 1) -> int:
    """Smallest number of passes whose row count covers ``max_steps`` optimizer steps.

    :param max_steps: Planned optimizer steps.
    :type max_steps: int
    :param n_rows: Dataset size.
    :type n_rows: int
    :param per_device_batch: ``per_device_train_batch_size``.
    :type per_device_batch: int
    :param grad_accum: ``gradient_accumulation_steps``.
    :type grad_accum: int
    :param world_size: Number of processes.
    :type world_size: int
    :return: Passes (at least 1).
    :rtype: int
    """
    need = rows_consumed(max_steps, per_device_batch, grad_accum, world_size)
    return max(1, -(-need // n_rows))


@dataclass
class BudgetState:
    """What the time-budget callback decided, for logging and tests.

    :param stopped: Whether training was ended by the budget.
    :param elapsed_s: Elapsed wall-clock seconds when the decision was taken.
    :param step: ``global_step`` at that point.
    """
    stopped: bool = False
    elapsed_s: float = 0.0
    step: int = 0


class TimeBudgetCallback:
    """Stop training cleanly (save + push) when a wall-clock budget is spent.

    Implemented against the ``transformers.TrainerCallback`` interface without importing
    transformers at module import time, so the pure logic is unit-testable on a CPU box.
    Attach with ``trainer.add_callback(TimeBudgetCallback(budget_s=21.5 * 3600))``.

    :param budget_s: Wall-clock budget in seconds measured from the first :meth:`on_train_begin`
        (from callback creation until then).
    :type budget_s: float
    :param min_steps_between_saves: Do not stop within this many steps of the last regular
        save unless the budget is exhausted anyway (avoids a redundant checkpoint).
    :type min_steps_between_saves: int
    :param clock: Time source (injectable for tests).
    """

    def __init__(self, budget_s: float, min_steps_between_saves: int = 0, clock=time.monotonic) -> None:
        self.budget_s = float(budget_s)
        self.min_steps_between_saves = int(min_steps_between_saves)
        self._clock = clock
        self._t0 = clock()
        self._started = False
        self.state = BudgetState()

    def on_train_begin(self, args=None, state=None, control=None, **kwargs):
        """Start the clock when training first starts.

        Only the first call resets it: an in-session retry calls ``train()`` again, and the
        budget has to keep counting from the session's first step or the retry would run past
        Colab's ceiling.

        :return: ``control`` unchanged.
        """
        if not self._started:
            self._t0 = self._clock()
            self._started = True
        return control

    def elapsed(self) -> float:
        """Seconds since the clock was (re)started.

        :return: Elapsed seconds.
        :rtype: float
        """
        return self._clock() - self._t0

    def should_stop(self, global_step: int) -> bool:
        """Pure decision: budget spent?

        :param global_step: Current optimizer step.
        :type global_step: int
        :return: True when training should end now.
        :rtype: bool
        """
        return self.elapsed() >= self.budget_s

    def on_step_end(self, args=None, state=None, control=None, **kwargs):
        """Request save + stop at the end of the step that crosses the budget.

        :return: ``control`` with ``should_save`` and ``should_training_stop`` set when stopping.
        """
        step = int(getattr(state, "global_step", 0) or 0)
        if not self.state.stopped and self.should_stop(step):
            self.state = BudgetState(stopped=True, elapsed_s=self.elapsed(), step=step)
            if control is not None:
                control.should_save = True
                control.should_training_stop = True
            print(f"[TimeBudgetCallback] budget {self.budget_s/3600:.1f} h spent at step {step}; saving and stopping", flush=True)
        return control


def describe_session(n_rows: int, max_steps: int, per_device_batch: int, grad_accum: int, global_step: int,
                     seed: int, world_size: int = 1) -> str:
    """Human-readable summary of what a session will train on.

    :param n_rows: Dataset size.
    :type n_rows: int
    :param max_steps: Planned optimizer steps.
    :type max_steps: int
    :param per_device_batch: ``per_device_train_batch_size``.
    :type per_device_batch: int
    :param grad_accum: ``gradient_accumulation_steps``.
    :type grad_accum: int
    :param global_step: Restored step (0 = fresh).
    :type global_step: int
    :param seed: Order seed.
    :type seed: int
    :param world_size: Processes.
    :type world_size: int
    :return: One-line description.
    :rtype: str
    """
    ep = epochs_for(max_steps, n_rows, per_device_batch, grad_accum, world_size)
    order = materialize_order(n_rows, ep, seed)
    rem = remaining_rows(order, global_step, per_device_batch, grad_accum, world_size)
    steps_left = len(rem) // (per_device_batch * grad_accum * world_size)
    return (f"{n_rows} rows × {ep} pass(es) = {len(order)} ordered rows; resumed at step {global_step} → "
            f"{len(rem)} rows / {steps_left} steps remain (target max_steps {max_steps}); seed {seed}")


def warmup_steps_for(max_steps: int, warmup_ratio: float) -> int:
    """Warm-up steps the trainer derives from a ratio (``TrainingArguments.get_warmup_steps``).

    :param max_steps: Planned optimizer steps.
    :type max_steps: int
    :param warmup_ratio: ``warmup_ratio`` training argument.
    :type warmup_ratio: float
    :return: ``ceil(max_steps × warmup_ratio)``.
    :rtype: int
    """
    return math.ceil(max_steps * warmup_ratio)


def cosine_warmup_lr(step: int, base_lr: float, warmup_steps: int, total_steps: int) -> float:
    """Learning rate after ``step`` scheduler steps under linear warm-up + half-cosine decay.

    Mirrors ``transformers.get_cosine_schedule_with_warmup`` (``num_cycles=0.5``): this is the
    rate the optimizer uses for optimizer step ``step + 1``, and what a correctly restored
    scheduler reports right after ``resume_from_checkpoint`` at ``global_step == step``.

    :param step: Scheduler steps taken so far (the restored ``global_step``).
    :type step: int
    :param base_lr: ``learning_rate`` training argument.
    :type base_lr: float
    :param warmup_steps: Warm-up length in steps.
    :type warmup_steps: int
    :param total_steps: ``max_steps``.
    :type total_steps: int
    :return: Scheduled learning rate.
    :rtype: float
    """
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def order_fingerprint(order: Sequence[int]) -> str:
    """SHA-256 of a row order (detects a changed permutation between sessions).

    :param order: Dataset indices in training order.
    :type order: Sequence[int]
    :return: Hex digest.
    :rtype: str
    """
    return hashlib.sha256(",".join(str(int(i)) for i in order).encode("ascii")).hexdigest()


def rows_fingerprint(columns: Sequence[Sequence[object]]) -> str:
    """SHA-256 over the rows of a dataset, given some of its columns (index → row mapping check).

    :param columns: Equal-length column lists (for example image hash, source, task, question, answer).
    :type columns: Sequence[Sequence[object]]
    :return: Hex digest.
    :rtype: str
    :raises ValueError: When the columns differ in length.
    """
    lengths = {len(c) for c in columns}
    if len(lengths) > 1:
        raise ValueError(f"columns differ in length: {sorted(lengths)}")
    digest = hashlib.sha256()
    for row in zip(*columns):
        digest.update(("\x1f".join("" if v is None else str(v) for v in row) + "\x1e").encode("utf-8"))
    return digest.hexdigest()


def build_session_plan(order: Sequence[int], **schedule: object) -> Dict[str, object]:
    """Everything that defines the training schedule, as one JSON-serialisable record.

    :param order: The materialised row order the run trains on.
    :type order: Sequence[int]
    :param schedule: Schedule-defining settings (steps, batch, accumulation, learning rate,
        warm-up, scheduler, optimiser, seeds, dataset repo and revision, row fingerprint, ...).
    :type schedule: object
    :return: The settings plus ``order_len`` and ``order_sha256``.
    :rtype: Dict[str, object]
    """
    plan: Dict[str, object] = dict(schedule)
    plan["order_len"] = len(order)
    plan["order_sha256"] = order_fingerprint(order)
    return plan


def diff_session_plan(stored: Dict[str, object], current: Dict[str, object],
                      ignore: Iterable[str] = PLAN_IGNORED_KEYS) -> List[str]:
    """Differences between the plan stored at the first launch and this session's plan.

    :param stored: Plan read back from the checkpoint repository.
    :type stored: Dict[str, object]
    :param current: Plan built from this session's settings.
    :type current: Dict[str, object]
    :param ignore: Keys that do not define the schedule (the order itself is taken from the stored
        plan, so its fingerprint is not compared here).
    :type ignore: Iterable[str]
    :return: One line per differing key; empty when the schedules are the same.
    :rtype: List[str]
    """
    skip = set(ignore)
    missing = object()
    problems: List[str] = []
    for key in sorted((set(stored) | set(current)) - skip):
        a, b = stored.get(key, missing), current.get(key, missing)
        if a is missing:
            problems.append(f"{key}: not in the stored plan, now {b!r}")
        elif b is missing:
            problems.append(f"{key}: stored {a!r}, missing now")
        elif a != b:
            problems.append(f"{key}: stored {a!r}, now {b!r}")
    return problems


def checkpoint_step(path: str) -> Optional[int]:
    """``global_step`` of a checkpoint folder that can restore the full training state.

    :param path: Checkpoint directory.
    :type path: str
    :return: The step, or ``None`` when trainer state, optimizer, scheduler or weights are missing
        or the trainer state does not parse (a folder still being written or a partial download).
    :rtype: Optional[int]
    """
    if not all(os.path.isfile(os.path.join(path, f)) for f in REQUIRED_CHECKPOINT_FILES):
        return None
    if not any(os.path.isfile(os.path.join(path, f)) for f in WEIGHT_FILES):
        return None
    with open(os.path.join(path, "trainer_state.json"), encoding="utf-8") as fh:
        text = fh.read()
    if not text.strip().endswith("}"):
        return None
    step = json.loads(text).get("global_step")
    return int(step) if isinstance(step, int) else None


def local_checkpoints(output_dir: str) -> List[str]:
    """The trainer's ``checkpoint-*`` folders under an output directory.

    :param output_dir: ``TrainingArguments.output_dir``.
    :type output_dir: str
    :return: Sorted directory paths (possibly empty).
    :rtype: List[str]
    """
    return sorted(d for d in glob.glob(os.path.join(output_dir, "checkpoint-*")) if os.path.isdir(d))


def newest_complete_checkpoint(paths: Sequence[str]) -> Tuple[Optional[str], int]:
    """The complete checkpoint with the highest ``global_step``.

    :param paths: Candidate checkpoint directories (missing or incomplete ones are skipped).
    :type paths: Sequence[str]
    :return: ``(path, step)``; ``(None, 0)`` when no candidate is complete.
    :rtype: Tuple[Optional[str], int]
    """
    best: Tuple[Optional[str], int] = (None, 0)
    for path in paths:
        if not os.path.isdir(path):
            continue
        step = checkpoint_step(path)
        if step is not None and (best[0] is None or step > best[1]):
            best = (path, step)
    return best


def resume_problems(global_step: int, expected_step: int, scheduler_step: Optional[int], lr: float,
                    expected_lr: float, n_batches: int, expected_batches: int, state_max_steps: int,
                    max_steps: int, optimizer_steps: Optional[Sequence[int]], rel_tol: float = 1e-6) -> List[str]:
    """Check the state a trainer is about to train from against the planned schedule.

    :param global_step: ``trainer.state.global_step`` after the checkpoint was loaded.
    :type global_step: int
    :param expected_step: Step of the checkpoint the session resumes from (0 = fresh).
    :type expected_step: int
    :param scheduler_step: ``lr_scheduler.last_epoch`` (``None`` when not exposed).
    :type scheduler_step: Optional[int]
    :param lr: Learning rate the scheduler reports now.
    :type lr: float
    :param expected_lr: :func:`cosine_warmup_lr` at ``expected_step``.
    :type expected_lr: float
    :param n_batches: Length of the prepared train dataloader.
    :type n_batches: int
    :param expected_batches: Batches left in the fixed row order.
    :type expected_batches: int
    :param state_max_steps: ``trainer.state.max_steps``.
    :type state_max_steps: int
    :param max_steps: Planned ``max_steps``.
    :type max_steps: int
    :param optimizer_steps: Per-parameter step counters of the optimizer (``None`` when the
        optimizer does not expose them, which is not treated as a problem).
    :type optimizer_steps: Optional[Sequence[int]]
    :param rel_tol: Relative tolerance on the learning rate.
    :type rel_tol: float
    :return: One line per mismatch; empty when the state is exactly the planned one.
    :rtype: List[str]
    """
    problems: List[str] = []
    if global_step != expected_step:
        problems.append(f"global_step is {global_step}, the checkpoint is step {expected_step}")
    if scheduler_step is not None and scheduler_step != expected_step:
        problems.append(f"the LR scheduler is at step {scheduler_step}, expected {expected_step} (schedule not restored)")
    if abs(lr - expected_lr) > rel_tol * max(abs(expected_lr), 1e-12):
        problems.append(f"learning rate is {lr:.6e}, the schedule gives {expected_lr:.6e} at step {expected_step}")
    if n_batches != expected_batches:
        problems.append(f"the train dataloader has {n_batches} batches, the fixed order has {expected_batches} left")
    if state_max_steps != max_steps:
        problems.append(f"the trainer will stop at step {state_max_steps}, the plan says {max_steps}")
    if optimizer_steps is not None:
        top = max(optimizer_steps) if optimizer_steps else 0
        if expected_step > 0 and not optimizer_steps:
            problems.append("the optimizer has no state: its moments were not restored from the checkpoint")
        elif top != expected_step:
            problems.append(f"the optimizer has taken {top} steps, expected {expected_step} (optimizer state not restored)")
    return problems


def may_retry(attempt: int, max_retries: int, failed_step: int, last_failed_step: Optional[int],
              checkpoint: Optional[int]) -> bool:
    """Whether a failed ``train()`` call may be re-entered inside the same session.

    :param attempt: Retries already used in this session.
    :type attempt: int
    :param max_retries: Retry budget.
    :type max_retries: int
    :param failed_step: ``global_step`` when the failure happened.
    :type failed_step: int
    :param last_failed_step: ``global_step`` of the previous failure in this session, if any.
    :type last_failed_step: Optional[int]
    :param checkpoint: Step of the newest complete checkpoint, or ``None`` when there is none.
    :type checkpoint: Optional[int]
    :return: True when retries remain, a complete checkpoint exists to restart from (without one
        the weights in memory are no longer the planned ones), and the failure is not a repeat at
        the same step (a row or state that fails deterministically needs a person).
    :rtype: bool
    """
    return attempt < max_retries and checkpoint is not None and failed_step != last_failed_step
