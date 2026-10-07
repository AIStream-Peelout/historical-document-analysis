# File name: derive_v22b.py
# Date: 9/28/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Derive the v22b Colab notebook from the v22a pilot notebook.

The pilot notebook is the source of truth for the training stack; this script applies the
v22b deltas as exact, asserted string edits so nothing else drifts:

* dataset → the full mixture (``isaacmg/genizah_v22_full`` at a pinned revision), with the
  size and share gates updated for the seven components;
* checkpoint repo / output dir / run name → ``v22b``;
* ``max_steps`` 2,500 (20,000 rows = one pass) and ``learning_rate`` 3e-5 (continued training on a
  converged model: adjust, do not re-carve);
* W&B run named at ``wandb.init`` time (the pilot's ``config.update`` before init produced a
  default run name);
* deterministic multi-epoch row order with **exact resume** (``train_session.py`` inlined:
  the resumed session trains on the rows the crashed one did not consume, under a sequential
  sampler) and a **time-budget stop** that saves + pushes before Colab's 24-hour ceiling;
* **a failed session resumes on the same schedule** (2026-10-02): the resume checkpoint is
  downloaded outside ``output_dir`` (inside it, every model push re-uploaded the stale folder and
  rolled the Hub's ``last-checkpoint`` back to the session's start: seen on the v19c, v20a and
  v21b resumed sessions), a Hub error is no longer read as "fresh start", the schedule and the
  row order are stored in the checkpoint repo on the first launch and must match afterwards,
  the restored step / learning rate / optimizer state / remaining rows are checked before the
  first step, a failure inside a session is retried from the newest complete checkpoint, and the
  session ends only once the Hub holds the checkpoint it stopped on.

Usage::

    .venv/bin/python -m src.finetuning.qwen_hebrew.colab.derive_v22b --revision <40-hex sha> \\
        [--src colab/genizah_v22a.ipynb] [--out colab/genizah_v22b.ipynb]
"""
import argparse
import copy
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
SESSION_MODULE = HERE.parent / "train_session.py"

FULL_REPO = "isaacmg/genizah_v22_full"
CKPT_REPO = "isaacmg/qwen3-vl-8b-hebrew-v22b-ckpt"
RUN_NAME = "genizah_v22b"
MAX_STEPS = 2500
LR = "3e-5"
TIME_BUDGET_H = 21.5
MAX_INSESSION_RETRIES = 2
ORDER_SEED = 3407
RESUME_ROOT = "/content/v22b_resume"
INLINE_MARKER = "# ---- inlined below this line into the Colab notebook by colab/derive_v22b.py (keep the marker) ----"
MIN_TRAIN_ROWS = 19000
SHARE_FLOORS = '(("pgp_qa", 0.10), ("pgp_editions", 0.20), ("documentary_grounding", 0.08), ("talmud_replay", 0.06), ("synthetic", 0.05))'


OLD_DISCOVERY = (
    'resume_dir = None\n'
    'try:\n'
    '    # hub_strategy="checkpoint" pushes a rolling "last-checkpoint/" folder\n'
    '    files = list_repo_files(CKPT_REPO)\n'
    '    if any(f.startswith("last-checkpoint/") for f in files):\n'
    '        snapshot_download(CKPT_REPO, allow_patterns="last-checkpoint/*", local_dir=OUT_DIR)\n'
    '        resume_dir = f"{OUT_DIR}/last-checkpoint"\n'
    '        print("resuming from last-checkpoint")\n'
    'except Exception as e:\n'
    '    print(f"no checkpoint repo yet ({type(e).__name__}) — fresh start")\n'
)
NEW_DISCOVERY = f'''# ---- v22b: where to resume from (helpers cell) ----
# The Hub's last-checkpoint is downloaded OUTSIDE output_dir: inside it, every model push re-uploads the stale folder
# and rolls the Hub's resume point back to this session's start (seen on the v19c / v20a / v21b resumed sessions).
# Only "the repository does not exist" is a fresh start; any other Hub error stops the notebook here.
RESUME_ROOT = "{RESUME_ROOT}"
resume_dir, _resume_step = discover_resume(CKPT_REPO, OUT_DIR, RESUME_ROOT)
print(f"resume point: {{resume_dir}} (step {{_resume_step}})" if resume_dir else "no checkpoint on the Hub or on this machine: fresh start from the warm start")
'''
OLD_TAIL = '''if resume_dir:
    # map-style resume: the Trainer would replay (collate) every seen batch on the CPU to skip it —
    # hours of idle GPU for nothing (v2.1b lesson). Restore optimizer/scheduler/step and start a fresh
    # shuffled epoch instead; a few rows repeat or are skipped, which is fine for a fine-tune.
    trainer.args.ignore_data_skip = True
    print("resume: optimizer/scheduler/step restored, data order restarted (ignore_data_skip=True)")
trainer.train(resume_from_checkpoint=resume_dir)'''
NEW_TAIL = '''# ---- v22b: schedule lock, fixed row order, exact resume (helpers cell) ----
_a = trainer.args
_order = materialize_order(len(train_ds), epochs_for(MAX_STEPS, len(train_ds), _a.per_device_train_batch_size, _a.gradient_accumulation_steps), seed=ORDER_SEED)
_plan = build_session_plan(
    _order, order_seed=ORDER_SEED, n_rows=len(train_ds), max_steps=_a.max_steps,
    per_device_batch=_a.per_device_train_batch_size, grad_accum=_a.gradient_accumulation_steps,
    learning_rate=_a.learning_rate, warmup_ratio=_a.warmup_ratio,
    lr_scheduler_type=str(getattr(_a.lr_scheduler_type, "value", _a.lr_scheduler_type)),
    weight_decay=_a.weight_decay, optim=str(getattr(_a.optim, "value", _a.optim)),
    dataset_repo=V22_REPO, dataset_revision=V22_REVISION, warm_start=f"{WARM_CKPT_REPO}@{WARM_REVISION}",
    rows_sha256=rows_fingerprint([train_ds.column(c) for c in ("image_sha1", "source", "task", "question", "answer")]))
_order = lock_session_plan(CKPT_REPO, _plan, _order, has_checkpoint=resume_dir is not None)   # first launch stores it; later sessions must match it
print(describe_session(len(train_ds), MAX_STEPS, _a.per_device_train_batch_size, _a.gradient_accumulation_steps, _resume_step, ORDER_SEED))
_gate = ResumeGateTrainerCallback()
trainer.add_callback(_gate)                                                      # step / LR / optimizer state / rows left are checked before the first step
trainer.add_callback(TimeBudgetTrainerCallback(budget_s=TIME_BUDGET_H * 3600))   # clean save + push before Colab's 24 h cap
_final_step = train_with_exact_resume(trainer, train_ds, _order, resume_dir, _gate, max_retries=MAX_INSESSION_RETRIES)
ensure_hub_checkpoint(CKPT_REPO, OUT_DIR, _final_step)                           # the session ends only once the Hub holds this step
print(f"session ended at step {_final_step} of {MAX_STEPS}" + ("" if _final_step >= MAX_STEPS else ": run the notebook again to continue from exactly here"))'''
INTRO_MD = f'''# Genizah v2.2b — the full v22 mixture, with exact resume

Same training stack as the v2.2a pilot (pinned install triplet, 6.5/7 MP resolution contract, LoRA r16 on tower +
language, merger FROZEN, adamw_8bit, cosine, warm start = **v2.1b step 1200**), on the full mixture
`{FULL_REPO}` (images-once layout, map-style, built by `src/finetuning/qwen_hebrew/build_v22_mixture.py`):

| component | share | what |
|---|---|---|
| ktiv_transcription | 0.28 | page / region / section / line transcription (KTIV): literary replay |
| pgp_editions | 0.22 | line-broken PGP edition pages + line-structure tasks |
| ktiv_grounding | 0.13 | locate / read_box / line_index / detect / crop (KTIV geometry) |
| documentary_grounding | 0.10 | locate / read_box on two-reader agreed lines (Kraken geometry) |
| pgp_qa | 0.12 | extractive QA v2: text-first answers, capped answer shares, balanced abstention, paraphrases |
| talmud_replay | 0.08 | Talmud page transcription (literary replay) |
| synthetic | 0.07 | synthetic Hebrew renders |

20,000 train rows = **one pass = {MAX_STEPS} steps** (batch 1 × accumulation 8), learning rate {LR}, 2 % warm-up, cosine.
Run name `{RUN_NAME}`; checkpoints in `{CKPT_REPO}`. Design and gates: `docs/v22_dataset.md`.

## Sessions and resuming

One Colab session cannot hold the run (about 65 s/step). Each session **stops itself after {TIME_BUDGET_H} h** with a save
and a Hub push, and the next one continues **on the same schedule**: same step, same learning rate, same optimizer
state, and the rows the previous sessions did not train on, in the same fixed order.

**To continue after a stop, a crash or a disconnect: run the whole notebook again (Runtime → Run all).** Nothing is
edited between sessions. The training cell then

1. takes the newest complete checkpoint (the Hub's `last-checkpoint`, or a newer one still on this machine);
2. checks this session's settings against `session_plan.json` in the checkpoint repo (written on the first launch:
   steps, batch, learning rate, warm-up, optimizer, dataset revision, row fingerprint and the row order) and refuses to
   train if anything differs;
3. checks, before the first step, that the restored step, learning rate, scheduler position and optimizer state are the
   planned ones and that the dataloader hands out exactly the rows that are left;
4. retries a failure inside the session up to {MAX_INSESSION_RETRIES} times from the newest checkpoint (never twice at the same step);
5. ends only once the Hub holds the checkpoint it stopped on.

A session that dies between two saves loses at most the steps since the last save (saves every 100 steps); the next
session replays exactly those rows.
'''


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    """Replace exactly one occurrence or fail loudly.

    :param text: Source text.
    :type text: str
    :param old: Anchor that must occur exactly once.
    :type old: str
    :param new: Replacement.
    :type new: str
    :param what: Label for the error message.
    :type what: str
    :return: Edited text.
    :rtype: str
    :raises ValueError: When the anchor is missing or ambiguous.
    """
    n = text.count(old)
    if n != 1:
        raise ValueError(f"{what}: anchor found {n} times: {old[:80]!r}")
    return text.replace(old, new, 1)


def _replace_line(text: str, pattern: str, new_line: str, what: str) -> str:
    """Replace exactly one line matching a regex (whole line).

    :param text: Source text.
    :type text: str
    :param pattern: Regex applied per line (``re.match``).
    :type pattern: str
    :param new_line: Replacement line.
    :type new_line: str
    :param what: Label.
    :type what: str
    :return: Edited text.
    :rtype: str
    :raises ValueError: When no or several lines match.
    """
    lines = text.splitlines()
    hits = [i for i, l in enumerate(lines) if re.match(pattern, l)]
    if len(hits) != 1:
        raise ValueError(f"{what}: {len(hits)} lines match {pattern!r}")
    lines[hits[0]] = new_line
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def session_helper_source() -> str:
    """The inlined helper cell: train_session.py's functions plus the notebook-only glue.

    :return: Python source for one code cell.
    :rtype: str
    :raises ValueError: When the inlining marker is missing from ``train_session.py``.
    """
    mod = SESSION_MODULE.read_text(encoding="utf-8")
    if mod.count(INLINE_MARKER) != 1:
        raise ValueError(f"train_session.py must contain the inlining marker exactly once ({mod.count(INLINE_MARKER)} found)")
    body = mod.split(INLINE_MARKER, 1)[1]
    glue = f'''
# ---- v22b session helpers (inlined from src/finetuning/qwen_hebrew/train_session.py; keep in sync) ----
import gc, glob, hashlib, json, math, os, time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset as _TorchDataset, SequentialSampler
from transformers import TrainerCallback
{{body}}

# ---- notebook-only glue (needs torch / transformers / the Hub) ----
PLAN_FILE = "session_plan.json"


class ResumeGateError(AssertionError):
    """The state about to be trained from is not the planned schedule. Never retried."""


class OrderedView(_TorchDataset):
    """A map-style view of ``base`` in a fixed index order (the rows a session still has to consume)."""

    def __init__(self, base, indices):
        self.base, self.indices = base, list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]

    def column(self, name):
        col = self.base.column(name)
        return [col[j] for j in self.indices]


class _RecordingView(_TorchDataset):
    """Records the positions a dataloader fetches (order gate)."""

    def __init__(self, ds):
        self.ds, self.seen = ds, []

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        self.seen.append(int(i))
        return self.ds[i]


class TimeBudgetTrainerCallback(TrainerCallback):
    """``TimeBudgetCallback`` as a real ``TrainerCallback`` (the core stays framework-free for tests)."""

    def __init__(self, budget_s: float):
        super().__init__()
        self.core = TimeBudgetCallback(budget_s)

    def on_train_begin(self, args, state, control, **kwargs):
        return self.core.on_train_begin(args, state, control, **kwargs)

    def on_step_end(self, args, state, control, **kwargs):
        return self.core.on_step_end(args, state, control, **kwargs)


def _optimizer_steps(optimizer) -> Optional[List[int]]:
    """Per-parameter step counters of an optimizer (AdamW and the 8-bit AdamW both keep ``state[p]["step"]``).

    :param optimizer: The trainer's optimizer (possibly wrapped by Accelerate).
    :return: Step counters, or ``None`` when the optimizer does not expose them.
    """
    state = getattr(getattr(optimizer, "optimizer", optimizer), "state", None)
    if state is None:
        return None
    steps = []
    for entry in state.values():
        if isinstance(entry, dict) and "step" in entry:
            value = entry["step"]
            steps.append(int(value.item()) if hasattr(value, "item") else int(value))
    return steps


class ResumeGateTrainerCallback(TrainerCallback):
    """Before the first step of every ``train()`` call: the restored state must be the planned schedule.

    Runs in ``on_train_begin``, i.e. after the trainer loaded weights, optimizer and scheduler from the
    checkpoint and built its train dataloader. Raises :class:`ResumeGateError` on any mismatch.
    """

    def __init__(self):
        super().__init__()
        self.expected_step, self.expected_rows, self.passed = 0, 0, []

    def expect(self, step: int, rows: int) -> None:
        """Set what the next ``train()`` call must start from.

        :param step: Step of the checkpoint (0 = fresh).
        :param rows: Rows left in the fixed order.
        """
        self.expected_step, self.expected_rows = int(step), int(rows)

    def on_train_begin(self, args, state, control, optimizer=None, lr_scheduler=None, train_dataloader=None, **kwargs):
        kind = str(getattr(args.lr_scheduler_type, "value", args.lr_scheduler_type))
        if kind != "cosine":
            raise ResumeGateError(f"the resume gate knows the cosine schedule, the trainer uses {{kind!r}}")
        inner = getattr(optimizer, "optimizer", optimizer)
        lr = float(inner.param_groups[0]["lr"])
        expected_lr = cosine_warmup_lr(self.expected_step, args.learning_rate, args.get_warmup_steps(args.max_steps), args.max_steps)
        counters = _optimizer_steps(optimizer)
        problems = resume_problems(
            global_step=int(state.global_step), expected_step=self.expected_step,
            scheduler_step=getattr(lr_scheduler, "last_epoch", None), lr=lr, expected_lr=expected_lr,
            n_batches=len(train_dataloader), expected_batches=-(-self.expected_rows // args.per_device_train_batch_size),
            state_max_steps=int(state.max_steps), max_steps=int(args.max_steps), optimizer_steps=counters)
        if problems:
            raise ResumeGateError("resume gate FAILED before the first step:\\n  - " + "\\n  - ".join(problems))
        self.passed.append(self.expected_step)
        seen = "not exposed" if counters is None else (f"{{len(counters)}} tensors at step {{max(counters)}}" if counters else "empty (fresh)")
        print(f"resume gate OK: step {{state.global_step}} of {{args.max_steps}}, lr {{lr:.6e}} (schedule {{expected_lr:.6e}}), "
              f"scheduler at {{getattr(lr_scheduler, 'last_epoch', '?')}}, optimizer state {{seen}}, {{self.expected_rows}} rows left", flush=True)
        return control


def order_gate(trainer, view, k: int = 3) -> None:
    """The PREPARED train dataloader must hand out the view's first rows, in order.

    :param trainer: The trainer (its sampler override is already installed).
    :param view: The ordered view it is about to train on.
    :param k: Rows to check.
    :raises ResumeGateError: When the dataloader fetches anything but positions ``0..k-1``.
    """
    rec = _RecordingView(view)
    saved_ds, saved_workers = trainer.train_dataset, trainer.args.dataloader_num_workers
    trainer.train_dataset, trainer.args.dataloader_num_workers = rec, 0   # in-process fetch: positions visible, nothing prefetched
    it = iter(trainer.get_train_dataloader())
    n = min(k, len(view))
    while len(rec.seen) < n:
        next(it)
    trainer.train_dataset, trainer.args.dataloader_num_workers = saved_ds, saved_workers
    del it
    if rec.seen[:n] != list(range(n)):
        raise ResumeGateError(f"order gate FAILED: the dataloader fetched view positions {{rec.seen[:n]}}, expected {{list(range(n))}} "
                              "(the sequential sampler is not in effect)")
    print(f"order gate OK: the prepared dataloader starts with rows {{[view.indices[i] for i in range(n)]}} of the fixed order")


def train_with_exact_resume(trainer, base_ds, order, resume_dir, gate, max_retries: int = 2) -> int:
    """Train over the fixed row order from ``resume_dir`` (or step 0); re-enter after an in-session failure.

    Every attempt starts from a complete checkpoint: weights, optimizer and scheduler are reloaded by the
    trainer, the dataset is the tail of ``order`` that checkpoint has not consumed, the sampler is
    sequential, and both gates run before the first step. A failure is retried at most ``max_retries``
    times, never without a checkpoint and never twice at the same step.

    :param trainer: The trainer (``max_steps`` set; callbacks ``gate`` and the time budget already added).
    :param base_ds: The full train dataset (map-style).
    :param order: The materialised row order of the whole run.
    :param resume_dir: Checkpoint to start from, or ``None`` for a fresh run.
    :param gate: The :class:`ResumeGateTrainerCallback` attached to ``trainer``.
    :param max_retries: In-session retry budget.
    :return: ``global_step`` when training returned.
    """
    batch, accum = trainer.args.per_device_train_batch_size, trainer.args.gradient_accumulation_steps
    attempt, last_failed_step = 0, None
    while True:
        step = 0
        if resume_dir is not None:
            step = checkpoint_step(resume_dir)
            if step is None:
                raise ResumeGateError(f"{{resume_dir}} is not a complete checkpoint (trainer state, optimizer, scheduler and weights are all needed)")
        remaining = remaining_rows(order, step, batch, accum)
        if step >= trainer.args.max_steps or not remaining:
            trainer._load_from_checkpoint(resume_dir)
            print(f"run already complete at step {{step}}: weights loaded from {{resume_dir}}, nothing left to train")
            return step
        view = OrderedView(base_ds, remaining)
        trainer.train_dataset = view
        trainer.args.ignore_data_skip = True     # the skip is done by construction: the view holds only unseen rows
        trainer._get_train_sampler = lambda *a, **k: SequentialSampler(trainer.train_dataset)   # the order is baked into the view
        order_gate(trainer, view)
        gate.expect(step, len(view))
        print(f"train() attempt {{attempt + 1}}: from step {{step}}, {{len(view)}} rows left in the fixed order", flush=True)
        failure = None
        try:
            trainer.train(resume_from_checkpoint=resume_dir)
        except Exception as error:   # the one deliberate catch-all: a long run re-enters from its last checkpoint
            if isinstance(error, AssertionError):
                raise                # gate failures are never retried
            failure = error
        if failure is None:
            return int(trainer.state.global_step)
        failed_step = int(trainer.state.global_step)
        candidates = local_checkpoints(trainer.args.output_dir) + ([resume_dir] if resume_dir else [])
        next_dir, next_step = newest_complete_checkpoint(candidates)
        if not may_retry(attempt, max_retries, failed_step, last_failed_step, next_step if next_dir else None):
            raise failure
        print(f"⚠️ train() failed at step {{failed_step}}: {{type(failure).__name__}}: {{str(failure)[:300]}}\\n"
              f"   in-session retry {{attempt + 1}}/{{max_retries}} from {{next_dir}} (step {{next_step}})", flush=True)
        attempt, last_failed_step, resume_dir, failure = attempt + 1, failed_step, next_dir, None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def discover_resume(ckpt_repo: str, out_dir: str, resume_root: str):
    """Newest complete checkpoint among the Hub's ``last-checkpoint`` and this machine's ``checkpoint-*`` folders.

    The Hub copy is downloaded to ``resume_root``, which must lie OUTSIDE ``out_dir``: the trainer's model
    push uploads everything in ``output_dir`` except ``checkpoint-*``, so a ``last-checkpoint`` folder
    inside it is re-uploaded at every save and rolls the Hub's resume point back to the session's start.
    Only "the repository does not exist" means a fresh start; any other Hub error propagates.

    :param ckpt_repo: Checkpoint repository id.
    :param out_dir: The trainer's ``output_dir``.
    :param resume_root: Download directory for the Hub checkpoint.
    :return: ``(checkpoint_dir or None, step)``.
    """
    from huggingface_hub import list_repo_files, snapshot_download
    from huggingface_hub.utils import RepositoryNotFoundError
    out_abs, root_abs = os.path.abspath(out_dir), os.path.abspath(resume_root)
    if root_abs == out_abs or root_abs.startswith(out_abs + os.sep):
        raise ResumeGateError(f"resume_root {{resume_root}} must not be inside output_dir {{out_dir}}")
    candidates = local_checkpoints(out_dir)
    try:
        files = list_repo_files(ckpt_repo)
    except RepositoryNotFoundError:
        files = []
        print(f"{{ckpt_repo}} does not exist yet")
    on_hub = any(f.startswith("last-checkpoint/") for f in files)
    if on_hub:
        snapshot_download(ckpt_repo, allow_patterns="last-checkpoint/*", local_dir=resume_root)
        candidates.append(os.path.join(resume_root, "last-checkpoint"))
    path, step = newest_complete_checkpoint(candidates)
    if on_hub and path is None:
        raise ResumeGateError(f"{{ckpt_repo}} has a last-checkpoint but no complete copy could be assembled (optimizer / scheduler / weights missing)")
    return path, step


def lock_session_plan(ckpt_repo: str, plan: Dict[str, object], order: Sequence[int], has_checkpoint: bool) -> List[int]:
    """Store the schedule and row order on the first launch; afterwards require the same schedule.

    :param ckpt_repo: Checkpoint repository id (exists: the trainer creates it).
    :param plan: This session's :func:`build_session_plan`.
    :param order: The row order materialised in this session.
    :param has_checkpoint: Whether a checkpoint was found to resume from.
    :return: The order to train on (the stored one from the second session on).
    :raises ResumeGateError: On any schedule difference, or a checkpoint without a stored plan.
    """
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    try:
        with open(hf_hub_download(ckpt_repo, PLAN_FILE, force_download=True), encoding="utf-8") as fh:
            stored = json.load(fh)
    except EntryNotFoundError:
        stored = None
    if stored is None:
        if has_checkpoint:
            raise ResumeGateError(f"{{ckpt_repo}} has a checkpoint but no {{PLAN_FILE}}: it was not started by this notebook, refusing to continue it")
        payload = dict(plan, order=[int(i) for i in order], created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        HfApi().upload_file(path_or_fileobj=json.dumps(payload).encode("utf-8"), path_in_repo=PLAN_FILE, repo_id=ckpt_repo,
                            commit_message="Session plan: schedule and fixed row order (exact resume)")
        print(f"session plan stored in {{ckpt_repo}}/{{PLAN_FILE}} ({{len(order)}} ordered rows, sha {{plan['order_sha256'][:12]}})")
        return [int(i) for i in order]
    problems = diff_session_plan(stored, plan)
    if problems:
        raise ResumeGateError("this session's settings differ from the plan the run was started with:\\n  - " + "\\n  - ".join(problems))
    stored_order = [int(i) for i in stored["order"]]
    if order_fingerprint(stored_order) != stored["order_sha256"] or len(stored_order) != stored["order_len"]:
        raise ResumeGateError(f"the row order stored in {{PLAN_FILE}} does not match its own fingerprint")
    if stored["order_sha256"] != plan["order_sha256"]:
        print("⚠️ this session regenerated a different permutation (library change); training on the STORED order")
    print(f"session plan verified against {{ckpt_repo}}/{{PLAN_FILE}} (started {{stored.get('created_at', '?')}})")
    return stored_order


def hub_checkpoint_step(ckpt_repo: str) -> Optional[int]:
    """``global_step`` of the Hub's ``last-checkpoint`` right now.

    :param ckpt_repo: Checkpoint repository id.
    :return: The step, or ``None`` when there is no ``last-checkpoint``.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    try:
        path = hf_hub_download(ckpt_repo, "last-checkpoint/trainer_state.json", force_download=True)
    except EntryNotFoundError:
        return None
    with open(path, encoding="utf-8") as fh:
        return int(json.load(fh)["global_step"])


def ensure_hub_checkpoint(ckpt_repo: str, out_dir: str, expected_step: int) -> None:
    """The session may only end once the Hub's resume point is the step training stopped on.

    :param ckpt_repo: Checkpoint repository id.
    :param out_dir: The trainer's ``output_dir``.
    :param expected_step: ``global_step`` when ``train()`` returned.
    :raises ResumeGateError: When the checkpoint cannot be confirmed on the Hub.
    """
    step = hub_checkpoint_step(ckpt_repo)
    if step != expected_step:
        from huggingface_hub import upload_folder
        path, local_step = newest_complete_checkpoint(local_checkpoints(out_dir))
        if path is None or local_step != expected_step:
            raise ResumeGateError(f"the Hub's last-checkpoint is step {{step}}, training stopped at {{expected_step}}, "
                                  f"and no complete local checkpoint of that step exists (newest: {{path}} at {{local_step}})")
        print(f"⚠️ the Hub's last-checkpoint is step {{step}}, training stopped at {{expected_step}}: pushing {{path}} now", flush=True)
        upload_folder(repo_id=ckpt_repo, folder_path=path, path_in_repo="last-checkpoint",
                      commit_message=f"Training in progress, step {{expected_step}}, checkpoint (verified re-push)")
        step = hub_checkpoint_step(ckpt_repo)
    if step != expected_step:
        raise ResumeGateError(f"the Hub's last-checkpoint is step {{step}} after the push, expected {{expected_step}}")
    print(f"Hub resume point verified: last-checkpoint is step {{step}}")


MAX_STEPS = {MAX_STEPS}
TIME_BUDGET_H = {TIME_BUDGET_H}
MAX_INSESSION_RETRIES = {MAX_INSESSION_RETRIES}
ORDER_SEED = {ORDER_SEED}
print(f"session helpers ready: MAX_STEPS={{MAX_STEPS}}, time budget {{TIME_BUDGET_H}} h, in-session retries {{MAX_INSESSION_RETRIES}}")
'''
    return glue.replace("{body}", body)


def derive(nb: Dict, revision: str) -> Dict:
    """Apply the v22b edits to a loaded v22a notebook.

    :param nb: Notebook JSON (not modified in place).
    :type nb: Dict
    :param revision: 40-hex dataset revision to pin (or ``PIN-AFTER-PUSH``).
    :type revision: str
    :return: The v22b notebook JSON.
    :rtype: Dict
    """
    nb = copy.deepcopy(nb)
    cells = nb["cells"]
    # --- data cell -----------------------------------------------------------------------------
    c2 = "".join(cells[2]["source"])
    c2 = _replace_once(c2, 'V22_REPO = "isaacmg/genizah_v22_pilot"', f'V22_REPO = "{FULL_REPO}"', "repo")
    c2 = _replace_line(c2, r'^V22_REVISION = "', f'V22_REVISION = "{revision}"   # v22-full mixture: rows + images_part*.tar (20,000 train rows, 7 components)', "revision")
    c2 = _replace_once(c2, 'LOCAL = "/content/genizah_v22_pilot"', 'LOCAL = "/content/genizah_v22_full"', "local dir")
    c2 = _replace_once(c2, 'assert len(train_ds) >= 7500 and len(eval_ds) >= 150, "pilot mixture incomplete"',
                       f'assert len(train_ds) >= {MIN_TRAIN_ROWS} and len(eval_ds) >= 150, "full mixture incomplete"', "size gate")
    c2 = _replace_once(c2, 'for _s, _lo in (("pgp_qa", 0.17), ("pgp_editions", 0.22), ("documentary_grounding", 0.08)):',
                       f'for _s, _lo in {SHARE_FLOORS}:', "share floors")
    cells[2]["source"] = c2.splitlines(keepends=True)
    # --- training cell -------------------------------------------------------------------------
    c7 = "".join(cells[7]["source"])
    c7 = _replace_once(c7, "# Cell 6 — v2.2a training:", "# Cell 6 — v2.2b training:", "cell title")
    c7 = _replace_once(c7, 'CKPT_REPO = "isaacmg/qwen3-vl-8b-hebrew-v22a-ckpt"', f'CKPT_REPO = "{CKPT_REPO}"', "ckpt repo")
    c7 = _replace_once(c7, 'OUT_DIR = "outputs_v22a"', 'OUT_DIR = "outputs_v22b"', "out dir")
    c7 = _replace_line(c7, r'^\s*max_steps=2000,', f'        max_steps=MAX_STEPS,               # 20,000 rows = one pass = {MAX_STEPS} steps; sessions end on the time budget and resume exactly', "max_steps")
    c7 = _replace_once(c7, '        learning_rate=5e-5,', f'        learning_rate={LR},                # continued training on a converged model: adjust, do not re-carve (v22a lesson)', "lr")
    c7 = _replace_once(c7, 'run_name="genizah_v22a"', f'run_name="{RUN_NAME}"', "run name")
    c7 = _replace_once(c7, 'wandb.config.update({', f'wandb.init(project="qwen-hebrew-finetune", name="{RUN_NAME}", config={{"resume_step": _resume_step, ', "wandb init")
    c7 = _replace_once(c7, '        hub_strategy="checkpoint", hub_private_repo=False,',
                       '        hub_strategy="checkpoint", hub_private_repo=False,\n'
                       '        hub_always_push=True,              # every save is pushed, also the time-budget stop that lands right after a regular save',
                       "hub always push")
    c7 = _replace_once(c7, OLD_DISCOVERY, NEW_DISCOVERY, "resume discovery")
    c7 = _replace_once(c7, OLD_TAIL, NEW_TAIL, "training tail")
    cells[7]["source"] = c7.splitlines(keepends=True)
    # --- intro cell: this notebook's own description (the pilot's table would be wrong here) -----
    if cells[0]["cell_type"] != "markdown":
        raise ValueError("cell 0 of the pilot notebook is expected to be the markdown intro")
    cells[0]["source"] = INTRO_MD.splitlines(keepends=True)
    # --- helper cell inserted before the training cell ------------------------------------------
    helper = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
              "source": session_helper_source().splitlines(keepends=True)}
    cells.insert(7, helper)
    # --- merged-model cell: names -----------------------------------------------------------------
    c9 = "".join(cells[9]["source"])
    c9 = c9.replace('"v22a-merged"', '"v22b-merged"')
    cells[9]["source"] = c9.splitlines(keepends=True)
    return nb


def main(argv: Optional[List[str]] = None) -> None:
    """CLI.

    :param argv: Arguments.
    :type argv: Optional[List[str]]
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--revision", required=True, help="40-hex dataset revision, or PIN-AFTER-PUSH")
    ap.add_argument("--src", type=Path, default=HERE / "genizah_v22a.ipynb")
    ap.add_argument("--out", type=Path, default=HERE / "genizah_v22b.ipynb")
    args = ap.parse_args(argv)
    if args.revision != "PIN-AFTER-PUSH" and not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise SystemExit("revision must be a 40-hex sha or PIN-AFTER-PUSH")
    nb = json.load(open(args.src, encoding="utf-8"))
    out = derive(nb, args.revision)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {args.out} ({len(out['cells'])} cells; revision {args.revision})")


if __name__ == "__main__":
    main()
