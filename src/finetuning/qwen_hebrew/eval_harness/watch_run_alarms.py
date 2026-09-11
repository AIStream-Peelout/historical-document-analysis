# File name: watch_run_alarms.py
# Date: 9/11/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Poll a W&B fine-tune run and ALARM on the two signals that would have caught the v21 image-blind run.

Alarm A (train-loss LEVEL): the first logged train loss is more than ``--level-factor`` times the reference
run's first logged train loss. Alarm B (warm-start regression): eval/loss exceeds ``--warm-eval`` by more
than ``--eval-margin`` on two consecutive evals. Both print loudly and set a non-zero exit code when the
watch ends; nothing is stopped automatically (the Colab kernel is the user's).

Usage (repo root):
    .venv/bin/python -m src.finetuning.qwen_hebrew.eval_harness.watch_run_alarms \\
        --run genizah_v21b --reference genizah_v20a --warm-eval 0.6733 --hours 8
    # self-test on the dead run (both alarms must fire):
    .venv/bin/python -m src.finetuning.qwen_hebrew.eval_harness.watch_run_alarms \\
        --run genizah_v21 --reference genizah_v20a --warm-eval 0.6733 --once
"""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

PROJECT = "qwen-hebrew-finetune"


def _points(run: Any, key: str) -> List[Tuple[int, float]]:
    """(global_step, value) pairs of ``key`` for a W&B run, in step order.

    :param run: ``wandb.apis.public.Run``.
    :param key: History key, e.g. ``train/loss``.
    :returns: Sorted list of (step, value).
    """
    rows = run.history(keys=["train/global_step", key], pandas=False)
    pts = [(int(r["train/global_step"]), float(r[key])) for r in rows if r.get(key) is not None]
    return sorted(pts)


def find_run(api: Any, project: str, name: str) -> Any:
    """Newest run named ``name`` (or with that id) in ``project``.

    :param api: ``wandb.Api``.
    :param project: ``entity/project`` path.
    :param name: Run display name or id.
    :returns: The run.
    :raises LookupError: When no run matches.
    """
    for r in api.runs(project, order="-created_at"):
        if r.name == name or r.id == name:
            return r
    raise LookupError(f"no run named {name!r} in {project}")


def check_level(train: List[Tuple[int, float]], ref_first: Optional[float], factor: float) -> Optional[str]:
    """Alarm A text when the first train-loss point is above ``factor`` × the reference's first point.

    :param train: Train-loss points of the watched run.
    :param ref_first: Reference run's first train loss (None disables the check).
    :param factor: Multiplicative threshold.
    :returns: Alarm message or None.
    """
    if not train or ref_first is None:
        return None
    step, val = train[0]
    if val > factor * ref_first:
        return (f"ALARM A — train loss {val:.3f} at step {step} is {val / ref_first:.1f}× the reference's first "
                f"point ({ref_first:.3f}); the model is probably not seeing its inputs. STOP THE RUN and gate the "
                f"prepared dataloader (docs/postmortem_v21_dispatch_truncation.md).")
    return None


def check_warm_regression(evals: List[Tuple[int, float]], warm_eval: Optional[float], margin: float) -> Optional[str]:
    """Alarm B text when two consecutive evals sit above the warm-start checkpoint's eval by ``margin``.

    :param evals: Eval-loss points of the watched run.
    :param warm_eval: Eval loss of the warm-start checkpoint (None disables the check).
    :param margin: Tolerated excess.
    :returns: Alarm message or None.
    """
    if warm_eval is None or len(evals) < 2:
        return None
    for (s1, v1), (s2, v2) in zip(evals, evals[1:]):
        if v1 > warm_eval + margin and v2 > warm_eval + margin:
            return (f"ALARM B — eval {v1:.4f}@{s1} and {v2:.4f}@{s2} both exceed the warm-start checkpoint's "
                    f"{warm_eval:.4f} by > {margin}; a warm start from the best checkpoint is regressing. STOP THE RUN.")
    return None


def main() -> int:
    """Poll until the run ends, the window expires, or ``--once``; exit 2 if any alarm fired."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="watched run name or id")
    ap.add_argument("--reference", default=None, help="reference run name/id for the train-loss level check")
    ap.add_argument("--warm-eval", type=float, default=None, help="eval/loss of the warm-start checkpoint")
    ap.add_argument("--level-factor", type=float, default=2.0)
    ap.add_argument("--eval-margin", type=float, default=0.02)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--interval", type=int, default=180, help="poll seconds")
    ap.add_argument("--once", action="store_true", help="single pass (self-test / historical run)")
    a = ap.parse_args()
    load_dotenv(Path(__file__).resolve().parents[4] / ".env")
    import wandb
    api = wandb.Api(timeout=60)
    project = f"{api.default_entity}/{PROJECT}"
    ref_first: Optional[float] = None
    if a.reference:
        ref_train = _points(find_run(api, project, a.reference), "train/loss")
        ref_first = ref_train[0][1] if ref_train else None
        print(f"reference {a.reference}: first train loss {ref_first}", flush=True)
    seen_train, seen_eval, fired = 0, 0, set()
    deadline = time.time() + a.hours * 3600
    while True:
        try:
            run = find_run(api, project, a.run)
            train, evals = _points(run, "train/loss"), _points(run, "eval/loss")
        except LookupError as e:
            print(f"{e} — waiting", flush=True)
            if a.once:
                return 1
            time.sleep(a.interval)
            continue
        if len(train) > seen_train:
            s, v = train[-1]
            print(f"TRAIN step {s}: {v:.3f} (first {train[0][1]:.3f}@{train[0][0]}) | state {run.state}", flush=True)
            seen_train = len(train)
        for s, v in evals[seen_eval:]:
            print(f"EVAL step {s}: {v:.4f}", flush=True)
        seen_eval = len(evals)
        for msg in (check_level(train, ref_first, a.level_factor), check_warm_regression(evals, a.warm_eval, a.eval_margin)):
            if msg and msg[:7] not in fired:
                fired.add(msg[:7])
                print("\n" + "!" * 100 + f"\n{msg}\n" + "!" * 100 + "\n", flush=True)
        if a.once or run.state != "running" or time.time() > deadline:
            print(f"watch ended: state={run.state} train points={seen_train} eval points={seen_eval} alarms={sorted(fired)}", flush=True)
            return 2 if fired else 0
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
