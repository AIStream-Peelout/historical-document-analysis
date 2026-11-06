# File name: derive_novel_v1.py
# Date: 11/3/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Derive the novel_v1 notebook (novel pages only, warm start v2.5a step 1200) and build its training rows.

novel_v1 trains on one dataset: the novel-rows build (``src/finetuning/qwen_hebrew/novel_build/build.py``): page rows
of manuscript sides whose documents no earlier fine-tune trained on, in the KTIV ``FEATURES`` schema, splits
``train_page`` and ``val``, section ``novel_page``, the script-neutral prompt on every row, and its images-once export.
Nothing else is mixed in: the replay quota is 0, no outside set is added and no script is upweighted.

**Rows** (subcommand ``rows``, :func:`build_rows`; ``plan`` previews them without writing): v2.5a's pass rule applied
to the dataset's train pages, with ``label_source`` as the sources (:func:`pass_plan`): every train page 2 or 3 times,
no source above 2.5 passes. The extra (third) rows are drawn uniformly at random over all train pages of every source
together (one seeded permutation of the pages, ``novel_v1/extras``; the per-source 2.5-pass cap is only a guard), so
every source and every script keeps its natural share of the rows up to sampling noise: nothing is upweighted. (v2.5a
handed the extras out round-robin by source, which gives a small source 2.5 passes while a large one stays near 2.1.)
``auto`` (:func:`auto_rows`) takes 2.2 x the train pages rounded to the nearest whole step (8 rows: batch 1 x
accumulation 8) inside the pass-rule range; ``--train-rows`` sets the rows explicitly, checked against the same range.
Val is the dataset's ``val`` split, every row once. Every question is set to ``prompts.SCRIPT_NEUTRAL_PROMPT`` and no
question may name a script. :func:`script_table` measures how far each script's share of the rows is from its share of
the pages; the round's gates refuse a drift above :data:`SCRIPT_SHARE_TOLERANCE` (a tiny set may need a wider one).
Rows of Russian National Library documents are left out unless ``--keep-rnl`` is given (that holder's terms: training only,
never in a published dataset). Written::

    <out>/rows/train.parquet, rows/val.parquet   images-once rows plus a ``source`` column (= ``label_source``)
    <out>/images/<sha1>.jpg                      the images those rows use
    <out>/mixture.json                           the plan (``config``): pages, rows, passes per source, scripts, val
    <out>/manifest.json                          images-once manifest (splits train and val)
    <out>/stats.json, README.md, PRIVATE.md      the dataset's stats and notice, the card of these rows
    <out>/documents.jsonl, build_record.json     local records (the round's push never uploads them): the document
                                                 behind every row, the inputs and their digests

**The notebook** (subcommand ``notebook``; only the Lambda edition is written, with ``lambda_setup.sh`` beside it) is
v2.5a's training stack (:func:`derive_v23a.derive` on the v2.2b notebook with a novel_v1 :class:`derive_v23a.Arm`,
then asserted edits): dataset ``isaacmg/genizah_novel_v1`` (the data cell **refuses to run** until the revision is
pinned); warm start v2.5a step 1200 by commit, weights only, checked against the checkpoint's ``trainer_state.json``;
LR 5e-5 on a fresh cosine, 3 % warm-up, seed 3407, LoRA r16 on tower and language, merger frozen; 50-step saves, evals
every 100 steps; checkpoints pushed and confirmed on the Hub at the quarter points of the run rounded up to a multiple
of 50 (at least 100; the last is the final step, :func:`milestone_steps`); the data-cell gates of this mixture (rows and
pages per source as planned, the pass rule, the neutral prompt, novel page rows only, answer hygiene, the scripts
printed with no floors); a guardrail batch of two novel pages; a step-0 ``trainer.evaluate()`` of the warm start
(logged to W&B; its value is stored in the checkpoint repository for the warm-start stop rule); a train-loss alarm:
the first logged train loss above ``LOSS_ALARM_FACTOR`` x max(``LOSS_ALARM_REFERENCE``, the step-0 eval loss) stops the
run (:func:`loss_alarm_threshold`; 1.20 is the floor); the Lambda safety
stop 1.3 x one pass at the measured step time; merged model ``isaacmg/qwen3-vl-8b-hebrew-novel-v1`` (private).

Usage::

    .venv/bin/python -m src.finetuning.qwen_hebrew.colab.derive_novel_v1 plan --export <dataset>_images_once [--train-rows N]
    .venv/bin/python -m src.finetuning.qwen_hebrew.colab.derive_novel_v1 rows --export <dataset>_images_once --out DIR
    .venv/bin/python -m src.finetuning.qwen_hebrew.colab.derive_novel_v1 notebook --revision <40-hex sha | PIN-AFTER-PUSH> \\
        (--plan DIR/mixture.json | --pages N [--train-rows N]) [--out-dir DIR]
    .venv/bin/python -m src.finetuning.qwen_hebrew.colab.derive_novel_v1 show --notebook NB [--field eval_steps] [--require-pinned]
"""
import argparse
import hashlib
import json
import math
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pyarrow as pa

from src.finetuning.qwen_hebrew import build_v22_mixture as bvm
from src.finetuning.qwen_hebrew.colab import derive_v23a as dv
from src.finetuning.qwen_hebrew.colab import derive_v24a as d4
from src.finetuning.qwen_hebrew.colab import host_lambda
from src.finetuning.qwen_hebrew.colab.derive_v22b import _replace_line, _replace_once
from src.finetuning.qwen_hebrew.images_once import SHA_COLUMN
from src.finetuning.qwen_hebrew.novel_build import build as novel_build
from src.finetuning.qwen_hebrew.novel_build.pool import is_rnl
from src.finetuning.qwen_hebrew.prompts import SCRIPT_NEUTRAL_PROMPT

HERE = Path(__file__).resolve().parent
SRC_NOTEBOOK = HERE / "genizah_v22b.ipynb"

# ----------------------------------------------------------------------------- names (one distinct name per run)
ROUND = "novel_v1"
TAG = "novel-v1"                       # derive_v23a.Arm tag: checkpoint repo, output and resume folders, merged folder
RUN_NAME = "genizah_novel_v1"          # trainer and W&B run name (W&B project qwen-hebrew-finetune)
DATA_REPO = "isaacmg/genizah_novel_v1"
LOCAL_LEAF = "genizah_novel_v1"
COLAB_NOTEBOOK = "genizah_novel_v1.ipynb"                       # the in-memory Colab edition (never written)
NOTEBOOK, SETUP_SCRIPT = host_lambda.lambda_files(COLAB_NOTEBOOK)   # genizah_novel_v1_lambda.ipynb, lambda_setup.sh
CKPT_REPO = f"isaacmg/qwen3-vl-8b-hebrew-{TAG}-ckpt"
MERGED_REPO = "isaacmg/qwen3-vl-8b-hebrew-novel-v1"
PLACEHOLDER = d4.PLACEHOLDER           # dataset revision before the push: the notebook refuses to run
PUSH_SCRIPT = "logs/next_round/novel_v1/push_novel_v1.sh"

# ----------------------------------------------------------------------------- warm start: v2.5a step 1200
# The v2.5a eval chain staged "v25a step 1200" from this commit of the checkpoint repo (the trainer's commit
# "Training in progress, step 1200, checkpoint"); the model cell checks trainer_state.json says step 1200.
WARM_PLACEHOLDER = "PIN-BEFORE-LAUNCH"
WARM_CKPT_REPO = "isaacmg/qwen3-vl-8b-hebrew-v25a-ckpt"
WARM_REVISION = "0a0e7e760f32ca19ea082a9266bfe54984d8c72d"
WARM_STEP = 1200

# ----------------------------------------------------------------------------- schedule (v2.5a's)
LR = "5e-5"
WARMUP_RATIO = "0.03"
SEED = d4.SEED                         # 3407
ROWS_PER_STEP = 8                      # batch 1 x accumulation 8
SAVE_STEPS = dv.SAVE_STEPS             # 50
EVAL_EVERY = 100                       # the trainer's eval_steps
MILESTONE_GRAIN = 50                   # milestones are quarter points rounded up to a multiple of this
MILESTONE_MIN = 100
MILESTONE_PARTS = 4
MEASURED_S_PER_STEP = d4.MEASURED_S_PER_STEP
BUDGET_FACTOR = d4.BUDGET_FACTOR
#: the train-loss alarm: v2.5a's first logged train loss (during its warm-up, on mostly seen rows) and the stop factor;
#: the threshold is LOSS_ALARM_FACTOR x max(LOSS_ALARM_REFERENCE, the warm start's step-0 eval loss)
LOSS_ALARM_REFERENCE = 0.60
LOSS_ALARM_FACTOR = 2.0
STEP0_FILE = "step0_eval_loss.json"    # the warm start's step-0 eval loss, stored in the checkpoint repo

# ----------------------------------------------------------------------------- the pass plan (v2.5a's)
COMPONENT = ROUND                      # names the seeded page order of each source (``novel_v1/<label_source>``)
AUTO = "auto"
AUTO_PASSES = Fraction(11, 5)          # auto rows = 2.2 x train pages, to the nearest whole step
EXTRAS_DRAW = f"{COMPONENT}/extras"     # the seeded draw of the pages that get a third row
MIN_PASSES = 2                         # every train page at least twice
MAX_PASSES = 2.5                       # no source above 2.5 passes (a page: 2 or 3 rows)
#: largest gap between a script's share of the train rows and its share of the train pages the gates accept
SCRIPT_SHARE_TOLERANCE = 0.005
#: train pages of the reference plan of the committed placeholder notebook (a pinned notebook carries the built plan)
REFERENCE_PAGES = 1000

# ----------------------------------------------------------------------------- the dataset
TRAIN_SPLIT = novel_build.SPLIT_NAMES["train"]     # train_page
VAL_SPLIT = novel_build.SPLIT_NAMES["val"]         # val
SECTION = novel_build.SECTION                      # novel_page
TASK = novel_build.TASK                            # fragment_transcribe
GROUP_COLUMN = "label_source"                      # the pass plan's sources
SOURCE_COLUMN = bvm.SOURCE_COLUMN                  # the column the rows step adds (= label_source)
SCRIPT_CLAUSES: Tuple[str, ...] = ("Arabic script", "Hebrew script")
SCRIPTS: Tuple[str, ...] = ("hebrew", "arabic", "mixed")
HARAKAT = re.compile(r"[\u0640\u064b-\u0652\u0670]")
FORBIDDEN_CARD: Tuple[str, ...] = tuple(dict.fromkeys((*bvm.FORBIDDEN_CARD_STRINGS, *novel_build.FORBIDDEN_README,
                                                       "genizah.org")))
PRIVATE_BANNER = ("**PRIVATE. Never publish or release these rows, their images or anything derived from them; they go to the "
                  "Hub only as the private repository the training run reads.**")


# ----------------------------------------------------------------------------- the pass plan


def extra_cap(pages: int) -> int:
    """Pages of a source that may get a third row.

    :param pages: Train pages of the source.
    :type pages: int
    :return: ``floor((MAX_PASSES - MIN_PASSES) x pages)``.
    :rtype: int
    """
    return int(math.floor((MAX_PASSES - MIN_PASSES) * pages + 1e-9))


def feasible_rows(pages: Mapping[str, int]) -> Tuple[int, int]:
    """The row counts the pass rule can meet with these pages.

    :param pages: source -> train pages.
    :type pages: Mapping[str, int]
    :return: (every page twice, every source at its extra cap).
    :rtype: Tuple[int, int]
    """
    low = sum(MIN_PASSES * n for n in pages.values())
    return low, low + sum(extra_cap(n) for n in pages.values())


def step_rows(pages: Mapping[str, int]) -> Tuple[int, int]:
    """The whole-step row counts inside the pass-rule range.

    :param pages: source -> train pages.
    :type pages: Mapping[str, int]
    :return: (lowest, highest) multiple of :data:`ROWS_PER_STEP` in :func:`feasible_rows`.
    :rtype: Tuple[int, int]
    :raises ValueError: When there are no pages or the range holds no multiple of 8.
    """
    if sum(pages.values()) <= 0:
        raise ValueError(f"no train pages: {dict(pages)}")
    low, high = feasible_rows(pages)
    lo8, hi8 = -(-low // ROWS_PER_STEP) * ROWS_PER_STEP, high // ROWS_PER_STEP * ROWS_PER_STEP
    if lo8 > hi8:
        raise ValueError(f"the pass rule allows {low} to {high} rows for the pages {dict(pages)}: no multiple of {ROWS_PER_STEP}")
    return lo8, hi8


def auto_rows(pages: Mapping[str, int]) -> int:
    """The ``auto`` row count: 2.2 x the train pages rounded to the nearest whole step (8 rows), inside the pass-rule range.

    :param pages: source -> train pages.
    :type pages: Mapping[str, int]
    :return: Rows (a multiple of :data:`ROWS_PER_STEP`).
    :rtype: int
    :raises ValueError: From :func:`step_rows`.
    """
    lo8, hi8 = step_rows(pages)
    return min(max(ROWS_PER_STEP * round(AUTO_PASSES * sum(pages.values()) / ROWS_PER_STEP), lo8), hi8)


def is_auto(request: Union[str, int]) -> bool:
    """Whether a rows request is ``auto``.

    :param request: ``auto`` (any case) or a number.
    :type request: Union[str, int]
    :return: True for ``auto``.
    :rtype: bool
    """
    return str(request).strip().lower() == AUTO


def resolve_rows(request: Union[str, int], pages: Mapping[str, int]) -> int:
    """The train rows for a request: :func:`auto_rows`, or the number given, checked against :func:`step_rows`.

    :param request: ``auto`` or a whole number of rows.
    :type request: Union[str, int]
    :param pages: source -> train pages.
    :type pages: Mapping[str, int]
    :return: Rows.
    :rtype: int
    :raises ValueError: On a request that is neither, or rows that are not a multiple of 8 inside the range.
    """
    if is_auto(request):
        return auto_rows(pages)
    if not re.fullmatch(r"\d+", str(request).strip()):
        raise ValueError(f"train rows must be '{AUTO}' or a whole number, got {request!r}")
    rows = int(str(request).strip())
    lo8, hi8 = step_rows(pages)
    if rows % ROWS_PER_STEP or not lo8 <= rows <= hi8:
        raise ValueError(f"{rows} train rows: the pages {dict(pages)} allow multiples of {ROWS_PER_STEP} from {lo8} to {hi8} "
                         f"(every page {MIN_PASSES} or {MIN_PASSES + 1} times, no source above {MAX_PASSES} passes)")
    return rows


def pass_plan(pages: Mapping[str, int], rows: int, seed: int = SEED) -> Dict[str, np.ndarray]:
    """Rows per page of every source: :data:`MIN_PASSES` each, the extra rows drawn uniformly over all pages together.

    Every page gets two rows. The pages of every source, in ``pages`` order and each source's own order, form one list;
    one seeded permutation of it (``build_v22_mixture.component_rng(seed, "novel_v1/extras")``) is walked from the start
    and each page met gets a third row until ``rows`` are given, so every page has the same chance of a third row
    whatever its source. The per-source cap (:func:`extra_cap`, at most :data:`MAX_PASSES`) is a guard only: a page of a
    source already at its cap is skipped. No page gets more than three rows.

    :param pages: source -> train pages (the order of the list the permutation runs over).
    :type pages: Mapping[str, int]
    :param rows: Rows wanted (:func:`resolve_rows`).
    :type rows: int
    :param seed: Build seed.
    :type seed: int
    :return: source -> rows per page (int64, the source's page order).
    :rtype: Dict[str, np.ndarray]
    :raises ValueError: When the rows are outside :func:`feasible_rows`.
    """
    low, high = feasible_rows(pages)
    if not low <= rows <= high:
        raise ValueError(f"{rows} rows cannot be met with every page {MIN_PASSES} or {MIN_PASSES + 1} times and no source above "
                         f"{MAX_PASSES} passes: the pages {dict(pages)} allow {low} to {high}")
    counts = {s: np.full(n, MIN_PASSES, dtype=np.int64) for s, n in pages.items()}
    caps = {s: extra_cap(n) for s, n in pages.items()}
    given = {s: 0 for s in pages}
    index = [(s, i) for s, n in pages.items() for i in range(n)]
    left = rows - low
    for k in bvm.component_rng(seed, EXTRAS_DRAW).permutation(len(index)):
        if not left:
            break
        s, i = index[k]
        if given[s] < caps[s]:                   # the cap only guards: a page of a source at its cap is skipped
            counts[s][i] += 1
            given[s] += 1
            left -= 1
    return counts


def page_counts(stems_by_source: Mapping[str, Sequence[str]], rows: int, seed: int = SEED) -> Dict[str, int]:
    """Rows of every train page under :func:`pass_plan`, the pages listed by sorted source, then sorted stem.

    :param stems_by_source: source -> stems of its train pages.
    :type stems_by_source: Mapping[str, Sequence[str]]
    :param rows: Train rows.
    :type rows: int
    :param seed: Build seed.
    :type seed: int
    :return: stem -> rows (2 or 3).
    :rtype: Dict[str, int]
    :raises ValueError: When a stem is listed twice.
    """
    ordered = {s: sorted(stems_by_source[s]) for s in sorted(stems_by_source)}
    stems = [st for group in ordered.values() for st in group]
    if len(set(stems)) != len(stems):
        raise ValueError(f"a train page is listed twice: {sorted(st for st, n in Counter(stems).items() if n > 1)[:5]}")
    counts = pass_plan({s: len(group) for s, group in ordered.items()}, rows, seed)
    return {st: int(n) for s, group in ordered.items() for st, n in zip(group, counts[s])}


def script_table(scripts: Mapping[str, str], counts: Mapping[str, int]) -> Dict[str, Dict[str, Any]]:
    """Train pages and train rows per script, their shares and the drift of the row share from the page share.

    :param scripts: stem -> script measured on its target (``hebrew`` / ``arabic`` / ``mixed``).
    :type scripts: Mapping[str, str]
    :param counts: stem -> rows of every train page.
    :type counts: Mapping[str, int]
    :return: script -> ``{"pages", "rows", "page_share", "row_share", "drift"}`` (shares and drift as fractions).
    :rtype: Dict[str, Dict[str, Any]]
    """
    pages, rows = Counter(), Counter()
    for stem, n in counts.items():
        pages[scripts[stem]] += 1
        rows[scripts[stem]] += n
    n_pages, n_rows = sum(pages.values()), sum(rows.values())
    names = [s for s in SCRIPTS if s in pages] + sorted(set(pages) - set(SCRIPTS))
    return {s: {"pages": pages[s], "rows": rows[s], "page_share": round(pages[s] / n_pages, 6),
                "row_share": round(rows[s] / n_rows, 6), "drift": round(rows[s] / n_rows - pages[s] / n_pages, 6)}
            for s in names}


def script_drift_problems(table: Mapping[str, Mapping[str, Any]], tolerance: float = SCRIPT_SHARE_TOLERANCE) -> List[str]:
    """Scripts whose share of the train rows is more than ``tolerance`` away from their share of the train pages.

    :param table: :func:`script_table`.
    :type table: Mapping[str, Mapping[str, Any]]
    :param tolerance: Largest accepted gap (a fraction: 0.005 = half a percentage point).
    :type tolerance: float
    :return: One message per script out of tolerance.
    :rtype: List[str]
    """
    return [f"{s}: {v['row_share']:.2%} of the train rows against {v['page_share']:.2%} of the train pages "
            f"(drift {v['drift'] * 100:+.2f} points, tolerance {tolerance * 100:.2f})"
            for s, v in table.items() if abs(v["drift"]) > tolerance + 1e-12]


def rows_within_tolerance(stems_by_source: Mapping[str, Sequence[str]], scripts: Mapping[str, str], seed: int = SEED,
                          tolerance: float = SCRIPT_SHARE_TOLERANCE) -> List[int]:
    """The whole-step row counts whose pass plan keeps every script within ``tolerance`` of its page share.

    :param stems_by_source: source -> stems of its train pages.
    :type stems_by_source: Mapping[str, Sequence[str]]
    :param scripts: stem -> script.
    :type scripts: Mapping[str, str]
    :param seed: Build seed.
    :type seed: int
    :param tolerance: Largest accepted drift (:func:`script_drift_problems`).
    :type tolerance: float
    :return: Row counts, ascending (empty when none qualifies).
    :rtype: List[int]
    """
    lo8, hi8 = step_rows({s: len(v) for s, v in stems_by_source.items()})
    return [rows for rows in range(lo8, hi8 + 1, ROWS_PER_STEP)
            if not script_drift_problems(script_table(scripts, page_counts(stems_by_source, rows, seed)), tolerance)]


# ----------------------------------------------------------------------------- the schedule


def max_steps_for(train_rows: int) -> int:
    """Steps of one pass over the train rows.

    :param train_rows: Train rows (a multiple of :data:`ROWS_PER_STEP`).
    :type train_rows: int
    :return: ``train_rows // 8``.
    :rtype: int
    :raises ValueError: When the rows are not a positive multiple of 8.
    """
    if train_rows <= 0 or train_rows % ROWS_PER_STEP:
        raise ValueError(f"train rows {train_rows} must be a positive multiple of {ROWS_PER_STEP}")
    return train_rows // ROWS_PER_STEP


def warmup_steps(train_rows: int) -> int:
    """Warm-up steps the Trainer derives from the ratio (``ceil(max_steps x warmup_ratio)``).

    :param train_rows: Train rows.
    :type train_rows: int
    :return: Steps of linear warm-up.
    :rtype: int
    """
    return math.ceil(max_steps_for(train_rows) * float(WARMUP_RATIO))


def lambda_budget_h(train_rows: int) -> float:
    """The Lambda session's safety stop: :data:`BUDGET_FACTOR` x one pass at the measured step time, rounded up.

    :param train_rows: Train rows.
    :type train_rows: int
    :return: Hours.
    :rtype: float
    """
    return float(math.ceil(BUDGET_FACTOR * max_steps_for(train_rows) * MEASURED_S_PER_STEP / 3600))


def milestone_steps(max_steps: int) -> Tuple[int, ...]:
    """Steps whose checkpoint the notebook pushes and confirms on the Hub before training goes on.

    The quarter points of the run rounded up to a multiple of :data:`MILESTONE_GRAIN`, at least :data:`MILESTONE_MIN`
    and at most the last step (the trainer saves the last step whatever the save interval).

    :param max_steps: Steps of the run.
    :type max_steps: int
    :return: Sorted distinct steps; the last is ``max_steps``.
    :rtype: Tuple[int, ...]
    :raises ValueError: On a run without steps.
    """
    if max_steps <= 0:
        raise ValueError(f"a run of {max_steps} steps has no milestones")
    steps = {min(max(math.ceil(Fraction(k * max_steps, MILESTONE_PARTS) / MILESTONE_GRAIN) * MILESTONE_GRAIN,
                     MILESTONE_MIN), max_steps)
             for k in range(1, MILESTONE_PARTS + 1)}
    return tuple(sorted(steps))


def eval_steps(max_steps: int) -> Tuple[int, ...]:
    """The checkpoints the Studio eval chain scores: every milestone after the first (the half, three-quarter and final
    points of the run), or the one milestone of a short run.

    :param max_steps: Steps of the run.
    :type max_steps: int
    :return: Steps.
    :rtype: Tuple[int, ...]
    """
    milestones = milestone_steps(max_steps)
    return milestones[1:] if len(milestones) > 1 else milestones


def loss_alarm_threshold(step0_eval_loss: float) -> float:
    """The train-loss alarm's stop level, as the notebook computes it after the step-0 eval.

    :param step0_eval_loss: The warm start's eval loss on the val split at step 0.
    :type step0_eval_loss: float
    :return: ``LOSS_ALARM_FACTOR x max(LOSS_ALARM_REFERENCE, step0_eval_loss)`` (at least 1.20).
    :rtype: float
    """
    return LOSS_ALARM_FACTOR * max(LOSS_ALARM_REFERENCE, float(step0_eval_loss))


def make_arm(train_rows: int) -> dv.Arm:
    """The run as a :class:`derive_v23a.Arm` (its share floors are replaced by this mixture's gates).

    :param train_rows: Train rows.
    :type train_rows: int
    :return: Tag ``novel-v1``: checkpoint repo ``isaacmg/qwen3-vl-8b-hebrew-novel-v1-ckpt``, run ``genizah_novel_v1``.
    :rtype: dv.Arm
    """
    max_steps_for(train_rows)
    return dv.Arm(TAG, RUN_NAME, "train.parquet", train_rows, "()", COLAB_NOTEBOOK)


def warm_start_pinned(revision: str = WARM_REVISION) -> bool:
    """Whether the warm start is pinned to a commit.

    :param revision: Warm-start revision.
    :type revision: str
    :return: True for a 40-hex sha.
    :rtype: bool
    """
    return bool(re.fullmatch(r"[0-9a-f]{40}", revision))


# ----------------------------------------------------------------------------- the rows step


def read_json(path: Path) -> Any:
    """A JSON file.

    :param path: File.
    :type path: Path
    :return: Its content.
    :rtype: Any
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    """sha256 of a file, streamed.

    :param path: File.
    :type path: Path
    :return: Hex digest.
    :rtype: str
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_export(export_dir: Path) -> Tuple[pa.Table, pa.Table, Dict[str, Any]]:
    """Train and val rows of the novel dataset's images-once export.

    :param export_dir: The export (``rows/train_page.parquet``, ``rows/val.parquet``, ``images/``, ``manifest.json``).
    :type export_dir: Path
    :return: (train rows, val rows in the train layout, the images-once metadata).
    :rtype: Tuple[pa.Table, pa.Table, Dict[str, Any]]
    :raises ValueError: When a split is missing or the layouts differ.
    """
    bvm.check_export(ROUND, export_dir)
    splits = list(read_json(export_dir / "manifest.json")["splits"])
    missing = [s for s in (TRAIN_SPLIT, VAL_SPLIT) if s not in splits]
    if missing:
        raise ValueError(f"{export_dir} lacks the split(s) {missing} (it has {splits})")
    train, meta = bvm.read_rows(export_dir, [TRAIN_SPLIT])
    val, val_meta = bvm.read_rows(export_dir, [VAL_SPLIT])
    bvm.check_layouts([meta, val_meta], [f"{export_dir.name}/{TRAIN_SPLIT}", f"{export_dir.name}/{VAL_SPLIT}"])
    return train, bvm.conform(val, train.schema), meta


def dataset_manifest_path(export_dir: Path) -> Path:
    """The dataset's ``manifest.jsonl``: in the saved dataset the export was made from.

    :param export_dir: The images-once export (its ``manifest.json`` names its source).
    :type export_dir: Path
    :return: ``<source dataset>/manifest.jsonl``.
    :rtype: Path
    """
    return Path(read_json(export_dir / "manifest.json")["source"]) / "manifest.jsonl"


def read_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    """Per-row records of the novel dataset (``schema.RowRecord`` plus ``rnl`` and ``shelfmark``), by stem.

    :param path: ``manifest.jsonl``.
    :type path: Path
    :return: stem -> record.
    :rtype: Dict[str, Dict[str, Any]]
    :raises ValueError: When a stem is listed twice.
    """
    records: Dict[str, Dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record["stem"] in records:
                raise ValueError(f"{path}: stem {record['stem']!r} listed twice")
            records[record["stem"]] = record
    return records


def script_clause_hits(questions: Sequence[str]) -> List[str]:
    """Questions that name a script.

    :param questions: Questions.
    :type questions: Sequence[str]
    :return: The distinct questions holding one of :data:`SCRIPT_CLAUSES`.
    :rtype: List[str]
    """
    return sorted({q for q in questions if any(clause in q for clause in SCRIPT_CLAUSES)})


def neutral_questions(table: pa.Table) -> Tuple[pa.Table, int]:
    """Set every row's question to the script-neutral prompt and check that no question names a script.

    :param table: Rows (``question``).
    :type table: pa.Table
    :return: (rows, how many questions were something else before).
    :rtype: Tuple[pa.Table, int]
    :raises ValueError: When a question still names a script (the neutral prompt itself would have to).
    """
    old = table["question"].to_pylist()
    i = table.schema.get_field_index("question")
    table = table.set_column(i, table.schema.field(i), pa.array([SCRIPT_NEUTRAL_PROMPT] * table.num_rows, table.schema.field(i).type))
    hits = script_clause_hits(table["question"].to_pylist())
    if hits:
        raise ValueError(f"a question names a script after the rewrite: {hits[:1]}")
    return table, sum(q != SCRIPT_NEUTRAL_PROMPT for q in old)


def rnl_record(record: Mapping[str, Any]) -> bool:
    """Whether a manifest record is a Russian National Library document (the build's flag or its holder patterns).

    :param record: Manifest record.
    :type record: Mapping[str, Any]
    :return: True when ``rnl`` is set or ``novel_build.pool.is_rnl`` matches.
    :rtype: bool
    """
    return bool(record.get("rnl")) or is_rnl(dict(record))


def keep_rows(table: pa.Table, stems: set) -> pa.Table:
    """Rows whose stem is not in ``stems``.

    :param table: Rows.
    :type table: pa.Table
    :param stems: Stems to drop.
    :type stems: set
    :return: The other rows.
    :rtype: pa.Table
    """
    return table.filter(pa.array([s not in stems for s in table["stem"].to_pylist()], pa.bool_()))


def manifest_problems(train: pa.Table, val: pa.Table, manifest: Mapping[str, Mapping[str, Any]]) -> List[str]:
    """Why the export and the manifest disagree: rows without a record (or records without a row), a record in the other
    split, a ``label_source`` that is not the one of the record's ledger source.

    :param train: Train rows.
    :type train: pa.Table
    :param val: Val rows.
    :type val: pa.Table
    :param manifest: :func:`read_manifest`.
    :type manifest: Mapping[str, Mapping[str, Any]]
    :return: Problems (empty: they agree).
    :rtype: List[str]
    """
    problems = []
    rows = {}
    for split, table in (("train", train), ("val", val)):
        for stem, label in zip(table["stem"].to_pylist(), table[GROUP_COLUMN].to_pylist()):
            rows[stem] = (split, label)
    unlisted = sorted(set(rows) - set(manifest))
    extra = sorted(set(manifest) - set(rows))
    if unlisted:
        problems.append(f"{len(unlisted)} rows have no manifest record, e.g. {unlisted[:3]}")
    if extra:
        problems.append(f"{len(extra)} manifest records have no row in the export, e.g. {extra[:3]}")
    moved = sorted(s for s, (split, _) in rows.items() if s in manifest and manifest[s].get("split") != split)
    if moved:
        problems.append(f"{len(moved)} rows sit in another split than their manifest record, e.g. {moved[:3]}")
    relabelled = sorted(s for s, (_, label) in rows.items()
                        if s in manifest and novel_build.LABEL_SOURCES.get(manifest[s].get("source")) != label)
    if relabelled:
        problems.append(f"{len(relabelled)} rows carry another label_source than their ledger source's, e.g. {relabelled[:3]}")
    return problems


@dataclass
class RowsPlan:
    """The training rows before anything is written (the ``plan`` preview and the ``rows`` step share it).

    :ivar train: Train rows: every page 2 or 3 times, sources in sorted order, ``source`` column added.
    :ivar val: Val rows, ``source`` column added.
    :ivar meta: images-once metadata of the rows (``source`` among the columns).
    :ivar record: The plan as ``mixture.json`` keeps it (``config``).
    :ivar documents: One record per train page and per val row (``documents.jsonl``).
    :ivar dropped: Manifest records of the rows left out as Russian National Library documents.
    :ivar stats: The dataset's ``stats.json``.
    :ivar counts: stem -> rows of every train page.
    """
    train: pa.Table
    val: pa.Table
    meta: Dict[str, Any]
    record: Dict[str, Any]
    documents: List[Dict[str, Any]]
    dropped: List[Dict[str, Any]]
    stats: Dict[str, Any]
    counts: Dict[str, int] = field(default_factory=dict)


def prepare_rows(export_dir: Path, request: Union[str, int] = AUTO, seed: int = SEED, keep_rnl: bool = False,
                 manifest_path: Optional[Path] = None) -> RowsPlan:
    """Read the export and its manifest, leave out RNL rows, set the neutral prompt, apply the pass plan (nothing written).

    :param export_dir: The novel dataset's images-once export.
    :type export_dir: Path
    :param request: ``auto`` or a whole number of train rows.
    :type request: Union[str, int]
    :param seed: Build seed.
    :type seed: int
    :param keep_rnl: Keep Russian National Library rows (default: left out and recorded).
    :type keep_rnl: bool
    :param manifest_path: The dataset's ``manifest.jsonl`` (None: :func:`dataset_manifest_path`).
    :type manifest_path: Optional[Path]
    :return: The plan.
    :rtype: RowsPlan
    :raises ValueError: When the export and its manifest disagree, a page has two train rows, a row is not a novel page
        row, or the rows request does not fit the pages.
    """
    train, val, meta = read_export(export_dir)
    stats = read_json(export_dir / "stats.json")
    manifest_path = manifest_path if manifest_path is not None else dataset_manifest_path(export_dir)
    manifest = read_manifest(manifest_path)
    problems = manifest_problems(train, val, manifest)
    if problems:
        raise ValueError(f"{export_dir} and {manifest_path.name} disagree: " + "; ".join(problems))
    kinds = {(s, t) for table in (train, val) for s, t in zip(table["section"].to_pylist(), table["task"].to_pylist())}
    if kinds != {(SECTION, TASK)}:
        raise ValueError(f"rows other than novel page transcriptions ({SECTION}, {TASK}): {sorted(kinds)}")
    dropped = [] if keep_rnl else [manifest[s] for s in train["stem"].to_pylist() + val["stem"].to_pylist() if rnl_record(manifest[s])]
    out = {d["stem"] for d in dropped}
    train, val = keep_rows(train, out), keep_rows(val, out)
    train, rewritten = neutral_questions(train)
    val, val_rewritten = neutral_questions(val)
    stems = train["stem"].to_pylist()
    labels = dict(zip(stems, train[GROUP_COLUMN].to_pylist()))
    if len(labels) != train.num_rows:
        raise ValueError(f"{export_dir}: a train page has more than one row (the pass rule counts pages)")
    by_source: Dict[str, List[str]] = {}
    for stem in stems:
        by_source.setdefault(labels[stem], []).append(stem)
    pages = {s: len(by_source[s]) for s in sorted(by_source)}
    rows = resolve_rows(request, pages)
    counts = page_counts(by_source, rows, seed)
    index = {stem: i for i, stem in enumerate(stems)}
    take = [index[stem] for s in sorted(by_source) for stem in sorted(by_source[s]) for _ in range(counts[stem])]
    train_rows = train.take(pa.array(take, pa.int64()))
    train_rows = bvm.with_source(train_rows, train_rows[GROUP_COLUMN].to_pylist())
    val_rows = bvm.with_source(val, val[GROUP_COLUMN].to_pylist())
    scripts = {stem: manifest[stem]["script"] for stem in stems}
    table = script_table(scripts, counts)
    rows_by_source = {s: sum(counts[st] for st in by_source[s]) for s in pages}
    low, high = feasible_rows(pages)
    steps = max_steps_for(rows)
    record = {
        "round": ROUND, "dataset": stats.get("name"), "dataset_built": stats.get("built"),
        "dataset_stats_sha256": file_sha256(export_dir / "stats.json"), "dataset_rows": stats.get("rows"),
        "rule": (f"every train page {MIN_PASSES} or {MIN_PASSES + 1} times, no source above {MAX_PASSES} passes; the extra rows "
                 f"drawn uniformly over all train pages together (one seeded permutation, {EXTRAS_DRAW}; the per-source "
                 f"cap only guards); sources = {GROUP_COLUMN}"),
        "requested": AUTO if is_auto(request) else int(str(request).strip()), "auto_passes": float(AUTO_PASSES),
        "rows_per_step": ROWS_PER_STEP, "seed": seed, "train_pages": sum(pages.values()), "pages_by_source": pages,
        "feasible_rows": [low, high], "step_rows": list(step_rows(pages)), "train_rows": rows, "max_steps": steps,
        "rows_by_source": rows_by_source,
        "passes_by_source": {s: round(rows_by_source[s] / pages[s], 4) for s in pages},
        "pages_by_rows": {str(k): v for k, v in sorted(Counter(counts.values()).items())},
        "val_rows": val_rows.num_rows, "val_by_source": dict(sorted(Counter(val_rows[SOURCE_COLUMN].to_pylist()).items())),
        "scripts": table, "val_scripts": dict(Counter(manifest[s]["script"] for s in val_rows["stem"].to_pylist())),
        "script_share_tolerance": SCRIPT_SHARE_TOLERANCE,
        "rows_within_script_tolerance": rows_within_tolerance(by_source, scripts, seed),
        "prompt": "SCRIPT_NEUTRAL_PROMPT", "questions_rewritten": {"train": rewritten, "val": val_rewritten},
        "section": SECTION, "task": TASK, "keep_rnl": keep_rnl,
        "rnl_dropped": {"train": sum(1 for d in dropped if d.get("split") == "train"),
                        "val": sum(1 for d in dropped if d.get("split") == "val")},
        "warm_start": f"{WARM_CKPT_REPO}@{WARM_REVISION} (step {WARM_STEP})",
        "milestones": list(milestone_steps(steps)), "eval_steps": list(eval_steps(steps)),
    }
    documents = [document_record(manifest[stem], "train", counts[stem]) for s in sorted(by_source) for stem in sorted(by_source[s])]
    documents += [document_record(manifest[stem], "val", 1) for stem in val_rows["stem"].to_pylist()]
    return RowsPlan(train_rows, val_rows, bvm.mixture_meta([meta], [export_dir.name]), record, documents, dropped, stats, counts)


def document_record(record: Mapping[str, Any], split: str, rows: int) -> Dict[str, Any]:
    """The ``documents.jsonl`` record of one row: its document, ledger verdict and script (no paths, no URLs).

    :param record: Manifest record.
    :type record: Mapping[str, Any]
    :param split: ``train`` or ``val``.
    :type split: str
    :param rows: Rows of the page in the split.
    :type rows: int
    :return: Record.
    :rtype: Dict[str, Any]
    """
    return {"stem": record["stem"], "split": split, "rows": rows, "row_id": record.get("row_id"),
            "label_source": novel_build.LABEL_SOURCES.get(record.get("source")),
            "document_group": record.get("document_group"), "canonical_ids": list(record.get("canonical_ids") or []),
            "pgpids": [str(p) for p in record.get("pgpids") or []], "shelfmark": record.get("shelfmark"),
            "script": record.get("script"), "verdict": record.get("verdict"), "rnl": rnl_record(record),
            "image_sha256": record.get("image_sha256")}


def card_text(record: Mapping[str, Any], stats: Mapping[str, Any]) -> str:
    """``README.md`` of the training rows (the dataset's PRIVATE banner first when the dataset is private).

    :param record: The plan (:attr:`RowsPlan.record`).
    :type record: Mapping[str, Any]
    :param stats: The dataset's ``stats.json``.
    :type stats: Mapping[str, Any]
    :return: Markdown.
    :raises ValueError: When the text names a forbidden source, host or storage path (:data:`FORBIDDEN_CARD`).
    """
    lines = [f"# {LOCAL_LEAF} (training rows of the {ROUND} fine-tune)", ""]
    if stats.get("private", True):
        lines += [f"{PRIVATE_BANNER} {'; '.join(stats.get('private_reasons') or [])}".rstrip(), ""]
    lines += [f"The rows a Qwen3-VL page reader trains on in the `{ROUND}` run, materialised from the novel-rows dataset "
              f"`{record['dataset']}` (built {record['dataset_built']}) by `src/finetuning/qwen_hebrew/colab/derive_novel_v1.py "
              "rows`: manuscript sides whose documents no earlier fine-tune trained on, nothing else (no replay, no outside "
              "sets, no script upweighted).", "",
              f"- train: {record['train_rows']:,} rows = {record['train_pages']:,} pages x 2 or 3 rows ({record['rule']}); "
              f"{record['max_steps']:,} steps at batch 1 x accumulation 8",
              f"- val: {record['val_rows']:,} rows, the dataset's own val split (no document group in both splits)",
              "- every question is the script-neutral prompt (`prompts.SCRIPT_NEUTRAL_PROMPT`); section "
              f"`{record['section']}`, task `{record['task']}`; the column `source` repeats `label_source`",
              f"- Russian National Library rows: {'kept (private use only)' if record['keep_rnl'] else 'left out'} "
              f"({record['rnl_dropped']['train']} train / {record['rnl_dropped']['val']} val left out)", "",
              "| source | train pages | train rows | passes | val rows |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {s} | {n} | {record['rows_by_source'][s]} | {record['passes_by_source'][s]:.2f} | {record['val_by_source'].get(s, 0)} |"
              for s, n in record["pages_by_source"].items()]
    lines += ["", "| script | train pages | page share | train rows | row share |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {s} | {v['pages']} | {v['page_share']:.2%} | {v['rows']} | {v['row_share']:.2%} |" for s, v in record["scripts"].items()]
    lines += ["", "`mixture.json` holds the plan; `stats.json` is the novel dataset's own; `manifest.json` lists the images-once "
                  "splits. Private, internal training material: no reproduction or redistribution of the text or the images."]
    text = "\n".join(lines) + "\n"
    check_card(text)
    return text


def check_card(text: str) -> None:
    """Refuse a card that names a forbidden source, host or storage path.

    :param text: Card text.
    :type text: str
    :raises ValueError: On a hit of :data:`FORBIDDEN_CARD`.
    """
    hits = [word for word in FORBIDDEN_CARD if word in text.lower()]
    if hits:
        raise ValueError(f"the card would name {hits}")


def build_rows(export_dir: Path, out_dir: Path, request: Union[str, int] = AUTO, seed: int = SEED, keep_rnl: bool = False,
               manifest_path: Optional[Path] = None, workers: int = 16) -> Dict[str, Any]:
    """Prepare (:func:`prepare_rows`) and write (:func:`write_rows`) the training rows.

    :param export_dir: The novel dataset's images-once export.
    :type export_dir: Path
    :param out_dir: Fresh folder (refused when it exists).
    :type out_dir: Path
    :param request: ``auto`` or a whole number of train rows.
    :type request: Union[str, int]
    :param seed: Build seed.
    :type seed: int
    :param keep_rnl: Keep Russian National Library rows.
    :type keep_rnl: bool
    :param manifest_path: The dataset's ``manifest.jsonl`` (None: the export's source dataset's).
    :type manifest_path: Optional[Path]
    :param workers: Parallel image copies.
    :type workers: int
    :return: ``mixture.json`` content.
    :rtype: Dict[str, Any]
    :raises FileExistsError: When ``out_dir`` exists.
    """
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists: the rows step writes a fresh folder")
    manifest_path = manifest_path if manifest_path is not None else dataset_manifest_path(export_dir)
    return write_rows(prepare_rows(export_dir, request, seed, keep_rnl, manifest_path), export_dir, out_dir, manifest_path,
                      str(request), seed, workers)


def write_rows(plan: RowsPlan, export_dir: Path, out_dir: Path, manifest_path: Path, request: str, seed: int,
               workers: int = 16) -> Dict[str, Any]:
    """Write prepared training rows as an images-once folder (layout in the module docstring).

    :param plan: :func:`prepare_rows` of ``export_dir``.
    :type plan: RowsPlan
    :param export_dir: The novel dataset's images-once export (the images are copied from it).
    :type export_dir: Path
    :param out_dir: Fresh folder (refused when it exists).
    :type out_dir: Path
    :param manifest_path: The dataset's ``manifest.jsonl`` the plan read (recorded with its digest).
    :type manifest_path: Path
    :param request: The rows request as given (recorded).
    :type request: str
    :param seed: Build seed (recorded).
    :type seed: int
    :param workers: Parallel image copies.
    :type workers: int
    :return: ``mixture.json`` content.
    :rtype: Dict[str, Any]
    :raises FileExistsError: When ``out_dir`` exists.
    """
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists: the rows step writes a fresh folder")
    card = card_text(plan.record, plan.stats)
    wanted = {sha: export_dir / "images" / f"{sha}.jpg"
              for table in (plan.train, plan.val) for sha in table[SHA_COLUMN].to_pylist()}
    out_dir.mkdir(parents=True)
    sizes = bvm.copy_images(wanted, out_dir / "images", workers)
    bvm.verify_images([plan.train, plan.val], out_dir / "images")
    bvm.write_rows(plan.train, plan.meta, out_dir / "rows" / "train.parquet")
    bvm.write_rows(plan.val, plan.meta, out_dir / "rows" / "val.parquet")
    splits = {"train": bvm.split_manifest(plan.train, sizes), "val": bvm.split_manifest(plan.val, sizes)}
    manifest = {"source": export_dir.name, "image_column": plan.meta["image_column"], "sha_column": SHA_COLUMN, "splits": splits,
                "n_rows": plan.train.num_rows + plan.val.num_rows, "n_images": len(sizes), "image_bytes": sum(sizes.values()),
                "row_image_bytes": sum(s["row_image_bytes"] for s in splits.values())}
    mixture = {"name": LOCAL_LEAF, "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "config": plan.record}
    shutil.copy2(export_dir / "stats.json", out_dir / "stats.json")
    if (export_dir / "PRIVATE.md").is_file():
        shutil.copy2(export_dir / "PRIVATE.md", out_dir / "PRIVATE.md")
    (out_dir / "README.md").write_text(card, encoding="utf-8")
    (out_dir / "documents.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in plan.documents), encoding="utf-8")
    bvm.write_json(out_dir / "build_record.json", {
        "export_dir": str(export_dir), "manifest_path": str(manifest_path), "manifest_sha256": file_sha256(manifest_path),
        "stats_sha256": plan.record["dataset_stats_sha256"], "request": request, "seed": seed, "keep_rnl": plan.record["keep_rnl"],
        "rnl_dropped": [{"stem": d["stem"], "split": d.get("split"), "script": d.get("script")} for d in plan.dropped]})
    bvm.write_json(out_dir / "mixture.json", mixture)
    bvm.write_json(out_dir / "manifest.json", manifest)          # written last: marks a finished folder
    return mixture


def plan_lines(plan: RowsPlan) -> List[str]:
    """What ``plan`` and ``rows`` print: pages, the range, the rows, the passes, the scripts and their drift, the steps.

    :param plan: :func:`prepare_rows`.
    :type plan: RowsPlan
    :return: Lines.
    :rtype: List[str]
    """
    r = plan.record
    drift = script_drift_problems(r["scripts"])
    within = sorted(r["rows_within_script_tolerance"], key=lambda n: (abs(n - r["train_rows"]), n))[:5]
    return [f"dataset {r['dataset']} (built {r['dataset_built']}): {r['train_pages']:,} train pages {r['pages_by_source']}",
            f"pass rule: {r['feasible_rows'][0]:,} to {r['feasible_rows'][1]:,} rows; whole steps {r['step_rows'][0]:,} to "
            f"{r['step_rows'][1]:,}; requested {r['requested']} -> {r['train_rows']:,} rows = {r['max_steps']:,} steps "
            f"({r['train_rows'] / r['train_pages']:.3f} passes; pages by rows {r['pages_by_rows']})",
            f"rows by source {r['rows_by_source']}; passes {r['passes_by_source']}",
            "scripts (train pages -> train rows): " + "; ".join(
                f"{s} {v['pages']} ({v['page_share']:.2%}) -> {v['rows']} ({v['row_share']:.2%}, {v['drift'] * 100:+.2f} points)"
                for s, v in r["scripts"].items()),
            ("script drift OK (within the gates' tolerance)" if not drift else
             "WARNING, the gates will refuse this plan: " + "; ".join(drift) + ". Row counts within the tolerance, nearest "
             f"first: {within or 'none'} (--train-rows N)"),
            f"val {r['val_rows']:,} rows {r['val_by_source']}; questions rewritten to the neutral prompt {r['questions_rewritten']}; "
            f"RNL rows left out {r['rnl_dropped']} (keep_rnl {r['keep_rnl']})",
            f"schedule: warm-up {warmup_steps(r['train_rows'])} steps, Hub-confirmed milestones {r['milestones']}, Studio evals "
            f"{r['eval_steps']}, Lambda safety stop {lambda_budget_h(r['train_rows']):.0f} h"]


# ----------------------------------------------------------------------------- notebook text


def _fill(template: str, **values: Any) -> str:
    """Replace every ``<<NAME>>`` of a template.

    :param template: Text with ``<<NAME>>`` placeholders.
    :type template: str
    :param values: NAME -> value.
    :type values: Any
    :return: The text.
    :rtype: str
    :raises ValueError: When a placeholder is left or a value is unused.
    """
    for name, value in values.items():
        if f"<<{name}>>" not in template:
            raise ValueError(f"placeholder <<{name}>> not in the template")
        template = template.replace(f"<<{name}>>", str(value))
    if "<<" in template and re.search(r"<<[A-Z0-9_]+>>", template):
        raise ValueError(f"placeholders left: {re.findall(r'<<[A-Z0-9_]+>>', template)}")
    return template


DATA_TITLE = f"# Cell 2 — data: {LOCAL_LEAF} (images-once, MAP-STYLE, materialised mixture — no streaming)"
PIN_GUARD = (f'# {ROUND}: never train on an unpinned dataset ({PUSH_SCRIPT.rsplit("/", 1)[1]} pins the pushed revision and regenerates this notebook)\n'
             'if V22_REVISION == "PIN-AFTER-PUSH" or len(V22_REVISION) != 40 or any(_c not in "0123456789abcdef" for _c in V22_REVISION):\n'
             f'    raise RuntimeError(f"dataset revision {{V22_REVISION!r}} is not a pinned 40-hex sha: run {PUSH_SCRIPT}, "\n'
             '                       "which pins the pushed revision and regenerates this notebook")\n')
REVISION_COMMENT = "# {round} mixture (novel pages only, no replay): rows + images_part*.tar ({rows:,} train rows over {pages:,} novel pages)"
REVISION_LINE = re.compile(r'^V22_REVISION = "([^"]+)"   # ' + re.escape(ROUND)
                           + r" mixture .*\(([\d,]+) train rows over ([\d,]+) novel pages\)$", re.M)
OLD_SIZE_GATE = f'assert len(train_ds) >= {{min_rows}} and len(eval_ds) >= {dv.MIN_VAL_ROWS}, "Arabic mixture incomplete"\n'
OLD_FLOORS = ('for _s, _lo in ():\n'
              '    assert _src[_s] / _n >= _lo, f"{_s} share {_src[_s] / _n:.3f} below {_lo} — wrong dataset build"\n')
OLD_VAL_COVERAGE = 'assert set(_src) <= set(eval_ds.column("source")), "val must cover every train source"\n'
PLAN_GATE = '''# <<ROUND>> plan gate: the pinned rows are the plan this notebook was derived with (derive_novel_v1.py) and the plan its
# rows step recorded in mixture.json; val is the novel dataset's own val split
NOVEL_TRAIN_ROWS = <<ROWS>>
NOVEL_TRAIN_PAGES = <<PAGES>>
_mix = MIX["config"]
assert len(train_ds) == NOVEL_TRAIN_ROWS == _mix["train_rows"] and NOVEL_TRAIN_PAGES == _mix["train_pages"], (
    f"{len(train_ds)} train rows over {_mix['train_pages']} pages in mixture.json ({_mix['train_rows']} planned there), the "
    f"notebook plans {NOVEL_TRAIN_ROWS} over {NOVEL_TRAIN_PAGES}: wrong dataset build")
assert len(eval_ds) == _mix["val_rows"] > 0, f"{len(eval_ds)} val rows, mixture.json plans {_mix['val_rows']}"
'''
PASS_GATE = '''# <<ROUND>> pass rule (derive_novel_v1.pass_plan): every train page 2 or 3 times, no source above 2.5 passes, rows and
# pages per source as planned. No share floors: the novel pages keep their natural shares.
_stems = train_ds.column("stem")
_per_page = Counter(_stems)
assert set(_per_page.values()) <= {2, 3}, f"train pages outside 2-3 rows: {Counter(_per_page.values())}"
_pages = Counter(dict(zip(_stems, train_ds.column("source"))).values())
assert len(_per_page) == NOVEL_TRAIN_PAGES and dict(_pages) == _mix["pages_by_source"] and dict(_src) == _mix["rows_by_source"], (
    f"pages {dict(_pages)} / rows {dict(_src)} per source are not the plan's {_mix['pages_by_source']} / {_mix['rows_by_source']}")
assert all(_src[_s] <= 2.5 * _p for _s, _p in _pages.items()), "a source above 2.5 passes"
print("passes by source:", {_s: round(_src[_s] / _p, 3) for _s, _p in sorted(_pages.items())})
'''
VAL_BLOCK = '''assert not set(_per_page) & set(eval_ds.column("stem")), "a val page is also a train page"
print("val by source:", dict(Counter(eval_ds.column("source"))))
'''
NOVEL_HYGIENE = '''# <<ROUND>> hygiene: every question is the script-neutral prompt (prompts.SCRIPT_NEUTRAL_PROMPT, verbatim below) and names
# no script; every row is a novel page row (section <<SECTION>>, task <<TASK>>); answers are non-empty and carry
# no harakat, tatweel or backtick (the mojibake gate above covers < and >). The script of every answer is measured from
# its letters (the novel build's measure): the novel Hebrew / Judaeo-Arabic rows, the Arabic-script rows (at least 40
# Arabic letters and 9 x the Hebrew ones) and the mixed rows are printed with their shares. There are no floors.
import re as _re
NEUTRAL_PROMPT = <<PROMPT>>
_HE, _AR = _re.compile(r"[\\u05d0-\\u05ea]"), _re.compile(r"[\\u0621-\\u064a\\u0671-\\u06d3]")
_MARKS = _re.compile(r"[\\u0640\\u064b-\\u0652\\u0670]")


def _script(text):
    """hebrew / arabic / mixed / none: the measure of novel_build.targets.measure_script."""
    _h, _r = len(_HE.findall(text)), len(_AR.findall(text))
    if not _h + _r:
        return "none"
    if min(_h, _r) >= 20 and min(_h, _r) >= 0.10 * (_h + _r):
        return "mixed"
    return "hebrew" if _h >= _r else "arabic"


for _name, _ds in (("train", train_ds), ("val", eval_ds)):
    _qs = set(_ds.column("question"))
    _other = sorted(_qs - {NEUTRAL_PROMPT})[:1]
    assert not _other, f"{_name}: a question is not the script-neutral prompt: {_other}"
    assert not any(_c in _x for _x in _qs for _c in ("Arabic script", "Hebrew script")), f"{_name}: a question names a script"
    _kinds = (set(_ds.column("section")), set(_ds.column("task")))
    assert _kinds == ({"<<SECTION>>"}, {"<<TASK>>"}), f"{_name}: rows other than novel page transcriptions: {_kinds}"
    _texts = _ds.column("answer")
    assert all(_x.strip() for _x in _texts), f"{_name}: an empty answer"
    assert not any(_MARKS.search(_x) for _x in _texts), f"{_name}: harakat, shadda or tatweel in an answer (the build strips them)"
    assert not any("`" in _x for _x in _texts), f"{_name}: a backtick in an answer"
    _measured = [_script(_x) for _x in _texts]
    assert "none" not in _measured, f"{_name}: an answer without Hebrew or Arabic letters"
    _weak = [_k for _x, _k in zip(_texts, _measured) if _k == "arabic"
             and not (len(_AR.findall(_x)) >= 40 and len(_AR.findall(_x)) >= 9 * len(_HE.findall(_x)))]
    assert not _weak, f"{_name}: {len(_weak)} Arabic-script answers are not mainly Arabic script"
_scripts = Counter(_script(_x) for _x in _ans)
_shares = {_k: round(_v / len(_ans), 3) for _k, _v in sorted(_scripts.items())}
print(f"hygiene OK: transcription {len(_transc)}; novel Hebrew/Judaeo-Arabic rows {_scripts['hebrew']}, Arabic-script rows "
      f"{_scripts['arabic']}, mixed rows {_scripts['mixed']} (shares {_shares}; natural shares, no floors)")
'''
OLD_PICKS = ('# guardrail: one real batch (a KTIV page + an Arabic-script page) must show page-res pixels + masked labels\n'
             '_i_page = next(i for i, (t, s) in enumerate(zip(train_ds.column("task"), train_ds.column("source")))\n'
             '               if t == "fragment_transcribe" and s.startswith("ktiv"))\n'
             '_i_qa = next(i for i, s in enumerate(train_ds.column("source")) if s.startswith("arabic_"))   # the Arabic page\n')
NEW_PICKS = ('# guardrail: one real batch (two novel pages: a Hebrew-script page and an Arabic-script page, or a second page when\n'
             '# the mixture has no Arabic-script page) must show page-res pixels + masked labels\n'
             '_scr4 = [_script(a) for a in train_ds.column("answer")]\n'
             '_stems4 = train_ds.column("stem")\n'
             '_i_page = next(i for i, s in enumerate(_scr4) if s == "hebrew")\n'
             '_i_qa = next((i for i, s in enumerate(_scr4) if s == "arabic"), next(i for i, st in enumerate(_stems4) if st != _stems4[_i_page]))\n')
PICK_PREFIXES: Tuple[str, ...] = ("_scr4 = ", "_stems4 = ", "_i_page = ", "_i_qa = ")
OLD_PICK_MESSAGE = "labels masked on both rows (Hebrew page + Arabic page)"
NEW_PICK_MESSAGE = "labels masked on both rows (two novel pages)"
MODEL_TITLE = f"# Cell 3 — model at page resolution + THE MERGER KNOB + warm start (v2.5a step {WARM_STEP})"
NEW_WARM = (f'# Warm start: v2.5a step {WARM_STEP} — the Arabic flagship, the end of the third Arabic-script run (LoRA r16,\n'
            '# merger frozen). Weights only: this run has its own optimizer and a fresh cosine schedule.\n'
            f'WARM_CKPT_REPO = "{WARM_CKPT_REPO}"   # private: the token must read it\n'
            f'WARM_REVISION = "{{warm_revision}}"  # commit "Training in progress, step {WARM_STEP}, checkpoint"\n'
            f'WARM_STEP = {WARM_STEP}\n')
NEW_WARM_DOWNLOAD = _replace_once(d4.NEW_WARM_DOWNLOAD, "# v24a: the pinned commit", f"# {ROUND}: the pinned commit", "warm step comment")
TRAIN_TITLE = f"# Cell 6 — {ROUND} (novel pages only, no replay, warm start v2.5a step {WARM_STEP}) training:"
OLD_PRIVATE = 'hub_strategy="checkpoint", hub_private_repo=True,    # Muharaf is CC BY-NC-SA: private until the licence question is decided'
NEW_PRIVATE = 'hub_strategy="checkpoint", hub_private_repo=True,    # private by default: nothing is published before the owner has vetted it'
SAVE_LINE = f"save_steps={SAVE_STEPS}, save_total_limit=2,"
EVAL_LINE = f'eval_strategy="steps", eval_steps={EVAL_EVERY}, per_device_eval_batch_size=1,'
DISPATCH_LINE = 'accelerator_config={"dispatch_batches": False},'
SESSION_LINE = ("print(describe_session(len(train_ds), MAX_STEPS, _a.per_device_train_batch_size, _a.gradient_accumulation_steps, "
                "_resume_step, ORDER_SEED))\n")
BUDGET_CALLBACK = "trainer.add_callback(TimeBudgetTrainerCallback(budget_s=TIME_BUDGET_H * 3600))   # clean save + push before Colab's 24 h cap\n"
STEP0_START = f"# ---- {ROUND}: step-0 eval of the warm start"
STEP0_BLOCK = '''<<START>> (W&B logs it at step 0) ----
# The warm-start stop rule reads every later eval against it: eval loss above this value for two consecutive evals means
# the run is broken (WarmStartWatchCallback below prints and alerts; the call is made by eye on W&B). A fresh start
# evaluates the warm start and stores the value in the checkpoint repo; a resumed session reads it back.
STEP0_FILE = "<<FILE>>"
if resume_dir is None:
    _m0 = trainer.evaluate()
    STEP0_EVAL_LOSS = float(_m0["eval_loss"])
    from huggingface_hub import HfApi as _HfApi
    _HfApi().upload_file(path_or_fileobj=json.dumps({"eval_loss": STEP0_EVAL_LOSS, "warm_start": f"{WARM_CKPT_REPO}@{WARM_REVISION}",
                                                     "val_rows": len(eval_ds)}).encode("utf-8"),
                         path_in_repo=STEP0_FILE, repo_id=CKPT_REPO, commit_message="Step-0 eval loss of the warm start (warm-start stop rule)")
    FastVisionModel.for_training(model)
else:
    from huggingface_hub import hf_hub_download as _hf_hub_download
    with open(_hf_hub_download(CKPT_REPO, STEP0_FILE), encoding="utf-8") as _fh:
        STEP0_EVAL_LOSS = float(json.load(_fh)["eval_loss"])
print(f"step-0 eval loss of the warm start: {STEP0_EVAL_LOSS:.4f} on {len(eval_ds)} val rows")
'''
CALLBACKS_START = f"# ---- {ROUND}: checkpoints pushed AND confirmed on the Hub at the quarter points of the run"
CALLBACKS_BLOCK = '''<<START>> ----
# Quarter points rounded up to a multiple of <<GRAIN>> (at least <<MIN>>; the last is the final step, which the trainer saves at the
# end). The Studio eval chain scores <<EVALS>> by their commit "Training in progress, step N, checkpoint"; the <<SAVE>>-step
# saves stay for resume. A milestone the trainer's own (asynchronous) push did not land is pushed again from the local
# checkpoint by ensure_hub_checkpoint (helpers cell), which fails loudly if it cannot confirm the step.
MILESTONES = <<MILESTONES>>
assert MILESTONES[-1] == MAX_STEPS and all(_m % trainer.args.save_steps == 0 or _m == MAX_STEPS for _m in MILESTONES), \\
    f"milestones {MILESTONES} must be save steps (save_steps {trainer.args.save_steps}) and end at MAX_STEPS {MAX_STEPS}"


class MilestonePushCallback(TrainerCallback):
    """At every milestone save (and at a train() start on a milestone), wait for the trainer's Hub pushes and confirm
    that the Hub's last-checkpoint is that step."""

    def __init__(self, trainer, ckpt_repo, out_dir, steps):
        super().__init__()
        self.trainer, self.ckpt_repo, self.out_dir, self.steps, self.confirmed = trainer, ckpt_repo, out_dir, {int(s) for s in steps}, []

    def confirm(self, step):
        if step not in self.steps or step in self.confirmed:
            return
        job = getattr(self.trainer, "push_in_progress", None)
        if job is not None:
            job.wait_until_done()          # the pushes run one after another: this step's is the last one queued
        ensure_hub_checkpoint(self.ckpt_repo, self.out_dir, step)
        self.confirmed.append(step)
        print(f"milestone step {step}: checkpoint confirmed on the Hub ({self.ckpt_repo})", flush=True)

    def on_train_begin(self, args, state, control, **kwargs):
        self.confirm(int(state.global_step))   # an in-session retry or a resume from a milestone confirms it first
        return control

    def on_save(self, args, state, control, **kwargs):
        self.confirm(int(state.global_step))
        return control


# ---- <<ROUND>>: train-loss alarm (gate the real train path) ----
# The first logged train loss above LOSS_ALARM_FACTOR x max(LOSS_ALARM_REFERENCE, STEP0_EVAL_LOSS) STOPS the run: a broken
# train path shows at its first log (the image-blind v2.1 run logged 3.0 against 0.55). The reference 0.60 is v2.5a's first
# logged train loss, on a mixture of mostly seen rows; novel pages start near the warm start's loss on unseen pages, which
# the step-0 eval measured, so that value raises the threshold when it is higher (2 x 0.60 = 1.20 stays the floor). A
# non-finite loss stops the run too. LossAlarmError is an AssertionError, which train_with_exact_resume never retries.
LOSS_ALARM_REFERENCE, LOSS_ALARM_FACTOR = <<REFERENCE>>, <<FACTOR>>
LOSS_ALARM_THRESHOLD = LOSS_ALARM_FACTOR * max(LOSS_ALARM_REFERENCE, STEP0_EVAL_LOSS)
LOSS_ALARM_RULE = f"{LOSS_ALARM_FACTOR} x max({LOSS_ALARM_REFERENCE:.2f}, step-0 eval {STEP0_EVAL_LOSS:.3f})"
print(f"loss alarm threshold {LOSS_ALARM_THRESHOLD:.2f} = {LOSS_ALARM_RULE}", flush=True)


class LossAlarmError(AssertionError):
    """A logged train loss crossed the alarm level: the run stops and is not retried."""


def loss_alarm(value, threshold):
    """Whether a logged loss is non-finite or above the threshold."""
    return not math.isfinite(value) or value > threshold


class LossAlarmCallback(TrainerCallback):
    """Stop the run at the first logged train loss above the alarm threshold (a print and a W&B alert first)."""

    def __init__(self, threshold, rule):
        super().__init__()
        self.threshold, self.rule = float(threshold), rule

    def on_log(self, args, state, control, logs=None, **kwargs):
        value = (logs or {}).get("loss")
        if value is not None and loss_alarm(float(value), self.threshold):
            text = (f"step {state.global_step}: train loss {float(value):.3f} > threshold {self.threshold:.2f} ({self.rule}): "
                    "the train path is broken, the run stops")
            print(f"LOSS ALARM: {text}", flush=True)
            if wandb.run is not None:
                wandb.alert(title="<<RUN>>: loss alarm, run stopped", text=text)
            raise LossAlarmError(text)
        return control


class WarmStartWatchCallback(TrainerCallback):
    """The warm-start stop rule, for the eye: print every eval loss next to the warm start's step-0 value and alert (the
    run goes on) once it has been above that value for two consecutive evals."""

    def __init__(self, step0_loss):
        super().__init__()
        self.step0_loss, self.above, self.alerted = float(step0_loss), 0, False

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        value = (metrics or {}).get("eval_loss")
        if value is None or int(state.global_step) == 0:
            return control
        self.above = self.above + 1 if float(value) > self.step0_loss else 0
        print(f"eval loss {float(value):.4f} at step {state.global_step} (warm start {self.step0_loss:.4f}; "
              f"{self.above} consecutive eval(s) above it)", flush=True)
        if self.above >= 2 and not self.alerted:
            self.alerted = True
            text = (f"eval loss above the warm start's step-0 value {self.step0_loss:.4f} for two consecutive evals "
                    f"(step {state.global_step}): the warm-start stop rule says the run is broken; check it")
            print(f"WARM-START STOP RULE: {text}", flush=True)
            if wandb.run is not None:
                wandb.alert(title="<<RUN>>: warm-start stop rule", text=text)
        return control


_milestones = MilestonePushCallback(trainer, CKPT_REPO, OUT_DIR, MILESTONES)
trainer.add_callback(_milestones)
trainer.add_callback(LossAlarmCallback(LOSS_ALARM_THRESHOLD, LOSS_ALARM_RULE))
trainer.add_callback(WarmStartWatchCallback(STEP0_EVAL_LOSS))
'''
OLD_MERGED = f'MERGED_REPO = "isaacmg/qwen3-vl-8b-hebrew-{TAG}-merged"'


def intro_md(train_rows: int, pages: int) -> str:
    """Markdown intro of the notebook.

    :param train_rows: Train rows.
    :type train_rows: int
    :param pages: Train pages.
    :type pages: int
    :return: Text of cell 0 (the Lambda edition puts its launch notes in front of it).
    :rtype: str
    """
    steps = max_steps_for(train_rows)
    milestones = ", ".join(f"{s:,}" for s in milestone_steps(steps))
    evals = ", ".join(f"{s:,}" for s in eval_steps(steps))
    return f"""# Genizah {ROUND} — novel pages only, no replay, on top of v2.5a

Same training stack as v2.5a (pinned install triplet, 6.5/7 MP resolution contract, LoRA r16 on tower + language,
merger FROZEN, adamw_8bit, cosine, seed {SEED}, fixed row order with exact resume, {SAVE_STEPS}-step saves, prepared-dataloader
gate before step 1). What changes:

| | v2.5a | this run |
|---|---|---|
| Warm start | v2.4a step 2000 | **v2.5a step {WARM_STEP}** (`{WARM_CKPT_REPO}@{WARM_REVISION[:8]}`), weights only: fresh optimizer, fresh cosine |
| Data | pooled Geniza Arabic pages + outside handwriting sets + Hebrew replay | `{DATA_REPO}`: only page rows of documents no earlier fine-tune trained on; replay quota 0, no outside sets, no script upweighted |
| Rows / steps | 9,600 / 1,200 | **{train_rows:,} / {steps:,}**: {pages:,} train pages, every page 2 or 3 times, the extra rows drawn uniformly over all pages (natural shares), no source above {MAX_PASSES} passes (batch 1 x accumulation 8) |
| Prompt | names the script | the script-neutral prompt on every row (`prompts.SCRIPT_NEUTRAL_PROMPT`) |
| Learning rate | 5e-5, 3% warm-up | {LR}, {float(WARMUP_RATIO):.0%} warm-up ({warmup_steps(train_rows)} steps), fresh cosine |
| Checkpoints | every 50 steps; 300, 600, 900, 1,200 confirmed on the Hub | every {SAVE_STEPS} steps (resume); steps {milestones} pushed **and confirmed** on the Hub (the quarter points, rounded up to a multiple of {MILESTONE_GRAIN}) |
| Gates | share floors, Arabic-script prompts | the plan, the pass rule, the neutral prompt, novel page rows, scripts printed without floors; a **train-loss alarm** (the first logged train loss above {LOSS_ALARM_FACTOR:g} x max({LOSS_ALARM_REFERENCE:.2f}, the step-0 eval loss), at least {loss_alarm_threshold(0):.2f}, stops the run) and a **step-0 eval** of the warm start (warm-start stop rule) |
| Session | Lambda: safety stop 28 h | Lambda A100-40GB: one pass ≈ {steps * MEASURED_S_PER_STEP / 3600:.1f} h at {MEASURED_S_PER_STEP} s/step; safety stop {lambda_budget_h(train_rows):.0f} h |

The data cell **refuses to run** until the dataset revision is pinned (`{PUSH_SCRIPT}` pins it and regenerates this
notebook); the model cell checks that the warm start is v2.5a's step-{WARM_STEP} checkpoint.

**To resume after a session ends: run every cell again, unedited.** The notebook finds the newest checkpoint on the
Hub, checks the stored schedule and row order, and continues with the rows not yet trained.

Run name `{RUN_NAME}`; checkpoints in `{CKPT_REPO}` (new, private); merged model `{MERGED_REPO}` (private). Steps
{evals} are scored in one sequential queue on the Studio: PGP-131 first, then religious-140, then the Arabic-script set
(covered images) as a regression monitor.
"""


def derive(nb: Dict, revision: str, train_rows: int, pages: int) -> Dict:
    """Apply the novel_v1 edits to a loaded v2.2b notebook (the Colab edition, kept in memory).

    :param nb: Notebook JSON of ``genizah_v22b.ipynb`` (not modified in place).
    :type nb: Dict
    :param revision: 40-hex revision of :data:`DATA_REPO`, or :data:`PLACEHOLDER` (the notebook then refuses to run).
    :type revision: str
    :param train_rows: Train rows of the pinned dataset (a multiple of 8).
    :type train_rows: int
    :param pages: Train pages they cover.
    :type pages: int
    :return: The Colab edition.
    :rtype: Dict
    :raises ValueError: When an anchor is missing or ambiguous, the rows do not fit the pages, or :data:`LR` is not a
        number.
    """
    if not MIN_PASSES * pages <= train_rows <= MAX_PASSES * pages:
        raise ValueError(f"{train_rows} train rows over {pages} pages: not {MIN_PASSES} to {MAX_PASSES} passes")
    arm = make_arm(train_rows)
    steps = arm.max_steps
    out = dv.derive(nb, revision, arm)
    cells = out["cells"]
    cells[0]["source"] = intro_md(train_rows, pages).splitlines(keepends=True)
    # --- data cell: repository, pinned revision only, local folder, this mixture's gates ----------------------------
    c2 = "".join(cells[2]["source"])
    c2 = _replace_once(c2, d4.OLD_DATA_TITLE, DATA_TITLE, "data cell title")
    c2 = _replace_once(c2, f'V22_REPO = "{dv.DATA_REPO}"', f'V22_REPO = "{DATA_REPO}"', "repo")
    c2 = _replace_line(c2, r'^V22_REVISION = "',
                       f'V22_REVISION = "{revision}"   ' + REVISION_COMMENT.format(round=ROUND, rows=train_rows, pages=pages), "revision")
    c2 = _replace_once(c2, d4.OLD_UNPINNED, PIN_GUARD, "placeholder guard")
    c2 = _replace_once(c2, f'LOCAL = "/content/{d4.V23A_LOCAL_LEAF}"', f'LOCAL = "/content/{LOCAL_LEAF}"', "local dir")
    c2 = _replace_once(c2, OLD_SIZE_GATE.format(min_rows=arm.min_train_rows), _fill(PLAN_GATE, ROUND=ROUND, ROWS=train_rows, PAGES=pages),
                       "size gate")
    c2 = _replace_once(c2, OLD_FLOORS, _fill(PASS_GATE, ROUND=ROUND), "share floors")
    c2 = _replace_once(c2, OLD_VAL_COVERAGE, VAL_BLOCK, "val coverage")
    c2 = _replace_once(c2, dv.NEW_HYGIENE, _fill(NOVEL_HYGIENE, ROUND=ROUND, SECTION=SECTION, TASK=TASK,
                                                 PROMPT=repr(SCRIPT_NEUTRAL_PROMPT)), "hygiene block")
    cells[2]["source"] = c2.splitlines(keepends=True)
    # --- model cell: warm start v2.5a step 1200, checked ---------------------------------------------------------
    c3 = "".join(cells[3]["source"])
    c3 = _replace_once(c3, d4.OLD_MODEL_TITLE, MODEL_TITLE, "model cell title")
    c3 = _replace_once(c3, d4.OLD_WARM, NEW_WARM.format(warm_revision=WARM_REVISION), "warm start pin")
    c3 = _replace_once(c3, d4.OLD_WARM_DOWNLOAD, NEW_WARM_DOWNLOAD, "warm start step check")
    cells[3]["source"] = c3.splitlines(keepends=True)
    # --- collator cell: the guardrail batch is two novel pages ----------------------------------------------------
    c4 = "".join(cells[4]["source"])
    c4 = _replace_once(c4, OLD_PICKS, NEW_PICKS, "guardrail picks")
    c4 = _replace_once(c4, OLD_PICK_MESSAGE, NEW_PICK_MESSAGE, "guardrail message")
    cells[4]["source"] = c4.splitlines(keepends=True)
    # --- training cell: title, schedule, step-0 eval, milestones, loss alarm ----------------------------------------
    c8 = "".join(cells[8]["source"])
    c8 = _replace_once(c8, d4.OLD_TRAIN_TITLE, TRAIN_TITLE, "cell title")
    float(LR)                                                    # a number literal, or fail here rather than in the notebook
    c8 = _replace_line(c8, d4.LR_LINE, f"        learning_rate={LR},                "
                       f"# v2.5a's rate on a fresh cosine: the novel pages on top of v2.5a step {WARM_STEP}", "lr")
    c8 = _replace_once(c8, d4.OLD_WARMUP, f'warmup_ratio={WARMUP_RATIO}, lr_scheduler_type="cosine", weight_decay=0.01,   '
                       f'# {warmup_steps(train_rows)} warm-up steps of {steps:,}; the resume gate knows cosine only', "warm-up")
    c8 = _replace_once(c8, OLD_PRIVATE, NEW_PRIVATE, "private checkpoints")
    for line, what in ((d4.SEED_LINE, "seed"), (SAVE_LINE, "50-step saves"), (EVAL_LINE, "100-step evals"),
                       (DISPATCH_LINE, "no dispatched batches"), (SESSION_LINE, "session description"), (BUDGET_CALLBACK, "time budget")):
        if c8.count(line) != 1:
            raise ValueError(f"{what}: {line.strip()!r} not found once in the training cell")
    c8 = _replace_once(c8, SESSION_LINE, SESSION_LINE + _fill(STEP0_BLOCK, START=STEP0_START, FILE=STEP0_FILE), "step-0 eval")
    c8 = _replace_once(c8, BUDGET_CALLBACK, BUDGET_CALLBACK + _fill(
        CALLBACKS_BLOCK, START=CALLBACKS_START, GRAIN=MILESTONE_GRAIN, MIN=MILESTONE_MIN, SAVE=SAVE_STEPS,
        EVALS=", ".join(f"{s:,}" for s in eval_steps(steps)), MILESTONES=repr(milestone_steps(steps)), ROUND=ROUND,
        REFERENCE=repr(LOSS_ALARM_REFERENCE), FACTOR=repr(LOSS_ALARM_FACTOR), RUN=RUN_NAME), "milestones and alarms")
    cells[8]["source"] = c8.splitlines(keepends=True)
    # --- merged-model cell: the run's own private repository ------------------------------------------------------
    c9 = "".join(cells[9]["source"])
    c9 = _replace_once(c9, OLD_MERGED, f'MERGED_REPO = "{MERGED_REPO}"', "merged repo")
    cells[9]["source"] = c9.splitlines(keepends=True)
    return out


def to_lambda(nb: Dict, train_rows: int, notebook_name: str = NOTEBOOK) -> Dict:
    """The Lambda edition with a session safety stop of :data:`BUDGET_FACTOR` x one pass.

    :param nb: :func:`derive` output.
    :type nb: Dict
    :param train_rows: Train rows (sets the budget).
    :type train_rows: int
    :param notebook_name: File name the edition is written as (quoted in the launch notes).
    :type notebook_name: str
    :return: The Lambda edition.
    :rtype: Dict
    """
    steps = max_steps_for(train_rows)
    out = host_lambda.to_lambda(nb, notebook_name)
    c7 = "".join(out["cells"][7]["source"])
    c7 = _replace_line(c7, r"^TIME_BUDGET_H = .*$",
                       f"TIME_BUDGET_H = {lambda_budget_h(train_rows)}   # Lambda, one session: {steps:,} steps x ~{MEASURED_S_PER_STEP} s "
                       f"(v2.3a on the A100-SXM4-40GB) = {steps * MEASURED_S_PER_STEP / 3600:.1f} h, x {BUDGET_FACTOR}; a safety stop only, "
                       "max_steps ends the run", "time budget")
    out["cells"][7]["source"] = c7.splitlines(keepends=True)
    return out


def lambda_edition(revision: str, train_rows: int, pages: int, src: Path = SRC_NOTEBOOK) -> Dict:
    """The Lambda edition derived from the v2.2b notebook.

    :param revision: Dataset revision (40-hex or :data:`PLACEHOLDER`).
    :type revision: str
    :param train_rows: Train rows.
    :type train_rows: int
    :param pages: Train pages.
    :type pages: int
    :param src: The v2.2b notebook.
    :type src: Path
    :return: The notebook.
    :rtype: Dict
    """
    return to_lambda(derive(read_json(src), revision, train_rows, pages), train_rows)


def notebook_plan(nb: Dict) -> Dict[str, Any]:
    """The plan a derived notebook carries (its revision line names the rows and pages).

    :param nb: A notebook written by :func:`lambda_edition`.
    :type nb: Dict
    :return: ``revision``, ``train_rows``, ``pages``, ``max_steps``, ``warmup_steps``, ``milestones``, ``eval_steps``,
        ``budget_h``, ``pinned``.
    :rtype: Dict[str, Any]
    :raises ValueError: When the revision line is not found.
    """
    m = REVISION_LINE.search("".join(nb["cells"][2]["source"]))
    if m is None:
        raise ValueError(f"no {ROUND} revision line in the data cell")
    rows, pages = int(m.group(2).replace(",", "")), int(m.group(3).replace(",", ""))
    steps = max_steps_for(rows)
    return {"revision": m.group(1), "train_rows": rows, "pages": pages, "max_steps": steps, "warmup_steps": warmup_steps(rows),
            "milestones": list(milestone_steps(steps)), "eval_steps": list(eval_steps(steps)), "budget_h": lambda_budget_h(rows),
            "pinned": bool(re.fullmatch(r"[0-9a-f]{40}", m.group(1)))}


def notebook_problems(nb: Dict, src: Path = SRC_NOTEBOOK) -> List[str]:
    """Why a Lambda edition may not be launched: an unpinned dataset or warm start, or a file that is not a fresh
    derivation at its own revision and plan.

    :param nb: The notebook as written.
    :type nb: Dict
    :param src: The v2.2b notebook.
    :type src: Path
    :return: Problems (empty: launchable).
    :rtype: List[str]
    """
    plan = notebook_plan(nb)
    problems = []
    if not plan["pinned"]:
        problems.append(f"dataset revision {plan['revision']!r} is not pinned: run {PUSH_SCRIPT}")
    if not warm_start_pinned():
        problems.append(f"warm start {WARM_CKPT_REPO}@{WARM_REVISION} is not pinned to a commit")
    if nb != lambda_edition(plan["revision"], plan["train_rows"], plan["pages"], src):
        problems.append("the notebook is not a fresh derivation at its revision and plan: re-derive it")
    return problems


def write_notebook(revision: str, train_rows: int, pages: int, out_dir: Path = HERE, src: Path = SRC_NOTEBOOK) -> Path:
    """Write the Lambda edition (and ``lambda_setup.sh`` beside it when it changed).

    :param revision: Dataset revision (40-hex or :data:`PLACEHOLDER`).
    :type revision: str
    :param train_rows: Train rows.
    :type train_rows: int
    :param pages: Train pages.
    :type pages: int
    :param out_dir: Folder.
    :type out_dir: Path
    :param src: The v2.2b notebook.
    :type src: Path
    :return: The notebook path.
    :rtype: Path
    """
    nb = lambda_edition(revision, train_rows, pages, src)
    if d4.write_if_changed(out_dir / SETUP_SCRIPT, host_lambda.setup_script()):
        print(f"wrote {out_dir / SETUP_SCRIPT}")
    path = out_dir / NOTEBOOK
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    steps = max_steps_for(train_rows)
    print(f"wrote {path} ({len(nb['cells'])} cells; dataset {DATA_REPO}@{revision}; run {RUN_NAME}, {train_rows:,} rows over "
          f"{pages:,} pages = {steps:,} steps; warm start {WARM_CKPT_REPO}@{WARM_REVISION[:8]} (step {WARM_STEP}); lr {LR}; "
          f"Hub-confirmed milestones {', '.join(str(s) for s in milestone_steps(steps))}; budget {lambda_budget_h(train_rows):.0f} h")
    if revision == PLACEHOLDER:
        print(f"NOTE: the notebook refuses to run until regenerated with the pushed revision ({PUSH_SCRIPT} does it)")
    return path


# ----------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the CLI (subcommands ``plan``, ``rows``, ``notebook``, ``show``).

    :return: The parser.
    :rtype: argparse.ArgumentParser
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name, text in (("plan", "preview the training rows of a novel dataset export (nothing written)"),
                       ("rows", "write the training rows of a novel dataset export")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--export", type=Path, required=True, help="the novel dataset's images-once export")
        p.add_argument("--train-rows", default=AUTO, help=f"'{AUTO}' (default: round(2.2 x pages) in whole steps) or a number")
        p.add_argument("--seed", type=int, default=SEED)
        p.add_argument("--keep-rnl", action="store_true", help="keep Russian National Library rows (default: left out)")
        p.add_argument("--manifest", type=Path, default=None, help="the dataset's manifest.jsonl (default: the export's source)")
        if name == "rows":
            p.add_argument("--out", type=Path, required=True, help="fresh folder for the training rows")
            p.add_argument("--workers", type=int, default=16)
    p = sub.add_parser("notebook", help=f"write {NOTEBOOK} (+ {SETUP_SCRIPT})")
    p.add_argument("--revision", required=True, help=f"40-hex revision of {DATA_REPO}, or {PLACEHOLDER}")
    p.add_argument("--plan", type=Path, default=None, help="mixture.json of the built rows (train rows and pages)")
    p.add_argument("--pages", type=int, default=None, help=f"train pages (placeholder derivations; reference {REFERENCE_PAGES})")
    p.add_argument("--train-rows", default=AUTO, help=f"with --pages: '{AUTO}' or a number")
    p.add_argument("--src", type=Path, default=SRC_NOTEBOOK)
    p.add_argument("--out-dir", type=Path, default=HERE)
    p = sub.add_parser("show", help="print the plan a derived notebook carries")
    p.add_argument("--notebook", type=Path, default=HERE / NOTEBOOK)
    p.add_argument("--field", default=None, help="print one field (lists space-separated), e.g. eval_steps")
    p.add_argument("--require-pinned", action="store_true", help="exit non-zero unless the dataset revision is pinned")
    return ap


def notebook_numbers(args: argparse.Namespace) -> Tuple[int, int]:
    """Train rows and pages of a ``notebook`` call: from the built plan, or from ``--pages`` (and ``--train-rows``).

    :param args: Parsed ``notebook`` arguments.
    :type args: argparse.Namespace
    :return: (train rows, pages).
    :rtype: Tuple[int, int]
    :raises SystemExit: On neither or both of ``--plan`` and ``--pages``.
    """
    if (args.plan is None) == (args.pages is None):
        raise SystemExit("give --plan (the built rows' mixture.json) or --pages (a placeholder derivation), not both")
    if args.plan is not None:
        config = read_json(args.plan)["config"]
        return int(config["train_rows"]), int(config["train_pages"])
    return resolve_rows(args.train_rows, {"pages": args.pages}), args.pages


def main(argv: Optional[List[str]] = None) -> None:
    """CLI: ``plan`` / ``rows`` (the training rows), ``notebook`` (the Lambda edition), ``show`` (a notebook's plan).

    :param argv: Arguments; None = ``sys.argv[1:]``.
    :type argv: Optional[List[str]]
    """
    args = build_parser().parse_args(argv)
    if args.command in ("plan", "rows"):
        manifest_path = args.manifest if args.manifest is not None else dataset_manifest_path(args.export)
        if args.command == "rows" and args.out.exists():
            raise SystemExit(f"{args.out} exists: the rows step writes a fresh folder")
        plan = prepare_rows(args.export, args.train_rows, args.seed, args.keep_rnl, manifest_path)
        print("\n".join(plan_lines(plan)))
        if args.command == "rows":
            write_rows(plan, args.export, args.out, manifest_path, str(args.train_rows), args.seed, args.workers)
            print(f"wrote {args.out} (train {plan.train.num_rows:,} rows, val {plan.val.num_rows:,} rows)")
        return
    if args.command == "notebook":
        if args.revision != PLACEHOLDER and not re.fullmatch(r"[0-9a-f]{40}", args.revision):
            raise SystemExit(f"revision must be a 40-hex sha or {PLACEHOLDER}")
        rows, pages = notebook_numbers(args)
        write_notebook(args.revision, rows, pages, args.out_dir, args.src)
        return
    plan = notebook_plan(read_json(args.notebook))
    if args.require_pinned and not plan["pinned"]:
        raise SystemExit(f"{args.notebook} is not pinned (dataset revision {plan['revision']!r})")
    if args.field is None:
        print(json.dumps(plan, indent=1))
    else:
        value = plan[args.field]
        print(" ".join(str(v) for v in value) if isinstance(value, list) else value)


if __name__ == "__main__":
    main()
