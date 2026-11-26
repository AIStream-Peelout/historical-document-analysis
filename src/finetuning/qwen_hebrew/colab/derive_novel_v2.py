# File name: derive_novel_v2.py
# Date: 11/19/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Derive the novel_v2 notebook (the genizah_novel_v2 pages only, warm start novel-v1 step 326) and build its training rows.

novel_v2 is novel_v1's recipe on the next novel dataset, ``genizah_novel_v2`` (the Judaeo-Arabic / Arabic round's
line-assembled edition pages, in the KTIV ``FEATURES`` schema, splits ``train_page`` and ``val``, the script-neutral
prompt on every row, and its images-once export). No replay in this run: the replay quota is 0, no outside set is added
and no script is upweighted. The schedule is novel_v1's: every train page 2 or 3 times (``auto`` = 2.2 passes, whole
steps of batch 1 x accumulation 8), LR 5e-5 on a fresh cosine, 3 % warm-up, seed 3407, 50-step saves, evals every 100
steps, exact resume, Hub-confirmed quarter-point milestones, the step-0 eval and the train-loss alarm.

**Reuse.** Every function of :mod:`derive_novel_v1` is reused: it is re-bound to this module's namespace
(:func:`rebind`), so it reads this round's names (repositories, run name, warm start, section, the seeded draw
``novel_v2/extras``) instead of novel_v1's. Only what differs is written here:

- *the dataset's own conventions*, handled explicitly and never by relabelling the export:

  - section ``ja_ar_page`` (novel_v1: ``novel_page``): :data:`SECTION`, read by the rows step's row-kind check, the
    notebook's hygiene gate and the round's gates (:data:`BUILD`); rows of another section are refused;
  - ``label_source`` ``legacy_store_edition`` on every row (novel_v1: several sources): the ledger sources of the
    manifest map to it through ``ja_ar_build.build_rows.LABEL_SOURCES`` (:data:`BUILD`; novel_v1's map does not know
    them); one source, so the pass plan's per-source cap is the whole set's;
  - the manifest is ``<dataset>_build/manifest.jsonl`` (novel_v1: ``<dataset>/manifest.jsonl``):
    :func:`dataset_manifest_path`;
  - the manifest's split is ``train_page`` / ``val`` (novel_v1: ``train`` / ``val``): :func:`read_manifest` maps it to
    the rows step's names (:data:`MANIFEST_SPLITS`, the original kept as ``manifest_split``);
  - the manifest carries a novelty ``tier`` and no ledger ``verdict``: :func:`document_record` records the tier and
    the verdict :data:`TIER_VERDICTS` gives it (``novel_strict`` -> ``novel``; any other tier is not novel);
  - ``stats.json`` has no ``name`` (the dataset is named from the export's source) and no per-split script counts (the
    round's script gate checks the whole-dataset counts instead, :func:`gate_scripts`), and its per-source counts name
    the internal ledger sources: the rows step writes :func:`hub_stats` (per-source counts under the rows'
    ``label_source``, no build date, no holder count) instead of copying it;
- *the warm start*: novel-v1 step 326, pinned by commit (``hub_verification.json`` of the novel_v1 Lambda run);
- *the texts*: the card, the notebook intro and the cells that name the warm start or this round;
- *the round's tooling* as subcommands (novel_v1 kept them as files in its git-ignored round folder): ``gates`` runs
  novel_v1's gates runner re-bound to this round, ``prepush`` repeats the push's checks and ``stage`` stages the files
  of the Hub push (both without the Hub). The upload itself is the user's (the runbook).

Outputs that hold rows or images are refused on the boot disk (:func:`on_boot_disk`): give a new folder on the NAS.

Usage::

    .venv/bin/python -m src.finetuning.qwen_hebrew.colab.derive_novel_v2 plan --export <dataset>_images_once [--train-rows N] [--keep-rnl]
    .venv/bin/python -m src.finetuning.qwen_hebrew.colab.derive_novel_v2 rows --export <dataset>_images_once --out DIR [--keep-rnl]
    .venv/bin/python -m src.finetuning.qwen_hebrew.colab.derive_novel_v2 gates --rows DIR --cairo-dir CAIRO [--min-long-side 800]
    .venv/bin/python -m src.finetuning.qwen_hebrew.colab.derive_novel_v2 prepush --rows DIR
    .venv/bin/python -m src.finetuning.qwen_hebrew.colab.derive_novel_v2 stage --rows DIR --stage NEWDIR
    .venv/bin/python -m src.finetuning.qwen_hebrew.colab.derive_novel_v2 notebook --revision <40-hex sha | PIN-AFTER-PUSH> \\
        (--plan DIR/mixture.json | --pages N [--train-rows N]) [--out-dir DIR]
    .venv/bin/python -m src.finetuning.qwen_hebrew.colab.derive_novel_v2 show --notebook NB [--field eval_steps] [--require-pinned]
"""
import argparse
import importlib.util
import json
import os
import re
import shutil
import tarfile
import types
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pyarrow.parquet as pq

from src.finetuning.qwen_hebrew import build_v22_mixture as bvm
from src.finetuning.qwen_hebrew.build_pgp_arabic_editions import TASK as EDITION_TASK
from src.finetuning.qwen_hebrew.colab import derive_novel_v1 as v1
from src.finetuning.qwen_hebrew.colab import derive_v24a as d4
from src.finetuning.qwen_hebrew.colab import host_lambda
from src.finetuning.qwen_hebrew.colab.derive_v22b import _replace_once
from src.finetuning.qwen_hebrew.images_once import SHA_COLUMN
from src.finetuning.qwen_hebrew.ja_ar_build import build_rows as ja_ar_rows
from src.finetuning.qwen_hebrew.ja_ar_build import release_audit
from src.finetuning.qwen_hebrew.novel_build import build as novel_build_v1
from src.finetuning.qwen_hebrew.prompts import SCRIPT_NEUTRAL_PROMPT

# ----------------------------------------------------------------------------- novel_v1's namespace, then this round's
# Every name of derive_novel_v1 (its constants, its imports and its functions) is copied here first; the names below
# replace the ones that differ, and rebind() then gives every novel_v1 function this namespace as its globals.
globals().update({k: v for k, v in vars(v1).items() if not k.startswith("__")})

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
SRC_NOTEBOOK = HERE / "genizah_v22b.ipynb"

# ----------------------------------------------------------------------------- names (one distinct name per run)
ROUND = "novel_v2"
TAG = "novel-v2"                       # derive_v23a.Arm tag: checkpoint repo, output and resume folders, merged folder
RUN_NAME = "genizah_novel_v2"          # trainer and W&B run name (W&B project qwen-hebrew-finetune)
DATA_REPO = "isaacmg/genizah_novel_v2_train"   # the training rows (new, private); isaacmg/genizah_novel_v2 is the source set
SOURCE_REPO = "isaacmg/genizah_novel_v2"       # the source set: never written by this round
LOCAL_LEAF = "genizah_novel_v2_train"
COLAB_NOTEBOOK = "genizah_novel_v2.ipynb"                       # the in-memory Colab edition (never written)
NOTEBOOK, SETUP_SCRIPT = host_lambda.lambda_files(COLAB_NOTEBOOK)   # genizah_novel_v2_lambda.ipynb, lambda_setup.sh
CKPT_REPO = f"isaacmg/qwen3-vl-8b-hebrew-{TAG}-ckpt"
MERGED_REPO = "isaacmg/qwen3-vl-8b-hebrew-novel-v2"
RUNBOOK = "logs/next_round/ja_ar_v1/novel1119/TRAINING_NOVEL_V2_RUNBOOK.md"
PUSH_SCRIPT = f"the user-run push ({RUNBOOK}, step 5)"
REVISION_FILE = "logs/next_round/ja_ar_v1/novel1119/push_novel_v2_train.revision"

# ----------------------------------------------------------------------------- warm start: novel-v1 step 326
# The novel_v1 Lambda run's final checkpoint, the one the production-path eval read (novel-v1 step 326 against v2.5a
# step 1200). hub_verification.json of that run: revision 944377c9..., title "Training in progress, step 326, checkpoint",
# global_step 326, private. The model cell checks trainer_state.json says step 326.
WARM_PLACEHOLDER = "PIN-BEFORE-LAUNCH"
WARM_CKPT_REPO = "isaacmg/qwen3-vl-8b-hebrew-novel-v1-ckpt"
WARM_REVISION = "944377c9d2626805202ad11679257b928d77d910"
WARM_STEP = 326
WARM_NAME = f"novel-v1 step {WARM_STEP}"

# ----------------------------------------------------------------------------- the pass plan (novel_v1's rule)
COMPONENT = ROUND                      # names the seeded draw of the extras (``novel_v2/extras``)
EXTRAS_DRAW = f"{COMPONENT}/extras"
#: train pages of the reference plan of a placeholder notebook (the export's train_page split)
REFERENCE_PAGES = 1739

# ----------------------------------------------------------------------------- the dataset's conventions
SECTION = ja_ar_rows.SECTION           # ja_ar_page (novel_v1: novel_page)
TASK = EDITION_TASK                    # fragment_transcribe
V1_SECTION = novel_build_v1.SECTION    # novel_page: refused in this round's rows
#: the manifest's split names -> the rows step's (the original stays in ``manifest_split``)
MANIFEST_SPLITS = {TRAIN_SPLIT: "train", "train": "train", VAL_SPLIT: "val"}   # noqa: F821  (train_page / val)
#: novelty tier of a manifest record -> its ledger verdict (anything else is not novel)
TIER_VERDICTS = {"novel_strict": "novel"}
MANIFEST_SUFFIX = "_build"             # <dataset>_build/manifest.jsonl beside the saved dataset
#: the build constants novel_v1's functions read as ``novel_build.*``, pointed at this round's build
BUILD = types.SimpleNamespace(SECTION=SECTION, TASK=TASK, LABEL_SOURCES=dict(ja_ar_rows.LABEL_SOURCES),
                              SPLIT_NAMES=dict(novel_build_v1.SPLIT_NAMES), HYGIENE_CHARS=ja_ar_rows.HYGIENE_CHARS,
                              FORBIDDEN_README=novel_build_v1.FORBIDDEN_README, LEGACY_STORE_SOURCES=ja_ar_rows.LEGACY_STORE_SOURCES)
novel_build = BUILD
#: internal ledger source names (and their prefixes): never on a Hub surface
LEDGER_NAMES: Tuple[str, ...] = tuple(sorted(set(ja_ar_rows.LEGACY_STORE_SOURCES)
                                             | {s.split("_")[0] for s in ja_ar_rows.LEGACY_STORE_SOURCES}))
FORBIDDEN_CARD: Tuple[str, ...] = tuple(dict.fromkeys((*v1.FORBIDDEN_CARD, *LEDGER_NAMES, "legacy_store_targeted")))
#: strings no row text column may hold (the push's list: no storage path check on Hebrew / Arabic text)
ROW_FORBIDDEN: Tuple[str, ...] = tuple(dict.fromkeys((*bvm.FORBIDDEN_CARD_STRINGS, *LEDGER_NAMES, "genizah.org")))
ROW_TEXT_COLUMNS: Tuple[str, ...] = ("question", "answer", "stem", "label_source", "section", "task", "source")
#: stats keys left out of the Hub copy (a build date, a count that names a holding)
STATS_DROPPED: Tuple[str, ...] = ("built", "rnl_rows")
SIDECARS: Tuple[str, ...] = ("README.md", "PRIVATE.md", "mixture.json", "stats.json", "manifest.json")
TAR_LIMIT = int(2.4 * 2 ** 30)         # HUB IMAGES AS ARCHIVES: tars of at most 2.4 GiB

# ----------------------------------------------------------------------------- the round's gates (novel_v1's runner)
V1_GATES = REPO_ROOT / "logs" / "next_round" / "novel_v1" / "run_notebook_gates_novel_v1.py"
MARKER = "notebook_gates_novel_v2.json"
GATES_VERSION = "novel_v2 gates 2026-11-19"
MIN_LONG_SIDE = 800

# ----------------------------------------------------------------------------- notebook texts naming this round
DATA_TITLE = f"# Cell 2 — data: {LOCAL_LEAF} (images-once, MAP-STYLE, materialised mixture — no streaming)"
PIN_GUARD = (f'# {ROUND}: never train on an unpinned dataset ({PUSH_SCRIPT} pins the pushed revision and regenerates this notebook)\n'
             'if V22_REVISION == "PIN-AFTER-PUSH" or len(V22_REVISION) != 40 or any(_c not in "0123456789abcdef" for _c in V22_REVISION):\n'
             f'    raise RuntimeError(f"dataset revision {{V22_REVISION!r}} is not a pinned 40-hex sha: run {PUSH_SCRIPT}, "\n'
             '                       "which pins the pushed revision and regenerates this notebook")\n')
REVISION_LINE = re.compile(r'^V22_REVISION = "([^"]+)"   # ' + re.escape(ROUND)
                           + r" mixture .*\(([\d,]+) train rows over ([\d,]+) novel pages\)$", re.M)
PLAN_GATE = v1.PLAN_GATE.replace("derive_novel_v1.py", "derive_novel_v2.py")
PASS_GATE = v1.PASS_GATE.replace("derive_novel_v1.pass_plan", "derive_novel_v2.pass_plan")
MODEL_TITLE = f"# Cell 3 — model at page resolution + THE MERGER KNOB + warm start ({WARM_NAME})"
NEW_WARM = (f'# Warm start: {WARM_NAME} — the end of the novel_v1 run (novel pages only, LoRA r16, merger frozen), the\n'
            '# checkpoint the production-path eval read. Weights only: this run has its own optimizer and a fresh cosine schedule.\n'
            f'WARM_CKPT_REPO = "{WARM_CKPT_REPO}"   # private: the token must read it\n'
            f'WARM_REVISION = "{{warm_revision}}"  # commit "Training in progress, step {WARM_STEP}, checkpoint"\n'
            f'WARM_STEP = {WARM_STEP}\n')
NEW_WARM_DOWNLOAD = _replace_once(d4.NEW_WARM_DOWNLOAD, "# v24a: the pinned commit", f"# {ROUND}: the pinned commit", "warm step comment")
TRAIN_TITLE = f"# Cell 6 — {ROUND} (genizah_novel_v2 pages only, no replay, warm start {WARM_NAME}) training:"
V1_LR_COMMENT = f"# v2.5a's rate on a fresh cosine: the novel pages on top of v2.5a step {WARM_STEP}"
LR_COMMENT = f"# novel_v1's rate on a fresh cosine: the genizah_novel_v2 pages on top of {WARM_NAME}"
STEP0_START = f"# ---- {ROUND}: step-0 eval of the warm start"
CALLBACKS_START = f"# ---- {ROUND}: checkpoints pushed AND confirmed on the Hub at the quarter points of the run"
OLD_MERGED = f'MERGED_REPO = "isaacmg/qwen3-vl-8b-hebrew-{TAG}-merged"'

#: names of this module that replace a novel_v1 value (string defaults equal to the novel_v1 value are swapped too)
OVERRIDES: Tuple[str, ...] = (
    "HERE", "SRC_NOTEBOOK", "ROUND", "TAG", "RUN_NAME", "DATA_REPO", "LOCAL_LEAF", "COLAB_NOTEBOOK", "NOTEBOOK",
    "SETUP_SCRIPT", "CKPT_REPO", "MERGED_REPO", "PUSH_SCRIPT", "WARM_PLACEHOLDER", "WARM_CKPT_REPO", "WARM_REVISION",
    "WARM_STEP", "COMPONENT", "EXTRAS_DRAW", "REFERENCE_PAGES", "SECTION", "TASK", "novel_build", "FORBIDDEN_CARD",
    "DATA_TITLE", "PIN_GUARD", "REVISION_LINE", "PLAN_GATE", "PASS_GATE", "MODEL_TITLE", "NEW_WARM", "NEW_WARM_DOWNLOAD",
    "TRAIN_TITLE", "STEP0_START", "CALLBACKS_START", "OLD_MERGED")


# ----------------------------------------------------------------------------- re-binding novel_v1's functions


def default_swaps(old: Mapping[str, Any], new: Mapping[str, Any], names: Sequence[str] = OVERRIDES) -> Dict[str, Any]:
    """String (and path) values of novel_v1 this round replaces, for swapping function defaults bound at definition.

    :param old: novel_v1's namespace.
    :type old: Mapping[str, Any]
    :param new: This round's namespace.
    :type new: Mapping[str, Any]
    :param names: Names this round replaces.
    :type names: Sequence[str]
    :return: novel_v1 value (as ``repr``) -> this round's value, for the names whose value changed.
    :rtype: Dict[str, Any]
    """
    return {repr(old[n]): new[n] for n in names
            if n in old and isinstance(old[n], (str, Path)) and old[n] != new[n]}


def rebind(func: types.FunctionType, namespace: Dict[str, Any], swaps: Mapping[str, Any]) -> types.FunctionType:
    """A copy of a function whose globals are ``namespace``, with defaults equal to a replaced value swapped.

    :param func: A novel_v1 function.
    :type func: types.FunctionType
    :param namespace: The globals the copy reads (this module's).
    :type namespace: Dict[str, Any]
    :param swaps: :func:`default_swaps`.
    :type swaps: Mapping[str, Any]
    :return: The copy (same code, name, docstring and annotations; ``__module__`` this module).
    :rtype: types.FunctionType
    """
    defaults = None if func.__defaults__ is None else tuple(
        swaps.get(repr(d), d) if isinstance(d, (str, Path)) else d for d in func.__defaults__)
    out = types.FunctionType(func.__code__, namespace, func.__name__, defaults, func.__closure__)
    out.__kwdefaults__ = None if func.__kwdefaults__ is None else {
        k: swaps.get(repr(d), d) if isinstance(d, (str, Path)) else d for k, d in func.__kwdefaults__.items()}
    out.__doc__, out.__qualname__, out.__annotations__ = func.__doc__, func.__qualname__, dict(func.__annotations__)
    out.__module__ = namespace["__name__"]
    return out


def rebind_module(module: types.ModuleType, namespace: Dict[str, Any], swaps: Mapping[str, Any]) -> List[str]:
    """Re-bind every function a module defines into ``namespace`` (under the same names).

    :param module: The module (its own functions only, not the ones it imports).
    :type module: types.ModuleType
    :param namespace: Target globals.
    :type namespace: Dict[str, Any]
    :param swaps: :func:`default_swaps`.
    :type swaps: Mapping[str, Any]
    :return: The names re-bound.
    :rtype: List[str]
    """
    names = [k for k, f in vars(module).items() if isinstance(f, types.FunctionType) and f.__module__ == module.__name__]
    for name in names:
        namespace[name] = rebind(getattr(module, name), namespace, swaps)
    return names


REBOUND = rebind_module(v1, globals(), default_swaps(vars(v1), globals()))
# the re-bound novel_v1 versions the functions below wrap
_v1_prepare_rows = prepare_rows            # noqa: F821  (re-bound above)
_v1_document_record = document_record      # noqa: F821
_v1_derive = derive                        # noqa: F821
_v1_build_parser = build_parser            # noqa: F821
_v1_main = main                            # noqa: F821


# ----------------------------------------------------------------------------- the dataset's conventions


def dataset_manifest_path(export_dir: Path) -> Path:
    """The dataset's ``manifest.jsonl``: beside the saved dataset the export names, or in its ``_build`` folder.

    :param export_dir: The images-once export (its ``manifest.json`` names its source).
    :type export_dir: Path
    :return: ``<dataset>/manifest.jsonl`` when it exists (novel_v1's layout), else ``<dataset>_build/manifest.jsonl``.
    :rtype: Path
    :raises FileNotFoundError: When neither exists.
    """
    source = Path(read_json(export_dir / "manifest.json")["source"])   # noqa: F821
    candidates = [source / "manifest.jsonl", source.parent / f"{source.name}{MANIFEST_SUFFIX}" / "manifest.jsonl"]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"no manifest.jsonl for {export_dir.name}: looked at {[str(p) for p in candidates]} (--manifest)")


def read_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    """Per-row records of the dataset by stem, the split mapped to the rows step's names (:data:`MANIFEST_SPLITS`).

    :param path: ``manifest.jsonl``.
    :type path: Path
    :return: stem -> record (``split`` = ``train`` / ``val``; ``manifest_split`` = the manifest's own value).
    :rtype: Dict[str, Dict[str, Any]]
    :raises ValueError: When a stem is listed twice or a split is unknown.
    """
    records = v1.read_manifest(path)
    unknown = sorted({str(r.get("split")) for r in records.values()} - set(MANIFEST_SPLITS))
    if unknown:
        raise ValueError(f"{path}: splits {unknown} are none of {sorted(MANIFEST_SPLITS)}")
    return {s: dict(r, split=MANIFEST_SPLITS[r["split"]], manifest_split=r["split"]) for s, r in records.items()}


def tier_verdict(record: Mapping[str, Any]) -> Optional[str]:
    """The ledger verdict of a manifest record: its own ``verdict``, else the one its novelty tier gives.

    :param record: Manifest record.
    :type record: Mapping[str, Any]
    :return: ``novel`` for a :data:`TIER_VERDICTS` tier, the tier itself for another one, None without either.
    :rtype: Optional[str]
    """
    if record.get("verdict"):
        return record["verdict"]
    tier = record.get("tier")
    return TIER_VERDICTS.get(tier, tier) if tier else None


def document_record(record: Mapping[str, Any], split: str, rows: int) -> Dict[str, Any]:
    """novel_v1's ``documents.jsonl`` record plus the novelty tier, the verdict from :func:`tier_verdict`.

    :param record: Manifest record (:func:`read_manifest`).
    :type record: Mapping[str, Any]
    :param split: ``train`` or ``val``.
    :type split: str
    :param rows: Rows of the page in the split.
    :type rows: int
    :return: Record (no paths, no URLs, no ledger source name).
    :rtype: Dict[str, Any]
    """
    return dict(_v1_document_record(record, split, rows), tier=record.get("tier"), verdict=tier_verdict(record))


def dataset_name(export_dir: Path, stats: Mapping[str, Any]) -> str:
    """The dataset's name: ``stats.json``'s ``name``, else the saved dataset's folder the export names.

    :param export_dir: The images-once export.
    :type export_dir: Path
    :param stats: The dataset's ``stats.json``.
    :type stats: Mapping[str, Any]
    :return: e.g. ``genizah_novel_v2``.
    :rtype: str
    """
    return stats.get("name") or Path(read_json(export_dir / "manifest.json")["source"]).name   # noqa: F821


def prepare_rows(export_dir: Path, request: Union[str, int] = AUTO, seed: int = SEED, keep_rnl: bool = False,  # noqa: F821
                 manifest_path: Optional[Path] = None) -> "RowsPlan":  # noqa: F821
    """novel_v1's rows plan on this round's dataset (section :data:`SECTION`, nothing written).

    :param export_dir: The dataset's images-once export.
    :type export_dir: Path
    :param request: ``auto`` or a whole number of train rows.
    :type request: Union[str, int]
    :param seed: Build seed.
    :type seed: int
    :param keep_rnl: Keep Russian National Library rows (default: left out and recorded, as novel_v1).
    :type keep_rnl: bool
    :param manifest_path: The dataset's ``manifest.jsonl`` (None: :func:`dataset_manifest_path`).
    :type manifest_path: Optional[Path]
    :return: The plan (``record["dataset"]`` named by :func:`dataset_name`).
    :rtype: RowsPlan
    :raises ValueError: As novel_v1's, including rows of another section than :data:`SECTION`.
    """
    plan = _v1_prepare_rows(export_dir, request, seed, keep_rnl, manifest_path)
    plan.record["dataset"] = dataset_name(export_dir, plan.stats)
    return plan


def hub_stats(stats: Mapping[str, Any], label_sources: Mapping[str, str] = BUILD.LABEL_SOURCES) -> Dict[str, Any]:
    """The dataset's stats for the Hub copy: per-source counts under the rows' ``label_source``, no date, no holder count.

    :param stats: The dataset's ``stats.json``.
    :type stats: Mapping[str, Any]
    :param label_sources: Ledger source -> ``label_source``.
    :type label_sources: Mapping[str, str]
    :return: Stats; every other key as it is.
    :rtype: Dict[str, Any]
    :raises ValueError: On a ledger source without a label source, or a result that still names a forbidden string.
    """
    out: Dict[str, Any] = {}
    for key, value in stats.items():
        if key in STATS_DROPPED:
            continue
        if key.endswith("_by_source") and isinstance(value, dict):
            unknown = sorted(set(value) - set(label_sources))
            if unknown:
                raise ValueError(f"stats {key}: sources without a label source {unknown}")
            summed: Counter = Counter()
            for source, n in value.items():
                summed[label_sources[source]] += n
            out[key] = dict(sorted(summed.items()))
        else:
            out[key] = value
    check_card(json.dumps(out, ensure_ascii=False))   # noqa: F821
    return out


def card_text(record: Mapping[str, Any], stats: Mapping[str, Any], mixture_name: str = LOCAL_LEAF) -> str:
    """``README.md`` of the training rows (the PRIVATE banner first when the dataset is private).

    :param record: The plan (:attr:`RowsPlan.record`).
    :type record: Mapping[str, Any]
    :param stats: The dataset's ``stats.json``.
    :type stats: Mapping[str, Any]
    :param mixture_name: Name of this training export.
    :return: Markdown.
    :rtype: str
    :raises ValueError: When the text names a forbidden source, host or storage path (:data:`FORBIDDEN_CARD`).
    """
    lines = [f"# {mixture_name} (training rows of the {ROUND} fine-tune)", ""]
    if stats.get("private", True):
        lines += [f"{PRIVATE_BANNER} {'; '.join(stats.get('private_reasons') or [])}".rstrip(), ""]   # noqa: F821
    lines += [f"The rows a Qwen3-VL page reader trains on in the `{ROUND}` run, materialised from the dataset "
              f"`{record['dataset']}` (built {record['dataset_built']}) by `src/finetuning/qwen_hebrew/colab/derive_novel_v2.py "
              "rows`: edition pages of documents no earlier fine-tune trained on, nothing else (no replay, no outside "
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
    lines += ["", "`mixture.json` holds the plan; `stats.json` is the dataset's own (per-source counts under the rows' "
                  "`label_source`); `manifest.json` lists the images-once splits. Private, internal training material: no "
                  "reproduction or redistribution of the text or the images."]
    text = "\n".join(lines) + "\n"
    check_card(text)   # noqa: F821
    return text


def write_rows(plan: "RowsPlan", export_dir: Path, out_dir: Path, manifest_path: Path, request: str, seed: int,  # noqa: F821
               workers: int = 16, *, mixture_name: str = LOCAL_LEAF) -> Dict[str, Any]:
    """Write prepared training rows as an images-once folder (novel_v1's layout; ``stats.json`` = :func:`hub_stats`).

    :param plan: :func:`prepare_rows` of ``export_dir``.
    :type plan: RowsPlan
    :param export_dir: The dataset's images-once export (the images are copied from it).
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
    :param mixture_name: Export identity for the mixture and its card. Defaults to the original run.
    :return: ``mixture.json`` content.
    :rtype: Dict[str, Any]
    :raises FileExistsError: When ``out_dir`` exists.
    """
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists: the rows step writes a fresh folder")
    card = card_text(plan.record, plan.stats, mixture_name)
    stats = hub_stats(plan.stats)
    wanted = {sha: export_dir / "images" / f"{sha}.jpg" for table in (plan.train, plan.val) for sha in table[SHA_COLUMN].to_pylist()}
    out_dir.mkdir(parents=True)
    sizes = bvm.copy_images(wanted, out_dir / "images", workers)
    bvm.verify_images([plan.train, plan.val], out_dir / "images")
    bvm.write_rows(plan.train, plan.meta, out_dir / "rows" / "train.parquet")
    bvm.write_rows(plan.val, plan.meta, out_dir / "rows" / "val.parquet")
    splits = {"train": bvm.split_manifest(plan.train, sizes), "val": bvm.split_manifest(plan.val, sizes)}
    manifest = {"source": export_dir.name, "image_column": plan.meta["image_column"], "sha_column": SHA_COLUMN, "splits": splits,
                "n_rows": plan.train.num_rows + plan.val.num_rows, "n_images": len(sizes), "image_bytes": sum(sizes.values()),
                "row_image_bytes": sum(s["row_image_bytes"] for s in splits.values())}
    mixture = {"name": mixture_name, "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "config": plan.record}
    bvm.write_json(out_dir / "stats.json", stats)
    if (export_dir / "PRIVATE.md").is_file():
        shutil.copy2(export_dir / "PRIVATE.md", out_dir / "PRIVATE.md")
    (out_dir / "README.md").write_text(card, encoding="utf-8")
    (out_dir / "documents.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in plan.documents), encoding="utf-8")
    bvm.write_json(out_dir / "build_record.json", {
        "export_dir": str(export_dir), "manifest_path": str(manifest_path), "manifest_sha256": file_sha256(manifest_path),  # noqa: F821
        "stats_sha256": plan.record["dataset_stats_sha256"], "request": request, "seed": seed, "keep_rnl": plan.record["keep_rnl"],
        "rnl_dropped": [{"stem": d["stem"], "split": d.get("split"), "script": d.get("script")} for d in plan.dropped]})
    bvm.write_json(out_dir / "mixture.json", mixture)
    bvm.write_json(out_dir / "manifest.json", manifest)          # written last: marks a finished folder
    return mixture


# ----------------------------------------------------------------------------- the notebook


def warm_start_pinned(revision: Optional[str] = None) -> bool:
    """Whether the warm start is pinned to a commit.

    :param revision: Warm-start revision (None: :data:`WARM_REVISION` as it is now).
    :type revision: Optional[str]
    :return: True for a 40-hex sha (False for :data:`WARM_PLACEHOLDER`).
    :rtype: bool
    """
    return bool(re.fullmatch(r"[0-9a-f]{40}", WARM_REVISION if revision is None else revision))


def intro_md(train_rows: int, pages: int) -> str:
    """Markdown intro of the notebook.

    :param train_rows: Train rows.
    :type train_rows: int
    :param pages: Train pages.
    :type pages: int
    :return: Text of cell 0 (the Lambda edition puts its launch notes in front of it).
    :rtype: str
    """
    steps = max_steps_for(train_rows)   # noqa: F821
    milestones = ", ".join(f"{s:,}" for s in milestone_steps(steps))   # noqa: F821
    evals = ", ".join(f"{s:,}" for s in eval_steps(steps))   # noqa: F821
    return f"""# Genizah {ROUND} — the genizah_novel_v2 pages only, no replay, on top of novel-v1

Same training stack as novel_v1 and v2.5a (pinned install triplet, 6.5/7 MP resolution contract, LoRA r16 on tower +
language, merger FROZEN, adamw_8bit, cosine, seed {SEED}, fixed row order with exact resume, {SAVE_STEPS}-step saves,
prepared-dataloader gate before step 1) and novel_v1's schedule. What changes:

| | novel_v1 | this run |
|---|---|---|
| Warm start | v2.5a step 1200 | **{WARM_NAME}** (`{WARM_CKPT_REPO}@{WARM_REVISION[:8]}`), weights only: fresh optimizer, fresh cosine |
| Data | genizah_novel_v1: 1,184 novel pages, several sources | `{DATA_REPO}`: the pages of `{SOURCE_REPO}` (edition pages of documents no earlier fine-tune trained on, section `{SECTION}`); replay quota 0, no outside sets, no script upweighted |
| Rows / steps | 2,608 / 326 | **{train_rows:,} / {steps:,}**: {pages:,} train pages, every page 2 or 3 times, the extra rows drawn uniformly over all pages (natural shares), no source above {MAX_PASSES} passes (batch 1 x accumulation 8) |
| Prompt | script-neutral | the script-neutral prompt on every row (`prompts.SCRIPT_NEUTRAL_PROMPT`) |
| Learning rate | 5e-5, 3% warm-up | {LR}, {float(WARMUP_RATIO):.0%} warm-up ({warmup_steps(train_rows)} steps), fresh cosine |
| Checkpoints | every 50 steps; 100, 200, 250, 326 confirmed on the Hub | every {SAVE_STEPS} steps (resume); steps {milestones} pushed **and confirmed** on the Hub (the quarter points, rounded up to a multiple of {MILESTONE_GRAIN}) |
| Gates | plan, pass rule, neutral prompt, novel page rows, loss alarm, step-0 eval | the same, on this round's rows: a **train-loss alarm** (the first logged train loss above {LOSS_ALARM_FACTOR:g} x max({LOSS_ALARM_REFERENCE:.2f}, the step-0 eval loss), at least {loss_alarm_threshold(0):.2f}, stops the run) and a **step-0 eval** of the warm start (warm-start stop rule) |
| Session | Lambda A100-40GB | Lambda A100-40GB: one pass ≈ {steps * MEASURED_S_PER_STEP / 3600:.1f} h at {MEASURED_S_PER_STEP} s/step; safety stop {lambda_budget_h(train_rows):.0f} h |

The data cell **refuses to run** until the dataset revision is pinned ({PUSH_SCRIPT} pins it and regenerates this
notebook); the model cell checks that the warm start is novel-v1's step-{WARM_STEP} checkpoint.

**To resume after a session ends: run every cell again, unedited.** The notebook finds the newest checkpoint on the
Hub, checks the stored schedule and row order, and continues with the rows not yet trained.

Run name `{RUN_NAME}`; checkpoints in `{CKPT_REPO}` (new, private); merged model `{MERGED_REPO}` (private). Steps
{evals} are scored in one sequential queue on the Studio: PGP-131 first, then religious-140, then the Arabic-script set
(covered images) as a regression monitor.
"""   # noqa: F821


def derive(nb: Dict, revision: str, train_rows: int, pages: int) -> Dict:
    """novel_v1's derivation with this round's names, warm start and texts (the Colab edition, kept in memory).

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
    :raises ValueError: As novel_v1's, or when the learning-rate comment is not found once.
    """
    out = _v1_derive(nb, revision, train_rows, pages)
    c8 = "".join(out["cells"][8]["source"])
    out["cells"][8]["source"] = _replace_once(c8, V1_LR_COMMENT, LR_COMMENT, "lr comment").splitlines(keepends=True)
    return out


# ----------------------------------------------------------------------------- boot-disk guard


def on_boot_disk(path: Path, home: Optional[Path] = None) -> bool:
    """Whether a path (or its nearest existing parent) is on the volume of the home folder (the boot disk).

    :param path: Output folder.
    :type path: Path
    :param home: Reference folder (None: the home folder).
    :type home: Optional[Path]
    :return: True when both are on one device.
    :rtype: bool
    """
    probe = Path(path).absolute()
    while not probe.exists():
        probe = probe.parent
    return os.stat(probe).st_dev == os.stat(home if home is not None else Path.home()).st_dev


def refuse_boot_disk(path: Path, what: str) -> None:
    """Stop when an output that holds rows or images would land on the boot disk.

    :param path: Output folder.
    :type path: Path
    :param what: What the folder is for (the message).
    :type what: str
    :raises SystemExit: When :func:`on_boot_disk`.
    """
    if on_boot_disk(path):
        raise SystemExit(f"{path} is on the boot disk: the {what} goes to a new, newly named folder on the NAS")


# ----------------------------------------------------------------------------- the round's gates


def load_v1_gates(path: Path = V1_GATES) -> types.ModuleType:
    """novel_v1's gates runner (git-ignored, in its round folder) as a module.

    :param path: ``run_notebook_gates_novel_v1.py``.
    :type path: Path
    :return: The module.
    :rtype: types.ModuleType
    :raises FileNotFoundError: When the file is not on this machine.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} is not here: novel_v2's gates re-use novel_v1's runner")
    spec = importlib.util.spec_from_file_location("run_notebook_gates_novel_v1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stats_for_script_gate(stats: Mapping[str, Any], val: Sequence[Mapping[str, Any]], documents: Mapping[str, Mapping[str, Any]],
                          dropped: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The stats layout novel_v1's script gate reads, from this dataset's whole-set ``rows_by_script``.

    The dataset's stats count rows per script over both splits (no per-split counts): the train pages per script are
    those counts less the val rows kept and the val rows left out; novel_v1's gate then subtracts the train rows left out.

    :param stats: The dataset's ``stats.json`` (``rows_by_script``).
    :type stats: Mapping[str, Any]
    :param val: Val rows.
    :type val: Sequence[Mapping[str, Any]]
    :param documents: stem -> ``documents.jsonl`` record.
    :type documents: Mapping[str, Mapping[str, Any]]
    :param dropped: Rows the rows step left out (``split``, ``script``).
    :type dropped: Sequence[Mapping[str, Any]]
    :return: ``{"rows_by_source_script_split": {"all": {script: {train_page: n}}}}``.
    :rtype: Dict[str, Any]
    """
    train = Counter({s: int(n) for s, n in (stats.get("rows_by_script") or {}).items()})
    train.subtract(Counter(documents[r["stem"]]["script"] for r in val if r["stem"] in documents))
    train.subtract(Counter(d.get("script") for d in dropped if d.get("split") == "val"))
    return {"rows_by_source_script_split": {"all": {s: {TRAIN_SPLIT: n} for s, n in train.items()}}}   # noqa: F821


def gates_namespace(module: types.ModuleType, min_long_side: int = MIN_LONG_SIDE) -> Dict[str, Any]:
    """novel_v1's gates re-bound to this round: this module as ``dn``, :data:`BUILD` as ``novel_build``, this round's
    marker, and a script gate that reads the whole-set script counts (:func:`stats_for_script_gate`).

    :param module: :func:`load_v1_gates`.
    :type module: types.ModuleType
    :param min_long_side: Shortest long side an image may have (novel_v1: 800 px).
    :type min_long_side: int
    :return: The namespace (``main`` runs the gates).
    :rtype: Dict[str, Any]
    """
    import sys
    namespace = dict(vars(module))
    namespace.update(dn=sys.modules[__name__], novel_build=BUILD, MARKER=MARKER, GATES_VERSION=GATES_VERSION,
                     MIN_LONG_SIDE=int(min_long_side), FORBIDDEN_ANSWER=set("<>`") | set(BUILD.HYGIENE_CHARS))
    namespace["__name__"] = f"{__name__}.gates"
    rebind_module(module, namespace, {})
    v1_gate_scripts = namespace["gate_scripts"]

    def gate_scripts(config: Mapping[str, Any], train: Sequence[Mapping[str, Any]], val: Sequence[Mapping[str, Any]],
                     documents: Mapping[str, Mapping[str, Any]], stats: Mapping[str, Any], dropped: Sequence[Mapping[str, Any]],
                     tolerance: float) -> Tuple[Dict[str, Any], List[str]]:
        """Gate 4 of novel_v1 on the stats layout it reads (:func:`stats_for_script_gate`).

        :param config: ``mixture.json`` ``config``.
        :param train: Train rows.
        :param val: Val rows.
        :param documents: stem -> ``documents.jsonl`` record.
        :param stats: The rows folder's ``stats.json``.
        :param dropped: Rows the rows step left out.
        :param tolerance: Largest accepted drift of a script's row share from its page share.
        :return: (summary, problems).
        """
        return v1_gate_scripts(config, train, val, documents, stats_for_script_gate(stats, val, documents, dropped), dropped, tolerance)

    namespace["gate_scripts"] = gate_scripts
    return namespace


def run_gates(out: Path, cairo_dir: Optional[Path] = None, export: Optional[Path] = None,
              tolerance: float = SCRIPT_SHARE_TOLERANCE, workers: int = 8, min_long_side: int = MIN_LONG_SIDE,  # noqa: F821
              held_out: Optional[Callable] = None, gates_path: Path = V1_GATES) -> Dict[str, Any]:
    """Run novel_v1's eight gates and the notebook's data cell on this round's rows; write :data:`MARKER` on success.

    :param out: The rows folder (``rows --out``).
    :type out: Path
    :param cairo_dir: The data folder holding the benchmark registries.
    :type cairo_dir: Optional[Path]
    :param export: The images-once export (None: ``build_record.json``'s).
    :type export: Optional[Path]
    :param tolerance: Largest accepted drift of a script's row share from its page share.
    :type tolerance: float
    :param workers: Parallel image decodes.
    :type workers: int
    :param min_long_side: Shortest accepted long side of an image (novel_v1: 800 px).
    :type min_long_side: int
    :param held_out: Loader of the held-out sets (None: ``derive_v25a.held_out_sets``; tests stub it).
    :type held_out: Optional[Callable]
    :param gates_path: novel_v1's runner.
    :type gates_path: Path
    :return: The marker.
    :rtype: Dict[str, Any]
    :raises SystemExit: When a gate fails (nothing written).
    """
    ns = gates_namespace(load_v1_gates(gates_path), min_long_side)
    return ns["main"](out, cairo_dir, export, tolerance, workers, held_out if held_out is not None else ns["d5"].held_out_sets)


# ----------------------------------------------------------------------------- the push's checks and staging (no Hub)


def text_hits(text: str, words: Sequence[str]) -> List[str]:
    """Forbidden words in a text (lower-case comparison).

    :param text: Text.
    :type text: str
    :param words: Forbidden words.
    :type words: Sequence[str]
    :return: The words found.
    :rtype: List[str]
    """
    low = text.lower()
    return [w for w in words if w in low]


def prepush_problems(src: Path) -> List[str]:
    """Why built rows may not be pushed: novel_v1's push checks on this round's names (nothing written, no Hub).

    The gates passed on exactly these files (:data:`MARKER`: sha256 of both row files, ``mixture.json`` and
    ``documents.jsonl``; rows and pages as planned); the warm start is pinned; every question is the script-neutral
    prompt; no Russian National Library row unless the rows step kept them; no forbidden name in a row text column, nor a
    forbidden name or storage path in a sidecar.

    :param src: The rows folder.
    :type src: Path
    :return: Problems (empty: the push may go ahead).
    :rtype: List[str]
    """
    needed = ["rows/train.parquet", "rows/val.parquet", "README.md", "mixture.json", "stats.json", "manifest.json",
              "documents.jsonl", MARKER]
    missing = [f for f in needed if not (src / f).is_file()] + ([] if (src / "images").is_dir() else ["images/"])
    if missing:
        return [f"{src}: missing {missing} (rows, then gates)"]
    problems = []
    gates = read_json(src / MARKER)   # noqa: F821
    for key, path in (("train_sha256", src / "rows" / "train.parquet"), ("val_sha256", src / "rows" / "val.parquet"),
                      ("mixture_sha256", src / "mixture.json"), ("documents_sha256", src / "documents.jsonl")):
        if gates.get(key) != file_sha256(path):   # noqa: F821
            problems.append(f"{path.name} changed since the notebook gates passed: run the gates again")
    if not warm_start_pinned():
        problems.append(f"the warm start {WARM_CKPT_REPO}@{WARM_REVISION} is not pinned to a commit")
    config = read_json(src / "mixture.json")["config"]   # noqa: F821
    if (gates.get("train_rows"), gates.get("train_pages")) != (config["train_rows"], config["train_pages"]):
        problems.append(f"the gates ran on {gates.get('train_rows')} rows / {gates.get('train_pages')} pages, mixture.json plans "
                        f"{config['train_rows']} / {config['train_pages']}")
    documents = [json.loads(line) for line in (src / "documents.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rnl = [d["stem"] for d in documents if d.get("rnl")]
    if rnl and not config.get("keep_rnl"):
        problems.append(f"{len(rnl)} Russian National Library rows, e.g. {rnl[:3]}, although the rows step left them out")
    for name in ("train", "val"):
        rows = pq.read_table(src / "rows" / f"{name}.parquet", columns=list(ROW_TEXT_COLUMNS)).to_pylist()
        other = [r["stem"] for r in rows if r["question"] != SCRIPT_NEUTRAL_PROMPT]
        if other:
            problems.append(f"{len(other)} {name} rows ask another question than the script-neutral prompt, e.g. {other[:3]}")
        hits = [r["stem"] for r in rows if text_hits(" ".join(str(r[c] or "") for c in ROW_TEXT_COLUMNS), ROW_FORBIDDEN)]
        if hits:
            problems.append(f"forbidden attribution in {len(hits)} {name} rows, e.g. {hits[:3]}")
    for name in SIDECARS:
        path = src / name
        if path.is_file():
            hits = text_hits(path.read_text(encoding="utf-8"), FORBIDDEN_CARD)
            if hits:
                problems.append(f"{name} names {hits}")
    return problems


def pack_images(images: Path, stage: Path, limit: int = TAR_LIMIT) -> List[str]:
    """Pack image files into tar archives of at most ``limit`` bytes (``images/<name>`` inside).

    :param images: The rows folder's ``images``.
    :type images: Path
    :param stage: Staging folder.
    :type stage: Path
    :param limit: Largest archive payload in bytes.
    :type limit: int
    :return: Archive names (``images_part1.tar``, ...).
    :rtype: List[str]
    """
    names: List[str] = []
    tar, size = None, 0
    for path in sorted(images.iterdir()):
        n = path.stat().st_size
        if tar is None or size + n > limit:
            if tar is not None:
                tar.close()
            names.append(f"images_part{len(names) + 1}.tar")
            tar, size = tarfile.open(stage / names[-1], "w"), 0
        tar.add(path, arcname=f"images/{path.name}")
        size += n
    if tar is not None:
        tar.close()
    return names


def stage_for_hub(src: Path, stage: Path, limit: int = TAR_LIMIT) -> Dict[str, Any]:
    """Stage the files the Hub push uploads: the two row files, the sidecars and the images as tar archives.

    ``documents.jsonl``, ``build_record.json`` and the gates marker stay local. Nothing goes to the Hub here.

    :param src: The rows folder (:func:`prepush_problems` must be empty).
    :type src: Path
    :param stage: Fresh staging folder.
    :type stage: Path
    :param limit: Largest archive payload in bytes.
    :type limit: int
    :return: Files, archives, archive SHA-256 digests and staged byte count.
    :rtype: Dict[str, Any]
    :raises FileExistsError: When ``stage`` exists.
    :raises ValueError: When the pre-push checks fail.
    """
    if stage.exists():
        raise FileExistsError(f"{stage} exists (an earlier staging): look at it, then remove it by hand")
    release = require_released(src)
    problems = prepush_problems(src)
    if problems:
        raise ValueError("the rows did not pass the pre-push checks: " + "; ".join(problems))
    (stage / "rows").mkdir(parents=True)
    for name in ("train.parquet", "val.parquet"):
        shutil.copy2(src / "rows" / name, stage / "rows" / name)
    files = ["rows/train.parquet", "rows/val.parquet"]
    for name in SIDECARS:
        if (src / name).is_file():
            shutil.copy2(src / name, stage / name)
            files.append(name)
    archives = pack_images(src / "images", stage, limit)
    receipts = release_audit.verify_staged_export(stage, archives, release)
    return {"files": files, "archives": archives, "archive_sha256": receipts,
            "bytes": sum(p.stat().st_size for p in stage.rglob("*") if p.is_file())}


def require_released(src: Path) -> Dict[str, Any]:
    """Refuse a push of training rows that no passing release audit covers (integrity audit 2026-11-22, change 5).

    The build's report folder is the folder of the manifest the rows step read (``build_record.json``); its
    ``RELEASE_AUDIT.json`` must have passed and be bound to exactly this folder's two row files
    (:func:`release_audit.require_release`). There is no override.

    :param src: The rows folder.
    :type src: Path
    :return: The release marker.
    :rtype: Dict[str, Any]
    :raises SystemExit: When the rows have no passing release audit.
    """
    record = read_json(src / "build_record.json")   # noqa: F821
    build_dir = Path(record["manifest_path"]).parent
    return release_audit.require_release(build_dir, [src / "rows" / "train.parquet", src / "rows" / "val.parquet"])


# ----------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    """novel_v1's parser (``plan``, ``rows``, ``notebook``, ``show``) plus ``gates``, ``prepush`` and ``stage``.

    :return: The parser.
    :rtype: argparse.ArgumentParser
    """
    ap = _v1_build_parser()
    sub = next(a for a in ap._actions if isinstance(a, argparse._SubParsersAction))
    p = sub.add_parser("gates", help=f"run the round's gates on built rows (writes {MARKER} on success)")
    p.add_argument("--rows", type=Path, required=True, help="the training rows (rows --out)")
    p.add_argument("--cairo-dir", type=Path, default=None, help="data folder with the benchmark registries")
    p.add_argument("--export", type=Path, default=None, help="the images-once export (default: build_record.json's)")
    p.add_argument("--script-share-tolerance", type=float, default=SCRIPT_SHARE_TOLERANCE)   # noqa: F821
    p.add_argument("--min-long-side", type=int, default=MIN_LONG_SIDE, help="shortest accepted long side (novel_v1: 800)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--gates-path", type=Path, default=V1_GATES)
    p = sub.add_parser("prepush", help="the push's checks on built and gated rows (nothing written, no Hub)")
    p.add_argument("--rows", type=Path, required=True)
    p = sub.add_parser("stage", help="stage the Hub push's files in a fresh folder off the boot disk (no Hub)")
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--stage", type=Path, required=True)
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    """CLI: novel_v1's subcommands on this round, and ``gates`` / ``prepush`` / ``stage``.

    :param argv: Arguments; None = ``sys.argv[1:]``.
    :type argv: Optional[List[str]]
    :raises SystemExit: On a failed check, or an output for rows or images on the boot disk.
    """
    args = build_parser().parse_args(argv)
    if args.command == "rows":
        refuse_boot_disk(args.out, "training rows")
    if args.command == "gates":
        run_gates(args.rows, args.cairo_dir, args.export, args.script_share_tolerance, args.workers, args.min_long_side,
                  gates_path=args.gates_path)
        return
    if args.command == "prepush":
        require_released(args.rows)
        problems = prepush_problems(args.rows)
        if problems:
            raise SystemExit("ABORT: " + "; ".join(problems))
        print(f"pre-push checks OK: {read_json(args.rows / 'mixture.json')['config']['train_rows']} train rows")   # noqa: F821
        return
    if args.command == "stage":
        refuse_boot_disk(args.stage, "Hub staging")
        staged = stage_for_hub(args.rows, args.stage)
        print(f"staged {args.stage}: {staged['files'] + staged['archives']} ({staged['bytes'] / 2 ** 30:.2f} GiB); upload is the user's")
        print(json.dumps({"archive_sha256": staged["archive_sha256"]}))
        return
    _v1_main(argv)


if __name__ == "__main__":
    main()
