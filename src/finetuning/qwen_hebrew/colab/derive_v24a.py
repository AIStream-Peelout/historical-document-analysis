# File name: derive_v24a.py
# Date: 10/15/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Derive the v24a notebooks (second Arabic-script run): Colab edition and Lambda Cloud edition.

v2.4a continues v2.3a with more Geniza Arabic pages: the PGP Arabic editions v2 (built from the pinned PGP export,
with the documents whose edition sides were matched to their images; ``pgp_arabic_editions_v2_images_once``). The
notebook is v2.3a's (:func:`derive_v23a.derive` with the v24a :class:`derive_v23a.Arm`: repositories, folders, run
name, size gate, share floors, ``max_steps``), then these exact, asserted edits:

* dataset repository ``isaacmg/genizah_v24a_arabic``, its revision comment and local folder; the notebook **refuses
  to run** while the revision is the ``PIN-AFTER-PUSH`` placeholder (v2.3a's notebook fell back to ``main``);
* warm start -> **v2.3a step 1500** (``isaacmg/qwen3-vl-8b-hebrew-v23a-ckpt`` at the commit of that checkpoint),
  weights only, and the model cell checks the downloaded checkpoint is step 1500;
* 16,000 rows = one pass = 2,000 steps; LR 1e-4 on a fresh cosine with 3 % warm-up (60 steps); seed 3407; LoRA r16;
* the mixture plan (:func:`planned_shares`): Geniza Arabic editions at ``ARABIC_QUOTA`` rows (default 2,500), the
  outside handwriting sets at exactly v2.3a's share of the total, the Hebrew replay the remainder in v2.3a's
  2,280 : 3,360 ratio; ``logs/next_round/arabic_v24a/build_v24a.sh`` builds from the same function, so the notebook's
  share floors and the build cannot disagree;
* intro, cell titles and comments; the checkpoint repository is new and private (Muharaf is CC BY-NC-SA);

then the Lambda edition (:func:`host_lambda.to_lambda`) with a safety stop sized for one Lambda session.

The build checks of the round live here too, so the build script, the local gates runner and the push share them:
:func:`check_editions_provenance` (the Arabic editions v2 export records the pinned PGP export for its edition text
and its queue) and :func:`check_train_registry` (no train row comes from a document or manuscript a registered
benchmark holds out).

Usage::

    .venv/bin/python -m src.finetuning.qwen_hebrew.colab.derive_v24a --revision <40-hex sha | PIN-AFTER-PUSH> \\
        [--arabic-quota 2500] [--host both|colab|lambda] [--src colab/genizah_v22b.ipynb] [--out-dir DIR]
"""
import argparse
import json
import math
import re
from pathlib import Path
from typing import AbstractSet, Dict, Iterable, List, Mapping, Optional, Tuple

from src.datasets.evaluations.benchmark_registry import registered_benchmark_documents, registered_benchmark_manuscripts
from src.finetuning.qwen_hebrew.colab import derive_v23a as dv
from src.finetuning.qwen_hebrew.colab import host_lambda
from src.finetuning.qwen_hebrew.colab.derive_v22b import _replace_line, _replace_once

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

# the shared spec's PGP export snapshot: the queue and the Arabic editions v2 are built from it (README steps 0, 1, 3)
PINNED_PGP_DIR = REPO / "artifacts" / "pgp_export_b2343a17"
PINNED_COMMIT_PREFIX = "b2343a17"

DATA_REPO = "isaacmg/genizah_v24a_arabic"
LOCAL_LEAF = "genizah_v24a_arabic"
V23A_LOCAL_LEAF = dv.DATA_REPO.split("/")[1]      # the v2.3a notebook's local folder (genizah_v23a_arabic)
PLACEHOLDER = "PIN-AFTER-PUSH"
TRAIN_ROWS = 16000                     # one pass = TRAIN_ROWS // 8 = 2,000 steps
ARABIC_QUOTA = 2500                    # default rows of the Geniza Arabic editions (ARABIC_QUOTA in build_v24a.sh)

# v2.3a's plan: SHARES in logs/next_round/arabic/build_v23a.sh at 12,000 train rows, in its key order (the builder
# concatenates the components in this order before its seeded shuffle)
V23A_TRAIN_ROWS = 12000
V23A_ROWS: Dict[str, int] = {"arabic_editions": 1320, "arabic_agapet": 960, "arabic_muharaf": 2640, "arabic_baybars": 1080,
                             "arabic_iskandar": 360, "pgp_edition_pages": 2280, "ktiv_transcription": 3360}
OUTSIDE = ("arabic_agapet", "arabic_muharaf", "arabic_baybars", "arabic_iskandar")
REPLAY = ("pgp_edition_pages", "ktiv_transcription")
FLOOR_FRACTION = 0.8                   # share floors of the data cell at about 0.8 of the planned shares, as in v2.3a

# warm start: the last checkpoint of v2.3a (logs/v23a_eval_chain.log: "stage v23a step 1500 (b5503b18...)")
WARM_CKPT_REPO = dv.MAIN.ckpt_repo
WARM_REVISION = "b5503b186a61e640b95277c3886fd65e503d97d1"
WARM_STEP = 1500

LR = "1e-4"
WARMUP_RATIO = "0.03"                  # 3 % of 2,000 steps = 60 warm-up steps (Trainer: ceil(max_steps * ratio))
SEED = 3407

# v2.3a on the Lambda A100-SXM4-40GB (W&B run e0mgttev): steps 810 -> 1500 in 12 h 11 min, evals and 50-step saves
# included = 63.6 s a step. The session budget is a safety stop at 1.3x one full pass (a stop saves, pushes, and the
# next launch resumes exactly).
MEASURED_S_PER_STEP = 63.6
BUDGET_FACTOR = 1.3

OLD_UNPINNED = ('if V22_REVISION == "PIN-AFTER-PUSH":\n'
                '    V22_REVISION = HfApi().dataset_info(V22_REPO).sha\n'
                '    print(f"⚠️ unpinned: {V22_REPO} main -> {V22_REVISION} — pin this SHA before the next launch")\n'
                'assert len(V22_REVISION) == 40, f"unpinned revision: {V22_REVISION!r}"\n')
PIN_GUARD = ('# v24a: never train on an unpinned dataset (push_v24a.sh pins the pushed revision and regenerates this notebook)\n'
             'if V22_REVISION == "PIN-AFTER-PUSH" or len(V22_REVISION) != 40 or any(_c not in "0123456789abcdef" for _c in V22_REVISION):\n'
             '    raise RuntimeError(f"dataset revision {V22_REVISION!r} is not a pinned 40-hex sha: run logs/next_round/arabic_v24a/push_v24a.sh, "\n'
             '                       "which pins the pushed revision and regenerates this notebook")\n')
OLD_DATA_TITLE = "# Cell 2 — data: genizah_v22_pilot (images-once, MAP-STYLE, materialised mixture — no streaming)"
OLD_MODEL_TITLE = "# Cell 3 — model at page resolution + THE MERGER KNOB + flagship warm start (v2.1b step 1200)"
OLD_WARM = ('# Warm start: v2.1b step 1200 — the flagship (religious CER 0.167, PGP 0.180, boxes on the right line 80%;\n'
            '# the 1300–1700 checkpoints were never promoted). The v2.2 data lever sits on top of it.\n'
            'WARM_CKPT_REPO = "isaacmg/qwen3-vl-8b-hebrew-v21b-ckpt"\n'
            'WARM_REVISION = "6724c32cd0f6296223978ae69635fc782d2dbb03"  # commit "Training in progress, step 1200, checkpoint"\n')
NEW_WARM = (f'# Warm start: v2.3a step {WARM_STEP} — the end of the first Arabic-script run (one pass over its mixture, LoRA r16,\n'
            '# merger frozen). Weights only: this run has its own optimizer and a fresh cosine schedule.\n'
            f'WARM_CKPT_REPO = "{WARM_CKPT_REPO}"   # private: the token must read it\n'
            f'WARM_REVISION = "{WARM_REVISION}"  # commit "Training in progress, step {WARM_STEP}, checkpoint"\n'
            f'WARM_STEP = {WARM_STEP}\n')
OLD_WARM_DOWNLOAD = ('local = snapshot_download(WARM_CKPT_REPO, revision=WARM_REVISION,\n'
                     '                          allow_patterns="last-checkpoint/adapter_model.safetensors")\n')
NEW_WARM_DOWNLOAD = ('local = snapshot_download(WARM_CKPT_REPO, revision=WARM_REVISION,\n'
                     '                          allow_patterns=["last-checkpoint/adapter_model.safetensors", "last-checkpoint/trainer_state.json"])\n'
                     '# v24a: the pinned commit must hold the checkpoint of the planned step\n'
                     'with open(f"{local}/last-checkpoint/trainer_state.json", encoding="utf-8") as _fh:\n'
                     '    _warm_step = int(json.load(_fh)["global_step"])\n'
                     'assert _warm_step == WARM_STEP, f"warm start {WARM_CKPT_REPO}@{WARM_REVISION[:8]} is step {_warm_step}, expected {WARM_STEP}"\n')
OLD_TRAIN_TITLE = "# Cell 6 — v2.3a (Arabic script) training:"
LR_LINE = r"^\s*learning_rate=[0-9.eE+-]+,"   # the TrainingArguments line with the rate literal (not the resume gate's _a.learning_rate)
OLD_WARMUP = f'warmup_ratio={dv.WARMUP_RATIO}, lr_scheduler_type="cosine", weight_decay=0.01,'
SEED_LINE = f'optim="adamw_8bit", seed={SEED}, data_seed={SEED}, output_dir=OUT_DIR,'


def planned_shares(arabic_quota: int = ARABIC_QUOTA, train_rows: int = TRAIN_ROWS) -> Dict[str, float]:
    """The v24a mixture plan as the builder's ``--shares``.

    The Geniza Arabic editions take ``arabic_quota`` rows; each outside handwriting set takes exactly its v2.3a share
    of the total; the Hebrew replay (full documentary pages, KTIV transcription) takes the remainder in v2.3a's
    2,280 : 3,360 ratio. Keys follow v2.3a's order.

    :param arabic_quota: Train rows of ``arabic_editions``.
    :type arabic_quota: int
    :param train_rows: Train budget.
    :type train_rows: int
    :return: component -> share (unrounded floats, summing to 1).
    :rtype: Dict[str, float]
    :raises ValueError: When the quota leaves no replay rows, or an outside set's rows would not be whole.
    """
    outside_rows = {c: V23A_ROWS[c] * train_rows / V23A_TRAIN_ROWS for c in OUTSIDE}
    if any(r != int(r) for r in outside_rows.values()):
        raise ValueError(f"{train_rows} train rows give the outside sets fractional rows: {outside_rows}")
    replay_rows = train_rows - arabic_quota - sum(outside_rows.values())
    if arabic_quota <= 0 or replay_rows <= 0:
        raise ValueError(f"arabic_quota {arabic_quota} must be positive and leave replay rows ({replay_rows:.0f} left of {train_rows})")
    replay_total = sum(V23A_ROWS[c] for c in REPLAY)
    shares = {}
    for comp in V23A_ROWS:
        if comp == "arabic_editions":
            shares[comp] = arabic_quota / train_rows
        elif comp in OUTSIDE:
            shares[comp] = V23A_ROWS[comp] / V23A_TRAIN_ROWS
        else:
            shares[comp] = replay_rows / train_rows * V23A_ROWS[comp] / replay_total
    return shares


def planned_quotas(arabic_quota: int = ARABIC_QUOTA, train_rows: int = TRAIN_ROWS) -> Dict[str, int]:
    """Rows per component of the plan, rounded as the mixture builder rounds (``round(share * train_rows)``).

    :param arabic_quota: Train rows of ``arabic_editions``.
    :type arabic_quota: int
    :param train_rows: Train budget.
    :type train_rows: int
    :return: component -> rows.
    :rtype: Dict[str, int]
    :raises ValueError: When the rounded rows do not add up to ``train_rows`` or miss the quota.
    """
    quotas = {comp: int(round(share * train_rows)) for comp, share in planned_shares(arabic_quota, train_rows).items()}
    if sum(quotas.values()) != train_rows or quotas["arabic_editions"] != arabic_quota:
        raise ValueError(f"the plan rounds to {quotas} (sum {sum(quotas.values())}), not {train_rows} rows with {arabic_quota} Arabic editions")
    return quotas


def share_floors(shares: Mapping[str, float]) -> str:
    """Source-share gate of the data cell: about :data:`FLOOR_FRACTION` of each planned share, rounded down.

    :param shares: component -> planned share.
    :type shares: Mapping[str, float]
    :return: The tuple literal the data cell's loop reads.
    :rtype: str
    """
    floors = [(comp, math.floor(FLOOR_FRACTION * share * 100 + 1e-9) / 100) for comp, share in shares.items()]
    return "(" + ", ".join(f'("{comp}", {floor:.2f})' for comp, floor in floors) + ")"


def arabic_quota_of(mixture: Mapping) -> int:
    """The Arabic quota a built mixture was planned with, checked against :func:`planned_shares`.

    :param mixture: ``mixture.json`` of a built mixture.
    :type mixture: Mapping
    :return: Its ``arabic_editions`` rows.
    :rtype: int
    :raises ValueError: When the build's train rows or shares are not the v24a plan for that quota.
    """
    config = mixture["config"]
    train_rows, shares = int(config["train_rows"]), dict(config["shares"])
    if train_rows != TRAIN_ROWS:
        raise ValueError(f"built with {train_rows} train rows, the v24a plan has {TRAIN_ROWS}")
    quota = int(round(shares["arabic_editions"] * train_rows))
    plan = planned_shares(quota, train_rows)
    if list(shares) != list(plan) or any(abs(shares[c] - plan[c]) > 1e-12 for c in plan):
        raise ValueError(f"the build's shares {shares} are not the v24a plan for {quota} Arabic rows: {plan}")
    return quota


# ----------------------------------------------------------------------------- build checks


#: Geniza sources of the mixture and how a stem names its document: (id kind, stem pattern with the id as group 1).
#: The outside handwriting sets (:data:`OUTSIDE`) hold no Geniza document.
GENIZA_STEMS: Dict[str, Tuple[str, "re.Pattern[str]"]] = {
    "arabic_editions": ("pgpid", re.compile(r"^pgpar_(\d+)_")),
    "pgp_edition_pages": ("pgpid", re.compile(r"^pgp_(\d+)_")),
    "ktiv_transcription": ("sys_num", re.compile(r"^ktiv_(\d+)_")),
}


def pinned_pgp_commit(pgp_dir: Optional[Path] = None, prefix: str = PINNED_COMMIT_PREFIX) -> str:
    """The commit of the pinned PGP export snapshot (its ``COMMIT`` file).

    :param pgp_dir: Export directory; None = :data:`PINNED_PGP_DIR`.
    :type pgp_dir: Optional[Path]
    :param prefix: Start of the commit the snapshot must be (the spec's ``b2343a17``).
    :type prefix: str
    :return: The 40-hex commit.
    :rtype: str
    :raises ValueError: When the ``COMMIT`` file does not hold a 40-hex commit starting with ``prefix``.
    """
    pgp_dir = pgp_dir if pgp_dir is not None else PINNED_PGP_DIR
    commit = (pgp_dir / "COMMIT").read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not commit.startswith(prefix):
        raise ValueError(f"{pgp_dir / 'COMMIT'} holds {commit[:48]!r}, not the pinned PGP export {prefix}...")
    return commit


def editions_provenance_problems(stats: Mapping, commit: str) -> List[str]:
    """Why an Arabic editions build (its ``stats.json``) is not from the pinned PGP export.

    The editions builder records the export of its edition text (``inputs.pgp_export_commit``, the export's ``COMMIT``)
    and the export its Arabic queue was built from (``inputs.queue_pgp_export_commit``). Both must be the pinned
    snapshot: a build from the unpinned ``raw_data`` export and the old queue records neither, and passes the builder's
    own queue check because the two empty values agree.

    :param stats: ``stats.json`` of the editions build (``images_once`` copies it into the export).
    :type stats: Mapping
    :param commit: The pinned export's commit.
    :type commit: str
    :return: The problems; empty when both inputs record ``commit``.
    :rtype: List[str]
    """
    inputs = stats.get("inputs") or {}
    problems = []
    for key, what in (("pgp_export_commit", "edition text"), ("queue_pgp_export_commit", "Arabic queue")):
        got = inputs.get(key) or ""
        if got != commit:
            problems.append(f"{what} from PGP export {got or '(unpinned or unrecorded)'}, not {commit}")
    return problems


def check_editions_provenance(editions_dir: Path, pgp_dir: Optional[Path] = None) -> str:
    """Fail unless an Arabic editions build, or its images-once export, comes from the pinned PGP export.

    :param editions_dir: The build or its images-once export (either holds the builder's ``stats.json``).
    :type editions_dir: Path
    :param pgp_dir: Pinned export directory; None = :data:`PINNED_PGP_DIR`.
    :type pgp_dir: Optional[Path]
    :return: The pinned commit both inputs record.
    :rtype: str
    :raises ValueError: When the edition text or the queue was built from another export (or an unpinned one).
    """
    pgp_dir = pgp_dir if pgp_dir is not None else PINNED_PGP_DIR
    commit = pinned_pgp_commit(pgp_dir)
    stats = json.loads((editions_dir / "stats.json").read_text(encoding="utf-8"))
    problems = editions_provenance_problems(stats, commit)
    if problems:
        raise ValueError(f"{editions_dir} is not built from the pinned PGP export {pgp_dir}: {'; '.join(problems)}. Build the queue "
                         f"with arabic_scrape_queue --pgp-dir {pgp_dir} --out-dir DIR and pass --pgp-dir {pgp_dir} --queue-csv "
                         f"DIR/arabic_scrape_priority_queue.csv to build_pgp_arabic_editions (logs/next_round/arabic_v24a/README.md)")
    return commit


def registry_contamination(rows: Iterable[Mapping[str, str]], held_pgpids: AbstractSet[str],
                           held_sys_nums: AbstractSet[str]) -> Dict[str, Dict[str, object]]:
    """Documents of registered benchmarks among the rows of a mixture.

    Geniza rows name their document in the stem (:data:`GENIZA_STEMS`); rows of the outside handwriting sets are not
    checked.

    :param rows: Rows with ``stem`` and ``source``.
    :type rows: Iterable[Mapping[str, str]]
    :param held_pgpids: PGP documents the registered benchmarks hold out.
    :type held_pgpids: AbstractSet[str]
    :param held_sys_nums: KTIV manuscripts (NLI system numbers) the registered benchmarks hold out.
    :type held_sys_nums: AbstractSet[str]
    :return: Geniza source -> ``{"documents": distinct documents in the rows, "held_out": sorted held-out ids among them}``.
    :rtype: Dict[str, Dict[str, object]]
    :raises ValueError: When a row's source is not a planned component, or a Geniza stem does not match its pattern
        (the check would be blind to that row).
    """
    seen: Dict[str, set] = {source: set() for source in GENIZA_STEMS}
    for row in rows:
        source, stem = row["source"], row["stem"]
        if source in OUTSIDE:
            continue
        if source not in GENIZA_STEMS:
            raise ValueError(f"row {stem!r} has source {source!r}, not a planned component")
        m = GENIZA_STEMS[source][1].match(stem)
        if m is None:
            raise ValueError(f"{source} stem {stem!r} does not match {GENIZA_STEMS[source][1].pattern}")
        seen[source].add(m.group(1))
    held = {"pgpid": held_pgpids, "sys_num": held_sys_nums}
    return {source: {"documents": len(ids), "held_out": sorted(ids & set(held[GENIZA_STEMS[source][0]]), key=int)}
            for source, ids in seen.items()}


def check_train_registry(rows: Iterable[Mapping[str, str]], held_pgpids: Optional[AbstractSet[str]] = None,
                         held_sys_nums: Optional[AbstractSet[str]] = None) -> Dict[str, int]:
    """Fail when a train row comes from a document or manuscript a registered benchmark holds out.

    The source exports of the mixture were built at different dates, some before a benchmark was registered
    (``sealed_v1``, 2026-10-12, is newer than the Hebrew edition pages and KTIV v4 exports), so the built train rows are
    checked against the registries as they are now. Val rows are not checked: the edition-page val split holds
    registered documents on purpose.

    :param rows: Train rows with ``stem`` and ``source``.
    :type rows: Iterable[Mapping[str, str]]
    :param held_pgpids: Held-out PGP documents; None = every registry of ``benchmark_registry``.
    :type held_pgpids: Optional[AbstractSet[str]]
    :param held_sys_nums: Held-out KTIV manuscripts; None = every registry of ``benchmark_registry``.
    :type held_sys_nums: Optional[AbstractSet[str]]
    :return: Geniza source -> documents checked.
    :rtype: Dict[str, int]
    :raises ValueError: When a train row is a held-out document (or a row cannot be checked, :func:`registry_contamination`).
    """
    held_pgpids = held_pgpids if held_pgpids is not None else registered_benchmark_documents()[1]
    held_sys_nums = held_sys_nums if held_sys_nums is not None else registered_benchmark_manuscripts()
    found = registry_contamination(rows, held_pgpids, held_sys_nums)
    hits = {source: info["held_out"] for source, info in found.items() if info["held_out"]}
    if hits:
        raise ValueError("train rows of registered benchmark documents: "
                         + "; ".join(f"{source} {len(ids)}, e.g. {ids[:20]}" for source, ids in hits.items()))
    return {source: int(info["documents"]) for source, info in found.items()}


def make_arm(arabic_quota: int = ARABIC_QUOTA) -> dv.Arm:
    """The v24a run as a :class:`derive_v23a.Arm`.

    :param arabic_quota: Train rows of ``arabic_editions`` (sets the share floors).
    :type arabic_quota: int
    :return: Tag ``v24a``: checkpoint repo ``isaacmg/qwen3-vl-8b-hebrew-v24a-ckpt``, run ``genizah_v24a``.
    :rtype: dv.Arm
    """
    planned_quotas(arabic_quota, TRAIN_ROWS)
    return dv.Arm("v24a", "genizah_v24a", "train.parquet", TRAIN_ROWS, share_floors(planned_shares(arabic_quota, TRAIN_ROWS)),
                  "genizah_v24a.ipynb")


ARM = make_arm(ARABIC_QUOTA)


def warmup_steps(arm: dv.Arm = ARM) -> int:
    """Warm-up steps the Trainer derives from the ratio (``ceil(max_steps * warmup_ratio)``).

    :param arm: The run.
    :type arm: dv.Arm
    :return: Steps of linear warm-up.
    :rtype: int
    """
    return math.ceil(arm.max_steps * float(WARMUP_RATIO))


def lambda_budget_h(arm: dv.Arm = ARM) -> float:
    """Session safety stop on Lambda: :data:`BUDGET_FACTOR` x one pass at the measured step time, rounded up.

    :param arm: The run.
    :type arm: dv.Arm
    :return: Hours.
    :rtype: float
    """
    return float(math.ceil(BUDGET_FACTOR * arm.max_steps * MEASURED_S_PER_STEP / 3600))


def intro_md(arabic_quota: int = ARABIC_QUOTA) -> str:
    """Markdown intro of the v24a notebook.

    :param arabic_quota: Train rows of ``arabic_editions``.
    :type arabic_quota: int
    :return: Text of cell 0 (the Lambda edition puts its launch notes in front of it).
    :rtype: str
    """
    arm, shares = make_arm(arabic_quota), planned_shares(arabic_quota, TRAIN_ROWS)
    quotas = planned_quotas(arabic_quota, TRAIN_ROWS)
    outside, replay = sum(shares[c] for c in OUTSIDE), sum(shares[c] for c in REPLAY)
    v23a_outside = sum(V23A_ROWS[c] for c in OUTSIDE) / V23A_TRAIN_ROWS
    v23a_replay = sum(V23A_ROWS[c] for c in REPLAY) / V23A_TRAIN_ROWS
    pass_h = arm.max_steps * MEASURED_S_PER_STEP / 3600
    return f"""# Genizah v2.4a — Arabic script, second run: PGP Arabic editions v2 on top of v2.3a

Same training stack as v2.3a (pinned install triplet, 6.5/7 MP resolution contract, LoRA r16 on tower + language,
merger FROZEN, adamw_8bit, cosine, seed {SEED}, fixed row order with exact resume, {dv.SAVE_STEPS}-step saves, prepared-dataloader
gate before step 1, Arabic-script and Hebrew-replay data gates). What changes:

| | v2.3a | this run |
|---|---|---|
| Warm start | v2.1b step 1200 | **v2.3a step {WARM_STEP}** (`{WARM_CKPT_REPO}@{WARM_REVISION[:8]}`), weights only: fresh optimizer, fresh cosine |
| Data | `{dv.DATA_REPO}` (Geniza Arabic editions v1) | `{DATA_REPO}`: Geniza Arabic editions **v2** (pinned PGP export, side-matched pages included) |
| Rows / steps | one pass over its mixture | **{arm.train_rows:,} / {arm.max_steps:,}** (one pass, batch 1 x accumulation 8) |
| Geniza Arabic editions | {V23A_ROWS["arabic_editions"] / V23A_TRAIN_ROWS:.0%} of the rows | {quotas["arabic_editions"]:,} rows ({shares["arabic_editions"]:.1%}) |
| Outside sets (Agapet, Muharaf, BAYBARS, ISKANDAR) | {v23a_outside:.0%} | {outside:.0%}: v2.3a's share of the total ({", ".join(f"{quotas[c]:,}" for c in OUTSIDE)} rows) |
| Hebrew replay (documentary pages, KTIV) | {v23a_replay:.0%} | the remainder, {replay:.1%} ({quotas["pgp_edition_pages"]:,} + {quotas["ktiv_transcription"]:,} rows, v2.3a's ratio) |
| Learning rate | {dv.LR}, {float(dv.WARMUP_RATIO):.0%} warm-up | {LR}, {float(WARMUP_RATIO):.0%} warm-up ({warmup_steps(arm)} steps), fresh cosine |
| Session | — | Lambda A100-40GB: one pass ≈ {pass_h:.1f} h at {MEASURED_S_PER_STEP} s/step; safety stop {lambda_budget_h(arm):.0f} h. Colab: {dv.TIME_BUDGET_H} h a session |

The data cell **refuses to run** until the dataset revision is pinned (`logs/next_round/arabic_v24a/push_v24a.sh` pins
it and regenerates this notebook); the model cell checks that the warm start is v2.3a's step-{WARM_STEP} checkpoint.

**To resume after a session ends: run every cell again, unedited.** The notebook finds the newest checkpoint on the
Hub, checks the stored schedule and row order, and continues with the rows not yet trained.

Run name `{arm.run_name}`; checkpoints in `{arm.ckpt_repo}` (new, private: Muharaf is CC BY-NC-SA). Checkpoints are scored
on the Arabic-script benchmark (`docs/arabic_script_benchmark.md`) and on PGP-131 / religious-140 for Hebrew-script regression.
"""


def derive(nb: Dict, revision: str, arabic_quota: int = ARABIC_QUOTA) -> Dict:
    """Apply the v24a edits to a loaded v22b notebook (Colab edition).

    :param nb: Notebook JSON of ``genizah_v22b.ipynb`` (not modified in place).
    :type nb: Dict
    :param revision: 40-hex revision of :data:`DATA_REPO` (or ``PIN-AFTER-PUSH``: the notebook then refuses to run).
    :type revision: str
    :param arabic_quota: Train rows of ``arabic_editions`` the dataset was built with.
    :type arabic_quota: int
    :return: The v24a Colab notebook (learning rate :data:`LR`, warm-up :data:`WARMUP_RATIO`, seed :data:`SEED`).
    :rtype: Dict
    :raises ValueError: When an anchor of the v2.3a notebook is missing or ambiguous, or :data:`LR` is not a number.
    """
    arm = make_arm(arabic_quota)
    out = dv.derive(nb, revision, arm)
    cells = out["cells"]
    cells[0]["source"] = intro_md(arabic_quota).splitlines(keepends=True)
    # --- data cell: repository, pinned revision only, local folder ----------------------------------
    c2 = "".join(cells[2]["source"])
    c2 = _replace_once(c2, OLD_DATA_TITLE, f"# Cell 2 — data: {LOCAL_LEAF} (images-once, MAP-STYLE, materialised mixture — no streaming)", "data cell title")
    c2 = _replace_once(c2, f'V22_REPO = "{dv.DATA_REPO}"', f'V22_REPO = "{DATA_REPO}"', "repo")
    c2 = _replace_line(c2, r'^V22_REVISION = "',
                       f'V22_REVISION = "{revision}"   # v24a Arabic mixture (PGP Arabic editions v2): rows + images_part*.tar '
                       f'({arm.train_rows:,} train rows, {arabic_quota:,} Geniza Arabic)', "revision")
    c2 = _replace_once(c2, OLD_UNPINNED, PIN_GUARD, "placeholder guard")
    c2 = _replace_once(c2, f'LOCAL = "/content/{V23A_LOCAL_LEAF}"', f'LOCAL = "/content/{LOCAL_LEAF}"', "local dir")
    cells[2]["source"] = c2.splitlines(keepends=True)
    # --- model cell: warm start v2.3a step 1500, checked --------------------------------------------
    c3 = "".join(cells[3]["source"])
    c3 = _replace_once(c3, OLD_MODEL_TITLE, f"# Cell 3 — model at page resolution + THE MERGER KNOB + warm start (v2.3a step {WARM_STEP})", "model cell title")
    c3 = _replace_once(c3, OLD_WARM, NEW_WARM, "warm start pin")
    c3 = _replace_once(c3, OLD_WARM_DOWNLOAD, NEW_WARM_DOWNLOAD, "warm start step check")
    cells[3]["source"] = c3.splitlines(keepends=True)
    # --- training cell: title, schedule ----------------------------------------------------------------
    c8 = "".join(cells[8]["source"])
    c8 = _replace_once(c8, OLD_TRAIN_TITLE, f"# Cell 6 — v2.4a (Arabic script, editions v2, warm start v2.3a step {WARM_STEP}) training:", "cell title")
    float(LR)                                                    # a number literal, or fail here rather than in the notebook
    c8 = _replace_line(c8, LR_LINE, f"        learning_rate={LR},                "
                       f"# v2.3a's rate on a fresh cosine: more Geniza Arabic pages on top of v2.3a step {WARM_STEP}", "lr")
    c8 = _replace_once(c8, OLD_WARMUP, f'warmup_ratio={WARMUP_RATIO}, lr_scheduler_type="cosine", weight_decay=0.01,   '
                       f'# {warmup_steps(arm)} warm-up steps of {arm.max_steps:,}; the resume gate knows cosine only', "warm-up")
    if c8.count(SEED_LINE) != 1:
        raise ValueError(f"seed line not found once in the training cell: {SEED_LINE!r}")
    cells[8]["source"] = c8.splitlines(keepends=True)
    return out


def to_lambda(nb: Dict, notebook_name: str, arabic_quota: int = ARABIC_QUOTA) -> Dict:
    """The Lambda edition with a session safety stop sized for one Lambda session.

    :param nb: :func:`derive` output.
    :type nb: Dict
    :param notebook_name: File name the edition is written as (quoted in the launch notes).
    :type notebook_name: str
    :param arabic_quota: Train rows of ``arabic_editions`` (only names the arm; the budget does not depend on it).
    :type arabic_quota: int
    :return: The Lambda edition.
    :rtype: Dict
    """
    arm = make_arm(arabic_quota)
    out = host_lambda.to_lambda(nb, notebook_name)
    c7 = "".join(out["cells"][7]["source"])
    c7 = _replace_line(c7, r"^TIME_BUDGET_H = .*$",
                       f"TIME_BUDGET_H = {lambda_budget_h(arm)}   # Lambda, one session: {arm.max_steps:,} steps x ~{MEASURED_S_PER_STEP} s "
                       f"(v2.3a on the A100-SXM4-40GB) = {arm.max_steps * MEASURED_S_PER_STEP / 3600:.1f} h, x {BUDGET_FACTOR}; a safety stop only, "
                       "max_steps ends the run", "time budget")
    out["cells"][7]["source"] = c7.splitlines(keepends=True)
    return out


def write_if_changed(path: Path, text: str) -> bool:
    """Write ``text`` unless the file already holds exactly it.

    :param path: Target file.
    :type path: Path
    :param text: Content.
    :type text: str
    :return: Whether the file was written.
    :rtype: bool
    """
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def main(argv: Optional[List[str]] = None) -> None:
    """CLI: write the Colab edition, the Lambda edition (+ ``lambda_setup.sh`` beside it), or both.

    :param argv: Arguments; None = ``sys.argv[1:]``.
    :type argv: Optional[List[str]]
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--revision", required=True, help=f"40-hex revision of {DATA_REPO}, or {PLACEHOLDER}")
    ap.add_argument("--arabic-quota", type=int, default=ARABIC_QUOTA,
                    help="train rows of arabic_editions the dataset was built with (ARABIC_QUOTA of build_v24a.sh)")
    ap.add_argument("--src", type=Path, default=HERE / "genizah_v22b.ipynb")
    ap.add_argument("--host", choices=["both", "colab", "lambda"], default="both",
                    help="both (default) = genizah_v24a.ipynb + genizah_v24a_lambda.ipynb (+ lambda_setup.sh)")
    ap.add_argument("--out-dir", type=Path, default=HERE)
    args = ap.parse_args(argv)
    if args.revision != PLACEHOLDER and not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise SystemExit(f"revision must be a 40-hex sha or {PLACEHOLDER}")
    arm = make_arm(args.arabic_quota)
    colab = derive(json.load(open(args.src, encoding="utf-8")), args.revision, args.arabic_quota)
    outputs = []
    if args.host in ("both", "colab"):
        outputs.append((args.out_dir / arm.notebook, colab))
    if args.host in ("both", "lambda"):
        nb_name, sh_name = host_lambda.lambda_files(arm.notebook)
        outputs.append((args.out_dir / nb_name, to_lambda(colab, nb_name, args.arabic_quota)))
        if write_if_changed(args.out_dir / sh_name, host_lambda.setup_script()):
            print(f"wrote {args.out_dir / sh_name}")
    for path, nb in outputs:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {path} ({len(nb['cells'])} cells; dataset {DATA_REPO}@{args.revision}; run {arm.run_name}, "
              f"{arm.train_rows:,} rows = {arm.max_steps:,} steps, Arabic editions {args.arabic_quota:,}; "
              f"warm start {WARM_CKPT_REPO}@{WARM_REVISION[:8]} (step {WARM_STEP})")
    if args.revision == PLACEHOLDER:
        print("NOTE: the notebooks refuse to run until regenerated with the pushed revision (push_v24a.sh does it)")


if __name__ == "__main__":
    main()
