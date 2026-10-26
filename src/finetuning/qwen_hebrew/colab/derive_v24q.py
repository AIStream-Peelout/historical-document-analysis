# File name: derive_v24q.py
# Date: 10/18/26
# Author: Isaac Godfried. Coded originally by Claude Sonnet 5.5.
"""Derive the v24q notebooks: the v23q page-parse / question pilot, warm-started from v2.4a step 2000.

v2.3q (:mod:`derive_v23q`) teaches two halves of one skill on ``isaacmg/genizah_v23q_vqa``: parse a page to JSON, then
fill fields and answer questions from the image plus a line-by-line reading in the prompt, citing the line. Its pilot
started from v2.1b step 1200. v24q repeats it **unchanged** from the final checkpoint of the Arabic round, v2.4a step 2000:
same mixture and pinned revision, same gates (size, share floors, page-parse JSON shape, collator guardrail, prepared
dataloader, session-plan lock, exact resume), same 1,000 steps, learning rate 1e-4, 5 % warm-up, cosine, seed 3407, LoRA
r16 with the merger frozen. The notebook is v23q's (:func:`derive_v23q.derive`), then these exact, asserted edits:

* the warm start -> v2.4a step 2000: the adapter at the commit of that checkpoint in ``isaacmg/qwen3-vl-8b-hebrew-v24a-ckpt``
  (weights only: fresh optimizer, fresh cosine), and the model cell checks that the downloaded checkpoint is step 2000.
  This is the mechanism of every run since v2.2b (``last-checkpoint/adapter_model.safetensors`` of a checkpoint
  repository at a commit), not the merged 16-bit repository: the model cell builds a fresh rank-16 adapter on the 4-bit
  base and loads the warm adapter into it, so a merged model could not be loaded there;
* names -> ``v24q``: checkpoint repository, output and resume folders, run and W&B names, merged repository and folder;
* the checkpoint repository is created **private** (v2.4a trained on Muharaf, CC BY-NC-SA, so what continues from it
  stays private until the owner decides), and saves every 50 steps (v23q: 100) so a terminated session costs at most
  50 steps;
* intro and the cell titles.

The data, helper, install and collator cells are v23q's, byte for byte. Then the Lambda edition
(:func:`host_lambda.to_lambda`) with a safety stop sized for one Lambda session; the Colab edition keeps v23q's
``TIME_BUDGET_H``.

Usage::

    .venv/bin/python -m src.finetuning.qwen_hebrew.colab.derive_v24q [--revision <40-hex sha>] \\
        [--host both|colab|lambda] [--src colab/genizah_v22b.ipynb] [--out-dir DIR]
"""
import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

from src.finetuning.qwen_hebrew.colab import derive_v23q as dq
from src.finetuning.qwen_hebrew.colab import derive_v24a as d4
from src.finetuning.qwen_hebrew.colab import host_lambda
from src.finetuning.qwen_hebrew.colab.derive_v22b import TIME_BUDGET_H as COLAB_TIME_BUDGET_H, _replace_line, _replace_once

HERE = Path(__file__).resolve().parent

TAG = "v24q"
RUN_NAME = f"genizah_{TAG}"
NOTEBOOK = f"genizah_{TAG}.ipynb"
CKPT_REPO = f"isaacmg/qwen3-vl-8b-hebrew-{TAG}-ckpt"      # new and private: never an existing run's repository
MERGED_REPO = f"isaacmg/qwen3-vl-8b-hebrew-{TAG}-merged"
DATA_REPO = dq.DATA_REPO                                  # the same mixture as v23q
DATA_REVISION = "a600dacd02365b1d16475529832e8c701d27b304"  # the revision genizah_v23q.ipynb pins

# warm start: the last checkpoint of v2.4a (logs/v24a_eval_chain.log: "stage v24a step 2000 (cafd125e60d0...)"; its
# last-checkpoint/trainer_state.json says global_step 2000 of 2000)
WARM_CKPT_REPO = "isaacmg/qwen3-vl-8b-hebrew-v24a-ckpt"
WARM_REVISION = "cafd125e60d0156b1ebed2af78f5b65df201ce94"
WARM_STEP = 2000

# v23q saved every 100 steps (v22b's); v23a and v24a (Colab and Lambda editions) save every 50, so a termination costs at most 50 steps
SAVE_STEPS = 50

NEW_MODEL_TITLE = f"# Cell 3 — model at page resolution + THE MERGER KNOB + warm start (v2.4a step {WARM_STEP})"
NEW_WARM = (f'# Warm start: v2.4a step {WARM_STEP} — the end of the second Arabic-script run (one pass over its mixture, LoRA r16,\n'
            '# merger frozen). Weights only: this run has its own optimizer and a fresh cosine schedule.\n'
            f'WARM_CKPT_REPO = "{WARM_CKPT_REPO}"   # private: the token must read it\n'
            f'WARM_REVISION = "{WARM_REVISION}"  # commit "Training in progress, step {WARM_STEP}, checkpoint"\n'
            f'WARM_STEP = {WARM_STEP}\n')
NEW_WARM_DOWNLOAD = ('local = snapshot_download(WARM_CKPT_REPO, revision=WARM_REVISION,\n'
                     '                          allow_patterns=["last-checkpoint/adapter_model.safetensors", "last-checkpoint/trainer_state.json"])\n'
                     f'# {TAG}: the pinned commit must hold the checkpoint of the planned step\n'
                     'with open(f"{local}/last-checkpoint/trainer_state.json", encoding="utf-8") as _fh:\n'
                     '    _warm_step = int(json.load(_fh)["global_step"])\n'
                     'assert _warm_step == WARM_STEP, f"warm start {WARM_CKPT_REPO}@{WARM_REVISION[:8]} is step {_warm_step}, expected {WARM_STEP}"\n')
OLD_SAVE = "save_steps=100, save_total_limit=2,"
NEW_SAVE = f"save_steps={SAVE_STEPS}, save_total_limit=2,   # a termination costs at most {SAVE_STEPS} steps (the Hub's last-checkpoint follows every save)"
OLD_PRIVATE = 'hub_strategy="checkpoint", hub_private_repo=False,'
NEW_PRIVATE = ('hub_strategy="checkpoint", hub_private_repo=True,    '
               '# the warm start (v2.4a) trained on Muharaf (CC BY-NC-SA): private until the licence question is decided')


def warmup_steps() -> int:
    """Warm-up steps the Trainer derives from the ratio (``ceil(max_steps * warmup_ratio)``).

    :return: Steps of linear warm-up (v23q's 5 % of its 1,000 steps).
    :rtype: int
    """
    return math.ceil(dq.MAX_STEPS * float(dq.WARMUP_RATIO))


def lambda_budget_h() -> float:
    """Session safety stop on Lambda: :data:`derive_v24a.BUDGET_FACTOR` x one pass at the measured step time, rounded up.

    :return: Hours.
    :rtype: float
    """
    return float(math.ceil(d4.BUDGET_FACTOR * dq.MAX_STEPS * d4.MEASURED_S_PER_STEP / 3600))


def intro_md() -> str:
    """Markdown intro of the v24q notebook.

    :return: Text of cell 0 (the Lambda edition puts its launch notes in front of it).
    :rtype: str
    """
    pass_h = dq.MAX_STEPS * d4.MEASURED_S_PER_STEP / 3600
    return f"""# Genizah v2.4q — page parse and questions answered from the page, on top of v2.4a (pilot re-run)

v2.3q's pilot, unchanged, from a better starting point: **the same mixture** (`{DATA_REPO}`, pinned revision), the same
gates and the same schedule, with the warm start moved from v2.1b step 1200 to **v2.4a step {WARM_STEP}**, the end of the second
Arabic-script run. The mixture teaches two halves of one skill: parse a page to JSON (lines, some with boxes), then fill
fields and answer questions from the image plus a line-by-line reading in the prompt, citing the line.

Same training stack as v2.3q (pinned install triplet, 6.5/7 MP resolution contract, LoRA r16 on tower + language,
merger FROZEN, adamw_8bit, cosine, seed 3407, fixed row order with exact resume, prepared-dataloader gate before step 1,
JSON-shape gates on every page-parse target). What changes:

| | v2.3q | this run |
|---|---|---|
| Warm start | v2.1b step 1200 | **v2.4a step {WARM_STEP}** (`{WARM_CKPT_REPO}@{WARM_REVISION[:8]}`), weights only: fresh optimizer, fresh cosine |
| Data | `{DATA_REPO}` | the same rows at the same pinned revision |
| Rows / steps | {dq.TRAIN_ROWS:,} / {dq.MAX_STEPS:,} | {dq.TRAIN_ROWS:,} / {dq.MAX_STEPS:,} (one pass, batch 1 x accumulation 8) |
| Learning rate | {dq.LR}, {float(dq.WARMUP_RATIO):.0%} warm-up | {dq.LR}, {float(dq.WARMUP_RATIO):.0%} warm-up ({warmup_steps()} steps), fresh cosine |
| Saves | every 100 steps | every {SAVE_STEPS} steps: a terminated session costs at most {SAVE_STEPS} steps |
| Checkpoint repository | public | new, **private**: v2.4a trained on Muharaf (CC BY-NC-SA) |
| Session | Colab, {COLAB_TIME_BUDGET_H} h budget | Lambda A100-40GB: one pass ≈ {pass_h:.1f} h at {d4.MEASURED_S_PER_STEP} s/step, safety stop {lambda_budget_h():.0f} h. The Colab edition keeps v2.3q's {COLAB_TIME_BUDGET_H} h budget |

The model cell checks that the downloaded warm start is the step-{WARM_STEP} checkpoint of v2.4a (its `trainer_state.json`).
The reference for this run is v2.4a itself on the parse harness before any training (`logs/next_round/vqa/baseline_v24a.sh`).

Every target text is a human edition line. The model's own reading appears only inside prompts, as context.

**To resume after a session ends: run every cell again, unedited.** The notebook finds the newest checkpoint on the
Hub, checks the stored schedule and row order, and continues with the rows not yet trained.

Run name `{RUN_NAME}`; checkpoints in `{CKPT_REPO}` (new, private); merged model in `{MERGED_REPO}` (private).
"""


def derive(nb: Dict, revision: str = DATA_REVISION) -> Dict:
    """Apply the v24q edits to a loaded v22b notebook (Colab edition).

    :param nb: Notebook JSON of ``genizah_v22b.ipynb`` (not modified in place).
    :type nb: Dict
    :param revision: 40-hex revision of :data:`DATA_REPO`; the default is the one ``genizah_v23q.ipynb`` pins. The
        ``PIN-AFTER-PUSH`` placeholder is refused: the mixture is already pushed, and a notebook must never fall back to ``main``.
    :type revision: str
    :return: The v24q Colab notebook.
    :rtype: Dict
    :raises ValueError: When ``revision`` is not a 40-hex sha, or an anchor of the v23q notebook is missing or ambiguous.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"dataset revision must be a 40-hex sha, not {revision!r}")
    out = dq.derive(nb, revision)
    cells = out["cells"]
    cells[0]["source"] = intro_md().splitlines(keepends=True)
    # --- model cell: warm start v2.4a step 2000, checked ---------------------------------------------
    c3 = "".join(cells[3]["source"])
    c3 = _replace_once(c3, d4.OLD_MODEL_TITLE, NEW_MODEL_TITLE, "model cell title")
    c3 = _replace_once(c3, d4.OLD_WARM, NEW_WARM, "warm start pin")
    c3 = _replace_once(c3, d4.OLD_WARM_DOWNLOAD, NEW_WARM_DOWNLOAD, "warm start step check")
    cells[3]["source"] = c3.splitlines(keepends=True)
    # --- training cell: names, private checkpoints, 50-step saves ----------------------------------------
    c8 = "".join(cells[8]["source"])
    c8 = _replace_once(c8, "# Cell 6 — v2.3q (page parse + questions) training:",
                       f"# Cell 6 — v2.4q (page parse + questions, warm start v2.4a step {WARM_STEP}) training:", "cell title")
    c8 = _replace_once(c8, f'CKPT_REPO = "{dq.CKPT_REPO}"', f'CKPT_REPO = "{CKPT_REPO}"', "ckpt repo")
    c8 = _replace_once(c8, f'OUT_DIR = "outputs_{dq.TAG}"', f'OUT_DIR = "outputs_{TAG}"', "out dir")
    c8 = _replace_once(c8, f'RESUME_ROOT = "/content/{dq.TAG}_resume"', f'RESUME_ROOT = "/content/{TAG}_resume"', "resume root")
    c8 = _replace_once(c8, OLD_SAVE, NEW_SAVE, "save steps")
    c8 = _replace_once(c8, OLD_PRIVATE, NEW_PRIVATE, "private checkpoints")
    c8 = _replace_once(c8, f'run_name="{dq.RUN_NAME}"', f'run_name="{RUN_NAME}"', "run name")
    c8 = _replace_once(c8, f'wandb.init(project="qwen-hebrew-finetune", name="{dq.RUN_NAME}", ',
                       f'wandb.init(project="qwen-hebrew-finetune", name="{RUN_NAME}", ', "wandb name")
    cells[8]["source"] = c8.splitlines(keepends=True)
    # --- merged-model cell: names --------------------------------------------------------------------
    c9 = "".join(cells[9]["source"])
    c9 = _replace_once(c9, f'MERGED_REPO = "{dq.MERGED_REPO}"', f'MERGED_REPO = "{MERGED_REPO}"', "merged repo")
    if c9.count(f'"{dq.TAG}-merged"') != 2:
        raise ValueError("merged-model folder name not found twice in the last cell")
    c9 = c9.replace(f'"{dq.TAG}-merged"', f'"{TAG}-merged"')
    cells[9]["source"] = c9.splitlines(keepends=True)
    return out


def to_lambda(nb: Dict, notebook_name: str) -> Dict:
    """The Lambda edition with a session safety stop sized for one Lambda session.

    :param nb: :func:`derive` output.
    :type nb: Dict
    :param notebook_name: File name the edition is written as (quoted in the launch notes).
    :type notebook_name: str
    :return: The Lambda edition.
    :rtype: Dict
    """
    out = host_lambda.to_lambda(nb, notebook_name)
    c7 = "".join(out["cells"][7]["source"])
    c7 = _replace_line(c7, r"^TIME_BUDGET_H = .*$",
                       f"TIME_BUDGET_H = {lambda_budget_h()}   # Lambda, one session: {dq.MAX_STEPS:,} steps x ~{d4.MEASURED_S_PER_STEP} s "
                       f"(v2.3a on the A100-SXM4-40GB) = {dq.MAX_STEPS * d4.MEASURED_S_PER_STEP / 3600:.1f} h, x {d4.BUDGET_FACTOR}; a safety stop only, "
                       "max_steps ends the run", "time budget")
    out["cells"][7]["source"] = c7.splitlines(keepends=True)
    return out


def main(argv: Optional[List[str]] = None) -> None:
    """CLI: write the Colab edition, the Lambda edition (+ ``lambda_setup.sh`` beside it when it differs), or both.

    :param argv: Arguments; None = ``sys.argv[1:]``.
    :type argv: Optional[List[str]]
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--revision", default=DATA_REVISION, help=f"40-hex revision of {DATA_REPO} (default: the one genizah_v23q.ipynb pins)")
    ap.add_argument("--src", type=Path, default=HERE / "genizah_v22b.ipynb")
    ap.add_argument("--host", choices=["both", "colab", "lambda"], default="both",
                    help="both (default) = genizah_v24q.ipynb + genizah_v24q_lambda.ipynb (+ lambda_setup.sh)")
    ap.add_argument("--out-dir", type=Path, default=HERE)
    args = ap.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise SystemExit("revision must be a 40-hex sha")
    colab = derive(json.load(open(args.src, encoding="utf-8")), args.revision)
    outputs = []
    if args.host in ("both", "colab"):
        outputs.append((args.out_dir / NOTEBOOK, colab))
    if args.host in ("both", "lambda"):
        nb_name, sh_name = host_lambda.lambda_files(NOTEBOOK)
        outputs.append((args.out_dir / nb_name, to_lambda(colab, nb_name)))
        if d4.write_if_changed(args.out_dir / sh_name, host_lambda.setup_script()):
            print(f"wrote {args.out_dir / sh_name}")
    for path, nb in outputs:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {path} ({len(nb['cells'])} cells; dataset {DATA_REPO}@{args.revision}; run {RUN_NAME}, {dq.TRAIN_ROWS:,} rows = "
              f"{dq.MAX_STEPS:,} steps; warm start {WARM_CKPT_REPO}@{WARM_REVISION[:8]} (step {WARM_STEP}))")


if __name__ == "__main__":
    main()
