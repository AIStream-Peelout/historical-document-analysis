# File name: derive_v23a.py
# Date: 10/5/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Derive the v23a Colab notebook (first Arabic-script run) from the v22b notebook.

The v22b notebook is the source of truth for the training stack (pinned install, 6.5/7 MP
resolution contract, LoRA r16 on tower + language, merger frozen, warm start v2.1b step 1200,
fixed row order with exact resume, time-budget stop). This script applies the v23a deltas as
exact, asserted string edits so nothing else drifts:

* dataset -> the Arabic mixture (``isaacmg/genizah_v23a_arabic`` at a pinned revision): Arabic-script
  pages of the Geniza and of outside handwriting sets, with Hebrew-script replay; the size and
  share gates are those of this mixture;
* the grounding and question-answering gates of the data cell are replaced by Arabic gates (an
  Arabic source's answer is mainly Arabic script and long enough, its prompt names Arabic script,
  outside targets carry no vowel marks or tatweel, the replay rows are still Hebrew script): the
  mixture has no box or QA rows;
* the collator guardrail checks a Hebrew page and an Arabic page instead of a page and a QA row;
* checkpoint repo / output dir / resume dir / run name -> ``v23a``; merged-model repo -> ``v23a``;
* ``max_steps`` for one pass over the mixture, and a higher learning rate with a longer warm-up: a
  new script has to be learned, where v22b only adjusted a converged model;
* the checkpoint repository is created private (the Hebrew runs publish theirs): part of the data is
  CC BY-NC-SA (Muharaf), and whether weights trained on it may be published is the owner's decision.

Three notebooks come out of it (:class:`Arm`):

* ``run``: the run itself;
* ``control``: the same run on ``rows/train_without_outside.parquet`` of the same dataset revision,
  i.e. the same rows in the same order minus the outside handwriting sets
  (``build_v22_mixture.write_train_variant``). Run and control answer whether the outside sets help
  on Geniza pages;
* ``r64``: the same run with a LoRA adapter of rank 64 instead of 16. The rank-16 warm start is
  placed into it without changing the model (:mod:`src.finetuning.qwen_hebrew.lora_rank`, whose
  function is copied into the model cell), so the run starts from the same model with four times
  the adapter capacity. Run and r64 answer whether the adapter's size limits a new script.

Usage::

    .venv/bin/python -m src.finetuning.qwen_hebrew.colab.derive_v23a --revision <40-hex sha> [--arm run|control|r64|all] \\
        [--src colab/genizah_v22b.ipynb] [--out FILE]
"""
import argparse
import copy
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.finetuning.qwen_hebrew import lora_rank
from src.finetuning.qwen_hebrew.colab.derive_v22b import _replace_line, _replace_once

HERE = Path(__file__).resolve().parent

DATA_REPO = "isaacmg/genizah_v23a_arabic"
LR = "1e-4"
WARMUP_RATIO = "0.05"
# Colab terminated the first v23a session at step 445 with the Hub at step 400 (2026-10-05); at ~65 s a step,
# saving every 50 steps caps what a termination costs at ~55 min. Evals stay at 100 (not in the session plan).
SAVE_STEPS = 50
# Both v23a sessions were killed 7 h 55 min after wandb.init (2026-10-05 20:18 and 2026-10-06 07:00 EDT, GPU busy,
# no resource pressure): an ~8 h runtime lifetime, not the documented 24 h. The session ends itself at 7.5 h with a
# clean save + push, so a kill costs nothing; raise it only after a session has provably run longer.
TIME_BUDGET_H = 7.5
MIN_VAL_ROWS = 100
# floors at about 0.8 of the planned shares (SHARES in logs/next_round/arabic/build_v23a.sh)
SHARE_FLOORS = ('(("arabic_editions", 0.09), ("arabic_agapet", 0.06), ("arabic_muharaf", 0.18), ("arabic_baybars", 0.07), '
                '("arabic_iskandar", 0.02), ("pgp_edition_pages", 0.15), ("ktiv_transcription", 0.22))')
# the control keeps the three other components at their row counts: shares 0.11, 0.19 and 0.28 of 12,000 over 6,960 rows
OUTSIDE_COMPONENTS = ("arabic_agapet", "arabic_muharaf", "arabic_baybars", "arabic_iskandar")
CONTROL_TRAIN_FILE = "train_without_outside.parquet"
CONTROL_SHARE_FLOORS = '(("arabic_editions", 0.15), ("pgp_edition_pages", 0.27), ("ktiv_transcription", 0.40))'


@dataclass(frozen=True)
class Arm:
    """One of the two v23a notebooks.

    :param tag: Run tag: names the checkpoint and merged repositories and the output and resume folders.
    :type tag: str
    :param run_name: Trainer and W&B run name.
    :type run_name: str
    :param train_file: Train rows file under ``rows/`` of the dataset.
    :type train_file: str
    :param train_rows: Planned rows of that file; one pass is ``train_rows // 8`` steps.
    :type train_rows: int
    :param share_floors: Source-share gate of the data cell, as the tuple literal its loop reads.
    :type share_floors: str
    :param notebook: Default output file name.
    :type notebook: str
    :param lora_rank: LoRA rank (``lora_alpha`` equals it, so the scale stays 1); above :data:`WARM_RANK`
        the warm start is expanded into the larger adapter.
    :type lora_rank: int
    """
    tag: str
    run_name: str
    train_file: str
    train_rows: int
    share_floors: str
    notebook: str
    lora_rank: int = 16

    @property
    def ckpt_repo(self) -> str:
        """Checkpoint repository.

        :return: ``isaacmg/qwen3-vl-8b-hebrew-<tag>-ckpt``, the name the eval harness derives from the tag.
        :rtype: str
        """
        return f"isaacmg/qwen3-vl-8b-hebrew-{self.tag}-ckpt"

    @property
    def merged_repo(self) -> str:
        """Merged-model repository.

        :return: ``isaacmg/qwen3-vl-8b-hebrew-<tag>-merged``.
        :rtype: str
        """
        return f"isaacmg/qwen3-vl-8b-hebrew-{self.tag}-merged"

    @property
    def max_steps(self) -> int:
        """Steps of one pass at batch 1 x accumulation 8.

        :return: ``train_rows // 8``.
        :rtype: int
        """
        return self.train_rows // 8

    @property
    def min_train_rows(self) -> int:
        """Size gate of the data cell.

        :return: 95 % of the planned rows.
        :rtype: int
        """
        return int(self.train_rows * 0.95)

    @property
    def is_control(self) -> bool:
        """Whether this arm trains without the outside handwriting sets.

        :return: ``True`` for the control.
        :rtype: bool
        """
        return self.train_file != "train.parquet"


WARM_RANK = 16                                 # rank of the warm-start adapter (v2.1b step 1200)
MAIN = Arm("v23a", "genizah_v23a", "train.parquet", 12000, SHARE_FLOORS, "genizah_v23a.ipynb")
CONTROL = Arm("v23actl", "genizah_v23a_ctl", CONTROL_TRAIN_FILE, 6960, CONTROL_SHARE_FLOORS, "genizah_v23a_ctl.ipynb")
R64 = Arm("v23ar64", "genizah_v23a_r64", "train.parquet", 12000, SHARE_FLOORS, "genizah_v23a_r64.ipynb", lora_rank=64)
ARMS: Dict[str, Arm] = {"run": MAIN, "control": CONTROL, "r64": R64}
# the run's own values under their old names
CKPT_REPO, MERGED_REPO, RUN_NAME = MAIN.ckpt_repo, MAIN.merged_repo, MAIN.run_name
TRAIN_ROWS, MAX_STEPS, MIN_TRAIN_ROWS = MAIN.train_rows, MAIN.max_steps, MAIN.min_train_rows

OLD_HYGIENE_START = "# Grounding hygiene: valid 0-1000 boxes, self-describing prompts (KTIV + documentary)\n"
OLD_HYGIENE_END = ('print(f"hygiene OK: transcription {len(_transc)}, locate {len(_loc)}, documentary {len(_doc)}, "\n'
                   '      f"read_box prompts {len(_rb)}, QA {len(_qa)} ({Counter(t for *_, t in _qa)})")\n')
NEW_HYGIENE = '''# Arabic hygiene: an Arabic source's answer is mainly Arabic script, its prompt names the script, outside targets
# carry no vowel marks; the replay rows stay Hebrew script. This mixture has no box rows and no QA rows.
import re as _re
_AR, _HE = _re.compile(r"[\\u0620-\\u064a]"), _re.compile(r"[\\u05d0-\\u05ea]")
_MARKS = _re.compile(r"[\\u0640\\u064b-\\u0652]")
_arabic = [(a, q, s) for a, q, s in zip(_ans, _q, _srcs) if s.startswith("arabic_")]
assert _arabic, "no Arabic-script rows in the mixture"
for a, q, s in _arabic:
    _n_ar, _n_he = len(_AR.findall(a)), len(_HE.findall(a))
    assert _n_ar >= 40 and _n_ar >= 9 * _n_he, f"{s}: answer is not mainly Arabic script ({_n_ar} Arabic / {_n_he} Hebrew letters)"
    assert "Arabic script" in q, f"{s}: the prompt does not name Arabic script"
assert all(not _MARKS.search(a) for a, q, s in _arabic if s != "arabic_editions"), "vowel marks or tatweel left in an outside target"
_replay = [a for a, s in zip(_ans, _srcs) if not s.startswith("arabic_")]
assert _replay and sum(len(_HE.findall(a)) for a in _replay) > 20 * sum(len(_AR.findall(a)) for a in _replay), \\
    "the replay rows are not Hebrew script"
assert not any(t in ("locate", "locate_word", "read_box", "read_box_word") or s == "pgp_qa" for t, s in zip(_t, _srcs)), \\
    "box or QA rows in an Arabic mixture"
print(f"hygiene OK: transcription {len(_transc)}, Arabic rows {len(_arabic)} "
      f"({Counter(s for *_, s in _arabic)}), Hebrew replay rows {len(_replay)}")
'''
CONTROL_GATE = f'''assert not set({OUTSIDE_COMPONENTS!r}) & set(_srcs), "outside Arabic rows in the control run"
'''
OLD_LORA_CONFIG = 'r=16, lora_alpha=16, lora_dropout=0.0, bias="none", random_state=3407,'
OLD_WARM_LOAD = "missing = set_peft_model_state_dict(model, _sd)\n"


def rank_expansion_block(rank: int) -> str:
    """Model-cell code that loads the rank-16 warm start into a larger adapter and checks nothing changed.

    :param rank: Rank of the new adapter.
    :type rank: int
    :return: Replacement for the cell's ``set_peft_model_state_dict`` line.
    :rtype: str
    """
    return f'''# ---- rank expansion {WARM_RANK} -> {rank} (function inlined from src/finetuning/qwen_hebrew/lora_rank.py) ----
from typing import Dict, Mapping
from peft import get_peft_model_state_dict


{inspect.getsource(lora_rank.expand_lora_state)}

_warm_sd = _sd
_fresh = {{k: v.detach().to("cpu", copy=True) for k, v in get_peft_model_state_dict(model).items()}}
_sd = expand_lora_state(_warm_sd, _fresh)
print(f"rank expansion {WARM_RANK} -> {rank}: {{sum(1 for k in _sd if k in _fresh and _sd[k].shape != _warm_sd[k].shape)}} LoRA tensors widened, "
      f"{{len(set(_fresh) - set(_sd))}} adapter tensors left at their fresh init")
missing = set_peft_model_state_dict(model, _sd)
# the larger adapter must compute what the warm one did, and its new directions must start at zero
_now = get_peft_model_state_dict(model)
_pairs = [k for k in _warm_sd if "lora_A" in k and _warm_sd[k].dim() == 2]
assert _pairs, "no LoRA matrices in the warm checkpoint"
for _ka in (_pairs[0], _pairs[len(_pairs) // 2], _pairs[-1]):
    _kb = _ka.replace("lora_A", "lora_B")
    _new = _now[_kb].detach().float().cpu() @ _now[_ka].detach().float().cpu()
    _old = _warm_sd[_kb].float() @ _warm_sd[_ka].float()
    assert float((_new - _old).abs().max()) <= 0.02 * float(_old.abs().max()) + 1e-6, f"rank expansion changed the adapter at {{_ka}}"
assert all(float(v[:, {WARM_RANK}:].abs().max()) == 0 for k, v in _now.items() if "lora_B" in k), "new LoRA directions are not zero at the start"
print("rank expansion OK: same function as the warm adapter, new directions at zero")
'''

def intro_md(arm: Arm = MAIN) -> str:
    """Markdown intro of an arm's notebook.

    :param arm: The run or its control.
    :type arm: Arm
    :return: Text of cell 0.
    :rtype: str
    """
    title = ("# Genizah v2.3a control — the Arabic-script run without the outside handwriting sets" if arm.is_control
             else f"# Genizah v2.3a r{arm.lora_rank} — the Arabic-script run with a larger adapter" if arm.lora_rank != WARM_RANK
             else "# Genizah v2.3a — first Arabic-script run")
    data = (f"`{DATA_REPO}`, file `rows/{arm.train_file}`: the rows of the v2.3a run in the same order, without Agapet, "
            "Muharaf, BAYBARS and ISKANDAR (Geniza Arabic editions + Hebrew-script replay only)" if arm.is_control
            else f"`{DATA_REPO}`: Arabic-script pages (Geniza editions, Agapet, Muharaf, BAYBARS, ISKANDAR) + Hebrew-script "
                 "replay (full documentary pages, KTIV transcription)")
    why = ("\nThis is the **control** of `genizah_v23a`: same Geniza Arabic pages, same replay rows, same settings, no outside "
           "data. Comparing the two on the Arabic-script benchmark shows whether the outside sets help on Geniza pages.\n"
           if arm.is_control else
           f"\nThis is `genizah_v23a` with a LoRA adapter of **rank {arm.lora_rank}** instead of {WARM_RANK}: same data, same order, same "
           f"settings. The rank-{WARM_RANK} warm start is placed into the larger adapter without changing the model (its rows of A and "
           "columns of B come first, the new columns of B are zero; the model cell checks this), so training starts from the same "
           "model with four times the adapter capacity. Comparing the two shows whether the adapter's size limits a new script.\n"
           if arm.lora_rank != WARM_RANK else "")
    return f"""{title}

Same training stack as v2.2b (pinned install triplet, 6.5/7 MP resolution contract, LoRA r16 on tower +
language, merger FROZEN, adamw_8bit, cosine, warm start = **v2.1b step 1200**, fixed row order with exact
resume, a session stops itself after 21.5 h with a save and a Hub push). What changes:
{why}
| | v2.2b | this run |
|---|---|---|
| Data | `isaacmg/genizah_v22_full` | {data} |
| Rows / steps | 20,000 / 2,500 | {arm.train_rows:,} / {arm.max_steps:,} |
| Learning rate | 3e-5, 2 % warm-up | {LR}, {float(WARMUP_RATIO):.0%} warm-up: a new script is learned, not adjusted |
| Gates | boxes, QA | Arabic script in Arabic sources, Hebrew script in replay |

**To resume after a session ends: run every cell again, unedited.** The notebook finds the newest checkpoint on the
Hub, checks the stored schedule and row order, and continues with the rows not yet trained.

Run name `{arm.run_name}`; checkpoints in `{arm.ckpt_repo}`. Checkpoints are scored on the Arabic-script benchmark
(`docs/arabic_script_benchmark.md`) and on the Hebrew-script benchmarks for regression.
"""


INTRO_MD = intro_md(MAIN)


def derive(nb: Dict, revision: str, arm: Arm = MAIN) -> Dict:
    """Apply the v23a edits to a loaded v22b notebook.

    :param nb: Notebook JSON of ``genizah_v22b.ipynb`` (not modified in place).
    :type nb: Dict
    :param revision: 40-hex dataset revision to pin (or ``PIN-AFTER-PUSH``).
    :type revision: str
    :param arm: The run (:data:`MAIN`) or its control (:data:`CONTROL`).
    :type arm: Arm
    :return: The arm's notebook JSON.
    :rtype: Dict
    :raises ValueError: When an anchor of the v22b notebook is missing or ambiguous.
    """
    nb = copy.deepcopy(nb)
    cells = nb["cells"]
    if cells[0]["cell_type"] != "markdown":
        raise ValueError("cell 0 of the v22b notebook is expected to be the markdown intro")
    cells[0]["source"] = intro_md(arm).splitlines(keepends=True)
    # --- data cell -----------------------------------------------------------------------------
    c2 = "".join(cells[2]["source"])
    c2 = _replace_once(c2, 'V22_REPO = "isaacmg/genizah_v22_full"', f'V22_REPO = "{DATA_REPO}"', "repo")
    c2 = _replace_line(c2, r'^V22_REVISION = "', f'V22_REVISION = "{revision}"   # v23a Arabic mixture: rows + images_part*.tar ({MAIN.train_rows:,} train rows)', "revision")
    c2 = _replace_once(c2, 'LOCAL = "/content/genizah_v22_full"', 'LOCAL = "/content/genizah_v23a_arabic"', "local dir")
    c2 = _replace_once(c2, 'assert len(train_ds) >= 19000 and len(eval_ds) >= 150, "full mixture incomplete"',
                       f'assert len(train_ds) >= {arm.min_train_rows} and len(eval_ds) >= {MIN_VAL_ROWS}, "Arabic mixture incomplete"', "size gate")
    if arm.is_control:
        c2 = _replace_once(c2, 'train_ds = ImagesOnceDataset(DATA_DIR / "rows/train.parquet", DATA_DIR / "images")',
                           f'train_ds = ImagesOnceDataset(DATA_DIR / "rows/{arm.train_file}", DATA_DIR / "images")   # control: no outside sets', "train rows file")
    c2 = _replace_line(c2, r'^for _s, _lo in \(\("pgp_qa", ', f'for _s, _lo in {arm.share_floors}:', "share floors")
    # KTIV val rows come from every KTIV task, also the box tasks this mixture does not train: val may hold more sources
    c2 = _replace_once(c2, 'assert set(eval_ds.column("source")) == set(_src), "val must cover every train source"',
                       'assert set(_src) <= set(eval_ds.column("source")), "val must cover every train source"', "val coverage")
    start, end = c2.find(OLD_HYGIENE_START), c2.find(OLD_HYGIENE_END)
    if start < 0 or end < 0 or c2.count(OLD_HYGIENE_START) != 1 or c2.count(OLD_HYGIENE_END) != 1:
        raise ValueError("grounding / QA hygiene block not found exactly once in the data cell")
    c2 = c2[:start] + NEW_HYGIENE + (CONTROL_GATE if arm.is_control else "") + c2[end + len(OLD_HYGIENE_END):]
    cells[2]["source"] = c2.splitlines(keepends=True)
    # --- model cell: a larger adapter takes the rank-16 warm start without changing the model ------
    if arm.lora_rank != WARM_RANK:
        c3 = "".join(cells[3]["source"])
        c3 = _replace_once(c3, OLD_LORA_CONFIG, f'r={arm.lora_rank}, lora_alpha={arm.lora_rank}, lora_dropout=0.0, bias="none", random_state=3407,   # scale alpha/r stays 1', "lora rank")
        c3 = _replace_once(c3, OLD_WARM_LOAD, rank_expansion_block(arm.lora_rank), "warm load")
        cells[3]["source"] = c3.splitlines(keepends=True)
    # --- collator cell: the second guardrail row is an Arabic page ----------------------------------
    c4 = "".join(cells[4]["source"])
    c4 = _replace_once(c4, "# guardrail: one real batch (a KTIV page + a QA row) must show page-res pixels + masked labels",
                       "# guardrail: one real batch (a KTIV page + an Arabic-script page) must show page-res pixels + masked labels", "guardrail comment")
    c4 = _replace_once(c4, '_i_qa = next(i for i, s in enumerate(train_ds.column("source")) if s == "pgp_qa")',
                       '_i_qa = next(i for i, s in enumerate(train_ds.column("source")) if s.startswith("arabic_"))   # the Arabic page', "guardrail row")
    c4 = _replace_once(c4, "labels masked on both rows (page + QA)", "labels masked on both rows (Hebrew page + Arabic page)", "guardrail message")
    cells[4]["source"] = c4.splitlines(keepends=True)
    # --- helper cell: steps ---------------------------------------------------------------------
    c7 = "".join(cells[7]["source"])
    c7 = _replace_line(c7, r"^MAX_STEPS = 2500$", f"MAX_STEPS = {arm.max_steps}", "max steps")
    c7 = _replace_line(c7, r"^TIME_BUDGET_H = 21.5$", f"TIME_BUDGET_H = {TIME_BUDGET_H}   # Colab killed both v23a sessions at ~8 h; see derive_v23a.TIME_BUDGET_H", "time budget")
    cells[7]["source"] = c7.splitlines(keepends=True)
    # --- training cell --------------------------------------------------------------------------
    c8 = "".join(cells[8]["source"])
    kind = ", control" if arm.is_control else f", rank {arm.lora_rank}" if arm.lora_rank != WARM_RANK else ""
    c8 = _replace_once(c8, "# Cell 6 — v2.2b training:", f"# Cell 6 — v2.3a (Arabic script{kind}) training:", "cell title")
    c8 = _replace_line(c8, r'^CKPT_REPO = "isaacmg/qwen3-vl-8b-hebrew-v22b-ckpt"',
                       f'CKPT_REPO = "{arm.ckpt_repo}"  # NEW repo: never reuse another run\'s (auto-resume would load its last-checkpoint)', "ckpt repo")
    c8 = _replace_once(c8, 'OUT_DIR = "outputs_v22b"', f'OUT_DIR = "outputs_{arm.tag}"', "out dir")
    c8 = _replace_once(c8, 'RESUME_ROOT = "/content/v22b_resume"', f'RESUME_ROOT = "/content/{arm.tag}_resume"', "resume root")
    c8 = _replace_line(c8, r"^\s*max_steps=MAX_STEPS,", f"        max_steps=MAX_STEPS,               # {arm.train_rows:,} rows = one pass = {arm.max_steps:,} steps; sessions end on the time budget and resume exactly", "max_steps")
    c8 = _replace_line(c8, r"^\s*learning_rate=3e-5,", f"        learning_rate={LR},                # a new script has to be learned; v22b's 3e-5 only adjusted a converged model", "lr")
    c8 = _replace_once(c8, 'warmup_ratio=0.02, lr_scheduler_type="cosine", weight_decay=0.01,',
                       f'warmup_ratio={WARMUP_RATIO}, lr_scheduler_type="cosine", weight_decay=0.01,', "warm-up")
    c8 = _replace_once(c8, "save_steps=100, save_total_limit=2,",
                       f"save_steps={SAVE_STEPS}, save_total_limit=2,   # a termination costs at most {SAVE_STEPS} steps (the Hub's last-checkpoint follows every save)", "save steps")
    # the Hebrew runs publish their checkpoints; this one trains on Muharaf (CC BY-NC-SA), so its repo starts private
    c8 = _replace_once(c8, 'hub_strategy="checkpoint", hub_private_repo=False,',
                       'hub_strategy="checkpoint", hub_private_repo=True,    # Muharaf is CC BY-NC-SA: private until the licence question is decided', "private checkpoints")
    c8 = _replace_once(c8, 'run_name="genizah_v22b"', f'run_name="{arm.run_name}"', "run name")
    c8 = _replace_once(c8, 'wandb.init(project="qwen-hebrew-finetune", name="genizah_v22b", ', f'wandb.init(project="qwen-hebrew-finetune", name="{arm.run_name}", ', "wandb name")
    cells[8]["source"] = c8.splitlines(keepends=True)
    # --- merged-model cell: names -----------------------------------------------------------------
    c9 = "".join(cells[9]["source"])
    c9 = _replace_once(c9, 'MERGED_REPO = "isaacmg/qwen3-vl-8b-hebrew-v22a-merged"', f'MERGED_REPO = "{arm.merged_repo}"', "merged repo")
    if '"v22b-merged"' not in c9:
        raise ValueError("merged-model folder name not found in the last cell")
    c9 = c9.replace('"v22b-merged"', f'"{arm.tag}-merged"')
    cells[9]["source"] = c9.splitlines(keepends=True)
    return nb


def main(argv: Optional[List[str]] = None) -> None:
    """CLI.

    :param argv: Arguments; None = ``sys.argv[1:]``.
    :type argv: Optional[List[str]]
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--revision", required=True, help="40-hex dataset revision, or PIN-AFTER-PUSH")
    ap.add_argument("--src", type=Path, default=HERE / "genizah_v22b.ipynb")
    ap.add_argument("--out", type=Path, default=None, help="default: the arm's own file under colab/ (one arm only)")
    ap.add_argument("--arm", choices=[*ARMS, "all"], default="run",
                    help="run = the run; control = without the outside sets; r64 = rank-64 adapter; all = the three files")
    ap.add_argument("--control", action="store_true", help="same as --arm control")
    ap.add_argument("--host", choices=["colab", "lambda"], default="colab",
                    help="lambda = the Lambda Cloud edition (<notebook>_lambda.ipynb + lambda_setup.sh beside it)")
    args = ap.parse_args(argv)
    if args.revision != "PIN-AFTER-PUSH" and not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise SystemExit("revision must be a 40-hex sha or PIN-AFTER-PUSH")
    names = list(ARMS) if args.arm == "all" else ["control" if args.control else args.arm]
    if args.out and len(names) > 1:
        raise SystemExit("--out names one file: use it with one arm")
    nb = json.load(open(args.src, encoding="utf-8"))
    for name in names:
        arm = ARMS[name]
        path = args.out or HERE / arm.notebook
        out = derive(nb, args.revision, arm)
        if args.host == "lambda":
            from src.finetuning.qwen_hebrew.colab import host_lambda
            nb_name, sh_name = host_lambda.lambda_files(path.name)
            path = path.with_name(nb_name)
            out = host_lambda.to_lambda(out, nb_name)
            (path.parent / sh_name).write_text(host_lambda.setup_script(), encoding="utf-8")
            print(f"wrote {path.parent / sh_name}")
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {path} ({len(out['cells'])} cells; revision {args.revision}; run {arm.run_name}, {arm.train_rows:,} rows, "
              f"rank {arm.lora_rank}; host {args.host})")


if __name__ == "__main__":
    main()
