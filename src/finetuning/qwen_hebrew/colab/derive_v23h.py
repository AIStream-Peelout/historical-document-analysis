# File name: derive_v23h.py
# Date: 10/12/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Derive the v23h notebook: the Hebrew-only ablation arm of v2.3a, Lambda Cloud edition by default.

v2.3a (the first Arabic-script run) lowered PGP-131 CER* from 0.160 (v2.2b) to 0.153, and the gain sits in the
Judaeo-Arabic / Arabic-language buckets. v23h asks whether the Arabic-script rows caused it (cross-script transfer)
or v2.3a's exposure and learning rate did. Its data is v2.3a's mixture minus every row of an Arabic-script source
(``isaacmg/genizah_v23h_hebonly``, written by :mod:`src.finetuning.qwen_hebrew.subset_mixture`: the same replay
rows in the same stored order, each as often as in v2.3a); its settings are v2.3a's (warm start v2.1b step 1200,
LR 1e-4, 5 % warm-up, cosine, LoRA r16, merger frozen, seed 3407, 50-step saves); one pass is 5,640 rows = 705 steps.
If v23h also reaches about 0.153 the gain came from exposure and LR; if it stays near 0.160 the Arabic data did it.

The notebook is v2.3a's (:func:`derive_v23a.derive` with the v23h :class:`derive_v23a.Arm`: repositories, folders,
run name, size gate, share floors, ``max_steps``), then these exact, asserted edits:

* dataset repository, revision comment and local folder -> the Hebrew-only mixture; the val size gate -> its 80 rows;
* the Arabic hygiene gate -> a Hebrew-only gate (no Arabic-source row, no prompt naming Arabic script, answers in
  Hebrew script, no box or QA rows);
* the collator guardrail's second row -> a full documentary page (the run has no Arabic page);
* intro, cell title and two comments; the checkpoint repository stays private, as v2.3a's;

then the Lambda edition (:func:`host_lambda.to_lambda`) with a session time budget sized for one Lambda session.

Usage::

    .venv/bin/python -m src.finetuning.qwen_hebrew.colab.derive_v23h --revision <40-hex sha | PIN-AFTER-PUSH> \\
        [--host lambda|colab] [--src colab/genizah_v22b.ipynb] [--out FILE]
"""
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from src.finetuning.qwen_hebrew.colab import derive_v23a as dv
from src.finetuning.qwen_hebrew.colab import host_lambda
from src.finetuning.qwen_hebrew.colab.derive_v22b import _replace_line, _replace_once

HERE = Path(__file__).resolve().parent

DATA_REPO = "isaacmg/genizah_v23h_hebonly"
LOCAL_LEAF = "genizah_v23h_hebonly"
SOURCE_REVISION = "3b762524067c77141c121ce84a3a724991618757"     # the v2.3a mixture the rows were taken from
TRAIN_ROWS = 5640                                                 # v2.3a's 2,280 documentary pages + 3,360 KTIV rows
# val = v2.3a's 80 Hebrew-script val rows (40 documentary pages + 40 KTIV; its 73 Arabic val rows are left out with their
# components, as the builder keeps val to trained components); the gate takes 95 % of that, as the train gate does
VAL_ROWS = 80
MIN_VAL_ROWS = 76
# floors at about 0.82 of the realised shares (pgp_edition_pages 0.404, ktiv_transcription 0.596), as in derive_v23a
SHARE_FLOORS = '(("pgp_edition_pages", 0.33), ("ktiv_transcription", 0.49))'
ARM = dv.Arm("v23h", "genizah_v23h", "train.parquet", TRAIN_ROWS, SHARE_FLOORS, "genizah_v23h.ipynb")
# v2.3a on the Lambda A100-SXM4-40GB (W&B run e0mgttev, logs/colab_watch.log): steps 810 -> 1500 in 12 h 11 min,
# evals and 50-step saves included = 63.6 s a step. One v23h session: 705 x 63.6 s = 12.5 h. The budget is a safety
# stop at about 1.4x that (a run that stops on it saves, pushes, and resumes exactly in a second session).
MEASURED_S_PER_STEP = 63.6
LAMBDA_TIME_BUDGET_H = 18.0

HEB_HYGIENE = '''# Hebrew-only hygiene (v23h = v2.3a minus every row of an Arabic-script source): no Arabic-source row, no prompt naming
# Arabic script, the answers are Hebrew script (v2.3a's replay carried 1,645 Arabic letters on 24 mixed-script documentary
# pages against 2.1 M Hebrew ones; the same rows are here). This mixture has no box rows and no QA rows.
import re as _re
_AR, _HE = _re.compile(r"[\\u0620-\\u064a]"), _re.compile(r"[\\u05d0-\\u05ea]")
assert not any(s.startswith("arabic_") for s in _srcs), "Arabic-source rows in the Hebrew-only run"
assert not any("Arabic script" in q for q in _q), "a prompt names Arabic script in the Hebrew-only run"
assert not any(len(_AR.findall(a)) >= 40 and len(_AR.findall(a)) >= 9 * len(_HE.findall(a)) for a in _ans), \\
    "a row is mainly Arabic script (v2.3a's rule for an Arabic row)"
_n_he, _n_ar = sum(len(_HE.findall(a)) for a in _ans), sum(len(_AR.findall(a)) for a in _ans)
assert _n_he > 20 * _n_ar, f"the rows are not Hebrew script ({_n_he} Hebrew / {_n_ar} Arabic letters)"
assert not any(t in ("locate", "locate_word", "read_box", "read_box_word") or s == "pgp_qa" for t, s in zip(_t, _srcs)), \\
    "box or QA rows in the Hebrew-only run"
print(f"hygiene OK: transcription {len(_transc)}, Hebrew-only rows {len(_ans)} ({Counter(_srcs)}), "
      f"Arabic letters {_n_ar} of {_n_he + _n_ar}")
'''


def intro_md() -> str:
    """Markdown intro of the v23h notebook.

    :return: Text of cell 0 (the Lambda edition puts its launch notes in front of it).
    :rtype: str
    """
    return f"""# Genizah v2.3h — v2.3a without its Arabic-script rows (Hebrew-only ablation)

v2.3a (the first Arabic-script run) lowered PGP-131 CER* from 0.160 (v2.2b) to 0.153, and the gain sits in the
Judaeo-Arabic / Arabic-language buckets. This run asks whether the Arabic-script rows caused it (cross-script
transfer) or v2.3a's exposure and learning rate did: **same replay rows, same order, same settings, no Arabic
rows.** If it also reaches about 0.153 the gain came from exposure + LR; if it stays near 0.160 the Arabic data did it.

Same training stack as v2.3a (pinned install triplet, 6.5/7 MP resolution contract, LoRA r16 on tower + language,
merger FROZEN, adamw_8bit, cosine, warm start = **v2.1b step 1200**, seed 3407, fixed row order with exact resume,
50-step saves, prepared-dataloader gate before step 1). What changes:

| | v2.3a | this run |
|---|---|---|
| Data | `{dv.DATA_REPO}` (Arabic-script pages + Hebrew-script replay) | `{DATA_REPO}`: v2.3a's rows minus every Arabic-source row (full documentary pages 2,280 + KTIV transcription 3,360, v2.3a's stored order) |
| Rows / steps | 12,000 / 1,500 | {ARM.train_rows:,} / {ARM.max_steps:,} (one pass = the exposure each of these rows had in v2.3a) |
| Learning rate | {dv.LR}, {float(dv.WARMUP_RATIO):.0%} warm-up (75 steps) | {dv.LR}, {float(dv.WARMUP_RATIO):.0%} warm-up (36 steps) |
| Gates | Arabic script in Arabic sources, Hebrew script in replay | no Arabic-source row, Hebrew script throughout |
| Val | 153 rows (73 Arabic) | the 80 Hebrew-script val rows of v2.3a |

**To resume after a session ends: run every cell again, unedited.** The notebook finds the newest checkpoint on the
Hub, checks the stored schedule and row order, and continues with the rows not yet trained.

Run name `{ARM.run_name}`; checkpoints in `{ARM.ckpt_repo}`. Checkpoints 300 / 550 / 705 (v2.3a's comparison points
600 / 1200 / 1500 at the same fraction of the schedule) are scored on PGP-131, then religious-140 at 705, on the
Studio (`logs/v23h_eval_chain.sh`).
"""


def derive(nb: Dict, revision: str) -> Dict:
    """Apply the v23h edits to a loaded v22b notebook (Colab edition).

    :param nb: Notebook JSON of ``genizah_v22b.ipynb`` (not modified in place).
    :type nb: Dict
    :param revision: 40-hex revision of :data:`DATA_REPO` (or ``PIN-AFTER-PUSH``).
    :type revision: str
    :return: The v23h Colab notebook.
    :rtype: Dict
    :raises ValueError: When an anchor of the v2.3a notebook is missing or ambiguous.
    """
    out = dv.derive(nb, revision, ARM)
    cells = out["cells"]
    cells[0]["source"] = intro_md().splitlines(keepends=True)
    # --- data cell ---------------------------------------------------------------------------------
    c2 = "".join(cells[2]["source"])
    c2 = _replace_once(c2, f'V22_REPO = "{dv.DATA_REPO}"', f'V22_REPO = "{DATA_REPO}"', "repo")
    c2 = _replace_line(c2, r'^V22_REVISION = "',
                       f'V22_REVISION = "{revision}"   # v23h Hebrew-only mixture (v2.3a minus its Arabic-script rows): '
                       f'rows + images_part*.tar ({ARM.train_rows:,} train rows)', "revision")
    c2 = _replace_once(c2, 'LOCAL = "/content/genizah_v23a_arabic"', f'LOCAL = "/content/{LOCAL_LEAF}"', "local dir")
    c2 = _replace_once(c2, f'assert len(train_ds) >= {ARM.min_train_rows} and len(eval_ds) >= {dv.MIN_VAL_ROWS}, "Arabic mixture incomplete"',
                       f'assert len(train_ds) >= {ARM.min_train_rows} and len(eval_ds) >= {MIN_VAL_ROWS}, "Hebrew-only mixture incomplete"   '
                       f'# {VAL_ROWS} val rows: v2.3a\'s Hebrew-script ones', "size gate")
    c2 = _replace_once(c2, dv.NEW_HYGIENE, HEB_HYGIENE, "hygiene gate")
    cells[2]["source"] = c2.splitlines(keepends=True)
    # --- collator cell: the second guardrail row is a full documentary page -------------------------
    c4 = "".join(cells[4]["source"])
    c4 = _replace_once(c4, "(a KTIV page + an Arabic-script page)", "(a KTIV page + a full documentary page)", "guardrail comment")
    c4 = _replace_once(c4, 'if s.startswith("arabic_"))   # the Arabic page',
                       'if s == "pgp_edition_pages")   # a full documentary page (no Arabic page in this run)', "guardrail row")
    c4 = _replace_once(c4, "(Hebrew page + Arabic page)", "(KTIV page + documentary page)", "guardrail message")
    cells[4]["source"] = c4.splitlines(keepends=True)
    # --- training cell -------------------------------------------------------------------------------
    c8 = "".join(cells[8]["source"])
    c8 = _replace_once(c8, "# Cell 6 — v2.3a (Arabic script) training:", "# Cell 6 — v2.3h (Hebrew-only ablation of v2.3a) training:", "cell title")
    c8 = _replace_once(c8, "# a new script has to be learned; v22b's 3e-5 only adjusted a converged model",
                       "# v2.3a's: the ablation changes the data only", "lr comment")
    c8 = _replace_once(c8, "# Muharaf is CC BY-NC-SA: private until the licence question is decided",
                       "# private like v2.3a's (PGP + KTIV data only: making it public is the owner's call)", "private comment")
    cells[8]["source"] = c8.splitlines(keepends=True)
    return out


def to_lambda(nb: Dict, notebook_name: str, budget_h: float = LAMBDA_TIME_BUDGET_H) -> Dict:
    """The Lambda edition with a session budget sized for one Lambda session.

    :param nb: :func:`derive` output.
    :type nb: Dict
    :param notebook_name: File name the edition is written as (quoted in the launch notes).
    :type notebook_name: str
    :param budget_h: Safety stop in hours.
    :type budget_h: float
    :return: The Lambda edition.
    :rtype: Dict
    """
    out = host_lambda.to_lambda(nb, notebook_name)
    c7 = "".join(out["cells"][7]["source"])
    c7 = _replace_line(c7, r"^TIME_BUDGET_H = .*$",
                       f"TIME_BUDGET_H = {budget_h}   # Lambda, one session: {ARM.max_steps} steps x ~{MEASURED_S_PER_STEP:.0f} s "
                       f"(v2.3a on the A100-SXM4-40GB) = {ARM.max_steps * MEASURED_S_PER_STEP / 3600:.1f} h; a safety stop only, "
                       "max_steps ends the run", "time budget")
    out["cells"][7]["source"] = c7.splitlines(keepends=True)
    return out


def main(argv: Optional[List[str]] = None) -> None:
    """CLI.

    :param argv: Arguments; None = ``sys.argv[1:]``.
    :type argv: Optional[List[str]]
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--revision", required=True, help="40-hex revision of the v23h dataset, or PIN-AFTER-PUSH")
    ap.add_argument("--src", type=Path, default=HERE / "genizah_v22b.ipynb")
    ap.add_argument("--host", choices=["lambda", "colab"], default="lambda",
                    help="lambda (default) = genizah_v23h.ipynb + lambda_setup.sh; colab = genizah_v23h_colab.ipynb")
    ap.add_argument("--out", type=Path, default=None, help="default: colab/genizah_v23h.ipynb (lambda) or colab/genizah_v23h_colab.ipynb")
    args = ap.parse_args(argv)
    if args.revision != "PIN-AFTER-PUSH" and not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise SystemExit("revision must be a 40-hex sha or PIN-AFTER-PUSH")
    nb = derive(json.load(open(args.src, encoding="utf-8")), args.revision)
    if args.host == "lambda":
        path = args.out or HERE / ARM.notebook
        nb = to_lambda(nb, path.name)
        (path.parent / host_lambda.SETUP_SCRIPT).write_text(host_lambda.setup_script(), encoding="utf-8")
        print(f"wrote {path.parent / host_lambda.SETUP_SCRIPT}")
    else:
        path = args.out or HERE / "genizah_v23h_colab.ipynb"
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {path} ({len(nb['cells'])} cells; dataset {DATA_REPO}@{args.revision}; run {ARM.run_name}, "
          f"{ARM.train_rows:,} rows = {ARM.max_steps} steps; host {args.host})")


if __name__ == "__main__":
    main()
