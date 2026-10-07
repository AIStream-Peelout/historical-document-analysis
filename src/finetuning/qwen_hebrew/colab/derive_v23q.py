# File name: derive_v23q.py
# Date: 10/5/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Derive the v23q Colab notebook (page parse + questions answered from the image and a parse) from v22b.

Why: v22b's direct question rows were never learned (the model answered with the most common month).
With the page's line-by-line reading pasted into the prompt the same checkpoint answered from the page.
``pgp_vqa_parse_v1`` (``build_vqa_parse.py``) teaches both halves: parse a page to JSON, then fill
fields and answer questions from the image plus a parse, citing the line. This notebook is the pilot
run of that set.

The v22b notebook is the source of truth for the training stack (pinned install, 6.5/7 MP resolution
contract, LoRA r16 on tower + language, merger frozen, warm start v2.1b step 1200, fixed row order with
exact resume, time-budget stop). The v23q deltas are exact, asserted string edits:

* dataset -> the page-parse mixture (``isaacmg/genizah_v23q_vqa`` at a pinned revision); size and share
  gates are those of this mixture;
* the grounding and question gates of the data cell are replaced by page-parse gates: every target of a
  parse, fields, question or look-up row is JSON of the agreed shape, every cited line number exists in
  the reading its prompt shows, and no row of the older question or box formats is present;
* the collator guardrail checks a page and a question row of the new form;
* checkpoint repo / output dir / resume dir / run name / merged repo -> ``v23q``;
* ``max_steps`` for one pass, learning rate 1e-4 with a 5 % warm-up (new task forms have to be learned;
  v22b's 3e-5 did not move the question rows).

Usage::

    .venv/bin/python -m src.finetuning.qwen_hebrew.colab.derive_v23q --revision <40-hex sha> \\
        [--src colab/genizah_v22b.ipynb] [--out colab/genizah_v23q.ipynb]
"""
import argparse
import copy
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from src.finetuning.qwen_hebrew.build_vqa_parse import PARSE_LEAD
from src.finetuning.qwen_hebrew.colab.derive_v22b import _replace_line, _replace_once
from src.finetuning.qwen_hebrew.colab.derive_v23a import OLD_HYGIENE_END, OLD_HYGIENE_START

HERE = Path(__file__).resolve().parent

DATA_REPO = "isaacmg/genizah_v23q_vqa"
TAG = "v23q"
CKPT_REPO = f"isaacmg/qwen3-vl-8b-hebrew-{TAG}-ckpt"
MERGED_REPO = f"isaacmg/qwen3-vl-8b-hebrew-{TAG}-merged"
RUN_NAME = "genizah_v23q"
TRAIN_ROWS = 8000
MAX_STEPS = TRAIN_ROWS // 8                    # batch 1 x accumulation 8: one pass, one Colab session
LR = "1e-4"
WARMUP_RATIO = "0.05"
MIN_TRAIN_ROWS = int(TRAIN_ROWS * 0.95)
MIN_VAL_ROWS = 100
# floors at about 0.8 of the planned shares (SHARES in logs/next_round/vqa/build_v23q.sh)
SHARE_FLOORS = ('(("vqa_parse_lines", 0.20), ("vqa_question", 0.17), ("vqa_lookup", 0.11), ("vqa_fields", 0.05), '
                '("vqa_parse_boxes", 0.02), ("pgp_edition_pages", 0.08), ("ktiv_transcription", 0.14))')

NEW_HYGIENE = f'''# Page-parse hygiene: every target of a parse, fields, question or look-up row is JSON of the agreed shape and
# every cited line number exists in the reading its prompt shows. No row of the older question or box formats.
_LEAD = {PARSE_LEAD!r}
def _ok_box(b):
    return len(b) == 4 and all(0 <= v <= 1000 for v in b) and b[2] > b[0] and b[3] > b[1]
def _ok_line(n, n_lines):
    return isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= n_lines
def _ok_cite(o, n_lines):
    return isinstance(o, dict) and isinstance(o.get("text"), str) and bool(o["text"].strip()) and _ok_line(o.get("line"), n_lines)
_vqa = [(a, q, t, s) for a, q, t, s in zip(_ans, _q, _t, _srcs) if s.startswith("vqa_")]
assert _vqa, "no page-parse rows in the mixture"
for a, q, t, s in _vqa:
    o = json.loads(a)
    assert "JSON" in q, f"{{s}}: the prompt does not ask for JSON"
    if t in ("parse_lines", "parse_lines_boxes"):
        assert isinstance(o, list) and o and [e["n"] for e in o] == list(range(1, len(o) + 1)), f"{{s}}: parse numbering"
        assert all(isinstance(e["text"], str) and e["text"].strip() for e in o), f"{{s}}: empty parse line"
        if t == "parse_lines_boxes":
            assert all(set(e) == {{"n", "text", "bbox_2d"}} and _ok_box(e["bbox_2d"]) for e in o) and "0-1000" in q, f"{{s}}: bad line box"
        else:
            assert all(set(e) == {{"n", "text"}} for e in o), f"{{s}}: parse keys"
        continue
    assert q.startswith(_LEAD) and "\\n]\\n" in q, f"{{s}}: the prompt does not open with the reading"
    _n = len(json.loads(q[len(_LEAD): q.index("\\n]\\n") + 2]))                       # lines of the reading shown
    if t == "fields_from_parse":
        assert isinstance(o, dict) and o, a
        assert all(v is None or _ok_cite(v, _n) or (isinstance(v, list) and v and all(_ok_cite(i, _n) for i in v))
                   for v in o.values()), f"{{s}}: bad field value in {{a}}"
    elif t == "question_from_parse":
        assert isinstance(o, dict) and "answer" in o, a
        _v = o["answer"]
        assert (_v is None and set(o) == {{"answer"}}) or (isinstance(_v, str) and _v.strip() and _ok_line(o.get("line"), _n)) \\
            or (isinstance(_v, list) and _v and all(_ok_cite(i, _n) for i in _v)), f"{{s}}: bad answer {{a}}"
    elif t == "lookup_from_parse":
        assert isinstance(o, dict) and _ok_line(o.get("line"), _n) and isinstance(o.get("text", o.get("answer")), str), \\
            f"{{s}}: bad look-up answer {{a}}"
    else:
        raise AssertionError(f"{{s}}: unknown page-parse task {{t}}")
assert any(json.loads(a) == {{"answer": None}} for a, q, t, s in _vqa if t == "question_from_parse"), "no 'not stated' rows"
assert not any(s in ("pgp_qa", "documentary_grounding", "ktiv_grounding") for s in _srcs), \\
    "rows of the older question or box formats in a page-parse mixture"
print(f"hygiene OK: transcription {{len(_transc)}}, page-parse rows {{len(_vqa)}} ({{Counter(s for *_, s in _vqa)}})")
'''

INTRO_MD = f"""# Genizah v2.3q — page parse and questions answered from the page (pilot)

Same training stack as v2.2b (pinned install triplet, 6.5/7 MP resolution contract, LoRA r16 on tower +
language, merger FROZEN, adamw_8bit, cosine, warm start = **v2.1b step 1200**, fixed row order with exact
resume, a session stops itself after 21.5 h with a save and a Hub push). What changes:

| | v2.2b | v2.3q |
|---|---|---|
| Data | `isaacmg/genizah_v22_full` | `{DATA_REPO}`: page parse to JSON (lines, some with boxes), fields and questions answered from the image plus a line-by-line reading in the prompt, generic look-ups; replay = full documentary pages and KTIV transcription |
| Rows / steps | 20,000 / 2,500 | {TRAIN_ROWS:,} / {MAX_STEPS:,} |
| Learning rate | 3e-5, 2 % warm-up | {LR}, {float(WARMUP_RATIO):.0%} warm-up: v2.2b's question rows were not learned at 3e-5 |
| Gates | boxes, old QA format | JSON shape of every page-parse target, cited lines exist in the reading shown |

Every target text is a human edition line. The model's own reading appears only inside prompts, as context.

**To resume after a session ends: run every cell again, unedited.** The notebook finds the newest checkpoint on the
Hub, checks the stored schedule and row order, and continues with the rows not yet trained.

Run name `{RUN_NAME}`; checkpoints in `{CKPT_REPO}`.
"""


def derive(nb: Dict, revision: str) -> Dict:
    """Apply the v23q edits to a loaded v22b notebook.

    :param nb: Notebook JSON of ``genizah_v22b.ipynb`` (not modified in place).
    :type nb: Dict
    :param revision: 40-hex dataset revision to pin (or ``PIN-AFTER-PUSH``).
    :type revision: str
    :return: The v23q notebook JSON.
    :rtype: Dict
    :raises ValueError: When an anchor of the v22b notebook is missing or ambiguous.
    """
    nb = copy.deepcopy(nb)
    cells = nb["cells"]
    if cells[0]["cell_type"] != "markdown":
        raise ValueError("cell 0 of the v22b notebook is expected to be the markdown intro")
    cells[0]["source"] = INTRO_MD.splitlines(keepends=True)
    # --- data cell -----------------------------------------------------------------------------
    c2 = "".join(cells[2]["source"])
    c2 = _replace_once(c2, 'V22_REPO = "isaacmg/genizah_v22_full"', f'V22_REPO = "{DATA_REPO}"', "repo")
    c2 = _replace_line(c2, r'^V22_REVISION = "', f'V22_REVISION = "{revision}"   # v23q page-parse mixture: rows + images_part*.tar ({TRAIN_ROWS:,} train rows)', "revision")
    c2 = _replace_once(c2, 'LOCAL = "/content/genizah_v22_full"', 'LOCAL = "/content/genizah_v23q_vqa"', "local dir")
    c2 = _replace_once(c2, 'assert len(train_ds) >= 19000 and len(eval_ds) >= 150, "full mixture incomplete"',
                       f'assert len(train_ds) >= {MIN_TRAIN_ROWS} and len(eval_ds) >= {MIN_VAL_ROWS}, "page-parse mixture incomplete"', "size gate")
    c2 = _replace_line(c2, r'^for _s, _lo in \(\("pgp_qa", ', f'for _s, _lo in {SHARE_FLOORS}:', "share floors")
    start, end = c2.find(OLD_HYGIENE_START), c2.find(OLD_HYGIENE_END)
    if start < 0 or end < 0 or c2.count(OLD_HYGIENE_START) != 1 or c2.count(OLD_HYGIENE_END) != 1:
        raise ValueError("grounding / QA hygiene block not found exactly once in the data cell")
    c2 = c2[:start] + NEW_HYGIENE + c2[end + len(OLD_HYGIENE_END):]
    cells[2]["source"] = c2.splitlines(keepends=True)
    # --- collator cell: the second guardrail row is a question answered from a reading -------------------
    c4 = "".join(cells[4]["source"])
    c4 = _replace_once(c4, "# guardrail: one real batch (a KTIV page + a QA row) must show page-res pixels + masked labels",
                       "# guardrail: one real batch (a KTIV page + a question row with a reading in its prompt) must show page-res pixels + masked labels", "guardrail comment")
    c4 = _replace_once(c4, '_i_qa = next(i for i, s in enumerate(train_ds.column("source")) if s == "pgp_qa")',
                       '_i_qa = next(i for i, s in enumerate(train_ds.column("source")) if s == "vqa_question")', "guardrail row")
    c4 = _replace_once(c4, "labels masked on both rows (page + QA)", "labels masked on both rows (page + question with a reading)", "guardrail message")
    cells[4]["source"] = c4.splitlines(keepends=True)
    # --- helper cell: steps ---------------------------------------------------------------------
    c7 = "".join(cells[7]["source"])
    c7 = _replace_line(c7, r"^MAX_STEPS = 2500$", f"MAX_STEPS = {MAX_STEPS}", "max steps")
    cells[7]["source"] = c7.splitlines(keepends=True)
    # --- training cell --------------------------------------------------------------------------
    c8 = "".join(cells[8]["source"])
    c8 = _replace_once(c8, "# Cell 6 — v2.2b training:", "# Cell 6 — v2.3q (page parse + questions) training:", "cell title")
    c8 = _replace_line(c8, r'^CKPT_REPO = "isaacmg/qwen3-vl-8b-hebrew-v22b-ckpt"',
                       f'CKPT_REPO = "{CKPT_REPO}"  # NEW repo: never reuse another run\'s (auto-resume would load its last-checkpoint)', "ckpt repo")
    c8 = _replace_once(c8, 'OUT_DIR = "outputs_v22b"', f'OUT_DIR = "outputs_{TAG}"', "out dir")
    c8 = _replace_once(c8, 'RESUME_ROOT = "/content/v22b_resume"', f'RESUME_ROOT = "/content/{TAG}_resume"', "resume root")
    c8 = _replace_line(c8, r"^\s*max_steps=MAX_STEPS,", f"        max_steps=MAX_STEPS,               # {TRAIN_ROWS:,} rows = one pass = {MAX_STEPS:,} steps; sessions end on the time budget and resume exactly", "max_steps")
    c8 = _replace_line(c8, r"^\s*learning_rate=3e-5,", f"        learning_rate={LR},                # new task forms have to be learned; v22b's 3e-5 did not move the question rows", "lr")
    c8 = _replace_once(c8, 'warmup_ratio=0.02, lr_scheduler_type="cosine", weight_decay=0.01,',
                       f'warmup_ratio={WARMUP_RATIO}, lr_scheduler_type="cosine", weight_decay=0.01,', "warm-up")
    c8 = _replace_once(c8, 'run_name="genizah_v22b"', f'run_name="{RUN_NAME}"', "run name")
    c8 = _replace_once(c8, 'wandb.init(project="qwen-hebrew-finetune", name="genizah_v22b", ', f'wandb.init(project="qwen-hebrew-finetune", name="{RUN_NAME}", ', "wandb name")
    cells[8]["source"] = c8.splitlines(keepends=True)
    # --- merged-model cell: names -----------------------------------------------------------------
    c9 = "".join(cells[9]["source"])
    c9 = _replace_once(c9, 'MERGED_REPO = "isaacmg/qwen3-vl-8b-hebrew-v22a-merged"', f'MERGED_REPO = "{MERGED_REPO}"', "merged repo")
    if '"v22b-merged"' not in c9:
        raise ValueError("merged-model folder name not found in the last cell")
    c9 = c9.replace('"v22b-merged"', f'"{TAG}-merged"')
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
    ap.add_argument("--out", type=Path, default=HERE / "genizah_v23q.ipynb")
    args = ap.parse_args(argv)
    if args.revision != "PIN-AFTER-PUSH" and not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise SystemExit("revision must be a 40-hex sha or PIN-AFTER-PUSH")
    nb = json.load(open(args.src, encoding="utf-8"))
    out = derive(nb, args.revision)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {args.out} ({len(out['cells'])} cells; revision {args.revision}; run {RUN_NAME}, {TRAIN_ROWS:,} rows)")


if __name__ == "__main__":
    main()
