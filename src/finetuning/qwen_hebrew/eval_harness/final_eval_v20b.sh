#!/bin/zsh
# v20b FINAL (step 2000): stage -> grounding trio (canonical results + preds) -> box geometry vs v20a
# -> layout-QA -> 10 site pages (raw template rate) -> full hard evals (religious-140 + PGP-131 + compare + W&B,
# then harness cleanup: NAS masters kept, local copies removed since v20b is not flagship) -> six-way table.
set -e
REPO=/Users/isaac/Documents/GitHub/historical-document-analysis
H=$REPO/src/finetuning/qwen_hebrew/eval_harness
G=$REPO/src/datasets/evaluations/grounding_eval
OUT=$REPO/src/datasets/raw_data/cairo_genizah/ai_reads
STEP=2000; SHA=4cae9d52d9cd167cb0d1ddd0248ed728c7308f29
NAME=qwen3-vl-8b-heb-v20b-step$STEP
echo "===== v20b FINAL EVAL step $STEP $(date) ====="
$H/hard_eval_ckpt.sh $STEP $SHA stage v20b || { echo "!!! NAS staging failed — ramdisk retry"; FORCE_LOCAL=1 $H/hard_eval_ckpt.sh $STEP $SHA stage v20b; }
for MODE in locate read_box grounded; do
  cd $REPO && .venv/bin/python $G/grounding_eval.py --run $MODE --model $NAME 2>&1 | grep -v "^INFO\|^DEBUG"
done
echo "===== box geometry $(date) ====="
cd $REPO && .venv/bin/python $G/box_quality.py --preds $NAME qwen3-vl-8b-heb-v20a-step1800 qwen3-vl-8b-heb-v20b-step1300 2>&1 | grep -v "^INFO\|^DEBUG"
echo "===== layout-QA $(date) ====="
cd $REPO && .venv/bin/python $H/layout_qa_eval.py --model $NAME 2>&1 | grep -v "^INFO\|^DEBUG" | tail -8
echo "===== 10 site pages $(date) ====="
cd $REPO && .venv/bin/python -m src.datasets.consensus.two_reader_lines --ids $OUT/jobs_v20b_sample10.jsonl --vlm-model $NAME \
  --out $OUT/ai_reads_${NAME}_sample10.jsonl --work-dir $OUT/images_v20b2000 2>&1 | grep -v "^INFO\|^DEBUG" | tail -3
cd $REPO && .venv/bin/python - <<PY
import json, statistics, sys
sys.path.insert(0, "$REPO")
from pathlib import Path
from src.datasets.evaluations.grounding_eval.box_quality import template_rate
RAW = Path("$OUT/raw"); jobs = [json.loads(l) for l in open("$OUT/jobs_v20b_sample10.jsonl")]
for key in ("qwen3-vl-8b-heb-v20a-step1800", "qwen3-vl-8b-heb-v20b-step1300", "$NAME"):
    tm = []
    for j in jobs:
        f = RAW / (j["doc_id"] + f"__{j['image_index']}__{key}.json")
        if f.exists():
            t = template_rate([l["box"] for l in json.load(open(f))["vlm_lines"]]); tm.append(t if t is not None else 0)
    print(f"  {key}: raw template rate mean {statistics.mean(tm):.2f} over {len(tm)} site pages, >=50%: {sum(t>=0.5 for t in tm)}")
PY
echo "===== full hard evals $(date) ====="
$H/hard_eval_ckpt.sh $STEP $SHA full v20b
echo "===== SIX-WAY GROUNDING TABLE $(date) ====="
cd $REPO && .venv/bin/python - <<PY
import json, statistics
from pathlib import Path
D = Path('src/datasets/evaluations/grounding_eval')
ARMS = {'v19a-1300 (frozen)': '', 'v19b-1300 (merger LoRA)': '_v19b_step1300', 'v19c-1800 (merger full)': '_v19c_step1800',
        'v20a-1500 (frozen+grounding)': '_v20a_step1500', 'v20a-1800 (frozen+grounding)': '_v20a_step1800',
        'v20b-2000 (merger LoRA+grounding)': '_v20b_step2000'}
print(f"{'arm':36s} {'locate hit':>10s} {'med IoU':>8s} {'IoU>=.5':>8s} | {'rb CER':>7s} | {'gr parse':>8s} {'ln match':>8s} {'ln IoU':>7s} {'ln CER':>7s}")
for arm, sfx in ARMS.items():
    L = lambda m: json.loads((D / f'grounding_eval_{m}{sfx}_results.json').read_text()) if (D / f'grounding_eval_{m}{sfx}_results.json').exists() else None
    loc, rb, gr = L('locate'), L('read_box'), L('grounded')
    if not all((loc, rb, gr)):
        print(f'{arm:36s} MISSING'); continue
    lok = [r for r in loc if r.get('ok')]; rok = [r for r in rb if r.get('ok')]; gok = [r for r in gr if r.get('ok')]
    print(f"{arm:36s} {sum(r['hit'] for r in lok):>4d}/{len(lok):<5d} {statistics.median(r['iou'] for r in lok):>8.3f} {sum(r['iou']>=0.5 for r in lok):>3d}/{len(lok):<4d} | "
          f"{statistics.median(r['cer'] for r in rok):>7.3f} | {len(gok):>3d}/24   {sum(r['matched'] for r in gok):>3d}/{sum(r['gt_n'] for r in gok):<4d} "
          f"{statistics.median(r['miou'] for r in gok):>7.3f} {statistics.median(r['mcer'] for r in gok if r.get('mcer') is not None):>7.3f}")
PY
echo "===== ALL DONE $(date) ====="
