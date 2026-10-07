#!/bin/zsh
# Resume the v20a eval from the layout-QA step: the model is already merged,
# converted and staged locally (stage pass done), and the grounding trio
# already ran. Runs layout-QA (v20a + v19a baseline) -> full hard evals
# (short-circuits merge/convert) -> 5-way grounding table -> loss tail.
set -e
REPO=/Users/isaac/Documents/GitHub/historical-document-analysis
H=$REPO/src/finetuning/qwen_hebrew/eval_harness
Q=$H/layout_qa_eval.py
STEP=${1:-1500}; SHA=${2:-91e940671e6a9fb164319a9aabf3ae77d9fe8eaa}
NAME=qwen3-vl-8b-heb-v20a-step$STEP
BASELINE=qwen3-vl-8b-heb-v19a-step1300
unset HF_HOME   # datasets + hub cache local for this pass (NAS EBADF on locks)

echo "===== v20a RESUME (step $STEP) $(date) ====="
[ -d $REPO/models/$NAME ] || { echo "ABORT: $NAME not staged locally"; exit 1; }
[ -d ~/.lmstudio/models/isaacmg/$NAME ] || { mkdir -p ~/.lmstudio/models/isaacmg; cp -c -R $REPO/models/$NAME ~/.lmstudio/models/isaacmg/$NAME; }

echo "===== layout-QA $(date) ====="
cd $REPO && .venv/bin/python $Q --model $NAME
~/.lmstudio/bin/lms unload $NAME 2>/dev/null || true
cd $REPO && .venv/bin/python $Q --model $BASELINE
~/.lmstudio/bin/lms unload $BASELINE 2>/dev/null || true

$H/hard_eval_ckpt.sh $STEP $SHA full v20a

echo "===== FIVE-WAY GROUNDING TABLE $(date) ====="
cd $REPO && .venv/bin/python - <<PY
import json, statistics
from pathlib import Path
D = Path('src/datasets/evaluations/grounding_eval')
ARMS = {'v19a-1300 (frozen)': '', 'v19b-1300 (merger LoRA)': '_v19b_step1300',
        'v19c-1200 (merger full)': '_v19c_step1200', 'v19c-1800 (merger full)': '_v19c_step1800',
        'v20a-$STEP (frozen+grounding data)': '_v20a_step$STEP'}
print(f"{'arm':36s} {'locate hit':>10s} {'med IoU':>8s} {'IoU>=.5':>8s} | {'rb CER':>7s} | "
      f"{'gr parse':>8s} {'ln match':>8s} {'ln IoU':>7s} {'ln CER':>7s}")
for arm, sfx in ARMS.items():
    L = lambda m: json.loads((D / f'grounding_eval_{m}{sfx}_results.json').read_text()) \
        if (D / f'grounding_eval_{m}{sfx}_results.json').exists() else None
    loc, rb, gr = L('locate'), L('read_box'), L('grounded')
    if not all((loc, rb, gr)):
        print(f'{arm:36s} MISSING'); continue
    lok = [r for r in loc if r.get('ok')]; rok = [r for r in rb if r.get('ok')]; gok = [r for r in gr if r.get('ok')]
    print(f"{arm:36s} {sum(r['hit'] for r in lok):>4d}/{len(lok):<5d} "
          f"{statistics.median(r['iou'] for r in lok):>8.3f} {sum(r['iou']>=0.5 for r in lok):>3d}/{len(lok):<4d} | "
          f"{statistics.median(r['cer'] for r in rok):>7.3f} | {len(gok):>3d}/24   "
          f"{sum(r['matched'] for r in gok):>3d}/{sum(r['gt_n'] for r in gok):<4d} "
          f"{statistics.median(r['miou'] for r in gok):>7.3f} "
          f"{statistics.median(r['mcer'] for r in gok if r.get('mcer') is not None):>7.3f}")
PY

echo "===== eval-loss tail (trainer_state @ $SHA) ====="
$REPO/.venv/bin/python - <<PY
import os, json
from dotenv import load_dotenv; load_dotenv('$REPO/.env')
from huggingface_hub import hf_hub_download
p = hf_hub_download('isaacmg/qwen3-vl-8b-hebrew-v20a-ckpt', 'last-checkpoint/trainer_state.json',
                    token=os.environ['HF1_TOKEN'], revision='$SHA')
for h in json.load(open(p))['log_history']:
    if 'eval_loss' in h and h['step'] >= 600:
        print(f"  step {h['step']:>4d}  eval_loss {h['eval_loss']:.4f}")
PY
echo "===== ALL DONE $(date) ====="
