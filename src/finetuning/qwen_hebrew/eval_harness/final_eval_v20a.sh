#!/bin/zsh
# v20a mid-run checkpoint through everything: stage latest hub checkpoint ->
# grounding trio -> layout-QA (v20a + v19a zero-shot baseline) -> full hard
# evals (religious-140 + PGP-131 + compare + W&B) -> cleanup -> 5-way
# grounding table -> eval-loss tail. Runs under nohup from the repo.
set -e
REPO=/Users/isaac/Documents/GitHub/historical-document-analysis
H=$REPO/src/finetuning/qwen_hebrew/eval_harness
G=$REPO/src/datasets/evaluations/grounding_eval/grounding_eval.py
Q=$H/layout_qa_eval.py
BASELINE=qwen3-vl-8b-heb-v19a-step1300
[ -d /Volumes/home/studio_offload ] && export HF_HOME=/Volumes/home/studio_offload/hf_home_merge

echo "===== v20a EVAL $(date) ====="
RES=$($REPO/.venv/bin/python - <<'PY'
import os, re
from dotenv import load_dotenv; load_dotenv('/Users/isaac/Documents/GitHub/historical-document-analysis/.env')
from huggingface_hub import HfApi
api = HfApi(token=os.environ['HF1_TOKEN'])
ck = next(c for c in api.list_repo_commits('isaacmg/qwen3-vl-8b-hebrew-v20a-ckpt') if 'checkpoint' in c.title)
step = int(re.search(r'step (\d+)', ck.title).group(1))
assert step >= 1300, f"expected >=1300, hub says {step}"
print(f"{step} {ck.commit_id}")
PY
)
STEP=$(echo $RES | awk '{print $1}'); SHA=$(echo $RES | awk '{print $2}')
NAME=qwen3-vl-8b-heb-v20a-step$STEP
echo "===== checkpoint: step $STEP sha $SHA $(date) ====="

$H/hard_eval_ckpt.sh $STEP $SHA stage v20a || {
  echo "!!! NAS-path staging failed (flaky SMB?) — retrying on the ramdisk path $(date)"
  rm -rf /Volumes/home/studio_offload/v19b_merge/$NAME-bf16 \
         /Volumes/home/studio_offload/v19b_merge/$NAME-mlx 2>/dev/null || true
  FORCE_LOCAL=1 $H/hard_eval_ckpt.sh $STEP $SHA stage v20a
}
for MODE in locate read_box grounded; do
  cd $REPO && .venv/bin/python $G --run $MODE --model $NAME
done
echo "===== layout-QA $(date) ====="
cd $REPO && .venv/bin/python $Q --model $NAME
~/.lmstudio/bin/lms unload $NAME 2>/dev/null || true
if [ ! -f $REPO/src/datasets/evaluations/grounding_eval/layout_qa_eval_${BASELINE}_results.json ]; then
  cd $REPO && .venv/bin/python $Q --model $BASELINE
  ~/.lmstudio/bin/lms unload $BASELINE 2>/dev/null || true
else
  echo "baseline layout-QA already scored — skipping $BASELINE"
fi

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
