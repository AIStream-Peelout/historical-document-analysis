#!/bin/zsh
# v19c TRUE FINAL (step 1800 — the resumed run died there): stage ->
# grounding trio -> full hard evals (religious-140 + PGP-131 + 3-way
# compare + W&B) -> cleanup -> loss tail. No polling: 1800 is known final.
set -e
REPO=/Users/isaac/Documents/GitHub/historical-document-analysis
SCRATCH=$REPO/src/finetuning/qwen_hebrew/eval_harness
G=$REPO/src/datasets/evaluations/grounding_eval/grounding_eval.py

echo "===== v19c FINAL EVAL (step 1800) $(date) ====="
RES=$($REPO/.venv/bin/python - <<'PY'
import os, re, sys
from dotenv import load_dotenv; load_dotenv('/Users/isaac/Documents/GitHub/historical-document-analysis/.env')
from huggingface_hub import HfApi
api = HfApi(token=os.environ['HF1_TOKEN'])
commits = api.list_repo_commits('isaacmg/qwen3-vl-8b-hebrew-v19c-ckpt')
ck = next(c for c in commits if 'checkpoint' in c.title)
step = int(re.search(r'step (\d+)', ck.title).group(1))
assert step == 1800, f"expected final step 1800, hub says {step}"
print(f"{step} {ck.commit_id}")
PY
)
STEP=$(echo $RES | awk '{print $1}'); SHA=$(echo $RES | awk '{print $2}')
NAME=qwen3-vl-8b-heb-v19c-step$STEP
echo "===== FINAL: step $STEP sha $SHA $(date) ====="

$SCRATCH/hard_eval_ckpt.sh $STEP $SHA stage v19c
for MODE in locate read_box grounded; do
  cd $REPO && .venv/bin/python $G --run $MODE --model $NAME
done
$SCRATCH/hard_eval_ckpt.sh $STEP $SHA full v19c

echo "===== eval-loss tail (trainer_state) ====="
$REPO/.venv/bin/python - <<PY
import os, json
from dotenv import load_dotenv; load_dotenv('$REPO/.env')
from huggingface_hub import hf_hub_download
p = hf_hub_download('isaacmg/qwen3-vl-8b-hebrew-v19c-ckpt', 'last-checkpoint/trainer_state.json',
                    token=os.environ['HF1_TOKEN'], revision='$SHA')
st = json.load(open(p))
for h in st['log_history']:
    if 'eval_loss' in h and h['step'] >= 1200:
        print(f"  step {h['step']:>4d}  eval_loss {h['eval_loss']:.4f}")
PY
echo "===== ALL DONE $(date) ====="
