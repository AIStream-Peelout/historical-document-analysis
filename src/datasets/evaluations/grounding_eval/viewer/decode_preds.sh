#!/bin/zsh
# Re-decode the grounding eval for the viewer: every mode for every model
# given on the command line (LM Studio keys), dumping per-query predictions to
# grounding_eval/preds/ WITHOUT touching the canonical *_results.json scores.
# Usage: viewer/decode_preds.sh qwen3-vl-8b-heb-v20a-step1800 [more keys...]
set -e
REPO=/Users/isaac/Documents/GitHub/historical-document-analysis
G=$REPO/src/datasets/evaluations/grounding_eval/grounding_eval.py
[ $# -ge 1 ] || { echo "usage: $0 MODEL_KEY [MODEL_KEY ...]"; exit 2 }
for MODEL in "$@"; do
  for MODE in locate read_box grounded; do
    echo "===== $MODEL / $MODE  $(date '+%F %T') ====="
    cd $REPO && .venv/bin/python $G --run $MODE --model $MODEL --no-results
  done
done
echo "===== DONE $(date '+%F %T') ====="
