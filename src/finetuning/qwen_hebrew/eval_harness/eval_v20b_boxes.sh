#!/bin/zsh
# v20b box check: waits for the staged MLX model and for the pilot chain to release LM Studio,
# then decodes the grounding trio (preds only) + 10 site pages, and prints box-quality metrics.
REPO=/Users/isaac/Documents/GitHub/historical-document-analysis
H=$REPO/src/finetuning/qwen_hebrew/eval_harness
G=$REPO/src/datasets/evaluations/grounding_eval
OUT=$REPO/src/datasets/raw_data/cairo_genizah/ai_reads
M=qwen3-vl-8b-heb-v20b-step1300
cd $REPO
for i in $(seq 1 360); do [ -d ~/.lmstudio/models/isaacmg/$M ] && ! pgrep -f "hard_eval_ckpt.sh 1300" >/dev/null && break; sleep 60; done
[ -d ~/.lmstudio/models/isaacmg/$M ] || { echo "ABORT: $M never staged"; exit 1; }
echo "===== staged; waiting for the pilot chain to finish $(date '+%F %T') ====="
while pgrep -f "chain_v2[.]sh" >/dev/null; do sleep 120; done
echo "===== v20b decodes $(date '+%F %T') ====="
for MODE in grounded locate read_box; do
  .venv/bin/python $G/grounding_eval.py --run $MODE --model $M --no-results 2>&1 | grep -v "^INFO\|^DEBUG"
done
echo "===== 10 site pages with v20b $(date '+%F %T') ====="
.venv/bin/python -m src.datasets.consensus.two_reader_lines --ids $OUT/jobs_v20b_sample10.jsonl --vlm-model $M \
  --out $OUT/ai_reads_${M}_sample10.jsonl --work-dir $OUT/images_v20b 2>&1 | grep -v "^INFO\|^DEBUG"
echo "===== box quality $(date '+%F %T') ====="
.venv/bin/python $G/box_quality.py --preds $M qwen3-vl-8b-heb-v20a-step1800 qwen3-vl-8b-heb-v19a-step1300 \
  --records $OUT/ai_reads_${M}_sample10.jsonl 2>&1 | grep -v "^INFO\|^DEBUG"
echo "===== DONE $(date '+%F %T') ====="
