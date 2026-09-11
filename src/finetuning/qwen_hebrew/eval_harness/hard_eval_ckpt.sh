#!/bin/zsh
# One series checkpoint through hard evals: merge -> convert -> LM Studio ->
# (lite: flip-slice ~66 items | full: religious-140 + PGP-131 + 3-way compare)
# -> W&B log (run v19c_hard_evals) -> local cleanup (NAS masters kept).
# Lives in the repo (NOT /private/tmp — that dir is wiped on every reboot).
set -e
STEP=$1; REV=$2; MODE=${3:-lite}; VER=${4:-v19c}
[ -n "$STEP" ] && [ ${#REV} -eq 40 ] || { echo "usage: hard_eval_ckpt.sh <step> <sha> [lite|full|stage] [ver]"; exit 1; }
REPO=/Users/isaac/Documents/GitHub/historical-document-analysis
SCRATCH=$REPO/src/finetuning/qwen_hebrew/eval_harness
NAME=qwen3-vl-8b-heb-$VER-step$STEP
BF16_NAS=/Volumes/home/studio_offload/v19b_merge/$NAME-bf16
BF16_LOCAL=$SCRATCH/$NAME-bf16
MLX_NAS=/Volumes/home/studio_offload/v19b_merge/$NAME-mlx
MLX_LOCAL=$REPO/models/$NAME
LMS_DIR=~/.lmstudio/models/isaacmg/$NAME

FREE=$(df -g /System/Volumes/Data | tail -1 | awk '{print $4}')
echo "local free: ${FREE}Gi ($MODE eval, step $STEP)"
FREE_MIN=18; [ "$MODE" = "stage" ] && FREE_MIN=12
[ "$FREE" -ge $FREE_MIN ] || { echo "ABORT: need >=${FREE_MIN}Gi free"; exit 1; }

NAS_UP=0; [ -d /Volumes/home/studio_offload ] && [ -z "$FORCE_LOCAL" ] && NAS_UP=1
echo "NAS_UP=$NAS_UP (FORCE_LOCAL=${FORCE_LOCAL:-0})"

if [ -d "$MLX_LOCAL" ]; then
  echo "=== [1-2/4] $NAME already staged locally — skipping merge/convert $(date) ==="
elif [ "$NAS_UP" = "1" ]; then
  echo "=== [1/4] merge $NAME (NAS mode) $(date) ==="
  if [ ! -f "$BF16_NAS/model.safetensors.index.json" ]; then
    $REPO/.venv/bin/python $SCRATCH/merge_ckpt_generic.py \
      --repo isaacmg/qwen3-vl-8b-hebrew-$VER-ckpt --revision $REV --out $BF16_NAS
  fi
  echo "=== [2/4] MLX convert $(date) ==="
  if [ ! -d "$MLX_NAS" ]; then
    rm -rf "$BF16_LOCAL"; cp -R "$BF16_NAS" "$BF16_LOCAL"
    cd $REPO && .venv-mlx/bin/python -m mlx_vlm convert \
      --hf-path "$BF16_LOCAL" --mlx-path "$MLX_NAS" -q --q-bits 8
    rm -rf "$BF16_LOCAL"
  fi
  echo "=== [3/4] stage + LM Studio $(date) ==="
  if [ ! -d "$MLX_LOCAL" ]; then cp -R "$MLX_NAS" "$MLX_LOCAL"; fi
else
  echo "=== [1/4] merge $NAME (RAMDISK mode — NAS unavailable) $(date) ==="
  if [ ! -d "$MLX_LOCAL" ]; then
    RD_BLOCKS=75497472   # 36G: holds HF base cache (mmap-held) + merged bf16
    MEMFREE=$(memory_pressure -Q 2>/dev/null | grep -oE '[0-9]+' | tail -1)
    [ "${MEMFREE:-0}" -ge 50 ] || { echo "ABORT: only ${MEMFREE}% RAM free, need >=50%"; exit 1; }
    RD_DEV=$(hdiutil attach -nomount ram://$RD_BLOCKS | awk '{print $1}')
    trap "cd /; hdiutil detach $RD_DEV -force >/dev/null 2>&1 || true" EXIT
    diskutil eraseVolume HFS+ V19CRAM $RD_DEV >/dev/null
    RD=/Volumes/V19CRAM
    export HF_HOME=$RD/hf   # base cache in RAM: local disk never holds it
    $REPO/.venv/bin/python $SCRATCH/merge_ckpt_generic.py \
      --repo isaacmg/qwen3-vl-8b-hebrew-$VER-ckpt --revision $REV --out $RD/$NAME-bf16
    echo "=== [2/4] MLX convert (ramdisk read -> local write) $(date) ==="
    cd $REPO && .venv-mlx/bin/python -m mlx_vlm convert \
      --hf-path "$RD/$NAME-bf16" --mlx-path "$MLX_LOCAL" -q --q-bits 8
    hdiutil detach $RD_DEV -force >/dev/null 2>&1 || true
    trap - EXIT
  fi
  echo "=== [3/4] stage + LM Studio $(date) ==="
fi
$REPO/.venv/bin/python -c "
import json
cfg = json.load(open('$MLX_LOCAL/preprocessor_config.json'))
assert cfg['min_pixels'] == 6_500_000 and cfg['max_pixels'] == 7_000_000, cfg
print('preprocessor OK')"
rm -rf "$LMS_DIR"; mkdir -p ~/.lmstudio/models/isaacmg
cp -c -R "$MLX_LOCAL" "$LMS_DIR"

if [ "$MODE" = "stage" ]; then
  echo "=== staged only (no eval, no cleanup) $(date) ==="; exit 0
fi
echo "=== [4/4] $MODE eval $(date) ==="
if [ "$MODE" = "full" ]; then
  cd $REPO/src/datasets/evaluations
  PYTHONPATH=$REPO $REPO/.venv/bin/python helper_eval_scripts/run_religious_benchmark.py \
    --vlm-model $NAME
  $REPO/.venv/bin/python $SCRATCH/run_pgp131_v19b.py $NAME
  PYTHONPATH=$REPO $REPO/.venv/bin/python helper_eval_scripts/score_genizah_offline.py \
    --benchmark verified --no-wandb | sed -n '15,45p'
  cd $REPO && .venv/bin/python $SCRATCH/compare_series.py --ver $VER --step $STEP --wandb
else
  cd $REPO && .venv/bin/python $SCRATCH/lite_eval.py --step $STEP --model-name $NAME
fi

echo "=== cleanup $(date) ==="
~/.lmstudio/bin/lms unload $NAME 2>/dev/null || true
# Flagship / current-best checkpoints stay on local disk + in LM Studio
# (user rule 2026-09-05: manual testing). Edit KEEP_LOCAL when the flagship changes.
KEEP_LOCAL=(qwen3-vl-8b-heb-v19a-step1300 qwen3-vl-8b-heb-v20a-step1800)
if (( ${KEEP_LOCAL[(Ie)$NAME]} )); then
  echo "keeping $NAME on disk (flagship/current-best list)"
else
  rm -rf "$MLX_LOCAL" "$LMS_DIR"
fi
df -g /System/Volumes/Data | tail -1 | awk '{print $4" Gi free at end"}'
echo "=== DONE step $STEP ($MODE) $(date) ==="
