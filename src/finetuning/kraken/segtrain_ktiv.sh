#!/bin/bash
# Genizah line-segmenter fine-tune: blla2026 -> KTIV PageXML (export_ktiv_pagexml.py), kraken 7.0.3, CPU.
#
# usage: segtrain_ktiv.sh stage|gate|baseline|train|resume <ckpt>
#   stage     rsync the NAS export (images + pagexml + manifests) to local disk (training reads it every epoch)
#   gate      seg_loader_gate.py through kraken's own data path; must exit 0 before training
#   baseline  ketos segtest of blla2026 on val.lst (the warm-start reference the first epochs must match)
#   train     ketos segtrain from blla2026 (container kraken-segtrain-ktiv-v1, detached, --restart no)
#   resume    continue from a Lightning checkpoint: segtrain_ktiv.sh resume /out/<ckpt>.ckpt
#
# Shared prod host (docs/shared_studio_runtime.md): 6 CPUs, 8 GiB (measured peak 6.7 GiB at the 2,600 px
# width cap, 1 loader worker), low cpu-shares so :8002 wins,
# threads pinned (torch otherwise sees all 16 host cores), outputs + logs on the NAS, nice 10.
set -euo pipefail
R=/Users/isaac/Documents/GitHub/historical-document-analysis
NAS=/Volumes/home/studio_offload/datasets/kraken_segmenter
SRC=$NAS/ktiv_pagexml_v1
STAGE=$HOME/kraken_segtrain_stage/ktiv_pagexml_v1
OUT=$NAS/runs/ktiv_seg_v1
SEGMODELS=/Volumes/home/studio_offload/models/kraken_segmenters
IMG=kraken-service:k7-base
NAME=kraken-segtrain-ktiv-v1
THREADS=6
COMMON=(--cpus $THREADS --memory 8g --cpu-shares 256 --shm-size 2g --log-opt max-size=50m --log-opt max-file=3
        -e OMP_NUM_THREADS=$THREADS -e PYTHONPATH=/code -w /data
        -v "$STAGE:/data:ro" -v "$OUT:/out" -v "$SEGMODELS:/segmodels:ro" -v "$R/src/finetuning/kraken:/code:ro")
KETOS=(nice -n 10 ketos --device cpu --threads $THREADS --workers 1 --seed 20260923)
TRAIN_ARGS=(segtrain -f page -t train.lst -e val.lst -i /segmodels/blla_2026/blla.mlmodel --resize fail
            --augment -q early --lag 8 --min-epochs 5 -N 60 -F 1 -r 1e-4 --schedule cosine --cos-max 60
            --cos-min-lr 1e-5 --line-width 8 -o /out/ktiv_seg)

case "${1:-}" in
  stage)
    mkdir -p "$STAGE"
    rsync -a --delete --exclude '.*' --include 'images/***' --include 'images_masked/***' --include 'pagexml/***' --include '*.lst' --include 'stats.json' \
      --include 'candidates_stats.json' --exclude '*' "$SRC/" "$STAGE/"
    echo "staged: $(ls "$STAGE/pagexml" | wc -l) xml, $(ls "$STAGE/images" | wc -l) images, $(du -sh "$STAGE" | cut -f1)"
    ;;
  gate)
    mkdir -p "$OUT/loader_gate"
    docker run --rm --name ${NAME}-gate "${COMMON[@]}" $IMG python /code/seg_loader_gate.py \
      --lst val.lst train.lst --items 300 --render 8 --out /out/loader_gate
    ;;
  baseline)
    # ketos 7.0.3 segtest crashes printing its pixel table (IndexError) after computing the metrics, and
    # does not log them; instead run segtrain's own validation on the unchanged model: 1 training page,
    # 1 step at lr 1e-12, full val.lst -> val_mean_iu / val_bl_* by the exact code path training reports.
    mkdir -p "$OUT"
    head -1 "$STAGE/train.lst" > "$OUT/baseline_train1.lst"
    docker run --rm --name ${NAME}-baseline "${COMMON[@]}" -v "$OUT/baseline_train1.lst:/baseline_train1.lst:ro" \
      $IMG "${KETOS[@]}" segtrain -f page -t /baseline_train1.lst -e val.lst -i /segmodels/blla_2026/blla.mlmodel \
      --resize fail -q fixed -N 1 -F 1 -r 1e-12 --line-width 8 -o /out/baseline_discard 2>&1 \
      | tr '\r' '\n' | tee "$OUT/baseline_blla2026_val.log" | grep -E "val_" | tail -2
    ;;
  train)
    mkdir -p "$OUT"
    [ -n "$(docker ps -aq -f name="^${NAME}$")" ] && { echo "container $NAME exists (resume or remove it)"; exit 1; }
    docker run -d --name $NAME --restart no "${COMMON[@]}" $IMG "${KETOS[@]}" "${TRAIN_ARGS[@]}"
    nohup sh -c "docker logs -f $NAME >> '$OUT/train.log' 2>&1" >/dev/null 2>&1 &
    echo "started $NAME; log persisted to $OUT/train.log"
    ;;
  resume)
    CKPT=${2:?checkpoint path inside the container, e.g. /out/ktiv_seg_12.ckpt}
    docker rm $NAME >/dev/null 2>&1 || true
    docker run -d --name $NAME --restart no "${COMMON[@]}" $IMG "${KETOS[@]}" "${TRAIN_ARGS[@]}" --resume "$CKPT"
    nohup sh -c "docker logs -f $NAME >> '$OUT/train.log' 2>&1" >/dev/null 2>&1 &
    echo "resumed $NAME from $CKPT; log appended to $OUT/train.log"
    ;;
  *) sed -n 2,11p "$0"; exit 1 ;;
esac
