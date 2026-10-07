#!/bin/bash
# Start the kraken 7 test container (kraken-k7, host :8003) with the chosen segmenter.
# usage: [PORT=8003] [KRAKEN_MULTISCALE=1] [KRAKEN_ORIENT=1] run_k7.sh default|blla2026|orli|<model path> [binarized|rgb]
#   PORT 8003 -> container kraken-k7, any other test port (8004) -> kraken-k7-<port>; never 8002 (prod)
#   /segmodels = NAS models/kraken_segmenters, /runs = NAS datasets/kraken_segmenter/runs (fine-tune checkpoints)
set -euo pipefail
R=/Users/isaac/Documents/GitHub/historical-document-analysis
SEGDIR=/Volumes/home/studio_offload/models/kraken_segmenters
RUNS=/Volumes/home/studio_offload/datasets/kraken_segmenter/runs
case "$1" in
  default)  SEG=default ;;
  blla2026) SEG=/segmodels/blla_2026/blla.mlmodel ;;
  orli)     SEG=/segmodels/orli_base/orli_base.safetensors ;;
  /segmodels/*|/runs/*) SEG=$1 ;;
  *) echo "unknown segmenter $1"; exit 1 ;;
esac
SEG_INPUT=${2:-binarized}
PORT=${PORT:-8003}
[ "$PORT" = 8002 ] && { echo "refusing :8002 (production)"; exit 1; }
NAME=kraken-k7; [ "$PORT" = 8003 ] || NAME=kraken-k7-$PORT
CPUS=${CPUS:-6}
docker rm -f $NAME >/dev/null 2>&1 || true   # only ever our own test container
docker run -d --name $NAME --restart no --cpus $CPUS --memory 8g --oom-score-adj 1000 --shm-size 2g -p $PORT:8002 \
  -e KRAKEN_SEGMENTER="$SEG" -e KRAKEN_SEG_INPUT="$SEG_INPUT" -e KRAKEN_THREADS=$CPUS \
  -e KRAKEN_MULTISCALE=${KRAKEN_MULTISCALE:-0} -e KRAKEN_ORIENT=${KRAKEN_ORIENT:-0} \
  -v "$R/src/datasets/raw_data/cairo_genizah/custom_model_weights:/app/models:ro" \
  -v "$SEGDIR:/segmodels:ro" -v "$RUNS:/runs:ro" ${IMAGE:-kraken-service:k7}
for i in $(seq 60); do curl -sf -m 60 localhost:$PORT/health && echo && exit 0; sleep 2; done
docker logs --tail 30 $NAME; exit 1
