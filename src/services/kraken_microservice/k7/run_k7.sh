#!/bin/bash
# Start the kraken 7 test container (kraken-k7, host :8003) with the chosen segmenter.
# usage: run_k7.sh default|blla2026|orli|<model path under /segmodels or /runs> [binarized|rgb]
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
docker rm -f kraken-k7 >/dev/null 2>&1 || true   # only ever our own test container
docker run -d --name kraken-k7 --restart no --cpus 6 --memory 8g --shm-size 2g -p 8003:8002 \
  -e KRAKEN_SEGMENTER="$SEG" -e KRAKEN_SEG_INPUT="$SEG_INPUT" -e KRAKEN_THREADS=6 \
  -v "$R/src/datasets/raw_data/cairo_genizah/custom_model_weights:/app/models:ro" \
  -v "$SEGDIR:/segmodels:ro" -v "$RUNS:/runs:ro" kraken-service:k7
for i in $(seq 60); do curl -sf -m 60 localhost:8003/health && echo && exit 0; sleep 2; done
docker logs --tail 30 kraken-k7; exit 1
