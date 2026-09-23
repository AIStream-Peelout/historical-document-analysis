#!/bin/bash
# Start the kraken 7 test container on host :8003 with the chosen segmenter.
# usage: run_k7.sh default|blla2026|orli
set -euo pipefail
R=/Users/isaac/Documents/GitHub/historical-document-analysis
SEGDIR=/Volumes/home/studio_offload/models/kraken_segmenters
case "$1" in
  default)  SEG=default ;;
  blla2026) SEG=/segmodels/blla_2026/blla.mlmodel ;;
  orli)     SEG=/segmodels/orli_base/orli_base.safetensors ;;
  *) echo "unknown segmenter $1"; exit 1 ;;
esac
docker rm -f kraken-k7 >/dev/null 2>&1 || true   # only ever our own test container
docker run -d --name kraken-k7 --restart no --cpus 6 --shm-size 2g -p 8003:8002 \
  -e KRAKEN_SEGMENTER="$SEG" -e KRAKEN_THREADS=6 \
  -v "$R/src/datasets/raw_data/cairo_genizah/custom_model_weights:/app/models:ro" \
  -v "$SEGDIR:/segmodels:ro" kraken-service:k7
for i in $(seq 60); do curl -sf localhost:8003/health && echo && exit 0; sleep 2; done
docker logs --tail 30 kraken-k7; exit 1
