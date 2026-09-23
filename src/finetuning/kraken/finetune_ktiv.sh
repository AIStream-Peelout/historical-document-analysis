#!/usr/bin/env bash
# Fine-tune the MiDRASH Kraken recogniser on exported KTIV lines.
#
# Runs `ketos compile` / `ketos train` inside a throw-away sibling container of
# the kraken-service image (same kraken 4.3.13 + torch as the :8002 service),
# CPU only and capped with --cpus/--memory so the shared Studio stays
# responsive.  Data and outputs live on the NAS; the base model is bind-mounted
# read-only from raw_data/cairo_genizah/custom_model_weights.
#
# Usage:  finetune_ktiv.sh compile|train|both
# Env:    DATA (export root, default the full KTIV export)  NAME (model stem)
#         CPUS MEM SHM (dataloader workers need /dev/shm) WORKERS EPOCHS
#         STAGE_DIR CHUNK (compile: local staging dir and lines per chunk) MIN_EPOCHS LAG LR BATCH WARMUP SCHEDULE
#         AUGMENT (--augment | --no-augment)  BASE (model path inside the container)
#         KEEP=1 keeps the finished container for inspection (docker logs / inspect)
set -euo pipefail
STAGE=${1:-both}
DATA=${DATA:-/Volumes/home/studio_offload/datasets/kraken_ktiv_lines}
NAME=${NAME:-ktiv_ft}
CPUS=${CPUS:-4}; MEM=${MEM:-12g}; SHM=${SHM:-4g}; WORKERS=${WORKERS:-$CPUS}
EPOCHS=${EPOCHS:-30}; MIN_EPOCHS=${MIN_EPOCHS:-8}; LAG=${LAG:-6}
LR=${LR:-1e-4}; BATCH=${BATCH:-32}; WARMUP=${WARMUP:-500}; SCHEDULE=${SCHEDULE:-constant}
AUGMENT=${AUGMENT:---no-augment}
BASE=${BASE:-/app/models/MiDRASH_Gen_01.mlmodel}
IMAGE=kraken-service:linewise
MODELS=/Users/isaac/Documents/GitHub/historical-document-analysis/src/datasets/raw_data/cairo_genizah/custom_model_weights
NAS_ROOT=$(dirname "$DATA"); DATA_DIR=$(basename "$DATA")
mkdir -p "$DATA/models"

run() {  # run <suffix> <cmd...> : one capped container, workdir = the export root
    local cname="kraken-train-${NAME}-$1"
    docker rm -f "$cname" >/dev/null 2>&1 || true
    docker run --name "$cname" --cpus "$CPUS" --memory "$MEM" --shm-size "$SHM" --log-opt max-size=50m --log-opt max-file=2 \
        -e OMP_NUM_THREADS="$WORKERS" -e PYTHONUNBUFFERED=1 \
        -v "$NAS_ROOT:/nas" -v "$MODELS:/app/models:ro" -w "/nas/$DATA_DIR" \
        "$IMAGE" "${@:2}" || true
    echo "$(date '+%F %T') $cname exit=$(docker inspect -f '{{.State.ExitCode}}' "$cname") oom_killed=$(docker inspect -f '{{.State.OOMKilled}}' "$cname")"
    [[ ${KEEP:-0} == 1 ]] || docker rm "$cname" >/dev/null
}

if [[ $STAGE == compile || $STAGE == both ]]; then
    # Chunked LOCAL staging: `ketos compile` with a worker pool over the NAS bind mount
    # (virtiofs -> SMB) dies with "Too many open files" after ~1k files, so each chunk of
    # line files is copied to local scratch, compiled there, and only the arrow goes to
    # the NAS.  Peak local disk use = one chunk (~350 MB).
    STAGE_DIR=${STAGE_DIR:-/private/tmp/kraken_stage}; CHUNK=${CHUNK:-4500}; COPY_STREAMS=${COPY_STREAMS:-4}
    mkdir -p "$DATA/arrow" "$STAGE_DIR"; rm -f "$DATA"/arrow/*.arrow "$STAGE_DIR"/*_part_*
    for split in train val; do
        (cd "$DATA" && split -l "$CHUNK" -a 2 "$split.txt" "$STAGE_DIR/${split}_part_")
        for part in "$STAGE_DIR"/${split}_part_*; do
            idx=${part##*_part_}
            rm -rf "$STAGE_DIR/data"; mkdir -p "$STAGE_DIR/data"
            { cat "$part"; sed 's/\.jpg$/.gt.txt/' "$part"; } > "$STAGE_DIR/files.txt"
            rm -f "$STAGE_DIR"/files_sub_*
            n_files=$(wc -l < "$STAGE_DIR/files.txt" | tr -d ' ')
            split -l $(( (n_files + COPY_STREAMS - 1) / COPY_STREAMS )) -a 1 "$STAGE_DIR/files.txt" "$STAGE_DIR/files_sub_"   # macOS split: no -n l/N
            for sub in "$STAGE_DIR"/files_sub_*; do   # parallel SMB reads: one stream tops out at ~11 files/s
                (cd "$DATA" && tar cf - -T "$sub") | tar xf - -C "$STAGE_DIR/data" &
            done; wait
            cp "$part" "$STAGE_DIR/data/list.txt"
            echo "$(date '+%F %T') compile $split part $idx ($(wc -l < "$part" | tr -d ' ') lines) -> arrow/${split}_part_${idx}.arrow"
            local_cname="kraken-train-${NAME}-compile-${split}-${idx}"
            docker rm -f "$local_cname" >/dev/null 2>&1 || true
            docker run --name "$local_cname" --cpus "$CPUS" --memory "$MEM" --log-opt max-size=50m --log-opt max-file=2 -e PYTHONUNBUFFERED=1 \
                -v "$STAGE_DIR/data:/data" -v "$NAS_ROOT:/nas" -w /data "$IMAGE" \
                ketos compile -f path --force-type baseline --workers "$WORKERS" -F list.txt \
                -o "/nas/$DATA_DIR/arrow/${split}_part_${idx}.arrow" 2>&1 | tee -a "$DATA/models/compile_$split.log" | grep -E "Output file|Error|Too many" || true
            rc=$(docker inspect -f '{{.State.ExitCode}}' "$local_cname"); docker rm "$local_cname" >/dev/null
            [[ $rc == 0 ]] || { echo "$(date '+%F %T') compile $split part $idx FAILED exit=$rc"; exit 1; }
        done
    done
    rm -rf "$STAGE_DIR/data" "$STAGE_DIR/files.txt" "$STAGE_DIR"/files_sub_* "$STAGE_DIR"/*_part_*
    (cd "$DATA" && ls arrow/train_part_*.arrow > train.manifest && ls arrow/val_part_*.arrow > val.manifest)
    echo "$(date '+%F %T') compiled: $(wc -l < "$DATA/train.manifest" | tr -d ' ') train arrows, $(wc -l < "$DATA/val.manifest" | tr -d ' ') val arrows"
fi
if [[ $STAGE == train || $STAGE == both ]]; then
    echo "$(date '+%F %T') train $NAME: base $BASE, cpus $CPUS, batch $BATCH, lr $LR, epochs $EPOCHS (min $MIN_EPOCHS, lag $LAG) $AUGMENT"
    [[ -s "$DATA/train.manifest" && -s "$DATA/val.manifest" ]] || { echo "no manifests — run the compile stage first"; exit 1; }   # -t/-e take path lists
    run train ketos -v train -f binary -i "$BASE" --resize union -t train.manifest -e val.manifest \
        -o "models/$NAME" -d cpu --workers "$WORKERS" -B "$BATCH" -r "$LR" -N "$EPOCHS" \
        --min-epochs "$MIN_EPOCHS" --lag "$LAG" --quit early --schedule "$SCHEDULE" --warmup "$WARMUP" \
        -u NFKD -n --base-dir R --reorder --pad 16 $AUGMENT --ignore-fixed-split \
        2>&1 | tee -a "$DATA/models/train_$NAME.log"
fi
