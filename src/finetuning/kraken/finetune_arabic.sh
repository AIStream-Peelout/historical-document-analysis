#!/usr/bin/env bash
# File name: finetune_arabic.sh
# Date: 10/19/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
#
# Fine-tune a Kraken 7.1.1 recogniser for Arabic handwriting from PP-OCRv6 medium (Apache-2.0).
#
# Usage:  [ENGINE=docker|native] finetune_arabic.sh compile|gate|baseline|train|test   (settings via env, below)
#
# ENGINE=docker (default): every stage runs in ONE throw-away capped container of kraken-train:k711 (Dockerfile.k711
# in this directory) named kraken-train-arabic-<stage>.  The script never touches any other container, and it refuses
# to start while any kraken-train-arabic-* container exists (one capped container at a time on the shared
# Studio).  Mounts: $DATA -> /data (workdir) and, read-only, at its own host path; $AR_DIR (the PP-OCRv6 model dir) ->
# /ar (read-only); this directory -> /work (read-only, for gate_lines.py and ketos_frozen_bn.py); every EXTRA_RO dir
# read-only at its own host path (build_arabic_lines.py writes absolute host paths into the lists and into PAGE
# imageFilename, so those resolve unchanged).  DATA and EXTRA_RO are made absolute (logical paths, so they
# match the unresolved /Volumes paths in the lists).  Container output is appended to
# $DATA/models/<stage>_$NAME.log.
#
# ENGINE=native: baseline, train and test run on the host through the kraken 7.1.1 venv ($VENV_PYTHON) on the Apple
# GPU (-d $DEVICE, default mps; about 30x faster than the CPU container), from the repo root, in their own process
# group; compile and gate are refused (docker only).  The manifests compile writes list container paths
# (/data/arrow/...), so a native stage writes host-path copies to $DATA/native/ (/data -> $DATA, /ar -> $AR_DIR; every
# arrow must exist) and passes BASE, MODEL, RESUME and the -o directory translated the same way; compile's own output
# is unchanged.  Same $DATA/models/$NAME, stage logs and results.tsv as docker.  There is no --memory cap (MEM, CPUS,
# SHM, IMAGE and EXTRA_RO are unused; THREADS still sets --threads), but one native run at a time: the running
# command's pid, which is also its process group, is in $DATA/models/native_<stage>_$NAME.pid, and a live pid file
# refuses every native stage and a docker compile (which rewrites the arrows); a native stage is refused while a
# kraken-train-arabic-compile container exists.  The command inherits the caller's environment (e.g.
# PYTORCH_ENABLE_MPS_FALLBACK=1) plus PYTHONUNBUFFERED=1, OMP_NUM_THREADS=$THREADS, KETOS_FROZEN_BN=$FROZEN_BN and
# TORCH_COMPILE_DISABLE=1 (unless COMPILE=1).
#
# LM Studio guard (native baseline/train/test): the GPU is shared with production LM Studio.
# A background loop runs `$LMS ps` every GUARD_POLL s; as soon as any loaded model's STATUS is not IDLE (or lms ps
# fails or hangs for 20 s: fail closed) it sends SIGSTOP to the ketos process group, and once every model has been
# IDLE for GUARD_IDLE consecutive seconds it sends SIGCONT.  Every pause and resume is logged with a timestamp to the
# stage log (plus a "still paused" line every 10 min); the guard ends when ketos exits.  A native stage refuses to
# start when lms ps cannot be read.  NO_GUARD=1 skips the guard (a WARNING on stderr and in the stage log: the run
# then does NOT yield the GPU to production).  INT, TERM and HUP to the driver are forwarded to the process group as
# SIGTERM followed by SIGCONT (so a paused run can act on it): stop a native run with Ctrl-C, kill <driver pid> or
# screen -X quit, never with kill -9 on the driver alone.
#
# BatchNorm (FROZEN_BN=1, the default): ketos train puts the whole network into train mode, so every BatchNorm
# re-estimates its running mean/variance from the new small batches; fine-tuning the PP-OCRv6 base that way collapsed
# validation accuracy from 90 % to 9 % (weights moved < 0.3 %, 103 of 122 BatchNorm running statistics by up to 76 %).
# With FROZEN_BN=1 the train stage runs ketos_frozen_bn.py (this directory; it imports only torch and kraken), a
# drop-in for the ketos executable that pins every BatchNorm to its running statistics (affine weights still train):
# docker `python /work/ketos_frozen_bn.py <ketos args>`, native `cd <repo root> && $VENV_PYTHON -m
# src.finetuning.kraken.ketos_frozen_bn <ketos args>`.  FROZEN_BN=0 runs plain ketos (A/B runs only).  baseline and
# test always run plain ketos (ketos test evaluates in eval mode).
#
# Stages:
#   compile   (docker only) ketos compile $DATA/lists/<set>_<split>.lst (split = train, val) for every set in $SETS into
#             $DATA/arrow/<set>_<split>.arrow.  PAGE sets: -f page --linetype baselines (TextLines without a
#             <Baseline> are dropped silently; imageFilename resolves relative to the XML).  Crop sets
#             ($CROP_SETS; foo.png + foo.gt.txt, stems without other dots): -f path --force-type baseline, so
#             they can share a run with the PAGE arrows.  List entries must resolve INSIDE the container:
#             /data/..., or absolute host paths under $DATA or an EXTRA_RO dir.  A missing list aborts before
#             anything runs; an empty one is skipped.  COMPILE_WORKERS > 1 is refused for a list whose first
#             entry (or its page image) is on the NAS (/Volumes): ketos compile --workers 5 over this bind
#             mount died with 'OSError: Too many open files' after ~1k files in the KTIV fine-tune.  Prints
#             kraken's line counts, then gate_lines.py count (rows and non-Arabic symbols; exit 3 = mixed
#             baseline/bbox arrows, exit 4 = a symbol the TEXT RULES forbid), and only when that passes writes
#             $DATA/train.manifest and $DATA/val.manifest (container paths /data/arrow/...).
#   gate      (docker only) gate_lines.py sample: $GATE_N seeded-random compiled lines per train arrow into
#             $DATA/gate/<arrow stem>/ (row_*.png, index.tsv, index.html).  LOOK at them before training:
#             train refuses to start until every train arrow has a gate dump newer than the arrow.
#   baseline  ketos test of $BASE on $EVAL_MANIFEST (Character Accuracy before fine-tuning).
#   train     ketos train from $BASE into $DATA/models/$NAME (a NAME already used is refused, except by RESUME).
#             QUIT=fixed (default): exactly EPOCHS epochs; ModelCheckpoint still keeps the best validation
#             checkpoint and writes best_<score>.safetensors, so early stopping would only save compute.
#             QUIT=early: ketos then IGNORES -N (it sets max_epochs=-1), so EPOCHS is no cap and the run ends
#             only if the trainer registers EarlyStopping.  A check (gate_lines.py early-check, log
#             train-check_$NAME.log; in the image under docker, with the venv's kraken under native) first prints the
#             stop/schedule lines of kraken and the train is refused when train/ppocr.py instantiates no EarlyStopping,
#             or when it reads trainer.estimated_stepping_batches (Lightning returns inf for max_epochs=-1, so a
#             step-based LR schedule never anneals or fails at fit start) unless EARLY_LR_OK=1 after reading those
#             lines.  The settings line (engine=, device=, frozen_bn=, ...) is appended to $DATA/models/train_$NAME.log.
#   test      ketos test of $MODEL on $EVAL_MANIFEST; MODEL defaults to the best_<score>.safetensors of $NAME.
#             baseline and test append "date stage model manifest char_acc word_acc engine frozen_bn" to
#             $DATA/models/results.tsv: engine = ENGINE:DEVICE; frozen_bn = the value the model's run recorded in
#             train_<run>.log ('-' for baseline or a model outside $DATA/models/<run>/, '?' for a run trained before
#             FROZEN_BN existed); rows written before these two columns existed have six fields.
#
# Env (default):
#   ENGINE (docker | native)  DEVICE (cpu under docker, the only choice there; mps under native, or cpu)
#   VENV_PYTHON (~/.venvs/kraken711/bin/python; native plain ketos = the ketos script beside it)  FROZEN_BN (1)
#   NO_GUARD (0)  GUARD_POLL (10)  GUARD_IDLE (30)  LMS (~/.lmstudio/bin/lms)
#   DATA (/Volumes/home/studio_offload/datasets/kraken_arabic_lines/v1)  SETS ("agapet rasam iskandar baybars")
#   CROP_SETS (baybars)  NAME (ar_ppocrv6m_v1)  BASE (/ar/ppocrv6_medium.safetensors)  MODEL (best of NAME)
#   AR_DIR (/Volumes/home/studio_offload/models/kraken_arabic: the dir behind /ar)
#   CPUS (4)  MEM (7g; docker only: refused above 7g and below 512m, 0 = unlimited in docker)  SHM (2g)  WORKERS (0)
#   THREADS (CPUS)  BATCH (2)  LR (1e-4)  QUIT (fixed | early)  EPOCHS (30: the run length under QUIT=fixed,
#   and --cos-max)  MIN_EPOCHS (5) and LAG (5) (QUIT=early only)  EARLY_LR_OK (0)  SCHEDULE (cosine)
#   WARMUP (200)  RESUME (unset; /data/models/$NAME/checkpoint_<NN>-<score>.ckpt continues that run)
#   AUGMENT (--augment | --no-augment)  COMPILE (0 sets TORCH_COMPILE_DISABLE=1; 1 lets PP-OCRv6 torch.compile)
#   IMAGE (kraken-train:k711)  TEST_BATCH (8)  EVAL_MANIFEST (val.manifest)  GATE_N (30)  GATE_SEED (0)
#   GATE_MANIFESTS ("train")  COMPILE_WORKERS (unset = kraken's default, single process)  SKIP_GATE_CHECK (0)
#   EXTRA_RO (/Volumes/home/studio_offload/datasets/arabic_external; space-separated host dirs without spaces)
#
# Memory (docker): the Docker VM has 16 GiB and production Kraken holds up to 10 GiB, so a training container stays at
# --memory 7g with --oom-score-adj 1000 (batch 8 was OOM-killed at 8 GiB; batch 2 fits in ~1.5 GiB).  Exit
# 137 is SIGKILL, i.e. almost always the memory cap.  Ketos global options (-d/--threads/--workers/-v) go
# BEFORE the subcommand (also after the wrapper).  An interrupted train leaves no best_*.safetensors: continue it with
# RESUME=/data/models/$NAME/checkpoint_<NN>-<score>.ckpt (same NAME and settings; ketos train --resume
# replaces -i $BASE --resize union, and the NAME-in-use refusal and the gate check are skipped), or convert
# the best checkpoint_*.ckpt by hand (ketos convert -o best.safetensors <ckpt>) and pass it as MODEL.  --resume
# restores the data configuration stored in the checkpoint (ketos then ignores -t/-e): a run started under
# ENGINE=docker stored /data/... manifest paths and resumes under docker only; a run started natively stored
# $DATA/native/... host paths, which resolve under both engines.
# LR schedule: kraken 7.0's VGSL trainer steps the cosine scheduler once per epoch (cos_t_max in epochs);
# the PP-OCRv6 trainer bakes warmup into its own LambdaLR and reads trainer.estimated_stepping_batches (the
# smoke log prints "Loading `train_dataloader` to estimate number of stepping batches"), so whether it honours
# --schedule/--cos-max, and in which unit, is unverified.  Check before tuning SCHEDULE or allowing QUIT=early:
#   docker run --rm kraken-train:k711 sh -c 'cd /usr/local/lib/python3.12/site-packages/kraken && grep -n -E "max_epochs|EarlyStopping|configure_callbacks|estimated_stepping_batches|cos_t_max" ketos/recognition.py train/ppocr.py'
set -euo pipefail

STAGE=${1:-}
ENGINE=${ENGINE:-docker}; DEVICE=${DEVICE:-}
VENV_PYTHON=${VENV_PYTHON:-$HOME/.venvs/kraken711/bin/python}
FROZEN_BN=${FROZEN_BN:-1}; NO_GUARD=${NO_GUARD:-0}; GUARD_POLL=${GUARD_POLL:-10}; GUARD_IDLE=${GUARD_IDLE:-30}
LMS=${LMS:-$HOME/.lmstudio/bin/lms}
DATA=${DATA:-/Volumes/home/studio_offload/datasets/kraken_arabic_lines/v1}
SETS=${SETS:-agapet rasam iskandar baybars}
CROP_SETS=${CROP_SETS:-baybars}
NAME=${NAME:-ar_ppocrv6m_v1}
AR_DIR=${AR_DIR:-/Volumes/home/studio_offload/models/kraken_arabic}
BASE=${BASE:-/ar/ppocrv6_medium.safetensors}
MODEL=${MODEL:-}
RESUME=${RESUME:-}
CPUS=${CPUS:-4}; MEM=${MEM:-7g}; SHM=${SHM:-2g}; WORKERS=${WORKERS:-0}; THREADS=${THREADS:-$CPUS}
BATCH=${BATCH:-2}; LR=${LR:-1e-4}; QUIT=${QUIT:-fixed}; EPOCHS=${EPOCHS:-30}; MIN_EPOCHS=${MIN_EPOCHS:-5}
LAG=${LAG:-5}; EARLY_LR_OK=${EARLY_LR_OK:-0}
SCHEDULE=${SCHEDULE:-cosine}; WARMUP=${WARMUP:-200}; AUGMENT=${AUGMENT:---augment}; COMPILE=${COMPILE:-0}
IMAGE=${IMAGE:-kraken-train:k711}
TEST_BATCH=${TEST_BATCH:-8}; EVAL_MANIFEST=${EVAL_MANIFEST:-val.manifest}
GATE_N=${GATE_N:-30}; GATE_SEED=${GATE_SEED:-0}; GATE_MANIFESTS=${GATE_MANIFESTS:-train}
COMPILE_WORKERS=${COMPILE_WORKERS:-}; SKIP_GATE_CHECK=${SKIP_GATE_CHECK:-0}
EXTRA_RO=${EXTRA_RO-/Volumes/home/studio_offload/datasets/arabic_external}   # EXTRA_RO= (empty) mounts none
MEM_CAP_BYTES=$((7 * 1024 * 1024 * 1024))
MEM_MIN_BYTES=$((512 * 1024 * 1024))
PREFIX=kraken-train-arabic
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
VENV_KETOS=${VENV_PYTHON%/*}/ketos
# perl one-liner that starts every native command: become a process-group leader, hand the pid (= the group id) to
# the driver through the file named by the first argument (written atomically), then exec the command (same pid).
SETPGRP_EXEC='setpgrp(0, 0) or die "setpgrp: $!\n"; my $f = shift @ARGV; open(my $h, ">", "$f.tmp") or die "$f.tmp: $!\n"; print $h "$$\n"; close($h) or die "$f.tmp: $!\n"; rename("$f.tmp", $f) or die "$f: $!\n"; exec { $ARGV[0] } @ARGV or die "exec $ARGV[0]: $!\n";'
# perl one-liner for `lms ps`: run the command (first argument = timeout in s) in its own process group and kill that
# whole group on timeout (exit 142), so no stray child keeps the output pipe open.
TIMEOUT_EXEC='my $t = shift @ARGV; my $pid = fork; defined $pid or die "fork: $!\n"; if (!$pid) { setpgrp(0, 0); exec { $ARGV[0] } @ARGV; die "exec $ARGV[0]: $!\n" } $SIG{ALRM} = sub { kill "KILL", -$pid; exit 142 }; alarm $t; waitpid($pid, 0); exit($? & 127 ? 128 + ($? & 127) : $? >> 8);'
NATIVE_PGID=""; NATIVE_GUARD_PID=""; NATIVE_ABORTED=0; NATIVE_PIDFILE=""
# Bash 3.2 (macOS /bin/bash) safe: no mapfile, no ${x,,}, never expand an empty array under set -u.

die() { echo "finetune_arabic.sh: $*" >&2; exit 1; }
ts() { date '+%F %T'; }

check_mem() {  # refuse MEM outside 512m..7g, or a value that cannot be parsed as docker's <int>[b|k|m|g]
    local v num mult=1 re='^([0-9]{1,12})([bkmg]?)i?b?$'
    v=$(printf '%s' "$MEM" | tr '[:upper:]' '[:lower:]')
    if ! [[ $v =~ $re ]]; then die "MEM=$MEM refused: give an integer with an optional b/k/m/g unit (e.g. 7g or 6144m)"; fi
    num=$((10#${BASH_REMATCH[1]}))
    case ${BASH_REMATCH[2]} in k) mult=1024 ;; m) mult=1048576 ;; g) mult=1073741824 ;; esac
    if (( num > MEM_CAP_BYTES / mult )); then
        die "MEM=$MEM refused: a training container on this host must stay at or below 7g (the Docker VM has 16 GiB and production Kraken holds up to 10 GiB); lower BATCH instead"
    fi
    if (( num * mult < MEM_MIN_BYTES )); then  # no overflow: num * mult <= 7 GiB after the cap check
        die "MEM=$MEM refused: 0 means unlimited in docker; give 512m..7g"
    fi
}

absolute_dir() {  # absolute_dir <dir>: its absolute logical path (symlinks kept, like the paths in the lists)
    (CDPATH='' cd -- "$1" && pwd)
}

check_knobs() {  # validate stage-independent settings before anything starts; make DATA/EXTRA_RO absolute
    local v ro="" re_uint='^[0-9]+$' re_cpus='^[0-9]+(\.[0-9]+)?$' re_name='^[A-Za-z0-9._-]+$' re_set='^[a-z0-9_]+$'
    if [[ $ENGINE == docker ]]; then check_mem; fi  # native has no --memory cap
    for v in WORKERS THREADS BATCH EPOCHS MIN_EPOCHS LAG WARMUP TEST_BATCH GATE_N GATE_SEED GUARD_POLL GUARD_IDLE; do
        if ! [[ ${!v} =~ $re_uint ]]; then die "$v=${!v} must be a non-negative integer"; fi
    done
    if (( 10#$GUARD_POLL < 1 )); then die "GUARD_POLL must be at least 1 (seconds between lms ps polls)"; fi
    if [[ -n $COMPILE_WORKERS ]] && ! [[ $COMPILE_WORKERS =~ $re_uint ]]; then die "COMPILE_WORKERS=$COMPILE_WORKERS must be an integer"; fi
    if ! [[ $CPUS =~ $re_cpus ]]; then die "CPUS=$CPUS must be a number"; fi
    if ! [[ $NAME =~ $re_name ]]; then die "NAME=$NAME: use letters, digits, '.', '_' or '-'"; fi
    for v in $SETS $CROP_SETS; do
        if ! [[ $v =~ $re_set ]]; then die "set name '$v' must match $re_set"; fi
    done
    for v in $GATE_MANIFESTS; do
        case $v in train|val) ;; *) die "GATE_MANIFESTS entries must be train or val, not '$v'" ;; esac
    done
    case $AUGMENT in --augment|--no-augment) ;; *) die "AUGMENT must be --augment or --no-augment" ;; esac
    case $COMPILE in 0|1) ;; *) die "COMPILE must be 0 or 1" ;; esac
    case $QUIT in fixed|early) ;; *) die "QUIT must be fixed or early" ;; esac
    case $EARLY_LR_OK in 0|1) ;; *) die "EARLY_LR_OK must be 0 or 1" ;; esac
    case $FROZEN_BN in 0|1) ;; *) die "FROZEN_BN must be 0 or 1" ;; esac
    case $NO_GUARD in 0|1) ;; *) die "NO_GUARD must be 0 or 1" ;; esac
    case $ENGINE:$DEVICE in
        docker:cpu|native:mps|native:cpu) ;;
        docker:*) die "DEVICE=$DEVICE: the $IMAGE image is CPU-only (DEVICE=cpu, or ENGINE=native for mps)" ;;
        *) die "DEVICE=$DEVICE: use mps or cpu" ;;
    esac
    if [[ -n $RESUME ]]; then
        case $RESUME in
            */../*) die "RESUME=$RESUME: no '..' in the path" ;;
            "/data/models/$NAME/"*.ckpt) ;;
            *) die "RESUME=$RESUME must be a checkpoint of this run: /data/models/$NAME/<file>.ckpt (set NAME to the interrupted run)" ;;
        esac
    fi
    if [[ -z ${SETS// /} ]]; then die "SETS is empty"; fi
    if [[ ! -d $DATA ]]; then die "DATA=$DATA is not a directory (NAS mounted? dataset built?)"; fi
    DATA=$(absolute_dir "$DATA")  # a trailing slash or a relative DATA would break visible() and the -v mounts
    if [[ ! -d $AR_DIR ]]; then die "$AR_DIR is not a directory (NAS mounted?)"; fi
    for v in $EXTRA_RO; do
        if [[ ! -d $v ]]; then die "EXTRA_RO entry $v is not a directory"; fi
        ro+="${ro:+ }$(absolute_dir "$v")"
    done
    EXTRA_RO=$ro
}

host_path() {  # host_path <container path>: print the host path behind /data or /ar ('' for other mounts)
    case $1 in
        /data/*) printf '%s\n' "$DATA/${1#/data/}" ;;
        /ar/*) printf '%s\n' "$AR_DIR/${1#/ar/}" ;;
        *) printf '\n' ;;
    esac
}

engine_path() {  # engine_path <container path>: as given under docker; under native the host path behind /data or /ar
    if [[ $ENGINE == native ]]; then
        case $1 in /data/*|/ar/*) host_path "$1"; return 0 ;; esac
    fi
    printf '%s\n' "$1"
}

require_file() {  # require_file <path>: die when it sits on a known mount (or, native, anywhere) and is missing there
    local h
    h=$(host_path "$1")
    if [[ -z $h && $ENGINE == native ]]; then h=$1; fi
    if [[ -n $h && ! -f $h ]]; then die "$1 not found (host path $h)"; fi
}

check_native_idle() {  # refuse while a native command of this driver runs under $DATA (a pid file with a live pid)
    local f pid
    for f in "$DATA"/models/native_*.pid; do
        if [[ ! -f $f ]]; then continue; fi
        pid=$(head -n 1 "$f" | tr -dc '0-9')
        if [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null; then
            die "a native run is active: $f (pid $pid: $(ps -o command= -p "$pid" 2>/dev/null | cut -c 1-150)); one native ketos at a time - wait for it (delete $f only if that pid is not this driver's)"
        fi
        echo "$(ts) removing stale $f (pid ${pid:-?} is gone)"
        rm -f "$f"
    done
}

preflight_docker() {  # image present, and no other kraken-train-arabic-* container (only names are read)
    local busy
    if ! docker version --format '{{.Server.Version}}' >/dev/null 2>&1; then die "docker daemon not reachable"; fi
    if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
        die "image $IMAGE not found; build it: docker build -f src/finetuning/kraken/Dockerfile.k711 -t kraken-train:k711 src/finetuning/kraken"
    fi
    busy=$(docker ps -a --filter "name=$PREFIX-" --format '{{.Names}}' | grep -E "^$PREFIX-" || true)
    if [[ -n $busy ]]; then
        die "container(s) $(echo $busy) already exist; one capped kraken-train-arabic container at a time - wait for it"
    fi
    if [[ $STAGE == compile ]]; then check_native_idle; fi  # compile rewrites the arrows a native run is reading
}

lms_state() {  # one `$LMS ps` (killed after 20 s): prints 'idle (<n> loaded)', 'busy: <id>=<STATUS> ...' or 'unknown (<why>)'
    local out rc=0
    out=$(perl -e "$TIMEOUT_EXEC" 20 "$LMS" ps 2>/dev/null < /dev/null) || rc=$?
    if [[ $rc != 0 ]]; then echo "unknown (lms ps exit $rc)"; return 0; fi
    # The table: optional blank line, a header with a STATUS column, one row per loaded model, a blank line.
    printf '%s\n' "$out" | awk '
        tolower($0) ~ /no models/ { none = 1 }
        !col { for (i = 1; i <= NF; i++) if ($i == "STATUS") col = i; next }
        NF == 0 { if (n) ended = 1; next }
        ended || NF < col { next }
        { n++; if (toupper($col) != "IDLE") busy = busy " " $1 "=" $col }
        END {
            if (!col) print (none ? "idle (no models loaded)" : "unknown (no STATUS column in lms ps output)")
            else if (busy != "") print "busy:" busy
            else print "idle (" n + 0 " loaded)"
        }'
}

guard_log() {  # guard_log <message>: timestamped line to the stage log and stderr; never fails
    local line
    line="$(ts) $*"
    printf '%s\n' "$line" >> "$DOCK_LOG" 2>/dev/null
    printf '%s\n' "$line" >&2 2>/dev/null
    return 0
}

lms_guard() {  # lms_guard <pgid> <state file>: the LM Studio guard loop (run in the background; ends when <pgid> exits)
    local pgid=$1 state_file=$2 state paused=0 idle_from=-1 paused_at=0 noted=0 pauses=0 total=0
    set +e  # never die on a stray failure: this loop may be holding the process group stopped
    trap '' HUP INT PIPE  # outlive a hung-up driver; keep guarding until the process group is gone
    echo "0 0" > "$state_file"
    guard_log "guard on: process group $pgid; $LMS ps every ${GUARD_POLL} s, SIGSTOP while any model is not IDLE, SIGCONT after ${GUARD_IDLE} s all IDLE"
    while kill -0 "$pgid" 2>/dev/null; do
        state=$(lms_state)
        if [[ $state == idle* ]]; then
            if [[ $paused == 1 ]]; then
                if (( idle_from < 0 )); then idle_from=$SECONDS; fi
                if (( SECONDS - idle_from >= GUARD_IDLE )); then
                    kill -CONT -- "-$pgid" 2>/dev/null
                    paused=0; total=$((total + SECONDS - paused_at))
                    guard_log "guard: LM Studio $state for $((SECONDS - idle_from)) s -> SIGCONT process group $pgid (paused $((SECONDS - paused_at)) s)"
                    echo "$pauses $total" > "$state_file"
                fi
            fi
        else
            idle_from=-1
            if [[ $paused == 0 ]]; then
                kill -STOP -- "-$pgid" 2>/dev/null
                paused=1; paused_at=$SECONDS; noted=$SECONDS; pauses=$((pauses + 1))
                guard_log "guard: LM Studio $state -> SIGSTOP process group $pgid"
                echo "$pauses $total" > "$state_file"
            elif (( SECONDS - noted >= 600 )); then
                noted=$SECONDS
                guard_log "guard: still paused after $((SECONDS - paused_at)) s: LM Studio $state"
            fi
        fi
        sleep "$GUARD_POLL" > /dev/null 2>&1 < /dev/null  # no inherited fds: a sleep orphaned by the driver's kill must not
    done                                                # hold a caller's output pipe open
    guard_log "guard: process group $pgid is gone; guard ends"
}

native_abort() {  # INT/TERM/HUP trap while a native command runs: stop the guard, then SIGTERM + SIGCONT the group
    trap '' PIPE
    NATIVE_ABORTED=1
    if [[ -n $NATIVE_GUARD_PID ]]; then kill "$NATIVE_GUARD_PID" 2>/dev/null || true; NATIVE_GUARD_PID=""; fi
    if [[ -n $NATIVE_PGID ]]; then
        kill -TERM -- "-$NATIVE_PGID" 2>/dev/null || true
        kill -CONT -- "-$NATIVE_PGID" 2>/dev/null || true
        guard_log "driver interrupted: SIGTERM + SIGCONT sent to process group $NATIVE_PGID"
    fi
}

preflight_native() {  # venv, perl and lms ps usable; no other native run of this driver; no compile container
    local names state
    if [[ ! -x $VENV_PYTHON ]]; then die "VENV_PYTHON=$VENV_PYTHON is not executable (ENGINE=native needs the kraken 7.1.1 venv)"; fi
    if [[ ! -f $VENV_KETOS ]]; then die "$VENV_KETOS not found: VENV_PYTHON must be the python of a venv bin/ that holds kraken's ketos script"; fi
    if ! command -v perl >/dev/null 2>&1; then die "perl not found (ENGINE=native starts the command in its own process group with it)"; fi
    check_native_idle
    if command -v docker >/dev/null 2>&1; then
        names=$(docker ps -a --filter "name=$PREFIX-compile" --format '{{.Names}}' 2>/dev/null || true)
        if grep -qx -- "$PREFIX-compile" <<< "$names"; then
            die "container $PREFIX-compile exists: compile rewrites the arrows and manifests a native stage reads - wait for it"
        fi
    fi
    if [[ $NO_GUARD == 1 ]]; then
        echo "$(ts) WARNING: NO_GUARD=1 - no LM Studio guard: this native $STAGE will NOT pause for production" >&2
    else
        if [[ ! -x $LMS ]]; then die "LMS=$LMS is not executable: the native guard needs lms ps (NO_GUARD=1 runs without it, unprotected)"; fi
        state=$(lms_state)
        case $state in
            unknown*) die "lms ps unreadable before the start: $state; the guard cannot see production - fix it, or NO_GUARD=1 (unprotected)" ;;
        esac
        echo "$(ts) LM Studio now: $state"
    fi
    echo "$(ts) ENGINE=native: $VENV_PYTHON, -d $DEVICE, cwd $REPO_ROOT, no --memory cap (MEM unused)"
}

build_docker_args() {  # the capped, --rm container settings shared by every stage
    local d
    DOCKER_ARGS=(--rm --cpus "$CPUS" --memory "$MEM" --oom-score-adj 1000 --shm-size "$SHM"
        --log-opt max-size=50m --log-opt max-file=2
        -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS="$THREADS"
        -v "$DATA:/data" -v "$DATA:$DATA:ro" -v "$AR_DIR:/ar:ro" -v "$SCRIPT_DIR:/work:ro" -w /data)
    if [[ $COMPILE != 1 ]]; then DOCKER_ARGS+=(-e TORCH_COMPILE_DISABLE=1); fi
    for d in $EXTRA_RO; do DOCKER_ARGS+=(-v "$d:$d:ro"); done
}

dock() {  # dock <stage> <command...>: run one capped container, append its output to the stage log, return its exit code
    local stage=$1 rc note=""
    shift
    local cname="$PREFIX-$stage"
    DOCK_LOG="$DATA/models/${stage}_${NAME}.log"
    touch "$DOCK_LOG"
    DOCK_START=$(wc -l < "$DOCK_LOG" | tr -d ' ')
    echo "$(ts) $cname: $*" | tee -a "$DOCK_LOG"
    set +e
    docker run --name "$cname" "${DOCKER_ARGS[@]}" "$IMAGE" "$@" 2>&1 | tee -a "$DOCK_LOG"
    rc=${PIPESTATUS[0]}
    set -e
    if [[ $rc == 137 ]]; then note=" (SIGKILL: almost always the --memory $MEM cap; lower BATCH)"; fi
    echo "$(ts) $cname exit=$rc$note" | tee -a "$DOCK_LOG"
    return "$rc"
}

native() {  # native <stage> <guard 0|1> <command...>: run a host command like dock() runs a container (ENGINE=native)
    local stage=$1 guard=$2 rc=1 note="" tmp pipe_pid pgid="" guard_pid="" pauses
    shift 2
    DOCK_LOG="$DATA/models/${stage}_${NAME}.log"
    touch "$DOCK_LOG"
    DOCK_START=$(wc -l < "$DOCK_LOG" | tr -d ' ')
    NATIVE_PIDFILE="$DATA/models/native_${stage}_${NAME}.pid"
    tmp=$(mktemp -d "${TMPDIR:-/tmp}/finetune_arabic.XXXXXX") || die "mktemp failed"
    echo "$(ts) native-$stage: cd $REPO_ROOT && $*" | tee -a "$DOCK_LOG"
    if [[ $guard == 1 && $NO_GUARD == 1 ]]; then
        echo "$(ts) WARNING native-$stage: NO_GUARD=1 - LM Studio guard OFF, this run will NOT pause for production" | tee -a "$DOCK_LOG" >&2
    fi
    NATIVE_PGID=""; NATIVE_GUARD_PID=""; NATIVE_ABORTED=0
    trap native_abort INT TERM HUP
    {
        native_rc=0
        export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="$THREADS" KETOS_FROZEN_BN="$FROZEN_BN"
        if [[ $COMPILE == 1 ]]; then unset TORCH_COMPILE_DISABLE; else export TORCH_COMPILE_DISABLE=1; fi
        cd "$REPO_ROOT" && perl -e "$SETPGRP_EXEC" "$tmp/pgid" "$@" || native_rc=$?
        echo "$native_rc" > "$tmp/rc"
    } 2>&1 | tee -a "$DOCK_LOG" &
    pipe_pid=$!
    while [[ ! -s $tmp/pgid ]] && kill -0 "$pipe_pid" 2>/dev/null; do sleep 0.1; done
    if [[ -s $tmp/pgid ]]; then
        pgid=$(cat "$tmp/pgid")
        echo "$pgid" > "$NATIVE_PIDFILE"
        echo "$(ts) native-$stage: process group $pgid (pid file $NATIVE_PIDFILE)" | tee -a "$DOCK_LOG"
        if [[ $guard == 1 && $NO_GUARD != 1 ]]; then
            lms_guard "$pgid" "$tmp/guard" &
            guard_pid=$!
        fi
        NATIVE_PGID=$pgid; NATIVE_GUARD_PID=$guard_pid
        if [[ $NATIVE_ABORTED == 1 ]]; then native_abort; fi  # a signal arrived before the group id was known
    fi
    while kill -0 "$pipe_pid" 2>/dev/null; do wait "$pipe_pid" 2>/dev/null || true; done  # a trapped signal ends a wait early
    trap - INT TERM HUP
    if [[ -n $guard_pid ]]; then
        kill "$guard_pid" 2>/dev/null || true
        wait "$guard_pid" 2>/dev/null || true
        pauses=$(cat "$tmp/guard" 2>/dev/null || echo "0 0")
        echo "$(ts) guard off: process group $pgid ended after ${pauses% *} pause(s), ${pauses#* } s paused" | tee -a "$DOCK_LOG"
    fi
    if [[ -s $tmp/rc ]]; then rc=$(cat "$tmp/rc"); fi
    rm -f "$NATIVE_PIDFILE" "$tmp/pgid" "$tmp/rc" "$tmp/guard"
    rmdir "$tmp" 2>/dev/null || true
    NATIVE_PGID=""; NATIVE_GUARD_PID=""
    case $rc in
        137) note=" (SIGKILL: killed from outside, e.g. macOS memory pressure or kill -9)" ;;
        143) note=" (SIGTERM)" ;;
    esac
    echo "$(ts) native-$stage exit=$rc$note" | tee -a "$DOCK_LOG"
    if [[ $NATIVE_ABORTED == 1 ]]; then die "interrupted: process group ${pgid:-?} was sent SIGTERM (log $DOCK_LOG)"; fi
    return "$rc"
}

run_stage() {  # run_stage <stage> <command...>: one capped container (docker) or a guarded host command (native)
    local stage=$1
    shift
    if [[ $ENGINE == native ]]; then native "$stage" 1 "$@"; else dock "$stage" "$@"; fi
}

set_ketos() {  # set_ketos <frozen 0|1>: KETOS = the ketos command prefix for ENGINE (1 = the frozen-BatchNorm wrapper)
    if [[ $1 == 1 && $ENGINE == native ]]; then
        if [[ ! -f $REPO_ROOT/src/finetuning/kraken/ketos_frozen_bn.py ]]; then
            die "$REPO_ROOT/src/finetuning/kraken/ketos_frozen_bn.py missing: FROZEN_BN=1 runs it as a module from the repo root"
        fi
        KETOS=("$VENV_PYTHON" -m src.finetuning.kraken.ketos_frozen_bn)  # native() runs every command from REPO_ROOT
    elif [[ $1 == 1 ]]; then
        if [[ ! -f $SCRIPT_DIR/ketos_frozen_bn.py ]]; then
            die "$SCRIPT_DIR/ketos_frozen_bn.py missing: FROZEN_BN=1 runs it from /work, the read-only mount of this directory"
        fi
        KETOS=(python /work/ketos_frozen_bn.py)
    elif [[ $ENGINE == native ]]; then
        KETOS=("$VENV_PYTHON" "$VENV_KETOS")
    else
        KETOS=(ketos)
    fi
}

native_manifest() {  # native_manifest <manifest under DATA>: write its host-path copy to $DATA/native/, print that path
    local dst="$DATA/native/${1//\//_}" a h
    read_manifest "$DATA/$1"
    mkdir -p "$DATA/native"
    : > "$dst.tmp"
    for a in "${MANIFEST[@]}"; do
        h=$(engine_path "$a")
        if [[ ! -f $h ]]; then die "$DATA/$1 lists $a: no such file on the host ($h)"; fi
        printf '%s\n' "$h" >> "$dst.tmp"
    done
    mv -f "$dst.tmp" "$dst"
    printf '%s\n' "$dst"
}

run_setting() {  # run_setting <run name> <key>: the last <key>=<value> its train log recorded ('' when none)
    local tok
    tok=$(grep -a -o -E "(^|[^A-Za-z_])$2=[A-Za-z0-9_]+" "$DATA/models/train_$1.log" 2>/dev/null | tail -n 1 || true)
    printf '%s\n' "${tok#*=}"
}

model_run() {  # model_run <model path>: <run> for /data/models/<run>/... or $DATA/models/<run>/... ('' otherwise)
    local rest
    case $1 in
        /data/models/*/*) rest=${1#/data/models/} ;;
        "$DATA"/models/*/*) rest=${1#"$DATA"/models/} ;;
        *) return 0 ;;
    esac
    printf '%s\n' "${rest%%/*}"
}

since_start() {  # output the last dock()/native() call appended to its log
    tail -n +"$((DOCK_START + 1))" "$DOCK_LOG" | tr '\r' '\n'
}

visible() {  # visible <path>: true when an absolute path resolves inside the container (under /data, $DATA or EXTRA_RO)
    local d
    case $1 in /data/*|"$DATA"/*) return 0 ;; esac
    for d in $EXTRA_RO; do
        case $1 in "${d%/}"/*) return 0 ;; esac
    done
    return 1
}

check_compile_workers() {  # check_compile_workers <list> <host path>: refuse COMPILE_WORKERS > 1 for a file on the NAS
    if [[ -z $COMPILE_WORKERS ]] || (( 10#$COMPILE_WORKERS <= 1 )); then return 0; fi
    case $2 in
        /Volumes/*) die "COMPILE_WORKERS=$COMPILE_WORKERS refused for $1: $2 is on the NAS, and ketos compile --workers 5 over this bind mount died with 'OSError: Too many open files' after ~1k files (KTIV fine-tune); leave COMPILE_WORKERS unset (or 0/1) or stage the files on local disk" ;;
    esac
}

check_list() {  # check_list <set> <host list>: its first entry (and, for PAGE, that file's image) must be reachable
    local first host xml_dir img
    first=$(grep -m 1 '[^[:space:]]' "$2" || true)
    if ! visible "$first"; then die "$2: entry $first is not visible in the container (add its dir to EXTRA_RO)"; fi
    host=$first
    case $first in /data/*) host="$DATA/${first#/data/}" ;; esac
    if [[ ! -f $host ]]; then die "$2: first entry $host does not exist"; fi
    check_compile_workers "$2" "$host"
    if is_crop_set "$1"; then
        if [[ ! -f ${host%.*}.gt.txt ]]; then die "$2: crop $host has no ${host%.*}.gt.txt"; fi
        return 0
    fi
    img=$(grep -o -m 1 'imageFilename="[^"]*"' "$host" | head -n 1 | sed 's/^imageFilename="//; s/"$//' || true)
    xml_dir=$(dirname "$host")
    case $img in
        "") die "$host has no imageFilename" ;;
        /*) if ! visible "$img"; then die "$host: image $img is not visible in the container (add its dir to EXTRA_RO)"; fi
            if [[ ! -f $img ]]; then die "$host: image $img does not exist"; fi
            check_compile_workers "$2" "$img" ;;
        *) if [[ ! -f $xml_dir/$img ]]; then die "$host: image $xml_dir/$img does not exist"; fi ;;
    esac
}

is_crop_set() {  # is_crop_set <set>: true for sets of pre-cropped line images
    case " $CROP_SETS " in *" $1 "*) return 0 ;; esac
    return 1
}

read_manifest() {  # read_manifest <host file>: its non-empty lines into MANIFEST
    local line
    MANIFEST=()
    if [[ ! -s $1 ]]; then die "$1 missing or empty: run the compile stage first"; fi
    while IFS= read -r line || [[ -n $line ]]; do
        if [[ -n $line ]]; then MANIFEST+=("$line"); fi
    done < "$1"
    if [[ ${#MANIFEST[@]} == 0 ]]; then die "$1 lists no arrows"; fi
}

stage_compile() {
    local s split lst n extracted arrow rc=0
    local -a fmt cmd train_arrows val_arrows report
    train_arrows=(); val_arrows=(); report=()
    for s in $SETS; do  # every list must exist (and its first entry resolve) before the first container starts
        for split in train val; do
            lst="$DATA/lists/${s}_${split}.lst"
            if [[ ! -f $lst ]]; then die "missing $lst (SETS=$SETS)"; fi
            if grep -q '[^[:space:]]' "$lst"; then check_list "$s" "$lst"; fi
        done
    done
    mkdir -p "$DATA/arrow"
    rm -f "$DATA/train.manifest" "$DATA/val.manifest"
    for s in $SETS; do
        for split in train val; do
            lst="$DATA/lists/${s}_${split}.lst"
            arrow="/data/arrow/${s}_${split}.arrow"
            n=$(grep -c '[^[:space:]]' "$lst" || true)
            if [[ $n == 0 ]]; then report+=("$s $split: empty list, skipped"); continue; fi
            if is_crop_set "$s"; then fmt=(-f path --force-type baseline); else fmt=(-f page --linetype baselines); fi
            cmd=(ketos)
            if [[ -n $COMPILE_WORKERS ]]; then cmd+=(--workers "$COMPILE_WORKERS"); fi
            cmd+=(compile "${fmt[@]}" -F "/data/lists/${s}_${split}.lst" -o "$arrow")
            dock compile "${cmd[@]}"
            extracted=$(since_start | grep -a 'Extracting lines' | grep -aoE '[0-9]+/[0-9]+' | tail -n 1 || true)
            report+=("$s $split: $n list entries ($(if is_crop_set "$s"; then echo crops; else echo PAGE files; fi)), kraken extracted ${extracted:-?} lines -> $arrow")
            if [[ $split == train ]]; then train_arrows+=("$arrow"); else val_arrows+=("$arrow"); fi
        done
    done
    echo "$(ts) compile summary (kraken's progress counts; the arrow rows follow):" | tee -a "$DOCK_LOG"
    printf '  %s\n' "${report[@]}" | tee -a "$DOCK_LOG"
    if [[ ${#train_arrows[@]} == 0 || ${#val_arrows[@]} == 0 ]]; then die "need at least one train and one val arrow"; fi
    dock compile python /work/gate_lines.py count "${train_arrows[@]}" "${val_arrows[@]}" || rc=$?
    case $rc in
        0) ;;
        3) die "gate_lines.py count: mixed baseline/bbox arrows (ketos train refuses them); manifests not written" ;;
        4) die "gate_lines.py count: TEXT RULE violation (symbols listed above); fix build_arabic_lines.py and rebuild; manifests not written" ;;
        *) die "gate_lines.py count failed (exit $rc, see $DOCK_LOG); manifests not written" ;;
    esac
    printf '%s\n' "${train_arrows[@]}" > "$DATA/train.manifest"
    printf '%s\n' "${val_arrows[@]}" > "$DATA/val.manifest"
    echo "$(ts) wrote $DATA/train.manifest (${#train_arrows[@]} arrows) and $DATA/val.manifest (${#val_arrows[@]}); next: gate" | tee -a "$DOCK_LOG"
}

stage_gate() {
    local m a
    local -a arrows
    arrows=()
    for m in $GATE_MANIFESTS; do
        read_manifest "$DATA/$m.manifest"
        arrows+=("${MANIFEST[@]}")
    done
    dock gate python /work/gate_lines.py sample --out-root /data/gate --n "$GATE_N" --seed "$GATE_SEED" "${arrows[@]}"
    echo "LOOK at these before training (does each image show exactly its text?):"
    for a in "${arrows[@]}"; do echo "  $DATA/gate/$(basename "$a" .arrow)/index.html"; done
}

check_gate() {  # every train arrow needs a gate dump written after the arrow itself
    local a stem missing=""
    read_manifest "$DATA/train.manifest"
    for a in "${MANIFEST[@]}"; do
        stem=$(basename "$a" .arrow)
        if [[ ! $DATA/gate/$stem/index.tsv -nt $(host_path "$a") ]]; then missing+=" $stem"; fi
    done
    if [[ -n $missing ]]; then
        die "no gate dump newer than the arrow for:$missing - run the gate stage and look at $DATA/gate first (SKIP_GATE_CHECK=1 overrides)"
    fi
}

check_early_stopping() {  # QUIT=early: refuse unless the PP-OCRv6 trainer (image or venv) can actually stop the run
    local rc=0 where=$IMAGE
    if [[ $ENGINE == native ]]; then
        where="the venv of $VENV_PYTHON"
        native train-check 0 "$VENV_PYTHON" "$SCRIPT_DIR/gate_lines.py" early-check || rc=$?
    else
        dock train-check python /work/gate_lines.py early-check || rc=$?
    fi
    case $rc in
        0) ;;
        5) die "QUIT=early refused: train/ppocr.py in $where instantiates no EarlyStopping and ketos sets max_epochs=-1 under -q early, so the run would never end (lines in $DOCK_LOG); use QUIT=fixed" ;;
        6) if [[ $EARLY_LR_OK != 1 ]]; then
               die "QUIT=early refused: train/ppocr.py reads trainer.estimated_stepping_batches, which Lightning returns as inf for max_epochs=-1 (a step-based LR schedule then never anneals or fails at fit start); read the lines in $DOCK_LOG and set EARLY_LR_OK=1 only if ppocr.py copes with that, else use QUIT=fixed"
           fi ;;
        *) die "QUIT=early refused: the trainer check failed (exit $rc, see $DOCK_LOG); use QUIT=fixed" ;;
    esac
}

run_test() {  # run_test <stage> <model path (container form; a host path also works natively)>: ketos test on EVAL_MANIFEST, record the accuracies
    local acc word eval_arg run frozen="-"
    read_manifest "$DATA/$EVAL_MANIFEST"
    require_file "$2"
    set_ketos 0
    if [[ $ENGINE == native ]]; then eval_arg=$(native_manifest "$EVAL_MANIFEST"); else eval_arg="/data/$EVAL_MANIFEST"; fi
    run_stage "$1" "${KETOS[@]}" -d "$DEVICE" --threads "$THREADS" --workers "$WORKERS" test -m "$(engine_path "$2")" -f binary \
        -e "$eval_arg" -u NFD -n --base-dir R -B "$TEST_BATCH"
    acc=$(since_start | grep -aE '%[[:space:]]+Character Accuracy[[:space:]]*$' | tail -n 1 | awk '{print $1}' || true)
    word=$(since_start | grep -aE '%[[:space:]]+Word Accuracy[[:space:]]*$' | tail -n 1 | awk '{print $1}' || true)
    if [[ $1 != baseline ]]; then
        run=$(model_run "$2")
        if [[ -n $run ]]; then frozen=$(run_setting "$run" frozen_bn); frozen=${frozen:-?}; fi
    fi
    echo "$(ts) $1: $2 on $EVAL_MANIFEST ($ENGINE:$DEVICE): Character Accuracy ${acc:-?}, Word Accuracy ${word:-?}" | tee -a "$DOCK_LOG"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(ts)" "$1" "$2" "$EVAL_MANIFEST" "${acc:-?}" "${word:-?}" "$ENGINE:$DEVICE" "$frozen" \
        >> "$DATA/models/results.tsv"
}

stage_train() {
    local out="$DATA/models/$NAME" source_note stop_note engine_note tman vman
    local -a args from stop
    read_manifest "$DATA/val.manifest"
    read_manifest "$DATA/train.manifest"
    if [[ -n $RESUME ]]; then  # continue an interrupted run of this NAME: its directory is expected to hold files
        require_file "$RESUME"
        from=(--resume "$(engine_path "$RESUME")"); source_note="resume $RESUME"
    else
        require_file "$BASE"
        if [[ -d $out && -n $(ls -A "$out") ]]; then
            die "$out already holds files: one name per run - pick a new NAME (or continue it: RESUME=/data/models/$NAME/checkpoint_<NN>-<score>.ckpt)"
        fi
        if [[ $SKIP_GATE_CHECK != 1 ]]; then check_gate; fi
        from=(-i "$(engine_path "$BASE")" --resize union); source_note="base $BASE"
    fi
    if [[ $QUIT == early ]]; then
        check_early_stopping
        stop=(-q early --min-epochs "$MIN_EPOCHS" --lag "$LAG")
        stop_note="early stopping (min $MIN_EPOCHS epochs, lag $LAG; no epoch cap)"
    else
        stop=(-q fixed -N "$EPOCHS")
        stop_note="exactly $EPOCHS epochs (best validation checkpoint kept)"
    fi
    set_ketos "$FROZEN_BN"
    if [[ $ENGINE == native ]]; then
        tman=$(native_manifest train.manifest)
        vman=$(native_manifest val.manifest)
        if [[ $NO_GUARD == 1 ]]; then engine_note="guard OFF (NO_GUARD=1)"; else engine_note="guard on (lms ps every ${GUARD_POLL} s, resume after ${GUARD_IDLE} s idle)"; fi
        engine_note="engine=native device=$DEVICE threads $THREADS, no memory cap, $engine_note"
    else
        tman=/data/train.manifest; vman=/data/val.manifest
        engine_note="engine=docker device=$DEVICE cpus $CPUS threads $THREADS mem $MEM"
    fi
    args=("${KETOS[@]}" -d "$DEVICE" --threads "$THREADS" --workers "$WORKERS" -v train -f binary "${from[@]}"
        -t "$tman" -e "$vman" -o "$(engine_path "/data/models/$NAME")" -B "$BATCH" -r "$LR" "${stop[@]}"
        --schedule "$SCHEDULE" --warmup "$WARMUP")
    if [[ $SCHEDULE == cosine ]]; then args+=(--cos-max "$EPOCHS"); fi
    args+=(-u NFD -n --base-dir R --reorder "$AUGMENT")
    echo "$(ts) train $NAME: ${#MANIFEST[@]} train arrows, $source_note, batch $BATCH, lr $LR, $stop_note, $SCHEDULE warmup $WARMUP, $AUGMENT, $engine_note, frozen_bn=$FROZEN_BN, torch.compile $COMPILE" \
        | tee -a "$DATA/models/train_${NAME}.log"
    run_stage train "${args[@]}"
    echo "$(ts) checkpoints in $out; next: ENGINE=$ENGINE NAME=$NAME $0 test"
}

best_model() {  # file name of the highest-scoring best_<score>.safetensors of NAME ('' when none)
    local f s
    for f in "$DATA/models/$NAME"/best_*.safetensors; do
        if [[ -f $f ]]; then
            s=${f##*/best_}
            printf '%s\t%s\n' "${s%.safetensors}" "${f##*/}"
        fi
    done | sort -g -k1,1 | tail -n 1 | cut -f2
}

stage_test() {
    local model=$MODEL best
    if [[ -z $model ]]; then
        best=$(best_model)
        if [[ -z $best ]]; then
            die "no $DATA/models/$NAME/best_*.safetensors (interrupted run? RESUME it, or ketos convert its best checkpoint_*.ckpt) - or set MODEL=/data/..."
        fi
        model="/data/models/$NAME/$best"
    fi
    run_test test "$model"
}

case $STAGE in
    compile|gate|baseline|train|test) ;;
    *) echo "usage: [ENGINE=docker|native] $0 compile|gate|baseline|train|test   (settings via env; see the header)" >&2; exit 2 ;;
esac
case $ENGINE in
    docker) DEVICE=${DEVICE:-cpu} ;;
    native) DEVICE=${DEVICE:-mps}
            case $STAGE in
                compile|gate) die "ENGINE=native runs baseline, train and test only: $STAGE stays in docker (ENGINE=docker; it writes and reads the arrows through the container's /data and /work mounts)" ;;
            esac ;;
    *) die "ENGINE=$ENGINE: use docker or native" ;;
esac
check_knobs
mkdir -p "$DATA/models"
DOCK_LOG="$DATA/models/${STAGE}_${NAME}.log"; DOCK_START=0
if [[ $ENGINE == native ]]; then
    preflight_native
else
    preflight_docker
    build_docker_args
fi
case $STAGE in
    compile) stage_compile ;;
    gate) stage_gate ;;
    baseline) run_test baseline "$BASE" ;;
    train) stage_train ;;
    test) stage_test ;;
esac
