#!/bin/zsh
# Convert a merged bf16 checkpoint to 8-bit MLX with the kit's low-memory wrapper, verify the result and record it.
#
# usage: run_convert.sh --hf-path <bf16 dir> --mlx-path <output dir> [--python <interpreter>] [--min-free-gib 12]
#                       [--identity-json <report>] [--no-hash] [--force]
#
# Refuses to start when: the source lacks config.json, the shard index, generation_config.json or a chat template;
# the output directory exists; <output>.partial exists (another converter's staging directory, never deleted here);
# <output>.lock exists (another run of this kit holds it); the interpreter lacks mlx or does not match
# requirements.txt; LM Studio or another GPU server runs on this Mac (Metal's watchdog then kills the conversion;
# --force overrides that one check); free memory is below --min-free-gib. The interpreter is $PYTHON, else
# ~/.venv-mlx-kit/bin/python when it exists, else python3.
# Runs under caffeinate (prevents idle sleep only: keep the lid open, or use clamshell mode on power with a display),
# stages in <output>.kit-partial, verifies, writes <output>.MANIFEST.json and <output>.CONVERTED.json, then renames
# the staging directory to <output> with a rename that fails instead of nesting if <output> appeared meanwhile.
# Not resumable: a failed attempt leaves <output>.kit-partial for inspection; the next run deletes it.
set -euo pipefail
KIT="${0:A:h}"
HF_PATH="" MLX_PATH="" PYTHON="${PYTHON:-}" MIN_FREE_GIB=12 IDENTITY_JSON="" FORCE=0 NO_HASH=0

die() { print -u2 -- "run_convert: $*"; exit 1; }
need() { (( $# >= 2 )) || die "$1 needs a value"; }

while (( $# )); do
  case "$1" in
    --hf-path) need "$@"; HF_PATH="$2"; shift 2 ;;
    --mlx-path) need "$@"; MLX_PATH="$2"; shift 2 ;;
    --python) need "$@"; PYTHON="$2"; shift 2 ;;
    --min-free-gib) need "$@"; MIN_FREE_GIB="$2"; shift 2 ;;
    --identity-json) need "$@"; IDENTITY_JSON="$2"; shift 2 ;;
    --no-hash) NO_HASH=1; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) die "unknown argument $1" ;;
  esac
done

# 1. the source must hold everything the verifier later requires in the output
[[ -n "$HF_PATH" && -n "$MLX_PATH" ]] || die "--hf-path and --mlx-path are required"
[[ -d "$HF_PATH" ]] || die "$HF_PATH is not a directory"
for f in config.json model.safetensors.index.json generation_config.json; do
  [[ -f "$HF_PATH/$f" ]] || die "$HF_PATH lacks $f (the output verification requires it)"
done
if [[ ! -f "$HF_PATH/chat_template.jinja" ]]; then
  grep -q '"chat_template"' "$HF_PATH/tokenizer_config.json" 2>/dev/null \
    || die "$HF_PATH has neither chat_template.jinja nor a chat_template in tokenizer_config.json"
fi

# 2. the output path: new, absolute, in an existing parent; never touch another converter's staging directory
MLX_PATH="${MLX_PATH%/}"
[[ -n "$MLX_PATH" && "$MLX_PATH" != "." && "$MLX_PATH" != ".." ]] || die "--mlx-path must name a new directory"
MLX_PATH="${MLX_PATH:A}"
[[ -d "${MLX_PATH:h}" ]] || die "parent directory ${MLX_PATH:h} does not exist (is the share mounted?)"
[[ -e "$MLX_PATH" ]] && die "$MLX_PATH exists; mlx_lm never writes into an existing directory (remove it or choose another name)"
[[ -e "$MLX_PATH.partial" ]] && die "$MLX_PATH.partial exists: another converter (for example the serving machine's pipeline) may be writing it; not deleting it. Remove it yourself only if that run is dead."
if [[ -n "$IDENTITY_JSON" ]]; then
  [[ -f "$IDENTITY_JSON" ]] || die "no identity report at $IDENTITY_JSON (gpu_quant_identity.py writes none when it exits 11)"
  IDENTITY_JSON="${IDENTITY_JSON:A}"
fi

# 3. one conversion of this model at a time, across machines: an atomic mkdir on the share
LOCK="$MLX_PATH.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  die "$LOCK exists ($(cat "$LOCK/owner" 2>/dev/null || print -- 'no owner file')): another run holds it; remove it only if that run is dead"
fi
print -- "$(scutil --get LocalHostName 2>/dev/null || hostname -s) pid $$ started $(date -u +%FT%TZ)" > "$LOCK/owner"
trap 'rm -rf "$LOCK"' EXIT INT TERM HUP
[[ -e "$MLX_PATH" ]] && die "$MLX_PATH appeared while taking the lock (made elsewhere)"

# 4. the interpreter: has mlx, matches the pins (the pins decide the tokenizer files' bytes)
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$HOME/.venv-mlx-kit/bin/python" ]]; then PYTHON="$HOME/.venv-mlx-kit/bin/python"; else PYTHON=python3; fi
fi
if [[ -n "$IDENTITY_JSON" ]]; then
  "$PYTHON" -c 'import json, sys; json.load(open(sys.argv[1]))' "$IDENTITY_JSON" 2>/dev/null || die "$IDENTITY_JSON is not valid JSON"
fi
"$PYTHON" -c 'import mlx.core, mlx_lm' 2>/dev/null || die "$PYTHON has no mlx / mlx_lm (make the venv from requirements.txt and pass --python or PYTHON=)"
"$PYTHON" "$KIT/kit_record.py" --check-pins "$KIT/requirements.txt" >/dev/null \
  || die "$PYTHON does not match requirements.txt (mismatches listed above); fix the venv so the result stays comparable"

# 5. nothing else may use the GPU
PATTERN='LM Studio|lms server|llama-server|ollama|mlx_lm.server|mlx_lm server|mlx_vlm|ComfyUI|vllm|convert_lowmem.py|gpu_quant_identity.py'
GPU_LIST="$(pgrep -f "$PATTERN" 2>/dev/null | grep -vx -- "$$" || true)"
if [[ -n "$GPU_LIST" ]]; then
  GPU_PIDS=(${(f)GPU_LIST})
  print -u2 -- "processes that use the GPU (pid, name):"
  ps -o pid=,comm= -p "${(j:,:)GPU_PIDS}" 2>/dev/null | cut -c1-120 | head -20 >&2
  (( FORCE )) || die "stop them first, or pass --force (Metal's watchdog kills a conversion that shares the GPU)"
fi

# 6. memory and power
PAGE=$(sysctl -n hw.pagesize)
FREE_GIB=$(vm_stat | awk -v page="$PAGE" '/Pages (free|inactive|speculative)/ {gsub(/\./, "", $NF); s += $NF} END {printf "%.1f", s * page / 1073741824}')
if (( FREE_GIB < MIN_FREE_GIB )); then
  die "only $FREE_GIB GiB free (+inactive, speculative); need $MIN_FREE_GIB (override with --min-free-gib)"
fi
pmset -g batt 2>/dev/null | grep -q "AC Power" || print -u2 -- "warning: not on AC power"

# 7. convert into the kit's own staging directory
PARTIAL="$MLX_PATH.kit-partial" LOG="$MLX_PATH.convert.log"
if [[ -e "$PARTIAL" ]]; then print -- "removing interrupted $PARTIAL"; rm -rf "$PARTIAL"; fi
print -- "=== $(date -u +%FT%TZ) convert $HF_PATH -> $PARTIAL on $(sysctl -n machdep.cpu.brand_string), $FREE_GIB GiB free, $PYTHON" | tee -a "$LOG"
START=$(date +%s)
if ! caffeinate -dims nice -n 10 "$PYTHON" -u "$KIT/convert_lowmem.py" --hf-path "$HF_PATH" --mlx-path "$PARTIAL" \
      --q-bits 8 --q-group-size 64 2>&1 | tee -a "$LOG"; then
  die "conversion failed after $(( $(date +%s) - START )) s; see $LOG ($PARTIAL left for inspection)"
fi
TOOK=$(( $(date +%s) - START ))
print -- "conversion finished in $TOOK s; verifying" | tee -a "$LOG"

# 8. verify, manifest, record, then the rename
HASH_FLAG=()
if (( NO_HASH )); then HASH_FLAG=(--no-hash); fi
"$PYTHON" "$KIT/verify_mlx_dir.py" "$PARTIAL" --manifest "$MLX_PATH.MANIFEST.json" "${HASH_FLAG[@]}" \
  || die "verification failed; $PARTIAL left for inspection"
IDENTITY_FLAG=()
if [[ -n "$IDENTITY_JSON" ]]; then IDENTITY_FLAG=(--identity-json "$IDENTITY_JSON"); fi
"$PYTHON" "$KIT/kit_record.py" --log "$LOG" --out "$MLX_PATH.CONVERTED.json" --hf-path "$HF_PATH" --mlx-path "$MLX_PATH" \
  --device gpu --seconds "$TOOK" --manifest "$MLX_PATH.MANIFEST.json" --requirements "$KIT/requirements.txt" \
  "${IDENTITY_FLAG[@]}" > /dev/null \
  || die "record failed; verified result left at $PARTIAL (write the record with kit_record.py, then rename it to $MLX_PATH yourself)"
if [[ -e "$MLX_PATH" ]]; then
  rm -f "$MLX_PATH.CONVERTED.json"
  die "$MLX_PATH appeared during the run (made elsewhere); verified result left at $PARTIAL"
fi
if ! "$PYTHON" -c 'import os, sys; os.rename(sys.argv[1], sys.argv[2])' "$PARTIAL" "$MLX_PATH"; then
  rm -f "$MLX_PATH.CONVERTED.json"
  die "rename failed; verified result left at $PARTIAL"
fi
print -- "done: $MLX_PATH ($TOOK s); manifest $MLX_PATH.MANIFEST.json; record $MLX_PATH.CONVERTED.json"
