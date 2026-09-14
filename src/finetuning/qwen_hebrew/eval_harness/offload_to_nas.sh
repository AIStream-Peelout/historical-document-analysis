#!/bin/zsh
# Move a cold local model directory to the NAS, verifying every file by size + SHA-1 BEFORE removing
# anything local.   offload_to_nas.sh <local_dir> <nas_dest_dir> [<extra_local_copy_to_delete> ...]
# Refuses to run if the model is currently loaded in LM Studio. Never touches anything else.
# (macOS ships openrsync: no --info/--itemize flags, hence the Python verifier.)
set -e
SRC=$1; DEST=$2; shift 2 || { echo "usage: offload_to_nas.sh <local_dir> <nas_dest_dir> [extra_copy ...]"; exit 1; }
[ -d "$SRC" ] || { echo "ABORT: $SRC missing"; exit 1; }
[ -d /Volumes/home/studio_offload ] || { echo "ABORT: NAS not mounted"; exit 1; }
NAME=$(basename "$SRC")
if ~/.lmstudio/bin/lms ps 2>/dev/null | grep -q "$NAME"; then echo "ABORT: $NAME is loaded in LM Studio"; exit 1; fi
echo "=== $(date '+%F %T') copy $SRC -> $DEST ($(du -sh "$SRC" | cut -f1)) ==="
mkdir -p "$DEST"
rsync -a "$SRC/" "$DEST/"
echo "=== $(date '+%F %T') verify (size + sha1 of every file) ==="
/Users/isaac/Documents/GitHub/historical-document-analysis/.venv/bin/python - "$SRC" "$DEST" <<'PY'
import hashlib, os, sys
src, dest = sys.argv[1], sys.argv[2]
def sha1(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""): h.update(chunk)
    return h.hexdigest()
n = 0
for root, _, files in os.walk(src):
    for f in files:
        a = os.path.join(root, f); b = os.path.join(dest, os.path.relpath(a, src))
        if not os.path.exists(b) or os.path.getsize(a) != os.path.getsize(b) or sha1(a) != sha1(b):
            print("MISMATCH:", os.path.relpath(a, src)); sys.exit(2)
        n += 1
print(f"verified {n} files identical")
PY
rm -rf "$SRC"; echo "removed $SRC"
for X in "$@"; do [ -d "$X" ] && { rm -rf "$X"; echo "removed $X"; }; done
df -h /System/Volumes/Data | tail -1 | awk '{print "local free now: "$4}'
echo "=== $(date '+%F %T') DONE $NAME ==="
