#!/bin/zsh
# Publish chain for the live playground (run when ready; safe to re-run — every step is idempotent):
#   1. wait for the genizah_ktiv_v3 push to finish (shares the uplink)
#   2. upload the merged v2.0a bf16 master (NAS) to the PUBLIC model repo isaacmg/qwen3-vl-8b-hebrew-v20a-merged
#   3. create the PRIVATE ZeroGPU Space isaacmg/genizah-reader from spaces/genizah_reader/ and poll its build
# Flip the Space to public in its settings once it runs the way you want it.
REPO=/Users/isaac/Documents/GitHub/historical-document-analysis
cd $REPO
echo "===== waiting for the dataset push to finish $(date '+%F %T') ====="
while pgrep -f "push_ktiv_dataset --src" >/dev/null; do sleep 120; done
echo "===== uploading merged model $(date '+%F %T') ====="
.venv/bin/python - <<'PY'
import os
from dotenv import load_dotenv; load_dotenv('/Users/isaac/Documents/GitHub/historical-document-analysis/.env')
from huggingface_hub import HfApi
api = HfApi(token=os.environ['HF1_TOKEN'])
MODEL = "isaacmg/qwen3-vl-8b-hebrew-v20a-merged"
SRC = "/Volumes/home/studio_offload/v19b_merge/qwen3-vl-8b-heb-v20a-step1800-bf16"
api.create_repo(MODEL, repo_type="model", private=False, exist_ok=True)
# upload_large_folder resumes and multi-commits reliably; plain upload_folder
# mis-skipped as "nothing modified" against a fresh repo when the weights'
# XET blobs already existed server-side (from the ckpt repo).
api.upload_large_folder(folder_path=SRC, repo_id=MODEL, repo_type="model")
shards = [f for f in api.list_repo_files(MODEL, repo_type="model") if f.endswith(".safetensors")]
assert len(shards) == 4, f"model upload incomplete: {len(shards)}/4 safetensors committed. " \
    f"Retry, or run: hf upload {MODEL} {SRC} ."
print("model uploaded:", MODEL, f"{len(shards)} safetensors + config committed", flush=True)
PY
echo "===== creating the Space (private, ZeroGPU) $(date '+%F %T') ====="
[ -f spaces/genizah_reader/README.md ] || { echo "ABORT: Space README missing"; exit 1; }
.venv/bin/python - <<'PY'
import os, time
from dotenv import load_dotenv; load_dotenv('/Users/isaac/Documents/GitHub/historical-document-analysis/.env')
from huggingface_hub import HfApi
api = HfApi(token=os.environ['HF1_TOKEN'])
SPACE = "isaacmg/genizah-reader"
api.create_repo(SPACE, repo_type="space", space_sdk="gradio", space_hardware="zero-a10g",
                private=True, exist_ok=True)
api.upload_folder(folder_path="spaces/genizah_reader", repo_id=SPACE, repo_type="space",
                  ignore_patterns=["__pycache__/*"],
                  commit_message="Genizah Reader: ZeroGPU demo of v2.0a (transcribe / lines with boxes / locate)")
try:
    api.request_space_hardware(SPACE, "zero-a10g")
except Exception as e:
    print("hardware request:", type(e).__name__, str(e)[:120])
for _ in range(40):
    rt = api.get_space_runtime(SPACE)
    print("runtime:", rt.stage, "hardware:", rt.hardware, flush=True)
    if rt.stage in ("RUNNING", "RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR"):
        break
    time.sleep(30)
print("space url: https://huggingface.co/spaces/" + SPACE, flush=True)
PY
echo "===== DONE $(date '+%F %T') ====="
