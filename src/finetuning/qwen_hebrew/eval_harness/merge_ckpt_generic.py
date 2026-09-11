"""Merge any series checkpoint (LoRA and/or modules_to_save) into base bf16.

Generalization of the v19b merge: handles v19c-style adapters whose merger
linears are full-weight clones (PEFT ``modules_to_save`` — ``merge_and_unload``
copies the clone over the original). Asserts the merger actually moved for
adapters that carry clone keys.

Disk-safe on the shared Studio: HF_HOME is pointed at the NAS so the 16GB
base download never lands on the local disk; the merged bf16 is written to
the NAS too.

Usage:
  merge_ckpt_generic.py --repo isaacmg/...-v19c-ckpt --revision <sha> \
      --out /Volumes/home/studio_offload/v19b_merge/<name>-bf16
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

NAS_UP = Path("/Volumes/home/studio_offload").exists() and not os.environ.get("FORCE_LOCAL")
if NAS_UP:
    os.environ.setdefault("HF_HOME", "/Volumes/home/studio_offload/hf_home_merge")
# NAS down: default local HF cache; the 16GB base cache is purged right after
# the model loads into RAM (before saving) so local disk never holds
# cache + merged bf16 simultaneously.
import dotenv

REPO = Path("/Users/isaac/Documents/GitHub/historical-document-analysis")
dotenv.load_dotenv(REPO / ".env")

import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: E402
from peft import PeftModel  # noqa: E402

BASE = "Qwen/Qwen3-VL-8B-Instruct"
V19A_PREPROC = REPO / "models/qwen3-vl-8b-heb-v19a-step1300/preprocessor_config.json"
NAS_PREPROC = Path("/Volumes/home/studio_offload/v19b_merge/qwen3-vl-8b-heb-v19b-step1300-bf16/preprocessor_config.json")
MIN_PIX, MAX_PIX = 6_500_000, 7_000_000

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--revision", required=True)
ap.add_argument("--out", required=True, type=Path)
args = ap.parse_args()
assert len(args.revision) == 40

token = os.environ["HF1_TOKEN"]
print("downloading adapter...", flush=True)
snap = snapshot_download(args.repo, revision=args.revision, token=token,
                         allow_patterns="last-checkpoint/*")
adapter_dir = Path(snap) / "last-checkpoint"
sd = load_file(adapter_dir / "adapter_model.safetensors")
has_merger_full = any("merger" in k and "lora" not in k for k in sd)
print(f"adapter: {len(sd)} tensors, merger full-weight keys: {has_merger_full}", flush=True)

print(f"loading base bf16 on CPU (cache on {'NAS' if NAS_UP else 'LOCAL, purged after load'})...",
      flush=True)
processor = AutoProcessor.from_pretrained(BASE, token=token)
# ALWAYS mmap-load (low_cpu_mem_usage=True): with the cache on a RAM-backed
# volume this is Metal-safe and halves process RAM — loading fully into RAM
# (=False) stacked with a large ramdisk caused the 2026-09-01 swap incident.
model = Qwen3VLForConditionalGeneration.from_pretrained(
    BASE, dtype=torch.bfloat16, low_cpu_mem_usage=True, token=token)
if not NAS_UP:
    _cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) \
        / "hub/models--Qwen--Qwen3-VL-8B-Instruct"
    if _cache.exists():
        shutil.rmtree(_cache)
        print(f"purged local base cache ({_cache})", flush=True)
ref = model.model.visual.merger.linear_fc1.weight.detach().clone()

print("attaching + merging adapter...", flush=True)
model = PeftModel.from_pretrained(model, adapter_dir)
model = model.merge_and_unload()

merged_w = model.model.visual.merger.linear_fc1.weight.detach()
delta = (merged_w.float() - ref.float()).abs().max().item()
print(f"merger linear_fc1 max |Δ| vs base after merge: {delta:.3e}", flush=True)
if has_merger_full:
    assert delta > 0, ("adapter carries merger clones but merged weights equal "
                       "base — modules_to_save merge failed")

args.out.mkdir(parents=True, exist_ok=True)
print(f"saving merged bf16 to {args.out} ...", flush=True)
model.save_pretrained(args.out, safe_serialization=True)
processor.save_pretrained(args.out)

# byte-identical resolution contract with the served v19-era models
src_preproc = NAS_PREPROC if NAS_PREPROC.exists() else V19A_PREPROC
shutil.copy2(src_preproc, args.out / "preprocessor_config.json")
cfg = json.loads((args.out / "preprocessor_config.json").read_text())
assert cfg["min_pixels"] == MIN_PIX and cfg["max_pixels"] == MAX_PIX, cfg
mcfg = json.loads((args.out / "config.json").read_text())
tcfg = mcfg.get("text_config", mcfg)
assert "rope_theta" in tcfg, "rope_theta missing — tf5-style config, needs patch"
print("merge complete + processor stamped 6.5/7MP + rope_theta OK", flush=True)
sys.exit(0)
