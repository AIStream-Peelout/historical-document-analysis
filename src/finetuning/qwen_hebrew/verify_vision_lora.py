# File name: verify_vision_lora.py
# Date: 8/9/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.
"""Post-hoc vision-LoRA check for a shipped/pushed Qwen3-VL adapter.

Every adapter through v1.7 shipped with all 108 vision ``lora_B`` tensors
exactly zero (Unsloth's gradient-checkpointing hook never reaches the
Qwen3-VL vision tower), i.e. the "vision fine-tune" was language-only.
requires_grad-based trainable-parameter printouts cannot catch this; only
the weights can. This script downloads (or reads locally) an adapter and
asserts, per tensor, whether the vision LoRA actually moved.

Run it against a checkpoint repo revision BEFORE spending anything on
evaluation:

    python -m src.finetuning.qwen_hebrew.verify_vision_lora \
        --repo isaacmg/qwen3-vl-8b-hebrew-v18b-ckpt --expect trained
    python -m src.finetuning.qwen_hebrew.verify_vision_lora \
        --repo isaacmg/qwen3-vl-8b-hebrew-v18a-ckpt --expect frozen

Exits nonzero when the adapter does not match the expectation.
"""
import argparse
import sys
from pathlib import Path

import dotenv

dotenv.load_dotenv()

EXPECTED_VISION_TENSORS = 108   # tower blocks; v19b+ adapters add 8 merger tensors
EXPECTED_MERGER_TENSORS = 8


def summarize_adapter(path: Path) -> dict:
    """Count nonzero vision/merger/language lora_B tensors in an adapter.

    The merger family (``visual.merger`` + ``visual.deepstack_merger_list``)
    is reported separately from the tower: v19b was the first run to adapt
    it, and pre-v19b adapters legitimately have zero merger tensors.

    :param path: Path to ``adapter_model.safetensors``.
    :returns: Dict with totals and per-family nonzero counts plus the max
        absolute vision update magnitude.
    """
    from safetensors.torch import load_file

    state = load_file(str(path))
    vis_b, mrg_b, lang_b = {}, {}, {}
    for n, t in state.items():
        if "lora_B" not in n:
            continue
        if ".visual." in n:
            (mrg_b if "merger" in n else vis_b)[n] = t
        else:
            lang_b[n] = t
    vis_nonzero = [n for n, t in vis_b.items() if t.abs().max().item() > 0]
    mrg_nonzero = [n for n, t in mrg_b.items() if t.abs().max().item() > 0]
    lang_nonzero = [n for n, t in lang_b.items() if t.abs().max().item() > 0]
    vis_max = max((t.abs().max().item() for t in vis_b.values()), default=0.0)
    return {
        "vision_total": len(vis_b),
        "vision_nonzero": len(vis_nonzero),
        "merger_total": len(mrg_b),
        "merger_nonzero": len(mrg_nonzero),
        "language_total": len(lang_b),
        "language_nonzero": len(lang_nonzero),
        "vision_max_abs": vis_max,
    }


def resolve_adapter(repo: str, revision: str, path: "Path | None") -> Path:
    """Locate the adapter file locally or download it from the hub.

    :param repo: HF repo id (ignored when ``path`` is given).
    :param revision: Revision to download (default branch when empty).
    :param path: Local adapter file or directory; ``None`` to use the hub.
    :returns: Path to ``adapter_model.safetensors``.
    """
    if path is not None:
        return path if path.is_file() else path / "adapter_model.safetensors"
    import os

    from huggingface_hub import hf_hub_download

    token = os.getenv("HF1_TOKEN") or os.getenv("HF_TOKEN")
    for name in ("last-checkpoint/adapter_model.safetensors",
                 "adapter_model.safetensors"):
        try:
            return Path(hf_hub_download(repo, name, revision=revision or None,
                                        token=token))
        except Exception:
            continue
    raise FileNotFoundError(f"no adapter_model.safetensors in {repo}@{revision}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="", help="HF adapter/checkpoint repo id")
    parser.add_argument("--revision", default="", help="repo revision (full SHA)")
    parser.add_argument("--path", type=Path, default=None,
                        help="local adapter file or directory (overrides --repo)")
    parser.add_argument("--expect", choices=("trained", "frozen"), required=True,
                        help="trained: all vision lora_B nonzero; frozen: all zero")
    args = parser.parse_args()
    if args.path is None and not args.repo:
        parser.error("give --repo or --path")

    adapter = resolve_adapter(args.repo, args.revision, args.path)
    s = summarize_adapter(adapter)
    print(f"{adapter}\n  tower lora_B  : {s['vision_nonzero']}/{s['vision_total']} "
          f"nonzero (max |ΔB| {s['vision_max_abs']:.3e})\n"
          f"  merger lora_B : {s['merger_nonzero']}/{s['merger_total']} nonzero\n"
          f"  language lora_B: {s['language_nonzero']}/{s['language_total']} nonzero")

    if s["vision_total"] != EXPECTED_VISION_TENSORS:
        print(f"FAIL: expected {EXPECTED_VISION_TENSORS} tower lora_B tensors, "
              f"found {s['vision_total']}")
        sys.exit(1)
    if s["merger_total"] not in (0, EXPECTED_MERGER_TENSORS):
        print(f"FAIL: expected 0 or {EXPECTED_MERGER_TENSORS} merger lora_B "
              f"tensors, found {s['merger_total']}")
        sys.exit(1)
    if args.expect == "trained" and s["vision_nonzero"] != s["vision_total"]:
        print("FAIL: vision LoRA did not train (the v1.5–v1.7 failure mode)")
        sys.exit(1)
    if args.expect == "trained" and s["merger_nonzero"] != s["merger_total"]:
        print("FAIL: merger LoRA present but did not train")
        sys.exit(1)
    if args.expect == "frozen" and s["vision_nonzero"] + s["merger_nonzero"] != 0:
        print("FAIL: vision/merger LoRA moved but this arm expected it frozen")
        sys.exit(1)
    print(f"OK: adapter matches --expect {args.expect}")
