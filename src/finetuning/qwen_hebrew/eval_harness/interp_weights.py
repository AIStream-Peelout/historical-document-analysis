# File name: interp_weights.py
# Date: 9/28/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Linear weight interpolation between two merged (bf16) checkpoints of the same architecture.

``theta = alpha * A + (1 - alpha) * B`` tensor by tensor (WiSE-FT style).  Both inputs are
HF ``save_pretrained`` directories with sharded safetensors; the output copies every
non-weight file (config, processor, tokenizer, chat template, generation config) from ``A``
and writes the interpolated weights with the same shard layout as ``A``.

Usage::

    .venv/bin/python -m src.finetuning.qwen_hebrew.eval_harness.interp_weights \\
        --a /Volumes/home/studio_offload/v19b_merge/qwen3-vl-8b-heb-v21b-step1200-bf16 \\
        --b /Volumes/home/studio_offload/v19b_merge/qwen3-vl-8b-heb-v22a-step1300-bf16 \\
        --alpha 0.5 --out /path/to/qwen3-vl-8b-heb-v22i-a50-bf16
"""
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def shard_map(d: Path) -> Dict[str, str]:
    """Tensor name -> shard file name for a safetensors directory.

    :param d: Model directory.
    :type d: Path
    :return: Mapping from the index file, or every tensor of the single shard.
    :rtype: Dict[str, str]
    """
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        return json.load(open(idx))["weight_map"]
    single = d / "model.safetensors"
    with safe_open(str(single), framework="pt") as f:
        return {k: "model.safetensors" for k in f.keys()}


def interpolate(a: Path, b: Path, alpha: float, out: Path) -> Dict[str, int]:
    """Write ``alpha*A + (1-alpha)*B`` to ``out``.

    :param a: First checkpoint directory (its non-weight files are copied).
    :type a: Path
    :param b: Second checkpoint directory.
    :type b: Path
    :param alpha: Weight on ``a`` in [0, 1].
    :type alpha: float
    :param out: Output directory (created).
    :type out: Path
    :return: Counts: tensors interpolated, tensors identical, shards written.
    :rtype: Dict[str, int]
    :raises ValueError: On mismatched tensor sets, shapes or dtypes.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be within [0, 1]")
    ma, mb = shard_map(a), shard_map(b)
    if set(ma) != set(mb):
        only_a, only_b = sorted(set(ma) - set(mb))[:5], sorted(set(mb) - set(ma))[:5]
        raise ValueError(f"tensor sets differ: only in A {only_a}, only in B {only_b}")
    out.mkdir(parents=True, exist_ok=True)
    for f in a.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            shutil.copy2(f, out / f.name)
    by_shard: Dict[str, List[str]] = {}
    for name, shard in ma.items():
        by_shard.setdefault(shard, []).append(name)
    stats = {"interpolated": 0, "identical": 0, "shards": 0}
    handles_b: Dict[str, safe_open] = {}
    for shard, names in sorted(by_shard.items()):
        tensors: Dict[str, torch.Tensor] = {}
        with safe_open(str(a / shard), framework="pt") as fa:
            for name in names:
                ta = fa.get_tensor(name)
                sb = mb[name]
                if sb not in handles_b:
                    handles_b[sb] = safe_open(str(b / sb), framework="pt")
                tb = handles_b[sb].get_tensor(name)
                if ta.shape != tb.shape or ta.dtype != tb.dtype:
                    raise ValueError(f"{name}: shape/dtype mismatch {tuple(ta.shape)}/{ta.dtype} vs {tuple(tb.shape)}/{tb.dtype}")
                if torch.equal(ta, tb):
                    tensors[name] = ta.contiguous(); stats["identical"] += 1
                else:
                    t = (ta.float() * alpha + tb.float() * (1.0 - alpha)).to(ta.dtype)
                    tensors[name] = t.contiguous(); stats["interpolated"] += 1
        save_file(tensors, str(out / shard), metadata={"format": "pt"})
        stats["shards"] += 1
        print(f"  wrote {shard}: {len(tensors)} tensors", flush=True)
    idx = a / "model.safetensors.index.json"
    if idx.exists():
        shutil.copy2(idx, out / idx.name)
    meta = {"a": str(a), "b": str(b), "alpha_on_a": alpha, **stats}
    (out / "interpolation.json").write_text(json.dumps(meta, indent=1))
    return stats


def main(argv: Optional[List[str]] = None) -> None:
    """CLI.

    :param argv: Arguments.
    :type argv: Optional[List[str]]
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True, help="checkpoint A (weight alpha; non-weight files copied from here)")
    ap.add_argument("--b", type=Path, required=True, help="checkpoint B (weight 1-alpha)")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    stats = interpolate(args.a, args.b, args.alpha, args.out)
    print(json.dumps({"out": str(args.out), **stats}))


if __name__ == "__main__":
    main()
