# File name: lora_rank.py
# Date: 10/5/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Warm-start a LoRA adapter of a higher rank from a lower-rank one without changing the model.

A LoRA layer adds ``scale * B @ A`` to a frozen weight, with ``A`` of shape ``[r, in]`` and ``B``
of shape ``[out, r]``. A trained rank-``r`` adapter fits into a rank-``R`` one (``R > r``) by
putting its rows of ``A`` and columns of ``B`` first, keeping the new adapter's fresh rows of ``A``
and setting the new columns of ``B`` to zero: the product is unchanged, so the model starts
exactly where the smaller adapter left it, and the extra ``R - r`` directions are free to learn.
The scale must be the same on both sides (``lora_alpha / r``: 16/16 and 64/64 are both 1).

The function is inlined into the training notebook by
:mod:`src.finetuning.qwen_hebrew.colab.derive_v23a` (its source is copied, so the notebook and
this module cannot drift apart).
"""
from typing import Dict, Mapping

import torch


def expand_lora_state(warm: Mapping[str, torch.Tensor], fresh: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Place a lower-rank LoRA state into a higher-rank adapter, function unchanged.

    :param warm: Saved state of the trained adapter (``get_peft_model_state_dict`` naming).
    :type warm: Mapping[str, torch.Tensor]
    :param fresh: State of the new adapter as created: ``lora_A`` initialised, ``lora_B`` zero.
    :type fresh: Mapping[str, torch.Tensor]
    :return: State to load into the new adapter. A tensor whose shape already matches (or whose key the
        new adapter does not have) is passed through; a ``lora_A`` tensor is the fresh one with the warm
        rows first; a ``lora_B`` tensor is zero with the warm columns first.
    :rtype: Dict[str, torch.Tensor]
    :raises ValueError: When a tensor differs in shape and is not a LoRA matrix that fits.
    """
    out: Dict[str, torch.Tensor] = {}
    for key, w in warm.items():
        f = fresh.get(key)
        if f is None or tuple(f.shape) == tuple(w.shape):
            out[key] = w
            continue
        t = f.detach().to("cpu", copy=True)
        if "lora_A" in key and w.shape[1:] == t.shape[1:] and w.shape[0] < t.shape[0]:
            t[: w.shape[0]] = w.to(t.dtype)
        elif "lora_B" in key and w.shape[0] == t.shape[0] and w.shape[2:] == t.shape[2:] and w.shape[1] < t.shape[1]:
            t.zero_()
            t[:, : w.shape[1]] = w.to(t.dtype)
        else:
            raise ValueError(f"{key}: trained shape {tuple(w.shape)} does not fit the new adapter's {tuple(t.shape)}")
        out[key] = t
    return out
