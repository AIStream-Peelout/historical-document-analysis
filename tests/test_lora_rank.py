# File name: test_lora_rank.py
# Date: 10/5/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Tests for the function-preserving LoRA rank expansion, on a real PEFT model."""
import pytest
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch import nn

from src.finetuning.qwen_hebrew.lora_rank import expand_lora_state


class Tiny(nn.Module):
    """Two linear layers and a convolution, each a LoRA target."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16, 24)
        self.fc2 = nn.Linear(24, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.tanh(self.fc1(self.conv(x).flatten(1))))


def _adapted(rank: int, seed: int) -> nn.Module:
    """A Tiny model (fixed base weights) with a rank-``rank`` adapter at scale 1."""
    torch.manual_seed(0)
    base = Tiny()
    torch.manual_seed(seed)
    return get_peft_model(base, LoraConfig(r=rank, lora_alpha=rank, target_modules=["conv", "fc1", "fc2"], lora_dropout=0.0))


def _trained_state(rank: int = 4) -> dict:
    """State of a rank-4 adapter whose B matrices are no longer zero."""
    model = _adapted(rank, seed=1)
    torch.manual_seed(2)
    for name, p in model.named_parameters():
        if "lora_B" in name:
            p.data.normal_(0, 0.3)
    return {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}, model


def test_expanded_adapter_computes_what_the_small_one_did():
    warm, small = _trained_state(4)
    big = _adapted(16, seed=7)
    fresh = {k: v.detach().clone() for k, v in get_peft_model_state_dict(big).items()}
    expanded = expand_lora_state(warm, fresh)
    result = set_peft_model_state_dict(big, expanded)
    assert not getattr(result, "unexpected_keys", [])
    x = torch.randn(6, 3, 4, 4)
    with torch.no_grad():
        assert torch.allclose(big(x), small(x), atol=1e-6)
    torch.manual_seed(0)
    assert not torch.allclose(big(x), Tiny()(x), atol=1e-3)            # the adapter does something: the test is not vacuous


def test_warm_rows_first_fresh_rows_kept_new_columns_zero():
    warm, _ = _trained_state(4)
    fresh = {k: v.detach().clone() for k, v in get_peft_model_state_dict(_adapted(16, seed=7)).items()}
    expanded = expand_lora_state(warm, fresh)
    assert set(expanded) == set(warm) == set(fresh)
    for key, w in warm.items():
        e, f = expanded[key], fresh[key]
        assert e.shape == f.shape
        if "lora_A" in key:
            assert torch.equal(e[:4], w) and torch.equal(e[4:], f[4:]) and float(e[4:].abs().max()) > 0
        else:
            assert torch.equal(e[:, :4], w) and float(e[:, 4:].abs().max()) == 0
    assert all(torch.equal(fresh[k], v) for k, v in get_peft_model_state_dict(_adapted(16, seed=7)).items())   # inputs untouched


def test_same_rank_and_unknown_keys_pass_through():
    warm, _ = _trained_state(4)
    same = expand_lora_state(warm, warm)
    assert all(same[k] is warm[k] for k in warm)
    extra = {**warm, "base_model.model.merger.weight": torch.ones(2, 2)}
    fresh = {k: v.detach().clone() for k, v in get_peft_model_state_dict(_adapted(16, seed=7)).items()}
    assert expand_lora_state(extra, fresh)["base_model.model.merger.weight"] is extra["base_model.model.merger.weight"]


def test_a_tensor_that_does_not_fit_is_an_error():
    warm, _ = _trained_state(4)
    smaller = {k: v.detach().clone() for k, v in get_peft_model_state_dict(_adapted(2, seed=7)).items()}
    with pytest.raises(ValueError, match="does not fit"):
        expand_lora_state(warm, smaller)                                  # a rank cannot shrink
    key = next(k for k in warm if "lora_A" in k and "fc1" in k)
    fresh = {k: v.detach().clone() for k, v in get_peft_model_state_dict(_adapted(16, seed=7)).items()}
    fresh[key] = torch.zeros(16, 99)                                      # another layer width
    with pytest.raises(ValueError, match="does not fit"):
        expand_lora_state(warm, fresh)
