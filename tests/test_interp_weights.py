# File name: test_interp_weights.py
# Date: 9/28/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Unit tests for the bf16 weight interpolation helper."""
import json

import pytest
import torch
from safetensors.torch import save_file

from src.finetuning.qwen_hebrew.eval_harness import interp_weights as iw


def _write_ckpt(d, tensors, shards=1):
    d.mkdir()
    names = list(tensors)
    per = max(1, len(names) // shards)
    weight_map = {}
    for i in range(shards):
        part = names[i * per:(i + 1) * per] if i < shards - 1 else names[i * per:]
        fname = f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
        save_file({n: tensors[n] for n in part}, str(d / fname), metadata={"format": "pt"})
        weight_map.update({n: fname for n in part})
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (d / "config.json").write_text('{"model_type": "test"}')
    (d / "preprocessor_config.json").write_text('{"min_pixels": 6500000}')


def test_interpolation_is_exact_average_and_keeps_shared_tensors(tmp_path):
    a = {"w1": torch.full((4, 3), 2.0, dtype=torch.bfloat16), "w2": torch.ones(5, dtype=torch.bfloat16), "same": torch.arange(6, dtype=torch.bfloat16)}
    b = {"w1": torch.full((4, 3), 4.0, dtype=torch.bfloat16), "w2": torch.zeros(5, dtype=torch.bfloat16), "same": torch.arange(6, dtype=torch.bfloat16)}
    _write_ckpt(tmp_path / "A", a, shards=2)
    _write_ckpt(tmp_path / "B", b, shards=1)  # different shard layout on purpose
    out = tmp_path / "out"
    stats = iw.interpolate(tmp_path / "A", tmp_path / "B", 0.5, out)
    assert stats == {"interpolated": 2, "identical": 1, "shards": 2}
    got = iw.shard_map(out)
    assert set(got) == {"w1", "w2", "same"} and (out / "config.json").exists() and (out / "preprocessor_config.json").exists()
    from safetensors import safe_open
    tensors = {}
    for shard in set(got.values()):
        with safe_open(str(out / shard), framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    assert torch.equal(tensors["w1"], torch.full((4, 3), 3.0, dtype=torch.bfloat16))
    assert torch.equal(tensors["w2"], torch.full((5,), 0.5, dtype=torch.bfloat16))
    assert torch.equal(tensors["same"], torch.arange(6, dtype=torch.bfloat16))
    meta = json.load(open(out / "interpolation.json"))
    assert meta["alpha_on_a"] == 0.5


def test_alpha_one_reproduces_a(tmp_path):
    a = {"w": torch.randn(3, 3).to(torch.bfloat16)}
    b = {"w": torch.randn(3, 3).to(torch.bfloat16)}
    _write_ckpt(tmp_path / "A", a); _write_ckpt(tmp_path / "B", b)
    iw.interpolate(tmp_path / "A", tmp_path / "B", 1.0, tmp_path / "out")
    from safetensors import safe_open
    with safe_open(str(tmp_path / "out" / "model-00001-of-00001.safetensors"), framework="pt") as f:
        assert torch.equal(f.get_tensor("w"), a["w"])


def test_mismatches_raise(tmp_path):
    _write_ckpt(tmp_path / "A", {"w": torch.ones(2, dtype=torch.bfloat16)})
    _write_ckpt(tmp_path / "B", {"w": torch.ones(3, dtype=torch.bfloat16)})
    with pytest.raises(ValueError, match="mismatch"):
        iw.interpolate(tmp_path / "A", tmp_path / "B", 0.5, tmp_path / "out1")
    _write_ckpt(tmp_path / "C", {"other": torch.ones(2, dtype=torch.bfloat16)})
    with pytest.raises(ValueError, match="tensor sets differ"):
        iw.interpolate(tmp_path / "A", tmp_path / "C", 0.5, tmp_path / "out2")
    with pytest.raises(ValueError, match="alpha"):
        iw.interpolate(tmp_path / "A", tmp_path / "A", 1.5, tmp_path / "out3")
