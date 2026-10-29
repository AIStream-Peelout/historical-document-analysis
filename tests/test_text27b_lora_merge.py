# File name: test_text27b_lora_merge.py
# Date: 10/27/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for text27b ``lora_merge``: adapter-config scale, PEFT key mapping, merge planning, the per-tensor merge,
the streaming shard writer (byte equality with ``save_file``, verbatim headers, contiguity), the driver (files,
index, sizes, report, verbatim shards, meta dir), resumption (record before rename, truncated progress lines, stale
index, re-hashing), the dry run, the refusals and the peak-RSS estimate against a real subprocess.

Everything runs on tiny synthetic models written to ``tmp_path``: no downloads, no machine paths.
"""
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import torch
from safetensors.torch import load_file, save_file

from src.finetuning.text27b import lora_merge as LM

#: The t27b training regex (logs/text_finetune_27b/genizah_t27_lambda.py, adapter_config.json).
REAL_REGEX = (r"(?!.*(?:visual|vision|mtp)).*\.layers\.\d+\.(?:linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)|"
              r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))")
SHARD1 = "model-00001-of-00002.safetensors"
SHARD2 = "model-00002-of-00002.safetensors"
QKV = "model.layers.0.linear_attn.in_proj_qkv.weight"
DOWN = "model.layers.0.mlp.down_proj.weight"
VISUAL = "model.visual.layers.0.mlp.down_proj.weight"
MTP = "mtp.layers.0.mlp.down_proj.weight"
NORM = "model.layers.0.input_layernorm.weight"
EMBED = "model.embed_tokens.weight"
LM_HEAD = "lm_head.weight"
#: (out, in) of every 2-D tensor in the synthetic model.
SHAPES = {EMBED: (8, 4), QKV: (6, 4), DOWN: (4, 5), VISUAL: (4, 5), MTP: (4, 5), LM_HEAD: (8, 4)}
LAYOUT = {SHARD1: [EMBED, NORM, QKV], SHARD2: [DOWN, VISUAL, MTP, LM_HEAD]}
RANK = 16
COPIED = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
          "merges.txt", "vocab.json"]
NOT_COPIED = ["README.md", ".gitattributes", "crc32.txt"]


def w_value(i: int, j: int, cols: int, salt: int) -> float:
    """Base weight entry (quarter steps in [-1, 1], exact in bf16).

    :param i: Row.
    :param j: Column.
    :param cols: Columns.
    :param salt: Per-tensor offset.
    :returns: Value.
    """
    return (((i * cols + j + salt) % 9) - 4) / 4


def base_lists() -> Dict[str, List[List[float]]]:
    """Plain-Python base weights of the synthetic model (the norm is 1-D).

    :returns: ``{key: rows}``.
    """
    weights = {key: [[w_value(i, j, cols, salt) for j in range(cols)] for i in range(rows)]
               for salt, (key, (rows, cols)) in enumerate(sorted(SHAPES.items()))}
    weights[NORM] = [[1.0, 0.5, -0.25, 2.0]]
    return weights


def a_rows(cols: int) -> List[List[float]]:
    """Non-zero rows 0 and 1 of ``lora_A`` (the other 14 rows are zero).

    :param cols: In-features.
    :returns: Two rows.
    """
    return [[(j % 3 - 1) * 0.5 for j in range(cols)], [0.25 * (j % 2) for j in range(cols)]]


def b_cols(rows: int) -> List[List[float]]:
    """Non-zero columns 0 and 1 of ``lora_B`` (the other 14 columns are zero).

    :param rows: Out-features.
    :returns: Two columns.
    """
    return [[(i % 3) * 0.25 for i in range(rows)], [(-0.5 if i % 2 else 0.5) for i in range(rows)]]


def lora_tensors(rows: int, cols: int) -> Dict[str, torch.Tensor]:
    """float32 ``A`` ``[16, in]`` and ``B`` ``[out, 16]`` with two non-zero components.

    :param rows: Out-features.
    :param cols: In-features.
    :returns: ``{"A": ..., "B": ...}``.
    """
    a = torch.zeros(RANK, cols, dtype=torch.float32)
    b = torch.zeros(rows, RANK, dtype=torch.float32)
    for k, row in enumerate(a_rows(cols)):
        a[k] = torch.tensor(row)
    for k, col in enumerate(b_cols(rows)):
        b[:, k] = torch.tensor(col)
    return {"A": a, "B": b}


def hand_merged(key: str, scale: float) -> List[List[float]]:
    """Expected merged weight computed with plain Python arithmetic.

    :param key: Targeted weight key.
    :param scale: LoRA scale.
    :returns: Rows of ``W + scale * sum_k B[:, k] A[k, :]``.
    """
    rows, cols = SHAPES[key]
    w, a, b = base_lists()[key], a_rows(cols), b_cols(rows)
    return [[w[i][j] + scale * (b[0][i] * a[0][j] + b[1][i] * a[1][j]) for j in range(cols)] for i in range(rows)]


def adapter_config(alpha: int = 32, rslora: bool = False, **extra: Any) -> Dict[str, Any]:
    """An adapter config shaped like the real t27b one.

    :param alpha: ``lora_alpha``.
    :param rslora: ``use_rslora``.
    :param extra: Overrides.
    :returns: Config dict.
    """
    config = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": RANK, "lora_alpha": alpha, "use_rslora": rslora,
              "target_modules": REAL_REGEX, "rank_pattern": {}, "alpha_pattern": {}, "modules_to_save": None,
              "use_dora": False, "bias": "none", "fan_in_fan_out": False, "lora_bias": False, "use_qalora": False,
              "qalora_group_size": 16, "layers_to_transform": None, "layers_pattern": None, "exclude_modules": None,
              "target_parameters": None, "trainable_token_indices": None, "base_model_name_or_path": "synthetic",
              "peft_version": "0.21.2"}
    config.update(extra)
    return config


def make_base(root: Path, metadata: Optional[Dict[str, str]] = None) -> Path:
    """Write the synthetic two-shard bf16 base with tokenizer/config files.

    :param root: Parent directory.
    :param metadata: Shard metadata (default ``{"format": "pt"}``).
    :returns: The base directory.
    """
    base = root / "base"
    base.mkdir()
    weights = base_lists()
    weight_map = {}
    total = 0
    for shard, keys in LAYOUT.items():
        tensors = {}
        for key in keys:
            tensor = torch.tensor(weights[key], dtype=torch.bfloat16)
            tensors[key] = tensor[0].clone() if key == NORM else tensor
            weight_map[key] = shard
            total += tensors[key].numel() * 2
        save_file(tensors, str(base / shard), metadata=metadata or {"format": "pt"})
    (base / LM.INDEX_NAME).write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map},
                                                 indent=2) + "\n")
    for name in COPIED + NOT_COPIED:
        (base / name).write_text(f"content of {name}\n")
    (base / ".cache").mkdir()
    (base / ".cache" / "lock").write_text("x")
    return base


def make_adapter(root: Path, name: str = "adapter", keys: Optional[List[str]] = None,
                 prefix: str = "base_model.model.", config: Optional[Dict[str, Any]] = None,
                 extra: Optional[Dict[str, torch.Tensor]] = None) -> Path:
    """Write a PEFT-style adapter directory.

    :param root: Parent directory.
    :param name: Directory name.
    :param keys: Weights that get a LoRA pair (default the two real targets).
    :param prefix: Key prefix.
    :param config: Adapter config (default :func:`adapter_config`).
    :param extra: Additional raw adapter tensors.
    :returns: The adapter directory.
    """
    adapter = root / name
    adapter.mkdir()
    tensors = {}
    for key in keys if keys is not None else [QKV, DOWN]:
        module = key[: -len(".weight")]
        pair = lora_tensors(*SHAPES[key])
        tensors[f"{prefix}{module}.lora_A.weight"] = pair["A"]
        tensors[f"{prefix}{module}.lora_B.weight"] = pair["B"]
    tensors.update(extra or {})
    save_file(tensors, str(adapter / LM.ADAPTER_WEIGHTS), metadata={"format": "pt"})
    (adapter / LM.ADAPTER_CONFIG).write_text(json.dumps(config or adapter_config()))
    return adapter


def raw_bytes(tensor: torch.Tensor) -> bytes:
    """The tensor's storage bytes.

    :param tensor: Tensor.
    :returns: Bytes.
    """
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


# --------------------------------------------------------------------------------------------------- config / keys

def test_lora_scale_plain_rslora_and_t27b() -> None:
    """alpha/r, alpha/sqrt(r), and the t27b r=16 alpha=16 config."""
    assert LM.lora_scale(adapter_config(alpha=32)) == 2.0
    assert LM.lora_scale(adapter_config(alpha=32, rslora=True)) == 8.0
    assert LM.lora_scale(adapter_config(alpha=16)) == 1.0


@pytest.mark.parametrize("field,value", [("rank_pattern", {"x": 8}), ("alpha_pattern", {"x": 8}), ("use_dora", True),
                                         ("modules_to_save", ["lm_head"]), ("fan_in_fan_out", True),
                                         ("bias", "all"), ("peft_type", "LOHA")])
def test_lora_scale_rejects_unsupported(field: str, value: Any) -> None:
    """Unsupported adapter features raise.

    :param field: Config field.
    :param value: Unsupported value.
    """
    with pytest.raises(ValueError):
        LM.lora_scale(adapter_config(**{field: value}))


def test_module_name_and_parse_adapter_key() -> None:
    """Prefixed and bare PEFT keys map to the base weight key; the role is A or B."""
    key = "base_model.model.model.layers.3.mlp.down_proj.lora_A.weight"
    assert LM.module_name(key) == "model.layers.3.mlp.down_proj.weight"
    assert LM.parse_adapter_key(key) == ("model.layers.3.mlp.down_proj.weight", "A")
    bare = "model.language_model.layers.3.linear_attn.in_proj_qkv.lora_B.weight"
    assert LM.parse_adapter_key(bare) == ("model.language_model.layers.3.linear_attn.in_proj_qkv.weight", "B")
    assert LM.module_name("custom.x.q_proj.lora_A.weight", prefixes=("custom.",)) == "x.q_proj.weight"


@pytest.mark.parametrize("key", ["base_model.model.model.layers.0.mlp.down_proj.lora_A.default.weight",
                                 "base_model.model.model.embed_tokens.lora_embedding_A.weight",
                                 "base_model.model.model.layers.0.mlp.down_proj.lora_magnitude_vector.weight",
                                 "base_model.model.model.layers.0.mlp.down_proj.bias"])
def test_parse_adapter_key_rejects_other_keys(key: str) -> None:
    """Keys that are not saved lora_A/lora_B weights raise.

    :param key: Adapter key.
    """
    with pytest.raises(ValueError):
        LM.parse_adapter_key(key)


def test_real_regex_selects_only_language_model_projections() -> None:
    """The negative lookahead skips visual and mtp; norms and embeddings are not targets."""
    keys = [EMBED, NORM, QKV, DOWN, VISUAL, MTP, LM_HEAD, "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_norm.weight", "model.layers.0.linear_attn.conv1d.weight"]
    assert LM.targeted_weight_keys(keys, REAL_REGEX) == [QKV, DOWN, "model.layers.0.self_attn.q_proj.weight"]


def test_target_regex_from_list_config() -> None:
    """A list target_modules matches modules equal to or ending in ``.<name>``."""
    regex = LM.target_regex_from_config(adapter_config(target_modules=["down_proj", "in_proj_qkv"]))
    assert LM.targeted_weight_keys([QKV, DOWN, VISUAL, NORM, "down_proj.weight"], regex) == sorted(
        [QKV, DOWN, VISUAL, "down_proj.weight"])


# ------------------------------------------------------------------------------------------------------- plan_merge

def adapter_keys(keys: List[str], prefix: str = "base_model.model.") -> List[str]:
    """Adapter A/B keys for weights.

    :param keys: Weight keys.
    :param prefix: Key prefix.
    :returns: Keys.
    """
    return [f"{prefix}{key[:-len('.weight')]}.lora_{role}.weight" for key in keys for role in "AB"]


def index_for() -> Dict[str, Any]:
    """Index of the synthetic layout.

    :returns: Index dict.
    """
    return {"weight_map": {key: shard for shard, names in LAYOUT.items() for key in names}}


def test_plan_merge_real_regex() -> None:
    """Pairs map to their shard and A/B keys."""
    plan = LM.plan_merge(index_for(), adapter_keys([QKV, DOWN]), REAL_REGEX)
    assert plan == {
        DOWN: (SHARD2, "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight",
               "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight"),
        QKV: (SHARD1, "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_A.weight",
              "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_B.weight"),
    }
    assert LM.plan_merge(index_for(), adapter_keys([QKV, DOWN], prefix=""), REAL_REGEX).keys() == plan.keys()


def test_plan_merge_unused_pair_raises() -> None:
    """An adapter pair whose base weight does not exist raises."""
    with pytest.raises(ValueError, match="no base weight"):
        LM.plan_merge(index_for(), adapter_keys([QKV, DOWN, "model.layers.0.mlp.up_proj.weight"]), REAL_REGEX)


def test_plan_merge_pair_outside_regex_raises() -> None:
    """A pair for a weight the regex skips (visual) raises instead of being merged or dropped."""
    with pytest.raises(ValueError, match="does not select"):
        LM.plan_merge(index_for(), adapter_keys([QKV, DOWN, VISUAL]), REAL_REGEX)


def test_plan_merge_target_without_adapter_raises() -> None:
    """A targeted base weight without a pair raises."""
    with pytest.raises(ValueError, match="no adapter pair"):
        LM.plan_merge(index_for(), adapter_keys([QKV]), REAL_REGEX)


def test_plan_merge_pair_matched_twice_raises() -> None:
    """Two keys resolving to the same weight and role raise."""
    keys = adapter_keys([QKV, DOWN]) + ["model.layers.0.mlp.down_proj.lora_A.weight"]
    with pytest.raises(ValueError, match="matched twice"):
        LM.plan_merge(index_for(), keys, REAL_REGEX)


def test_plan_merge_incomplete_pair_raises() -> None:
    """A lora_A without its lora_B raises."""
    keys = adapter_keys([QKV]) + ["base_model.model.model.layers.0.mlp.down_proj.lora_A.weight"]
    with pytest.raises(ValueError, match="missing lora_A or lora_B"):
        LM.plan_merge(index_for(), keys, REAL_REGEX)


# ------------------------------------------------------------------------------------------------------ merge_shard

def shard_tensors(shard: str) -> Dict[str, torch.Tensor]:
    """In-memory bf16 tensors of one synthetic shard.

    :param shard: Shard name.
    :returns: Tensors.
    """
    weights = base_lists()
    return {key: (torch.tensor(weights[key], dtype=torch.bfloat16)[0].clone() if key == NORM
                  else torch.tensor(weights[key], dtype=torch.bfloat16)) for key in LAYOUT[shard]}


def adapter_dict(keys: List[str]) -> Dict[str, torch.Tensor]:
    """In-memory adapter tensors for weights.

    :param keys: Weight keys.
    :returns: Tensors keyed like the adapter file.
    """
    out = {}
    for key in keys:
        pair = lora_tensors(*SHAPES[key])
        module = f"base_model.model.{key[:-len('.weight')]}"
        out[f"{module}.lora_A.weight"], out[f"{module}.lora_B.weight"] = pair["A"], pair["B"]
    return out


@pytest.mark.parametrize("scale", [2.0, 8.0])
def test_merge_shard_hand_computed(scale: float) -> None:
    """Targeted weight equals the hand computation; untouched tensors are the same objects.

    :param scale: LoRA scale (alpha 32 / r 16, and the rslora 32 / sqrt(16)).
    """
    tensors = shard_tensors(SHARD1)
    plan = LM.plan_merge(index_for(), adapter_keys([QKV, DOWN]), REAL_REGEX)
    pairs = {key: pair for key, pair in plan.items() if pair[0] == SHARD1}
    merged, stats = LM.merge_shard(tensors, pairs, adapter_dict([QKV, DOWN]), scale)
    assert torch.equal(merged[QKV], torch.tensor(hand_merged(QKV, scale), dtype=torch.bfloat16))
    assert merged[QKV].dtype == torch.bfloat16 and merged[QKV].shape == (6, 4)
    assert merged[EMBED] is tensors[EMBED] and merged[NORM] is tensors[NORM]
    assert torch.equal(tensors[QKV], torch.tensor(base_lists()[QKV], dtype=torch.bfloat16))  # input untouched
    assert [entry["key"] for entry in stats] == [QKV]
    entry = stats[0]
    assert entry["shape"] == [6, 4] and entry["rank"] == RANK and entry["dtype"] == "bfloat16"
    delta = torch.tensor(hand_merged(QKV, scale)) - torch.tensor(base_lists()[QKV])
    w_fro = float(torch.linalg.vector_norm(torch.tensor(base_lists()[QKV])))
    assert entry["max_abs_delta"] == pytest.approx(float(delta.abs().max()))
    assert entry["delta_fro_over_w_fro"] == pytest.approx(float(torch.linalg.vector_norm(delta)) / w_fro)
    assert entry["lora_fro_over_w_fro"] == pytest.approx(entry["delta_fro_over_w_fro"])  # exact in bf16
    assert entry["changed_fraction"] == pytest.approx(float((delta != 0).float().mean()))


def test_merge_tensor_matches_formula_bitwise_and_keeps_f32_input() -> None:
    """Random bf16 and f32 weights: result is bitwise the brief's formula; inputs are not modified."""
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(RANK, 33, generator=generator) * 0.01
    b = torch.randn(17, RANK, generator=generator) * 0.01
    for dtype in (torch.bfloat16, torch.float32):
        weight = torch.randn(17, 33, generator=generator).to(dtype)
        before = weight.clone()
        merged, _ = LM.merge_tensor(weight, a, b, 1.0, "w")
        expected = (weight.float() + 1.0 * (b.float() @ a.float())).to(weight.dtype)
        assert torch.equal(merged, expected) and merged.dtype == dtype
        assert torch.equal(weight, before)


def test_merge_tensor_shape_mismatch_raises() -> None:
    """A/B that do not fit W raise."""
    with pytest.raises(ValueError, match="incompatible"):
        LM.merge_tensor(torch.zeros(4, 5, dtype=torch.bfloat16), torch.zeros(RANK, 4), torch.zeros(4, RANK), 1.0)
    with pytest.raises(ValueError, match="not floating point"):
        LM.merge_tensor(torch.zeros(4, 5, dtype=torch.uint8), torch.zeros(RANK, 5), torch.zeros(4, RANK), 1.0)


# ---------------------------------------------------------------------------------------------------- merge_adapter

@pytest.mark.parametrize("rslora,scale", [(False, 2.0), (True, 8.0)])
def test_merge_adapter_end_to_end(tmp_path: Path, rslora: bool, scale: float) -> None:
    """Merged values, byte-identical untouched tensors, index, copied files, sizes and the report.

    :param tmp_path: Pytest temporary directory.
    :param rslora: Whether the adapter uses rsLoRA.
    :param scale: The expected scale.
    """
    base = make_base(tmp_path)
    adapter = make_adapter(tmp_path, config=adapter_config(alpha=32, rslora=rslora))
    out, report_path = tmp_path / "out", tmp_path / "reports" / "merge.json"
    report = LM.merge_adapter(base, adapter, out, report_path)

    merged = {**load_file(str(out / SHARD1)), **load_file(str(out / SHARD2))}
    original = {**load_file(str(base / SHARD1)), **load_file(str(base / SHARD2))}
    for key in (QKV, DOWN):
        assert torch.equal(merged[key], torch.tensor(hand_merged(key, scale), dtype=torch.bfloat16))
    for key in (EMBED, NORM, VISUAL, MTP, LM_HEAD):
        assert merged[key].dtype == torch.bfloat16 and raw_bytes(merged[key]) == raw_bytes(original[key])
    assert (out / LM.INDEX_NAME).read_bytes() == (base / LM.INDEX_NAME).read_bytes()
    for shard in LAYOUT:
        assert (out / shard).stat().st_size == (base / shard).stat().st_size
        assert not (out / f"{shard}.partial").exists()
    for name in COPIED:
        assert (out / name).read_text() == f"content of {name}\n"
    for name in NOT_COPIED + [".cache"]:
        assert not (out / name).exists()

    assert json.loads(report_path.read_text()) == report
    assert report["n_targets"] == 2 and report["scale"] == scale and report["n_shards"] == 2
    assert [entry["key"] for entry in report["tensors"]] == sorted([QKV, DOWN])
    assert {row["file"]: row["status"] for row in report["shards"]} == {SHARD1: "merged", SHARD2: "merged"}
    assert report["resumed_shards"] == [] and sorted(report["copied_files"]) == sorted(COPIED)
    assert report["total_bytes"] == sum((base / shard).stat().st_size for shard in LAYOUT)
    assert report["wall_seconds"] >= 0 and report["fingerprint"]["scale"] == scale


def test_shard_without_targets_is_copied_byte_for_byte(tmp_path: Path) -> None:
    """With a regex selecting only down_proj, shard 1 has no target and is copied unchanged.

    :param tmp_path: Pytest temporary directory.
    """
    base = make_base(tmp_path)
    adapter = make_adapter(tmp_path, keys=[DOWN], config=adapter_config(target_modules=["down_proj"]))
    report = LM.merge_adapter(base, adapter, tmp_path / "out", tmp_path / "r.json",
                              target_regex=r"(?!.*(?:visual|mtp)).*\.mlp\.down_proj")
    assert {row["file"]: row["status"] for row in report["shards"]} == {SHARD1: "copied", SHARD2: "merged"}
    assert (tmp_path / "out" / SHARD1).read_bytes() == (base / SHARD1).read_bytes()
    assert report["n_targets"] == 1


def test_merge_adapter_unused_pair_raises_before_writing(tmp_path: Path) -> None:
    """An adapter pair with no base weight stops the run before any shard is written.

    :param tmp_path: Pytest temporary directory.
    """
    base = make_base(tmp_path)
    extra = {f"base_model.model.model.layers.0.mlp.up_proj.lora_{role}.weight": torch.zeros(RANK, 4)
             for role in "AB"}
    adapter = make_adapter(tmp_path, extra=extra)
    with pytest.raises(ValueError, match="no base weight"):
        LM.merge_adapter(base, adapter, tmp_path / "out", tmp_path / "r.json")
    assert not (tmp_path / "out").exists()


def test_merge_adapter_target_without_adapter_raises(tmp_path: Path) -> None:
    """A targeted weight (down_proj) with no pair in the adapter raises.

    :param tmp_path: Pytest temporary directory.
    """
    base = make_base(tmp_path)
    adapter = make_adapter(tmp_path, keys=[QKV])
    with pytest.raises(ValueError, match="no adapter pair"):
        LM.merge_adapter(base, adapter, tmp_path / "out", tmp_path / "r.json")


def test_merge_adapter_shape_mismatch_raises(tmp_path: Path) -> None:
    """A/B whose shapes do not fit the base weight are caught from headers.

    :param tmp_path: Pytest temporary directory.
    """
    base = make_base(tmp_path)
    adapter = make_adapter(tmp_path, keys=[QKV], extra={
        "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight": torch.zeros(RANK, 4),
        "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight": torch.zeros(4, RANK)})
    with pytest.raises(ValueError, match="does not fit"):
        LM.merge_adapter(base, adapter, tmp_path / "out", tmp_path / "r.json")


def test_stray_safetensors_and_memory_budget_raise(tmp_path: Path) -> None:
    """A shard missing from the index, or an estimate above the memory budget, refuses to start.

    :param tmp_path: Pytest temporary directory.
    """
    base = make_base(tmp_path)
    adapter = make_adapter(tmp_path)
    with pytest.raises(ValueError, match="memory|RSS"):
        LM.merge_adapter(base, adapter, tmp_path / "out", tmp_path / "r.json", memory_budget_bytes=1)
    save_file({"x": torch.zeros(2)}, str(base / "extra.safetensors"))
    with pytest.raises(ValueError, match="not in the index"):
        LM.merge_adapter(base, adapter, tmp_path / "out", tmp_path / "r.json")


def test_out_dir_equal_to_base_raises(tmp_path: Path) -> None:
    """Merging in place is refused.

    :param tmp_path: Pytest temporary directory.
    """
    base = make_base(tmp_path)
    with pytest.raises(ValueError, match="differ"):
        LM.merge_adapter(base, make_adapter(tmp_path), base, tmp_path / "r.json")


def test_written_size_mismatch_fails_before_rename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A merged tensor that does not fill its base span (simulated writer bug) raises before the rename: no final
    shard, no progress record, no index.

    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch.
    """
    base = make_base(tmp_path)
    real = LM.merge_tensor

    def short(weight: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor, scale: float,
              key: str = "") -> Any:
        """``merge_tensor`` with the last row dropped.

        :param weight: Base weight.
        :param lora_a: lora_A.
        :param lora_b: lora_B.
        :param scale: Scale.
        :param key: Key.
        :returns: Truncated merge and stats.
        """
        merged, stats = real(weight, lora_a, lora_b, scale, key)
        return merged[:-1].clone(), stats

    monkeypatch.setattr(LM, "merge_tensor", short)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="bytes, base span"):
        LM.merge_adapter(base, make_adapter(tmp_path), out, tmp_path / "r.json")
    assert not (out / SHARD1).exists() and not (out / LM.INDEX_NAME).exists()
    assert not (out / LM.PROGRESS_NAME).exists()


def test_base_metadata_and_header_bytes_kept_verbatim(tmp_path: Path) -> None:
    """A base shard with extra metadata keeps its exact header bytes (the old ``save_file`` writer changed them).

    :param tmp_path: Pytest temporary directory.
    """
    base = make_base(tmp_path, metadata={"format": "pt", "producer": "some-other-writer-with-a-long-name"})
    out = tmp_path / "out"
    LM.merge_adapter(base, make_adapter(tmp_path), out, tmp_path / "r.json")
    for shard in LAYOUT:
        assert LM.read_header_bytes(out / shard)[0] == LM.read_header_bytes(base / shard)[0]
    assert torch.equal(load_file(str(out / SHARD1))[QKV], torch.tensor(hand_merged(QKV, 2.0), dtype=torch.bfloat16))


def test_streaming_output_equals_save_file_output(tmp_path: Path) -> None:
    """Every streamed shard is byte-identical to merging in memory and saving with ``safetensors.save_file``.

    :param tmp_path: Pytest temporary directory.
    """
    base = make_base(tmp_path)
    adapter_dir = make_adapter(tmp_path)
    out = tmp_path / "out"
    LM.merge_adapter(base, adapter_dir, out, tmp_path / "r.json")
    adapter = load_file(str(adapter_dir / LM.ADAPTER_WEIGHTS))
    plan = LM.plan_merge(index_for(), list(adapter), REAL_REGEX)
    for shard in LAYOUT:
        pairs = {key: pair for key, pair in plan.items() if pair[0] == shard}
        merged, _ = LM.merge_shard(load_file(str(base / shard)), pairs, adapter, 2.0)
        reference = tmp_path / f"reference-{shard}"
        save_file(merged, str(reference), metadata={"format": "pt"})
        assert (out / shard).read_bytes() == reference.read_bytes()


def write_raw_safetensors(path: Path, header: Dict[str, Any], data: bytes) -> None:
    """Write a safetensors file from a hand-made header (to build malformed layouts).

    :param path: Destination.
    :param header: Header dict.
    :param data: Data section.
    """
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)


@pytest.mark.parametrize("offsets,data_len,match", [([[0, 8], [16, 24]], 24, "gap or overlap"),
                                                    ([[0, 8], [4, 12]], 12, "gap or overlap"),
                                                    ([[0, 8], [8, 16]], 24, "tensors end at"),
                                                    ([[0, 8], [8, 14]], 14, "span of 6 bytes")])
def test_data_layout_rejects_gaps_overlaps_and_bad_spans(tmp_path: Path, offsets: List[List[int]], data_len: int,
                                                         match: str) -> None:
    """The writer refuses a data section with a gap, an overlap, trailing bytes or a span of the wrong length.

    :param tmp_path: Pytest temporary directory.
    :param offsets: data_offsets of tensors ``a`` and ``b`` (BF16 [4] = 8 bytes each).
    :param data_len: Data-section length.
    :param match: Expected message fragment.
    """
    path = tmp_path / "bad.safetensors"
    header = {"__metadata__": {"format": "pt"},
              "a.weight": {"dtype": "BF16", "shape": [4], "data_offsets": offsets[0]},
              "b.weight": {"dtype": "BF16", "shape": [4], "data_offsets": offsets[1]}}
    write_raw_safetensors(path, header, bytes(data_len))
    raw, parsed = LM.read_header_bytes(path)
    with pytest.raises(ValueError, match=match):
        LM.data_layout(parsed, path.stat().st_size - parsed["__header_bytes__"])


# ------------------------------------------------------------------------------------------------------- resumption

def test_resume_skips_existing_shard(tmp_path: Path) -> None:
    """A finished shard from the same merge is skipped (file untouched, stats from the record); the missing one is
    re-merged to the same bytes.

    :param tmp_path: Pytest temporary directory.
    """
    base = make_base(tmp_path)
    adapter = make_adapter(tmp_path)
    out = tmp_path / "out"
    first = LM.merge_adapter(base, adapter, out, tmp_path / "r1.json")
    shard2_bytes = (out / SHARD2).read_bytes()
    (out / SHARD2).unlink()
    (out / LM.INDEX_NAME).unlink()
    before = (out / SHARD1).stat()
    second = LM.merge_adapter(base, adapter, out, tmp_path / "r2.json")
    after = (out / SHARD1).stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert second["resumed_shards"] == [SHARD1]
    assert {row["file"]: row["status"] for row in second["shards"]} == {SHARD1: "resumed", SHARD2: "merged"}
    assert (out / SHARD2).read_bytes() == shard2_bytes
    assert second["tensors"] == first["tensors"]
    assert (out / LM.INDEX_NAME).read_bytes() == (base / LM.INDEX_NAME).read_bytes()


def test_resume_refuses_shard_from_another_merge(tmp_path: Path) -> None:
    """Right-sized output shards without a matching progress record (unknown origin, or a different adapter) raise.

    :param tmp_path: Pytest temporary directory.
    """
    base = make_base(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    shutil.copyfile(base / SHARD1, out / SHARD1)
    with pytest.raises(ValueError, match="no progress record"):
        LM.merge_adapter(base, make_adapter(tmp_path), out, tmp_path / "r.json")
    (out / SHARD1).unlink()
    LM.merge_adapter(base, make_adapter(tmp_path, name="adapter_x", config=adapter_config(alpha=32)), out,
                     tmp_path / "r.json")
    with pytest.raises(ValueError, match="no progress record"):
        LM.merge_adapter(base, make_adapter(tmp_path, name="adapter_y", config=adapter_config(alpha=16)), out,
                         tmp_path / "r.json")


def test_crash_between_record_and_rename_is_redone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The progress record is written before the rename: a crash in between leaves a record and a ``.partial`` but
    no final shard, and the next run redoes that shard to the same bytes as a clean run.

    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch.
    """
    base = make_base(tmp_path)
    adapter = make_adapter(tmp_path)
    clean = tmp_path / "clean"
    LM.merge_adapter(base, adapter, clean, tmp_path / "clean.json")
    real_replace = os.replace

    def crash_on_shard1(src: Any, dst: Any) -> None:
        """``os.replace`` that fails for shard 1.

        :param src: Source.
        :param dst: Destination.
        :raises OSError: For shard 1.
        """
        if str(dst).endswith(SHARD1):
            raise OSError("simulated crash before the rename")
        real_replace(src, dst)

    out = tmp_path / "out"
    monkeypatch.setattr(os, "replace", crash_on_shard1)
    with pytest.raises(OSError, match="simulated"):
        LM.merge_adapter(base, adapter, out, tmp_path / "r.json")
    monkeypatch.setattr(os, "replace", real_replace)
    records = [json.loads(line) for line in (out / LM.PROGRESS_NAME).read_text().splitlines()]
    assert [record["shard"] for record in records] == [SHARD1] and records[0]["out_sha256"]
    assert (out / f"{SHARD1}.partial").exists() and not (out / SHARD1).exists()
    report = LM.merge_adapter(base, adapter, out, tmp_path / "r.json", verify_resumed=True)
    assert report["resumed_shards"] == []
    for name in list(LAYOUT) + [LM.INDEX_NAME]:
        assert (out / name).read_bytes() == (clean / name).read_bytes()
    assert not (out / f"{SHARD1}.partial").exists()


def test_truncated_progress_line_is_ignored_and_dropped(tmp_path: Path) -> None:
    """An unterminated last progress line (crash mid-append) is ignored on load and removed before the next append;
    an unparsable newline-terminated line still raises.

    :param tmp_path: Pytest temporary directory.
    """
    path = tmp_path / LM.PROGRESS_NAME
    good = {"fingerprint": {"f": 1}, "shard": "a", "out_sha256": "x"}
    path.write_text(json.dumps(good) + "\n" + '{"fingerprint": {"f": 1}, "sha')
    assert LM._load_progress(path, {"f": 1}) == {"a": good}
    second = {"fingerprint": {"f": 1}, "shard": "b", "out_sha256": "y"}
    LM._append_progress(path, second)
    assert path.read_text() == json.dumps(good) + "\n" + json.dumps(second) + "\n"
    assert LM._load_progress(path, {"f": 1}) == {"a": good, "b": second}
    path.write_text("{broken\n" + json.dumps(good) + "\n")
    with pytest.raises(json.JSONDecodeError):
        LM._load_progress(path, {"f": 1})


def test_stale_index_removed_when_a_run_starts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An index left by an earlier complete merge is removed before any shard is written, so a run that fails
    part-way never leaves "index present" on an incomplete directory.

    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch.
    """
    base = make_base(tmp_path)
    adapter = make_adapter(tmp_path)
    out = tmp_path / "out"
    LM.merge_adapter(base, adapter, out, tmp_path / "r.json")
    (out / SHARD2).unlink()
    assert (out / LM.INDEX_NAME).exists()

    def fail(*args: Any, **kwargs: Any) -> None:
        """Fail every merge.

        :param args: Ignored.
        :param kwargs: Ignored.
        :raises RuntimeError: Always.
        """
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(LM, "merge_tensor", fail)
    with pytest.raises(RuntimeError, match="simulated"):
        LM.merge_adapter(base, adapter, out, tmp_path / "r.json")
    assert not (out / LM.INDEX_NAME).exists() and not (out / SHARD2).exists()


def test_verify_resumed_catches_a_tampered_shard(tmp_path: Path) -> None:
    """A resumed shard of the right size whose bytes changed is trusted without ``verify_resumed`` and refused with it.

    :param tmp_path: Pytest temporary directory.
    """
    base = make_base(tmp_path)
    adapter = make_adapter(tmp_path)
    out = tmp_path / "out"
    LM.merge_adapter(base, adapter, out, tmp_path / "r.json")
    data = bytearray((out / SHARD1).read_bytes())
    data[-1] ^= 0xFF
    (out / SHARD1).write_bytes(bytes(data))
    with pytest.raises(ValueError, match="sha256 differs"):
        LM.merge_adapter(base, adapter, out, tmp_path / "r.json", verify_resumed=True)
    assert LM.merge_adapter(base, adapter, out, tmp_path / "r.json")["resumed_shards"] == [SHARD1, SHARD2]


# -------------------------------------------------------------------------------- real key naming, verbatim, meta

R_EMBED = "model.language_model.embed_tokens.weight"
R_QKV = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
R_CONV = "model.language_model.layers.0.linear_attn.conv1d.weight"
R_ALOG = "model.language_model.layers.0.linear_attn.A_log"
R_OUT = "model.language_model.layers.0.linear_attn.out_proj.weight"
R_Q = "model.language_model.layers.1.self_attn.q_proj.weight"
R_QNORM = "model.language_model.layers.1.self_attn.q_norm.weight"
R_GATE = "model.language_model.layers.1.mlp.gate_proj.weight"
R_NORM = "model.language_model.norm.weight"
R_VISUAL = "model.visual.blocks.0.attn.qkv.weight"
R_MTP = "mtp.layers.0.self_attn.k_proj.weight"
R_HEAD = "lm_head.weight"
R_SHARDS = ("model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors")
R_TARGETS = [R_QKV, R_OUT, R_Q, R_GATE]


def real_tensors() -> Dict[str, Dict[str, torch.Tensor]]:
    """Qwen3.8-named synthetic tensors laid out like the t27b shards 00003 (embed only), a layer shard and 00018
    (lm_head + mtp), plus 1-D / 3-D untouched tensors.

    :returns: ``{shard: {key: bf16 tensor}}``.
    """
    generator = torch.Generator().manual_seed(7)

    def rand(*shape: int) -> torch.Tensor:
        """Random bf16 tensor.

        :param shape: Shape.
        :returns: Tensor.
        """
        return (torch.randn(*shape, generator=generator) * 0.05).to(torch.bfloat16)

    return {
        R_SHARDS[0]: {R_EMBED: rand(32, 8)},
        R_SHARDS[1]: {R_QKV: rand(24, 8), R_CONV: rand(24, 1, 4), R_ALOG: rand(3), R_OUT: rand(8, 12),
                      R_Q: rand(16, 8), R_QNORM: rand(4), R_GATE: rand(20, 8), R_NORM: rand(8),
                      R_VISUAL: rand(12, 6)},
        R_SHARDS[2]: {R_HEAD: rand(32, 8), R_MTP: rand(4, 8)},
    }


def write_model(directory: Path, layout: Dict[str, Dict[str, torch.Tensor]], files: Dict[str, str]) -> Path:
    """Write shards, the index and non-weight files.

    :param directory: Model directory (created).
    :param layout: ``{shard: {key: tensor}}``.
    :param files: ``{name: text}`` non-weight files.
    :returns: The directory.
    """
    directory.mkdir(parents=True)
    weight_map = {}
    for shard, tensors in layout.items():
        save_file(tensors, str(directory / shard), metadata={"format": "pt"})
        weight_map.update({key: shard for key in tensors})
    total = sum(t.numel() * t.element_size() for tensors in layout.values() for t in tensors.values())
    (directory / LM.INDEX_NAME).write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}))
    for name, text in files.items():
        (directory / name).write_text(text)
    return directory


def real_adapter(root: Path) -> Path:
    """PEFT adapter with ``base_model.model.`` keys over the real module paths (float32 A/B, r=16, alpha=16).

    :param root: Parent directory.
    :returns: The adapter directory.
    """
    generator = torch.Generator().manual_seed(11)
    shapes = {key: tuple(tensors[key].shape) for tensors in real_tensors().values() for key in tensors}
    tensors = {}
    for key in R_TARGETS:
        rows, cols = shapes[key]
        module = f"base_model.model.{key[:-len('.weight')]}"
        tensors[f"{module}.lora_A.weight"] = torch.randn(RANK, cols, generator=generator) * 0.1
        tensors[f"{module}.lora_B.weight"] = torch.randn(rows, RANK, generator=generator) * 0.1
    adapter = root / "real_adapter"
    adapter.mkdir()
    save_file(tensors, str(adapter / LM.ADAPTER_WEIGHTS), metadata={"format": "pt"})
    (adapter / LM.ADAPTER_CONFIG).write_text(json.dumps(adapter_config(alpha=16)))
    return adapter


def f16_round_trip(tensor: torch.Tensor) -> torch.Tensor:
    """bf16 -> f16 -> bf16, the training-time copy of embed / lm_head (tiny values lose precision).

    :param tensor: bf16 tensor.
    :returns: Round-tripped tensor.
    """
    return tensor.to(torch.float16).to(torch.bfloat16)


def test_real_naming_with_verbatim_shards_and_meta_dir(tmp_path: Path) -> None:
    """Real Qwen3.8 key naming with the PEFT prefix: the regex selects exactly the four projections, the two
    target-free shards come byte for byte from the "final" directory, the layer shard is merged bit-exactly, and
    the non-weight files are the meta dir's (base-only extras and crc32.txt are not copied).

    :param tmp_path: Pytest temporary directory.
    """
    layout = real_tensors()
    base = write_model(tmp_path / "base", layout, {"config.json": "base config", "tokenizer.json": "base tok",
                                                   "vocab.json": "v", "merges.txt": "m", "crc32.txt": "c",
                                                   "LICENSE": "l"})
    final_layout = {R_SHARDS[0]: {R_EMBED: f16_round_trip(layout[R_SHARDS[0]][R_EMBED])},
                    R_SHARDS[2]: {R_HEAD: f16_round_trip(layout[R_SHARDS[2]][R_HEAD]),
                                  R_MTP: layout[R_SHARDS[2]][R_MTP]}}
    final = write_model(tmp_path / "final_shards", final_layout, {})
    (final / LM.INDEX_NAME).unlink()
    meta = write_model(tmp_path / "meta", {}, {"config.json": "final config", "tokenizer.json": "final tok",
                                               "generation_config.json": "g", "processor_config.json": "p"})
    adapter = real_adapter(tmp_path)
    keys = [key for tensors in layout.values() for key in tensors]
    assert LM.targeted_weight_keys(keys, REAL_REGEX) == sorted(R_TARGETS)
    assert LM.untargeted_shards(json.loads((base / LM.INDEX_NAME).read_text()), REAL_REGEX) == [R_SHARDS[0],
                                                                                               R_SHARDS[2]]
    plan = LM.plan_merge(json.loads((base / LM.INDEX_NAME).read_text()),
                         list(load_file(str(adapter / LM.ADAPTER_WEIGHTS))), REAL_REGEX)
    assert sorted(plan) == sorted(R_TARGETS) and {pair[0] for pair in plan.values()} == {R_SHARDS[1]}

    out = tmp_path / "out"
    report = LM.merge_adapter(base, adapter, out, tmp_path / "r.json", verbatim_dir=final, meta_dir=meta)
    assert {row["file"]: row["status"] for row in report["shards"]} == {
        R_SHARDS[0]: "verbatim", R_SHARDS[1]: "merged", R_SHARDS[2]: "verbatim"}
    for shard in (R_SHARDS[0], R_SHARDS[2]):
        assert (out / shard).read_bytes() == (final / shard).read_bytes()
    merged = load_file(str(out / R_SHARDS[1]))
    lora = load_file(str(adapter / LM.ADAPTER_WEIGHTS))
    for key in R_TARGETS:
        module = f"base_model.model.{key[:-len('.weight')]}"
        weight = layout[R_SHARDS[1]][key]
        expected = (weight.float() + 1.0 * (lora[f"{module}.lora_B.weight"] @ lora[f"{module}.lora_A.weight"]))
        assert torch.equal(merged[key], expected.to(torch.bfloat16))
    for key in (R_CONV, R_ALOG, R_QNORM, R_NORM, R_VISUAL):
        assert raw_bytes(merged[key]) == raw_bytes(layout[R_SHARDS[1]][key])
    assert sorted(report["copied_files"]) == ["config.json", "generation_config.json", "processor_config.json",
                                              "tokenizer.json"]
    assert (out / "tokenizer.json").read_text() == "final tok"
    for name in ("vocab.json", "merges.txt", "crc32.txt", "LICENSE"):
        assert not (out / name).exists()
    assert report["fingerprint"]["verbatim_shards"] == {R_SHARDS[0]: (base / R_SHARDS[0]).stat().st_size,
                                                        R_SHARDS[2]: (base / R_SHARDS[2]).stat().st_size}
    second = LM.merge_adapter(base, adapter, out, tmp_path / "r2.json", verbatim_dir=final, meta_dir=meta,
                              verify_resumed=True)
    assert second["resumed_shards"] == list(R_SHARDS)
    with pytest.raises(ValueError, match="no progress record"):  # dropping the verbatim dir changes the fingerprint
        LM.merge_adapter(base, adapter, out, tmp_path / "r3.json", meta_dir=meta)


def test_verbatim_shard_refusals(tmp_path: Path) -> None:
    """A verbatim shard that holds targets, has other header bytes, or is not in the base index is refused.

    :param tmp_path: Pytest temporary directory.
    """
    layout = real_tensors()
    base = write_model(tmp_path / "base", layout, {"config.json": "c"})
    adapter = real_adapter(tmp_path)
    targeted = write_model(tmp_path / "targeted", {R_SHARDS[1]: layout[R_SHARDS[1]]}, {})
    with pytest.raises(ValueError, match="hold LoRA targets"):
        LM.describe_plan(base, adapter, verbatim_dir=targeted)
    other = write_model(tmp_path / "other", {R_SHARDS[0]: {R_EMBED: layout[R_SHARDS[0]][R_EMBED][:16].clone()}}, {})
    with pytest.raises(ValueError, match="header bytes differ"):
        LM.describe_plan(base, adapter, verbatim_dir=other)
    stray = write_model(tmp_path / "stray", {"model-00009-of-00009.safetensors": {"x.weight": torch.zeros(2)}}, {})
    with pytest.raises(ValueError, match="not in the base index"):
        LM.describe_plan(base, adapter, verbatim_dir=stray)
    with pytest.raises(ValueError, match="no config.json"):
        LM.describe_plan(base, adapter, meta_dir=write_model(tmp_path / "meta", {}, {"tokenizer.json": "t"}))


# ------------------------------------------------------------------------------------------------------ peak RSS

def test_estimate_peak_rss_formula() -> None:
    """adapter + 12 x numel + one copy chunk + interpreter allowance; t27b's largest target is 17408 x 5120."""
    assert LM.estimate_peak_rss(100, 10) == 100 + 120 + LM.COPY_CHUNK + LM.RSS_OVERHEAD
    t27b = LM.estimate_peak_rss(435_280_528, 17408 * 5120)
    assert 1.9e9 < t27b < 2.2e9 and t27b < LM.DEFAULT_MEMORY_BUDGET


def test_subprocess_peak_rss_within_estimate(tmp_path: Path) -> None:
    """A real CLI merge of a shard with a 256 MiB untargeted tensor stays within the estimate (the previous
    ``load_file`` + ``save_file`` writer peaked near 2-3x the shard, above it).

    :param tmp_path: Pytest temporary directory.
    """
    base = tmp_path / "base"
    base.mkdir()
    shard = "model-00001-of-00001.safetensors"
    tensors = {EMBED: torch.ones(16384, 8192, dtype=torch.bfloat16), QKV: torch.ones(64, 48, dtype=torch.bfloat16)}
    save_file(tensors, str(base / shard), metadata={"format": "pt"})
    del tensors
    (base / LM.INDEX_NAME).write_text(json.dumps({"weight_map": {EMBED: shard, QKV: shard}}))
    (base / "config.json").write_text("{}")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    save_file({"base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_A.weight": torch.ones(RANK, 48) * 0.01,
               "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_B.weight": torch.ones(64, RANK) * 0.01},
              str(adapter / LM.ADAPTER_WEIGHTS))
    (adapter / LM.ADAPTER_CONFIG).write_text(json.dumps(adapter_config(alpha=16)))
    out = tmp_path / "out"
    script = ("import json, resource, sys\n"
              "from src.finetuning.text27b import lora_merge as LM\n"
              "LM.main(sys.argv[1:])\n"
              "scale = 1 if sys.platform == 'darwin' else 1024\n"
              "print('PEAK', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale)\n")
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo)}
    result = subprocess.run([sys.executable, "-c", script, "--base-dir", str(base), "--adapter-dir", str(adapter),
                             "--out-dir", str(out), "--report", str(tmp_path / "r.json"), "--threads", "2",
                             "--max-local-gb", "1"], cwd=repo, env=env, capture_output=True, text=True, check=True)
    peak = int(result.stdout.strip().splitlines()[-1].split()[1])
    estimate = json.loads((tmp_path / "r.json").read_text())["peak_rss_estimate"]
    shard_bytes = (base / shard).stat().st_size
    assert (out / shard).stat().st_size == shard_bytes
    assert peak <= estimate, f"peak RSS {peak} > estimate {estimate} (shard {shard_bytes})"
    (base / shard).unlink()  # keep the boot disk clean (pytest keeps recent tmp dirs)
    (out / shard).unlink()


# -------------------------------------------------------------------------------------------------------------- CLI

def test_dry_run_prints_plan(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """``--dry-run`` prints counts and bytes and writes nothing.

    :param tmp_path: Pytest temporary directory.
    :param capsys: Output capture.
    """
    base = make_base(tmp_path)
    adapter = make_adapter(tmp_path)
    LM.main(["--base-dir", str(base), "--adapter-dir", str(adapter), "--out-dir", str(tmp_path / "out"),
             "--report", str(tmp_path / "r.json"), "--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["n_targets"] == 2 and plan["n_shards"] == 2 and plan["n_shards_with_targets"] == 2
    assert plan["base_bytes"] == sum((base / shard).stat().st_size for shard in LAYOUT)
    assert plan["adapter_bytes"] == (adapter / LM.ADAPTER_WEIGHTS).stat().st_size
    assert plan["scale"] == 2.0 and plan["target_regex"] == REAL_REGEX
    assert plan["target_dtypes"] == {"BF16": 2} and plan["max_target_numel"] == 24
    assert plan["targets_per_shard"] == {SHARD1: 1, SHARD2: 1}
    assert not (tmp_path / "out").exists() and not (tmp_path / "r.json").exists()


def test_cli_run(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """A full CLI run writes the merged directory and the report.

    :param tmp_path: Pytest temporary directory.
    :param capsys: Output capture.
    """
    base = make_base(tmp_path)
    adapter = make_adapter(tmp_path)
    LM.main(["--base-dir", str(base), "--adapter-dir", str(adapter), "--out-dir", str(tmp_path / "out"),
             "--report", str(tmp_path / "r.json")])
    assert json.loads((tmp_path / "r.json").read_text())["n_targets"] == 2
    assert (tmp_path / "out" / LM.INDEX_NAME).exists()
    assert '"n_targets": 2' in capsys.readouterr().out
