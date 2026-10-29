# File name: lora_merge.py
# Date: 10/27/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Merge a PEFT LoRA adapter into a sharded bf16 safetensors checkpoint, streaming one tensor at a time.

Use: merge an intermediate t27b checkpoint adapter (e.g. the step-1000 ``last-checkpoint/``) into a bf16 base so it
can be converted and evaluated like the final merged model. The arithmetic per targeted linear weight ``W`` (shape
``[out, in]``) with ``A`` (``[r, in]``) and ``B`` (``[out, r]``) is::

    W' = (W.float() + scale * (B.float() @ A.float())).to(W.dtype)
    scale = lora_alpha / r            (lora_alpha / sqrt(r) when use_rslora)

PEFT key convention (verified against peft 0.17.1 ``utils/save_and_load.get_peft_model_state_dict`` and
``tuners/lora/layer.py``): a LoRA ``Linear`` holds ``lora_A`` / ``lora_B`` as ``nn.ModuleDict`` of ``nn.Linear``
keyed by adapter name, so the in-memory key is ``<peft prefix><module>.lora_A.<adapter>.weight``; on save the
``.<adapter>`` part is removed (``k.replace(f".{adapter_name}", "")``), giving
``base_model.model.<module>.lora_A.weight`` where ``<module>`` is the module path inside the wrapped model (for a
``*ForCausalLM`` / ``*ForConditionalGeneration`` that is the same path as the checkpoint key minus ``.weight``).
The prefix(es) stripped are a parameter (:data:`PEFT_PREFIXES`); a key without any of them is taken as bare.
Targeting follows PEFT: a string ``target_modules`` is applied with ``re.fullmatch`` to the module name; a list
matches a module equal to an entry or ending in ``.<entry>`` (:func:`target_regex_from_config`).

Writer (:func:`merge_adapter`): every output shard is streamed into ``<name>.partial`` and renamed when complete.
The base shard's safetensors header bytes are copied verbatim (tensor order, offsets, padding and ``__metadata__``
stay the base's), then its data section is walked in offset order (:func:`data_layout` asserts the spans tile it
with no gap); a targeted tensor is read into one buffer, merged and written, every other byte range is copied
through one :data:`COPY_CHUNK` buffer. Nothing is memory-mapped and no shard is ever resident as a whole. Shards with
no target are byte-copied, from the base or, for the shards present in ``verbatim_dir``, from that directory: such a
shard must hold no target and have exactly the base shard's header bytes and size (t27b takes the final merge's
target-free embed / lm_head shards this way; they carry the training-time f16 round trip of those two tensors).
Non-weight files come from ``meta_dir`` when given, else from the base (minus ``crc32.txt``, whose checksums would be
stale). The index is removed when a run starts and written last, so its presence marks a complete merge.

Peak RSS is about ``adapter + 12 x largest targeted numel + COPY_CHUNK`` plus the interpreter
(:func:`estimate_peak_rss`): the targeted tensor's bf16 buffer (2n), its float32 working copy (4n), the float32
``scale * B @ A`` (4n) and the bf16 result (2n). The run refuses to start when the estimate exceeds the budget.

Resumption: before a shard is renamed into place, a progress record (merge fingerprint, verified size and the
sha256 of the bytes written) is appended to ``lora_merge_progress.jsonl``; a final-named shard of the base size with
a matching record is skipped (and re-hashed against the record when ``verify_resumed``). A crash between the record
and the rename leaves only the ``.partial``, which the next run redoes; a crash mid-append leaves an unterminated last
line, which is ignored and dropped before the next append.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Sequence, Tuple

import torch
from safetensors.torch import load_file

#: Base checkpoint shard index file name.
INDEX_NAME = "model.safetensors.index.json"
#: PEFT adapter weight / config file names.
ADAPTER_WEIGHTS = "adapter_model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"
#: Per-shard progress records (JSON lines) kept in the output directory for resumption.
PROGRESS_NAME = "lora_merge_progress.jsonl"
#: Key prefixes PEFT puts in front of the wrapped model's module path in a saved adapter.
PEFT_PREFIXES: Tuple[str, ...] = ("base_model.model.",)
#: Default peak-RSS budget for the merge process (bytes).
DEFAULT_MEMORY_BUDGET = 12 * 10 ** 9
#: Fixed allowance for the interpreter, torch and bookkeeping in the peak-RSS estimate (bytes; ~0.2 GB measured).
RSS_OVERHEAD = 512 * 1024 ** 2
#: Size of the one buffer through which untargeted byte ranges are copied (bytes).
COPY_CHUNK = 64 * 1024 ** 2
#: Bytes of float temporaries per element of a targeted weight (bf16 buffer + f32 copy + f32 B@A + bf16 result).
TEMP_BYTES_PER_ELEMENT = 12
#: Free space kept in reserve on the output filesystem (bytes).
FREE_SPACE_MARGIN = 1024 ** 3
#: Default torch intra-op threads (the Studio is a shared production machine).
DEFAULT_THREADS = 6
#: Safetensors dtypes a targeted base weight may have, and their torch dtypes.
TORCH_DTYPES: Dict[str, torch.dtype] = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
FLOAT_DTYPES = frozenset(TORCH_DTYPES)
#: Element sizes of the safetensors dtypes (used to check every span against dtype x shape).
DTYPE_BYTES: Dict[str, int] = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "U16": 2, "I16": 2, "F16": 2,
                               "BF16": 2, "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8}
#: Non-weight files never copied into the merged directory.
EXCLUDED_FILES = frozenset({"crc32.txt", PROGRESS_NAME})
#: Files that are weights in another format; their presence in the base directory is an error.
FOREIGN_WEIGHT_SUFFIXES = frozenset({".bin", ".pt", ".pth", ".gguf", ".ckpt", ".h5", ".msgpack", ".npz"})
#: Adapter-config fields that must be empty / falsy (features this merge does not implement).
UNSUPPORTED_NONEMPTY = ("rank_pattern", "alpha_pattern", "modules_to_save", "exclude_modules", "layers_to_transform",
                        "layers_pattern", "target_parameters", "trainable_token_indices", "layer_replication")
UNSUPPORTED_TRUE = ("use_dora", "fan_in_fan_out", "lora_bias", "use_qalora", "use_bdlora")
_ADAPTER_KEY = re.compile(r"(?P<module>.+)\.lora_(?P<role>[AB])\.weight")
_MAX_HEADER_BYTES = 100 * 1024 ** 2

Pair = Tuple[str, str, str]


def check_supported(adapter_config: Dict[str, Any]) -> None:
    """Raise when the adapter config uses a PEFT feature this merge does not implement.

    :param adapter_config: Parsed ``adapter_config.json``.
    :raises ValueError: On a non-LoRA adapter, a bias other than ``none`` or any field in
        :data:`UNSUPPORTED_NONEMPTY` / :data:`UNSUPPORTED_TRUE` being set.
    """
    peft_type = adapter_config.get("peft_type", "LORA")
    if str(peft_type).upper() != "LORA":
        raise ValueError(f"unsupported peft_type {peft_type!r}; only plain LoRA is merged")
    bad = [name for name in UNSUPPORTED_NONEMPTY if adapter_config.get(name)]
    bad += [name for name in UNSUPPORTED_TRUE if adapter_config.get(name)]
    if adapter_config.get("bias", "none") not in (None, "none"):
        bad.append("bias")
    if bad:
        raise ValueError(f"unsupported adapter_config fields set: {', '.join(bad)}")


def lora_scale(adapter_config: Dict[str, Any]) -> float:
    """LoRA scaling factor applied to ``B @ A``.

    :param adapter_config: Parsed ``adapter_config.json`` (needs ``r`` and ``lora_alpha``).
    :returns: ``lora_alpha / r``, or ``lora_alpha / sqrt(r)`` when ``use_rslora`` is true.
    :raises ValueError: On unsupported features (:func:`check_supported`) or a non-positive rank.
    """
    check_supported(adapter_config)
    rank = int(adapter_config["r"])
    alpha = float(adapter_config["lora_alpha"])
    if rank <= 0:
        raise ValueError(f"LoRA rank must be positive, got {rank}")
    if adapter_config.get("use_rslora"):
        return alpha / math.sqrt(rank)
    return alpha / rank


def target_regex_from_config(adapter_config: Dict[str, Any]) -> str:
    """The module-name regex equivalent to the adapter's ``target_modules``.

    :param adapter_config: Parsed ``adapter_config.json``.
    :returns: A regex for ``re.fullmatch`` against module names (a string ``target_modules`` is returned as is;
        a list becomes ``(?:.*\\.)?(?:name1|name2|...)``, PEFT's "equal or ends with ``.name``" rule).
    :raises ValueError: When ``target_modules`` is missing, empty or ``all-linear``.
    """
    targets = adapter_config.get("target_modules")
    if isinstance(targets, str):
        if targets == "all-linear":
            raise ValueError("target_modules='all-linear' cannot be resolved without the model; pass a regex")
        return targets
    if not targets:
        raise ValueError("adapter_config has no target_modules; pass target_regex explicitly")
    names = "|".join(re.escape(str(name)) for name in sorted(targets))
    return rf"(?:.*\.)?(?:{names})"


def parse_adapter_key(adapter_key: str, prefixes: Sequence[str] = PEFT_PREFIXES) -> Tuple[str, str]:
    """Split a saved PEFT LoRA key into the base weight key it updates and its role.

    :param adapter_key: E.g. ``base_model.model.model.layers.3.mlp.down_proj.lora_A.weight`` or the bare
        ``model.layers.3.mlp.down_proj.lora_B.weight``.
    :param prefixes: Prefixes to strip (the first one that matches is removed); a key with none is bare.
    :returns: ``(weight_key, role)``, e.g. ``("model.layers.3.mlp.down_proj.weight", "A")``. The weight key is
        the pair id.
    :raises ValueError: For anything that is not a ``lora_A`` / ``lora_B`` weight with the adapter name already
        removed (``lora_A.default.weight``, ``lora_embedding_A``, ``lora_magnitude_vector``, biases, ...).
    """
    match = _ADAPTER_KEY.fullmatch(adapter_key)
    if match is None:
        raise ValueError(f"not a saved PEFT LoRA A/B weight key: {adapter_key!r} "
                         "(expected <module>.lora_A.weight or <module>.lora_B.weight)")
    module = match.group("module")
    for prefix in prefixes:
        if module.startswith(prefix):
            module = module[len(prefix):]
            break
    if not module:
        raise ValueError(f"empty module path in adapter key {adapter_key!r}")
    return f"{module}.weight", match.group("role")


def module_name(adapter_key: str, prefixes: Sequence[str] = PEFT_PREFIXES) -> str:
    """Base checkpoint weight key updated by a saved PEFT LoRA key.

    :param adapter_key: E.g. ``base_model.model.model.layers.3.mlp.down_proj.lora_A.weight``.
    :param prefixes: Prefixes to strip, see :func:`parse_adapter_key`.
    :returns: E.g. ``model.layers.3.mlp.down_proj.weight``.
    :raises ValueError: See :func:`parse_adapter_key`.
    """
    return parse_adapter_key(adapter_key, prefixes)[0]


def _module_of(weight_key: str) -> str:
    """Module path of a ``<module>.weight`` key.

    :param weight_key: Checkpoint key ending in ``.weight``.
    :returns: The key without the ``.weight`` suffix.
    """
    return weight_key[: -len(".weight")]


def targeted_weight_keys(weight_keys: Sequence[str], target_regex: str) -> List[str]:
    """Checkpoint weights whose module the regex selects.

    :param weight_keys: Checkpoint tensor keys.
    :param target_regex: Module-name regex applied with ``re.fullmatch``.
    :returns: Sorted ``*.weight`` keys whose module path fully matches.
    """
    pattern = re.compile(target_regex)
    return sorted(key for key in weight_keys if key.endswith(".weight") and pattern.fullmatch(_module_of(key)))


def untargeted_shards(base_index: Dict[str, Any], target_regex: str) -> List[str]:
    """Shards of a checkpoint index that hold no weight the regex selects.

    :param base_index: Parsed ``model.safetensors.index.json`` (needs ``weight_map``).
    :param target_regex: Module-name regex applied with ``re.fullmatch``.
    :returns: Sorted shard file names.
    """
    weight_map: Dict[str, str] = base_index["weight_map"]
    targeted = {weight_map[key] for key in targeted_weight_keys(list(weight_map), target_regex)}
    return sorted(set(weight_map.values()) - targeted)


def _examples(keys: Sequence[str], limit: int = 5) -> str:
    """Short listing of keys for an error message.

    :param keys: Keys.
    :param limit: How many to show.
    :returns: ``n=<count>: a, b, ...``.
    """
    keys = sorted(keys)
    more = ", ..." if len(keys) > limit else ""
    return f"n={len(keys)}: " + ", ".join(keys[:limit]) + more


def plan_merge(base_index: Dict[str, Any], adapter_keys: Sequence[str], target_regex: str,
               prefixes: Sequence[str] = PEFT_PREFIXES) -> Dict[str, Pair]:
    """Pair every adapter A/B with the base weight it updates and the shard holding that weight.

    :param base_index: Parsed ``model.safetensors.index.json`` (needs ``weight_map``).
    :param adapter_keys: Keys of the adapter safetensors file.
    :param target_regex: Module-name regex (``re.fullmatch``) the adapter was trained with.
    :param prefixes: Adapter key prefixes to strip, see :func:`parse_adapter_key`.
    :returns: ``{weight_key: (shard_file, lora_A_key, lora_B_key)}`` sorted by weight key.
    :raises ValueError: When the adapter is empty, a pair is matched twice (two keys for the same weight and
        role) or lacks its A or B, an adapter pair has no base weight, an adapter pair updates a weight outside
        the regex, or a base weight matching the regex has no adapter pair.
    """
    weight_map = base_index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("base index has no weight_map")
    pairs: Dict[str, Dict[str, str]] = {}
    for key in adapter_keys:
        weight_key, role = parse_adapter_key(key, prefixes)
        slot = pairs.setdefault(weight_key, {})
        if role in slot:
            raise ValueError(f"adapter pair for {weight_key} matched twice (lora_{role}): {slot[role]!r} and {key!r}")
        slot[role] = key
    if not pairs:
        raise ValueError("adapter holds no LoRA pairs")
    incomplete = [key for key, slot in pairs.items() if set(slot) != {"A", "B"}]
    if incomplete:
        raise ValueError(f"adapter pairs missing lora_A or lora_B ({_examples(incomplete)})")
    no_base = [key for key in pairs if key not in weight_map]
    if no_base:
        raise ValueError(f"adapter pairs with no base weight ({_examples(no_base)})")
    pattern = re.compile(target_regex)
    outside = [key for key in pairs if not pattern.fullmatch(_module_of(key))]
    if outside:
        raise ValueError(f"adapter pairs for weights the target regex does not select ({_examples(outside)})")
    without_pair = [key for key in targeted_weight_keys(list(weight_map), target_regex) if key not in pairs]
    if without_pair:
        raise ValueError(f"base weights matching the target regex have no adapter pair ({_examples(without_pair)})")
    return {key: (weight_map[key], pairs[key]["A"], pairs[key]["B"]) for key in sorted(pairs)}


def merge_tensor(weight: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor, scale: float,
                 key: str = "") -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Merge one LoRA pair into one weight; the inputs are not modified.

    Computes ``(W.float() + scale * (B.float() @ A.float())).to(W.dtype)`` with in-place float32 steps
    (bitwise identical to that expression) to keep temporaries near ``10 x numel`` bytes beside ``W``.

    :param weight: Base weight ``[out, in]`` (floating point; bf16 for t27b).
    :param lora_a: ``lora_A`` weight ``[r, in]``.
    :param lora_b: ``lora_B`` weight ``[out, r]``.
    :param scale: LoRA scale (:func:`lora_scale`).
    :param key: Weight key, for messages and stats.
    :returns: ``(merged, stats)``; ``merged`` is a new tensor of ``weight.dtype``. ``stats`` has ``key``,
        ``shape``, ``dtype``, ``rank``, ``delta_fro_over_w_fro`` and ``max_abs_delta`` (of the realised
        ``W' - W`` after the cast back), ``lora_fro_over_w_fro`` (of ``scale * B @ A`` before the cast) and
        ``changed_fraction`` (share of elements the cast result actually changed).
    :raises ValueError: On non-2-D tensors, incompatible shapes or a non-floating-point weight.
    """
    if weight.ndim != 2 or lora_a.ndim != 2 or lora_b.ndim != 2:
        raise ValueError(f"{key}: expected 2-D W, A, B; got {tuple(weight.shape)}, {tuple(lora_a.shape)}, "
                         f"{tuple(lora_b.shape)}")
    out_features, in_features = weight.shape
    if lora_a.shape[1] != in_features or lora_b.shape[0] != out_features or lora_b.shape[1] != lora_a.shape[0]:
        raise ValueError(f"{key}: W {tuple(weight.shape)} incompatible with A {tuple(lora_a.shape)} and "
                         f"B {tuple(lora_b.shape)}")
    if not weight.is_floating_point():
        raise ValueError(f"{key}: base weight dtype {weight.dtype} is not floating point (quantised base?)")
    work = weight.to(torch.float32, copy=True)
    product = lora_b.to(torch.float32) @ lora_a.to(torch.float32)
    product.mul_(scale)
    w_fro = float(torch.linalg.vector_norm(work))
    lora_fro = float(torch.linalg.vector_norm(product))
    work.add_(product)
    merged = work.to(weight.dtype, copy=True)
    del work
    product.copy_(merged)
    product.sub_(weight)
    delta_fro = float(torch.linalg.vector_norm(product))
    changed = int(torch.count_nonzero(product))
    max_abs = float(product.abs_().max()) if product.numel() else 0.0
    del product
    stats = {
        "key": key,
        "shape": [int(out_features), int(in_features)],
        "dtype": str(weight.dtype).replace("torch.", ""),
        "rank": int(lora_a.shape[0]),
        "delta_fro_over_w_fro": delta_fro / w_fro if w_fro > 0 else None,
        "lora_fro_over_w_fro": lora_fro / w_fro if w_fro > 0 else None,
        "max_abs_delta": max_abs,
        "changed_fraction": changed / weight.numel() if weight.numel() else 0.0,
    }
    return merged, stats


def _check_pairs_available(keys: Sequence[str], pairs: Dict[str, Pair], adapter: Dict[str, torch.Tensor]) -> None:
    """Require every planned weight to be among ``keys`` and every planned A/B to be in the adapter.

    :param keys: Tensor keys of the shard.
    :param pairs: Plan entries for the shard.
    :param adapter: Adapter tensors.
    :raises ValueError: On a missing weight or adapter tensor.
    """
    present = set(keys)
    missing = [key for key in pairs if key not in present]
    if missing:
        raise ValueError(f"planned weights absent from the shard ({_examples(missing)})")
    absent = [name for key in pairs for name in pairs[key][1:] if name not in adapter]
    if absent:
        raise ValueError(f"planned adapter tensors absent from the adapter ({_examples(absent)})")


def merge_shard(tensors: Dict[str, torch.Tensor], pairs: Dict[str, Pair], adapter: Dict[str, torch.Tensor],
                scale: float) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    """Merge the given pairs into one in-memory shard's tensors (reference path; the driver streams instead).

    :param tensors: The shard's tensors (not modified).
    :param pairs: ``{weight_key: (shard_file, lora_A_key, lora_B_key)}`` for weights in this shard (a subset of
        :func:`plan_merge`'s result).
    :param adapter: Adapter tensors keyed as in the adapter file.
    :param scale: LoRA scale.
    :returns: ``(merged_tensors, stats)``: a new dict in which targeted weights are new merged tensors and every
        other entry is the same object as in ``tensors``; one :func:`merge_tensor` stats dict per pair, by key.
    :raises ValueError: When a pair's weight is not in ``tensors`` or its A/B is not in ``adapter``.
    """
    _check_pairs_available(list(tensors), pairs, adapter)
    merged = dict(tensors)
    stats = []
    for key in sorted(pairs):
        _, a_key, b_key = pairs[key]
        merged[key], entry = merge_tensor(tensors[key], adapter[a_key], adapter[b_key], scale, key)
        stats.append(entry)
    return merged, stats


def read_header_bytes(path: Path) -> Tuple[bytes, Dict[str, Any]]:
    """Raw header bytes and the parsed header of a safetensors file, without touching the tensor data.

    :param path: Safetensors file.
    :returns: ``(raw, header)``: ``raw`` is the 8-byte length prefix plus the JSON header exactly as stored
        (padding included); ``header`` is the parsed dict (``__metadata__`` plus ``{key: {dtype, shape,
        data_offsets}}``) with the size of ``raw`` added as ``__header_bytes__``.
    :raises ValueError: On a truncated file or an implausible header length.
    """
    with Path(path).open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: too short for a safetensors header")
        (length,) = struct.unpack("<Q", prefix)
        if length > _MAX_HEADER_BYTES:
            raise ValueError(f"{path}: implausible safetensors header length {length}")
        raw = handle.read(length)
    if len(raw) != length:
        raise ValueError(f"{path}: header truncated ({len(raw)} of {length} bytes)")
    header = json.loads(raw)
    header["__header_bytes__"] = 8 + length
    return prefix + raw, header


def read_safetensors_header(path: Path) -> Dict[str, Any]:
    """Parsed safetensors header (see :func:`read_header_bytes`).

    :param path: Safetensors file.
    :returns: The header dict with ``__header_bytes__``.
    :raises ValueError: See :func:`read_header_bytes`.
    """
    return read_header_bytes(path)[1]


def tensor_specs(header: Dict[str, Any]) -> Dict[str, Tuple[str, Tuple[int, ...]]]:
    """``{key: (dtype, shape)}`` of a safetensors header.

    :param header: Output of :func:`read_safetensors_header`.
    :returns: Specs of every tensor entry.
    """
    return {key: (value["dtype"], tuple(value["shape"])) for key, value in header.items() if not key.startswith("__")}


def _data_bytes(header: Dict[str, Any]) -> int:
    """Size of the data section described by a safetensors header.

    :param header: Output of :func:`read_safetensors_header`.
    :returns: Bytes.
    """
    ends = [value["data_offsets"][1] for key, value in header.items() if not key.startswith("__")]
    return max(ends, default=0)


def data_layout(header: Dict[str, Any], data_bytes: int) -> List[Tuple[str, int, int]]:
    """Tensors of a safetensors header in data-offset order, checked to tile the data section exactly.

    :param header: Output of :func:`read_safetensors_header`.
    :param data_bytes: Size of the data section (file size minus ``__header_bytes__``).
    :returns: ``[(key, start, end)]`` sorted by offset (offsets relative to the data section).
    :raises ValueError: On a gap or overlap between consecutive spans, a span whose length is not
        ``itemsize x numel`` (for known dtypes), or bytes after the last tensor.
    """
    entries = sorted((int(value["data_offsets"][0]), int(value["data_offsets"][1]), key)
                     for key, value in header.items() if not key.startswith("__"))
    position = 0
    for start, end, key in entries:
        if start != position or end < start:
            raise ValueError(f"{key}: data span [{start}, {end}) does not continue at {position} "
                             "(gap or overlap in the safetensors data section)")
        width = DTYPE_BYTES.get(header[key]["dtype"])
        if width is not None and end - start != width * math.prod(header[key]["shape"]):
            raise ValueError(f"{key}: span of {end - start} bytes != {header[key]['dtype']} x {header[key]['shape']}")
        position = end
    if position != data_bytes:
        raise ValueError(f"data section is {data_bytes} bytes but the tensors end at {position}")
    return [(key, start, end) for start, end, key in entries]


def sha256_file(path: Path, chunk: int = 8 * 1024 ** 2) -> str:
    """SHA-256 of a file, streamed.

    :param path: File.
    :param chunk: Read size in bytes.
    :returns: Hex digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def estimate_peak_rss(adapter_bytes: int, max_target_numel: int) -> int:
    """Estimated peak RSS of :func:`merge_adapter` with the streaming writer.

    The adapter tensors stay resident; per targeted weight there are :data:`TEMP_BYTES_PER_ELEMENT` bytes of
    buffers per element (bf16 input buffer, float32 copy, float32 ``B @ A``, bf16 result); untargeted bytes pass
    through one :data:`COPY_CHUNK` buffer; :data:`RSS_OVERHEAD` covers the interpreter and torch.

    :param adapter_bytes: Adapter file size.
    :param max_target_numel: Elements of the largest targeted weight.
    :returns: Bytes.
    """
    return adapter_bytes + TEMP_BYTES_PER_ELEMENT * max_target_numel + COPY_CHUNK + RSS_OVERHEAD


def _copyable_files(source_dir: Path) -> List[Path]:
    """Top-level non-weight files of a directory that belong in the merged directory.

    :param source_dir: Base checkpoint directory, or the ``meta_dir`` that replaces its non-weight files.
    :returns: Sorted files: everything except ``*.safetensors``, the index (written last by the driver),
        ``*.partial``, :data:`EXCLUDED_FILES` (``crc32.txt``: stale for merged shards), hidden files
        (``.gitattributes``, ``.cache``) and ``README*``.
    :raises ValueError: When the directory holds weights in another format.
    """
    files = []
    for path in sorted(Path(source_dir).iterdir()):
        if not path.is_file() or path.name.startswith(".") or path.name.upper().startswith("README"):
            continue
        if path.suffix == ".safetensors" or path.name == INDEX_NAME or path.name.endswith(".partial"):
            continue
        if path.name in EXCLUDED_FILES:
            continue
        if path.suffix.lower() in FOREIGN_WEIGHT_SUFFIXES:
            raise ValueError(f"unexpected weight file in {source_dir}: {path.name}")
        files.append(path)
    return files


def _verbatim_shards(verbatim_dir: Path, base_dir: Path, shards: Sequence[str], targets_per_shard: Dict[str, int],
                     base_raw_headers: Dict[str, bytes], shard_bytes: Dict[str, int]) -> Dict[str, Path]:
    """Validate the shards of ``verbatim_dir`` that replace target-free base shards byte for byte.

    :param verbatim_dir: Directory of replacement shards (e.g. the final merge's target-free shards).
    :param base_dir: Base checkpoint directory (for messages).
    :param shards: Base shard names.
    :param targets_per_shard: Planned targets per base shard.
    :param base_raw_headers: Raw header bytes of every base shard.
    :param shard_bytes: Size of every base shard.
    :returns: ``{shard: path in verbatim_dir}``.
    :raises ValueError: When the directory has no shard, a shard is not in the base index, holds planned
        targets, or differs from the base shard in header bytes or size.
    """
    found = sorted(path.name for path in Path(verbatim_dir).glob("*.safetensors"))
    if not found:
        raise ValueError(f"{verbatim_dir}: no *.safetensors shards to copy verbatim")
    unknown = sorted(set(found) - set(shards))
    if unknown:
        raise ValueError(f"{verbatim_dir}: shards not in the base index ({_examples(unknown)})")
    with_targets = [name for name in found if targets_per_shard.get(name)]
    if with_targets:
        raise ValueError(f"{verbatim_dir}: shards that hold LoRA targets cannot be copied verbatim "
                         f"({_examples(with_targets)})")
    verbatim = {}
    for name in found:
        path = Path(verbatim_dir) / name
        if read_header_bytes(path)[0] != base_raw_headers[name]:
            raise ValueError(f"{path}: header bytes differ from {base_dir / name}")
        if path.stat().st_size != shard_bytes[name]:
            raise ValueError(f"{path}: size {path.stat().st_size} != base {shard_bytes[name]}")
        verbatim[name] = path
    return verbatim


def describe_plan(base_dir: Path, adapter_dir: Path, target_regex: Optional[str] = None,
                  prefixes: Sequence[str] = PEFT_PREFIXES, verbatim_dir: Optional[Path] = None,
                  meta_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Validate the inputs from headers only and describe the merge (no tensor data is read).

    :param base_dir: Base checkpoint directory (``model.safetensors.index.json`` + shards).
    :param adapter_dir: PEFT adapter directory (``adapter_config.json`` + ``adapter_model.safetensors``).
    :param target_regex: Module regex; ``None`` uses the adapter config's ``target_modules``.
    :param prefixes: Adapter key prefixes to strip.
    :param verbatim_dir: Optional directory whose shards replace target-free base shards byte for byte.
    :param meta_dir: Optional directory whose non-weight files replace the base's.
    :returns: Dict with ``plan``, ``adapter_config``, ``scale``, ``target_regex``, ``shards`` (sorted names),
        ``shard_bytes``, ``targets_per_shard``, ``n_targets``, ``base_bytes``, ``adapter_bytes``,
        ``max_target_numel``, ``target_dtypes``, ``base_metadata``, ``verbatim`` (``{shard: path}``),
        ``meta_source``, ``meta_files`` and ``peak_rss_estimate``.
    :raises ValueError: On any plan, header, shape, rank, layout or file-set inconsistency.
    """
    base_dir = Path(base_dir)
    adapter_config = json.loads((Path(adapter_dir) / ADAPTER_CONFIG).read_text())
    scale = lora_scale(adapter_config)
    regex = target_regex if target_regex is not None else target_regex_from_config(adapter_config)
    index = json.loads((base_dir / INDEX_NAME).read_text())
    adapter_path = Path(adapter_dir) / ADAPTER_WEIGHTS
    adapter_header = read_safetensors_header(adapter_path)
    adapter_specs = tensor_specs(adapter_header)
    plan = plan_merge(index, list(adapter_specs), regex, prefixes)
    weight_map: Dict[str, str] = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    on_disk = {path.name for path in base_dir.glob("*.safetensors")}
    if set(shards) - on_disk:
        raise ValueError(f"index names shards that are missing ({_examples(sorted(set(shards) - on_disk))})")
    if on_disk - set(shards):
        stray = sorted(on_disk - set(shards))
        raise ValueError(f"safetensors files not in the index would be dropped ({_examples(stray)})")
    _copyable_files(base_dir)
    rank = int(adapter_config["r"])
    shard_bytes: Dict[str, int] = {}
    raw_headers: Dict[str, bytes] = {}
    base_metadata: Counter = Counter()
    target_dtypes: Counter = Counter()
    max_target_numel = 0
    for shard in shards:
        raw_headers[shard], header = read_header_bytes(base_dir / shard)
        shard_bytes[shard] = (base_dir / shard).stat().st_size
        data_layout(header, shard_bytes[shard] - header["__header_bytes__"])
        base_metadata[json.dumps(header.get("__metadata__"), sort_keys=True)] += 1
        specs = tensor_specs(header)
        expected = {key for key, name in weight_map.items() if name == shard}
        if set(specs) != expected:
            raise ValueError(f"{shard}: tensors differ from the index weight_map "
                             f"(extra {_examples(sorted(set(specs) - expected))}; "
                             f"missing {_examples(sorted(expected - set(specs)))})")
        for key, (_, a_key, b_key) in plan.items():
            if weight_map[key] != shard:
                continue
            dtype, shape = specs[key]
            a_shape, b_shape = adapter_specs[a_key][1], adapter_specs[b_key][1]
            if dtype not in FLOAT_DTYPES or len(shape) != 2:
                raise ValueError(f"{key}: base weight {dtype} {shape} is not a 2-D float tensor")
            if len(a_shape) != 2 or len(b_shape) != 2 or a_shape != (rank, shape[1]) or b_shape != (shape[0], rank):
                raise ValueError(f"{key}: W {shape} with r={rank} does not fit A {a_shape} / B {b_shape}")
            target_dtypes[dtype] += 1
            max_target_numel = max(max_target_numel, shape[0] * shape[1])
    targets_per_shard = Counter(shard for shard, _, _ in plan.values())
    verbatim: Dict[str, Path] = {}
    if verbatim_dir is not None:
        verbatim = _verbatim_shards(Path(verbatim_dir), base_dir, shards, targets_per_shard, raw_headers, shard_bytes)
    meta_source = Path(meta_dir) if meta_dir is not None else base_dir
    meta_files = [path.name for path in _copyable_files(meta_source)]
    if meta_dir is not None and "config.json" not in meta_files:
        raise ValueError(f"{meta_dir}: meta_dir has no config.json")
    adapter_bytes = adapter_path.stat().st_size
    return {
        "plan": plan,
        "adapter_config": adapter_config,
        "scale": scale,
        "target_regex": regex,
        "shards": shards,
        "shard_bytes": shard_bytes,
        "targets_per_shard": {shard: targets_per_shard.get(shard, 0) for shard in shards},
        "n_targets": len(plan),
        "base_bytes": sum(shard_bytes.values()),
        "adapter_bytes": adapter_bytes,
        "max_target_numel": max_target_numel,
        "target_dtypes": dict(target_dtypes),
        "base_metadata": dict(base_metadata),
        "verbatim": verbatim,
        "meta_source": meta_source,
        "meta_files": meta_files,
        "peak_rss_estimate": estimate_peak_rss(adapter_bytes, max_target_numel),
    }


def _load_progress(path: Path, fingerprint: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Progress records of earlier runs of the same merge.

    Only newline-terminated lines are records: an unterminated last line is what a crash during
    :func:`_append_progress` leaves, and is ignored (that append's rename never happened).

    :param path: Progress JSON-lines file in the output directory.
    :param fingerprint: Fingerprint of this merge.
    :returns: ``{shard: record}`` for records whose fingerprint equals ``fingerprint`` (last one wins).
    :raises json.JSONDecodeError: On an unparsable newline-terminated line (the file was edited or corrupted).
    """
    if not path.exists():
        return {}
    complete = path.read_text().rpartition("\n")[0]
    records = {}
    for line in complete.splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("fingerprint") == fingerprint:
                records[record["shard"]] = record
    return records


def _append_progress(path: Path, record: Dict[str, Any]) -> None:
    """Append one durable progress record, first dropping an unterminated last line left by a crash.

    :param path: Progress JSON-lines file.
    :param record: Record to append.
    """
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        if size:
            handle.seek(size - 1)
            if handle.read(1) != b"\n":
                handle.seek(0)
                handle.truncate(handle.read().rfind(b"\n") + 1)
        handle.write((json.dumps(record) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())


def _read_into(handle: BinaryIO, view: memoryview) -> None:
    """Fill ``view`` from ``handle``.

    :param handle: Binary file positioned at the data to read.
    :param view: Writable buffer to fill completely.
    :raises RuntimeError: On end of file before ``view`` is full.
    """
    position = 0
    while position < len(view):
        count = handle.readinto(view[position:])
        if not count:
            raise RuntimeError(f"short read: {position} of {len(view)} bytes")
        position += count


def _copy_bytes(source: BinaryIO, destination: BinaryIO, digest: Any, nbytes: int, buffer: bytearray) -> None:
    """Copy ``nbytes`` from the current position of ``source`` to ``destination`` through one buffer.

    :param source: Binary input file.
    :param destination: Binary output file.
    :param digest: ``hashlib`` object updated with the bytes written.
    :param nbytes: Bytes to copy.
    :param buffer: Reused copy buffer.
    :raises RuntimeError: On end of file before ``nbytes`` were copied.
    """
    view = memoryview(buffer)
    remaining = nbytes
    while remaining:
        count = source.readinto(view[: min(len(buffer), remaining)])
        if not count:
            raise RuntimeError(f"short read: {nbytes - remaining} of {nbytes} bytes")
        destination.write(view[:count])
        digest.update(view[:count])
        remaining -= count


def _copy_file(source: Path, partial: Path) -> str:
    """Byte-copy a shard through one buffer, hashing what is written, then fsync.

    :param source: Shard to copy.
    :param partial: Temporary output path.
    :returns: sha256 of the bytes written.
    """
    digest = hashlib.sha256()
    size = source.stat().st_size
    buffer = bytearray(min(COPY_CHUNK, max(size, 1)))
    with source.open("rb") as src, partial.open("wb") as dst:
        _copy_bytes(src, dst, digest, size, buffer)
        dst.flush()
        os.fsync(dst.fileno())
    return digest.hexdigest()


def _write_merged_shard(source: Path, partial: Path, pairs: Dict[str, Pair], adapter: Dict[str, torch.Tensor],
                        scale: float) -> Tuple[List[Dict[str, Any]], str]:
    """Stream one base shard to ``partial``, merging its targets one tensor at a time.

    The header bytes are copied verbatim; the data section is walked in offset order: a targeted tensor is read
    into one buffer, merged (:func:`merge_tensor`) and written; every other span is copied through one
    :data:`COPY_CHUNK` buffer.

    :param source: Base shard.
    :param partial: Temporary output path.
    :param pairs: Plan entries for this shard.
    :param adapter: Adapter tensors.
    :param scale: LoRA scale.
    :returns: ``(stats, sha256)``: per-tensor stats in offset order and the sha256 of the bytes written.
    :raises ValueError: On a non-contiguous data section, a planned weight or adapter tensor that is missing, or
        a targeted dtype outside :data:`FLOAT_DTYPES`.
    :raises RuntimeError: On a big-endian host, a short read, or a merged tensor whose size differs from its span.
    """
    if sys.byteorder != "little":
        raise RuntimeError("safetensors data is little-endian; the streaming writer needs a little-endian host")
    header_raw, header = read_header_bytes(source)
    offset = header["__header_bytes__"]
    layout = data_layout(header, source.stat().st_size - offset)
    _check_pairs_available([key for key, _, _ in layout], pairs, adapter)
    spans = [end - start for key, start, end in layout if key not in pairs]
    buffer = bytearray(min(COPY_CHUNK, max(spans, default=1) or 1))
    digest = hashlib.sha256()
    stats = []
    with source.open("rb") as src, partial.open("wb") as dst:
        dst.write(header_raw)
        digest.update(header_raw)
        src.seek(offset)
        for key, start, end in layout:
            if key not in pairs:
                _copy_bytes(src, dst, digest, end - start, buffer)
                continue
            spec = header[key]
            dtype = TORCH_DTYPES.get(spec["dtype"])
            if dtype is None:
                raise ValueError(f"{key}: targeted dtype {spec['dtype']} is not one of {sorted(FLOAT_DTYPES)}")
            raw = bytearray(end - start)
            _read_into(src, memoryview(raw))
            weight = (torch.frombuffer(raw, dtype=dtype).reshape(spec["shape"]) if raw
                      else torch.empty(spec["shape"], dtype=dtype))
            _, a_key, b_key = pairs[key]
            merged, entry = merge_tensor(weight, adapter[a_key], adapter[b_key], scale, key)
            del weight, raw
            payload = merged.contiguous().view(torch.uint8).reshape(-1).numpy()
            if payload.nbytes != end - start:
                raise RuntimeError(f"{key}: merged tensor is {payload.nbytes} bytes, base span {end - start}")
            dst.write(payload)
            digest.update(payload)
            del merged, payload
            stats.append(entry)
        dst.flush()
        os.fsync(dst.fileno())
    return stats, digest.hexdigest()


def _check_same_layout(base: Path, out: Path) -> int:
    """Require an output shard to have the base shard's exact header bytes and size.

    :param base: Base shard.
    :param out: Output shard (final or ``.partial``).
    :returns: The output size in bytes.
    :raises RuntimeError: On a header or size difference (the message separates header and data sections).
    """
    (base_raw, base_header), (out_raw, out_header) = read_header_bytes(base), read_header_bytes(out)
    base_size, out_size = base.stat().st_size, out.stat().st_size
    if out_raw != base_raw:
        detail = ("tensor dtypes/shapes differ" if tensor_specs(base_header) != tensor_specs(out_header)
                  else "header bytes differ")
        raise RuntimeError(f"{out.name}: {detail} from the base shard; left in place for inspection")
    if base_size != out_size:
        raise RuntimeError(f"{out.name}: size {out_size} != base {base_size} (header {out_header['__header_bytes__']} "
                           f"vs {base_header['__header_bytes__']}, data {_data_bytes(out_header)} vs "
                           f"{_data_bytes(base_header)}); left in place for inspection")
    return out_size


def _resumable_shards(info: Dict[str, Any], out_dir: Path, progress: Dict[str, Dict[str, Any]],
                      verify_resumed: bool) -> List[str]:
    """Output shards a previous run of the same merge finished (final name, base size, matching record).

    :param info: :func:`describe_plan` result.
    :param out_dir: Output directory.
    :param progress: Records of this merge (:func:`_load_progress`).
    :param verify_resumed: Re-hash each resumable shard against its record's ``out_sha256``.
    :returns: Shard names to skip.
    :raises ValueError: On a base-sized output shard without a record of this merge, a record without the
        verified size / sha256, or (``verify_resumed``) a shard whose sha256 differs from its record.
    """
    resumable = []
    for shard in info["shards"]:
        out = out_dir / shard
        if not out.exists() or out.stat().st_size != info["shard_bytes"][shard]:
            continue
        record = progress.get(shard)
        if record is None:
            raise ValueError(f"{out} has the base shard's size but no progress record from this merge "
                             f"(different adapter, base, regex or verbatim shards?); remove it to re-merge")
        if record.get("out_bytes") != out.stat().st_size or not record.get("out_sha256"):
            raise ValueError(f"{out}: its progress record lacks the verified size / sha256; remove it to re-merge")
        if verify_resumed and sha256_file(out) != record["out_sha256"]:
            raise ValueError(f"{out}: sha256 differs from its progress record (corrupt or replaced); "
                             f"remove it to re-merge")
        resumable.append(shard)
    return resumable


def merge_adapter(base_dir: Path, adapter_dir: Path, out_dir: Path, report_path: Path,
                  target_regex: Optional[str] = None, prefixes: Sequence[str] = PEFT_PREFIXES,
                  memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET, verbatim_dir: Optional[Path] = None,
                  meta_dir: Optional[Path] = None, verify_resumed: bool = False) -> Dict[str, Any]:
    """Merge a LoRA adapter into a sharded checkpoint, shard by shard, streaming and resumably.

    :param base_dir: Base checkpoint directory (``model.safetensors.index.json`` + shards + tokenizer/config).
    :param adapter_dir: PEFT adapter directory (``adapter_config.json`` + ``adapter_model.safetensors``).
    :param out_dir: Output directory (created); shards keep their base file names.
    :param report_path: Where the JSON report is written.
    :param target_regex: Module regex; ``None`` uses the adapter config's ``target_modules``.
    :param prefixes: Adapter key prefixes to strip.
    :param memory_budget_bytes: Refuse to start when :func:`estimate_peak_rss` exceeds this.
    :param verbatim_dir: Optional directory whose shards (target-free, base header bytes and size) are copied
        instead of the same-named base shards.
    :param meta_dir: Optional directory whose non-weight files are copied instead of the base's.
    :param verify_resumed: Re-hash resumed shards against their progress records before trusting them.
    :returns: The report: ``n_targets``, ``scale``, ``target_regex``, ``fingerprint``, ``shards`` (per shard:
        ``file``, ``status`` merged / copied / verbatim / resumed, ``in_bytes``, ``out_bytes``, ``out_sha256``,
        ``n_targets``, ``seconds``), ``resumed_shards``, ``verbatim_shards``, ``tensors`` (per-tensor stats),
        ``summary``, ``copied_files``, ``meta_source``, ``total_bytes``, ``peak_rss_estimate``, ``wall_seconds``.
    :raises ValueError: On invalid inputs (see :func:`describe_plan`), ``out_dir`` equal to ``base_dir``,
        ``verbatim_dir`` or ``meta_dir``, a memory estimate over budget, too little free space, or an output shard
        of the right size left by a different merge (see :func:`_resumable_shards`).
    :raises RuntimeError: When a written shard differs in header/size from its base shard or a pair is not
        consumed exactly once.
    """
    started = time.monotonic()
    base_dir, adapter_dir, out_dir = Path(base_dir), Path(adapter_dir), Path(out_dir)
    for label, other in (("base_dir", base_dir), ("verbatim_dir", verbatim_dir), ("meta_dir", meta_dir)):
        if other is not None and Path(other).resolve() == out_dir.resolve():
            raise ValueError(f"out_dir must differ from {label} (shards and files are replaced by name)")
    info = describe_plan(base_dir, adapter_dir, target_regex, prefixes, verbatim_dir, meta_dir)
    plan: Dict[str, Pair] = info["plan"]
    verbatim: Dict[str, Path] = info["verbatim"]
    if info["peak_rss_estimate"] > memory_budget_bytes:
        raise ValueError(f"estimated peak RSS {info['peak_rss_estimate']} bytes exceeds the budget "
                         f"{memory_budget_bytes}")
    fingerprint = {
        "adapter_sha256": sha256_file(adapter_dir / ADAPTER_WEIGHTS),
        "base_index_sha256": sha256_file(base_dir / INDEX_NAME),
        "scale": info["scale"],
        "target_regex": info["target_regex"],
        "prefixes": list(prefixes),
        "verbatim_shards": {name: info["shard_bytes"][name] for name in sorted(verbatim)},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / INDEX_NAME).unlink(missing_ok=True)  # written again only once every shard of this run is in place
    progress_path = out_dir / PROGRESS_NAME
    progress = _load_progress(progress_path, fingerprint)
    resumable = _resumable_shards(info, out_dir, progress, verify_resumed)
    copy_files = [info["meta_source"] / name for name in info["meta_files"]]
    to_write = sum(info["shard_bytes"][shard] for shard in info["shards"] if shard not in resumable)
    to_write += sum(path.stat().st_size for path in copy_files) + (base_dir / INDEX_NAME).stat().st_size
    free = shutil.disk_usage(out_dir).free
    if free < to_write + FREE_SPACE_MARGIN:
        raise ValueError(f"{out_dir}: {free} bytes free, need {to_write} + {FREE_SPACE_MARGIN} margin")

    adapter: Optional[Dict[str, torch.Tensor]] = None
    consumed: Counter = Counter()
    shard_rows: List[Dict[str, Any]] = []
    tensor_stats: List[Dict[str, Any]] = []
    for shard in info["shards"]:
        shard_started = time.monotonic()
        pairs = {key: pair for key, pair in plan.items() if pair[0] == shard}
        source, out = base_dir / shard, out_dir / shard
        if shard in resumable:
            record = progress[shard]
            consumed.update(list(pairs))
            tensor_stats.extend(record["stats"])
            shard_rows.append({"file": shard, "status": "resumed", "in_bytes": info["shard_bytes"][shard],
                               "out_bytes": out.stat().st_size, "out_sha256": record["out_sha256"],
                               "n_targets": len(pairs), "seconds": record.get("seconds")})
            print(f"{shard}: resumed (merged earlier, {len(pairs)} targets)", flush=True)
            continue
        partial = out_dir / f"{shard}.partial"
        partial.unlink(missing_ok=True)
        if pairs:
            if adapter is None:
                adapter = load_file(str(adapter_dir / ADAPTER_WEIGHTS))
            stats, out_sha256 = _write_merged_shard(source, partial, pairs, adapter, info["scale"])
            status = "merged"
        elif shard in verbatim:
            stats, out_sha256, status = [], _copy_file(verbatim[shard], partial), "verbatim"
        else:
            stats, out_sha256, status = [], _copy_file(source, partial), "copied"
        out_bytes = _check_same_layout(source, partial)
        seconds = round(time.monotonic() - shard_started, 2)
        _append_progress(progress_path, {"fingerprint": fingerprint, "shard": shard, "status": status,
                                         "in_bytes": info["shard_bytes"][shard], "out_bytes": out_bytes,
                                         "out_sha256": out_sha256, "seconds": seconds, "stats": stats})
        os.replace(partial, out)
        consumed.update(stats_entry["key"] for stats_entry in stats)
        tensor_stats.extend(stats)
        shard_rows.append({"file": shard, "status": status, "in_bytes": info["shard_bytes"][shard],
                           "out_bytes": out_bytes, "out_sha256": out_sha256, "n_targets": len(pairs),
                           "seconds": seconds})
        print(f"{shard}: {status} ({len(pairs)} targets, {out_bytes} bytes, {seconds} s)", flush=True)
    del adapter

    if set(consumed) != set(plan) or any(count != 1 for count in consumed.values()):
        twice = [key for key, count in consumed.items() if count != 1]
        raise RuntimeError(f"adapter pairs not consumed exactly once (missing "
                           f"{_examples(sorted(set(plan) - set(consumed)))}; repeated {_examples(twice)})")
    copied = []
    for path in copy_files:
        shutil.copyfile(path, out_dir / path.name)
        copied.append(path.name)
    for shard in info["shards"]:
        _check_same_layout(base_dir / shard, out_dir / shard)
    shutil.copyfile(base_dir / INDEX_NAME, out_dir / INDEX_NAME)  # last: its presence marks a complete merge

    ratios = sorted(entry["delta_fro_over_w_fro"] for entry in tensor_stats
                    if entry.get("delta_fro_over_w_fro") is not None)
    changed = sorted(entry["changed_fraction"] for entry in tensor_stats)
    report = {
        "base_dir": str(base_dir),
        "adapter_dir": str(adapter_dir),
        "out_dir": str(out_dir),
        "adapter_config": {name: info["adapter_config"].get(name) for name in
                           ("r", "lora_alpha", "use_rslora", "base_model_name_or_path", "peft_version")},
        "n_targets": len(plan),
        "scale": info["scale"],
        "target_regex": info["target_regex"],
        "fingerprint": fingerprint,
        "target_dtypes": info["target_dtypes"],
        "n_shards": len(info["shards"]),
        "shards": shard_rows,
        "resumed_shards": [row["file"] for row in shard_rows if row["status"] == "resumed"],
        "verbatim_shards": {name: str(path) for name, path in sorted(verbatim.items())},
        "tensors": sorted(tensor_stats, key=lambda entry: entry["key"]),
        "summary": {
            "delta_fro_over_w_fro": _quantiles(ratios),
            "changed_fraction": _quantiles(changed),
            "max_abs_delta": max((entry["max_abs_delta"] for entry in tensor_stats), default=None),
        },
        "copied_files": copied,
        "meta_source": str(info["meta_source"]),
        "total_bytes": sum(row["out_bytes"] for row in shard_rows),
        "peak_rss_estimate": info["peak_rss_estimate"],
        "wall_seconds": round(time.monotonic() - started, 2),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = report_path.with_name(report_path.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(tmp, report_path)
    return report


def _quantiles(values: Sequence[float]) -> Optional[Dict[str, float]]:
    """Min / median / max of sorted values.

    :param values: Sorted numbers.
    :returns: ``{"min", "median", "max"}`` or ``None`` when empty.
    """
    if not values:
        return None
    return {"min": values[0], "median": values[len(values) // 2], "max": values[-1]}


def _on_home_volume(path: Path) -> bool:
    """Whether ``path`` (or its nearest existing parent) is on the same filesystem as the home directory.

    :param path: Candidate output path.
    :returns: True for the local boot data volume.
    """
    probe = path.resolve()
    while not probe.exists():
        probe = probe.parent
    return probe.stat().st_dev == Path.home().stat().st_dev


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface.

    :returns: The parser.
    """
    parser = argparse.ArgumentParser(description="Merge a PEFT LoRA adapter into a sharded bf16 checkpoint.")
    parser.add_argument("--base-dir", type=Path, required=True, help="bf16 base (index + shards + tokenizer)")
    parser.add_argument("--adapter-dir", type=Path, required=True,
                        help="adapter_config.json + adapter_model.safetensors")
    parser.add_argument("--out-dir", type=Path, required=True, help="merged output directory")
    parser.add_argument("--report", type=Path, required=True, help="JSON report path")
    parser.add_argument("--target-regex", default=None, help="module regex (default: adapter target_modules)")
    parser.add_argument("--adapter-prefix", action="append", default=None,
                        help=f"adapter key prefix to strip (repeatable; default {PEFT_PREFIXES[0]!r})")
    parser.add_argument("--verbatim-shard-dir", type=Path, default=None,
                        help="target-free shards here (base header bytes + size) replace the base's byte for byte")
    parser.add_argument("--meta-dir", type=Path, default=None,
                        help="take the non-weight files (config, tokenizer, template) from here instead of the base")
    parser.add_argument("--verify-resumed", action="store_true",
                        help="re-hash resumed shards against their progress records")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="torch intra-op threads")
    parser.add_argument("--memory-budget-gb", type=float, default=DEFAULT_MEMORY_BUDGET / 1e9,
                        help="refuse to start above this estimated peak RSS (decimal GB)")
    parser.add_argument("--max-local-gb", type=float, default=2.0,
                        help="refuse to write more than this to the home volume (decimal GB)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point.

    :param argv: Arguments (default ``sys.argv[1:]``).
    :raises SystemExit: When a non-dry run would write more than ``--max-local-gb`` to the home volume.
    """
    args = build_parser().parse_args(argv)
    torch.set_num_threads(max(1, args.threads))
    prefixes = tuple(args.adapter_prefix) if args.adapter_prefix else PEFT_PREFIXES
    if args.dry_run:
        info = describe_plan(args.base_dir, args.adapter_dir, args.target_regex, prefixes, args.verbatim_shard_dir,
                             args.meta_dir)
        summary = {
            "n_targets": info["n_targets"],
            "n_shards": len(info["shards"]),
            "n_shards_with_targets": sum(1 for count in info["targets_per_shard"].values() if count),
            "base_bytes": info["base_bytes"],
            "adapter_bytes": info["adapter_bytes"],
            "scale": info["scale"],
            "target_regex": info["target_regex"],
            "target_dtypes": info["target_dtypes"],
            "base_metadata": info["base_metadata"],
            "max_target_numel": info["max_target_numel"],
            "peak_rss_estimate": info["peak_rss_estimate"],
            "targets_per_shard": info["targets_per_shard"],
            "verbatim_shards": sorted(info["verbatim"]),
            "meta_source": str(info["meta_source"]),
            "meta_files": info["meta_files"],
        }
        print(json.dumps(summary, indent=2))
        return
    base_bytes = sum(path.stat().st_size for path in args.base_dir.glob("*.safetensors"))
    if _on_home_volume(args.out_dir) and base_bytes > args.max_local_gb * 1e9:
        raise SystemExit(f"refusing to write {base_bytes} bytes to the home volume ({args.out_dir}); use the NAS")
    report = merge_adapter(args.base_dir, args.adapter_dir, args.out_dir, args.report, args.target_regex, prefixes,
                           int(args.memory_budget_gb * 1e9), verbatim_dir=args.verbatim_shard_dir,
                           meta_dir=args.meta_dir, verify_resumed=args.verify_resumed)
    print(json.dumps({key: report[key] for key in ("n_targets", "scale", "n_shards", "resumed_shards",
                                                   "verbatim_shards", "total_bytes", "wall_seconds", "summary")},
                     indent=2))


if __name__ == "__main__":
    main()
