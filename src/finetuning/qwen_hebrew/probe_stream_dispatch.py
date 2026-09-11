"""Reproduce streamed Qwen3-VL image truncation on CPU without model downloads.

Run with ``.venv/bin/python -m src.finetuning.qwen_hebrew.probe_stream_dispatch``.
An optional ``--loader-source`` imports a saved upstream Accelerate data_loader.py
under the installed package, allowing its dispatcher to be tested independently.
This does not modify the installed packages or access production models.
"""

import argparse
import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import accelerate
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, IterableDataset
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel


class PatchRows(IterableDataset):
    """Yield already-collated rows with a packed image-patch dimension."""

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Yield two deterministic images without fetching external data.

        :returns: An iterator over batches with sixteen image patches each.
        """
        for _ in range(2):
            yield {
                "input_ids": torch.arange(8).reshape(1, 8),
                "attention_mask": torch.ones(1, 8, dtype=torch.long),
                "pixel_values": torch.arange(192).reshape(16, 12).float() / 192,
                "image_grid_thw": torch.tensor([[1, 4, 4]]),
            }


def load_dispatch_module(source: Path | None) -> ModuleType:
    """Load the installed or explicitly supplied Accelerate dispatcher.

    :param source: Optional path to a trusted upstream data_loader.py file.
    :returns: The module whose prepare_data_loader function will be exercised.
    """
    if source is None:
        return accelerate.data_loader
    spec = importlib.util.spec_from_file_location("accelerate.v21_probe_loader", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Check dispatcher corruption, its workaround, and silent vision broadcast.

    :returns: None; print evidence and raise if the expected checks fail.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loader-source", type=Path)
    args = parser.parse_args()
    module = load_dispatch_module(args.loader_source)
    Accelerator(cpu=True)
    raw = next(iter(DataLoader(PatchRows(), batch_size=None)))
    batches = {}
    wrappers = {}
    for name, dispatch in (("default", None), ("disabled", False)):
        prepared = module.prepare_data_loader(
            DataLoader(PatchRows(), batch_size=None),
            device=torch.device("cpu"),
            put_on_device=True,
            dispatch_batches=dispatch,
        )
        batches[name] = next(iter(prepared))
        wrappers[name] = type(prepared).__name__

    bad, good = batches["default"], batches["disabled"]
    expected = int(raw["image_grid_thw"].prod(dim=-1).sum())
    assert expected == 16
    assert bad["pixel_values"].shape[0] == 1
    assert torch.equal(bad["pixel_values"], raw["pixel_values"][:1])
    assert all(torch.equal(good[key], value) for key, value in raw.items())
    assert torch.equal(bad["input_ids"], raw["input_ids"])
    assert torch.equal(bad["image_grid_thw"], raw["image_grid_thw"])

    torch.manual_seed(3407)
    vision = Qwen3VLVisionModel(Qwen3VLVisionConfig(
        depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
        out_hidden_size=32, patch_size=2, temporal_patch_size=1,
        spatial_merge_size=2, num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )).eval()
    with torch.no_grad():
        damaged, _ = vision(bad["pixel_values"], bad["image_grid_thw"])
        repeated, _ = vision(
            bad["pixel_values"].repeat(expected, 1), bad["image_grid_thw"]
        )
        intact, _ = vision(good["pixel_values"], good["image_grid_thw"])
    assert damaged.shape == intact.shape == (4, 32)
    assert torch.allclose(damaged, repeated, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(damaged, intact, atol=1e-6, rtol=1e-5)
    print(json.dumps({
        "installed_accelerate": accelerate.__version__,
        "loader_source": str(module.__file__),
        "wrappers": wrappers,
        "expected_patch_rows": expected,
        "default_patch_rows": bad["pixel_values"].shape[0],
        "disabled_patch_rows": good["pixel_values"].shape[0],
        "disabled_preserves_every_tensor": True,
        "damaged_vision_output_shape": list(damaged.shape),
        "damaged_equals_repeated_first_patch": True,
    }, indent=2))


if __name__ == "__main__":
    main()
