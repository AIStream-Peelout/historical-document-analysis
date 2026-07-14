"""Fuse trained adapters into a standalone model and export for LM Studio.

mlx-vlm 0.6.x has no top-level fuse CLI, so this script does the merge
explicitly:

1. Load the base model with adapters applied (LoRA layers + any full-weight
   vision-tower deltas saved by training with ``train_vision``).
2. Fuse every LoRA/DoRA layer back into plain linear layers.
3. Save weights + config + processor into a self-contained model directory.
4. Sanity-generate on one validation crop before declaring success.

The exported directory loads directly in LM Studio (MLX engine): import it
via ``lms import <dir>`` or copy it under ``~/.lmstudio/models/local/``. The
resulting LM Studio model id is what you pass to the benchmark via
``--lm_studio_models``.

  .venv-mlx/bin/python -m src.finetuning.qwen_hebrew.export_lmstudio \\
      --adapter models/adapters/qwen3vl_hebrew_stage1 \\
      --output models/qwen3-vl-8b-heb-stage1
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Optional

from src.finetuning.qwen_hebrew.quick_eval import (
    DEFAULT_DATASET_DIR,
    SAMPLE_SEED,
    load_model,
    resolve_image_path,
    select_val_samples,
    transcribe,
)


def fuse_lora_layers(model) -> int:
    """Fuse all LoRA/DoRA layers in-place into plain linear layers.

    :param model: Model with adapter layers applied.
    :returns: Number of layers fused.
    """
    from mlx_vlm.trainer.dora_layers import DoRAEmbedding, DoRALinear
    from mlx_vlm.trainer.lora_layers import (
        LoRAEmbedding,
        LoRALinear,
        LoRASwitchLinear,
    )
    from mlx_vlm.trainer.utils import set_module_by_name

    fusable = (LoRALinear, LoRASwitchLinear, LoRAEmbedding, DoRALinear, DoRAEmbedding)
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, fusable)
    ]
    for name, module in targets:
        set_module_by_name(model, name, module.fuse())
    return len(targets)


def export(
    model_path: str,
    adapter_path: str,
    output_dir: Path,
    dataset_dir: Path,
    skip_sanity: bool = False,
) -> None:
    """Fuse and export the fine-tuned model.

    :param model_path: Base model (HF hub id or local path).
    :param adapter_path: Adapter directory produced by training.
    :param output_dir: Destination for the fused, self-contained model.
    :param dataset_dir: Dataset dir used for the sanity generation.
    :param skip_sanity: Skip the post-fuse sanity generation.
    """
    import tempfile

    from mlx_vlm.utils import save_config, save_weights

    print(f"Loading {model_path} + adapters {adapter_path}")
    model, processor, config = load_model(model_path, adapter_path)

    n_fused = fuse_lora_layers(model)
    print(f"Fused {n_fused} adapter layers")
    if n_fused == 0:
        raise RuntimeError(
            f"No adapter layers found to fuse — is {adapter_path} a valid "
            f"training output (adapter_config.json + adapters.safetensors)?"
        )

    if not skip_sanity:
        print("Sanity generation on one validation crop…")
        sample = select_val_samples(dataset_dir, 1, SAMPLE_SEED)[0]
        with tempfile.TemporaryDirectory() as tmp:
            image_path = resolve_image_path(sample["image"], Path(tmp))
            text = transcribe(model, processor, config, image_path, sample["question"])
        preview = text.strip().replace("\n", " ")[:120]
        print(f"  → {preview!r}")
        if not preview:
            raise RuntimeError("Fused model generated empty output — aborting export.")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving weights → {output_dir}")
    save_weights(output_dir, model)

    cfg = dict(config)
    cfg.pop("lora", None)
    save_config(cfg, output_dir / "config.json")
    processor.save_pretrained(output_dir)

    # Carry over auxiliary files (chat template, generation config) if present
    # locally — harmless to skip when the base came from the hub cache.
    base = Path(model_path)
    if base.is_dir():
        for aux in ("generation_config.json", "chat_template.json"):
            src = base / aux
            if src.exists() and not (output_dir / aux).exists():
                shutil.copy(src, output_dir / aux)

    print(
        f"\n✅ Exported → {output_dir}\n"
        f"   Load in LM Studio:  lms import {output_dir}\n"
        f"   (or copy the directory under ~/.lmstudio/models/local/)\n"
        f"   Then benchmark with: --lm_studio_models <its LM Studio id>"
    )


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point.

    :param argv: Optional argument list (defaults to ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description="Fuse adapters and export for LM Studio.")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3-VL-8B-Instruct-bf16")
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--skip_sanity", action="store_true")
    args = parser.parse_args(argv)

    export(
        model_path=args.model,
        adapter_path=args.adapter,
        output_dir=args.output,
        dataset_dir=args.dataset_dir,
        skip_sanity=args.skip_sanity,
    )


if __name__ == "__main__":
    main()
