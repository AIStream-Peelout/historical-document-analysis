"""Local MLX LoRA training wrapper for Qwen3-VL Hebrew fine-tuning.

Drives ``mlx_vlm``'s SFT trainer through its Python API (the ``mlx_vlm.lora``
CLI only accepts hub datasets via ``load_dataset``; our artifact is a
``save_to_disk`` directory). Adds on top of the stock trainer:

- YAML experiment configs (see ``configs/``)
- warmup + cosine LR schedule (stock CLI is constant-LR)
- label masking always on (``train_on_completions=True`` — the old Gemma
  pipeline's unmasked-labels bug must never recur)
- vision-tower presence assertion before training
- image-resolution policy applied to the processor (min/max pixels)
- W&B logging by tee-parsing the trainer's stdout loss lines

Run inside the dedicated training venv:

  .venv-mlx/bin/python -m src.finetuning.qwen_hebrew.mlx_train \\
      --config src/finetuning/qwen_hebrew/configs/smoke.yaml
"""

import argparse
import io
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

_REPO = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_DIR = _REPO / "src/datasets/processed/qwen_hebrew_talmud"

# Matches the trainer's per-report line, e.g.
# "Iter 10: Train loss 2.31400000, Learning Rate 1.000e-04, It/sec 0.42, ..."
_TRAIN_LINE_RE = re.compile(
    r"Iter (\d+): Train loss (?:\x1b\[\d+m)?([\d.]+)(?:\x1b\[\d+m)? ?, "
    r"Learning Rate ([\d.e+-]+), It/sec ([\d.]+), Tokens/sec ([\d.]+)"
)
_VAL_LINE_RE = re.compile(r"Iter (\d+): Val loss ([\d.]+)")


@dataclass
class ExperimentConfig:
    """Training experiment configuration loaded from YAML.

    :param model_path: HF hub id or local path of the base model.
    :param dataset_dir: ``save_to_disk`` dataset directory.
    :param train_split: Split used for training.
    :param val_split: Split used for validation (empty string disables).
    :param task_filter: Keep only rows with this ``task`` value ('' = all).
    :param max_samples: Optional cap on training rows (0 = all).
    :param adapter_dir: Output directory for adapter checkpoints.
    :param resume_adapter: Optional adapter path to resume from.
    :param train_vision: Unfreeze the vision tower + projector.
    :param lora_rank: LoRA rank.
    :param lora_alpha: LoRA alpha.
    :param lora_dropout: LoRA dropout.
    :param learning_rate: Peak learning rate.
    :param min_learning_rate: Final LR after cosine decay.
    :param warmup_fraction: Fraction of total steps used for linear warmup.
    :param batch_size: Per-step batch size.
    :param gradient_accumulation_steps: Gradient accumulation steps.
    :param epochs: Epochs over the (filtered) training split.
    :param max_seq_length: Max combined sequence length (vision + text tokens).
    :param grad_checkpoint: Enable gradient checkpointing.
    :param grad_clip: Gradient clipping value.
    :param min_pixels: Processor minimum image pixels.
    :param max_pixels: Processor maximum image pixels.
    :param steps_per_report: Steps between loss reports.
    :param steps_per_eval: Steps between validation passes.
    :param steps_per_save: Steps between adapter checkpoints.
    :param val_batches: Validation batches per eval (-1 = full val set).
    :param wandb_project: W&B project name ('' disables logging).
    :param run_name: W&B run name.
    """

    model_path: str = "mlx-community/Qwen3-VL-8B-Instruct-bf16"
    dataset_dir: str = str(DEFAULT_DATASET_DIR)
    train_split: str = "train"
    val_split: str = "val"
    task_filter: str = "crop_transcribe"
    max_samples: int = 0
    adapter_dir: str = "models/adapters/qwen3vl_hebrew"
    resume_adapter: str = ""
    train_vision: bool = True
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    learning_rate: float = 5e-5
    min_learning_rate: float = 1e-6
    warmup_fraction: float = 0.03
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    epochs: int = 2
    max_seq_length: int = 6144
    grad_checkpoint: bool = True
    grad_clip: float = 1.0
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 2_800_000
    steps_per_report: int = 10
    steps_per_eval: int = 250
    steps_per_save: int = 500
    val_batches: int = 8
    wandb_project: str = "qwen-hebrew-finetune"
    run_name: str = ""


def load_config(path: Path) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a YAML file.

    :param path: Path to the YAML config.
    :returns: Populated config; unknown keys raise ``TypeError``.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ExperimentConfig(**raw)


def assert_vision_model(model) -> None:
    """Fail fast if the loaded model has no vision tower.

    The legacy Gemma pipeline once trained ``gemma-3-1b-it`` (text-only) in a
    "vision" configuration; this assertion makes that class of mistake loud.

    :param model: Loaded mlx-vlm model.
    :raises ValueError: If no vision stack is present.
    """
    has_vision = any(
        hasattr(model, attr) and getattr(model, attr) is not None
        for attr in ("vision_model", "vision_tower")
    )
    vision_config = getattr(getattr(model, "config", None), "vision_config", None)
    if not (has_vision or vision_config):
        raise ValueError(
            f"Model has no vision tower — refusing to train. "
            f"Check model_path (got a text-only model?)."
        )


def apply_resolution_policy(processor, min_pixels: int, max_pixels: int) -> None:
    """Set the processor's dynamic-resolution bounds.

    Dense Hebrew script is unreadable at low resolution (the legacy pipeline's
    224×224 resize was a root-cause failure); this pins the policy explicitly.

    :param processor: HF processor owning an ``image_processor``.
    :param min_pixels: Minimum total pixels per image.
    :param max_pixels: Maximum total pixels per image.
    :raises AttributeError: If the processor exposes no known size interface.
    """
    ip = getattr(processor, "image_processor", processor)
    if hasattr(ip, "min_pixels"):
        ip.min_pixels = min_pixels
        ip.max_pixels = max_pixels
    elif hasattr(ip, "size") and isinstance(ip.size, dict) and "min_pixels" in ip.size:
        ip.size["min_pixels"] = min_pixels
        ip.size["max_pixels"] = max_pixels
    else:
        raise AttributeError(
            f"Cannot set resolution policy on {type(ip).__name__} — "
            f"inspect its size attributes and extend apply_resolution_policy."
        )


class _WandbTee(io.TextIOBase):
    """Stdout tee that mirrors trainer loss lines into W&B.

    The stock mlx-vlm trainer has no logging callback; it prints metrics.
    This tee passes everything through to the real stdout and logs parsed
    train/val loss lines.
    """

    def __init__(self, wandb_run, stream):
        """
        :param wandb_run: Active ``wandb`` run object.
        :param stream: Underlying stream (the real ``sys.stdout``).
        """
        self._run = wandb_run
        self._stream = stream
        self._buffer = ""

    def write(self, text: str) -> int:
        """Write through to the underlying stream and parse metric lines.

        :param text: Text chunk from the trainer.
        :returns: Number of characters written.
        """
        self._stream.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._parse(line)
        return len(text)

    def flush(self) -> None:
        """Flush the underlying stream."""
        self._stream.flush()

    def _parse(self, line: str) -> None:
        """Parse one output line and log metrics if it matches.

        :param line: A single line of trainer output.
        """
        clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = _TRAIN_LINE_RE.search(clean)
        if m:
            it, loss, lr, it_sec, tok_sec = m.groups()
            self._run.log(
                {
                    "train/loss": float(loss),
                    "train/learning_rate": float(lr),
                    "train/it_per_sec": float(it_sec),
                    "train/tokens_per_sec": float(tok_sec),
                },
                step=int(it),
            )
            return
        m = _VAL_LINE_RE.search(clean)
        if m:
            self._run.log({"val/loss": float(m.group(2))}, step=int(m.group(1)))


def run_training(cfg: ExperimentConfig) -> Path:
    """Run one training experiment.

    :param cfg: Experiment configuration.
    :returns: Path to the final adapter file.
    """
    import mlx.optimizers as optim
    from datasets import load_from_disk
    from mlx_vlm.lora import setup_model_for_training, transform_dataset_to_messages
    from mlx_vlm.trainer.datasets import VisionDataset
    from mlx_vlm.trainer.sft_trainer import TrainingArgs, train
    from mlx_vlm.trainer.utils import print_trainable_parameters
    from mlx_vlm.utils import load

    print(f"Loading model: {cfg.model_path}")
    model, processor = load(
        cfg.model_path, processor_config={"trust_remote_code": True}
    )
    assert_vision_model(model)
    apply_resolution_policy(processor, cfg.min_pixels, cfg.max_pixels)

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    config = model.config.__dict__

    print(f"Loading dataset: {cfg.dataset_dir}")
    dataset_dict = load_from_disk(cfg.dataset_dir)
    train_ds = dataset_dict[cfg.train_split]
    if cfg.task_filter:
        train_ds = train_ds.filter(lambda t: t == cfg.task_filter, input_columns="task")
    if cfg.max_samples:
        train_ds = train_ds.shuffle(seed=0).select(range(min(cfg.max_samples, len(train_ds))))
    train_ds = transform_dataset_to_messages(train_ds, model_type)

    val_dataset = None
    if cfg.val_split:
        val_ds = dataset_dict[cfg.val_split]
        if cfg.task_filter:
            val_ds = val_ds.filter(lambda t: t == cfg.task_filter, input_columns="task")
        val_ds = transform_dataset_to_messages(val_ds, model_type)
        val_dataset = VisionDataset(
            val_ds, config, processor, train_on_completions=True
        )

    train_dataset = VisionDataset(
        train_ds, config, processor, train_on_completions=True
    )

    steps_per_epoch = max(len(train_ds) // cfg.batch_size, 1)
    total_iters = steps_per_epoch * cfg.epochs
    warmup_steps = max(int(total_iters * cfg.warmup_fraction), 1)

    lr_schedule = optim.join_schedules(
        [
            optim.linear_schedule(0.0, cfg.learning_rate, warmup_steps),
            optim.cosine_decay(
                cfg.learning_rate, total_iters - warmup_steps, cfg.min_learning_rate
            ),
        ],
        [warmup_steps],
    )
    optimizer = optim.AdamW(learning_rate=lr_schedule)

    train_args_ns = argparse.Namespace(
        full_finetune=False,
        train_vision=cfg.train_vision,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
    )
    model = setup_model_for_training(
        model, train_args_ns, cfg.resume_adapter or None
    )
    print_trainable_parameters(model)

    adapter_dir = Path(cfg.adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = adapter_dir / "adapters.safetensors"

    training_args = TrainingArgs(
        batch_size=cfg.batch_size,
        iters=total_iters,
        steps_per_report=cfg.steps_per_report,
        steps_per_eval=cfg.steps_per_eval,
        steps_per_save=cfg.steps_per_save,
        val_batches=cfg.val_batches,
        max_seq_length=cfg.max_seq_length,
        adapter_file=str(adapter_file),
        grad_checkpoint=cfg.grad_checkpoint,
        learning_rate=cfg.learning_rate,
        grad_clip=cfg.grad_clip,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    )

    print(
        f"Training: {len(train_ds)} samples, {total_iters} iters "
        f"({cfg.epochs} epochs), warmup {warmup_steps}, "
        f"train_vision={cfg.train_vision}, label masking ON"
    )

    wandb_run = None
    if cfg.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.run_name or None,
            config=cfg.__dict__,
        )
        sys.stdout = _WandbTee(wandb_run, sys.stdout)

    try:
        train(
            model=model,
            optimizer=optimizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            args=training_args,
            train_on_completions=True,
        )
    finally:
        if wandb_run:
            sys.stdout = sys.stdout._stream
            wandb_run.finish()

    print(f"✅ Adapters saved → {adapter_file}")
    return adapter_file


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point.

    :param argv: Optional argument list (defaults to ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description="Train Qwen3-VL on Hebrew documents (MLX).")
    parser.add_argument("--config", type=Path, required=True, help="YAML experiment config.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Override the config's max_samples cap.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.max_samples is not None:
        cfg.max_samples = args.max_samples
    run_training(cfg)


if __name__ == "__main__":
    main()
