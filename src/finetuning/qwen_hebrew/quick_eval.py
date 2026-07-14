"""Fast CER check on held-out section crops via local MLX generation.

The between-checkpoint iteration loop: minutes, not hours. Scores N fixed
validation crops with the same prompts used in training, using the shared
benchmark metrics (:mod:`src.datasets.evaluations.metrics` — single source
of truth for CER).

Run inside the training venv:

  .venv-mlx/bin/python -m src.finetuning.qwen_hebrew.quick_eval \\
      --model mlx-community/Qwen3-VL-8B-Instruct-bf16 \\
      [--adapter models/adapters/qwen3vl_hebrew_stage1] [--num_samples 30]
"""

import argparse
import random
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from src.datasets.evaluations.metrics import (
    cer_pair,
    char_count_ratio,
    flag_failure_modes,
    normalize_whitespace,
)

_REPO = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_DIR = _REPO / "src/datasets/processed/qwen_hebrew_talmud"
SAMPLE_SEED = 20260714
MAX_GEN_TOKENS = 4500


def load_model(model_path: str, adapter_path: Optional[str] = None):
    """Load an mlx-vlm model, optionally with trained adapters.

    :param model_path: HF hub id or local path of the (base or fused) model.
    :param adapter_path: Optional adapter directory from training.
    :returns: Tuple of (model, processor, config dict).
    """
    from mlx_vlm.utils import load

    model, processor = load(model_path, adapter_path=adapter_path or None)
    return model, processor, model.config.__dict__


def transcribe(model, processor, config: Dict, image_path: str, prompt: str) -> str:
    """Generate a transcription for one image.

    :param model: Loaded mlx-vlm model.
    :param processor: Matching processor.
    :param config: Model config dict.
    :param image_path: Path to the image file.
    :param prompt: Plain-text instruction.
    :returns: Generated text.
    """
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    formatted = apply_chat_template(processor, config, prompt, num_images=1)
    result = generate(
        model,
        processor,
        formatted,
        image=[image_path],
        max_tokens=MAX_GEN_TOKENS,
        temperature=0.0,
        verbose=False,
    )
    return result.text


def resolve_image_path(row_image: Dict, scratch_dir: Path) -> str:
    """Return a filesystem path for a dataset image (decode=False encoding).

    :param row_image: ``{"path": ..., "bytes": ...}`` dict from the Image feature.
    :param scratch_dir: Directory for materialising byte-only images.
    :returns: Path to an image file on disk.
    """
    path = row_image.get("path")
    if path and Path(path).exists():
        return path
    out = scratch_dir / f"img_{abs(hash(row_image.get('path') or id(row_image)))}.png"
    out.write_bytes(row_image["bytes"])
    return str(out)


def select_val_samples(dataset_dir: Path, num_samples: int, seed: int) -> List[Dict]:
    """Pick a fixed, seeded subset of validation crop samples.

    :param dataset_dir: ``save_to_disk`` dataset directory.
    :param num_samples: Number of samples to evaluate.
    :param seed: RNG seed (fixed default keeps runs comparable).
    :returns: List of dataset rows with undecoded images.
    """
    from datasets import Image as HFImage
    from datasets import load_from_disk

    val = load_from_disk(str(dataset_dir))["val"]
    val = val.filter(lambda t: t == "crop_transcribe", input_columns="task")
    val = val.cast_column("image", HFImage(decode=False))
    indices = list(range(len(val)))
    random.Random(seed).shuffle(indices)
    return [val[i] for i in indices[:num_samples]]


def evaluate_samples(
    model, processor, config: Dict, samples: List[Dict], shift_images: int = 0
) -> List[Dict]:
    """Transcribe and score a list of samples.

    :param model: Loaded model.
    :param processor: Matching processor.
    :param config: Model config dict.
    :param samples: Dataset rows (undecoded images).
    :param shift_images: Cyclic image shift — 0 pairs each sample with its own
        image; the probe uses 1 to pair every sample with the WRONG image.
    :returns: Per-sample result dicts (section, cer_strict, cer_lenient,
        char_ratio, flags, hypothesis).
    """
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        n = len(samples)
        for i, row in enumerate(samples):
            image_row = samples[(i + shift_images) % n]["image"]
            image_path = resolve_image_path(image_row, scratch)
            hyp = transcribe(model, processor, config, image_path, row["question"])
            hyp_n = normalize_whitespace(hyp)
            ref_n = normalize_whitespace(row["answer"])
            strict, lenient = cer_pair(hyp_n, ref_n)
            results.append(
                {
                    "stem": row["stem"],
                    "section": row["section"],
                    "cer_strict": strict,
                    "cer_lenient": lenient,
                    "char_ratio": char_count_ratio(hyp_n, ref_n),
                    "flags": flag_failure_modes(hyp_n, ref_n),
                    "hypothesis": hyp,
                }
            )
            print(
                f"  [{i + 1}/{n}] {row['stem']}/{row['section']}: "
                f"CER {strict:.3f} (lenient {lenient:.3f}) "
                f"ratio {results[-1]['char_ratio']:.2f} {results[-1]['flags'] or ''}"
            )
    return results


def summarize(results: List[Dict]) -> Dict:
    """Aggregate per-sample results into summary metrics.

    :param results: Output of :func:`evaluate_samples`.
    :returns: Summary dict (overall + per-section mean CER, flag counts).
    """
    by_section: Dict[str, List[Dict]] = {}
    for r in results:
        by_section.setdefault(r["section"], []).append(r)

    flags = Counter(f for r in results for f in r["flags"])
    return {
        "num_samples": len(results),
        "cer_strict_mean": statistics.mean(r["cer_strict"] for r in results),
        "cer_lenient_mean": statistics.mean(r["cer_lenient"] for r in results),
        "char_ratio_mean": statistics.mean(r["char_ratio"] for r in results),
        "per_section": {
            s: {
                "n": len(rs),
                "cer_strict": statistics.mean(r["cer_strict"] for r in rs),
                "cer_lenient": statistics.mean(r["cer_lenient"] for r in rs),
            }
            for s, rs in sorted(by_section.items())
        },
        "failure_flags": dict(flags),
    }


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point.

    :param argv: Optional argument list (defaults to ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description="Quick CER eval on held-out crops.")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3-VL-8B-Instruct-bf16")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--wandb", action="store_true", help="Log summary to W&B.")
    args = parser.parse_args(argv)

    print(f"Model: {args.model}" + (f" + adapter {args.adapter}" if args.adapter else ""))
    model, processor, config = load_model(args.model, args.adapter)
    samples = select_val_samples(args.dataset_dir, args.num_samples, args.seed)
    results = evaluate_samples(model, processor, config, samples)
    summary = summarize(results)

    print("\n── Quick eval summary ──")
    print(f"  samples:        {summary['num_samples']}")
    print(f"  CER strict:     {summary['cer_strict_mean']:.4f}")
    print(f"  CER lenient:    {summary['cer_lenient_mean']:.4f}")
    print(f"  char ratio:     {summary['char_ratio_mean']:.3f}")
    for section, m in summary["per_section"].items():
        print(f"  {section:8s} (n={m['n']}): CER {m['cer_strict']:.4f} / lenient {m['cer_lenient']:.4f}")
    if summary["failure_flags"]:
        print(f"  failure flags:  {summary['failure_flags']}")

    if args.wandb:
        import wandb

        run = wandb.init(project="qwen-hebrew-finetune", job_type="quick_eval",
                         config={"model": args.model, "adapter": args.adapter})
        run.log({f"quick_eval/{k}": v for k, v in summary.items()
                 if isinstance(v, (int, float))})
        run.finish()


if __name__ == "__main__":
    main()
