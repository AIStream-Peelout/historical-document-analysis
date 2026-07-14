"""Shuffle-image control: is the model actually reading the image?

The legacy Gemma pipeline trained without images reaching the model, producing
a blind Hebrew LM whose loss kept "improving". This probe makes that failure
detectable in minutes at generation time, with no trainer internals:

- Transcribe N validation crops with the CORRECT image.
- Transcribe the same N prompts with images cyclically shifted by one
  (every sample sees the WRONG image).
- A model that reads the image must get much worse on shifted images.

Pass criterion: ``CER(shifted) - CER(correct) > 0.3``. Run on the base model
first (records the pre-training margin), and after every training stage.

  .venv-mlx/bin/python -m src.finetuning.qwen_hebrew.image_dependence_probe \\
      --model mlx-community/Qwen3-VL-8B-Instruct-bf16 [--adapter <dir>]
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from src.finetuning.qwen_hebrew.quick_eval import (
    DEFAULT_DATASET_DIR,
    SAMPLE_SEED,
    evaluate_samples,
    load_model,
    select_val_samples,
    summarize,
)

PASS_MARGIN = 0.3


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point.

    :param argv: Optional argument list (defaults to ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description="Shuffle-image dependence probe.")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3-VL-8B-Instruct-bf16")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    args = parser.parse_args(argv)

    print(f"Model: {args.model}" + (f" + adapter {args.adapter}" if args.adapter else ""))
    model, processor, config = load_model(args.model, args.adapter)
    samples = select_val_samples(args.dataset_dir, args.num_samples, args.seed)

    print(f"\n[1/2] Correct images ({len(samples)} samples)")
    correct = summarize(evaluate_samples(model, processor, config, samples))
    print(f"\n[2/2] Shifted images (every sample gets the wrong image)")
    shifted = summarize(
        evaluate_samples(model, processor, config, samples, shift_images=1)
    )

    delta = shifted["cer_lenient_mean"] - correct["cer_lenient_mean"]
    passed = delta > PASS_MARGIN

    print("\n── Image-dependence probe ──")
    print(f"  CER (correct images): {correct['cer_lenient_mean']:.4f}")
    print(f"  CER (shifted images): {shifted['cer_lenient_mean']:.4f}")
    print(f"  margin:               {delta:.4f}  (pass threshold {PASS_MARGIN})")
    if passed:
        print("  ✅ PASS — output depends on the image.")
    else:
        print(
            "  ❌ FAIL — the model scores about the same with the WRONG image.\n"
            "     It is ignoring the image (blind training, memorization, or\n"
            "     canonical recitation). Do not proceed with this checkpoint."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
