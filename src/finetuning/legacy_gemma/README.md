# Legacy Gemma 3 fine-tuning pipeline (superseded — do not use)

Archived 2026-07 when the Qwen3-VL pipeline (`src/finetuning/qwen_hebrew/`)
replaced it. Kept for reference because the *way* it failed informed the new
pipeline's guardrails.

## Why it never worked

Training loss plateaued while hallucination "improved" because the model was
being trained as a **blind, unconditional Hebrew LM** and then evaluated
multimodally:

1. **Images never reached the model.** `gemma_three_collator.py` loaded
   images into `all_images` and then discarded them — `apply_chat_template`
   was called on text-only messages, so `pixel_values` was always `None`.
2. **Labels were never masked.** `labels = input_ids.clone()` with no `-100`
   masking — loss was computed on prompt and padding tokens.
3. **A "diagnostic fixer" silently clobbered the training strategy.**
   `gemma_trainer_diagnostic.fix_parameter_connection` re-froze the whole
   model and unfroze only embeddings, `lm_head`, and 3 text layers.
4. One config trained `gemma-3-1b-it`, which has **no vision tower**, in a
   "vision" configuration; images were resized to 224×224 (dense Hebrew is
   unreadable at that resolution).

The new pipeline's counters: mandatory collator assertions (pixel_values +
label masking), an image-dependence probe, a vision-tower assertion, an
explicit min/max-pixels resolution policy, and an overfit-8 sanity gate.
See `src/finetuning/qwen_hebrew/`.

Note: these files also import a nonexistent `src.finetuning_scripts` package
(the directory was renamed to `src/finetuning` at some point) — they do not
run as-is.
