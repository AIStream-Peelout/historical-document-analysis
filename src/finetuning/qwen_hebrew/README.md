# Qwen3-VL Hebrew fine-tuning pipeline

Fine-tunes Qwen3-VL-8B to actually read Hebrew script (the base model
collapses on the Talmud benchmark — garbled repetition, CER ≫ Gemini/Claude).
Replaces the failed Gemma pipeline archived in `../legacy_gemma/` (see its
README for the post-mortem that motivated this pipeline's guardrails).

## Environments

- **Main venv** (`.venv`): data prep, tests, benchmark. mlx-vlm must NEVER be
  installed here (it force-upgrades transformers/huggingface-hub past the pins
  kraken/doctr/colpali need).
- **Training venv** (`.venv-mlx`): `python3 -m venv .venv-mlx &&
  .venv-mlx/bin/pip install -r requirements-mlx.txt`. Runs mlx_train,
  quick_eval, image_dependence_probe, export_lmstudio.
- **Colab A100**: `colab/qwen3_vl_hebrew_unsloth.ipynb` for heavy runs; it
  consumes the same dataset from a private HF hub repo.

## Workflow

```bash
# 1. Crop the full corpus (~5,373 pages → ~16k section crops)
.venv/bin/python -m src.finetuning.qwen_hebrew.crop_full_corpus            # scan + crop
#    review crop_outliers.json for pages needing boundary overrides

# 2. Build the dataset (holds out the 66 benchmark pages; excludes
#    rashi/tosafot pairs on non-standard-layout pages, e.g. Nedarim)
.venv/bin/python -m src.finetuning.qwen_hebrew.build_dataset [--push_to_hub user/repo]
.venv/bin/python -m pytest tests/test_qwen_hebrew_dataset.py    # artifact checks

# 3. Baselines BEFORE training (record the collapse + probe margin)
.venv-mlx/bin/python -m src.finetuning.qwen_hebrew.quick_eval
.venv-mlx/bin/python -m src.finetuning.qwen_hebrew.image_dependence_probe

# 4. Overfit-8 sanity gate — must reach ~0 loss or STOP and debug
.venv-mlx/bin/python -m src.finetuning.qwen_hebrew.mlx_train --config src/finetuning/qwen_hebrew/configs/smoke.yaml

# 5. Real training (joint vision+language; 4k-sample scout run first)
.venv-mlx/bin/python -m src.finetuning.qwen_hebrew.mlx_train --config src/finetuning/qwen_hebrew/configs/stage1_joint.yaml --max_samples 4000
#    between checkpoints:
.venv-mlx/bin/python -m src.finetuning.qwen_hebrew.quick_eval --adapter models/adapters/qwen3vl_hebrew_stage1
#    after each stage:
.venv-mlx/bin/python -m src.finetuning.qwen_hebrew.image_dependence_probe --adapter models/adapters/qwen3vl_hebrew_stage1

# 6. Export → LM Studio → benchmark before/after
.venv-mlx/bin/python -m src.finetuning.qwen_hebrew.export_lmstudio \
    --adapter models/adapters/qwen3vl_hebrew_stage1 --output models/qwen3-vl-8b-heb-stage1
lms import models/qwen3-vl-8b-heb-stage1
.venv/bin/python -m src.datasets.evaluations.talmud_evaluation \
    --lm_studio_models qwen/qwen3-vl-8b,<lm-studio-id>
# pure-reading (crop) track:
.venv/bin/python -m src.datasets.evaluations.talmud_crop_evaluation \
    --lm_studio_models qwen/qwen3-vl-8b,<lm-studio-id> --wandb
```

## Guardrails (each maps to a root cause of the legacy failure)

| Guardrail | Catches |
|---|---|
| `image_dependence_probe` (CER margin > 0.3 on shuffled images) | blind training / memorization |
| Notebook cell 5 asserts `pixel_values` + label masking | collator bugs |
| `mlx_train` asserts a vision tower exists | text-only base model |
| explicit `min_pixels`/`max_pixels` in configs | destructive downscaling |
| overfit-8 gate (`configs/smoke.yaml`) | any silent data-plumbing break |
| `build_dataset` hard-asserts zero benchmark stems | eval leakage |

## Roadmap

Phase 2 (newspapers, `gs://backupmacf/newspapers/`) and Phase 3
(Genizah/HebrewBooks self-training) extend the same dataset schema via
`label_source` — see the plan in the repo discussion / EACL notes.
