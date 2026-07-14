"""Fine-tuning pipeline for Qwen3-VL on Hebrew historical documents.

Modules
-------
- ``prompts``                 — training prompts (crop transcription + page extraction)
- ``crop_full_corpus``        — crop the full Talmud corpus into section images
- ``build_dataset``           — build the HF training dataset (images + messages)
- ``mlx_train``               — local MLX LoRA training wrapper
- ``quick_eval``              — fast CER check on held-out crops
- ``image_dependence_probe``  — shuffle-image control (guards against blind training)
- ``export_lmstudio``         — fuse adapters and export for LM Studio
"""
