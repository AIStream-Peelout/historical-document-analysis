---
title: Genizah Reader
emoji: 📜
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Read handwritten Hebrew manuscripts with boxes
models:
  - isaacmg/qwen3-vl-8b-hebrew-v20a-merged
---

# Genizah Reader

Upload a page of a handwritten Hebrew-script manuscript and watch a fine-tuned Qwen3-VL-8B
transcribe it — as plain text, line by line with bounding boxes drawn as they arrive, or by
locating a phrase you type. This is the live playground of the
[Cairo Genizah AI](https://cairogenizah.ai) project; the site's own reads are computed offline.

**Experimental research model — do not cite.** It invents plausible text on damaged regions,
sometimes repeats itself, and its boxes are approximate. Not trained on Arabic, Latin or Greek
script. Uploads are not stored.

Model: [isaacmg/qwen3-vl-8b-hebrew-v20a-merged](https://huggingface.co/isaacmg/qwen3-vl-8b-hebrew-v20a-merged).
Training-data credit: the National Library of Israel's KTIV project and the Princeton Geniza
Project, with images from the holding institutions. Runs on Hugging Face ZeroGPU; GPU time is
metered per visitor (sign in for a larger daily quota).
