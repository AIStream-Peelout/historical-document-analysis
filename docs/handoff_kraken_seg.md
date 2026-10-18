# Handoff — build `kraken_seg` (reading-order baseline) for genizah_religious_v1

Paste the block below into a fresh Claude Code session in this repo. Uncommitted
working note; delete when done.

---

## Objective
Add a **fair Kraken baseline** (`kraken_seg`) to the Genizah religious-text
benchmark and re-score it against the flagship VLM. Right now the benchmark only
ran `kraken_raw` (flat line output in segmentation order), which on multi-column
Talmud MS pages is NOT reading order — so Kraken's F1 collapsed and my scorer
mislabeled it "hallucinated." That label is wrong and this task fixes the
comparison.

## The finding that motivates this (verified, do not re-litigate)
On `genizah_religious_v1` (140 Talmud-MS pages), an order-independent diagnostic
showed Kraken's **bag-of-chars overlap with GT = 0.94 single-col / 0.98 multi-col,
with 0 "misread" cases**. Kraken *recognizes the glyphs excellently*; its low F1
(0.393) and low 5-gram precision (0.16/0.39) are **pure reading-order scramble**
(2-gram stays high at 0.74/0.91, 5-gram collapses = short runs right, long runs
cross line/column boundaries wrongly). 66/130 pages are clean order-scramble.
"Hallucinated 36%" was a scorer artifact: `classify()` (VLM-centric) buckets any
ngramP<0.10 output as "hallucinated" regardless of model. Kraken has no language
prior and cannot hallucinate. So v19a's Talmud win is **document-structure /
reading-order**, NOT better glyph perception. (See docs/talmud_ocr_segmentation.md:
the reading-order step is load-bearing on Talmud, harmful on single-col
documentary.)

## What to build — `kraken_seg`

**Path A (PRIMARY — deterministic geometry reorder, no LLM, no prod contention):**
1. The Kraken microservice (`localhost:8002`, container owned by THIS repo) already
   returns per-line geometry. See `src/services/kraken_microservice/main.py`:
   `_segment_and_recognize_lines()` builds `LineInfo {text, bbox:[x0,y0,x1,y1]}`
   in ORIGINAL-image pixel coords, exposed as `TranscriptionResponse.lines`.
   BUT the flat `/transcribe` path may return only `text`. Check the endpoint;
   if `lines` isn't in the JSON response, expose it (add a field or a
   `?geometry=1` param) — do NOT change existing `text` behavior (other callers
   depend on it).
2. In the client (`src/models/ocr/kraken_transcriber.py`), add
   `transcribe_with_kraken_lines()` that returns the `lines` list (text+bbox).
3. Reorder into Hebrew reading order with the SAME column-clustering logic already
   written for GT word-boxes: `src/finetuning/qwen_hebrew/ktiv_layout.py` →
   `reconstruct_page()` clusters boxes into columns and orders columns
   right→left, lines top→bottom. Adapt it (or factor out a shared
   `columns_from_boxes(boxes)`), feeding Kraken LINE bboxes instead of word boxes.
   One line = one unit; join within a column top-to-bottom, columns right-to-left.
   This makes `kraken_seg` GT-order-consistent by construction (GT uses the same
   reconstructor), so the comparison isolates OCR quality from order.

**Path B (fallback / second baseline — LLM reflow):**
Only if Path A's geometry is unusable. Feed Kraken flat text to a reading-order
LLM. Machinery exists in `src/datasets/evaluations/talmud_evaluation.py`
(`_SEGMENTATION_PROMPTS`, `_call_ocr_segmentation()`) BUT it is Vilna-printed
specific (3 sections: gemara/rashi/tosafot) and will NOT fit running-text /
2-column MSS pages — you'd need a new simple "reflow into reading order, copy
exactly, do not correct" prompt. Default seg model is Gemini Flash; LM Studio
text model via `--segmentation_base_url`. If using LM Studio: production serves
on :1234 with a one-request-at-a-time asyncio lock — do NOT eject pinned prod
models (qwen3-4b router, qwen3.6-35b synthesis). Prefer Path A to avoid this.

## Wire into the scorer
`src/datasets/evaluations/helper_eval_scripts/run_religious_benchmark.py`:
- Add `kraken_seg` as a third saved output per page (alongside
  `v19a_step1300` and `kraken_raw`), resumable (skip if file exists).
- **Also finish the in-progress scorer honesty patch** (was mid-edit): add an
  ORDER-INDEPENDENT diagnostic to each score row + the report — `bag_overlap`
  (multiset char overlap vs GT) and an `order_scramble` flag
  (`bag_overlap>=0.6 and ngram_precision<0.2`). Replace the misleading `halluc`
  aggregate column with `readOK` (median bag_overlap) + `orderScr` (% scrambled)
  so a line-OCR model is never again labeled "hallucinated." Do this ONLY in
  run_religious_benchmark.py — do NOT touch score_genizah_offline.classify()
  (the PGP documentary benchmark numbers must stay frozen).
- Report ALL / talmud / multi-col / single-col for all three models.

## Expected result
`kraken_seg` should rescue most of the 66 order-scramble pages: 5-gram precision
and F1 jump toward the bag-overlap ceiling (~0.94–0.98 char level), especially
multi-col. v19a likely still wins (end-to-end), but the honest framing becomes
"VLM vs OCR + reading-order," not "VLM reads better." Report the before/after
per slice.

## Key paths / constants
- Benchmark: `src/datasets/raw_data/cairo_genizah/evaluations/genizah_religious_v1/`
  (`genizah_religious_v1.json` catalog with per-page `gt`,`n_columns`,`is_talmud`,
  `image`; `raw_outputs/<doc_id>/*.txt`). 140 pages, all decontaminated.
- Kraken model: `src/datasets/raw_data/cairo_genizah/custom_model_weights/MiDRASH_Gen_01.mlmodel`
- VLM served: `qwen3-vl-8b-heb-v19a-step1300` (LM Studio :1234).
- Reorder logic to reuse: `src/finetuning/qwen_hebrew/ktiv_layout.py::reconstruct_page`.

## HARD CONSTRAINTS (shared PRODUCTION Mac Studio — read docs/shared_studio_runtime.md)
- api.cairogenizah.ai runs here. NEVER stop/restart/prune other Docker containers,
  NEVER eject pinned LM Studio prod models, NEVER kill cloudflared. This repo owns
  ONLY the Kraken container (:8002). Never bind web-app ports
  (8000/8001/3000/7681/7475/9200/5601/8010).
- Before any long job: `df -h`, `~/.lmstudio/bin/lms ps`, `docker ps`. RAM ceiling
  is swap (~17–19GB); ask before taking anything prod offline. Big outputs → NAS
  `/Volumes/home/studio_offload/`.
- Git: working branch is `ktiv_merging_logic_images`. NEVER commit to `kg_work`
  (reserved for a separate agent).
- The benchmark is already decontaminated (disjoint from the 906 v1.9a-trained
  KTIV sys_nums + genizah_clean_v2 + PGP benchmark). Don't re-open that; just
  don't ADD new pages without re-running DecontamGate.
- One MLX process at a time; mlx-vlm never in the main .venv (not needed here).

## First steps
1. `docker ps` to confirm the Kraken container is up on :8002 (this repo's, don't
   touch others); `curl localhost:8002/health` or hit `/transcribe` on one image.
2. Inspect `src/services/kraken_microservice/main.py` `/transcribe` response — is
   `lines` (geometry) already in the JSON? If yes, Path A is trivial.
3. Prototype the geometry reorder on ONE known multi-col page (its raw_outputs
   already exist), confirm 5-gram precision jumps vs kraken_raw, then run all 140.
