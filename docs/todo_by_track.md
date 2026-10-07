# Open items by track — 2026-09-14

Status key: **done** · **running** · **queued** (ready, needs a go) · **decision** (yours) · **idea** (not started).
Sources: this week's v21/v21b work, the offline log (`docs/v21b_offline_status.md`), the post-mortem, and the
2026-09-14 discussions (fusion, sample efficiency, Kraken, KTIV recall, VQA).

## 1. Transcription / CER

| item | status | notes |
|---|---|---|
| v21b-1200 promoted: LM Studio staged, KEEP_LOCAL, pipeline default | done | commit `5a116bb` |
| Resume v21b 1200 → 2000 (cosine tail) | decision | ~14 h Colab; v2.0a gained ~0.01 eval loss over its last 500 steps; optional |
| Hub merged upload `isaacmg/qwen3-vl-8b-hebrew-v21b-merged` + Space swap | decision | card credits NLI-KTIV + PGP only (no FJMS) |
| Decode policy: column detection → per-column/band decode → loop/length guards → page fallback | idea | probe: under-read two-column pages 0.612 → 0.239 CER; prompt-only sectioning does not work |
| Section rows in training ("lines N–M", wider `grounded_crop`, column crops) | idea | also the only route past the 12k-token page limit |
| Agreed-line pseudo-labels from the consensus pipeline (two-reader filter, 96–100 % precise) | idea | after the v21b re-read of the high-res docs finishes |
| Arabic-script GT (PGP footnotes, 756 docs) into eval + train | idea | the Arabic gap is unmeasured today |
| Synthetic confusable drills re-weighted by the confusion matrix (ד/ר, ב/כ, ו/י) | idea | synth3 exists |
| Word-level grounding rows (locate/read_box at parity with v2.0a) | idea | v2.2 data lever |
| v4 dataset: pages stored once + `page_id` task rows, map-style loading, `short_fragment` family | queued (gated on the scrape) | approved 09-14; **do not build until the KTIV scrape lands — target ≥ 400–500 new documents in v4** (intake ledger in track 6) |
| Eval slices: two-column, JA, Arabic script reported per checkpoint | idea | anchors stay religious-140 + PGP-131 |

## 2. Kraken / second reader

| item | status | notes |
|---|---|---|
| Fine-tune MiDRASH recognition on KTIV lines | running (compile relaunched 09-14 13:00 after an overnight `Too many open files` failure on the NAS mount; chunked local staging fixed it) | `src/finetuning/kraken/export_ktiv_lines.py` (49k clean lines / 1.43M letters from 3,556 pages; ink-centred crops, NFKD, codec-checked, split by manuscript) → `filter_manifests.py` (aspect ≤ 25, aspect/letter ≤ 1.0: kills merged-column and run-on boxes that OOM-ed batches) → `finetune_ktiv.sh` (sibling container of kraken-service:linewise, `--cpus 5 --memory 12g --shm-size 4g`, batch 16, lr 1e-4, early stop lag 5, ≤ 20 epochs; ~7 lines/s on 3 threads → ~1.5–2 h/epoch). Data + models on the NAS (`datasets/kraken_ktiv_lines/models/`). Eval: `run_religious_benchmark.py --kraken-model … --kraken-tag ktivft` (new). CC BY-NC-SA derivative — publish back |
| Retrain the segmenter on line polygons (KTIV word-box lines + agreement-filtered VLM boxes) | idea | the 4-letter-fragment problem; biggest two-reader coverage lever, zero language-prior risk |
| PGP-aligned line crops (VLM geometry, PGP text as the label) | idea | after v4; guards against forgetting the documentary hands |
| Line-level eval + two-reader agreement rate as the operational metric | idea | `kraken_seg` CERsub 0.375 vs VLM 0.165 today |
| Never train Kraken on VLM text | rule | keeps the two readers' errors uncorrelated |

## 3. VQA on Genizah documents

| item | status | notes |
|---|---|---|
| Catalog VQA from PGP metadata (type, language, date, people, places) with grounded answers (evidence-line box) | idea | thousands of curated docs; catalog-metadata generation turned into supervision |
| Entity QA (KG-curated for PGP; Gemini Pro text-only for religious KTIV + catalog; none on JA) | idea | roadmap v2.2 |
| Canonical-comparison VQA (Sefaria alignment; variant readings are the answer) | idea | roadmap v2.3; never as conditioning |
| Abstention rows + a 150–200-question held-out VQA benchmark with hallucination rate | idea | build before the first VQA run |
| v22 = v21b mixture + section rows + VQA families at ~20 %, transcription share held | idea | one run, two evals |

## 4. Multimodal research (fusion, interpretability, sample efficiency)

| item | status | notes |
|---|---|---|
| `probe_fusion_layers.py`: per-layer LoRA delta norms over ~180 checkpoints; visual attention mass per layer; attention knockout by layer band; activation patching v20a→v21b | queued | approved 09-14; Studio only (PyTorch on Metal), merged masters on the NAS |
| Attention probe on `locate_word`: attends-right-but-decodes-wrong vs never-attends | queued | the gate for any architecture change |
| Within-run adaptation curves: box + lite CER at every v21b checkpoint 100–1200 (+ v20a) | idea | 12 × ~2 h on the Studio via the eval chain |
| Pages-vs-quality curve (250/500/1000/2000/all pages, fixed steps) | idea | Colab; the sample-efficiency headline |
| Supervision-density ablation (drop word / line-index / crop rows) | idea | Colab, one run |
| v2.1b = merger LoRA + v2.1 data | idea | one-knob ablation; the honest "is the merger weak" test (prior from v20b: no) |
| Gated cross-attention at DeepStack layers | idea (research arm only) | after the above; needs its own serving path; Q-Former replacement: no |
| Weight-spectra / stencil-prior-dissolution figures for the paper | idea | `article/handoff_weight_probing.md` is the template |

## 5. Consensus pipeline / site

| item | status | notes |
|---|---|---|
| v21b re-read of the 6,065 high-res docs → then the JTS-heavy remainder (chained) | running | `run_documentary_v21b.log`; ~5 days total |
| Agreement rate v2.0a vs v21b on the first few hundred pages | queued | promised; v2.0a slice: 6 % of lines, 11 % of pages |
| Site session: `--apply` v21b reads (loader keyed by `vlm_model`), rule v2 unchanged | decision (site) | v2.0a file stays valid |
| JTS pages arrive at 1,440 px — two-reader impossible until higher-res JTS images are indexed | idea (site/index) | 1,261 of the remaining jobs are JTS |
| Unload v2.0a from LM Studio once nothing calls it | decision | files stay (flagship rule) |

## 6. Data / KTIV recall

| item | status | notes |
|---|---|---|
| Partial-page re-probe: 204 manuscripts / 297 images with no captured annotation page (the booklets) | queued | `ktiv-scraper/ktiv_queue_partial_pages.csv`; **annotations only — images already on disk** |
| 457 open never-visited index items | queued | `ktiv-scraper/ktiv_queue_never_visited_open.csv` |
| 100-manuscript false-negative re-probe | queued | `ktiv-scraper/ktiv_queue_reprobe_sample100.csv` |
| Probe-only extension mode (enumerate the 80k listing, annotation probe per manuscript, scrape positives only) | idea | ~1 day of unattended Chrome; exact transcribed set |
| Capture–recapture estimate with ~8 new probe words in list-only mode | idea | bounds what is still unfound |
| KTIV transcriptions outside the Genizah collection | idea | same extension; Kraken/adaptation data |
| Arabic KTIV scrape priority queue (785 rows) | queued | columns fixed to the scraper's format |
| Builder: rescue the 52 transcribed pages whose boxes fall outside the image frame (scale check between annotation frame and downloaded derivative) | idea | median 940 letters each — real pages |
| Re-download images for the 3 transcribed manuscripts with no zip (34 pages, one 16-page booklet) | queued | needs KTIV image serving back |
| `short_fragment` family: transcribed pages under the 150-letter gate | approved 09-14 | 251 pages / 151 mss after id-level benchmark exclusion (316 before); all have in-frame images; 229 have ≥ 4 lines. Gate: letters ≥ 40, damage share ≤ 0.30 (gap tokens + words with dots/brackets over all words), image in frame, not shingle-contaminated → 169 pages / 109 mss (73 mss new to the dataset), ~16k letters. Heavily damaged strips (damage > 0.30, e.g. 35 pages) stay out per Isaac. Own family + own eval slice; 4 pages carry an `@`+combining-mark sigla artifact — check normalisation in the builder |
| **Scrape haul 09-13/14 night (measured 09-14 15:00):** 511 new bundle files / 507 mss touched, 1,264 new image zips (KTIV images serve again); 185 brand-new transcribed manuscripts (184 with images), 2 DOM→API upgrades, 3 mss gained pages (990001398720205171: 13 → 42 pages) → **+575 text pages, +490k Hebrew letters** before gating; 317 re-scrapes with unchanged coverage; the never-visited/reprobe queue CSVs carry shelf-mark ids while bundles carry PNX ids, so per-queue attribution needs the scraper's visited log | done | 
| v4 intake ledger (toward ≥ 400–500 new docs): short pages 169 (109 mss, 73 new) · out-of-frame rescue 52 pages · no-image re-download 34 pages (3 mss) · partial-page re-probe up to 297 pages (204 existing mss — pages, not new docs) · never-visited open 457 mss (yield unknown; the swing item) · false-negative re-probe 100 (low yield expected). New-manuscript count depends mostly on the never-visited yield; page count clears 500 without it | running (scrape by Isaac) | started 09-14 |
| KTIV page audit result: 4,136 transcribed pages; 3,310 in v3; the rest = 316 short, 312 PGP-shingle hold-out, ~125 religious-140 hold-out, 52 out-of-frame, 34 no image, 2 gaps — nothing dropped by accident | done | 2026-09-14 |
| Full 80k image scrape | long-term | months; KTIV image serving currently erroring |

## 7. Infra / ops

| item | status | notes |
|---|---|---|
| Disk batch A re-plan: four cold models (73 GB) — run with the pipeline paused, assistant healthy | decision | site models excluded; `offload_to_nas.sh` verifies before deleting |
| Batch C (your experiments, 22 GB) / batch D (Docker prune, 50 GB) | decision | |
| `qwen3.8-27b` stays unloaded; keep ≥ 25 GB RAM headroom for the site's 35B-A3B | rule | saved |
| Confirm the site assistant recovered | decision | one retry in the chat |
| Push the branch (13+ unpushed commits) | decision | |
| Rented GPU VM with SSH for long runs (my direct access; no disk/streaming constraints) | decision | after v4 |
| Article: v21b rows, v21 post-mortem methods note, fusion-probe figures | idea | |
