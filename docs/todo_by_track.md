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
| v4 dataset: pages stored once + `page_id` task rows, map-style loading | decision | approved in principle; removes streaming and makes new families cheap |
| Eval slices: two-column, JA, Arabic script reported per checkpoint | idea | anchors stay religious-140 + PGP-131 |

## 2. Kraken / second reader

| item | status | notes |
|---|---|---|
| Fine-tune MiDRASH recognition on KTIV lines (`ketos` in the container, NFKD, held-out split) | queued | ~10k human-transcribed lines with geometry; CC BY-NC-SA derivative — publish back |
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
