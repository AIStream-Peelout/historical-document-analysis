# Handoff: upgrade the Kraken service to kraken 5/6, then train a Genizah segmenter

Written 2026-09-23 for a fresh session. Everything below was measured in the v22 session; nothing here
is speculation unless marked. Repo: `/Users/isaac/Documents/GitHub/historical-document-analysis`.

## Why

Our Kraken reads are the weak reader in the two-reader consensus pipeline and the reason line
agreement sits at ~10% on documentary pages. On 2026-09-23 we scored MiDRASH's published automatic
transcriptions of the NLI Geniza images against our religious benchmark (140 KTIV pages, 138 matched;
same visible-ink CER scorer as the harness):

| Reader | CER median | aligned F1 |
|---|---|---|
| MiDRASH Zenodo transcriptions (kraken 5.3.1.dev56, classifier-chosen Genizah segmenters + recognition) | **0.240** | 0.834 |
| our `kraken_raw` (service, default `blla` segmenter, Gen_01) | 0.677 | 0.424 |
| our `kraken_seg` (same, lines re-ordered by geometry) | 0.635 | 0.500 |
| our `kraken_seg_ktivft` (Gen_01 fine-tuned on 45k KTIV lines) | 0.592 | 0.541 |
| VLM v21b-1200 (for reference) | 0.167 | 0.869 |

MiDRASH beats our Kraken on 129 of 138 pages, by a median of 0.4 CER. The recognition model is the
same lineage (their "MiDRASH Geniza 01 HTR" v1 file is our `MiDRASH_Gen_01.mlmodel`, 23 MB). The gap is
segmentation, reading order and the kraken version, not recognition — our own recognition fine-tune
moved 0.64 → 0.59 only. Details and per-page numbers: memory note `project_zenodo_midrash_transcriptions`,
scratch `zenodo_vs_kraken_religious.json` (per-page CERs for all five readers).

## What exists

- **Our service:** container `kraken-linewise` on **:8002** (the only port this repo owns), image
  `kraken-service:linewise`, kraken **4.3.13**, code at `/app/main.py` inside the container, models
  mounted from `src/datasets/raw_data/cairo_genizah/custom_model_weights/` (`MiDRASH_Gen_01.mlmodel`,
  `ktiv_ft_r1_ep10.mlmodel`). Segmentation call: `blla.segment(im, text_direction="horizontal-rl")` with
  the built-in default model, optional `binarization.nlbin`. Client: `src/models/ocr/kraken_transcriber.py`
  (`transcribe_with_kraken`, `transcribe_with_kraken_lines`). Dockerfile/compose: search the repo for
  `kraken-service:linewise` (`docs/shared_studio_runtime.md` documents the port rules).
- **Religious benchmark:** `src/datasets/raw_data/cairo_genizah/evaluations/genizah_religious_v1/`
  (`genizah_religious_v1.json`, 140 docs with `sys_num`, `fl`, `gt`, `n_columns`, `is_talmud`; images
  under `images/`; cached outputs per doc in `raw_outputs/<doc_id>/` incl. `kraken_raw.txt`,
  `kraken_seg.txt`, `kraken_lines.json`; scores in `religious_scores_long.csv`). Runner:
  `src/datasets/evaluations/helper_eval_scripts/run_religious_benchmark.py` (`--kraken-tag` writes
  outputs under a new key so nothing is overwritten). Scorer: `cer_pair(normalize_ink_hypothesis(hyp),
  genizah_visible_ink_gt(gt))` from `src/datasets/evaluations/metrics.py` and `aligned_prf` from
  `helper_eval_scripts/score_genizah_offline.py` (see `eval_harness/lite_eval.py` lines 28–31).
- **PGP benchmark:** `evaluations/genizah_test_v1/genizah_test_v1_verified.json` (131 docs), cached
  Kraken outputs in `src/datasets/evaluations/transcription_raw_outputs/<doc_id>/kraken_raw.txt`;
  benchmark-metric CER median 0.358 for `kraken_raw`, 0.442 for `kraken_seg`. Score both benchmarks.
- **Public models (Zenodo, kraken model community, Apache 2.0, Benjamin Kiessling):**
  - new general `blla` segmentation model, 2026-09-21, record 22879549 (`blla.mlmodel`, 5 MB; retrained
    on ~11,200 pages from six datasets incl. Muharaf Arabic handwriting) — the new kraken default;
  - **Orli**, 2026-06-05, record 20558179 (`orli_base.safetensors`, 594 MB): baseline detection +
    reading order in one model, high-res variant 1920×1440 (paper arXiv 2606.04166). Needs a kraken
    version with Orli support (check kraken ≥6 docs at kraken.re).
  - MiDRASH's own Genizah segmenters are **not published** (checked Zenodo, the NetLay paper, the HTR
    model records). Only the recognition model is (record 17931017 v1, CC-BY-NC-SA).
- **MiDRASH transcriptions on the NAS:** `/Volumes/home/studio_offload/datasets/midrash_zenodo/Transcriptions.txt`
  (1.37 GB, 948,549 page records, header `==> <sys_num>_IE<ie>_P<page>_FL<fl> <==`). Their FL ids are
  the master file ids = our KTIV zip derivative FL − 3 (118/138 pages; a few −5/−7): match by sys_num +
  page position (`NNNN_` prefix of the zip member). Text only, no geometry.
- **Segmentation training data we already hold:** every API-shape KTIV bundle
  (`raw_data/cairo_genizah/ktiv/ktiv_PNX_MANUSCRIPTS<sys>-1_transcription*.json`) carries one W3C
  annotation per word with an `SvgSelector` rectangle in IIIF pixel coordinates, `fl` → zip member
  `NNNN_<FL>.jpg`. `src/finetuning/qwen_hebrew/ktiv_layout.py` already rebuilds lines from word boxes
  (column split by x-gap, y-clustering at 0.5×median height, RTL order, gap merge); the v4 build
  (`/Volumes/home/studio_offload/datasets/genizah_ktiv_v4`, 4,230 pages, 1,750 mss; images-once export
  next to it with 18,383 JPEGs) passed the geometry gates. `build_ktiv_dataset.py` has the per-page
  line geometry (`line_index_rows`, `norm_box`) and the decontamination lists
  (`raw_data/cairo_genizah/decontam/`, religious-benchmark exclusions). Line polygons/baselines for
  kraken `segtrain` (ALTO or PageXML) can be derived from those line boxes: baseline = bottom edge of
  the word-box union (or its vertical centre band), polygon = the union box; regions = the column
  clusters. Do not use benchmark manuscripts (`genizah_religious_v1.json` sys_nums, `benchmark_ids.json`).
- **Recognition fine-tune tooling** (reusable for the segmenter run's data plumbing):
  `src/finetuning/kraken/` (`export_ktiv_lines.py`, `filter_manifests.py`, `finetune_ktiv.sh`: ketos in
  Docker with `--shm-size 4g`, log caps; memory note `project_kraken_ktiv_finetune`).

## Constraints (read `docs/shared_studio_runtime.md` first)

- Shared production Mac Studio: never stop/restart other containers, LM Studio models or cloudflared.
- **Do not restart or replace `kraken-linewise` while the consensus pipeline is running** — it calls
  :8002 continuously (pid file `logs/two_reader_v21b_v22b_0923.pid`, expected to finish the edition
  pages ~2026-09-25/26, then the rest of its queue for several more days). Coordinate the cut-over: the
  pipeline is resumable (`--ids`, `--out`, done keys), so a short stop between pages is fine if agreed.
- First test in a scratch venv or a second image on a **different, unused port** (not 8000/8001/3000/
  7681/7475/9200/5601/8010/1234). Big outputs to the NAS. Check `df -h` (local ~35 GB free).
- No hub pushes, no GCS uploads (user-run). Kraken/ketos jobs: `nice -n 10`, CPU only (no GPU here).

## Plan

1. **Scratch environment:** `.venv-kraken6` (or Docker image `kraken-service:k6`) with the current
   kraken release; confirm `MiDRASH_Gen_01.mlmodel` still loads and recognises identically on a few
   lines (kraken 4→5/6 changed segmentation/polygon handling; recognition should be stable).
2. **Segmenter A/B on the religious benchmark (and PGP benchmark):** for each page produce `kraken_raw`
   and `kraken_seg` under tags such as `k6_blla_new`, `k6_orli`, `k4_default` (the cached baseline), with
   Gen_01 recognition; score with the harness scorer; report median/mean CER and F1, per `n_columns` and
   `is_talmud`, and the paired win counts against the cached 0.64/0.68 rows and against the MiDRASH 0.24.
   Also dump line counts and reading-order sanity (rows per GT line; MiDRASH noted left/right region
   order errors and vertical text loss).
3. **Genizah segmenter fine-tune** (only if step 2 leaves a large gap): build PageXML/ALTO from the KTIV
   word boxes for a decontaminated subset (start ~1,000 pages, mixed one- and two-column, mss-level
   split, hold out the benchmark mss), `ketos segtrain` from the new general `blla.mlmodel`, evaluate as
   in step 2. Report the region/line detection metrics kraken prints plus the end-to-end CER.
4. **Service cut-over proposal:** new image with the chosen kraken + segmenter, same API (`/transcribe`,
   `/preload`, line endpoint used by `transcribe_with_kraken_lines`), same mount; a one-page smoke test
   against the pipeline's client; then a coordinated swap of `kraken-linewise` on :8002 when the
   pipeline is paused. Keep the old image tagged for rollback.

## Deliverables

- A markdown results table (per-tag CER/F1 on both benchmarks, paired wins, timing per page) checked
  into `docs/` next to this file, plus the raw outputs under new `--kraken-tag` keys.
- The scratch environment recipe (Dockerfile or venv steps) and, if trained, the segmenter model on the
  NAS (`/Volumes/home/studio_offload/datasets/kraken_segmenter/`) with its training manifest.
- A go/no-go note for the cut-over with the expected effect on the consensus pipeline (agreed-line
  rate is the number that matters: currently 9.6% of lines on documentary pages, ~17% on easier ones).

## Related option: the MiDRASH text as a second reader (measured 2026-09-23)

The Zenodo transcriptions are text only (line-broken, reading order, no coordinates), so they cannot
replace segmentation geometry, but they can replace our Kraken's *text* wherever they exist: 29,795 of
the 139,159 pages in the served index are NLI-hosted (`/KTIV/<sys>/` image paths, 8,404 docs), and
Zenodo covers 7,709 of those 8,403 manuscripts (92%) — mostly KTIV-only literary manuscripts, plus
1,874 letter and 1,185 legal pages. The current documentary pipeline queue is PGP photos (only 283 of
5,181 edition pages are NLI-hosted), so this does not help the documentary batch. Design if pursued: a
third reader `midrash_zenodo_0.8` matched by sys_num + page position (FL − 3), VLM↔Zenodo line
agreement by text similarity with monotonic alignment, box from our Kraken fragments where present
else the VLM box (80% on the right line on KTIV scans), new rule version (site must accept it).
Expected effect: verified-line rate on KTIV-hosted pages rises from the Kraken-limited ~10–17% to
something near the two readers' agreement (both ≤0.24 CER).

## Decision 2026-09-23 and the re-run plan

The user chose to keep line geometry from the consensus pipeline's own Kraken (not the Zenodo text) and
to **prioritise this upgrade**. The pipeline keeps running meanwhile: its VLM outputs are cached per
page and stay valid; only the Kraken fragments need redoing. `two_reader_lines.py` is getting a
`--rekraken` mode (Kraken-only pass over the cached VLM outputs, new fragments stored under
`frags_by_htr[<htr_model_name>]`, records rebuilt and `ai_read.htr_model` restamped; `--kraken-model`,
`--htr-model-name`). When the new service is live, the re-run over the ~8.6k pages read so far is one
command at roughly 24 s/page (≈57 h for the 8,577 records read by 2026-09-23, ≈6 days for all ~21k once
the pipeline finishes), resumable. **Added 2026-09-23 (`tests/test_two_reader_rekraken.py`, 12 tests, smoke-tested on one
page: identical fragments and n_agreed):** `--rekraken --kraken-model PATH --htr-model-name NAME
[--kraken-cache-suffix KEY] [--all-cache] [--force] [--raw-dir DIR]`; new fragments go to
`frags_by_htr[KEY]`, legacy `frags` = `MiDRASH_Gen_01` untouched; `--all-cache` groups cache entries by
image hash (8,608 Kraken reads instead of 14,718). Caveat: the service keeps ONE model loaded and serves
one request at a time, so a new-model pass alongside the running pipeline would reload the model on
every request — run it after the pipeline finishes, or against a second service instance via
`KRAKEN_MICROSERVICE_URL`.

## Open points

- kraken 5 vs 6: pick the version Orli needs; check that Gen_01 (`.mlmodel`, kraken 4 format) loads.
- Timing: the current service takes ~23 s/page median on edition pages; a bigger segmenter or Orli at
  1920×1440 may be slower on CPU — measure, the pipeline budget is ~40 s/page.
- Licences: Gen_01 is CC-BY-NC-SA (internal use), the Kiessling models Apache 2.0.
