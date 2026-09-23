# Kraken 7 segmenter A/B — results and cut-over note (2026-09-23)

Follow-up to `docs/handoff_kraken5_segmenter.md`. Steps 1–2 done; **step 4 (cut-over) done 2026-09-23
12:26 EDT**; step 3 (Genizah segmenter fine-tune) in progress — see the two sections at the end.

## TL;DR

* **Upgrade target is kraken 7.0.3**, not 5/6. The current release is 7.1.1, but Orli (pinned at commit
  `7996f29`, see below) requires `kraken~=7.0.2`, and 7.0.3 carries the fix for loading pre-1.0 CoreML
  models like `MiDRASH_Gen_01.mlmodel`.
* **Best configuration: kraken 7.0.3 + the new general blla model (Zenodo 22879549, 2026-09-21).**
  Religious `kraken_seg` CER 0.635 → 0.556, PGP `kraken_raw` 0.358 → 0.297, wins roughly 2:1 page-paired on
  both benchmarks, and it runs in ~10–13 s/page on 6 cores versus ~23 s for the current service.
* **Consensus effect (the number that matters): agreed lines 11.9 % → 14.8 %** on 60 documentary pages
  (758 VLM lines, current line rule v3, cached v21b VLM lines reused). 13 pages up, 1 down.
* **Orli is a no-go** here: good on two-column pages but it misses lines on single-column fragments
  (0.75 detected lines per GT line), is 3× slower, and *lowers* agreement to 7.3 %.
* **The MiDRASH gap is not closed** (0.556 vs 0.240). Two causes, both visible on the pages:
  (1) **rotated scans**: KTIV images whose text runs vertically (e.g. T-S F2(2).75), where blla finds
  only fragments; (2) **line fragmentation on skewed or damaged pages**, where blla splits each line into
  overlapping pieces and also detects the stitched border, the ruler and the shelf-mark label as text lines.
  The error is mostly *missing* text (aligned precision 0.77, recall 0.48, output 68 % of GT length).
  Excluding the 23 pages that recover < 20 % of the GT letters gives 0.486 vs MiDRASH 0.222 on the rest.

## Benchmarks

Same scorer as the harnesses: `cer_pair(normalize_ink_hypothesis(hyp), genizah_visible_ink_gt(gt))`
and `aligned_prf` F1, pages with ≥ 50 GT letters. Recognition is `MiDRASH_Gen_01` throughout.
The rows `kraken_raw`/`kraken_seg` are the cached kraken 4.3.13 service outputs; they reproduce the
published numbers (religious 0.677/0.635 on the 138 MiDRASH-matched pages, PGP 0.358). `_raw` =
segmenter reading order, `_seg` = lines re-ordered by geometry (`ktiv_layout.reorder_ocr_lines`).

### Religious benchmark (140 pages; MiDRASH comparison on the 138 matched pages)

| key | CER med | CER mean | F1 med | wins/losses vs k4 | CER med (138) | wins vs MiDRASH |
|---|---:|---:|---:|---:|---:|---:|
| kraken_raw (k4) | 0.680 | 0.614 | 0.421 | — | 0.677 | 8/138 |
| kraken_seg (k4) | 0.641 | 0.561 | 0.489 | — | 0.635 | 9/138 |
| kraken_raw_k7_default | 0.658 | 0.577 | 0.463 | 96/37 | 0.654 | 9/138 |
| kraken_seg_k7_default | 0.592 | 0.546 | 0.496 | 88/41 | 0.575 | 10/138 |
| **kraken_raw_k7_blla2026** | 0.587 | 0.552 | 0.536 | 94/45 | 0.582 | 12/138 |
| **kraken_seg_k7_blla2026** | **0.574** | **0.528** | **0.547** | 90/48 | **0.556** | 13/138 |
| kraken_raw_k7_orli | 0.703 | 0.634 | 0.334 | 59/81 | 0.703 | 7/138 |
| kraken_seg_k7_orli | 0.706 | 0.637 | 0.345 | 52/88 | 0.706 | 9/138 |
| MiDRASH Zenodo transcriptions | | | | | 0.240 | |

By layout (CER median; the benchmark's `is_talmud` flag is True on 138/140 pages, so that slice is uninformative):

| key | two-column (33) | single-column (107) |
|---|---:|---:|
| kraken_seg (k4) | 0.504 | 0.654 |
| kraken_seg_k7_default | 0.409 | 0.618 |
| kraken_seg_k7_blla2026 | **0.399** | **0.587** |
| kraken_raw_k7_orli | 0.393 | 0.753 |

Lines emitted per GT line (median): k7_default 1.92, k7_blla2026 1.39, Orli 0.75 (over- vs under-segmentation).

### PGP benchmark (131 pages)

| key | CER med | CER mean | F1 med | wins/losses vs k4 |
|---|---:|---:|---:|---:|
| kraken_raw (k4) | 0.358 | 0.394 | 0.730 | — |
| kraken_seg (k4) | 0.437 | 0.455 | 0.679 | — |
| kraken_raw_k7_default | 0.347 | 0.385 | 0.743 | 79/49 |
| kraken_seg_k7_default | 0.340 | 0.364 | 0.754 | 95/25 |
| **kraken_raw_k7_blla2026** | **0.297** | 0.346 | **0.782** | 99/31 |
| **kraken_seg_k7_blla2026** | **0.295** | **0.335** | **0.782** | 101/20 |
| kraken_raw_k7_orli | 0.356 | 0.377 | 0.748 | 62/69 |
| kraken_seg_k7_orli | 0.396 | 0.407 | 0.712 | 68/54 |

(PGP GT has no line breaks, so lines-per-GT-line is not defined there.)

### Timing (6-CPU container, full `/transcribe_lines` incl. nlbin + two-pass fallback)

| tag | religious s/page med (p90) | PGP s/page med (p90) |
|---|---:|---:|
| k7_default | 12.1 (19.1) | 10.1 (13.4) |
| k7_blla2026 | 13.0 (19.6) | 10.2 (13.3) |
| k7_orli (bf16, CPU) | 37.6 (70.8) | 39.6 (64.2) |

The current k4 service is ~23 s/page median on edition pages. The pipeline budget is ~40 s/page, which
blla2026 is well inside.

### Two-reader agreement probe (60 documentary pages, seed 0)

`src/datasets/consensus/probe_segmenter_agreement.py`. The VLM reads the whole page, not Kraken crops,
so the cached v21b `vlm_lines` are reused. Both sidecars are rebuilt under the current rule
(`lines-v3-20260917`), so base and candidate are like-for-like. Images are fresh downloads checked
against the cached sha256 and oriented size. Nothing under `ai_reads/` is written.

| reader | agreed / 758 VLM lines | letter (401) | legal (297) | ketubah (39) | pages up / down |
|---|---:|---:|---:|---:|---:|
| k4 cached Gen_01 frags | 90 (11.9 %) | 8 % | 16 % | 28 % | — |
| k7_default | 95 (12.5 %) | 9 % | 16 % | 28 % | 5 / 2 |
| **k7_blla2026** | **112 (14.8 %)** | **11 %** | **19 %** | 28 % | **13 / 1** |
| k7_orli | 55 (7.3 %) | 6 % | 7 % | 28 % | 9 / 8 |

Per-page rows: `docs/kraken7_segmenter_ab/agreement_k7_<tag>.jsonl`.

## Other findings

* **Recognition is stable across versions.** Gen_01 loads in 7.0.3. On identical segmentation, the
  7.x task-API recogniser at batch size 1 is byte-identical to legacy `rpred`. Batch 16 changes the output
  slightly (CTC padding), so the service keeps batch 1.
* **The k4 → k7 difference with the same default model is post-processing.** kraken 4.3.13 and 7.0.3
  ship the *same* `blla.mlmodel` (md5 `0e3f0e1c…`). kraken 7 puts the right-hand column first on RTL
  two-column pages (kraken 4 started on the left), which is most of the `k7_default` gain on two-column
  pages. Baseline geometry also shifts by a few pixels.
* **Thread pinning matters.** Inside a `--cpus 6` container torch still sees 16 cores; unpinned,
  recognition took 84 s on a page that takes 8.5 s with `torch.set_num_threads(6)`. The service reads
  `KRAKEN_THREADS`.
* **Orli caveats**: bf16 only; upstream HEAD after commit `7bf383e` (2026-08-29) loops to the line cap
  with the published weights ([greekOCR#142](https://github.com/kkkamur07/greekOCR/issues/142)), hence the
  pin at `7996f29`. It emits baselines only; the service asks it to polygonize with kraken's polygonizer.
* The old `blla.segment()` API is deprecated (removal in kraken 8) and cannot load the 2026 blla model;
  the service uses `SegmentationTaskModel`.

## Recipe

* Image: `src/services/kraken_microservice/k7/` (Dockerfile, requirements, `main.py`: same API as the
  kraken 4 service: `/health /preload /transcribe /transcribe_lines`). Build with
  `docker build -t kraken-service:k7 src/services/kraken_microservice/k7`.
* Run the test container: `src/services/kraken_microservice/k7/run_k7.sh default|blla2026|orli` → container
  `kraken-k7` on host **:8003**, `--cpus 6`, same read-only model mount, segmenter models mounted from
  the NAS (`/Volumes/home/studio_offload/models/kraken_segmenters/{blla_2026,orli_base}`, md5-verified,
  Apache-2.0).
* A/B driver: `src/datasets/evaluations/helper_eval_scripts/kraken_segmenter_ab.py --benchmark
  religious|pgp --tag k7_<seg>` (one `/transcribe_lines` call per page; writes
  `kraken_{raw,seg}_<tag>.txt` + `kraken_lines_<tag>.json` next to the cached outputs; refuses :8002).
  Scorer: `kraken_segmenter_ab_score.py` → `docs/kraken7_segmenter_ab/ab_scores.json`, `ab_tables.md`.
* MiDRASH per-page CERs (from the 2026-09-23 session): `docs/kraken7_segmenter_ab/zenodo_vs_kraken_religious.json`.

## Go / no-go for the cut-over (done — see "Cut-over record" below)

**Go for kraken 7.0.3 + blla2026**, as a coordinated swap:

1. Bake the segmenter into the image (`COPY` blla_2026/blla.mlmodel, `ENV KRAKEN_SEGMENTER=…`) so production
   doesn't depend on the NAS mount; tag `kraken-service:k7-blla2026`. Keep `kraken-service:linewise`
   tagged for rollback.
2. One-page smoke test of the pipeline's own client against :8003 (`run_kraken` → `sidecar`).
3. When the consensus pipeline is between pages (it resumes by done keys), stop `kraken-linewise` and
   start the new image as `kraken-linewise` on :8002 with `--restart unless-stopped` and no `--cpus` cap,
   or keep the cap. Rollback = the same swap back.
4. Stamp the change: pages read after the swap should record a different `htr_model` or cache key
   (e.g. `MiDRASH_Gen_01@k7-blla2026`) so agreement numbers before and after aren't mixed. Optionally
   `--rekraken` the already-read documentary pages later (a ~10 s/page Kraken-only pass, no VLM).

Expected effect: about +3 points of agreed lines on documentary pages (11.9 → 14.8 % in the probe), and
Kraken time per page roughly halves. It doesn't change the VLM side.

## Proposed next step (needs a go): Genizah segmenter fine-tune

The remaining gap is segmentation on Genizah-specific layouts (rotation, skew/damage fragmentation,
border/label false positives). Plan:

* **Orientation (cheap, first):** detect vertical-text pages and rotate before segmentation: try
  0/90/270 when the first pass yields short fragments (e.g. median line < 5 letters), keep the rotation
  with the most recognised letters × confidence. Measurable on the 23 low-recovery religious pages.
* **Fine-tune blla2026 with `ketos segtrain`** on ~1,000 decontaminated KTIV API-shape pages (mixed one-
  and two-column, manuscript-level split, benchmark manuscripts held out via
  `build_ktiv_dataset.exclude_benchmark_manuscripts`). PageXML from `ktiv_layout.reconstruct_page`:
  line polygon = word-box union, baseline = ink-centred bottom edge (`export_ktiv_lines.ink_centre`),
  regions = column clusters. Pages where KTIV transcribes only part of the page need a mask or
  exclusion, otherwise untranscribed text becomes negatives. Evaluate end to end as above plus the
  agreement probe.
* **Cost:** CPU-only here. Kraken segtrain at 1800 px on 6 cores is likely hours per epoch for 1k pages, so
  a multi-day run, or wait for the ROCm box (ETA Oct 23–26). Colab is an alternative if the data is staged
  there.

## Cut-over record (2026-09-23)

* Plan reviewed by a 3-lens workflow (pipeline runtime / Docker infra / data provenance) before execution;
  its fixes were applied: socket-state pause gate instead of log silence, measured memory cap (peak 7.0 GiB
  on a 95 MP pipeline page → `--memory 10g`, not the planned 6g), read-only weight mount, STOP budget with
  automatic rollback, boundary recorded by job, verification only on k7-served jobs.
* `logs/kraken_cutover_0923.sh` (dry-run first): paused the running pipeline with SIGSTOP only when `lsof`
  showed its LM Studio socket open and no :8002 socket (6 unsafe windows declined, 7th taken), swapped
  containers, preloaded Gen_01, resumed — **4 s STOP**, no Kraken failures (failures.jsonl 78 → 78).
* `:8002` = `kraken-linewise` on `kraken-service:k7-blla2026` (`--restart unless-stopped --cpus 6 --memory 10g`,
  weights `:ro`); rollback copy `kraken-linewise-k4` (old image, stopped, restart=no). Runtime doc updated.
* Record `logs/kraken_cutover_0923.json`: jobs 926–927 were in flight and carry kraken-4 fragments; job
  928 (`Cambridge_CUL_T_S_AS_146_210#0`) onwards is k7-blla2026.
* **Pending provenance stamp.** The running pipeline stores k7 fragments under the legacy key
  `MiDRASH_Gen_01` (no stamp option in its code). After it exits:
  `.venv/bin/python -m src.datasets.consensus.stamp_htr_cache_key` (dry-run first) sets raw-cache
  `htr_model` and record `ai_read.htr_cache_key` = `MiDRASH_Gen_01@k7.0.3-blla2026` for the k7 set derived
  from the pipeline logs (fragment counts cross-checked). Until then, agreement-probe "k4 baselines" and
  default-key `--rekraken` runs over post-cut-over pages are unreliable.

## Genizah segmenter fine-tune (in progress)

Training data: `/Volumes/home/studio_offload/datasets/kraken_segmenter/ktiv_pagexml_v1/` (2,000 decontaminated
KTIV API-shape candidate pages, ≤ 3 per manuscript, 112 val manuscripts held out; benchmark manuscripts
excluded by `exclude_benchmark_manuscripts` + shingle decontamination).

* `src/finetuning/kraken/export_ktiv_pagexml.py` — candidates / relines / write (see its docstring);
  `seg_gate.py` — blla2026 predictions per page in the container; `seg_geometry.py` — shared cover/match
  geometry; `seg_loader_gate.py` — the export through kraken's own training data path;
  `segtrain_ktiv.sh` — stage / gate / baseline / train / resume.
* Targets: lines blla2026 already segments 1:1 take its baseline (the convention Gen_01 reads); other
  lines get an ink-estimated baseline (body bottom, calibrated against blla2026: shift 0.076 pitch,
  residual spread 0.033 pitch) with the page's consensus slope, refitted per line when an end is off;
  unplaceable lines, vertical marginal words and untranscribed writing blla2026 reads confidently are
  painted out with parchment colour so no writing is taught as background.
* Two independent visual audits (6 auditors each, 50–60 pages): v1 exporter 36 % bad pages, 7.2 wrong +
  4.1 missing targets / 100 lines → **v2 15 % bad, 3.8 wrong + 0.8 missing / 100 lines: GO**; the two main
  remaining patterns (drift on warped lines, blla lines stopping a word short) were fixed after the audit
  (per-line refit on end mismatch, ink-extent clipping, blla extension to the GT words).
* Memory: training peak 5.8 GiB at 1,350 px width, 6.7 GiB at 2,500, > 10 GiB at 4,000 (OOM) → pages wider
  than 2,600 px excluded (6.5 %); bf16 barely saves memory and is 55× slower on this CPU.
* Loader gate on the interim export: all XML parse, per-class counts equal the XML, no failed samples,
  every baseline left→right, input height 1800.
* ketos 7.0.3 `segtest` crashes printing its pixel table; the warm-start reference is instead segtrain's
  own validation of the unchanged blla2026 (1 train page, 1 step at lr 1e-12) — `segtrain_ktiv.sh baseline`.
* Run: `logs/kraken_segtrain_launch_0923.sh` chains write → stage → loader gate → baseline → train
  (`kraken-segtrain-ktiv-v1`, 6 CPUs, 8 GiB, cpu-shares 256, lr 1e-4 cosine, augment, early stop lag 8,
  ≤ 60 epochs; checkpoints + `train.log` in `…/kraken_segmenter/runs/ktiv_seg_v1/`).
* Evaluation of checkpoints: `run_k7.sh /runs/ktiv_seg_v1/<model>.safetensors rgb` on :8003, then
  `kraken_segmenter_ab.py` (both benchmarks) + `kraken_segmenter_ab_score.py` + the agreement probe; also
  blla2026 with `KRAKEN_SEG_INPUT=rgb` (the fine-tune trains on RGB pages; the service segments nlbin output
  by default).

## State

* `:8002` runs k7-blla2026 (see cut-over record); test image `kraken-service:k7` (+ `KRAKEN_SEG_INPUT`),
  `kraken-service:k7-base` kept; test container `kraken-k7` only on :8003 when evaluating.
* New cache files only: `kraken_{raw,seg}_k7_*.txt`, `kraken_lines_k7_*.json` in both benchmarks'
  raw-output dirs. `religious_scores_*.csv` and the paper tables were not rewritten.
