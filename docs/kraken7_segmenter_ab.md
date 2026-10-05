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

### Mid-run evaluation after epoch 3 (2026-09-23 22:08–00:00, training paused and resumed)

Validation (129 images, segtrain's own metrics): `val_bl_f1` blla2026 0.763 → epoch 1 0.935 → 2 0.943 → 3 0.945
(`val_mean_iu` 0.389 → 0.482 / 0.482 / 0.488; early stopping monitors mean IoU, which is insensitive —
checkpoints are chosen by baseline F1 and benchmark CER instead; all epoch checkpoints are kept).

End to end, same harness scorer, Gen_01 recognition, epoch-3 checkpoint converted to safetensors and served
with `KRAKEN_SEG_INPUT=rgb` (the fine-tune trains on colour pages):

| religious (140), CER med | all | two-col (33) | single-col (107) | wins vs k4 |
|---|---:|---:|---:|---:|
| kraken_raw k4 | 0.680 | 0.679 | 0.681 | — |
| kraken_raw blla2026, binarised (**prod since 12:26**) | 0.587 | 0.564 | 0.593 | 94/45 |
| kraken_raw blla2026, rgb | 0.655 | 0.568 | 0.698 | 62/78 |
| **kraken_raw fine-tuned e03, rgb** | **0.410** | **0.316** | **0.467** | 105/35 |
| kraken_seg fine-tuned e03, rgb | 0.406 | 0.316 | 0.454 | 102/38 |
| MiDRASH Zenodo (138) | 0.240 | | | |

| PGP (131), CER med | raw | seg | wins vs k4 (raw) |
|---|---:|---:|---:|
| k4 | 0.358 | 0.437 | — |
| blla2026 binarised (prod) | 0.297 | 0.295 | 99/31 |
| blla2026 rgb | 0.305 | 0.297 | 95/36 |
| **fine-tuned e03 rgb** | **0.270** | **0.274** | 113/18 |

RGB input alone makes blla2026 worse on the religious pages (0.587 → 0.655), so the gain is the fine-tune's.
PGP images are out-of-domain for the fine-tune (photographs, documentary hands) and still improve.
Lines per GT line (religious): blla2026 1.39 → fine-tuned 0.98 (fragmentation gone).

Agreement probe — **caveat: sample drift.** The probe's seeded sample is drawn from the growing v21b records
file, so this run sampled a different 60 pages (848 VLM lines, 4 overlapping) than the earlier probes; the
numbers are comparable only within the run: cached k4 fragments 9.1 % → blla2026 rgb 11.2 % (7 up / 8 down)
→ fine-tuned e03 rgb 12.1 % (9 up / 3 down). No post-cut-over (k7) pages were in the sample. The probe now
takes `--pages-from <earlier probe jsonl>` so later runs reuse identical pages; the next evaluation runs prod
(blla2026 binarised), blla2026 rgb and the best checkpoint on the union of both samples (120 pages).
The agreement gain is much smaller than the CER gain: agreement needs whole lines at ≥ 0.8 similarity, which
on documentary hands is limited by Gen_01 recognition rather than segmentation.

### Final evaluation (training stopped after epoch 10, 2026-09-24)

Training: `val_bl_f1` 0.935 → 0.952 by epoch 4, flat through epoch 10 (stopped by decision; checkpoints e1–e10 kept,
e8–e10 converted in `runs/ktiv_seg_v1/eval/`). **e10 chosen** (best on both benchmarks among e3/e9/e10).

Test-time multi-view reading (k7 test image `kraken-service:k7-mv`, `KRAKEN_MULTISCALE=1 KRAKEN_ORIENT=1`, `views.py`):
the remaining MiDRASH gap was mostly *text scale* — blla resizes every page to 1800 px height, and a third of the
religious benchmark's pages reach the model with writing larger than 97 % of the training pages (recovery fell to
0.53 there, 14 near-total failures) — plus ~5 rotated scans. Views shrink the writing (bottom padding, scale 0.5 /
0.33, recognition at full resolution) and try 90°/270° when the upright read is fragmentary; the best-reading view
wins. Reviewed by a 3-lens workflow + verification (fixed: bottom lines lost in shrunk views, over-eager rotation
trigger, rotated reads breaking geometry reorder, Orli distortion).

| CER median | religious raw / seg | 2-col / 1-col (seg) | PGP raw / seg |
|---|---:|---:|---:|
| k4 (old service) | 0.680 / 0.641 | 0.504 / 0.654 | 0.358 / 0.437 |
| blla2026 binarised (**prod now**) | 0.587 / 0.574 | 0.399 / 0.587 | 0.297 / 0.295 |
| blla2026 + multi-view | 0.462 / 0.397 | 0.299 / 0.404 | 0.272 / 0.287 |
| fine-tuned e10 (rgb) | 0.409 / 0.399 | 0.314 / 0.461 | 0.264 / 0.267 |
| **fine-tuned e10 + multi-view** | **0.336 / 0.334** | **0.248 / 0.361** | **0.253** / 0.267 |
| MiDRASH Zenodo (138 matched) | 0.240 | | |

Agreement probe on the 89 frozen documentary pages every configuration completed (1,178 VLM lines, rule v3):

| reader | agreed | pages up/down vs k4 | Kraken s/page med / p90 |
|---|---:|---:|---:|
| k4 cached | 11.1 % | — | — |
| blla2026 binarised (prod) | 12.8 % | 17 / 5 | 10.1 / 17.3 |
| blla2026 + multi-view | 14.5 % | 22 / 5 | 25.4 / 37.1 |
| fine-tuned e10 | 15.0 % | 19 / 2 | 8.4 / 16.0 |
| **fine-tuned e10 + multi-view** | **16.6 %** | **26 / 2** | 22.4 / 33.6 |

(Single-view e10 on all 116 frozen pages: 10.2 % → 14.4 %; prod 12.1 %.)

**Robustness issue found:** the multi-view test containers were OOM-killed (8 GiB cap) after ~360 requests — the
service's memory grows across requests (the same glibc fragmentation as training); 27 / 9 probe pages failed at the
tail. Must be fixed (malloc arena/trim settings, `malloc_trim` per request) and soak-tested before any prod use.

**Incident (2026-09-24 11:45 EDT):** with three test containers running beside prod, the 16 GiB Docker VM ran out
of memory and the kernel killed prod `kraken-linewise` (not a container OOM; Docker restarted it). One pipeline page
(job 415 of the v22c run) failed with `kraken` and is in `failures.jsonl` for retry. Prevention: test containers get
`--oom-score-adj 1000`, at most two at a time.

### Plan for the second swap (fine-tuned e10 + multi-view), agreed with the v22 consensus session

* **When:** between pipeline runs, after v22c reaches DONE (ETA ~2026-09-25 12:30 EDT); a mid-run swap would
  mislabel pages, because the pipeline's HTR key is fixed for the life of a process.
* **Before:** memory-leak fix soak-tested (`src/services/kraken_microservice/k7/soak_test.py`, image
  `kraken-service:k7-mv2`); v22 session runs `stamp_htr_cache_key.py --all-k7-logs logs/two_reader_v21b_v22c_0924.log`
  on the finished v22c output (retroactive k7 key).
* **Swap:** new prod image = k7-mv2 code + e10 weights + `KRAKEN_SEG_INPUT=rgb KRAKEN_MULTISCALE=1 KRAKEN_ORIENT=1`;
  guards: a real `/transcribe_lines` request must succeed on the new container before anything resumes, disk ≥ 10 GB
  free, no test containers running; old image kept for rollback; boundary logged.
* **After:** the next pipeline run (and its watchdog's auto-resume line) is launched with
  `--htr-cache-key MiDRASH_Gen_01@k7.0.3-ktivseg-e10-mv` (flag added in commit fbbb516 on kg/pipeline-version-stamp;
  byte-identical to the stamp tool). A bare `--rekraken` without `--kraken-cache-suffix` is being made to refuse
  unless `--legacy-key` is passed.
* **Optional before that:** a same-config leak-fix swap of prod (`kraken-service:k7-blla2026-lf`, current key kept)
  only with the user's OK, after `check_kraken_identity.py` shows identical fragments on ≥ 20 v22c pages.

### Leak-fix swap (2026-09-24 23:35 EDT, user-approved)

`:8002` → `kraken-service:k7-blla2026-lf` (same model and settings; `MALLOC_ARENA_MAX=2`, mmap/trim thresholds,
`gc.collect()` + `malloc_trim(0)` after every request). Gates: soak 542 requests flat at 3.9 GiB with 0 failures
(`soak_test.py`); identity 25/25 v22c pages, 745 fragments identical to prod's cache (`check_kraken_identity.py`).
32 s STOP including a real `/transcribe_lines` check; boundary: jobs 1602–1603 old, 1604 onward new; no
`failure=kraken`. In production afterwards: anon memory flat at 4.4–5.0 GiB over 611 requests / 6 h (the unfixed
image climbed 4.8 → 8.4 GiB over ~500 requests toward its 10 GiB cap). Record `logs/kraken_leakfix_swap_0924.json`,
trace `logs/kraken_prod_memtrace_0924.log`; rollback container `kraken-linewise-k7a` (stopped).

### Second swap — fine-tuned e10 + multi-view in production (2026-09-25 15:12 EDT, user-approved)

Between runs, agreed with the v22 session: it stopped the v22c cycle guard, watchdog and pipeline at 15:09 (last
page job 3176), backed up the records file and stamped v22c (4,560 records → `MiDRASH_Gen_01@k7.0.3-blla2026`;
9,412 pre-cut-over records untouched). `logs/kraken_ktivseg_swap_0925.sh` then swapped `:8002` to
`kraken-service:k7-ktivseg-e10` (container ffc67056b29e; the prod image reproduced the evaluated outputs on 12/12
benchmark pages, incl. rotated and rescaled views, before the swap) and verified health, preload and a real
`/transcribe_lines` (71 lines). Rollback: `kraken-linewise-lf`. The next pipeline run (v22d) carries
`--htr-cache-key MiDRASH_Gen_01@k7.0.3-ktivseg-e10-mv` on its launch and watchdog resume line.

## State

* `:8002` runs k7-blla2026 (see cut-over record); test image `kraken-service:k7` (+ `KRAKEN_SEG_INPUT`),
  `kraken-service:k7-base` kept; test container `kraken-k7` only on :8003 when evaluating.
* New cache files only: `kraken_{raw,seg}_k7_*.txt`, `kraken_lines_k7_*.json` in both benchmarks'
  raw-output dirs. `religious_scores_*.csv` and the paper tables were not rewritten.
