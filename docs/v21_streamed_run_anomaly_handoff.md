# v21 streamed-run anomaly — investigation handoff (2026-09-11)

Hand-off for a second investigator (local model with the same directories). Everything below is
verified unless marked *hypothesis*. Absolute paths are on the Mac Studio; `REPO` =
`/Users/isaac/Documents/GitHub/historical-document-analysis`, `SCRATCH` =
`/private/tmp/claude-501/-Users-isaac-Documents-GitHub-historical-document-analysis/58185f3f-1ea5-4c5c-a8f9-fdf6db6e6fcd/scratchpad`.
Python for the repo: `REPO/.venv/bin/python`. Secrets in `REPO/.env` (`HF1_TOKEN`, `WANDB_API_KEY`).

## 1. What the run is

* **v21** = first Qwen3-VL-8B QLoRA run whose training mixture is **streamed** (`datasets`
  IterableDatasets) instead of map-style. Warm start from v2.0a step 1800
  (`isaacmg/qwen3-vl-8b-hebrew-v20a-ckpt@af9df6a0ad4743bc8493a7cf0c14bc3b4b796dd8`), MERGER_MODE
  frozen, LR 5e-5 cosine (warmup 0.02), batch 1 × grad-accum 8, max_seq 12288, 2000 steps, eval every
  100 on 230 rows. Mixture: v2.0a's transcription backbone rescaled to 80 % + 20 % grounding split over
  ten families (six new in v2.1). Colab A100-40GB.
* Running notebook **with outputs** (authoritative): `REPO/src/datasets/raw_data/cairo_genizah/ktiv/genizah_v21(1).ipynb`
  (repo copy without outputs, since renamed to `REPO/src/finetuning/qwen_hebrew/colab/genizah_v21b_dispatchfix.ipynb`; cell sources dumped to
  `SCRATCH/v21_cell{2,4,5,8}.py`).
* Reference run **v2.0a** notebook with outputs: `REPO/src/datasets/raw_data/cairo_genizah/ktiv/genizah_v20a_5.ipynb`
  (map-style mixture, otherwise the same cells — diff confirmed: collator identical except the smoke row,
  model cell differs only in the warm-start lines).
* Data: hub `isaacmg/genizah_ktiv_v3` (PRIVATE) revision `45fe477baf11029e89014633dbcad974591e3363`,
  163 parquet files, 76.7 GB, splits `train_<task>` (14 families) + `val` (2,754 rows). Local arrow source of
  the push: `/Volumes/home/studio_offload/datasets/genizah_ktiv_v3` (57,246 train rows; columns
  `image, question, answer, task, section, stem, label_source, target_chars, target_tokens, image_width,
  image_height`). Push script `REPO/src/finetuning/qwen_hebrew/push_ktiv_dataset.py` (index-view `select`
  per task, `push_to_hub`); builder `REPO/src/finetuning/qwen_hebrew/build_ktiv_dataset.py`.
* Other sources in the mixture (map-style on Colab, converted with `as_stream()` =
  `ds.shuffle(seed=3407).to_iterable_dataset(num_shards=4)`): `isaacmg/genizah_clean_v2@57366ad3…`,
  `isaacmg/synthetic_hebrew_v3@59abcf7c…`, `isaacmg/talmud_finetune_v2` (unpinned).
* Checkpoints: `isaacmg/qwen3-vl-8b-hebrew-v21-ckpt` (`last-checkpoint/adapter_model.safetensors`,
  `trainer_state.json`; rolling, save_total_limit 2).

## 2. The symptom (numbers)

| step | v21 eval/loss | v2.0a eval/loss | v21 train/loss | v2.0a train/loss |
|---|---|---|---|---|
| 100 | 0.6830 | 0.7457 | 3.03 (window mean 3.08) | 0.60 (0.57) |
| 200 | 0.7059 | 0.7650 | 2.87 | 0.52 |
| 300 | 0.7202 | 0.7489 | 2.83 | 0.64 |
| 400 | 0.7534 | 0.7372 | 3.08 | 0.38 |
| 500 | 0.7558 | 0.7225 | 3.23 | 0.64 |
| 600 | 0.7598 | 0.7285 | 3.05 | 0.52 |
| 700 | 0.7644 | 0.7127 | 2.84 | 0.55 |

* Eval loss **rises monotonically** and crossed above v2.0a's curve at step 400. Best v21 checkpoint is
  step 100 and it is still worse than v2.0a-1800 (0.6733). Note v21's eval set has 35 extra rows (new
  grounding families) vs v2.0a's 195 — the curves are close but not identical metrics.
* Train loss is **~3.0 from the first log (step 10: 3.20)**, flat, low variance (±0.15 per 80-row window),
  vs 0.51 at step 10 and ~0.55 mean for v2.0a. grad_norm 1.4–2.2 (v2.0a 2–4). Step time 62.3 s vs
  62.5 s; GPU memory 35.7 % vs 38.3 % median → same-sized batches.
* User-observed consequence: the model is a transcription regression at every checkpoint since 100.
  Recommendation already given: interrupt the run (no upside), keep the kernel for diagnostics.

## 3. What is eliminated (evidence + where to find it)

1. **Software stack** — W&B `requirements.txt` of v20a / v20b / v21 are identical (unsloth 2026.8.9,
   unsloth_zoo 2026.8.6, transformers 4.57.6, trl 0.24.0, datasets 4.3.0, accelerate 1.14.0, peft 0.20.0,
   torch 2.11.0+cu128, bitsandbytes 0.50.2, xformers 0.0.35, pillow 11.3.0) **except pyarrow 23.0.1 (v21)
   vs 25.0.1**. Downloaded copies: `SCRATCH/wandb_req_genizah_v2{1,0a,0b}/`. The same Unsloth banner,
   "Double buffering", "smartly offload", "compile graph cache reset" messages appear in v20a_5's outputs.
2. **pyarrow** — a scratch venv with exactly pyarrow 23.0.1 + datasets 4.3.0 (`SCRATCH/venv23`) streams
   byte-identical rows (`SCRATCH/stream_hash_probe.py` → `SCRATCH/probe_pa23.txt`; `locate_word` rows share
   their page's pixel hash with the `fragment_transcribe` row of that page).
3. **Streaming reader** — first 3 rows of `train_fragment_transcribe` via `load_dataset(streaming=True)`
   are pixel-, question- and answer-identical to the parquet row group read with pyarrow.
4. **Row alignment on the hub** — `SCRATCH/vlm_hub_row_check.py` (→ `SCRATCH/vlm_hub_row_check.log`)
   fetched row-group 0 of the first train shard and of `val-00000-of-00008` by HTTP range and ran
   v2.0a-1800 through LM Studio (`qwen3-vl-8b-heb-v20a-step1800`, Studio :1234, client
   `REPO/src/models/ocr/lms_transcriber.py:transcribe_with_lm_studio`): **train CER 0.019 / 0.024 / 0.101,
   val CER 0.163 / 0.130 / 0.131** (letters-only, `src.datasets.consensus.line_rule.letters` + Levenshtein).
   Images and answers are correctly paired in both splits.
5. **Mixture composition** — under datasets 4.3.0, a notebook-identical `interleave_datasets`
   (probabilities, seed 3407, `all_exhausted`, buffered streams + `as_stream`) honours the probabilities over
   16k draws (grounding 19.9 % vs 20 % designed; per-source ratios 0.87–1.12). Inline script, see §6.
6. **Trainer/loop code paths** — read in `SCRATCH/unsloth_src/` (`us/` = unsloth 2026.8.9, `zoo86/` =
   unsloth_zoo 2026.8.6, `trl/` = trl 0.24.0, `tf4576/transformers/trainer.py` = transformers 4.57.6):
   * transformers `_inner_training_loop`: `tr_loss += training_step(...)` per micro-batch, logged
     `loss = tr_loss / (optimizer steps in window)`; iterable case uses `steps_in_epoch = max_steps × GA`,
     `get_batch_samples(epoch_iterator, 8)` → identical to map-style.
   * Unsloth replaces `Trainer.get_batch_samples` with `_unsloth_get_batch_samples`
     (`zoo86/unsloth_zoo/loss_utils.py:239`), which counts `labels[...,1:] != -100 & attention_mask[...,1:] != 0`
     over the 8 micro-batches (and calls `torch._dynamo.mark_static/mark_dynamic` on `input_ids`/`attention_mask`),
     and `compute_loss` with `_unsloth_pre_compute_loss` (`us/unsloth/models/_utils.py:3055`) which injects
     `inputs["num_items_in_batch"]`. `_unsloth_training_step` is the 4.57.6 source re-exec'd (its regex
     replacements do not match 4.57.6, so it is behaviourally the original). No IterableDataset-specific
     branch affects any of this. TRL 0.24's `SFTTrainer` skips dataset preparation (`skip_prepare_dataset`),
     uses our collator (`UnslothVisionDataCollator`, `resize="max"`, `train_on_responses_only=True`).
7. **In-kernel diagnostic 1** (`SCRATCH/colab_diag_v21.py`, run by the user in the live kernel at ~step 670,
   model.eval() + no_grad unless stated):
   ```
   mixture[0] fragment_transcribe (2128,3054) loss=1.091 seq=6847 answer_tok=322
   mixture[1] fragment_transcribe (1440,1293) loss=2.658 seq=6982 answer_tok=429   # genizah JA row
   mixture[2] page_extract        (1734,2248) loss=0.032 seq=7997 answer_tok=1461  # talmud, memorised
   mixture[3] crop_transcribe     (1056,2114) loss=0.029 seq=7035 answer_tok=517
   mixture[4] fragment_transcribe (2209,2942) loss=0.832 seq=6775 answer_tok=334
   mixture[5] fragment_transcribe (1251,2000) loss=0.253 seq=6972 answer_tok=415
   mixture[6] fragment_transcribe (2968,2189) loss=0.584 seq=7091 answer_tok=512
   mixture[7] section_transcribe  (2311,1739) loss=1.753 seq=6534 answer_tok=93
   ktiv_pages stream              loss=0.462 answer_tok=171
   eval_ds[0] map-style           loss=0.917 answer_tok=466
   genizah[0] map-style           loss=2.967 answer_tok=103   # Arabic-script row with (؟) markers
   genizah via as_stream          loss=2.658 answer_tok=429
   TRAIN-MODE trainer.compute_loss(eval_ds[0], n=own tokens)  = 0.917
   TRAIN-MODE trainer.compute_loss(ktiv_pages stream, n=own)  = 0.462
   ```
   (`fragment_transcribe` is the task name in BOTH the KTIV and genizah_clean datasets.) Token-weighted
   mean over the 8 streamed rows ≈ **0.59**; train-mode per-row loss equals eval-mode. ⇒ rows, kernels and
   train-mode forward are all fine **per row**. The 3.0 is manufactured between the per-row losses and the
   logged number, i.e. in the 8-micro-batch accumulation/normalisation of one optimizer step.

## 4. Open hypotheses (ranked) and the decisive test

**H1 (fits every number): `num_items_in_batch` does not take effect in the loop.** Then each micro-batch
returns its own per-row token-mean, `training_step` does not divide by GA (because `num_items_in_batch` is
not None and `model_accepts_loss_kwargs` is True), the log shows **8 × example-weighted mean** (≈3.0 when
many rows are near-memorised Talmud/synth pages), the accumulated gradient is 8 × the average per-row
gradient (grad_norm 1.4–2.2 = 8 × ~0.2 — consistent), and the objective silently becomes
**example-weighted instead of token-weighted**: a 20-token `locate_word` answer weighs as much as a
2,000-token page → grounding effectively up-weighted ~10× → transcription eval drifts up. Why it would
differ from v20a is unexplained (same code); candidates: the fused-loss `n_items` extraction in the
compiled forward (`zoo86/unsloth_zoo/compiler.py:1995-2085`, several fallbacks reading
`kwargs`/`all_locals`), or `has_kwargs` detection in `_unsloth_get_batch_samples` (cached per model class
name in `ALLOWED_NUM_ITEMS_IN_BATCH`).

**H2: the train-only `mark_dynamic` + torch.compile graph.** `_unsloth_get_batch_samples` marks dim 1 of
`input_ids`/`attention_mask` dynamic; the loop's batches therefore hit a dynamic-shape compiled graph that
diag 1's un-marked batches never used. A mis-traced dynamic graph (mRoPE / image-token scatter) would give
LM-only loss (~3) in the loop while eval (`unsloth_prediction_step`, no marks, no_grad) stays correct.

**H3: the loop's own dataloader yields something other than `iter(mixture)`** (accelerate wrapper,
`set_epoch`, skip logic). Least likely — single process, no resume.

**Decisive test — `SCRATCH/colab_diag2b_v21.py`** (paste into the live kernel after interrupting
`trainer.train()`; ≈3 min, 16 backward passes):

```python
import torch
dl = trainer.get_train_dataloader()
bs, n_items = trainer.get_batch_samples(iter(dl), trainer.args.gradient_accumulation_steps, model.device)
per = [int((b["labels"] != -100).sum()) for b in bs]
print(f"num_items_in_batch={n_items} | per-micro answer tokens={per} sum={sum(per)} | accepts_loss_kwargs={trainer.model_accepts_loss_kwargs}")
def ev(b):
    model.eval()
    with torch.no_grad():
        return model(**{k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in b.items()}).loss.item()
rows = []
for i, b in enumerate(bs):
    e = ev(b); model.train()
    a = trainer.training_step(model, dict(b), num_items_in_batch=n_items).item(); model.zero_grad(set_to_none=True)
    z = trainer.training_step(model, dict(b), num_items_in_batch=None).item();    model.zero_grad(set_to_none=True)
    rows.append((e, a, z))
    print(f"micro[{i}] tok={per[i]:>5} eval_mean={e:.3f} | step(N_total)={a:.4f} expected={e*per[i]/int(n_items):.4f} | step(None)={z:.4f} expected={e/8:.4f}")
A = sum(r[1] for r in rows); Z = sum(r[2] for r in rows); tw = sum(r[0]*k for r, k in zip(rows, per)) / sum(per)
print(f"SUM step(N_total)={A:.3f}  SUM step(None)={Z:.3f}  eval token-weighted={tw:.3f}  8×example-mean={sum(r[0] for r in rows):.3f}")
FastVisionModel.for_training(model);
```
Reading: `SUM step(N_total)` is what the loop logs per step. If each `step(N_total)` ≈ its `eval_mean`
and the SUM ≈ `8×example-mean` → **H1 confirmed** (fix: make the count reach the loss — e.g. verify
`trainer.model_accepts_loss_kwargs`, `_unsloth_get_batch_samples`'s `has_kwargs`, and the compiled
forward's `n_items`; or as a blunt workaround use `per_device_train_batch_size=8, gradient_accumulation_steps=1`
so there is no cross-micro-batch normalisation). If per-micro values are neither `expected` nor `eval_mean`
but uniformly high → **H2** (workaround: `os.environ["UNSLOTH_COMPILE_DISABLE"]="1"` before importing
unsloth, or `torch._dynamo.config.suppress_errors`; verify by re-running the cell). If the SUM ≈ 0.6 while the
run logs 3.0 → **H3**, inspect `trainer.get_train_dataloader()` wrappers and `trainer._train_batch_size`.

## 5. Dataset size question (answered)

76.7 GB = 1.52 MB/row × ~45k page-level rows: every page-level family (fragment, region, locate,
locate_word, line_index, read_box, read_box_word, layout_qa, line_of_phrase, grounded_detect,
grounded_page) embeds the full page JPEG, so each of the 3,158 pages is stored ~14×. v2 was 35.2 GB for the
same reason with fewer families. Per-family sizes: locate_word 14.3 GB, locate 9.6, line_index 9.4,
region 6.6, fragment/read_box_word/layout_qa/line_of_phrase/read_box ≈ 4.8 each, grounded_detect 3.9,
val 3.6, grounded_crop 2.3, section 2.0, line 0.8, grounded_page 0.4. Fix for v4: one `pages` split +
`page_id` on task rows joined in the collator (~5 GB), which also removes the need to stream.

## 6. How to get the logs

**W&B** (project `igodfried/qwen-hebrew-finetune`; run ids: v21 = `js3ku3tw`, v2.0a = `genizah-v20a`,
v2.0b = `s21stpko`; key in `REPO/.env`):
```python
from dotenv import load_dotenv; load_dotenv("/Users/isaac/Documents/GitHub/historical-document-analysis/.env")
import wandb
api = wandb.Api(timeout=60); proj = f"{api.default_entity}/qwen-hebrew-finetune"
run = api.run(f"{proj}/js3ku3tw")
train = [h for h in run.history(keys=["train/global_step","train/loss","train/grad_norm","train/learning_rate","_timestamp"], pandas=False) if h.get("train/loss") is not None]
evals = [h for h in run.history(keys=["train/global_step","eval/loss","eval/runtime"], pandas=False) if h.get("eval/loss") is not None]
sysm  = run.history(stream="events", pandas=False)                       # GPU memory/power etc.
run.file("requirements.txt").download(root="/tmp/v21", replace=True)     # exact package versions
run.file("wandb-metadata.json").download(root="/tmp/v21", replace=True)
print(run.config)                                                        # full SFTConfig
```
(`run.history()` with `keys=` returns only rows having those keys; `run.scan_history()` for everything.)
A polling watcher pattern is in `SCRATCH/watch_v21_eval.py` (log `SCRATCH/watch_v21_eval3.log`).

**Notebook outputs** (TRL's progress table is `text/html` in the training cell):
```python
import json, re, html
nb = json.load(open("REPO/src/datasets/raw_data/cairo_genizah/ktiv/genizah_v21(1).ipynb"))
for c in nb["cells"]:
    for o in c.get("outputs", []):
        h = "".join(o.get("data", {}).get("text/html", []))
        for r in re.findall(r"<tr>(.*?)</tr>", h, flags=re.S):
            print([html.unescape(re.sub(r"<.*?>", "", x)).strip() for x in re.findall(r"<t[hd]>(.*?)</t[hd]>", r, flags=re.S)])
```
Text outputs are under `outputs[*]["text"]`; the Unsloth/torch version banner is in cell 3's output.

**Hub**: `HfApi(token=HF1_TOKEN).dataset_info("isaacmg/genizah_ktiv_v3", revision=REV, files_metadata=True)`
for per-file sizes; `load_dataset_builder(...).info.splits` for row counts; streaming reads as in the notebook
(`load_dataset(REPO, split="train_<task>", revision=REV, streaming=True)`).

## 7. Environment rules for whoever picks this up

* The Mac Studio is shared production (`docs/shared_studio_runtime.md`): never stop/restart other processes
  or Docker containers, never eject LM Studio models (`qwen3-vl-8b-heb-v20a-step1800` is loaded and used by the
  consensus pipeline), check `df -h` / `~/.lmstudio/bin/lms ps` / `docker ps` before long jobs.
* Hub throughput from the Studio is ~3.7 MB/s; NAS arrow reads over SMB are very slow (a 60k-row column
  scan took >15 min) — prefer the HTTP-range approach in `SCRATCH/vlm_hub_row_check.py`.
* `datasets` locally is 4.4.1 / pyarrow 22; Colab's exact pair (4.3.0 / 23.0.1) lives in `SCRATCH/venv23`.
* Findings are also in memory: `~/.claude/projects/-Users-isaac-Documents-GitHub-historical-document-analysis/memory/project_v21_run_anomaly.md`.

## 8. Update 2026-09-11 ~13:45 — diag 2b result (H1 eliminated; fault is on the trainer's batch path)

In-kernel output (`SCRATCH/colab_diag2b_v21.py`, first five micro-batches):
```
num_items_in_batch=4083 | per-micro answer tokens=[322, 429, 1461, 517, 334, 415, 512, 93] sum=4083 | accepts_loss_kwargs=True
micro[0] tok=  322 eval_mean=3.853 | step(N_total)=0.3039 expected=0.3038 | step(None)=0.4816 expected=0.4816
micro[1] tok=  429 eval_mean=2.729 | step(N_total)=0.2871 expected=0.2867 | step(None)=0.3415 expected=0.3411
micro[2] tok= 1461 eval_mean=2.389 | step(N_total)=0.8547 expected=0.8547 | step(None)=0.2986 expected=0.2986
micro[3] tok=  517 eval_mean=2.568 | step(N_total)=0.3252 expected=0.3252 | step(None)=0.3210 expected=0.3211
micro[4] tok=  334 eval_mean=2.873 | step(N_total)=0.2349 expected=0.2350 | step(None)=0.3590 expected=0.3591
```
* Normalisation is correct: the count equals the token sum, `step(N_total)` == `expected`, `step(None)` == `eval/8`.
  **H1 is dead.** (The "does not accept num_items_in_batch" warning is from the explicit `None` call.)
* The SAME eight rows as diag 1 (identical token counts) now score **3.85 / 2.73 / 2.39 / 2.57 / 2.87 in eval
  mode** versus 1.09 / 2.66 / 0.03 / 0.03 / 0.83 in diag 1. Only difference: these batches came through
  `trainer.get_train_dataloader()` → `trainer.get_batch_samples()` (accelerate `send_to_device` + Unsloth's
  `torch._dynamo.mark_static/mark_dynamic` on `input_ids`/`attention_mask`), diag 1 called `collator(rows)`
  directly. A memorised Talmud page goes 0.03 → 2.39. **The loop's batches are unreadable to the model even
  in eval mode; that is the 3.0, and every v21 checkpoint was trained on it** (LoRA drifting toward
  image-blind transcription = the eval rise).
* Next split (`diag 3`, given to the user): dataloader batch unmarked → eval; apply the two marks → eval;
  `.clone()` copy (drops marks) → eval; `torch._dynamo.reset()` → eval; `torch._dynamo.config.disable=True`
  (eager) → eval; plus `dl.collate_fn is collator`. Expected if H2: unmarked ≈0.5, marked ≈3, cloned ≈0.5.
  Fix for H2: `os.environ["UNSLOTH_COMPILE_DISABLE"]="1"` before `import unsloth` (cell 1) — or neutralise the
  marks (`unsloth_zoo.loss_utils.mark_dynamic = lambda *a, **k: None`) — then a FRESH v21 from the v2.0a warm
  start. If unmarked is already ≈3 the corruption is in the dataloader/device move (H3) → inspect
  `accelerate` `DataLoaderShard.send_to_device(non_blocking)` + `dataloader_pin_memory`.
* Why v20a (same code, map-style) did not show this is still unexplained; do not assume the marks are the
  whole story until diag 3 says so.

**Studio note (13:47):** an `lms`-spawned duplicate `LM Studio` app instance (responsible process: ChatGPT)
aborted at launch (`HIServices _RegisterApplication`, SIGABRT). The serving instance (PID 2909, up since
Sep 4) was unaffected. Do not run `lms server start` / `lms load` / `open -a "LM Studio"` on the Studio.

## 9. RESOLVED 2026-09-11 (Astra, `docs/v21_independent_dispatch_findings.md`) — Accelerate's dispatcher truncated the image

With an **iterable** train set Accelerate defaults to `DataLoaderDispatcher` (single device included), which
slices every batch tensor along dim 0 to the inferred batch size (1). Qwen3-VL packs image patches along
dim 0, so `pixel_values` `[25728, 1536]` reached the model as `[1, 1536]` while `image_grid_thw`,
`input_ids` and `labels` survived; the vision tower broadcasts the single patch over the grid and returns
correctly-shaped garbage. Confirmed live in the kernel (`pixel_values: torch.Size([1, 1536])`, grid
`[[1, 192, 134]]`, 25,728 expected) and reproduced on CPU by
`src/finetuning/qwen_hebrew/probe_stream_dispatch.py`. Map-style eval and the map-style v20a run went
through `DataLoaderShard` and were intact — which is why every observation in §2–§8 lined up the way it did
(uniform image-blind loss ≈3.0 from step 10, direct-collate probes fine, dispatcher batches at 3.85 even
cloned/eager, normalisation correct, unchanged step time and memory). H1–H3 are retired.

**Fix applied to `src/finetuning/qwen_hebrew/colab/genizah_v21b_dispatchfix.ipynb` (renamed from `genizah_v21.ipynb`; tests in
`tests/test_colab_notebook_v21.py`):** `accelerator_config={"dispatch_batches": False}` in `make_sft_config`;
a gate before `trainer.train()` that takes batches from the PREPARED train dataloader and asserts
`type` is not a Dispatcher, `pixel_values.shape[0] == image_grid_thw.prod(-1).sum()` for two batches, the
prepared-batch loss equals the direct-collate loss of the same row (±5 %), and the eval batch is intact;
fresh destinations `isaacmg/qwen3-vl-8b-hebrew-v21b-ckpt` / `outputs_v21b` / W&B `genizah_v21b` so the
auto-resume cannot load the image-blind `last-checkpoint/` of `v21-ckpt` (kept as evidence). Warm start
unchanged (v2.0a-1800), fresh optimizer/schedule.
