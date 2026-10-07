# Post-mortem — v21 trained image-blind for a night, then a day of circular debugging

Date: 2026-09-11. Author: Fable (Claude), on the user's request. Resolution by an external audit (Astra,
`docs/v21_independent_dispatch_findings.md`). This document is about the *process failure*, not the bug.

## What happened

* 03:52 UTC — v21 launched on Colab A100: the first run whose training mixture is **streamed**
  (`datasets` IterableDatasets, because `genizah_ktiv_v3` is 77 GB). Warm start from v2.0a-1800, 2000 steps.
* From the first log (step 10, ~04:05 UTC) train loss was **3.2** — v2.0a's was 0.51 at the same step on the
  same stack. Nobody reacted. The eval curve then rose monotonically (0.683 → 0.706 → 0.720 → 0.753 …).
* ~12:00 local — the user asked "why isn't eval descending". I spent most of the day eliminating hypotheses
  (data bytes, hub row alignment, pyarrow, library versions, mixture composition, loss normalisation,
  torch.compile) with many tool calls, LM Studio requests, a scratch venv, source reads of three libraries, and
  five round-trips of diagnostic cells through the user's kernel. Each eliminated something real; none found
  the cause.
* The user brought in a second model. It looked at the **actual objects on the failing path** — the prepared
  train dataloader's type and one batch's tensor shapes — read Accelerate's `data_loader.py`, reproduced the
  bug on CPU, and had the answer: Accelerate's default `DataLoaderDispatcher` (used for *any* iterable
  dataset, single device included) slices every batch tensor along dim 0 to the batch size, so Qwen3-VL's
  packed `pixel_values` `[25728, 1536]` reached the model as `[1, 1536]`. The model trained image-blind all
  night; every v21 checkpoint is scrap.

Cost: ~8 h of A100, ~700 steps of a 2000-step run, most of a working day, and a large token bill.

## The signals that were available and ignored

| When | Signal | What it meant | What I did |
|---|---|---|---|
| step 10 | train loss 3.2 vs 0.51 for the reference run | the model cannot see the image | nothing — I had a watcher on eval only |
| step 200 | eval 0.706 > warm-start checkpoint's 0.673 for the 2nd consecutive eval | a warm start from the best checkpoint should not lose 0.03 | rationalised as "warm-start settling, v2.0a also rose early" |
| diag 1 vs diag 2b | same 8 rows: 0.03–1.1 collated directly, 2.4–3.9 from the trainer's dataloader | the *input tensors* differ between the two paths | tested two more loss-level hypotheses instead of diffing the tensors |
| all along | the only path component I had not read was Accelerate's dataloader prep | my belief "single-process prepare is a no-op" was unverified | stated it as fact in my own reasoning, twice |

## Root causes

1. **Gates tested components, not the path.** The notebook had data pins, grounding hygiene, a collator
   smoke test and a gradient-flow gate — all on rows and the collator. Nothing examined a batch *after*
   `trainer.get_train_dataloader()` / `accelerator.prepare`, which is the only thing the model ever sees.
   The grad-gate cell even computed a loss on a streamed row and never printed it — and it would not have
   caught this anyway, because it bypassed the dataloader.
2. **A load-bearing belief went unverified.** I read Unsloth, TRL and transformers sources to rule things
   out, and skipped Accelerate because I "knew" it did nothing on one GPU. That belief was wrong, and it sat
   exactly on the one component that differs between map-style and iterable datasets — the thing the user
   asked about directly ("anything that could've messed stuff when we went to streaming?"). I answered from
   a mental model instead of from the source.
3. **I misapplied a user instruction about train loss.** "I wouldn't care about training loss … it is the
   eval loss descending that matters" was about the *trend* being noisy under a mixed-task curriculum. I
   turned it into "don't look at train loss at all" and did not compare the *level* to the previous run.
   A 6× jump at step 10 on an identical stack is not noise; it is the loudest alarm the run could give.
4. **No stop rule for a warm-started run.** When a run starts from the best checkpoint, eval above the
   warm-start value for two consecutive evals should halt the run, not prompt a story about settling.
5. **Hypothesis-first instead of observation-first.** Once diag 1 and diag 2b disagreed on identical rows,
   the discrepancy was in the inputs by construction. The right next step was a five-line diff of the two
   batch dicts (keys, shapes, dtypes, then values). I instead designed two more experiments around losses
   (normalisation, then compile/marks), each costing a kernel round-trip.
6. **Thoroughness without a budget.** Hub row alignment via LM Studio, a pyarrow-23 venv, an interleave
   simulation, W&B requirement diffs — each was a reasonable check in isolation, but several tested branches
   the evidence had already made improbable (eval@100 = 0.683 on val rows from the same builder had
   effectively cleared the data hours earlier). No per-hypothesis budget, no point at which I switched from
   "test the next hypothesis" to "instrument one batch end to end".
7. **Serial diagnostics through the user's kernel.** Five cells, one hypothesis each, each needing the user
   to paste output. One cell that dumped everything about one batch on both paths — dataloader type, every
   tensor's shape and dtype, grid product vs patch rows, loss on both — would have ended it in one trip.
8. **Two changes in one overnight launch.** The data path changed (map-style → streamed) *and* a 2000-step
   run was launched, with no 20-step smoke comparing train loss against the map-style run at the same steps.

## Why the external audit was fast

Fresh priors, and it started from the one hard fact — the loop's batches are bad and directly collated
batches are good — then inspected the object on that path rather than theorising about it: `type(dl)`,
`pixel_values.shape` against `image_grid_thw.prod().sum()`. It read the Accelerate source for the exact
version and wrote a CPU reproduction. It also benefited from the fault already being localised to "between
`get_train_dataloader()` and the model"; that localisation was real work, but the final step was cheap and I
should have taken it hours earlier.

## What is already changed

* `genizah_v21b_dispatchfix.ipynb`: `accelerator_config={"dispatch_batches": False}`; a gate before
  `trainer.train()` on the **prepared** dataloader (no Dispatcher; `pixel_values.shape[0] ==
  image_grid_thw.prod(-1).sum()` on two batches; prepared-batch loss == direct-collate loss ±5 %; eval batch
  intact); fresh destinations (`v21b-ckpt`, `outputs_v21b`, W&B `genizah_v21b`); distinct file name.
  `tests/test_colab_notebook_v21.py` pins all of it (11 tests).
* Memory rule saved: any IterableDataset + HF Trainer + a VLM whose batch tensors are not batch-major needs
  `dispatch_batches=False` and a prepared-path shape gate; the collator alone proves nothing.

## What changes from here (rules, in priority order)

0. **The change is the prior (the user's rule).** When a run breaks right after a known change — here
   map-style → streamed — enumerate every code path that *branches on that change* (Trainer, TRL,
   Accelerate all branch on `IterableDataset`), not just the artefacts it produces, and check those first.
   I verified the streamed *bytes* and called streaming cleared; the dataset *type* was the change.
   Corollary: a structural change that "didn't fail" proves nothing — this failure was silent by
   construction (a truncated tensor with a valid shape, broadcast into a valid output). Only an explicit
   invariant counts as evidence.
1. **Gate the real path, always.** Every training notebook checks one batch from the prepared train and
   eval dataloaders before step 1: dataloader class, tensor shapes vs grid, prepared-vs-direct loss. Turn the
   v21b gate into a shared helper so it cannot be forgotten.
2. **Alarm on train-loss level, not just eval trend.** The run watcher compares the first logged train loss
   against the reference run's (>2× → page immediately) and halts on a warm-started run whose eval exceeds
   the warm-start checkpoint for two consecutive evals. Both would have fired before 05:00.
3. **Smoke A/B before any data-path change.** 20 steps on the new path vs 20 on the old, same seed, compare
   train loss; only then the overnight launch.
4. **Debugging protocol.** (a) Write down every component on the path and every assumption about it, marked
   verified/unverified; verify the unverified ones against source before forming hypotheses. (b) When two
   paths disagree on the same input, diff the inputs (shape → dtype → values) before anything else. (c) One
   instrumented cell that captures everything, not one hypothesis per round-trip.
5. **Budget and escalate.** Two failed hypotheses or ~90 minutes without a mechanism → stop, write the
   handoff, bring in a second reader. The handoff format worked; the delay in using it did not.
6. **One run, one name.** Distinct notebook file names and checkpoint repos per run; never a name that can
   be confused with a run whose checkpoints must not be resumed.
