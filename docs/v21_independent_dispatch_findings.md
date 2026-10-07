# V21 independent investigation: streamed image patches truncated by Accelerate

2026-09-11. Investigation requested as a second look alongside Fable.

**Finding:** Accelerate's default iterable-dataset dispatcher can reduce Qwen3-VL's
packed `pixel_values` to its first patch on a single device with batch size 1.
Qwen then broadcasts that patch over the full positional grid, silently producing
valid-shaped vision features from a corrupted image. This is reproduced locally
and **confirmed in the live V21 Colab kernel**:

```text
DataLoaderDispatcher
pixel_values: torch.Size([1, 1536])
grid: [[1, 192, 134]]
expected patch rows: 25728
```

The loader retains one of 25,728 required patches. This establishes the broken
training input path. A corrected Colab forward/scout run remains to verify recovery.

The original handoff is `docs/v21_streamed_run_anomaly_handoff.md`. This note
supersedes its ranking of hypotheses; its original observations are preserved.

## Evidence from the user's second diagnostic

The user supplied the diagnostic output during this investigation:

```text
num_items_in_batch=4083
per-micro tokens=[322,429,1461,517,334,415,512,93]
accepts_loss_kwargs=True
SUM step(N_total)=2.779
SUM step(None)=3.017
eval token-weighted=2.779
8×example-mean=24.132
```

Every microbatch's `step(N_total)` matches `eval_mean * tokens / 4083`;
`step(None)` matches `eval_mean / 8`, within rounding/numerical differences.
**H1 (missing token denominator) is contradicted for these batches.** The warning
about unsupported `num_items_in_batch` occurs when the diagnostic deliberately
passes `None`; it is not evidence that the counted path ignored its denominator.

The dataloader batches already have high eval losses (2.389–3.853). The first
diagnostic's direct-collator batches had much lower losses, including a Talmud row
at 0.032 versus 2.389 in the second diagnostic. The identical token-count sequence
supports comparing the rows, but token counts alone do not establish tensor
identity. Compare the actual tensors to close that gap.

## Mechanism and source evidence

1. The V21 notebook supplies an iterable training dataset and leaves
   `accelerator_config.dispatch_batches` unspecified. Its evaluation dataset is
   map-style. V20a training was also map-style.
2. Accelerate selects dispatching by default for iterable datasets when moving
   batches to a device. The dispatcher determines batch size from a tensor and
   slices the batch tensors along dimension zero—even with one process. See
   [Accelerate v1.14.0 data_loader.py](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/data_loader.py).
3. With `input_ids.shape[0] == 1`, the slice is `[0:1]`. Qwen's `pixel_values`
   stores patches along dimension zero, so this removes all but the first patch.
   `image_grid_thw`, input IDs and labels can remain unchanged.
4. Qwen's patch embed accepts one patch. The vision forward adds the full grid's
   positional embeddings to it; tensor broadcasting expands the single patch
   across the grid. Later vision operations therefore still receive the expected
   sequence length. See
   [Transformers v4.57.6 Qwen3-VL](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py),
   `Qwen3VLVisionPatchEmbed.forward` and `Qwen3VLVisionModel.forward`.

This accounts for good direct-collator probes, high dataloader losses, and
map-style eval remaining readable while training damages transcription. Since
the vision sequence is expanded before its transformer blocks, nearly unchanged
training time and memory do not rule it out. No torch.compile defect is needed
to reproduce the corruption. The reproduction and the live tensor shapes support
this diagnosis; the exact share of the observed metric regression attributable
to this defect still requires a corrected run.

## Reproduction and validation

`src/finetuning/qwen_hebrew/probe_stream_dispatch.py` uses tiny deterministic image
tensors and a randomly initialized, tiny Qwen3-VL vision model on CPU. It downloads
no models and performs no training or service requests.

```bash
.venv/bin/python -m src.finetuning.qwen_hebrew.probe_stream_dispatch
```

Also tested with the downloaded **v1.14.0 data_loader.py** using:

```bash
.venv/bin/python -m src.finetuning.qwen_hebrew.probe_stream_dispatch \
  --loader-source /private/tmp/v21_independent_investigation/data_loader_1_14.py
```

The latter executes the exact v1.14.0 dispatcher module under the local
Accelerate 1.12.0 package's supporting utilities. It is not a full replica of
Colab's environment. Local Transformers is exactly 4.57.6; Unsloth and CUDA are
not used in this CPU reproduction.

Verified results:

| Check | Default dispatch | `dispatch_batches=False` |
|---|---|---|
| Prepared loader | `DataLoaderDispatcher` | `DataLoaderShard` |
| Image patch rows (16 expected) | 1 | 16 |
| Input IDs and image grid | Preserved | Preserved |
| Every input tensor preserved | No | Yes |
| Vision output shape | `(4, 32)` | `(4, 32)` |

The damaged vision output matches feeding a complete grid of repeated copies of
the first patch. It differs from the intact image's output. Both assertions pass.

## Minimal live confirmation

Reuse `bs` and `dl` from diagnostic 2b; no forward pass is needed:

```python
b = bs[0]
print(type(dl).__name__)
print('pixel_values:', b['pixel_values'].shape)
print('grid:', b['image_grid_thw'].tolist())
print('expected patch rows:', b['image_grid_thw'].prod(dim=-1).sum().item())
```

The user ran this check and returned the output recorded above, confirming
corruption in the live training path.

## Proposed fix and restart gate

Add this argument to V21's `make_sft_config(...)` call:

```python
accelerator_config={"dispatch_batches": False},
```

Keep batch size 1 and gradient accumulation 8. Create a new Trainer so the new
Accelerator configuration takes effect. Changing the configuration on an already
constructed dataloader will not repair that loader.

Before training, inspect batches from **the prepared training dataloader**, not
only from the collator. For each image batch require:

```python
assert b['pixel_values'].shape[0] == int(
    b['image_grid_thw'].prod(dim=-1).sum()
)
```

This is the pre-merge patch count; do not divide by the spatial merge factor.
For the same fixed rows, compare all tensors before and after preparation, then
repeat diagnostic 2b. The accumulated loss should match intact-image forwards.
Check a same-checkpoint, same-eval-set baseline before a short training scout.

Restart from the pinned V20a-1800 weights with a fresh optimizer/schedule and a
fresh output/checkpoint destination. The current notebook automatically resumes
from `last-checkpoint/`; leaving that destination unchanged can reload a damaged
V21 checkpoint despite a correct warm-start cell. Do not overwrite the existing
run's evidence. This investigation did not change or restart the training notebook.

Increasing batch size to 8 does not fix this dispatcher incompatibility: it
still slices patch tensors as examples and may instead trigger a shape error.
Disabling compilation or suppressing compile errors does not repair a tensor
already truncated by the dataloader.
