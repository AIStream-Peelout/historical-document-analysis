# MLX 8-bit conversion kit for a second Apple Silicon Mac

Converts a merged bf16 checkpoint (Hugging Face safetensors layout) to an 8-bit MLX model with exactly the recipe
the reference machine uses (`mlx_lm convert -q --q-bits 8 --q-group-size 64`), on a different Mac, so the reference
machine's GPU stays free for serving. The kit also answers, before any conversion, whether this Mac quantises
bit-identically to the reference machine, and it leaves a manifest and a record next to the output so the receiving
side can check the copy.

Copy the whole directory to the other Mac (the scripts import each other by directory). No credentials are needed:
everything is read from and written to shared storage.

| file | role |
|---|---|
| `requirements.txt` | pinned `mlx`, `mlx-lm`, `transformers`, `tokenizers`, … (the versions the reference conversion used) |
| `selftest.py` | one-minute end-to-end run on a tiny random model; proves the environment with `--device cpu` and `--device gpu` |
| `gpu_quant_identity.py` | quantises reference tensors here and compares them bit for bit with the reference MLX copy |
| `run_convert.sh` | the conversion: safety checks, lock, `caffeinate`, staging directory, verification, manifest, record |
| `convert_lowmem.py` | the stock `mlx_lm` conversion, each weight read before its kernel runs so a network-volume source cannot trip Metal's GPU watchdog (`--quantize-order`, section 4) |
| `memory_floor.py` | estimates the free memory the eager order needs from the source's shard headers (stdlib only; `run_convert.sh` calls it) |
| `verify_mlx_dir.py` | completeness check of a model directory; writes or compares a size/sha256 manifest (stdlib only) |
| `kit_record.py` | writes `<output>.CONVERTED.json`; `--check-pins` compares the installed libraries with `requirements.txt` |

Placeholders below: `<NAS>` is the mount point of the shared storage, `<model>` the checkpoint's name.

## 1. Requirements

* Apple Silicon Mac, macOS 14 or later. Python 3.12 or newer, 3.13 like the reference machine if possible: the pinned
  numpy needs 3.12, and macOS's own `/usr/bin/python3` (3.9) will not work.
* Memory, in the default eager quantisation order (section 4): free memory ≥ the packed model (the bf16 checkpoint ×
  (bits + 32 / group size) / 16, about 0.53 of it at 8 bits) + the largest bf16 tensor (usually the token embedding or
  the output head, a few GB) + a few GB for the process. For a dense 27B model at 8 bits the packed model is ≈26.6 GiB
  (28.6 GB) and the peak ≈29 GiB (31 GB), so about 33 GiB must be free, which a 48 GB Mac has only with little else
  running. `--memory-limit-gib` does not cap this: MLX's limit is a guideline (above it the scheduler waits for
  in-flight work, then the allocation proceeds). `run_convert.sh` computes this floor from the source's shard headers
  (`memory_floor.py`: packed estimate + largest tensor + 4 GiB) and refuses to start below it; in the lazy order its
  floor is 12 GiB; `--min-free-gib` overrides either. If the Mac also runs other services, keep their usual headroom
  on top. The lazy order needs about 5 GiB, but only works from a fast local SSD. To see the estimate beforehand:
  `python3 memory_floor.py <NAS>/<model>-bf16 --json`.
* The GPU must be otherwise idle for the whole conversion. Metal's watchdog kills a command buffer that takes too
  long, and a killed buffer poisons the conversion's Metal context. Two things make one take too long: another
  process monopolising the GPU, and (in the lazy order only) a kernel waiting on a slow read from the source, which
  the default eager order prevents (section 4). `run_convert.sh` refuses to start while LM Studio or a model server
  is running; do not start one until it finishes.
* The shared storage mounted (SMB). The conversion reads the whole bf16 checkpoint (≈56 GB for a 27B model), writes
  ≈29 GB back and reads those 29 GB again for the manifest hashes (`--no-hash` skips that). A wired connection
  matters: at 100 MB/s the reading alone takes about ten minutes. Reading the source over the network is why the
  default quantisation order is eager (section 4).
* Sleep: `caffeinate` prevents idle sleep only. Keep the lid open, or use clamshell mode with power and an external
  display. Run long steps inside `screen` or `tmux` so a closed terminal window does not end them.

## 2. Install

```bash
cd <kit directory>
python3.13 -m venv ~/.venv-mlx-kit          # e.g. after: brew install python@3.13
~/.venv-mlx-kit/bin/python -c 'import sys; assert sys.version_info >= (3, 12), sys.version'
~/.venv-mlx-kit/bin/pip install -r requirements.txt
```

Then the self-test, first on the CPU (checks Python and the packages), then on the GPU (checks Metal):

```bash
~/.venv-mlx-kit/bin/python selftest.py --device cpu
~/.venv-mlx-kit/bin/python selftest.py --device gpu
```

Both print a JSON report ending in `"ok": true`. Nothing outside a temporary directory is touched. The self-test
converts the tiny model in the eager order (`--quantize-order lazy` makes the lazy order the recorded one), converts it
again in the other order and compares the weight files byte for byte (`orders_bit_identical`), and checks from the
eager conversion's log that every quantised weight was read first and released (`eager_hooks_ran`);
`bits_per_weight` is 8.511 for the tiny model. With `--device gpu` it also checks that the CPU and the GPU compute the
model fix-ups the eager order moves to the CPU (a bf16 `x + 1.0` norm shift and a float16 → bf16 cast, over every
16-bit pattern) bit for bit (`fixups_cpu_gpu_identical`). Run the GPU self-test before the first real conversion on a
machine: it is the check of this implementation's eager order on the GPU. It exercises the conversion machinery on a
small llama model, not the exact architecture of the real checkpoint.

`run_convert.sh` uses `$PYTHON` when set, else `~/.venv-mlx-kit/bin/python` when it exists, else `python3`, and it
refuses an interpreter whose libraries do not match `requirements.txt` exactly (those versions write the tokenizer
files, which the receiving side compares byte for byte with the reference copy's).

## 3. Identity test (once per machine)

```bash
~/.venv-mlx-kit/bin/python gpu_quant_identity.py \
    --shards-dir <NAS>/<reference bf16 shards> \
    --mlx-dir   <NAS>/<reference MLX 8-bit copy> \
    --out quant_identity.json
```

It quantises the token embedding and the output head (the two largest tensors) from the reference bf16 shards on
this Mac's GPU and compares packed weights, scales and biases with the reference MLX copy. It reads about 8 GB over
the network (the two bf16 tensors and the reference's two quantised tensors). `--shards-dir` may be the whole bf16
checkpoint or just the shards that hold those two tensors; `--pair HF_KEY=MLX_BASE` chooses other tensors.

* `IDENTICAL` (exit 0): conversions made here are bit-paired with the reference machine's. A model converted here can
  be compared with one converted there as if one machine had made both.
* `DIFFERENCES_FOUND` (exit 10): still a valid 8-bit model, but not bit-identical. Expect one-ulp differences in a
  fraction of a percent of the scales (the reference machine's CPU against its own GPU: 0.46 % of the 8-bit weights,
  1.3 % of the packed 32-bit words, 0.17 % of the scales, biases identical). Use such a conversion for serving or
  exploratory evaluation,
  not for a paired comparison that attributes small score differences to the weights.
* Exit 11: a requested tensor was not found (no report is written). Exit 1 or 2 is a crash or an argument error.

Keep `quant_identity.json`; `run_convert.sh --identity-json` copies its verdict into the conversion record.

## 4. Convert

```bash
./run_convert.sh \
    --hf-path  <NAS>/<model>-bf16 \
    --mlx-path <NAS>/<model>-MLX-8bit \
    --identity-json quant_identity.json
```

The runner computes the eager order's free-memory floor from the source (section 1) and prints it in the log's first
line; `--min-free-gib` overrides it.

`--mlx-path` must be exactly the name the serving machine's pipeline expects, in the place it expects it.

What happens, in order: the source is checked (config, shard index, generation config, chat template); the output
must not exist, and a `<model>-MLX-8bit.partial` directory, which another converter may be writing, is never
deleted; a lock directory `<model>-MLX-8bit.lock` is taken with an atomic `mkdir` on the share (another run of the
kit, on any Mac, refuses while it exists) and removed when the script ends; the interpreter is checked against
`requirements.txt`; GPU users and free memory are checked; `caffeinate -dims` keeps the Mac from idle sleep;
`convert_lowmem.py` writes into `<model>-MLX-8bit.kit-partial`; `verify_mlx_dir.py` checks every shard (header,
length, tensor set) and writes `<model>-MLX-8bit.MANIFEST.json` (sizes and sha256); `<model>-MLX-8bit.CONVERTED.json`
is written; only then is the staging directory renamed to `<model>-MLX-8bit`, with a rename that fails instead of
nesting if the final directory appeared meanwhile. The log is `<model>-MLX-8bit.convert.log`.

`--quantize-order eager|lazy` is passed to `convert_lowmem.py` (default eager; see below).

Timings for a 27B model: an eager conversion with the source on a network volume took about 25 minutes on the
reference machine's class of GPU with the GPU idle (reading, quantising, writing); the hash pass (≈29 GB read again)
comes on top. Peak memory: the packed model plus one bf16 tensor in the eager order (≈29 GiB, 31 GB, for a dense 27B
model), about 5 GiB in the lazy order.

### Quantisation order (`--quantize-order`)

In the stock `mlx_lm` order (*lazy*) each quantised tensor is evaluated only when its shard is written, as one graph:
the read of the bf16 tensor from the source file (on the CPU), then the quantisation kernel (on the GPU). The Metal
command buffer that runs the kernel then waits for the read. With the source on a network volume (SMB), a multi-GB
tensor can take longer to arrive than Metal's GPU watchdog allows, and the conversion dies a few minutes in with
`[METAL] Command buffer execution failed: Caused GPU Timeout Error`, even with the GPU otherwise idle. The same lazy
conversion works from a fast local SSD.

`eager` (the default) reads each weight into memory on the CPU first and only then runs its quantisation kernel on
data already in memory, so no command buffer waits on the network. Loading, the dtype cast and the model's own weight
fix-ups run on the CPU; each source weight is released as soon as its quantised copy exists, for every layer type that
defines a `to_quantized` method (linear and embedding layers, and `mlx_lm`'s mixture-of-experts `SwitchLinear` and
similar). The log's `[lowmem] eager:` line counts quantised and released weights and is followed by a warning when
they differ.

What stays the same: the quantisation kernel, the device it runs on and its arguments. For a source already stored in
the config dtype (bf16 here) the kernel's input is the bytes read from the file, so the quantised tensors are
bit-identical to the lazy order's. What moves: the fix-ups before the kernel (renames, transposes, and arithmetic such
as a norm weight shifted by 1.0, which is not quantised) run on the CPU instead of the GPU; the GPU self-test checks
that both compute them identically. Evidence: a prototype of this order (same wrapped quantisation, default device
left on the GPU, no weight release) converted a 27B model on the GPU, and the tensors it shares verbatim with an
earlier checkpoint matched that checkpoint's lazily converted copy bit for bit; this implementation is compared with
the lazy order byte for byte by `selftest.py` on the device it runs on, which is why the GPU self-test comes before the
first real conversion. The cost is memory: the packed model is held until it is written (sizing rule in section 1).

`--quantize-order lazy` keeps the stock order and its ≈5 GiB footprint. Use it only when the source is on a fast
local SSD, for example on a Mac with too little free memory for eager: copy the bf16 checkpoint to the internal disk
first, then run `run_convert.sh --hf-path <local copy> … --quantize-order lazy`.

A conversion is not resumable. After a crash or a network drop, read the log, fix the cause, and run again: the script
deletes its own `.kit-partial` directory itself. Never create the final directory by hand. If the script stopped after
the verification (for example the record step failed), the verified result is still in `.kit-partial`: write the
record with `kit_record.py` and rename the directory yourself rather than converting again.

## 5. Hand the result to the serving machine

On the receiving side (no MLX environment needed; `verify_mlx_dir.py` runs on any Python 3):

```bash
python3 verify_mlx_dir.py <NAS>/<model>-MLX-8bit --compare <NAS>/<model>-MLX-8bit.MANIFEST.json
```

Finder's `.DS_Store` and `._*` files are ignored, so a folder opened in Finder still verifies. Then install it for
LM Studio by copying the directory under its models folder as `<org>/<model>-MLX-8bit` (a local copy loads faster
than a network mount). LM Studio derives the model key from the folder name and strips quantisation tokens from it,
so read the key back from `GET /v1/models` rather than assuming it. A serving pipeline that already waits for the MLX
copy at the agreed storage path picks it up by itself, and waits while the kit's lock or staging directory exists.

## 6. Troubleshooting

| symptom | cause and fix |
|---|---|
| `… .partial exists: another converter … may be writing it` | the serving machine's own pipeline stages there; wait for it, or remove that directory only when that run is dead |
| `… .lock exists (<owner>)` | another kit run holds the lock (the owner file names the Mac and pid); remove the lock only when that run is dead |
| `does not match requirements.txt` | the interpreter has drifted versions; recreate the venv from `requirements.txt` or pass `--python` |
| `Cannot save to the path … as it already exists` | stale `.kit-partial`; the runner removes it on the next run, never create the final directory by hand |
| `GPU Timeout Error` / `kIOGPUCommandBufferCallbackErrorTimeout` / `command buffer execution failed` | with `--quantize-order lazy` and the source on a network volume: the read was awaited inside the GPU command buffer; use the default eager order, or a local copy of the source. Otherwise another process used the GPU during the conversion (or the Mac slept): stop it, run again |
| `only … GiB free …; need … GiB (eager order estimate …)` | the eager order holds the packed model (section 1): close other applications, or copy the source to a local SSD and use `--quantize-order lazy` |
| heavy swapping, or the process killed for memory, during an eager conversion | too little free memory for the packed model (sizing rule in section 1; was the floor overridden with `--min-free-gib`?): close other applications, or copy the source to a local SSD and use `--quantize-order lazy` |
| `[lowmem] warning: … quantised eagerly but … released` | a layer type quantised outside the kit's hooks; the result is still correct, but peak memory can exceed the sizing rule: watch memory, or use `--quantize-order lazy` from a local SSD copy |
| conversion very slow, low GPU use | reading over Wi-Fi; use a wired connection or copy the bf16 checkpoint locally first |
| `verify_mlx_dir.py` reports a shard shorter than its header implies | truncated write (network drop, disk full); run again |
| `selftest.py --device gpu` fails but `--device cpu` passes | Metal problem: macOS or `mlx-metal` version; check `pip list` matches `requirements.txt` |
| different `bits per weight` than the reference log | different `mlx-lm` version or quantisation flags; the figure for this recipe is logged by the reference conversion |

Exit codes: `run_convert.sh`, `selftest.py`, `verify_mlx_dir.py` and `kit_record.py --check-pins` return 0 on success
and 1 on failure; `gpu_quant_identity.py` returns 0 identical, 10 differences found, 11 tensor missing.
