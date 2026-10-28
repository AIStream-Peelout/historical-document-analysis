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
| `convert_lowmem.py` | the stock `mlx_lm` conversion with a bounded Metal working set (≈5 GiB for a 27B model) |
| `verify_mlx_dir.py` | completeness check of a model directory; writes or compares a size/sha256 manifest (stdlib only) |
| `kit_record.py` | writes `<output>.CONVERTED.json`; `--check-pins` compares the installed libraries with `requirements.txt` |

Placeholders below: `<NAS>` is the mount point of the shared storage, `<model>` the checkpoint's name.

## 1. Requirements

* Apple Silicon Mac, macOS 14 or later. Python 3.12 or newer, 3.13 like the reference machine if possible: the pinned
  numpy needs 3.12, and macOS's own `/usr/bin/python3` (3.9) will not work.
* Memory: the conversion needs about 5 GiB on the GPU plus a few GiB for the process. If the Mac also runs other
  services, keep their usual headroom; `run_convert.sh` refuses to start below 12 GiB free (`--min-free-gib`).
* The GPU must be otherwise idle for the whole conversion. Metal kills a long command buffer when another process
  monopolises the GPU, and a killed buffer poisons the conversion's Metal context. `run_convert.sh` refuses to start
  while LM Studio or a model server is running; do not start one until it finishes.
* The shared storage mounted (SMB). The conversion reads the whole bf16 checkpoint (≈56 GB for a 27B model), writes
  ≈27 GB back and reads those 27 GB again for the manifest hashes (`--no-hash` skips that). A wired connection
  matters: at 100 MB/s the reading alone takes about ten minutes.
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
exercises the conversion machinery on a small llama model, not the exact architecture of the real checkpoint.

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

Timings for a 27B model: about 15–20 minutes of GPU work on the reference machine's class of GPU with the GPU idle,
plus the storage transfers (≈56 GB read, ≈27 GB written, ≈27 GB read again for the hashes). Peak Metal memory about 5 GiB.

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
| `GPU Timeout Error` / `kIOGPUCommandBufferCallbackErrorTimeout` / `command buffer execution failed` | another process used the GPU during the conversion (or the Mac slept); stop it, run again |
| conversion very slow, low GPU use | reading over Wi-Fi; use a wired connection or copy the bf16 checkpoint locally first |
| `verify_mlx_dir.py` reports a shard shorter than its header implies | truncated write (network drop, disk full); run again |
| `selftest.py --device gpu` fails but `--device cpu` passes | Metal problem: macOS or `mlx-metal` version; check `pip list` matches `requirements.txt` |
| different `bits per weight` than the reference log | different `mlx-lm` version or quantisation flags; the figure for this recipe is logged by the reference conversion |

Exit codes: `run_convert.sh`, `selftest.py`, `verify_mlx_dir.py` and `kit_record.py --check-pins` return 0 on success
and 1 on failure; `gpu_quant_identity.py` returns 0 identical, 10 differences found, 11 tensor missing.
