# Handoff: weight probing & feature analysis of the Hebrew Qwen3-VL fine-tunes

For a fresh chat whose goal is: probe the fine-tuned weights, identify exactly which
parameters moved, and run visual analysis of intermediate features — fine-tuned vs
vanilla vs vision-trained variants. Everything below was verified against the actual
files on 2026-08-24.

## 1. The one trap to avoid

The merged models in `models/` are **8-bit quantized MLX exports**
(`quantization: {bits: 8, group_size: 64, mode: affine}` — BF16 + packed U32
tensors). **Do not diff them against the base** — quantization noise will swamp the
fine-tune deltas. For exact parameter analysis use:

- **Base (vanilla), full BF16, 17.5 GB:**
  `~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/`
- **LoRA adapters, full F32** (the definitive record of what training changed):
  `~/.cache/huggingface/hub/models--isaacmg--qwen3-vl-8b-hebrew-{GEN}-ckpt/snapshots/<pin>/last-checkpoint/adapter_model.safetensors`

The exact update is `ΔW = B·A × (alpha/r) = B·A × 1.0` — no scaling factor to
remember. The quantized `models/` copies are fine for *running* the models locally
(LM Studio), just not for numerics.

## 2. Adapter inventory (all verified by loading)

| generation | HF repo (isaacmg/…) | snapshot used | vision lora_B | notes |
|---|---|---|---|---|
| v1.5 | qwen3-vl-8b-hebrew-rashi-ckpt | (single) | — unverified | earliest, synthetic-rashi era |
| v1.6 | qwen3-vl-8b-hebrew-v16-ckpt | latest | **0/108 nonzero** | dead vision |
| v1.7 | qwen3-vl-8b-hebrew-v17-ckpt | `068a2cc1…` | **0/108 nonzero** | language-only flagship |
| v1.8a | qwen3-vl-8b-hebrew-v18a-ckpt | `9dbd0070…` (2 snapshots exist) | **0/108 nonzero** | deliberate vision-frozen control |
| v1.8b | qwen3-vl-8b-hebrew-v18b-ckpt | `c80313f8…` (3 snapshots exist) | **108/108 nonzero**, max\|B\|=0.199 | first true vision train |
| v1.9a | qwen3-vl-8b-hebrew-v19a-ckpt | latest | **108/108 nonzero** | current best on Genizah |

Multiple snapshots per repo = intermediate pushes; pin by hash (this session used
`sorted(glob)[-1]`). Private repos: pass `HF1_TOKEN` from `.env`
(see `verify_vision_lora.py` docstring).

**Adapter anatomy** (identical config every generation): PEFT LoRA, r=16, alpha=16,
dropout 0, bias none. 720 tensors = 360 wrapped modules:

- vision (27 blocks × 4): `base_model.model.model.visual.blocks.{N}.attn.qkv|attn.proj|mlp.linear_fc1|mlp.linear_fc2`
- language (36 layers × 7): `base_model.model.model.language_model.layers.{N}.self_attn.{q,k,v,o}_proj|mlp.{gate,up,down}_proj`
- **Never wrapped in any generation:** `visual.merger` (40.1M), the 3
  `visual.deepstack_merger_list` (120.4M — taps at blocks 8/16/24), patch/pos
  embed, `embed_tokens`, `lm_head`, all norms. The vision→language bridge is
  virgin territory.

**Config-vs-reality warning:** the adapter_config target regex *claims* vision in
every generation — only the weights tell the truth. Also the MLX YAMLs in
`src/finetuning/qwen_hebrew/configs/` say alpha=32; the shipped (Unsloth/Colab)
adapters are alpha=16. Don't mix the two provenances.

## 3. Architecture cheat sheet (from the base config, verified)

- total 8,767,123,696 params; LoRA touches 51,346,944 (0.586%)
- language: 36 layers, hidden 4096, 32 q / 8 kv heads, head_dim 128, SwiGLU 12288, vocab 151,936, interleaved MRoPE (sections 24/20/20)
- vision: 27 blocks, hidden 1152, 16 heads, ffn 4304, patch 16 (temporal 2), 2×2 spatial merge → 4096-d visual tokens, deepstack taps [8, 16, 24]

## 4. Results already computed (build on, don't redo)

- `article/analysis_data/archdata.json` — param counts per component, per-layer
  mean ‖B·A‖_F for v1.6/1.7/1.8a/1.8b (`perlayer`), module shapes, verify table
- `article/analysis_data/langgrid.json` — per-layer × per-projection RMS of B·A
  for v16/v17/v18a/v18b/**v19a** (language side; `grid[layer][module]`)
- Known patterns to contrast against: language updates concentrate in
  `gate_proj` (≈2× attention) and at both ends of the stack (mid-layers
  quietest); language deltas are near-identical across generations
  (mean RMS 5.6e-4→6.2e-4 v16→v19a); v1.8b vision updates ~20× smaller than
  language (mean ‖ΔW‖ 0.167 vs 3.79), largest at block 25, smallest block 2
- Rendered figures: `article/assets/A1–A8` + the companion artifact
  ("Where the Gradient Went"); repo tool:
  `src/finetuning/qwen_hebrew/verify_vision_lora.py` (per-tensor zero check,
  `--expect trained|frozen`)

## 5. Stimuli for intermediate-feature analysis

- **Genizah benchmark images:**
  `src/datasets/raw_data/cairo_genizah/evaluations/genizah_test_v1/images/` (131
  verified fragments; catalog + script tags alongside). Staged per-fragment
  copies: `src/datasets/evaluations/genizah_images/<fragment>/`
- **Talmud pages:** `src/datasets/raw_data/cairo_genizah/evaluations/talmud_sample/converted_images/` (65)
- **Every model's raw output per stimulus:**
  `src/datasets/evaluations/transcription_raw_outputs/<doc>/<model>.txt`
- High-signal probe stimuli from this session's analysis:
  - `Cambridge_CUL_T_S_10J4_10` — one fragment where base loops, Claude abstains,
    Sonnet hallucinates, v1.8b reads (all four failure modes)
  - `01_10_page_001` / `01_10b_page_001` — base-model loop-collapse pages
    (flat Kraken OCR preserved in `article/_preserved/`)
  - hypothesis to test at feature level: v1.8a→v1.8b attention/feature changes on
    *degraded manuscript* inputs should dwarf changes on *printed* inputs
    (behaviorally: hallucination 22.1%→9.2% on Genizah, print unchanged)

## 6. Machine & environment rules (shared prod box — hard rules)

- Read `docs/shared_studio_runtime.md` first. api.cairogenizah.ai runs here;
  never stop other containers / LM Studio models / cloudflared.
- Loading base + adapters in torch ≈ 18+ GB RAM: **check `~/.lmstudio/bin/lms ps`
  and free RAM first**; LM Studio often has a 10 GB model resident and may be
  mid-inference. Exhaust alternatives, ask before anything that pressures prod.
- Python: use `.venv/bin/python` (torch, safetensors, matplotlib, PIL, numpy —
  all present; this is what all §4 numbers were computed with). `.venv-mlx` is
  for MLX training only; never install mlx-vlm into `.venv`.
- Running the models for activations: MLX/LM Studio serve the *quantized* merged
  weights; for faithful intermediate features load base+adapter in transformers
  (bf16, `device_map` to MPS) — mind the RAM rule above.

## 7. Suggested opening moves for the new chat

```python
from safetensors.torch import load_file
st = load_file("<v18b adapter path>/adapter_model.safetensors")
dW = {k[:-len(".lora_B.weight")]: (st[k].float() @ st[k.replace("lora_B","lora_A")].float())
      for k in st if "lora_B" in k}          # exact update, scale 1.0
```

Cleanest contrast pairs:
- **base vs v1.8a** → the pure language-side fine-tune (vision identical to base)
- **v1.8a vs v1.8b** → the pure vision-side delta (same data/steps/hparams)
- **v1.8b vs v1.9a** → what the KTIV data + continued training added
- SVD of per-module ΔW (rank ≤16 by construction) → effective rank / top
  singular directions; project vision-block ΔW onto activation space of
  manuscript vs print stimuli.
