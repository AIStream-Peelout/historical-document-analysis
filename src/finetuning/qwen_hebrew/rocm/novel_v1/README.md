# Reproduce novel-v1 training on this ROCm workstation

Please prepare and run the same 326-step novel-v1 training workload previously completed on a Lambda A100, so we can compare this AMD workstation with that run. This is the controlled reproduction we previously discussed. Do not introduce replay, extend training, rebuild the dataset, or launch multiple experimental arms. Only the final step326 checkpoint needs a quality evaluation.

You are running locally on the new workstation. Inspect its actual hardware and existing setup; do not assume the GPU is an R9700 or that the OS is still Windows. This tracked handoff contains run configuration and aggregate reference measurements. Training data and model artifacts remain in the private Hub repositories. This is the current scope; do not revive older checkpoint/prompt sweeps.

## 1. Begin with local preparation

- This entire handoff is tracked under `src/finetuning/qwen_hebrew/rocm/novel_v1/`. Its `benchmark_observer.py`, `lambda_benchmark_manifest.json`, `lambda_benchmark_summary.json` and `quality_summary.json` are the portable reference files. No handoff ZIP or files from the original machine are required for training. The Lambda timing summary describes the historical training run; use `quality_summary.json` for the subsequently completed evaluations.
- Repository: https://github.com/AIStream-Peelout/historical-document-analysis . Fetch branch `novel-rows-v1`, which contains this handoff, and read `AGENTS.md`. Create an isolated checkout/worktree and branch `codex/novel-v1-rocm` from that fetched handoff tip; preserve any existing work. The canonical training source remains pinned to the commit below; keep the new handoff files available rather than resetting the checkout to the older commit.
- Canonical source commit: `84d6c0038530566954786210265c3044f47178a5`, from branch `novel-rows-v1`, PR44. Verify the object rather than trusting a moving branch tip.
- Canonical notebook: `src/finetuning/qwen_hebrew/colab/genizah_novel_v1_lambda.ipynb`.
- Its SHA256 must be `a3530a4902bfe30cefee44232d73aab87881d9567b1398126634ef4dd4229861`.
- Inventory GPU/count/VRAM, RAM, free local storage, OS, driver, Python, ROCm, and any existing WSL/Linux/container environment. Prefer one GPU for this single-GPU comparison. Explain any other topology.
- Verify current official support for the actual GPU and OS before selecting the runtime. Preserve a working existing installation. Check https://rocm.docs.amd.com/ and follow the current compatibility matrix; the older Radeon documentation stops at7.2.1. Check https://unsloth.ai/docs/get-started/install/amd and https://github.com/unslothai/unsloth/blob/main/docker/DOCKERHUB-ROCM.md . Pin the selected package versions or container digest and save these sources/date.
- Use a dedicated environment. **Do not run `lambda_setup.sh` on AMD:** it installs CUDA wheels. Verify the installed torch is HIP-backed, the intended GPU is selected, and there is no silent CPU fallback. `torch.cuda` APIs alone do not establish CUDA versus HIP.
- Broad AMD/Unsloth support does not prove this exact Qwen3-VL 4-bit vision-training path works. Verify it. Do not replace drivers, repartition/reinstall the OS, enable remote services, or reboot without explaining the required user action first. Routine environment preparation, downloads, tests and the gated run are authorized; do not stop after a plan or repeatedly ask whether to continue.

## 2. Immutable inputs and workload

Use these private Hugging Face artifacts through secure local login/environment credentials. Never put tokens into chat, code, logs or committed files.

- Dataset: `isaacmg/genizah_novel_v1`, revision `18fa0f8bb7eab4cea899a262de3647fe245ec4a7`, **repo_type=dataset**. Its pinned export already contains `rows/train.parquet`, `rows/val.parquet`, metadata and image archives. Download and unpack using the notebook's logic; do not scrape, regenerate OCR or rerun the dataset builder.
- Train rows SHA256: `fe73dcbdfdb5e38a779b41bfb3e1c0bc7d513ad4e01046029840088e4573e9cf`.
- Val rows SHA256: `56b766a2205dfc04fb3051c0b541f74fbe9a15470f2ee609b37098f1defac0f6`.
- Expected:1184 unique train pages/951 document groups,2608 training draws;54 validation pages/47 disjoint groups. No replay. The row list represents about2.2 passes over unique pages; do not multiply it by2.2 again.
- ORIGINAL warm start: `isaacmg/qwen3-vl-8b-hebrew-v25a-ckpt`, revision `0a0e7e760f32ca19ea082a9266bfe54984d8c72d`, directory `last-checkpoint`, confirmed global_step1200. Load adapter weights only, with a fresh optimizer/scheduler.
- Actual resolved Lambda base: `unsloth/qwen3-vl-8b-instruct-unsloth-bnb-4bit`, revision `b5b904c3fcdc7541adf2a2bb219b0ed95288c794`. Assert the actual downloaded artifact/revision; Unsloth can internally redirect base loading. Do not silently substitute another quantization or floating `main`.
- Match the canonical notebook: BF16 compute,4-bit base loading, LoRA r16/alpha16 on vision and language, merger frozen,6.5–7MP processor policy,12288-token context, batch1/gradient accumulation8, AdamW8-bit, LR5e-5, cosine schedule,3% warmup, weight decay0.01, seed/data_seed3407, fixed row order,326 optimizer updates, save every50 steps, validate every100. Preserve the script-neutral training prompt and all masking/EOS/image preprocessing behavior.
- Expected fixed row-order SHA256: `cfb9a51ca43aea4b1f1eb3be5bfa8da0e303a96323164b7d8f06bacb06caead8`.
- Expected trainable tensor-spec SHA256: `8383eebe7a4b859a1c91dadf036eeb01ed57ffd6c9915cea4b523f1ea9915928`.

Derive a separate AMD runtime from the canonical source, with an explicit diff limited to paths, output identities, compatible backend/package setup, observer integration and measured session-time budgeting. Keep original hashes and files. Preserve version validation; replace CUDA-specific checks with explicit recorded AMD pins rather than bypassing them. The Lambda package manifest is beside this README. If a necessary change affects optimizer, precision, base representation, data, image policy, batch, trainable layers or schedule, explain the incompatibility and ask before making this a different experiment. Do not reduce the workload simply to fit.

## 3. Independent outputs and real training gates

Use only these new identities; verify any existing contents before writing:

- Private checkpoint repo: `isaacmg/qwen3-vl-8b-hebrew-novel-v1-rocm-ckpt`.
- Private merged-model repo: `isaacmg/qwen3-vl-8b-hebrew-novel-v1-rocm`.
- Output directory: `outputs_novel-v1-rocm`.
- W&B project: `qwen-hebrew-finetune`; new run name `genizah_novel_v1_rocm`, independent run ID/resume metadata.

Never resume the AMD run from the completed novel-v1 CUDA checkpoint. Both hardware runs start from the original v2.5a step1200 adapter. A resumed AMD session may use only its own exact checkpoint and saved sample/optimizer/scheduler/RNG state.

Run the notebook's applicable data, label, pixel/grid, warm-start, vision-gradient, dataloader and resume gates. All108 vision LoRA tensors must receive gradients, with the merger frozen. Prepared-dataloader loss must satisfy the existing comparison against direct collation. Exercise a genuine optimizer update and checkpoint save/reload in a separate smoke directory, using the real workload configuration; inference-only success is insufficient. Keep smoke output out of formal repositories and reset weights/optimizer/RNG in a fresh formal process afterward. Keep loss alarms and save-on-stop behavior; do not relax a failed gate.

When the gates pass and no workload-changing incompatibility remains, start the formal326-step run in a persistent, logged session and verify it actually makes progress. The original8-hour Lambda safety budget is not an AMD runtime prediction: estimate a safe AMD session budget from observed steps, preserving the326-step target and exact resume behavior. Use the existing checkpoint milestones100/200/250/326 for resilience; this does not authorize a checkpoint-evaluation sweep.

## 4. Hardware comparison

Attach the accompanying `benchmark_observer.py` with backend `rocm` after pre-training gates and before the formal `train_with_exact_resume`, matching the Lambda instrumentation. Preserve `benchmark_manifest.json` and `benchmark_metrics.jsonl`. Compare actual data/order/trainable-spec hashes and immutable model revisions before interpreting speed.

Lambda reference: A100-SXM4-40GB,326 steps, training runtime including validation/checkpoints20808.8183 seconds (5h46m49s), synchronized optimizer-step median60.1812 seconds after excluding the first10 steps, p9061.8270 seconds. Peak PyTorch allocated16280187392 bytes; reserved40846229504 bytes. Reserved includes allocator cache and is not the minimum required VRAM. Do not count this as separately measured total setup/merge/upload time.

Record package/driver/OS versions, setup/download/compile time, first-step and steady timing, median/p90 after the same first10-step exclusion, validation/checkpoint/export/upload time, memory, interruptions and total wall time. Collect vendor GPU utilization/physical memory where supported. Use synchronized timing. Measure actual work; do not compare the fastest step with Lambda's median. Report every unavoidable software/backend difference and avoid promising identical floating-point results.

## 5. Quality comparison and newly discovered failures

Primary goal remains JA CER below0.14. Arabic LER0.50 is a secondary target, with failure and over-reading counts. Evaluate only the completed AMD step326 model against saved results; do not regenerate baseline inference.

The established CUDA quality measurements used MLX8-bit inference in LM Studio on the original Mac, not CUDA inference. For a controlled training-quality comparison, hand the pinned final AMD adapter/merged-model revisions back for the SAME conversion, prompts, benchmark lists and decoding protocol on that evaluator. An AMD/BF16 inference result would be a separately labelled protocol difference. If benchmark assets/the evaluator are unavailable locally, finish training/export and provide the final-evaluation handoff; do not fabricate assets, re-OCR targets, or contact another host without authorization.

Fixed evaluation: PGP131 with the established `build_fragment_prompt(doc_id)` and8192 output tokens; religious140 with the same prompt function and2500 tokens; first60 Arabic benchmark documents/117 photographs with Arabic `trained` prompt and2500 tokens. Temperature0.1. The script-neutral prompt is the TRAINING prompt; do not silently switch the standard evaluation to neutral. The private benchmark assets may require transfer from the original evaluator; they are not included in this public checkout. Their absence does not block the training reproduction. The standard scoring code is in the repository.

CUDA final results versus v2.5a, same protocol:
- JA median CER,82 matched pages:0.1456 ->0.1448. All82 are scored; target not reached.
- PGP131 median CER:0.1451 ->0.1337. **Mean CER worsened0.1926 ->0.4518**: two extreme, length-limited outputs were labelled substantive by the existing classifier. JA mean worsened0.1957 ->0.4438. Median alone is misleading.
- Religious median CER on substantive answers:0.1866 ->0.1782;138/140 substantive in each model. CUDA novel-v1 has2 hallucinated answers.
- Arabic covered-image median LER, matched56 documents/67 images:0.5459 ->0.4669, but loop documents4 ->11 and mean LER1.2748 ->2.0617. All117 images were transcribed. Whole-document scorer LER worsened0.7253 ->0.8165. Do not announce an unqualified Arabic success based on the median.

Report medians AND means, same-document coverage, successful requests versus model failures, truncation/token-limit hits, overlong output, loops, hallucinations and worst pages. Retain raw answers and request metadata. Do not alter decoding, normalize away long failures, drop difficult pages, or quietly fix the failure classifier inside this reproduction. Any new classifier/decoding policy belongs in a separately disclosed diagnostic comparison using preserved raw answers.

## 6. Deliverables

Persist a compact `ROCM_RUN_STATE.json` with environment, exact source/runtime hashes, PIDs/session, artifact revisions, log/output paths, passed/failed gates, latest step and resume instructions. Give a brief readiness/progress update, then finish the authorized run rather than stopping at preparation. If a real blocker requires user action, report the precise failure, attempted supported fixes and smallest needed action.

At completion verify global_step326, private Hub artifacts, the merge and processor contract, then provide the hardware comparison, final immutable revisions and an evaluation handoff. Preserve small logs, manifests, timings and a backend diff. No extra cloud rental, replay experiment, recipe change or five-model comparison is authorized by this prompt.
