# v21b — offline status log (Rosh Hashanah, 2026-09-11 → 13)

Everything running unattended on the Studio while the user is away, and what it produced.
Newest entries at the bottom. Logs live in `logs/` (gitignored) and the session scratchpad.

## What is running (2026-09-11 17:30 local)

| job | what | log |
|---|---|---|
| Colab `genizah_v21b` (W&B `kozus0pg`) | the dispatcher-fixed v2.1 run, 2000 steps at ~62 s/step, checkpoints to `isaacmg/qwen3-vl-8b-hebrew-v21b-ckpt` every 100 steps | W&B |
| `watch_run_alarms` | alarm A: first train loss > 2× v2.0a's 0.513; alarm B: eval > 0.6733 + 0.02 twice in a row | scratchpad `watch_v21b_alarms.log` |
| `watch_ckpt.py` | one line per pushed checkpoint with its eval loss; stale-run alert after 3.2 h without a push | `logs/watch_ckpt_v21b.log` |
| `auto_eval_series --ver v21b` | stage → box evals → lite CER at steps 600/900/1200/1500/1800, FULL hard evals at 1800/2000 (fallback: full on the newest evaluated step ≥ 900 if the run stalls 6 h); gated on ≥20 GB disk, ≥35 % RAM, NAS, LM Studio; cleans up after every checkpoint | `logs/auto_eval_v21b.log`, state `logs/auto_eval_v21b_state.json` |
| `probe_column_transcription` | v2.0a on the 33 two-column religious-140 pages: page prompt vs per-column prompts vs ink-gap crops | `logs/probe_column_transcription.log`, results `src/datasets/evaluations/probes/` |
| consensus pipeline (PID 68844) | documentary slice, ~1 record/min, resumable | `ai_reads/run_documentary.log` |

Prod (Docker stack, LM Studio's `qwen3.8-27b` and v20a, cloudflared) is not touched by any of these.

## Start state

- v21b at launch +1 h: train loss 0.386 at step 10 (v2.0a: 0.513) — the dataloader gate passed and the model
  sees images; eval@100 = 0.6965 (+0.023 vs the warm-start checkpoint, at the alarm-B margin — expected
  early bump from six new task families; the step-200 point decides whether it is turning).
- Local disk 29 GB free (each staged candidate takes ~9 GB while it is being evaluated), RAM 66 % free.

## 2026-09-11 20:55 local — pass 1: probe finished, run healthy

**v21b run** (W&B `kozus0pg`, ~62 s/step, step ~300 at 20:45): eval 0.6965 @100 → **0.6890 @200 → 0.6954 @300**
(warm-start checkpoint 0.6733; hovering +0.02, not climbing — alarm B needs two consecutive evals > 0.6933 and has
not fired); train loss 0.44–0.62 (v2.0a-level; alarm A quiet). Checkpoints 200 and 300 pushed on schedule. The
orchestrator is waiting for step 600 (~01:40 local). Disk 28 GB free, RAM 65 % free, all five jobs alive,
consensus pipeline at 1853/8277.

**Column-transcription probe — DONE** (`src/datasets/evaluations/probes/column_transcription_qwen3_vl_8b_heb_v20a_step1800.{csv,md}`,
v2.0a on the 33 two-column religious-140 pages; A = benchmarked page prompt, B = "ONLY the right/left column"
prompts on the full page, C = page split at its widest vertical ink gap and each half decoded with the page prompt):

| condition | n | median CER | mean CER | under-read (<0.8) | over-gen (>1.2) |
|---|---|---|---|---|---|
| A page prompt | 33 | 0.243 | 0.322 | 12/33 | 1/33 |
| B column prompts | 33 | 0.256 | 0.334 | 1/33 | 8/33 |
| C ink-gap crops | 30 | 0.151 | 0.282 | 4/30 | 1/30 |

Paired on the same pages:
* **C crops fix the under-read class**: on the 10 pages the page prompt under-read, median CER **0.612 → 0.239**
  (mean 0.559 → 0.278), 7/10 no longer under-read; on the 20 on-target pages median unchanged (0.126 → 0.121).
  Net paired mean ΔCER −0.030 (10 wins > 0.05, 4 losses, 16 within ±0.05).
* **B prompt-only does not work**: it recovers the under-read pages partially (0.607 → 0.448) but the model ignores
  "only the right column" on pages it already read fully and transcribes both columns twice (len ratio ≈ 1.5 on
  4 pages) — net zero.
* **The crop regressions are the guard-rail spec**: one crop looped (FL203725568: CER 1.28 → 2.79, 3.6× the GT
  length), one page split badly (FL203737423: 0.14 → 0.57, half the text lost), 3 pages had no detectable gap.
  So the production rule is: detect two columns → decode crops → per-crop degeneracy check (length vs the page
  decode, repetition) → fall back to the page decode when a crop fails. Under those guards this is worth roughly
  −0.37 median CER on the third of two-column pages the page prompt truncates — the coverage lever the CER
  breakdown predicted. It also argues for the v2.2 data design: column-crop rows (grounded_crop is already in
  v2.1) rather than "transcribe only column X" prompts.
