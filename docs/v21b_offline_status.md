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
