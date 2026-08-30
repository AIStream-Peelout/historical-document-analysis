# Mac Studio Memory Budget & Outage Report

**Machine:** Mac Studio, **128 GB** unified memory, macOS (Darwin 25.5). Dynamic
swap (observed swap-file totals grew 12 GB → 17 GB → 19 GB as pressure rose).
**Role:** shared production (api.cairogenizah.ai) **+** ML eval/training-prep box.

**Prepared:** 2026-08-21, from direct observation this session + a repo survey of
per-component footprints. Numbers are tagged **[measured]** (seen live via
`memory_pressure` / `lms ps` / `ps` / Force-Quit dialog), **[on-disk]** (model
file size ≈ resident for MLX/GGUF), or **[inferred]** (not stated in code).

---

## 1. The core problem in one paragraph

At 128 GB the machine is comfortable for **any one** of its jobs, but it runs
**several large-memory subsystems that were each sized independently and none of
which know about the others**: LM Studio (multiple multi-GB models, no global
cap), the Docker prod stack (Neo4j with no heap limit), the KG-generation
pipeline (a resident 22 GB model at 64K context), the VLM eval/merge pipeline
(9–17 GB spikes), and a stack of Electron GUI apps (~13 GB). When two or three
peak together, committed memory exceeds ~110 GB, **swap fills (it caps near
17–19 GB), and macOS's memory killer fires** — evicting LM Studio models and/or
GUI apps, which on this box means degrading production and losing work. There is
**no admission control** anywhere: every component grabs what it wants.

---

## 2. Outage / incident log (this session, 2026-08-19 → 08-21)

| # | When | Type | What happened | My job's role |
|---|------|------|---------------|---------------|
| 1 | 08-19 | **Disk** | Disk hit **100% (267 MiB free)** during a ~11 GB KTIV dataset build. Docker's VM could no longer write → **frontend nginx 500s**; killed the font probe at 115/510. | **Cause.** I launched an ~11 GB write without a `df` pre-flight on an already-tight disk. Fixed after: NAS offload + df guard. |
| 2 | 08-20 | **RAM** | **4 VLMs + qwen3.8-27b resident in LM Studio (~50 GB)** + Docker 18.7 GB + browsers ~13 GB. Swap exhausted (**16.2 / 17.4 GB**). macOS "system has run out of application memory" → **killed Firefox AND evicted every LM Studio model** (server process survived). Prod backend stayed HTTP 200 (chat JIT-reloads). User hard-restarted. | **Not my cause.** My gauntlet was idle-gated and never ran. The pileup was interactive model-loading. |
| 3 | 08-21 | **RAM** | My v19a benchmark JIT-loaded v19a (~10 GB) onto an already-heavy box → **20% free, swap 18.5 / 19.5 GB** → user forced a **2nd hard restart**. | **Contributor.** The 10 GB was the straw, not the haystack — but I added it. My "check memory once, then launch" gate did not protect against *cumulative* load that built up after the check. |
| — | 08-21 | GPU (related) | MLX `convert` died with `kIOGPUCommandBufferCallbackErrorTimeout` — its bf16 source was on the **SMB NAS**, so lazy evaluation pulled 16 GB over the network mid-Metal-command-buffer and tripped the GPU watchdog. | **Cause.** My disk-saving optimization (bf16 → NAS) backfired. Fixed: stage source locally before convert. Not a RAM fault. |

**Common thread:** none of these were a single runaway process. Each was
**cumulative load crossing the swap ceiling**, then macOS killing something. The
box has no headroom margin and no coordination between subsystems.

---

## 3. Memory-pressure observations (free % and swap, live)

| Moment | Free RAM | Swap used / total | Note |
|--------|---------:|-------------------|------|
| Steady (prod + 1 experiment) | 41–49% | — | normal working range |
| Before crash #2 | **20%** | **18.5 / 19.5 GB** | crisis; hard restart followed |
| Before crash #1 | (est. <15%) | **16.2 / 17.4 GB** | swap essentially full |
| Just after model eviction | 65% | — | killer recovered space |
| Post-hard-restart | **86–96%** | 0 | clean slate |

**Reading:** the actionable signal is **`sysctl vm.swapusage` free < ~2 GB**,
not free-RAM % alone — macOS lets swap absorb pressure right up until it can't,
so free-% can look survivable (20%) while swap is already at the cliff.

---

## 4. Component footprint reference (the authoritative table)

### LM Studio models (resident ≈ file size)
| Model | On-disk | Resident [measured] | Used by |
|-------|--------:|--------------------:|---------|
| `qwen3.6-35b-a3b` (GGUF, MoE) | 21 GB | **22.07 GB** | prod synthesis+verification **and** KG generation |
| `qwen3.8-27b` (MLX-4bit) | 15 GB | **16.08 GB** | user experiments |
| `qwen3.5-27b` / `qwen3.6-27b` (GGUF/MLX) | 16 / 15 GB | 16.08 GB | experiments |
| `c4ai-command-r-v01-4bit` | 21 GB | — | legacy KG default (code); overridden by .env |
| `gemma-4-31b-it-MLX` | 27 GB | — | eval only, ~9 min/section (overnight) |
| `gemma-4-31b-qat-GGUF` | 18 GB | — | **FORBIDDEN — crashes the LM Studio server** |
| Fine-tuned VLMs v16/v17/v18a/v18b/**v19a** (MLX q8) | 9.2 GB ea | **9.87 GB** | benchmarks; prod keeps v18b+v17 pinned |
| `qwen/qwen3-4b-2507` (router) | 2.1 GB | **2.28 GB** | prod query routing; KG coreference |
| **Model catalog on disk (all)** | **~250 GB** | — | most are cold |

### Non-LM-Studio consumers
| Component | Footprint | Source |
|-----------|-----------|--------|
| Docker Desktop VM (whole prod stack) | **~18.7 GB RSS** [measured] | Force-Quit dialog |
| PEFT merge (bf16 base on CPU) | **~17 GB RAM peak**, ~26 GB disk peak | docs/shared_studio_runtime.md:47 |
| Prod LM Studio resident baseline | **~54 GB** when chat models + VLMs all loaded | shared_studio_runtime.md:65 |
| Firefox / ChatGPT / Claude / Cursor / Gemini / GitHub Desktop | 7.2 / 3.3 / 1.7 / 0.8 / 0.4 / 0.3 GB ≈ **~14 GB** [measured] | Force-Quit dialog |
| HF base cache (Qwen3-VL-8B, needed for every merge) | 16 GB **disk** | shared_studio_runtime.md:48 |

---

## 5. KG generation memory needs (surveyed — the part you asked about)

**Bottom line: KG generation's RAM demand is one large resident LLM, and it is
the *same* model production chat uses.** The Neo4j import code itself uses **no
model** — it only opens a bolt driver and MERGEs JSON that the LLM passes
produced.

| Pass / script | Model (default → overnight) | Local RAM | Cite |
|---------------|-----------------------------|-----------|------|
| **Orchestrator** `run_kg_overnight.py` | `qwen3.6-35b-a3b` held resident **at 65536 context** for the entire corpus run, one book at a time, shared across passes | **~22 GB weights + large KV cache** (64K ctx on a 35B MoE adds several GB) [inferred] | run_kg_overnight.py:47, 99-102 |
| Pass 1 vision/structured (`StructuredJSONLLM`) | `qwen3-vl:8b` → overridden to `qwen3.6-35b-a3b` (8B VL "proved unreliable, garbled") | 9–22 GB | structured_json_llm.py:88-92; complete_pipeline_example.py:52-61 |
| Pass 2 `EntityTagger` | **gemini-3.5-flash (cloud)** default; local `qwen3:8b` option | 0 (cloud) or ~9 GB | entity_tagger.py:666-679 |
| Pass 2.5/3 `CoreferenceResolver` | `qwen/qwen3-4b-2507` (2.3 GB) default; 35b overnight; **max_tokens escalates to 16384** because reasoning models burn budget on `reasoning_content` | 2.3–22 GB | coreference_resolver.py:106-115, 325-345 |
| Pass 4 `RelationExtractor` | **gemini-3.5-flash (cloud)** default; local `qwen3.6-35b-a3b` | 0 or ~22 GB | relationship_extractor.py:958-961 |
| `enrich_node_relations.py` | `qwen3:8b` local default | ~9 GB | enrich_node_relations.py:90-108 |
| Neo4j import/enrich scripts (7 files) | **none** — bolt driver only | 0 (Neo4j server RAM only) | academic_kg_import.py:86-88, etc. |
| Bibliography embedding (ES, **not** KG) | `Qwen3-Embedding-0.6B` local SentenceTransformer, batch 2 | **~1.2 GB** [inferred] | qwen_text_embedding.py:28-30 |

**Two implications for your solutions:**
1. **A full local KG run is not additive with prod chat if both use
   `qwen3.6-35b-a3b`** — LM Studio can share the one resident copy. But **Pass 1
   (VL model) and Pass 3 (qwen3-4b) are *different* models**, so an overnight
   run that touches all passes wants **3 distinct models resident** (VL 9 GB +
   35b 22 GB + 4b 2.3 GB ≈ **33 GB**) unless serialized per-pass with unloads.
2. The **64K context** is the hidden cost: the KV cache for a 35B MoE at 65,536
   tokens is several GB *on top of* the 22 GB weights, and it grows with
   concurrency (`PARALLEL 4` was observed on the chat models).

---

## 6. VLM eval / merge memory needs

The eval path is **deliberately single-resident** and well-behaved:
- A process-wide `asyncio` lock in `lms_transcriber.py` guarantees **one VLM
  inference in RAM at a time**, process-wide (lms_transcriber.py:84-111).
- `fragment_evals` runs documents **strictly one at a time**; extra models run
  **sequentially**, never co-resident (fragment_evals.py:1386-1436).
- So a benchmark's *own* footprint is just **one ~10 GB q8 VLM** + its KV/vision
  activations. Images are sent at **native resolution** (no eval-time resize),
  and the training/infer contract pins **6.5–7.0 MP**, which sets vision-patch
  count (the memory driver per request).

The merge pipeline is the real spike: **PEFT bf16 base on CPU ≈ 17 GB RAM,
~26 GB disk**, one MLX process at a time (shared_studio_runtime.md:47). Kraken
holds one HTR model + a transient ~2× image during binarization.

**Net:** an eval + merge cycle needs ~17 GB transiently (merge) then ~10 GB
(benchmark). The problem has never been the eval's own size — it's that it
lands *on top of* whatever else is already resident.

---

## 7. Docker / prod-stack memory (a gap worth fixing)

Surveyed `genizah_search/docker-compose.yml`: **only one memory control exists.**

| Service | Memory limit / heap | Risk |
|---------|---------------------|------|
| elasticsearch | `ES_JAVA_OPTS=-Xms1g -Xmx2g` (2 GB heap), memlock unlimited | bounded ✓ |
| **neo4j** | **NO heap, NO pagecache, NO mem_limit** → Neo4j 2026.04.0 image **auto-sizes to a fraction of host RAM** | **on a 128 GB host this can silently claim 10s of GB** |
| backend / frontend / embedding / kibana | **no `mem_limit`, no `deploy.resources`** | unbounded; embedding loads Qwen3-Embedding-0.6B (~1.2 GB) |
| — | `LM_STUDIO_MODEL_TTL=3600s` idle-unload | helps, but 1 h is long |

**The Neo4j default-heap gap is the single highest-leverage config fix** in this
report: pinning `server.memory.heap.max_size` and `server.memory.pagecache.size`
to explicit modest values would remove an uncapped multi-GB consumer that
nobody is currently accounting for.

---

## 8. Root-cause patterns (what actually keeps happening)

1. **No global admission control.** LM Studio, Docker/Neo4j, KG runs, evals, and
   GUI apps each size themselves; nothing enforces a machine-wide ceiling.
2. **Swap is the real ceiling, and it's small.** ~17–19 GB of dynamic swap on a
   128 GB box means the *effective* hard wall is ~145 GB committed, and macOS
   starts killing well before that as pressure builds.
3. **Interactive model-loading is the biggest single risk** — crash #1 was four
   VLMs + a 27B loaded by hand in LM Studio (~50 GB in one app), not any script.
4. **"Check once, then launch" is insufficient** for a shared box — load that
   builds up *after* a pre-flight check (a prod JIT-load, a new browser tab, a
   manual model load) is invisible to a one-time gate.
5. **GUI apps are a quiet ~14 GB** (Docker Desktop 18.7 GB + Electron stack) that
   competes with the same pool as the models.

---

## 9. Recommended memory-management solutions (ranked)

**Config fixes (free, high leverage):**
1. **Cap Neo4j** — set `server.memory.heap.max_size` (e.g. 4 GB) and
   `server.memory.pagecache.size` (e.g. 4 GB) in the compose env. Removes the
   single largest *uncapped* consumer. *(prod change — yours to make.)*
2. **Cap LM Studio resident set** — in LM Studio 0.4.16, set a max-loaded-models
   limit and/or shorter idle TTL so the "4 models resident" pileup (crash #1)
   cannot recur. Keep only prod chat models + the one VLM under test.
3. **Add `mem_limit`** to the backend/embedding/kibana compose services so a leak
   can't run the host to zero.

**Process fixes (I can implement in this repo):**
4. **Swap-aware admission guard in the eval/merge runners** — check
   `vm.swapusage` free *before each document/stage* and pause when it drops below
   a floor (e.g. 25 GB free RAM **or** <4 GB swap free), instead of one-time
   idle-gating. This directly prevents crash #2's "straw" pattern.
5. **Serialize KG passes with explicit unloads** — for a full local overnight KG
   run, don't hold VL + 35b + 4b co-resident; run pass-major with an unload
   between, or route Pass 2/4 to Gemini (cloud, already the code default) so only
   the 35b is ever local.

**Hardware / topology (your call):**
6. **Offload eval + KG generation to the 48 GB MacBook Pro** (serves no prod →
   can be saturated freely). Fits one VLM benchmark or one 35b KG pass with
   room; the plan is already scoped. This is the clean structural fix — it
   removes the eval/KG load from the prod box entirely.
7. **Treat the model catalog like the NAS'd datasets** — ~250 GB of models on
   disk, most cold; keep only the hot set on local SSD, the rest on NAS (LM
   Studio can load from a mounted path, ~2–3 min/16 GB over GbE).

**Monitoring (cheap insurance):**
8. A lightweight always-on watcher that logs `vm.swapusage` + `lms ps` every
   minute and alerts when swap-free < 4 GB — so the next event is caught at the
   cliff edge instead of after the hard restart.

---

## 10. Quick-reference: the budget that must hold

Rough committed-memory ceiling to stay under (128 GB − swap headroom − OS):

```
  prod LM Studio chat (router 2.3 + synth/verify 22)      ~24 GB   [must stay up]
  Docker prod stack (with Neo4j capped)                   ~12 GB   [must stay up]
  GUI apps (browsers + Electron + Docker Desktop)         ~14 GB   [user-controlled]
  --------------------------------------------------------------
  prod + desktop baseline                                 ~50 GB
  → leaves ~60–65 GB for experiments before swap engages
     ONE VLM benchmark (~10) OR one merge (~17) OR one     fits
       KG pass with the 35b (~24 incl. KV) fits;
     TWO of {second 27B model, VL+35b+4b KG, 4 VLMs}       does NOT
```

The recurring failure is loading a **second** large thing into that ~60 GB
window while the first is still resident. Every fix above is ultimately about
enforcing that one-large-experiment-at-a-time budget automatically instead of by
memory.
