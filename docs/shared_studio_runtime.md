# Shared Studio Runtime — READ BEFORE RUNNING ANYTHING LONG

The Mac Studio is **shared production**. `api.cairogenizah.ai` is served from
Docker containers, LM Studio and a cloudflared tunnel on this machine (owned by
the `genizah_search` repo — its "Shared Runtime" section is canonical). This
repo's evaluations, consensus runs, dataset builds, merges and probes share the
same disk, RAM, Docker daemon and LM Studio server. One process killing another
it was not aware of has happened repeatedly (port clashes, disk exhaustion,
model evictions). Treat everything below as live infrastructure.

## 1. Pre-flight before any long or service-touching job

```bash
df -h /System/Volumes/Data        # need >= 20 GB headroom beyond what the job writes
~/.lmstudio/bin/lms ps            # what is loaded / GENERATING right now
docker ps --format '{{.Names}}\t{{.Ports}}\t{{.Status}}'
```

* Big outputs (built datasets, image dumps, merged models beyond the served
  pair, run dirs) go to the NAS: `/Volumes/home/studio_offload/` — see the
  NAS policy in `project-nas-offload` memory. Disk at 100% stalls Docker's VM
  and takes the web app down.
* If LM Studio is `GENERATING` for someone else, queue behind it (the font
  probe's `lms ps` idle gate is the pattern).

## 2. Port map (host). Do not bind anything else to these.

| port | owner | what |
|---|---|---|
| 8000 | genizah_search backend | **PROD API** (cloudflared → api.cairogenizah.ai) |
| 3000 | genizah_search frontend | nginx, React build |
| 8001 | genizah_search embedding | Qwen3-Embedding-0.6B (pinned revision) |
| 7681 / 7475 | genizah_search neo4j | bolt / browser (non-default ports) — **prod graph** |
| 9200 / 5601 | genizah_search ES / Kibana | local ES 8.18.2 (experiments + backups; prod search uses remote ES) |
| 1234 | LM Studio (host) | prod LLM server AND our eval server |
| 11434 | Ollama (host) | legacy, not live |
| 8010 | genizah_search dev backend | `scripts/dev_backend_local.py` |
| **8002** | **this repo: Kraken microservice** | the ONLY port/container this repo owns |

## 3. What THIS repo runs on the shared machine

| component | where | notes |
|---|---|---|
| Kraken/MiDRASH HTR microservice | Docker image `kraken-service:linewise`, container `kraken-linewise`, **:8002**, `--restart unless-stopped`, bind-mount `src/datasets/raw_data/cairo_genizah/custom_model_weights` → `/app/models` | endpoints `/health /preload /transcribe /transcribe_lines`. Only ONE kraken container may run; the legacy `kraken-service` container (`adoring_fermat`) is stopped with restart=no. If :8002 is "already allocated" after a Docker restart, stop the *kraken* duplicate — never a `genizah_search-*` container. Source: `src/services/kraken_microservice/`. |
| Fine-tuned VLMs in LM Studio | `qwen3-vl-8b-heb-v18b-step700` (production/benchmark), `qwen3-vl-8b-heb-v17-step800` (paper Talmud flagship), `qwen3-vl-8b-heb-v18a-step700`, `qwen3-vl-8b-heb-v16-step1000` (ablations); ~9–10 GB each, MLX q8, APFS-cloned from `models/` | **Never eject v18b/v17.** Our harnesses serialize to ONE LM Studio request at a time (`lms_transcriber` lock). Long runs: 131-fragment benchmark (~1–2 h), consensus corpus runs (~55 s/doc, days), probes (hours). |
| LLM judge / segmentation (text-only) | LM Studio `qwen/qwen3.6-35b-a3b` (judge, same model as prod synthesis/verification), Gemini Flash (cloud) | hundreds of calls per eval run → contends with prod chat; run off-hours or pick a non-prod local model. |
| Merge pipeline | `.venv` PEFT merge (bf16 base on CPU, ~17 GB RAM) → `.venv-mlx` `mlx_vlm convert` → APFS clone into `~/.lmstudio/models/isaacmg/` | ONE MLX process at a time; needs ~26 GB free disk at peak; RAM spike can pressure LM Studio's resident models. |
| HF caches | `~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct` (16 GB, needed for every merge); adapters/ckpts small | cold caches (colqwen 31 GB) already archived to NAS. |
| KG / indexing scripts | `src/datasets/indexing/neo4j/*` → genizah_search-neo4j-1 (bolt 7681); `src/datasets/indexing/*elastic*` → ES | **these write to prod/backup stores — confirm with the user before any import or re-index.** |
| GCS bucket `cairo-genizah-es-json` | images for the web app + our KTIV/FJP image uploads | uploads fine; never delete/overwrite objects. |
| Training | Colab A100 only (`src/finetuning/qwen_hebrew/colab/`) — never local | local = evals, merges, probes, data builds. |
| W&B | project `cairo-genizah-transcription`, entity `igodfried` | run dirs are synced; local copies archived to NAS. |

## 4. Never do these from this repo (ask the user; they act from genizah_search)

`docker compose down|restart|up --force-recreate`, `docker system prune`,
`docker volume rm`, `docker rm|kill` of any `genizah_search-*` container,
restarting Docker Desktop, `pkill cloudflared`, `lms unload --all`, ejecting
models in the LM Studio UI, loading `google/gemma-4-31b-qat` (crashes the whole
LM Studio server), `docker compose build backend && up -d backend` (that is a
production deploy), modifying prod databases.

## 5. RAM: a balancing act, not a free-for-all

LM Studio keeps ~54 GB of prod models resident on a 128 GB machine; the web app
containers need more. A job that needs exceptional RAM must:

1. First exhaust the alternatives — a smaller/quantized model, sequential instead
   of concurrent requests, smaller context, running off-hours, waiting for idle.
2. Only when there is no other option, the Genizah web app or the RAG framework
   **may** be taken offline temporarily for that job — **ask the user first and
   get explicit approval for that specific occasion**, never pre-emptively or
   silently, and restore it immediately afterwards.
3. Never let memory pressure crash the Mac: that takes down everything,
   production included. The objective is to run experiments while the website
   keeps serving uninterrupted.

## 6. If something shared looks broken

Diagnose read-only — `docker ps`, `docker logs genizah_search-backend-1`,
`curl localhost:8000/health`, `curl localhost:1234/api/v0/models`,
`df -h` — and report to the user. Do not restart anything to "fix" it.
