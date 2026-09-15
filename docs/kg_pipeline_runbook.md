# Cairo Genizah KG — Pipeline Runbook (v3)

**Audience: an agent or engineer picking up KG work with no prior context.**
Read this before running anything. It states what each pass does, what to run
when, and — most importantly — what *not* to run.

Current pipeline data version: **v3** (`PIPELINE_VERSION` in
`src/models/llm/academic/relationship_extractor.py`).

---

## 0. Ground rules (read first)

| Rule | Why |
|---|---|
| **Never re-run Pass 1 on a book that already has `page_*_structured.json`.** | Pass 1 calls the **Google Cloud Vision API, billed per page**. ~3,805 pages are already processed. Re-running costs real money and produces nothing new. |
| **Never run `enhanced_kg_import.py`.** | Dead code, superseded by `academic_kg_import.py`. It has an unvalidated `_write_generic` fallback that writes malformed triples. |
| **Check LM Studio context length before any Pass 2–4 run.** | It defaults low (8192 and 32768 both seen). At 8192 **Pass 4 silently produces zero relations**; at 32768 **Pass 3 dedup batches fail** (see below). Must be 65536. Verify: `lms ps` (CONTEXT column) or `curl -s localhost:1234/api/v0/models \| jq '.data[]\|{id,loaded_context_length}'`. Fix: `lms unload <id>; lms load <id> --context-length 65536 -y`. |
| **One LM Studio / MLX process at a time, and check free RAM first.** | Concurrent model processes cause OOM and silent truncation. A 2026-07-30 test run with the 35B plus two 8B vision models loaded came close to crashing the machine. Unload what you are not using before a corpus run. |
| **Load the model under the EXACT identifier the pipeline requests.** | See the two-instance trap below — this is the single easiest way to silently run the whole corpus at 8192 context. |

### ⚠️ The two-instance trap (found 2026-08-02)
`run_kg_overnight.py` sets `MODEL = "qwen3.6-35b-a3b"` — **unprefixed**. LM Studio
lists the model as `qwen/qwen3.6-35b-a3b` (**prefixed**). These are *different
identifiers*: requesting the unprefixed name makes LM Studio **JIT-load a second,
separate instance at its default context (8192)** while your carefully loaded
65536 instance sits idle under the prefixed name. Both show up in `lms ps`:

```
qwen/qwen3.6-35b-a3b   ...  ctx=65536   <- the one you loaded, unused
qwen3.6-35b-a3b        ...  ctx=8192    <- the one the pipeline actually used
```

Everything then fails in the confusing ways described below (truncation,
timeouts, empty content) because the real context is 8192, not what you set.

**Correct way to load** — force the served identifier to match:
```bash
lms unload qwen3.6-35b-a3b; lms unload qwen/qwen3.6-35b-a3b     # clear both
lms load qwen/qwen3.6-35b-a3b --identifier qwen3.6-35b-a3b --context-length 65536 -y
```
**Verify before every run** that exactly ONE instance is loaded at 65536:
```bash
curl -s localhost:1234/api/v0/models \
  | jq -r '.data[]|select(.state=="loaded")|"\(.id) ctx=\(.loaded_context_length)"'
```

### ⚠️ Silent degradation: Pass 3 dedup
Pass 3 catches LM Studio failures per batch and logs
`WARNING Dedup skipped for batch (LMS call failed)`, then **continues**. The run
still "succeeds" and writes a sentinel — but the book's people/places were never
deduplicated, so near-identical names (`al-Hakim` / `Al-Hakim` / `al-hakim`)
each become their own node.

**This happened 227 times during the June 2026 v2 run that produced the current
production graph** (an earlier count of 454 double-counted: that run was
launched with `nohup ... > kg_overnight.log` *while* the script also attached a
FileHandler to the same file, so every June log line is written twice — always
`sort -u` timestamps when counting events in this log). June forensics, unique
events: 227 dedup-skips = ~206 silent empty-content truncations + 21 read
timeouts + **zero HTTP 400s**. This is the most likely explanation for the
duplicate Person nodes in Neo4j.

**Root cause — reasoning-token truncation (measured 2026-08-02).** The dedup
model `run_kg_overnight.py` passes in is `qwen3.6-35b-a3b`, a *reasoning*
model. It spends the `max_tokens` budget on `reasoning_content` **before**
emitting any `content`. `/no_think` is present in `_DEDUP_SYSTEM` but **this
model ignores it** and reasons anyway (~7k characters on a 50-name batch). When
the budget runs out mid-reasoning the API returns **200 OK** with
`finish_reason="length"` and an **empty `content`** — no error, no 400, no
exception. The caller's `if not raw` then treats it as a failed call and drops
the batch. Measured on a 50-name batch:

| `max_tokens` | `finish_reason` | reasoning | `content` |
|---|---|---|---|
| 2048 | `length` | 7,753 chars | **empty → batch dropped** |
| 8192 | `stop`   | 6,976 chars | valid JSON |

Fixed in `_call_lms`: reserves **16384**; on an empty-content/`length` response
it **doubles the budget and retries**, then falls back to extracting JSON from
`reasoning_content`, and if all else fails logs a *specific* error
(`returned no usable content`) rather than failing silently. A genuine 400
(context overflow) still *halves* the reservation — the two cases need opposite
corrections, so they are handled separately. Also switched `response_format`
from `{"type":"json_object"}` (which this LM Studio build rejects outright:
`'response_format.type' must be 'json_schema' or 'text'`) to `"text"`.

**Second failure mode — read timeout.** Raising the token budget traded
truncation for a timeout: 50-name batches then failed at *exactly* 120.0s, the
old `_LMS_TIMEOUT`. Two further changes:

- `_LMS_TIMEOUT` 120 → **600s**
- `_BATCH_SIZE` 50 → **25** (reasoning cost scales with batch size; smaller
  batches also mean a single failure loses less work)

Measured after both: a 25-name batch completes in **44s** with valid content and
correct variant merging.

⚠️ **Do not "save context" by lowering `max_tokens` on a reasoning model.**
Reserved output tokens are consumed by reasoning *first*. An 8192→2048 change
made during this investigation produced a **100% dedup failure rate**.

> **Cheaper alternative worth considering:** `CoreferenceResolver`'s own default
> model is `qwen/qwen3-4b-2507` — non-reasoning and fast, and dedup is a simple
> name-matching task. `run_kg_overnight.py` overrides it with the 35B, which is
> both slower and reasoning-heavy. Using the 4B for Pass 3 would sidestep this
> whole failure class.

**Since 2026-08 the driver enforces this for you:** if any dedup batch is
skipped, `run_kg_overnight.py` logs `DEGRADED` and **refuses to write the
`.v3_complete` sentinel**, so the book is retried on the next run instead of
being silently marked done. You can still audit manually:
```bash
grep "Dedup skipped" ~/kg_overnight.log | sort -u | wc -l   # sort -u: June lines are double-written
grep "DEGRADED" ~/kg_overnight.log
```

### Other 2026-08 integrity guards (do not remove these)
- **No `reasoning_content` fallback.** `_call_lms` used to return the raw
  reasoning trace when content was empty. A truncated trace often holds a
  complete-looking *draft* grouping the model was about to retract
  (`...wait, Abraham Maimonides is his SON`) — it parsed cleanly and merged
  two different real people. A dropped batch is recoverable; a wrong merge is
  silent corruption. It now returns `None`.
- **Dedup groups are validated against the batch** (`_validate_dedup_groups`).
  A canonical not in the batch drops the group (it would rename real entities
  to a never-extracted string); variants not in the batch are filtered (one
  naming a *different* real entity used to destroy it).
- **Pass 4 refuses to fall back to the untagged resolved file.** It used to
  silently read the previous generation's `book_entities_resolved.json` and
  still stamp `pipeline_version: v3`. It now raises `FileNotFoundError`, so the
  driver marks the book failed and it is retried.
- **Discovery is run-tag aware.** Both `CoreferenceResolver.resolve_all` and
  `RelationExtractor.extract_all` glob the tagged names. Previously a tagged
  run globbed untagged paths — the resolver CLI found 0 books, and Pass 4's
  `extract_all` found 5 stale Gemini-era books.
- **The relations log line separates rejects from exact duplicates.**
  `_compact_relations` drops exact dupes silently; the old line counted them as
  "rejected", so the log disagreed with the rejected file.
| **Neo4j is Community Edition — one database only (`neo4j`).** | `CREATE DATABASE` is Enterprise-only. To get a clean target, run a **second container on another port**, not a second database. |
| **The default Neo4j is the live site's DB.** | It lives in `genizah_search/docker-compose.yml`; the `backend` service (api.cairogenizah.ai) depends on it. Validate on dev before touching it. |

### ⚠️ LM Studio degrades silently — restart the SERVER, not just the model
Observed 2026-08-06 mid-run: throughput collapsed to **~5 tok/s** for a 35B MoE
(should be 85-95 tok/s). Every dedup batch then hit the 600s timeout exactly,
and the book ground on for hours producing nothing. **Unloading and reloading
the model did NOT fix it. Only `lms server stop && lms server start` did** —
5 tok/s → 86-94 tok/s, an ~18x recovery. Something in the server's GPU/Metal
state degrades and survives a model reload.

**Measure throughput before every long run** (this is the canary — it catches
the degraded state that no status field reports):
```bash
curl -s localhost:1234/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"/no_think Say: ready"}],
       "max_tokens":512,"response_format":{"type":"text"}}' \
  | jq '.usage.completion_tokens'   # time it: expect ~2s / 85-95 tok/s
```
If it is slow: `lms server stop; lms server start`, reload the model, re-measure.

**Always load with `--parallel 1`.** The pipeline is strictly single-threaded,
but LM Studio defaults to `PARALLEL 4` and reserves KV cache for four
concurrent predictions. At 65536 context that wasted **11.6 GB** (free memory
went 16.5 → 28.1 GB on changing it) and contributed to the memory pressure
that crashed the machine:
```bash
lms load qwen/qwen3.6-35b-a3b --identifier qwen3.6-35b-a3b \
         --context-length 65536 --parallel 1 -y
```

### Targeting a dev database
`load_dotenv` does **not** override exported environment variables (verified), so
no code change is needed — just prefix the command:

```bash
NEO4J_URI=bolt://127.0.0.1:7682 PYTHONPATH=. .venv/bin/python <script>
```

Every import script reads `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` /
`NEO4J_DATABASE` from the environment, falling back to `.env`.

> **Trap:** all import scripts hardcode a default of `genizah-prod` for
> `NEO4J_DATABASE`, which *cannot exist* on Community Edition. It only works
> because `.env` sets `NEO4J_DATABASE=neo4j`. If `.env` ever drifts, imports
> fail confusingly.

---

## 1. The passes

### Pass 1 — OCR → structured JSON  *(PAID; run only for new material)*
- **Script:** `src/datasets/indexing/bibliography/batch_pipeline_runner.py`
- **Uses:** `BookOCRService` (Google Cloud Vision) + `StructuredJSONLLM`
  (LM Studio, or Gemini if `use_gemini=True`)
- **Input:** page images / PDFs
- **Output:** `<book_dir>/**/page_NNN_structured.json` with `full_main_text`,
  `summary`, `shelf_marks_mentioned`, `classification`
- **Independent of v2/v3.** No pipeline-version change has ever affected Pass 1
  output, so it is *never* part of a rebuild — only of adding new material.

### Pass 2 — per-page entity tagging
- **Class:** `EntityTagger.tag_book` (`src/models/llm/academic/entity_tagger.py`)
- **Input:** `page_*_structured.json`
- **Output:** `page_NNN_entities_<run_tag>.json` — people / places /
  institutions / shelf_marks per page
- Includes `_is_leaked_entity()`, which strips prompt-template echoes
  ("Full name as it appears", angle-bracket placeholders).

### Pass 3 — book-wide coreference + classification
- **Class:** `CoreferenceResolver.resolve_book` (`coreference_resolver.py`)
- **Input:** all Pass-2 page entity files for one book
- **Output:** `<book_dir>/book_entities_resolved_<run_tag>.json`
- **Does:** dedups names across pages (keeping `pages` lists), stamps
  `person_class` ∈ {biblical, scholar, historical}, and filters shelfmarks via
  `ShelfmarkNormalizer.classify_shelfmark`.
- `is_known_historical_author()` protects Maimonides/Saadia/ha-Levi-type figures
  from being mislabeled `scholar` just because they authored classical works.

### Pass 4 — relation extraction + deterministic compaction
- **Class:** `RelationExtractor.extract_book` (`relationship_extractor.py`)
- **Input:** the Pass-3 resolved file
- **Output:** `relations_v3/<book>/<run_tag>/book_relations.json` (accepted)
  and `book_relations_rejected.json` (quarantined, **not** imported)
- **Critical:** must use the constrained JSON schema + `/no_think`. Without the
  schema the model emits prose instead of JSON → zero relations.
- **Compaction rejects ~49% of raw relations.** Rules, in order: authoritative
  retyping → self-loop → dedup → citation-evidence (English **and Hebrew**) →
  language/concept endpoints → vague endpoints → shelfmark-as-Institution →
  ORIGINATED_FROM provenance → type-pairing → biblical scope.

### Driver for Passes 2–4
```bash
PYTHONPATH=. .venv/bin/python run_kg_overnight.py                    # whole corpus
PYTHONPATH=. .venv/bin/python run_kg_overnight.py --only <book_dir>  # one book
PYTHONPATH=. .venv/bin/python run_kg_overnight.py --list             # dry inventory
```
Runs all three passes per book, sequentially. **Resumable:** skips any book with
a `relations_v3/<book>/<run_tag>/.v3_complete` sentinel. Failures write no
sentinel, so re-running retries only what failed. Roughly ~60–80 min/book; the
full 53-book corpus took ~61 hours.

`--only` **ignores sentinels**, so it works on a fresh version tree (where no
sentinels exist yet) and lets you re-test a single book without deleting state.
This is the flag to use for adding new material and for testing a pipeline
change. `--list` shows each discovered book as `done`/`pending` without running
anything.

---

## 2. Neo4j import order

Run **in this order** after extraction. Steps 1, 2 and 4 are *not* about
academic books — they load PGP / bibliography / FJP+KTIV data.

| # | Script | Scope | Needed when adding one new book? |
|---|---|---|---|
| 1 | `knowlege_graph_poc.py` | Princeton PGP CSVs | **No** |
| 2 | `biblio_import.py` | `biblio.json` citations | **No** |
| 3 | `academic_kg_import.py` | **Pass-4 relations** | **Yes** |
| 4 | `merged_shelfmarks_import.py` | merged FJP/KTIV shelfmarks | **No** |
| 4b | `backfill_es_doc_id.py` | `Fragment.es_doc_id` (the Neo4j→ES join the site reads) | **No** |
| 5 | `geocode_places.py` | Place + Institution coordinates | Yes (cheap, incremental) |
| 6 | `enrich_fragment_people.py --run-tag v3_lms_qwen3_6-35b-a3b` | Fragment↔Person links from the Pass-3 resolved files (**the run tag is required** — the untagged glob finds nothing) | Yes |
| 7 | `pgp_person_relations_import.py` | PGP-website person→fragment role edges | **No** |
| 8 | `enrich_kg_identifiers.py --apply` | Crossref/OpenLibrary links on BookArticle; slow, network-bound; can run after cutover | **No** |

All importers use `MERGE`, so they are **idempotent** — re-running does not
duplicate nodes or edges. Steps 3–7 take `--dry-run`; use it first.

**Step 4b exists because nothing else writes `es_doc_id`.** The 44.9k values on
the June graph came from an uncommitted one-off; a rebuild has none until
`backfill_es_doc_id.py` runs (after step 4, so the merged Fragments exist).
It joins `merged_shelfmarks.jsonl` records (whose `canonical_id` *is* the ES
`_id`) onto Fragments by first join component → pgpid → full shelfmark,
verifies every id against the served index (`genizah_merged_v5`), and fills
NULLs only. Verified against the June graph on 2026-09-15: 43,847 identical,
629 differ (JSONL id-shape drift, both shapes are in the index), 998 new.

**Generation stamp.** Since 2026-09-15 `academic_kg_import.py` sets
`pipeline_version` (from each `book_relations.json`) on every node and
relationship it *creates* — `ON CREATE` only, so shared PGP/biblio elements
are never claimed. Retiring a generation is therefore
`MATCH ()-[r]->() WHERE r.pipeline_version = 'v3' DELETE r` plus the node
equivalent, instead of a wipe.

**Timing.** June figures are for a local Neo4j. Every importer issues one
statement per row, so against a Neo4j on another machine the wall time scales
with round-trip latency: the 2026-09-15 MBP build over Wi-Fi (≈11 ms per
statement vs 1 ms local) took 43 min for step 1, 26 min for step 2, 17 min
each for steps 3 and 4, ~3 min for 4b–7 together. Run from a checkout on the
DB host (or over Ethernet) if that matters.

`academic_kg_import.py` defaults to the **current version's** relations root
(`relations_v3`). To import a previous generation for comparison:
```bash
PYTHONPATH=. .venv/bin/python src/datasets/indexing/neo4j/academic_kg_import.py \
    --relations-only --relations-root src/datasets/raw_data/cairo_genizah/academic_literature/relations_v2
```

`geocode_places.py` only touches nodes where `lat IS NULL` unless `--all` is
passed, so it is naturally incremental. Use `--institutions-only` /
`--places-only` to scope it.

---

## 3. Adding a new book/article  ← the common case

Everything below is per-book. **Do not run the full pipeline.**

1. **Place the book** under
   `src/datasets/raw_data/cairo_genizah/academic_literature/<collection>/<book_dir>/`.

2. **Add a `*_metadata.json` next to it** with an `authors` list. This is not
   optional: `ScholarRegistry` reads these to distinguish real authors from
   hallucinated ones and to drive the `STUDIED` augmentation. A missing
   metadata file degrades author attribution silently.

3. **Pass 1 for that book only** (the one legitimate paid run):
   ```bash
   PYTHONPATH=. .venv/bin/python -m src.datasets.indexing.bibliography.batch_pipeline_runner --help
   ```
   Use its directory-scoping flag to target only the new book. Confirm you get
   `page_NNN_structured.json` files before continuing.

4. **Passes 2–4 — just run the normal driver.** It is already incremental:
   every existing book has a `.v3_complete` sentinel and is skipped, so only
   the new book is processed.
   ```bash
   PYTHONPATH=. .venv/bin/python run_kg_overnight.py
   ```

5. **Import the new relations** (idempotent; existing books re-MERGE harmlessly):
   ```bash
   PYTHONPATH=. .venv/bin/python src/datasets/indexing/neo4j/academic_kg_import.py --relations-only
   ```

6. **Geocode the new nodes** (skips everything already placed):
   ```bash
   PYTHONPATH=. .venv/bin/python src/datasets/indexing/neo4j/geocode_places.py
   ```

**Do NOT, for a new book:** re-run Pass 1 on existing books · run steps 1/2/4 of
the import order · bump `PIPELINE_VERSION` · delete sentinels · wipe the DB.

---

## 4. Versioning rules

`PIPELINE_VERSION` lives in `relationship_extractor.py` and flows into the
relations output dir, the run tag (`v3_lms_qwen3_6-35b-a3b`, which names the
Pass-3 resolved files), the sentinel filename, and a `pipeline_version` field
in every output JSON.

- **Bump it** when a change to Pass 2/3/4 logic would produce *different output
  for the same inputs*. The new run lands in `relations_v4/` and leaves the
  previous generation intact for comparison.
- **Do not bump it** for adding data (new books), importer-only changes, or
  geocoding changes — none of those alter Pass 2–4 output.
- After bumping, **all books re-run** (no sentinels exist in the new tree).
  That is intentional: mixed-version output in one graph is the thing this
  mechanism exists to prevent.

### Version history
- **v2** — 2026-06-21 full-corpus run (qwen3.6-35b-a3b), 47 done / 1 failed /
  5 skipped of 53. This is the generation currently in the production graph.
- **v3** — adds the Hebrew citation-evidence detector, the
  shelfmark-as-Institution reject rule, the historical-author gazetteer, and
  the Pass-3 `_call_lms` dedup fix (all written 2026-07, **never yet executed
  against the corpus**). A partial single-book test on 2026-07-30 validated
  Passes 2 and 3 end-to-end and exposed the dedup bug; its artifacts were
  deleted because they predate the fix. **No valid v3 output exists yet.**

---

## 5. Known open issues

- **`enrich_fragment_people.py`** was ported to the Pass-3 resolved files on
  2026-08-05 and first ran on 2026-09-15 (v3 MBP build: 622 edges, 122
  definite / 500 possible). Its first dry-run crashed on every book with
  edges (`KeyError: 'shelfmark'` in the logging lines; fixed the same day).
  On the June graph `certainty` is still null on all ~5,078 `MENTIONS_PERSON`
  edges because that run never happened there.
- `cairo_to_manchester_2` failed in the v2 run (`TypeError: object of type
  'bool' has no len()`, since fixed) and its only output is from an older
  model. A v3 run resolves this naturally.
- 60 book dirs have Pass-1 output but only 53 entered the KG run — probably
  alternate structured-dir variants collapsed by discovery, but worth
  confirming before a full run so nothing is silently uncovered.
- Junk Place nodes (dates, comma-joined strings), junk concept-Institutions
  ("court", "synagogue", גאון סורא), and case-only Person duplicates remain
  un-filtered at source.
- `Cambridge University Library / Bodleian Library Oxford` is a compound
  holdings node (1,313 rels). Do **not** merge it into Cambridge — it means
  fragments split between two libraries and needs a real split.
  **Regression 2026-09-15:** the ingest normalisation added on 2026-07-18
  (`InstitutionNormalizer` in steps 1/2/4) folded exactly this node onto
  `Cambridge University Library` in the MBP v3 rebuild. The normaliser now
  keeps compound names verbatim; a graph built between those dates needs the
  1,313 Lewis-Gibson (`L_G_*`) `HELD_AT` edges re-pointed.
