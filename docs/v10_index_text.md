# Index v10: `semantic_v4` text, catalogue enrichment, text hash

Status 2026-10-11 (05:00): **`genizah_merged_v10` built and verified** (79,524 docs / 65,522 vectors from the 01:47 merge, sha `c3930caade2f54d5…`; verifier PASS); **`bibliography_text_only_0.8` built** (62 books / 4,944 pages, same embedder; metadata-attribution bug fixed first); full verifier PASS on both indices (the Passover probe now reads `scholarly_frames`, where v10 carries the identification); `_meta.source` correction script pending (`logs/v10/fix_v10_meta_source.py`); MBP handoff with paste-ready prompt in `docs/handoffs/v10_index_switch_mbp.md`. v9 is abandoned (its embedder regressed on
festival / holiday queries); the site goes from v8 straight to v10. The lead plan is the site repo's
`docs/handoffs/v10_embedder/V10_EMBEDDER_PLAN.md` (section B = index side) with the audit
`audits/text-pipeline.md`; this note records what this repo implements and the numbers measured here.

## 1. What a v10 document carries (document model, `GenizahDocument.from_merged_format`)

| field | source | in the embedded text? |
|---|---|---|
| `description` | merge precedence PGP > KTIV title > Bodleian TEI > legacy label; **2026-10-11: a KTIV title that is a subject-heading label ("פיוט;תפילה וברכות;תפילות קבע", "מקרא [טקסט") is replaced by the catalogued varying title ("Common Prayers: Sukkot Amidah", English before Hebrew)** — `merged_description` in the document model (so older merged files get it at index/updater time) and `ktiv_description` in the merge; 7,366 descriptions change in the 01:47 merge, 39 records lose a boilerplate-only vector, none gains one. See `docs/handoffs/v10_festival_cards.md` | yes |
| `alt_titles` | KTIV "Varying form of title" (label stripped, de-duplicated) | yes, folded into `Description:` by the v4 layout |
| `scholarly_frames` | KTIV scholarly entries: cleaned `frame` + non-generic work `title` (own attribute, plan B1a) | **no**: the v10 embedder is trained on the frames-off text (`semantic_v4_nf`), so the index default is frames off; `--scholarly-frames` is the opt-in for the on-arm ablation |
| `subjects` | KTIV `full_catalog.subjects` + **the headings of a KTIV label title** (`ktiv_heading_parts`, 19,052 records) + generic work titles + PGP tags (minus collection-wide labels incl. the Hebrew "קטע גניזה" / "תחום לא מזוהה", numeric ids, PGP workflow tags, legacy-store names) | **no** (`Subjects:` is written in the v8 text and dropped by the v4 layout; the words would be training labels) |
| `genres` | KTIV scholarly domain levels | no (facets / BM25) |
| `language` | PGP > legacy > **KTIV `basic_catalog.language`** (plan B1c; frozen map, ~13.9k records) | yes (`Language:` line) |
| transcriptions | legacy-catalogue dict/list items, **legacy string items now kept** (plan B1d), KTIV academic transcription | yes (capped) |
| `embedded_text_sha256` | sha256 of the exact embedded text, on every document, eligible or not (plan B2) | — |

## 2. The text (`src/embeddings/semantic_text.py`, `text_representation.py`)

* `semantic_v3` stays the frozen reference (byte-identical functions with the site copy; v9 parity).
* `semantic_v4 = layout(semantic_v3(v8))`: fold continuation lines into their field line; drop `Subjects:` and
  `Document Type:`; fold the `Alternate titles:` parts into `Description:` (normalised whole-word de-dupe,
  `MAX_ALT_CHARS = 1000`). Bibliography pages keep the v3 variant.
* Eligibility v4 is decided on the v3 text: the frozen 64 labels, length measured without `Subjects:`, alternate-title
  parts compared without brackets, plus a frozen admission list for short texts naming a festival or topic
  (`src/embeddings/data/eligibility_admit_v4.json`, PROVISIONAL copy expanded from the site repo's rules, sha256
  `928a77f8…`; the v10 round publishes the pinned file).
* `embed_text_for_version(v8_text, version)` returns `(text, eligible, text_sha256)`; `DEFAULT_TEXT_VERSION` flips to
  `semantic_v4` in the same commit that pins the v10 embedder.
* Golden fixtures `tests/data/semantic_v{3,4}_golden.jsonl` pin both transforms on ~200 real records.

## 3. The build (`index_merged_genizah.py`, `elastic_index_genizah.py`, `verify_reembedded_indexes.py`)

* `DEFAULT_INDEX_NAME = genizah_merged_v10`; `INDEX_SCHOLARLY_FRAMES = False` (one constant that flips together with
  `FINETUNED_MODEL_*` and `DEFAULT_TEXT_VERSION`); CLI `--scholarly-frames` / `--no-scholarly-frames` (default off, must
  match the pinned embedder's training text, which for v10 is the frames-off `semantic_v4`). Until the model pin flips
  `DEFAULT_TEXT_VERSION`, pass `--text-version semantic_v4` explicitly: a bare run would build a v10-named index with the
  v3 text and the v9 model.
* Mapping adds `scholarly_frames`, `subjects`, `genres` (text + keyword) and `embedded_text_sha256` (keyword, written on
  every document, ineligible ones included).
* Embeds once per text hash: a per-build cache `text_sha256 → vector` (cap 100k entries, ≈230 MB for the ~56k distinct
  eligible texts), stats `texts_embedded` / `text_cache_hits`.
* `_meta` adds `text_hash` (the rule), `document_canary.text_sha256`, `text_function{version, scholarly_frames,
  golden_fixture_sha256, hda_commit (-dirty when the tree is dirty)}`, `source{merged_path, merged_records,
  merged_sha256}` and, after the last batch, `counts{documents, eligible, distinct_text_sha256,
  distinct_eligible_text_sha256}` (read from the index itself, so a `--skip` resume still counts everything). The writer
  guard also refuses a resume with the other frames flag; a resume with a different merged file is not guarded yet.
* Verifier: `text_hash` and `text_function` required; the contract line reports the render flag, `hda_commit` and the
  fixture sha; the document canary's `text_sha256` is checked; self-consistency prints a TEXT DRIFT line (stored hash vs
  the recomputed text, FAIL with ids) and a separate vector-drift line; generated-description documents are recomputed
  from their stored text.
* In-place updater (`update_catalogue_enrichment`, for re-merge deltas): renders exactly as the index was built
  (`index_scholarly_frames(_meta)`), compares description / alt_titles / subjects / genres / scholarly_frames (and the
  stored hash once present), writes `embedded_text_sha256`, and `--refresh-document-canary` renews the canary with its
  hash. `update_generated_descriptions` works under both text versions.
* The bibliography index (`bibliography_text_only_0.8`) must be built with the same v10 embedder before the switch.

## 4. Inputs and order of operations

1. Merge: Isaac re-ran it 2026-10-10 20:50 → 78,864 records (751 new, 201 changed texts, 0 removals vs the Oct-8 file
   the v10 embedder trains on). Frozen copy for the build: `/Volumes/home/studio_offload/merged_snapshots/merged_shelfmarks_20261010.jsonl`
   (sha256 `fa28601cd94af786d4344edd869c3199da91848f61bf52577eac32f50dfcfdf1`), with its merge report beside it; pass it as `--merged` so `_meta.source` records it.
   Later batches: re-merge, upload new KTIV images, freeze again.
2. **v10 embedder = `v4-final-lora-run1` step 211** (Isaac, 2026-10-11: "v4 solves some of the issues", build with
   V4 211). Trained in Colab on the frozen Oct-8 file, text `semantic_v4_nf` (frames off), data `isaacmg/genizah-embed-train`
   tag `v4-final`; LoRA r16/α32/dropout 0.05 on q/k/v/o/gate/up/down, 211 steps = 1 epoch, final loss 2.71. The runs
   repo (`isaacmg/genizah-embed-runs` @ 8dee47ce) holds the adapter checkpoints only, so the servable model was produced
   on the Studio with the trainer's own recipe (`train_embedder_v3.load_checkpoint` on base
   `Qwen/Qwen3-Embedding-0.6B` @ 97b0c614, merged-vs-adapter min cosine 0.9999999 on 64 texts, then `save_finalized`:
   max_seq 8192, no truncation/padding baked in, prompts `{query: QUERY_PROMPT, document: ""}`; `check_saved_model`
   passed). Folder: `/Volumes/home/studio_offload/models/genizah-embed-qwen3-0.6b-v4-final-lora-run1-step211/`
   (`training_info.json` records provenance and that the Colab gate/selection cells had not run).
3. **Pushed and pinned (2026-10-11).** The lead session pushed the same merged weights (model.safetensors sha256
   `107d00fd…`, identical to the Studio merge) to `isaacmg/genizah-embed-qwen3-0.6b` as tag **`v10-lora-s0211`**, commit
   **`af0252255d3e9b8b8432e09de7c5f0689d7ada34`** (now `main`; `v3-final-run1` still points at `dbfab73f`). HDA pins
   that sha in `FINETUNED_MODEL_REVISION` and `DEFAULT_TEXT_VERSION = semantic_v4` in the same change;
   `INDEX_SCHOLARLY_FRAMES` stays False. Eval on the full 64,392-record pool (lead's report): holiday P@10 0.81
   (stock 0.64, v9 0.62), holiday canon share 0.05 (v9 0.26), focus AP 0.165 (v9 0.151), seen AP 0.37 (v9 0.44),
   known-item MRR 0.23 (v9 0.27), collection lift 2.77; Isaac shipped it knowing the strict gate fails some sub-checks.
4. **Build (Isaac's commands; ES writes are not run from this session).** Isaac's rule: a new index takes ALL scraped
   data, so the build reads the current `merged_shelfmarks.jsonl` (the indexer default), not a frozen copy; `_meta.source`
   records that file's path, record count and sha256. The exact input of the v10 build is the 2026-10-11 01:24 merge:
   79,397 records, sha256 `9ed38c959f6bc8baea81557ba2077dd134d6b315485a08aac845b8cfe7113554` (+533 KTIV-scraped records,
   156 changed texts, 0 removed vs the 20:50 file; the zip cache served all 15,263 archives with 0 misses); its provenance
   copy is `/Volumes/home/studio_offload/merged_snapshots/merged_shelfmarks_20261011.jsonl` (kept by the lead session,
   report beside it). The lead's pipeline counts 65,423 eligible records on it with frames off. Two records lost
   eligibility only because the merge precedence let a KTIV boilerplate title displace a rich description
   (Cambridge_CUL_Or_1080_1_50 → "לוח", Manchester_JRL_C_161) — the precedence fix is a follow-up for the next merge.

   ```bash
   PYTHONPATH=$PWD .venv/bin/python -m src.datasets.indexing.index_merged_genizah
   PYTHONPATH=$PWD .venv/bin/python -m src.datasets.indexing.bibliography.index_all_bibliography
   PYTHONPATH=$PWD .venv/bin/python -m src.datasets.indexing.verify_reembedded_indexes
   ```

   Defaults: index `genizah_merged_v10`, text `semantic_v4`, frames off, the v10 model (`v10-lora-s0211`), three canaries
   regenerated with it, `counts` written at the end; resume an interrupted run with `--skip N` (the log names the value).
   The bibliography index (`bibliography_text_only_0.8`) must be built with the same embedder: the backend's startup
   gate checks both indices against one model. Then the MBP handoff `docs/handoffs/v10_index_switch_mbp.md`.

## 5. Measurements (filled in as they are made)

Measured 2026-10-10 on the Oct-10 merged file (78,864 records), text block byte-identical to the site copy
(`evals/embedding_audit/semantic_text.py` @ 2026-10-09 16:36, block sha `721e3db0…`; admission file sha `2922b774…`,
boilerplate sha `d57204f7…`; parity script in the session scratchpad: 0 text and 0 reason differences on every record,
frames on and off; v3 identical on all 85,521 texts):

| | count |
|---|---|
| eligible under frozen v3 on the HEAD-shaped text (what a v9-style build would embed) | 64,964 |
| eligible under v4, frames on / frames off | 65,067 / 64,994 |
| records v4 removes vs v3 | **0** (adds 103, all via the admission list or a frame) |
| v4 reasons | eligible 65,067 · boilerplate_only 9,385 · empty 2,455 · short 1,957 |
| texts that differ with frames on vs off | 4,905 |
| distinct v4 texts among eligible (frames on) | 57,208 (7,859 duplicate texts → embed once per hash) |
| v4 text length, chars | p50 153 · p95 1,351 · max 6,580 |

Golden fixtures: 200 rows each, 12 strata (seed 13; the new `alt_dedupe_across_parts` stratum is what tells the
site block apart from the audit's draft); v3 fixture sha `9ea37e0f…`, v4 fixture sha `770f12ef…`. 247 tests pass across
the text, document-model, updater, generator and v9-build suites.
