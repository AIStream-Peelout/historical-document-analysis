# Handoff — switch the site to index v10 (v10 embedder `v10-lora-s0211`, `semantic_v4` text, frames off)

> For the MBP session that runs the site backend (`genizah_search`, branch `prod-mbp`). Supersedes
> `v9_index_switch_mbp.md` (v9 is abandoned: its embedder regressed on festival queries; the site goes v8 → v10).
> Per the site repo's CLAUDE.md: no backend rebuild, index swap or container restart without Isaac's explicit approval.

## Paste this to start (MBP session, `genizah_search` on `prod-mbp`)

> Read `/Users/isaac/Documents/GitHub/historical-document-analysis/docs/handoffs/v10_index_switch_mbp.md` in full, then your
> repo's CLAUDE.md. Index v10 is BUILT and verified on the cluster (`genizah_merged_v10`: 79,524 documents, 65,522 with
> a vector, every document stamped `embedded_text_sha256`; verifier PASS on contract, three canaries, eligibility vs
> vectors, text drift 0/40, vector drift min 1.00000). It was embedded with a NEW model revision, so the backend must
> re-pull the weights before anything else: `isaacmg/genizah-embed-qwen3-0.6b` at commit
> `af0252255d3e9b8b8432e09de7c5f0689d7ada34` (tag `v10-lora-s0211`; private repo, needs `HF_TOKEN` in the embedding
> container). Steps, each production step only with Isaac's explicit OK: (1) set `EMBEDDING_MODEL_NAME` /
> `EMBEDDING_MODEL_REVISION` to that repo and sha in the compose env and `embedding_models.py` defaults, give the
> container the token, restart the embedding service so sentence-transformers downloads the revision (≈2.4 GB) — or
> pre-fill its `hf_home` volume — and prove it loaded the right weights by re-embedding `_meta.canary.string` and
> comparing with `_meta.canary.vector` (cosine > 0.99) and the same for `query_canary` under the query prompt; load with
> `max_seq_length = 8192`, fp32, documents raw, queries with the prompt passed explicitly; (2) `ELASTICSEARCH_INDEX=
> genizah_merged_v10`; `ELASTICSEARCH_BIBLIOGRAPHY_INDEX=bibliography_text_only_0.8` (built 2026-10-11 with the same model, 4,944
> pages, verifier PASS; your startup gate checks both indices against one embedder); (3) startup gate: `text_representation_version == "semantic_v4"`,
> `embedding_model_revision == af0252…`, canary + query_canary; (4) query side: every vector leg filters
> `semantic_eligible: true`; shelf-mark / collection routing to `search_by_shelfmark`; new BM25 / facet fields
> `scholarly_frames^2`, `subjects^1.5`, `genres`; optional collapse on `embedded_text_sha256`; (5) dev backend on the
> two indices → chat eval sets + known-item spot checks including a festival query ("Hoshana Rabbah liturgy", the v10
> fix) → Isaac's approval → prod env switch → restart → smoke queries → keep v8 until he says. Report the canary
> cosines, the startup-gate output and the eval numbers back to Isaac.

## 0. State (2026-10-11 03:00, Studio)

- **`genizah_merged_v10` is built** (Isaac, 2026-10-11 01:48–02:35 on the Studio): 79,524 documents = the 01:47 merge
  (`merged_shelfmarks.jsonl`, sha256 `c3930caade2f54d52c4d16c3f99c7fc580aa4e5e3c3cd8134f6f48d37ac34bef`, provenance copy
  `merged_snapshots/merged_shelfmarks_20261011_0147.jsonl`), 65,522 with a vector (`semantic_eligible: true`), 57,310
  distinct eligible texts (one embedding each), every document carries `embedded_text_sha256`, `text_representation_version`
  = `semantic_v4` on all. Fragment-index verifier: PASS (contract, canary, query canary, document canary, eligibility ↔
  vectors, text drift 0 of 40, vector drift min 1.00000, unrelated-pair cosine median 0.151). One cosmetic defect:
  `_meta.source` still names the 01:24 merge because the index was created by a first attempt and resumed after the
  re-merge; `logs/v10/fix_v10_meta_source.py` (Studio, Isaac) rewrites just that key.
- **`bibliography_text_only_0.8` is built** (2026-10-11 14:10–14:40, same embedder): 62 books, 4,944 page documents,
  all with a vector (= the page count of 0.7; 26 pages shared between two book directories are stored once, 2 pages
  failed schema validation). Discovery bug fixed first: with several metadata files in one folder the alphabetically
  first was taken, so in 0.7 two Cambridge articles carried the Hekhalot article's title/author and the Rylands
  "Searching for the last Genizah fragment" article carried another's; 0.8 attributes them correctly.
- **Full verifier on both indices:** every contract, canary, eligibility-vs-vector, text-drift, vector-drift and
  discrimination and known-item check PASSES (§0b).
- Code is landed in `historical-document-analysis` (uncommitted; Isaac picks the branch).

## 0b. Verifier status (both indices)

`verify_reembedded_indexes` passes every check on `genizah_merged_v10` and `bibliography_text_only_0.8`: contract,
string / query / document canaries at cosine 1.0, eligibility vs vectors, text drift 0 of 40, vector drift 1.0,
discrimination, and the known-item probes. One probe ("Passover Haggadah seder") first reported FAIL because it only
looked for the word in `description`; v10's top-10 are all Haggadah fragments identified in `scholarly_frames`
("[תוספות של סידור]: הגדה … של פסח"), and 97 of its top-100 name the Haggadah in a catalogue field. The probe now reads
those fields too and passes. Judge v10 on the chat eval sets and festival queries in step 2.

## 1. The index contract (`_meta` on both indices)

| key | value / meaning |
|---|---|
| `model_name` / `embedding_model` | `isaacmg/genizah-embed-qwen3-0.6b` (private Hub repo) |
| `model_revision` / `embedding_model_revision` | `af0252255d3e9b8b8432e09de7c5f0689d7ada34` (tag `v10-lora-s0211`; pin the sha, never the tag) |
| `query_prompt` | exactly `Instruct: Given a search query, retrieve relevant passages\nQuery: ` |
| `text_representation_version` | `semantic_v4` (layout over the frozen v3 text: continuation lines folded, `Subjects:` and `Document Type:` dropped, alternate titles folded into `Description:`; eligibility decided on the v3 text plus a frozen admission list) |
| `text_function` | `{version: semantic_v4, scholarly_frames: false, golden_fixture_sha256, hda_commit}` — the documents were rendered WITHOUT the KTIV scholarly frames, exactly as the model was trained |
| `text_hash` | rule for `embedded_text_sha256` (sha256 of the UTF-8 bytes of the embedded text; on every document) |
| `source` | `{merged_path, merged_records: 79524, merged_sha256: c3930caade2f54d5…}` after the fix script (the 01:47 merge; provenance copy `merged_snapshots/merged_shelfmarks_20261011_0147.jsonl`) |
| `counts` | `{documents: 79524, eligible: 65522, distinct_text_sha256: 57832, distinct_eligible_text_sha256: 57310}` |
| `canary`, `query_canary`, `document_canary` | regenerated with the v10 model; `document_canary` also carries `text_sha256` |
| `dims`, `max_seq_length` | 1024, 8192 |

Per document: `semantic_eligible`, `text_representation_version`, `embedded_text_sha256`; new searchable fields
`scholarly_frames`, `subjects`, `genres` (multilingual text + `.keyword`) next to `alt_titles`.

## 2. Backend changes (genizah_search)

1. Embedder env: `EMBEDDING_MODEL_NAME=isaacmg/genizah-embed-qwen3-0.6b`,
   `EMBEDDING_MODEL_REVISION=af0252255d3e9b8b8432e09de7c5f0689d7ada34`; the repo is private (`HF_TOKEN` to the container,
   never in logs). Load with `max_seq_length = 8192`, fp32; documents raw; queries with the prompt above passed
   explicitly (the saved model's `prompts` are `{query: <prompt>, document: ""}`, `default_prompt_name` None).
2. Index names: `ELASTICSEARCH_INDEX=genizah_merged_v10`, `ELASTICSEARCH_BIBLIOGRAPHY_INDEX=bibliography_text_only_0.8`.
3. Startup gate: `canary` + `embedding_model` + `embedding_model_revision`, `text_representation_version == "semantic_v4"`,
   `query_canary` (cosine > 0.99 under the query prompt).
4. New fields: `subjects` and `genres` as facets and BM25 fields (Studio weights `subjects^1.5`, `genres`);
   `scholarly_frames^2` next to `alt_titles^2` for tractate / folio / verse-range queries.
5. Optional: collapse duplicate catalogue texts in vector results on `embedded_text_sha256` (about 7.9k redundant
   vectors; `distinct_eligible_text_sha256` in `counts` says how many distinct vectors exist).

## 3. Query-side changes that must ship with the switch (unchanged from the v9 note)

- Every vector leg filters `semantic_eligible: true` (vector-less documents otherwise break `script_score`).
- Shelf-mark and collection routing in `/search` and `/search-hybrid` (shelf marks are masked out of the vectors).
- `SIMILARITY_THRESHOLD = 0.4` stays on the RRF-normalised bibliography score; no raw-cosine gate.

## 4. Order of operations

1. Isaac builds v10 + bibliography 0.8 on the Studio; `verify_reembedded_indexes` passes on both.
2. Dev backend on the two indices; chat eval sets (agentic_rag v1, crosslingual, multiturn) + known-item spot checks
   (Hebrew query, English paraphrase, shelf-mark query, bibliography query, a festival query such as "Hoshana Rabbah
   liturgy" — the v10 fix).
3. Isaac's approval → switch env → restart → the same smoke queries on prod → keep v8 until he says.

## 5. Pointers

- Studio: `docs/v10_index_text.md`, `src/embeddings/{semantic_text,text_representation,qwen_text_embedding}.py`,
  `src/datasets/indexing/{index_merged_genizah,elastic_index_genizah,verify_reembedded_indexes}.py`.
- Site: `docs/handoffs/v10_embedder/V10_EMBEDDER_PLAN.md`, `evals/embedding_audit/semantic_text.py` (the v4 reference),
  the pushed model's `genizah_eval/` reports.
