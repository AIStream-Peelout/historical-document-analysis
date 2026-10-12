> **SUPERSEDED 2026-10-09 (afternoon).** The v9 embedder regressed on festival / holiday queries (bare festival names
> trained onto canon records, "Purim → Esther"), so the site will not switch to v9: the next index is **v10**, built
> on a fresh merge with the `semantic_v4` text contract and the v10 embedder (plan: genizah_search
> `docs/handoffs/v10_embedder/V10_EMBEDDER_PLAN.md`, index-side items B1-B7). Nothing more is written into v9; the
> catalogue enrichment below ships in v10 (frames as their own attribute with a render flag, `subjects` / `genres`
> as fields, `Subjects:` dropped from the embedded text by the v4 layout). The rest of this note stays as the record
> of what v9 is.

# Handoff — switch the site to index v9 (fine-tuned embedder, `semantic_v3` text)

> Paste everything below the line into the MBP session that runs the site backend (`genizah_search`, branch
> `prod-mbp`). It is self-contained. The index was built on the Studio from `historical-document-analysis` on
> 2026-10-08; this note says what is in it, what the backend must change to use it, and the order to do it in.
> Per the site repo's CLAUDE.md: no backend rebuild, index swap or container restart without Isaac's explicit approval
> for each production step.

---

## 0. State (2026-10-08 01:06, Studio)

- `genizah_merged_v9` **is built**: 78,113 documents = the merged record count (nothing dropped), 63,652 with a
  vector, 14,461 with `semantic_eligible: false` and **no vector** (empty / under-40-character / boilerplate-only
  catalogue text — by design, parity with the frozen eval). They remain in the index for shelf-mark and BM25 lookup.
- `bibliography_text_only_0.8` **is not built yet**. The backend's startup gate checks both indices against one
  embedder, so the bibliography must be re-embedded before any switch. It is built from `historical-document-analysis`
  on the Studio: `python -m src.datasets.indexing.bibliography.index_all_bibliography` (default index name is the
  code constant `DEFAULT_BIBLIOGRAPHY_INDEX`; `--dry-run` lists the books first). Ask Isaac or the Studio session.
- **2026-10-09 catalogue enrichment (in progress on the Studio, same contract):** the document model now lifts the
  KTIV scholarly identifications (``scholarly_entries[].subsections.writing_characteristics.frame`` / non-generic
  ``title``, e.g. "[Talmud Yerushalmi]: Sanhedrin 10:2 – 3") into `alt_titles` (spellings differing only in brackets,
  quotes or case collapse to one), adds a `subjects` keyword field (KTIV catalogue subjects and PGP tags; minus
  collection-wide labels, numeric authority ids, PGP workflow tags and anything naming the legacy catalogue store)
  that also appears in the embedding text as a `Subjects:` line, and a `genres` field (KTIV scholarly domain levels
  such as "Rabbinic Literature # Talmud Bavli", generic work titles such as "Talmud Bavli") for facets and BM25 only,
  never embedded (the 2026-10-09 eval showed taxonomy terms pull distinctive records towards their genre centroid). The in-place updater
  `src.datasets.indexing.update_catalogue_enrichment` writes the new fields into v9 and re-embeds only the documents
  whose text changed; `_meta` stays `semantic_v3` (a `Subjects:` line never makes a record eligible on its own, so the
  frozen eligibility of texts without it is unchanged). Expect `semantic_eligible` to flip to `true` for a few
  hundred previously vector-less KTIV records, and `subjects` to be absent (not `[]`) on documents the updater has not
  touched. Numbers and the run log: `docs/v9_index_embedding.md`. Whether it has landed is visible in the mapping (`genres` present) and on documents (`catalogue_fields_updated_at`); until then v9 has neither field and its vectors are the Oct-8 ones.
- Verify both before touching the backend (from `historical-document-analysis`, `.venv`):
  `python -m src.datasets.indexing.verify_reembedded_indexes` (defaults to the two new index names; checks `_meta`,
  reproduces all three canaries, re-derives texts with the same function, checks eligibility ↔ vector consistency).

## 1. The index contract (`_meta` on both indices)

| key | value / meaning |
|---|---|
| `model_name` / `embedding_model` | `isaacmg/genizah-embed-qwen3-0.6b` (private Hub repo) |
| `model_revision` / `embedding_model_revision` | `dbfab73f1e91a6b95eba728541cc5a391a88d933` (commit sha, never the tag) |
| `query_prompt` | exactly `Instruct: Given a search query, retrieve relevant passages\nQuery: ` |
| `text_representation_version` | `semantic_v3` — catalogue text with `Document ID:` / `Shelf Mark:` lines removed, editor credits stripped, empty field lines dropped, own and cited shelf marks and PGPIDs masked as `[shelfmark]` |
| `dims`, `max_seq_length` | 1024, 8192 |
| `canary` | the existing document-mode canary (string + vector) — what the current startup gate reads |
| `query_canary` | the same string embedded **with the query prompt** — catches the wrong-prompt trap |
| `document_canary` | one real record: `doc_id`, its `semantic_v3` text and its vector |

Per document: `semantic_eligible` (boolean) and `text_representation_version` (keyword). Writers on the Studio side
refuse to add to an index whose `_meta` differs on model, revision or text version, so anything that lands in v9 later
(see §5) carries the same contract.

## 2. Backend changes (genizah_search)

1. Embedder: `EMBEDDING_MODEL_NAME=isaacmg/genizah-embed-qwen3-0.6b`, `EMBEDDING_MODEL_REVISION=dbfab73f1e91a6b95eba728541cc5a391a88d933`
   (compose env + the defaults in `embedding_models.py`). The repo is private: give the embedding container the Hub
   token (`HF_TOKEN`) or a pre-filled `hf_home` volume — never in logs or images.
2. Loading: `max_seq_length = 8192` set explicitly; documents embedded raw; queries always with the prompt above passed
   explicitly; **never** `prompt_name="query"` (the saved model's default prompt is the stock "web search" one).
   Pin fp32 as the eval did. The 401-on-`processor_config.json` trap: sentence-transformers 5.6 does not forward the
   token to its processor probe on a private repo — export `HF_TOKEN` in the process environment (the Studio loader does
   exactly this).
3. Index names: `ELASTICSEARCH_INDEX=genizah_merged_v9`, `ELASTICSEARCH_BIBLIOGRAPHY_INDEX=bibliography_text_only_0.8`.
4. Startup gate: keep reading `canary` + `embedding_model` + `embedding_model_revision`; add
   `text_representation_version == "semantic_v3"` and the `query_canary` (cosine > 0.99 under the query prompt).
5. New fields `subjects` and `genres` (both `text` with `multilingual` analyzer + `.keyword`): usable as facets
   ("Subjects" / "Genre" filters) and in the BM25 leg (the Studio indexer weights `subjects^1.5` and `genres` next to
   `alt_titles^2`); `alt_titles` now also holds the scholarly identifications, so the tractate / folio / verse-range
   strings are searchable lexically and semantically. Only `subjects` is in the embedding text.

## 3. Query-side changes that must ship with the switch

- **Every vector leg filters `semantic_eligible: true`.** `script_score` with `cosineSimilarity` errors on the 14,461
  vector-less documents otherwise. BM25 and shelf-mark legs are unchanged.
- **Shelf-mark and collection routing** in `/search` and `/search-hybrid`: shelf marks are no longer inside the
  vectors, so "T-S 13J" or "Mosseri" must be detected and routed to `search_by_shelfmark` / a collection filter
  (handoff item 1 of `EMBEDDER_PRODUCTION_HANDOFF_2026-10-05.md`).
- `SIMILARITY_THRESHOLD = 0.4` stays: it applies to the RRF-normalised bibliography score, not raw cosine; do not add
  a raw-cosine gate (the tuned model's cosines sit ~0.1 lower).
- Hybrid weights unchanged (primary 50/50 default; bibliography RRF). Watch English top-1 on paraphrased queries — the
  one flat metric in the eval; the cheap fix if it regresses is a heavier keyword weight in `bibliography_hybrid`.

## 4. Order of operations

1. Bibliography index built and both indices verified (§0).
2. Dev backend on the two new indices; run the chat eval sets (agentic_rag v1, crosslingual, multiturn) and a
   known-item spot check (Hebrew query, English paraphrase, shelf-mark query, bibliography query).
3. Isaac's approval → switch env → restart → the same four smoke queries on prod → keep the v8 index until he says.

## 5. What arrives behind the switch (don't be surprised)

The 14,461 vector-less documents are going through the consensus pipeline (12,108 have images). Their consensus reads
will be turned into **machine-generated catalogue descriptions** by a text-only model (never the image), with
provenance fields (`description_source: generated_from_consensus`, agreed-line share, model + prompt version), and
embedded with the same `semantic_v3` function into **the same v9 index** (the writer guard permits it: same model, same
text version). For those documents `semantic_eligible` flips to `true` over time. The site should label them as
machine-derived wherever the description is shown; the agreed lines remain the citable layer.

## 6. Production bugs found on the way (separate fixes, not blockers)

- The chat's `primary_hybrid` action always fails: `SearchRequest` has no `semanticWeight` field.
- `search_service.search_hybrid` with Advanced Search filters builds a filter-only bool query, scores 0, and
  `boost_mode: multiply` zeroes every hybrid score.
- `search_by_author` sorts `(cos+1)/2` scores together with RRF scores; merge by rank if plans start mixing them.

## 7. Pointers

- Studio side: `docs/v9_index_embedding.md`, `src/embeddings/{qwen_text_embedding,semantic_text,text_representation}.py`,
  `src/datasets/indexing/{index_merged_genizah,elastic_index_genizah,verify_reembedded_indexes}.py`,
  `src/datasets/indexing/bibliography/index_all_bibliography.py`.
- Site side: `docs/handoffs/v9_embedder/V9_EMBEDDER_HANDOFF.md`, `docs/handoffs/EMBEDDER_PRODUCTION_HANDOFF_2026-10-05.md`,
  `evals/embedding_audit/` (frozen eval + reference text function, branch `eval-retrieval`).
