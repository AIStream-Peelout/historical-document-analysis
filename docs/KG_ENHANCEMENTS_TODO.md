# Knowledge Graph Enhancement TODO

Items identified while improving the RAG pipeline in `genizah_search`
(2026-07-26). These belong to this repo because they change graph/index
content; the serving repo only reads.

## High value

1. **Scholar biographical summaries (precomputed + curatable).**
   Add a `bio_summary` (and optionally `bio_sources`) property on `Scholar`
   nodes for major figures: dates, career, fields, why they matter to Genizah
   studies. General queries like "Who is S.D. Goitein?" currently synthesize a
   profile from scratch on every request and can only describe what the
   retrieved pages happen to support. A precomputed summary gives instant,
   stable, citable biography; hand-curate the top scholars (Goitein, Friedman,
   Schechter, Khan, …) and LLM-draft the long tail with a `curated: bool`
   flag so curated text is never overwritten by regeneration.

2. **Scholar node dedup / canonicalization.**
   "Shelomo Dov Goitein" (106 works), "Goitein, S. D." (106), "S.D. Goitein"
   (0), "Goitein, Shelomo Dov Fritz" (1) are separate nodes. Merge onto one
   canonical node with `aliases: [...]`, or add `SAME_AS` edges the serving
   side can follow. Also remove non-person Scholar nodes ("Cairo Genizah" is
   literally a Scholar node; "Unknown Author of '…'" nodes should be typed
   differently).

3. **BookArticle dedup + metadata completion.**
   Duplicate nodes per work with complementary fields (one has publisher, the
   other has year as a string). Merge; normalize `year` to int; where DOI is
   discoverable (CrossRef lookup by title+author), backfill `doi` — the chat
   UI's book popups link DOI → WorldCat/Scholar in that priority order.

4. **Bibliography page-id extraction bug (`bibliography_text_only_0.7`).**
   860 of 3,621 pages share duplicated `doc_id`s: patterns `<book>_p[]` (empty)
   and `<book>_p[119]` (a constant stamped across many pages), plus missing
   `extracted_page_number` on the same records. Breaks page-level citations
   for the newly added books. The serving repo now dedupes by ES `_id` so
   ranking is safe, but the page numbers in citations need this fixed.

5. **Fragment `occasion` / holiday facet.**
   Tag fragments (and bibliography pages) with a normalized occasion field
   (holiday, fast day, liturgical genre: qinot, piyyut, haggadah, ketubba…) via
   a local-LLM pass. Dense embeddings alone cannot reliably distinguish
   "Shavuot piyyut" from "Yom Kippur piyyut"; a structured facet makes holiday
   queries filterable and fixes the worst semantic-search confusions.

## Medium value

6. **Display-form shelf marks on Fragment nodes.**
   `canonical_shelfmark` is underscore-dialect (`T_S_12_388`). Add a
   `display_shelfmark` ("T-S 12.388") so RAG answers and map popups can show
   the scholarly citation form. (Serving side currently links the underscore
   forms via `es_doc_id`, but they read as raw ids.)

7. **`es_doc_id` coverage + retarget to `genizah_merged_v2` + reverse-lookup index.**
   Coverage verified 2026-07-27: 44,896/49,074 Fragment nodes carry es_doc_id
   and resolve against `genizah_merged_v2`; 11,956 have ≥1 REFERENCES from a
   BookArticle. Backfill the missing ~4.2k, and **CREATE INDEX for
   (f:Fragment) ON (f.es_doc_id)** — the serving repo's planned
   fragment-context lookup (open a fragment → show its scholarship) queries by
   es_doc_id and currently would scan; only canonical_shelfmark/pgpid/
   shelfmark are indexed.

8. **Scraping worklist integration.**
   The serving repo now records shelf marks cited in scholarship but absent
   from the fragment index in ES index `genizah_missing_fragments_v1`
   (occurrence counts, citing works, rejected near-matches; exposed at
   `GET /missing-fragments` on the backend). Use it to prioritize fragment
   scraping — first entry is T-S H3.111 (a T-S H3-series gap).

9. **Holiday/topic relationships in the graph.**
   Once fragments carry occasion facets, add e.g. `(:Fragment)-[:FOR_OCCASION]->
   (:Occasion {name: "Tisha B'Av"})` so the RAG planner can eventually route
   holiday queries graph-first (currently deliberately deprioritized because
   the graph has nothing to offer them).

## Notes

- Embedding contract for any re-index: `Qwen/Qwen3-Embedding-0.6B` @
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, 1024-dim, L2-normalized, docs
  raw / queries instruction-prefixed, max_seq_length 8192, canary in `_meta`.
  Keep `_id`s stable across rebuilds — the eval oracle and doc links depend
  on them.
