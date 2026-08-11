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
   other has year as a string). Merge; normalize `year` to int.

   **Direct-link identifiers are the highest-value gap here.** Coverage as of
   2026-07-31, out of 6,309 BookArticle nodes:
   - `doi`: **0** — the serving repo's "View via DOI" path can never fire.
   - `url`: **222** (JSTOR stable 79, Cambridge T-S 37, Academia.edu 35,
     doi.org 23, Persée/NLI/Princeton the rest). These are genuinely useful and
     the chat now links them directly.
   - `isbn`: **0** — no direct WorldCat item links are possible for books.

   Because ~96% of works have no direct identifier, the chat has to fall back
   to a WorldCat/Google Scholar *search*, which is a poor experience: the
   reader gets a result list instead of the work. Backfill, in priority order:
   1. **DOI via CrossRef** (query by title + first author + year); store the
      bare DOI, not a URL.
   2. **ISBN for books** (OpenLibrary / WorldCat search API) — enables direct
      `https://search.worldcat.org/isbn/<isbn>` item links.
   3. **`url` for everything else** — publisher/JSTOR/library landing page.
      Where the source bibliography already prints a stable URL, capture it at
      ingestion rather than discarding it; that is where the existing 222 came
      from and the same extractor should be applied to the rest.
   Normalize: strip trailing punctuation (some stored URLs end in "."), prefer
   https, and prefer `doi.org` form when a DOI is discoverable from the URL.

4. **Bibliography page-id extraction bug (`bibliography_text_only_0.7`).**
   860 of 3,621 pages share duplicated `doc_id`s: patterns `<book>_p[]` (empty)
   and `<book>_p[119]` (a constant stamped across many pages), plus missing
   `extracted_page_number` on the same records. Breaks page-level citations
   for the newly added books. The serving repo now dedupes by ES `_id` so
   ranking is safe, but the page numbers in citations need this fixed.

5. **`shelf_marks_mentioned` extraction emits non-shelf-marks (bibliography index).**
   The extractor is capturing publication references and prose phrases as if
   they were manuscript shelf marks. Observed on Friedman, *Jewish Marriage in
   Palestine* p. 159: `["DJD II, no. 20", "DJD II, no. 21", "Babata's Ketubba"]`,
   and on p. 24 a bare `"TS"`. DJD = *Discoveries in the Judaean Desert*, a
   publication series for Dead Sea Scrolls / Judaean Desert material (Babatha's
   ketubba is from Nahal Hever) — these are NOT Cairo Genizah shelf marks and
   must not be indexed as such. They surfaced in chat answers as "manuscripts
   cited in these sources", which misleads readers and pollutes the
   missing-fragment scraping worklist.
   Fix at extraction time: validate candidates against a shelf-mark grammar
   (collection prefix + number) and reject publication-series citations
   (`DJD`, `P.Oxy`, `no. N`, volume/issue forms), bare collection
   abbreviations with no number, and possessive prose ("X's Ketubba").
   The serving repo now filters these defensively at read time
   (`filter_manuscript_shelfmarks` in `src/backend/lms_agentic_search.py`),
   but the index should not carry them.

6. **Author-internal reference systems must map to canonical shelf marks.**
   Some works cite manuscripts through a private numbering scheme rather than
   institutional shelf marks. Goitein's *India Traders* is the clearest case:
   its `shelf_marks_mentioned` values are entries like `"III, 29"`, `"II, 24"`,
   `"III, 10"` — his own document numbers (volume, item), not shelf marks.
   These are unusable downstream: they cannot be linked, and they pollute the
   shelf-mark space with ambiguous strings.
   Needed here: resolve author-internal identifiers to canonical shelf marks
   (Goitein's India Book numbering → T-S / ENA / Bodl. marks; the concordances
   exist in the published volumes and in Friedman's later editions), and store
   the mapping so the graph can express "work W discusses fragment F" even when
   the printed citation is internal. Until then, the serving repo filters these
   out as non-shelf-marks. NOTE: this is index-side work by design — the chat
   app must not guess at these mappings.

7. **`REFERENCES` coverage and title matching for the work→manuscript bridge.**
   The serving repo now surfaces "manuscripts these works are based on" by
   joining retrieved bibliography pages to `(:BookArticle)-[:REFERENCES]->
   (:Fragment)` (see docs/DESIGN_PRECEPTS.md §1 in genizah_search). Two
   limitations observed 2026-07-29:
   - **Title join is fuzzy.** ES bibliography titles and KG BookArticle titles
     differ (subtitles, editor suffixes, bilingual forms), so the serving side
     matches on a leading-word probe. Add a stable shared key — e.g. store the
     bibliography index's `doc_id`/book identifier on the BookArticle node, or
     an `es_title` property — so the join is deterministic.
   - **Coverage is uneven.** Friedman's *Jewish Marriage in Palestine* has 126
     REFERENCES edges (excellent); Krakowski & Stern's "oldest dated document
     of the Cairo Genizah" article has none, so a reader gets no manuscript
     bridge for it. Extend REFERENCES extraction to article-length works.
   - Duplicate BookArticle nodes (item 3) split REFERENCES across copies;
     merging consolidates each work's fragment list.

8. **Hebrew-scholarship bridges in the bibliography index (audited 2026-08-03).**
   523/3,621 pages are Hebrew-dominant (Gil's *Be-Malkhut Yishma'el* alone is
   350 = 67% of them). English queries reach them only through English
   metadata, and those bridges are weak:
   - **86 of 523 Hebrew pages have Hebrew-only descriptions** — no English
     bridge at all. Generate English descriptions for these.
   - **subject_keywords are boilerplate**: the same ~6 generic tags ("Cairo
     Genizah", "Jewish history", "Islamic world"…) on effectively every Gil
     page. Regenerate page-specific topical keywords (in English) — for
     Hebrew pages they are a primary retrieval bridge, and generic tags make
     pages match every query weakly and no query strongly.
   - **language / main_language fields are dead**: empty and literal
     "Unknown" respectively on ALL 3,621 docs. Populate them (the serving
     side currently infers language from character ratios at request time).
   - Embedding note: full_text_content begins with an English metadata
     preamble (title, subjects, AI summary) which is why stored vectors show
     no cross-lingual penalty — PRESERVE that preamble structure in any
     re-embed, and improving the preamble's topical specificity directly
     improves Hebrew pages' semantic reachability.
   Serving-side mitigations shipped (query-time Hebrew search variant, Hebrew
   alias slots, cross-script gates); the durable fix is richer English
   metadata here.

9. **Fragment `occasion` / holiday facet.**
   Tag fragments (and bibliography pages) with a normalized occasion field
   (holiday, fast day, liturgical genre: qinot, piyyut, haggadah, ketubba…) via
   a local-LLM pass. Dense embeddings alone cannot reliably distinguish
   "Shavuot piyyut" from "Yom Kippur piyyut"; a structured facet makes holiday
   queries filterable and fixes the worst semantic-search confusions.

## Medium value

10. **Display-form shelf marks on Fragment nodes.**
   `canonical_shelfmark` is underscore-dialect (`T_S_12_388`). Add a
   `display_shelfmark` ("T-S 12.388") so RAG answers and map popups can show
   the scholarly citation form. (Serving side currently links the underscore
   forms via `es_doc_id`, but they read as raw ids.)

11. **`es_doc_id` coverage + retarget to `genizah_merged_v2` + reverse-lookup index.**
   Coverage verified 2026-07-27: 44,896/49,074 Fragment nodes carry es_doc_id
   and resolve against `genizah_merged_v2`; 11,956 have ≥1 REFERENCES from a
   BookArticle. Backfill the missing ~4.2k, and **CREATE INDEX for
   (f:Fragment) ON (f.es_doc_id)** — the serving repo's planned
   fragment-context lookup (open a fragment → show its scholarship) queries by
   es_doc_id and currently would scan; only canonical_shelfmark/pgpid/
   shelfmark are indexed.

12. **Scraping worklist integration.**
   The serving repo now records shelf marks cited in scholarship but absent
   from the fragment index in ES index `genizah_missing_fragments_v1`
   (occurrence counts, citing works, rejected near-matches; exposed at
   `GET /missing-fragments` on the backend). Use it to prioritize fragment
   scraping — first entry is T-S H3.111 (a T-S H3-series gap).

13. **Holiday/topic relationships in the graph.**
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
