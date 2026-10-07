# KTIV alternate titles and keyword search (2026-09-16)

Input for the broader keyword-query rework. Records a gap found while
investigating why "Megillah 2a" did not surface T-S NS 219.40, what was
added to close it, and what the rework should build on.

## Finding

NLI's KTIV catalogue carries two title fields per manuscript:

| field | example (T-S NS 219.40) | role |
|---|---|---|
| `basic_catalog.title` | `ספרות הלכתית ופרשנות תלמודית;ספרות חז"ל;תלמוד בבלי [טקסט` | Hebrew subject heading (genre classification) |
| `basic_catalog.varying_form_of_title` | `Talmud Bavli: Megillah 2 a – b` + `תלמוד בבלי: מגילה ב ע"א – ע"ב` | the actual work / tractate / folio identification, in English and Hebrew |

The scraper captured both. The merge lifted only the subject heading into
the top-level `description`. `GenizahDocument.from_merged_format` read only
that description, so the varying title never reached Elasticsearch: not the
lexical fields, not `full_text_content`, not the embedding text.

Scale in `merged_shelfmarks.jsonl` (72,642 records):

| | count |
|---|---|
| records with a KTIV source | 10,295 |
| with a varying title | 8,815 |
| where the varying title adds tokens the description lacks | 8,810 |
| where the description is only the Hebrew subject heading | 7,463 |

For roughly three quarters of KTIV records the varying title is the only
English statement of what the fragment is, and it was discarded. Before the
fix, "Megillah" matched about 15 documents across every merged index
combined.

## What was added

- `GenizahDocument.alt_titles: List[str]`, filled from the KTIV varying
  title (the `Title in English:` label is stripped; Hebrew forms kept
  verbatim). Feeds `full_text_content`, the embedding text
  (`Alternate titles: ...`), the ES document, and the embedding cache key.
- ES mapping: `alt_titles` as `text` (multilingual analyzer) with a
  `keyword` sub-field (`ignore_above` 512). Kept separate from
  `description` so it can be weighted independently.
- This repo's `search_documents` multi-match: `alt_titles^2`.
- `src/datasets/indexing/backfill_alt_titles.py`: patches an existing index
  in place (adds mapping, bulk-updates `alt_titles` + recomputed
  `full_text_content` by canonical id, no re-embed).
- `tests/test_alt_titles.py`.

Applied to the live MacBook Elasticsearch on 2026-09-16:

| index | updated | missing | "Megillah" hits after |
|---|---|---|---|
| genizah_merged_v6 | 8,815 | 0 | 60 (57 via `alt_titles`) |
| genizah_merged_v5 | 8,278 | 537 (records newer than v5) | 58 |

Embedding vectors were not refreshed; the next full re-index picks the
titles up automatically.

## Why it matters for the rework

1. **The web app's keyword query does not use the new field yet.**
   `genizah_search/src/backend/search_service.py` (multi-match near line
   1356) searches `transcription_full_text`, `translation_full_text`,
   `description`, `title`, `document_type`, `content_type`, `collection`,
   `language`, `script_type`, `material`. It does not search
   `full_text_content` either, so the backfill is invisible to it until
   `alt_titles` is added. Suggested starting weight: `alt_titles^2.5`,
   on par with `title`, since it is a curated identification string rather
   than free text.
2. **`alt_titles` is the right place for a work / tractate / folio boost.**
   The English form is short, canonical-ish ("Talmud Bavli: <tractate>
   <folio>"), and cleanly tokenised. A `match_phrase` boost on it (as
   `search_bibliography.py` already does for `full_text_content`) would
   make tractate-plus-folio queries precise.
3. **`alt_titles.keyword` enables exact-title facets** (e.g. all fragments
   of "Talmud Bavli: Megillah") without touching the analysed field.
4. **Description quality for KTIV-only records is poor by construction.**
   7,463 records have a description that is just the Hebrew genre heading.
   The rework should consider composing a display / search description for
   KTIV-only records from `alt_titles` + heading, or at least stop weighting
   `description` as if it were prose for those records.
5. **Other KTIV fields still not surfaced** that the same investigation
   passed over: `full_catalog.subjects`, `full_catalog.notes`,
   `scholarly_entries[].subsections.writing_characteristics.{title,frame,domain}`
   (FGP-derived, e.g. `frame: "[Talmud Bavli]: Megillah 2 a – b"`,
   `domain: "Rabbinic Literature # Talmud Bavli # ..."`). `domain` in
   particular is a ready-made hierarchical genre facet. All are present in
   `sources.ktiv` in the merged JSONL and would need the same lift-and-map
   treatment; a re-index or another in-place backfill would then apply them.
