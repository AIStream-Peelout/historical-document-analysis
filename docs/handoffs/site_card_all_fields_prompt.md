# Prompt for the genizah_search session: show every catalogue field on the result card and the detail view

Paste from the line below into the site session. Written 2026-10-11 from the `genizah_merged_v10` index on the Studio
and the `genizah_search` checkout (branch eval-retrieval, same card/detail code as origin/master).

---

Task: make the search result card and the document detail view display every catalogue field a v10 record carries,
and stop inventing values. Today a Rosh Hashanah search on `genizah_merged_v10` shows six cards whose only text is
"פיוט;תפילה וברכות;תפילות קבע" with "Unknown · Hebrew · Parchment" under it, while the record itself says
"Common Prayers: Musaf Rosh ha-Shanah Amidah". The search is right; the card throws the information away. Work in
this repo on a branch (never master); backend + frontend + a browser check; no index writes.

## 1. What a v10 record carries (fields of `_source`, coverage of 79,524 documents)

| field | type | present on | meaning |
|---|---|---|---|
| `description` | text | 79,524 (may be "") | merged description; for ~16.9k KTIV-only records today it is NLI's catalogue *title*, which is a `;`-joined list of Hebrew subject headings ("פיוט;תפילה וברכות;תפילות קבע" = piyyut; prayers and blessings; fixed prayers). An index update that is about to run replaces those with the English identification and moves the headings to `subjects`; handle both states. |
| `alt_titles` | keyword[] | 15,631 | KTIV "varying form of title": the catalogued identification, English then Hebrew ("Common Prayers: Sukkot Amidah", "תפילות קבע: סוכות עמידה", "Talmud Bavli: Shabbat 56 b – 57 b"). This is the title a reader needs. |
| `scholarly_frames` | keyword[] | 7,021 | KTIV scholarly identifications, one per scholar entry: passage or liturgical unit identified ("[Common Prayers]: Musaf Rosh ha-Shanah Amidah", "יום הכיפורים וידוי; מוסף יום הכיפורים עמידה", "[Bible]: Numbers 2:12 – 3:29"). More specific than the title; may be several. |
| `subjects` | keyword[] | 21,818 | keywords: KTIV catalogue subjects, PGP tags, generic work names ("Mishneh Torah", "India", "trade", "Halfon b. Menashshe"); after the pending update also the Hebrew headings. |
| `genres` | keyword[] | 15,180 | KTIV domain taxonomy ("Common Prayers", "Liturgy and Brakhot", "Piyyut", "Letters # Documentary"). Facets, not prose. |
| `language`, `main_language`, `other_languages`, `script_type` | keyword | all | "Hebrew", "Judaeo-Arabic", … ; `language` is never missing (filled from KTIV), `script_type` is often "other". |
| `date_display`, `date_start`, `date_end`, `century`, `date_certainty` | keyword/int | 8,186 dated | show the date only when `date_display` exists; `date_certainty` is "uncertain" on most records and means *no* date, not an uncertain one. |
| `document_type` | keyword | 37,786 | "Legal", "Letter", "Liturgical", … |
| `institution`, `collection`, `sub_collection`, `shelf_mark` | keyword | ~all | holding data; `shelf_mark` is the display shelf mark. |
| `image_urls[]`, `primary_image_index`, `actual_image_url`, `has_images`, `image_preferred_source`, `has_ktiv_images`, `has_bodleian_images`, `ktiv_iiif_manifest_url` (19,250), `bodleian_catalogue_url` (1,156) | | 59,199 with images | gallery + two legitimate outbound links (NLI IIIF manifest, Bodleian catalogue). |
| `transcriptions[]` (`has_transcriptions` 7,848, `has_ktiv_transcription` 2,088), `translations[]` | nested | | transcription sections `{name, lines}` and translations. |
| `bibliography[]` (`has_bib` 23,229) | object[] | | `{citation, title, authors[], year, location, relations[], url, source}`; `url` is usually null. |
| `joins_data` | object | 2,297 | join information for the fragment. |
| `named_entities{persons,places,organizations,dates}` | object | 4,029 with persons | extracted entities. |
| `sources_present[]` | keyword[] | all | which catalogues contributed ("pgp", "ktiv", "bodleian", …). |
| `semantic_eligible`, `text_representation_version`, `embedded_text_sha256`, `content_quality`, `completeness_score` | | all | technical; `semantic_eligible: false` (14,002 records) means the record has no vector and is reachable by keyword/shelf mark only. Worth one small line on the detail page ("not in semantic search: no catalogue text"). |

One real record, `Cambridge_CUL_T_S_NS_158_111` (rank 1 for "Yom Kippur" on v10):
`description` "פיוט;תפילה וברכות;תפילות קבע" · `alt_titles` ["Common Prayers: Yom ha-Kippurim", "תפילות קבע: יום הכיפורים"] ·
`scholarly_frames` ["יום הכיפורים וידוי; [תפילות קבע]: מוסף יום הכיפורים עמידה; מנחה יום הכיפורים עמידה"] · `genres`
["Common Prayers", "Liturgy and Brakhot", "Piyyut", …] · `language` "Hebrew" · 18 `image_urls` + `ktiv_iiif_manifest_url` ·
2 bibliography entries (Shivtiel & Niessen 2006; Ehrlich, Kenishta 4) · `institution` Cambridge University Library ·
`collection` Taylor-Schechter · `sub_collection` New Series · no date. The card shows only the first field and three
fallbacks.

## 2. What drops the information today (file:line from the eval-retrieval checkout; master is +~100 lines)

- Backend `search_service.py:479 _extract_metadata` builds `DocumentMetadata` (`:24`), which has no `alt_titles`,
  `scholarly_frames`, `subjects`, `genres`, `bibliography` (only `has_bib`), `date_display`, `sources_present`,
  `ktiv_iiif_manifest_url`, `bodleian_catalogue_url`, `named_entities`, `other_languages`, `semantic_eligible`.
  Every search endpoint (`/search` :818, hybrid :1779, keyword :1408, shelfmark :1223) ships that projection, so the
  frontend never receives the fields even though the hit's `_source` has them.
- Card `SearchResults.jsx:49-56` and `:118-120`: fallbacks `description: "Historical manuscript from the Cairo Genizah
  collection."`, `date: "Unknown"`, `language: "Hebrew"`, `material: "Parchment"`, and an Unsplash bookshelf photo.
  "Parchment" and "Unknown" appear on every card because no record has `material` or `date`; they are fabricated.
- Detail view `DocumentDetailView.jsx` renders none of `alt_titles`, `scholarly_frames`, `subjects`, `genres`,
  `bibliography`, `joins_data`, `named_entities`, IIIF link.
- Lexical leg `search_service.py:1341-1359` / `:1704-1720` queries `title`, `transcription_full_text`,
  `translation_full_text`, `content_type`, which do not exist in merged indices, and not `alt_titles`,
  `scholarly_frames`, `subjects`, `genres`; so keyword search is `description` only.

## 3. Card spec (compact; every value comes from the record or is omitted)

1. **Title line**: first Latin-script `alt_titles` entry; else first `alt_titles` entry; else `description`
   (truncated); else the shelf mark. Never "Document <id>" when a title exists.
2. **Description line**: `description` when it is not identical to the title line. If it is a Hebrew heading list
   (matches `^[^A-Za-z]*;[^A-Za-z]*$`), render its `;` parts as chips instead of prose, in the subjects row.
3. **Identification row**: up to two `scholarly_frames`, each truncated at ~90 chars, "+n more" to the detail view.
4. **Keywords row**: `subjects` then `genres`, at most five chips total, de-duplicated against the title words.
5. **Facts row**: `language` (plus `other_languages`), `date_display` only if present, `document_type` only if
   present, `institution` · `collection`. No material, no "Unknown", no placeholder date.
6. **Signals**: image count ("18 images"), "transcription" when `has_transcriptions` or `has_ktiv_transcription`,
   "n references" when `has_bib` (count from `bibliography`), "join" when `joins_data`. Records with
   `semantic_eligible: false` get a muted "catalogue stub" tag so a user understands why it is thin.
7. **Image**: first valid `image_urls` entry; no image → a neutral "no image" tile, never a stock photo.
8. Keep the "% match" badge; it is the cosine on the Semantic tab.

## 4. Detail view spec (complete)

Sections in this order, each omitted when empty: Identification (`alt_titles`, English first), Scholarly
identifications (`scholarly_frames`, one per line), Description, Subjects, Genres, Language(s) and script, Date
(`date_display`, `date_start`–`date_end`, `century`, `date_certainty` only when a date exists), Holding
(`institution`, `collection`, `sub_collection`, `shelf_mark`), Images (gallery over `image_urls`, `primary_image_index`
first; links "View on NLI (IIIF manifest)" from `ktiv_iiif_manifest_url` and "Bodleian catalogue" from
`bodleian_catalogue_url`), Transcriptions (sections with their `name`, lines in numeric order) and Translations,
Bibliography (`authors`, `year`, `title`, `location`, `relations`; `citation` as fallback; a link only when `url`
is non-null and passes the rule in §6), Joins (`joins_data`), Named entities, Sources (`sources_present`), and a
small technical footer (`text_representation_version`, semantic eligibility, `indexed_at`).

## 5. Backend changes

- Extend `DocumentMetadata` and `_extract_metadata` with: `alt_titles`, `scholarly_frames`, `subjects`, `genres`,
  `date_display`, `date_start`, `date_end`, `century`, `date_certainty`, `other_languages`, `script_type`,
  `sub_collection`, `sources_present`, `image_urls` (already), `primary_image_index`, `ktiv_iiif_manifest_url`,
  `bodleian_catalogue_url`, `bibliography` (trim to the first 5 on list responses, full on the detail endpoint),
  `named_entities`, `semantic_eligible`, `has_ktiv_transcription`. Use `source.get` with defaults; older indices
  (v8) lack several of these and must keep working.
- Apply the same projection on every endpoint that returns documents (semantic, hybrid, keyword, shelfmark, detail).
- Lexical leg: replace the stale field list with `description^2`, `alt_titles^2.5`, `scholarly_frames^2`,
  `subjects^1.5`, `genres`, `shelf_mark`, `full_text_content`, `transcriptions.text` (nested) if you keep a flat
  multi_match; drop the fields that do not exist. (Known separate bug: the Hybrid `function_score` uses the
  multi_match as a flat +0.5 filter and any filter zeroes every score; fix or document, but do not block this task on it.)

## 6. Rules that must hold

- Never render or link the legacy catalogue store (its name or domains) anywhere: bibliography `url` values and
  `joins_data` may still carry such links in older documents; drop any URL whose host is not NLI/KTIV, Bodleian,
  Cambridge, PGP or another holding institution. Outbound links on a record: NLI IIIF manifest, Bodleian catalogue,
  PGP when a PGP id exists.
- No fabricated values: remove the "Parchment", "Unknown", "Hebrew", Unsplash and "Historical manuscript…" fallbacks.
- Hebrew and Arabic strings are RTL: `dir="auto"` on every text node that may hold them.
- Do not change ranking, the embedder or the index; this is display and pass-through only.

## 7. Acceptance (check in the browser against `genizah_merged_v10` on the Semantic tab)

- "Rosh Hashanah": cards for `Cambridge_CUL_T_S_NS_152_45`, `Cambridge_CUL_Add_3160_10`, `Cambridge_CUL_T_S_H8_59`
  show "Common Prayers: Rosh ha-Shanah" / "Common Prayers: Musaf Rosh ha-Shanah (Amidah)" as the title line.
- "Yom Kippur": `Cambridge_CUL_T_S_NS_158_111` shows "Common Prayers: Yom ha-Kippurim", its scholarly identification
  line, "18 images", "2 references".
- "Sukkot": `Cambridge_CUL_T_S_H7_28` shows "Common Prayers: Sukkot Amidah".
- A PGP record with a long description (`Cairo_ENL_MIAC_74`) still shows the description, plus its subjects
  ("Rosh Hashanah", "prayer").
- A record with no image (`Cambridge_CUL_Add_3160_10`) shows the neutral tile, not a bookshelf.
- No card anywhere shows "Parchment" or "Unknown". The detail page of `Cambridge_CUL_T_S_NS_158_111` lists every
  section in §4 that has data. v8 cards still render.
- Add component tests for the title-line choice and the heading-list-to-chips rule, and a backend test that the
  projection carries the new fields and tolerates their absence.

Report back with the branch name, screenshots of the three queries, and anything in the data that did not fit the spec.
