# Cairo Genizah Knowledge Graph — Schema (for the frontend)

Neo4j database `neo4j` (env `NEO4J_DATABASE=neo4j`). Counts are from the current
rebuilt graph (2026-06-21). This is the authoritative schema after the recent
pipeline overhaul — **read the "Frontend action items" section at the end**, it's
what changed.

## Node labels

| Label | Count | Key | Notable properties |
|---|---|---|---|
| `Fragment` | 49,074 | `canonical_shelfmark` | `shelfmark`, `collection_name`, `date`, `language`, `description`, `notes`, `subjects` (list), `genres` (list), `catalog_title`, `catalog_author`, `catalog_persons`, `scholarly_entry_count`, `paleographic_note`, `has_transcription`, `has_translation`, `is_multifragment`, `iiif_url`, `pgpid`, `pgpids` (list), `data_sources`, `source_books` |
| `Person` | 7,318 | `name` | `name_variants`, `description`, `gender`, `home_base`, `social_roles`, `related_documents_count`, `data_sources`, `source_books` |
| `Scholar` | 3,319 | `name` | same shape as `Person` |
| `BiblicalPerson` | 6 | `name` | `data_sources`, `source_books` |
| `Place` | 1,693 | `name` | `lat`, `lng`, `city`, `country`, `region`, `place_type`, `is_region`, `geocode_source`, `geocode_suspect`, `osm_display_name`, `name_variants`, `data_sources` |
| `Institution` | 415 | `name` | `city`, `country`, `region`, `collection`, `lat`, `lng`, `osm_display_name`, `data_sources` |
| `BookArticle` | 6,309 | `article_id` | `title`, `citation`, `journal`, `volume`, `pages`, `publisher`, `year`, `source_type`, `data_sources` |
| `Tag` | 2,699 | `name` | (topical subject tags) |
| `Language` | 44 | `name` | |

### Person vs Scholar vs BiblicalPerson (important for the map)
- **`Person`** = historical Genizah-era individuals (merchants, community members, letter-writers). The primary "people on the map" type.
- **`Scholar`** = modern post-discovery figures (academics, editors, collectors/dealers).
- **`BiblicalPerson`** = canonical Tanakh/Mishnah/Talmud/classical figures. Kept isolated (only fragment-mention edges) so they don't pollute Person/Scholar. **Exclude from the people map.**

A person/scholar can be either label — the map should match `(p:Person OR p:Scholar)` if it wants both, but **not** `BiblicalPerson`.

## Relationship types

`(subject) -[REL]-> (object)`, with the dominant label patterns and counts:

| Relation | Pattern(s) | Count |
|---|---|---|
| `HELD_AT` | Fragment → Institution | 42,807 |
| `HAS_TAG` | Fragment → Tag | 35,425 |
| `REFERENCES` | BookArticle → Fragment | 31,518 |
| `WRITTEN_IN` | Fragment → Language | 17,161 |
| `MENTIONS_PLACE` | Fragment → Place | 7,886 |
| `WROTE` | Scholar → BookArticle (also Person →) | 6,929 |
| `MENTIONS_PERSON` | Fragment → Person (also → Scholar) | 5,079 |
| `JOINED_WITH` | Fragment → Fragment | 4,306 |
| `RELATED_TO` | Person → Person | 3,450 |
| `ORIGINATED_FROM` | Fragment → Place; Person → Place | 2,190 |
| `WRITTEN_AT` | Fragment → Place | 1,758 |
| `COLLABORATED_WITH` | Scholar → Scholar (also Person →) | 1,407 |
| `SENT_TO` | Fragment → Place | 1,199 |
| `LIVED_IN` | Person → Place (also Scholar →) | 1,192 |
| `STUDIED` | Scholar → Fragment | 961 |
| `TRAVELED_TO` | Person → Place (also Scholar →) | 679 |
| `AFFILIATED_WITH` | Person/Scholar → Institution | 558 |
| `ORIGINATED_IN` | Fragment → Place | 546 |
| `CITED_IN` | Fragment → BookArticle | 524 |
| `MARRIED_TO` | Person → Person | 316 |
| `TRANSCRIBED` | Scholar/Person → Fragment | 234 |

### Relationship properties
- **Academic-literature edges** (`LIVED_IN`, `TRAVELED_TO`, `ORIGINATED_FROM`, `MENTIONS_PERSON`, `MENTIONS_PLACE`, `WROTE`, `STUDIED`, `AFFILIATED_WITH`, `CITED_IN`, `COLLABORATED_WITH`, `MARRIED_TO`, `RELATED_TO`, `TRANSCRIBED`): `evidence`, `evidence_page`, `confidence` (`high`/`medium`), `book_uuid`, `page_uuid`, `source`, `source_books`, `data_sources`.
- **`REFERENCES`** (bibliography): `pages`, `has_discussion`, `has_transcription`, `has_translation`, `transcription_extent`, `translation_extent`, `data_sources`.
- **`HELD_AT`**: `sub_collection` (+ `data_sources`); academic-derived ones also carry the evidence/uuid set.
- **`HAS_TAG` / `WRITTEN_IN` / `JOINED_WITH`**: no properties.

## `data_sources` taxonomy
Every node and most edges carry a `data_sources` list recording origin:
- `pgp` — Princeton Geniza Project CSV (base fragments/people/places, tags, joins, languages)
- `biblio` — bibliography citations (BookArticle/Scholar/WROTE/REFERENCES)
- `extracted` / `enriched` — academic-literature LLM pipeline (entities + relations). **`enriched` is the current pipeline's tag and carries the bulk of the new person/place edges.**
- `merged` — merged_shelfmarks enrichment (FJP related people/places + KTIV catalog)

`source_books` lists the specific book slugs that contributed a node/edge.

## Cross-store identifiers (Neo4j ↔ Elasticsearch)
Academic relations carry **`book_uuid`** and **`page_uuid`** (deterministic UUIDs).
The ES page document's `_id` **is** the `page_uuid`. So:
- KG triplet → ES page text: read `r.page_uuid`, fetch the ES doc with that `_id`.
- The page component is the **filename sequence index** (`page_001` → 1), not the printed page.
ES page docs also carry `book_uuid`, `page_seq`, `doi`, `shelf_marks_mentioned`,
`full_text_content`, `transcriptions`, `summary`.

---

## Frontend action items (what changed)

1. **New label `BiblicalPerson`** — exclude from the people layer/map.
2. **People are split into `Person` + `Scholar`.** If a query was `MATCH (:Person)…`, it now misses ~3,300 `Scholar` nodes. Use `(:Person OR :Scholar)` where both are wanted; historical-people views can stay `:Person`.
3. **`data_sources` now includes `enriched` and `merged`.** The bulk of the new academic person/place edges are tagged `enriched`. **Any source filter must include `enriched`/`merged`** or it will hide most new data.
4. **Map "home" = `LIVED_IN`, journeys = `TRAVELED_TO`.** People with only `ORIGINATED_FROM` (origin, no residence) are **not plotted** — this is why some people (e.g. Rachel the Byzantine → `ORIGINATED_FROM Constantinople`) disappeared. Consider adding an **origin layer for `ORIGINATED_FROM`**.
5. **New relation properties `book_uuid` / `page_uuid`** enable linking a triplet to its ES source page (`page_uuid` == ES `_id`).
6. **New `Fragment` properties** from KTIV/merged: `subjects`, `notes`, `genres`, `catalog_title/author/persons`, `scholarly_entry_count`, `paleographic_note`, `date`, `language`, `pgpids` — available for fragment detail panels / faceting.
7. **Place geocoding is partial:** only ~495 of 1,693 `Place` nodes have `lat`/`lng` (the Princeton historical gazetteer + a few). Many academic-extracted places (regions, concepts, Hebrew-script variants) are **ungeocoded**, so people connected only to those won't appear on the map. Filter map places on `lat IS NOT NULL`.
8. **Known data-quality caveats (consolidation pending):** the same person can appear as several variant nodes (e.g. `Goitein` / `S. D. Goitein` / `Shelomo Dov Goitein`; `Ibn 'Awkal` ×6), and the same place as variants (`Baghdad` / `Bagdad` / `בגדאד`; `Egypt` / `מצרים`). De-dupe defensively in any aggregate counts.
