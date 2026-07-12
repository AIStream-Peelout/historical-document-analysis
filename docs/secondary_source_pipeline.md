# Secondary Source Processing Pipeline

How academic secondary sources (books, journal articles) about the Cairo Genizah
are turned into two downstream systems:

- **Elasticsearch** — full-text + semantic search for the RAG chatbot
- **Neo4j** — knowledge graph for GraphRAG and interactive exploration

This document covers the LLM extraction passes, the knowledge-graph schema, the
relation-quality cleanup, the cross-store identifiers that link ES ↔ Neo4j, and
the import/rebuild order.

---

## Design Principles

- **Each pass is pure file I/O.** No extraction pass reads or writes a database;
  DB import is a separate final step.
- **Each pass builds on the prior pass's files** and can be re-run independently.
- **Entity extraction precedes coreference precedes relations.** You can't resolve
  "he traveled there" before knowing who "he" is, or extract a relation before the
  entities exist.
- **The page key is the filename sequence index, never the printed page.** Pages
  are identified by their `page_NNN` filename position (`page_001` → 1). The OCR/
  vision-extracted *printed* page number (`extracted_page_number`) is unreliable
  (missing, duplicated, roman numerals) and is kept only as a display field
  (`printed_page`). Every cross-stage and cross-store join uses the sequence index.
- **LLM output is constrained, not trusted.** The local model (LM Studio) is driven
  with JSON-schema constrained decoding + `/no_think`; outputs are then validated/
  compacted deterministically before anything reaches the graph.
- **ES and Neo4j share deterministic IDs** (`book_uuid` / `page_uuid`) so a KG
  triplet can be traced back to its source page's full text in ES, and vice versa.

---

## Backends and Run Tags

Passes 2–4 run through `src/models/llm/academic/llm_client.py`, which supports two
backends:

- **LM Studio** (local, default) — model `qwen3.6-35b-a3b` via the `lmstudio`
  Python SDK with constrained JSON-schema decoding. **Load it at a large context
  (65536) with no TTL** — the default low context (8192) silently fails Pass 4.
- **Gemini** — `gemini-3.5-flash`, schema enforced via prompt.

A **run tag** keeps LM Studio and Gemini outputs separate on disk. For LM Studio it
is auto-derived from the model: `qwen3.6-35b-a3b` → `lms_qwen3_6-35b-a3b`. Gemini
runs use no tag (the default paths).

| Artifact | LM Studio (`run_tag`) | Gemini (default) |
|---|---|---|
| Pass 2 entities | `entities_<run_tag>/page_NNN_entities.json` | `entities/page_NNN_entities.json` |
| Pass 3 resolved | `book_entities_resolved_<run_tag>.json` | `book_entities_resolved.json` |
| Pass 4 relations | `relations_v2/<book>/<run_tag>/book_relations.json` | `relations_v2/<book>/book_relations.json` |

---

## Pipeline Overview

```
PDF / scan
   │  [OCR or embedded-text extraction]            → raw text + page images
   ▼
[Pass 1] structured_json_llm.py                    → page_NNN_structured.json
   │
   ├───────────────────────────────────────────────→ ELASTICSEARCH (per page)
   ▼
[Pass 2] entity_tagger.py                          → page_NNN_entities.json
   ▼
[Pass 3] coreference_resolver.py                   → book_entities_resolved[_tag].json
   ▼
[Pass 4] relationship_extractor.py                 → book_relations.json  (+ _rejected.json)
   ▼
[NEO4J IMPORT]  academic_kg_import.py + others     → knowledge graph
```

---

## Prerequisite: OCR / text extraction

`src/models/ocr/book_ocr_service.py`. Scans → **Google Cloud Vision** (best for
Hebrew/non-Latin). Born-digital PDFs → embedded-text extraction (no external OCR).
Output: per-page raw text + page images in the book directory.

---

## Pass 1 — Structured Page Extraction

**Script:** `src/models/llm/academic/structured_json_llm.py` · **per page** ·
output `page_NNN_structured.json`

Imposes structure on raw OCR text, one page at a time (no entity/relation work).
Fields: `shelf_marks_mentioned`, `footnotes`, `transcriptions`, `classification`
(`general_academic` / `transcription` / `catalog` / …), `full_main_text`,
`summary`, `extracted_page_number` (printed page — unreliable, see below).

Default Pass-1 model is the 35B MoE (`qwen3.6-35b-a3b`); the old 8B VL model is
unreliable (garbled `extracted_page_number`, sparse extraction). Output dirs follow
the `*_structured_<model>/[<model>/]page_NNN_structured.json` convention.

---

## Pass 2 — Per-Page Entity Tagging

**Script:** `src/models/llm/academic/entity_tagger.py` · 2-page window ·
output `page_NNN_entities.json`

Extracts named entities per page: **people** (role hint `historical_person` /
`scholar` / `collector`), **places**, **institutions**, **shelf marks**. The
2-page window catches entities whose name appears at the bottom of the previous
page and short form at the top of the current one.

Key behaviors:

- **Constrained decoding** (`_ENTITY_JSON_SCHEMA`) + `/no_think` so the LLM emits
  valid JSON, not chain-of-thought.
- **`page_number` = filename sequence index**; the printed page is preserved as
  `printed_page`.
- **Template-leak guard.** On low-signal pages the model sometimes echoes the
  prompt's JSON skeleton verbatim (entities literally named "Full name as it
  appears" / "Place name", role = the enum string `historical_person|scholar|…`).
  `_is_leaked_entity` drops these at parse time, and the coreference aggregator
  applies the same guard so older entity files are cleaned without re-running Pass 2.

---

## Pass 3 — Coreference Resolution + Entity Classification

**Script:** `src/models/llm/academic/coreference_resolver.py` ·
input all `page_NNN_entities.json` · output `book_entities_resolved[_tag].json`

Aggregates per-page entities into deduplicated, page-attributed canonical entities,
then classifies and cleans them:

1. **Aggregate** every name → its page set + contexts (page = filename seq index,
   with a fallback that re-derives it from the filename for older entity files).
2. **LLM dedup** (cheap, name-lists only): group surface variants → one canonical
   ("Lebdi" / "Joseph b. David Lebdi"). The OpenAI-compat 400-on-`response_format`
   is retried without it.
3. **`person_class` classification** — each person tagged `biblical` / `scholar` /
   `historical`:
   - `biblical` via `biblical_person_classifier.py` (gazetteer of unambiguous
     Tanakh/Mishnah/classical figures; ambiguous overlap names like *Abraham*,
     *Joseph* are LLM-adjudicated with context, defaulting to historical).
   - `scholar` via role hints (scholar/author/editor/collector/dealer/…) +
     `ScholarRegistry` (ground-truth metadata authors). Scholar wins over
     ambiguous-biblical.
   - else `historical`.
4. **Shelf-mark validation** — `ShelfmarkNormalizer.classify_shelfmark` keeps only
   genuine shelf marks (recognised collection anchor + a digit), dropping LLM
   reasoning prose, prompt echoes, vague descriptors ("poetic Geniza fragments"),
   and Davidson *Thesaurus* poem refs (`D. I Nr. 1012`). Berlin Gemeindebibliothek
   refs are kept with an explicit Berlin prefix (canonical linking deferred).

---

## Pass 4 — Relation Extraction + Compaction

**Script:** `src/models/llm/academic/relationship_extractor.py` · per entity, scoped
to that entity's pages · output `book_relations.json` (accepted) +
`book_relations_rejected.json` (quarantine)

Extracts typed relations per entity (constrained schema + `/no_think`), each with an
evidence quote, evidence page, and confidence. Then two deterministic, no-LLM steps:

### Augmentation
`author → STUDIED → fragment` is added for each ground-truth book author × the
fragments it actually edits/discusses (gated on a transcription page or extended
discussion) — the core scholarly edges the LLM produces inconsistently.

### Compaction (rejected rows are quarantined, not deleted)
Applied in order; each dropped relation carries a `reject_reason`:

| Rule | Drops |
|---|---|
| authoritative retyping | (not a drop) endpoints that are resolved people are retyped to their canonical label (Scholar / BiblicalPerson / Person) from Pass-3 `person_class` |
| self-loop | subject == object |
| exact dedup | identical (subject, relation, object) |
| **bibliography/citation** | evidence is a reference-list citation (co-editor `COLLABORATED_WITH` cliques, per-citation `WROTE`, publisher cities) — the single biggest noise source |
| language/concept | endpoint is a language/script/linguistic term (Judaeo-Arabic, imāla, Hebrew) |
| vague entity | bare "Fragment" / "the manuscript" / etc. |
| provenance | `ORIGINATED_FROM` from a non-`Person`/`Fragment` subject (publisher cities, institution addresses, scholar regions) |
| type-pairing | `(subject_type, object_type)` violates `ALLOWED_RELATIONS` (Person/Scholar/BiblicalPerson interchangeable) |
| biblical scope | a `BiblicalPerson` edge whose other end isn't a `Fragment` (no biblical person↔person edges) |

Cross-store IDs (`book_uuid`, `page_uuid`) are stamped on each accepted relation
(see *Cross-Store Identifiers*).

---

## Knowledge Graph Schema

### Node labels

| Label | Meaning | Key |
|---|---|---|
| `Fragment` | A Genizah document fragment | `canonical_shelfmark` |
| `Person` | Genizah-era historical individual (the priority node type) | `name` |
| `Scholar` | Modern post-discovery figure: academic, editor, dealer/collector | `name` |
| `BiblicalPerson` | Canonical Tanakh/Mishnah/Talmud/classical figure | `name` |
| `Place` | Geographic location (lat/lng after geocoding) | `name` |
| `Institution` | Library / university / archive | `name` |
| `BookArticle` | Secondary source | `article_id` (sha1 of title|author|year) |
| `Entity` | Fallback when a type is unknown | `name` |

**Person vs Scholar vs BiblicalPerson.** `Person` = historical Genizah-era people
(the priority). `Scholar` = modern researchers/editors/collectors — wins for anyone
matching a ground-truth metadata author or scholar/collector role. `BiblicalPerson`
= canonical ancient figures, kept *only* so they don't conflate with Person/Scholar
and to support "fragments discussing Abraham" queries — they keep **only
`Fragment → BiblicalPerson` edges** (no person↔person biblical graph). Classification
happens in Pass 3 (`person_class`) and is applied at import; biblical detection is
conservative so a real person named "David" is never demoted.

### Relation types (`ALLOWED_RELATIONS`)

| Relation | Subject → Object |
|---|---|
| `LIVED_IN`, `TRAVELED_TO`, `ORIGINATED_FROM` | Person → Place |
| `AFFILIATED_WITH` | Scholar → Institution |
| `WROTE` | Scholar → BookArticle |
| `TRANSCRIBED`, `STUDIED` | Scholar → Fragment |
| `COLLABORATED_WITH` | Scholar → Scholar |
| `MARRIED_TO`, `RELATED_TO` | Person → Person |
| `MENTIONS_PERSON` | Fragment → Person |
| `MENTIONS_PLACE`, `ORIGINATED_IN` | Fragment → Place |
| `HELD_AT` | Fragment → Institution |
| `CITED_IN` | Fragment → BookArticle |

### Provenance properties

Every node/edge carries `data_sources` (`pgp`, `biblio`, `extracted`, `enriched`,
`merged`) and `source_books`, so any element's origin is traceable and re-imports
are idempotent.

---

## Cross-Store Identifiers (ES ↔ Neo4j)

**Module:** `src/datasets/document_models/corpus_ids.py`

Deterministic UUIDs shared by both stores, so the web app can jump from a KG triplet
to its source page's text in ES and back:

- **book key** = the book's DOI when present (`metadata.identifiers.doi`), else the
  book directory stem (`pdf_name` / `source_book`).
- `book_uuid = uuid5(GENIZAH_NS, book_key)` — uniform UUID either way.
- `page_uuid = uuid5(GENIZAH_NS, "<book_key>#p<seq>")`, **seq = filename index**.

Both stores must use the sequence index for the page component (the printed page is
unreliable). On the ES side the page document's `_id` **is** the `page_uuid`; on the
KG side each academic relation carries `r.book_uuid` / `r.page_uuid`. DOI coverage is
currently small (≈3 books); the rest fall back to the stem.

---

## Elasticsearch Indexing

**Scripts:** `src/datasets/indexing/bibliography/index_all_bibliography.py` (driver),
`index_bibliography.py`, `elastic_index_genizah.py`. Consumes **Pass 1 output only**.

Discovery is **book-directory based** (mirrors the KG's book selection): one task per
book, choosing the structured-dir variant with the most pages, resolving each book's
real `*_metadata.json` (the root `example_book_metadata.json` template is excluded).
This guarantees the ES book set matches the KG's exactly.

Each page document carries: `full_text_content` (footnotes separated),
`transcriptions`, `shelf_marks_mentioned`, book metadata, `page_number` (printed),
**`page_seq`** (filename index), **`book_uuid`**, **`page_uuid`**, `doi`. Enables
semantic (NOMIC/CLIP) + BM25 + transcription + filtered search. Re-index under a new
index version since the `_id` scheme is now `page_uuid`.

---

## Neo4j Import / Rebuild

Import scripts consume the JSON files and write the graph; they never feed back into
extraction. **Order matters** (later steps enrich earlier nodes):

```bash
# 1. Princeton PGP CSV → Fragment / Person / Place base
python -m src.datasets.indexing.neo4j.knowlege_graph_poc

# 2. Bibliography citations (RICHEST citation source — keep, not deprecated)
python -m src.datasets.indexing.neo4j.biblio_import

# 3. Academic literature relations (Pass 4 book_relations.json; stamps book_uuid/page_uuid)
python -m src.datasets.indexing.neo4j.academic_kg_import      # add --include-legacy for old *_enhanced.json

# 4. Merged-shelfmarks enrichment (FJP related people/places + KTIV catalog)
python -m src.datasets.indexing.neo4j.merged_shelfmarks_import

# 5. Geocode Place nodes
python -m src.datasets.indexing.neo4j.geocode_places
```

What each adds:

- **`biblio_import`** (`biblio.json`, ~9.9k shelf marks / 28.7k citations) →
  `BookArticle`, `Scholar`, `WROTE`, `REFERENCES`, `HELD_AT`. This is the **richest
  citation source** (the merged file's FJP bibliography has less than half as many
  citations), so it is *not* superseded by the merged import.
- **`academic_kg_import`** → the academic relations from Pass 4. By default it imports
  **only** the new `book_relations.json` (Pass 4); legacy `*_enhanced.json` /
  `enriched_relations/` require `--include-legacy`. Resolves Person/Scholar/
  BiblicalPerson labels and stamps `book_uuid`/`page_uuid`.
- **`merged_shelfmarks_import`** (NEW) → from `merged_shelfmarks.jsonl`, the rich +
  already-in-KG subset only: FJP `related_people`/`related_places` →
  `MENTIONS_PERSON`/`MENTIONS_PLACE`; date/language/description and KTIV catalog
  metadata (`subjects`, `notes`, paleography, `catalog_author`, `catalog_persons`,
  `genres`) as Fragment properties; `HELD_AT`. Fragments keyed by
  `to_canonical_id(shelfmark_display)` so they merge with existing nodes. KTIV
  scholarly **bibliography** is *not* imported yet — the "FGP Catalogue record" field
  that carries it is not captured by the current KTIV scraper.

---

## Running the Full Extraction (LM Studio)

**Per book** (resumable driver over the whole corpus): `run_kg_overnight.py` walks
every book directory and runs Passes 2→4 (`overwrite=True`), writing a
`.v2_complete` sentinel per finished book so a restart skips completed work. Per-book
/ per-pass errors are isolated.

```bash
# Ensure the model is loaded at a large context first (critical):
lms load qwen/qwen3.6-35b-a3b --identifier qwen3.6-35b-a3b --context-length 65536 -y

PYTHONPATH=. nohup .venv/bin/python run_kg_overnight.py >> ~/kg_overnight.log 2>&1 &
```

**Single book** (per-pass CLI; each derives `--run-tag` from `--lms-model`):

```bash
python -m src.models.llm.academic.entity_tagger        --dir <book> --backend lm_studio --lms-model qwen3.6-35b-a3b
python -m src.models.llm.academic.coreference_resolver --dir <book> --lms-model qwen3.6-35b-a3b
python -m src.models.llm.academic.relationship_extractor --dir <book> --backend lm_studio --lms-model qwen3.6-35b-a3b
```

---

## Operational Notes

- **LM Studio context is the #1 gotcha.** Verify with `lms ps` (CONTEXT column) before
  a run. At 8192, Pass 1 drops pages and Pass 4 yields **zero** relations. Load one
  instance at 65536 under the exact identifier the pipeline requests (`qwen3.6-35b-a3b`),
  no TTL, so it can't bind to a stale 8192 instance.
- **Resumability.** `.v2_complete` sentinels live in the repo (under `relations_v2/`),
  so they survive reboots and `/tmp` cleanups; re-running the driver continues where
  it left off.
- **Rebuild = full wipe.** To reflect the current pipeline, wipe the Neo4j DB and run
  the import order above — `academic_kg_import` MERGEs, so importing onto an old graph
  would mix new clean relations with stale noisy ones.
- **Database.** `.env` `NEO4J_DATABASE=neo4j` (the data lives in `neo4j`, not
  `genizah-prod`); scripts read this env var.

---

## Legacy / Status

| Script | Status |
|---|---|
| `biblio_import.py` | **Active** — richest citation source; complements `merged_shelfmarks_import`. |
| `merged_shelfmarks_import.py` | **Active (new)** — fragment enrichment from the merged corpus. |
| `secondary_llm_processing.py` | Deprecated — combined extraction/coref/triplets in one pass, no page attribution. Replaced by Pass 2 + Pass 3. |
| `enrich_node_relations.py` | Deprecated — queried Neo4j mid-pipeline. Replaced by Pass 4. |
| `enhanced_kg_import.py` | Superseded by `academic_kg_import.py`. |
| `enrich_fragment_people.py` | Still reads legacy `*_enhanced.json`; pending port to the new outputs. |
