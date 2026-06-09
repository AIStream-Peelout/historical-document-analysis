# Secondary Source Processing Pipeline

This document describes the multi-stage pipeline used to process academic secondary
sources (books, journal articles) about the Cairo Genizah into two downstream systems:

- **Elasticsearch** — full-text and semantic search for the RAG chatbot
- **Neo4j** — knowledge graph for GraphRAG and interactive exploration

---

## Design Principles

- **Each pass is pure file I/O.** No pass reads from or writes to any database.
  Database import is a separate final step.
- **Each pass builds on the output of the prior pass.** The pipeline is linear
  and reproducible; any pass can be re-run independently.
- **Entity extraction precedes coreference resolution.** You cannot resolve "he
  traveled there" without first knowing who "he" is. Per-page entity extraction
  (Pass 2) runs before the long-context coreference pass (Pass 3).
- **Elasticsearch and Neo4j are fed from different pipeline stages.** Elasticsearch
  needs only Pass 1 output (full text + transcriptions). Neo4j needs the full chain
  through Pass 4 (relations).

---

## Pipeline Overview

```
PDF / Scanned Document
        │
        ▼
[Prerequisite: OCR]                     → raw text + page images
        │
        ▼
[Pass 1: Structured Page Extraction]    → page_NNN_structured.json  (per page)
        │
        ├──────────────────────────────→ ELASTICSEARCH INDEXING
        │                                 (full text + transcriptions + metadata)
        ▼
[Pass 2: Per-Page Entity Tagging]       → page_NNN_entities.json    (per page)
        │
        ▼
[Pass 3: Coreference Resolution]        → book_entities_resolved.json
        │
        ▼
[Pass 4: Relation Extraction]           → book_relations.json
        │
        ▼
[NEO4J IMPORT]                          → knowledge graph
```

---

## Prerequisite: OCR

If the document is not already digitized, OCR must run first. Supported backends:

- **Google Cloud Vision API** — recommended; best results for Hebrew and non-Latin scripts
- **Doctr**
- **Unstructured**

Output: per-page raw text alongside page images, stored in the book's directory.
See `src/models/ocr/book_ocr_service.py`.

If the source is already a digital text (e.g. a downloaded PDF with embedded text),
OCR can be skipped.

---

## Pass 1 — Structured Page Extraction

**Script:** `src/models/llm/structured_json_llm.py`  
**Input:** raw OCR text + page image (per page)  
**Output:** `page_NNN_structured.json` (per page)  
**Window:** single page  

The first LLM pass processes each page independently. It does **not** attempt
entity extraction or relationship inference — its sole job is imposing structure
on the raw OCR text.

Each structured page contains:

| Field | Description |
|---|---|
| `shelf_marks_mentioned` | Shelf marks cited on this page with brief context description |
| `footnotes` | Footnote number → full footnote text, separated from main body |
| `transcriptions` | Hebrew/Aramaic transcriptions keyed by shelf mark, then line number |
| `classification` | Page type: `general_academic`, `transcription`, `catalog`, `diagram/map` |
| `full_main_text` | Cleaned main body text with footnotes removed |
| `summary` | One-paragraph summary of page content |
| `extracted_page_number` | Actual printed page number(s), not PDF position |

**Design rationale:** Keeping Pass 1 narrow and single-page makes it fast,
parallelizable, and cheap to re-run. Separating footnotes at this stage prevents
footnote text from contaminating entity extraction in later passes. Shelf mark
detection here also gives Pass 2 a head start.

---

## Pass 2 — Per-Page Entity Tagging

**Script:** `src/models/llm/entity_tagger.py` *(planned)*  
**Input:** `page_NNN_structured.json` for current page + previous page  
**Output:** `page_NNN_entities.json` (per page)  
**Window:** 2-page sliding window  

A fast, focused pass that asks one narrow question per page: *what named entities
appear here?*

Each entity file records for this page:

- **People** — name, role hint (`historical_person`, `scholar`, `collector`)
- **Places** — name, type hint (`city`, `region`, `body_of_water`, etc.)
- **Institutions** — name, type hint (`library`, `university`, `archive`, etc.)
- **Shelf marks** — confirmed/augmented from Pass 1

The 2-page window exists to catch entities whose full name appears at the bottom
of the previous page but whose pronoun or short form appears at the top of the
current page.

**Design rationale:** Every entity must carry the page number where it appears.
Document-level extraction (the legacy approach) produces entities with no page
anchor, making them invisible to relation extraction — you cannot cite evidence
for a relationship if you don't know where in the text it was asserted.

---

## Pass 3 — Coreference Resolution

**Script:** `src/models/llm/coreference_resolver.py` *(planned)*  
**Input:** all `page_NNN_entities.json` for a book  
**Output:** `book_entities_resolved.json`  
**Window:** 10-page sliding window (may be reduced if context budget is exceeded)  

The long-context pass. Given a window of already-extracted per-page entities, the
LLM resolves ambiguous references back to the named entities established in Pass 2:

- **Pronouns:** "he traveled there" → "Joseph b. David Lebdi traveled to Aden"
- **Definite descriptions:** "the merchant", "the manuscript", "the city"
- **Shelf mark shorthand:** "the above-mentioned document", abbreviated forms
- **Cross-page name variants:** "Lebdi" on page 30 = "Joseph b. David Lebdi" from page 1

Output is a consolidated, deduplicated entity list with full page-range attribution
for each entity across the book.

**Design rationale:** Coreference resolution only works well *after* per-page entity
extraction has run. Without a named-entity anchor list from Pass 2, the LLM has no
referent to resolve pronouns to. The 10-page window balances context richness against
model context limits; the window slides through the book chapter by chapter.

---

## Pass 4 — Relation Extraction

**Script:** `src/models/llm/relation_extractor.py` *(planned — replaces `enrich_node_relations.py`)*  
**Input:** `book_entities_resolved.json` + relevant `page_NNN_structured.json` pages  
**Output:** `book_relations.json`  
**Window:** per entity, scoped to only the pages where that entity appears (from Pass 3)  

With fully resolved, page-attributed entities, this pass extracts typed relationships.
Each relation cites the evidence page, a direct quote or close paraphrase, and a
confidence level (`high` = direct statement, `medium` = clear implication).

### Allowed Relation Types

| Relation | Subject | Object | Notes |
|---|---|---|---|
| `LIVED_IN` | Person | Place | historical residence |
| `TRAVELED_TO` | Person | Place | journey attested in source |
| `ORIGINATED_FROM` | Person / Fragment | Place | provenance |
| `AFFILIATED_WITH` | Scholar | Institution | academic affiliation |
| `WROTE` | Scholar | BookArticle | authorship |
| `TRANSCRIBED` | Scholar | Fragment | who produced the transcription |
| `STUDIED` | Scholar | Fragment | scholarly focus |
| `COLLABORATED_WITH` | Scholar | Scholar | co-authorship or acknowledgment |
| `MARRIED_TO` | Person | Person | attested in Genizah documents |
| `RELATED_TO` | Person | Person | family relation |
| `MENTIONS_PERSON` | Fragment | Person | person named in document |
| `MENTIONS_PLACE` | Fragment | Place | place named in document |
| `ORIGINATED_IN` | Fragment | Place | document's origin location |
| `HELD_AT` | Fragment | Institution | current holding institution |
| `CITED_IN` | Fragment | BookArticle | fragment discussed in article |

**Design rationale:** Scoping relation extraction to the pages where an entity
actually appears prevents spurious relations from unrelated sections of a long book
and keeps LLM prompts focused and evidence-grounded.

---

## Knowledge Graph Schema

### Node Types

| Label | Description | Key Properties |
|---|---|---|
| `Fragment` | A Cairo Genizah document fragment | `canonical_shelfmark`, `shelfmark`, `collection` |
| `Person` | A historical figure | `name`, `dates`, `origin` |
| `Scholar` | A modern academic | `name`, `institution`, `era` |
| `Place` | A geographic location | `name`, `lat`, `lng`, `region`, `period` |
| `Institution` | Library, university, or archive | `name`, `city`, `country` |
| `BookArticle` | Secondary source (book or article) | `title`, `author`, `year`, `article_id` |
| `Transcription` | A scholarly transcription of a fragment | `text`, `language`, `script` |

### Person vs Scholar

These are intentionally distinct node types because their meaningful relationships
differ fundamentally:

**`Person`** — a historical individual attested in or discussed by Genizah documents.
Relationships that matter: where they lived, where they traveled, who they married,
which fragments mention them, where they originated from.

**`Scholar`** — a modern academic who studies the Genizah. Relationships that matter:
which fragments they transcribed or edited, which articles they wrote, which
institutions they are affiliated with, which colleagues they collaborated with.

**Collectors** (individuals who assembled Genizah collections, e.g. Elkan Adler,
David Kaufmann) are kept as `Person` nodes. Although their relationship to fragments
differs from that of historical correspondents, they are historical actors whose
provenance connections to fragments and institutions are meaningful and distinct from
modern scholarship.

### Transcription Provenance

A `Transcription` node links to **both** the `Fragment` it transcribes and the
`BookArticle` in which it was published:

```
(Scholar)-[:TRANSCRIBED]->(Transcription)-[:OF]->(Fragment)
(Transcription)-[:PUBLISHED_IN]->(BookArticle)
```

This captures the full provenance chain: *who* transcribed it, *where* it was
published, and *which fragment* it represents. Multiple transcriptions of the same
fragment by different scholars are modeled as separate `Transcription` nodes, all
pointing to the same `Fragment`.

---

## Elasticsearch Indexing

Elasticsearch consumes **Pass 1 output only** and is indexed independently of the
Neo4j pipeline. Each page becomes a searchable document with:

- Full main text (footnotes separated)
- Transcriptions (searchable Hebrew/Aramaic primary source text)
- Shelf marks as keyword filters
- Book/article metadata (author, year, collection)
- Page classification and summary

This enables:
- **Semantic search** via dense vector embeddings (NOMIC, CLIP)
- **Keyword search** via BM25
- **Primary source search** directly against transcription text
- **Filtered search** by shelf mark, author, institution, date range

---

## Neo4j Import

All Neo4j import scripts are **read-only with respect to the extraction pipeline** —
they consume JSON files produced by Passes 1–4 and write to the graph. They do not
feed back into the extraction pipeline.

```bash
# Backup before any destructive rebuild
python src/datasets/indexing/neo4j/backup_to_gcs.py

# Princeton PGP CSV (Fragment nodes + basic metadata)
python src/datasets/indexing/neo4j/knowledge_graph_poc.py

# Academic literature (entities + relations from Passes 2–4)
python src/datasets/indexing/neo4j/academic_kg_import.py

# Geocode Place nodes (runs after import)
python src/datasets/indexing/neo4j/geocode_places.py
```

---

## Running the Full Pipeline

```bash
# Step 0: OCR (skip if already digitized)
python src/models/ocr/book_ocr_service.py --input path/to/book.pdf

# Step 1: Structured extraction
python src/models/llm/structured_json_llm.py --dir academic_literature/my_book

# Step 2: Per-page entity tagging  (planned)
python src/models/llm/entity_tagger.py --dir academic_literature/my_book

# Step 3: Coreference resolution  (planned)
python src/models/llm/coreference_resolver.py --dir academic_literature/my_book

# Step 4: Relation extraction  (planned)
python src/models/llm/relation_extractor.py --dir academic_literature/my_book

# Index to Elasticsearch (can run after Step 1)
python src/datasets/indexing/elasticsearch/elastic_index_genizah.py

# Import to Neo4j (run after Steps 1–4 complete)
python src/datasets/indexing/neo4j/knowledge_graph_poc.py
python src/datasets/indexing/neo4j/academic_kg_import.py
python src/datasets/indexing/neo4j/geocode_places.py
```

---

## Legacy Scripts

The following scripts predate this architecture and are being phased out:

| Script | Issue | Replacement |
|---|---|---|
| `secondary_llm_processing.py` | Combined entity extraction, coreference, and KG triplets in one pass; document-level people with no page attribution | Pass 2 + Pass 3 |
| `enrich_node_relations.py` | Queried Neo4j for candidate entities, creating a DB dependency mid-pipeline | Pass 4 |
| `enrich_fragment_people.py` | Wrote Fragment→Person edges directly to Neo4j mid-pipeline | Pass 4 output → `academic_kg_import.py` |
| `biblio_import.py` | Deprecated; functionality merged into `academic_kg_import.py` | `academic_kg_import.py` |
