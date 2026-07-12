# Knowledge Graph — Follow-up Work (next PR)

Deferred from the big pipeline-overhaul PR (new Pass 2–4 extraction, relation-quality
compaction, BiblicalPerson, cross-store `book_uuid`/`page_uuid`, `merged_shelfmarks_import`).
Those landed; the items below are the next round, ordered by impact. Each is a
**post-import** graph pass unless noted, so none disturb the extraction pipeline.

Schema reference for any of this: `docs/kg_schema.md`. Pipeline: `docs/secondary_source_pipeline.md`.

---

## 1. Place consolidation — highest impact (the map is broken without it)

**Problem.** Over half of all person→place edges point to **un-geocoded** places, so
journeys/homes don't render (e.g. the India-trade routes are all present as edges but
invisible):
- `TRAVELED_TO 378/677 (56%)`, `LIVED_IN 652/1189 (55%)`, `ORIGINATED_FROM 512/687 (75%)` → un-geocoded.

Three causes:
1. **Diacritic/spelling variant fragmentation.** The academic pipeline (`data_sources:
   enriched`) minted new Place nodes that duplicate geocoded Princeton nodes but differ
   by an apostrophe/macron/qualifier, so they never got coordinates:
   - `'Aydhāb` (enriched, no coords) vs `ʿAydhāb` (pgp, GEO)
   - `Qayrawan` vs `Qayrawān`; `Tripoli` vs `Tripoli (Ifrīqiya)`
   `PlaceNormalizer` did not fold these onto the canonical form.
2. **Regions/vague destinations** never in the Princeton gazetteer: `Gujarāt`, `Malabar`,
   `Indian Ocean`, `northern/western India`, `Morocco`, `Libya`, `Egypt`-as-country.
3. **Junk "places"**: role phrases extracted as places — `"residence of the bishop"`,
   `"gardens"`, `"her current place of residence"`, `"his residence (manzil)"`.

**Fix.** New `src/datasets/indexing/neo4j/consolidate_places.py` (post-import, run
read-only first to print merge candidates):
- For each un-geocoded place, match a geocoded place by **ascii-folded name** (strip
  diacritics, apostrophes/ʿayin, macrons, parenthetical qualifiers); on match, **redirect
  its edges to the canonical geocoded node** and delete the duplicate.
- Geocode the major regions (extend `geocode_places.py` `HISTORICAL_OVERRIDES` /
  region handling for Gujarat, Malabar, Indian Ocean, etc.).
- Drop vague "place" phrases (a stoplist + a "looks like a place" check).
- Reuse `PlaceNormalizer` and the `geocode_places` overrides.

**Context:** `geocode_places.py` only processes a curated ~511-place set (Princeton
gazetteer + overrides); the ~1,180 `enriched` places are out of its scope, which is why
they have no coordinates. Folding them onto canonical nodes is the leverage point.

---

## 2. Person consolidation

**Problem.** Important recurring people fragment into many nodes, splitting their
relations/routes:
- Joseph Lebdi → ~10 nodes (`Joseph Lebdi`, `Joseph b. David Lebdi`, `Lebdi`,
  `David Lebdi`, `Joseph b. David al-Lebdi`, …)
- `Ibn 'Awkal` → 6, `S. D. Goitein` → 3 (`Goitein`/`Shelomo Dov Goitein`/`S. D. Goitein`),
  `Udovitch` → 2.

Pass-3 LLM dedup misses complex Arabic/Hebrew name variants, and each book is resolved
independently so the same merchant isn't merged across the India-trader volumes.
`PersonNormalizer` only knows famous Gaonim/Maimonides, not these merchants.

**Fix.** New `src/datasets/indexing/neo4j/consolidate_people.py` (post-import,
read-only-first):
- Merge `Person`/`Scholar` nodes by **name-component overlap** (surname + matching
  given-name/patronymic), **not** surname alone — distinct people who share a surname must
  stay separate (`Abu 'l-Khayr b. Barhūn` ≠ `Abu 'l-Surür b. Barhūn`; `Jacob` vs `Joseph`
  Ibn ʿAwkal may differ).
- Bias to **high-degree (important) nodes** where fragmentation hurts most.
- LLM adjudication for ambiguous clusters (same conservative pattern as
  `biblical_person_classifier`), defaulting to keep-separate.
- Merge variants → one node, union edges + `name_variants`.

---

## 3. Generic-person / vague-place stoplist (quick)

Junk leaked as nodes: `"writer"`, `"the writer"`, `"Writer"`, `"the merchant"`, bare
single given-names without a patronymic (people); the vague "place" phrases above. The
Pass-4 compaction `_is_vague_entity` only covers fragment-ish words. Extend it with a
generic-person + generic-place stoplist (in `relationship_extractor.py` compaction, and/or
the consolidation passes). Low risk, removes obvious noise.

---

## 4. KTIV bibliography — scraper gap, then import

KTIV/NLI records have an **"FGP Catalogue record"** field carrying FJP-style bibliography
(scholarly editions), but the **KTIV scraper does not capture it**, so it's absent from
`merged_shelfmarks.jsonl`. Steps:
1. Fix the KTIV scraper to capture the "FGP Catalogue record" field.
2. Then extend `merged_shelfmarks_import.py` to import it as `BookArticle` + `REFERENCES`
   (it currently imports related people/places + catalog props only).

Note: KTIV `scholarly_entries.contributor` already names published catalogues (`Danzig
Catalog 1997`, `Halper Catalog 1924`, `Worman`, `Lutzki`, `Lieberman`, `Schocken/Zulay`).
Modeling those as `BookArticle -REFERENCES{reference_type:'catalogue'}-> Fragment` was
designed but **put on hold** pending the scraper fix. `biblio_import` stays the richest
citation source regardless (28.7k citations vs merged's 13.6k — do NOT retire it).

---

## 5. Map / frontend changes (cross-cutting — see `docs/kg_schema.md`)

Tracked for the frontend session, but listed here so the data work aligns:
- Map plots `LIVED_IN` as "home" and `TRAVELED_TO` as journeys; **`ORIGINATED_FROM` is not
  plotted** → add an origin layer (restores people like Rachel the Byzantine who have
  `ORIGINATED_FROM Constantinople` but no `LIVED_IN`).
- Query `(:Person OR :Scholar)`, exclude `BiblicalPerson`.
- Source filter must include `data_sources` `enriched` and `merged`.
- Filter map places on `lat IS NOT NULL`.

---

## 6. Legacy: `enrich_fragment_people.py`

Still reads legacy `*_enhanced.json`. Port to the new outputs (resolved entities +
structured pages) or retire — see `docs/secondary_source_pipeline.md` legacy table.

---

## Suggested PR slicing

- **PR A — Place consolidation + region geocoding** (#1): biggest visible win (lights up
  the map / trade routes).
- **PR B — Person consolidation** (#2) + generic stoplist (#3).
- **PR C — KTIV scraper + bibliography import** (#4): gated on the scraper fix.

Run #1 and #2 read-only first and review merge candidates before applying — both can
over-merge if naive.
