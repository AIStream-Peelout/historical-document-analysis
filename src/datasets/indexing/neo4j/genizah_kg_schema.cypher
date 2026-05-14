// ============================================================================
// Cairo Genizah Knowledge Graph Schema
// ============================================================================

// ============================================================================
// CONSTRAINTS AND INDEXES
// ============================================================================

CREATE CONSTRAINT fragment_shelfmark     IF NOT EXISTS FOR (f:Fragment)     REQUIRE f.canonical_shelfmark IS UNIQUE;
CREATE CONSTRAINT book_article_id        IF NOT EXISTS FOR (b:BookArticle)  REQUIRE b.article_id IS UNIQUE;
CREATE CONSTRAINT scholar_name           IF NOT EXISTS FOR (s:Scholar)      REQUIRE s.name IS UNIQUE;
CREATE CONSTRAINT place_name             IF NOT EXISTS FOR (pl:Place)       REQUIRE pl.name IS UNIQUE;
CREATE CONSTRAINT person_name            IF NOT EXISTS FOR (p:Person)       REQUIRE p.name IS UNIQUE;
CREATE CONSTRAINT institution_name       IF NOT EXISTS FOR (i:Institution)  REQUIRE i.name IS UNIQUE;

CREATE INDEX fragment_shelfmark_display  IF NOT EXISTS FOR (f:Fragment)    ON (f.shelfmark);
CREATE INDEX fragment_pgpid              IF NOT EXISTS FOR (f:Fragment)    ON (f.pgpid);
CREATE INDEX book_year                   IF NOT EXISTS FOR (b:BookArticle) ON (b.year);
CREATE INDEX scholar_position            IF NOT EXISTS FOR (s:Scholar)     ON (s.position);
CREATE INDEX person_period               IF NOT EXISTS FOR (p:Person)      ON (p.period);

// ============================================================================
// NODE: Fragment
//   A physical manuscript fragment held at an institution.
// ============================================================================
// Properties:
//   canonical_shelfmark  — normalised ID (e.g. T_S_AS_18_170)
//   shelfmark            — display form  (e.g. T-S AS 18.170)
//   ktiv_id              — KTIV / external catalogue ID
//   transcription        — full transcription text
//   translation          — English (or other) translation text
//   period               — date / period string (e.g. "11th century")
//   year_start / year_end — numeric year range when known
//   pgpid                — Princeton Geniza Project ID
//   iiif_url             — IIIF image manifest URL
//   url                  — human-readable catalogue URL

// ============================================================================
// NODE: BookArticle
//   A scholarly publication (book, journal article, chapter, etc.)
// ============================================================================
// Properties:
//   article_id  — deterministic ID: sha1(title|author|year)
//   title
//   year
//   publisher
//   language
//   journal     — journal / book series name if applicable
//   pages       — page range within volume
//   url

// ============================================================================
// NODE: Scholar
//   A modern researcher / author.
// ============================================================================
// Properties:
//   name
//   position    — academic position / affiliation

// ============================================================================
// NODE: Place
//   A geographic location.
// ============================================================================
// Properties:
//   name
//   latitude
//   longitude
//   region
//   country

// ============================================================================
// NODE: Person
//   A historical individual mentioned in or associated with fragments.
// ============================================================================
// Properties:
//   name
//   period      — estimated life period (e.g. "fl. 1050-1080")
//   dob_year    — estimated birth year (numeric)

// ============================================================================
// NODE: Institution
//   A library or archive that holds fragments.
// ============================================================================
// Properties:
//   name
//   collection  — primary collection name within institution
//   city
//   country

// ============================================================================
// RELATIONSHIPS
// ============================================================================
//
//   Scholar      -[:WROTE]->          BookArticle
//   BookArticle  -[:CITES]->          BookArticle
//   BookArticle  -[:REFERENCES {pages, mention_type, has_transcription,
//                               has_translation, location, doc_relation,
//                               notes, has_discussion,
//                               transcription_extent}]-> Fragment
//   Fragment     -[:MENTIONS]->       Person
//   Person       -[:LIVED_IN]->       Place
//   Fragment     -[:MENTIONS_PLACE]-> Place
//   Person       -[:TRAVELED_TO]->    Place
//   Person       -[:WROTE]->          Fragment
//   Fragment     -[:JOINED_WITH {confidence, notes}]-> Fragment
//   Fragment     -[:HELD_AT {sub_collection}]->        Institution
