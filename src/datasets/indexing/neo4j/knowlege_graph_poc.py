#!/usr/bin/env python3
"""
Princeton Genizah Project - Neo4j Knowledge Graph Import Script

This script ingests Princeton CSV data into Neo4j to create a comprehensive
knowledge graph of Cairo Genizah documents.

Shelfmarks are normalised via ShelfmarkNormalizer so every Fragment node
carries a stable canonical_shelfmark ID that can be matched across datasets
(Princeton CSV, Cambridge TEI, biblio.json, user queries, etc.).

Requirements:
    pip install neo4j pandas python-dotenv tqdm

Environment variables (.env file):
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your_password

CSV Files (place in ./data/ directory):
    - documents.csv
    - fragments.csv
    - people.csv
    - places.csv
    - sources.csv
    - footnotes.csv
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
from tqdm import tqdm

from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer
from src.datasets.document_models.institution_normalizer import InstitutionNormalizer
from src.datasets.document_models import biblical_person_classifier as bib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('genizah_import.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PrincetonGenizahKG:
    """Handler for Princeton Genizah Knowledge Graph in Neo4j"""

    def __init__(self, uri: str, user: str, password: str, data_dir: str = "./data", database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.data_dir = Path(data_dir)
        self.database = database

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        logger.info(f"Connected to Neo4j at {uri}")
        logger.info(f"Using database: {self.database}")
        logger.info(f"Using data directory: {self.data_dir}")

    def close(self):
        self.driver.close()
        logger.info("Closed Neo4j connection")

    def ensure_database_exists(self):
        """Create the target database if it doesn't already exist."""
        if self.database == "neo4j":
            return  # default db always exists
        logger.info(f"Ensuring database '{self.database}' exists...")
        with self.driver.session(database="system") as session:
            session.run(f"CREATE DATABASE `{self.database}` IF NOT EXISTS")
        logger.info(f"✓ Database '{self.database}' ready")

    def verify_csv_files(self) -> Dict[str, Path]:
        required_files = [
            'documents.csv',
            'fragments.csv',
            'people.csv',
            'places.csv',
            'sources.csv',
            'footnotes.csv',
        ]
        file_paths = {}
        missing_files = []
        for filename in required_files:
            filepath = self.data_dir / filename
            if filepath.exists():
                file_paths[filename] = filepath
                logger.info(f"✓ Found {filename}")
            else:
                missing_files.append(filename)
                logger.warning(f"✗ Missing {filename}")
        if missing_files:
            raise FileNotFoundError(f"Missing required CSV files: {', '.join(missing_files)}")
        return file_paths

    def create_constraints(self):
        """Create database constraints and indexes matching genizah_kg_schema.cypher"""
        logger.info("Creating constraints and indexes...")
        constraints = [
            # Unique constraints
            "CREATE CONSTRAINT fragment_canonical   IF NOT EXISTS FOR (f:Fragment)    REQUIRE f.canonical_shelfmark IS UNIQUE",
            "CREATE CONSTRAINT scholar_name         IF NOT EXISTS FOR (s:Scholar)     REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT book_article_id      IF NOT EXISTS FOR (b:BookArticle) REQUIRE b.article_id IS UNIQUE",
            "CREATE CONSTRAINT place_name           IF NOT EXISTS FOR (pl:Place)      REQUIRE pl.name IS UNIQUE",
            "CREATE CONSTRAINT person_name          IF NOT EXISTS FOR (p:Person)      REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT institution_name     IF NOT EXISTS FOR (i:Institution) REQUIRE i.name IS UNIQUE",
            # Indexes
            "CREATE INDEX fragment_shelfmark   IF NOT EXISTS FOR (f:Fragment)    ON (f.shelfmark)",
            "CREATE INDEX fragment_pgpid       IF NOT EXISTS FOR (f:Fragment)    ON (f.pgpid)",
            "CREATE INDEX book_year            IF NOT EXISTS FOR (b:BookArticle) ON (b.year)",
            "CREATE INDEX person_period        IF NOT EXISTS FOR (p:Person)      ON (p.period)",
            "CREATE INDEX scholar_position    IF NOT EXISTS FOR (s:Scholar)     ON (s.position)",
        ]
        with self.driver.session(database=self.database) as session:
            for constraint in tqdm(constraints, desc="Creating constraints"):
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.warning(f"Constraint/index warning: {e}")
        logger.info("✓ Constraints and indexes created")

    # ------------------------------------------------------------------
    # Documents → Fragment nodes
    # ------------------------------------------------------------------
    def import_documents(self, df: pd.DataFrame):
        logger.info(f"Importing {len(df)} documents as Fragment nodes...")

        def process_batch(tx, batch):
            for _, row in batch.iterrows():
                doc_data = row.where(pd.notnull(row), None).to_dict()
                shelfmark = doc_data.get('shelfmark') or ''
                canonical = ShelfmarkNormalizer.to_canonical_id(shelfmark)

                query = """
                MERGE (f:Fragment {canonical_shelfmark: $canonical_shelfmark})
                SET f.shelfmark            = $shelfmark,
                    f.pgpid                = $pgpid,
                    f.url                  = $url,
                    f.type                 = $type,
                    f.description          = $description,
                    f.period               = $inferred_date_display,
                    f.doc_date_original    = $doc_date_original,
                    f.doc_date_standard    = $doc_date_standard,
                    f.inferred_date_standard = $inferred_date_standard,
                    f.language_note        = $language_note,
                    f.has_transcription    = $has_transcription,
                    f.has_translation      = $has_translation,
                    f.region               = $region,
                    f.side                 = $side,
                    f.data_sources         = CASE
                        WHEN 'pgp' IN coalesce(f.data_sources, []) THEN coalesce(f.data_sources, [])
                        ELSE coalesce(f.data_sources, []) + ['pgp']
                    END

                WITH f
                WHERE $tags IS NOT NULL
                WITH f, split($tags, ', ') AS tag_list
                UNWIND tag_list AS tag_name
                WITH f, trim(tag_name) AS tag_name
                WHERE tag_name <> ''
                MERGE (t:Tag {name: tag_name})
                MERGE (f)-[:HAS_TAG]->(t)

                WITH f
                WHERE $languages_primary IS NOT NULL
                WITH f, split($languages_primary, ',') AS primary_langs
                UNWIND primary_langs AS lang_name
                WITH f, trim(lang_name) AS lang_name
                WHERE lang_name <> ''
                MERGE (l:Language {name: lang_name})
                MERGE (f)-[:WRITTEN_IN {primary: true}]->(l)
                """

                params = {
                    'canonical_shelfmark': canonical,
                    'shelfmark':           shelfmark,
                    'pgpid':               int(doc_data['pgpid']) if doc_data.get('pgpid') else None,
                    'url':                 doc_data.get('url'),
                    'type':                doc_data.get('type'),
                    'description':         doc_data.get('description'),
                    'doc_date_original':   doc_data.get('doc_date_original'),
                    'doc_date_standard':   doc_data.get('doc_date_standard'),
                    'inferred_date_display': doc_data.get('inferred_date_display'),
                    'inferred_date_standard': doc_data.get('inferred_date_standard'),
                    'language_note':       doc_data.get('language_note'),
                    'tags':                doc_data.get('tags'),
                    'languages_primary':   doc_data.get('languages_primary'),
                    'has_transcription':   doc_data.get('has_transcription') == 'Y',
                    'has_translation':     doc_data.get('has_translation') == 'Y',
                    'region':              doc_data.get('region'),
                    'side':                doc_data.get('side'),
                }
                tx.run(query, **params)

                # Fragment → Place relationships from place columns
                # mentioned / possibly_mentioned are comma-separated place names
                place_links = [
                    ('mentioned',           'MENTIONS_PLACE',    {}),
                    ('possibly_mentioned',  'MENTIONS_PLACE',    {'certain': False}),
                    ('destination',         'SENT_TO',           {}),
                    ('location',            'WRITTEN_AT',        {}),
                    ('origin',              'ORIGINATED_FROM',   {}),
                ]
                for col, rel_type, rel_props in place_links:
                    raw = doc_data.get(col)
                    if not raw:
                        continue
                    # Some columns are single values, others comma-separated lists
                    place_names = [p.strip() for p in str(raw).split(',') if p.strip()]
                    for place_name in place_names:
                        tx.run(
                            f"""
                            MATCH (f:Fragment {{canonical_shelfmark: $csm}})
                            MERGE (pl:Place {{name: $place}})
                            SET pl.place_type   = 'historical',
                                pl.data_sources = CASE
                                WHEN 'pgp' IN coalesce(pl.data_sources, []) THEN coalesce(pl.data_sources, [])
                                ELSE coalesce(pl.data_sources, []) + ['pgp']
                            END
                            MERGE (f)-[r:{rel_type}]->(pl)
                            SET r += $rel_props
                            """,
                            csm=canonical,
                            place=place_name,
                            rel_props=rel_props,
                        )

        batch_size = 100
        with self.driver.session(database=self.database) as session:
            for i in tqdm(range(0, len(df), batch_size), desc="Importing fragments"):
                session.execute_write(process_batch, df.iloc[i:i + batch_size])

        logger.info("✓ Fragments imported")

    # ------------------------------------------------------------------
    # Fragments CSV → HELD_AT Institution (via ShelfmarkNormalizer)
    # ------------------------------------------------------------------
    def import_fragments(self, df: pd.DataFrame):
        logger.info(f"Linking {len(df)} fragments to institutions...")

        def process_batch(tx, batch):
            for _, row in batch.iterrows():
                doc_data = row.where(pd.notnull(row), None).to_dict()
                shelfmark = doc_data.get('shelfmark') or ''
                canonical = ShelfmarkNormalizer.to_canonical_id(shelfmark)
                inst_info = ShelfmarkNormalizer.get_institution_info(shelfmark)
                institution = inst_info.get('institution') or doc_data.get('library')
                # Canonicalise before it ever becomes a MERGE key. Without this,
                # this PGP-CSV path creates its own un-normalized Institution
                # node ("Jewish Theological Seminary Library") alongside the
                # canonical one other pipelines create via InstitutionNormalizer
                # ("Jewish Theological Seminary") -- same real institution, two
                # nodes, "circles inside circles" duplicate map pins.
                if institution:
                    institution = InstitutionNormalizer.normalize(institution)
                collection  = inst_info.get('collection')
                subcollection = inst_info.get('subcollection')

                # Ensure the Fragment node exists (may have been created in import_documents)
                base_query = """
                MERGE (f:Fragment {canonical_shelfmark: $canonical_shelfmark})
                ON CREATE SET f.shelfmark = $shelfmark
                SET f.is_multifragment = $is_multifragment,
                    f.url              = $url,
                    f.iiif_url         = $iiif_url,
                    f.collection_name  = $collection_name,
                    f.provenance_display = $provenance_display,
                    f.data_sources     = CASE
                        WHEN 'pgp' IN coalesce(f.data_sources, []) THEN coalesce(f.data_sources, [])
                        ELSE coalesce(f.data_sources, []) + ['pgp']
                    END
                """
                tx.run(base_query, {
                    'canonical_shelfmark': canonical,
                    'shelfmark':           shelfmark,
                    'is_multifragment':    doc_data.get('is_multifragment') == 'Y',
                    'url':                 doc_data.get('url'),
                    'iiif_url':            doc_data.get('iiif_url'),
                    'collection_name':     doc_data.get('collection_name'),
                    'provenance_display':  doc_data.get('provenance_display'),
                })

                # Link to Institution
                if institution:
                    inst_query = """
                    MERGE (f:Fragment {canonical_shelfmark: $canonical_shelfmark})
                    MERGE (i:Institution {name: $institution})
                    ON CREATE SET i.collection = $collection
                    SET i.data_sources = CASE
                        WHEN 'pgp' IN coalesce(i.data_sources, []) THEN coalesce(i.data_sources, [])
                        ELSE coalesce(i.data_sources, []) + ['pgp']
                    END
                    MERGE (f)-[r:HELD_AT]->(i)
                    SET r.sub_collection = $subcollection,
                        r.data_sources   = CASE
                            WHEN 'pgp' IN coalesce(r.data_sources, []) THEN coalesce(r.data_sources, [])
                            ELSE coalesce(r.data_sources, []) + ['pgp']
                        END
                    """
                    tx.run(inst_query, {
                        'canonical_shelfmark': canonical,
                        'institution':         institution,
                        'collection':          collection,
                        'subcollection':       subcollection,
                    })

                # Link to documents via PGPIDs
                pgpids = doc_data.get('pgpids')
                if pgpids:
                    join_query = """
                    MERGE (f:Fragment {canonical_shelfmark: $canonical_shelfmark})
                    WITH f, split($pgpids, ';') AS pgpid_list
                    UNWIND pgpid_list AS pgpid_str
                    WITH f, trim(pgpid_str) AS pgpid_str
                    WHERE pgpid_str <> ''
                    MATCH (d:Fragment {pgpid: toInteger(pgpid_str)})
                    WHERE d <> f
                    MERGE (f)-[:JOINED_WITH]->(d)
                    """
                    tx.run(join_query, {
                        'canonical_shelfmark': canonical,
                        'pgpids': pgpids,
                    })

        batch_size = 100
        with self.driver.session(database=self.database) as session:
            for i in tqdm(range(0, len(df), batch_size), desc="Linking institutions"):
                session.execute_write(process_batch, df.iloc[i:i + batch_size])

        logger.info("✓ Fragments linked to institutions")

    # ------------------------------------------------------------------
    # People → Person nodes
    # ------------------------------------------------------------------
    def import_people(self, df: pd.DataFrame):
        logger.info(f"Importing {len(df)} people...")

        def process_batch(tx, batch):
            for _, row in batch.iterrows():
                doc_data = row.where(pd.notnull(row), None).to_dict()
                name = doc_data['name']

                # Create / update Person node
                tx.run("""
                MERGE (p:Person {name: $name})
                SET p.name_variants           = $name_variants,
                    p.gender                  = $gender,
                    p.social_roles            = $social_roles,
                    p.description             = $description,
                    p.home_base               = $home_base,
                    p.related_documents_count = $related_documents_count,
                    p.url                     = $url,
                    p.data_sources            = CASE
                        WHEN 'pgp' IN coalesce(p.data_sources, []) THEN coalesce(p.data_sources, [])
                        ELSE coalesce(p.data_sources, []) + ['pgp']
                    END
                """, {
                    'name':                    name,
                    'name_variants':           doc_data.get('name_variants'),
                    'gender':                  doc_data.get('gender'),
                    'social_roles':            doc_data.get('social_roles'),
                    'description':             doc_data.get('description'),
                    'home_base':               doc_data.get('home_base'),
                    'related_documents_count': doc_data.get('related_documents_count'),
                    'url':                     doc_data.get('url'),
                })

                # Person -[:LIVED_IN]-> Place  (home_base is a single value)
                if doc_data.get('home_base'):
                    tx.run("""
                        MERGE (p:Person {name: $name})
                        MERGE (pl:Place {name: $place})
                        SET pl.place_type   = 'historical',
                            pl.data_sources = CASE
                            WHEN 'pgp' IN coalesce(pl.data_sources, []) THEN coalesce(pl.data_sources, [])
                            ELSE coalesce(pl.data_sources, []) + ['pgp']
                        END
                        MERGE (p)-[r:LIVED_IN]->(pl)
                        SET r.data_sources = CASE
                            WHEN 'pgp' IN coalesce(r.data_sources, []) THEN coalesce(r.data_sources, [])
                            ELSE coalesce(r.data_sources, []) + ['pgp']
                        END
                    """, name=name, place=doc_data['home_base'].strip())

                # Person -[:TRAVELED_TO]-> Place  (traveled_to is comma-separated)
                if doc_data.get('traveled_to'):
                    for place in str(doc_data['traveled_to']).split(','):
                        place = place.strip()
                        if place:
                            tx.run("""
                                MERGE (p:Person {name: $name})
                                MERGE (pl:Place {name: $place})
                                SET pl.place_type   = 'historical',
                                    pl.data_sources = CASE
                                    WHEN 'pgp' IN coalesce(pl.data_sources, []) THEN coalesce(pl.data_sources, [])
                                    ELSE coalesce(pl.data_sources, []) + ['pgp']
                                END
                                MERGE (p)-[r:TRAVELED_TO]->(pl)
                                SET r.data_sources = CASE
                                    WHEN 'pgp' IN coalesce(r.data_sources, []) THEN coalesce(r.data_sources, [])
                                    ELSE coalesce(r.data_sources, []) + ['pgp']
                                END
                            """, name=name, place=place)

        batch_size = 100
        with self.driver.session(database=self.database) as session:
            for i in tqdm(range(0, len(df), batch_size), desc="Importing people"):
                session.execute_write(process_batch, df.iloc[i:i + batch_size])

        logger.info("✓ People imported")

    # ------------------------------------------------------------------
    # Places → Place nodes
    # ------------------------------------------------------------------
    def import_places(self, df: pd.DataFrame):
        logger.info(f"Importing {len(df)} places...")

        def process_batch(tx, batch):
            for _, row in batch.iterrows():
                doc_data = row.where(pd.notnull(row), None).to_dict()
                query = """
                MERGE (pl:Place {name: $name})
                SET pl.name_variants           = $name_variants,
                    pl.is_region               = $is_region,
                    pl.coordinates             = $coordinates,
                    pl.geographic_area         = $geographic_area,
                    pl.notes                   = $notes,
                    pl.related_documents_count = $related_documents_count,
                    pl.url                     = $url,
                    pl.place_type              = 'historical',
                    pl.data_sources            = CASE
                        WHEN 'pgp' IN coalesce(pl.data_sources, []) THEN coalesce(pl.data_sources, [])
                        ELSE coalesce(pl.data_sources, []) + ['pgp']
                    END
                """
                tx.run(query, {
                    'name':                    doc_data['name'],
                    'name_variants':           doc_data.get('name_variants'),
                    'is_region':               doc_data.get('is_region') == 'Y',
                    'coordinates':             doc_data.get('coordinates'),
                    'geographic_area':         doc_data.get('geographic_area'),
                    'notes':                   doc_data.get('notes'),
                    'related_documents_count': doc_data.get('related_documents_count'),
                    'url':                     doc_data.get('url'),
                })

        batch_size = 100
        with self.driver.session(database=self.database) as session:
            for i in tqdm(range(0, len(df), batch_size), desc="Importing places"):
                session.execute_write(process_batch, df.iloc[i:i + batch_size])

        logger.info("✓ Places imported")

    # ------------------------------------------------------------------
    # Sources → BookArticle + Scholar nodes
    # ------------------------------------------------------------------
    def import_sources(self, df: pd.DataFrame):
        logger.info(f"Importing {len(df)} sources as BookArticle nodes...")

        def process_batch(tx, batch):
            for _, row in batch.iterrows():
                doc_data = row.where(pd.notnull(row), None).to_dict()
                citation = doc_data.get('citation', '')
                article_id = _make_article_id(
                    doc_data.get('title', citation),
                    doc_data.get('authors', ''),
                    str(doc_data.get('year', '')),
                )
                # Drop known pre-modern historical authors (Maimonides,
                # Saadia Gaon, ...) from the Scholar-creation list below —
                # PGP bibliography "authors" is almost always modern
                # secondary scholarship, but a primary-source edition/
                # translation entry could legitimately list one of them,
                # and WROTE->BookArticle would otherwise mislabel them
                # Scholar the same way the academic-literature pipeline once
                # did (see biblical_person_classifier.is_known_historical_author).
                raw_authors = doc_data.get('authors', '') or ''
                authors = ';'.join(
                    a for a in raw_authors.split(';')
                    if a.strip() and not bib.is_known_historical_author(a.strip())
                ) or None
                query = """
                MERGE (b:BookArticle {article_id: $article_id})
                SET b.title        = $title,
                    b.year         = $year,
                    b.publisher    = $publisher,
                    b.journal      = $journal_book,
                    b.volume       = $volume,
                    b.pages        = $page_range,
                    b.url          = $url,
                    b.citation     = $citation,
                    b.source_type  = $source_type,
                    b.data_sources = CASE
                        WHEN 'pgp' IN coalesce(b.data_sources, []) THEN coalesce(b.data_sources, [])
                        ELSE coalesce(b.data_sources, []) + ['pgp']
                    END

                WITH b
                WHERE $authors IS NOT NULL
                WITH b, split($authors, ';') AS author_list
                UNWIND author_list AS author_name
                WITH b, trim(author_name) AS author_name
                WHERE author_name <> ''
                MERGE (s:Scholar {name: author_name})
                SET s.data_sources = CASE
                    WHEN 'pgp' IN coalesce(s.data_sources, []) THEN coalesce(s.data_sources, [])
                    ELSE coalesce(s.data_sources, []) + ['pgp']
                END
                MERGE (s)-[r:WROTE]->(b)
                SET r.data_sources = CASE
                    WHEN 'pgp' IN coalesce(r.data_sources, []) THEN coalesce(r.data_sources, [])
                    ELSE coalesce(r.data_sources, []) + ['pgp']
                END
                """
                tx.run(query, {
                    'article_id':   article_id,
                    'title':        doc_data.get('title', citation),
                    'year':         doc_data.get('year'),
                    'publisher':    doc_data.get('publisher'),
                    'journal_book': doc_data.get('journal_book'),
                    'volume':       doc_data.get('volume'),
                    'page_range':   doc_data.get('page_range'),
                    'url':          doc_data.get('url'),
                    'citation':     citation,
                    'source_type':  doc_data.get('source_type'),
                    'authors':      authors,
                })

        batch_size = 100
        with self.driver.session(database=self.database) as session:
            for i in tqdm(range(0, len(df), batch_size), desc="Importing sources"):
                session.execute_write(process_batch, df.iloc[i:i + batch_size])

        logger.info("✓ Sources imported")

    # ------------------------------------------------------------------
    # Footnotes → BookArticle -[:REFERENCES]-> Fragment
    # ------------------------------------------------------------------
    def import_footnotes(self, df: pd.DataFrame):
        logger.info(f"Importing {len(df)} footnote relationships...")

        def process_batch(tx, batch):
            for _, row in batch.iterrows():
                doc_data = row.where(pd.notnull(row), None).to_dict()
                document_id = doc_data.get('document_id')
                source      = doc_data.get('source')
                if document_id is None or source is None:
                    continue
                query = """
                MATCH (f:Fragment {pgpid: $document_id})
                MATCH (b:BookArticle {citation: $source})
                MERGE (b)-[r:REFERENCES]->(f)
                SET r.location     = $location,
                    r.doc_relation = $doc_relation,
                    r.notes        = $notes,
                    r.data_sources = CASE
                        WHEN 'pgp' IN coalesce(r.data_sources, []) THEN coalesce(r.data_sources, [])
                        ELSE coalesce(r.data_sources, []) + ['pgp']
                    END
                """
                try:
                    tx.run(query, {
                        'document_id': int(document_id),
                        'source':      source,
                        'location':    doc_data.get('location'),
                        'doc_relation': doc_data.get('doc_relation'),
                        'notes':       doc_data.get('notes'),
                    })
                except Exception as e:
                    logger.debug(f"Footnote skip: {e}")

        batch_size = 100
        with self.driver.session(database=self.database) as session:
            for i in tqdm(range(0, len(df), batch_size), desc="Importing footnotes"):
                session.execute_write(process_batch, df.iloc[i:i + batch_size])

        logger.info("✓ Footnotes imported")

    # ------------------------------------------------------------------
    # Document → Person edges (mentioned / possibly_mentioned)
    # ------------------------------------------------------------------
    def import_document_people(self, df: pd.DataFrame):
        """Create Fragment -[:MENTIONS_PERSON]-> Person edges from documents.csv.

        The `mentioned` column contains comma-separated person names who appear
        in a document. `possibly_mentioned` is the same but with lower certainty.
        Both are imported; the relationship carries a `certainty` property so
        queries can filter by confidence level.

        Note: only `mentioned` and `possibly_mentioned` columns are currently
        processed. The `destination` / `origin` columns (SENT_TO / AUTHORED_BY)
        are not yet implemented here.
        """
        logger.info("Importing document–person relationships...")

        # Build a set of known Person names for quick lookup so we don't create
        # spurious Person nodes from what are actually place names in these columns.
        with self.driver.session(database=self.database) as session:
            known_people = {
                r['name'] for r in
                session.run("MATCH (p:Person) RETURN p.name AS name").data()
                if r['name']
            }

        logger.info(f"  Known Person nodes for matching: {len(known_people):,}")

        def process_batch(tx, batch):
            for _, row in batch.iterrows():
                doc_data   = row.where(pd.notnull(row), None).to_dict()
                pgpid      = doc_data.get('pgpid')
                shelfmark  = doc_data.get('shelfmark', '')
                if not pgpid:
                    continue

                # Parse and deduplicate a semicolon/comma-separated name field
                def _names(field):
                    raw = doc_data.get(field) or ''
                    names = [n.strip() for n in raw.replace(';', ',').split(',') if n.strip()]
                    return list(dict.fromkeys(names))

                # Fragment -[:MENTIONS_PERSON {certainty:'definite'}]-> Person
                for person_name in _names('mentioned'):
                    if person_name not in known_people:
                        continue
                    tx.run("""
                        MATCH (f:Fragment {pgpid: $pgpid})
                        MATCH (p:Person   {name:  $name})
                        MERGE (f)-[r:MENTIONS_PERSON]->(p)
                        SET r.certainty   = 'definite',
                            r.data_sources = CASE
                                WHEN 'pgp' IN coalesce(r.data_sources, []) THEN coalesce(r.data_sources, [])
                                ELSE coalesce(r.data_sources, []) + ['pgp']
                            END
                    """, pgpid=int(pgpid), name=person_name)

                # Fragment -[:MENTIONS_PERSON {certainty:'possible'}]-> Person
                for person_name in _names('possibly_mentioned'):
                    if person_name not in known_people:
                        continue
                    tx.run("""
                        MATCH (f:Fragment {pgpid: $pgpid})
                        MATCH (p:Person   {name:  $name})
                        MERGE (f)-[r:MENTIONS_PERSON]->(p)
                        ON CREATE SET r.certainty = 'possible'
                        SET r.data_sources = CASE
                                WHEN 'pgp' IN coalesce(r.data_sources, []) THEN coalesce(r.data_sources, [])
                                ELSE coalesce(r.data_sources, []) + ['pgp']
                            END
                    """, pgpid=int(pgpid), name=person_name)

        batch_size = 200
        with self.driver.session(database=self.database) as session:
            for i in tqdm(range(0, len(df), batch_size), desc="Importing doc–person edges"):
                session.execute_write(process_batch, df.iloc[i:i + batch_size])

        # Report how many people got connected
        with self.driver.session(database=self.database) as session:
            stats = session.run("""
                MATCH (:Fragment)-[r:MENTIONS_PERSON]->(:Person)
                RETURN r.certainty AS certainty, count(*) AS cnt
                ORDER BY certainty
            """).data()
            for s in stats:
                logger.info(f"  MENTIONS_PERSON [{s['certainty']}]: {s['cnt']:,} edges")

        logger.info("✓ Document–person relationships imported")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def run_import(self):
        logger.info("=" * 60)
        logger.info("PRINCETON GENIZAH PROJECT - NEO4J IMPORT")
        logger.info("=" * 60)

        try:
            file_paths = self.verify_csv_files()
            self.create_constraints()

            logger.info("\nLoading CSV files...")
            documents_df = pd.read_csv(file_paths['documents.csv'])
            self.import_documents(documents_df)
            self.import_fragments(pd.read_csv(file_paths['fragments.csv']))
            self.import_people(pd.read_csv(file_paths['people.csv']))
            self.import_places(pd.read_csv(file_paths['places.csv']))
            self.import_sources(pd.read_csv(file_paths['sources.csv']))
            self.import_footnotes(pd.read_csv(file_paths['footnotes.csv']))
            # Must run after both people and fragments are imported
            self.import_document_people(documents_df)

            logger.info("\n" + "=" * 60)
            logger.info("✓ IMPORT COMPLETE!")
            logger.info("=" * 60)
            self.print_statistics()

        except Exception as e:
            logger.error(f"Import failed: {e}", exc_info=True)
            raise

    def print_statistics(self):
        logger.info("\nKnowledge Graph Statistics:")
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] AS label, count(*) AS count
                ORDER BY count DESC
            """)
            print("\nNode Counts:")
            for record in result:
                print(f"  {record['label']}: {record['count']:,}")

            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS relationship, count(*) AS count
                ORDER BY count DESC
            """)
            print("\nRelationship Counts:")
            for record in result:
                print(f"  {record['relationship']}: {record['count']:,}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_article_id(title: str, author: str, year: str) -> str:
    """Deterministic ID for a BookArticle node."""
    import hashlib
    key = f"{title}|{author}|{year}".lower().strip()
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def main():
    load_dotenv()
    neo4j_uri      = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user     = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    neo4j_database = os.getenv('NEO4J_DATABASE', 'genizah-prod')
    data_dir       = os.getenv('DATA_DIR', '/Users/isaac/Documents/GitHub/historical-document-analysis/src/datasets/raw_data/cairo_genizah/pgp_raw/data')

    if not neo4j_password:
        logger.error("NEO4J_PASSWORD environment variable not set")
        sys.exit(1)

    kg = PrincetonGenizahKG(neo4j_uri, neo4j_user, neo4j_password, data_dir, database=neo4j_database)
    try:
        kg.ensure_database_exists()
        kg.run_import()
    finally:
        kg.close()


if __name__ == "__main__":
    main()
