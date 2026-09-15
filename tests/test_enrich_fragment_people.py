"""``enrich_fragment_people`` on Pass-3 resolved files.

The first dry-run on the v3 resolved files (2026-09-15) crashed on every
book that produced at least one edge: the edge dicts carry
``display_shelfmark`` but the dry-run/write-failure log lines read
``e['shelfmark']``. These tests pin the edge shape and exercise the dry-run
path end to end on a resolved file with real edges.
"""

import json
from pathlib import Path

import pytest

from src.datasets.indexing.neo4j import enrich_fragment_people as efp


RESOLVED = {
    "source_book": "some_book",
    "people": [
        {"name": "Joseph Lebdi", "pages": [4], "aliases": ["Lebdi"],
         "person_class": "historical"},
        {"name": "S. D. Goitein", "pages": [4], "aliases": [], "person_class": "scholar"},
        {"name": "Ben Yiju", "pages": [9], "aliases": [], "person_class": "historical"},
    ],
    "shelf_marks": [
        # Strategy 1: name in description, page-consistent.
        {"mark": "T-S 12.345", "pages": [4], "aliases": [],
         "description": "Letter of Joseph Lebdi from Aden", "mark_class": "standard"},
        # Strategy 2: sole historical person on the shared page.
        {"mark": "T-S 8J20.2", "pages": [9], "aliases": [], "description": "",
         "mark_class": "standard"},
    ],
}


def test_edges_carry_display_shelfmark_not_shelfmark():
    edges = efp._extract_edges_from_resolved(RESOLVED, "some_book")
    assert {e["certainty"] for e in edges} == {"definite", "possible"}
    for e in edges:
        assert set(e) == {"canonical_shelfmark", "display_shelfmark", "person_name",
                          "certainty", "evidence", "source_book"}
        assert "shelfmark" not in e
    names = {e["person_name"] for e in edges}
    assert names == {"Joseph Lebdi", "Ben Yiju"}   # the scholar is never linked


def test_dry_run_survives_a_file_with_edges(tmp_path: Path):
    path = tmp_path / "book_entities_resolved_v3_test.json"
    path.write_text(json.dumps(RESOLVED), encoding="utf-8")
    enricher = efp.FragmentPeopleEnricher.__new__(efp.FragmentPeopleEnricher)
    enricher.driver = None          # dry-run must never touch the driver
    enricher.database = "neo4j"
    counts = enricher.enrich_file(path, dry_run=True)
    assert counts == {"edges_definite": 1, "edges_possible": 1}
