"""``academic_kg_import`` must stamp ``pipeline_version`` on what it creates.

Nothing in the graph used to record the extraction generation, so retiring
one generation's academic layer meant wiping the database. These tests pin
the contract: every node and relationship *created* by the Pass-4 importer
carries ``pipeline_version``, the value comes from the relations file (not
the code constant), and the stamp lives in ``ON CREATE SET`` so shared
PGP/biblio elements are never claimed by a generation.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from src.datasets.indexing.neo4j import academic_kg_import as aki


class _Result:
    """Stand-in for a neo4j ``Result`` whose lookups never match."""

    def single(self):
        return None


class FakeTx:
    """Records every ``run`` call so the Cypher and its parameters can be checked."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def run(self, query: str, **params: Any) -> _Result:
        self.calls.append((query, params))
        return _Result()


class FakeSession:
    def __init__(self, tx: FakeTx) -> None:
        self.tx = tx

    def execute_write(self, fn, *args):
        return fn(self.tx, *args)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeDriver:
    def __init__(self, tx: FakeTx) -> None:
        self.tx = tx

    def session(self, database: str = "neo4j") -> FakeSession:
        return FakeSession(self.tx)

    def close(self) -> None:
        pass


def _merge_calls(tx: FakeTx) -> List[Tuple[str, Dict[str, Any]]]:
    """Return only the calls that MERGE a node or relationship."""
    return [(q, p) for q, p in tx.calls if "MERGE" in q]


def _assert_stamp_on_create(query: str, params: Dict[str, Any], version: str) -> None:
    assert params.get("pipeline_version") == version, query
    assert "pipeline_version = $pipeline_version" in query, query
    # The stamp must sit inside ON CREATE SET, never a bare SET: a bare SET
    # would re-label pre-existing PGP/biblio elements as this generation.
    on_create = re.findall(r"ON CREATE SET(.*?)(?:\n\s*(?:SET|ON MATCH)\b|\"\"\"|$)",
                           query, flags=re.S)
    assert any("pipeline_version = $pipeline_version" in blk for blk in on_create), query
    for blk in re.findall(r"(?<!ON CREATE )(?<!ON MATCH )\bSET\b(.*?)(?=\n\s*(?:MERGE|MATCH|ON|SET)\b|\"\"\"|$)",
                          query, flags=re.S):
        assert "pipeline_version" not in blk, query


@pytest.fixture
def relations_file(tmp_path: Path) -> Path:
    book = tmp_path / "relations_v2" / "some_book" / "v2_tag"
    book.mkdir(parents=True)
    path = book / "book_relations.json"
    path.write_text(json.dumps({
        "source_book": "some_book",
        "pipeline_version": "v2",
        "relations": [
            {"subject": "Yosef the Trader", "subject_type": "Person",
             "relation": "TRAVELED_TO", "object": "Aden", "object_type": "Place",
             "evidence": "went to Aden", "evidence_page": 3, "confidence": "high"},
            {"subject": "T-S 12.345", "subject_type": "Fragment",
             "relation": "CITED_IN", "object": "Letters of Medieval Traders",
             "object_type": "BookArticle", "evidence": "cited", "evidence_page": None,
             "confidence": "medium"},
        ],
    }), encoding="utf-8")
    return path


def test_stamp_version_helper_binds_parameter():
    assert aki._stamp_version("n") == "n.pipeline_version = $pipeline_version"


def test_merge_person_stamps_version_on_create_only():
    tx = FakeTx()
    label, name = aki._merge_person(tx, "Yosef the Trader", "Person", "enriched",
                                    "some_book", pipeline_version="v9")
    assert label == "Person" and name
    merges = _merge_calls(tx)
    assert len(merges) == 1
    _assert_stamp_on_create(*merges[0], version="v9")


def test_merge_person_defaults_to_current_pipeline_version():
    tx = FakeTx()
    aki._merge_person(tx, "Yosef the Trader", "Person", "enriched", "some_book")
    (_, params), = _merge_calls(tx)
    assert params["pipeline_version"] == aki.PIPELINE_VERSION


def test_relations_import_stamps_every_created_element_with_file_version(relations_file):
    tx = FakeTx()
    importer = aki.AcademicKGImporter.__new__(aki.AcademicKGImporter)
    importer.driver = FakeDriver(tx)
    importer.database = "neo4j"

    counts = importer.import_relations_file(relations_file)
    assert counts["relations_written"] == 2

    merges = _merge_calls(tx)
    # Two relations → subject node, object node, relationship each.
    assert len(merges) == 6
    for query, params in merges:
        _assert_stamp_on_create(query, params, version="v2")

    # The file said v2 even though the code constant is newer.
    assert aki.PIPELINE_VERSION != "v2"
    rel_queries = [q for q, _ in merges if "-[r:" in q]
    assert len(rel_queries) == 2
    for q in rel_queries:
        assert "r.pipeline_version = $pipeline_version" in q


def test_relations_import_falls_back_to_pipeline_version_when_file_lacks_it(tmp_path):
    path = tmp_path / "book_relations.json"
    path.write_text(json.dumps({
        "source_book": "b",
        "relations": [{"subject": "A", "subject_type": "Person", "relation": "RELATED_TO",
                       "object": "B", "object_type": "Person", "evidence": "", "confidence": "high"}],
    }), encoding="utf-8")
    tx = FakeTx()
    importer = aki.AcademicKGImporter.__new__(aki.AcademicKGImporter)
    importer.driver = FakeDriver(tx)
    importer.database = "neo4j"
    importer.import_relations_file(path)
    for _, params in _merge_calls(tx):
        assert params["pipeline_version"] == aki.PIPELINE_VERSION
