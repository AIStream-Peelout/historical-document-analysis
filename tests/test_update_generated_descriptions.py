# File name: test_update_generated_descriptions.py
# Date: 10/8/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Writing generated descriptions into the index: writer guard, canary, refusals, field set, dry run, resume.

No network and no model: Elasticsearch, the bulk helper and the embedder are faked. No ``.env`` is read.
"""
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest

from src.datasets.consensus import describe_from_reads as dfr
from src.datasets.indexing import update_generated_descriptions as ugd
from src.datasets.indexing import verify_reembedded_indexes as vri
from src.datasets.indexing.elastic_index_genizah import IndexMetaMismatchError
from src.embeddings import qwen_text_embedding as qte
from src.embeddings.text_representation import embed_text_for_version, is_text_eligible, text_for_version

DIMS = 8
INDEX = "genizah_merged_v9_test"
CANARY = "canary string"
TRANSCRIPTION = "ליוסף אחי יקירי שלום\nשלחתי לך פלפל ושמן בספינה\n[...] בפסטאט בחדש ניסן"


def fake_vector(text: str, salt: str = "") -> np.ndarray:
    """Deterministic unit vector (a different ``salt`` stands for a different model).

    :param text: Text.
    :param salt: Model stand-in.
    :return: Unit vector of :data:`DIMS` floats.
    """
    seed = int(hashlib.sha256((salt + text).encode("utf-8")).hexdigest()[:8], 16)
    vector = np.random.default_rng(seed).normal(size=DIMS).astype(np.float32)
    return vector / np.linalg.norm(vector)


class FakeEmbedder:
    """Stands in for QwenTextEmbedding; records every text it embeds."""

    def __init__(self, salt: str = "", revision: str = qte.FINETUNED_MODEL_REVISION) -> None:
        """:param salt: Vector-space stand-in.
        :param revision: Reported revision.
        """
        self.model_name = qte.FINETUNED_MODEL_NAME
        self.revision = revision
        self.salt = salt
        self.documents: List[str] = []

    def get_text_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """:param texts: Texts (an empty one is a test failure).
        :return: (n, DIMS) array.
        """
        assert texts and all(t.strip() for t in texts), "an empty string reached the embedder"
        self.documents.extend(texts)
        return np.stack([fake_vector(t, self.salt) for t in texts])


def matches(doc: Dict[str, Any], query: Optional[Dict[str, Any]]) -> bool:
    """Evaluate the verifier's query shapes (term, exists, bool filter / must / must_not).

    :param doc: ``_source``.
    :param query: Query DSL (None = match all).
    :return: Whether the document matches.
    """
    if not query:
        return True
    if "term" in query:
        ((name, value),) = query["term"].items()
        return doc.get(name) == value
    if "exists" in query:
        return doc.get(query["exists"]["field"]) is not None
    clause = query["bool"]
    return (all(matches(doc, q) for q in clause.get("filter", []) + clause.get("must", []))
            and not any(matches(doc, q) for q in clause.get("must_not", [])))


class FakeIndices:
    """``es.indices``: mappings by index, and a record of ``put_mapping`` calls."""

    def __init__(self) -> None:
        """Start with no index."""
        self.mappings: Dict[str, Dict[str, Any]] = {}
        self.put: List[Dict[str, Any]] = []

    def exists(self, index: str) -> bool:
        """:param index: Index.
        :return: Whether it exists.
        """
        return index in self.mappings

    def get_mapping(self, index: str) -> Dict[str, Any]:
        """:param index: Index.
        :return: ``{index: {"mappings": ...}}``.
        """
        return {index: {"mappings": copy.deepcopy(self.mappings[index])}}

    def put_mapping(self, index: str, properties: Dict[str, Any]) -> None:
        """:param index: Index.
        :param properties: New fields (``_meta`` is not touched, as in Elasticsearch).
        """
        self.put.append(properties)
        self.mappings[index].setdefault("properties", {}).update(properties)


class FakeES:
    """In-memory index: documents by id, ``mget`` and ``count``."""

    def __init__(self) -> None:
        """Start empty."""
        self.indices = FakeIndices()
        self.docs: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.bulk_calls: List[List[Dict[str, Any]]] = []

    def mget(self, index: str, ids: Sequence[str], source_includes: Optional[List[str]] = None) -> Dict[str, Any]:
        """:param index: Index.
        :param ids: Ids.
        :param source_includes: Ignored (whole ``_source`` returned).
        :return: ``{"docs": [...]}``.
        """
        store = self.docs.get(index, {})
        return {"docs": [{"_id": i, "found": i in store, **({"_source": copy.deepcopy(store[i])} if i in store else {})}
                         for i in ids]}

    def count(self, index: str, query: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
        """:param index: Index.
        :param query: Query DSL.
        :return: ``{"count": n}``.
        """
        return {"count": sum(matches(d, query) for d in self.docs.get(index, {}).values())}


@pytest.fixture
def es(monkeypatch: pytest.MonkeyPatch) -> FakeES:
    """A v9-style index with a vector-less document, a catalogue-vector document and one more vector-less one.

    :param monkeypatch: pytest monkeypatch (routes the module's bulk helper to the fake).
    :return: The fake cluster.
    """
    cluster = FakeES()
    cluster.indices.mappings[INDEX] = {"_meta": index_meta(FakeEmbedder()),
                                       "properties": {"embedding_vector": {"type": "dense_vector", "dims": DIMS}}}
    cluster.docs[INDEX] = {
        "Doc_A": {"doc_id": "Doc_A", "semantic_eligible": False, "text_representation_version": "semantic_v3"},
        "Doc_B": {"doc_id": "Doc_B", "semantic_eligible": True, "text_representation_version": "semantic_v3",
                  "embedding_vector": fake_vector("catalogue").tolist()},
        "Doc_D": {"doc_id": "Doc_D", "semantic_eligible": False, "text_representation_version": "semantic_v3"},
    }

    def fake_bulk(client: FakeES, actions: List[Dict[str, Any]], refresh: str) -> tuple:
        """:param client: Fake cluster.
        :param actions: Bulk update actions.
        :param refresh: Ignored.
        :return: (successes, failures).
        """
        client.bulk_calls.append(actions)
        for action in actions:
            assert action["_op_type"] == "update"
            client.docs[action["_index"]][action["_id"]].update(copy.deepcopy(action["doc"]))
        return len(actions), []

    monkeypatch.setattr(ugd, "bulk", fake_bulk)
    return cluster


def index_meta(embedder: FakeEmbedder, text_version: str = "semantic_v3", **overrides: Any) -> Dict[str, Any]:
    """The ``_meta`` an index built with ``embedder`` carries.

    :param embedder: Embedder the index was built with.
    :param text_version: Text version.
    :param overrides: Keys to replace.
    :return: ``_meta``.
    """
    meta = {"model_name": embedder.model_name, "model_revision": embedder.revision,
            "text_representation_version": text_version, "dims": DIMS,
            "canary": {"string": CANARY, "vector": fake_vector(CANARY, embedder.salt).tolist()}}
    meta.update(overrides)
    return meta


def gen_row(doc_id: str, summary: Optional[str] = None, generated_at: str = "2026-10-08T12:00:00+00:00",
            source: str = dfr.DESCRIPTION_SOURCE, confidence: float = 0.8) -> Dict[str, Any]:
    """A generator row built with the generator's own functions.

    :param doc_id: Document.
    :param summary: Summary (default: a real two-sentence summary).
    :param generated_at: Generation time.
    :param source: ``description_source``.
    :param confidence: Model confidence.
    :return: Row.
    """
    reply = {"genre": "documentary", "document_type": "Letter", "language": ["Judaeo-Arabic"], "script": "Hebrew",
             "summary": summary if summary is not None else
             f"Letter ({doc_id}) to Yosef about a shipment of pepper and oil sent by ship. It mentions Fustat.",
             "identified_work": None,
             "entities": {"people": ["יוסף"], "places": ["פסטאט"], "dates": [], "commodities": ["פלפל", "קנמון"],
                          "institutions": []},
             "themes": ["trade"], "confidence": confidence, "notes": ""}
    report = dfr.grounding_check(reply, TRANSCRIPTION)
    grounded = report.grounded_json
    eligible, reason = dfr.embed_decision(grounded)
    return {"doc_id": doc_id, "n_pages": 1, "n_lines": 3, "n_agreed": 2, "agreed_lines_pct": 66.7,
            "readers": {"vlm": ["qwen3-vl-8b-heb-v21b-step1200"], "htr": ["MiDRASH_Gen_01@k7"]},
            "source_files": ["ai_reads.jsonl"], "model": dfr.DEFAULT_MODEL, "prompt_version": dfr.PROMPT_VERSION,
            "reasoning": "off", "generated_at": generated_at, "json": grounded, "grounding": report.to_dict(),
            "description_text": dfr.description_text(grounded), "embed_eligible": eligible, "embed_reason": reason,
            "description_source": source, "machine_derived": True}


def write_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> Path:
    """:param path: Destination.
    :param rows: Rows.
    :return: The path.
    """
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def no_model(*args: Any, **kwargs: Any) -> None:
    """Stands in for the model loader in runs that must not load it.

    :raises AssertionError: Always.
    """
    raise AssertionError("the embedder was loaded")


def standard_rows(tmp_path: Path) -> Path:
    """Rows for A (updatable), B (catalogue vector), C (not indexed), D (not eligible), E (other source).

    :param tmp_path: Temporary directory.
    :return: The generator file.
    """
    return write_rows(tmp_path / "gen.jsonl", [gen_row("Doc_A"), gen_row("Doc_B"), gen_row("Doc_C"),
                                               gen_row("Doc_D", summary="Damaged."),
                                               gen_row("Doc_E", source="catalogue")])


# --------------------------------------------------------------------------------------------- tests


def test_dry_run_prints_the_plan_and_neither_embeds_nor_writes(tmp_path, es, monkeypatch, capsys):
    monkeypatch.setattr(qte, "QwenTextEmbedding", no_model)
    before = copy.deepcopy(es.docs)
    result = ugd.update_generated_descriptions(standard_rows(tmp_path), index_name=INDEX, dry_run=True, es_client=es, text_version="semantic_v3")
    assert result["dry_run"] and result["updated"] == 0 and result["to_update"] == 1 and result["refused"] == 4
    assert result["refused_by_reason"] == {"has a catalogue vector": 1, "not in the index": 1,
                                           "embed_eligible false (short_summary)": 1,
                                           "description_source is 'catalogue'": 1}
    assert es.docs == before and es.bulk_calls == [] and es.indices.put == []
    out = capsys.readouterr().out
    assert "update Doc_A: Description: Letter (Doc_A)" in out and "nothing embedded or written" in out
    assert "mapping: would add description_source, generated_description, generated_description_meta" in out


def test_update_writes_the_vector_flags_text_and_provenance(tmp_path, es):
    embedder = FakeEmbedder()
    rows_path = standard_rows(tmp_path)
    row_a = gen_row("Doc_A")
    result = ugd.update_generated_descriptions(rows_path, index_name=INDEX, es_client=es, embedding_model=embedder, text_version="semantic_v3")
    assert result["updated"] == 1
    doc = es.docs[INDEX]["Doc_A"]
    embedded = text_for_version(row_a["description_text"], "semantic_v3")
    assert embedder.documents == [CANARY, embedded]          # the canary, then only the eligible text
    assert np.allclose(doc["embedding_vector"], fake_vector(embedded))
    assert doc["semantic_eligible"] is True and doc["text_representation_version"] == "semantic_v3"
    assert doc["description_source"] == "generated_from_consensus"
    assert doc["generated_description"] == row_a["description_text"]
    assert dfr.MACHINE_TAG in doc["generated_description"]
    meta = doc["generated_description_meta"]
    assert (meta["model"], meta["prompt_version"], meta["agreed_lines_pct"], meta["generated_at"]) == (
        dfr.DEFAULT_MODEL, dfr.PROMPT_VERSION, 66.7, "2026-10-08T12:00:00+00:00")
    assert meta["grounding"] == {"entities_kept": 3, "entities_dropped": 1, "summary_spans_absent": 0,
                                 "confidence_capped": True, "identified_work": "none",
                                 "normalisation": dfr.NORMALISATION}
    assert meta["machine_derived"] is True and meta["embedder_revision"] == qte.FINETUNED_MODEL_REVISION
    assert meta["confidence"] == dfr.CONFIDENCE_CAP and meta["readers"]["vlm"] == ["qwen3-vl-8b-heb-v21b-step1200"]
    assert es.docs[INDEX]["Doc_B"]["embedding_vector"] == fake_vector("catalogue").tolist()     # untouched
    assert "description_source" not in es.docs[INDEX]["Doc_D"]
    assert set(es.indices.put[0]) == set(ugd.GENERATED_FIELDS_MAPPING)
    assert es.indices.mappings[INDEX]["_meta"] == index_meta(FakeEmbedder())                   # _meta untouched
    assert vri.check_eligibility_vectors(es, INDEX, "semantic_v3")                             # flags match vectors
    assert ugd.embedded_text_from_source(doc, "semantic_v3") == embedded
    assert ugd.embedded_text_from_source(es.docs[INDEX]["Doc_B"], "semantic_v3") is None


@pytest.mark.parametrize("meta", [
    index_meta(FakeEmbedder(revision="0" * 40)),
    index_meta(FakeEmbedder(), text_version="v8"),
    {"embedding_model": "Qwen/Qwen3-Embedding-0.6B", "embedding_model_revision": "97b0c614"},
])
def test_writer_guard_refuses_a_mismatching_index(tmp_path, es, meta):
    es.indices.mappings[INDEX]["_meta"] = meta
    embedder = FakeEmbedder()
    with pytest.raises(IndexMetaMismatchError):
        ugd.update_generated_descriptions(standard_rows(tmp_path), index_name=INDEX, es_client=es,
                                          embedding_model=embedder, text_version="semantic_v3")
    assert embedder.documents == [] and es.bulk_calls == []


def test_missing_index_and_live_names_are_refused(tmp_path, es):
    rows = standard_rows(tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        ugd.update_generated_descriptions(rows, index_name="genizah_merged_v99", es_client=es, dry_run=True, text_version="semantic_v3")
    with pytest.raises(ValueError, match="live"):
        ugd.update_generated_descriptions(rows, index_name="genizah_merged_v8", es_client=es, dry_run=True, text_version="semantic_v3")


def test_an_embedder_that_misses_the_canary_is_refused(tmp_path, es):
    embedder = FakeEmbedder(salt="another vector space")
    with pytest.raises(IndexMetaMismatchError, match="canary"):
        ugd.update_generated_descriptions(standard_rows(tmp_path), index_name=INDEX, es_client=es,
                                          embedding_model=embedder, text_version="semantic_v3")
    assert es.bulk_calls == [] and embedder.documents == [CANARY]


def test_a_wrong_vector_size_is_refused_before_writing(tmp_path, es):
    es.indices.mappings[INDEX]["_meta"]["dims"] = 1024
    with pytest.raises(ValueError, match="dims"):
        ugd.update_generated_descriptions(standard_rows(tmp_path), index_name=INDEX, es_client=es,
                                          embedding_model=FakeEmbedder(), text_version="semantic_v3")
    assert es.bulk_calls == []


def test_rerun_skips_applied_generations_newer_rows_and_force_reapply(tmp_path, es):
    rows = standard_rows(tmp_path)
    ugd.update_generated_descriptions(rows, index_name=INDEX, es_client=es, embedding_model=FakeEmbedder(), text_version="semantic_v3")
    again = ugd.update_generated_descriptions(rows, index_name=INDEX, es_client=es, embedding_model=FakeEmbedder(), text_version="semantic_v3")
    assert (again["skipped"], again["updated"], len(es.bulk_calls)) == (1, 0, 1)
    newer = write_rows(tmp_path / "newer.jsonl", [gen_row("Doc_A"), gen_row("Doc_A", generated_at="2026-10-09T00:00:00+00:00")])
    result = ugd.update_generated_descriptions(newer, index_name=INDEX, es_client=es, embedding_model=FakeEmbedder(), text_version="semantic_v3")
    assert result["rows_read"] == 2 and result["updated"] == 1
    assert es.docs[INDEX]["Doc_A"]["generated_description_meta"]["generated_at"] == "2026-10-09T00:00:00+00:00"
    forced = ugd.update_generated_descriptions(newer, index_name=INDEX, es_client=es, embedding_model=FakeEmbedder(),
                                               force=True, text_version="semantic_v3")
    assert forced["updated"] == 1


def test_min_confidence_and_limit(tmp_path, es):
    rows = write_rows(tmp_path / "gen.jsonl", [gen_row("Doc_A", confidence=0.3), gen_row("Doc_D")])
    result = ugd.update_generated_descriptions(rows, index_name=INDEX, es_client=es, dry_run=True,
                                               min_confidence=0.4, text_version="semantic_v3")
    assert result["to_update"] == 1 and result["refused_by_reason"] == {"confidence 0.3 < 0.4": 1}
    limited = ugd.update_generated_descriptions(rows, index_name=INDEX, es_client=es, embedding_model=FakeEmbedder(),
                                                limit=1, text_version="semantic_v3")
    assert limited["updated"] == 1 and "description_source" not in es.docs[INDEX]["Doc_D"]


@pytest.mark.parametrize("text_version", ["semantic_v3", "semantic_v4"])
def test_update_follows_the_index_text_version_and_writes_the_text_hash(tmp_path, es, text_version):
    """Eligibility and text come from ``embed_text_for_version`` (``is_text_eligible`` refuses semantic_v4 fragments),
    and the embedded text's hash is written as the indexer writes it."""
    es.indices.mappings[INDEX]["_meta"]["text_representation_version"] = text_version
    embedder = FakeEmbedder()
    row_a = gen_row("Doc_A")
    label_only = dict(gen_row("Doc_D"), description_text="Description: קטע גניזה\nLanguage: Hebrew")  # a frozen label
    rows = write_rows(tmp_path / "gen.jsonl", [row_a, label_only])
    result = ugd.update_generated_descriptions(rows, index_name=INDEX, text_version=text_version, es_client=es,
                                               embedding_model=embedder)
    expected = embed_text_for_version(row_a["description_text"], text_version)
    assert expected.eligible and result["updated"] == 1
    assert result["refused_by_reason"] == {f"text not eligible under {text_version}": 1}
    assert embedder.documents == [CANARY, expected.text]
    doc = es.docs[INDEX]["Doc_A"]
    assert doc["text_representation_version"] == text_version
    assert doc["embedded_text_sha256"] == expected.text_sha256 == hashlib.sha256(expected.text.encode()).hexdigest()
    assert np.allclose(doc["embedding_vector"], fake_vector(expected.text))
    assert ugd.embedded_text_from_source(doc, text_version) == expected.text
    assert ("Document Type: Letter" in expected.text) == (text_version == "semantic_v3")   # v4 drops the type guess
    assert es.indices.put[0]["embedded_text_sha256"] == {"type": "keyword"}
    if text_version == "semantic_v4":
        with pytest.raises(ValueError, match="embed_text_for_version"):
            is_text_eligible(expected.text, text_version)       # the pre-v10 call cannot decide a v4 text


def test_cli_defaults_follow_the_indexer_constants():
    args = ugd.build_parser().parse_args(["--generated", "g.jsonl"])
    assert args.index == ugd.DEFAULT_INDEX_NAME == "genizah_merged_v10"
    assert args.text_version == "semantic_v4" and not args.dry_run and not args.allow_live
    assert ugd.build_parser().parse_args(["--generated", "g.jsonl", "--dry-run"]).dry_run
