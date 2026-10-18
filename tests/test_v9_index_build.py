# File name: test_v9_index_build.py
# Date: 10/7/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Index v9 build path: pinned fine-tuned loader, ``_meta`` contract, eligible-only vectors, writer guard, verifier.

No network and no model: SentenceTransformer, the embedder and Elasticsearch are all faked. No ``.env`` is read.
"""

import hashlib
import importlib
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest
import torch

from src.datasets.document_models.genizah_document import GenizahDocument
from src.datasets.indexing import elastic_index_genizah as eig
from src.datasets.indexing import index_merged_genizah as img
from src.datasets.indexing import verify_reembedded_indexes as vri
from src.embeddings import qwen_text_embedding as qte
from src.embeddings.text_representation import (
    BIBLIOGRAPHY,
    DocumentCanary,
    create_text_representation,
    text_for_version,
)

DIMS = qte.EMBEDDING_DIMS
INDEX = "genizah_merged_v9_test"


def fake_vector(text: str, salt: str = "") -> np.ndarray:
    """Deterministic unit vector for a text (different ``salt`` = a different model).

    :param text: Text.
    :param salt: Model stand-in.
    :return: Unit vector of :data:`DIMS` floats.
    """
    seed = int(hashlib.sha256((salt + text).encode("utf-8")).hexdigest()[:8], 16)
    vector = np.random.default_rng(seed).normal(size=DIMS).astype(np.float32)
    return vector / np.linalg.norm(vector)


class FakeEmbedder:
    """Stands in for QwenTextEmbedding: deterministic vectors, and a record of every text it embeds."""

    def __init__(self, model_name: str = qte.FINETUNED_MODEL_NAME, revision: str = qte.FINETUNED_MODEL_REVISION,
                 salt: str = "") -> None:
        """Create the fake.

        :param model_name: Reported model name.
        :param revision: Reported revision.
        :param salt: Vector-space stand-in.
        """
        self.model_name = model_name
        self.revision = revision
        self.max_seq_length = qte.MAX_SEQ_LENGTH
        self.salt = salt
        self.documents: List[str] = []
        self.queries: List[str] = []

    def get_text_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Document-mode vectors; an empty text is a test failure.

        :param texts: Texts.
        :return: (n, DIMS) array.
        """
        assert texts and all(t.strip() for t in texts), "an empty string reached the embedder"
        self.documents.extend(texts)
        return np.stack([fake_vector(t, self.salt) for t in texts])

    def get_query_embedding(self, query: str) -> np.ndarray:
        """Query-mode vector (prefix included).

        :param query: Query.
        :return: (DIMS,) array.
        """
        self.queries.append(query)
        return fake_vector(qte.QUERY_PROMPT + query, self.salt)


def matches(doc: Dict[str, Any], query: Optional[Dict[str, Any]]) -> bool:
    """Evaluate the few query shapes the verifier sends (term, exists, bool filter/must/must_not).

    :param doc: Document ``_source``.
    :param query: Query DSL (``None`` = match all).
    :return: Whether the document matches.
    """
    if not query:
        return True
    if "term" in query:
        ((field, value),) = query["term"].items()
        return doc.get(field) == value
    if "exists" in query:
        return doc.get(query["exists"]["field"]) is not None
    clause = query["bool"]
    return (all(matches(doc, q) for q in clause.get("filter", []) + clause.get("must", []))
            and not any(matches(doc, q) for q in clause.get("must_not", [])))


class FakeIndices:
    """``es.indices``: mappings by index name."""

    def __init__(self) -> None:
        """Start with no index."""
        self.mappings: Dict[str, Dict[str, Any]] = {}
        self.created: Dict[str, Dict[str, Any]] = {}

    def exists(self, index: str) -> bool:
        """:param index: Index name.
        :return: Whether it exists.
        """
        return index in self.mappings

    def create(self, index: str, body: Dict[str, Any]) -> None:
        """:param index: Index name.
        :param body: Settings + mappings.
        """
        self.created[index] = body
        self.mappings[index] = body["mappings"]

    def get_mapping(self, index: str) -> Dict[str, Any]:
        """:param index: Index name.
        :return: ``{index: {"mappings": ...}}``.
        """
        return {index: {"mappings": self.mappings[index]}}

    def put_mapping(self, index: str, meta: Optional[Dict[str, Any]] = None,
                    properties: Optional[Dict[str, Any]] = None) -> None:
        """:param index: Index name.
        :param meta: Replacement ``_meta`` (the whole block, as in Elasticsearch; the build's final ``counts``).
        :param properties: New fields.
        """
        if meta is not None:
            self.mappings[index]["_meta"] = meta
        if properties:
            self.mappings[index].setdefault("properties", {}).update(properties)


class FakeES:
    """In-memory Elasticsearch: indices, a bulk sink, and count / mget over what was written."""

    def __init__(self) -> None:
        """Start empty."""
        self.indices = FakeIndices()
        self.actions: List[Dict[str, Any]] = []

    def docs(self, index: str) -> List[Dict[str, Any]]:
        """:param index: Index name.
        :return: ``_source`` of every document written to it.
        """
        return [a["_source"] for a in self.actions if a["_index"] == index]

    def count(self, index: str, query: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
        """:param index: Index name.
        :param query: Query DSL.
        :return: ``{"count": n}``.
        """
        return {"count": sum(matches(doc, query) for doc in self.docs(index))}

    def mget(self, index: str, ids: Sequence[str], source: Any = None,
             source_includes: Optional[List[str]] = None) -> Dict[str, Any]:
        """:param index: Index name.
        :param ids: Document ids.
        :param source: Ignored.
        :param source_includes: Ignored (whole ``_source`` returned).
        :return: ``{"docs": [...]}``.
        """
        by_id = {a["_id"]: a["_source"] for a in self.actions if a["_index"] == index}
        return {"docs": [{"_id": i, "found": i in by_id, "_source": by_id.get(i, {})} for i in ids]}

    def search(self, index: str, size: int = 0, aggs: Optional[Dict[str, Any]] = None,
               query: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """The search a build ends with: a paged composite ``terms`` aggregation (``_meta.counts``).

        :param index: Index name.
        :param size: Ignored (no hits).
        :param aggs: ``{name: {"composite": {"size", "sources": [{key: {"terms": {"field": ...}}}], "after"?}}}``.
        :param query: Query DSL (``None`` = match all).
        :return: ``{"aggregations": {name: {"buckets": [...], "after_key"?: {...}}}}``.
        """
        ((name, agg),) = aggs.items()
        composite = agg["composite"]
        ((key, source),) = composite["sources"][0].items()
        field = source["terms"]["field"]
        values = sorted({d[field] for d in self.docs(index) if matches(d, query) and d.get(field) is not None})
        after = (composite.get("after") or {}).get(key)
        page = [v for v in values if after is None or v > after][:composite["size"]]
        result: Dict[str, Any] = {"buckets": [{"key": {key: v}, "doc_count": 1} for v in page]}
        if page:
            result["after_key"] = {key: page[-1]}
        return {"aggregations": {name: result}}


@pytest.fixture
def fake_es(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeES:
    """Route the processor's client and bulk helper to one FakeES (cwd moved so caches land in tmp).

    :param monkeypatch: pytest monkeypatch.
    :param tmp_path: Temporary directory.
    :return: The fake cluster.
    """
    es = FakeES()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(eig, "Elasticsearch", lambda **kwargs: es)

    def fake_bulk(client: FakeES, actions: List[Dict[str, Any]], refresh: str) -> tuple:
        """:param client: Fake cluster.
        :param actions: Bulk actions.
        :param refresh: Ignored.
        :return: (successes, failures).
        """
        client.actions.extend(actions)
        return len(actions), []

    monkeypatch.setattr(eig, "bulk", fake_bulk)
    return es


def record(cid: str, shelfmark: str, description: str, **sources: Any) -> Dict[str, Any]:
    """A merged_shelfmarks.jsonl record (PGP / KTIV sources only).

    :param cid: Canonical id.
    :param shelfmark: Display shelf mark.
    :param description: Catalogue description.
    :param sources: Source blocks.
    :return: Record dict.
    """
    return {"canonical_id": cid, "shelfmark_display": shelfmark, "institution": "Cambridge University Library",
            "sources_present": sorted(sources), "description": description, "date": None, "images": {},
            "sources": sources}


BOILER_ID, RICH_ID, BARE_ID, RICH2_ID = ("Cambridge_CUL_T_S_NS_1_1", "Cambridge_CUL_T_S_13J8_14",
                                         "Cambridge_CUL_T_S_NS_1_2", "Cambridge_CUL_T_S_10J5_3")
RECORDS = [
    record(BOILER_ID, "T-S NS 1.1", "Newly treated and encapsulated, must be examined"),
    record(RICH_ID, "T-S 13J8.14", "Letter from Nahray b. Nissim about the flax trade. Joins ENA 2558.5.",
           pgp={"documents": [{"pgpid": "12345", "languages_primary": "Judaeo-Arabic", "location": "Fustat"}]}),
    record(BARE_ID, "T-S NS 1.2", ""),
    record(RICH2_ID, "T-S 10J5.3", "Marriage contract from Fustat with a list of the bride's trousseau.",
           ktiv_transcription={"pages": [{"text": "בשם רחמנא", "image_name": "Frag. 001r"}], "page_count": 1}),
]


def write_merged(path: Path, records: List[Dict[str, Any]]) -> str:
    """Write records as a merged JSONL.

    :param path: Destination.
    :param records: Records.
    :return: The path as a string.
    """
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return str(path)


def semantic(rec: Dict[str, Any]) -> str:
    """:param rec: Merged record.
    :return: Its semantic_v3 text.
    """
    return create_text_representation(GenizahDocument.from_merged_format(rec), "semantic_v3")


# --------------------------------------------------------------------------------------------- the model loader
class FakeSentenceTransformer:
    """Records how QwenTextEmbedding loads and calls the model."""

    last: "FakeSentenceTransformer"

    def __init__(self, model_name_or_path: str, **kwargs: Any) -> None:
        """:param model_name_or_path: Hub id.
        :param kwargs: Loader kwargs.
        """
        self.name = model_name_or_path
        self.kwargs = kwargs
        self.max_seq_length = 384           # a saved 384-token training window must not survive
        self.default_prompt_name = "query"  # a saved default prompt must not survive loading
        self.encode_calls: List[tuple] = []
        FakeSentenceTransformer.last = self

    def tokenizer(self, texts: List[str], truncation: bool, max_length: int) -> Dict[str, List[List[int]]]:
        """:param texts: Texts.
        :param truncation: Ignored.
        :param max_length: Cap.
        :return: Fake token ids.
        """
        return {"input_ids": [[0] * min(len(t), max_length) for t in texts]}

    def encode(self, texts: List[str], **kwargs: Any) -> np.ndarray:
        """:param texts: Texts.
        :param kwargs: Encode kwargs (recorded).
        :return: Ones.
        """
        self.encode_calls.append((list(texts), kwargs))
        return np.ones((len(texts), DIMS), dtype=np.float32)


@pytest.fixture
def fake_st(monkeypatch: pytest.MonkeyPatch) -> type:
    """Replace the sentence_transformers module (nothing is downloaded or loaded).

    :param monkeypatch: pytest monkeypatch.
    :return: The fake class (``.last`` is the latest instance).
    """
    module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer, __version__="fake")
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return FakeSentenceTransformer


def test_loader_defaults_to_the_pinned_finetune(fake_st, monkeypatch):
    monkeypatch.setenv("HF1_TOKEN", "hf1-test-value")
    monkeypatch.setenv("HF_TOKEN", "hf-test-value")
    model = qte.QwenTextEmbedding(device="cpu")
    st = fake_st.last
    assert (st.name, st.kwargs["revision"]) == ("isaacmg/genizah-embed-qwen3-0.6b",
                                                "af0252255d3e9b8b8432e09de7c5f0689d7ada34")
    assert (model.model_name, model.revision) == (qte.FINETUNED_MODEL_NAME, qte.FINETUNED_MODEL_REVISION)
    assert st.kwargs["token"] == "hf1-test-value"
    assert st.kwargs["model_kwargs"] == {"dtype": torch.float32}
    assert st.max_seq_length == model.max_seq_length == 8192
    assert st.default_prompt_name is None


def test_loader_token_fallback_and_base_model(fake_st, monkeypatch):
    monkeypatch.delenv("HF1_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf-test-value")
    qte.QwenTextEmbedding(device="cpu")
    assert fake_st.last.kwargs["token"] == "hf-test-value"
    monkeypatch.delenv("HF_TOKEN")
    qte.QwenTextEmbedding(model_name=qte.MODEL_ID, revision=qte.MODEL_REVISION, device="cpu")
    assert fake_st.last.kwargs["token"] is None
    assert (fake_st.last.name, fake_st.last.kwargs["revision"]) == (qte.MODEL_ID, qte.MODEL_REVISION)


def test_queries_get_the_explicit_prefix_and_documents_none(fake_st):
    assert qte.QUERY_PROMPT == "Instruct: Given a search query, retrieve relevant passages\nQuery: "
    model = qte.QwenTextEmbedding(device="cpu")
    model.get_query_embedding("laws of lulav")
    texts, kwargs = fake_st.last.encode_calls[-1]
    assert texts == ["laws of lulav"] and kwargs["prompt"] == qte.QUERY_PROMPT and "prompt_name" not in kwargs
    model.get_text_embeddings_batch(["Description: a letter about flax"])
    texts, kwargs = fake_st.last.encode_calls[-1]
    assert texts == ["Description: a letter about flax"]
    assert "prompt" not in kwargs and "prompt_name" not in kwargs


# --------------------------------------------------------------------------------------------- _meta
def test_build_index_meta_records_the_contract_and_three_canaries():
    embedder = FakeEmbedder()
    canary = DocumentCanary(RICH_ID, "Description: Letter about the flax trade in Alexandria.")
    meta = qte.build_index_meta(embedder, "semantic_v3", canary)
    assert meta["model_name"] == "isaacmg/genizah-embed-qwen3-0.6b"
    assert meta["model_revision"] == "af0252255d3e9b8b8432e09de7c5f0689d7ada34"
    assert meta["query_prompt"] == qte.QUERY_PROMPT
    assert meta["text_representation_version"] == "semantic_v3"
    assert meta["max_seq_length"] == 8192
    # pre-v9 keys the site's startup gate reads
    assert (meta["embedding_model"], meta["embedding_model_revision"]) == (meta["model_name"], meta["model_revision"])
    assert meta["canary"]["string"] == qte.CANARY_STRING
    assert np.allclose(meta["canary"]["vector"], fake_vector(qte.CANARY_STRING), atol=1e-7)
    assert meta["query_canary"]["prompt"] == qte.QUERY_PROMPT
    assert np.allclose(meta["query_canary"]["vector"], fake_vector(qte.QUERY_PROMPT + qte.CANARY_STRING), atol=1e-7)
    doc = meta["document_canary"]
    assert (doc["doc_id"], doc["text"]) == canary
    assert np.allclose(doc["vector"], fake_vector(canary.text), atol=1e-7) and len(doc["vector_sha256_6dp"]) == 64
    assert json.loads(json.dumps(meta)) == meta


def test_build_index_meta_rejects_bad_input():
    embedder = FakeEmbedder()
    with pytest.raises(ValueError, match="unknown text representation version"):
        qte.build_index_meta(embedder, "semantic_v99", None)
    with pytest.raises(ValueError, match="empty"):
        qte.build_index_meta(embedder, "semantic_v3", DocumentCanary("x", "  "))
    assert qte.build_index_meta(embedder, "semantic_v3", None)["document_canary"] is None


# --------------------------------------------------------------------------------------------- the indexer
def test_index_merged_embeds_only_eligible_records_and_writes_flags(tmp_path, fake_es):
    embedder = FakeEmbedder()
    path = write_merged(tmp_path / "merged.jsonl", RECORDS)
    result = img.index_merged(text_version="semantic_v3", index_name=INDEX, merged_path=path, embedding_model=embedder, elasticsearch_config={})
    assert result == {"indexed": 4, "without_vector": 2}

    by_id = {a["_id"]: a["_source"] for a in fake_es.actions}
    for cid in (BOILER_ID, BARE_ID):
        assert by_id[cid]["semantic_eligible"] is False and "embedding_vector" not in by_id[cid]
        assert by_id[cid]["shelf_mark"]  # still there for shelf-mark / BM25 lookup
    for rec in (RECORDS[1], RECORDS[3]):
        src = by_id[rec["canonical_id"]]
        assert src["semantic_eligible"] is True
        assert np.allclose(src["embedding_vector"], fake_vector(semantic(rec)))
    assert {s["text_representation_version"] for s in by_id.values()} == {"semantic_v3"}
    # the model only ever saw the canaries and the eligible semantic texts
    assert semantic(RECORDS[0]) not in embedder.documents and "" not in embedder.documents
    assert not any(t.startswith("Document ID:") for t in embedder.documents)

    body = fake_es.indices.created[INDEX]
    properties = body["mappings"]["properties"]
    assert properties["semantic_eligible"] == {"type": "boolean"}
    assert properties["text_representation_version"] == {"type": "keyword"}
    meta = body["mappings"]["_meta"]
    assert meta["text_representation_version"] == "semantic_v3"
    assert meta["document_canary"]["doc_id"] == RICH_ID  # the first eligible record
    assert meta["document_canary"]["text"] == semantic(RECORDS[1])


def test_v8_text_version_embeds_every_record(tmp_path, fake_es):
    embedder = FakeEmbedder()
    path = write_merged(tmp_path / "merged.jsonl", RECORDS)
    result = img.index_merged(index_name=INDEX, merged_path=path, text_version="v8", embedding_model=embedder,
                              elasticsearch_config={})
    assert result == {"indexed": 4, "without_vector": 0}
    assert all(s["semantic_eligible"] and s["text_representation_version"] == "v8" for s in fake_es.docs(INDEX))
    v8 = GenizahDocument.from_merged_format(RECORDS[0]).create_text_representation()
    assert v8 in embedder.documents


@pytest.mark.parametrize("name", sorted(img.LIVE_INDEX_NAMES))
def test_live_index_names_are_refused(name, tmp_path, fake_es):
    embedder = FakeEmbedder()
    path = write_merged(tmp_path / "merged.jsonl", RECORDS)
    with pytest.raises(ValueError, match="i-know-this-is-live"):
        img.index_merged(text_version="semantic_v3", index_name=name, merged_path=path, embedding_model=embedder, elasticsearch_config={})
    assert embedder.documents == [] and fake_es.actions == []


def test_live_flag_still_meets_the_writer_guard(tmp_path, fake_es):
    fake_es.indices.mappings["genizah_merged_v8"] = {"_meta": {
        "embedding_model": qte.MODEL_ID, "embedding_model_revision": qte.MODEL_REVISION}}
    path = write_merged(tmp_path / "merged.jsonl", RECORDS)
    with pytest.raises(eig.IndexMetaMismatchError, match="pre-v9"):
        img.index_merged(text_version="semantic_v3", index_name="genizah_merged_v8", allow_live=True, merged_path=path,
                         embedding_model=FakeEmbedder(), elasticsearch_config={})
    assert fake_es.actions == []


@pytest.mark.parametrize("existing, refused_key", [
    ({"model_name": qte.MODEL_ID, "model_revision": qte.MODEL_REVISION, "text_representation_version": "semantic_v3"},
     "model_name"),
    ({"model_name": qte.FINETUNED_MODEL_NAME, "model_revision": "0" * 40, "text_representation_version": "semantic_v3"},
     "model_revision"),
    ({"model_name": qte.FINETUNED_MODEL_NAME, "model_revision": qte.FINETUNED_MODEL_REVISION,
      "text_representation_version": "v8"}, "text_representation_version"),
])
def test_writer_guard_refuses_a_mismatching_meta(existing, refused_key, tmp_path, fake_es):
    fake_es.indices.mappings[INDEX] = {"_meta": existing}
    path = write_merged(tmp_path / "merged.jsonl", RECORDS)
    with pytest.raises(eig.IndexMetaMismatchError, match=refused_key):
        img.index_merged(text_version="semantic_v3", index_name=INDEX, merged_path=path, embedding_model=FakeEmbedder(), elasticsearch_config={})
    assert fake_es.actions == [] and INDEX not in fake_es.indices.created


def test_writer_guard_lets_a_matching_run_resume(tmp_path, fake_es):
    fake_es.indices.mappings[INDEX] = {"_meta": {
        "model_name": qte.FINETUNED_MODEL_NAME, "model_revision": qte.FINETUNED_MODEL_REVISION,
        "text_representation_version": "semantic_v3"}}
    path = write_merged(tmp_path / "merged.jsonl", RECORDS)
    result = img.index_merged(text_version="semantic_v3", index_name=INDEX, merged_path=path, skip=2, embedding_model=FakeEmbedder(),
                              elasticsearch_config={})
    assert result == {"indexed": 2, "without_vector": 1}
    assert INDEX not in fake_es.indices.created  # existing index and its _meta left alone


def test_meta_mismatch_helper():
    expected = {"model_name": "m", "model_revision": "r", "text_representation_version": "semantic_v3", "x": 1}
    assert eig.index_meta_mismatches(dict(expected, x=2), expected) == []
    problems = eig.index_meta_mismatches({"model_name": "m"}, expected)
    assert [p.split(":")[0] for p in problems] == ["model_revision", "text_representation_version"]


def test_versioned_processor_needs_matching_meta(fake_es):
    with pytest.raises(ValueError, match="index_meta"):
        eig.ElasticsearchGenizahProcessor(FakeEmbedder(), elasticsearch_config={}, index_name=INDEX,
                                          text_version="semantic_v3")
    with pytest.raises(ValueError, match="index_meta"):
        eig.ElasticsearchGenizahProcessor(FakeEmbedder(), elasticsearch_config={}, index_name=INDEX,
                                          index_meta={"text_representation_version": "v8"},
                                          text_version="semantic_v3")


def test_legacy_processor_path_is_unchanged(fake_es):
    embedder = FakeEmbedder()
    docs = [GenizahDocument.from_merged_format(r) for r in RECORDS]
    expected = [d.create_text_representation() for d in docs]
    processor = eig.ElasticsearchGenizahProcessor(embedder, elasticsearch_config={}, index_name="legacy_test",
                                                  embedding_dims=DIMS)
    processor.process_documents(docs)
    assert embedder.documents == expected
    assert all("embedding_vector" in s and "semantic_eligible" not in s for s in fake_es.docs("legacy_test"))


def test_cli_index_default_is_the_code_constant_and_never_a_live_name():
    args = img.build_parser().parse_args([])
    assert args.index == img.DEFAULT_INDEX_NAME == "genizah_merged_v11"
    assert args.index not in img.LIVE_INDEX_NAMES and args.text_version == "semantic_v4"


# --------------------------------------------------------------------------------------------- bibliography
@pytest.fixture
def bibliography_indexer(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Import the bibliography indexer without letting it load ``.env``.

    :param monkeypatch: pytest monkeypatch.
    :return: The ``index_bibliography`` module.
    """
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    module = importlib.import_module("src.datasets.indexing.bibliography.index_bibliography")
    monkeypatch.setattr(module, "es_config_from_env", lambda: {})
    return module


def test_bibliography_indexer_embeds_page_text_without_the_pdf_line(tmp_path, fake_es, bibliography_indexer):
    book = tmp_path / "book_structured"
    book.mkdir()
    (book / "page_001_structured.json").write_text(json.dumps({
        "metadata": {"pdf_name": "book", "page_number": 1},
        "shelf_marks_mentioned": ["T-S 13J8.14"],
        "summary": "Discusses T-S 13J8.14, a letter on the flax trade.",
        "full_main_text": "The letter T-S 13J8.14 names the partner.",
    }), encoding="utf-8")
    (book / "page_002_structured.json").write_text(json.dumps({"metadata": {"pdf_name": "book", "page_number": 2}}),
                                                   encoding="utf-8")
    metadata = tmp_path / "book_metadata.json"
    metadata.write_text("{}", encoding="utf-8")
    embedder = FakeEmbedder()
    bibliography_indexer.index_bibliography_to_elasticsearch(
        root_dir=str(book), metadata_file=str(metadata), index_name="bibliography_v9_test",
        embedding_mode="text_only", embeddings_model=embedder, text_version="semantic_v3")

    pages = {s["doc_id"]: s for s in fake_es.docs("bibliography_v9_test")}
    first, second = pages["book_p001"], pages["book_p002"]
    assert first["full_text_content"].startswith("Source PDF: book, Page: 1\n")  # stored (lexical) text unchanged
    embedded = text_for_version(first["full_text_content"], "semantic_v3", BIBLIOGRAPHY)
    assert embedded.startswith("Shelf marks mentioned: T-S 13J8.14\n")
    assert first["semantic_eligible"] is True and np.allclose(first["embedding_vector"], fake_vector(embedded))
    assert second["semantic_eligible"] is False and "embedding_vector" not in second  # only the PDF line
    meta = fake_es.indices.created["bibliography_v9_test"]["mappings"]["_meta"]
    assert meta["text_representation_version"] == "semantic_v3" and meta["document_canary"]["text"] == embedded
    canary_id = meta["document_canary"]["doc_id"]
    assert vri.biblio_text(fake_es, "bibliography_v9_test", canary_id, "semantic_v3") == embedded
    assert vri.check_eligibility_vectors(fake_es, "bibliography_v9_test", "semantic_v3")


def test_index_all_bibliography_stops_at_the_writer_guard(tmp_path, monkeypatch, fake_st, bibliography_indexer):
    module = importlib.import_module("src.datasets.indexing.bibliography.index_all_bibliography")
    task = {"metadata_file": tmp_path / "m.json", "structured_dir": tmp_path, "image_dir": None, "book": "b"}
    monkeypatch.setattr(module, "discover_indexing_tasks", lambda root_dir, suffix: [task, dict(task, book="c")])
    calls: List[Dict[str, Any]] = []

    def refuse(**kwargs: Any) -> None:
        """:param kwargs: Call arguments (recorded).
        :raises eig.IndexMetaMismatchError: Always.
        """
        calls.append(kwargs)
        raise eig.IndexMetaMismatchError("refused")

    monkeypatch.setattr(module, "index_bibliography_to_elasticsearch", refuse)
    with pytest.raises(eig.IndexMetaMismatchError):
        module.index_all_bibliography(root_dir=str(tmp_path), index_name="bibliography_v9_test",
                                      embedding_mode="text_only")
    assert len(calls) == 1 and calls[0]["text_version"] == "semantic_v4"


# --------------------------------------------------------------------------------------------- the verifier
def test_document_canary_check():
    embedder = FakeEmbedder()
    canary = DocumentCanary(RICH_ID, "Description: Letter about the flax trade in Alexandria.")
    meta = qte.build_index_meta(embedder, "semantic_v3", canary)
    assert vri.check_document_canary(embedder, INDEX, meta, lambda doc_id: canary.text)
    assert not vri.check_document_canary(embedder, INDEX, meta, lambda doc_id: canary.text + " (changed)")
    assert not vri.check_document_canary(embedder, INDEX, meta, lambda doc_id: None)
    assert not vri.check_document_canary(FakeEmbedder(salt="base"), INDEX, meta, lambda doc_id: canary.text)
    assert not vri.check_document_canary(embedder, INDEX, dict(meta, document_canary=None), lambda d: canary.text)


def test_meta_contract_check():
    embedder = FakeEmbedder()
    meta = qte.build_index_meta(embedder, "semantic_v3", DocumentCanary(RICH_ID, "Description: a letter on flax."))
    assert vri.check_meta_contract(INDEX, meta, embedder)
    assert not vri.check_meta_contract(INDEX, meta, FakeEmbedder(qte.MODEL_ID, qte.MODEL_REVISION))
    assert not vri.check_meta_contract(INDEX, dict(meta, query_prompt="Instruct: Given a web search query"), embedder)
    legacy = {k: v for k, v in meta.items() if k not in vri.REQUIRED_META_KEYS or k == "canary"}
    assert not vri.check_meta_contract(INDEX, legacy, embedder)
    assert vri.index_text_version(legacy) == "v8"


def test_verifier_passes_the_built_index_and_fails_another_model(tmp_path, fake_es):
    embedder = FakeEmbedder()
    path = write_merged(tmp_path / "merged.jsonl", RECORDS)
    img.index_merged(text_version="semantic_v3", index_name=INDEX, merged_path=path, embedding_model=embedder, elasticsearch_config={})

    def recompute(doc_id: str, version: str) -> Optional[str]:
        """:param doc_id: Canonical id.
        :param version: Text version.
        :return: The record's text.
        """
        return vri.merged_texts([doc_id], version, path).get(doc_id)

    assert vri.verify_index(fake_es, embedder, INDEX, recompute)
    assert not vri.verify_index(fake_es, FakeEmbedder(qte.MODEL_ID, qte.MODEL_REVISION, salt="base"), INDEX, recompute)


def test_eligibility_check_catches_vector_flag_disagreement(tmp_path, fake_es):
    path = write_merged(tmp_path / "merged.jsonl", RECORDS)
    img.index_merged(text_version="semantic_v3", index_name=INDEX, merged_path=path, embedding_model=FakeEmbedder(), elasticsearch_config={})
    assert vri.check_eligibility_vectors(fake_es, INDEX, "semantic_v3")
    ineligible = next(s for s in fake_es.docs(INDEX) if not s["semantic_eligible"])
    ineligible["embedding_vector"] = fake_vector("x").tolist()
    assert not vri.check_eligibility_vectors(fake_es, INDEX, "semantic_v3")
    del ineligible["embedding_vector"]
    eligible = next(s for s in fake_es.docs(INDEX) if s["semantic_eligible"])
    del eligible["embedding_vector"]
    assert not vri.check_eligibility_vectors(fake_es, INDEX, "semantic_v3")


def test_merged_texts_follow_the_text_version(tmp_path):
    path = write_merged(tmp_path / "merged.jsonl", RECORDS)
    texts = vri.merged_texts([RICH_ID, "missing"], "semantic_v3", path)
    assert texts == {RICH_ID: semantic(RECORDS[1])}
    v8 = vri.merged_texts([RICH_ID], "v8", path)[RICH_ID]
    assert v8 == GenizahDocument.from_merged_format(RECORDS[1]).create_text_representation()


def test_verifier_cli_defaults_follow_the_builders():
    args = vri.build_parser().parse_args([])
    assert args.genizah_index == img.DEFAULT_INDEX_NAME
    assert args.biblio_index == "bibliography_text_only_0.8" and args.biblio_index != "bibliography_text_only_0.7"


def test_loader_passes_the_hub_token_to_every_download_path(monkeypatch):
    """A private repo answers the processor probe with 401 unless the token reaches it (seen 2026-10-08)."""
    import sentence_transformers
    from src.embeddings import qwen_text_embedding as qte

    seen = {}

    class FakeST:
        def __init__(self, name, **kwargs):
            seen.update(kwargs)
            self.max_seq_length = 512
            self.default_prompt_name = "query"

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeST)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF1_TOKEN", "hf_dummy_token_for_test")
    model = qte.QwenTextEmbedding(device="cpu")
    assert seen["token"] == "hf_dummy_token_for_test"
    assert seen["processor_kwargs"] == {"token": "hf_dummy_token_for_test"}
    assert os.environ["HF_TOKEN"] == "hf_dummy_token_for_test"
    assert model.max_seq_length == qte.MAX_SEQ_LENGTH == 8192 and model.model.default_prompt_name is None
