# File name: test_v10_index_build.py
# Date: 10/10/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Index v10 build path: per-document text hash, one embedding per distinct text, the scholarly-frames render flag
(default off), the extended ``_meta`` contract with its final ``counts``, and the verifier's text-drift check.

No network, no model and no git: Elasticsearch, the embedder and ``hda_commit`` are faked. No ``.env`` is read.
"""

import copy
import hashlib
import inspect
import json
import subprocess
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest

from src.datasets.document_models.genizah_document import GenizahDocument
from src.datasets.indexing import elastic_index_genizah as eig
from src.datasets.indexing import index_merged_genizah as img
from src.datasets.indexing import update_generated_descriptions as ugd
from src.datasets.indexing import verify_reembedded_indexes as vri
from src.embeddings import qwen_text_embedding as qte
from src.embeddings.text_representation import DocumentCanary, embed_text_for_version, embedding_text

DIMS = 32
INDEX = "genizah_merged_v10_test"
V4 = "semantic_v4"
COMMIT = "0123456789abcdef0123456789abcdef01234567-dirty"
LABEL = "קטע גניזה"     # a frozen boilerplate label: no vector on its own


def sha(text: str) -> str:
    """:param text: Text.
    :return: sha256 hex of its UTF-8 bytes.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fake_vector(text: str, salt: str = "") -> np.ndarray:
    """Deterministic unit vector for a text (a different ``salt`` stands for a different model).

    :param text: Text.
    :param salt: Model stand-in.
    :return: Unit vector of :data:`DIMS` floats.
    """
    seed = int(sha(salt + text)[:8], 16)
    vector = np.random.default_rng(seed).normal(size=DIMS).astype(np.float32)
    return vector / np.linalg.norm(vector)


class FakeEmbedder:
    """Stands in for QwenTextEmbedding: deterministic vectors, and a record of every call."""

    def __init__(self, salt: str = "") -> None:
        """:param salt: Vector-space stand-in."""
        self.model_name = qte.FINETUNED_MODEL_NAME
        self.revision = qte.FINETUNED_MODEL_REVISION
        self.max_seq_length = qte.MAX_SEQ_LENGTH
        self.salt = salt
        self.calls: List[List[str]] = []

    @property
    def documents(self) -> List[str]:
        """:return: Every document-mode text embedded, in order."""
        return [text for call in self.calls for text in call]

    def get_text_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """:param texts: Texts (an empty one is a test failure).
        :return: (n, DIMS) array.
        """
        assert texts and all(t.strip() for t in texts), "an empty string reached the embedder"
        self.calls.append(list(texts))
        return np.stack([fake_vector(t, self.salt) for t in texts])

    def get_query_embedding(self, query: str) -> np.ndarray:
        """:param query: Query.
        :return: (DIMS,) array.
        """
        return fake_vector(qte.QUERY_PROMPT + query, self.salt)


def matches(doc: Dict[str, Any], query: Optional[Dict[str, Any]]) -> bool:
    """Evaluate the query shapes the indexer and the verifier send (match_all, term, exists, bool).

    :param doc: ``_source``.
    :param query: Query DSL (``None`` = match all).
    :return: Whether the document matches.
    """
    if not query or "match_all" in query:
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
    """``es.indices``: mappings by index; ``put_mapping`` replaces ``_meta`` whole, as Elasticsearch does."""

    def __init__(self, log: List[tuple]) -> None:
        """:param log: The cluster's operation log."""
        self.mappings: Dict[str, Dict[str, Any]] = {}
        self.created: Dict[str, Dict[str, Any]] = {}
        self.log = log

    def exists(self, index: str) -> bool:
        """:param index: Index.
        :return: Whether it exists.
        """
        return index in self.mappings

    def create(self, index: str, body: Dict[str, Any]) -> None:
        """:param index: Index.
        :param body: Settings + mappings (deep-copied: what the index was created with stays inspectable).
        """
        self.created[index] = copy.deepcopy(body)
        self.mappings[index] = copy.deepcopy(body["mappings"])

    def get_mapping(self, index: str) -> Dict[str, Any]:
        """:param index: Index.
        :return: ``{index: {"mappings": ...}}``.
        """
        return {index: {"mappings": copy.deepcopy(self.mappings[index])}}

    def put_mapping(self, index: str, meta: Optional[Dict[str, Any]] = None,
                    properties: Optional[Dict[str, Any]] = None) -> None:
        """:param index: Index.
        :param meta: Replacement ``_meta``.
        :param properties: New fields.
        """
        self.log.append(("put_mapping", sorted(meta or {}), sorted(properties or {})))
        if meta is not None:
            self.mappings[index]["_meta"] = copy.deepcopy(meta)
        if properties:
            self.mappings[index].setdefault("properties", {}).update(properties)


class FakeES:
    """In-memory Elasticsearch: a bulk sink (last write per id wins), count, mget, the verifier's random sample and a
    paged composite aggregation."""

    def __init__(self) -> None:
        """Start empty."""
        self.log: List[tuple] = []
        self.indices = FakeIndices(self.log)
        self.actions: List[Dict[str, Any]] = []

    def docs(self, index: str) -> Dict[str, Dict[str, Any]]:
        """:param index: Index.
        :return: ``_id`` -> ``_source`` (the live objects: tests may tamper with them).
        """
        return {a["_id"]: a["_source"] for a in self.actions if a["_index"] == index}

    def count(self, index: str, query: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
        """:param index: Index.
        :param query: Query DSL.
        :return: ``{"count": n}``.
        """
        return {"count": sum(matches(d, query) for d in self.docs(index).values())}

    def mget(self, index: str, ids: Sequence[str], source: Any = None,
             source_includes: Optional[List[str]] = None) -> Dict[str, Any]:
        """:param index: Index.
        :param ids: Ids.
        :param source: Ignored.
        :param source_includes: Ignored (whole ``_source``).
        :return: ``{"docs": [...]}``.
        """
        docs = self.docs(index)
        return {"docs": [{"_id": i, "found": i in docs, "_source": docs.get(i, {})} for i in ids]}

    def search(self, index: str, body: Optional[Dict[str, Any]] = None, size: int = 0,
               query: Optional[Dict[str, Any]] = None, aggs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """The verifier's sample (``body``: docs with a vector, in write order, ``_source`` filtered to the requested
        fields) or a paged composite ``terms`` aggregation (``aggs``).

        :param index: Index.
        :param body: ``_sample_docs`` body.
        :param size: Ignored for aggregations.
        :param query: Aggregation query.
        :param aggs: ``{name: {"composite": {...}}}``.
        :return: Search response.
        """
        docs = self.docs(index)
        if body is not None:
            fields = body["_source"]
            hits = [{"_id": i, "_source": {k: s[k] for k in fields if k in s}}
                    for i, s in docs.items() if s.get("embedding_vector") is not None]
            return {"hits": {"hits": hits[:body["size"]]}}
        ((name, agg),) = aggs.items()
        composite = agg["composite"]
        ((key, source),) = composite["sources"][0].items()
        field = source["terms"]["field"]
        values = sorted({d[field] for d in docs.values() if matches(d, query) and d.get(field) is not None})
        after = (composite.get("after") or {}).get(key)
        page = [v for v in values if after is None or v > after][:composite["size"]]
        result: Dict[str, Any] = {"buckets": [{"key": {key: v}, "doc_count": 1} for v in page]}
        if page:
            result["after_key"] = {key: page[-1]}
        return {"aggregations": {name: result}}


@pytest.fixture(autouse=True)
def fixed_commit(monkeypatch: pytest.MonkeyPatch) -> str:
    """Keep git out of the tests: ``hda_commit`` returns a fixed value.

    :param monkeypatch: pytest monkeypatch.
    :return: The value.
    """
    monkeypatch.setattr(qte, "hda_commit", lambda: COMMIT)
    return COMMIT


@pytest.fixture
def fake_es(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeES:
    """Route the processor's client and bulk helper to one FakeES (cwd moved so the processor's cache dir lands in tmp).

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
        client.log.append(("bulk", len(actions)))
        return len(actions), []

    monkeypatch.setattr(eig, "bulk", fake_bulk)
    return es


def ktiv_record(cid: str, n: int, description: str, frames: Sequence[str] = ()) -> Dict[str, Any]:
    """A KTIV merged record with scholarly frames (the shape of ``tests/test_update_catalogue_enrichment``).

    :param cid: Canonical id.
    :param n: Shelf-mark number.
    :param description: Merged description.
    :param frames: KTIV scholarly frames (``scholarly_frames``, on the ``Alternate titles:`` line when rendered).
    :return: Record.
    """
    entries = [{"subsections": {"writing_characteristics": {"frame": frame}}} for frame in frames]
    return {"canonical_id": cid, "shelfmark_display": f"Cambridge University Library, Cambridge, England Ms. T-S NS 219.{n}",
            "institution": "Cambridge University Library", "sources_present": ["ktiv"], "description": description,
            "date": None, "images": {"ktiv": {"image_urls": []}, "preferred_source": None},
            "sources": {"pgp": None, "fjp": [], "bodleian": None,
                        "ktiv": {"sys_num": "990051590520205171",
                                 "basic_catalog": {"title": LABEL, "varying_form_of_title": []},
                                 "full_catalog": {"subjects": []}, "scholarly_entries": entries}}}


def plain_record(cid: str, shelfmark: str, description: str) -> Dict[str, Any]:
    """A catalogue-only merged record.

    :param cid: Canonical id.
    :param shelfmark: Display shelf mark.
    :param description: Description.
    :return: Record.
    """
    return {"canonical_id": cid, "shelfmark_display": shelfmark, "institution": "Cambridge University Library",
            "sources_present": [], "description": description, "date": None, "images": {}, "sources": {}}


FRAMED_ID, BOILER_ID, BARE_ID, DUP_A_ID, DUP_B_ID, LABEL_FRAME_ID = (
    "Cambridge_CUL_T_S_NS_219_1", "Cambridge_CUL_T_S_NS_1_1", "Cambridge_CUL_T_S_NS_1_2",
    "Manchester_A_1", "Manchester_A_2", "Cambridge_CUL_T_S_NS_219_3")
GENESIS = "Book of Genesis. Bible: Texts in Hebrew with Babylonian vocalisation."
RECORDS = [
    ktiv_record(FRAMED_ID, 1, "Liturgical poem for the Day of Atonement in an oriental square hand.",
                frames=["[Piyyut]: Avodah for Yom Kippur"]),                   # eligible, text depends on the flag
    plain_record(BOILER_ID, "T-S NS 1.1", "Newly treated and encapsulated, must be examined"),   # boilerplate
    plain_record(BARE_ID, "T-S NS 1.2", ""),                                                   # empty text
    plain_record(DUP_A_ID, "A 1", GENESIS),                                    # two records, one text
    plain_record(DUP_B_ID, "A 2", GENESIS),
    ktiv_record(LABEL_FRAME_ID, 3, LABEL, frames=["[Talmud Bavli]: Yoma 41 b – c"]),   # eligible only with frames
]


def write_merged(path: Path, records: Sequence[Dict[str, Any]], blank_after: Optional[int] = None) -> str:
    """Write records as a merged JSONL.

    :param path: Destination.
    :param records: Records.
    :param blank_after: Insert an empty line after this many records (not a record).
    :return: The path as a string.
    """
    lines = [json.dumps(r, ensure_ascii=False) + "\n" for r in records]
    if blank_after is not None:
        lines.insert(blank_after, "\n")
    path.write_text("".join(lines), encoding="utf-8")
    return str(path)


def item(rec: Dict[str, Any], frames: bool = False, version: str = V4) -> Any:
    """:param rec: Merged record.
    :param frames: Render flag.
    :param version: Text version.
    :return: Its ``EmbeddingText``.
    """
    return embedding_text(GenizahDocument.from_merged_format(rec), version, include_scholarly_frames=frames)


def build(tmp_path: Path, es: FakeES, embedder: Optional[FakeEmbedder] = None, **kwargs: Any) -> Dict[str, int]:
    """Run ``index_merged`` (semantic_v4) on :data:`RECORDS` into :data:`INDEX` of the fake cluster.

    :param tmp_path: Temporary directory.
    :param es: The fake cluster (routed by the ``fake_es`` fixture).
    :param embedder: Embedder (default: a new FakeEmbedder).
    :param kwargs: More ``index_merged`` arguments (they override the defaults here; without ``merged_path``,
        :data:`RECORDS` are written to ``tmp_path / "merged.jsonl"``).
    :return: ``index_merged``'s result.
    """
    if "merged_path" not in kwargs:
        kwargs["merged_path"] = write_merged(tmp_path / "merged.jsonl", RECORDS)
    options = {"index_name": INDEX, "text_version": V4, "embedding_model": embedder or FakeEmbedder(),
               "elasticsearch_config": {}, **kwargs}
    return img.index_merged(**options)


# --------------------------------------------------------------------------------------------- text hash (B2)
def test_every_document_carries_the_hash_of_its_text(tmp_path, fake_es):
    result = build(tmp_path, fake_es)
    assert result == {"indexed": 6, "without_vector": 3}
    stored = fake_es.docs(INDEX)
    for rec in RECORDS:
        expected = item(rec)
        source = stored[rec["canonical_id"]]
        assert source["embedded_text_sha256"] == expected.text_sha256 == sha(expected.text)
        assert source["semantic_eligible"] is expected.eligible
        assert ("embedding_vector" in source) is expected.eligible
    assert stored[BARE_ID]["embedded_text_sha256"] == sha("")              # the empty text has a hash too
    for cid in (BOILER_ID, BARE_ID, LABEL_FRAME_ID):
        assert stored[cid]["semantic_eligible"] is False and "embedding_vector" not in stored[cid]
    properties = fake_es.indices.created[INDEX]["mappings"]["properties"]
    assert properties["embedded_text_sha256"] == {"type": "keyword"} == eig.TEXT_VERSION_MAPPING["embedded_text_sha256"]


# --------------------------------------------------------------------------------------------- embed once (B4)
def test_a_duplicate_text_is_embedded_once_across_batches(tmp_path, fake_es):
    embedder = FakeEmbedder()
    build(tmp_path, fake_es, embedder, batch_size=1)          # the two Genesis records land in separate batches
    genesis = item(RECORDS[3]).text
    assert genesis == item(RECORDS[4]).text
    assert embedder.documents.count(genesis) == 1
    stored = fake_es.docs(INDEX)
    assert stored[DUP_A_ID]["embedding_vector"] == stored[DUP_B_ID]["embedding_vector"]
    assert np.allclose(stored[DUP_B_ID]["embedding_vector"], fake_vector(genesis))
    assert stored[DUP_A_ID]["embedded_text_sha256"] == stored[DUP_B_ID]["embedded_text_sha256"]


def test_the_processor_counts_reused_vectors_within_and_across_batches(fake_es):
    embedder = FakeEmbedder()
    docs = [GenizahDocument.from_merged_format(r) for r in RECORDS]
    meta = qte.build_index_meta(embedder, V4, None)
    processor = eig.ElasticsearchGenizahProcessor(embedder, elasticsearch_config={}, index_name=INDEX,
                                                  embedding_dims=DIMS, index_meta=meta, text_version=V4)
    embedder.calls.clear()
    processor.process_documents(docs, batch_size=10)
    assert embedder.calls == [[item(RECORDS[0]).text, item(RECORDS[3]).text]]    # eligible, distinct, in order
    assert (processor.stats["texts_embedded"], processor.stats["text_cache_hits"]) == (2, 1)
    processor.process_documents(docs[3:5], batch_size=10)                       # the Genesis pair again
    assert len(embedder.calls) == 1 and processor.stats["text_cache_hits"] == 3
    assert all(isinstance(v, np.ndarray) and v.dtype == np.float32 for v in processor.text_vectors.values())


def test_past_the_cache_cap_a_text_is_embedded_once_per_batch(fake_es):
    embedder = FakeEmbedder()
    meta = qte.build_index_meta(embedder, V4, None)
    processor = eig.ElasticsearchGenizahProcessor(embedder, elasticsearch_config={}, index_name=INDEX,
                                                  embedding_dims=DIMS, index_meta=meta, text_version=V4)
    processor.text_vectors_max = 0
    embedder.calls.clear()
    pair = [GenizahDocument.from_merged_format(r) for r in RECORDS[3:5]]
    processor.process_documents(pair, batch_size=2)
    processor.process_documents(pair, batch_size=2)
    assert embedder.calls == [[item(RECORDS[3]).text]] * 2 and processor.text_vectors == {}


# --------------------------------------------------------------------------------------------- _meta (B6)
def test_meta_records_the_hash_rule_text_function_source_and_final_counts(tmp_path, fake_es):
    path = write_merged(tmp_path / "merged.jsonl", RECORDS, blank_after=2)
    build(tmp_path, fake_es, merged_path=path)
    created = fake_es.indices.created[INDEX]["mappings"]["_meta"]
    meta = fake_es.indices.mappings[INDEX]["_meta"]
    assert "counts" not in created and {k: v for k, v in meta.items() if k != "counts"} == created
    assert meta["text_representation_version"] == V4
    assert meta["text_hash"] == qte.TEXT_HASH_RULE == "sha256 of the UTF-8 bytes of the embedded text (embedded_text_sha256)"
    canary = meta["document_canary"]
    assert canary["doc_id"] == FRAMED_ID and canary["text"] == item(RECORDS[0]).text
    assert canary["text_sha256"] == sha(canary["text"])
    fixture = Path(qte.__file__).resolve().parents[2] / "tests" / "data" / "semantic_v4_golden.jsonl"
    assert meta["text_function"] == {"version": V4, "scholarly_frames": False,
                                     "golden_fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                                     "hda_commit": COMMIT}
    assert meta["source"] == {"merged_path": path, "merged_records": 6,
                              "merged_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    assert meta["counts"] == {"documents": 6, "eligible": 3, "distinct_text_sha256": 5,
                              "distinct_eligible_text_sha256": 2}
    # counts come last: one _meta write after every bulk write
    assert fake_es.log[-1][0] == "put_mapping" and [op for op, *_ in fake_es.log].count("put_mapping") == 1
    assert "counts" in fake_es.log[-1][1]
    assert json.loads(json.dumps(meta)) == meta


def test_counts_cover_the_whole_index_after_a_resumed_build(tmp_path, fake_es):
    build(tmp_path, fake_es, limit=3)
    assert fake_es.indices.mappings[INDEX]["_meta"]["counts"]["documents"] == 3
    build(tmp_path, fake_es, skip=3)                          # the writer guard lets the matching run resume
    counts = fake_es.indices.mappings[INDEX]["_meta"]["counts"]
    assert counts == {"documents": 6, "eligible": 3, "distinct_text_sha256": 5, "distinct_eligible_text_sha256": 2}
    assert eig.distinct_values(fake_es, INDEX, "embedded_text_sha256", page_size=2) == 5        # three pages


def test_build_index_meta_defaults_and_rejects_a_mismatched_text_function():
    embedder = FakeEmbedder()
    meta = qte.build_index_meta(embedder, "semantic_v3", DocumentCanary(DUP_A_ID, "Description: a letter on flax."))
    assert meta["text_function"] == qte.text_function_block("semantic_v3")
    assert meta["text_function"]["scholarly_frames"] is False and meta["text_function"]["hda_commit"] == COMMIT
    assert "source" not in meta                                # only the merged indexer passes one
    with pytest.raises(ValueError, match="text_function"):
        qte.build_index_meta(embedder, V4, None, text_function=qte.text_function_block("semantic_v3"))


def test_golden_fixture_per_text_version():
    for version in ("semantic_v3", V4):
        assert qte.golden_fixture_sha256(version) == hashlib.sha256(
            Path(qte.GOLDEN_FIXTURES[version]).read_bytes()).hexdigest()
    assert qte.golden_fixture_sha256("semantic_v3") != qte.golden_fixture_sha256(V4)
    assert qte.golden_fixture_sha256("v8") is None             # the production text has no fixture


def test_git_commit_state_marks_a_dirty_checkout(monkeypatch):
    def runner(status_out: str, head_code: int = 0) -> Any:
        """:param status_out: ``git status --porcelain`` output.
        :param head_code: ``git rev-parse`` return code.
        :return: A fake ``subprocess.run``.
        """
        def run(cmd: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
            """:param cmd: Command.
            :param kwargs: Ignored.
            :return: Completed process.
            """
            assert cmd[:3] == ["git", "--no-optional-locks", "-C"]          # read-only: never takes index.lock
            if cmd[-2:] == ["rev-parse", "HEAD"]:
                return subprocess.CompletedProcess(cmd, head_code, stdout="abc123\n", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout=status_out, stderr="")
        return run

    monkeypatch.setattr(qte.shutil, "which", lambda name: "/usr/bin/git")
    assert qte.git_commit_state("/repo", run=runner("")) == "abc123"
    assert qte.git_commit_state("/repo", run=runner("?? src/new_file.py\n")) == "abc123-dirty"
    assert qte.git_commit_state("/repo", run=runner("", head_code=128)) is None      # not a checkout
    monkeypatch.setattr(qte.shutil, "which", lambda name: None)
    assert qte.git_commit_state("/repo", run=runner("")) is None                     # no git at all


def test_merged_source_counts_records_and_hashes_the_file(tmp_path, monkeypatch):
    path = write_merged(tmp_path / "merged.jsonl", RECORDS[:2], blank_after=1)
    monkeypatch.setattr(img, "_REPO_ROOT", str(tmp_path))
    assert img.merged_source(path) == {"merged_path": "merged.jsonl", "merged_records": 2,
                                       "merged_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


# --------------------------------------------------------------------------------------------- the render flag
def test_scholarly_frames_default_off_and_every_default_reads_the_constant():
    assert img.INDEX_SCHOLARLY_FRAMES is False
    parser = img.build_parser()
    assert parser.parse_args([]).include_scholarly_frames is False
    assert parser.parse_args(["--scholarly-frames"]).include_scholarly_frames is True
    assert parser.parse_args(["--no-scholarly-frames"]).include_scholarly_frames is False
    with pytest.raises(SystemExit):
        parser.parse_args(["--scholarly-frames", "--no-scholarly-frames"])

    def default(function: Any, name: str = "include_scholarly_frames") -> Any:
        """:param function: Callable.
        :param name: Parameter.
        :return: Its default.
        """
        return inspect.signature(function).parameters[name].default

    for function in (img.index_merged, img.find_document_canary, eig.ElasticsearchGenizahProcessor.__init__,
                     eig.pick_document_canary, qte.text_function_block, vri.merged_texts):
        assert default(function) is img.INDEX_SCHOLARLY_FRAMES, function.__qualname__


def test_the_flag_changes_the_canary_the_texts_and_the_meta(tmp_path, monkeypatch):
    path = write_merged(tmp_path / "merged.jsonl", RECORDS)
    off, on = img.find_document_canary(path, V4), img.find_document_canary(path, V4, include_scholarly_frames=True)
    assert off.doc_id == on.doc_id == FRAMED_ID
    assert "[Piyyut]: Avodah for Yom Kippur" in on.text and "Piyyut" not in off.text
    monkeypatch.chdir(tmp_path)
    builds: Dict[bool, FakeES] = {}
    for frames in (False, True):
        es = builds[frames] = FakeES()
        monkeypatch.setattr(eig, "Elasticsearch", lambda **kwargs: es)
        monkeypatch.setattr(eig, "bulk", lambda client, actions, refresh: (client.actions.extend(actions), (0, []))[1])
        img.index_merged(index_name=INDEX, merged_path=path, text_version=V4, include_scholarly_frames=frames,
                         embedding_model=FakeEmbedder(), elasticsearch_config={})
        meta = es.indices.mappings[INDEX]["_meta"]
        assert meta["text_function"]["scholarly_frames"] is frames and eig.index_scholarly_frames(meta) is frames
        assert meta["document_canary"]["text"] == (on if frames else off).text
    off_docs, on_docs = builds[False].docs(INDEX), builds[True].docs(INDEX)
    assert off_docs[FRAMED_ID]["embedded_text_sha256"] != on_docs[FRAMED_ID]["embedded_text_sha256"]
    assert on_docs[FRAMED_ID]["embedded_text_sha256"] == item(RECORDS[0], frames=True).text_sha256
    assert off_docs[LABEL_FRAME_ID]["semantic_eligible"] is False and on_docs[LABEL_FRAME_ID]["semantic_eligible"] is True
    assert off_docs[DUP_A_ID]["embedded_text_sha256"] == on_docs[DUP_A_ID]["embedded_text_sha256"]    # no frames


def test_the_writer_guard_refuses_resuming_with_the_other_flag(tmp_path, fake_es):
    build(tmp_path, fake_es, limit=2)
    embedder = FakeEmbedder()
    written = len(fake_es.actions)
    with pytest.raises(eig.IndexMetaMismatchError, match="scholarly_frames: index has False, this run would write True"):
        build(tmp_path, fake_es, embedder, skip=2, include_scholarly_frames=True)
    assert len(fake_es.actions) == written
    existing = fake_es.indices.mappings[INDEX]["_meta"]
    assert eig.index_meta_mismatches({k: v for k, v in existing.items() if k != "text_function"},
                                     dict(existing, text_function={"scholarly_frames": True})) == []   # pre-v10 index


def test_the_processor_refuses_a_meta_built_for_the_other_flag(fake_es):
    meta = qte.build_index_meta(FakeEmbedder(), V4, None,
                                text_function=qte.text_function_block(V4, include_scholarly_frames=True))
    with pytest.raises(ValueError, match="scholarly_frames"):
        eig.ElasticsearchGenizahProcessor(FakeEmbedder(), elasticsearch_config={}, index_name=INDEX,
                                          index_meta=meta, text_version=V4)


def test_only_fragment_documents_take_the_flag():
    fragment = GenizahDocument.from_merged_format(RECORDS[0])
    page = types.SimpleNamespace(embedding_text_kind="bibliography")
    assert eig.document_scholarly_frames(fragment, False) is False and eig.document_scholarly_frames(fragment, True)
    assert eig.document_scholarly_frames(page, False) is True       # a page has one rendering (no keyword passed)


# --------------------------------------------------------------------------------------------- index name
def test_v10_is_live_the_builder_defaults_to_v11_and_the_updaters_to_the_live_index(monkeypatch):
    """The served index is protected like its predecessors; a default build goes to a new name; the in-place updaters
    target the served index and therefore always need --i-know-this-is-live."""
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)   # index_bibliography loads it on import
    assert img.LIVE_INDEX_NAME == "genizah_merged_v10" and img.LIVE_INDEX_NAME in img.LIVE_INDEX_NAMES
    assert img.DEFAULT_INDEX_NAME == "genizah_merged_v11" and img.DEFAULT_INDEX_NAME not in img.LIVE_INDEX_NAMES
    assert img.build_parser().parse_args([]).index == img.DEFAULT_INDEX_NAME
    assert vri.build_parser().parse_args([]).genizah_index == img.DEFAULT_INDEX_NAME
    assert ugd.build_parser().parse_args(["--generated", "g.jsonl"]).index == img.LIVE_INDEX_NAME
    for live in ("genizah_merged_v8", "genizah_merged_v10"):
        with pytest.raises(ValueError, match="genizah_merged_v11"):
            img.check_index_name(live)
    assert img.check_index_name("genizah_merged_v10", allow_live=True) == "genizah_merged_v10"


# --------------------------------------------------------------------------------------------- the verifier
def built_index(tmp_path: Path, es: FakeES, embedder: FakeEmbedder) -> str:
    """Build :data:`INDEX` (frames off) and return the merged path.

    :param tmp_path: Temporary directory.
    :param es: The fake cluster.
    :param embedder: Embedder.
    :return: The merged file the index was built from.
    """
    path = write_merged(tmp_path / "merged.jsonl", RECORDS)
    build(tmp_path, es, embedder, merged_path=path)
    return path


def test_self_consistency_passes_on_the_built_index(tmp_path, fake_es, capsys):
    embedder = FakeEmbedder()
    path = built_index(tmp_path, fake_es, embedder)
    meta = fake_es.indices.mappings[INDEX]["_meta"]
    assert vri.check_self_consistency_genizah(fake_es, embedder, INDEX, V4, n=3, merged_path=path,
                                              include_scholarly_frames=eig.index_scholarly_frames(meta),
                                              require_hash=True)
    out = capsys.readouterr().out
    assert "[PASS] genizah_merged_v10_test: text drift over 3 hashed docs: 3 match" in out
    assert "[PASS] genizah_merged_v10_test: vector drift over 3 docs" in out


def test_text_drift_fails_with_the_ids_and_the_rest_is_still_cosine_checked(tmp_path, fake_es, capsys):
    embedder = FakeEmbedder()
    built_index(tmp_path, fake_es, embedder)
    changed = copy.deepcopy(RECORDS)
    changed[4]["description"] = GENESIS + " With a colophon."                # the merged file moved on
    path = write_merged(tmp_path / "changed.jsonl", changed)
    assert not vri.check_self_consistency_genizah(fake_es, embedder, INDEX, V4, n=3, merged_path=path,
                                                  require_hash=True)
    out = capsys.readouterr().out
    assert f"[FAIL] genizah_merged_v10_test: text drift over 3 hashed docs: 2 match the recomputed text; " \
           f"TEXT DRIFT in 1: {DUP_B_ID}" in out
    assert "[PASS] genizah_merged_v10_test: vector drift over 2 docs" in out      # model drift reported separately


def test_a_tampered_hash_or_the_wrong_render_flag_is_text_drift(tmp_path, fake_es, capsys):
    embedder = FakeEmbedder()
    path = built_index(tmp_path, fake_es, embedder)
    assert not vri.check_self_consistency_genizah(fake_es, embedder, INDEX, V4, n=3, merged_path=path,
                                                  include_scholarly_frames=True, require_hash=True)
    assert f"TEXT DRIFT in 1: {FRAMED_ID}" in capsys.readouterr().out        # frames on, but the index has them off
    fake_es.docs(INDEX)[DUP_A_ID]["embedded_text_sha256"] = "0" * 64
    assert not vri.check_self_consistency_genizah(fake_es, embedder, INDEX, V4, n=3, merged_path=path)
    assert f"TEXT DRIFT in 1: {DUP_A_ID}" in capsys.readouterr().out


def test_a_missing_hash_fails_only_when_the_index_promises_one(tmp_path, fake_es, capsys):
    embedder = FakeEmbedder()
    path = built_index(tmp_path, fake_es, embedder)
    for source in fake_es.docs(INDEX).values():
        source.pop("embedded_text_sha256")
    assert vri.check_self_consistency_genizah(fake_es, embedder, INDEX, V4, n=3, merged_path=path)
    assert "[SKIP]" in capsys.readouterr().out                                  # a v9 index: vector check only
    assert not vri.check_self_consistency_genizah(fake_es, embedder, INDEX, V4, n=3, merged_path=path,
                                                  require_hash=True)
    assert "3 without embedded_text_sha256" in capsys.readouterr().out


def test_vector_drift_is_reported_as_such(tmp_path, fake_es, capsys):
    path = built_index(tmp_path, fake_es, FakeEmbedder())
    assert not vri.check_self_consistency_genizah(fake_es, FakeEmbedder(salt="another model"), INDEX, V4, n=3,
                                                  merged_path=path, require_hash=True)
    out = capsys.readouterr().out
    assert "[PASS] genizah_merged_v10_test: text drift" in out and "[FAIL] genizah_merged_v10_test: vector drift" in out


def test_a_generated_description_is_recomputed_from_its_stored_text(tmp_path, fake_es, capsys):
    embedder = FakeEmbedder()
    path = built_index(tmp_path, fake_es, embedder)
    generated = "Description: Letter to Yosef about pepper and oil sent by ship from Fustat. [machine-generated]"
    text = embed_text_for_version(generated, V4).text
    fake_es.docs(INDEX)[BOILER_ID].update({
        "description_source": "generated_from_consensus", "generated_description": generated,
        "semantic_eligible": True, "embedding_vector": fake_vector(text).tolist(), "embedded_text_sha256": sha(text)})
    assert vri.check_self_consistency_genizah(fake_es, embedder, INDEX, V4, n=4, merged_path=path, require_hash=True)
    assert "text drift over 4 hashed docs: 4 match" in capsys.readouterr().out


def test_document_canary_checks_its_text_hash(tmp_path, fake_es):
    embedder = FakeEmbedder()
    path = built_index(tmp_path, fake_es, embedder)
    meta = fake_es.indices.mappings[INDEX]["_meta"]

    def recompute(doc_id: str) -> Optional[str]:
        """:param doc_id: Canonical id.
        :return: Its text under the index's version and flag.
        """
        return vri.merged_texts([doc_id], V4, path, eig.index_scholarly_frames(meta)).get(doc_id)

    assert vri.check_document_canary(embedder, INDEX, meta, recompute)
    tampered = dict(meta, document_canary=dict(meta["document_canary"], text_sha256="0" * 64))
    assert not vri.check_document_canary(embedder, INDEX, tampered, recompute)
    legacy = dict(meta, document_canary={k: v for k, v in meta["document_canary"].items() if k != "text_sha256"})
    assert vri.check_document_canary(embedder, INDEX, legacy, recompute)        # a v9 canary has no hash


def test_meta_contract_needs_the_hash_rule_and_text_function(tmp_path, fake_es, capsys):
    embedder = FakeEmbedder()
    path = built_index(tmp_path, fake_es, embedder)
    meta = fake_es.indices.mappings[INDEX]["_meta"]
    assert vri.check_meta_contract(INDEX, meta, embedder)
    out = capsys.readouterr().out
    assert f"hda_commit {COMMIT}" in out and "local fixture identical" in out and "scholarly frames off" in out
    for broken in ({k: v for k, v in meta.items() if k != "text_hash"},
                   {k: v for k, v in meta.items() if k != "text_function"},
                   dict(meta, text_hash="md5 of the text"),
                   dict(meta, text_function=dict(meta["text_function"], version="semantic_v3"))):
        assert not vri.check_meta_contract(INDEX, broken, embedder)

    def recompute(doc_id: str, version: str) -> Optional[str]:
        """:param doc_id: Canonical id.
        :param version: Text version.
        :return: The record's text with the index's render flag.
        """
        return vri.merged_texts([doc_id], version, path, eig.index_scholarly_frames(meta)).get(doc_id)

    assert vri.verify_index(fake_es, embedder, INDEX, recompute)
