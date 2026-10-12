# File name: test_update_catalogue_enrichment.py
# Date: 10/9/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Catalogue enrichment updater: decisions, writer guard, canary, field sets, text-hash change detection, dry run,
idempotency, CLI.

No network and no model: Elasticsearch, the bulk helper and the embedder are faked. No ``.env`` is read.
"""
import copy
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest

from src.datasets.consensus.describe_from_reads import DESCRIPTION_SOURCE
from src.datasets.document_models.genizah_document import GenizahDocument
from src.datasets.indexing import update_catalogue_enrichment as uce
from src.datasets.indexing.elastic_index_genizah import (
    ALT_TITLES_MAPPING,
    GENRES_MAPPING,
    SCHOLARLY_FRAMES_MAPPING,
    SUBJECTS_MAPPING,
    IndexMetaMismatchError,
)
from src.embeddings import qwen_text_embedding as qte
from src.embeddings.text_representation import embedding_text

DIMS = 8
INDEX = "genizah_merged_v9_test"
CANARY = "canary string"
LETTER = "Letter from Nahray b. Nissim to his partner in Alexandria about flax shipments and prices"
DEED = "Deed of sale of a house in Fustat witnessed by the court of the Palestinian Jews"
LABEL = "קטע גניזה"            # a label of the frozen boilerplate list: no vector on its own
SHORT = "אלמקדמאת"             # specific, but under the 40-character floor (semantic_v3 measures the whole text)
NEW_FIELDS = ["scholarly_frames", "subjects", "genres", "embedded_text_sha256", "catalogue_fields_updated_at"]


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


class FakeIndices:
    """``es.indices``: mappings by index, and a record of ``put_mapping`` calls."""

    def __init__(self) -> None:
        """Start with no index."""
        self.mappings: Dict[str, Dict[str, Any]] = {}
        self.put: List[Dict[str, Any]] = []
        self.put_meta: List[Dict[str, Any]] = []

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

    def put_mapping(self, index: str, properties: Optional[Dict[str, Any]] = None,
                    meta: Optional[Dict[str, Any]] = None) -> None:
        """:param index: Index.
        :param properties: New fields (``_meta`` is not touched by a properties-only call, as in Elasticsearch).
        :param meta: A whole ``_meta`` block to store.
        """
        if properties:
            self.put.append(copy.deepcopy(properties))
            self.mappings[index].setdefault("properties", {}).update(properties)
        if meta is not None:
            self.put_meta.append(copy.deepcopy(meta))
            self.mappings[index]["_meta"] = copy.deepcopy(meta)


class FakeES:
    """In-memory index: documents by id and ``mget`` (honouring ``source_includes`` on top-level fields)."""

    def __init__(self) -> None:
        """Start empty."""
        self.indices = FakeIndices()
        self.docs: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.bulk_calls: List[List[Dict[str, Any]]] = []
        self.mget_calls: List[Dict[str, Any]] = []

    def mget(self, index: str, ids: Sequence[str], source_includes: Optional[List[str]] = None) -> Dict[str, Any]:
        """:param index: Index.
        :param ids: Ids.
        :param source_includes: Top-level ``_source`` fields to return.
        :return: ``{"docs": [...]}``.
        """
        self.mget_calls.append({"ids": list(ids), "source_includes": list(source_includes or [])})
        store = self.docs.get(index, {})
        out = []
        for i in ids:
            if i not in store:
                out.append({"_id": i, "found": False})
                continue
            source = {k: copy.deepcopy(v) for k, v in store[i].items() if source_includes is None or k in source_includes}
            out.append({"_id": i, "found": True, "_source": source})
        return {"docs": out}


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


def merged_record(cid: str, n: int, description: str, frames: Sequence[str] = (), subjects: Sequence[str] = (),
                  tags: str = "", domains: Sequence[str] = (), alt_titles: Sequence[str] = ()) -> Dict[str, Any]:
    """A merged record in the shape of ``tests/test_alt_titles._merged`` with KTIV frames, subjects, domains, varying
    titles and PGP tags.

    :param cid: Canonical id.
    :param n: Shelf-mark number.
    :param description: Merged description.
    :param frames: KTIV scholarly frames (become ``scholarly_frames``, listed on the ``Alternate titles:`` line).
    :param subjects: KTIV catalogue subjects.
    :param tags: PGP tag string.
    :param domains: KTIV scholarly domain paths, one scholarly entry each (``"Rabbinic Literature # Talmud Bavli"``;
        their levels become ``genres``).
    :param alt_titles: KTIV varying forms of the title (become ``alt_titles``).
    :return: Record.
    """
    entries = [{"subsections": {"writing_characteristics": {"frame": frame}}} for frame in frames]
    entries += [{"subsections": {"writing_characteristics": {"domain": domain}}} for domain in domains]
    return {
        "canonical_id": cid,
        "shelfmark_display": f"Cambridge University Library, Cambridge, England Ms. T-S NS 219.{n}",
        "institution": "Cambridge University Library",
        "sources_present": ["ktiv"],
        "description": description,
        "date": None,
        "images": {"ktiv": {"image_urls": []}, "preferred_source": None},
        "sources": {
            "pgp": {"documents": [{"tags": tags}]} if tags else None, "fjp": [], "bodleian": None,
            "ktiv": {"sys_num": "990051590520205171",
                     "basic_catalog": {"title": LABEL, "varying_form_of_title": list(alt_titles)},
                     "full_catalog": {"subjects": list(subjects)},
                     "scholarly_entries": entries},
        },
    }


def standard_records() -> List[Dict[str, Any]]:
    """One merged record per decision, in file order.

    Under the frozen ``semantic_v3`` rule a ``Subjects:`` line is content and counts towards the 40-character floor, so
    the records that must stay without a vector carry nothing that makes them eligible: ``Doc_ShortNoFlip`` and
    ``Doc_Refused`` one short keyword (their texts stay under the floor: ``short``), ``Doc_FieldsOnly`` a varying title
    that is itself a frozen label (``boilerplate_only``).

    :return: Records.
    """
    return [
        merged_record("Doc_Reembed", 1, LETTER, frames=["[Talmud Bavli]: Megillah 2 a – b"], subjects=["Jewish law"],
                      domains=["Rabbinic Literature # Talmud Bavli"]),
        merged_record("Doc_Flip", 2, LABEL, frames=["[Talmud Bavli]: Sukkah 29 a – b"]),
        merged_record("Doc_ShortNoFlip", 3, SHORT, subjects=["Poetry"]),
        merged_record("Doc_FieldsOnly", 4, LABEL, alt_titles=["Unidentified"]),
        merged_record("Doc_Refused", 5, LABEL, tags="Bible"),
        merged_record("Doc_Unchanged", 6, LETTER, frames=["[Bible]: Numbers Bemidbar 2:12 – 3:29"], subjects=["Bible"],
                      domains=["Bible: Texts # Bible: Texts and Translations"]),
        merged_record("Doc_Generated", 7, LABEL, frames=["[Talmud Bavli]: Yoma 41 b – c"]),
        merged_record("Doc_Description", 8, DEED),
        merged_record("Doc_Missing", 9, LETTER, subjects=["Jewish law"]),
    ]


def document(record: Dict[str, Any]) -> GenizahDocument:
    """:param record: Merged record.
    :return: Its document under the current model.
    """
    return GenizahDocument.from_merged_format(record)


def v9_source(record: Dict[str, Any], eligible: bool, **overrides: Any) -> Dict[str, Any]:
    """The ``_source`` a v9-era build holds for a record: today's description, no enrichment fields, no hash.

    :param record: Merged record.
    :param eligible: ``semantic_eligible`` (a vector is stored when True).
    :param overrides: Fields to replace.
    :return: ``_source``.
    """
    source = {"doc_id": record["canonical_id"], "description": document(record).description, "alt_titles": [],
              "full_text_content": "old full text", "semantic_eligible": eligible,
              "text_representation_version": "semantic_v3"}
    if eligible:
        source["embedding_vector"] = fake_vector("old " + record["canonical_id"]).tolist()
    source.update(overrides)
    return source


def write_merged(path: Path, records: Sequence[Dict[str, Any]]) -> Path:
    """:param path: Destination.
    :param records: Merged records.
    :return: The path.
    """
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return path


def no_model(*args: Any, **kwargs: Any) -> None:
    """Stands in for the model loader in runs that must not load it.

    :raises AssertionError: Always.
    """
    raise AssertionError("the embedder was loaded")


@pytest.fixture
def records() -> Dict[str, Dict[str, Any]]:
    """:return: The standard merged records by canonical id."""
    return {r["canonical_id"]: r for r in standard_records()}


@pytest.fixture
def merged(tmp_path: Path) -> Path:
    """:param tmp_path: Temporary directory.
    :return: The standard merged JSONL.
    """
    return write_merged(tmp_path / "merged.jsonl", standard_records())


@pytest.fixture
def es(monkeypatch: pytest.MonkeyPatch, records: Dict[str, Dict[str, Any]]) -> FakeES:
    """A v9-era index holding every standard record but ``Doc_Missing``; its mapping lacks ``subjects``.

    :param monkeypatch: pytest monkeypatch (routes the module's bulk helper to the fake).
    :param records: Standard merged records.
    :return: The fake cluster.
    """
    cluster = FakeES()
    cluster.indices.mappings[INDEX] = {"_meta": index_meta(FakeEmbedder()), "properties": {
        "description": {"type": "text"}, **ALT_TITLES_MAPPING,
        "embedding_vector": {"type": "dense_vector", "dims": DIMS}}}
    unchanged = document(records["Doc_Unchanged"])
    cluster.docs[INDEX] = {
        "Doc_Reembed": v9_source(records["Doc_Reembed"], eligible=True),
        "Doc_Flip": v9_source(records["Doc_Flip"], eligible=False),
        "Doc_ShortNoFlip": v9_source(records["Doc_ShortNoFlip"], eligible=False),
        "Doc_FieldsOnly": v9_source(records["Doc_FieldsOnly"], eligible=False),
        "Doc_Refused": v9_source(records["Doc_Refused"], eligible=True),
        "Doc_Unchanged": v9_source(records["Doc_Unchanged"], eligible=True, alt_titles=unchanged.alt_titles,
                                   scholarly_frames=unchanged.scholarly_frames, subjects=unchanged.subjects,
                                   genres=unchanged.genres),
        "Doc_Generated": v9_source(records["Doc_Generated"], eligible=True, description_source=DESCRIPTION_SOURCE,
                                   generated_description="Description: a letter [machine-generated]"),
        "Doc_Description": v9_source(records["Doc_Description"], eligible=True, description="Old catalogue label"),
    }

    def fake_bulk(client: FakeES, actions: List[Dict[str, Any]], refresh: str) -> tuple:
        """:param client: Fake cluster.
        :param actions: Bulk update actions.
        :param refresh: Must be ``wait_for``.
        :return: (successes, failures).
        """
        assert refresh == "wait_for"
        client.bulk_calls.append(copy.deepcopy(actions))
        for action in actions:
            assert action["_op_type"] == "update" and action["_id"] in client.docs[action["_index"]]
            client.docs[action["_index"]][action["_id"]].update(copy.deepcopy(action["doc"]))
        return len(actions), []

    monkeypatch.setattr(uce, "bulk", fake_bulk)
    return cluster


def run(merged_path: Path, es_client: FakeES, **kwargs: Any) -> Dict[str, Any]:
    """:param merged_path: Merged JSONL.
    :param es_client: Fake cluster.
    :param kwargs: Further :func:`update_catalogue_enrichment` arguments.
    :return: Its summary.
    """
    kwargs.setdefault("text_version", "semantic_v3")      # the fixtures are a v9-era (semantic_v3) index
    return uce.update_catalogue_enrichment(merged_path, index_name=INDEX, es_client=es_client, **kwargs)


def target(**kwargs: Any) -> uce.EnrichedRecord:
    """A target state for the pure decision tests.

    :param kwargs: Fields to replace.
    :return: Record.
    """
    values = {"doc_id": "D", "description": "d", "alt_titles": ["a"], "scholarly_frames": ["f"], "subjects": ["s"],
              "genres": ["g"], "full_text_content": "d a f s g",
              "text": "Description: d\nAlternate titles: a; f\nSubjects: s", "eligible": True}
    values.update(kwargs)
    return uce.EnrichedRecord(**values)


SAME = {"description": "d", "alt_titles": ["a"], "scholarly_frames": ["f"], "subjects": ["s"], "genres": ["g"]}
HASH = hashlib.sha256(target().text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------------- decisions


@pytest.mark.parametrize("state, record, expected", [
    (None, target(), "missing"),
    # a generated description holds the vector: catalogue fields only, never the vector
    ({**SAME, "alt_titles": [], "description_source": DESCRIPTION_SOURCE, "semantic_eligible": True}, target(),
     "generated"),
    ({**SAME, "description_source": DESCRIPTION_SOURCE, "semantic_eligible": True, "embedded_text_sha256": "x"},
     target(), "unchanged"),
    # no stored hash (every v9 document): the catalogue fields decide
    ({**SAME, "semantic_eligible": True}, target(), "unchanged"),
    ({**SAME, "alt_titles": [], "semantic_eligible": True}, target(), "update/reembed"),
    ({"description": "d", "alt_titles": ["a"], "scholarly_frames": ["f"], "semantic_eligible": False}, target(),
     "update/reembed"),                                                                        # subjects absent
    ({"description": "d", "alt_titles": ["a"], "subjects": ["s"], "genres": ["g"], "semantic_eligible": False},
     target(), "update/reembed"),                                     # scholarly_frames absent (every v9 document)
    ({**SAME, "scholarly_frames": [], "semantic_eligible": True}, target(), "update/reembed"),
    ({**SAME, "description": "old label", "semantic_eligible": True}, target(), "update/reembed"),
    ({**SAME, "subjects": [], "semantic_eligible": False}, target(eligible=False), "update/fields_only"),
    ({**SAME, "subjects": []}, target(eligible=False), "update/fields_only"),                           # flag absent
    ({**SAME, "subjects": [], "semantic_eligible": True}, target(eligible=False), "refused:would_lose_vector"),
    # genres are compared like the other fields; without a hash a genres-only change counts as a text change
    ({**SAME, "genres": [], "semantic_eligible": True}, target(), "update/reembed"),
    ({"description": "d", "alt_titles": ["a"], "scholarly_frames": ["f"], "subjects": ["s"],
      "semantic_eligible": False}, target(eligible=False), "update/fields_only"),                  # genres absent
    # a stored hash: the text decides the vector, the fields decide a field update
    ({**SAME, "semantic_eligible": True, "embedded_text_sha256": HASH}, target(), "unchanged"),
    ({**SAME, "subjects": ["t"], "semantic_eligible": True, "embedded_text_sha256": HASH}, target(),
     "update/fields_only"),
    ({**SAME, "genres": ["h"], "semantic_eligible": True, "embedded_text_sha256": HASH}, target(),
     "update/fields_only"),                                                     # genres never reach the hashed text
    ({**SAME, "semantic_eligible": True, "embedded_text_sha256": "0" * 64}, target(), "update/reembed"),
    ({**SAME, "semantic_eligible": True, "embedded_text_sha256": "0" * 64}, target(eligible=False),
     "refused:would_lose_vector"),
    ({**SAME, "semantic_eligible": False, "embedded_text_sha256": "0" * 64}, target(eligible=False), "unchanged"),
    ({**SAME, "subjects": [], "semantic_eligible": False, "embedded_text_sha256": "0" * 64}, target(eligible=False),
     "update/fields_only"),
])
def test_decide_covers_every_branch(state, record, expected):
    assert uce.decide(state, record) == expected


def test_flips_change_reasons_and_eligibility_drift():
    assert uce.change_reasons({**SAME, "description": "old", "subjects": []}, target()) == ["description", "subjects"]
    assert uce.change_reasons(SAME, target()) == ["text_only"]
    assert uce.change_reasons({**SAME, "genres": []}, target()) == ["genres"]
    assert uce.change_reasons({**SAME, "scholarly_frames": [], "subjects": []}, target()) == ["scholarly_frames",
                                                                                              "subjects"]
    assert uce.is_flip({"semantic_eligible": False}, "update/reembed")
    assert uce.is_flip({}, "update/reembed")                                    # flag absent counts as no vector
    assert not uce.is_flip({"semantic_eligible": True}, "update/reembed")
    assert not uce.is_flip({"semantic_eligible": False}, "update/fields_only")
    assert uce.eligibility_drift({**SAME, "semantic_eligible": False}, target(), "unchanged")
    assert not uce.eligibility_drift({**SAME, "semantic_eligible": True}, target(), "unchanged")
    assert not uce.eligibility_drift({"semantic_eligible": False}, target(), "update/reembed")


def test_summary_counts_examples_text_versions_and_canary():
    summary = uce.EnrichmentSummary(index=INDEX, text_version="semantic_v3", merged_path="m", dry_run=True,
                                    document_canary={"doc_id": "D", "text": "not scanned",
                                                     "expected_text": target().text})
    for i in range(7):
        summary.add({"semantic_eligible": False, "text_representation_version": "v8"}, target(doc_id=f"F{i}"),
                    "update/reembed")
    summary.add({**SAME, "semantic_eligible": True}, target(), "unchanged")
    data = summary.to_dict()
    assert (data["scanned"], data["flips"], data["other_text_version"]) == (8, 7, 7)
    assert data["examples"]["flips"] == ["F0", "F1", "F2", "F3", "F4"]       # five examples per category
    assert data["document_canary"] == {"doc_id": "D", "text": "unchanged"}
    assert json.loads(json.dumps(data)) == data


# --------------------------------------------------------------------------------------------- runs


def test_dry_run_reports_without_loading_the_model_or_writing(tmp_path, merged, es, monkeypatch, capsys):
    monkeypatch.setattr(qte, "QwenTextEmbedding", no_model)
    before = copy.deepcopy(es.docs)
    report = tmp_path / "report.json"
    result = run(merged, es, dry_run=True, report_path=report)
    assert result["counts"] == {"missing": 1, "generated": 1, "unchanged": 1, "update/reembed": 3,
                                "update/fields_only": 2, "refused:would_lose_vector": 1}
    assert (result["scanned"], result["flips"], result["refused"], result["eligibility_drift"]) == (9, 1, 1, 0)
    assert result["examples"]["flips"] == ["Doc_Flip"]
    assert result["examples"]["update/reembed"] == ["Doc_Reembed", "Doc_Flip", "Doc_Description"]
    assert result["examples"]["update/fields_only"] == ["Doc_ShortNoFlip", "Doc_FieldsOnly"]
    assert result["examples"]["missing"] == ["Doc_Missing"]
    assert result["changed"] == {"description": 1, "alt_titles": 1, "scholarly_frames": 3, "subjects": 3, "genres": 1,
                                 "text_only": 0}
    assert result["mapping_missing"] == NEW_FIELDS and result["updated"] == {
        "update/reembed": 0, "update/fields_only": 0, "generated": 0}
    assert result["dry_run"] and result["canary_cosine"] is None
    assert es.docs == before and es.bulk_calls == [] and es.indices.put == []
    assert json.loads(report.read_text(encoding="utf-8")) == result
    assert all(len(c["ids"]) <= uce.MGET_CHUNK and c["source_includes"] == uce.STATE_FIELDS for c in es.mget_calls)
    out = capsys.readouterr().out
    assert "update/reembed 3" in out and "eligibility flips (indexed without a vector, eligible now): 1" in out
    assert "flips: Doc_Flip" in out and "dry run: nothing embedded or written" in out
    assert ("lacks scholarly_frames, subjects, genres, embedded_text_sha256, catalogue_fields_updated_at "
            "(a real run adds them)") in out


def test_real_run_writes_vectors_flags_hash_and_fields(merged, es, records):
    embedder = FakeEmbedder()
    before = copy.deepcopy(es.docs)
    result = run(merged, es, embedding_model=embedder)
    assert result["updated"] == {"update/reembed": 3, "update/fields_only": 2, "generated": 1}
    assert result["canary_cosine"] > uce.CANARY_COSINE_FLOOR
    stored = es.docs[INDEX]
    reembedded = ["Doc_Reembed", "Doc_Flip", "Doc_Description"]
    texts = {cid: embedding_text(document(records[cid]), "semantic_v3").text for cid in reembedded}
    assert embedder.documents == [CANARY] + [texts[cid] for cid in reembedded]   # the canary, then only these
    for cid in ["Doc_Reembed", "Doc_Flip", "Doc_ShortNoFlip", "Doc_Description", "Doc_FieldsOnly", "Doc_Generated"]:
        doc = document(records[cid])
        assert (stored[cid]["description"], stored[cid]["alt_titles"], stored[cid]["scholarly_frames"],
                stored[cid]["subjects"], stored[cid]["genres"], stored[cid]["full_text_content"]) == (
            doc.description, doc.alt_titles, doc.scholarly_frames, doc.subjects, doc.genres,
            doc.create_full_text_content())
        assert stored[cid]["catalogue_fields_updated_at"].endswith("+00:00")
    for cid in reembedded:
        assert np.allclose(stored[cid]["embedding_vector"], fake_vector(texts[cid]))
        assert stored[cid]["semantic_eligible"] is True and stored[cid]["text_representation_version"] == "semantic_v3"
        assert stored[cid]["embedded_text_sha256"] == hashlib.sha256(texts[cid].encode("utf-8")).hexdigest()
    assert stored["Doc_Description"]["description"] == DEED
    # genres are written and reach full_text_content, never the embedded text
    assert stored["Doc_Reembed"]["genres"] == ["Rabbinic Literature", "Talmud Bavli"]
    assert "Rabbinic Literature" in stored["Doc_Reembed"]["full_text_content"]
    assert "Rabbinic Literature" not in texts["Doc_Reembed"]
    # a short label plus one keyword stays without a vector: semantic_v3 counts the Subjects line towards the floor,
    # and the text is still under it
    assert stored["Doc_ShortNoFlip"]["subjects"] == ["Poetry"]
    assert "embedding_vector" not in stored["Doc_ShortNoFlip"] and stored["Doc_ShortNoFlip"]["semantic_eligible"] is False
    # fields only (a varying title that is itself a frozen label): no vector, flag and hash untouched
    assert stored["Doc_FieldsOnly"]["alt_titles"] == ["Unidentified"]
    assert "embedding_vector" not in stored["Doc_FieldsOnly"] and stored["Doc_FieldsOnly"]["semantic_eligible"] is False
    assert "embedded_text_sha256" not in stored["Doc_FieldsOnly"]
    # a generated description keeps its vector and flags
    generated = stored["Doc_Generated"]
    assert generated["embedding_vector"] == before[INDEX]["Doc_Generated"]["embedding_vector"]
    assert generated["semantic_eligible"] is True and generated["description_source"] == DESCRIPTION_SOURCE
    assert generated["alt_titles"] == [] and generated["scholarly_frames"] == ["[Talmud Bavli]: Yoma 41 b – c"]
    assert "embedded_text_sha256" not in generated
    # refused, unchanged and missing documents are not written; nothing is created
    assert stored["Doc_Refused"] == before[INDEX]["Doc_Refused"]
    assert stored["Doc_Unchanged"] == before[INDEX]["Doc_Unchanged"]
    assert set(stored) == set(before[INDEX])
    # mapping completed (properties only), _meta untouched; eligible flags match vectors
    assert es.indices.put == [{name: uce.ENRICHMENT_MAPPING[name] for name in NEW_FIELDS}]
    assert es.indices.put[0]["scholarly_frames"] == SCHOLARLY_FRAMES_MAPPING["scholarly_frames"]
    assert es.indices.put[0]["subjects"] == SUBJECTS_MAPPING["subjects"]
    assert es.indices.put[0]["genres"] == GENRES_MAPPING["genres"]
    assert es.indices.mappings[INDEX]["_meta"] == index_meta(FakeEmbedder())
    assert all((d.get("semantic_eligible") is True) == ("embedding_vector" in d) for d in stored.values())


def test_a_second_run_finds_every_written_document_unchanged(merged, es):
    run(merged, es, embedding_model=FakeEmbedder())
    calls = len(es.bulk_calls)
    embedder = FakeEmbedder()
    again = run(merged, es, embedding_model=embedder)
    assert again["counts"] == {"missing": 1, "generated": 0, "unchanged": 7, "update/reembed": 0,
                               "update/fields_only": 0, "refused:would_lose_vector": 1}
    assert again["updated"] == {"update/reembed": 0, "update/fields_only": 0, "generated": 0}
    assert again["mapping_missing"] == [] and len(es.bulk_calls) == calls and embedder.documents == [CANARY]


def test_the_stored_hash_catches_text_changes_outside_the_catalogue_fields(tmp_path, es, records):
    first = write_merged(tmp_path / "first.jsonl", standard_records())
    run(first, es, embedding_model=FakeEmbedder())
    changed = standard_records()
    for record in changed:
        if record["canonical_id"] in ("Doc_Reembed", "Doc_Unchanged"):   # with and without a stored hash
            record["sources"]["ktiv_transcription"] = {"pages": [{"text": "ליוסף אחי\nשלום רב", "image_name": "001r"}],
                                                       "page_count": 1}
    result = run(write_merged(tmp_path / "second.jsonl", changed), es, embedding_model=FakeEmbedder())
    assert result["counts"]["update/reembed"] == 1 and result["examples"]["update/reembed"] == ["Doc_Reembed"]
    assert result["counts"]["unchanged"] == 6 and "Doc_Unchanged" in result["examples"]["unchanged"]   # no hash: fields decide
    assert result["changed"] == {"description": 0, "alt_titles": 0, "scholarly_frames": 0, "subjects": 1, "genres": 0,
                                 "text_only": 1}                                       # Doc_Refused: subjects
    text = embedding_text(document(changed[0]), "semantic_v3").text
    assert "Transcription: ליוסף אחי שלום רב" in text
    stored = es.docs[INDEX]["Doc_Reembed"]
    assert stored["embedded_text_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert np.allclose(stored["embedding_vector"], fake_vector(text))
    assert "ליוסף" in stored["full_text_content"]


def test_a_genres_only_change_is_a_field_update_on_a_hashed_document(tmp_path, es, records):
    run(write_merged(tmp_path / "first.jsonl", standard_records()), es, embedding_model=FakeEmbedder())
    hashed_before = copy.deepcopy(es.docs[INDEX]["Doc_Reembed"])
    changed = standard_records()
    for record in changed:
        if record["canonical_id"] in ("Doc_Reembed", "Doc_Unchanged"):   # with and without a stored hash
            record["sources"]["ktiv"]["scholarly_entries"].append(
                {"subsections": {"writing_characteristics": {"domain": "Halakhah # Responsa"}}})
    embedder = FakeEmbedder()
    result = run(write_merged(tmp_path / "second.jsonl", changed), es, embedding_model=embedder)
    assert result["examples"]["update/fields_only"] == ["Doc_Reembed"]
    assert result["examples"]["update/reembed"] == ["Doc_Unchanged"]          # no hash: any differing field counts
    assert result["changed"]["genres"] == 2 and result["flips"] == 0
    stored = es.docs[INDEX]["Doc_Reembed"]
    assert stored["genres"] == ["Rabbinic Literature", "Talmud Bavli", "Halakhah", "Responsa"]
    assert "Responsa" in stored["full_text_content"]
    for unchanged_field in ("embedding_vector", "embedded_text_sha256", "semantic_eligible"):
        assert stored[unchanged_field] == hashed_before[unchanged_field]
    unchanged_text = embedding_text(document(changed[5]), "semantic_v3").text
    assert embedder.documents == [CANARY, unchanged_text] and "Responsa" not in unchanged_text
    assert es.docs[INDEX]["Doc_Unchanged"]["genres"][-2:] == ["Halakhah", "Responsa"]


@pytest.mark.parametrize("text_version, decision", [("semantic_v3", "update/reembed"),
                                                    ("semantic_v4", "update/fields_only")])
def test_keywords_lift_a_short_label_under_semantic_v3_but_not_under_semantic_v4(tmp_path, es, text_version, decision):
    """Two keywords take a short label over the floor under the frozen v3 rule (the whole text counts, the Subjects
    line included); semantic_v4 measures the text without its Subjects line, so the record stays without a vector."""
    record = merged_record("Doc_ShortNoFlip", 3, SHORT, subjects=["Poetry", "Liturgy"])
    es.indices.mappings[INDEX]["_meta"]["text_representation_version"] = text_version
    result = run(write_merged(tmp_path / "keywords.jsonl", [record]), es, dry_run=True, text_version=text_version)
    assert result["examples"][decision] == ["Doc_ShortNoFlip"] and result["changed"]["subjects"] == 1
    assert result["flips"] == (1 if decision == "update/reembed" else 0)


def test_a_complete_mapping_is_left_alone_and_a_bare_one_gets_alt_titles_too(merged, es):
    es.indices.mappings[INDEX]["properties"].pop("alt_titles")
    assert run(merged, es, dry_run=True)["mapping_missing"] == ["alt_titles"] + NEW_FIELDS
    run(merged, es, embedding_model=FakeEmbedder())
    assert es.indices.put[0]["alt_titles"] == ALT_TITLES_MAPPING["alt_titles"]
    run(merged, es, embedding_model=FakeEmbedder())
    assert len(es.indices.put) == 1


@pytest.mark.parametrize("meta", [
    index_meta(FakeEmbedder(revision="0" * 40)),
    index_meta(FakeEmbedder(), text_version="v8"),
    {"embedding_model": "Qwen/Qwen3-Embedding-0.6B", "embedding_model_revision": "97b0c614"},
])
def test_writer_guard_refuses_a_mismatching_index(merged, es, meta):
    es.indices.mappings[INDEX]["_meta"] = meta
    embedder = FakeEmbedder()
    with pytest.raises(IndexMetaMismatchError):
        run(merged, es, embedding_model=embedder)
    with pytest.raises(IndexMetaMismatchError):
        run(merged, es, dry_run=True)
    assert embedder.documents == [] and es.bulk_calls == [] and es.indices.put == [] and es.mget_calls == []


def test_missing_index_and_live_names_are_refused(merged, es):
    with pytest.raises(ValueError, match="does not exist"):
        uce.update_catalogue_enrichment(merged, index_name="genizah_merged_v99", es_client=es, dry_run=True, text_version="semantic_v3")
    with pytest.raises(ValueError, match="live"):
        uce.update_catalogue_enrichment(merged, index_name="genizah_merged_v8", es_client=es, dry_run=True, text_version="semantic_v3")
    assert es.mget_calls == []


def test_an_embedder_that_misses_the_canary_is_refused(merged, es):
    embedder = FakeEmbedder(salt="another vector space")
    with pytest.raises(IndexMetaMismatchError, match="canary"):
        run(merged, es, embedding_model=embedder)
    assert embedder.documents == [CANARY] and es.bulk_calls == [] and es.indices.put == [] and es.mget_calls == []


def test_the_default_embedder_must_be_the_one_the_guard_checked(merged, es, monkeypatch):
    monkeypatch.setattr(qte, "QwenTextEmbedding", lambda: FakeEmbedder(revision="f" * 40))
    with pytest.raises(IndexMetaMismatchError, match="not the one the guard checked"):
        run(merged, es)
    assert es.bulk_calls == [] and es.indices.put == []


def test_a_wrong_vector_size_is_refused_before_writing(merged, es):
    es.indices.mappings[INDEX]["_meta"]["dims"] = 1024
    with pytest.raises(ValueError, match="dims"):
        run(merged, es, embedding_model=FakeEmbedder())
    assert es.bulk_calls == []


def test_skip_limit_chunking_and_progress_log(merged, es, monkeypatch, caplog):
    monkeypatch.setattr(uce, "MGET_CHUNK", 2)
    monkeypatch.setattr(uce, "PROGRESS_EVERY", 2)
    with caplog.at_level(logging.INFO, logger=uce.__name__):
        result = run(merged, es, dry_run=True, skip=1, limit=5)
    assert result["scanned"] == 5 and (result["skip"], result["limit"]) == (1, 5)
    assert [c["ids"] for c in es.mget_calls] == [["Doc_Flip", "Doc_ShortNoFlip"], ["Doc_FieldsOnly", "Doc_Refused"],
                                                 ["Doc_Unchanged"]]
    assert "scanned 2 merged records (safe to resume with --skip 3)" in caplog.text
    assert "scanned 4 merged records (safe to resume with --skip 5)" in caplog.text


@pytest.mark.parametrize("canary_id, canary_text, limit, expected", [
    ("Doc_Unchanged", None, None, "unchanged"),
    ("Doc_Reembed", "Description: an older text of this record", None, "changed"),
    ("Doc_Description", "whatever", 2, "not scanned"),
])
def test_document_canary_status(merged, es, records, canary_id, canary_text, limit, expected):
    text = canary_text or embedding_text(document(records[canary_id]), "semantic_v3").text
    es.indices.mappings[INDEX]["_meta"]["document_canary"] = {"doc_id": canary_id, "text": text}
    result = run(merged, es, dry_run=True, limit=limit)
    assert result["document_canary"] == {"doc_id": canary_id, "text": expected}


def test_cli_defaults_follow_the_indexer_constants():
    args = uce.build_parser().parse_args([])
    assert args.index == uce.DEFAULT_INDEX_NAME == "genizah_merged_v10" and args.merged == uce.DEFAULT_MERGED
    assert args.text_version == "semantic_v4" and not args.dry_run and not args.allow_live
    assert (args.limit, args.skip, args.batch_size, args.report) == (None, 0, uce.DEFAULT_BATCH_SIZE, None)
    assert uce.build_parser().parse_args(["--i-know-this-is-live", "--dry-run"]).allow_live


def test_main_runs_a_dry_run_and_writes_the_report(tmp_path, merged, es, monkeypatch, capsys):
    monkeypatch.setattr(uce, "default_es_client", lambda: es)
    monkeypatch.setattr(qte, "QwenTextEmbedding", no_model)
    report = tmp_path / "out.json"
    uce.main(["--merged", str(merged), "--index", INDEX, "--text-version", "semantic_v3", "--dry-run", "--report", str(report)])
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["dry_run"] and data["scanned"] == 9 and data["flips"] == 1 and es.bulk_calls == []
    assert "dry run: nothing embedded or written" in capsys.readouterr().out


def test_refresh_document_canary_rewrites_only_that_meta_block_after_the_writes(merged, es, records):
    stale = {"doc_id": "Doc_Unchanged", "text": "Description: a stale canary text", "vector": [0.0] * DIMS}
    es.indices.mappings[INDEX]["_meta"]["document_canary"] = stale
    before = copy.deepcopy(es.indices.mappings[INDEX]["_meta"])
    result = run(merged, es, embedding_model=FakeEmbedder(), refresh_canary=True)
    text = embedding_text(document(records["Doc_Unchanged"]), "semantic_v3").text
    meta = es.indices.mappings[INDEX]["_meta"]
    canary = meta["document_canary"]
    assert canary["doc_id"] == "Doc_Unchanged" and canary["text"] == text
    assert canary["text_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert np.allclose(canary["vector"], fake_vector(text), atol=1e-6) and len(canary["vector_sha256_6dp"]) == 64
    assert {k: v for k, v in meta.items() if k != "document_canary"} == {
        k: v for k, v in before.items() if k != "document_canary"}
    assert len(es.indices.put_meta) == 1 and es.indices.put       # the properties put happened too
    assert result["document_canary"] == {"doc_id": "Doc_Unchanged", "text": "changed", "refresh": "refreshed",
                                         "refresh_text_changed": True}
    assert result["updated"]["update/reembed"] == 3


def test_refresh_document_canary_is_only_noted_in_a_dry_run(merged, es, monkeypatch, capsys):
    monkeypatch.setattr(qte, "QwenTextEmbedding", no_model)
    es.indices.mappings[INDEX]["_meta"]["document_canary"] = {"doc_id": "Doc_Unchanged", "text": "x",
                                                              "vector": [0.0] * DIMS}
    result = run(merged, es, dry_run=True, refresh_canary=True)
    assert result["document_canary"]["refresh"] == "skipped (dry run)" and es.indices.put_meta == []
    assert "document canary Doc_Unchanged: text changed, skipped (dry run)" in capsys.readouterr().out


@pytest.mark.parametrize("canary", [None, {"doc_id": "Doc_Refused", "text": "x"}, {"doc_id": "Doc_Nowhere", "text": "x"}])
def test_refresh_document_canary_refuses_missing_or_ineligible_canaries(merged, es, canary):
    if canary is not None:
        es.indices.mappings[INDEX]["_meta"]["document_canary"] = canary
    with pytest.raises(ValueError):
        uce.refresh_document_canary(es, INDEX, merged, FakeEmbedder(), "semantic_v3")
    assert es.indices.put_meta == []


def test_find_merged_document_streams_the_file(merged, records):
    doc = uce.find_merged_document(merged, "Doc_Description")
    assert doc is not None and doc.doc_id == "Doc_Description" and doc.description == DEED
    assert uce.find_merged_document(merged, "Doc_Nowhere") is None


def test_cli_has_the_canary_refresh_flag():
    parser = uce.build_parser()
    assert parser.parse_args([]).refresh_canary is False
    assert parser.parse_args(["--refresh-document-canary"]).refresh_canary is True


def test_the_updater_renders_frames_exactly_as_the_index_was_built(merged, es, monkeypatch):
    """A frames-off index (``_meta.text_function.scholarly_frames: false``) is updated with frames-off texts."""
    monkeypatch.setattr(qte, "QwenTextEmbedding", no_model)
    es.indices.mappings[INDEX]["_meta"]["text_function"] = {"version": "semantic_v3", "scholarly_frames": False}
    result = run(merged, es, dry_run=True)
    # Doc_Flip is a boilerplate label plus a frame: eligible only when the frame is rendered
    assert result["flips"] == 0 and "Doc_Flip" in result["examples"]["update/fields_only"]
    es.indices.mappings[INDEX]["_meta"]["text_function"]["scholarly_frames"] = True
    again = run(merged, es, dry_run=True)
    assert again["flips"] == 1 and again["examples"]["flips"] == ["Doc_Flip"]
