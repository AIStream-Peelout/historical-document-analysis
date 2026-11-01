# File name: test_update_image_corrections.py
# Date: 10/31/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Image-corrections updater: target partial documents, the dry run's read-only contract, refusal rules, apply with
read-back, rollback, report contents, CLI.

Synthetic merged records only. No network: Elasticsearch and the bulk helper are faked. No ``.env`` is read.
"""
import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest import mock

import pytest

from src.datasets.indexing import update_image_corrections as uic

INDEX = "synthetic_served_index_q7"
OTHER_INDEX = "synthetic_other_index_q7"
BASE = "https://storage.example.test/bucket-q7/images"


# ----------------------------------------------------------------------------------------------- fixtures


def merged_record(cid: str, legacy: Sequence[str] = (), preferred: Optional[str] = None,
                  ktiv_urls: Sequence[str] = (), **image_extra: Any) -> Dict[str, Any]:
    """A synthetic merged record.

    :param cid: Canonical id.
    :param legacy: ``images.fjp`` file names.
    :param preferred: ``images.preferred_source``.
    :param ktiv_urls: ``images.ktiv.image_urls``.
    :param image_extra: Further ``images`` keys (correction markers).
    :return: Record.
    """
    ktiv = ({"image_urls": list(ktiv_urls), "iiif_manifest_url": f"https://iiif.example.test/{cid}/manifest"}
            if ktiv_urls else None)
    return {"canonical_id": cid, "shelfmark_display": cid.replace("_", " "), "institution": "Synthetic Library",
            "collection": "Synthetic", "subcollection": None, "sources_present": ["fjp"],
            "description": f"Synthetic fragment {cid}", "date": None, "pgpids": [],
            "images": {"fjp": list(legacy), "ktiv": ktiv, "bodleian": None, "preferred_source": preferred,
                       **image_extra},
            "sources": {}}


def served_source(record: Dict[str, Any]) -> Dict[str, Any]:
    """The ``_source`` an index built from ``record`` holds (image fields, metadata, plus an unrelated field).

    :param record: Merged record.
    :return: Source document.
    """
    from src.datasets.document_models.genizah_document import GenizahDocument

    doc = GenizahDocument.from_merged_format(copy.deepcopy(record)).to_elasticsearch_document()
    return json.loads(json.dumps({k: v for k, v in doc.items() if k != "indexed_at"}, default=str))


def mapping_properties() -> Dict[str, Any]:
    """An index mapping with every image field mapped.

    :return: ``properties``.
    """
    props: Dict[str, Any] = {name: {"type": "keyword"} for name in uic.TOP_LEVEL_IMAGE_FIELDS}
    props["canonical_id"] = {"type": "keyword"}
    props["description"] = {"type": "text"}
    props[uic.METADATA_FIELD] = {"properties": {key: {"type": "keyword"} for key in uic.METADATA_IMAGE_KEYS}}
    return props


def select(source: Dict[str, Any], includes: Optional[List[str]]) -> Dict[str, Any]:
    """``_source`` filtering by dotted paths, as Elasticsearch does.

    :param source: Document.
    :param includes: Paths (None = everything).
    :return: Filtered copy.
    """
    if includes is None:
        return copy.deepcopy(source)
    out: Dict[str, Any] = {}
    for path in includes:
        head, _, tail = path.partition(".")
        if head not in source:
            continue
        if tail:
            if isinstance(source[head], dict) and tail in source[head]:
                out.setdefault(head, {})[tail] = copy.deepcopy(source[head][tail])
        else:
            out[head] = copy.deepcopy(source[head])
    return out


def deep_merge(target: Dict[str, Any], partial: Dict[str, Any]) -> None:
    """Merge a partial document into a source the way an Elasticsearch ``doc`` update does.

    :param target: Stored source (changed in place).
    :param partial: Partial document.
    """
    for key, value in partial.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


class FakeIndices:
    """``es.indices`` with one index and a record of every call."""

    def __init__(self, names: Sequence[str]) -> None:
        """:param names: Existing indices."""
        self.names = set(names)
        self.calls: List[str] = []
        self.properties = mapping_properties()

    def exists(self, index: str) -> bool:
        """:param index: Index.
        :return: Whether it exists.
        """
        self.calls.append("exists")
        return index in self.names

    def get_mapping(self, index: str) -> Dict[str, Any]:
        """:param index: Index.
        :return: Mapping response.
        """
        self.calls.append("get_mapping")
        return {index: {"mappings": {"properties": copy.deepcopy(self.properties)}}}


class FakeES:
    """In-memory cluster: one store per index, ``mget`` with dotted ``source_includes``."""

    def __init__(self, docs: Dict[str, Dict[str, Any]], index: str = INDEX) -> None:
        """:param docs: Sources by id for ``index``.
        :param index: Index name.
        """
        self.store: Dict[str, Dict[str, Dict[str, Any]]] = {index: copy.deepcopy(docs)}
        self.indices = FakeIndices([index])
        self.mget_calls: List[List[str]] = []
        self.bulk_calls: List[List[Dict[str, Any]]] = []

    def mget(self, index: str, ids: Sequence[str], source_includes: Optional[List[str]] = None) -> Dict[str, Any]:
        """:param index: Index.
        :param ids: Ids.
        :param source_includes: Paths.
        :return: ``{"docs": [...]}``.
        """
        self.mget_calls.append(list(ids))
        docs = self.store.get(index, {})
        return {"docs": [{"_index": index, "_id": i, "found": True, "_source": select(docs[i], source_includes)}
                         if i in docs else {"_index": index, "_id": i, "found": False} for i in ids]}


@pytest.fixture
def fake_bulk(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Replace the bulk helper by one that deep-merges partial updates into the fake store.

    :param monkeypatch: Pytest monkeypatch.
    :return: Calls recorded as ``(refresh, actions)``.
    """
    calls: List[Tuple[str, List[Dict[str, Any]]]] = []

    def run_bulk(client: FakeES, actions: List[Dict[str, Any]], refresh: str, raise_on_error: bool) -> tuple:
        """:param client: Fake cluster.
        :param actions: Actions.
        :param refresh: Refresh policy.
        :param raise_on_error: Must be False (errors are recorded, not raised).
        :return: ``(successes, errors)``.
        """
        assert raise_on_error is False
        calls.append((refresh, copy.deepcopy(actions)))
        for action in actions:
            assert action["_op_type"] == "update" and set(action) == {"_op_type", "_index", "_id", "doc"}
            deep_merge(client.store[action["_index"]][action["_id"]], action["doc"])
        return len(actions), []

    monkeypatch.setattr(uic, "bulk", run_bulk)
    return calls


def misfiled_scene(tmp_path: Path) -> Dict[str, Any]:
    """A misfiled photograph: A's ``A_1r.jpg`` shows B; the corrected merge moves it to B (which had no image).

    C is unrelated; D carries a removal marker without a correction row; E is named by a correction but has no merged
    record; F is in the merged file but not in the index.

    :param tmp_path: Pytest temp dir.
    :return: Paths, the fake cluster and the served sources.
    """
    original = {"A": merged_record("A", ["A_1r.jpg", "A_1v.jpg"], "fjp"),
                "B": merged_record("B", [], None),
                "C": merged_record("C", ["C_1r.jpg"], "fjp"),
                "D": merged_record("D", ["D_1r.jpg", "D_2r.jpg"], "fjp"),
                "F": merged_record("F", ["F_1r.jpg"], "fjp")}
    corrected = {"A": merged_record("A", ["A_1v.jpg"], "fjp", fjp_corrections=[{"image": "A_1r.jpg", "to": "B"}]),
                 "B": merged_record("B", ["A_1r.jpg"], "fjp", fjp_corrections=[{"image": "A_1r.jpg", "from": "A"}]),
                 "C": original["C"],
                 "D": merged_record("D", ["D_1r.jpg"], "fjp", fjp_removed=["D_2r.jpg"]),
                 "F": merged_record("F", [], None, fjp_corrections=[{"image": "F_1r.jpg", "to": "E"}])}
    merged = tmp_path / "merged.jsonl"
    merged.write_text("".join(json.dumps(rec) + "\n" for rec in corrected.values()), encoding="utf-8")
    corrections = tmp_path / "image_corrections.jsonl"
    corrections.write_text(json.dumps({"from_canonical_id": "A", "to_canonical_id": "B", "image": "A_1r.jpg"}) + "\n"
                           + json.dumps({"from_canonical_id": "F", "to_canonical_id": "E", "image": "F_1r.jpg"})
                           + "\n", encoding="utf-8")
    served = {cid: served_source(original[cid]) for cid in ("A", "B", "C", "D")}
    old = time.time() - 600
    os.utime(merged, (old, old))
    return {"merged": merged, "corrections": corrections, "es": FakeES(served), "served": served,
            "plan": tmp_path / "out" / "plan.jsonl", "report": tmp_path / "out" / "report.md",
            "result": tmp_path / "out" / "result.jsonl", "rollback_result": tmp_path / "out" / "rollback.jsonl"}


def dry(scene: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    """Run the dry run of a scene.

    :param scene: :func:`misfiled_scene`.
    :param kwargs: Further :func:`uic.dry_run` arguments.
    :return: Its summary.
    """
    kwargs.setdefault("es_client", scene["es"])
    return uic.dry_run(scene["merged"], scene["corrections"], index_name=INDEX, plan_path=scene["plan"],
                       report_path=scene["report"], **kwargs)


def apply(scene: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    """Apply a scene's plan.

    :param scene: :func:`misfiled_scene`.
    :param kwargs: Further :func:`uic.apply_plan` arguments.
    :return: Its summary.
    """
    kwargs.setdefault("index_name", INDEX)
    return uic.apply_plan(scene["plan"], merged_path=scene["merged"], corrections_path=scene["corrections"],
                          result_path=scene["result"], es_client=scene["es"], **kwargs)


def rows_by_id(plan_path: Path) -> Dict[str, Dict[str, Any]]:
    """Plan rows by canonical id.

    :param plan_path: Plan.
    :return: Rows.
    """
    return {row["canonical_id"]: row for row in uic.read_plan(plan_path)[1]}


def written_ids(calls: List[Tuple[str, List[Dict[str, Any]]]]) -> List[str]:
    """Document ids of every bulk action.

    :param calls: :func:`fake_bulk` record.
    :return: Ids.
    """
    return [action["_id"] for _, actions in calls for action in actions]


# ----------------------------------------------------------------------------------------------- unit


def test_correction_ids_and_reader(tmp_path: Path) -> None:
    """Sources and targets both count; a removal has no target; a row without a source is refused."""
    assert uic.correction_ids([{"from_canonical_id": "A", "to_canonical_id": "B"},
                               {"from_canonical_id": "C", "to_canonical_id": None},
                               {"from_canonical_id": "D"}]) == {"A", "B", "C", "D"}
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"from_canonical_id": "A"}\n\n{"to_canonical_id": "B"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 3"):
        uic.read_corrections(bad)


def test_line_canonical_id_prefix_and_fallback() -> None:
    """The leading key is read from the prefix; another key order falls back to parsing."""
    assert uic.line_canonical_id('{"canonical_id": "X_\\"1", "images": {}}') == 'X_"1'
    assert uic.line_canonical_id('{"images": {}, "canonical_id": "Y"}') == "Y"


def test_image_file_name_never_shows_host_or_bucket() -> None:
    """Bucket URLs, bare names, IIIF image and manifest URLs give file names only."""
    assert uic.image_file_name(f"{BASE}/A_1r.jpg") == "A_1r.jpg"
    assert uic.image_file_name("A_1r.jpg") == "A_1r.jpg"
    assert uic.image_file_name("https://iiif.example.test/iiif/image/abc-123/full/full/0/default.jpg") == "abc-123"
    assert uic.image_file_name("https://iiif.example.test/DOCID/PNX_1-1/manifest") == "PNX_1-1/manifest"
    assert uic.image_file_name(None) == "(none)"


def test_partial_doc_nests_metadata_keys() -> None:
    """Dotted metadata paths become one nested object; top-level fields stay flat."""
    assert uic.partial_doc({"image_urls": ["u"], "tei_metadata.image_preferred_source": "x",
                            "tei_metadata.ktiv_images": []}) == {
        "image_urls": ["u"], "tei_metadata": {"image_preferred_source": "x", "ktiv_images": []}}


def test_scan_merged_finds_named_and_marked_records(tmp_path: Path) -> None:
    """Named ids are collected; a non-empty marker adds a record; an empty marker does not."""
    path = tmp_path / "m.jsonl"
    records = [merged_record("A", ["a.jpg"]), merged_record("D", [], fjp_removed=["d.jpg"]),
               merged_record("G", ["g.jpg"], fjp_corrections=[]), merged_record("H", ["h.jpg"])]
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    found, marked = uic.scan_merged(path, {"A", "Z"})
    assert set(found) == {"A", "D"} and marked == {"D"}


# ----------------------------------------------------------------------------------------------- dry run


def test_target_partial_docs_for_a_moved_image(tmp_path: Path) -> None:
    """The plan's changed fields per document are exactly the image fields the correction changes."""
    scene = misfiled_scene(tmp_path)
    summary = dry(scene)
    rows = rows_by_id(scene["plan"])
    assert {cid: row["status"] for cid, row in rows.items()} == {
        "A": "changed", "B": "changed", "D": "changed", "E": "not_in_merged", "F": "missing_in_index"}
    assert "C" not in rows                                      # neither named nor marked
    a, b, d = rows["A"], rows["B"], rows["D"]
    moved_url, kept_url = a["before"]["image_urls"]
    assert uic.partial_doc({f: a["after"][f] for f in a["changed_fields"]}) == {
        "image_urls": [kept_url], "actual_image_url": kept_url}
    assert uic.partial_doc({f: b["after"][f] for f in b["changed_fields"]}) == {
        "image_urls": [moved_url], "primary_image_index": 0, "has_images": True, "actual_image_url": moved_url,
        "image_preferred_source": "fjp", "tei_metadata": {"image_preferred_source": "fjp"}}
    assert d["changed_fields"] == ["image_urls"] and d["images_removed"] == ["D_2r.jpg"]
    assert (a["images_added"], a["images_removed"]) == ([], ["A_1r.jpg"])
    assert (b["images_added"], b["images_removed"]) == (["A_1r.jpg"], [])
    assert summary["changed"] == 3 and summary["images_added"] == 1 and summary["images_removed"] == 2
    assert summary["would_refuse"] == []


def test_dry_run_performs_no_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With a strict mock client the dry run only calls read methods; no update, bulk or index call happens."""
    scene = misfiled_scene(tmp_path)
    fake = scene["es"]
    client = mock.MagicMock(name="es")
    client.mget.side_effect = fake.mget
    client.indices.exists.side_effect = fake.indices.exists
    client.indices.get_mapping.side_effect = fake.indices.get_mapping
    bulk_helper = mock.MagicMock(name="bulk")
    monkeypatch.setattr(uic, "bulk", bulk_helper)
    dry(scene, es_client=client)
    bulk_helper.assert_not_called()
    for name in ("update", "bulk", "index", "create", "delete", "update_by_query", "reindex"):
        getattr(client, name).assert_not_called()
    for name in ("put_mapping", "create", "delete", "refresh", "put_settings"):
        getattr(client.indices, name).assert_not_called()
    called = {c[0].split(".")[0] if c[0] else "" for c in client.mock_calls}
    assert called <= {"mget", "indices"}
    assert scene["plan"].exists() and scene["report"].exists()


def test_report_names_file_names_only_and_no_index(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """The report shows file names, never full URLs, the bucket or the index name; nor do the plan or stdout."""
    scene = misfiled_scene(tmp_path)
    dry(scene, note="synthetic corrections")
    report = scene["report"].read_text(encoding="utf-8")
    assert "`A_1r.jpg`" in report and "synthetic corrections" in report
    assert "://" not in report and "bucket-q7" not in report and "storage.example" not in report
    assert "| would change | 3 |" in report and "| missing in the index (never created) | 1 |" in report
    for text in (report, scene["plan"].read_text(encoding="utf-8"), capsys.readouterr().out):
        assert INDEX not in text


def test_dry_run_reports_over_limit_and_unmapped(tmp_path: Path) -> None:
    """The dry run notes, without refusing, what would block an apply."""
    scene = misfiled_scene(tmp_path)
    del scene["es"].indices.properties[uic.METADATA_FIELD]
    summary = dry(scene, max_docs=2)
    assert summary["would_refuse"] == ["3 > --max-docs 2", "unmapped field tei_metadata.image_preferred_source"]
    assert "REFUSED" in scene["report"].read_text(encoding="utf-8")


# ----------------------------------------------------------------------------------------------- refusals


def test_refuses_plan_older_than_merged_file(tmp_path: Path, fake_bulk: list) -> None:
    """A merged file touched after the plan blocks the apply."""
    scene = misfiled_scene(tmp_path)
    dry(scene)
    future = time.time() + 60
    os.utime(scene["merged"], (future, future))
    with pytest.raises(uic.ApplyRefused, match="older than the merged file"):
        apply(scene)
    assert fake_bulk == []


def test_refuses_changed_corrections_or_other_files(tmp_path: Path, fake_bulk: list) -> None:
    """Corrections edited after the plan, or another corrections / merged path, block the apply."""
    scene = misfiled_scene(tmp_path)
    dry(scene)
    other = tmp_path / "other_corrections.jsonl"
    other.write_text(scene["corrections"].read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(uic.ApplyRefused, match="re-run the dry run"):
        uic.apply_plan(scene["plan"], merged_path=scene["merged"], corrections_path=other, index_name=INDEX,
                       result_path=scene["result"], es_client=scene["es"])
    with open(scene["corrections"], "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"from_canonical_id": "C"}) + "\n")
    with pytest.raises(uic.ApplyRefused, match="corrections file changed"):
        apply(scene)
    assert fake_bulk == []


def test_refuses_more_than_max_docs(tmp_path: Path, fake_bulk: list) -> None:
    """More changed documents than --max-docs blocks the apply."""
    scene = misfiled_scene(tmp_path)
    dry(scene)
    with pytest.raises(uic.ApplyRefused, match="more than --max-docs 2"):
        apply(scene, max_docs=2)
    assert fake_bulk == []


def test_refuses_another_index_drift_and_unmapped(tmp_path: Path, fake_bulk: list) -> None:
    """Another target index, a document changed since the plan, or an unmapped field blocks the apply."""
    scene = misfiled_scene(tmp_path)
    dry(scene)
    with pytest.raises(uic.ApplyRefused, match="another index"):
        apply(scene, index_name=OTHER_INDEX)
    scene["es"].store[INDEX]["B"]["image_urls"] = [f"{BASE}/someone_else.jpg"]
    with pytest.raises(uic.ApplyRefused, match="1 documents do not hold the plan's before values"):
        apply(scene)
    scene["es"].store[INDEX]["B"]["image_urls"] = []
    del scene["es"].indices.properties["has_images"]
    with pytest.raises(uic.ApplyRefused, match="unmapped fields: has_images"):
        apply(scene)
    assert fake_bulk == []


# ----------------------------------------------------------------------------------------------- apply / rollback


def test_apply_writes_partial_updates_verifies_and_is_idempotent(tmp_path: Path, fake_bulk: list) -> None:
    """Only changed image fields are written, with refresh; every document is read back; a re-run writes nothing."""
    scene = misfiled_scene(tmp_path)
    dry(scene)
    rows = rows_by_id(scene["plan"])
    summary = apply(scene)
    assert summary["written"] == 3 and summary["verified"] == 3 and summary["unverified"] == summary["failed"] == 0
    assert [refresh for refresh, _ in fake_bulk] == ["wait_for"] and sorted(written_ids(fake_bulk)) == ["A", "B", "D"]
    for _, actions in fake_bulk:
        for action in actions:
            row = rows[action["_id"]]
            assert action["_index"] == INDEX
            assert action["doc"] == uic.partial_doc({f: row["after"][f] for f in row["changed_fields"]})
    store = scene["es"].store[INDEX]
    for cid in ("A", "B", "D"):
        assert uic.image_fields(store[cid]) == rows[cid]["after"]
        assert store[cid]["description"] == scene["served"][cid]["description"]          # untouched
        assert store[cid]["tei_metadata"]["canonical_id"] == cid                          # merged, not replaced
    assert store["C"] == scene["served"]["C"]
    result = [json.loads(line) for line in scene["result"].read_text(encoding="utf-8").splitlines()]
    assert result[0]["kind"] == "header" and result[-1]["kind"] == "summary"
    assert {r["canonical_id"]: r["verified"] for r in result if r["kind"] == "document"} == {
        "A": True, "B": True, "D": True}
    archived = list(scene["plan"].parent.glob("plan.applied-*.jsonl"))
    assert len(archived) == 1 and summary["archived_plan"].endswith(archived[0].name)
    again = apply(scene)
    assert again["written"] == 0 and again["already_done"] == 3 and len(fake_bulk) == 1


def test_rollback_is_gated_on_apply_and_restores_before_values(tmp_path: Path, fake_bulk: list) -> None:
    """Without --apply a rollback only checks; with it the before values come back and verify."""
    scene = misfiled_scene(tmp_path)
    dry(scene)
    before = {cid: uic.image_fields(scene["es"].store[INDEX][cid]) for cid in ("A", "B", "D")}
    apply(scene)
    archived = next(scene["plan"].parent.glob("plan.applied-*.jsonl"))
    check = uic.rollback_plan(archived, execute=False, index_name=INDEX, result_path=scene["rollback_result"],
                              es_client=scene["es"])
    assert check["to_write"] == 3 and check["dry_run"] is True and len(fake_bulk) == 1
    restored = uic.rollback_plan(archived, execute=True, index_name=INDEX, result_path=scene["rollback_result"],
                                 es_client=scene["es"])
    assert restored["written"] == 3 and restored["verified"] == 3 and len(fake_bulk) == 2
    for cid, fields in before.items():
        assert uic.image_fields(scene["es"].store[INDEX][cid]) == fields
    assert uic.rollback_plan(archived, execute=True, index_name=INDEX, result_path=scene["rollback_result"],
                             es_client=scene["es"])["already_done"] == 3


def test_rollback_refuses_drifted_documents(tmp_path: Path, fake_bulk: list) -> None:
    """A document changed after the apply blocks the whole rollback."""
    scene = misfiled_scene(tmp_path)
    dry(scene)
    apply(scene)
    archived = next(scene["plan"].parent.glob("plan.applied-*.jsonl"))
    scene["es"].store[INDEX]["A"]["image_urls"] = [f"{BASE}/later.jpg"]
    with pytest.raises(uic.ApplyRefused, match="plan's after values"):
        uic.rollback_plan(archived, execute=True, index_name=INDEX, result_path=scene["rollback_result"],
                          es_client=scene["es"])
    assert len(fake_bulk) == 1


# ----------------------------------------------------------------------------------------------- CLI


def test_cli_defaults_to_dry_run_and_returns_refusal_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                            fake_bulk: list) -> None:
    """No flag = dry run (no write); --apply with a stale plan exits 2 without writing."""
    scene = misfiled_scene(tmp_path)
    monkeypatch.setattr(uic, "default_es_client", lambda: scene["es"])
    common = ["--merged", str(scene["merged"]), "--corrections", str(scene["corrections"]), "--index", INDEX,
              "--plan", str(scene["plan"]), "--report", str(scene["report"]), "--result", str(scene["result"])]
    assert uic.main(common) == 0 and scene["plan"].exists() and fake_bulk == []
    future = time.time() + 60
    os.utime(scene["merged"], (future, future))
    assert uic.main([*common, "--apply"]) == 2 and fake_bulk == []
