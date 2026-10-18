# File name: test_build_vectorless_jobs.py
# Date: 10/8/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for the vector-less consensus job pool builder (toy records only, no data files)."""
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional

import pytest

BUILDER = Path(__file__).resolve().parents[1] / "src/datasets/consensus/build_vectorless_jobs.py"
BASE = "https://storage.googleapis.com/cairo-genizah-es-json"


def _load() -> ModuleType:
    """Import the builder from its path (``logs/`` is not a package).

    :return: The module.
    :rtype: ModuleType
    """
    spec = importlib.util.spec_from_file_location("build_vectorless_jobs", BUILDER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bv = _load()
LEGACY = bv.LEGACY_IMAGE_KEY


def _images(legacy: Optional[List[str]] = None, ktiv_urls: Optional[List[str]] = None,
            bodleian_urls: Optional[List[str]] = None, preferred: Optional[str] = None,
            ktiv_extra: Optional[Dict[str, Any]] = None, bodleian_extra: Optional[Dict[str, Any]] = None
            ) -> Dict[str, Any]:
    """A toy ``images`` block in the merged export's shape.

    :param legacy: Legacy-store entries (bare names or URLs).
    :type legacy: Optional[List[str]]
    :param ktiv_urls: KTIV ``image_urls``.
    :type ktiv_urls: Optional[List[str]]
    :param bodleian_urls: Bodleian ``image_urls``.
    :type bodleian_urls: Optional[List[str]]
    :param preferred: ``preferred_source``.
    :type preferred: Optional[str]
    :param ktiv_extra: Extra KTIV keys (``zip_files``, ``iiif_manifest_url``...).
    :type ktiv_extra: Optional[Dict[str, Any]]
    :param bodleian_extra: Extra Bodleian keys.
    :type bodleian_extra: Optional[Dict[str, Any]]
    :return: Images block.
    :rtype: Dict[str, Any]
    """
    ktiv = {"image_urls": ktiv_urls or [], **(ktiv_extra or {})} if (ktiv_urls or ktiv_extra) else {}
    bodleian = {"image_urls": bodleian_urls or [], **(bodleian_extra or {})} if (bodleian_urls or bodleian_extra) \
        else None
    return {LEGACY: legacy or [], "ktiv": ktiv, "bodleian": bodleian, "preferred_source": preferred}


def _cand(doc_id: str, tranche: str, urls: List[str], reason: str = "empty") -> Any:
    """A toy candidate whose images all come from the legacy store.

    :param doc_id: Document id.
    :type doc_id: str
    :param tranche: Tranche label.
    :type tranche: str
    :param urls: Image URLs.
    :type urls: List[str]
    :param reason: Vector-less reason.
    :type reason: str
    :return: Candidate.
    :rtype: Any
    """
    return bv.Candidate(doc_id=doc_id, tranche=tranche, reason=reason, document_type=None, institution="Inst",
                        images=[("legacy", u) for u in urls], availability="legacy_store", ktiv_status="none",
                        bodleian_status="none")


def test_vectorless_reason_maps_and_rejects_unknown() -> None:
    """Eligibility reasons map to the job labels; eligible records get none."""
    assert bv.vectorless_reason("empty") == "empty"
    assert bv.vectorless_reason("short") == "short"
    assert bv.vectorless_reason("boilerplate_only") == "boilerplate"
    assert bv.vectorless_reason("eligible") is None
    with pytest.raises(ValueError):
        bv.vectorless_reason("mystery")


@pytest.mark.parametrize("record,tranche", [
    ({"sources_present": ["pgp", LEGACY]}, "V1_pgp_documentary"),
    ({"sources_present": ["ktiv", "pgp"]}, "V1_pgp_documentary"),
    ({"sources_present": [], "sources": {"pgp": {"documents": [{}]}}}, "V1_pgp_documentary"),
    ({"sources_present": ["ktiv", LEGACY]}, "V2_ktiv_literary"),
    ({"sources_present": ["bodleian", "ktiv"]}, "V2_ktiv_literary"),
    ({"sources_present": [LEGACY]}, "V3_legacy_scraps"),
    ({"sources_present": ["bodleian", LEGACY], "sources": {"pgp": {}, "ktiv": None}}, "V3_legacy_scraps"),
])
def test_assign_tranche(record: Dict[str, Any], tranche: str) -> None:
    """PGP wins, then KTIV; empty source blocks do not count.

    :param record: Toy merged record.
    :type record: Dict[str, Any]
    :param tranche: Expected tranche.
    :type tranche: str
    """
    assert bv.assign_tranche(record) == tranche


def test_ordered_image_urls_preferred_first_and_legacy_prefixed() -> None:
    """Preferred source leads, the rest follow ktiv > bodleian > legacy; bare legacy names get the bucket prefix."""
    images = _images(legacy=["A_1r.jpg", f"{BASE}/images/B 2.jpg"], ktiv_urls=[f"{BASE}/KTIV/9/0001_FL1.jpg"],
                     bodleian_urls=[f"{BASE}/BODLEIAN/x/1a.jpg"], preferred=LEGACY)
    assert bv.ordered_image_urls(images) == [
        ("legacy", f"{BASE}/images/A_1r.jpg"), ("legacy", f"{BASE}/images/B 2.jpg"),
        ("ktiv", f"{BASE}/KTIV/9/0001_FL1.jpg"), ("bodleian", f"{BASE}/BODLEIAN/x/1a.jpg")]
    images["preferred_source"] = None
    assert [s for s, _ in bv.ordered_image_urls(images)] == ["ktiv", "bodleian", "legacy", "legacy"]
    assert bv.ordered_image_urls({}) == []


@pytest.mark.parametrize("preferred", [None, "ktiv", "bodleian", LEGACY])
def test_ordered_image_urls_match_the_index_builder(preferred: Optional[str]) -> None:
    """The per-source attribution reproduces ``genizah_document._merged_image_urls`` exactly.

    :param preferred: ``preferred_source`` value.
    :type preferred: Optional[str]
    """
    from src.datasets.document_models.genizah_document import _merged_image_urls
    images = _images(legacy=["L_1r.jpg", "L_1v.jpg"], ktiv_urls=[f"{BASE}/KTIV/1/0001_FL2.jpg"],
                     bodleian_urls=[f"{BASE}/BODLEIAN/b/1.jpg"], preferred=preferred)
    assert [u for _, u in bv.ordered_image_urls(images)] == _merged_image_urls(images)


def test_image_status_availability_and_unresolved_reason() -> None:
    """KTIV / Bodleian states, the one-class availability precedence and the unresolved reasons."""
    manifest = {"iiif_manifest_url": "https://iiif.nli.org.il/IIIFv21/DOCID/1-1/manifest"}
    iiif = _images(ktiv_extra=manifest)
    zips = _images(ktiv_extra={"zip_files": ["k.zip"], **manifest})
    gcs = _images(ktiv_urls=[f"{BASE}/KTIV/1/0001_FL1.jpg"])
    bod_empty = _images(bodleian_extra={"populated": False})
    assert [bv.ktiv_image_status(x) for x in (iiif, zips, gcs, {})] == ["iiif_only", "zip_unpopulated", "gcs", "none"]
    assert bv.bodleian_image_status(bod_empty) == "unpopulated"
    assert bv.bodleian_image_status(_images(bodleian_urls=["u"])) == "gcs"
    assert bv.availability_class(_images(legacy=["a.jpg"], ktiv_extra=manifest)) == "legacy_store"
    assert [bv.availability_class(x) for x in (zips, gcs, iiif, bod_empty, {})] == [
        "ktiv_local", "ktiv_local", "ktiv_iiif_only", "bodleian", "none"]
    assert [bv.unresolved_reason(x) for x in (iiif, zips, bod_empty, {})] == [
        "ktiv_iiif_only", "ktiv_zip_not_populated", "bodleian_not_populated", "no_images"]


def test_make_candidate_fields() -> None:
    """A candidate carries the tranche, ordered images, KTIV identifiers and the unresolved reason."""
    record = {"sources_present": ["ktiv"], "images": _images(
        ktiv_extra={"iiif_manifest_url": "https://iiif.nli.org.il/m", "sys_num": "990001"})}
    c = bv.make_candidate(record, "Doc_1", "short", None, "Inst")
    assert (c.tranche, c.images, c.ktiv_status, c.sys_num, c.unresolved) == (
        "V2_ktiv_literary", [], "iiif_only", "990001", "ktiv_iiif_only")
    assert bv.iiif_lines([c])[1] == "Doc_1\tV2_ktiv_literary\tshort\t990001\thttps://iiif.nli.org.il/m\t0"


def test_iter_jsonl_skips_blank_and_torn_lines(tmp_path: Path) -> None:
    """A half-written last line (a paused writer) is skipped.

    :param tmp_path: Pytest temporary directory.
    :type tmp_path: Path
    """
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n\n{"a": 2}\n{"a": 3, "b"', encoding="utf-8")
    assert [r["a"] for r in bv.iter_jsonl(p)] == [1, 2]


def test_read_files_excludes_failure_logs(tmp_path: Path) -> None:
    """Failure logs are not reads.

    :param tmp_path: Pytest temporary directory.
    :type tmp_path: Path
    """
    for name in ("ai_reads_m.jsonl", "ai_reads_m.failures.jsonl", "jobs_x.jsonl", "ai_reads_n_sample10.jsonl"):
        (tmp_path / name).write_text("", encoding="utf-8")
    assert [p.name for p in bv.read_files(tmp_path)] == ["ai_reads_m.jsonl", "ai_reads_n_sample10.jsonl"]


def test_index_reads_keys_by_url() -> None:
    """Reads are keyed by image URL; rows without a read or URL are ignored."""
    rows = [{"doc_id": "D", "image_index": 3, "image_url": "u1", "ai_read": {"vlm_model": "m1", "n_lines": 7}},
            {"doc_id": "D", "image_index": 0, "image_url": "u2", "failure": "download"},
            {"doc_id": "E", "image_index": 0, "image_url": None, "ai_read": {"vlm_model": "m1"}}]
    more = [{"doc_id": "D2", "image_index": 0, "image_url": "u1", "ai_read": {"vlm_model": "m2", "n_lines": 0}}]
    reads = bv.index_reads([("f1.jsonl", rows), ("f2.jsonl", more)])
    assert set(reads) == {"u1"}
    assert reads["u1"] == [bv.ReadRef("m1", "f1.jsonl", "D", 7), bv.ReadRef("m2", "f2.jsonl", "D2", 0)]


def test_doc_read_class() -> None:
    """Document class from its images' classes."""
    assert bv.doc_read_class([]) == "unresolved"
    assert bv.doc_read_class(["already_read", "already_read"]) == "already_read"
    assert bv.doc_read_class(["already_read", "never_read"]) == "partially_read"
    assert bv.doc_read_class(["never_read"]) == "never_read"


def test_job_rows_format_reads_and_pending() -> None:
    """Rows lead with the pipeline's keys; read images carry ``read_by``, pending never-read ones ``pending_in``."""
    c = _cand("Doc_A", "V3_legacy_scraps", [f"{BASE}/images/a.jpg", f"{BASE}/images/b.jpg", f"{BASE}/images/c.jpg"],
              reason="boilerplate")
    reads = {f"{BASE}/images/a.jpg": [bv.ReadRef("m1", "ai_reads_m1.jsonl", "Doc_A", 4)]}
    pending = {f"{BASE}/images/a.jpg": ["jobs_routed_literary"], f"{BASE}/images/b.jpg": ["jobs_routed_literary"]}
    rows = bv.job_rows(c, reads, pending)
    assert list(rows[0])[:7] == ["doc_id", "image_index", "image_url", "document_type", "institution", "queue",
                                 "reread_class"]
    assert [r["image_index"] for r in rows] == [0, 1, 2]
    assert [r["reread_class"] for r in rows] == ["already_read", "never_read", "never_read"]
    assert rows[0]["read_by"] == [{"vlm_model": "m1", "file": "ai_reads_m1.jsonl", "n_lines": 4}]
    assert "pending_in" not in rows[0] and rows[1]["pending_in"] == ["jobs_routed_literary"]
    assert "pending_in" not in rows[2] and "read_by" not in rows[2]
    assert {r["pool"] for r in rows} == {"vectorless_v9"}
    assert {(r["vectorless_reason"], r["queue"], r["doc_n_images"], r["image_source"]) for r in rows} == {
        ("boilerplate", "V3_legacy_scraps", 3, "legacy")}
    assert json.loads(json.dumps(rows[1]))["document_type"] is None
    assert bv.job_rows(_cand("Doc_B", "V1_pgp_documentary", []), reads) == []


def test_co_owners_and_also_in_fields() -> None:
    """``also_in`` lists other records (capped); ``also_in_fragment`` counts only fragment-level ones."""
    owners = {"u1": ["Box", "Frag_1", "Frag_1"], "u2": ["Box"], "u3": ["Box"] + [f"F{i}" for i in range(8)],
              "u4": ["Frag_2", "BigBox"]}
    n_images = {"Box": 30, "BigBox": 400, "Frag_1": 2, "Frag_2": 1, **{f"F{i}": 1 for i in range(8)}}
    assert bv.co_owners("u1", "Box", owners) == ["Frag_1"]
    assert bv.co_owners("u2", "Box", owners) == [] and bv.co_owners("zz", "Box", owners) == []
    rows = bv.job_rows(_cand("Box", "V3_legacy_scraps", ["u1", "u2", "u3"]), {}, None, owners, {"Frag_1"}, n_images)
    assert rows[0]["also_in"] == ["Frag_1"] and rows[0]["also_in_fragment"] == "vector"
    assert "also_in" not in rows[1] and "also_in_fragment" not in rows[1]
    assert len(rows[2]["also_in"]) == bv.MAX_CO_OWNERS and rows[2]["also_in_fragment"] == "vectorless"
    frag = bv.job_rows(_cand("Frag_2", "V3_legacy_scraps", ["u4"]), {}, None, owners, {"BigBox"}, n_images)
    assert frag[0]["also_in"] == ["BigBox"] and "also_in_fragment" not in frag[0]


def test_fragment_overlap_ignores_containers() -> None:
    """A box-level co-owner never makes a fragment's image redundant."""
    n_images = {"Box": 20, "Frag": 19}
    assert bv.fragment_overlap(["Box"], n_images, {"Box"}) is None
    assert bv.fragment_overlap(["Box", "Frag"], n_images, {"Box"}) == "vectorless"
    assert bv.fragment_overlap(["Frag"], n_images, {"Frag"}) == "vector"
    assert bv.fragment_overlap([], n_images, set()) is None


def test_sort_rows_by_tranche_then_doc_then_index() -> None:
    """V1 before V2 before V3; then doc_id; then image_index."""
    rows = [{"queue": "V3_legacy_scraps", "doc_id": "A", "image_index": 0},
            {"queue": "V1_pgp_documentary", "doc_id": "Z", "image_index": 1},
            {"queue": "V1_pgp_documentary", "doc_id": "Z", "image_index": 0},
            {"queue": "V2_ktiv_literary", "doc_id": "B", "image_index": 0},
            {"queue": "V1_pgp_documentary", "doc_id": "M", "image_index": 5}]
    got = [(r["queue"][:2], r["doc_id"], r["image_index"]) for r in bv.sort_rows(rows)]
    assert got == [("V1", "M", 5), ("V1", "Z", 0), ("V1", "Z", 1), ("V2", "B", 0), ("V3", "A", 0)]


def test_select_pilot_is_deterministic_never_read_and_reachable() -> None:
    """Lowest SHA-1 per tranche, never-read documents only, at least one image outside the blocked URLs."""
    cands = [_cand(f"D{i}", "V1_pgp_documentary", [f"u{i}"]) for i in range(10)]
    cands += [_cand("R1", "V1_pgp_documentary", ["r1"]), _cand("X1", "V1_pgp_documentary", ["blocked"]),
              _cand("N1", "V1_pgp_documentary", []), _cand("T1", "V2_ktiv_literary", ["t1"])]
    classes = {c.doc_id: bv.NEVER_READ for c in cands}
    classes["R1"], classes["N1"] = bv.ALREADY_READ, bv.UNRESOLVED
    got = bv.select_pilot(cands, classes, {"V1_pgp_documentary": 4, "V2_ktiv_literary": 3, "V3_legacy_scraps": 2},
                          blocked_urls={"blocked"})
    expected_v1 = sorted([f"D{i}" for i in range(10)], key=bv.sha1_key)[:4]
    assert got["V1_pgp_documentary"] == expected_v1
    assert got["V2_ktiv_literary"] == ["T1"] and got["V3_legacy_scraps"] == []
    assert bv.select_pilot(list(reversed(cands)), classes, {"V1_pgp_documentary": 4},
                           blocked_urls={"blocked"})["V1_pgp_documentary"] == expected_v1


def test_compare_positions() -> None:
    """Same position, moved within the list, or no longer listed; unknown documents are not compared."""
    lists = {"D": ["k1", "l1", "l2"]}
    ref = [("D", 1, "l1"), ("D", 0, "l1"), ("D", 2, "gone.jpg"), ("D", 9, "l2"), ("E", 0, "x")]
    assert dict(bv.compare_positions(lists, ref)) == {"same": 1, "moved": 2, "gone": 1}


def test_percentile_and_image_stats() -> None:
    """Nearest-rank percentiles and the per-document image statistics."""
    assert bv.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.9) == 9
    assert bv.percentile([5], 0.99) == 5
    st = bv.image_stats([1, 2, 2, 3, 40])
    assert (st["docs"], st["images"], st["median"], st["max"]) == (5, 48, 2, 40)
    assert bv.image_stats([])["docs"] == 0


def test_tranche_table_totals() -> None:
    """The summary table counts documents per class and resolvable images, with tranche and grand totals."""
    cands = [_cand("A", "V1_pgp_documentary", ["a1", "a2"], "empty"), _cand("B", "V1_pgp_documentary", [], "short"),
             _cand("C", "V3_legacy_scraps", ["c1"], "boilerplate")]
    classes = {"A": bv.NEVER_READ, "B": bv.UNRESOLVED, "C": bv.ALREADY_READ}
    table = bv.tranche_table(cands, classes)
    assert "| **total** | all | 3 | 1 | 0 | 1 | 1 | 3 |" in table
    assert "| **V1_pgp_documentary** | all | 2 | 1 | 0 | 0 | 1 | 2 |" in table
    assert "| V3_legacy_scraps | boilerplate | 1 | 0 | 0 | 1 | 0 | 1 |" in table
