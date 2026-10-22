# File name: test_text27b_exposure.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for :mod:`src.finetuning.text27b.exposure` on synthetic reader mixtures (no real corpus)."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.finetuning.text27b import exposure as ex
from src.finetuning.text27b import paths as P

SYS_V3 = "990000000005555"
SYS_V4 = "990000000006666"
SYS_PILOT = "990000000001234"
IMG_C = b"synthetic-image-C"
IMG_D = b"synthetic-image-D"


def _sha(b: bytes) -> str:
    """sha256 hex of bytes.

    :param b: Bytes.
    :returns: Hex digest.
    """
    return hashlib.sha256(b).hexdigest()


def _jsonl(path: Path, rows: List[Dict[str, Any]]) -> Path:
    """Write rows as JSONL.

    :param path: Destination.
    :param rows: Rows.
    :returns: The path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def _clean_dataset(root: Path, items: List[Dict[str, Any]]) -> Path:
    """Write a minimal ``save_to_disk``-style train split (arrow stream with ``stem`` and ``image``).

    :param root: Dataset directory.
    :param items: ``[{"stem": ..., "bytes": ...}]``.
    :returns: The dataset directory.
    """
    (root / "train").mkdir(parents=True, exist_ok=True)
    img_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    schema = pa.schema([("image", img_type), ("stem", pa.string())])
    batch = pa.record_batch([pa.array([{"bytes": it["bytes"], "path": None} for it in items], type=img_type),
                             pa.array([it["stem"] for it in items])], schema=schema)
    with pa.OSFile(str(root / "train" / "data-00000-of-00001.arrow"), "wb") as sink:
        with pa.ipc.new_stream(sink, schema) as writer:
            writer.write_batch(batch)
    return root


def _ktiv_dir(root: Path, pages: Dict[str, List[str]]) -> Path:
    """Write a KTIV page directory ``<root>/<sys>/<file>``.

    :param root: Directory.
    :param pages: ``{sys_num: [file names]}``.
    :returns: The directory.
    """
    for sys_num, names in pages.items():
        (root / sys_num).mkdir(parents=True, exist_ok=True)
        for n in names:
            (root / sys_num / n).write_bytes(b"x")
    (root / "not_a_manuscript").mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def world(tmp_path: Path) -> Dict[str, Any]:
    """Synthetic reader mixtures, reads, corpus maps and row files.

    :param tmp_path: pytest temporary directory.
    :returns: ``{"sources": ExposureSources, "tmp": Path, "rows": [files]}``.
    """
    nas = tmp_path / "nas"
    nas.mkdir()
    pq.write_table(pa.table({"stem": ["pgp_100_0_page", "pgpqa_200_1_qa_date_0", "dg_Lib_X_1__0_loc0",
                                      f"ktiv_{SYS_PILOT}_FL111_sec0", "mystery_stem"],
                             "source": ["pgp_editions", "pgp_qa", "documentary_grounding", "ktiv_transcription",
                                        "other"]}), nas / "pilot_train.parquet")
    pq.write_table(pa.table({"stem": ["pgp_300_0_page"]}), nas / "pilot_val.parquet")
    manifests = {
        "pgp_editions": _jsonl(nas / "m_ed.jsonl", [{"pgpid": "100", "canonical_id": "Lib_A_1", "image_index": 0,
                                                     "image_url": "http://img/a1r.jpg", "image_sha256": "shaA1r",
                                                     "rows": ["pgp_100_0_page"]}]),
        "pgp_qa": _jsonl(nas / "m_qa.jsonl", [{"stem": "pgpqa_200_1_qa_date_0", "pgpid": "200", "canonical_id": "Lib_B_2",
                                               "image_index": 1, "image_url": "http://img/b2v.jpg"}]),
        "documentary_grounding": _jsonl(nas / "m_dg.jsonl", [{"doc_id": "Lib_X_1", "image_index": 0, "stem": "dg_Lib_X_1__0_loc0",
                                                              "image_url": "http://img/x1r.jpg"}]),
    }
    clean = {"clean_v2": _clean_dataset(tmp_path / "clean_v2", [{"stem": "Lib_C_3", "bytes": IMG_C}]),
             "clean_v1": _clean_dataset(tmp_path / "clean_v1", [{"stem": "Lib_D_4", "bytes": IMG_D}])}
    ktiv = {"ktiv_v3": _ktiv_dir(nas / "kv3", {SYS_V3: ["FL555.jpg", "FL555_line0.jpg"]}),
            "ktiv_v4": _ktiv_dir(nas / "kv4", {SYS_V4: ["FL666.jpg"], SYS_V3: ["FL555.jpg"]})}
    ktiv_url = lambda s, fl: f"http://img/KTIV/{s}/0001_{fl}.jpg"  # noqa: E731
    reads_a50 = [
        {"doc_id": "Lib_A_1", "image_index": 0, "image_url": "http://img/a1r.jpg", "image_sha256": "shaA1r"},
        {"doc_id": "Lib_A_1", "image_index": 1, "image_url": "http://img/a1v.jpg", "image_sha256": "shaA1v"},
        {"doc_id": "Lib_C_3", "image_index": 0, "image_url": "http://img/c3r.jpg", "image_sha256": _sha(IMG_C)},
        {"doc_id": "Old_Name_F", "image_index": 0, "image_url": "http://img/f6r.jpg", "image_sha256": "shaF"},
        {"doc_id": "Lib_F_6", "image_index": 1, "image_url": "http://img/f6v.jpg", "image_sha256": "shaF"},
        {"doc_id": "Not_PGP", "image_index": 0, "image_url": "http://img/zz.jpg", "image_sha256": "shaZ"},
    ]
    read_files = {"a50": _jsonl(tmp_path / "reads_a50.jsonl", reads_a50),
                  "v21b": _jsonl(tmp_path / "reads_v21b.jsonl", reads_a50[:3])}
    merged_map = {"Lib_A_1": {"pgpids": ["100"], "image_urls": ["http://img/a1r.jpg"]},
                  "Lib_B_2": {"pgpids": ["200", "201"], "image_urls": []},
                  "Lib_C_3": {"pgpids": ["400"], "image_urls": []},
                  "Lib_E_5": {"pgpids": ["600"], "image_urls": []},
                  "Lib_F_6": {"pgpids": ["700"], "image_urls": ["http://img/f6r.jpg"]},
                  "Lib_K_7": {"pgpids": ["500"], "image_urls": []}}
    (tmp_path / "merged_map.json").write_text(json.dumps(merged_map), encoding="utf-8")
    _jsonl(tmp_path / "merged.jsonl", [
        {"canonical_id": "Lib_E_5", "images": {"ktiv": {"sys_num": SYS_V3}}, "sources": {"ktiv": {"sys_num": SYS_V3}}},
        {"canonical_id": "Lib_A_1", "images": {}, "sources": {}},
        {"canonical_id": "Unwanted", "images": {"ktiv": {"sys_num": SYS_V4}}},
    ])

    def rd(model: str, pages: List[Dict[str, Any]], cer: float, cov: float = 1.0) -> Dict[str, Any]:
        """One parallel-corpus read.

        :param model: Reader.
        :param pages: Pages.
        :param cer: CER.
        :param cov: Coverage.
        :returns: Read dict.
        """
        return {"model": model, "pages": pages, "text": "t", "cer_vs_edition": cer, "coverage": cov}

    pg = lambda d, i, u: {"doc_id": d, "image_index": i, "image_url": u}  # noqa: E731
    _jsonl(tmp_path / "parallel.jsonl", [
        {"pgpid": "100", "merged_doc_ids": ["Lib_A_1"], "reads": [
            rd("a50", [pg("Lib_A_1", 0, "http://img/a1r.jpg"), pg("Lib_A_1", 1, "http://img/a1v.jpg")], 0.05),
            rd("v21b", [pg("Lib_A_1", 0, "http://img/a1r.jpg")], 0.30)]},
        {"pgpid": "200", "merged_doc_ids": ["Lib_B_2"], "reads": [rd("a50", [pg("Lib_B_2", 0, "http://img/b2r.jpg")], 0.2)]},
        {"pgpid": "201", "merged_doc_ids": ["Lib_B_2"], "reads": [rd("a50", [pg("Lib_B_2", 0, "http://img/b2r.jpg")], 0.4),
                                                                   rd("v21b", [pg("Lib_B_2", 0, "http://img/b2r.jpg")], 0.5)]},
        {"pgpid": "300", "merged_doc_ids": ["Lib_V_9"], "reads": [rd("a50", [pg("Lib_V_9", 0, "http://img/v9r.jpg")], 0.3)]},
        {"pgpid": "400", "merged_doc_ids": ["Lib_C_3"], "reads": [
            rd("v21b", [pg("Lib_C_3", 0, "http://img/c3r.jpg"), pg("Lib_C_3", 1, "http://img/c3v.jpg")], 0.1)]},
        {"pgpid": "500", "merged_doc_ids": ["Lib_K_7"], "reads": [
            rd("v21b", [pg("Lib_K_7", 0, ktiv_url(SYS_V3, "FL555"))], 0.2),
            rd("a50", [pg("Lib_K_7", 1, ktiv_url(SYS_V4, "FL666"))], 0.2, cov=3.0)]},
        {"pgpid": "501", "merged_doc_ids": ["Lib_K_8"], "reads": [
            rd("v21b", [pg("Lib_K_8", 0, ktiv_url(SYS_V3, "FL999"))], 0.6),
            rd("a50", [pg("Lib_K_8", 0, ktiv_url(SYS_PILOT, "FL111"))], 0.6)]},
        {"pgpid": "600", "merged_doc_ids": ["Lib_E_5"], "reads": [rd("v21b", [pg("Lib_E_5", 0, "http://img/e5r.jpg")], 0.7)]},
        {"pgpid": "800", "merged_doc_ids": [], "reads": []},
    ])
    v01 = tmp_path / "v01"
    meta = lambda p, m, c, split, cond="read": {"pgpid": p, "read_model": m, "input_cer": c, "split": split,  # noqa: E731
                                                "condition": cond, "task": "describe_pgp"}
    rows = [_jsonl(v01 / "train.jsonl", [
        {"messages": [], "meta": meta("100", "a50", 0.03, "train")},
        {"messages": [], "meta": meta("201", "v21b", 0.5, "train")},
        {"messages": [], "meta": meta("700", "a50", None, "train")},
        {"messages": [], "meta": meta("100", None, None, "train", cond="edition")},
    ]), _jsonl(v01 / "test.jsonl", [{"messages": [], "meta": meta("400", "v21b", 0.1, "test")}])]
    src = ex.ExposureSources(pilot_train=nas / "pilot_train.parquet", pilot_val=nas / "pilot_val.parquet",
                             manifests=manifests, clean=clean, ktiv_dirs=ktiv, read_files=read_files,
                             pgp_parallel=tmp_path / "parallel.jsonl", merged_pgpid_map=tmp_path / "merged_map.json",
                             merged_shelfmarks=tmp_path / "merged.jsonl")
    return {"sources": src, "tmp": tmp_path, "rows": rows, "ktiv_url": ktiv_url}


# ------------------------------------------------------------------------------------- helpers
def test_id_key_and_page_key() -> None:
    """Canonical-id keys drop case and punctuation; page keys join id and index.

    :returns: None.
    """
    assert ex.id_key("Cambridge_CUL_T_S_13J2_5") == "cambridgecults13j25"
    assert ex.id_key("") == ""
    assert ex.page_key("Lib_A_1", 2) == "Lib_A_1:2"


@pytest.mark.parametrize("stem,expected", [
    ("pgp_446_0_page", ("pgpid", "446", 0)),
    ("vqa_pgp_12_3_x", ("pgpid", "12", 3)),
    ("pgpqa_499_1_qa_date_0", ("pgpid", "499", 1)),
    ("dg_New_York_JTS_ENA_3697_8__0_loc0", ("cid", "New_York_JTS_ENA_3697_8", 0)),
    (f"ktiv_{SYS_PILOT}_FL202834361_gcrop0", ("ktiv", f"{SYS_PILOT}/FL202834361", None)),
    ("synthetic_hebrew_1", None),
])
def test_parse_pilot_stem(stem: str, expected: Any) -> None:
    """Pilot stems are classified by kind.

    :param stem: Stem.
    :param expected: Expected parse.
    :returns: None.
    """
    assert ex.parse_pilot_stem(stem) == expected


def test_ktiv_page_of_url() -> None:
    """KTIV-hosted URLs give (sys_num, FL id); others give None.

    :returns: None.
    """
    assert ex.ktiv_page_of_url(f"https://host/x/KTIV/{SYS_V3}/0002_FL168307758.jpg") == (SYS_V3, "FL168307758")
    assert ex.ktiv_page_of_url(f"https://host/x/KTIV/{SYS_V3}/FL1.png") == (SYS_V3, "FL1")
    assert ex.ktiv_page_of_url("https://host/images/VII_A_23_1r.jpg") is None
    assert ex.ktiv_page_of_url(None) is None


def test_level_constants() -> None:
    """Level ranks follow rows.EXPOSURE_LEVELS (page strongest, none weakest).

    :returns: None.
    """
    assert ex.LEVEL_RANK["page"] > ex.LEVEL_RANK["pgpid"] > ex.LEVEL_RANK["fragment"] > ex.LEVEL_RANK["none"] == 0
    assert ex._max_level([]) == "none"
    assert ex._max_level(["fragment", "page", "none"]) == "page"
    assert set(ex.READER_MIXTURES["v21b"]) < set(ex.READER_MIXTURES["a50"])


# ------------------------------------------------------------------------------- training index
def test_training_index_add_and_hits() -> None:
    """Index keys record their mixtures; hits are restricted to the reader's mixtures.

    :returns: None.
    """
    idx = ex.TrainingIndex()
    idx.add("url", "u", "v22a_pilot")
    idx.add("url", "u", "clean_v2")
    idx.add("url", "", "clean_v2")
    assert idx.hits("url", "u", ["clean_v2"]) == {"clean_v2"}
    assert idx.hits("url", "u", ["clean_v1"]) == set()
    assert idx.hits("url", None, ["clean_v2"]) == set()
    assert idx.sizes()["url"] == {"clean_v2": 1, "v22a_pilot": 1}
    with pytest.raises(KeyError):
        idx.add("bogus", "k", "m")


def test_build_training_index(world: Dict[str, Any]) -> None:
    """Pilot, clean and KTIV sources fill the expected keys.

    :param world: Synthetic fixture.
    :returns: None.
    """
    idx = ex.build_training_index(world["sources"], page_sys={SYS_V3})
    st = idx.stats
    assert st["v22a_pilot"]["rows"] == 5 and st["v22a_pilot"]["unknown_stems"] == 1
    assert st["v22a_pilot"]["pgpids"] == 2 and st["v22a_pilot"]["unmapped_pages"] == 0
    assert idx.hits("pgpid", "100", ["v22a_pilot"]) and idx.hits("pgpid", "200", ["v22a_pilot"])
    assert idx.hits("url", "http://img/a1r.jpg", ["v22a_pilot"])
    assert idx.hits("sha256", "shaA1r", ["v22a_pilot"])
    assert idx.hits("url", "http://img/b2v.jpg", ["v22a_pilot"])
    assert idx.hits("url", "http://img/x1r.jpg", ["v22a_pilot"])
    assert idx.hits("fragment", ex.id_key("Lib_B_2"), ["v22a_pilot"])      # pgp_qa row's fragment
    assert idx.hits("fragment", ex.id_key("Lib_X_1"), ["v22a_pilot"])      # dg row
    assert idx.hits("ktiv_page", f"{SYS_PILOT}/FL111", ["v22a_pilot"])
    assert idx.hits("ktiv_sys", SYS_PILOT, ["v22a_pilot"])
    assert idx.hits("fragment", ex.id_key("Lib_C_3"), ["clean_v2"]) and idx.hits("sha256", _sha(IMG_C), ["clean_v2"])
    assert idx.hits("fragment", ex.id_key("Lib_D_4"), ["clean_v1"]) and idx.hits("sha256", _sha(IMG_D), ["clean_v1"])
    assert idx.hits("ktiv_sys", SYS_V3, ["ktiv_v3"]) == {"ktiv_v3"}
    assert idx.hits("ktiv_page", f"{SYS_V3}/FL555", ex.KTIV_MIXTURES) == {"ktiv_v3", "ktiv_v4"}
    assert idx.hits("ktiv_sys", SYS_V4, ["ktiv_v4"]) and not idx.hits("ktiv_page", f"{SYS_V4}/FL666", ["ktiv_v4"])
    assert st["ktiv"]["ktiv_v3"] == {"manuscripts": 1, "listed": 1, "pages": 1}
    assert idx.val_pgpids == {"300"}
    assert st["missing_sources"] == []


def test_build_training_index_missing_sources(world: Dict[str, Any], tmp_path: Path) -> None:
    """Absent sources are skipped and named, not fatal.

    :param world: Synthetic fixture.
    :param tmp_path: pytest temporary directory.
    :returns: None.
    """
    src = world["sources"]
    src.pilot_train = tmp_path / "nope.parquet"
    src.ktiv_dirs = {"ktiv_v1": tmp_path / "nope_dir"}
    src.clean = {"clean_v1": tmp_path / "nope_clean"}
    idx = ex.build_training_index(src)
    assert "pilot_train" in idx.stats["missing_sources"] and "ktiv:ktiv_v1" in idx.stats["missing_sources"]
    assert idx.stats["ktiv"]["ktiv_v1"]["missing"] is True
    assert not idx.pgpid


# ------------------------------------------------------------------------------------------ reads
def test_load_read_pages_url_fallback(world: Dict[str, Any]) -> None:
    """Renamed ids are attached through the image-URL map; unknown ids are dropped.

    :param world: Synthetic fixture.
    :returns: None.
    """
    rp = ex.load_read_pages(world["sources"].read_files["a50"], known_ids={"Lib_A_1", "Lib_C_3", "Lib_F_6"},
                            url_to_id={"http://img/f6r.jpg": "Lib_F_6"})
    assert rp["by_cid"]["Lib_F_6"] == [0, 1]
    assert "Not_PGP" not in rp["by_cid"] and "Old_Name_F" not in rp["by_cid"]
    assert rp["sha_of_url"]["http://img/zz.jpg"] == "shaZ"
    everything = ex.load_read_pages(world["sources"].read_files["a50"])
    assert "Not_PGP" in everything["by_cid"]


def test_collect_reads(world: Dict[str, Any]) -> None:
    """Parallel reads keep their pages (sha filled); wanted reads are rebuilt from the reads file.

    :param world: Synthetic fixture.
    :returns: None.
    """
    src = world["sources"]
    cids_of, url_to_id = ex._merged_maps(src.merged_pgpid_map)
    known = {c for cs in cids_of.values() for c in cs}
    rp = {m: ex.load_read_pages(p, known, url_to_id) for m, p in src.read_files.items()}
    reads = ex.collect_reads(src.pgp_parallel, rp, cids_of, wanted={("700", "a50"), ("100", "a50")})
    by = {(r.pgpid, r.model): r for r in reads}
    assert by[("100", "a50")].source == "pgp_parallel"
    assert [p["image_sha256"] for p in by[("100", "a50")].pages] == ["shaA1r", "shaA1v"]
    assert by[("201", "a50")].cids == ["Lib_B_2"]
    pool = by[("700", "a50")]
    assert pool.source == "reads_file" and pool.cer is None
    assert len(pool.pages) == 1 and pool.pages[0]["image_sha256"] == "shaF"   # two pages, same sha → one
    assert len(ex.collect_reads(src.pgp_parallel, rp, cids_of, sample_docs=1)) == 2


def test_merged_sys_nums(world: Dict[str, Any]) -> None:
    """Only wanted ids with a KTIV system number are returned.

    :param world: Synthetic fixture.
    :returns: None.
    """
    got = ex.merged_sys_nums(world["sources"].merged_shelfmarks, {"Lib_E_5", "Lib_A_1"})
    assert got == {"Lib_E_5": {SYS_V3}}
    assert ex.merged_sys_nums(None, {"Lib_E_5"}) == {}


# --------------------------------------------------------------------------------- classification
@pytest.fixture()
def index(world: Dict[str, Any]) -> ex.TrainingIndex:
    """Training index of the synthetic world.

    :param world: Synthetic fixture.
    :returns: The index.
    """
    return ex.build_training_index(world["sources"], page_sys=None)


def _read(pid: str, model: str, pages: List[Any], cids: List[str]) -> ex.Read:
    """Build a Read.

    :param pid: pgpid.
    :param model: Reader.
    :param pages: ``[(doc_id, idx, url, sha)]``.
    :param cids: Canonical ids.
    :returns: The read.
    """
    return ex.Read(pid, model, [{"doc_id": d, "image_index": i, "image_url": u, "image_sha256": s}
                                for d, i, u, s in pages], cids)


def test_classify_page_and_pgpid(index: ex.TrainingIndex) -> None:
    """A trained page is page level; another page of the same document is pgpid level.

    :param index: Synthetic index.
    :returns: None.
    """
    recs = ex.classify_read(_read("100", "a50", [("Lib_A_1", 0, "http://img/a1r.jpg", "shaA1r"),
                                                  ("Lib_A_1", 1, "http://img/a1v.jpg", "shaA1v")], ["Lib_A_1"]),
                            index, {})
    assert [r["level"] for r in recs] == ["page", "pgpid"]
    assert {r["read_level"] for r in recs} == {"page"}
    assert recs[0]["routes"] == ["v22a_pilot:sha256", "v22a_pilot:url"]
    assert recs[1]["routes"] == ["v22a_pilot:pgpid"]
    assert recs[0]["page_key"] == "Lib_A_1:0"
    # v21b never saw the pilot
    v21 = ex.classify_read(_read("100", "v21b", [("Lib_A_1", 0, "http://img/a1r.jpg", "shaA1r")], ["Lib_A_1"]), index, {})
    assert v21[0]["level"] == "none" and v21[0]["routes"] == []


def test_classify_fragment_and_val(index: ex.TrainingIndex) -> None:
    """Another document on a trained fragment is fragment level; pilot val rows are not exposure.

    :param index: Synthetic index.
    :returns: None.
    """
    r201 = ex.classify_read(_read("201", "a50", [("Lib_B_2", 0, "http://img/b2r.jpg", "s")], ["Lib_B_2"]), index, {})
    assert r201[0]["level"] == "fragment" and r201[0]["routes"] == ["v22a_pilot:fragment"]
    r200 = ex.classify_read(_read("200", "a50", [("Lib_B_2", 0, "http://img/b2r.jpg", "s")], ["Lib_B_2"]), index, {})
    assert r200[0]["level"] == "pgpid"
    r300 = ex.classify_read(_read("300", "a50", [("Lib_V_9", 0, "http://img/v9r.jpg", "s")], ["Lib_V_9"]), index, {})
    assert r300[0]["level"] == "none"


def test_classify_clean_sha(index: ex.TrainingIndex) -> None:
    """The clean set's embedded image matches by sha256 (page); the document's other page is fragment.

    :param index: Synthetic index.
    :returns: None.
    """
    pages = [("Lib_C_3", 0, "http://img/c3r.jpg", _sha(IMG_C)), ("Lib_C_3", 1, "http://img/c3v.jpg", "other")]
    for model in ("v21b", "a50"):
        recs = ex.classify_read(_read("400", model, pages, ["Lib_C_3"]), index, {})
        assert [r["level"] for r in recs] == ["page", "fragment"]
        assert recs[0]["routes"] == ["clean_v2:sha256"] and recs[1]["routes"] == ["clean_v2:fragment"]


def test_classify_ktiv(world: Dict[str, Any], index: ex.TrainingIndex) -> None:
    """KTIV pages match by (sys, FL); a reader only sees the KTIV builds of its lineage.

    :param world: Synthetic fixture.
    :param index: Synthetic index.
    :returns: None.
    """
    url = world["ktiv_url"]
    r = ex.classify_read(_read("500", "v21b", [("Lib_K_7", 0, url(SYS_V3, "FL555"), "s")], []), index, {})
    assert r[0]["level"] == "page" and r[0]["routes"] == ["ktiv_v3:ktiv_page"]
    # same manuscript, another page → fragment via the system number
    r = ex.classify_read(_read("501", "v21b", [("Lib_K_8", 0, url(SYS_V3, "FL999"), "s")], []), index, {})
    assert r[0]["level"] == "fragment" and r[0]["routes"] == ["ktiv_v3:ktiv_sys"]
    # ktiv_v4 is a50-only
    r4 = [ex.classify_read(_read("500", m, [("Lib_K_7", 1, url(SYS_V4, "FL666"), "s")], []), index, {})[0]["level"]
          for m in ("a50", "v21b")]
    assert r4 == ["page", "none"]
    # the pilot's own KTIV rows
    rp = ex.classify_read(_read("501", "a50", [("Lib_K_8", 0, url(SYS_PILOT, "FL111"), "s")], []), index, {})
    assert rp[0]["level"] == "page" and rp[0]["routes"] == ["v22a_pilot:ktiv_page"]
    # merged-corpus system number of the fragment
    rm = ex.classify_read(_read("600", "v21b", [("Lib_E_5", 0, "http://img/e5r.jpg", "s")], ["Lib_E_5"]), index,
                          {"Lib_E_5": {SYS_V3}})
    assert rm[0]["level"] == "fragment" and "ktiv_v3:ktiv_sys" in rm[0]["routes"]


def test_classify_pageless_and_unknown_model(index: ex.TrainingIndex) -> None:
    """A read without pages yields one record; an unknown reader is skipped by build_exposure.

    :param index: Synthetic index.
    :returns: None.
    """
    recs = ex.classify_read(_read("200", "a50", [], ["Lib_B_2"]), index, {})
    assert len(recs) == 1 and recs[0]["page_key"] is None and recs[0]["level"] == "pgpid"
    with pytest.raises(KeyError):
        ex.classify_read(_read("200", "v99", [], []), index, {})
    out = ex.build_exposure([_read("200", "v99", [], []), _read("200", "a50", [], [])], index, {})
    assert [r["read_model"] for r in out] == ["a50"]


# ------------------------------------------------------------------------------------------ tagging
def test_exposure_lookup_and_tag_rows(tmp_path: Path) -> None:
    """Lookup keeps the strongest level per read; tag_rows sets the two meta fields.

    :param tmp_path: pytest temporary directory.
    :returns: None.
    """
    recs = [{"pgpid": "1", "read_model": "a50", "level": "pgpid", "read_level": "page"},
            {"pgpid": "1", "read_model": "a50", "level": "page", "read_level": "page"},
            {"pgpid": "2", "read_model": "v21b", "level": "none", "read_level": "none"},
            {"pgpid": 3, "read_model": "a50", "level": "fragment"}]
    path = _jsonl(tmp_path / "exposure.jsonl", recs)
    lk = ex.exposure_lookup(path)
    assert lk == {("1", "a50"): "page", ("2", "v21b"): "none", ("3", "a50"): "fragment"}
    assert ex.exposure_lookup(lk) == lk and ex.exposure_lookup(recs) == lk
    rows = [{"meta": {"condition": "read", "pgpid": "1", "read_model": "a50", "input_cer": 0.02}},
            {"meta": {"condition": "read_hard", "pgpid": "2", "read_model": "v21b", "input_cer": 0.6}},
            {"meta": {"condition": "read_partial", "pgpid": "9", "read_model": "a50"}},
            {"meta": {"condition": "noised_edition", "pgpid": "1"}},
            {"meta": {"condition": "edition", "pgpid": "1"}}]
    stats: collections.Counter = collections.Counter()
    out = ex.tag_rows(iter(rows), path, stats=stats)
    assert [r["meta"]["exposure_level"] for r in out] == ["page", "none", None, None, None]
    assert [r["meta"]["reader_exposed"] for r in out] == [True, False, None, False, False]
    assert stats == {"level:page": 1, "level:none": 1, "level:missing": 1, "level:n/a": 2}
    assert ex.near_duplicate(out[0]) and not ex.near_duplicate(out[1]) and not ex.near_duplicate(out[2])
    assert not ex.near_duplicate(out[0], cer_max=0.01)


# ------------------------------------------------------------------------------------------ report
def test_cer_summary() -> None:
    """Quantiles and low-CER shares; empty input gives n=0.

    :returns: None.
    """
    assert ex.cer_summary([]) == {"n": 0}
    assert ex.cer_summary([None]) == {"n": 0}
    s = ex.cer_summary([0.0, 0.1, 0.2, 0.3, None])
    assert s["n"] == 4 and s["mean"] == pytest.approx(0.15) and s["p50"] == pytest.approx(0.15)
    assert s["share_lt_0_05"] == 0.25 and s["share_lt_0_10"] == 0.25


def test_report_rows() -> None:
    """Row report counts levels, documents and near duplicates per split and reader.

    :returns: None.
    """
    lk = {("1", "a50"): "page", ("2", "a50"): "none", ("3", "v21b"): "fragment"}
    metas = [{"condition": "read", "pgpid": "1", "read_model": "a50", "split": "train", "input_cer": 0.01},
             {"condition": "read", "pgpid": "1", "read_model": "a50", "split": "train", "input_cer": 0.2},
             {"condition": "read", "pgpid": "2", "read_model": "a50", "split": "train", "input_cer": 0.3},
             {"condition": "read", "pgpid": "3", "read_model": "v21b", "split": "dev", "input_cer": 0.1},
             {"condition": "read", "pgpid": "4", "read_model": "v21b", "split": "dev", "input_cer": 0.1},
             {"condition": "edition", "pgpid": "1", "split": "train"}]
    rep = ex.report_rows(metas, lk)
    a = rep["train"]["a50"]
    assert a["rows"] == 3 and a["exposed_rows"] == 2 and a["docs"] == 2 and a["exposed_docs"] == 1
    assert a["rows_by_level"] == {"none": 1, "page": 2} and a["near_duplicates"] == 1
    assert a["cer_exposed"]["n"] == 2 and a["cer_unexposed"]["n"] == 1 and a["cer_by_level"]["page"]["n"] == 2
    d = rep["dev"]["all"]
    assert d["rows"] == 2 and d["missing"] == 1 and d["rows_by_level"] == {"fragment": 1, "missing": 1}


def test_report_reads_and_paired() -> None:
    """Parallel-read report splits in-gate reads; the paired report groups documents by who is exposed.

    :returns: None.
    """
    reads = [ex.Read("1", "a50", [], [], 0.05, 1.0), ex.Read("1", "v21b", [], [], 0.25, 1.0),
             ex.Read("2", "a50", [], [], 0.30, 1.0), ex.Read("2", "v21b", [], [], 0.32, 0.3),
             ex.Read("3", "a50", [], [], None, None, "reads_file")]
    lk = {("1", "a50"): "page", ("1", "v21b"): "none", ("2", "a50"): "none", ("2", "v21b"): "none"}
    rep = ex.report_reads(reads, lk)
    assert rep["a50"]["reads"] == 3 and rep["a50"]["reads_by_level"] == {"missing": 1, "none": 1, "page": 1}
    assert rep["a50"]["exposed_docs"] == 1
    assert rep["a50"]["cer_all"]["exposed"]["n"] == 1 and rep["a50"]["cer_all"]["page"]["p50"] == 0.05
    assert rep["v21b"]["cer_in_gate"]["unexposed"]["n"] == 1 and rep["v21b"]["cer_all"]["unexposed"]["n"] == 2
    pr = ex.report_paired(reads, lk)
    assert pr["all"]["a50_only"]["n"] == 1 and pr["all"]["a50_only"]["mean"] == pytest.approx(-0.2)
    assert pr["all"]["neither"]["mean"] == pytest.approx(-0.02)
    assert pr["all"]["did_vs_neither"]["a50_only"]["mean_diff"] == pytest.approx(-0.18)
    assert "neither" not in pr["in_gate"] and "did_vs_neither" not in pr["in_gate"]


def test_bootstrap_mean_diff() -> None:
    """Mean difference with a seeded interval that brackets it; empty samples give None.

    :returns: None.
    """
    a = [0.1, 0.2, 0.3, 0.2, 0.25]
    b = [0.4, 0.5, 0.45, 0.35]
    r = ex.bootstrap_mean_diff(a, b, n_boot=500, seed=1)
    assert r["mean_diff"] == pytest.approx(0.21 - 0.425, abs=1e-4)
    assert r["ci95"][0] <= r["mean_diff"] <= r["ci95"][1] < 0
    assert r == ex.bootstrap_mean_diff(a, b, n_boot=500, seed=1)
    assert ex.bootstrap_mean_diff([], b)["mean_diff"] is None


def test_compare_prototype() -> None:
    """Agreement with the probe evidence format, page level and any level.

    :returns: None.
    """
    overlap = {"docs": {"1": {"read_model": "a50", "a50_trained": True, "v22a_page_url_hits": ["u"]},
                        "2": {"read_model": "a50", "a50_trained": True, "v22a_page_url_hits": []},
                        "3": {"read_model": "a50", "a50_trained": False},
                        "4": {"read_model": "v21b", "a50_trained": False},
                        "5": {"read_model": "a50", "a50_trained": False}}}
    lk = {("1", "a50"): "page", ("2", "a50"): "none", ("3", "a50"): "fragment"}
    res = ex.compare_prototype(lk, overlap)
    assert res["compared"] == 3 and res["exposed_agree"] == 1
    assert res["exposed_only_probe"] == ["2"] and res["exposed_only_here"] == ["3"]
    assert res["page_agree"] == 3


# ------------------------------------------------------------------------------------- end to end
def _paths_json(world: Dict[str, Any]) -> Path:
    """Write a complete synthetic ``paths.json`` for the CLI.

    :param world: Synthetic fixture.
    :returns: The config path.
    """
    t: Path = world["tmp"]
    src: ex.ExposureSources = world["sources"]
    cfg = {k: str(t / k) for k in ("probe_dir", "pgp_export", "document_treatments", "analysis_pages_index",
                                   "exclusions_v1", "samples", "benchmark_ids", "academic_literature",
                                   "tokenizer_json", "chat_template", "scratch", "pgp_raw")}
    cfg.update({"repo": str(t), "text_ft_v0_1": str(t / "v01"), "pgp_parallel": str(src.pgp_parallel),
                "merged_pgpid_map": str(src.merged_pgpid_map), "merged_shelfmarks": str(src.merged_shelfmarks),
                "a50_overlap": str(t / "overlap.json"), "ai_reads": str(t), "book_editions_nas": str(t / "nas" / "x"),
                "out_dir": str(t / "out"),
                "exposure_sources": {"pilot_train": str(src.pilot_train), "pilot_val": str(src.pilot_val),
                                     **{f"manifest:{k}": str(v) for k, v in src.manifests.items()},
                                     "clean:clean_v2": str(src.clean["clean_v2"]),
                                     "clean:clean_v1": str(src.clean["clean_v1"]),
                                     "ktiv:ktiv_v1": str(t / "absent_ktiv_v1"), "ktiv:ktiv_v2": str(t / "absent_ktiv_v2"),
                                     "ktiv:ktiv_v3": str(src.ktiv_dirs["ktiv_v3"]),
                                     "ktiv:ktiv_v4": str(src.ktiv_dirs["ktiv_v4"]),
                                     "read:a50": str(src.read_files["a50"]), "read:v21b": str(src.read_files["v21b"])}})
    (t / "overlap.json").write_text(json.dumps({"docs": {"100": {"read_model": "a50", "a50_trained": True,
                                                                  "v22a_page_url_hits": ["http://img/a1r.jpg"]}}}))
    path = t / "paths.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def test_sources_from_paths(world: Dict[str, Any]) -> None:
    """Defaults resolve against the NAS root and repo; paths.json and overrides replace entries.

    :param world: Synthetic fixture.
    :returns: None.
    """
    p = P.load_paths(_paths_json(world))
    src = ex.ExposureSources.from_paths(p)
    assert src.pilot_train == world["sources"].pilot_train
    assert set(src.ktiv_dirs) == set(ex.KTIV_MIXTURES) and set(src.read_files) == {"a50", "v21b"}
    assert src.missing() == ["ktiv:ktiv_v1", "ktiv:ktiv_v2"]
    assert src.describe()["ktiv:ktiv_v1"]["exists"] is False
    bare = ex.ExposureSources.from_paths(p.__class__(**{**p.__dict__, "extra": {}}),
                                         overrides={"clean:clean_v1": "/x/y"})
    assert bare.pilot_train == p.book_editions_nas.parent / ex.NAS_SOURCES["pilot_train"]
    assert bare.clean["clean_v2"] == p.repo / ex.REPO_SOURCES["clean:clean_v2"]
    assert bare.clean["clean_v1"] == Path("/x/y")


def test_main_end_to_end(world: Dict[str, Any], capsys: pytest.CaptureFixture) -> None:
    """The CLI writes exposure.jsonl and the report, covering parallel and row-only reads.

    :param world: Synthetic fixture.
    :param capsys: pytest output capture.
    :returns: None.
    """
    cfg = _paths_json(world)
    assert ex.main(["--paths", str(cfg)]) == 2                          # two KTIV dirs are absent
    assert not (world["tmp"] / "out" / "exposure.jsonl").exists()
    assert ex.main(["--paths", str(cfg), "--allow-missing"]) == 0
    out = world["tmp"] / "out"
    recs = [json.loads(x) for x in (out / "exposure.jsonl").read_text().splitlines()]
    lk = ex.exposure_lookup(recs)
    assert lk[("100", "a50")] == "page" and lk[("100", "v21b")] == "none"
    assert lk[("201", "a50")] == "fragment" and lk[("201", "v21b")] == "none"
    assert lk[("400", "v21b")] == "page" and lk[("500", "v21b")] == "page" and lk[("500", "a50")] == "page"
    assert lk[("501", "v21b")] == "fragment" and lk[("600", "v21b")] == "fragment" and lk[("300", "a50")] == "none"
    assert ("700", "a50") in lk                                          # read-only pool read from the reads file
    rep = json.loads((out / "exposure_report.json").read_text())
    assert rep["reads"]["by_source"] == {"pgp_parallel": 12, "reads_file": 1}
    assert rep["rows"]["train"]["a50"]["near_duplicates"] == 1
    assert rep["rows"]["test"]["v21b"]["rows_by_level"] == {"page": 1}
    assert rep["prototype_check"]["exposed_agree"] == 1 and rep["prototype_check"]["page_agree"] == 1
    assert rep["pilot_val_only_reads"] == 1
    assert rep["training_index"]["stats"]["missing_sources"] == ["ktiv:ktiv_v1", "ktiv:ktiv_v2"]
    assert "rows train" in capsys.readouterr().out
    # tag the v0_1-style rows with the written file
    rows = [json.loads(x) for x in world["rows"][0].read_text().splitlines()]
    ex.tag_rows(rows, out / "exposure.jsonl")
    assert [r["meta"]["reader_exposed"] for r in rows] == [True, False, False, False]
