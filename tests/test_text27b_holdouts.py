# File name: test_text27b_holdouts.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for text27b ``holdouts``: the hold-out set H built from canonical ids, pgpids and
shelfmark keys, the group closure, the eval tier, persistence and the CLI.

Everything runs on a synthetic world in ``tmp_path`` (merged corpus, PGP export CSVs, a pgpid
map that misses the Bodleian spelling, v0_1 split files, probe files, extra benchmark files);
:func:`make_world` is reused by ``test_text27b_contamination.py``.
"""
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.finetuning.text27b import holdouts as H
from src.finetuning.text27b import paths as P

#: Registered benchmark documents of the synthetic world (canonical ids, pgpids).
REGISTRY = ({"Cambridge_CUL_T_S_NS_999_5"}, {"9005"})
#: The synthetic PGP-131 list: one id the pgpid map knows, two Bodleian ids it does not.
B131 = ["Cambridge_CUL_T_S_99J1_2", "Oxford_Bodleian_MS_heb_d_99_7", "Oxford_Bodleian_MS_heb_f_77_3"]


def _write_csv(path: Path, header: List[str], rows: List[Dict[str, str]]) -> None:
    """Write a CSV with a BOM, as the PGP export does.

    :param path: Destination.
    :param header: Column names.
    :param rows: Rows.
    :returns: None.
    """
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def _jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write JSONL.

    :param path: Destination.
    :param rows: Objects.
    :returns: None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def v01_row(pgpid: str, split: str, shelfmark: str, task: str = "describe_pgp", condition: str = "edition",
            target: str = "A letter.", document: str = "שלום עליכם", vision: bool = False,
            group: Optional[str] = None) -> Dict[str, Any]:
    """A v0_1-style row.

    :param pgpid: PGP id.
    :param split: Split.
    :param shelfmark: Shelfmark.
    :param task: Task.
    :param condition: Input condition.
    :param target: Assistant text.
    :param document: Text inside ``<document>``.
    :param vision: ``held_out_vision_benchmark``.
    :param group: ``shelfmark_group`` (default ``g<pgpid>``).
    :returns: The row.
    """
    return {"messages": [{"role": "system", "content": "sys"},
                         {"role": "user", "content": f"Do it.\n\n<document>\n{document}\n</document>"},
                         {"role": "assistant", "content": target}],
            "meta": {"task": task, "condition": condition, "pgpid": pgpid, "shelfmark": shelfmark,
                     "shelfmark_group": group or f"g{pgpid}", "split": split, "held_out_vision_benchmark": vision}}


def make_world(tmp_path: Path, v01_extra: Optional[Dict[str, List[Dict[str, Any]]]] = None,
               pairs_extra: Optional[List[Dict[str, Any]]] = None) -> P.Paths:
    """Write the synthetic world and its ``paths.json``.

    pgpids: 9001 (PGP-131, mapped; fragment shared with train doc 9011), 9002 (PGP-131 Bodleian d,
    unmapped, v0_1 train), 9003 (PGP-131 Bodleian f, unmapped, found only through the PGP record),
    9004 / 9005 (registry), 9006 (v0_1 vision flag; fragment shared with 9012), 9010 (plain train),
    9020 (dev), 9021 (test),
    9030 / 9031 (probe), 9040 (extra benchmark pgpid), 9041 (extra benchmark by shelfmark).

    :param tmp_path: Pytest temporary directory (the synthetic repo root).
    :param v01_extra: Extra v0_1 rows per split.
    :param pairs_extra: Extra ``pgp_parallel`` rows.
    :returns: Loaded :class:`Paths`.
    """
    data = tmp_path / "data"
    pgp = data / "pgp"
    reads = data / "reads"
    pgp.mkdir(parents=True)
    reads.mkdir()
    merged = [
        {"canonical_id": "Cambridge_CUL_T_S_99J1_2", "shelfmark_display": "T-S 99J1.2", "pgpids": [9001]},
        {"canonical_id": "Oxford_Bodleian_Bodl_MS_heb_d_99_7", "shelfmark_display": "Bodl. MS heb. d 99/7", "pgpids": [9002]},
        {"canonical_id": "Oxford_Bodleian_Bodl_MS_heb_f_77_3", "shelfmark_display": "Bodl. MS heb. f 77/3", "pgpids": []},
        {"canonical_id": "Cambridge_CUL_T_S_NS_999_5", "shelfmark_display": "T-S NS 999.5", "pgpids": [9004]},
        {"canonical_id": "Cambridge_CUL_T_S_88J2_1", "shelfmark_display": "T-S 88J2.1", "pgpids": [9010]},
    ]
    _jsonl(data / "merged.jsonl", [dict(r, sources_present=["pgp"], description="") for r in merged])
    docs = {"9001": "T-S 99J1.2", "9002": "Bodl. MS heb. d 99/7", "9003": "Bodl. MS heb. f 77/3", "9004": "T-S NS 999.5",
            "9005": "T-S 44J4.4", "9006": "T-S 33J3.3", "9010": "T-S 88J2.1", "9011": "T-S 99J1.2 + T-S 99J1.9",
            "9020": "T-S 77J3.4", "9021": "T-S 76J3.4", "9030": "T-S 22J2.2", "9031": "T-S 21J2.2", "9040": "T-S 66J5.5",
            "9041": "T-S 55J6.6", "9012": "T-S 33J3.8"}
    _write_csv(pgp / "documents.csv", ["pgpid", "shelfmark", "type", "description", "has_transcription", "has_translation"],
               [{"pgpid": k, "shelfmark": v, "description": f"Description of document {k}."} for k, v in docs.items()])
    _write_csv(pgp / "footnotes.csv", ["document_id", "source_slug", "location", "doc_relation", "content"], [])
    _write_csv(pgp / "fragments.csv", ["shelfmark", "pgpids", "shelfmarks_historic"],
               [{"shelfmark": "T-S 99J1.2", "pgpids": "9001;9011"}, {"shelfmark": "T-S 88J2.1", "pgpids": "9010"},
                {"shelfmark": "T-S 33J3.3", "pgpids": "9006;9012"},
                {"shelfmark": "Bodl. MS heb. f 77/3", "pgpids": "9003"}])
    _write_csv(pgp / "sources.csv", ["slug", "title", "year", "authors"], [])
    json.dump({"Cambridge_CUL_T_S_99J1_2": {"pgpids": ["9001"]}, "Cambridge_CUL_T_S_NS_999_5": {"pgpids": ["9004"]},
               "Cambridge_CUL_T_S_88J2_1": {"pgpids": ["9010"]}}, open(data / "merged_map.json", "w"))
    json.dump(B131, open(data / "benchmark_ids.json", "w"))
    v01 = {"train": [v01_row("9002", "train", "Bodl. MS heb. d 99/7"), v01_row("9002", "train", "Bodl. MS heb. d 99/7", task="catalogue_json"),
                     v01_row("9003", "train", "Bodl. MS heb. f 77/3"), v01_row("9010", "train", "T-S 88J2.1"),
                     v01_row("9011", "train", "T-S 99J1.2 + T-S 99J1.9")],
           "dev": [v01_row("9020", "dev", "T-S 77J3.4")],
           "test": [v01_row("9006", "test", "T-S 33J3.3", vision=True), v01_row("9021", "test", "T-S 76J3.4")]}
    for sp, rows in (v01_extra or {}).items():
        v01.setdefault(sp, []).extend(rows)
    for sp, rows in v01.items():
        _jsonl(data / "v01" / f"{sp}.jsonl", rows)
    json.dump({"meta": {}, "sets": {"S1": {"entries": [{"pgpid": "9030"}], "pgpids": ["9030"]}}}, open(data / "samples.json", "w"))
    json.dump({"meta": {}, "entries": [{"pgpid": "9031", "reasons": ["x"]}]}, open(data / "exclusions.json", "w"))
    pairs = [{"pgpid": "9001", "shelfmark": "T-S 99J1.2", "merged_doc_ids": ["Cambridge_CUL_T_S_99J1_2"],
              "held_out_vision_benchmark": False, "split": "test", "edition_text": "", "translation_en": None}]
    _jsonl(data / "pairs.jsonl", pairs + list(pairs_extra or []))
    json.dump({"benchmark": "sealed_x", "ids": ["Cambridge_CUL_T_S_66J5_5"], "pgpids": ["9040"]}, open(data / "sealed.json", "w"))
    json.dump({"benchmark": "relig_x", "docs": [{"doc_id": "K1", "shelf_mark": "Cambridge University Library, Cambridge, England Ms. T-S 55J6.6"}]},
              open(data / "relig.json", "w"))
    cfg: Dict[str, Any] = {"repo": str(tmp_path), "merged_shelfmarks": "data/merged.jsonl", "pgp_export": "data/pgp",
                           "ai_reads": "data/reads", "out_dir": "out/v0_2", "merged_pgpid_map": "data/merged_map.json",
                           "benchmark_ids": "data/benchmark_ids.json", "text_ft_v0_1": "data/v01",
                           "samples": "data/samples.json", "exclusions_v1": "data/exclusions.json",
                           "pgp_parallel": "data/pairs.jsonl", "document_treatments": "data/treatments.jsonl",
                           "analysis_pages_index": "data/page_index.jsonl", "academic_literature": "data/lit",
                           "extra_benchmarks": ["data/sealed.json", "data/relig.json"]}
    for k in ["probe_dir", "a50_overlap", "book_editions_nas", "tokenizer_json", "chat_template", "scratch"]:
        cfg[k] = f"data/{k}"
    f = tmp_path / "paths.json"
    f.write_text(json.dumps(cfg))
    return P.load_paths(f)


@pytest.fixture()
def world(tmp_path: Path) -> P.Paths:
    """The synthetic world.

    :param tmp_path: Pytest temporary directory.
    :returns: Its :class:`Paths`.
    """
    return make_world(tmp_path)


@pytest.fixture()
def held(world: P.Paths) -> H.Holdouts:
    """H built over the synthetic world.

    :param world: Synthetic paths.
    :returns: The hold-out set.
    """
    return H.build_holdouts(world, registry=REGISTRY)


# ------------------------------------------------------------------------------------ build
def test_unmapped_bodleian_ids_resolve_by_key(held: H.Holdouts) -> None:
    """Benchmark ids missing from the pgpid map resolve through the shelfmark key (d- and f-size leaves).

    :param held: Hold-out set.
    :returns: None.
    """
    assert held.is_benchmark(pgpid="9002") == "key:pgp131"
    assert held.is_benchmark(pgpid="9003") == "key:pgp131"
    assert held.is_benchmark(shelfmark="Bodl. MS heb. d 99/7")
    assert held.is_benchmark(shelfmark="MS heb. f 77/3")
    assert held.is_benchmark(canonical="Oxford_Bodleian_Bodl_MS_heb_d_99_7")
    res = {r["canonical_id"]: r for r in held.meta["pgp131_unmapped_resolution"]}
    assert set(res) == {"Oxford_Bodleian_MS_heb_d_99_7", "Oxford_Bodleian_MS_heb_f_77_3"}
    assert res["Oxford_Bodleian_MS_heb_d_99_7"]["pgpids"] == ["9002"]
    assert res["Oxford_Bodleian_MS_heb_d_99_7"]["v0_1_rows"] == {"9002": {"train": 2}}
    assert res["Oxford_Bodleian_MS_heb_f_77_3"]["pgpids"] == ["9003"]
    assert "pgp_documents" in res["Oxford_Bodleian_MS_heb_f_77_3"]["via"]["9003"]
    assert held.meta["counts"]["pgp131_unmapped_in_pgpid_map"] == 2
    assert held.meta["counts"]["pgp131_without_any_pgpid"] == 0


def test_direct_reasons(held: H.Holdouts) -> None:
    """Mapped PGP-131, registry ids and pgpids, the v0_1 vision flag and probe documents are benchmark.

    :param held: Hold-out set.
    :returns: None.
    """
    assert held.is_benchmark(pgpid="9001") == "pgp131"
    assert held.is_benchmark(pgpid="9004") == "registry"
    assert held.is_benchmark(pgpid=9005) == "registry"
    assert held.is_benchmark(pgpid="9006") == "vision_flag"
    assert held.is_benchmark(pgpid="9030") == "probe"
    assert held.is_benchmark(pgpid="9031") == "probe"
    assert held.is_benchmark(pgpid="9010") is None
    assert held.is_eval_doc(pgpid="9010") is None
    assert held.is_held(pgpid="9010", shelfmark="T-S 88J2.1", canonical="Cambridge_CUL_T_S_88J2_1") is None


def test_group_closure(held: H.Holdouts) -> None:
    """A document sharing a fragment with a benchmark document is held out (``group:``).

    :param held: Hold-out set.
    :returns: None.
    """
    assert held.is_benchmark(pgpid="9011") == "group:pgp131"
    assert held.benchmark["9011"] == ["group:pgp131", "key:pgp131"]
    assert held.is_benchmark(pgpid="9012") == "group:vision_flag"
    assert held.meta["counts"]["benchmark_pgpids_added_by_group"] == 1
    assert set(held.meta["benchmark_in_v0_1_train"]) == {"9002", "9003", "9011"}


def test_joined_shelfmark_matches_any_part(held: H.Holdouts) -> None:
    """Every part of a joined shelfmark is checked against the benchmark keys.

    :param held: Hold-out set.
    :returns: None.
    """
    assert held.is_benchmark(shelfmark="T-S 12J9.9 + T-S 99J1.2") == "key:pgp131"
    assert held.is_benchmark(shelfmark="T-S 12J9.9") is None


def test_eval_tier(held: H.Holdouts) -> None:
    """v0_1 dev / test documents and extra benchmarks form the eval tier, separate from the benchmark tier.

    :param held: Hold-out set.
    :returns: None.
    """
    assert held.is_eval_doc(pgpid="9020") == "dev"
    assert held.is_eval_doc(pgpid="9021") == "test"
    assert held.is_benchmark(pgpid="9020") is None
    assert held.is_eval_doc(shelfmark="T-S 77J3.4") == "key:dev"
    assert held.is_eval_doc(pgpid="9040") == "extra:sealed_x"
    assert held.is_eval_doc(pgpid="9041") == "key:extra:relig_x"
    assert held.is_eval_doc(canonical="Cambridge_CUL_T_S_66J5_5") == "extra:sealed_x"
    assert held.is_held(pgpid="9021") == "test"
    assert held.is_held(pgpid="9006") == "vision_flag"
    assert held.meta["extra_benchmarks"]["relig_x"]["shelfmarks"] == 1


def test_match_any_and_best_reason(held: H.Holdouts) -> None:
    """``match_any`` returns the most important reason over several identifiers.

    :param held: Hold-out set.
    :returns: None.
    """
    assert held.match_any(pgpids=["9021", "9001"]) == "pgp131"
    assert held.match_any(pgpids=["9021"], include_eval=False) is None
    assert held.match_any(shelfmarks=["T-S 76J3.4"]) == "key:test"
    assert H.best_reason(["key:test", "group:probe", "dev"]) == "group:probe"
    assert H.best_reason(["key:pgp131", "registry"]) == "key:pgp131"
    assert H.best_reason(["key:group:pgp131", "key:pgp131"]) == "key:pgp131"
    assert H.base_reason("key:group:test") == "test" and H.derive("group:", "key:dev") == "group:dev"
    assert H.best_reason(["extra:x", "test"]) == "extra:x"
    assert H.best_reason([]) is None


def test_no_extra_files(world: P.Paths) -> None:
    """``extra_files=[]`` drops the extra benchmarks.

    :param world: Synthetic paths.
    :returns: None.
    """
    h = H.build_holdouts(world, registry=REGISTRY, extra_files=[])
    assert h.is_eval_doc(pgpid="9040") is None
    assert h.meta["extra_benchmarks"] == {}


def test_save_load_roundtrip(held: H.Holdouts, tmp_path: Path) -> None:
    """``save`` / ``load`` keep every map and the report.

    :param held: Hold-out set.
    :param tmp_path: Pytest temporary directory.
    :returns: None.
    """
    f = held.save(tmp_path / "h.json")
    h2 = H.Holdouts.load(f)
    assert h2.pgpids == held.pgpids and h2.eval_pgpids == held.eval_pgpids
    assert h2.keys == held.keys and h2.eval_keys == held.eval_keys
    assert h2.is_benchmark(pgpid="9002") == "key:pgp131"
    assert h2.meta["counts"] == held.meta["counts"]
    data = json.loads(f.read_text())
    assert list(data["benchmark"]) == sorted(data["benchmark"], key=int)
    assert data["format_version"] == H.FORMAT_VERSION
    assert "mshebd99_7" in data["keys"]


def test_reasons_of(held: H.Holdouts) -> None:
    """``reasons_of`` lists the stored reasons of both tiers.

    :param held: Hold-out set.
    :returns: None.
    """
    assert held.reasons_of("9006")[0] == "vision_flag"
    assert "test" in held.reasons_of("9006")
    assert held.reasons_of("424242") == []


# ------------------------------------------------------------------------------------ sources
def test_load_id_list_shapes(tmp_path: Path) -> None:
    """A plain list and a registry object are read; anything else raises.

    :param tmp_path: Pytest temporary directory.
    :returns: None.
    """
    (tmp_path / "a.json").write_text(json.dumps(["X_1"]))
    (tmp_path / "b.json").write_text(json.dumps({"ids": ["Y_2"]}))
    (tmp_path / "c.json").write_text(json.dumps({"nope": 1}))
    assert H.load_id_list(tmp_path / "a.json") == ["X_1"]
    assert H.load_id_list(tmp_path / "b.json") == ["Y_2"]
    with pytest.raises(ValueError):
        H.load_id_list(tmp_path / "c.json")


def test_walk_pgpids_and_probe(tmp_path: Path) -> None:
    """``pgpid`` and ``pgpids`` values are collected at any depth; missing probe files are skipped.

    :param tmp_path: Pytest temporary directory.
    :returns: None.
    """
    assert H.walk_pgpids({"a": [{"pgpid": 1}, {"b": {"pgpids": ["2", 3]}}], "pgpid": ""}) == {"1", "2", "3"}
    (tmp_path / "s.json").write_text(json.dumps({"meta": {"pgpid": "99"}, "sets": {"x": {"pgpids": ["5"]}}}))
    assert H.probe_pgpids(tmp_path / "s.json", tmp_path / "missing.json") == {"5"}


def test_load_extra_benchmark(tmp_path: Path) -> None:
    """Registry-shaped and doc-list-shaped benchmark files are both read.

    :param tmp_path: Pytest temporary directory.
    :returns: None.
    """
    (tmp_path / "e.json").write_text(json.dumps({"benchmark": "b1", "ids": ["A_1"], "pgpids": ["7"],
                                                 "docs": [{"shelf_mark": "T-S 1J1.1"}, {"shelfmark": "T-S 1J1.1"}]}))
    eb = H.load_extra_benchmark(tmp_path / "e.json")
    assert (eb.name, eb.ids, eb.pgpids, eb.shelfmarks) == ("b1", {"A_1"}, {"7"}, ["T-S 1J1.1"])


def test_scan_v01_and_pairs(world: P.Paths) -> None:
    """The v0_1 scan counts rows per split, flags and groups; ``scan_pairs`` reads the vision flag.

    :param world: Synthetic paths.
    :returns: None.
    """
    v = H.scan_v01(world.text_ft_v0_1)
    assert v["9002"].splits == {"train": 2} and v["9002"].tasks == {"describe_pgp": 1, "catalogue_json": 1}
    assert v["9006"].vision and v["9006"].group == "g9006"
    assert H.scan_v01(world.text_ft_v0_1, splits=("nope",)) == {}
    pairs = H.scan_pairs(world.pgp_parallel)
    assert pairs["9001"].merged_ids == ["Cambridge_CUL_T_S_99J1_2"] and not pairs["9001"].vision
    assert H.scan_pairs(None) == {}


def test_pairs_vision_flag(tmp_path: Path) -> None:
    """A ``pgp_parallel`` vision flag (string form) also holds a document out.

    :param tmp_path: Pytest temporary directory.
    :returns: None.
    """
    p = make_world(tmp_path, pairs_extra=[{"pgpid": "9050", "shelfmark": "T-S 1J5.5", "held_out_vision_benchmark": "True"}])
    h = H.build_holdouts(p, registry=REGISTRY)
    assert h.is_benchmark(pgpid="9050") == "vision_flag"


def test_union_find_groups() -> None:
    """Connected components with the smallest numeric id as root.

    :returns: None.
    """
    g = H.build_groups([["30", "4"], ["4", "100"], ["7"]], universe=["8"])
    assert g["100"] == ["4", "30", "100"] and g["7"] == ["7"] and g["8"] == ["8"]
    uf = H.UnionFind()
    uf.union("12", "3")
    assert uf.find("12") == "3"


def test_key_index_sources() -> None:
    """The key index links keys to pgpids and records where each link came from.

    :returns: None.
    """
    ki = H.KeyIndex()
    ki.add_shelfmark("T-S 99J1.2 + T-S 99J1.9", ["1"], "docs")
    ki.add_canonical("Oxford_Bodleian_Bodl_MS_heb_d_99_7", ["2"], "merged", "Bodl. MS heb. d 99/7")
    assert ki.lookup("ts99j1_9") == {"1"}
    assert ki.lookup("mshebd99_7") == {"2"} and ki.sources[("mshebd99_7", "2")] == {"merged"}
    assert ki.lookup("") == set()


# ------------------------------------------------------------------------------------ CLI
def test_cli_build_show_check(world: P.Paths, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """``build`` writes ``holdouts.json`` under ``out_dir`` and prints the Bodleian resolution; ``check`` answers.

    :param world: Synthetic paths.
    :param monkeypatch: Pytest monkeypatch (synthetic registry).
    :param capsys: Pytest output capture.
    :returns: None.
    """
    monkeypatch.setattr(H._registry, "registered_benchmark_documents", lambda: REGISTRY)
    cfg = str(world.source_file)
    assert H.main(["--paths", cfg, "build"]) == 0
    out = capsys.readouterr().out
    assert "Oxford_Bodleian_MS_heb_d_99_7 key=mshebd99_7 pgpids=['9002']" in out
    assert world.out(H.HOLDOUTS_FILE).exists()
    assert H.main(["--paths", cfg, "show"]) == 0
    assert "benchmark_pgpids" in capsys.readouterr().out
    assert H.main(["--paths", cfg, "check", "--shelfmark", "Bodl. MS heb. f 77/3"]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["benchmark"] == "key:pgp131"
    h = H.Holdouts.load(p=world)
    assert h.meta["build_args"] == {"no_extra": False, "out": None}
    assert h.meta["inputs"]["benchmark_ids"]["path"] == "data/benchmark_ids.json"


def test_pgp_parallel_dev_test_documents_without_v01_rows_join_the_eval_tier(tmp_path: Path) -> None:
    """Fix pass F1: a ``pgp_parallel`` dev / test document that v0_1 built no rows for is an eval document too, so
    literature naming it is kept out; ``pgp_parallel`` train documents are not.

    :param tmp_path: Pytest temporary directory.
    :returns: None.
    """
    p = make_world(tmp_path, pairs_extra=[
        {"pgpid": "9050", "shelfmark": "T-S 55J5.5", "merged_doc_ids": [], "held_out_vision_benchmark": False,
         "split": "test", "edition_text": "", "translation_en": None},
        {"pgpid": "9051", "shelfmark": "T-S 55J5.6", "merged_doc_ids": [], "held_out_vision_benchmark": False,
         "split": "train", "edition_text": "", "translation_en": None}])
    h = H.build_holdouts(p, registry=REGISTRY)
    assert h.is_eval_doc("9050") == "test" and h.is_eval_doc(shelfmark="T-S 55J5.5") == "key:test"
    assert h.is_eval_doc("9051") is None and h.is_benchmark("9050") is None
    # 9050 and the base world's PGP-131 pair row 9001 (split test, no v0_1 row; it stays in the benchmark tier)
    assert h.meta["counts"]["eval_pgpids_pgp_parallel_split_only"] == 2 and h.is_benchmark("9001") == "pgp131"
