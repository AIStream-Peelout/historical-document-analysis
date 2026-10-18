# File name: test_benchmark_registry.py
# Date: 10/2/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Tests for the registry of held-out benchmark documents."""
import json

import pytest

from src.datasets.evaluations import benchmark_registry as reg


def test_write_and_read_a_registry(tmp_path):
    reg.write_registry(tmp_path / "arabic_v0.json", "arabic_v0", ["B_2", "A_1"], ["10", 9, "10"], "2026-10-02")
    stored = json.loads((tmp_path / "arabic_v0.json").read_text(encoding="utf-8"))
    assert stored == {"benchmark": "arabic_v0", "built": "2026-10-02", "ids": ["A_1", "B_2"], "pgpids": ["9", "10"]}
    (tmp_path / "other.json").write_text(json.dumps({"ids": ["C_3"]}), encoding="utf-8")          # a registry without PGP ids
    assert reg.registered_benchmark_documents(["arabic_v0", "other"], tmp_path) == ({"A_1", "B_2", "C_3"}, {"9", "10"})
    assert reg.registered_benchmark_documents([], tmp_path) == (set(), set())


def test_a_listed_registry_that_is_missing_stops_the_caller(tmp_path):
    with pytest.raises(FileNotFoundError):
        reg.registered_benchmark_documents(["not_there"], tmp_path)


def test_the_real_registries_exist_and_hold_documents():
    ids, pgpids = reg.registered_benchmark_documents()
    assert len(ids) >= 100 and len(pgpids) >= 100 and all(pid.isdigit() for pid in pgpids)


def test_manuscripts_round_trip_and_the_key_is_optional(tmp_path):
    reg.write_registry(tmp_path / "lit.json", "lit", ["KTIV_990000000000000002_FL1"], [], "2026-10-12",
                       sys_nums=["990000000000000002", "990000000000000001"])
    stored = json.loads((tmp_path / "lit.json").read_text(encoding="utf-8"))
    assert stored["sys_nums"] == ["990000000000000001", "990000000000000002"] and stored["pgpids"] == []
    reg.write_registry(tmp_path / "doc.json", "doc", ["A_1"], ["10"], "2026-10-12")
    assert "sys_nums" not in json.loads((tmp_path / "doc.json").read_text(encoding="utf-8"))
    assert reg.registered_benchmark_manuscripts(["lit", "doc"], tmp_path) == {"990000000000000001", "990000000000000002"}
    assert reg.registered_benchmark_documents(["lit"], tmp_path) == ({"KTIV_990000000000000002_FL1"}, set())
    with pytest.raises(FileNotFoundError):
        reg.registered_benchmark_manuscripts(["not_there"], tmp_path)


def test_a_registry_can_live_next_to_its_benchmark(tmp_path, monkeypatch):
    monkeypatch.setitem(reg.REGISTRY_LOCATIONS, "elsewhere_v0", tmp_path)
    reg.write_registry(tmp_path / "elsewhere_v0.json", "elsewhere_v0", ["X"], ["7"], "2026-10-12",
                       sys_nums=["990000000000000003"])
    assert reg.registry_path("elsewhere_v0") == tmp_path / "elsewhere_v0.json"
    assert reg.registry_path("elsewhere_v0", tmp_path / "other") == tmp_path / "other" / "elsewhere_v0.json"
    assert reg.registry_path("arabic_script_benchmark_v0") == reg.REGISTRY_DIR / "arabic_script_benchmark_v0.json"
    assert reg.registered_benchmark_documents(["elsewhere_v0"]) == ({"X"}, {"7"})
    assert reg.registered_benchmark_manuscripts(["elsewhere_v0"]) == {"990000000000000003"}


def test_the_sealed_set_is_registered_with_both_halves():
    assert "sealed_v1" in reg.BENCHMARK_REGISTRIES
    ids, pgpids = reg.registered_benchmark_documents(["sealed_v1"])
    sys_nums = reg.registered_benchmark_manuscripts(["sealed_v1"])
    literary = {i for i in ids if i.startswith("KTIV_")}
    assert len(literary) >= 100 and len(ids - literary) >= 50 and len(pgpids) >= 50 and len(sys_nums) >= 100
    assert {i.split("_")[1] for i in literary} <= sys_nums
    assert reg.registered_benchmark_manuscripts() >= sys_nums
