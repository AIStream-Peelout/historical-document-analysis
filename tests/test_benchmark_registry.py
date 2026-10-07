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
