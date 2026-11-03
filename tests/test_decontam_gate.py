# File name: test_decontam_gate.py
# Date: 11/3/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for the decontamination gate's shelf-mark layer (layer 2: ``clean_v2`` / benchmark loose keys).

Until PR-4 the gate keyed a raw mark with its institution stripped (``T-S 12.301`` -> ``ts12|301``) while the
``clean_v2_keys`` inventory is institution-qualified (``cambridgecults12|301``), so the layer never fired on a raw
mark (audit finding F4). The gate now keys with the facade ``loose_key``.
"""
from __future__ import annotations

from typing import Iterable

import pytest

from src.datasets import shelfmarks
from src.datasets.evaluations.helper_eval_scripts import decontam_gate as dg
from src.datasets.evaluations.helper_eval_scripts.decontam_gate import DecontamGate


def fake_gate(clean_ids: Iterable[str] = (), bench_ids: Iterable[str] = ()) -> DecontamGate:
    """A gate whose inventories hold the loose keys of some ES ids (no files read).

    :param clean_ids: ES ids of the ``clean_v2`` training set.
    :type clean_ids: Iterable[str]
    :param bench_ids: ES ids of the benchmark.
    :type bench_ids: Iterable[str]
    :return: The gate.
    :rtype: DecontamGate
    """
    gate = DecontamGate.__new__(DecontamGate)
    gate.ktiv_train_sys = {"990000000000000001"}
    gate.clean_keys = {shelfmarks.loose_key(i) for i in clean_ids}
    gate.bench_keys = {shelfmarks.loose_key(i) for i in bench_ids}
    return gate


def test_raw_mark_hits_its_es_id_in_clean_v2() -> None:
    """A raw ``T-S 12.301`` (any spelling, KTIV head included) whose ES id is in ``clean_v2_keys`` is rejected."""
    gate = fake_gate(clean_ids=["Cambridge_CUL_T_S_12_301"])
    assert gate.clean_keys == {"cambridgecults12|301"}
    for mark in ("T-S 12.301", "TS 12.301", "T-S 12:301", "Cambridge_CUL_T_S_12_301",
                 "Cambridge University Library, Cambridge, England Ms. T-S 12.301"):
        assert gate.check("", mark) == (False, "shelfmark_in_clean_v2 (cambridgecults12|301)"), mark
    assert gate.check("", "T-S 12.3011") == (True, "clean")
    assert gate.check("", "ENA 12.301") == (True, "clean")             # another institution's 12.301


def test_benchmark_layer_and_sys_num_layer() -> None:
    """The benchmark keys and the KTIV sys_num list still reject; an unkeyable mark falls through to clean."""
    gate = fake_gate(bench_ids=["New_York_JTS_ENA_2557_1"])
    assert gate.check("", "ENA 2557.1") == (False, "shelfmark_in_benchmark (newyorkjtsena2557|1)")
    assert gate.check("990000000000000001", "ENA 2557.1") == (False, "sysnum_in_ktiv_v19a_train")
    assert gate.check("", "") == (True, "clean")
    assert DecontamGate.loose_key("Or. 5566") == ""                 # institution undetermined: no key


def test_gate_key_is_the_facade_key() -> None:
    """``DecontamGate.loose_key`` is the facade's ``loose_key`` (the gate defines no key of its own)."""
    assert DecontamGate.loose_key is shelfmarks.loose_key
    assert fake_gate().loose_key("T-S 12.301") == shelfmarks.loose_key("Cambridge_CUL_T_S_12_301")


@pytest.mark.skipif(not (dg._DECONTAM / "clean_v2_keys.json").exists(), reason="decontam inventories absent")
def test_real_inventory_rejects_a_raw_clean_v2_mark() -> None:
    """On the real inventories a raw KTIV mark of a clean_v2 fragment is rejected (it was passed before PR-4)."""
    gate = DecontamGate()
    assert "cambridgecults16|1" in gate.clean_keys
    clean, reason = gate.check("", "Cambridge University Library, Cambridge, England Ms. T-S 16.1")
    assert not clean and reason.startswith("shelfmark_in_clean_v2")
