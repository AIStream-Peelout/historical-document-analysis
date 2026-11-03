"""T4 snapshot of the FROZEN text27b shelf-mark keys (shelf-mark migration plan, PR-6).

``src/finetuning/text27b/shelfmarks.py`` ``keys`` / ``key`` / ``core_key`` / ``compact`` / ``canonical_to_key`` are the
key format of the t27b hold-outs and mixtures, and ``core_of`` / ``token_of`` stay with them (no byte-identical facade
function); ``prep_shelfmark`` now lives in the facade and is re-imported. Their outputs on every probe mark and
expected ES id of ``tests/fixtures/shelfmark_probes.json`` were captured before PR-6 in
``tests/fixtures/text27b_shelfmarks_frozen.json`` and must not drift.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.datasets import shelfmarks as F
from src.finetuning.text27b import shelfmarks as SM

SNAPSHOT = Path(__file__).resolve().parent / "fixtures" / "text27b_shelfmarks_frozen.json"
ROWS: List[Dict[str, Any]] = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["rows"]
FUNCTIONS = ("keys", "key", "canonical_to_key", "core_key", "compact", "prep_shelfmark", "core_of", "token_of")


@pytest.mark.parametrize("fn", FUNCTIONS)
def test_frozen_function_matches_snapshot(fn: str) -> None:
    """Each frozen function gives its snapshot value on every probe input.

    :param fn: Function name in ``text27b.shelfmarks``.
    :type fn: str
    """
    drift = [(r["id"], r["value"], r[fn], getattr(SM, fn)(r["value"])) for r in ROWS if getattr(SM, fn)(r["value"]) != r[fn]]
    assert not drift, drift[:5]


def test_snapshot_covers_every_probe() -> None:
    """The snapshot has one row per non-empty probe mark and per expected ES id (309 inputs)."""
    probes = json.loads((SNAPSHOT.parent / "shelfmark_probes.json").read_text(encoding="utf-8"))["probes"]
    assert len(ROWS) == sum(bool(p["mark"]) + bool(p["expected_es_id"]) for p in probes) == 309


def test_core_of_and_token_of_have_no_byte_identical_facade_twin() -> None:
    """Why ``core_of`` / ``token_of`` stay frozen: the facade differs on many probe inputs (measured for PR-6)."""
    core_diff = sum(1 for r in ROWS if F.core_id(r["value"]) != r["core_of"])
    token_diff = sum(1 for r in ROWS if F.split_id(F.canonical_id(r["value"]) or "")[0] != r["token_of"])
    assert (core_diff, token_diff) == (153, 38)
