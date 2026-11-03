"""T1 golden test: every ``src.datasets.shelfmarks`` function against its reference column, per probe.

The fixture ``tests/fixtures/shelfmark_probes.json`` holds the 170 audit probes with the reference columns
(``REF_*``). In PR-1 the facade reproduces them byte for byte; the only departure is the id-input rule (an ES id
passed as the mark gives its own core), allowed for the probes in :data:`ID_INPUT_PROBES` only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pytest

from src.datasets import shelfmarks
from src.datasets.merging.institution_tokens import _REGISTRY_TOKENS, combine

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DATA: Dict[str, Any] = json.loads((FIXTURES / "shelfmark_probes.json").read_text(encoding="utf-8"))
PROBES: List[Dict[str, Any]] = DATA["probes"]
ES_IDS: Set[str] = set((FIXTURES / "es_id_sample.txt").read_text(encoding="utf-8").split())
KNOWN = frozenset(DATA["cudl_known_fragments"])

#: Probes whose mark already is an ES id (PR-1 id-input rule): ``core_id`` / ``core_key`` depart from the reference.
#: P032 is an id the ES holds; P097 is a pre-fix Oxford id form (re-parsing it to the current id is PR-2).
ID_INPUT_PROBES = {"P032": True, "P097": False}


def _ids(probes: List[Dict[str, Any]]) -> List[str]:
    """Probe ids for pytest parametrisation.

    :param probes: Probe dictionaries.
    :type probes: List[Dict[str, Any]]
    :return: The probe ids.
    :rtype: List[str]
    """
    return [p["id"] for p in probes]


def _assert_id_input(probe: Dict[str, Any], core: str) -> None:
    """Check an allowed id-input departure: the core is the id's own core, and the id is accounted for.

    :param probe: The probe.
    :type probe: Dict[str, Any]
    :param core: The facade's ``core_id`` of the probe mark.
    :type core: str
    :return: None.
    :rtype: None
    """
    token, split_core = shelfmarks.split_id(probe["mark"])
    assert token is not None and core == split_core
    assert combine(token, core) == probe["mark"]
    assert probe["expected_es_id"] in ES_IDS
    assert (probe["mark"] in ES_IDS) is ID_INPUT_PROBES[probe["id"]]


def test_fixture_shape() -> None:
    """The fixture carries all 170 probes and the 56 ES ids they name.

    :return: None.
    :rtype: None
    """
    assert len(PROBES) == 170 and len({p["id"] for p in PROBES}) == 170
    assert ES_IDS == {p["expected_es_id"] for p in PROBES if p["expected_in_es"]}
    assert shelfmarks.NORMALISER_VERSION == "2026.11.03-1"


def test_id_input_probes_are_exactly_the_es_id_inputs() -> None:
    """The id-input rule fires on P032 and P097 and on no other probe.

    :return: None.
    :rtype: None
    """
    fired = {p["id"] for p in PROBES if shelfmarks._id_input_core(p["mark"]) is not None}
    assert fired == set(ID_INPUT_PROBES)


@pytest.mark.parametrize("probe", PROBES, ids=_ids(PROBES))
def test_core_id(probe: Dict[str, Any]) -> None:
    """``core_id`` == ``REF_core`` (id-input probes: the id's own core).

    :param probe: One fixture probe (id, family, mark, institution, ``REF_*`` columns, expected ES id).
    :type probe: Dict[str, Any]
    :return: None.
    :rtype: None
    """
    got = shelfmarks.core_id(probe["mark"])
    if probe["id"] in ID_INPUT_PROBES:
        assert got != probe["REF_core"]
        _assert_id_input(probe, got)
    else:
        assert got == probe["REF_core"]


@pytest.mark.parametrize("probe", PROBES, ids=_ids(PROBES))
def test_canonical_id(probe: Dict[str, Any]) -> None:
    """``canonical_id(mark, institution)`` == ``REF_canonical_id`` on every probe.

    :param probe: One fixture probe (id, family, mark, institution, ``REF_*`` columns, expected ES id).
    :type probe: Dict[str, Any]
    :return: None.
    :rtype: None
    """
    assert shelfmarks.canonical_id(probe["mark"], probe["institution"]) == probe["REF_canonical_id"]


@pytest.mark.parametrize("probe", PROBES, ids=_ids(PROBES))
def test_loose_key(probe: Dict[str, Any]) -> None:
    """``loose_key(mark, institution)`` == ``REF_loose_of_cid`` on every probe.

    :param probe: One fixture probe (id, family, mark, institution, ``REF_*`` columns, expected ES id).
    :type probe: Dict[str, Any]
    :return: None.
    :rtype: None
    """
    assert shelfmarks.loose_key(probe["mark"], probe["institution"]) == probe["REF_loose_of_cid"]


@pytest.mark.parametrize("probe", PROBES, ids=_ids(PROBES))
def test_core_key(probe: Dict[str, Any]) -> None:
    """``core_key`` == ``REF_loose_key`` (id-input probes: the key of the id's own core).

    :param probe: One fixture probe (id, family, mark, institution, ``REF_*`` columns, expected ES id).
    :type probe: Dict[str, Any]
    :return: None.
    :rtype: None
    """
    got = shelfmarks.core_key(probe["mark"])
    if probe["id"] in ID_INPUT_PROBES:
        assert got != probe["REF_loose_key"]
        _assert_id_input(probe, shelfmarks.core_id(probe["mark"]))
    else:
        assert got == probe["REF_loose_key"]


@pytest.mark.parametrize("probe", PROBES, ids=_ids(PROBES))
def test_split_id(probe: Dict[str, Any]) -> None:
    """``split_id(REF_canonical_id)`` gives ``REF_token`` when that is a registry token, else ``None``.

    :param probe: One fixture probe (id, family, mark, institution, ``REF_*`` columns, expected ES id).
    :type probe: Dict[str, Any]
    :return: None.
    :rtype: None
    """
    token, core = shelfmarks.split_id(probe["REF_canonical_id"])
    expected: Optional[str] = probe["REF_token"] if probe["REF_token"] in _REGISTRY_TOKENS else None
    assert token == expected
    if token is None:
        assert core == probe["REF_canonical_id"]
    else:
        assert combine(token, core) == probe["REF_canonical_id"]


@pytest.mark.parametrize("probe", PROBES, ids=_ids(PROBES))
def test_cudl_manifest(probe: Dict[str, Any]) -> None:
    """``cudl_manifest`` == ``REF_cudl`` (with the PGP fragments that matter for these probes).

    :param probe: One fixture probe (id, family, mark, institution, ``REF_*`` columns, expected ES id).
    :type probe: Dict[str, Any]
    :return: None.
    :rtype: None
    """
    assert shelfmarks.cudl_manifest(probe["mark"], KNOWN) == probe["REF_cudl"]


def test_cudl_manifest_returns_the_parent_manifest_for_a_sub_numbered_child() -> None:
    """Documented hazard: P013 (child .14) shares its parent's manifest, so it is not an identity key.

    :return: None.
    :rtype: None
    """
    child = next(p for p in PROBES if p["id"] == "P013")
    parent = next(p for p in PROBES if p["id"] == "P010")
    assert shelfmarks.cudl_manifest(child["mark"], KNOWN) == shelfmarks.cudl_manifest(parent["mark"], KNOWN)


@pytest.mark.parametrize("probe", PROBES, ids=_ids(PROBES))
def test_cudl_manifest_is_the_scrape_queue_function(probe: Dict[str, Any]) -> None:
    """The facade's ``cudl_manifest`` is a thin re-export of ``arabic_scrape_queue.cudl_manifest`` (moves in PR-6).

    :param probe: One fixture probe (id, family, mark, institution, ``REF_*`` columns, expected ES id).
    :type probe: Dict[str, Any]
    :return: None.
    :rtype: None
    """
    from src.datasets.indexing import arabic_scrape_queue

    assert shelfmarks.cudl_manifest(probe["mark"], KNOWN) == arabic_scrape_queue.cudl_manifest(probe["mark"], KNOWN)


@pytest.mark.parametrize("probe", PROBES, ids=_ids(PROBES))
def test_is_shelfmark_and_institution_info_wrap_the_engine(probe: Dict[str, Any]) -> None:
    """``is_shelfmark`` / ``institution_info`` are the engine's validator and taxonomy.

    :param probe: One fixture probe (id, family, mark, institution, ``REF_*`` columns, expected ES id).
    :type probe: Dict[str, Any]
    :return: None.
    :rtype: None
    """
    from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer

    mark = probe["mark"]
    assert shelfmarks.is_shelfmark(mark) == (ShelfmarkNormalizer.classify_shelfmark(mark) in ("standard", "berlin"))
    assert shelfmarks.institution_info(mark) == ShelfmarkNormalizer.get_institution_info(mark)


def test_garbage_is_not_a_shelfmark() -> None:
    """Every garbage probe fails the validator.

    :return: None.
    :rtype: None
    """
    assert not any(shelfmarks.is_shelfmark(p["mark"]) for p in PROBES if p["family"] == "garbage")
