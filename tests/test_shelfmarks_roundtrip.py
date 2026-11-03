"""T3 round-trip test: a raw mark, its canonical id and its ES id give one ``loose_key``; an ES id gives its own core.

``loose_key(mark) == loose_key(canonical_id(mark)) == loose_key(es_id)`` for P001 / P028 / P029 / P032 and the
Oxford probes (P088–P097). With the probe's institution this holds today except for P097 (a pre-fix Oxford id,
re-parsed in PR-2). Without an institution it needs institution inference, which is PR-2; those cases are strict
xfails so they flag themselves once PR-2 lands.

The id-input rule (§1.3 of the migration plan): an ES id comes back from ``canonical_id`` unchanged when no
institution is given or the institution's token is the id's own token. The engine alone does not guarantee that
(it re-normalises the core: every ``__sys`` id loses an underscore), so the facade returns the id itself. One
exception to "an ES id has a non-empty core": the bare multi-word token id ``Cambridge_Lewis_Gibson`` has core ``""``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.datasets import shelfmarks
from src.datasets.merging import institution_tokens
from src.datasets.merging.institution_tokens import SERIES_TOKENS, _REGISTRY_TOKENS, combine

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PROBES: Dict[str, Dict[str, Any]] = {
    p["id"]: p for p in json.loads((FIXTURES / "shelfmark_probes.json").read_text(encoding="utf-8"))["probes"]
}
ES_IDS: List[str] = sorted(set((FIXTURES / "es_id_sample.txt").read_text(encoding="utf-8").split()))

ROUND_TRIP = ["P001", "P028", "P029", "P032"] + [f"P{n:03d}" for n in range(88, 98)]
#: Probes that do not round-trip yet even with their institution (fixed by the PR-2 facade pre-pass).
PR2_WITH_INSTITUTION = {"P097"}
#: Probes that round-trip without an institution today (the id-input rule); the rest need PR-2 inference.
NO_INSTITUTION_OK = {"P032"}


def _case(pid: str, pr2: bool) -> Any:
    """A parametrisation case, a strict ``PR-2`` xfail when *pr2* is true.

    :param pid: Probe id.
    :type pid: str
    :param pr2: Whether the case is expected to pass only after PR-2.
    :type pr2: bool
    :return: A ``pytest.param``.
    :rtype: Any
    """
    marks = [pytest.mark.xfail(reason="PR-2", strict=True)] if pr2 else []
    return pytest.param(pid, id=pid, marks=marks)


@pytest.mark.parametrize("pid", [_case(p, p in PR2_WITH_INSTITUTION) for p in ROUND_TRIP])
def test_loose_key_round_trip_with_institution(pid: str) -> None:
    """Mark + institution, its canonical id, and the expected ES id share one loose key.

    :param pid: Probe id from the fixture.
    :type pid: str
    :return: None.
    :rtype: None
    """
    probe = PROBES[pid]
    mark, institution = probe["mark"], probe["institution"]
    key = shelfmarks.loose_key(mark, institution)
    assert key == shelfmarks.loose_key(shelfmarks.canonical_id(mark, institution))
    assert key == shelfmarks.loose_key(probe["expected_es_id"])


@pytest.mark.parametrize("pid", [_case(p, p not in NO_INSTITUTION_OK) for p in ROUND_TRIP])
def test_loose_key_round_trip_without_institution(pid: str) -> None:
    """The bare mark (institution inferred) gives the ES id's loose key: PR-2 except for an ES-id input.

    :param pid: Probe id from the fixture.
    :type pid: str
    :return: None.
    :rtype: None
    """
    probe = PROBES[pid]
    key = shelfmarks.loose_key(probe["mark"])
    assert key == shelfmarks.loose_key(shelfmarks.canonical_id(probe["mark"]))
    assert key == shelfmarks.loose_key(probe["expected_es_id"])


@pytest.mark.parametrize("es_id", ES_IDS)
def test_es_id_gives_its_own_core_and_itself(es_id: str) -> None:
    """``core_id(es_id) == split_id(es_id)[1]``; ``canonical_id(es_id)`` is the id unchanged.

    :param es_id: An ES id from ``tests/fixtures/es_id_sample.txt``.
    :type es_id: str
    :return: None.
    :rtype: None
    """
    token, core = shelfmarks.split_id(es_id)
    assert token is not None
    assert shelfmarks.core_id(es_id) == core
    assert shelfmarks.canonical_id(es_id) == es_id
    # with its own token as the institution the facade returns the id unchanged (the engine does not always:
    # 306 ``__sys`` ids and the two Budapest_Rabbinical_Seminary ids whose token is not idempotent)
    assert shelfmarks.canonical_id(es_id, token) == es_id
    assert shelfmarks.loose_key(es_id) == shelfmarks.loose_key(es_id, token)


@pytest.mark.parametrize("token", _REGISTRY_TOKENS)
def test_split_id_round_trips_every_registry_token(token: str) -> None:
    """Every registry token is recovered by the longest-prefix split, alone and with a core.

    :param token: A registry token of ``institution_tokens``.
    :type token: str
    :return: None.
    :rtype: None
    """
    assert shelfmarks.split_id(token) == (token, "")
    assert shelfmarks.split_id(token + "_T_S_12_301") == (token, "T_S_12_301")
    assert combine(token, "Q_12_301") == token + "_Q_12_301"


def test_split_id_without_a_registry_token() -> None:
    """Slug, ``Unknown_*`` and bare numeric ids carry no registry token.

    :return: None.
    :rtype: None
    """
    for cid in ("Unknown_N_A", "Russian_State_Library_Guenzburg_1181", "987007606066605171", "Cambridgeshire_1"):
        assert shelfmarks.split_id(cid) == (None, cid)


def test_series_tokens_name_registry_tokens() -> None:
    """Every ``SERIES_TOKENS`` value is a registry token, and a bare ``Or.`` names no institution.

    :return: None.
    :rtype: None
    """
    assert all(v is None or v in _REGISTRY_TOKENS for v in SERIES_TOKENS.values())
    assert SERIES_TOKENS["Or."] is None
    assert SERIES_TOKENS["T-S"] == "Cambridge_CUL" and SERIES_TOKENS["ENA"] == "New_York_JTS"


#: ES ids the engine does not round-trip with their own institution (a ``__sys`` suffix), with an institution spelling
#: whose token is the id's own token.
SYS_IDS = [
    ("London_BL_Add_27002__sys990000659290205171", "British Library"),
    ("Oxford_Bodleian_Bodl_MS_heb_a_3__sys990001648120205171", "Bodleian Library"),
]


@pytest.mark.parametrize("es_id,institution", SYS_IDS, ids=[cid for cid, _ in SYS_IDS])
def test_es_id_with_its_own_institution_comes_back_unchanged(es_id: str, institution: str) -> None:
    """A ``__sys`` id with a matching institution is returned as is, although the engine would change it.

    :param es_id: An ES id with a ``__sys`` suffix.
    :type es_id: str
    :param institution: An institution spelling whose token is the id's token.
    :type institution: str
    :return: None.
    :rtype: None
    """
    token, core = shelfmarks.split_id(es_id)
    assert institution_tokens.institution_token(institution) == token
    assert institution_tokens.canonical_id(institution, es_id) != es_id  # the engine alone drops an underscore
    assert shelfmarks.canonical_id(es_id, institution) == es_id
    assert shelfmarks.canonical_id(es_id) == es_id
    assert shelfmarks.core_id(es_id) == core and "__sys" in core
    assert shelfmarks.loose_key(es_id, institution) == shelfmarks.loose_key(es_id)


def test_bare_token_id_has_an_empty_core_and_comes_back_unchanged() -> None:
    """``Cambridge_Lewis_Gibson`` (an ES id that is a registry token alone) has core ``""``; documented exception.

    A one-word token alone (``Geneva``) is not taken as an ES id: it is left to the engine.

    :return: None.
    :rtype: None
    """
    cid = "Cambridge_Lewis_Gibson"
    assert shelfmarks.split_id(cid) == (cid, "")
    assert shelfmarks.core_id(cid) == ""
    assert shelfmarks.canonical_id(cid) == cid
    assert shelfmarks.canonical_id(cid, "Lewis-Gibson") == cid
    assert shelfmarks.canonical_id(cid, "Jewish Theological Seminary") == institution_tokens.canonical_id(
        "Jewish Theological Seminary", cid)
    assert shelfmarks._id_input_core("Geneva") is None
    assert shelfmarks.core_id("Geneva") == shelfmarks.ShelfmarkNormalizer.to_canonical_id("Geneva")


def test_es_id_with_a_conflicting_institution_defers_to_the_engine() -> None:
    """Intended PR-1 behaviour: the id-input rule applies without an institution or with the id's own institution.

    A conflicting institution is not silently ignored: the engine's output stands, exactly as before the facade.
    Without an institution the ES id comes back unchanged (the engine would give ``Unknown_<id>``).

    :return: None.
    :rtype: None
    """
    es_id = "Cambridge_CUL_T_S_12_301"
    for institution in ("Jewish Theological Seminary", "any institution"):
        expected = institution_tokens.canonical_id(institution, es_id)
        assert expected != es_id
        assert shelfmarks.canonical_id(es_id, institution) == expected
    assert shelfmarks.canonical_id(es_id, "Cambridge University Library") == es_id
    assert shelfmarks.canonical_id(es_id) == es_id != institution_tokens.canonical_id(None, es_id)
