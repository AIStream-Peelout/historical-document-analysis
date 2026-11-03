"""T3 round-trip test: a raw mark, its canonical id and its ES id give one ``loose_key``; an ES id gives its own core.

``loose_key(mark) == loose_key(canonical_id(mark)) == loose_key(es_id)`` for P001 / P028 / P029 / P032 and the
Oxford probes (P088–P097), with the probe's institution and without one (institution inference, PR-2), and for the
side-suffix probes (one id per leaf, PR-2): ``loose_key("T-S 12.301 recto") == loose_key("Cambridge_CUL_T_S_12_301")``.
Every ES id, slug-token and ``Unknown_…`` ids included, keeps a non-empty loose key of its own, and ``canonical_id`` is
idempotent on its outputs. The single-sided ES ids (R3) are pinned as a known re-key.

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
#: Probes that name one side of a leaf: they share the leaf's loose key (side_suffix, PR-2).
SIDE_PROBES = ["P048", "P049", "P050", "P051", "P052", "P053", "P054", "P105"]


@pytest.mark.parametrize("pid", ROUND_TRIP + SIDE_PROBES)
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
    assert key
    assert key == shelfmarks.loose_key(shelfmarks.canonical_id(mark, institution))
    assert key == shelfmarks.loose_key(probe["expected_es_id"])


@pytest.mark.parametrize("pid", ROUND_TRIP + SIDE_PROBES)
def test_loose_key_round_trip_without_institution(pid: str) -> None:
    """The bare mark (institution inferred from the head inside it, a series word or a Bodleian parse) gives the ES
    id's loose key.

    :param pid: Probe id from the fixture.
    :type pid: str
    :return: None.
    :rtype: None
    """
    probe = PROBES[pid]
    key = shelfmarks.loose_key(probe["mark"])
    assert key
    assert key == shelfmarks.loose_key(shelfmarks.canonical_id(probe["mark"]))
    assert key == shelfmarks.loose_key(probe["expected_es_id"])


def test_side_suffix_round_trip() -> None:
    """A side of a leaf and the leaf's ES id share one loose key; the side does not enter the id.

    :return: None.
    :rtype: None
    """
    assert shelfmarks.loose_key("T-S 12.301 recto") == shelfmarks.loose_key("Cambridge_CUL_T_S_12_301")
    assert shelfmarks.loose_key("T-S 12.301 recto") == "cambridgecults12|301"
    assert shelfmarks.canonical_id("T-S 10J5.17, 1v") == "Cambridge_CUL_T_S_10J5_17"
    assert shelfmarks.loose_key("AIU XII.87v", "Alliance Israélite Universelle") == shelfmarks.loose_key(
        "Paris_AIU_XII_87")
    assert shelfmarks.loose_key("T-S Misc. 28.79, f. 14r") == shelfmarks.loose_key("Cambridge_CUL_T_S_Misc_28_79_14")


def test_institution_inference_rules() -> None:
    """Head inside the mark, then series words; a bare ``Or.`` names no institution; no institution gives None.

    :return: None.
    :rtype: None
    """
    assert shelfmarks.canonical_id("Cambridge University Library T-S AS 181.163") == "Cambridge_CUL_T_S_AS_181_163"
    assert shelfmarks.canonical_id("ENA NS 2.26") == "New_York_JTS_ENA_NS_2_26"
    assert shelfmarks.canonical_id("Or. 1080 1.85") == "Cambridge_CUL_Or_1080_1_85"
    assert shelfmarks.canonical_id("Or. 5566") is None and shelfmarks.loose_key("Or. 5566") == ""
    assert shelfmarks.canonical_id("BL Or. 5566C.9") == "London_BL_Or_5566C_9"
    assert shelfmarks.canonical_id("Paris Mosseri: Moss. IV10") == "Cambridge_Mosseri_IV10"
    assert shelfmarks.canonical_id("PER H 22") is None
    assert shelfmarks.canonical_id("PER H 22", "Austrian National Library") == "Vienna_ONB_PER_H_22"
    assert shelfmarks.canonical_id("Unknown_N_A") is None
    assert shelfmarks.canonical_id("987007606066605171") is None
    assert shelfmarks.canonical_id("T. S. 12.301") == "Cambridge_CUL_T_S_12_301"
    assert shelfmarks.canonical_id("T-S 12.301", "Unknown") == "Cambridge_CUL_T_S_12_301"  # names no institution
    assert shelfmarks.canonical_id("Unknown Library", "Unknown Library") is None
    # Lewis-Gibson under a Cambridge University Library KTIV head keeps the sub-collection token (as the ES does)
    assert shelfmarks.canonical_id("Cambridge University Library, Cambridge, England Ms. L-G Add. 3") == (
        "Cambridge_Lewis_Gibson_L_G_Add_3")


#: ``MS heb`` alone does not name Oxford: NLI / JTS marks with a ``Ms. Heb.`` series word, no institution given.
NOT_OXFORD = ["Ms. Heb. 24°10285", "MS Heb. 38°5702", "Ms. Heb. 4° 577/3", "Ms. heb. 28 (JTS)"]


@pytest.mark.parametrize("mark", NOT_OXFORD)
def test_ms_heb_alone_does_not_infer_oxford(mark: str) -> None:
    """Oxford is inferred only for a mark the engine parses as a Bodleian shelf mark (never with ``°``).

    :param mark: A non-Bodleian ``Ms. Heb.`` mark.
    :type mark: str
    :return: None.
    :rtype: None
    """
    assert shelfmarks.canonical_id(mark) is None
    assert shelfmarks.loose_key(mark) == ""
    assert shelfmarks.canonical_id("MS Heb. c. 13/25") == "Oxford_Bodleian_Bodl_MS_heb_c_13_25"


#: Bare cores and malformed marks written with underscores: not slug ids, so never echoed back.
NOT_SLUG_IDS = ["T_S_12_301", "ENA_2727_5", "A_2", "DK_12", "12_301", "T-S_12.301", "Mingana_12"]


@pytest.mark.parametrize("mark", NOT_SLUG_IDS)
def test_a_bare_core_is_not_taken_as_an_es_id(mark: str) -> None:
    """``canonical_id`` of an underscore core is ``None`` or an inferred registry-token id, never the input.

    :param mark: An underscore-joined core with no registry token.
    :type mark: str
    :return: None.
    :rtype: None
    """
    assert not shelfmarks._is_slug_id(mark)
    cid = shelfmarks.canonical_id(mark)
    assert cid != mark
    assert cid is None or shelfmarks.split_id(cid)[0] is not None
    assert shelfmarks.canonical_id("T-S_12.301") == "Cambridge_CUL_T_S_12_301"


#: Probes whose institution is inferred from the mark alone and gives the same id as with the probe's institution:
#: a ``T.-S.`` / ``T.S.`` spelling (P014, P017), a KTIV ``… Cod.`` head (P132), and an institution needle in the
#: digit-free prefix (P112, P120, P133, P137, P153).
INFERRED_AS_GIVEN = ["P014", "P017", "P112", "P120", "P132", "P133", "P137", "P153"]


@pytest.mark.parametrize("pid", INFERRED_AS_GIVEN)
def test_institution_inferred_from_the_mark(pid: str) -> None:
    """Without an institution the mark alone gives its reference id (the id it gives with its institution).

    :param pid: Probe id from the fixture.
    :type pid: str
    :return: None.
    :rtype: None
    """
    probe = PROBES[pid]
    cid = shelfmarks.canonical_id(probe["mark"])
    assert cid == shelfmarks.canonical_id(probe["mark"], probe["institution"]) == probe["REF_canonical_id"]
    if probe["expected_in_es"]:
        assert shelfmarks.loose_key(probe["mark"]) == shelfmarks.loose_key(probe["expected_es_id"])


def test_strasbourg_bare_mark_infers_the_es_id() -> None:
    """P129 ``Strasbourg 4038`` alone gives the ES id; with the probe's slug institution ``BNU`` it keeps the reference
    slug id (not a section 1.3 change).

    :return: None.
    :rtype: None
    """
    probe = PROBES["P129"]
    assert shelfmarks.canonical_id(probe["mark"]) == probe["expected_es_id"] == "Strasbourg_4038"
    assert shelfmarks.canonical_id(probe["mark"], probe["institution"]) == probe["REF_canonical_id"]


#: Digit-free marks that name a holding when an institution comes with them (merged-corpus records whose ES ids
#: these are): the garbage rule does not apply once an institution is given.
DIGIT_FREE_WITH_INSTITUTION = [
    ("MS Meunier", "Ben Zvi Institute", "Ben_Zvi_Institute_MS_Meunier"),
    ("Benayahu, Meir, Jerusalem, Israel", "Benayahu", "Benayahu_Meir_Jerusalem_Israel"),
    ("Unknown private collection", "Private Collection", "Private_Collection_private_collection"),
]


@pytest.mark.parametrize("mark,institution,es_id", DIGIT_FREE_WITH_INSTITUTION,
                         ids=[cid for _, _, cid in DIGIT_FREE_WITH_INSTITUTION])
def test_digit_free_mark_with_an_institution_keeps_its_id(mark: str, institution: str, es_id: str) -> None:
    """A digit-free mark with an institution is not garbage; without one it is.

    :param mark: Digit-free raw mark.
    :type mark: str
    :param institution: The record's institution spelling.
    :type institution: str
    :param es_id: The record's ES id.
    :type es_id: str
    :return: None.
    :rtype: None
    """
    assert shelfmarks.canonical_id(mark, institution) == es_id
    assert shelfmarks.loose_key(mark, institution) == shelfmarks.loose_key(es_id) != ""
    assert shelfmarks.canonical_id(es_id) == es_id  # a slug id comes back unchanged


@pytest.mark.parametrize("mark", ["N/A", "see above", "(shelfmark unknown)", "None", "———", "Item page", ""])
def test_placeholders_are_garbage_with_any_institution(mark: str) -> None:
    """A placeholder names no mark, even with an institution.

    :param mark: Placeholder.
    :type mark: str
    :return: None.
    :rtype: None
    """
    for institution in (None, "Cambridge University Library", "Item page"):
        assert shelfmarks.canonical_id(mark, institution) is None
        assert shelfmarks.loose_key(mark, institution) == ""


#: Ids without a registry token: slug-token ids (P146 / P147 / P129 reference ids) and ``Unknown_…`` ids.
SLUG_IDS = [PROBES[pid]["REF_canonical_id"] for pid in ("P129", "P146", "P147")] + [
    "Ben_Zvi_Institute_MS_Meunier", "Unknown_DK_123", "Unknown_Cod_Heid_Or_78"]


@pytest.mark.parametrize("cid", ES_IDS + SLUG_IDS)
def test_every_es_id_has_a_loose_key_of_its_own(cid: str) -> None:
    """Every ES id (registry-token or slug) keys non-empty; a slug id comes back unchanged, an ``Unknown_…`` id gives
    ``None`` (R7) but is still keyed on itself.

    :param cid: ES id.
    :type cid: str
    :return: None.
    :rtype: None
    """
    key = shelfmarks.loose_key(cid)
    assert key
    if cid.startswith("Unknown_"):
        assert shelfmarks.canonical_id(cid) is None
        assert key == shelfmarks.ShelfmarkNormalizer.loose_key(cid)
    else:
        assert shelfmarks.canonical_id(cid) == cid


@pytest.mark.parametrize("probe", list(PROBES.values()), ids=list(PROBES))
def test_canonical_id_is_idempotent_on_its_outputs(probe: Dict[str, Any]) -> None:
    """``canonical_id(canonical_id(m, inst)) == canonical_id(m, inst)``, with and without the probe's institution;
    no output is an ``Unknown_…`` id.

    :param probe: One fixture probe.
    :type probe: Dict[str, Any]
    :return: None.
    :rtype: None
    """
    for institution in (probe["institution"], None):
        cid = shelfmarks.canonical_id(probe["mark"], institution)
        if cid is None:
            continue
        assert not cid.startswith("Unknown")
        assert shelfmarks.canonical_id(cid) == cid
        assert shelfmarks.loose_key(cid) == shelfmarks.loose_key(probe["mark"], institution)


#: R3 known re-key (migration plan section 4): single-sided ES ids whose raw marks now give the side-less leaf
#: (``mark, institution, ES id, facade id``). The first three side-less ids are not ES ids: ``resolve()`` (PR-6) falls
#: back to the side-ful id until the ES re-key. The last two are the Tier-1 duplicate pair (the ES also holds the
#: side-less ``Cambridge_CUL_T_S_NS_225_25``).
R3_KNOWN_REKEY = [
    ("T-S NS J401r", "Cambridge University Library", "Cambridge_CUL_T_S_NS_J401r", "Cambridge_CUL_T_S_NS_J401"),
    ("Cambridge University Library, Cambridge, England Ms. T-S H 3.104r", "Cambridge University Library",
     "Cambridge_CUL_T_S_H3_104r", "Cambridge_CUL_T_S_H3_104"),
    ("Dār al-kutub inv. no. 404r", "Egyptian National Library (Dār al-Kutub)", "Cairo_ENL_Dār_al_kutub_inv_no_404r",
     "Cairo_ENL_Dār_al_kutub_inv_no_404"),
    ("T-S NS 225.25r", "Cambridge University Library", "Cambridge_CUL_T_S_NS_225_25r", "Cambridge_CUL_T_S_NS_225_25"),
    ("T-S NS 225.25v", "Cambridge University Library", "Cambridge_CUL_T_S_NS_225_25v", "Cambridge_CUL_T_S_NS_225_25"),
]


@pytest.mark.parametrize("mark,institution,es_id,facade_id", R3_KNOWN_REKEY, ids=[row[2] for row in R3_KNOWN_REKEY])
def test_single_sided_es_ids_are_a_known_rekey(mark: str, institution: str, es_id: str, facade_id: str) -> None:
    """Pinned: a single-sided ES id comes back unchanged, but its raw mark gives the side-less leaf.

    :param mark: The record's display mark.
    :type mark: str
    :param institution: The record's institution.
    :type institution: str
    :param es_id: The record's (side-ful) ES id.
    :type es_id: str
    :param facade_id: What the facade gives for the raw mark.
    :type facade_id: str
    :return: None.
    :rtype: None
    """
    assert shelfmarks.canonical_id(es_id) == es_id
    assert shelfmarks.canonical_id(mark, institution) == facade_id != es_id
    assert shelfmarks.loose_key(mark, institution) != shelfmarks.loose_key(es_id)
    assert shelfmarks.side_suffix(mark)[1] in ("r", "v")


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


#: PR-4 colon fix: a colon between two digits (no space) separates class-mark parts. Before the fix the engine read
#: it as a catalogue head, so these marks shared a key with their last number alone.
CLASS_COLON_PAIRS = [("TS 12:301", "TS 5:301"), ("T-S NS 327:140", "T-S NS 140"), ("T-S Ar.41:18", "T-S Ar.18")]


@pytest.mark.parametrize("mark,other", CLASS_COLON_PAIRS, ids=[m for m, _ in CLASS_COLON_PAIRS])
def test_class_colon_round_trip_and_distinct(mark: str, other: str) -> None:
    """A digit-colon mark, its canonical id and its dotted form give one key; another item gives another key.

    :param mark: A mark with a digit-colon-digit separator.
    :type mark: str
    :param other: A different item that the old head reading merged with *mark*.
    :type other: str
    :return: None.
    :rtype: None
    """
    cid = shelfmarks.canonical_id(mark)
    assert cid is not None
    assert shelfmarks.loose_key(mark) == shelfmarks.loose_key(cid) == shelfmarks.loose_key(mark.replace(":", "."))
    assert shelfmarks.canonical_id(cid) == cid and shelfmarks.core_id(cid) == shelfmarks.core_id(mark)
    assert shelfmarks.loose_key(mark) != shelfmarks.loose_key(other)
    assert shelfmarks.core_key(mark) != shelfmarks.core_key(other)
