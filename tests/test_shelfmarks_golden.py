"""T1 golden test: every ``src.datasets.shelfmarks`` function against its reference column, per probe.

The fixture ``tests/fixtures/shelfmark_probes.json`` holds the 170 audit probes with the reference columns
(``REF_*``). The facade reproduces them byte for byte except for the probes in :data:`INTENDED_CHANGES`, the
departures of section 1.3 of the migration plan (PR-1: the id-input rule; PR-2: sides, the glued-prefix /
pre-fix-Oxford / spaced-head pre-pass, series folds, ``None`` for joins, NLI system numbers and garbage). Every
intended ``canonical_id`` is an id of ``tests/fixtures/es_id_sample.txt`` or ``None``. The new PR-2 functions
(``split``, ``side_suffix``, ``variants``, ``parent``) have golden tables of their own below.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

from src.datasets import shelfmarks
from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer
from src.datasets.merging.institution_tokens import _REGISTRY_TOKENS, combine
from src.finetuning.text27b import lit_segment

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DATA: Dict[str, Any] = json.loads((FIXTURES / "shelfmark_probes.json").read_text(encoding="utf-8"))
PROBES: List[Dict[str, Any]] = DATA["probes"]
BY_ID: Dict[str, Dict[str, Any]] = {p["id"]: p for p in PROBES}
ES_IDS: Set[str] = set((FIXTURES / "es_id_sample.txt").read_text(encoding="utf-8").split())
KNOWN = frozenset(DATA["cudl_known_fragments"])

#: Probes whose mark already is an ES id (PR-1 id-input rule). P032 is an id the ES holds; P097 is a pre-fix
#: Oxford id form, re-parsed to the current id in PR-2.
ID_INPUT_PROBES = {"P032": True, "P097": False}

_CUL_12_301 = "Cambridge_CUL_T_S_12_301"
_CUL_10J5_17 = "Cambridge_CUL_T_S_10J5_17"
_OXFORD_C_13_25 = "Oxford_Bodleian_Bodl_MS_heb_c_13_25"
_JOINS = ("P154", "P155", "P157", "P158")
_NLI_NUMBERS = ("P150", "P151")
_GARBAGE = tuple(f"P{n}" for n in range(161, 171))

#: Section 1.3 of the migration plan: probe id -> function -> new value. The only allowed departures from the
#: reference columns. ``loose_key`` / ``core_key`` follow from these by definition (:func:`_intended`).
INTENDED_CHANGES: Dict[str, Dict[str, Optional[str]]] = {
    # R3 sides: one id per leaf, the side is a separate value (side_suffix)
    **{pid: {"canonical_id": _CUL_12_301} for pid in ("P048", "P049", "P050", "P051", "P052")},
    "P053": {"canonical_id": _CUL_10J5_17},
    "P054": {"canonical_id": _CUL_10J5_17},
    "P105": {"canonical_id": "Paris_AIU_XII_87"},
    # R2 glued prefix
    "P041": {"canonical_id": _CUL_12_301},
    # R11 pre-fix Oxford id (and the PR-1 id-input rule for core_id)
    "P097": {"canonical_id": _OXFORD_C_13_25, "core_id": "Bodl_MS_heb_c_13_25"},
    # PR-1 id-input rule: an ES id gives its own core
    "P032": {"core_id": "T_S_12_301"},
    # R5 spaced head stripped in the pre-pass
    "P031": {"canonical_id": "Cambridge_CUL_T_S_AS_181_163"},
    # R4 series folds
    "P016": {"canonical_id": _CUL_12_301},
    "P026": {"canonical_id": "Cambridge_CUL_T_S_Ar_53_70"},
    "P027": {"canonical_id": "Cambridge_CUL_T_S_Ar_53_70"},
    "P073": {"canonical_id": "Cambridge_CUL_Or_1080_1_85"},
    "P083": {"canonical_id": "Cambridge_Lewis_Gibson_L_G_Ar_I_49"},
    "P086": {"canonical_id": "Cambridge_Lewis_Gibson_L_G_Misc_10"},
    # R1 / R10 joins, R7 NLI system numbers and garbage: None (and an empty core)
    **{pid: {"canonical_id": None, "core_id": ""} for pid in _JOINS + _NLI_NUMBERS + _GARBAGE},
}

#: Function -> the fixture's reference column.
REFERENCE_COLUMNS = {"canonical_id": "REF_canonical_id", "loose_key": "REF_loose_of_cid", "core_id": "REF_core",
                     "core_key": "REF_loose_key"}


def _intended(pid: str, function: str) -> Tuple[bool, Optional[str]]:
    """The intended value of *function* on probe *pid*, if the probe is an intended change for it.

    ``loose_key`` is ``ShelfmarkNormalizer.loose_key`` of the intended ``canonical_id`` (``""`` for ``None``) and
    ``core_key`` that of the intended ``core_id``.

    :param pid: Probe id.
    :type pid: str
    :param function: One of :data:`REFERENCE_COLUMNS`.
    :type function: str
    :return: ``(listed, value)``; ``(False, None)`` when the probe keeps its reference value.
    :rtype: Tuple[bool, Optional[str]]
    """
    row = INTENDED_CHANGES.get(pid, {})
    source = {"loose_key": "canonical_id", "core_key": "core_id"}.get(function, function)
    if source not in row:
        return False, None
    value = row[source]
    if function in ("loose_key", "core_key"):
        return True, ShelfmarkNormalizer.loose_key(value) if value else ""
    return True, value


def _golden(probe: Dict[str, Any], function: str) -> Optional[str]:
    """The value *function* must give on *probe*: the intended value if listed, else the reference column.

    :param probe: One fixture probe.
    :type probe: Dict[str, Any]
    :param function: One of :data:`REFERENCE_COLUMNS`.
    :type function: str
    :return: The golden value.
    :rtype: Optional[str]
    """
    listed, value = _intended(probe["id"], function)
    return value if listed else probe[REFERENCE_COLUMNS[function]]


def _ids(probes: List[Dict[str, Any]]) -> List[str]:
    """Probe ids for pytest parametrisation.

    :param probes: Probe dictionaries.
    :type probes: List[Dict[str, Any]]
    :return: The probe ids.
    :rtype: List[str]
    """
    return [p["id"] for p in probes]


def test_fixture_shape() -> None:
    """The fixture carries all 170 probes and the 56 ES ids they name.

    :return: None.
    :rtype: None
    """
    assert len(PROBES) == 170 and len({p["id"] for p in PROBES}) == 170
    assert ES_IDS == {p["expected_es_id"] for p in PROBES if p["expected_in_es"]}
    assert shelfmarks.NORMALISER_VERSION == "2026.11.03-3"


def test_id_input_probes_are_exactly_the_es_id_inputs() -> None:
    """The id-input rule fires on P032 and P097 and on no other probe.

    :return: None.
    :rtype: None
    """
    fired = {p["id"] for p in PROBES if shelfmarks._id_input_core(p["mark"]) is not None}
    assert fired == set(ID_INPUT_PROBES)
    assert all((BY_ID[pid]["mark"] in ES_IDS) is in_es for pid, in_es in ID_INPUT_PROBES.items())


def test_intended_changes_land_on_es_ids_or_none() -> None:
    """Every intended ``canonical_id`` is an ES id of the snapshot or ``None``; every intended core is ``""`` or the
    core of the probe's expected ES id; every listed ``canonical_id`` really departs from the reference.

    :return: None.
    :rtype: None
    """
    for pid, row in INTENDED_CHANGES.items():
        probe = BY_ID[pid]
        if "canonical_id" in row:
            value = row["canonical_id"]
            assert value is None or value in ES_IDS, pid
            assert value == probe["expected_es_id"], pid
            assert value != probe["REF_canonical_id"], pid
        if "core_id" in row:
            core = row["core_id"]
            assert core == "" or (probe["expected_es_id"] in ES_IDS
                                  and core == shelfmarks.split_id(probe["expected_es_id"])[1]), pid


@pytest.mark.parametrize("function", list(REFERENCE_COLUMNS))
@pytest.mark.parametrize("probe", PROBES, ids=_ids(PROBES))
def test_golden(probe: Dict[str, Any], function: str) -> None:
    """Each facade function gives its reference column, or the intended value for a probe in the table.

    ``canonical_id`` and ``loose_key`` take the probe's institution; ``core_id`` and ``core_key`` are
    institution-free.

    :param probe: One fixture probe (id, family, mark, institution, ``REF_*`` columns, expected ES id).
    :type probe: Dict[str, Any]
    :param function: Facade function name (a key of :data:`REFERENCE_COLUMNS`).
    :type function: str
    :return: None.
    :rtype: None
    """
    args = (probe["mark"], probe["institution"]) if function in ("canonical_id", "loose_key") else (probe["mark"],)
    assert getattr(shelfmarks, function)(*args) == _golden(probe, function)


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


# ------------------------------------------------------------------------------------ PR-2 functions

#: ``split`` goldens: probe id -> the single marks.
SPLIT_GOLDEN: Dict[str, List[str]] = {
    "P154": ["T-S 16.22", "T-S 10J5.17"],
    "P155": ["TS 16.44", "TS 12.613"],
    "P157": ["Cambridge CUL: T-S 16.44", "Cambridge CUL: T-S 12.613"],
    "P158": ["T-S 12.613", "T-S 16.44"],
    "P156": ["T-S 8J5.2a", "T-S 8J5.2b", "T-S 8J5.2c"],
}


@pytest.mark.parametrize("pid", sorted(SPLIT_GOLDEN))
def test_split_golden(pid: str) -> None:
    """Join probes split into their single marks, with no trailing join punctuation.

    :param pid: Probe id.
    :type pid: str
    :return: None.
    :rtype: None
    """
    assert shelfmarks.split(BY_ID[pid]["mark"]) == SPLIT_GOLDEN[pid]


@pytest.mark.parametrize("probe", [p for p in PROBES if p["id"] not in SPLIT_GOLDEN and p["mark"].strip()],
                         ids=_ids([p for p in PROBES if p["id"] not in SPLIT_GOLDEN and p["mark"].strip()]))
def test_split_keeps_every_single_mark(probe: Dict[str, Any]) -> None:
    """Every other probe is one mark: ``split`` returns it unchanged (institution words, ``and`` inside a KTIV
    head, ranges and piece ranges such as ``c-e`` are not joins).

    :param probe: One fixture probe.
    :type probe: Dict[str, Any]
    :return: None.
    :rtype: None
    """
    assert shelfmarks.split(probe["mark"]) == [probe["mark"]]


def test_split_unions_the_two_splitters() -> None:
    """``split`` covers both splitters' cases (merge: ``Inst:`` prefixes carried forward; lit_segment: ``and``,
    prefix inheritance, numbered pieces) and calls them rather than copying them.

    :return: None.
    :rtype: None
    """
    assert shelfmarks.split("Cambridge CUL: T-S A.1 T-S B.2") == ["Cambridge CUL: T-S A.1", "Cambridge CUL: T-S B.2"]
    assert shelfmarks.split("TS NS J 110 and 152") == ["TS NS J 110", "TS NS J 152"]
    assert shelfmarks.split("TS NS J 185, nos. 5, 6") == ["TS NS J 185, no. 5", "TS NS J 185, no. 6"]
    assert shelfmarks.split("CAJS Halper 472") == ["CAJS Halper 472"]
    assert shelfmarks.split("") == [] and shelfmarks.split("   ") == []
    for mark in ("TS 16.22 and TS 10 J 5, f. 17", "ENA 1234.5"):
        assert shelfmarks.split(mark) == (lit_segment.split_joins(mark) if len(lit_segment.split_joins(mark)) > 1
                                          else [mark])


def test_canonical_id_of_each_split_part() -> None:
    """A join gives ``None``; each of its parts gives its own ES id.

    :return: None.
    :rtype: None
    """
    for pid in _JOINS:
        probe = BY_ID[pid]
        assert shelfmarks.canonical_id(probe["mark"], probe["institution"]) is None
        assert shelfmarks.core_id(probe["mark"]) == ""
        assert all(shelfmarks.canonical_id(part, probe["institution"]) for part in shelfmarks.split(probe["mark"]))
    assert [shelfmarks.canonical_id(p) for p in shelfmarks.split(BY_ID["P154"]["mark"])] == [
        "Cambridge_CUL_T_S_16_22", _CUL_10J5_17]


#: ``side_suffix`` goldens: probe id -> (side, the ES id of the base).
SIDE_GOLDEN: Dict[str, Tuple[str, str]] = {
    "P048": ("r", _CUL_12_301), "P049": ("r", _CUL_12_301), "P050": ("v", _CUL_12_301),
    "P051": ("r", _CUL_12_301), "P052": ("v", _CUL_12_301),
    "P053": ("1r", _CUL_10J5_17), "P054": ("1v", _CUL_10J5_17),
    "P105": ("v", "Paris_AIU_XII_87"),
}


@pytest.mark.parametrize("pid", sorted(SIDE_GOLDEN))
def test_side_suffix_golden(pid: str) -> None:
    """The side is stripped and normalised; the base gives the expected ES id.

    :param pid: Probe id.
    :type pid: str
    :return: None.
    :rtype: None
    """
    probe = BY_ID[pid]
    base, side = shelfmarks.side_suffix(probe["mark"])
    side_expected, es_id = SIDE_GOLDEN[pid]
    assert side == side_expected
    assert es_id in ES_IDS and es_id == probe["expected_es_id"]
    assert shelfmarks.canonical_id(base, probe["institution"]) == es_id
    assert shelfmarks.canonical_id(base) == es_id


@pytest.mark.parametrize("probe", [p for p in PROBES if p["id"] not in SIDE_GOLDEN],
                         ids=_ids([p for p in PROBES if p["id"] not in SIDE_GOLDEN]))
def test_side_suffix_leaves_other_probes_alone(probe: Dict[str, Any]) -> None:
    """No other probe ends in a side marker (P167 ``Plate IV`` included: roman numerals are never stripped).

    :param probe: One fixture probe.
    :type probe: Dict[str, Any]
    :return: None.
    :rtype: None
    """
    assert shelfmarks.side_suffix(probe["mark"]) == (probe["mark"], None)


def test_side_suffix_never_strips_ranges_piece_letters_or_numerals() -> None:
    """Folio ranges, piece letters and upper-case / roman letters stay; one marker only is stripped.

    :return: None.
    :rtype: None
    """
    for mark in ("Codex 1700, ff. 9r-10r", "Codex 1700 9r - 10r", "Codex_1700_9r_10r", "T-S 12a",
                 "Plate IV", "F 1908.44A", "T-S F3.74B", "Mosseri IV",
                 "T-S 12.301 recto and verso", "T-S 12.301 recto + verso", "T-S 12.301 recto & verso",
                 "T-S 12.301 r-v"):
        assert shelfmarks.side_suffix(mark) == (mark, None), mark
    assert shelfmarks.side_suffix("T-S 12.301 r") == ("T-S 12.301", "r")
    assert shelfmarks.side_suffix("T-S 12.301r verso") == ("T-S 12.301r", "v")
    assert shelfmarks.side_suffix("ENA 2727, 5r") == ("ENA 2727, 5", "r")
    # a folio after a box-and-item mark is a side of the item, however it is spelled: one leaf for all four
    for mark, side in (("T-S 12.301 1v", "1v"), ("T-S 12.301 fol. 1v", "1v"), ("T-S 12.301, f. 1v", "1v"),
                       ("T-S 12.301 1 r", "1r")):
        assert shelfmarks.side_suffix(mark) == ("T-S 12.301", side), mark
        assert shelfmarks.canonical_id(mark) == _CUL_12_301, mark
    # a Bodleian folio is a leaf of its own (Oxford_Bodleian_Bodl_MS_heb_c_13_25): only the side is stripped
    assert shelfmarks.side_suffix("MS Heb. d 66, fol. 9r") == ("MS Heb. d 66, fol. 9", "r")
    # a join string, or a base ending in a join and a number, keeps its sides (call split first)
    for mark in ("T-S 12.301 f. 1r and 2v", "T-S 16.22v / T-S 10J5.17r", "T-S 12.301 1r + 2v"):
        assert shelfmarks.side_suffix(mark) == (mark, None), mark
        assert shelfmarks.canonical_id(mark) is None, mark
    assert shelfmarks.side_suffix("Ms. Heb. 4° 577/3v") == ("Ms. Heb. 4° 577/3", "v")  # a glued ``/`` is no join


#: A folio other than 1 after a box-and-item mark is a child leaf: (mark, side). All give the child's ES id.
CHILD_FOLIOS: Tuple[Tuple[str, str], ...] = (
    ("T-S Misc. 28.79, f. 14r", "14r"), ("T-S Misc. 28.79 14r", "14r"), ("T-S Misc. 28.79, fol. 14v", "14v"),
    ("T-S Misc. 28.79, 14v", "14v"),
)


@pytest.mark.parametrize("mark,side", CHILD_FOLIOS, ids=[m for m, _ in CHILD_FOLIOS])
def test_a_child_folio_keeps_its_number(mark: str, side: str) -> None:
    """Only the side letter of a child folio is stripped; its number stays in the id (not merged into the parent).

    Folio 1 is the item's own leaf (P053 / P054: ``T-S 10J5.17 1r`` -> ``Cambridge_CUL_T_S_10J5_17``).

    :param mark: A child-folio mark with a side.
    :type mark: str
    :param side: The expected side.
    :type side: str
    :return: None.
    :rtype: None
    """
    child = "Cambridge_CUL_T_S_Misc_28_79_14"
    assert child in ES_IDS and "Cambridge_CUL_T_S_Misc_28_79" in ES_IDS
    base, got = shelfmarks.side_suffix(mark)
    assert got == side and base.endswith("14")
    assert shelfmarks.canonical_id(mark) == shelfmarks.canonical_id(base) == child
    assert shelfmarks.canonical_id("T-S Misc. 28.79, f. 14") == shelfmarks.canonical_id(BY_ID["P013"]["mark"]) == child
    for pid in ("P053", "P054"):
        assert shelfmarks.canonical_id(BY_ID[pid]["mark"]) == _CUL_10J5_17


def test_join_punctuation_is_trimmed() -> None:
    """``/`` is join punctuation for ``split``; a trailing join character does not leak into an id.

    :return: None.
    :rtype: None
    """
    assert shelfmarks.split("T-S 12.301 / T-S 12.302") == ["T-S 12.301", "T-S 12.302"]
    assert shelfmarks.canonical_id("T-S 12.301 / T-S 12.302") is None
    assert shelfmarks.canonical_id("ENA NS 1.1 ; ") == "New_York_JTS_ENA_NS_1_1"
    assert shelfmarks.canonical_id("T-S 12.301 +") == _CUL_12_301


#: ``variants`` goldens (R4): probe id -> the variant that the series fold produces.
VARIANT_GOLDEN: Dict[str, str] = {
    "P016": "T-S 12.301",
    "P026": "T-S Ar. 53.70",
    "P027": "T-S Ar. 53, f. 70",
    "P073": "CUL Or 1080 Box 1, f. 85",
    "P083": "L-G Ar. I.49",
    "P086": "L-G Misc. 10",
}


@pytest.mark.parametrize("pid", sorted(VARIANT_GOLDEN))
def test_variants_golden(pid: str) -> None:
    """The input comes first, the series fold second, and the fold gives the expected ES id.

    :param pid: Probe id.
    :type pid: str
    :return: None.
    :rtype: None
    """
    probe = BY_ID[pid]
    got = shelfmarks.variants(probe["mark"])
    assert got[:2] == [probe["mark"], VARIANT_GOLDEN[pid]]
    assert probe["expected_es_id"] in ES_IDS
    assert shelfmarks.canonical_id(got[1], probe["institution"]) == probe["expected_es_id"]


#: R6 probes of the legacy-spelling families (not changed in the facade).
R6_PROBES = frozenset({"P080"})


@pytest.mark.parametrize("probe", [p for p in PROBES if p["family"] in ("legacy_ts", "lewis_gibson", "cul_or_add")],
                         ids=_ids([p for p in PROBES if p["family"] in ("legacy_ts", "lewis_gibson", "cul_or_add")]))
def test_variants_reach_the_expected_id(probe: Dict[str, Any]) -> None:
    """For the legacy-spelling families, some variant gives the expected ES id (when the probe names one).

    R6 (case / class-mark collapse beyond T-S, P080 ``Or.1080 J 262``) is not a spelling variant: the ES holds
    both forms and folding them is an ES re-key (section 4 of the migration plan), so it is exempt.

    :param probe: One fixture probe.
    :type probe: Dict[str, Any]
    :return: None.
    :rtype: None
    """
    got = shelfmarks.variants(probe["mark"])
    assert got[0] == probe["mark"] and len(got) == len(set(got))
    if probe["expected_es_id"] and probe["id"] not in R6_PROBES:
        assert probe["expected_es_id"] in {shelfmarks.canonical_id(v, probe["institution"]) for v in got}


#: lit_segment.legacy_variants cases (``tests/test_text27b_lit_segment.py``): the facade's variants include them.
LEGACY_CASES = ["TS 8 J 34, f. 8", "TS NS Box 226, f. 11", "TS Arabic 30.123", "TS 8.223 Verso",
                "Bodleian MS Heb. d 66.5", "ENA 2805, fol. 16B", "TS Misc Box 28, f. 79, no 1", "TS NS J 318(a)",
                "ULC Or Box 5, f. 13", "Westminster College, Misc. 34", "E 16.510", "JTS Krengel 5.123"]


@pytest.mark.parametrize("mark", LEGACY_CASES)
def test_variants_include_the_lit_segment_variants(mark: str) -> None:
    """``variants`` calls ``lit_segment.legacy_variants`` and keeps all of its candidates.

    :param mark: A printed shelf mark.
    :type mark: str
    :return: None.
    :rtype: None
    """
    got = shelfmarks.variants(mark)
    assert got[0] == mark
    assert set(lit_segment.legacy_variants(mark)) <= set(got)


def test_variants_safe_cleaners_and_alt_classmark() -> None:
    """The text27b cleaning, the safe key-cleaner spellings and the alternate classmark are candidates; the
    BnF fold is a candidate only (``canonical_id`` keeps the reference on P142).

    :return: None.
    :rtype: None
    """
    assert "T-S 24.31" in shelfmarks.variants("TS Box 24.31 (secondary use)")
    assert "Mosseri II 127.1" in shelfmarks.variants("Mosseri L 129.1 (II 127.1) 36")
    assert "Plate IV" in shelfmarks.variants("Plate IV") and "Plate I" not in shelfmarks.variants("Plate IV")
    bnf = BY_ID["P142"]
    assert "Paris hebr. 1234" in shelfmarks.variants(bnf["mark"])
    assert shelfmarks.canonical_id(bnf["mark"], bnf["institution"]) == bnf["REF_canonical_id"]


def test_parent_golden() -> None:
    """P013 (child .14) has the parent P010's id when the ES holds it; P010 itself has no parent there.

    :return: None.
    :rtype: None
    """
    child, parent_probe = BY_ID["P013"], BY_ID["P010"]
    parent_id = parent_probe["expected_es_id"]
    assert parent_id == "Cambridge_CUL_T_S_Misc_28_79" and parent_id in ES_IDS
    assert shelfmarks.parent(child["mark"], ES_IDS) == parent_id
    assert shelfmarks.parent(child["mark"], ES_IDS, child["institution"]) == parent_id
    assert shelfmarks.parent(parent_probe["mark"], ES_IDS) is None
    assert shelfmarks.parent(child["mark"]) is None  # no ES set given: resolve() (PR-6) supplies it
    assert shelfmarks.parent(child["mark"], frozenset()) is None
    assert shelfmarks.parent("Plate IV", ES_IDS) is None
