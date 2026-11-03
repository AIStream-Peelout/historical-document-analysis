"""PR-6 resolver tests: ``ShelfmarkIndex`` + ``resolve`` of the shelf-mark facade, the moved printed-mark helpers,
``cudl_manifest`` (now owned by the facade) and the text27b ``match_shelfmark`` that calls ``resolve``.

Golden: the ``expected_es_id`` of the 140 single-item probes of ``tests/fixtures/shelfmark_probes.json``, resolved
through an index built from ``tests/fixtures/shelfmark_index_fixture.jsonl`` (71 scrubbed merged-corpus records: the
56 ids of ``es_id_sample.txt`` plus 15 records that share a loose or core key with a probe; only ids, display
marks, institutions, PGP ids, historic spellings and source shelf-mark strings).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.datasets import shelfmarks as F
from src.finetuning.text27b import lit_segment as LS
from src.finetuning.text27b import shelfmarks as SM

FIXTURES = Path(__file__).resolve().parent / "fixtures"
INDEX_FIXTURE = FIXTURES / "shelfmark_index_fixture.jsonl"
PROBES: Dict[str, Any] = json.loads((FIXTURES / "shelfmark_probes.json").read_text(encoding="utf-8"))

#: In-ES probes that resolve to ANOTHER ES id of the same leaf (Tier-1 duplicates of the migration plan, section 4:
#: the mark is spelled exactly as that duplicate record's own id / display form).
KNOWN_DUPLICATES: Dict[str, str] = {
    "P056": "New_York_JTS_MS_9160_7",  # R9 JTS ``MS`` namespace (expected New_York_JTS_9160_7)
    "P101": "Cambridge_Mosseri_IV10",  # R6 class-mark collapse (expected Cambridge_Mosseri_IV_10)
    "P159": "Cambridge_CUL_T_S_F3_74B",  # R6 case (expected Cambridge_CUL_T_S_F3_74b; the record's own KTIV mark)
}
#: In-ES probes the resolver leaves unmatched (spellings the ES does not hold in any form: the ``Kaufmann`` and
#: ``BNU`` holder words stay in the core; Q13 / R4 of the plan).
KNOWN_UNMATCHED = frozenset({"P123", "P139"})
#: Probes whose expected id is NOT in the ES, but the ES holds the leaf under another spelling (R4/R5/R6
#: namespaces): the resolver returns that real id.
NOT_IN_ES_RESOLVED: Dict[str, List[str]] = {
    "P127": ["Strasbourg_Stras_4038_10"],
    "P132": ["Vienna_ONB_Austrian_National_Library_Vienna_Austria_Cod_hebr_65"],
    "P140": ["Strasbourg_Stras_4038_10"],
    "P149": ["Heidelberg_p_Heid_Arab_2"],
}


@pytest.fixture(scope="module")
def index() -> F.ShelfmarkIndex:
    """The fixture index (built from the scrubbed JSONL path).

    :return: The index.
    :rtype: F.ShelfmarkIndex
    """
    return F.ShelfmarkIndex(INDEX_FIXTURE)


def _single_item_probes() -> List[Dict[str, Any]]:
    """The probes that name one item.

    :return: Probes with an ``expected_es_id``.
    :rtype: List[Dict[str, Any]]
    """
    return [p for p in PROBES["probes"] if p["expected_es_id"]]


# ------------------------------------------------------------------------------------ fixture hygiene

def test_index_fixture_is_scrubbed_and_covers_the_sample() -> None:
    """The fixture holds only the allowed fields, no URLs, nothing of the excluded holding (its registry token and
    series words, taken from the registry so they are not spelled out here), and every sampled ES id."""
    records = [json.loads(line) for line in INDEX_FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) <= 200
    allowed = {"canonical_id", "shelfmark_display", "institution", "pgpids", "aliases", "source_marks"}
    assert all(set(r) <= allowed for r in records)
    text = INDEX_FIXTURE.read_text(encoding="utf-8")
    assert not re.search(r"https?://", text)
    excluded = F.institution_token("National Library of " + "Russia")
    words = [w for w, tok in F._engine.SERIES_TOKENS.items() if tok == excluded]
    assert excluded.count("_") == 1 and len(words) >= 3
    assert not re.search("|".join([re.escape(excluded.split("_")[1])] + [rf"\b{re.escape(w)}\b" for w in words]), text)
    sample = set((FIXTURES / "es_id_sample.txt").read_text(encoding="utf-8").split())
    assert sample <= {r["canonical_id"] for r in records}


# ------------------------------------------------------------------------------------ golden

def test_resolve_golden_single_item_probes(index: F.ShelfmarkIndex) -> None:
    """Every single-item probe (with its own institution) resolves to its expected ES id, or to a pinned exception.

    122 probes expect an id the ES holds: 117 resolve to it exactly, 3 to a Tier-1 duplicate id of the same leaf,
    2 stay unmatched. 18 expect an id the ES does not hold: 14 stay unmatched and 4 resolve to the ES's own
    spelling of that leaf. Nothing else. (The same counts hold on the full merged corpus, measured for PR-6.)

    :param index: The fixture index.
    """
    resolved, unmatched, wrong = 0, [], []
    for p in _single_item_probes():
        res = F.resolve(p["mark"], index=index, institution=p["institution"])
        assert set(res.cids) <= index.ids and res.how in F.HOW_ORDER
        if p["expected_in_es"]:
            want = [KNOWN_DUPLICATES.get(p["id"], p["expected_es_id"])] if p["id"] not in KNOWN_UNMATCHED else []
        else:
            want = NOT_IN_ES_RESOLVED.get(p["id"], [])
        if res.cids == [p["expected_es_id"]]:
            resolved += 1
        elif not res.cids:
            unmatched.append(p["id"])
        else:
            wrong.append(p["id"])
        assert res.cids == want, (p["id"], p["mark"], res)
    assert resolved == 117
    assert sorted(unmatched) == sorted(KNOWN_UNMATCHED | (
        {p["id"] for p in _single_item_probes() if not p["expected_in_es"]} - set(NOT_IN_ES_RESOLVED)))
    assert sorted(wrong) == sorted(set(KNOWN_DUPLICATES) | set(NOT_IN_ES_RESOLVED))


def test_resolve_never_returns_an_id_outside_the_index(index: F.ShelfmarkIndex) -> None:
    """No probe mark, expected id or variant of either resolves to an id the index does not hold.

    :param index: The fixture index.
    """
    for p in PROBES["probes"]:
        for x in [p["mark"], p["expected_es_id"] or ""] + F.variants(p["mark"]):
            for inst in (None, p["institution"]):
                assert set(F.resolve(x, index=index, institution=inst).cids) <= index.ids


def test_resolve_es_ids_and_unknown_ids_are_exact(index: F.ShelfmarkIndex) -> None:
    """Every id of the index resolves to itself; an id the index lacks does not.

    :param index: The fixture index.
    """
    for cid in index.ids:
        assert F.resolve(cid, index=index) == F.Resolution([cid], "exact")
    assert F.resolve("Cambridge_CUL_T_S_99J9_99", index=index).cids == []


# ------------------------------------------------------------------------------------ chain steps

def _mini(records: List[Dict[str, Any]], fragments: Any = None) -> F.ShelfmarkIndex:
    """A small in-memory index.

    :param records: Slim records.
    :type records: List[Dict[str, Any]]
    :param fragments: Optional fragment rows.
    :type fragments: Any
    :return: The index.
    :rtype: F.ShelfmarkIndex
    """
    return F.ShelfmarkIndex(records, fragments=fragments)


def test_resolve_alias_side_parent_and_loose(index: F.ShelfmarkIndex) -> None:
    """Historic spellings resolve by alias; a side is stripped; a child resolves to its parent; case is loose.

    :param index: The fixture index.
    """
    assert F.resolve("Dropsie 472", index=index) == F.Resolution(["Philadelphia_CAJS_Halper_472"], "alias")
    assert F.resolve("T-S 12.301 verso", index=index) == F.Resolution(["Cambridge_CUL_T_S_12_301"], "exact")
    small = _mini([{"canonical_id": "Cambridge_CUL_T_S_Misc_28_79"}, {"canonical_id": "Cambridge_CUL_T_S_Misc_28_79_1"},
                   {"canonical_id": "Cambridge_CUL_T_S_F3_74b"}])
    assert F.resolve("T-S Misc.28.79.14", index=small) == F.Resolution(["Cambridge_CUL_T_S_Misc_28_79"], "parent")
    assert F.resolve("T-S F3.74B", index=small) == F.Resolution(["Cambridge_CUL_T_S_F3_74b"], "loose")


def test_resolve_never_lossy_and_never_a_box_parent() -> None:
    """A dropped note number or a box / volume id is not a resolution (it would name another leaf or every leaf)."""
    small = _mini([{"canonical_id": "Cambridge_CUL_T_S_AS_18_11"}, {"canonical_id": "Cambridge_CUL_T_S_NS_322"},
                   {"canonical_id": "Strasbourg_4038"}])
    assert "T-S AS 18.11" in F.variants("T-S AS 18.1163") and "T-S AS 18.11" not in F.variants("T-S AS 18.1163", False)
    assert F.resolve("T-S AS 18.1163", index=small) == F.Resolution([], "unmatched")
    assert F.resolve("T-S NS 322.64", index=small) == F.Resolution([], "unmatched")
    assert F.resolve("Strasbourg 4038.10", index=small) == F.Resolution([], "unmatched")


def test_resolve_side_ful_es_id_fallback() -> None:
    """The three ES ids keyed on one side (R3) are still reached from their raw marks."""
    bare = _mini([{"canonical_id": "Cambridge_CUL_T_S_H3_104r"}])
    assert F.canonical_id("T-S H3.104r") == "Cambridge_CUL_T_S_H3_104"
    assert F.resolve("T-S H3.104r", index=bare) == F.Resolution(["Cambridge_CUL_T_S_H3_104r"], "exact")
    # with its display mark the record also answers to the side-less alias id
    full = _mini([{"canonical_id": "Cambridge_CUL_T_S_H3_104r", "shelfmark_display": "T-S H3.104r"}])
    assert F.resolve("T-S H3.104", index=full) == F.Resolution(["Cambridge_CUL_T_S_H3_104r"], "alias")


def test_resolve_core_key_only_when_one_institution() -> None:
    """An institution-free core match is used only when its ids carry one institution token."""
    one = _mini([{"canonical_id": "Birmingham_Mingana_10", "shelfmark_display": "Birmingham: Mingana 10"}])
    two = _mini([{"canonical_id": "Birmingham_Mingana_10"}, {"canonical_id": "Budapest_MTA_10"}])
    assert F.core_key("Budapest_MTA_10") == F.core_key("10") == "10"
    assert two.institutions("10") == 2 and two.core_unique("10") == set()
    assert F.resolve("10", index=two) == F.Resolution([], "unmatched")
    assert one.institutions("mingana10") == 1
    assert F.resolve("Mingana 10", index=one).cids == ["Birmingham_Mingana_10"]


def test_resolve_garbage_without_a_number_stays_unmatched() -> None:
    """A spelling without a number gives no alias id or key, and resolves only as an ES id itself."""
    small = _mini([{"canonical_id": "London_BL_Or_9999_1", "shelfmark_display": "BL Or. 9999.1", "aliases": ["BL"]},
                   {"canonical_id": "Oxford_Bodleian_Bodl_MS_heb_d_99_1", "aliases": ["Bodl."]},
                   {"canonical_id": "Manchester_JRL_A_9", "aliases": ["JRL"]},
                   {"canonical_id": "Paris_AIU_VII_C", "aliases": ["AIU"]}])
    assert not [k for k in small.alias if not F._has_digit(k)]
    for garbage in ("BL", "Bodl.", "JRL", "AIU", "London_BL", "Paris_AIU"):
        assert F.resolve(garbage, index=small) == F.Resolution([], "unmatched"), garbage
    # a Roman numeral is a number: the AIU box still resolves
    assert F.resolve("AIU VII C", index=small) == F.Resolution(["Paris_AIU_VII_C"], "exact")
    assert F.resolve("BL Or. 9999.1", index=small).cids == ["London_BL_Or_9999_1"]


def test_resolve_core_step_needs_a_strong_key() -> None:
    """Bare numbers, short keys and page / folio references never take the institution-free core step."""
    small = _mini([{"canonical_id": "Birmingham_Mingana_5"}, {"canonical_id": "Frankfurt_152"},
                   {"canonical_id": "Manchester_JRL_P_23"}, {"canonical_id": "Washington_Freer_F_12r"},
                   {"canonical_id": "Manchester_JRL_A_5"}, {"canonical_id": "Washington_Freer_F_1908_44BB"},
                   {"canonical_id": "Cambridge_CUL_Genizah_5"}, {"canonical_id": "Cambridge_CUL_Letter_12"},
                   {"canonical_id": "Cambridge_CUL_Fig_12"}, {"canonical_id": "Vienna_ONB_PER_H_99"}])
    for weak in ("5", "152", "p. 23", "f. 12r", "a5", "-a Z+564XbY 0:a5_-+", "fol. 12r", "pp. 23-24",
                 "Genizah 5", "Letter 12", "Fig. 12"):
        res = F.resolve(weak, index=small)
        assert (res.cids, res.how, res.ambiguous) == ([], "unmatched", False), weak
    assert F.resolve("pp. 23-24", index=small).note == "range"
    # a Freer mark is not a folio reference; a siglum (``PER``) is not prose
    assert F.resolve("F 1908.44BB", index=small) == F.Resolution(["Washington_Freer_F_1908_44BB"], "core")
    assert F.resolve("PER H 99", index=small).cids == ["Vienna_ONB_PER_H_99"]
    assert not F._core_key_usable(F.core_key("Genizah 5"), "Genizah 5")
    assert F._core_key_usable(F.core_key("Genizah 5.3"), "Genizah 5.3")  # two digit groups


def test_resolve_bare_number_core_never_crosses_institutions() -> None:
    """An institution acronym plus a bare number (``JTS 5``, ``CUL 132``) never core-matches the one record of
    ANOTHER institution that happens to carry that number; the record's own spellings still resolve."""
    small = _mini([{"canonical_id": "Birmingham_Mingana_5", "shelfmark_display": "Ms. Mingana 5"},
                   {"canonical_id": "Budapest_MTA_132", "shelfmark_display": "DK 132"},
                   {"canonical_id": "New_York_JTS_ENA_5", "shelfmark_display": "ENA 5"},
                   {"canonical_id": "Budapest_MTA_133a"}, {"canonical_id": "Paris_AIU_IX"}])
    # the bare-number core keys are unique in this index: the hazard the rule guards against
    assert small.core_unique("5") == {"Birmingham_Mingana_5"}
    assert small.core_unique("132") == {"Budapest_MTA_132"}
    for mark in ("JTS 5", "CUL 5", "CUL 132", "JTS 132", "JTS 133a", "CUL IX"):
        res = F.resolve(mark, index=small)
        assert res.cids == [] and (res.how == "unmatched" or res.ambiguous), (mark, res)
    # the same bare number given with another institution
    res = F.resolve("JTS 5", index=small, institution="Cambridge University Library")
    assert res.cids == [] and (res.how == "unmatched" or res.ambiguous), res
    assert F.resolve("Mingana 5", index=small) == F.Resolution(["Birmingham_Mingana_5"], "exact")
    assert F.resolve("Ms. Mingana 5", index=small).cids == ["Birmingham_Mingana_5"]
    assert F.resolve("ENA 5", index=small) == F.Resolution(["New_York_JTS_ENA_5"], "exact")
    assert F.resolve("DK 132", index=small).cids == ["Budapest_MTA_132"]
    assert F._is_bare_number_core("JTS 5") and F._is_bare_number_core("JTS 12.301")
    assert F._is_bare_number_core("JTS 133a") and F._is_bare_number_core("CUL IX")
    assert F.resolve("AIU IX", index=small) == F.Resolution(["Paris_AIU_IX"], "exact")
    assert not F._is_bare_number_core("ENA 5") and not F._is_bare_number_core("Mingana 5")
    # a core that keeps a series or holder word may still cross the inferred institution (legacy-store head)
    legacy = _mini([{"canonical_id": "Cambridge_CUL_T_S_AS_62_262"}])
    hit = F.resolve("New York JTS: T-S AS 62.262", index=legacy)
    assert hit == F.Resolution(["Cambridge_CUL_T_S_AS_62_262"], "core")


@pytest.mark.parametrize("mark, is_range", [
    ("T-S 10J5.17-18", True), ("T-S 12.6-7", True), ("ff. 9-10", True), ("T-S 12.6\u20137", True),
    ("BL Or. 5566c ff. 9-10", True), ("Yevr.-Arab. I 1700.21r-21v", True), ("BM Or 5552, fol. 2-2a", True),
    ("T-S AS 148.161-63", True), ("Strasbourg: 4028/1 - 4028/14", True),
    ("T-S 12.6", False), ("T-S 12.7-6", False), ("MS-TS-00012-00301", False), ("MS-OR-01080-00001-00085", False),
    ("FR 9-004/20", False), ("Princeton 584 (1) AM 2005-125", False), ("T-S NS 322.64", False),
])
def test_is_range(mark: str, is_range: bool) -> None:
    """Numeric ranges are recognised; CUDL ids, zero-padded box numbers and accession numbers are not ranges.

    :param mark: Raw mark.
    :param is_range: Expected verdict.
    """
    assert F._is_range(mark) is is_range


def test_resolve_range_never_collapses_to_its_first_leaf() -> None:
    """A numeric range is ``unmatched`` with ``note="range"`` (Q11 open): no first leaf, no parent, no core / variant
    step; only the ES's own record of that very range is returned (``exact``)."""
    small = _mini([{"canonical_id": "Cambridge_CUL_T_S_10J5_17", "shelfmark_display": "T-S 10J5.17"},
                   {"canonical_id": "Cambridge_CUL_T_S_10J5_17_1"},
                   {"canonical_id": "Cambridge_CUL_T_S_12_6", "shelfmark_display": "T-S 12.6"},
                   {"canonical_id": "London_BL_Or_5566c_9", "shelfmark_display": "BL Or. 5566c, f. 9"},
                   {"canonical_id": "Birmingham_Mingana_9"},
                   {"canonical_id": "New_York_JTS_ENA_9846_3_4", "shelfmark_display": "ENA 9846.3-4"}])
    for mark in ("T-S 10J5.17-18", "T-S 12.6-7", "ff. 9-10", "BL Or. 5566c ff. 9-10", "T-S 12.6-7v"):
        assert F.resolve(mark, index=small) == F.Resolution([], "unmatched", False, "range"), mark
    assert F.parent("T-S 10J5.17-18", small.ids) is None
    assert F.resolve("ENA 9846.3-4", index=small) == F.Resolution(["New_York_JTS_ENA_9846_3_4"], "exact", False,
                                                                  "range")
    # a join with a range part: the other part's ids, ``unmatched``, the note kept
    assert F.resolve("T-S 12.6 + T-S 10J5.17-18", index=small) == F.Resolution(
        ["Cambridge_CUL_T_S_12_6"], "unmatched", False, "range")
    assert F.resolve("T-S 12.6", index=small) == F.Resolution(["Cambridge_CUL_T_S_12_6"], "exact")


def test_parent_needs_numbered_children_in_the_index() -> None:
    """The parent fallback takes a two-segment leaf only when the index holds numbered children of it."""
    leaf = _mini([{"canonical_id": "Cambridge_CUL_T_S_10J5_17"}])
    assert F.resolve("T-S 10J5.17.3", index=leaf) == F.Resolution([], "unmatched")
    assert leaf.numbered_parents() == set()
    parent = _mini([{"canonical_id": "Cambridge_CUL_T_S_10J5_17"}, {"canonical_id": "Cambridge_CUL_T_S_10J5_17_1"}])
    assert parent.numbered_parents() == {"Cambridge_CUL_T_S_10J5_17"}
    assert F.resolve("T-S 10J5.17.3", index=parent) == F.Resolution(["Cambridge_CUL_T_S_10J5_17"], "parent")


def test_resolve_multi_id_hits_are_ambiguous_unless_case_exact() -> None:
    """Several ids for one alias / loose / core key are flagged ``ambiguous``; the case-exact core wins when unique."""
    dup = _mini([{"canonical_id": "London_BL_Or_9998__sys1", "shelfmark_display": "Or. 9998"},
                 {"canonical_id": "London_BL_Or_9998__sys2", "shelfmark_display": "Or. 9998"}])
    res = F.resolve("BL Or. 9998", index=dup)
    assert res == F.Resolution(["London_BL_Or_9998__sys1", "London_BL_Or_9998__sys2"], "alias", True)
    assert F.resolve("BL Or. 9998 + BL Or. 9997", index=dup).ambiguous
    case = _mini([{"canonical_id": "Heidelberg_P_Heid_Hebr_99"}, {"canonical_id": "Heidelberg_p_Heid_Hebr_99"}])
    assert F.resolve("p. Heid. Hebr. 99", index=case) == F.Resolution(["Heidelberg_p_Heid_Hebr_99"], "core")
    assert F.resolve("Heidelberg_P_Heid_Hebr_99", index=case) == F.Resolution(["Heidelberg_P_Heid_Hebr_99"], "exact")


def test_fragment_rows_link_pgp_ids_to_their_record() -> None:
    """A fragment row whose mark resolves to one id adds its PGP ids to that id (merged records miss them)."""
    small = _mini([{"canonical_id": "Cambridge_CUL_T_S_99J9_1", "shelfmark_display": "T-S 99J9.1", "pgpids": []},
                   {"canonical_id": "Cambridge_CUL_T_S_99J9_2", "shelfmark_display": "T-S 99J9.2", "pgpids": ["9006"]}],
                  fragments=[{"shelfmark": "T-S 99J9.1", "pgpids": "9005;9006"},
                             {"shelfmark": "T-S 99J9", "pgpids": "9007"}])
    assert small.pgpids["Cambridge_CUL_T_S_99J9_1"] == ["9005", "9006"]
    assert F.resolve("pgpid:9006", index=small) == F.Resolution(
        ["Cambridge_CUL_T_S_99J9_1", "Cambridge_CUL_T_S_99J9_2"], "pgpid")
    assert F.resolve("pgpid:9007", index=small) == F.Resolution([], "unmatched")


def test_resolve_joins_union_with_weakest_how(index: F.ShelfmarkIndex) -> None:
    """A join gives the union of its parts' ids and the ``how`` of its weakest part.

    :param index: The fixture index.
    """
    res = F.resolve("T-S 16.22 + T-S 10J5.17", index=index)
    assert res == F.Resolution(["Cambridge_CUL_T_S_10J5_17", "Cambridge_CUL_T_S_16_22"], "exact")
    res = F.resolve("T-S 16.22 + Dropsie 472", index=index)
    assert res == F.Resolution(["Cambridge_CUL_T_S_16_22", "Philadelphia_CAJS_Halper_472"], "alias")
    res = F.resolve("T-S 16.22; T-S 99J9.99", index=index)
    assert res == F.Resolution(["Cambridge_CUL_T_S_16_22"], "unmatched")


def test_resolve_pgpid_records_then_fragments(index: F.ShelfmarkIndex) -> None:
    """``pgpid:<n>`` gives the records carrying the PGP id, else its fragment marks' ES ids, else nothing.

    :param index: The fixture index.
    """
    assert F.resolve("pgpid:3455", index=index) == F.Resolution(
        ["Cambridge_CUL_T_S_10J5_17", "Cambridge_CUL_T_S_16_22"], "pgpid")
    small = _mini([{"canonical_id": "New_York_JTS_ENA_2727_5", "pgpids": []}],
                  fragments=[{"shelfmark": "ENA 2727.5", "pgpids": "77;78"}, {"shelfmark": "ENA 1.1", "pgpids": ["79"]}])
    assert F.resolve("pgpid:78", index=small) == F.Resolution(["New_York_JTS_ENA_2727_5"], "pgpid")
    assert F.resolve("pgpid:79", index=small) == F.Resolution([], "unmatched")
    assert F.resolve("", index=small) == F.Resolution([], "unmatched")


def test_record_marks_full_and_slim_forms() -> None:
    """Spellings of a full merged record (any source name) and of a slim record; join strings are skipped."""
    full = {"canonical_id": "New_York_JTS_ENA_2727_5", "shelfmark_display": "ENA 2727.5",
            "sources": {"pgp": {"fragment": {"shelfmark": "ENA 2727.5", "shelfmarks_historic": "ENA 2727, f. 5"}},
                        "legacy": [{"shelf_mark": "New York JTS: ENA 2727.5"},
                                   {"shelf_mark": "New York JTS: ENA 2727.5 - ENA 2727.6"}],
                        "ktiv": {"shelf_mark": "ENA 2727.5", "shelfmarks": {"shelfmark": {"value": "ENA 2727, 5"}}}}}
    assert F.record_marks(full) == ["New_York_JTS_ENA_2727_5", "ENA 2727.5", "ENA 2727, f. 5",
                                    "New York JTS: ENA 2727.5", "ENA 2727, 5"]
    slim = {"canonical_id": "X_1", "shelfmark_display": "X 1", "aliases": ["Old 1"], "source_marks": ["X 1", ""]}
    assert F.record_marks(slim) == ["X_1", "X 1", "Old 1"]


# ------------------------------------------------------------------------------------ cache

def test_index_cache_round_trip_and_invalidation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The pickled index is reused while the file and the normaliser version are unchanged.

    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    merged = tmp_path / "merged.jsonl"
    merged.write_text(INDEX_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    cache = tmp_path / "cache"
    first = F.ShelfmarkIndex(merged, cache_dir=cache)
    files = list(cache.glob("*.pkl"))
    assert len(files) == 1
    stamp = files[0].stat().st_mtime_ns
    again = F.ShelfmarkIndex(merged, cache_dir=cache)
    assert files[0].stat().st_mtime_ns == stamp and again.ids == first.ids and again.loose == first.loose
    monkeypatch.setattr(F, "NORMALISER_VERSION", "test-bump")
    F.ShelfmarkIndex(merged, cache_dir=cache)
    assert files[0].stat().st_mtime_ns != stamp


def test_index_cache_never_under_raw_data(tmp_path: Path) -> None:
    """A cache directory under ``raw_data`` is refused.

    :param tmp_path: Pytest temporary directory.
    """
    merged = tmp_path / "merged.jsonl"
    merged.write_text('{"canonical_id": "Geneva_119"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        F.ShelfmarkIndex(merged, cache_dir=tmp_path / "raw_data" / "cache")


# ------------------------------------------------------------------------------------ moved helpers, cudl

def test_lit_segment_reexports_the_facade_helpers() -> None:
    """``lit_segment`` and ``text27b.shelfmarks`` re-import the helpers the facade now owns."""
    for name in ("split_joins", "split_piece_lists", "legacy_variants", "_clean_sm", "alt_classmark"):
        assert getattr(LS, name) is getattr(F, name)
    assert SM.prep_shelfmark is F.prep_shelfmark


def test_cudl_manifest_matches_the_queue_copy() -> None:
    """The facade's ``cudl_manifest`` and the scrape queue's copy give identical URLs on the probe set."""
    from src.datasets.indexing import arabic_scrape_queue as Q

    known = frozenset(PROBES["cudl_known_fragments"])
    for p in PROBES["probes"]:
        assert F.cudl_manifest(p["mark"], known) == Q.cudl_manifest(p["mark"], known) == p["REF_cudl"], p["id"]
    assert F.CUDL_BASE == Q.CUDL_BASE


# ------------------------------------------------------------------------------------ text27b match_shelfmark

@pytest.fixture()
def corpus() -> SM.Corpus:
    """A synthetic text27b corpus (made-up numbers): two records and one fragment row without a record.

    :return: The corpus.
    :rtype: SM.Corpus
    """
    merged = {"Cambridge_CUL_T_S_99J1_2": {"disp": "T-S 99J1.2", "pgpids": ["9001"]},
              "Oxford_Bodleian_Bodl_MS_heb_d_99_7": {"disp": "Bodl. MS heb. d 99/7", "pgpids": ["9002"]},
              "New_York_JTS_ENA_9999_1": {"disp": "ENA 9999.1", "pgpids": []}}
    frags = [{"shelfmark": "T-S 99J1.2", "pgpids": ["9001"], "historic": ""},
             {"shelfmark": "T-S NS 999.5", "pgpids": ["9004"], "historic": ""}]
    return SM.Corpus(merged, {}, {}, {"9001": {"Cambridge_CUL_T_S_99J1_2"}, "9002": {"Oxford_Bodleian_Bodl_MS_heb_d_99_7"}},
                     {"T_S_NS_999_5": {"9004"}, "T_S_99J1_2": {"9001"}}, {}, {}, [], set(), {}, {}, frags)


def test_match_shelfmark_exact(corpus: SM.Corpus) -> None:
    """An exact mark gives its id and PGP id.

    :param corpus: The synthetic text27b corpus.
    """
    m = SM.match_shelfmark("T-S 99J1.2", corpus)
    assert m["how"] == "exact" and m["pgpids"] == ["9001"] and m["cids"] == ["Cambridge_CUL_T_S_99J1_2"]


def test_match_shelfmark_exact_bodleian(corpus: SM.Corpus) -> None:
    """A Bodleian display mark is exact.

    :param corpus: The synthetic text27b corpus.
    """
    m = SM.match_shelfmark("Bodl. MS heb. d 99/7", corpus)
    assert m["how"] == "exact" and m["pgpids"] == ["9002"]


def test_match_shelfmark_bare_core(corpus: SM.Corpus) -> None:
    """A mark without its series word does not borrow an institution: the series word is part of the core key.

    :param corpus: The synthetic text27b corpus.
    """
    assert F.core_key("99J1.2") != F.core_key("Cambridge_CUL_T_S_99J1_2")
    m = SM.match_shelfmark("99J1.2", corpus)
    assert m["how"] == "unmatched" and m["cids"] == [] and m["pgpids"] == []


def test_match_shelfmark_keeps_fragment_pgp_ids_and_join_siblings() -> None:
    """A record without PGP ids still gets the PGP ids of its ``fragments.csv`` row and the other records of those
    PGP documents (the real corpus misses them on multi-document fragments)."""
    merged = {"Cambridge_CUL_T_S_99J9_1": {"disp": "T-S 99J9.1", "pgpids": []},
              "New_York_JTS_ENA_9999_9": {"disp": "ENA 9999.9", "pgpids": ["9006"]}}
    frags = [{"shelfmark": "T-S 99J9.1", "pgpids": ["9005", "9006"], "historic": ""}]
    corpus = SM.Corpus(merged, {}, {}, {"9006": {"New_York_JTS_ENA_9999_9"}}, {"T_S_99J9_1": {"9005", "9006"}},
                       {}, {}, [], set(), {}, {}, frags)
    m = SM.match_shelfmark("T-S 99J9.1", corpus)
    assert m["how"] == "exact" and m["pgpids"] == ["9005", "9006"]
    assert m["cids"] == ["Cambridge_CUL_T_S_99J9_1", "New_York_JTS_ENA_9999_9"]


def test_match_shelfmark_ambiguous_single_mark() -> None:
    """A single mark naming several records is ``ambiguous`` with no ids, as before PR-6."""
    merged = {"London_BL_Or_9998__sys1": {"disp": "Or. 9998", "pgpids": []},
              "London_BL_Or_9998__sys2": {"disp": "Or. 9998", "pgpids": []}}
    corpus = SM.Corpus(merged, {}, {}, {}, {}, {}, {}, [], set(), {}, {}, [])
    want = {"norm": "BL Or. 9998", "core": SM.core_of("BL Or. 9998"), "how": "ambiguous", "how_detail": "alias",
            "ambiguous": True, "cids": [], "pgpids": []}
    assert SM.match_shelfmark("BL Or. 9998", corpus) == want


def test_match_shelfmark_pgp_fragment(corpus: SM.Corpus) -> None:
    """A fragment row with no ES record gives its PGP ids and no id.

    :param corpus: The synthetic text27b corpus.
    """
    m = SM.match_shelfmark("T-S NS 999.5", corpus)
    assert m["how"] == "pgp_fragment" and m["pgpids"] == ["9004"] and m["cids"] == []


def test_match_shelfmark_junk(corpus: SM.Corpus) -> None:
    """A string without a usable core is junk.

    :param corpus: The synthetic text27b corpus.
    """
    assert SM.match_shelfmark("Recto", corpus)["how"] == "junk"


def test_match_shelfmark_unmatched(corpus: SM.Corpus) -> None:
    """A mark the corpus does not hold is unmatched.

    :param corpus: The synthetic text27b corpus.
    """
    m = SM.match_shelfmark("T-S 98J1.1", corpus)
    assert m["how"] == "unmatched" and m["cids"] == [] and m["pgpids"] == []


def test_match_shelfmark_how_keeps_the_pre_pr6_vocabulary() -> None:
    """``how`` stays in the pre-PR-6 names (``rag_l3`` counts them); ``how_detail`` gives the facade step."""
    merged = {"Philadelphia_CAJS_Halper_9472": {"disp": "Dropsie 9472", "pgpids": []},
              "Cambridge_CUL_T_S_F9_74b": {"disp": "", "pgpids": []}}
    small = SM.Corpus(merged, {}, {}, {}, {}, {}, {}, [], set(), {}, {}, [])
    m = SM.match_shelfmark("Dropsie 9472", small)
    assert (m["how"], m["how_detail"], m["cids"]) == ("core", "alias", ["Philadelphia_CAJS_Halper_9472"])
    m = SM.match_shelfmark("T-S F9.74B", small)
    assert (m["how"], m["how_detail"], m["cids"]) == ("core_compact", "loose", ["Cambridge_CUL_T_S_F9_74b"])
    assert set(SM._HOW_TO_LEGACY) == set(F.HOW_ORDER)


def test_match_shelfmark_range_reproduces_the_pre_pr6_result() -> None:
    """A numeric range gets exactly the pre-PR-6 result (Q11 open), never the first leaf of the range."""
    merged = {"Cambridge_CUL_T_S_10J5_17": {"disp": "T-S 10J5.17", "pgpids": ["9101"]},
              "Cambridge_CUL_T_S_10J5_17_1": {"disp": "T-S 10J5.17.1", "pgpids": []},
              "New_York_JTS_ENA_9846_3_4": {"disp": "ENA 9846.3-4", "pgpids": ["9102"]}}
    core_idx = {"T_S_10J5_17": {"Cambridge_CUL_T_S_10J5_17"}, "T_S_10J5_17_1": {"Cambridge_CUL_T_S_10J5_17_1"},
                "ENA_9846_3_4": {"New_York_JTS_ENA_9846_3_4"}}
    corpus = SM.Corpus(merged, core_idx, {k.lower(): v for k, v in core_idx.items()},
                       {"9101": {"Cambridge_CUL_T_S_10J5_17"}, "9102": {"New_York_JTS_ENA_9846_3_4"}},
                       {}, {}, {}, [], set(), {}, {}, [])
    for mark in ("T-S 10J5.17-18", "ENA 9846.3-4", "ff. 9-10"):
        m = SM.match_shelfmark(mark, corpus)
        head = SM._match_shelfmark_head(mark, corpus)
        assert {k: m[k] for k in head} == head and m["how_detail"] == "range", mark
        assert m["ambiguous"] is False
    assert SM.match_shelfmark("T-S 10J5.17-18", corpus)["cids"] == []
    assert SM.match_shelfmark("ENA 9846.3-4", corpus)["cids"] == ["New_York_JTS_ENA_9846_3_4"]


def test_match_shelfmark_ambiguous_join() -> None:
    """A join with a part that names several records is ``ambiguous`` with no ids, like a single mark."""
    merged = {"London_BL_Or_9998__sys1": {"disp": "Or. 9998", "pgpids": ["9201"]},
              "London_BL_Or_9998__sys2": {"disp": "Or. 9998", "pgpids": []},
              "Cambridge_CUL_T_S_99J1_2": {"disp": "T-S 99J1.2", "pgpids": ["9001"]}}
    corpus = SM.Corpus(merged, {}, {}, {"9001": {"Cambridge_CUL_T_S_99J1_2"}}, {}, {}, {}, [], set(), {}, {}, [])
    m = SM.match_shelfmark("T-S 99J1.2 + BL Or. 9998", corpus)
    assert (m["how"], m["ambiguous"], m["cids"], m["pgpids"]) == ("ambiguous", True, [], [])
    m = SM.match_shelfmark("T-S 99J1.2", corpus)
    assert (m["how"], m["ambiguous"], m["cids"]) == ("exact", False, ["Cambridge_CUL_T_S_99J1_2"])


def test_match_shelfmark_one_record_is_not_expanded_by_fragment_links() -> None:
    """A mark naming one record keeps the pre-PR-6 ids: the index's fragment-linked PGP ids (a historic spelling
    in ``fragments.csv``) do not add the other records of that PGP id; they do for a mark already naming several."""
    merged = {"Cambridge_CUL_T_S_99J9_1": {"disp": "T-S 99J9.1", "pgpids": []},
              "New_York_JTS_ENA_9999_9": {"disp": "ENA 9999.9", "pgpids": ["9006"]},
              "New_York_JTS_ENA_9999_8": {"disp": "ENA 9999.8", "pgpids": ["9007"]},
              "New_York_JTS_ENA_9999_7": {"disp": "ENA 9999.7", "pgpids": ["9007"]}}
    frags = [{"shelfmark": "T-S 99J9.1", "pgpids": ["9006"], "historic": ""},
             {"shelfmark": "ENA 9999.8", "pgpids": ["9008"], "historic": ""},
             {"shelfmark": "T-S 99J9.5", "pgpids": ["9008"], "historic": ""}]
    p2c = {"9006": {"New_York_JTS_ENA_9999_9"}, "9007": {"New_York_JTS_ENA_9999_8", "New_York_JTS_ENA_9999_7"}}
    corpus = SM.Corpus(merged, {}, {}, p2c, {}, {}, {}, [], set(), {}, {}, frags)
    idx = SM.corpus_index(corpus)
    assert idx.pgpids["Cambridge_CUL_T_S_99J9_1"] == ["9006"]
    m = SM.match_shelfmark("T-S 99J9.1", corpus)
    assert (m["cids"], m["pgpids"]) == (["Cambridge_CUL_T_S_99J9_1"], [])
    # a mark that already names several records (PGP 9007) gains the fragment links (9008 of ENA 9999.8)
    m = SM.match_shelfmark("ENA 9999.7", corpus)
    assert m["cids"] == ["New_York_JTS_ENA_9999_7", "New_York_JTS_ENA_9999_8"] and m["pgpids"] == ["9007", "9008"]


def test_corpus_index_cache_keeps_no_corpus_alive() -> None:
    """The ``corpus_index`` cache holds a weak reference only: a corpus dropped by its caller is freed (and its
    cached index with it) even when no other corpus is indexed; a stand-in without weak references is not cached."""
    import gc
    import weakref
    from types import SimpleNamespace

    def make() -> SM.Corpus:
        """A one-record corpus.

        :return: The corpus.
        :rtype: SM.Corpus
        """
        return SM.Corpus({"Cambridge_CUL_T_S_99J1_2": {"disp": "T-S 99J1.2", "pgpids": []}}, {}, {}, {}, {}, {}, {},
                         [], set(), {}, {}, [])

    first = make()
    index = SM.corpus_index(first)
    assert SM.corpus_index(first) is index
    corpus_ref, index_ref = weakref.ref(first), weakref.ref(index)
    del first, index
    gc.collect()
    assert corpus_ref() is None and index_ref() is None
    assert SM._INDEX_CACHE == {}
    second, third = make(), make()
    SM.corpus_index(second)
    SM.corpus_index(third)
    assert SM._INDEX_CACHE["ref"]() is third and "corpus" not in SM._INDEX_CACHE
    stand_in = SimpleNamespace(merged={"Cambridge_CUL_T_S_99J1_2": {"disp": "T-S 99J1.2", "pgpids": []}}, fragments=[])
    assert SM.corpus_index(stand_in).ids == {"Cambridge_CUL_T_S_99J1_2"}
    assert SM._INDEX_CACHE["ref"]() is third


def test_resolve_a_records_own_spelling_and_historic_marks() -> None:
    """A record's own display form reaches it although its id has no registry token; a ``fragments`` row's historic
    marks reach the row's record (a weak-looking ``L 999`` too); a spelling two records share stays unmatched."""
    records = [{"canonical_id": "Example_Collection_Haifa_Israel_Ms_9188",
                "shelfmark_display": "Example Collection, Haifa, Israel Ms. 9188"},
               {"canonical_id": "Manchester_JRL_L_999", "shelfmark_display": "JRL L 999"},
               {"canonical_id": "Twin_A_Ms_77", "shelfmark_display": "Twin Holder Ms. 77"},
               {"canonical_id": "Twin_B_Ms_77", "shelfmark_display": "Twin Holder Ms. 77"}]
    plain = _mini(records)
    own = F.Resolution(["Example_Collection_Haifa_Israel_Ms_9188"], "alias")
    assert F.resolve("Example Collection, Haifa, Israel Ms. 9188", index=plain) == own
    assert F.resolve("example collection,  haifa, israel ms. 9188", index=plain) == own
    for mark in ("L 999", "Twin Holder Ms. 77", "Genizah 999", "Example Collection"):
        assert F.resolve(mark, index=plain) == F.Resolution([], "unmatched"), mark
    linked = _mini(records, fragments=[{"shelfmark": "Manchester: L 999", "pgpids": "9401",
                                        "historic": "L 999; Gaster 999b"}])
    for mark in ("L 999", "Gaster 999b"):
        assert F.resolve(mark, index=linked) == F.Resolution(["Manchester_JRL_L_999"], "alias"), mark
    assert F.resolve("Genizah 999", index=linked) == F.Resolution([], "unmatched")
    # a historic mark whose core is another record's (``L 999`` of a Mosseri fragment vs the JRL id) names neither
    clash = _mini(records + [{"canonical_id": "Cambridge_Mosseri_II_999", "shelfmark_display": "Moss. II,999"}],
                  fragments=[{"shelfmark": "Moss. II,999", "pgpids": "9403", "historic": "L 999"}])
    assert F.resolve("L 999", index=clash) == F.Resolution([], "unmatched")
    assert F.resolve("Moss. II,999", index=clash).cids == ["Cambridge_Mosseri_II_999"]


def test_match_shelfmark_own_display_and_historic_mark() -> None:
    """``match_shelfmark`` reaches a record by its own display form and by its PGP historic mark, as before PR-6."""
    merged = {"Example_Collection_Haifa_Israel_Ms_9188": {"disp": "Example Collection, Haifa, Israel Ms. 9188",
                                                          "pgpids": ["9402"]},
              "Manchester_JRL_L_999": {"disp": "Manchester: L 999", "pgpids": ["9401"]}}
    frags = [{"shelfmark": "Manchester: L 999", "pgpids": ["9401"], "historic": "L 999"}]
    corpus = SM.Corpus(merged, {}, {}, {"9401": {"Manchester_JRL_L_999"}}, {}, {}, {}, [], set(), {}, {}, frags)
    m = SM.match_shelfmark("Example Collection, Haifa, Israel Ms. 9188", corpus)
    assert (m["how"], m["cids"], m["pgpids"]) == ("core", ["Example_Collection_Haifa_Israel_Ms_9188"], ["9402"])
    m = SM.match_shelfmark("L 999", corpus)
    assert (m["how"], m["cids"], m["pgpids"]) == ("core", ["Manchester_JRL_L_999"], ["9401"])


def test_match_shelfmark_core_lc_keeps_fragment_pgp_ids() -> None:
    """A mark whose core differs from the record's only by case is ``core_lc`` (as before PR-6) and keeps the PGP
    ids of its ``fragments.csv`` core."""
    cid = "Cambridge_CUL_T_S_H99_11a"
    frags = [{"shelfmark": "T-S H99.11A", "pgpids": ["9301"], "historic": ""}]
    corpus = SM.Corpus({cid: {"disp": "T-S H99.11a", "pgpids": []}}, {"T_S_H99_11a": {cid}}, {"t_s_h99_11a": {cid}},
                       {}, {"T_S_H99_11A": {"9301"}}, {}, {}, [], set(), {}, {}, frags)
    m = SM.match_shelfmark("T-S H99.11A", corpus)
    assert (m["how"], m["how_detail"], m["cids"], m["pgpids"]) == ("core_lc", "loose", [cid], ["9301"])
    head = SM._match_shelfmark_head("T-S H99.11A", corpus)
    assert (head["how"], head["cids"], head["pgpids"]) == ("core_lc", [cid], ["9301"])
