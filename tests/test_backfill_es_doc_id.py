"""Candidate ordering and write decisions of ``backfill_es_doc_id``."""

from collections import Counter

from src.datasets.indexing.neo4j import backfill_es_doc_id as bf

BY_PGPID = {448: "Cambridge_Mosseri_IV_130_2", 472: "Cambridge_CUL_T_S_10J8_9"}
BY_CANON = {
    "T_S_20_169": "Cambridge_CUL_T_S_20_169",
    "T_S_10J8_9": "Cambridge_CUL_T_S_10J8_9",
    "VII_A_34": "Paris_AIU_VII_A_34",
    "T_S_99_1": "Cambridge_CUL_T_S_99_1",   # in the JSONL but NOT in the served index
}


def test_first_join_component_wins_then_pgpid_then_full_mark():
    cands = bf.candidate_ids("T_S_20_169_+_T_S_10J8_9", 472, BY_PGPID, BY_CANON)
    assert cands == ["Cambridge_CUL_T_S_20_169", "Cambridge_CUL_T_S_10J8_9"]


def test_pgpid_only_fragment_resolves_and_none_when_unknown():
    assert bf.candidate_ids("Mosseri_IV_14_2_+_AIU_VII_E_119", 448, BY_PGPID, BY_CANON) == \
        ["Cambridge_Mosseri_IV_130_2"]
    assert bf.candidate_ids("Unknown_1", None, BY_PGPID, BY_CANON) == []
    assert bf.candidate_ids(None, None, BY_PGPID, BY_CANON) == []


def test_resolve_fills_nulls_keeps_existing_and_respects_es_verification():
    frags = [
        {"c": "VII_A_34", "p": None, "e": None},                       # set
        {"c": "T_S_20_169", "p": None, "e": "Cambridge_CUL_T_S_20_169"},  # same
        {"c": "T_S_10J8_9", "p": None, "e": "Stale_Id"},               # kept unless overwrite
        {"c": "Nope", "p": None, "e": None},                           # unresolved
        {"c": "T_S_99_1_+_T_S_10J8_9", "p": 472, "e": None},           # first comp not in ES → pgpid
    ]
    existing = {"Paris_AIU_VII_A_34", "Cambridge_CUL_T_S_20_169", "Cambridge_CUL_T_S_10J8_9"}
    rows, stats = bf.resolve(frags, BY_PGPID, BY_CANON, existing, overwrite=False)
    assert rows == [{"c": "VII_A_34", "e": "Paris_AIU_VII_A_34"},
                    {"c": "T_S_99_1_+_T_S_10J8_9", "e": "Cambridge_CUL_T_S_10J8_9"}]
    assert stats == Counter(set=2, already_set_same=1, already_set_different_kept=1,
                            unresolved=1, candidates_not_in_es=1)

    rows, stats = bf.resolve(frags, BY_PGPID, BY_CANON, existing, overwrite=True)
    assert {"c": "T_S_10J8_9", "e": "Cambridge_CUL_T_S_10J8_9"} in rows
    assert stats["overwritten"] == 1


def test_resolve_without_verification_takes_first_candidate():
    frags = [{"c": "T_S_20_169_+_T_S_10J8_9", "p": 472, "e": None}]
    rows, _ = bf.resolve(frags, BY_PGPID, BY_CANON, None, overwrite=False)
    assert rows == [{"c": "T_S_20_169_+_T_S_10J8_9", "e": "Cambridge_CUL_T_S_20_169"}]
