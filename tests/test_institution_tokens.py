"""Tests for descriptive institution tokens in canonical shelfmark ids.

The canonical id must always carry an institution token (never a bare number
that collides across collections), and the *same* fragment must produce the
*same* id whether it comes from PGP, FJP or KTIV.
"""

import pytest

from src.datasets.merging.institution_tokens import (
    canonical_id,
    combine,
    institution_token,
    resolve_token,
)


@pytest.mark.parametrize("text,token", [
    ("CUL Cambridge University Library", "Cambridge_CUL"),
    ("Cambridge CUL", "Cambridge_CUL"),
    ("Cambridge University Library", "Cambridge_CUL"),
    ("Mosseri", "Cambridge_Mosseri"),
    ("Cambridge Lewis-Gibson", "Cambridge_Lewis_Gibson"),
    ("CUL & Bodl.", "Cambridge_Lewis_Gibson"),
    ("New York JTS", "New_York_JTS"),
    ("JTS Jewish Theological Seminary Library", "New_York_JTS"),
    ("Manchester", "Manchester_JRL"),
    ("JRL John Rylands Library", "Manchester_JRL"),
    ("Paris AIU", "Paris_AIU"),
    ("AIU Alliance Israélite Universelle", "Paris_AIU"),
    ("Budapest MTA", "Budapest_MTA"),
    ("Geneva", "Geneva"),
    ("Cincinnati HUC", "Cincinnati_HUC"),
])
def test_resolve_token(text, token):
    assert resolve_token(text) == token


def test_unmapped_falls_back_to_slug_not_empty():
    # Never returns empty / numeric — an unknown institution still gets a token.
    assert institution_token("Some Tiny Private Collection") == "Some_Tiny_Private_Collection"
    assert institution_token("") == "Unknown"


def test_combine_dedupes_overlap():
    assert combine("Manchester_JRL", "JRL_B_1924") == "Manchester_JRL_B_1924"
    assert combine("Geneva", "Geneva_119") == "Geneva_119"
    assert combine("Cambridge_CUL", "T_S_10J5_6") == "Cambridge_CUL_T_S_10J5_6"
    assert combine("Cincinnati_HUC", "1002") == "Cincinnati_HUC_1002"


def test_no_bare_numeric_id():
    # The original bug: "Budapest MTA: 101" must not become bare "101".
    cid = canonical_id("Budapest MTA", "Budapest MTA: 101")
    assert cid == "Budapest_MTA_101"
    assert not cid[0].isdigit()


@pytest.mark.parametrize("rows", [
    # (institution_text, shelfmark) spellings of the SAME fragment per source.
    [("CUL Cambridge University Library", "T-S 10J5.6"),
     ("Cambridge CUL", "Cambridge CUL: T-S 10J5.6"),
     ("Cambridge University Library",
      "Cambridge University Library, Cambridge, England Ms. T-S 10 J 5.6")],
    [("JTS Jewish Theological Seminary Library", "ENA 1055.27"),
     ("New York JTS", "New York JTS: ENA 1055.27")],
    [("AIU Alliance Israélite Universelle", "AIU VII.E.5"),
     ("Paris AIU", "Paris AIU: VII.E.5")],
    [("Geneva", "Geneva 119"), ("Geneva", "Geneva: 119")],
])
def test_same_fragment_same_id_across_sources(rows):
    ids = {canonical_id(text, sm) for text, sm in rows}
    assert len(ids) == 1


def test_different_collections_same_number_do_not_collide():
    a = canonical_id("Budapest MTA", "Budapest MTA: 101")
    b = canonical_id("Geneva", "Geneva: 101")
    c = canonical_id("Frankfurt", "Frankfurt: 101")
    assert len({a, b, c}) == 3
