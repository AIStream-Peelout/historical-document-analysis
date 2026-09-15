"""Compound (joint-holdings) institution names must survive normalisation.

The June 2026 graph carried ``Cambridge University Library / Bodleian Library
Oxford`` (the Lewis-Gibson collection, 1,313 fragments) as its own node. The
2026-07-18 ingest normalisation folded it onto ``Cambridge University Library``
in the 2026-09-15 rebuild — the merge ``docs/kg_pipeline_runbook.md`` §5
explicitly forbids. These tests pin the guard.
"""

from src.datasets.document_models.institution_normalizer import InstitutionNormalizer as N

LG = "Cambridge University Library / Bodleian Library Oxford"


def test_compound_holdings_name_is_kept_verbatim():
    assert N.normalize(LG) == LG
    assert N.is_compound(LG)


def test_members_still_normalise_on_their_own():
    assert N.normalize("Cambridge University Library") == "Cambridge University Library"
    assert N.normalize("Bodleian Library Oxford") == "Bodleian Library, Oxford"


def test_two_spellings_of_one_institution_are_not_compound():
    # Both halves resolve to the same canonical → ordinary normalisation.
    both = "Bodleian Library Oxford / Bodleian Library, Oxford"
    assert not N.is_compound(both)
    assert N.normalize(both) == "Bodleian Library, Oxford"
    assert not N.is_compound("")
    assert not N.is_compound("T-S / misc")
