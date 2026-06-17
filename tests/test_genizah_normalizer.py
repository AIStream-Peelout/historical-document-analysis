"""Regression tests for the hardened shelfmark normalizer.

Covers the cross-source canonicalisation behaviours added for the PGP/FJP/KTIV
merge: institution-prefix stripping (incl. KTIV "Ms." and compound FJP
prefixes), Taylor-Schechter class-mark spacing collapse, folio-designator
removal, the British Museum -> British Library rename, and equivalence across
the three source spellings of the same fragment.
"""

import pytest

from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer as SN


# ── institution prefix stripping ──────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("T-S 10J5.6", "T_S_10J5_6"),
    ("Cambridge CUL: T-S 10J5.6", "T_S_10J5_6"),
    ("New York JTS: ENA 4020.50", "ENA_4020_50"),
    ("Oxford: MS heb. b.18/2", "MS_heb_b_18_2"),
    ("Cambridge University Library, Cambridge, England Ms. T-S 20.113", "T_S_20_113"),
])
def test_institution_prefixes_stripped(raw, expected):
    assert SN.to_canonical_id(raw) == expected


# ── Taylor-Schechter class-mark spacing collapse ──────────────────────────────

@pytest.mark.parametrize("spaced,closed", [
    ("T-S 10 J 5.6", "T-S 10J5.6"),
    ("T-S 8 K 20.2", "T-S 8K20.2"),
    ("T-S K 17.1", "T-S K17.1"),
    ("T-S J 1.1", "T-S J1.1"),
])
def test_ts_classmark_spacing_collapses(spaced, closed):
    assert SN.to_canonical_id(spaced) == SN.to_canonical_id(closed)


@pytest.mark.parametrize("raw,expected", [
    # Multi-letter subseries must NOT be collapsed into the box notation.
    ("T-S AS 100.2", "T_S_AS_100_2"),
    ("T-S NS 318.6", "T_S_NS_318_6"),
    ("T-S Misc.28.199", "T_S_Misc_28_199"),
])
def test_subseries_not_collapsed(raw, expected):
    assert SN.to_canonical_id(raw) == expected


# ── folio designators and institution rename ──────────────────────────────────

def test_folio_designator_dropped_number_kept():
    assert SN.to_canonical_id("BM Or 10110, f. 23") == "BL_Or_10110_23"
    assert SN.to_canonical_id("BM Or 10110, fol. 9") == "BL_Or_10110_9"


def test_bm_renamed_to_bl():
    assert SN.are_equivalent("BM Or 10110.23", "BL OR 10110.23")


@pytest.mark.parametrize("jrl,manchester", [
    ("JRL C 121", "Manchester: C 121"),
    ("JRL B 6785", "Manchester: B 6785"),
    ("JRL A 1", "Manchester: A 1"),
    ("JRL Gaster 1389", "Manchester: Gaster 1389"),
    ("JRL Genizah 836", "Manchester: Genizah 836"),
])
def test_jrl_equals_manchester(jrl, manchester):
    # John Rylands Library (PGP "JRL") is the same as Manchester (FJP/KTIV).
    assert SN.to_canonical_id(jrl) == SN.to_canonical_id(manchester)
    assert SN.to_canonical_id(jrl).startswith("JRL_")


def test_jrl_ktiv_full_string():
    ktiv = "The University of Manchester Library, Manchester, England Ms. B 3567"
    assert SN.to_canonical_id(ktiv) == SN.to_canonical_id("JRL B 3567")


def test_single_collection_letter_survives_folio_rule():
    # "L" (Lewis-Gibson) and "P"/"C" (Gaster) must not be mistaken for folio
    # designators, which require a trailing dot.
    assert SN.to_canonical_id("L-G Ar.I.130") == "L_G_Ar_I_130"


# ── cross-source equivalence (the whole point of the merge) ───────────────────

def test_three_sources_same_fragment_equivalent():
    pgp = "T-S 10J5.6"
    fjp = "Cambridge CUL: T-S 10J5.6"
    ktiv = "Cambridge University Library, Cambridge, England Ms. T-S 10 J 5.6"
    canonical = SN.to_canonical_id(pgp)
    assert SN.to_canonical_id(fjp) == canonical
    assert SN.to_canonical_id(ktiv) == canonical


def test_empty_and_none_safe():
    assert SN.to_canonical_id("") == ""
    assert SN.to_canonical_id(None) == ""
