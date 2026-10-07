"""Institution-token and id-style cleanups found in the pre-v8 check of new KTIV records.

* The JTS needle ``ena`` matched inside ``Menachem`` / ``Benayahu`` / ``Modena``;
  needles now match whole words, and genuine ENA shelfmarks still resolve to JTS.
* The ``tel aviv`` city needle filed the Gross, Einhorn and Eretz Israel Museum
  collections under Tel Aviv University; ``st. petersburg`` filed the Institute
  of Oriental Manuscripts under the NLR; Budapest's "Jewish Theological Seminary
  - University of Jewish Studies" was filed under the New York JTS.
* KTIV British Library marks (``The British Library, London, England Or. …``)
  leaked the location head into the id; they now take PGP's ``London_BL_Or_…``.
  A mark naming no leaf (``Or. 10129``) is keyed per item, so it neither
  collapses with other items in the volume nor rides a PGP leaf alias.
* KTIV "Unknown Library" items all collapsed onto ``Unknown_Library``; they are
  keyed by their single former-owner shelfmark, else per item.
* Unambiguous institutions that were slugs get tokens.
"""

import re
from typing import Optional

import pytest

from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer as SN
from src.datasets.merging import merge_shelfmarks as M
from src.datasets.merging.institution_tokens import (
    _REGISTRY,
    _needle_pattern,
    canonical_id,
    resolve_token,
)

JTS_HEAD = "The Jewish Theological Seminary of America, New York, NY, USA Ms. "
BL_HEAD = "The British Library, London, England "
BL_PGP = {"library_abbrev": "BL", "library": "British Library"}


def _ktiv(mark: str, sys_num: str = "990000000000205171", **doc: object) -> str:
    """KTIV canonical id for a single-shelfmark record.

    :param mark: The full KTIV ``shelf_mark``.
    :param sys_num: KTIV system number.
    :param doc: Extra KTIV record fields (``shelfmarks`` …).
    :returns: The canonical id the merge assigns (before PGP aliasing).
    """
    record = {"shelf_mark": mark, "sys_num": sys_num, **doc}
    (_, cid, _), = M.ktiv_record_ids(record, [mark])
    return cid


def _unknown(additional: Optional[str] = None, sys_num: str = "990001341400205171") -> str:
    """KTIV canonical id for an "Unknown Library" item.

    :param additional: ``shelfmarks.additional`` (former owner shelfmark), or ``None``.
    :param sys_num: KTIV system number.
    :returns: The canonical id the merge assigns.
    """
    shelfmarks = {"shelfmark": {"value": "Unknown Library"}}
    if additional is not None:
        shelfmarks["additional"] = additional
    return _ktiv("Unknown Library", sys_num=sys_num, shelfmarks=shelfmarks)


# ── 1. whole-word needles: "ena" is not inside Menachem / Benayahu / Modena ──

@pytest.mark.parametrize("mark,expected", [
    ("Feldman, Menachem, Jerusalem, Israel Ms. 147",
     "Feldman_Menachem_Jerusalem_Israel_Ms_147"),
    ("Benayahu, Meir, Jerusalem, Israel Ms. TEU 17.2",
     "Benayahu_Meir_Jerusalem_Israel_Ms_TEU_17_2"),
    ("State Archives of Modena, Modena, Italy Ms. 184", "Modena_State_Archives_184"),
    ("Capitular Archives of Modena, Modena, Italy Ms. 88", "Modena_Capitular_Archives_88"),
])
def test_ena_inside_a_word_is_not_jts(mark, expected):
    assert resolve_token(mark) != "New_York_JTS"
    assert _ktiv(mark) == expected


@pytest.mark.parametrize("cid,expected", [
    (canonical_id("JTS Jewish Theological Seminary Library", "ENA 2808.59"),
     "New_York_JTS_ENA_2808_59"),
    (canonical_id("JTS Jewish Theological Seminary Library", "ENA NS 5.29"),
     "New_York_JTS_ENA_NS_5_29"),
    (M.fjp_canonical_id("New York JTS: ENA 2808.59", {"collection": "New York JTS"}),
     "New_York_JTS_ENA_2808_59"),
    (M.fjp_canonical_id("ENA NS 5.29", {"collection": "", "institution": "Unknown"}),
     "New_York_JTS_ENA_NS_5_29"),
    (M.fjp_canonical_id("JTS ENA 1055.27", {}), "New_York_JTS_ENA_1055_27"),
    (_ktiv(JTS_HEAD + "ENA 2808.59"), "New_York_JTS_ENA_2808_59"),
    (_ktiv(JTS_HEAD + "ENA NS 5.29"), "New_York_JTS_ENA_NS_5_29"),
])
def test_genuine_ena_shelfmarks_still_resolve_to_jts(cid, expected):
    assert cid == expected


@pytest.mark.parametrize("text", [
    "ENA 2808.59", "ENA NS 5.29", "JTS ENA", "Ms. ENA 1205.55", "ENA2808.59",
])
def test_ena_needle_matches_as_a_token(text):
    assert resolve_token(text) == "New_York_JTS"


@pytest.mark.parametrize("needle,text,hit", [
    ("ena", "menachem", False),
    ("ena", "ms. ena 1", True),
    ("t-s", "t-s10j5.6", True),          # a digit is a boundary
    ("moss.", "moss.ii 1", True),        # no boundary after the needle's own "."
    ("cul", "faculty of arts", False),
])
def test_needle_pattern_whole_word(needle, text, hit):
    assert bool(re.search(_needle_pattern(needle), text)) is hit


def test_registry_tokens_unique_and_needles_lowercase():
    tokens = [token for token, _ in _REGISTRY]
    assert len(tokens) == len(set(tokens))
    assert all(n == n.lower() for _, needles in _REGISTRY for n in needles)


# ── 2. Gross (and the other Tel Aviv / St Petersburg / Budapest misfiles) ───

@pytest.mark.parametrize("mark,expected", [
    ("Gross, William L., Tel Aviv, Israel Ms. MO.011.090", "TelAviv_Gross_MO_11_90"),
    ("Einhorn, Isaac, Tel Aviv, Israel Ms. 38", "TelAviv_Einhorn_38"),
    ("Eretz Israel Museum, Tel Aviv, Israel Ms. 33", "TelAviv_Eretz_Israel_Museum_33"),
    ("Institute of Oriental Manuscripts, the Russian Academy of Sciences, "
     "St. Petersburg, Russia Ms. A 69", "StPetersburg_IOM_A_69"),
    ("The Jewish Theological Seminary - University of Jewish Studies, Budapest, "
     "Hungary Ms. K 177", "Budapest_Rabbinical_Seminary_K_177"),
])
def test_misfiled_ktiv_collections_get_their_own_token(mark, expected):
    assert _ktiv(mark) == expected


@pytest.mark.parametrize("cid,expected", [
    (_ktiv("Tel Aviv University Library, Tel Aviv, Israel Ms. 7"), "TelAviv_TAU_7"),
    (canonical_id("TAU Sourasky Central Library, Tel Aviv University", "TAU 1"),
     "TelAviv_TAU_1"),
    (M.fjp_canonical_id("Tel Aviv: 16", {"collection": "Tel Aviv", "institution": "Tel Aviv"}),
     "TelAviv_TAU_16"),
    (_ktiv("The National Library of Russia, St. Petersburg, Russia Ms. EVR ARAB II 986"),
     "StPetersburg_NLR_EVR_ARAB_II_986"),
    (canonical_id("SPIOS Institute of Oriental Manuscripts, Russian Academy of Sciences",
                  "D 55"), "StPetersburg_IOM_D_55"),
    (_ktiv(JTS_HEAD + "Lutzki 515, fol. 30"), "New_York_JTS_Lutzki_515_30"),
])
def test_genuine_neighbours_unchanged(cid, expected):
    assert cid == expected


# ── 3. KTIV British Library marks take PGP's London_BL_Or_… style ──────────

@pytest.mark.parametrize("ktiv_core,pgp_shelfmark", [
    ("Or. 10110.23", "BL OR 10110.23"),
    ("Or. 12369.24", "BL OR 12369.24"),
    ("Or. 5543.5", "BL OR 5543.5"),
    ("Or. 5557K.11", "BL Or. 5557K.11"),
    ("Or. 5566D.24", "BL OR 5566D.24"),
])
def test_ktiv_bl_joins_its_pgp_twin(ktiv_core, pgp_shelfmark):
    pgp_cid = M.pgp_canonical_id({**BL_PGP, "shelfmark": pgp_shelfmark})[1]
    assert _ktiv(BL_HEAD + ktiv_core) == pgp_cid
    assert pgp_cid.startswith("London_BL_Or_")


@pytest.mark.parametrize("ktiv_core,expected", [
    ("Or. 5557O.43", "London_BL_Or_5557O_43"),
    ("Ms. Or. 1.2", "London_BL_Or_1_2"),
    # Volume / folder only: keyed per item (see the tests below).
    ("Add. 27002", "London_BL_Add_27002__sys990000000000205171"),
    ("Harley 1204", "London_BL_Harley_1204__sys990000000000205171"),
    ("Or. 5557C", "London_BL_Or_5557C__sys990000000000205171"),
    ("Ms. Or. 1", "London_BL_Or_1__sys990000000000205171"),
])
def test_ktiv_bl_head_never_leaks_into_the_id(ktiv_core, expected):
    assert _ktiv(BL_HEAD + ktiv_core) == expected


def test_ktiv_bl_volume_only_items_do_not_collapse():
    # KTIV files distinct Or. 10124 fragments under the bare volume.
    a = _ktiv(BL_HEAD + "Or. 10124", sys_num="990053523540205171")
    b = _ktiv(BL_HEAD + "Or. 10124", sys_num="990053523620205171")
    assert a == "London_BL_Or_10124__sys990053523540205171"
    assert a != b


def test_ktiv_bl_volume_only_is_not_captured_by_a_pgp_leaf_alias():
    # PGP lists "BL OR 10129" as a historic spelling of "BL OR 10129.1–25" (a
    # Zohar); KTIV's Or. 10129 is a deed fragment from somewhere in the volume.
    row = {**BL_PGP, "shelfmark": "BL OR 10129.1–25", "shelfmarks_historic": "BL OR 10129"}
    token, primary = M.pgp_canonical_id(row)
    accepted, _ = M.pgp_variant_aliases(token, primary, row["shelfmark"], ["BL OR 10129"])
    alias = {primary: primary, **{v: primary for v in accepted}}
    assert alias["London_BL_Or_10129"] == "London_BL_Or_10129_1_25"
    cid = _ktiv(BL_HEAD + "Or. 10129", sys_num="990036962610205171")
    assert alias.get(cid, cid) == "London_BL_Or_10129__sys990036962610205171"


def test_ktiv_bl_volume_only_without_sys_num_keeps_the_volume_id():
    mark = BL_HEAD + "Or. 10129"
    assert M.ktiv_mark_cid({"shelf_mark": mark}, mark) == ("London_BL_Or_10129", None)


def test_bl_normaliser_core_matches_pgp_and_historic_bm():
    ktiv = SN.to_canonical_id(BL_HEAD + "Or. 10110.23")
    assert ktiv == SN.to_canonical_id("BL OR 10110.23")
    assert ktiv == SN.to_canonical_id("BM Or 10110, f. 23") == "BL_Or_10110_23"


def test_ktiv_bl_merged_institution_is_still_the_british_library():
    # The fix lives in to_canonical_id only: the display's institution lookup
    # must not start reading "Or." as Cambridge's CUL Or. series.
    ktiv = {"sys_num": "990053561090205171", "shelf_mark": BL_HEAD + "Or. 10110.23"}
    rec = M.build_merged_record("London_BL_Or_10110_23", None, [], ktiv, {}, {})
    assert rec["institution"] == "The British Library"


# ── 4. "Unknown Library" items and newly registered institutions ─────────────

@pytest.mark.parametrize("additional,expected", [
    ("Sassoon, David Solomon, London, England Ms. 524", "London_Sassoon_524"),
    ("Sassoon, David Solomon, London, England Ms. 713", "London_Sassoon_713"),
    ("Mehlman, Israel, Jerusalem, Israel Ms. 103", "Mehlman_Israel_Jerusalem_Israel_Ms_103"),
    ("Wallach, Isaac, Israel Ms. 340", "Wallach_Isaac_Israel_Ms_340"),
])
def test_unknown_library_keyed_by_former_owner(additional, expected):
    assert _unknown(additional) == expected


def test_unknown_library_items_never_share_an_id():
    ids = {
        _unknown("Sassoon, David Solomon, London, England Ms. 216", "990001303940205171"),
        _unknown("Sassoon, David Solomon, London, England Ms. 527", "990001341430205171"),
        _unknown("Shapira, Bernard, Jerusalem, Israel Ms. 1* Benayahu, Meir, Jerusalem, "
                 "Israel Ms. TEU 91", "997009538336105171"),
        _unknown(None, "990000000000000001"),
        _unknown("Film MSS-D", "990000000000000002"),
    }
    assert len(ids) == 5
    assert "Unknown_Library" not in ids


def test_unknown_library_with_two_owners_is_keyed_per_item():
    additional = ("Shapira, Bernard, Jerusalem, Israel Ms. 1* Benayahu, Meir, Jerusalem, "
                  "Israel Ms. TEU 91")
    cid = _unknown(additional, "997009538336105171")
    assert cid == "Unknown_Library__sys997009538336105171"
    assert M.ktiv_former_owner_mark({"shelfmarks": {"additional": additional}}) is None


def test_unknown_library_without_sys_num_keeps_placeholder():
    doc = {"shelf_mark": "Unknown Library", "shelfmarks": {}}
    assert M.ktiv_mark_cid(doc, "Unknown Library") == ("Unknown_Library", None)


def test_unknown_library_former_owner_matches_direct_ktiv_style():
    # A Wallach item KTIV files under Wallach directly and one it files under
    # "Unknown Library" get ids in the same style.
    direct = _ktiv("Wallach, Isaac, Israel Ms. 327")
    via_unknown = _unknown("Wallach, Isaac, Israel Ms. 340")
    assert direct == "Wallach_Isaac_Israel_Ms_327"
    assert via_unknown.rsplit("_", 1)[0] == direct.rsplit("_", 1)[0]


def test_fjp_sassoon_uses_the_same_token_as_ktiv():
    assert M.fjp_canonical_id(
        "London Sassoon: Sassoon 410", {"collection": "London Sassoon", "institution": "Oslo"}
    ) == "London_Sassoon_410"
    assert _unknown("Sassoon, David Solomon, London, England Ms. 410") == "London_Sassoon_410"


@pytest.mark.parametrize("mark,expected", [
    ("The University of Haifa Library, Haifa, Israel Ms. GEN B Haf. 1",
     "Haifa_University_GEN_B_Haf_1"),
    ("Victor Emmanuel III National Library, Naples, Italy Ms. F 10bis", "Naples_BNN_F_10bis"),
    ("Topkapu Palace Library of Ahmet III, Istanbul, Turkey Ms. G. Islami 101-112",
     "Istanbul_Topkapi_G_Islami_101_112"),
    ("Library of the Emanuel Ringelblum Jewish Historical Institute, Warsaw, Poland Ms. 647",
     "Warsaw_JHI_647"),
    ("Historical Archive of the City of Cologne, Cologne, Germany Ms. Best. 7020 "
     "(Handschriften (W*)), 332/18", "Cologne_City_Archive_Best_7020_332_18"),
])
def test_new_institution_tokens(mark, expected):
    assert _ktiv(mark) == expected
