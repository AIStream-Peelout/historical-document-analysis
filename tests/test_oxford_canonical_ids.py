"""One canonical id per Oxford (Bodleian) leaf across PGP, FJP, KTIV, TEI and the scrape.

Every source spells a Bodleian leaf differently (PGP ``Bodl. MS heb. b 3/5``,
FJP ``Oxford: MS heb. b 3/5``, KTIV ``… England Ms. heb. b. 3.5``, TEI
``MS. Heb. b. 3/5``); all must land on the PGP-style id
``Oxford_Bodleian_Bodl_MS_heb_b_3_5`` while every existing PGP id and every
non-Oxford id stays byte-identical. Also covered: other libraries' ``Ms. Heb.``
shelfmarks are not filed under Oxford, the ``f.`` size letter survives, PGP
historic aliases cannot capture another leaf, volume-only KTIV items stay
distinct, and a Bodleian TEI part that is not verified for the leaf never
supplies its description, date or images.
"""

import csv
import json
import os

import pytest

from src.datasets.document_models.genizah_document import _merged_image_urls
from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer as SN
from src.datasets.merging import merge_shelfmarks as M
from src.datasets.merging.bodleian_images import (
    bodleian_folio_verified,
    bodleian_image_manifest,
    bodleian_images_verified,
    leaf_folios,
)
from src.datasets.merging.institution_tokens import OXFORD_TOKEN, resolve_token

KTIV_HEAD = "The Bodleian Libraries, University of Oxford, Oxford, England Ms. "
PGP_LIBRARY = {"library_abbrev": "Bodl.", "library": "Bodleian Library"}

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_RAW = os.path.join(_REPO_ROOT, "src", "datasets", "raw_data", "cairo_genizah")
_PGP_FRAGMENTS = os.path.join(_RAW, "pgp_raw", "data", "fragments.csv")
_MERGE_STATE = os.path.join(_RAW, "merged", "merge_state.json")


def _pgp(shelfmark: str) -> str:
    """PGP primary id for an Oxford fragment row.

    :param shelfmark: PGP ``shelfmark``.
    :returns: The canonical id the merge assigns.
    """
    return M.pgp_canonical_id({"shelfmark": shelfmark, **PGP_LIBRARY})[1]


def _fjp(mark: str) -> str:
    """FJP id for one Oxford segment.

    :param mark: FJP segment (``Oxford: MS heb. …``).
    :returns: The canonical id the merge assigns.
    """
    return M.fjp_canonical_id(mark, {"collection": "Oxford", "institution": "Bodleian"})


def _ktiv(core: str, sys_num: str = "990000000000205171", **doc) -> str:
    """KTIV id for a Bodleian item.

    :param core: Shelfmark after ``Ms.`` (``heb. b. 3.5``).
    :param sys_num: KTIV system number.
    :param doc: Extra KTIV record fields.
    :returns: The canonical id the merge assigns.
    """
    mark = KTIV_HEAD + core
    return M.ktiv_mark_cid({"shelf_mark": mark, "sys_num": sys_num, **doc}, mark)[0]


def _bod(shelf_mark: str, **fields) -> dict:
    """A Bodleian scrape record queued under *shelf_mark*.

    :param shelf_mark: The priority-queue shelfmark.
    :param fields: Overrides (``match``, ``tei``, ``images``, ``canonical_id`` …).
    :returns: The record dict.
    """
    return {"canonical_id": "Oxford_Bodleian_queued", "shelf_mark": shelf_mark,
            "match": "folio", "tei": None, "images": [], **fields}


# ── every spelling of one leaf -> one PGP-style id ────────────────────────────

@pytest.mark.parametrize("expected,ids", [
    ("Oxford_Bodleian_Bodl_MS_heb_b_3_5", lambda: [
        _pgp("Bodl. MS heb. b 3/5"),
        _fjp("Oxford: MS heb. b 3/5"),
        _fjp("Oxford: MS heb. b.3/5"),
        _ktiv("heb. b. 3.5"),
        M.bodleian_canonical_id(_bod("MS. Heb. b. 3/5")),          # TEI idno
    ]),
    ("Oxford_Bodleian_Bodl_MS_heb_f_56_1", lambda: [
        _pgp("Bodl. MS heb. f 56/1"),
        _fjp("Oxford: MS heb. f 56/1"),
        _fjp("Oxford: MS heb. f.56/1"),
        _ktiv("heb. f. 56.1"),
        M.bodleian_canonical_id(_bod("MS. Heb. f. 56/1")),
        M.bodleian_canonical_id(_bod("Bodleian Library MS Heb. f. 56, fol. 1")),
    ]),
    ("Oxford_Bodleian_Bodl_MS_heb_e_34_6", lambda: [
        _pgp("Bodl. MS heb. e 34/6"), _fjp("Oxford: MS heb. e.34/6"), _ktiv("heb. e. 34.6"),
    ]),
    ("Oxford_Bodleian_Bodl_MS_heb_c_10_1", lambda: [
        _pgp("Bodl. MS heb. c 10/1"), _fjp("MS heb. c.10/1"), _ktiv("heb. c. 10.1"),
    ]),
    ("Oxford_Bodleian_Bodl_MS_heb_d_68_102", lambda: [
        _pgp("Bodl. MS heb. d 68/102"), _fjp("Oxford: MS heb. d 68/102"),
        _ktiv("heb. d. 68. 102"),
    ]),
    ("Oxford_Bodleian_Bodl_MS_heb_f_57_7", lambda: [
        _pgp("Bod. MS Heb. f 57/7"), _pgp("Bodl. MS heb. f 57/7"), _ktiv("heb. f. 57.7"),
    ]),
    ("Oxford_Bodleian_Bodl_MS_Arab_c_56_30", lambda: [
        _pgp("Bodl. MS Arab. c 56.30"), _fjp("Oxford: MS Arab. c.56/30"),
        M.bodleian_canonical_id(_bod("MS. Arab. c. 56/30")),
    ]),
    ("Oxford_Bodleian_Bodl_MS_heb_d_44_28_29", lambda: [
        _pgp("Bodl. MS heb. d 44/28-29"), _pgp("Bodl. MS heb. d 44/28–29"),
        _fjp("Oxford: MS heb. d.44/28-29"),
    ]),
    ("Oxford_Bodleian_Bodl_MS_heb_b_12_13a", lambda: [
        _pgp("Bodl. MS heb. b 12/13a"), _fjp("Oxford: MS heb. b 12/13a"),
    ]),
])
def test_all_spellings_of_a_leaf_share_one_id(expected, ids):
    assert set(ids()) == {expected}


@pytest.mark.parametrize("raw,key", [
    ("Bodl. MS Heb. b 3 (Cat. 2806), f. 5", ("heb", "b", "3", ("5",))),
    ("Bodl. MS Heb. d 65, fol. 26", ("heb", "d", "65", ("26",))),
    ("Bodl. MS Heb e74/26", ("heb", "e", "74", ("26",))),
    ("Bodl. Ms Heb. d 66/137", ("heb", "d", "66", ("137",))),
    ("Bodl. MS Arab. c 56.49–50", ("arab", "c", "56", ("49", "50"))),
    (KTIV_HEAD + "heb. d. 05", ("heb", "d", "5", ())),
])
def test_parse_oxford_physical_key(raw, key):
    assert SN.parse_oxford(raw).physical_key == key


@pytest.mark.parametrize("raw", [
    "The National Library of Israel, Jerusalem, Israel Ms. Heb. 38°11343",
    "The National Library of France, Paris, France Ms. hebr. 719",
    "Harvard University Library, Cambridge, MA, USA Ms. Heb. 29, fol. 1",
    "Cambridge [Ma.] Harvard: MS Heb. 29 fol. 1",
    "Cambridge CUL: T-S 10J5.6",
    "bodl",
    "Oxford: MS heb. e.34/6 - MS heb. e.34/7",   # an unsplit join string
])
def test_parse_oxford_rejects_non_bodleian_and_unparseable(raw):
    assert SN.parse_oxford(raw) is None


def test_letter_ranges_and_leaf_folios():
    assert SN.parse_oxford("Bodl. MS heb. c 13/6-8").folios == {6, 7, 8}
    assert SN.parse_oxford("Bodl. MS heb. b 12/13a").folios == {13}
    assert SN.parse_oxford(KTIV_HEAD + "heb. a. 3").folios == set()


# ── existing PGP ids and non-Oxford ids are unchanged ─────────────────────────

@pytest.mark.parametrize("shelfmark,current_id", [
    ("Bodl. MS heb. b 3/5", "Oxford_Bodleian_Bodl_MS_heb_b_3_5"),
    ("Bodl. MS heb. f 101/42", "Oxford_Bodleian_Bodl_MS_heb_f_101_42"),
    ("Bodl. MS heb. d.47/44", "Oxford_Bodleian_Bodl_MS_heb_d_47_44"),
    ("Bodl. MS heb. d 73/4–8", "Oxford_Bodleian_Bodl_MS_heb_d_73_4_8"),
    ("Bodl. MS heb. c 13/6-8", "Oxford_Bodleian_Bodl_MS_heb_c_13_6_8"),
    ("Bodl. MS heb. b 12/13a", "Oxford_Bodleian_Bodl_MS_heb_b_12_13a"),
    ("Bodl. MS heb. g 2/2", "Oxford_Bodleian_Bodl_MS_heb_g_2_2"),
    ("Bodl. MS Arab. c 56.18", "Oxford_Bodleian_Bodl_MS_Arab_c_56_18"),
    ("Bodl. MS Arab. c 56.49–50", "Oxford_Bodleian_Bodl_MS_Arab_c_56_49_50"),
])
def test_pgp_style_ids_byte_identical(shelfmark, current_id):
    assert _pgp(shelfmark) == current_id


@pytest.mark.parametrize("institution,shelfmark,current_id", [
    ("CUL Cambridge University Library", "T-S 10J5.6", "Cambridge_CUL_T_S_10J5_6"),
    ("JTS Jewish Theological Seminary Library", "ENA 2713.17", "New_York_JTS_ENA_2713_17"),
    ("CUL & Bodl. Bodleian Library and Cambridge University Library (jointly owned) ",
     "L-G Ar.I.130", "Cambridge_Lewis_Gibson_L_G_Ar_I_130"),
    ("NLI National Library of Israel", "NLI 4°8333.27", "Jerusalem_NLI_4°8333_27"),
    ("BL British Library", "BL Or 10110, f. 23", "London_BL_Or_10110_23"),
])
def test_non_oxford_pgp_ids_unchanged(institution, shelfmark, current_id):
    abbrev, _, library = institution.partition(" ")
    assert M.pgp_canonical_id(
        {"shelfmark": shelfmark, "library_abbrev": abbrev, "library": library})[1] == current_id


@pytest.mark.skipif(not (os.path.exists(_PGP_FRAGMENTS) and os.path.exists(_MERGE_STATE)),
                    reason="needs local PGP data and merge_state.json")
def test_pgp_source_ids_match_current_merge():
    """Every PGP id (Oxford in PGP's own style, and every non-Oxford id) is a current id.

    Only Oxford rows written off PGP's house style ("Bodl. MS Heb.", "Bodl. Ms",
    "Bod.", a letter glued to its volume) change, to the case-folded PGP style.
    """
    with open(_MERGE_STATE, encoding="utf-8") as fh:
        current = set(json.load(fh)["all_ids"])
    exceptions, non_oxford = [], 0
    with open(_PGP_FRAGMENTS, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            token, cid = M.pgp_canonical_id(row)
            if not cid:
                continue
            if token != OXFORD_TOKEN:
                non_oxford += 1
                assert cid in current, row["shelfmark"]
            elif cid not in current:
                exceptions.append(row["shelfmark"])
    assert non_oxford > 30000
    assert exceptions, "the PGP off-style rows are expected to be renamed"
    for shelfmark in exceptions:
        assert not shelfmark.startswith(("Bodl. MS heb. ", "Bodl. MS Arab. ")), shelfmark
        assert _pgp(shelfmark).startswith(("Oxford_Bodleian_Bodl_MS_heb_",
                                           "Oxford_Bodleian_Bodl_MS_Arab_"))


# ── other libraries' "Ms. Heb." shelfmarks are not Oxford ─────────────────────

@pytest.mark.parametrize("text,token", [
    ("The National Library of Israel, Jerusalem, Israel Ms. Heb. 38°11343", "Jerusalem_NLI"),
    ("The National Library of France, Paris, France Ms. hebr. 719", "Paris_BNF"),
    ("Harvard University Library, Cambridge, MA, USA Ms. Heb. 29, fol. 1", "Harvard"),
    ("Cambridge [Ma.] Harvard: MS Heb. 29 fol. 1", "Harvard"),
    # Unchanged neighbours: private Jerusalem collections and Strasbourg's BNU.
    ("The Schocken Institute for Jewish Research, Jerusalem, Israel Ms. 3639.1", None),
    ("BNUS Bibliothèque Nationale et Universitaire de Strasbourg", "Strasbourg"),
    # A bare "MS heb." with no institution named is still the Bodleian.
    ("MS heb. c.10/1", OXFORD_TOKEN),
    (KTIV_HEAD + "heb. b. 3.5", OXFORD_TOKEN),
])
def test_institution_tokens_nli_bnf_harvard_before_oxford(text, token):
    assert resolve_token(text) == token


def test_nli_ktiv_record_id_moves_out_of_oxford():
    mark = "The National Library of Israel, Jerusalem, Israel Ms. Heb. 38°11343"
    assert M.ktiv_mark_cid({"shelf_mark": mark}, mark)[0] == "Jerusalem_NLI_Heb_38°11343"


# ── the "f." size letter survives FOLIO_RE ────────────────────────────────────

def test_f_size_letter_kept():
    assert _ktiv("heb. f. 101.42") == "Oxford_Bodleian_Bodl_MS_heb_f_101_42"
    assert _ktiv("heb. f. 101.42") == _pgp("Bodl. MS heb. f 101/42")
    assert SN.to_canonical_id("MS. Heb. f. 56/1") == "Bodl_MS_heb_f_56_1"
    # FOLIO_RE still strips a real folio designator elsewhere.
    assert SN.to_canonical_id("BM Or 10110, f. 23") == "BL_Or_10110_23"


# ── PGP historic alias guard ──────────────────────────────────────────────────

@pytest.mark.parametrize("primary,historic", [
    ("Bodl. MS Heb. f 57/1", "Bodl. MS Heb. d 57/1"),
    ("Bod. MS Heb. f 57/7", "Bod. MS Heb. d 57/7"),
    ("Bodl. MS heb. d 44/28-29", "Bodl. MS heb. d 44/3"),
    ("Bodl. MS heb. d.47/44", "Bodl. MS. Heb. d. 47/18"),
    ("Bodl. MS Arab. c 56.18", "Bodl. MS Arab. c 56.19"),
    ("Bodl. MS Arab. c 56.23", "Bodl. MS Arab. c 58.23"),
    ("Bodl. MS Arab. c 56.30", "Bodl. MS Arab. c 58.30"),
    ("Bodl. MS Arab. c 56.50", "Bodl. MS Arab. c 56.49–50"),
])
def test_cross_leaf_pgp_aliases_are_skipped(primary, historic):
    accepted, skipped = M.pgp_variant_aliases(OXFORD_TOKEN, _pgp(primary), primary, [historic])
    assert accepted == []
    assert len(skipped) == 1 and skipped[0]["variant"] == historic
    assert skipped[0]["reason"] in ("different_leaf", "overlapping_range")


def test_same_leaf_historic_alias_accepted_and_unparseable_skipped():
    primary = _pgp("Bodl. MS heb. b 3/5")
    accepted, skipped = M.pgp_variant_aliases(
        OXFORD_TOKEN, primary, "Bodl. MS heb. b 3/5",
        ["Bodl. MS Heb. b 3 (Cat. 2806), f. 5", "bodl"])
    assert accepted == [primary]
    assert [s["reason"] for s in skipped] == ["unparseable"]


def test_alias_allowlist_admits_reviewed_pair(monkeypatch):
    primary = _pgp("Bodl. MS Heb. f 57/1")
    variant = _pgp("Bodl. MS Heb. d 57/1")
    monkeypatch.setattr(M, "OXFORD_ALIAS_ALLOWLIST", frozenset({(primary, variant)}))
    accepted, skipped = M.pgp_variant_aliases(
        OXFORD_TOKEN, primary, "Bodl. MS Heb. f 57/1", ["Bodl. MS Heb. d 57/1"])
    assert accepted == [variant] and skipped == []


def test_non_oxford_aliases_keep_current_behaviour():
    accepted, skipped = M.pgp_variant_aliases(
        "London_BL", "London_BL_Or_10110_23", "BL Or 10110.23", ["BM Or 10110, f. 23", "Or 1"])
    assert accepted == ["London_BL_Or_10110_23", "London_BL_Or_1"] and skipped == []


def test_load_pgp_reports_skipped_aliases(tmp_path):
    frags = tmp_path / "fragments.csv"
    docs = tmp_path / "documents.csv"
    docs.write_text("pgpid,description\n", encoding="utf-8")
    with open(frags, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["shelfmark", "shelfmarks_historic", "pgpids",
                                                "library_abbrev", "library"])
        writer.writeheader()
        writer.writerow({"shelfmark": "Bodl. MS Heb. f 57/1",
                         "shelfmarks_historic": "Bodl. MS Heb. d 57/1; Bodl. MS Heb. f 57, fol. 1",
                         "pgpids": "", **PGP_LIBRARY})
    records, alias, skipped = M.load_pgp(str(frags), str(docs))
    assert set(records) == {"Oxford_Bodleian_Bodl_MS_heb_f_57_1"}
    assert "Oxford_Bodleian_Bodl_MS_heb_d_57_1" not in alias   # KTIV d.57.1 stays d.57.1
    assert [s["variant"] for s in skipped] == ["Bodl. MS Heb. d 57/1"]


# ── KTIV volume-only items stay distinct ──────────────────────────────────────

def test_ktiv_volume_only_keyed_by_sys_num():
    a = _ktiv("heb. a. 3", sys_num="990001648420205171")
    b = _ktiv("heb. a. 3", sys_num="990001648120205171")
    assert a == "Oxford_Bodleian_Bodl_MS_heb_a_3__sys990001648420205171"
    assert a != b


def test_ktiv_volume_only_uses_marc_leaf_note():
    marc = '<datafield tag="500"><subfield code="a">Shelfmark Range: From leaf 5 to Leaf 8</subfield>'
    mark = KTIV_HEAD + "heb. a. 3"
    assert M.ktiv_mark_cid({"shelf_mark": mark, "sys_num": "1", "marc_xml": marc}, mark) == (
        "Oxford_Bodleian_Bodl_MS_heb_a_3_5", "leaf_note")
    notes = {"full_catalog": {"notes": "Shelfmark Range: From leaf 42"}}
    assert M.ktiv_leaf_note(notes) == "42"
    # Item-level marks ignore the note (their own number is the folio).
    assert _ktiv("heb. e. 106.25", marc_xml=marc) == "Oxford_Bodleian_Bodl_MS_heb_e_106_25"


def test_load_ktiv_does_not_collapse_volume_only_items(tmp_path, monkeypatch):
    for i, sys_num in enumerate(["990001648420205171", "990001648120205171"]):
        doc = {"shelf_mark": KTIV_HEAD + "heb. a. 3", "sys_num": sys_num}
        (tmp_path / f"ktiv_bodl_{i}.json").write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.setattr(M, "KTIV_GLOB", str(tmp_path / "*.json"))
    by_cid, stats = M.load_ktiv({})
    assert len(by_cid) == 2
    assert {r["resolved_by"] for r in stats["oxford_volume_only"]} == {"sys_num"}
    assert len(stats["oxford_volume_only"]) == 2


# ── Bodleian scrape: re-keying and TEI / image precedence ─────────────────────

_WRONG_PART = dict(  # KTIV d.57.2 bound to TEI part 2 (fols 11–18): another leaf
    canonical_id="Oxford_Bodleian_heb_d_57_2",
    match="part",
    tei={"part_xml_id": "MS_Heb_d_57-part2", "titles": ["Arabic vocabulary to Judges"],
         "orig_date": {"text": "1500"}, "locus": [{"from": "11a", "to": "18b"}]},
    images=[{"stem": "MS_HEB_d_57_11a", "folio": "11", "side": "a",
             "local_path": "images/Oxford_Bodleian_heb_d_57_2/MS_HEB_d_57_11a.jpg"}],
)


def test_folio_verification_rules():
    wrong = _bod(KTIV_HEAD + "heb. d. 57.2", **_WRONG_PART)
    assert leaf_folios(wrong) == {2}
    assert not bodleian_folio_verified(wrong) and not bodleian_images_verified(wrong)
    part_ok = _bod("Bodl. MS heb. f 56/1", match="part", tei={"locus": [{"from": "1a", "to": "19b"}]})
    assert bodleian_folio_verified(part_ok)
    assert bodleian_folio_verified(_bod("Bodl. MS heb. e 34/67", match="folio"))
    # No numbered TEI parts, but the images are folio N's own facsimiles.
    none = _bod(KTIV_HEAD + "heb. e. 106.23", match="none",
                images=[{"folio": "23", "side": "a"}, {"folio": "23", "side": "b"}])
    assert not bodleian_folio_verified(none) and bodleian_images_verified(none)
    assert not bodleian_images_verified(_bod("Bodl. MS heb. a 1/1", match="none"))


def test_unverified_tei_part_never_outranks_fjp():
    bod = _bod("Oxford: MS heb. d.57/2", **_WRONG_PART)
    fjp = [("Oxford: MS heb. d.57/2", {"description": "תפלה למגפה (FJP)",
                                      "date": {"standard_date": "1839"},
                                      "images": ["fjp_small.jpg"]})]
    paths = {"Oxford_Bodleian_Bodl_MS_heb_d_57_2": ["BODLEIAN/x/MS_HEB_d_57_11a.jpg"]}
    rec = M.build_merged_record("Oxford_Bodleian_Bodl_MS_heb_d_57_2", None, fjp, None,
                                bodleian=bod, bodleian_images=paths)
    assert rec["description"] == "תפלה למגפה (FJP)"
    assert rec["date"] == "1839"
    assert rec["images"]["preferred_source"] == "fjp"
    blk = rec["images"]["bodleian"]
    assert blk["folio_verified"] is False and blk["images_verified"] is False
    # Another leaf's images are not published (the indexer serves every
    # source's image_urls), only kept aside; nor is the wrong part linked.
    assert blk["populated"] is False and blk["image_urls"] == [] and blk["images"] == []
    assert blk["unverified_images"] == ["BODLEIAN/x/MS_HEB_d_57_11a.jpg"]
    assert blk["tei_part_id"] is None and blk["catalogue_url"] is None


def test_unverified_bodleian_images_alone_are_not_an_image():
    bod = _bod(KTIV_HEAD + "heb. d. 57.2", **_WRONG_PART)
    paths = {"C": ["BODLEIAN/x/MS_HEB_d_57_11a.jpg"]}
    rec = M.build_merged_record("C", None, [], None, bodleian=bod, bodleian_images=paths)
    assert rec["images"]["preferred_source"] is None
    assert rec["description"] is None and rec["date"] is None
    assert not M.record_has_image(rec)
    # Nothing downstream can surface them either (GenizahDocument image_urls).
    assert _merged_image_urls(rec["images"]) == []


def test_no_tei_parts_record_keeps_volume_link_and_folio_labelled_images():
    bod = _bod(KTIV_HEAD + "heb. e. 106.23", match="none",
               tei={"catalogue_url": "https://hebrew.bodleian.ox.ac.uk/catalog/volume_1"},
               images=[{"folio": "23", "side": "a", "stem": "MS_HEB_e_106_23a"}])
    rec = M.build_merged_record("C", None, [], None, bodleian=bod,
                                bodleian_images={"C": ["BODLEIAN/x/MS_HEB_e_106_23a.jpg"]})
    blk = rec["images"]["bodleian"]
    assert blk["populated"] is True and blk["unverified_images"] == []
    assert blk["catalogue_url"] == "https://hebrew.bodleian.ox.ac.uk/catalog/volume_1"
    assert rec["images"]["preferred_source"] == "bodleian" and M.record_has_image(rec)


def test_verified_tei_part_still_fills_description():
    bod = _bod("Bodl. MS heb. f 56/1", match="part",
               tei={"titles": ["Calendar"], "locus": [{"from": "1a", "to": "19b"}]},
               images=[{"folio": "1", "stem": "MS_HEB_f_56_1a"}])
    rec = M.build_merged_record("C", None, [("Oxford: MS heb. f 56/1", {"description": "FJP"})],
                                None, bodleian=bod, bodleian_images={"C": ["BODLEIAN/x/a.jpg"]})
    assert rec["description"] == "Calendar"
    assert rec["images"]["preferred_source"] == "bodleian"


def test_load_bodleian_rekeys_every_queue_spelling_onto_one_leaf(tmp_path):
    records = tmp_path / "records"
    records.mkdir()
    pgp_style = _bod("Bodl. MS heb. a 2/6", canonical_id="Oxford_Bodleian_Bodl_MS_heb_a_2_6",
                     images=[{"stem": "MS_HEB_a_2_6a", "folio": "6",
                              "local_path": "images/Oxford_Bodleian_Bodl_MS_heb_a_2_6/MS_HEB_a_2_6a.jpg"}],
                     image_count=1)
    ktiv_style = _bod(KTIV_HEAD + "heb. a. 2.6", canonical_id="Oxford_Bodleian_heb_a_2_6",
                      match="part", tei={"locus": [{"from": "11a", "to": "12b"}]},
                      images=[{"stem": "MS_HEB_a_2_11a", "folio": "11",
                               "local_path": "images/Oxford_Bodleian_heb_a_2_6/MS_HEB_a_2_11a.jpg"},
                              {"stem": "MS_HEB_a_2_12a", "folio": "12",
                               "local_path": "images/Oxford_Bodleian_heb_a_2_6/MS_HEB_a_2_12a.jpg"}],
                      image_count=2)
    (records / "a.json").write_text(json.dumps(pgp_style), encoding="utf-8")
    (records / "b.json").write_text(json.dumps(ktiv_style), encoding="utf-8")
    by_cid, stats = M.load_bodleian(str(records / "*.json"))
    assert list(by_cid) == ["Oxford_Bodleian_Bodl_MS_heb_a_2_6"]
    # The folio-verified record wins over the richer, wrong-part one.
    assert by_cid["Oxford_Bodleian_Bodl_MS_heb_a_2_6"]["canonical_id"] == \
        "Oxford_Bodleian_Bodl_MS_heb_a_2_6"
    assert stats["rekeyed"] == 1
    assert stats["collapsed"] == [{"canonical_id": "Oxford_Bodleian_Bodl_MS_heb_a_2_6",
                                   "kept": "Oxford_Bodleian_Bodl_MS_heb_a_2_6",
                                   "dropped": "Oxford_Bodleian_heb_a_2_6"}]


def test_rekeyed_bodleian_images_keep_their_uploaded_object_path(tmp_path):
    stored = "Oxford_Bodleian_heb_e_106_23"
    img_dir = tmp_path / "images" / stored
    img_dir.mkdir(parents=True)
    (img_dir / "MS_HEB_e_106_23a.jpg").write_bytes(b"\xff\xd8")
    rec = _bod(KTIV_HEAD + "heb. e. 106.23", canonical_id=stored, match="none",
               images=[{"stem": "MS_HEB_e_106_23a", "folio": "23",
                        "local_path": f"images/{stored}/MS_HEB_e_106_23a.jpg"}])
    new_cid = M.bodleian_canonical_id(rec)
    assert new_cid == "Oxford_Bodleian_Bodl_MS_heb_e_106_23"
    assert bodleian_image_manifest({new_cid: rec}, str(tmp_path)) == {
        new_cid: [f"BODLEIAN/{stored}/MS_HEB_e_106_23a.jpg"]}
