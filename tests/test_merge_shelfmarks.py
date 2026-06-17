"""Tests for FJP multi-shelfmark splitting in the merge pipeline.

The vast majority of FJP records carry a single shelfmark, but ~2% enumerate
several fragments in one string (joins / ranges). Those must be exploded so each
constituent shelfmark is captured rather than lost to a garbage canonical id.
"""

from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer as SN
from src.datasets.merging.merge_shelfmarks import (
    _ktiv_ena_alias,
    build_merged_record,
    diff_against_snapshot,
    index_ktiv_zips,
    looks_multi,
    record_has_image,
    record_is_rich,
    split_shelfmarks,
)


def test_single_shelfmark_not_split():
    assert split_shelfmarks("Cambridge CUL: T-S 10J5.6") == ["Cambridge CUL: T-S 10J5.6"]
    assert split_shelfmarks("New York JTS: ENA 4020.50") == ["New York JTS: ENA 4020.50"]
    assert not looks_multi("Cambridge CUL: T-S 10J5.6")


def test_multi_same_institution_explodes_and_carries_prefix():
    raw = "Cambridge CUL: T-S AS 62.64 T-S AS 62.645"
    marks = split_shelfmarks(raw)
    assert [SN.to_canonical_id(m) for m in marks] == ["T_S_AS_62_64", "T_S_AS_62_645"]
    # The institution context is carried forward to the second mark.
    assert all(m.startswith("Cambridge CUL:") for m in marks)


def test_range_endpoints_become_separate_marks():
    raw = "Cambridge CUL: T-S NS 266 - T-S NS 270"
    cids = [SN.to_canonical_id(m) for m in split_shelfmarks(raw)]
    assert cids == ["T_S_NS_266", "T_S_NS_270"]


def test_multi_institution_explodes_with_each_context():
    raw = ("Cambridge CUL: T-S A38.2 T-S NS 247.4 "
           "Oxford: MS heb. e.43/3 Paris AIU: IX.A.13")
    cids = [SN.to_canonical_id(m) for m in split_shelfmarks(raw)]
    assert cids == ["T_S_A38_2", "T_S_NS_247_4", "MS_heb_e_43_3", "IX_A_13"]


def test_parenthetical_alt_not_treated_as_second_shelfmark():
    # "(Alt: 1)" must not split on its inner colon into a bogus "1)" mark.
    assert split_shelfmarks("Cincinnati HUC: 1001 (Alt: 1)") == ["Cincinnati HUC: 1001"]
    marks = split_shelfmarks("Paris Mosseri: Moss. VIII399.1 (Alt: 2nd Series: P 764)")
    assert [SN.to_canonical_id(m) for m in marks] == ["Mosseri_VIII399_1"]


# ── image / richness predicates ───────────────────────────────────────────────

def test_record_has_image():
    assert record_has_image({"images": {"fjp": ["x_1r.jpg"], "ktiv": None}})
    assert record_has_image({"images": {"fjp": [], "ktiv": {"pnx_id": "PNX_1"}}})
    assert not record_has_image({"images": {"fjp": [], "ktiv": None}})
    assert not record_has_image({"images": {"fjp": [], "ktiv": {"pnx_id": None}}})


def test_record_is_rich():
    assert record_is_rich({"description": "Letter from X", "sources": {}})
    assert record_is_rich({"description": None, "sources": {"fjp": [{"transcriptions": [1]}]}})
    assert not record_is_rich({"description": None, "sources": {"fjp": [{}], "ktiv": None}})


# ── run-over-run diff ─────────────────────────────────────────────────────────

# ── KTIV JTS dual-shelfmark (ENA) bridge ──────────────────────────────────────

def test_ktiv_lutzki_keys_under_ena_from_additional():
    # A Lutzki/MS item carries its old ENA number in shelfmarks.additional;
    # it must re-key onto the ENA form so it joins PGP/FJP.
    doc = {
        "shelf_mark": "The Jewish Theological Seminary of America, New York Ms. Lutzki 825, fol. 55",
        "shelfmarks": {"additional": "Adler, Elkan Nathan Ms. 1205.55"},
    }
    assert _ktiv_ena_alias(doc) == "New_York_JTS_ENA_1205_55"


def test_ktiv_ena_alias_absent_without_adler():
    assert _ktiv_ena_alias({"shelfmarks": {"additional": "Film MSS-D"}}) is None
    assert _ktiv_ena_alias({"shelfmarks": {}}) is None


# ── KTIV image-zip pointers ───────────────────────────────────────────────────

def test_index_ktiv_zips_groups_by_sysnum(tmp_path):
    for name in [
        "ktiv_PNX_MANUSCRIPTS990051236050205171-1_images.zip",
        "ktiv_PNX_MANUSCRIPTS990051236050205171-1_images(1).zip",
        "ktiv_PNX_MANUSCRIPTS990039474250205171-1_images.zip",
    ]:
        (tmp_path / name).write_bytes(b"")
    index = index_ktiv_zips(str(tmp_path / "*.zip"))
    assert len(index["990051236050205171"]) == 2          # two downloads of one ms
    assert index["990039474250205171"][0].endswith("images.zip")


def test_merged_record_points_at_ktiv_zip():
    ktiv = {"pnx_id": "PNX_1", "sys_num": "990051236050205171", "shelf_mark": "x"}
    zips = {"990051236050205171": ["ktiv_PNX_MANUSCRIPTS990051236050205171-1_images.zip"]}
    rec = build_merged_record("Cambridge_CUL_T_S_10J5_6", None, [], ktiv, zips)
    assert rec["images"]["ktiv"]["zip_files"] == zips["990051236050205171"]
    assert rec["images"]["ktiv"]["zip_downloaded"] is True
    assert rec["images"]["preferred_source"] == "ktiv"


def test_merged_record_ktiv_zip_absent_when_not_downloaded():
    ktiv = {"pnx_id": "PNX_2", "sys_num": "999999999999999999", "shelf_mark": "x"}
    rec = build_merged_record("X_1", None, [], ktiv, {})
    assert rec["images"]["ktiv"]["zip_files"] == []
    assert rec["images"]["ktiv"]["zip_downloaded"] is False


def test_diff_baseline_when_no_previous():
    state = {"all_ids": ["A"], "pgp_covered_ids": [], "ktiv_only_ids": [], "no_image_ids": ["A"]}
    assert diff_against_snapshot(state, None)["baseline"] is True


def test_diff_detects_new_ktiv_and_coverage_and_imaging():
    prev = {
        "all_ids": ["A", "B"],
        "pgp_covered_ids": [],
        "ktiv_only_ids": [],
        "no_image_ids": ["A", "B"],
    }
    cur = {
        "all_ids": ["A", "B", "C"],   # C is newly scraped
        "pgp_covered_ids": ["A"],     # A just gained FJP/KTIV coverage
        "ktiv_only_ids": ["C"],       # C is KTIV-only
        "no_image_ids": ["B"],        # A just gained an image
    }
    diff = diff_against_snapshot(cur, prev)
    assert diff["new_records"] == ["C"]
    assert diff["new_ktiv_only"] == ["C"]
    assert diff["newly_pgp_covered"] == ["A"]
    assert diff["newly_imaged"] == ["A"]
    assert diff["counts"]["new_records"] == 1
