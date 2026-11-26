# File name: test_legacy_store_inventory.py
# Date: 10/30/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Tests of the legacy-store transcription inventory: join keys, the corpus index, captures -> inventory, the
join statuses and the per-collection summary (synthetic merged corpus, no raw data)."""
import json
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from src.datasets.merging import legacy_store_inventory as fi


def _merged(cid: str, display: str, *, pgp_mark: Optional[str] = None, pgp_edition: bool = False,
            pgp_lang: str = "", ktiv_mark: Optional[str] = None, ktiv_text: bool = False,
            legacy_marks: Optional[List[str]] = None, legacy_text: bool = False, historic: str = "") -> dict:
    """A minimal ``merged_shelfmarks.jsonl`` record.

    :param cid: Canonical id.
    :param display: Display shelf mark.
    :param pgp_mark: PGP fragment shelf mark (adds a PGP block).
    :param pgp_edition: Whether the attached PGP document has a transcription.
    :param pgp_lang: PGP primary language.
    :param ktiv_mark: KTIV catalogue shelf mark (adds a KTIV block).
    :param ktiv_text: Whether our KTIV transcription scrape holds text.
    :param legacy_marks: Legacy-store scrape shelf-mark strings.
    :param legacy_text: Whether the legacy-store scrape holds text.
    :param historic: PGP historic shelf marks (``;``-separated).
    :returns: The record.
    """
    sources: Dict[str, object] = {"pgp": None, fi.LEGACY_SOURCE_KEY: [], "ktiv": None, "ktiv_transcription": None, "bodleian": None}
    present = []
    if pgp_mark:
        present.append("pgp")
        sources["pgp"] = {"fragment": {"shelfmark": pgp_mark, "shelfmarks_historic": historic},
                          "documents": [{"has_transcription": "Y" if pgp_edition else "N", "languages_primary": pgp_lang}]}
    if ktiv_mark:
        present.append("ktiv")
        sources["ktiv"] = {"shelf_mark": ktiv_mark, "shelfmarks": {"shelfmark": {"value": ktiv_mark}},
                           "basic_catalog": {"language": "Hebrew Judeo-Arabic"}}
        sources["ktiv_transcription"] = {"pages": [{"text": "x"}]} if ktiv_text else None
    for mark in legacy_marks or []:
        present.append(fi.LEGACY_SOURCE_KEY)
        sources[fi.LEGACY_SOURCE_KEY].append({"shelf_mark": mark, "transcriptions": {"1": "..."} if legacy_text else {}})
    return {"canonical_id": cid, "shelfmark_display": display, "sources_present": sorted(set(present)),
            "sources": sources}


@pytest.fixture
def merged_path(tmp_path: Path) -> Path:
    """Write a synthetic merged corpus and return its path.

    :param tmp_path: pytest temp dir.
    :returns: Path of the JSONL.
    """
    records = [
        _merged("T_S_8J5_1", "T-S 8J5.1", pgp_mark="T-S 8J5.1", pgp_edition=True, pgp_lang="Judaeo-Arabic"),
        _merged("T_S_AS_14_46", "T-S AS 14.46", pgp_mark="T-S AS 14.46", pgp_edition=False),
        _merged("T_S_AS_144_6", "T-S AS 144.6", pgp_mark="T-S AS 144.6", pgp_edition=True),
        _merged("StPetersburg_NLR_EVR_ARAB_II_1036",
                "The National Library of Russia, St. Petersburg, Russia Ms. EVR ARAB II 1036",
                ktiv_mark="The National Library of Russia, St. Petersburg, Russia Ms. EVR ARAB II 1036", ktiv_text=True),
        _merged("StPetersburg_NLR_RNL_Yevr_II_C1", "RNL Yevr. II C1", pgp_mark="RNL Yevr. II C1", pgp_edition=True),
        _merged("StPetersburg_NLR_Yevr_III_B_1089b", "St. Petersburg St. Peterburg: Yevr. III B 1089b",
                legacy_marks=["Cambridge CUL: T-S Ar.16.48 St. Petersburg St. Peterburg: Yevr. III B 1089b"], legacy_text=True),
        _merged("ENA_NS_2_26", "ENA NS 2.26", pgp_mark="ENA NS 2.26", pgp_edition=True, historic="ENA NS 2.26 (old)"),
        _merged("Halper_331", "Halper 331", pgp_mark="Halper 331", pgp_edition=True),
    ]
    path = tmp_path / "merged.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_join_key_unifies_holder_and_series_spellings() -> None:
    """RNL / NLR / Ms / Yevr vs EVR collapse; digit boundaries survive."""
    assert fi.join_key("RNL Yevr.-Arab. II 1201") == fi.join_key(
        "The National Library of Russia, St. Petersburg, Russia Ms. EVR ARAB II 1201")
    assert fi.join_key("Cambridge CUL: T-S 8 J 5.1") == fi.join_key("T-S 8J5.1")
    assert fi.join_key("T-S AS 14.46") != fi.join_key("T-S AS 144.6")
    assert fi.join_key("ENA NS 2.26") != fi.join_key("ENA NS 22.6")
    assert fi.join_key("Firkovich Evr. II A 733") == fi.join_key("Yevr. II A 733")
    assert fi.join_key("") == ""
    assert fi.join_key("  T-S   8J5.1 ") == fi.join_key("T-S 8J5.1")


def test_reference_index_keys_every_spelling(merged_path: Path) -> None:
    """Canonical id, display, PGP historic, legacy-store constituents and KTIV marks all resolve."""
    index = fi.build_reference_index(str(merged_path))
    assert [h["canonical_id"] for h in index[fi.join_key("T-S 8J5.1")]] == ["T_S_8J5_1"]
    assert [h["canonical_id"] for h in index[fi.join_key("ENA NS 2.26 (old)")]] == ["ENA_NS_2_26"]
    assert [h["canonical_id"] for h in index[fi.join_key("Yevr. III B 1089b")]] == ["StPetersburg_NLR_Yevr_III_B_1089b"]
    # A multi-mark legacy-store string must not index the whole string's other constituent on this record.
    assert all(h["canonical_id"] != "StPetersburg_NLR_Yevr_III_B_1089b" for h in index.get(fi.join_key("T-S Ar.16.48"), []))
    holdings = index[fi.join_key("EVR ARAB II 1036")][0]
    assert holdings["ktiv_transcription"] and not holdings["pgp_edition"]
    assert holdings["ktiv_language"] == "Hebrew Judeo-Arabic"


def test_record_marks_is_the_facade_lift(merged_path: Path) -> None:
    """``record_marks`` is the shelf-mark facade's (PR-6): same spellings, join strings skipped."""
    from src.datasets import shelfmarks as F

    for line in merged_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        assert list(fi.record_marks(record)) == F.record_marks(record)
    record = json.loads(merged_path.read_text(encoding="utf-8").splitlines()[5])
    assert list(fi.record_marks(record)) == ["StPetersburg_NLR_Yevr_III_B_1089b",
                                             "St. Petersburg St. Peterburg: Yevr. III B 1089b"]


def test_load_captures_dedupes_and_cleans(tmp_path: Path) -> None:
    """Later captures of the same item win; blank marks are dropped; per-item fields override the capture's."""
    cap = tmp_path / "captures"
    cap.mkdir()
    (cap / "a.json").write_text(json.dumps({
        "listing_url": "https://x/list?p=1", "collection": "Cambridge", "captured_at": "2026-10-30T00:00:00Z",
        "items": [
            {"shelf_mark": "T-S  8J5.1", "item_url": "https://x/item/1", "has_transcription": True, "transcription_length": 10},
            {"shelf_mark": "", "item_url": "https://x/item/2", "has_transcription": True},
            {"shelf_mark": "T-S AS 14.46", "has_transcription": False, "collection": "Cambridge AS"},
        ]}), encoding="utf-8")
    (cap / "b.json").write_text(json.dumps({
        "listing_url": "https://x/list?p=2", "collection": "Cambridge",
        "items": [{"shelf_mark": "T-S 8J5.1", "item_url": "https://x/item/1", "has_transcription": True,
                   "transcription_length": 12, "language": "Judaeo-Arabic"}]}), encoding="utf-8")
    rows = fi.load_captures(str(cap / "*.json"))
    assert [r["shelf_mark"] for r in rows] == ["T-S 8J5.1", "T-S AS 14.46"]
    assert rows[0]["transcription_length"] == 12 and rows[0]["language"] == "Judaeo-Arabic"
    assert rows[0]["listing_url"] == "https://x/list?p=2"
    assert rows[1]["collection"] == "Cambridge AS" and rows[1]["has_transcription"] is False
    out = tmp_path / "inventory.jsonl"
    assert fi.write_inventory(rows, str(out)) == 2
    assert fi.read_inventory(str(out))[0]["captured_at"] == "2026-10-30T00:00:00Z"


def test_join_statuses_and_summary(merged_path: Path, tmp_path: Path) -> None:
    """Every status is reachable; the digit-boundary trap stays apart; not_held counts only transcribed items."""
    index = fi.build_reference_index(str(merged_path))
    rows = [
        {"shelf_mark": "T-S 8J5.1", "collection": "Cambridge", "has_transcription": True},
        {"shelf_mark": "T-S AS 14.46", "collection": "Cambridge", "has_transcription": True},
        {"shelf_mark": "T-S AS 144.6", "collection": "Cambridge", "has_transcription": False},
        {"shelf_mark": "RNL Yevr.-Arab. II 1036", "collection": "St. Petersburg", "has_transcription": True},
        {"shelf_mark": "RNL Yevr. II C1", "collection": "St. Petersburg", "has_transcription": True},
        {"shelf_mark": "Yevr. III B 1089b", "collection": "St. Petersburg", "has_transcription": True},
        {"shelf_mark": "RNL Yevr.-Arab. II 1201", "collection": "St. Petersburg", "has_transcription": True},
    ]
    joined = fi.join_inventory(rows, index)
    by_mark = {r["shelf_mark"]: r for r in joined}
    assert by_mark["T-S 8J5.1"]["match_status"] == fi.STATUS_PGP
    assert by_mark["T-S 8J5.1"]["corpus_languages"] == "Judaeo-Arabic"
    assert by_mark["T-S AS 14.46"]["match_status"] == fi.STATUS_CORPUS
    assert by_mark["T-S AS 14.46"]["matched_canonical_id"] == "T_S_AS_14_46"
    assert by_mark["T-S AS 144.6"]["matched_canonical_id"] == "T_S_AS_144_6"
    assert by_mark["RNL Yevr.-Arab. II 1036"]["match_status"] == fi.STATUS_KTIV
    assert by_mark["RNL Yevr. II C1"]["match_status"] == fi.STATUS_PGP
    assert by_mark["Yevr. III B 1089b"]["match_status"] == fi.STATUS_LEGACY_ONLY
    assert by_mark["RNL Yevr.-Arab. II 1201"]["match_status"] == fi.STATUS_UNMATCHED
    assert not any(r["ambiguous"] for r in joined)
    summary = fi.summarise(joined)
    assert summary["ALL"]["listed"] == 7 and summary["ALL"]["transcribed"] == 6
    assert summary["ALL"]["not_held"] == 3  # AS 14.46 (corpus only), 1089b (legacy only), 1201 (unmatched)
    assert summary["St. Petersburg"][fi.STATUS_KTIV] == 1 and summary["St. Petersburg"]["not_held"] == 2
    assert "Cambridge" in summary and summary["Cambridge"]["not_held"] == 1


def test_pick_candidate_prefers_matching_institution() -> None:
    """Two fragments sharing a key: the one whose id starts with the item's institution token wins."""
    cands = [{"canonical_id": "Cambridge_CUL_Misc_12"}, {"canonical_id": "Manchester_JRL_Misc_12"}]
    chosen, ambiguous = fi._pick_candidate(cands, "Manchester", "Misc 12")
    assert chosen["canonical_id"] == "Manchester_JRL_Misc_12" and not ambiguous
    chosen, ambiguous = fi._pick_candidate(cands, "", "Misc 12")
    assert chosen is cands[0] and ambiguous


def test_run_build_and_join_end_to_end(merged_path: Path, tmp_path: Path) -> None:
    """build -> join writes the inventory, both CSVs and the summary."""
    out = tmp_path / "inv"
    (out / "captures").mkdir(parents=True)
    (out / "captures" / "c.json").write_text(json.dumps({
        "collection": "Cambridge", "listing_url": "https://x/l",
        "items": [{"shelf_mark": "T-S 8J5.1", "has_transcription": True, "item_url": "https://x/i/1"},
                  {"shelf_mark": "T-S NS 999.1", "has_transcription": True, "item_url": "https://x/i/2"}]}), encoding="utf-8")
    assert fi.run_build(str(out), str(out / "captures" / "*.json")) == 2
    summary = fi.run_join(str(out), str(merged_path))
    assert summary["ALL"] == {"listed": 2, "transcribed": 2, fi.STATUS_PGP: 1, fi.STATUS_UNMATCHED: 1, "not_held": 1}
    not_held = (out / fi.NOT_HELD_CSV).read_text(encoding="utf-8").splitlines()
    assert len(not_held) == 2 and "T-S NS 999.1" in not_held[1]
    assert (out / fi.JOIN_CSV).exists() and (out / fi.SUMMARY_FILE).exists()


def test_load_list_csv_and_jsonl(tmp_path: Path, merged_path: Path) -> None:
    """A hand-made CSV counts rows without a flag column as transcribed; yes/no words parse; JSONL works the same."""
    csv_path = tmp_path / "list.csv"
    csv_path.write_text(
        "shelf_mark,collection,language,has_transcription,transcription_length,item_url,notes\n"
        "T-S 8J5.1,Cambridge,Judaeo-Arabic,yes,\"1,234\",https://x/i/1,whatever\n"
        "T-S AS 14.46,Cambridge,,No,,,\n"
        "RNL Yevr.-Arab. II 1036,St. Petersburg,Judaeo-Arabic,,,,\n"
        ",Cambridge,,,,,\n",
        encoding="utf-8")
    rows = fi.load_list(str(csv_path))
    assert [r["shelf_mark"] for r in rows] == ["T-S 8J5.1", "T-S AS 14.46", "RNL Yevr.-Arab. II 1036"]
    assert rows[0]["has_transcription"] is True and rows[0]["transcription_length"] == 1234
    assert rows[1]["has_transcription"] is False and rows[1]["transcription_length"] is None
    assert rows[2]["has_transcription"] is True  # absent flag on a download list = fetched
    jsonl_path = tmp_path / "list.jsonl"
    jsonl_path.write_text(json.dumps({"shelf_mark": "ENA NS 2.26"}) + "\n" + json.dumps({"shelf_mark": "ENA NS 2.26", "language": "Hebrew"}) + "\n",
                          encoding="utf-8")
    rows = fi.load_list(str(jsonl_path))
    assert len(rows) == 1 and rows[0]["language"] == "Hebrew" and rows[0]["has_transcription"] is True
    with pytest.raises(ValueError):
        fi.load_list(str(tmp_path / "list.txt"))
    with pytest.raises(ValueError):
        fi.parse_flag("maybe", True)
    out = tmp_path / "inv"
    summary = fi.run_join(str(out), str(merged_path), input_path=str(csv_path))
    assert summary["ALL"]["listed"] == 3 and summary["ALL"]["transcribed"] == 2
    assert summary["St. Petersburg"][fi.STATUS_KTIV] == 1 and summary["Cambridge"][fi.STATUS_PGP] == 1
    assert (out / fi.INVENTORY_FILE).exists() and (out / fi.SUMMARY_FILE).exists()
