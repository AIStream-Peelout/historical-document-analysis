# File name: test_ktiv_transcription_gap.py
# Date: 10/30/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests of the KTIV transcription gap report: enumeration union, bundle / item-scrape / merged-corpus
joins, item-page text on disk, Genizah and token-group decisions, tiers, the extension queue CSV and
the report numbers. Everything is synthetic under ``tmp_path``; no raw data is read."""
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from src.datasets.merging import ktiv_transcription_gap as gap

SYS_HELD = "990000000010205171"       # bundle with text (and an item scrape with full text)
SYS_EMPTY = "990000000020205171"      # only an empty bundle
SYS_ITEM = "990000000030205171"       # item scrape, no bundle (Genizah, Judeo-Arabic catalogue)
SYS_MERGED = "990000000040205171"     # merged corpus only
SYS_NEW_JA = "990000000050205171"     # unknown, Judaeo-Arabic hits
SYS_NEW_HE = "990000000060205171"     # unknown, Hebrew, T-S shelf mark
SYS_NON_GEN = "990000000070205171"    # item scrape says not Genizah; its item page holds the text
SYS_WEAK = "990000000080205171"       # only the weak JA token "אן"
SYS_ARABIC = "990000000090205171"     # Arabic-script hit
SYS_JA_NON_GEN = "990000000100205171"  # Judaeo-Arabic, item scrape says not Genizah
SYS_NOT_ENUM = "990000000990205171"   # our bundle, missed by the enumeration

TOKEN_GROUPS = {"في": "arabic_script", "דלך": "judaeo_arabic", "אלדי": "judaeo_arabic",
                "אן": "judaeo_arabic", "אשר": "hebrew", "של": "hebrew"}
COMPLETE_TOKEN_STATS = [
    {"token": "אשר", "group": "hebrew", "total_hits": 9, "pages_fetched": 1, "done": True,
     "end_reason": "all_hits"},
    {"token": "של", "group": "hebrew", "total_hits": 420, "pages_fetched": 5, "done": True,
     "end_reason": "saturated"},
    {"token": "في", "group": "arabic_script", "total_hits": None, "pages_fetched": 0, "done": False,
     "end_reason": None},
]


def _ms(sysnum: str, tokens: List[str], shelfmark: Optional[str] = None, pages: Optional[List[int]] = None,
        fields: Optional[dict] = None) -> dict:
    """One enumerated manuscript in the extension's export shape.

    :param sysnum: System number.
    :param tokens: Tokens that hit it.
    :param shelfmark: ``lds06`` shelf mark.
    :param pages: Hit pages.
    :param fields: Kept search fields.
    :returns: The record.
    """
    return {"sysnum": sysnum, "shelfmark": shelfmark, "ie": "IE1", "team": "team", "thumbnail": None,
            "pages": pages or [1], "snippets": [], "tokens": tokens,
            "groups": sorted({TOKEN_GROUPS[t] for t in tokens}), "fields": fields or {}}


def _write(path: Path, data: object) -> None:
    """Write JSON.

    :param path: Target file.
    :param data: JSON-serialisable value.
    :returns: None.
    """
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _export(manuscripts: List[dict], complete: bool = True, generated_at: str = "2026-10-30T10:00:00Z",
            token_stats: Optional[List[dict]] = None) -> dict:
    """An enumeration file.

    :param manuscripts: Manuscript records.
    :param complete: Whether the run completed.
    :param generated_at: Timestamp.
    :param token_stats: Per-word stats (default: a single partial entry).
    :returns: The file content.
    """
    return {"generated_at": generated_at, "complete": complete, "stopped_reason": None if complete else "rate_limited",
            "bulk_size": 50, "multi_facet_query": "", "single_facet_query": "",
            "tokens": list(TOKEN_GROUPS), "token_groups": TOKEN_GROUPS, "total_requests": 12,
            "token_stats": token_stats or [{"token": "אשר", "total_hits": 9}], "manuscripts": manuscripts}


def _item(sysnum: str, shelf_mark: str, subjects: List[str], language: str = "Hebrew",
          full_text: Optional[dict] = None) -> dict:
    """An item-page scrape.

    :param sysnum: System number.
    :param shelf_mark: Shelf mark.
    :param subjects: Catalogue subjects.
    :param language: Catalogue language.
    :param full_text: The scrape's ``full_text`` block (None = no block).
    :returns: The record.
    """
    return {"source": "nli_ktiv", "sys_num": sysnum, "pnx_id": f"PNX_MANUSCRIPTS{sysnum}", "shelf_mark": shelf_mark,
            "page_title": f"{shelf_mark} | Ktiv | Item page", "basic_catalog": {"title": "x", "language": language},
            "full_catalog": {"subjects": subjects}, "full_text": full_text}


@pytest.fixture
def world(tmp_path: Path) -> Dict[str, Path]:
    """A synthetic KTIV folder, merged corpus and output folder.

    :param tmp_path: Pytest temp dir.
    :returns: Paths: ``ktiv``, ``merged``, ``out``.
    """
    ktiv = tmp_path / "ktiv"
    ktiv.mkdir()
    cul = "Cambridge University Library, Cambridge, England Ms. T-S"
    _write(ktiv / "ktiv_fts_transcribed_20261029_partial.json", _export([
        _ms(SYS_HELD, ["אשר"], f"{cul} 8J5.1"),
        _ms(SYS_NEW_JA, ["דלך"], "The Jewish Theological Seminary of America, New York, NY, USA Ms. ENA 1.2",
            pages=[1, 2]),
    ], complete=False, generated_at="2026-10-29T10:00:00Z"))
    _write(ktiv / "ktiv_fts_transcribed_20261030.json", _export([
        _ms(SYS_HELD, ["אשר", "של"], f"{cul} 8J5.1"),
        _ms(SYS_EMPTY, ["אשר"], f"{cul} 10J1.1"),
        _ms(SYS_ITEM, ["אשר"], None, pages=[1, 2, 3]),
        _ms(SYS_MERGED, ["אשר"], "The British Library, London, England Or. 5566"),
        _ms(SYS_NEW_JA, ["אשר"], None, pages=[1, 2, 3, 4]),
        _ms(SYS_NEW_HE, ["של"], f"{cul} NS 1.1", pages=[1, 2, 3, 4, 5]),
        _ms(SYS_NON_GEN, ["אשר"], "Ets Haim Library, Amsterdam Ms. 48 B 13"),
        _ms(SYS_WEAK, ["אן", "אשר"], "Some Library, Somewhere Ms. 1"),
        _ms(SYS_ARABIC, ["في"], "Unknown Collection Ms. 5", fields={"local29": "Cairo Genizah"}),
        _ms(SYS_JA_NON_GEN, ["דלך"], "Ets Haim Library, Amsterdam Ms. 47 E 1"),
    ], token_stats=COMPLETE_TOKEN_STATS))
    # Bundles: one with text (plus an empty duplicate), one empty, one the enumeration missed.
    _write(ktiv / f"ktiv_PNX_MANUSCRIPTS{SYS_HELD}-1_transcription.json",
           {"pages": [{"fl": "FL1", "annotation_page": {"items": [{"id": "w1"}]}}]})
    (ktiv / f"ktiv_PNX_MANUSCRIPTS{SYS_HELD}-1_transcription(1).json").write_text("", encoding="utf-8")
    _write(ktiv / f"ktiv_PNX_MANUSCRIPTS{SYS_EMPTY}-1_transcription.json", {"pages": [{"lines": ["", " "]}]})
    _write(ktiv / f"ktiv_PNX_MANUSCRIPTS{SYS_NOT_ENUM}-1_transcription.json", {"pages": [{"lines": ["שלום"]}]})
    # Item scrapes (one duplicated with a (1) suffix), a junk JSON, and a ZIP. Two item pages carry
    # their transcription text (one of them also has a bundle); one has an empty full_text.text.
    _write(ktiv / "ktiv_Cambridge_T-S_AS_1.1.json",
           _item(SYS_ITEM, f"{cul} AS 1.1", ["Cairo Genizah fragments"], "Hebrew Judeo-Arabic"))
    _write(ktiv / "ktiv_Cambridge_T-S_AS_1.1(1).json",
           _item(SYS_ITEM, f"{cul} AS 1.1", ["Cairo Genizah fragments"], "Hebrew Judeo-Arabic"))
    _write(ktiv / "ktiv_Ets_Haim_48_B_13.json",
           _item(SYS_NON_GEN, "Ets Haim Library, Amsterdam Ms. 48 B 13", ["Sermons"],
                 full_text={"text": ["שורה ראשונה", "שורה שנייה"]}))
    _write(ktiv / "ktiv_Ets_Haim_47_E_1.json",
           _item(SYS_JA_NON_GEN, "Ets Haim Library, Amsterdam Ms. 47 E 1", ["Liturgy"], "Judeo-Arabic",
                 full_text={"text": ["  "]}))
    _write(ktiv / "ktiv_Cambridge_T-S_8J5.1.json",
           _item(SYS_HELD, f"{cul} 8J5.1", ["Cairo Genizah fragments"], full_text={"text": ["אשר"]}))
    _write(ktiv / "ktiv_search_dump.json", [1, 2, 3])
    (ktiv / "ktiv_broken.json").write_text("{not json", encoding="utf-8")
    (ktiv / f"ktiv_PNX_MANUSCRIPTS{SYS_ITEM}-1_images.zip").write_bytes(b"PK")
    merged = tmp_path / "merged_shelfmarks.jsonl"
    merged.write_text("\n".join(json.dumps(r) for r in [
        {"canonical_id": "Cambridge_CUL_AS_1_1", "sources": {"ktiv": {"sys_num": SYS_ITEM}}},
        {"canonical_id": "London_BL_Or_5566", "sources": {"ktiv": {"pnx_id": f"PNX_MANUSCRIPTS{SYS_MERGED}"}}},
        {"canonical_id": "Cambridge_CUL_8J5_1", "sources": {"ktiv": {"sys_num": SYS_HELD}}},
        {"canonical_id": "PGP_only", "sources": {"ktiv": None, "pgp": {"x": 1}}},
    ]) + "\n", encoding="utf-8")
    return {"ktiv": ktiv, "merged": merged, "out": tmp_path / "out"}


def test_load_enumerations_unions_files_and_applies_the_weak_token_rule(world: Dict[str, Path]) -> None:
    """Records from several files merge by system number; ``אן`` alone is not Judaeo-Arabic.

    :param world: The synthetic folders.
    :returns: None.
    """
    paths = sorted(str(p) for p in world["ktiv"].glob(gap.FTS_PATTERN))
    by_sysnum, meta = gap.load_enumerations(paths)
    assert len(by_sysnum) == 10
    assert [f["file"] for f in meta["files"]] == ["ktiv_fts_transcribed_20261029_partial.json",
                                                  "ktiv_fts_transcribed_20261030.json"]
    assert meta["files"][1]["bulk_size"] == 50 and meta["files"][1]["facet_filtered"] is False
    held = by_sysnum[SYS_HELD]
    assert held["tokens"] == {"אשר", "של"} and held["groups"] == {"hebrew"}
    ja = by_sysnum[SYS_NEW_JA]
    assert ja["groups"] == {"judaeo_arabic", "hebrew"}
    assert ja["pages"] == {1, 2, 3, 4}
    assert ja["shelfmark"].endswith("ENA 1.2")  # the partial file's shelf mark fills the gap
    assert by_sysnum[SYS_WEAK]["groups"] == {"hebrew"}
    assert by_sysnum[SYS_ARABIC]["groups"] == {"arabic_script"}


def test_token_group_falls_back_to_script_detection() -> None:
    """Tokens missing from the file's mapping are grouped by their script; weak JA tokens need company.

    :returns: None.
    """
    assert gap.token_group("على", {}) == "arabic_script"
    assert gap.token_group("כל", {}) == "hebrew"
    assert gap.token_group("הדא", {"הדא": "judaeo_arabic"}) == "judaeo_arabic"
    assert gap.ja_hit({"אן", "ואן"}) is True
    assert gap.ja_hit({"אן"}) is False
    assert gap.ja_hit({"דלך"}) is True
    # הדא is Palestinian Aramaic too: alone it is weak, with another JA token it counts.
    assert "הדא" in gap.WEAK_JA_TOKENS
    assert gap.ja_hit({"הדא"}) is False
    assert gap.ja_hit({"הדא", "אן"}) is True
    assert gap.ja_hit({"הדא", "דלך"}) is True


def test_scan_bundles_reads_names_and_tells_text_from_empty(world: Dict[str, Path]) -> None:
    """Bundle duplicates collapse to one system number; zero-byte and blank bundles have no text.

    :param world: The synthetic folders.
    :returns: None.
    """
    bundles = gap.scan_bundles(str(world["ktiv"]))
    assert bundles == {
        SYS_HELD: {"files": 2, "with_text": True},
        SYS_EMPTY: {"files": 1, "with_text": False},
        SYS_NOT_ENUM: {"files": 1, "with_text": True},
    }


def test_scan_item_scrapes_reads_genizah_language_full_text_and_skips_junk(world: Dict[str, Path]) -> None:
    """Item scrapes are keyed by system number; non-dict and broken JSON files are skipped.

    :param world: The synthetic folders.
    :returns: None.
    """
    items = gap.scan_item_scrapes(str(world["ktiv"]))
    assert set(items) == {SYS_ITEM, SYS_NON_GEN, SYS_JA_NON_GEN, SYS_HELD}
    assert items[SYS_ITEM]["files"] == 2
    assert items[SYS_ITEM]["genizah"] is True
    assert items[SYS_ITEM]["language"] == "Hebrew Judeo-Arabic"
    assert items[SYS_NON_GEN]["genizah"] is False
    assert {s for s, item in items.items() if item["full_text"]} == {SYS_NON_GEN, SYS_HELD}


def test_item_full_text_needs_a_non_blank_line() -> None:
    """Only a ``full_text.text`` with real text counts as transcription text on disk.

    :returns: None.
    """
    assert gap.item_full_text({"full_text": {"text": ["", "שלום"]}}) is True
    assert gap.item_full_text({"full_text": {"text": "שלום"}}) is True
    assert gap.item_full_text({"full_text": {"text": [" ", ""]}}) is False
    assert gap.item_full_text({"full_text": {"text": []}}) is False
    assert gap.item_full_text({"full_text": {}}) is False
    assert gap.item_full_text({"full_text": None}) is False
    assert gap.item_full_text({}) is False


def test_item_genizah_flag_sources() -> None:
    """Subjects, titles and MARC can each mark the Genizah; no catalogue data means undecided.

    :returns: None.
    """
    assert gap.item_genizah_flag({"full_catalog": {"subjects": ["Cairo Genizah fragments"]}}) is True
    assert gap.item_genizah_flag({"page_title": "Ms. T-S 1, CUL, קטע גניזה | Ktiv"}) is True
    assert gap.item_genizah_flag({"basic_catalog": {"title": "x"}, "marc_xml": "<a>Geniza</a>"}) is True
    assert gap.item_genizah_flag({"basic_catalog": {"title": "Siddur"}}) is False
    assert gap.item_genizah_flag({"page_title": "Just a moment..."}) is None


def test_load_merged_ktiv_keys_canonical_ids_by_system_number(world: Dict[str, Path]) -> None:
    """Only records with a KTIV source count; ``pnx_id`` works when ``sys_num`` is absent.

    :param world: The synthetic folders.
    :returns: None.
    """
    merged = gap.load_merged_ktiv(str(world["merged"]))
    assert merged == {SYS_ITEM: ["Cambridge_CUL_AS_1_1"], SYS_MERGED: ["London_BL_Or_5566"],
                      SYS_HELD: ["Cambridge_CUL_8J5_1"]}


def test_language_groups_and_tiers() -> None:
    """Catalogue languages and token groups drive the tier; tier 1 excludes known non-Genizah material.

    :returns: None.
    """
    assert gap.language_groups("Hebrew Judeo-Arabic") == {"judaeo_arabic"}
    assert gap.language_groups("Arabic Hebrew") == {"arabic_script"}
    assert gap.language_groups("Judaeo-Arabic Arabic") == {"judaeo_arabic", "arabic_script"}
    assert gap.language_groups(None) == set()
    assert gap.assign_tier({"hebrew", "judaeo_arabic"}, True) == 1
    assert gap.assign_tier({"judaeo_arabic"}, None) == 1
    assert gap.assign_tier({"arabic_script"}, None) == 1
    # Known non-Genizah Judaeo-Arabic / Arabic material no longer outranks Genizah Hebrew.
    assert gap.assign_tier({"hebrew", "judaeo_arabic"}, False) == 3
    assert gap.assign_tier({"arabic_script"}, False) == 3
    assert gap.assign_tier({"hebrew"}, True) == 2
    assert gap.assign_tier({"hebrew"}, False) == 3
    assert gap.assign_tier({"hebrew"}, None) == 3
    assert gap.primary_group({"hebrew", "judaeo_arabic"}) == "judaeo_arabic"
    assert gap.primary_group(set()) == "unknown"


def test_decide_genizah_precedence() -> None:
    """Item scrape beats search fields beats the collection-head fallback, which only says yes.

    :returns: None.
    """
    assert gap.decide_genizah({"genizah": False}, {"local29": "Cairo Genizah"}, "Ms. T-S 1") == (False, "item_scrape")
    assert gap.decide_genizah(None, {"local29": ["x", "Cairo Genizah"]}, None) == (True, "search_fields")
    assert gap.decide_genizah(None, {}, "Cambridge University Library Ms. T-S NS 1.1") == (True, "shelfmark")
    assert gap.decide_genizah(None, {}, "JTS Ms. ENA 2808.59") == (True, "shelfmark")
    assert gap.decide_genizah(None, {}, "Ets Haim Library Ms. 48 B 13") == (None, "undecided")
    assert gap.decide_genizah({"genizah": None}, {}, None) == (None, "undecided")


def test_library_of_uses_institution_tokens() -> None:
    """Shelf-mark heads resolve to the merge's institution tokens, with a plain fallback.

    :returns: None.
    """
    assert gap.library_of("Cambridge University Library, Cambridge, England Ms. T-S NS 25.288") == "Cambridge_CUL"
    assert gap.library_of("Nowhere Library, Atlantis Ms. 1") == "Nowhere Library"
    assert gap.library_of(None) == "unknown"


def test_words_not_fully_paged_flags_words_short_of_their_total_hits() -> None:
    """Words whose pages stop short of TotalHits are listed; unknown totals are not.

    :returns: None.
    """
    words = gap.words_not_fully_paged(COMPLETE_TOKEN_STATS, 50)
    assert words == [{"token": "של", "group": "hebrew", "total_hits": 420, "pages_fetched": 5,
                      "hits_not_listed": 170, "done": True, "end_reason": "saturated"}]
    assert gap.words_not_fully_paged([{"token": "x", "total_hits": 9}], 50)[0]["hits_not_listed"] == 9
    assert gap.words_not_fully_paged([], 50) == []


def test_run_writes_report_and_extension_queue(world: Dict[str, Path]) -> None:
    """End to end: statuses, tiers, ranks, queue columns the extension parses, report numbers.

    :param world: The synthetic folders.
    :returns: None.
    """
    summary = gap.run(str(world["ktiv"]), str(world["merged"]), str(world["out"]))
    assert summary["enumerated_manuscripts"] == 10
    assert summary["by_status"] == {
        "local_bundle": 1, "item_page_text_no_bundle": 1, "local_bundle_empty": 1,
        "item_scrape_no_bundle": 2, "merged_corpus_no_bundle": 1, "unknown_to_us": 4,
    }
    assert summary["missing_transcriptions"] == 9
    assert summary["recall_check"] == {"local_bundles_with_text": 2, "found_by_enumeration": 1,
                                       "not_found_by_enumeration": 1}
    assert summary["text_on_disk"] == {
        "note": "item scrapes whose full_text.text holds the transcription (no word boxes)",
        "item_page_text_sysnums": 2, "item_page_text_with_bundle": 1, "item_page_text_without_bundle": 1,
        "enumerated_item_page_text_no_bundle": 1, "queue_rows_item_page_text": 1,
    }
    assert summary["enumeration_complete"] is True
    assert summary["token_stats"] == COMPLETE_TOKEN_STATS
    assert [w["token"] for w in summary["words_not_fully_paged"]] == ["של"]
    assert summary["queue"]["rows"] == 9
    assert summary["queue"]["by_tier"] == {"1": 3, "2": 2, "3": 4}
    assert summary["queue"]["zip_on_disk"] == 1
    assert summary["queue"]["text_on_disk_item_page"] == 1
    assert summary["genizah"] == {"genizah": 6, "non_genizah": 2, "undecided": 2}
    assert summary["group_membership"]["judaeo_arabic"] == {"enumerated": 3, "missing": 3, "item_page_text": 0}
    assert summary["group_membership"]["hebrew"]["item_page_text"] == 1
    assert summary["run_estimate"]["seconds_per_item"] == 8
    assert summary["run_estimate"]["hours_without_text_on_disk"] <= summary["run_estimate"]["hours"]
    assert summary["by_library"]["Cambridge_CUL"]["enumerated"] == 4

    with open(world["out"] / gap.QUEUE_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames[:5] == ["rank", "tier", "canonical_id", "shelf_mark", "ktiv_manifest_url"]
        assert reader.fieldnames[-1] == "text_on_disk"
        rows = list(reader)
    by_sys = {row["sysnum"]: row for row in rows}
    assert SYS_HELD not in by_sys
    assert [row["rank"] for row in rows] == [str(i) for i in range(1, 10)]
    assert [row["tier"] for row in rows] == ["1", "1", "1", "2", "2", "3", "3", "3", "3"]
    # Tier 1 ordered by hit pages: SYS_NEW_JA (4) > SYS_ITEM (3) > SYS_ARABIC (1).
    assert [row["sysnum"] for row in rows[:3]] == [SYS_NEW_JA, SYS_ITEM, SYS_ARABIC]
    assert [row["sysnum"] for row in rows[3:5]] == [SYS_NEW_HE, SYS_EMPTY]
    item = by_sys[SYS_ITEM]
    assert item["canonical_id"] == "Cambridge_CUL_AS_1_1"
    assert item["shelf_mark"].endswith("T-S AS 1.1")  # from the item scrape (no lds06)
    assert item["ktiv_manifest_url"] == (
        f"https://iiif.nli.org.il/IIIFv21/DOCID/PNX_MANUSCRIPTS{SYS_ITEM}-1/manifest")
    assert item["zip_on_disk"] == "yes" and item["genizah"] == "genizah"
    assert item["text_on_disk"] == "no"
    assert by_sys[SYS_NEW_HE]["canonical_id"] == SYS_NEW_HE  # unknown to the corpus → sysnum
    non_gen = by_sys[SYS_NON_GEN]
    assert non_gen["genizah"] == "non_genizah" and non_gen["tier"] == "3"
    # Text already on disk from the item page: still queued at its tier, flagged so it can be skipped.
    assert non_gen["status"] == "item_page_text_no_bundle" and non_gen["text_on_disk"] == "item_page"
    # Non-Genizah Judaeo-Arabic ranks with the rest of tier 3, not ahead of Genizah Hebrew.
    ja_non_gen = by_sys[SYS_JA_NON_GEN]
    assert ja_non_gen["primary_group"] == "judaeo_arabic" and ja_non_gen["tier"] == "3"
    assert ja_non_gen["text_on_disk"] == "no"  # a blank full_text.text does not count
    assert by_sys[SYS_WEAK]["primary_group"] == "hebrew" and by_sys[SYS_WEAK]["tier"] == "3"
    assert by_sys[SYS_MERGED]["status"] == "merged_corpus_no_bundle"
    assert all(row["canonical_id"] and row["shelf_mark"] for row in rows)  # the extension drops blank ones

    report = json.loads((world["out"] / gap.REPORT_JSON).read_text(encoding="utf-8"))
    assert report["queue"]["rows"] == 9
    markdown = (world["out"] / gap.REPORT_MD).read_text(encoding="utf-8")
    assert "Transcribed manuscripts enumerated: **10**" in markdown
    assert "1 / 2 (50.0%)" in markdown
    assert "| item_page_text_no_bundle | 1 / 10 (10.0%) |" in markdown
    assert "Transcription text on disk (item page): 2 item scrapes" in markdown
    assert "| של | hebrew | 420 | 5 | 170 | saturated |" in markdown


def test_queue_canonical_ids_stay_unique() -> None:
    """Two rows sharing a merged canonical id fall back to their system numbers.

    :returns: None.
    """
    base = {"status": gap.STATUS_UNKNOWN, "tier": 2, "hit_pages": 1, "genizah": True,
            "primary_group": "hebrew", "library": "x", "zip_on_disk": False, "text_on_disk": "no",
            "ktiv_manifest_url": "u"}
    rows = [dict(base, sysnum="1" * 18, shelf_mark="A", canonical_id="Join_1"),
            dict(base, sysnum="2" * 18, shelf_mark="B", canonical_id="Join_1"),
            dict(base, sysnum="3" * 18, shelf_mark=None, canonical_id=None)]
    queue = gap.build_queue(rows)
    # Same tier and pages: ordered by shelf mark (a missing one sorts first).
    assert [row["canonical_id"] for row in queue] == ["3" * 18, "1" * 18, "2" * 18]
    assert queue[0]["shelf_mark"] == "3" * 18


def test_run_without_enumeration_files_explains_what_to_do(tmp_path: Path) -> None:
    """No ``ktiv_fts_transcribed_*.json`` → a clear error, nothing written.

    :param tmp_path: Pytest temp dir.
    :returns: None.
    """
    (tmp_path / "ktiv").mkdir()
    with pytest.raises(FileNotFoundError, match="enumerator"):
        gap.run(str(tmp_path / "ktiv"), str(tmp_path / "missing.jsonl"), str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()
