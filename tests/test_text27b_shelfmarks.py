# File name: test_text27b_shelfmarks.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for text27b ``shelfmarks``: key construction (incl. benchmark-style canonical ids vs PGP
spellings), prompt-example leaks, the located-in-text rule and corpus matching on a synthetic
merged corpus / PGP export written to ``tmp_path``. All shelfmark numbers are made up.
"""
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.finetuning.text27b import paths as P
from src.finetuning.text27b import shelfmarks as S


# ----------------------------------------------------------------------------------- keys
def test_canonical_to_key_matches_pgp_bodleian_spelling() -> None:
    """A benchmark-style Bodleian id (no ``Bodl_``) and the PGP display string give one key (two synthetic leaves)."""
    assert S.canonical_to_key("Oxford_Bodleian_MS_heb_d_99_7") == S.key("Bodl. MS heb. d 99/7") == "mshebd99_7"
    assert S.canonical_to_key("Oxford_Bodleian_Bodl_MS_heb_d_99_7") == "mshebd99_7"
    assert S.canonical_to_key("Oxford_Bodleian_MS_heb_f_98_12") == S.key("Bodl. MS heb. f 98/12") == "mshebf98_12"


@pytest.mark.parametrize("spelling", ["Bodl. MS heb. d 99/7", "MS Heb. d. 99, f. 7", "Oxford: MS heb. d 99/7",
                                      "The Bodleian Libraries, University of Oxford, Oxford, England Ms. heb. d. 99.7"])
def test_key_bodleian_spellings(spelling: str) -> None:
    """PGP, TEI-ish, catalogue-style and KTIV-style spellings of one Bodleian leaf share a key.

    :param spelling: Source spelling.
    """
    assert S.key(spelling) == "mshebd99_7"


def test_size_letter_f_and_ts_f_series_survive() -> None:
    """The ``f`` size letter and the T-S ``F`` series are not taken for folio marks (prototype bug)."""
    assert S.prep_shelfmark("Bodl. MS heb. f 98/12") == "Bodl. MS heb. f 98/12"
    assert S.key("T-S F99.22") == S.canonical_to_key("Cambridge_CUL_T_S_F99_22") == "tsf99_22"
    assert S.key("T-S 99J1 fol. 3v") == S.key("T-S 99J1, f. 3") == "ts99j1_3"


def test_keys_institution_free_and_digit_boundaries() -> None:
    """Institution words are dropped on both sides; separators between digit runs are kept."""
    assert S.key("T-S 99J1.2") == S.canonical_to_key("Cambridge_CUL_T_S_99J1_2") == "ts99j1_2"
    assert S.key("ENA 9999.1") == S.canonical_to_key("New_York_JTS_ENA_9999_1") == "ena9999_1"
    assert S.key("Mosseri IX.99.1") == S.canonical_to_key("Cambridge_Mosseri_IX_99_1")
    assert S.key("JRL A 9990") == S.key("Rylands A 9990") == S.canonical_to_key("Manchester_JRL_A_9990")
    assert S.key("ENA NS 7.71") != S.key("ENA NS 77.1")
    assert S.key("T_S_99J1_2") == S.key("T-S 99J1.2")


def test_key_side_letters_sys_suffix_and_junk() -> None:
    """Side letters and KTIV ``__sys`` suffixes are ignored; strings without a usable core are junk."""
    assert S.key("T-S NS 999.25r") == S.canonical_to_key("Cambridge_CUL_T_S_NS_999_25r") == "tsns999_25"
    assert S.canonical_to_key("London_BL_Or_99999__sys990000000000205171") == S.key("BL Or. 99999")
    assert S.key("Recto") == "" and S.key("") == "" and S.key(None) == ""
    assert S.canonical_to_key("Berlin_JCB_A2") == ""
    assert S.canonical_to_key("Manchester_JRL_L_98") == S.key("JRL L 98") == "l98"


def test_keys_split_joins() -> None:
    """Joined shelfmarks give one key per part, without duplicates."""
    assert S.keys("T-S 99J1.2 + Bodl. MS heb. d 99/7") == ["ts99j1_2", "mshebd99_7"]
    assert S.keys("Bodl. MS heb. b 99/6 + Bodl. MS heb. b 99/6") == ["mshebb99_6"]
    assert S.key("T-S 99J1.2 + ENA 9999.1") == "ts99j1_2"
    assert S.keys(None) == []


def test_leaked_prompt_examples() -> None:
    """The structuring prompt's example shelfmarks have cores and keys."""
    assert len(S.LEAKED_CORES) >= 10 and len(S.LEAKED_KEYS) >= 10


# ----------------------------------------------------------------------------------- location
def test_locate_variants_and_negatives() -> None:
    """Spacing / folio variants are located at a token boundary; other numbers are not."""
    text = "As shown in T-S 99 J 1.2 (fol. 3) and in ENA 9999.1, but not ENA 99991."
    assert S.locate("T-S 99J1.2", text) == [text.index("T-S 99 J")]
    assert S.locate("ENA 9999.1", text) == [text.index("ENA 9999.1")]
    assert S.locate("T-S 99J1.3", text) == []
    assert S.locate("Recto", text) == []
    assert S.is_located("ENA 9999.1", text) and not S.is_located("ENA 9999.2", text)


def test_located_keys_drops_fake_keys() -> None:
    """Only keys present in the page text survive, in input order."""
    text = "The letter T-S 99J1.2 is discussed below; see also ENA 9999.1."
    keys = ["ENA 9999.1", "T-S A43.1", "T-S 99J1.2", "ENA 9999.1"]
    assert S.located_keys(keys, text) == ["ENA 9999.1", "T-S 99J1.2"]


# ----------------------------------------------------------------------------------- corpus
def _write_csv(path: Path, header: List[str], rows: List[Dict[str, str]]) -> None:
    """Write a CSV with a BOM, as the PGP export does.

    :param path: Destination.
    :param header: Column names.
    :param rows: Rows.
    :returns: None.
    """
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


@pytest.fixture()
def corpus_paths(tmp_path: Path) -> P.Paths:
    """A synthetic config, merged corpus, PGP export and reads file.

    :param tmp_path: Pytest temporary directory.
    :returns: Loaded :class:`Paths`.
    """
    data = tmp_path / "data"
    pgp = data / "pgp"
    reads = data / "reads"
    pgp.mkdir(parents=True)
    reads.mkdir()
    merged = [
        {"canonical_id": "Cambridge_CUL_T_S_99J1_2", "shelfmark_display": "T-S 99J1.2", "pgpids": [9001],
         "sources_present": ["pgp"], "description": "Letter."},
        {"canonical_id": "Oxford_Bodleian_Bodl_MS_heb_d_99_7", "shelfmark_display": "Bodl. MS heb. d 99/7",
         "pgpids": [9002], "sources_present": ["pgp", "ktiv"], "description": "Deed."},
        {"canonical_id": "New_York_JTS_ENA_9999_1", "shelfmark_display": "ENA 9999.1", "pgpids": [],
         "sources_present": ["ktiv"]},
    ]
    with open(data / "merged.jsonl", "w", encoding="utf-8") as fh:
        for r in merged:
            fh.write(json.dumps(r) + "\n")
    _write_csv(pgp / "documents.csv", ["pgpid", "shelfmark", "type", "description", "has_transcription", "has_translation"],
               [{"pgpid": "9001", "shelfmark": "T-S 99J1.2", "has_transcription": "Y", "has_translation": "Y"},
                {"pgpid": "9002", "shelfmark": "Bodl. MS heb. d 99/7", "has_transcription": "Y"},
                {"pgpid": "9004", "shelfmark": "T-S NS 999.5"}])
    _write_csv(pgp / "footnotes.csv", ["document_id", "source_slug", "location", "doc_relation", "content"],
               [{"document_id": "9001", "source_slug": "s1", "doc_relation": "Digital Edition", "content": "שלום"},
                {"document_id": "9001", "source_slug": "s1", "doc_relation": "Digital Translation", "content": "Peace"},
                {"document_id": "9002", "source_slug": "s2", "doc_relation": "Discussion", "content": "x"}])
    _write_csv(pgp / "fragments.csv", ["shelfmark", "pgpids", "shelfmarks_historic"],
               [{"shelfmark": "T-S 99J1.2", "pgpids": "9001"}, {"shelfmark": "T-S NS 999.5", "pgpids": "9004"}])
    _write_csv(pgp / "sources.csv", ["slug", "title", "year", "authors"],
               [{"slug": "s1", "title": "T", "year": "2000", "authors": "Smith, J."}])
    (reads / "reader_x.jsonl").write_text(json.dumps({"doc_id": "Cambridge_CUL_T_S_99J1_2", "text": "r"}) + "\n")
    (reads / "reader_x.failures.jsonl").write_text(json.dumps({"doc_id": "New_York_JTS_ENA_9999_1"}) + "\n")
    cfg: Dict[str, Any] = {"repo": str(tmp_path), "merged_shelfmarks": "data/merged.jsonl", "pgp_export": "data/pgp",
                           "ai_reads": "data/reads", "out_dir": "out"}
    for k in ["probe_dir", "text_ft_v0_1", "pgp_parallel", "merged_pgpid_map", "document_treatments",
              "analysis_pages_index", "exclusions_v1", "samples", "a50_overlap", "benchmark_ids", "academic_literature",
              "book_editions_nas", "tokenizer_json", "chat_template", "scratch"]:
        cfg[k] = f"data/{k}"
    f = tmp_path / "paths.json"
    f.write_text(json.dumps(cfg))
    return P.load_paths(f)


def test_load_corpus_fields_and_reads(corpus_paths: P.Paths) -> None:
    """The loader indexes merged cores, PGP pgpids, fragments, footnotes and read ids (failures skipped).

    :param corpus_paths: Synthetic configuration.
    """
    c = S.load_corpus(corpus_paths)
    assert set(c.merged) == {"Cambridge_CUL_T_S_99J1_2", "Oxford_Bodleian_Bodl_MS_heb_d_99_7", "New_York_JTS_ENA_9999_1"}
    assert c.pgp_to_cids["9002"] == {"Oxford_Bodleian_Bodl_MS_heb_d_99_7"}
    assert c.frag_core_pgpids["T_S_NS_999_5"] == {"9004"}
    assert c.read_ids == {"Cambridge_CUL_T_S_99J1_2"} and set(c.read_ids_by_model) == {"reader_x.jsonl"}
    assert c.fn_rel["9001"] == {"Digital Edition", "Digital Translation"}
    assert [r["content"] for r in c.fn_rows if r["pgpid"] == "9002"] == [""]
    assert c.sources["s1"]["authors"] == "Smith, J."


def test_load_corpus_cache_invalidation(corpus_paths: P.Paths) -> None:
    """The pickle cache is reused until an input file changes.

    :param corpus_paths: Synthetic configuration.
    """
    S.load_corpus(corpus_paths)
    cache = corpus_paths.out("cache", "corpus.pkl")
    assert cache.exists()
    stamp = cache.stat().st_mtime_ns
    assert S.load_corpus(corpus_paths).read_ids == {"Cambridge_CUL_T_S_99J1_2"}
    assert cache.stat().st_mtime_ns == stamp
    reads = corpus_paths.ai_reads / "reader_x.jsonl"
    reads.write_text(reads.read_text() + json.dumps({"doc_id": "New_York_JTS_ENA_9999_1"}) + "\n")
    os.utime(reads, ns=(reads.stat().st_atime_ns, reads.stat().st_mtime_ns + 10_000_000))
    assert "New_York_JTS_ENA_9999_1" in S.load_corpus(corpus_paths).read_ids


def test_match_shelfmark_paths(corpus_paths: P.Paths) -> None:
    """Exact, core, fragment-only, junk and unmatched outcomes.

    :param corpus_paths: Synthetic configuration.
    """
    c = S.load_corpus(corpus_paths, cache=False)
    m = S.match_shelfmark("T-S 99J1.2", c)
    assert m["how"] == "exact" and m["pgpids"] == ["9001"] and m["cids"] == ["Cambridge_CUL_T_S_99J1_2"]
    m = S.match_shelfmark("Bodl. MS heb. d 99/7", c)
    assert m["how"] == "exact" and m["pgpids"] == ["9002"]
    m = S.match_shelfmark("99J1.2", c)
    assert m["how"] in ("core", "core_lc", "core_compact", "unmatched")
    m = S.match_shelfmark("T-S NS 999.5", c)
    assert m["how"] == "pgp_fragment" and m["pgpids"] == ["9004"] and m["cids"] == []
    assert S.match_shelfmark("Recto", c)["how"] == "junk"
    assert S.match_shelfmark("T-S 98J1.1", c)["how"] == "unmatched"
    flags = S.pgp_flags(["9001"], ["Cambridge_CUL_T_S_99J1_2"], c)
    assert flags == {"has_pgp_digital_edition": True, "has_pgp_digital_translation": True, "has_our_read": True}
    assert S.pgp_flags(["9002"], [], c)["has_pgp_digital_edition"] is False
