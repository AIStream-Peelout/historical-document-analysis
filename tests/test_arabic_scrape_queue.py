# File name: test_arabic_scrape_queue.py
# Date: 10/15/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for the Arabic-script scraping queue: PGP export and output directories, lazy fragment list, image-store
photographs as permitted (never benchmark-grade) images and the merged-record joins."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pytest

from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer
from src.datasets.indexing import arabic_scrape_queue as asq

REPO = Path(__file__).resolve().parents[1]
ARABIC_LONG = "بسم الله الرحمن الرحيم هذا ما اشترى الشيخ ابو الفضل من القاضي " * 6       # over 200 Arabic letters
DOCUMENT_COLUMNS = ["pgpid", "url", "iiif_urls", "shelfmark", "multifragment", "side", "type", "languages_primary",
                    "doc_date_standard", "has_transcription"]
DOCUMENTS = [
    {"pgpid": "1", "shelfmark": "T-S 13J4.15.1", "type": "Letter", "languages_primary": "Arabic", "has_transcription": "Y"},
    {"pgpid": "2", "shelfmark": "T-S Ar.1.2", "iiif_urls": "https://lib/m2", "type": "Legal document", "languages_primary": "Arabic",
     "has_transcription": "Y"},
    {"pgpid": "3", "shelfmark": "T-S 8J1.3", "iiif_urls": "https://lib/m3", "languages_primary": "Judaeo-Arabic", "has_transcription": "Y"},
    {"pgpid": "4", "shelfmark": "T-S Ar.1.4", "iiif_urls": "https://lib/m4", "type": "List or table", "languages_primary": "Arabic"},
]


def _csv(path: Path, columns: List[str], rows: List[Dict[str, str]]) -> None:
    """Write a PGP-export-style CSV (UTF-8 with BOM)."""
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def _pgp_dir(path: Path, fragments: List[str], extra_documents: Sequence[Dict[str, str]] = ()) -> Path:
    """A tiny PGP export: documents (plus ``extra_documents``), footnotes (editions of 1 and 2) and the given fragment shelfmarks."""
    path.mkdir(parents=True)
    _csv(path / "documents.csv", DOCUMENT_COLUMNS,
         [dict(row, url=f"https://pgp/{row['pgpid']}") for row in list(DOCUMENTS) + list(extra_documents)])
    _csv(path / "footnotes.csv", ["document", "document_id", "doc_relation", "content"],
         [{"document_id": pid, "doc_relation": "Digital Edition", "content": ARABIC_LONG} for pid in ("1", "2")]
         + [{"document_id": "4", "doc_relation": "Discussion", "content": ARABIC_LONG}])
    _csv(path / "fragments.csv", ["shelfmark", "library", "collection", "iiif_url"],
         [{"shelfmark": shelfmark, "library": "CUL", "collection": "T-S"} for shelfmark in fragments])
    return path


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Path]:
    """Two PGP exports (with and without the fragment ``T-S 13J4.15``) and the non-PGP inputs, all in tmp_path."""
    merged = tmp_path / "merged_shelfmarks.jsonl"
    merged.write_text(json.dumps({"canonical_id": "Cambridge_T_S_Ar_1_2", "pgpids": ["2"], "sources": {}, "images": {}}) + "\n",
                      encoding="utf-8")
    (tmp_path / "keys.json").write_text("[]", encoding="utf-8")
    (tmp_path / "ids.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(asq, "MERGED", merged)
    monkeypatch.setattr(asq, "CLEAN_V2_KEYS", tmp_path / "keys.json")
    monkeypatch.setattr(asq, "CLEAN_V2_IDS", tmp_path / "ids.json")
    monkeypatch.setattr(asq, "LOCAL_IMAGE_DIRS", [])
    default_out = tmp_path / "raw_merged"
    default_out.mkdir()
    for name in ("OUT_CSV", "OUT_BACKLOG", "OUT_SUMMARY"):
        monkeypatch.setattr(asq, name, default_out / getattr(asq, name).name)
    return {"with": _pgp_dir(tmp_path / "pgp_with", ["T-S 13J4.15", "T-S Ar.1.2"]),
            "without": _pgp_dir(tmp_path / "pgp_without", ["T-S Ar.1.2"]), "default_out": default_out}


def _read(path: Path) -> List[Dict[str, str]]:
    """Rows of a queue CSV."""
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.mark.parametrize("shelfmark, url", [
    ("T-S 13J1.10", "MS-TS-00013-J-00001-00010"), ("T-S Ar.18(1).135", "MS-TS-AR-00018-00001-00135"),
    ("CUL Or.1080 J10", "MS-OR-01080-J-00010"), ("CUL Add.3430", "MS-ADD-03430"), ("Moss. I,1", "MS-MOSSERI-I-00001"),
    ("T-S 13J4.15.1", "MS-TS-00013-J-00004-00015-00001"), ("T-S 13J1.10 + T-S 13J1.11", None), ("ENA 2558.1", None),
])
def test_cudl_manifest_is_unchanged(shelfmark, url):
    assert asq.cudl_manifest(shelfmark, frozenset()) == (asq.CUDL_BASE + url if url else None)


def test_cudl_manifest_drops_a_document_number_of_a_known_fragment(inputs, monkeypatch):
    assert asq.cudl_manifest("T-S 13J4.15.1", {"T-S 13J4.15"}) == asq.CUDL_BASE + "MS-TS-00013-J-00004-00015"
    monkeypatch.setattr(asq, "PGP", inputs["with"])                    # no list given: the export under PGP, read when needed
    assert asq.cudl_manifest("T-S 13J4.15.1") == asq.CUDL_BASE + "MS-TS-00013-J-00004-00015"
    monkeypatch.setattr(asq, "PGP", inputs["without"])
    assert asq.cudl_manifest("T-S 13J4.15.1") == asq.CUDL_BASE + "MS-TS-00013-J-00004-00015-00001"


def test_known_fragments_reads_the_chosen_export(inputs, tmp_path):
    assert asq.known_fragments(inputs["with"]) == {"T-S 13J4.15", "T-S Ar.1.2"}
    assert asq.known_fragments(inputs["without"]) == {"T-S Ar.1.2"} and asq.known_fragments(tmp_path / "nowhere") == frozenset()


def test_import_reads_no_pgp_file():
    code = ("import builtins, json\n"
            "opened, real = [], builtins.open\n"
            "builtins.open = lambda file, *a, **k: opened.append(str(file)) or real(file, *a, **k)\n"
            "from src.datasets.indexing import arabic_scrape_queue as m\n"
            "builtins.open = real\n"
            "print(json.dumps({'opened': [f for f in opened if f.endswith('.csv') or 'pgp_raw' in f],\n"
            "                  'cached': m._fragment_shelfmarks.cache_info().currsize, 'eager': hasattr(m, '_KNOWN_FRAGMENTS')}))\n")
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True,
                            env={**os.environ, "PYTHONPATH": str(REPO)})
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"opened": [], "cached": 0, "eager": False}


def test_build_reads_the_chosen_pgp_dir(inputs, monkeypatch):
    rows, summary = asq.build(pgp_dir=inputs["with"])
    by_id = {r["pgpid"]: r for r in rows}
    assert summary["arabic_primary_docs"] == 3 and sorted(by_id) == ["1", "2", "4"]           # 3 is Judaeo-Arabic
    assert (by_id["1"]["tier"], by_id["2"]["tier"], by_id["4"]["tier"]) == (6, 1, 5)
    assert by_id["1"]["suggested_iiif"] == asq.CUDL_BASE + "MS-TS-00013-J-00004-00015" and by_id["2"]["canonical_id"] == "Cambridge_T_S_Ar_1_2"
    assert by_id["2"]["pgp_transcription_letters"] == len(asq._ARABIC.findall(ARABIC_LONG)) and by_id["2"]["library"] == "CUL"
    rows, _ = asq.build(pgp_dir=inputs["without"])
    assert {r["pgpid"]: r["suggested_iiif"] for r in rows}["1"] == asq.CUDL_BASE + "MS-TS-00013-J-00004-00015-00001"
    monkeypatch.setattr(asq, "PGP", inputs["with"])                    # default: the export under PGP
    assert asq.build() == asq.build(pgp_dir=inputs["with"])


def test_main_writes_to_the_out_dir(inputs, tmp_path, capsys):
    out = tmp_path / "next_round" / "queue"
    asq.main(["--pgp-dir", str(inputs["with"]), "--out-dir", str(out)])
    assert sorted(p.name for p in out.iterdir()) == ["arabic_image_only_backlog.csv", "arabic_scrape_priority_queue.csv",
                                                      "arabic_scrape_priority_summary.json"]
    assert list(inputs["default_out"].iterdir()) == []                                  # the default location is untouched
    queue = _read(out / "arabic_scrape_priority_queue.csv")
    assert [(r["rank"], r["pgpid"], r["tier"]) for r in queue] == [("1", "2", "1"), ("2", "1", "6")]
    assert list(queue[0]) == asq.COLUMNS and [r["pgpid"] for r in _read(out / "arabic_image_only_backlog.csv")] == ["4"]
    summary = json.loads((out / "arabic_scrape_priority_summary.json").read_text())
    assert summary["main_queue_rows"] == 2 and summary["backlog_rows"] == 1 and str(out) in capsys.readouterr().out


def test_the_summary_records_the_export_and_the_builder_can_read_it(inputs, tmp_path):
    (inputs["with"] / "COMMIT").write_text("b2343a17\n", encoding="utf-8")
    out = tmp_path / "queue"
    asq.main(["--pgp-dir", str(inputs["with"]), "--out-dir", str(out)])
    summary = json.loads((out / "arabic_scrape_priority_summary.json").read_text())
    assert (summary["pgp_dir"], summary["pgp_export_commit"]) == (str(inputs["with"]), "b2343a17")
    assert asq.queue_provenance(out / "arabic_scrape_priority_queue.csv") == {"pgp_dir": str(inputs["with"]), "pgp_export_commit": "b2343a17"}
    assert asq.pgp_export_commit(inputs["with"]) == "b2343a17" and asq.pgp_export_commit(inputs["without"]) == ""
    assert asq.queue_provenance(tmp_path / "elsewhere" / "queue.csv") == {"pgp_dir": "", "pgp_export_commit": ""}   # no summary


def test_main_refuses_to_replace_the_default_queue_with_another_export(inputs, monkeypatch, capsys):
    default_out = inputs["default_out"]
    upper = default_out.with_name(default_out.name.upper())                     # the same directory on APFS and the NAS share
    cases = [([], "give --out-dir"), (["--out-dir", str(default_out)], "must not replace it")]
    if upper.exists():
        cases.append((["--out-dir", str(upper)], "must not replace it"))
    for extra, message in cases:
        with pytest.raises(SystemExit):
            asq.main(["--pgp-dir", str(inputs["with"])] + extra)
        assert message in capsys.readouterr().err
        assert list(default_out.iterdir()) == []                                                    # nothing written over
    assert asq.out_dir_problem(inputs["with"], None).startswith("--pgp-dir") and asq.out_dir_problem(None, None) == ""
    assert asq.out_dir_problem(inputs["with"], inputs["default_out"].parent / "next_round") == ""
    monkeypatch.setattr(asq, "PGP", inputs["with"])                    # naming the default export is the default run
    asq.main(["--pgp-dir", str(inputs["with"])])
    assert sorted(p.name for p in inputs["default_out"].iterdir()) == ["arabic_image_only_backlog.csv", "arabic_scrape_priority_queue.csv",
                                                                       "arabic_scrape_priority_summary.json"]


def test_main_defaults_write_the_same_files_to_the_merged_dir(inputs, tmp_path, monkeypatch):
    assert asq.output_paths() == (asq.OUT_CSV, asq.OUT_BACKLOG, asq.OUT_SUMMARY)
    assert asq.output_paths(tmp_path) == (tmp_path / "arabic_scrape_priority_queue.csv", tmp_path / "arabic_image_only_backlog.csv",
                                          tmp_path / "arabic_scrape_priority_summary.json")
    asq.main(["--pgp-dir", str(inputs["with"]), "--out-dir", str(tmp_path / "flagged")])
    monkeypatch.setattr(asq, "PGP", inputs["with"])
    asq.main([])
    for path in (asq.OUT_CSV, asq.OUT_BACKLOG, asq.OUT_SUMMARY):
        assert path.parent == inputs["default_out"] and path.read_bytes() == (tmp_path / "flagged" / path.name).read_bytes()


# --- image-store photographs (permitted since the 2026-11-01 rule change) and merged-record joins -------------------

def _row(letters: int, gt_source: str, image_source: str, in_training: bool = False, store_letters: int = 0) -> Dict[str, Any]:
    """A queue row with only the facts :func:`asq.tier_for` reads.

    :param letters: PGP transcription Arabic letters.
    :type letters: int
    :param gt_source: Ground-truth source.
    :type gt_source: str
    :param image_source: Image source class.
    :type image_source: str
    :param in_training: Whether the document is in the training set.
    :type in_training: bool
    :param store_letters: Arabic letters of the image store's transcription.
    :type store_letters: int
    :return: The row.
    :rtype: Dict[str, Any]
    """
    return {"pgp_transcription_letters": letters, "gt_source": gt_source, "image_source": image_source,
            "in_training_clean_v2": in_training, "fjp_transcription_letters": store_letters}


@pytest.mark.parametrize("row, tier", [
    (_row(300, "pgp_export", "fjp_only"), 2),                       # benchmark-grade text, store photographs: never tier 1
    (_row(300, "pgp_export", "iiif_pgp"), 1), (_row(300, "pgp_export", "iiif_cudl_constructed"), 1),
    (_row(300, "pgp_export", "fjp_only", in_training=True), 2), (_row(50, "pgp_export", "fjp_only"), 2),
    (_row(0, "pgp_site", "fjp_only"), 3), (_row(0, "fjp_only", "fjp_only", store_letters=50), 4),
    (_row(0, "none", "fjp_only"), 5),                               # was 0 (not listed) before 2026-11-01
    (_row(300, "pgp_export", "none"), 6), (_row(0, "none", "none"), 0),
])
def test_tier_for_permits_store_photographs_but_never_as_tier_1(row: Dict[str, Any], tier: int) -> None:
    """Store photographs are a permitted image source; the cascade is unchanged except that tier 1 needs a library image.

    :param row: Queue row.
    :type row: Dict[str, Any]
    :param tier: Expected tier.
    :type tier: int
    """
    assert asq.tier_for(row) == tier
    assert "fjp_only" in asq.PERMITTED_IMAGE_SOURCES and "fjp_only" not in asq.LIBRARY_IMAGE_SOURCES


def _record(canonical_id: str, display: str, pgpids: Sequence[str] = (), store: Sequence[str] = (),
            ktiv: Sequence[str] = (), store_text: str = "") -> Dict[str, Any]:
    """A merged-corpus record (one JSONL line, parsed).

    :param canonical_id: Canonical id.
    :type canonical_id: str
    :param display: Display shelf mark.
    :type display: str
    :param pgpids: Linked PGP ids.
    :type pgpids: Sequence[str]
    :param store: Image-store photographs as stored (bare names or URLs).
    :type store: Sequence[str]
    :param ktiv: Ktiv images.
    :type ktiv: Sequence[str]
    :param store_text: Image-store transcription text.
    :type store_text: str
    :return: The record.
    :rtype: Dict[str, Any]
    """
    return {"canonical_id": canonical_id, "shelfmark_display": display, "pgpids": list(pgpids),
            "sources": {"fjp": {"transcription": store_text}} if store_text else {},
            "images": {"fjp": list(store), "ktiv": list(ktiv) or None, "bodleian": None}}


def _view(canonical_id: str, display: str, store: Sequence[str] = (), ktiv: bool = False) -> Dict[str, Any]:
    """The queue's view of a merged record, as :func:`asq.pick_merged_record` receives it.

    :param canonical_id: Canonical id.
    :type canonical_id: str
    :param display: Display shelf mark.
    :type display: str
    :param store: Image-store photographs.
    :type store: Sequence[str]
    :param ktiv: Whether it lists Ktiv images.
    :type ktiv: bool
    :return: The view.
    :rtype: Dict[str, Any]
    """
    return asq._merged_record(_record(canonical_id, display, store=store, ktiv=["k.jpg"] if ktiv else []))


def test_pick_merged_record_keeps_the_first_unless_it_has_no_images_and_a_later_one_answers_to_the_shelf() -> None:
    """First record wins; an image-less first record yields to a later record with images whose key is the shelf's."""
    bare, stored = _view("Cambridge_T_S_Ar_1_2", "T-S Ar.1.2"), _view("Cambridge_T_S_Ar_1_2_b", "T-S Ar. 1.2", store=["a_1r.jpg"])
    other = _view("Cambridge_T_S_Ar_1_20", "T-S Ar.1.20", store=["b_1r.jpg"])
    assert asq.pick_merged_record([], "T-S Ar.1.2") == {}
    assert asq.pick_merged_record([bare, stored], "T-S Ar.1.2") is stored                  # the exception
    assert asq.pick_merged_record([bare, other, stored], "T-S Ar.1.2") is stored           # the first later one that answers
    assert asq.pick_merged_record([bare, other], "T-S Ar.1.2") is bare                     # images, but another fragment
    assert asq.pick_merged_record([stored, _view("x", "T-S Ar.1.2", ktiv=True)], "T-S Ar.1.2") is stored   # first has images
    assert asq.pick_merged_record([bare, _view("y", "T-S Ar.1.2")], "T-S Ar.1.2") is bare  # the later one has no images
    by_id = _view("Cambridge_T_S_Ar_1_2", "", ktiv=True)                                   # no display: the canonical id's key
    assert asq.pick_merged_record([bare, by_id], "Cambridge_T_S_Ar_1_2") is by_id


def test_load_merged_index_keeps_every_linked_record_and_indexes_unlinked_ones_by_shelf_key(inputs: Dict[str, Path]) -> None:
    """Linked records stay in file order per PGP id; unlinked ones answer to their display and canonical-id keys.

    :param inputs: Fixture inputs (``asq.MERGED`` is rewritten here).
    :type inputs: Dict[str, Path]
    """
    records = [_record("Cambridge_T_S_Ar_1_2", "T-S Ar.1.2", pgpids=["2"]),
               _record("Cambridge_T_S_Ar_1_2_b", "T-S Ar.1.2", pgpids=["2", "2"], store=["a_1r.jpg", "https://store/a_1v.jpg"]),
               _record("Paris_AIU_VII_E_43", "AIU VII.E.43", store=["VII_E_43_1r.jpg"], store_text="بسم الله الرحمن الرحيم والصلاة")]
    asq.MERGED.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    linked, unlinked = asq.load_merged_index()
    assert [r["canonical_id"] for r in linked["2"]] == ["Cambridge_T_S_Ar_1_2", "Cambridge_T_S_Ar_1_2_b"]   # a repeated id counts once
    assert linked["2"][1]["fjp_images"] == ["a_1r.jpg", "https://store/a_1v.jpg"] and linked["2"][1]["fjp"] is True
    first = asq.load_merged_by_pgpid()["2"]                                                # the first record, as before, plus the new keys
    assert (first["canonical_id"], first["shelfmark_display"], first["fjp_images"], first["fjp"]) == ("Cambridge_T_S_Ar_1_2", "T-S Ar.1.2", [], False)
    assert set(unlinked) == {asq.loose_key("AIU VII.E.43"), ShelfmarkNormalizer.loose_key("Paris_AIU_VII_E_43")}
    assert all(recs[0]["canonical_id"] == "Paris_AIU_VII_E_43" and recs[0]["fjp_arabic"] > 0 for recs in unlinked.values())
    assert asq.load_merged_unlinked_by_key() == unlinked


def test_join_merged_falls_back_to_one_unlinked_record_and_refuses_several() -> None:
    """No linked record: one unlinked record answering to the shelf mark is used; several are ambiguous (none used)."""
    one, twin = _view("Paris_AIU_VII_E_43", "AIU VII.E.43", store=["a.jpg"]), _view("Paris_AIU_VII_E_43_dup", "AIU VII.E.43")
    linked = {"7": [_view("Cambridge_T_S_Ar_1_2", "T-S Ar.1.2")]}
    assert asq.join_merged("7", "AIU VII.E.43", linked, {asq.loose_key("AIU VII.E.43"): [one]}) == (linked["7"][0], "pgpid", "")
    assert asq.join_merged("8", "AIU VII.E.43", linked, {asq.loose_key("AIU VII.E.43"): [one]}) == (one, "shelfmark", "shelfmark_fallback_used")
    assert asq.join_merged("8", "AIU VII.E.43", linked, {asq.loose_key("AIU VII.E.43"): [one, twin]}) == ({}, "", "shelfmark_fallback_ambiguous")
    assert asq.join_merged("8", "AIU VII.E.44", linked, {asq.loose_key("AIU VII.E.43"): [one]}) == ({}, "", "")


STORE_DOCUMENT = {"pgpid": "5", "shelfmark": "AIU XII.80", "type": "Letter", "languages_primary": "Arabic"}


@pytest.fixture
def store_inputs(inputs: Dict[str, Path], tmp_path: Path) -> Dict[str, Path]:
    """An export with a fifth document and a merged corpus exercising every join.

    Document 1 (edition, no manifest) has store photographs only through an unlinked record; document 2's first linked
    record has no images and its second (Ktiv images) answers to its shelf mark; two unlinked records answer to
    document 4's shelf mark; document 5 (no text, no manifest) has store photographs through its linked record.

    :param inputs: Fixture inputs.
    :type inputs: Dict[str, Path]
    :param tmp_path: Pytest temporary directory.
    :type tmp_path: Path
    :return: ``inputs`` plus ``"store"``: the export with document 5.
    :rtype: Dict[str, Path]
    """
    records = [_record("Cambridge_T_S_Ar_1_2_stub", "T-S Ar.1.20", pgpids=["2"]),
               _record("Cambridge_T_S_Ar_1_2", "T-S Ar.1.2", pgpids=["2"], ktiv=["IE1/FL2"]),
               _record("Cambridge_CUL_T_S_13J4_15_1", "T-S 13J4.15.1", store=["T_S_13J4_15_1_1r.jpg", "https://store/T_S_13J4_15_1_1v.jpg"]),
               _record("Cambridge_T_S_Ar_1_4", "T-S Ar.1.4", store=["a.jpg"]), _record("Cambridge_CUL_T_S_Ar_1_4", "T-S Ar. 1.4"),
               _record("Paris_AIU_XII_80", "AIU XII.80", pgpids=["5"], store=["XII_80_1r.jpg", "XII_80_1v.jpg"])]
    asq.MERGED.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return dict(inputs, store=_pgp_dir(tmp_path / "pgp_store", ["T-S Ar.1.2"], [STORE_DOCUMENT]))


def test_build_lists_store_photographs_and_records_the_joins(store_inputs: Dict[str, Path]) -> None:
    """Store-only rows are listed (tier 2 with an edition, tier 5 without); joins and summary counts are recorded.

    :param store_inputs: Fixture inputs with the store merged corpus.
    :type store_inputs: Dict[str, Path]
    """
    rows, summary = asq.build(pgp_dir=store_inputs["store"])
    by_id = {r["pgpid"]: r for r in rows}
    assert sorted(by_id) == ["1", "2", "4", "5"]                                            # 5 was not listed before 2026-11-01
    one = by_id["1"]                                                                        # was tier 6 (no permitted image)
    assert (one["tier"], one["image_source"], one["merged_join"], one["canonical_id"]) == (2, "fjp_only", "shelfmark", "Cambridge_CUL_T_S_13J4_15_1")
    assert one["fjp_images"] == "T_S_13J4_15_1_1r.jpg;https://store/T_S_13J4_15_1_1v.jpg"
    two = by_id["2"]                                                                        # multi-link resolved by the shelf mark
    assert (two["tier"], two["image_source"], two["merged_join"], two["canonical_id"]) == (1, "iiif_pgp+ktiv", "pgpid", "Cambridge_T_S_Ar_1_2")
    assert (by_id["4"]["merged_join"], by_id["4"]["canonical_id"], by_id["4"]["fjp_images"]) == ("", "", "")   # ambiguous: no record
    assert (by_id["5"]["tier"], by_id["5"]["image_source"], by_id["5"]["merged_join"]) == (5, "fjp_only", "pgpid")
    assert {k: summary[k] for k in ("fjp_only_listed", "shelfmark_fallback_used", "multilink_resolved_by_shelfmark",
                                    "shelfmark_fallback_ambiguous")} == {"fjp_only_listed": 2, "shelfmark_fallback_used": 1,
                                                                         "multilink_resolved_by_shelfmark": 1, "shelfmark_fallback_ambiguous": 1}


def test_a_resolving_library_manifest_replaces_store_photographs(store_inputs: Dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """A document with store photographs is still a CUDL candidate; a resolving manifest makes it tier 1.

    :param store_inputs: Fixture inputs with the store merged corpus.
    :type store_inputs: Dict[str, Path]
    :param monkeypatch: Pytest monkeypatch.
    :type monkeypatch: pytest.MonkeyPatch
    """
    asked: List[str] = []
    monkeypatch.setattr(asq, "verify_urls", lambda urls, delay_s=0.25: asked.extend(urls) or {u: True for u in urls})
    rows, summary = asq.build(verify_cudl=True, pgp_dir=store_inputs["store"])
    one = {r["pgpid"]: r for r in rows}["1"]
    assert asked == [asq.CUDL_BASE + "MS-TS-00013-J-00004-00015-00001"] and summary["cudl_verified_resolving"] == 1
    assert (one["tier"], one["image_source"], one["fjp_images"].count(";")) == (1, "iiif_cudl_constructed", 1)
    assert summary["fjp_only_listed"] == 1                                                  # only document 5 now


def test_main_writes_the_store_columns_and_they_round_trip(store_inputs: Dict[str, Path], tmp_path: Path) -> None:
    """``fjp_images`` and ``merged_join`` are the trailing columns of both CSVs and read back as written.

    :param store_inputs: Fixture inputs with the store merged corpus.
    :type store_inputs: Dict[str, Path]
    :param tmp_path: Pytest temporary directory.
    :type tmp_path: Path
    """
    out = tmp_path / "queue_store"
    asq.main(["--pgp-dir", str(store_inputs["store"]), "--out-dir", str(out)])
    queue, backlog = _read(out / "arabic_scrape_priority_queue.csv"), _read(out / "arabic_image_only_backlog.csv")
    assert asq.COLUMNS[-2:] == ["fjp_images", "merged_join"] and list(queue[0]) == asq.COLUMNS == list(backlog[0])
    assert [(r["pgpid"], r["tier"], r["merged_join"]) for r in queue] == [("2", "1", "pgpid"), ("1", "2", "shelfmark")]
    assert queue[1]["fjp_images"].split(";") == ["T_S_13J4_15_1_1r.jpg", "https://store/T_S_13J4_15_1_1v.jpg"]
    assert [(r["pgpid"], r["image_source"], r["fjp_images"], r["merged_join"]) for r in backlog] == [   # letters before lists
        ("5", "fjp_only", "XII_80_1r.jpg;XII_80_1v.jpg", "pgpid"), ("4", "iiif_pgp", "", "")]
    summary = json.loads((out / "arabic_scrape_priority_summary.json").read_text())
    assert summary["fjp_only_listed"] == 2 and summary["shelfmark_fallback_ambiguous"] == 1
