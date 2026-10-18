# File name: test_arabic_scrape_queue.py
# Date: 10/15/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for the Arabic-script scraping queue: PGP export and output directories, lazy fragment list."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest

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


def _pgp_dir(path: Path, fragments: List[str]) -> Path:
    """A tiny PGP export: documents, footnotes (editions of 1 and 2) and the given fragment shelfmarks."""
    path.mkdir(parents=True)
    _csv(path / "documents.csv", DOCUMENT_COLUMNS, [dict(row, url=f"https://pgp/{row['pgpid']}") for row in DOCUMENTS])
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
