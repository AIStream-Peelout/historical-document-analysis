"""Tests for ``novel_build.pool``: ledger join, normalisation, filters, RNL flag, images, stage file (synthetic)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.finetuning.qwen_hebrew.novel_build import paths
from src.finetuning.qwen_hebrew.novel_build import pool as P
from src.finetuning.qwen_hebrew.novel_build import targets as T
from src.finetuning.qwen_hebrew.novel_build.schema import Candidate, read_jsonl, write_jsonl

# Synthetic lines: invented letter groups, no real transcription.
R1, R2, R3 = "אבגד הוזח טיכל מנסע", "פצקר שתאב גדהו זחטי", "כלמנ סעפצ קרשת אבגד"
V1, V2 = "הוזח טיכל מנסע פצקר", "שתאב גדהו זחטי כלמנ"
A1, A3 = "ابتث جحخد ذرزس", "نهوي ابتث جحخد"
BUCKET = "https://example.invalid/bucket"


def _jsonl(path: Path, rows: List[Dict[str, Any]]) -> Path:
    """Write rows as JSON lines.

    :param path: File.
    :param rows: Rows.
    :return: The path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def _ledger(row_id: str, source: str, verdict: str = "novel", flags: Any = None, **fields: Any) -> Dict[str, Any]:
    """One ledger record with every field the build reads.

    :param row_id: Row id.
    :param source: Source.
    :param verdict: Verdict.
    :param flags: Extra ``flags`` entries.
    :param fields: Field overrides.
    :return: The record.
    """
    record: Dict[str, Any] = {
        "row_id": row_id, "source": source, "stem": None, "family": "document", "image_sha256": None, "phash": None,
        "image_url": None, "image_path": None, "pgpid": None, "canonical_id": None, "shelfmark": "Testville Box 1",
        "shelf_key": "box1", "ktiv_id": None, "side": None, "script": "hebrew", "n_letters": 40, "notes": "",
        "unit": "document", "n_images_known": 0, "image_sources": ["none"], "edition_source": "synthetic",
        "held_out_hint": None, "languages": None, "cids_es": [], "shelf_keys_clean": [], "source_file": source,
        "document_group": f"canonical_id:{row_id}"}
    record.update(fields)
    record["flags"] = {"image": [], "image_uncertain": [], "document": [], "document_uncertain": [],
                       "text_best_definite": None, "text_best_uncertain": None, "witnesses": [], "held_out": None,
                       "verdict": verdict, "page_level_novel": verdict == "novel", **(flags or {})}
    return record


@pytest.fixture()
def inputs(tmp_path: Path) -> Dict[str, Path]:
    """A ledger, candidate files and the per-source inputs of every target path.

    :param tmp_path: Pytest temp dir.
    :return: Paths by name.
    """
    local = tmp_path / "store" / "Testville_Lib_A_1" / "a_1r.jpg"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"jpeg")
    ledger = [
        _ledger("pgp:101", "pgp", pgpid="101", canonical_id="Testville_Lib_TS_1", cids_es=["Testville_Lib_TS_1_es"],
                canonical_ids=["Testville_Lib_TS_1", "Otherplace_Lib_X_2"], image_sources=["iiif_pgp"],
                image_url="https://example.invalid/iiif/manifest/101",
                languages={"primary": "Hebrew", "secondary": ""}),
        _ledger("pgp:102", "pgp", pgpid="102", script="arabic", image_sources=["fjp_pgpid_mirror", "iiif_pgp"],
                image_url=f"{BUCKET}/102_image.jpg"),
        _ledger("pgp:103", "pgp", verdict="uncertain_history", pgpid="103",
                flags={"image_uncertain": ["phash_far:setx@6"], "text_best_definite": [0.2, 31, "setx:abc"]}),
        _ledger("pgp:104", "pgp", verdict="held_out", pgpid="104",
                flags={"document": ["pgpid:bench_set"], "held_out": "benchmark document"}),
        _ledger("pgp:105", "pgp", pgpid="105", image_path=str(local), image_sha256="cd" * 32, phash="f0" * 8,
                image_sources=["fjp_local", "fjp_store_named", "iiif_cudl"],
                image_url="https://example.invalid/iiif/manifest/105"),
        _ledger("Testville_Lib_A_1:0#r1", "fjp", unit="side", canonical_id="Testville_Lib_A_1",
                cids_es=["Testville_Lib_A_1"], pgpid="201", pgpids=["201", "202"], shelfmark="Testville: A 1",
                images=[{"file_name": "a_1r.jpg", "path": str(local), "url": f"{BUCKET}/a_1r.jpg",
                         "image_sha256": "aa" * 32, "phash": "0f" * 8, "side_token": "1r", "slip_side": "v",
                         "slip_outcome": "confirmed"},
                        {"file_name": "a_1v.jpg", "path": None, "url": f"{BUCKET}/a_1v.jpg", "image_sha256": None,
                         "phash": None, "side_token": "1v", "slip_side": None, "slip_outcome": None}]),
        _ledger("Testville_Lib_B_2:s0", "fjp_scrape", unit="side", canonical_id="Testville_Lib_B_2",
                notes="shape=string; part=str; entry=0/1; block=0/1",
                images=[{"file_name": "b_1r.jpg", "url": f"{BUCKET}/b_1r.jpg", "path": None, "side_token": "1r",
                         "image_sha256": None, "phash": None}]),
        _ledger("990000000000205171:FL100", "ktiv", unit="page", ktiv_id="990000000000205171", image_sources=["ktiv"],
                image_url="https://nli.example.invalid/IIIFv21/FL100/full/2800,/0/default.jpg",
                notes="shape=api; bundle=ktiv_test_bundle.json", edition_source="nli_ktiv_transcription_api"),
        _ledger("lit:testwork:partA:p13:tsx1", "literature", edition_source="literature:testwork/partA pages 13,14",
                resolution={"status": "resolved", "raw_keys": ["TS X 1"], "canonical_ids": [], "pgpids": ["301"]}),
        _ledger("lit:testbook:1", "lit_run2", pgpid="401", shelfmark="RNL Yevr. II A 999",
                resolution={"status": "resolved", "canonical_ids": [], "pgpids": ["401", "402"], "ambiguous": False}),
        _ledger("apd_v3:apd_test_0_page", "apd_v3", stem="apd_test_0_page", script="arabic", unit="page",
                image_sources=["iiif_other", "apd_local"], image_path=str(tmp_path / "elsewhere.jpg")),
    ]
    cdir = tmp_path / "candidates"
    _jsonl(cdir / "pgp.jsonl", [
        {"row_id": "pgp:101", "edition_text": "\n".join(["Recto", R1, R2, "", "Verso", V1]), "text_shingles": ["x"],
         "shelfmark": "ignored: the ledger wins"},
        {"row_id": "pgp:102", "edition_text": "\n".join(["Recto", A1, "Verso", A3])},
        {"row_id": "pgp:105", "edition_text": R3}])
    _jsonl(cdir / "fjp.jsonl", [{"row_id": "Testville_Lib_A_1:0#r1", "edition_text": V1, "funnel": "accepted"}])
    _jsonl(cdir / "fjp_scrape.jsonl", [{"row_id": "Testville_Lib_B_2:s0", "edition_text": "1 אבגד הוזח [טיכל] מנסע",
                                        "side": None, "notes": "shape=string; part=str; entry=0/1; block=0/1"}])
    _jsonl(cdir / "ktiv.jsonl", [{"row_id": "990000000000205171:FL100", "edition_text": f"{R1}\n{R2}",
                                  "notes": "shape=api; bundle=ktiv_test_bundle.json",
                                  "edition_source": "nli_ktiv_transcription_api"}])
    _jsonl(cdir / "literature.jsonl", [{"row_id": "lit:testwork:partA:p13:tsx1", "edition_text": f"{R1}\n{R2}"}])
    _jsonl(cdir / "apd_v3.jsonl", [{"row_id": "apd_v3:apd_test_0_page", "edition_text": f"{A1}\n{A3}"}])
    sections = _jsonl(tmp_path / "cleaned_sections.jsonl", [
        {"canonical_id": "Testville_Lib_A_1", "section_index": 0, "side": "1r", "side_decision": "marker", "text": R1},
        {"canonical_id": "Testville_Lib_A_1", "section_index": 0, "side": "1v", "side_decision": "marker", "text": V1}])
    page = "testwork/partA/s/page_{:03d}_structured.json"
    blocks = _jsonl(tmp_path / "lit_blocks.jsonl", [
        {"work": "testwork", "part": "partA", "page_file": page.format(14), "shelf_key": "TS X 1", "text": R2},
        {"work": "testwork", "part": "partA", "page_file": page.format(13), "shelf_key": "TS X 1", "text": R1},
        {"work": "testwork", "part": "partA", "page_file": page.format(13), "shelf_key": "TS X 2", "text": V2}])
    run2 = _jsonl(tmp_path / "run2.jsonl", [{"doc_id": "lit:testbook:1", "transcription_lines": [
        {"n": 1, "side": "recto", "part": None, "text": R1}, {"n": 2, "side": "recto", "part": None, "text": R2},
        {"n": 1, "side": "verso", "part": None, "text": V1}]}])
    root = tmp_path / "datasets"
    apd_dir = root / "arabic_external" / "apd" / "apd_arabic_pages_v3"
    apd_image = tmp_path / "apd_0.jpg"
    apd_image.write_bytes(b"jpeg")
    _jsonl(apd_dir / "manifest.jsonl", [{"stem": "apd_test_0_page", "side": "verso", "canvas": 0,
                                         "image_path": str(apd_image),
                                         "image_url": "https://example.invalid/apd/0.jpg"}])
    return {"pool": _jsonl(tmp_path / "pool.jsonl", ledger), "candidates": cdir, "sections": sections,
            "blocks": blocks, "run2": run2, "root": root, "work": tmp_path / "work", "local": local}


def _args(inputs: Dict[str, Path], *extra: str) -> List[str]:
    """CLI arguments pointing at the fixture inputs.

    :param inputs: Fixture paths.
    :param extra: More arguments.
    :return: The argument list.
    """
    return ["--pool", str(inputs["pool"]), "--candidates-dir", str(inputs["candidates"]),
            "--work-dir", str(inputs["work"]), "--sections", str(inputs["sections"]),
            "--lit-blocks", str(inputs["blocks"]), "--run2", str(inputs["run2"]),
            "--datasets-root", str(inputs["root"]), *extra]


def test_normalisation_of_ids_languages_and_notes(inputs: Dict[str, Path]) -> None:
    """Ids merged into lists, ledger values kept, candidate-only fields joined, edition text not carried."""
    cands = {c.row_id: c for c in P.load_candidates(inputs["pool"], inputs["candidates"])}
    pgp = cands["pgp:101"]
    assert pgp.canonical_ids == ["Testville_Lib_TS_1_es", "Testville_Lib_TS_1", "Otherplace_Lib_X_2"]
    assert pgp.pgpids == ["101"] and pgp.languages_stated == {"primary": "Hebrew", "secondary": ""}
    assert (pgp.verdict, pgp.witnesses, pgp.unit, pgp.n_letters, pgp.script) == ("novel", [], "document", 40, "hebrew")
    assert pgp.shelfmark == "Testville Box 1" and pgp.document_group == "canonical_id:pgp:101"
    assert pgp.notes["candidate_record"] == "joined" and pgp.notes["edition_source"] == "synthetic"
    assert "edition_text" not in pgp.notes and "text_shingles" not in pgp.notes
    assert pgp.notes["ledger_flags"]["verdict"] == "novel" and pgp.targets == []
    fjp = cands["Testville_Lib_A_1:0#r1"]
    assert fjp.pgpids == ["201", "202"] and fjp.notes["funnel"] == "accepted"
    assert cands["lit:testbook:1"].pgpids == ["401", "402"]
    assert cands["990000000000205171:FL100"].ktiv_id == "990000000000205171"


def test_witnesses_only_outside_novel(inputs: Dict[str, Path]) -> None:
    """Evidence strings for uncertain and held-out records; a novel record has none."""
    cands = {c.row_id: c for c in P.load_candidates(inputs["pool"], inputs["candidates"], sources=["pgp"],
                                                    verdicts=P.VERDICTS)}
    assert cands["pgp:103"].witnesses == ["phash_far:setx@6", "text_definite:setx:abc@0.200/31"]
    assert cands["pgp:104"].witnesses == ["pgpid:bench_set", "held_out:benchmark document"]
    assert cands["pgp:101"].witnesses == []


def test_source_verdict_and_limit_filters(inputs: Dict[str, Path]) -> None:
    """Sources and verdicts filter the ledger; the limit caps every source separately, in ledger order."""
    pgp_only = P.load_candidates(inputs["pool"], inputs["candidates"], sources=["pgp"])
    assert [c.row_id for c in pgp_only] == ["pgp:101", "pgp:102", "pgp:105"]
    uncertain = P.load_candidates(inputs["pool"], inputs["candidates"], verdicts=("uncertain_history",))
    assert [c.row_id for c in uncertain] == ["pgp:103"]
    limited = P.load_candidates(inputs["pool"], inputs["candidates"], limit=1)
    assert [c.row_id for c in limited if c.source == "pgp"] == ["pgp:101"]
    assert sorted({c.source for c in limited}) == sorted(set(P.SOURCES) - {"aodeh_v2_1"})
    assert P.load_candidates(inputs["pool"], inputs["candidates"], sources=["ktiv", "fjp"], limit=0) == []


def test_rnl_flag(inputs: Dict[str, Path]) -> None:
    """The holder flag fires on RNL sigla and collection names in any id, mark or label, and nowhere else."""
    cands = {c.row_id: c.rnl for c in P.load_candidates(inputs["pool"], inputs["candidates"])}
    assert cands.pop("lit:testbook:1") is True and not any(cands.values())
    for record in ({"canonical_id": "StPetersburg_NLR_Yevr_II_A_9"}, {"shelfmark": "Firkovitch II 9999 f. 1"},
                   {"shelfmark": "MS St Petersburg Evr Ar II 9999"}, {"shelfmark": "Antonin 999"},
                   {"resolution": {"labels": ["(Firkovich II) 9999, f. 1"]}}, {"shelf_key": "rnlyevrarab9999"},
                   {"notes": "library=NLR"}):
        assert P.is_rnl(record), record
    for record in ({"shelfmark": "T-S 99.1"}, {"notes": "library=CUL"}, {"shelfmark": "Testville Evrard 9"}):
        assert not P.is_rnl(record), record


def test_legacy_store_images_and_labels(inputs: Dict[str, Path]) -> None:
    """Every listed photograph; local files present, URL-only listed; the slip side replaces the file token."""
    cands = {c.row_id: c for c in P.load_candidates(inputs["pool"], inputs["candidates"])}
    first, second = cands["Testville_Lib_A_1:0#r1"].images
    assert (first.image_id, first.origin, first.status, first.path) == ("fjp_local:0", "fjp_local", "present",
                                                                        str(inputs["local"]))
    assert (first.sha256, first.phash, first.label, first.slip_outcome) == ("aa" * 32, "0f" * 8, "v", "confirmed")
    assert first.note == "side_token=1r; slip_side=v; conflict"
    assert (second.image_id, second.origin, second.status, second.label, second.url) == (
        "fjp_bucket:1", "fjp_bucket", "listed", "1v", f"{BUCKET}/a_1v.jpg")
    scrape = cands["Testville_Lib_B_2:s0"].images[0]
    assert (scrape.origin, scrape.status, scrape.label) == ("fjp_bucket", "listed", "1r")
    assert P.image_label("1v", "v") == "1v" and P.image_label("", "") == ""


def test_seed_image_of_other_sources(inputs: Dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """One reference: the local file first, else a photograph URL; a manifest URL is none; existence is stubbable."""
    monkeypatch.setattr(P, "URL_ORIGINS", (("nli.example.invalid", "iiif_nli"),))
    cands = {c.row_id: c for c in P.load_candidates(inputs["pool"], inputs["candidates"])}
    assert cands["pgp:101"].images == []
    mirror = cands["pgp:102"].images[0]
    assert (mirror.origin, mirror.url, mirror.path, mirror.status) == ("pgpid_mirror", f"{BUCKET}/102_image.jpg", None,
                                                                       "listed")
    local = cands["pgp:105"].images[0]
    assert (local.image_id, local.origin, local.path, local.url, local.status) == (
        "fjp_local:0", "fjp_local", str(inputs["local"]), None, "present")
    assert (local.sha256, local.phash) == ("cd" * 32, "f0" * 8)
    assert cands["990000000000205171:FL100"].images[0].origin == "iiif_nli"
    assert P.url_origin("https://elsewhere.example.invalid/a/b.jpg") == "iiif_other"
    assert not P.is_photo_url("https://example.invalid/iiif/manifest.json")
    monkeypatch.setattr(P, "path_exists", lambda path: False)
    stubbed = {c.row_id: c for c in P.load_candidates(inputs["pool"], inputs["candidates"], sources=["pgp", "fjp"])}
    assert stubbed["pgp:105"].images[0].status == "listed"
    assert [i.status for i in stubbed["Testville_Lib_A_1:0#r1"].images] == ["listed", "listed"]


def test_main_writes_the_stage_file_with_targets(inputs: Dict[str, Path], capsys: pytest.CaptureFixture) -> None:
    """``main``: load, attach targets, write the stage file, print per-source counts and empty-target reasons."""
    P.main(_args(inputs))
    report = json.loads(capsys.readouterr().out)
    stage = paths.stage_path(inputs["work"], "candidates")
    assert report["stage_file"] == str(stage) and report["candidates"] == 9
    assert report["by_source"]["pgp"] == {"candidates": 3, "with_targets": 3, "sides": 5, "images": 2,
                                          "images_present": 1, "rnl": 0}
    assert report["empty_targets"] == {}
    stored = {c.row_id: c for c in read_jsonl(stage, Candidate)}
    sides = {row_id: [(t.side_key, t.lines) for t in c.targets] for row_id, c in stored.items()}
    assert sides["pgp:101"] == [("recto", [R1, R2]), ("verso", [V1])]
    assert sides["pgp:102"] == [("recto", [A1]), ("verso", [A3])]
    assert sides["Testville_Lib_A_1:0#r1"] == [("1v", [V1])]
    assert sides["Testville_Lib_B_2:s0"] == [("", ["אבגד הוזח [...] מנסע"])]
    assert sides["990000000000205171:FL100"] == [("page", [R1, R2])]
    assert sides["lit:testwork:partA:p13:tsx1"] == [("", [R1, R2])]
    assert sides["lit:testbook:1"] == [("recto", [R1, R2]), ("verso", [V1])]
    assert sides["apd_v3:apd_test_0_page"] == [("verso", [A1, A3])]
    pgp = stored["pgp:101"].targets[0]
    assert (pgp.letters_hebrew, pgp.letters_arabic, pgp.pointer["pgpid"]) == (32, 0, "101")
    assert stored["990000000000205171:FL100"].targets[0].pointer["bundle"] == "ktiv_test_bundle.json"
    assert stored["lit:testwork:partA:p13:tsx1"].targets[0].pointer["page_files"] == [
        "testwork/partA/s/page_013_structured.json", "testwork/partA/s/page_014_structured.json"]
    apd_image = stored["apd_v3:apd_test_0_page"].images
    assert [(i.origin, i.status) for i in apd_image] == [("apd_local", "present")]


def test_main_filters_and_rejects_unknown_names(inputs: Dict[str, Path], capsys: pytest.CaptureFixture) -> None:
    """``--sources`` / ``--verdicts`` / ``--limit`` reach the loader; an unknown name is a usage error."""
    P.main(_args(inputs, "--sources", "pgp,ktiv", "--verdicts", "novel,held_out", "--limit", "2"))
    report = json.loads(capsys.readouterr().out)
    assert {s: v["candidates"] for s, v in report["by_source"].items()} == {"ktiv": 1, "pgp": 2}
    stored = [c.row_id for c in read_jsonl(paths.stage_path(inputs["work"], "candidates"), Candidate)]
    assert stored == ["pgp:101", "pgp:102", "990000000000205171:FL100"]
    with pytest.raises(SystemExit):
        P.main(_args(inputs, "--sources", "pgp,nonsense"))
    with pytest.raises(SystemExit):
        P.main(_args(inputs, "--verdicts", "maybe"))


def test_missing_candidate_record_is_reported(inputs: Dict[str, Path], capsys: pytest.CaptureFixture) -> None:
    """A ledger record without a candidate record is kept, marked, and left without targets with a reason."""
    ledger = list(read_jsonl(inputs["pool"])) + [_ledger("pgp:777", "pgp", pgpid="777")]
    _jsonl(inputs["pool"], ledger)
    P.main(_args(inputs, "--sources", "pgp"))
    report = json.loads(capsys.readouterr().out)
    assert report["empty_targets"] == {"pgp": {"no_candidate_record": {"count": 1, "examples": ["pgp:777"]}}}
    stored = {c.row_id: c for c in read_jsonl(paths.stage_path(inputs["work"], "candidates"), Candidate)}
    assert stored["pgp:777"].notes["candidate_record"] == "missing" and stored["pgp:777"].targets == []


def test_json_round_trip(inputs: Dict[str, Path], tmp_path: Path) -> None:
    """Candidates with targets and images survive ``write_jsonl`` / ``read_jsonl`` unchanged."""
    cands = P.load_candidates(inputs["pool"], inputs["candidates"], verdicts=P.VERDICTS)
    T.attach_targets(cands, candidates_dir=inputs["candidates"], sections_path=inputs["sections"],
                     lit_blocks_path=inputs["blocks"], run2_path=inputs["run2"], datasets_root=inputs["root"])
    assert any(c.targets for c in cands) and any(c.images for c in cands)
    path = tmp_path / "round_trip.jsonl"
    assert write_jsonl(path, cands) == len(cands)
    assert list(read_jsonl(path, Candidate)) == cands
