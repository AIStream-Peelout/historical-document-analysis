"""Tests of the novel-rows funnel: stage counts, document groups, script switch, reasons, review queue, header digests.

A four-candidate synthetic pool (two candidates share one document group) plus three non-novel ledger records, with
stage files in the schema types written to ``tmp_path``. No network, no NAS.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.finetuning.qwen_hebrew.novel_build import funnel, paths
from src.finetuning.qwen_hebrew.novel_build.schema import (Candidate, HoldoutResult, ImageRef, QualityResult,
                                                           RowRecord, SideDecision, TargetSide, read_jsonl,
                                                           write_jsonl)

N1, N2, N3, N4 = "pgp:11", "Cambridge_CUL_T_S_97_4321:0", "990000000000000001:FL1", "lit:test_work:p1:k1"
G1, G2, G3 = "canonical_id:Cambridge_CUL_T_S_97_4321", "ktiv:990000000000000001", "canonical_id:Cambridge_CUL_Ar_9_9"
NOW = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)
TEXT = "אבג דהו\nזחט יכל\nמנס עפצ"


def _ledger(row_id: str, source: str, group: str, verdict: str = "novel", script: str = "hebrew",
            witnesses: Sequence[str] = (), held_out: Optional[str] = None,
            image_uncertain: Sequence[str] = ()) -> Dict[str, Any]:
    """A ledger record.

    :param row_id: Candidate id.
    :param source: Source.
    :param group: Document group.
    :param verdict: Verdict.
    :param script: Script.
    :param witnesses: Witness datasets.
    :param held_out: Held-out reason.
    :param image_uncertain: Uncertain image witnesses.
    :return: The record.
    """
    return {"row_id": row_id, "source": source, "script": script, "document_group": group,
            "flags": {"verdict": verdict, "witnesses": list(witnesses), "held_out": held_out, "image": [],
                      "document": [], "image_uncertain": list(image_uncertain), "document_uncertain": []}}


def _cand(row_id: str, source: str, group: str, sides: Sequence[str], n_images: int, script: str = "hebrew") -> Candidate:
    """A candidate with targets and found photographs.

    :param row_id: Candidate id.
    :param source: Source.
    :param group: Document group.
    :param sides: Target side keys.
    :param n_images: Present photographs.
    :param script: Ledger script.
    :return: The candidate.
    """
    targets = [TargetSide(side_key=s, text=TEXT, raw_text=TEXT, lines=TEXT.splitlines(), pointer={"side": s})
               for s in sides]
    images = [ImageRef(image_id=f"iiif_cudl:{i}", origin="iiif_cudl", path=f"/x/{row_id}/{i}.jpg",
                       sha256=f"{row_id}-{i}", status="present") for i in range(n_images)]
    return Candidate(row_id=row_id, source=source, document_group=group, canonical_ids=[], pgpids=[], ktiv_id=None,
                     shelfmark="T-S 97.4321", script=script, languages_stated=[], verdict="novel", witnesses=[],
                     unit="side", n_letters=40, targets=targets, images=images)


def _accepted(row_id: str, side: str) -> SideDecision:
    """An accepted decision.

    :param row_id: Candidate id.
    :param side: Side key.
    :return: The decision.
    """
    return SideDecision(row_id=row_id, side_key=side, outcome="accepted", reasons=[], image_sha256=f"{row_id}-0",
                        basis="p99", reader="kraken:test", recall=0.5)


def _ok(row_id: str, side: str, script: str = "hebrew") -> QualityResult:
    """A passed quality result.

    :param row_id: Candidate id.
    :param side: Side key.
    :param script: Measured script.
    :return: The result.
    """
    return QualityResult(row_id=row_id, side_key=side, ok=True, reasons=[], script=script, letters_hebrew=18,
                         letters_arabic=0, tokens=100, text=TEXT)


def _row(row_id: str, side: str, source: str, group: str, split: str, script: str = "hebrew") -> RowRecord:
    """A stage6 manifest record.

    :param row_id: Candidate id.
    :param side: Side key.
    :param source: Source.
    :param group: Document group.
    :param split: ``train`` or ``val``.
    :param script: Measured script.
    :return: The record.
    """
    return RowRecord(stem=f"novel_{source}_{row_id}_{side}", row_id=row_id, side_key=side, source=source,
                     document_group=group, canonical_ids=[], pgpids=[], image_sha256=f"{row_id}-0", image_path="/x",
                     image_url=None, image_origin="iiif_cudl", target_pointer={}, side_basis={"basis": "p99"},
                     script=script, languages_stated=[], verdict="novel", witnesses=[], quality_flags=[],
                     holdout_reasons=[], split=split, letters=18, target_tokens=100)


def make_work(tmp_path: Path) -> Path:
    """Write the ledger and the stage files of the four-candidate pool.

    N1 (two sides) becomes two training rows, its verso measured ``mixed``; N2 shares N1's document group and goes to
    review; N3 becomes a validation row; N4 has no photograph.

    :param tmp_path: Pytest temp dir.
    :return: The work dir (the ledger is ``tmp_path / "pool.jsonl"``).
    """
    work = tmp_path / "work"
    write_jsonl(tmp_path / "pool.jsonl", [
        _ledger(N1, "pgp", G1), _ledger(N2, "fjp", G1), _ledger(N3, "ktiv", G2), _ledger(N4, "literature", G3, script="arabic"),
        _ledger("pgp:20", "pgp", "canonical_id:G4", "held_out", held_out="registered_benchmark_ms"),
        _ledger("pgp:21", "pgp", "canonical_id:G5", "previously_trained", witnesses=["genizah_test_a", "genizah_test_b"]),
        _ledger("pgp:22", "pgp", "canonical_id:G6", "uncertain_history", image_uncertain=["phash_far:x"])])
    cands = [_cand(N1, "pgp", G1, ["recto", "verso"], 2), _cand(N2, "fjp", G1, [""], 1),
             _cand(N3, "ktiv", G2, ["page"], 1), _cand(N4, "literature", G3, ["recto"], 0, script="arabic")]
    write_jsonl(paths.stage_path(work, "candidates"), cands)
    write_jsonl(paths.stage_path(work, "images"), cands)
    write_jsonl(paths.stage_path(work, "sides"), [
        _accepted(N1, "recto"), _accepted(N1, "verso"),
        SideDecision(row_id=N2, side_key="", outcome="needs_review", reasons=["ownership_contested"]),
        _accepted(N3, "page")])
    write_jsonl(paths.stage_path(work, "quality"), [_ok(N1, "recto"), _ok(N1, "verso", "mixed"), _ok(N3, "page")])
    write_jsonl(paths.stage_path(work, "holdout"), [
        HoldoutResult(row_id=N1, side_key="recto", held_out=False, reasons=[], split="train", rnl=False),
        HoldoutResult(row_id=N1, side_key="verso", held_out=False, reasons=[], split="train", rnl=False),
        HoldoutResult(row_id=N3, side_key="page", held_out=False, reasons=[], split="val", rnl=False)])
    write_jsonl(paths.stage_path(work, "rows"), [
        _row(N1, "recto", "pgp", G1, "train"), _row(N1, "verso", "pgp", G1, "train", "mixed"),
        _row(N3, "page", "ktiv", G2, "val")])
    return work


def test_stage_counts_are_monotone_on_a_four_candidate_pool(tmp_path: Path) -> None:
    """Totals per stage from the ledger and the stage files; no stage exceeds the one before it."""
    work = make_work(tmp_path)
    result = funnel.compute_funnel(work, tmp_path / "pool.jsonl", now=NOW)
    total = result["counts"]["total"]
    got = {key: (cell["candidates"], cell["sides"], cell["groups"]) for key, cell in total.items()}
    assert got == {"inventoried": (7, None, 6), "held_out_ledger": (1, None, 1), "previously_trained": (1, None, 1),
                   "uncertain_history": (1, None, 1), "novel": (4, None, 3), "target_extracted": (4, 5, 3),
                   "image_found": (3, 4, 2), "side_verified": (2, 3, 2), "quality_passed": (2, 3, 2),
                   "held_out_builder": (0, 0, 0), "holdout_cleared": (2, 3, 2), "val_rows": (1, 1, 1),
                   "train_rows": (1, 2, 1), "rows": (2, 3, 2)}
    assert result["monotone_violations"] == []
    assert funnel.monotone_violations(result["counts"]) == []


def test_document_groups_are_counted_beside_candidates(tmp_path: Path) -> None:
    """Two candidates of one document group count as one group; sides of one candidate as one candidate."""
    work = make_work(tmp_path)
    by_source = funnel.compute_funnel(work, tmp_path / "pool.jsonl", now=NOW)["counts"]["by_source"]
    assert by_source["pgp"]["novel"] == {"candidates": 1, "sides": None, "groups": 1}
    assert by_source["fjp"]["image_found"] == {"candidates": 1, "sides": 1, "groups": 1}
    assert by_source["pgp"]["train_rows"] == {"candidates": 1, "sides": 2, "groups": 1}


def test_script_is_measured_from_quality_passed_on(tmp_path: Path) -> None:
    """Up to side verified the ledger's script counts; from quality passed on, the measured one."""
    work = make_work(tmp_path)
    counts = funnel.compute_funnel(work, tmp_path / "pool.jsonl", now=NOW)["counts"]
    assert counts["by_script"]["mixed"]["side_verified"]["sides"] == 0
    assert counts["by_script"]["mixed"]["quality_passed"]["sides"] == 1
    assert counts["by_script"]["hebrew"]["side_verified"]["sides"] == 3
    assert counts["by_source_script"]["pgp"]["mixed"]["train_rows"]["sides"] == 1
    assert counts["scripts"] == ["hebrew", "arabic", "mixed"]


def test_reasons_table_and_review_queue(tmp_path: Path) -> None:
    """Every stage's drop reasons per source, the validation rows listed apart, the review queue by reason."""
    work = make_work(tmp_path)
    result = funnel.compute_funnel(work, tmp_path / "pool.jsonl", now=NOW)
    assert result["reasons"] == {
        "held_out_ledger": {"registered_benchmark_ms": {"pgp": 1}},
        "previously_trained": {"genizah_test_a": {"pgp": 1}, "genizah_test_b": {"pgp": 1}},
        "uncertain_history": {"image_uncertain": {"pgp": 1}},
        "image_found": {"no_image_listed": {"literature": 1}},
        "side_verified": {"ownership_contested": {"fjp": 1}},
        "val_rows": {"validation_split": {"ktiv": 1}}}
    assert result["review_queue"] == {"ownership_contested": {"fjp": 1}}
    text = funnel.render(result)
    assert "| side verified | ownership_contested | 1 | 0 | 0 | 0 | 1 |" in text
    assert "Ledger review queue: 1 uncertain-history candidates (1 document groups)" in text


def test_trace_stops_at_the_first_stage_not_passed(tmp_path: Path) -> None:
    """The trace gives each side the last stage it reached and why it stopped."""
    work = make_work(tmp_path)
    data = funnel.load_stage_data(work, tmp_path / "pool.jsonl")
    got = {(r["row_id"], r["side_key"]): (r["stage"], r["reasons"]) for r in funnel.trace(data)}
    assert got == {(N1, "recto"): ("holdout_cleared", []), (N1, "verso"): ("holdout_cleared", []),
                   (N2, ""): ("image_found", ["ownership_contested"]),
                   (N3, "page"): ("holdout_cleared", []), (N4, "recto"): ("target_extracted", ["no_image_listed"])}
    assert [r["row_id"] for r in funnel.trace(data, limit=1)] == [N1, N1]


def test_header_lists_every_input_with_its_sha256(tmp_path: Path) -> None:
    """The header gives the generation time and the sha256 of every input file; a missing one says so."""
    work = make_work(tmp_path)
    text = funnel.render(funnel.compute_funnel(work, tmp_path / "pool.jsonl", now=NOW))
    assert "Generated 2026-11-03T12:00:00Z" in text
    for path in [tmp_path / "pool.jsonl"] + [paths.stage_path(work, key) for key in
                                             ("candidates", "images", "sides", "quality", "holdout", "rows")]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() in text
    assert "| stages/stage6_decisions.jsonl | stage6 decisions | missing | missing |" in text
    assert str(tmp_path) not in text and "fjp" not in text.lower()


def test_a_stale_stage_file_breaks_monotonicity(tmp_path: Path) -> None:
    """A quality result for a side stage2 never accepted shows up as a count above the stage before it."""
    work = make_work(tmp_path)
    quality = list(read_jsonl(paths.stage_path(work, "quality"), QualityResult)) + [_ok(N2, "")]
    write_jsonl(paths.stage_path(work, "quality"), quality)
    result = funnel.compute_funnel(work, tmp_path / "pool.jsonl", now=NOW)
    assert any("all sources: sides at quality passed (4) exceed side verified (3)" in v
               for v in result["monotone_violations"])
    assert any(v.startswith("source legacy_store:") for v in result["monotone_violations"])
    assert "NOT MONOTONE" in funnel.render(result)


def test_build_refusals_come_from_the_stage6_decisions(tmp_path: Path) -> None:
    """A side that cleared the hold-outs but is not a row takes the build's reasons, else ``not_built``."""
    work = make_work(tmp_path)
    rows = list(read_jsonl(paths.stage_path(work, "rows"), RowRecord))
    write_jsonl(paths.stage_path(work, "rows"), rows[:2])
    pool = tmp_path / "pool.jsonl"
    assert funnel.compute_funnel(work, pool, now=NOW)["reasons"]["rows"] == {"not_built": {"ktiv": 1}}
    write_jsonl(work / funnel.DECISIONS_FILE, [{"row_id": N3, "side_key": "page", "stage": "holdout_cleared",
                                                "reasons": ["side_read_missing"]}])
    assert funnel.compute_funnel(work, pool, now=NOW)["reasons"]["rows"] == {"side_read_missing": {"ktiv": 1}}


def test_duplicate_keys_are_reported(tmp_path: Path) -> None:
    """A key listed twice in a stage file is counted once and reported."""
    work = make_work(tmp_path)
    sides = list(read_jsonl(paths.stage_path(work, "sides"), SideDecision))
    write_jsonl(paths.stage_path(work, "sides"), sides + sides[:1])
    result = funnel.compute_funnel(work, tmp_path / "pool.jsonl", now=NOW)
    assert result["duplicates"]["sides"] == 1 and result["counts"]["total"]["side_verified"]["sides"] == 3
    assert "Keys listed more than once" in funnel.render(result)


def test_main_writes_funnel_md_in_the_work_dir(tmp_path: Path, capsys: Any) -> None:
    """The CLI writes FUNNEL.md into the work dir by default, with legacy-store sources under their public name."""
    work = make_work(tmp_path)
    funnel.main(["--work-dir", str(work), "--pool", str(tmp_path / "pool.jsonl")])
    text = (work / "FUNNEL.md").read_text()
    assert text.startswith("# Novel-rows funnel") and "## legacy_store" in text
    assert "| newly usable training rows | rows | 1 (1) | 0 (0) | 1 (1) | 2 (1) |" in text
    assert "rows 3 (2 groups)" in capsys.readouterr().out
    out = tmp_path / "elsewhere" / "F.md"
    funnel.main(["--work-dir", str(work), "--pool", str(tmp_path / "pool.jsonl"), "--out", str(out)])
    assert out.read_text().startswith("# Novel-rows funnel")
