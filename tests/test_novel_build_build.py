"""Tests of the novel-rows build (stage 6): row assembly, the acceptance checks, the dataset writer and the CLI.

Synthetic stage files in the schema types (invented shelf marks, a few letters of text, tiny JPEGs) in ``tmp_path``.
``save_dataset`` and ``export_images_once`` are stubbed, except in one round trip through the real writer. No
network, no LM Studio, no NAS.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest
from PIL import Image

from src.finetuning.qwen_hebrew.build_ktiv_dataset import FEATURES
from src.finetuning.qwen_hebrew.novel_build import build, funnel, paths
from src.finetuning.qwen_hebrew.novel_build.schema import (Candidate, HoldoutResult, ImageRef, QualityResult,
                                                           ReadRecord, RowRecord, SideDecision, TargetSide,
                                                           read_jsonl, write_jsonl)
from src.finetuning.qwen_hebrew.prompts import SCRIPT_NEUTRAL_PROMPT

HEBREW = "אבג דהו\nזחט יכל\nמנס עפצ"
READER = "kraken:test_model"
A, B, C, D, E, F = ("pgp:9001", "Cambridge_CUL_T_S_97_4322:0", "990000000000000001:FL1", "lit:test_work:p1:k1",
                    "pgp:9002", "pgp:9003")
GROUP_A, GROUP_B = "canonical_id:Cambridge_CUL_T_S_97_4321", "canonical_id:Cambridge_CUL_T_S_97_4322"


@dataclass
class World:
    """A synthetic work dir.

    :param work: Work dir.
    :param pool: Ledger file.
    :param out: Dataset directory to write.
    """

    work: Path
    pool: Path
    out: Path


def _jpeg(path: Path, size: Tuple[int, int] = (64, 48)) -> str:
    """Write a small JPEG.

    :param path: Destination.
    :param size: (width, height).
    :return: The path as a string.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (200, 190, 180)).save(path, "JPEG", quality=90)
    return str(path)


def _image(work: Path, name: str, origin: str = "iiif_cudl") -> ImageRef:
    """A found photograph with a real file.

    :param work: Work dir.
    :param name: File stem.
    :param origin: Image origin.
    :return: The image reference.
    """
    path = _jpeg(work / "images" / f"{name}.jpg")
    sha = hashlib.sha256(Path(path).read_bytes() + name.encode()).hexdigest()
    return ImageRef(image_id=f"{origin}:0", origin=origin, url=f"https://example.invalid/{name}.jpg", path=path,
                    sha256=sha, width=64, height=48, status="present")


def _target(side: str, text: str = HEBREW) -> TargetSide:
    """A target side.

    :param side: Side key.
    :param text: Target text.
    :return: The side.
    """
    return TargetSide(side_key=side, text=text, raw_text=text, lines=text.splitlines(),
                      pointer={"source": "test_edition", "side": side, "lines": [1, 3]}, letters_hebrew=18)


def _cand(row_id: str, source: str, group: str, sides: Sequence[str], images: Sequence[ImageRef] = (),
          script: str = "hebrew", shelfmark: str = "T-S 97.4321", rnl: bool = False) -> Candidate:
    """A stage0/stage1 candidate.

    :param row_id: Candidate id.
    :param source: Ledger source.
    :param group: Document group.
    :param sides: Target side keys.
    :param images: Photographs.
    :param script: Ledger script.
    :param shelfmark: Shelf mark.
    :param rnl: RNL holder flag.
    :return: The candidate.
    """
    return Candidate(row_id=row_id, source=source, document_group=group, canonical_ids=[group.split(":", 1)[1]],
                     pgpids=[], ktiv_id=None, shelfmark=shelfmark, script=script, languages_stated=["Hebrew"],
                     verdict="novel", witnesses=[], unit="side", n_letters=40, targets=[_target(s) for s in sides],
                     images=list(images), rnl=rnl)


def _ledger(row_id: str, source: str, group: str, verdict: str = "novel", script: str = "hebrew",
            witnesses: Sequence[str] = (), held_out: Optional[str] = None) -> Dict[str, Any]:
    """A ledger record.

    :param row_id: Candidate id.
    :param source: Source.
    :param group: Document group.
    :param verdict: Verdict.
    :param script: Script.
    :param witnesses: Witness datasets.
    :param held_out: Held-out reason.
    :return: The record.
    """
    return {"row_id": row_id, "source": source, "script": script, "document_group": group, "languages": [],
            "flags": {"verdict": verdict, "witnesses": list(witnesses), "held_out": held_out, "image": [],
                      "document": [], "image_uncertain": [], "document_uncertain": []}}


def _accepted(row_id: str, side: str, image: ImageRef, reader: str = READER) -> SideDecision:
    """An accepted side decision.

    :param row_id: Candidate id.
    :param side: Side key.
    :param image: The photograph.
    :param reader: Reader name.
    :return: The decision.
    """
    return SideDecision(row_id=row_id, side_key=side, outcome="accepted", reasons=[], image_sha256=image.sha256,
                        basis="p99", reader=reader, recall=0.61, null_p95=0.1, null_p99=0.2, margin=0.3,
                        competitors=[{"sha256": image.sha256, "recall": 0.61, "coverage": {"read_letters_ratio": 1.1}}])


def _quality(row_id: str, side: str, ok: bool = True, reasons: Sequence[str] = (), text: str = HEBREW,
             tokens: Optional[int] = 900, script: str = "hebrew") -> QualityResult:
    """A quality result.

    :param row_id: Candidate id.
    :param side: Side key.
    :param ok: Passed.
    :param reasons: Reason codes.
    :param text: Final text.
    :param tokens: Measured tokens.
    :param script: Measured script.
    :return: The result.
    """
    return QualityResult(row_id=row_id, side_key=side, ok=ok, reasons=list(reasons), script=script,
                         letters_hebrew=18, letters_arabic=0, tokens=tokens, text=text)


def _read(image: ImageRef, reader: str = READER) -> ReadRecord:
    """A read of a photograph.

    :param image: The photograph.
    :param reader: Reader name.
    :return: The read.
    """
    return ReadRecord(sha256=image.sha256, reader=reader, text=HEBREW, lines=HEBREW.splitlines(), n_lines=3,
                      letters_hebrew=18, letters_arabic=0, finish="ok", seconds=1.0, read_at="2026-11-03T00:00:00Z")


def make_world(tmp_path: Path) -> World:
    """Six novel candidates (three rows: two training sides of one document, one validation side with a
    legacy-store photograph; one side in review, one without a photograph, one failing quality, one held out) and
    three non-novel ledger records.

    :param tmp_path: Pytest temp dir.
    :return: The world.
    """
    work = tmp_path / "work"
    img = {name: _image(work, name, origin) for name, origin in
           (("a_r", "iiif_cudl"), ("a_v", "iiif_cudl"), ("b", "fjp_local"), ("c", "ktiv_zip"), ("e", "iiif_pgp"),
            ("f", "iiif_pgp"))}
    cands = [_cand(A, "pgp", GROUP_A, ["recto", "verso"], [img["a_r"], img["a_v"]]),
             _cand(B, "fjp", GROUP_B, [""], [img["b"]], shelfmark="T-S 97.4322"),
             _cand(C, "ktiv", "ktiv:990000000000000001", ["page"], [img["c"]]),
             _cand(D, "literature", "canonical_id:Cambridge_CUL_T_S_Ar_97_1", ["recto"], [], script="arabic"),
             _cand(E, "pgp", "canonical_id:Cambridge_CUL_T_S_97_4323", ["recto"], [img["e"]]),
             _cand(F, "pgp", "canonical_id:Cambridge_CUL_T_S_95_1", ["recto"], [img["f"]])]
    ledger = [_ledger(c.row_id, c.source, c.document_group, script=c.script) for c in cands]
    ledger += [_ledger("pgp:9100", "pgp", "canonical_id:Cambridge_CUL_T_S_90_1", "held_out",
                       held_out="registered_benchmark_ms"),
               _ledger("pgp:9101", "pgp", "canonical_id:Cambridge_CUL_T_S_90_2", "previously_trained",
                       witnesses=["genizah_test_a"]),
               _ledger("990000000000000002:FL2", "ktiv", "ktiv:990000000000000002", "uncertain_history")]
    pool = tmp_path / "pool.jsonl"
    write_jsonl(pool, ledger)
    write_jsonl(paths.stage_path(work, "candidates"), [replace(c, images=[]) for c in cands])
    write_jsonl(paths.stage_path(work, "images"), cands)
    write_jsonl(paths.stage_path(work, "sides"), [
        _accepted(A, "recto", img["a_r"]), _accepted(A, "verso", img["a_v"]), _accepted(B, "", img["b"]),
        SideDecision(row_id=C, side_key="page", outcome="needs_review", reasons=["read_recall_below_p95"]),
        _accepted(E, "recto", img["e"]), _accepted(F, "recto", img["f"])])
    write_jsonl(paths.stage_path(work, "quality"), [
        _quality(A, "recto"), _quality(A, "verso", tokens=1200), _quality(B, "", reasons=["tokens_unmeasured"],
                                                                          tokens=None),
        _quality(E, "recto", ok=False, reasons=["hygiene_chars"], text=HEBREW + " &"), _quality(F, "recto")])
    write_jsonl(paths.stage_path(work, "holdout"), [
        HoldoutResult(row_id=A, side_key="recto", held_out=False, reasons=[], split="train", rnl=False),
        HoldoutResult(row_id=A, side_key="verso", held_out=False, reasons=[], split="train", rnl=False),
        HoldoutResult(row_id=B, side_key="", held_out=False, reasons=[], split="val", rnl=False),
        HoldoutResult(row_id=F, side_key="recto", held_out=True, reasons=["benchmark:pgp131:Cambridge_CUL_T_S_95_1"],
                      split="excluded", rnl=False)])
    write_jsonl(work / build.READS_FILE, [_read(img[name]) for name in ("a_r", "a_v", "b", "c", "e", "f")])
    return World(work=work, pool=pool, out=tmp_path / "datasets" / "genizah_novel_v1")


def _assembled(world: World) -> Tuple[List[Dict[str, Any]], List[RowRecord], List[Dict[str, Any]], Dict[str, Any]]:
    """Rows, manifest, decisions and stage counts of a world.

    :param world: The world.
    :return: The four.
    """
    data = funnel.load_stage_data(world.work, world.pool, build_outputs=False)
    rows, manifest, decisions = build.assemble_rows(world.work, pool=world.pool, data=data)
    counts = funnel.stage_counts(replace(data, rows=list(manifest)))
    return rows, manifest, decisions, counts


def _snapshot(root: Path) -> Dict[str, bytes]:
    """Every file under a directory with its bytes.

    :param root: Directory.
    :return: ``relative path -> bytes``.
    """
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


class Calls:
    """Stub of ``save_dataset`` / ``export_images_once`` that records its calls."""

    def __init__(self) -> None:
        """Start with no calls."""
        self.saved: List[Tuple[Dict[str, List[Dict[str, Any]]], Path, Dict[str, str]]] = []
        self.exported: List[Tuple[Path, Path]] = []

    def save(self, splits: Dict[str, List[Dict[str, Any]]], out_dir: Path, sidecars: Dict[str, str]) -> None:
        """Record a save and write the sidecars, as the real writer leaves them.

        :param splits: Split rows.
        :param out_dir: Destination.
        :param sidecars: Sidecar texts.
        """
        self.saved.append((splits, out_dir, sidecars))
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, text in sidecars.items():
            (out_dir / name).write_text(text, encoding="utf-8")

    def export(self, dataset_dir: Path, out_dir: Path) -> Dict[str, Any]:
        """Record an export.

        :param dataset_dir: Saved dataset.
        :param out_dir: Export directory.
        :return: A manifest-like dict.
        """
        self.exported.append((dataset_dir, out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        return {"source": str(dataset_dir), "n_rows": 3, "n_images": 3, "dedupe_ratio": 1.0, "splits": {}}


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> Calls:
    """Stub the dataset writer and the export.

    :param monkeypatch: Pytest monkeypatch.
    :return: The call recorder.
    """
    recorder = Calls()
    monkeypatch.setattr(build, "save_dataset", recorder.save)
    monkeypatch.setattr(build, "export_images_once", recorder.export)
    return recorder


# ------------------------------------------------------------------------------------------------------------ assembly

def test_rows_are_page_rows_in_the_ktiv_schema(tmp_path: Path) -> None:
    """Three rows with exactly the KTIV fields, the neutral prompt, the page task and section, the local image."""
    world = make_world(tmp_path)
    rows, manifest, _, _ = _assembled(world)
    assert [row["stem"] for row in rows] == ["novel_pgp_pgp_9001_recto_page", "novel_pgp_pgp_9001_verso_page",
                                             "novel_fjp_Cambridge_CUL_T_S_97_4322_0_page_page"]
    for row in rows:
        assert set(row) == set(FEATURES)
        assert row["question"] == SCRIPT_NEUTRAL_PROMPT
        assert row["task"] == build.TASK == "fragment_transcribe"
        assert row["section"] == "novel_page"
        assert Path(row["image"]).is_file()
        assert row["answer"] == HEBREW and row["target_chars"] == len(HEBREW)
        assert (row["image_width"], row["image_height"]) == (64, 48)
    assert [row["label_source"] for row in rows] == ["pgp_edition", "pgp_edition", "legacy_store_diplomatic"]
    assert [row["target_tokens"] for row in rows] == [900, 1200, 0]
    assert [m.split for m in manifest] == ["train", "train", "val"]
    assert [m.image_origin for m in manifest] == ["iiif_cudl", "iiif_cudl", "fjp_local"]
    first = manifest[0]
    assert first.verdict == "novel" and first.document_group == GROUP_A and first.script == "hebrew"
    assert first.side_basis["basis"] == "p99" and first.side_basis["read"]["reader"] == READER
    assert first.side_basis["coverage"] == {"read_letters_ratio": 1.1}
    assert first.target_pointer == {"source": "test_edition", "side": "recto", "lines": [1, 3]}
    assert manifest[2].quality_flags == ["tokens_unmeasured"]


def test_label_source_covers_every_ledger_source() -> None:
    """Every ledger source maps to its label source."""
    assert build.LABEL_SOURCES == {"pgp": "pgp_edition", "fjp": "legacy_store_diplomatic",
                                   "fjp_scrape": "legacy_store_scrape", "ktiv": "ktiv_nli",
                                   "literature": "scanned_edition", "lit_run2": "run2_edition",
                                   "apd_v3": "apd_edition", "aodeh_v2_1": "book_edition"}


def test_decisions_give_every_side_its_last_stage_and_reasons(tmp_path: Path) -> None:
    """One decision per (candidate, side); the excluded ones stop where the stage files say, with reasons."""
    _, _, decisions, _ = _assembled(make_world(tmp_path))
    by_key = {(d["row_id"], d["side_key"]): d for d in decisions}
    assert len(decisions) == 7
    assert {k: (d["stage"], d["reasons"]) for k, d in by_key.items() if d["stage"] != "row"} == {
        (C, "page"): ("image_found", ["read_recall_below_p95"]),
        (D, "recto"): ("target_extracted", ["no_image_listed"]),
        (E, "recto"): ("side_verified", ["hygiene_chars"]),
        (F, "recto"): ("quality_passed", ["benchmark:pgp131:Cambridge_CUL_T_S_95_1"])}
    assert by_key[(B, "")]["stem"] == "novel_fjp_Cambridge_CUL_T_S_97_4322_0_page_page"
    assert by_key[(B, "")]["label_source"] == "legacy_store_diplomatic"


def test_build_gates_refuse_a_missing_read_a_missing_file_and_a_non_novel_verdict(tmp_path: Path) -> None:
    """A side without the read that placed it, a photograph not on disk and a non-novel ledger verdict stop a row."""
    world = make_world(tmp_path)
    reads = [r for r in read_jsonl(world.work / build.READS_FILE)]
    write_jsonl(world.work / build.READS_FILE, reads[:1] + reads[2:])         # drop the read of A verso
    ledger = list(read_jsonl(world.pool))
    for rec in ledger:
        if rec["row_id"] == A:
            rec["flags"]["verdict"] = "uncertain_history"
    write_jsonl(world.pool, ledger)
    missing = {str(world.work / "images" / "b.jpg")}
    rows, _, decisions = build.assemble_rows(world.work, pool=world.pool, is_file=lambda p: str(p) not in missing)
    by_key = {(d["row_id"], d["side_key"]): d for d in decisions}
    assert rows == []
    assert by_key[(A, "recto")]["reasons"] == ["ledger_verdict_uncertain_history"]
    assert by_key[(A, "verso")]["reasons"] == ["ledger_verdict_uncertain_history", "side_read_missing"]
    assert by_key[(B, "")]["reasons"] == ["image_file_missing"]
    assert {d["stage"] for d in by_key.values() if d["row_id"] in (A, B)} == {"holdout_cleared"}


def test_find_read_accepts_a_reader_kind_naming_one_read() -> None:
    """A decision that names only the reader kind finds the single read of that kind; two make it ambiguous."""
    index = {"s1": {"kraken:m1": {"reader": "kraken:m1"}}, "s2": {"kraken:m1": {}, "kraken:m2": {}}}
    assert build.find_read(index, "s1", "kraken") == {"reader": "kraken:m1"}
    assert build.find_read(index, "s1", "kraken:m1") == {"reader": "kraken:m1"}
    assert build.find_read(index, "s2", "kraken") is None
    assert build.find_read(index, "s1", None) is None


def test_duplicate_stems_raise(tmp_path: Path) -> None:
    """Two target sides with one key would give one stem twice: the build refuses."""
    world = make_world(tmp_path)
    cands = list(read_jsonl(paths.stage_path(world.work, "candidates"), Candidate))
    cands[0].targets.append(_target("recto"))
    write_jsonl(paths.stage_path(world.work, "candidates"), cands)
    with pytest.raises(ValueError, match="stem"):
        build.assemble_rows(world.work, pool=world.pool)


def test_stems_are_ascii_and_distinct_for_non_ascii_keys() -> None:
    """Non-ASCII keys keep distinct stems through a digest."""
    first = build.row_stem("literature", "lit:w:p1:כג", 'ע"א')
    second = build.row_stem("literature", "lit:w:p1:כג", 'ע"ב')
    assert first != second and first.isascii() and first.startswith("novel_literature_lit_w_p1_")
    assert build.row_stem("pgp", "pgp:444", "") == "novel_pgp_pgp_444_page_page"


def test_image_size_applies_the_exif_orientation(tmp_path: Path) -> None:
    """A photograph stored sideways with an orientation tag reports its upright size."""
    path = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (64, 48))
    exif = image.getexif()
    exif[build.EXIF_ORIENTATION] = 6
    image.save(path, "JPEG", exif=exif.tobytes())
    assert build.image_size(path) == (48, 64)


# ---------------------------------------------------------------------------------------------------------- acceptance

def test_clean_build_has_no_violations(tmp_path: Path) -> None:
    """The synthetic world passes every acceptance check."""
    assert build.acceptance_checks(*_assembled(make_world(tmp_path))) == []


def _mutate(kind: str, rows: List[Dict[str, Any]], manifest: List[RowRecord], decisions: List[Dict[str, Any]],
            counts: Dict[str, Any]) -> None:
    """Break one acceptance rule in place.

    :param kind: Which rule.
    :param rows: Rows.
    :param manifest: Manifest.
    :param decisions: Decisions.
    :param counts: Stage counts.
    """
    if kind == "sha256":
        manifest[0] = replace(manifest[0], image_sha256="")
    elif kind == "side_basis":
        manifest[0] = replace(manifest[0], side_basis={"outcome": "needs_review"})
    elif kind == "holdout_result":
        manifest[0] = replace(manifest[0], split="excluded")
    elif kind == "holdout_reasons":
        manifest[0] = replace(manifest[0], holdout_reasons=["benchmark:test:x"])
    elif kind == "verdict":
        manifest[0] = replace(manifest[0], verdict="uncertain_history")
    elif kind == "question":
        rows[0]["question"] = "Transcribe this page."
    elif kind == "tokens":
        rows[0]["target_tokens"] = build.TOKEN_BUDGET + 1
    elif kind == "monotone":
        counts["total"]["rows"]["sides"] = 99
    elif kind == "silent_exclusion":
        decisions.append({"row_id": "pgp:1", "side_key": "recto", "stage": "image_found", "reasons": []})
    elif kind == "stems":
        rows.append(dict(rows[0]))
        manifest.append(manifest[0])
    elif kind == "groups":
        manifest[2] = replace(manifest[2], document_group=manifest[0].document_group)
    elif kind == "no_rows":
        rows.clear()
        manifest.clear()


@pytest.mark.parametrize("kind, message", [
    ("sha256", "no image sha256"), ("side_basis", "no side basis"), ("holdout_result", "no hold-out result"),
    ("holdout_reasons", "hold-out reasons on a row"), ("verdict", "not novel"),
    ("question", "not SCRIPT_NEUTRAL_PROMPT"), ("tokens", "over the budget"), ("monotone", "not monotone"),
    ("silent_exclusion", "excluded without a reason"), ("stems", "stem not unique"),
    ("groups", "shares a document group"), ("no_rows", "no rows assembled")])
def test_each_violation_is_detected(tmp_path: Path, kind: str, message: str) -> None:
    """Every acceptance rule, broken on its own, is reported."""
    rows, manifest, decisions, counts = _assembled(make_world(tmp_path))
    counts = copy.deepcopy(counts)
    _mutate(kind, rows, manifest, decisions, counts)
    violations = build.acceptance_checks(rows, manifest, decisions, counts)
    assert any(message in v for v in violations), violations


@pytest.mark.parametrize("char", list(build.HYGIENE_CHARS))
def test_every_hygiene_character_is_a_violation(tmp_path: Path, char: str) -> None:
    """An answer holding any hygiene character is reported."""
    rows, manifest, decisions, counts = _assembled(make_world(tmp_path))
    rows[1]["answer"] += f" {char}"
    violations = build.acceptance_checks(rows, manifest, decisions, counts)
    assert any("hygiene character" in v and rows[1]["stem"] in v for v in violations)


def test_privacy_reasons_name_legacy_store_and_rnl_rows(tmp_path: Path) -> None:
    """A legacy-store photograph and transcription force privacy; so does an RNL row."""
    _, manifest, decisions, _ = _assembled(make_world(tmp_path))
    reasons = build.privacy_reasons(manifest, decisions)
    assert any("photographs from the project's legacy image store" in r for r in reasons)
    assert any("transcriptions from the project's legacy store" in r for r in reasons)
    library_only = [m for m in manifest if m.source == "pgp"]
    assert build.privacy_reasons(library_only, decisions) == []
    rnl = [dict(d, rnl=True) if d.get("stem") == library_only[0].stem else d for d in decisions]
    assert any("Russian National Library" in r for r in build.privacy_reasons(library_only, rnl))


# ---------------------------------------------------------------------------------------------------------------- CLI

def _argv(world: World, *extra: str) -> List[str]:
    """CLI arguments for a world.

    :param world: The world.
    :param extra: More arguments.
    :return: Arguments.
    """
    return ["--work-dir", str(world.work), "--pool", str(world.pool), "--out", str(world.out), *extra]


def test_dry_run_writes_nothing(tmp_path: Path, calls: Calls, capsys: pytest.CaptureFixture) -> None:
    """``--dry-run`` prints the counts and writes no file anywhere."""
    world = make_world(tmp_path)
    before = _snapshot(tmp_path)
    build.main(_argv(world, "--dry-run"))
    assert _snapshot(tmp_path) == before
    assert calls.saved == [] and calls.exported == []
    printed = capsys.readouterr().out
    assert "rows: 3 (train_page 2, val 1" in printed and "violations: none" in printed


def test_public_refuses_a_legacy_store_image(tmp_path: Path, calls: Calls, capsys: pytest.CaptureFixture) -> None:
    """``--public`` refuses a build with a legacy-store row and writes nothing."""
    world = make_world(tmp_path)
    before = _snapshot(tmp_path)
    with pytest.raises(SystemExit) as stop:
        build.main(_argv(world, "--public"))
    assert stop.value.code == 1
    assert calls.saved == [] and not world.out.exists() and _snapshot(tmp_path) == before
    assert "public build refused" in capsys.readouterr().out


def test_violations_stop_the_build(tmp_path: Path, calls: Calls) -> None:
    """A stale stage file that breaks monotonicity stops the build before any write."""
    world = make_world(tmp_path)
    quality = list(read_jsonl(paths.stage_path(world.work, "quality"), QualityResult))
    quality += [_quality(C, "page"), _quality(D, "recto"), _quality("pgp:7777", "recto")]
    write_jsonl(paths.stage_path(world.work, "quality"), quality)
    with pytest.raises(SystemExit):
        build.main(_argv(world))
    assert calls.saved == [] and not world.out.exists()


def test_main_writes_dataset_sidecars_manifest_decisions_and_funnel(tmp_path: Path, calls: Calls) -> None:
    """A build writes the splits with README, PRIVATE and stats sidecars, the manifest, the decisions, the export,
    the stage6 files and FUNNEL.md; the README names no forbidden word, host or path."""
    world = make_world(tmp_path)
    build.main(_argv(world))
    (splits, out_dir, sidecars), = calls.saved
    assert out_dir == world.out
    assert {name: len(rows) for name, rows in splits.items()} == {"train_page": 2, "val": 1}
    assert set(sidecars) == {"README.md", "PRIVATE.md", "stats.json"}
    readme = sidecars["README.md"]
    assert readme.startswith("# genizah_novel_v1\n\n" + build.PRIVATE_BANNER)
    assert "legacy image store" in readme and "| legacy_store | hebrew | 0 | 1 |" in readme
    assert not any(word in readme.lower() for word in build.FORBIDDEN_README)
    stats = json.loads((world.out / "stats.json").read_text())
    assert stats["rows"] == {"train_page": 2, "val": 1} and stats["private"] is True
    assert stats["images_once"]["n_rows"] == 3 and "source" not in stats["images_once"]
    assert all(not i["name"].startswith("/") for i in stats["inputs"])
    assert stats["funnel"]["total"]["rows"] == {"candidates": 2, "sides": 3, "groups": 2}
    manifest = list(read_jsonl(world.out / "manifest.jsonl"))
    assert [m["rnl"] for m in manifest] == [False, False, False] and manifest[2]["shelfmark"] == "T-S 97.4322"
    assert [m.stem for m in read_jsonl(world.out / "manifest.jsonl", RowRecord)] == [m["stem"] for m in manifest]
    assert len(list(read_jsonl(world.out / "decisions.jsonl"))) == 7
    export_dir = world.out.with_name("genizah_novel_v1_images_once")
    assert calls.exported == [(world.out, export_dir)]
    assert (export_dir / "PRIVATE.md").is_file() and (export_dir / "README.md").read_text() == readme
    assert len(list(read_jsonl(paths.stage_path(world.work, "rows"), RowRecord))) == 3
    assert len(list(read_jsonl(world.work / funnel.DECISIONS_FILE))) == 7
    funnel_md = (world.work / funnel.FUNNEL_FILE).read_text()
    assert "| newly usable training rows | rows | 2 (1) |" in funnel_md


def test_no_export_skips_the_export(tmp_path: Path, calls: Calls) -> None:
    """``--no-export`` writes the dataset without the images-once copy."""
    world = make_world(tmp_path)
    build.main(_argv(world, "--no-export"))
    assert len(calls.saved) == 1 and calls.exported == []


def test_round_trip_through_the_real_writer(tmp_path: Path) -> None:
    """The rows load back from the real ``save_dataset`` with the KTIV features and embedded images, and the
    real images-once export carries the README and the PRIVATE notice."""
    from datasets import load_from_disk

    world = make_world(tmp_path)
    rows, manifest, decisions, counts = _assembled(world)
    private = build.privacy_reasons(manifest, decisions)
    stats = build.build_stats(world.out.name, manifest, decisions, counts, [], private)
    build.write_dataset(build.rows_by_split(rows, manifest), manifest, decisions, world.out, export=True,
                        private=False, stats=stats, readme=build.readme_text(stats))
    saved = load_from_disk(str(world.out))
    assert {name: saved[name].num_rows for name in saved} == {"train_page": 2, "val": 1}
    assert saved["train_page"].features == FEATURES
    assert saved["val"][0]["image"].size == (64, 48) and saved["val"][0]["stem"] == manifest[2].stem
    assert (world.out / "PRIVATE.md").is_file()
    export_dir = world.out.with_name("genizah_novel_v1_images_once")
    assert (export_dir / "PRIVATE.md").is_file() and (export_dir / "README.md").is_file()
    assert json.loads((world.out / "stats.json").read_text())["images_once"]["n_rows"] == 3


def test_write_dataset_forces_private_and_the_banner(tmp_path: Path, calls: Calls) -> None:
    """A legacy-store row makes the dataset private even when the caller says otherwise."""
    world = make_world(tmp_path)
    rows, manifest, decisions, counts = _assembled(world)
    stats = build.build_stats(world.out.name, manifest, decisions, counts, [], [])
    out = build.write_dataset(build.rows_by_split(rows, manifest), manifest, decisions, world.out, export=False,
                              private=False, stats=stats, readme=build.readme_text(stats))
    sidecars = calls.saved[0][2]
    assert out["private"] is True and "PRIVATE.md" in sidecars
    assert build.PRIVATE_BANNER in sidecars["README.md"]


def test_check_out_dir_refuses_overlap_and_foreign_names(tmp_path: Path) -> None:
    """The output may not overlap the work dir and must be a genizah_novel_v<N> directory."""
    work = tmp_path / "work"
    with pytest.raises(ValueError, match="overlaps"):
        build.check_out_dir(work / "genizah_novel_v1", work)
    with pytest.raises(ValueError, match="genizah_novel_v"):
        build.check_out_dir(tmp_path / "genizah_clean_v2", work)
    build.check_out_dir(tmp_path / "genizah_novel_v1", work)


def test_dry_run_reports_a_foreign_output_name(tmp_path: Path, calls: Calls, capsys: pytest.CaptureFixture) -> None:
    """A dry run with an output that is not a genizah_novel_v<N> directory fails and writes nothing."""
    world = make_world(tmp_path)
    with pytest.raises(SystemExit):
        build.main(["--work-dir", str(world.work), "--pool", str(world.pool), "--out",
                    str(tmp_path / "genizah_clean_v2"), "--dry-run"])
    assert "is not a genizah_novel_v<N> directory" in capsys.readouterr().out and calls.saved == []


def test_readme_refuses_forbidden_words() -> None:
    """README text naming a forbidden word or a host is refused."""
    with pytest.raises(ValueError):
        build.check_public_text("images from https://example.invalid")
    build.check_public_text("photographs from the project's legacy image store")
