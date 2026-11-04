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


def _assembled(world: World, reviews: Optional[build.Reviews] = None
               ) -> Tuple[List[Dict[str, Any]], List[RowRecord], List[Dict[str, Any]], Dict[str, Any]]:
    """Rows, manifest, decisions and stage counts of a world.

    :param world: The world.
    :param reviews: Approvals and policy (default: none, i.e. the default policy and nothing approved).
    :return: The four.
    """
    data = funnel.load_stage_data(world.work, world.pool, build_outputs=False)
    rows, manifest, decisions = build.assemble_rows(world.work, pool=world.pool, data=data, reviews=reviews)
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


@pytest.fixture(autouse=True)
def no_reviews_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test picks up a reviews file from the environment.

    :param monkeypatch: Pytest monkeypatch.
    """
    monkeypatch.delenv(build.REVIEWS_ENV, raising=False)


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


# ------------------------------------------------------------------------------------------------------- review gate

LIT_POINTER = {"source": "scanned_edition", "work": "dissertations", "part": "test_thesis_1965",
               "page_files": ["page_001_structured.json"], "shelf_key": "k1"}
LIT_KEY = "literature:dissertations/test_thesis_1965"
PENDING_LIT = f"review_pending_work:{LIT_KEY}"
RUN2_POINTER = {"source": "run2_edition", "file": "run2_test.jsonl", "doc_id": "lit:sample_1970:29,30",
                "line_numbers": [1, 2, 3]}
RUN2_KEY = "lit_run2:lit:sample_1970"
PENDING_ROW = "review_pending_row"


def _retarget(world: World, row_id: str, source: str, pointer: Dict[str, Any]) -> None:
    """Make a candidate a scanned-edition one: its ledger source, its source in stage0 and stage1, its pointers.

    :param world: The world.
    :param row_id: Candidate id.
    :param source: New ledger source (``literature`` / ``lit_run2``).
    :param pointer: The target pointer every side of the candidate gets.
    """
    ledger = list(read_jsonl(world.pool))
    for record in ledger:
        if record["row_id"] == row_id:
            record["source"] = source
    write_jsonl(world.pool, ledger)
    for stage in ("candidates", "images"):
        path = paths.stage_path(world.work, stage)
        cands = list(read_jsonl(path, Candidate))
        for cand in cands:
            if cand.row_id == row_id:
                cand.source = source
                for target in cand.targets:
                    target.pointer = dict(pointer)
        write_jsonl(path, cands)


def _recover(world: World, row_id: str, side_key: str, reason: str, basis: str) -> None:
    """Make an accepted side a rule-recovered one (the stage2 reason and basis of the ratio rules).

    :param world: The world.
    :param row_id: Candidate id.
    :param side_key: Target side.
    :param reason: ``spill_ratio_ok`` or ``conflict_ratio_ok``.
    :param basis: ``p99_ratio2`` or ``p99_ratio2_mirror``.
    """
    path = paths.stage_path(world.work, "sides")
    sides = list(read_jsonl(path, SideDecision))
    for side in sides:
        if (side.row_id, side.side_key) == (row_id, side_key):
            side.reasons, side.basis = [reason], basis
    write_jsonl(path, sides)


def _recovered_world(tmp_path: Path) -> World:
    """The base world with both sides of A rule-recovered (recto by the spill rule, verso by the mirror rule).

    :param tmp_path: Pytest temp dir.
    :return: The world.
    """
    world = make_world(tmp_path)
    _recover(world, A, "recto", "spill_ratio_ok", "p99_ratio2")
    _recover(world, A, "verso", "conflict_ratio_ok", "p99_ratio2_mirror")
    return world


def _approved_work(key: str) -> Dict[str, str]:
    """An ``approved_works`` entry.

    :param key: ``<source>:<work>``.
    :return: The entry.
    """
    return {"work": key, "reviewed_by": "reviewer_a", "date": "2026-11-05", "sheet": "sheets/eye_sheet_test.md"}


def _verdict(row_id: str, side_key: str, verdict: str) -> Dict[str, str]:
    """An ``approved_rows`` entry.

    :param row_id: Candidate id.
    :param side_key: Target side.
    :param verdict: One of ``build.REVIEW_VERDICTS``.
    :return: The entry.
    """
    return {"row_id": row_id, "side_key": side_key, "verdict": verdict, "note": "checked by eye"}


def _reviews(works: Sequence[str] = (), rows: Sequence[Dict[str, str]] = (), required: Sequence[str] = (),
             **policy: str) -> build.Reviews:
    """Reviews from approved works, row entries, required works and policy values.

    :param works: Work keys with an approval.
    :param rows: ``approved_rows`` entries.
    :param required: Work keys that are ``require`` whatever the policy says.
    :param policy: Policy values by class.
    :return: The parsed reviews.
    """
    return build.parse_reviews({"approved_works": [_approved_work(key) for key in works],
                                "approved_rows": list(rows), "required_works": list(required),
                                "review_policy": dict(policy)})


def _by_key(decisions: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Decision records by (candidate, side).

    :param decisions: Decision records.
    :return: ``(row_id, side_key) -> record``.
    """
    return {(d["row_id"], d["side_key"]): d for d in decisions}


def _write_reviews(path: Path, **content: Any) -> Path:
    """Write a reviews file.

    :param path: Destination.
    :param content: Top-level keys of the file.
    :return: The path.
    """
    path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
    return path


def test_work_key_is_the_pointers_work_and_part_or_the_doc_id_minus_its_item() -> None:
    """Literature: ``<work>/<part>`` (the folder alone would merge theses); run 2: the doc id without its item."""
    assert build.work_key("literature", LIT_POINTER) == LIT_KEY
    assert build.work_key("literature", {"work": "dissertations"}) == "literature:dissertations"
    assert build.work_key("lit_run2", RUN2_POINTER) == RUN2_KEY
    assert build.work_key("lit_run2", {"doc_id": "cohen_2013:1"}) == "lit_run2:cohen_2013"
    assert (build.work_key("lit_run2", {"doc_id": "web:blog_misc:article-3_fragment-1:2"})
            == "lit_run2:web:blog_misc:article-3_fragment-1")
    assert build.work_key("lit_run2", {"doc_id": "no_colon"}) == "lit_run2:no_colon"
    assert build.work_key("literature", {}) == "literature:unknown"
    assert build.work_key("lit_run2", {"doc_id": None}) == "lit_run2:unknown"


def test_a_scanned_edition_work_without_approval_is_excluded_under_require(tmp_path: Path) -> None:
    """Policy ``require`` (the default, with or without a reviews file): the sides stay at ``holdout_cleared`` with
    ``review_pending_work:<source>:<work>``; the other rows are untouched."""
    world = make_world(tmp_path)
    _retarget(world, A, "literature", LIT_POINTER)
    for reviews in (None, _reviews(typewritten_works="require")):
        _, manifest, decisions = build.assemble_rows(world.work, pool=world.pool, reviews=reviews)
        by_key = _by_key(decisions)
        assert [m.row_id for m in manifest] == [B]
        for side in ("recto", "verso"):
            assert (by_key[(A, side)]["stage"], by_key[(A, side)]["reasons"]) == ("holdout_cleared", [PENDING_LIT])
            assert by_key[(A, side)]["stem"] is None
        assert by_key[(B, "")]["stage"] == "row"


def test_flag_policy_admits_a_scanned_edition_work_with_its_flag(tmp_path: Path) -> None:
    """Policy ``flag``: the sides become rows and the manifest's ``quality_flags`` carry the pending-work code."""
    world = make_world(tmp_path)
    _retarget(world, A, "literature", LIT_POINTER)
    rows, manifest, decisions = build.assemble_rows(world.work, pool=world.pool,
                                                    reviews=_reviews(typewritten_works="flag"))
    assert [(m.row_id, m.side_key, m.quality_flags) for m in manifest] == [
        (A, "recto", [PENDING_LIT]), (A, "verso", [PENDING_LIT]), (B, "", ["tokens_unmeasured"])]
    assert [row["label_source"] for row in rows] == ["scanned_edition", "scanned_edition", "legacy_store_diplomatic"]
    assert _by_key(decisions)[(A, "recto")]["stage"] == "row"


@pytest.mark.parametrize("policy", ["require", "flag"])
def test_an_approved_work_is_admitted_without_a_flag(tmp_path: Path, policy: str) -> None:
    """A work listed in ``approved_works`` is admitted under either policy and carries no review flag."""
    world = make_world(tmp_path)
    _retarget(world, A, "literature", LIT_POINTER)
    _, manifest, _ = build.assemble_rows(world.work, pool=world.pool,
                                         reviews=_reviews(works=[LIT_KEY], typewritten_works=policy))
    assert [(m.row_id, m.quality_flags) for m in manifest] == [(A, []), (A, []), (B, ["tokens_unmeasured"])]


def test_an_approval_covers_one_work_of_one_source_only(tmp_path: Path) -> None:
    """Another thesis of the same folder, or the same name under the other source, does not approve a work."""
    world = make_world(tmp_path)
    _retarget(world, A, "literature", LIT_POINTER)
    decoys = [LIT_KEY.replace("test_thesis_1965", "other_thesis_1970"), "literature:dissertations",
              LIT_KEY.replace("literature:", "lit_run2:")]
    _, manifest, decisions = build.assemble_rows(world.work, pool=world.pool, reviews=_reviews(works=decoys))
    assert [m.row_id for m in manifest] == [B]
    assert _by_key(decisions)[(A, "recto")]["reasons"] == [PENDING_LIT]
    _retarget(world, A, "lit_run2", RUN2_POINTER)
    _, manifest, decisions = build.assemble_rows(world.work, pool=world.pool, reviews=_reviews(works=decoys))
    assert _by_key(decisions)[(A, "recto")]["reasons"] == [f"review_pending_work:{RUN2_KEY}"]
    _, manifest, _ = build.assemble_rows(world.work, pool=world.pool, reviews=_reviews(works=[RUN2_KEY]))
    assert [m.row_id for m in manifest] == [A, A, B] and manifest[0].source == "lit_run2"


def test_a_row_level_ok_does_not_approve_an_unapproved_work(tmp_path: Path) -> None:
    """The work gate depends on ``approved_works`` alone."""
    world = make_world(tmp_path)
    _retarget(world, A, "literature", LIT_POINTER)
    reviews = _reviews(rows=[_verdict(A, "recto", "ok"), _verdict(A, "verso", "ok")])
    _, manifest, decisions = build.assemble_rows(world.work, pool=world.pool, reviews=reviews)
    assert [m.row_id for m in manifest] == [B]
    assert _by_key(decisions)[(A, "recto")]["reasons"] == [PENDING_LIT]


def test_rule_recovered_rows_are_flagged_under_the_default_flag_policy(tmp_path: Path) -> None:
    """Spill- and mirror-rule placements are admitted with ``review_pending_row`` in ``quality_flags``; the
    manifest's side basis still names the ratio rule."""
    world = _recovered_world(tmp_path)
    for reviews in (None, _reviews(rule_recovered="flag")):
        _, manifest, _ = build.assemble_rows(world.work, pool=world.pool, reviews=reviews)
        assert [(m.side_key, m.quality_flags) for m in manifest if m.row_id == A] == [
            ("recto", [PENDING_ROW]), ("verso", [PENDING_ROW])]
        assert [m.side_basis["basis"] for m in manifest if m.row_id == A] == ["p99_ratio2", "p99_ratio2_mirror"]
        assert manifest[-1].quality_flags == ["tokens_unmeasured"]


def test_rule_recovered_rows_are_excluded_under_require(tmp_path: Path) -> None:
    """Policy ``require`` with no ``approved_rows`` entry: ``review_pending_row``, the side stays at
    ``holdout_cleared``."""
    world = _recovered_world(tmp_path)
    reviews = _reviews(rule_recovered="require")
    _, manifest, decisions = build.assemble_rows(world.work, pool=world.pool, reviews=reviews)
    by_key = _by_key(decisions)
    assert [m.row_id for m in manifest] == [B]
    for side in ("recto", "verso"):
        assert (by_key[(A, side)]["stage"], by_key[(A, side)]["reasons"]) == ("holdout_cleared", [PENDING_ROW])
        assert by_key[(A, side)]["rule_recovered"] is True
    assert by_key[(B, "")]["rule_recovered"] is False


@pytest.mark.parametrize("policy", ["require", "flag"])
def test_an_ok_approval_admits_a_rule_recovered_row_without_a_flag(tmp_path: Path, policy: str) -> None:
    """An ``ok`` entry clears the recto of A; the verso, without one, is excluded (``require``) or flagged
    (``flag``)."""
    world = _recovered_world(tmp_path)
    reviews = _reviews(rows=[_verdict(A, "recto", "ok")], rule_recovered=policy)
    _, manifest, decisions = build.assemble_rows(world.work, pool=world.pool, reviews=reviews)
    by_key = _by_key(decisions)
    assert by_key[(A, "recto")]["stage"] == "row" and manifest[0].quality_flags == []
    if policy == "require":
        assert [m.side_key for m in manifest if m.row_id == A] == ["recto"]
        assert by_key[(A, "verso")]["reasons"] == [PENDING_ROW]
    else:
        assert [(m.side_key, m.quality_flags) for m in manifest if m.row_id == A] == [
            ("recto", []), ("verso", [PENDING_ROW])]


@pytest.mark.parametrize("verdict", ["wrong_side", "wrong_image", "partial", "bad_ocr"])
@pytest.mark.parametrize("policy", ["require", "flag"])
def test_any_other_verdict_excludes_the_row_whatever_the_policy(tmp_path: Path, verdict: str, policy: str) -> None:
    """A verdict other than ``ok`` gives ``review_rejected:<verdict>`` alone (the row was reviewed, so it is not
    also pending); the recto of A, without an entry, follows the policy."""
    world = _recovered_world(tmp_path)
    reviews = _reviews(rows=[_verdict(A, "verso", verdict)], rule_recovered=policy)
    _, manifest, decisions = build.assemble_rows(world.work, pool=world.pool, reviews=reviews)
    by_key = _by_key(decisions)
    assert by_key[(A, "verso")]["stage"] == "holdout_cleared"
    assert by_key[(A, "verso")]["reasons"] == [f"review_rejected:{verdict}"]
    assert (by_key[(A, "recto")]["stage"] == "row") == (policy == "flag")
    assert (A, "verso") not in {(m.row_id, m.side_key) for m in manifest}


def test_a_verdict_rejects_a_row_of_any_class(tmp_path: Path) -> None:
    """A rejection applies to an ordinary row too (not scanned-edition, not rule-recovered), with an unlabelled side."""
    world = make_world(tmp_path)
    _, manifest, decisions = build.assemble_rows(world.work, pool=world.pool,
                                                 reviews=_reviews(rows=[_verdict(B, "", "wrong_image")]))
    assert [m.row_id for m in manifest] == [A, A]
    assert _by_key(decisions)[(B, "")]["reasons"] == ["review_rejected:wrong_image"]


def test_a_side_in_two_review_classes_lists_both_reasons(tmp_path: Path) -> None:
    """A rule-recovered side of an unapproved work under ``require`` for both is pending twice."""
    world = _recovered_world(tmp_path)
    _retarget(world, A, "literature", LIT_POINTER)
    reviews = _reviews(rule_recovered="require")
    _, _, decisions = build.assemble_rows(world.work, pool=world.pool, reviews=reviews)
    assert _by_key(decisions)[(A, "recto")]["reasons"] == [PENDING_LIT, PENDING_ROW]


def test_review_reasons_wait_for_the_build_gates(tmp_path: Path) -> None:
    """A side that fails a build gate shows only that gate, so "pending" means "would be a row"."""
    world = _recovered_world(tmp_path)
    missing = {str(world.work / "images" / "a_r.jpg")}
    _, _, decisions = build.assemble_rows(world.work, pool=world.pool, is_file=lambda p: str(p) not in missing,
                                          reviews=_reviews(rule_recovered="require"))
    by_key = _by_key(decisions)
    assert by_key[(A, "recto")]["reasons"] == ["image_file_missing"]
    assert by_key[(A, "verso")]["reasons"] == [PENDING_ROW]


def test_quality_flags_keep_the_stage3_flags_first_and_add_each_review_flag_once(tmp_path: Path) -> None:
    """Stage3 flags stay in front; the same review flag is not listed twice."""
    world = make_world(tmp_path)
    _recover(world, B, "", "spill_ratio_ok", "p99_ratio2")
    _, manifest, _ = build.assemble_rows(world.work, pool=world.pool)
    assert manifest[-1].row_id == B and manifest[-1].quality_flags == ["tokens_unmeasured", PENDING_ROW]


# -------------------------------------------------------------------------------------------------- the reviews file

def test_parse_reviews_reads_the_documented_format() -> None:
    """Approved works with reviewer, date and sheet; row verdicts keyed by (row id, side); the policy."""
    reviews = build.parse_reviews({
        "approved_works": [_approved_work(LIT_KEY), {**_approved_work(RUN2_KEY), "note": "60 lines read"}],
        "approved_rows": [_verdict(A, "recto", "ok"), {"row_id": B, "side_key": "", "verdict": "partial"}],
        "review_policy": {"typewritten_works": "flag", "rule_recovered": "require"}})
    assert reviews.policy == {"typewritten_works": "flag", "rule_recovered": "require"}
    assert reviews.approved_works[LIT_KEY] == _approved_work(LIT_KEY)
    assert reviews.approved_works[RUN2_KEY]["note"] == "60 lines read"
    assert reviews.approved_rows[(A, "recto")]["verdict"] == "ok"
    assert reviews.approved_rows[(B, "")]["verdict"] == "partial"


def test_missing_parts_of_the_reviews_file_take_the_defaults() -> None:
    """Scanned-edition works ``require``, rule-recovered placements ``flag``; no file at all is the same state."""
    default = {"typewritten_works": "require", "rule_recovered": "flag"}
    assert build.Reviews().policy == default and build.parse_reviews({}).policy == default
    assert build.parse_reviews({"review_policy": {"rule_recovered": "require"}}).policy == {
        "typewritten_works": "require", "rule_recovered": "require"}
    assert build.parse_reviews({}).approved_works == {} and build.parse_reviews({}).approved_rows == {}


@pytest.mark.parametrize("raw, message", [
    ([], "one JSON object"),
    ({"approved_row": []}, "unknown top-level key 'approved_row'"),
    ({"review_policy": {"typewritten_works": "optional"}}, "'optional' is not one of require, flag"),
    ({"review_policy": {"all_rows": "flag"}}, "unknown class 'all_rows'"),
    ({"review_policy": "flag"}, "review_policy: must be an object"),
    ({"approved_works": [LIT_KEY]}, "approved_works[0]: must be an object"),
    ({"approved_works": {"work": LIT_KEY}}, "approved_works: must be a list"),
    ({"approved_works": [{**_approved_work("pgp:123")}]}, "must read <source>:<work>"),
    ({"approved_works": [{**_approved_work("literature:")}]}, "must read <source>:<work>"),
    ({"approved_works": [{**_approved_work(LIT_KEY), "reviewed_by": " "}]}, "reviewed_by: must not be blank"),
    ({"approved_works": [{k: v for k, v in _approved_work(LIT_KEY).items() if k != "sheet"}]}, "missing field 'sheet'"),
    ({"approved_works": [{**_approved_work(LIT_KEY), "reviewer": "x"}]}, "unknown field 'reviewer'"),
    ({"approved_works": [_approved_work(LIT_KEY), _approved_work(LIT_KEY)]}, "listed twice"),
    ({"approved_works": [_approved_work(" " + LIT_KEY)]}, "leading or trailing whitespace"),
    ({"approved_rows": [{"row_id": A, "side_key": "recto", "verdict": "okay"}]}, "'okay' is not one of ok, wrong_side"),
    ({"approved_rows": [{"row_id": A, "side_key": "recto"}]}, "missing field 'verdict'"),
    ({"approved_rows": [{"row_id": A, "side_key": 1, "verdict": "ok"}]}, "side_key: must be a string"),
    ({"approved_rows": [{"row_id": "", "side_key": "", "verdict": "ok"}]}, "row_id: must not be blank"),
    ({"approved_rows": [{"row_id": A, "side_key": "recto ", "verdict": "ok"}]}, "leading or trailing whitespace"),
    ({"approved_rows": [_verdict(A, "recto", "ok"), _verdict(A, "recto", "wrong_side")]}, "listed twice"),
    ({"required_works": LIT_KEY}, "required_works: must be a list"),
    ({"required_works": [1]}, "required_works[0]: must be a string"),
    ({"required_works": ["pgp:123"]}, "required_works[0]: 'pgp:123' must read <source>:<work>"),
    ({"required_works": ["literature:"]}, "must read <source>:<work>"),
    ({"required_works": [" literature:x/y"]}, "' literature:x/y' has leading or trailing whitespace"),
    ({"required_works": ["literature:x/y", "literature:x/y"]}, "required_works[1]: 'literature:x/y' is listed twice"),
])
def test_invalid_reviews_files_are_refused(raw: Any, message: str) -> None:
    """Unknown keys, verdicts and policies, blank or missing fields and duplicate entries are errors: a typo must
    not quietly turn a rejection into nothing."""
    with pytest.raises(ValueError) as error:
        build.parse_reviews(raw)
    assert message in str(error.value)


def test_every_problem_of_a_reviews_file_is_listed_at_once() -> None:
    """One error names all problems, so the file is fixed in one pass."""
    with pytest.raises(ValueError) as error:
        build.parse_reviews({"approved_row": [], "review_policy": {"rule_recovered": "maybe"},
                             "approved_rows": [{"row_id": A, "side_key": "recto", "verdict": "fine"}]})
    text = str(error.value)
    assert "unknown top-level key" in text and "'maybe'" in text and "'fine'" in text


def test_load_reviews_reads_a_file_and_refuses_a_missing_one(tmp_path: Path) -> None:
    """No path is the default state; a configured path that does not exist is an error, never "no approvals"."""
    assert build.load_reviews(None) == build.Reviews()
    path = _write_reviews(tmp_path / "reviews.json", approved_works=[_approved_work(LIT_KEY)],
                          approved_rows=[_verdict(A, "recto", "ok")], review_policy={"typewritten_works": "flag"})
    loaded = build.load_reviews(path)
    assert list(loaded.approved_works) == [LIT_KEY] and loaded.policy["typewritten_works"] == "flag"
    with pytest.raises(ValueError, match="not found"):
        build.load_reviews(tmp_path / "missing.json")


def test_the_reviews_argument_wins_over_the_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--reviews`` first, then ``NOVEL_BUILD_REVIEWS``, else none."""
    assert build.reviews_path(None) is None
    monkeypatch.setenv(build.REVIEWS_ENV, str(tmp_path / "env.json"))
    assert build.reviews_path(None) == tmp_path / "env.json"
    assert build.reviews_path(tmp_path / "arg.json") == tmp_path / "arg.json"


# ---------------------------------------------------------------------------------------------- acceptance and the CLI

def test_acceptance_asserts_the_review_gate_on_the_admitted_rows(tmp_path: Path) -> None:
    """Rows built under ``flag`` pass a ``flag`` policy; checked against ``require`` (or with a verdict, or with their
    flag removed) they violate: the check reads the manifest, not the gate's code path."""
    world = _recovered_world(tmp_path)
    _retarget(world, A, "literature", LIT_POINTER)
    flag = _reviews(typewritten_works="flag", rule_recovered="flag")
    rows, manifest, decisions, counts = _assembled(world, flag)
    assert [m.row_id for m in manifest] == [A, A, B]
    assert build.acceptance_checks(rows, manifest, decisions, counts, flag) == []
    # the default (no reviews) is the strictest class: a scanned-edition row needs its work approved
    assert any("work is not in approved_works" in v and "2 row(s)" in v
               for v in build.acceptance_checks(rows, manifest, decisions, counts))
    require_rows = _reviews(typewritten_works="flag", rule_recovered="require")
    assert any("rule-recovered row without an ok entry" in v and "2 row(s)" in v
               for v in build.acceptance_checks(rows, manifest, decisions, counts, require_rows))
    one_ok = _reviews(works=[LIT_KEY], rows=[_verdict(A, "recto", "ok")], rule_recovered="require")
    assert any("rule-recovered row without an ok entry" in v and "1 row(s)" in v
               for v in build.acceptance_checks(rows, manifest, decisions, counts, one_ok))
    rejected = _reviews(works=[LIT_KEY], rows=[_verdict(A, "verso", "wrong_side")])
    assert any("verdict other than ok" in v and manifest[1].stem in v
               for v in build.acceptance_checks(rows, manifest, decisions, counts, rejected))
    stripped = [replace(m, quality_flags=[f for f in m.quality_flags if not f.startswith("review_")])
                for m in manifest]
    assert any("without its review_pending flag" in v and "2 row(s)" in v
               for v in build.acceptance_checks(rows, stripped, decisions, counts, flag))


def test_acceptance_is_clean_when_every_class_is_approved(tmp_path: Path) -> None:
    """Approved work and ``ok`` rows under ``require`` for both classes: no violation, no flag."""
    world = _recovered_world(tmp_path)
    _retarget(world, A, "literature", LIT_POINTER)
    reviews = _reviews(works=[LIT_KEY], rows=[_verdict(A, "recto", "ok"), _verdict(A, "verso", "ok")],
                       typewritten_works="require", rule_recovered="require")
    rows, manifest, decisions, counts = _assembled(world, reviews)
    assert [m.quality_flags for m in manifest] == [[], [], ["tokens_unmeasured"]]
    assert build.acceptance_checks(rows, manifest, decisions, counts, reviews) == []


def test_a_rejection_that_matches_no_side_is_a_violation(tmp_path: Path) -> None:
    """A verdict other than ``ok`` works only on an exact (row id, side): a mistyped key would leave the row in the
    dataset, so it is reported; an ``ok`` that matches nothing merely leaves its row pending."""
    world = make_world(tmp_path)
    data = funnel.load_stage_data(world.work, world.pool, build_outputs=False)
    reviews = _reviews(rows=[_verdict(A, "rectum", "wrong_side"), _verdict("pgp:0", "recto", "ok"),
                             _verdict(A, "verso", "bad_ocr"), _verdict(B, "", "partial")])
    assert build.unmatched_rejections(reviews, data) == [
        "review verdict names no (candidate, side) of stage0: 1, e.g. pgp:9001#rectum"]
    assert build.unmatched_rejections(build.Reviews(), data) == []


def test_dry_run_prints_the_review_gate_with_the_default_policy(tmp_path: Path, calls: Calls,
                                                                 capsys: pytest.CaptureFixture) -> None:
    """Without a reviews file the scanned-edition rows wait for approval; the flagged rule-recovered row is admitted."""
    world = make_world(tmp_path)
    _retarget(world, A, "literature", LIT_POINTER)
    _recover(world, B, "", "spill_ratio_ok", "p99_ratio2")
    build.main(_argv(world, "--dry-run"))
    printed = capsys.readouterr().out
    assert "rows: 1 (train_page 0, val 1" in printed and "violations: none" in printed
    assert ("review (typewritten works require, rule-recovered flag): pending work 2 sides (1 works), pending row 0, "
            "rejected 0, admitted with flags 1; rule-recovered admitted 1, excluded 0 (0 by review)") in printed
    assert calls.saved == []


def test_the_reviews_env_var_feeds_the_cli(tmp_path: Path, calls: Calls, capsys: pytest.CaptureFixture,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``NOVEL_BUILD_REVIEWS`` set and no flag, the approved work is admitted; the policy line shows ``require``
    for rule-recovered rows from the file, and the one rule-recovered row waiting for approval is counted."""
    world = make_world(tmp_path)
    _retarget(world, A, "literature", LIT_POINTER)
    _recover(world, B, "", "spill_ratio_ok", "p99_ratio2")
    path = _write_reviews(tmp_path / "private_reviews.json", approved_works=[_approved_work(LIT_KEY)],
                          review_policy={"rule_recovered": "require"})
    monkeypatch.setenv(build.REVIEWS_ENV, str(path))
    build.main(_argv(world, "--dry-run"))
    printed = capsys.readouterr().out
    assert "rows: 2 (train_page 2, val 0" in printed
    assert ("review (typewritten works require, rule-recovered require): pending work 0 sides (0 works), "
            "pending row 1, rejected 0, admitted with flags 0; rule-recovered admitted 0, excluded 1 (1 by review)"
            ) in printed


def test_main_writes_the_review_flags_policy_digest_and_funnel_table(tmp_path: Path, calls: Calls) -> None:
    """An approved work and a flagged rule-recovered row: the manifest shows the flag, ``stats.json`` the policy and
    the digest of the (privately named) reviews file, the README the policy, FUNNEL.md the review table."""
    world = make_world(tmp_path)
    _retarget(world, A, "literature", LIT_POINTER)
    _recover(world, B, "", "spill_ratio_ok", "p99_ratio2")
    path = _write_reviews(tmp_path / "private_reviews.json", approved_works=[_approved_work(LIT_KEY)])
    build.main(_argv(world, "--reviews", str(path)))
    stats = json.loads((world.out / "stats.json").read_text())
    assert stats["review"] == {"policy": {"typewritten_works": "require", "rule_recovered": "flag"},
                               "required_works": 0, "approved_works": 1, "row_verdicts": {},
                               "rows_admitted_with_flags": 1}
    digest, = [i for i in stats["inputs"] if i["name"] == build.REVIEWS_INPUT_NAME]
    assert digest["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest() and digest["records"] == 1
    assert "private_reviews" not in (world.out / "stats.json").read_text() and str(tmp_path) not in json.dumps(stats)
    manifest = list(read_jsonl(world.out / "manifest.jsonl"))
    assert [m["quality_flags"] for m in manifest] == [[], [], ["tokens_unmeasured", PENDING_ROW]]
    readme = (world.out / "README.md").read_text()
    assert ("human review gate: scanned-edition works `require`, rule-recovered placements `flag`" in readme
            and "1 row(s) carry such a flag" in readme)
    decisions = _by_key(list(read_jsonl(world.out / "decisions.jsonl")))
    assert decisions[(B, "")]["rule_recovered"] is True and decisions[(A, "recto")]["rule_recovered"] is False
    text = (world.work / funnel.FUNNEL_FILE).read_text()
    assert "## Human review gate" in text and "(no work awaits review)" in text
    assert "| legacy_store | 0 | 0 | 0 | 1 | 1 | 0 | 0 |" in text and "| all | 0 | 0 | 0 | 1 | 1 | 0 | 0 |" in text


def test_main_with_a_pending_work_lists_it_in_the_funnel(tmp_path: Path, calls: Calls) -> None:
    """Without approval the work is excluded and named in FUNNEL.md exactly as ``approved_works`` must spell it."""
    world = make_world(tmp_path)
    _retarget(world, A, "literature", LIT_POINTER)
    build.main(_argv(world))
    text = (world.work / funnel.FUNNEL_FILE).read_text()
    assert f"| {LIT_KEY} | 2 |" in text and "| literature | 2 | 0 | 0 | 0 | 0 | 0 | 0 |" in text
    stats = json.loads((world.out / "stats.json").read_text())
    assert stats["exclusion_reasons"]["review_pending_work:literature"] == 2
    assert stats["rows"] == {"val": 1}
    assert LIT_KEY not in (world.out / "stats.json").read_text()


def test_an_unmatched_rejection_stops_the_build(tmp_path: Path, calls: Calls, capsys: pytest.CaptureFixture) -> None:
    """The violation is printed and nothing is written."""
    world = make_world(tmp_path)
    path = _write_reviews(tmp_path / "reviews.json", approved_rows=[_verdict(A, "rectum", "wrong_side")])
    before = _snapshot(tmp_path)
    with pytest.raises(SystemExit) as stop:
        build.main(_argv(world, "--reviews", str(path)))
    assert stop.value.code == 1 and calls.saved == [] and not world.out.exists() and _snapshot(tmp_path) == before
    assert "review verdict names no (candidate, side) of stage0: 1, e.g. pgp:9001#rectum" in capsys.readouterr().out


def test_a_missing_or_invalid_reviews_file_stops_the_cli(tmp_path: Path, calls: Calls) -> None:
    """A configured reviews file that cannot be used is an error before anything is assembled or written."""
    world = make_world(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        build.main(_argv(world, "--reviews", str(tmp_path / "missing.json")))
    bad = _write_reviews(tmp_path / "bad.json", approved_rows=[_verdict(A, "recto", "fine")])
    with pytest.raises(ValueError, match="'fine' is not one of"):
        build.main(_argv(world, "--reviews", str(bad)))
    assert calls.saved == [] and not world.out.exists()


# ---------------------------------------------------------------------------------------------------- required works

OTHER_POINTER = {"source": "scanned_edition", "work": "dissertations", "part": "printed_edition_1990",
                 "page_files": ["page_002_structured.json"], "shelf_key": "k2"}
OTHER_KEY = "literature:dissertations/printed_edition_1990"
PENDING_OTHER = f"review_pending_work:{OTHER_KEY}"


def _two_works_world(tmp_path: Path) -> World:
    """The base world with A (two sides) in the thesis LIT_KEY and B (one unlabelled side) in the printed edition
    OTHER_KEY, both scanned-edition works.

    :param tmp_path: Pytest temp dir.
    :return: The world.
    """
    world = make_world(tmp_path)
    _retarget(world, A, "literature", LIT_POINTER)
    _retarget(world, B, "literature", OTHER_POINTER)
    return world


def test_required_works_are_parsed_and_default_to_none() -> None:
    """The optional list holds work keys; a required work is ``require`` whatever ``typewritten_works`` says."""
    reviews = build.parse_reviews({"required_works": [LIT_KEY, RUN2_KEY],
                                   "review_policy": {"typewritten_works": "flag"}})
    assert reviews.required_works == frozenset({LIT_KEY, RUN2_KEY})
    assert reviews.work_policy(LIT_KEY) == "require" and reviews.work_policy(RUN2_KEY) == "require"
    assert reviews.work_policy(OTHER_KEY) == "flag"
    assert build.parse_reviews({}).required_works == frozenset() == build.Reviews().required_works
    assert build.Reviews().work_policy(OTHER_KEY) == "require"                  # the default policy applies to the rest
    assert _reviews(typewritten_works="flag").work_policy(LIT_KEY) == "flag"


@pytest.mark.parametrize("policy", ["require", "flag"])
def test_a_required_work_is_excluded_and_the_policy_governs_the_other_works(tmp_path: Path, policy: str) -> None:
    """A required thesis waits for its approval under either policy; a printed edition follows ``typewritten_works``:
    excluded under ``require``, admitted with its flag under ``flag``."""
    world = _two_works_world(tmp_path)
    _, manifest, decisions = build.assemble_rows(world.work, pool=world.pool,
                                                 reviews=_reviews(required=[LIT_KEY], typewritten_works=policy))
    by_key = _by_key(decisions)
    for side in ("recto", "verso"):
        assert (by_key[(A, side)]["stage"], by_key[(A, side)]["reasons"]) == ("holdout_cleared", [PENDING_LIT])
    if policy == "require":
        assert manifest == [] and by_key[(B, "")]["reasons"] == [PENDING_OTHER]
    else:
        assert [(m.row_id, m.quality_flags) for m in manifest] == [(B, ["tokens_unmeasured", PENDING_OTHER])]


def test_an_approved_required_work_is_admitted_without_a_flag(tmp_path: Path) -> None:
    """The eye sheet of a required thesis lifts its rows; the other work stays flagged under ``flag``."""
    world = _two_works_world(tmp_path)
    reviews = _reviews(works=[LIT_KEY], required=[LIT_KEY], typewritten_works="flag")
    _, manifest, _ = build.assemble_rows(world.work, pool=world.pool, reviews=reviews)
    assert [(m.row_id, m.side_key, m.quality_flags) for m in manifest] == [
        (A, "recto", []), (A, "verso", []), (B, "", ["tokens_unmeasured", PENDING_OTHER])]


def test_a_required_work_does_not_touch_the_rule_recovered_policy(tmp_path: Path) -> None:
    """``rule_recovered`` keeps its own policy: an approved, required thesis side that was rule-recovered is flagged
    under ``flag`` and excluded under ``require``."""
    world = _two_works_world(tmp_path)
    _recover(world, A, "recto", "spill_ratio_ok", "p99_ratio2")
    for policy, admitted in (("flag", True), ("require", False)):
        reviews = _reviews(works=[LIT_KEY], required=[LIT_KEY], typewritten_works="flag", rule_recovered=policy)
        _, manifest, decisions = build.assemble_rows(world.work, pool=world.pool, reviews=reviews)
        assert ((A, "recto") in {(m.row_id, m.side_key) for m in manifest}) == admitted
        if admitted:
            assert manifest[0].quality_flags == [PENDING_ROW]
        else:
            assert _by_key(decisions)[(A, "recto")]["reasons"] == [PENDING_ROW]


def test_acceptance_holds_a_required_work_to_require_while_a_flagged_work_passes(tmp_path: Path) -> None:
    """Rows built with nothing required pass; checked with the thesis required, only its rows violate (the flagged
    printed edition still carries its flag)."""
    world = _two_works_world(tmp_path)
    flag = _reviews(typewritten_works="flag")
    rows, manifest, decisions, counts = _assembled(world, flag)
    assert [m.row_id for m in manifest] == [A, A, B]
    assert build.acceptance_checks(rows, manifest, decisions, counts, flag) == []
    required = _reviews(required=[LIT_KEY], typewritten_works="flag")
    violations = build.acceptance_checks(rows, manifest, decisions, counts, required)
    assert len(violations) == 1 and "work is not in approved_works" in violations[0] and "2 row(s)" in violations[0]
    approved = _reviews(works=[LIT_KEY], required=[LIT_KEY], typewritten_works="flag")
    assert build.acceptance_checks(rows, manifest, decisions, counts, approved) == []


def test_a_required_work_that_matches_no_work_is_a_violation(tmp_path: Path) -> None:
    """A mistyped required key would leave the intended work under the looser policy, so it is reported."""
    world = _two_works_world(tmp_path)
    data = funnel.load_stage_data(world.work, world.pool, build_outputs=False)
    reviews = _reviews(required=[LIT_KEY, LIT_KEY.replace("1965", "1956"), RUN2_KEY])
    assert build.unmatched_required_works(reviews, data) == [
        f"required_works names no work of stage0: 2, e.g. {RUN2_KEY}, {LIT_KEY.replace('1965', '1956')}"]
    assert build.unmatched_required_works(_reviews(required=[LIT_KEY, OTHER_KEY]), data) == []
    assert build.unmatched_required_works(build.Reviews(), data) == []


def test_a_mistyped_required_work_stops_the_build(tmp_path: Path, calls: Calls, capsys: pytest.CaptureFixture) -> None:
    """The violation is printed and nothing is written."""
    world = _two_works_world(tmp_path)
    path = _write_reviews(tmp_path / "reviews.json", required_works=[LIT_KEY.replace("1965", "1956")],
                          review_policy={"typewritten_works": "flag"})
    before = _snapshot(tmp_path)
    with pytest.raises(SystemExit) as stop:
        build.main(_argv(world, "--reviews", str(path)))
    assert stop.value.code == 1 and calls.saved == [] and not world.out.exists() and _snapshot(tmp_path) == before
    assert ("required_works names no work of stage0: 1, e.g. literature:dissertations/test_thesis_1956"
            in capsys.readouterr().out)


def test_dry_run_prints_the_number_of_required_works(tmp_path: Path, calls: Calls,
                                                     capsys: pytest.CaptureFixture) -> None:
    """The review line names the policy of the other works and the count of required ones."""
    world = _two_works_world(tmp_path)
    path = _write_reviews(tmp_path / "reviews.json", required_works=[LIT_KEY],
                          review_policy={"typewritten_works": "flag", "rule_recovered": "flag"})
    build.main(_argv(world, "--dry-run", "--reviews", str(path)))
    assert ("review (typewritten works flag, 1 required works, rule-recovered flag): pending work 2 sides (1 works), "
            "pending row 0, rejected 0, admitted with flags 1; rule-recovered admitted 0, excluded 0 (0 by review)"
            ) in capsys.readouterr().out


def test_main_counts_the_required_works_in_the_stats_digest_and_readme(tmp_path: Path, calls: Calls) -> None:
    """``stats.json`` and the reviews digest count the required works; the README says how many are always
    ``require``."""
    world = _two_works_world(tmp_path)
    path = _write_reviews(tmp_path / "reviews.json", required_works=[LIT_KEY], approved_works=[_approved_work(LIT_KEY)],
                          review_policy={"typewritten_works": "flag"})
    build.main(_argv(world, "--reviews", str(path)))
    stats = json.loads((world.out / "stats.json").read_text())
    assert stats["review"] == {"policy": {"typewritten_works": "flag", "rule_recovered": "flag"}, "required_works": 1,
                               "approved_works": 1, "row_verdicts": {}, "rows_admitted_with_flags": 1}
    digest, = [i for i in stats["inputs"] if i["name"] == build.REVIEWS_INPUT_NAME]
    assert digest["records"] == 2
    readme = (world.out / "README.md").read_text()
    assert "scanned-edition works `flag`" in readme and "1 named work(s) are always `require`" in readme
    assert LIT_KEY not in readme and LIT_KEY not in (world.out / "stats.json").read_text()


# ------------------------------------------------------------------------------------------------------- --list-works

def test_list_works_counts_the_sides_of_each_scanned_edition_work(tmp_path: Path) -> None:
    """Only the scanned-edition sources have work keys; most sides first, ties by key."""
    world = _two_works_world(tmp_path)
    _retarget(world, E, "lit_run2", RUN2_POINTER)
    cands = list(read_jsonl(paths.stage_path(world.work, "candidates"), Candidate))
    assert build.list_works(cands) == {LIT_KEY: 2, RUN2_KEY: 1, OTHER_KEY: 1, "literature:unknown": 1}
    assert list(build.list_works(cands)) == [LIT_KEY, RUN2_KEY, OTHER_KEY, "literature:unknown"]
    assert build.list_works([c for c in cands if c.source == "pgp"]) == {}
    assert build.list_works([]) == {}


def test_works_lines_give_totals_per_source_then_one_line_per_work() -> None:
    """The header counts works and sides per source; each work line starts with its side count."""
    lines = build.works_lines({LIT_KEY: 12, OTHER_KEY: 3, RUN2_KEY: 2})
    assert lines[0] == ("scanned-edition works in stage0: 3 works, 17 sides "
                        "(literature 2 works / 15 sides, lit_run2 1 works / 2 sides)")
    assert lines[1].startswith("sides  work key")
    assert lines[2:] == [f"   12  {LIT_KEY}", f"    3  {OTHER_KEY}", f"    2  {RUN2_KEY}"]
    assert build.works_lines({})[0].startswith("scanned-edition works in stage0: 0 works, 0 sides")


def test_list_works_cli_prints_the_keys_and_writes_nothing(tmp_path: Path, calls: Calls,
                                                           capsys: pytest.CaptureFixture) -> None:
    """``--list-works`` reads only stage0: no ledger, no reviews, no output files."""
    world = _two_works_world(tmp_path)
    _retarget(world, E, "lit_run2", RUN2_POINTER)
    before = _snapshot(tmp_path)
    build.main(["--work-dir", str(world.work), "--list-works", "--pool", str(tmp_path / "no_ledger.jsonl")])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == ("scanned-edition works in stage0: 4 works, 5 sides "
                        "(literature 3 works / 4 sides, lit_run2 1 works / 1 sides)")
    assert lines[2:] == [f"    2  {LIT_KEY}", f"    1  {RUN2_KEY}", f"    1  {OTHER_KEY}", "    1  literature:unknown"]
    assert _snapshot(tmp_path) == before and calls.saved == []


def test_list_works_cli_needs_a_stage0_file(tmp_path: Path) -> None:
    """A work dir without stage0 is an error, not an empty list."""
    with pytest.raises(ValueError, match="stage0 candidates not found"):
        build.main(["--work-dir", str(tmp_path / "nowhere"), "--list-works"])
