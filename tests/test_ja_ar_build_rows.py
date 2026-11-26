"""Synthetic tests of the ja_ar_v1 row builder (selection, fallbacks, tiers, hold-outs, quality, derived images)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from PIL import Image

from src.finetuning.qwen_hebrew.ja_ar_build import approval, assign, build_rows, clean_v2

LINES_A = ["אבגדהוזחטי כלמנסעפצקר", "שתאבגדהוז חטיכלמנסעפ", "צקרשתאבגד הוזחטיכלמנ"]
LINES_B = ["סעפצקרשתא בגדהוזחטיכ", "למנסעפצקר שתאבגדהוזח", "טיכלמנסעפ צקרשתאבגדה"]


def make_image(tmp: Path, name: str = "p.jpg", shade: int = 255) -> str:
    """A small stored photograph.

    :param tmp: Folder.
    :param name: File name.
    :param shade: Grey level (another shade gives another photograph, with another digest).
    :return: Path.
    """
    path = tmp / name
    Image.new("RGB", (400, 300), (shade, shade, shade)).save(path)
    return str(path)


def cand(row_id: str, side_key: str, lines: List[str], source: str = "pgp") -> Dict[str, Any]:
    """A candidate record with one target side.

    :param row_id: Candidate id.
    :param side_key: Side key.
    :param lines: Target lines.
    :param source: Source.
    :return: Record.
    """
    return {"row_id": row_id, "source": source, "document_group": f"g:{row_id}", "canonical_ids": [], "pgpids": [],
            "shelfmark": "T-S X", "rnl": False, "targets": [{"side_key": side_key, "lines": lines}]}


def pair(row_id: str, side_key: str, lines: List[str], image: str, sha: Optional[str] = None,
         dup_rank: Optional[int] = None, crop: Optional[List[int]] = None, rotation: int = 0,
         source: str = "pgp") -> Dict[str, Any]:
    """An accepted pair keeping every line, with no unresolved line beside its text.

    :param row_id: Candidate id.
    :param side_key: Side key.
    :param lines: Kept lines.
    :param image: Photograph path.
    :param sha: Photograph digest (default: the file's own).
    :param dup_rank: Duplicate rank.
    :param crop: Crop box.
    :param rotation: Rotation.
    :param source: Source.
    :return: Record.
    """
    sha = sha or build_rows.file_sha256(image)
    return {"open_lines": [], "pair_id": f"{row_id}|{side_key}|{sha[:12]}", "row_id": row_id, "side_key": side_key,
            "side_keys": [side_key], "source": source, "document_group": f"g:{row_id}", "script": "hebrew",
            "image_sha256": sha, "image_path": image, "decision": "accept", "remedies": ["crop"] if crop else [],
            "line_indices": list(range(len(lines))), "line_refs": [[side_key, i] for i in range(len(lines))],
            "lines": lines, "rotation": rotation, "crop": crop, "dup_rank": dup_rank, "flags": [], "reasons": [],
            "evidence": {"target_letters": 60}}


def verdict(row_id: str, side_key: str, tier: str = "novel_strict", hold: bool = False) -> Dict[str, Any]:
    """A hold-out v2 verdict.

    :param row_id: Candidate id.
    :param side_key: Side key.
    :param tier: Tier.
    :param hold: Eval hold-out.
    :return: Record.
    """
    return {"row_id": row_id, "side_key": side_key, "tier": tier, "pair_tiers": {}, "eval_holdout": hold,
            "eval_families": ["new_eval"] if hold else []}


def approved(pairs: List[Dict[str, Any]], cands: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Bind every pair's approval to its final answer, as the adjudication step does for accepted pairs.

    :param pairs: Accepted pairs (changed in place).
    :param cands: Candidates by row id.
    :return: The pairs.
    """
    for p in pairs:
        assert build_rows.stamp_approval(p, cands, {}, set(), "rule") is None
    return pairs


def run(pairs: List[Dict[str, Any]], cands: List[Dict[str, Any]], verdicts: List[Dict[str, Any]], tmp: Path) -> Any:
    """Build without tokenizer or 3-gram models; every pair is approved first.

    :param pairs: Pairs.
    :param cands: Candidates.
    :param verdicts: Verdicts.
    :param tmp: Folder for derived images.
    :return: ``(rows, manifest, decisions)``.
    """
    by_id = {c["row_id"]: c for c in cands}
    return build_rows.build(approved(pairs, by_id), by_id,
                            {(v["row_id"], v["side_key"]): v for v in verdicts}, {}, set(), None,
                            build_rows.DEFAULT_TIERS, tmp / "derived", assign.GateParams(),
                            remedy_classes=("as_is", "margin_block", "edge_kept", "crop"))


def test_primary_edition_wins_and_fallback_is_shadowed(tmp_path: Path) -> None:
    """Two editions of one text on one photograph: one row, the other shadowed."""
    img = make_image(tmp_path)
    pairs = [pair("r1", "recto", LINES_A, img, dup_rank=0), pair("r2", "recto", LINES_A, img, dup_rank=1, source="fjp")]
    rows, manifest, decisions = run(pairs, [cand("r1", "recto", LINES_A), cand("r2", "recto", LINES_A, "fjp")],
                                    [verdict("r1", "recto"), verdict("r2", "recto")], tmp_path)
    assert len(rows) == 1 and manifest[0]["row_id"] == "r1"
    assert any(d["decision"] == "shadowed" for d in decisions)


def test_fallback_promoted_when_primary_is_trained(tmp_path: Path) -> None:
    """When the primary edition was trained, the next edition becomes the row."""
    img = make_image(tmp_path)
    pairs = [pair("r1", "recto", LINES_A, img, dup_rank=0), pair("r2", "recto", LINES_A, img, dup_rank=1, source="fjp")]
    rows, manifest, _ = run(pairs, [cand("r1", "recto", LINES_A), cand("r2", "recto", LINES_A, "fjp")],
                            [verdict("r1", "recto", "previously_trained"), verdict("r2", "recto")], tmp_path)
    assert [m["row_id"] for m in manifest] == ["r2"]


def test_eval_holdout_and_uncertain_are_excluded(tmp_path: Path) -> None:
    """Held-out and uncertain sides never become rows."""
    img = make_image(tmp_path)
    pairs = [pair("r1", "recto", LINES_A, img), pair("r2", "recto", LINES_B, make_image(tmp_path, "b.jpg", 200))]
    rows, _, decisions = run(pairs, [cand("r1", "recto", LINES_A), cand("r2", "recto", LINES_B)],
                             [verdict("r1", "recto", hold=True), verdict("r2", "recto", "uncertain_history")], tmp_path)
    assert rows == []
    assert {d["stage"] for d in decisions} == {"holdout", "novelty"}


def test_too_few_letters_is_a_quality_exclusion(tmp_path: Path) -> None:
    """A pair under the letter minimum is excluded at the quality stage."""
    img = make_image(tmp_path)
    short = ["אבג דה"]
    rows, _, decisions = run([pair("r1", "recto", short, img)], [cand("r1", "recto", short)],
                             [verdict("r1", "recto")], tmp_path)
    assert rows == [] and decisions[0]["reasons"] == ["too_few_letters"]


def test_crop_writes_a_derived_image(tmp_path: Path) -> None:
    """A crop remedy gives the row a derived photograph of the cropped size."""
    img = make_image(tmp_path)
    rows, manifest, _ = run([pair("r1", "recto", LINES_A, img, crop=[10, 20, 210, 220])],
                            [cand("r1", "recto", LINES_A)], [verdict("r1", "recto")], tmp_path)
    assert rows[0]["image"] != img and (rows[0]["image_width"], rows[0]["image_height"]) == (200, 200)
    assert manifest[0]["crop"] == [10, 20, 210, 220]


def test_split_is_stable_per_document_group() -> None:
    """The split of a document group never changes and about 3 % of groups go to val."""
    groups = [f"g:{k}" for k in range(4000)]
    first = [build_rows.split_of(g) for g in groups]
    assert first == [build_rows.split_of(g) for g in groups]
    share = first.count("val") / len(first)
    assert 0.015 < share < 0.045


def test_repaired_pairs_go_to_the_repair_queue(tmp_path: Path) -> None:
    """With the default remedy classes, a cropped pair is held for repair, a plain one becomes a row."""
    img = make_image(tmp_path)
    pairs = [pair("r1", "recto", LINES_A, img), pair("r2", "recto", LINES_B, make_image(tmp_path, "b.jpg", 200),
                                                    crop=[0, 0, 100, 100])]
    cands = {c["row_id"]: c for c in [cand("r1", "recto", LINES_A), cand("r2", "recto", LINES_B)]}
    rows, manifest, decisions = build_rows.build(approved(pairs, cands), cands,
                                                 {(v["row_id"], v["side_key"]): v for v in [verdict("r1", "recto"),
                                                                                           verdict("r2", "recto")]},
                                                 {}, set(), None, build_rows.DEFAULT_TIERS, tmp_path / "derived",
                                                 assign.GateParams())
    assert [m["row_id"] for m in manifest] == ["r1"]
    assert any(d["stage"] == "repair_queue" and d["reasons"] == ["remedy:crop"] for d in decisions)


def test_prior_build_photo_and_side_are_not_built_again(tmp_path: Path) -> None:
    """An incremental build skips a photograph or a side that an earlier build already made into a row."""
    img = make_image(tmp_path)
    pairs = [pair("r1", "recto", LINES_A, img), pair("r2", "recto", LINES_B, make_image(tmp_path, "b.jpg", 200)),
             pair("r3", "verso", LINES_B, make_image(tmp_path, "c.jpg", 150))]
    cands = {c["row_id"]: c for c in [cand("r1", "recto", LINES_A), cand("r2", "recto", LINES_B),
                                      cand("r3", "verso", LINES_B)]}
    verdicts = {(v["row_id"], v["side_key"]): v for v in [verdict("r1", "recto"), verdict("r2", "recto"),
                                                         verdict("r3", "verso")]}
    prior = {"images": {pairs[0]["image_sha256"]}, "sides": {("r2", "recto")}}
    rows, manifest, decisions = build_rows.build(approved(pairs, cands), cands, verdicts, {}, set(), None,
                                                 build_rows.DEFAULT_TIERS, tmp_path / "derived", assign.GateParams(),
                                                 prior=prior)
    assert [m["row_id"] for m in manifest] == ["r3"]
    reasons = {d["pair_id"].split("|")[0]: d["reasons"] for d in decisions if d["stage"] == "prior_build"}
    assert reasons == {"r1": ["prior_build_photo"], "r2": ["prior_build_side"]}


def test_pair_on_a_registered_eval_photo_is_held_out(tmp_path: Path) -> None:
    """A pair whose photograph is a registered eval photograph (same digest or near pHash) never becomes a row."""
    img = make_image(tmp_path)
    pairs = [pair("r1", "recto", LINES_A, img), pair("r2", "recto", LINES_B, make_image(tmp_path, "b.jpg", 200)),
             pair("r3", "verso", LINES_B, make_image(tmp_path, "c.jpg", 150))]
    cands = {c["row_id"]: c for c in [cand("r1", "recto", LINES_A), cand("r2", "recto", LINES_B),
                                      cand("r3", "verso", LINES_B)]}
    cands["r2"]["images"] = [{"sha256": pairs[1]["image_sha256"], "phash": "ffff0000ffff0000"}]
    cands["r3"]["images"] = [{"sha256": pairs[2]["image_sha256"], "phash": "0f0f0f0f0f0f0f0f"}]
    verdicts = {(v["row_id"], v["side_key"]): v for v in [verdict("r1", "recto"), verdict("r2", "recto"),
                                                         verdict("r3", "verso")]}
    eval_photos = {"sha256": {pairs[0]["image_sha256"]}, "phash": {"ffff0000ffff0001"}}
    rows, manifest, decisions = build_rows.build(approved(pairs, cands), cands, verdicts, {}, set(), None,
                                                 build_rows.DEFAULT_TIERS, tmp_path / "derived", assign.GateParams(),
                                                 eval_photos=eval_photos)
    assert [m["row_id"] for m in manifest] == ["r3"]
    held = sorted(d["pair_id"].split("|")[0] for d in decisions if d["reasons"] == ["eval_holdout:eval_photo"])
    assert held == ["r1", "r2"]


def test_load_eval_photos_reads_registered_items(tmp_path: Path) -> None:
    """Digests and pHashes come from every registered set's items."""
    import json
    (tmp_path / "a.json").write_text(json.dumps({"items": [{"image_sha256": ["x" * 64], "phash": ["00ff"]}]}))
    (tmp_path / "b.json").write_text(json.dumps({"items": [{"image_sha256": [], "phash": []}]}))
    got = build_rows.load_eval_photos(tmp_path)
    assert got == {"sha256": {"x" * 64}, "phash": {"00ff"}}


def test_side_without_verdict_takes_the_document_verdict(tmp_path: Path) -> None:
    """A document-level verdict (side key "") covers every side: a novel document builds, a held one does not."""
    img = make_image(tmp_path)
    pairs = [pair("r1", "recto", LINES_A, img), pair("r2", "verso", LINES_B, make_image(tmp_path, "b.jpg", 200))]
    cands = {c["row_id"]: c for c in [cand("r1", "recto", LINES_A), cand("r2", "verso", LINES_B)]}
    verdicts = {("r1", ""): verdict("r1", ""), ("r2", ""): verdict("r2", "", hold=True)}
    rows, manifest, decisions = build_rows.build(approved(pairs, cands), cands, verdicts, {}, set(), None,
                                                 build_rows.DEFAULT_TIERS, tmp_path / "derived", assign.GateParams())
    assert [m["row_id"] for m in manifest] == ["r1"]
    assert any(d["stage"] == "holdout" for d in decisions)


def build_one(p: Dict[str, Any], lines: List[str], tmp: Path, source: str = "pgp") -> Any:
    """Build a single pair as it is (no approval is added here).

    :param p: The pair.
    :param lines: Its candidate's lines.
    :param tmp: Folder for derived images.
    :param source: Candidate source.
    :return: ``(rows, manifest, decisions)``.
    """
    cands = {"r1": cand("r1", "recto", lines, source)}
    return build_rows.build([p], cands, {("r1", "recto"): verdict("r1", "recto")}, {}, set(), None,
                            build_rows.DEFAULT_TIERS, tmp / "derived", assign.GateParams())


def test_a_pair_without_an_approval_builds_no_row(tmp_path: Path) -> None:
    """An accepted pair nobody approved with its final answer is held."""
    rows, _, decisions = build_one(pair("r1", "recto", LINES_A, make_image(tmp_path)), LINES_A, tmp_path)
    assert rows == [] and decisions[0]["stage"] == "approval" and decisions[0]["reasons"] == ["no_approval"]


def test_a_pair_changed_after_its_approval_builds_no_row(tmp_path: Path) -> None:
    """Losing a line, or getting another crop, after the approval voids it (integrity audit 2026-11-22)."""
    img = make_image(tmp_path)
    cands = {"r1": cand("r1", "recto", LINES_A)}
    shorter = approved([pair("r1", "recto", LINES_A, img)], cands)[0]
    for key in ("lines", "line_refs", "line_indices"):
        shorter[key] = shorter[key][:-1]
    rows, _, decisions = build_one(shorter, LINES_A, tmp_path)
    assert rows == [] and decisions[0]["reasons"] == ["approval_void"]
    cropped = approved([pair("r1", "recto", LINES_A, img)], cands)[0]
    cropped["crop"], cropped["remedies"] = [0, 0, 200, 200], []
    rows, _, decisions = build_one(cropped, LINES_A, tmp_path)
    assert rows == [] and decisions[0]["reasons"] == ["approval_void"]
    whole = approved([pair("r1", "recto", LINES_A, img)], cands)[0]
    rows, manifest, _ = build_one(whole, LINES_A, tmp_path)
    assert rows[0]["answer"].split("\n") == LINES_A and manifest[0]["approval"]["by"] == "rule"
    assert manifest[0]["approval"]["scope"] == approval.SCOPE and len(manifest[0]["row_image_sha256"]) == 64


def test_an_approval_does_not_survive_a_change_of_the_source_text(tmp_path: Path) -> None:
    """A re-ingest that changes another line of the pair's side voids the approval of the pair."""
    img = make_image(tmp_path)
    cands = {"r1": cand("r1", "recto", LINES_A + LINES_B)}
    p = pair("r1", "recto", LINES_A, img)
    approved([p], cands)
    rows, _, decisions = build_one(p, LINES_A + LINES_B[:2], tmp_path)
    assert rows == [] and decisions[0]["reasons"] == ["approval_void"]


def test_a_line_the_character_model_flags_stays_in_the_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A scrambled-line flag never removes a line: the answer keeps every approved line and records the flag."""
    real = clean_v2.clean_side

    def flagged(lines: List[str], **kwargs: Any) -> Any:
        """The real cleaner, with the second line flagged as scrambled.

        :param lines: The side's lines.
        :param kwargs: Passed through.
        :return: The clean result.
        """
        result = real(lines, **kwargs)
        result.flags.append({"flag": "scrambled_candidate", "line_index": 1, "out_index": 1, "score": -9.0})
        return result

    monkeypatch.setattr(build_rows.clean_v2, "clean_side", flagged)
    cands = {"r1": cand("r1", "recto", LINES_A)}
    p = pair("r1", "recto", LINES_A, make_image(tmp_path))
    answer = build_rows.final_answer(p, cands, {}, set())
    assert answer.lines == LINES_A and answer.scrambled_lines == 1 and answer.letters_kept
    assert answer.scrambled_at == [2]
    unread = pair("r1", "recto", LINES_A, p["image_path"])
    unread["unread_lines"] = [2]                                          # ... and no read supports that line either
    assert build_rows.stamp_approval(unread, cands, {}, set(), "rule") == "unlikely_unread_line"
    assert build_rows.final_answer(unread, cands, {}, set()).lines == LINES_A   # held for a judge, never shortened
    assert build_rows.stamp_approval(unread, cands, {}, set(), "judges") is None
    rows, manifest, _ = build_one(approved([p], cands)[0], LINES_A, tmp_path)
    assert rows[0]["answer"].split("\n") == LINES_A
    assert manifest[0]["scrambled_lines"] == 1 and "scrambled_candidate" in manifest[0]["clean_flags"]


def test_unsettled_lines_next_to_a_text_block_its_approval(tmp_path: Path) -> None:
    """A pair with unresolved lines beside its text cannot be approved, and is held if they appear afterwards."""
    img = make_image(tmp_path)
    cands = {"r1": cand("r1", "recto", LINES_A + LINES_B)}
    open_entry = {"after_line": 3, "before_line": None, "lines": LINES_B[:1], "refs": [["recto", 3]]}
    p = pair("r1", "recto", LINES_A, img)
    p["open_lines"] = [dict(open_entry)]
    assert build_rows.stamp_approval(p, cands, {}, set(), "rule") == "unresolved_lines"
    p["open_lines"][0]["settled"] = "absent"
    assert build_rows.stamp_approval(p, cands, {}, set(), "judges") is None
    p["open_lines"] = [dict(open_entry)]                                  # reopened after the approval
    rows, _, decisions = build_one(p, LINES_A + LINES_B, tmp_path)
    assert rows == [] and decisions[0]["reasons"] == ["unresolved_lines"]


def test_a_pair_that_does_not_fit_the_candidates_file_is_never_built(tmp_path: Path) -> None:
    """Pair lines that are not the candidate's lines at the pair's references mean stale records."""
    cands = {"r1": cand("r1", "recto", LINES_B)}
    p = pair("r1", "recto", LINES_A, make_image(tmp_path))
    assert build_rows.source_segments(p, cands) is None
    assert build_rows.stamp_approval(p, cands, {}, set(), "rule") == "pair_lines_differ_from_candidate"
    p["approval"] = {"by": "rule", "scope": approval.SCOPE, "fingerprint": "0" * 64}
    rows, _, decisions = build_one(p, LINES_B, tmp_path)
    assert rows == [] and decisions[0]["reasons"][0] == "pair_lines_differ_from_candidate"


def test_an_answer_that_cleaning_shortened_needs_judges_who_saw_it(tmp_path: Path) -> None:
    """When a text rule removes letters, the rule's approval of the longer lines does not cover the answer."""
    doubled = [LINES_A[0], LINES_A[0], LINES_A[1], LINES_A[2]]            # the cleaner drops a line set twice
    cands = {"r1": cand("r1", "recto", doubled)}
    p = pair("r1", "recto", doubled, make_image(tmp_path))
    answer = build_rows.final_answer(p, cands, {}, set())
    assert answer.lines == LINES_A and not answer.letters_kept
    assert build_rows.stamp_approval(p, cands, {}, set(), "rule") == "answer_changed_by_cleaning"
    assert "approval" not in p
    assert build_rows.stamp_approval(p, cands, {}, set(), "judges", answer_seen=True) is None
    rows, _, _ = build_one(p, doubled, tmp_path)
    assert rows[0]["answer"].split("\n") == LINES_A


def test_a_photograph_whose_file_changed_since_the_approval_builds_no_row(tmp_path: Path) -> None:
    """The approval names the photograph by digest; other bytes under the same path are another photograph."""
    img = make_image(tmp_path)
    cands = {"r1": cand("r1", "recto", LINES_A)}
    p = approved([pair("r1", "recto", LINES_A, img)], cands)[0]
    rows, manifest, _ = build_one(p, LINES_A, tmp_path)
    assert len(rows) == 1 and manifest[0]["row_image_sha256"] == p["image_sha256"]
    make_image(tmp_path, shade=10)                                        # the stored file is replaced
    rows, _, decisions = build_one(p, LINES_A, tmp_path)
    assert rows == [] and decisions[0]["reasons"] == ["image_file_changed"]


def test_no_rule_approves_a_pair_whose_left_out_lines_are_not_accounted_for(tmp_path: Path) -> None:
    """A gate that says nothing about the lines beside a pair's text cannot vouch that the text is whole."""
    cands = {"r1": cand("r1", "recto", LINES_A)}
    p = pair("r1", "recto", LINES_A, make_image(tmp_path))
    del p["open_lines"]
    assert build_rows.stamp_approval(p, cands, {}, set(), "rule") == "no_line_accounting"
    assert build_rows.stamp_approval(p, cands, {}, set(), "judges") is None   # judges looked at the photograph
