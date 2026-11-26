"""Check source-review binding, photograph identity, and training split in the edition path."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.finetuning.qwen_hebrew.ja_ar_build import assign, build_rows as br, edition_candidate as ec
from src.finetuning.qwen_hebrew.ja_ar_build import release_audit


def test_edition_review_binds_view_and_retains_rnl_train_split(tmp_path: Path) -> None:
    """A reviewed RNL crop stays in train even if the document hashes to validation.

    :param tmp_path: Temporary photograph and derived output directory.
    """
    image_path = tmp_path / "source.jpg"
    Image.new("RGB", (160, 140), "white").save(image_path)
    lines = ["זה הוא הספר אשר נתן לו על ידי השליח", "והוא עוד עתיד לבוא אל הבית הגדול מחר"]
    group = next(f"g{i}" for i in range(200) if br.split_of(f"g{i}") == "val")
    cand = {"row_id": "r", "document_group": group, "rnl": True, "source": "fgp_targeted",
            "targets": [{"side_key": "pdf1", "lines": lines}]}
    pair = {"pair_id": "p", "row_id": "r", "side_keys": ["pdf1"], "source": "fgp_targeted", "script": "hebrew",
            "document_group": group, "lines": lines, "line_refs": [["pdf1", 0], ["pdf1", 1]], "flags": [],
            "image_path": str(image_path), "image_sha256": br.file_sha256(str(image_path)),
            "crop": [4, 8, 126, 112], "rotation": 180, "open_lines": [{"lines": ["printed gap"]}],
            "decision": "review", "reasons": [],
            "remedies": ["assemble", "crop"], "evidence": {"target_letters": 55}}
    view, _, _ = br.derived_image(pair, tmp_path / "preview")
    pair["edition"] = {"pdf_sha256": "source", "view_sha256": br.file_sha256(view), "line_ids": ["a", "b"]}
    candidates = {"r": cand}
    answer = br.final_answer(pair, candidates, {}, set())
    record = {"review_key": ec.review_key(pair, answer, candidates), "reviewer": "inspection", "decision": "complete",
              "observation": "Two printed lines cover the upright cropped leaf."}
    pair["edition"]["line_ids"].reverse()
    with pytest.raises(ValueError, match="stale inspection"):
        ec.apply_review(pair, answer, record, candidates, {}, set())
    pair["edition"]["line_ids"].reverse()
    ec.apply_review(pair, answer, record, candidates, {}, set())
    verdicts = {("r", "pdf1"): {"tier": "novel_strict", "eval_holdout": False}}
    args = (candidates, verdicts, {}, set(), None, ["novel_strict"], tmp_path / "out", assign.GateParams())
    rows, manifest, _ = br.build([pair], *args, remedy_classes=["assemble_crop"])
    assert len(rows) == 1 and rows[0]["answer"] == answer.text
    assert manifest[0]["split"] == "train_page"
    assert release_audit.approval_kind(manifest[0]) == "source_review"
    pair["edition"]["view_sha256"] = "stale"
    rows, _, decisions = br.build([pair], *args, remedy_classes=["assemble_crop"])
    assert not rows and decisions[0]["reasons"] == ["approval_void"]
    # A binding made against a different rendering must still fail on the delivered bytes.
    pair["approval"]["review_key"] = ec.review_key(pair, answer, candidates)
    with pytest.raises(ValueError, match="reviewed view"):
        br.build([pair], *args, remedy_classes=["assemble_crop"])


def test_declared_unread_lines_cover_every_placed_blank() -> None:
    """A missing whitespace-only line in a view's unread ledger must invalidate it."""
    view = {"row_id": "r", "image_sha256": "photo", "line_ids": ["A", "B"],
            "draft_line_ids": ["A"], "placed_without_reading": []}
    lines = [{"line_id": "A", "placement": view, "text": "ספר"},
             {"line_id": "B", "placement": view, "text": "  "}]
    with pytest.raises(ValueError, match="Placed line IDs"):
        ec.make_pair(view, {"lines": lines}, {}, {}, {})


def test_explicit_group_hold_blocks_active_sibling(tmp_path: Path) -> None:
    """A positive view-level manuscript hold still applies when its own check has no text.

    :param tmp_path: Workspace for the synthetic views and exposure records.
    """
    import argparse
    from src.finetuning.qwen_hebrew.ja_ar_build.edition_assembly import write_records
    views = tmp_path / "views.jsonl"
    checks = tmp_path / "checks.jsonl"
    write_records(views, [{"row_id": "a", "document_group": "g", "image_sha256": "a",
                           "exposure_status": "manuscript_overlap_hold"},
                          {"row_id": "b", "document_group": "g", "image_sha256": "b",
                           "exposure_status": "passed_completed_checks"}])
    write_records(checks, [])
    with pytest.raises(ValueError, match="Conflicting exposure"):
        ec.prepare(argparse.Namespace(views=views, checks=checks), tmp_path)
