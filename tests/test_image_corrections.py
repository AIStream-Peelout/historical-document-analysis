"""Tests for the durable image-corrections record, its converters and its application in the merge.

Every id, file name and slip text here is synthetic.
"""

import copy
import hashlib
import json

import pytest

from src.datasets.document_models.genizah_document import GenizahDocument
from src.datasets.merging.image_corrections import (
    BASIS_EYE,
    BASIS_SLIP_READ,
    KIND_TRANSCRIPTION,
    REVIEW_FILE_NAME_MISMATCH,
    REVIEW_HAS_REVIEWED_CORRECTION,
    SOURCE_LABEL_PILOT,
    SOURCE_SLIP_RELABEL,
    append_image_corrections,
    build_transcription_correction,
    correction_kind,
    from_label_pilot,
    from_relabels,
    hold_for_reviewed,
    image_name,
    iter_corrections,
    load_image_corrections,
    load_transcription_corrections,
    new_corrections,
    section_digest,
    short_evidence,
    split_relabels,
    transcription_sections,
    validate_correction,
)
from src.datasets.merging.merge_shelfmarks import (
    MOVED_FROM_KEY,
    apply_image_corrections,
    apply_transcription_corrections,
    preferred_image_source,
)

BASE = "https://example.org/images"


def _sha(text: str) -> str:
    """Synthetic sha256 for a file name.

    :param text: Any string.
    :returns: 64 hex characters.
    """
    return hashlib.sha256(text.encode()).hexdigest()


def _correction(**overrides) -> dict:
    """A valid synthetic move correction, with *overrides* applied.

    :param overrides: Fields to replace.
    :returns: The record.
    """
    record = {
        "image": "Q_7_1r.jpg",
        "url": f"{BASE}/Q_7_1r.jpg",
        "sha256": _sha("Q_7_1r.jpg"),
        "from_canonical_id": "Testville_LIB_Q_7",
        "to_canonical_id": "Testville_LIB_Q_6",
        "to_shelfmark_text": "Q 6",
        "side": "v",
        "basis": BASIS_SLIP_READ,
        "evidence": "LIB Q 6 verso",
        "reviewed": False,
        "recorded_at": "2026-10-31",
        "source": SOURCE_SLIP_RELABEL,
    }
    record.update(overrides)
    return record


def _write_jsonl(path, rows) -> None:
    """Write *rows* as JSONL.

    :param path: Destination.
    :param rows: JSON-serialisable rows.
    """
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _record(cid: str, fjp=(), ktiv_images=()) -> dict:
    """A minimal merged record with an ``images`` block shaped like the merge's.

    :param cid: Canonical id.
    :param fjp: ``images.fjp`` entries.
    :param ktiv_images: KTIV object paths (populated KTIV block when non-empty).
    :returns: The record.
    """
    images = {
        "fjp": list(fjp),
        "ktiv": {"images": list(ktiv_images), "image_urls": [f"{BASE}/{p}" for p in ktiv_images],
                 "image_count": len(ktiv_images), "populated": bool(ktiv_images)} if ktiv_images else None,
        "bodleian": None,
    }
    images["preferred_source"] = preferred_image_source(images)
    return {"canonical_id": cid, "shelfmark_display": cid.replace("_", " "), "images": images,
            "sources": {}, "sources_present": ["fjp"]}


# ───────────────────────────── schema ─────────────────────────────

def test_validate_accepts_a_move_and_turns_an_integer_side_into_text():
    out = validate_correction(_correction(side=2))
    assert out["side"] == "2"
    assert correction_kind(out) == "move"


def test_correction_kinds():
    assert correction_kind(_correction(to_canonical_id=None)) == "detach"
    assert correction_kind(_correction(to_canonical_id="Testville_LIB_Q_7")) == "side_only"
    assert correction_kind(_correction(to_canonical_id="Testville_LIB_Q_7", side=None)) == "revoke"


@pytest.mark.parametrize("overrides", [
    {"basis": "guess"},
    {"evidence": "x" * 61},
    {"recorded_at": "31/10/2026"},
    {"recorded_at": "2026-02-30"},
    {"sha256": "ABC"},
    {"url": f"{BASE}/other.jpg"},
    {"image": "dir/Q_7_1r.jpg"},
    {"reviewed": "yes"},
    {"to_canonical_id": None, "to_shelfmark_text": None},
    {"from_canonical_id": ""},
    {"side": ""},
    {"extra_field": 1},
])
def test_validate_rejects_malformed_rows(overrides):
    with pytest.raises(ValueError):
        validate_correction(_correction(**overrides))


def test_validate_rejects_a_missing_field():
    row = _correction()
    del row["basis"]
    with pytest.raises(ValueError, match="missing"):
        validate_correction(row)


def test_short_evidence_collapses_space_and_cuts():
    assert short_evidence("  A   B ") == "A B"
    cut = short_evidence("y" * 100)
    assert len(cut) == 60 and cut.endswith("...")
    assert short_evidence(None) == ""


def test_image_name_handles_urls_and_bare_names():
    assert image_name(f"{BASE}/123_image.jpg") == "123_image.jpg"
    assert image_name("Q_7_1r.jpg") == "Q_7_1r.jpg"


# ───────────────────────────── file I/O ─────────────────────────────

def test_load_later_line_overrides_earlier_for_the_same_image(tmp_path):
    path = tmp_path / "c.jsonl"
    _write_jsonl(path, [_correction(), _correction(to_canonical_id="Testville_LIB_Q_5", to_shelfmark_text="Q 5"),
                        _correction(image="Z_1.jpg", url=None)])
    loaded = load_image_corrections(str(path))
    assert set(loaded) == {"Q_7_1r.jpg", "Z_1.jpg"}
    assert loaded["Q_7_1r.jpg"]["to_canonical_id"] == "Testville_LIB_Q_5"


def test_load_names_the_line_of_a_malformed_row(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps(_correction()) + "\n\n" + json.dumps(_correction(basis="x")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"c\.jsonl:3"):
        load_image_corrections(str(path))
    path.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"c\.jsonl:1: not valid JSON"):
        load_image_corrections(str(path))


def test_append_creates_the_file_and_skips_duplicates(tmp_path):
    path = tmp_path / "c.jsonl"
    assert append_image_corrections(str(path), [_correction(), _correction()]) == 1
    # Same image + target + side again: skipped; a new target for the image: appended (and wins on load).
    again = [_correction(evidence="re-read"),
             _correction(to_canonical_id="Testville_LIB_Q_5", to_shelfmark_text="Q 5")]
    assert append_image_corrections(str(path), again) == 1
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert load_image_corrections(str(path))["Q_7_1r.jpg"]["to_canonical_id"] == "Testville_LIB_Q_5"


def test_append_writes_nothing_when_one_record_is_bad(tmp_path):
    path = tmp_path / "c.jsonl"
    with pytest.raises(ValueError):
        append_image_corrections(str(path), [_correction(), _correction(basis="x")])
    assert not path.exists()


def test_a_revoke_appended_later_cancels_a_correction(tmp_path):
    path = tmp_path / "c.jsonl"
    append_image_corrections(str(path), [_correction()])
    revoke = _correction(to_canonical_id="Testville_LIB_Q_7", side=None, basis=BASIS_EYE, reviewed=True)
    assert append_image_corrections(str(path), [revoke]) == 1
    assert correction_kind(load_image_corrections(str(path))["Q_7_1r.jpg"]) == "revoke"


# ───────────────────────────── relabel converter ─────────────────────────────

def _manifest_row(cid: str, name: str, side_token: str = "r", new_cid: str = "", sha_of: str = "") -> dict:
    """A synthetic image-manifest row (v2 shape when *new_cid* is given: ``canonical_id`` rewritten by the run).

    :param cid: Record the file is filed under (``old_canonical_id`` in a v2 row).
    :param name: File name.
    :param side_token: Side token from the file name.
    :param new_cid: v2 only: the record the run moved the row to (``None`` for a detach is written as ``"-"``).
    :param sha_of: Name whose synthetic sha256 the file has (default: *name*; lets two files share bytes).
    :returns: The row.
    """
    row = {"canonical_id": cid, "url": f"{BASE}/{name}", "path": f"/data/{cid}/{name}",
           "sha256": _sha(sha_of or name), "side_token": side_token, "file_name": name}
    if new_cid:
        row.update(old_canonical_id=cid, canonical_id=None if new_cid == "-" else new_cid)
    return row


def _relabel(cid: str, name: str, outcome: str, resolved=None, parsed=None, side=None, slip="",
             file_name=True, sha_of: str = "") -> dict:
    """A synthetic relabel-run row, shaped like the run's own (``file_name``, ``parsed_shelf_mark``).

    :param cid: Old canonical id.
    :param name: File name (its sha256 is synthetic).
    :param outcome: Run outcome.
    :param resolved: Resolved canonical id or ``None``.
    :param parsed: Parsed shelf mark.
    :param side: Slip side.
    :param slip: Slip text.
    :param file_name: Write ``file_name`` (the run does; older rows may not).
    :param sha_of: Name whose synthetic sha256 the file has (default: *name*).
    :returns: The row.
    """
    row = {"sha256": _sha(sha_of or name), "old_canonical_id": cid, "slip_text": slip, "parsed_shelf_mark": parsed,
           "resolved_canonical_id": resolved, "slip_side": side, "outcome": outcome}
    if file_name:
        row["file_name"] = name
    return row


@pytest.fixture
def relabel_files(tmp_path):
    """Synthetic manifest + relabel run covering every outcome.

    :param tmp_path: pytest tmp dir.
    :returns: ``(relabels_path, manifest_path, tmp_path)``.
    """
    names = {
        "A": ("Testville_LIB_A_2", "A_2_1r.jpg"),   # known document -> move
        "B": ("Testville_LIB_A_2", "A_2_1v.jpg"),   # confirmed -> nothing
        "C": ("Testville_LIB_C_9", "C_9_1r.jpg"),   # unknown document -> detach
        "D": ("Testville_LIB_D_1", "D_1_1r.jpg"),   # needs_eye -> review
        "E": ("Testville_LIB_E_1", "E_1_1r.jpg"),   # no_slip
        "F": ("Testville_LIB_F_1", "F_1_1r.jpg"),   # unreadable
        "G": ("Testville_LIB_G_1", "G_1_1r.jpg"),   # known document without resolved id -> review
        "H": ("Testville_LIB_H_1", "H_1_1r.jpg"),   # shared file, two targets -> review
        "H2": ("Testville_LIB_H_2", "H_1_1r.jpg"),
        "I": ("Testville_LIB_I_1", "I_1_1r.jpg"),   # unknown document, resolved outside the set -> move
        "J": ("Testville_LIB_J_1", "J_1_1r.jpg"),   # resolved to its own record -> review
    }
    manifest = [_manifest_row(cid, name) for cid, name in names.values()]
    relabels = [
        _relabel(*names["A"], "relabel_to_known_document", resolved="Testville_LIB_A_1", parsed="LIB A 1", side="v",
                 slip="Testville Library   A 1 verso"),
        _relabel(*names["B"], "confirmed", resolved="Testville_LIB_A_2", slip="LIB A 2"),
        _relabel(*names["C"], "relabel_to_unknown_document", parsed="LIB C 99", slip="LIB C 99"),
        _relabel(*names["D"], "needs_eye", slip="LIB D ?"),
        _relabel(*names["E"], "no_slip"),
        _relabel(*names["F"], "unreadable"),
        _relabel(*names["G"], "relabel_to_known_document", slip="LIB G 2"),
        _relabel(*names["H"], "relabel_to_known_document", resolved="Testville_LIB_H_3", slip="LIB H 3"),
        _relabel(*names["H2"], "relabel_to_known_document", resolved="Testville_LIB_H_4", slip="LIB H 4"),
        _relabel("Testville_LIB_Z_1", "Z_1_1r.jpg", "relabel_to_known_document", resolved="Testville_LIB_Z_2"),
        _relabel(*names["I"], "relabel_to_unknown_document", resolved="Testville_LIB_I_7", parsed="LIB I 7",
                 slip="LIB I 7", side="2"),
        _relabel(*names["J"], "relabel_to_known_document", resolved="Testville_LIB_J_1", slip="LIB J 1"),
    ]
    _write_jsonl(tmp_path / "manifest.jsonl", manifest)
    _write_jsonl(tmp_path / "relabels.jsonl", relabels)
    return str(tmp_path / "relabels.jsonl"), str(tmp_path / "manifest.jsonl"), tmp_path


def test_relabels_become_moves_and_detaches_only(relabel_files):
    relabels, manifest, _ = relabel_files
    corrections, review = split_relabels(relabels, manifest, recorded_at="2026-10-31")
    by_image = {c["image"]: c for c in corrections}
    assert set(by_image) == {"A_2_1r.jpg", "C_9_1r.jpg", "I_1_1r.jpg"}
    move = by_image["A_2_1r.jpg"]
    assert (move["from_canonical_id"], move["to_canonical_id"], move["side"]) == (
        "Testville_LIB_A_2", "Testville_LIB_A_1", "v")
    assert move["url"] == f"{BASE}/A_2_1r.jpg" and move["sha256"] == _sha("A_2_1r.jpg")
    assert move["evidence"] == "Testville Library A 1 verso"
    assert (move["basis"], move["reviewed"], move["source"]) == (BASIS_SLIP_READ, False, SOURCE_SLIP_RELABEL)
    detach = by_image["C_9_1r.jpg"]
    assert detach["to_canonical_id"] is None and detach["to_shelfmark_text"] == "LIB C 99"
    assert correction_kind(detach) == "detach"
    # An unknown-document outcome with a resolved id names a corpus record outside the set: a move, not a review.
    outside = by_image["I_1_1r.jpg"]
    assert (outside["to_canonical_id"], outside["side"], correction_kind(outside)) == (
        "Testville_LIB_I_7", "2", "move")
    reasons = sorted(r["review_reason"] for r in review)
    assert reasons == ["conflicting_corrections_for_image", "conflicting_corrections_for_image",
                       "image_not_in_manifest", "known_document_without_resolved_id", "needs_eye",
                       "resolved_to_own_record"]


def test_from_relabels_writes_the_review_file_and_never_a_needs_eye_correction(relabel_files):
    relabels, manifest, tmp_path = relabel_files
    review_path = tmp_path / "review.jsonl"
    corrections = from_relabels(relabels, manifest, review_path=str(review_path), recorded_at="2026-10-31")
    review = [json.loads(line) for line in review_path.read_text(encoding="utf-8").splitlines()]
    assert len(review) == 6
    needs_eye = [r for r in review if r["review_reason"] == "needs_eye"]
    assert needs_eye[0]["image"] == "D_1_1r.jpg"
    assert "D_1_1r.jpg" not in {c["image"] for c in corrections}
    # The converter output is a valid corrections file.
    out = tmp_path / "corrections.jsonl"
    assert append_image_corrections(str(out), corrections) == 3
    assert len(load_image_corrections(str(out))) == 3


def test_relabels_join_a_v2_manifest_by_the_old_record_and_the_file_name(tmp_path):
    """The v2 manifest rewrites canonical_id on moved / detached rows; the join keys on old_canonical_id and the
    run's file_name, so bytes held by two records do not send the row to review."""
    manifest = [
        _manifest_row("Testville_LIB_V_1", "V_1_1r.jpg", new_cid="Testville_LIB_V_0"),        # moved by the run
        _manifest_row("Testville_LIB_V_9", "V_9_1r.jpg", sha_of="V_1_1r.jpg"),                 # same bytes elsewhere
        _manifest_row("Testville_LIB_W_1", "W_1_1r.jpg", new_cid="-"),                         # detached by the run
        _manifest_row("Testville_LIB_W_9", "W_9_1r.jpg", sha_of="W_1_1r.jpg"),
        _manifest_row("Testville_LIB_X_1", "X_1_1r.jpg"),
    ]
    relabels = [
        _relabel("Testville_LIB_V_1", "V_1_1r.jpg", "relabel_to_known_document", resolved="Testville_LIB_V_0",
                 slip="LIB V 0"),
        _relabel("Testville_LIB_W_1", "W_1_1r.jpg", "relabel_to_unknown_document", parsed="LIB W 77",
                 slip="Rylands W 77"),
        # The run's file name differs from the manifest url: a person must look.
        _relabel("Testville_LIB_X_1", "X_1_1v.jpg", "relabel_to_known_document", resolved="Testville_LIB_X_2",
                 slip="LIB X 2", sha_of="X_1_1r.jpg"),
    ]
    _write_jsonl(tmp_path / "m.jsonl", manifest)
    _write_jsonl(tmp_path / "r.jsonl", relabels)
    corrections, review = split_relabels(str(tmp_path / "r.jsonl"), str(tmp_path / "m.jsonl"),
                                         recorded_at="2026-10-31")
    by_image = {c["image"]: c for c in corrections}
    assert set(by_image) == {"V_1_1r.jpg", "W_1_1r.jpg"}
    assert by_image["V_1_1r.jpg"]["url"] == f"{BASE}/V_1_1r.jpg"
    assert by_image["V_1_1r.jpg"]["from_canonical_id"] == "Testville_LIB_V_1"
    assert by_image["W_1_1r.jpg"]["to_canonical_id"] is None
    assert [r["review_reason"] for r in review] == [REVIEW_FILE_NAME_MISMATCH]


def test_relabel_rows_without_file_name_fall_back_to_the_manifest_url(tmp_path):
    _write_jsonl(tmp_path / "m.jsonl", [_manifest_row("Testville_LIB_Y_1", "Y_1_1r.jpg")])
    _write_jsonl(tmp_path / "r.jsonl", [_relabel("Testville_LIB_Y_1", "Y_1_1r.jpg", "relabel_to_known_document",
                                                 resolved="Testville_LIB_Y_2", slip="LIB Y 2", file_name=False)])
    corrections, review = split_relabels(str(tmp_path / "r.jsonl"), str(tmp_path / "m.jsonl"))
    assert [c["image"] for c in corrections] == ["Y_1_1r.jpg"] and review == []


def test_hold_for_reviewed_never_lets_an_automatic_row_override_a_reviewed_one():
    reviewed = validate_correction(_correction(image="R_1_1r.jpg", url=None, from_canonical_id="Testville_LIB_R_1",
                                               to_canonical_id="Testville_LIB_R_0", side="2", basis=BASIS_EYE,
                                               reviewed=True, source=SOURCE_LABEL_PILOT))
    unreviewed = validate_correction(_correction(image="U_1_1r.jpg", url=None, from_canonical_id="Testville_LIB_U_1",
                                                 to_canonical_id="Testville_LIB_U_0"))
    established = {"R_1_1r.jpg": reviewed, "U_1_1r.jpg": unreviewed}
    candidates = [
        _correction(image="R_1_1r.jpg", url=None, from_canonical_id="Testville_LIB_R_1",
                    to_canonical_id="Testville_LIB_R_5"),                                  # disagrees -> review
        _correction(image="R_1_1r.jpg", url=None, from_canonical_id="Testville_LIB_R_1",
                    to_canonical_id="Testville_LIB_R_0", side="2"),                        # agrees -> dropped
        _correction(image="U_1_1r.jpg", url=None, from_canonical_id="Testville_LIB_U_1",
                    to_canonical_id="Testville_LIB_U_9"),                                  # unreviewed before -> kept
        _correction(image="N_1_1r.jpg", url=None, from_canonical_id="Testville_LIB_N_1"),  # new image -> kept
    ]
    kept, held, agreeing = hold_for_reviewed(candidates, established)
    assert [c["image"] for c in kept] == ["U_1_1r.jpg", "N_1_1r.jpg"] and agreeing == 1
    assert [(h["image"], h["review_reason"], h["reviewed_correction"]["to_canonical_id"]) for h in held] == [
        ("R_1_1r.jpg", REVIEW_HAS_REVIEWED_CORRECTION, "Testville_LIB_R_0")]


def test_relabels_reject_an_unknown_outcome(tmp_path):
    _write_jsonl(tmp_path / "m.jsonl", [_manifest_row("Testville_LIB_A_2", "A_2_1r.jpg")])
    _write_jsonl(tmp_path / "r.jsonl", [_relabel("Testville_LIB_A_2", "A_2_1r.jpg", "maybe")])
    with pytest.raises(ValueError, match="unknown outcome"):
        split_relabels(str(tmp_path / "r.jsonl"), str(tmp_path / "m.jsonl"))


# ───────────────────────────── label-pilot converter ─────────────────────────────

def test_label_pilot_converts_only_misfilings_with_eye_basis(tmp_path):
    eye = [
        {"key": "p001", "canonical_id": "Testville_LIB_K_5", "file": "K_5_1r.jpg", "label_read": "LIB K 4-2",
         "eye": "wrong fragment", "note": ""},
        {"key": "p002", "canonical_id": "Testville_LIB_M_1", "file": "77_image.jpg", "label_read": "OTHER 3",
         "eye": "join partner", "note": ""},
        {"key": "p003", "canonical_id": "Testville_LIB_N_1", "file": "N_1_1.jpg", "label_read": "N 8",
         "eye": "wrong fragment", "note": ""},
        {"key": "p004", "canonical_id": "Testville_LIB_P_1", "file": "P_1_1r.jpg", "label_read": "LIB P 1",
         "eye": "match", "note": ""},
    ]
    sample = [{"key": k, "image_url": f"{BASE}/{f}", "sha256": _sha(f)}
              for k, f in (("p001", "K_5_1r.jpg"), ("p002", "77_image.jpg"), ("p003", "N_1_1.jpg"),
                           ("p004", "P_1_1r.jpg"))]
    _write_jsonl(tmp_path / "eye.jsonl", eye)
    _write_jsonl(tmp_path / "sample.jsonl", sample)
    targets = {
        "p001": {"to_canonical_id": "Testville_LIB_K_4", "to_shelfmark_text": "LIB K 4", "side": "2"},
        "p003": {"to_canonical_id": None, "to_shelfmark_text": "LIB N 8", "side": None},
    }
    records = from_label_pilot(str(tmp_path / "eye.jsonl"), str(tmp_path / "sample.jsonl"), targets)
    assert [r["image"] for r in records] == ["K_5_1r.jpg", "N_1_1.jpg"]
    assert all(r["basis"] == BASIS_EYE and r["reviewed"] and r["source"] == SOURCE_LABEL_PILOT for r in records)
    assert records[0]["side"] == "2" and records[0]["evidence"] == "LIB K 4-2"
    assert correction_kind(records[1]) == "detach"
    with pytest.raises(KeyError):
        from_label_pilot(str(tmp_path / "eye.jsonl"), str(tmp_path / "sample.jsonl"), {"p001": targets["p001"]})


# ───────────────────────────── application ─────────────────────────────

def _corrections(*records) -> dict:
    """Key validated corrections by image, as the loader does.

    :param records: Correction records.
    :returns: ``{image: record}``.
    """
    return {r["image"]: validate_correction(r) for r in records}


def test_move_keeps_order_records_provenance_and_side():
    records = {
        "Testville_LIB_Q_7": _record("Testville_LIB_Q_7", ["Q_7_1r.jpg", "Q_7_1v.jpg", f"{BASE}/55_image.jpg"]),
        "Testville_LIB_Q_6": _record("Testville_LIB_Q_6", ["Q_6_1r.jpg", "Q_6_1v.jpg"]),
    }
    stats = apply_image_corrections(records, _corrections(_correction()))
    assert records["Testville_LIB_Q_7"]["images"]["fjp"] == ["Q_7_1v.jpg", f"{BASE}/55_image.jpg"]
    dst = records["Testville_LIB_Q_6"]["images"]
    assert dst["fjp"] == ["Q_6_1r.jpg", "Q_6_1v.jpg", "Q_7_1r.jpg"]
    assert dst["fjp_sides"] == {"Q_7_1r.jpg": "v"}
    assert dst["fjp_corrections"] == [{"image": "Q_7_1r.jpg", "from": "Testville_LIB_Q_7", "basis": BASIS_SLIP_READ,
                                       "source": SOURCE_SLIP_RELABEL}]
    assert records["Testville_LIB_Q_7"]["images"]["fjp_removed"] == [
        {"image": "Q_7_1r.jpg", "to": "Testville_LIB_Q_6", "to_shelfmark_text": "Q 6", "basis": BASIS_SLIP_READ,
         "source": SOURCE_SLIP_RELABEL}]
    assert (stats["applied"], stats["moved"], stats["changed_records"]) == (1, 1, 2)


def test_move_of_a_url_entry_keeps_the_url_and_sets_preferred_source_on_an_imageless_target():
    records = {
        "Testville_LIB_R_1": _record("Testville_LIB_R_1", [f"{BASE}/91_image.jpg"]),
        "Testville_LIB_R_2": _record("Testville_LIB_R_2"),
    }
    assert records["Testville_LIB_R_2"]["images"]["preferred_source"] is None
    apply_image_corrections(records, _corrections(_correction(
        image="91_image.jpg", url=f"{BASE}/91_image.jpg", sha256=None, from_canonical_id="Testville_LIB_R_1",
        to_canonical_id="Testville_LIB_R_2", side=None)))
    assert records["Testville_LIB_R_2"]["images"]["fjp"] == [f"{BASE}/91_image.jpg"]
    assert records["Testville_LIB_R_2"]["images"]["preferred_source"] == "fjp"
    assert "fjp_sides" not in records["Testville_LIB_R_2"]["images"]
    assert records["Testville_LIB_R_1"]["images"]["fjp"] == []
    assert records["Testville_LIB_R_1"]["images"]["preferred_source"] is None


def test_move_into_a_record_with_ktiv_scans_keeps_ktiv_preferred_and_does_not_duplicate():
    records = {
        "Testville_LIB_S_1": _record("Testville_LIB_S_1", ["S_1_1r.jpg"]),
        "Testville_LIB_S_2": _record("Testville_LIB_S_2", ["S_1_1r.jpg"], ktiv_images=["K/1/a.jpg"]),
    }
    apply_image_corrections(records, _corrections(_correction(
        image="S_1_1r.jpg", url=None, from_canonical_id="Testville_LIB_S_1", to_canonical_id="Testville_LIB_S_2")))
    assert records["Testville_LIB_S_2"]["images"]["fjp"] == ["S_1_1r.jpg"]
    assert records["Testville_LIB_S_2"]["images"]["preferred_source"] == "ktiv"


def test_detach_side_only_revoke_and_misses():
    records = {
        "Testville_LIB_T_1": _record("Testville_LIB_T_1", ["T_1_1r.jpg", "T_1_1v.jpg"]),
        "Testville_LIB_T_2": _record("Testville_LIB_T_2", ["T_2_1r.jpg", "T_2_1v.jpg"]),
        "Testville_LIB_T_3": _record("Testville_LIB_T_3", ["T_3_1r.jpg"]),
    }
    before = copy.deepcopy(records)
    corrections = _corrections(
        _correction(image="T_1_1r.jpg", url=None, from_canonical_id="Testville_LIB_T_1", to_canonical_id=None,
                    to_shelfmark_text="LIB T 40", side=None),
        _correction(image="T_2_1v.jpg", url=None, from_canonical_id="Testville_LIB_T_2",
                    to_canonical_id="Testville_LIB_T_2", side="r"),
        _correction(image="T_3_1r.jpg", url=None, from_canonical_id="Testville_LIB_T_3",
                    to_canonical_id="Testville_LIB_T_3", side=None),
        _correction(image="U_1_1r.jpg", url=None, from_canonical_id="Testville_LIB_U_1"),
        _correction(image="T_2_9r.jpg", url=None, from_canonical_id="Testville_LIB_T_2"),
        _correction(image="T_2_1r.jpg", url=None, from_canonical_id="Testville_LIB_T_2",
                    to_canonical_id="Testville_LIB_NOWHERE"),
    )
    stats = apply_image_corrections(records, corrections)
    t1 = records["Testville_LIB_T_1"]["images"]
    assert t1["fjp"] == ["T_1_1v.jpg"]
    assert t1["fjp_removed"] == [{"image": "T_1_1r.jpg", "to": None, "to_shelfmark_text": "LIB T 40",
                                  "basis": BASIS_SLIP_READ, "source": SOURCE_SLIP_RELABEL}]
    t2 = records["Testville_LIB_T_2"]["images"]
    assert t2["fjp"] == ["T_2_1r.jpg", "T_2_1v.jpg"] and t2["fjp_sides"] == {"T_2_1v.jpg": "r"}
    assert records["Testville_LIB_T_3"] == before["Testville_LIB_T_3"]
    assert {k: stats[k] for k in ("applied", "moved", "detached", "side_only", "revoked", "unknown_source_record",
                                  "unknown_target_record", "image_not_in_source")} == {
        "applied": 2, "moved": 0, "detached": 1, "side_only": 1, "revoked": 1, "unknown_source_record": 1,
        "unknown_target_record": 1, "image_not_in_source": 1}
    assert stats["changed_ids"] == ["Testville_LIB_T_1", "Testville_LIB_T_2"]


def test_the_index_document_serves_the_moved_image_on_the_target_only():
    records = {
        "Testville_LIB_Q_7": _record("Testville_LIB_Q_7", ["Q_7_1r.jpg", "Q_7_1v.jpg"]),
        "Testville_LIB_Q_6": _record("Testville_LIB_Q_6", ["Q_6_1r.jpg"]),
    }
    apply_image_corrections(records, _corrections(_correction()))
    target = GenizahDocument.from_merged_format(records["Testville_LIB_Q_6"])
    source = GenizahDocument.from_merged_format(records["Testville_LIB_Q_7"])
    assert [image_name(u) for u in target.image_urls] == ["Q_6_1r.jpg", "Q_7_1r.jpg"]
    assert [image_name(u) for u in source.image_urls] == ["Q_7_1v.jpg"]
    assert all(u.startswith("http") for u in target.image_urls)


# ───────────────────────────── transcription corrections ─────────────────────────────

_RESPONSUM = "synthetic responsum line one\nsynthetic responsum line two\nsynthetic responsum line three"
_SCRAP = "synthetic verse fragment"


def _text_record(cid: str, doc_id: str = "", transcriptions=None, fjp_extra=None) -> dict:
    """A merged record with one legacy-store entry holding *transcriptions* (no entry when *doc_id* is empty).

    :param cid: Canonical id.
    :param doc_id: The entry's ``doc_id``.
    :param transcriptions: The entry's ``transcriptions`` (dict or list).
    :param fjp_extra: Further entry fields.
    :returns: The record.
    """
    record = _record(cid)
    entries = [{"doc_id": doc_id, "transcriptions": transcriptions if transcriptions is not None else {},
                "translations": {}, "language": "Hebrew", **(fjp_extra or {})}] if doc_id else []
    record["sources"] = {"pgp": None, "fjp": entries, "ktiv": None, "ktiv_transcription": None, "bodleian": None}
    return record


def _text_correction(**overrides) -> dict:
    """A valid synthetic transcription move, with *overrides* applied.

    :param overrides: Fields to replace.
    :returns: The record.
    """
    record = {
        "kind": KIND_TRANSCRIPTION,
        "from_canonical_id": "Testville_LIB_T_10",
        "to_canonical_id": "Testville_LIB_T_11",
        "fjp_doc_id": "doc-10",
        "sections": [{"key": "2", "sha256": section_digest(_RESPONSUM)}],
        "basis": BASIS_EYE,
        "evidence": "photographs show a verse scrap",
        "reviewed": True,
        "recorded_at": "2026-10-31",
        "source": "synthetic_eye_list",
    }
    record.update(overrides)
    return record


def test_transcription_correction_schema_and_kinds():
    out = validate_correction(_text_correction(sections=[{"key": 2, "sha256": None}], fjp_doc_id=10))
    assert out["sections"] == [{"key": "2", "sha256": None}] and out["fjp_doc_id"] == "10"
    assert out["to_shelfmark_text"] is None and "image" not in out
    assert correction_kind(out) == "move"
    assert correction_kind(_text_correction(to_canonical_id=None)) == "detach"
    assert correction_kind(_text_correction(to_canonical_id="Testville_LIB_T_10")) == "revoke"


@pytest.mark.parametrize("overrides", [
    {"kind": "caption"},
    {"sections": []},
    {"sections": [{"key": ""}]},
    {"sections": [{"key": "1", "sha256": "abc"}]},
    {"sections": [{"key": "1"}, {"key": "1"}]},
    {"sections": [{"key": "1", "text": "x"}]},
    {"fjp_doc_id": ""},
    {"image": "Q_7_1r.jpg"},
    {"evidence": "x" * 61},
    {"basis": "guess"},
])
def test_transcription_validation_rejects_malformed_rows(overrides):
    with pytest.raises(ValueError):
        validate_correction(_text_correction(**overrides))


def test_transcription_sections_and_digest():
    assert transcription_sections({"1": "a", 2: "b"}) == [("1", "a"), ("2", "b")]
    assert transcription_sections([{"name": "n", "lines": {"1": "x"}}, "plain"]) == [
        ("1", {"name": "n", "lines": {"1": "x"}}), ("2", "plain")]
    assert transcription_sections(None) == [] and transcription_sections("") == []
    assert section_digest("abc") == hashlib.sha256(b"abc").hexdigest()
    assert section_digest({"b": 1, "a": "x"}) == section_digest({"a": "x", "b": 1})


def test_both_kinds_share_the_file_and_each_loader_reads_its_own(tmp_path):
    path = tmp_path / "c.jsonl"
    assert append_image_corrections(str(path), [_correction(), _text_correction(),
                                                _text_correction(evidence="same correction again")]) == 2
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert "kind" not in lines[0] and lines[1]["kind"] == KIND_TRANSCRIPTION      # image lines keep their shape
    assert set(load_image_corrections(str(path))) == {"Q_7_1r.jpg"}
    texts = load_transcription_corrections(str(path))
    assert set(texts) == {("Testville_LIB_T_10", "doc-10", "2")}
    # A later revoke of the same section wins; a new target for it is a new line.
    revoke = _text_correction(to_canonical_id="Testville_LIB_T_10")
    assert new_corrections(str(path), [revoke, _text_correction()]) == [validate_correction(revoke)]
    append_image_corrections(str(path), [revoke])
    assert correction_kind(load_transcription_corrections(str(path))[("Testville_LIB_T_10", "doc-10", "2")]) == \
        "revoke"
    assert len(list(iter_corrections(str(path)))) == 3


def test_build_transcription_correction_fills_digests_from_the_record():
    record = _text_record("Testville_LIB_T_10", "doc-10", {"1": _SCRAP, "2": _RESPONSUM})
    built = build_transcription_correction(record, "doc-10", ["2"], "Testville_LIB_T_11", basis=BASIS_EYE,
                                           evidence="photographs show a verse scrap", reviewed=True,
                                           source="synthetic_eye_list", recorded_at="2026-10-31")
    assert built == validate_correction(_text_correction())
    with pytest.raises(KeyError):
        build_transcription_correction(record, "doc-10", ["9"], None, basis=BASIS_EYE, evidence="", reviewed=True,
                                       source="s")
    with pytest.raises(KeyError):
        build_transcription_correction(record, "doc-99", ["1"], None, basis=BASIS_EYE, evidence="", reviewed=True,
                                       source="s")


def _text_corrections(*records) -> dict:
    """Key validated transcription corrections per section, as the loader does.

    :param records: Correction records.
    :returns: ``{(from, fjp_doc_id, key): record}``.
    """
    out = {}
    for record in records:
        valid = validate_correction(record)
        for section in valid["sections"]:
            out[(valid["from_canonical_id"], valid["fjp_doc_id"], section["key"])] = valid
    return out


def test_a_moved_section_leaves_the_source_and_lands_in_a_container_on_the_target():
    shared_raw = {"doc_id": "doc-10", "transcriptions": {"1": _SCRAP, "2": _RESPONSUM}, "translations": {},
                  "language": "Hebrew", "date": {"standard_date": "1100"}}
    source = _text_record("Testville_LIB_T_10")
    source["sources"]["fjp"] = [shared_raw]
    join_partner = _text_record("Testville_LIB_T_12")
    join_partner["sources"]["fjp"] = [shared_raw]            # same raw dict, as the merge's join explosion does
    records = {"Testville_LIB_T_10": source, "Testville_LIB_T_11": _text_record("Testville_LIB_T_11")}
    stats = apply_transcription_corrections(records, _text_corrections(_text_correction()))
    assert (stats["moved"], stats["applied"], stats["changed_ids"]) == (
        1, 1, ["Testville_LIB_T_10", "Testville_LIB_T_11"])
    assert source["sources"]["fjp"][0]["transcriptions"] == {"1": _SCRAP}
    assert source["sources"]["fjp"][0]["date"] == {"standard_date": "1100"}       # other fields kept
    assert shared_raw["transcriptions"] == {"1": _SCRAP, "2": _RESPONSUM}           # raw payload untouched
    assert join_partner["sources"]["fjp"][0]["transcriptions"] == {"1": _SCRAP, "2": _RESPONSUM}
    target = records["Testville_LIB_T_11"]
    assert target["sources"]["fjp"] == [
        {"doc_id": "doc-10", "transcriptions": {"2": _RESPONSUM}, MOVED_FROM_KEY: "Testville_LIB_T_10"}]
    provenance = {"fjp_doc_id": "doc-10", "section": "2", "sha256": section_digest(_RESPONSUM), "basis": BASIS_EYE,
                  "source": "synthetic_eye_list"}
    assert source["transcriptions_corrections"] == [{**provenance, "direction": "out", "to": "Testville_LIB_T_11"}]
    assert target["transcriptions_corrections"] == [{**provenance, "direction": "in", "from": "Testville_LIB_T_10"}]


def test_list_shaped_sections_move_by_position_and_detach_keeps_the_text():
    first = {"name": "Editor: A", "lines": {"1": "synthetic first"}}
    second = {"name": "Editor: B", "lines": {"1": "synthetic second"}}
    third = {"name": "Editor: C", "lines": {"1": "synthetic third"}}
    records = {
        "Testville_LIB_L_1": _text_record("Testville_LIB_L_1", "doc-L", [first, second, third]),
        "Testville_LIB_L_2": _text_record("Testville_LIB_L_2", "doc-L2", {"1": "own text"}),
    }
    corrections = _text_corrections(
        _text_correction(from_canonical_id="Testville_LIB_L_1", to_canonical_id="Testville_LIB_L_2",
                         fjp_doc_id="doc-L", sections=[{"key": "1", "sha256": section_digest(first)},
                                                       {"key": "3", "sha256": None}]),
        _text_correction(from_canonical_id="Testville_LIB_L_1", to_canonical_id=None, to_shelfmark_text="LIB L 40",
                         fjp_doc_id="doc-L", sections=[{"key": "2", "sha256": section_digest(second)}]),
    )
    stats = apply_transcription_corrections(records, corrections)
    assert (stats["moved"], stats["detached"]) == (2, 1)
    source, target = records["Testville_LIB_L_1"], records["Testville_LIB_L_2"]
    assert source["sources"]["fjp"][0]["transcriptions"] == []
    assert target["sources"]["fjp"][0]["transcriptions"] == {"1": "own text"}       # its own entry untouched
    assert target["sources"]["fjp"][1]["transcriptions"] == [first, third]
    assert source["transcriptions_removed"] == [{
        "fjp_doc_id": "doc-L", "section": "2", "sha256": section_digest(second), "basis": BASIS_EYE,
        "source": "synthetic_eye_list", "to_shelfmark_text": "LIB L 40", "value": second}]


def test_transcription_revoke_and_misses_change_nothing():
    records = {
        "Testville_LIB_T_10": _text_record("Testville_LIB_T_10", "doc-10", {"1": _SCRAP, "2": _RESPONSUM}),
        "Testville_LIB_T_11": _text_record("Testville_LIB_T_11"),
    }
    before = copy.deepcopy(records)
    cases = [
        _text_correction(to_canonical_id="Testville_LIB_T_10"),                                # revoke
        _text_correction(sections=[{"key": "1", "sha256": section_digest("re-scraped text")}]),  # section_changed
        _text_correction(sections=[{"key": "7", "sha256": None}]),                             # section missing
        _text_correction(fjp_doc_id="doc-other"),                                              # entry missing
        _text_correction(from_canonical_id="Testville_LIB_NOWHERE"),                           # unknown source
        _text_correction(to_canonical_id="Testville_LIB_NOWHERE", sections=[{"key": "9", "sha256": None}]),
    ]
    corrections = {}
    for case in cases:
        corrections.update(_text_corrections(case))
    assert len(corrections) == len(cases)
    stats = apply_transcription_corrections(records, corrections)
    assert records == before
    assert {k: stats[k] for k in ("applied", "revoked", "section_changed", "section_not_in_source",
                                  "fjp_record_not_in_source", "unknown_source_record", "unknown_target_record")} == {
        "applied": 0, "revoked": 1, "section_changed": 1, "section_not_in_source": 1, "fjp_record_not_in_source": 1,
        "unknown_source_record": 1, "unknown_target_record": 1}


def test_the_index_document_serves_a_moved_section_on_the_target_only():
    records = {
        "Testville_LIB_T_10": _text_record("Testville_LIB_T_10", "doc-10", {"2": _RESPONSUM}),
        "Testville_LIB_T_11": _text_record("Testville_LIB_T_11"),
    }
    records["Testville_LIB_T_11"]["sources"]["ktiv"] = {"basic_catalog": {"language": "Hebrew"}}
    before_target = GenizahDocument.from_merged_format(copy.deepcopy(records["Testville_LIB_T_11"]))
    apply_transcription_corrections(records, _text_corrections(_text_correction()))
    target_doc = GenizahDocument.from_merged_format(records["Testville_LIB_T_11"])
    source_doc = GenizahDocument.from_merged_format(records["Testville_LIB_T_10"])
    assert [t.name for t in target_doc.transcriptions] == ["Transcription 2"]
    assert list(target_doc.transcriptions[0].lines.values())[-1] == "synthetic responsum line three"
    assert source_doc.transcriptions == []
    target_es = target_doc.to_elasticsearch_document()
    source_es = source_doc.to_elasticsearch_document()
    assert target_es["has_transcriptions"] is True and "responsum line two" in target_es["transcriptions"][0]["text"]
    assert "responsum" in target_es["full_text_content"]
    assert source_es["has_transcriptions"] is False and source_es["transcriptions"] == []
    assert "responsum" not in source_es["full_text_content"]
    # The container brings no catalogue fields of its own: the target's language and date are unchanged.
    assert (target_doc.language, target_doc.date) == (before_target.language, before_target.date)
