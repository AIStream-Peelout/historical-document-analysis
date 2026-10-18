"""Tests for the KTIV zip mover (local scrape folder -> NAS bucket layout).

Every test builds a synthetic "local" scrape folder and a synthetic "NAS" root
under ``tmp_path``; nothing touches the real scrape folder or the real share.
The mount guard is disabled in ``run`` tests because both fake roots share a
filesystem (the guard itself is tested separately).
"""

from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from src.datasets.tools import ktiv_zip_mover as mover

SYS_A = "990051000000205171"
SYS_B = "990052000000205171"
SYS_C = "990053000000205171"

Usage = collections.namedtuple("Usage", "total used free")


def _zip_name(sysnum: str, tail: str = "") -> str:
    """Return a KTIV image-zip filename.

    :param sysnum: Manuscript sysnum.
    :param tail: Suffix after ``_images`` (e.g. ``(1)`` for a re-download).
    :return: The filename.
    """
    return f"ktiv_PNX_MANUSCRIPTS{sysnum}-1_images{tail}.zip"


def _make_zip(path: Path, n_bytes: int = 2000) -> Path:
    """Write a valid zip with one random stored member of ``n_bytes``.

    :param path: Destination path (parents are created).
    :param n_bytes: Member payload size.
    :return: ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0001_FL100000001.jpg", os.urandom(n_bytes))
    return path


def _make_crc_corrupt_zip(path: Path) -> Path:
    """Write a zip whose member bytes no longer match the stored CRC.

    :param path: Destination path.
    :return: ``path``.
    """
    payload = b"GENIZAH-PAGE-" * 100
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0001_FL100000002.jpg", payload)
    data = path.read_bytes()
    at = data.index(payload)
    path.write_bytes(data[:at] + b"X" + data[at + 1:])
    return path


def _age(path: Path, hours: float) -> Path:
    """Backdate a file's mtime (and atime) by ``hours``.

    :param path: File to backdate.
    :param hours: Age to give it.
    :return: ``path``.
    """
    then = time.time() - hours * 3600
    os.utime(path, (then, then), follow_symlinks=False)
    return path


def _dirs(tmp_path: Path) -> Tuple[Path, Path]:
    """Create the fake local scrape folder and return it with a (not yet created) NAS root.

    :param tmp_path: Pytest temp dir.
    :return: ``(local_dir, nas_root)``.
    """
    local = tmp_path / "ktiv"
    local.mkdir()
    return local, tmp_path / "nas" / "ktiv_zips"


def _cfg(local: Path, nas: Path, **overrides: object) -> mover.MoverConfig:
    """Build a test config: mount guard off, manifest outside the NAS root, 1 h min age.

    :param local: Local scrape folder.
    :param nas: NAS root.
    :param overrides: Field overrides.
    :return: The config.
    """
    base: Dict[str, object] = dict(
        local_dir=local, nas_root=nas, manifest=local.parent / "manifest.jsonl",
        min_age_hours=1.0, nas_min_free_gb=0.0, check_mount=False,
    )
    base.update(overrides)
    return mover.MoverConfig(**base)


def _snapshot(root: Path) -> Dict[str, Tuple[int, int]]:
    """Record every path under ``root`` with its size and mtime.

    :param root: Tree to record.
    :return: ``relative path -> (size, mtime_ns)``.
    """
    return {
        str(p.relative_to(root)): (p.lstat().st_size, p.lstat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
    }


def _read_manifest(path: Path) -> List[dict]:
    """Parse a JSONL manifest.

    :param path: Manifest file.
    :return: One dict per line.
    """
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ─────────────────────────── names, layout, selection ───────────────────────


def test_zip_sysnum_and_destination(tmp_path: Path) -> None:
    """Image-zip names (incl. ``(N)`` re-downloads) parse; partials, JSON and others do not."""
    assert mover.zip_sysnum(_zip_name(SYS_A)) == SYS_A
    assert mover.zip_sysnum(_zip_name(SYS_A, "(3)")) == SYS_A
    assert mover.zip_sysnum(_zip_name(SYS_A, " (3)")) == SYS_A
    assert mover.zip_sysnum(_zip_name(SYS_A) + ".part") is None
    assert mover.zip_sysnum(f"ktiv_PNX_MANUSCRIPTS{SYS_A}-1_transcription.json") is None
    assert mover.zip_sysnum(f"PNX_MANUSCRIPTS{SYS_A}-1_IE12345.zip") is None
    assert mover.nas_destination(tmp_path, _zip_name(SYS_A, "(1)")) == tmp_path / "990051" / _zip_name(SYS_A, "(1)")
    with pytest.raises(ValueError):
        mover.nas_destination(tmp_path, "notes.zip")


def test_scan_candidates_filters_and_orders(tmp_path: Path) -> None:
    """Only settled regular image zips are selected, oldest first; everything else is counted or ignored."""
    local, _ = _dirs(tmp_path)
    newer = _age(_make_zip(local / _zip_name(SYS_A)), 30)
    older = _age(_make_zip(local / _zip_name(SYS_B, "(1)")), 50)
    _make_zip(local / _zip_name(SYS_C))  # too young (just written)
    _age(_make_zip(local / (_zip_name(SYS_C, "(2)") + ".part")), 50)  # partial download
    _age(_make_zip(local / "Caffeine_1.6.4.zip"), 50)  # unrelated zip
    (local / f"ktiv_PNX_MANUSCRIPTS{SYS_A}-1_bundle.json").write_text("{}")
    (local / f"ktiv_PNX_MANUSCRIPTS{SYS_B}-1_images").mkdir()  # extracted folder
    target = _age(_make_zip(tmp_path / "elsewhere" / _zip_name(SYS_C, "(5)")), 50)
    (local / _zip_name(SYS_C, "(5)")).symlink_to(target)

    selection = mover.scan_candidates(local, min_age_hours=24)

    assert [c.path for c in selection.candidates] == [older, newer]
    assert [c.sysnum for c in selection.candidates] == [SYS_B, SYS_A]
    assert selection.skipped == {"too-young": 1, "partial-download": 1, "symlink": 1}
    assert selection.zips_seen == 4


def test_scan_candidates_require_manifest(tmp_path: Path) -> None:
    """With a required-sysnum set, other sysnums are held back."""
    local, _ = _dirs(tmp_path)
    _age(_make_zip(local / _zip_name(SYS_A)), 30)
    _age(_make_zip(local / _zip_name(SYS_B)), 30)
    selection = mover.scan_candidates(local, 24, required_sysnums={SYS_A})
    assert [c.sysnum for c in selection.candidates] == [SYS_A]
    assert selection.skipped == {"not-in-required-manifest": 1}


def test_load_required_sysnums_shapes(tmp_path: Path) -> None:
    """JSON lists / keyed objects / ``sys_nums`` lists and JSONL records all yield sysnums."""
    as_list = tmp_path / "a.json"
    as_list.write_text(json.dumps([SYS_A, int(SYS_B), {"sys_num": SYS_C}, "no id here"]))
    keyed = tmp_path / "b.json"
    keyed.write_text(json.dumps({SYS_A: {"pages": 3}, SYS_B: {"pages": 1}}))
    listed = tmp_path / "c.json"
    listed.write_text(json.dumps({"sys_nums": [SYS_C], "bucket": "cairo-genizah-es-json"}))
    lines = tmp_path / "d.jsonl"
    lines.write_text("\n".join([
        json.dumps({"sysnum": SYS_A, "fl": "FL1"}),
        json.dumps(f"KTIV/{SYS_B}/0001_FL100000001.jpg"),
        "",
        json.dumps(_zip_name(SYS_C, "(1)")),
    ]))
    assert mover.load_required_sysnums(as_list) == {SYS_A, SYS_B, SYS_C}
    assert mover.load_required_sysnums(keyed) == {SYS_A, SYS_B}
    assert mover.load_required_sysnums(listed) == {SYS_C}
    assert mover.load_required_sysnums(lines) == {SYS_A, SYS_B, SYS_C}


# ───────────────────────────── integrity, copy, dest ────────────────────────


def test_check_zip_integrity(tmp_path: Path) -> None:
    """A good zip passes; a CRC-corrupt and a truncated zip fail with a reason."""
    good = _make_zip(tmp_path / "good.zip")
    crc_bad = _make_crc_corrupt_zip(tmp_path / "crc.zip")
    truncated = tmp_path / "trunc.zip"
    truncated.write_bytes(good.read_bytes()[:500])

    assert mover.check_zip_integrity(good) == (True, "")
    ok, detail = mover.check_zip_integrity(crc_bad)
    assert not ok and "0001_FL100000002.jpg" in detail
    ok, detail = mover.check_zip_integrity(truncated)
    assert not ok and "BadZipFile" in detail


def test_copy_verified_success(tmp_path: Path) -> None:
    """The copy lands under its final name with identical bytes, source mtime and no temp."""
    src = _age(_make_zip(tmp_path / _zip_name(SYS_A)), 30)
    dst = tmp_path / "nas" / "990051" / src.name
    result = mover.copy_verified(src, dst, os.lstat(src))

    assert result.ok and result.reason == ""
    assert result.sha1 == hashlib.sha1(src.read_bytes()).hexdigest()
    assert dst.read_bytes() == src.read_bytes()
    assert int(dst.stat().st_mtime) == int(src.stat().st_mtime)
    assert not mover.temp_path_for(dst).exists()
    assert src.exists()  # copying never deletes


def test_copy_verified_mismatch_deletes_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the NAS read-back hash differs, the temp is removed and nothing reaches the final name."""
    src = _make_zip(tmp_path / _zip_name(SYS_A))
    dst = tmp_path / "nas" / "990051" / src.name
    monkeypatch.setattr(mover, "sha1_file", lambda path, chunk_bytes=0: "0" * 40)

    result = mover.copy_verified(src, dst, os.lstat(src))

    assert not result.ok and result.reason == "verify-mismatch"
    assert not dst.exists() and not mover.temp_path_for(dst).exists()
    assert src.exists()


def test_verify_copy_size_mismatch(tmp_path: Path) -> None:
    """A short NAS copy fails verification on size before any hashing."""
    src = _make_zip(tmp_path / "src.zip")
    tmp = tmp_path / ".dst.zip.moving"
    tmp.write_bytes(src.read_bytes()[:-1])
    st = os.lstat(src)
    reason, detail = mover.verify_copy(src, tmp, tmp_path / "dst.zip", st, st.st_size, "x")
    assert reason == "verify-mismatch" and "size" in detail


def test_destination_state(tmp_path: Path) -> None:
    """Absent, identical (duplicate) and different (conflict) destinations are told apart."""
    src = _make_zip(tmp_path / "src.zip")
    size = src.stat().st_size
    same = tmp_path / "same.zip"
    shutil.copyfile(src, same)
    same_size_other_bytes = tmp_path / "other.zip"
    same_size_other_bytes.write_bytes(os.urandom(size))
    shorter = tmp_path / "short.zip"
    shorter.write_bytes(b"x")

    assert mover.destination_state(src, tmp_path / "missing.zip", size) == (mover.DEST_ABSENT, None)
    state, sha1 = mover.destination_state(src, same, size)
    assert state == mover.DEST_DUPLICATE and sha1 == hashlib.sha1(src.read_bytes()).hexdigest()
    assert mover.destination_state(src, same_size_other_bytes, size)[0] == mover.DEST_CONFLICT
    assert mover.destination_state(src, shorter, size) == (mover.DEST_CONFLICT, None)


def test_manifest_record_fields(tmp_path: Path) -> None:
    """A manifest line carries the agreed fields with a UTC timestamp."""
    cand = mover.Candidate(tmp_path / _zip_name(SYS_A), SYS_A, 123, 0.0)
    manifest = tmp_path / "m" / "manifest.jsonl"
    when = dt.datetime(2026, 9, 24, 12, 0, tzinfo=dt.timezone.utc)
    mover.append_manifest(manifest, mover.manifest_record(cand, tmp_path / "d.zip", "moved", 123, "ab", when=when))

    (row,) = _read_manifest(manifest)
    assert row == {
        "sysnum": SYS_A, "filename": _zip_name(SYS_A), "size": 123, "sha1": "ab",
        "src": str(cand.path), "dst": str(tmp_path / "d.zip"),
        "moved_at": "2026-09-24T12:00:00+00:00", "action": "moved",
    }


# ─────────────────────────────── guards ────────────────────────────────────


def test_nas_has_room_and_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The free-space floor and the per-run cap are enforced, including dry-run projection."""
    monkeypatch.setattr(mover.shutil, "disk_usage", lambda path: Usage(10 * mover.GB, 0, 3 * mover.GB))
    assert mover.nas_has_room(tmp_path, mover.GB, min_free_gb=2)
    assert not mover.nas_has_room(tmp_path, mover.GB + 1, min_free_gb=2)
    assert not mover.nas_has_room(tmp_path, mover.GB, min_free_gb=2, pending_bytes=1)
    assert mover.within_cap(0, 10**15, max_gb=0)
    assert mover.within_cap(mover.GB, mover.GB, max_gb=2)
    assert not mover.within_cap(mover.GB, mover.GB + 1, max_gb=2)


def test_free_space_guard_stops_apply_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The run stops once the next copy would leave the NAS below the floor."""
    local, nas = _dirs(tmp_path)
    first = _age(_make_zip(local / _zip_name(SYS_A)), 50)
    _age(_make_zip(local / _zip_name(SYS_B)), 40)
    floor = 1_000
    start_free = floor + first.stat().st_size + 10  # room for exactly one file

    def fake_usage(path: Path) -> Usage:
        """Shrink free space by whatever has been written under the NAS root."""
        used = sum(p.stat().st_size for p in nas.rglob("*") if p.is_file()) if nas.exists() else 0
        return Usage(10**12, used, start_free - used)

    monkeypatch.setattr(mover.shutil, "disk_usage", fake_usage)
    summary = mover.run(_cfg(local, nas, apply=True, nas_min_free_gb=floor / mover.GB))

    assert summary.actions == {"moved": 1}
    assert summary.stop_reason == "nas-min-free"
    assert not first.exists() and (local / _zip_name(SYS_B)).exists()


def test_dry_run_projects_free_space(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dry run counts planned bytes against the floor although nothing is written."""
    local, nas = _dirs(tmp_path)
    first = _age(_make_zip(local / _zip_name(SYS_A)), 50)
    _age(_make_zip(local / _zip_name(SYS_B)), 40)
    free = 1_000 + first.stat().st_size + 10
    monkeypatch.setattr(mover.shutil, "disk_usage", lambda path: Usage(10**12, 0, free))

    summary = mover.run(_cfg(local, nas, nas_min_free_gb=1_000 / mover.GB))

    assert summary.actions == {"would-move": 1} and summary.stop_reason == "nas-min-free"


def test_max_gb_cap_stops_run(tmp_path: Path) -> None:
    """``max_gb`` stops the run before the copy that would exceed it."""
    local, nas = _dirs(tmp_path)
    first = _age(_make_zip(local / _zip_name(SYS_A)), 50)
    _age(_make_zip(local / _zip_name(SYS_B)), 40)
    summary = mover.run(_cfg(local, nas, apply=True, max_gb=(first.stat().st_size + 1) / mover.GB))
    assert summary.actions == {"moved": 1} and summary.stop_reason == "max-gb"


def test_mount_guard(tmp_path: Path) -> None:
    """A NAS root on the local filesystem (share not mounted) is refused."""
    local, nas = _dirs(tmp_path)
    assert not mover.nas_on_separate_device(nas, local)
    with pytest.raises(mover.MoverError):
        mover.run(_cfg(local, nas, check_mount=True))


# ─────────────────────────── dry run vs apply ──────────────────────────────


def _mixed_scrape(local: Path, nas: Path) -> Dict[str, Path]:
    """Populate a local folder covering every per-file outcome.

    :param local: Local scrape folder.
    :param nas: NAS root (a duplicate and a conflict are pre-placed there).
    :return: Named paths of interest.
    """
    good = _age(_make_zip(local / _zip_name(SYS_A)), 50)
    good_rerun = _age(_make_zip(local / _zip_name(SYS_A, "(1)")), 49)
    corrupt = _age(_make_crc_corrupt_zip(local / _zip_name(SYS_B)), 48)
    dup = _age(_make_zip(local / _zip_name(SYS_B, "(1)")), 47)
    nas_dup = nas / "990052" / dup.name
    nas_dup.parent.mkdir(parents=True)
    shutil.copyfile(dup, nas_dup)
    conflict = _age(_make_zip(local / _zip_name(SYS_C)), 46)
    nas_conflict = _make_zip(nas / "990053" / conflict.name, n_bytes=10)
    young = _make_zip(local / _zip_name(SYS_C, "(1)"))
    part = _age(_make_zip(local / (_zip_name(SYS_C, "(2)") + ".part")), 50)
    bundle = local / f"ktiv_PNX_MANUSCRIPTS{SYS_A}-1_transcription.json"
    bundle.write_text("{}")
    return dict(good=good, good_rerun=good_rerun, corrupt=corrupt, dup=dup, nas_dup=nas_dup,
                conflict=conflict, nas_conflict=nas_conflict, young=young, part=part, bundle=bundle)


def test_dry_run_creates_and_deletes_nothing(tmp_path: Path) -> None:
    """A dry run reports every outcome but leaves both trees byte-for-byte unchanged."""
    local, nas = _dirs(tmp_path)
    _mixed_scrape(local, nas)
    before = _snapshot(tmp_path)

    summary = mover.run(_cfg(local, nas))

    assert _snapshot(tmp_path) == before
    assert not (local.parent / "manifest.jsonl").exists()
    assert summary.actions == {
        "would-move": 2, "skipped:corrupt": 1, "would-remove-duplicate": 1, "skipped:dest-conflict": 1,
    }
    assert summary.selection_skips == {"too-young": 1, "partial-download": 1}
    assert summary.stop_reason == "done"


def test_dry_run_does_not_create_missing_nas_root(tmp_path: Path) -> None:
    """With no NAS tree yet, a dry run still creates no directory and no manifest."""
    local, nas = _dirs(tmp_path)
    _age(_make_zip(local / _zip_name(SYS_A)), 50)
    summary = mover.run(_cfg(local, nas, manifest=None))
    assert summary.actions == {"would-move": 1}
    assert not nas.exists() and not (tmp_path / "nas").exists()


def test_apply_moves_verifies_and_logs(tmp_path: Path) -> None:
    """``--apply`` moves good zips into buckets, removes verified duplicates, keeps the rest, logs each."""
    local, nas = _dirs(tmp_path)
    paths = _mixed_scrape(local, nas)
    good_bytes = paths["good"].read_bytes()
    conflict_bytes = paths["nas_conflict"].read_bytes()

    summary = mover.run(_cfg(local, nas, apply=True))

    moved = nas / "990051" / paths["good"].name
    assert moved.read_bytes() == good_bytes and not paths["good"].exists()
    assert (nas / "990051" / paths["good_rerun"].name).exists() and not paths["good_rerun"].exists()
    assert not paths["dup"].exists() and paths["nas_dup"].exists()
    assert paths["corrupt"].exists() and not (nas / "990052" / paths["corrupt"].name).exists()
    assert paths["conflict"].exists() and paths["nas_conflict"].read_bytes() == conflict_bytes
    for kept in ("young", "part", "bundle"):
        assert paths[kept].exists()
    assert not list(nas.rglob("*.moving"))
    assert summary.actions == {
        "moved": 2, "skipped:corrupt": 1, "duplicate-removed": 1, "skipped:dest-conflict": 1,
    }

    rows = {row["filename"]: row for row in _read_manifest(local.parent / "manifest.jsonl")}
    assert len(rows) == 5
    good_row = rows[paths["good"].name]
    assert good_row["action"] == "moved" and good_row["sysnum"] == SYS_A
    assert good_row["sha1"] == hashlib.sha1(good_bytes).hexdigest()
    assert good_row["size"] == len(good_bytes)
    assert good_row["src"] == str(paths["good"]) and good_row["dst"] == str(moved)
    assert dt.datetime.fromisoformat(good_row["moved_at"]).utcoffset() == dt.timedelta(0)
    assert rows[paths["dup"].name]["action"] == "duplicate-removed"
    assert rows[paths["corrupt"].name]["action"] == "skipped:corrupt"
    assert rows[paths["corrupt"].name]["detail"]
    assert rows[paths["conflict"].name]["action"] == "skipped:dest-conflict"


def test_apply_rerun_is_idempotent(tmp_path: Path) -> None:
    """A second ``--apply`` finds nothing new to move and changes nothing on the NAS."""
    local, nas = _dirs(tmp_path)
    _age(_make_zip(local / _zip_name(SYS_A)), 50)
    mover.run(_cfg(local, nas, apply=True))
    nas_before = _snapshot(nas)
    summary = mover.run(_cfg(local, nas, apply=True))
    assert summary.candidates == 0 and summary.processed == 0
    assert _snapshot(nas) == nas_before


def test_limit_caps_files_processed(tmp_path: Path) -> None:
    """``limit`` processes only the N oldest candidates."""
    local, nas = _dirs(tmp_path)
    oldest = _age(_make_zip(local / _zip_name(SYS_A)), 50)
    _age(_make_zip(local / _zip_name(SYS_B)), 40)
    _age(_make_zip(local / _zip_name(SYS_C)), 30)
    summary = mover.run(_cfg(local, nas, apply=True, limit=1))
    assert summary.processed == 1 and summary.stop_reason == "limit"
    assert not oldest.exists() and (nas / "990051" / oldest.name).exists()


@pytest.mark.skipif(shutil.which("lsof") is None, reason="lsof not available")
def test_open_file_is_skipped(tmp_path: Path) -> None:
    """A zip some process holds open is detected by lsof and left untouched."""
    local, nas = _dirs(tmp_path)
    held = _age(_make_zip(local / _zip_name(SYS_A)), 50)
    free = _age(_make_zip(local / _zip_name(SYS_B)), 40)
    with open(held, "rb"):
        assert mover.open_paths([held, free], shutil.which("lsof")) == {held}
        summary = mover.run(_cfg(local, nas, apply=True))
    assert summary.actions == {"skipped:open": 1, "moved": 1}
    assert held.exists() and not (nas / "990051" / held.name).exists()
    assert not free.exists()
    assert mover.open_paths([held], None) == set()


# ─────────────────────────────── main / CLI ────────────────────────────────


def test_main_dry_run_exit_code_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The CLI dry run exits 0 with a summary; a failed mount guard exits 2."""
    local, nas = _dirs(tmp_path)
    _age(_make_zip(local / _zip_name(SYS_A)), 50)
    argv = ["--local-dir", str(local), "--nas-root", str(nas), "--nas-min-free-gb", "0", "--limit", "5"]

    assert mover.main(argv) == 2  # same filesystem: looks like an unmounted share
    monkeypatch.setattr(mover, "nas_on_separate_device", lambda nas_root, local_dir: True)
    assert mover.main(argv) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "would-move=1" in out
    assert not nas.exists()


# ─────────────────────────────── zip lookup ────────────────────────────────


def test_resolve_zip_local_nas_and_miss(tmp_path: Path) -> None:
    """Lookup prefers the local folder, falls back to the NAS buckets and returns None on a miss."""
    local, nas = _dirs(tmp_path)
    local_a = _make_zip(local / _zip_name(SYS_A))
    shutil.copyfile(local_a, _make_zip(nas / "990051" / local_a.name))  # also on the NAS
    nas_b = _make_zip(nas / "990052" / _zip_name(SYS_B, "(1)"))
    nas_b_base = _make_zip(nas / "990052" / _zip_name(SYS_B))
    (nas / "990052" / (_zip_name(SYS_B, "(2)") + ".part")).write_bytes(b"partial")
    dirs = [local, nas]

    assert mover.resolve_zip(SYS_A, dirs) == local_a  # local hit wins
    assert mover.resolve_zips(SYS_A, dirs) == [local_a]  # one entry per filename
    assert mover.resolve_zip(SYS_B, dirs) == nas_b  # NAS hit
    assert mover.resolve_zips(SYS_B, dirs) == [nas_b, nas_b_base]  # every re-download, no .part
    assert mover.resolve_zip(str(Path("/any/dir") / nas_b_base.name), dirs) == nas_b_base  # by filename
    assert mover.resolve_zip(SYS_C, dirs) is None  # miss
    assert mover.resolve_zip(_zip_name(SYS_C), dirs) is None
    with pytest.raises(ValueError):
        mover.resolve_zip("not-a-sysnum", dirs)


def test_zip_search_dirs_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``KTIV_ZIP_DIRS`` (colon-separated) sets the search path; unset falls back to local then NAS."""
    local, nas = _dirs(tmp_path)
    nas_b = _make_zip(nas / "990052" / _zip_name(SYS_B))
    monkeypatch.setenv(mover.KTIV_ZIP_DIRS_ENV, f"{local}:{nas}:")
    assert mover.zip_search_dirs() == [local, nas]
    assert mover.resolve_zip(SYS_B) == nas_b
    monkeypatch.delenv(mover.KTIV_ZIP_DIRS_ENV)
    assert mover.zip_search_dirs() == [mover.DEFAULT_LOCAL_DIR, mover.DEFAULT_NAS_ROOT]
