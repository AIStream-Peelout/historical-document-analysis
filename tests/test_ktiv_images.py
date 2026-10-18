"""Tests for KTIV image source resolution (zip archives + IIIF-fallback folders).

The scraper leaves images locally in two forms: NLI bulk-download zips, and
loose files in ``*_images/`` folders when the zip path failed. Both must yield
identical bucket object paths, zips must win when usable, and Chrome's
``name(1).jpg`` re-download copies must collapse onto one object each.
"""

import os
import zipfile

from src.datasets.merging.ktiv_images import (
    _dedupe_uniquified,
    gcs_object_path,
    image_folders,
    ktiv_image_manifest,
    ktiv_image_sources,
)

SYS_A = "990000000000000001"
SYS_B = "990000000000000002"


def _make_zip(path, members):
    """Create a zip at *path* whose *members* hold one byte each.

    :param path: Destination zip path.
    :param members: Member names to create.
    """
    with zipfile.ZipFile(path, "w") as zf:
        for name in members:
            zf.writestr(name, b"x")


def _make_folder(root, name, files):
    """Create an image folder with one-byte *files* and return its path.

    :param root: Parent directory.
    :param name: Folder name.
    :param files: Filenames to create.
    """
    folder = root / name
    folder.mkdir()
    for fname in files:
        (folder / fname).write_bytes(b"x")
    return str(folder)


# ── uniquify dedupe ───────────────────────────────────────────────────────────

def test_dedupe_prefers_base_copy_over_uniquified():
    picked = _dedupe_uniquified(["a_1r.jpg", "a_1r(1).jpg", "a_1r (2).jpg"])
    assert picked == [("a_1r.jpg", "a_1r.jpg")]


def test_dedupe_keeps_orphan_uniquified_copy_under_canonical_name():
    # Only a "(1)" copy exists: the object path drops the suffix, but the
    # on-disk source name keeps it so the uploader can find the file.
    assert _dedupe_uniquified(["a_1r(1).jpg"]) == [("a_1r.jpg", "a_1r(1).jpg")]


def test_dedupe_distinct_images_all_survive():
    picked = _dedupe_uniquified(["b_1v.jpg", "a_1r.jpg", "a_1r(1).jpg"])
    assert picked == [("a_1r.jpg", "a_1r.jpg"), ("b_1v.jpg", "b_1v.jpg")]


# ── source resolution ─────────────────────────────────────────────────────────

def test_zip_wins_over_folder_when_it_has_images(tmp_path):
    zp = tmp_path / f"ktiv_PNX_MANUSCRIPTS{SYS_A}-1_images.zip"
    _make_zip(zp, ["0001.jpg", "0000_header.pdf"])
    folder = _make_folder(
        tmp_path, f"ktiv_PNX_MANUSCRIPTS{SYS_A}-1_images", ["001_other.jpg"]
    )
    sources = ktiv_image_sources([str(zp)], [folder])
    assert sources[SYS_A].kind == "zip"
    assert [op for op, _ in sources[SYS_A].entries] == [
        gcs_object_path(SYS_A, "0001.jpg")
    ]


def test_folder_covers_manuscript_with_empty_zip(tmp_path):
    # The NLI download service sometimes serves an image-less archive; the
    # IIIF-fallback folder must then supply the pointers.
    zp = tmp_path / f"ktiv_PNX_MANUSCRIPTS{SYS_A}-1_images.zip"
    _make_zip(zp, ["0000_header.pdf"])
    folder = _make_folder(
        tmp_path,
        f"ktiv_PNX_MANUSCRIPTS{SYS_A}-1_images",
        ["001_Frag._001r.jpg", "001_Frag._001r(1).jpg", "002_Frag._001v.jpg"],
    )
    sources = ktiv_image_sources([str(zp)], [folder])
    assert sources[SYS_A].kind == "folder"
    assert [op for op, _ in sources[SYS_A].entries] == [
        gcs_object_path(SYS_A, "001_Frag._001r.jpg"),
        gcs_object_path(SYS_A, "002_Frag._001v.jpg"),
    ]


def test_folder_only_manuscript_gets_pointers(tmp_path):
    folder = _make_folder(
        tmp_path, f"PNX_MANUSCRIPTS{SYS_B}-1_IE123", ["001.jpg", "notes.txt"]
    )
    manifest = ktiv_image_manifest([], [folder])
    assert manifest == {SYS_B: [gcs_object_path(SYS_B, "001.jpg")]}


def test_richest_folder_wins_per_manuscript(tmp_path):
    small = _make_folder(
        tmp_path, f"ktiv_PNX_MANUSCRIPTS{SYS_A}-1_images", ["001.jpg"]
    )
    big = _make_folder(
        tmp_path, f"PNX_MANUSCRIPTS{SYS_A}-1_IE9", ["001.jpg", "002.jpg"]
    )
    sources = ktiv_image_sources([], [small, big])
    assert sources[SYS_A].container == big
    assert len(sources[SYS_A].entries) == 2


def test_folder_without_sysnum_or_images_ignored(tmp_path):
    no_digits = _make_folder(tmp_path, "Caffeine.app", ["icon.png"])
    imageless = _make_folder(
        tmp_path, f"ktiv_PNX_MANUSCRIPTS{SYS_A}-1_images", ["header.pdf"]
    )
    assert ktiv_image_sources([], [no_digits, imageless]) == {}


def test_image_folders_lists_only_directories(tmp_path):
    (tmp_path / "loose.jpg").write_bytes(b"x")
    folder = _make_folder(tmp_path, f"ktiv_PNX_MANUSCRIPTS{SYS_A}-1_images", [])
    assert image_folders(str(tmp_path)) == [folder]
    assert image_folders(str(tmp_path / "does-not-exist")) == []
