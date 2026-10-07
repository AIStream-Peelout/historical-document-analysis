#!/usr/bin/env python3
"""KTIV image archives & folders -> GCS object pointers.

KTIV scan images arrive locally in two forms, depending on which path the
scraper extension took for a manuscript:

* **Zip archives** (``*PNX_MANUSCRIPTS{sys_num}*.zip``) from the NLI bulk
  download service, containing numbered fragment images plus a ``header.pdf``
  cover page. These are read WITHOUT extracting to disk.
* **Image folders** (``*PNX_MANUSCRIPTS{sys_num}*_images/`` and similar) — the
  extension's per-image IIIF fallback saves loose files into a folder when the
  zip download fails, and some archives have also been extracted by hand.
  Chrome's conflict handling can leave ``name(1).jpg`` duplicates next to
  ``name.jpg``; those are collapsed onto the base name.

Both forms are mapped, per manuscript ``sys_num``, to the ordered list of GCS
object paths the images live at once uploaded:

    KTIV/<sys_num>/<original_member_name>

paths are relative to the bucket base (``cairo-genizah-es-json``), matching how
FJP images are stored (the web app prepends the bucket URL). ``sys_num`` is
immutable, so these object paths are stable even if the canonical id changes.
When a manuscript has BOTH a usable zip and a folder, the zip wins (folders are
usually just its extracted copy); a folder is used whenever no zip yields
images, so IIIF-fallback manuscripts get real pointers without re-scraping.

Used by:
* the merge, to populate ``images.ktiv`` with real pointers + a ``populated`` flag;
* :mod:`upload_ktiv_images`, which streams the same members to GCS.
"""

from __future__ import annotations

import os
import re
import zipfile
from typing import Dict, List, NamedTuple, Optional, Tuple

# Bucket layout (shared with FJP, which lives under "images/").
GCS_BUCKET = "cairo-genizah-es-json"
GCS_BASE_URL = f"https://storage.googleapis.com/{GCS_BUCKET}"
KTIV_PREFIX = "KTIV"

_SYSNUM_RE = re.compile(r"(\d{15,})")
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
# Chrome download-conflict suffix on a filename stem: "name(1)" / "name (2)".
_UNIQUIFY_RE = re.compile(r"\s?\(\d+\)$")


class ManuscriptImages(NamedTuple):
    """The local image source chosen for one manuscript.

    :ivar kind: ``"zip"`` or ``"folder"``.
    :ivar container: Path to the zip archive or the image folder.
    :ivar entries: Ordered ``(object_path, source_name)`` pairs, where
        ``source_name`` is the zip member name or the on-disk filename inside
        the folder (which may carry a ``(N)`` uniquify suffix the object path
        does not).
    """

    kind: str
    container: str
    entries: List[Tuple[str, str]]


def image_folders(ktiv_dir: str) -> List[str]:
    """List candidate image folders (direct subdirectories) of *ktiv_dir*.

    Non-manuscript directories are harmless here — anything without a 15+-digit
    system number in its name is ignored downstream.

    :param ktiv_dir: Directory holding the KTIV downloads.
    :returns: Sorted list of subdirectory paths.
    """
    try:
        entries = sorted(os.listdir(ktiv_dir))
    except OSError:
        return []
    return [
        path for name in entries
        if os.path.isdir(path := os.path.join(ktiv_dir, name))
    ]


def gcs_object_path(sys_num: str, member_name: str) -> str:
    """Return the bucket-relative GCS object path for a KTIV image.

    :param sys_num: Manuscript system number.
    :param member_name: Image filename inside the zip/folder (basename only).
    :returns: ``KTIV/<sys_num>/<basename>``.
    """
    return f"{KTIV_PREFIX}/{sys_num}/{os.path.basename(member_name)}"


def gcs_url(object_path: str) -> str:
    """Return the full public URL for a bucket-relative *object_path*.

    :param object_path: Bucket-relative path (e.g. ``KTIV/990.../0001.jpg``).
    :returns: Full ``https://storage.googleapis.com/...`` URL.
    """
    return f"{GCS_BASE_URL}/{object_path}"


def _is_image(name: str) -> bool:
    """Return True if a zip member / file is a fragment image (not the header PDF).

    :param name: Zip member name or filename.
    :returns: True for supported image extensions.
    """
    return name.lower().endswith(_IMAGE_EXTS)


def _sysnum_of(path: str) -> Optional[str]:
    """Extract the manuscript ``sys_num`` from a zip/folder basename.

    :param path: Path whose basename embeds a 15+-digit system number.
    :returns: The ``sys_num`` string, or ``None`` when absent.
    """
    m = _SYSNUM_RE.search(os.path.basename(path))
    return m.group(1) if m else None


def _dedupe_uniquified(names: List[str]) -> List[Tuple[str, str]]:
    """Collapse Chrome ``name(1).jpg`` re-download copies onto their base name.

    Files are grouped by their de-uniquified name; each group keeps the exact
    base copy when present, else the first sorted copy. The de-uniquified name
    becomes the canonical (object-path) name either way, so bucket paths stay
    stable no matter how many times an image was re-downloaded.

    :param names: Image filenames inside one folder.
    :returns: Sorted ``(canonical_name, actual_filename)`` pairs.
    """
    groups: Dict[str, List[str]] = {}
    for name in names:
        stem, ext = os.path.splitext(name)
        canonical = _UNIQUIFY_RE.sub("", stem) + ext
        groups.setdefault(canonical, []).append(name)
    picked: List[Tuple[str, str]] = []
    for canonical, copies in groups.items():
        actual = canonical if canonical in copies else sorted(copies)[0]
        picked.append((canonical, actual))
    return sorted(picked)


def _folder_images(folder: str) -> List[Tuple[str, str]]:
    """List the deduplicated image files directly inside *folder*.

    :param folder: Path to an image folder.
    :returns: Sorted ``(canonical_name, actual_filename)`` pairs (may be empty).
    """
    try:
        names = [n for n in os.listdir(folder) if _is_image(n)]
    except OSError:
        return []
    return _dedupe_uniquified(names)


def _pick_zip_per_sysnum(zip_paths: List[str]) -> Dict[str, str]:
    """Choose one archive per ``sys_num`` (the base download, not ``(N)`` copies).

    :param zip_paths: Paths to KTIV ``*.zip`` archives.
    :returns: ``sys_num -> chosen zip path``.
    """
    by_sysnum: Dict[str, List[str]] = {}
    for zp in zip_paths:
        sysnum = _sysnum_of(zp)
        if sysnum:
            by_sysnum.setdefault(sysnum, []).append(zp)
    chosen: Dict[str, str] = {}
    for sysnum, paths in by_sysnum.items():
        # Prefer the base archive (no "(1)" suffix); fall back to the first sorted.
        base = [p for p in paths if "(" not in os.path.basename(p)]
        chosen[sysnum] = sorted(base or paths)[0]
    return chosen


def _pick_folder_per_sysnum(dir_paths: List[str]) -> Dict[str, str]:
    """Choose one image folder per ``sys_num`` (the one holding the most images).

    A manuscript can leave several folders behind (an extracted zip copy plus an
    IIIF-fallback batch); their file sets use different naming schemes, so they
    are not unioned — the richest single folder wins to avoid duplicate pages.

    :param dir_paths: Paths to candidate image folders.
    :returns: ``sys_num -> chosen folder path`` (folders with no images omitted).
    """
    by_sysnum: Dict[str, List[str]] = {}
    for dp in dir_paths:
        sysnum = _sysnum_of(dp)
        if sysnum and os.path.isdir(dp):
            by_sysnum.setdefault(sysnum, []).append(dp)
    chosen: Dict[str, str] = {}
    for sysnum, paths in by_sysnum.items():
        best = max(sorted(paths), key=lambda p: len(_folder_images(p)))
        if _folder_images(best):
            chosen[sysnum] = best
    return chosen


def _zip_entries(sys_num: str, zip_path: str) -> List[Tuple[str, str]]:
    """List a zip's image members as ``(object_path, member_name)`` pairs.

    :param sys_num: Manuscript system number.
    :param zip_path: Path to the archive.
    :returns: Sorted pairs; empty when the archive is unreadable or imageless.
    """
    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = [n for n in zf.namelist() if _is_image(n)]
    except (zipfile.BadZipFile, OSError):
        return []
    return [(gcs_object_path(sys_num, n), n) for n in sorted(members)]


def ktiv_image_sources(
    zip_paths: List[str],
    folder_paths: Optional[List[str]] = None,
) -> Dict[str, ManuscriptImages]:
    """Resolve the local image source for every manuscript with images on disk.

    Zips are authoritative: a manuscript whose chosen archive contains images is
    sourced from it. Folders cover the rest — manuscripts whose zip download
    failed (empty archive or none at all) but whose images were saved loose by
    the IIIF fallback. Sharing this resolution between the merge (pointer
    writing) and the uploader (object writing) keeps the two in lockstep.

    :param zip_paths: Paths to KTIV ``*.zip`` archives.
    :param folder_paths: Paths to candidate image folders, or ``None``.
    :returns: ``sys_num -> ManuscriptImages`` (manuscripts with no usable
        images in either form are omitted).
    """
    sources: Dict[str, ManuscriptImages] = {}
    for sys_num, zp in _pick_zip_per_sysnum(zip_paths).items():
        entries = _zip_entries(sys_num, zp)
        if entries:
            sources[sys_num] = ManuscriptImages("zip", zp, entries)
    for sys_num, folder in _pick_folder_per_sysnum(folder_paths or []).items():
        if sys_num in sources:
            continue
        entries = [
            (gcs_object_path(sys_num, canonical), actual)
            for canonical, actual in _folder_images(folder)
        ]
        sources[sys_num] = ManuscriptImages("folder", folder, entries)
    return sources


def ktiv_image_manifest(
    zip_paths: List[str],
    folder_paths: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Map each manuscript ``sys_num`` to its ordered GCS image object paths.

    Thin wrapper over :func:`ktiv_image_sources` for callers (the merge) that
    only need the object paths, not where they come from.

    :param zip_paths: Paths to KTIV ``*.zip`` archives.
    :param folder_paths: Paths to candidate image folders, or ``None``.
    :returns: ``sys_num -> [bucket-relative object paths]``. A manuscript with
        no usable images is omitted.
    """
    return {
        sys_num: [object_path for object_path, _ in source.entries]
        for sys_num, source in ktiv_image_sources(zip_paths, folder_paths).items()
    }
