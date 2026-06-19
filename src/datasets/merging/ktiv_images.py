#!/usr/bin/env python3
"""KTIV image archives -> GCS object pointers.

KTIV scan images arrive as per-manuscript zip archives
(``ktiv_PNX_MANUSCRIPTS{sys_num}-1_images.zip``) containing numbered fragment
images plus a ``header.pdf`` cover page. This module reads those archives WITHOUT
extracting to disk and produces, per manuscript ``sys_num``, the ordered list of
GCS object paths the images will live at once uploaded:

    KTIV/<sys_num>/<original_member_name>

paths are relative to the bucket base (``cairo-genizah-es-json``), matching how
FJP images are stored (the web app prepends the bucket URL). ``sys_num`` is
immutable, so these object paths are stable even if the canonical id changes.

Used by:
* the merge, to populate ``images.ktiv`` with real pointers + a ``populated`` flag;
* :mod:`upload_ktiv_images`, which streams the same members to GCS.
"""

from __future__ import annotations

import os
import re
import zipfile
from typing import Dict, List

# Bucket layout (shared with FJP, which lives under "images/").
GCS_BUCKET = "cairo-genizah-es-json"
GCS_BASE_URL = f"https://storage.googleapis.com/{GCS_BUCKET}"
KTIV_PREFIX = "KTIV"

_SYSNUM_RE = re.compile(r"(\d{15,})")
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def gcs_object_path(sys_num: str, member_name: str) -> str:
    """Return the bucket-relative GCS object path for a KTIV image.

    :param sys_num: Manuscript system number.
    :param member_name: Image filename inside the zip (basename only).
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
    """Return True if a zip member is a fragment image (not the header PDF).

    :param name: Zip member name.
    :returns: True for supported image extensions.
    """
    return name.lower().endswith(_IMAGE_EXTS)


def _pick_zip_per_sysnum(zip_paths: List[str]) -> Dict[str, str]:
    """Choose one archive per ``sys_num`` (the base download, not ``(N)`` copies).

    :param zip_paths: Paths to KTIV ``*.zip`` archives.
    :returns: ``sys_num -> chosen zip path``.
    """
    by_sysnum: Dict[str, List[str]] = {}
    for zp in zip_paths:
        m = _SYSNUM_RE.search(os.path.basename(zp))
        if m:
            by_sysnum.setdefault(m.group(1), []).append(zp)
    chosen: Dict[str, str] = {}
    for sysnum, paths in by_sysnum.items():
        # Prefer the base archive (no "(1)" suffix); fall back to the first sorted.
        base = [p for p in paths if "(" not in os.path.basename(p)]
        chosen[sysnum] = sorted(base or paths)[0]
    return chosen


def ktiv_image_manifest(zip_paths: List[str]) -> Dict[str, List[str]]:
    """Map each manuscript ``sys_num`` to its ordered GCS image object paths.

    Reads the chosen archive per manuscript and lists its image members (the
    ``header.pdf`` and any non-image members are skipped), preserving archive
    order.

    :param zip_paths: Paths to KTIV ``*.zip`` archives.
    :returns: ``sys_num -> [bucket-relative object paths]``. A manuscript with no
        image members is omitted.
    """
    manifest: Dict[str, List[str]] = {}
    for sys_num, zp in _pick_zip_per_sysnum(zip_paths).items():
        try:
            with zipfile.ZipFile(zp) as zf:
                members = [n for n in zf.namelist() if _is_image(n)]
        except (zipfile.BadZipFile, OSError):
            continue
        if members:
            manifest[sys_num] = [gcs_object_path(sys_num, n) for n in sorted(members)]
    return manifest
