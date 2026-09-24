#!/usr/bin/env python3
"""Move settled KTIV image zips from the local scrape folder to the NAS, verified.

The KTIV scraper (a Chrome extension) writes
``ktiv_PNX_MANUSCRIPTS<sysnum>-1_images*.zip`` archives -- next to JSON bundles
and metadata, which this tool never touches -- into
``src/datasets/raw_data/cairo_genizah/ktiv/`` at up to ~10 GB/h. This tool
relocates settled zips to the NAS share::

    /Volumes/home/studio_offload/ktiv_zips/<first BUCKET_DIGITS digits of sysnum>/<original filename>

Per file, in order (each rule is its own unit-tested function):

1. **Selection** (:func:`scan_candidates`): regular files only (never
   symlinks) named like an image zip (never ``.part``, never ``.json``), older
   than ``--min-age-hours`` by mtime and, with ``--require-manifest``, whose
   sysnum is listed there (:func:`load_required_sysnums`) -- the hook for
   "pages already uploaded to GCS".
2. **Open check** (:func:`open_paths`): ``lsof`` on the candidate paths
   (batched, just before each batch is processed); a file any process has
   open is skipped.
3. **Destination** (:func:`destination_state`): an existing NAS file with the
   same size and SHA-1 is a verified duplicate (under ``--apply`` the local
   copy is deleted); anything else there is a conflict and is never
   overwritten.
4. **Integrity** (:func:`check_zip_integrity`): ``ZipFile.testzip()`` must pass.
5. **Guards** (:func:`within_cap`, :func:`nas_has_room`): the run stops before
   it would exceed ``--max-gb`` or leave the NAS with less than
   ``--nas-min-free-gb`` free.
6. **Copy** (:func:`copy_verified`, :func:`verify_copy`): stream to
   ``.<name>.moving`` in the destination directory (SHA-1 computed on the
   fly), fsync, re-read the NAS copy and compare size + SHA-1, then rename into
   place. On mismatch the NAS temp is deleted and the file skipped, so a final
   name on the NAS only ever holds verified bytes.
7. **Delete**: the local file is removed only under ``--apply``, only after
   verification, and only if it is unchanged (inode, size, mtime) since it was
   read.
8. **Manifest** (:func:`append_manifest`): one JSON line per processed file --
   sysnum, filename, size, sha1, src, dst, moved_at (UTC), action
   (``moved`` | ``duplicate-removed`` | ``skipped:<reason>``), plus ``detail``
   for skips. Selection-level filters (too young, ``.part``, not in the
   required manifest) are only counted in the summary.

The default mode is a DRY RUN: it reads (directory listing, ``lsof``,
``testzip``, stats and hashes of existing NAS copies) but creates, writes and
deletes nothing -- not even the manifest. ``--apply`` performs the moves. A
mount guard refuses to run when the NAS root is not on a different filesystem
than the local folder (an unmounted share would otherwise resolve to the local
disk under ``/Volumes``).

Zip lookup for dataset builders
-------------------------------
:func:`resolve_zip` (first hit) and :func:`resolve_zips` (every ``(N)``
re-download copy, like the builders' current ``sorted(glob(...))``) find a
manuscript's archive by sysnum or by filename. They search the directories
named by the ``KTIV_ZIP_DIRS`` environment variable: a colon-separated list,
searched left to right, each directory checked both flat
(``<dir>/<filename>``, the scrape folder) and bucketed
(``<dir>/<bucket>/<filename>``, the NAS layout)::

    export KTIV_ZIP_DIRS="$REPO/src/datasets/raw_data/cairo_genizah/ktiv:/Volumes/home/studio_offload/ktiv_zips"

When unset, the search path is exactly that pair: the local scrape folder
first, then the NAS.

Usage::

    .venv/bin/python -m src.datasets.tools.ktiv_zip_mover --limit 20          # dry run
    .venv/bin/python -m src.datasets.tools.ktiv_zip_mover --apply --limit 20  # trial move
    .venv/bin/python -m src.datasets.tools.ktiv_zip_mover --apply --max-gb 100

Run one instance at a time. Sizes use decimal units (1 GB = 10^9 bytes). A
copy interrupted mid-file leaves only a ``.<name>.moving`` temp on the NAS,
which the next attempt overwrites and which never matches the lookup pattern.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Sequence, Set, Tuple

log = logging.getLogger("ktiv_zip_mover")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCAL_DIR = REPO_ROOT / "src" / "datasets" / "raw_data" / "cairo_genizah" / "ktiv"
DEFAULT_NAS_ROOT = Path("/Volumes/home/studio_offload/ktiv_zips")
MANIFEST_NAME = "manifest.jsonl"
KTIV_ZIP_DIRS_ENV = "KTIV_ZIP_DIRS"

#: Leading sysnum digits that name the NAS bucket directory. Shared by the
#: mover and :func:`resolve_zips`; change it only before the first ``--apply``.
#: KTIV sysnums are 18-digit NLI ids that almost all start ``99005``, so six
#: digits yields few, large buckets; nine digits spreads them far more evenly.
BUCKET_DIGITS = 6
GB = 10**9
CHUNK_BYTES = 8 * 1024 * 1024
LSOF_BATCH = 32

ZIP_NAME_RE = re.compile(r"^ktiv_PNX_MANUSCRIPTS(?P<sysnum>\d{15,})-1_images[^/]*\.zip$")
_SYSNUM_RE = re.compile(r"(?<!\d)\d{15,}(?!\d)")
_SYSNUM_KEYS = ("sys_num", "sysnum")
_SYSNUM_LIST_KEYS = ("sys_nums", "sysnums")
# Everything ``ZipFile`` / ``testzip`` raises for an archive it cannot fully read.
_ZIP_READ_ERRORS = (
    zipfile.BadZipFile, zlib.error, EOFError, NotImplementedError, RuntimeError, ValueError, OSError,
)

DEST_ABSENT = "absent"
DEST_DUPLICATE = "duplicate"
DEST_CONFLICT = "conflict"
_STOP_PREFIX = "stop:"


# ───────────────────────────── names & layout ──────────────────────────────


def zip_sysnum(name: str) -> Optional[str]:
    """Return the sysnum of a KTIV image-zip filename, or ``None`` for any other name.

    Accepts ``ktiv_PNX_MANUSCRIPTS<sysnum>-1_images.zip`` and Chrome's
    re-download variants (``..._images(1).zip``); rejects partial downloads
    (anything containing ``.part``), JSON bundles and every other file.

    :param name: File name without a directory part.
    :type name: str
    :return: The 15+ digit sysnum, or ``None`` when ``name`` is not an image zip.
    :rtype: Optional[str]
    """
    if ".part" in name.lower():
        return None
    match = ZIP_NAME_RE.match(name)
    return match.group("sysnum") if match else None


def bucket_name(sysnum: str, digits: int = BUCKET_DIGITS) -> str:
    """Return the NAS bucket directory name for a sysnum.

    :param sysnum: Manuscript system number.
    :type sysnum: str
    :param digits: Number of leading digits that name the bucket.
    :type digits: int
    :return: The first ``digits`` digits of ``sysnum``.
    :rtype: str
    """
    return sysnum[:digits]


def nas_destination(nas_root: Path, filename: str, digits: int = BUCKET_DIGITS) -> Path:
    """Return where a zip lives on the NAS: ``<nas_root>/<bucket>/<filename>``.

    :param nas_root: Root of the NAS zip store.
    :type nas_root: Path
    :param filename: Original zip filename (kept unchanged).
    :type filename: str
    :param digits: Number of leading sysnum digits that name the bucket.
    :type digits: int
    :return: Destination path.
    :rtype: Path
    :raises ValueError: When ``filename`` is not a KTIV image-zip name.
    """
    sysnum = zip_sysnum(filename)
    if sysnum is None:
        raise ValueError(f"not a KTIV image-zip name: {filename!r}")
    return nas_root / bucket_name(sysnum, digits) / filename


# ─────────────────────────────── selection ─────────────────────────────────


@dataclass(frozen=True)
class Candidate:
    """A local image zip selected for moving.

    :param path: Local path as listed (never resolved through symlinks).
    :param sysnum: Manuscript system number parsed from the filename.
    :param size: Size in bytes at selection time.
    :param mtime: Modification time (epoch seconds) at selection time.
    """

    path: Path
    sysnum: str
    size: int
    mtime: float


@dataclass
class Selection:
    """Result of scanning the local folder.

    :param candidates: Selected zips, oldest first.
    :param skipped: Counts of zip-like files left out, by reason.
    :param zips_seen: Number of image-zip names seen (selected or not).
    """

    candidates: List[Candidate] = field(default_factory=list)
    skipped: collections.Counter = field(default_factory=collections.Counter)
    zips_seen: int = 0


def is_settled(mtime: float, now: float, min_age_hours: float) -> bool:
    """Return True when a file's mtime is at least ``min_age_hours`` old.

    :param mtime: File modification time (epoch seconds).
    :type mtime: float
    :param now: Current time (epoch seconds).
    :type now: float
    :param min_age_hours: Minimum age in hours.
    :type min_age_hours: float
    :return: Whether the file is old enough to move.
    :rtype: bool
    """
    return now - mtime >= min_age_hours * 3600.0


def scan_candidates(
    local_dir: Path,
    min_age_hours: float,
    now: Optional[float] = None,
    required_sysnums: Optional[Set[str]] = None,
) -> Selection:
    """List the image zips in ``local_dir`` that are eligible to move.

    Only regular files are considered -- symlinks are skipped without being
    followed -- and only names accepted by :func:`zip_sysnum`, so ``.part``
    downloads, JSON bundles, extracted ``*_images/`` folders and unrelated
    files are never selected.

    :param local_dir: The scrape folder (not searched recursively).
    :type local_dir: Path
    :param min_age_hours: Minimum age by mtime.
    :type min_age_hours: float
    :param now: Current time (epoch seconds); defaults to ``time.time()``.
    :type now: Optional[float]
    :param required_sysnums: When given, only these sysnums are eligible.
    :type required_sysnums: Optional[Set[str]]
    :return: Candidates (oldest first) plus skip counts.
    :rtype: Selection
    """
    now = time.time() if now is None else now
    selection = Selection()
    with os.scandir(local_dir) as entries:
        for entry in entries:
            sysnum = zip_sysnum(entry.name)
            if sysnum is None:
                if entry.name.endswith(".zip.part") and "_images" in entry.name:
                    selection.skipped["partial-download"] += 1
                continue
            selection.zips_seen += 1
            if entry.is_symlink():
                selection.skipped["symlink"] += 1
            elif not entry.is_file(follow_symlinks=False):
                selection.skipped["not-a-regular-file"] += 1
            else:
                st = entry.stat(follow_symlinks=False)
                if not is_settled(st.st_mtime, now, min_age_hours):
                    selection.skipped["too-young"] += 1
                elif required_sysnums is not None and sysnum not in required_sysnums:
                    selection.skipped["not-in-required-manifest"] += 1
                else:
                    selection.candidates.append(
                        Candidate(Path(entry.path), sysnum, st.st_size, st.st_mtime)
                    )
    selection.candidates.sort(key=lambda cand: (cand.mtime, cand.path.name))
    return selection


def _sysnums_in(item: object) -> Set[str]:
    """Extract the sysnum from one manifest item.

    :param item: A sysnum (str/int), a string containing one (a zip name or a
        ``KTIV/<sysnum>/...`` object path), or a dict with a ``sys_num`` /
        ``sysnum`` key.
    :type item: object
    :return: Zero or one sysnum.
    :rtype: Set[str]
    """
    if isinstance(item, dict):
        key = next((k for k in _SYSNUM_KEYS if k in item), None)
        return _sysnums_in(item[key]) if key is not None else set()
    if isinstance(item, (str, int)) and not isinstance(item, bool):
        match = _SYSNUM_RE.search(str(item))
        return {match.group(0)} if match else set()
    return set()


def load_required_sysnums(path: Path) -> Set[str]:
    """Load the sysnums listed in a ``--require-manifest`` file.

    ``.jsonl`` / ``.ndjson`` files hold one item per line. Other files are one
    JSON document: a list of items, an object with a ``sys_nums`` /
    ``sysnums`` list, a single record with a ``sys_num`` / ``sysnum`` key, or
    an object keyed by sysnum. An item is a sysnum, a string containing one
    (zip name, ``KTIV/<sysnum>/<member>`` object path) or a record with a
    ``sys_num`` / ``sysnum`` key.

    :param path: JSON or JSONL file.
    :type path: Path
    :return: The set of listed sysnums.
    :rtype: Set[str]
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        items: List[object] = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            list_key = next((k for k in _SYSNUM_LIST_KEYS if k in data), None)
            if list_key is not None:
                items = list(data[list_key])
            elif any(k in data for k in _SYSNUM_KEYS):
                items = [data]
            else:
                items = list(data.keys())
        else:
            items = [data]
    sysnums: Set[str] = set()
    for item in items:
        sysnums |= _sysnums_in(item)
    return sysnums


# ────────────────────────────── open-file check ────────────────────────────


def _lsof_reports_open(lsof: str, paths: Sequence[Path]) -> bool:
    """Return True when ``lsof`` finds any process holding any of ``paths`` open.

    Only the output counts: ``lsof`` exits 1 whenever ANY named file is not
    open, even while it prints the PIDs holding the others, so the exit code
    is meaningless for a batch. With ``-t`` the output is PIDs only.

    :param lsof: Path of the ``lsof`` executable.
    :type lsof: str
    :param paths: Files to check in one ``lsof`` call.
    :type paths: Sequence[Path]
    :return: Whether at least one of the files is open.
    :rtype: bool
    """
    proc = subprocess.run(
        [lsof, "-n", "-P", "-t", "--", *(str(p) for p in paths)],
        capture_output=True, text=True, check=False,
    )
    return bool(proc.stdout.strip())


def open_paths(paths: Sequence[Path], lsof: Optional[str], batch_size: int = LSOF_BATCH) -> Set[Path]:
    """Return the subset of ``paths`` that some process currently has open.

    One ``lsof`` call covers a whole batch (a call costs ~0.5 s however many
    files it names); only a batch that reports an open file is re-checked file
    by file.

    :param paths: Candidate paths.
    :type paths: Sequence[Path]
    :param lsof: Path of the ``lsof`` executable, or ``None`` when unavailable
        (then nothing is reported open).
    :type lsof: Optional[str]
    :param batch_size: Paths per ``lsof`` call.
    :type batch_size: int
    :return: Paths that are open.
    :rtype: Set[Path]
    """
    if not lsof:
        return set()
    busy: Set[Path] = set()
    for start in range(0, len(paths), batch_size):
        batch = list(paths[start:start + batch_size])
        if batch and _lsof_reports_open(lsof, batch):
            busy.update(p for p in batch if _lsof_reports_open(lsof, [p]))
    return busy


# ──────────────────────────── file I/O helpers ─────────────────────────────


def _nocache(fd: int) -> None:
    """Keep this descriptor's data out of the page cache (macOS ``F_NOCACHE``).

    Stops the NAS read-back from being answered by pages we just wrote, and
    stops a bulk move from churning the page cache of a shared machine. A
    no-op where ``F_NOCACHE`` does not exist.

    :param fd: Open file descriptor.
    :type fd: int
    """
    if hasattr(fcntl, "F_NOCACHE"):
        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)


def _open_read(path: Path) -> BinaryIO:
    """Open a file for binary reading, refusing a symlink as the final component.

    :param path: File to open.
    :type path: Path
    :return: A buffered binary reader.
    :rtype: BinaryIO
    """
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    _nocache(fd)
    return os.fdopen(fd, "rb")


def sha1_file(path: Path, chunk_bytes: int = CHUNK_BYTES) -> str:
    """Return the SHA-1 hex digest of a file, streamed in chunks.

    :param path: File to hash.
    :type path: Path
    :param chunk_bytes: Read size.
    :type chunk_bytes: int
    :return: Hex digest.
    :rtype: str
    """
    digest = hashlib.sha1(usedforsecurity=False)
    with _open_read(path) as fh:
        while chunk := fh.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _lstat_or_none(path: Path) -> Optional[os.stat_result]:
    """``os.lstat`` that returns ``None`` for a missing path.

    :param path: Path to stat (symlinks are not followed).
    :type path: Path
    :return: The stat result, or ``None`` when nothing exists there.
    :rtype: Optional[os.stat_result]
    """
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None


def same_file_state(path: Path, before: os.stat_result) -> bool:
    """Return True when ``path`` is still the same, unmodified regular file.

    :param path: Local file.
    :type path: Path
    :param before: Its ``lstat`` result from before it was read.
    :type before: os.stat_result
    :return: Whether inode, size and mtime are unchanged.
    :rtype: bool
    """
    now = _lstat_or_none(path)
    return (
        now is not None
        and stat.S_ISREG(now.st_mode)
        and (now.st_ino, now.st_size, now.st_mtime_ns) == (before.st_ino, before.st_size, before.st_mtime_ns)
    )


# ─────────────────────── destination, integrity, guards ────────────────────


def destination_state(src: Path, dst: Path, src_size: int) -> Tuple[str, Optional[str]]:
    """Classify what already sits at a zip's NAS destination.

    :param src: Local zip.
    :type src: Path
    :param dst: Its NAS destination.
    :type dst: Path
    :param src_size: Local size in bytes.
    :type src_size: int
    :return: ``(state, local sha1 or None)`` where state is
        :data:`DEST_ABSENT`, :data:`DEST_DUPLICATE` (a regular file with the
        same size and SHA-1) or :data:`DEST_CONFLICT` (anything else, never
        overwritten). The SHA-1 is only computed when the sizes match.
    :rtype: Tuple[str, Optional[str]]
    """
    dst_stat = _lstat_or_none(dst)
    if dst_stat is None:
        return DEST_ABSENT, None
    if not stat.S_ISREG(dst_stat.st_mode) or dst_stat.st_size != src_size:
        return DEST_CONFLICT, None
    src_sha1 = sha1_file(src)
    return (DEST_DUPLICATE if sha1_file(dst) == src_sha1 else DEST_CONFLICT), src_sha1


def check_zip_integrity(path: Path) -> Tuple[bool, str]:
    """Run ``ZipFile.testzip()`` over an archive (reads every member, checks CRCs).

    :param path: Zip file (a symlink is refused).
    :type path: Path
    :return: ``(ok, detail)``; ``detail`` explains a failure.
    :rtype: Tuple[bool, str]
    """
    try:
        with _open_read(path) as fh, zipfile.ZipFile(fh) as zf:
            bad_member = zf.testzip()
    except _ZIP_READ_ERRORS as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if bad_member is not None:
        return False, f"CRC mismatch in member {bad_member!r}"
    return True, ""


def existing_ancestor(path: Path) -> Path:
    """Return ``path`` or its nearest ancestor that exists.

    :param path: Possibly not-yet-created path.
    :type path: Path
    :return: The deepest existing directory on the way to ``path``.
    :rtype: Path
    """
    current = Path(os.path.abspath(path))
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def nas_on_separate_device(nas_root: Path, local_dir: Path) -> bool:
    """Return True when the NAS root is on a different filesystem than ``local_dir``.

    Guards against an unmounted share: ``/Volumes/home/...`` would then resolve
    to the local disk, and the "move" would fill the very disk it should free.

    :param nas_root: NAS zip-store root (may not exist yet).
    :type nas_root: Path
    :param local_dir: Local scrape folder.
    :type local_dir: Path
    :return: Whether the two live on different devices.
    :rtype: bool
    """
    return os.stat(existing_ancestor(nas_root)).st_dev != os.stat(local_dir).st_dev


def nas_free_bytes(nas_root: Path) -> int:
    """Return the free bytes on the filesystem holding ``nas_root``.

    :param nas_root: NAS zip-store root (may not exist yet).
    :type nas_root: Path
    :return: Free bytes.
    :rtype: int
    """
    return shutil.disk_usage(existing_ancestor(nas_root)).free


def nas_has_room(nas_root: Path, needed_bytes: int, min_free_gb: float, pending_bytes: int = 0) -> bool:
    """Return True when copying ``needed_bytes`` keeps the NAS above its free-space floor.

    :param nas_root: NAS zip-store root.
    :type nas_root: Path
    :param needed_bytes: Size of the next copy.
    :type needed_bytes: int
    :param min_free_gb: Floor in GB that must remain free afterwards.
    :type min_free_gb: float
    :param pending_bytes: Bytes planned but not written (dry-run projection).
    :type pending_bytes: int
    :return: Whether the copy may proceed.
    :rtype: bool
    """
    return nas_free_bytes(nas_root) - pending_bytes - needed_bytes >= min_free_gb * GB


def within_cap(done_bytes: int, next_bytes: int, max_gb: float) -> bool:
    """Return True when the next copy keeps the run within its ``--max-gb`` cap.

    :param done_bytes: Bytes copied (or planned) so far in this run.
    :type done_bytes: int
    :param next_bytes: Size of the next copy.
    :type next_bytes: int
    :param max_gb: Per-run cap in GB; ``0`` or less means unlimited.
    :type max_gb: float
    :return: Whether the copy may proceed.
    :rtype: bool
    """
    return max_gb <= 0 or done_bytes + next_bytes <= max_gb * GB


# ─────────────────────────────── copy & verify ─────────────────────────────


@dataclass(frozen=True)
class CopyResult:
    """Outcome of :func:`copy_verified`.

    :param ok: True when the verified copy is in place under its final name.
    :param sha1: SHA-1 of the bytes read from the local file.
    :param reason: Skip reason when not ok (``verify-mismatch``,
        ``source-changed``, ``dest-appeared``).
    :param detail: Human-readable explanation.
    """

    ok: bool
    sha1: str
    reason: str = ""
    detail: str = ""


def temp_path_for(dst: Path) -> Path:
    """Return the temp name a copy is written to before its final rename.

    :param dst: Final destination.
    :type dst: Path
    :return: ``<dst dir>/.<name>.moving`` (hidden, never matches the zip pattern).
    :rtype: Path
    """
    return dst.with_name(f".{dst.name}.moving")


def verify_copy(
    src: Path, tmp: Path, dst: Path, src_stat: os.stat_result, copied: int, src_sha1: str
) -> Tuple[str, str]:
    """Check a freshly written NAS temp copy before it is renamed into place.

    :param src: Local source file.
    :type src: Path
    :param tmp: NAS temp copy.
    :type tmp: Path
    :param dst: Final destination (must still be free).
    :type dst: Path
    :param src_stat: ``lstat`` of the source taken before the copy.
    :type src_stat: os.stat_result
    :param copied: Bytes read from the source and written.
    :type copied: int
    :param src_sha1: SHA-1 of the bytes read from the source.
    :type src_sha1: str
    :return: ``("", "")`` when verified, else ``(reason, detail)``.
    :rtype: Tuple[str, str]
    """
    if copied != src_stat.st_size or not same_file_state(src, src_stat):
        return "source-changed", f"read {copied} B of {src_stat.st_size} B, or the local file changed"
    nas_size = os.lstat(tmp).st_size
    if nas_size != copied:
        return "verify-mismatch", f"NAS size {nas_size} != local {copied}"
    nas_sha1 = sha1_file(tmp)
    if nas_sha1 != src_sha1:
        return "verify-mismatch", f"NAS sha1 {nas_sha1} != local {src_sha1}"
    if os.path.lexists(dst):
        return "dest-appeared", "the destination appeared during the copy"
    return "", ""


def copy_verified(src: Path, dst: Path, src_stat: os.stat_result) -> CopyResult:
    """Copy ``src`` to ``dst`` through a temp name, verify it, then rename it into place.

    The SHA-1 of the source is computed while streaming; the NAS temp is then
    fsynced, re-read and compared (size and SHA-1). On any mismatch the temp is
    deleted and nothing is left at ``dst``. The source mtime is carried over.

    :param src: Local zip.
    :type src: Path
    :param dst: NAS destination (its directory is created if needed).
    :type dst: Path
    :param src_stat: ``lstat`` of ``src`` taken before the copy.
    :type src_stat: os.stat_result
    :return: The copy outcome.
    :rtype: CopyResult
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = temp_path_for(dst)
    digest = hashlib.sha1(usedforsecurity=False)
    copied = 0
    out_fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o644)
    _nocache(out_fd)
    with _open_read(src) as fin, os.fdopen(out_fd, "wb") as fout:
        while chunk := fin.read(CHUNK_BYTES):
            digest.update(chunk)
            fout.write(chunk)
            copied += len(chunk)
        fout.flush()
        os.fsync(fout.fileno())
    src_sha1 = digest.hexdigest()
    reason, detail = verify_copy(src, tmp, dst, src_stat, copied, src_sha1)
    if reason:
        tmp.unlink()
        return CopyResult(False, src_sha1, reason, detail)
    os.utime(tmp, ns=(src_stat.st_atime_ns, src_stat.st_mtime_ns))
    os.rename(tmp, dst)
    return CopyResult(True, src_sha1)


# ─────────────────────────────────── manifest ──────────────────────────────


def manifest_record(
    cand: Candidate,
    dst: Path,
    action: str,
    size: int,
    sha1: Optional[str],
    detail: str = "",
    when: Optional[dt.datetime] = None,
) -> Dict[str, object]:
    """Build one manifest line.

    :param cand: The processed candidate.
    :type cand: Candidate
    :param dst: Its NAS destination.
    :type dst: Path
    :param action: ``moved``, ``duplicate-removed`` or ``skipped:<reason>``.
    :type action: str
    :param size: Size in bytes.
    :type size: int
    :param sha1: SHA-1 when computed, else ``None``.
    :type sha1: Optional[str]
    :param detail: Optional explanation (skips).
    :type detail: str
    :param when: Event time; defaults to now (UTC).
    :type when: Optional[dt.datetime]
    :return: The JSON-serialisable record.
    :rtype: Dict[str, object]
    """
    when = when or dt.datetime.now(dt.timezone.utc)
    record: Dict[str, object] = {
        "sysnum": cand.sysnum,
        "filename": cand.path.name,
        "size": size,
        "sha1": sha1,
        "src": str(cand.path),
        "dst": str(dst),
        "moved_at": when.astimezone(dt.timezone.utc).isoformat(timespec="seconds"),
        "action": action,
    }
    if detail:
        record["detail"] = detail
    return record


def append_manifest(path: Path, record: Dict[str, object]) -> None:
    """Append one JSON line to the manifest and flush it to storage.

    :param path: Manifest file (created with its directory if needed).
    :type path: Path
    :param record: Record from :func:`manifest_record`.
    :type record: Dict[str, object]
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ──────────────────────────────────── run ──────────────────────────────────


class MoverError(RuntimeError):
    """A precondition failed before anything was touched."""


@dataclass
class MoverConfig:
    """Settings for one run (mirrors the CLI flags).

    :param local_dir: Scrape folder holding the zips.
    :param nas_root: NAS zip-store root.
    :param manifest: Manifest path; ``None`` means ``<nas_root>/manifest.jsonl``.
    :param apply: Perform moves; False is a read-only dry run.
    :param min_age_hours: Minimum age by mtime.
    :param required_sysnums: When set, only these sysnums may move.
    :param nas_min_free_gb: Stop before the NAS would drop below this many GB free.
    :param max_gb: Per-run copy cap in GB (0 = unlimited).
    :param limit: Process at most this many candidates (0 = unlimited).
    :param check_mount: Refuse to run unless the NAS root is on another filesystem.
    """

    local_dir: Path = DEFAULT_LOCAL_DIR
    nas_root: Path = DEFAULT_NAS_ROOT
    manifest: Optional[Path] = None
    apply: bool = False
    min_age_hours: float = 24.0
    required_sysnums: Optional[Set[str]] = None
    nas_min_free_gb: float = 200.0
    max_gb: float = 0.0
    limit: int = 0
    check_mount: bool = True

    @property
    def manifest_path(self) -> Path:
        """Return the effective manifest path.

        :return: ``manifest`` or ``<nas_root>/manifest.jsonl``.
        :rtype: Path
        """
        return self.manifest if self.manifest is not None else self.nas_root / MANIFEST_NAME


@dataclass
class Summary:
    """Running totals and final report of a run.

    :param apply: Whether the run moved files (False = dry run).
    :param zips_seen: Image-zip names seen in the local folder.
    :param selection_skips: Zip-like files left out at selection, by reason.
    :param candidates: Number of selected candidates.
    :param candidate_bytes: Their total size.
    :param processed: Candidates that reached an action.
    :param actions: Count per action.
    :param bytes_copied: Bytes copied to the NAS (planned, in a dry run).
    :param bytes_freed: Local bytes deleted (that would be, in a dry run).
    :param nas_free_start: NAS free bytes when the run started.
    :param stop_reason: ``done``, ``limit``, ``max-gb`` or ``nas-min-free``.
    :param elapsed: Seconds spent processing.
    """

    apply: bool
    zips_seen: int = 0
    selection_skips: Dict[str, int] = field(default_factory=dict)
    candidates: int = 0
    candidate_bytes: int = 0
    processed: int = 0
    actions: collections.Counter = field(default_factory=collections.Counter)
    bytes_copied: int = 0
    bytes_freed: int = 0
    nas_free_start: int = 0
    stop_reason: str = ""
    elapsed: float = 0.0


@dataclass(frozen=True)
class Outcome:
    """What happened (or, in a dry run, would happen) to one candidate.

    :param action: ``moved``, ``duplicate-removed``, ``would-move``,
        ``would-remove-duplicate``, ``skipped:<reason>`` or ``stop:<reason>``.
    :param size: Size in bytes.
    :param sha1: SHA-1 when computed.
    :param detail: Explanation for skips.
    """

    action: str
    size: int = 0
    sha1: Optional[str] = None
    detail: str = ""

    @property
    def stops_run(self) -> bool:
        """Return True when this outcome ends the run (a guard tripped).

        :return: Whether the action is ``stop:<reason>``.
        :rtype: bool
        """
        return self.action.startswith(_STOP_PREFIX)


def process_candidate(cand: Candidate, cfg: MoverConfig, busy: Set[Path], bytes_done: int) -> Outcome:
    """Decide on one candidate and, under ``cfg.apply``, carry the move out.

    :param cand: The candidate.
    :type cand: Candidate
    :param cfg: Run settings.
    :type cfg: MoverConfig
    :param busy: Paths ``lsof`` reported open.
    :type busy: Set[Path]
    :param bytes_done: Bytes copied (or planned) so far this run.
    :type bytes_done: int
    :return: The outcome; a ``stop:*`` outcome means nothing was done to this file.
    :rtype: Outcome
    """
    if cand.path in busy:
        return Outcome("skipped:open", cand.size, detail="a process has the file open")
    src_stat = _lstat_or_none(cand.path)
    if src_stat is None or not stat.S_ISREG(src_stat.st_mode):
        return Outcome("skipped:vanished", cand.size, detail="no longer a regular file")
    size = src_stat.st_size
    dst = nas_destination(cfg.nas_root, cand.path.name)
    state, sha1 = destination_state(cand.path, dst, size)
    if state == DEST_DUPLICATE:
        if not cfg.apply:
            return Outcome("would-remove-duplicate", size, sha1)
        if not same_file_state(cand.path, src_stat):
            return Outcome("skipped:source-changed", size, sha1, "local file changed while hashing")
        os.unlink(cand.path)
        return Outcome("duplicate-removed", size, sha1)
    if state == DEST_CONFLICT:
        return Outcome("skipped:dest-conflict", size, sha1, "a different file already exists at the destination")
    ok, detail = check_zip_integrity(cand.path)
    if not ok:
        return Outcome("skipped:corrupt", size, None, detail)
    if not within_cap(bytes_done, size, cfg.max_gb):
        return Outcome(f"{_STOP_PREFIX}max-gb", size)
    if not nas_has_room(cfg.nas_root, size, cfg.nas_min_free_gb, pending_bytes=0 if cfg.apply else bytes_done):
        return Outcome(f"{_STOP_PREFIX}nas-min-free", size)
    if not cfg.apply:
        return Outcome("would-move", size)
    result = copy_verified(cand.path, dst, src_stat)
    if not result.ok:
        return Outcome(f"skipped:{result.reason}", size, result.sha1, result.detail)
    if not same_file_state(cand.path, src_stat):
        return Outcome("skipped:source-changed", size, result.sha1,
                       "local file changed after the verified copy; both copies kept")
    os.unlink(cand.path)
    return Outcome("moved", size, result.sha1)


def _record_outcome(summary: Summary, cand: Candidate, outcome: Outcome, cfg: MoverConfig) -> None:
    """Fold one outcome into the totals and, under ``--apply``, the manifest.

    :param summary: Running totals (updated in place).
    :type summary: Summary
    :param cand: The processed candidate.
    :type cand: Candidate
    :param outcome: Its outcome (not a ``stop:*``).
    :type outcome: Outcome
    :param cfg: Run settings.
    :type cfg: MoverConfig
    """
    summary.processed += 1
    summary.actions[outcome.action] += 1
    if outcome.action in ("moved", "would-move"):
        summary.bytes_copied += outcome.size
    if outcome.action in ("moved", "would-move", "duplicate-removed", "would-remove-duplicate"):
        summary.bytes_freed += outcome.size
    if cfg.apply:
        dst = nas_destination(cfg.nas_root, cand.path.name)
        append_manifest(
            cfg.manifest_path,
            manifest_record(cand, dst, outcome.action, outcome.size, outcome.sha1, outcome.detail),
        )


def _progress_line(index: int, total: int, cand: Candidate, outcome: Outcome, summary: Summary, elapsed: float) -> str:
    """Format one per-file progress line with running totals.

    :param index: 1-based position in this run.
    :type index: int
    :param total: Files this run will process at most.
    :type total: int
    :param cand: The candidate.
    :type cand: Candidate
    :param outcome: Its outcome.
    :type outcome: Outcome
    :param summary: Running totals.
    :type summary: Summary
    :param elapsed: Seconds since processing started.
    :type elapsed: float
    :return: The log line.
    :rtype: str
    """
    copied, freed = ("moved", "freed") if summary.apply else ("to move", "to free")
    rate = summary.bytes_copied / 1e6 / elapsed if elapsed > 0 else 0.0
    line = (
        f"[{index}/{total}] {outcome.action} {cand.path.name} ({outcome.size / 1e6:.1f} MB) | "
        f"{summary.processed} files, {copied} {summary.bytes_copied / GB:.2f} GB, "
        f"{freed} {summary.bytes_freed / GB:.2f} GB, {rate:.1f} MB/s"
    )
    return f"{line} | {outcome.detail}" if outcome.detail else line


def run(cfg: MoverConfig, now: Optional[float] = None) -> Summary:
    """Select candidates and process them in order (oldest first).

    :param cfg: Run settings.
    :type cfg: MoverConfig
    :param now: Current time for the age filter; defaults to ``time.time()``.
    :type now: Optional[float]
    :return: Totals and the stop reason.
    :rtype: Summary
    :raises MoverError: When the local folder is missing or the NAS root is not
        on a separate filesystem (share not mounted) and ``cfg.check_mount``.
    """
    if not cfg.local_dir.is_dir():
        raise MoverError(f"local dir not found: {cfg.local_dir}")
    if cfg.check_mount and not nas_on_separate_device(cfg.nas_root, cfg.local_dir):
        raise MoverError(
            f"{cfg.nas_root} is on the same filesystem as {cfg.local_dir}; is the NAS share mounted?"
        )
    selection = scan_candidates(cfg.local_dir, cfg.min_age_hours, now, cfg.required_sysnums)
    summary = Summary(
        apply=cfg.apply,
        zips_seen=selection.zips_seen,
        selection_skips=dict(selection.skipped),
        candidates=len(selection.candidates),
        candidate_bytes=sum(c.size for c in selection.candidates),
        nas_free_start=nas_free_bytes(cfg.nas_root),
    )
    todo = selection.candidates[: cfg.limit] if cfg.limit > 0 else selection.candidates
    log.info(
        "%s: %d candidates (%.2f GB) of %d zips; processing %d; NAS free %.1f GB",
        "APPLY" if cfg.apply else "DRY RUN", summary.candidates, summary.candidate_bytes / GB,
        summary.zips_seen, len(todo), summary.nas_free_start / GB,
    )
    lsof = shutil.which("lsof")
    if lsof is None:
        log.warning("lsof not found: the open-file check is disabled")
    started = time.monotonic()
    for start in range(0, len(todo), LSOF_BATCH):
        batch = todo[start:start + LSOF_BATCH]
        busy = open_paths([c.path for c in batch], lsof)
        for cand in batch:
            outcome = process_candidate(cand, cfg, busy, summary.bytes_copied)
            if outcome.stops_run:
                summary.stop_reason = outcome.action[len(_STOP_PREFIX):]
                log.warning("stopping before %s: %s (NAS free %.1f GB)", cand.path.name,
                            summary.stop_reason, nas_free_bytes(cfg.nas_root) / GB)
                break
            _record_outcome(summary, cand, outcome, cfg)
            log.info(_progress_line(summary.processed, len(todo), cand, outcome, summary,
                                    time.monotonic() - started))
        if summary.stop_reason:
            break
    if not summary.stop_reason:
        summary.stop_reason = "limit" if len(todo) < len(selection.candidates) else "done"
    summary.elapsed = time.monotonic() - started
    return summary


def format_summary(summary: Summary, cfg: MoverConfig) -> str:
    """Render the end-of-run summary.

    :param summary: Final totals.
    :type summary: Summary
    :param cfg: Run settings.
    :type cfg: MoverConfig
    :return: Multi-line summary text.
    :rtype: str
    """
    mode = "APPLY" if summary.apply else "DRY RUN (nothing was created, written or deleted)"
    skips = ", ".join(f"{k}={v}" for k, v in sorted(summary.selection_skips.items())) or "none"
    actions = ", ".join(f"{k}={v}" for k, v in sorted(summary.actions.items())) or "none"
    rate = summary.bytes_copied / 1e6 / summary.elapsed if summary.elapsed > 0 else 0.0
    copied, freed = ("copied", "freed locally") if summary.apply else ("would copy", "would free locally")
    return "\n".join([
        f"KTIV zip mover -- {mode}",
        f"  local dir   : {cfg.local_dir}",
        f"  NAS root    : {cfg.nas_root} (free at start {summary.nas_free_start / GB:,.1f} GB, "
        f"floor {cfg.nas_min_free_gb:g} GB)",
        f"  zips seen   : {summary.zips_seen} (selection skips: {skips})",
        f"  candidates  : {summary.candidates} ({summary.candidate_bytes / GB:,.2f} GB)",
        f"  processed   : {summary.processed} (limit {cfg.limit or 'none'}, max-gb {cfg.max_gb or 'none'})",
        f"  actions     : {actions}",
        f"  bytes       : {copied} {summary.bytes_copied / GB:,.2f} GB; {freed} {summary.bytes_freed / GB:,.2f} GB",
        f"  stop reason : {summary.stop_reason}",
        f"  elapsed     : {summary.elapsed:.1f} s ({rate:.1f} MB/s)",
        f"  manifest    : {cfg.manifest_path if summary.apply else 'not written (dry run)'}",
    ])


# ──────────────────────────────── zip lookup ───────────────────────────────


def zip_search_dirs(env_value: Optional[str] = None) -> List[Path]:
    """Return the zip search path: ``$KTIV_ZIP_DIRS`` or the local folder then the NAS.

    :param env_value: Colon-separated directory list; ``None`` reads the
        ``KTIV_ZIP_DIRS`` environment variable.
    :type env_value: Optional[str]
    :return: Directories in search order.
    :rtype: List[Path]
    """
    raw = os.environ.get(KTIV_ZIP_DIRS_ENV, "") if env_value is None else env_value
    dirs = [Path(part.strip()).expanduser() for part in raw.split(":") if part.strip()]
    return dirs or [DEFAULT_LOCAL_DIR, DEFAULT_NAS_ROOT]


def _zips_in(directory: Path, sysnum: str, filename: Optional[str]) -> List[Path]:
    """List a manuscript's zips in one directory.

    :param directory: Directory to look in (may not exist).
    :type directory: Path
    :param sysnum: Manuscript system number.
    :type sysnum: str
    :param filename: Exact filename to look for, or ``None`` for every
        ``ktiv_PNX_MANUSCRIPTS<sysnum>-1_images*.zip``.
    :type filename: Optional[str]
    :return: Matching files, sorted by name.
    :rtype: List[Path]
    """
    if filename is not None:
        path = directory / filename
        return [path] if path.is_file() else []
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob(f"ktiv_PNX_MANUSCRIPTS{sysnum}-1_images*.zip") if p.is_file())


def resolve_zips(sysnum_or_filename: str, search_dirs: Optional[Sequence[Path]] = None) -> List[Path]:
    """Find every image zip of a manuscript (or one zip by filename) across the search path.

    Each search directory is checked flat (``<dir>/<name>``) and in the NAS
    bucket layout (``<dir>/<bucket>/<name>``). A filename found in several
    directories is reported once, from the earliest directory.

    :param sysnum_or_filename: A sysnum, or a zip filename (a directory part
        is ignored).
    :type sysnum_or_filename: str
    :param search_dirs: Directories in search order; ``None`` uses
        :func:`zip_search_dirs` (``$KTIV_ZIP_DIRS`` or local, then NAS).
    :type search_dirs: Optional[Sequence[Path]]
    :return: Matching zips in search order (sorted by name within a directory).
    :rtype: List[Path]
    :raises ValueError: When the argument is neither a sysnum nor a zip filename.
    """
    key = str(sysnum_or_filename).strip()
    filename: Optional[str] = os.path.basename(key)
    sysnum = zip_sysnum(filename)
    if sysnum is None:
        if not key.isdigit():
            raise ValueError(f"expected a sysnum or a KTIV image-zip filename, got {sysnum_or_filename!r}")
        sysnum, filename = key, None
    found: Dict[str, Path] = {}
    for base in (zip_search_dirs() if search_dirs is None else search_dirs):
        for directory in (Path(base), Path(base) / bucket_name(sysnum)):
            for path in _zips_in(directory, sysnum, filename):
                found.setdefault(path.name, path)
    return list(found.values())


def resolve_zip(sysnum_or_filename: str, search_dirs: Optional[Sequence[Path]] = None) -> Optional[Path]:
    """Find a manuscript's image zip: the local folder first, then the NAS layout.

    :param sysnum_or_filename: A sysnum, or a zip filename.
    :type sysnum_or_filename: str
    :param search_dirs: Directories in search order; ``None`` uses
        ``$KTIV_ZIP_DIRS`` or the local folder then the NAS.
    :type search_dirs: Optional[Sequence[Path]]
    :return: The first match of :func:`resolve_zips`, or ``None``.
    :rtype: Optional[Path]
    """
    hits = resolve_zips(sysnum_or_filename, search_dirs)
    return hits[0] if hits else None


# ─────────────────────────────────── CLI ───────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    :return: The parser.
    :rtype: argparse.ArgumentParser
    """
    ap = argparse.ArgumentParser(
        prog="python -m src.datasets.tools.ktiv_zip_mover",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--apply", action="store_true", help="move files (default: read-only dry run)")
    ap.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR, help="scrape folder (default: %(default)s)")
    ap.add_argument("--nas-root", type=Path, default=DEFAULT_NAS_ROOT, help="NAS zip store (default: %(default)s)")
    ap.add_argument("--manifest", type=Path, default=None, help="JSONL log (default: <nas-root>/manifest.jsonl)")
    ap.add_argument("--min-age-hours", type=float, default=24.0, help="minimum age by mtime (default: 24)")
    ap.add_argument("--require-manifest", type=Path, default=None,
                    help="JSON/JSONL listing the only sysnums allowed to move (e.g. those already on GCS)")
    ap.add_argument("--nas-min-free-gb", type=float, default=200.0,
                    help="stop before the NAS would have less than this free (default: 200)")
    ap.add_argument("--max-gb", type=float, default=0.0, help="per-run copy cap in GB (default: 0 = unlimited)")
    ap.add_argument("--limit", type=int, default=0, help="process at most N candidates (default: 0 = all)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the mover from the command line.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :type argv: Optional[Sequence[str]]
    :return: Exit code: 0 after a run (with summary), 2 when a precondition failed.
    :rtype: int
    """
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    required: Optional[Set[str]] = None
    if args.require_manifest is not None:
        if not args.require_manifest.is_file():
            ap.error(f"--require-manifest not found: {args.require_manifest}")
        required = load_required_sysnums(args.require_manifest)
        log.info("--require-manifest: %d sysnums from %s", len(required), args.require_manifest)
    cfg = MoverConfig(
        local_dir=args.local_dir, nas_root=args.nas_root, manifest=args.manifest, apply=args.apply,
        min_age_hours=args.min_age_hours, required_sysnums=required, nas_min_free_gb=args.nas_min_free_gb,
        max_gb=args.max_gb, limit=args.limit,
    )
    try:
        summary = run(cfg)
    except MoverError as exc:
        log.error("%s", exc)
        return 2
    print(format_summary(summary, cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
