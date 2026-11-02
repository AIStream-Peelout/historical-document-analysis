# File name: raw_package.py
# Date: 10/30/26
# Author: Isaac Godfried. Coded by Claude.
"""Package every raw legacy-store download into one manifest: one record per shelf mark.

The legacy image store's material reached the project at different times and through two
browser extensions: an old scrape (folders of ``<shelf mark>_<n><r|v>.jpg`` images,
``cataloging_`` / ``joins_`` / ``bibliography_`` / ``metadata_`` JSON and transcription PDFs),
a public bucket snapshot of one older run, and new hand downloads mixed into the raw KTIV
folder. This module walks a configurable list of roots and builds one record per shelf mark.

Steps (each a public function):

1. :func:`classify_name` sorts files by NAME SHAPE only. Files of any other shape are counted as
   ``unrelated`` and never opened (the raw folders also hold personal and scraper files).
2. :func:`list_bucket` pages through the public bucket's XML listing (``prefix=`` / ``marker=``,
   no credentials) and :func:`fetch_snapshot` downloads the legacy-store-shaped objects over
   HTTPS, resumably and read-only (GET only; MD5 checked against the ETag);
   :func:`verify_snapshot` re-hashes what was fetched.
3. :func:`walk_root` lists a root's files with their kind, run folder and timestamps.
4. :func:`dedupe_images` groups images by SHA-256 across roots (every path kept as ``{root, rel}``,
   one canonical; :func:`image_group_record` is the ``images.jsonl`` line);
   :func:`label_race_status` dates each run folder against the image-saver's label-race fix and
   :func:`image_token_kind` says whether a folder's ``<n><r|v>`` tokens are sides at all.
5. :func:`read_metadata_json` reads catalogue, joins, bibliography and transcription-metadata JSON
   (URLs dropped; credit strings reduced to their kind, :func:`scrub_credit`), :func:`dedupe_json`
   collapses re-downloads.
6. Transcription PDFs are grouped by text digest and parsed with
   :mod:`src.datasets.indexing.legacy_store.pdf_transcriptions` in either style
   (:func:`pdf_transcriptions.detect_style`); the shelf mark comes from the header (new style)
   or the file name (old style, :func:`pdf_transcriptions.parse_side_file_name`). Side
   re-downloads are folded into the side they copy, distinct sides numbered by suffix order, and
   near-duplicate texts flagged (:func:`flag_near_duplicate_texts`).
7. :func:`package_key` gives every spelling (catalogue string, header, file-name stem, image stem)
   one join key; :func:`build_merged_index` and :func:`ktiv_status` look the shelf mark up in the
   merged corpus, and :func:`image_policy` applies the user's rule (2026-10-30): prefer the KTIV
   photograph wherever the manuscript is on KTIV; a legacy-only image needs the shelf-mark-in-image
   label check and content matching before it is trusted.
8. :func:`package_roots` runs the steps over every root and returns the manifest records (one per
   shelf mark), the transcription unit records and the unresolved texts.

No image is paired with a folio here: image-to-folio pairing stays unverified.
"""
import collections
import datetime as dt
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Callable, Counter, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import fitz

from src.datasets.indexing.legacy_store import pdf_transcriptions as P
from src.datasets.merging.legacy_store_inventory import LEGACY_SOURCE_KEY, join_key, record_marks
from src.datasets.merging.merge_shelfmarks import looks_multi, split_shelfmarks

# ───────────────────────────── constants ─────────────────────────────
# The image-saver extension's label-race fix (shelf mark captured at click time, not at image
# load time; commit "Improve shelfmark capture for image downloads"). Images saved before it
# showed a 52.8 % image/shelf-mark mismatch (recto/verso swaps, off-by-one documents).
LABEL_RACE_FIX_UTC = dt.datetime(2025, 10, 26, 22, 30, 13, tzinfo=dt.timezone.utc)
# A run folder whose tokens never end in ``v`` over this many images uses viewer sequence numbers.
VIEWER_SEQUENCE_MIN_IMAGES = 20
BUCKET_HOST = "https://storage.googleapis.com"
S3_NS = {"s": "http://doc.s3.amazonaws.com/2006-03-01"}
_UUID = r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
TRANSCRIPTION_PDF_RE = re.compile(
    rf"^transcription_(?:manual_(?P<stem>.+?)|(?P<response>response_)?(?P<uuid>{_UUID}))"
    r"(?P<copy>\(\d+\)|-\d+)?\.pdf$", re.IGNORECASE)
# Old-run PDFs named after the item (``a_12.pdf``, ``rylands_fragment_42.pdf``): lower-case series.
OTHER_SHELFMARK_PDF_RE = re.compile(r"^(?:[a-z]{1,2}|rylands_fragment)_\d+[a-z]?\.pdf$")
IMAGE_RE = re.compile(r"^(?P<stem>[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*?)_(?P<token>\d+[rv])"
                      r"(?:\((?P<copy>\d+)\))?\.jpe?g$", re.IGNORECASE)
JSON_RE = re.compile(r"^(?P<kind>cataloging|joins|bibliography|metadata)_(?P<rest>.+?)"
                     r"(?P<copy>\(\d+\)|-\d+)?\.json$")
# Pages the store returned instead of a record: no shelf mark in them.
JUNK_JSON_STEMS = ("Please_select_shelfmarks", "There_are_no_images_in_the_search_results", "No_Joins")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
# Holder abbreviations some sources put before the core (``JRL B 7961``, ``RNL Yevr. II C1``).
HOLDER_PREFIX_RE = re.compile(r"^(?:JRL|RNL|NLR)(?:[\s_.]+|$)", re.IGNORECASE)
KINDS = ("transcription_pdf", "other_shelfmark_pdf", "image", "cataloging_json", "joins_json",
         "bibliography_json", "metadata_json", "junk_json", "partial_download", "unrelated")
FETCH_KINDS = frozenset(KINDS) - {"unrelated", "partial_download"}
# The store's own credit formula (a team, a team's head, a site user). Credit strings of this form
# name the store's teams; they are reduced to their kind (:func:`scrub_credit`).
STORE_CREDIT_RE = re.compile(r"team|\bhead\s+of\b|^\s*site\s+user\b", re.IGNORECASE)
DEFAULT_INSTITUTION_HEADS = P.DEFAULT_INSTITUTION_HEADS
POLICY_PREFER_KTIV = "prefer_ktiv"
POLICY_LEGACY_ONLY = "legacy_only_needs_label_check"
POLICY_NO_IMAGE = "no_image"


@dataclass
class RootSpec:
    """One raw root.

    :param name: Short name used in run folders and the report.
    :param path: Local folder (for a snapshot: the folder the bucket objects were fetched into).
    :param recursive: Whether sub-folders are walked (hidden folders and ``.app`` bundles never are).
    :param listing: Bucket listing (:func:`list_bucket`) for a snapshot root, else None; its
        ``modified`` times date the snapshot's files and its unfetched objects are counted.
    :param prefix: Bucket key prefix of a snapshot root ("" for a local root).
    """

    name: str
    path: str
    recursive: bool = False
    listing: Optional[List[Dict[str, Any]]] = None
    prefix: str = ""


@dataclass
class FileEntry:
    """A file found under a root.

    :param root: Root name.
    :param path: Absolute path ("" for a snapshot object that was not fetched).
    :param rel: Path relative to the root.
    :param name: File name.
    :param kind: One of :data:`KINDS`.
    :param run_folder: ``<root>/<relative folder>``.
    :param size: Bytes.
    :param mtime: Modification time (local) or bucket upload time (snapshot), POSIX seconds.
    :param time_source: ``"file_mtime"`` or ``"bucket_upload"``.
    :param fetched: False for a snapshot object left in the bucket.
    :param extra: Free-form annotations (unused by the packager itself).
    """

    root: str
    path: str
    rel: str
    name: str
    kind: str
    run_folder: str
    size: int
    mtime: float
    time_source: str = "file_mtime"
    fetched: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


# ───────────────────────────── names ─────────────────────────────
def classify_name(name: str) -> str:
    """Kind of a file from its name alone.

    :param name: File name (no folder).
    :return: One of :data:`KINDS`; ``partial_download`` for a browser's unfinished ``.part`` file
        (counted, never read); ``unrelated`` for every other shape this module does not read.
    """
    if name.endswith(".part"):
        return "partial_download"
    if TRANSCRIPTION_PDF_RE.match(name):
        return "transcription_pdf"
    if OTHER_SHELFMARK_PDF_RE.match(name):
        return "other_shelfmark_pdf"
    if IMAGE_RE.match(name):
        return "image"
    m = JSON_RE.match(name)
    if m:
        if m.group("rest") in JUNK_JSON_STEMS:
            return "junk_json"
        return f"{m.group('kind')}_json"
    return "unrelated"


def parse_image_name(name: str) -> Optional[Dict[str, Any]]:
    """Stem, side token and Firefox copy number of an image file name.

    :param name: ``T_S_AS_95_124_1v.jpg``, ``B_7961_1r(1).jpg``.
    :return: ``stem``, ``token``, ``leaf`` (int), ``side`` (``r``/``v``), ``copy`` (int or None);
        None for another shape.
    """
    m = IMAGE_RE.match(name)
    if not m:
        return None
    token = m.group("token").lower()
    return {"stem": m.group("stem"), "token": token, "leaf": int(token[:-1]), "side": token[-1],
            "copy": int(m.group("copy")) if m.group("copy") else None}


# ───────────────────────────── shelf-mark keys ─────────────────────────────
strip_institution = P.strip_institution


def package_key(mark: Optional[str], heads: Sequence[str] = DEFAULT_INSTITUTION_HEADS) -> str:
    """One join key for every spelling of a legacy-store shelf mark.

    The institution head is dropped (:func:`strip_institution`), and so is a leading holder
    abbreviation (:data:`HOLDER_PREFIX_RE`); parentheses become separators
    (``T-S F1(1).33`` and the file-name ``T_S_F1_1_33`` agree; the merging normaliser would drop
    ``(1)``), and the rest goes through :func:`legacy_store_inventory.join_key` (separator- and
    case-insensitive, digit boundaries kept, ``Yevr``/``Evr`` unified).

    :param mark: Shelf mark, file-name stem or image stem.
    :param heads: Known institution heads.
    :return: The key ("" for an empty mark).
    """
    if not mark:
        return ""
    core, _ = strip_institution(mark, heads)
    core = HOLDER_PREFIX_RE.sub("", core.strip())
    core = re.sub(r"[()]", " ", core.replace("_", " "))
    return join_key(core)


def display_from_stem(stem: str, heads: Sequence[str] = DEFAULT_INSTITUTION_HEADS) -> str:
    """Readable shelf mark from an underscore stem (``T_S_AS_97_1`` -> ``T S AS 97 1``).

    :param stem: File-name or image stem.
    :param heads: Known institution heads.
    :return: The core with underscores as spaces.
    """
    core, _ = strip_institution(stem, heads)
    return re.sub(r"\s+", " ", core.replace("_", " ")).strip()


# ───────────────────────────── bucket snapshot ─────────────────────────────
def parse_listing_page(xml_bytes: bytes) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Objects of one XML listing page and the marker of the next page.

    :param xml_bytes: Response body of ``GET <bucket>?prefix=...&marker=...``.
    :return: (objects with ``key``, ``size``, ``etag``, ``modified``; next marker or None when the
        listing is complete).
    """
    root = ET.fromstring(xml_bytes)
    objects = []
    for c in root.findall("s:Contents", S3_NS):
        objects.append({"key": c.find("s:Key", S3_NS).text, "size": int(c.find("s:Size", S3_NS).text),
                        "etag": (c.find("s:ETag", S3_NS).text or "").strip('"'),
                        "modified": c.find("s:LastModified", S3_NS).text})
    truncated = (root.findtext("s:IsTruncated", default="false", namespaces=S3_NS) or "").lower() == "true"
    if not truncated:
        return objects, None
    marker = root.findtext("s:NextMarker", default=None, namespaces=S3_NS)
    return objects, marker or (objects[-1]["key"] if objects else None)


def list_bucket(bucket: str, prefix: str, opener: Callable[..., Any] = urllib.request.urlopen,
                delay: float = 0.3, sleep: Callable[[float], None] = time.sleep,
                host: str = BUCKET_HOST) -> List[Dict[str, Any]]:
    """Every object under a prefix of a public bucket, from the XML listing (no credentials).

    :param bucket: Bucket name.
    :param prefix: Key prefix (with its trailing slash).
    :param opener: ``urlopen``-like callable (injected in tests).
    :param delay: Seconds between requests.
    :param sleep: ``time.sleep``-like callable.
    :param host: Storage host.
    :return: Objects (``key``, ``size``, ``etag``, ``modified``) in listing order.
    """
    objects: List[Dict[str, Any]] = []
    marker = ""
    while True:
        query = urllib.parse.urlencode({"prefix": prefix, "marker": marker})
        with opener(f"{host}/{bucket}?{query}", timeout=60) as resp:
            page, marker = parse_listing_page(resp.read())
        objects.extend(page)
        if marker is None:
            return objects
        sleep(delay)


def file_md5(path: str, chunk: int = 1 << 20) -> str:
    """MD5 of a file (the bucket's ETag of a non-composite object).

    :param path: File.
    :param chunk: Read size.
    :return: 32 hex digits.
    """
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_rel(key: str, prefix: str) -> str:
    """Path of a bucket object below the snapshot folder.

    Keys that could leave the snapshot folder are refused: a key starting with ``/`` or a path
    with a ``..`` component (before or after the prefix is removed).

    :param key: Object key.
    :param prefix: Snapshot prefix.
    :return: The key without the prefix.
    :raises ValueError: For an unsafe key.
    """
    rel = key[len(prefix):] if key.startswith(prefix) else key
    for path in (key, rel):
        if path.startswith("/") or ".." in path.replace("\\", "/").split("/"):
            raise ValueError(f"unsafe bucket key refused: {key!r}")
    return rel


def read_state(state_path: str) -> Dict[str, Dict[str, Any]]:
    """Objects already fetched and verified (one JSON line per object, last line wins).

    :param state_path: State file (may not exist).
    :return: Key -> state record.
    """
    state: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    state[rec["key"]] = rec
    return state


def fetch_snapshot(listing: Sequence[Mapping[str, Any]], bucket: str, prefix: str, dest_dir: str, state_path: str,
                   kinds: Iterable[str] = FETCH_KINDS, opener: Callable[..., Any] = urllib.request.urlopen,
                   delay: float = 0.3, sleep: Callable[[float], None] = time.sleep, host: str = BUCKET_HOST,
                   log: Optional[Callable[[str], None]] = None, force: Iterable[str] = ()) -> Dict[str, Any]:
    """Download the legacy-store-shaped objects of a listing into ``dest_dir`` (resumable, GET only).

    An object is fetched when its file name classifies as one of ``kinds``; every other object
    (personal documents, scraper files) is left in the bucket. A fetched object's MD5 is checked
    against the ETag in memory, then the file is written and a state line appended (the state is
    the record of completed files: a file without a state line is fetched again); an object whose
    state line has the same ETag and whose file has the listed size is skipped.

    :param listing: :func:`list_bucket` output.
    :param bucket: Bucket name.
    :param prefix: Snapshot prefix (stripped from keys to make local paths).
    :param dest_dir: Local snapshot folder.
    :param state_path: Append-only state file.
    :param kinds: Kinds to fetch.
    :param opener: ``urlopen``-like callable.
    :param delay: Seconds between downloads.
    :param sleep: ``time.sleep``-like callable.
    :param host: Storage host.
    :param log: Progress callback (None: silent).
    :param force: Keys fetched again even when the state says they are present (after a failed
        :func:`verify_snapshot`).
    :return: ``fetched``, ``skipped_present``, ``not_fetched_kind`` (counts), ``bytes_fetched``,
        ``seconds``, ``md5_mismatch`` (keys whose download did not match the ETag; not written).
    """
    wanted = set(kinds)
    forced = set(force)
    made: Set[str] = set()
    state = read_state(state_path)
    stats: Dict[str, Any] = {"fetched": 0, "skipped_present": 0, "not_fetched_kind": 0, "bytes_fetched": 0,
                             "md5_mismatch": []}
    started = time.time()
    with open(state_path, "a", encoding="utf-8") as state_fh:
        for i, obj in enumerate(listing):
            rel = snapshot_rel(obj["key"], prefix)
            if not rel or rel.endswith("/") or classify_name(os.path.basename(rel)) not in wanted:
                stats["not_fetched_kind"] += 1
                continue
            target = os.path.join(dest_dir, rel)
            done = state.get(obj["key"])
            if obj["key"] not in forced and done and done.get("etag") == obj["etag"] and os.path.exists(target) \
                    and os.path.getsize(target) == obj["size"]:
                stats["skipped_present"] += 1
                continue
            folder = os.path.dirname(target)
            if folder not in made:
                os.makedirs(folder, exist_ok=True)
                made.add(folder)
            url = f"{host}/{bucket}/{urllib.parse.quote(obj['key'])}"
            with opener(url, timeout=120) as resp:
                body = resp.read()              # objects are small (images < 10 MB): checked in memory
            md5 = hashlib.md5(body).hexdigest()
            if re.fullmatch(r"[0-9a-f]{32}", obj["etag"]) and md5 != obj["etag"]:
                stats["md5_mismatch"].append(obj["key"])
                continue
            with open(target, "wb") as out:     # a crash mid-write leaves no state line: re-fetched next run
                out.write(body)
            state_fh.write(json.dumps({"key": obj["key"], "etag": obj["etag"], "size": obj["size"], "md5": md5,
                                       "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
                           + "\n")
            if stats["fetched"] % 25 == 0:
                state_fh.flush()
            stats["fetched"] += 1
            stats["bytes_fetched"] += obj["size"]
            if log and stats["fetched"] % 200 == 0:
                log(f"{i + 1}/{len(listing)} objects seen, {stats['fetched']} fetched, "
                    f"{stats['bytes_fetched'] / 1e9:.2f} GB")
            sleep(delay)
    stats["seconds"] = round(time.time() - started, 1)
    return stats


def verify_snapshot(listing: Sequence[Mapping[str, Any]], prefix: str, dest_dir: str, state_path: str
                    ) -> Dict[str, List[str]]:
    """Re-hash every fetched file against its listing ETag (a check after interrupted or overlapping runs).

    :param listing: :func:`list_bucket` output.
    :param prefix: Snapshot prefix.
    :param dest_dir: Local snapshot folder.
    :param state_path: Fetch state file.
    :return: ``ok``, ``bad`` (size or MD5 differs) and ``missing`` (in the state, not on disk) keys.
    """
    state = read_state(state_path)
    out: Dict[str, List[str]] = {"ok": [], "bad": [], "missing": []}
    for obj in listing:
        if obj["key"] not in state:
            continue
        target = os.path.join(dest_dir, snapshot_rel(obj["key"], prefix))
        if not os.path.exists(target):
            out["missing"].append(obj["key"])
        elif os.path.getsize(target) != obj["size"] or (re.fullmatch(r"[0-9a-f]{32}", obj["etag"])
                                                         and file_md5(target) != obj["etag"]):
            out["bad"].append(obj["key"])
        else:
            out["ok"].append(obj["key"])
    return out


# ───────────────────────────── walking roots ─────────────────────────────
def iso_to_epoch(text: Optional[str]) -> Optional[float]:
    """POSIX seconds of an ISO-8601 time (``2025-11-18T14:32:19.032Z``).

    :param text: ISO time or None.
    :return: Seconds, or None for a missing or unparsable value.
    """
    if not text:
        return None
    m = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.\d+)?(Z|[+-]\d{2}:?\d{2})?$", text.strip())
    if not m:
        return None
    zone = m.group(2) or "Z"
    stamp = dt.datetime.fromisoformat(m.group(1) + ("+00:00" if zone == "Z" else zone))
    return stamp.timestamp()


def _walk_dirs(path: str, recursive: bool) -> Iterable[Tuple[str, List[str]]]:
    """Folders and their file names, skipping hidden folders and ``.app`` bundles.

    :param path: Root folder.
    :param recursive: Whether to descend.
    :return: ``(folder, file names)`` pairs.
    """
    for folder, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not d.endswith(".app"))
        yield folder, sorted(files)
        if not recursive:
            return


def walk_root(spec: RootSpec) -> List[FileEntry]:
    """Every file of a root with its kind (files are stat-ed, never opened here).

    For a snapshot root the bucket listing gives each fetched file's upload time
    (``time_source = "bucket_upload"``, an upper bound on when it was scraped) and adds a
    ``fetched=False`` entry for every listed object that is not on disk (other shapes were left
    in the bucket; hidden object names included, so the listing is counted in full). Local hidden
    files (``.DS_Store``) are skipped; a browser's unfinished ``.part`` download is listed with kind
    ``partial_download``.

    :param spec: The root.
    :return: Entries in folder and name order.
    """
    by_rel: Dict[str, Mapping[str, Any]] = {}
    if spec.listing is not None:
        by_rel = {snapshot_rel(o["key"], spec.prefix): o for o in spec.listing}
    out: List[FileEntry] = []
    seen: Set[str] = set()
    for folder, files in _walk_dirs(spec.path, spec.recursive):
        rel_dir = os.path.relpath(folder, spec.path)
        for name in files:
            if name.startswith("."):
                continue
            path = os.path.join(folder, name)
            rel = os.path.normpath(os.path.join(rel_dir, name))
            st = os.stat(path)
            obj = by_rel.get(rel)
            mtime, source = st.st_mtime, "file_mtime"
            if obj is not None:
                mtime, source = iso_to_epoch(obj["modified"]) or st.st_mtime, "bucket_upload"
            out.append(FileEntry(spec.name, path, rel, name, classify_name(name),
                                 f"{spec.name}/{rel_dir}" if rel_dir != "." else spec.name, st.st_size, mtime, source))
            seen.add(rel)
    for rel, obj in by_rel.items():
        if rel in seen or not rel or rel.endswith("/"):
            continue
        name = os.path.basename(rel)
        rel_dir = os.path.dirname(rel)
        out.append(FileEntry(spec.name, "", rel, name, classify_name(name),
                             f"{spec.name}/{rel_dir}" if rel_dir else spec.name, obj["size"],
                             iso_to_epoch(obj["modified"]) or 0.0, "bucket_upload", fetched=False))
    return out


# ───────────────────────────── images ─────────────────────────────
def label_race_status(extracted_at: Sequence[float], file_times: Sequence[float], upper_bound: bool,
                      fix: dt.datetime = LABEL_RACE_FIX_UTC) -> str:
    """Whether a run folder's images were saved before the image-saver's label-race fix.

    The extension's own ``extractedAt`` stamps (catalogue / joins / bibliography JSON saved in
    the same run folder) are preferred; without them, the files' times are used: download times
    (``file_mtime``) or bucket upload times, which only bound the save time from above
    (``upper_bound``: a time after the fix proves nothing).

    :param extracted_at: ``extractedAt`` stamps of the folder's JSON (POSIX seconds).
    :param file_times: File times of the folder's images.
    :param upper_bound: Whether ``file_times`` are upper bounds (bucket uploads).
    :param fix: Time of the fix.
    :return: ``pre_fix``, ``post_fix``, ``spans_fix`` or ``unknown``.
    """
    cut = fix.timestamp()
    times = list(extracted_at) or list(file_times)
    if not times:
        return "unknown"
    if max(times) < cut:
        return "pre_fix"
    if not extracted_at and upper_bound:
        return "unknown"
    if min(times) >= cut:
        return "post_fix"
    return "spans_fix"


def image_token_kind(tokens: Sequence[str], min_images: int = VIEWER_SEQUENCE_MIN_IMAGES) -> str:
    """Whether a run folder's ``<n><r|v>`` image tokens are sides at all.

    The new hand downloads append ``r`` to the viewer's image sequence number: over many images
    no token ends in ``v``. Such tokens name neither a folio nor a side.

    :param tokens: Tokens of the folder's images (copies excluded).
    :param min_images: Fewest images for a verdict.
    :return: ``viewer_sequence_suspected`` or ``side_token``.
    """
    if len(tokens) >= min_images and not any(t.endswith("v") for t in tokens):
        return "viewer_sequence_suspected"
    return "side_token"


def dedupe_images(entries: Sequence[FileEntry], root_order: Sequence[str],
                  sha: Callable[[str], str] = P.file_sha256) -> List[Dict[str, Any]]:
    """Group image files by SHA-256 across roots; every path is kept, one is canonical.

    The canonical path is in the earliest root of ``root_order``, then without a ``(n)`` copy
    suffix, then the shortest name. Paths are stored as ``{root, rel}`` (root name and path below
    it, as transcription files are): the roots file resolves them, and no absolute path (whose
    folder names may name a source) reaches the package.

    :param entries: Image entries (fetched files only).
    :param root_order: Root names in preference order.
    :param sha: File hasher.
    :return: One dict per distinct image: ``sha256``, ``size``, ``canonical`` (``root``, ``rel``),
        ``paths`` (``root``, ``rel``, ``run_folder``, ``name``, ``stem``, ``token``, ``copy``),
        ``stems`` (sorted).
    """
    rank = {name: i for i, name in enumerate(root_order)}
    groups: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        info = parse_image_name(e.name)
        digest = sha(e.path)
        g = groups.setdefault(digest, {"sha256": digest, "size": e.size, "paths": []})
        g["paths"].append({"root": e.root, "rel": e.rel, "run_folder": e.run_folder, "name": e.name,
                           "stem": info["stem"], "token": info["token"], "copy": info["copy"]})
    out = []
    for g in groups.values():
        g["paths"].sort(key=lambda p: (rank.get(p["root"], len(rank)), p["copy"] is not None, len(p["name"]), p["name"]))
        g["canonical"] = {"root": g["paths"][0]["root"], "rel": g["paths"][0]["rel"]}
        g["stems"] = sorted({p["stem"] for p in g["paths"]})
        out.append(g)
    return sorted(out, key=lambda g: (rank.get(g["canonical"]["root"], len(rank)), g["canonical"]["rel"]))


def image_group_record(group: Mapping[str, Any]) -> Dict[str, Any]:
    """The ``images.jsonl`` line of one image group (paths as ``{root, rel}``, no absolute path).

    :param group: :func:`dedupe_images` group, annotated by :func:`package_roots`.
    :return: ``sha256``, ``size``, ``canonical``, ``stems``, ``paths``, ``label_race``,
        ``token_kind``, ``in_corpus_legacy_names``, ``flags``.
    """
    return {"sha256": group["sha256"], "size": group["size"], "canonical": dict(group["canonical"]),
            "stems": list(group["stems"]), "paths": [dict(p) for p in group["paths"]],
            "label_race": group["label_race"], "token_kind": group["token_kind"],
            "in_corpus_legacy_names": group["in_corpus_legacy_names"], "flags": list(group["flags"])}


# ───────────────────────────── metadata JSON ─────────────────────────────
def _drop_urls(value: Any) -> Any:
    """Remove URL strings (and ``links`` lists) from a JSON value.

    :param value: JSON-like value.
    :return: The value without URLs (a string that is only a URL becomes "").
    """
    if isinstance(value, str):
        return URL_RE.sub("", value).strip()
    if isinstance(value, dict):
        return {k: _drop_urls(v) for k, v in value.items() if k not in ("links", "pageUrl", "url")}
    if isinstance(value, list):
        return [_drop_urls(v) for v in value]
    return value


def scrub_credit(text: Any) -> Any:
    """A credit string in the store's own formula reduced to its kind (other values unchanged).

    Credits naming the store's teams, a team's head or a site user (:data:`STORE_CREDIT_RE`)
    carry the store's name; they become ``legacy store <kind>`` (:func:`pdf_transcriptions.credit_kind`).
    Published catalogue names (``Some Catalog``) are kept.

    :param text: Credit value (any JSON value).
    :return: The value, or its kind label.
    """
    if not isinstance(text, str) or not STORE_CREDIT_RE.search(text):
        return text
    return "legacy store " + str(P.credit_kind(text)).replace("_", " ")


def split_marks(mark: str) -> List[str]:
    """Individual shelf marks of a catalogue string (join strings name several).

    Single marks are returned unchanged (the merge's splitter drops parentheses, which would turn
    ``T-S F1(1).33`` into ``T-S F1.33``).

    :param mark: Shelf-mark string.
    :return: Marks (at least one for a non-empty string).
    """
    mark = (mark or "").strip()
    if not mark:
        return []
    return split_shelfmarks(mark) if looks_multi(mark) else [mark]


def read_metadata_json(path: str, kind: str) -> Dict[str, Any]:
    """Shelf marks and payload of one catalogue / joins / bibliography / metadata JSON.

    URLs are dropped from the payload. Credits are never kept verbatim: a catalogue's ``Source``
    entries and ``sources`` list and a join's ``source`` pass through :func:`scrub_credit`; a
    bibliography's ``source`` and a transcription metadata ``source`` (the transcription credit)
    are dropped and replaced by their kind (``source_kind`` / ``credit_kind``,
    :func:`pdf_transcriptions.credit_kind`). A record whose shelf mark has no digit is junk (the
    store answered with a placeholder page).

    :param path: JSON file.
    :param kind: ``cataloging_json``, ``joins_json``, ``bibliography_json`` or ``metadata_json``.
    :return: ``kind``, ``marks`` (shelf marks the record is about; for joins the main mark only),
        ``multi`` (the string named several marks), ``payload``, ``extracted_at`` (POSIX seconds
        or None), ``junk``, ``unreadable``.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:       # truncated download: counted, never fatal
        return {"kind": kind, "marks": [], "multi": False, "payload": {}, "extracted_at": None, "junk": True,
                "unreadable": True}
    if not isinstance(data, dict):
        return {"kind": kind, "marks": [], "multi": False, "payload": {}, "extracted_at": None, "junk": True,
                "unreadable": True}
    meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    extracted = iso_to_epoch(meta.get("extractedAt") or data.get("extractedAt"))
    raw_mark, payload = "", {}
    if kind == "cataloging_json":
        ms = data.get("manuscript") or {}
        raw_mark = ms.get("shelfmark") or ""
        payload = {"entries": [{"section": e.get("section"), "field": e.get("field"),
                                "value": scrub_credit(e.get("value")) if str(e.get("field") or "").strip().lower()
                                == "source" else e.get("value")}
                               for e in ms.get("catalogingEntries") or [] if isinstance(e, dict)],
                   "sources": [scrub_credit(x.get("name")) for x in ms.get("sources") or [] if isinstance(x, dict)]}
    elif kind == "joins_json":
        raw_mark = data.get("mainShelfmark") or ""
        payload = {"joined": [j.get("shelfmark") for j in data.get("joinedManuscripts") or [] if isinstance(j, dict)],
                   "source": scrub_credit(data.get("source"))}
    elif kind == "bibliography_json":
        entries = {e.get("key"): e.get("value") for e in data.get("entries") or [] if isinstance(e, dict)}
        raw_mark = entries.get("shelfmark") or ""
        payload = {k: v for k, v in entries.items() if k not in ("shelfmark", "source")}
        if "source" in entries:
            payload["source_kind"] = P.credit_kind(entries.get("source"))
    elif kind == "metadata_json":
        raw_mark = data.get("shelfmark") or ""
        payload = {"credit_kind": P.credit_kind(data.get("source")), "inventory_id": data.get("inventoryId")}
    marks = split_marks(raw_mark)
    junk = not re.search(r"\d", raw_mark)
    return {"kind": kind, "marks": marks, "multi": len(marks) > 1, "payload": _drop_urls(payload),
            "extracted_at": extracted, "junk": junk, "unreadable": False, "raw_mark": raw_mark}


def dedupe_json(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse re-downloads: equal kind, marks and payload are one record.

    :param records: :func:`read_metadata_json` outputs with ``path`` and ``root`` added.
    :return: Distinct records, each with ``copies`` (paths) and ``digest``.
    """
    groups: Dict[str, Dict[str, Any]] = {}
    for r in records:
        digest = hashlib.sha1(json.dumps([r["kind"], r["marks"], r["payload"]], sort_keys=True,
                                         ensure_ascii=False).encode("utf-8")).hexdigest()
        g = groups.get(digest)
        if g is None:
            g = groups[digest] = dict(r, digest=digest, copies=[])
        g["copies"].append(r["path"])
    return list(groups.values())


# ───────────────────────────── merged corpus ─────────────────────────────
def merged_holdings(record: Mapping[str, Any]) -> Dict[str, Any]:
    """What the merged corpus holds for one record: KTIV presence and images of every source.

    :param record: One ``merged_shelfmarks.jsonl`` line.
    :return: ``canonical_id``, ``display``, ``institution``, ``on_ktiv``, ``ktiv_images_held``
        (images fetched to the project store), ``ktiv_iiif_manifest``, ``ktiv_image_count``,
        ``ktiv_transcription``, ``legacy_image_names`` (legacy-store image file names already in
        the corpus), ``bodleian_images``, ``preferred_source``.
    """
    sources = record.get("sources") or {}
    images = record.get("images") or {}
    ktiv_img = images.get("ktiv") or {}
    return {
        "canonical_id": record.get("canonical_id") or "", "display": record.get("shelfmark_display") or "",
        "institution": record.get("institution") or "", "on_ktiv": bool(sources.get("ktiv")),
        "ktiv_images_held": bool(ktiv_img.get("populated") and ktiv_img.get("image_urls")),
        "ktiv_iiif_manifest": bool(ktiv_img.get("iiif_manifest_url")),
        "ktiv_image_count": int(ktiv_img.get("image_count") or 0),
        "ktiv_transcription": bool(sources.get("ktiv_transcription")),
        "legacy_image_names": list(images.get(LEGACY_SOURCE_KEY) or []),
        "bodleian_images": bool(images.get("bodleian")), "preferred_source": images.get("preferred_source"),
    }


def build_merged_index(merged_path: str, heads: Sequence[str] = DEFAULT_INSTITUTION_HEADS
                       ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[str]]]:
    """Index the merged corpus by every key its spellings give, and by legacy image file name.

    :param merged_path: ``merged_shelfmarks.jsonl``.
    :param heads: Institution heads for :func:`package_key`.
    :return: (key -> holdings list (:func:`merged_holdings`), legacy image file name -> canonical
        ids). Keys are :func:`package_key` and :func:`legacy_store_inventory.join_key` of every
        spelling (:func:`legacy_store_inventory.record_marks`). A record whose legacy-store images
        are all named for OTHER shelf marks (a box such as ``T-S AS 97`` holding the images of
        ``T-S AS 97.1`` ...) gets ``box_level = True``: it proves no item exists under its key.
    """
    index: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    names: Dict[str, List[str]] = collections.defaultdict(list)
    with open(merged_path, encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            holdings = merged_holdings(record)
            keys: Set[str] = set()
            for mark in record_marks(record):
                if mark:
                    keys.add(package_key(mark, heads))
                    keys.add(join_key(mark))
            keys.discard("")
            stems = {package_key(info["stem"], heads) for info in
                     (parse_image_name(n) for n in holdings["legacy_image_names"]) if info}
            holdings["box_level"] = bool(stems) and not (stems & keys)
            for key in keys:
                index[key].append(holdings)
            for name in holdings["legacy_image_names"]:
                names[name].append(holdings["canonical_id"])
    return dict(index), dict(names)


_INSTITUTION_HINT_TOKENS = {"cambridge": ("Cambridge", "CUL", "T_S"), "manchester": ("Manchester", "JRL"),
                            "petersburg": ("Jerusalem_NLI_RNL", "RNL", "Petersburg")}


def ktiv_status(key: str, hints: Iterable[str], index: Mapping[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """KTIV presence and images for one package key.

    When several corpus records answer to the key, those whose canonical id fits an institution
    hint win; a remaining tie is ``ambiguous`` and the flags are OR-ed over the candidates.

    :param key: :func:`package_key`.
    :param hints: Institution heads seen for the shelf mark.
    :param index: :func:`build_merged_index` key index.
    :return: ``matched``, ``ambiguous``, ``canonical_ids``, ``on_ktiv``, ``ktiv_images_held``,
        ``ktiv_iiif_manifest``, ``ktiv_image_count``, ``ktiv_transcription``,
        ``corpus_legacy_images``, ``bodleian_images``.
    """
    cands = list(index.get(key, []))
    if len(cands) > 1:
        wanted = [tok for h in hints for word, toks in _INSTITUTION_HINT_TOKENS.items() if word in h.lower()
                  for tok in toks]
        fit = [c for c in cands if any(t.lower() in c["canonical_id"].lower() for t in wanted)]
        if fit:
            cands = fit
    by_id = {c["canonical_id"]: c for c in cands}
    cands = list(by_id.values())
    return {
        "matched": bool(cands), "ambiguous": len(cands) > 1, "canonical_ids": sorted(by_id),
        "on_ktiv": any(c["on_ktiv"] for c in cands), "ktiv_images_held": any(c["ktiv_images_held"] for c in cands),
        "ktiv_iiif_manifest": any(c["ktiv_iiif_manifest"] for c in cands),
        "ktiv_image_count": max((c["ktiv_image_count"] for c in cands), default=0),
        "ktiv_transcription": any(c["ktiv_transcription"] for c in cands),
        "corpus_legacy_images": sum(len(c["legacy_image_names"]) for c in cands),
        "bodleian_images": any(c["bodleian_images"] for c in cands),
    }


def image_policy(ktiv: Mapping[str, Any], legacy_images: int) -> str:
    """The user's image rule (2026-10-30) for one shelf mark.

    :param ktiv: :func:`ktiv_status` output.
    :param legacy_images: Distinct legacy-store images of the shelf mark in this package.
    :return: :data:`POLICY_PREFER_KTIV` when the manuscript is on KTIV with a photograph (held,
        or listed in a IIIF manifest); else :data:`POLICY_LEGACY_ONLY` when legacy-store images
        exist (shelf-mark-in-image label check and content matching needed before trust); else
        :data:`POLICY_NO_IMAGE`.
    """
    if ktiv.get("on_ktiv") and (ktiv.get("ktiv_images_held") or ktiv.get("ktiv_iiif_manifest")):
        return POLICY_PREFER_KTIV
    if legacy_images:
        return POLICY_LEGACY_ONLY
    return POLICY_NO_IMAGE


# ───────────────────────────── transcription PDFs ─────────────────────────────
def scan_pdf(entry: FileEntry, side_file_evidence: bool = False) -> Dict[str, Any]:
    """Junk check, text digest and style of one PDF.

    :param entry: The PDF's entry (fetched).
    :param side_file_evidence: Whether the file name's side suffix is confirmed by a sibling file
        (passed to :func:`pdf_transcriptions.detect_style`).
    :return: ``digest`` (None for an unreadable file), ``junk``, ``style``, ``pages``,
        ``unreadable``, ``side_file_evidence``.
    """
    if entry.size == 0:
        return {"digest": None, "junk": False, "style": "unknown", "pages": 0, "unreadable": True,
                "side_file_evidence": side_file_evidence}
    try:
        doc = fitz.open(entry.path)
    except fitz.FileDataError:         # damaged download: counted, never fatal
        return {"digest": None, "junk": False, "style": "unknown", "pages": 0, "unreadable": True,
                "side_file_evidence": side_file_evidence}
    with doc:
        junk = P.is_junk(doc)
        return {"digest": P.content_digest(doc), "junk": junk, "pages": len(doc), "unreadable": False,
                "style": "junk" if junk else P.detect_style(doc, entry.name, side_file_evidence),
                "side_file_evidence": side_file_evidence}


def pdf_name_key(name: str, heads: Sequence[str] = DEFAULT_INSTITUTION_HEADS) -> str:
    """Package key of the shelf mark a PDF's file name gives (side suffix read as the naming rule).

    :param name: File name (``transcription_manual_<stem>[_<n>].pdf`` or an item name ``a_152.pdf``).
    :param heads: Institution heads.
    :return: The key, or "" for a name without a shelf mark (uuid names).
    """
    parsed = P.parse_side_file_name(name, heads=heads)
    if parsed:
        return package_key(parsed["stem"], heads)
    if OTHER_SHELFMARK_PDF_RE.match(name):
        return package_key(os.path.splitext(name)[0].replace("_", " "), heads)
    return ""


def side_known_checker(known_keys: Set[str], heads: Sequence[str] = DEFAULT_INSTITUTION_HEADS
                       ) -> Callable[[str], bool]:
    """``is_known`` callable for :func:`pdf_transcriptions.parse_side_file_name`.

    A key counts as a known ITEM when it is in ``known_keys`` and no other known key extends it
    by a number (``tsas97`` is a box when ``tsas97|4`` is known: the store catalogues boxes too).

    :param known_keys: Package keys of known items (corpus, catalogue JSON, image stems).
    :param heads: Institution heads.
    :return: Callable telling whether a core shelf mark is a known item.
    """
    containers = set()
    for key in known_keys:
        parts = key.split("|")
        for i in range(1, len(parts)):
            containers.add("|".join(parts[:i]))

    def is_item(core: str) -> bool:
        """Whether a core shelf mark is a known item.

        :param core: Core shelf mark (underscores or any separators).
        :return: True for a known, non-container key.
        """
        key = package_key(core, heads)
        return key in known_keys and key not in containers

    return is_item


def assign_side(names: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """One shelf mark and side for a text whose copies carry several manual file names.

    Names without a download-copy suffix win; among them the lowest side (a later file with the
    same text is a re-download under the next free suffix: the higher sides of the same stem are
    re-download names, counted in the flag ``side_redownload_copies`` and listed in
    ``redownload_name_sides``, as :func:`pdf_transcriptions.merge_side_redownloads` treats a copy
    whose text layer differs; no conflict). Copies named for different shelf marks (one text saved
    for several items) are flagged ``shelfmark_name_conflict`` (the first name is used, all are
    listed); a side known only from a copy-suffixed name ``side_from_copy_name_unverified``.

    :param names: :func:`pdf_transcriptions.parse_side_file_name` outputs (with ``file``).
    :return: The chosen parse plus ``flags`` (list), ``redownload_name_sides`` and
        ``name_candidates``.
    """
    plain = [n for n in names if not n["copy_suffix"]]
    pool = plain or list(names)
    best = min(pool, key=lambda n: (n["side_index"], n["file"]))
    flags = []
    if len({re.sub(r"_+", "_", n["stem"]) for n in pool}) > 1:
        flags.append("shelfmark_name_conflict")
    redownload_sides = sorted({n["side_index"] for n in pool if n["stem"] == best["stem"]} - {best["side_index"]})
    if redownload_sides:
        flags.append("side_redownload_copies")
    if not plain:
        flags.append("side_from_copy_name_unverified")
    if best["side_ambiguous"]:
        flags.append("side_index_ambiguous_name")
    return dict(best, flags=flags, redownload_name_sides=redownload_sides,
                name_candidates=[{"file": n["file"], "stem": n["stem"], "side": n["side_index"],
                                  "resolution": n["side_resolution"]} for n in names])


# ───────────────────────────── packaging ─────────────────────────────
@dataclass
class ShelfRecord:
    """Everything the package holds for one shelf mark (built by :func:`package_roots`).

    :param key: :func:`package_key`.
    :param spellings: Shelf-mark spellings seen (catalogue strings, headers, file-name stems).
    :param institutions: Institution heads seen.
    :param images: Image groups by SHA-256 (:func:`dedupe_images`).
    :param transcriptions: Transcription entries by text digest.
    :param attached: JSON kind -> attached catalogue / joins / bibliography / metadata records.
    :param flags: Record flags.
    """

    key: str
    spellings: Set[str] = field(default_factory=set)
    institutions: Set[str] = field(default_factory=set)
    images: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    transcriptions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    attached: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: collections.defaultdict(list))
    flags: Set[str] = field(default_factory=set)


def _shelf(records: Dict[str, ShelfRecord], key: str) -> ShelfRecord:
    """The record of a key, created on first use.

    :param records: Records by key.
    :param key: Package key.
    :return: The record.
    """
    if key not in records:
        records[key] = ShelfRecord(key)
    return records[key]


def _folder_status(entries: Sequence[FileEntry], json_times: Mapping[str, List[float]]
                   ) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Label-race status and token kind of every run folder holding images.

    :param entries: All entries.
    :param json_times: Run folder -> ``extractedAt`` stamps of its JSON.
    :return: (run folder -> :func:`label_race_status`, run folder -> :func:`image_token_kind`).
    """
    times: Dict[str, List[float]] = collections.defaultdict(list)
    upper: Dict[str, bool] = {}
    tokens: Dict[str, List[str]] = collections.defaultdict(list)
    for e in entries:
        if e.kind != "image" or not e.fetched:
            continue
        times[e.run_folder].append(e.mtime)
        upper[e.run_folder] = e.time_source == "bucket_upload"
        info = parse_image_name(e.name)
        if info["copy"] is None:
            tokens[e.run_folder].append(info["token"])
    race = {f: label_race_status(json_times.get(f, []), t, upper[f]) for f, t in times.items()}
    kind = {f: image_token_kind(t) for f, t in tokens.items()}
    return race, kind


def _group_race(statuses: Iterable[str]) -> str:
    """Best label-race evidence over an image's copies (a post-fix copy settles it).

    :param statuses: Run-folder statuses of the copies.
    :return: ``post_fix``, ``pre_fix``, ``spans_fix`` or ``unknown``.
    """
    found = set(statuses)
    for status in ("post_fix", "pre_fix", "spans_fix"):
        if status in found:
            return status
    return "unknown"


def package_roots(specs: Sequence[RootSpec], merged_index: Mapping[str, List[Dict[str, Any]]],
                  merged_names: Mapping[str, List[str]], lexicon: Optional[Mapping[str, int]],
                  heads: Sequence[str] = DEFAULT_INSTITUTION_HEADS, sha: Callable[[str], str] = P.file_sha256,
                  log: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Build the package over every root: manifest records, transcription records and counts.

    Side files of one shelf mark are cleaned together, re-downloads saved under a higher suffix are
    folded into the side they copy (:func:`pdf_transcriptions.merge_side_redownloads`: no unit of
    their own, their files listed as copies), the distinct sides are numbered by suffix order
    (:func:`pdf_transcriptions.renumber_sides`), and the editor-note document rule runs per shelf
    mark. Across all texts, a later near-duplicate of an earlier text is flagged
    ``near_duplicate_document`` (:func:`flag_near_duplicate_texts`).

    :param specs: Roots in preference order (the first root's copy of an image is canonical).
    :param merged_index: :func:`build_merged_index` key index.
    :param merged_names: :func:`build_merged_index` legacy image name index.
    :param lexicon: Word -> count for finals inference and repairs.
    :param heads: Institution heads.
    :param sha: Image hasher.
    :param log: Progress callback.
    :return: ``manifest`` (one dict per shelf mark, sorted by key), ``transcriptions`` (one dict
        per folio or side), ``unresolved`` (transcriptions without a shelf mark), ``entries`` (every
        file of every root), ``images`` (distinct image groups), ``pdf_groups`` (by text digest),
        ``scans`` (PDF path -> :func:`scan_pdf`), ``json_records`` (every JSON read),
        ``json_distinct``, ``json_unattached`` (kind -> distinct records without a shelf record),
        ``folder_race`` and ``folder_token_kind`` (run folder -> status), ``side_redownloads``
        (folded copies, with ``key``), ``near_duplicates`` (:func:`flag_near_duplicate_texts`).
    """
    say = log or (lambda m: None)
    entries: List[FileEntry] = []
    for spec in specs:
        found = walk_root(spec)
        say(f"{spec.name}: {len(found)} entries")
        entries.extend(found)
    root_order = [s.name for s in specs]

    # metadata JSON
    json_recs: List[Dict[str, Any]] = []
    json_times: Dict[str, List[float]] = collections.defaultdict(list)
    for e in entries:
        if e.fetched and e.kind in ("cataloging_json", "joins_json", "bibliography_json", "metadata_json"):
            r = read_metadata_json(e.path, e.kind)
            r.update(path=e.path, root=e.root, run_folder=e.run_folder)
            json_recs.append(r)
            if r["extracted_at"]:
                json_times[e.run_folder].append(r["extracted_at"])
    distinct_json = dedupe_json([r for r in json_recs if not r["junk"]])
    say(f"json: {len(json_recs)} read, {len(distinct_json)} distinct")

    records: Dict[str, ShelfRecord] = {}
    # Item evidence for reading ``_<n>`` file-name suffixes: corpus items (box-level records
    # excluded), catalogue and metadata items, image stems.
    known_keys: Set[str] = {k for k, hs in merged_index.items() if any(not h.get("box_level") for h in hs)}
    for r in distinct_json:
        for mark in r["marks"]:
            k = package_key(mark, heads)
            if not k:
                continue
            if r["kind"] in ("cataloging_json", "metadata_json"):
                known_keys.add(k)
            if r["kind"] in ("cataloging_json", "metadata_json") and not r["multi"]:
                _shelf(records, k)

    # images
    race, token_kind = _folder_status(entries, json_times)
    image_entries = [e for e in entries if e.kind == "image" and e.fetched]
    groups = dedupe_images(image_entries, root_order, sha)
    say(f"images: {len(image_entries)} files, {len(groups)} distinct")
    for g in groups:
        g["label_race"] = _group_race(race.get(p["run_folder"], "unknown") for p in g["paths"])
        kinds = {token_kind.get(p["run_folder"], "side_token") for p in g["paths"]}
        g["token_kind"] = "viewer_sequence_suspected" if kinds == {"viewer_sequence_suspected"} else "side_token"
        g["in_corpus_legacy_names"] = any(p["name"] in merged_names for p in g["paths"])
        g["flags"] = []
        if len(g["stems"]) > 1:
            g["flags"].append("same_image_multiple_names")
        if g["label_race"] != "post_fix":
            g["flags"].append("label_race_possible")
        if g["token_kind"] != "side_token":
            g["flags"].append("token_not_a_side")
        for stem in g["stems"]:
            k = package_key(stem, heads)
            known_keys.add(k)
            rec = _shelf(records, k)
            rec.spellings.add(display_from_stem(stem, heads))
            rec.images[g["sha256"]] = g

    # transcription PDFs
    pdf_entries = [e for e in entries if e.fetched and e.kind in ("transcription_pdf", "other_shelfmark_pdf")]
    siblings: Dict[str, Set[str]] = collections.defaultdict(set)
    for e in pdf_entries:
        m = P.SIDE_FILE_RE.match(e.name)
        if m:
            siblings[e.run_folder].add(m.group("stem"))
    rank = {name: i for i, name in enumerate(root_order)}
    is_known = side_known_checker(known_keys, heads)
    pdf_groups: Dict[str, Dict[str, Any]] = {}
    scans: Dict[str, Dict[str, Any]] = {}
    for e in pdf_entries:
        parsed = P.parse_side_file_name(e.name, siblings[e.run_folder], is_known, heads)
        evidence = bool(parsed and parsed["side_resolution"] == "sibling_file")
        scan = scan_pdf(e, evidence)
        scans[e.path] = scan
        if scan["digest"] is None:
            continue
        g = pdf_groups.setdefault(scan["digest"], {"digest": scan["digest"], "style": scan["style"],
                                                    "junk": scan["junk"], "pages": scan["pages"], "copies": []})
        g["copies"].append(e)
        if evidence and scan["style"] == "side_per_file":
            g["style"] = "side_per_file"           # one copy's sibling-confirmed side name settles the style
    for g in pdf_groups.values():
        g["copies"].sort(key=lambda e: (rank.get(e.root, len(rank)), e.name))
    say(f"pdfs: {len(pdf_entries)} files, {len(pdf_groups)} text digests")
    transcriptions: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []
    texts: List[Dict[str, Any]] = []
    side_items: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    solo_side_items: List[Dict[str, Any]] = []
    credits = {package_key(m, heads): r["payload"].get("credit_kind") for r in distinct_json
               if r["kind"] == "metadata_json" for m in r["marks"] if r["payload"].get("credit_kind")}

    def file_refs(group: Mapping[str, Any]) -> List[Dict[str, str]]:
        """Root and relative path of every copy of a PDF text.

        :param group: PDF group.
        :return: References.
        """
        return [{"root": c.root, "rel": c.rel} for c in group["copies"]]

    def names_of(group: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Side-file name parses of a group's manual copies.

        :param group: PDF group.
        :return: Parses with ``file``.
        """
        out = []
        for c in group["copies"]:
            parsed = P.parse_side_file_name(c.name, siblings[c.run_folder], is_known, heads)
            if parsed:
                out.append(dict(parsed, file=c.name))
        return out

    def other_name_mark(group: Mapping[str, Any]) -> Optional[str]:
        """Shelf mark guessed from an item-named PDF (``a_12.pdf``), only when the item is known.

        :param group: PDF group.
        :return: The mark, or None.
        """
        for c in group["copies"]:
            if c.kind == "other_shelfmark_pdf":
                mark = os.path.splitext(c.name)[0].replace("_", " ")
                if package_key(mark, heads) in known_keys:
                    return mark
        return None

    for g in pdf_groups.values():
        if g["junk"] or g["style"] == "unknown":
            continue
        canonical = g["copies"][0]
        names = names_of(g)
        if g["style"] == "header_folio":
            hint = assign_side(names)["shelfmark"] if names else other_name_mark(g)
            recs = P.ingest(canonical.path, None, lexicon, index={}, shelfmark_hint=hint)
            mark = recs[0]["shelfmark"] if recs else None
            texts.append({"group": g, "recs": recs, "key": package_key(mark, heads) if mark else "", "mark": mark,
                          "institution": names[0]["institution"] if names else None})
            continue
        flags: List[str] = []
        if names:
            chosen = assign_side(names)
            mark, side, flags = chosen["shelfmark"], chosen["side_index"], list(chosen["flags"])
            extra = {"side_resolution": chosen["side_resolution"], "side_ambiguous": chosen["side_ambiguous"],
                     "copy_suffix": chosen["copy_suffix"], "canonical_id": chosen["canonical_id"],
                     "institution": chosen["institution"], "name_candidates": chosen["name_candidates"],
                     "redownload_name_sides": chosen["redownload_name_sides"]}
        else:
            mark, side = other_name_mark(g), 1
            extra = {"side_resolution": "item_file_name" if mark else "none", "side_ambiguous": True,
                     "copy_suffix": None, "canonical_id": None, "institution": None, "name_candidates": [],
                     "redownload_name_sides": []}
            flags.append("shelfmark_from_item_file_name_unverified" if mark else "side_unknown")
        item = {"path": canonical.path, "shelfmark": mark, "side_index": side, "digest": g["digest"],
                "name_flags": flags, **extra}
        if mark:
            side_items[package_key(mark, heads)].append(item)
        else:
            solo_side_items.append(item)

    by_digest = {g["digest"]: g for g in pdf_groups.values()}
    redownloads: List[Dict[str, Any]] = []
    for key, items in side_items.items():
        recs = P.ingest_side_files(items, lexicon)
        for r in recs:
            r["flags"].update({f: 1 for f in r.pop("name_flags", [])})
        recs, copies = P.merge_side_redownloads(recs)
        redownloads.extend(dict(c, key=key) for c in copies)
        P.renumber_sides(recs)
        P.group_side_records(recs, key=lambda r, k=key: k)
        P.flag_editor_note_document(recs)
        for digest in sorted({r["digest"] for r in recs}):
            mine = [r for r in recs if r["digest"] == digest]
            mine_copies = [c for c in copies if c["copy_of_digest"] == digest]
            extra_files = [dict(ref, copy="side_redownload", suffix_side_index=c["suffix_side_index"])
                           for c in mine_copies for ref in file_refs(by_digest[c["digest"]])]
            texts.append({"group": by_digest[digest], "recs": mine, "key": key, "mark": mine[0]["shelfmark"],
                          "institution": mine[0].get("institution"), "extra_files": extra_files,
                          "redownloads": [{"digest": c["digest"], "suffix_side_index": c["suffix_side_index"],
                                           "flag": c["flag"], "files": file_refs(by_digest[c["digest"]])}
                                          for c in mine_copies]})
    for item in solo_side_items:
        recs = P.ingest_side_files([item], lexicon)
        for r in recs:
            r["flags"].update({f: 1 for f in r.pop("name_flags", [])})
            r["flags"]["shelfmark_unresolved"] = 1
        P.flag_editor_note_document(recs)
        if recs:
            texts.append({"group": by_digest[item["digest"]], "recs": recs, "key": "", "mark": None,
                          "institution": None})
    near = flag_near_duplicate_texts(texts, root_order)
    for text in texts:
        _finish_transcription(text, records, transcriptions, unresolved, credits)
    say(f"transcriptions: {len(transcriptions)} units, {len(unresolved)} unresolved texts, "
        f"{len(redownloads)} side re-downloads folded, {len(near)} near-duplicate texts")

    # catalogue, joins, bibliography, metadata
    unattached: Counter = collections.Counter()
    for r in distinct_json:
        keys = [package_key(m, heads) for m in r["marks"]]
        hit = False
        for mark, k in zip(r["marks"], keys):
            if k and k in records:
                rec = records[k]
                rec.attached[r["kind"]].append({"mark": mark, "multi": r["multi"], "payload": r["payload"],
                                                "copies": len(r["copies"]), "extracted_at": r["extracted_at"]})
                rec.spellings.add(mark)
                head = strip_institution(mark, heads)[1]
                if head:
                    rec.institutions.add(head)
                hit = True
        if not hit:
            unattached[r["kind"]] += 1

    manifest = [_manifest_record(rec, merged_index, merged_names, records) for rec in records.values()]
    manifest.sort(key=lambda m: m["key"])
    return {"manifest": manifest, "transcriptions": transcriptions, "unresolved": unresolved,
            "entries": entries, "images": groups, "pdf_groups": pdf_groups, "scans": scans,
            "json_records": json_recs, "json_distinct": distinct_json, "json_unattached": dict(unattached),
            "folder_race": race, "folder_token_kind": token_kind, "side_redownloads": redownloads,
            "near_duplicates": near}


def _text_priority(group: Mapping[str, Any], rank: Mapping[str, int]) -> Tuple[int, float, str]:
    """Order of a PDF text among near-duplicates: earlier root, then earlier download, then name.

    :param group: PDF group (``copies``: entries, canonical first).
    :param rank: Root name -> preference rank.
    :return: Sort key.
    """
    first = group["copies"][0]
    mtime = min(c.mtime for c in group["copies"] if c.root == first.root)
    return rank.get(first.root, len(rank)), mtime, first.name


def flag_near_duplicate_texts(texts: Sequence[Dict[str, Any]], root_order: Sequence[str],
                              ratio: float = P.NEAR_DUPLICATE_RATIO) -> List[Dict[str, Any]]:
    """Flag every unit of a text whose letters nearly equal an earlier text's (any shelf marks).

    Texts are ordered by :func:`_text_priority` (the PDF runner keeps the earliest download too);
    pairs come from :func:`pdf_transcriptions.near_duplicate_pairs` (SequenceMatcher ratio at or
    above ``ratio`` on the cleaned letters). The later text's units get ``near_duplicate_document``
    (a page-row exclusion) and the text records ``near_duplicate_of``. Changes are made in place.

    :param texts: Pending texts (``group``, ``recs``, ``key``).
    :param root_order: Root names in preference order.
    :param ratio: Smallest letter similarity.
    :return: One dict per pair: ``key``, ``digest`` (the flagged text), ``of_key``, ``of_digest``,
        ``ratio``, ``units`` (flagged units).
    """
    rank = {name: i for i, name in enumerate(root_order)}
    order = sorted(range(len(texts)), key=lambda i: _text_priority(texts[i]["group"], rank))
    letters = [P.text_letters(texts[i]["recs"]) for i in order]
    out: List[Dict[str, Any]] = []
    for a, b, value in P.near_duplicate_pairs(letters, ratio=ratio):
        orig, dup = texts[order[a]], texts[order[b]]
        for r in dup["recs"]:
            r["flags"]["near_duplicate_document"] = 1
        dup.setdefault("near_duplicate_of", {"key": orig["key"] or None, "digest": orig["group"]["digest"],
                                             "ratio": round(value, 5)})
        out.append({"key": dup["key"] or None, "digest": dup["group"]["digest"], "of_key": orig["key"] or None,
                    "of_digest": orig["group"]["digest"], "ratio": round(value, 5), "units": len(dup["recs"])})
    return out


def _finish_transcription(text: Mapping[str, Any], records: Dict[str, ShelfRecord],
                          transcriptions: List[Dict[str, Any]], unresolved: List[Dict[str, Any]],
                          credits: Mapping[str, str]) -> None:
    """Attach one PDF text's folio or side records to its shelf mark (or to the unresolved list).

    :param text: Pending text: ``group`` (PDF group: ``digest``, ``style``, ``pages``, ``copies``),
        ``recs`` (its folio or side records), ``key`` (package key, "" when unresolved), ``mark``
        (shelf mark as given), ``institution`` (head from the file name, or None), optional
        ``extra_files`` (folded re-downloads, flagged ``copy = "side_redownload"``),
        ``redownloads`` and ``near_duplicate_of``.
    :param records: Shelf records by key (updated).
    :param transcriptions: Unit records (appended).
    :param unresolved: Unresolved texts (appended).
    :param credits: Package key -> transcription credit kind.
    """
    group, recs, key, mark, institution = text["group"], text["recs"], text["key"], text["mark"], text["institution"]
    files = [{"root": c.root, "rel": c.rel} for c in group["copies"]] + list(text.get("extra_files") or [])
    if not recs:
        return
    style = group["style"]
    for r in recs:
        r["page_row_exclusions"] = P.page_row_exclusions(r)
        r.update(key=key or None, digest=group["digest"], style=style, files=files)
        transcriptions.append(r)
    signals = P.provenance_signals(recs, style, credits.get(key) if key else None)
    flags = sorted({f for r in recs for f in r["flags"]})
    units = [{"label": r["folio"], "side_index": r.get("side_index"), "suffix_side_index": r.get("suffix_side_index"),
              "heading_shelfmark": r.get("heading_shelfmark"), "letters": r["letters"],
              "cleaned_lines": len(r["cleaned_lines"]), "columns": len(r.get("columns") or []),
              "flags": sorted(r["flags"]), "page_row_exclusions": r["page_row_exclusions"]} for r in recs]
    entry = {"digest": group["digest"], "style": style, "files": files, "pages": group["pages"],
             "shelfmark_given": mark, "shelfmark_source": recs[0].get("shelfmark_source"),
             "institution_line": recs[0].get("institution"), "title": recs[0].get("title"), "units": units,
             "letters": sum(int(r["letters"]) for r in recs), "provenance": signals, "flags": flags,
             "usable_units": sum(1 for u in units if not u["page_row_exclusions"]),
             "side_redownloads": list(text.get("redownloads") or []),
             "near_duplicate_of": text.get("near_duplicate_of")}
    if not key:
        unresolved.append(entry)
        return
    rec = _shelf(records, key)
    rec.transcriptions[group["digest"]] = entry
    if mark:
        rec.spellings.add(mark)
    if institution:
        rec.institutions.add(institution)
    if recs[0].get("institution"):
        rec.institutions.add(str(recs[0]["institution"]))


def _manifest_record(rec: ShelfRecord, merged_index: Mapping[str, List[Dict[str, Any]]],
                     merged_names: Mapping[str, List[str]], records: Mapping[str, ShelfRecord]) -> Dict[str, Any]:
    """The manifest line of one shelf mark.

    A record with a transcription but without images of its own borrows the images of its parent
    mark (the key without its last number: ``Evr. IIA 223.2`` -> ``Yevr_II_A_223``, whose image
    names drop the sub-number), flagged ``images_from_parent_shelfmark_unverified``.

    :param rec: Shelf record.
    :param merged_index: Corpus key index.
    :param merged_names: Corpus legacy image name index.
    :param records: All shelf records (for the parent lookup).
    :return: The manifest dict.
    """
    flags = set(rec.flags)
    images = list(rec.images.values())
    parent_images: List[Dict[str, Any]] = []
    if not images and rec.transcriptions and "|" in rec.key:
        parent = records.get(rec.key.rsplit("|", 1)[0])
        if parent and parent.images:
            parent_images = list(parent.images.values())
            flags.add("images_from_parent_shelfmark_unverified")
    ktiv = ktiv_status(rec.key, rec.institutions, merged_index)
    n_images = len(images) + len(parent_images)
    policy = image_policy(ktiv, n_images)
    if policy == POLICY_LEGACY_ONLY:
        flags.add("needs_label_check")
        if rec.transcriptions:
            flags.add("needs_content_matching")
    if any("label_race_possible" in g["flags"] for g in images + parent_images):
        flags.add("label_race_possible")
    if rec.transcriptions and n_images == 0 and not ktiv["ktiv_images_held"] and not ktiv["ktiv_iiif_manifest"] \
            and not ktiv["corpus_legacy_images"] and not ktiv["bodleian_images"]:
        flags.add("transcription_without_image_anywhere")
    spellings = sorted(rec.spellings)
    catalogue = rec.attached.get("cataloging_json", [])
    display = next((x["mark"] for x in catalogue if not x["multi"]), None) or \
        next((t["shelfmark_given"] for t in rec.transcriptions.values() if t["shelfmark_given"]), None) or \
        (spellings[0] if spellings else rec.key)

    def image_ref(g: Mapping[str, Any]) -> Dict[str, Any]:
        """Manifest form of an image group.

        :param g: Image group.
        :return: The reference.
        """
        return {"sha256": g["sha256"], "size": g["size"], "canonical": dict(g["canonical"]),
                "names": sorted({p["name"] for p in g["paths"]}), "tokens": sorted({p["token"] for p in g["paths"]}),
                "token_kind": g["token_kind"], "label_race": g["label_race"],
                "run_folders": sorted({p["run_folder"] for p in g["paths"]}),
                "paths": [{"root": p["root"], "rel": p["rel"]} for p in g["paths"]],
                "in_corpus_legacy_names": g["in_corpus_legacy_names"], "flags": g["flags"]}

    sides = collections.Counter(t[-1] for g in images for t in {p["token"] for p in g["paths"]})
    return {
        "key": rec.key, "shelfmark": display, "spellings": spellings, "institutions": sorted(rec.institutions),
        "images": [image_ref(g) for g in sorted(images, key=lambda g: (g["canonical"]["root"],
                                                                        g["canonical"]["rel"]))],
        "parent_images": [image_ref(g) for g in parent_images], "image_count": len(images),
        "image_tokens": dict(sides), "image_folio_pairing": "unverified",
        "transcriptions": sorted(rec.transcriptions.values(), key=lambda t: t["digest"]),
        "side_indices": sorted({u["side_index"] for t in rec.transcriptions.values() if t["style"] == "side_per_file"
                                for u in t["units"]}),
        "catalogue": catalogue, "joins": rec.attached.get("joins_json", []),
        "bibliography": rec.attached.get("bibliography_json", []),
        "transcription_metadata": rec.attached.get("metadata_json", []),
        "ktiv": ktiv, "image_policy": policy, "flags": sorted(flags),
    }
