"""Stage 1 of the novel-rows build: every photograph known for each candidate, listed, verified and fetched.

Reads the stage-0 candidates and completes each candidate's ``images`` list from the sources the corpus already has:

* ``pgp`` -- the PGP export's IIIF manifests (``documents.csv`` ``iiif_urls``), expanded canvas by canvas, plus the
  images of every merged record the candidate names (legacy-store photographs, the KTIV image list, Bodleian
  masters, the record's own IIIF manifest), with the durable image corrections (move / detach) applied first;
* ``fjp`` / ``fjp_scrape`` -- the legacy-store photographs the candidate lists, verified on disk;
* ``ktiv`` -- the page image inside the manuscript's KTIV image zip, extracted once into a cache directory;
* ``literature`` / ``lit_run2`` -- the candidate's own image, the merged images of its records, the run-2 local
  library downloads and the legacy-store files of its canonical ids;
* ``apd_v3`` / ``aodeh_v2_1`` -- the built row's image (set at stage 0), verified.

A ``pgp`` / ``literature`` / ``lit_run2`` candidate left with no image at all gets the Cambridge library manifest of
its Cambridge canonical ids (the shelf-mark facade's ``cudl_manifest``).

Fetch cap: a candidate with more than ``--max-per-candidate`` photographs known only by URL fetches at most that many,
chosen by file name through the facade (the stem names the candidate's own mark, then a side of it); the rest become
``box_listing`` (kept listed, never fetched), and a candidate none of whose URLs matches is flagged
``box_listing_unmatched`` for review. This is a fetch selection only: sides are still placed by reads.

Local files get ``sha256`` (of the stored bytes), ``phash`` and the upright size; ``fetch_missing`` downloads what
is only known by URL (a HEAD check first on the image bucket), keeps the source bytes' ``sha256``, turns the image
upright and stores it once per distinct ``sha256``. Outcomes of every URL are appended to a state log so a second run
fetches nothing it already has.

``phash`` is the novelty ledger's 64-bit DCT hash (the ``imagehash.phash`` recipe re-implemented without that
package, with the ledger's own DCT path), always computed on the decoded stored bytes before any EXIF rotation, so
it compares directly with the ledger's ``phash`` field.

Run from the repository root::

    .venv/bin/python -m src.finetuning.qwen_hebrew.novel_build.images --inventory [--limit 40]
    .venv/bin/python -m src.finetuning.qwen_hebrew.novel_build.images --fetch --sources pgp,literature
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import re
import threading
import time
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageOps

from src.datasets.merging.image_corrections import image_name, load_image_corrections
from src.datasets.merging.merge_shelfmarks import apply_image_corrections, correction_record_ids
from src.datasets.shelfmarks import cudl_manifest, loose_key, split_id
from src.finetuning.qwen_hebrew.build_ktiv_dataset import find_zip_member
from src.finetuning.qwen_hebrew.build_legacy_store_rows import sides_conflict
from src.finetuning.qwen_hebrew.novel_build import paths
from src.finetuning.qwen_hebrew.novel_build.schema import Candidate, ImageRef, read_jsonl, write_jsonl

# ----------------------------------------------------------------------------------------------------- constants

#: A photograph whose long side (upright) is under this many pixels is kept but marked ``too_small``.
MIN_LONG_SIDE = 800
#: JPEG quality of re-encoded (rotated / non-JPEG) images.
JPEG_QUALITY = 92
#: IIIF size parameter used for the Cambridge image service (the corpus' convention).
CUDL_IMAGE_SIZE = "3245,"
#: IIIF size parameter for every other image service.
DEFAULT_IMAGE_SIZE = "full"
#: Base URL of the legacy store's photographs (a bare ``images.fjp`` file name lives under it).
LEGACY_BUCKET_BASE = os.environ.get("NOVEL_BUILD_LEGACY_BUCKET_BASE",
                                    "https://storage.googleapis.com/cairo-genizah-es-json/images")
#: Hosts of the image bucket: their URLs get a HEAD check before any GET.
BUCKET_HOSTS = frozenset({urlparse(LEGACY_BUCKET_BASE).hostname or "", "storage.googleapis.com"})
#: Top-level folders of the bucket that mirror another source, and the origin of their files.
BUCKET_FOLDER_ORIGINS = {"KTIV": "iiif_nli", "BODLEIAN": "bodleian"}
#: Image-service and viewer hosts by origin (manifests and direct image URLs are both classified by host).
CUDL_HOSTS = frozenset({"cudl.lib.cam.ac.uk", "images.lib.cam.ac.uk"})
PGP_HOSTS = frozenset({"princetongenizalab.github.io", "figgy.princeton.edu"})
NLI_HOSTS = frozenset({"iiif.nli.org.il"})
BODLEIAN_HOSTS = frozenset({"iiif.bodleian.ox.ac.uk", "digital.bodleian.ox.ac.uk"})
#: Origins this module writes; a stage-0 origin outside this set is re-derived from the URL or path.
KNOWN_ORIGINS = frozenset({"ktiv_zip", "iiif_nli", "iiif_cudl", "bodleian", "iiif_pgp", "iiif_other", "cudl_local",
                           "fjp_local", "fjp_bucket", "pgpid_mirror", "apd_local", "aodeh_local", "local_other"})
#: Sources whose candidates also list the images of their merged records (the spec names ``pgp``; the two
#: literature sources are added so that both sides of a fragment are listed, not only the first image URL).
MERGED_SOURCES = frozenset({"pgp", "literature", "lit_run2"})
#: Sources whose image is given by the built row (verified, never extended).
PASS_THROUGH_SOURCES = frozenset({"apd_v3", "aodeh_v2_1"})
#: Cambridge canonical-id prefixes and the mark prefix the facade's ``cudl_manifest`` reads; a candidate of
#: :data:`MERGED_SOURCES` left with no image at all is given the library manifest of its Cambridge ids.
CUDL_ID_PREFIXES = (("Cambridge_CUL_T_S_", "T-S "), ("Cambridge_CUL_Or_", "CUL Or."),
                    ("Cambridge_CUL_Add_", "CUL Add."), ("Cambridge_CUL_ADD_", "CUL Add."))
#: Statuses of a usable local file.
USABLE_STATUSES = ("present", "fetched")
#: Default per-candidate fetch cap (``--max-per-candidate``); URL-only photographs beyond it, or whose file name does
#: not name the candidate's own mark, become ``box_listing`` (kept listed, never fetched).
MAX_PER_CANDIDATE = 8
_STEM_SEPARATORS_RE = re.compile(r"[_\-\s]+")
_LEGACY_SIDE_RE = re.compile(r"^(\d{0,3})([rv])$", re.I)
#: HTTP statuses retried as transient.
TRANSIENT_HTTP = frozenset({408, 425, 429, 500, 502, 503, 504})
#: First back-off (seconds) between attempts; doubled per attempt.
RETRY_BACKOFF_S = 2.0
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".jp2", ".gif", ".webp")
_IIIF_IMAGE_RE = re.compile(r"/full/[^/]+/\d+(?:\.\d+)?/(?:default|native|color|gray|grey)\.\w+$", re.I)
_MIRROR_RE = re.compile(r"^\d+_image\.\w+$")
_SIDE_TOKEN_RE = re.compile(r"(?:^|_)(\d{0,3}[rv])$", re.I)
_MERGED_ID_RE = re.compile(r'^\{\s*"(?:_id|canonical_id)"\s*:\s*"((?:[^"\\]|\\.)*)"')
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_HAS_SCIPY = importlib.util.find_spec("scipy") is not None

DEFAULT_PGP_EXPORT_DIR = paths.REPO_ROOT / "artifacts/pgp_export_b2343a17"
DEFAULT_MERGED = paths.REPO_ROOT / "src/datasets/raw_data/cairo_genizah/merged/merged_shelfmarks.jsonl"
DEFAULT_CORRECTIONS = paths.REPO_ROOT / "src/datasets/raw_data/cairo_genizah/merged/image_corrections.jsonl"
DEFAULT_LEGACY_MANIFESTS = (paths.DATASETS_ROOT / "legacy_store_diplomatic/images_manifest_v2.jsonl",
                            paths.DATASETS_ROOT / "legacy_store_diplomatic/images_manifest_chain.jsonl")
DEFAULT_RUN2_LOCAL_IMAGES = paths.REPO_ROOT / "logs/run2_data_2026-10-27/cache/local_images.json"
DEFAULT_KTIV_DIR = paths.REPO_ROOT / "src/datasets/raw_data/cairo_genizah/ktiv"
DEFAULT_BODLEIAN_DIR = paths.REPO_ROOT / "src/datasets/raw_data/cairo_genizah/bodleian/images"
#: Work-dir relative locations owned by this stage.
MANIFESTS_DIR = "manifests"
IMAGES_DIR = "images"
MERGED_INDEX_FILE = "merged_images_index.json"
LOCAL_KEYS_FILE = "local_image_keys.json"
FETCH_STATE_FILE = "fetch_state.jsonl"
INVENTORY_FILE = "stage1_inventory.md"
#: Key of this stage's provenance inside ``Candidate.notes``.
NOTES_KEY = "images_stage1"

ManifestFetcher = Callable[[str], Dict[str, Any]]
SleepFn = Callable[[float], None]


class PermanentFetchError(RuntimeError):
    """A fetch failure that retrying cannot fix (an HTTP 4xx other than 408/425/429, a body that is not JSON)."""


class TransientStatusError(RuntimeError):
    """A HEAD answered with a status worth retrying (429, 5xx, ...)."""


# --------------------------------------------------------------------------------------- perceptual hash (ledger)


def _dct2(grid: np.ndarray) -> np.ndarray:
    """Orthonormal 2-D DCT-II, by the same code path as the ledger (SciPy when installed, else the matrix form).

    :param grid: Square float64 array.
    :return: DCT coefficients.
    """
    if _HAS_SCIPY:
        from scipy.fft import dct
        return dct(dct(grid, axis=0, norm="ortho"), axis=1, norm="ortho")
    n = grid.shape[0]
    k = np.arange(n, dtype=np.float64)
    c = np.ascontiguousarray(np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n)) * np.sqrt(2.0 / n))
    c[0, :] = c[0, :] / np.sqrt(2.0)
    return np.dot(np.dot(c, grid), c.T)


def phash_image(img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """64-bit DCT perceptual hash, identical to the novelty ledger's ``phash`` (comparable hex strings).

    Grey, LANCZOS-resize to 32 x 32, 2-D DCT, the 8 x 8 low frequencies against their median, bits row-major and
    most significant first.

    :param img: Decoded image (not EXIF-rotated, as the ledger hashed it).
    :param hash_size: Side of the hash grid (8 -> 64 bits).
    :param highfreq_factor: Resize factor before the DCT.
    :return: 16 hex characters.
    """
    size = hash_size * highfreq_factor
    grey = img.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    coeffs = _dct2(np.ascontiguousarray(np.asarray(grey, dtype=np.float64)))
    low = coeffs[:hash_size, :hash_size]
    bits = (low > np.median(low)).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def phash_distance(a: Optional[str], b: Optional[str]) -> Optional[int]:
    """Hamming distance of two perceptual hashes.

    :param a: Hex hash or None.
    :param b: Hex hash or None.
    :return: Differing bits, or None when either hash is missing.
    """
    if not a or not b:
        return None
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def upright_size(img: Image.Image) -> Tuple[int, int]:
    """Width and height of an image as shown after its EXIF orientation is applied.

    :param img: Opened image.
    :return: ``(width, height)``.
    """
    orientation = int(img.getexif().get(0x0112, 1) or 1)
    return (img.height, img.width) if orientation in (5, 6, 7, 8) else (img.width, img.height)


def image_keys(data: bytes) -> Dict[str, Any]:
    """Identity keys of encoded image bytes: sha256 of the bytes, phash and upright size of the decoded image.

    :param data: Encoded image bytes.
    :return: ``{"sha256", "phash", "width", "height", "orientation"}``.
    :raises OSError: When the bytes do not decode (``PIL.UnidentifiedImageError`` included).
    """
    with Image.open(io.BytesIO(data)) as img:
        img.load()
        width, height = upright_size(img)
        return {"sha256": hashlib.sha256(data).hexdigest(), "phash": phash_image(img), "width": width,
                "height": height, "orientation": int(img.getexif().get(0x0112, 1) or 1)}


def write_upright(data: bytes, dest: Path) -> bool:
    """Write a photograph upright at ``dest``.

    An upright RGB or greyscale JPEG is written verbatim (the file's sha256 is then the source's); anything else
    (EXIF rotation, another format, another mode) is turned upright with ``ImageOps.exif_transpose``, converted to
    RGB and re-encoded at :data:`JPEG_QUALITY` (the pattern of ``build_pgp_editions.fetch_image``).

    :param data: Source bytes.
    :param dest: Target ``.jpg`` path (parents are created).
    :return: True when the image was re-encoded.
    :raises OSError: When the bytes do not decode or the file cannot be written.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(data)) as img:
        orientation = int(img.getexif().get(0x0112, 1) or 1)
        verbatim = img.format == "JPEG" and orientation in (0, 1) and img.mode in ("RGB", "L")
        if verbatim:
            dest.write_bytes(data)
        else:
            ImageOps.exif_transpose(img).convert("RGB").save(dest, "JPEG", quality=JPEG_QUALITY)
    return not verbatim


def store_image(data: bytes, dest: Path) -> Dict[str, Any]:
    """Write a photograph upright at ``dest`` (:func:`write_upright`) and return its keys.

    :param data: Source bytes.
    :param dest: Target ``.jpg`` path.
    :return: :func:`image_keys` plus ``reencoded`` and ``source_sha256``; ``sha256`` is the WRITTEN file's digest
        when the bytes were re-encoded, else the source's (identical bytes).
    :raises OSError: When the bytes do not decode or the file cannot be written.
    """
    keys = image_keys(data)
    reencoded = write_upright(data, dest)
    out = {**keys, "reencoded": reencoded, "source_sha256": keys["sha256"]}
    if reencoded:
        # ``sha256`` names the file every later stage hashes; the source bytes' digest stays as ``source_sha256``
        # (the ledger's image identity), and ``phash`` is still the source's.
        out["sha256"] = hashlib.sha256(dest.read_bytes()).hexdigest()
    return out


# ------------------------------------------------------------------------------------------------- URL helpers


def split_urls(cell: Optional[str]) -> List[str]:
    """Split a multi-valued URL cell of the PGP export (``;``- or whitespace-separated).

    :param cell: Cell text.
    :return: Non-empty URLs in order, without repeats.
    """
    return list(dict.fromkeys(u.strip() for u in re.split(r"\s*;\s*|\s+", cell or "") if u.strip()))


def host_of(url: Optional[str]) -> str:
    """Lower-case host of a URL.

    :param url: URL or None.
    :return: Host, ``""`` when absent.
    """
    return (urlparse(url).hostname or "").lower() if url else ""


def is_bucket_url(url: Optional[str]) -> bool:
    """Whether a URL points into the image bucket (HEAD-checked before a GET).

    :param url: URL or None.
    :return: True for bucket URLs.
    """
    return host_of(url) in BUCKET_HOSTS


def is_image_url(url: str) -> bool:
    """Whether a URL names an image file or an IIIF image request (anything else is treated as a manifest).

    :param url: URL.
    :return: True for an image URL.
    """
    path = urlparse(url).path
    return path.lower().endswith(IMAGE_SUFFIXES) or bool(_IIIF_IMAGE_RE.search(path))


def is_pgpid_mirror(name: str) -> bool:
    """Whether a legacy-store file name is the one-per-document ``<pgpid>_image`` copy.

    :param name: File name.
    :return: True for the mirror copy.
    """
    return bool(_MIRROR_RE.match(name))


def legacy_url(entry: str) -> str:
    """Full bucket URL of an ``images.fjp`` entry (a bare file name, or already a URL).

    :param entry: Entry as stored in the merged corpus.
    :return: The URL.
    """
    return entry if str(entry).startswith("http") else f"{LEGACY_BUCKET_BASE}/{image_name(entry)}"


def origin_of_url(url: str) -> str:
    """Origin of a photograph (or of a manifest's canvases) from its URL.

    :param url: Image or manifest URL.
    :return: One of :data:`KNOWN_ORIGINS`.
    """
    host = host_of(url)
    if host in BUCKET_HOSTS:
        parts = [p for p in urlparse(url).path.split("/") if p]
        folder = parts[1] if len(parts) > 2 else ""
        if folder in BUCKET_FOLDER_ORIGINS:
            return BUCKET_FOLDER_ORIGINS[folder]
        return "pgpid_mirror" if is_pgpid_mirror(image_name(url)) else "fjp_bucket"
    if host in CUDL_HOSTS:
        return "iiif_cudl"
    if host in NLI_HOSTS:
        return "iiif_nli"
    if host in BODLEIAN_HOSTS:
        return "bodleian"
    if host in PGP_HOSTS:
        return "iiif_pgp"
    return "iiif_other"


def side_token(name: str) -> Optional[str]:
    """Trailing side token of a photograph's file name (``X_1_1r.jpg`` -> ``1r``); review evidence only.

    :param name: File name.
    :return: The token or None.
    """
    match = _SIDE_TOKEN_RE.search(Path(name).stem)
    return match.group(1) if match else None


def safe_name(text: str) -> str:
    """File-system-safe form of an id (any run of characters outside ``[A-Za-z0-9._-]`` becomes ``_``).

    :param text: Id.
    :return: Safe name (``_`` when nothing is left).
    """
    return _UNSAFE_RE.sub("_", text).strip("._") or "_"


def group_name(cand: Candidate) -> str:
    """Image directory of a candidate: its first canonical id, else its row id, made file-system safe.

    :param cand: Candidate.
    :return: Directory name.
    """
    return safe_name(cand.canonical_ids[0] if cand.canonical_ids else cand.row_id)


def add_note(ref: ImageRef, note: str) -> None:
    """Append a note to an image reference once.

    :param ref: Image reference (mutated).
    :param note: Note text.
    """
    if note and note not in ref.note.split("; "):
        ref.note = f"{ref.note}; {note}" if ref.note else note


def _atomic_write_text(path: Path, text: str) -> None:
    """Write a text file through a temporary sibling and a rename, so a reader never sees half a file.

    :param path: Target.
    :param text: Content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------------- IIIF


def _label_text(label: Any) -> str:
    """Plain text of a IIIF label (v2 string / list / ``@value`` object, v3 language map).

    :param label: Label value.
    :return: Text (``""`` when absent).
    """
    if label is None:
        return ""
    if isinstance(label, str):
        return label.strip()
    if isinstance(label, list):
        return " ".join(t for t in (_label_text(x) for x in label) if t)
    if isinstance(label, dict):
        if "@value" in label:
            return _label_text(label["@value"])
        return " ".join(t for t in (_label_text(v) for v in label.values()) if t)
    return str(label).strip()


def _service_id(service: Any) -> Optional[str]:
    """The id of the first image service in a ``service`` value (object or list; ``@id`` or ``id``).

    :param service: Service value.
    :return: Service URL without a trailing slash, or None.
    """
    if isinstance(service, list):
        for item in service:
            sid = _service_id(item)
            if sid:
                return sid
        return None
    if isinstance(service, dict):
        sid = service.get("@id") or service.get("id")
        return str(sid).rstrip("/") if sid else None
    return None


def canvas_image_url(service: Optional[str], fallback: Optional[str]) -> Optional[str]:
    """Image URL of a canvas: the service's full image (the Cambridge service at its corpus width), else the
    resource's own URL.

    :param service: Image service id or None.
    :param fallback: The image resource's own id.
    :return: URL or None.
    """
    if service:
        size = CUDL_IMAGE_SIZE if host_of(service) in CUDL_HOSTS else DEFAULT_IMAGE_SIZE
        return f"{service}/full/{size}/0/default.jpg"
    return str(fallback) if fallback else None


def _v3_body(canvas: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """The painting body of a Presentation 3 canvas (first annotation of the first page; a Choice's first item).

    :param canvas: Canvas object.
    :return: Body object or None.
    """
    pages = canvas.get("items") or []
    annos = (pages[0].get("items") or []) if pages and isinstance(pages[0], dict) else []
    body = annos[0].get("body") if annos and isinstance(annos[0], dict) else None
    if isinstance(body, list):
        body = body[0] if body else None
    if isinstance(body, dict) and body.get("type") == "Choice":
        body = (body.get("items") or [None])[0]
    return body if isinstance(body, dict) else None


def manifest_canvases(manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Canvases of a IIIF Presentation 2 (``sequences[0].canvases``) or 3 (``items``) manifest, in order.

    v2 image: ``images[0].resource.service['@id']``; v3 image: ``items[0].items[0].body.service[0].id``; without a
    service the resource's own id is the image URL.

    :param manifest: Parsed manifest.
    :return: ``[{"canvas_index" (0-based), "label", "service", "image_url"}]``; a canvas without an image has
        ``image_url`` None.
    """
    out: List[Dict[str, Any]] = []
    if manifest.get("sequences"):
        sequence = manifest["sequences"][0] or {}
        for index, canvas in enumerate(sequence.get("canvases") or []):
            images = canvas.get("images") or []
            resource = (images[0].get("resource") or {}) if images else {}
            service = _service_id(resource.get("service"))
            out.append({"canvas_index": index, "label": _label_text(canvas.get("label")), "service": service,
                        "image_url": canvas_image_url(service, resource.get("@id") or resource.get("id"))})
        return out
    for index, canvas in enumerate(manifest.get("items") or []):
        if not isinstance(canvas, dict) or canvas.get("type", "Canvas") != "Canvas":
            continue
        body = _v3_body(canvas) or {}
        service = _service_id(body.get("service"))
        out.append({"canvas_index": index, "label": _label_text(canvas.get("label")), "service": service,
                    "image_url": canvas_image_url(service, body.get("id") or body.get("@id"))})
    return out


def manifest_cache_path(manifests_dir: Path, url: str) -> Path:
    """Cache file of a manifest: ``<sha1(url)>.json``.

    :param manifests_dir: Cache directory.
    :param url: Manifest URL.
    :return: Path.
    """
    return manifests_dir / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()}.json"


def cached_manifest(manifests_dir: Path, url: str) -> Optional[Dict[str, Any]]:
    """A cached manifest, if any.

    :param manifests_dir: Cache directory.
    :param url: Manifest URL.
    :return: Parsed manifest or None.
    """
    path = manifest_cache_path(manifests_dir, url)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


class HostPacer:
    """Keeps at least ``delay`` seconds between two requests to one host, across threads."""

    def __init__(self, delay: float, sleep: SleepFn = time.sleep, clock: Callable[[], float] = time.monotonic) -> None:
        """Create a pacer.

        :param delay: Minimum seconds between requests to one host.
        :param sleep: Sleep function (tests pass a no-op).
        :param clock: Monotonic clock.
        """
        self.delay = max(0.0, float(delay))
        self._sleep = sleep
        self._clock = clock
        self._next: Dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, url: str) -> None:
        """Block until a request to the URL's host is allowed, and book the next slot.

        :param url: URL about to be requested.
        """
        if self.delay <= 0:
            return
        host = host_of(url)
        with self._lock:
            now = self._clock()
            start = max(now, self._next.get(host, 0.0))
            self._next[host] = start + self.delay
        if start > now:
            self._sleep(start - now)


def call_with_retries(fn: Callable[[str], Any], url: str, *, retries: int, pacer: HostPacer,
                      sleep: SleepFn = time.sleep) -> Tuple[Any, Optional[str]]:
    """Call an injected HTTP callable with retries; its failure is returned as text, never raised.

    :param fn: ``fn(url)``; raises on failure (:class:`PermanentFetchError` stops the retries at once).
    :param url: URL.
    :param retries: Extra attempts after the first.
    :param pacer: Per-host pacer.
    :param sleep: Sleep function for the back-off.
    :return: ``(result, None)`` or ``(None, error text)``.
    """
    error = ""
    for attempt in range(retries + 1):
        pacer.wait(url)
        try:
            return fn(url), None
        except PermanentFetchError as exc:
            return None, f"{type(exc).__name__}: {exc}"[:200]
        except Exception as exc:  # noqa: BLE001 - any failure of an injected HTTP callable is recorded as a status
            error = f"{type(exc).__name__}: {exc}"[:200]
            if attempt < retries:
                sleep(RETRY_BACKOFF_S * (2 ** attempt))
    return None, error


def fetch_manifests(urls: Sequence[str], manifests_dir: Path, fetch_manifest: ManifestFetcher, *, workers: int = 2,
                    delay: float = 0.3, retries: int = 2, sleep: SleepFn = time.sleep) -> Dict[str, str]:
    """Fetch the manifests not cached yet and cache them as ``<sha1(url)>.json``.

    Network failures are recorded (returned), not raised.

    :param urls: Manifest URLs.
    :param manifests_dir: Cache directory.
    :param fetch_manifest: ``fetch_manifest(url) -> parsed JSON``; raises on failure.
    :param workers: Threads.
    :param delay: Minimum seconds between requests to one host.
    :param retries: Extra attempts after the first.
    :param sleep: Sleep function.
    :return: ``{url: error}`` for the manifests that could not be fetched.
    """
    todo = [u for u in dict.fromkeys(urls) if not manifest_cache_path(manifests_dir, u).is_file()]
    pacer = HostPacer(delay, sleep)
    errors: Dict[str, str] = {}

    def one(url: str) -> None:
        """Fetch and cache one manifest.

        :param url: Manifest URL.
        """
        data, err = call_with_retries(fetch_manifest, url, retries=retries, pacer=pacer, sleep=sleep)
        if err is not None or not isinstance(data, dict):
            errors[url] = err or "not_a_json_object"
            return
        _atomic_write_text(manifest_cache_path(manifests_dir, url), json.dumps(data, ensure_ascii=False))

    if todo:
        print(f"  manifests: {len(todo):,} to fetch", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(one, _interleave_hosts(todo)))
    return errors


def refs_from_manifest(url: str, manifest: Mapping[str, Any]) -> List[ImageRef]:
    """One image reference per canvas of a manifest (origin by the manifest's host).

    :param url: Manifest URL.
    :param manifest: Parsed manifest.
    :return: References with ``url``, ``canvas_index`` and ``label``; status ``listed``.
    """
    origin = origin_of_url(url)
    refs = []
    for canvas in manifest_canvases(manifest):
        if canvas["image_url"]:
            refs.append(ImageRef(image_id="", origin=origin, url=canvas["image_url"],
                                 canvas_index=canvas["canvas_index"], label=canvas["label"] or None,
                                 status="listed", note=f"manifest {url}"))
    return refs


def url_items(urls: Iterable[str]) -> List[Union[ImageRef, str]]:
    """Listing items of IIIF URL fields: an image URL becomes a reference, anything else stays a manifest URL.

    :param urls: URLs from a manifest field (some exports put a direct image URL there).
    :return: References and manifest URLs, in order.
    """
    return [ImageRef(image_id="", origin=origin_of_url(u), url=u, status="listed") if is_image_url(u) else u
            for u in urls]


def _interleave_hosts(urls: Sequence[str]) -> List[str]:
    """Order URLs round-robin over their hosts, so parallel workers do not queue on one host.

    :param urls: URLs.
    :return: The same URLs, interleaved by host.
    """
    by_host: Dict[str, List[str]] = defaultdict(list)
    for url in urls:
        by_host[host_of(url)].append(url)
    queues = list(by_host.values())
    out: List[str] = []
    for i in range(max((len(q) for q in queues), default=0)):
        out.extend(q[i] for q in queues if i < len(q))
    return out


# ------------------------------------------------------------------------------------------------ source indexes


def load_pgp_manifest_urls(pgp_export_dir: Path) -> Dict[str, List[str]]:
    """The PGP export's IIIF URLs per pgpid (``documents.csv`` ``iiif_urls``).

    :param pgp_export_dir: Export directory holding ``documents.csv``.
    :return: ``{pgpid: [url]}``.
    """
    out: Dict[str, List[str]] = {}
    with open(pgp_export_dir / "documents.csv", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            urls = split_urls(row.get("iiif_urls"))
            if urls:
                out[str(row["pgpid"]).strip()] = urls
    return out


@dataclass
class LegacyIndex:
    """Legacy-store photographs on disk, from the store's image manifest(s).

    :param by_file: ``{file name: manifest record}`` (first manifest wins).
    :param by_cid: ``{canonical id: [records]}`` in image order (detached files have no canonical id).
    :param paths: Every listed local path.
    """

    by_file: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_cid: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    paths: Set[str] = field(default_factory=set)


def load_legacy_manifest(manifests: Union[Path, Sequence[Path]]) -> LegacyIndex:
    """Index the legacy store's image manifest(s) (``canonical_id``, ``file_name``, ``path``, ``sha256``, ``width``,
    ``height``, ``side_token``, ``slip_side``, ``slip_outcome``, ...). A missing file is skipped with a notice.

    :param manifests: One manifest path or several.
    :return: The index.
    """
    index = LegacyIndex()
    by_cid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for path in ([manifests] if isinstance(manifests, (str, Path)) else list(manifests)):
        path = Path(path)
        if not path.is_file():
            print(f"  legacy manifest absent, skipped: {path.name}", flush=True)
            continue
        for rec in read_jsonl(path):
            name = rec.get("file_name") or (image_name(rec["url"]) if rec.get("url") else None)
            if not name:
                continue
            index.by_file.setdefault(name, rec)
            if rec.get("path"):
                index.paths.add(str(rec["path"]))
            if rec.get("canonical_id"):
                by_cid[rec["canonical_id"]].append(rec)
    index.by_cid = {cid: sorted(recs, key=lambda r: (r.get("image_index") is None, r.get("image_index") or 0))
                    for cid, recs in by_cid.items()}
    return index


def load_run2_local(path: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    """Run-2 local library downloads per canonical id, and how many ids share each download folder.

    :param path: ``local_images.json`` (``{canonical_id: [{path, canvas_label, manifest_id}]}``); absent -> empty.
    :return: ``(by canonical id, {folder: number of ids})``.
    """
    if not path.is_file():
        print(f"  run-2 local image index absent, skipped: {path.name}", flush=True)
        return {}, {}
    data: Dict[str, List[Dict[str, Any]]] = json.loads(path.read_text(encoding="utf-8"))
    folders: Dict[str, Set[str]] = defaultdict(set)
    for cid, recs in data.items():
        for rec in recs:
            folders[str(Path(rec["path"]).parent)].add(cid)
    return data, {folder: len(ids) for folder, ids in folders.items()}


def merged_images_index(merged_path: Path, ids: Iterable[str], cache_path: Path) -> Dict[str, Dict[str, Any]]:
    """Image facts of the merged records with the given ids, from ONE streaming pass, cached as JSON.

    Each kept record is ``{"images": <the record's images block>, "pgp_iiif": <sources.pgp.fragment.iiif_url>,
    "pgpids": [...]}``. The cache is reused while it names the same merged file (path, size, mtime) and covers every
    requested id; otherwise one pass over the file rebuilds it for the union of the old and new ids. Lines are
    pre-filtered on their leading id key, so only wanted records are parsed.

    :param merged_path: ``merged_shelfmarks.jsonl``.
    :param ids: Canonical ids wanted.
    :param cache_path: Cache file.
    :return: ``{canonical id: facts}`` (ids absent from the merged file are simply not keys).
    """
    wanted = set(ids)
    stat = merged_path.stat()
    signature = {"merged": str(merged_path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("signature") == signature:
            if wanted <= set(cached.get("ids") or []):
                return cached["records"]
            wanted |= set(cached.get("ids") or [])
    t0 = time.time()
    records: Dict[str, Dict[str, Any]] = {}
    with open(merged_path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if i % 20000 == 0:
                print(f"  merged: {i:,} lines, {len(records):,} kept", flush=True)
            if not line.strip():
                continue
            match = _MERGED_ID_RE.match(line)
            if match is not None:
                raw = match.group(1)
                if (json.loads(f'"{raw}"') if "\\" in raw else raw) not in wanted:
                    continue
            rec = json.loads(line)
            cid = rec.get("_id") or rec.get("canonical_id")
            if cid not in wanted:
                continue
            frag = ((rec.get("sources") or {}).get("pgp") or {}).get("fragment") or {}
            records[cid] = {"images": rec.get("images") or {}, "pgp_iiif": frag.get("iiif_url") or "",
                            "pgpids": rec.get("pgpids") or []}
    print(f"  merged index: {len(records):,}/{len(wanted):,} ids found in {time.time() - t0:.0f}s", flush=True)
    _atomic_write_text(cache_path, json.dumps({"signature": signature, "ids": sorted(wanted), "records": records},
                                              ensure_ascii=False))
    return records


def corrected_records(index: Mapping[str, Dict[str, Any]], corrections: Optional[Mapping[str, Mapping[str, Any]]]
                      ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """A copy of the merged image facts with the durable image corrections applied (moves and detaches; side-only
    corrections set ``images.fjp_sides``), by :func:`merge_shelfmarks.apply_image_corrections`.

    :param index: Output of :func:`merged_images_index` (left unchanged).
    :param corrections: ``{image: correction}`` from ``load_image_corrections``, or None to apply nothing.
    :return: ``(records keyed by canonical id, correction counts)``.
    """
    records = {cid: {**rec, "images": json.loads(json.dumps(rec.get("images") or {}))} for cid, rec in index.items()}
    if corrections is None:
        return records, {"loaded": 0}
    stats = apply_image_corrections(records, corrections)
    stats["changed_ids"] = len(stats.get("changed_ids") or [])
    return records, stats


# ------------------------------------------------------------------------------------------- per-source listing


def legacy_label(token: str, slip_side: str) -> str:
    """Side label of a legacy-store photograph, by the rule stage 0 applies to the store's own records: the side its
    shelf-mark slip names replaces the file-name token, except that an agreeing token adds the leaf to a bare slip
    side (slip ``v`` + token ``1v`` -> ``1v``).

    :param token: Folio + side token from the file name (``1r``) or ``""``.
    :param slip_side: Slip side or ``""``.
    :return: The label, ``""`` when neither is known.
    """
    if not slip_side:
        return token
    if len(slip_side) == 1 and len(token) > 1 and not sides_conflict(token, slip_side):
        return token
    return slip_side


def legacy_ref(rec: Mapping[str, Any], url: Optional[str] = None, label: Optional[str] = None) -> ImageRef:
    """Reference to a legacy-store photograph listed by the store's image manifest.

    :param rec: Manifest record.
    :param url: Bucket URL (default: the record's own).
    :param label: Side label overriding the record's (a side set by an image correction).
    :return: ``fjp_local`` reference (presence is checked later).
    """
    token = f"{rec.get('folio') or ''}{rec.get('side_token') or ''}" if rec.get("side_token") else ""
    slip = str(rec.get("slip_side") or "")
    ref = ImageRef(image_id="", origin="fjp_local", url=url or rec.get("url"), path=rec.get("path"),
                   sha256=rec.get("sha256"), width=rec.get("width"), height=rec.get("height"),
                   label=label or legacy_label(token, slip) or None, slip_outcome=rec.get("slip_outcome"),
                   status="listed")
    if slip:
        conflict = "; conflict" if sides_conflict(token, slip) else ""
        add_note(ref, f"side_token={token or '-'}; slip_side={slip}{conflict}")
    return ref


def merged_record_listing(cid: str, rec: Mapping[str, Any], legacy: LegacyIndex,
                          bodleian_dir: Optional[Path]) -> List[Union[ImageRef, str]]:
    """Images of one (corrected) merged record: legacy-store entries, KTIV image URLs, Bodleian masters, and the
    record's own IIIF manifest(s) (as manifest URLs, expanded later).

    :param cid: Canonical id of the record.
    :param rec: Facts from :func:`merged_images_index` after :func:`corrected_records`.
    :param legacy: Legacy-store index.
    :param bodleian_dir: Directory of local Bodleian masters (``<canonical id>/<file>``) or None.
    :return: Image references and manifest URLs, in record order.
    """
    images = rec.get("images") or {}
    sides = images.get("fjp_sides") or {}
    out: List[Union[ImageRef, str]] = []
    for entry in images.get("fjp") or []:
        name = image_name(str(entry))
        url = legacy_url(str(entry))
        side = sides.get(name)
        if name in legacy.by_file:
            ref = legacy_ref(legacy.by_file[name], url=url, label=side)
            filed_under = legacy.by_file[name].get("canonical_id")
            if filed_under != cid:  # the store manifest's own relabel disagrees: evidence for review
                add_note(ref, f"legacy manifest files it under {filed_under or 'no record'}")
            out.append(ref)
        else:
            out.append(ImageRef(image_id="", origin="pgpid_mirror" if is_pgpid_mirror(name) else "fjp_bucket",
                                url=url, label=side or side_token(name), status="listed"))
    ktiv = images.get("ktiv") or {}
    for url in ktiv.get("image_urls") or []:
        out.append(ImageRef(image_id="", origin="iiif_nli", url=str(url), label=Path(image_name(str(url))).stem,
                            status="listed"))
    bodleian = images.get("bodleian") or {}
    for url in bodleian.get("image_urls") or bodleian.get("source_urls") or []:
        local = bodleian_dir / cid / image_name(str(url)) if bodleian_dir is not None else None
        out.append(ImageRef(image_id="", origin="bodleian", url=str(url),
                            path=str(local) if local is not None and local.is_file() else None,
                            label=Path(image_name(str(url))).stem, status="listed"))
    out.extend(url_items(split_urls(rec.get("pgp_iiif"))))
    return out


def cudl_manifest_for_id(cid: str) -> Optional[str]:
    """The Cambridge library manifest of a Cambridge canonical id, built by the facade's ``cudl_manifest`` (its
    tokens are the id's letters and numbers, so the id's underscores stand for the mark's spaces and dots).

    :param cid: Canonical id.
    :return: Manifest URL, or None for a non-Cambridge id.
    """
    for prefix, mark in CUDL_ID_PREFIXES:
        if cid.startswith(prefix) and len(cid) > len(prefix):
            return cudl_manifest(mark + cid[len(prefix):].replace("_", " "), known=frozenset())
    return None


def ktiv_fl(cand: Candidate) -> Tuple[Optional[str], Optional[str]]:
    """KTIV system number and page FL id of a KTIV candidate (row id ``<sys>:<page id>[~suffix]``).

    :param cand: Candidate.
    :return: ``(sys_num, fl)``; ``fl`` is None when the page id is not an ``FL`` id.
    """
    head, _, tail = cand.row_id.partition(":")
    sys_num = cand.ktiv_id or head or None
    page = tail.split("~", 1)[0]
    return sys_num, (page if page.upper().startswith("FL") else None)


def ktiv_page_ref(cand: Candidate, ktiv_dir: Path, ktiv_cache: Path) -> Tuple[Optional[ImageRef], str]:
    """The candidate's page image from its manuscript's KTIV image zip, extracted once to
    ``ktiv_cache/<sys>/<fl>.jpg`` (with a ``.json`` sidecar of its keys).

    The zip member's bytes are kept (``sha256`` is theirs) and written upright by :func:`store_image`, instead of
    ``build_ktiv_dataset.load_page_image``, which returns decoded pixels only. Reading the member and writing the
    cache file are I/O whose failure is returned as a reason (the candidate then keeps its page URL).

    :param cand: KTIV candidate.
    :param ktiv_dir: Directory of the KTIV image zips.
    :param ktiv_cache: Cache directory.
    :return: ``(reference or None, reason)``; the reason is ``""`` when a file was found.
    """
    sys_num, fl = ktiv_fl(cand)
    if not sys_num or not fl:
        return None, "ktiv_page_without_fl"
    dest = ktiv_cache / safe_name(sys_num) / f"{safe_name(fl)}.jpg"
    meta_path = dest.with_suffix(".json")
    if dest.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if "source_sha256" not in meta:  # sidecar from before the file-digest rule: re-key on the file
            meta = {**meta, "source_sha256": meta["sha256"], "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
            _atomic_write_text(meta_path, json.dumps(meta))
    else:
        hit = find_zip_member(ktiv_dir, sys_num, fl)
        if hit is None:
            return None, "ktiv_zip_member_not_found"
        zpath, member = hit
        try:
            with zipfile.ZipFile(zpath) as zf:
                data = zf.read(member)
            keys = store_image(data, dest)
        except (OSError, zipfile.BadZipFile, zlib.error, Image.DecompressionBombError) as exc:
            return None, f"ktiv_zip_unreadable {zpath.name}:{member}: {type(exc).__name__}"
        meta = {**keys, "zip": zpath.name, "member": member}
        _atomic_write_text(meta_path, json.dumps(meta))
    ref = ImageRef(image_id="", origin="ktiv_zip", path=str(dest), sha256=meta["sha256"], phash=meta["phash"],
                   width=meta["width"], height=meta["height"], status="listed",
                   note=f"zip {meta.get('zip')} member {meta.get('member')}")
    if meta.get("reencoded"):
        add_note(ref, "reencoded")
    return ref, ""


def split_existing(ref: ImageRef) -> List[Union[ImageRef, str]]:
    """Normalise one stage-0 image reference.

    Stage 0 fills one reference per candidate from the ledger's ``image_url`` and ``image_path``, which need not be
    the same photograph (the URL can be a IIIF manifest, the path another side's local file), and takes its origin
    from the ledger's first image source, which need not describe either. A reference whose URL and path name the
    same file stays whole (origin from the URL, made local when the file is on disk); otherwise the path (with the
    ledger's sha256 / phash) becomes a ``local_other`` item (refined later from the directory it lives in) and the
    URL a separate item: a manifest URL to expand, or an image reference with its origin from the URL.

    :param ref: Stage-0 reference.
    :return: References and manifest URLs.
    """
    url = ref.url.strip() if ref.url else None
    same_file = bool(url and ref.path and image_name(url) == Path(ref.path).name)
    out: List[Union[ImageRef, str]] = []
    if ref.path:
        local = ImageRef(**{**ref.__dict__, "url": url if same_file else None})
        if same_file and url:
            derived = origin_of_url(url)
            local.origin = "fjp_local" if derived == "fjp_bucket" else derived
        else:
            if ref.origin != "local_other":
                add_note(local, f"stage0 origin {ref.origin}")
            local.origin = "local_other"
            local.label, local.slip_outcome, local.canvas_index = None, None, None
        out.append(local)
    if url and not same_file:
        if not is_image_url(url):
            out.append(url)
        else:
            out.append(ImageRef(image_id="", origin=origin_of_url(url), url=url, status="listed"))
    return out


def _norm_url(url: Optional[str]) -> Optional[str]:
    """Comparison form of a URL (trimmed, no trailing slash, ``http`` as ``https``).

    :param url: URL or None.
    :return: Normalised URL or None.
    """
    if not url:
        return None
    text = url.strip().rstrip("/")
    return "https://" + text[len("http://"):] if text.startswith("http://") else text


def merge_ref(into: ImageRef, other: ImageRef) -> None:
    """Fold a second listing of the same photograph into the first: missing fields are filled; when the second
    brings the local file, its origin comes with it.

    :param into: Kept reference (mutated).
    :param other: Duplicate listing.
    """
    if into.path is None and other.path is not None:
        into.path = other.path
        into.origin = other.origin
        into.status = other.status
    for name in ("url", "sha256", "phash", "width", "height", "canvas_index", "label", "slip_outcome"):
        if getattr(into, name) is None and getattr(other, name) is not None:
            setattr(into, name, getattr(other, name))
    for part in (other.note or "").split("; "):
        add_note(into, part)


def dedup_refs(refs: Sequence[ImageRef]) -> List[ImageRef]:
    """Merge references that list one photograph twice (same URL, then same local path), keeping first-seen order.

    :param refs: References.
    :return: Distinct references.
    """
    out: List[ImageRef] = []
    by_url: Dict[str, ImageRef] = {}
    for ref in refs:
        key = _norm_url(ref.url)
        if key is not None and key in by_url:
            merge_ref(by_url[key], ref)
            continue
        out.append(ref)
        if key is not None:
            by_url[key] = ref
    final: List[ImageRef] = []
    by_path: Dict[str, ImageRef] = {}
    for ref in out:
        if ref.path and ref.path in by_path:
            merge_ref(by_path[ref.path], ref)
            continue
        final.append(ref)
        if ref.path:
            by_path[ref.path] = ref
    return final


def mark_duplicates(refs: Sequence[ImageRef]) -> int:
    """Within one candidate, mark every later reference whose bytes (sha256) equal an earlier usable one as
    ``duplicate``.

    :param refs: The candidate's references (mutated).
    :return: Number marked.
    """
    first: Dict[str, ImageRef] = {}
    marked = 0
    for ref in refs:
        if not ref.sha256 or ref.status not in USABLE_STATUSES + ("too_small", "duplicate"):
            continue
        kept = first.get(ref.sha256)
        if kept is None:
            if ref.status != "duplicate":
                first[ref.sha256] = ref
            continue
        if ref is not kept and ref.status != "duplicate":
            ref.status = "duplicate"
            add_note(ref, f"same bytes as {kept.image_id}")
            marked += 1
    return marked


def assign_ids(refs: Sequence[ImageRef]) -> None:
    """Give each reference its in-candidate id ``<origin>:<index>``.

    :param refs: The candidate's references (mutated).
    """
    for index, ref in enumerate(refs):
        ref.image_id = f"{ref.origin}:{index}"


# ------------------------------------------------------------------------------------------------- local keys


def _file_signature(path: Path) -> str:
    """Size and modification time of a file, for the key cache.

    :param path: File.
    :return: ``"<size>:<mtime_ns>"``.
    """
    st = path.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def local_image_keys(paths_: Iterable[str], cache_path: Path, workers: int = 4) -> Dict[str, Dict[str, Any]]:
    """sha256 / phash / upright size of local image files, cached by path, size and mtime.

    Reading or decoding a file is I/O whose failure is recorded (``{"error": ...}``), not raised.

    :param paths_: Local files.
    :param cache_path: JSON cache.
    :param workers: Threads.
    :return: ``{path: keys}`` (keys of :func:`image_keys`, or ``{"error": text}``).
    """
    cache: Dict[str, Dict[str, Any]] = (json.loads(cache_path.read_text(encoding="utf-8"))
                                        if cache_path.is_file() else {})
    out: Dict[str, Dict[str, Any]] = {}
    todo: List[Tuple[str, str]] = []
    for path in dict.fromkeys(paths_):
        sig = _file_signature(Path(path))
        hit = cache.get(path)
        if hit is not None and hit.get("sig") == sig:
            out[path] = hit
        else:
            todo.append((path, sig))

    def one(item: Tuple[str, str]) -> Tuple[str, Dict[str, Any]]:
        """Keys of one file.

        :param item: ``(path, signature)``.
        :return: ``(path, keys)``.
        """
        path, sig = item
        try:
            keys = image_keys(Path(path).read_bytes())
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            keys = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        return path, {**keys, "sig": sig}

    if todo:
        print(f"  local image keys: {len(out):,} cached, {len(todo):,} to hash", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for path, keys in pool.map(one, todo):
                out[path] = keys
                cache[path] = keys
        _atomic_write_text(cache_path, json.dumps(cache))
    return out


def size_status(ref: ImageRef, usable: str) -> str:
    """Status of a local file by its size.

    :param ref: Reference with ``width`` / ``height``.
    :param usable: Status when large enough (``present`` or ``fetched``).
    :return: ``usable`` or ``too_small``.
    """
    if ref.width and ref.height and max(ref.width, ref.height) < MIN_LONG_SIDE:
        return "too_small"
    return usable


# ----------------------------------------------------------------------------------------------- fetch cap


def stem_mark(stem: str) -> Tuple[str, Optional[str]]:
    """A file-name stem read as a shelf mark: ``_`` and ``-`` are separators, and a trailing legacy-store side token
    (``<folio><r|v>``, e.g. ``1r``) is split off. Folio 1 (or none) is the item's own leaf, so its side names the
    mark itself; another folio stays in the mark (a child leaf, as the facade reads ``f. 2v``).

    :param stem: File name without extension.
    :return: ``(mark text, side token)``; the side is None when the stem ends in no side token.
    """
    tokens = [t for t in _STEM_SEPARATORS_RE.split(stem) if t]
    match = _LEGACY_SIDE_RE.match(tokens[-1]) if len(tokens) > 1 else None
    if match is None:
        return " ".join(tokens), None
    folio = match.group(1)
    base = tokens[:-1] + ([folio] if folio and folio != "1" else [])
    return " ".join(base), tokens[-1]


def own_mark_keys(cand: Candidate) -> Tuple[Set[str], List[Optional[str]]]:
    """The facade's loose keys of a candidate's own marks, and the institutions to read its file names with.

    :param cand: Candidate.
    :return: ``(keys of its canonical ids and shelf mark, registry tokens of its canonical ids plus None)``.
    """
    institutions: List[Optional[str]] = list(dict.fromkeys(split_id(cid)[0] for cid in cand.canonical_ids))
    if None not in institutions:
        institutions.append(None)
    keys = {loose_key(cid) for cid in cand.canonical_ids}
    if cand.shelfmark:
        keys |= {loose_key(cand.shelfmark, institution=inst) for inst in institutions}
    keys.discard("")
    return keys, institutions


def stem_tier(name: str, own_keys: Set[str], institutions: Sequence[Optional[str]]) -> int:
    """How a photograph's file name relates to the candidate's own mark, through the facade's ``loose_key`` (the
    stem's ``_`` / ``-`` read as separators, keyed with each of the candidate's institutions).

    :param name: File name.
    :param own_keys: :func:`own_mark_keys` keys.
    :param institutions: :func:`own_mark_keys` institutions.
    :return: 1 = the stem names the own mark; 2 = it names a side of that mark; 0 = unrelated.
    """
    stem = Path(name).stem
    whole = " ".join(t for t in _STEM_SEPARATORS_RE.split(stem) if t)
    base, side = stem_mark(stem)
    if any(loose_key(whole, institution=inst) in own_keys for inst in institutions):
        return 1
    if side is not None and any(loose_key(base, institution=inst) in own_keys for inst in institutions):
        return 2
    return 0


def cap_fetch_selection(cand: Candidate, max_per_candidate: Optional[int]) -> Dict[str, Any]:
    """Choose which URL-only photographs of a candidate may be fetched (the others become ``box_listing``).

    Only when the candidate has more than ``max_per_candidate`` photographs known only by URL (status ``listed``):
    up to that many are kept, first those whose file name names the candidate's own mark, then those naming a side
    of it (in listing order); a canvas of an expanded IIIF manifest counts as naming the own mark (its identity is
    the manifest the candidate's own record or pgpid names; its file name is the IIIF ``default``). Unrelated names
    never fill the rest. When none matches, every URL-only photograph becomes ``box_listing`` and the candidate is
    flagged ``box_listing_unmatched``. Local files are never touched (a fetch selection only; sides are placed by
    reads, never by names).

    :param cand: Candidate (mutated: statuses, notes of its references).
    :param max_per_candidate: The cap, or None / 0 for no cap.
    :return: ``{"listed", "selected", "box_listing"}`` counts over all the candidate's images, plus
        ``box_listing_unmatched: True`` when no URL matched.
    """
    fetchable = [r for r in cand.images if r.status == "listed" and r.url and not r.path]
    out: Dict[str, Any] = {}
    if max_per_candidate and len(fetchable) > max_per_candidate:
        own_keys, institutions = own_mark_keys(cand)
        tiers = {id(r): (1 if r.canvas_index is not None
                         else stem_tier(image_name(r.url or ""), own_keys, institutions)) for r in fetchable}
        ranked = [r for tier in (1, 2) for r in fetchable if tiers[id(r)] == tier]
        keep = {id(r) for r in ranked[:max_per_candidate]}
        for ref in fetchable:
            if id(ref) in keep:
                continue
            ref.status = "box_listing"
            add_note(ref, "fetch_cap: over the cap" if tiers[id(ref)] else "fetch_cap: file name names another mark")
        if not ranked:
            out["box_listing_unmatched"] = True
    n_box = sum(1 for r in cand.images if r.status == "box_listing")
    out.update({"listed": len(cand.images), "selected": len(cand.images) - n_box, "box_listing": n_box})
    return out


# ---------------------------------------------------------------------------------------------- resolve_images


@dataclass
class _Sources:
    """Indexes the resolvers share.

    :param pgp_urls: PGP export IIIF URLs per pgpid.
    :param legacy: Legacy-store image index.
    :param merged: Corrected merged image facts per canonical id.
    :param run2: Run-2 local downloads per canonical id.
    :param run2_folders: Ids per run-2 download folder.
    :param bodleian_dir: Local Bodleian masters or None.
    :param ktiv_dir: KTIV image zips.
    :param ktiv_cache: KTIV page cache.
    """

    pgp_urls: Dict[str, List[str]]
    legacy: LegacyIndex
    merged: Dict[str, Dict[str, Any]]
    run2: Dict[str, List[Dict[str, Any]]]
    run2_folders: Dict[str, int]
    bodleian_dir: Optional[Path]
    ktiv_dir: Path
    ktiv_cache: Path


def _candidate_listing(cand: Candidate, src: _Sources, notes: Dict[str, Any]) -> List[Union[ImageRef, str]]:
    """Every image reference and manifest URL known for one candidate, before expansion and dedup.

    :param cand: Candidate.
    :param src: Shared indexes.
    :param notes: The candidate's stage-1 notes (mutated: misses and corrections).
    :return: References and manifest URLs, in listing order.
    """
    items: List[Union[ImageRef, str]] = []
    if cand.source in PASS_THROUGH_SOURCES:
        return [ImageRef(**ref.__dict__) for ref in cand.images]
    if cand.source == "ktiv":
        ref, reason = ktiv_page_ref(cand, src.ktiv_dir, src.ktiv_cache)
        if ref is not None:  # the zip page replaces the same page's library URL (kept as provenance)
            ref.url = next((r.url for r in cand.images if r.url), None)
            return [ref]
        notes["misses"].append(reason)
    for ref in cand.images:
        items.extend(split_existing(ref))
    if cand.source == "pgp":
        for pgpid in cand.pgpids:
            items.extend(url_items(src.pgp_urls.get(str(pgpid), [])))
    if cand.source in MERGED_SOURCES:
        for cid in cand.canonical_ids:
            rec = src.merged.get(cid)
            if rec is None:
                notes["misses"].append(f"merged_record_absent {cid}")
                continue
            items.extend(merged_record_listing(cid, rec, src.legacy, src.bodleian_dir))
            images = rec.get("images") or {}
            moved_in = [c["image"] for c in images.get("fjp_corrections") or []]
            removed = [r["image"] for r in images.get("fjp_removed") or []]
            if moved_in or removed:
                notes["corrections"].append({"canonical_id": cid, "moved_in": moved_in, "removed": removed})
            ktiv = images.get("ktiv") or {}
            if ktiv.get("iiif_manifest_url") and not ktiv.get("image_urls"):
                notes["misses"].append(f"ktiv_manifest_not_expanded {cid}")
    if cand.source in ("literature", "lit_run2"):
        for cid in cand.canonical_ids:
            for rec in src.run2.get(cid, []):
                ref = ImageRef(image_id="", origin="cudl_local", path=str(rec["path"]),
                               label=rec.get("canvas_label") or None, status="listed",
                               note=f"run2 manifest {rec.get('manifest_id')}")
                shared = src.run2_folders.get(str(Path(rec["path"]).parent), 1)
                if shared > 1:
                    add_note(ref, f"run2 folder shared by {shared} ids")
                items.append(ref)
            for rec in src.legacy.by_cid.get(cid, []):
                items.append(legacy_ref(rec))
    if cand.source in MERGED_SOURCES and not items:
        derived = [u for u in dict.fromkeys(cudl_manifest_for_id(cid) for cid in cand.canonical_ids) if u]
        if derived:
            notes["misses"].append("no_listed_image cudl_manifest_derived")
            items.extend(derived)
    return items


def _local_origin(path: str, src: "_Sources", run2_paths: Set[str]) -> str:
    """Origin of a local file listed without a trustworthy origin, from where it lives.

    :param path: Local path.
    :param src: Shared indexes.
    :param run2_paths: Run-2 local download paths.
    :return: ``fjp_local`` | ``cudl_local`` | ``ktiv_zip`` (the KTIV page cache, or a page unpacked next to the
        zips) | ``bodleian`` | ``local_other``.
    """
    if path in src.legacy.paths:
        return "fjp_local"
    if path in run2_paths:
        return "cudl_local"
    parents = Path(path).parents
    if Path(src.ktiv_cache) in parents or Path(src.ktiv_dir) in parents:
        return "ktiv_zip"
    if src.bodleian_dir is not None and Path(src.bodleian_dir) in parents:
        return "bodleian"
    return "local_other"


def _complete_ref(ref: ImageRef, src: "_Sources", run2_paths: Set[str]) -> None:
    """Settle one reference's origin and presence before hashing: a recorded path that is absent is dropped (the
    URL, if any, stays to be fetched); legacy-store manifest fields fill gaps.

    :param ref: Reference (mutated).
    :param src: Shared indexes.
    :param run2_paths: Run-2 local download paths.
    """
    if ref.path:
        if ref.origin == "local_other":
            ref.origin = _local_origin(ref.path, src, run2_paths)
        rec = src.legacy.by_file.get(Path(ref.path).name)
        if rec is not None and str(rec.get("path")) == ref.path:
            for name in ("width", "height", "sha256", "slip_outcome"):
                if getattr(ref, name) is None and rec.get(name) is not None:
                    setattr(ref, name, rec[name])
        if not Path(ref.path).is_file():
            add_note(ref, f"path_absent {ref.path}")
            ref.path = None
            if ref.origin == "fjp_local" and ref.url:
                ref.origin = origin_of_url(ref.url)
            ref.status = "listed" if ref.url else "missing"
            return
        if ref.origin == "fjp_bucket":
            ref.origin = "fjp_local"
        ref.status = "present"
        return
    if ref.status not in ("head_404", "missing", "fetched", "too_small", "duplicate"):
        ref.status = "listed" if ref.url else "missing"


def resolve_images(cands: List[Candidate], *, pgp_export_dir: Path, merged_path: Path,
                   corrections_path: Optional[Path], work_dir: Path, legacy_manifest: Union[Path, Sequence[Path]],
                   run2_local_images: Path, ktiv_dir: Path, ktiv_cache: Path,
                   fetch_manifest: Optional[ManifestFetcher], bodleian_dir: Optional[Path] = None,
                   merged_ids: Optional[Iterable[str]] = None, workers: int = 4, delay: float = 0.3,
                   retries: int = 2, sleep: SleepFn = time.sleep,
                   max_per_candidate: Optional[int] = MAX_PER_CANDIDATE) -> Counter:
    """Complete every candidate's ``images`` in place (see the module docstring for the per-source rules).

    Manifests are read from ``work_dir/manifests/<sha1(url)>.json``; uncached ones are fetched through
    ``fetch_manifest`` when it is given, and stay unexpanded (named in ``notes['images_stage1']``) when it is None
    (offline / inventory mode). Local files get sha256 / phash / size (cached in ``work_dir/manifests``); status is
    ``present`` (or ``too_small``) for a local file, ``listed`` for a URL still to fetch, ``missing`` for neither,
    ``box_listing`` for a URL left out by the fetch cap (:func:`cap_fetch_selection`, counts in the notes).

    :param cands: Stage-0 candidates (mutated).
    :param pgp_export_dir: PGP export directory (``documents.csv``).
    :param merged_path: Merged corpus JSONL.
    :param corrections_path: Image corrections JSONL, or None for none.
    :param work_dir: Build work dir.
    :param legacy_manifest: Legacy-store image manifest(s).
    :param run2_local_images: Run-2 ``local_images.json``.
    :param ktiv_dir: KTIV image zips.
    :param ktiv_cache: KTIV page cache directory.
    :param fetch_manifest: ``fetch_manifest(url) -> parsed JSON`` (raises on failure), or None for offline.
    :param bodleian_dir: Local Bodleian masters, or None.
    :param merged_ids: Canonical ids to index from the merged file (default: those of ``cands``); pass the whole
        stage-0 set so a later full run reuses the cached index.
    :param workers: Threads for manifests and local hashing.
    :param delay: Minimum seconds between manifest requests to one host.
    :param retries: Extra manifest attempts.
    :param sleep: Sleep function.
    :param max_per_candidate: Fetch cap per candidate (None / 0: no cap).
    :return: Counts (references by origin and status, manifests, corrections).
    """
    stats: Counter = Counter()
    manifests_dir = work_dir / MANIFESTS_DIR
    sources = {c.source for c in cands}
    pgp_urls = load_pgp_manifest_urls(pgp_export_dir) if "pgp" in sources else {}
    legacy = load_legacy_manifest(legacy_manifest) if sources & ({"pgp", "fjp", "fjp_scrape"} | MERGED_SOURCES) \
        else LegacyIndex()
    merged: Dict[str, Dict[str, Any]] = {}
    if sources & MERGED_SOURCES:
        ids = set(merged_ids) if merged_ids is not None else set()
        ids |= {cid for c in cands if c.source in MERGED_SOURCES for cid in c.canonical_ids}
        corrections = load_image_corrections(str(corrections_path)) if corrections_path is not None else None
        # every record a correction names is indexed too, so a move INTO a candidate's record finds its source
        corr_ids = correction_record_ids(corrections) if corrections else set()
        index = merged_images_index(merged_path, ids | corr_ids, manifests_dir / MERGED_INDEX_FILE)
        merged, corr_stats = corrected_records(index, corrections)
        for key, value in corr_stats.items():
            if isinstance(value, int):
                stats[f"corrections_{key}"] += value
    run2, run2_folders = load_run2_local(run2_local_images) if sources & {"literature", "lit_run2"} else ({}, {})
    run2_paths = {str(r["path"]) for recs in run2.values() for r in recs}
    src = _Sources(pgp_urls=pgp_urls, legacy=legacy, merged=merged, run2=run2, run2_folders=run2_folders,
                   bodleian_dir=bodleian_dir, ktiv_dir=ktiv_dir, ktiv_cache=ktiv_cache)

    listings: List[Tuple[Candidate, List[Union[ImageRef, str]], Dict[str, Any]]] = []
    for cand in cands:
        notes: Dict[str, Any] = {"manifests_expanded": [], "manifests_unexpanded": [], "misses": [], "corrections": []}
        listings.append((cand, _candidate_listing(cand, src, notes), notes))

    manifest_urls = list(dict.fromkeys(i for _, items, _ in listings for i in items if isinstance(i, str)))
    errors: Dict[str, str] = {}
    if fetch_manifest is not None:
        errors = fetch_manifests(manifest_urls, manifests_dir, fetch_manifest, workers=workers, delay=delay,
                                 retries=retries, sleep=sleep)
    manifests = {u: cached_manifest(manifests_dir, u) for u in manifest_urls}
    stats["manifests_listed"] += len(manifest_urls)
    stats["manifests_unavailable"] += sum(1 for m in manifests.values() if m is None)

    for cand, items, notes in listings:
        refs: List[ImageRef] = []
        seen_manifests: Set[str] = set()
        for item in items:
            if isinstance(item, ImageRef):
                refs.append(item)
                continue
            if item in seen_manifests:
                continue
            seen_manifests.add(item)
            manifest = manifests.get(item)
            if manifest is None:
                reason = errors.get(item) or ("manifest_not_cached" if fetch_manifest is None else "manifest_failed")
                notes["manifests_unexpanded"].append({"url": item, "reason": reason})
                continue
            canvas_refs = refs_from_manifest(item, manifest)
            notes["manifests_expanded"].append({"url": item, "canvases": len(canvas_refs)})
            refs.extend(canvas_refs)
        for ref in refs:
            _complete_ref(ref, src, run2_paths)
        cand.images = dedup_refs(refs)
        cand.notes[NOTES_KEY] = {k: v for k, v in notes.items() if v}

    present = [r for c in cands for r in c.images if r.path and r.status == "present"]
    need = [r.path for r in present if r.sha256 is None or r.phash is None or r.width is None or r.height is None]
    keys = local_image_keys([p for p in need if p], manifests_dir / LOCAL_KEYS_FILE, workers=workers) if need else {}
    for ref in present:
        hit = keys.get(ref.path or "")
        if hit is not None and "error" in hit:
            ref.status = "missing"
            add_note(ref, f"undecodable {hit['error']}")
            continue
        if hit is not None:
            if ref.sha256 and hit.get("sha256") and ref.sha256 != hit["sha256"]:  # the file on disk wins
                add_note(ref, f"sha256 differs from the listing ({ref.sha256[:12]})")
                ref.sha256, ref.phash = hit["sha256"], None
            for name in ("sha256", "phash", "width", "height"):
                if getattr(ref, name) is None:
                    setattr(ref, name, hit.get(name))
            if hit.get("orientation") not in (None, 0, 1):
                add_note(ref, f"exif_orientation {hit['orientation']}")
        ref.status = size_status(ref, "present")
    for cand in cands:
        assign_ids(cand.images)
        stats["duplicates_marked"] += mark_duplicates(cand.images)
        cap = cap_fetch_selection(cand, max_per_candidate)
        cand.notes[NOTES_KEY].update(cap)
        stats["cands_box_listing_unmatched"] += int(bool(cap.get("box_listing_unmatched")))
        for ref in cand.images:
            stats[f"origin_{ref.origin}"] += 1
            stats[f"status_{ref.status}"] += 1
    return stats


# ---------------------------------------------------------------------------------------------------- fetching


class FetchState:
    """Append-only log of every fetched URL's outcome (``work_dir/images/fetch_state.jsonl``), loaded at start so a
    re-run reuses files, 404s and hashes instead of fetching again."""

    def __init__(self, path: Path) -> None:
        """Load the log (the last line per URL wins).

        :param path: Log file (created on first append).
        """
        self.path = path
        self.lock = threading.Lock()
        self.by_url: Dict[str, Dict[str, Any]] = {}
        self.by_sha: Dict[str, Dict[str, Any]] = {}
        self.reserved: Set[str] = set()
        if path.is_file():
            for rec in read_jsonl(path):
                self._index(rec)

    def _index(self, rec: Dict[str, Any]) -> None:
        """Index one outcome.

        :param rec: Outcome record.
        """
        self.by_url[rec["url"]] = rec
        if rec.get("sha256") and rec.get("path") and rec["sha256"] not in self.by_sha:
            self.by_sha[rec["sha256"]] = rec
        if rec.get("path"):
            self.reserved.add(rec["path"])

    def append(self, rec: Dict[str, Any]) -> None:
        """Record one outcome (written and flushed at once; the caller holds :attr:`lock`).

        :param rec: Outcome record.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
        self._index(rec)

    def reusable(self, url: str, retry_missing: bool) -> Optional[Dict[str, Any]]:
        """The recorded outcome of a URL when it needs no new request.

        :param url: URL.
        :param retry_missing: Treat a recorded ``missing`` as not final.
        :return: The outcome, or None when the URL must be fetched.
        """
        rec = self.by_url.get(url)
        if rec is None:
            return None
        if rec["status"] in ("fetched", "too_small"):
            return rec if rec.get("path") and Path(rec["path"]).is_file() else None
        if rec["status"] == "head_404":
            return rec
        if rec["status"] == "missing" and not retry_missing:
            return rec
        return None


def apply_outcome(ref: ImageRef, rec: Mapping[str, Any]) -> None:
    """Copy a fetch outcome onto a reference.

    :param ref: Reference (mutated).
    :param rec: Outcome record.
    """
    ref.status = rec["status"]
    if rec.get("path"):
        ref.path = rec["path"]
    for name in ("sha256", "phash", "width", "height"):
        if rec.get(name) is not None:
            setattr(ref, name, rec[name])
    for part in str(rec.get("note") or "").split("; "):
        add_note(ref, part)
    reencoded = bool(rec.get("reencoded")) or "reencoded" in str(rec.get("note") or "")
    if reencoded and rec.get("source_sha256") is None and ref.path and Path(ref.path).is_file():
        # a record from before the file-digest rule: ``sha256`` was the source bytes' digest, re-key on the file
        add_note(ref, f"source_sha256 {ref.sha256}")
        ref.sha256 = hashlib.sha256(Path(ref.path).read_bytes()).hexdigest()


def _needs_fetch(ref: ImageRef) -> bool:
    """Whether a reference is a photograph known only by URL.

    :param ref: Reference.
    :return: True when it has a URL and no local file.
    """
    return bool(ref.url) and not ref.path and ref.status in ("listed", "missing", "head_404")


def apply_fetch_state(cands: Sequence[Candidate], work_dir: Path) -> Counter:
    """Offline: copy every recorded fetch outcome onto the matching references (no request is made).

    :param cands: Candidates (mutated).
    :param work_dir: Build work dir.
    :return: Counts of applied outcomes by status.
    """
    state = FetchState(work_dir / IMAGES_DIR / FETCH_STATE_FILE)
    counts: Counter = Counter()
    for cand in cands:
        for ref in cand.images:
            if _needs_fetch(ref) and ref.url in state.by_url:
                rec = state.reusable(ref.url, retry_missing=False)
                if rec is not None:
                    apply_outcome(ref, rec)
                    counts[rec["status"]] += 1
        mark_duplicates(cand.images)
    return counts


def _dest_for(state: FetchState, images_dir: Path, group: str, index: int) -> Path:
    """A free file name in a candidate's image directory: ``<index>.jpg`` when free, else the next free number.

    :param state: Fetch state (its lock must be held).
    :param images_dir: ``work_dir/images``.
    :param group: Candidate directory name.
    :param index: Preferred number (the reference's index in its candidate).
    :return: Reserved path.
    """
    n = index
    while True:
        dest = images_dir / group / f"{n}.jpg"
        if str(dest) not in state.reserved and not dest.exists():
            state.reserved.add(str(dest))
            return dest
        n += 1


def fetch_missing(cands: Sequence[Candidate], work_dir: Path, *, head: Callable[[str], int],
                  get: Callable[[str], bytes], delay: float = 0.3, limit: Optional[int] = None, retries: int = 2,
                  workers: int = 1, sleep: SleepFn = time.sleep) -> Counter:
    """Fetch every photograph known only by URL; failures are recorded as statuses, never raised.

    Per distinct URL: a bucket URL gets a HEAD first (404 -> ``head_404``, another 4xx -> ``missing``, no GET); the
    GET's bytes give ``sha256`` (of the source bytes) and ``phash`` (decoded, before rotation); bytes already stored
    (same sha256, here or as a local file of any candidate) reuse the first path; otherwise the image is written
    upright to ``work_dir/images/<first canonical id or row id>/<index>.jpg`` (:func:`store_image`). A long side
    under :data:`MIN_LONG_SIDE` gives ``too_small`` (file kept). Every outcome is appended to the fetch state, so a
    second run requests nothing it has (a recorded ``missing`` is retried, a ``head_404`` is not).

    :param cands: Candidates (mutated).
    :param work_dir: Build work dir.
    :param head: ``head(url) -> HTTP status``; raises on a network failure.
    :param get: ``get(url) -> body bytes``; raises on failure (:class:`PermanentFetchError` is not retried).
    :param delay: Minimum seconds between two requests to one host.
    :param limit: Fetch at most this many URLs (None: all).
    :param retries: Extra attempts per request.
    :param workers: Threads.
    :param sleep: Sleep function (tests pass a no-op).
    :return: Counts by outcome.
    """
    images_dir = work_dir / IMAGES_DIR
    state = FetchState(images_dir / FETCH_STATE_FILE)
    pacer = HostPacer(delay, sleep)
    counts: Counter = Counter()
    counts_lock = threading.Lock()
    local_by_sha: Dict[str, str] = {}
    for cand in cands:
        for ref in cand.images:
            if ref.sha256 and ref.path and ref.status in ("present", "too_small"):
                local_by_sha.setdefault(ref.sha256, ref.path)
    todo: Dict[str, List[Tuple[Candidate, ImageRef, int]]] = {}
    for cand in cands:
        for index, ref in enumerate(cand.images):
            if not _needs_fetch(ref):
                continue
            rec = state.reusable(str(ref.url), retry_missing=True)
            if rec is not None:
                apply_outcome(ref, rec)
                counts[f"reused_{rec['status']}"] += 1
                continue
            todo.setdefault(str(ref.url), []).append((cand, ref, index))
    urls = _interleave_hosts(list(todo))
    if limit is not None:
        counts["skipped_limit"] += max(0, len(urls) - limit)
        urls = urls[:limit]

    def one(url: str) -> None:
        """Fetch one URL and apply its outcome to every reference listing it.

        :param url: Image URL.
        """
        cand, _, index = todo[url][0]
        rec = _fetch_one(url, head=head, get=get, retries=retries, pacer=pacer, sleep=sleep, state=state,
                         local_by_sha=local_by_sha, images_dir=images_dir, group=group_name(cand), index=index)
        for _, ref, _ in todo[url]:
            apply_outcome(ref, rec)
        with counts_lock:
            counts[rec["status"]] += 1

    if urls:
        print(f"  fetch: {len(urls):,} URLs ({len(todo):,} pending)", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(one, urls))
    for cand in cands:
        counts["duplicates_marked"] += mark_duplicates(cand.images)
    return counts


def _fetch_one(url: str, *, head: Callable[[str], int], get: Callable[[str], bytes], retries: int, pacer: HostPacer,
               sleep: SleepFn, state: FetchState, local_by_sha: Mapping[str, str], images_dir: Path,
               group: str, index: int) -> Dict[str, Any]:
    """Fetch, hash and store one URL, and record the outcome.

    HTTP calls, decoding the body and writing the file are the I/O whose failure becomes the ``missing`` status.

    :param url: Image URL.
    :param head: HEAD callable.
    :param get: GET callable.
    :param retries: Extra attempts.
    :param pacer: Per-host pacer.
    :param sleep: Sleep function.
    :param state: Fetch state.
    :param local_by_sha: Local files (paths) of the run's candidates by sha256.
    :param images_dir: ``work_dir/images``.
    :param group: Directory of the first candidate listing the URL.
    :param index: The reference's index in that candidate.
    :return: The recorded outcome.
    """
    def record(rec: Dict[str, Any]) -> Dict[str, Any]:
        """Append the outcome to the state under its lock.

        :param rec: Outcome.
        :return: The same outcome.
        """
        with state.lock:
            state.append(rec)
        return rec

    rec: Dict[str, Any] = {"url": url, "status": "missing", "path": None, "sha256": None, "phash": None,
                           "width": None, "height": None, "note": "",
                           "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if is_bucket_url(url):
        def checked_head(u: str) -> int:
            """HEAD with transient statuses raised for a retry.

            :param u: URL.
            :return: Status code.
            """
            code = int(head(u))
            if code in TRANSIENT_HTTP:
                raise TransientStatusError(f"status {code}")
            return code

        code, err = call_with_retries(checked_head, url, retries=retries, pacer=pacer, sleep=sleep)
        if err is not None:
            return record({**rec, "note": f"head_failed {err}"})
        if code == 404:
            return record({**rec, "status": "head_404", "note": "head 404"})
        if code >= 400:
            return record({**rec, "note": f"head_status_{code}"})
    data, err = call_with_retries(get, url, retries=retries, pacer=pacer, sleep=sleep)
    if err is not None or not data:
        return record({**rec, "note": f"get_failed {err}" if err is not None else "get_failed empty body"})
    try:
        keys = image_keys(data)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        return record({**rec, "note": f"undecodable {type(exc).__name__}: {exc}"[:200]})
    rec.update({k: keys[k] for k in ("sha256", "phash", "width", "height")})
    notes = [f"exif_orientation {keys['orientation']}"] if keys["orientation"] not in (0, 1) else []
    with state.lock:
        earlier = state.by_sha.get(keys["sha256"])
        same = earlier["path"] if earlier is not None and Path(earlier["path"]).is_file() \
            else local_by_sha.get(keys["sha256"])
        rec["path"] = same or str(_dest_for(state, images_dir, group, index))
    if same:
        notes.append(f"same bytes as {same}")
    else:
        try:
            reencoded = write_upright(data, Path(rec["path"]))
        except (OSError, ValueError) as exc:
            return record({**rec, "path": None, "note": f"write_failed {type(exc).__name__}: {exc}"[:200]})
        if reencoded:
            notes.append("reencoded")
    long_side = max(int(rec["width"] or 0), int(rec["height"] or 0))
    return record({**rec, "status": "too_small" if long_side < MIN_LONG_SIDE else "fetched", "note": "; ".join(notes)})


# --------------------------------------------------------------------------------------------------- inventory

_INV_COLUMNS = ("candidates", "none", "listed", "present", "fetched", "url-only", "missing", "too_small", "head_404",
                "duplicate", "box_listing", "distinct sha256", "usable cands", "box unmatched cands",
                "unexpanded manifests", "max/cand")


def _inventory_row(cands: Sequence[Candidate]) -> Dict[str, int]:
    """Counts of one inventory row.

    :param cands: Candidates of the row.
    :return: Counts by column.
    """
    refs = [r for c in cands for r in c.images]
    status = Counter(r.status for r in refs)
    return {
        "candidates": len(cands),
        "none": sum(1 for c in cands if not c.images),
        "listed": len(refs),
        "present": status["present"],
        "fetched": status["fetched"],
        "url-only": sum(1 for r in refs if r.status == "listed" and r.url and not r.path),
        "missing": status["missing"],
        "too_small": status["too_small"],
        "head_404": status["head_404"],
        "duplicate": status["duplicate"],
        "box_listing": status["box_listing"],
        "distinct sha256": len({r.sha256 for r in refs if r.sha256}),
        "usable cands": sum(1 for c in cands if any(r.status in USABLE_STATUSES for r in c.images)),
        "box unmatched cands": sum(1 for c in cands if (c.notes.get(NOTES_KEY) or {}).get("box_listing_unmatched")),
        "unexpanded manifests": sum(len((c.notes.get(NOTES_KEY) or {}).get("manifests_unexpanded") or [])
                                    for c in cands),
        "max/cand": max((len(c.images) for c in cands), default=0),
    }


def inventory(cands: Sequence[Candidate]) -> str:
    """Markdown inventory of the candidates' images: one row per source (and a total), then counts by origin.

    Columns: ``candidates``; ``none`` = candidates without any image; ``listed`` = image references;
    ``present`` = local files found; ``fetched`` = downloaded; ``url-only`` = known by URL, not fetched yet;
    ``missing`` = failed (see the reference notes); ``too_small`` (long side under the minimum, file kept);
    ``head_404``; ``duplicate`` (same bytes as another image of the candidate); ``box_listing`` = left out by the
    fetch cap; ``distinct sha256``; ``usable cands`` = candidates with at least one present or fetched image;
    ``box unmatched cands`` = candidates whose URL listing named none of their marks (review cases);
    ``unexpanded manifests`` = IIIF manifests not available (offline or failed); ``max/cand`` = most images of one
    candidate.

    :param cands: Candidates with completed images.
    :return: Markdown text.
    """
    by_source: Dict[str, List[Candidate]] = defaultdict(list)
    for cand in cands:
        by_source[cand.source].append(cand)
    lines = ["# Stage 1 image inventory", "",
             f"{len(cands):,} candidates; minimum long side {MIN_LONG_SIDE} px.", "",
             "| source | " + " | ".join(_INV_COLUMNS) + " |",
             "|---|" + "---:|" * len(_INV_COLUMNS)]
    for source in sorted(by_source):
        row = _inventory_row(by_source[source])
        lines.append(f"| {source} | " + " | ".join(f"{row[c]:,}" for c in _INV_COLUMNS) + " |")
    total = _inventory_row(cands)
    lines.append("| **total** | " + " | ".join(f"{total[c]:,}" for c in _INV_COLUMNS) + " |")
    origin_status: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for cand in cands:
        for ref in cand.images:
            origin_status[(cand.source, ref.origin)][ref.status] += 1
    statuses = ("present", "fetched", "listed", "missing", "too_small", "head_404", "duplicate", "box_listing")
    lines += ["", "## Images by origin", "", "| source | origin | images | " + " | ".join(statuses) + " |",
              "|---|---|---:|" + "---:|" * len(statuses)]
    for (source, origin), counts in sorted(origin_status.items()):
        lines.append(f"| {source} | {origin} | {sum(counts.values()):,} | "
                     + " | ".join(f"{counts[s]:,}" for s in statuses) + " |")
    misses: Counter = Counter()
    for cand in cands:
        for miss in (cand.notes.get(NOTES_KEY) or {}).get("misses") or []:
            misses[(cand.source, str(miss).split(" ", 1)[0])] += 1
        for item in (cand.notes.get(NOTES_KEY) or {}).get("manifests_unexpanded") or []:
            misses[(cand.source, f"manifest:{item.get('reason', '')[:40]}")] += 1
    if misses:
        lines += ["", "## Resolution misses", "", "| source | reason | count |", "|---|---|---:|"]
        for (source, reason), n in sorted(misses.items()):
            lines.append(f"| {source} | {reason} | {n:,} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------- CLI

_SESSIONS = threading.local()
USER_AGENT = "novel-build-images/1.0 (research corpus builder; python-requests)"


def _session() -> Any:
    """A per-thread ``requests`` session.

    :return: Session.
    """
    import requests
    if getattr(_SESSIONS, "session", None) is None:
        _SESSIONS.session = requests.Session()
        _SESSIONS.session.headers["User-Agent"] = USER_AGENT
    return _SESSIONS.session


def _raise_for_status(status: int, url: str) -> None:
    """Raise for an HTTP error status (4xx other than the transient ones are permanent).

    :param status: HTTP status.
    :param url: URL (for the message).
    :raises PermanentFetchError: On a permanent client error.
    :raises RuntimeError: On another error status.
    """
    if status >= 400:
        if status < 500 and status not in TRANSIENT_HTTP:
            raise PermanentFetchError(f"HTTP {status}")
        raise RuntimeError(f"HTTP {status} for {host_of(url)}")


def http_head(url: str, timeout: float = 30.0) -> int:
    """HEAD request.

    :param url: URL.
    :param timeout: Seconds.
    :return: HTTP status.
    """
    return int(_session().head(url, timeout=timeout, allow_redirects=True).status_code)


def http_get(url: str, timeout: float = 120.0) -> bytes:
    """GET request for an image.

    :param url: URL.
    :param timeout: Seconds.
    :return: Body bytes.
    """
    resp = _session().get(url, timeout=timeout)
    _raise_for_status(resp.status_code, url)
    return resp.content


def http_get_json(url: str, timeout: float = 60.0) -> Dict[str, Any]:
    """GET request for a IIIF manifest.

    :param url: URL.
    :param timeout: Seconds.
    :return: Parsed JSON object.
    :raises PermanentFetchError: When the body is not a JSON object.
    """
    resp = _session().get(url, timeout=timeout, headers={"Accept": "application/ld+json, application/json"})
    _raise_for_status(resp.status_code, url)
    try:
        data = resp.json()
    except ValueError as exc:
        raise PermanentFetchError(f"not JSON ({type(exc).__name__})") from exc
    if not isinstance(data, dict):
        raise PermanentFetchError("JSON is not an object")
    return data


def merge_stage1(stage1_path: Path, all_cands: Sequence[Candidate], processed: Sequence[Candidate]) -> List[Candidate]:
    """Stage-1 records: this run's candidates, plus earlier stage-1 images of candidates not processed now (their
    other fields refreshed from stage 0), in stage-0 order.

    :param stage1_path: Existing stage-1 file (may be absent).
    :param all_cands: Every stage-0 candidate.
    :param processed: Candidates resolved in this run.
    :return: Records to write.
    """
    done = {c.row_id: c for c in processed}
    earlier = {c.row_id: c for c in read_jsonl(stage1_path, Candidate)} if stage1_path.is_file() else {}
    out: List[Candidate] = []
    for cand in all_cands:
        if cand.row_id in done:
            out.append(done[cand.row_id])
        elif cand.row_id in earlier:
            prev = earlier[cand.row_id]
            cand.images = prev.images
            if NOTES_KEY in prev.notes:
                cand.notes[NOTES_KEY] = prev.notes[NOTES_KEY]
            out.append(cand)
    return out


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface.

    :return: Parser.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--work-dir", type=Path, default=paths.DEFAULT_WORK_DIR, help="build work dir")
    p.add_argument("--inventory", action="store_true", help="resolve offline and write the inventory (no requests)")
    p.add_argument("--fetch", action="store_true", help="fetch uncached manifests and images known only by URL")
    p.add_argument("--sources", default="", help="comma-separated sources (default: all)")
    p.add_argument("--limit", type=int, default=None,
                   help="at most N candidates per source, in stage-0 order (as the stage-0 CLI counts)")
    p.add_argument("--workers", type=int, default=2, help="threads for requests and local hashing")
    p.add_argument("--delay", type=float, default=0.3, help="minimum seconds between requests to one host")
    p.add_argument("--retries", type=int, default=2, help="extra attempts per request")
    p.add_argument("--max-per-candidate", type=int, default=MAX_PER_CANDIDATE,
                   help="fetch at most N URL-only photographs per candidate, chosen by file name (0: no cap)")
    p.add_argument("--pgp-export-dir", type=Path, default=DEFAULT_PGP_EXPORT_DIR)
    p.add_argument("--merged", type=Path, default=DEFAULT_MERGED, help="merged corpus JSONL")
    p.add_argument("--corrections", type=Path, default=DEFAULT_CORRECTIONS, help="image corrections JSONL")
    p.add_argument("--no-corrections", action="store_true", help="apply no image corrections")
    p.add_argument("--legacy-manifest", type=Path, nargs="+", default=list(DEFAULT_LEGACY_MANIFESTS),
                   help="legacy-store image manifest(s)")
    p.add_argument("--run2-local-images", type=Path, default=DEFAULT_RUN2_LOCAL_IMAGES)
    p.add_argument("--ktiv-dir", type=Path, default=DEFAULT_KTIV_DIR, help="KTIV image zips")
    p.add_argument("--ktiv-cache", type=Path, default=None, help="KTIV page cache (default: <work-dir>/ktiv_cache)")
    p.add_argument("--bodleian-dir", type=Path, default=DEFAULT_BODLEIAN_DIR, help="local Bodleian masters")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    """Resolve (and with ``--fetch`` download) the images of the stage-0 candidates; write stage 1 and the inventory.

    :param argv: Arguments (default: ``sys.argv``).
    """
    args = build_parser().parse_args(argv)
    if not (args.inventory or args.fetch):
        raise SystemExit("pass --inventory (offline) and/or --fetch")
    work_dir: Path = args.work_dir
    stage0 = paths.stage_path(work_dir, "candidates")
    all_cands: List[Candidate] = list(read_jsonl(stage0, Candidate))
    wanted = {s.strip() for s in args.sources.split(",") if s.strip()}
    taken: Counter = Counter()
    selected: List[Candidate] = []
    for cand in all_cands:
        if (wanted and cand.source not in wanted) or (args.limit is not None and taken[cand.source] >= args.limit):
            continue
        taken[cand.source] += 1
        selected.append(cand)
    merged_ids = {cid for c in all_cands if c.source in MERGED_SOURCES for cid in c.canonical_ids}
    t0 = time.time()
    stats = resolve_images(
        selected, pgp_export_dir=args.pgp_export_dir, merged_path=args.merged,
        corrections_path=None if args.no_corrections else args.corrections, work_dir=work_dir,
        legacy_manifest=args.legacy_manifest, run2_local_images=args.run2_local_images, ktiv_dir=args.ktiv_dir,
        ktiv_cache=args.ktiv_cache or work_dir / "ktiv_cache", fetch_manifest=http_get_json if args.fetch else None,
        bodleian_dir=args.bodleian_dir, merged_ids=merged_ids, workers=args.workers, delay=args.delay,
        retries=args.retries, max_per_candidate=args.max_per_candidate)
    print(f"resolved {len(selected):,}/{len(all_cands):,} candidates in {time.time() - t0:.0f}s: "
          f"{dict(sorted(stats.items()))}", flush=True)
    if args.fetch:
        counts = fetch_missing(selected, work_dir, head=http_head, get=http_get, delay=args.delay,
                               retries=args.retries, workers=args.workers)
    else:
        counts = apply_fetch_state(selected, work_dir)
    print(f"fetch outcomes: {dict(sorted(counts.items()))}", flush=True)
    stage1 = paths.stage_path(work_dir, "images")
    records = merge_stage1(stage1, all_cands, selected)
    n = write_jsonl(stage1, records)
    report = inventory(records)
    _atomic_write_text(work_dir / INVENTORY_FILE, report)
    print(f"wrote {n:,} candidates -> {stage1}\nwrote {work_dir / INVENTORY_FILE}\n", flush=True)
    print(report, flush=True)


if __name__ == "__main__":
    main()
