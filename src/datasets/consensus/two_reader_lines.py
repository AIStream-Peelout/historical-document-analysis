# File name: two_reader_lines.py
# Date: 9/8/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Two-reader line reads for the website: Kraken lines + grounded VLM lines.

For every (document, image) job this reads the image with the Kraken line
service (``/transcribe_lines``) and with the grounded Qwen3-VL checkpoint
(JSON lines with ``bbox_2d``), matches the two under
:mod:`src.datasets.consensus.line_rule` and appends ONE JSONL record in the
envelope the site's loader ingests as-is
(``genizah_search/scripts/load_ai_transcriptions.py``)::

    {"doc_id": <ES _id in the merged index>, "source_index": <reference index, default the newest genizah_merged_v<N>>,
     "image_index": <position in the doc's image_urls>, "image_url": <that URL>,
     "image_width": W, "image_height": H,            # after EXIF orientation
     "image_sha256": <hash of the downloaded bytes>,
     "ai_read": {"vlm_model", "vlm_revision", "htr_model", "rule_version",
                 "decoded_at", "parsed", "n_lines", "n_agreed", "lines": [...]}}

Coordinates are 0-1000 normalised to the oriented image both readers saw.
Each line's ``bbox`` is the evidence box (VLM box ∪ its Kraken fragments);
``raw/<doc>__<image>__<vlm>.json`` keeps both readers' outputs so a rule change
is a ``--rematch`` (seconds per page) rather than a re-read, and a better Kraken
model is a ``--rekraken`` (one Kraken read per image, no LM Studio): its
fragments are cached under ``frags_by_htr[<key>]`` beside the original reader's
``frags`` and the records are rebuilt from the cached VLM lines.  A Kraken
service swap that keeps the recognition model (new kraken version or
segmenter) is a ``--htr-cache-key KEY`` run: the raw cache files that run's
``frags`` under ``htr_model = KEY`` and every record carries
``ai_read.htr_cache_key = KEY`` (``htr_model`` stays the model's name), the
layout ``stamp_htr_cache_key.py`` gives reads made before the option existed.
``parsed`` is True when the VLM reply was a JSON array with at least one
valid line object (a truncated array still counts; prose or nothing does
not).  Sequential on purpose (LM Studio must see one request at a time),
resumable by ``(doc_id, image_index)``; the next image's download + Kraken read
(CPU) overlaps the current VLM read (GPU); images deleted after each job; stops
when free disk drops under ``--min-free-gb``.  Infrastructure failures
(download, Kraken, LM Studio) are logged to ``<out>.failures.jsonl`` and NOT
recorded, so a re-run retries them.

Usage (repo root; long runs under nohup):
    .venv/bin/python -m src.datasets.consensus.two_reader_lines --from-consensus high
    .venv/bin/python -m src.datasets.consensus.two_reader_lines --ids jobs.jsonl --limit 50
    .venv/bin/python -m src.datasets.consensus.two_reader_lines --ids jobs.jsonl --htr-cache-key MiDRASH_Gen_01@k7.0.3-blla2026
    .venv/bin/python -m src.datasets.consensus.two_reader_lines --rekraken --kraken-model NEW.mlmodel
"""
import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps
from dotenv import load_dotenv

load_dotenv()

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
from src.datasets.consensus.line_rule import (  # noqa: E402
    RULE_VERSION, assign_fragments, evidence_box, letters, line_status, similarity)
from src.models.ocr.kraken_transcriber import transcribe_with_kraken_lines  # noqa: E402
from src.models.ocr.lms_transcriber import transcribe_with_lm_studio  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

OUT_DIR = _REPO / "src/datasets/raw_data/cairo_genizah/ai_reads"
RAW_DIR = OUT_DIR / "raw"      # per-job reader outputs; a rule change is a re-match, not a re-read
CONSENSUS_RESULTS = _REPO / "src/datasets/raw_data/cairo_genizah/consensus_pilot/pilot_results.jsonl"
KRAKEN_MODEL = str(_REPO / "src/datasets/raw_data/cairo_genizah/custom_model_weights/MiDRASH_Gen_01.mlmodel")
HTR_MODEL_NAME = "MiDRASH_Gen_01"
LEGACY_HTR_MODEL = "MiDRASH_Gen_01"  # reader of every raw-cache ``frags`` list written before entries named theirs; never change
REKRAKEN_MAX_KRAKEN_FAILURES = 10    # --rekraken stops after this many Kraken failures in a row (service down / model not mounted)
VLM_MODEL = "qwen3-vl-8b-heb-v21b-step1200"
VLM_REVISION = "6724c32c"          # hub commit of the v21b step-1200 checkpoint (flagship 2026-09-13; v2.0a-1800 was af9df6a0)
SITE_ENV = Path.home() / "Documents/GitHub/genizah_search/src/backend/.env"
# Byte-identical to grounding_eval.P_GROUNDED — the prompt the probe numbers were measured with.
P_GROUNDED = ('Transcribe this manuscript page line by line. Respond with ONLY '
              'a JSON array; each element {"text": "...", "bbox_2d": [x1, y1, '
              'x2, y2]} gives one line\'s transcription and its bounding box. '
              'Preserve reading order.')
VLM_MAX_TOKENS = 3500
_NUM = (int, float)


# ---------------------------------------------------------------------------
# Elasticsearch (read-only lookups of the served merged index)
# ---------------------------------------------------------------------------

def load_site_env(path: Path) -> None:
    """Load ES credentials from the site's env file into ``os.environ`` (no overrides).

    :param path: ``.env`` file of the website backend.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _es_base_url() -> str:
    """Resolve the Elasticsearch base URL for the consensus pipeline.

    ``ES_URL`` wins; otherwise the URL is assembled from the repo-wide
    ``ELASTIC_SEARCH_SCHEME`` / ``ELASTIC_SEARCH_HOST`` / ``ELASTIC_SEARCH_PORT``
    settings in ``.env`` (the same ones ``es_config_from_env`` uses), so this
    script writes to the same cluster as every other indexer. Falls back to the
    local Docker node only when nothing is configured.

    :return: Base URL without a trailing slash.
    """
    explicit = os.environ.get("ES_URL")
    if explicit:
        return explicit.rstrip("/")
    host = os.environ.get("ELASTIC_SEARCH_HOST")
    if not host:
        return "http://localhost:9200"
    scheme = os.environ.get("ELASTIC_SEARCH_SCHEME", "https")
    port = os.environ.get("ELASTIC_SEARCH_PORT", "443")
    return f"{scheme}://{host}:{port}"


def _es_request(method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Any:
    """Authenticated JSON request against the Elasticsearch selected by :func:`_es_base_url`.

    Credentials come from ``ELASTICSEARCH_USERNAME`` / ``ELASTICSEARCH_USER`` /
    ``ELASTICSEARCH_PASSWORD`` when exported, else the repo-wide ``ELASTIC_USER`` /
    ``ELASTIC_PASSWORD`` from ``.env``.
    """
    url = _es_base_url()
    user = (os.environ.get("ELASTICSEARCH_USERNAME") or os.environ.get("ELASTICSEARCH_USER")
            or os.environ.get("ELASTIC_USER", ""))
    password = os.environ.get("ELASTICSEARCH_PASSWORD") or os.environ.get("ELASTIC_PASSWORD", "")
    auth = base64.b64encode(f"{user}:{password}".encode()).decode()
    req = urllib.request.Request(url + path, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": "Basic " + auth, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def latest_merged_index() -> str:
    """Highest-numbered ``genizah_merged_v<N>`` index (the successor index is always the reference).

    :returns: Index name.
    """
    names = [row["index"] for row in _es_request("GET", "/_cat/indices/genizah_merged_v*?format=json&h=index")]
    nums = [(int(m.group(1)), n) for n in names if (m := re.fullmatch(r"genizah_merged_v(\d+)", n))]
    if not nums:
        raise RuntimeError("no genizah_merged_v<N> index found")
    return max(nums)[1]


def es_get_docs(ids: List[str], index: str, fields: str = "image_urls") -> Dict[str, Dict[str, Any]]:
    """Fetch ``_source`` of the given ids via ``_mget``.

    :param ids: Elasticsearch ``_id`` values.
    :param index: Index name.
    :param fields: Comma-separated ``_source`` fields.
    :returns: ``{_id: _source}`` for the ids that exist.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for i in range(0, len(ids), 200):
        got = _es_request("POST", f"/{index}/_mget?_source={urllib.parse.quote(fields)}", {"ids": ids[i:i + 200]})
        for d in got["docs"]:
            if d.get("found"):
                out[d["_id"]] = d["_source"]
    return out


def restamp(out: Path, index: str) -> None:
    """Re-resolve every record in ``out`` against ``index`` (source_index + image_index by image_url).

    Image lists change between merged-index versions (files renamed, images
    prepended), so a record's position is recomputed from its exact URL; a
    record whose URL is no longer listed is dropped with a note.
    :param out: JSONL of records.
    :param index: Reference merged index.
    """
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    docs = es_get_docs([r["doc_id"] for r in rows], index)
    kept, changed, dropped = [], 0, 0
    for r in rows:
        urls = (docs.get(r["doc_id"]) or {}).get("image_urls") or []
        if r["image_url"] not in urls:
            dropped += 1
            continue
        idx = urls.index(r["image_url"])
        if (r["source_index"], r["image_index"]) != (index, idx):
            changed += 1
        r["source_index"], r["image_index"] = index, idx
        kept.append(r)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    print(f"restamped {len(kept)} records to {index} ({changed} changed, {dropped} dropped)")


def build_jobs(args: argparse.Namespace, index: str) -> List[Dict[str, Any]]:
    """Resolve the job list: ``(doc_id, image_index, image_url)`` triples.

    :param args: CLI arguments (``--from-consensus`` or ``--ids``).
    :param index: Served merged index (for ``image_urls`` lookups).
    :returns: Jobs in input order, unresolvable ones dropped with a note.
    """
    wanted: List[Dict[str, Any]] = []
    if args.from_consensus:
        for line in CONSENSUS_RESULTS.read_text().splitlines():
            r = json.loads(line)
            if r.get("tier") == args.from_consensus and r.get("image_url"):
                wanted.append({"doc_id": r["canonical_id"], "image_url": r["image_url"]})
    if args.ids:
        for line in Path(args.ids).read_text().splitlines():
            if line.strip():
                wanted.append(json.loads(line))
    if getattr(args, "no_resolve", False):
        # Pre-index runs (e.g. a freshly scraped image not yet in the served index): trust the
        # job's own image_index / image_url, and let ``local_path`` supply the bytes.
        jobs = [{"doc_id": w["doc_id"], "image_index": int(w["image_index"]), "image_url": w["image_url"],
                 **({"local_path": w["local_path"]} if w.get("local_path") else {})}
                for w in wanted if w.get("image_url") is not None and "image_index" in w]
        print(f"{len(jobs)} jobs taken as given (--no-resolve; source_index stamp {index})")
        return jobs
    docs = es_get_docs([w["doc_id"] for w in wanted], index)
    jobs, dropped = [], 0
    for w in wanted:
        urls = (docs.get(w["doc_id"]) or {}).get("image_urls") or []
        if "image_index" in w and w["image_index"] < len(urls):
            idx = int(w["image_index"])
        elif w.get("image_url") in urls:
            idx = urls.index(w["image_url"])
        else:
            dropped += 1
            continue
        jobs.append({"doc_id": w["doc_id"], "image_index": idx, "image_url": urls[idx]})
    if dropped:
        print(f"dropped {dropped} jobs whose doc/image is not in {index}")
    return jobs


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def download(url: str, timeout: float = 60.0, tries: int = 3) -> Optional[bytes]:
    """Download ``url`` with retries.

    :param url: Image URL.
    :param timeout: Per-attempt timeout in seconds.
    :param tries: Attempts.
    :returns: Bytes, or None when every attempt failed.
    """
    # A few served image names carry a literal space (macOS "name 2.jpg" duplicates); urllib refuses
    # those as InvalidURL, while the object exists under the percent-encoded name.
    url = urllib.parse.quote(url, safe=":/?&=%#")
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 — network; retried, then reported by the caller
            if attempt == tries - 1:
                print(f"    download failed: {exc}")
            time.sleep(2 * (attempt + 1))
    return None


def prepare_image(data: bytes, dest: Path) -> Tuple[int, int]:
    """Write the image both readers will see, EXIF orientation applied.

    :param data: Downloaded bytes.
    :param dest: Path to write (JPEG).
    :returns: ``(width, height)`` of the oriented image.
    """
    with Image.open(io.BytesIO(data)) as im:
        oriented = ImageOps.exif_transpose(im)
        orientation = (im.getexif() or {}).get(0x0112, 1)
        if orientation in (None, 1) and im.format == "JPEG":
            dest.write_bytes(data)
        else:
            oriented.convert("RGB").save(dest, format="JPEG", quality=95)
        return oriented.width, oriented.height


def _clamp_box(box: List[float]) -> List[int]:
    """Clamp to 0-1000 ints and order corners (mirrors the site's ``clamp_bbox``)."""
    x1, y1, x2, y2 = (int(round(max(0.0, min(1000.0, float(v))))) for v in box)
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def parse_grounded(raw: Optional[str]) -> Tuple[bool, List[Dict[str, Any]]]:
    """Parse the VLM's JSON-array reply, tolerating truncation and trailing prose.

    :param raw: Model output.
    :returns: ``(parsed, lines)`` with ``lines`` as ``{text, box}`` (box 0-1000 floats).
    """
    if not raw or not raw.strip():
        return False, []
    text = raw.strip()
    objs: List[Any] = []
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        try:
            arr = json.loads(m.group(0))
            objs = arr if isinstance(arr, list) else []
        except json.JSONDecodeError:
            objs = []
    if not objs:                                     # truncated array: recover closed objects
        depth, start, in_str, esc = 0, None, False, False
        for i, ch in enumerate(text):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        objs.append(json.loads(text[start:i + 1]))
                    except json.JSONDecodeError:
                        pass
                    start = None
    lines = []
    for o in objs:
        if not isinstance(o, dict):
            continue
        t, b = str(o.get("text", "")).strip(), o.get("bbox_2d")
        if t and isinstance(b, list) and len(b) == 4 and all(isinstance(v, _NUM) for v in b):
            lines.append({"text": t, "box": [float(v) for v in b]})
    parsed = bool(lines) and text.startswith("[")
    return parsed, lines


def drop_loop_duplicates(vlm_lines: List[Dict[str, Any]], min_letters: int = 8) -> List[Dict[str, Any]]:
    """Drop later exact repeats (letters-only) of a line of >= ``min_letters`` letters.

    A looping decode re-emits the same line many times; repeats can never be
    confirmed (fragments are assigned once) and would only clutter the page.
    Short genuine repeats (e.g. a two-word refrain) are kept.
    :param vlm_lines: Parsed lines in reading order.
    :param min_letters: Minimum letters for a line to be subject to dedupe.
    :returns: Lines with loop repeats removed.
    """
    seen, out = set(), []
    for ln in vlm_lines:
        key = letters(ln["text"])
        if len(key) >= min_letters and key in seen:
            continue
        seen.add(key)
        out.append(ln)
    return out


def sidecar(vlm_lines: List[Dict[str, Any]], frags: List[Dict[str, Any]], parsed: bool,
            vlm_model: str, vlm_revision: str, htr_model: str = HTR_MODEL_NAME) -> Dict[str, Any]:
    """Match the two readings and build the ``ai_read`` block.

    :param vlm_lines: ``{text, box}`` from :func:`parse_grounded`.
    :param frags: Kraken fragments ``{text, conf, box}`` (0-1000).
    :param parsed: Whether the VLM reply parsed.
    :param vlm_model: LM Studio key.
    :param vlm_revision: Checkpoint revision string.
    :param htr_model: Name of the Kraken model that produced ``frags``.
    :returns: Sidecar dict in the loader's schema.
    """
    vlm_lines = drop_loop_duplicates(vlm_lines)
    groups = assign_fragments(vlm_lines, frags)
    lines = []
    for i, ln in enumerate(vlm_lines):
        htr_text = " ".join(frags[j]["text"] for j in groups[i]).strip() or None
        agreement = round(similarity(ln["text"], htr_text), 4) if htr_text else None
        lines.append({
            "index": i, "text": ln["text"],
            "bbox": _clamp_box(evidence_box(ln["box"], [frags[j]["box"] for j in groups[i]])),
            "agreement": agreement, "status": line_status(agreement, htr_text),
            "htr_text": htr_text, "htr_fragments": [_clamp_box(frags[j]["box"]) for j in groups[i]],
        })
    return {
        "vlm_model": vlm_model, "vlm_revision": vlm_revision, "htr_model": htr_model,
        "rule_version": RULE_VERSION,
        "decoded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parsed": parsed, "n_lines": len(lines),
        "n_agreed": sum(l["status"] == "agreed" for l in lines), "lines": lines,
    }


async def run_kraken(path: Path, model_path: str, width: int, height: int,
                     timeout: float = 300.0) -> Optional[List[Dict[str, Any]]]:
    """Kraken-read one prepared image into fragments with 0-1000 boxes.

    :param path: Oriented JPEG written by :func:`prepare_image`.
    :param model_path: Kraken model (host path; the service finds it by basename in its ``/app/models`` mount).
    :param width: Oriented image width in pixels.
    :param height: Oriented image height in pixels.
    :param timeout: Service request timeout in seconds.
    :returns: ``{text, conf, box}`` for every line with text, or None when the service call failed.
    """
    kr = await transcribe_with_kraken_lines(model_path, str(path), timeout=timeout)
    if kr is None:
        return None
    return [{"text": l["text"], "conf": l.get("confidence"),
             "box": [1000 * l["bbox"][0] / width, 1000 * l["bbox"][1] / height,
                     1000 * l["bbox"][2] / width, 1000 * l["bbox"][3] / height]}
            for l in kr.get("lines", []) if l.get("text", "").strip()]


async def stage1(job: Dict[str, Any], work_dir: Path) -> Dict[str, Any]:
    """Download, orient and Kraken-read one image (CPU side).

    :param job: ``{doc_id, image_index, image_url}``.
    :param work_dir: Where the oriented JPEG is written.
    :returns: ``info`` with ``path``, ``width``, ``height``, ``sha256``, ``frags`` — or a ``failure`` key.
    """
    info: Dict[str, Any] = {"doc_id": job["doc_id"], "image_index": job["image_index"]}
    t0 = time.time()
    if job.get("local_path"):
        data = Path(job["local_path"]).read_bytes() if Path(job["local_path"]).exists() else None
    else:
        data = download(job["image_url"])
    if data is None:
        info["failure"] = "download"
        return info
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", job["doc_id"]) + f"__{job['image_index']}"
    path = work_dir / f"{stem}.jpg"
    try:
        width, height = prepare_image(data, path)
    except Exception as exc:  # noqa: BLE001 — corrupt/unsupported image is a per-job failure
        info["failure"] = f"image: {type(exc).__name__}"
        return info
    info.update(path=path, width=width, height=height, sha256=hashlib.sha256(data).hexdigest(),
                download_s=round(time.time() - t0, 1))
    t0 = time.time()
    frags = await run_kraken(path, KRAKEN_MODEL, width, height)
    info["kraken_s"] = round(time.time() - t0, 1)
    if frags is None:
        info["failure"] = "kraken"
        return info
    info["frags"] = frags
    return info


async def stage2(job: Dict[str, Any], info: Dict[str, Any], source_index: str, vlm_model: str,
                 vlm_revision: str, htr_cache_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """VLM-read the prepared image, match, and build the loader record (GPU side).

    With ``htr_cache_key`` the fragments are filed under that key exactly as
    ``stamp_htr_cache_key.py`` files an earlier read after the fact: the raw-cache
    entry keeps them as its top-level ``frags`` and its ``htr_model`` names the key
    (so :func:`cached_frags` answers them for the key and no longer for
    :data:`HTR_MODEL_NAME`), and the record's ``ai_read`` gets ``htr_cache_key``
    after ``lines`` while ``ai_read.htr_model`` stays the recognition model's name,
    as :func:`rebuild_record` stamps a ``--rekraken --kraken-cache-suffix`` record
    (so ``--rematch`` and a same-key ``--rekraken`` find the fragments again).
    :param job: The job.
    :param info: Output of :func:`stage1` (no ``failure``).
    :param source_index: Reference merged index stamped into the record.
    :param vlm_model: LM Studio key.
    :param vlm_revision: Checkpoint revision string.
    :param htr_cache_key: HTR cache key of ``info['frags']`` (``--htr-cache-key``); None writes exactly what this
        function wrote before the option existed, as does :data:`HTR_MODEL_NAME` (the default key).
    :returns: The record, or None (``info['failure']`` set) when LM Studio failed.
    """
    t0 = time.time()
    raw = await transcribe_with_lm_studio(vlm_model, str(info["path"]), P_GROUNDED, max_tokens=VLM_MAX_TOKENS)
    info["vlm_s"] = round(time.time() - t0, 1)
    if raw is None:
        info["failure"] = "vlm"
        return None
    parsed, vlm_lines = parse_grounded(raw)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_cache_path(job, vlm_model, for_write=True).write_text(json.dumps(dict(
        doc_id=job["doc_id"], image_index=job["image_index"], image_url=job["image_url"],
        width=info["width"], height=info["height"], sha256=info["sha256"], vlm_model=vlm_model,
        vlm_raw=raw, vlm_lines=vlm_lines, parsed=parsed, frags=info["frags"],
        htr_model=htr_cache_key or HTR_MODEL_NAME), ensure_ascii=False))
    ai_read = sidecar(vlm_lines, info["frags"], parsed, vlm_model, vlm_revision)
    if htr_cache_key and htr_cache_key != ai_read["htr_model"]:
        ai_read["htr_cache_key"] = htr_cache_key      # how --rematch finds these fragments again
    info.update(n_frag=len(info["frags"]), n_lines=ai_read["n_lines"], n_agreed=ai_read["n_agreed"], parsed=parsed)
    return {
        "doc_id": job["doc_id"], "source_index": source_index, "image_index": job["image_index"],
        "image_url": job["image_url"], "image_width": info["width"], "image_height": info["height"],
        "image_sha256": info["sha256"], "ai_read": ai_read,
    }


def raw_stem(job: Dict[str, Any], vlm_model: str) -> str:
    """File stem of a job's raw-cache entry (per VLM checkpoint)."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", job["doc_id"]) + f"__{job['image_index']}__{vlm_model}"


def raw_cache_path(job: Dict[str, Any], vlm_model: str, for_write: bool = False,
                   raw_dir: Optional[Path] = None) -> Path:
    """Raw-cache file for a job.

    Writes always use the per-model name.  Reads fall back to the model-less
    name of the first pilot runs only when that file was produced by the same
    VLM (checked from its ``vlm_model`` field), so one checkpoint can never
    be served another's outputs.
    :param job: ``{doc_id, image_index}``.
    :param vlm_model: LM Studio key.
    :param for_write: True when the caller is about to write the entry.
    :param raw_dir: Cache directory (default :data:`RAW_DIR`).
    :returns: Path (may not exist yet).
    """
    raw_dir = RAW_DIR if raw_dir is None else raw_dir
    new = raw_dir / f"{raw_stem(job, vlm_model)}.json"
    if for_write or new.exists():
        return new
    old = raw_dir / (re.sub(r"[^A-Za-z0-9_.-]", "_", job["doc_id"]) + f"__{job['image_index']}.json")
    if old.exists():
        try:
            if json.loads(old.read_text()).get("vlm_model") == vlm_model:
                return old
        except json.JSONDecodeError:
            pass
    return new


async def rematch(out: Path, work_dir: Path, source_index: str, vlm_revision: str) -> None:
    """Rebuild every record in ``out`` under the current rule from the raw cache.

    Records without a cache entry (written before the cache existed) get one:
    the VLM lines come from the record (their stored box is the raw VLM box in
    rule v1 records) and Kraken is re-run on the image; LM Studio is not used.
    Each record is matched against the fragments it was built from (its
    ``htr_model`` / ``htr_cache_key``), so a rule change never undoes a ``--rekraken``.
    :param out: JSONL of records (rewritten in place).
    :param work_dir: Scratch dir for images.
    :param source_index: Index stamped into rebuilt records.
    :param vlm_revision: Revision string for rebuilt records.
    """
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rebuilt, failed = [], 0
    for n, r in enumerate(rows, 1):
        cache = raw_cache_path(r, r["ai_read"]["vlm_model"])
        if cache.exists():
            c = json.loads(cache.read_text())
        else:
            info = await stage1(r, work_dir)
            if "failure" in info:
                print(f"  {n}/{len(rows)} {r['doc_id']}#{r['image_index']}: rematch FAILED {info['failure']} (record kept as is)", flush=True)
                rebuilt.append(r)
                failed += 1
                continue
            if info.get("path"):
                Path(info["path"]).unlink(missing_ok=True)
            c = dict(doc_id=r["doc_id"], image_index=r["image_index"], image_url=r["image_url"],
                     width=info["width"], height=info["height"], sha256=info["sha256"],
                     vlm_model=r["ai_read"]["vlm_model"], vlm_raw=None,
                     vlm_lines=[{"text": l["text"], "box": [float(v) for v in l["bbox"]]} for l in r["ai_read"]["lines"]],
                     parsed=r["ai_read"]["parsed"], frags=info["frags"], htr_model=HTR_MODEL_NAME)
            cache.write_text(json.dumps(c, ensure_ascii=False))
        htr_model = r["ai_read"].get("htr_model") or HTR_MODEL_NAME
        key = r["ai_read"].get("htr_cache_key") or htr_model
        frags = cached_frags(c, key)
        if frags is None:
            print(f"  {n}/{len(rows)} {r['doc_id']}#{r['image_index']}: rematch SKIPPED, no {key} fragments in "
                  f"the cache (record kept as is)", flush=True)
            rebuilt.append(r)
            failed += 1
            continue
        ai_read = sidecar(c["vlm_lines"], frags, c["parsed"], c["vlm_model"], vlm_revision, htr_model=htr_model)
        if key != htr_model:
            ai_read["htr_cache_key"] = key
        rebuilt.append({"doc_id": r["doc_id"], "source_index": source_index, "image_index": r["image_index"],
                        "image_url": r["image_url"], "image_width": c["width"], "image_height": c["height"],
                        "image_sha256": c["sha256"], "ai_read": ai_read})
        print(f"  {n}/{len(rows)} {r['doc_id']}#{r['image_index']}: agreed {r['ai_read']['n_agreed']} -> {ai_read['n_agreed']} "
              f"of {ai_read['n_lines']}", flush=True)
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rebuilt))
    print(f"rematched {len(rebuilt) - failed} records under {RULE_VERSION}; {failed} left unchanged -> {out}", flush=True)


# ---------------------------------------------------------------------------
# Kraken-only re-run (--rekraken)
# ---------------------------------------------------------------------------

def cached_frags(entry: Dict[str, Any], key: str) -> Optional[List[Dict[str, Any]]]:
    """Kraken fragments a raw-cache entry holds for the HTR cache key ``key``.

    ``frags_by_htr[key]`` (written by :func:`rekraken`) wins.  The top-level
    ``frags`` list is the original reader's: the model named in the entry's
    ``htr_model`` field, or :data:`LEGACY_HTR_MODEL` for entries written
    before that field existed; it answers for that key only.
    :param entry: Parsed raw-cache JSON.
    :param key: HTR cache key (the HTR model name unless ``--kraken-cache-suffix`` set another).
    :returns: The fragments, or None when the entry has none for ``key``.
    """
    by_htr = entry.get("frags_by_htr") or {}
    if key in by_htr:
        return by_htr[key]
    if key == entry.get("htr_model", LEGACY_HTR_MODEL):
        return entry.get("frags")
    return None


def write_json_atomic(path: Path, obj: Any) -> None:
    """Replace ``path`` with ``obj`` as compact JSON via a temp file and a rename.

    A raw-cache entry holds a VLM read that cost a GPU minute; an interrupted
    rewrite must never leave it truncated.
    :param path: Destination file.
    :param obj: JSON-serialisable value.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    os.replace(tmp, path)


def store_frags(path: Path, cache_key: str, frags: List[Dict[str, Any]]) -> None:
    """Add ``frags`` to a raw-cache entry as ``frags_by_htr[cache_key]``; every other field stays as it was.

    :param path: Raw-cache file.
    :param cache_key: HTR cache key.
    :param frags: Fragments from :func:`run_kraken`.
    """
    entry = json.loads(path.read_text())
    entry.setdefault("frags_by_htr", {})[cache_key] = frags
    write_json_atomic(path, entry)


def rekraken_names(kraken_model: Optional[str], htr_model_name: Optional[str],
                   cache_suffix: Optional[str]) -> Tuple[str, str, str]:
    """Resolve ``--rekraken``'s Kraken model path, stamped HTR model name and cache key.

    :param kraken_model: ``--kraken-model`` (None = :data:`KRAKEN_MODEL`).
    :param htr_model_name: ``--htr-model-name`` (None = the model file's stem, which is
        :data:`HTR_MODEL_NAME` for the default model), so a new model file is never
        stamped with the old model's name.
    :param cache_suffix: ``--kraken-cache-suffix`` (None = the HTR model name).
    :returns: ``(model_path, htr_model, cache_key)``.
    """
    model_path = kraken_model or KRAKEN_MODEL
    htr_model = htr_model_name or (Path(kraken_model).stem if kraken_model else HTR_MODEL_NAME)
    return model_path, htr_model, cache_suffix or htr_model


def rebuild_record(record: Dict[str, Any], entry: Dict[str, Any], cache_key: str,
                   htr_model: str) -> Optional[Dict[str, Any]]:
    """Re-match a record's cached VLM read against the entry's ``cache_key`` fragments.

    The envelope (``source_index``, ``image_index``, ``image_url``) and the VLM
    side (lines, ``parsed``, model, revision) are kept; matching runs under the
    current rule.
    :param record: Record from ``--out``.
    :param entry: Its raw-cache entry.
    :param cache_key: ``frags_by_htr`` key to match against.
    :param htr_model: Name stamped as ``ai_read.htr_model``.
    :returns: The rebuilt record, or None when the entry has no fragments for ``cache_key``.
    """
    frags = cached_frags(entry, cache_key)
    if frags is None:
        return None
    ai_read = sidecar(entry["vlm_lines"], frags, entry["parsed"], record["ai_read"]["vlm_model"],
                      record["ai_read"].get("vlm_revision"), htr_model=htr_model)
    if cache_key != htr_model:
        ai_read["htr_cache_key"] = cache_key          # how --rematch finds these fragments again
    return {"doc_id": record["doc_id"], "source_index": record["source_index"], "image_index": record["image_index"],
            "image_url": record["image_url"], "image_width": entry["width"], "image_height": entry["height"],
            "image_sha256": entry["sha256"], "ai_read": ai_read}


def _image_stem(doc_id: str, image_index: int) -> str:
    """File stem :func:`stage1` gives a job's prepared image in ``--work-dir``.

    :param doc_id: Document id.
    :param image_index: Image position.
    :returns: Stem without extension.
    """
    return re.sub(r"[^A-Za-z0-9_.-]", "_", doc_id) + f"__{image_index}"


async def rekraken_page(paths: List[Path], meta: Dict[Path, Dict[str, Any]], model_path: str, cache_key: str,
                        work_dir: Path, keep_images: bool = False) -> Dict[str, Any]:
    """Kraken-read one image anew and store the fragments in each of its raw-cache entries.

    The image is a kept ``work_dir`` copy whose bytes hash to the cached
    ``sha256``, else a fresh download that must hash to it: the VLM read those
    exact bytes, so a changed image is skipped, never re-read under old lines.
    :param paths: Raw-cache files of one image (one ``sha256``).
    :param meta: Per file ``doc_id``, ``image_index``, ``image_url``, ``sha256``, ``width``, ``height``.
    :param model_path: Kraken model.
    :param cache_key: ``frags_by_htr`` key to write.
    :param work_dir: Where kept images are looked up and the download is prepared.
    :param keep_images: Keep the prepared download.
    :returns: ``status`` (``ok``, or the failure: ``download``, ``sha256``, ``image``, ``kraken``), ``note``,
        ``download_s``, ``kraken_s`` and, on success, ``frags``.
    """
    first = meta[paths[0]]
    res: Dict[str, Any] = {"status": "ok", "note": "", "download_s": 0.0, "kraken_s": 0.0}
    width, height = first["width"], first["height"]
    t0 = time.time()
    path, fresh = None, False
    for m in (meta[p] for p in paths):
        kept = work_dir / f"{_image_stem(m['doc_id'], m['image_index'])}.jpg"
        if kept.exists() and hashlib.sha256(kept.read_bytes()).hexdigest() == first["sha256"]:
            path, res["note"] = kept, "kept image"
            break
    if path is None:
        data = None
        for url in dict.fromkeys(meta[p]["image_url"] for p in paths):
            data = download(url)
            if data is not None:
                break
        if data is None:
            return {**res, "status": "download"}
        sha = hashlib.sha256(data).hexdigest()
        if sha != first["sha256"]:
            return {**res, "status": "sha256", "note": f"served {sha[:12]} != cached {(first['sha256'] or '-')[:12]}"}
        path = work_dir / f"{_image_stem(first['doc_id'], first['image_index'])}.jpg"
        fresh = not path.exists()
        try:
            size: Any = prepare_image(data, path)
        except Exception as exc:  # noqa: BLE001 — corrupt/unsupported image is a per-page failure
            size = type(exc).__name__
        if size != (width, height):   # the new fragments must share the frame the cached VLM boxes were drawn in
            if fresh:
                path.unlink(missing_ok=True)
            return {**res, "status": "image", "note": f"oriented {size} != cached {(width, height)}"}
    res["download_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    frags = await run_kraken(path, model_path, width, height)
    res["kraken_s"] = round(time.time() - t0, 1)
    if fresh and not keep_images:
        path.unlink(missing_ok=True)
    if frags is None:
        return {**res, "status": "kraken"}
    for p in paths:
        store_frags(p, cache_key, frags)
    return {**res, "frags": frags}


def _complete_lines(text: str) -> List[str]:
    """Non-blank, newline-terminated lines of ``text`` (a record still being appended is not one yet).

    :param text: File contents.
    :returns: The lines, without their newlines.
    """
    return [l for l in text[:text.rfind("\n") + 1].splitlines() if l.strip()]


def replace_records(out: Path, lines: List[str], n_read: int) -> int:
    """Atomically replace ``out`` with ``lines`` plus any records appended to it since it was read.

    The main loop only ever appends, so records past the first ``n_read`` were
    added by a run going on meanwhile; they are carried over unchanged (the
    next ``--rekraken`` picks them up).  A record appended in the instant of
    the swap is lost from ``out`` and re-read by that loop's next run, which
    resumes by ``(doc_id, image_index)``.
    :param out: Records file.
    :param lines: Serialised records replacing its first ``n_read``.
    :param n_read: Records it held when read.
    :returns: Number of records carried over.
    :raises RuntimeError: When ``out`` now holds fewer than ``n_read`` records (someone else rewrote it).
    """
    now = _complete_lines(out.read_text()) if out.exists() else []
    if len(now) < n_read:
        raise RuntimeError(f"{out} shrank from {n_read} to {len(now)} records during the pass; left as it is "
                           f"(the new fragments are in the raw cache: re-run to rebuild the records)")
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text("".join(l + "\n" for l in lines + now[n_read:]))
    os.replace(tmp, out)
    return len(now) - n_read


async def rekraken(out: Path, work_dir: Path, raw_dir: Path, model_path: str, htr_model: str, cache_key: str,
                   all_cache: bool = False, force: bool = False, limit: int = 0, min_free_gb: float = 5.0,
                   keep_images: bool = False) -> "Counter[str]":
    """Kraken-only re-run: new fragments for pages already read, records rebuilt without LM Studio.

    Pages are the images behind the records in ``out`` (plus every raw-cache
    entry with ``all_cache``), grouped by image hash so one Kraken read serves
    every VLM checkpoint's entry for an image.  A page whose entries all hold
    ``cache_key`` fragments is not re-read unless ``force`` (so an interrupted
    pass resumes); entries lacking them beside a sibling that has them are
    filled without a read.  Then every record whose entry holds ``cache_key``
    fragments is rebuilt (envelope and VLM side kept, ``htr_model`` stamped)
    and ``out`` is rewritten once; the other records (failed or mismatched
    pages, no cache entry) are kept byte-for-byte.
    :param out: Records file (rewritten; may be absent with ``all_cache``).
    :param work_dir: Kept images, and scratch for downloads.
    :param raw_dir: Raw-cache directory.
    :param model_path: Kraken model.
    :param htr_model: Name stamped as ``ai_read.htr_model``.
    :param cache_key: ``frags_by_htr`` key for the new fragments.
    :param all_cache: Also re-read raw-cache entries that have no record in ``out``.
    :param force: Re-read pages that already hold ``cache_key`` fragments.
    :param limit: Read at most this many pages (0 = no limit).
    :param min_free_gb: Stop reading when free disk under ``raw_dir`` drops below this.
    :param keep_images: Keep downloaded images in ``work_dir``.
    :returns: Counts: pages ``krakened`` (``entries`` written), ``skipped``, entries ``filled``, failures
        ``download`` / ``image`` / ``kraken``, ``sha256`` mismatches, records ``rebuilt`` / ``kept`` / ``carried``.
    """
    counts: Counter = Counter()
    lines = _complete_lines(out.read_text()) if out.exists() else []
    meta: Dict[Path, Dict[str, Any]] = {}          # per raw-cache file: what the re-read needs (entries stay on disk)
    row_paths: List[Optional[Path]] = []           # per record: its usable raw-cache file

    def remember(p: Path, entry: Dict[str, Any]) -> None:
        """Keep the few fields of ``entry`` the pass needs.

        :param p: Raw-cache file.
        :param entry: Its parsed JSON.
        """
        meta[p] = {**{k: entry.get(k) for k in ("doc_id", "image_index", "image_url", "sha256", "width", "height")},
                   "n_frags": len(entry.get("frags") or []), "has_key": cached_frags(entry, cache_key) is not None}

    for line in lines:
        r = json.loads(line)
        p = raw_cache_path(r, r["ai_read"]["vlm_model"], raw_dir=raw_dir)
        entry = json.loads(p.read_text()) if p.exists() else {}
        usable = entry.get("image_url") == r["image_url"] and entry.get("vlm_model") == r["ai_read"]["vlm_model"]
        row_paths.append(p if usable else None)
        if usable and p not in meta:
            remember(p, entry)
    if all_cache:
        for p in sorted(raw_dir.glob("*.json")):
            if p not in meta:
                remember(p, json.loads(p.read_text()))
    pages: Dict[str, List[Path]] = {}
    for p, m in meta.items():
        pages.setdefault(m["sha256"] or str(p), []).append(p)     # no hash: its own page (skipped as unverifiable)
    todo = []
    for paths in pages.values():
        have = [p for p in paths if meta[p]["has_key"]]
        if force or not have:
            todo.append(paths)
            continue
        counts["skipped"] += 1
        missing = [p for p in paths if not meta[p]["has_key"]]
        if missing:                                # same bytes, so a sibling's fragments are this entry's
            frags = cached_frags(json.loads(have[0].read_text()), cache_key)
            for p in missing:
                store_frags(p, cache_key, frags)
                meta[p]["has_key"] = True
            counts["filled"] += len(missing)
    todo = todo[:limit] if limit else todo
    rows_of: Dict[Path, List[int]] = {}
    for i, p in enumerate(row_paths):
        if p is not None:
            rows_of.setdefault(p, []).append(i)
    print(f"{len(lines)} records in {out.name} ({row_paths.count(None)} without a usable cache entry); "
          f"{len(meta)} cache entries = {len(pages)} images; {counts['skipped']} already hold {cache_key} fragments "
          f"({counts['filled']} sibling entries filled); {len(todo)} to read; kraken={Path(model_path).name} "
          f"htr={htr_model} rule={RULE_VERSION}", flush=True)

    rebuilt: Dict[int, str] = {}
    streak = 0
    for n, paths in enumerate(todo, 1):
        free_gb = shutil.disk_usage(raw_dir).free / 1e9
        if free_gb < min_free_gb:
            print(f"STOP: {free_gb:.1f} GB free < {min_free_gb}", flush=True)
            break
        res = await rekraken_page(paths, meta, model_path, cache_key, work_dir, keep_images)
        m = meta[paths[0]]
        tag = f"  {n}/{len(todo)} {m['doc_id']}#{m['image_index']}"
        timing = f"[dl {res['download_s']}s kraken {res['kraken_s']}s]"
        if res["status"] != "ok":
            counts[res["status"]] += 1
            streak = streak + 1 if res["status"] == "kraken" else 0
            print(f"{tag}: {'SKIPPED' if res['status'] == 'sha256' else 'FAILED'} {res['status']} {res['note']} "
                  f"(record kept as is)  {timing}", flush=True)
            if streak >= REKRAKEN_MAX_KRAKEN_FAILURES:
                print(f"STOP: {streak} Kraken failures in a row (service down, or {Path(model_path).name} not in "
                      f"its /app/models mount?)", flush=True)
                break
            continue
        streak = 0
        counts["krakened"] += 1
        counts["entries"] += len(paths)
        change = f" ({len(paths)} cache entries, no record in {out.name})"
        for p in paths:
            meta[p]["has_key"] = True
            for i in rows_of.get(p, []):
                old = json.loads(lines[i])
                new = rebuild_record(old, json.loads(p.read_text()), cache_key, htr_model)
                if new is not None:
                    rebuilt[i] = json.dumps(new, ensure_ascii=False)
                    change = (f" agreed {old['ai_read']['n_agreed']} -> {new['ai_read']['n_agreed']} "
                              f"of {new['ai_read']['n_lines']}")
        note = f" ({res['note']})" if res["note"] else ""
        print(f"{tag}: frags {m['n_frags']} -> {len(res['frags'])}{change}{note}  {timing}", flush=True)

    for i, p in enumerate(row_paths):              # pages read earlier (resume) or filled from a sibling
        if i not in rebuilt and p is not None and meta[p]["has_key"]:
            new = rebuild_record(json.loads(lines[i]), json.loads(p.read_text()), cache_key, htr_model)
            if new is not None:
                rebuilt[i] = json.dumps(new, ensure_ascii=False)
    counts["rebuilt"], counts["kept"] = len(rebuilt), len(lines) - len(rebuilt)
    if out.exists():
        counts["carried"] = replace_records(out, [rebuilt.get(i, l) for i, l in enumerate(lines)], len(lines))
    failed = counts["download"] + counts["image"] + counts["kraken"]
    print(f"rekraken {htr_model} [key {cache_key}]: {counts['krakened']} pages re-krakened ({counts['entries']} "
          f"cache entries), {counts['skipped']} skipped (fragments already cached), {failed} failed (download "
          f"{counts['download']}, image {counts['image']}, kraken {counts['kraken']}), {counts['sha256']} skipped on "
          f"sha256 mismatch; records: {counts['rebuilt']} rebuilt under {RULE_VERSION}, {counts['kept']} kept as is, "
          f"{counts['carried']} appended meanwhile carried over -> {out}", flush=True)
    return counts


def done_keys(out: Path) -> set:
    """``(doc_id, image_index)`` pairs already recorded in ``out``."""
    keys = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                keys.add((r["doc_id"], r["image_index"]))
    return keys


def htr_cache_key_error(args: argparse.Namespace) -> Optional[str]:
    """Why ``--htr-cache-key`` cannot go with the mode given, or None when it can (or is not given).

    The key stamps the normal read path only; the rewrite modes keep their own key handling.
    :param args: Parsed CLI arguments.
    :returns: The error message for ``ap.error``, or None.
    """
    if not getattr(args, "htr_cache_key", None):
        return None
    reasons = {"--rekraken": "names its fragments with --kraken-cache-suffix",
               "--rematch": "matches each record against the fragments of its own key",
               "--restamp": "reads no fragments"}
    given = [f for f, on in (("--rekraken", args.rekraken), ("--rematch", args.rematch),
                             ("--restamp", args.restamp)) if on]
    if not given:
        return None
    return "--htr-cache-key is for normal runs: " + "; ".join(f"{f} {reasons[f]}" for f in given)


async def main_async(args: argparse.Namespace) -> None:
    """Resolve jobs, run them sequentially, append records (or run one of the rewrite modes).

    ``args.htr_cache_key`` (``--htr-cache-key``) files the run's fragments under
    that key (see :func:`stage2`); without it nothing differs from before the
    option existed.  Cache semantics under a key, consistent with ``--rekraken``
    (where fragments cached under another key never count as the key's: those
    pages are re-read): this path reads no fragments back from the raw cache,
    so every job not yet in ``--out`` is Kraken-read afresh and its entry is
    rewritten under the key, even when the entry held legacy
    (``MiDRASH_Gen_01``) fragments; a job already in ``--out`` is skipped
    whatever its key (the resume is by ``(doc_id, image_index)`` only, since
    records are only ever appended), and its record keeps the stamp of the
    reader that made it (a legacy one is neither re-read nor relabelled; to
    re-read it under the key run ``--rekraken --kraken-cache-suffix KEY``).
    :param args: Parsed CLI arguments.
    """
    if args.rekraken:                 # raw cache + Kraken only: no Elasticsearch, no LM Studio
        model_path, htr_model, cache_key = rekraken_names(args.kraken_model, args.htr_model_name,
                                                          args.kraken_cache_suffix)
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        await rekraken(Path(args.out), work_dir, Path(args.raw_dir) if args.raw_dir else RAW_DIR, model_path,
                       htr_model, cache_key, all_cache=args.all_cache, force=args.force, limit=args.limit,
                       min_free_gb=args.min_free_gb, keep_images=args.keep_images)
        return
    load_site_env(Path(args.site_env))
    source_index = latest_merged_index() if args.source_index in (None, "latest") else args.source_index
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    failures = out.with_suffix(".failures.jsonl")

    if args.restamp:
        restamp(out, source_index)
        return
    if args.rematch:
        await rematch(out, work_dir, source_index, args.vlm_revision)
        return
    jobs = build_jobs(args, source_index)
    done = done_keys(out)
    todo = [j for j in jobs if (j["doc_id"], j["image_index"]) not in done]
    if args.limit:
        todo = todo[:args.limit]
    htr_cache_key = getattr(args, "htr_cache_key", None)
    key_note = f" htr_cache_key={htr_cache_key}" if htr_cache_key else ""
    print(f"{len(jobs)} jobs resolved against {source_index}; {len(done)} done; {len(todo)} to run; "
          f"vlm={args.vlm_model} rule={RULE_VERSION}{key_note}", flush=True)
    def log_failure(info: Dict[str, Any]) -> None:
        with failures.open("a") as fh:
            fh.write(json.dumps({k: v for k, v in info.items() if k not in ("path", "frags")}, ensure_ascii=False) + "\n")

    pending: Optional[asyncio.Task] = asyncio.create_task(stage1(todo[0], work_dir)) if todo else None
    for n, job in enumerate(todo, 1):
        free_gb = shutil.disk_usage(out.parent).free / 1e9
        if free_gb < args.min_free_gb:
            print(f"STOP: {free_gb:.1f} GB free < {args.min_free_gb}", flush=True)
            if pending:
                pending.cancel()
            break
        info = await pending
        pending = asyncio.create_task(stage1(todo[n], work_dir)) if n < len(todo) else None
        record = None
        if "failure" not in info:
            record = await stage2(job, info, source_index, args.vlm_model, args.vlm_revision,
                                  htr_cache_key=htr_cache_key)
        if not args.keep_images and info.get("path"):
            Path(info["path"]).unlink(missing_ok=True)
        if record is None:
            log_failure(info)
            print(f"  {n}/{len(todo)} {job['doc_id']}#{job['image_index']}: FAILED {info.get('failure')}", flush=True)
            continue
        with out.open("a") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  {n}/{len(todo)} {job['doc_id']}#{job['image_index']}: lines {info['n_lines']} agreed {info['n_agreed']} "
              f"frags {info['n_frag']} parsed={info['parsed']}  "
              f"[dl {info['download_s']}s kraken {info['kraken_s']}s vlm {info['vlm_s']}s]", flush=True)
    print(f"done {datetime.now().isoformat(timespec='seconds')} -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-consensus", choices=("high", "standard", "escalate"), default=None,
                    help="jobs = consensus pilot rows of this tier (doc_id + the image the page gate read)")
    ap.add_argument("--ids", default=None, help="JSONL of {doc_id, image_index|image_url}")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_DIR / f"ai_reads_{VLM_MODEL}.jsonl"))
    ap.add_argument("--work-dir", default=str(OUT_DIR / "images"))
    ap.add_argument("--source-index", default="latest",
                    help="reference merged index; 'latest' = highest genizah_merged_v<N> (the successor index)")
    ap.add_argument("--rematch", action="store_true",
                    help="rebuild the records in --out under the current rule from the raw cache (Kraken re-run "
                         "for records that predate the cache; no LM Studio), then exit")
    ap.add_argument("--restamp", action="store_true",
                    help="only re-resolve the records already in --out against --source-index, then exit")
    ap.add_argument("--rekraken", action="store_true",
                    help="Kraken-only re-run: re-read the pages behind the records in --out (every raw-cache entry "
                         "with --all-cache) with --kraken-model, cache the fragments under frags_by_htr[KEY] beside "
                         "the original reader's 'frags', rebuild the records from the cached VLM lines and rewrite "
                         "--out (no LM Studio, no ES); resumable (pages holding KEY fragments are not re-read), "
                         "then exit")
    ap.add_argument("--kraken-model", default=None,
                    help=f"--rekraken: Kraken model path (default {Path(KRAKEN_MODEL).name}; the service finds it "
                         f"by basename in its /app/models mount)")
    ap.add_argument("--htr-model-name", default=None,
                    help=f"--rekraken: name stamped as ai_read.htr_model (default the --kraken-model file stem, "
                         f"i.e. {HTR_MODEL_NAME} for the default model)")
    ap.add_argument("--kraken-cache-suffix", default=None, metavar="KEY",
                    help="--rekraken: frags_by_htr key for the new fragments (default the HTR model name); give a "
                         "distinct one to re-read with a same-named model (e.g. under a new segmenter) without "
                         "replacing its earlier fragments")
    ap.add_argument("--all-cache", action="store_true",
                    help="--rekraken: every raw-cache entry, not only those behind --out (one read per image "
                         "hash; records are rebuilt in --out only)")
    ap.add_argument("--force", action="store_true", help="--rekraken: re-read pages that already hold KEY fragments")
    ap.add_argument("--raw-dir", default=None, help="--rekraken: raw-cache directory (default ai_reads/raw)")
    ap.add_argument("--site-env", default=str(SITE_ENV), help="website .env for ES credentials")
    ap.add_argument("--vlm-model", default=VLM_MODEL)
    ap.add_argument("--vlm-revision", default=VLM_REVISION)
    ap.add_argument("--htr-cache-key", default=None, metavar="KEY",
                    help=f"normal runs: file this run's Kraken fragments under KEY (raw-cache htr_model = KEY, "
                         f"records' ai_read.htr_cache_key = KEY, ai_read.htr_model stays {HTR_MODEL_NAME}), the "
                         f"layout stamp_htr_cache_key.py gives earlier reads; for a Kraken service swap (kraken "
                         f"version, segmenter) that keeps the recognition model, e.g. "
                         f"{HTR_MODEL_NAME}@k7.0.3-blla2026. Jobs already in --out are skipped whatever their key")
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    ap.add_argument("--no-resolve", action="store_true",
                    help="trust image_index/image_url (and optional local_path) in --ids as given; skip the index lookup")
    ap.add_argument("--keep-images", action="store_true")
    a = ap.parse_args()
    if not (a.from_consensus or a.ids or a.restamp or a.rematch or a.rekraken):
        ap.error("give --from-consensus TIER, --ids FILE, --restamp, --rematch, or --rekraken")
    rekraken_only = [f for f, v in (("--kraken-model", a.kraken_model), ("--htr-model-name", a.htr_model_name),
                                    ("--kraken-cache-suffix", a.kraken_cache_suffix), ("--all-cache", a.all_cache),
                                    ("--force", a.force), ("--raw-dir", a.raw_dir)) if v]
    if rekraken_only and not a.rekraken:
        ap.error(f"{', '.join(rekraken_only)}: --rekraken options only")
    key_error = htr_cache_key_error(a)
    if key_error:
        ap.error(key_error)
    if a.rekraken and (a.from_consensus or a.ids or a.restamp or a.rematch):
        ap.error("--rekraken re-reads the records in --out (or the raw cache): no jobs, no other mode")
    if a.rekraken and not (a.all_cache or Path(a.out).exists()):
        ap.error(f"--rekraken needs an existing --out ({a.out}) or --all-cache")
    asyncio.run(main_async(a))
