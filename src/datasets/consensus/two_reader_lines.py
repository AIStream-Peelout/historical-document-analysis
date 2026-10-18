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
``frags`` and the records are rebuilt from the cached VLM lines (``<key>`` is
``--kraken-cache-suffix``, naming the Kraken service; the bare model name only
with ``--legacy-key``).  A Kraken
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

``--decode-policy`` reads each page the way the benchmark decode policy does
(:mod:`src.datasets.consensus.pipeline_decode_policy`): a page Kraken splits
into columns (with a central ink gap) is read as column crops, rightmost first;
a single-column page whose whole-page read fails health is re-read as two
halves; every other page is read whole, as without the flag.  The merged lines
go through the same :func:`sidecar`; the raw cache keeps every reply
(``vlm_raw`` is a list when a page took more than one request) and the policy's
trace (``decode_policy``), and the record's ``ai_read`` gains a short
``decode_policy`` stamp (the site loader ignores unknown ``ai_read`` keys, as it
does ``htr_cache_key``).  ``--revisit`` applies the policy to pages already in
``--out`` from their cached Kraken fragments and whole-page read: only the pages
it would change are downloaded and re-read, and their records are rewritten in
place; ``--revisit --dry-run`` plans every page without a request or a write.

Usage (repo root; long runs under nohup):
    .venv/bin/python -m src.datasets.consensus.two_reader_lines --from-consensus high
    .venv/bin/python -m src.datasets.consensus.two_reader_lines --ids jobs.jsonl --limit 50
    .venv/bin/python -m src.datasets.consensus.two_reader_lines --ids jobs.jsonl --htr-cache-key MiDRASH_Gen_01@k7.0.3-blla2026
    .venv/bin/python -m src.datasets.consensus.two_reader_lines --rekraken --kraken-model NEW.mlmodel --kraken-cache-suffix NEW@k7.0.3-blla2026
    .venv/bin/python -m src.datasets.consensus.two_reader_lines --ids jobs.jsonl --decode-policy --htr-cache-key KEY
    .venv/bin/python -m src.datasets.consensus.two_reader_lines --revisit --dry-run --out ai_reads_MODEL.jsonl --vlm-model MODEL [--ids jobs.jsonl]
    .venv/bin/python -m src.datasets.consensus.two_reader_lines --revisit --decode-policy --out ai_reads_MODEL.jsonl --vlm-model MODEL [--ids jobs.jsonl]
"""
import argparse
import asyncio
import base64
import dataclasses
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
from src.datasets.consensus import decode_policy as dp  # noqa: E402
from src.datasets.consensus import pipeline_decode_policy as pdp  # noqa: E402
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
REVISIT_MAX_VLM_FAILURES = 5         # --revisit stops after this many LM Studio failures in a row (server down / model gone)
REVISIT_FLUSH_EVERY = 25             # --revisit rewrites --out after this many re-read pages (the loader sees progress)
REVISIT_REPORT_DIR = _REPO / "logs/decode_policy"   # --revisit --dry-run reports
SECONDS_PER_CALL = 45.0              # dry-run cost estimate per model request (documentary page reads take 30-60 s)
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
# one line entry of a grounded reply in the prompt's key order. The text runs to the `", "bbox_2d": [` delimiter, so
# a plain `"` inside it (Hebrew abbreviations: ס"ז, כ"ז) does not end it, and it may not run into the next entry.
_ENTRY_RE = re.compile(r'\{\s*"text"\s*:\s*"((?:(?!"\s*\}\s*,?\s*\{\s*"text"\s*:).)*?)"\s*,\s*"bbox_2d"\s*:\s*'
                       r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\s*\}',
                       re.S)


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


def _entry_text(field: str) -> str:
    """Decode the text field of a line entry matched by :data:`_ENTRY_RE`.

    :param field: The characters between the field's quotes, still JSON-escaped; may hold a plain ``"``.
    :type field: str
    :return: The text. A field that is valid JSON string content is decoded as such; otherwise only the
        escapes ``\\"``, ``\\\\`` and ``\\n`` are undone.
    :rtype: str
    """
    try:
        return json.loads(f'"{field}"')
    except json.JSONDecodeError:
        return field.replace('\\"', '"').replace("\\n", " ").replace("\\\\", "\\")


def entry_objects(text: str) -> List[Dict[str, Any]]:
    """Line entries of a grounded reply found by their shape instead of by JSON parsing.

    Used when the reply is not valid JSON: a plain ``"`` inside a text (Hebrew abbreviations) unbalances the
    quotes, and a parser that tracks strings then drops every entry up to the next stray quote.

    :param text: Raw reply.
    :type text: str
    :return: ``{"text", "bbox_2d"}`` objects of the complete entries, in reply order.
    :rtype: List[Dict[str, Any]]
    """
    return [{"text": _entry_text(m.group(1)), "bbox_2d": [float(v) for v in m.groups()[1:]]} for m in _ENTRY_RE.finditer(text)]


def parse_grounded(raw: Optional[str]) -> Tuple[bool, List[Dict[str, Any]]]:
    """Parse the VLM's JSON-array reply, tolerating truncation, trailing prose and plain quotes inside a text.

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
    by_shape = entry_objects(text) if not objs else []
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
    if len(by_shape) >= len(objs):                   # unbalanced quotes hide entries from the scan above
        objs = by_shape
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
                 vlm_revision: str, htr_cache_key: Optional[str] = None,
                 policy: Optional[dp.PolicyThresholds] = None, keep_images: bool = False,
                 rescue_min_kraken_letters: int = 0) -> Optional[Dict[str, Any]]:
    """VLM-read the prepared image, match, and build the loader record (GPU side).

    With ``htr_cache_key`` the fragments are filed under that key exactly as
    ``stamp_htr_cache_key.py`` files an earlier read after the fact: the raw-cache
    entry keeps them as its top-level ``frags`` and its ``htr_model`` names the key
    (so :func:`cached_frags` answers them for the key and no longer for
    :data:`HTR_MODEL_NAME`), and the record's ``ai_read`` gets ``htr_cache_key``
    after ``lines`` while ``ai_read.htr_model`` stays the recognition model's name,
    as :func:`rebuild_record` stamps a ``--rekraken --kraken-cache-suffix`` record
    (so ``--rematch`` and a same-key ``--rekraken`` find the fragments again).
    With ``policy`` the page is read under the decode policy (:func:`stage2_policy`).
    :param job: The job.
    :param info: Output of :func:`stage1` (no ``failure``).
    :param source_index: Reference merged index stamped into the record.
    :param vlm_model: LM Studio key.
    :param vlm_revision: Checkpoint revision string.
    :param htr_cache_key: HTR cache key of ``info['frags']`` (``--htr-cache-key``); None writes exactly what this
        function wrote before the option existed, as does :data:`HTR_MODEL_NAME` (the default key).
    :param policy: Decode-policy thresholds (``--decode-policy``); None reads the page whole, exactly as before the
        option existed.
    :param keep_images: With ``policy``: keep the crop JPEGs beside the page image (``--keep-images``).
    :param rescue_min_kraken_letters: With ``policy``: never rescue a page whose Kraken fragments hold fewer letters
        (``--rescue-min-kraken-letters``; 0 = the validated policy).
    :returns: The record, or None (``info['failure']`` set) when LM Studio failed.
    """
    if policy is not None:
        return await stage2_policy(job, info, source_index, vlm_model, vlm_revision, policy, htr_cache_key,
                                   keep_images, rescue_min_kraken_letters)
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


# ---------------------------------------------------------------------------
# Decode policy (--decode-policy, --revisit)
# ---------------------------------------------------------------------------

class PolicyPageReader:
    """One prepared page image read for the decode policy: the page itself, or crops of it, with :data:`P_GROUNDED`.

    Crops are written beside the page image as ``<page stem>__<unit>.jpg`` (JPEG quality
    :data:`~src.datasets.consensus.pipeline_decode_policy.CROP_JPEG_QUALITY`) and removed by :meth:`remove_crops`.
    Requests go out one at a time, like every other LM Studio request of the pipeline.
    """

    def __init__(self, page: Path, width: int, height: int, vlm_model: str) -> None:
        """:param page: The oriented page JPEG (:func:`prepare_image`).
        :param width: Its width in pixels.
        :param height: Its height in pixels.
        :param vlm_model: LM Studio key.
        """
        self.page, self.width, self.height, self.vlm_model = page, width, height, vlm_model
        self.crops: List[Path] = []
        self._image: Optional[Image.Image] = None

    def image(self) -> Image.Image:
        """The page, decoded once on first use.

        :returns: RGB image (the oriented page, so its frame is the fragments' and the whole-page boxes').
        """
        if self._image is None:
            with Image.open(self.page) as im:
                self._image = im.convert("RGB")
        return self._image

    def ink_gap(self) -> Optional[int]:
        """The page's central vertical ink gap (:func:`~src.datasets.consensus.decode_policy.central_ink_gap`).

        :returns: Gap x in pixels, or None when the page shows none.
        """
        return dp.central_ink_gap(self.image())

    async def read(self, unit: str, role: str,
                   crop: Optional[Tuple[int, int, int, int]]) -> Optional[pdp.Reply]:
        """Read the page (``crop`` None) or one crop of it with the grounded prompt.

        :param unit: Unit name (also names the crop file).
        :param role: Unit role.
        :param crop: Pixel crop box, or None for the whole page.
        :returns: The reply with its lines in the page's 0-1000 frame, or None when LM Studio failed.
        """
        path = self.page
        if crop is not None:
            path = self.page.with_name(f"{self.page.stem}__{unit}.jpg")
            self.image().crop(crop).save(path, format="JPEG", quality=pdp.CROP_JPEG_QUALITY)
            self.crops.append(path)
        t0 = time.time()
        raw = await transcribe_with_lm_studio(self.vlm_model, str(path), P_GROUNDED, max_tokens=VLM_MAX_TOKENS)
        if raw is None:
            return None
        parsed, lines = parse_grounded(raw)
        if crop is not None:
            lines = pdp.crop_lines_to_page(lines, crop, self.width, self.height)
        return pdp.Reply(unit, role, crop, raw, parsed, lines, round(time.time() - t0, 1))

    def remove_crops(self) -> None:
        """Delete the crop JPEGs written so far."""
        for path in self.crops:
            path.unlink(missing_ok=True)
        self.crops.clear()


async def stage2_policy(job: Dict[str, Any], info: Dict[str, Any], source_index: str, vlm_model: str,
                        vlm_revision: str, th: dp.PolicyThresholds, htr_cache_key: Optional[str] = None,
                        keep_images: bool = False, rescue_min_kraken_letters: int = 0) -> Optional[Dict[str, Any]]:
    """:func:`stage2` under the decode policy: read the page as its plan says, match the merged lines, build the record.

    The raw-cache entry has the fields :func:`stage2` writes, with ``vlm_lines`` / ``parsed`` the merged page's and
    ``vlm_raw`` the reply (a list of every reply, in the order of ``decode_policy.requests``, when the page took
    more than one request), plus ``decode_policy``: the policy's trace. The record is :func:`stage2`'s, its
    ``ai_read`` ending with the short stamp :func:`~src.datasets.consensus.pipeline_decode_policy.policy_stamp`.
    :param job: The job.
    :param info: Output of :func:`stage1` (no ``failure``); gains ``policy`` (the action, ``keep(blank_kraken_<n>)``
        for a rescue skipped on a Kraken-blank page), ``calls`` and ``policy_fallback``.
    :param source_index: Reference merged index stamped into the record.
    :param vlm_model: LM Studio key.
    :param vlm_revision: Checkpoint revision string.
    :param th: Decode-policy thresholds.
    :param htr_cache_key: As for :func:`stage2`.
    :param keep_images: Keep the crop JPEGs.
    :param rescue_min_kraken_letters: Never rescue a page whose Kraken fragments hold fewer letters (0 = validated).
    :returns: The record, or None (``info['failure']`` = ``vlm``) when a request failed.
    """
    reader = PolicyPageReader(Path(info["path"]), info["width"], info["height"], vlm_model)
    t0 = time.time()
    result = await pdp.decode_page(info["frags"], info["width"], info["height"], th, reader.read, reader.ink_gap,
                                   rescue_min_kraken_letters=rescue_min_kraken_letters)
    info["vlm_s"] = round(time.time() - t0, 1)
    if not keep_images:
        reader.remove_crops()
    if result is None:
        info["failure"] = "vlm"
        return None
    trace = {**result.trace, "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_cache_path(job, vlm_model, for_write=True).write_text(json.dumps(dict(
        doc_id=job["doc_id"], image_index=job["image_index"], image_url=job["image_url"],
        width=info["width"], height=info["height"], sha256=info["sha256"], vlm_model=vlm_model,
        vlm_raw=result.vlm_raw, vlm_lines=result.lines, parsed=result.parsed, frags=info["frags"],
        htr_model=htr_cache_key or HTR_MODEL_NAME, decode_policy=trace), ensure_ascii=False))
    ai_read = sidecar(result.lines, info["frags"], result.parsed, vlm_model, vlm_revision)
    if htr_cache_key and htr_cache_key != ai_read["htr_model"]:
        ai_read["htr_cache_key"] = htr_cache_key
    ai_read["decode_policy"] = pdp.policy_stamp(trace)
    info.update(n_frag=len(info["frags"]), n_lines=ai_read["n_lines"], n_agreed=ai_read["n_agreed"],
                parsed=result.parsed, calls=result.calls,
                policy=f"keep({result.plan.reason})" if pdp.is_blank_skip(result.plan) else result.plan.action,
                policy_fallback=trace["page_fallback"])
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
        if c.get("decode_policy"):                    # a page read under --decode-policy keeps its stamp
            ai_read["decode_policy"] = pdp.policy_stamp(c["decode_policy"])
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
    :param cache_suffix: ``--kraken-cache-suffix`` (None = the HTR model name, which the CLI allows only with
        ``--legacy-key``: see :func:`rekraken_key_error`).
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
    current rule.  An entry read under the decode policy gives the record its
    ``ai_read.decode_policy`` stamp.
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
    if entry.get("decode_policy"):
        ai_read["decode_policy"] = pdp.policy_stamp(entry["decode_policy"])
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


# ---------------------------------------------------------------------------
# Decode policy over recorded pages (--revisit)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class RevisitSelection:
    """The pages of an ids file, for ``--revisit --ids`` (matched locally: no index lookup).

    :param path: The ids file.
    :param keys: ``(doc_id, image_index)`` of its jobs that name an image index (how :func:`done_keys` resumes).
    :param url_keys: ``(doc_id, image_url)`` of its jobs that name only a URL.
    :param n_jobs: Jobs in the file.
    """

    path: Path
    keys: frozenset
    url_keys: frozenset
    n_jobs: int

    def selects(self, record: Dict[str, Any]) -> bool:
        """Whether a record is one of the file's pages.

        :param record: Record from ``--out``.
        :returns: True when its ``(doc_id, image_index)`` or ``(doc_id, image_url)`` is listed.
        """
        return ((record["doc_id"], record["image_index"]) in self.keys
                or (record["doc_id"], record["image_url"]) in self.url_keys)


def revisit_selection(ids: Optional[Path]) -> Optional[RevisitSelection]:
    """Read ``--ids`` for ``--revisit``.

    :param ids: JSONL of ``{doc_id, image_index|image_url}`` (None: every record in ``--out``).
    :returns: The selection, or None without a file.
    """
    if ids is None:
        return None
    keys, url_keys, n = set(), set(), 0
    for line in ids.read_text().splitlines():
        if not line.strip():
            continue
        job = json.loads(line)
        n += 1
        if "image_index" in job:
            keys.add((job["doc_id"], int(job["image_index"])))
        elif job.get("image_url"):
            url_keys.add((job["doc_id"], job["image_url"]))
    return RevisitSelection(ids, frozenset(keys), frozenset(url_keys), n)


def record_htr(record: Dict[str, Any]) -> Tuple[str, str]:
    """The HTR model name and cache key a record was matched under (as :func:`rematch` reads them).

    :param record: Record from ``--out``.
    :returns: ``(htr_model, cache_key)``.
    """
    htr_model = record["ai_read"].get("htr_model") or HTR_MODEL_NAME
    return htr_model, record["ai_read"].get("htr_cache_key") or htr_model


def revisit_entry(record: Dict[str, Any], raw_dir: Path,
                  vlm_model: str) -> Tuple[str, Path, Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    """A record's raw-cache entry, if the decode policy can be planned from it.

    :param record: Record from ``--out``.
    :param raw_dir: Raw-cache directory.
    :param vlm_model: ``--vlm-model``: only its records are revisited (the crops are read with the same model).
    :returns: ``(status, cache path, entry, fragments)``. Status ``ok`` (fragments of the record's HTR key and a
        whole-page reply to judge), ``done`` (the entry already carries a ``decode_policy`` trace), or why it cannot
        be planned: ``other_model``, ``no_cache``, ``cache_mismatch`` (another image or model), ``no_whole_read``
        (no reply text, e.g. an entry ``--rematch`` built from a record), ``no_frags``.
    """
    path = raw_cache_path(record, vlm_model, raw_dir=raw_dir)
    if record["ai_read"]["vlm_model"] != vlm_model:
        return "other_model", path, None, None
    if not path.exists():
        return "no_cache", path, None, None
    entry = json.loads(path.read_text())
    if entry.get("image_url") != record["image_url"] or entry.get("vlm_model") != vlm_model:
        return "cache_mismatch", path, None, None
    if entry.get("decode_policy"):
        return "done", path, entry, None
    if not isinstance(entry.get("vlm_raw"), str):
        return "no_whole_read", path, entry, None
    frags = cached_frags(entry, record_htr(record)[1])
    return ("no_frags" if frags is None else "ok"), path, entry, frags


def fetch_cached_image(doc_id: str, image_index: int, entry: Dict[str, Any],
                       work_dir: Path) -> Tuple[Optional[Path], str, str, bool]:
    """The image a raw-cache entry was read from, prepared in ``work_dir``; only the exact bytes the readers saw.

    A kept ``work_dir`` copy hashing to the cached ``sha256`` is used as is; otherwise the image is downloaded and
    must hash to it and orient to the cached size (the cached boxes were drawn in that frame), as
    :func:`rekraken_page` requires.
    :param doc_id: Document id.
    :param image_index: Image position.
    :param entry: Raw-cache entry (``image_url``, ``sha256``, ``width``, ``height``).
    :param work_dir: Where kept images are looked up and the download is prepared.
    :returns: ``(path, status, note, fresh)``: status ``ok`` with the path, else ``download``, ``sha256`` or
        ``image`` with no path; ``fresh`` is True when the file was written now (the caller deletes it).
    """
    path = work_dir / f"{_image_stem(doc_id, image_index)}.jpg"
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]:
        return path, "ok", "kept image", False
    data = download(entry["image_url"])
    if data is None:
        return None, "download", "", False
    sha = hashlib.sha256(data).hexdigest()
    if sha != entry["sha256"]:
        return None, "sha256", f"served {sha[:12]} != cached {(entry['sha256'] or '-')[:12]}", False
    fresh = not path.exists()
    try:
        size: Any = prepare_image(data, path)
    except Exception as exc:  # noqa: BLE001 — corrupt/unsupported image is a per-page failure
        size = type(exc).__name__
    if size != (entry["width"], entry["height"]):
        if fresh:
            path.unlink(missing_ok=True)
        return None, "image", f"oriented {size} != cached {(entry['width'], entry['height'])}", False
    return path, "ok", "", fresh


def revisit_report_path(out: Path, ids: Optional[Path], rescue_min_kraken_letters: int = 0) -> Path:
    """Default report path of ``--revisit --dry-run``.

    :param out: Records file.
    :param ids: ``--ids`` file, or None.
    :param rescue_min_kraken_letters: ``--rescue-min-kraken-letters`` (0: not set).
    :returns: ``logs/decode_policy/revisit_<out stem>[__<ids stem>][__rescue_min<N>].json``.
    """
    suffix = (f"__{ids.stem}" if ids else "") + (f"__rescue_min{rescue_min_kraken_letters}"
                                                 if rescue_min_kraken_letters else "")
    return REVISIT_REPORT_DIR / f"revisit_{out.stem}{suffix}.json"


def revisit_report(rows: List[Dict[str, Any]], counts: Counter, out: Path, n_records: int,
                   selection: Optional[RevisitSelection], recorded: set, vlm_model: str, th: dp.PolicyThresholds,
                   seconds_per_call: float, rescue_min_kraken_letters: int = 0) -> Dict[str, Any]:
    """The ``--revisit --dry-run`` report.

    :param rows: One :func:`~src.datasets.consensus.pipeline_decode_policy.plan_row` per planned page (with
        ``doc_id`` / ``image_index``), planned from Kraken's split alone.
    :param counts: Selected records by raw-cache status (:func:`revisit_entry`), plus ``not_selected``.
    :param out: Records file.
    :param n_records: Records in it.
    :param selection: ``--ids`` selection, or None.
    :param recorded: ``(doc_id, image_index)`` of every record in ``out``.
    :param vlm_model: ``--vlm-model``.
    :param th: The thresholds a live revisit would use (the dry run plans with ``require_ink_gap`` off).
    :param seconds_per_call: Seconds per model request for the time estimates.
    :param rescue_min_kraken_letters: ``--rescue-min-kraken-letters`` the pages were planned with.
    :returns: JSON-ready report; ``pages`` lists the pages a revisit would re-read. ``actions`` counts the pages kept
        only because their Kraken fragments were too blank to rescue (``skipped_blank``) apart from ``keep``.
    """
    actions = Counter(r["action"] for r in rows)
    blank = [r for r in rows if r["action"] == dp.ACTION_KEEP and r["reason"].startswith(pdp.BLANK_KRAKEN_REASON)]
    labels = Counter(r["whole_page_label"] for r in rows)
    base, worst = (sum(r["calls_revisit"][k] for r in rows) for k in (0, 1))
    candidates = [r for r in rows if r["action"] != dp.ACTION_KEEP]
    report: Dict[str, Any] = {
        "out": str(out), "ids": str(selection.path) if selection else None, "vlm_model": vlm_model,
        "policy_version": pdp.POLICY_VERSION, "thresholds": th.to_dict(),
        "column_detection": ("Kraken's split alone: the images are not on disk, so the dry run has no ink-gap test. "
                             + ("A live --revisit downloads each candidate page and keeps a split only when the page "
                                "shows a central vertical ink gap (religious-140: 25 true and 1 false Kraken splits "
                                "-> 24 and 0 with the gap); a page the gate refuses is planned as one column: rescue "
                                "if its whole-page read fails health, else keep (no request)."
                                if th.require_ink_gap else "The live run is set to the same (--no-require-ink-gap).")),
        "rescue_min_kraken_letters": rescue_min_kraken_letters,
        "records": n_records, "status": dict(counts), "planned": len(rows),
        "actions": {dp.ACTION_COLUMNS: actions.get(dp.ACTION_COLUMNS, 0),
                    dp.ACTION_RESCUE: actions.get(dp.ACTION_RESCUE, 0), "skipped_blank": len(blank),
                    dp.ACTION_KEEP: actions.get(dp.ACTION_KEEP, 0) - len(blank)},
        "skipped_blank_labels": dict(Counter(r["whole_page_label"] for r in blank).most_common()),
        "reasons": dict(Counter(r["reason"] for r in rows).most_common()),
        "kraken_columns": dict(sorted(Counter(r["kraken_columns"] for r in rows).items())),
        "whole_page_labels": dict(labels.most_common()),
        "whole_page_flags": dict(Counter(f for r in rows for f in r["whole_page_flags"]).most_common()),
        "whole_page_labels_on_column_pages": dict(Counter(
            r["whole_page_label"] for r in rows if r["action"] == dp.ACTION_COLUMNS).most_common()),
        "rescue_labels": dict(Counter(
            r["whole_page_label"] for r in rows if r["action"] == dp.ACTION_RESCUE).most_common()),
        # Kraken read < min_kraken_letters: the halves cover the whole page and, without a Kraken budget or a healthy
        # whole-page read, a half read of more than overgen_min_letters fails as over-generated.
        "rescue_kraken_blank": dict(Counter(
            r["whole_page_label"] for r in rows
            if r["action"] == dp.ACTION_RESCUE and r["reason"].endswith("_kraken_blank")).most_common()),
        "revisit_cost": {"downloads": len(candidates), "calls_base": base, "calls_worst": worst,
                         "hours_base": round(base * seconds_per_call / 3600, 1),
                         "hours_worst": round(worst * seconds_per_call / 3600, 1),
                         "seconds_per_call": seconds_per_call,
                         "note": "base = one request per planned crop (a column page: one per column; a rescue: "
                                 "two halves); the worst case adds two half reads for every column whose crop read "
                                 "fails. The whole-page read is cached, so a revisit never repeats it."},
    }
    if selection is not None:
        pending = len(selection.keys - recorded)
        live = [sum(r["calls_live"][k] for r in rows) / len(rows) if rows else 1.0 for k in (0, 1)]
        report["jobs"] = {
            "file": str(selection.path), "jobs": selection.n_jobs, "pages": len(selection.keys),
            "pages_named_by_url_only": len(selection.url_keys), "recorded": len(selection.keys & recorded),
            "pending": pending, "calls_per_page_base": round(live[0], 3), "calls_per_page_worst": round(live[1], 3),
            "pending_calls_without_policy": pending, "pending_calls_base": round(pending * live[0]),
            "pending_calls_worst": round(pending * live[1]),
            "pending_extra_hours_base": round(pending * (live[0] - 1) * seconds_per_call / 3600, 1),
            "pending_extra_hours_worst": round(pending * (live[1] - 1) * seconds_per_call / 3600, 1),
            "note": "pending = pages of the ids file with no record in --out, counted before the live run resolves "
                    "them against the index (it drops unresolvable ones); their per-page cost uses this file's plan "
                    "rates (keep 1 request, rescue 3, a k-column page k to 3k+1)"}
    report["pages"] = candidates
    return report


def format_revisit_summary(report: Dict[str, Any]) -> str:
    """A few lines of a ``--revisit --dry-run`` report for the console.

    :param report: Output of :func:`revisit_report`.
    :returns: Text.
    """
    a, cost = report["actions"], report["revisit_cost"]
    lines = [f"revisit dry run over {Path(report['out']).name} ({report['vlm_model']}): {report['records']} records; "
             f"status {report['status']}",
             f"  plan of {report['planned']} pages (Kraken's split alone, no ink-gap test): columns "
             f"{a[dp.ACTION_COLUMNS]}, rescue {a[dp.ACTION_RESCUE]}, skipped as blank (Kraken < "
             f"{report['rescue_min_kraken_letters']} letters) {a['skipped_blank']} {report['skipped_blank_labels']}, "
             f"keep {a[dp.ACTION_KEEP]}",
             f"  Kraken columns per page: {report['kraken_columns']}",
             f"  whole-page labels: {report['whole_page_labels']}",
             f"  ... on column pages: {report['whole_page_labels_on_column_pages']}; failed reads rescued: "
             f"{report['rescue_labels']} (Kraken-blank pages among them: {report['rescue_kraken_blank']})",
             f"  revisit cost: {cost['downloads']} downloads, {cost['calls_base']} model calls (worst case "
             f"{cost['calls_worst']}), {cost['hours_base']} h ({cost['hours_worst']} h) at {cost['seconds_per_call']} s"]
    if "jobs" in report:
        j = report["jobs"]
        lines.append(f"  jobs {Path(j['file']).name}: {j['jobs']} jobs / {j['pages']} pages, {j['recorded']} recorded, "
                     f"{j['pending']} pending -> {j['pending_calls_base']} calls under the policy (worst "
                     f"{j['pending_calls_worst']}) instead of {j['pending_calls_without_policy']}: +"
                     f"{j['pending_extra_hours_base']} h ({j['pending_extra_hours_worst']} h)")
    return "\n".join(lines)


async def revisit(out: Path, work_dir: Path, raw_dir: Path, vlm_model: str, th: dp.PolicyThresholds,
                  ids: Optional[Path] = None, dry_run: bool = False, report: Optional[Path] = None, limit: int = 0,
                  min_free_gb: float = 5.0, keep_images: bool = False, flush_every: int = REVISIT_FLUSH_EVERY,
                  seconds_per_call: float = SECONDS_PER_CALL, rescue_min_kraken_letters: int = 0) -> Dict[str, Any]:
    """Apply the decode policy to pages already in ``out``: re-read only the pages it would change.

    Each selected record is planned from its raw-cache entry, without a request: the Kraken fragments it was matched
    against and its whole-page reply. Pages planned ``columns`` (Kraken's split; the ink gap needs the image) or
    ``rescue`` are downloaded (the exact bytes the readers saw, :func:`fetch_cached_image`) and decoded with the
    cached whole-page read standing in for a new one (:func:`~src.datasets.consensus.pipeline_decode_policy.decode_page`;
    with ``th.require_ink_gap`` a split without a central ink gap becomes one column again, kept unless its read
    failed). The entry is then rewritten (``vlm_raw`` = [cached reply, new replies...], ``vlm_lines`` / ``parsed`` of
    the merged page, the trace under ``decode_policy`` with the cached lines) and the record rebuilt by
    :func:`rebuild_record` under its own HTR key, envelope and revision. ``out`` is rewritten every ``flush_every``
    pages and at the end (:func:`replace_records`); every other record stays byte for byte. Resumable: an entry
    holding a trace is not re-read, and its record is rebuilt if a pass died before writing it.
    :param out: Records file.
    :param work_dir: Kept images, and scratch for downloads and crops.
    :param raw_dir: Raw-cache directory.
    :param vlm_model: LM Studio key; only its records are revisited.
    :param th: Decode-policy thresholds of the live revisit.
    :param ids: Restrict to the pages of this jobs file (None: every record).
    :param dry_run: Plan only: no request, no download, no write except the report.
    :param report: Dry-run report path (default :func:`revisit_report_path`).
    :param limit: Re-read at most this many pages (0 = no limit).
    :param min_free_gb: Stop when free disk under ``raw_dir`` drops below this.
    :param keep_images: Keep downloads and crops in ``work_dir``.
    :param flush_every: Rewrite ``out`` after this many re-read pages (0 = at the end only).
    :param seconds_per_call: Seconds per request for the dry run's estimates.
    :param rescue_min_kraken_letters: Never rescue a page whose Kraken fragments hold fewer letters: it is kept as it
        is, not downloaded (0 = the validated policy).
    :returns: The dry-run report, or the run's counts.
    """
    lines = _complete_lines(out.read_text())
    selection = revisit_selection(ids)
    kraken_only = dataclasses.replace(th, require_ink_gap=False)
    counts: Counter = Counter()
    rows: List[Dict[str, Any]] = []
    todo: List[Tuple[int, Path]] = []
    resume: List[Tuple[int, Path]] = []
    duplicates: List[Tuple[int, Path]] = []      # a second record of a page planned already: rebuilt from its entry
    planned_paths = set()
    recorded = set()
    for i, line in enumerate(lines):
        r = json.loads(line)
        recorded.add((r["doc_id"], r["image_index"]))
        if selection is not None and not selection.selects(r):
            counts["not_selected"] += 1
            continue
        status, path, entry, frags = revisit_entry(r, raw_dir, vlm_model)
        if status == "ok" and path in planned_paths:
            counts["duplicate"] += 1
            duplicates.append((i, path))
            continue
        counts[status] += 1
        if status == "done" and entry["decode_policy"].get("action") != dp.ACTION_KEEP \
                and "decode_policy" not in r["ai_read"]:
            resume.append((i, path))
        if status != "ok":
            continue
        planned_paths.add(path)
        plan = pdp.plan_from_frags(frags, entry["width"], entry["height"], entry["vlm_raw"], entry["vlm_lines"],
                                   None, kraken_only, rescue_min_kraken_letters)
        rows.append({"doc_id": r["doc_id"], "image_index": r["image_index"], **pdp.plan_row(plan)})
        if plan.action != dp.ACTION_KEEP:
            todo.append((i, path))
    if dry_run:
        rep = revisit_report(rows, counts, out, len(lines), selection, recorded, vlm_model, th, seconds_per_call,
                             rescue_min_kraken_letters)
        path = report or revisit_report_path(out, ids, rescue_min_kraken_letters)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
        print(format_revisit_summary(rep), flush=True)
        print(f"report: {path}", flush=True)
        return rep

    todo = todo[:limit] if limit else todo
    work_dir.mkdir(parents=True, exist_ok=True)
    planned = Counter(r["action"] for r in rows)
    blank = sum(1 for r in rows if r["action"] == dp.ACTION_KEEP and r["reason"].startswith(pdp.BLANK_KRAKEN_REASON))
    counts["skipped_blank"] = blank
    print(f"{len(lines)} records in {out.name}; selected by status {dict(counts)}; plan (Kraken's split alone): "
          f"columns {planned[dp.ACTION_COLUMNS]}, rescue {planned[dp.ACTION_RESCUE]}, skipped as blank (Kraken < "
          f"{rescue_min_kraken_letters} letters) {blank}, keep {planned[dp.ACTION_KEEP] - blank}; "
          f"{len(todo)} to re-read; {len(resume)} records to rebuild from an earlier pass; vlm={vlm_model} "
          f"policy={pdp.POLICY_VERSION} ink_gap={'required' if th.require_ink_gap else 'not required'}", flush=True)
    rebuilt: Dict[int, str] = {}

    def rebuild_from_entry(i: int, path: Path) -> bool:
        """Rebuild record ``i`` from a raw-cache entry the policy already decided (no request).

        :param i: Record position in ``out``.
        :param path: Its raw-cache file.
        :returns: True when the entry holds a policy read the record was rebuilt from.
        """
        r, entry = json.loads(lines[i]), json.loads(path.read_text())
        if (entry.get("decode_policy") or {}).get("action") in (None, dp.ACTION_KEEP):
            return False
        htr_model, key = record_htr(r)
        new = rebuild_record(r, entry, key, htr_model)
        if new is not None:
            rebuilt[i] = json.dumps(new, ensure_ascii=False)
        return new is not None

    for i, path in resume:                       # read by a pass that died before rewriting --out
        rebuild_from_entry(i, path)

    def flush() -> int:
        """Rewrite ``out`` with every record rebuilt so far.

        :returns: Records appended to ``out`` meanwhile (carried over).
        """
        return replace_records(out, [rebuilt.get(i, l) for i, l in enumerate(lines)], len(lines)) if rebuilt else 0

    streak, stopped, carried, reread = 0, False, 0, 0
    for n, (i, path) in enumerate(todo, 1):
        free_gb = shutil.disk_usage(raw_dir).free / 1e9
        if free_gb < min_free_gb:
            print(f"revisit STOP: {free_gb:.1f} GB free < {min_free_gb}", flush=True)
            stopped = True
            break
        r, entry = json.loads(lines[i]), json.loads(path.read_text())
        htr_model, key = record_htr(r)
        tag = f"  {n}/{len(todo)} {r['doc_id']}#{r['image_index']}"
        if entry.get("decode_policy"):           # decided meanwhile by another pass: nothing to read
            counts["decided_meanwhile"] += 1
            rebuild_from_entry(i, path)
            print(f"{tag}: already decided ({entry['decode_policy'].get('action')}), record rebuilt from the cache",
                  flush=True)
            continue
        t0 = time.time()
        image, status, note, fresh = fetch_cached_image(r["doc_id"], r["image_index"], entry, work_dir)
        dl_s = round(time.time() - t0, 1)
        if image is None:
            counts[status] += 1
            print(f"{tag}: {'SKIPPED' if status == 'sha256' else 'FAILED'} {status} {note} (record kept as is)  "
                  f"[dl {dl_s}s]", flush=True)
            continue
        reader = PolicyPageReader(image, entry["width"], entry["height"], vlm_model)
        whole = pdp.Reply(pdp.PAGE_UNIT, pdp.PAGE_UNIT, None, entry["vlm_raw"], bool(entry["parsed"]),
                          list(entry["vlm_lines"]), cached=True)
        t0 = time.time()
        result = await pdp.decode_page(cached_frags(entry, key), entry["width"], entry["height"], th, reader.read,
                                       reader.ink_gap, whole=whole, rescue_min_kraken_letters=rescue_min_kraken_letters)
        vlm_s = round(time.time() - t0, 1)
        if not keep_images:
            reader.remove_crops()
            if fresh:
                image.unlink(missing_ok=True)
        timing = f"[dl {dl_s}s vlm {vlm_s}s]"
        if result is None:
            counts["vlm"] += 1
            streak += 1
            print(f"{tag}: FAILED vlm (record kept as is)  {timing}", flush=True)
            if streak >= REVISIT_MAX_VLM_FAILURES:
                print(f"revisit STOP: {streak} LM Studio failures in a row (is {vlm_model} served?)", flush=True)
                stopped = True
                break
            continue
        streak = 0
        if result.plan.action == dp.ACTION_KEEP:
            counts["gate_kept"] += 1
            print(f"{tag}: keep ({result.plan.reason}; Kraken's {result.plan.kraken_columns} columns without a "
                  f"central ink gap)  {timing}", flush=True)
            continue
        trace = {**result.trace, "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "cached_whole_page": {"vlm_lines": entry["vlm_lines"], "parsed": entry["parsed"]}}
        entry.update(vlm_raw=result.vlm_raw, vlm_lines=result.lines, parsed=result.parsed, decode_policy=trace)
        write_json_atomic(path, entry)
        new = rebuild_record(r, entry, key, htr_model)
        rebuilt[i] = json.dumps(new, ensure_ascii=False)
        reread += 1
        counts[f"read_{result.plan.action}"] += 1
        counts["calls"] += result.calls
        fallback = f" fallback={trace['page_fallback']}" if trace["page_fallback"] else ""
        print(f"{tag}: {result.plan.action}/{result.plan.reason} calls={result.calls} lines "
              f"{r['ai_read']['n_lines']} -> {new['ai_read']['n_lines']} agreed {r['ai_read']['n_agreed']} -> "
              f"{new['ai_read']['n_agreed']} final={trace['final_source']}{fallback}  {timing}", flush=True)
        if flush_every and reread % flush_every == 0:
            carried += flush()
    for i, path in duplicates:                   # the page's other record follows its re-read
        rebuild_from_entry(i, path)
    carried += flush()
    counts["rebuilt"], counts["resumed"], counts["carried"] = len(rebuilt), len(resume), carried
    print(f"revisit {out.name}: {reread} pages re-read (columns {counts['read_columns']}, rescue "
          f"{counts['read_rescue']}; {counts['calls']} model calls), {counts['gate_kept']} kept by the ink-gap gate, "
          f"{counts['skipped_blank']} rescues skipped as blank; "
          f"failed: download {counts['download']}, image {counts['image']}, vlm {counts['vlm']}; {counts['sha256']} "
          f"skipped on sha256 mismatch; records: {len(rebuilt)} rebuilt ({len(resume)} from an earlier pass), "
          f"{carried} appended meanwhile carried over -> {out}", flush=True)
    if not stopped:
        print(f"revisit done {datetime.now().isoformat(timespec='seconds')} -> {out}", flush=True)
    return dict(counts)


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
               "--restamp": "reads no fragments",
               "--revisit": "matches each record against the fragments of its own key"}
    given = [f for f, on in (("--rekraken", args.rekraken), ("--rematch", args.rematch),
                             ("--restamp", args.restamp), ("--revisit", getattr(args, "revisit", False))) if on]
    if not given:
        return None
    return "--htr-cache-key is for normal runs: " + "; ".join(f"{f} {reasons[f]}" for f in given)


def policy_args_error(args: argparse.Namespace) -> Optional[str]:
    """Why the decode-policy options cannot go with the mode given, or None when they can (or are not given).

    ``--decode-policy`` changes how new pages are read (normal runs) and ``--revisit``
    re-reads recorded pages, so neither goes with the rewrite modes, which read
    nothing with the VLM; ``--dry-run`` and ``--report`` belong to ``--revisit``, and
    ``--rescue-min-kraken-letters`` (>= 0) to either policy mode.
    :param args: Parsed CLI arguments.
    :returns: The error message for ``ap.error``, or None.
    """
    revisit_on, policy_on = getattr(args, "revisit", False), getattr(args, "decode_policy", False)
    modes = [f for f, on in (("--rekraken", args.rekraken), ("--rematch", args.rematch), ("--restamp", args.restamp),
                             ("--from-consensus", getattr(args, "from_consensus", None))) if on]
    if revisit_on and modes:
        return f"--revisit re-reads the records in --out (only those of --ids, when given): not with {', '.join(modes)}"
    if policy_on and [m for m in modes if m != "--from-consensus"]:
        return f"--decode-policy reads pages: not with {', '.join(m for m in modes if m != '--from-consensus')}"
    if revisit_on and not Path(args.out).exists():
        return f"--revisit needs an existing --out ({args.out})"
    if getattr(args, "dry_run", False) and not revisit_on:
        return "--dry-run plans a --revisit: give --revisit"
    rescue_min = getattr(args, "rescue_min_kraken_letters", 0) or 0
    if rescue_min < 0:
        return "--rescue-min-kraken-letters must be >= 0"
    if rescue_min and not (policy_on or revisit_on):
        return "--rescue-min-kraken-letters shapes the decode policy: give --decode-policy or --revisit"
    if getattr(args, "report", None) and not getattr(args, "dry_run", False):
        return "--report names the --revisit --dry-run report: give --revisit --dry-run"
    return None


def rekraken_key_error(args: argparse.Namespace) -> Optional[str]:
    """Why a ``--rekraken`` run's HTR cache key is refused, or None when it is explicit (or there is no ``--rekraken``).

    The bare key (the HTR model name: :data:`HTR_MODEL_NAME` for the default model)
    names no Kraken service, so a re-read filed under it relabels every page stamped
    with a service key (``--htr-cache-key`` or ``stamp_htr_cache_key.py``) as that
    reader's: the key must be named with ``--kraken-cache-suffix``, or the bare key
    taken on purpose with ``--legacy-key``.
    :param args: Parsed CLI arguments.
    :returns: The error message for ``ap.error``, or None.
    """
    if not args.rekraken:
        return None
    legacy = getattr(args, "legacy_key", False)
    if legacy and args.kraken_cache_suffix:
        return "--legacy-key and --kraken-cache-suffix are exclusive: the bare key, or KEY"
    if legacy or args.kraken_cache_suffix:
        return None
    _, htr_model, _ = rekraken_names(args.kraken_model, args.htr_model_name, None)
    return (f"--rekraken needs --kraken-cache-suffix KEY naming the Kraken service (e.g. {htr_model}@k7.0.3-blla2026): "
            f"the bare key {htr_model} names none, and a re-read filed under it relabels pages stamped with a "
            f"service key (--htr-cache-key, stamp_htr_cache_key.py) as {htr_model}'s; pass --legacy-key to use the "
            f"bare key on purpose")


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
    ``args.decode_policy`` (``--decode-policy``) reads each new page under the
    decode policy (:func:`stage2_policy`); ``args.revisit`` (``--revisit``) applies
    it to the pages already in ``--out`` instead (:func:`revisit`; raw cache and
    LM Studio only, nothing at all with ``--dry-run``).  Arguments without these
    attributes read exactly as before the options existed.
    :param args: Parsed CLI arguments.
    """
    if getattr(args, "revisit", False):   # raw cache + LM Studio only: no Elasticsearch, no Kraken
        await revisit(Path(args.out), Path(args.work_dir), Path(args.raw_dir) if args.raw_dir else RAW_DIR, args.vlm_model,
                      dp.thresholds_for(VLM_MAX_TOKENS, getattr(args, "require_ink_gap", True)),
                      ids=Path(args.ids) if args.ids else None, dry_run=args.dry_run,
                      report=Path(args.report) if args.report else None, limit=args.limit,
                      min_free_gb=args.min_free_gb, keep_images=args.keep_images, flush_every=args.flush_every,
                      seconds_per_call=args.seconds_per_call,
                      rescue_min_kraken_letters=getattr(args, "rescue_min_kraken_letters", 0))
        return
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
    policy = (dp.thresholds_for(VLM_MAX_TOKENS, getattr(args, "require_ink_gap", True))
              if getattr(args, "decode_policy", False) else None)
    rescue_min = getattr(args, "rescue_min_kraken_letters", 0)
    if policy is not None:
        key_note += f" decode_policy={pdp.POLICY_VERSION}{'' if policy.require_ink_gap else ' (no ink-gap gate)'}"
        key_note += f" rescue_min_kraken_letters={rescue_min}" if rescue_min else ""
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
                                  htr_cache_key=htr_cache_key, policy=policy, keep_images=args.keep_images,
                                  rescue_min_kraken_letters=rescue_min)
        if not args.keep_images and info.get("path"):
            Path(info["path"]).unlink(missing_ok=True)
        if record is None:
            log_failure(info)
            print(f"  {n}/{len(todo)} {job['doc_id']}#{job['image_index']}: FAILED {info.get('failure')}", flush=True)
            continue
        with out.open("a") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        policy_note = ""
        if "policy" in info:
            policy_note = (f" policy={info['policy']} calls={info['calls']}"
                           + (f" fallback={info['policy_fallback']}" if info["policy_fallback"] else ""))
        print(f"  {n}/{len(todo)} {job['doc_id']}#{job['image_index']}: lines {info['n_lines']} agreed {info['n_agreed']} "
              f"frags {info['n_frag']} parsed={info['parsed']}{policy_note}  "
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
                    help=f"--rekraken: frags_by_htr key for the new fragments, naming the Kraken service that reads "
                         f"them (e.g. {HTR_MODEL_NAME}@k7.0.3-blla2026); required unless --legacy-key, so a re-read "
                         f"with a same-named model under a new kraken or segmenter never replaces or relabels "
                         f"earlier fragments")
    ap.add_argument("--legacy-key", action="store_true",
                    help=f"--rekraken: file the re-read under the bare key, the HTR model name ({HTR_MODEL_NAME} "
                         f"for the default model), on purpose; pages stamped with a service key (--htr-cache-key, "
                         f"stamp_htr_cache_key.py) are then re-read and relabelled as that reader's")
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
    ap.add_argument("--decode-policy", action="store_true",
                    help="read each page under the benchmark decode policy: a page Kraken splits into columns (with a "
                         "central ink gap) is read as column crops, a single-column page whose whole-page read fails "
                         "health is re-read as two halves, every other page is read whole as without the flag; the "
                         "raw cache keeps every reply and the policy's trace (decode_policy), the record's ai_read "
                         "gets a short decode_policy stamp")
    ap.add_argument("--require-ink-gap", action=argparse.BooleanOptionalAction, default=True,
                    help="--decode-policy / --revisit: a Kraken column split counts only when the page shows a central "
                         "vertical ink gap, as on the benchmark (default: on)")
    ap.add_argument("--revisit", action="store_true",
                    help="apply the decode policy to the pages already in --out (only those of --ids, when given; no "
                         "index lookup): plan each from its raw-cache Kraken fragments and whole-page read, download "
                         "and re-read only the pages planned as columns or rescue, rewrite their raw-cache entries "
                         "and records in place (others stay byte for byte); resumable; implies --decode-policy")
    ap.add_argument("--dry-run", action="store_true",
                    help="--revisit: plan every page (Kraken's split alone: no image, so no ink-gap test) and write a "
                         "report; no request, no download, no change to --out or the raw cache")
    ap.add_argument("--report", default=None,
                    help="--revisit --dry-run: report path (default logs/decode_policy/revisit_<out stem>"
                         "[__<ids stem>].json)")
    ap.add_argument("--rescue-min-kraken-letters", type=int, default=0, metavar="N",
                    help="--decode-policy / --revisit: never rescue a page whose Kraken fragments hold fewer than N "
                         "letters: keep its read as it is (plan reason blank_kraken_<letters>); 0 = the validated "
                         "policy (default). Column pages are unaffected")
    ap.add_argument("--flush-every", type=int, default=REVISIT_FLUSH_EVERY,
                    help="--revisit: rewrite --out after this many re-read pages (0 = at the end only)")
    ap.add_argument("--seconds-per-call", type=float, default=SECONDS_PER_CALL,
                    help="--revisit --dry-run: seconds per model request for the time estimates")
    a = ap.parse_args()
    if not (a.from_consensus or a.ids or a.restamp or a.rematch or a.rekraken or a.revisit):
        ap.error("give --from-consensus TIER, --ids FILE, --restamp, --rematch, --rekraken, or --revisit")
    rekraken_only = [f for f, v in (("--kraken-model", a.kraken_model), ("--htr-model-name", a.htr_model_name),
                                    ("--kraken-cache-suffix", a.kraken_cache_suffix), ("--legacy-key", a.legacy_key),
                                    ("--all-cache", a.all_cache), ("--force", a.force),
                                    ("--raw-dir", a.raw_dir and not a.revisit)) if v]
    if rekraken_only and not a.rekraken:
        ap.error(f"{', '.join(rekraken_only)}: --rekraken options only")
    key_error = htr_cache_key_error(a) or policy_args_error(a)
    if key_error:
        ap.error(key_error)
    if a.rekraken and (a.from_consensus or a.ids or a.restamp or a.rematch):
        ap.error("--rekraken re-reads the records in --out (or the raw cache): no jobs, no other mode")
    if a.rekraken and not (a.all_cache or Path(a.out).exists()):
        ap.error(f"--rekraken needs an existing --out ({a.out}) or --all-cache")
    rekraken_error = rekraken_key_error(a)
    if rekraken_error:
        ap.error(rekraken_error)
    asyncio.run(main_async(a))
