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
``raw/<doc>__<image>.json`` keeps both readers' outputs so a rule change is
a ``--rematch`` (seconds per page) rather than a re-read.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps

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


def _es_request(method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Any:
    """Authenticated JSON request against the Elasticsearch in ``ES_URL``."""
    url = os.environ.get("ES_URL", "http://localhost:9200")
    user = os.environ.get("ELASTICSEARCH_USERNAME") or os.environ.get("ELASTICSEARCH_USER", "")
    auth = base64.b64encode(f"{user}:{os.environ.get('ELASTICSEARCH_PASSWORD', '')}".encode()).decode()
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
            vlm_model: str, vlm_revision: str) -> Dict[str, Any]:
    """Match the two readings and build the ``ai_read`` block.

    :param vlm_lines: ``{text, box}`` from :func:`parse_grounded`.
    :param frags: Kraken fragments ``{text, conf, box}`` (0-1000).
    :param parsed: Whether the VLM reply parsed.
    :param vlm_model: LM Studio key.
    :param vlm_revision: Checkpoint revision string.
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
        "vlm_model": vlm_model, "vlm_revision": vlm_revision, "htr_model": HTR_MODEL_NAME,
        "rule_version": RULE_VERSION,
        "decoded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parsed": parsed, "n_lines": len(lines),
        "n_agreed": sum(l["status"] == "agreed" for l in lines), "lines": lines,
    }


async def stage1(job: Dict[str, Any], work_dir: Path) -> Dict[str, Any]:
    """Download, orient and Kraken-read one image (CPU side).

    :param job: ``{doc_id, image_index, image_url}``.
    :param work_dir: Where the oriented JPEG is written.
    :returns: ``info`` with ``path``, ``width``, ``height``, ``sha256``, ``frags`` — or a ``failure`` key.
    """
    info: Dict[str, Any] = {"doc_id": job["doc_id"], "image_index": job["image_index"]}
    t0 = time.time()
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
    kr = await transcribe_with_kraken_lines(KRAKEN_MODEL, str(path), timeout=300.0)
    info["kraken_s"] = round(time.time() - t0, 1)
    if kr is None:
        info["failure"] = "kraken"
        return info
    info["frags"] = [{"text": l["text"], "conf": l.get("confidence"),
                      "box": [1000 * l["bbox"][0] / width, 1000 * l["bbox"][1] / height,
                              1000 * l["bbox"][2] / width, 1000 * l["bbox"][3] / height]}
                     for l in kr.get("lines", []) if l.get("text", "").strip()]
    return info


async def stage2(job: Dict[str, Any], info: Dict[str, Any], source_index: str, vlm_model: str,
                 vlm_revision: str) -> Optional[Dict[str, Any]]:
    """VLM-read the prepared image, match, and build the loader record (GPU side).

    :param job: The job.
    :param info: Output of :func:`stage1` (no ``failure``).
    :param source_index: Reference merged index stamped into the record.
    :param vlm_model: LM Studio key.
    :param vlm_revision: Checkpoint revision string.
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
        vlm_raw=raw, vlm_lines=vlm_lines, parsed=parsed, frags=info["frags"]), ensure_ascii=False))
    ai_read = sidecar(vlm_lines, info["frags"], parsed, vlm_model, vlm_revision)
    info.update(n_frag=len(info["frags"]), n_lines=ai_read["n_lines"], n_agreed=ai_read["n_agreed"], parsed=parsed)
    return {
        "doc_id": job["doc_id"], "source_index": source_index, "image_index": job["image_index"],
        "image_url": job["image_url"], "image_width": info["width"], "image_height": info["height"],
        "image_sha256": info["sha256"], "ai_read": ai_read,
    }


def raw_stem(job: Dict[str, Any], vlm_model: str) -> str:
    """File stem of a job's raw-cache entry (per VLM checkpoint)."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", job["doc_id"]) + f"__{job['image_index']}__{vlm_model}"


def raw_cache_path(job: Dict[str, Any], vlm_model: str, for_write: bool = False) -> Path:
    """Raw-cache file for a job.

    Writes always use the per-model name.  Reads fall back to the model-less
    name of the first pilot runs only when that file was produced by the same
    VLM (checked from its ``vlm_model`` field), so one checkpoint can never
    be served another's outputs.
    :param job: ``{doc_id, image_index}``.
    :param vlm_model: LM Studio key.
    :param for_write: True when the caller is about to write the entry.
    :returns: Path (may not exist yet).
    """
    new = RAW_DIR / f"{raw_stem(job, vlm_model)}.json"
    if for_write or new.exists():
        return new
    old = RAW_DIR / (re.sub(r"[^A-Za-z0-9_.-]", "_", job["doc_id"]) + f"__{job['image_index']}.json")
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
                     parsed=r["ai_read"]["parsed"], frags=info["frags"])
            cache.write_text(json.dumps(c, ensure_ascii=False))
        ai_read = sidecar(c["vlm_lines"], c["frags"], c["parsed"], c["vlm_model"], vlm_revision)
        rebuilt.append({"doc_id": r["doc_id"], "source_index": source_index, "image_index": r["image_index"],
                        "image_url": r["image_url"], "image_width": c["width"], "image_height": c["height"],
                        "image_sha256": c["sha256"], "ai_read": ai_read})
        print(f"  {n}/{len(rows)} {r['doc_id']}#{r['image_index']}: agreed {r['ai_read']['n_agreed']} -> {ai_read['n_agreed']} "
              f"of {ai_read['n_lines']}", flush=True)
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rebuilt))
    print(f"rematched {len(rebuilt) - failed} records under {RULE_VERSION}; {failed} left unchanged -> {out}", flush=True)


def done_keys(out: Path) -> set:
    """``(doc_id, image_index)`` pairs already recorded in ``out``."""
    keys = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                keys.add((r["doc_id"], r["image_index"]))
    return keys


async def main_async(args: argparse.Namespace) -> None:
    """Resolve jobs, run them sequentially, append records."""
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
    print(f"{len(jobs)} jobs resolved against {source_index}; {len(done)} done; {len(todo)} to run; "
          f"vlm={args.vlm_model} rule={RULE_VERSION}", flush=True)
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
            record = await stage2(job, info, source_index, args.vlm_model, args.vlm_revision)
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
    ap.add_argument("--site-env", default=str(SITE_ENV), help="website .env for ES credentials")
    ap.add_argument("--vlm-model", default=VLM_MODEL)
    ap.add_argument("--vlm-revision", default=VLM_REVISION)
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    ap.add_argument("--keep-images", action="store_true")
    a = ap.parse_args()
    if not (a.from_consensus or a.ids or a.restamp or a.rematch):
            ap.error("give --from-consensus TIER, --ids FILE, --restamp, or --rematch")
    asyncio.run(main_async(a))
