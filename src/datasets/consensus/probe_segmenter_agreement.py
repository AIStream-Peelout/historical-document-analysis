"""Read-only probe: how a different Kraken service changes the two-reader agreed-line rate.

Samples documentary pages already read by the VLM (raw cache of
``ai_reads_<vlm>.jsonl``), Kraken-reads the same oriented image through a test
service (``--url``, never :8002 while the consensus pipeline uses it) and
rebuilds both sidecars under the current line rule from the cached VLM lines:

* baseline — the cached ``MiDRASH_Gen_01`` fragments (kraken 4 service),
* candidate — the new fragments.

The VLM reads the whole page, so its lines do not depend on the segmenter and
are reused. Images are fresh downloads that must hash to the cached
``image_sha256`` and orient to the cached size (same guard as ``rekraken``).
Nothing in ``ai_reads/`` is written; results go to ``--out`` as JSONL
(resumable per page) and a summary is printed.
"""
import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

_AI = _REPO / "src/datasets/raw_data/cairo_genizah/ai_reads"
_VLM = "qwen3-vl-8b-heb-v21b-step1200"
_BASE_KEY = "MiDRASH_Gen_01"


def sample_pages(n: int, seed: int) -> list:
    """Documentary v21b records with a raw-cache entry holding Gen_01 fragments.

    :param n: Sample size.
    :type n: int
    :param seed: RNG seed.
    :type seed: int
    :return: ``(record, cache entry)`` pairs.
    :rtype: list
    """
    from src.datasets.consensus.two_reader_lines import cached_frags, raw_cache_path
    doc_types = {}
    for l in open(_AI / "jobs_documentary_v5.jsonl"):
        j = json.loads(l)
        doc_types[j["doc_id"]] = j.get("document_type")
    cands = []
    for line in open(_AI / f"ai_reads_{_VLM}.jsonl"):
        r = json.loads(line)
        if r["doc_id"] not in doc_types or not r["ai_read"].get("parsed") or not r.get("image_sha256"):
            continue
        cache = raw_cache_path(r, _VLM)
        if cache.exists():
            r["document_type"] = doc_types[r["doc_id"]]
            cands.append((r, cache))
    random.Random(seed).shuffle(cands)
    out = []
    for r, cache in cands:
        c = json.loads(cache.read_text())
        if cached_frags(c, _BASE_KEY) is not None and c.get("vlm_lines"):
            out.append((r, c))
        if len(out) == n:
            break
    return out


async def probe_page(r: dict, c: dict, model_path: str, work_dir: Path) -> dict:
    """Kraken-read one page through the test service and score both readers.

    :param r: Output record (image url, sha256, size).
    :type r: dict
    :param c: Raw-cache entry (vlm_lines, frags).
    :type c: dict
    :param model_path: Recognition model path.
    :type model_path: str
    :param work_dir: Where oriented images are kept between tags.
    :type work_dir: Path
    :return: Per-page result row.
    :rtype: dict
    """
    from src.datasets.consensus.two_reader_lines import (
        _image_stem, cached_frags, download, prepare_image, run_kraken, sidecar)
    row = {"doc_id": r["doc_id"], "image_index": r["image_index"]}
    w, h = r["image_width"], r["image_height"]
    path = work_dir / f"{_image_stem(r['doc_id'], r['image_index'])}.jpg"
    if not path.exists():
        data = download(r["image_url"])
        if data is None:
            return {**row, "status": "download"}
        if hashlib.sha256(data).hexdigest() != r["image_sha256"]:
            return {**row, "status": "sha256"}
        if prepare_image(data, path) != (w, h):
            path.unlink(missing_ok=True)
            return {**row, "status": "image"}
    t0 = time.time()
    frags = await run_kraken(path, model_path, w, h, timeout=900.0)
    row["kraken_s"] = round(time.time() - t0, 1)
    if frags is None:
        return {**row, "status": "kraken"}
    base = sidecar(c["vlm_lines"], cached_frags(c, _BASE_KEY), c["parsed"], _VLM, "")
    cand = sidecar(c["vlm_lines"], frags, c["parsed"], _VLM, "")
    return {**row, "status": "ok", "n_lines": base["n_lines"],
            "agreed_base": base["n_agreed"], "agreed_cand": cand["n_agreed"],
            "n_frags_base": len(cached_frags(c, _BASE_KEY)), "n_frags_cand": len(frags)}


def summarise(rows: list) -> str:
    """Pooled agreed-line rates for baseline vs candidate.

    :param rows: Result rows with status ``ok``.
    :type rows: list
    :return: One-line summary.
    :rtype: str
    """
    ok = [r for r in rows if r.get("status") == "ok"]
    n = sum(r["n_lines"] for r in ok)
    b = sum(r["agreed_base"] for r in ok)
    k = sum(r["agreed_cand"] for r in ok)
    up = sum(r["agreed_cand"] > r["agreed_base"] for r in ok)
    down = sum(r["agreed_cand"] < r["agreed_base"] for r in ok)
    secs = sorted(r["kraken_s"] for r in ok)
    med = secs[len(secs) // 2] if secs else None
    return (f"{len(ok)} pages, {n} VLM lines: agreed base {b} ({b / max(n, 1):.1%}) -> "
            f"candidate {k} ({k / max(n, 1):.1%}); pages up {up} / down {down}; kraken s/page median {med}")


async def main_async(args: argparse.Namespace) -> None:
    """Run the probe for one tag.

    :param args: Parsed CLI arguments.
    :type args: argparse.Namespace
    """
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if args.out.exists():
        for line in open(args.out):
            x = json.loads(line)
            done[(x["doc_id"], x["image_index"])] = x
    pages = sample_pages(args.n, args.seed)
    from src.models.ocr.kraken_transcriber import preload_kraken_model
    preload_kraken_model(args.kraken_model)
    with open(args.out, "a") as fh:
        for i, (r, c) in enumerate(pages, 1):
            if (r["doc_id"], r["image_index"]) in done:
                continue
            row = await probe_page(r, c, args.kraken_model, args.work_dir)
            row["document_type"] = r.get("document_type")
            done[(r["doc_id"], r["image_index"])] = row
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"  {i}/{len(pages)} {row}", flush=True)
    print(summarise(list(done.values())))


def main() -> None:
    """Parse arguments and run."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--url", default="http://localhost:8003")
    p.add_argument("--out", type=Path, required=True, help="result JSONL for this tag")
    p.add_argument("--work-dir", type=Path, required=True, help="oriented image cache (shared across tags)")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kraken-model", default=str(
        _REPO / "src/datasets/raw_data/cairo_genizah/custom_model_weights/MiDRASH_Gen_01.mlmodel"))
    args = p.parse_args()
    if args.url.rstrip("/").endswith(":8002"):
        sys.exit("refusing :8002 — that is the production service the consensus pipeline uses")
    os.environ["KRAKEN_MICROSERVICE_URL"] = args.url
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
