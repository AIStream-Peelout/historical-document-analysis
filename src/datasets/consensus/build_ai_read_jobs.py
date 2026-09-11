# File name: build_ai_read_jobs.py
# Date: 9/8/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Build a job list for ``two_reader_lines.py --ids`` from the newest merged index.

Selects untranscribed documents with images (optionally by document type and
institution), emits ``{doc_id, image_index, image_url}`` per image — the
first image per document by default (usually the recto), or every image with
``--all-images``.  Read-only against Elasticsearch.

Usage (repo root):
    .venv/bin/python -m src.datasets.consensus.build_ai_read_jobs --types letter legal ketubah business \\
        --out src/datasets/raw_data/cairo_genizah/ai_reads/jobs_documentary.jsonl
"""
import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
from src.datasets.consensus.two_reader_lines import (  # noqa: E402
    SITE_ENV, _es_request, latest_merged_index, load_site_env)


def scan(index: str, types: List[str], institutions: List[str], transcribed: bool) -> List[Dict[str, Any]]:
    """Scroll the matching documents.

    :param index: Merged index.
    :param types: ``document_type`` values to keep (empty = any).
    :param institutions: Substrings that must appear in ``institution`` (empty = any).
    :param transcribed: Keep documents that already have transcriptions (default: only untranscribed).
    :returns: ``_source`` dicts with ``_id`` added.
    """
    must: List[Dict[str, Any]] = [{"exists": {"field": "image_urls"}}]
    must_not: List[Dict[str, Any]] = [] if transcribed else [{"term": {"has_transcriptions": True}}]
    if types:
        must.append({"terms": {"document_type": types}})
    query = {"bool": {"must": must, "must_not": must_not}}
    res = _es_request("POST", f"/{index}/_search?scroll=2m",
                      {"size": 5000, "query": query, "_source": ["image_urls", "document_type", "institution", "shelf_mark"]})
    sid, hits, docs = res["_scroll_id"], res["hits"]["hits"], []
    while hits:
        for h in hits:
            src = h["_source"]
            src["_id"] = h["_id"]
            if not institutions or any(s.lower() in str(src.get("institution", "")).lower() for s in institutions):
                docs.append(src)
        res = _es_request("POST", "/_search/scroll", {"scroll": "2m", "scroll_id": sid})
        sid, hits = res["_scroll_id"], res["hits"]["hits"]
    _es_request("DELETE", "/_search/scroll", {"scroll_id": sid})
    return docs


def main() -> None:
    """Write the job list and print its composition."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--types", nargs="*", default=[], help="document_type values (e.g. letter legal ketubah)")
    ap.add_argument("--institutions", nargs="*", default=[], help="substrings of institution")
    ap.add_argument("--all-images", action="store_true", help="every image, not just the first")
    ap.add_argument("--include-transcribed", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap on documents (in index order)")
    ap.add_argument("--source-index", default="latest")
    ap.add_argument("--site-env", default=str(SITE_ENV))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    load_site_env(Path(a.site_env))
    index = latest_merged_index() if a.source_index == "latest" else a.source_index
    docs = scan(index, a.types, a.institutions, a.include_transcribed)
    docs.sort(key=lambda d: d["_id"])
    if a.limit:
        docs = docs[:a.limit]
    jobs = []
    for d in docs:
        urls = d.get("image_urls") or []
        for i, u in enumerate(urls if a.all_images else urls[:1]):
            jobs.append({"doc_id": d["_id"], "image_index": i, "image_url": u,
                         "document_type": d.get("document_type"), "institution": d.get("institution")})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in jobs))
    print(f"{index}: {len(docs)} documents -> {len(jobs)} image jobs -> {out}")
    print("  types:", collections.Counter(str(d.get('document_type')) for d in docs).most_common(8))
    print("  institutions:", collections.Counter(str(d.get('institution')).split('/')[0].strip()[:28] for d in docs).most_common(6))
    print(f"  Studio estimate at ~55 s/image with prefetch: {len(jobs) * 55 / 3600:.1f} h; at 90 s: {len(jobs) * 90 / 3600:.1f} h")


if __name__ == "__main__":
    main()
