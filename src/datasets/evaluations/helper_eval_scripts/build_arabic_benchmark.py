# File name: build_arabic_benchmark.py
# Date: 10/2/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Build the Arabic-script transcription benchmark (v0).

Documents: the "benchmark-ready" tier of ``merged/arabic_scrape_priority_queue.csv`` (built by
``src/datasets/indexing/arabic_scrape_queue.py``): Arabic-primary documents with a Princeton Geniza
Project edition of at least 200 Arabic letters and an image we are permitted to use. Two kinds of
document are left out here:

* **joins**: the edition covers several fragments, so its text cannot be attributed to the one
  fragment in the images;
* **text too dense for the images**: more than 1,000 edition letters per megapixel of image (a
  long deed on a 1,440-pixel photograph: a letter is about ten pixels wide, nobody can read it);
* **documents in a training source set**: checked against the manifests of the source sets the
  fine-tuning mixtures are sampled from (PGP editions, PGP question rows, agreed documentary
  lines) and the ``decontam`` id inventory of the earlier documentary training set. A missing
  manifest stops the build: a benchmark must not be built without this check.

Ground truth: the PGP edition (``pgp_raw/data/footnotes.csv``), reduced to what is on the page by
``arabic_script.split_sections`` (restorations, editor's notes and struck text removed). When a
document has several editions the one with the most visible Arabic letters is used.

Images, in this order: the image store (``image_urls`` of the document in the served merged
index), then the holding library's IIIF manifest listed in the queue. A manifest that describes a
whole volume (more canvases than ``--max-canvases``) is skipped: the folio cannot be identified
from it. Images are capped at ``--max-side`` pixels on the long side.

Output directory (default on the NAS, never under ``raw_data``)::

    benchmark.jsonl        one record per document (metadata, images, ground truth by side), in a
                           fixed pseudo-random order: the first N records are an unbiased sample
    images/<doc_id>__<n>.jpg
    images_manifest.json   every image: source URL, canvas label, bytes, sha256, or the error
    build_report.json      counts, every excluded document and its reason

A full build into the default directory also rewrites the benchmark's registry
(``decontam/arabic_script_benchmark_v0.json``, see ``benchmark_registry``), which training-set
builders read to keep these documents out.

Usage::

    .venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.build_arabic_benchmark \\
        [--out DIR] [--no-iiif] [--max-side 4000] [--limit N] [--training-manifest FILE ...]
"""
import argparse
import csv
import hashlib
import json
import os
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image as PILImage

from src.datasets.evaluations import arabic_script as ar
from src.datasets.evaluations.benchmark_registry import REGISTRY_DIR, write_registry

REPO = Path(__file__).resolve().parents[4]
QUEUE_CSV = REPO / "src/datasets/raw_data/cairo_genizah/merged/arabic_scrape_priority_queue.csv"
FOOTNOTES_CSV = REPO / "src/datasets/raw_data/cairo_genizah/pgp_raw/data/footnotes.csv"
NAS_DATASETS = Path("/Volumes/home/studio_offload/datasets")
DEFAULT_OUT = NAS_DATASETS / "arabic_script_benchmark_v0"
TRAINING_MANIFESTS = tuple(NAS_DATASETS / name / "manifest.jsonl"
                           for name in ("pgp_editions_v1", "pgp_qa_v1", "pgp_qa_v2", "documentary_grounding_v1"))
TRAINING_ID_LISTS = (REPO / "src/datasets/raw_data/cairo_genizah/decontam/clean_v2_ids.json",)
DEFAULT_REGISTRY = REGISTRY_DIR / "arabic_script_benchmark_v0.json"
MIN_ARABIC_LETTERS = 200
MAX_LETTERS_PER_MEGAPIXEL = 1000
USER_AGENT = "historical-document-analysis benchmark builder (research; contact igodfried@isaac26.com)"


def tier1_documents(queue_csv: Path) -> Dict[str, Dict[str, Any]]:
    """Benchmark-ready documents of the Arabic queue, one entry per canonical id.

    The queue has one row per PGP document and fragment; rows that share a canonical id are
    merged (their PGP ids and manifests are collected).

    :param queue_csv: ``arabic_scrape_priority_queue.csv``.
    :type queue_csv: Path
    :return: ``{doc_id: {pgpids, shelfmark, library, doc_type, doc_date, pgp_side, single_fragment, manifests}}``
        in queue order; ``single_fragment`` is False when any of the document's PGP records is a join.
    :rtype: Dict[str, Dict[str, Any]]
    """
    docs: Dict[str, Dict[str, Any]] = {}
    with open(queue_csv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["tier"] != "1" or not row["canonical_id"]:
                continue
            doc = docs.setdefault(row["canonical_id"], {
                "pgpids": [], "shelfmark": row["shelfmark"], "library": row["library"], "doc_type": row["doc_type"],
                "doc_date": row["doc_date"], "pgp_side": row["side"], "single_fragment": True, "manifests": []})
            doc["single_fragment"] = doc["single_fragment"] and row["single_fragment"] == "True"
            if row["pgpid"] not in doc["pgpids"]:
                doc["pgpids"].append(row["pgpid"])
            for url in re.split(r"[;|\s]+", (row["iiif_urls"] or row["suggested_iiif"] or "").strip()):
                if url.startswith("http") and url not in doc["manifests"]:
                    doc["manifests"].append(url)
    return docs


def training_documents(manifests: Iterable[Path], id_lists: Iterable[Path]) -> Dict[str, str]:
    """Documents that are in a training source set.

    :param manifests: ``manifest.jsonl`` of each source set: one JSON object per page or row, with
        ``canonical_id`` or ``doc_id`` and, in the PGP-derived sets, ``pgpid``.
    :type manifests: Iterable[Path]
    :param id_lists: JSON lists of canonical ids (the ``decontam`` inventories).
    :type id_lists: Iterable[Path]
    :return: ``{key: name of the set it was first seen in}``; a key is a canonical id or
        ``"pgp:<pgpid>"``.
    :rtype: Dict[str, str]
    """
    seen: Dict[str, str] = {}
    for path in id_lists:
        with open(path, encoding="utf-8") as fh:
            for doc_id in json.load(fh):
                seen.setdefault(doc_id, path.stem)
    for path in manifests:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                for doc_id in (row.get("canonical_id"), row.get("doc_id")):
                    if doc_id:
                        seen.setdefault(doc_id, path.parent.name)
                if row.get("pgpid"):
                    seen.setdefault(f"pgp:{row['pgpid']}", path.parent.name)
    return seen


def training_set_of(doc_id: str, pgpids: Iterable[str], seen: Dict[str, str]) -> str:
    """The training source set a document is in, if any.

    :param doc_id: Canonical id.
    :type doc_id: str
    :param pgpids: The document's PGP ids.
    :type pgpids: Iterable[str]
    :param seen: :func:`training_documents` result.
    :type seen: Dict[str, str]
    :return: Name of the set, or ``""`` when the document is in none.
    :rtype: str
    """
    for key in [doc_id] + [f"pgp:{pid}" for pid in pgpids]:
        if key in seen:
            return seen[key]
    return ""


def load_editions(footnotes_csv: Path, pgpids: Iterable[str]) -> Dict[str, List[str]]:
    """Edition texts per PGP document, each edition kept separate.

    :param footnotes_csv: PGP export ``footnotes.csv``.
    :type footnotes_csv: Path
    :param pgpids: PGP document ids wanted.
    :type pgpids: Iterable[str]
    :return: ``{pgpid: [edition text, ...]}`` for rows whose relation contains "Edition".
    :rtype: Dict[str, List[str]]
    """
    wanted, out = set(pgpids), {}
    with open(footnotes_csv, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            pid = row.get("document_id") or row.get("document") or ""
            if pid in wanted and "Edition" in row["doc_relation"] and row["content"].strip():
                out.setdefault(pid, []).append(row["content"])
    return out


def ground_truth(editions: List[str]) -> Dict[str, Any]:
    """Visible-text ground truth of one document from its editions.

    :param editions: Edition texts of the document's PGP ids.
    :type editions: List[str]
    :return: ``sections`` (``[{side, lines}]`` of the edition with the most visible Arabic letters),
        ``text``, ``arabic_letters``, ``hebrew_letters``, ``editions`` (how many there were).
    :rtype: Dict[str, Any]
    """
    best: Tuple[int, List[Tuple[str, List[str]]]] = (-1, [])
    for content in editions:
        sections = ar.split_sections(content)
        letters = len(ar.arabic_letters("\n".join(line for _, lines in sections for line in lines)))
        if letters > best[0]:
            best = (letters, sections)
    text = "\n".join(line for _, lines in best[1] for line in lines)
    arabic, hebrew, _ = ar.script_share(text)
    return {"sections": [{"side": side, "lines": lines} for side, lines in best[1]], "text": text,
            "arabic_letters": len(ar.arabic_letters(text)), "hebrew_letters": hebrew, "editions": len(editions)}


def iiif_canvases(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Canvases of a IIIF Presentation manifest (version 2 or 3), in reading order.

    :param manifest: Parsed manifest JSON.
    :type manifest: Dict[str, Any]
    :return: ``[{label, width, height, service, resource}]``; ``service`` is the IIIF Image API base
        URL when the manifest gives one, ``resource`` the image URL itself.
    :rtype: List[Dict[str, Any]]
    """
    canvases = []
    v2 = (manifest.get("sequences") or [{}])[0].get("canvases")
    for canvas in v2 if v2 is not None else manifest.get("items") or []:
        if "images" in canvas:                                   # Presentation 2
            body = canvas["images"][0]["resource"]
            label = canvas.get("label")
        else:                                                    # Presentation 3
            body = canvas["items"][0]["items"][0]["body"]
            label = canvas.get("label")
            if isinstance(label, dict):
                label = " ".join(v for values in label.values() for v in values)
        service = body.get("service") or {}
        service = service[0] if isinstance(service, list) and service else service
        canvases.append({"label": str(label or ""), "width": int(canvas.get("width") or 0), "height": int(canvas.get("height") or 0),
                         "service": (service.get("@id") or service.get("id") or "").rstrip("/") if isinstance(service, dict) else "",
                         "resource": body.get("@id") or body.get("id") or ""})
    return canvases


def iiif_image_url(canvas: Dict[str, Any], max_side: int) -> str:
    """Download URL of a canvas, capped at ``max_side`` pixels on the long side.

    Uses the Image API ``w,`` size form (required at compliance level 1, so it works on every
    server seen so far) and ``full`` when the image is already small enough.

    :param canvas: One entry of :func:`iiif_canvases`.
    :type canvas: Dict[str, Any]
    :param max_side: Largest allowed long side in pixels.
    :type max_side: int
    :return: Image URL (the plain resource URL when the canvas has no image service).
    :rtype: str
    """
    if not canvas["service"]:
        return canvas["resource"]
    width, height = canvas["width"], canvas["height"]
    if not width or not height or max(width, height) <= max_side:
        return f"{canvas['service']}/full/full/0/default.jpg"
    target = max_side if width >= height else round(max_side * width / height)
    return f"{canvas['service']}/full/{target},/0/default.jpg"


def side_of_label(label: str) -> str:
    """Recto / verso from a canvas label such as ``1r``, ``16 verso`` or ``p. 2``.

    :param label: Canvas label.
    :type label: str
    :return: ``"recto"``, ``"verso"`` or ``""`` when the label does not say.
    :rtype: str
    """
    low = label.strip().lower()
    if re.search(r"\brecto\b|\d\s*r\b", low):
        return "recto"
    if re.search(r"\bverso\b|\d\s*v\b", low):
        return "verso"
    return ""


def http_get(url: str, timeout: int = 90) -> bytes:
    """GET a URL with the builder's user agent.

    :param url: URL.
    :type url: str
    :param timeout: Seconds.
    :type timeout: int
    :return: Response body.
    :rtype: bytes
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def save_image(url: str, dst: Path) -> Dict[str, Any]:
    """Download one image unless it is already there.

    :param url: Image URL.
    :type url: str
    :param dst: Destination file.
    :type dst: Path
    :return: ``{file, bytes, sha256}``.
    :rtype: Dict[str, Any]
    """
    if dst.exists() and dst.stat().st_size > 0:
        data = dst.read_bytes()
    else:
        data = http_get(url)
        tmp = dst.with_suffix(".part")
        tmp.write_bytes(data)
        os.replace(tmp, dst)
    return {"file": dst.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def megapixels(path: Path) -> float:
    """Size of an image file in megapixels.

    :param path: Image file.
    :type path: Path
    :return: Width times height, in millions of pixels.
    :rtype: float
    """
    with PILImage.open(path) as image:
        return image.size[0] * image.size[1] / 1e6


def planned_images(doc_id: str, doc: Dict[str, Any], index_urls: List[str], use_iiif: bool, max_side: int,
                   max_canvases: int, delay_s: float) -> Tuple[List[Dict[str, Any]], str]:
    """Where a document's images come from.

    :param doc_id: Canonical id.
    :type doc_id: str
    :param doc: Entry of :func:`tier1_documents`.
    :type doc: Dict[str, Any]
    :param index_urls: ``image_urls`` of the document in the merged index (may be empty).
    :type index_urls: List[str]
    :param use_iiif: Whether library manifests may be used when the index has no image.
    :type use_iiif: bool
    :param max_side: Long-side cap for IIIF images.
    :type max_side: int
    :param max_canvases: Manifests with more canvases are treated as whole volumes and skipped.
    :type max_canvases: int
    :param delay_s: Pause after each manifest request.
    :type delay_s: float
    :return: ``([{url, label, side, source}], note)``; the note says why the list is empty or partial.
    :rtype: Tuple[List[Dict[str, Any]], str]
    """
    if index_urls:
        return [{"url": url, "label": "", "side": "", "source": "image_store"} for url in index_urls], ""
    if not use_iiif:
        return [], "no image in the index (IIIF disabled)"
    if not doc["manifests"]:
        return [], "no image in the index and no manifest"
    images, notes = [], []
    for manifest_url in doc["manifests"]:
        try:
            canvases = iiif_canvases(json.loads(http_get(manifest_url)))
        except (OSError, ValueError, KeyError, IndexError) as error:     # unreachable host, bad JSON, unexpected manifest shape
            notes.append(f"{manifest_url}: {type(error).__name__}")
            continue
        finally:
            time.sleep(delay_s)
        if len(canvases) > max_canvases:
            notes.append(f"{manifest_url}: whole volume ({len(canvases)} canvases)")
            continue
        images += [{"url": iiif_image_url(c, max_side), "label": c["label"], "side": side_of_label(c["label"]),
                    "source": manifest_url} for c in canvases]
    return images, "; ".join(notes)


def build(out: Path, use_iiif: bool = True, max_side: int = 4000, max_canvases: int = 8, limit: Optional[int] = None,
          delay_s: float = 0.5, index: str = "genizah_merged_v8", training_manifests: Sequence[Path] = TRAINING_MANIFESTS,
          training_id_lists: Sequence[Path] = TRAINING_ID_LISTS, registry: Optional[Path] = None) -> Dict[str, Any]:
    """Build the benchmark directory.

    :param out: Output directory.
    :type out: Path
    :param use_iiif: Fetch from library manifests when the image store has no image.
    :type use_iiif: bool
    :param max_side: Long-side cap for IIIF images.
    :type max_side: int
    :param max_canvases: Whole-volume threshold for manifests.
    :type max_canvases: int
    :param limit: Only the first N queue documents (for a dry run).
    :type limit: Optional[int]
    :param delay_s: Pause between requests to library servers.
    :type delay_s: float
    :param index: Served merged index to read ``image_urls`` from.
    :type index: str
    :param training_manifests: Source-set manifests for the training check (:func:`training_documents`).
    :type training_manifests: Sequence[Path]
    :param training_id_lists: Canonical-id inventories for the training check.
    :type training_id_lists: Sequence[Path]
    :param registry: Registry file to write the benchmark's ids to (not written when None).
    :type registry: Optional[Path]
    :return: The build report (also written to ``build_report.json``).
    :rtype: Dict[str, Any]
    """
    from src.datasets.consensus.two_reader_lines import es_get_docs   # needs the site's index credentials

    docs = tier1_documents(QUEUE_CSV)
    if limit:
        docs = dict(list(docs.items())[:limit])
    in_training = training_documents(training_manifests, training_id_lists)
    editions = load_editions(FOOTNOTES_CSV, {pid for d in docs.values() for pid in d["pgpids"]})
    indexed = es_get_docs(sorted(docs), index, "image_urls")
    (out / "images").mkdir(parents=True, exist_ok=True)
    records, image_log, excluded = [], [], {}
    for doc_id, doc in docs.items():
        if not doc["single_fragment"]:
            excluded[doc_id] = "join: the edition covers several fragments"
            continue
        training_set = training_set_of(doc_id, doc["pgpids"], in_training)
        if training_set:
            excluded[doc_id] = f"in a training source set ({training_set})"
            continue
        gt = ground_truth([text for pid in doc["pgpids"] for text in editions.get(pid, [])])
        if gt["arabic_letters"] < MIN_ARABIC_LETTERS:
            excluded[doc_id] = f"fewer than {MIN_ARABIC_LETTERS} visible Arabic letters"
            continue
        planned, note = planned_images(doc_id, doc, (indexed.get(doc_id) or {}).get("image_urls") or [], use_iiif, max_side,
                                       max_canvases, delay_s)
        images = []
        for n, item in enumerate(planned):
            entry = {"doc_id": doc_id, "image_index": n, "image_url": item["url"], "label": item["label"], "side": item["side"],
                     "source": item["source"]}
            try:
                entry.update(save_image(item["url"], out / "images" / f"{doc_id}__{n}.jpg"))
                images.append({k: entry[k] for k in ("file", "image_index", "label", "side", "source")})
            except OSError as error:                                 # 404 in the store, refused by a library server
                entry["error"] = f"{type(error).__name__}: {getattr(error, 'code', '')}".strip(": ")
            if item["source"] != "image_store":
                time.sleep(delay_s)
            image_log.append(entry)
        if not images:
            excluded[doc_id] = note or "every image failed to download"
            continue
        density = gt["arabic_letters"] / sum(megapixels(out / "images" / image["file"]) for image in images)
        if density > MAX_LETTERS_PER_MEGAPIXEL:
            excluded[doc_id] = f"text too dense for the images (over {MAX_LETTERS_PER_MEGAPIXEL} letters per megapixel)"
            continue
        records.append({"id": doc_id, "pgpids": doc["pgpids"], "shelfmark": doc["shelfmark"], "library": doc["library"],
                        "doc_type": doc["doc_type"], "doc_date": doc["doc_date"], "pgp_side": doc["pgp_side"], "images": images,
                        "gt": gt, "letters_per_megapixel": round(density), "image_note": note})
    records.sort(key=lambda record: hashlib.sha1(record["id"].encode("utf-8")).hexdigest())   # any prefix is an unbiased sample
    with open(out / "benchmark.jsonl", "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(out / "images_manifest.json", "w", encoding="utf-8") as fh:
        json.dump({"source_index": index, "images": image_log}, fh, ensure_ascii=False, indent=1)
    if registry is not None:
        write_registry(registry, out.name, [r["id"] for r in records], [pid for r in records for pid in r["pgpids"]],
                       time.strftime("%Y-%m-%d"))
    report = {"queue_documents": len(docs), "benchmark_documents": len(records),
              "images": sum(len(r["images"]) for r in records),
              "image_sources": dict(Counter("image_store" if i["source"] == "image_store" else "library_iiif"
                                            for r in records for i in r["images"])),
              "failed_images": sum(1 for e in image_log if "error" in e),
              "excluded": dict(Counter(re.sub(r"^https?://\S+: ", "", reason) for reason in excluded.values())),
              "excluded_documents": excluded, "training_sets_checked": [str(p) for p in list(training_id_lists) + list(training_manifests)],
              "by_library": dict(Counter(r["library"] for r in records)), "by_type": dict(Counter(r["doc_type"] for r in records)),
              "arabic_letters_total": sum(r["gt"]["arabic_letters"] for r in records),
              "with_hebrew_text": sum(1 for r in records if r["gt"]["hebrew_letters"] >= 20),
              "single_image": sum(1 for r in records if len(r["images"]) == 1)}
    with open(out / "build_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    return report


def main(argv: Optional[List[str]] = None) -> None:
    """CLI.

    :param argv: Arguments.
    :type argv: Optional[List[str]]
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-iiif", action="store_true", help="only use images already in the image store")
    parser.add_argument("--max-side", type=int, default=4000)
    parser.add_argument("--max-canvases", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--training-manifest", type=Path, nargs="*", default=list(TRAINING_MANIFESTS),
                        help="manifest.jsonl of every training source set to check against")
    args = parser.parse_args(argv)
    full_default_build = args.out == DEFAULT_OUT and args.limit is None        # only that build defines what is held out
    report = build(args.out, use_iiif=not args.no_iiif, max_side=args.max_side, max_canvases=args.max_canvases, limit=args.limit,
                   training_manifests=args.training_manifest, registry=DEFAULT_REGISTRY if full_default_build else None)
    print(json.dumps({k: v for k, v in report.items() if k != "excluded_documents"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
