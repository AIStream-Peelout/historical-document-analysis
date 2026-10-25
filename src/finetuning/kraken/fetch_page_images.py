# File name: fetch_page_images.py
# Date: 10/19/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Fetch the page images that the RASAM and Iskandar PAGE XML releases reference, over IIIF.

The Calfa repositories ship PAGE XML without images.  ``list-images.tsv`` (RASAM) and
``iskandar-images-list.csv`` (Iskandar) name the IIIF resources:

* RASAM: ``https://bina.bulac.fr/iiif/2/<IIIF image ID>`` (BULAC; the served full size differs from the
  XML's ``imageWidth``/``imageHeight`` -- both sizes are recorded so the dataset builder can rescale).
* Iskandar: the Gotha manifests at ``dhb.thulb.uni-jena.de`` (one per manuscript); the page's image is
  the canvas whose image id ends in ``_<Image_id>.tif``.  Gallica (BnF) and Leeds rows are skipped:
  Gallica refuses scripted requests and Leeds lists no IIIF resource.

Images are written next to the XML as ``images/<stem>.jpg`` with a ``manifest.jsonl`` of
``{stem, url, sha256, bytes, served_width, served_height, xml_width, xml_height}``.  Resumable.

Usage (repo root)::

    .venv/bin/python -m src.finetuning.kraken.fetch_page_images rasam [--limit N] [--sleep 0.5]
    .venv/bin/python -m src.finetuning.kraken.fetch_page_images iskandar
"""
import argparse
import csv
import hashlib
import io
import json
import logging
import re
import time
import urllib.request
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

EXTERNAL_DIR = Path("/Volumes/home/studio_offload/datasets/arabic_external")
RASAM_DIR = EXTERNAL_DIR / "rasam_pages"
ISKANDAR_DIR = EXTERNAL_DIR / "iskandar_pages"
USER_AGENT = "historical-document-analysis research fetcher (IIIF; contact via repository)"
TIMEOUT = 120


def http_get(url: str, timeout: int = TIMEOUT) -> bytes:
    """GET a URL (follows redirects) and return the body.

    :param url: URL to fetch.
    :type url: str
    :param timeout: Socket timeout in seconds.
    :type timeout: int
    :return: Response body.
    :rtype: bytes
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def xml_size(xml_path: Path) -> Tuple[int, int]:
    """Width and height declared by a PAGE file's ``<Page>`` element.

    :param xml_path: PAGE XML file.
    :type xml_path: Path
    :return: ``(imageWidth, imageHeight)``; ``(0, 0)`` when absent.
    :rtype: Tuple[int, int]
    """
    head = xml_path.read_text(encoding="utf-8", errors="replace")[:4000]
    w = re.search(r'imageWidth="(\d+)"', head)
    h = re.search(r'imageHeight="(\d+)"', head)
    return (int(w.group(1)) if w else 0, int(h.group(1)) if h else 0)


def rasam_jobs(root: Path = RASAM_DIR) -> Iterator[Tuple[str, str, Path]]:
    """``(stem, image url, xml path)`` for every RASAM page with an IIIF image id.

    :param root: Clone of ``calfa-co/rasam-dataset``.
    :type root: Path
    :return: Jobs in list order.
    :rtype: Iterator[Tuple[str, str, Path]]
    """
    with open(root / "list-images.tsv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            stem = row["FileName"]
            image_id = row["IIIF image ID"].strip()
            if not image_id:
                continue
            sub = row["Dataset"].lower()                      # RASAM1 -> rasam1
            xml = root / "page" / sub / f"{stem}.xml"
            if xml.exists():
                yield stem, f"https://bina.bulac.fr/iiif/2/{image_id}/full/full/0/default.jpg", xml


def gotha_image_services(manifest_url: str) -> Dict[str, str]:
    """Map the trailing page number of each Gotha canvas image to its IIIF image service.

    :param manifest_url: IIIF Presentation v2 manifest.
    :type manifest_url: str
    :return: ``{"006": "https://.../Ms-orient-A-02390_006.tif", ...}``.
    :rtype: Dict[str, str]
    """
    manifest = json.loads(http_get(manifest_url))
    services: Dict[str, str] = {}
    for canvas in manifest["sequences"][0]["canvases"]:
        resource = canvas["images"][0]["resource"]
        service = (resource.get("service") or {}).get("@id") or resource["@id"]
        match = re.search(r"_(\d+)\.tif$", service)
        if match:
            services[match.group(1)] = service
    return services


def iskandar_jobs(root: Path = ISKANDAR_DIR) -> Iterator[Tuple[str, str, Path]]:
    """``(stem, image url, xml path)`` for the Iskandar pages hosted at Gotha (IIIF).

    :param root: Clone of ``lipa/iskandar``.
    :type root: Path
    :return: Jobs in list order; Gallica and Leeds rows are skipped.
    :rtype: Iterator[Tuple[str, str, Path]]
    """
    services_by_manifest: Dict[str, Dict[str, str]] = {}
    with open(root / "iskandar-images-list.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            manifest_url = row["IIIF_manifest"].strip()
            if "thulb.uni-jena.de" not in manifest_url:
                continue
            stem = row["XML_File_Name"].replace(".xml", "")
            xml = root / "XML-pages" / f"{stem}.xml"
            if not xml.exists():
                continue
            if manifest_url not in services_by_manifest:
                services_by_manifest[manifest_url] = gotha_image_services(manifest_url)
            service = services_by_manifest[manifest_url].get(row["Image_id"].strip().zfill(3))
            if service:
                yield stem, f"{service}/full/full/0/default.jpg", xml


def fetch_all(jobs: List[Tuple[str, str, Path]], out_dir: Path, sleep: float = 0.5) -> Dict[str, int]:
    """Download every job's image, verify it decodes, and append a manifest row.

    :param jobs: ``(stem, url, xml path)`` triples.
    :type jobs: List[Tuple[str, str, Path]]
    :param out_dir: Target folder (``images/`` next to the XML).
    :type out_dir: Path
    :param sleep: Pause between requests in seconds.
    :type sleep: float
    :return: Counts of ``fetched``, ``skipped`` (already present) and ``failed``.
    :rtype: Dict[str, int]
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    done = set()
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as fh:
            done = {json.loads(line)["stem"] for line in fh if line.strip()}
    counts = {"fetched": 0, "skipped": 0, "failed": 0}
    with open(manifest_path, "a", encoding="utf-8") as mf:
        for n, (stem, url, xml) in enumerate(jobs, 1):
            if stem in done:
                counts["skipped"] += 1
                continue
            try:
                data = http_get(url)
                with Image.open(io.BytesIO(data)) as im:
                    im.load()
                    served_w, served_h = im.size
            except Exception as exc:  # network or decode failure: log and continue with the next page
                counts["failed"] += 1
                logger.warning("%s: %r", stem, exc)
                time.sleep(sleep)
                continue
            (out_dir / f"{stem}.jpg").write_bytes(data)
            xw, xh = xml_size(xml)
            row = {"stem": stem, "url": url, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                   "served_width": served_w, "served_height": served_h, "xml_width": xw, "xml_height": xh}
            mf.write(json.dumps(row) + "\n")
            mf.flush()
            counts["fetched"] += 1
            logger.info("%4d/%d %s %dx%d (xml %dx%d) %.1f MB", n, len(jobs), stem, served_w, served_h, xw, xh, len(data) / 1e6)
            time.sleep(sleep)
    return counts


def main(argv: Optional[List[str]] = None) -> None:
    """CLI.

    :param argv: Arguments.
    :type argv: Optional[List[str]]
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["rasam", "iskandar"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = RASAM_DIR if args.dataset == "rasam" else ISKANDAR_DIR
    jobs = list(rasam_jobs(root) if args.dataset == "rasam" else iskandar_jobs(root))
    if args.limit:
        jobs = jobs[: args.limit]
    counts = fetch_all(jobs, root / "images", sleep=args.sleep)
    print(json.dumps({"dataset": args.dataset, "jobs": len(jobs), **counts}))


if __name__ == "__main__":
    main()
