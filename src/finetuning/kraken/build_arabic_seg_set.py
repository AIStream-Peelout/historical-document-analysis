# File name: build_arabic_seg_set.py
# Date: 10/19/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Downscaled page set for a Kraken segmenter fine-tune on Arabic layouts.

``ketos segtrain`` memory grows with page width (the KTIV segmenter run capped pages at 2,600 px), and the cleaned
PAGE copies written by :mod:`build_arabic_lines` reference the full-resolution images.  This script copies the
PAGE-set pages (Agapet, RASAM, Iskandar) into ``<out-root>/images/<set>/<stem>.jpg`` resized so that the longer side
is at most ``--max-side`` pixels, rewrites every ``Coords``/``Baseline`` points string with
:func:`build_arabic_lines.rescale_points`, points ``imageFilename`` at the absolute resized image and writes the XML
to ``<out-root>/page_xml/<set>/<stem>.xml``.  Lists ``<out-root>/lists/{train,val}.lst`` follow the line sets'
manuscript splits (``<lines-root>/lists/<set>_{train,val}.lst``), so no manuscript sits in both.

Usage (repo root)::

    .venv/bin/python -m src.finetuning.kraken.build_arabic_seg_set [--lines-root DIR] [--out-root DIR] [--max-side 2600] [--sets agapet rasam iskandar]
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lxml import etree
from PIL import Image

from src.finetuning.kraken.build_arabic_lines import rescale_points

logger = logging.getLogger(__name__)

LINES_ROOT = Path("/Volumes/home/studio_offload/datasets/kraken_arabic_lines/v1")
OUT_ROOT = Path("/Volumes/home/studio_offload/datasets/kraken_arabic_seg/v1")
SETS = ("agapet", "rasam", "iskandar")
MAX_SIDE = 2600
JPEG_QUALITY = 92


def _local(element: etree._Element) -> str:
    """Local tag name of an element.

    :param element: XML element.
    :type element: etree._Element
    :return: Tag without namespace.
    :rtype: str
    """
    return etree.QName(element).localname


def rescale_page_xml(data: bytes, image_path: str, scale: float, width: int, height: int) -> bytes:
    """Rewrite a PAGE file for a resized image.

    :param data: Original PAGE XML bytes.
    :type data: bytes
    :param image_path: Absolute path of the resized image.
    :type image_path: str
    :param scale: Resize factor applied to both axes.
    :type scale: float
    :param width: Resized image width.
    :type width: int
    :param height: Resized image height.
    :type height: int
    :return: Rewritten PAGE XML bytes.
    :rtype: bytes
    """
    root = etree.fromstring(data)
    for element in root.iter():
        if not isinstance(element.tag, str):
            continue
        name = _local(element)
        if name == "Page":
            element.set("imageFilename", image_path)
            element.set("imageWidth", str(width))
            element.set("imageHeight", str(height))
        elif name in ("Coords", "Baseline") and element.get("points"):
            element.set("points", rescale_points(element.get("points"), scale, scale, width, height))
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def source_image(xml_path: Path) -> Optional[Path]:
    """The image a cleaned PAGE file points at.

    :param xml_path: PAGE file whose ``imageFilename`` is absolute.
    :type xml_path: Path
    :return: Image path, or None when it does not exist.
    :rtype: Optional[Path]
    """
    root = etree.parse(str(xml_path)).getroot()
    for element in root.iter():
        if isinstance(element.tag, str) and _local(element) == "Page":
            name = element.get("imageFilename") or ""
            path = Path(name) if name.startswith("/") else xml_path.parent / name
            return path if path.exists() else None
    return None


def convert_page(xml_path: Path, out_root: Path, set_name: str, max_side: int) -> Optional[Tuple[Path, float]]:
    """Resize one page's image and rewrite its PAGE file.

    :param xml_path: Cleaned PAGE file.
    :type xml_path: Path
    :param out_root: Output root.
    :type out_root: Path
    :param set_name: Set folder name.
    :type set_name: str
    :param max_side: Longest allowed image side.
    :type max_side: int
    :return: ``(written xml path, scale)`` or None when the image is missing.
    :rtype: Optional[Tuple[Path, float]]
    """
    image_path = source_image(xml_path)
    if image_path is None:
        return None
    out_img = out_root / "images" / set_name / f"{xml_path.stem}.jpg"
    out_xml = out_root / "page_xml" / set_name / f"{xml_path.stem}.xml"
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        scale = min(1.0, max_side / max(im.size))
        if scale < 1.0:
            im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
        im.save(out_img, "JPEG", quality=JPEG_QUALITY)
        width, height = im.size
    out_xml.write_bytes(rescale_page_xml(xml_path.read_bytes(), str(out_img), scale, width, height))
    return out_xml, scale


def build(lines_root: Path, out_root: Path, sets: List[str], max_side: int) -> Dict[str, Dict[str, int]]:
    """Convert every listed page of every set and write the train/val lists.

    :param lines_root: Root of the line dataset (its ``lists/<set>_{train,val}.lst`` give the splits).
    :type lines_root: Path
    :param out_root: Output root.
    :type out_root: Path
    :param sets: Set names.
    :type sets: List[str]
    :param max_side: Longest allowed image side.
    :type max_side: int
    :return: Per-set counts of pages converted and skipped per split.
    :rtype: Dict[str, Dict[str, int]]
    """
    (out_root / "lists").mkdir(parents=True, exist_ok=True)
    stats: Dict[str, Dict[str, int]] = {}
    outputs: Dict[str, List[str]] = {"train": [], "val": []}
    for set_name in sets:
        stats[set_name] = {"train": 0, "val": 0, "skipped": 0, "downscaled": 0}
        for split in ("train", "val"):
            list_path = lines_root / "lists" / f"{set_name}_{split}.lst"
            if not list_path.exists():
                continue
            for line in list_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                result = convert_page(Path(line.strip()), out_root, set_name, max_side)
                if result is None:
                    stats[set_name]["skipped"] += 1
                    continue
                outputs[split].append(str(result[0]))
                stats[set_name][split] += 1
                stats[set_name]["downscaled"] += int(result[1] < 1.0)
        logger.info("%s: %s", set_name, stats[set_name])
    for split, paths in outputs.items():
        (out_root / "lists" / f"{split}.lst").write_text("\n".join(paths) + ("\n" if paths else ""), encoding="utf-8")
    (out_root / "stats.json").write_text(json.dumps({"max_side": max_side, "sets": stats,
                                                     "pages": {k: len(v) for k, v in outputs.items()}}, indent=1), encoding="utf-8")
    return stats


def main(argv: Optional[List[str]] = None) -> None:
    """CLI.

    :param argv: Arguments.
    :type argv: Optional[List[str]]
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lines-root", type=Path, default=LINES_ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--max-side", type=int, default=MAX_SIDE)
    parser.add_argument("--sets", nargs="+", default=list(SETS))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stats = build(args.lines_root, args.out_root, args.sets, args.max_side)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
