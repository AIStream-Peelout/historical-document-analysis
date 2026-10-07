"""Drop over-wide or tiny line crops from a Kraken line export's manifests.

Kraken pads every training batch to its widest line and has no length
bucketing, so a handful of very wide crops (merged columns, marginalia,
run-on lines) dominate memory and time; MiDRASH's own lines are ordinary
single-column lines.  This rewrites ``train.txt`` / ``val.txt`` in place
(the unfiltered lists are kept as ``*.all.txt``) using ``index.jsonl`` when
the exporter wrote one, or image headers and ``.gt.txt`` files otherwise.
Two cuts: an absolute aspect cap, and a sparseness cap (aspect per Hebrew
letter) that catches boxes whose geometry does not match their text —
merged columns, run-on boxes, mis-centred strips.

Usage (from repo root):
    PYTHONPATH=. python -m src.finetuning.kraken.filter_manifests \\
        --out /Volumes/home/studio_offload/datasets/kraken_ktiv_lines \\
        [--max-aspect 25] [--max-aspect-per-letter 1.0] [--min-height 20]
"""

import argparse
import json
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, Tuple

from PIL import Image as PILImage

MAX_ASPECT = 25.0        # width / height at any scale; ~3000 px at Kraken's 120-px input
MAX_ASPECT_PER_LETTER = 1.0   # a real line runs ~0.3-0.5 aspect units per letter
MIN_HEIGHT = 20
_HEB_RE = re.compile(r"[א-ת]")
log = logging.getLogger("filter_manifests")


def line_shape(path: Path) -> Tuple[int, int]:
    """Image (width, height) from the header only.

    :param path: Line image.
    :type path: Path
    :return: ``(width, height)`` in px.
    :rtype: Tuple[int, int]
    """
    with PILImage.open(path) as im:
        return im.size


def _header_record(path: Path) -> Tuple[int, int, int]:
    """``(width, height, hebrew letters)`` for one line from its files.

    :param path: Line image (``.gt.txt`` sits next to it).
    :type path: Path
    :return: Width, height and Hebrew letter count.
    :rtype: Tuple[int, int, int]
    """
    w, h = line_shape(path)
    text = path.with_suffix("").with_suffix(".gt.txt").read_text(encoding="utf-8")
    return w, h, len(_HEB_RE.findall(text))


def load_records(out: Path, rels: Iterable[str]) -> Dict[str, Tuple[int, int, int]]:
    """``(width, height, letters)`` per relative path, from ``index.jsonl`` when available.

    :param out: Export root.
    :type out: Path
    :param rels: Relative image paths.
    :type rels: Iterable[str]
    :return: ``rel -> (width, height, letters)``.
    :rtype: Dict[str, Tuple[int, int, int]]
    """
    rels = list(rels)
    recs: Dict[str, Tuple[int, int, int]] = {}
    index = out / "index.jsonl"
    if index.exists():
        for line in index.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            recs[rec["path"]] = (rec["w"], rec["h"], rec["letters"])
    missing = [r for r in rels if r not in recs]
    if missing:
        with ThreadPoolExecutor(16) as pool:
            for rel, rec in zip(missing, pool.map(lambda r: _header_record(out / r), missing)):
                recs[rel] = rec
    return recs


def filter_manifest(out: Path, name: str, max_aspect: float, min_height: int,
                    max_aspect_per_letter: float = MAX_ASPECT_PER_LETTER) -> Counter:
    """Rewrite one manifest without the over-wide / sparse / tiny crops.

    :param out: Export root.
    :type out: Path
    :param name: Manifest file name (``train.txt`` or ``val.txt``).
    :type name: str
    :param max_aspect: Maximum width / height.
    :type max_aspect: float
    :param min_height: Minimum height, px.
    :type min_height: int
    :param max_aspect_per_letter: Maximum (width / height) / Hebrew letters.
    :type max_aspect_per_letter: float
    :return: Counts: ``kept``, ``wide``, ``sparse``, ``tiny``.
    :rtype: Counter
    """
    src = out / name
    keep_all = out / name.replace(".txt", ".all.txt")
    if not keep_all.exists():
        keep_all.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    rels = [r for r in keep_all.read_text(encoding="utf-8").splitlines() if r]
    recs = load_records(out, rels)
    counts: Counter = Counter()
    kept = []
    for rel in rels:
        w, h, letters = recs[rel]
        if h < min_height:
            counts["tiny"] += 1
        elif w / h > max_aspect:
            counts["wide"] += 1
        elif w / h > max_aspect_per_letter * max(1, letters):
            counts["sparse"] += 1
        else:
            kept.append(rel)
            counts["kept"] += 1
    src.write_text("\n".join(kept) + "\n", encoding="utf-8")
    log.info("%s: %s", name, dict(counts))
    return counts


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-aspect", type=float, default=MAX_ASPECT)
    ap.add_argument("--max-aspect-per-letter", type=float, default=MAX_ASPECT_PER_LETTER)
    ap.add_argument("--min-height", type=int, default=MIN_HEIGHT)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for name in ("train.txt", "val.txt"):
        filter_manifest(args.out, name, args.max_aspect, args.min_height, args.max_aspect_per_letter)


if __name__ == "__main__":
    main()
