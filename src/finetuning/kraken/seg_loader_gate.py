"""Verify a PageXML export through kraken's own segmentation training data path (runs INSIDE the container).

``ketos segtrain`` fails soft in three places: an XML that does not parse is
dropped with a warning, a line or region whose type is not in the class map is
skipped, and an image that fails to load is silently replaced by a random other
sample.  This gate makes each of those hard, using the same classes the trainer
uses (``XMLPage(...).to_container()`` → ``BaselineSet`` with blla2026's class
mapping and an ``ImageInputTransforms`` at the model's input height):

* every manifest entry parses, as ``baselines`` type, with ≥ 1 line and ≥ 1 region;
* the dataset's per-class counts equal the XML line / region counts;
* every baseline runs left→right (the start separator must sit at the left end,
  as blla2026 predicts on Hebrew; a reversed baseline yields upside-down lines);
* ``__getitem__`` over the checked items leaves ``failed_samples`` empty, and every
  target has baseline pixels;
* renders ``--render`` targets over their images (baselines red, start separator
  green, end separator blue, regions yellow tint) for eyeballing.

Exit status is non-zero on any failure.
"""
import argparse
import json
import random
import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from kraken.lib.dataset import BaselineSet, ImageInputTransforms  # noqa: E402
from kraken.lib.xml import XMLPage  # noqa: E402

CLASS_MAPPING = {"aux": {"_start_separator": 0, "_end_separator": 1},
                 "baselines": {"default": 2}, "regions": {"text": 3}}   # = blla2026 user_metadata['class_mapping']


def render(sample: dict, dest: Path) -> None:
    """Overlay one dataset sample's target channels on its input image.

    :param sample: ``BaselineSet`` item (``image`` C×H×W float, ``target`` K×H×W).
    :type sample: dict
    :param dest: Output PNG path.
    :type dest: Path
    """
    im = sample["image"]
    arr = (im.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    t = sample["target"].numpy() > 0
    out = arr.astype(np.float32)
    out[t[3]] = out[t[3]] * 0.7 + np.array([255, 255, 0]) * 0.3
    out[t[2]] = [255, 0, 0]
    out[t[0]] = [0, 220, 0]
    out[t[1]] = [0, 80, 255]
    Image.fromarray(out.astype(np.uint8)).save(dest)


def check(lst: Path, height: int, line_width: int, n_items: int, n_render: int, out: Path, seed: int) -> dict:
    """Run every gate for one manifest.

    :param lst: Manifest of PageXML paths.
    :type lst: Path
    :param height: Model input height (1800 for blla).
    :type height: int
    :param line_width: ``--line-width`` the trainer will use.
    :type line_width: int
    :param n_items: Items to load through ``__getitem__`` (0 = all).
    :type n_items: int
    :param n_render: Items to render.
    :type n_render: int
    :param out: Render directory.
    :type out: Path
    :param seed: Sampling seed.
    :type seed: int
    :return: Report dict with an ``errors`` list.
    :rtype: dict
    """
    files = [l.strip() for l in lst.read_text().splitlines() if l.strip()]
    errors, docs = [], []
    n_lines = n_regions = reversed_bl = 0
    for f in files:
        try:
            doc = XMLPage(f, filetype="page").to_container()
        except Exception as e:  # noqa: BLE001 — every parse failure is reported, not raised
            errors.append(f"parse {f}: {e}")
            continue
        if doc.type != "baselines" or not doc.lines or not doc.regions:
            errors.append(f"empty/typed {f}: type={doc.type} lines={len(doc.lines)} regions={len(doc.regions)}")
            continue
        for ln in doc.lines:
            if ln.baseline[0][0] >= ln.baseline[-1][0]:
                reversed_bl += 1
        n_lines += len(doc.lines)
        n_regions += sum(len(v) for v in doc.regions.values())
        docs.append(doc)
    if reversed_bl:
        errors.append(f"{reversed_bl} baselines do not run left→right")
    ds = BaselineSet(class_mapping=json.loads(json.dumps(CLASS_MAPPING)), line_width=line_width,
                     im_transforms=ImageInputTransforms(1, height, 0, 3, (0, 0), valid_norm=False))
    for doc in docs:
        ds.add(doc)
    stats = {k: dict(v) for k, v in ds.class_stats.items()}
    if stats.get("baselines", {}).get("default", 0) != n_lines:
        errors.append(f"dataset baselines {stats.get('baselines')} != XML lines {n_lines}")
    if stats.get("regions", {}).get("text", 0) != n_regions:
        errors.append(f"dataset regions {stats.get('regions')} != XML regions {n_regions}")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    if n_items:
        idx = idx[:n_items]
    shapes = Counter()
    empty_targets = 0
    for k, i in enumerate(idx):
        s = ds[i]
        shapes[tuple(s["image"].shape[:2])] += 1
        if s["target"][2].sum() == 0:
            empty_targets += 1
        if k < n_render:
            render(s, out / f"{lst.stem}_{k:02d}_{Path(ds.imgs[i]).stem}.png")
    if ds.failed_samples:
        errors.append(f"{len(ds.failed_samples)} samples failed to load (would be silently replaced)")
    if empty_targets:
        errors.append(f"{empty_targets} samples have no baseline pixels")
    heights = Counter(sh[1] for sh in shapes.elements())
    return {"manifest": str(lst), "files": len(files), "parsed": len(docs), "lines": n_lines,
            "regions": n_regions, "class_stats": stats, "items_loaded": len(idx),
            "input_heights": dict(heights), "errors": errors}


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--lst", type=Path, nargs="+", required=True)
    p.add_argument("--height", type=int, default=1800)
    p.add_argument("--line-width", type=int, default=8)
    p.add_argument("--items", type=int, default=0, help="items per manifest through __getitem__ (0 = all)")
    p.add_argument("--render", type=int, default=6)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    torch.set_num_threads(4)
    a.out.mkdir(parents=True, exist_ok=True)
    reports = [check(l, a.height, a.line_width, a.items, a.render, a.out, a.seed) for l in a.lst]
    (a.out / "loader_gate.json").write_text(json.dumps(reports, indent=1, ensure_ascii=False))
    print(json.dumps(reports, indent=1, ensure_ascii=False)[:4000])
    sys.exit(1 if any(r["errors"] for r in reports) else 0)


if __name__ == "__main__":
    main()
