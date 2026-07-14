"""Crop the full Talmud corpus (~5,373 pages) into section images for training.

Thin driver around the proven benchmark cropper
(:func:`src.datasets.evaluations.helper_eval_scripts.talmud_crop.run_crop_pipeline`),
pointed at ``talmud_full/talmud_complete`` instead of the benchmark sample.

Adds one safety layer on top of the existing pipeline: an aspect-ratio /
width outlier scan. The percentage boundaries are calibrated to ~1734 px-wide
Vilna scans; pages that deviate strongly are flagged into
``crop_outliers.json`` for manual boundary overrides rather than silently
producing mis-aligned crops.

Usage
-----
  # Calibration preview on a few pages spread across the corpus
  python -m src.finetuning.qwen_hebrew.crop_full_corpus --preview --max_pages 5

  # Outlier scan only (fast, no crops written)
  python -m src.finetuning.qwen_hebrew.crop_full_corpus --scan_only

  # Full crop run
  python -m src.finetuning.qwen_hebrew.crop_full_corpus
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from src.datasets.evaluations.helper_eval_scripts.talmud_crop import (
    IMAGE_EXTENSIONS,
    run_crop_pipeline,
)

_REPO = Path(__file__).resolve().parents[3]

TALMUD_FULL_DIR = (
    _REPO / "src/datasets/raw_data/cairo_genizah/evaluations/talmud_full/talmud_complete"
)
FULL_IMAGES_DIR = TALMUD_FULL_DIR / "converted_images"
FULL_CROPS_DIR = TALMUD_FULL_DIR / "crops"
FULL_BOUNDARY_CONFIG = TALMUD_FULL_DIR / "crop_boundaries.json"
OUTLIERS_PATH = TALMUD_FULL_DIR / "crop_outliers.json"

# Aspect ratio (height / width) deviating more than this fraction from the
# corpus median is flagged for manual review.
ASPECT_DEVIATION_THRESHOLD = 0.10


def scan_outliers(
    images_dir: Path = FULL_IMAGES_DIR,
    outliers_path: Path = OUTLIERS_PATH,
    threshold: float = ASPECT_DEVIATION_THRESHOLD,
) -> List[Dict]:
    """Scan page dimensions and flag aspect-ratio outliers.

    :param images_dir: Directory of full-page images.
    :param outliers_path: Where to write the outlier report JSON.
    :param threshold: Max fractional deviation of height/width from the
        corpus median before a page is flagged.
    :returns: List of outlier entries (``stem``, ``width``, ``height``,
        ``aspect``, ``deviation``).
    """
    dims: Dict[str, tuple] = {}
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        with Image.open(p) as img:
            dims[p.stem] = img.size  # (w, h) — lazy, does not decode pixels

    if not dims:
        raise FileNotFoundError(f"No images found in {images_dir}")

    aspects = {stem: h / w for stem, (w, h) in dims.items()}
    median_aspect = statistics.median(aspects.values())
    median_width = statistics.median(w for w, _ in dims.values())

    outliers = []
    for stem, aspect in aspects.items():
        deviation = abs(aspect - median_aspect) / median_aspect
        if deviation > threshold:
            w, h = dims[stem]
            outliers.append(
                {
                    "stem": stem,
                    "width": w,
                    "height": h,
                    "aspect": round(aspect, 4),
                    "deviation": round(deviation, 4),
                }
            )

    report = {
        "pages_scanned": len(dims),
        "median_width": median_width,
        "median_aspect": round(median_aspect, 4),
        "threshold": threshold,
        "outliers": sorted(outliers, key=lambda o: -o["deviation"]),
    }
    outliers_path.parent.mkdir(parents=True, exist_ok=True)
    outliers_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Scanned {len(dims)} pages — median width {median_width}px, "
        f"median aspect {median_aspect:.3f}; {len(outliers)} outliers "
        f"(>{threshold:.0%} aspect deviation) → {outliers_path}"
    )
    return outliers


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point.

    :param argv: Optional argument list (defaults to ``sys.argv``).
    """
    parser = argparse.ArgumentParser(
        description="Crop the full Talmud corpus into gemara/rashi/tosafot section images."
    )
    parser.add_argument("--images_dir", type=Path, default=FULL_IMAGES_DIR)
    parser.add_argument("--output_dir", type=Path, default=FULL_CROPS_DIR)
    parser.add_argument("--boundary_config", type=Path, default=FULL_BOUNDARY_CONFIG)
    parser.add_argument("--max_pages", type=int, default=None)
    parser.add_argument("--preview", action="store_true", help="Show boundary overlays.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--scan_only", action="store_true",
        help="Only run the aspect-ratio outlier scan; write crop_outliers.json and exit.",
    )
    parser.add_argument(
        "--skip_scan", action="store_true",
        help="Skip the outlier scan before cropping.",
    )
    args = parser.parse_args(argv)

    if not args.skip_scan:
        outliers = scan_outliers(images_dir=args.images_dir)
        if outliers:
            print(
                f"⚠️  {len(outliers)} pages deviate from the calibrated geometry — "
                f"review {OUTLIERS_PATH} and add per-page overrides to "
                f"{args.boundary_config} if their crops look wrong."
            )
    if args.scan_only:
        return

    run_crop_pipeline(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        boundary_config_path=args.boundary_config,
        max_pages=args.max_pages,
        preview=args.preview,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
