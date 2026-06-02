"""
talmud_crop.py
Crop Vilna-edition Talmud pages into three benchmark regions:
  gemara (centre), rashi (inner margin), tosafot (outer margin)

Recto pages (stem does NOT end in 'b'): Rashi is on the right, Tosafot on the left.
Verso pages (stem ends in 'b'):          Rashi is on the left, Tosafot on the right.

Default percentage boundaries are calibrated to the scans in the Talmud sample
(~1734 px wide).  Per-page overrides can be stored in a JSON boundary config so
every crop in the benchmark release is fully reproducible.

Outputs
-------
  <output_dir>/
    <stem>_gemara.png
    <stem>_rashi.png
    <stem>_tosafot.png
  crop_manifest.json   — pixel + percentage boundaries for every crop produced

Usage
-----
  # Dry-run preview (draws coloured boxes, does not write crops)
  python -m src.datasets.evaluations.talmud_crop --preview --max_pages 3

  # Generate all crops with defaults
  python -m src.datasets.evaluations.talmud_crop

  # Use a custom boundary config and custom output dir
  python -m src.datasets.evaluations.talmud_crop \\
      --boundary_config my_boundaries.json \\
      --output_dir /path/to/crops
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image, ImageDraw

# ── Default image directories (same defaults as talmud_evaluation.py) ────────

_REPO = Path(__file__).resolve().parents[3]   # project root

TALMUD_IMAGES_DIR = (
    _REPO / "src/datasets/raw_data/cairo_genizah/evaluations/talmud_sample/converted_images"
)
DEFAULT_OUTPUT_DIR = (
    _REPO / "src/datasets/raw_data/cairo_genizah/evaluations/talmud_sample/crops"
)
DEFAULT_BOUNDARY_CONFIG = (
    _REPO / "src/datasets/raw_data/cairo_genizah/evaluations/talmud_sample/crop_boundaries.json"
)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# ── Section names (canonical order) ──────────────────────────────────────────

SECTIONS = ("gemara", "rashi", "tosafot")

# ── Default crop boundaries (percentage of image width / height) ──────────────
#
# Calibrated to the Vilna-edition scans in the Talmud sample dataset.
# All values are fractions in [0, 1]:  x1, y1 = top-left;  x2, y2 = bottom-right
#
# Recto pages (odd folios, no 'b' suffix):
#   inner margin (Rashi) is on the RIGHT side of the image.
# Verso pages (even folios, 'b' suffix):
#   inner margin (Rashi) is on the LEFT side of the image.

VILNA_RECTO_DEFAULTS: Dict[str, Dict[str, float]] = {
    "tosafot": {"x1": 0.00, "y1": 0.04, "x2": 0.19, "y2": 0.97},
    "gemara":  {"x1": 0.19, "y1": 0.04, "x2": 0.80, "y2": 0.97},
    "rashi":   {"x1": 0.80, "y1": 0.04, "x2": 1.00, "y2": 0.97},
}

VILNA_VERSO_DEFAULTS: Dict[str, Dict[str, float]] = {
    "rashi":   {"x1": 0.00, "y1": 0.04, "x2": 0.20, "y2": 0.97},
    "gemara":  {"x1": 0.20, "y1": 0.04, "x2": 0.81, "y2": 0.97},
    "tosafot": {"x1": 0.81, "y1": 0.04, "x2": 1.00, "y2": 0.97},
}

# Colours used for the preview overlay (section → RGB)
_PREVIEW_COLOURS = {
    "gemara":  (30,  144, 255),   # dodger blue
    "rashi":   (255, 140,  0),    # orange
    "tosafot": (50,  205,  50),   # lime green
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_verso(stem: str) -> bool:
    """Return True if the page stem indicates a verso (back-of-folio) page."""
    # Stems like '01_10b_page_001' or '01_10b'
    return stem.lower().replace("_page_001", "").endswith("b")


def _pct_to_pixels(
    bounds: Dict[str, float], width: int, height: int
) -> Tuple[int, int, int, int]:
    """Convert percentage bounds dict → integer pixel box (left, top, right, bottom)."""
    left   = max(0, int(bounds["x1"] * width))
    top    = max(0, int(bounds["y1"] * height))
    right  = min(width,  int(bounds["x2"] * width))
    bottom = min(height, int(bounds["y2"] * height))
    return left, top, right, bottom


def _default_bounds(stem: str) -> Dict[str, Dict[str, float]]:
    """Return the appropriate recto/verso default boundary dict for a page stem."""
    return VILNA_VERSO_DEFAULTS if _is_verso(stem) else VILNA_RECTO_DEFAULTS


# ── Boundary config I/O ───────────────────────────────────────────────────────

def load_boundary_config(config_path: Path) -> Dict:
    """Load boundary config JSON; return empty dict if file does not exist."""
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def save_boundary_config(config: Dict, config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"  💾 Boundary config saved → {config_path}")


def _resolve_bounds(
    stem: str,
    section: str,
    overrides: Dict,
) -> Dict[str, float]:
    """Return the boundary dict for one stem+section, applying overrides if present."""
    page_overrides = overrides.get("overrides", {}).get(stem, {})
    if section in page_overrides:
        return page_overrides[section]
    global_defaults = overrides.get("defaults", {})
    if section in global_defaults:
        return global_defaults[section]
    return _default_bounds(stem)[section]


# ── Core crop logic ───────────────────────────────────────────────────────────

def crop_page(
    image_path: Path,
    output_dir: Path,
    boundary_config: Dict,
    dry_run: bool = False,
) -> Optional[Dict]:
    """Crop one Talmud page into three section images.

    Returns a manifest entry dict, or None if the image cannot be opened.
    Does not write anything when dry_run=True.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  ✗ Cannot open {image_path.name}: {e}")
        return None

    stem = image_path.stem          # e.g. '01_10_page_001'
    w, h = img.size

    manifest_entry: Dict = {
        "stem": stem,
        "source_image": str(image_path),
        "width": w,
        "height": h,
        "verso": _is_verso(stem),
        "crops": {},
    }

    for section in SECTIONS:
        bounds = _resolve_bounds(stem, section, boundary_config)
        left, top, right, bottom = _pct_to_pixels(bounds, w, h)
        crop_img = img.crop((left, top, right, bottom))

        crop_filename = f"{stem}_{section}.png"
        crop_path = output_dir / crop_filename

        manifest_entry["crops"][section] = {
            "path": str(crop_path),
            "bounds_pct": bounds,
            "bounds_px":  {"x1": left, "y1": top, "x2": right, "y2": bottom},
            "width":  right - left,
            "height": bottom - top,
        }

        if not dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            crop_img.save(crop_path, "PNG")

    return manifest_entry


def preview_page(image_path: Path, boundary_config: Dict) -> None:
    """Draw coloured crop-boundary boxes on the image and display it."""
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  ✗ Cannot open {image_path.name}: {e}")
        return

    stem = image_path.stem
    w, h = img.size
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")

    for section in SECTIONS:
        bounds = _resolve_bounds(stem, section, boundary_config)
        left, top, right, bottom = _pct_to_pixels(bounds, w, h)
        colour = _PREVIEW_COLOURS[section]
        # Semi-transparent fill + solid border
        draw.rectangle(
            [left, top, right, bottom],
            fill=(*colour, 40),
            outline=(*colour, 220),
            width=4,
        )
        # Label in the top-left of the region
        label_x = left + 8
        label_y = top + 8
        draw.text((label_x + 1, label_y + 1), section, fill=(0, 0, 0, 220))
        draw.text((label_x, label_y), section, fill=(*colour, 255))

    # Scale down for display (max 900px tall)
    scale = min(1.0, 900 / h)
    disp = overlay.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    try:
        disp.show(title=f"Crop preview — {stem}")
        print(f"  👁  Preview opened for {stem} ({w}×{h}px, scale={scale:.2f})")
    except Exception:
        # Headless environment — save to /tmp instead
        tmp = Path(f"/tmp/talmud_preview_{stem}.png")
        disp.save(tmp)
        print(f"  💾 Preview saved (headless) → {tmp}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_crop_pipeline(
    images_dir: Path = TALMUD_IMAGES_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    boundary_config_path: Path = DEFAULT_BOUNDARY_CONFIG,
    max_pages: Optional[int] = None,
    preview: bool = False,
    dry_run: bool = False,
    save_config_template: bool = False,
) -> None:
    """
    Main entry point.

    Args:
        images_dir:           Directory containing full-page Talmud PNG/TIFF images.
        output_dir:           Where to write crop images (created if absent).
        boundary_config_path: JSON file with per-page overrides (need not exist).
        max_pages:            Limit processing to the first N pages.
        preview:              Show boundary overlay before cropping each page.
        dry_run:              Parse and show what would be done; don't write files.
        save_config_template: Write a starter boundary config with the built-in
                              defaults and exit — useful for creating an override file.
    """

    boundary_config = load_boundary_config(boundary_config_path)

    if save_config_template:
        template = {
            "_comment": (
                "Per-page override example.  Add stems under 'overrides' to "
                "adjust individual pages.  Values are fractions 0–1."
            ),
            "defaults": {
                "recto": VILNA_RECTO_DEFAULTS,
                "verso": VILNA_VERSO_DEFAULTS,
            },
            "overrides": {
                "EXAMPLE_01_10_page_001": {
                    "rashi":   {"x1": 0.79, "y1": 0.04, "x2": 1.00, "y2": 0.97},
                    "gemara":  {"x1": 0.18, "y1": 0.04, "x2": 0.79, "y2": 0.97},
                    "tosafot": {"x1": 0.00, "y1": 0.04, "x2": 0.18, "y2": 0.97},
                }
            },
        }
        save_boundary_config(template, boundary_config_path)
        print("Template saved. Edit the 'overrides' section then re-run without --save_config_template.")
        return

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(f"✗ No images found in {images_dir}")
        sys.exit(1)

    if max_pages:
        image_paths = image_paths[:max_pages]

    print(f"\n{'DRY RUN — ' if dry_run else ''}Cropping {len(image_paths)} Talmud pages")
    print(f"  Images dir:  {images_dir}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Config:      {boundary_config_path} ({'loaded' if boundary_config else 'using defaults'})")
    print()

    manifest = []
    skipped  = 0

    for i, image_path in enumerate(image_paths, 1):
        stem  = image_path.stem
        verso = _is_verso(stem)
        print(f"[{i}/{len(image_paths)}] {stem}  ({'verso' if verso else 'recto'})")

        if preview:
            preview_page(image_path, boundary_config)

        entry = crop_page(image_path, output_dir, boundary_config, dry_run=dry_run)
        if entry is None:
            skipped += 1
            continue

        manifest.append(entry)

        # Print per-section pixel sizes for quick sanity check
        for section, info in entry["crops"].items():
            px = info["bounds_px"]
            print(
                f"    {section:8s}: "
                f"x={px['x1']}-{px['x2']} ({info['width']}px)  "
                f"y={px['y1']}-{px['y2']} ({info['height']}px)"
            )

    # ── Save manifest ─────────────────────────────────────────────────────────
    if not dry_run and manifest:
        manifest_path = output_dir / "crop_manifest.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_pages": len(manifest),
                    "total_crops": len(manifest) * 3,
                    "sections": list(SECTIONS),
                    "boundary_config_path": str(boundary_config_path),
                    "pages": manifest,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\n✅ Done — {len(manifest)} pages × 3 regions = {len(manifest) * 3} crops")
        print(f"   Crops:    {output_dir}")
        print(f"   Manifest: {manifest_path}")
        if skipped:
            print(f"   ⚠️  Skipped: {skipped} images (could not open)")
    elif dry_run:
        print(f"\n[DRY RUN] Would produce {len(manifest)} × 3 = {len(manifest) * 3} crops.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop Vilna-edition Talmud pages into gemara / rashi / tosafot regions."
    )
    parser.add_argument(
        "--images_dir", type=Path, default=TALMUD_IMAGES_DIR,
        help="Directory of full-page Talmud images.",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Where to write crop images (default: talmud_sample/crops/).",
    )
    parser.add_argument(
        "--boundary_config", type=Path, default=DEFAULT_BOUNDARY_CONFIG,
        help="JSON file with per-page boundary overrides.",
    )
    parser.add_argument(
        "--max_pages", type=int, default=None,
        help="Process only the first N pages (useful for smoke tests).",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Overlay coloured crop boxes on each page before writing crops.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Show what would be done without writing any files.",
    )
    parser.add_argument(
        "--save_config_template", action="store_true",
        help="Write a starter crop_boundaries.json with built-in defaults and exit.",
    )

    args = parser.parse_args()

    run_crop_pipeline(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        boundary_config_path=args.boundary_config,
        max_pages=args.max_pages,
        preview=args.preview,
        dry_run=args.dry_run,
        save_config_template=args.save_config_template,
    )
