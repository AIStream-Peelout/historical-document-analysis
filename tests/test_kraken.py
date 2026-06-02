"""
Quick Kraken test — run a single image through the microservice and print output.

Usage:
    python scripts/test_kraken.py path/to/image.png
    python scripts/test_kraken.py path/to/image.png --model /path/to/model.mlmodel
    python scripts/test_kraken.py path/to/image.png --url http://localhost:8002
"""

import argparse
import asyncio
import sys
from pathlib import Path

import aiohttp

DEFAULT_MODEL = (
    "/Users/isaac/Documents/GitHub/historical-document-analysis"
    "/src/datasets/raw_data/cairo_genizah/custom_model_weights/MiDRASH_Gen_01.mlmodel"
)
DEFAULT_URL = "http://localhost:8002"


async def run(image_path: str, model_path: str, base_url: str) -> None:
    print(f"Image : {image_path}")
    print(f"Model : {model_path}")
    print(f"Server: {base_url}")
    print()

    # ── 1. Preload model ──────────────────────────────────────────────────────
    print("Preloading model...")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/preload",
            json={"model_path": model_path},
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status != 200:
                print(f"  ✗ Preload failed: {resp.status} {await resp.text()}")
                sys.exit(1)
            print(f"  ✓ {await resp.text()}")

    # ── 2. Transcribe ─────────────────────────────────────────────────────────
    print("\nTranscribing...")
    async with aiohttp.ClientSession() as session:
        with open(image_path, "rb") as f:
            form = aiohttp.FormData()
            form.add_field("model_path", model_path)
            form.add_field("image", f, filename=Path(image_path).name)

            async with session.post(
                f"{base_url}/transcribe",
                data=form,
                timeout=aiohttp.ClientTimeout(total=180),
            ) as resp:
                if resp.status != 200:
                    print(f"  ✗ Transcription failed: {resp.status} {await resp.text()}")
                    sys.exit(1)
                result = await resp.json()

    text             = result.get("text", "")
    polygon_failures = result.get("polygon_failures", 0)
    used_binarize    = result.get("used_binarization", False)

    print(f"  ✓ {len(text)} chars  |  {len(text.splitlines())} lines  "
          f"|  polygon_failures={polygon_failures}  |  binarized={used_binarize}")
    print()
    print("─" * 60)
    print(text)
    print("─" * 60)


if __name__ == "__main__":

    asyncio.run(run(image_path="/src/datasets/raw_data/cairo_genizah/evaluations/talmud_sample/crops/01_4_page_001_rashi.png", model_path=DEFAULT_MODEL, base_url=DEFAULT_URL))
