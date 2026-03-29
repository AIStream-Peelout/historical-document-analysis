"""
kraken_transcriber.py
Async wrapper around synchronous Kraken HTR for the transcription pipeline.

Key design decisions:
- Kraken is entirely synchronous; we run it in a thread executor so it doesn't
  block the asyncio event loop (and so LangGraph timeouts remain usable).
- The Kraken model is expensive to load (~2–3 s). We cache it in a module-level
  singleton so repeated calls within the same process pay load cost only once.
- Binarization is always applied — Genizah and Talmud pages both benefit from it.
- Segmentation uses 'horizontal-rl' (right-to-left baselines) which is required
  for Hebrew manuscript material.

Usage:
    from kraken_transcriber import transcribe_with_kraken, preload_kraken_model

    # Optional: preload once at startup to amortize cost
    preload_kraken_model("/path/to/MiDRASH_Gen_01.mlmodel")

    # Then call from any async context
    text = await transcribe_with_kraken(model_path, image_path)
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional
from kraken.lib import models  # lazy import
from kraken import blla, binarization, rpred
from kraken.lib import models
from PIL import Image

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Module-level model cache                                                     #
# --------------------------------------------------------------------------- #

_cached_model = None
_cached_model_path: Optional[str] = None


def preload_kraken_model(model_path: str) -> None:
    """Load the Kraken model into the module cache.

    Call this once at the start of an evaluation run to avoid paying the load
    cost on every document.  Safe to call multiple times with the same path.
    """
    global _cached_model, _cached_model_path

    if _cached_model_path == model_path and _cached_model is not None:
        return  # Already loaded



    logger.info(f"Loading Kraken model: {model_path}")
    _cached_model = models.load_any(model_path)
    _cached_model_path = model_path
    logger.info("Kraken model loaded and cached.")


# --------------------------------------------------------------------------- #
# Synchronous core (runs in a thread executor)                                 #
# --------------------------------------------------------------------------- #

def _transcribe_sync(model_path: str, image_path: str) -> str:
    """Run the full Kraken pipeline synchronously.

    Steps:
      1. Load image with PIL
      2. Binarize (nlbin) — critical for degraded manuscript material
      3. Segment into baselines with RTL direction
      4. Run recognition with the loaded model
      5. Join predicted lines into a single string (preserving line order)

    Raises on hard failures; the async wrapper catches and logs them.
    """
    # Use cached model if available; otherwise load now (slower)
    global _cached_model, _cached_model_path
    if _cached_model_path == model_path and _cached_model is not None:
        model = _cached_model
    else:
        logger.warning("Kraken model not pre-cached — loading on-demand (slow).")
        model = models.load_any(model_path)

    im = Image.open(image_path)

    # Binarize: converts to 1-bit image Kraken expects for best accuracy
    binarized = binarization.nlbin(im)

    # Baseline segmentation — handles complex manuscript layouts better than
    # bounding-box approaches; RTL direction is required for Hebrew
    seg = blla.segment(binarized, text_direction="horizontal-rl")

    # Recognition — rpred returns an iterable of ocr_record objects
    records = list(rpred.rpred(model, binarized, seg))

    # Filter empty predictions and join lines
    lines = [r.prediction for r in records if r.prediction.strip()]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Public async interface                                                       #
# --------------------------------------------------------------------------- #

async def transcribe_with_kraken(
    model_path: str,
    image_path: str,
    timeout: Optional[float] = 120.0,
) -> Optional[str]:
    """Run Kraken transcription in a thread executor (non-blocking).

    Args:
        model_path: Path to the .mlmodel file (MiDRASH_Gen_01.mlmodel or similar).
        image_path: Path to the manuscript image.
        timeout: Maximum seconds to wait; returns None on timeout.

    Returns:
        Transcribed text, or None on failure.
    """
    loop = asyncio.get_event_loop()
    try:
        text = await asyncio.wait_for(
            loop.run_in_executor(None, _transcribe_sync, model_path, image_path),
            timeout=timeout,
        )
        return text

    except asyncio.TimeoutError:
        logger.error(f"Kraken timed out after {timeout}s on {image_path}")
        return None

    except Exception as exc:
        logger.error(
            f"Kraken failed on {image_path}: {type(exc).__name__}: {str(exc)[:300]}"
        )
        return None