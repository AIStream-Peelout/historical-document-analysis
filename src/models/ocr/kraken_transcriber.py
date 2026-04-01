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

    import warnings
    from kraken.lib import models  # lazy import

    logger.info(f"Loading Kraken model: {model_path}")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You will not be able to run predict",
            category=RuntimeWarning,
        )
        _cached_model = models.load_any(model_path)
    _cached_model_path = model_path
    logger.info("Kraken model loaded and cached.")


# --------------------------------------------------------------------------- #
# Synchronous core (runs in a thread executor)                                 #
# --------------------------------------------------------------------------- #

def _segment_and_recognize(model, im, use_binarization: bool):
    """Run one segmentation + recognition pass and return (lines, failed_polygon_count).

    The 'Polygonizer failed' / TopologyException messages come from Kraken's
    internal polygon inflation step (shapely/GEOS) when baselines on complex
    multi-column layouts (Talmud: Gemara center + Rashi/Tosafot margins) produce
    self-intersecting geometry.  They are logged at WARNING level by kraken.blla
    and are non-fatal — affected lines are dropped from the output.

    We silence the kraken.blla logger during segmentation and count failures via
    a custom handler instead, so the eval output stays readable while we still
    get the failure rate as a diagnostic metric.
    """
    import logging as _logging
    from kraken import blla, binarization, rpred

    # ── Count polygonizer failures without printing them ──────────────────────
    class _PolyFailCounter(_logging.Handler):
        def __init__(self):
            super().__init__()
            self.count = 0

        def emit(self, record):
            if "Polygonizer failed" in record.getMessage() or \
               "TopologyException" in record.getMessage() or \
               "Self-intersection" in record.getMessage() or \
               "side location conflict" in record.getMessage():
                self.count += 1

    blla_logger = _logging.getLogger("kraken.blla")
    counter = _PolyFailCounter()
    # Temporarily redirect blla warnings through our counter only
    original_level = blla_logger.level
    original_propagate = blla_logger.propagate
    blla_logger.setLevel(_logging.WARNING)
    blla_logger.propagate = False
    blla_logger.addHandler(counter)

    try:
        target_im = binarization.nlbin(im) if use_binarization else im
        seg = blla.segment(target_im, text_direction="horizontal-rl")
        records = list(rpred.rpred(model, target_im, seg))
    finally:
        blla_logger.removeHandler(counter)
        blla_logger.propagate = original_propagate
        blla_logger.setLevel(original_level)

    lines = [r.prediction for r in records if r.prediction.strip()]
    return lines, counter.count


def _transcribe_sync(model_path: str, image_path: str) -> str:
    """Run the full Kraken pipeline synchronously.

    Strategy:
      1. Try with nlbin binarization (standard path).
      2. If >30% of detected lines fail polygon inflation, retry on the raw
         image — complex Talmud layouts sometimes produce worse geometry after
         binarization sharpens edges into tight curves.
      3. Return whichever pass produced more output lines.

    Polygonizer failures are counted and logged as a single summary line rather
    than flooding the console with per-line geometry exceptions.
    """
    import warnings
    from kraken.lib import models
    from PIL import Image

    # ── Load model (use cache if available) ───────────────────────────────────
    global _cached_model, _cached_model_path
    if _cached_model_path == model_path and _cached_model is not None:
        model = _cached_model
    else:
        logger.warning("Kraken model not pre-cached — loading on-demand (slow).")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You will not be able to run predict",
                category=RuntimeWarning,
            )
            model = models.load_any(model_path)

    im = Image.open(image_path)

    # ── Pass 1: with binarization ─────────────────────────────────────────────
    lines_bin, failures_bin = _segment_and_recognize(model, im, use_binarization=True)
    total_bin = len(lines_bin) + failures_bin
    fail_rate_bin = failures_bin / max(total_bin, 1)

    logger.info(
        f"  Kraken binarized: {len(lines_bin)} lines kept, "
        f"{failures_bin} polygon failures ({fail_rate_bin:.0%})"
    )

    # ── Pass 2: raw image fallback if failure rate is high ───────────────────
    # Threshold: if more than 30% of baselines failed polygon inflation,
    # retry without binarization. nlbin can sharpen curves into geometry that
    # GEOS can't polygonize; the raw image sometimes produces cleaner baselines.
    FALLBACK_THRESHOLD = 0.30
    if fail_rate_bin > FALLBACK_THRESHOLD:
        logger.info(
            f"  Polygon failure rate {fail_rate_bin:.0%} > {FALLBACK_THRESHOLD:.0%} — "
            f"retrying on raw image without binarization"
        )
        lines_raw, failures_raw = _segment_and_recognize(model, im, use_binarization=False)
        logger.info(
            f"  Kraken raw: {len(lines_raw)} lines kept, "
            f"{failures_raw} polygon failures"
        )
        # Use whichever pass recovered more lines
        if len(lines_raw) > len(lines_bin):
            logger.info("  Using raw-image pass (more lines recovered)")
            return "\n".join(lines_raw)
        else:
            logger.info("  Binarized pass still better — keeping it")

    return "\n".join(lines_bin)


# --------------------------------------------------------------------------- #
# Public async interface                                                       #
# --------------------------------------------------------------------------- #

async def transcribe_with_kraken(
    model_path: str,
    image_path: str,
    timeout: Optional[float] = 120.0,
) -> Optional[str]:
    """Run Kraken transcription in a thread executor (non-blocking).

    Returns the transcribed text, or None on hard failure.
    Polygon failure diagnostics are logged at INFO level as a single summary
    line rather than the raw per-line geometry exceptions from blla.
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