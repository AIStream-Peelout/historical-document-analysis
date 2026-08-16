import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

# Suppress kraken warnings before importing it
import warnings
warnings.filterwarnings("ignore", message="You will not be able to run predict", category=RuntimeWarning)

from kraken.lib import models
from kraken import blla, binarization, rpred
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Kraken Transcription Microservice")

# Module-level model cache
_cached_model = None
_cached_model_path: Optional[str] = None


class PreloadRequest(BaseModel):
    model_path: str


class TranscriptionResponse(BaseModel):
    text: str
    polygon_failures: int
    used_binarization: bool
    model_path: str


class LineInfo(BaseModel):
    text: str
    bbox: list  # [x0, y0, x1, y1] in original-image pixel coordinates
    confidence: Optional[float] = None


class LinesResponse(BaseModel):
    lines: list  # list[LineInfo]
    polygon_failures: int
    used_binarization: bool
    model_path: str
    image_size: list  # [width, height]


@app.post("/preload")
async def preload_model(req: PreloadRequest):
    global _cached_model, _cached_model_path

    model_path = req.model_path

    # Standardize the path to inside the container assuming /app/models is volume mounted
    # If the user passes an absolute path like /Users/isaac/.../models/MiDRASH_Gen_01.mlmodel,
    # the frontend might map it directly, or maybe just pass the filename. Let's try to handle both.
    
    # If it is inside /app/models, then use it as is, however the client probably sends its local machine path!
    # Let's see if we can resolve the model path. We'll try loading it directly.
    if not os.path.exists(model_path):
        # Maybe they provided a host path and we need to map it?
        # A simpler way is to just assume the file is mounted exactly at the path provided OR at /app/models
        # For safety, let's also check if it exists in /app/models/
        filename = os.path.basename(model_path)
        alt_path = os.path.join("/app/models", filename)
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise HTTPException(status_code=400, detail=f"Model path {model_path} not found in container.")

    if _cached_model_path == model_path and _cached_model is not None:
        return {"status": "already loaded", "model_path": model_path}

    logger.info(f"Loading Kraken model: {model_path}")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="You will not be able to run predict", category=RuntimeWarning)
            _cached_model = models.load_any(model_path)
        _cached_model_path = model_path
        logger.info("Kraken model loaded and cached.")
        return {"status": "loaded", "model_path": model_path}
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


def _segment_and_recognize(model, im: Image.Image, use_binarization: bool):
    class _PolyFailCounter(logging.Handler):
        def __init__(self):
            super().__init__()
            self.count = 0

        def emit(self, record):
            msg = record.getMessage()
            if "Polygonizer failed" in msg or \
               "TopologyException" in msg or \
               "Self-intersection" in msg or \
               "side location conflict" in msg:
                self.count += 1

    blla_logger = logging.getLogger("kraken.blla")
    counter = _PolyFailCounter()
    original_level = blla_logger.level
    original_propagate = blla_logger.propagate
    blla_logger.setLevel(logging.WARNING)
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


def _record_bbox(record):
    """Bounding box [x0, y0, x1, y1] from a kraken ocr_record's geometry.

    Prefers the polygon boundary; falls back to the baseline extent with a
    synthetic height when no boundary is available.
    """
    pts = getattr(record, "boundary", None) or []
    if not pts:
        base = getattr(record, "baseline", None) or []
        if not base:
            return None
        xs = [p[0] for p in base]
        ys = [p[1] for p in base]
        pad = 20  # synthetic half-height around the baseline
        return [min(xs), max(min(ys) - pad, 0), max(xs), max(ys) + pad]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def _segment_and_recognize_lines(model, im: Image.Image, use_binarization: bool):
    """Like _segment_and_recognize but returns per-line dicts with geometry.

    Segmentation runs on the (optionally binarized) image, which nlbin does
    not resize, so bboxes are valid on the original image coordinates.
    """
    class _PolyFailCounter(logging.Handler):
        def __init__(self):
            super().__init__()
            self.count = 0

        def emit(self, record):
            msg = record.getMessage()
            if "Polygonizer failed" in msg or \
               "TopologyException" in msg or \
               "Self-intersection" in msg or \
               "side location conflict" in msg:
                self.count += 1

    blla_logger = logging.getLogger("kraken.blla")
    counter = _PolyFailCounter()
    original_level = blla_logger.level
    original_propagate = blla_logger.propagate
    blla_logger.setLevel(logging.WARNING)
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

    infos = []
    for r in records:
        if not r.prediction.strip():
            continue
        bbox = _record_bbox(r)
        if bbox is None:
            continue
        confs = getattr(r, "confidences", None) or []
        infos.append({
            "text": r.prediction,
            "bbox": [int(v) for v in bbox],
            "confidence": (sum(confs) / len(confs)) if confs else None,
        })
    return infos, counter.count


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    model_path: str = Form(...),
    image: UploadFile = File(...)
):
    global _cached_model, _cached_model_path

    # Try mapping the path as we did in preload
    if not os.path.exists(model_path):
        filename = os.path.basename(model_path)
        alt_path = os.path.join("/app/models", filename)
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise HTTPException(status_code=400, detail=f"Model path {model_path} not found in container.")

    if _cached_model_path == model_path and _cached_model is not None:
        model = _cached_model
    else:
        logger.warning("Kraken model not pre-cached — loading on-demand (slow).")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="You will not be able to run predict", category=RuntimeWarning)
                model = models.load_any(model_path)
            _cached_model = model
            _cached_model_path = model_path
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    try:
        im = Image.open(image.file)
        if im.mode != "RGB":
            im = im.convert("RGB")

        # Pass 1: binarization
        lines_bin, failures_bin = _segment_and_recognize(model, im, use_binarization=True)
        total_bin = len(lines_bin) + failures_bin
        fail_rate_bin = failures_bin / max(total_bin, 1)

        logger.info(f"Kraken binarized: {len(lines_bin)} lines kept, {failures_bin} polygon failures ({fail_rate_bin:.0%})")

        FALLBACK_THRESHOLD = 0.30
        if fail_rate_bin > FALLBACK_THRESHOLD:
            logger.info(f"Polygon failure rate {fail_rate_bin:.0%} > {FALLBACK_THRESHOLD:.0%} retry raw")
            lines_raw, failures_raw = _segment_and_recognize(model, im, use_binarization=False)
            logger.info(f"Kraken raw: {len(lines_raw)} lines kept, {failures_raw} polygon failures")
            if len(lines_raw) > len(lines_bin):
                logger.info("Using raw-image pass")
                return TranscriptionResponse(
                    text="\n".join(lines_raw),
                    polygon_failures=failures_raw,
                    used_binarization=False,
                    model_path=model_path
                )
            else:
                logger.info("Binarized pass still better")
                
        return TranscriptionResponse(
            text="\n".join(lines_bin),
            polygon_failures=failures_bin,
            used_binarization=True,
            model_path=model_path
        )

    except Exception as e:
        logger.error(f"Failed transcription processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe_lines", response_model=LinesResponse)
async def transcribe_lines(
    model_path: str = Form(...),
    image: UploadFile = File(...)
):
    """Per-line transcription with geometry, for line-wise VLM pipelines.

    Mirrors /transcribe's two-pass binarization fallback, but returns each
    kept line's text, bbox and mean recognition confidence instead of the
    joined text.
    """
    global _cached_model, _cached_model_path

    if not os.path.exists(model_path):
        filename = os.path.basename(model_path)
        alt_path = os.path.join("/app/models", filename)
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise HTTPException(status_code=400, detail=f"Model path {model_path} not found in container.")

    if _cached_model_path == model_path and _cached_model is not None:
        model = _cached_model
    else:
        logger.warning("Kraken model not pre-cached — loading on-demand (slow).")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="You will not be able to run predict", category=RuntimeWarning)
                model = models.load_any(model_path)
            _cached_model = model
            _cached_model_path = model_path
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    try:
        im = Image.open(image.file)
        if im.mode != "RGB":
            im = im.convert("RGB")

        infos_bin, failures_bin = _segment_and_recognize_lines(model, im, use_binarization=True)
        total_bin = len(infos_bin) + failures_bin
        fail_rate_bin = failures_bin / max(total_bin, 1)
        logger.info(f"Kraken lines binarized: {len(infos_bin)} kept, {failures_bin} polygon failures ({fail_rate_bin:.0%})")

        FALLBACK_THRESHOLD = 0.30
        if fail_rate_bin > FALLBACK_THRESHOLD:
            infos_raw, failures_raw = _segment_and_recognize_lines(model, im, use_binarization=False)
            logger.info(f"Kraken lines raw: {len(infos_raw)} kept, {failures_raw} polygon failures")
            if len(infos_raw) > len(infos_bin):
                return LinesResponse(
                    lines=infos_raw, polygon_failures=failures_raw,
                    used_binarization=False, model_path=model_path,
                    image_size=[im.width, im.height],
                )

        return LinesResponse(
            lines=infos_bin, polygon_failures=failures_bin,
            used_binarization=True, model_path=model_path,
            image_size=[im.width, im.height],
        )

    except Exception as e:
        logger.error(f"Failed line transcription processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
