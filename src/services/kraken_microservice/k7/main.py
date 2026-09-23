"""Kraken 7 transcription microservice (test build, same API as the kraken 4 service).

Endpoints mirror ``../main.py``: ``/health``, ``/preload``, ``/transcribe`` and
``/transcribe_lines``. The segmenter is chosen once per container through
environment variables so an A/B is a container restart, not a client change:

* ``KRAKEN_SEGMENTER`` — ``default`` (kraken's bundled blla.mlmodel, the same
  weights kraken 4.3.13 ships), or a path to a blla ``.mlmodel`` or an Orli
  ``.safetensors`` file.
* ``KRAKEN_THREADS`` — torch intra-op threads; set it to the container's
  ``--cpus`` value (torch otherwise sees every host core and oversubscribes).
* ``KRAKEN_SEG_INPUT`` — ``binarized`` (default: blla segments the nlbin image,
  as the kraken 4 service did) or ``rgb`` (segment the original colour page, the
  input ``ketos segtrain`` fine-tunes on). Recognition always follows the
  two-pass binarization policy. Orli always segments the original image.

Segmentation uses kraken 7's task API (the only path that loads the 2026 blla
and Orli). Recognition uses ``rpred.rpred``, which gives output identical to the
task API at batch size 1 and is the fastest path on CPU.
"""
import logging
import os
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image
from pydantic import BaseModel

from kraken import binarization, rpred
from kraken.configs import SegmentationInferenceConfig
from kraken.lib import models
from kraken.models import load_models
from kraken.tasks.segmentation import SegmentationTaskModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEGMENTER = os.getenv("KRAKEN_SEGMENTER", "default")
THREADS = int(os.getenv("KRAKEN_THREADS", "6"))
SEG_INPUT = os.getenv("KRAKEN_SEG_INPUT", "binarized")
IS_ORLI = "orli" in Path(SEGMENTER).name.lower()
FALLBACK_THRESHOLD = 0.30
_POLY_FAIL_MARKERS = ("Polygonizer failed", "TopologyException", "Self-intersection",
                      "side location conflict")

torch.set_num_threads(THREADS)

app = FastAPI(title="Kraken 7 Transcription Microservice")

_cached_model = None
_cached_model_path: Optional[str] = None


def _load_segmenter(spec: str) -> SegmentationTaskModel:
    """Load the page segmenter named by ``KRAKEN_SEGMENTER``.

    :param spec: ``default`` or a path to a blla ``.mlmodel`` / Orli ``.safetensors`` file.
    :type spec: str
    :return: A segmentation task wrapping the model(s).
    :rtype: SegmentationTaskModel
    """
    if spec == "default":
        return SegmentationTaskModel.load_model()
    return SegmentationTaskModel(load_models(spec, tasks=["segmentation"]))


def _seg_config() -> SegmentationInferenceConfig:
    """Build the segmentation inference config for the active segmenter.

    Orli only works in bf16 (other precisions cause runaway generation) and
    emits baselines only, so it is asked to polygonize with kraken's
    polygonizer; blla runs in fp32.

    :return: Inference config for :meth:`SegmentationTaskModel.predict`.
    :rtype: SegmentationInferenceConfig
    """
    common = dict(text_direction="horizontal-rl", accelerator="cpu", device=1)
    if IS_ORLI:
        from orli.configs import OrliSegmentationInferenceConfig
        return OrliSegmentationInferenceConfig(precision="bf16-true", polygonize=True, **common)
    return SegmentationInferenceConfig(precision="32-true", **common)


logger.info(f"Loading segmenter {SEGMENTER} (orli={IS_ORLI}, threads={THREADS})")
_segmenter = _load_segmenter(SEGMENTER)
_seg_cfg = _seg_config()


class PreloadRequest(BaseModel):
    model_path: str


class TranscriptionResponse(BaseModel):
    text: str
    polygon_failures: int
    used_binarization: bool
    model_path: str


class LinesResponse(BaseModel):
    lines: list  # list of {text, bbox [x0, y0, x1, y1], confidence}
    polygon_failures: int
    used_binarization: bool
    model_path: str
    image_size: list  # [width, height]


class _PolyFailCounter(logging.Handler):
    """Counts kraken polygonizer failures logged during one page."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        """Increment the counter for polygonizer failure messages.

        :param record: A log record from the ``kraken`` logger tree.
        :type record: logging.LogRecord
        """
        if any(m in record.getMessage() for m in _POLY_FAIL_MARKERS):
            self.count += 1


def _resolve_model_path(model_path: str) -> str:
    """Map a client (host) model path onto the ``/app/models`` mount.

    :param model_path: Path as sent by the client.
    :type model_path: str
    :return: A path that exists inside the container.
    :rtype: str
    :raises HTTPException: If neither the path nor its basename under ``/app/models`` exists.
    """
    if os.path.exists(model_path):
        return model_path
    alt_path = os.path.join("/app/models", os.path.basename(model_path))
    if os.path.exists(alt_path):
        return alt_path
    raise HTTPException(status_code=400, detail=f"Model path {model_path} not found in container.")


def _get_model(model_path: str):
    """Return the cached recognition model, loading it if the path changed.

    :param model_path: Resolved in-container model path.
    :type model_path: str
    :return: A kraken ``TorchSeqRecognizer``.
    """
    global _cached_model, _cached_model_path
    if _cached_model_path != model_path or _cached_model is None:
        logger.info(f"Loading Kraken recognition model: {model_path}")
        _cached_model = models.load_any(model_path)
        _cached_model_path = model_path
    return _cached_model


def _segment_and_recognize(model, im: Image.Image, use_binarization: bool):
    """Segment a page and recognise every line.

    blla segments the (optionally binarized) image, or the original colour image
    with ``KRAKEN_SEG_INPUT=rgb``. Orli is trained on colour scans, so it always
    segments the original image; recognition still runs on the binarized image
    when ``use_binarization`` is set.
    nlbin does not resize, so geometry is valid on the original image.

    :param model: Recognition model.
    :param im: RGB page image.
    :type im: Image.Image
    :param use_binarization: Binarize with nlbin before segmentation/recognition.
    :type use_binarization: bool
    :return: (ocr records, polygonizer failure count).
    :rtype: tuple
    """
    kraken_logger = logging.getLogger("kraken")
    counter = _PolyFailCounter()
    kraken_logger.addHandler(counter)
    try:
        target_im = binarization.nlbin(im) if use_binarization else im
        seg = _segmenter.predict(im if (IS_ORLI or SEG_INPUT == "rgb") else target_im, _seg_cfg)
        # Orli lines whose polygonization failed carry no boundary; rpred cannot extract them.
        # Orli's polygonizer yields numpy arrays; normalise to point lists.
        for ln in seg.lines:
            if ln.boundary is not None:
                ln.boundary = [tuple(map(int, p)) for p in ln.boundary]
        kept = [ln for ln in seg.lines if ln.boundary]
        counter.count += len(seg.lines) - len(kept)
        seg.lines = kept
        records = list(rpred.rpred(model, target_im, seg))
    finally:
        kraken_logger.removeHandler(counter)
    return records, counter.count


def _record_bbox(record) -> Optional[list]:
    """Bounding box [x0, y0, x1, y1] from a kraken ocr_record's geometry.

    Prefers the polygon boundary; falls back to the baseline extent with a
    synthetic height when no boundary is available.

    :param record: A kraken ``ocr_record``.
    :return: Integer-able bbox, or None when the record has no geometry.
    :rtype: Optional[list]
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


def _line_infos(records) -> list:
    """Per-line dicts (text, bbox, mean confidence) for non-empty records.

    :param records: kraken ``ocr_record`` list.
    :type records: list
    :return: Line dicts in reading order.
    :rtype: list
    """
    infos = []
    for r in records:
        if not r.prediction.strip():
            continue
        bbox = _record_bbox(r)
        if bbox is None:
            continue
        confs = getattr(r, "confidences", None) or []
        infos.append({"text": r.prediction, "bbox": [int(v) for v in bbox],
                      "confidence": (sum(confs) / len(confs)) if confs else None})
    return infos


def _two_pass(model, im: Image.Image):
    """Binarized pass, with a raw-image retry when polygonization mostly fails.

    Same policy as the kraken 4 service: keep the raw pass only if its failure
    rate triggered the retry and it yields more non-empty lines.

    :param model: Recognition model.
    :param im: RGB page image.
    :type im: Image.Image
    :return: (line infos, polygon failures, used_binarization).
    :rtype: tuple
    """
    recs_bin, fail_bin = _segment_and_recognize(model, im, use_binarization=True)
    infos_bin = _line_infos(recs_bin)
    rate = fail_bin / max(len(infos_bin) + fail_bin, 1)
    logger.info(f"binarized: {len(infos_bin)} lines kept, {fail_bin} polygon failures ({rate:.0%})")
    if rate > FALLBACK_THRESHOLD:
        recs_raw, fail_raw = _segment_and_recognize(model, im, use_binarization=False)
        infos_raw = _line_infos(recs_raw)
        logger.info(f"raw: {len(infos_raw)} lines kept, {fail_raw} polygon failures")
        if len(infos_raw) > len(infos_bin):
            return infos_raw, fail_raw, False
    return infos_bin, fail_bin, True


def _open_rgb(upload: UploadFile) -> Image.Image:
    """Open an uploaded image as RGB.

    :param upload: Multipart image upload.
    :type upload: UploadFile
    :return: RGB PIL image.
    :rtype: Image.Image
    """
    im = Image.open(upload.file)
    return im if im.mode == "RGB" else im.convert("RGB")


@app.post("/preload")
async def preload_model(req: PreloadRequest):
    """Load and cache a recognition model.

    :param req: Request carrying the model path.
    :type req: PreloadRequest
    :return: Status dict.
    :rtype: dict
    """
    model_path = _resolve_model_path(req.model_path)
    already = _cached_model_path == model_path and _cached_model is not None
    _get_model(model_path)
    return {"status": "already loaded" if already else "loaded", "model_path": model_path}


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(model_path: str = Form(...), image: UploadFile = File(...)):
    """Transcribe a page to flat text in segmenter reading order.

    :param model_path: Recognition model path (host or container).
    :type model_path: str
    :param image: Page image upload.
    :type image: UploadFile
    :return: Joined text plus diagnostics.
    :rtype: TranscriptionResponse
    """
    model_path = _resolve_model_path(model_path)
    model = _get_model(model_path)
    infos, failures, used_bin = _two_pass(model, _open_rgb(image))
    return TranscriptionResponse(text="\n".join(i["text"] for i in infos), polygon_failures=failures,
                                 used_binarization=used_bin, model_path=model_path)


@app.post("/transcribe_lines", response_model=LinesResponse)
async def transcribe_lines(model_path: str = Form(...), image: UploadFile = File(...)):
    """Per-line transcription with geometry, for line-wise pipelines.

    :param model_path: Recognition model path (host or container).
    :type model_path: str
    :param image: Page image upload.
    :type image: UploadFile
    :return: Line dicts plus diagnostics and image size.
    :rtype: LinesResponse
    """
    model_path = _resolve_model_path(model_path)
    model = _get_model(model_path)
    im = _open_rgb(image)
    infos, failures, used_bin = _two_pass(model, im)
    return LinesResponse(lines=infos, polygon_failures=failures, used_binarization=used_bin,
                         model_path=model_path, image_size=[im.width, im.height])


@app.get("/health")
def health():
    """Liveness probe that also reports the active segmenter.

    :return: Status dict.
    :rtype: dict
    """
    return {"status": "ok", "kraken": "7.0.3", "segmenter": SEGMENTER, "seg_input": SEG_INPUT, "threads": THREADS}
