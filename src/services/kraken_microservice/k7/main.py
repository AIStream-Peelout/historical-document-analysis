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
* ``KRAKEN_MULTISCALE=1`` — also read the page with its writing shrunk (views at
  scale 0.5, then 0.33 if 0.5 wins; see ``views.py``) and keep the view whose
  lines read best (Σ Hebrew letters × mean confidence). Large-script fragments
  otherwise reach the fixed 1800 px model input with text far larger than the
  segmenter was trained on.
* ``KRAKEN_ORIENT=1`` — when the upright full-scale read looks like fragments, also
  try the page rotated 90° and 270° (a rotation must score ≥ 2× upright and read as
  text itself); the scale search then runs in the chosen orientation. Boxes are
  always returned in the original page's coordinates (plus ``view_bbox`` in the
  rotated frame for geometry re-ordering); the chosen view is reported in ``view``.

Segmentation uses kraken 7's task API (the only path that loads the 2026 blla
and Orli). Recognition uses ``rpred.rpred``, which gives output identical to the
task API at batch size 1 and is the fastest path on CPU.
"""
import ctypes
import gc
import logging
import os
import re
import statistics
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

from views import bbox_to_original, rotate, seg_view

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEGMENTER = os.getenv("KRAKEN_SEGMENTER", "default")
THREADS = int(os.getenv("KRAKEN_THREADS", "6"))
SEG_INPUT = os.getenv("KRAKEN_SEG_INPUT", "binarized")
IS_ORLI = "orli" in Path(SEGMENTER).name.lower()
MULTISCALE = os.getenv("KRAKEN_MULTISCALE", "0") == "1" and not IS_ORLI   # Orli resizes to a fixed W×H: padding
                                                                          # one axis would distort, not shrink
ORIENT = os.getenv("KRAKEN_ORIENT", "0") == "1"
WEAK_LONG_LINE = 8           # a read is "fragments" when under WEAK_LONG_SHARE of its letters sit in lines with
WEAK_LONG_SHARE = 0.3        # at least WEAK_LONG_LINE letters (letter-weighted: 1-2-letter noise lines don't count)...
WEAK_TOTAL_LETTERS = 40      # ...or it has fewer letters in total
ROTATION_MARGIN = 2.0        # a rotated read must score this many times the upright one (and not be fragments itself)
_HEB = re.compile(r"[\u05d0-\u05ea]")
FALLBACK_THRESHOLD = 0.30
_POLY_FAIL_MARKERS = ("Polygonizer failed", "TopologyException", "Self-intersection",
                      "side location conflict", "Extracting line failed")

torch.set_num_threads(THREADS)

app = FastAPI(title="Kraken 7 Transcription Microservice")

try:
    _LIBC = ctypes.CDLL("libc.so.6")
except OSError:          # not glibc (e.g. macOS dev runs): nothing to trim
    _LIBC = None


def _release_memory() -> None:
    """Return freed memory to the OS after a request.

    Pages of very different sizes fragment glibc's heap; without this the
    service's resident memory grows request after request until the container
    is OOM-killed (seen after ~360 multi-view requests at 8 GiB, and on prod).
    Works together with the ``MALLOC_*`` settings in the Dockerfile.
    """
    gc.collect()
    if _LIBC is not None:
        _LIBC.malloc_trim(0)

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
    view: Optional[dict] = None  # {rotation, scale, scores} when multi-view reading is on


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


def _segment_and_recognize(model, rgb: Image.Image, bim: Optional[Image.Image], use_binarization: bool,
                           scale: float = 1.0):
    """Segment one view of a page and recognise every line at full resolution.

    blla segments the binarized page, or the colour page with ``KRAKEN_SEG_INPUT=rgb``
    (Orli always segments the colour page). At ``scale == 1`` the segmenter gets
    the full-resolution image exactly as before; at ``scale < 1`` it gets the
    text-shrinking view of :func:`views.seg_view` and its geometry is mapped back
    to full resolution (lines found in the padding are dropped). Recognition
    always runs on the full-resolution (binarized) page.

    :param model: Recognition model.
    :param rgb: RGB page (already rotated for the view).
    :type rgb: Image.Image
    :param bim: nlbin-binarized page in the same frame, or None when ``use_binarization`` is False.
    :type bim: Optional[Image.Image]
    :param use_binarization: Recognise (and, unless rgb input, segment) the binarized page.
    :type use_binarization: bool
    :param scale: Text shrink factor of the segmentation view.
    :type scale: float
    :return: (ocr records, polygonizer failure count).
    :rtype: tuple
    """
    kraken_logger = logging.getLogger("kraken")
    counter = _PolyFailCounter()
    kraken_logger.addHandler(counter)
    try:
        target_im = bim if use_binarization else rgb
        seg_src = rgb if (IS_ORLI or SEG_INPUT == "rgb") else target_im
        if scale >= 1.0:
            seg = _segmenter.predict(seg_src, _seg_cfg)
        else:
            view, f, content_h = seg_view(seg_src, scale)
            seg = _segmenter.predict(view, _seg_cfg)
            kept_lines = []
            w_full, h_full = seg_src.size
            # blla bounds polygons by the padded view, so a bottom line's lower seam can reach into the padding:
            # clamp every mapped point to the full-resolution page or rpred cannot extract the line
            to_full = lambda p: (min(max(int(p[0] / f), 0), w_full - 1), min(max(int(p[1] / f), 0), h_full - 1))  # noqa: E731
            for ln in seg.lines:
                if min(p[1] for p in ln.baseline) >= content_h:
                    continue                                   # a "line" in the padding
                ln.baseline = [to_full(p) for p in ln.baseline]
                if ln.boundary is not None:
                    ln.boundary = [to_full(p) for p in ln.boundary]
                kept_lines.append(ln)
            seg.lines = kept_lines
        # Orli's polygonizer yields numpy arrays; normalise to point lists. Lines without a boundary cannot
        # be extracted by rpred.
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


def _two_pass(model, rgb: Image.Image, bim: Image.Image, scale: float = 1.0):
    """Binarized pass, with a raw-image retry when polygonization mostly fails (one view).

    Same policy as the kraken 4 service: keep the raw pass only if its failure
    rate triggered the retry and it yields more non-empty lines.

    :param model: Recognition model.
    :param rgb: RGB page in the view's frame.
    :type rgb: Image.Image
    :param bim: Binarized page in the same frame.
    :type bim: Image.Image
    :param scale: Text shrink factor of the segmentation view.
    :type scale: float
    :return: (line infos, polygon failures, used_binarization).
    :rtype: tuple
    """
    recs_bin, fail_bin = _segment_and_recognize(model, rgb, bim, True, scale)
    infos_bin = _line_infos(recs_bin)
    rate = fail_bin / max(len(infos_bin) + fail_bin, 1)
    logger.info(f"binarized (scale {scale}): {len(infos_bin)} lines kept, {fail_bin} polygon failures ({rate:.0%})")
    if rate > FALLBACK_THRESHOLD:
        recs_raw, fail_raw = _segment_and_recognize(model, rgb, None, False, scale)
        infos_raw = _line_infos(recs_raw)
        logger.info(f"raw (scale {scale}): {len(infos_raw)} lines kept, {fail_raw} polygon failures")
        if len(infos_raw) > len(infos_bin):
            return infos_raw, fail_raw, False
    return infos_bin, fail_bin, True


def _read_score(infos: list) -> float:
    """How much text a read recovered: Σ Hebrew letters × mean line confidence.

    :param infos: Line dicts from :func:`_line_infos`.
    :type infos: list
    :return: Score.
    :rtype: float
    """
    return sum(len(_HEB.findall(i["text"])) * (i["confidence"] or 0.0) for i in infos)


def _is_fragmentary(infos: list) -> bool:
    """Whether a read looks like scraps (rotated page, wrong scale) rather than lines of text.

    :param infos: Line dicts.
    :type infos: list
    :return: True when the page has fewer than :data:`WEAK_TOTAL_LETTERS` letters or less than
        :data:`WEAK_LONG_SHARE` of them sit in lines of at least :data:`WEAK_LONG_LINE` letters.
    :rtype: bool
    """
    letters = [len(_HEB.findall(i["text"])) for i in infos]
    total = sum(letters)
    if total < WEAK_TOTAL_LETTERS:
        return True
    return sum(n for n in letters if n >= WEAK_LONG_LINE) / total < WEAK_LONG_SHARE


def _read_page(model, im: Image.Image):
    """Read a page, searching scales / orientations when enabled, and return original-frame lines.

    :param model: Recognition model.
    :param im: RGB page.
    :type im: Image.Image
    :return: (line infos in original coordinates, polygon failures, used_binarization, view dict or None).
    :rtype: tuple
    """
    bim = binarization.nlbin(im)
    reads = {}

    def run(rot: int, scale: float) -> None:
        rgb_v, bim_v = rotate(im, rot), rotate(bim, rot)
        infos, fails, used_bin = _two_pass(model, rgb_v, bim_v, scale)
        reads[(rot, scale)] = (infos, fails, used_bin, _read_score(infos))

    run(0, 1.0)
    if not (MULTISCALE or ORIENT):
        infos, fails, used_bin, _ = reads[(0, 1.0)]
        return infos, fails, used_bin, None
    rot = 0
    if ORIENT and _is_fragmentary(reads[(0, 1.0)][0]):
        # orientation first, at full scale; a rotation must win clearly and read as text itself
        for r in (90, 270):
            run(r, 1.0)
        cand = max((90, 270), key=lambda r: reads[(r, 1.0)][3])
        if (reads[(cand, 1.0)][3] >= ROTATION_MARGIN * reads[(0, 1.0)][3]
                and not _is_fragmentary(reads[(cand, 1.0)][0])):
            rot = cand
    if MULTISCALE:
        run(rot, 0.5)
        if reads[(rot, 0.5)][3] > reads[(rot, 1.0)][3]:
            run(rot, 0.33)
    scale = max((k for k in reads if k[0] == rot), key=lambda k: reads[k][3])[1]
    infos, fails, used_bin, _ = reads[(rot, scale)]
    if rot:
        # original-frame boxes for the pipeline; view-frame boxes for consumers that re-order by geometry
        infos = [{**i, "bbox": bbox_to_original(i["bbox"], rot, im.width, im.height), "view_bbox": i["bbox"]}
                 for i in infos]
    view = {"rotation": rot, "scale": scale, "scores": {f"{r}/{s}": round(v[3], 1) for (r, s), v in reads.items()}}
    logger.info(f"view chosen: rotation {rot}, scale {scale}; scores {view['scores']}")
    return infos, fails, used_bin, view


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
    try:
        infos, failures, used_bin, _ = _read_page(model, _open_rgb(image))
    finally:
        _release_memory()
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
    try:
        infos, failures, used_bin, view = _read_page(model, im)
    finally:
        _release_memory()
    return LinesResponse(lines=infos, polygon_failures=failures, used_binarization=used_bin,
                         model_path=model_path, image_size=[im.width, im.height], view=view)


@app.get("/health")
def health():
    """Liveness probe that also reports the active segmenter.

    :return: Status dict.
    :rtype: dict
    """
    return {"status": "ok", "kraken": "7.0.3", "segmenter": SEGMENTER, "seg_input": SEG_INPUT,
            "multiscale": MULTISCALE, "orient": ORIENT, "threads": THREADS}
