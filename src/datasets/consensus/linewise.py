"""Line-wise VLM inference: kraken segmentation + per-line v18b reads.

Motivation (P2 of v1.9): the whole-page VLM under-reads long pages (benchmark
coverage 90% vs kraken 97%; user-observed early stopping on strong docs like
Paris AIU III.A.66) and is markedly stronger on small regions.  This module
reads every kraken-segmented line individually and reassembles the page in
kraken's reading order, so page length can no longer truncate the read.

Two modes:

* ``blind`` — the VLM sees only the line crop.  Output is pure VLM text, so
  benchmark comparisons against whole-page mode are clean, and per-line
  cross-family agreement (vs the kraken line read) stays valid evidence.
* ``adversarial`` — the VLM sees kraken's read of the same line and must
  confirm or correct it strictly from the ink (never deferential — anchoring
  risk is the known failure mode, per the v1.9 handoff).

Per-line details (bbox, kraken text, VLM text, kraken confidence) are saved
as a JSONL sidecar for reconciliation experiments.
"""

import asyncio
import io
import json
import tempfile
from pathlib import Path

import aiohttp
from PIL import Image

from src.datasets.consensus.consensus_gate import _series_from_doc_id
from src.models.ocr.lms_transcriber import transcribe_with_lm_studio

KRAKEN_URL = "http://localhost:8002"
KRAKEN_MODEL = "/app/models/MiDRASH_Gen_01.mlmodel"
LINE_MAX_TOKENS = 400
PAD_X_FRAC = 0.02
PAD_Y_FRAC = 0.35


def _line_prompt(doc_id: str) -> str:
    """Blind per-line transcription prompt.

    Keeps the calibrated whole-page prompt's framing and rules at line scale.

    :param doc_id: Document identifier (for the series slot).
    :type doc_id: str
    :return: Prompt text.
    :rtype: str
    """
    series = _series_from_doc_id(doc_id)
    return f"""This is a single line cropped from a Cairo Genizah manuscript ({series} collection).
The text may be in Hebrew, Aramaic, or Judaeo-Arabic (Arabic written in Hebrew script).

Transcribe the text in this image exactly as written. Do not normalize or correct the text.
Mark damaged or unclear characters with [?].

Return ONLY the transcription of this one line with no commentary."""


def _adversarial_prompt(doc_id: str, htr_line: str) -> str:
    """Adversarial reconciliation prompt: confirm or refute the HTR read.

    :param doc_id: Document identifier (for the series slot).
    :type doc_id: str
    :param htr_line: Kraken's read of the same line.
    :type htr_line: str
    :return: Prompt text.
    :rtype: str
    """
    return f"""{_line_prompt(doc_id)}

A separate HTR system read this line as:
{htr_line}

That reading may contain errors. Verify it STRICTLY against the ink in the
image — confirm what matches, correct what does not, and never copy the HTR
reading where the ink disagrees. Return ONLY your final reading of the line."""


async def fetch_lines(image_path: str, timeout_s: float = 300.0) -> dict:
    """Get per-line text + geometry from the kraken microservice.

    :param image_path: Path to the document image.
    :type image_path: str
    :param timeout_s: Total request timeout in seconds.
    :type timeout_s: float
    :return: Parsed /transcribe_lines response.
    :rtype: dict
    """
    url = f"{KRAKEN_URL}/transcribe_lines"
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        with open(image_path, "rb") as fh:
            form = aiohttp.FormData()
            form.add_field("model_path", KRAKEN_MODEL)
            form.add_field("image", fh, filename=Path(image_path).name)
            async with session.post(url, data=form) as resp:
                resp.raise_for_status()
                return await resp.json()


def crop_line(im: Image.Image, bbox: list) -> Image.Image:
    """Crop one line with proportional context padding.

    :param im: Full document image.
    :type im: Image.Image
    :param bbox: [x0, y0, x1, y1] line bounding box.
    :type bbox: list
    :return: Cropped line image.
    :rtype: Image.Image
    """
    x0, y0, x1, y1 = bbox
    h = max(y1 - y0, 1)
    w = max(x1 - x0, 1)
    pad_x = int(w * PAD_X_FRAC)
    pad_y = int(h * PAD_Y_FRAC)
    return im.crop((
        max(x0 - pad_x, 0), max(y0 - pad_y, 0),
        min(x1 + pad_x, im.width), min(y1 + pad_y, im.height),
    ))


async def transcribe_linewise(
    image_path: str,
    doc_id: str,
    vlm_model: str,
    mode: str = "blind",
) -> dict:
    """Read every kraken line with the VLM and reassemble the page.

    :param image_path: Path to the document image.
    :type image_path: str
    :param doc_id: Document identifier.
    :type doc_id: str
    :param vlm_model: LM Studio model id.
    :type vlm_model: str
    :param mode: ``blind`` or ``adversarial``.
    :type mode: str
    :return: Dict with ``text`` (assembled page) and ``lines`` (per-line
        details: bbox, kraken text/confidence, VLM text).
    :rtype: dict
    """
    seg = await fetch_lines(image_path)
    im = Image.open(image_path)
    if im.mode != "RGB":
        im = im.convert("RGB")

    details = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, line in enumerate(seg["lines"]):
            crop = crop_line(im, line["bbox"])
            crop_path = str(Path(tmp) / f"line_{i:03d}.jpg")
            crop.save(crop_path, "JPEG", quality=92)
            prompt = (_adversarial_prompt(doc_id, line["text"])
                      if mode == "adversarial" else _line_prompt(doc_id))
            vlm_text = await transcribe_with_lm_studio(
                model_id=vlm_model, image_path=crop_path, prompt=prompt,
                max_tokens=LINE_MAX_TOKENS,
            ) or ""
            details.append(dict(
                idx=i, bbox=line["bbox"], kraken_text=line["text"],
                kraken_confidence=line.get("confidence"),
                vlm_text=vlm_text.strip(),
            ))

    text = "\n".join(d["vlm_text"] for d in details if d["vlm_text"])
    return dict(
        text=text, lines=details, mode=mode,
        used_binarization=seg["used_binarization"],
        polygon_failures=seg["polygon_failures"],
        image_size=seg["image_size"],
    )


async def _demo(image_path: str, doc_id: str, vlm_model: str) -> None:
    """Manual smoke helper: print the assembled text for one image.

    :param image_path: Path to the document image.
    :type image_path: str
    :param doc_id: Document identifier.
    :type doc_id: str
    :param vlm_model: LM Studio model id.
    :type vlm_model: str
    """
    out = await transcribe_linewise(image_path, doc_id, vlm_model)
    print(f"{len(out['lines'])} lines")
    print(out["text"][:800])


if __name__ == "__main__":
    import sys

    asyncio.run(_demo(sys.argv[1], sys.argv[2],
                      sys.argv[3] if len(sys.argv) > 3
                      else "qwen3-vl-8b-heb-v18b-step700"))
