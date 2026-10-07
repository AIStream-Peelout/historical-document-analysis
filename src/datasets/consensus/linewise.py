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


def _ink_envelope(boxes: list, im_w: int, im_h: int) -> tuple:
    """Bounding envelope of the segmented ink, padded slightly.

    :param boxes: Line bboxes [x0, y0, x1, y1].
    :type boxes: list
    :param im_w: Image width.
    :type im_w: int
    :param im_h: Image height.
    :type im_h: int
    :return: (x0, y0, x1, y1) envelope.
    :rtype: tuple
    """
    if not boxes:
        return 0, 0, im_w, im_h
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    pad = 25
    return (max(x0 - pad, 0), max(y0 - pad, 0),
            min(x1 + pad, im_w), min(y1 + pad, im_h))


def build_bands(boxes: list, im_w: int, im_h: int,
                target_band_px: int = 550, overlap_frac: float = 0.15) -> list:
    """Split the ink envelope into overlapping horizontal bands.

    Sections of ~6-10 manuscript lines keep the VLM at a scale close to its
    whole-page training distribution — single kraken segments proved to be
    word-fragments on which the model hallucinates (2026-08-16 smoke).

    :param boxes: Segmented line bboxes.
    :type boxes: list
    :param im_w: Image width.
    :type im_w: int
    :param im_h: Image height.
    :type im_h: int
    :param target_band_px: Nominal band height in pixels.
    :type target_band_px: int
    :param overlap_frac: Fraction of band height shared with the next band.
    :type overlap_frac: float
    :return: List of band bboxes [x0, y0, x1, y1], top to bottom.
    :rtype: list
    """
    x0, y0, x1, y1 = _ink_envelope(boxes, im_w, im_h)
    height = y1 - y0
    n_bands = max(1, min(10, round(height / target_band_px)))
    band_h = height / n_bands
    overlap = int(band_h * overlap_frac)
    bands = []
    for i in range(n_bands):
        top = int(y0 + i * band_h) - (overlap if i else 0)
        bottom = int(y0 + (i + 1) * band_h) + (overlap if i < n_bands - 1 else 0)
        bands.append([x0, max(top, 0), x1, min(bottom, im_h)])
    return bands


def _dedup_join(band_texts: list, tail_lines: int = 4) -> str:
    """Join band texts, dropping overlap-duplicated boundary lines.

    A leading line of band *k+1* is dropped when its letter stream is
    near-contained in one of band *k*'s trailing lines (or vice versa).

    :param band_texts: Per-band assembled text, top to bottom.
    :type band_texts: list
    :param tail_lines: How many boundary lines to compare on each side.
    :type tail_lines: int
    :return: Deduplicated page text.
    :rtype: str
    """
    from src.datasets.evaluations.helper_eval_scripts.audit_genizah_benchmark import (
        letters_only,
    )

    def similar(a: str, b: str) -> bool:
        la, lb = letters_only(a), letters_only(b)
        if len(la) < 6 or len(lb) < 6:
            return la == lb and bool(la)
        short, long_ = (la, lb) if len(la) <= len(lb) else (lb, la)
        grams = [short[i:i + 5] for i in range(len(short) - 4)]
        hits = sum(g in long_ for g in grams)
        return hits / len(grams) >= 0.7

    out_lines: list = []
    for text in band_texts:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if out_lines:
            tail = out_lines[-tail_lines:]
            while lines and any(similar(lines[0], t) for t in tail):
                lines.pop(0)
        out_lines.extend(lines)
    return "\n".join(out_lines)


BAND_MAX_TOKENS = 2000


def _kraken_letters_in_band(seg_lines: list, band: list) -> int:
    """Letters kraken read inside a band (by line-center membership).

    :param seg_lines: /transcribe_lines records (bbox + text).
    :type seg_lines: list
    :param band: Band bbox [x0, y0, x1, y1].
    :type band: list
    :return: Total kraken letters whose line centre falls in the band.
    :rtype: int
    """
    from src.datasets.evaluations.helper_eval_scripts.audit_genizah_benchmark import (
        letters_only,
    )
    total = 0
    for line in seg_lines:
        cy = (line["bbox"][1] + line["bbox"][3]) / 2
        if band[1] <= cy < band[3]:
            total += len(letters_only(line["text"]))
    return total


async def _read_band(im, band: list, doc_id: str, vlm_model: str, tmp: str,
                     seg_lines: list, depth: int = 0) -> tuple:
    """Read one band; bisect once on loop suspicion, truncate as last resort.

    Long-period loops evade the 12-gram loop_ratio screen (observed 19k-char
    band at ratio 0.29), so suspicion is volume-based: far more VLM letters
    than kraken saw ink in the band.  A smaller crop reliably changes the
    decode attractor, hence bisection before truncation.

    :param im: Full page image.
    :type im: Image.Image
    :param band: Band bbox.
    :type band: list
    :param doc_id: Document identifier.
    :type doc_id: str
    :param vlm_model: LM Studio model id.
    :type vlm_model: str
    :param tmp: Temp dir for crops.
    :type tmp: str
    :param seg_lines: Kraken line records for volume estimation.
    :type seg_lines: list
    :param depth: Current bisection depth (max 2).
    :type depth: int
    :return: (text, flags) where flags notes bisect/truncate events.
    :rtype: tuple
    """
    from src.datasets.consensus.consensus_gate import build_fragment_prompt
    from src.datasets.evaluations.helper_eval_scripts.audit_genizah_benchmark import (
        letters_only,
    )

    crop = im.crop(tuple(band))
    crop_path = str(Path(tmp) / f"band_d{depth}_{band[1]}.jpg")
    crop.save(crop_path, "JPEG", quality=92)
    text = (await transcribe_with_lm_studio(
        model_id=vlm_model, image_path=crop_path,
        prompt=build_fragment_prompt(doc_id), max_tokens=BAND_MAX_TOKENS,
    ) or "").strip()

    kraken_letters = _kraken_letters_in_band(seg_lines, band)
    vlm_letters = len(letters_only(text))
    suspect = vlm_letters > max(3 * kraken_letters, 400)
    if not suspect:
        return text, []

    if depth < 2 and (band[3] - band[1]) > 220:
        mid = (band[1] + band[3]) // 2
        overlap = int((band[3] - band[1]) * 0.08)
        top_half = [band[0], band[1], band[2], mid + overlap]
        bottom_half = [band[0], mid - overlap, band[2], band[3]]
        t1, f1 = await _read_band(im, top_half, doc_id, vlm_model, tmp,
                                  seg_lines, depth + 1)
        t2, f2 = await _read_band(im, bottom_half, doc_id, vlm_model, tmp,
                                  seg_lines, depth + 1)
        return _dedup_join([t1, t2]), ["bisect"] + f1 + f2

    # Last resort: keep leading lines up to ~2x kraken's ink volume.
    budget = max(2 * kraken_letters, 200)
    kept, count = [], 0
    for ln in text.splitlines():
        kept.append(ln)
        count += len(letters_only(ln))
        if count >= budget:
            break
    return "\n".join(kept), ["truncate"]


async def transcribe_sections(
    image_path: str,
    doc_id: str,
    vlm_model: str,
) -> dict:
    """Read the page as overlapping horizontal sections and reassemble.

    Uses the byte-identical calibrated whole-page prompt on each band (a
    band is simply a smaller manuscript image; the prompt never referenced
    page extent).

    :param image_path: Path to the document image.
    :type image_path: str
    :param doc_id: Document identifier.
    :type doc_id: str
    :param vlm_model: LM Studio model id.
    :type vlm_model: str
    :return: Dict with ``text``, per-band details, and segmentation stats.
    :rtype: dict
    """
    seg = await fetch_lines(image_path)
    im = Image.open(image_path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    boxes = [l["bbox"] for l in seg["lines"]]
    bands = build_bands(boxes, im.width, im.height)

    band_details = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, band in enumerate(bands):
            text, flags = await _read_band(im, band, doc_id, vlm_model, tmp,
                                           seg["lines"])
            band_details.append(dict(idx=i, bbox=band, vlm_text=text,
                                     flags=flags))

    text = _dedup_join([b["vlm_text"] for b in band_details])
    return dict(
        text=text, bands=band_details, n_segments=len(seg["lines"]),
        used_binarization=seg["used_binarization"],
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
