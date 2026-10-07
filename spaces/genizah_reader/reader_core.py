# File name: reader_core.py
# Date: 9/10/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Model-free core of the Genizah reader Space: prompts, parsing, drawing.

Kept separate from ``app.py`` (which imports torch/spaces/gradio) so it can
be unit-tested on any machine. The prompt strings are byte-identical to the
training/eval prompts in ``historical-document-analysis`` — the wording is
load-bearing for this fine-tune.
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

PROMPTS: Dict[str, str] = {
    "transcribe": (
        "This image is a manuscript fragment from the Cairo Genizah — handwritten Hebrew "
        "script (the language may be Hebrew, Judeo-Arabic, or Aramaic).\n\n"
        "Transcribe the text exactly as written, in reading order. Mark unclear characters with [?].\n"
        "Where text is lost or illegible due to damage, write [...].\n"
        "Do NOT correct, restore, or complete from memory.\n\n"
        "Return ONLY the transcription."),
    "lines": (
        'Transcribe this manuscript page line by line. Respond with ONLY a JSON array; each '
        'element {"text": "...", "bbox_2d": [x1, y1, x2, y2]} gives one line\'s transcription '
        'and its bounding box, coordinates normalized to 0-1000. Preserve reading order.'),
    "locate": (
        'Locate the exact Hebrew phrase "{phrase}" on this manuscript page. Respond with ONLY '
        'a JSON object {{"bbox_2d": [x1, y1, x2, y2]}} giving the phrase\'s bounding box, '
        'coordinates normalized to 0-1000. No other text.'),
}
MAX_NEW_TOKENS = {"transcribe": 4096, "lines": 3500, "locate": 64}
GPU_SECONDS = {"transcribe": 150, "lines": 150, "locate": 40}
_JSON_BOX = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")
_NUM = (int, float)


def clamp_box(box: List[float]) -> List[int]:
    """Clamp a 0-1000 box to range and order its corners."""
    x1, y1, x2, y2 = (int(round(max(0.0, min(1000.0, float(v))))) for v in box)
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def parse_locate(text: str) -> Optional[List[int]]:
    """First ``[x1, y1, x2, y2]`` in the reply (the eval's rule), clamped; None if absent."""
    m = _JSON_BOX.search(text or "")
    if not m:
        return None
    b = clamp_box([int(v) for v in m.groups()])
    return b if b[2] > b[0] and b[3] > b[1] else None


def parse_lines(text: str) -> Tuple[List[Dict[str, Any]], bool]:
    """Recover ``{text, bbox_2d}`` objects from a (possibly still streaming) JSON array.

    Scans for closed top-level objects, so lines can be drawn as they arrive
    and a truncated array still yields its complete elements.

    :param text: Model output so far.
    :returns: ``(lines, complete)`` — ``complete`` when the array closed.
    """
    objs: List[Dict[str, Any]] = []
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(text or ""):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objs.append(json.loads(text[start:i + 1]))
                except json.JSONDecodeError:
                    pass
                start = None
    lines = []
    for o in objs:
        if not isinstance(o, dict):
            continue
        t, b = str(o.get("text", "")).strip(), o.get("bbox_2d")
        if t and isinstance(b, list) and len(b) == 4 and all(isinstance(v, _NUM) for v in b):
            cb = clamp_box(b)
            if cb[2] > cb[0] and cb[3] > cb[1]:
                lines.append({"text": t, "bbox_2d": cb})
    complete = (text or "").rstrip().endswith("]") and depth == 0
    return lines, complete


def looping(text: str, window: int = 24, repeats: int = 6) -> bool:
    """True when the tail repeats itself — the decode has entered a loop."""
    t = text or ""
    if len(t) < window * repeats:
        return False
    tail = t[-window:]
    return t.count(tail) >= repeats


def draw_boxes(image: Image.Image, boxes: List[List[int]], labels: Optional[List[str]] = None,
               color: Tuple[int, int, int] = (214, 51, 108), width: int = 4) -> Image.Image:
    """Draw 0-1000 boxes on a copy of the image (scaled to its pixel size)."""
    im = image.convert("RGB").copy()
    d = ImageDraw.Draw(im, "RGBA")
    W, H = im.size
    for k, b in enumerate(boxes):
        x1, y1, x2, y2 = b[0] * W / 1000, b[1] * H / 1000, b[2] * W / 1000, b[3] * H / 1000
        d.rectangle([x1, y1, x2, y2], outline=color + (255,), width=width, fill=color + (28,))
        label = labels[k] if labels and k < len(labels) else str(k + 1)
        pad = max(2, width)
        d.rectangle([x2 - 14 * len(label) - 2 * pad, y1 - 22, x2, y1], fill=color + (230,))
        d.text((x2 - 14 * len(label) - pad, y1 - 20), label, fill=(255, 255, 255, 255))
    return im


def rtl(text: str) -> str:
    """Wrap text so browsers render it right-to-left (Hebrew)."""
    return "‫" + (text or "") + "‬"
