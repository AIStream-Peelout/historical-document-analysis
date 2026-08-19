"""Layout reconstruction for KTIV (NLI) word-level transcription annotations.

The KTIV viewer serves each transcribed page as a W3C AnnotationPage: one
Annotation per *word* with its text, editorial sigla and a rectangular
SvgSelector in the page image's pixel frame.  Three facts about that payload
(measured 2026-08-18 on 850 manuscripts) shape this module:

* Served order is NOT reliable reading order: on multi-column pages the
  columns interleave, and ~12% of items are duplicates (one copy per sigla
  code on the same word).  Reading order is therefore rebuilt from geometry
  alone: dedupe -> split columns on x-gaps -> cluster lines on y -> words
  right-to-left.
* Transcribers split damaged words into per-fragment annotations, so the raw
  single-letter token rate is 0.18 (vs 0.016 in edited Genizah GT).  The gap
  between consecutive boxes on a line is bimodal — touching (< ~0.05 line
  heights) for fragments of one word, 0.1-0.3 for real spaces — so touching
  boxes are merged back into one word (rate drops to ~0.06, threshold-
  insensitive over 0.03-0.10).
* Illegible runs are dot-run words (``.....``); they become the ``[...]`` gap
  token.  Editorial symbols that are not ink (end-of-paragraph ``⟡``, line
  filler ``~``, supralinear digits, uncertainty ``?``) are stripped.

Everything here is pure functions over plain dicts, so it is unit-testable
without images.
"""

import re
import statistics
import unicodedata
from typing import Dict, List, Optional, Tuple

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_HEB_RE = re.compile(r"[א-ת]")
_DOTS_RE = re.compile(r"^[.·…]+$")
_STRIP_CHARS = "⟡~¹²³?«»‹›"

GAP_TOKEN = "[...]"
MERGE_GAP_LINE_HEIGHTS = 0.10   # boxes closer than this (in line heights) = one word
LINE_TOL_LINE_HEIGHTS = 0.5     # y-distance below which words share a line
COLUMN_GAP_LINE_HEIGHTS = 1.5   # empty x-band at least this wide splits columns
COLUMN_MIN_SHARE = 0.15         # each column must hold this share of words


def parse_box(selector_value: str) -> Optional[Tuple[float, float, float, float]]:
    """Parse an SvgSelector path into an axis-aligned box.

    :param selector_value: The ``<svg><path d="M x,y x,y ..."/></svg>`` string.
    :type selector_value: str
    :return: (x0, y0, x1, y1) or None when fewer than four points are present.
    :rtype: Optional[Tuple[float, float, float, float]]
    """
    nums = [float(v) for v in _NUM_RE.findall(selector_value or "")]
    if len(nums) < 8:
        return None
    xs, ys = nums[0::2], nums[1::2]
    return (min(xs), min(ys), max(xs), max(ys))


def clean_word(value: str) -> Tuple[str, bool]:
    """Apply the sigla policy to one annotation value.

    :param value: Raw annotation text.
    :type value: str
    :return: (cleaned text, is_gap).  Cleaned text is empty when nothing
        remains after stripping non-ink symbols.
    :rtype: Tuple[str, bool]
    """
    text = unicodedata.normalize("NFC", (value or "").strip())
    if not text:
        return "", False
    if _DOTS_RE.match(text):
        return GAP_TOKEN, True
    text = "".join(ch for ch in text if ch not in _STRIP_CHARS)
    text = text.replace("...", "").replace("…", "").strip()
    return text, False


def extract_words(items: List[dict]) -> List[Dict]:
    """Turn AnnotationPage items into deduplicated word dicts.

    :param items: The ``items`` list of one AnnotationPage.
    :type items: List[dict]
    :return: Words as ``{"text", "box", "gap", "sigla"}`` in served order,
        duplicates (same text and box) removed, break markers dropped.
    :rtype: List[Dict]
    """
    seen = set()
    words: List[Dict] = []
    for item in items:
        item_id = str(item.get("id") or "").lower()
        if "breakline" in item_id or "breackline" in item_id:
            continue
        body = item.get("body") or {}
        text, is_gap = clean_word(body.get("value") or "")
        if not text:
            continue
        box = parse_box(((item.get("target") or {}).get("selector") or {}).get("value") or "")
        if box is None:
            continue
        key = (text, tuple(round(v) for v in box))
        if key in seen:
            continue
        seen.add(key)
        words.append({"text": text, "box": box, "gap": is_gap,
                      "sigla": body.get("sigla") or ""})
    return words


def _median_height(words: List[Dict]) -> float:
    """Median box height of non-gap words (gap boxes can span many lines).

    :param words: Word dicts.
    :type words: List[Dict]
    :return: Median height in pixels (>= 1).
    :rtype: float
    """
    hs = [w["box"][3] - w["box"][1] for w in words if not w["gap"]] or \
         [w["box"][3] - w["box"][1] for w in words]
    return max(statistics.median(hs), 1.0) if hs else 1.0


def split_columns(words: List[Dict], line_h: float) -> List[List[Dict]]:
    """Split words into columns ordered right-to-left using empty x-bands.

    Word boxes are projected onto x and merged into occupied intervals; an
    unoccupied band wider than :data:`COLUMN_GAP_LINE_HEIGHTS` line heights
    splits columns, provided each side keeps :data:`COLUMN_MIN_SHARE` of the
    words (so a hole in a single column does not masquerade as a gutter).

    :param words: Word dicts.
    :type words: List[Dict]
    :param line_h: Median line height in pixels.
    :type line_h: float
    :return: Columns, rightmost first; each a list of word dicts.
    :rtype: List[List[Dict]]
    """
    if len(words) < 8:
        return [words]
    intervals = sorted((w["box"][0], w["box"][2]) for w in words if not w["gap"]) \
        or sorted((w["box"][0], w["box"][2]) for w in words)
    merged = [list(intervals[0])]
    for a, b in intervals[1:]:
        if a <= merged[-1][1] + 0.2 * line_h:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    cuts = []
    for (a0, b0), (a1, b1) in zip(merged, merged[1:]):
        if a1 - b0 >= COLUMN_GAP_LINE_HEIGHTS * line_h:
            cuts.append((b0 + a1) / 2)
    if not cuts:
        return [words]
    bounds = [-1e12] + sorted(cuts) + [1e12]
    cols = []
    for lo, hi in zip(bounds, bounds[1:]):
        col = [w for w in words if lo <= (w["box"][0] + w["box"][2]) / 2 < hi]
        if col:
            cols.append(col)
    if any(len(c) < COLUMN_MIN_SHARE * len(words) for c in cols):
        return [words]
    cols.sort(key=lambda c: -statistics.median((w["box"][0] + w["box"][2]) / 2 for w in c))
    return cols


def cluster_lines(words: List[Dict], line_h: float) -> List[List[Dict]]:
    """Group one column's words into lines by vertical centre.

    :param words: Word dicts of one column.
    :type words: List[Dict]
    :param line_h: Median line height in pixels.
    :type line_h: float
    :return: Lines top-to-bottom, words within a line ordered right-to-left.
    :rtype: List[List[Dict]]
    """
    ordered = sorted(words, key=lambda w: (w["box"][1] + w["box"][3]) / 2)
    lines: List[List[Dict]] = []
    centres: List[float] = []
    for w in ordered:
        yc = (w["box"][1] + w["box"][3]) / 2
        if lines and abs(yc - centres[-1]) < LINE_TOL_LINE_HEIGHTS * line_h:
            lines[-1].append(w)
            centres[-1] = 0.7 * centres[-1] + 0.3 * yc
        else:
            lines.append([w])
            centres.append(yc)
    for line in lines:
        line.sort(key=lambda w: -w["box"][2])
    return lines


def merge_line_words(line: List[Dict], line_h: float) -> List[str]:
    """Rejoin split word fragments and collapse repeated gaps on one line.

    :param line: Words of one line, right-to-left.
    :type line: List[Dict]
    :param line_h: Median line height in pixels.
    :type line_h: float
    :return: Tokens (words and gap tokens) in reading order.
    :rtype: List[str]
    """
    tokens: List[str] = []
    prev: Optional[Dict] = None
    for w in line:
        if w["gap"]:
            if tokens and tokens[-1] == GAP_TOKEN:
                prev = w
                continue
            tokens.append(GAP_TOKEN)
        elif prev is not None and not prev["gap"] and tokens and tokens[-1] != GAP_TOKEN \
                and (prev["box"][0] - w["box"][2]) < MERGE_GAP_LINE_HEIGHTS * line_h:
            tokens[-1] = tokens[-1] + w["text"]
        else:
            tokens.append(w["text"])
        prev = w
    return tokens


def reconstruct_page(items: List[dict]) -> Dict:
    """Rebuild a page's reading-order text and line geometry from annotations.

    :param items: AnnotationPage ``items``.
    :type items: List[dict]
    :return: ``{"columns": [ {"lines": [ {"text", "box", "n_words"} ] } ],
        "lines": flat line list in reading order, "text": page text,
        "line_h": median line height, "n_words", "n_gaps"}``.
    :rtype: Dict
    """
    words = extract_words(items)
    if not words:
        return {"columns": [], "lines": [], "text": "", "line_h": 0.0,
                "n_words": 0, "n_gaps": 0}
    line_h = _median_height(words)
    columns = []
    flat_lines = []
    for col_words in split_columns(words, line_h):
        col_lines = []
        for line in cluster_lines(col_words, line_h):
            tokens = merge_line_words(line, line_h)
            text = " ".join(tokens).strip()
            if not text:
                continue
            box = (min(w["box"][0] for w in line), min(w["box"][1] for w in line),
                   max(w["box"][2] for w in line), max(w["box"][3] for w in line))
            entry = {"text": text, "box": box,
                     "n_words": sum(1 for t in tokens if t != GAP_TOKEN)}
            col_lines.append(entry)
            flat_lines.append(entry)
        columns.append({"lines": col_lines})
    return {
        "columns": columns,
        "lines": flat_lines,
        "text": "\n".join(ln["text"] for ln in flat_lines),
        "line_h": line_h,
        "n_words": sum(1 for w in words if not w["gap"]),
        "n_gaps": sum(1 for w in words if w["gap"]),
    }


def hebrew_letters(text: str) -> int:
    """Count Hebrew letters in a string.

    :param text: Any text.
    :type text: str
    :return: Number of characters in the Hebrew letter block.
    :rtype: int
    """
    return len(_HEB_RE.findall(text))
