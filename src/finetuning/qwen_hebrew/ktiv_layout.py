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
  boxes are merged back into one word (rate drops to ~0.06).  Densely written
  pages compress real spaces to ~0.08-0.10 line heights, so the fragment cut
  is chosen per page from the gap distribution (:func:`_page_merge_cut`).
* Geometry is not uniformly trustworthy (GT-mash audit, 2026-08): some pages
  draw boxes 2-3x taller than the line pitch, chaining neighbouring lines
  into one cluster whose x-sort interleaves their words (fix: targeted
  rescue via :func:`_split_interleaved`), and some carry no y information at
  all — every selector at y=0 with mirrored x (fix: served order split at
  BreakLine markers, :func:`served_lines`).
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
MERGE_GAP_LINE_HEIGHTS = 0.10   # default merge cut (in line heights); see _page_merge_cut
MERGE_CUT_MIN = 0.03            # adaptive cut is clamped into this range
MERGE_CUT_MAX = 0.10
OVERLAP_MERGE_FLOOR = 0.15      # boxes overlapping deeper than this are NOT fragments
MAX_MERGED_WORD_LETTERS = 18    # no Hebrew word is longer; guard against chain-merges
LINE_TOL_LINE_HEIGHTS = 0.5     # y-distance below which words share a line
COLUMN_GAP_LINE_HEIGHTS = 1.5   # empty x-band at least this wide splits columns
COLUMN_MIN_SHARE = 0.15         # each column must hold this share of words
DEGENERATE_Y_SPAN = 4.0         # all-word y-range below this = geometry-free page
INTERLEAVE_GAP_LINE_HEIGHTS = 0.3  # RTL gap this deep into the previous box = interleave


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


def _char_width(words: List[Dict]) -> float:
    """Median per-letter box width — a scale anchor independent of box height.

    :param words: Word dicts.
    :type words: List[Dict]
    :return: Median of width/letters over non-gap words with >= 2 Hebrew
        letters (falling back to >= 1); 0.0 when no word carries letters.
    :rtype: float
    """
    for min_letters in (2, 1):
        ratios = []
        for w in words:
            if w["gap"]:
                continue
            n = len(_HEB_RE.findall(w["text"]))
            if n >= min_letters:
                ratios.append((w["box"][2] - w["box"][0]) / n)
        if ratios:
            return statistics.median(ratios)
    return 0.0


def _is_interleaved(line: List[Dict], line_h: float) -> bool:
    """Whether a clustered "line" carries words from several physical lines.

    When boxes are drawn 2-3x taller than the line pitch, clustering at 0.5
    box heights chains neighbouring lines together; the right-to-left x-sort
    then interleaves their words, which shows up as consecutive words whose
    boxes overlap deeply in x (strongly negative RTL gaps) — something that
    essentially never happens between the true neighbours of one line.

    :param line: Clustered words, right-to-left.
    :type line: List[Dict]
    :param line_h: Median box height in pixels.
    :type line_h: float
    :return: True when the interleave signature is present.
    :rtype: bool
    """
    bad = total = 0
    for prev, w in zip(line, line[1:]):
        if prev["gap"] or w["gap"]:
            continue
        total += 1
        if (prev["box"][0] - w["box"][2]) < -INTERLEAVE_GAP_LINE_HEIGHTS * line_h:
            bad += 1
    return bad >= 3 or (bad >= 2 and total > 0 and bad / total >= 0.15)


def _split_interleaved(line: List[Dict], line_h: float) -> List[List[Dict]]:
    """Recursively split a chained line cluster at its widest y-centre gap.

    Only fires on clusters showing the :func:`_is_interleaved` signature, so
    healthy pages pass through byte-identical; on pathological clusters the
    largest internal y-centre delta is the boundary between the chained
    physical lines, and recursion peels them apart top-to-bottom without
    needing any global pitch estimate.

    :param line: Clustered words, right-to-left.
    :type line: List[Dict]
    :param line_h: Median box height in pixels.
    :type line_h: float
    :return: One or more lines, top-to-bottom, each right-to-left.
    :rtype: List[List[Dict]]
    """
    if len(line) < 4 or not _is_interleaved(line, line_h):
        return [line]
    by_y = sorted(line, key=lambda w: (w["box"][1] + w["box"][3]) / 2)
    ycs = [(w["box"][1] + w["box"][3]) / 2 for w in by_y]
    d_max, i_max = 0.0, -1
    for i, (a, b) in enumerate(zip(ycs, ycs[1:])):
        if b - a > d_max:
            d_max, i_max = b - a, i
    if d_max <= 1.0:
        return [line]
    top, bottom = by_y[:i_max + 1], by_y[i_max + 1:]
    for part in (top, bottom):
        part.sort(key=lambda w: -w["box"][2])
    return _split_interleaved(top, line_h) + _split_interleaved(bottom, line_h)


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


def _page_merge_cut(gaps_lh: List[float]) -> float:
    """Adaptive fragment/space threshold from a page's own gap distribution.

    Fragment gaps sit near 0 line heights and real spaces at 0.1-0.3 on most
    pages, but densely written pages compress real spaces to ~0.08-0.10 —
    under the fixed 0.10 cut whole lines merged into space-free runs.  A 1-D
    2-means over the page's small gaps finds the valley between the touching
    mode and the space mode; the cut is clamped to
    [:data:`MERGE_CUT_MIN`, :data:`MERGE_CUT_MAX`], and pages without clear
    bimodality (or with too few gaps to tell) keep the historical default.

    :param gaps_lh: Intra-line gaps between adjacent non-gap words, in line
        heights.
    :type gaps_lh: List[float]
    :return: Merge cut in line heights.
    :rtype: float
    """
    pop = sorted(g for g in gaps_lh if -0.02 < g < 0.35)
    if len(pop) < 30:
        return MERGE_GAP_LINE_HEIGHTS
    c0, c1 = pop[0], pop[-1]
    if c1 - c0 < 0.06:
        return MERGE_GAP_LINE_HEIGHTS
    for _ in range(10):
        lo = [g for g in pop if abs(g - c0) <= abs(g - c1)]
        hi = [g for g in pop if abs(g - c0) > abs(g - c1)]
        if not lo or not hi:
            return MERGE_GAP_LINE_HEIGHTS
        c0, c1 = statistics.fmean(lo), statistics.fmean(hi)
    # Only adapt when the space mode is genuinely squeezed toward the cut —
    # on ordinary pages (spaces ~0.17-0.25) the historical 0.10 already sits
    # in the valley, and "adapting" it a hair lower only churns knife-edge
    # fragment merges.
    if c1 - c0 < 0.06 or c1 >= 0.16:
        return MERGE_GAP_LINE_HEIGHTS
    return min(max((c0 + c1) / 2, MERGE_CUT_MIN), MERGE_CUT_MAX)


def merge_line_words(line: List[Dict], line_h: float,
                     merge_cut: float = MERGE_GAP_LINE_HEIGHTS) -> List[str]:
    """Rejoin split word fragments and collapse repeated gaps on one line.

    A fragment merge requires the boxes to be *touching*: closer than
    ``merge_cut`` but not overlapping deeper than
    :data:`OVERLAP_MERGE_FLOOR` line heights (deep overlap means the words
    came from different physical lines that were clustered together — merging
    those manufactures giant unspaced runs).  A merged word is never allowed
    to exceed :data:`MAX_MERGED_WORD_LETTERS` Hebrew letters.

    :param line: Words of one line, right-to-left.
    :type line: List[Dict]
    :param line_h: Effective line height in pixels.
    :type line_h: float
    :param merge_cut: Fragment/space threshold in line heights (per page,
        from :func:`_page_merge_cut`; the module default preserves the
        historical behaviour for callers that do not pass one).
    :type merge_cut: float
    :return: Tokens (words and gap tokens) in reading order.
    :rtype: List[str]
    """
    tokens: List[str] = []
    prev: Optional[Dict] = None
    for w in line:
        gap_px = (prev["box"][0] - w["box"][2]) if prev is not None else 0.0
        if w["gap"]:
            if tokens and tokens[-1] == GAP_TOKEN:
                prev = w
                continue
            tokens.append(GAP_TOKEN)
        elif prev is not None and not prev["gap"] and tokens and tokens[-1] != GAP_TOKEN \
                and -OVERLAP_MERGE_FLOOR * line_h < gap_px < merge_cut * line_h \
                and len(_HEB_RE.findall(tokens[-1] + w["text"])) <= MAX_MERGED_WORD_LETTERS:
            tokens[-1] = tokens[-1] + w["text"]
        else:
            tokens.append(w["text"])
        prev = w
    return tokens


def _is_breakline(item: dict) -> bool:
    """Whether an annotation item is a line-break marker.

    :param item: AnnotationPage item.
    :type item: dict
    :return: True for BreakLine/BreackLine markers.
    :rtype: bool
    """
    item_id = str(item.get("id") or "").lower()
    return "breakline" in item_id or "breackline" in item_id


def served_lines(items: List[dict]) -> List[List[Dict]]:
    """Split served-order words into lines at break-line markers.

    Fallback for pages whose selectors carry no usable geometry (all-zero y,
    mirrored x): there the served order IS reading order and the viewer's
    BreakLine annotations delimit the physical lines — verified to yield
    coherent running text on every such benchmark page.  Word cleaning and
    (text, box) dedup follow :func:`extract_words` exactly.

    :param items: AnnotationPage ``items``.
    :type items: List[dict]
    :return: Lines of word dicts in served order.
    :rtype: List[List[Dict]]
    """
    lines: List[List[Dict]] = []
    cur: List[Dict] = []
    seen = set()
    for item in items:
        if _is_breakline(item):
            if cur:
                lines.append(cur)
                cur = []
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
        cur.append({"text": text, "box": box, "gap": is_gap,
                    "sigla": body.get("sigla") or ""})
    if cur:
        lines.append(cur)
    return lines


def _merge_served_line(line: List[Dict], char_w: float) -> List[str]:
    """Rejoin touching fragments on one served-order line.

    Reading order is the served order (not re-sorted: geometry-free pages
    offer nothing to sort by), so touching is judged by the x-interval
    distance between consecutive boxes — direction-free, hence valid in the
    mirrored frame these pages use — against a char-width scale (box heights
    are zero here).

    :param line: Words of one line in served order.
    :type line: List[Dict]
    :param char_w: Median per-letter box width in pixels.
    :type char_w: float
    :return: Tokens (words and gap tokens) in reading order.
    :rtype: List[str]
    """
    tokens: List[str] = []
    prev: Optional[Dict] = None
    for w in line:
        if w["gap"]:
            if not tokens or tokens[-1] != GAP_TOKEN:
                tokens.append(GAP_TOKEN)
            prev = w
            continue
        touching = False
        if prev is not None and not prev["gap"] and tokens and tokens[-1] != GAP_TOKEN \
                and char_w > 0:
            a, b = prev["box"], w["box"]
            dist = max(b[0] - a[2], a[0] - b[2])
            touching = dist < 0.15 * char_w and \
                len(_HEB_RE.findall(tokens[-1] + w["text"])) <= MAX_MERGED_WORD_LETTERS
        if touching:
            tokens[-1] = tokens[-1] + w["text"]
        else:
            tokens.append(w["text"])
        prev = w
    return tokens


def reconstruct_page(items: List[dict]) -> Dict:
    """Rebuild a page's reading-order text and line geometry from annotations.

    Pages with usable geometry go through columns -> line clustering ->
    interleave rescue (:func:`_split_interleaved`, which un-chains clusters
    that swallowed several physical lines) -> per-page adaptive fragment
    merging.  Pages whose boxes carry no vertical information at all (y-span
    under :data:`DEGENERATE_Y_SPAN` px) fall back to served order split at
    BreakLine markers, when those are present.

    :param items: AnnotationPage ``items``.
    :type items: List[dict]
    :return: ``{"columns": [ {"lines": [ {"text", "box", "n_words"} ] } ],
        "lines": flat line list in reading order, "text": page text,
        "line_h": effective line height, "n_words", "n_gaps",
        "mode": "geometry" | "served_breaklines", "merge_cut"}``.
    :rtype: Dict
    """
    words = extract_words(items)
    if not words:
        return {"columns": [], "lines": [], "text": "", "line_h": 0.0,
                "n_words": 0, "n_gaps": 0, "mode": "geometry",
                "merge_cut": MERGE_GAP_LINE_HEIGHTS}
    ycs = [(w["box"][1] + w["box"][3]) / 2 for w in words if not w["gap"]] or \
          [(w["box"][1] + w["box"][3]) / 2 for w in words]
    degenerate = (max(ycs) - min(ycs)) < DEGENERATE_Y_SPAN
    if degenerate and any(_is_breakline(it) for it in items):
        char_w = _char_width(words)
        flat_lines = []
        for line in served_lines(items):
            tokens = _merge_served_line(line, char_w)
            text = " ".join(tokens).strip()
            if not text:
                continue
            box = (min(w["box"][0] for w in line), min(w["box"][1] for w in line),
                   max(w["box"][2] for w in line), max(w["box"][3] for w in line))
            flat_lines.append({"text": text, "box": box,
                               "n_words": sum(1 for t in tokens if t != GAP_TOKEN)})
        return {
            "columns": [{"lines": flat_lines}],
            "lines": flat_lines,
            "text": "\n".join(ln["text"] for ln in flat_lines),
            "line_h": 0.0,
            "n_words": sum(1 for w in words if not w["gap"]),
            "n_gaps": sum(1 for w in words if w["gap"]),
            "mode": "served_breaklines",
            "merge_cut": MERGE_GAP_LINE_HEIGHTS,
        }
    box_h = _median_height(words)
    col_line_lists = [
        [seg for line in cluster_lines(col_words, box_h)
         for seg in _split_interleaved(line, box_h)]
        for col_words in split_columns(words, box_h)]
    gaps_lh = []
    for col_lines in col_line_lists:
        for line in col_lines:
            for prev, w in zip(line, line[1:]):
                if not prev["gap"] and not w["gap"]:
                    gaps_lh.append((prev["box"][0] - w["box"][2]) / box_h)
    merge_cut = _page_merge_cut(gaps_lh)
    columns = []
    flat_lines = []
    for col_lines in col_line_lists:
        entries = []
        for line in col_lines:
            tokens = merge_line_words(line, box_h, merge_cut)
            text = " ".join(tokens).strip()
            if not text:
                continue
            box = (min(w["box"][0] for w in line), min(w["box"][1] for w in line),
                   max(w["box"][2] for w in line), max(w["box"][3] for w in line))
            entry = {"text": text, "box": box,
                     "n_words": sum(1 for t in tokens if t != GAP_TOKEN)}
            entries.append(entry)
            flat_lines.append(entry)
        columns.append({"lines": entries})
    return {
        "columns": columns,
        "lines": flat_lines,
        "text": "\n".join(ln["text"] for ln in flat_lines),
        "line_h": box_h,
        "n_words": sum(1 for w in words if not w["gap"]),
        "n_gaps": sum(1 for w in words if w["gap"]),
        "mode": "geometry",
        "merge_cut": merge_cut,
    }


OCR_SUBSTANTIAL_LETTERS = 4    # units below this cannot define or bridge a column
OCR_MIN_COL_LETTER_SHARE = 0.12  # letter-poor bands merge into a neighbour


def _split_columns_weighted(units: List[Dict], line_h: float) -> List[List[Dict]]:
    """Column split for OCR line/fragment boxes, robust to gutter strays.

    Differs from :func:`split_columns` (word annotations) in two ways needed
    for OCR output: only units with >= :data:`OCR_SUBSTANTIAL_LETTERS` Hebrew
    letters define the occupied x-intervals (so a stray short word sitting in
    the gutter cannot bridge it or fabricate a column), and letter-poor bands
    are merged into their nearest neighbour instead of vetoing the whole
    split (so marginalia never force a single-column fallback).

    :param units: Dicts with ``box`` (x0, y0, x1, y1) and ``letters`` counts.
    :type units: List[Dict]
    :param line_h: Median line height in pixels.
    :type line_h: float
    :return: Columns, rightmost first; each a list of unit dicts.
    :rtype: List[List[Dict]]
    """
    substantial = [u for u in units if u["letters"] >= OCR_SUBSTANTIAL_LETTERS]
    cuts: List[float] = []
    if len(substantial) >= 8:
        intervals = sorted((u["box"][0], u["box"][2]) for u in substantial)
        merged = [list(intervals[0])]
        for a, b in intervals[1:]:
            if a <= merged[-1][1] + 0.2 * line_h:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        for (_a0, b0), (a1, _b1) in zip(merged, merged[1:]):
            if a1 - b0 >= COLUMN_GAP_LINE_HEIGHTS * line_h:
                cuts.append((b0 + a1) / 2)
    bounds = [-1e12] + sorted(cuts) + [1e12]
    bands = []
    for lo, hi in zip(bounds, bounds[1:]):
        band = [u for u in units if lo <= (u["box"][0] + u["box"][2]) / 2 < hi]
        if band:
            bands.append(band)

    def _cx(band: List[Dict]) -> float:
        return statistics.median((u["box"][0] + u["box"][2]) / 2 for u in band)

    total_letters = sum(u["letters"] for u in units) or 1
    merged_any = True
    while merged_any and len(bands) > 1:
        merged_any = False
        for i, band in enumerate(bands):
            if sum(u["letters"] for u in band) / total_letters < OCR_MIN_COL_LETTER_SHARE:
                nbrs = [j for j in (i - 1, i + 1) if 0 <= j < len(bands)]
                j = min(nbrs, key=lambda j: abs(_cx(bands[j]) - _cx(band)))
                bands[j].extend(band)
                bands.pop(i)
                merged_any = True
                break
    bands.sort(key=lambda b: -_cx(b))
    return bands


def reorder_ocr_lines(lines: List[dict]) -> str:
    """Rebuild Hebrew reading order from OCR line fragments with bboxes.

    Input is OCR output (e.g. the Kraken microservice's ``/transcribe_lines``)
    whose served order is segmentation order, not reading order.  Each unit is
    treated as atomic text: columns are split right-to-left with
    :func:`_split_columns_weighted`, units are clustered into visual lines
    with :func:`cluster_lines`, ordered right-to-left within a line, and
    touching fragments are rejoined by :func:`merge_line_words` — the same
    geometry pipeline the ground truth is built with, so a reordered
    hypothesis is order-consistent with the GT by construction.

    :param lines: Dicts with ``text`` and ``bbox`` ``[x0, y0, x1, y1]`` in
        image pixels (extra keys ignored).
    :type lines: List[dict]
    :return: Page text, one reconstructed line per ``\\n``, columns
        rightmost-first.
    :rtype: str
    """
    units = []
    for ln in lines:
        text = unicodedata.normalize("NFC", (ln.get("text") or "").strip())
        if not text:
            continue
        x0, y0, x1, y1 = ln["bbox"]
        units.append({"text": text,
                      "box": (float(x0), float(y0), float(x1), float(y1)),
                      "gap": bool(_DOTS_RE.match(text)),
                      "letters": len(_HEB_RE.findall(text))})
    if not units:
        return ""
    heights = [u["box"][3] - u["box"][1] for u in units if u["letters"] >= 2] or \
              [u["box"][3] - u["box"][1] for u in units]
    line_h = max(statistics.median(heights), 1.0)
    out_lines = []
    for band in _split_columns_weighted(units, line_h):
        for cluster in cluster_lines(band, line_h):
            text = " ".join(merge_line_words(cluster, line_h)).strip()
            if text:
                out_lines.append(text)
    return "\n".join(out_lines)


def hebrew_letters(text: str) -> int:
    """Count Hebrew letters in a string.

    :param text: Any text.
    :type text: str
    :return: Number of characters in the Hebrew letter block.
    :rtype: int
    """
    return len(_HEB_RE.findall(text))
