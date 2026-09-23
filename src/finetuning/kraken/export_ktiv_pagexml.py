"""Export KTIV word-box geometry as PageXML for ``ketos segtrain`` (Genizah line segmenter).

Steps, run in order:

1. ``candidates`` (host): gate manuscripts and pages the way the VLM builder
   does (benchmark id exclusion + shingle decontamination, frame check, see
   :mod:`src.finetuning.qwen_hebrew.build_ktiv_dataset`), then apply
   segmentation-specific gates: geometry-mode pages only (no served-breakline
   fallback), no rotated scans (text running vertically), at least
   :data:`MIN_LINES` lines, at most :data:`MAX_COLUMNS` columns.  Sample up to
   ``--pages`` pages (at most ``--per-ms`` per manuscript), split by manuscript,
   export page images at most :data:`MAX_H` px high (blla's input height, so
   nothing the network sees is lost) and ``candidates.jsonl``.  Lines are the
   physical lines of :mod:`ktiv_layout` (columns → line clustering → interleave
   rescue) with their exact word membership; gap tokens and vertical marginal
   words are kept out of line geometry (vertical words are painted out later).
   ``relines`` recomputes that geometry for an existing ``candidates.jsonl``
   without changing the page selection, images or split.
2. ``seg_gate.py`` (inside the kraken 7 container): blla2026 predictions per
   page, with Gen_01 text for detected lines not covered by any GT line.
3. ``write`` (host), per page:

   * a GT line that blla2026 already segments 1:1 (≥ 80 % of its baseline in the
     line's box, spanning ≥ 85 % of its width) takes blla2026's baseline — the
     convention the recogniser is used to;
   * every other line is split at horizontal gaps wider than
     :data:`SEG_GAP_PITCHES` pitches (holes, marginal additions) and each segment
     gets a straight baseline on the ink body bottom with the slope fixed to the
     page's consensus slope (median of the adopted blla2026 baselines), shifted
     by the systematic offset measured between the estimator and blla2026 on the
     matched lines;
   * lines that cannot be placed (fit failures, oversized KTIV boxes, lines
     crowding a neighbour) are painted out of the image together with vertical
     marginal words and confidently read blla2026 lines on untranscribed text,
     so no real writing is ever taught as background; pages with too many such
     lines, or wider than :data:`MAX_W` (training memory), are rejected;
   * PageXML: one untyped ``TextRegion`` per column (kraken class ``text``),
     untyped ``TextLine`` per baseline segment (``default``) — blla2026's class
     mapping — plus ``train.lst`` / ``val.lst`` / ``stats.json`` and QA overlays.

Usage (repo root):
    .venv/bin/python -m src.finetuning.kraken.export_ktiv_pagexml candidates --out DIR [--pages 2000]
    .venv/bin/python -m src.finetuning.kraken.export_ktiv_pagexml relines --out DIR
    (run seg_gate.py in the container, see segtrain_ktiv.sh)
    .venv/bin/python -m src.finetuning.kraken.export_ktiv_pagexml write --out DIR
"""
import argparse
import json
import logging
import random
import statistics
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image as PILImage, ImageDraw

from src.finetuning.kraken.export_ktiv_lines import column_pitch, ink_centre, split_manuscripts
from src.finetuning.kraken.seg_geometry import covered_share, match_lines, polyline_box, y_at
from src.finetuning.qwen_hebrew import build_ktiv_dataset as B
from src.finetuning.qwen_hebrew.ktiv_layout import (
    DEGENERATE_Y_SPAN, _median_height, _page_merge_cut, _split_interleaved, cluster_lines, extract_words,
    hebrew_letters, merge_line_words, reconstruct_page, split_columns)

log = logging.getLogger(__name__)

MAX_H = 1800                 # blla input height ([1,1800,0,3]); taller pages are downscaled
MAX_W = 2600                 # widest training page at MAX_H: training peak ~6.7 GiB at 2,500 px, OOM > 10 GiB at 4,000
MIN_LINES = 3
MAX_COLUMNS = 3
ROTATED_SHARE = 0.5          # >= this share of 3+-letter words taller than wide = rotated scan
VERTICAL_MIN_H = 1.6         # a vertical marginal word is also this many page word-heights tall
VAL_FRACTION = 0.08
SPLIT_SEED = 20260923
JPEG_QUALITY = 92
BODY_HALF = 0.5              # baseline search window half-height around the ink centre, in pitches
CHUNK_PITCHES = 1.5          # words are grouped into chunks at least this wide for baseline sampling
SEG_GAP_PITCHES = 2.5        # a horizontal gap this wide between words splits a line into baseline segments
BLUE_MARGIN = 40             # B - R above this (and B >= G) = blue backing board showing through, not ink
FIT_TOL = 0.2                # chunk residual (pitches) around the fixed-slope baseline
FIT_MIN_INLIERS = 0.6        # share of chunks that must agree with it
END_TOL = 0.33              # an estimated baseline end this far (pitches) off its chunk estimate is refitted per line
FREE_SLOPE_MAX_DEV = 0.08    # ...with its own slope, if that stays within this of the page slope; else painted out
INK_COL_FRAC = 0.15          # a column counts as ink when its band darkness exceeds this share of the line's maximum
MATCH_MIN_SHARE = 0.8        # blla2026 baseline share inside the GT line box to adopt it...
MATCH_MIN_SPAN = 0.85        # ...and its span relative to the GT line width
TALL_BOXES = 1.8             # line's median word height / page median word height above this = unreliable KTIV boxes
MIN_LINE_GAP = 0.45          # min vertical distance between baselines overlapping in x, in pitches
MASK_MAX_LINES = 3           # pages with at most max(this, MASK_MAX_SHARE × lines) painted-out lines are kept
MASK_MAX_SHARE = 0.25
UNCOVERED_MIN_LETTERS = 5    # an uncovered blla2026 line with this many letters...
UNCOVERED_MIN_CONF = 0.6     # ...at this mean confidence is untranscribed writing: painted out
UNCOVERED_MIN_PITCHES = 2.0  # ...if at least this long
PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"


# ---------------------------------------------------------------- line geometry from KTIV

def _is_vertical(w: Dict, word_h: float) -> bool:
    """A 3+-letter word written vertically (marginal writing): taller than wide AND long vertically.

    KTIV word boxes are often taller than wide on ordinary lines (loose boxes), so
    height > width alone would flag real text; a vertically written word is also
    much taller than the page's ordinary word height.

    :param w: Word dict from :func:`extract_words`.
    :type w: Dict
    :param word_h: Median word-box height of the page, px.
    :type word_h: float
    :return: True for vertical words.
    :rtype: bool
    """
    h, wd = w["box"][3] - w["box"][1], w["box"][2] - w["box"][0]
    return hebrew_letters(w["text"]) >= 3 and h > wd and h > VERTICAL_MIN_H * word_h


def is_rotated(words: Sequence[Dict]) -> bool:
    """Whether a page's text runs vertically (most 3+-letter word boxes taller than wide).

    :param words: Words from :func:`extract_words`.
    :type words: Sequence[Dict]
    :return: True when at least :data:`ROTATED_SHARE` of them are vertical.
    :rtype: bool
    """
    long_words = [w for w in words if not w["gap"] and hebrew_letters(w["text"]) >= 3]
    tall = sum((w["box"][3] - w["box"][1]) > (w["box"][2] - w["box"][0]) for w in long_words)
    return bool(long_words) and tall / len(long_words) >= ROTATED_SHARE


def _union(boxes: Sequence[Sequence[float]]) -> List[float]:
    """Union box.

    :param boxes: ``(x0, y0, x1, y1)`` boxes.
    :type boxes: Sequence[Sequence[float]]
    :return: Their bounding box.
    :rtype: List[float]
    """
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def page_lines(items: List[dict]) -> Optional[Dict]:
    """Physical lines of a KTIV page with their exact words (geometry mode only).

    Mirrors :func:`ktiv_layout.reconstruct_page` (same column split, line
    clustering, interleave rescue and adaptive merge cut for the text) but keeps
    each line's words.

    :param items: AnnotationPage ``items``.
    :type items: List[dict]
    :return: ``{"columns": [[{"text", "box", "words": [{"box", "text"}]}]], "vertical": [box, ...]}``, or
        None for pages without usable geometry.
    :rtype: Optional[Dict]
    """
    words = extract_words(items)
    if not words:
        return None
    ycs = [(w["box"][1] + w["box"][3]) / 2 for w in words if not w["gap"]] or \
          [(w["box"][1] + w["box"][3]) / 2 for w in words]
    if (max(ycs) - min(ycs)) < DEGENERATE_Y_SPAN:
        return None
    box_h = _median_height(words)
    col_line_lists = [[seg for line in cluster_lines(col_words, box_h) for seg in _split_interleaved(line, box_h)]
                      for col_words in split_columns(words, box_h)]
    gaps_lh = [(prev["box"][0] - w["box"][2]) / box_h
               for col_lines in col_line_lists for line in col_lines
               for prev, w in zip(line, line[1:]) if not prev["gap"] and not w["gap"]]
    merge_cut = _page_merge_cut(gaps_lh)
    columns = []
    for col_lines in col_line_lists:
        entries = []
        for line in col_lines:
            text = " ".join(merge_line_words(line, box_h, merge_cut)).strip()
            geo = [w for w in line if not w["gap"] and not _is_vertical(w, box_h)]
            if text and geo:
                entries.append({"text": text, "box": _union([w["box"] for w in geo]),
                                "words": [{"box": list(w["box"]), "text": w["text"]} for w in geo]})
        if entries:
            columns.append(entries)
    return {"columns": columns,
            "vertical": [list(w["box"]) for w in words if not w["gap"] and _is_vertical(w, box_h)]}


def page_candidates(doc: Dict, ktiv_dir: Path, shingles: set, stats: Counter) -> List[Dict]:
    """Gated page records of one manuscript (no images written yet).

    :param doc: API-shape KTIV bundle.
    :type doc: Dict
    :param ktiv_dir: KTIV directory with the image zips.
    :type ktiv_dir: Path
    :param shingles: Benchmark shingles for manuscript-level decontamination.
    :type shingles: set
    :param stats: Counter updated in place with gate outcomes.
    :type stats: Counter
    :return: Page dicts with ``sys_num``, ``fl``, ``zip``, ``member``, ``size``, ``columns``, ``vertical``.
    :rtype: List[Dict]
    """
    sys_num = doc["sys_num"]
    pages = []
    for p in doc.get("pages") or []:
        items = ((p.get("annotation_page") or {}).get("items")) or []
        page = reconstruct_page(items)
        if page["lines"]:
            pages.append((p.get("fl") or "", page, items))
    if not pages:
        return []
    if shingles and any(B.shingle_hits(pg["text"], shingles) >= B.DECONTAM_MIN_HITS for _, pg, _ in pages):
        stats["ms_contaminated"] += 1
        return []
    stats["ms_kept"] += 1
    out = []
    for fl, page, items in pages:
        if page["mode"] != "geometry":
            stats["page_served_breaklines"] += 1
            continue
        if is_rotated(extract_words(items)):
            stats["page_rotated"] += 1
            continue
        geo = page_lines(items)
        if not geo or sum(len(c) for c in geo["columns"]) < MIN_LINES:
            stats["page_few_lines"] += 1
            continue
        if len(geo["columns"]) > MAX_COLUMNS:
            stats["page_many_columns"] += 1
            continue
        found = B.find_zip_member(ktiv_dir, sys_num, fl)
        if not found:
            stats["page_no_image"] += 1
            continue
        zpath, member = found
        with zipfile.ZipFile(zpath) as zf, zf.open(member) as fh:
            width, height = PILImage.open(fh).size          # header only; the zips sit on the NAS
        max_x = max(ln["box"][2] for col in geo["columns"] for ln in col)
        max_y = max(ln["box"][3] for col in geo["columns"] for ln in col)
        if max_x > width * 1.02 or max_y > height * 1.02 or min(width, height) < B.MIN_IMAGE_SIDE_PX:
            stats["page_out_of_frame"] += 1
            continue
        stats["page_candidate"] += 1
        out.append({"sys_num": sys_num, "fl": fl, "zip": str(zpath), "member": member,
                    "size": [width, height], **geo})
    return out


def scale_page(rec: Dict, s: float) -> Dict:
    """Scale every box of a candidate by ``s``.

    :param rec: Candidate page dict.
    :type rec: Dict
    :param s: Scale factor (<= 1).
    :type s: float
    :return: A copy with scaled ``columns`` and ``vertical`` boxes.
    :rtype: Dict
    """
    sc = lambda b: [round(v * s, 1) for v in b]  # noqa: E731
    cols = [[{"box": sc(ln["box"]), "text": ln["text"],
              "words": [{"box": sc(w["box"]), "text": w["text"]} for w in ln["words"]]}
             for ln in col] for col in rec["columns"]]
    return {**rec, "columns": cols, "vertical": [sc(b) for b in rec.get("vertical", [])]}


def candidates(out: Path, n_pages: int, per_ms: int, seed: int, val_fraction: float) -> Dict:
    """Step 1: gate, sample and split pages; export images and ``candidates.jsonl``.

    :param out: Dataset root (created).
    :type out: Path
    :param n_pages: Pages to sample in total.
    :type n_pages: int
    :param per_ms: Max pages per manuscript.
    :type per_ms: int
    :param seed: Sampling / split seed.
    :type seed: int
    :param val_fraction: Share of sampled manuscripts held out for validation.
    :type val_fraction: float
    :return: Summary dict (also written to ``candidates_stats.json``).
    :rtype: Dict
    """
    (out / "images").mkdir(parents=True, exist_ok=True)
    bundles = B.load_bundles(B.KTIV_DIR)
    bundles, excl = B.exclude_benchmark_manuscripts(bundles, B.KTIV_DIR)
    shingles = B.benchmark_shingles(B.BENCH_PATH) | B.benchmark_shingles(B.RELIGIOUS_BENCH_PATH)
    stats: Counter = Counter({f"excluded_{k}": v for k, v in excl.items()})
    rng = random.Random(seed)
    by_ms = {}
    for doc in bundles:
        recs = page_candidates(doc, B.KTIV_DIR, shingles, stats)
        if recs:
            rng.shuffle(recs)
            by_ms[doc["sys_num"]] = recs[:per_ms]
    pool = [r for recs in by_ms.values() for r in recs]
    rng.shuffle(pool)
    chosen = pool[:n_pages]
    val_ms = split_manuscripts(sorted({r["sys_num"] for r in chosen}), val_fraction, seed)
    with (out / "candidates.jsonl").open("w", encoding="utf-8") as fh:
        for k, rec in enumerate(chosen, 1):
            im = B.load_page_image(Path(rec["zip"]), rec["member"]).convert("RGB")
            s = min(1.0, MAX_H / im.height)
            if s < 1.0:
                im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), PILImage.LANCZOS)
            page_id = f"{rec['sys_num']}_{rec['fl']}"
            im.save(out / "images" / f"{page_id}.jpg", "JPEG", quality=JPEG_QUALITY)
            srec = scale_page(rec, s)
            srec.update(page_id=page_id, image=f"images/{page_id}.jpg", image_size=[im.width, im.height],
                        scale=s, split="val" if rec["sys_num"] in val_ms else "train")
            fh.write(json.dumps(srec, ensure_ascii=False) + "\n")
            if k % 100 == 0:
                log.info("exported %d/%d page images", k, len(chosen))
    ncols = Counter(len(r["columns"]) for r in chosen)
    summary = {"stats": dict(stats), "pool_pages": len(pool), "pool_manuscripts": len(by_ms),
               "chosen_pages": len(chosen), "val_manuscripts": len(val_ms),
               "val_pages": sum(r["sys_num"] in val_ms for r in chosen),
               "columns": {str(k): v for k, v in sorted(ncols.items())},
               "per_ms": per_ms, "seed": seed, "max_h": MAX_H}
    (out / "candidates_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("candidates: %s", json.dumps(summary))
    return summary


def relines(out: Path) -> Dict:
    """Recompute line geometry (:func:`page_lines`) for an existing ``candidates.jsonl``.

    Page selection, images, scale and split are kept; the previous file is saved as
    ``candidates.prev.jsonl``.  Pages that no longer yield :data:`MIN_LINES` lines are dropped.

    :param out: Dataset root.
    :type out: Path
    :return: Counts.
    :rtype: Dict
    """
    src = out / "candidates.jsonl"
    recs = [json.loads(l) for l in src.open(encoding="utf-8")]
    want = {(r["sys_num"], r["fl"]) for r in recs}
    items_by = {}
    for doc in B.load_bundles(B.KTIV_DIR):
        for p in doc.get("pages") or []:
            if (doc["sys_num"], p.get("fl") or "") in want:
                items_by[(doc["sys_num"], p.get("fl") or "")] = ((p.get("annotation_page") or {}).get("items")) or []
    stats: Counter = Counter()
    new = []
    for r in recs:
        geo = page_lines(items_by.get((r["sys_num"], r["fl"]), []))
        if not geo or sum(len(c) for c in geo["columns"]) < MIN_LINES:
            stats["dropped_few_lines"] += 1
            continue
        new.append(scale_page({**r, **geo}, r["scale"]))
        stats["kept"] += 1
    src.rename(out / "candidates.prev.jsonl")
    with src.open("w", encoding="utf-8") as fh:
        for r in new:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("relines: %s", dict(stats))
    return dict(stats)


# ---------------------------------------------------------------- baseline estimation

def ink_gray(im: PILImage.Image) -> np.ndarray:
    """Grayscale page with the blue backing board neutralised to background.

    KTIV scans (e.g. Cambridge T-S) show a dark-blue board through holes and
    around the fragment; in plain grayscale it reads as ink.

    :param im: Page image.
    :type im: PILImage.Image
    :return: uint8 2-D array.
    :rtype: np.ndarray
    """
    a = np.asarray(im.convert("RGB")).astype(np.int16)
    gray = a.mean(axis=2)
    gray[((a[..., 2] - a[..., 0]) > BLUE_MARGIN) & (a[..., 2] >= a[..., 1])] = 235
    return gray.astype(np.uint8)


def _body_bottom(page: np.ndarray, x0: int, x1: int, centre: float, pitch: float, method: str) -> Optional[float]:
    """Estimate the y of the letter-body bottom in one horizontal chunk of a line.

    :param page: Ink map from :func:`ink_gray`.
    :type page: np.ndarray
    :param x0: Chunk left edge, px.
    :type x0: int
    :param x1: Chunk right edge, px.
    :type x1: int
    :param centre: Ink centre of the chunk, px.
    :type centre: float
    :param pitch: Column line pitch, px.
    :type pitch: float
    :param method: ``quantile`` (row above which 90 % of the window's ink lies) or ``gradient``
        (steepest drop of the row-ink profile below the centre).
    :type method: str
    :return: Body-bottom y, px, or None when the chunk carries no ink.
    :rtype: Optional[float]
    """
    h = page.shape[0]
    r0 = max(0, int(centre - BODY_HALF * pitch))
    r1 = min(h, int(centre + BODY_HALF * pitch) + 1)
    if r1 - r0 < 4 or x1 - x0 < 2:
        return None
    dark = 255.0 - page[r0:r1, max(0, x0):x1].mean(axis=1)
    dark = np.clip(dark - np.percentile(dark, 20), 0, None)
    k = max(3, int(pitch // 10))
    dark = np.convolve(dark, np.ones(k) / k, mode="same")
    if dark.sum() <= 0:
        return None
    if method == "quantile":
        cum = np.cumsum(dark) / dark.sum()
        return float(r0 + int(np.searchsorted(cum, 0.9)))
    lo = max(0, int(round(centre)) - r0)
    grad = np.diff(dark)
    return float(r0 + lo + int(np.argmin(grad[lo:])) + 1) if lo < len(grad) else None


def chunk_estimates(page: np.ndarray, words: Sequence[Dict], pitch: float, method: str) -> List[Tuple[float, float]]:
    """Body-bottom samples ``(x, y)`` along one line or segment, one per word chunk.

    Words are grouped left→right into chunks at least :data:`CHUNK_PITCHES`
    pitches wide; each chunk is re-centred on its ink (:func:`ink_centre`,
    searched around the chunk's own box centre).

    :param page: Ink map.
    :type page: np.ndarray
    :param words: Word dicts (``box``).
    :type words: Sequence[Dict]
    :param pitch: Column line pitch, px.
    :type pitch: float
    :param method: See :func:`_body_bottom`.
    :type method: str
    :return: Chunk centre x and body-bottom y, for chunks with ink.
    :rtype: List[Tuple[float, float]]
    """
    ws = sorted((w["box"] for w in words), key=lambda b: b[0])
    chunks, cur = [], None
    for b in ws:
        cur = list(b) if cur is None else _union([cur, b])
        if cur[2] - cur[0] >= CHUNK_PITCHES * pitch:
            chunks.append(cur)
            cur = None
    if cur is not None:
        if chunks and cur[2] - cur[0] < 0.5 * CHUNK_PITCHES * pitch:
            cur = _union([chunks.pop(), cur])
        chunks.append(cur)
    out = []
    for c in chunks:
        x0, x1 = int(c[0]), int(c[2])
        yb = _body_bottom(page, x0, x1, ink_centre(page, x0, x1, (c[1] + c[3]) / 2, pitch), pitch, method)
        if yb is not None:
            out.append(((x0 + x1) / 2, yb))
    return out


def fixed_slope_fit(samples: Sequence[Tuple[float, float]], slope: float, tol: float) -> Tuple[Optional[float], float]:
    """Intercept of a line with known slope through chunk samples, robust to outliers.

    :param samples: ``(x, y)`` chunk estimates.
    :type samples: Sequence[Tuple[float, float]]
    :param slope: Page consensus slope.
    :type slope: float
    :param tol: Inlier tolerance, px.
    :type tol: float
    :return: ``(intercept or None, inlier share)``; the intercept is the mean over inliers.
    :rtype: Tuple[Optional[float], float]
    """
    if not samples:
        return None, 0.0
    resid = [y - slope * x for x, y in samples]
    med = statistics.median(resid)
    inl = [r for r in resid if abs(r - med) <= tol]
    return (sum(inl) / len(inl) if inl else None), len(inl) / len(resid)


def split_segments(words: Sequence[Dict], pitch: float) -> List[List[Dict]]:
    """Split a line's words left→right at horizontal gaps wider than :data:`SEG_GAP_PITCHES` pitches.

    :param words: Word dicts of one line.
    :type words: Sequence[Dict]
    :param pitch: Column pitch, px.
    :type pitch: float
    :return: Word groups, each becoming its own baseline.
    :rtype: List[List[Dict]]
    """
    ws = sorted(words, key=lambda w: w["box"][0])
    segs, right = [[ws[0]]], ws[0]["box"][2]
    for w in ws[1:]:
        if w["box"][0] - right > SEG_GAP_PITCHES * pitch:
            segs.append([w])
        else:
            segs[-1].append(w)
        right = max(right, w["box"][2])
    return segs


def robust_line(xs: Sequence[float], ys: Sequence[float], tol: float) -> Tuple[float, float, float]:
    """Outlier-robust straight line: Theil–Sen slope and median intercept, then a least-squares refit on inliers.

    :param xs: Abscissae, px.
    :type xs: Sequence[float]
    :param ys: Ordinates, px.
    :type ys: Sequence[float]
    :param tol: Inlier tolerance, px.
    :type tol: float
    :return: ``(slope, intercept, inlier share)``.
    :rtype: Tuple[float, float, float]
    """
    if len(xs) == 1:
        return 0.0, float(ys[0]), 1.0
    slopes = [(ys[j] - ys[i]) / (xs[j] - xs[i])
              for i in range(len(xs)) for j in range(i + 1, len(xs)) if xs[j] != xs[i]]
    m = statistics.median(slopes) if slopes else 0.0
    b = statistics.median(y - m * x for x, y in zip(xs, ys))
    inl = [(x, y) for x, y in zip(xs, ys) if abs(y - (m * x + b)) <= tol]
    if len(inl) >= 2 and len({x for x, _ in inl}) >= 2:
        m, b = (float(v) for v in np.polyfit([x for x, _ in inl], [y for _, y in inl], 1))
    return m, b, len(inl) / len(xs)


def _slope(pts: Sequence[Sequence[float]]) -> float:
    """End-to-end slope of a polyline.

    :param pts: Points sorted by x.
    :type pts: Sequence[Sequence[float]]
    :return: dy/dx (0 for vertical-degenerate input).
    :rtype: float
    """
    dx = pts[-1][0] - pts[0][0]
    return (pts[-1][1] - pts[0][1]) / dx if dx else 0.0


def _ltr(points: Sequence[Sequence[float]]) -> List[Tuple[int, int]]:
    """Polyline sorted left→right with strictly increasing x.

    :param points: Points.
    :type points: Sequence[Sequence[float]]
    :return: Cleaned points.
    :rtype: List[Tuple[int, int]]
    """
    out: List[Tuple[int, int]] = []
    for x, y in sorted((int(p[0]), int(p[1])) for p in points):
        if not out or x > out[-1][0]:
            out.append((x, y))
    return out


def ink_extent(ink: np.ndarray, x0: float, x1: float, pts: Sequence[Sequence[float]], pitch: float) -> Tuple[int, int]:
    """Horizontal extent of the ink sitting on a baseline, searched within ``[x0, x1]``.

    Column darkness is measured in the band 0.6 pitch above the baseline;
    leading / trailing columns below :data:`INK_COL_FRAC` of the maximum are
    trimmed.  Falls back to ``[x0, x1]`` when the band carries no ink or the
    trimmed extent would be under half of it.

    :param ink: Ink map.
    :type ink: np.ndarray
    :param x0: Left search limit, px.
    :type x0: float
    :param x1: Right search limit, px.
    :type x1: float
    :param pts: Baseline (any x-extent; extrapolated linearly by :func:`y_at_ext`).
    :type pts: Sequence[Sequence[float]]
    :param pitch: Column pitch, px.
    :type pitch: float
    :return: ``(left, right)`` px.
    :rtype: Tuple[int, int]
    """
    h, w = ink.shape
    a, b = max(0, int(x0)), min(w, int(x1) + 1)
    if b - a < 4:
        return int(x0), int(x1)
    dark = np.zeros(b - a)
    for x in range(a, b, 1):
        y = y_at_ext(pts, x)
        r0, r1 = max(0, int(y - 0.6 * pitch)), min(h, int(y) + 1)
        if r1 > r0:
            dark[x - a] = 255.0 - ink[r0:r1, x].mean()
    dark = np.clip(dark - np.percentile(dark, 10), 0, None)
    k = max(3, int(pitch // 4))
    dark = np.convolve(dark, np.ones(k) / k, mode="same")
    if dark.max() <= 0:
        return int(x0), int(x1)
    cols = np.nonzero(dark >= INK_COL_FRAC * dark.max())[0]
    lo, hi = a + int(cols[0]), a + int(cols[-1])
    return (lo, hi) if hi - lo >= 0.5 * (x1 - x0) else (int(x0), int(x1))


def y_at_ext(pts: Sequence[Sequence[float]], x: float) -> float:
    """Polyline y at ``x``, extrapolating the end segments linearly outside its x-range.

    :param pts: Points sorted by x (≥ 2).
    :type pts: Sequence[Sequence[float]]
    :param x: Abscissa, px.
    :type x: float
    :return: y, px.
    :rtype: float
    """
    if x < pts[0][0]:
        (xa, ya), (xb, yb) = pts[0], pts[1]
    elif x > pts[-1][0]:
        (xa, ya), (xb, yb) = pts[-2], pts[-1]
    else:
        return y_at(pts, x)
    return ya if xb == xa else ya + (yb - ya) * (x - xa) / (xb - xa)


def extend_to_words(pts: List[Tuple[int, int]], words: Sequence[Dict], ink: np.ndarray, pitch: float) -> List[Tuple[int, int]]:
    """Extend an adopted blla2026 baseline to the ink of its GT line's outermost words.

    blla2026 often stops a word short of the line end; the GT words say the
    line goes on.  Each end is extended (linearly) to the ink extent inside the
    GT word span, never shortened.

    :param pts: blla2026 baseline, left→right.
    :type pts: List[Tuple[int, int]]
    :param words: The GT line's words.
    :type words: Sequence[Dict]
    :param ink: Ink map.
    :type ink: np.ndarray
    :param pitch: Column pitch, px.
    :type pitch: float
    :return: Baseline points, left→right.
    :rtype: List[Tuple[int, int]]
    """
    wx0, wx1 = min(w["box"][0] for w in words), max(w["box"][2] for w in words)
    if pts[0][0] <= wx0 + 0.3 * pitch and pts[-1][0] >= wx1 - 0.3 * pitch:
        return pts
    lo, hi = ink_extent(ink, wx0, wx1, pts, pitch)
    out = list(pts)
    if lo < out[0][0] - 0.3 * pitch:
        out.insert(0, (int(lo), int(round(y_at_ext(pts, lo)))))
    if hi > out[-1][0] + 0.3 * pitch:
        out.append((int(hi), int(round(y_at_ext(pts, hi)))))
    return out


# ---------------------------------------------------------------- per-page targets

def page_pitch(rec: Dict) -> Tuple[float, List[float], float]:
    """Page and column pitches plus the page's median word height.

    :param rec: Candidate page.
    :type rec: Dict
    :return: ``(page pitch, [column pitch, ...], median word height)``.
    :rtype: Tuple[float, List[float], float]
    """
    hs = sorted(ln["box"][3] - ln["box"][1] for col in rec["columns"] for ln in col)
    h_med = hs[len(hs) // 2]
    pitches = [column_pitch([{"box": ln["box"]} for ln in col], h_med) for col in rec["columns"]]
    wh = statistics.median(w["box"][3] - w["box"][1] for col in rec["columns"] for ln in col for w in ln["words"])
    return float(statistics.median(pitches)), pitches, float(wh)


def page_targets(rec: Dict, pred: Dict, ink: np.ndarray, method: str, shift: float) -> Dict:
    """Baseline targets of one page, plus the lines to paint out and calibration samples.

    :param rec: Candidate page.
    :type rec: Dict
    :param pred: ``seg_gate.py`` output for the page.
    :type pred: Dict
    :param ink: Ink map of the page image.
    :type ink: np.ndarray
    :param method: Body-bottom estimator (see :func:`_body_bottom`).
    :type method: str
    :param shift: Systematic estimator − blla2026 offset, in pitches (subtracted from estimates).
    :type shift: float
    :return: ``{"segments": [{"ci", "li", "pts", "source", "text"}], "failed": {(ci, li): reason},
        "calib": [offset in pitches, ...], "page_slope", "n_blla", "n_est"}``.
    :rtype: Dict
    """
    page_p, pitches, page_wh = page_pitch(rec)
    flat = [(ci, li, ln) for ci, col in enumerate(rec["columns"]) for li, ln in enumerate(col)]
    preds = [_ltr(ln["baseline"]) for ln in pred["lines"]]
    matches = match_lines(preds, [tuple(ln["box"]) for _, _, ln in flat], page_p,
                          min_share=MATCH_MIN_SHARE, min_span=MATCH_MIN_SPAN)
    adopted = {g: preds[j] for g, j in enumerate(matches) if j is not None and len(preds[j]) >= 2}
    slopes = [_slope(p) for p in adopted.values()]
    if len(slopes) < 2:                 # few blla-matched lines: the page's own robust per-line slopes
        for ci, li, ln in flat:
            samples = chunk_estimates(ink, ln["words"], pitches[ci], method)
            if len(samples) >= 3:
                m, _, share = robust_line([x for x, _ in samples], [y for _, y in samples], FIT_TOL * pitches[ci])
                if share >= 0.8:
                    slopes.append(m)
    page_slope = statistics.median(slopes) if len(slopes) >= 2 else 0.0
    segments, failed, calib = [], {}, []
    refits = 0
    for g, (ci, li, ln) in enumerate(flat):
        p = pitches[ci]
        if g in adopted:
            segments.append({"ci": ci, "li": li, "pts": extend_to_words(adopted[g], ln["words"], ink, p),
                             "source": "blla", "text": ln["text"]})
            samples = chunk_estimates(ink, ln["words"], p, method)
            b, share = fixed_slope_fit(samples, page_slope, FIT_TOL * p)
            if b is not None and share >= FIT_MIN_INLIERS:
                d = [(page_slope * x + b) - yb for x, _ in samples if (yb := y_at(adopted[g], x)) is not None]
                if d:
                    calib.append(statistics.median(d) / p)
            continue
        if statistics.median(w["box"][3] - w["box"][1] for w in ln["words"]) > TALL_BOXES * page_wh:
            failed[(ci, li)] = "tall_boxes"
            continue
        segs = []
        for seg in split_segments(ln["words"], p):
            x0, x1 = min(w["box"][0] for w in seg), max(w["box"][2] for w in seg)
            if x1 - x0 < 0.5 * p:
                continue                       # a lone letter across a hole: too short to be a line
            samples = chunk_estimates(ink, seg, p, method)
            b, share = fixed_slope_fit(samples, page_slope, FIT_TOL * p)
            if b is None or share < FIT_MIN_INLIERS:
                segs = None
                break
            slope = page_slope
            ends = [min(samples), max(samples)]
            if max(abs(yy - (slope * xx + b)) for xx, yy in ends) > END_TOL * p:
                # this line sits at its own angle (warped / drooping / separate piece): fit it on its own
                if len(samples) < 2:
                    segs = None
                    break
                m, b2, share2 = robust_line([x for x, _ in samples], [y for _, y in samples], FIT_TOL * p)
                if share2 < 0.8 or abs(m - page_slope) > FREE_SLOPE_MAX_DEV:
                    segs = None
                    break
                slope, b = m, b2
                refits += 1
            y = lambda x, s_=slope, b_=b: s_ * x + b_ - shift * p  # noqa: E731
            line_pts = [(x0, y(x0)), (x1, y(x1))]
            cx0, cx1 = ink_extent(ink, x0, x1, line_pts, p)
            segs.append({"ci": ci, "li": li, "pts": [(int(cx0), int(round(y(cx0)))), (int(cx1), int(round(y(cx1))))],
                         "source": "est", "text": " ".join(w["text"] for w in sorted(seg, key=lambda w: -w["box"][2]))})
        if not segs:
            failed[(ci, li)] = "fit"
            continue
        segments.extend(segs)
    # crowding: two kept baselines overlapping in x closer than MIN_LINE_GAP pitches -> fail the smaller line
    by_col = defaultdict(list)
    for s in segments:
        by_col[s["ci"]].append(s)
    for ci, segs in by_col.items():
        for i, a in enumerate(segs):
            for b in segs[i + 1:]:
                if a["li"] == b["li"]:
                    continue
                lo, hi = max(a["pts"][0][0], b["pts"][0][0]), min(a["pts"][-1][0], b["pts"][-1][0])
                if hi <= lo:
                    continue
                xm = (lo + hi) / 2
                if abs(y_at(a["pts"], xm) - y_at(b["pts"], xm)) < MIN_LINE_GAP * pitches[ci]:
                    la = hebrew_letters(rec["columns"][ci][a["li"]]["text"])
                    lb = hebrew_letters(rec["columns"][ci][b["li"]]["text"])
                    if min(la, lb) > 0.5 * max(la, lb):
                        failed.setdefault((ci, a["li"]), "crowded")
                        failed.setdefault((ci, b["li"]), "crowded")
                    else:
                        failed.setdefault((ci, a["li"] if la < lb else b["li"]), "crowded")
    segments = [s for s in segments if (s["ci"], s["li"]) not in failed]
    return {"segments": segments, "failed": failed, "calib": calib, "page_slope": page_slope,
            "n_blla": sum(s["source"] == "blla" for s in segments), "n_est": sum(s["source"] == "est" for s in segments),
            "n_refit": refits}


def uncovered_writing(pred: Dict, page_p: float) -> List[List[List[int]]]:
    """Boundaries of blla2026 lines outside every GT line that read as real writing.

    :param pred: ``seg_gate.py`` output.
    :type pred: Dict
    :param page_p: Page pitch, px.
    :type page_p: float
    :return: Polygons to paint out.
    :rtype: List[List[List[int]]]
    """
    out = []
    for ln in pred["lines"]:
        if ln.get("covered") or not ln.get("text") or not ln.get("boundary"):
            continue
        bx = polyline_box(ln["baseline"])
        if (hebrew_letters(ln["text"]) >= UNCOVERED_MIN_LETTERS and (ln.get("conf") or 0) >= UNCOVERED_MIN_CONF
                and bx[2] - bx[0] >= UNCOVERED_MIN_PITCHES * page_p):
            out.append(ln["boundary"])
    return out


def paint(im: PILImage.Image, rec: Dict, pred: Dict, targets: Dict, extra_polys: List) -> Tuple[PILImage.Image, int]:
    """Paint failed lines, vertical marginal words and untranscribed writing out of the page.

    Failed lines are covered by their word boxes (padded 0.25 pitch vertically, 0.15
    horizontally) plus the blla2026 line polygons lying mostly inside that area;
    nothing inside the text bands of kept baselines (−0.7 … +0.25 pitch around
    them) is touched.  The fill is the parchment colour of the text regions
    (non-blue pixels between the 50th and 90th brightness percentile).

    :param im: Page image.
    :type im: PILImage.Image
    :param rec: Candidate page.
    :type rec: Dict
    :param pred: blla2026 predictions.
    :type pred: Dict
    :param targets: :func:`page_targets` output.
    :type targets: Dict
    :param extra_polys: Additional polygons to paint (untranscribed writing).
    :type extra_polys: List
    :return: ``(painted image, painted pixel count)``.
    :rtype: Tuple[PILImage.Image, int]
    """
    a = np.asarray(im.convert("RGB")).copy()
    h, w = a.shape[:2]
    page_p, pitches, _ = page_pitch(rec)
    paint_m = PILImage.new("1", (w, h), 0)
    dp = ImageDraw.Draw(paint_m)
    for (ci, li) in targets["failed"]:
        p = pitches[ci]
        ln = rec["columns"][ci][li]
        area = _union([w_["box"] for w_ in ln["words"]])
        for wd in ln["words"]:
            b = wd["box"]
            dp.rectangle([b[0] - 0.15 * p, b[1] - 0.25 * p, b[2] + 0.15 * p, b[3] + 0.25 * p], fill=1)
        for pl in pred["lines"]:
            if pl.get("boundary") and covered_share(pl["baseline"], [tuple(area)], 0.3 * p) >= 0.5:
                dp.polygon([tuple(q) for q in pl["boundary"]], fill=1)
    for b in rec.get("vertical", []):
        dp.rectangle([b[0] - 0.1 * page_p, b[1] - 0.1 * page_p, b[2] + 0.1 * page_p, b[3] + 0.1 * page_p], fill=1)
    for poly in extra_polys:
        dp.polygon([tuple(q) for q in poly], fill=1)
    protect_m = PILImage.new("1", (w, h), 0)
    dq = ImageDraw.Draw(protect_m)
    for s in targets["segments"]:
        p = pitches[s["ci"]]
        pts = s["pts"]
        dq.polygon([(x, y - 0.7 * p) for x, y in pts] + [(x, y + 0.25 * p) for x, y in reversed(pts)], fill=1)
    mask = np.asarray(paint_m, dtype=bool) & ~np.asarray(protect_m, dtype=bool)
    n = int(mask.sum())
    if n:
        gray = a.mean(axis=2)
        region = np.zeros((h, w), bool)
        for ci, col in enumerate(rec["columns"]):
            x0, y0, x1, y1 = region_box(col, pitches[ci], (w, h))
            region[y0:y1 + 1, x0:x1 + 1] = True
        not_blue = ~(((a[..., 2].astype(int) - a[..., 0]) > BLUE_MARGIN) & (a[..., 2] >= a[..., 1]))
        sel = region & not_blue
        if sel.sum() > 100:
            lo, hi = np.percentile(gray[sel], [50, 90])
            sel &= (gray >= lo) & (gray <= hi)
        fill = np.median(a[sel] if sel.sum() else a.reshape(-1, 3), axis=0).astype(np.uint8)
        a[mask] = fill
    return PILImage.fromarray(a), n


# ---------------------------------------------------------------- write

def region_box(col: Sequence[Dict], pitch: float, size: Sequence[int]) -> Tuple[int, int, int, int]:
    """Text-region rectangle of a column: its line boxes' union padded by 0.3 pitch.

    :param col: Column lines.
    :type col: Sequence[Dict]
    :param pitch: Column pitch, px.
    :type pitch: float
    :param size: ``[width, height]`` of the page image.
    :type size: Sequence[int]
    :return: Clipped integer box.
    :rtype: Tuple[int, int, int, int]
    """
    x0, y0, x1, y1 = _union([ln["box"] for ln in col])
    pad = 0.3 * pitch
    return (max(0, int(x0 - pad)), max(0, int(y0 - pad)), min(size[0] - 1, int(x1 + pad)), min(size[1] - 1, int(y1 + pad)))


def _pts(points: Sequence[Sequence[float]]) -> str:
    """PageXML ``points`` attribute.

    :param points: ``[(x, y), ...]``.
    :type points: Sequence[Sequence[float]]
    :return: ``"x,y x,y ..."``.
    :rtype: str
    """
    return " ".join(f"{int(round(x))},{int(round(y))}" for x, y in points)


def _rect(box: Sequence[float]) -> List[Tuple[float, float]]:
    """Clockwise rectangle polygon of a box.

    :param box: ``(x0, y0, x1, y1)``.
    :type box: Sequence[float]
    :return: Four corner points.
    :rtype: List[Tuple[float, float]]
    """
    x0, y0, x1, y1 = box
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def page_xml(rec: Dict, image: str, segments: Sequence[Dict], pitches: Sequence[float]) -> str:
    """Serialise one page as PageXML 2019 (one region per column, one line per baseline segment).

    Line ``Coords`` are the band −0.7 … +0.3 pitch around the baseline (segtrain
    ignores them; they only keep the file valid).

    :param rec: Candidate page.
    :type rec: Dict
    :param image: Image path relative to the dataset root.
    :type image: str
    :param segments: Kept baseline segments.
    :type segments: Sequence[Dict]
    :param pitches: Column pitches.
    :type pitches: Sequence[float]
    :return: XML text.
    :rtype: str
    """
    w, h = rec["image_size"]
    clip = lambda x, y: (min(max(x, 0), w - 1), min(max(y, 0), h - 1))  # noqa: E731
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    parts = [f'<?xml version="1.0" encoding="UTF-8"?>\n<PcGts xmlns="{PAGE_NS}">',
             f"<Metadata><Creator>export_ktiv_pagexml</Creator><Created>{now}</Created>"
             f"<LastChange>{now}</LastChange></Metadata>",
             # kraken resolves imageFilename against the XML's own directory (pagexml/)
             f'<Page imageFilename="../{escape(image)}" imageWidth="{w}" imageHeight="{h}">']
    for ci, col in enumerate(rec["columns"]):
        parts.append(f'<TextRegion id="r{ci}"><Coords points="{_pts(_rect(region_box(col, pitches[ci], (w, h))))}"/>')
        for k, s in enumerate(x for x in segments if x["ci"] == ci):
            p = pitches[ci]
            bl = [clip(x, y) for x, y in s["pts"]]
            band = [clip(x, y - 0.7 * p) for x, y in bl] + [clip(x, y + 0.3 * p) for x, y in reversed(bl)]
            parts.append(f'<TextLine id="r{ci}l{k}"><Coords points="{_pts(band)}"/><Baseline points="{_pts(bl)}"/>'
                         f"<TextEquiv><Unicode>{escape(s['text'])}</Unicode></TextEquiv></TextLine>")
        parts.append("</TextRegion>")
    parts.append("</Page></PcGts>\n")
    return "\n".join(parts)


def overlay(im: PILImage.Image, rec: Dict, targets: Dict, pred: Optional[Dict], dest: Path) -> None:
    """QA image: KTIV line boxes (blue), targets (red = blla-adopted, orange = estimated),
    blla2026 baselines (green), painted-out lines (magenta box).

    :param im: The training image (painted, if any painting happened).
    :type im: PILImage.Image
    :param rec: Candidate page.
    :type rec: Dict
    :param targets: :func:`page_targets` output.
    :type targets: Dict
    :param pred: blla2026 predictions.
    :type pred: Optional[Dict]
    :param dest: Output JPEG path.
    :type dest: Path
    """
    im = im.convert("RGB").copy()
    dr = ImageDraw.Draw(im)
    for ci, col in enumerate(rec["columns"]):
        for li, ln in enumerate(col):
            dr.rectangle(ln["box"], outline=(255, 0, 255) if (ci, li) in targets["failed"] else (40, 90, 255),
                         width=3 if (ci, li) in targets["failed"] else 2)
    for ln in (pred or {}).get("lines", []):
        dr.line([tuple(p) for p in ln["baseline"]], fill=(0, 200, 0), width=2)
    for s in targets["segments"]:
        dr.line([tuple(p) for p in s["pts"]], fill=(255, 0, 0) if s["source"] == "blla" else (255, 140, 0), width=4)
    im.thumbnail((1600, 1600))
    im.save(dest, "JPEG", quality=85)


def write(out: Path, method: str, n_overlays: int, max_w: int) -> Dict:
    """Step 3: targets, painting, PageXML and manifests (see the module docstring).

    :param out: Dataset root holding ``candidates.jsonl``, ``images/`` and ``preds/``.
    :type out: Path
    :param method: Body-bottom estimator ``quantile`` / ``gradient``, or ``auto`` (smaller residual
        spread against blla2026 on matched lines).
    :type method: str
    :param n_overlays: QA overlays per category.
    :type n_overlays: int
    :param max_w: Pages wider than this are rejected (training memory).
    :type max_w: int
    :return: Summary dict (also ``stats.json``).
    :rtype: Dict
    """
    recs = [json.loads(l) for l in (out / "candidates.jsonl").open(encoding="utf-8")]
    stats: Counter = Counter()
    pages = []
    for rec in recs:
        if rec["image_size"][0] > max_w:
            stats["page_rejected_too_wide"] += 1
            continue
        pf = out / "preds" / f"{rec['page_id']}.json"
        if not pf.exists():
            stats["page_no_pred"] += 1
            continue
        pages.append((rec, json.loads(pf.read_text())))
    # pass 1: calibration of the estimator against blla2026 on 1:1-matched lines
    offsets = defaultdict(list)
    methods = ("quantile", "gradient") if method == "auto" else (method,)
    inks = {}
    for rec, pred in pages:
        inks[rec["page_id"]] = ink = ink_gray(PILImage.open(out / rec["image"]))
        for m in methods:
            offsets[m].extend(page_targets(rec, pred, ink, m, 0.0)["calib"])
    calib = {}
    for m, v in offsets.items():
        med = statistics.median(v) if v else 0.0
        calib[m] = {"n": len(v), "median": med,
                    "spread": statistics.median(abs(x - med) for x in v) if v else None,
                    "p90_abs_resid": sorted(abs(x - med) for x in v)[int(0.9 * (len(v) - 1))] if v else None}
    chosen = min(calib, key=lambda m: calib[m]["spread"] if calib[m]["spread"] is not None else 1e9)
    shift = calib[chosen]["median"]
    for sub in ("pagexml", "qa", "images_masked"):      # outputs of an earlier write pass are rebuilt from scratch
        (out / sub).mkdir(exist_ok=True)
        for f in (out / sub).iterdir():                # files only: removing the dir itself fails on the SMB NAS
            if not f.name.startswith("."):             # .smbdelete* = SMB pending-delete placeholders
                f.unlink()
    manifests = {"train": [], "val": []}
    shown = Counter()
    for rec, pred in pages:
        page_id = rec["page_id"]
        t = page_targets(rec, pred, inks.pop(page_id), chosen, shift)
        n_lines = sum(len(col) for col in rec["columns"])
        for reason in t["failed"].values():
            stats[f"lines_failed_{reason}"] += 1
        im = PILImage.open(out / rec["image"])
        if len(t["failed"]) > max(MASK_MAX_LINES, MASK_MAX_SHARE * n_lines) or not t["segments"]:
            stats["page_rejected_geometry"] += 1
            if shown["rejected"] < n_overlays:
                shown["rejected"] += 1
                overlay(im, rec, t, pred, out / "qa" / f"rejected_{page_id}.jpg")
            continue
        page_p, pitches, _ = page_pitch(rec)
        unc = uncovered_writing(pred, page_p)
        image = rec["image"]
        painted_im, n_px = paint(im, rec, pred, t, unc) if (t["failed"] or rec.get("vertical") or unc) else (im, 0)
        if n_px:
            image = f"images_masked/{page_id}.jpg"
            painted_im.save(out / image, "JPEG", quality=JPEG_QUALITY)
            stats["pages_painted"] += 1
            stats["lines_painted"] += len(t["failed"])
            stats["uncovered_writing_painted"] += len(unc)
            stats["vertical_words_painted"] += len(rec.get("vertical", []))
        (out / "pagexml" / f"{page_id}.xml").write_text(page_xml(rec, image, t["segments"], pitches), encoding="utf-8")
        manifests[rec["split"]].append(f"pagexml/{page_id}.xml")
        stats[f"pages_written_{rec['split']}"] += 1
        stats["gt_lines_kept"] += n_lines - len(t["failed"])
        stats["segments_blla"] += t["n_blla"]
        stats["segments_est"] += t["n_est"]
        stats["segments_est_refit_own_slope"] += t["n_refit"]
        cat = "painted" if n_px else rec["split"]
        if shown[cat] < n_overlays:
            shown[cat] += 1
            overlay(painted_im, rec, t, pred, out / "qa" / f"{cat}_{page_id}.jpg")
    # manifests are relative to the dataset root, so they hold at any mount point (ketos runs with cwd = root)
    for split, paths in manifests.items():
        (out / f"{split}.lst").write_text("\n".join(sorted(paths)) + "\n", encoding="utf-8")
    summary = {"stats": dict(stats), "baseline_method": chosen, "calibration": calib, "shift_pitches": shift,
               "params": {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float))}}
    (out / "stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("write: %s", json.dumps(summary["stats"]))
    return summary


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--pages", type=int, default=2000)
    c.add_argument("--per-ms", type=int, default=3)
    c.add_argument("--seed", type=int, default=SPLIT_SEED)
    c.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    r = sub.add_parser("relines")
    r.add_argument("--out", type=Path, required=True)
    w = sub.add_parser("write")
    w.add_argument("--out", type=Path, required=True)
    w.add_argument("--method", choices=("auto", "gradient", "quantile"), default="auto")
    w.add_argument("--overlays", type=int, default=12)
    w.add_argument("--max-w", type=int, default=MAX_W)
    a = p.parse_args()
    if a.cmd == "candidates":
        candidates(a.out, a.pages, a.per_ms, a.seed, a.val_fraction)
    elif a.cmd == "relines":
        relines(a.out)
    else:
        write(a.out, a.method, a.overlays, a.max_w)


if __name__ == "__main__":
    main()
