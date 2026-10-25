# File name: filter_aligned_lines.py
# Date: 10/20/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""High-precision subset of a forced-alignment run by FILTERING placed lines (no re-alignment).

``align_edition_lines.py`` writes, per run, ``page_xml/<group>/<stem>.xml`` (kraken PAGE XML; a placed
line is a ``<TextLine>`` whose ``<Unicode>`` holds the edition text, a rejected line keeps an empty
``<Unicode>``), ``pairs.jsonl`` (one row per edition line, ``written`` true for placed lines, with the
geometry diagnostics used here: ``extended_lh_right``, ``block_limited`` ``[left, right]``,
``own_ink_outside``, ``lost``, ``line_height``, ``segments``) and ``lists/genizah_{train,val}.lst``.

``filter`` copies the run's page XML into a NEW out-root and removes the ``<TextLine>`` elements of the
dropped placed lines (everything else, including ``imageFilename`` and the empty rejected lines, is kept).
A placed line is dropped when (:func:`drop_reasons`, first matching reason in this order):

* ``page``: its page is in the drop-page list (the whole page is left out);
* ``named``: ``(stem, line_id)`` is in the drop-line list;
* ``lost``: ``lost`` above ``max_lost``;
* ``own_ink_outside``: ``own_ink_outside`` above ``max_own_ink_outside``;
* ``ext_right``: ``extended_lh_right`` at least ``max_ext_right_lh``;
* ``block_right``: the right end is block limited and ``extended_lh_right`` is at least ``block_ext_right_lh``;
* ``neighbour_ink``: ink that belongs to another segmenter baseline covers at least ``neighbour_ink_max``
  line heights squared inside the polygon (:func:`neighbour_ink_area`, a port of the PGP geometry audit's
  ``nbbase.py``).

A page left without placed lines is not written. ``lists/{train,val}.lst`` keep the source run's split of
each page; ``filter_report.json`` holds the rules, kept/dropped counts per reason (first reason and any
reason), letters kept and one decision row per placed line.

``filter-custom`` handles an older run without ``pairs.jsonl`` (e.g. ``v1_align2``): the per-line record is the
TextLine ``custom`` attribute (``alignment {tier:anchor; sim:0.74; ...; lost:0.00;}``); a placed line is kept
only when its tier is in ``--keep-tiers``, ``lost`` is present and at most ``--max-lost`` and its id is not a
merged one (no underscore, :func:`custom_drop_reasons`).

All list files are written in one go (hidden temp file, then ``os.replace``) and only after the page XML they
name is complete, so a watcher never sees a partial list.

``combine`` concatenates the lists of several filtered roots (``--list NAME=ROOT[,ROOT...]``) into
``<out-dir>/NAME_{train,val}.lst`` and writes ``totals.json`` (pages, lines, Arabic letters U+0621 to U+064A
per list, counted from the XML)::

    python -m src.finetuning.kraken.filter_aligned_lines filter --run RUN --out-root OUT \\
        --drop-pages pgpar_8270_0_page --drop-lines pgpar_8652_0_page:l6 --max-ext-right-lh 0.8 \\
        --block-ext-right-lh 0.3 --max-own-ink-outside 0.10 --max-lost 0 --neighbour-ink-max 0.02
    python -m src.finetuning.kraken.filter_aligned_lines combine --out-dir LISTS \\
        --list pgp_only=OUT_PGP --list pgp_khan=OUT_PGP,OUT_KHAN
"""
import argparse
import json
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
NS = "{%s}" % PAGE_NS
SPLITS = ("train", "val")
ARABIC_FIRST = 0x0621
ARABIC_LAST = 0x064A
REASON_ORDER = ("page", "named", "lost", "own_ink_outside", "ext_right", "block_right", "neighbour_ink")
DEFAULT_LINE_HEIGHT = 60.0

LineKey = Tuple[str, str]
Point = Tuple[int, int]

ET.register_namespace("", PAGE_NS)
ET.register_namespace("xsi", XSI_NS)


@dataclass
class FilterRules:
    """Drop rules for placed lines; a threshold of ``None`` switches that rule off.

    :param drop_pages: page stems left out entirely.
    :param drop_lines: ``(stem, line_id)`` pairs dropped by name.
    :param max_lost: drop when ``lost`` is above this.
    :param max_own_ink_outside: drop when ``own_ink_outside`` is above this.
    :param max_ext_right_lh: drop when ``extended_lh_right`` is at least this.
    :param block_ext_right_lh: drop when the right end is block limited and ``extended_lh_right`` is at least this.
    :param neighbour_ink_max: drop when the neighbour-assigned ink inside the polygon (lh^2) is at least this.
    """

    drop_pages: Set[str] = field(default_factory=set)
    drop_lines: Set[LineKey] = field(default_factory=set)
    max_lost: Optional[float] = 0.0
    max_own_ink_outside: Optional[float] = 0.10
    max_ext_right_lh: Optional[float] = None
    block_ext_right_lh: Optional[float] = None
    neighbour_ink_max: Optional[float] = None

    def to_json(self) -> dict:
        """JSON-ready copy (sets as sorted lists).

        :return: dict of the rules.
        """
        d = asdict(self)
        d["drop_pages"] = sorted(self.drop_pages)
        d["drop_lines"] = sorted("%s:%s" % k for k in self.drop_lines)
        return d


def parse_line_key(spec: str) -> LineKey:
    """Parse ``STEM:LINE_ID`` into a key.

    :param spec: e.g. ``pgpar_8652_0_page:l6``.
    :return: ``(stem, line_id)``.
    """
    stem, sep, lid = spec.rpartition(":")
    if not sep or not stem or not lid:
        raise ValueError("line spec must be STEM:LINE_ID, got %r" % spec)
    return stem, lid


def load_pairs(path: Path) -> List[dict]:
    """Read a run's ``pairs.jsonl``.

    :param path: file path.
    :return: rows in file order.
    """
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def placed_rows(rows: Iterable[dict]) -> Dict[LineKey, dict]:
    """Placed (written) lines keyed by ``(stem, line_id)``.

    :param rows: pairs rows.
    :return: mapping in file order.
    """
    return {(r["stem"], r["line_id"]): r for r in rows if r.get("written")}


def _num(row: dict, key: str) -> float:
    """Numeric field with ``None``/missing read as 0.

    :param row: pairs row.
    :param key: field name.
    :return: float value.
    """
    return float(row.get(key) or 0.0)


def drop_reasons(row: dict, rules: FilterRules, neighbour_ink: Optional[float] = None) -> List[str]:
    """Every rule a placed line trips, in :data:`REASON_ORDER`.

    :param row: pairs row of a placed line.
    :param rules: drop rules.
    :param neighbour_ink: neighbour-assigned ink inside the polygon (lh^2), or ``None`` when not measured.
    :return: reasons (empty list when the line is kept).
    """
    out = []
    key = (row["stem"], row["line_id"])
    ext_r = _num(row, "extended_lh_right")
    limited = row.get("block_limited") or [False, False]
    if row["stem"] in rules.drop_pages:
        out.append("page")
    if key in rules.drop_lines:
        out.append("named")
    if rules.max_lost is not None and _num(row, "lost") > rules.max_lost:
        out.append("lost")
    if rules.max_own_ink_outside is not None and _num(row, "own_ink_outside") > rules.max_own_ink_outside:
        out.append("own_ink_outside")
    if rules.max_ext_right_lh is not None and ext_r >= rules.max_ext_right_lh:
        out.append("ext_right")
    if rules.block_ext_right_lh is not None and bool(limited[1]) and ext_r >= rules.block_ext_right_lh:
        out.append("block_right")
    if rules.neighbour_ink_max is not None and neighbour_ink is not None and neighbour_ink >= rules.neighbour_ink_max:
        out.append("neighbour_ink")
    return out


def decide(placed: Dict[LineKey, dict], rules: FilterRules,
           neighbour: Optional[Dict[LineKey, float]] = None) -> Dict[LineKey, List[str]]:
    """Drop reasons of every placed line.

    :param placed: placed lines keyed by ``(stem, line_id)``.
    :param rules: drop rules.
    :param neighbour: measured neighbour ink per line (lh^2); missing lines are not judged by that rule.
    :return: mapping key -> reasons (empty = kept).
    """
    neighbour = neighbour or {}
    return {k: drop_reasons(r, rules, neighbour.get(k)) for k, r in placed.items()}


def tally(decisions: Dict[LineKey, List[str]]) -> Dict[str, Dict[str, int]]:
    """Dropped counts per reason, attributed to the first reason and to any reason.

    :param decisions: mapping key -> reasons.
    :return: ``{"first": {...}, "any": {...}}`` with every reason of :data:`REASON_ORDER` present.
    """
    first = Counter(rs[0] for rs in decisions.values() if rs)
    anyr = Counter(r for rs in decisions.values() for r in rs)
    return {"first": {r: first.get(r, 0) for r in REASON_ORDER}, "any": {r: anyr.get(r, 0) for r in REASON_ORDER}}


def arabic_letters(text: str) -> int:
    """Count of Arabic letters U+0621 to U+064A.

    :param text: line text.
    :return: letter count.
    """
    return sum(1 for ch in text if ARABIC_FIRST <= ord(ch) <= ARABIC_LAST)


def remove_text_lines(xml_bytes: bytes, drop_ids: Set[str]) -> Tuple[bytes, List[str]]:
    """PAGE XML with the ``<TextLine>`` elements of ``drop_ids`` removed and everything else kept.

    :param xml_bytes: source document.
    :param drop_ids: TextLine ids to remove.
    :return: new document bytes and the ids actually removed.
    """
    root = ET.fromstring(xml_bytes)
    removed = []
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag == NS + "TextLine" and child.get("id") in drop_ids:
                parent.remove(child)
                removed.append(child.get("id"))
    body = ET.tostring(root, encoding="unicode")
    return ("<?xml version='1.0' encoding='UTF-8'?>\n" + body).encode("utf-8"), removed


def text_lines(xml_bytes: bytes) -> List[Tuple[str, str]]:
    """Ids and texts of the TextLines with non-empty text.

    :param xml_bytes: PAGE XML document.
    :return: ``(id, text)`` in document order.
    """
    root = ET.fromstring(xml_bytes)
    out = []
    for tl in root.iter(NS + "TextLine"):
        u = tl.find("%sTextEquiv/%sUnicode" % (NS, NS))
        text = (u.text or "").strip() if u is not None else ""
        if text:
            out.append((tl.get("id"), text))
    return out


def line_polygons(xml_bytes: bytes) -> Tuple[str, Dict[str, List[Point]]]:
    """Image path and the polygons of the non-empty TextLines.

    :param xml_bytes: PAGE XML document.
    :return: ``imageFilename`` and mapping id -> polygon points.
    """
    root = ET.fromstring(xml_bytes)
    img = root.find(".//%sPage" % NS).get("imageFilename")
    keep = {lid for lid, _ in text_lines(xml_bytes)}
    polys = {}
    for tl in root.iter(NS + "TextLine"):
        if tl.get("id") in keep:
            pts = tl.find(NS + "Coords").get("points").split()
            polys[tl.get("id")] = [tuple(int(float(v)) for v in p.split(",")) for p in pts]
    return img, polys


def read_split(lists_dir: Path) -> Dict[str, str]:
    """Train/val assignment of each page stem from a run's ``genizah_{train,val}.lst``.

    :param lists_dir: the run's ``lists`` folder.
    :return: mapping stem -> ``train`` or ``val``.
    """
    split = {}
    for name in SPLITS:
        f = lists_dir / ("genizah_%s.lst" % name)
        if not f.exists():
            continue
        for line in f.read_text().split():
            split[Path(line).stem] = name
    return split


# --------------------------------------------------------------------------------------------------------------
# Neighbour ink (port of the PGP geometry audit's nbbase.py)
# --------------------------------------------------------------------------------------------------------------

def rasterise(poly: Sequence[Point], x0: int, y0: int, width: int, height: int) -> np.ndarray:
    """Rasterise a polygon into a window.

    :param poly: polygon points in page coordinates.
    :param x0: window left.
    :param y0: window top.
    :param width: window width.
    :param height: window height.
    :return: bool mask of shape (height, width).
    """
    m = Image.new("L", (width, height), 0)
    ImageDraw.Draw(m).polygon([(x - x0, y - y0) for x, y in poly], fill=1)
    return np.asarray(m).astype(bool)


def baseline_y_at(baseline: Sequence[Sequence[float]], x: float, reach: float) -> Optional[float]:
    """Baseline y at column ``x``, carried flat up to ``reach`` pixels beyond its ends.

    :param baseline: baseline points.
    :param x: column.
    :param reach: pixels beyond the ends that still count.
    :return: y, or ``None`` outside the reach.
    """
    pts = sorted(baseline)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if x < xs[0] - reach or x > xs[-1] + reach:
        return None
    return float(np.interp(x, xs, ys))


def page_arrays(image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """Grey page and blue-mount mask as used by the neighbour-ink measure.

    :param image: page image.
    :return: ``(gray uint8, blue bool)``.
    """
    rgb = image.convert("RGB")
    a = np.asarray(rgb)
    gray = (0.3 * a[..., 0].astype(float) + 0.59 * a[..., 1].astype(float)
            + 0.11 * a[..., 2].astype(float)).astype(np.uint8)
    hsv = np.asarray(rgb.convert("HSV")).astype(int)
    blue = (hsv[..., 0] > 120) & (hsv[..., 0] < 200) & (hsv[..., 1] > 90)
    return gray, blue


def neighbour_ink_area(poly: Sequence[Point], line_height: float, segments: Set[int], src_lines: Sequence[dict],
                       gray: np.ndarray, blue: np.ndarray) -> Tuple[float, List[Tuple[int, int, int]]]:
    """Ink inside the polygon whose components sit nearer another segmenter baseline than the line's own.

    Otsu-thresholded ink (12 grey levels darker than the threshold, mount excluded, opened once) in a window of
    one line height left/right and 1.5 above/below is split into connected components. A component of at least
    (0.05 lh)^2 pixels that has pixels inside the polygon counts as neighbour ink when its centroid is closer, by
    more than 0.1 lh, to another segmenter baseline (carried 2.5 lh beyond its ends, and more than 0.4 lh away)
    than to the line's own baselines (carried 3 lh; the polygon's mean y when none reaches).

    :param poly: final polygon of the placed line.
    :param line_height: line height in pixels.
    :param segments: segmenter line indices the placed line was built from.
    :param src_lines: segmenter lines of the page (``index``, ``baseline``).
    :param gray: grey page.
    :param blue: mount mask.
    :return: neighbour pixels inside the polygon in lh^2, and the components ``(cx, cy, pixels inside)``.
    """
    lh = float(line_height)
    H, W = gray.shape
    xs = [q[0] for q in poly]
    ys = [q[1] for q in poly]
    x0 = max(0, int(min(xs) - lh))
    x1 = min(W, int(max(xs) + lh))
    y0 = max(0, int(min(ys) - 1.5 * lh))
    y1 = min(H, int(max(ys) + 1.5 * lh))
    g = gray[y0:y1, x0:x1]
    b = blue[y0:y1, x0:x1]
    paper = g[~b]
    if paper.size == 0:
        return 0.0, []
    hist = np.bincount(paper.ravel(), minlength=256).astype(float)
    p = hist / hist.sum()
    w = np.cumsum(p)
    m = np.cumsum(p * np.arange(256))
    t = int(np.argmax((m[-1] * w - m) ** 2 / (w * (1 - w) + 1e-12)))
    ink = ndimage.binary_opening((g < t - 12) & ~b, iterations=1)
    lab, _ = ndimage.label(ink)
    me = rasterise(poly, x0, y0, x1 - x0, y1 - y0)
    own_bl = [ln["baseline"] for ln in src_lines if ln["index"] in segments]
    oth_bl = [ln["baseline"] for ln in src_lines if ln["index"] not in segments]
    mean_y = float(np.mean(ys))
    tot = 0
    comps = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        cm = lab[sl] == i
        if int(cm.sum()) < (0.05 * lh) ** 2:
            continue
        inside = int((cm & me[sl]).sum())
        if inside == 0:
            continue
        cy, cx = ndimage.center_of_mass(cm)
        cy += sl[0].start + y0
        cx += sl[1].start + x0
        d_own = min([abs(cy - y) for y in (baseline_y_at(bl, cx, 3 * lh) for bl in own_bl) if y is not None]
                    or [1e9])
        if d_own > 1e8:
            d_own = abs(cy - mean_y)
        cand = [(cy - y) for y in (baseline_y_at(bl, cx, 2.5 * lh) for bl in oth_bl) if y is not None]
        d_oth = min([abs(c) for c in cand if abs(c) > 0.4 * lh] or [1e9])
        if d_oth + 0.1 * lh < d_own:
            tot += inside
            comps.append((int(cx), int(cy), inside))
    return tot / lh ** 2, comps


def measure_neighbour_ink(xml_dir: Path, placed: Dict[LineKey, dict], pages_dir: Path,
                          skip_pages: Set[str]) -> Dict[LineKey, float]:
    """Neighbour ink (lh^2) of every placed line of a run, page by page.

    :param xml_dir: the run's ``page_xml/<group>`` folder.
    :param placed: placed lines keyed by ``(stem, line_id)``.
    :param pages_dir: segmenter page JSONs (``<stem>.json`` with ``lines``).
    :param skip_pages: stems not measured (dropped whole).
    :return: mapping key -> area.
    """
    by_stem: Dict[str, List[str]] = {}
    for stem, lid in placed:
        if stem not in skip_pages:
            by_stem.setdefault(stem, []).append(lid)
    out = {}
    for stem in sorted(by_stem):
        img_path, polys = line_polygons((xml_dir / (stem + ".xml")).read_bytes())
        src = json.loads((pages_dir / (stem + ".json")).read_text())
        with Image.open(img_path) as im:
            gray, blue = page_arrays(im)
        for lid in by_stem[stem]:
            row = placed[(stem, lid)]
            area, _ = neighbour_ink_area(polys[lid], float(row.get("line_height") or DEFAULT_LINE_HEIGHT),
                                         set(row.get("segments") or []), src["lines"], gray, blue)
            out[(stem, lid)] = round(area, 4)
    return out


# --------------------------------------------------------------------------------------------------------------
# Filter a run
# --------------------------------------------------------------------------------------------------------------

def find_xml_dir(run: Path) -> Path:
    """The single ``page_xml/<group>`` folder of a run.

    :param run: run folder.
    :return: path of the group folder.
    """
    groups = sorted(p for p in (run / "page_xml").iterdir() if p.is_dir())
    if len(groups) != 1:
        raise ValueError("expected one page_xml group in %s, found %s" % (run, [g.name for g in groups]))
    return groups[0]


def atomic_write_text(path: Path, text: str) -> None:
    """Write a file in one go: a hidden temp file in the same folder, then ``os.replace``.

    :param path: destination.
    :param text: content.
    """
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def check_out_root(run: Path, out_xml_dir: Path) -> None:
    """Refuse an output inside the source run or an existing non-empty output folder.

    :param run: resolved source run folder.
    :param out_xml_dir: resolved output ``page_xml/<group>`` folder.
    """
    if out_xml_dir == run or run in out_xml_dir.parents:
        raise ValueError("output %s lies inside the source run %s" % (out_xml_dir, run))
    if out_xml_dir.exists() and any(out_xml_dir.iterdir()):
        raise FileExistsError("%s is not empty; pick a new out-root" % out_xml_dir)


def write_filtered(xml_dir: Path, out_root: Path, drop_ids: Dict[str, Set[str]], drop_pages: Set[str],
                   split: Dict[str, str]) -> dict:
    """Copy page XML without the dropped TextLines, then write ``lists/{train,val}.lst`` atomically.

    :param xml_dir: source ``page_xml/<group>`` folder.
    :param out_root: output root (XML goes to ``out_root/page_xml/<group>``).
    :param drop_ids: TextLine ids to remove per page stem.
    :param drop_pages: stems left out entirely.
    :param split: stem -> ``train``/``val`` (pages missing here are not written).
    :return: ``pages_out``, ``lines_out``, ``letters_kept`` per split, ``pages_emptied``, ``pages_unlisted``.
    """
    out_xml_dir = out_root / "page_xml" / xml_dir.name
    out_xml_dir.mkdir(parents=True, exist_ok=True)
    lists: Dict[str, List[str]] = {s: [] for s in SPLITS}
    letters = {s: 0 for s in SPLITS}
    lines_kept = {s: 0 for s in SPLITS}
    emptied, unlisted = [], []
    for xml in sorted(xml_dir.glob("*.xml")):
        stem = xml.stem
        if stem in drop_pages:
            continue
        want = drop_ids.get(stem, set())
        new_bytes, removed = remove_text_lines(xml.read_bytes(), want)
        if set(removed) != want:
            raise RuntimeError("%s: could not remove %s" % (stem, sorted(want - set(removed))))
        kept = text_lines(new_bytes)
        if not kept:
            emptied.append(stem)
            continue
        if stem not in split:
            unlisted.append(stem)
            continue
        dst = out_xml_dir / xml.name
        dst.write_bytes(new_bytes)
        s = split[stem]
        lists[s].append(str(dst))
        lines_kept[s] += len(kept)
        letters[s] += sum(arabic_letters(t) for _, t in kept)
    (out_root / "lists").mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        atomic_write_text(out_root / "lists" / ("%s.lst" % s), "".join(p + "\n" for p in lists[s]))
    return {"pages_out": {s: len(lists[s]) for s in SPLITS}, "lines_out": lines_kept, "letters_kept": letters,
            "pages_emptied": emptied, "pages_unlisted": unlisted}


def filter_run(run: Path, out_root: Path, rules: FilterRules, pages_dir: Optional[Path] = None) -> dict:
    """Write the filtered copy of a run (decisions from ``pairs.jsonl``) and return its report.

    :param run: source run folder (read only).
    :param out_root: NEW output folder (must not lie inside ``run``).
    :param rules: drop rules.
    :param pages_dir: segmenter page JSONs for the neighbour-ink rule (default: ``pages_dir`` of the run's
        ``alignment_stats.json``).
    :return: report dict (also written to ``out_root/filter_report.json``).
    """
    run = run.resolve()
    out_root = out_root.resolve()
    xml_dir = find_xml_dir(run)
    check_out_root(run, out_root / "page_xml" / xml_dir.name)
    rows = load_pairs(run / "pairs.jsonl")
    placed = placed_rows(rows)
    stems_with_lines = {k[0] for k in placed}
    all_stems = {r["stem"] for r in rows}
    missing_pages = sorted(s for s in rules.drop_pages if s not in stems_with_lines)
    missing_lines = sorted("%s:%s" % k for k in rules.drop_lines if k not in placed)
    neighbour: Dict[LineKey, float] = {}
    if rules.neighbour_ink_max is not None:
        if pages_dir is None:
            pages_dir = Path(json.loads((run / "alignment_stats.json").read_text())["pages_dir"])
        neighbour = measure_neighbour_ink(xml_dir, placed, pages_dir, rules.drop_pages)
    decisions = decide(placed, rules, neighbour)
    drop_ids: Dict[str, Set[str]] = {}
    for (stem, lid), rs in decisions.items():
        if rs:
            drop_ids.setdefault(stem, set()).add(lid)
    written = write_filtered(xml_dir, out_root, drop_ids, rules.drop_pages, read_split(run / "lists"))
    stats = run / "alignment_stats.json"
    if stats.exists():
        shutil.copy2(stats, out_root / "source_alignment_stats.json")
    report = {
        "source_run": str(run),
        "out_root": str(out_root),
        "mode": "pairs",
        "rules": rules.to_json(),
        "pages_dir": str(pages_dir) if pages_dir else None,
        "placed_lines_in": len(placed),
        "pages_in": len(stems_with_lines),
        "kept_lines": sum(1 for rs in decisions.values() if not rs),
        "dropped_lines": sum(1 for rs in decisions.values() if rs),
        "dropped_by_reason": tally(decisions),
        **written,
        "drop_pages_without_placed_lines": [{"stem": s, "in_pairs": s in all_stems} for s in missing_pages],
        "drop_lines_not_placed": missing_lines,
        "decisions": [
            {"stem": k[0], "line_id": k[1], "reasons": rs, "neighbour_ink": neighbour.get(k),
             "extended_lh_right": placed[k].get("extended_lh_right"), "block_limited": placed[k].get("block_limited"),
             "own_ink_outside": placed[k].get("own_ink_outside"), "lost": placed[k].get("lost")}
            for k, rs in decisions.items()
        ],
    }
    (out_root / "filter_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return report


# --------------------------------------------------------------------------------------------------------------
# Runs without pairs.jsonl (decisions from the TextLine custom attribute)
# --------------------------------------------------------------------------------------------------------------

CUSTOM_REASON_ORDER = ("tier", "lost", "merged")


def parse_custom(custom: Optional[str]) -> Dict[str, str]:
    """Key/value pairs of an ``alignment {tier:anchor; sim:0.74; lost:0.00;}`` custom attribute.

    :param custom: attribute value (or ``None``).
    :return: mapping key -> raw string value.
    """
    return dict(re.findall(r"(\w+):([^;}]*);", custom or ""))


def custom_drop_reasons(line_id: str, custom: Optional[str], keep_tiers: Set[str], max_lost: float = 0.0,
                        allow_merged: bool = False) -> List[str]:
    """Reasons to drop a placed line judged only from its id and custom attribute.

    :param line_id: TextLine id (a merged line has an underscore, e.g. ``l4_5``).
    :param custom: TextLine custom attribute.
    :param keep_tiers: tiers kept (e.g. ``{"anchor"}``).
    :param max_lost: drop when ``lost`` is above this; a missing or unparsable ``lost`` is dropped too.
    :param allow_merged: keep merged ids.
    :return: reasons in :data:`CUSTOM_REASON_ORDER` (empty = kept).
    """
    kv = parse_custom(custom)
    out = []
    if kv.get("tier") not in keep_tiers:
        out.append("tier")
    lost = kv.get("lost")
    if lost is None or not re.fullmatch(r"\d+(\.\d+)?", lost.strip()) or float(lost) > max_lost:
        out.append("lost")
    if not allow_merged and "_" in line_id:
        out.append("merged")
    return out


def filter_custom_run(run: Path, out_root: Path, keep_tiers: Set[str], max_lost: float = 0.0,
                      allow_merged: bool = False) -> dict:
    """Write the filtered copy of a run that has no ``pairs.jsonl`` and return its report.

    :param run: source run folder (read only).
    :param out_root: NEW output folder.
    :param keep_tiers: tiers kept.
    :param max_lost: largest ``lost`` kept.
    :param allow_merged: keep merged ids.
    :return: report dict (also written to ``out_root/filter_report.json``).
    """
    run = run.resolve()
    out_root = out_root.resolve()
    xml_dir = find_xml_dir(run)
    check_out_root(run, out_root / "page_xml" / xml_dir.name)
    decisions: Dict[LineKey, List[str]] = {}
    customs: Dict[LineKey, str] = {}
    for xml in sorted(xml_dir.glob("*.xml")):
        root = ET.fromstring(xml.read_bytes())
        placed_ids = {lid for lid, _ in text_lines(xml.read_bytes())}
        for tl in root.iter(NS + "TextLine"):
            if tl.get("id") in placed_ids:
                key = (xml.stem, tl.get("id"))
                customs[key] = tl.get("custom") or ""
                decisions[key] = custom_drop_reasons(tl.get("id"), tl.get("custom"), keep_tiers, max_lost, allow_merged)
    drop_ids: Dict[str, Set[str]] = {}
    for (stem, lid), rs in decisions.items():
        if rs:
            drop_ids.setdefault(stem, set()).add(lid)
    written = write_filtered(xml_dir, out_root, drop_ids, set(), read_split(run / "lists"))
    first = Counter(rs[0] for rs in decisions.values() if rs)
    anyr = Counter(r for rs in decisions.values() for r in rs)
    report = {
        "source_run": str(run),
        "out_root": str(out_root),
        "mode": "custom",
        "rules": {"keep_tiers": sorted(keep_tiers), "max_lost": max_lost, "allow_merged": allow_merged,
                  "lost_missing_is_dropped": True},
        "placed_lines_in": len(decisions),
        "pages_in": len({k[0] for k in decisions}),
        "kept_lines": sum(1 for rs in decisions.values() if not rs),
        "dropped_lines": sum(1 for rs in decisions.values() if rs),
        "dropped_by_reason": {"first": {r: first.get(r, 0) for r in CUSTOM_REASON_ORDER},
                              "any": {r: anyr.get(r, 0) for r in CUSTOM_REASON_ORDER}},
        "placed_tiers": dict(Counter(parse_custom(c).get("tier") for c in customs.values())),
        **written,
        "decisions": [{"stem": k[0], "line_id": k[1], "reasons": rs, "custom": customs[k]}
                      for k, rs in decisions.items()],
    }
    (out_root / "filter_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return report


# --------------------------------------------------------------------------------------------------------------
# Combined lists
# --------------------------------------------------------------------------------------------------------------

def list_totals(paths: Sequence[str]) -> Dict[str, int]:
    """Pages, text lines and Arabic letters of a list of PAGE XML files.

    :param paths: absolute XML paths.
    :return: ``{"pages", "lines", "letters"}``.
    """
    lines = letters = 0
    for p in paths:
        kept = text_lines(Path(p).read_bytes())
        lines += len(kept)
        letters += sum(arabic_letters(t) for _, t in kept)
    return {"pages": len(paths), "lines": lines, "letters": letters}


def combine_lists(out_dir: Path, groups: Dict[str, List[Path]]) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Write ``NAME_{train,val}.lst`` per group of filtered roots and ``totals.json``.

    :param out_dir: lists folder to write.
    :param groups: list name -> filtered roots whose ``lists/{train,val}.lst`` are concatenated.
    :return: totals per list name and split.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    totals: Dict[str, Dict[str, Dict[str, int]]] = {}
    for name, roots in groups.items():
        totals[name] = {}
        for s in SPLITS:
            paths: List[str] = []
            for root in roots:
                paths += (root / "lists" / ("%s.lst" % s)).read_text().split()
            if len(set(paths)) != len(paths):
                raise ValueError("%s/%s lists a page twice" % (name, s))
            atomic_write_text(out_dir / ("%s_%s.lst" % (name, s)), "".join(p + "\n" for p in paths))
            totals[name][s] = list_totals(paths)
        totals[name]["roots"] = [str(r) for r in roots]
    atomic_write_text(out_dir / "totals.json", json.dumps(totals, indent=1))
    return totals


def parse_groups(specs: Sequence[str]) -> Dict[str, List[Path]]:
    """Parse ``NAME=ROOT[,ROOT...]`` specs.

    :param specs: command-line specs.
    :return: mapping name -> roots.
    """
    out = {}
    for spec in specs:
        name, sep, roots = spec.partition("=")
        if not sep or not name or not roots:
            raise ValueError("list spec must be NAME=ROOT[,ROOT...], got %r" % spec)
        out[name] = [Path(r) for r in roots.split(",")]
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point.

    :param argv: arguments (default ``sys.argv[1:]``).
    :return: exit status.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("filter", help="write a filtered copy of a run")
    f.add_argument("--run", type=Path, required=True)
    f.add_argument("--out-root", type=Path, required=True)
    f.add_argument("--drop-pages", nargs="*", default=[])
    f.add_argument("--drop-lines", nargs="*", default=[], help="STEM:LINE_ID")
    f.add_argument("--max-lost", type=float, default=0.0)
    f.add_argument("--max-own-ink-outside", type=float, default=0.10)
    f.add_argument("--max-ext-right-lh", type=float, default=None)
    f.add_argument("--block-ext-right-lh", type=float, default=None)
    f.add_argument("--neighbour-ink-max", type=float, default=None)
    f.add_argument("--pages-dir", type=Path, default=None)
    fc = sub.add_parser("filter-custom", help="filter a run without pairs.jsonl by the TextLine custom attribute")
    fc.add_argument("--run", type=Path, required=True)
    fc.add_argument("--out-root", type=Path, required=True)
    fc.add_argument("--keep-tiers", nargs="+", default=["anchor"])
    fc.add_argument("--max-lost", type=float, default=0.0)
    fc.add_argument("--allow-merged", action="store_true")
    c = sub.add_parser("combine", help="combine filtered lists and count totals")
    c.add_argument("--out-dir", type=Path, required=True)
    c.add_argument("--list", dest="lists", action="append", required=True, help="NAME=ROOT[,ROOT...]")
    args = ap.parse_args(argv)
    if args.cmd == "filter":
        rules = FilterRules(drop_pages=set(args.drop_pages), drop_lines={parse_line_key(s) for s in args.drop_lines},
                            max_lost=args.max_lost, max_own_ink_outside=args.max_own_ink_outside,
                            max_ext_right_lh=args.max_ext_right_lh, block_ext_right_lh=args.block_ext_right_lh,
                            neighbour_ink_max=args.neighbour_ink_max)
        rep = filter_run(args.run, args.out_root, rules, args.pages_dir)
        summary = {k: rep[k] for k in ("placed_lines_in", "kept_lines", "dropped_lines", "dropped_by_reason",
                                       "pages_out", "lines_out", "letters_kept", "pages_emptied", "pages_unlisted",
                                       "drop_pages_without_placed_lines", "drop_lines_not_placed")}
        print(json.dumps(summary, indent=1))
    elif args.cmd == "filter-custom":
        rep = filter_custom_run(args.run, args.out_root, set(args.keep_tiers), args.max_lost, args.allow_merged)
        print(json.dumps({k: v for k, v in rep.items() if k != "decisions"}, indent=1))
    else:
        print(json.dumps(combine_lists(args.out_dir, parse_groups(args.lists)), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
