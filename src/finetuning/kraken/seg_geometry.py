"""Pure-Python line geometry shared by the segmenter exporter and its in-container gate.

No numpy / PIL / repo imports, so ``seg_gate.py`` can import it inside the
kraken container with only this directory on ``PYTHONPATH``.  Boxes are
``(x0, y0, x1, y1)`` in page pixels; baselines are ``[(x, y), ...]`` point
lists.
"""
from typing import List, Optional, Sequence, Tuple

Box = Tuple[float, float, float, float]
Point = Tuple[float, float]

COVER_MIN_SHARE = 0.3        # share of a detected line's extent inside GT lines to count as covered
COVER_PAD_PITCH = 0.3        # GT boxes are padded by this many line pitches for the cover test


def polyline_box(points: Sequence[Sequence[float]]) -> Box:
    """Axis-aligned bounding box of a point list.

    :param points: ``[(x, y), ...]``.
    :type points: Sequence[Sequence[float]]
    :return: ``(x0, y0, x1, y1)``.
    :rtype: Box
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def pad_box(box: Box, pad: float) -> Box:
    """Grow a box by ``pad`` px on every side.

    :param box: Box to grow.
    :type box: Box
    :param pad: Padding, px.
    :type pad: float
    :return: Padded box.
    :rtype: Box
    """
    return box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad


def y_at(baseline: Sequence[Sequence[float]], x: float) -> Optional[float]:
    """Linear interpolation of a polyline's y at ``x`` (None outside its x-range).

    :param baseline: Points in any x order.
    :type baseline: Sequence[Sequence[float]]
    :param x: Query abscissa, px.
    :type x: float
    :return: Interpolated y, or None.
    :rtype: Optional[float]
    """
    pts = sorted((float(p[0]), float(p[1])) for p in baseline)
    if not pts or x < pts[0][0] or x > pts[-1][0]:
        return None
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        if xa <= x <= xb:
            return ya if xb == xa else ya + (yb - ya) * (x - xa) / (xb - xa)
    return pts[-1][1]


def covered_share(baseline: Sequence[Sequence[float]], gt_boxes: Sequence[Box], pad: float,
                  samples: int = 16) -> float:
    """Share of a detected baseline whose sample points fall inside a (padded) GT line box.

    :param baseline: Detected baseline points.
    :type baseline: Sequence[Sequence[float]]
    :param gt_boxes: GT line boxes on the page.
    :type gt_boxes: Sequence[Box]
    :param pad: Padding applied to every GT box, px.
    :type pad: float
    :param samples: Number of evenly spaced x samples along the baseline.
    :type samples: int
    :return: Covered share in ``[0, 1]``.
    :rtype: float
    """
    x0, _, x1, _ = polyline_box(baseline)
    padded = [pad_box(b, pad) for b in gt_boxes]
    hits = n = 0
    for i in range(samples):
        x = x0 + (x1 - x0) * (i + 0.5) / samples
        y = y_at(baseline, x)
        if y is None:
            continue
        n += 1
        hits += any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in padded)
    return hits / n if n else 0.0


def is_covered(baseline: Sequence[Sequence[float]], gt_boxes: Sequence[Box], pitch: float) -> bool:
    """Whether a detected line lies on transcribed (GT) text.

    :param baseline: Detected baseline points.
    :type baseline: Sequence[Sequence[float]]
    :param gt_boxes: GT line boxes.
    :type gt_boxes: Sequence[Box]
    :param pitch: Page line pitch, px.
    :type pitch: float
    :return: True when at least :data:`COVER_MIN_SHARE` of it lies in padded GT boxes.
    :rtype: bool
    """
    return covered_share(baseline, gt_boxes, COVER_PAD_PITCH * pitch) >= COVER_MIN_SHARE


def match_lines(pred_baselines: List[Sequence[Sequence[float]]], gt_boxes: Sequence[Box],
                pitch: float, min_share: float = 0.8, min_span: float = 0.6) -> List[Optional[int]]:
    """One-to-one matches of detected baselines to GT lines (for calibration only).

    A detected line matches GT line ``g`` when at least ``min_share`` of its samples fall in ``g``'s
    padded box and its x-extent spans at least ``min_span`` of ``g``'s width; GT lines claimed by
    more than one detection are left unmatched (fragmented lines are not calibration material).

    :param pred_baselines: Detected baselines.
    :type pred_baselines: List[Sequence[Sequence[float]]]
    :param gt_boxes: GT line boxes.
    :type gt_boxes: Sequence[Box]
    :param pitch: Page line pitch, px.
    :type pitch: float
    :param min_share: Minimum covered share inside the single GT box.
    :type min_share: float
    :param min_span: Minimum x-span ratio detected/GT.
    :type min_span: float
    :return: Per GT line, the index of its unique matching detection, or None.
    :rtype: List[Optional[int]]
    """
    pad = COVER_PAD_PITCH * pitch
    claims: List[List[int]] = [[] for _ in gt_boxes]
    for j, bl in enumerate(pred_baselines):
        bx = polyline_box(bl)
        for g, box in enumerate(gt_boxes):
            if covered_share(bl, [box], pad) < min_share:
                continue
            if (bx[2] - bx[0]) >= min_span * max(1.0, box[2] - box[0]):
                claims[g].append(j)
    owner = {}
    for g, js in enumerate(claims):
        for j in js:
            owner.setdefault(j, []).append(g)
    return [js[0] if len(js) == 1 and len(owner[js[0]]) == 1 else None for js in claims]
