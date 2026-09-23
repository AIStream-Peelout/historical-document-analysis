# File name: line_rule.py
# Date: 9/8/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Two-reader line rule ``lines-v2-20260909``: Kraken fragments vs VLM lines.

The website (``genizah_search/src/backend/ai_transcriptions.py``) re-derives
each line's status from ``agreement`` with the same threshold and rejects
records that disagree, so the constants here and there must change together
(and with a new ``RULE_VERSION``).

Rule v2 (2026-09-09; v1 measured in the 2026-09-08 probe, 24 KTIV pages):

* Candidates for a VLM line are the Kraken fragments with at least half of
  their height inside the line's vertical band.  Candidates are clustered by
  horizontal gaps (columns); the cluster overlapping the VLM box most (else
  the nearest to its centre) is taken WHOLE, so a narrow or template VLM box
  no longer drops the fragments at the ends of the row.  Each fragment goes
  to the single line whose band covers it best; fragments are concatenated
  right-to-left.  (v1 required half of each fragment's width inside the VLM
  box — correct for column-wide boxes, lossy for the narrow repeated boxes
  the model emits on most site images.)
* Agreement = ``1 - Levenshtein / max(len)`` on Hebrew letters only.
* A line is *agreed* when agreement >= 0.8 and Kraken read something.
* The stored line box is the EVIDENCE box: union of the VLM box and the
  line's assigned fragments (the VLM box alone when there are none).
"""
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import Levenshtein

RULE_VERSION = "lines-v3-20260917"
AGREED_MIN = 0.8
COLUMN_GAP = 60.0   # 0-1000 units of x between fragments that starts a new column cluster
MIN_X_OVERLAP = 0.1  # v3: a fragment must overlap the VLM box in x by >= this share of its own width
_HEB = re.compile(r"[א-ת]")


def letters(s: str) -> str:
    """Hebrew letters only (the scorer's letters-only convention).

    :param s: Any text.
    :returns: The Hebrew letters of ``s`` in order.
    """
    return "".join(_HEB.findall(s or ""))


def similarity(a: str, b: str) -> float:
    """Symmetric letters-only similarity ``1 - lev / max(len)``; 0 if either is empty.

    :param a: One reading.
    :param b: The other reading.
    :returns: Similarity in [0, 1].
    """
    a, b = letters(a), letters(b)
    if not a or not b:
        return 0.0
    return 1.0 - Levenshtein.distance(a, b) / max(len(a), len(b))


def overlap_frac(a: Sequence[float], b: Sequence[float], axis: int) -> float:
    """Share of box ``b``'s extent on ``axis`` (0 = x, 1 = y) that lies inside box ``a``.

    :param a: Containing box ``[x1, y1, x2, y2]``.
    :param b: Box whose extent is measured.
    :param axis: 0 for horizontal, 1 for vertical.
    :returns: Fraction in [0, 1].
    """
    lo, hi = axis, axis + 2
    ov = max(0.0, min(a[hi], b[hi]) - max(a[lo], b[lo]))
    ext = b[hi] - b[lo]
    return ov / ext if ext > 0 else 0.0


def _clusters_by_gap(cands: List[Tuple[int, Dict[str, Any]]], gap: float) -> List[List[Tuple[int, Dict[str, Any]]]]:
    """Split ``(index, fragment)`` candidates into column clusters at horizontal gaps > ``gap``.

    :param cands: Fragments in one vertical band.
    :param gap: Minimum empty x-span (0-1000 units) that separates columns.
    :returns: Clusters ordered left to right.
    """
    if not cands:
        return []
    cands = sorted(cands, key=lambda c: c[1]["box"][0])
    out, cur, right = [], [cands[0]], cands[0][1]["box"][2]
    for j, f in cands[1:]:
        if f["box"][0] - right > gap:
            out.append(cur)
            cur = []
        cur.append((j, f))
        right = max(right, f["box"][2])
    out.append(cur)
    return out


def assign_fragments(vlm_lines: Sequence[Dict[str, Any]], frags: Sequence[Dict[str, Any]],
                     gap: float = COLUMN_GAP, min_x: float = MIN_X_OVERLAP) -> List[List[int]]:
    """Assign Kraken fragments to VLM lines (rule v3: band first, whole column cluster,
    horizontally gated).

    v3 adds ``min_x``: a fragment is a candidate only if it overlaps the VLM box
    horizontally by at least that share of its own width.  Under v2, on a two-page
    opening whose gutter is narrower than ``gap`` both pages' fragments formed one
    cluster, so every line swallowed the facing page's words (evidence boxes spanning
    the gutter, agreement destroyed; Or.1080 1.55, 2026-09-17).

    :param vlm_lines: Dicts with ``box`` (0-1000 ``[x1, y1, x2, y2]``).
    :param frags: Dicts with ``box`` in the same space.
    :param gap: Column-splitting gap, see :data:`COLUMN_GAP`.
    :param min_x: Horizontal gate, see :data:`MIN_X_OVERLAP`.
    :returns: Per VLM line, the indices of its fragments sorted right-to-left.
    """
    claims: Dict[int, Tuple[float, int]] = {}      # fragment -> (vertical overlap, line)
    for i, ln in enumerate(vlm_lines):
        cands = [(j, f) for j, f in enumerate(frags)
                 if overlap_frac(ln["box"], f["box"], 1) >= 0.5
                 and overlap_frac(f["box"], ln["box"], 0) >= min_x]
        if not cands:
            continue
        bx = ln["box"]
        cx = (bx[0] + bx[2]) / 2

        def score(cluster: List[Tuple[int, Dict[str, Any]]]) -> Tuple[float, float]:
            lo = min(f["box"][0] for _, f in cluster)
            hi = max(f["box"][2] for _, f in cluster)
            return (max(0.0, min(hi, bx[2]) - max(lo, bx[0])), -abs((lo + hi) / 2 - cx))

        for j, f in max(_clusters_by_gap(cands, gap), key=score):
            vy = overlap_frac(ln["box"], f["box"], 1)
            if j not in claims or vy > claims[j][0]:
                claims[j] = (vy, i)
    groups: List[List[int]] = [[] for _ in vlm_lines]
    for j, (_, i) in claims.items():
        groups[i].append(j)
    for g in groups:
        g.sort(key=lambda j: -(frags[j]["box"][0] + frags[j]["box"][2]))
    return groups


def evidence_box(vlm_box: Sequence[float], frag_boxes: Sequence[Sequence[float]]) -> List[float]:
    """Union of the VLM box and its assigned fragments (the VLM box alone without fragments).

    :param vlm_box: The model's line box.
    :param frag_boxes: Boxes of the fragments assigned to the line.
    :returns: ``[x1, y1, x2, y2]``.
    """
    boxes = [list(vlm_box)] + [list(b) for b in frag_boxes]
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def line_status(agreement: Optional[float], htr_text: Optional[str]) -> str:
    """``"agreed"`` when both readers concur, else ``"unconfirmed"`` (mirrors the site).

    :param agreement: Similarity between the two readings, or None without Kraken evidence.
    :param htr_text: Kraken's concatenated reading.
    :returns: Line status.
    """
    if agreement is None or not (htr_text or "").strip():
        return "unconfirmed"
    return "agreed" if agreement >= AGREED_MIN else "unconfirmed"
