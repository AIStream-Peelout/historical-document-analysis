# File name: reread.py
# Date: 10/5/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Pure helpers for re-reading pages whose first VLM read failed (see :mod:`src.datasets.consensus.read_health`).

Used by the re-read experiment (``logs/next_round/failed_reads/reread_experiment.py``, which makes the
LM Studio requests, and ``score_reread.py``, which scores them); kept here, free of I/O, so the parts a
pipeline remedy would reuse are unit-tested:

* :class:`Variant` / :data:`VARIANTS` / :func:`variant_plan`: the request variants of the experiment and
  which of them apply to a job's failure class.
* :func:`half_boxes` / :func:`crop_halves` / :func:`to_page_box`: a page cut into a top and a bottom half
  that overlap, and line boxes mapped from a half back to the page.
* :func:`line_match` / :func:`join_halves`: the two halves' line reads joined into one page read, the lines
  both halves read in the overlap kept once (the longer reading of each, since a crop edge can cut a line).
* :func:`select_jobs`: the experiment's job list, balanced over failure classes (loops on near-blank images apart).
* :func:`paired_summary`: a variant's re-reads against the failed reads they replace.
"""
import hashlib
import re
import statistics as st
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import Levenshtein
from PIL import Image

from src.datasets.consensus.read_health import (
    LABEL_CAPPED, LABEL_LOOP, LABEL_NEAR_EMPTY, LABEL_OK, LABEL_SKIPPED, REREAD_LABELS)

OVERLAP_SHARE = 0.12          # the halves share 12 % of the page height: >= 3 lines of a 25-line page sit in both crops
OVERLAP_WINDOW_LINES = 10     # the overlap holds at most ~10 lines (a 12 % band of an 80-line page)
JOIN_MIN_SIMILARITY = 0.6     # line_rule's similarity scale: two readings of one line by one model score well above 0.6
CONTAIN_MIN_LETTERS = 6       # a line cut by a crop edge is matched inside the full line only when it keeps >= 6 letters
SAME_MARGIN = 0.02            # paired CER changes within +-0.02 count as unchanged (page-to-page noise of a re-read)
CLASS_ORDER = (LABEL_CAPPED, LABEL_NEAR_EMPTY, LABEL_SKIPPED, LABEL_LOOP)   # rarest first: every class gets jobs under a small --limit

_LETTER = re.compile(r"[א-תء-ي]")


# ----------------------------------------------------------------------------- variants


@dataclass(frozen=True)
class Variant:
    """One request variant of the re-read experiment.

    :param name: Output folder name.
    :param description: What the variant changes against the pipeline's request.
    :param max_tokens: ``max_tokens`` (None = the pipeline's value).
    :param temperature: Sampling temperature (None = the client's default, which the pipeline uses).
    :param extra_payload: Extra OpenAI-compatible request fields (``seed``, ``repeat_penalty``).
    :param halves: Read the page as two overlapping halves and join them.
    :param only_classes: Run only for jobs of these failure classes (empty = every job).
    """

    name: str
    description: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    extra_payload: Dict[str, Any] = field(default_factory=dict)
    halves: bool = False
    only_classes: Tuple[str, ...] = ()

    def requests_per_job(self) -> int:
        """LM Studio requests one job of this variant costs.

        :return: 2 for the halves variant, else 1.
        :rtype: int
        """
        return 2 if self.halves else 1


def default_variants(seed: int = 20261005, repeat_penalty: float = 1.1, capped_max_tokens: int = 6000) -> Dict[str, Variant]:
    """The experiment's variants, in run order.

    :param seed: Seed of the sampling variant.
    :type seed: int
    :param repeat_penalty: ``repeat_penalty`` of the penalty variant (LM Studio's own default value).
    :type repeat_penalty: float
    :param capped_max_tokens: ``max_tokens`` of the raised-cap variant.
    :type capped_max_tokens: int
    :return: ``name -> Variant``.
    :rtype: Dict[str, Variant]
    """
    variants = [
        Variant("same", "the pipeline's request again, unchanged (is the failure deterministic?)"),
        Variant(f"max_tokens_{capped_max_tokens}", f"max_tokens {capped_max_tokens} instead of the pipeline's cap",
                max_tokens=capped_max_tokens, only_classes=(LABEL_CAPPED,)),
        Variant("temp03_seed", f"temperature 0.3 with seed {seed}", temperature=0.3, extra_payload={"seed": seed}),
        Variant("repeat_penalty", f"repeat_penalty {repeat_penalty}", extra_payload={"repeat_penalty": repeat_penalty}),
        Variant("halves", "top and bottom half read separately (overlapping), the line reads joined", halves=True),
    ]
    return {v.name: v for v in variants}


VARIANTS = default_variants()


def variant_plan(failure_class: str, names: Sequence[str], variants: Mapping[str, Variant] = VARIANTS) -> List[Variant]:
    """The variants that apply to one job.

    :param failure_class: The job's failure class.
    :type failure_class: str
    :param names: Requested variant names, in run order.
    :type names: Sequence[str]
    :param variants: Known variants.
    :type variants: Mapping[str, Variant]
    :return: The requested variants whose ``only_classes`` admit the class.
    :rtype: List[Variant]
    :raises KeyError: For an unknown variant name.
    """
    plan = [variants[n] for n in names]
    return [v for v in plan if not v.only_classes or failure_class in v.only_classes]


# ----------------------------------------------------------------------------- half-page reads


def half_boxes(width: int, height: int, overlap: float = OVERLAP_SHARE) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """Pixel boxes of the top and bottom half of a page, overlapping around the middle.

    :param width: Page width in pixels.
    :type width: int
    :param height: Page height in pixels.
    :type height: int
    :param overlap: Share of the page height both halves contain (centred on the middle).
    :type overlap: float
    :return: ``((0, 0, width, top_bottom), (0, bottom_top, width, height))`` (PIL crop boxes).
    :rtype: Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]
    :raises ValueError: For an overlap outside [0, 1) or an empty page.
    """
    if not 0 <= overlap < 1 or width <= 0 or height <= 0:
        raise ValueError(f"bad half split: {width}x{height}, overlap {overlap}")
    half_band = overlap * height / 2
    top_end = min(height, int(round(height / 2 + half_band)))
    bottom_start = max(0, int(round(height / 2 - half_band)))
    return (0, 0, width, top_end), (0, bottom_start, width, height)


def crop_halves(image: Image.Image, overlap: float = OVERLAP_SHARE) -> Tuple[Image.Image, Image.Image]:
    """The top and bottom half of a page image (see :func:`half_boxes`).

    :param image: Oriented page image.
    :type image: Image.Image
    :param overlap: Share of the page height both halves contain.
    :type overlap: float
    :return: ``(top, bottom)`` crops.
    :rtype: Tuple[Image.Image, Image.Image]
    """
    top_box, bottom_box = half_boxes(image.width, image.height, overlap)
    return image.crop(top_box), image.crop(bottom_box)


def to_page_box(box: Sequence[float], crop_box: Sequence[int], width: int, height: int) -> List[float]:
    """Map a 0-1000 box drawn on a crop to the page's 0-1000 frame.

    :param box: ``[x1, y1, x2, y2]`` in the crop's 0-1000 frame.
    :type box: Sequence[float]
    :param crop_box: Pixel box of the crop on the page ``(x1, y1, x2, y2)``.
    :type crop_box: Sequence[int]
    :param width: Page width in pixels.
    :type width: int
    :param height: Page height in pixels.
    :type height: int
    :return: The box in the page's 0-1000 frame (rounded to 0.1).
    :rtype: List[float]
    """
    cx1, cy1, cx2, cy2 = crop_box
    cw, ch = cx2 - cx1, cy2 - cy1
    xs = [(cx1 + cw * box[i] / 1000) * 1000 / width for i in (0, 2)]
    ys = [(cy1 + ch * box[i] / 1000) * 1000 / height for i in (1, 3)]
    return [round(xs[0], 1), round(ys[0], 1), round(xs[1], 1), round(ys[1], 1)]


def _letters(text: str) -> str:
    """Hebrew and Arabic letters of a text, in order.

    :param text: Any text.
    :type text: str
    :return: Letters only.
    :rtype: str
    """
    return "".join(_LETTER.findall(text or ""))


def line_match(a: str, b: str, min_contained: int = CONTAIN_MIN_LETTERS) -> float:
    """How well two line readings match: letter similarity, or containment of a cut line in a full one.

    :param a: One line.
    :type a: str
    :param b: Another line.
    :type b: str
    :param min_contained: Shortest line (letters) matched by containment.
    :type min_contained: int
    :return: ``max(1 - lev / max_len, best 1 - lev / len_short over windows of the longer line)`` on letters;
        0.0 when either line has no letters.
    :rtype: float
    """
    la, lb = _letters(a), _letters(b)
    if not la or not lb:
        return 0.0
    full = 1 - Levenshtein.distance(la, lb) / max(len(la), len(lb))
    short, long_ = (la, lb) if len(la) <= len(lb) else (lb, la)
    if len(short) < min_contained or len(short) == len(long_):
        return full
    n = len(short)
    contained = max(1 - Levenshtein.distance(short, long_[i:i + n]) / n for i in range(len(long_) - n + 1))
    return max(full, contained)


def join_halves(top: Sequence[Mapping[str, Any]], bottom: Sequence[Mapping[str, Any]],
                window: int = OVERLAP_WINDOW_LINES,
                min_sim: float = JOIN_MIN_SIMILARITY) -> Tuple[List[Dict[str, Any]], int]:
    """Join the line reads of a page's top and bottom half into one read.

    The bottom read starts with lines the top read ends with.  Each of the bottom's first ``window`` lines is
    matched against the top's last ``window`` lines (:func:`line_match`); the bottom's lines up to the last
    one that matches are the overlap and are dropped, and a matched top line is replaced by its bottom
    reading when that holds more letters (the top crop's edge may have cut it).  With no match the reads
    are simply concatenated.

    :param top: Top half's lines in reading order (dicts with ``text``; other keys are carried along).
    :type top: Sequence[Mapping[str, Any]]
    :param bottom: Bottom half's lines in reading order.
    :type bottom: Sequence[Mapping[str, Any]]
    :param window: Lines compared on each side of the overlap.
    :type window: int
    :param min_sim: Minimum :func:`line_match` for two lines to be one line.
    :type min_sim: float
    :return: ``(joined lines, number of bottom lines dropped as overlap)``.
    :rtype: Tuple[List[Dict[str, Any]], int]
    """
    merged = [dict(ln) for ln in top]
    tail_start = max(0, len(merged) - window)
    matched: Dict[int, int] = {}
    for i, line in enumerate(bottom[:window]):
        best, best_j = 0.0, -1
        for j in range(tail_start, len(top)):
            score = line_match(line.get("text") or "", top[j].get("text") or "")
            if score > best or (score == best and j > best_j):
                best, best_j = score, j
        if best >= min_sim:
            matched[i] = best_j
    if not matched:
        return merged + [dict(ln) for ln in bottom], 0
    last = max(matched)
    for i, j in sorted(matched.items()):
        if len(_letters(bottom[i].get("text") or "")) > len(_letters(merged[j].get("text") or "")):
            merged[j] = dict(bottom[i])
    return merged + [dict(ln) for ln in bottom[last + 1:]], last + 1


# ----------------------------------------------------------------------------- jobs


def stable_rank(key: str) -> int:
    """Deterministic pseudo-random rank of a key (stable across runs and machines).

    :param key: Any string.
    :type key: str
    :return: Integer derived from the key's SHA-1.
    :rtype: int
    """
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:12], 16)


def failure_stratum(candidate: Mapping[str, Any]) -> str:
    """Stratum of a candidate for job balancing: its failure class, with a near-blank image kept apart.

    :param candidate: Candidate with ``failure_class`` and optionally ``blank_image`` (Kraken read almost nothing).
    :type candidate: Mapping[str, Any]
    :return: ``failure_class``, or ``failure_class + "/blank_image"``.
    :rtype: str
    """
    return candidate["failure_class"] + ("/blank_image" if candidate.get("blank_image") else "")


def select_jobs(candidates: Sequence[Mapping[str, Any]], limit: int, held_out_only: bool = True,
                class_order: Sequence[str] = CLASS_ORDER,
                stratum: Callable[[Mapping[str, Any]], str] = failure_stratum) -> List[Dict[str, Any]]:
    """Pick the re-read jobs: flagged reads a re-read can fix, balanced over failure strata.

    Candidates are filtered to ``failure_class`` in :data:`~src.datasets.consensus.read_health.REREAD_LABELS`
    (a parse loss needs a re-parse, not a re-read) and, with ``held_out_only``, to ``held_out`` ones; one job
    per ``job_id``.  The strata (by default the failure class, loops on near-blank images apart) then take
    turns (``class_order`` first, any other stratum after, sorted), each in a stable pseudo-random order with
    held-out reads first, until ``limit`` jobs are chosen; so the first N jobs of the list are balanced for any N,
    and without ``held_out_only`` the held-out jobs are a subset of the list.

    :param candidates: Flagged reads with ``job_id``, ``failure_class`` and ``held_out``.
    :type candidates: Sequence[Mapping[str, Any]]
    :param limit: Maximum number of jobs (0 = all eligible).
    :type limit: int
    :param held_out_only: Keep only reads of pages the model never trained on.
    :type held_out_only: bool
    :param class_order: Strata served first, in turn order.
    :type class_order: Sequence[str]
    :param stratum: Stratum of a candidate.
    :type stratum: Callable[[Mapping[str, Any]], str]
    :return: Jobs in turn order (copies of the candidates).
    :rtype: List[Dict[str, Any]]
    """
    by_class: Dict[str, List[Dict[str, Any]]] = {}
    seen = set()
    for c in sorted(candidates, key=lambda c: (not c.get("held_out"), stable_rank(c["job_id"]))):  # held-out reads first
        if c["failure_class"] not in REREAD_LABELS or (held_out_only and not c.get("held_out")):
            continue
        if c["job_id"] in seen:
            continue
        seen.add(c["job_id"])
        by_class.setdefault(stratum(c), []).append(dict(c))
    order = [k for k in class_order if k in by_class] + sorted(k for k in by_class if k not in class_order)
    queues = [by_class[k] for k in order]
    jobs: List[Dict[str, Any]] = []
    while any(queues) and (not limit or len(jobs) < limit):
        for q in queues:
            if q and (not limit or len(jobs) < limit):
                jobs.append(q.pop(0))
    return jobs


# ----------------------------------------------------------------------------- scoring


def paired_summary(rows: Sequence[Mapping[str, Any]], margin: float = SAME_MARGIN) -> Dict[str, Any]:
    """A variant's re-reads against the failed reads they would replace (same pages, same ground truth).

    :param rows: One per job with ``cer``, ``err``, ``ref`` (the re-read), ``base_cer``, ``base_err`` (the cached
        failed read) and ``label`` (the re-read's health label).
    :type rows: Sequence[Mapping[str, Any]]
    :param margin: CER changes within this margin count as unchanged.
    :type margin: float
    :return: ``n``, ``better`` / ``worse`` / ``same`` counts, median and pooled CER of the re-reads and of the
        failed reads, ``healthy`` (re-reads the detector passes) and ``policy_pooled_cer`` (keep the re-read
        only when it is healthy, else the failed read).
    :rtype: Dict[str, Any]
    """
    if not rows:
        return {"n": 0}
    ref = sum(r["ref"] for r in rows)
    keep = [r["err"] if r["label"] == LABEL_OK else r["base_err"] for r in rows]
    return {"n": len(rows),
            "better": sum(1 for r in rows if r["cer"] < r["base_cer"] - margin),
            "worse": sum(1 for r in rows if r["cer"] > r["base_cer"] + margin),
            "same": sum(1 for r in rows if abs(r["cer"] - r["base_cer"]) <= margin),
            "median_cer": st.median(r["cer"] for r in rows), "median_base_cer": st.median(r["base_cer"] for r in rows),
            "pooled_cer": sum(r["err"] for r in rows) / max(1, ref),
            "pooled_base_cer": sum(r["base_err"] for r in rows) / max(1, ref),
            "healthy": sum(1 for r in rows if r["label"] == LABEL_OK),
            "policy_pooled_cer": sum(keep) / max(1, ref)}
