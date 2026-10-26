# File name: pipeline_decode_policy.py
# Date: 10/12/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""The benchmark decode policy (:mod:`src.datasets.consensus.decode_policy`) for the two-reader consensus pipeline.

:mod:`~src.datasets.consensus.two_reader_lines` reads every page whole with its grounded JSON prompt. Under
``--decode-policy`` a page is read the way the decode policy validated on religious-140 and PGP-131 (2026-10-12)
reads it:

``columns``  Kraken's line geometry splits the page into two or more columns (and, by default, the image shows a
             central vertical ink gap). Each column crop is read on its own, rightmost first. A crop read that fails
             health is read again as the crop's top and bottom halves. A unit both fail takes the whole-page read's
             lines whose box centres fall inside its crop; the whole page is read for that only when a unit fails.
``rescue``   a single column whose whole-page read fails health (loop, capped, near-empty, skipped text). The
             column's top and bottom halves (12 % overlap) are read and joined.
``keep``     a single column whose whole-page read is healthy. That read stands (one request, as before).

Kept from the benchmark: the plan (:func:`~src.datasets.consensus.decode_policy.plan_page` on the Kraken fragments),
the per-unit fallback order (crop, then a re-read, then the whole-page share), the page fallbacks of
:func:`~src.datasets.consensus.decode_policy.assemble` and its two safety rules. An assembly holding fewer than 0.8x
the letters of a healthy whole-page read gives way to that read. On a page where Kraken under-segmented the text,
over-generation is judged against 1.5x the whole-page read instead of 3x Kraken
(:func:`~src.datasets.consensus.decode_policy.weak_kraken_limit`).

Changed for the pipeline, and why:

* A reply is the pipeline's grounded JSON, so it is judged with
  :func:`~src.datasets.consensus.read_health.read_health` on the raw reply (its ``capped`` test looks for a closed
  JSON array) plus the policy's over-generation guard (:func:`judge_reply`), not with the plain-text
  :func:`~src.datasets.consensus.decode_policy.health_of`. Texts joined from several replies (halves, shares) are
  judged as the benchmark judges joined texts: :func:`~src.datasets.consensus.decode_policy.judge_read`, no cap.
* Lines keep their boxes. Crop boxes are mapped to the page frame (:func:`crop_lines_to_page`), columns are
  concatenated rightmost first (:func:`merge_column_lines`) and halves are joined by box and text
  (:func:`merge_half_lines`). ``two_reader_lines.sidecar`` then matches the page's lines against Kraken under the
  unchanged line rule.
* A failed column crop is re-read as two halves of the crop (the benchmark reads bands), and the whole-page share is
  geometric (the benchmark matches whole-page lines to Kraken lines by text).

Pure logic: no file or network I/O. :func:`decode_page` drives the reads through an async reader the caller injects
(the pipeline's reader writes each crop JPEG and asks LM Studio: ``two_reader_lines.PolicyPageReader``).
"""
import dataclasses
import statistics
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from src.datasets.consensus import decode_policy as dp
from src.datasets.consensus.read_health import ReadHealth, letter_count, read_health
from src.datasets.consensus.reread import JOIN_MIN_SIMILARITY, line_match, to_page_box

POLICY_VERSION = "pipeline-decode-policy-v1-20261012"   # stamped into every trace; bump when a decision rule changes
PAGE_UNIT = "page"                 # unit name and role of the whole-page request
SOURCE_HALVES = "halves"           # a failed column crop re-read as its two halves (where the benchmark reads bands)
CROP_JPEG_QUALITY = 92             # the benchmark driver's (and linewise's) crop quality
MODE_LIVE = "live"                 # a new page: every request is made now
MODE_REVISIT = "revisit"           # a recorded page: its whole-page read comes from the raw cache
OVERLAP_REACH = 1.0                # a half's line counts as in the overlap within one overlap height of it (box drift)
BLANK_KRAKEN_REASON = "blank_kraken_"   # plan reason prefix of a rescue skipped for too few Kraken letters

Box = Tuple[int, int, int, int]
Line = Dict[str, Any]


# ----------------------------------------------------------------------------- replies


@dataclass
class Reply:
    """One VLM request's outcome, its lines already in the page's 0-1000 frame.

    :param unit: Unit name (``page``, ``col0``, ``col0.top``, ``top``, ...).
    :param role: ``page``, :data:`~src.datasets.consensus.decode_policy.ROLE_COLUMN` or
        :data:`~src.datasets.consensus.decode_policy.ROLE_HALF`.
    :param crop: Pixel crop box on the page (None for the whole page).
    :param raw: The raw reply.
    :param parsed: Whether the reply parsed as a JSON array (``parse_grounded``).
    :param lines: ``{"text", "box"}`` per parsed line, page frame.
    :param seconds: Request time.
    :param cached: The reply comes from the raw cache (a revisit's whole-page read), not from a request made now.
    """

    unit: str
    role: str
    crop: Optional[Box]
    raw: str
    parsed: bool
    lines: List[Line] = field(default_factory=list)
    seconds: float = 0.0
    cached: bool = False

    @property
    def texts(self) -> List[str]:
        """Texts of the parsed lines.

        :return: Line texts in reply order.
        :rtype: List[str]
        """
        return [ln["text"] for ln in self.lines]

    def to_trace(self) -> Dict[str, Any]:
        """JSON-ready summary for the raw-cache trace (the reply itself is stored in ``vlm_raw``).

        :return: Unit, role, crop, parse result, line and letter counts, reply length, seconds, cached.
        :rtype: Dict[str, Any]
        """
        return {"unit": self.unit, "role": self.role, "crop": list(self.crop) if self.crop else None,
                "parsed": self.parsed, "n_lines": len(self.lines), "letters": letter_count(_text(self.lines)),
                "reply_chars": len(self.raw or ""), "seconds": self.seconds, "cached": self.cached}


@dataclass
class UnitReplies:
    """What was read for one unit of the plan.

    :param unit: The unit (a column, or one half of a rescued column).
    :param crop: The unit's crop read (None: not read).
    :param halves: For a column whose crop read failed: its two half units with their replies, top first.
    """

    unit: dp.ReadUnit
    crop: Optional[Reply] = None
    halves: List[Tuple[dp.ReadUnit, Reply]] = field(default_factory=list)


ReadFn = Callable[[str, str, Optional[Box]], Awaitable[Optional[Reply]]]


@dataclass
class PolicyResult:
    """A page decoded under the policy.

    :param plan: The final plan.
    :param lines: The page's lines (``{"text", "box"}``, page frame) for ``sidecar``.
    :param parsed: The page-level ``parsed`` flag.
    :param replies: Every reply in request order (a revisit's cached whole-page read first).
    :param trace: JSON-ready account of every decision.
    """

    plan: dp.PagePlan
    lines: List[Line]
    parsed: bool
    replies: List[Reply]
    trace: Dict[str, Any]

    @property
    def calls(self) -> int:
        """Model requests made for the page now (the cached whole-page read not counted).

        :return: Request count.
        :rtype: int
        """
        return sum(1 for r in self.replies if not r.cached)

    @property
    def vlm_raw(self) -> Union[str, List[str]]:
        """The raw-cache ``vlm_raw`` value: the reply when there is one, else every reply in request order.

        :return: A string, or a list parallel to ``trace["requests"]``.
        :rtype: Union[str, List[str]]
        """
        return self.replies[0].raw if len(self.replies) == 1 else [r.raw for r in self.replies]


# ----------------------------------------------------------------------------- Kraken fragments as benchmark units


def frag_pixel_box(box: Sequence[float], width: int, height: int) -> List[float]:
    """A fragment's 0-1000 box in the page's pixels (the inverse of ``two_reader_lines.run_kraken``'s scaling).

    :param box: ``[x1, y1, x2, y2]`` in 0-1000 units.
    :type box: Sequence[float]
    :param width: Page width in pixels.
    :type width: int
    :param height: Page height in pixels.
    :type height: int
    :return: The box in pixels, rounded to 0.01.
    :rtype: List[float]
    """
    return [round(box[0] * width / 1000, 2), round(box[1] * height / 1000, 2),
            round(box[2] * width / 1000, 2), round(box[3] * height / 1000, 2)]


def kraken_json_from_frags(frags: Sequence[Mapping[str, Any]], width: int, height: int) -> Dict[str, Any]:
    """The pipeline's Kraken fragments as the ``/transcribe_lines`` response the benchmark policy plans from.

    :param frags: ``{"text", "conf", "box"}`` fragments (0-1000 boxes, page frame).
    :type frags: Sequence[Mapping[str, Any]]
    :param width: Page width in pixels.
    :type width: int
    :param height: Page height in pixels.
    :type height: int
    :return: ``{"lines": [{"text", "bbox", "confidence"}], "image_size", "view"}`` with pixel boxes and no rotation
        (the fragments are already in the oriented page's frame), so a line's index is its fragment's index.
    :rtype: Dict[str, Any]
    """
    return {"lines": [{"text": f.get("text") or "", "bbox": frag_pixel_box(f["box"], width, height),
                       "confidence": f.get("conf")} for f in frags],
            "image_size": [int(width), int(height)], "view": {"rotation": 0}}


def units_from_frags(frags: Sequence[Mapping[str, Any]], width: int,
                     height: int) -> Tuple[List[Dict[str, Any]], float, Tuple[int, int]]:
    """The benchmark's layout units, built from the pipeline's fragments.

    :param frags: ``{"text", "conf", "box"}`` fragments (0-1000 boxes).
    :type frags: Sequence[Mapping[str, Any]]
    :param width: Page width in pixels.
    :type width: int
    :param height: Page height in pixels.
    :type height: int
    :return: :func:`~src.datasets.consensus.decode_policy.kraken_units` of :func:`kraken_json_from_frags`:
        ``(units, line_h, (width, height))``, each unit ``{"idx", "text", "box" (pixels), "gap", "letters"}`` with
        ``letters`` the Hebrew letters the column split counts (Kraken lines with Arabic letters only never define a
        column, as on the benchmark).
    :rtype: Tuple[List[Dict[str, Any]], float, Tuple[int, int]]
    """
    return dp.kraken_units(kraken_json_from_frags(frags, width, height))


# ----------------------------------------------------------------------------- health


def judge_reply(reply: Reply, kraken_texts: Sequence[str], cap: int, th: dp.PolicyThresholds,
                runaway_letters: Optional[float] = None) -> ReadHealth:
    """Verdict on one grounded JSON reply: ``read_health`` on the raw reply plus the policy's over-generation guard.

    The guard is the one :func:`~src.datasets.consensus.decode_policy.judge_read` adds to plain-text reads.

    :param reply: The reply.
    :type reply: Reply
    :param kraken_texts: Kraken line texts of the region the reply read.
    :type kraken_texts: Sequence[str]
    :param cap: ``max_tokens`` of the request.
    :type cap: int
    :param th: Policy thresholds.
    :type th: dp.PolicyThresholds
    :param runaway_letters: The page's :func:`~src.datasets.consensus.decode_policy.weak_kraken_limit` (None:
        Kraken's rule).
    :type runaway_letters: Optional[float]
    :return: The verdict; ``overgen`` is added to the flags of an over-generated read and becomes its label when
        nothing else failed.
    :rtype: ReadHealth
    """
    health = read_health(reply.raw or "", reply.texts, list(kraken_texts), cap, th=th.health)
    if not dp.is_overgenerated(health.facts, th, runaway_letters):
        return health
    return ReadHealth(health.label if health.failed else dp.LABEL_OVERGEN, health.flags + (dp.LABEL_OVERGEN,),
                      health.facts)


# ----------------------------------------------------------------------------- planning


def plan_from_frags(frags: Sequence[Mapping[str, Any]], width: int, height: int, whole_raw: Optional[str],
                    whole_lines: Optional[Sequence[Mapping[str, Any]]], ink_gap_x: Optional[int],
                    thresholds: dp.PolicyThresholds, rescue_min_kraken_letters: int = 0) -> dp.PagePlan:
    """Plan one page from its Kraken fragments and (when there is one) its whole-page read.

    The geometry is :func:`~src.datasets.consensus.decode_policy.plan_page`'s: columns rightmost first with pixel
    crops clipped to the gutters, and for a rescue the single column's top and bottom halves with
    ``thresholds.half_overlap`` overlap (the whole page halved when Kraken read under 100 letters). The whole-page
    read is judged on its raw JSON reply (:func:`~src.datasets.consensus.read_health.read_health`, cap
    ``thresholds.page_token_cap``).

    :param frags: ``{"text", "conf", "box"}`` fragments (0-1000 boxes).
    :type frags: Sequence[Mapping[str, Any]]
    :param width: Page width in pixels.
    :type width: int
    :param height: Page height in pixels.
    :type height: int
    :param whole_raw: Raw whole-page reply (None: not read yet).
    :type whole_raw: Optional[str]
    :param whole_lines: Its parsed lines (``{"text", "box"}``).
    :type whole_lines: Optional[Sequence[Mapping[str, Any]]]
    :param ink_gap_x: :func:`~src.datasets.consensus.decode_policy.central_ink_gap` of the page (None: no gap, or
        not measured; with ``thresholds.require_ink_gap`` a Kraken split then stays one column).
    :type ink_gap_x: Optional[int]
    :param thresholds: Policy thresholds.
    :type thresholds: dp.PolicyThresholds
    :param rescue_min_kraken_letters: A page whose Kraken fragments hold fewer letters than this is never planned as
        a rescue: it is kept (reason ``blank_kraken_<letters>``). 0 = the validated policy. Columns are unaffected.
    :type rescue_min_kraken_letters: int
    :return: ``columns`` (whole-page health attached when given); without a whole-page read, ``rescue`` with reason
        ``whole_page_missing`` for a single column; with one, ``rescue`` (reason ``whole_page_<label>``) when its
        label is in ``thresholds.rescue_labels``, else ``keep``; a rescue of a Kraken-blank page as above.
    :rtype: dp.PagePlan
    """
    kraken = kraken_json_from_frags(frags, width, height)
    plan = dp.plan_page(kraken, None, thresholds, ink_gap_x=ink_gap_x)   # geometry; the JSON read is judged below
    units, line_h, size = dp.kraken_units(kraken)

    def keep(reason: str, whole: Optional[ReadHealth]) -> dp.PagePlan:
        """The page kept as read: no units, the column itself (not the halved page) as its one column.

        :param reason: Plan reason.
        :type reason: str
        :param whole: The whole-page read's health (None: not read yet).
        :type whole: Optional[ReadHealth]
        :return: The ``keep`` plan.
        :rtype: dp.PagePlan
        """
        columns = dp.detect_columns(units, line_h, size, thresholds.require_ink_gap, ink_gap_x,
                                    thresholds.crop_margin_line_heights)
        return dataclasses.replace(plan, action=dp.ACTION_KEEP, reason=reason, columns=tuple(columns), units=(),
                                   whole_page=whole)

    whole = None
    if whole_raw is not None:
        whole = read_health(whole_raw, [ln["text"] for ln in whole_lines or []], [u["text"] for u in units],
                            thresholds.page_token_cap, th=thresholds.health)
        if plan.action == dp.ACTION_COLUMNS:
            return dataclasses.replace(plan, whole_page=whole)
        if whole.label not in thresholds.rescue_labels:
            return keep(f"whole_page_{whole.label}", whole)
        plan = dataclasses.replace(plan, reason=plan.reason.replace("missing", whole.label, 1), whole_page=whole)
    if plan.action == dp.ACTION_RESCUE and plan.kraken_letters < rescue_min_kraken_letters:
        return keep(f"{BLANK_KRAKEN_REASON}{plan.kraken_letters}", whole)
    return plan


def is_blank_skip(plan: dp.PagePlan) -> bool:
    """Whether a plan keeps a page only because its Kraken fragments were too blank to rescue it.

    :param plan: A plan from :func:`plan_from_frags`.
    :type plan: dp.PagePlan
    :return: True for a ``keep`` whose reason is ``blank_kraken_<letters>``.
    :rtype: bool
    """
    return plan.action == dp.ACTION_KEEP and plan.reason.startswith(BLANK_KRAKEN_REASON)


def column_halves(plan: dp.PagePlan, unit: dp.ReadUnit, units: Sequence[Dict[str, Any]],
                  th: dp.PolicyThresholds) -> List[dp.ReadUnit]:
    """The two halves a failed column crop is re-read as (:func:`~src.datasets.consensus.decode_policy.half_units`).

    :param plan: The page's plan.
    :type plan: dp.PagePlan
    :param unit: A column unit of the plan (``col<k>``).
    :type unit: dp.ReadUnit
    :param units: Layout units from :func:`units_from_frags`.
    :type units: Sequence[Dict[str, Any]]
    :param th: Policy thresholds (``half_overlap``).
    :type th: dp.PolicyThresholds
    :return: ``col<k>.top`` and ``col<k>.bottom``, each with the column's Kraken lines whose centre falls inside it.
    :rtype: List[dp.ReadUnit]
    """
    column = next(c for c in plan.columns if f"col{c.index}" == unit.name)
    return [dataclasses.replace(h, name=f"{unit.name}.{h.name}") for h in dp.half_units(column, units, th.half_overlap)]


# ----------------------------------------------------------------------------- boxes and line merging


def _text(lines: Sequence[Mapping[str, Any]]) -> str:
    """Line texts joined with newlines.

    :param lines: ``{"text", ...}`` lines.
    :type lines: Sequence[Mapping[str, Any]]
    :return: The joined text.
    :rtype: str
    """
    return "\n".join(ln["text"] for ln in lines)


def crop_box_to_page(crop: Sequence[float], width: int, height: int) -> List[float]:
    """A pixel box on the page in the page's 0-1000 frame.

    :param crop: ``(x1, y1, x2, y2)`` in pixels.
    :type crop: Sequence[float]
    :param width: Page width in pixels.
    :type width: int
    :param height: Page height in pixels.
    :type height: int
    :return: The box in 0-1000 units.
    :rtype: List[float]
    """
    return [1000 * crop[0] / width, 1000 * crop[1] / height, 1000 * crop[2] / width, 1000 * crop[3] / height]


def crop_lines_to_page(lines: Sequence[Mapping[str, Any]], crop_box: Sequence[int], width: int,
                       height: int) -> List[Line]:
    """Lines read on a crop, their boxes mapped to the page's 0-1000 frame
    (:func:`~src.datasets.consensus.reread.to_page_box`).

    :param lines: ``{"text", "box"}`` with boxes in the crop's 0-1000 frame (``parse_grounded`` of the crop's reply).
    :type lines: Sequence[Mapping[str, Any]]
    :param crop_box: Pixel box of the crop on the page.
    :type crop_box: Sequence[int]
    :param width: Page width in pixels.
    :type width: int
    :param height: Page height in pixels.
    :type height: int
    :return: ``{"text", "box"}`` with page-frame boxes (rounded to 0.1), in reply order.
    :rtype: List[Line]
    """
    return [{"text": ln["text"], "box": to_page_box(ln["box"], crop_box, width, height)} for ln in lines]


def merge_column_lines(columns_lines: Sequence[Sequence[Mapping[str, Any]]]) -> List[Line]:
    """Column reads joined into the page's reading order: rightmost column first, each column's lines in its order.

    :param columns_lines: Each column's lines (``{"text", "box"}``, page frame) in reading order.
    :type columns_lines: Sequence[Sequence[Mapping[str, Any]]]
    :return: The columns concatenated by decreasing median line centre x (ties and empty columns keep the order
        given, empty ones last).
    :rtype: List[Line]
    """
    def centre_x(lines: Sequence[Mapping[str, Any]]) -> float:
        """Median horizontal centre of a column's lines.

        :param lines: The column's lines.
        :type lines: Sequence[Mapping[str, Any]]
        :return: The median centre, or -inf for a column without lines.
        :rtype: float
        """
        if not lines:
            return float("-inf")
        return statistics.median((ln["box"][0] + ln["box"][2]) / 2 for ln in lines)

    order = sorted(range(len(columns_lines)), key=lambda k: -centre_x(columns_lines[k]))
    return [dict(ln) for k in order for ln in columns_lines[k]]


def half_overlap_box(top_crop: Sequence[int], bottom_crop: Sequence[int], width: int, height: int) -> List[float]:
    """The band both halves of a split read contain, in the page's 0-1000 frame.

    :param top_crop: Pixel crop of the top half.
    :type top_crop: Sequence[int]
    :param bottom_crop: Pixel crop of the bottom half.
    :type bottom_crop: Sequence[int]
    :param width: Page width in pixels.
    :type width: int
    :param height: Page height in pixels.
    :type height: int
    :return: ``[x1, y1, x2, y2]``: the halves' x-extent, from the bottom half's top to the top half's bottom.
    :rtype: List[float]
    """
    x0, x1 = min(top_crop[0], bottom_crop[0]), max(top_crop[2], bottom_crop[2])
    return crop_box_to_page((x0, bottom_crop[1], x1, top_crop[3]), width, height)


def merge_half_lines(top: Sequence[Mapping[str, Any]], bottom: Sequence[Mapping[str, Any]],
                     overlap_box: Sequence[float], min_sim: float = JOIN_MIN_SIMILARITY,
                     reach: float = OVERLAP_REACH) -> Tuple[List[Line], int]:
    """Join the line reads of a top and a bottom half, keeping each line both halves read once.

    A bottom line is a duplicate of a top line when both lie in the overlap band (box centres within ``reach``
    overlap heights of ``overlap_box``, which absorbs the model's box drift on a crop) and their texts match
    (:func:`~src.datasets.consensus.reread.line_match` >= ``min_sim``). Pairs are taken best match first, each line
    in at most one pair. A duplicate is dropped from the bottom read, and when it holds more letters than its top
    reading it replaces that reading (the top crop's edge may have cut the line). A matching text elsewhere on the
    page (a repeated formula) is never a duplicate.

    :param top: Top half's lines (``{"text", "box"}``, page frame) in reading order.
    :type top: Sequence[Mapping[str, Any]]
    :param bottom: Bottom half's lines in reading order.
    :type bottom: Sequence[Mapping[str, Any]]
    :param overlap_box: :func:`half_overlap_box` of the two crops.
    :type overlap_box: Sequence[float]
    :param min_sim: Minimum line match for two readings of one line.
    :type min_sim: float
    :param reach: How far beyond the overlap band (in band heights) a line still counts as in it.
    :type reach: float
    :return: ``(top lines then the bottom's remaining lines, number of bottom lines dropped)``.
    :rtype: Tuple[List[Line], int]
    """
    band_y0, band_y1 = sorted((overlap_box[1], overlap_box[3]))
    margin = reach * max(band_y1 - band_y0, 1.0)
    lo, hi = band_y0 - margin, band_y1 + margin

    def in_band(line: Mapping[str, Any]) -> bool:
        """Whether a line's box centre lies in the (widened) overlap band.

        :param line: A line.
        :type line: Mapping[str, Any]
        :return: True inside the band.
        :rtype: bool
        """
        return lo <= (line["box"][1] + line["box"][3]) / 2 <= hi

    pairs = []
    for i, b in enumerate(bottom):
        if not in_band(b):
            continue
        for j, t in enumerate(top):
            if in_band(t):
                score = line_match(t["text"], b["text"])
                if score >= min_sim:
                    pairs.append((score, i, j))
    merged = [dict(t) for t in top]
    dropped, taken = set(), set()
    for _, i, j in sorted(pairs, key=lambda p: (-p[0], p[1], p[2])):
        if i in dropped or j in taken:
            continue
        dropped.add(i)
        taken.add(j)
        if letter_count(bottom[i]["text"]) > letter_count(merged[j]["text"]):
            merged[j] = dict(bottom[i])
    return merged + [dict(b) for i, b in enumerate(bottom) if i not in dropped], len(dropped)


def share_lines(whole_lines: Sequence[Mapping[str, Any]], crop: Sequence[int], width: int, height: int) -> List[Line]:
    """The whole-page read's share of a unit: its lines whose box centre falls inside the unit's crop.

    :param whole_lines: Whole-page lines (``{"text", "box"}``, page frame).
    :type whole_lines: Sequence[Mapping[str, Any]]
    :param crop: The unit's pixel crop.
    :type crop: Sequence[int]
    :param width: Page width in pixels.
    :type width: int
    :param height: Page height in pixels.
    :type height: int
    :return: Those lines, in whole-page order.
    :rtype: List[Line]
    """
    box = crop_box_to_page(crop, width, height)
    return [dict(ln) for ln in whole_lines if dp._centre_inside(ln["box"], box)]


# ----------------------------------------------------------------------------- assembly


def _attempt(source: str, lines: Sequence[Mapping[str, Any]], health: ReadHealth) -> Dict[str, Any]:
    """Trace record of one candidate reading of a unit.

    :param source: ``crop``, ``halves`` or ``share``.
    :type source: str
    :param lines: The candidate's lines.
    :type lines: Sequence[Mapping[str, Any]]
    :param health: Its verdict.
    :type health: ReadHealth
    :return: Source, letters, line count and the verdict.
    :rtype: Dict[str, Any]
    """
    return {"source": source, "letters": letter_count(_text(lines)), "n_lines": len(lines),
            "health": health.to_dict()}


def resolve_unit_lines(replies: UnitReplies, whole: Optional[Reply], th: dp.PolicyThresholds,
                       runaway_letters: Optional[float], width: int,
                       height: int) -> Tuple[List[Line], bool, Dict[str, Any]]:
    """Pick one unit's lines: the first healthy candidate of crop, then halves, then the whole-page share.

    The crop reply is judged by :func:`judge_reply` against the unit's Kraken lines. The halves pass when each half's
    reply passes :func:`judge_reply` against its own Kraken lines and their joined text passes
    :func:`~src.datasets.consensus.decode_policy.judge_read` (no cap) against the unit's. The share
    (:func:`share_lines`) is judged like a joined text. With no healthy candidate the unit is ``unresolved`` and its
    lines are the best effort, as in :func:`~src.datasets.consensus.decode_policy.resolve_unit`: halves, else crop,
    else share.

    :param replies: What was read for the unit.
    :type replies: UnitReplies
    :param whole: The whole-page read (None: none, so no share).
    :type whole: Optional[Reply]
    :param th: Policy thresholds.
    :type th: dp.PolicyThresholds
    :param runaway_letters: The page's :func:`~src.datasets.consensus.decode_policy.weak_kraken_limit` (None:
        Kraken's rule).
    :type runaway_letters: Optional[float]
    :param width: Page width in pixels.
    :type width: int
    :param height: Page height in pixels.
    :type height: int
    :return: ``(lines, parsed, trace)``; ``parsed`` is the chosen candidate's (every reply it joins parsed); the
        trace's ``used`` names the source or says ``unresolved``.
    :rtype: Tuple[List[Line], bool, Dict[str, Any]]
    """
    unit = replies.unit
    texts = unit.kraken_texts
    candidates: Dict[str, Tuple[List[Line], bool]] = {}
    attempts: List[Dict[str, Any]] = []
    used: Optional[str] = None
    if replies.crop is not None:
        health = judge_reply(replies.crop, texts, th.crop_token_cap, th, runaway_letters)
        candidates[dp.SOURCE_CROP] = (list(replies.crop.lines), replies.crop.parsed)
        attempts.append(_attempt(dp.SOURCE_CROP, replies.crop.lines, health))
        used = None if health.failed else dp.SOURCE_CROP
    if used is None and replies.halves:
        (top_unit, top), (bottom_unit, bottom) = replies.halves
        lines, dropped = merge_half_lines(top.lines, bottom.lines,
                                          half_overlap_box(top_unit.crop, bottom_unit.crop, width, height))
        verdicts = [judge_reply(r, h.kraken_texts, th.crop_token_cap, th, runaway_letters) for h, r in replies.halves]
        joined = dp.judge_read(_text(lines), texts, None, th, runaway_letters)
        health = next((v for v in verdicts if v.failed), joined)
        candidates[SOURCE_HALVES] = (lines, all(r.parsed for _, r in replies.halves if r.lines))
        attempts.append({**_attempt(SOURCE_HALVES, lines, health), "overlap_lines_dropped": dropped,
                         "halves": [{"unit": h.name, "label": v.label, "flags": list(v.flags)}
                                    for (h, _), v in zip(replies.halves, verdicts)]})
        used = None if health.failed else SOURCE_HALVES
    if used is None and whole is not None:
        lines = share_lines(whole.lines, unit.crop, width, height)
        if lines:
            health = dp.judge_read(_text(lines), texts, None, th, runaway_letters)
            candidates[dp.SOURCE_SHARE] = (lines, whole.parsed)
            attempts.append(_attempt(dp.SOURCE_SHARE, lines, health))
            used = None if health.failed else dp.SOURCE_SHARE
    if used is not None:
        lines, parsed = candidates[used]
    else:
        lines, parsed = next((candidates[s] for s in (SOURCE_HALVES, dp.SOURCE_CROP, dp.SOURCE_SHARE)
                              if candidates.get(s) and candidates[s][0]), ([], False))
    trace = {**unit.to_dict(), "attempts": attempts, "used": used or "unresolved", "resolved": used is not None,
             "letters": letter_count(_text(lines)), "n_lines": len(lines)}
    return lines, parsed and bool(lines), trace


def assemble_lines(plan: dp.PagePlan, replies: Mapping[str, UnitReplies], whole: Optional[Reply],
                   th: dp.PolicyThresholds, width: int, height: int) -> Tuple[List[Line], bool, Dict[str, Any]]:
    """The page's lines from the unit reads, with the page fallbacks of
    :func:`~src.datasets.consensus.decode_policy.assemble`.

    * ``keep``: the whole-page read's lines.
    * Every unit resolved: the columns rightmost first (:func:`merge_column_lines`), or the halves joined
      (:func:`merge_half_lines`).
    * Some unit unresolved and a healthy whole-page read, or every unit unresolved and any whole-page read: that
      read (``page_fallback`` says why).
    * Some unit unresolved and the whole-page read unhealthy or missing: the assembly with each unresolved unit's
      best effort (``assembled_partial``).
    * An assembly holding fewer than ``th.assembled_min_ratio`` x a healthy whole-page read's letters: that read
      (``page_fallback`` :data:`~src.datasets.consensus.decode_policy.FALLBACK_SHORTER`).

    Units are judged with :func:`~src.datasets.consensus.decode_policy.weak_kraken_limit` when Kraken under-segmented
    a page whose whole-page read is healthy.

    :param plan: The page's plan (``whole_page`` = the whole-page read's health, when there is one).
    :type plan: dp.PagePlan
    :param replies: ``unit name -> UnitReplies`` (missing units count as not read).
    :type replies: Mapping[str, UnitReplies]
    :param whole: The whole-page read (None when the page was not read whole).
    :type whole: Optional[Reply]
    :param th: Policy thresholds.
    :type th: dp.PolicyThresholds
    :param width: Page width in pixels.
    :type width: int
    :param height: Page height in pixels.
    :type height: int
    :return: ``(lines, parsed, trace)``. ``parsed`` is the whole-page read's when the page falls back to it, else
        True when the page has lines and every unit that gave lines took them from parsed replies.
    :rtype: Tuple[List[Line], bool, Dict[str, Any]]
    """
    whole_text = None if whole is None else _text(whole.lines)
    whole_letters = None if whole_text is None else letter_count(whole_text)
    runaway = dp.weak_kraken_limit(plan, whole_text, th)
    trace: Dict[str, Any] = {"action": plan.action, "reason": plan.reason, "units": [], "page_fallback": None,
                             "whole_page_letters": whole_letters, "kraken_letters": plan.kraken_letters,
                             "weak_kraken_relaxation": runaway is not None, "overgen_runaway_letters": runaway,
                             "assembled_letters": None}
    if plan.action == dp.ACTION_KEEP:
        lines = list(whole.lines) if whole is not None else []
        trace.update(final_source=dp.FINAL_WHOLE_PAGE, final_letters=letter_count(_text(lines)))
        return lines, whole is not None and whole.parsed, trace
    unit_lines: List[List[Line]] = []
    unit_parsed: List[bool] = []
    for unit in plan.units:
        lines, parsed, unit_trace = resolve_unit_lines(replies.get(unit.name) or UnitReplies(unit), whole, th,
                                                       runaway, width, height)
        unit_lines.append(lines)
        unit_parsed.append(parsed)
        trace["units"].append(unit_trace)
    resolved = [u["resolved"] for u in trace["units"]]
    whole_ok = whole is not None and plan.whole_page is not None and not plan.whole_page.failed
    if not all(resolved) and whole is not None and (whole_ok or not any(resolved)):
        trace.update(page_fallback="unit_unresolved" if whole_ok else "all_units_unresolved",
                     final_source=dp.FINAL_WHOLE_PAGE, final_letters=whole_letters)
        return list(whole.lines), whole.parsed, trace
    if plan.action == dp.ACTION_RESCUE and len(plan.units) == 2:
        top, bottom = plan.units
        final, dropped = merge_half_lines(unit_lines[0], unit_lines[1],
                                          half_overlap_box(top.crop, bottom.crop, width, height))
        trace["overlap_lines_dropped"] = dropped
    else:
        final = merge_column_lines(unit_lines)
    trace["assembled_letters"] = letter_count(_text(final))
    if whole_ok and trace["assembled_letters"] < th.assembled_min_ratio * whole_letters:
        # The units resolved, but together they hold far less text than a healthy whole-page read: some unit's
        # source lost text, so the whole-page read stands.
        trace.update(page_fallback=dp.FALLBACK_SHORTER, final_source=dp.FINAL_WHOLE_PAGE, final_letters=whole_letters)
        return list(whole.lines), whole.parsed, trace
    trace.update(final_source=dp.FINAL_ASSEMBLED if all(resolved) else dp.FINAL_PARTIAL,
                 final_letters=trace["assembled_letters"])
    parsed = bool(final) and all(p for lines, p in zip(unit_lines, unit_parsed) if lines)
    return final, parsed, trace


# ----------------------------------------------------------------------------- the page


async def decode_page(frags: Sequence[Mapping[str, Any]], width: int, height: int, th: dp.PolicyThresholds,
                      read: ReadFn, ink_gap: Callable[[], Optional[int]], whole: Optional[Reply] = None,
                      rescue_min_kraken_letters: int = 0) -> Optional[PolicyResult]:
    """Decode one page under the policy, one request at a time.

    Live (``whole`` None): a column page reads its column crops; a single-column page is read whole, and on a
    failed read its halves are read. Revisit (``whole`` = the page's cached whole-page read): only the units are read.
    A column crop that fails :func:`judge_reply` is re-read as its two halves (:func:`column_halves`). A live column
    page is read whole only when some unit stays unresolved after that, for its share and the page fallbacks.

    :param frags: Kraken fragments ``{"text", "conf", "box"}`` (0-1000 boxes) the page's record is matched against.
    :type frags: Sequence[Mapping[str, Any]]
    :param width: Page width in pixels.
    :type width: int
    :param height: Page height in pixels.
    :type height: int
    :param th: Policy thresholds (``page_token_cap`` / ``crop_token_cap`` = the requests' ``max_tokens``).
    :type th: dp.PolicyThresholds
    :param read: ``await read(unit_name, role, pixel_crop_or_None)`` makes one request (crop None = the whole page)
        and returns its :class:`Reply` with page-frame lines, or None when the request failed.
    :type read: ReadFn
    :param ink_gap: Returns :func:`~src.datasets.consensus.decode_policy.central_ink_gap` of the page; called only
        when Kraken splits the page and ``th.require_ink_gap`` is set.
    :type ink_gap: Callable[[], Optional[int]]
    :param whole: The cached whole-page read of a recorded page (``cached=True``), or None for a live page.
    :type whole: Optional[Reply]
    :param rescue_min_kraken_letters: Never rescue a page whose Kraken fragments hold fewer letters
        (:func:`plan_from_frags`; 0 = the validated policy).
    :type rescue_min_kraken_letters: int
    :return: The decoded page, or None when a request failed (nothing is decided then: the page is retried later).
    :rtype: Optional[PolicyResult]
    """
    mode = MODE_LIVE if whole is None else MODE_REVISIT
    replies: List[Reply] = [] if whole is None else [whole]
    probe = plan_from_frags(frags, width, height, None, None, None, th)
    gap = ink_gap() if th.require_ink_gap and probe.kraken_columns >= 2 else None

    def plan_with(reply: Optional[Reply]) -> dp.PagePlan:
        """The page's plan given the whole-page read so far.

        :param reply: The whole-page read, or None.
        :type reply: Optional[Reply]
        :return: The plan.
        :rtype: dp.PagePlan
        """
        if reply is None:
            return plan_from_frags(frags, width, height, None, None, gap, th, rescue_min_kraken_letters)
        return plan_from_frags(frags, width, height, reply.raw, reply.lines, gap, th, rescue_min_kraken_letters)

    async def read_whole() -> Optional[Reply]:
        """Read the page whole and keep the reply.

        :return: The reply, or None when the request failed.
        :rtype: Optional[Reply]
        """
        reply = await read(PAGE_UNIT, PAGE_UNIT, None)
        if reply is not None:
            replies.append(reply)
        return reply

    plan = plan_with(whole)
    if plan.action != dp.ACTION_COLUMNS and whole is None:       # live single column: the whole-page read decides
        whole = await read_whole()
        if whole is None:
            return None
        plan = plan_with(whole)
    unit_replies: Dict[str, UnitReplies] = {}
    if plan.action != dp.ACTION_KEEP:
        units, _, _ = units_from_frags(frags, width, height)
        for unit in plan.units:
            crop = await read(unit.name, unit.role, unit.crop)
            if crop is None:
                return None
            replies.append(crop)
            done = UnitReplies(unit, crop)
            if unit.role == dp.ROLE_COLUMN and judge_reply(crop, unit.kraken_texts, th.crop_token_cap, th).failed:
                for half in column_halves(plan, unit, units, th):
                    reply = await read(half.name, half.role, half.crop)
                    if reply is None:
                        return None
                    replies.append(reply)
                    done.halves.append((half, reply))
            unit_replies[unit.name] = done
        if whole is None and not all(resolve_unit_lines(r, None, th, None, width, height)[2]["resolved"]
                                     for r in unit_replies.values()):
            whole = await read_whole()                           # live column page: the share needs the whole page
            if whole is None:
                return None
            plan = plan_with(whole)
    lines, parsed, assembly = assemble_lines(plan, unit_replies, whole, th, width, height)
    trace = {"version": POLICY_VERSION, "mode": mode, **assembly,
             "calls": sum(1 for r in replies if not r.cached), "requests": [r.to_trace() for r in replies],
             "require_ink_gap": th.require_ink_gap, "rescue_min_kraken_letters": rescue_min_kraken_letters,
             "token_cap": th.crop_token_cap, "plan": plan.to_dict()}
    return PolicyResult(plan, lines, parsed, replies, trace)


def policy_stamp(trace: Mapping[str, Any]) -> Dict[str, Any]:
    """The short ``ai_read["decode_policy"]`` stamp of a record decoded under the policy (the trace stays in the raw cache).

    :param trace: The raw-cache entry's ``decode_policy`` trace.
    :type trace: Mapping[str, Any]
    :return: ``version``, ``action``, ``reason``, ``final_source`` and ``calls``.
    :rtype: Dict[str, Any]
    """
    return {k: trace.get(k) for k in ("version", "action", "reason", "final_source", "calls")}


# ----------------------------------------------------------------------------- dry-run arithmetic


def estimated_calls(plan: dp.PagePlan, whole_cached: bool) -> Tuple[int, int]:
    """Model requests a page costs under the policy.

    :param plan: The page's plan.
    :type plan: dp.PagePlan
    :param whole_cached: The whole-page read exists already (a revisit), so it is never requested.
    :type whole_cached: bool
    :return: ``(base, worst case)``. Base: one request per unit, plus the whole-page read of a live single-column
        page. Worst case adds two half reads per column and, for a live column page, the whole-page read.
    :rtype: Tuple[int, int]
    """
    whole = 0 if whole_cached else 1
    if plan.action == dp.ACTION_KEEP:
        return whole, whole
    if plan.action == dp.ACTION_RESCUE:
        return whole + len(plan.units), whole + len(plan.units)
    return len(plan.units), 3 * len(plan.units) + whole


def plan_row(plan: dp.PagePlan) -> Dict[str, Any]:
    """Dry-run report row of one planned page.

    :param plan: The page's plan.
    :type plan: dp.PagePlan
    :return: Action, reason, column counts, Kraken letters, whole-page label / flags / letter ratio, unit names and
        crops, and the request estimates ``[base, worst case]`` (:func:`estimated_calls`) of a revisit
        (``calls_revisit``: the whole-page read is cached) and of a live read (``calls_live``).
    :rtype: Dict[str, Any]
    """
    whole = plan.whole_page
    return {"action": plan.action, "reason": plan.reason, "kraken_columns": plan.kraken_columns,
            "n_columns": len(plan.columns), "kraken_letters": plan.kraken_letters,
            "n_kraken_lines": plan.n_kraken_lines, "whole_page_label": whole.label if whole else "missing",
            "whole_page_flags": list(whole.flags) if whole else [],
            "whole_page_letter_ratio": whole.facts.letter_ratio if whole else None,
            "units": [{"name": u.name, "crop": list(u.crop), "kraken_letters": u.kraken_letters} for u in plan.units],
            "calls_revisit": list(estimated_calls(plan, True)), "calls_live": list(estimated_calls(plan, False))}
