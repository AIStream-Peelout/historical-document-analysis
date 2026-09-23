# File name: build_documentary_grounding.py
# Date: 9/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Build ``documentary_grounding_v1``: line-grounding rows on documentary page images.

Why: the flagship VLM's grounding rows were trained on KTIV literary scans only,
and on documentary photos its boxes follow a page-shaped prior instead of the
ink.  The two-reader batch (:mod:`src.datasets.consensus.two_reader_lines`)
matches each VLM line to Kraken line fragments; a line both readers agree on
has verified text (the VLM's reading, the better reader) and trustworthy ink
geometry (Kraken's fragments).  Those lines become the two KTIV grounding
families — ``locate`` (phrase -> box) and ``read_box`` (box -> text) — with the
prompt strings imported from :mod:`build_ktiv_dataset`, so the rows are the
same tasks and mix into a run as more rows of the existing families.

Line rule (conservative):

* ``status == "agreed"`` and ``agreement >= 0.8``;
* the VLM text (NFC, whitespace collapsed) has >= 8 Hebrew letters, no
  ``[...]``, no letter of another script (decoding glitches such as ``higher``
  or Arabic letters inside a Hebrew word) and its letters occur exactly once on
  the page (an ambiguous phrase is no locate target, as in the KTIV builder);
* its box is the union of the line's Kraken fragments (``htr_fragments``) —
  never the evidence box or the VLM box: >= 1 fragment, positive, height at
  most 3x the page's median Kraken fragment height (all fragments of the page,
  from the raw cache) and at least 60 px wide at image scale.

Per page image the usable lines are ranked by letter count; the top 4 become
``locate`` rows and the top 2 ``read_box`` rows; pages with fewer than 2 usable
lines are skipped.  Boxes are 0-1000 of the EXIF-oriented original image (the
frame both readers saw), which is stored at original resolution.

Decontamination (per document, before any download) drops a doc when:
its canonical id is a PGP-131 benchmark id; one of the two ids is the other
plus a trailing ``_<n>`` join suffix; its image URL is a benchmark image; its
loose shelfmark key (:meth:`DecontamGate.loose_key`, from the canonical id,
the join base and the merged record's shelfmarks) is a benchmark key; or it
maps to a religious-benchmark manuscript by KTIV sys_num (merged record or
``/KTIV/<sys>/`` image URL) or by loose shelfmark key.  Ids that merely share a
volume after stripping BOTH trailing numbers (T-S 16.106 vs T-S 16.110) are
different fragments and kept unless ``--volume-level-decontam`` is given.

Split: 5 % of DOCUMENTS go to ``val`` by a seeded hash of the doc id, so a
rebuild over a grown batch keeps every earlier val document in val.

Usage (repo root; pass a snapshot copy while the pipeline is appending):
    nice -n 10 .venv/bin/python -m src.finetuning.qwen_hebrew.build_documentary_grounding \\
        --batch <snapshot.jsonl> \\
        --out-dir /Volumes/home/studio_offload/datasets/documentary_grounding_v1 --workers 8
"""

import argparse
import hashlib
import io
import json
import logging
import re
import statistics
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.client import HTTPException
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from datasets import Dataset, DatasetDict
from PIL import Image as PILImage
from PIL import ImageOps, UnidentifiedImageError

from src.finetuning.qwen_hebrew.build_ktiv_dataset import (
    BENCH_PATH,
    FEATURES,
    RELIGIOUS_BENCH_PATH,
    SPLIT_SEED,
    VAL_FRACTION,
    _LOCATE_PROMPT,
    _READBOX_PROMPT,
    _positive,
)
from src.finetuning.qwen_hebrew.ktiv_layout import GAP_TOKEN, hebrew_letters

try:  # optional layer: without it the id / image-url / sys_num layers still apply
    from src.datasets.evaluations.helper_eval_scripts.decontam_gate import DecontamGate
    from src.datasets.merging.institution_tokens import canonical_id
except ImportError:
    DecontamGate = None
    canonical_id = None

PILImage.MAX_IMAGE_PIXELS = None
logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[3]
AI_READS_DIR = _REPO / "src/datasets/raw_data/cairo_genizah/ai_reads"
DEFAULT_BATCH = AI_READS_DIR / "ai_reads_qwen3-vl-8b-heb-v21b-step1200.jsonl"
DEFAULT_RAW_DIR = AI_READS_DIR / "raw"
DEFAULT_OUT = Path("/Volumes/home/studio_offload/datasets/documentary_grounding_v1")
BENCH_IDS_PATH = _REPO / "src/datasets/raw_data/cairo_genizah/decontam/benchmark_ids.json"
MERGED_PATH = _REPO / "src/datasets/raw_data/cairo_genizah/merged/merged_shelfmarks.jsonl"

LABEL_SOURCE = "two_reader_agreed"
SECTION = "line"
MIN_AGREEMENT = 0.8
MIN_LINE_LETTERS = 8
MAX_HEIGHT_MEDIANS = 3.0         # union height <= this many median fragment heights
MIN_WIDTH_PX = 60                # union width at image scale
LOCATE_ROWS_PER_PAGE = 4
READBOX_ROWS_PER_PAGE = 2
MIN_USABLE_LINES = 2
DOWNLOAD_TRIES = 3
DOWNLOAD_TIMEOUT_S = 60.0

_HEB_RE = re.compile(r"[א-ת]")
_JOIN_SUFFIX_RE = re.compile(r"_\d+$")
_KTIV_URL_SYS_RE = re.compile(r"/KTIV/(\d{15,})/")
_URL_PREFIX_RE = re.compile(r"^https?://[^/]+/[^/]+/([^/]+)/")
_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]")
_MERGED_ID_RE = re.compile(r'^\{"canonical_id": "((?:[^"\\]|\\.)*)"')


# ---------------------------------------------------------------------------
# Line rule
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UsableLine:
    """One agreed line that passed every check.

    :ivar index: The line's ``index`` in the batch record.
    :ivar text: Cleaned VLM text (the row target).
    :ivar box: Union of the line's Kraken fragments, 0-1000 ints.
    :ivar letters: Hebrew letter count of ``text``.
    :ivar agreement: Two-reader letters-only similarity.
    """

    index: int
    text: str
    box: Tuple[int, int, int, int]
    letters: int
    agreement: float


def letters_of(text: str) -> str:
    """Return the Hebrew letters of ``text`` in order (letters-only convention).

    :param text: Any text.
    :type text: str
    :return: The letters.
    :rtype: str
    """
    return "".join(_HEB_RE.findall(text or ""))


def clean_text(text: str) -> str:
    """NFC-normalize a VLM line and collapse whitespace runs.

    :param text: Raw VLM line text.
    :type text: str
    :return: Cleaned text.
    :rtype: str
    """
    return unicodedata.normalize("NFC", " ".join((text or "").split()))


def has_foreign_letters(text: str) -> bool:
    """True when ``text`` holds a letter outside the Hebrew block (U+0590-U+05FF).

    :param text: Cleaned line text.
    :type text: str
    :return: Whether a Latin/Arabic/CJK/... letter is present.
    :rtype: bool
    """
    return any(ch.isalpha() and not "֐" <= ch <= "׿" for ch in text)


def safe_stem(doc_id: str, image_index: int) -> str:
    """Filesystem-safe ``<doc_id>__<image_index>`` (the pipeline's raw-cache convention).

    :param doc_id: Canonical document id.
    :type doc_id: str
    :param image_index: Position of the image in the doc's ``image_urls``.
    :type image_index: int
    :return: Stem.
    :rtype: str
    """
    return f"{_SAFE_RE.sub('_', doc_id)}__{image_index}"


def union_box(boxes: Sequence[Sequence[float]]) -> Tuple[int, int, int, int]:
    """Union of 0-1000 boxes as ints clamped to 0-1000 (``norm_box``'s clamping).

    :param boxes: Non-empty list of ``[x0, y0, x1, y1]`` in 0-1000 units.
    :type boxes: Sequence[Sequence[float]]
    :return: ``(x0, y0, x1, y1)``.
    :rtype: Tuple[int, int, int, int]
    """
    def c(v: float) -> int:
        return max(0, min(1000, round(v)))
    return (c(min(b[0] for b in boxes)), c(min(b[1] for b in boxes)),
            c(max(b[2] for b in boxes)), c(max(b[3] for b in boxes)))


def median_height(boxes: Iterable[Sequence[float]]) -> float:
    """Median height of the positive-height boxes (0.0 when there are none).

    :param boxes: Boxes in 0-1000 units.
    :type boxes: Iterable[Sequence[float]]
    :return: Median height in 0-1000 units of the image height.
    :rtype: float
    """
    heights = [b[3] - b[1] for b in boxes if b[3] > b[1]]
    return float(statistics.median(heights)) if heights else 0.0


def box_rejection(box: Sequence[int], median_h: float, image_width: int) -> Optional[str]:
    """Geometric sanity check of a line box.

    :param box: Union box, 0-1000 ints.
    :type box: Sequence[int]
    :param median_h: The page's median Kraken fragment height (0-1000 units).
    :type median_h: float
    :param image_width: Oriented image width in pixels.
    :type image_width: int
    :return: Rejection reason, or None when the box passes.
    :rtype: Optional[str]
    """
    if not _positive(list(box)):
        return "box_not_positive"
    if median_h <= 0:
        return "no_page_median"
    if box[3] - box[1] > MAX_HEIGHT_MEDIANS * median_h:
        return "box_too_tall"
    if (box[2] - box[0]) * image_width / 1000 < MIN_WIDTH_PX:
        return "box_too_narrow"
    return None


def text_rejection(line: Dict, page_letters: str) -> Optional[str]:
    """Status and text checks of one batch line (the box is checked separately).

    :param line: Line entry of ``ai_read.lines``.
    :type line: Dict
    :param page_letters: Letters of every VLM line of the page, concatenated.
    :type page_letters: str
    :return: Rejection reason, or None when the line passes.
    :rtype: Optional[str]
    """
    if line.get("status") != "agreed":
        return "unconfirmed"
    if (line.get("agreement") or 0.0) < MIN_AGREEMENT:
        return "agreement_below_min"
    text = clean_text(line.get("text", ""))
    if GAP_TOKEN in text:
        return "gap_marker"
    if hebrew_letters(text) < MIN_LINE_LETTERS:
        return "too_few_letters"
    if has_foreign_letters(text):
        return "foreign_script"
    if page_letters.count(letters_of(text)) != 1:
        return "not_unique_on_page"
    if not line.get("htr_fragments"):
        return "no_fragments"
    return None


def usable_lines(lines: Sequence[Dict], median_h: float,
                 image_width: int) -> Tuple[List[UsableLine], Counter]:
    """Apply the line rule to every line of a page.

    :param lines: ``ai_read.lines`` of one batch record.
    :type lines: Sequence[Dict]
    :param median_h: The page's median Kraken fragment height (0-1000 units).
    :type median_h: float
    :param image_width: Oriented image width in pixels.
    :type image_width: int
    :return: (usable lines in page order, rejection reasons of agreed lines).
    :rtype: Tuple[List[UsableLine], Counter]
    """
    page_letters = "".join(letters_of(clean_text(ln.get("text", ""))) for ln in lines)
    usable: List[UsableLine] = []
    reasons: Counter = Counter()
    for ln in lines:
        why = text_rejection(ln, page_letters)
        if why is None:
            box = union_box(ln["htr_fragments"])
            why = box_rejection(box, median_h, image_width)
        if why == "unconfirmed":
            continue
        reasons["agreed_seen"] += 1
        if why:
            reasons[why] += 1
            continue
        text = clean_text(ln["text"])
        usable.append(UsableLine(int(ln["index"]), text, box, hebrew_letters(text),
                                 float(ln["agreement"])))
    reasons["usable"] += len(usable)
    return usable, reasons


def select_lines(usable: Sequence[UsableLine]) -> Tuple[List[UsableLine], List[UsableLine]]:
    """Choose the lines with the most letters for the two row families.

    :param usable: Usable lines of one page.
    :type usable: Sequence[UsableLine]
    :return: (locate lines, read_box lines); both empty when the page has
        fewer than :data:`MIN_USABLE_LINES` usable lines.
    :rtype: Tuple[List[UsableLine], List[UsableLine]]
    """
    if len(usable) < MIN_USABLE_LINES:
        return [], []
    ranked = sorted(usable, key=lambda u: (-u.letters, u.index))
    return ranked[:LOCATE_ROWS_PER_PAGE], ranked[:READBOX_ROWS_PER_PAGE]


# ---------------------------------------------------------------------------
# Pages and rows
# ---------------------------------------------------------------------------

@dataclass
class PagePlan:
    """Rows planned for one page image (before its download).

    :ivar doc_id: Canonical document id.
    :ivar image_index: Image position in the doc.
    :ivar image_url: Public GCS URL.
    :ivar image_sha256: Hash of the bytes the readers saw.
    :ivar width: Oriented image width.
    :ivar height: Oriented image height.
    :ivar locate: Lines for ``locate`` rows.
    :ivar read_box: Lines for ``read_box`` rows.
    """

    doc_id: str
    image_index: int
    image_url: str
    image_sha256: str
    width: int
    height: int
    locate: List[UsableLine]
    read_box: List[UsableLine]

    @property
    def key(self) -> str:
        """Filesystem-safe page key (image file stem).

        :return: ``<doc_id>__<image_index>``.
        :rtype: str
        """
        return safe_stem(self.doc_id, self.image_index)


def page_fragment_boxes(record: Dict, raw_dir: Path) -> Tuple[List[Sequence[float]], str]:
    """All Kraken fragment boxes of a page, for its median fragment height.

    :param record: Batch record.
    :type record: Dict
    :param raw_dir: The pipeline's raw cache directory.
    :type raw_dir: Path
    :return: (boxes, source) — the raw cache's full fragment list when present,
        else the fragments assigned to the record's lines.
    :rtype: Tuple[List[Sequence[float]], str]
    """
    vlm_model = record["ai_read"]["vlm_model"]
    path = raw_dir / f"{safe_stem(record['doc_id'], record['image_index'])}__{vlm_model}.json"
    if path.exists():
        return [f["box"] for f in json.loads(path.read_text())["frags"]], "raw_cache"
    return [b for ln in record["ai_read"]["lines"] for b in ln.get("htr_fragments") or []], "record"


def plan_page(record: Dict, frag_boxes: Sequence[Sequence[float]]) -> Tuple[Optional[PagePlan], Counter]:
    """Apply the line rule and the per-page caps to one batch record.

    :param record: Batch record.
    :type record: Dict
    :param frag_boxes: All Kraken fragment boxes of the page.
    :type frag_boxes: Sequence[Sequence[float]]
    :return: (plan or None when the page has too few usable lines, line reasons).
    :rtype: Tuple[Optional[PagePlan], Counter]
    """
    usable, reasons = usable_lines(record["ai_read"]["lines"], median_height(frag_boxes),
                                   int(record["image_width"]))
    locate, read_box = select_lines(usable)
    if not locate:
        return None, reasons
    return PagePlan(record["doc_id"], int(record["image_index"]), record["image_url"],
                    record.get("image_sha256") or "", int(record["image_width"]),
                    int(record["image_height"]), locate, read_box), reasons


def feature_row(image_path: Path, question: str, answer: str, task: str, stem: str,
                width: int, height: int) -> Dict:
    """Assemble one row in the KTIV feature schema (mirrors ``build_ktiv_dataset._row``).

    :param image_path: Page image file.
    :type image_path: Path
    :param question: Prompt.
    :type question: str
    :param answer: Target.
    :type answer: str
    :param task: ``locate`` or ``read_box``.
    :type task: str
    :param stem: Row id.
    :type stem: str
    :param width: Image width.
    :type width: int
    :param height: Image height.
    :type height: int
    :return: Feature dict.
    :rtype: Dict
    """
    answer = unicodedata.normalize("NFC", answer).strip()
    return {
        "image": str(image_path), "question": question, "answer": answer,
        "task": task, "section": SECTION, "stem": stem, "label_source": LABEL_SOURCE,
        "target_chars": len(answer), "target_tokens": 0,
        "image_width": width, "image_height": height,
    }


def locate_question(phrase: str) -> str:
    """KTIV ``locate`` prompt for a phrase.

    :param phrase: The line text.
    :type phrase: str
    :return: Prompt.
    :rtype: str
    """
    return _LOCATE_PROMPT.format(phrase=phrase)


def locate_answer(box: Sequence[int]) -> str:
    """KTIV ``locate`` answer: ``{"bbox_2d": [x0, y0, x1, y1]}``.

    :param box: 0-1000 int box.
    :type box: Sequence[int]
    :return: JSON string.
    :rtype: str
    """
    return json.dumps({"bbox_2d": [int(v) for v in box]})


def read_box_question(box: Sequence[int]) -> str:
    """KTIV ``read_box`` prompt for a box.

    :param box: 0-1000 int box.
    :type box: Sequence[int]
    :return: Prompt.
    :rtype: str
    """
    return _READBOX_PROMPT.format(x0=box[0], y0=box[1], x1=box[2], y1=box[3])


def page_rows(plan: PagePlan, image_path: Path) -> List[Tuple[Dict, Dict]]:
    """Feature rows plus their manifest entries for one downloaded page.

    :param plan: The page's plan.
    :type plan: PagePlan
    :param image_path: The stored page image.
    :type image_path: Path
    :return: ``(feature row, manifest entry)`` pairs, locate rows first.
    :rtype: List[Tuple[Dict, Dict]]
    """
    out = []
    families = (("locate", "loc", plan.locate), ("read_box", "rb", plan.read_box))
    for task, tag, lines in families:
        for j, ln in enumerate(lines):
            stem = f"dg_{plan.key}_{tag}{j}"
            if task == "locate":
                q, a = locate_question(ln.text), locate_answer(ln.box)
            else:
                q, a = read_box_question(ln.box), ln.text
            meta = {"doc_id": plan.doc_id, "image_index": plan.image_index,
                    "image_url": plan.image_url, "task": task, "stem": stem,
                    "box": list(ln.box), "text": ln.text, "agreement": ln.agreement,
                    "line_index": ln.index, "url_prefix": url_prefix(plan.image_url)}
            out.append((feature_row(image_path, q, a, task, stem, plan.width, plan.height), meta))
    return out


def url_prefix(url: str) -> str:
    """Top-level bucket folder of an image URL (``images`` photo vs ``KTIV`` scan).

    :param url: GCS image URL.
    :type url: str
    :return: Folder name, or ``other``.
    :rtype: str
    """
    m = _URL_PREFIX_RE.match(url or "")
    return m.group(1) if m else "other"


# ---------------------------------------------------------------------------
# Decontamination
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkIndex:
    """Everything a batch document is checked against.

    :ivar ids: PGP-131 canonical ids.
    :ivar image_urls: PGP-131 image URLs.
    :ivar loose_keys: Loose shelfmark keys of the PGP-131 docs.
    :ivar religious_sys: Religious-benchmark KTIV sys_nums.
    :ivar religious_keys: Loose shelfmark keys of the religious docs.
    """

    ids: Set[str]
    image_urls: Set[str] = field(default_factory=set)
    loose_keys: Set[str] = field(default_factory=set)
    religious_sys: Set[str] = field(default_factory=set)
    religious_keys: Set[str] = field(default_factory=set)

    @property
    def bases(self) -> Set[str]:
        """Benchmark ids with a trailing ``_<n>`` stripped (only those that had one).

        :return: Set of bases.
        :rtype: Set[str]
        """
        return {join_base(i) for i in self.ids} - self.ids


@dataclass
class DocRefs:
    """Cross-source references of one document from the merged corpus.

    :ivar sys_nums: KTIV system numbers attached to the record.
    :ivar shelfmarks: Display / PGP shelfmark strings.
    """

    sys_nums: Set[str] = field(default_factory=set)
    shelfmarks: Set[str] = field(default_factory=set)


def join_base(doc_id: str) -> str:
    """Strip one trailing ``_<n>`` (join / part suffix) from a canonical id.

    :param doc_id: Canonical id.
    :type doc_id: str
    :return: The id without its trailing number segment.
    :rtype: str
    """
    return _JOIN_SUFFIX_RE.sub("", doc_id)


def loose_key(shelfmark: str) -> Optional[str]:
    """:meth:`DecontamGate.loose_key`, or None when the gate did not import.

    :param shelfmark: Canonical id or shelfmark string.
    :type shelfmark: str
    :return: Loose key or None.
    :rtype: Optional[str]
    """
    return DecontamGate.loose_key(shelfmark) if DecontamGate is not None else None


def build_benchmark_index(ids: Iterable[str], image_urls: Iterable[str] = (),
                          religious_docs: Iterable[Dict] = (),
                          extra_keys: Iterable[str] = ()) -> BenchmarkIndex:
    """Assemble the decontamination index from benchmark contents.

    :param ids: PGP-131 canonical ids.
    :type ids: Iterable[str]
    :param image_urls: PGP-131 image URLs.
    :type image_urls: Iterable[str]
    :param religious_docs: Religious-benchmark docs (``sys_num``, ``shelf_mark``).
    :type religious_docs: Iterable[Dict]
    :param extra_keys: Precomputed benchmark loose keys (the gate's inventory).
    :type extra_keys: Iterable[str]
    :return: The index.
    :rtype: BenchmarkIndex
    """
    ids = set(ids)
    religious_docs = list(religious_docs)
    marks = [d["shelf_mark"] for d in religious_docs if d.get("shelf_mark")]
    rel_keys = {loose_key(m) for m in marks}
    if canonical_id is not None:
        rel_keys |= {loose_key(canonical_id(m, m)) for m in marks}
    return BenchmarkIndex(
        ids=ids, image_urls=set(image_urls),
        loose_keys=({loose_key(i) for i in ids} | set(extra_keys)) - {None},
        religious_sys={str(d["sys_num"]) for d in religious_docs if d.get("sys_num")},
        religious_keys=rel_keys - {None})


def load_benchmark_index(bench_ids_path: Path = BENCH_IDS_PATH, bench_path: Path = BENCH_PATH,
                         religious_path: Path = RELIGIOUS_BENCH_PATH) -> BenchmarkIndex:
    """Load the benchmark files that exist into a :class:`BenchmarkIndex`.

    :param bench_ids_path: ``decontam/benchmark_ids.json``.
    :type bench_ids_path: Path
    :param bench_path: Verified PGP-131 benchmark (ids + image URLs).
    :type bench_path: Path
    :param religious_path: Religious benchmark (KTIV sys_nums + shelfmarks).
    :type religious_path: Path
    :return: The index.
    :rtype: BenchmarkIndex
    """
    ids: Set[str] = set(json.loads(bench_ids_path.read_text())) if bench_ids_path.exists() else set()
    urls: Set[str] = set()
    if bench_path.exists():
        docs = json.loads(bench_path.read_text())["docs"]
        ids |= {d["doc_id"] for d in docs if d.get("doc_id")}
        urls = {d["image_url"] for d in docs if d.get("image_url")}
    religious = json.loads(religious_path.read_text())["docs"] if religious_path.exists() else []
    extra = DecontamGate().bench_keys if DecontamGate is not None else set()
    return build_benchmark_index(ids, urls, religious, extra)


def _as_list(value: object) -> List[Dict]:
    """Normalize a merged-record source block (None / dict / list) to a list of dicts.

    :param value: Block value.
    :type value: object
    :return: List of dict blocks.
    :rtype: List[Dict]
    """
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    return []


def doc_refs_from_record(rec: Dict) -> DocRefs:
    """KTIV sys_nums and shelfmarks of one merged record.

    :param rec: Line of ``merged_shelfmarks.jsonl``.
    :type rec: Dict
    :return: The references.
    :rtype: DocRefs
    """
    refs = DocRefs()
    blocks = _as_list((rec.get("images") or {}).get("ktiv")) + _as_list((rec.get("sources") or {}).get("ktiv"))
    refs.sys_nums = {str(b["sys_num"]) for b in blocks if b.get("sys_num")}
    pgp = (rec.get("sources") or {}).get("pgp") or {}
    marks = [rec.get("shelfmark_display"), ((pgp.get("fragment") or {}) if isinstance(pgp, dict) else {}).get("shelfmark")]
    refs.shelfmarks = {m for m in marks if m}
    return refs


def load_doc_refs(merged_path: Path, doc_ids: Set[str]) -> Dict[str, DocRefs]:
    """Scan the merged corpus for the wanted docs' cross-source references.

    :param merged_path: ``merged_shelfmarks.jsonl`` (skipped when absent).
    :type merged_path: Path
    :param doc_ids: Canonical ids to look up.
    :type doc_ids: Set[str]
    :return: ``doc_id -> DocRefs`` for the docs found.
    :rtype: Dict[str, DocRefs]
    """
    out: Dict[str, DocRefs] = {}
    if not merged_path.exists():
        return out
    with merged_path.open(encoding="utf-8") as fh:
        for line in fh:
            m = _MERGED_ID_RE.match(line)
            cid = json.loads(f'"{m.group(1)}"') if m else None
            if cid is None:
                cid = json.loads(line).get("canonical_id") if line.strip() else None
            if cid in doc_ids:
                out[cid] = doc_refs_from_record(json.loads(line))
    return out


def decontam_reason(doc_id: str, image_urls: Iterable[str], refs: DocRefs,
                    index: BenchmarkIndex, volume_level: bool = False) -> Optional[str]:
    """Why a document must be excluded, or None when it is clean.

    :param doc_id: Canonical id.
    :type doc_id: str
    :param image_urls: The doc's image URLs in the batch.
    :type image_urls: Iterable[str]
    :param refs: The doc's merged-corpus references.
    :type refs: DocRefs
    :param index: Benchmark index.
    :type index: BenchmarkIndex
    :param volume_level: Also drop docs whose id shares a benchmark id's base
        after stripping BOTH trailing numbers (whole volumes, e.g. all of T-S 16).
    :type volume_level: bool
    :return: Reason string or None.
    :rtype: Optional[str]
    """
    if doc_id in index.ids:
        return "benchmark_id"
    base = join_base(doc_id)
    if base in index.ids or doc_id in index.bases:
        return "benchmark_join_suffix"
    if volume_level and base in {join_base(i) for i in index.ids}:
        return "benchmark_volume"
    urls = set(image_urls)
    if urls & index.image_urls:
        return "benchmark_image_url"
    keys = {k for k in (loose_key(x) for x in (doc_id, base, *refs.shelfmarks)) if k}
    if keys & index.loose_keys:
        return "benchmark_loose_key"
    sys_nums = set(refs.sys_nums)
    for u in urls:
        m = _KTIV_URL_SYS_RE.search(u)
        if m:
            sys_nums.add(m.group(1))
    if sys_nums & index.religious_sys:
        return "religious_sys_num"
    if keys & index.religious_keys:
        return "religious_shelfmark"
    return None


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def orient_and_save(data: bytes, dest: Path) -> Tuple[int, int]:
    """Write the image both readers saw (mirrors ``two_reader_lines.prepare_image``).

    An upright JPEG is written byte-for-byte; anything else is EXIF-transposed
    and re-encoded as JPEG quality 95.  Original resolution either way.

    :param data: Downloaded bytes.
    :type data: bytes
    :param dest: Output path.
    :type dest: Path
    :return: ``(width, height)`` of the oriented image.
    :rtype: Tuple[int, int]
    """
    with PILImage.open(io.BytesIO(data)) as im:
        orientation = (im.getexif() or {}).get(0x0112, 1)
        if orientation in (None, 1) and im.format == "JPEG":
            dest.write_bytes(data)
            return im.width, im.height
        oriented = ImageOps.exif_transpose(im)
        oriented.convert("RGB").save(dest, format="JPEG", quality=95)
        return oriented.width, oriented.height


def image_dims(path: Path) -> Optional[Tuple[int, int]]:
    """Header dimensions of an image file.

    :param path: Image file.
    :type path: Path
    :return: ``(width, height)``, or None when the file is not a readable image.
    :rtype: Optional[Tuple[int, int]]
    """
    try:
        with PILImage.open(path) as im:
            return im.size
    except (UnidentifiedImageError, OSError):
        return None


def download_bytes(url: str, timeout: float = DOWNLOAD_TIMEOUT_S,
                   tries: int = DOWNLOAD_TRIES) -> Optional[bytes]:
    """Download ``url`` with retries (same quoting as the pipeline's ``download``).

    :param url: Image URL.
    :type url: str
    :param timeout: Per-attempt timeout in seconds.
    :type timeout: float
    :param tries: Attempts.
    :type tries: int
    :return: Bytes, or None when every attempt failed.
    :rtype: Optional[bytes]
    """
    quoted = urllib.parse.quote(url, safe=":/?&=%#")
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(quoted), timeout=timeout) as resp:
                return resp.read()
        except (OSError, HTTPException, ValueError) as exc:  # network: retried, then reported
            logger.warning("download %s attempt %d failed: %s", url, attempt + 1, exc)
            if attempt < tries - 1:
                time.sleep(2 * (attempt + 1))
    return None


def fetch_page_image(plan: PagePlan, images_dir: Path) -> Tuple[Optional[Path], Optional[str]]:
    """Make sure the plan's page image is stored and matches the record.

    A present file is reused when its dimensions equal the record's; a fresh
    download must hash to the record's ``image_sha256`` (the bytes the readers
    saw) and orient to the record's dimensions.  Files are written atomically.

    :param plan: Page plan.
    :type plan: PagePlan
    :param images_dir: Image cache directory.
    :type images_dir: Path
    :return: (path, None) on success, (None, failure reason) otherwise.
    :rtype: Tuple[Optional[Path], Optional[str]]
    """
    dest = images_dir / f"{plan.key}.jpg"
    if dest.exists() and image_dims(dest) == (plan.width, plan.height):
        return dest, None
    data = download_bytes(plan.image_url)
    if data is None:
        return None, "download"
    if plan.image_sha256 and hashlib.sha256(data).hexdigest() != plan.image_sha256:
        return None, "sha256_mismatch"
    tmp = dest.with_name(dest.name + ".part")
    dims = orient_and_save(data, tmp)
    if dims != (plan.width, plan.height):
        tmp.unlink(missing_ok=True)
        return None, "dims_mismatch"
    tmp.replace(dest)
    return dest, None


def fetch_all(plans: Sequence[PagePlan], images_dir: Path,
              workers: int) -> Tuple[Dict[str, Path], List[Dict]]:
    """Fetch every planned page image in parallel.

    :param plans: Page plans.
    :type plans: Sequence[PagePlan]
    :param images_dir: Image cache directory.
    :type images_dir: Path
    :param workers: Download threads.
    :type workers: int
    :return: (page key -> image path, failure records).
    :rtype: Tuple[Dict[str, Path], List[Dict]]
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    failures: List[Dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        results = ex.map(lambda p: fetch_page_image(p, images_dir), plans)
        for n, (plan, (path, why)) in enumerate(zip(plans, results), 1):
            if path is None:
                failures.append({"doc_id": plan.doc_id, "image_index": plan.image_index,
                                 "image_url": plan.image_url, "reason": why})
            else:
                paths[plan.key] = path
            if n % 100 == 0 or n == len(plans):
                logger.info("images %d/%d (%d failed)", n, len(plans), len(failures))
    return paths, failures


# ---------------------------------------------------------------------------
# Split, batch, build
# ---------------------------------------------------------------------------

def _split_draw(doc_id: str, seed: int) -> float:
    """Uniform [0, 1) draw from a seeded hash of the doc id.

    :param doc_id: Canonical id.
    :type doc_id: str
    :param seed: Split seed.
    :type seed: int
    :return: Draw.
    :rtype: float
    """
    return int(hashlib.sha256(f"{seed}:{doc_id}".encode()).hexdigest()[:12], 16) / 16 ** 12


def val_documents(doc_ids: Iterable[str], fraction: float = VAL_FRACTION,
                  seed: int = SPLIT_SEED) -> Set[str]:
    """Documents assigned to ``val`` (stable when the batch grows).

    :param doc_ids: Documents in the dataset.
    :type doc_ids: Iterable[str]
    :param fraction: Expected val share of documents.
    :type fraction: float
    :param seed: Split seed.
    :type seed: int
    :return: Val document ids (at least one when there are two or more docs).
    :rtype: Set[str]
    """
    docs = sorted(set(doc_ids))
    val = {d for d in docs if _split_draw(d, seed) < fraction}
    if not val and len(docs) >= 2:
        val = {min(docs, key=lambda d: _split_draw(d, seed))}
    return val


def load_batch(path: Path) -> Tuple[List[Dict], Dict[str, int]]:
    """Read the batch JSONL (last record per ``(doc_id, image_index)`` wins).

    A trailing line without a newline (an append in progress) is skipped.

    :param path: Batch JSONL.
    :type path: Path
    :return: (records in file order, counts of lines / duplicates / partial tail).
    :rtype: Tuple[List[Dict], Dict[str, int]]
    """
    by_key: Dict[Tuple[str, int], Dict] = {}
    counts = Counter()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            if not line.endswith("\n"):
                counts["partial_tail_skipped"] += 1
                continue
            rec = json.loads(line)
            counts["lines"] += 1
            by_key[(rec["doc_id"], int(rec["image_index"]))] = rec
    counts["duplicates"] = counts["lines"] - len(by_key)
    return list(by_key.values()), dict(counts)


def _arrow_bytes(out_dir: Path) -> int:
    """Bytes of the saved arrow shards.

    :param out_dir: ``save_to_disk`` directory.
    :type out_dir: Path
    :return: Total size.
    :rtype: int
    """
    return sum(p.stat().st_size for split in ("train", "val")
               for p in (out_dir / split).glob("*.arrow"))


def build(batch_path: Path, out_dir: Path, images_dir: Path, raw_dir: Path = DEFAULT_RAW_DIR,
          merged_path: Path = MERGED_PATH, limit: int = 0, workers: int = 8,
          volume_level: bool = False) -> Dict:
    """Build, save and describe the dataset.

    :param batch_path: Two-reader batch JSONL (a snapshot while the pipeline runs).
    :type batch_path: Path
    :param out_dir: ``save_to_disk`` destination (+ stats.json, manifest.jsonl).
    :type out_dir: Path
    :param images_dir: Page image cache.
    :type images_dir: Path
    :param raw_dir: The pipeline's raw cache (all Kraken fragments per page).
    :type raw_dir: Path
    :param merged_path: Merged corpus JSONL for cross-source decontamination.
    :type merged_path: Path
    :param limit: Max pages with agreed lines to consider (0 = all).
    :type limit: int
    :param workers: Download threads.
    :type workers: int
    :param volume_level: See :func:`decontam_reason`.
    :type volume_level: bool
    :return: The stats written to ``stats.json``.
    :rtype: Dict
    """
    t0 = time.time()
    records, batch_counts = load_batch(batch_path)
    considered = [r for r in records if any(ln.get("status") == "agreed" for ln in r["ai_read"]["lines"])]
    if limit:
        considered = considered[:limit]
    docs = sorted({r["doc_id"] for r in considered})
    logger.info("batch: %d records, %d pages with agreed lines (%d docs)",
                len(records), len(considered), len(docs))

    index = load_benchmark_index()
    refs = load_doc_refs(merged_path, set(docs))
    urls_by_doc: Dict[str, Set[str]] = defaultdict(set)
    for r in considered:
        urls_by_doc[r["doc_id"]].add(r["image_url"])
    excluded = {}
    volume_extra = 0
    for d in docs:
        why = decontam_reason(d, urls_by_doc[d], refs.get(d, DocRefs()), index, volume_level)
        if why:
            excluded[d] = why
        elif not volume_level and decontam_reason(d, urls_by_doc[d], refs.get(d, DocRefs()), index, True):
            volume_extra += 1
    logger.info("decontam: %d docs excluded %s", len(excluded), dict(Counter(excluded.values())))

    plans: List[PagePlan] = []
    line_counts: Counter = Counter()
    median_src: Counter = Counter()
    page_counts: Counter = Counter()
    for r in considered:
        if r["doc_id"] in excluded:
            page_counts["pages_decontam_excluded"] += 1
            continue
        boxes, src = page_fragment_boxes(r, raw_dir)
        median_src[src] += 1
        plan, reasons = plan_page(r, boxes)
        line_counts.update(reasons)
        if plan is None:
            page_counts["pages_too_few_usable_lines"] += 1
            continue
        plans.append(plan)
    logger.info("planned %d pages; downloading with %d workers into %s", len(plans), workers, images_dir)

    paths, failures = fetch_all(plans, images_dir, workers)
    kept = [p for p in plans if p.key in paths]
    val_docs = val_documents(p.doc_id for p in kept)
    train, val, manifest = [], [], []
    for plan in kept:
        split = "val" if plan.doc_id in val_docs else "train"
        for row, meta in page_rows(plan, paths[plan.key]):
            (val if split == "val" else train).append(row)
            manifest.append({**meta, "split": split})

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for m in manifest:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    DatasetDict({"train": Dataset.from_list(train, features=FEATURES),
                 "val": Dataset.from_list(val, features=FEATURES)}).save_to_disk(str(out_dir))

    tasks = Counter(m["task"] for m in manifest)
    letters = Counter()
    for m in manifest:
        letters[m["task"]] += hebrew_letters(m["text"])
    unique_lines = {(m["doc_id"], m["image_index"], m["line_index"]): m["text"] for m in manifest}
    stats = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "batch": str(batch_path), "batch_counts": batch_counts,
        "rule_versions": dict(Counter(r["ai_read"].get("rule_version") for r in records)),
        "vlm_models": dict(Counter(r["ai_read"].get("vlm_model") for r in records)),
        "records_in_batch": len(records),
        "pages_considered": len(considered), "docs_considered": len(docs),
        "decontam": {
            "docs_excluded": len(excluded), "pages_excluded": page_counts["pages_decontam_excluded"],
            "by_reason": dict(Counter(excluded.values())),
            "excluded_docs": [{"doc_id": d, "reason": w} for d, w in sorted(excluded.items())],
            "volume_level_rule": volume_level,
            "volume_level_would_also_exclude_docs": volume_extra,
            "loose_key_layer": DecontamGate is not None,
            "merged_refs_found": len(refs),
            "index_sizes": {"ids": len(index.ids), "image_urls": len(index.image_urls),
                            "loose_keys": len(index.loose_keys),
                            "religious_sys": len(index.religious_sys),
                            "religious_keys": len(index.religious_keys)},
        },
        "lines": dict(line_counts),
        "median_height_source": dict(median_src),
        "pages_too_few_usable_lines": page_counts["pages_too_few_usable_lines"],
        "pages_planned": len(plans),
        "download_failures": len(failures),
        "download_failures_by_reason": dict(Counter(f["reason"] for f in failures)),
        "download_failure_list": failures,
        "pages_kept": len(kept),
        "pages_kept_by_url_prefix": dict(Counter(url_prefix(p.image_url) for p in kept)),
        "docs_kept": len({p.doc_id for p in kept}),
        "docs_val": len(val_docs),
        "rows": {**dict(tasks), "total": len(manifest)},
        "rows_train": len(train), "rows_val": len(val),
        "rows_by_split_task": dict(Counter(f"{m['split']}/{m['task']}" for m in manifest)),
        "letters": {**dict(letters), "unique_lines": sum(hebrew_letters(t) for t in unique_lines.values()),
                    "n_unique_lines": len(unique_lines)},
        "image_bytes": sum(paths[p.key].stat().st_size for p in kept),
        "arrow_bytes": _arrow_bytes(out_dir),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    logger.info("saved %d rows (%d train / %d val) to %s", len(manifest), len(train), len(val), out_dir)
    return stats


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=Path, default=DEFAULT_BATCH,
                    help="two-reader batch JSONL (pass a snapshot copy while the pipeline appends)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--images-dir", type=Path, default=None, help="page image cache (default <out-dir>/images)")
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--merged", type=Path, default=MERGED_PATH)
    ap.add_argument("--limit", type=int, default=0, help="max pages with agreed lines (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--volume-level-decontam", action="store_true",
                    help="also drop docs sharing a benchmark id's volume (both trailing numbers stripped)")
    a = ap.parse_args()
    stats = build(a.batch, a.out_dir, a.images_dir or a.out_dir / "images", a.raw_dir, a.merged,
                  a.limit, a.workers, a.volume_level_decontam)
    print(json.dumps({k: v for k, v in stats.items()
                      if k not in ("download_failure_list", "decontam")}, indent=1, ensure_ascii=False))
    print("decontam:", json.dumps({k: v for k, v in stats["decontam"].items() if k != "excluded_docs"}))


if __name__ == "__main__":
    main()
