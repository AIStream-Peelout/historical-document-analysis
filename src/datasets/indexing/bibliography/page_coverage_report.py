#!/usr/bin/env python3
"""Report which printed pages of each scanned book are missing from the corpus.

A book reaches the search index through a chain of stages, and it can drop out
at any one of them::

    printed page -> page image -> OCR text -> structured JSON -> ES page doc
                                                              -> KG relations

Conflating those stages is what makes "is this book complete?" hard to answer,
so this report keeps them separate. The distinction is operationally important:

* **scan gap** — no image for a printed page. Only a trip back to the library
  fixes this.
* **OCR gap** — image exists, no OCR text. Re-run Pass 1 OCR.
* **structuring gap** — OCR text exists, no ``page_*_structured.json``. Re-run
  the Pass-1 structuring step; *do not re-photograph*.
* **index gap** — structured JSON exists but no Elasticsearch page document.
  Re-run :mod:`src.datasets.indexing.bibliography.index_all_bibliography`.
* **KG gap** — indexed but no Neo4j relation cites the page. Often legitimate
  (a page with no entities), so it is reported but never called "missing".

Printed-page coverage is derived from the *leading page number* of each page's
text. Most of this corpus was photographed as two-page spreads, so one image
carries a verso and a recto (``188 … PROPOSAL AND ACCEPTANCE 189``). Spread
stride is detected per book from the data rather than assumed, then used to
expand each image into the printed pages it covers.

Usage::

    python -m src.datasets.indexing.bibliography.page_coverage_report
    python -m src.datasets.indexing.bibliography.page_coverage_report \\
        --book kettubah_palestine --detail

Elasticsearch defaults to the local Docker node (``localhost:9200``, http),
because ``.env`` points ``ELASTIC_SEARCH_HOST`` at the Cloudflare-forwarded
endpoint, which is unreachable whenever the tunnel is down. Use ``--es-host``
to read a remote node instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import dotenv

from src.datasets.document_models.corpus_ids import book_key, book_uuid, page_uuid
from src.datasets.indexing.bibliography.index_all_bibliography import discover_indexing_tasks

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROOT = _REPO_ROOT / "src/datasets/raw_data/cairo_genizah/academic_literature"
DEFAULT_OUT_DIR = _REPO_ROOT / "artifacts/page_coverage"
DEFAULT_ES_INDEX = "bibliography_text_only_0.7"

#: Filename sequence index, e.g. ``page_007_structured.json`` -> 7.
_SEQ_RE = re.compile(r"page_(\d+)")

#: A printed page number at the very start of a page's text. Arabic numerals
#: only — roman front-matter numbers are not comparable to the body range.
_LEADING_PAGE_RE = re.compile(r"^\W*(\d{1,4})\b")

#: Printed page numbers above this are OCR noise (a footnote or a year read as
#: the running head). No volume in this corpus runs past it.
_MAX_PLAUSIBLE_PAGE = 1200


@dataclass
class PaginationFit:
    """A robust linear fit of printed page number against sheet sequence.

    :param slope: Printed pages gained per sheet (≈2 for a two-page spread).
    :param intercept: Fitted printed page number at sequence 0.
    :param stride: Printed pages per sheet, rounded from the slope.
    :param accepted: Readable numbers that agree with the fit.
    :param rejected: Readable numbers discarded as OCR noise.
    """

    slope: float
    intercept: float
    stride: int
    accepted: Dict[int, int] = field(default_factory=dict)
    rejected: Dict[int, int] = field(default_factory=dict)

    def predict(self, seq: int) -> int:
        """Predict the printed page number of a sheet from the fit.

        :param seq: Filename sequence index.
        :returns: The predicted (verso) printed page number, at least 1.
        """
        return max(1, round(self.intercept + self.slope * seq))


@dataclass
class PageRecord:
    """Per-image stage-by-stage state for one scanned sheet.

    :param seq: Filename sequence index (``page_007`` → 7).
    :param printed_start: First printed page number on the sheet, if known.
    :param printed_end: Last printed page number on the sheet (verso+stride-1).
    :param printed_inferred: The printed span came from the pagination fit
        rather than from a page number read off this sheet.
    :param has_image: A page image file exists.
    :param ocr_chars: Characters of OCR text (0 = OCR gap).
    :param has_structured: A ``page_*_structured.json`` exists.
    :param in_es: An Elasticsearch page document exists.
    :param kg_relations: Neo4j relations citing this page.
    """

    seq: int
    printed_start: Optional[int] = None
    printed_end: Optional[int] = None
    printed_inferred: bool = False
    has_image: bool = False
    ocr_chars: int = 0
    has_structured: bool = False
    in_es: bool = False
    kg_relations: int = 0

    @property
    def stage_reached(self) -> str:
        """Name the furthest pipeline stage this sheet reached.

        :returns: One of ``kg``, ``indexed``, ``structured``, ``ocr``,
            ``image_only``, ``missing``.
        """
        if self.kg_relations:
            return "kg"
        if self.in_es:
            return "indexed"
        if self.has_structured:
            return "structured"
        if self.ocr_chars:
            return "ocr"
        if self.has_image:
            return "image_only"
        return "missing"

    @property
    def printed_pages(self) -> List[int]:
        """Printed page numbers carried by this sheet.

        :returns: The inclusive printed range, or ``[]`` if unreadable.
        """
        if self.printed_start is None:
            return []
        return list(range(self.printed_start, (self.printed_end or self.printed_start) + 1))


@dataclass
class ScanCoverage:
    """Coverage of one scanned unit (one PDF / one book directory).

    :param scan: Book directory name (the corpus's ``source_book`` stem).
    :param collection: Parent directory grouping several scans of one work.
    :param title: Title from the resolved metadata file.
    :param book_uuid: Cross-store book id.
    :param volume: Volume number parsed from the scan name, if stated.
    :param stride: Printed pages per image (2 = two-page spread).
    :param fit: The pagination fit used to place sheets on printed pages.
    :param pages: Per-image records, ordered by sequence.
    :param printed_gaps: Printed page numbers absent inside the observed range.
    :param known_page_count: The volume's true extent, when the metadata states
        it. Without it the report cannot tell whether a scan that stops at
        p. 201 finished the book or stopped a third of the way in.
    """

    scan: str
    collection: str
    title: str
    book_uuid: str
    volume: Optional[int] = None
    stride: int = 1
    fit: Optional[PaginationFit] = None
    pages: List[PageRecord] = field(default_factory=list)
    printed_gaps: List[int] = field(default_factory=list)
    known_page_count: Optional[int] = None

    @property
    def image_count(self) -> int:
        """:returns: Number of page images on disk."""
        return sum(1 for p in self.pages if p.has_image)

    @property
    def structured_count(self) -> int:
        """:returns: Number of structured JSON pages."""
        return sum(1 for p in self.pages if p.has_structured)

    @property
    def es_count(self) -> int:
        """:returns: Number of pages present in Elasticsearch."""
        return sum(1 for p in self.pages if p.in_es)

    @property
    def kg_count(self) -> int:
        """:returns: Number of pages cited by at least one Neo4j relation."""
        return sum(1 for p in self.pages if p.kg_relations)

    @property
    def ocr_gaps(self) -> List[int]:
        """:returns: Sequences with an image but no OCR text."""
        return [p.seq for p in self.pages if p.has_image and not p.ocr_chars]

    @property
    def structuring_gaps(self) -> List[int]:
        """:returns: Sequences with OCR text but no structured JSON."""
        return [p.seq for p in self.pages if p.ocr_chars and not p.has_structured]

    @property
    def index_gaps(self) -> List[int]:
        """:returns: Sequences with structured JSON but no ES document."""
        return [p.seq for p in self.pages if p.has_structured and not p.in_es]

    @property
    def printed_span(self) -> Optional[Tuple[int, int]]:
        """:returns: ``(first, last)`` readable printed page, or ``None``."""
        known = [n for p in self.pages for n in p.printed_pages]
        return (min(known), max(known)) if known else None

    @property
    def is_complete(self) -> bool:
        """:returns: True when no stage gap remains for this scan."""
        return not (
            self.printed_gaps or self.ocr_gaps or self.structuring_gaps or self.index_gaps
        )


def parse_seq(path: Path) -> Optional[int]:
    """Extract the filename sequence index from a page path.

    :param path: A ``page_NNN*`` image or JSON path.
    :returns: The integer sequence, or ``None`` if the name does not carry one.
    """
    match = _SEQ_RE.search(path.name)
    return int(match.group(1)) if match else None


def leading_page_number(text: Optional[str]) -> Optional[int]:
    """Read the printed page number at the start of a page's text.

    On a two-page spread the running head of the verso comes first, so this
    returns the *lower* of the sheet's two printed numbers.

    :param text: OCR or structured page text.
    :returns: The printed page number, or ``None`` when absent/implausible.
    """
    if not text:
        return None
    match = _LEADING_PAGE_RE.match(text)
    if not match:
        return None
    number = int(match.group(1))
    return number if 0 < number <= _MAX_PLAUSIBLE_PAGE else None


def detect_stride(printed_by_seq: Dict[int, int]) -> int:
    """Infer how many printed pages each image covers.

    Uses the modal step between consecutive readable page numbers: a two-page
    spread advances by 2 per sheet, a single-page scan by 1.

    :param printed_by_seq: Map of sequence index to leading printed number.
    :returns: 1 or 2.
    """
    ordered = [printed_by_seq[seq] for seq in sorted(printed_by_seq)]
    steps = [
        b - a
        for a, b in zip(ordered, ordered[1:])
        if 0 < b - a <= 4  # ignore jumps across missing sheets and OCR noise
    ]
    if not steps:
        return 1
    return 2 if statistics.median(steps) >= 1.5 else 1


def fit_pagination(printed_by_seq: Dict[int, int], tolerance: int = 3) -> Optional[PaginationFit]:
    """Fit printed page number as a linear function of sheet sequence.

    Printed pages advance at a fixed stride through a scan, so the readable
    numbers should lie on a line. A per-sheet number is *not* trustworthy on its
    own: OCR reads a footnote marker, a plate caption or a year as the running
    head, and one such outlier at either end stretches the apparent range by
    dozens of pages and invents an enormous "missing" span. This fits the line
    robustly (Theil–Sen: the median pairwise slope, which tolerates a large
    minority of bad points) and reports which observations agree with it.

    :param printed_by_seq: Map of sequence index to leading printed number.
    :param tolerance: Maximum residual, in printed pages, for an observation to
        count as agreeing with the fit.
    :returns: The fit, or ``None`` when fewer than three readable numbers exist.
    """
    points = sorted(printed_by_seq.items())
    if len(points) < 3:
        return None
    slopes = [
        (y2 - y1) / (x2 - x1)
        for i, (x1, y1) in enumerate(points)
        for x2, y2 in points[i + 1:]
        if x2 != x1
    ]
    if not slopes:
        return None
    slope = statistics.median(slopes)
    intercept = statistics.median(y - slope * x for x, y in points)
    accepted = {
        seq: value
        for seq, value in points
        if abs(value - (intercept + slope * seq)) <= tolerance
    }
    # Refit on the agreeing subset so the line is not dragged by the outliers.
    if len(accepted) >= 3:
        refit = sorted(accepted.items())
        slopes = [
            (y2 - y1) / (x2 - x1)
            for i, (x1, y1) in enumerate(refit)
            for x2, y2 in refit[i + 1:]
            if x2 != x1
        ]
        if slopes:
            slope = statistics.median(slopes)
            intercept = statistics.median(y - slope * x for x, y in refit)
    stride = 2 if slope >= 1.5 else 1
    return PaginationFit(
        slope=slope,
        intercept=intercept,
        stride=stride,
        accepted=accepted,
        rejected={seq: v for seq, v in points if seq not in accepted},
    )


def load_ocr_pages(book_dir: Path) -> Tuple[Dict[int, int], Dict[int, str]]:
    """Read OCR text length and head text per sequence from a book's OCR file.

    These files carry the full OCR text of every sheet and can run to tens of
    megabytes, so both outputs come from a single parse. Only the head of each
    page's text is retained — enough to read the running-head page number.

    :param book_dir: Book root directory.
    :returns: ``(chars_by_seq, head_text_by_seq)``; both empty if no OCR file.
    """
    chars: Dict[int, int] = {}
    heads: Dict[int, str] = {}
    for path in book_dir.glob("*_ocr_results.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for page in payload.get("pages") or []:
            seq = page.get("page_number")
            if not isinstance(seq, int):
                continue
            result = page.get("ocr_result")
            text = (result or {}).get("full_text") if isinstance(result, dict) else None
            chars[seq] = max(chars.get(seq, 0), len(text or ""))
            if text:
                heads.setdefault(seq, text[:200])
    return chars, heads


def _printed_number(structured: Optional[Dict[str, Any]], ocr_text: Optional[str]) -> Optional[int]:
    """Resolve a sheet's leading printed page number from its text.

    The model's ``extracted_page_number`` is deliberately *not* consulted. On a
    two-page spread it usually names the recto, which shifts the sheet's span by
    one and manufactures a phantom one-page gap at every sheet whose running
    head OCR missed. The running head at the start of the text is the verso and
    is the only reading consistent across sheets.

    :param structured: Parsed structured page JSON, if any.
    :param ocr_text: Raw OCR text for the same sheet, if any.
    :returns: The printed page number, or ``None``.
    """
    if structured:
        number = leading_page_number(structured.get("full_main_text"))
        if number is not None:
            return number
    return leading_page_number(ocr_text)


def assign_spans(
    seqs: Sequence[int],
    fit: Optional[PaginationFit],
) -> Dict[int, Tuple[int, bool]]:
    """Place every sheet on the printed page it starts, using nearby anchors.

    Sheets whose running head was read and agrees with the fit are anchors. The
    rest are interpolated from the nearest anchor at the scan's stride, which
    keeps verso/recto parity intact — a plain linear prediction would round to
    the wrong parity and split the printed range with one-page holes.

    :param seqs: All sheet sequence indexes present in the scan, ascending.
    :param fit: The scan's pagination fit, or ``None`` if it could not be made.
    :returns: Map of sequence to ``(printed_start, inferred)``.
    """
    if not fit or not fit.accepted:
        return {}
    anchors = sorted(fit.accepted.items())
    spans: Dict[int, Tuple[int, bool]] = {}
    for seq in seqs:
        if seq in fit.accepted:
            spans[seq] = (fit.accepted[seq], False)
            continue
        # Step out from the nearest anchor at the scan's stride.
        nearest_seq, nearest_page = min(anchors, key=lambda a: abs(a[0] - seq))
        spans[seq] = (max(1, nearest_page + fit.stride * (seq - nearest_seq)), True)
    return spans


def find_printed_gaps(pages: Sequence[PageRecord]) -> List[int]:
    """List printed pages lost to sheets that are absent from the scan.

    A gap is only claimed where a sheet is genuinely missing from the sequence
    (``page_007`` exists, ``page_008`` does not). Gaps are deliberately *not*
    derived by subtracting the covered page set from the observed range: running
    heads are misread often enough that such a comparison invents one-page holes
    on pages that were photographed perfectly well.

    :param pages: Per-image records for one scan.
    :returns: Sorted printed page numbers behind a missing sheet.
    """
    placed = [page for page in pages if page.printed_start is not None]
    if len(placed) < 2:
        return []
    by_seq = {page.seq: page for page in placed}
    gaps: List[int] = []
    ordered = sorted(by_seq)
    for previous, following in zip(ordered, ordered[1:]):
        if following == previous + 1:
            continue
        start = (by_seq[previous].printed_end or by_seq[previous].printed_start) + 1
        gaps.extend(range(start, by_seq[following].printed_start))
    return gaps


def build_scan_coverage(
    task: Dict[str, Any],
    root: Path,
    es_seqs: Optional[Set[int]] = None,
    kg_counts: Optional[Dict[str, int]] = None,
) -> ScanCoverage:
    """Assemble stage-by-stage coverage for one discovered book task.

    :param task: A task dict from
        :func:`~src.datasets.indexing.bibliography.index_all_bibliography.discover_indexing_tasks`.
    :param root: The academic_literature root (for the collection label).
    :param es_seqs: Sequence indexes present in Elasticsearch for this book.
    :param kg_counts: Map of page_uuid to Neo4j relation count.
    :returns: The populated coverage record.
    """
    structured_dir = Path(task["structured_dir"])
    book_dir = _book_dir(structured_dir)
    metadata = json.loads(Path(task["metadata_file"]).read_text(encoding="utf-8"))
    key = book_key(metadata, stem=book_dir.name)

    image_seqs: Set[int] = set()
    if task.get("image_dir"):
        for path in Path(task["image_dir"]).iterdir():
            seq = parse_seq(path) if path.is_file() else None
            if seq is not None:
                image_seqs.add(seq)

    structured: Dict[int, Dict[str, Any]] = {}
    for path in structured_dir.glob("page_*_structured.json"):
        seq = parse_seq(path)
        if seq is not None:
            structured[seq] = json.loads(path.read_text(encoding="utf-8"))

    ocr_chars, ocr_texts = load_ocr_pages(book_dir)

    printed_by_seq: Dict[int, int] = {}
    for seq in sorted(set(structured) | set(ocr_texts)):
        number = _printed_number(structured.get(seq), ocr_texts.get(seq))
        if number is not None:
            printed_by_seq[seq] = number
    fit = fit_pagination(printed_by_seq)
    stride = fit.stride if fit else detect_stride(printed_by_seq)

    volume = infer_volume(book_dir.name, metadata)
    coverage = ScanCoverage(
        scan=book_dir.name,
        collection=_collection_label(book_dir, root),
        volume=volume,
        title=str(metadata.get("title") or "").strip(),
        book_uuid=book_uuid(key),
        stride=stride,
        fit=fit,
        known_page_count=known_page_count(metadata, volume),
    )
    all_seqs = sorted(image_seqs | set(structured) | set(ocr_chars))
    spans = assign_spans(all_seqs, fit)
    for seq in all_seqs:
        # Trust a sheet's own number only when it agrees with the pagination
        # fit; otherwise step out from the nearest anchor, so a sheet whose
        # running head was misread still accounts for the pages it carries.
        start, inferred = spans.get(seq, (printed_by_seq.get(seq), False))
        record = PageRecord(
            seq=seq,
            printed_start=start,
            printed_end=None if start is None else start + stride - 1,
            printed_inferred=inferred,
            has_image=seq in image_seqs,
            ocr_chars=ocr_chars.get(seq, 0),
            has_structured=seq in structured,
            in_es=bool(es_seqs and seq in es_seqs),
            kg_relations=(kg_counts or {}).get(page_uuid(key, seq), 0),
        )
        coverage.pages.append(record)
    coverage.printed_gaps = find_printed_gaps(coverage.pages)
    return coverage


def known_page_count(metadata: Dict[str, Any], volume: Optional[int]) -> Optional[int]:
    """Read a volume's true page count out of the book metadata.

    Looks for ``volumes.<n>.page_count`` first (multi-volume sets state each
    volume separately), then a top-level ``page_count`` / ``publication.pages``.
    Fill these in from a library catalogue: without them the report can only say
    where the scans stop, not whether the book stops there too.

    :param metadata: Parsed book metadata.
    :param volume: The volume number this scan belongs to, if known.
    :returns: The page count, or ``None`` when the metadata does not state one.
    """
    volumes = metadata.get("volumes")
    if volume is not None and isinstance(volumes, dict):
        entry = volumes.get(str(volume)) or volumes.get(volume)
        if isinstance(entry, dict):
            count = _coerce_page_count(entry.get("page_count"))
            if count:
                return count
    count = _coerce_page_count(metadata.get("page_count"))
    if count:
        return count
    publication = metadata.get("publication")
    if isinstance(publication, dict):
        return _coerce_page_count(publication.get("pages"))
    return None


def _coerce_page_count(value: Any) -> Optional[int]:
    """Coerce a metadata page-count value to a positive int.

    :param value: Raw metadata value (int, float or string).
    :returns: The page count, or ``None`` when unusable.
    """
    try:
        count = int(float(value))
    except (TypeError, ValueError):
        return None
    return count if 0 < count <= _MAX_PLAUSIBLE_PAGE else None


def infer_volume(scan: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """Resolve which volume of a multi-volume work a scan belongs to.

    Each volume restarts its pagination at 1, so printed page numbers may only
    be pooled within a volume. Most directory names carry it (``vol_1``,
    ``vol2``, ``v2``), but not all: ``friedman_vol_intro`` is in fact volume 1
    pp. 1-107 and runs straight into ``friedman_108_201_vol_1``. A
    ``scan_volumes`` map in the book metadata states those cases explicitly
    rather than leaving them to be guessed from a misleading name.

    :param scan: Book directory name.
    :param metadata: Parsed book metadata, consulted for ``scan_volumes``.
    :returns: The volume number, or ``None`` when neither source states one.
    """
    overrides = (metadata or {}).get("scan_volumes")
    if isinstance(overrides, dict) and scan in overrides:
        try:
            return int(overrides[scan])
        except (TypeError, ValueError):
            logger.warning("Bad scan_volumes entry for %s: %r", scan, overrides[scan])
    match = re.search(r"vol(?:ume)?[_\- ]?(\d+)|_v(\d+)(?:_|$)", scan.lower())
    if not match:
        return None
    return int(match.group(1) or match.group(2))


def _book_dir(structured_dir: Path) -> Path:
    """Return the book root for a structured directory.

    :param structured_dir: A ``*_structured*`` dir or its model subdirectory.
    :returns: The book root directory.
    """
    return structured_dir.parent if "_structured" in structured_dir.name else structured_dir.parent.parent


def _collection_label(book_dir: Path, root: Path) -> str:
    """Return the collection (work) a scan belongs to.

    Several scans of one multi-volume work live side by side under a shared
    parent directory (``kettubah_palestine/friedman_vol_intro`` …), which is the
    level at which "is this book complete?" should be answered.

    :param book_dir: The book (scan) directory.
    :param root: The academic_literature root.
    :returns: The parent directory name, or the scan name if it sits at root.
    """
    parent = book_dir.parent
    return book_dir.name if parent == root else parent.name


# ---------------------------------------------------------------------------
# Store lookups
# ---------------------------------------------------------------------------


def es_sequences_by_book(index: str = DEFAULT_ES_INDEX) -> Dict[str, Set[int]]:
    """Fetch the page sequences present in Elasticsearch, keyed by book_uuid.

    :param index: Bibliography index to read.
    :returns: Map of book_uuid to the set of indexed ``page_seq`` values.
    """
    from elasticsearch import Elasticsearch

    from src.datasets.indexing.elastic_index_genizah import es_config_from_env

    # Fail fast rather than retrying into a long stall when the host is wrong.
    es = Elasticsearch(**es_config_from_env(), request_timeout=30, max_retries=1)
    by_book: Dict[str, Set[int]] = {}
    after: Optional[Sequence[Any]] = None
    while True:
        kwargs: Dict[str, Any] = {
            "index": index,
            "size": 2000,
            "query": {"exists": {"field": "book_uuid"}},
            "sort": [{"page_uuid": "asc"}],
            "source_includes": ["book_uuid", "page_seq"],
        }
        if after is not None:
            kwargs["search_after"] = after
        hits = es.search(**kwargs)["hits"]["hits"]
        if not hits:
            return by_book
        for hit in hits:
            source = hit["_source"]
            seq = source.get("page_seq")
            if isinstance(seq, int):
                by_book.setdefault(source["book_uuid"], set()).add(seq)
        after = hits[-1]["sort"]


def kg_relation_counts() -> Dict[str, int]:
    """Count Neo4j relations per cited page.

    Neo4j holds no page nodes — page provenance lives on relation properties —
    so this is a relation tally, not a page inventory.

    :returns: Map of page_uuid to the number of relations citing it.
    """
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
        notifications_min_severity="OFF",
    )
    query = """
    MATCH ()-[r]->() WHERE r.page_uuid IS NOT NULL
    RETURN r.page_uuid AS page_uuid, count(*) AS relations
    """
    try:
        with driver.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as session:
            return {row["page_uuid"]: row["relations"] for row in session.run(query)}
    finally:
        driver.close()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _format_ranges(numbers: Iterable[int]) -> str:
    """Collapse a sorted number sequence into compact ranges.

    :param numbers: Page or sequence numbers.
    :returns: A string such as ``"12, 18-24, 31"``.
    """
    values = sorted(set(numbers))
    if not values:
        return ""
    parts: List[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        parts.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    parts.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(parts)


def write_scan_csv(coverages: Sequence[ScanCoverage], path: Path) -> None:
    """Write the per-scan summary CSV.

    :param coverages: Coverage records to write.
    :param path: Destination CSV path.
    """
    columns = [
        "collection", "volume", "scan", "title", "stride", "printed_first", "printed_last",
        "images", "ocr", "structured", "in_es", "in_kg",
        "printed_gap_count", "printed_gaps",
        "ocr_gaps", "structuring_gaps", "index_gaps",
        "page_numbers_rejected", "complete",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for cov in coverages:
            span = cov.printed_span
            writer.writerow({
                "collection": cov.collection,
                "volume": "" if cov.volume is None else cov.volume,
                "scan": cov.scan,
                "title": cov.title,
                "stride": cov.stride,
                "page_numbers_rejected": len(cov.fit.rejected) if cov.fit else 0,
                "printed_first": span[0] if span else "",
                "printed_last": span[1] if span else "",
                "images": cov.image_count,
                "ocr": sum(1 for p in cov.pages if p.ocr_chars),
                "structured": cov.structured_count,
                "in_es": cov.es_count,
                "in_kg": cov.kg_count,
                "printed_gap_count": len(cov.printed_gaps),
                "printed_gaps": _format_ranges(cov.printed_gaps),
                "ocr_gaps": _format_ranges(cov.ocr_gaps),
                "structuring_gaps": _format_ranges(cov.structuring_gaps),
                "index_gaps": _format_ranges(cov.index_gaps),
                "complete": cov.is_complete,
            })


def write_page_csv(coverages: Sequence[ScanCoverage], path: Path) -> None:
    """Write the per-page detail CSV (one row per scanned sheet).

    :param coverages: Coverage records to write.
    :param path: Destination CSV path.
    """
    columns = [
        "collection", "volume", "scan", "seq", "printed_start", "printed_end", "printed_inferred",
        "has_image", "ocr_chars", "has_structured", "in_es", "kg_relations", "stage_reached",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for cov in coverages:
            for page in cov.pages:
                row = asdict(page)
                row.update(
                    collection=cov.collection,
                    volume="" if cov.volume is None else cov.volume,
                    scan=cov.scan,
                    stage_reached=page.stage_reached,
                )
                writer.writerow({column: row[column] for column in columns})


def write_summary_json(coverages: Sequence[ScanCoverage], path: Path) -> Dict[str, Any]:
    """Write the machine-readable rollup.

    :param coverages: Coverage records to write.
    :param path: Destination JSON path.
    :returns: The report dict that was written.
    """
    report = {
        "scans": len(coverages),
        "complete_scans": sum(1 for c in coverages if c.is_complete),
        "totals": {
            "images": sum(c.image_count for c in coverages),
            "structured": sum(c.structured_count for c in coverages),
            "in_es": sum(c.es_count for c in coverages),
            "in_kg": sum(c.kg_count for c in coverages),
            "printed_gaps": sum(len(c.printed_gaps) for c in coverages),
            "ocr_gaps": sum(len(c.ocr_gaps) for c in coverages),
            "structuring_gaps": sum(len(c.structuring_gaps) for c in coverages),
            "index_gaps": sum(len(c.index_gaps) for c in coverages),
        },
        "volumes": _collection_rollup(coverages),
        "scan_detail": [
            {
                **{k: v for k, v in asdict(cov).items() if k not in ("pages", "fit")},
                "printed_span": cov.printed_span,
                "stride": cov.stride,
                "page_numbers_rejected": (
                    sorted(cov.fit.rejected.items()) if cov.fit else []
                ),
                "images": cov.image_count,
                "structured": cov.structured_count,
                "in_es": cov.es_count,
                "in_kg": cov.kg_count,
                "ocr_gaps": cov.ocr_gaps,
                "structuring_gaps": cov.structuring_gaps,
                "index_gaps": cov.index_gaps,
                "complete": cov.is_complete,
            }
            for cov in coverages
        ],
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def tail_gap(members: Sequence[ScanCoverage]) -> Optional[Tuple[int, int]]:
    """Pages after the last scanned one that the volume is known to still have.

    This is the difference between "our scans stop at p. 201" and "the book ends
    at p. 201". It can only be answered when a page count is recorded in the
    book's metadata (``volumes.<n>.page_count`` or a top-level ``page_count``),
    which is why that field is worth filling in from a library catalogue.

    :param members: Scans belonging to one volume.
    :returns: The inclusive ``(first, last)`` unscanned tail, or ``None`` when
        the extent is unknown or the scans already reach the end.
    """
    extent = next((cov.known_page_count for cov in members if cov.known_page_count), None)
    if not extent:
        return None
    spans = [cov.printed_span for cov in members if cov.printed_span]
    if not spans:
        return (1, extent)
    last = max(end for _, end in spans)
    return (last + 1, extent) if extent > last else None


def volume_gaps(members: Sequence[ScanCoverage]) -> List[int]:
    """Printed pages confirmed missing: those behind a sheet absent from a scan.

    Only sheet-level evidence counts here. The space *between* two scans is
    handled separately by :func:`seam_spans`, because a directory often holds
    several independent works whose page ranges are unrelated.

    :param members: Scans belonging to one volume.
    :returns: Sorted missing printed page numbers.
    """
    gaps: Set[int] = set()
    for cov in members:
        gaps.update(cov.printed_gaps)
    return sorted(gaps)


def seam_spans(members: Sequence[ScanCoverage]) -> List[Tuple[int, int]]:
    """Printed ranges lying between one scan's end and the next scan's start.

    Whether a seam is a real gap depends on whether the scans are parts of one
    work. ``india_traders`` is one book scanned as pp. 1-75 and 350-501, so its
    seam is 274 genuinely unscanned pages. ``rylands_articles`` holds six
    separate journal articles at pp. 7-27, 75-108, 488-500 …, so its seams are
    other people's articles that were never in scope. The report cannot tell
    these apart reliably — the corpus's metadata resolution assigns several
    distinct articles the same metadata file — so seams are reported as
    unverified rather than counted as missing pages.

    :param members: Scans belonging to one volume.
    :returns: Inclusive ``(first, last)`` uncovered ranges between scans.
    """
    spans = sorted(
        (cov.printed_span for cov in members if cov.printed_span), key=lambda s: s[0]
    )
    if not spans:
        return []
    seams: List[Tuple[int, int]] = []
    reach = spans[0][1]
    for start, end in spans[1:]:
        if start > reach + 1:
            seams.append((reach + 1, start - 1))
        reach = max(reach, end)
    return seams


def _collection_rollup(coverages: Sequence[ScanCoverage]) -> List[Dict[str, Any]]:
    """Group scans by the volume of the work they belong to, worst gap first.

    Printed page numbers are pooled per *volume*, never per work: each volume of
    a multi-volume set restarts at page 1, so unioning them would hide real gaps
    behind another volume's pages.

    :param coverages: Coverage records.
    :returns: One dict per (collection, volume), ordered by gap size descending.
    """
    groups: Dict[Tuple[str, Optional[int]], List[ScanCoverage]] = {}
    for cov in coverages:
        groups.setdefault((cov.collection, cov.volume), []).append(cov)
    rows: List[Dict[str, Any]] = []
    for (collection, volume), members in groups.items():
        printed_covered: Set[int] = {
            n for cov in members for page in cov.pages for n in page.printed_pages
        }
        interior_gaps = volume_gaps(members)
        seams = seam_spans(members)
        tail = tail_gap(members)
        extent = next((cov.known_page_count for cov in members if cov.known_page_count), None)
        rows.append({
            "known_page_count": extent,
            "tail_gap": list(tail) if tail else None,
            "tail_pages": (tail[1] - tail[0] + 1) if tail else 0,
            "collection": collection,
            "volume": volume,
            "label": collection if volume is None else f"{collection} vol.{volume}",
            "title": next((cov.title for cov in members if cov.title), ""),
            "scans": [cov.scan for cov in members],
            "printed_span": (
                [min(printed_covered), max(printed_covered)] if printed_covered else None
            ),
            "printed_pages_covered": len(printed_covered),
            "printed_gaps": interior_gaps,
            "printed_gaps_ranges": _format_ranges(interior_gaps),
            # Unverified: only a real gap when the scans are parts of one work.
            "seam_spans": [list(seam) for seam in seams],
            "seam_pages": sum(end - start + 1 for start, end in seams),
            "one_work": len({cov.title for cov in members if cov.title}) <= 1,
            "structuring_gap_sheets": sum(len(cov.structuring_gaps) for cov in members),
            "ocr_gap_sheets": sum(len(cov.ocr_gaps) for cov in members),
            "index_gap_sheets": sum(len(cov.index_gaps) for cov in members),
            "complete": (
                all(cov.is_complete for cov in members)
                and not interior_gaps
                and tail is None
            ),
        })
    return sorted(
        rows,
        key=lambda r: (
            -(r["tail_pages"] + len(r["printed_gaps"]) + r["structuring_gap_sheets"]
              + r["ocr_gap_sheets"] + r["index_gap_sheets"]),
            r["label"],
        ),
    )


def print_report(coverages: Sequence[ScanCoverage], report: Dict[str, Any], detail: bool) -> None:
    """Print the human-readable report.

    :param coverages: Coverage records (used for the detail section).
    :param report: The rollup returned by :func:`write_summary_json`.
    :param detail: Whether to print per-scan gap breakdowns.
    """
    totals = report["totals"]
    print(
        f"{report['scans']} scans, {report['complete_scans']} with no gap at any stage.\n"
        f"images={totals['images']} structured={totals['structured']} "
        f"in_es={totals['in_es']} in_kg={totals['in_kg']}\n"
        f"gaps: printed_pages={totals['printed_gaps']} ocr_sheets={totals['ocr_gaps']} "
        f"structuring_sheets={totals['structuring_gaps']} index_sheets={totals['index_gaps']}\n"
    )
    header = (
        f"{'work / volume':32s} {'pp.span':>10s} {'of':>5s} {'tail':>10s} "
        f"{'pp.gap':>6s} {'seam?':>6s} {'struct':>6s} {'ocr':>4s} {'index':>5s}"
    )
    print(header)
    print("-" * len(header))
    for row in report["volumes"]:
        span = row["printed_span"]
        span_text = f"{span[0]}-{span[1]}" if span else "-"
        tail = row["tail_gap"]
        tail_text = f"{tail[0]}-{tail[1]}" if tail else ("ok" if row["known_page_count"] else "?")
        print(
            f"{row['label'][:32]:32s} {span_text:>10s} "
            f"{str(row['known_page_count'] or '?'):>5s} {tail_text:>10s} "
            f"{len(row['printed_gaps']):6d} {row['seam_pages']:6d} "
            f"{row['structuring_gap_sheets']:6d} "
            f"{row['ocr_gap_sheets']:4d} {row['index_gap_sheets']:5d}"
        )
    print(
        "\nof     = the volume's true page count, when the metadata records one.\n"
        "tail   = pages after the last scanned page that the book still has —\n"
        "         the never-photographed remainder. '?' means no page count is\n"
        "         recorded, so whether the scans finished the book is UNKNOWN.\n"
        "pp.gap = pages behind a sheet missing from a scan (confirmed).\n"
        "seam?  = pages between two scans in the same folder — a real gap only\n"
        "         when those scans are parts of one work, which the corpus's\n"
        "         metadata cannot always establish. Check before acting."
    )
    if not detail:
        return
    print("\nPer-scan detail (scans with any gap):")
    for cov in coverages:
        if cov.is_complete:
            continue
        span = cov.printed_span
        span_text = f", printed {span[0]}-{span[1]}" if span else ""
        volume_text = "" if cov.volume is None else f" vol.{cov.volume}"
        print(f"\n  {cov.collection}{volume_text}/{cov.scan}  (stride={cov.stride}{span_text})")
        print(f"    images={cov.image_count} structured={cov.structured_count} "
              f"in_es={cov.es_count} in_kg={cov.kg_count}")
        if cov.printed_gaps:
            print(f"    missing printed pages : {_format_ranges(cov.printed_gaps)}")
        if cov.ocr_gaps:
            print(f"    no OCR text (sheets)  : {_format_ranges(cov.ocr_gaps)}")
        if cov.structuring_gaps:
            print(f"    OCR but not structured: {_format_ranges(cov.structuring_gaps)}")
        if cov.index_gaps:
            print(f"    structured but not ES : {_format_ranges(cov.index_gaps)}")
        if cov.fit and cov.fit.rejected:
            rejected = ", ".join(f"sheet {s}→{v}" for s, v in sorted(cov.fit.rejected.items()))
            print(f"    page nos. rejected as OCR noise: {rejected}")


def main() -> None:
    """CLI entry point: build the page-coverage report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="academic_literature root")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--index", default=DEFAULT_ES_INDEX, help="Bibliography ES index")
    parser.add_argument(
        "--book",
        action="append",
        help="Limit to scans whose collection or scan name contains this text (repeatable)",
    )
    parser.add_argument("--detail", action="store_true", help="Print per-scan gap breakdowns")
    parser.add_argument("--no-es", action="store_true", help="Skip the Elasticsearch lookup")
    parser.add_argument("--no-kg", action="store_true", help="Skip the Neo4j lookup")
    parser.add_argument("--es-host", default="localhost", help="ES host (default: local Docker)")
    parser.add_argument("--es-port", default="9200", help="ES port")
    parser.add_argument("--es-scheme", default="http", help="ES scheme")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    dotenv.load_dotenv(_REPO_ROOT / ".env")
    # Override the .env endpoint: it names the Cloudflare tunnel, which stalls
    # on connect when the tunnel is down.
    os.environ["ELASTIC_SEARCH_HOST"] = args.es_host
    os.environ["ELASTIC_SEARCH_PORT"] = args.es_port
    os.environ["ELASTIC_SEARCH_SCHEME"] = args.es_scheme

    tasks = discover_indexing_tasks(str(args.root))
    if args.book:
        # Filter before parsing any page JSON — a single-book run should not
        # read the whole corpus.
        needles = [needle.lower() for needle in args.book]
        tasks = [
            task for task in tasks
            if any(needle in str(task["structured_dir"]).lower() for needle in needles)
        ]
    es_by_book = {} if args.no_es else es_sequences_by_book(args.index)
    kg_counts = {} if args.no_kg else kg_relation_counts()

    coverages: List[ScanCoverage] = []
    for task in tasks:
        cov = build_scan_coverage(task, args.root, es_seqs=None, kg_counts=kg_counts)
        cov_es = es_by_book.get(cov.book_uuid)
        if cov_es:
            for page in cov.pages:
                page.in_es = page.seq in cov_es
        coverages.append(cov)
    coverages.sort(key=lambda c: (c.collection, c.scan))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scan_csv = args.out_dir / "page_coverage_by_scan.csv"
    page_csv = args.out_dir / "page_coverage_by_page.csv"
    summary_json = args.out_dir / "page_coverage_summary.json"
    write_scan_csv(coverages, scan_csv)
    write_page_csv(coverages, page_csv)
    report = write_summary_json(coverages, summary_json)

    print_report(coverages, report, args.detail)
    print(f"\nWrote {scan_csv}\nWrote {page_csv}\nWrote {summary_json}")


if __name__ == "__main__":
    main()
