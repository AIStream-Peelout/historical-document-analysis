#!/usr/bin/env python3
"""Per-PDF stage ledger for the academic secondary-source pipeline.

Every PDF saved under ``academic_literature/`` should pass through::

    metadata -> OCR (page images + *_ocr_results.json) -> Pass 1 structured
             -> Pass 2 entities -> Pass 3 coreference -> Pass 4 relations

The batch drivers decide "done" by file existence alone (any
``*_structured.json`` counts as a finished book), so partial runs, failed
pages and never-started PDFs are invisible to them. This report checks each
stage at page granularity, offline, and names the next step for every PDF.

Pass-1 structured variants
--------------------------
A book may hold several ``*_structured*`` directories (one per model run).
The downstream readers do not agree on which copy of a page they use:
Elasticsearch indexes one directory (the one with the most page files, ties
broken by filesystem order), Pass 2 with ``overwrite`` lets the last-sorted
copy win, and Pass 4 takes the first copy ``rglob`` returns. So the ledger
measures Pass-1 gaps against one *target* variant (most valid pages, then
most files, then name) — the directory ``fill_initial_stages`` writes into —
and separately flags books whose other variants hold failed or text-less
copies that Passes 2/4 or ES may still read.

It is read-only: no LLM, OCR, Elasticsearch or Neo4j calls.

Usage::

    python -m src.datasets.indexing.bibliography.academic_pipeline_status
    python -m src.datasets.indexing.bibliography.academic_pipeline_status --todo \\
        --markdown artifacts/academic_pipeline_status/status.md \\
        --json artifacts/academic_pipeline_status/status.json

PDFs listed (one relative path or glob per line; lines starting with ``#`` are
comments) in ``academic_literature/.pipeline_ignore`` are skipped — use it for
files that are not scholarship or are duplicates. ``raw_data`` is gitignored,
so that list stays local.
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import fitz

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROOT = _PROJECT_ROOT / "src" / "datasets" / "raw_data" / "cairo_genizah" / "academic_literature"
TEMPLATE_NAME = "example_book_metadata.json"
IGNORE_FILE = ".pipeline_ignore"
# Pages whose OCR text is shorter than this are treated as blank (plates,
# separator sheets) and are not expected to have Pass-1 output.
MIN_PAGE_CHARS = 50
PIPELINE_VERSION = "v3"  # mirrors relationship_extractor.PIPELINE_VERSION
RUN_TAG = f"{PIPELINE_VERSION}_lms_qwen3_6-35b-a3b"  # mirrors run_kg_overnight.RUN_TAG


@dataclass
class VariantScan:
    """Pages found in one ``*_structured*`` directory.

    :param files: Number of ``page_*_structured.json`` files.
    :param ok: Page numbers with a valid (non-failed) file.
    :param failed: Page numbers whose file is a validation-failure stub.
    :param no_text: Page numbers whose file lacks ``full_main_text``.
    :param strays: File name -> ``metadata.page_number`` for files whose name disagrees with it.
    :param newest_mtime: Latest modification time among its page files.
    """

    files: int = 0
    ok: Set[int] = field(default_factory=set)
    failed: Set[int] = field(default_factory=set)
    no_text: Set[int] = field(default_factory=set)
    strays: Dict[str, int] = field(default_factory=dict)
    newest_mtime: float = 0.0


@dataclass
class PdfStatus:
    """Stage status of one source PDF.

    :param pdf: PDF path relative to the corpus root.
    :param pages: PDF page count.
    :param text_chars_per_page: Mean embedded-text characters over the first pages.
    :param book_dir: Output directory relative to the root, if any.
    :param metadata: Resolved metadata file (relative) or None.
    :param metadata_kind: How it was resolved: exact, in_book_dir, shared, ambiguous, invalid, none.
    :param metadata_es: Metadata file Elasticsearch's resolver picks for the book dir, when it differs.
    :param images: Number of rendered page images.
    :param ocr_pages: Pages recorded in the OCR JSON (None when no OCR JSON).
    :param ocr_mode: ``vision``, ``text_only`` or ``unreadable``.
    :param ocr_missing: PDF pages absent from the OCR JSON or recorded as errors.
    :param text_pages: OCR pages with at least ``MIN_PAGE_CHARS`` characters.
    :param pass1_variants: Structured-dir variant -> page-file count.
    :param pass1_primary: Target variant (most valid pages, then files, then name).
    :param pass1_es_variant: Variant ES would index (most files) when it differs from the target or is tied.
    :param pass1_ok: Valid pages in the target variant.
    :param pass1_failed: Failure stubs in the target variant.
    :param pass1_no_full_text: Pages in any variant missing ``full_main_text`` (Passes 2/4 read every variant).
    :param pass1_gap_pages: Text-bearing OCR pages without a valid file in the target variant.
    :param pass1_strays: ``variant/file -> metadata page`` for misnamed page files.
    :param variant_hazards: Non-target variants holding failed or text-less copies downstream may read.
    :param pass2_pages: Pages with current-version entity files.
    :param pass3: Current-version resolved-entities file exists.
    :param pass4: Current-version relations sentinel exists.
    :param pass4_stale: Pass-1 output (any variant) is newer than the relations sentinel.
    :param relations: Accepted relation count (current version).
    :param next_step: Earliest incomplete stage.
    """

    pdf: str
    pages: Optional[int] = None
    text_chars_per_page: Optional[int] = None
    book_dir: Optional[str] = None
    metadata: Optional[str] = None
    metadata_kind: str = "none"
    metadata_es: Optional[str] = None
    images: int = 0
    ocr_pages: Optional[int] = None
    ocr_mode: Optional[str] = None
    ocr_missing: List[int] = field(default_factory=list)
    text_pages: int = 0
    pass1_variants: Dict[str, int] = field(default_factory=dict)
    pass1_primary: Optional[str] = None
    pass1_es_variant: Optional[str] = None
    pass1_ok: int = 0
    pass1_failed: int = 0
    pass1_no_full_text: int = 0
    pass1_gap_pages: List[int] = field(default_factory=list)
    pass1_strays: Dict[str, int] = field(default_factory=dict)
    variant_hazards: Dict[str, str] = field(default_factory=dict)
    pass2_pages: int = 0
    pass3: bool = False
    pass4: bool = False
    pass4_stale: bool = False
    relations: Optional[int] = None
    next_step: str = ""


def find_ignore_file(root: Path) -> Optional[Path]:
    """Locate ``.pipeline_ignore`` at *root* or the nearest ancestor holding one.

    Looking upward means pointing ``--root`` at a sub-collection still honours
    the corpus-wide list.

    :param root: Corpus root or a folder inside it.
    :returns: Path of the ignore file, or None.
    """
    for d in (root, *root.parents):
        if (d / IGNORE_FILE).exists():
            return d / IGNORE_FILE
    return None


def load_ignore_patterns(root: Path) -> List[str]:
    """Read ignore globs as absolute patterns anchored at the ignore file's folder.

    :param root: Corpus root (or a folder inside it).
    :returns: Absolute POSIX glob patterns (empty when no ignore file exists).
    """
    path = find_ignore_file(root)
    if path is None:
        return []
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    base = glob.escape(path.parent.as_posix())
    return [f"{base}/{line}" for line in lines if line and not line.startswith("#")]


def discover_pdfs(root: Path, ignore: Sequence[str] = ()) -> List[Path]:
    """List source PDFs under *root*, skipping generated and ignored paths.

    :param root: Corpus root.
    :param ignore: Absolute glob patterns to skip (from :func:`load_ignore_patterns`).
    :returns: Sorted PDF paths.
    """
    out = []
    for pdf in root.rglob("*.pdf"):
        rel = pdf.relative_to(root)
        if any(part.endswith("_images") or part.startswith("relations_v") for part in rel.parts[:-1]):
            continue
        if any(fnmatch.fnmatch(pdf.as_posix(), pat) for pat in ignore):
            continue
        out.append(pdf)
    return sorted(out)


def _has_structured_output(d: Path) -> bool:
    """Return True if *d* contains any Pass-1 structured page file.

    :param d: Directory to test.
    :returns: Whether a ``page_*_structured.json`` exists beneath it.
    """
    return any(True for _ in d.glob("*_structured*/**/page_*_structured.json"))


def find_book_dir(pdf: Path) -> Optional[Path]:
    """Return the output directory for *pdf*.

    The convention is a sibling ``<stem>/``; a few older books were written
    under a longer slug that starts with the stem (``searching-for`` ->
    ``searching-for-the-last-genizah-fragment-…``), accepted only when it holds
    Pass-1 output.

    :param pdf: Source PDF.
    :returns: The book directory, or None when nothing has been written yet.
    """
    exact = pdf.parent / pdf.stem
    if exact.is_dir():
        return exact
    for d in sorted(pdf.parent.iterdir()):
        if d.is_dir() and d.name.startswith(pdf.stem + "-") and _has_structured_output(d):
            return d
    return None


def _json_ok(path: Path) -> bool:
    """Return True when *path* holds a JSON object with a title.

    :param path: Metadata file.
    :returns: Validity flag.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError):
        return False
    return isinstance(data, dict) and bool(data.get("title"))


def _owned_by_sibling(meta: Path, pdf: Path) -> bool:
    """Return True when a folder-level metadata file belongs to another PDF there.

    ``french/Ashtor_1_1963_metadata.json`` describes ``french/Ashtor_1_1963.pdf``;
    it must not become the context of an unrelated PDF elsewhere under ``french/``.

    :param meta: Folder-level metadata file.
    :param pdf: The PDF being resolved.
    :returns: Whether *meta* is stem-matched to a different PDF or book dir in its folder.
    """
    stem = meta.name[: -len("_metadata.json")]
    if stem == pdf.stem:
        return False
    return (meta.parent / f"{stem}.pdf").exists() or (meta.parent / stem).is_dir()


def resolve_metadata(pdf: Path, book_dir: Optional[Path], root: Path) -> Tuple[Optional[Path], str]:
    """Find the metadata file that describes *pdf*, stem-first.

    Order: ``<stem>_metadata.json`` beside the PDF or in the book dir; any
    ``*_metadata.json`` inside the book dir; the single ``*_metadata.json`` of
    the nearest ancestor folder below the root, unless that file is stem-matched
    to a different PDF in its folder. An unparseable file never hides a valid
    one (the 0-byte ``bjrl-article-p159_metadata.json`` case). A folder holding
    several candidate files with no stem match is ``ambiguous``.

    :param pdf: Source PDF.
    :param book_dir: Its output directory (may be None).
    :param root: Corpus root.
    :returns: ``(path or None, kind)``.
    """
    exact = [pdf.parent / f"{pdf.stem}_metadata.json"]
    if book_dir is not None:
        exact.append(book_dir / f"{book_dir.name}_metadata.json")
    invalid = None
    for cand in exact:
        if cand.exists():
            if _json_ok(cand):
                return cand, "exact"
            invalid = invalid or cand
    if book_dir is not None:
        for cand in sorted(book_dir.glob("*_metadata.json")):
            if _json_ok(cand):
                return cand, "in_book_dir"
            invalid = invalid or cand
    if invalid is not None:
        return invalid, "invalid"
    folder = pdf.parent
    while folder != root and root in folder.parents:
        shared = sorted(p for p in folder.glob("*_metadata.json")
                        if p.name != TEMPLATE_NAME and not _owned_by_sibling(p, pdf))
        if len(shared) == 1:
            return shared[0], ("shared" if _json_ok(shared[0]) else "invalid")
        if len(shared) > 1:
            return None, "ambiguous"
        folder = folder.parent
    return None, "none"


def es_metadata_for(book_dir: Path, root: Path) -> Optional[Path]:
    """Return the metadata file the Elasticsearch indexer would attach to *book_dir*.

    :param book_dir: Book directory.
    :param root: Corpus root.
    :returns: Path chosen by ``index_all_bibliography.find_book_metadata``.
    """
    from src.datasets.indexing.bibliography.index_all_bibliography import find_book_metadata

    return find_book_metadata(book_dir, root)


def _page_num(name: str) -> Optional[int]:
    """Parse the sequence index from ``page_NNN_…`` file names.

    :param name: File name.
    :returns: The integer index, or None.
    """
    parts = name.split("_")
    if len(parts) >= 2 and parts[0] == "page" and parts[1].isdigit():
        return int(parts[1])
    return None


def read_ocr(book_dir: Path) -> Tuple[Optional[int], Optional[str], Set[int], Set[int]]:
    """Summarise ``<book>/<book>_ocr_results.json``.

    :param book_dir: Book directory.
    :returns: ``(page_count, mode, text-bearing pages, good pages)``; good pages
        have an OCR result and no error. Mode is ``unreadable`` when the JSON
        cannot be parsed (e.g. caught mid-rewrite).
    """
    path = book_dir / f"{book_dir.name}_ocr_results.json"
    if not path.exists():
        return None, None, set(), set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return 0, "unreadable", set(), set()
    pages = data.get("pages", [])
    mode = data.get("processing_info", {}).get("mode") or "vision"
    text_pages = {
        p["page_number"] for p in pages
        if isinstance(p.get("ocr_result"), dict)
        and len((p["ocr_result"].get("full_text") or "").strip()) >= MIN_PAGE_CHARS
    }
    good = {p["page_number"] for p in pages if isinstance(p.get("ocr_result"), dict) and not p.get("error")}
    return len(pages), mode, text_pages, good


def scan_variants(book_dir: Path) -> Dict[str, VariantScan]:
    """Scan every ``*_structured*`` directory (one- or two-level) under *book_dir*.

    :param book_dir: Book directory.
    :returns: Variant path (relative to the book dir) -> scan.
    """
    out: Dict[str, VariantScan] = {}
    for f in book_dir.glob("*_structured*/**/page_*_structured.json"):
        if "_structured" not in f.parent.name and "_structured" not in f.parent.parent.name:
            continue
        v = out.setdefault(f.parent.relative_to(book_dir).as_posix(), VariantScan())
        v.files += 1
        v.newest_mtime = max(v.newest_mtime, f.stat().st_mtime)
        n = _page_num(f.name)
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            v.failed.add(n)
            continue
        meta_n = data.get("metadata", {}).get("page_number")
        if isinstance(meta_n, int) and meta_n != n:
            v.strays[f.name] = meta_n
            continue
        (v.failed if data.get("metadata", {}).get("validation_failed") else v.ok).add(n)
        if not (data.get("full_main_text") or "").strip():
            v.no_text.add(n)
    return out


def pick_target_variant(variants: Dict[str, VariantScan]) -> Optional[str]:
    """Choose the variant gaps are measured against and new pages are written to.

    :param variants: Output of :func:`scan_variants`.
    :returns: Variant with the most valid pages, then most files, then the smallest name.
    """
    if not variants:
        return None
    return sorted(variants, key=lambda k: (-len(variants[k].ok), -variants[k].files, k))[0]


def es_variant(variants: Dict[str, VariantScan]) -> Tuple[Optional[str], bool]:
    """Return the variant the ES indexer picks (most files) and whether that pick is a tie.

    :param variants: Output of :func:`scan_variants`.
    :returns: ``(variant, tied)``.
    """
    if not variants:
        return None, False
    top = max(v.files for v in variants.values())
    leaders = sorted(k for k, v in variants.items() if v.files == top)
    return leaders[0], len(leaders) > 1


def read_pass1(book_dir: Path) -> Tuple[Dict[str, VariantScan], Optional[str]]:
    """Scan Pass-1 output and choose the target variant.

    :param book_dir: Book directory.
    :returns: ``(variants, target)``.
    """
    variants = scan_variants(book_dir)
    return variants, pick_target_variant(variants)


def gap_pages(text_pages: Set[int], variants: Dict[str, VariantScan], target: Optional[str]) -> List[int]:
    """Text-bearing pages without a valid file in the target variant.

    Pages covered by a misnamed file in the target (``page_0301`` holding page
    30) are not gaps — writing ``page_030`` would duplicate them downstream.

    :param text_pages: OCR pages with text.
    :param variants: Output of :func:`scan_variants`.
    :param target: Target variant.
    :returns: Sorted page numbers.
    """
    if target is None:
        return sorted(text_pages)
    v = variants[target]
    return sorted(text_pages - v.ok - set(v.strays.values()))


def next_step(s: PdfStatus) -> str:
    """Name the earliest incomplete stage for *s*.

    :param s: Filled status record.
    :returns: Short step label.
    """
    prefix = "metadata+" if s.metadata_kind in ("none", "invalid", "ambiguous") else ""
    if s.ocr_pages is None or s.ocr_missing or s.ocr_mode == "unreadable":
        step = "ocr"
    elif s.pass1_ok == 0:
        step = "pass1"
    elif s.pass1_strays:
        step = "stray_pages"
    elif s.pass1_gap_pages:
        step = "pass1_gaps"
    elif s.pass1_no_full_text:
        step = "full_text"
    elif not s.pass4:
        step = "pass2-4"
    elif s.pass4_stale:
        step = "pass2-4_stale"
    else:
        step = "done"
    return prefix + step


def survey_pdf(pdf: Path, root: Path) -> PdfStatus:
    """Build the stage record for one PDF.

    :param pdf: Source PDF.
    :param root: Corpus root.
    :returns: Status record.
    """
    s = PdfStatus(pdf=pdf.relative_to(root).as_posix())
    with fitz.open(pdf) as doc:
        s.pages = doc.page_count
        sample = range(min(doc.page_count, 5))
        s.text_chars_per_page = int(sum(len(doc[i].get_text().strip()) for i in sample) / max(1, len(sample)))
    book_dir = find_book_dir(pdf)
    meta, s.metadata_kind = resolve_metadata(pdf, book_dir, root)
    s.metadata = meta.relative_to(root).as_posix() if meta else None
    if book_dir is None:
        s.next_step = next_step(s)
        return s
    s.book_dir = book_dir.relative_to(root).as_posix()
    s.images = sum(1 for _ in book_dir.glob("*_images/page_*.png"))
    s.ocr_pages, s.ocr_mode, text_pages, good = read_ocr(book_dir)
    if s.ocr_pages is not None and s.ocr_mode != "unreadable":
        s.ocr_missing = sorted(set(range(1, s.pages + 1)) - good)
    s.text_pages = len(text_pages)

    variants, target = read_pass1(book_dir)
    s.pass1_variants = {k: v.files for k, v in variants.items()}
    s.pass1_primary = target
    if target:
        tv = variants[target]
        s.pass1_ok, s.pass1_failed = len(tv.ok), len(tv.failed)
        s.pass1_strays = {f"{target}/{name}": n for name, n in tv.strays.items()}
    s.pass1_no_full_text = sum(len(v.no_text) for v in variants.values())
    s.pass1_gap_pages = gap_pages(text_pages, variants, target) if s.ocr_pages else []
    for name, v in variants.items():
        if name != target and (v.failed or v.no_text):
            s.variant_hazards[name] = f"{v.files} files, {len(v.failed)} failed, {len(v.no_text)} without full text"
    es_pick, tied = es_variant(variants)
    if es_pick and (es_pick != target or tied):
        s.pass1_es_variant = es_pick + (" (tie)" if tied else "")
    es_meta = es_metadata_for(book_dir, root) if variants else None
    if es_meta is not None and (meta is None or es_meta.resolve() != meta.resolve()):
        s.metadata_es = es_meta.relative_to(root).as_posix()

    s.pass2_pages = sum(1 for _ in book_dir.glob(f"entities_{RUN_TAG}/page_*_entities.json"))
    s.pass3 = (book_dir / f"book_entities_resolved_{RUN_TAG}.json").exists()
    rel_dir = root / f"relations_{PIPELINE_VERSION}" / book_dir.name / RUN_TAG
    sentinel = rel_dir / f".{PIPELINE_VERSION}_complete"
    s.pass4 = sentinel.exists()
    if s.pass4 and variants:
        s.pass4_stale = max(v.newest_mtime for v in variants.values()) > sentinel.stat().st_mtime
    if (rel_dir / "book_relations.json").exists():
        rel = json.loads((rel_dir / "book_relations.json").read_text(encoding="utf-8"))
        s.relations = len(rel.get("relations", rel) if isinstance(rel, dict) else rel)
    s.next_step = next_step(s)
    return s


def find_phantom_books(root: Path, book_dirs: Set[str]) -> List[str]:
    """List directories the downstream drivers treat as books but no PDF owns.

    ``entity_tagger._book_dir_for_structured`` makes the parent of any
    ``*_structured*`` directory a book, so a stray structured dir at collection
    level turns the whole collection into a "book" that sweeps in its children.

    :param root: Corpus root.
    :param book_dirs: Book dirs (relative) owned by a PDF.
    :returns: Relative paths of phantom book directories.
    """
    phantoms = set()
    for f in root.rglob("page_*_structured.json"):
        parent = f.parent
        book = parent.parent if "_structured" in parent.name else parent.parent.parent
        rel = book.relative_to(root).as_posix()
        if rel not in book_dirs and "relations_v" not in rel:
            phantoms.add(rel)
    return sorted(phantoms)


def _ranges(nums: Sequence[int]) -> str:
    """Compress sorted integers into ``1-3,7`` form.

    :param nums: Sorted page numbers.
    :returns: Range string.
    """
    out, start, prev = [], None, None
    for n in nums:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = n
    if start is not None:
        out.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(out)


def to_markdown(rows: List[PdfStatus], phantoms: List[str], ignored: List[str]) -> str:
    """Render the ledger as a Markdown table grouped by next step, plus hazard notes.

    :param rows: Status records.
    :param phantoms: Phantom book directories.
    :param ignored: Ignore patterns in effect.
    :returns: Markdown text.
    """
    lines = ["| PDF | pages | metadata | OCR | Pass 1 ok / text pages | Pass-1 gaps | P2 | P3 | P4 | next |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for s in sorted(rows, key=lambda r: (r.next_step == "done", r.next_step, r.pdf)):
        ocr = f"{s.ocr_pages} ({s.ocr_mode})" if s.ocr_pages is not None else "—"
        gaps = _ranges(s.pass1_gap_pages) if s.pass1_gap_pages else ""
        if len(gaps) > 40:
            gaps = f"{len(s.pass1_gap_pages)} pages ({gaps[:37]}…)"
        meta = f" `{Path(s.metadata).name}`" if s.metadata_kind == "shared" else ""
        p4 = ("stale" if s.pass4_stale else "✓") if s.pass4 else ""
        lines.append(f"| `{s.pdf}` | {s.pages} | {s.metadata_kind}{meta} | {ocr} | {s.pass1_ok} / {s.text_pages} | "
                     f"{gaps} | {s.pass2_pages or ''} | {'✓' if s.pass3 else ''} | {p4} | {s.next_step} |")
    notes = []
    for s in rows:
        if s.metadata_es:
            notes.append(f"- `{s.pdf}`: Elasticsearch attaches `{s.metadata_es}`, not `{s.metadata}`")
        if s.pass1_es_variant:
            notes.append(f"- `{s.pdf}`: Elasticsearch indexes `{s.pass1_es_variant}`, target is `{s.pass1_primary}`")
        for name, why in s.variant_hazards.items():
            notes.append(f"- `{s.pdf}`: other variant `{name}` ({why}) can reach Passes 2/4")
        for name, n in s.pass1_strays.items():
            notes.append(f"- `{s.pdf}`: misnamed `{name}` holds page {n}")
    if notes:
        lines += ["", "### Hazards", *notes]
    if phantoms:
        lines += ["", "Phantom book dirs (treated as books downstream, no PDF owns them): "
                  + ", ".join(f"`{p}`" for p in phantoms)]
    if ignored:
        lines += ["", f"Ignored via `{IGNORE_FILE}`: " + ", ".join(f"`{Path(p).name}`" for p in ignored)]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    :param argv: Arguments (defaults to ``sys.argv[1:]``).
    :returns: Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Corpus root (academic_literature).")
    parser.add_argument("--json", type=Path, help="Write the full ledger as JSON here.")
    parser.add_argument("--markdown", type=Path, help="Write a Markdown table here.")
    parser.add_argument("--todo", action="store_true", help="Print only PDFs that are not done.")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    ignored = load_ignore_patterns(root)
    rows = [survey_pdf(p, root) for p in discover_pdfs(root, ignored)]
    phantoms = find_phantom_books(root, {r.book_dir for r in rows if r.book_dir})

    for s in rows:
        if args.todo and s.next_step == "done":
            continue
        gaps = f" gaps={len(s.pass1_gap_pages)}" if s.pass1_gap_pages else ""
        flags = "".join((" [es-meta]" if s.metadata_es else "", " [variants]" if s.variant_hazards else "",
                         " [es-variant]" if s.pass1_es_variant else ""))
        print(f"{s.next_step:22s} {s.pdf[:64]:64s} pg={s.pages!s:>4} md={s.metadata_kind:11s} "
              f"ocr={s.ocr_pages if s.ocr_pages is not None else '-'!s:>4} p1={s.pass1_ok:>4}/{s.text_pages:<4}{gaps}{flags}")
    done = sum(s.next_step == "done" for s in rows)
    print(f"\n{len(rows)} PDFs: {done} done, {len(rows) - done} with work left; "
          f"{len(phantoms)} phantom book dir(s); {len(ignored)} ignore pattern(s)")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"rows": [asdict(r) for r in rows], "phantom_book_dirs": phantoms,
                                         "ignored": ignored}, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(to_markdown(rows, phantoms, ignored), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
