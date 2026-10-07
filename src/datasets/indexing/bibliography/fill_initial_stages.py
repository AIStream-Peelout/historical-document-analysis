#!/usr/bin/env python3
"""Bring academic PDFs through OCR and Pass 1, filling page-level gaps.

The existing drivers (``batch_pipeline_runner``, ``complete_pipeline_example``)
treat a book as finished once any ``*_structured.json`` exists, always run the
deprecated ``_enhanced`` step, and default to the legacy ``qwen3-vl:8b`` model.
This driver does only the initial stages, for explicitly named PDFs, and is
resumable at page granularity:

1. **OCR** — ``BookOCRService`` into ``<pdf_dir>/<stem>/``. ``--ocr-mode auto``
   uses the free embedded-text path for born-digital PDFs and Google Cloud
   Vision for scans, including scans that carry a third-party OCR layer
   (HathiTrust, Persée, manchesterhive, Acrobat Paper Capture) and PDFs whose
   embedded Hebrew is stored in visual (reversed) order. An existing
   text-layer OCR that fails the same checks is reported and Pass 1 is not run
   on it unless ``--accept-suspect-ocr`` is given.
2. **Pass 1** — ``StructuredJSONLLM`` on every text-bearing page that has no
   valid file in the book's *target* structured directory (see
   :mod:`academic_pipeline_status`): missing pages and ``[VALIDATION FAILED]``
   stubs. Pages go into that existing directory so a book keeps one complete
   variant; a new book gets the model's usual directory name.
3. **Step 2.5** — ``full_main_text`` is copied from the OCR text into every
   page written here, and back-filled into any page, in any variant, that
   lacks it (Passes 2/4 read every variant).

Passes 2–4 are not run here; use ``run_kg_overnight.py --only <book_dir_name>``.

Usage::

    PYTHONPATH=. .venv/bin/python -m src.datasets.indexing.bibliography.fill_initial_stages \\
        --pdf engagement_docs_heb/engage_betr_geniz.pdf --dry-run
    PYTHONPATH=. .venv/bin/python -m src.datasets.indexing.bibliography.fill_initial_stages \\
        --pdf thesis_bible/ej_arrant.pdf --backend gemini --concurrency 4

``--pdf`` paths are relative to the corpus root. PDFs matched by
``academic_literature/.pipeline_ignore`` are refused.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import dotenv
import fitz

from src.datasets.indexing.bibliography.academic_pipeline_status import (
    DEFAULT_ROOT,
    MIN_PAGE_CHARS,
    discover_pdfs,
    find_book_dir,
    gap_pages,
    load_ignore_patterns,
    read_pass1,
    resolve_metadata,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
logger = logging.getLogger("fill_initial_stages")

DEFAULT_LMS_MODEL = "qwen3.6-35b-a3b"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
USABLE_METADATA = ("exact", "in_book_dir", "shared", "override")
_HEBREW_WORD = re.compile(r"[א-ת]{2,}")
_FINAL_FORMS = set("ךםןףץ")
# Matched case-insensitively against the first line of a fallback's error message
# (a provider refusal appends the whole response body after the first line).
_TRANSIENT = re.compile(
    r"\b(429|500|502|503|504)\b|resource_exhausted|unavailable|deadline_exceeded|timed out|timeout|"
    r"overloaded|not reachable|connection|disconnected|nodename nor servname|name or service not known",
    re.IGNORECASE,
)


@dataclass
class BookPlan:
    """What the driver will do for one PDF.

    :param pdf: Source PDF.
    :param book_dir: Output directory (may not exist yet).
    :param ocr_mode: ``text``, ``vision`` or None when OCR is already complete.
    :param ocr_reason: Why that OCR mode was chosen.
    :param suspect_ocr: Why a complete existing OCR should not be trusted (None when fine).
    :param metadata_path: Metadata file passed to Pass 1 as book context.
    :param metadata_kind: How the metadata was resolved.
    :param variant_dir: Structured directory pages will be written into (None until OCR exists).
    :param todo_pages: Pages needing Pass 1 (None until OCR exists).
    :param strays: Misnamed page files in the target variant (file -> page they hold).
    """

    pdf: Path
    book_dir: Path
    ocr_mode: Optional[str]
    ocr_reason: str
    suspect_ocr: Optional[str]
    metadata_path: Optional[Path]
    metadata_kind: str
    variant_dir: Optional[Path] = None
    todo_pages: Optional[List[int]] = None
    strays: Optional[Dict[str, int]] = None


def hebrew_final_form_starts(text: str) -> Tuple[int, int]:
    """Count Hebrew words, and those that begin with a final-form letter.

    Final forms (ך ם ן ף ץ) only end words in logical order, so many words
    *starting* with one means the text layer stores Hebrew visually (reversed).

    :param text: Extracted page text.
    :returns: ``(hebrew_words, words_starting_with_a_final_form)``.
    """
    words = _HEBREW_WORD.findall(text)
    return len(words), sum(w[0] in _FINAL_FORMS for w in words)


def _looks_scanned(page: fitz.Page) -> bool:
    """Return True when *page* is a raster scan, whatever text layer it carries.

    Either one image covers most of the page, or most of the text is drawn
    invisibly (render mode 3) — how OCR layers are laid over scans, including
    manchesterhive pages whose image sits inside a larger download frame.

    :param page: PDF page.
    :returns: Scan verdict.
    """
    area = page.rect.width * page.rect.height
    covered = max((abs(fitz.Rect(info["bbox"])) for info in page.get_image_info()), default=0)
    if area and covered / area >= 0.85:
        return True
    trace = page.get_texttrace()
    total = sum(len(span["chars"]) for span in trace)
    invisible = sum(len(span["chars"]) for span in trace if span.get("type") == 3)
    return bool(total) and invisible * 2 > total


def choose_ocr_mode(pdf: Path, sample: int = 8) -> Tuple[str, str]:
    """Pick embedded-text extraction or Cloud Vision for *pdf*.

    Vision is chosen when most sampled pages are scans (see :func:`_looks_scanned`),
    when the text layer is thin, or when embedded Hebrew looks reversed (see
    :func:`hebrew_final_form_starts`).

    :param pdf: Source PDF.
    :param sample: Number of pages to sample, spread evenly.
    :returns: ``(mode, reason)`` with mode ``text`` or ``vision``.
    """
    with fitz.open(pdf) as doc:
        n = doc.page_count
        idx = sorted({round(i * (n - 1) / max(1, sample - 1)) for i in range(min(sample, n))})
        scanned, chars, heb_words, heb_final_start = 0, [], 0, 0
        for i in idx:
            page = doc[i]
            scanned += _looks_scanned(page)
            text = page.get_text()
            chars.append(len(text.strip()))
            words, finals = hebrew_final_form_starts(text)
            heb_words += words
            heb_final_start += finals
    median_chars = sorted(chars)[len(chars) // 2] if chars else 0
    if scanned * 2 >= len(idx):
        return "vision", f"{scanned}/{len(idx)} sampled pages are scans (full-page image or invisible OCR layer)"
    if median_chars < 200:
        return "vision", f"thin text layer (median {median_chars} chars/page)"
    if heb_words >= 30 and heb_final_start / heb_words > 0.05:
        return "vision", f"embedded Hebrew looks reversed ({heb_final_start}/{heb_words} words start with a final form)"
    return "text", f"born-digital text layer (median {median_chars} chars/page)"


def ocr_json_path(book_dir: Path) -> Path:
    """Return the conventional OCR results path for *book_dir*.

    :param book_dir: Book directory.
    :returns: ``<book_dir>/<name>_ocr_results.json``.
    """
    return book_dir / f"{book_dir.name}_ocr_results.json"


def _read_ocr_json(book_dir: Path) -> Optional[Dict[str, Any]]:
    """Load the OCR JSON, or None when it is missing or unreadable.

    :param book_dir: Book directory.
    :returns: Parsed JSON or None.
    """
    path = ocr_json_path(book_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        logger.warning(f"{path} is not valid JSON (truncated checkpoint?) — OCR will be redone")
        return None


def ocr_missing_pages(pdf: Path, book_dir: Path) -> List[int]:
    """List PDF pages without a usable OCR result (absent or recorded as an error).

    :param pdf: Source PDF.
    :param book_dir: Book directory.
    :returns: Sorted page numbers (every page when there is no readable OCR JSON).
    """
    with fitz.open(pdf) as doc:
        total = doc.page_count
    data = _read_ocr_json(book_dir)
    if data is None:
        return list(range(1, total + 1))
    good = {p["page_number"] for p in data.get("pages", []) if p.get("ocr_result") and not p.get("error")}
    return sorted(set(range(1, total + 1)) - good)


def existing_ocr_suspect(pdf: Path, book_dir: Path) -> Optional[str]:
    """Return why a complete text-layer OCR should not be trusted, else None.

    :param pdf: Source PDF.
    :param book_dir: Book directory.
    :returns: Reason string or None (Vision OCR is always trusted).
    """
    data = _read_ocr_json(book_dir)
    if data is None or data.get("processing_info", {}).get("mode") != "text_only":
        return None
    mode, reason = choose_ocr_mode(pdf)
    return f"existing OCR came from the PDF text layer, but {reason}" if mode == "vision" else None


def load_ocr_pages(book_dir: Path) -> Dict[int, str]:
    """Map page number to OCR ``full_text`` for *book_dir*.

    :param book_dir: Book directory.
    :returns: ``{page_number: full_text}`` (empty when unreadable).
    """
    data = _read_ocr_json(book_dir) or {}
    return {p["page_number"]: (p.get("ocr_result") or {}).get("full_text", "") for p in data.get("pages", [])}


def target_variant_dir(book_dir: Path, default_name: str, new_variant: bool) -> Path:
    """Choose the structured directory to write into.

    :param book_dir: Book directory.
    :param default_name: Directory name the model would use for a fresh book.
    :param new_variant: Force *default_name* even if the book has a variant already.
    :returns: Directory path.
    """
    target = read_pass1(book_dir)[1] if book_dir.exists() else None
    if target and not new_variant:
        return book_dir / target
    return book_dir / default_name


def pages_needing_pass1(book_dir: Path, ocr_pages: Dict[int, str], variant_dir: Path) -> List[int]:
    """List text-bearing pages without a valid file in *variant_dir*.

    :param book_dir: Book directory.
    :param ocr_pages: ``{page_number: full_text}``.
    :param variant_dir: The structured directory pages will be written into.
    :returns: Sorted page numbers.
    """
    variants, _ = read_pass1(book_dir)
    text_pages = {n for n, t in ocr_pages.items() if len(t.strip()) >= MIN_PAGE_CHARS}
    key = variant_dir.relative_to(book_dir).as_posix()
    return gap_pages(text_pages, variants, key if key in variants else None)


def load_metadata(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Load book metadata for the Pass-1 prompt, dropping keys that are noise there.

    ``external_ids`` and ``extraction_notes`` are bookkeeping, not context.

    :param path: Metadata JSON path or None.
    :returns: Metadata dict or None.
    """
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if k not in ("external_ids", "extraction_notes")}


def plan_book(pdf: Path, root: Path, ocr_mode: str, metadata_override: Optional[Path],
              variant_name: str, new_variant: bool) -> BookPlan:
    """Work out the steps needed for one PDF without doing any of them.

    :param pdf: Source PDF.
    :param root: Corpus root.
    :param ocr_mode: ``auto``, ``text`` or ``vision``.
    :param metadata_override: Explicit metadata path, if given.
    :param variant_name: Structured directory name for a fresh book (``{stem}`` placeholder allowed).
    :param new_variant: Force a new structured directory.
    :returns: The plan.
    """
    book_dir = find_book_dir(pdf) or (pdf.parent / pdf.stem)
    if metadata_override:
        meta_path, meta_kind = metadata_override, "override"
    else:
        meta_path, meta_kind = resolve_metadata(pdf, book_dir if book_dir.exists() else None, root)
    suspect = None
    if not ocr_missing_pages(pdf, book_dir):
        mode, reason = None, "OCR complete"
        suspect = existing_ocr_suspect(pdf, book_dir)
    elif ocr_mode == "auto":
        mode, reason = choose_ocr_mode(pdf)
    else:
        mode, reason = ocr_mode, "forced by --ocr-mode"
    plan = BookPlan(pdf=pdf, book_dir=book_dir, ocr_mode=mode, ocr_reason=reason, suspect_ocr=suspect,
                    metadata_path=meta_path, metadata_kind=meta_kind)
    if _read_ocr_json(book_dir) is not None:
        plan.variant_dir = target_variant_dir(book_dir, variant_name.format(stem=book_dir.name), new_variant)
        plan.todo_pages = pages_needing_pass1(book_dir, load_ocr_pages(book_dir), plan.variant_dir)
        variants, _ = read_pass1(book_dir)
        key = plan.variant_dir.relative_to(book_dir).as_posix()
        plan.strays = variants[key].strays if key in variants else {}
    return plan


def estimate_ocr_gb(pdf: Path, sample: int = 3, json_mb_per_page: float = 0.1) -> float:
    """Estimate disk written by OCR by rendering sampled pages as 300-DPI PNGs.

    Scans produce far larger PNGs than born-digital pages (up to ~7 MB/page in
    this corpus), so a flat per-page constant under-reserves on a full disk.

    :param pdf: Source PDF.
    :param sample: Pages to render, spread evenly.
    :param json_mb_per_page: Allowance for the OCR JSON.
    :returns: Estimated gigabytes.
    """
    with fitz.open(pdf) as doc:
        n = doc.page_count
        idx = sorted({round(i * (n - 1) / max(1, sample - 1)) for i in range(min(sample, n))})
        mat = fitz.Matrix(300 / 72, 300 / 72)
        worst_mb = max(len(doc[i].get_pixmap(matrix=mat).tobytes("png")) for i in idx) / 1e6
    return n * (worst_mb + json_mb_per_page) / 1000


def run_ocr(plan: BookPlan, credentials: Optional[str]) -> None:
    """Run (or resume) OCR for *plan* into its book directory.

    :param plan: Book plan with ``ocr_mode`` set.
    :param credentials: Google service-account JSON path (Vision only).
    """
    from src.models.ocr.book_ocr_service import BookOCRService

    service = BookOCRService(credentials_path=credentials)
    plan.book_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"OCR ({plan.ocr_mode}) {plan.pdf.name} -> {plan.book_dir}")
    if plan.ocr_mode == "text":
        service.process_pdf_text_only_to_file(str(plan.pdf), output_path=str(plan.book_dir), save_images=True)
    else:
        service.process_pdf_to_file(str(plan.pdf), output_path=str(plan.book_dir))


def _is_transient(result: Dict[str, Any]) -> bool:
    """Return True when a fallback page looks like a retryable API/transport error.

    Only the first line of the error is inspected: a provider refusal appends
    the raw response body, whose numbers would otherwise match status codes.

    :param result: Page result from ``StructuredJSONLLM``.
    :returns: Whether a retry is worthwhile.
    """
    meta = result.get("metadata", {})
    first_line = str(meta.get("error_message", "")).split("\n", 1)[0]
    return bool(meta.get("validation_failed")) and bool(_TRANSIENT.search(first_line))


async def _process_one(llm: Any, page: int, text: str, image: Path, pdf_name: str, out_dir: Path,
                       sem: asyncio.Semaphore, retries: int) -> bool:
    """Run Pass 1 on one page, retrying transient failures, and write the file.

    ``full_main_text`` is always attached (as ``add_full_text_to_structured``
    does), so even a failure stub carries the page text to ES and Passes 2/4.

    :param llm: ``StructuredJSONLLM`` instance.
    :param page: Page sequence number.
    :param text: OCR text.
    :param image: Page image path (passed through; the current pydantic-ai drops it).
    :param pdf_name: Book name recorded in page metadata.
    :param out_dir: Structured directory.
    :param sem: Concurrency limiter.
    :param retries: Extra attempts for transient failures.
    :returns: True when the page validated.
    """
    async with sem:
        for attempt in range(retries + 1):
            if image.exists():
                result = await llm.process_page(text, image, page, pdf_name)
            else:
                result = await llm.process_page_text_only(text, page, pdf_name)
            if not _is_transient(result) or attempt == retries:
                break
            wait = 15 * (3 ** attempt)
            logger.warning(f"{pdf_name} p{page}: transient failure, retry {attempt + 1}/{retries} in {wait}s")
            await asyncio.sleep(wait)
    failed = bool(result.get("metadata", {}).get("validation_failed"))
    result["full_main_text"] = text  # step 2.5
    (out_dir / f"page_{page:03d}_structured.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"{pdf_name} p{page}: {'FAILED' if failed else 'ok'}")
    return not failed


async def run_pass1(plan: BookPlan, llm: Any, concurrency: int, retries: int) -> Tuple[int, int]:
    """Run Pass 1 over ``plan.todo_pages``.

    :param plan: Book plan with ``variant_dir`` and ``todo_pages`` set.
    :param llm: ``StructuredJSONLLM`` instance.
    :param concurrency: Pages in flight at once.
    :param retries: Extra attempts per page for transient failures.
    :returns: ``(ok, failed)`` page counts.
    """
    ocr_pages = load_ocr_pages(plan.book_dir)
    plan.variant_dir.mkdir(parents=True, exist_ok=True)
    images = next(iter(sorted(plan.book_dir.glob("*_images"))), plan.book_dir / f"{plan.book_dir.name}_images")
    sem = asyncio.Semaphore(concurrency)
    results = await asyncio.gather(*(
        _process_one(llm, n, ocr_pages[n], images / f"page_{n:03d}.png", plan.book_dir.name,
                     plan.variant_dir, sem, retries)
        for n in plan.todo_pages))
    return sum(results), len(results) - sum(results)


def backfill_full_text(variant_dir: Path, ocr_pages: Dict[int, str]) -> int:
    """Set ``full_main_text`` on pages of *variant_dir* that lack it (step 2.5).

    Failure stubs are included, matching ``add_full_text_to_structured``. Only
    pages missing the field are rewritten, so untouched pages keep their mtimes;
    misnamed files are matched by ``metadata.page_number``, as step 2.5 does.

    :param variant_dir: Structured directory.
    :param ocr_pages: ``{page_number: full_text}``.
    :returns: Number of pages updated.
    """
    updated = 0
    for f in sorted(variant_dir.glob("page_*_structured.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        text = ocr_pages.get(data.get("metadata", {}).get("page_number"))
        if (data.get("full_main_text") or "").strip() or not text:
            continue
        data["full_main_text"] = text
        f.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        updated += 1
    return updated


def build_llm(backend: str, lms_model: str, gemini_model: str, metadata: Optional[Dict[str, Any]]) -> Any:
    """Construct the Pass-1 client for *backend*.

    :param backend: ``gemini`` or ``lm_studio``.
    :param lms_model: LM Studio model identifier.
    :param gemini_model: Gemini model name.
    :param metadata: Book metadata for the prompt.
    :returns: ``StructuredJSONLLM`` instance.
    """
    from src.models.llm.academic.structured_json_llm import StructuredJSONLLM

    if backend == "gemini":
        return StructuredJSONLLM(use_gemini=True, gemini_model=gemini_model, book_metadata=metadata)
    return StructuredJSONLLM(model_name=lms_model, book_metadata=metadata)


def default_variant_name(backend: str, lms_model: str, gemini_model: str) -> str:
    """Return the structured directory name a fresh book gets, with a ``{stem}`` placeholder.

    Mirrors ``StructuredJSONLLM._get_output_directory_name``.

    :param backend: ``gemini`` or ``lm_studio``.
    :param lms_model: LM Studio model identifier.
    :param gemini_model: Gemini model name.
    :returns: Name template.
    """
    if backend == "gemini":
        return "{stem}_structured_gemini_" + gemini_model.replace(":", "_").replace("-", "_")
    return "{stem}_structured_" + lms_model.replace(":", "_").replace("-", "_")


def free_gb(path: Path) -> float:
    """Free space on the volume holding *path*, in GB.

    :param path: Any existing path on the volume.
    :returns: Gigabytes free.
    """
    return shutil.disk_usage(path).free / 1e9


def _rel(path: Optional[Path], root: Path) -> Optional[str]:
    """Show *path* relative to *root* when it lies inside it.

    :param path: Any path or None.
    :param root: Corpus root.
    :returns: Display string or None.
    """
    if path is None:
        return None
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)


def process_book(pdf: Path, root: Path, args: argparse.Namespace, variant_name: str) -> int:
    """Plan and run OCR, Pass 1 and step 2.5 for one PDF.

    :param pdf: Source PDF.
    :param root: Corpus root.
    :param args: Parsed CLI arguments.
    :param variant_name: Structured directory name template for a fresh book.
    :returns: Number of problems (failed pages, OCR errors, skips); -1 when the disk floor stops the run.
    """
    plan = plan_book(pdf, root, args.ocr_mode, args.metadata, variant_name, args.new_variant)
    todo = "after OCR" if plan.todo_pages is None else len(plan.todo_pages)
    variant = _rel(plan.variant_dir, root) or variant_name.format(stem=plan.book_dir.name)
    print(f"{pdf.relative_to(root)}\n  ocr: {plan.ocr_mode or 'done'} ({plan.ocr_reason})\n"
          f"  metadata: {_rel(plan.metadata_path, root)} [{plan.metadata_kind}]\n  pass1 -> {variant}: {todo} page(s)")
    for name, n in (plan.strays or {}).items():
        print(f"  NOTE: page {n} is held by misnamed {name}; not refilled — rename it to page_{n:03d}_structured.json")
    if plan.suspect_ocr:
        print(f"  WARNING: {plan.suspect_ocr}")
    usable_meta = plan.metadata_kind in USABLE_METADATA
    if not usable_meta and not args.allow_no_metadata:
        print("  SKIP: no unambiguous metadata (create <stem>_metadata.json, pass --metadata, or --allow-no-metadata)")
        return 1
    if args.dry_run:
        return 0

    problems = 0
    if plan.ocr_mode:
        need = estimate_ocr_gb(pdf)
        if free_gb(root) - need < args.min_free_gb:
            print(f"  STOP: {free_gb(root):.1f} GB free, OCR needs ~{need:.1f} GB, floor {args.min_free_gb} GB")
            return -1
        run_ocr(plan, args.credentials)
        missing = ocr_missing_pages(pdf, plan.book_dir)
        if missing:
            print(f"  OCR: {len(missing)} page(s) missing or errored ({missing[:10]}…); re-run to retry them")
            problems += len(missing)
        plan = plan_book(pdf, root, args.ocr_mode, args.metadata, variant_name, args.new_variant)
    if args.skip_pass1 or plan.variant_dir is None:
        return problems
    if plan.suspect_ocr and not args.accept_suspect_ocr:
        print("  SKIP Pass 1: re-OCR with Vision first, or pass --accept-suspect-ocr")
        return problems + 1
    if plan.todo_pages:
        metadata = load_metadata(plan.metadata_path) if usable_meta else None
        llm = build_llm(args.backend, args.lms_model, args.gemini_model, metadata)
        t0 = time.time()
        ok, bad = asyncio.run(run_pass1(plan, llm, args.concurrency, args.retries))
        problems += bad
        print(f"  pass1: {ok} ok, {bad} failed in {time.time() - t0:.0f}s -> {_rel(plan.variant_dir, root)}")
    ocr_pages = load_ocr_pages(plan.book_dir)
    filled = sum(backfill_full_text(d, ocr_pages) for d in {f.parent for f in plan.book_dir.glob(
        "*_structured*/**/page_*_structured.json")})
    if filled:
        print(f"  step 2.5: full_main_text added to {filled} page(s)")
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    :param argv: Arguments (defaults to ``sys.argv[1:]``).
    :returns: 0 when everything succeeded, 1 when some pages or books need a re-run, 2 on the disk floor.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", action="append", required=True, help="PDF path relative to the corpus root (repeatable).")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--ocr-mode", choices=("auto", "text", "vision"), default="auto")
    parser.add_argument("--backend", choices=("gemini", "lm_studio"), default="gemini")
    parser.add_argument("--lms-model", default=DEFAULT_LMS_MODEL)
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--metadata", type=Path, help="Explicit metadata JSON (only with a single --pdf).")
    parser.add_argument("--allow-no-metadata", action="store_true",
                        help="Run Pass 1 without book context when no usable metadata resolves.")
    parser.add_argument("--accept-suspect-ocr", action="store_true",
                        help="Run Pass 1 even when the existing text-layer OCR looks like a scan or reversed Hebrew.")
    parser.add_argument("--new-variant", action="store_true", help="Write a new structured dir even if one exists.")
    parser.add_argument("--skip-pass1", action="store_true", help="Only do OCR.")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--credentials", help="Google service-account JSON (default: GOOGLE_APPLICATION_CREDENTIALS).")
    parser.add_argument("--min-free-gb", type=float, default=8.0, help="Refuse OCR that would leave less free disk.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    dotenv.load_dotenv(_PROJECT_ROOT / ".env")
    root = args.root.resolve()
    if args.metadata:
        if len(args.pdf) != 1:
            parser.error("--metadata applies to exactly one --pdf")
        args.metadata = args.metadata.resolve()
        if not args.metadata.exists():
            parser.error(f"--metadata {args.metadata} does not exist")

    allowed = {p.resolve() for p in discover_pdfs(root, load_ignore_patterns(root))}
    pdfs = []
    for rel in args.pdf:
        pdf = (root / rel).resolve()
        if pdf not in allowed:
            parser.error(f"{rel}: not a corpus PDF, or listed in .pipeline_ignore")
        pdfs.append(pdf)

    variant_name = default_variant_name(args.backend, args.lms_model, args.gemini_model)
    problems = 0
    for pdf in pdfs:
        result = process_book(pdf, root, args, variant_name)
        if result < 0:
            return 2
        problems += result
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
