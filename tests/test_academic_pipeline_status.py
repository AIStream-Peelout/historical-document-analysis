"""Tests for the academic-literature stage ledger and the initial-stage filler.

The drivers this replaces judged a book "done" if any structured file existed,
so the cases that matter are the partial ones: failed-validation pages, pages
with OCR text but no structured file, several structured variants in one book,
folders holding several metadata files, and PDFs that must never be processed.
"""

import json
from pathlib import Path

import fitz
import pytest

from src.datasets.indexing.bibliography.academic_pipeline_status import (
    RUN_TAG,
    discover_pdfs,
    find_book_dir,
    find_phantom_books,
    load_ignore_patterns,
    read_pass1,
    resolve_metadata,
    survey_pdf,
)
from src.datasets.indexing.bibliography.fill_initial_stages import (
    _is_transient,
    backfill_full_text,
    choose_ocr_mode,
    default_variant_name,
    existing_ocr_suspect,
    hebrew_final_form_starts,
    load_metadata,
    ocr_missing_pages,
    pages_needing_pass1,
    target_variant_dir,
)

LONG_TEXT = "Genizah fragment discussed at length on this page. " * 12


def _pdf(path: Path, pages: int = 3, text: bool = True) -> Path:
    """Write a small PDF, born-digital (text) or scan-like (full-page image).

    :param path: Output path.
    :param pages: Page count.
    :param text: True for a text layer, False for an image-only page.
    :returns: The path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        if text:
            page.insert_textbox(page.rect + (36, 36, -36, -36), LONG_TEXT)
        else:
            pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 60, 80), False)
            pix.clear_with(200)
            page.insert_image(page.rect, pixmap=pix)
    doc.save(path)
    return path


def _ocr(book_dir: Path, texts: dict) -> Path:
    """Write an OCR results JSON in the service's shape.

    :param book_dir: Book directory.
    :param texts: ``{page_number: full_text}``.
    :returns: The JSON path.
    """
    book_dir.mkdir(parents=True, exist_ok=True)
    out = book_dir / f"{book_dir.name}_ocr_results.json"
    out.write_text(json.dumps({
        "pdf_path": str(book_dir) + ".pdf",
        "pages": [{"page_number": n, "ocr_result": {"full_text": t}} for n, t in texts.items()],
        "processing_info": {"mode": "text_only"},
    }))
    return out


def _page(variant: Path, n: int, failed: bool = False, full_text: str = "x", name: str = "") -> None:
    """Write one structured page file.

    :param variant: Structured directory.
    :param n: Page number recorded in ``metadata.page_number``.
    :param failed: Mark as a validation failure.
    :param full_text: ``full_main_text`` value ('' to omit).
    :param name: File name override (default ``page_NNN_structured.json``).
    """
    variant.mkdir(parents=True, exist_ok=True)
    data = {"summary": "s", "metadata": {"page_number": n}}
    if failed:
        data["metadata"]["validation_failed"] = True
    if full_text:
        data["full_main_text"] = full_text
    (variant / (name or f"page_{n:03d}_structured.json")).write_text(json.dumps(data))


def _meta(path: Path, title: str = "T") -> Path:
    """Write a metadata JSON.

    :param path: Output path.
    :param title: Title value.
    :returns: The path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"title": title, "authors": ["A B"]}))
    return path


@pytest.fixture
def corpus(tmp_path):
    """Build a corpus with one done book, one partial book and one new PDF.

    :param tmp_path: pytest temp dir.
    :returns: Corpus root.
    """
    root = tmp_path / "academic_literature"
    # done: coll_a/done.pdf with OCR, full Pass 1 and a v3 sentinel
    _pdf(root / "coll_a" / "done.pdf", pages=2)
    _meta(root / "coll_a" / "done_metadata.json")
    bd = root / "coll_a" / "done"
    _ocr(bd, {1: LONG_TEXT, 2: LONG_TEXT})
    for n in (1, 2):
        _page(bd / "done_structured_gemini_x", n)
    (bd / f"book_entities_resolved_{RUN_TAG}.json").write_text("{}")
    rel = root / "relations_v3" / "done" / RUN_TAG
    rel.mkdir(parents=True)
    (rel / ".v3_complete").write_text("t")
    (rel / "book_relations.json").write_text(json.dumps({"relations": [1, 2, 3]}))
    # partial: coll_b/part.pdf — page 2 failed, page 3 missing, page 4 blank
    _pdf(root / "coll_b" / "part.pdf", pages=4)
    _meta(root / "coll_b" / "only_metadata.json")
    pb = root / "coll_b" / "part"
    _ocr(pb, {1: LONG_TEXT, 2: LONG_TEXT, 3: LONG_TEXT, 4: "  "})
    _page(pb / "part_structured_old", 1)
    _page(pb / "part_structured_old", 2, failed=True)
    # new PDF in a folder with two unrelated metadata files
    _pdf(root / "coll_c" / "fresh.pdf", pages=1)
    _meta(root / "coll_c" / "x_metadata.json")
    _meta(root / "coll_c" / "y_metadata.json")
    # template and a non-scholarly file that the ignore list excludes
    _meta(root / "example_book_metadata.json")
    _pdf(root / "coll_c" / "Mail - receipt.pdf", pages=1)
    (root / ".pipeline_ignore").write_text("# personal\ncoll_c/Mail - *.pdf\n")
    return root


def test_ignore_list_excludes_personal_files(corpus):
    """Ignored PDFs never reach the ledger or the driver."""
    pdfs = discover_pdfs(corpus, load_ignore_patterns(corpus))
    names = {p.name for p in pdfs}
    assert "Mail - receipt.pdf" not in names
    assert names == {"done.pdf", "part.pdf", "fresh.pdf"}


def test_done_book(corpus):
    """A book with full Pass 1 and a v3 sentinel is done."""
    s = survey_pdf(corpus / "coll_a" / "done.pdf", corpus)
    assert (s.metadata_kind, s.pass1_ok, s.pass1_gap_pages, s.pass4, s.relations) == ("exact", 2, [], True, 3)
    assert s.next_step == "done"


def test_new_pass1_pages_make_relations_stale(corpus):
    """Pass-1 pages written after the relations sentinel mark Passes 2-4 stale."""
    import os
    sentinel = corpus / "relations_v3" / "done" / RUN_TAG / ".v3_complete"
    os.utime(sentinel, (1_000_000, 1_000_000))
    s = survey_pdf(corpus / "coll_a" / "done.pdf", corpus)
    assert s.pass4_stale and s.next_step == "pass2-4_stale"


def test_partial_book_counts_failed_and_missing_pages_as_gaps(corpus):
    """A validation-failed page and a missing page are gaps; a blank page is not."""
    s = survey_pdf(corpus / "coll_b" / "part.pdf", corpus)
    assert s.metadata_kind == "shared"
    assert (s.pass1_ok, s.pass1_failed, s.text_pages) == (1, 1, 3)
    assert s.pass1_gap_pages == [2, 3]
    assert s.next_step == "pass1_gaps"


def test_unrelated_folder_metadata_is_ambiguous_not_borrowed(corpus):
    """Two metadata files in the folder and no stem match -> ambiguous, never a guess."""
    s = survey_pdf(corpus / "coll_c" / "fresh.pdf", corpus)
    assert (s.metadata, s.metadata_kind, s.next_step) == (None, "ambiguous", "metadata+ocr")


def test_metadata_in_book_dir_and_template_skipped(corpus, tmp_path):
    """Metadata inside the book dir resolves; the root template is never used."""
    root = corpus
    pdf = _pdf(root / "solo.pdf", pages=1)
    assert resolve_metadata(pdf, None, root) == (None, "none")
    _meta(root / "solo" / "solo_inner_metadata.json")
    path, kind = resolve_metadata(pdf, root / "solo", root)
    assert kind == "in_book_dir" and path.name == "solo_inner_metadata.json"


def test_invalid_metadata_reported(corpus):
    """A zero-byte metadata file is reported as invalid."""
    (corpus / "coll_c" / "fresh_metadata.json").write_text("")
    _, kind = resolve_metadata(corpus / "coll_c" / "fresh.pdf", None, corpus)
    assert kind == "invalid"


def test_invalid_metadata_does_not_hide_valid_book_dir_file(corpus):
    """A 0-byte folder-level file is skipped when the book dir holds a valid one."""
    (corpus / "coll_c" / "fresh_metadata.json").write_text("")
    _meta(corpus / "coll_c" / "fresh" / "fresh_metadata.json")
    path, kind = resolve_metadata(corpus / "coll_c" / "fresh.pdf", corpus / "coll_c" / "fresh", corpus)
    assert (kind, path.parent.name) == ("exact", "fresh")


def test_alias_book_dir_needs_structured_output(corpus):
    """A longer slug is accepted as the book dir only when it holds Pass-1 output."""
    pdf = _pdf(corpus / "coll_d" / "searching-for.pdf", pages=1)
    alias = corpus / "coll_d" / "searching-for-the-last-fragment"
    alias.mkdir(parents=True)
    assert find_book_dir(pdf) is None
    _page(alias / "searching-for-the-last-fragment_structured_g", 1)
    assert find_book_dir(pdf) == alias


def test_phantom_collection_level_book(corpus):
    """A stray structured dir at collection level is reported as a phantom book."""
    _page(corpus / "coll_a" / "done_intro_structured", 1)
    assert find_phantom_books(corpus, {"coll_a/done", "coll_b/part"}) == ["coll_a"]


def test_pages_needing_pass1_and_target_variant(corpus):
    """Gap pages are text pages without a valid file in the target; writes go to the target."""
    pb = corpus / "coll_b" / "part"
    ocr = {1: LONG_TEXT, 2: LONG_TEXT, 3: LONG_TEXT, 4: ""}
    assert pages_needing_pass1(pb, ocr, pb / "part_structured_old") == [2, 3]
    _page(pb / "part_structured_tiny", 9)
    assert target_variant_dir(pb, "part_structured_new", new_variant=False) == pb / "part_structured_old"
    assert target_variant_dir(pb, "part_structured_new", new_variant=True) == pb / "part_structured_new"
    assert pages_needing_pass1(pb, ocr, pb / "part_structured_new") == [1, 2, 3]
    fresh = corpus / "coll_c" / "fresh"
    assert target_variant_dir(fresh, "fresh_structured_new", new_variant=False) == fresh / "fresh_structured_new"


def test_failed_stub_variant_never_wins_and_gaps_use_the_target(corpus):
    """india_trader_426_500: an all-failed variant with as many files loses to the valid one.

    A page valid only in a non-target variant is still a gap in the target.
    """
    bd = corpus / "coll_b" / "part"
    for n in (1, 2, 3):
        _page(bd / "part_structured_zz_failed", n, failed=True)
    variants, target = read_pass1(bd)
    assert target == "part_structured_old"
    s = survey_pdf(corpus / "coll_b" / "part.pdf", corpus)
    assert s.pass1_gap_pages == [2, 3]
    assert "part_structured_zz_failed" in s.variant_hazards
    assert s.pass1_es_variant == "part_structured_zz_failed"


def test_misnamed_page_is_a_stray_not_a_gap(corpus):
    """page_0301 holding page 3 is reported, and page 3 is not refilled as a duplicate."""
    bd = corpus / "coll_b" / "part"
    _page(bd / "part_structured_old", 3, name="page_0301_structured.json")
    s = survey_pdf(corpus / "coll_b" / "part.pdf", corpus)
    assert s.pass1_strays == {"part_structured_old/page_0301_structured.json": 3}
    assert s.pass1_gap_pages == [2]
    assert s.next_step == "stray_pages"


def test_missing_ocr_pages_are_reported(corpus):
    """An OCR JSON that stops short of the PDF, or has error pages, is not complete."""
    bd = corpus / "coll_b" / "part"
    _ocr(bd, {1: LONG_TEXT, 2: LONG_TEXT})
    assert ocr_missing_pages(corpus / "coll_b" / "part.pdf", bd) == [3, 4]
    s = survey_pdf(corpus / "coll_b" / "part.pdf", corpus)
    assert s.ocr_missing == [3, 4] and s.next_step == "ocr"


def test_folder_metadata_owned_by_a_sibling_pdf_is_not_shared(corpus):
    """french/Ashtor_1_1963_metadata.json must not describe another PDF under french/."""
    _pdf(corpus / "coll_f" / "ashtor.pdf", pages=1)
    _meta(corpus / "coll_f" / "ashtor_metadata.json")
    other = _pdf(corpus / "coll_f" / "sub" / "chapira.pdf", pages=1)
    assert resolve_metadata(other, None, corpus) == (None, "none")
    _meta(corpus / "coll_g" / "series_metadata.json")
    part = _pdf(corpus / "coll_g" / "series_1_50.pdf", pages=1)
    assert resolve_metadata(part, None, corpus)[1] == "shared"


def test_ignore_file_found_from_a_sub_root(corpus):
    """Pointing --root at a sub-collection still honours the corpus ignore list."""
    sub = corpus / "coll_c"
    names = {p.name for p in discover_pdfs(sub, load_ignore_patterns(sub))}
    assert names == {"fresh.pdf"}


def test_backfill_full_text_fills_missing_including_failed_stubs(corpus):
    """Step 2.5 fills empty full_main_text (failure stubs too) and leaves filled pages alone."""
    v = corpus / "coll_e" / "b" / "b_structured_x"
    _page(v, 1, full_text="")
    _page(v, 2, full_text="kept")
    _page(v, 3, failed=True, full_text="")
    assert backfill_full_text(v, {1: "one", 2: "two", 3: "three"}) == 2
    read = lambda n: json.loads((v / f"page_{n:03d}_structured.json").read_text()).get("full_main_text")
    assert (read(1), read(2), read(3)) == ("one", "kept", "three")


def test_suspect_existing_text_layer_ocr(tmp_path):
    """Text-layer OCR of a scan (Schirmann) is flagged; of a born-digital PDF it is not."""
    scan = _pdf(tmp_path / "scan.pdf", pages=2, text=False)
    _ocr(tmp_path / "scan", {1: LONG_TEXT, 2: LONG_TEXT})
    assert "scans" in existing_ocr_suspect(scan, tmp_path / "scan")
    digital = _pdf(tmp_path / "dig.pdf", pages=2)
    _ocr(tmp_path / "dig", {1: LONG_TEXT, 2: LONG_TEXT})
    assert existing_ocr_suspect(digital, tmp_path / "dig") is None


def test_invisible_ocr_layer_counts_as_scan(tmp_path):
    """A partial-page image under an invisible text layer (manchesterhive BJRL) goes to Vision."""
    path = tmp_path / "bjrl.pdf"
    doc = fitz.open()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 60, 80), False)
    pix.clear_with(200)
    for _ in range(3):
        page = doc.new_page()
        page.insert_image(fitz.Rect(60, 60, page.rect.width - 60, page.rect.height * 0.75), pixmap=pix)
        page.insert_textbox(page.rect + (36, 36, -36, -36), LONG_TEXT, render_mode=3)
    doc.save(path)
    assert choose_ocr_mode(path)[0] == "vision"


def test_choose_ocr_mode(tmp_path):
    """Born-digital text -> embedded text; full-page raster -> Vision."""
    assert choose_ocr_mode(_pdf(tmp_path / "digital.pdf", pages=3))[0] == "text"
    assert choose_ocr_mode(_pdf(tmp_path / "scan.pdf", pages=3, text=False))[0] == "vision"


def test_reversed_hebrew_detection():
    """Visual-order Hebrew (words starting with final forms) is detected; logical order is not."""
    logical = "שלום מלך נתן אברהם בן יוסף " * 10
    visual = " ".join(w[::-1] for w in logical.split())
    words, finals = hebrew_final_form_starts(logical)
    assert words == 60 and finals == 0
    words, finals = hebrew_final_form_starts(visual)
    assert finals / words > 0.05


def test_transient_detection():
    """Rate-limit and transport errors are retried; schema failures are not."""
    fail = lambda msg: {"metadata": {"validation_failed": True, "error_message": msg}}
    assert _is_transient(fail("429 RESOURCE_EXHAUSTED"))
    assert _is_transient(fail("503 UNAVAILABLE"))
    assert _is_transient(fail("Server disconnected without sending a response."))
    assert _is_transient(fail("All connection attempts failed"))
    assert _is_transient(fail("504 DEADLINE_EXCEEDED"))
    assert not _is_transient(fail("1 validation error for StructuredPageData"))
    assert not _is_transient(fail("Content filter 'RECITATION' triggered, body:\n{\"tokens\": 1500}"))
    assert not _is_transient({"metadata": {}})


def test_variant_name_and_metadata_prompt_filter(tmp_path):
    """Directory names mirror StructuredJSONLLM; bookkeeping keys stay out of the prompt."""
    assert default_variant_name("gemini", "m", "gemini-3.5-flash").format(stem="b") == "b_structured_gemini_gemini_3.5_flash"
    assert default_variant_name("lm_studio", "qwen3.6-35b-a3b", "g").format(stem="b") == "b_structured_qwen3.6_35b_a3b"
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"title": "T", "external_ids": {"doi": "x"}, "extraction_notes": "n"}))
    assert load_metadata(p) == {"title": "T"}
