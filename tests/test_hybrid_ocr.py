"""Tests for the hybrid OCR mode (text layer vs Cloud Vision per page).

Measured on OBP Semitic Languages and Cultures volumes: the text layer of a
born-digital PDF scrambles right-to-left lines, while Vision drops short
Hebrew/Arabic words set inline in English prose. The cases that matter are
the routing of each page kind, Vision being rejected when it loses letters,
and a Vision failure being retried on the next run.
"""

import json
from pathlib import Path
from typing import Callable, List

import fitz
import pytest

from src.datasets.indexing.bibliography.fill_initial_stages import (
    existing_ocr_suspect,
    ocr_checkpoint_mode,
    ocr_missing_pages,
    plan_book,
)
from src.models.ocr.book_ocr_service import BookOCRService
from src.models.ocr.script_routing import (
    HybridThresholds,
    letter_agreement,
    semitic_share,
    suspect_glyphs,
    wants_vision,
)

HEBREW = "בשרתי צדק בקהל רב הנה אל ישועתי אבטח ולא אפחד כי עזי וזמרת יה"
ENGLISH = "The letter from the community of Alexandria is discussed at length on this page. " * 3
MIXED = ENGLISH + " The word <span dir='rtl'>שלום</span> appears once."


def _pdf(path: Path, pages: List[str]) -> Path:
    """Write a PDF whose pages carry the given HTML snippets (Hebrew via MuPDF's fallback fonts).

    :param path: Output PDF path.
    :param pages: One HTML snippet per page.
    :returns: *path*.
    """
    doc = fitz.open()
    for html in pages:
        page = doc.new_page()
        page.insert_htmlbox(page.rect + (50, 50, -50, -50), html)
    doc.save(path)
    return path


def _service(vision: Callable[[bytes], dict]) -> BookOCRService:
    """Build a BookOCRService with a stand-in for the Vision call (no client, no credentials).

    :param vision: Replacement for ``extract_text_from_image``.
    :returns: Service instance.
    """
    service = BookOCRService.__new__(BookOCRService)
    service.extract_text_from_image = vision
    return service


class _FakeVision:
    """Returns queued texts (or raises queued exceptions) and counts calls."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, image_bytes: bytes) -> dict:
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return {"full_text": item, "text_blocks": [], "confidence_score": 0.99}


def test_semitic_share_separates_page_kinds():
    """Hebrew paragraphs route to Vision; English prose with an inline Hebrew word does not."""
    th = HybridThresholds()
    assert semitic_share(HEBREW) == pytest.approx(1.0)
    assert semitic_share(ENGLISH) == 0.0
    assert 0 < semitic_share(ENGLISH + " שלום") < th.min_share
    assert wants_vision(HEBREW, th) and not wants_vision(ENGLISH + " שלום", th)
    assert semitic_share("") == 0.0


def test_letter_agreement_ignores_order_marks_and_presentation_forms():
    """Word order, vowel points and Arabic presentation forms do not count as disagreement."""
    assert letter_agreement("עולם שָׁלוֹם", "שלום עולם") == (1.0, 1.0)
    assert letter_agreement("الاقرار", "ﺍﻻﻗﺮﺍﺭ") == (1.0, 1.0)
    recall, precision = letter_agreement("שלום", "שלום עולם")
    assert recall == pytest.approx(0.5) and precision == 1.0
    assert letter_agreement("", "") == (1.0, 1.0)


def test_suspect_glyphs_only_on_semitic_lines():
    """Greek letters inside Arabic lines (Hathaway's lam-alef map) count; Greek in English prose does not."""
    assert suspect_glyphs("اλقرار الشرعي\nاعμه") == 2
    assert suspect_glyphs("the Greek word λόγος") == 0
    assert suspect_glyphs("\uFE8Dλ\uFED7\uFEAE\uFE8D\uFEAD") == 1  # line stored as presentation forms


def test_hybrid_routes_each_page_kind(tmp_path):
    """Only the Hebrew page is sent to Vision; prose with inline Hebrew and plain prose keep the text layer."""
    pdf = _pdf(tmp_path / "book.pdf", [f"<p dir='rtl'>{HEBREW}</p>", MIXED, ENGLISH])
    vision = _FakeVision(" ".join(reversed(HEBREW.split())))
    book_dir = tmp_path / "book"
    out = _service(vision).process_pdf_hybrid_to_file(str(pdf), str(book_dir))
    data = json.loads(Path(out["ocr_results_path"]).read_text(encoding="utf-8"))
    pages = {p["page_number"]: p for p in data["pages"]}
    assert vision.calls == 1
    assert [pages[n]["ocr_source"] for n in (1, 2, 3)] == ["vision", "text_layer", "text_layer"]
    assert pages[1]["routing"]["vision_letter_recall"] == 1.0 and not pages[1]["routing"]["vision_rejected"]
    assert "שלום" in pages[2]["ocr_result"]["full_text"]
    assert all(Path(p["image_path"]).exists() for p in data["pages"])
    info = data["processing_info"]
    assert (info["mode"], info["vision_pages"], info["text_layer_pages"]) == ("hybrid", 1, 2)
    assert not ocr_missing_pages(pdf, book_dir)
    assert existing_ocr_suspect(pdf, book_dir) is None


def test_hybrid_keeps_text_layer_when_vision_loses_letters(tmp_path):
    """Vision that drops most of the Hebrew is rejected and the page keeps its text layer."""
    pdf = _pdf(tmp_path / "book.pdf", [f"<p dir='rtl'>{HEBREW}</p>"])
    vision = _FakeVision("בשרתי צדק")
    out = _service(vision).process_pdf_hybrid_to_file(str(pdf), str(tmp_path / "book"))
    data = json.loads(Path(out["ocr_results_path"]).read_text(encoding="utf-8"))
    page = data["pages"][0]
    assert page["ocr_source"] == "text_layer" and page["routing"]["vision_rejected"]
    assert page["routing"]["vision_letter_recall"] < 0.97
    assert "ישועתי" in page["ocr_result"]["full_text"]
    assert data["processing_info"]["vision_rejected_pages"] == [1]


def test_hybrid_vision_failure_is_retried_on_resume(tmp_path):
    """A Vision error leaves a retryable stub; the next run calls Vision for that page only."""
    pdf = _pdf(tmp_path / "book.pdf", [f"<p dir='rtl'>{HEBREW}</p>", ENGLISH, f"<p dir='rtl'>{HEBREW}</p>"])
    book_dir = tmp_path / "book"
    first = _FakeVision(RuntimeError("quota"), HEBREW)
    _service(first).process_pdf_hybrid_to_file(str(pdf), str(book_dir))
    assert first.calls == 2 and ocr_missing_pages(pdf, book_dir) == [1]
    second = _FakeVision(HEBREW)
    out = _service(second).process_pdf_hybrid_to_file(str(pdf), str(book_dir))
    data = json.loads(Path(out["ocr_results_path"]).read_text(encoding="utf-8"))
    assert second.calls == 1 and not ocr_missing_pages(pdf, book_dir)
    assert [p["ocr_source"] for p in data["pages"]] == ["vision", "text_layer", "vision"]


def _scan(path: Path, pages: int = 2) -> Path:
    """Write an image-only PDF (a full-page raster per page, no text layer).

    :param path: Output PDF path.
    :param pages: Page count.
    :returns: *path*.
    """
    doc = fitz.open()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 60, 80), False)
    pix.clear_with(200)
    for _ in range(pages):
        page = doc.new_page()
        page.insert_image(page.rect, pixmap=pix)
    doc.save(path)
    return path


def test_hybrid_on_a_scan_falls_back_to_vision_and_its_output_is_suspect(tmp_path):
    """A scan has no text layer to keep: planning switches to Vision, and hybrid output on it is not trusted."""
    pdf = _scan(tmp_path / "scan.pdf")
    plan = plan_book(pdf, tmp_path, "hybrid", None, "{stem}_structured_x", False)
    assert plan.ocr_mode == "vision" and "hybrid needs a born-digital text layer" in plan.ocr_reason
    vision = _FakeVision()
    _service(vision).process_pdf_hybrid_to_file(str(pdf), str(tmp_path / "scan"))
    assert vision.calls == 0
    assert "scans" in existing_ocr_suspect(pdf, tmp_path / "scan")


def test_hybrid_flags_reversed_hebrew_it_kept_from_the_layer(tmp_path):
    """Visual-order Hebrew inline on English pages stays on the text layer, so the output is flagged."""
    reversed_words = " ".join(w[::-1] for w in ("שלום מלך נתן אברהם בן יוסף " * 6).split())
    prose = ENGLISH * 4 + " " + reversed_words
    pdf = _pdf(tmp_path / "rev.pdf", [prose, prose])
    _service(_FakeVision()).process_pdf_hybrid_to_file(str(pdf), str(tmp_path / "rev"))
    assert "reversed Hebrew" in existing_ocr_suspect(pdf, tmp_path / "rev")


def test_unfinished_hybrid_resumes_as_hybrid(tmp_path):
    """auto and --ocr-mode hybrid (the documented re-run) both resume an unfinished hybrid checkpoint in hybrid
    mode; a conflicting forced mode is refused."""
    pdf = _pdf(tmp_path / "book.pdf", [f"<p dir='rtl'>{HEBREW}</p>", ENGLISH])
    _service(_FakeVision(RuntimeError("quota"))).process_pdf_hybrid(
        str(pdf), str(tmp_path / "book"), str(tmp_path / "book" / "book_ocr_results.json"))
    assert ocr_checkpoint_mode(tmp_path / "book") == "hybrid"
    for requested in ("auto", "hybrid"):
        plan = plan_book(pdf, tmp_path, requested, None, "{stem}_structured_x", False)
        assert plan.ocr_mode == "hybrid" and plan.blocked is None, requested
        assert plan.ocr_reason == "resuming an unfinished hybrid OCR"
    forced = plan_book(pdf, tmp_path, "text", None, "{stem}_structured_x", False)
    assert forced.ocr_mode is None and "--ocr-mode hybrid" in forced.blocked


def test_overprinted_fake_bold_does_not_reject_a_correct_vision_read(tmp_path):
    """Glyphs drawn twice for fake bold are counted once, so Vision's single copy matches the layer."""
    path = tmp_path / "bold.pdf"
    doc = fitz.open()
    page = doc.new_page()
    for dx in (0, 0.4):
        page.insert_htmlbox(page.rect + (50 + dx, 50, -50 + dx, -50), f"<p dir='rtl'>{HEBREW}</p>")
    doc.save(path)
    out = _service(_FakeVision(HEBREW)).process_pdf_hybrid_to_file(str(path), str(tmp_path / "bold"))
    page_result = json.loads(Path(out["ocr_results_path"]).read_text(encoding="utf-8"))["pages"][0]
    assert page_result["ocr_source"] == "vision" and page_result["routing"]["vision_letter_recall"] == 1.0


def test_hybrid_refuses_to_finish_another_modes_checkpoint(tmp_path):
    """Forcing hybrid over an unfinished text-layer run would mix unrouted pages into a 'hybrid' file."""
    pdf = _pdf(tmp_path / "book.pdf", [ENGLISH, ENGLISH])
    book_dir = tmp_path / "book"
    book_dir.mkdir()
    _service(_FakeVision()).process_pdf_text_only(str(pdf), end_page=0, output_dir=str(book_dir),
                                                  checkpoint_path=str(book_dir / "book_ocr_results.json"))
    assert ocr_missing_pages(pdf, book_dir) == [2]
    plan = plan_book(pdf, tmp_path, "hybrid", None, "{stem}_structured_x", False)
    assert plan.ocr_mode is None and "move book_ocr_results.json aside" in plan.blocked
