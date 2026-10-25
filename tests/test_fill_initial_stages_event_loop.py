"""Tests for the multi-PDF "Event loop is closed" failure in the initial-stage filler.

pydantic-ai keeps one httpx client per provider for the whole process. Its pooled
connections belong to the event loop that opened them, so when every book ran
under its own ``asyncio.run()``, each book after the first lost its first pages to
``RuntimeError("Event loop is closed")``, which ``StructuredJSONLLM`` turns into a
``[VALIDATION FAILED]`` stub. The fake client below reproduces that shape without
any network or model.
"""

import asyncio
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import fitz
import pytest

from src.datasets.indexing.bibliography import fill_initial_stages as fis

LONG_TEXT = "Genizah fragment discussed at length on this page. " * 12


class LoopBoundClient:
    """Stand-in for pydantic-ai's process-wide cached httpx client.

    The pooled connection belongs to the event loop of the request that opened it.
    Using it after that loop has closed raises ``RuntimeError("Event loop is closed")``
    and drops it, so the next request opens a fresh connection on the running loop.
    """

    def __init__(self, stale_loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Create the client, optionally holding a connection from an already-closed loop.

        :param stale_loop: Loop the pooled connection was opened on (None for an empty pool).
        """
        self.loop = stale_loop
        self.closed_loop_errors = 0

    async def send(self) -> None:
        """Make one request.

        :raises RuntimeError: When the pooled connection's loop has closed.
        """
        if self.loop is not None and self.loop.is_closed():
            self.loop = None
            self.closed_loop_errors += 1
            raise RuntimeError("Event loop is closed")
        self.loop = asyncio.get_running_loop()


class FakeLLM:
    """Mimics ``StructuredJSONLLM``: a client exception becomes a failure stub, never a raise."""

    def __init__(self, client: LoopBoundClient) -> None:
        """Wrap *client*.

        :param client: Shared HTTP client stand-in.
        """
        self.client = client
        self.calls: Counter = Counter()

    async def process_page_text_only(self, text: str, page: int, pdf_name: str) -> Dict[str, Any]:
        """Return a valid page, or the fallback stub ``StructuredJSONLLM`` writes on an exception.

        :param text: OCR text.
        :param page: Page number.
        :param pdf_name: Book name.
        :returns: Page result.
        """
        self.calls[(pdf_name, page)] += 1
        try:
            await self.client.send()
        except RuntimeError as e:
            return {"summary": f"[VALIDATION FAILED] Page processing failed: {e}",
                    "metadata": {"page_number": page, "validation_failed": True, "error_message": str(e)}}
        return {"summary": "ok", "metadata": {"page_number": page}}

    async def process_page(self, text: str, image: Path, page: int, pdf_name: str) -> Dict[str, Any]:
        """Image variant; the fake ignores the image.

        :param text: OCR text.
        :param image: Page image path.
        :param page: Page number.
        :param pdf_name: Book name.
        :returns: Page result.
        """
        return await self.process_page_text_only(text, page, pdf_name)


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """Retry without the production backoff wait.

    :param monkeypatch: pytest monkeypatch.
    """
    monkeypatch.setattr(fis, "RETRY_BASE_WAIT_S", 0)


def _book(root: Path, name: str, pages: int = 2) -> Path:
    """Write a born-digital PDF with exact-stem metadata and complete text-layer OCR.

    :param root: Corpus root.
    :param name: Collection folder and PDF stem.
    :param pages: Page count.
    :returns: The PDF path.
    """
    pdf = root / name / f"{name}.pdf"
    pdf.parent.mkdir(parents=True)
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_textbox(page.rect + (36, 36, -36, -36), LONG_TEXT)
    doc.save(pdf)
    (pdf.parent / f"{name}_metadata.json").write_text(json.dumps({"title": name, "authors": ["A B"]}))
    book_dir = pdf.parent / name
    book_dir.mkdir()
    fis.ocr_json_path(book_dir).write_text(json.dumps({
        "pages": [{"page_number": n, "ocr_result": {"full_text": LONG_TEXT}} for n in range(1, pages + 1)],
        "processing_info": {"mode": "text_only"},
    }))
    return pdf


def _plan(pdf: Path) -> fis.BookPlan:
    """Build a Pass-1 plan for every page of a :func:`_book` PDF.

    :param pdf: PDF written by :func:`_book`.
    :returns: Plan writing into ``<stem>_structured_fake``.
    """
    book_dir = pdf.parent / pdf.stem
    pages = sorted(fis.load_ocr_pages(book_dir))
    return fis.BookPlan(pdf=pdf, book_dir=book_dir, ocr_mode=None, ocr_reason="OCR complete", suspect_ocr=None,
                        metadata_path=None, metadata_kind="none", variant_dir=book_dir / f"{pdf.stem}_structured_fake",
                        todo_pages=pages)


def _failed(plan: fis.BookPlan, page: int) -> bool:
    """Read a written page's ``validation_failed`` flag.

    :param plan: Plan whose variant dir holds the page.
    :param page: Page number.
    :returns: The flag.
    """
    data = json.loads((plan.variant_dir / f"page_{page:03d}_structured.json").read_text())
    return bool(data["metadata"].get("validation_failed"))


def test_closed_loop_error_is_transient():
    """The stub's message is retryable; a schema failure still is not."""
    fail = lambda msg: {"metadata": {"validation_failed": True, "error_message": msg}}
    assert fis._is_transient(fail("Event loop is closed"))
    assert not fis._is_transient(fail("1 validation error for StructuredPageData"))


def test_closed_loop_page_is_retried_and_completes(tmp_path, caplog):
    """A client that raises "Event loop is closed" once: the page is retried, logged, and written valid."""
    dead = asyncio.new_event_loop()
    dead.close()
    llm = FakeLLM(LoopBoundClient(stale_loop=dead))
    plan = _plan(_book(tmp_path, "book"))
    with caplog.at_level(logging.INFO, logger="fill_initial_stages"):
        ok, bad = asyncio.run(fis.run_pass1(plan, llm, concurrency=4, retries=2))
    assert (ok, bad) == (2, 0)
    assert llm.client.closed_loop_errors == 1
    assert llm.calls == Counter({("book", 1): 2, ("book", 2): 1})
    assert not _failed(plan, 1) and not _failed(plan, 2)
    assert "book p1: transient failure (Event loop is closed), retry 1/2 in 0s" in caplog.text
    assert "book p1: ok (retries: 1)" in caplog.text


def test_separate_loops_per_book_reproduce_the_failure_and_retry_recovers(tmp_path):
    """One asyncio.run per book (the old driver) fails book 2's first page; a retry recovers it."""
    first, second = _plan(_book(tmp_path, "first")), _plan(_book(tmp_path, "second"))
    llm = FakeLLM(LoopBoundClient())
    assert asyncio.run(fis.run_pass1(first, llm, concurrency=4, retries=0)) == (2, 0)
    assert asyncio.run(fis.run_pass1(second, llm, concurrency=4, retries=0)) == (1, 1)
    assert _failed(second, 1) and not _failed(second, 2)

    llm = FakeLLM(LoopBoundClient())
    assert asyncio.run(fis.run_pass1(first, llm, concurrency=4, retries=2)) == (2, 0)
    assert asyncio.run(fis.run_pass1(second, llm, concurrency=4, retries=2)) == (2, 0)
    assert llm.client.closed_loop_errors == 1 and llm.calls[("second", 1)] == 2
    assert not _failed(second, 1)


def test_main_runs_every_book_on_one_event_loop(tmp_path, monkeypatch):
    """Several --pdf in one process: no closed-loop error even with retries off, and every page validates."""
    root = tmp_path / "academic_literature"
    pdfs = [_book(root, name) for name in ("alpha", "beta", "gamma")]
    client = LoopBoundClient()  # shared across books, like pydantic-ai's cached client
    llms = []

    def build_llm(*_args: Any) -> FakeLLM:
        llms.append(FakeLLM(client))
        return llms[-1]

    monkeypatch.setattr(fis, "build_llm", build_llm)
    monkeypatch.setattr(fis, "_PROJECT_ROOT", tmp_path)  # no .env here
    monkeypatch.setattr(fis.logging, "basicConfig", lambda **_kwargs: None)
    argv = ["--root", str(root), "--retries", "0"]
    for pdf in pdfs:
        argv += ["--pdf", pdf.relative_to(root).as_posix()]
    assert fis.main(argv) == 0
    assert len(llms) == 3 and client.closed_loop_errors == 0
    for pdf in pdfs:
        variant = pdf.parent / pdf.stem / fis.default_variant_name("gemini", "", fis.DEFAULT_GEMINI_MODEL).format(
            stem=pdf.stem)
        pages = [json.loads(f.read_text()) for f in sorted(variant.glob("page_*_structured.json"))]
        assert len(pages) == 2
        assert all(not p["metadata"].get("validation_failed") and p["full_main_text"] == LONG_TEXT for p in pages)
