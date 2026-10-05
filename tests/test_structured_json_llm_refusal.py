"""A provider refusal in Pass 1 must come back as a failed page, not a valid empty one.

Gemini content-filter stops (RECITATION, SAFETY, ...) raise
``UnexpectedModelBehavior`` whose ``body`` is the raw API response. The old
fallback path pulled "JSON" out of that body, filled defaults and saved an
empty page that passed validation, so no driver ever retried it.
"""

import asyncio

from pydantic_ai.exceptions import UnexpectedModelBehavior

from src.models.llm.academic.structured_json_llm import StructuredJSONLLM


class _RaisingAgent:
    """Agent stub whose ``run`` raises the given exception."""

    def __init__(self, exc: Exception):
        """:param exc: Exception to raise from ``run``."""
        self.exc = exc

    async def run(self, *args, **kwargs):
        """Raise the configured exception.

        :param args: Ignored.
        :param kwargs: Ignored.
        """
        raise self.exc


def _llm(tmp_path, exc: Exception) -> StructuredJSONLLM:
    """Build a StructuredJSONLLM without touching any model provider.

    :param tmp_path: pytest temp dir for debug outputs.
    :param exc: Exception the agent raises.
    :returns: Instance wired to the raising stub.
    """
    llm = object.__new__(StructuredJSONLLM)
    llm.raw_data_dir = tmp_path
    llm.failed_outputs_dir = tmp_path / "failed"
    llm.raw_outputs_dir = tmp_path / "raw"
    llm.failed_outputs_dir.mkdir()
    llm.raw_outputs_dir.mkdir()
    llm.use_gemini = True
    llm.model_name = "gemini-test"
    llm.book_metadata = None
    llm.agent = _RaisingAgent(exc)
    return llm


def test_content_filter_refusal_is_a_failed_page(tmp_path):
    """RECITATION with a JSON body yields validation_failed, not an empty valid page."""
    body = '{"candidates": [{"finish_reason": "RECITATION"}], "usage_metadata": {"total_token_count": 1500}}'
    llm = _llm(tmp_path, UnexpectedModelBehavior("Content filter 'RECITATION' triggered", body))
    out = asyncio.run(llm.process_page_text_only("some page text", 7, "book"))
    assert out["metadata"]["validation_failed"] is True
    assert "RECITATION" in out["metadata"]["error_message"]


def test_retry_exhaustion_without_body_still_attempts_json_rescue(tmp_path):
    """Output-validation exhaustion (no body) keeps the old rescue path, here yielding a failure stub."""
    llm = _llm(tmp_path, UnexpectedModelBehavior("Exceeded maximum retries (3) for output validation"))
    out = asyncio.run(llm.process_page_text_only("some page text", 8, "book"))
    assert out["metadata"]["validation_failed"] is True
