# File name: test_lms_transcriber_meta.py
# Date: 10/24/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests of the LM Studio transcriber's completion metadata (finish reason, token counts) with a fake aiohttp session."""
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from src.models.ocr import lms_transcriber as lms

REPLY = {"choices": [{"message": {"content": "  بسم الله \n"}, "finish_reason": "length"}],
         "usage": {"prompt_tokens": 1200, "completion_tokens": 2500, "total_tokens": 3700}}


# --- completion_meta --------------------------------------------------------------------------------------------------

def test_completion_meta() -> None:
    assert lms.completion_meta(REPLY) == {"text": "بسم الله", "finish_reason": "length", "prompt_tokens": 1200,
                                          "completion_tokens": 2500, "total_tokens": 3700}


@pytest.mark.parametrize("choice", [{"message": {"content": ""}}, {"message": {"content": None}}, {"message": {}},
                                    {"message": None}, {}])
def test_completion_meta_empty_content_is_none(choice: Dict[str, Any]) -> None:
    meta = lms.completion_meta({"choices": [dict(choice, finish_reason="stop")]})
    assert meta["text"] is None and meta["finish_reason"] == "stop"


def test_completion_meta_without_usage_or_finish_reason() -> None:
    meta = lms.completion_meta({"choices": [{"message": {"content": "x"}}]})
    assert meta == {"text": "x", "finish_reason": None, "prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    assert lms.completion_meta({"choices": [{"message": {"content": "x"}}], "usage": None})["prompt_tokens"] is None
    partial = lms.completion_meta({"choices": [{"message": {"content": "x"}, "finish_reason": "eos"}], "usage": {"completion_tokens": 4}})
    assert (partial["finish_reason"], partial["completion_tokens"], partial["prompt_tokens"]) == ("eos", 4, None)


def test_completion_meta_whitespace_content_is_an_empty_string() -> None:
    # Unchanged from the text-only function this wraps: whitespace is stripped to "", not None (the ablation scorer
    # counts "" as an "empty" abstention).
    assert lms.completion_meta({"choices": [{"message": {"content": " \n "}}]})["text"] == ""


# --- the request ------------------------------------------------------------------------------------------------------

class FakeResponse:
    """Fake aiohttp response."""

    def __init__(self, status: int, body: Any) -> None:
        """
        :param status: HTTP status.
        :type status: int
        :param body: JSON body (dict) or error text (str).
        :type body: Any
        """
        self.status = status
        self.body = body

    async def json(self) -> Dict[str, Any]:
        """Reply body.

        :return: The body.
        :rtype: Dict[str, Any]
        """
        return self.body

    async def text(self) -> str:
        """Error body.

        :return: The body as text.
        :rtype: str
        """
        return str(self.body)

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


def fake_session(replies: Sequence[Tuple[int, Any]], sent: List[Dict[str, Any]]) -> type:
    """A fake ``aiohttp.ClientSession`` class answering posts with ``replies`` in turn.

    :param replies: ``(status, body)`` per request.
    :type replies: Sequence[Tuple[int, Any]]
    :param sent: Receives each posted body.
    :type sent: List[Dict[str, Any]]
    :return: Session class.
    :rtype: type
    """
    queue = list(replies)

    class Session:
        """Fake session."""

        def __init__(self, timeout: Any = None) -> None:
            self.timeout = timeout

        def post(self, url: str, json: Dict[str, Any]) -> FakeResponse:
            """Answer one request.

            :param url: Endpoint.
            :type url: str
            :param json: Body.
            :type json: Dict[str, Any]
            :return: The next reply.
            :rtype: FakeResponse
            """
            sent.append(dict(json, _url=url))
            return FakeResponse(*queue.pop(0))

        async def __aenter__(self) -> "Session":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

    return Session


@pytest.fixture()
def image(tmp_path: Path) -> str:
    """A small file standing in for a page photo (the transcriber only base64-encodes it).

    :param tmp_path: Directory.
    :type tmp_path: Path
    :return: Its path.
    :rtype: str
    """
    path = tmp_path / "page.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
    return str(path)


def _call(monkeypatch: pytest.MonkeyPatch, replies: Sequence[Tuple[int, Any]], image: str, meta: bool = True,
          **kwargs: Any) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
    """Run one transcription against the fake session.

    :param monkeypatch: Pytest monkeypatch.
    :type monkeypatch: pytest.MonkeyPatch
    :param replies: ``(status, body)`` per request.
    :type replies: Sequence[Tuple[int, Any]]
    :param image: Image path.
    :type image: str
    :param meta: Call the meta function (else the text wrapper).
    :type meta: bool
    :return: ``(result, posted bodies)``.
    :rtype: Tuple[Optional[Any], List[Dict[str, Any]]]
    """
    sent: List[Dict[str, Any]] = []
    monkeypatch.setattr(lms.aiohttp, "ClientSession", fake_session(replies, sent))
    fn = lms.transcribe_with_lm_studio_meta if meta else lms.transcribe_with_lm_studio
    return asyncio.run(fn("m", image, "p", base_url="http://fake/v1", retry_delay=0.0, **kwargs)), sent


def test_meta_returns_the_completion_meta(monkeypatch: pytest.MonkeyPatch, image: str) -> None:
    out, sent = _call(monkeypatch, [(200, REPLY)], image, max_tokens=2500, temperature=0.1)
    assert out == lms.completion_meta(REPLY)
    assert len(sent) == 1 and sent[0]["_url"] == "http://fake/v1/chat/completions"
    assert sent[0]["model"] == "m" and sent[0]["max_tokens"] == 2500 and sent[0]["temperature"] == 0.1
    assert sent[0]["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_text_wrapper_returns_exactly_the_meta_text(monkeypatch: pytest.MonkeyPatch, image: str) -> None:
    out, _ = _call(monkeypatch, [(200, REPLY)], image, meta=False)
    assert out == "بسم الله"
    empty, _ = _call(monkeypatch, [(200, {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]})], image, meta=False)
    assert empty is None


def test_http_error_after_retries_is_none(monkeypatch: pytest.MonkeyPatch, image: str) -> None:
    out, sent = _call(monkeypatch, [(400, '{"error": "Model unloaded."}')], image, max_retries=0)
    assert out is None and len(sent) == 1
    text, _ = _call(monkeypatch, [(500, "boom")], image, meta=False, max_retries=0)
    assert text is None


def test_retry_then_success(monkeypatch: pytest.MonkeyPatch, image: str) -> None:
    out, sent = _call(monkeypatch, [(400, "unloaded"), (200, REPLY)], image, max_retries=1)
    assert out == lms.completion_meta(REPLY) and len(sent) == 2


def test_empty_choices_is_none(monkeypatch: pytest.MonkeyPatch, image: str) -> None:
    out, _ = _call(monkeypatch, [(200, {"choices": []})], image, max_retries=0)
    assert out is None
    missing, _ = _call(monkeypatch, [(200, {"object": "chat.completion"})], image, max_retries=0)
    assert missing is None
