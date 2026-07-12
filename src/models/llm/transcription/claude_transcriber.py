"""
claude_transcriber.py
Async transcription via the Anthropic Claude API.

Claude is used as an optional, flag-gated benchmark model in the evaluation
scripts (tokens are expensive, so it is off by default).  The module mirrors
the calling conventions of ``lms_transcriber.py``: a single async function
that takes an image path and a prompt and returns raw text (or ``None`` on
failure) so eval loops can log the error and continue.

Authentication resolves from the environment (``ANTHROPIC_API_KEY``) or an
``ant auth login`` profile — no key is ever passed in code.

Usage:
    from src.models.llm.transcription.claude_transcriber import (
        transcribe_with_claude,
        ClaudeConfig,
    )

    raw = await transcribe_with_claude(
        image_path="/path/to/page.jpg",
        prompt="Transcribe this manuscript...",
    )
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
from typing import Optional, Tuple

import anthropic
from PIL import Image


# ============================================================================
# Config
# ============================================================================

class ClaudeConfig:
    """Runtime configuration for Claude transcription calls."""

    DEFAULT_MODEL: str = os.getenv("CLAUDE_EVAL_MODEL", "claude-opus-4-8")
    """Model used for evaluation runs.  Override via CLI flag or env var."""

    MAX_TOKENS: int = 16_000
    """Output budget — full-page Talmud sections fit comfortably."""

    TIMEOUT_S: float = 600.0
    """Wall-clock timeout per request (SDK-level)."""

    MAX_RETRIES: int = 2
    """SDK auto-retries on 429/5xx/connection errors."""

    MAX_LONG_EDGE_PX: int = 2576
    """Claude Opus 4.8 native high-res vision limit — larger images are
    downscaled server-side anyway, so we downscale client-side to save
    upload bytes and image tokens."""

    MAX_IMAGE_BYTES: int = 4_500_000
    """Stay under the API's per-image size limit with headroom."""


# Lazily-constructed singleton client (reused across calls for connection
# pooling; safe because AsyncAnthropic is coroutine-safe).
_client: Optional[anthropic.AsyncAnthropic] = None


def _get_client() -> anthropic.AsyncAnthropic:
    """Return the shared AsyncAnthropic client, creating it on first use.

    :return: Shared async Anthropic client instance
    :rtype: anthropic.AsyncAnthropic
    """
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(
            timeout=ClaudeConfig.TIMEOUT_S,
            max_retries=ClaudeConfig.MAX_RETRIES,
        )
    return _client


# ============================================================================
# Image encoding
# ============================================================================

def _encode_image_for_claude(image_path: str) -> Tuple[str, str]:
    """Prepare an image for the Claude API.

    Converts TIFF/other formats to PNG, downscales to the model's native
    high-resolution limit, and re-encodes as JPEG if the payload would
    exceed the per-image size limit.

    :param image_path: Path to the source image (JPEG, PNG, or TIFF)
    :type image_path: str
    :return: Tuple of (base64-encoded data, media type)
    :rtype: Tuple[str, str]
    :raises FileNotFoundError: If the image does not exist
    """
    img = Image.open(image_path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    long_edge = max(img.size)
    if long_edge > ClaudeConfig.MAX_LONG_EDGE_PX:
        ratio = ClaudeConfig.MAX_LONG_EDGE_PX / long_edge
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw = buf.getvalue()
    media_type = "image/png"

    if len(raw) > ClaudeConfig.MAX_IMAGE_BYTES:
        rgb = img.convert("RGB") if img.mode != "RGB" else img
        for quality in (92, 85, 75, 60):
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=quality)
            raw = buf.getvalue()
            media_type = "image/jpeg"
            if len(raw) <= ClaudeConfig.MAX_IMAGE_BYTES:
                break

    return base64.b64encode(raw).decode("utf-8"), media_type


# ============================================================================
# Core transcription call
# ============================================================================

async def transcribe_with_claude(
    image_path: str,
    prompt: str,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> Optional[str]:
    """Transcribe an image with Claude and return the raw text.

    Uses streaming with ``get_final_message()`` so long outputs never hit
    HTTP timeouts, and adaptive thinking for better transcription quality.
    Note that Opus 4.8 rejects ``temperature`` — determinism is steered via
    the prompt only.

    :param image_path: Path to the document image
    :type image_path: str
    :param prompt: Transcription instruction sent alongside the image
    :type prompt: str
    :param model: Claude model ID; defaults to :attr:`ClaudeConfig.DEFAULT_MODEL`
    :type model: Optional[str]
    :param max_tokens: Output token budget; defaults to :attr:`ClaudeConfig.MAX_TOKENS`
    :type max_tokens: Optional[int]
    :return: Transcribed text, or ``None`` on refusal / unrecoverable failure
    :rtype: Optional[str]
    """
    model = model or ClaudeConfig.DEFAULT_MODEL
    max_tokens = max_tokens or ClaudeConfig.MAX_TOKENS

    try:
        # PIL work is CPU-bound and synchronous — keep it off the event loop.
        b64, media_type = await asyncio.to_thread(_encode_image_for_claude, image_path)
    except FileNotFoundError:
        print(f"    ✗ [claude] Image not found: {image_path}")
        return None
    except Exception as exc:
        print(f"    ✗ [claude] Image encoding failed: {exc}")
        return None

    client = _get_client()

    try:
        async with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        ) as stream:
            response = await stream.get_final_message()
    except anthropic.AuthenticationError:
        print(
            "    ✗ [claude] Authentication failed — set ANTHROPIC_API_KEY "
            "or run `ant auth login`."
        )
        return None
    except anthropic.RateLimitError as exc:
        print(f"    ✗ [claude] Rate limited after SDK retries: {exc}")
        return None
    except anthropic.APIStatusError as exc:
        print(f"    ✗ [claude] API error {exc.status_code}: {str(exc)[:200]}")
        return None
    except anthropic.APIConnectionError as exc:
        print(f"    ✗ [claude] Connection error: {exc}")
        return None
    except Exception as exc:
        print(f"    ✗ [claude] Unexpected error: {type(exc).__name__}: {str(exc)[:200]}")
        return None

    if response.stop_reason == "refusal":
        print(f"    ⛔ [claude] Refusal (model declined to transcribe) — skipping")
        return None

    text = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()

    if response.stop_reason == "max_tokens":
        print(f"    ⚠️  [claude] Hit max_tokens — output may be truncated ({len(text)} chars)")

    return text or None
