"""
lm_studio_transcriber.py
Async transcription via LM Studio's OpenAI-compatible local API.

LM Studio exposes a POST /v1/chat/completions endpoint that accepts the same
multimodal message format as OpenAI, so any vision-capable model loaded in
LM Studio (e.g. qwen/qwen3-vl-8b) works here without changes.

Usage:
    from src.models.ocr.lm_studio_transcriber import (
        transcribe_with_lm_studio,
        check_lm_studio_health,
        LMStudioConfig,
    )

    # Single model
    raw = await transcribe_with_lm_studio(
        model_id="qwen/qwen3-vl-8b",
        image_path="/path/to/page.jpg",
        prompt="Transcribe this...",
    )

    # Check which models are currently loaded
    models = await check_lm_studio_health()
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import aiohttp


# ============================================================================
# Config
# ============================================================================

@dataclass
class LMStudioConfig:
    """Runtime config for LM Studio.  Override via CLI args or env vars."""

    base_url: str = "http://localhost:1234/v1"
    """LM Studio server root — change if you use a non-default port."""

    default_models: List[str] = field(
        default_factory=lambda: ["qwen/qwen3-vl-8b"]
    )
    """Models to include in the benchmark when none are specified on the CLI."""

    temperature: float = 0.1
    """Low temperature → less creative hallucination (matches Gemini calls)."""

    max_tokens: int = 8192
    """Headroom for a full section transcription PLUS thinking tokens —
    reasoning models (e.g. Gemma 4) burn hundreds of tokens on a hidden
    reasoning channel before emitting content; too small a budget returns
    finish_reason=length with EMPTY content."""

    timeout_s: float = 600.0
    """Large local models are slow: gemma-4-31b-qat generates ~16 tok/s on
    Apple Silicon, so a full Gemara section (thinking + ~4k output tokens)
    can take 4-5 minutes. 180s produced spurious timeouts."""

    connect_timeout_s: float = 5.0
    """Fail fast if LM Studio isn't running."""

    max_retries: int = 4
    """Retry on transient errors (OOM recovery, context window overflow).
    Multi-model benchmark runs evict/reload models on every swap, and the
    first request(s) to a just-evicted model can 400 with "Model unloaded"
    before the JIT load completes — extra attempts make swaps reliable."""

    retry_delay_s: float = 5.0


# Singleton used by the eval loop; callers may replace it.
DEFAULT_CONFIG = LMStudioConfig()


# ============================================================================
# Global serialization — ONE LM Studio inference request at a time
# ============================================================================
# The local server holds one large model in memory; concurrent requests make
# it evict/reload models (400 "Model unloaded") and thrash limited RAM.
# Every inference entry point in this module acquires this lock, so no two
# LM Studio requests are ever in flight simultaneously — process-wide,
# regardless of how callers schedule their coroutines.

_lms_lock: Optional[asyncio.Lock] = None
_lms_lock_loop: Optional[asyncio.AbstractEventLoop] = None


def _get_lms_lock() -> asyncio.Lock:
    """Return the module-wide LM Studio lock for the running event loop.

    Created lazily per event loop (an asyncio.Lock cannot be shared across
    loops, and tests/scripts may call asyncio.run() several times).

    :return: Lock serializing all LM Studio inference calls
    :rtype: asyncio.Lock
    """
    global _lms_lock, _lms_lock_loop
    loop = asyncio.get_running_loop()
    if _lms_lock is None or _lms_lock_loop is not loop:
        _lms_lock = asyncio.Lock()
        _lms_lock_loop = loop
    return _lms_lock


# ============================================================================
# Health check — call once at startup to verify the server is reachable
# and log which models are currently loaded.
# ============================================================================

async def check_lm_studio_health(
    base_url: str = DEFAULT_CONFIG.base_url,
) -> List[str]:
    """Return list of model IDs currently available in LM Studio.

    Raises `RuntimeError` if the server is unreachable so the caller can
    skip LM Studio gracefully rather than timing out per document.
    """
    url = f"{base_url}/models"
    timeout = aiohttp.ClientTimeout(total=DEFAULT_CONFIG.connect_timeout_s)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.json()
                models = [m["id"] for m in data.get("data", [])]
                return models
    except aiohttp.ClientConnectorError:
        raise RuntimeError(
            f"LM Studio is not reachable at {base_url}. "
            "Start LM Studio and enable the local server (port 1234 by default)."
        )
    except Exception as exc:
        raise RuntimeError(f"LM Studio health check failed: {exc}") from exc


# ============================================================================
# Image encoding helper
# ============================================================================

def _encode_image(image_path: str) -> tuple[str, str]:
    """Return (base64_data, mime_type) for the image at *image_path*.

    Supports JPEG, PNG, TIFF (converted to PNG bytes for the API call since
    most vision models don't accept TIFF natively).
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime, _ = mimetypes.guess_type(str(path))
    raw_bytes = path.read_bytes()

    # TIFF → PNG conversion so VLMs that don't accept TIFF still work
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            from PIL import Image
            import io
            img = Image.open(path)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            raw_bytes = buf.getvalue()
            mime = "image/png"
        except ImportError:
            pass  # fall through and send raw — model may still handle it

    mime = mime or "image/jpeg"
    return base64.b64encode(raw_bytes).decode("utf-8"), mime


# ============================================================================
# Core transcription call
# ============================================================================

async def transcribe_with_lm_studio(
    model_id: str,
    image_path: str,
    prompt: str,
    base_url: str = DEFAULT_CONFIG.base_url,
    temperature: float = DEFAULT_CONFIG.temperature,
    max_tokens: int = DEFAULT_CONFIG.max_tokens,
    timeout: float = DEFAULT_CONFIG.timeout_s,
    max_retries: int = DEFAULT_CONFIG.max_retries,
    retry_delay: float = DEFAULT_CONFIG.retry_delay_s,
) -> Optional[str]:
    """Call a vision-capable model loaded in LM Studio and return the raw text.

    The function sends the image as a base64 data URI inside a standard
    OpenAI multimodal message.  The prompt should ask the model to return
    JSON (section-aware Talmud prompts already do this).

    Returns None on unrecoverable failure so the caller can log the error
    and continue with other models.
    """
    b64, mime = _encode_image(image_path)
    data_uri = f"data:{mime};base64,{b64}"

    payload = {
        "model": model_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    }

    url = f"{base_url}/chat/completions"
    client_timeout = aiohttp.ClientTimeout(total=timeout)

    # Hold the module-wide lock for the entire call (retries included):
    # LM Studio must never see two inference requests at once.
    async with _get_lms_lock():
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                async with aiohttp.ClientSession(timeout=client_timeout) as session:
                    async with session.post(url, json=payload) as resp:
                        if resp.status != 200:
                            # Always surface LM Studio's error body — e.g.
                            # {"error": "Model unloaded."} when another large
                            # model evicted this one from memory.
                            body = await resp.text()
                            print(
                                f"    ✗ [{model_id}] HTTP {resp.status} "
                                f"(attempt {attempt + 1}/{max_retries + 1}): {body[:200]}"
                            )
                            if attempt < max_retries:
                                await asyncio.sleep(retry_delay * (attempt + 1))
                                continue
                            return None

                        data = await resp.json()

                        # OpenAI-compatible response shape
                        choices = data.get("choices", [])
                        if not choices:
                            print(f"    ✗ [{model_id}] Empty choices in response")
                            return None

                        content = choices[0].get("message", {}).get("content", "")
                        return content.strip() if content else None

            except aiohttp.ClientConnectorError:
                print(f"    ✗ [{model_id}] Cannot connect to LM Studio at {base_url}")
                return None
            except asyncio.TimeoutError:
                last_exc = asyncio.TimeoutError(f"Timed out after {timeout}s")
                print(f"    ⏱ [{model_id}] Timeout on attempt {attempt + 1}")
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay)
            except Exception as exc:
                last_exc = exc
                print(f"    ✗ [{model_id}] Error on attempt {attempt + 1}: {exc}")
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay)

        print(f"    ✗ [{model_id}] All {max_retries + 1} attempts failed. Last: {last_exc}")
        return None


# ============================================================================
# Text-only completion (no image) — used for OCR segmentation/extraction
# ============================================================================

async def complete_text(
    model_id: str,
    prompt: str,
    base_url: str = DEFAULT_CONFIG.base_url,
    temperature: float = DEFAULT_CONFIG.temperature,
    max_tokens: int = DEFAULT_CONFIG.max_tokens,
    timeout: float = 120.0,
) -> Optional[str]:
    """Text-only chat completion against an LM Studio model — no image.

    Used when an open-weight text model is preferred over Gemini for the
    OCR segmentation / extraction step.

    :param model_id: LM Studio model ID
    :type model_id: str
    :param prompt: Full text prompt
    :type prompt: str
    :param base_url: LM Studio server root
    :type base_url: str
    :param temperature: Sampling temperature
    :type temperature: float
    :param max_tokens: Output token budget
    :type max_tokens: int
    :param timeout: Total request timeout in seconds
    :type timeout: float
    :return: Model output text, or None on failure
    :rtype: Optional[str]
    """
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    # Same module-wide lock as image transcription — LM Studio must never
    # see two inference requests at once (limited RAM, model eviction).
    async with _get_lms_lock():
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                async with session.post(f"{base_url}/chat/completions", json=payload) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"] or ""
                    return content.strip() or None
        except Exception as exc:
            print(f"    ✗ LM Studio text-only failed ({model_id}): {exc}")
            return None


# ============================================================================
# Convenience: run the same prompt across multiple models concurrently
# ============================================================================

async def transcribe_batch(
    model_ids: List[str],
    image_path: str,
    prompt: str,
    base_url: str = DEFAULT_CONFIG.base_url,
    **kwargs,
) -> dict[str, Optional[str]]:
    """Transcribe one image with multiple LM Studio models SEQUENTIALLY.

    Sequential on purpose: concurrent requests to two large models make
    LM Studio evict one to load the other, and every request to the evicted
    model fails instantly with 400 ``{"error": "Model unloaded."}``.
    On a single local GPU concurrency buys nothing anyway.

    :param model_ids: LM Studio model IDs to run
    :type model_ids: List[str]
    :param image_path: Path to the document image
    :type image_path: str
    :param prompt: Transcription prompt sent to every model
    :type prompt: str
    :param base_url: LM Studio server root
    :type base_url: str
    :return: Mapping of model_id to raw text (or None on failure)
    :rtype: dict[str, Optional[str]]
    """
    out: dict[str, Optional[str]] = {}
    for model_id in model_ids:
        try:
            out[model_id] = await transcribe_with_lm_studio(
                model_id, image_path, prompt, base_url=base_url, **kwargs
            )
        except Exception as exc:
            print(f"    ✗ [{model_id}] Exception: {exc}")
            out[model_id] = None
    return out