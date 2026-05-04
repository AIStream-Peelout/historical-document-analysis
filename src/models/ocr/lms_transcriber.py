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

    max_tokens: int = 4096
    """Enough headroom for a full JSON response with three sections."""

    timeout_s: float = 180.0
    """LM Studio on a local GPU is fast; generous timeout for CPU fallback."""

    connect_timeout_s: float = 5.0
    """Fail fast if LM Studio isn't running."""

    max_retries: int = 2
    """Retry on transient errors (OOM recovery, context window overflow)."""

    retry_delay_s: float = 5.0


# Singleton used by the eval loop; callers may replace it.
DEFAULT_CONFIG = LMStudioConfig()


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

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 503:
                        # Model still loading — wait and retry
                        body = await resp.text()
                        print(
                            f"    ⏳ LM Studio 503 (model loading?) — "
                            f"attempt {attempt + 1}/{max_retries + 1}: {body[:120]}"
                        )
                        if attempt < max_retries:
                            await asyncio.sleep(retry_delay * (attempt + 1))
                            continue
                        return None

                    resp.raise_for_status()
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
# Convenience: run the same prompt across multiple models concurrently
# ============================================================================

async def transcribe_batch(
    model_ids: List[str],
    image_path: str,
    prompt: str,
    base_url: str = DEFAULT_CONFIG.base_url,
    **kwargs,
) -> dict[str, Optional[str]]:
    """Transcribe one image with multiple LM Studio models concurrently.

    Returns {model_id: raw_text_or_None}.
    """
    tasks = {
        model_id: transcribe_with_lm_studio(
            model_id, image_path, prompt, base_url=base_url, **kwargs
        )
        for model_id in model_ids
    }
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    out: dict[str, Optional[str]] = {}
    for model_id, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            print(f"    ✗ [{model_id}] Exception: {result}")
            out[model_id] = None
        else:
            out[model_id] = result
    return out