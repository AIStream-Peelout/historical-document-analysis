"""
kraken_transcriber.py
Async client wrapper hitting the Kraken FastAPI microservice.

Key design decisions:
- Kraken is run in a separate Docker container because it requires an older Python environment.
- The microservice exposes /preload and /transcribe endpoints.
"""

import asyncio
import logging
import os
import aiohttp
from typing import Optional

logger = logging.getLogger(__name__)

KRAKEN_MICROSERVICE_URL = os.getenv("KRAKEN_MICROSERVICE_URL", "http://localhost:8002")

def preload_kraken_model(model_path: str) -> None:
    """Preload the Kraken model by hitting the microservice synchronously."""
    import requests
    url = f"{KRAKEN_MICROSERVICE_URL}/preload"
    logger.info(f"Triggering preload via {url} for model: {model_path}")
    try:
        response = requests.post(url, json={"model_path": model_path}, timeout=60)
        response.raise_for_status()
        logger.info("Kraken model preloaded in microservice.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to preload model in microservice: {e}")

async def transcribe_with_kraken(
    model_path: str,
    image_path: str,
    timeout: Optional[float] = 120.0,
) -> Optional[str]:
    """Send image to the Kraken microservice for transcription."""
    url = f"{KRAKEN_MICROSERVICE_URL}/transcribe"
    
    try:
        # Provide timeout for aiohttp
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            with open(image_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("model_path", model_path)
                data.add_field("image", f, filename=os.path.basename(image_path))
                
                async with session.post(url, data=data) as response:
                    if response.status != 200:
                        err = await response.text()
                        logger.error(f"Microservice error: {response.status} {err}")
                        return None
                    
                    result = await response.json()
                    
                    # Log diagnostics similar to previous implementation
                    fails = result.get("polygon_failures", 0)
                    binarized = result.get("used_binarization", True)
                    text = result.get("text", "")
                    
                    logger.info(
                        f"  Kraken service (binarize={binarized}): {len(text.splitlines())} lines kept, "
                        f"{fails} polygon failures"
                    )
                    
                    return text

    except asyncio.TimeoutError:
        logger.error(f"Kraken microservice request timed out after {timeout}s on {image_path}")
        return None
    except Exception as exc:
        logger.error(f"Kraken microservice request failed on {image_path}: {type(exc).__name__}: {str(exc)[:300]}")
        return None