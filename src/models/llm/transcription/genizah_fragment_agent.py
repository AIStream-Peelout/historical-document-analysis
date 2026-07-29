"""
Cairo Genizah Transcription Agent
Core transcription logic for production and evaluation use
"""

import asyncio
import json
import os
import re
import ssl
from pathlib import Path
from typing import TypedDict, Optional, Dict, List
from urllib.parse import urlparse
import aiohttp
import certifi
from langgraph.graph import StateGraph, END
import google.generativeai as genai
from google.cloud import vision
import time
import dotenv
from PIL import Image

dotenv.load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

class AgentConfig:
    """Core agent configuration"""

    # API keys
    GOOGLE_VISION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # Models
    # NOTE: there is no "gemini-3.5-pro" — the API's Pro line currently tops
    # out at gemini-3.1-pro-preview (verify with genai.list_models()).
    GEMINI_FLASH_MODEL = "gemini-3.5-flash"
    GEMINI_PRO_MODEL = "gemini-3.1-pro-preview"
    GEMINI_ANALYSIS_MODEL = "gemini-3.5-flash"

    # Model selection
    USE_VISION_OCR = True
    USE_GEMINI_FLASH = True
    USE_GEMINI_PRO = True
    USE_ANALYSIS = True

    # Timeouts
    GEMINI_FLASH_TIMEOUT = 180   # 3 minutes
    # gemini-3.1-pro-preview is a heavy thinking model — dense sections
    # (rashi/tosafot) routinely exceed 300s, wasting all 3 retries.
    GEMINI_PRO_TIMEOUT   = 600   # 10 minutes

    # Output token budget.
    # gemini-3.5-pro is a thinking model — thinking tokens count against
    # this limit.  With 8 192 the model exhausts the budget mid-transcription on
    # large Gemara sections (finish_reason=2).  32 768 gives the thinking pass
    # ~10-15k tokens and still leaves room for a full section output.
    # NOTE: the deprecated google.generativeai SDK may not honour this correctly
    # on newer models — proper fix is migrating to google.genai.
    GEMINI_FLASH_MAX_OUTPUT_TOKENS = 16_384
    GEMINI_PRO_MAX_OUTPUT_TOKENS   = 32_768

    # Overridable for long benchmark runs: Gemini Pro 504s burn the full
    # timeout per attempt on pages/fragments it cannot finish, so run-once
    # benchmarks set GEMINI_MAX_ATTEMPTS=1 and report the failure as
    # coverage (project protocol) instead of paying the retry tax.
    MAX_RETRIES = int(os.getenv("GEMINI_MAX_ATTEMPTS", "3"))
    RETRY_DELAY = 10  # seconds

    # Image preprocessing
    MAX_IMAGE_PIXELS = 8_000_000  # ~3000x3000

    # Rate limiting
    DELAY_BETWEEN_DOCS = 3  # seconds
    KRAKEN_MODEL_PATH = os.getenv("KRAKEN_MODEL_PATH", "/Users/isaac1/Documents/historical-document-analysis/src/datasets/raw_data/cairo_genizah/custom_model_weights/MiDRASH_Gen_01.mlmodel")
    USE_KRAKEN = True  # Kraken is very slow and has no API rate limits, so disable by default
    KRAKEN_TIMEOUT = 120  # seconds per document

# ============================================================================
# Image Preprocessing
# ============================================================================

def prepare_image_for_gemini(image_path: str, max_pixels: int = None) -> str:
    """Optimize image for Gemini processing while preserving quality."""
    if max_pixels is None:
        max_pixels = AgentConfig.MAX_IMAGE_PIXELS

    img = Image.open(image_path)
    current_pixels = img.width * img.height

    print(f"    📐 Image: {img.width}x{img.height} ({current_pixels:,} pixels)")

    if current_pixels > max_pixels:
        ratio = (max_pixels / current_pixels) ** 0.5
        new_size = (int(img.width * ratio), int(img.height * ratio))

        print(f"    📐 Resizing to {new_size[0]}x{new_size[1]} ({new_size[0]*new_size[1]:,} pixels)")

        img = img.resize(new_size, Image.Resampling.LANCZOS)

        resized_path = str(Path(image_path).parent / f"prepared_{Path(image_path).name}")
        img.save(resized_path, quality=95, optimize=True)

        return resized_path

    return image_path


# ============================================================================
# State Definition
# ============================================================================

class TranscriptionState(TypedDict):
    """State object passed between LangGraph nodes during transcription pipeline."""
    # Input
    doc_id: str
    image_path: str
    catalog_metadata: dict
    ground_truth: Optional[str]

    # Model results
    vision_ocr_result: Optional[dict]
    gemini_flash_result: Optional[dict]
    gemini_pro_result: Optional[dict]
    kraken_result: Optional[dict]

    # Individual model metrics (for eval mode)
    vision_ocr_metrics: Optional[dict]
    gemini_flash_metrics: Optional[dict]
    gemini_pro_metrics: Optional[dict]
    kraken_metrics: Optional[dict]

    # Consensus
    all_results: list
    disagreements: list
    needs_review: bool
    final_transcription: str
    confidence_score: float
    consensus_strategy: str

    # Consensus metrics (for eval mode)
    consensus_metrics: Optional[dict]

    # Analysis
    analysis_result: Optional[dict]

    # Timing
    processing_time: float
    model_times: dict


# ============================================================================
# Robust Gemini Call with PROPER ASYNC HANDLING
# ============================================================================

async def call_gemini_with_retry(
    model_name: str,
    image_path: str,
    prompt: str,
    temperature: float = 0.1,
    max_retries: int = None,
    timeout: int = None
) -> Optional[str]:
    """Call Gemini API with robust retry and proper async timeout handling."""
    if max_retries is None:
        max_retries = AgentConfig.MAX_RETRIES

    if timeout is None:
        timeout = (
            AgentConfig.GEMINI_PRO_TIMEOUT
            if "pro" in model_name.lower()
            else AgentConfig.GEMINI_FLASH_TIMEOUT
        )

    genai.configure(api_key=AgentConfig.GEMINI_API_KEY)
    model = genai.GenerativeModel(model_name)

    # Resize image once before the retry loop
    prepared_image = prepare_image_for_gemini(image_path)

    # Upload image ONCE before the retry loop.
    # genai.upload_file is synchronous — run it in a thread so it doesn't
    # block the event loop during parallel model execution.
    print(f"    📤 Uploading image to Gemini Files API...")
    try:
        uploaded_file = await asyncio.to_thread(genai.upload_file, prepared_image)
    except Exception as e:
        print(f"    ❌ Upload failed: {type(e).__name__}: {str(e)[:200]}")
        return None

    for attempt in range(max_retries):
        try:
            print(f"    🔄 Attempt {attempt + 1}/{max_retries} (timeout: {timeout}s)...")

            max_tokens = (
                AgentConfig.GEMINI_PRO_MAX_OUTPUT_TOKENS
                if "pro" in model_name.lower()
                else AgentConfig.GEMINI_FLASH_MAX_OUTPUT_TOKENS
            )
            response = await asyncio.wait_for(
                model.generate_content_async(
                    [prompt, uploaded_file],
                    generation_config=genai.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                ),
                timeout=timeout,
            )

            if not response.candidates:
                raise ValueError("No candidates in response")

            candidate = response.candidates[0]
            finish_reason = getattr(candidate, "finish_reason", None)
            print(f"    📋 finish_reason={finish_reason}")

            # SAFETY block (finish_reason=3) — model won't change its mind on retry
            if finish_reason == 3:
                safety_info = (
                    ", ".join(
                        f"{r.category}:{r.probability}"
                        for r in candidate.safety_ratings
                    )
                    if hasattr(candidate, "safety_ratings")
                    else "no details"
                )
                print(f"    ⛔ SAFETY block ({safety_info}) — not retrying")
                return None

            # RECITATION block (finish_reason=4) — model recognised canonical text.
            # Try to salvage partial content before giving up; don't retry.
            if finish_reason == 4:
                partial = ""
                if candidate.content and candidate.content.parts:
                    partial = "".join(
                        p.text for p in candidate.content.parts
                        if hasattr(p, "text") and p.text
                    ).strip()
                if partial:
                    print(f"    ⚠️  RECITATION block — salvaged {len(partial)} chars")
                    return partial
                print(f"    ⛔ RECITATION block with no extractable content — not retrying")
                return None

            # Empty content for any other finish_reason — retry may help
            if not candidate.content or not candidate.content.parts:
                raise ValueError(f"Empty content (finish_reason={finish_reason})")

            text = "".join(
                p.text for p in candidate.content.parts
                if hasattr(p, "text") and p.text
            ).strip()

            if not text:
                raise ValueError(f"Empty text in content parts (finish_reason={finish_reason})")

            print(f"    ✅ Success — {len(text)} chars")
            return text

        except asyncio.TimeoutError:
            print(f"    ⏱️  Timeout after {timeout}s (attempt {attempt + 1})")
            if attempt < max_retries - 1:
                wait_time = AgentConfig.RETRY_DELAY * (attempt + 1)
                print(f"    ⏳ Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
            else:
                print(f"    ❌ All {max_retries} attempts timed out")
                return None

        except Exception as e:
            error_str = str(e)
            print(f"    ❌ {type(e).__name__}: {error_str[:200]}")

            if "429" in error_str or "rate limit" in error_str.lower():
                if attempt < max_retries - 1:
                    print(f"    ⏳ Rate limited — waiting 60s...")
                    await asyncio.sleep(60)
                    continue

            if "quota" in error_str.lower() or "exhausted" in error_str.lower():
                print(f"    ⛔ Quota exhausted — not retrying")
                return None

            if attempt < max_retries - 1:
                wait_time = AgentConfig.RETRY_DELAY * (attempt + 1)
                print(f"    ⏳ Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
            else:
                print(f"    ❌ All {max_retries} attempts failed")
                return None

    return None


async def call_gemini_text_only(
    model_name: str,
    prompt: str,
    temperature: float = 0.1,
    timeout: int = 60,
) -> Optional[str]:
    """Call Gemini with a text-only prompt — no image upload.

    Used for the OCR segmentation step where the input is already extracted
    text, not an image.  Keeps the same finish_reason handling as the image
    variant so RECITATION/SAFETY blocks are surfaced consistently.
    """
    genai.configure(api_key=AgentConfig.GEMINI_API_KEY)
    model = genai.GenerativeModel(model_name)

    try:
        max_tokens = (
            AgentConfig.GEMINI_PRO_MAX_OUTPUT_TOKENS
            if "pro" in model_name.lower()
            else AgentConfig.GEMINI_FLASH_MAX_OUTPUT_TOKENS
        )
        response = await asyncio.wait_for(
            model.generate_content_async(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        print(f"    ⏱️  Text-only timeout after {timeout}s ({model_name})")
        return None
    except Exception as e:
        print(f"    ❌ Text-only call failed: {type(e).__name__}: {str(e)[:200]}")
        return None

    if not response.candidates:
        return None

    candidate = response.candidates[0]
    finish_reason = getattr(candidate, "finish_reason", None)

    if finish_reason == 3:   # SAFETY
        print(f"    ⛔ Text-only SAFETY block ({model_name})")
        return None
    if finish_reason == 4:   # RECITATION — try partial salvage
        if candidate.content and candidate.content.parts:
            partial = "".join(
                p.text for p in candidate.content.parts
                if hasattr(p, "text") and p.text
            ).strip()
            if partial:
                return partial
        return None

    if not candidate.content or not candidate.content.parts:
        return None

    return "".join(
        p.text for p in candidate.content.parts
        if hasattr(p, "text") and p.text
    ).strip() or None


# ============================================================================
# Helper Functions
# ============================================================================

def _series_from_doc_id(doc_id: str) -> str:
    """Extract the collection series from a Genizah document ID.

    Examples: 'T-S NS J295' → 'T-S NS', 'ENA 2727.3' → 'ENA', 'T-S 12.721' → 'T-S'
    """
    s = doc_id.strip()
    if re.match(r"T-S\s+NS", s, re.IGNORECASE):
        return "T-S NS"
    if re.match(r"T-S", s, re.IGNORECASE):
        return "T-S"
    parts = s.split()
    return parts[0] if parts else "Unknown"


async def download_image(
    url_or_filename: str,
    doc_id: str,
    output_dir: Path,
    image_prefix: str = "https://storage.googleapis.com/cairo-genizah-es-json/images/"
) -> Path:
    """Download image from URL or locate local file."""
    url_or_filename = image_prefix + url_or_filename
    if url_or_filename.startswith("http"):
        filename = Path(urlparse(url_or_filename).path).name
        output_path = output_dir / doc_id / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            return output_path

        print(f"  ⬇ Downloading: {filename}")
        # Use certifi's CA bundle explicitly — macOS Python installs often
        # lack system certs, causing SSLCertVerificationError otherwise.
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        async with aiohttp.ClientSession() as session:
            async with session.get(url_or_filename, ssl=ssl_ctx) as response:
                response.raise_for_status()
                content = await response.read()
                output_path.write_bytes(content)
                return output_path
    else:
        local_path = output_dir / doc_id / url_or_filename
        return local_path


# ============================================================================
# LangGraph Nodes - Core Transcription Logic
# ============================================================================

async def vision_ocr_node(state: TranscriptionState) -> TranscriptionState:
    """Execute Google Vision OCR transcription."""
    if not AgentConfig.USE_VISION_OCR:
        print(f"  ⏭️  Vision OCR: Skipped")
        state['vision_ocr_result'] = None
        state['vision_ocr_metrics'] = None
        return state

    print(f"  🔍 Vision OCR...")
    start_time = time.time()

    def _run_vision_ocr(image_path: str):
        client = vision.ImageAnnotatorClient()
        with open(image_path, 'rb') as f:
            img = vision.Image(content=f.read())
        return client.text_detection(image=img)

    try:
        # Run synchronous Vision API call in a thread to avoid blocking the event loop
        response = await asyncio.to_thread(_run_vision_ocr, state['image_path'])

        text = response.text_annotations[0].description if response.text_annotations else ""
        elapsed = time.time() - start_time

        state['vision_ocr_result'] = {
            'text': text,
            'confidence': 0.85 if text else 0.0,
            'model': 'google_vision_ocr',
            'char_count': len(text),
            'processing_time': elapsed
        }
        state['model_times']['vision_ocr'] = elapsed
        print(f"    ✓ Extracted {len(text)} chars in {elapsed:.1f}s")

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"    ✗ Failed: {type(e).__name__}: {str(e)[:100]}")
        state['vision_ocr_result'] = None
        state['vision_ocr_metrics'] = None
        state['model_times']['vision_ocr'] = elapsed

    return state


async def gemini_flash_node(state: TranscriptionState) -> TranscriptionState:
    """Execute Gemini Flash transcription."""
    if not AgentConfig.USE_GEMINI_FLASH:
        print(f"  ⏭️  Gemini Flash: Skipped")
        state['gemini_flash_result'] = None
        state['gemini_flash_metrics'] = None
        return state

    print(f"  ⚡ Gemini Flash...")
    start_time = time.time()

    try:
        series = _series_from_doc_id(state['doc_id'])
        prompt = f"""This is a manuscript from the Cairo Genizah ({series} collection).
The text may be in Hebrew, Aramaic, or Judaeo-Arabic (Arabic written in Hebrew script).

Transcribe the text in this image exactly as written. Do not normalize or correct the text.
Preserve all vocalization marks (nikud) and line structure.
Mark damaged or unclear characters with [?].

Return ONLY the transcription with no commentary."""

        text = await call_gemini_with_retry(
            AgentConfig.GEMINI_FLASH_MODEL,
            state['image_path'],
            prompt,
            temperature=0.1
        )

        if text is None:
            raise ValueError("Failed to get transcription after retries")

        elapsed = time.time() - start_time

        state['gemini_flash_result'] = {
            'text': text,
            'confidence': 0.80,
            'model': 'gemini_3_flash',
            'char_count': len(text),
            'processing_time': elapsed
        }

        state['model_times']['gemini_flash'] = elapsed

        print(f"    ✓ Extracted {len(text)} chars in {elapsed:.1f}s")

    except Exception as e:
        elapsed = time.time() - start_time
        error_type = type(e).__name__
        error_msg = str(e)[:200]
        print(f"    ✗ Failed: {error_type}: {error_msg}")
        state['gemini_flash_result'] = None
        state['gemini_flash_metrics'] = None
        state['model_times']['gemini_flash'] = elapsed

    return state


async def kraken_node(state) -> dict:
    """Execute Kraken / MiDRASH transcription.

    Kraken is synchronous; we run it in a thread executor so it doesn't block
    the event loop and LangGraph timeouts remain effective.

    Note: The model is loaded per-call here for LangGraph compatibility.
    For batch evaluation, use talmud_eval.py which preloads the model once.
    """

    if not AgentConfig.USE_KRAKEN:
        print("  ⏭️  Kraken: Skipped")
        state["kraken_result"] = None
        state["kraken_metrics"] = None
        return state

    print("  🐙 Kraken / MiDRASH...")
    start_time = time.time()

    try:
        from src.models.ocr.kraken_transcriber import transcribe_with_kraken

        text = await transcribe_with_kraken(
            AgentConfig.KRAKEN_MODEL_PATH,
            state["image_path"],
            timeout=AgentConfig.KRAKEN_TIMEOUT,
        )

        if text is None:
            raise ValueError("Kraken returned None")

        elapsed = time.time() - start_time

        state["kraken_result"] = {
            "text": text,
            "confidence": None,  # Kraken doesn't expose a scalar confidence
            "model": "kraken_midrash_gen_01",
            "char_count": len(text),
            "processing_time": elapsed,
        }
        state["model_times"]["kraken"] = elapsed

        print(f"    ✓ Extracted {len(text)} chars in {elapsed:.1f}s")

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"    ✗ Kraken failed: {type(e).__name__}: {str(e)[:200]}")
        state["kraken_result"] = None
        state["kraken_metrics"] = None
        state["model_times"]["kraken"] = elapsed

    return state

async def gemini_pro_node(state: TranscriptionState) -> TranscriptionState:
    """Execute Gemini Pro transcription with FIXED timeout handling."""
    if not AgentConfig.USE_GEMINI_PRO:
        print(f"  ⏭️  Gemini Pro: Skipped")
        state['gemini_pro_result'] = None
        state['gemini_pro_metrics'] = None
        return state

    print(f"  🎯 Gemini Pro...")
    start_time = time.time()

    try:
        series = _series_from_doc_id(state['doc_id'])
        prompt = f"""This is a manuscript from the Cairo Genizah ({series} collection).
The text may be in Hebrew, Aramaic, or Judaeo-Arabic (Arabic written in Hebrew characters).

Step 1 — Identify the script type (square Hebrew, cursive Hebrew, Rashi script, etc.)
Step 2 — Transcribe methodically, right to left, top to bottom.
Step 3 — Preserve all details: vocalization marks (nikud), line breaks, marginal notes.
         Mark damaged or unclear characters with [?].
         Do NOT normalize or correct the text — every variant and scribal error is scholarly data.

Return ONLY the transcription."""

        text = await call_gemini_with_retry(
            AgentConfig.GEMINI_PRO_MODEL,
            state['image_path'],
            prompt,
            temperature=0.1
        )

        if text is None:
            raise ValueError("Failed to get transcription after retries")

        elapsed = time.time() - start_time

        state['gemini_pro_result'] = {
            'text': text,
            'confidence': 0.85,
            'model': 'gemini_3_pro',
            'char_count': len(text),
            'processing_time': elapsed
        }

        state['model_times']['gemini_pro'] = elapsed

        print(f"    ✓ Extracted {len(text)} chars in {elapsed:.1f}s")

    except Exception as e:
        elapsed = time.time() - start_time
        error_type = type(e).__name__
        error_msg = str(e)[:200]
        print(f"    ✗ Failed: {error_type}: {error_msg}")
        state['gemini_pro_result'] = None
        state['gemini_pro_metrics'] = None
        state['model_times']['gemini_pro'] = elapsed

    return state


async def consensus_node(state: TranscriptionState) -> TranscriptionState:
    """Compute consensus transcription from all model outputs."""
    print(f"  🤝 Computing consensus...")

    # Collect results
    results = []
    for key in ['vision_ocr_result', 'gemini_flash_result', 'gemini_pro_result']:
        if state[key] is not None:
            results.append(state[key])

    state['all_results'] = results

    if not results:
        state['final_transcription'] = ""
        state['confidence_score'] = 0.0
        state['needs_review'] = True
        state['consensus_strategy'] = "failed"
        state['disagreements'] = []
        return state

    # Analyze disagreements
    from difflib import SequenceMatcher
    disagreements = []
    for i, r1 in enumerate(results):
        for j, r2 in enumerate(results[i+1:], i+1):
            similarity = SequenceMatcher(None, r1['text'], r2['text']).ratio()
            if similarity < 0.90:
                disagreements.append({
                    'models': [r1['model'], r2['model']],
                    'similarity': similarity,
                })

    state['disagreements'] = disagreements

    # Consensus strategy - PREFER PRO for quality
    if state['gemini_pro_result']:
        state['final_transcription'] = state['gemini_pro_result']['text']
        state['confidence_score'] = 0.90
        state['consensus_strategy'] = "pro_preferred"

    elif state['gemini_flash_result']:
        state['final_transcription'] = state['gemini_flash_result']['text']
        state['confidence_score'] = 0.80
        state['consensus_strategy'] = "flash_fallback"

    elif state['vision_ocr_result']:
        state['final_transcription'] = state['vision_ocr_result']['text']
        state['confidence_score'] = 0.70
        state['consensus_strategy'] = "ocr_fallback"

    else:
        state['final_transcription'] = results[0]['text']
        state['confidence_score'] = 0.50
        state['consensus_strategy'] = "fallback_first"

    state['needs_review'] = len(disagreements) > 0

    print(f"    ✓ Strategy: {state['consensus_strategy']}, "
          f"Confidence: {state['confidence_score']:.2f}")

    return state


async def analysis_node(state: TranscriptionState) -> TranscriptionState:
    """Analyze all transcriptions using Gemini for coherence and translation."""
    if not AgentConfig.USE_ANALYSIS:
        print(f"  ⏭️  Analysis: Skipped")
        state['analysis_result'] = None
        return state

    print(f"  🔬 Analyzing transcriptions...")
    start_time = time.time()

    try:
        genai.configure(api_key=AgentConfig.GEMINI_API_KEY)
        model = genai.GenerativeModel(AgentConfig.GEMINI_ANALYSIS_MODEL)

        # Gather all available transcriptions
        transcriptions_summary = []

        if state.get('vision_ocr_result'):
            transcriptions_summary.append(f"""
**Google Vision OCR**:
{state['vision_ocr_result']['text'][:1000]}...
""")

        if state.get('gemini_flash_result'):
            transcriptions_summary.append(f"""
**Gemini Flash**:
{state['gemini_flash_result']['text'][:1000]}...
""")

        if state.get('gemini_pro_result'):
            transcriptions_summary.append(f"""
**Gemini Pro**:
{state['gemini_pro_result']['text'][:1000]}...
""")

        # Construct analysis prompt
        prompt = f"""You are a Hebrew manuscript expert analyzing transcription quality for the Cairo Genizah project.

**CATALOG INFORMATION:**
Document ID: {state['doc_id']}
Description: {state['catalog_metadata'].get('description', 'No description available')}
Date: {state['catalog_metadata'].get('date', 'Unknown')}
Type: {state['catalog_metadata'].get('type', 'Unknown')}

**TRANSCRIPTION ATTEMPTS:**
{chr(10).join(transcriptions_summary)}

**CONSENSUS TRANSCRIPTION** (Strategy: {state['consensus_strategy']}):
{state['final_transcription'][:1000]}...

**YOUR TASK:**
Analyze these transcriptions and provide:

1. **Coherence Assessment**: Do the transcriptions make sense as Hebrew text? Are there obvious OCR errors, garbled text, or nonsensical character sequences?

2. **Catalog Alignment**: How well does the transcribed content align with the catalog description? Does it match the expected document type (legal, liturgical, biblical, etc.)?

3. **Recommended Transcription**: Which model's output appears most accurate and why?

4. **Content Summary**: Provide a 2-3 sentence English summary of what this document contains.

5. **Key Observations**: Note any interesting textual variants, damage patterns, or paleographic features.

6. **Translation Sample**: Translate the first 100-200 characters of the best transcription to English.

Please respond in JSON format:
{{
    "coherence_assessment": "...",
    "catalog_alignment": "...",
    "recommended_transcription": "vision_ocr|gemini_flash|gemini_pro|consensus",
    "confidence_reasoning": "...",
    "content_summary": "...",
    "key_observations": "...",
    "translation_sample": "..."
}}"""

        # Use async API here too
        response = await model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(temperature=0.3),
        )

        # Parse JSON response
        response_text = response.text.strip()

        # Extract JSON from markdown code blocks if present
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)

        analysis_data = json.loads(response_text)

        elapsed = time.time() - start_time

        state['analysis_result'] = {
            'summary': analysis_data.get('content_summary', ''),
            'translation': analysis_data.get('translation_sample', ''),
            'coherence_assessment': analysis_data.get('coherence_assessment', ''),
            'recommended_transcription': analysis_data.get('recommended_transcription', 'consensus'),
            'confidence_reasoning': analysis_data.get('confidence_reasoning', ''),
            'catalog_alignment': analysis_data.get('catalog_alignment', ''),
            'key_observations': analysis_data.get('key_observations', ''),
            'processing_time': elapsed
        }

        print(f"    ✓ Analysis complete in {elapsed:.2f}s")
        print(f"    🎯 Recommends: {state['analysis_result']['recommended_transcription']}")

    except Exception as e:
        elapsed = time.time() - start_time
        error_type = type(e).__name__
        error_msg = str(e)[:200]
        print(f"    ✗ Analysis failed: {error_type}: {error_msg}")

        state['analysis_result'] = {
            'summary': 'Analysis failed',
            'translation': '',
            'coherence_assessment': f'Error: {error_type}',
            'recommended_transcription': 'consensus',
            'confidence_reasoning': error_msg,
            'catalog_alignment': 'Unable to assess',
            'key_observations': '',
            'processing_time': elapsed
        }

    return state


# ============================================================================
# Build Graph - For Production and Evaluation Use
# ============================================================================

def build_transcription_graph():
    """Build LangGraph workflow for transcription.

    Sequential execution to avoid rate limits:
    - OCR and Flash run in parallel (fast, high limits)
    - Pro runs sequentially after (slower, lower limits)
    """

    async def parallel_models_node(state: TranscriptionState) -> TranscriptionState:
        """Execute models with Pro running sequentially to avoid rate limits."""

        # Run OCR and Flash in parallel (they're fast and have high limits)
        parallel_tasks = []
        if AgentConfig.USE_VISION_OCR:
            parallel_tasks.append(vision_ocr_node(dict(state)))
        if AgentConfig.USE_GEMINI_FLASH:
            parallel_tasks.append(gemini_flash_node(dict(state)))

        if parallel_tasks:
            results = await asyncio.gather(*parallel_tasks)

            for result in results:
                if 'vision_ocr_result' in result and result['vision_ocr_result'] is not None:
                    state['vision_ocr_result'] = result['vision_ocr_result']
                    state['vision_ocr_metrics'] = result.get('vision_ocr_metrics')
                    if 'vision_ocr' in result.get('model_times', {}):
                        state['model_times']['vision_ocr'] = result['model_times']['vision_ocr']

                if 'gemini_flash_result' in result and result['gemini_flash_result'] is not None:
                    state['gemini_flash_result'] = result['gemini_flash_result']
                    state['gemini_flash_metrics'] = result.get('gemini_flash_metrics')
                    if 'gemini_flash' in result.get('model_times', {}):
                        state['model_times']['gemini_flash'] = result['model_times']['gemini_flash']

        # Run Pro SEQUENTIALLY (avoid rate limits, allow full timeout)
        if AgentConfig.USE_GEMINI_PRO:
            # Small delay before Pro to space out requests
            await asyncio.sleep(3)
            pro_result = await gemini_pro_node(state)

            if 'gemini_pro_result' in pro_result and pro_result['gemini_pro_result'] is not None:
                state['gemini_pro_result'] = pro_result['gemini_pro_result']
                state['gemini_pro_metrics'] = pro_result.get('gemini_pro_metrics')
                if 'gemini_pro' in pro_result.get('model_times', {}):
                    state['model_times']['gemini_pro'] = pro_result['model_times']['gemini_pro']
        if AgentConfig.USE_KRAKEN:
            await asyncio.sleep(2)  # brief pause between API systems
            kraken_result_state = await kraken_node(state)
            if kraken_result_state.get("kraken_result") is not None:
                state["kraken_result"] = kraken_result_state["kraken_result"]
                state["kraken_metrics"] = kraken_result_state.get("kraken_metrics")
                if "kraken" in kraken_result_state.get("model_times", {}):
                    state["model_times"]["kraken"] = kraken_result_state["model_times"]["kraken"]

        return state

    workflow = StateGraph(TranscriptionState)

    workflow.add_node("parallel_models", parallel_models_node)
    workflow.add_node("consensus", consensus_node)
    workflow.add_node("analysis", analysis_node)

    workflow.set_entry_point("parallel_models")
    workflow.add_edge("parallel_models", "consensus")
    workflow.add_edge("consensus", "analysis")
    workflow.add_edge("analysis", END)

    return workflow.compile()


# ============================================================================
# Main Processing Function
# ============================================================================

async def transcribe_document(
    doc_id: str,
    metadata: dict,
    image_path: Path,
    ground_truth: Optional[str] = None
) -> TranscriptionState:
    """Process a single document through the transcription pipeline.

    Args:
        doc_id: Document identifier
        metadata: Catalog metadata for the document
        image_path: Path to the document image
        ground_truth: Optional ground truth transcription (for evaluation)

    Returns:
        Final state with all transcription results
    """
    print(f"\n📄 {doc_id}")
    print(f"   {metadata.get('description', '')[:80]}...")

    # Initialize state
    start_time = time.time()

    initial_state = TranscriptionState(
        doc_id=doc_id,
        image_path=str(image_path),
        catalog_metadata=metadata,
        ground_truth=ground_truth,
        vision_ocr_result=None,
        gemini_flash_result=None,
        gemini_pro_result=None,
        vision_ocr_metrics=None,
        gemini_flash_metrics=None,
        gemini_pro_metrics=None,
        all_results=[],
        disagreements=[],
        needs_review=False,
        final_transcription="",
        confidence_score=0.0,
        consensus_strategy="",
        consensus_metrics=None,
        analysis_result=None,
        processing_time=0.0,
        model_times={}
    )

    # Build and execute graph
    graph = build_transcription_graph()
    final_state = await graph.ainvoke(initial_state)
    final_state['processing_time'] = time.time() - start_time

    print(f"  ✅ Complete in {final_state['processing_time']:.2f}s")

    return final_state