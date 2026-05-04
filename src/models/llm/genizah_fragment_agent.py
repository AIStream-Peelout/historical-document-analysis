"""
Cairo Genizah Transcription Agent
Core transcription logic for production and evaluation use
"""

import asyncio
import json
import os
import re
from pathlib import Path
from typing import TypedDict, Optional, Dict, List
from urllib.parse import urlparse
import aiohttp
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
    GEMINI_FLASH_MODEL = "gemini-3-flash-preview"
    GEMINI_PRO_MODEL = "gemini-3.1-pro-preview"
    GEMINI_ANALYSIS_MODEL = "gemini-2.0-flash-exp"

    # Model selection
    USE_VISION_OCR = True
    USE_GEMINI_FLASH = True
    USE_GEMINI_PRO = True
    USE_ANALYSIS = True

    # Timeouts - Pro needs more time but should NOT take 15 minutes
    GEMINI_FLASH_TIMEOUT = 180  # 3 minutes
    GEMINI_PRO_TIMEOUT = 300    # 5 minutes (should be enough)
    MAX_RETRIES = 3
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
    """Call Gemini API with robust retry logic and PROPER async timeout handling.

    KEY FIX: Uses async API methods (generate_content_async) so timeout actually works!
    """
    if max_retries is None:
        max_retries = AgentConfig.MAX_RETRIES

    if timeout is None:
        # Default timeout based on model
        if "pro" in model_name.lower():
            timeout = AgentConfig.GEMINI_PRO_TIMEOUT
        else:
            timeout = AgentConfig.GEMINI_FLASH_TIMEOUT

    genai.configure(api_key=AgentConfig.GEMINI_API_KEY)
    model = genai.GenerativeModel(model_name)

    # Prepare image
    prepared_image = prepare_image_for_gemini(image_path)

    for attempt in range(max_retries):
        try:
            print(f"    🔄 Attempt {attempt + 1}/{max_retries}")

            # Upload file - this is synchronous but fast, so it's ok
            print(f"    📤 Uploading image...")
            uploaded_file = genai.upload_file(prepared_image)

            # Small delay after upload
            await asyncio.sleep(2)

            # Generate with PROPER async timeout
            print(f"    🧠 Generating transcription (timeout: {timeout}s)...")

            # THE KEY FIX: Use the ASYNC version of generate_content!
            # This allows asyncio.wait_for to actually interrupt it
            response = await asyncio.wait_for(
                model.generate_content_async(
                    [prompt, uploaded_file],
                    generation_config=genai.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=8192,  # Ensure we don't get truncated
                    ),
                ),
                timeout=timeout
            )

            # Validate response
            if not response.candidates:
                raise ValueError("No candidates returned")

            candidate = response.candidates[0]

            # Check for safety blocks
            if candidate.finish_reason == 3:  # SAFETY
                safety_info = "\n".join([
                    f"        {rating.category}: {rating.probability}"
                    for rating in candidate.safety_ratings
                ]) if hasattr(candidate, 'safety_ratings') else "No safety info"
                raise ValueError(f"Blocked by safety filters:\n{safety_info}")

            # Check for content
            if not candidate.content or not candidate.content.parts:
                raise ValueError(f"No content (finish_reason={candidate.finish_reason})")

            first_part = candidate.content.parts[0]
            if not hasattr(first_part, 'text') or not first_part.text:
                raise ValueError("Content part has no text")

            text = first_part.text.strip()
            print(f"    ✅ Success! Extracted {len(text)} chars")
            return text

        except asyncio.TimeoutError:
            print(f"    ⏱️  Timeout after {timeout}s on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                wait_time = AgentConfig.RETRY_DELAY * (attempt + 1)
                print(f"    ⏳ Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
            else:
                print(f"    ❌ Failed after {max_retries} timeout attempts")
                return None

        except Exception as e:
            error_str = str(e)
            print(f"    ❌ Error: {type(e).__name__}: {error_str[:200]}")

            # Check for rate limit
            if "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower():
                if attempt < max_retries - 1:
                    wait_time = 60  # Wait 1 minute for rate limits
                    print(f"    ⏳ Rate limited, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue

            # Check for quota exhausted
            if "quota" in error_str.lower() or "exhausted" in error_str.lower():
                print(f"    ⛔ Quota exhausted - cannot retry")
                return None

            # Other errors - retry with delay
            if attempt < max_retries - 1:
                wait_time = AgentConfig.RETRY_DELAY * (attempt + 1)
                print(f"    ⏳ Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
            else:
                print(f"    ❌ Failed after {max_retries} attempts")
                return None

    return None


# ============================================================================
# Helper Functions
# ============================================================================

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
        async with aiohttp.ClientSession() as session:
            async with session.get(url_or_filename) as response:
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

    try:
        client = vision.ImageAnnotatorClient()

        with open(state['image_path'], 'rb') as f:
            image = vision.Image(content=f.read())

        response = client.text_detection(image=image)

        if response.text_annotations:
            text = response.text_annotations[0].description
            confidence = 0.85
        else:
            text = ""
            confidence = 0.0

        elapsed = time.time() - start_time

        state['vision_ocr_result'] = {
            'text': text,
            'confidence': confidence,
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
        catalog_hint = state['catalog_metadata'].get('description', '')

        prompt = f"""Transcribe this Hebrew manuscript image.

Catalog context: {catalog_hint}

CRITICAL INSTRUCTIONS:
1. Transcribe EXACTLY what you see - character by character
2. Do NOT "correct" text to match expected biblical versions
3. Textual variants are valuable - preserve them exactly
4. Include all vocalization marks if present
5. Preserve spacing and line structure

Return ONLY the Hebrew transcription with no commentary."""

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
        catalog_hint = state['catalog_metadata'].get('description', '')

        prompt = f"""The following is an image of a document from the Cairo Genizah. The text may be in Hebrew, Arabic, Aramaic, Judeo-Arabic or some combination thereof.

Catalog context: {catalog_hint}

CRITICAL INSTRUCTIONS:
1. Transcribe EXACTLY what you see - character by character
2. Pay careful attention to:
   - Vocalization marks (nikud)
   - Text direction and layout
   - Damaged or faded sections (mark with [?] if unclear)
   - Marginal notes and annotations
3. Do NOT normalize or "correct" the text to match known versions
4. Textual variants and scribal errors are valuable - preserve them exactly
5. Preserve line breaks and spatial layout where significant

Think step-by-step:
1. First, identify the script(s) and language(s) present
2. Then, transcribe methodically from right to left, top to bottom
3. For ambiguous characters, note alternatives in brackets

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