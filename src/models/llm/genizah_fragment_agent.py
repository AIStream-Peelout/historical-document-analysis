"""
Cairo Genizah Transcription Evaluation Pipeline
HIGH-QUALITY MODE: Optimized for accuracy over speed
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
from google.api_core import retry as retry_module
import wandb
from difflib import SequenceMatcher
import time
from collections import defaultdict
import dotenv
from PIL import Image

dotenv.load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Pipeline configuration - QUALITY MODE"""
    catalog_path = "/Users/isaac1/Documents/historical-document-analysis/src/datasets/raw_data/merged_princeton_friedberger_all_documents_with_transcriptions.json"
    IMAGES_DIR = Path("./genizah_images")
    RESULTS_DIR = Path("./transcription_results")
    RAW_OUTPUTS_DIR = Path("./transcription_raw_outputs")
    CATALOG_PATH = Path(catalog_path)

    # API keys
    GOOGLE_VISION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # W&B
    WANDB_PROJECT = "cairo-genizah-transcription"
    WANDB_ENTITY = os.getenv("WANDB_ENTITY")

    # Models
    GEMINI_FLASH_MODEL = "gemini-3-flash-preview"
    GEMINI_PRO_MODEL = "gemini-3-pro-preview"
    GEMINI_ANALYSIS_MODEL = "gemini-2.0-flash-exp"

    # Model selection - ALL ENABLED for quality
    USE_VISION_OCR = True
    USE_GEMINI_FLASH = True
    USE_GEMINI_PRO = True
    USE_ANALYSIS = True

    # Evaluation
    MIN_CONFIDENCE_THRESHOLD = 0.7

    # Timeouts - GENEROUS for complex manuscripts
    GEMINI_TIMEOUT = 900  # 15 minutes per transcription
    MAX_RETRIES = 3
    RETRY_DELAY = 10  # seconds

    # Image preprocessing
    MAX_IMAGE_PIXELS = 8_000_000  # ~3000x3000 - balance quality vs processing time

    # Rate limiting (stay under 25 RPM for Pro)
    DELAY_BETWEEN_DOCS = 3  # seconds

    # Incremental results file
    INCREMENTAL_RESULTS_FILE = Path("./transcription_results/incremental_results.jsonl")


# ============================================================================
# Image Preprocessing
# ============================================================================

def prepare_image_for_gemini(image_path: str, max_pixels: int = None) -> str:
    """Optimize image for Gemini processing while preserving quality.

    :param image_path: Path to original image
    :type image_path: str
    :param max_pixels: Maximum total pixels (width * height)
    :type max_pixels: int
    :return: Path to prepared image
    :rtype: str
    """
    if max_pixels is None:
        max_pixels = Config.MAX_IMAGE_PIXELS

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
# Metrics Calculation
# ============================================================================

def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate"""
    if not reference:
        return 1.0 if hypothesis else 0.0

    matcher = SequenceMatcher(None, reference, hypothesis)
    operations = matcher.get_opcodes()
    errors = sum(max(j2-j1, i2-i1) for op, i1, i2, j1, j2 in operations if op != 'equal')

    return errors / len(reference)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate"""
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if not ref_words:
        return 1.0 if hyp_words else 0.0

    matcher = SequenceMatcher(None, ref_words, hyp_words)
    operations = matcher.get_opcodes()
    errors = sum(max(j2-j1, i2-i1) for op, i1, i2, j1, j2 in operations if op != 'equal')

    return errors / len(ref_words)


def calculate_similarity(reference: str, hypothesis: str) -> float:
    """Calculate character-level similarity ratio"""
    return SequenceMatcher(None, reference, hypothesis).ratio()


def extract_ground_truth(transcriptions) -> str:
    """Extract ground truth text from transcription data"""
    if not transcriptions:
        return ""

    # Format 1: Dictionary with string keys
    if isinstance(transcriptions, dict):
        sorted_keys = sorted(transcriptions.keys(), key=lambda x: int(x) if x.isdigit() else 0)
        lines = [transcriptions[k] for k in sorted_keys]
        return '\n'.join(lines)

    # Format 2: List of transcription objects
    if isinstance(transcriptions, list):
        if not transcriptions:
            return ""

        first_transcription = transcriptions[0]

        if isinstance(first_transcription, dict) and 'lines' in first_transcription:
            lines_dict = first_transcription['lines']
            sorted_keys = sorted(lines_dict.keys(), key=lambda x: int(x) if x.isdigit() else 0)
            lines = [lines_dict[k] for k in sorted_keys]
            return '\n'.join(lines)

    return ""


# ============================================================================
# Output Saving
# ============================================================================

def save_raw_output(doc_id: str, model_name: str, text: str, ground_truth: str = None):
    """Save raw transcription output to file"""
    output_dir = Config.RAW_OUTPUTS_DIR / doc_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model output
    output_file = output_dir / f"{model_name}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)

    # Save ground truth once
    if ground_truth and model_name == "ground_truth":
        gt_file = output_dir / "ground_truth.txt"
        with open(gt_file, 'w', encoding='utf-8') as f:
            f.write(ground_truth)


def save_incremental_result(result: dict):
    """Append result to incremental JSONL file"""
    Config.INCREMENTAL_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(Config.INCREMENTAL_RESULTS_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False, default=str) + '\n')

    print(f"  💾 Saved to: {Config.INCREMENTAL_RESULTS_FILE}")


def create_comparison_html(doc_id: str, ground_truth: str,
                          ocr_text: str, flash_text: str, pro_text: str,
                          consensus_text: str, consensus_strategy: str) -> str:
    """Create HTML comparison of all transcriptions"""

    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .comparison {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
            .section {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
            .section h3 {{ margin-top: 0; color: #666; }}
            .text {{ white-space: pre-wrap; font-family: "Courier New", monospace; 
                     direction: rtl; text-align: right; line-height: 1.6; }}
            .consensus {{ background-color: #f0f8ff; }}
            .ground-truth {{ background-color: #f0fff0; }}
        </style>
    </head>
    <body>
        <h1>Transcription Comparison: {doc_id}</h1>
        <p><strong>Consensus Strategy:</strong> {consensus_strategy}</p>
        
        <div class="comparison">
            <div class="section ground-truth">
                <h3>Ground Truth ({len(ground_truth)} chars)</h3>
                <div class="text">{ground_truth[:500]}...</div>
            </div>
            
            <div class="section consensus">
                <h3>Consensus - {consensus_strategy} ({len(consensus_text)} chars)</h3>
                <div class="text">{consensus_text[:500]}...</div>
            </div>
            
            <div class="section">
                <h3>Google Vision OCR ({len(ocr_text)} chars)</h3>
                <div class="text">{ocr_text[:500]}...</div>
            </div>
            
            <div class="section">
                <h3>Gemini Flash ({len(flash_text)} chars)</h3>
                <div class="text">{flash_text[:500]}...</div>
            </div>
            
            <div class="section">
                <h3>Gemini Pro ({len(pro_text)} chars)</h3>
                <div class="text">{pro_text[:500]}...</div>
            </div>
        </div>
    </body>
    </html>
    """

    return html


# ============================================================================
# State Definition
# ============================================================================

class TranscriptionState(TypedDict):
    """State object passed between LangGraph nodes during transcription pipeline."""
    # Input
    doc_id: str
    image_path: str
    catalog_metadata: dict
    ground_truth: str

    # Model results
    vision_ocr_result: Optional[dict]
    gemini_flash_result: Optional[dict]
    gemini_pro_result: Optional[dict]

    # Individual model metrics
    vision_ocr_metrics: Optional[dict]
    gemini_flash_metrics: Optional[dict]
    gemini_pro_metrics: Optional[dict]

    # Consensus
    all_results: list
    disagreements: list
    needs_review: bool
    final_transcription: str
    confidence_score: float
    consensus_strategy: str

    # Consensus metrics
    consensus_metrics: Optional[dict]

    # Analysis
    analysis_result: Optional[dict]

    # Timing
    processing_time: float
    model_times: dict


# ============================================================================
# Helper Functions
# ============================================================================

async def download_image(url_or_filename: str, doc_id: str, output_dir: Path, image_prefix="https://storage.googleapis.com/cairo-genizah-es-json/images/") -> Path:
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


def has_ground_truth(metadata: dict) -> bool:
    """Check if document has valid ground truth transcription."""
    transcriptions = metadata.get('transcriptions', [])
    if not transcriptions:
        return False

    ground_truth = extract_ground_truth(transcriptions)
    return len(ground_truth.strip()) > 10


def evaluate_transcription(ground_truth: str, hypothesis: str, model_name: str) -> dict:
    """Calculate all evaluation metrics for a transcription."""
    # Clean whitespace for comparison
    gt_clean = ' '.join(ground_truth.split())
    hyp_clean = ' '.join(hypothesis.split())

    metrics = {
        'model': model_name,
        'cer': calculate_cer(gt_clean, hyp_clean),
        'wer': calculate_wer(gt_clean, hyp_clean),
        'similarity': calculate_similarity(gt_clean, hyp_clean),
        'exact_match': gt_clean == hyp_clean,
        'char_count': len(hypothesis),
        'gt_char_count': len(ground_truth),
        'char_diff': abs(len(hypothesis) - len(ground_truth))
    }

    return metrics


# ============================================================================
# Robust Gemini Call with Retry Logic
# ============================================================================

async def call_gemini_with_retry(
    model_name: str,
    image_path: str,
    prompt: str,
    temperature: float = 0.1,
    max_retries: int = None
) -> Optional[str]:
    """Call Gemini API with robust retry logic and timeout handling.

    :param model_name: Gemini model identifier
    :param image_path: Path to image file
    :param prompt: Transcription prompt
    :param temperature: Generation temperature
    :param max_retries: Maximum retry attempts
    :return: Transcribed text or None if failed
    """
    if max_retries is None:
        max_retries = Config.MAX_RETRIES

    genai.configure(api_key=Config.GEMINI_API_KEY)
    model = genai.GenerativeModel(model_name)

    # Prepare image
    prepared_image = prepare_image_for_gemini(image_path)

    for attempt in range(max_retries):
        try:
            print(f"    🔄 Attempt {attempt + 1}/{max_retries}")

            # Upload file
            print(f"    📤 Uploading image...")
            uploaded_file = genai.upload_file(prepared_image)

            # Small delay after upload
            await asyncio.sleep(2)

            # Generate with VERY generous timeout using asyncio
            print(f"    🧠 Generating transcription (timeout: {Config.GEMINI_TIMEOUT}s)...")

            async def _generate():
                return model.generate_content(
                    [prompt, uploaded_file],
                    generation_config=genai.GenerationConfig(temperature=temperature),
                )

            # Use asyncio.wait_for for timeout control
            response = await asyncio.wait_for(_generate(), timeout=Config.GEMINI_TIMEOUT)

            # Validate response
            if not response.candidates:
                raise ValueError("No candidates returned")

            candidate = response.candidates[0]

            if candidate.finish_reason == 3:  # SAFETY
                safety_info = "\n".join([
                    f"        {rating.category}: {rating.probability}"
                    for rating in candidate.safety_ratings
                ]) if hasattr(candidate, 'safety_ratings') else "No safety info"
                raise ValueError(f"Blocked by safety filters:\n{safety_info}")

            if not candidate.content or not candidate.content.parts:
                raise ValueError(f"No content (finish_reason={candidate.finish_reason})")

            first_part = candidate.content.parts[0]
            if not hasattr(first_part, 'text') or not first_part.text:
                raise ValueError("Content part has no text")

            text = first_part.text.strip()
            print(f"    ✅ Success! Extracted {len(text)} chars")
            return text

        except asyncio.TimeoutError:
            print(f"    ⏱️  Timeout after {Config.GEMINI_TIMEOUT}s")
            if attempt < max_retries - 1:
                wait_time = Config.RETRY_DELAY * (attempt + 1)
                print(f"    ⏳ Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
            else:
                print(f"    ❌ Failed after {max_retries} attempts")
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

            # Other errors - retry with delay
            if attempt < max_retries - 1:
                wait_time = Config.RETRY_DELAY * (attempt + 1)
                print(f"    ⏳ Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
            else:
                print(f"    ❌ Failed after {max_retries} attempts")
                return None

    return None


# ============================================================================
# LangGraph Nodes
# ============================================================================

async def vision_ocr_node(state: TranscriptionState) -> TranscriptionState:
    """Execute Google Vision OCR transcription and evaluate results."""
    if not Config.USE_VISION_OCR:
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

        # Save raw output
        save_raw_output(state['doc_id'], 'vision_ocr', text)

        state['vision_ocr_result'] = {
            'text': text,
            'confidence': confidence,
            'model': 'google_vision_ocr',
            'char_count': len(text),
            'processing_time': elapsed
        }

        # Evaluate against ground truth
        if state['ground_truth']:
            state['vision_ocr_metrics'] = evaluate_transcription(
                state['ground_truth'],
                text,
                'google_vision_ocr'
            )
            state['vision_ocr_metrics']['processing_time'] = elapsed

            print(f"    ✓ CER: {state['vision_ocr_metrics']['cer']:.3f}, "
                  f"Similarity: {state['vision_ocr_metrics']['similarity']:.3f}")

        state['model_times']['vision_ocr'] = elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"    ✗ Failed: {type(e).__name__}: {str(e)[:100]}")
        state['vision_ocr_result'] = None
        state['vision_ocr_metrics'] = None
        state['model_times']['vision_ocr'] = elapsed

    return state


async def gemini_flash_node(state: TranscriptionState) -> TranscriptionState:
    """Execute Gemini Flash transcription and evaluate results."""
    if not Config.USE_GEMINI_FLASH:
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
            Config.GEMINI_FLASH_MODEL,
            state['image_path'],
            prompt,
            temperature=0.1
        )

        if text is None:
            raise ValueError("Failed to get transcription after retries")

        elapsed = time.time() - start_time

        # Save raw output
        save_raw_output(state['doc_id'], 'gemini_flash', text)

        state['gemini_flash_result'] = {
            'text': text,
            'confidence': 0.80,
            'model': 'gemini_3_flash',
            'char_count': len(text),
            'processing_time': elapsed
        }

        # Evaluate
        if state['ground_truth']:
            state['gemini_flash_metrics'] = evaluate_transcription(
                state['ground_truth'],
                text,
                'gemini_3_flash'
            )
            state['gemini_flash_metrics']['processing_time'] = elapsed

            print(f"    ✓ CER: {state['gemini_flash_metrics']['cer']:.3f}, "
                  f"Similarity: {state['gemini_flash_metrics']['similarity']:.3f}")

        state['model_times']['gemini_flash'] = elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        error_type = type(e).__name__
        error_msg = str(e)[:200]
        print(f"    ✗ Failed: {error_type}: {error_msg}")
        state['gemini_flash_result'] = None
        state['gemini_flash_metrics'] = None
        state['model_times']['gemini_flash'] = elapsed

    return state


async def gemini_pro_node(state: TranscriptionState) -> TranscriptionState:
    """Execute Gemini Pro transcription and evaluate results."""
    if not Config.USE_GEMINI_PRO:
        print(f"  ⏭️  Gemini Pro: Skipped (disabled)")
        state['gemini_pro_result'] = None
        state['gemini_pro_metrics'] = None
        return state

    print(f"  🎯 Gemini Pro (HIGH QUALITY MODE)...")
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
            Config.GEMINI_PRO_MODEL,
            state['image_path'],
            prompt,
            temperature=0.1
        )

        if text is None:
            raise ValueError("Failed to get transcription after retries")

        elapsed = time.time() - start_time

        # Save raw output
        save_raw_output(state['doc_id'], 'gemini_pro', text)

        state['gemini_pro_result'] = {
            'text': text,
            'confidence': 0.85,
            'model': 'gemini_3_pro',
            'char_count': len(text),
            'processing_time': elapsed
        }

        # Evaluate
        if state['ground_truth']:
            state['gemini_pro_metrics'] = evaluate_transcription(
                state['ground_truth'],
                text,
                'gemini_3_pro'
            )
            state['gemini_pro_metrics']['processing_time'] = elapsed

            print(f"    ✓ CER: {state['gemini_pro_metrics']['cer']:.3f}, "
                  f"Similarity: {state['gemini_pro_metrics']['similarity']:.3f}")

        state['model_times']['gemini_pro'] = elapsed

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

    # Save consensus output
    save_raw_output(state['doc_id'], 'consensus', state['final_transcription'])

    # Evaluate consensus
    if state['ground_truth']:
        state['consensus_metrics'] = evaluate_transcription(
            state['ground_truth'],
            state['final_transcription'],
            f"consensus_{state['consensus_strategy']}"
        )

        print(f"    ✓ Consensus CER: {state['consensus_metrics']['cer']:.3f}, "
              f"Strategy: {state['consensus_strategy']}")

    state['needs_review'] = len(disagreements) > 0

    return state


async def analysis_node(state: TranscriptionState) -> TranscriptionState:
    """Analyze all transcriptions using Gemini for coherence and translation."""
    if not Config.USE_ANALYSIS:
        print(f"  ⏭️  Analysis: Skipped")
        state['analysis_result'] = None
        return state

    print(f"  🔬 Analyzing transcriptions...")
    start_time = time.time()

    try:
        genai.configure(api_key=Config.GEMINI_API_KEY)
        model = genai.GenerativeModel(Config.GEMINI_ANALYSIS_MODEL)

        # Gather all available transcriptions
        transcriptions_summary = []

        if state.get('vision_ocr_result'):
            cer = state.get('vision_ocr_metrics', {}).get('cer', 'N/A')
            cer_str = f"{cer:.3f}" if isinstance(cer, (int, float)) else cer
            transcriptions_summary.append(f"""
**Google Vision OCR** (CER: {cer_str}):
{state['vision_ocr_result']['text'][:1000]}...
""")

        if state.get('gemini_flash_result'):
            cer = state.get('gemini_flash_metrics', {}).get('cer', 'N/A')
            cer_str = f"{cer:.3f}" if isinstance(cer, (int, float)) else cer
            transcriptions_summary.append(f"""
**Gemini Flash** (CER: {cer_str}):
{state['gemini_flash_result']['text'][:1000]}...
""")

        if state.get('gemini_pro_result'):
            cer = state.get('gemini_pro_metrics', {}).get('cer', 'N/A')
            cer_str = f"{cer:.3f}" if isinstance(cer, (int, float)) else cer
            transcriptions_summary.append(f"""
**Gemini Pro** (CER: {cer_str}):
{state['gemini_pro_result']['text'][:1000]}...
""")

        # Construct rich analysis prompt
        prompt = f"""You are a Hebrew manuscript expert analyzing transcription quality for the Cairo Genizah project.

**CATALOG INFORMATION:**
Document ID: {state['doc_id']}
Description: {state['catalog_metadata'].get('description', 'No description available')}
Date: {state['catalog_metadata'].get('date', 'Unknown')}
Type: {state['catalog_metadata'].get('type', 'Unknown')}

**GROUND TRUTH REFERENCE:**
{state['ground_truth'][:2000]}...

**TRANSCRIPTION ATTEMPTS:**
{chr(10).join(transcriptions_summary)}

**CONSENSUS TRANSCRIPTION** (Strategy: {state['consensus_strategy']}):
{state['final_transcription'][:1000]}...

**YOUR TASK:**
Analyze these transcriptions and provide:

1. **Coherence Assessment**: Do the transcriptions make sense as Hebrew text? Are there obvious OCR errors, garbled text, or nonsensical character sequences?

2. **Catalog Alignment**: How well does the transcribed content align with the catalog description? Does it match the expected document type (legal, liturgical, biblical, etc.)?

3. **Recommended Transcription**: Which model's output appears most accurate and why? Consider both the metrics and the actual Hebrew text quality.

4. **Content Summary**: Provide a 2-3 sentence English summary of what this document contains based on the transcriptions.

5. **Key Observations**: Note any interesting textual variants, damage patterns, or paleographic features visible in the transcriptions.

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

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
            ),
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
        print(f"    📋 Summary: {state['analysis_result']['summary'][:100]}...")
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


def should_review(state: TranscriptionState) -> str:
    """Determine if consensus result needs human review."""
    return "review" if state['needs_review'] else "end"


async def review_node(state: TranscriptionState) -> TranscriptionState:
    """Placeholder node for human review workflow."""
    return state


# ============================================================================
# Build Graph - Sequential for Pro to avoid rate limits
# ============================================================================

def build_evaluation_graph():
    """Build LangGraph workflow - Sequential Pro execution"""

    async def parallel_models_node(state: TranscriptionState) -> TranscriptionState:
        """Execute models with Pro running sequentially to avoid rate limits"""

        # Save ground truth
        save_raw_output(state['doc_id'], 'ground_truth', state['ground_truth'], state['ground_truth'])

        # Run OCR and Flash in parallel (they're fast and have high limits)
        parallel_tasks = []
        if Config.USE_VISION_OCR:
            parallel_tasks.append(vision_ocr_node(dict(state)))
        if Config.USE_GEMINI_FLASH:
            parallel_tasks.append(gemini_flash_node(dict(state)))

        if parallel_tasks:
            results = await asyncio.gather(*parallel_tasks)

            for result in results:
                if 'vision_ocr_result' in result and result['vision_ocr_result'] is not None:
                    state['vision_ocr_result'] = result['vision_ocr_result']
                    state['vision_ocr_metrics'] = result['vision_ocr_metrics']
                    if 'vision_ocr' in result.get('model_times', {}):
                        state['model_times']['vision_ocr'] = result['model_times']['vision_ocr']

                if 'gemini_flash_result' in result and result['gemini_flash_result'] is not None:
                    state['gemini_flash_result'] = result['gemini_flash_result']
                    state['gemini_flash_metrics'] = result['gemini_flash_metrics']
                    if 'gemini_flash' in result.get('model_times', {}):
                        state['model_times']['gemini_flash'] = result['model_times']['gemini_flash']

        # Run Pro SEQUENTIALLY (avoid rate limits, allow full timeout)
        if Config.USE_GEMINI_PRO:
            # Small delay before Pro to space out requests
            await asyncio.sleep(3)
            pro_result = await gemini_pro_node(state)

            if 'gemini_pro_result' in pro_result and pro_result['gemini_pro_result'] is not None:
                state['gemini_pro_result'] = pro_result['gemini_pro_result']
                state['gemini_pro_metrics'] = pro_result['gemini_pro_metrics']
                if 'gemini_pro' in pro_result.get('model_times', {}):
                    state['model_times']['gemini_pro'] = pro_result['model_times']['gemini_pro']

        return state

    workflow = StateGraph(TranscriptionState)

    workflow.add_node("parallel_models", parallel_models_node)
    workflow.add_node("consensus", consensus_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("review", review_node)

    workflow.set_entry_point("parallel_models")
    workflow.add_edge("parallel_models", "consensus")
    workflow.add_edge("consensus", "analysis")
    workflow.add_conditional_edges(
        "analysis",
        should_review,
        {"review": "review", "end": END}
    )
    workflow.add_edge("review", END)

    return workflow.compile()


# ============================================================================
# W&B Logging
# ============================================================================

def log_to_wandb(state: TranscriptionState, run_name: str) -> tuple:
    """Log comprehensive metrics and return table rows for batch logging."""

    # Log scalar metrics for time series
    metrics = {
        'doc_id': state['doc_id'],
        'processing_time_total': state['processing_time'],
    }

    # Log individual model metrics as scalars
    for model_key in ['vision_ocr', 'gemini_flash', 'gemini_pro']:
        model_metrics = state.get(f'{model_key}_metrics')
        if model_metrics:
            for key, value in model_metrics.items():
                if isinstance(value, (int, float)):
                    metrics[f'{model_key}/{key}'] = value

    # Log consensus metrics
    if state.get('consensus_metrics'):
        for key, value in state['consensus_metrics'].items():
            if isinstance(value, (int, float)):
                metrics[f'consensus/{key}'] = value

    # Log strategy and metadata
    metrics['consensus/strategy'] = state['consensus_strategy']
    metrics['consensus/num_disagreements'] = len(state['disagreements'])
    metrics['consensus/needs_review'] = state['needs_review']

    # Log timing
    for model, duration in state['model_times'].items():
        metrics[f'timing/{model}'] = duration

    wandb.log(metrics)

    # ========================================================================
    # TEXT COMPARISON ROW
    # ========================================================================

    image_path = state['image_path']
    wandb_image = wandb.Image(image_path) if Path(image_path).exists() else None

    text_comparison_row = [
        state['doc_id'],
        wandb_image,
        state['ground_truth'],
        (state.get('vision_ocr_result') or {}).get('text', ''),
        (state.get('gemini_flash_result') or {}).get('text', ''),
        (state.get('gemini_pro_result') or {}).get('text', ''),
        state['final_transcription'],
        state['consensus_strategy']
    ]

    # ========================================================================
    # METRICS ROWS
    # ========================================================================

    metrics_rows = []

    for model_key, model_name in [
        ('vision_ocr', 'Google Vision OCR'),
        ('gemini_flash', 'Gemini Flash'),
        ('gemini_pro', 'Gemini Pro'),
        ('consensus', 'Consensus')
    ]:
        metrics_key = f'{model_key}_metrics'
        model_metrics = state.get(metrics_key)

        if model_metrics:
            metrics_rows.append([
                state['doc_id'],
                model_name,
                model_metrics.get('cer'),
                model_metrics.get('wer'),
                model_metrics.get('similarity'),
                model_metrics.get('char_count'),
                model_metrics.get('gt_char_count'),
                model_metrics.get('char_diff'),
                model_metrics.get('processing_time'),
                model_metrics.get('exact_match', False)
            ])

    # ========================================================================
    # ANALYSIS ROW
    # ========================================================================

    analysis_row = None
    if state.get('analysis_result'):
        analysis_row = [
            state['doc_id'],
            state['catalog_metadata'].get('description', '')[:200],
            state['analysis_result'].get('summary', ''),
            state['analysis_result'].get('translation', ''),
            state['analysis_result'].get('coherence_assessment', ''),
            state['analysis_result'].get('catalog_alignment', ''),
            state['analysis_result'].get('recommended_transcription', ''),
            state['analysis_result'].get('confidence_reasoning', ''),
            state['analysis_result'].get('key_observations', ''),
            state['analysis_result'].get('processing_time', 0.0)
        ]

    # ========================================================================
    # HTML COMPARISON
    # ========================================================================

    ocr_text = (state.get('vision_ocr_result') or {}).get('text', '')
    flash_text = (state.get('gemini_flash_result') or {}).get('text', '')
    pro_text = (state.get('gemini_pro_result') or {}).get('text', '')

    if ocr_text or flash_text or pro_text:
        comparison_html = create_comparison_html(
            state['doc_id'],
            state['ground_truth'],
            ocr_text,
            flash_text,
            pro_text,
            state['final_transcription'],
            state['consensus_strategy']
        )

        # Save HTML file
        html_path = Config.RAW_OUTPUTS_DIR / state['doc_id'] / "comparison.html"
        html_path.write_text(comparison_html, encoding='utf-8')

        # Log to W&B
        wandb.log({f"comparison_html/{state['doc_id']}": wandb.Html(comparison_html)})

    return text_comparison_row, metrics_rows, analysis_row


# ============================================================================
# Main Pipeline
# ============================================================================

async def process_document(doc_id: str, metadata: dict, graph, wandb_run) -> dict:
    """Process a single document through the complete transcription pipeline."""

    print(f"\n📄 {doc_id}")
    print(f"   {metadata.get('description', '')[:80]}...")

    images = metadata.get('images', [])
    if not images:
        print(f"  ⚠️  No images found - skipping")
        return None

    image_path = await download_image(images[0], doc_id, Config.IMAGES_DIR)
    if not image_path.exists():
        print(f"  ⚠️  Image missing: {image_path} - skipping")
        return None

    # Extract ground truth
    ground_truth = extract_ground_truth(metadata.get('transcriptions', []))
    if not ground_truth:
        print(f"  ⚠️  No ground truth found - skipping")
        return None

    print(f"  📝 Ground truth: {len(ground_truth)} chars")

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

    # Execute
    final_state = await graph.ainvoke(initial_state)
    final_state['processing_time'] = time.time() - start_time

    # Log to W&B and get table rows
    text_row, metrics_rows, analysis_row = log_to_wandb(final_state, wandb_run.name)

    # Store rows in state for later batch logging
    final_state['_text_comparison_row'] = text_row
    final_state['_metrics_rows'] = metrics_rows
    final_state['_analysis_row'] = analysis_row

    # Save incrementally to file
    save_incremental_result(final_state)

    print(f"  ✅ Complete in {final_state['processing_time']:.2f}s")
    print(f"  📂 Raw outputs: {Config.RAW_OUTPUTS_DIR / doc_id}")

    return final_state


"""
Cairo Genizah Transcription Evaluation Pipeline - BATCH PROCESSING MODE
Process corpus in manageable batches with rate limit protection
"""


# ============================================================================
# Batch Configuration
# ============================================================================

class BatchConfig:
    """Batch processing configuration"""

    # Batch parameters
    BATCH_SIZE = 50  # Documents per batch
    START_DOC = 0  # Starting index (0-based)

    # Rate limiting
    DELAY_BETWEEN_DOCS = 3  # seconds
    DELAY_BETWEEN_BATCHES = 300  # 5 minutes between batches
    MAX_RETRIES_PER_DOC = 3

    # Progress tracking
    BATCH_TRACKING_FILE = Path("./transcription_results/batch_progress.json")

    @classmethod
    def load_batch_progress(cls) -> dict:
        """Load batch processing progress"""
        if cls.BATCH_TRACKING_FILE.exists():
            with open(cls.BATCH_TRACKING_FILE, 'r') as f:
                return json.load(f)
        return {
            'completed_batches': [],
            'failed_docs': [],
            'last_processed_doc': None,
            'total_processed': 0
        }

    @classmethod
    def save_batch_progress(cls, progress: dict):
        """Save batch processing progress"""
        cls.BATCH_TRACKING_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(cls.BATCH_TRACKING_FILE, 'w') as f:
            json.dump(progress, f, indent=2)


# ============================================================================
# Batch Manager
# ============================================================================

class BatchManager:
    """Manage batch processing and resume capability"""

    def __init__(self, total_docs: int):
        self.total_docs = total_docs
        self.progress = BatchConfig.load_batch_progress()

    def get_batch_info(self, start_doc: int, num_docs: int) -> dict:
        """Calculate batch information"""

        end_doc = min(start_doc + num_docs, self.total_docs)
        batch_num = start_doc // num_docs + 1
        total_batches = (self.total_docs + num_docs - 1) // num_docs

        return {
            'batch_num': batch_num,
            'total_batches': total_batches,
            'start_doc': start_doc,
            'end_doc': end_doc,
            'num_docs': end_doc - start_doc,
            'progress_pct': (end_doc / self.total_docs) * 100
        }

    def is_batch_completed(self, batch_num: int) -> bool:
        """Check if batch was already processed"""
        return batch_num in self.progress['completed_batches']

    def mark_batch_completed(self, batch_num: int, doc_ids: list):
        """Mark batch as completed"""
        if batch_num not in self.progress['completed_batches']:
            self.progress['completed_batches'].append(batch_num)
        self.progress['total_processed'] += len(doc_ids)
        self.progress['last_processed_doc'] = doc_ids[-1] if doc_ids else None
        BatchConfig.save_batch_progress(self.progress)

    def mark_doc_failed(self, doc_id: str, error: str):
        """Track failed documents"""
        self.progress['failed_docs'].append({
            'doc_id': doc_id,
            'error': error,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        BatchConfig.save_batch_progress(self.progress)

    def get_resume_point(self) -> int:
        """Get index to resume from"""
        if self.progress['last_processed_doc']:
            # Resume from next document
            return self.progress['total_processed']
        return 0

    def generate_progress_report(self) -> str:
        """Generate progress report"""

        completed = len(self.progress['completed_batches'])
        failed = len(self.progress['failed_docs'])
        processed = self.progress['total_processed']
        remaining = self.total_docs - processed

        report = f"""
{'=' * 80}
BATCH PROCESSING PROGRESS
{'=' * 80}

Total documents: {self.total_docs}
Processed: {processed} ({processed / self.total_docs * 100:.1f}%)
Remaining: {remaining}

Completed batches: {completed}
Failed documents: {failed}

Last processed: {self.progress['last_processed_doc']}
"""

        if self.progress['failed_docs']:
            report += "\n\nFailed Documents:\n"
            for fail in self.progress['failed_docs'][-10:]:  # Show last 10
                report += f"  - {fail['doc_id']}: {fail['error'][:50]}\n"

        return report


# ============================================================================
# Updated Main with Batch Processing
# ============================================================================

async def main(
        start_doc: int = None,
        num_docs: int = None,
        resume: bool = False
):
    """Main evaluation pipeline with batch processing support.

    :param start_doc: Starting document index (0-based). If None, uses BatchConfig.START_DOC
    :type start_doc: int, optional
    :param num_docs: Number of documents to process. If None, uses BatchConfig.BATCH_SIZE
    :type num_docs: int, optional
    :param resume: Resume from last processed document
    :type resume: bool
    """

    print("=" * 80)
    print("Cairo Genizah Transcription Evaluation - BATCH MODE")
    print("=" * 80)

    # Setup
    Config.IMAGES_DIR.mkdir(exist_ok=True)
    Config.RESULTS_DIR.mkdir(exist_ok=True)
    Config.RAW_OUTPUTS_DIR.mkdir(exist_ok=True)

    # Verify API keys
    assert Config.GEMINI_API_KEY, "GEMINI_API_KEY environment variable not set"
    assert Config.GOOGLE_VISION_CREDENTIALS, "GOOGLE_APPLICATION_CREDENTIALS not set"

    # Load catalog
    print(f"\n📚 Loading catalog...")
    with open(Config.CATALOG_PATH) as f:
        catalog = json.load(f)

    # Filter for documents with ground truth
    eval_docs = {
        doc_id: metadata
        for doc_id, metadata in catalog.items()
        if has_ground_truth(metadata)
    }

    print(f"   Total docs in catalog: {len(catalog)}")
    print(f"   With ground truth: {len(eval_docs)}")

    # Initialize batch manager
    batch_manager = BatchManager(len(eval_docs))

    # Determine batch parameters
    if resume:
        start_doc = batch_manager.get_resume_point()
        print(f"\n🔄 Resuming from document {start_doc}")
    elif start_doc is None:
        start_doc = BatchConfig.START_DOC

    if num_docs is None:
        num_docs = BatchConfig.BATCH_SIZE

    # Get batch info
    batch_info = batch_manager.get_batch_info(start_doc, num_docs)

    print(f"\n📦 Batch Configuration:")
    print(f"   Batch: {batch_info['batch_num']}/{batch_info['total_batches']}")
    print(f"   Documents: {start_doc} to {batch_info['end_doc']} ({batch_info['num_docs']} docs)")
    print(f"   Overall progress: {batch_info['progress_pct']:.1f}%")

    # Check if batch already completed
    if batch_manager.is_batch_completed(batch_info['batch_num']):
        print(f"\n⚠️  Batch {batch_info['batch_num']} already completed!")
        proceed = input("Process anyway? (y/n): ")
        if proceed.lower() != 'y':
            print("Exiting...")
            return

    # Slice document list for this batch
    doc_ids = list(eval_docs.keys())
    batch_doc_ids = doc_ids[start_doc:batch_info['end_doc']]
    batch_docs = {doc_id: eval_docs[doc_id] for doc_id in batch_doc_ids}

    print(f"\n📋 Processing {len(batch_docs)} documents in this batch")
    print(f"   First doc: {batch_doc_ids[0]}")
    print(f"   Last doc: {batch_doc_ids[-1]}")

    # Show sample
    print("\nSample documents:")
    for doc_id in batch_doc_ids[:3]:
        gt = extract_ground_truth(batch_docs[doc_id].get('transcriptions', []))
        print(f"  • {doc_id}: {len(gt)} chars")

    # Estimated time
    est_time_per_doc = 120  # 2 minutes per doc with Pro
    est_total_time = len(batch_docs) * est_time_per_doc / 60
    print(f"\n⏱️  Estimated time: {est_total_time:.0f} minutes ({est_total_time / 60:.1f} hours)")

    # Config summary
    print(f"\n⚙️  Configuration:")
    print(f"   Vision OCR: {'✓' if Config.USE_VISION_OCR else '✗'}")
    print(f"   Gemini Flash: {'✓' if Config.USE_GEMINI_FLASH else '✗'}")
    print(f"   Gemini Pro: {'✓' if Config.USE_GEMINI_PRO else '✗'}")
    print(f"   Analysis: {'✓' if Config.USE_ANALYSIS else '✗'}")
    print(f"   Timeout per model: {Config.GEMINI_TIMEOUT}s")
    print(f"   Delay between docs: {Config.DELAY_BETWEEN_DOCS}s")

    # Clear incremental results for this batch
    batch_results_file = Config.RESULTS_DIR / f"batch_{batch_info['batch_num']}_incremental.jsonl"
    if batch_results_file.exists():
        print(f"\n⚠️  Found existing batch results file")
        proceed = input("Overwrite? (y/n): ")
        if proceed.lower() == 'y':
            batch_results_file.unlink()

    # Initialize W&B with batch-specific name
    wandb_run = wandb.init(
        project=Config.WANDB_PROJECT,
        entity=Config.WANDB_ENTITY,
        name=f"batch-{batch_info['batch_num']}-of-{batch_info['total_batches']}-{time.strftime('%Y%m%d-%H%M%S')}",
        tags=[
            f"batch_{batch_info['batch_num']}",
            f"docs_{start_doc}_to_{batch_info['end_doc']}",
        ],
        config={
            "mode": "batch_processing",
            "batch_num": batch_info['batch_num'],
            "total_batches": batch_info['total_batches'],
            "start_doc": start_doc,
            "end_doc": batch_info['end_doc'],
            "num_docs": len(batch_docs),
            "models": [
                m for m, enabled in [
                    ("google_vision_ocr", Config.USE_VISION_OCR),
                    ("gemini_3_flash", Config.USE_GEMINI_FLASH),
                    ("gemini_3_pro", Config.USE_GEMINI_PRO)
                ] if enabled
            ],
            "timeout_per_model": Config.GEMINI_TIMEOUT,
            "max_retries": Config.MAX_RETRIES,
        }
    )

    # Build graph
    graph = build_evaluation_graph()

    # Process documents - collect table rows
    results = []
    text_comparison_rows = []
    all_metrics_rows = []
    analysis_rows = []

    failed_docs = []

    for i, (doc_id, metadata) in enumerate(batch_docs.items(), 1):
        global_index = start_doc + i
        print(f"\n[{i}/{len(batch_docs)}] (Global: {global_index}/{len(eval_docs)})", end=" ")

        try:
            result = await process_document(doc_id, metadata, graph, wandb_run)
            if result:
                results.append(result)
                # Collect table rows
                text_comparison_rows.append(result['_text_comparison_row'])
                all_metrics_rows.extend(result['_metrics_rows'])
                if result.get('_analysis_row'):
                    analysis_rows.append(result['_analysis_row'])

                # Save incrementally to batch-specific file
                with open(batch_results_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False, default=str) + '\n')
            else:
                failed_docs.append({'doc_id': doc_id, 'reason': 'No result returned'})
                batch_manager.mark_doc_failed(doc_id, 'No result returned')

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"\n  ❌ FATAL ERROR: {error_msg}")
            failed_docs.append({'doc_id': doc_id, 'reason': error_msg})
            batch_manager.mark_doc_failed(doc_id, error_msg)

        # Rate limit protection - delay between documents
        if i < len(batch_docs):
            print(f"\n  ⏳ Cooling down for {Config.DELAY_BETWEEN_DOCS}s...")
            await asyncio.sleep(Config.DELAY_BETWEEN_DOCS)

    # ========================================================================
    # LOG BATCH TABLES TO W&B
    # ========================================================================

    print("\n📊 Logging batch tables to W&B...")

    # Text comparison table
    if text_comparison_rows:
        text_comparison_table = wandb.Table(
            columns=[
                "fragment_id",
                "image",
                "ground_truth",
                "vision_ocr",
                "gemini_flash",
                "gemini_pro",
                "consensus",
                "consensus_strategy"
            ],
            data=text_comparison_rows
        )
        wandb.log({f"batch_{batch_info['batch_num']}_text_comparison": text_comparison_table})

    # Metrics table
    if all_metrics_rows:
        metrics_table = wandb.Table(
            columns=[
                "fragment_id",
                "model",
                "cer",
                "wer",
                "similarity",
                "char_count",
                "gt_char_count",
                "char_diff",
                "processing_time_sec",
                "exact_match"
            ],
            data=all_metrics_rows
        )
        wandb.log({f"batch_{batch_info['batch_num']}_metrics": metrics_table})

    # Analysis table
    if analysis_rows:
        analysis_table = wandb.Table(
            columns=[
                "fragment_id",
                "catalog_description",
                "content_summary",
                "translation_sample",
                "coherence_assessment",
                "catalog_alignment",
                "recommended_model",
                "reasoning",
                "key_observations",
                "analysis_time_sec"
            ],
            data=analysis_rows
        )
        wandb.log({f"batch_{batch_info['batch_num']}_analysis": analysis_table})
        print(f"   ✓ Logged {len(analysis_rows)} analysis records")

    # ========================================================================
    # BATCH SUMMARY
    # ========================================================================

    print("\n" + "=" * 80)
    print(f"BATCH {batch_info['batch_num']} SUMMARY")
    print("=" * 80)

    print(f"\nProcessed: {len(results)}/{len(batch_docs)}")
    print(f"Failed: {len(failed_docs)}")

    if results:
        # Individual model stats
        for model_name in ['vision_ocr', 'gemini_flash', 'gemini_pro']:
            metrics_key = f'{model_name}_metrics'
            model_results = [r[metrics_key] for r in results if r.get(metrics_key)]

            if model_results:
                avg_cer = sum(m['cer'] for m in model_results) / len(model_results)
                avg_similarity = sum(m['similarity'] for m in model_results) / len(model_results)
                avg_time = sum(m['processing_time'] for m in model_results) / len(model_results)

                print(f"\n{model_name.upper().replace('_', ' ')}:")
                print(f"  Avg CER: {avg_cer:.3f}")
                print(f"  Avg Similarity: {avg_similarity:.3f}")
                print(f"  Avg Time: {avg_time:.1f}s")
                print(f"  Processed: {len(model_results)}/{len(results)}")

        # Consensus stats
        consensus_results = [r['consensus_metrics'] for r in results if r.get('consensus_metrics')]
        if consensus_results:
            avg_consensus_cer = sum(m['cer'] for m in consensus_results) / len(consensus_results)
            avg_consensus_sim = sum(m['similarity'] for m in consensus_results) / len(consensus_results)

            print(f"\nCONSENSUS:")
            print(f"  Avg CER: {avg_consensus_cer:.3f}")
            print(f"  Avg Similarity: {avg_consensus_sim:.3f}")

    if failed_docs:
        print("\n\n❌ FAILED DOCUMENTS:")
        for fail in failed_docs:
            print(f"  - {fail['doc_id']}: {fail['reason'][:80]}")

    # Save batch results
    batch_output_file = Config.RESULTS_DIR / f"batch_{batch_info['batch_num']}_results.json"
    with open(batch_output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'batch_info': batch_info,
            'results': results,
            'failed_docs': failed_docs,
            'summary_stats': {
                'processed': len(results),
                'failed': len(failed_docs),
                'success_rate': len(results) / len(batch_docs) if batch_docs else 0
            }
        }, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n💾 Batch results: {batch_output_file}")
    print(f"💾 Incremental results: {batch_results_file}")
    print(f"📊 W&B dashboard: {wandb_run.url}")

    # Mark batch as completed
    batch_manager.mark_batch_completed(batch_info['batch_num'], batch_doc_ids)

    # Show overall progress
    print("\n" + batch_manager.generate_progress_report())

    wandb.finish()

    # Suggest next batch
    if batch_info['end_doc'] < len(eval_docs):
        next_start = batch_info['end_doc']
        print(f"\n💡 Next batch command:")
        print(f"   python genizah_fragment_agent.py --start-doc {next_start} --num-docs {num_docs}")


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Cairo Genizah Transcription Evaluation - Batch Processing'
    )
    parser.add_argument(
        '--start-doc',
        type=int,
        default=25,
        help='Starting document index (0-based). Default: 0'
    )
    parser.add_argument(
        '--num-docs',
        type=int,
        default=None,
        help=f'Number of documents to process. Default: {BatchConfig.BATCH_SIZE}'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last processed document'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Set default batch size'
    )

    args = parser.parse_args()

    # Update batch config if specified
    if args.batch_size:
        BatchConfig.BATCH_SIZE = args.batch_size

    # Run batch processing
    asyncio.run(main(
        start_doc=args.start_doc,
        num_docs=args.num_docs,
        resume=args.resume
    ))


if __name__ == "__main__":
    asyncio.run(main())