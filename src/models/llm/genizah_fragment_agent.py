"""
Cairo Genizah Transcription Evaluation Pipeline
Incremental saving + optional Pro model
"""

import asyncio
import json
import os
from pathlib import Path
from typing import TypedDict, Optional, Dict, List
from urllib.parse import urlparse
import aiohttp
from langgraph.graph import StateGraph, END
import google.generativeai as genai
from google.cloud import vision
import wandb
from difflib import SequenceMatcher
import time
from collections import defaultdict
import dotenv
dotenv.load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Pipeline configuration"""
    catalog_path = "/Users/isaac1/Documents/historical-document-analysis/src/datasets/raw_data/merged_princeton_friedberger_first_25_with_transcriptions.json"
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

    # Model selection
    USE_VISION_OCR = True
    USE_GEMINI_FLASH = True
    USE_GEMINI_PRO = False  # Set to False to skip Pro (quota issues)

    # Evaluation
    MIN_CONFIDENCE_THRESHOLD = 0.7

    # Incremental results file
    INCREMENTAL_RESULTS_FILE = Path("./transcription_results/incremental_results.jsonl")


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
    """State passed between nodes"""
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

    # Timing
    processing_time: float
    model_times: dict


# ============================================================================
# Helper Functions
# ============================================================================

async def download_image(url_or_filename: str, doc_id: str, output_dir: Path, image_prefix="https://storage.googleapis.com/cairo-genizah-es-json/images/") -> Path:
    """Download image from URL or locate local file"""
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
    """Check if document has ground truth transcription"""
    transcriptions = metadata.get('transcriptions', [])
    if not transcriptions:
        return False

    ground_truth = extract_ground_truth(transcriptions)
    return len(ground_truth.strip()) > 10


def evaluate_transcription(ground_truth: str, hypothesis: str, model_name: str) -> dict:
    """Calculate all metrics for a transcription"""

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
# LangGraph Nodes
# ============================================================================
async def vision_ocr_node(state: TranscriptionState) -> TranscriptionState:
    """Call Google Vision OCR and evaluate"""
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
    """Call Gemini Flash and evaluate"""
    if not Config.USE_GEMINI_FLASH:
        print(f"  ⏭️  Gemini Flash: Skipped")
        state['gemini_flash_result'] = None
        state['gemini_flash_metrics'] = None
        return state

    print(f"  ⚡ Gemini Flash...")
    start_time = time.time()

    try:
        genai.configure(api_key=Config.GEMINI_API_KEY)
        model = genai.GenerativeModel(Config.GEMINI_FLASH_MODEL)

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

        uploaded_file = genai.upload_file(str(state['image_path']))

        response = model.generate_content(
            [prompt, uploaded_file],
            generation_config=genai.GenerationConfig(
                temperature=0.1,
            )
        )

        # Check if we have any candidates
        if not response.candidates:
            raise ValueError(
                f"Gemini Flash returned no candidates\n"
                f"Prompt feedback: {getattr(response, 'prompt_feedback', 'None')}"
            )

        candidate = response.candidates[0]

        # Check for safety blocks
        if candidate.finish_reason == 3:  # SAFETY
            safety_info = "\n".join([
                f"      {rating.category}: {rating.probability}"
                for rating in candidate.safety_ratings
            ]) if hasattr(candidate, 'safety_ratings') else "No safety info"
            raise ValueError(f"Blocked by safety filters\n{safety_info}")

        # Check if we have content
        if not candidate.content or not candidate.content.parts:
            raise ValueError(f"No content returned (finish_reason={candidate.finish_reason})")

        # Extract text
        first_part = candidate.content.parts[0]
        if not hasattr(first_part, 'text') or not first_part.text:
            raise ValueError("Content part has no text")

        text = first_part.text.strip()
        elapsed = time.time() - start_time

        print(f"    ✓ Extracted {len(text)} chars")

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
    """Call Gemini Pro and evaluate"""
    if not Config.USE_GEMINI_PRO:
        print(f"  ⏭️  Gemini Pro: Skipped (disabled)")
        state['gemini_pro_result'] = None
        state['gemini_pro_metrics'] = None
        return state

    print(f"  🎯 Gemini Pro...")
    start_time = time.time()

    try:
        genai.configure(api_key=Config.GEMINI_API_KEY)
        model = genai.GenerativeModel(Config.GEMINI_PRO_MODEL)

        catalog_hint = state['catalog_metadata'].get('description', '')

        prompt = f"""Transcribe this Hebrew manuscript image with maximum accuracy.

Context: {catalog_hint}

CRITICAL: Transcribe exactly what appears on the page. Do not normalize or correct the text.

Return only Hebrew transcription."""

        uploaded_file = genai.upload_file(str(state['image_path']))

        response = model.generate_content(
            [prompt, uploaded_file],
            generation_config=genai.GenerationConfig(
                temperature=0.1,
            )
        )

        text = response.text.strip()
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
    """Compute consensus and evaluate it"""
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

    # Consensus strategy
    vision_result = state['vision_ocr_result']

    if vision_result and vision_result['confidence'] > Config.MIN_CONFIDENCE_THRESHOLD:
        state['final_transcription'] = vision_result['text']
        state['confidence_score'] = vision_result['confidence']
        state['consensus_strategy'] = "ocr_primary"

    elif state['gemini_pro_result'] and state['gemini_flash_result']:
        state['final_transcription'] = state['gemini_pro_result']['text']
        state['confidence_score'] = 0.82
        state['consensus_strategy'] = "llm_pro_preferred"

    elif state['gemini_pro_result']:
        state['final_transcription'] = state['gemini_pro_result']['text']
        state['confidence_score'] = 0.80
        state['consensus_strategy'] = "llm_pro_only"

    elif state['gemini_flash_result']:
        state['final_transcription'] = state['gemini_flash_result']['text']
        state['confidence_score'] = 0.75
        state['consensus_strategy'] = "llm_flash_only"

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


def should_review(state: TranscriptionState) -> str:
    """Conditional routing"""
    return "review" if state['needs_review'] else "end"


async def review_node(state: TranscriptionState) -> TranscriptionState:
    """Placeholder for review"""
    return state


# ============================================================================
# Build Graph
# ============================================================================

def build_evaluation_graph():
    """Build LangGraph workflow with parallel model execution"""

    async def parallel_models_node(state: TranscriptionState) -> TranscriptionState:
        """Execute all models in parallel"""

        # Save ground truth
        save_raw_output(state['doc_id'], 'ground_truth', state['ground_truth'], state['ground_truth'])

        # Build list of tasks based on config
        tasks = []
        if Config.USE_VISION_OCR:
            tasks.append(vision_ocr_node(dict(state)))
        if Config.USE_GEMINI_FLASH:
            tasks.append(gemini_flash_node(dict(state)))
        if Config.USE_GEMINI_PRO:
            tasks.append(gemini_pro_node(dict(state)))

        if not tasks:
            raise ValueError("No models enabled! Enable at least one model in Config.")

        results = await asyncio.gather(*tasks)

        # Merge results - only update fields each model is responsible for
        for result in results:
            # Vision OCR fields
            if 'vision_ocr_result' in result and result['vision_ocr_result'] is not None:
                state['vision_ocr_result'] = result['vision_ocr_result']
                state['vision_ocr_metrics'] = result['vision_ocr_metrics']
                if 'vision_ocr' in result.get('model_times', {}):
                    state['model_times']['vision_ocr'] = result['model_times']['vision_ocr']

            # Gemini Flash fields
            if 'gemini_flash_result' in result and result['gemini_flash_result'] is not None:
                state['gemini_flash_result'] = result['gemini_flash_result']
                state['gemini_flash_metrics'] = result['gemini_flash_metrics']
                if 'gemini_flash' in result.get('model_times', {}):
                    state['model_times']['gemini_flash'] = result['model_times']['gemini_flash']

            # Gemini Pro fields
            if 'gemini_pro_result' in result and result['gemini_pro_result'] is not None:
                state['gemini_pro_result'] = result['gemini_pro_result']
                state['gemini_pro_metrics'] = result['gemini_pro_metrics']
                if 'gemini_pro' in result.get('model_times', {}):
                    state['model_times']['gemini_pro'] = result['model_times']['gemini_pro']

        return state
    workflow = StateGraph(TranscriptionState)

    workflow.add_node("parallel_models", parallel_models_node)
    workflow.add_node("consensus", consensus_node)
    workflow.add_node("review", review_node)

    workflow.set_entry_point("parallel_models")
    workflow.add_edge("parallel_models", "consensus")
    workflow.add_conditional_edges(
        "consensus",
        should_review,
        {"review": "review", "end": END}
    )
    workflow.add_edge("review", END)

    return workflow.compile()


# ============================================================================
# W&B Logging
# ============================================================================
def log_to_wandb(state: TranscriptionState, run_name: str) -> tuple:
    """Log comprehensive metrics and return table rows for batch logging.

    :param state: Completed pipeline state with all results
    :type state: TranscriptionState
    :param run_name: W&B run identifier
    :type run_name: str
    :return: Tuple of (text_comparison_row, metrics_rows)
    :rtype: tuple
    """

    # Log scalar metrics for time series (keep this - logs immediately)
    metrics = {
        'doc_id': state['doc_id'],
        'processing_time_total': state['processing_time'],
    }

    # ... all the metrics logging code stays the same ...

    wandb.log(metrics)

    # ========================================================================
    # TEXT COMPARISON ROW - Return for batch table
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
    # METRICS ROWS - Return for batch table
    # ========================================================================

    metrics_rows = []

    # Add row for each model that ran
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
    # HTML COMPARISON (keep this - logs immediately with unique key)
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

        # Log to W&B with unique key per document
        wandb.log({f"comparison_html/{state['doc_id']}": wandb.Html(comparison_html)})

    return text_comparison_row, metrics_rows
    # ========================================================================
    # HTML COMPARISON (existing, keep for detailed view)
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


# ============================================================================
# Main Pipeline
# ============================================================================

async def process_document(doc_id: str, metadata: dict, graph, wandb_run) -> dict:
    """Process single document - saves results incrementally"""

    print(f"\n📄 {doc_id}")
    print(f"   {metadata.get('description', '')[:80]}...")

    images = metadata.get('images', [])
    assert images, f"No images found for {doc_id}"

    image_path = await download_image(images[0], doc_id, Config.IMAGES_DIR)
    if not image_path.exists():
        print(f"  ⚠️  Image download failed for {doc_id}, skipping.")

        return None
    assert image_path.exists(), f"Image file missing: {image_path}"

    # Extract ground truth
    ground_truth = extract_ground_truth(metadata.get('transcriptions', []))
    assert ground_truth, f"No ground truth found for {doc_id}"

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
        processing_time=0.0,
        model_times={}
    )

    # Execute
    final_state = await graph.ainvoke(initial_state)
    final_state['processing_time'] = time.time() - start_time

    # Log to W&B
    log_to_wandb(final_state, wandb_run.name)

    # Save incrementally to file
    save_incremental_result(final_state)

    print(f"  ✅ Complete in {final_state['processing_time']:.2f}s")
    print(f"  📂 Raw outputs: {Config.RAW_OUTPUTS_DIR / doc_id}")

    return final_state


async def main():
    """Main evaluation pipeline"""

    print("=" * 80)
    print("Cairo Genizah Transcription Evaluation")
    print("=" * 80)

    # Setup
    Config.IMAGES_DIR.mkdir(exist_ok=True)
    Config.RESULTS_DIR.mkdir(exist_ok=True)
    Config.RAW_OUTPUTS_DIR.mkdir(exist_ok=True)

    # Clear incremental results file for fresh run
    if Config.INCREMENTAL_RESULTS_FILE.exists():
        Config.INCREMENTAL_RESULTS_FILE.unlink()

    # Verify API keys
    assert Config.GEMINI_API_KEY, "GEMINI_API_KEY environment variable not set"
    assert Config.GOOGLE_VISION_CREDENTIALS, "GOOGLE_APPLICATION_CREDENTIALS not set"

    # Show config
    print(f"\n⚙️  Configuration:")
    print(f"   Vision OCR: {'✓' if Config.USE_VISION_OCR else '✗'}")
    print(f"   Gemini Flash: {'✓' if Config.USE_GEMINI_FLASH else '✗'}")
    print(f"   Gemini Pro: {'✓' if Config.USE_GEMINI_PRO else '✗ (disabled - quota issues)'}")

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

    print(f"   Total docs: {len(catalog)}")
    print(f"   With ground truth: {len(eval_docs)}")

    # Limit for initial test
    TEST_LIMIT = 10
    test_docs = dict(list(eval_docs.items())[:TEST_LIMIT])
    print(f"   Testing on: {len(test_docs)} documents\n")

    # Show sample
    print("Sample documents:")
    for doc_id in list(test_docs.keys())[:3]:
        gt = extract_ground_truth(test_docs[doc_id].get('transcriptions', []))
        print(f"  • {doc_id}: {len(gt)} chars")
    print()

    # Initialize W&B
    wandb_run = wandb.init(
        project=Config.WANDB_PROJECT,
        entity=Config.WANDB_ENTITY,
        name=f"eval-{time.strftime('%Y%m%d-%H%M%S')}",
        config={
            "models": [
                m for m, enabled in [
                    ("google_vision_ocr", Config.USE_VISION_OCR),
                    ("gemini_3_flash", Config.USE_GEMINI_FLASH),
                    ("gemini_3_pro", Config.USE_GEMINI_PRO)
                ] if enabled
            ],
            "num_documents": len(test_docs),
            "consensus_threshold": Config.MIN_CONFIDENCE_THRESHOLD
        }
    )

    # Build graph
    graph = build_evaluation_graph()

    # Process documents
    results = []
    for i, (doc_id, metadata) in enumerate(test_docs.items(), 1):
        print(f"\n[{i}/{len(test_docs)}]", end=" ")
        result = await process_document(doc_id, metadata, graph, wandb_run)
        results.append(result)

    # Aggregate statistics
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    # Individual model stats
    for model_name in ['vision_ocr', 'gemini_flash', 'gemini_pro']:
        metrics_key = f'{model_name}_metrics'
        model_results = [r[metrics_key] for r in results if r.get(metrics_key)]

        if model_results:
            avg_cer = sum(m['cer'] for m in model_results) / len(model_results)
            avg_similarity = sum(m['similarity'] for m in model_results) / len(model_results)

            print(f"\n{model_name.upper().replace('_', ' ')}:")
            print(f"  Avg CER: {avg_cer:.3f}")
            print(f"  Avg Similarity: {avg_similarity:.3f}")
            print(f"  Processed: {len(model_results)}/{len(results)}")

    # Consensus stats
    consensus_results = [r['consensus_metrics'] for r in results if r.get('consensus_metrics')]
    if consensus_results:
        avg_consensus_cer = sum(m['cer'] for m in consensus_results) / len(consensus_results)
        avg_consensus_sim = sum(m['similarity'] for m in consensus_results) / len(consensus_results)

        print(f"\nCONSENSUS:")
        print(f"  Avg CER: {avg_consensus_cer:.3f}")
        print(f"  Avg Similarity: {avg_consensus_sim:.3f}")

        # Strategy breakdown
        strategies = defaultdict(int)
        for r in results:
            strategies[r['consensus_strategy']] += 1

        print(f"\n  Strategies used:")
        for strategy, count in strategies.items():
            print(f"    {strategy}: {count}")

    # Compare
    print(f"\n{'='*80}")
    print("CONSENSUS vs BEST INDIVIDUAL MODEL")
    print(f"{'='*80}")

    for r in results:
        if not r.get('consensus_metrics'):
            continue

        individual_cers = []
        if r.get('vision_ocr_metrics'):
            individual_cers.append(('vision_ocr', r['vision_ocr_metrics']['cer']))
        if r.get('gemini_flash_metrics'):
            individual_cers.append(('gemini_flash', r['gemini_flash_metrics']['cer']))
        if r.get('gemini_pro_metrics'):
            individual_cers.append(('gemini_pro', r['gemini_pro_metrics']['cer']))

        if individual_cers:
            best_model, best_cer = min(individual_cers, key=lambda x: x[1])
            consensus_cer = r['consensus_metrics']['cer']

            improvement = best_cer - consensus_cer
            print(f"{r['doc_id'][:50]:50s} | Best: {best_model:15s} ({best_cer:.3f}) | "
                  f"Consensus: {consensus_cer:.3f} | Δ: {improvement:+.3f}")

    # Save final aggregated results
    output_file = Config.RESULTS_DIR / f"final_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n💾 Final results: {output_file}")
    print(f"💾 Incremental results: {Config.INCREMENTAL_RESULTS_FILE}")
    print(f"📂 Raw outputs: {Config.RAW_OUTPUTS_DIR}")
    print(f"📊 W&B dashboard: {wandb_run.url}")

    wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())