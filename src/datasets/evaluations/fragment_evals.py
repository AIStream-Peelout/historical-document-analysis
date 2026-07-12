"""
Cairo Genizah Transcription Evaluation Pipeline
Batch processing with metrics, W&B logging, and auto-resume
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
from difflib import SequenceMatcher
import wandb

# Import core agent
from src.models.llm.transcription.genizah_fragment_agent import (
    AgentConfig,
    TranscriptionState,
    transcribe_document,
    download_image,
    call_gemini_text_only,
    _series_from_doc_id,
)

# Local VLMs via LM Studio (e.g. Gemma 4)
from src.models.ocr.lms_transcriber import (
    transcribe_batch as lm_studio_transcribe_batch,
    complete_text as lm_studio_complete_text,
    check_lm_studio_health,
)

# Claude (flag-gated — expensive, off by default)
from src.models.llm.transcription.claude_transcriber import (
    transcribe_with_claude,
    ClaudeConfig,
)

# Shared benchmark metrics
from src.datasets.evaluations.metrics import (
    cer_pair,
    wer_pair,
    char_count_ratio,
    flag_failure_modes,
)


# ============================================================================
# Evaluation Configuration
# ============================================================================

class EvalConfig:
    """Evaluation-specific configuration"""
    catalog_path = "/Users/isaac/Documents/GitHub/historical-document-analysis/src/datasets/raw_data/cairo_genizah/evaluations/genizah_sample/merged_princeton_friedberger_first_25_with_transcriptions.json"
    IMAGES_DIR = Path("./genizah_images")
    RESULTS_DIR = Path("./transcription_results")
    RAW_OUTPUTS_DIR = Path("./transcription_raw_outputs")
    CATALOG_PATH = Path(catalog_path)

    # W&B
    WANDB_PROJECT = "cairo-genizah-transcription"
    WANDB_ENTITY = None  # Set via environment or command line

    # Batch parameters
    BATCH_SIZE = 50  # Documents per batch
    DELAY_BETWEEN_DOCS = 3  # seconds
    DELAY_BETWEEN_BATCHES = 300  # 5 minutes between batches

    # Progress tracking
    BATCH_TRACKING_FILE = Path("./transcription_results/batch_progress.json")
    INCREMENTAL_RESULTS_FILE = Path("./transcription_results/incremental_results.jsonl")

    # LM Studio (local open-weight VLMs) — top models from independent labs,
    # both scored in every run.
    LM_STUDIO_MODELS = ["qwen/qwen3-vl-8b", "google/gemma-4-31b-qat"]
    LM_STUDIO_BASE_URL = "http://localhost:1234/v1"


# ============================================================================
# Extra benchmark models (LM Studio VLMs + optional Claude)
# Populated once in main() before any documents are evaluated.
# ============================================================================

_lm_studio_models: List[str] = []
_lm_studio_base_url: str = EvalConfig.LM_STUDIO_BASE_URL
_use_claude: bool = False

# Text-only LLM used to reconstruct reading-order text from raw OCR output.
# BOTH OCR engines (Google Vision, Kraken) are routed through this same model
# so the two "specialized OCR + LLM extraction" pipelines are directly
# comparable to each other and to the end-to-end VLMs.
_segmentation_model: str = AgentConfig.GEMINI_FLASH_MODEL
_segmentation_base_url: Optional[str] = None  # set → served via LM Studio


def model_key_from_id(model_id: str) -> str:
    """Stable short identifier for a model ID, safe for W&B column names.

    e.g. "google/gemma-4-31b-qat" → "gemma_4_31b_qat"

    :param model_id: Provider-qualified model ID
    :type model_id: str
    :return: Sanitized snake_case key
    :rtype: str
    """
    name = model_id.split("/")[-1]
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def extra_model_keys() -> List[str]:
    """Return the state/W&B keys for all configured extra models.

    :return: List of model keys (LM Studio models first, then Claude)
    :rtype: List[str]
    """
    keys = [model_key_from_id(m) for m in _lm_studio_models]
    if _use_claude:
        keys.append("claude")
    return keys


def scored_model_keys() -> List[str]:
    """Return every model key that gets CER/WER metrics in this run.

    Two families are compared:

    * **End-to-end VLMs** — Gemini Flash/Pro, LM Studio VLMs, Claude.
    * **Specialized OCR + text-only LLM extraction** — Google Vision OCR
      and Kraken, each routed through the SAME extraction model
      (``_segmentation_model``) so the two pipelines are directly
      comparable.

    Raw OCR output is deliberately never scored: it is not necessarily in
    reading order, so scoring it directly would conflate OCR quality with
    line ordering.  Raw outputs are saved and shown for inspection only,
    so OCR-model failures can be told apart from extraction-LLM failures.

    :return: Ordered list of model state keys
    :rtype: List[str]
    """
    keys = ['gemini_flash', 'gemini_pro'] + extra_model_keys()
    if AgentConfig.USE_VISION_OCR:
        keys.append('vision_ocr_seg')
    if AgentConfig.USE_KRAKEN:
        keys.append('kraken_seg')
    return keys


def raw_ocr_keys() -> List[str]:
    """Return the enabled OCR engines whose RAW output is saved and shown
    for inspection but never scored.

    :return: Ordered list of OCR state-key prefixes
    :rtype: List[str]
    """
    keys = []
    if AgentConfig.USE_VISION_OCR:
        keys.append('vision_ocr')
    if AgentConfig.USE_KRAKEN:
        keys.append('kraken')
    return keys


_OCR_EXTRACTION_PROMPT = """\
The following text is raw OCR output from a Cairo Genizah manuscript fragment.
The text may be in Hebrew, Aramaic, or Judaeo-Arabic (Arabic written in Hebrew
script).  The OCR engine reads text block by block, so lines may be OUT of
natural reading order and may contain stray non-text artifacts.

Reconstruct the transcription in natural reading order (right to left, top to
bottom).  Keep the text EXACTLY as the OCR produced it — do NOT correct,
complete, or normalize anything from memory; only reorder lines and drop
obvious non-text artifacts.

Return ONLY the reconstructed transcription — no labels, no commentary.

OCR text:
{ocr_text}"""


async def _call_ocr_extraction(flat_text: str) -> Optional[str]:
    """Route flat OCR output through the shared text-only extraction LLM.

    The extraction model can be a Gemini model (default) or an LM Studio
    model when ``_segmentation_base_url`` is set — mirroring the Talmud
    eval's ``--segmentation_model`` / ``--segmentation_base_url`` options.

    :param flat_text: Raw OCR text (possibly out of reading order)
    :type flat_text: str
    :return: Reconstructed reading-order transcription, or None on failure
    :rtype: Optional[str]
    """
    prompt = _OCR_EXTRACTION_PROMPT.format(ocr_text=flat_text[:15_000])
    if _segmentation_base_url:
        return await lm_studio_complete_text(
            model_id=_segmentation_model,
            prompt=prompt,
            base_url=_segmentation_base_url,
        )
    return await call_gemini_text_only(_segmentation_model, prompt, timeout=90)


def _fragment_prompt(doc_id: str) -> str:
    """Build the standard fragment transcription prompt for extra models.

    Mirrors the Gemini Flash prompt in genizah_fragment_agent so all models
    are benchmarked with identical instructions.

    :param doc_id: Document identifier (used to derive the collection series)
    :type doc_id: str
    :return: Transcription prompt
    :rtype: str
    """
    series = _series_from_doc_id(doc_id)
    return f"""This is a manuscript from the Cairo Genizah ({series} collection).
The text may be in Hebrew, Aramaic, or Judaeo-Arabic (Arabic written in Hebrew script).

Transcribe the text in this image exactly as written. Do not normalize or correct the text.
Preserve all vocalization marks (nikud) and line structure.
Mark damaged or unclear characters with [?].

Return ONLY the transcription with no commentary."""


# ============================================================================
# Metrics Calculation
# ============================================================================

def calculate_similarity(reference: str, hypothesis: str) -> float:
    """Character-level similarity ratio (kept for W&B diagnostics)."""
    return SequenceMatcher(None, reference, hypothesis).ratio()


def evaluate_transcription(ground_truth: str, hypothesis: str, model_name: str) -> dict:
    """Calculate all benchmark metrics for one model output.

    Uses Levenshtein CER/WER (HTR community standard for published numbers).
    NOT clamped at 1.0 — values > 1 document decoder collapse.
    """
    gt_clean = ' '.join(ground_truth.split())
    hyp_clean = ' '.join(hypothesis.split())

    cer_s, cer_l = cer_pair(hyp_clean, gt_clean)
    wer_s, wer_l = wer_pair(hyp_clean, gt_clean)

    return {
        'model': model_name,
        'cer': cer_s,
        'cer_lenient': cer_l,
        'wer': wer_s,
        'wer_lenient': wer_l,
        'char_count_ratio': char_count_ratio(hypothesis, ground_truth),
        'similarity': calculate_similarity(gt_clean, hyp_clean),
        'exact_match': gt_clean == hyp_clean,
        'char_count': len(hypothesis),
        'gt_char_count': len(ground_truth),
        'char_diff': abs(len(hypothesis) - len(ground_truth)),
    }


# ============================================================================
# Ground Truth Extraction
# ============================================================================

def extract_ground_truth(transcriptions) -> str:
    """Extract ground truth text from transcription data"""
    if not transcriptions:
        return ""

    # Numeric keys sort numerically ("2" < "10"); non-numeric keys sort
    # lexically after them instead of all collapsing to position 0.
    def _line_key(x: str):
        return (0, int(x), "") if x.isdigit() else (1, 0, x)

    # Format 1: Dictionary with string keys
    if isinstance(transcriptions, dict):
        sorted_keys = sorted(transcriptions.keys(), key=_line_key)
        lines = [transcriptions[k] for k in sorted_keys]
        return '\n'.join(lines)

    # Format 2: List of transcription objects
    if isinstance(transcriptions, list):
        if not transcriptions:
            return ""

        first_transcription = transcriptions[0]

        if len(transcriptions) > 1:
            print(f"  ⚠️  {len(transcriptions)} transcription versions found — using the first")

        if isinstance(first_transcription, dict) and 'lines' in first_transcription:
            lines_dict = first_transcription['lines']
            sorted_keys = sorted(lines_dict.keys(), key=_line_key)
            lines = [lines_dict[k] for k in sorted_keys]
            return '\n'.join(lines)

    return ""


def has_ground_truth(metadata: dict) -> bool:
    """Check if document has valid ground truth transcription."""
    transcriptions = metadata.get('transcriptions', [])
    if not transcriptions:
        return False

    ground_truth = extract_ground_truth(transcriptions)
    return len(ground_truth.strip()) > 10


# ============================================================================
# Output Saving
# ============================================================================

def save_raw_output(doc_id: str, model_name: str, text: str, ground_truth: str = None):
    """Save raw transcription output to file"""
    output_dir = EvalConfig.RAW_OUTPUTS_DIR / doc_id
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


def save_incremental_result(result: dict, batch_num: int = None):
    """Append result to incremental JSONL file"""
    if batch_num:
        results_file = EvalConfig.RESULTS_DIR / f"batch_{batch_num}_incremental.jsonl"
    else:
        results_file = EvalConfig.INCREMENTAL_RESULTS_FILE

    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False, default=str) + '\n')


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
# Batch Manager
# ============================================================================

class BatchManager:
    """Manage batch processing and resume capability"""

    def __init__(self, total_docs: int):
        self.total_docs = total_docs
        self.progress = self.load_batch_progress()

    @staticmethod
    def load_batch_progress() -> dict:
        """Load batch processing progress"""
        if EvalConfig.BATCH_TRACKING_FILE.exists():
            with open(EvalConfig.BATCH_TRACKING_FILE, 'r') as f:
                progress = json.load(f)
            progress.setdefault('completed_docs', [])
            return progress
        return {
            'completed_batches': [],
            'completed_docs': [],
            'failed_docs': [],
            'last_processed_doc': None,
            'total_processed': 0
        }

    @staticmethod
    def save_batch_progress(progress: dict):
        """Save batch processing progress"""
        EvalConfig.BATCH_TRACKING_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(EvalConfig.BATCH_TRACKING_FILE, 'w') as f:
            json.dump(progress, f, indent=2)

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

    def get_next_uncompleted_batch(self, batch_size: int) -> int:
        """Find the next uncompleted batch start index"""
        # Get all completed batch numbers
        completed = sorted(self.progress['completed_batches'])

        # Find first missing batch
        expected_batch = 1
        for batch_num in completed:
            if batch_num == expected_batch:
                expected_batch += 1
            else:
                break

        # Convert batch number to start index
        start_doc = (expected_batch - 1) * batch_size

        # Make sure we don't go past the total
        if start_doc >= self.total_docs:
            return None

        return start_doc

    def is_doc_completed(self, doc_id: str) -> bool:
        """Check whether a document already succeeded in a previous run.

        :param doc_id: Document identifier
        :type doc_id: str
        :return: True if the document was already evaluated successfully
        :rtype: bool
        """
        return doc_id in self.progress['completed_docs']

    def mark_doc_completed(self, doc_id: str):
        """Record a successfully evaluated document.

        Also clears any earlier failure entries for the same document, so
        the failure list only reflects documents that still need attention.

        :param doc_id: Document identifier
        :type doc_id: str
        """
        if doc_id not in self.progress['completed_docs']:
            self.progress['completed_docs'].append(doc_id)
        self.progress['failed_docs'] = [
            f for f in self.progress['failed_docs'] if f['doc_id'] != doc_id
        ]
        self.progress['last_processed_doc'] = doc_id
        self.progress['total_processed'] = len(self.progress['completed_docs'])
        self.save_batch_progress(self.progress)

    def mark_batch_completed(self, batch_num: int, doc_ids: list):
        """Mark a batch completed — only if EVERY document in it succeeded.

        Batches with failures stay uncompleted so ``--auto-next`` revisits
        them; already-succeeded documents are skipped per-doc, so only the
        failed ones are retried (no API tokens wasted on re-runs).

        :param batch_num: 1-based batch number
        :type batch_num: int
        :param doc_ids: All document IDs belonging to this batch
        :type doc_ids: list
        """
        if all(d in self.progress['completed_docs'] for d in doc_ids):
            if batch_num not in self.progress['completed_batches']:
                self.progress['completed_batches'].append(batch_num)
        else:
            remaining = [d for d in doc_ids if d not in self.progress['completed_docs']]
            print(f"\n⚠️  Batch {batch_num} left uncompleted — {len(remaining)} doc(s) "
                  f"still failing and will be retried on the next run: {remaining[:5]}")
        self.progress['total_processed'] = len(self.progress['completed_docs'])
        self.save_batch_progress(self.progress)

    def mark_doc_failed(self, doc_id: str, error: str):
        """Track failed documents (one entry per document — latest error wins)."""
        self.progress['failed_docs'] = [
            f for f in self.progress['failed_docs'] if f['doc_id'] != doc_id
        ]
        self.progress['failed_docs'].append({
            'doc_id': doc_id,
            'error': error,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        self.save_batch_progress(self.progress)

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
# W&B Logging
# ============================================================================

def log_to_wandb(state: TranscriptionState, run_name: str) -> tuple:
    """Log comprehensive metrics and return table rows for batch logging."""

    metrics = {
        'doc_id': state['doc_id'],
        'processing_time_total': state['processing_time'],
        # Benchmark context fields (always False — no metadata given to models)
        'catalog_metadata_provided': False,
        'corpus': state.get('_benchmark_corpus', 'genizah'),
        'track': state.get('_benchmark_track', 'track1'),
        'script_type': state.get('_benchmark_script_type', 'unknown'),
    }

    # Per-model scalars (CER, WER, char_count_ratio, lenient variants)
    for model_key in scored_model_keys():
        model_metrics = state.get(f'{model_key}_metrics')
        if model_metrics:
            for key, value in model_metrics.items():
                if isinstance(value, (int, float)):
                    metrics[f'{model_key}/{key}'] = value
        # Failure mode flags as a comma-separated string
        flags = state.get(f'_{model_key}_failure_flags', [])
        if flags:
            metrics[f'{model_key}/failure_mode_flags'] = ','.join(flags)

    # Consensus metrics
    if state.get('consensus_metrics'):
        for key, value in state['consensus_metrics'].items():
            if isinstance(value, (int, float)):
                metrics[f'consensus/{key}'] = value

    metrics['consensus/strategy'] = state['consensus_strategy']
    metrics['consensus/num_disagreements'] = len(state['disagreements'])
    metrics['consensus/needs_review'] = state['needs_review']

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
        state.get('ground_truth', ''),
    ]
    # One text column per scored model (VLMs + OCR→LLM pipelines) — columns
    # added in the same order by main() when building the W&B table.
    for key in scored_model_keys():
        text_comparison_row.append((state.get(f'{key}_result') or {}).get('text', ''))
    # Raw OCR text is shown for inspection only — never scored (line order
    # is not reading order).
    for ocr_key in raw_ocr_keys():
        text_comparison_row.append((state.get(f'{ocr_key}_result') or {}).get('text', ''))
    text_comparison_row += [
        state['final_transcription'],
        state['consensus_strategy']
    ]

    # ========================================================================
    # METRICS ROWS
    # ========================================================================

    metrics_rows = []

    _FIXED_DISPLAY_NAMES = {
        'gemini_flash': 'Gemini Flash',
        'gemini_pro': 'Gemini Pro',
        'claude': f'Claude ({ClaudeConfig.DEFAULT_MODEL})',
        'vision_ocr_seg': 'Vision OCR → LLM extraction',
        'kraken_seg': 'Kraken → LLM extraction',
    }
    model_display_names = [
        (key, _FIXED_DISPLAY_NAMES.get(key, key)) for key in scored_model_keys()
    ]
    model_display_names.append(('consensus', 'Consensus'))

    for model_key, model_name in model_display_names:
        metrics_key = f'{model_key}_metrics'
        model_metrics = state.get(metrics_key)

        if model_metrics:
            metrics_rows.append([
                state['doc_id'],
                model_name,
                model_metrics.get('cer'),
                model_metrics.get('cer_lenient'),
                model_metrics.get('wer'),
                model_metrics.get('wer_lenient'),
                model_metrics.get('char_count_ratio'),
                model_metrics.get('similarity'),
                model_metrics.get('char_count'),
                model_metrics.get('gt_char_count'),
                model_metrics.get('char_diff'),
                model_metrics.get('processing_time'),
                model_metrics.get('exact_match', False),
                ','.join(state.get(f'_{model_key}_failure_flags', [])),
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
            state.get('ground_truth', ''),
            ocr_text,
            flash_text,
            pro_text,
            state['final_transcription'],
            state['consensus_strategy']
        )

        # Save HTML file
        html_path = EvalConfig.RAW_OUTPUTS_DIR / state['doc_id'] / "comparison.html"
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(comparison_html, encoding='utf-8')

        # Log to W&B
        wandb.log({f"comparison_html/{state['doc_id']}": wandb.Html(comparison_html)})

    return text_comparison_row, metrics_rows, analysis_row


# ============================================================================
# Main Evaluation Function
# ============================================================================

async def evaluate_document(doc_id: str, metadata: dict, wandb_run, batch_num: int = None) -> Optional[Dict]:
    """Evaluate a single document with ground truth metrics."""

    images = metadata.get('images', [])
    if not images:
        print(f"  ⚠️  No images found - skipping")
        return None

    # Download image
    image_path = await download_image(images[0], doc_id, EvalConfig.IMAGES_DIR)
    if not image_path.exists():
        print(f"  ⚠️  Image missing: {image_path} - skipping")
        return None

    # Extract ground truth
    ground_truth = extract_ground_truth(metadata.get('transcriptions', []))
    if not ground_truth:
        print(f"  ⚠️  No ground truth found - skipping")
        return None

    print(f"  📝 Ground truth: {len(ground_truth)} chars")

    # Save ground truth
    save_raw_output(doc_id, 'ground_truth', ground_truth, ground_truth)

    # Run transcription
    final_state = await transcribe_document(doc_id, metadata, image_path, ground_truth)

    # Add benchmark context fields
    final_state['_benchmark_corpus'] = 'genizah'
    final_state['_benchmark_track'] = 'track1'
    final_state['_benchmark_script_type'] = metadata.get('script_type', 'unknown')

    # ── Extra benchmark models (same prompt as Gemini Flash) ────────────────
    prompt = _fragment_prompt(doc_id)

    if _lm_studio_models:
        print(f"  🖥  LM Studio: {', '.join(model_key_from_id(m) for m in _lm_studio_models)}...")
        t0 = time.time()
        lm_results = await lm_studio_transcribe_batch(
            model_ids=_lm_studio_models,
            image_path=str(image_path),
            prompt=prompt,
            base_url=_lm_studio_base_url,
        )
        lm_elapsed = time.time() - t0
        for model_id, text in lm_results.items():
            key = model_key_from_id(model_id)
            if text:
                final_state[f'{key}_result'] = {
                    'text': text,
                    'model': key,
                    'char_count': len(text),
                    'processing_time': lm_elapsed,
                }
                final_state['model_times'][key] = lm_elapsed
            else:
                final_state[f'{key}_result'] = None
                print(f"    ✗ [{key}] no output")

    if _use_claude:
        print(f"  🤖 Claude ({ClaudeConfig.DEFAULT_MODEL})...")
        t0 = time.time()
        claude_text = await transcribe_with_claude(str(image_path), prompt)
        claude_elapsed = time.time() - t0
        if claude_text:
            final_state['claude_result'] = {
                'text': claude_text,
                'model': ClaudeConfig.DEFAULT_MODEL,
                'char_count': len(claude_text),
                'processing_time': claude_elapsed,
            }
            final_state['model_times']['claude'] = claude_elapsed
            print(f"    ✓ Extracted {len(claude_text)} chars in {claude_elapsed:.1f}s")
        else:
            final_state['claude_result'] = None

    # ── OCR → text-only LLM extraction pipelines ────────────────────────────
    # Raw OCR (Vision, Kraken) is NEVER scored — line order is not reading
    # order.  Both engines are routed through the SAME extraction LLM and the
    # end-to-end result is scored, making the OCR+LLM pipelines directly
    # comparable to the end-to-end VLMs.  Raw output is saved so OCR-model
    # failures can be told apart from extraction-LLM failures.
    for ocr_key in raw_ocr_keys():
        raw_text = (final_state.get(f'{ocr_key}_result') or {}).get('text', '')
        if not raw_text:
            final_state[f'{ocr_key}_seg_result'] = None
            continue
        save_raw_output(doc_id, f'{ocr_key}_raw', raw_text)
        print(f"  🔀 {ocr_key} → {_segmentation_model} extraction...")
        t0 = time.time()
        seg_text = await _call_ocr_extraction(raw_text)
        seg_elapsed = time.time() - t0
        if seg_text:
            final_state[f'{ocr_key}_seg_result'] = {
                'text': seg_text,
                'model': f'{ocr_key}+{_segmentation_model}',
                'char_count': len(seg_text),
                'processing_time': seg_elapsed,
            }
            final_state['model_times'][f'{ocr_key}_seg'] = seg_elapsed
            print(f"    ✓ [{ocr_key}_seg] {len(seg_text)} chars in {seg_elapsed:.1f}s")
        else:
            final_state[f'{ocr_key}_seg_result'] = None
            print(f"    ✗ [{ocr_key}_seg] extraction returned nothing")

    # Compute evaluation metrics and failure flags for each model
    for model_key in scored_model_keys():
        result = final_state.get(f'{model_key}_result')
        if result:
            metrics = evaluate_transcription(ground_truth, result['text'], result['model'])
            metrics['processing_time'] = result.get('processing_time', 0.0)
            final_state[f'{model_key}_metrics'] = metrics
            final_state[f'_{model_key}_failure_flags'] = flag_failure_modes(
                result['text'], ground_truth, catalog_metadata_provided=False
            )

            save_raw_output(doc_id, model_key, result['text'])

            print(f"    {result['model']}: CER={metrics['cer']:.3f} "
                  f"(lenient={metrics['cer_lenient']:.3f}), "
                  f"ratio={metrics['char_count_ratio']:.2f}")

    # Compute consensus metrics
    if final_state['final_transcription']:
        consensus_metrics = evaluate_transcription(
            ground_truth,
            final_state['final_transcription'],
            f"consensus_{final_state['consensus_strategy']}"
        )
        final_state['consensus_metrics'] = consensus_metrics

        # Save consensus output
        save_raw_output(doc_id, 'consensus', final_state['final_transcription'])

        print(f"    Consensus: CER={consensus_metrics['cer']:.3f}, "
              f"Strategy={final_state['consensus_strategy']}")

    # Log to W&B and get table rows
    text_row, metrics_rows, analysis_row = log_to_wandb(final_state, wandb_run.name)

    # Store rows for batch logging
    final_state['_text_comparison_row'] = text_row
    final_state['_metrics_rows'] = metrics_rows
    final_state['_analysis_row'] = analysis_row

    # Save incrementally
    save_incremental_result(final_state, batch_num)

    return final_state


# ============================================================================
# Main Pipeline
# ============================================================================

async def main(
        start_doc: Optional[int] = None,
        num_docs: Optional[int] = None,
        auto_next: bool = True,
        lm_studio_models: Optional[List[str]] = None,
        lm_studio_base_url: str = EvalConfig.LM_STUDIO_BASE_URL,
        use_claude: bool = False,
        claude_model: Optional[str] = None,
        segmentation_model: str = AgentConfig.GEMINI_FLASH_MODEL,
        segmentation_base_url: Optional[str] = None,
):
    """Main evaluation pipeline with batch processing support.

    :param start_doc: Starting document index (0-based). If None and
        auto_next=True, finds next uncompleted batch automatically.
    :type start_doc: Optional[int]
    :param num_docs: Number of documents to process. If None, uses
        EvalConfig.BATCH_SIZE.
    :type num_docs: Optional[int]
    :param auto_next: If True and start_doc is None, automatically find the
        next uncompleted batch.
    :type auto_next: bool
    :param lm_studio_models: LM Studio VLM model IDs to benchmark (e.g.
        ["google/gemma-4-31b-qat"]). Empty list disables LM Studio.
    :type lm_studio_models: Optional[List[str]]
    :param lm_studio_base_url: LM Studio API base URL.
    :type lm_studio_base_url: str
    :param use_claude: Include Claude in the benchmark. Off by default —
        tokens are expensive.
    :type use_claude: bool
    :param claude_model: Override the Claude model ID.
    :type claude_model: Optional[str]
    :param segmentation_model: Text-only LLM that reconstructs reading-order
        text from raw OCR output. Both Vision OCR and Kraken are routed
        through this same model.
    :type segmentation_model: str
    :param segmentation_base_url: If set, the segmentation model is served
        via LM Studio at this URL instead of Gemini.
    :type segmentation_base_url: Optional[str]
    """
    global _lm_studio_models, _lm_studio_base_url, _use_claude
    global _segmentation_model, _segmentation_base_url

    print("=" * 80)
    print("Cairo Genizah Transcription Evaluation - BATCH MODE")
    print("=" * 80)

    # ── Extra model configuration ────────────────────────────────────────────
    _lm_studio_models = list(lm_studio_models) if lm_studio_models else []
    _lm_studio_base_url = lm_studio_base_url
    _use_claude = use_claude
    _segmentation_model = segmentation_model
    _segmentation_base_url = segmentation_base_url
    if claude_model:
        ClaudeConfig.DEFAULT_MODEL = claude_model

    if _lm_studio_models:
        print(f"\n🖥  Checking LM Studio at {_lm_studio_base_url} ...")
        try:
            available = await check_lm_studio_health(_lm_studio_base_url)
            print(f"   Available models: {available}")
            missing = [m for m in _lm_studio_models if m not in available]
            if missing:
                print(
                    f"   ⚠️  These models are not loaded in LM Studio: {missing}\n"
                    f"   Load them in LM Studio or they will produce empty results."
                )
        except RuntimeError as exc:
            print(f"   ⚠️  {exc}\n   Disabling LM Studio for this run.")
            _lm_studio_models = []

    if _use_claude and not os.getenv("ANTHROPIC_API_KEY"):
        print(
            "\n⚠️  ANTHROPIC_API_KEY is not set — Claude auth will fall back to "
            "an `ant auth login` profile if one exists."
        )

    # Setup
    EvalConfig.IMAGES_DIR.mkdir(exist_ok=True)
    EvalConfig.RESULTS_DIR.mkdir(exist_ok=True)
    EvalConfig.RAW_OUTPUTS_DIR.mkdir(exist_ok=True)

    # Verify API keys
    assert AgentConfig.GEMINI_API_KEY, "GEMINI_API_KEY environment variable not set"
    assert AgentConfig.GOOGLE_VISION_CREDENTIALS, "GOOGLE_APPLICATION_CREDENTIALS not set"

    # Load catalog
    print(f"\n📚 Loading catalog...")
    with open(EvalConfig.CATALOG_PATH) as f:
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
    if num_docs is None:
        num_docs = EvalConfig.BATCH_SIZE

    if start_doc is None and auto_next:
        start_doc = batch_manager.get_next_uncompleted_batch(num_docs)
        if start_doc is None:
            print("\n🎉 All batches completed!")
            print(batch_manager.generate_progress_report())
            return
        print(f"\n🔍 Auto-detected next uncompleted batch starting at doc {start_doc}")
    elif start_doc is None:
        start_doc = 0

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
    est_time_per_doc = 90  # ~1.5 minutes per doc with FIXED async API
    est_total_time = len(batch_docs) * est_time_per_doc / 60
    print(f"\n⏱️  Estimated time: {est_total_time:.0f} minutes ({est_total_time / 60:.1f} hours)")

    # Config summary
    print(f"\n⚙️  Configuration:")
    print(f"   Vision OCR: {'✓' if AgentConfig.USE_VISION_OCR else '✗'}")
    print(f"   Kraken: {'✓' if AgentConfig.USE_KRAKEN else '✗'}")
    print(
        f"   Gemini Flash: {'✓' if AgentConfig.USE_GEMINI_FLASH else '✗'} (timeout: {AgentConfig.GEMINI_FLASH_TIMEOUT}s)")
    print(f"   Gemini Pro: {'✓' if AgentConfig.USE_GEMINI_PRO else '✗'} (timeout: {AgentConfig.GEMINI_PRO_TIMEOUT}s)")
    print(f"   Analysis: {'✓' if AgentConfig.USE_ANALYSIS else '✗'}")
    print(f"   LM Studio: {', '.join(_lm_studio_models) if _lm_studio_models else '✗'}")
    print(f"   Claude: {ClaudeConfig.DEFAULT_MODEL if _use_claude else '✗'}")
    print(f"   OCR extraction model: {_segmentation_model}"
          + (f" (LM Studio @ {_segmentation_base_url})" if _segmentation_base_url else " (Gemini)"))
    print(f"   Delay between docs: {EvalConfig.DELAY_BETWEEN_DOCS}s")

    # Initialize W&B
    wandb_run = wandb.init(
        project=EvalConfig.WANDB_PROJECT,
        entity=EvalConfig.WANDB_ENTITY,
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
                    ("google_vision_ocr", AgentConfig.USE_VISION_OCR),
                    ("gemini_3_flash", AgentConfig.USE_GEMINI_FLASH),
                    ("gemini_3_pro", AgentConfig.USE_GEMINI_PRO)
                ] if enabled
            ] + _lm_studio_models + ([ClaudeConfig.DEFAULT_MODEL] if _use_claude else []),
            "flash_timeout": AgentConfig.GEMINI_FLASH_TIMEOUT,
            "pro_timeout": AgentConfig.GEMINI_PRO_TIMEOUT,
            "lm_studio_models": _lm_studio_models,
            "lm_studio_base_url": _lm_studio_base_url if _lm_studio_models else None,
            "claude_model": ClaudeConfig.DEFAULT_MODEL if _use_claude else None,
            "ocr_segmentation_model": _segmentation_model,
            "ocr_segmentation_base_url": _segmentation_base_url,
        }
    )

    # Process documents
    results = []
    text_comparison_rows = []
    all_metrics_rows = []
    analysis_rows = []
    failed_docs = []

    for i, (doc_id, metadata) in enumerate(batch_docs.items(), 1):
        global_index = start_doc + i

        # Per-document resume: skip anything that already succeeded so
        # failed-doc retries never re-pay API tokens for completed docs.
        if batch_manager.is_doc_completed(doc_id):
            print(f"\n[{i}/{len(batch_docs)}] ⏭️  {doc_id} already completed — skipping")
            continue

        print(f"\n[{i}/{len(batch_docs)}] (Global: {global_index}/{len(eval_docs)})", end=" ")

        try:
            result = await evaluate_document(doc_id, metadata, wandb_run, batch_info['batch_num'])
            if result:
                results.append(result)
                text_comparison_rows.append(result['_text_comparison_row'])
                all_metrics_rows.extend(result['_metrics_rows'])
                if result.get('_analysis_row'):
                    analysis_rows.append(result['_analysis_row'])
                batch_manager.mark_doc_completed(doc_id)
            else:
                failed_docs.append({'doc_id': doc_id, 'reason': 'No result returned'})
                batch_manager.mark_doc_failed(doc_id, 'No result returned')

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"\n  ❌ FATAL ERROR: {error_msg}")
            failed_docs.append({'doc_id': doc_id, 'reason': error_msg})
            batch_manager.mark_doc_failed(doc_id, error_msg)

        # Rate limit protection
        if i < len(batch_docs):
            print(f"\n  ⏳ Cooling down for {EvalConfig.DELAY_BETWEEN_DOCS}s...")
            await asyncio.sleep(EvalConfig.DELAY_BETWEEN_DOCS)

    # ========================================================================
    # LOG BATCH TABLES TO W&B
    # ========================================================================

    print("\n📊 Logging batch tables to W&B...")

    if text_comparison_rows:
        text_table = wandb.Table(
            columns=[
                "fragment_id", "image", "ground_truth",
                *scored_model_keys(),
                # raw OCR shown for inspection only — no metrics
                *[f"{k}_raw" for k in raw_ocr_keys()],
                "consensus", "consensus_strategy"
            ],
            data=text_comparison_rows
        )
        wandb.log({f"batch_{batch_info['batch_num']}_text_comparison": text_table})

    if all_metrics_rows:
        metrics_table = wandb.Table(
            columns=[
                "fragment_id", "model",
                "cer", "cer_lenient",
                "wer", "wer_lenient",
                "char_count_ratio", "similarity",
                "char_count", "gt_char_count", "char_diff",
                "processing_time_sec", "exact_match",
                "failure_mode_flags",
            ],
            data=all_metrics_rows
        )
        wandb.log({f"batch_{batch_info['batch_num']}_metrics": metrics_table})

    if analysis_rows:
        analysis_table = wandb.Table(
            columns=[
                "fragment_id", "catalog_description", "content_summary",
                "translation_sample", "coherence_assessment", "catalog_alignment",
                "recommended_model", "reasoning", "key_observations", "analysis_time_sec"
            ],
            data=analysis_rows
        )
        wandb.log({f"batch_{batch_info['batch_num']}_analysis": analysis_table})

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
        for model_name in scored_model_keys():
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
    batch_output_file = EvalConfig.RESULTS_DIR / f"batch_{batch_info['batch_num']}_results.json"
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
    print(f"📊 W&B dashboard: {wandb_run.url}")

    # Mark batch as completed
    batch_manager.mark_batch_completed(batch_info['batch_num'], batch_doc_ids)

    # Show overall progress
    print("\n" + batch_manager.generate_progress_report())

    wandb.finish()

    # Suggest next batch
    if batch_info['end_doc'] < len(eval_docs):
        next_start = batch_manager.get_next_uncompleted_batch(num_docs)
        if next_start is not None:
            print(f"\n💡 Next uncompleted batch:")
            print(f"   python genizah_fragment_eval.py --auto-next")
            print(f"   (or manually: --start-doc {next_start} --num-docs {num_docs})")


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
        default=None,
        help='Starting document index (0-based). If not specified with --auto-next, finds next uncompleted batch.'
    )
    parser.add_argument(
        '--num-docs',
        type=int,
        default=None,
        help=f'Number of documents to process. Default: {EvalConfig.BATCH_SIZE}'
    )
    parser.add_argument(
        '--auto-next',
        action='store_true',
        default=True,
        help='Automatically find and process next uncompleted batch (default: True)'
    )
    parser.add_argument(
        '--no-auto-next',
        dest='auto_next',
        action='store_false',
        help='Disable auto-finding next batch'
    )
    parser.add_argument(
        '--lm-studio-models',
        type=lambda s: [m.strip() for m in s.split(',') if m.strip()],
        default=EvalConfig.LM_STUDIO_MODELS,
        metavar='MODEL[,MODEL,...]',
        help='LM Studio VLM model IDs to benchmark (comma-separated). '
             f'Default: {",".join(EvalConfig.LM_STUDIO_MODELS)}'
    )
    parser.add_argument(
        '--lm-studio-base-url',
        default=EvalConfig.LM_STUDIO_BASE_URL,
        help='LM Studio API base URL'
    )
    parser.add_argument(
        '--skip-lm-studio',
        action='store_true',
        help='Disable LM Studio models for this run'
    )
    parser.add_argument(
        '--use-claude',
        action='store_true',
        help='Include Claude in the benchmark (off by default — tokens are expensive)'
    )
    parser.add_argument(
        '--claude-model',
        default=None,
        help=f'Claude model ID (default: {ClaudeConfig.DEFAULT_MODEL})'
    )
    parser.add_argument(
        '--segmentation-model',
        default=AgentConfig.GEMINI_FLASH_MODEL,
        help='Text-only LLM that reconstructs reading-order text from raw OCR '
             'output (Vision OCR and Kraken both use it). Default: Gemini Flash. '
             'Set to an LM Studio model ID and provide --segmentation-base-url '
             'to use an open-source text model instead.'
    )
    parser.add_argument(
        '--segmentation-base-url',
        default=None,
        help='If set, the segmentation model is served via LM Studio at this '
             'URL. Leave unset to use Gemini (default).'
    )

    args = parser.parse_args()

    # Run evaluation
    asyncio.run(main(
        start_doc=args.start_doc,
        num_docs=args.num_docs,
        auto_next=args.auto_next,
        lm_studio_models=[] if args.skip_lm_studio else args.lm_studio_models,
        lm_studio_base_url=args.lm_studio_base_url,
        use_claude=args.use_claude,
        claude_model=args.claude_model,
        segmentation_model=args.segmentation_model,
        segmentation_base_url=args.segmentation_base_url,
    ))