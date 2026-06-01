"""
Cairo Genizah Transcription Evaluation Pipeline
Batch processing with metrics, W&B logging, and auto-resume
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from difflib import SequenceMatcher
import wandb

# Import core agent
from src.models.llm.genizah_fragment_agent import (
    AgentConfig,
    TranscriptionState,
    transcribe_document,
    download_image
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
                return json.load(f)
        return {
            'completed_batches': [],
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

    def mark_batch_completed(self, batch_num: int, doc_ids: list):
        """Mark batch as completed"""
        if batch_num not in self.progress['completed_batches']:
            self.progress['completed_batches'].append(batch_num)
        self.progress['total_processed'] += len(doc_ids)
        self.progress['last_processed_doc'] = doc_ids[-1] if doc_ids else None
        self.save_batch_progress(self.progress)

    def mark_doc_failed(self, doc_id: str, error: str):
        """Track failed documents"""
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
    for model_key in ['vision_ocr', 'gemini_flash', 'gemini_pro']:
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

    # Compute evaluation metrics and failure flags for each model
    for model_key in ['vision_ocr', 'gemini_flash', 'gemini_pro']:
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
        auto_next: bool = True
):
    """Main evaluation pipeline with batch processing support.

    Args:
        start_doc: Starting document index (0-based). If None and auto_next=True,
                   finds next uncompleted batch automatically.
        num_docs: Number of documents to process. If None, uses EvalConfig.BATCH_SIZE
        auto_next: If True and start_doc is None, automatically find next uncompleted batch
    """

    print("=" * 80)
    print("Cairo Genizah Transcription Evaluation - BATCH MODE")
    print("=" * 80)

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
            ],
            "flash_timeout": AgentConfig.GEMINI_FLASH_TIMEOUT,
            "pro_timeout": AgentConfig.GEMINI_PRO_TIMEOUT,
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
        print(f"\n[{i}/{len(batch_docs)}] (Global: {global_index}/{len(eval_docs)})", end=" ")

        try:
            result = await evaluate_document(doc_id, metadata, wandb_run, batch_info['batch_num'])
            if result:
                results.append(result)
                text_comparison_rows.append(result['_text_comparison_row'])
                all_metrics_rows.extend(result['_metrics_rows'])
                if result.get('_analysis_row'):
                    analysis_rows.append(result['_analysis_row'])
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
                "fragment_id", "image", "ground_truth", "vision_ocr",
                "gemini_flash", "gemini_pro", "consensus", "consensus_strategy"
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

    args = parser.parse_args()

    # Run evaluation
    asyncio.run(main(
        start_doc=args.start_doc,
        num_docs=args.num_docs,
        auto_next=args.auto_next
    ))