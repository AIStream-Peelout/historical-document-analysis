#!/usr/bin/env python3
"""
Batch Pipeline Runner — Cairo Genizah Academic Literature

Scans every PDF under academic_literature/, resolves its metadata JSON,
and resumes the processing pipeline from wherever it last stopped.

Each file is checked in order:
  1. OCR         → skipped if {stem}_ocr_results.json + {stem}_images/ exist
  2. Structured  → skipped if any *_structured.json found anywhere under the output dir
  3. Enhanced    → skipped unless --rerun-enhanced is set, or _enhanced.json is missing

Usage examples
--------------
# Dry-run: show status of every file without processing anything
python batch_pipeline_runner.py --dry-run

# Resume all incomplete files
python batch_pipeline_runner.py

# Force re-run the enhanced step on everything (e.g. after prompt update)
python batch_pipeline_runner.py --rerun-enhanced

# Process only a specific subdirectory
python batch_pipeline_runner.py --dir kettubah_palestine

# Skip any file that hasn't been OCR'd yet (avoids GCP costs during structured-only runs)
python batch_pipeline_runner.py --skip-unocred
"""

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

import dotenv

# ---------------------------------------------------------------------------
# Bootstrap paths
# ---------------------------------------------------------------------------
project_root = Path(__file__).parent.parent.parent.parent.parent
data_root = project_root / "src" / "datasets" / "raw_data" / "cairo_genizah" / "academic_literature"
sys.path.append(str(project_root))
dotenv.load_dotenv(project_root / ".env")

from src.models.ocr.book_ocr_service import BookOCRService
from src.models.llm.structured_json_llm import StructuredJSONLLM
from src.models.llm.secondary_llm_processing import SecondaryLLMProcessor
from src.models.llm.add_full_text_to_structured import add_full_text_to_structured

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metadata discovery
# ---------------------------------------------------------------------------

def find_metadata(pdf_path: Path) -> Optional[Path]:
    """Return the best-matching metadata JSON for *pdf_path*, or None.

    Search order (all within the same directory as the PDF):
    1. {stem}_metadata.json          — most common convention
    2. {stem}.json                   — some Rylands articles use bare stem
    3. Single *_metadata.json file   — shared metadata (e.g. friedman_metadata.json)
    4. Walk parent dirs up to data_root for a shared *_metadata.json
    """
    stem = pdf_path.stem
    parent = pdf_path.parent

    # 1. exact stem match
    for candidate in (
        parent / f"{stem}_metadata.json",
        parent / f"{stem}.json",
    ):
        if candidate.exists():
            return candidate

    # 2. any *_metadata.json in the same directory
    meta_files = [f for f in parent.glob("*_metadata.json") if f.is_file()]
    if len(meta_files) == 1:
        return meta_files[0]
    if len(meta_files) > 1:
        # Pick the one whose stem most closely matches — shortest edit distance heuristic
        def _score(p: Path) -> int:
            # number of chars in stem that appear in metadata filename
            return sum(c in p.stem for c in stem.lower())
        return max(meta_files, key=_score)

    # 3. walk up to data_root looking for a shared metadata file
    check = parent.parent
    while check != data_root.parent:
        meta_files = [f for f in check.glob("*_metadata.json") if f.is_file()]
        if meta_files:
            return meta_files[0]
        if check == data_root:
            break
        check = check.parent

    return None


# ---------------------------------------------------------------------------
# Per-file status helpers
# ---------------------------------------------------------------------------

def _find_structured_files(output_dir: Path) -> list[Path]:
    """Recursively find all *_structured.json files under *output_dir*."""
    if not output_dir.exists():
        return []
    return list(output_dir.rglob("*_structured.json"))


def get_file_status(pdf_path: Path) -> dict:
    """Return a status dict for a PDF describing which pipeline steps are done."""
    stem = pdf_path.stem
    # Output lives next to the PDF, in a directory named after the PDF stem
    output_dir = pdf_path.parent / stem

    ocr_json   = output_dir / f"{stem}_ocr_results.json"
    images_dir = output_dir / f"{stem}_images"
    enhanced   = output_dir / f"{stem}_enhanced.json"
    structured = _find_structured_files(output_dir)

    return {
        "pdf_path":      pdf_path,
        "output_dir":    output_dir,
        "ocr_done":      ocr_json.exists() and images_dir.exists(),
        "ocr_json":      ocr_json,
        "images_dir":    images_dir,
        "structured_done": len(structured) > 0,
        "structured_dir":  structured[0].parent if structured else None,
        "structured_count": len(structured),
        "enhanced_done": enhanced.exists(),
        "enhanced_path": enhanced,
        "metadata":      find_metadata(pdf_path),
    }


def print_status_table(statuses: list[dict]) -> None:
    """Print a compact table showing pipeline completion for every file."""
    col = 60
    header = f"{'PDF':<{col}}  OCR   STRUCT  ENHANCED  METADATA"
    print(header)
    print("-" * len(header))
    for s in statuses:
        rel = str(s["pdf_path"].relative_to(data_root))
        ocr     = "✅" if s["ocr_done"]        else "❌"
        struct  = "✅" if s["structured_done"]  else "❌"
        enh     = "✅" if s["enhanced_done"]    else "❌"
        meta    = "✅" if s["metadata"]         else "⚠️ "
        print(f"{rel:<{col}}  {ocr}     {struct}     {enh}        {meta}")


# ---------------------------------------------------------------------------
# Single-file pipeline runner
# ---------------------------------------------------------------------------

def run_file(
    status: dict,
    use_gemini: bool = True,
    rerun_enhanced: bool = False,
    skip_unocred: bool = False,
) -> str:
    """
    Run (or resume) the full pipeline for one PDF.

    Returns a short outcome string: "skipped", "completed", or "failed:<reason>".
    """
    pdf_path   : Path           = status["pdf_path"]
    output_dir : Path           = status["output_dir"]
    metadata   : Optional[Path] = status["metadata"]
    stem       : str            = pdf_path.stem

    ocr_json        = output_dir / f"{stem}_ocr_results.json"
    images_dir      = output_dir / f"{stem}_images"
    enhanced_path   = output_dir / f"{stem}_enhanced.json"

    rel = str(pdf_path.relative_to(data_root))
    logger.info(f"▶ {rel}")

    # ------------------------------------------------------------------
    # Load book metadata
    # ------------------------------------------------------------------
    book_metadata = None
    if metadata and metadata.exists():
        try:
            with open(metadata, encoding="utf-8") as f:
                book_metadata = json.load(f)
            logger.info(f"  Metadata: {metadata.name} — {book_metadata.get('title', '?')}")
        except Exception as e:
            logger.warning(f"  Could not load metadata {metadata}: {e}")
    else:
        logger.info("  No metadata file found — proceeding without book context")

    # ------------------------------------------------------------------
    # Step 1: OCR
    # ------------------------------------------------------------------
    ocr_result = None
    if status["ocr_done"]:
        logger.info(f"  ✅ OCR already done ({ocr_json.name})")
        ocr_result = {
            "ocr_results_path": str(ocr_json),
            "images_dir":       str(images_dir),
        }
    else:
        if skip_unocred:
            logger.info("  ⏭  No OCR yet and --skip-unocred set — skipping this file")
            return "skipped"

        logger.info("  Step 1: Running OCR…")
        try:
            ocr_service = BookOCRService()
            ocr_result  = ocr_service.process_pdf_to_file(
                pdf_path    = str(pdf_path),
                start_page  = 0,
                output_path = str(output_dir),
            )
            logger.info(f"  OCR complete → {ocr_result['ocr_results_path']}")
        except Exception as e:
            logger.error(f"  OCR failed: {e}")
            return f"failed:ocr:{e}"

    # ------------------------------------------------------------------
    # Step 2: Structured LLM processing
    # ------------------------------------------------------------------
    structured_dir = None
    if status["structured_done"]:
        structured_dir = status["structured_dir"]
        logger.info(f"  ✅ Structured already done ({status['structured_count']} files in {structured_dir.name})")
    else:
        logger.info("  Step 2: Running structured LLM processing…")
        try:
            structured_llm = StructuredJSONLLM(
                use_gemini    = use_gemini,
                book_metadata = book_metadata,
            )
            structured_result = structured_llm.process_from_ocr_service(
                ocr_service_result=ocr_result,
            )
            structured_dir = Path(structured_result["structured_dir"])
            logger.info(f"  Structured complete → {structured_dir}")
        except Exception as e:
            logger.error(f"  Structured LLM failed: {e}\n{traceback.format_exc()}")
            return f"failed:structured:{e}"

    if structured_dir is None or not structured_dir.exists():
        logger.error(f"  Structured directory missing after step 2: {structured_dir}")
        return "failed:structured:directory_missing"

    # ------------------------------------------------------------------
    # Step 2.5: Re-add full text (idempotent)
    # ------------------------------------------------------------------
    try:
        add_results = add_full_text_to_structured(
            structured_dir    = str(structured_dir),
            ocr_results_path  = ocr_result["ocr_results_path"],
        )
        logger.info(f"  Full text added to {add_results['files_updated']} files")
    except Exception as e:
        logger.warning(f"  add_full_text_to_structured failed (non-fatal): {e}")

    # ------------------------------------------------------------------
    # Step 3: Enhanced / secondary processing
    # ------------------------------------------------------------------
    if status["enhanced_done"] and not rerun_enhanced:
        logger.info(f"  ✅ Enhanced already done ({enhanced_path.name})")
    else:
        if rerun_enhanced and enhanced_path.exists():
            logger.info("  --rerun-enhanced set — deleting existing enhanced output")
            enhanced_path.unlink()

        logger.info("  Step 3: Running enhanced (secondary) processing…")
        try:
            gemini_key = os.environ.get("GEMINI_API_KEY")
            if gemini_key:
                processor = SecondaryLLMProcessor(
                    backend        = "gemini",
                    gemini_model   = "gemini-3-flash-preview",
                    gemini_api_key = gemini_key,
                )
            else:
                logger.warning("  GEMINI_API_KEY not set — falling back to Ollama")
                processor = SecondaryLLMProcessor()

            enhanced_data = processor.process_from_structured_dir(
                structured_dir,
                output_path=enhanced_path,
            )
            meta = enhanced_data.get("processing_metadata", {})
            people    = enhanced_data.get("people_locations", {}).get("people",    [])
            locations = enhanced_data.get("people_locations", {}).get("locations", [])
            logger.info(
                f"  Enhanced complete — pages={meta.get('total_pages',0)}, "
                f"people={len(people)}, locations={len(locations)}"
            )
        except Exception as e:
            logger.error(f"  Enhanced processing failed: {e}\n{traceback.format_exc()}")
            return f"failed:enhanced:{e}"

    logger.info(f"  ✅ {rel} — done")
    return "completed"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch-resume the Cairo Genizah academic literature pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Print status table for all files without processing anything.",
    )
    parser.add_argument(
        "--rerun-enhanced", "-e",
        action="store_true",
        help="Force re-run the enhanced/secondary step even if _enhanced.json exists. "
             "Use this after updating the extraction prompt.",
    )
    parser.add_argument(
        "--dir", "-d",
        metavar="SUBDIR",
        default=None,
        help="Limit processing to a specific subdirectory under academic_literature "
             "(e.g. kettubah_palestine or french/article_guy).",
    )
    parser.add_argument(
        "--skip-unocred",
        action="store_true",
        help="Skip any PDF that hasn't been OCR'd yet (avoids GCP Vision API costs).",
        default=True
    )
    parser.add_argument(
        "--no-gemini",
        action="store_true",
        help="Use Ollama/LM Studio for structured processing instead of Gemini.",
        default=False,
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Print the status table then exit (alias for --dry-run).",
    )
    args = parser.parse_args()

    use_gemini     = not args.no_gemini
    dry_run        = args.dry_run or args.status_only
    rerun_enhanced = args.rerun_enhanced
    skip_unocred   = args.skip_unocred

    # ------------------------------------------------------------------
    # Discover PDFs
    # ------------------------------------------------------------------
    search_root = data_root
    if args.dir:
        search_root = data_root / args.dir
        if not search_root.exists():
            logger.error(f"Directory not found: {search_root}")
            sys.exit(1)

    pdf_files = sorted(search_root.rglob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDFs found under {search_root}")
        sys.exit(1)

    logger.info(f"Found {len(pdf_files)} PDF(s) under {search_root.relative_to(project_root)}")

    # ------------------------------------------------------------------
    # Build status for every file
    # ------------------------------------------------------------------
    statuses = [get_file_status(p) for p in pdf_files]

    # Always print the status table
    print()
    print_status_table(statuses)
    print()

    if dry_run:
        # Summary counts
        n_ocr      = sum(1 for s in statuses if s["ocr_done"])
        n_struct   = sum(1 for s in statuses if s["structured_done"])
        n_enhanced = sum(1 for s in statuses if s["enhanced_done"])
        n_meta     = sum(1 for s in statuses if s["metadata"])
        print(f"Summary: {len(statuses)} files  |  OCR {n_ocr}  |  Structured {n_struct}  |  Enhanced {n_enhanced}  |  Has metadata {n_meta}")
        return

    # ------------------------------------------------------------------
    # Process files
    # ------------------------------------------------------------------
    outcomes: dict[str, list[str]] = {"completed": [], "skipped": [], "failed": []}

    for status in statuses:
        rel = str(status["pdf_path"].relative_to(data_root))

        # Decide if this file needs any work
        needs_ocr        = not status["ocr_done"]
        needs_structured = not status["structured_done"]
        needs_enhanced   = not status["enhanced_done"] or rerun_enhanced

        if not (needs_ocr or needs_structured or needs_enhanced):
            logger.info(f"⏭  {rel} — fully complete, skipping")
            outcomes["skipped"].append(rel)
            continue

        outcome = run_file(
            status,
            use_gemini     = use_gemini,
            rerun_enhanced = rerun_enhanced,
            skip_unocred   = skip_unocred,
        )
        if outcome.startswith("failed"):
            outcomes["failed"].append(f"{rel}: {outcome}")
        elif outcome == "skipped":
            outcomes["skipped"].append(rel)
        else:
            outcomes["completed"].append(rel)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"Batch complete — {len(pdf_files)} file(s) processed")
    print(f"  ✅ Completed : {len(outcomes['completed'])}")
    print(f"  ⏭  Skipped   : {len(outcomes['skipped'])}")
    print(f"  ❌ Failed    : {len(outcomes['failed'])}")
    if outcomes["failed"]:
        print("\nFailed files:")
        for f in outcomes["failed"]:
            print(f"  • {f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
