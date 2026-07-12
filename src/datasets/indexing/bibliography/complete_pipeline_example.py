#!/usr/bin/env python3
"""
Complete end-to-end pipeline for a single academic PDF — current format.

Stages
------
1.   OCR / text extraction  (BookOCRService — GCP Vision for scans,
                             PyMuPDF text-only for born-digital PDFs)
2.   Pass 1 — structured page JSONs        (StructuredJSONLLM)
2.5  Re-attach full OCR text to structured pages (idempotent)
3.   Pass 2 — per-page entity tagging      (EntityTagger)
4.   Pass 3 — book-level coreference resolution (CoreferenceResolver)
5.   Pass 4 — relation extraction          (RelationExtractor)

After the pipeline completes, import into Neo4j with::

    python src/datasets/indexing/neo4j/academic_kg_import.py

which by default imports only the Pass 4 ``book_relations.json`` files
(legacy *_enhanced.json sources require ``--include-legacy``).

The legacy SecondaryLLMProcessor flow lives on in
``secondary_processing_example.py`` for historical reference only.
"""
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import dotenv

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
data_root = project_root / "src" / "datasets" / "raw_data" / "cairo_genizah" / "academic_literature"
sys.path.append(str(project_root))

dotenv.load_dotenv(project_root / ".env")

from src.models.ocr.book_ocr_service import BookOCRService
from src.models.llm.academic.structured_json_llm import StructuredJSONLLM
from src.models.llm.academic.add_full_text_to_structured import add_full_text_to_structured
from src.models.llm.academic.llm_client import LLMClient
from src.models.llm.academic.entity_tagger import EntityTagger, _safe_tag
from src.models.llm.academic.coreference_resolver import CoreferenceResolver
from src.models.llm.academic.relationship_extractor import RelationExtractor

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pass 1 vision model. The 8b VL model proved unreliable (garbled
# extracted_page_number, sparse extraction) — default to the 35B MoE.
DEFAULT_LMS_MODEL = "qwen3.6-35b-a3b"

lms_model_names_to_extensions = {
    "qwen3_vl_8b":    "qwen3-vl:8b",          # legacy — unreliable, kept for reference
    "qwen_model":     "qwen/qwen3-vl-8b",     # legacy — unreliable, kept for reference
    "qwen3_6_35b":    "qwen3.6-35b-a3b",
}
DEFAULT_MODEL_EXTENSION = "qwen3_6_35b"


def load_book_metadata(book_file: Optional[str]) -> Optional[dict]:
    """Load ground-truth book metadata from a JSON file.

    :param book_file: Path under ``academic_literature`` to the metadata
        JSON (e.g. ``"arrant_posegay/posegayandarrant_metadata.json"``).
    :returns: Metadata dict, or None when missing.
    """
    if not book_file:
        return None
    metadata_file = data_root / book_file
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    logger.warning(f"Metadata file not found: {metadata_file}")
    return None


def _find_structured_files(structured_dir: Path, full_output_path: Path) -> List[Path]:
    """Locate Pass 1 structured page files, tolerating model subdirectories.

    :param structured_dir: Expected structured directory.
    :param full_output_path: Book output root, searched as a fallback.
    :returns: List of structured page JSON paths (possibly empty).
    """
    if structured_dir.exists():
        files = list(structured_dir.rglob("*_structured.json"))
        if files:
            return files
    if full_output_path.exists():
        return list(full_output_path.rglob("*_structured.json"))
    return []


def run_complete_pipeline(
    example_file: str,
    book_file: Optional[str] = None,
    born_digital: bool = False,
    use_gemini_structured: bool = False,
    model_extension: str = DEFAULT_MODEL_EXTENSION,
    lms_model: str = DEFAULT_LMS_MODEL,
    backend: str = "lm_studio",
    overwrite: bool = False,
) -> Dict:
    """Run the full current-format pipeline on one PDF, resuming completed steps.

    :param example_file: Path under ``academic_literature`` to the PDF.
    :param book_file: Path under ``academic_literature`` to the metadata JSON.
    :param born_digital: Extract embedded text via PyMuPDF instead of GCP OCR.
    :param use_gemini_structured: Use Gemini for Pass 1 instead of LM Studio.
    :param model_extension: Suffix for the structured output directory; also
        selects the Pass 1 LM Studio model via ``lms_model_names_to_extensions``.
    :param lms_model: LM Studio model for Passes 2–4 (and Pass 1 when the
        extension has no mapping).
    :param backend: ``"lm_studio"`` or ``"gemini"`` for Passes 2 and 4.
    :param overwrite: Re-run Passes 2–4 even when outputs exist.
    :returns: Summary dict of per-stage results.
    """
    print(f"Processing file: {example_file}")
    example_file_path = Path(example_file)
    pdf_name = example_file_path.stem
    full_output_path = data_root / example_file_path.parent / pdf_name
    full_pdf_path = data_root / example_file_path

    ocr_json_path  = full_output_path / f"{pdf_name}_ocr_results.json"
    images_dir     = full_output_path / f"{pdf_name}_images"
    structured_dir = full_output_path / f"{pdf_name}_structured_{model_extension}"
    summary: Dict = {"book_dir": str(full_output_path)}

    # ------------------------------------------------------------------
    # Step 1: OCR / text extraction
    # ------------------------------------------------------------------
    if ocr_json_path.exists() and images_dir.exists():
        print("✅ Step 1: OCR already completed — skipping")
        ocr_result = {
            'ocr_results_path': str(ocr_json_path),
            'images_dir': str(images_dir),
        }
    else:
        ocr_service = BookOCRService()
        if born_digital:
            print("Step 1: Extracting embedded text and saving images (no external OCR)...")
            ocr_result = ocr_service.process_pdf_text_only_to_file(
                pdf_path=str(full_pdf_path),
                output_path=str(full_output_path),
                start_page=0,
                end_page=None,
                save_images=True,
            )
        else:
            print("Step 1: Running OCR processing...")
            ocr_result = ocr_service.process_pdf_to_file(
                pdf_path=str(full_pdf_path),
                start_page=0,
                output_path=str(full_output_path),
            )
        print(f"OCR result: {ocr_result}")
    summary["ocr"] = ocr_result

    # ------------------------------------------------------------------
    # Step 2: Pass 1 — structured page JSONs
    # ------------------------------------------------------------------
    existing_structured = _find_structured_files(structured_dir, full_output_path)
    if existing_structured:
        structured_dir = existing_structured[0].parent
        print(f"✅ Step 2: structured processing already completed "
              f"({len(existing_structured)} files in {structured_dir})")
    else:
        print("Step 2: Running Pass 1 structured LLM processing...")
        book_metadata = load_book_metadata(book_file)
        if book_metadata:
            print(f"✅ Loaded book metadata: {book_metadata.get('title', 'Unknown title')}")
        if use_gemini_structured:
            structured_llm = StructuredJSONLLM(book_metadata=book_metadata, use_gemini=True)
        else:
            model_name = lms_model_names_to_extensions.get(model_extension, lms_model)
            structured_llm = StructuredJSONLLM(
                model_name=model_name,
                book_metadata=book_metadata,
                use_gemini=False,
            )
        structured_result = structured_llm.process_from_ocr_service(ocr_result)
        print(f"Structured processing completed: {structured_result}")
        if structured_result.get('structured_dir'):
            structured_dir = Path(structured_result['structured_dir'])

    if not structured_dir.exists():
        print(f"ERROR: Structured directory doesn't exist: {structured_dir}")
        return summary
    summary["structured_dir"] = str(structured_dir)

    # ------------------------------------------------------------------
    # Step 2.5: Re-attach full OCR text (idempotent)
    # ------------------------------------------------------------------
    print("Step 2.5: Adding full OCR text to structured files...")
    add_results = add_full_text_to_structured(
        structured_dir=str(structured_dir),
        ocr_results_path=ocr_result['ocr_results_path'],
    )
    print(f"Added full text to {add_results['files_updated']} files")

    # ------------------------------------------------------------------
    # Steps 3–5: Passes 2–4 (entity tagging → coref → relations)
    # ------------------------------------------------------------------
    run_tag = _safe_tag(lms_model) if backend == "lm_studio" else ""
    client = LLMClient(backend=backend, lms_model=lms_model)

    print(f"Step 3: Pass 2 entity tagging (run_tag={run_tag or 'default'})...")
    tagger = EntityTagger(client, run_tag=run_tag)
    summary["entities"] = tagger.tag_book(full_output_path, overwrite=overwrite)
    print(f"Entity tagging: {summary['entities']}")

    print("Step 4: Pass 3 coreference resolution...")
    resolver = CoreferenceResolver(lms_model=lms_model, run_tag=run_tag)
    summary["resolved"] = resolver.resolve_book(full_output_path, overwrite=overwrite)
    print(f"Coreference resolution written: {summary['resolved']}")

    print("Step 5: Pass 4 relation extraction...")
    extractor = RelationExtractor(client, run_tag=run_tag)
    summary["relations"] = extractor.extract_book(full_output_path, overwrite=overwrite)
    print(f"Relations extracted: {summary['relations']}")

    print(f"Complete pipeline finished for {example_file}")
    print("Import into Neo4j with: python src/datasets/indexing/neo4j/academic_kg_import.py")
    print("-" * 80)
    return summary


def check_pipeline_status(example_file: str, lms_model: str = DEFAULT_LMS_MODEL) -> None:
    """Report which pipeline stages have completed for a PDF.

    :param example_file: Path under ``academic_literature`` to the PDF.
    :param lms_model: LM Studio model whose run-tagged outputs to check.
    """
    example_file_path = Path(example_file)
    pdf_name = example_file_path.stem
    full_output_path = data_root / example_file_path.parent / pdf_name
    run_tag = _safe_tag(lms_model)

    ocr_json_path = full_output_path / f"{pdf_name}_ocr_results.json"
    structured = list(full_output_path.rglob("*_structured.json")) if full_output_path.exists() else []
    entities_dir = full_output_path / f"entities_{run_tag}"
    entity_files = list(entities_dir.glob("page_*_entities.json")) if entities_dir.exists() else []
    resolved_path = full_output_path / f"book_entities_resolved_{run_tag}.json"
    relations_path = (data_root / "relations_v2" / pdf_name / run_tag / "book_relations.json")

    print(f"Status for {example_file}  (run_tag={run_tag})")
    print(f"  OCR:        {'✅' if ocr_json_path.exists() else '❌'}  {ocr_json_path}")
    print(f"  Pass 1:     {'✅' if structured else '❌'}  {len(structured)} structured page files")
    print(f"  Pass 2:     {'✅' if entity_files else '❌'}  {len(entity_files)} entity files in {entities_dir.name}")
    print(f"  Pass 3:     {'✅' if resolved_path.exists() else '❌'}  {resolved_path.name}")
    print(f"  Pass 4:     {'✅' if relations_path.exists() else '❌'}  {relations_path}")
    print("-" * 80)


def run_all_pdf_in_dir(directory_to_process: str, book_metadata_file: str, **kwargs) -> None:
    """Run the complete pipeline on every PDF in a directory.

    :param directory_to_process: Directory path under ``academic_literature``.
    :param book_metadata_file: Shared metadata JSON path under ``academic_literature``.
    :param kwargs: Forwarded to :func:`run_complete_pipeline`.
    """
    for pdf in sorted((data_root / directory_to_process).glob("*.pdf")):
        rel = pdf.relative_to(data_root)
        run_complete_pipeline(example_file=str(rel), book_file=book_metadata_file, **kwargs)


if __name__ == "__main__":
    # End-to-end test of the new-format pipeline on a born-digital PDF
    # (embedded text — no external OCR cost).
    run_complete_pipeline(
        example_file="berlin/Schirmann-DiepoetischenGenizafragmente-1932.pdf",
        book_file="berlin/schirmann_metadata.json",
        born_digital=True,
    )
