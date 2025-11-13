#!/usr/bin/env python3
"""
Add full OCR text back to structured JSON files.

This script takes structured JSON files (produced by StructuredJSONLLM) and adds
the full OCR text from the original OCR results back to each page.

This is useful when processing large documents where including the full text in
the LLM prompt causes token limit issues. Instead, we can add it back after processing.

Usage:
    python add_full_text_to_structured.py <structured_dir> <ocr_results_path>
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_ocr_results(ocr_results_path: str) -> Dict[str, Any]:
    """
    Load OCR results from JSON file.
    
    Args:
        ocr_results_path: Path to OCR results JSON file
        
    Returns:
        Dictionary containing OCR results indexed by page number
    """
    with open(ocr_results_path, 'r', encoding='utf-8') as f:
        ocr_data = json.load(f)
    
    # Create a dictionary mapping page numbers to OCR results
    ocr_by_page = {}
    for page in ocr_data.get('pages', []):
        page_num = page.get('page_number')
        if page_num is not None:
            ocr_result = page.get('ocr_result', {})
            full_text = ocr_result.get('full_text', '')
            ocr_by_page[page_num] = full_text
    
    logger.info(f"Loaded OCR results for {len(ocr_by_page)} pages")
    return ocr_by_page


def load_structured_files(structured_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all structured JSON files from the directory.
    
    Args:
        structured_dir: Directory containing structured JSON files
        
    Returns:
        List of structured data dictionaries with their file paths
    """
    structured_files = []
    
    for json_file in sorted(structured_dir.glob("*_structured.json")):
        with open(json_file, 'r', encoding='utf-8') as f:
            structured_data = json.load(f)
            structured_files.append({
                'path': json_file,
                'data': structured_data
            })
    
    logger.info(f"Loaded {len(structured_files)} structured files")
    return structured_files


def add_full_text_to_structured(structured_dir: str, ocr_results_path: str) -> Dict[str, Any]:
    """
    Add full OCR text to structured JSON files.
    
    Args:
        structured_dir: Directory containing structured JSON files
        ocr_results_path: Path to OCR results JSON file
        
    Returns:
        Dictionary with processing statistics
    """
    logger.info(f"Processing structured files in {structured_dir}")
    logger.info(f"Using OCR results from {ocr_results_path}")
    
    # Load OCR results
    ocr_by_page = load_ocr_results(ocr_results_path)
    
    # Load structured files
    structured_dir_path = Path(structured_dir)
    structured_files = load_structured_files(structured_dir_path)
    
    results = {
        'total_files': len(structured_files),
        'files_updated': 0,
        'files_not_found': 0,
        'pages_processed': 0
    }
    
    # Process each structured file
    for file_info in structured_files:
        structured_data = file_info['data']
        file_path = file_info['path']
        
        # Get page number from metadata
        metadata = structured_data.get('metadata', {})
        page_number = metadata.get('page_number')
        
        if page_number is None:
            logger.warning(f"No page number in metadata for {file_path.name}")
            results['files_not_found'] += 1
            continue
        
        # Get OCR text for this page
        ocr_text = ocr_by_page.get(page_number)
        
        if ocr_text is None:
            logger.warning(f"No OCR text found for page {page_number}")
            results['files_not_found'] += 1
            continue
        
        # Add full_main_text from OCR to the structured data
        structured_data['full_main_text'] = ocr_text
        logger.debug(f"Added full_main_text to page {page_number}")
        
        # Save the updated structured data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)
        
        results['files_updated'] += 1
        results['pages_processed'] += 1
    
    logger.info(f"Processing complete:")
    logger.info(f"  - Files updated: {results['files_updated']}")
    logger.info(f"  - Files not found: {results['files_not_found']}")
    
    return results


def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Add full OCR text to structured JSON files"
    )
    parser.add_argument(
        "structured_dir",
        help="Directory containing structured JSON files"
    )
    parser.add_argument(
        "ocr_results_path",
        help="Path to OCR results JSON file"
    )
    
    args = parser.parse_args()
    
    try:
        results = add_full_text_to_structured(
            structured_dir=args.structured_dir,
            ocr_results_path=args.ocr_results_path
        )
        
        print(f"\nResults:")
        print(f"  Total files: {results['total_files']}")
        print(f"  Files updated: {results['files_updated']}")
        print(f"  Files not found: {results['files_not_found']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())


