#!/usr/bin/env python3
"""
Book OCR Service for Bibliography Processing

This module provides a service to load scanned PDFs and extract text using
Google Cloud Vision OCR API. The service processes PDFs page by page and
stores the raw OCR text for further processing.

Usage:
    from book_ocr_service import BookOCRService
    
    ocr_service = BookOCRService()
    text_results = ocr_service.process_pdf("path/to/document.pdf")
"""

import io
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import fitz  # PyMuPDF
from PIL import Image
import base64
import dotenv

from google.cloud import vision
from google.cloud.vision_v1 import types
from google.api_core import exceptions as gcp_exceptions

# Load environment variables from project root
# Get the path to the project root (5 levels up from this file)
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
dotenv.load_dotenv(project_root / ".env")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


class BookOCRService:
    """
    Service for extracting text from scanned PDF documents using Google Cloud Vision OCR.
    
    This class handles:
    - Loading PDF documents
    - Converting PDF pages to images
    - Calling Google Cloud Vision API for OCR
    - Storing raw OCR results
    """
    
    def __init__(self, 
                 raw_data_dir: Optional[str] = None,
                 credentials_path: Optional[str] = None):
        """
        Initialize the BookOCRService.
        
        Args:
            raw_data_dir: Directory where raw PDF files are stored (optional, defaults to biblio/raw_data)
            credentials_path: Path to Google Cloud credentials JSON file (optional, uses GOOGLE_APPLICATION_CREDENTIALS env var)
        """
        # Set up raw data directory - use portable path
        if raw_data_dir is None:
            # Get the directory of this file and create raw_data subdirectory
            current_file_dir = Path(__file__).parent
            self.raw_data_dir = current_file_dir / "raw_data"
        else:
            self.raw_data_dir = Path(raw_data_dir)
        
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Google Cloud Vision client
        try:
            # Load credentials from environment or provided path
            if credentials_path:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
                logger.info(f"Using credentials from: {credentials_path}")
            elif os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
                logger.info(f"Using credentials from environment: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
            else:
                logger.warning("No Google Cloud credentials found. Make sure to set GOOGLE_APPLICATION_CREDENTIALS environment variable")
            
            # Create client with explicit project configuration (following working Colab pattern)
            client_options = {}
            if os.getenv('GOOGLE_CLOUD_PROJECT'):
                client_options['quota_project_id'] = os.getenv('GOOGLE_CLOUD_PROJECT')
                logger.info(f"Using quota project: {os.getenv('GOOGLE_CLOUD_PROJECT')}")
            
            self.vision_client = vision.ImageAnnotatorClient(client_options=client_options)
            logger.info("Google Cloud Vision client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud Vision client: {e}")
            raise
    
    def load_pdf(self, pdf_path: str) -> fitz.Document:
        """
        Load a PDF document using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            PyMuPDF Document object
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If PDF cannot be loaded
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            doc = fitz.open(str(pdf_path))
            logger.info(f"Successfully loaded PDF: {pdf_path} ({len(doc)} pages)")
            return doc
        except Exception as e:
            logger.error(f"Failed to load PDF {pdf_path}: {e}")
            raise
    
    def pdf_page_to_image(self, page: fitz.Page, dpi: int = 300) -> bytes:
        """
        Convert a PDF page to image bytes.
        
        Args:
            page: PyMuPDF Page object
            dpi: Resolution for the image (default: 300)
            
        Returns:
            Image bytes in PNG format
        """
        try:
            # Create transformation matrix for desired DPI
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            
            return img_data
        except Exception as e:
            logger.error(f"Failed to convert PDF page to image: {e}")
            raise
    
    def extract_text_from_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Extract text from image using Google Cloud Vision OCR.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Dictionary containing OCR results with text and confidence scores
        """
        try:
            # Create Vision API image object
            image = types.Image(content=image_bytes)
            
            # Configure OCR request with language hints for Hebrew/Arabic documents.
            image_context = types.ImageContext(language_hints=['he', 'ar', 'en'])
            
            # Perform OCR using document_text_detection for better results
            response = self.vision_client.document_text_detection(
                image=image,
                image_context=image_context
            )
            
            if response.error.message:
                raise Exception(f"Vision API error: {response.error.message}")
            
            # Extract text and annotations
            full_text = response.full_text_annotation.text if response.full_text_annotation else ""
            
            # Extract individual text blocks with confidence scores
            text_blocks = []
            if response.text_annotations:
                for annotation in response.text_annotations[1:]:  # Skip first (full text)
                    text_blocks.append({
                        'text': annotation.description,
                        'confidence': getattr(annotation, 'confidence', 0.0),
                        'bounding_box': self._extract_bounding_box(annotation.bounding_poly)
                    })
            
            result = {
                'full_text': full_text,
                'text_blocks': text_blocks,
                'page_text': full_text,
                'confidence_score': getattr(response.full_text_annotation, 'confidence', 0.0) if response.full_text_annotation else 0.0,
                'language_hints': self._extract_language_hints(response.full_text_annotation) if response.full_text_annotation else []
            }
            
            logger.info(f"OCR completed. Extracted {len(full_text)} characters")
            return result
            
        except gcp_exceptions.GoogleAPIError as e:
            logger.error(f"Google Cloud Vision API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to extract text from image: {e}")
            raise
    
    def _extract_bounding_box(self, bounding_poly) -> Dict[str, int]:
        """Extract bounding box coordinates from Vision API response."""
        if not bounding_poly or not bounding_poly.vertices:
            return {'x': 0, 'y': 0, 'width': 0, 'height': 0}
        
        vertices = bounding_poly.vertices
        x_coords = [v.x for v in vertices]
        y_coords = [v.y for v in vertices]
        
        return {
            'x': min(x_coords),
            'y': min(y_coords),
            'width': max(x_coords) - min(x_coords),
            'height': max(y_coords) - min(y_coords)
        }
    
    def _extract_language_hints(self, full_text_annotation) -> List[str]:
        """Extract detected language hints from Vision API response."""
        if not full_text_annotation or not hasattr(full_text_annotation, 'pages'):
            return []
        
        languages = []
        for page in full_text_annotation.pages:
            if hasattr(page, 'property') and hasattr(page.property, 'detected_languages'):
                for lang in page.property.detected_languages:
                    languages.append(lang.language_code)
        
        return list(set(languages))  # Remove duplicates
    
    def process_pdf(self, pdf_path: str, start_page: int = 0, end_page: Optional[int] = None, 
                   save_images: bool = True, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a PDF document page by page using OCR.
        
        Args:
            pdf_path: Path to the PDF file
            start_page: Starting page number (0-indexed)
            end_page: Ending page number (0-indexed, None for all pages)
            save_images: Whether to save page images for later LLM processing
            output_dir: Directory to save images (optional, defaults to self.raw_data_dir)
            
        Returns:
            Dictionary containing OCR results for all processed pages
        """
        pdf_path = Path(pdf_path)
        
        try:
            # Load PDF
            doc = self.load_pdf(str(pdf_path))
            
            # Determine page range
            total_pages = len(doc)
            if end_page is None:
                end_page = total_pages - 1
            else:
                end_page = min(end_page, total_pages - 1)
            
            logger.info(f"Processing PDF: {pdf_path.name} (pages {start_page}-{end_page} of {total_pages})")
            
            # Create images directory if saving images
            images_dir = None
            if save_images:
                # Use output_dir if provided, otherwise use self.raw_data_dir
                base_dir = Path(output_dir) if output_dir else self.raw_data_dir
                images_dir = base_dir / f"{pdf_path.stem}_images"
                images_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Images will be saved to: {images_dir}")
            
            # Process each page
            results = {
                'pdf_path': str(pdf_path),
                'total_pages': total_pages,
                'processed_pages': end_page - start_page + 1,
                'images_dir': str(images_dir) if images_dir else None,
                'pages': []
            }
            
            for page_num in range(start_page, end_page + 1):
                try:
                    logger.info(f"Processing page {page_num + 1}/{total_pages}")
                    
                    # Get page
                    page = doc[page_num]
                    
                    # Convert to image
                    image_bytes = self.pdf_page_to_image(page)
                    
                    # Save image if requested
                    image_path = None
                    if save_images and images_dir:
                        image_filename = f"page_{page_num + 1:03d}.png"
                        image_path = images_dir / image_filename
                        with open(image_path, 'wb') as f:
                            f.write(image_bytes)
                        logger.debug(f"Saved image: {image_path}")
                    
                    # Extract text
                    ocr_result = self.extract_text_from_image(image_bytes)
                    
                    # Add page metadata
                    page_result = {
                        'page_number': page_num + 1,  # 1-indexed for user convenience
                        'ocr_result': ocr_result,
                        'image_path': str(image_path) if image_path else None,
                        'image_size': {
                            'width': page.rect.width,
                            'height': page.rect.height
                        }
                    }
                    
                    results['pages'].append(page_result)
                    
                except Exception as e:
                    logger.error(f"Failed to process page {page_num + 1}: {e}")
                    # Add error page result
                    results['pages'].append({
                        'page_number': page_num + 1,
                        'error': str(e),
                        'ocr_result': None,
                        'image_path': None
                    })
            
            # Close document
            doc.close()
            
            logger.info(f"Completed processing PDF: {pdf_path.name}")
            if save_images:
                logger.info(f"Page images saved to: {images_dir}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            raise
    
    def process_pdf_to_file(self, pdf_path: str, output_path: Optional[str] = None, 
                           start_page: int = 0, end_page: Optional[int] = None, 
                           save_images: bool = True) -> Dict[str, str]:
        """
        Process a PDF and save OCR results to a JSON file.
        
        Args:
            pdf_path: Path to the PDF file
            output_path: Path for output JSON file (optional). If provided without .json extension, 
                        treats it as a directory and creates {pdf_stem}_ocr_results.json inside it.
            start_page: Starting page number (0-indexed)
            end_page: Ending page number (0-indexed, None for all pages)
            save_images: Whether to save page images for later LLM processing
            
        Returns:
            Dictionary containing:
            - 'ocr_results_path': Path to the OCR results JSON file
            - 'images_dir': Path to the images directory (if save_images=True)
        """
        import json
        
        pdf_path = Path(pdf_path)
        
        # Generate output path if not provided
        if output_path is None:
            output_path = self.raw_data_dir / f"{pdf_path.stem}_ocr_results.json"
        else:
            output_path = Path(output_path)
            # If output_path doesn't have .json extension, treat it as a directory and create proper filename
            if output_path.suffix != '.json':
                # Ensure the parent directory exists before creating the file path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path = output_path / f"{pdf_path.stem}_ocr_results.json"
        
        # Process PDF
        # Use the parent directory of output_path for images if output_path is provided
        output_dir = output_path.parent if output_path else None
        results = self.process_pdf(str(pdf_path), start_page, end_page, save_images, str(output_dir) if output_dir else None)
        
        # Add metadata
        results['processing_info'] = {
            'processed_at': str(Path().cwd()),
            'service_version': '1.0.0',
            'start_page': start_page,
            'end_page': end_page,
            'save_images': save_images
        }
        
        # Save results
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"OCR results saved to: {output_path}")
            if save_images and results.get('images_dir'):
                logger.info(f"Page images saved to: {results['images_dir']}")
            
            # Return both paths
            return {
                'ocr_results_path': str(output_path),
                'images_dir': results.get('images_dir')
            }
            
        except Exception as e:
            logger.error(f"Failed to save OCR results to {output_path}: {e}")
            raise

    def process_pdf_text_only(self, pdf_path: str, start_page: int = 0, end_page: Optional[int] = None,
                              save_images: bool = True, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract raw text from a (born-digital) PDF using PyMuPDF, optionally saving page images.

        This alternative flow skips external OCR and relies on embedded PDF text extraction.

        :param pdf_path: Path to the PDF file to process.
        :type pdf_path: str
        :param start_page: Starting page number (0-indexed).
        :type start_page: int
        :param end_page: Ending page number (0-indexed). If None, processes to last page.
        :type end_page: Optional[int]
        :param save_images: Whether to render and save page images for later multimodal LLM use.
        :type save_images: bool
        :param output_dir: Directory to save images; defaults to `self.raw_data_dir` if None.
        :type output_dir: Optional[str]
        :returns: Dictionary containing extraction results compatible with `StructuredJSONLLM.process_ocr_results`.
        :rtype: Dict[str, Any]

        Example::

            service = BookOCRService()
            results = service.process_pdf_text_only(
                pdf_path="/path/to/file.pdf",
                start_page=0,
                end_page=None,
                save_images=True,
                output_dir="/tmp/output"
            )
        """
        pdf_path = Path(pdf_path)

        try:
            doc = self.load_pdf(str(pdf_path))

            total_pages = len(doc)
            if end_page is None:
                end_page = total_pages - 1
            else:
                end_page = min(end_page, total_pages - 1)

            logger.info(f"Text-only processing PDF: {pdf_path.name} (pages {start_page}-{end_page} of {total_pages})")

            images_dir: Optional[Path] = None
            if save_images:
                base_dir = Path(output_dir) if output_dir else self.raw_data_dir
                images_dir = base_dir / f"{pdf_path.stem}_images"
                images_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Images will be saved to: {images_dir}")

            results: Dict[str, Any] = {
                'pdf_path': str(pdf_path),
                'total_pages': total_pages,
                'processed_pages': end_page - start_page + 1,
                'images_dir': str(images_dir) if images_dir else None,
                'pages': []
            }

            for page_num in range(start_page, end_page + 1):
                logger.info(f"Extracting text (no OCR) from page {page_num + 1}/{total_pages}")
                page = doc[page_num]

                # Extract embedded text from the PDF page
                # Using "text" preserves a readable flow in most cases; alternatives include "blocks" or "page"
                page_text: str = page.get_text("text") or ""

                # Optionally render page image for consistent downstream multimodal processing
                image_path: Optional[Path] = None
                if save_images and images_dir is not None:
                    img_bytes = self.pdf_page_to_image(page)
                    image_filename = f"page_{page_num + 1:03d}.png"
                    image_path = images_dir / image_filename
                    with open(image_path, 'wb') as f:
                        f.write(img_bytes)

                # Build page entry compatible with OCR schema expected by StructuredJSONLLM
                page_result: Dict[str, Any] = {
                    'page_number': page_num + 1,
                    'ocr_result': {
                        'full_text': page_text,
                        'text_blocks': [],
                        'page_text': page_text,
                        'confidence_score': 1.0,
                        'language_hints': []
                    },
                    'image_path': str(image_path) if image_path else None,
                    'image_size': {
                        'width': page.rect.width,
                        'height': page.rect.height
                    }
                }

                results['pages'].append(page_result)

            doc.close()
            logger.info(f"Completed text-only processing: {pdf_path.name}")
            return results

        except Exception as e:
            logger.error(f"Failed text-only processing for PDF {pdf_path}: {e}")
            raise

    def process_pdf_text_only_to_file(self, pdf_path: str, output_path: Optional[str] = None,
                                      start_page: int = 0, end_page: Optional[int] = None,
                                      save_images: bool = True) -> Dict[str, str]:
        """
        Run `process_pdf_text_only` and persist results to a JSON file, mirroring the OCR output shape.

        :param pdf_path: Path to the PDF file to process.
        :type pdf_path: str
        :param output_path: Output JSON file path or directory. If directory or without *.json*, creates
                           `{pdf_stem}_ocr_results.json` inside it to match downstream expectations.
        :type output_path: Optional[str]
        :param start_page: Starting page number (0-indexed).
        :type start_page: int
        :param end_page: Ending page number (0-indexed). If None, processes to last page.
        :type end_page: Optional[int]
        :param save_images: Whether to render and save page images.
        :type save_images: bool
        :returns: Dict with keys `ocr_results_path` and `images_dir`.
        :rtype: Dict[str, str]

        Example::

            service = BookOCRService()
            out = service.process_pdf_text_only_to_file(
                pdf_path="/path/file.pdf",
                output_path="/tmp/out_dir",
                start_page=0,
                end_page=None,
                save_images=True
            )
        """
        import json

        pdf_path_obj = Path(pdf_path)

        if output_path is None:
            output_path_obj = self.raw_data_dir / f"{pdf_path_obj.stem}_ocr_results.json"
        else:
            output_path_obj = Path(output_path)
            if output_path_obj.suffix != '.json':
                output_path_obj.parent.mkdir(parents=True, exist_ok=True)
                output_path_obj = output_path_obj / f"{pdf_path_obj.stem}_ocr_results.json"

        output_dir = output_path_obj.parent if output_path_obj else None
        results = self.process_pdf_text_only(
            pdf_path=str(pdf_path_obj),
            start_page=start_page,
            end_page=end_page,
            save_images=save_images,
            output_dir=str(output_dir) if output_dir else None
        )

        # Tag processing mode for clarity
        results['processing_info'] = {
            'processed_at': str(Path().cwd()),
            'service_version': '1.0.0',
            'start_page': start_page,
            'end_page': end_page,
            'save_images': save_images,
            'mode': 'text_only'
        }

        with open(output_path_obj, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Text-only results saved to: {output_path_obj}")
        if save_images and results.get('images_dir'):
            logger.info(f"Page images saved to: {results['images_dir']}")

        return {
            'ocr_results_path': str(output_path_obj),
            'images_dir': results.get('images_dir')
        }
    
    def list_pdf_files(self) -> List[Path]:
        """
        List all PDF files in the raw data directory.
        
        Returns:
            List of Path objects for PDF files
        """
        pdf_files = list(self.raw_data_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {self.raw_data_dir}")
        return pdf_files


def main():
    """Example usage of the BookOCRService."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process PDF documents with OCR")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--start-page", type=int, default=0, help="Starting page (0-indexed)")
    parser.add_argument("--end-page", type=int, help="Ending page (0-indexed)")
    parser.add_argument("--credentials", help="Path to Google Cloud credentials JSON")
    
    args = parser.parse_args()

    ocr_service = BookOCRService(credentials_path=args.credentials)

    # Process PDF
    output_path = ocr_service.process_pdf_to_file(
        pdf_path=args.pdf_path,
        output_path=args.output,
        start_page=args.start_page,
        end_page=args.end_page
    )

    print(f"OCR processing completed. Results saved to: {output_path}")





if __name__ == "__main__":
    main()
