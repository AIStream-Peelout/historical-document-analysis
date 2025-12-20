#!/usr/bin/env python3
"""
Structured JSON LLM Service for Bibliography Processing (Pydantic AI Version)

This module provides a service to process OCR text and images through an LLM
(Ollama or Gemini) to extract structured JSON data from Cairo Genizah bibliography documents.

Usage:
    from structured_json_llm import StructuredJSONLLM

    llm_service = StructuredJSONLLM()
    structured_data = llm_service.process_page(ocr_text, image_path)
"""

import json
import logging
import os
import base64
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from PIL import Image

import dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.ollama import OllamaProvider

# Load environment variables from project root
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
dotenv.load_dotenv(project_root / ".env")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


class StructuredPageData(BaseModel):
    """Schema for structured page data from Cairo Genizah bibliography"""
    shelf_marks_mentioned: Dict[str, str] = Field(
        default_factory=dict,
        description="Shelf marks found with brief summary of their mention"
    )
    footnotes: Dict[str, str] = Field(
        default_factory=dict,
        description="Footnotes with their numbers as keys"
    )
    transcriptions: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="Transcriptions organized by shelf mark, then line number"
    )
    extracted_page_number: Optional[List[int]] = Field(
        None,
        description="Page number extracted from the document"
    )
    summary: str = Field(
        ...,
        description="Summary of the page content"
    )
    classification: str = Field(
        ...,
        description="Document classification: general_academic, transcription, catalog, or diagram/map"
    )


class StructuredJSONLLM:
    """
    Service for processing OCR text and images through LLM to extract structured JSON data.

    This class handles:
    - Loading OCR results and images
    - Calling Ollama or Gemini LLM with structured prompts
    - Automatic validation and retry via Pydantic AI
    - Saving structured results
    """

    def __init__(self,
                 raw_data_dir: Optional[str] = None, #str = "http://localhost:1234/v1",
                 ollama_url: str = "http://localhost:1234/v1",
                 model_name: str = "qwen3-vl:8b",
                 use_gemini: bool = False,
                 gemini_api_key: Optional[str] = None,
                 gemini_model: str = "gemini-2.0-flash-exp",
                 gemini_safety_settings: Optional[List[Dict[str, str]]] = None,
                 book_metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the StructuredJSONLLM service.

        :param raw_data_dir: Directory where raw data files are stored (optional, defaults to biblio/raw_data)
        :param ollama_url: URL for Ollama API (default: http://localhost:11434/v1). Should follow OpenAI style.
        :param model_name: Name of the Ollama model to use (default: qwen3-vl:8b)
        :param use_gemini: Whether to use Gemini instead of Ollama (default: False)
        :param gemini_api_key: Gemini API key (if None, will try to get from environment)
        :param gemini_model: Gemini model name (default: gemini-2.0-flash-exp)
        :param gemini_safety_settings: Custom safety settings for Gemini
        :param book_metadata: Optional dictionary containing book metadata
        """
        # Set up raw data directory
        if raw_data_dir is None:
            current_file_dir = Path(__file__).parent
            self.raw_data_dir = current_file_dir / "raw_data"
        else:
            self.raw_data_dir = Path(raw_data_dir)

        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.use_gemini = use_gemini
        self.book_metadata = book_metadata
        self.model_name = model_name

        # Set up model and agent
        if self.use_gemini:
            self._setup_gemini(gemini_model, gemini_api_key, gemini_safety_settings)
        else:
            self._setup_ollama(model_name, ollama_url)

        # Create agent with structured output
        self.agent = Agent(
            self.model,
            output_type=StructuredPageData,
            system_prompt=self._get_system_prompt(),
            retries=2  # Pydantic AI will automatically retry on validation failures
        )

        logger.info(
            f"StructuredJSONLLM initialized with {'Gemini' if use_gemini else 'Ollama'} model: {self.model_name}")

    def _setup_ollama(self, model_name: str, ollama_url: str):
        """Setup Ollama model"""
        self.model = OpenAIChatModel(
            model_name,
            provider=OllamaProvider(base_url=ollama_url)
        )
        self.model_name = model_name

    def _setup_gemini(self, gemini_model: str, gemini_api_key: Optional[str],
                      gemini_safety_settings: Optional[List[Dict[str, str]]]):
        """Setup Gemini model"""
        self.gemini_model = gemini_model
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')

        if not self.gemini_api_key:
            raise ValueError(
                "Gemini API key is required when use_gemini=True. "
                "Set GEMINI_API_KEY environment variable or pass gemini_api_key parameter."
            )

        # Set up safety settings for Hebrew text processing
        if gemini_safety_settings is None:
            self.gemini_safety_settings = {
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
            }
        else:
            # Convert list format to dict format if needed
            self.gemini_safety_settings = {}
            for setting in gemini_safety_settings:
                self.gemini_safety_settings[setting['category']] = setting['threshold']

        self.model = GoogleModel(
            gemini_model,
            api_key=self.gemini_api_key,
            safety_settings=self.gemini_safety_settings
        )
        self.model_name = gemini_model

    def _get_system_prompt(self, book_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create the system prompt for the LLM.

        :param book_metadata: Optional dictionary containing book metadata
        :return: System prompt string
        """
        # Build metadata context if available
        metadata_context = ""
        if book_metadata:
            metadata_context = "\n\n## Book Context\n"
            metadata_context += "You are processing pages from a specific book. Here is the book's metadata:\n"
            for key, value in book_metadata.items():
                if value:
                    value_str = str(value)
                    value_str = value_str.replace("{", "(").replace("}", ")")
                    metadata_context += f"- {key.replace('_', ' ').title()}: {value_str}\n"
            metadata_context += "\nUse this context to better understand the pages you are processing.\n"

        prompt = """You are an expert Judaic studies professor PhD and data annotator. You will be given both text and images of books on the Cairo Genizah. Your job is to parse the text into a consistent JSON structured format based on both the raw text extracted by OCR and your understanding from the visual of the document.""" + metadata_context + """

Use the image ONLY to determine the structure of the document. Do not EVER attempt to transcribe from the image of the document. Your job is only to arrange the text given based on the image.

Classification: Based on the text and image of the document. Please classify the document as one of the following:
    - general_academic: General Academic Analysis of Cairo Genizah documents. 
    - transcription: A transcription of a specific Cairo Genizah shelf-mark. Usually these pages will have a lot of lines
    and old Hebrew/Arabic text. They will usually be laid out in very specific style with each line. 
    - catalog: These pages will generally be very structured. Likely listing many shelf-marks and associations.
    - diagram/map: These pages are generally image based.

**IMPORTANT: Extract page numbers**
Look at the actual printed/visible page numbers in the document image (usually at top or bottom of page).
- Single page scan: return one number, e.g., [119]
- Two-page spread scan: return both numbers in order, e.g., [118, 119]
- No visible page numbers: return empty list []
These are NOT the PDF page numbers - extract the actual numbers printed on the document.


Guidelines:
1. Extract all shelf marks mentioned in the text. Common patterns include:
   - Cambridge: T-S followed by classification (T-S A43.1, T-S NS 322.64, T-S Ar.31.30)
   - JTS: ENA series (ENA 2689.11, ENA NS 12.5)
   - Oxford: MS Heb. series (MS Heb. d. 32)
   - Manchester Rylands: Multiple formats including:
     * Letter + space + number (A 589, B 3239, not A589 or B3239)
     * Rylands Genizah + number (Rylands Genizah 2)
     * Gaster series (Gaster Hebrew, Gaster Arabic, Gaster Printed)
     * Language series (Ar. 400)
   - UPenn: Halper numbers (Halper 331)
   - Paris AIU: Roman numerals (AIU VI C 6)
   Note: Pay attention to spacing - Rylands uses "A 589" while others might use "ENA2689"
   Note: Shelf marks may use various separators (periods, spaces, hyphens)
2. Identify footnotes and extract their content with proper numbering
3. Extract Hebrew/Arabic transcriptions and associate them with shelf marks
4. If no footnotes exist, use empty object
5. If no transcriptions exist, use empty object. DO NOT EVER TRY TO TRANSCRIBE IMAGES OF THE DOCUMENT. YOU ARE TO USE ONLY THE PROVIDED OCR TEXT. 
6. If no shelf marks are mentioned, use empty object
7. Extract the page number or page numbers from the bottom as a list. 
"""

        return prompt

    def _load_book_metadata(self, pdf_name: str) -> Optional[Dict[str, Any]]:
        """
        Load book metadata from a JSON file named `{pdf_name}_metadata.json`.

        :param pdf_name: Name of the PDF file (without extension)
        :return: Dictionary containing book metadata, or None if file not found
        """
        # Try to find the metadata file in the same directory as this file
        metadata_file = Path(__file__).parent / f"{pdf_name}_metadata.json"

        # If not found, try in the raw_data directory
        if not metadata_file.exists():
            metadata_file = self.raw_data_dir / f"{pdf_name}_metadata.json"

        # Try relative to current working directory
        if not metadata_file.exists():
            metadata_file = Path.cwd() / f"{pdf_name}_metadata.json"

        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    logger.info(f"Loaded book metadata from {metadata_file}")
                    return metadata
            except Exception as e:
                logger.warning(f"Failed to load book metadata from {metadata_file}: {e}")
                return None
        else:
            logger.debug(f"Book metadata file not found: {metadata_file}")
            return None

    def _get_output_directory_name(self, pdf_name: str) -> str:
        """
        Generate output directory name with provider and model information.

        :param pdf_name: Name of the source PDF
        :return: Directory name string with provider and model info
        """
        if self.use_gemini:
            clean_model = self.gemini_model.replace(":", "_").replace("-", "_")
            return f"{pdf_name}_structured_gemini_{clean_model}"
        else:
            clean_model = self.model_name.replace(":", "_").replace("-", "_")
            return f"{pdf_name}_structured_{clean_model}"

    def _encode_image_to_base64(self, image_path: Union[str, Path]) -> str:
        """
        Encode image to base64 for API.

        :param image_path: Path to the image file
        :return: Base64 encoded image string
        """
        try:
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                base64_string = base64.b64encode(image_data).decode('utf-8')
                return base64_string
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise

    async def process_page(self, ocr_text: str, image_path: Union[str, Path],
                           page_number: int, pdf_name: str) -> Dict[str, Any]:
        """
        Process a single page through the LLM to extract structured data.

        :param ocr_text: Raw OCR text from the page
        :param image_path: Path to the page image
        :param page_number: Page number for identification
        :param pdf_name: Name of the source PDF
        :return: Structured JSON data
        """
        try:
            logger.info(f"Processing page {page_number} from {pdf_name}")

            # Use provided book metadata or load from file
            book_metadata = self.book_metadata
            if book_metadata is None:
                book_metadata = self._load_book_metadata(pdf_name)

            # Update system prompt with metadata if needed
            if book_metadata:
                system_prompt = self._get_system_prompt(book_metadata)
                # Create a new agent with updated prompt for this page
                agent = Agent(
                    self.model,
                    output_type=StructuredPageData,
                    system_prompt=system_prompt,
                    retries=1
                )
            else:
                agent = self.agent

            # Prepare prompt with OCR text
            prompt = f"""Please process the following document page and extract structured data.

OCR Text:
{ocr_text}

Extract the structured data according to the schema."""

            # Encode image
            image_b64 = self._encode_image_to_base64(image_path)

            # Run agent with image - Pydantic AI handles validation and retries!
            # For Ollama models, we pass the image in a special way
            if not self.use_gemini:
                # Ollama approach - pass image as base64 in the prompt context
                result = await agent.run(
                    prompt,
                    message_history=[{
                        'role': 'user',
                        'content': prompt,
                        'images': [image_b64]  # Ollama format
                    }]
                )
            else:
                # Gemini approach - use PIL Image
                pil_image = Image.open(image_path)
                result = await agent.run(
                    prompt,
                    message_history=[{
                        'role': 'user',
                        'content': [prompt, pil_image]
                    }]
                )

            # result.output is already validated StructuredPageData!
            structured_data = result.output.model_dump()

            # Add metadata
            structured_data['metadata'] = {
                'page_number': page_number,
                'pdf_name': pdf_name,
                'processed_at': str(Path().cwd()),
                'model_used': self.model_name,
                'provider': 'gemini' if self.use_gemini else 'ollama'
            }

            logger.info(f"Successfully processed page {page_number}")
            return structured_data

        except Exception as e:
            logger.error(f"Failed to process page {page_number}: {e}")
            raise

    def process_page_sync(self, ocr_text: str, image_path: Union[str, Path],
                          page_number: int, pdf_name: str) -> Dict[str, Any]:
        """Synchronous wrapper for process_page"""
        import asyncio
        return asyncio.run(self.process_page(ocr_text, image_path, page_number, pdf_name))

    async def process_page_text_only(self, ocr_text: str, page_number: int,
                                     pdf_name: str) -> Dict[str, Any]:
        """
        Process a page with text-only (no image) for cases where image is not available.

        :param ocr_text: Raw OCR text from the page
        :param page_number: Page number for identification
        :param pdf_name: Name of the source PDF
        :return: Structured JSON data
        """
        try:
            logger.info(f"Processing page {page_number} from {pdf_name} (text-only)")

            # Use provided book metadata or load from file
            book_metadata = self.book_metadata
            if book_metadata is None:
                book_metadata = self._load_book_metadata(pdf_name)

            # Update system prompt with metadata if needed
            if book_metadata:
                system_prompt = self._get_system_prompt(book_metadata)
                agent = Agent(
                    self.model,
                    output_type=StructuredPageData,
                    system_prompt=system_prompt,
                    retries=2
                )
            else:
                agent = self.agent

            # Prepare prompt
            prompt = f"""Please process the following document page and extract structured data.

OCR Text:
{ocr_text}

Note: No image is available for this page, so use only the text to determine structure.

Extract the structured data according to the schema."""

            # Run agent without image
            result = await agent.run(prompt)

            # result.output is already validated StructuredPageData!
            structured_data = result.output.model_dump()

            # Add metadata
            structured_data['metadata'] = {
                'page_number': page_number,
                'pdf_name': pdf_name,
                'processed_at': str(Path().cwd()),
                'model_used': self.model_name,
                'processing_mode': 'text_only',
                'provider': 'gemini' if self.use_gemini else 'ollama'
            }

            logger.info(f"Successfully processed page {page_number} (text-only)")
            return structured_data

        except Exception as e:
            logger.error(f"Failed to process page {page_number} (text-only): {e}")
            raise

    def process_page_text_only_sync(self, ocr_text: str, page_number: int,
                                    pdf_name: str) -> Dict[str, Any]:
        """Synchronous wrapper for process_page_text_only"""
        import asyncio
        return asyncio.run(self.process_page_text_only(ocr_text, page_number, pdf_name))

    async def process_ocr_results(self, ocr_results_path: Union[str, Path],
                                  output_dir: Optional[Path] = None,
                                  starting_page_number: int = 1) -> Dict[str, Any]:
        """
        Process OCR results file and create structured JSON for each page.

        :param ocr_results_path: Path to the OCR results JSON file
        :param output_dir: Directory to save structured results (optional)
        :param starting_page_number: Page number to start from (default: 1)
        :return: Dictionary with processing results
        """
        ocr_results_path = Path(ocr_results_path)

        try:
            # Load OCR results
            with open(ocr_results_path, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)

            pdf_name = Path(ocr_data['pdf_path']).stem
            logger.info(f"Processing OCR results for PDF: {pdf_name}")

            # Create output directory
            structured_dir_name = self._get_output_directory_name(pdf_name)
            if output_dir is None:
                structured_dir = self.raw_data_dir / structured_dir_name
            else:
                structured_dir = output_dir / structured_dir_name
            structured_dir.mkdir(parents=True, exist_ok=True)

            results = {
                'pdf_name': pdf_name,
                'total_pages': len(ocr_data['pages']),
                'processed_pages': 0,
                'failed_pages': 0,
                'structured_files': [],
                'error_summary': {
                    'validation_errors': 0,
                    'api_errors': 0,
                    'other_errors': 0
                },
                'failed_pages_details': []
            }

            # Process each page
            for page_data in ocr_data['pages']:
                page_number = page_data['page_number']
                if page_number < starting_page_number:
                    continue

                try:
                    # Check if OCR result exists
                    if 'ocr_result' not in page_data or not page_data['ocr_result']:
                        logger.warning(f"No OCR result for page {page_number}, skipping")
                        results['failed_pages'] += 1
                        continue

                    ocr_result = page_data['ocr_result']
                    full_text = ocr_result.get('full_text', '')

                    if not full_text.strip():
                        logger.warning(f"No text content for page {page_number}, skipping")
                        results['failed_pages'] += 1
                        continue

                    # Check if image path is available
                    image_path = page_data.get('image_path')
                    if image_path and Path(image_path).exists():
                        # Process with image
                        structured_data = await self.process_page(
                            full_text, image_path, page_number, pdf_name
                        )
                    else:
                        # Fallback to text-only processing
                        logger.warning(f"No image available for page {page_number}, using text-only")
                        structured_data = await self.process_page_text_only(
                            full_text, page_number, pdf_name
                        )

                    # Save structured data
                    output_file = structured_dir / f"page_{page_number:03d}_structured.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(structured_data, f, indent=2, ensure_ascii=False)

                    results['structured_files'].append(str(output_file))
                    results['processed_pages'] += 1

                    logger.info(f"Processed page {page_number}/{len(ocr_data['pages'])}")

                except Exception as e:
                    logger.error(f"Failed to process page {page_number}: {e}")
                    results['failed_pages'] += 1

                    # Categorize the error
                    error_message = str(e)
                    error_type = 'other_errors'

                    if 'validation' in error_message.lower():
                        error_type = 'validation_errors'
                    elif 'API' in error_message or 'api' in error_message:
                        error_type = 'api_errors'

                    results['error_summary'][error_type] += 1
                    results['failed_pages_details'].append({
                        'page_number': page_number,
                        'error_type': error_type,
                        'error_message': error_message
                    })

            # Save summary
            summary_file = structured_dir / "processing_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Completed processing {pdf_name}: "
                f"{results['processed_pages']} pages processed, "
                f"{results['failed_pages']} failed"
            )

            # Log error summary
            if results['failed_pages'] > 0:
                logger.warning("Error Summary:")
                for error_type, count in results['error_summary'].items():
                    if count > 0:
                        logger.warning(f"  {error_type}: {count} pages")

            return results

        except Exception as e:
            logger.error(f"Failed to process OCR results: {e}")
            raise

    def process_ocr_results_sync(self, ocr_results_path: Union[str, Path],
                                 output_dir: Optional[Path] = None,
                                 starting_page_number: int = 1) -> Dict[str, Any]:
        """Synchronous wrapper for process_ocr_results"""
        import asyncio
        return asyncio.run(
            self.process_ocr_results(ocr_results_path, output_dir, starting_page_number)
        )

    def process_from_ocr_service(self, ocr_service_result: Dict[str, str],
                                 page_number: int = 1) -> Dict[str, Any]:
        """
        Process OCR results directly from BookOCRService.process_pdf_to_file() output.

        :param ocr_service_result: Dictionary from BookOCRService.process_pdf_to_file()
            containing 'ocr_results_path' and 'images_dir'
        :param page_number: Starting page number (default: 1)
        :return: Dictionary with processing results
        """
        ocr_results_path = ocr_service_result['ocr_results_path']
        images_dir = ocr_service_result.get('images_dir')

        logger.info(f"Processing OCR results from service: {ocr_results_path}")
        if images_dir:
            logger.info(f"Using images directory: {images_dir}")

        return self.process_ocr_results_sync(
            ocr_results_path,
            output_dir=Path(ocr_results_path).parent,
            starting_page_number=page_number
        )


def main():
    """Example usage of the StructuredJSONLLM service."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process OCR results with LLM to extract structured JSON"
    )
    parser.add_argument("ocr_results_path", help="Path to OCR results JSON file")
    parser.add_argument("--model", default="qwen3-vl:8b", help="Ollama model name")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama API URL")
    parser.add_argument("--use-gemini", action="store_true", help="Use Gemini instead of Ollama")
    parser.add_argument("--gemini-model", default="gemini-2.0-flash-exp", help="Gemini model name")
    parser.add_argument("--gemini-api-key", help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--starting-page", type=int, default=1, help="Starting page number")

    args = parser.parse_args()

    try:
        # Initialize service
        llm_service = StructuredJSONLLM(
            model_name=args.model,
            ollama_url=args.ollama_url,
            use_gemini=args.use_gemini,
            gemini_model=args.gemini_model,
            gemini_api_key=args.gemini_api_key
        )

        # Process OCR results
        results = llm_service.process_ocr_results_sync(
            args.ocr_results_path,
            starting_page_number=args.starting_page
        )

        print(f"Processing completed:")
        print(f"  - Processed pages: {results['processed_pages']}")
        print(f"  - Failed pages: {results['failed_pages']}")
        print(f"  - Structured files: {len(results['structured_files'])}")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    main()