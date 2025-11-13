#!/usr/bin/env python3
"""
Structured JSON LLM Service for Bibliography Processing

This module provides a service to process OCR text and images through an LLM
(Ollama) to extract structured JSON data from Cairo Genizah bibliography documents.

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
import requests
from PIL import Image
import io

import dotenv
import google.generativeai as genai

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


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator to retry function calls on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Don't retry on certain types of errors
                    if "blocked by Gemini safety filters" in str(e):
                        logger.warning("Content blocked by safety filters - not retrying")
                        raise e
                    elif "truncated due to token limit" in str(e):
                        logger.warning("Response truncated due to token limit - not retrying")
                        raise e
                    
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed")
                        raise last_exception
            
            return None
        return wrapper
    return decorator


class StructuredJSONLLM:
    """
    Service for processing OCR text and images through LLM to extract structured JSON data.
    
    This class handles:
    - Loading OCR results and images
    - Calling Ollama LLM with structured prompts
    - Parsing LLM responses into JSON format
    - Saving structured results
    """
    
    def __init__(self, 
                 raw_data_dir: Optional[str] = None,
                 ollama_url: str = "http://localhost:11434",
                 model_name: str = "llama3.1:8b",
                 use_gemini: bool = False,
                 gemini_api_key: Optional[str] = None,
                 gemini_model: str = "gemini-2.5-flash",
                 gemini_safety_settings: Optional[List[Dict[str, str]]] = None,
                 book_metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the StructuredJSONLLM service.
        
        Args:
            raw_data_dir: Directory where raw data files are stored (optional, defaults to biblio/raw_data)
            ollama_url: URL for Ollama API (default: http://localhost:11434)
            model_name: Name of the Ollama model to use (default: llama3.1:8b)
            use_gemini: Whether to use Gemini GCP instead of Ollama (default: False)
            gemini_api_key: Gemini API key (if None, will try to get from environment)
            gemini_model: Gemini model name (default: gemini-1.5-flash for free tier)
            gemini_safety_settings: Custom safety settings for Gemini (default: minimal blocking for Hebrew text)
            book_metadata: Optional dictionary containing book metadata (title, authors, summary, etc.) to provide context to the LLM
        
        Example:
            metadata = {
                "title": "From Cairo to Manchester",
                "authors": ["Renate Smithuis", "Philip S. Alexander"],
                "summary": "University of Manchester Library holds a collection..."
            }
            llm_service = StructuredJSONLLM(use_gemini=True, book_metadata=metadata)
        """
        # Set up raw data directory - use portable path
        if raw_data_dir is None:
            # Get the directory of this file and create raw_data subdirectory
            current_file_dir = Path(__file__).parent
            self.raw_data_dir = current_file_dir / "raw_data"
        else:
            self.raw_data_dir = Path(raw_data_dir)
        
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.use_gemini = use_gemini
        self.book_metadata = book_metadata
        
        if self.use_gemini:
            # Gemini configuration
            self.gemini_model = gemini_model
            self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
            
            if not self.gemini_api_key:
                raise ValueError("Gemini API key is required when use_gemini=True. Set GEMINI_API_KEY environment variable or pass gemini_api_key parameter.")
            
            # Configure Gemini
            genai.configure(api_key=self.gemini_api_key)
            
            # Set up safety settings for Hebrew text processing
            if gemini_safety_settings is None:
                # Default minimal safety settings for Hebrew historical documents
                self.gemini_safety_settings = [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH", 
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    }
                ]
            else:
                self.gemini_safety_settings = gemini_safety_settings
            
            self.gemini_model_instance = genai.GenerativeModel(
                self.gemini_model,
                safety_settings=self.gemini_safety_settings
            )
            
            logger.info(f"StructuredJSONLLM initialized with Gemini model: {gemini_model}")
            logger.info(f"Safety settings configured for Hebrew text processing")
        else:
            # Ollama configuration
            self.ollama_url = ollama_url
            self.model_name = model_name
            
            # Test Ollama connection
            self._test_ollama_connection()
            
            logger.info(f"StructuredJSONLLM initialized with Ollama model: {model_name}")
    
    def _test_ollama_connection(self):
        """Test connection to Ollama API."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                logger.info(f"Connected to Ollama. Available models: {model_names}")
                
                if not any(self.model_name in name for name in model_names):
                    logger.warning(f"Model '{self.model_name}' not found. Available models: {model_names}")
            else:
                logger.error(f"Failed to connect to Ollama: HTTP {response.status_code}")
                raise Exception(f"Cannot connect to Ollama at {self.ollama_url}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise Exception(f"Cannot connect to Ollama at {self.ollama_url}")
    
    def _load_book_metadata(self, pdf_name: str) -> Optional[Dict[str, Any]]:
        """
        Load book metadata from a JSON file named `{pdf_name}_metadata.json`.
        
        Args:
            pdf_name: Name of the PDF file (without extension)
            
        Returns:
            Dictionary containing book metadata, or None if file not found
            
        Example:
            metadata = self._load_book_metadata("friedberger_bibliography")
            if metadata:
                print(metadata.get('title', 'No title'))
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
        
        Args:
            pdf_name: Name of the source PDF
            
        Returns:
            Directory name string with provider and model info
        """
        if self.use_gemini:
            # Clean model name for directory (remove colons and special chars)
            clean_model = self.gemini_model.replace(":", "_").replace("-", "_")
            return f"{pdf_name}_structured_gemini_{clean_model}"
        else:
            # Clean model name for directory (remove colons and special chars)
            clean_model = self.model_name.replace(":", "_").replace("-", "_")
            return f"{pdf_name}_structured_ollama_{clean_model}"

    def _create_prompt(self, ocr_text: str, book_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create the structured prompt for the LLM.

        Args:
            ocr_text: Raw OCR text from the document
            book_metadata: Optional dictionary containing book metadata (title, author, etc.)

        Returns:
            Formatted prompt string
        """
        # Build metadata context if available
        metadata_context = ""
        if book_metadata:
            metadata_context = "\n\n## Book Context\n"
            metadata_context += "You are processing pages from a specific book. Here is the book's metadata:\n"
            for key, value in book_metadata.items():
                if value:
                    # Convert value to string, handling lists and nested structures
                    value_str = str(value)
                    # Replace braces to avoid format string issues - using parentheses instead
                    value_str = value_str.replace("{", "(").replace("}", ")")
                    # Use simple string concatenation to avoid any formatting issues
                    metadata_context += "- " + key.replace('_', ' ').title() + ": " + value_str + "\n"
            metadata_context += "\nUse this context to better understand the pages you are processing.\n"
        
        # Build the prompt without using f-string to avoid issues with escaped braces
        prompt = """You are an expert Judaic studies professor PhD and data annotator. You will be given both text and images of books on the Cairo Genizah. Your job is to parse the text into a consistent JSON structured format based on both the raw text extracted by OCR and your understanding from the visual of the document.""" + metadata_context + """
        Use the image ONLY to determine the structure of the document. Do not EVER attempt to transcribe from the image of the document. Your job is only is arrange the text given to 

The JSON structure should follow this exact format:
```json
{{
    "shelf_marks_mentioned": {{"TS-MS-1": "One to two line summary of mention of shelf-mark",
    "TS-MS-2":"One to two line summary of mention of shelf-mark"}},
    "footnotes": {{
        "1": "footnote 1 text here",
        "2": "footnote 2 text here"
    }},
      "transcriptions": {{
    "TS-MS-1": {{
      "0": "ברוך",
      "1": "ייי הללויה",
      "2": "מציון",
      "3": "שיכון",
      "4": "ירושלים",
      "5": "ברוך"
    }}
  }},
    "extracted_page_number": 200,
    "summary": "Summary of book page based on both main text and image",
    "classification": "transcription"
}}
```
Transcriptions should be a JSON object of lines as shown above. There are rarely multiple transcriptions 
per page but if there are then please do a list. 

Classification: Based on the text and image of the document. Please classify the document as one of the following:
    - general_academic: General Academic Analysis of Cairo Genizah documents. 
    - transcription: A transcription of a specific Cairo Genizah shelf-mark. Usually these pages will have a lot of lines
    and old Hebrew/Arabic text. They will usually be laid out in very specific style with each line. 
    - catalog: These pages will generally be very structured. Likely listing many shelf-marks and associations.
    - diagram/map: These pages are generally image based.

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
4. Provide the main text content excluding footnotes and shelf mark references
5. If no footnotes exist, use empty object {{}}
6. If no transcriptions exist, use empty object {{}}. DO NOT EVER TRY TO TRANSCRIBE IMAGES OF THE DOCUMENT. YOU ARE TO USE ONLY THE PROVIDED OCR TEXT. 
7. If no shelf marks are mentioned, use empty array []
8. Return ONLY valid JSON, no additional text or explanations

Please process the following document text:

""" + ocr_text + """

Return the structured JSON:"""

        return prompt

    def _encode_image_to_base64(self, image_path: Union[str, Path]) -> str:
        """
        Encode image to base64 for Ollama API.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                base64_string = base64.b64encode(image_data).decode('utf-8')
                return base64_string
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise
    
    def _call_ollama_with_image(self, prompt: str, image_path: Union[str, Path]) -> str:
        """
        Call Ollama API with text prompt and image.
        
        Args:
            prompt: Text prompt for the LLM
            image_path: Path to the image file
            
        Returns:
            LLM response text
        """
        try:
            # Encode image to base64
            image_base64 = self._encode_image_to_base64(image_path)
            
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "format": "json"  # Request JSON format
            }
            
            # Make API call
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=400  # 2 minute timeout for processing
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                logger.error(f"Ollama API error: HTTP {response.status_code}")
                logger.error(f"Response: {response.text}")
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to call Ollama API: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in Ollama API call: {e}")
            raise
    
    @retry_on_failure(max_retries=2, delay=2.0)
    def _call_gemini_gcp(self, prompt: str, image_path: Union[str, Path]) -> str:
        """
        Call Gemini GCP API with text prompt and image.
        
        Args:
            prompt: Text prompt for the LLM
            image_path: Path to the image file
            
        Returns:
            LLM response text
        """
        try:
            # Load and prepare image
            image = Image.open(image_path)
            
            # Create content with text and image
            content = [prompt, image]
            
            # Generate response
            response = self.gemini_model_instance.generate_content(
                content,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for structured output
                    max_output_tokens=8192,  # Increased limit for Hebrew text processing
                )
            )
            
            # Check for blocked content or other issues
            if not response.candidates:
                logger.error("Gemini returned no candidates")
                raise Exception("No candidates returned from Gemini API")
            
            candidate = response.candidates[0]
            
            # Check finish reason
            if hasattr(candidate, 'finish_reason'):
                finish_reason = candidate.finish_reason
                if finish_reason == 3:  # SAFETY
                    logger.warning("Gemini blocked content due to safety filters")
                    raise Exception("Content blocked by Gemini safety filters")
                elif finish_reason == 4:  # RECITATION
                    logger.warning("Gemini blocked content due to recitation filters")
                    raise Exception("Content blocked by Gemini recitation filters")
                elif finish_reason == 2:  # MAX_TOKENS
                    logger.warning("Gemini response truncated due to token limit")
                    raise Exception("Response truncated due to token limit - try reducing prompt size")
                elif finish_reason == 5:  # OTHER
                    logger.warning("Gemini blocked content for other reasons")
                    raise Exception("Content blocked by Gemini for other reasons")
                elif finish_reason == 6:  # LANGUAGE
                    logger.warning("Gemini blocked content due to language restrictions")
                    raise Exception("Content blocked by Gemini language restrictions")
                elif finish_reason == 7:  # BLOCKLIST
                    logger.warning("Gemini blocked content due to blocklist")
                    raise Exception("Content blocked by Gemini blocklist")
                elif finish_reason == 8:  # PROHIBITED_CONTENT
                    logger.warning("Gemini blocked content due to prohibited content")
                    raise Exception("Content blocked by Gemini prohibited content")
                elif finish_reason == 9:  # SPII
                    logger.warning("Gemini blocked content due to SPII restrictions")
                    raise Exception("Content blocked by Gemini SPII restrictions")
                elif finish_reason == 10:  # MALFORMED_FUNCTION_CALL
                    logger.warning("Gemini blocked content due to malformed function call")
                    raise Exception("Content blocked by Gemini malformed function call")
                elif finish_reason == 11:  # IMAGE_SAFETY
                    logger.warning("Gemini blocked content due to image safety")
                    raise Exception("Content blocked by Gemini image safety")
            
            # Check if response has text
            if hasattr(candidate, 'content') and candidate.content.parts:
                text_content = candidate.content.parts[0].text
                if text_content:
                    return text_content
            
            # Fallback to response.text if available
            if response.text:
                return response.text
            
            logger.error("Gemini returned empty response")
            raise Exception("Empty response from Gemini API")
                
        except Exception as e:
            logger.error(f"Error in Gemini API call: {e}")
            raise
    
    @retry_on_failure(max_retries=2, delay=2.0)
    def _call_gemini_gcp_text_only(self, prompt: str) -> str:
        """
        Call Gemini GCP API with text prompt only (no image).
        
        Args:
            prompt: Text prompt for the LLM
            
        Returns:
            LLM response text
        """
        try:
            # Generate response
            response = self.gemini_model_instance.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for structured output
                    max_output_tokens=8192,  # Increased limit for Hebrew text processing
                )
            )
            
            # Check for blocked content or other issues
            if not response.candidates:
                logger.error("Gemini returned no candidates")
                raise Exception("No candidates returned from Gemini API")
            
            candidate = response.candidates[0]
            
            # Check finish reason
            if hasattr(candidate, 'finish_reason'):
                finish_reason = candidate.finish_reason
                if finish_reason == 3:  # SAFETY
                    logger.warning("Gemini blocked content due to safety filters")
                    raise Exception("Content blocked by Gemini safety filters")
                elif finish_reason == 4:  # RECITATION
                    logger.warning("Gemini blocked content due to recitation filters")
                    raise Exception("Content blocked by Gemini recitation filters")
                elif finish_reason == 2:  # MAX_TOKENS
                    logger.warning("Gemini response truncated due to token limit")
                    raise Exception("Response truncated due to token limit - try reducing prompt size")
                elif finish_reason == 5:  # OTHER
                    logger.warning("Gemini blocked content for other reasons")
                    raise Exception("Content blocked by Gemini for other reasons")
                elif finish_reason == 6:  # LANGUAGE
                    logger.warning("Gemini blocked content due to language restrictions")
                    raise Exception("Content blocked by Gemini language restrictions")
                elif finish_reason == 7:  # BLOCKLIST
                    logger.warning("Gemini blocked content due to blocklist")
                    raise Exception("Content blocked by Gemini blocklist")
                elif finish_reason == 8:  # PROHIBITED_CONTENT
                    logger.warning("Gemini blocked content due to prohibited content")
                    raise Exception("Content blocked by Gemini prohibited content")
                elif finish_reason == 9:  # SPII
                    logger.warning("Gemini blocked content due to SPII restrictions")
                    raise Exception("Content blocked by Gemini SPII restrictions")
                elif finish_reason == 10:  # MALFORMED_FUNCTION_CALL
                    logger.warning("Gemini blocked content due to malformed function call")
                    raise Exception("Content blocked by Gemini malformed function call")
                elif finish_reason == 11:  # IMAGE_SAFETY
                    logger.warning("Gemini blocked content due to image safety")
                    raise Exception("Content blocked by Gemini image safety")
            
            # Check if response has text
            if hasattr(candidate, 'content') and candidate.content.parts:
                text_content = candidate.content.parts[0].text
                if text_content:
                    return text_content
            
            # Fallback to response.text if available
            if response.text:
                return response.text
            
            logger.error("Gemini returned empty response")
            raise Exception("Empty response from Gemini API")
                
        except Exception as e:
            logger.error(f"Error in Gemini API call: {e}")
            raise
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response and extract JSON with robust error handling.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Parsed JSON dictionary
        """
        try:
            # Try to extract JSON from response
            response_text = response_text.strip()
            
            # Look for JSON block in markdown format
            if '```json' in response_text:
                start = response_text.find('```json') + 7
                end = response_text.find('```', start)
                if end != -1:
                    json_text = response_text[start:end].strip()
                else:
                    json_text = response_text[start:].strip()
            elif '```' in response_text:
                start = response_text.find('```') + 3
                end = response_text.find('```', start)
                if end != -1:
                    json_text = response_text[start:end].strip()
                else:
                    json_text = response_text[start:].strip()
            else:
                json_text = response_text
            
            # Try to fix common JSON issues
            json_text = self._fix_common_json_issues(json_text)
            
            # Parse JSON
            parsed_json = json.loads(json_text)
            
            # Validate required fields
            required_fields = ['shelf_marks_mentioned', 'footnotes', 'transcriptions', 'full_main_text']
            for field in required_fields:
                if field not in parsed_json:
                    logger.warning(f"Missing required field '{field}' in LLM response")
                    parsed_json[field] = [] if field == 'shelf_marks_mentioned' else {}
            
            return parsed_json
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.error(f"Response text: {response_text[:500]}...")
            
            # Try to extract partial data from malformed JSON
            partial_data = self._extract_partial_json(response_text)
            if partial_data:
                logger.info("Successfully extracted partial data from malformed JSON")
                return partial_data
            
            # Return default structure if parsing fails completely
            return {
                'shelf_marks_mentioned': [],
                'footnotes': {},
                'transcriptions': {},
                'full_main_text': response_text[:1000] if response_text else "Failed to parse response"
            }
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            raise
    
    def _fix_common_json_issues(self, json_text: str) -> str:
        """
        Fix common JSON formatting issues.
        
        Args:
            json_text: Raw JSON text
            
        Returns:
            Fixed JSON text
        """
        # Remove trailing commas before closing braces/brackets
        json_text = json_text.replace(',}', '}').replace(',]', ']')
        
        # Fix unterminated strings by finding the last quote and adding closing quote
        # This is a simple heuristic - might need more sophisticated handling
        lines = json_text.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Check for unterminated strings (odd number of quotes)
            quote_count = line.count('"')
            if quote_count % 2 == 1 and line.strip().endswith('"'):
                # Line ends with quote but has odd count - might be unterminated
                line = line.rstrip() + '"'
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _extract_partial_json(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Try to extract partial JSON data from malformed response.
        
        Args:
            response_text: Raw response text
            
        Returns:
            Partial JSON data or None if extraction fails
        """
        try:
            # Look for specific patterns in the response
            partial_data = {
                'shelf_marks_mentioned': [],
                'footnotes': {},
                'transcriptions': {},
                'full_main_text': response_text[:1000] if response_text else "Failed to parse response"
            }
            
            # Try to extract shelf marks using regex
            import re
            
            # Look for shelf mark patterns
            shelf_patterns = [
                r'"([A-Z]\s+\d+[-\d]*)"',  # A 1649-1, B 3014-2, etc.
                r'"([T-S]\s+[A-Z0-9\.]+)"',  # T-S A43.1, etc.
                r'"([ENA]\s+\d+[\.\d]*)"',  # ENA 2689.11, etc.
            ]
            
            shelf_marks = []
            for pattern in shelf_patterns:
                matches = re.findall(pattern, response_text)
                shelf_marks.extend(matches)
            
            if shelf_marks:
                partial_data['shelf_marks_mentioned'] = list(set(shelf_marks))  # Remove duplicates
            
            return partial_data
            
        except Exception as e:
            logger.debug(f"Failed to extract partial JSON: {e}")
            return None
    
    def process_page(self, ocr_text: str, image_path: Union[str, Path], 
                    page_number: int, pdf_name: str) -> Dict[str, Any]:
        """
        Process a single page through the LLM to extract structured data.
        
        Args:
            ocr_text: Raw OCR text from the page
            image_path: Path to the page image
            page_number: Page number for identification
            pdf_name: Name of the source PDF
            
        Returns:
            Structured JSON data
        """
        if self.use_gemini:
            return self.process_page_with_gemini(ocr_text, image_path, page_number, pdf_name)
        else:
            return self.process_page_with_ollama(ocr_text, image_path, page_number, pdf_name)
    
    def process_page_with_ollama(self, ocr_text: str, image_path: Union[str, Path], 
                               page_number: int, pdf_name: str) -> Dict[str, Any]:
        """
        Process a single page through Ollama LLM to extract structured data.
        
        Args:
            ocr_text: Raw OCR text from the page
            image_path: Path to the page image
            page_number: Page number for identification
            pdf_name: Name of the source PDF
            
        Returns:
            Structured JSON data
        """
        try:
            logger.info(f"Processing page {page_number} from {pdf_name} with Ollama")
            
            # Use provided book metadata or load from file
            book_metadata = self.book_metadata
            if book_metadata is None:
                book_metadata = self._load_book_metadata(pdf_name)
            
            # Create prompt with metadata
            prompt = self._create_prompt(ocr_text, book_metadata=book_metadata)
            
            # Call Ollama with image
            llm_response = self._call_ollama_with_image(prompt, image_path)
            
            # Parse response
            structured_data = self._parse_llm_response(llm_response)
            
            # Add metadata
            structured_data['metadata'] = {
                'page_number': page_number,
                'pdf_name': pdf_name,
                'processed_at': str(Path().cwd()),
                'model_used': self.model_name,
                'provider': 'ollama'
            }
            
            logger.info(f"Successfully processed page {page_number} with Ollama")
            return structured_data
            
        except Exception as e:
            logger.error(f"Failed to process page {page_number} with Ollama: {e}")
            raise
    
    def process_page_with_gemini(self, ocr_text: str, image_path: Union[str, Path], 
                               page_number: int, pdf_name: str) -> Dict[str, Any]:
        """
        Process a single page through Gemini GCP to extract structured data.
        
        Args:
            ocr_text: Raw OCR text from the page
            image_path: Path to the page image
            page_number: Page number for identification
            pdf_name: Name of the source PDF
            
        Returns:
            Structured JSON data
        """
        try:
            logger.info(f"Processing page {page_number} from {pdf_name} with Gemini")
            
            # Use provided book metadata or load from file
            book_metadata = self.book_metadata
            if book_metadata is None:
                book_metadata = self._load_book_metadata(pdf_name)
            
            # Create prompt with metadata
            prompt = self._create_prompt(ocr_text, book_metadata=book_metadata)
            
            # Call Gemini with image
            llm_response = self._call_gemini_gcp(prompt, image_path)
            
            # Parse response
            structured_data = self._parse_llm_response(llm_response)
            
            # Add metadata
            structured_data['metadata'] = {
                'page_number': page_number,
                'pdf_name': pdf_name,
                'processed_at': str(Path().cwd()),
                'model_used': self.gemini_model,
                'provider': 'gemini_gcp'
            }
            
            logger.info(f"Successfully processed page {page_number} with Gemini")
            return structured_data
            
        except Exception as e:
            logger.error(f"Failed to process page {page_number} with Gemini: {e}")
            raise
    
    def process_from_ocr_service(self, ocr_service_result: Dict[str, str], page_number=1) -> Dict[str, Any]:
        """
        Process OCR results directly from BookOCRService.process_pdf_to_file() output.
        
        Args:
            ocr_service_result: Dictionary from BookOCRService.process_pdf_to_file()
                containing 'ocr_results_path' and 'images_dir'
            
        Returns:
            Dictionary with processing results
        """
        ocr_results_path = ocr_service_result['ocr_results_path']
        images_dir = ocr_service_result.get('images_dir')
        
        logger.info(f"Processing OCR results from service: {ocr_results_path}")
        if images_dir:
            logger.info(f"Using images directory: {images_dir}")
        
        # Use the parent directory of OCR results as the base output directory
        # The process_ocr_results method will create the appropriately named subdirectory
        return self.process_ocr_results(ocr_results_path, output_dir=Path(ocr_results_path).parent, starting_page_number=page_number)
    
    def process_ocr_results(self, ocr_results_path: Union[str, Path], output_dir: Optional[Path] = None, starting_page_number=1) -> Dict[str, Any]:
        """
        Process OCR results file and create structured JSON for each page.
        
        Args:
            ocr_results_path: Path to the OCR results JSON file
            output_dir: Directory to save structured results (optional, defaults to self.raw_data_dir)
            
        Returns:
            Dictionary with processing results
        """
        ocr_results_path = Path(ocr_results_path)
        
        try:
            # Load OCR results
            with open(ocr_results_path, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
            
            pdf_name = Path(ocr_data['pdf_path']).stem
            logger.info(f"Processing OCR results for PDF: {pdf_name}")
            
            # Create output directory for structured results with provider and model info
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
                    'gemini_safety_blocks': 0,
                    'gemini_token_limit': 0,
                    'gemini_language_blocks': 0,
                    'json_parsing_errors': 0,
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
                        structured_data = self.process_page(full_text, image_path, page_number, pdf_name)
                    else:
                        # Fallback to text-only processing
                        logger.warning(f"No image available for page {page_number}, using text-only processing")
                        structured_data = self.process_page_text_only(full_text, page_number, pdf_name)
                    
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
                    
                    if 'blocked by Gemini safety filters' in error_message:
                        error_type = 'gemini_safety_blocks'
                    elif 'truncated due to token limit' in error_message:
                        error_type = 'gemini_token_limit'
                    elif 'language restrictions' in error_message:
                        error_type = 'gemini_language_blocks'
                    elif 'Failed to parse JSON' in error_message:
                        error_type = 'json_parsing_errors'
                    elif 'API error' in error_message or 'API call' in error_message:
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
            
            logger.info(f"Completed processing {pdf_name}: {results['processed_pages']} pages processed, {results['failed_pages']} failed")
            
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
    
    def process_page_text_only(self, ocr_text: str, page_number: int, pdf_name: str) -> Dict[str, Any]:
        """
        Process a page with text-only (no image) for cases where image is not available.
        
        Args:
            ocr_text: Raw OCR text from the page
            page_number: Page number for identification
            pdf_name: Name of the source PDF
            
        Returns:
            Structured JSON data
        """
        if self.use_gemini:
            return self.process_page_text_only_gemini(ocr_text, page_number, pdf_name)
        else:
            return self.process_page_text_only_ollama(ocr_text, page_number, pdf_name)
    
    def process_page_text_only_ollama(self, ocr_text: str, page_number: int, pdf_name: str) -> Dict[str, Any]:
        """
        Process a page with text-only using Ollama (no image) for cases where image is not available.
        
        Args:
            ocr_text: Raw OCR text from the page
            page_number: Page number for identification
            pdf_name: Name of the source PDF
            
        Returns:
            Structured JSON data
        """
        try:
            logger.info(f"Processing page {page_number} from {pdf_name} (text-only with Ollama)")
            
            # Use provided book metadata or load from file
            book_metadata = self.book_metadata
            if book_metadata is None:
                book_metadata = self._load_book_metadata(pdf_name)
            
            # Create prompt with metadata
            prompt = self._create_prompt(ocr_text, book_metadata=book_metadata)
            
            # Call Ollama without image
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=410
            )
            
            if response.status_code == 200:
                result = response.json()
                llm_response = result.get('response', '')
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
            
            # Parse response
            structured_data = self._parse_llm_response(llm_response)
            
            # Add metadata
            structured_data['metadata'] = {
                'page_number': page_number,
                'pdf_name': pdf_name,
                'processed_at': str(Path().cwd()),
                'model_used': self.model_name,
                'processing_mode': 'text_only',
                'provider': 'ollama'
            }
            
            logger.info(f"Successfully processed page {page_number} (text-only with Ollama)")
            return structured_data
            
        except Exception as e:
            logger.error(f"Failed to process page {page_number} (text-only with Ollama): {e}")
            raise
    
    def process_page_text_only_gemini(self, ocr_text: str, page_number: int, pdf_name: str) -> Dict[str, Any]:
        """
        Process a page with text-only using Gemini (no image) for cases where image is not available.
        
        Args:
            ocr_text: Raw OCR text from the page
            page_number: Page number for identification
            pdf_name: Name of the source PDF
            
        Returns:
            Structured JSON data
        """
        try:
            logger.info(f"Processing page {page_number} from {pdf_name} (text-only with Gemini)")
            
            # Use provided book metadata or load from file
            book_metadata = self.book_metadata
            if book_metadata is None:
                book_metadata = self._load_book_metadata(pdf_name)
            
            # Create prompt with metadata
            prompt = self._create_prompt(ocr_text, book_metadata=book_metadata)
            
            # Call Gemini without image
            llm_response = self._call_gemini_gcp_text_only(prompt)
            
            # Parse response
            structured_data = self._parse_llm_response(llm_response)
            
            # Add metadata
            structured_data['metadata'] = {
                'page_number': page_number,
                'pdf_name': pdf_name,
                'processed_at': str(Path().cwd()),
                'model_used': self.gemini_model,
                'processing_mode': 'text_only',
                'provider': 'gemini_gcp'
            }
            
            logger.info(f"Successfully processed page {page_number} (text-only with Gemini)")
            return structured_data
            
        except Exception as e:
            logger.error(f"Failed to process page {page_number} (text-only with Gemini): {e}")
            raise


def main():
    """Example usage of the StructuredJSONLLM service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process OCR results with LLM to extract structured JSON")
    parser.add_argument("ocr_results_path", help="Path to OCR results JSON file")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model name")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama API URL")
    parser.add_argument("--use-gemini", action="store_true", help="Use Gemini GCP instead of Ollama")
    parser.add_argument("--gemini-model", default="gemini-1.5-flash", help="Gemini model name")
    parser.add_argument("--gemini-api-key", help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--gemini-safety-minimal", action="store_true", help="Use minimal safety settings for Hebrew text")
    
    args = parser.parse_args()
    
    # Set up safety settings if requested
    gemini_safety_settings = None
    if args.use_gemini and args.gemini_safety_minimal:
        gemini_safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
    
    try:
        # Initialize service
        llm_service = StructuredJSONLLM(
            model_name=args.model,
            ollama_url=args.ollama_url,
            use_gemini=args.use_gemini,
            gemini_model=args.gemini_model,
            gemini_api_key=args.gemini_api_key,
            gemini_safety_settings=gemini_safety_settings
        )
        
        # Process OCR results
        results = llm_service.process_ocr_results(args.ocr_results_path)
        
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
