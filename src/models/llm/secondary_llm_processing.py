#!/usr/bin/env python3
"""
Secondary LLM Processing for Enhanced Bibliography Analysis

This module provides a second-stage LLM processing service that addresses context loss
from the initial structured_json_llm.py processing. It performs:

1. Multi-page context analysis to better link shelf marks to transcriptions
2. Explicit extraction of people and locations mentioned in documents
3. Cross-reference analysis to improve accuracy of shelf mark associations
4. Enhanced structured output with improved linking

Usage:
    from secondary_llm_processing import SecondaryLLMProcessor
    
    processor = SecondaryLLMProcessor()
    enhanced_data = processor.process_multiple_pages(structured_data_files)
"""

import json
import logging
import os
import base64
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import requests
from collections import defaultdict, Counter
import re

import dotenv

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


class SecondaryLLMProcessor:
    """
    Secondary LLM processing service for enhanced bibliography analysis.
    
    This class handles:
    - Multi-page context analysis
    - Improved shelf mark to transcription linking
    - People and location extraction
    - Cross-page reference resolution
    - Enhanced structured output
    """
    
    def __init__(self, 
                 ollama_url: str = "http://localhost:11434",
                 model_name: str = "llama3.1:8b",
                 context_window_pages: int = 5):
        """
        Initialize the SecondaryLLMProcessor service.
        
        Args:
            ollama_url: URL for Ollama API (default: http://localhost:11434)
            model_name: Name of the Ollama model to use (default: llama3.1:8b)
            context_window_pages: Number of pages to include in context window (default: 5)
        """
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.context_window_pages = context_window_pages
        
        # Test Ollama connection
        self._test_ollama_connection()
        
        logger.info(f"SecondaryLLMProcessor initialized with model: {model_name}")
    
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
    
    def _create_context_analysis_prompt(self, pages_data: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for multi-page context analysis.
        
        Args:
            pages_data: List of structured data from multiple pages
            
        Returns:
            Formatted prompt string for context analysis
        """
        # Prepare context information
        context_info = []
        for i, page_data in enumerate(pages_data):
            page_num = page_data.get('metadata', {}).get('page_number', i+1)
            
            # Safely get full_main_text as a string
            main_text = page_data.get('full_main_text', '')
            if not isinstance(main_text, str):
                main_text = str(main_text)
            main_text_excerpt = main_text[:200] if main_text else ''
            
            context_info.append(f"""
Page {page_num}:
- Shelf marks mentioned: {page_data.get('shelf_marks_mentioned', [])}
- Transcriptions: {list(page_data.get('transcriptions', {}).keys())}
- Main text excerpt: {main_text_excerpt}...
""")
        
        context_text = "\n".join(context_info)
        
        prompt = f"""You are an expert Judaic studies professor PhD specializing in Cairo Genizah manuscripts. You are analyzing multiple pages from a bibliography book to improve the linking between shelf marks and transcriptions, and to extract people and locations.

CONTEXT - Multiple pages from the same document:
{context_text}

Your task is to perform a comprehensive analysis across these pages to:

1. IMPROVE SHELF MARK TO TRANSCRIPTION LINKING:
   - Identify shelf marks that are mentioned but don't have associated transcriptions
   - Find transcriptions that don't have clear shelf mark associations
   - Use cross-page context to link shelf marks to their transcriptions
   - Distinguish between shelf marks that are just mentioned vs. those with actual transcriptions

2. EXTRACT PEOPLE AND LOCATIONS:
   - Identify all personal names (Hebrew, Arabic, English, etc.)
   - Identify all geographical locations and institutions
   - Note the context in which they appear (author, scribe, place of origin, etc.)

3. CROSS-PAGE REFERENCE RESOLUTION:
   - Find references that span multiple pages
   - Resolve ambiguous shelf mark references
   - Identify patterns in how shelf marks are presented

Return your analysis in this exact JSON format:
```json
{{
    "enhanced_shelf_mark_transcriptions": {{
        "TS-MS-1": {{
            "transcription": "דוגמה לתעתוק עברי",
            "confidence": "high",
            "context_pages": [1, 2],
            "linking_evidence": "Explicitly linked on page 1, transcription appears on page 2"
        }},
        "TS-MS-2": {{
            "transcription": "עוד דוגמה לתעתוק",
            "confidence": "medium",
            "context_pages": [1],
            "linking_evidence": "Mentioned in footnote, transcription in main text"
        }}
    }},
    "people_mentioned": [
        {{
            "name": "שמואל בן יהודה",
            "name_variants": ["Samuel b. Judah", "Shmuel ben Yehuda"],
            "role": "author",
            "context": "Author of the manuscript discussed",
            "pages_mentioned": [1, 3]
        }},
        {{
            "name": "Maimonides",
            "name_variants": ["Rambam", "Moses ben Maimon"],
            "role": "subject",
            "context": "Subject of the manuscript",
            "pages_mentioned": [2, 4]
        }}
    ],
    "locations_mentioned": [
        {{
            "name": "Cairo",
            "name_variants": ["Fustat", "القاهرة"],
            "type": "city",
            "context": "Place where manuscript was found",
            "pages_mentioned": [1, 2]
        }},
        {{
            "name": "Cambridge University Library",
            "name_variants": ["CUL", "Cambridge"],
            "type": "institution",
            "context": "Current repository",
            "pages_mentioned": [1, 3]
        }}
    ],
    "cross_page_references": [
        {{
            "reference_type": "shelf_mark_continuation",
            "description": "TS-MS-1 mentioned on page 1, transcription continues on page 2",
            "pages_involved": [1, 2],
            "confidence": "high"
        }}
    ],
    "analysis_summary": "Summary of key findings and improvements made"
}}
```

Guidelines:
1. Be conservative with confidence levels - only mark as "high" when you're certain
2. Include all name variants you can identify
3. Focus on improving the shelf mark-transcription linking using cross-page context
4. Extract both Hebrew/Arabic and English names
5. Include institutional names as locations
6. Return ONLY valid JSON, no additional text

Analyze the provided pages and return the enhanced structured data:"""
        
        return prompt
    
    def _create_people_locations_prompt(self, text_content: str) -> str:
        """
        Create a focused prompt for people and location extraction.
        
        Args:
            text_content: Combined text content from multiple pages
            
        Returns:
            Formatted prompt string for people/location extraction
        """
        prompt = f"""You are an expert in Judaic studies and Middle Eastern history. Extract all people and locations mentioned in this text from Cairo Genizah bibliography documents.

Text to analyze:
{text_content[:3000]}...

Extract and return in this exact JSON format:
```json
{{
    "people": [
        {{
            "name": "Full name as it appears",
            "name_variants": ["Alternative spellings", "Transliterations"],
            "role": "author|scribe|subject|scholar|other",
            "context": "Brief description of how they're mentioned",
            "confidence": "high|medium|low"
        }}
    ],
    "locations": [
        {{
            "name": "Location name as it appears",
            "name_variants": ["Alternative names", "Transliterations"],
            "type": "city|region|country|institution|synagogue|other",
            "context": "Brief description of how it's mentioned",
            "confidence": "high|medium|low"
        }}
    ]
}}
```

Guidelines:
1. Include Hebrew, Arabic, and English names
2. Capture all variants and transliterations
3. Be specific about roles and types
4. Include institutional names (libraries, universities, etc.)
5. Only include entities you're confident about
6. Return ONLY valid JSON

Extract the people and locations:"""
        
        return prompt
    
    def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama API with text prompt.
        
        Args:
            prompt: Text prompt for the LLM
            
        Returns:
            LLM response text
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                logger.error(f"Ollama API error: HTTP {response.status_code}")
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to call Ollama API: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in Ollama API call: {e}")
            raise
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response and extract JSON.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Parsed JSON dictionary
        """
        try:
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
            
            parsed_json = json.loads(json_text)
            return parsed_json
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.error(f"Response text: {response_text[:500]}...")
            return {}
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return {}
    
    def _load_structured_data_files(self, structured_dir: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load all structured data files from a directory.
        
        Args:
            structured_dir: Directory containing structured JSON files
            
        Returns:
            List of structured data dictionaries
        """
        structured_dir = Path(structured_dir)
        pages_data = []
        
        # Find all structured JSON files
        json_files = sorted(structured_dir.glob("*_structured.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    page_data = json.load(f)
                    pages_data.append(page_data)
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")
        
        # Sort by page number
        pages_data.sort(key=lambda x: x.get('metadata', {}).get('page_number', 0))
        
        logger.info(f"Loaded {len(pages_data)} structured data files")
        return pages_data
    
    def _analyze_shelf_mark_patterns(self, pages_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze shelf mark patterns across pages to improve linking.
        
        Args:
            pages_data: List of structured data from multiple pages
            
        Returns:
            Analysis of shelf mark patterns
        """
        shelf_mark_mentions = defaultdict(list)
        transcriptions_by_page = defaultdict(dict)
        
        # Collect shelf mark mentions and transcriptions
        for page_data in pages_data:
            page_num = page_data.get('metadata', {}).get('page_number', 0)
            
            # Track shelf mark mentions
            for shelf_mark in page_data.get('shelf_marks_mentioned', []):
                shelf_mark_mentions[shelf_mark].append(page_num)
            
            # Track transcriptions
            transcriptions = page_data.get('transcriptions', {})
            transcriptions_by_page[page_num] = transcriptions
        
        # Analyze patterns
        analysis = {
            'shelf_mark_frequency': dict(shelf_mark_mentions),
            'transcriptions_by_page': dict(transcriptions_by_page),
            'potential_links': [],
            'unlinked_shelf_marks': [],
            'unlinked_transcriptions': []
        }
        
        # Find potential links
        all_transcription_shelf_marks = set()
        for page_transcriptions in transcriptions_by_page.values():
            all_transcription_shelf_marks.update(page_transcriptions.keys())
        
        all_mentioned_shelf_marks = set(shelf_mark_mentions.keys())
        
        # Find unlinked shelf marks
        analysis['unlinked_shelf_marks'] = list(all_mentioned_shelf_marks - all_transcription_shelf_marks)
        
        # Find unlinked transcriptions
        analysis['unlinked_transcriptions'] = list(all_transcription_shelf_marks - all_mentioned_shelf_marks)
        
        return analysis
    
    def process_multiple_pages(self, structured_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Process multiple pages with enhanced context analysis.
        
        Args:
            structured_dir: Directory containing structured JSON files from first-stage processing
            
        Returns:
            Enhanced structured data with improved linking and extracted entities
        """
        try:
            logger.info(f"Starting secondary processing of structured data in {structured_dir}")
            
            # Load structured data files
            pages_data = self._load_structured_data_files(structured_dir)
            
            if not pages_data:
                logger.warning("No structured data files found")
                return {}
            
            # Analyze shelf mark patterns
            pattern_analysis = self._analyze_shelf_mark_patterns(pages_data)
            
            # Create context analysis prompt
            context_prompt = self._create_context_analysis_prompt(pages_data)
            
            # Call LLM for context analysis
            logger.info("Performing multi-page context analysis...")
            context_response = self._call_ollama(context_prompt)
            context_analysis = self._parse_llm_response(context_response)
            
            # Extract people and locations from combined text
            combined_text = " ".join([
                page_data.get('full_main_text', '') + " " + 
                " ".join(page_data.get('footnotes', {}).values())
                for page_data in pages_data
            ])
            
            people_locations_prompt = self._create_people_locations_prompt(combined_text)
            logger.info("Extracting people and locations...")
            people_locations_response = self._call_ollama(people_locations_prompt)
            people_locations = self._parse_llm_response(people_locations_response)
            
            # Combine results
            enhanced_data = {
                'original_pages': pages_data,
                'pattern_analysis': pattern_analysis,
                'context_analysis': context_analysis,
                'people_locations': people_locations,
                'processing_metadata': {
                    'total_pages': len(pages_data),
                    'model_used': self.model_name,
                    'context_window_pages': self.context_window_pages,
                    'processing_timestamp': str(Path().cwd())
                }
            }
            
            logger.info("Secondary processing completed successfully")
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Failed to process multiple pages: {e}")
            raise
    
    def save_enhanced_data(self, enhanced_data: Dict[str, Any], 
                          output_path: Union[str, Path]) -> None:
        """
        Save enhanced data to file.
        
        Args:
            enhanced_data: Enhanced structured data
            output_path: Path to save the enhanced data
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Enhanced data saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save enhanced data: {e}")
            raise
    
    def process_from_structured_dir(self, structured_dir: Union[str, Path], 
                                   output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Complete processing pipeline from structured directory to enhanced output.
        
        Args:
            structured_dir: Directory containing structured JSON files
            output_path: Optional path to save enhanced data
            
        Returns:
            Enhanced structured data
        """
        try:
            # Process multiple pages
            enhanced_data = self.process_multiple_pages(structured_dir)
            
            # Save if output path provided
            if output_path:
                self.save_enhanced_data(enhanced_data, output_path)
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Failed to process from structured directory: {e}")
            raise


def main():
    """Example usage of the SecondaryLLMProcessor service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Secondary LLM processing for enhanced bibliography analysis")
    parser.add_argument("structured_dir", help="Directory containing structured JSON files")
    parser.add_argument("--output", help="Output path for enhanced data")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model name")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama API URL")
    
    args = parser.parse_args()
    
    try:
        # Initialize service
        processor = SecondaryLLMProcessor(
            model_name=args.model,
            ollama_url=args.ollama_url
        )
        
        # Process structured data
        enhanced_data = processor.process_from_structured_dir(
            args.structured_dir,
            args.output
        )
        
        print(f"Secondary processing completed:")
        print(f"  - Pages processed: {enhanced_data.get('processing_metadata', {}).get('total_pages', 0)}")
        print(f"  - Enhanced shelf mark transcriptions: {len(enhanced_data.get('context_analysis', {}).get('enhanced_shelf_mark_transcriptions', {}))}")
        print(f"  - People extracted: {len(enhanced_data.get('people_locations', {}).get('people', []))}")
        print(f"  - Locations extracted: {len(enhanced_data.get('people_locations', {}).get('locations', []))}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
