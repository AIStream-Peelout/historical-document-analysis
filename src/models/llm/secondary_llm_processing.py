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
import asyncio
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

# ---------------------------------------------------------------------------
# Backend constants
# ---------------------------------------------------------------------------
BACKEND_OLLAMA  = "ollama"   # local LM Studio / Ollama endpoint
BACKEND_GEMINI  = "gemini"   # Google Gemini via google-generativeai SDK

GEMINI_FLASH_MODEL = "gemini-3-flash-preview"   # matches AgentConfig


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
                 backend: str = BACKEND_OLLAMA,
                 ollama_url: str = "http://localhost:1234/v1",
                 model_name: str = "c4ai-command-r-v01",
                 gemini_model: str = GEMINI_FLASH_MODEL,
                 gemini_api_key: Optional[str] = None,
                 context_window_pages: int = 5):
        """
        Initialize the SecondaryLLMProcessor service.

        Args:
            backend:             "ollama" (LM Studio / local) or "gemini".
                                 Automatically promoted to "gemini" when a
                                 *gemini_api_key* is supplied and *backend* was
                                 not explicitly set to "ollama".
            ollama_url:          Base URL for the Ollama/LM Studio endpoint.
            model_name:          Model name used when backend="ollama".
            gemini_model:        Gemini model name when backend="gemini".
                                 Defaults to gemini-3-flash-preview.
            gemini_api_key:      Gemini API key. Falls back to GEMINI_API_KEY env var.
            context_window_pages: Number of pages included in each context window.
        """
        # Resolve the API key first so the auto-promote check can use it.
        resolved_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

        # Auto-promote to Gemini when a key is available and the caller did not
        # explicitly request Ollama.  This avoids a silent fallback to a local
        # model that may not be running.
        if backend == BACKEND_OLLAMA and resolved_api_key:
            backend = BACKEND_GEMINI
            logger.info("gemini_api_key provided — switching backend to 'gemini' automatically")

        if backend not in (BACKEND_OLLAMA, BACKEND_GEMINI):
            raise ValueError(f"backend must be '{BACKEND_OLLAMA}' or '{BACKEND_GEMINI}', got '{backend}'")

        self.backend              = backend
        self.ollama_url           = ollama_url
        self.model_name           = model_name
        self.gemini_model         = gemini_model
        self.gemini_api_key       = resolved_api_key
        self.context_window_pages = context_window_pages

        if self.backend == BACKEND_OLLAMA:
            self._test_ollama_connection()
        else:
            self._test_gemini_connection()

        logger.info(f"SecondaryLLMProcessor initialised — backend={backend}, "
                    f"model={gemini_model if backend == BACKEND_GEMINI else model_name}")

    def _test_ollama_connection(self):
        """Test connection to Ollama / LM Studio API."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                logger.info(f"Connected to Ollama. Available models: {model_names}")
                if not any(self.model_name in name for name in model_names):
                    logger.warning(f"Model '{self.model_name}' not found in {model_names}")
            else:
                raise Exception(f"Cannot connect to Ollama at {self.ollama_url}: HTTP {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Cannot connect to Ollama at {self.ollama_url}: {e}")

    def _test_gemini_connection(self):
        """Verify that the Gemini API key is present."""
        if not self.gemini_api_key:
            raise ValueError("Gemini backend requires GEMINI_API_KEY env var or gemini_api_key param.")
    
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

    def _create_kg_triplets_prompt(self, text_content: str, people: list, locations: list) -> str:
        """
        Create a prompt for extracting KG triplets (subject → relation → object).

        Uses the already-extracted people and locations as anchors so the LLM
        focuses on relationships rather than re-doing NER.

        Supported relation types mirror the Neo4j schema:
          Person  → LIVED_IN       → Place
          Person  → TRAVELED_TO    → Place
          Person  → WROTE          → Fragment   (shelf mark)
          Person  → MENTIONED_IN   → Fragment   (shelf mark)
          Place   → MENTIONED_IN   → Fragment   (shelf mark)
          Scholar → WROTE          → BookArticle (title)
          BookArticle → CITES      → BookArticle (title)

        Args:
            text_content: Combined text from all pages (truncated)
            people: List of person dicts already extracted
            locations: List of location dicts already extracted

        Returns:
            Formatted prompt string
        """
        people_list  = ", ".join(p['name'] for p in people[:30])  if people    else "none identified"
        loc_list     = ", ".join(l['name'] for l in locations[:30]) if locations else "none identified"

        prompt = f"""You are an expert in Judaic studies building a knowledge graph of Cairo Genizah scholarship.

Previously identified entities:
  People:    {people_list}
  Locations: {loc_list}

Text to analyze (excerpt):
{text_content[:4000]}

Extract knowledge-graph triplets connecting these entities. Use ONLY these relation types:
  - LIVED_IN        : person permanently resided in a place
  - TRAVELED_TO     : person traveled to or visited a place
  - WROTE           : person wrote a manuscript (use shelf mark) or book/article (use title)
  - MENTIONED_IN    : person or place is mentioned in a manuscript (use shelf mark)
  - ORIGINATED_FROM : document or person originated from a place
  - SENT_TO         : document was sent to a place or person
  - CITES           : one book/article cites another (use titles)

Return ONLY valid JSON in this exact format:
```json
{{
    "triplets": [
        {{
            "subject":       "Exact name of person, place, or shelf mark",
            "subject_type":  "Person|Place|Fragment|BookArticle",
            "relation":      "LIVED_IN",
            "object":        "Exact name of person, place, or shelf mark",
            "object_type":   "Person|Place|Fragment|BookArticle",
            "evidence":      "The sentence or phrase that supports this triplet",
            "confidence":    "high|medium|low"
        }}
    ]
}}
```

Rules:
1. Only extract triplets directly supported by the text — do not infer.
2. Use the canonical name form from the entity lists above where possible.
3. For shelf marks use the format as written (e.g. "T-S 13J7.4", "ENA 2713.17").
4. Omit triplets with confidence "low" unless the evidence is a direct quote.
5. Return ONLY valid JSON, no commentary.

Extract the triplets:"""

        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Route the prompt to whichever backend is active and return the text response."""
        if self.backend == BACKEND_GEMINI:
            return self._call_gemini(prompt)
        return self._call_ollama(prompt)

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama / LM Studio API with a text prompt."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            }
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=300,
            )
            if response.status_code == 200:
                return response.json().get('response', '')
            raise Exception(f"Ollama API error: HTTP {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to call Ollama API: {e}")
            raise

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini Flash via google-generativeai SDK (sync wrapper)."""
        try:
            import google.generativeai as genai  # noqa: PLC0415
        except ImportError:
            raise ImportError("google-generativeai is required for the Gemini backend. "
                              "Run: pip install google-generativeai")

        genai.configure(api_key=self.gemini_api_key)
        model = genai.GenerativeModel(self.gemini_model)

        async def _async_call():
            response = await model.generate_content_async(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )
            return response.text

        try:
            # Re-use a running loop if one exists (e.g. Jupyter), else create one.
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, _async_call())
                    return future.result(timeout=300)
            else:
                return loop.run_until_complete(_async_call())
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
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
            context_response = self._call_llm(context_prompt)
            context_analysis = self._parse_llm_response(context_response)
            
            # Extract people and locations from combined text
            combined_text = " ".join([
                page_data.get('full_main_text', '') + " " +
                " ".join(page_data.get('footnotes', {}).values())
                for page_data in pages_data
            ])

            people_locations_prompt = self._create_people_locations_prompt(combined_text)
            logger.info("Extracting people and locations...")
            people_locations_response = self._call_llm(people_locations_prompt)
            people_locations = self._parse_llm_response(people_locations_response)

            # Extract KG triplets using identified entities as anchors
            logger.info("Extracting knowledge graph triplets...")
            kg_triplets_prompt = self._create_kg_triplets_prompt(
                combined_text,
                people_locations.get('people', []),
                people_locations.get('locations', []),
            )
            kg_triplets_response = self._call_llm(kg_triplets_prompt)
            kg_triplets = self._parse_llm_response(kg_triplets_response)

            # Combine results
            enhanced_data = {
                'original_pages':   pages_data,
                'pattern_analysis': pattern_analysis,
                'context_analysis': context_analysis,
                'people_locations': people_locations,
                'kg_triplets':      kg_triplets.get('triplets', []),
                'processing_metadata': {
                    'total_pages':          len(pages_data),
                    'model_used':           self.model_name,
                    'context_window_pages': self.context_window_pages,
                    'processing_timestamp': str(Path().cwd()),
                    'triplet_count':        len(kg_triplets.get('triplets', [])),
                }
            }
            
            logger.info("Secondary processing completed successfully")
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Failed to process multiple pages: {e}")
            raise
    
    def push_triplets_to_neo4j(
        self,
        triplets: List[Dict[str, Any]],
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        database: str = "neo4j",
    ) -> int:
        """
        Push extracted KG triplets into Neo4j.

        Merges nodes and relationships so it is safe to call multiple times
        on the same data (idempotent).  Only triplets with confidence
        'high' or 'medium' are written.

        Args:
            triplets:       List of triplet dicts from kg_triplets output.
            neo4j_uri:      Bolt URI, e.g. "bolt://localhost:7687".
            neo4j_user:     Neo4j username.
            neo4j_password: Neo4j password.
            database:       Target database name.

        Returns:
            Number of triplets written.
        """
        from neo4j import GraphDatabase as _GDB  # local import to keep neo4j optional

        # Relation → Cypher pattern  (subject_label)-[rel]->(object_label)
        REL_MAP = {
            'LIVED_IN':        ('Person',      'Place'),
            'TRAVELED_TO':     ('Person',      'Place'),
            'WROTE':           ('Person',      None),      # object type varies
            'MENTIONED_IN':    (None,          'Fragment'),
            'ORIGINATED_FROM': (None,          'Place'),
            'SENT_TO':         ('Fragment',    None),
            'CITES':           ('BookArticle', 'BookArticle'),
        }

        driver = _GDB.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        written = 0
        try:
            with driver.session(database=database) as session:
                for t in triplets:
                    if t.get('confidence', 'low') == 'low':
                        continue

                    rel      = t.get('relation', '').upper()
                    subj     = t.get('subject', '').strip()
                    obj      = t.get('object', '').strip()
                    s_label  = t.get('subject_type', 'Person')
                    o_label  = t.get('object_type',  'Place')

                    if not (rel and subj and obj):
                        continue

                    # Use LLM-provided labels, falling back to REL_MAP hints
                    if rel in REL_MAP:
                        hint_s, hint_o = REL_MAP[rel]
                        if hint_s:
                            s_label = hint_s
                        if hint_o:
                            o_label = hint_o

                    # name property differs by label
                    s_prop = 'canonical_shelfmark' if s_label == 'Fragment' else 'name'
                    o_prop = 'canonical_shelfmark' if o_label == 'Fragment' else 'name'

                    cypher = (
                        f"MERGE (s:{s_label} {{{s_prop}: $subj}}) "
                        f"MERGE (o:{o_label} {{{o_prop}: $obj}}) "
                        f"MERGE (s)-[r:{rel}]->(o) "
                        f"SET r.evidence = $evidence, r.confidence = $confidence"
                    )
                    try:
                        session.run(cypher, subj=subj, obj=obj,
                                    evidence=t.get('evidence', ''),
                                    confidence=t.get('confidence', 'medium'))
                        written += 1
                    except Exception as e:
                        logger.warning(f"Triplet write failed ({subj} -{rel}-> {obj}): {e}")
        finally:
            driver.close()

        logger.info(f"✓ Pushed {written}/{len(triplets)} triplets to Neo4j ({database})")
        return written

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
    parser.add_argument("--output",       help="Output path for enhanced data JSON")
    # Backend toggle
    parser.add_argument("--backend",      default=BACKEND_OLLAMA,
                        choices=[BACKEND_OLLAMA, BACKEND_GEMINI],
                        help="LLM backend to use (default: ollama)")
    # Ollama / LM Studio options
    parser.add_argument("--model",        default="c4ai-command-r-v01", help="Ollama model name")
    parser.add_argument("--ollama-url",   default="http://localhost:1234/v1", help="Ollama/LM Studio API URL")
    # Gemini options
    parser.add_argument("--gemini-model", default=GEMINI_FLASH_MODEL,
                        help=f"Gemini model name (default: {GEMINI_FLASH_MODEL})")
    # Neo4j options — if provided, triplets are pushed automatically
    parser.add_argument("--neo4j-uri",      default=None, help="Neo4j bolt URI (enables KG push)")
    parser.add_argument("--neo4j-user",     default="neo4j")
    parser.add_argument("--neo4j-password", default=None)
    parser.add_argument("--neo4j-database", default="neo4j")

    args = parser.parse_args()

    try:
        processor = SecondaryLLMProcessor(
            backend=args.backend,
            model_name=args.model,
            ollama_url=args.ollama_url,
            gemini_model=args.gemini_model,
        )

        enhanced_data = processor.process_from_structured_dir(
            args.structured_dir,
            args.output,
        )

        meta = enhanced_data.get('processing_metadata', {})
        print(f"\nSecondary processing completed:")
        print(f"  Pages processed:               {meta.get('total_pages', 0)}")
        print(f"  Enhanced SM transcriptions:    {len(enhanced_data.get('context_analysis', {}).get('enhanced_shelf_mark_transcriptions', {}))}")
        print(f"  People extracted:              {len(enhanced_data.get('people_locations', {}).get('people', []))}")
        print(f"  Locations extracted:           {len(enhanced_data.get('people_locations', {}).get('locations', []))}")
        print(f"  KG triplets extracted:         {len(enhanced_data.get('kg_triplets', []))}")

        # Optionally push triplets to Neo4j
        if args.neo4j_uri and args.neo4j_password:
            written = processor.push_triplets_to_neo4j(
                triplets=enhanced_data.get('kg_triplets', []),
                neo4j_uri=args.neo4j_uri,
                neo4j_user=args.neo4j_user,
                neo4j_password=args.neo4j_password,
                database=args.neo4j_database,
            )
            print(f"  Triplets written to Neo4j:     {written}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
