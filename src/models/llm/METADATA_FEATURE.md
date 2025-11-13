# Book Metadata Feature for StructuredJSONLLM

## Overview
The `StructuredJSONLLM` class now supports providing book metadata as context to improve the quality of structured data extraction from PDFs.

## Features
1. **Pass metadata as constructor parameter** - Load metadata once and reuse for all pages
2. **Automatic file loading** - Falls back to loading `{pdf_name}_metadata.json` if not provided
3. **Context-aware prompts** - LLM receives book information to better understand the content

## Usage

### Method 1: Pass metadata when initializing

```python
import json
from src.models.llm.structured_json_llm import StructuredJSONLLM

# Load metadata from a JSON file
with open('book_metadata.json', 'r') as f:
    book_metadata = json.load(f)

# Initialize with metadata
llm_service = StructuredJSONLLM(
    use_gemini=True,
    book_metadata=book_metadata
)

# Process OCR results - metadata will be included in prompts
results = llm_service.process_from_ocr_service(ocr_result)
```

### Method 2: Automatic file loading

Place a file named `{pdf_name}_metadata.json` in one of these locations:
- Same directory as `structured_json_llm.py` (`src/models/llm/`)
- `raw_data_dir` 
- Current working directory

The system will automatically load and use it:

```python
# No metadata parameter needed - will auto-load
llm_service = StructuredJSONLLM(use_gemini=True)
results = llm_service.process_from_ocr_service(ocr_result)
```

### Method 3: Using in complete_pipeline_example.py

The `complete_pipeline_example.py` script has been updated to automatically load and use metadata:

```python
# The script automatically loads metadata from files like:
# - cairo_to_manchester_1_metadata.json
# - cairo_to_manchester_intro_metadata.json
# etc.

resume_specific_file("cairo_to_manchester/cairo_to_manchester_1.pdf")
```

## Metadata Format

The metadata should be a JSON file with book information. Example:

```json
{
  "title": "From Cairo to Manchester: studies in the Rylands Genizah fragments",
  "authors": [
    "Renate Smithuis",
    "Philip S. Alexander"
  ],
  "summary": "University of Manchester Library holds a collection...",
  "publication": {
    "type": "Print Book",
    "language": "English",
    "year": 2013,
    "publisher": "Oxford University Press",
    "location": "Oxford"
  },
  "subjects": [
    "Cairo Genizah",
    "Genizah",
    "Manuscripts, Hebrew"
  ]
}
```

Any fields can be included - they will all be formatted and added to the prompt.

## How it Works

When metadata is provided (either as a parameter or loaded from a file), it's formatted as a "Book Context" section and added to the LLM prompt:

```
## Book Context
You are processing pages from a specific book. Here is the book's metadata:
- Title: From Cairo to Manchester: studies in the Rylands Genizah fragments
- Authors: Renate Smithuis, Philip S. Alexander
- Summary: University of Manchester Library holds a collection...
...
Use this context to better understand the pages you are processing.
```

This context helps the LLM:
- Understand the book's subject matter and scope
- Recognize specialized terminology
- Extract more accurate shelf marks and transcriptions
- Provide better classifications

## Files Modified

1. **src/models/llm/structured_json_llm.py**
   - Added `book_metadata` parameter to `__init__`
   - Updated all processing methods to use metadata
   - Added `_load_book_metadata()` helper method
   - Updated `_create_prompt()` to include metadata context

2. **src/datasets/cairo_genizah/indexing/biblio/complete_pipeline_example.py**
   - Added `load_book_metadata()` function
   - Updated `run_specific_file()` and `resume_specific_file()` to load and use metadata
   - Added console logging when metadata is loaded

3. **src/models/llm/example_book_metadata.json**
   - Created example metadata file showing the format


