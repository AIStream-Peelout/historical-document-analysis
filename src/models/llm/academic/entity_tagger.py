#!/usr/bin/env python3
"""
Pass 2 — Per-Page Entity Tagger
================================

Processes each page's structured JSON (Pass 1 output) and extracts named
entities with page-level attribution.  Uses a 2-page sliding window so that
an entity named at the bottom of the previous page is still available as
context when its pronoun appears at the top of the current page.

This pass is intentionally narrow: it only asks *what entities are on this
page* — not what they did or how they relate.  Relationship extraction is
handled in Pass 4 after coreference resolution (Pass 3) has consolidated
entity mentions across the full book.

Input
-----
``page_NNN_structured.json`` files produced by ``structured_json_llm.py``
(Pass 1), found inside a book subdirectory under ``academic_literature/``.

Output
------
``page_NNN_entities.json`` alongside each structured JSON, with the schema::

    {
        "page_number": 28,
        "source_book": "india_trader_1_50",
        "extracted_at": "2026-06-08T12:00:00Z",
        "model_used": "gemini-3.5-flash",
        "people": [
            {
                "name": "Joseph b. David Lebdi",
                "role": "historical_person",
                "context": "Central figure, merchant from Tripoli"
            }
        ],
        "places": [
            {"name": "Tripoli", "type": "city", "context": "Lebdi family origin"}
        ],
        "institutions": [
            {"name": "Rabbinical Court of Fustat", "type": "court", "context": "..."}
        ],
        "shelf_marks": [
            {"mark": "T-S 20.115", "description": "Letter from Lebdi", "source": "pass1"}
        ]
    }

``role`` values for people:  ``historical_person`` | ``scholar`` | ``collector``
``type`` values for places:  ``city`` | ``region`` | ``country`` | ``body_of_water``
                              | ``trade_route`` | ``other``
``type`` values for institutions: ``library`` | ``university`` | ``archive``
                                   | ``synagogue`` | ``court`` | ``other``

Usage
-----
# Tag all pages in a book directory
python entity_tagger.py --dir academic_literature/india_traders/india_trader_1_50

# Tag all books under academic_literature/
python entity_tagger.py

# Dry-run: show what would be processed without calling the LLM
python entity_tagger.py --dry-run

# Force re-tag even if output file already exists
python entity_tagger.py --overwrite

# Use LM Studio instead of Gemini
python entity_tagger.py --backend lm_studio --lms-model qwen3:8b
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dotenv

_project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(_project_root))
dotenv.load_dotenv(_project_root / ".env")

from src.models.llm.academic.llm_client import LLMClient  # noqa: E402

logger = logging.getLogger(__name__)

_data_root = _project_root / "src" / "datasets" / "raw_data" / "cairo_genizah" / "academic_literature"

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

_ENTITY_FILE_SUFFIX = "_entities.json"


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def _is_structured_dir(p: Path) -> bool:
    """Return True if *p* is a structured-output directory.

    Matches both single-level (``book_structured_gemini/``) and two-level
    (``book_structured_qwen/qwen3_vl_8b/``) layouts.

    :param p: Directory path to test.
    :returns: True if the directory or its parent contains ``_structured``.
    """
    return "_structured" in p.name or "_structured" in p.parent.name


def _book_dir_for_structured(p: Path) -> Path:
    """Return the book root directory for a structured JSON file path.

    :param p: Path to a ``page_NNN_structured.json`` file.
    :returns: Book root directory (2 or 3 levels up depending on layout).
    """
    # p.parent = the structured dir (single-level) or model subdir (two-level)
    if "_structured" in p.parent.name:
        return p.parent.parent          # single-level: book/book_structured_*/
    return p.parent.parent.parent       # two-level:    book/book_structured_*/model/

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert in medieval Jewish history, the Cairo Genizah, and academic
scholarship on the medieval Mediterranean world.

Your task is to identify named entities on a specific page of an academic book.
Extract only what is directly stated on the CURRENT PAGE — use the PREVIOUS PAGE
only as context to help identify entities that continue across a page boundary.

Output ONLY valid JSON. No markdown, no code fences, no explanation.
"""


def _build_prompt(
    current_page: Dict,
    prev_page: Optional[Dict],
    source_book: str,
) -> str:
    """Build the entity tagging prompt for one page.

    :param current_page: Structured JSON for the page being tagged.
    :param prev_page: Structured JSON for the previous page (context only).
    :param source_book: Book stem name for context.
    :returns: Prompt string ready to send to the LLM.
    """
    page_num = current_page.get("extracted_page_number", "unknown")
    main_text = (current_page.get("full_main_text") or "").strip()
    footnotes = current_page.get("footnotes") or {}
    _sm = current_page.get("shelf_marks_mentioned") or {}
    pass1_marks = list(_sm.keys()) if isinstance(_sm, dict) else list(_sm)
    classification = current_page.get("classification", "general_academic")

    footnote_block = ""
    if footnotes:
        lines = [f"  [{k}] {v}" for k, v in list(footnotes.items())[:10]]
        footnote_block = "Footnotes:\n" + "\n".join(lines)

    prev_block = ""
    if prev_page:
        prev_text = (prev_page.get("full_main_text") or "").strip()[-600:]
        prev_num = prev_page.get("extracted_page_number", "?")
        prev_block = f"""
PREVIOUS PAGE (page {prev_num}) — for context only, do not extract entities from here:
{prev_text}

---
"""

    marks_hint = ""
    if pass1_marks:
        marks_hint = f"Shelf marks already identified by Pass 1 (confirm or add to these): {pass1_marks}\n"

    return f"""\
Book: {source_book}
Page: {page_num}
Classification: {classification}
{marks_hint}
{prev_block}CURRENT PAGE text:
{main_text[:2000]}

{footnote_block}

---
Extract all named entities that appear on the CURRENT PAGE above.

Return ONLY this JSON structure (no markdown, no code fences):
{{
  "people": [
    {{
      "name": "Full name as it appears",
      "role": "historical_person|scholar|collector",
      "context": "One sentence describing who this person is in this context"
    }}
  ],
  "places": [
    {{
      "name": "Place name",
      "type": "city|region|country|body_of_water|trade_route|other",
      "context": "Brief context"
    }}
  ],
  "institutions": [
    {{
      "name": "Institution name",
      "type": "library|university|archive|synagogue|court|other",
      "context": "Brief context"
    }}
  ],
  "shelf_marks": [
    {{
      "mark": "Shelf mark exactly as written",
      "description": "What this fragment contains or discusses",
      "source": "pass1|pass2"
    }}
  ]
}}

Rules:
- "historical_person": a person who lived before ~1900 and is discussed as a historical figure
- "scholar": a modern academic (post ~1850) who writes about or studies historical material
- "collector": a person who collected or acquired Genizah fragments (e.g. Elkan Adler, David Kaufmann)
- Include shelf marks already listed in Pass 1 with source="pass1"; add newly spotted ones with source="pass2"
- Use the most complete form of a name you can find on the current page
- If no entities of a type exist, return an empty list []
"""


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)


def _parse_response(raw: str) -> Optional[Dict]:
    """Extract and validate the JSON entity dict from an LLM response.

    :param raw: Raw LLM output string.
    :returns: Parsed dict with people/places/institutions/shelf_marks keys,
              or None if parsing fails.
    """
    # Strip thinking tags (some models emit <think>...</think>)
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    match = _JSON_RE.search(raw)
    if not match:
        logger.warning("  No JSON found in LLM response")
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as e:
        logger.warning(f"  JSON parse error: {e}")
        return None

    # Normalise — ensure all expected lists exist
    return {
        "people":       [p for p in (data.get("people")       or []) if p.get("name")],
        "places":       [p for p in (data.get("places")       or []) if p.get("name")],
        "institutions": [p for p in (data.get("institutions") or []) if p.get("name")],
        "shelf_marks":  [p for p in (data.get("shelf_marks")  or []) if p.get("mark")],
    }


# ---------------------------------------------------------------------------
# Core tagger
# ---------------------------------------------------------------------------

class EntityTagger:
    """Tags named entities in structured page JSON files (Pass 1 output).

    :param client: Initialised LLMClient to use for generation.
    """

    def __init__(self, client: LLMClient):
        self.client = client

    def tag_book(
        self,
        book_dir: Path,
        dry_run: bool = False,
        overwrite: bool = False,
    ) -> Dict[str, int]:
        """Tag all pages in a book directory.

        Structured JSONs live inside ``*_structured_*/`` subdirectories
        (e.g. ``india_trader_1_50_structured_gemini_gemini_2.5_flash/``).
        Entity files are written alongside them in the same subdir.

        :param book_dir: Book root directory (parent of the ``*_structured_*/`` subdir).
        :param dry_run: If True, count pages without calling the LLM.
        :param overwrite: If True, re-tag pages that already have entity files.
        :returns: Counts dict with ``tagged``, ``skipped``, ``failed`` keys.
        """
        source_book  = book_dir.name
        entities_dir = book_dir / "entities"
        _complete_marker = entities_dir / ".complete"

        # Fast-path: book was fully tagged in a previous run
        if _complete_marker.exists() and not overwrite:
            logger.info(f"  {source_book}: already complete (delete entities/.complete to re-run)")
            return {"tagged": 0, "skipped": 0, "failed": 0, "book_complete": True}

        structured_files = sorted(
            p for p in book_dir.rglob("page_*_structured.json")
            if _is_structured_dir(p.parent)
        )

        if not structured_files:
            logger.warning(f"  No structured JSON files found in {book_dir}")
            return {"tagged": 0, "skipped": 0, "failed": 0}

        counts = {"tagged": 0, "skipped": 0, "failed": 0}
        entities_dir.mkdir(exist_ok=True)

        already_done = sum(
            1 for p in structured_files
            if (entities_dir / p.name.replace("_structured.json", _ENTITY_FILE_SUFFIX)).exists()
        )
        remaining = len(structured_files) - already_done
        logger.info(
            f"  {source_book}: {len(structured_files)} pages total, "
            f"{already_done} done, {remaining} remaining"
        )

        prev_data: Optional[Dict] = None

        for i, struct_path in enumerate(structured_files):
            # e.g. page_026_structured.json → entities/page_026_entities.json
            entity_filename = struct_path.name.replace("_structured.json", _ENTITY_FILE_SUFFIX)
            entity_path = entities_dir / entity_filename

            if entity_path.exists() and not overwrite:
                counts["skipped"] += 1
                # Still load for use as prev_page context
                try:
                    prev_data = json.load(open(struct_path, encoding="utf-8"))
                except Exception:
                    prev_data = None
                continue

            try:
                current_data = json.load(open(struct_path, encoding="utf-8"))
            except Exception as e:
                logger.error(f"  Could not load {struct_path.name}: {e}")
                counts["failed"] += 1
                prev_data = None
                continue

            page_num = current_data.get("extracted_page_number")

            if dry_run:
                logger.debug(f"  [dry-run] Would tag {struct_path.name} (page {page_num})")
                counts["tagged"] += 1
                prev_data = current_data
                continue

            prompt = _build_prompt(current_data, prev_data, source_book)

            try:
                raw = self.client.complete(_SYSTEM_PROMPT, prompt)
                parsed = _parse_response(raw)
            except Exception as e:
                logger.error(f"  LLM error on {struct_path.name}: {e}")
                counts["failed"] += 1
                prev_data = current_data
                continue

            if parsed is None:
                counts["failed"] += 1
                prev_data = current_data
                continue

            # Merge pass1 shelf marks that the LLM may have missed
            _sm2 = current_data.get("shelf_marks_mentioned") or {}
            pass1_marks = set(_sm2.keys()) if isinstance(_sm2, dict) else set(_sm2)
            tagged_marks = {sm["mark"] for sm in parsed["shelf_marks"]}
            for mark, desc in (_sm2.items() if isinstance(_sm2, dict) else {}):

                if mark not in tagged_marks:
                    parsed["shelf_marks"].append({
                        "mark": mark,
                        "description": desc,
                        "source": "pass1",
                    })

            output = {
                "page_number":   page_num,
                "source_book":   source_book,
                "extracted_at":  datetime.now(timezone.utc).isoformat(),
                "model_used":    self.client.lms_model
                                 if self.client.backend == "lm_studio"
                                 else "gemini",
                **parsed,
            }

            with open(entity_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            people_count = len(parsed["people"])
            place_count  = len(parsed["places"])
            mark_count   = len(parsed["shelf_marks"])
            logger.info(
                f"  {struct_path.name} (p.{page_num}): "
                f"{people_count} people, {place_count} places, {mark_count} marks"
            )

            counts["tagged"] += 1
            prev_data = current_data

        # Write completion marker if all pages succeeded (none failed)
        if not dry_run and counts["failed"] == 0 and remaining > 0:
            _complete_marker.touch()
            logger.info(f"  {source_book}: marked complete ✓")

        return counts

    def tag_all(
        self,
        root: Path,
        dry_run: bool = False,
        overwrite: bool = False,
    ) -> Dict[str, int]:
        """Tag all books found recursively under *root*.

        :param root: Root directory to search for book subdirectories.
        :param dry_run: Count pages without calling the LLM.
        :param overwrite: Re-tag pages that already have entity files.
        :returns: Aggregated counts across all books.
        """
        # A "book directory" is the parent of any *_structured_*/ subdir
        # that contains page_NNN_structured.json files.
        book_dirs: List[Path] = []
        seen: set = set()
        for p in root.rglob("page_*_structured.json"):
            if _is_structured_dir(p.parent):
                book_dir = _book_dir_for_structured(p)
                if book_dir not in seen:
                    book_dirs.append(book_dir)
                    seen.add(book_dir)
        book_dirs.sort()

        totals = {"tagged": 0, "skipped": 0, "failed": 0, "books_complete": 0, "books_processed": 0}
        logger.info(f"Found {len(book_dirs)} book directories under {root}")

        for book_dir in book_dirs:
            logger.info(f"Book: {book_dir.relative_to(root)}")
            counts = self.tag_book(book_dir, dry_run=dry_run, overwrite=overwrite)
            for k in ("tagged", "skipped", "failed"):
                totals[k] += counts.get(k, 0)
            if counts.get("book_complete"):
                totals["books_complete"] += 1
            else:
                totals["books_processed"] += 1

        return totals


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("entity_tagger.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dir", "-d", metavar="SUBDIR", default=None,
        help="Subdirectory under academic_literature/ to process (default: all).",
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Count pages without calling the LLM.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-tag pages that already have entity files.",
    )
    parser.add_argument(
        "--backend", choices=["gemini", "lm_studio"], default="gemini",
        help="LLM backend to use (default: gemini).",
    )
    parser.add_argument(
        "--lms-model", default="qwen3:8b",
        help="LM Studio model name (only used with --backend lm_studio).",
    )
    parser.add_argument(
        "--lms-url", default="http://localhost:1234/v1",
        help="LM Studio API URL.",
    )
    parser.add_argument(
        "--gemini-model", default="gemini-3.5-flash",
        help="Gemini model name (default: gemini-3.5-flash.",
    )
    args = parser.parse_args()

    client = LLMClient(
        backend=args.backend,
        lms_url=args.lms_url,
        lms_model=args.lms_model,
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=args.gemini_model,
    )

    tagger = EntityTagger(client)

    root = _data_root
    if args.dir:
        root = _data_root / args.dir
        if not root.exists():
            logger.error(f"Directory not found: {root}")
            sys.exit(1)

    totals = tagger.tag_all(root, dry_run=args.dry_run, overwrite=args.overwrite)

    tag = "[DRY RUN] " if args.dry_run else ""
    print(f"\n{tag}Entity tagging complete")
    print("=" * 40)
    print(f"  Books already complete: {totals['books_complete']:,} (fast-skipped)")
    print(f"  Books processed:        {totals['books_processed']:,}")
    print(f"  Pages tagged:           {totals['tagged']:,}")
    print(f"  Pages skipped:          {totals['skipped']:,} (file exists)")
    print(f"  Pages failed:           {totals['failed']:,}")
    print("=" * 40)


if __name__ == "__main__":
    main()
