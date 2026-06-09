#!/usr/bin/env python3
"""
Pass 3 — Coreference Resolution
=================================

Reads all per-page entity files (Pass 2 output) for a book and resolves
ambiguous references — pronouns, definite descriptions, name variants,
shelf mark shorthand — back to the named entities established in Pass 2.

Uses a 5-page sliding window so cross-page references like "he", "the
manuscript", or "the above-mentioned merchant" can be traced back to the
last full name that appeared in context.  If the model times out or hits
a context limit the window is halved and retried (down to a minimum of 2 pages).

Output is a single consolidated ``book_entities_resolved.json`` file per
book with a deduplicated entity list, each entry carrying the full list of
page numbers where that entity appears.

Input
-----
``page_NNN_entities.json`` files produced by ``entity_tagger.py`` (Pass 2).

Output
------
``book_entities_resolved.json`` in the book directory::

    {
        "source_book": "india_trader_1_50",
        "resolved_at": "2026-06-08T12:00:00Z",
        "model_used":  "gemini-3.5-flash",
        "people": [
            {
                "name":         "Joseph b. David Lebdi",
                "role":         "historical_person",
                "pages":        [1, 9, 13, 15, 19, 22, 26, 27, 28, 29, 30, 31],
                "aliases":      ["Lebdi", "Joseph Lebdi", "the merchant", "Ibn al-Lebdi"],
                "description":  "Merchant from Tripoli, central figure of section 1"
            }
        ],
        "places": [...],
        "institutions": [...],
        "shelf_marks": [...]
    }

Usage
-----
python coreference_resolver.py --dir academic_literature/india_traders/india_trader_1_50
python coreference_resolver.py                    # all books
python coreference_resolver.py --dry-run          # show page counts only
python coreference_resolver.py --overwrite        # re-resolve even if output exists
python coreference_resolver.py --window 5         # smaller window if hitting context limits
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dotenv

_project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(_project_root))
dotenv.load_dotenv(_project_root / ".env")

from src.models.llm.entity_relation_extractor import LLMClient  # noqa: E402

logger = logging.getLogger(__name__)

_data_root  = _project_root / "src" / "datasets" / "raw_data" / "cairo_genizah" / "academic_literature"
_OUTPUT_FILE = "book_entities_resolved.json"
_DEFAULT_WINDOW = 5
_MIN_WINDOW     = 2

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert in medieval Jewish history, the Cairo Genizah, and academic
scholarship on the medieval Mediterranean world.

Your task is to consolidate and resolve a list of named entities extracted
page-by-page from an academic book.  Many entities will appear under different
forms across pages (short names, pronouns, definite descriptions).  Your job
is to merge these into canonical entries with full page attribution.

Output ONLY valid JSON. No markdown, no code fences, no explanation.
"""


def _build_window_prompt(
    pages_block: List[Dict],
    source_book: str,
    existing_entities: Optional[Dict] = None,
) -> str:
    """Build the coreference resolution prompt for a window of pages.

    :param pages_block: List of entity dicts (one per page, from Pass 2).
    :param source_book: Book stem name for context.
    :param existing_entities: Entities already resolved from prior windows
                               (passed as context so the model can extend them).
    :returns: Prompt string ready to send to the LLM.
    """
    page_summaries = []
    for page in pages_block:
        pg = page.get("page_number", "?")
        people = [p["name"] for p in page.get("people", [])]
        places = [p["name"] for p in page.get("places", [])]
        insts  = [p["name"] for p in page.get("institutions", [])]
        marks  = [p["mark"] for p in page.get("shelf_marks", [])]
        parts = [f"Page {pg}:"]
        if people:
            parts.append(f"  People: {', '.join(people)}")
        if places:
            parts.append(f"  Places: {', '.join(places)}")
        if insts:
            parts.append(f"  Institutions: {', '.join(insts)}")
        if marks:
            parts.append(f"  Shelf marks: {', '.join(marks)}")
        page_summaries.append("\n".join(parts))

    existing_block = ""
    if existing_entities:
        existing_block = "\nEntities resolved from earlier pages (extend these, do not duplicate):\n"
        for label in ("people", "places", "institutions", "shelf_marks"):
            items = existing_entities.get(label, [])
            if items:
                key = "name" if label != "shelf_marks" else "mark"
                names = [e[key] for e in items]
                existing_block += f"  {label.title()}: {', '.join(names)}\n"

    return f"""\
Book: {source_book}
{existing_block}
Per-page entity extractions for pages in this window:

{chr(10).join(page_summaries)}

---
Consolidate these into a single resolved entity list.  For each entity:
- Use the most complete canonical name (e.g. "Joseph b. David Lebdi" not "Lebdi")
- List ALL page numbers where this entity appears (including from existing resolved entities above)
- List known aliases / short forms / pronouns used in this window for this entity
- Write a brief description based on context

Return ONLY this JSON (no markdown, no code fences):
{{
  "people": [
    {{
      "name":        "Canonical full name",
      "role":        "historical_person|scholar|collector",
      "pages":       [1, 9, 13, 26, 27],
      "aliases":     ["Short name", "Pronoun or description used in text"],
      "description": "One sentence identifying this person in context"
    }}
  ],
  "places": [
    {{
      "name":    "Canonical place name",
      "type":    "city|region|country|body_of_water|trade_route|other",
      "pages":   [13, 27],
      "aliases": ["Alternative forms used in text"],
      "description": "Brief context"
    }}
  ],
  "institutions": [
    {{
      "name":    "Canonical institution name",
      "type":    "library|university|archive|synagogue|court|other",
      "pages":   [5],
      "aliases": [],
      "description": "Brief context"
    }}
  ],
  "shelf_marks": [
    {{
      "mark":        "Canonical shelf mark (e.g. T-S 20.115)",
      "pages":       [27, 28],
      "aliases":     ["Abbreviated forms used in text"],
      "description": "What this fragment contains or discusses"
    }}
  ]
}}

Rules:
- Merge entries that clearly refer to the same entity (variant names, abbreviations,
  pronouns, definite descriptions like "the merchant" or "the manuscript")
- Keep entries separate if there is any doubt they are the same entity
- Include ALL entities from the existing resolved list above, extended with new pages
  and aliases found in this window
- Pages list must be sorted ascending
"""


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)


def _parse_response(raw: str) -> Optional[Dict]:
    """Parse LLM JSON response into a resolved entities dict.

    :param raw: Raw LLM output string.
    :returns: Dict with people/places/institutions/shelf_marks keys, or None.
    """
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

    def _clean_list(items: List, key: str) -> List:
        seen, out = set(), []
        for item in (items or []):
            val = (item.get(key) or "").strip()
            if val and val not in seen:
                seen.add(val)
                # Ensure pages is a sorted list of ints
                pages = sorted(set(int(p) for p in (item.get("pages") or [])
                                   if str(p).lstrip("-").isdigit()))
                item["pages"] = pages
                item["aliases"] = [a for a in (item.get("aliases") or []) if a]
                out.append(item)
        return out

    return {
        "people":       _clean_list(data.get("people",       []), "name"),
        "places":       _clean_list(data.get("places",       []), "name"),
        "institutions": _clean_list(data.get("institutions", []), "name"),
        "shelf_marks":  _clean_list(data.get("shelf_marks",  []), "mark"),
    }


# ---------------------------------------------------------------------------
# Merge helper — combines two resolved-entity dicts
# ---------------------------------------------------------------------------

def _merge_resolved(base: Dict, update: Dict) -> Dict:
    """Merge *update* into *base*, deduplicating by canonical name.

    Entities that appear in both dicts have their page lists and alias lists
    merged; the description from *update* takes precedence if non-empty.

    :param base: Existing resolved entity dict.
    :param update: New resolved entity dict from the latest window.
    :returns: Merged dict.
    """
    def _merge_list(base_list: List, update_list: List, key: str) -> List:
        index: Dict[str, Dict] = {e[key]: e for e in base_list}
        for item in update_list:
            k = item[key]
            if k in index:
                existing = index[k]
                # Merge pages
                all_pages = sorted(set(existing.get("pages", []) + item.get("pages", [])))
                existing["pages"] = all_pages
                # Merge aliases
                all_aliases = list(dict.fromkeys(
                    existing.get("aliases", []) + item.get("aliases", [])
                ))
                existing["aliases"] = all_aliases
                # Update description if new one is non-empty
                if item.get("description"):
                    existing["description"] = item["description"]
            else:
                index[k] = item
        return list(index.values())

    return {
        "people":       _merge_list(base.get("people", []),       update.get("people", []),       "name"),
        "places":       _merge_list(base.get("places", []),       update.get("places", []),       "name"),
        "institutions": _merge_list(base.get("institutions", []), update.get("institutions", []), "name"),
        "shelf_marks":  _merge_list(base.get("shelf_marks", []),  update.get("shelf_marks", []),  "mark"),
    }


# ---------------------------------------------------------------------------
# Core resolver
# ---------------------------------------------------------------------------

class CoreferenceResolver:
    """Resolves cross-page entity references for a book (Pass 3).

    :param client: Initialised LLMClient to use for generation.
    :param window_size: Number of pages per sliding window (default 10).
    """

    def __init__(self, client: LLMClient, window_size: int = _DEFAULT_WINDOW):
        self.client      = client
        self.window_size = window_size

    def resolve_book(
        self,
        book_dir: Path,
        dry_run: bool = False,
        overwrite: bool = False,
    ) -> bool:
        """Resolve coreferences for all pages in a book directory.

        :param book_dir: Directory containing ``page_NNN_entities.json`` files.
        :param dry_run: If True, report page count without calling the LLM.
        :param overwrite: If True, re-resolve even if output already exists.
        :returns: True if resolution was written, False otherwise.
        """
        output_path = book_dir / _OUTPUT_FILE
        if output_path.exists() and not overwrite:
            logger.info(f"  {book_dir.name}: output exists, skipping (use --overwrite)")
            return False

        # Entity files live in book_dir/entities/ (Pass 2 output directory)
        entity_files = sorted((book_dir / "entities").glob("page_*_entities.json"))

        if not entity_files:
            logger.warning(f"  {book_dir.name}: no entity files found (run Pass 2 first)")
            return False

        source_book = book_dir.name
        logger.info(f"  {source_book}: resolving {len(entity_files)} pages "
                    f"(window={self.window_size})")

        if dry_run:
            return False

        # Load all pages
        pages: List[Dict] = []
        for ef in entity_files:
            try:
                pages.append(json.load(open(ef, encoding="utf-8")))
            except Exception as e:
                logger.warning(f"  Could not load {ef.name}: {e}")

        if not pages:
            return False

        # Slide the window through the pages, accumulating resolved entities
        resolved: Dict = {"people": [], "places": [], "institutions": [], "shelf_marks": []}
        window = self.window_size

        i = 0
        while i < len(pages):
            chunk = pages[i: i + window]
            prompt = _build_window_prompt(chunk, source_book, resolved if i > 0 else None)

            # Retry with smaller window on context, timeout, or disconnect errors
            current_window = window
            result = None
            attempts = 0
            max_attempts = 4
            while current_window >= _MIN_WINDOW and attempts < max_attempts:
                chunk = pages[i: i + current_window]
                prompt = _build_window_prompt(chunk, source_book, resolved if i > 0 else None)
                try:
                    raw    = self.client.complete(_SYSTEM_PROMPT, prompt)
                    result = _parse_response(raw)
                    break
                except Exception as e:
                    err = str(e).lower()
                    attempts += 1
                    # Shrink window on any retryable error: context overflow, timeout,
                    # or server disconnect (which signals the payload is too large)
                    retryable = (
                        "context" in err or "token" in err or "400" in err
                        or "disconnect" in err or "timeout" in err
                        or "timed out" in err or "server" in err
                    )
                    if retryable and current_window > _MIN_WINDOW:
                        current_window = max(_MIN_WINDOW, current_window // 2)
                        logger.warning(
                            f"  Retryable error (attempt {attempts}) — "
                            f"reducing window to {current_window}: {e}"
                        )
                    else:
                        logger.error(f"  LLM error on window {i}-{i+current_window}: {e}")
                        break

            if result:
                resolved = _merge_resolved(resolved, result)
                logger.info(
                    f"  Window pages {i+1}–{i+len(chunk)}: "
                    f"{len(resolved['people'])} people, "
                    f"{len(resolved['places'])} places, "
                    f"{len(resolved['shelf_marks'])} marks"
                )
            else:
                logger.warning(f"  Window {i}-{i+window}: no result, continuing")

            # Slide by window - 2 so the last two pages of each chunk overlap
            # with the first two of the next (helps catch cross-window references)
            i += max(1, current_window - 2)

        output = {
            "source_book": source_book,
            "resolved_at": datetime.now(timezone.utc).isoformat(),
            "model_used":  self.client.lms_model
                           if self.client.backend == "lm_studio"
                           else "gemini",
            "page_count":  len(pages),
            **resolved,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(
            f"  {source_book}: wrote {output_path.name} — "
            f"{len(resolved['people'])} people, "
            f"{len(resolved['places'])} places, "
            f"{len(resolved['institutions'])} institutions, "
            f"{len(resolved['shelf_marks'])} shelf marks"
        )
        return True

    def resolve_all(
        self,
        root: Path,
        dry_run: bool = False,
        overwrite: bool = False,
    ) -> Dict[str, int]:
        """Resolve all books found under *root*.

        :param root: Root directory to search for book directories.
        :param dry_run: Report counts without calling the LLM.
        :param overwrite: Re-resolve books that already have output.
        :returns: Counts dict with ``resolved``, ``skipped``, ``no_entities`` keys.
        """
        # Find book dirs that contain entity files
        # Book dirs are any directory containing an entities/ subdirectory with entity files
        book_dirs = sorted(set(
            p.parent.parent
            for p in root.rglob("entities/page_*_entities.json")
        ))

        counts = {"resolved": 0, "skipped": 0, "no_entities": 0}
        logger.info(f"Found {len(book_dirs)} book directories with entity files")

        for book_dir in book_dirs:
            logger.info(f"Book: {book_dir.relative_to(root)}")
            if not list(book_dir.rglob("*_entities.json")):
                counts["no_entities"] += 1
                continue
            written = self.resolve_book(book_dir, dry_run=dry_run, overwrite=overwrite)
            if written:
                counts["resolved"] += 1
            else:
                counts["skipped"] += 1

        return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("coreference_resolver.log"),
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
        help="Report page counts without calling the LLM.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-resolve books that already have output.",
    )
    parser.add_argument(
        "--window", type=int, default=_DEFAULT_WINDOW,
        help=f"Sliding window size in pages (default: {_DEFAULT_WINDOW}).",
    )
    parser.add_argument(
        "--backend", choices=["gemini", "lm_studio"], default="gemini",
    )
    parser.add_argument("--lms-model", default="qwen3:8b")
    parser.add_argument("--lms-url",   default="http://localhost:1234/v1")
    parser.add_argument("--gemini-model", default="gemini-3.5-flash")
    args = parser.parse_args()

    client = LLMClient(
        backend=args.backend,
        lms_url=args.lms_url,
        lms_model=args.lms_model,
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=args.gemini_model,
    )

    resolver = CoreferenceResolver(client, window_size=args.window)

    root = _data_root
    if args.dir:
        root = _data_root / args.dir
        if not root.exists():
            logger.error(f"Directory not found: {root}")
            sys.exit(1)

    counts = resolver.resolve_all(root, dry_run=args.dry_run, overwrite=args.overwrite)

    tag = "[DRY RUN] " if args.dry_run else ""
    print(f"\n{tag}Coreference resolution complete")
    print("=" * 40)
    print(f"  Resolved:    {counts['resolved']:,}")
    print(f"  Skipped:     {counts['skipped']:,}")
    print(f"  No entities: {counts['no_entities']:,}")
    print("=" * 40)


if __name__ == "__main__":
    main()
