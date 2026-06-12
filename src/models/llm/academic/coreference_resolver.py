#!/usr/bin/env python3
"""
Pass 3 — Coreference Resolution
=================================

Reads all per-page entity files (Pass 2 output) for a book and resolves
ambiguous references — name variants, abbreviations, and alternate spellings
— into canonical entities with full page attribution across the book.

Architecture
------------
Phase 1 (pure Python, free):
  - Load all ``page_NNN_entities.json`` files
  - Aggregate every name mention with its page numbers
  - Exact-deduplicate within each category

Phase 2 (one small LM Studio call per entity type, cheap):
  - Send only the flat list of unique names (no page text, no context)
  - Ask the model to group variants → canonical name
  - Apply the mapping back to the aggregated data

This avoids the sliding-window approach which hit Gemini timeouts and
consumed expensive tokens on growing context.  The LLM sees only a compact
name list (a few hundred tokens), so even a small local model (qwen3:8b)
handles it reliably.

Gemini is intentionally NOT used here — reserve quota for Pass 2 (tagging)
and Pass 4 (relation extraction) which need full page text.

Input
-----
``page_NNN_entities.json`` files produced by ``entity_tagger.py`` (Pass 2),
located in ``book_dir/entities/``.

Output
------
``book_entities_resolved.json`` in the book directory::

    {
        "source_book": "india_trader_1_50",
        "resolved_at": "2026-06-09T00:00:00Z",
        "model_used":  "qwen3:8b",
        "page_count":  48,
        "people": [
            {
                "name":        "Joseph b. David Lebdi",
                "role":        "historical_person",
                "pages":       [1, 9, 13, 15, 22, 27],
                "aliases":     ["Lebdi", "Joseph Lebdi", "Ibn al-Lebdi"],
                "description": "Merchant from Tripoli, central figure of section 1"
            }
        ],
        "places": [...],
        "institutions": [...],
        "shelf_marks": [...]
    }

Usage
-----
python coreference_resolver.py --dir india_traders/india_trader_1_50
python coreference_resolver.py                     # all books
python coreference_resolver.py --dry-run           # show counts only
python coreference_resolver.py --overwrite         # re-resolve even if output exists
python coreference_resolver.py --no-llm            # Phase 1 only, skip LLM dedup
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

import requests
import dotenv

_project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(_project_root))
dotenv.load_dotenv(_project_root / ".env")

logger = logging.getLogger(__name__)

_data_root   = _project_root / "src" / "datasets" / "raw_data" / "cairo_genizah" / "academic_literature"
_OUTPUT_FILE = "book_entities_resolved.json"

# ---------------------------------------------------------------------------
# Default LM Studio settings (cheap local model for name-list dedup)
# ---------------------------------------------------------------------------
_LMS_URL     = "http://localhost:1234/v1"
_LMS_MODEL   = "qwen/qwen3-4b-2507"
_LMS_TIMEOUT = 120  # seconds


# ---------------------------------------------------------------------------
# Phase 1 — Pure Python aggregation
# ---------------------------------------------------------------------------

def _aggregate_entities(entity_files: List[Path]) -> Dict:
    """Aggregate all per-page entity files into flat name-keyed dicts.

    For each category (people, places, institutions, shelf_marks) we collect:
    - the set of page numbers the name appears on
    - all ``context`` strings seen for that name
    - the ``role``/``type`` most commonly assigned to that name

    :param entity_files: Sorted list of ``page_NNN_entities.json`` paths.
    :returns: Dict with keys ``people``, ``places``, ``institutions``,
              ``shelf_marks``, each a dict keyed by raw name/mark string.
    """
    # Structure: category → name → {"pages": set, "contexts": list, "meta": Counter}
    people:       Dict[str, Dict] = defaultdict(lambda: {"pages": set(), "contexts": [], "roles": []})
    places:       Dict[str, Dict] = defaultdict(lambda: {"pages": set(), "contexts": [], "types": []})
    institutions: Dict[str, Dict] = defaultdict(lambda: {"pages": set(), "contexts": [], "types": []})
    shelf_marks:  Dict[str, Dict] = defaultdict(lambda: {"pages": set(), "descriptions": []})

    for ef in entity_files:
        try:
            data = json.load(open(ef, encoding="utf-8"))
        except Exception as e:
            logger.warning(f"  Could not load {ef.name}: {e}")
            continue

        pg = data.get("page_number")
        if pg is None:
            continue

        for p in (data.get("people") or []):
            name = (p.get("name") or "").strip()
            if not name:
                continue
            people[name]["pages"].add(pg)
            if p.get("context"):
                people[name]["contexts"].append(p["context"])
            if p.get("role"):
                people[name]["roles"].append(p["role"])

        for pl in (data.get("places") or []):
            name = (pl.get("name") or "").strip()
            if not name:
                continue
            places[name]["pages"].add(pg)
            if pl.get("context"):
                places[name]["contexts"].append(pl["context"])
            if pl.get("type"):
                places[name]["types"].append(pl["type"])

        for inst in (data.get("institutions") or []):
            name = (inst.get("name") or "").strip()
            if not name:
                continue
            institutions[name]["pages"].add(pg)
            if inst.get("context"):
                institutions[name]["contexts"].append(inst["context"])
            if inst.get("type"):
                institutions[name]["types"].append(inst["type"])

        for sm in (data.get("shelf_marks") or []):
            mark = (sm.get("mark") or "").strip()
            if not mark:
                continue
            shelf_marks[mark]["pages"].add(pg)
            if sm.get("description"):
                shelf_marks[mark]["descriptions"].append(sm["description"])

    return {
        "people":       dict(people),
        "places":       dict(places),
        "institutions": dict(institutions),
        "shelf_marks":  dict(shelf_marks),
    }


def _most_common(lst: List[str]) -> Optional[str]:
    """Return the most frequently occurring string in *lst*, or None if empty.

    :param lst: List of strings.
    :returns: Most common string or None.
    """
    if not lst:
        return None
    return max(set(lst), key=lst.count)


def _build_resolved_entry(name: str, agg: Dict, category: str) -> Dict:
    """Convert an aggregated raw entry into a resolved entity dict.

    :param name: Raw name string.
    :param agg: Aggregation dict for this name.
    :param category: One of ``people``, ``places``, ``institutions``, ``shelf_marks``.
    :returns: Resolved entity dict ready for output.
    """
    pages = sorted(int(p) for p in agg["pages"] if str(p).lstrip("-").isdigit())
    desc  = agg["contexts"][0] if agg.get("contexts") else (
            agg["descriptions"][0] if agg.get("descriptions") else "")

    if category == "people":
        return {
            "name":        name,
            "role":        _most_common(agg.get("roles", [])) or "historical_person",
            "pages":       pages,
            "aliases":     [],
            "description": desc,
        }
    if category == "places":
        return {
            "name":        name,
            "type":        _most_common(agg.get("types", [])) or "other",
            "pages":       pages,
            "aliases":     [],
            "description": desc,
        }
    if category == "institutions":
        return {
            "name":        name,
            "type":        _most_common(agg.get("types", [])) or "other",
            "pages":       pages,
            "aliases":     [],
            "description": desc,
        }
    # shelf_marks
    return {
        "mark":        name,
        "pages":       pages,
        "aliases":     [],
        "description": desc,
    }


# ---------------------------------------------------------------------------
# Phase 2 — LM Studio name-list deduplication
# ---------------------------------------------------------------------------

_DEDUP_SYSTEM = """\
/no_think
You are an expert in medieval Jewish history and the Cairo Genizah.
Your task is to group name variants that refer to the same entity.
Output ONLY valid JSON. No markdown, no code fences, no explanation.
"""


def _build_dedup_prompt(category: str, names: List[str]) -> str:
    """Build a compact deduplication prompt for one entity category.

    :param category: Human-readable category label for the prompt.
    :param names: List of unique raw name strings to deduplicate.
    :returns: Prompt string.
    """
    names_block = "\n".join(f"  - {n}" for n in names)
    return f"""\
The following {category} names were extracted page-by-page from an academic book about \
the Cairo Genizah.  Many are variants of the same entity (short forms, abbreviations, \
transliterations, alternative spellings).

Names to deduplicate:
{names_block}

Group these names so that each group shares ONE canonical name (the most complete form) \
and lists the variants.  Only merge names you are confident refer to the same entity.

Return ONLY this JSON (no markdown, no code fences):
{{
  "groups": [
    {{
      "canonical": "Most complete / formal name",
      "variants":  ["Shorter form", "Alternate spelling", ...]
    }}
  ]
}}

If a name has no variants just include it as a group with an empty variants list.
"""


def _call_lms(prompt: str, lms_url: str, lms_model: str, timeout: int) -> Optional[str]:
    """Send a prompt to an LM Studio / Ollama OpenAI-compatible endpoint.

    :param prompt: User prompt text.
    :param lms_url: Base URL of the LM Studio endpoint.
    :param lms_model: Model name to request.
    :param timeout: HTTP timeout in seconds.
    :returns: Response text, or None on failure.
    """
    payload = {
        "model": lms_model,
        "messages": [
            {"role": "system", "content": _DEDUP_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens":  8192,
        "response_format": {"type": "json_object"},
    }
    try:
        r = requests.post(
            f"{lms_url.rstrip('/')}/chat/completions",
            json=payload,
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"  LM Studio call failed: {e}")
        return None


_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)


def _parse_dedup_response(raw: str) -> List[Dict]:
    """Extract the ``groups`` list from an LLM dedup response.

    :param raw: Raw LLM output string.
    :returns: List of ``{"canonical": ..., "variants": [...]}`` dicts.
    """
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    match = _JSON_RE.search(raw)
    if not match:
        logger.warning("  No JSON in dedup response")
        return []
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as e:
        logger.warning(f"  JSON parse error in dedup response: {e}")
        return []
    return [
        g for g in (data.get("groups") or [])
        if g.get("canonical")
    ]


def _apply_dedup(
    resolved_list: List[Dict],
    groups: List[Dict],
    name_key: str,
) -> List[Dict]:
    """Apply LLM deduplication groups to a resolved entity list.

    Merges variant entries into their canonical entry (union of pages,
    union of aliases).

    :param resolved_list: List of resolved entity dicts from Phase 1.
    :param groups: Dedup groups from Phase 2 LLM call.
    :param name_key: Key used for the entity name (``"name"`` or ``"mark"``).
    :returns: Deduplicated list with aliases populated.
    """
    # Build variant → canonical mapping
    variant_to_canonical: Dict[str, str] = {}
    canonical_variants:   Dict[str, List[str]] = {}
    for g in groups:
        canon = g["canonical"].strip()
        variants = [v.strip() for v in (g.get("variants") or []) if v.strip()]
        canonical_variants[canon] = variants
        for v in variants:
            variant_to_canonical[v] = canon

    # Index resolved list by name
    index: Dict[str, Dict] = {e[name_key]: e for e in resolved_list}

    merged: Dict[str, Dict] = {}
    for entry in resolved_list:
        raw_name = entry[name_key]
        canon    = variant_to_canonical.get(raw_name, raw_name)

        if canon not in merged:
            # Start with this entry, using the canonical name
            new_entry = dict(entry)
            new_entry[name_key] = canon
            merged[canon] = new_entry
        else:
            # Merge pages and aliases into existing canonical entry
            existing = merged[canon]
            all_pages = sorted(set(existing["pages"] + entry["pages"]))
            existing["pages"] = all_pages
            # Add raw_name as alias if it differs from canonical
            if raw_name != canon and raw_name not in existing.get("aliases", []):
                existing.setdefault("aliases", []).append(raw_name)
            # Prefer longer description
            if len(entry.get("description", "")) > len(existing.get("description", "")):
                existing["description"] = entry["description"]

    # Populate aliases from the LLM groups for canonicals that had no variants merged
    for canon, variants in canonical_variants.items():
        if canon in merged:
            existing_aliases = merged[canon].get("aliases", [])
            for v in variants:
                if v not in existing_aliases and v != canon:
                    existing_aliases.append(v)
            merged[canon]["aliases"] = existing_aliases

    return list(merged.values())


# ---------------------------------------------------------------------------
# Core resolver
# ---------------------------------------------------------------------------

class CoreferenceResolver:
    """Resolves cross-page entity coreferences for a book (Pass 3).

    Phase 1 (free): aggregates all per-page entity files into a flat
    name-keyed dict with page attribution.

    Phase 2 (cheap): sends only the unique name list for each category to
    a local LM Studio model for deduplication/canonicalization.  Gemini is
    intentionally not used to conserve quota.

    :param lms_url: LM Studio base URL (default: http://localhost:1234/v1).
    :param lms_model: Local model name to use for dedup (default: qwen3:8b).
    :param lms_timeout: HTTP timeout for LM Studio calls in seconds.
    :param use_llm: If False, skip Phase 2 (aggregation only).
    """

    def __init__(
        self,
        lms_url:     str  = _LMS_URL,
        lms_model:   str  = _LMS_MODEL,
        lms_timeout: int  = _LMS_TIMEOUT,
        use_llm:     bool = True,
        run_tag:     str  = "",
    ):
        self.lms_url     = lms_url
        self.lms_model   = lms_model
        self.lms_timeout = lms_timeout
        self.use_llm     = use_llm
        self.run_tag     = run_tag  # e.g. "lms_qwen3-32b"; empty = default output

        if use_llm:
            self._verify_lms()

    def _verify_lms(self) -> None:
        """Verify LM Studio is reachable and the requested model is loaded.

        Logs a warning (does not raise) if unreachable so the caller can
        still run Phase 1 only.
        """
        try:
            r = requests.get(f"{self.lms_url.rstrip('/')}/models", timeout=5)
            models = [m["id"] for m in r.json().get("data", [])]
            logger.info(f"LM Studio connected. Available: {models}")
            if not any(self.lms_model in m for m in models):
                logger.warning(f"Model '{self.lms_model}' not in loaded models: {models}")
        except Exception as e:
            logger.warning(f"Could not reach LM Studio at {self.lms_url}: {e}")

    # Maximum names per LMS batch — keeps response well under 8192 tokens
    _BATCH_SIZE = 50

    def _dedup_category(
        self,
        category_label: str,
        resolved_list: List[Dict],
        name_key: str,
    ) -> List[Dict]:
        """Run Phase 2 deduplication for one entity category.

        The name list is chunked into batches of ``_BATCH_SIZE`` to keep
        each LMS response well within the token limit.  Groups from all
        batches are merged before applying the final dedup mapping.

        :param category_label: Human-readable label for the prompt.
        :param resolved_list: Phase 1 resolved entities for this category.
        :param name_key: Key used for the canonical name (``"name"`` or ``"mark"``).
        :returns: Deduplicated list.
        """
        if not self.use_llm or len(resolved_list) < 2:
            return resolved_list

        names = [e[name_key] for e in resolved_list]
        all_groups: List[Dict] = []

        for batch_start in range(0, len(names), self._BATCH_SIZE):
            batch = names[batch_start: batch_start + self._BATCH_SIZE]
            logger.info(
                f"    {category_label} dedup batch "
                f"{batch_start + 1}–{batch_start + len(batch)} of {len(names)}"
            )
            prompt = _build_dedup_prompt(category_label, batch)
            raw = _call_lms(prompt, self.lms_url, self.lms_model, self.lms_timeout)
            if not raw:
                logger.warning(f"  Dedup skipped for batch (LMS call failed)")
                continue
            groups = _parse_dedup_response(raw)
            all_groups.extend(groups)

        if not all_groups:
            return resolved_list

        result = _apply_dedup(resolved_list, all_groups, name_key)
        before = len(resolved_list)
        after  = len(result)
        if before != after:
            logger.info(f"    {category_label}: {before} → {after} after dedup")
        return result

    def resolve_book(
        self,
        book_dir: Path,
        dry_run:   bool = False,
        overwrite: bool = False,
    ) -> bool:
        """Resolve coreferences for a single book directory.

        :param book_dir: Book root directory containing ``entities/`` subdir.
        :param dry_run: If True, report counts without writing output.
        :param overwrite: If True, re-resolve even if output already exists.
        :returns: True if output was written, False otherwise.
        """
        # Tagged output file for LMS runs so Gemini and LMS results coexist
        output_filename = (_OUTPUT_FILE.replace(".json", f"_{self.run_tag}.json")
                           if self.run_tag else _OUTPUT_FILE)
        output_path = book_dir / output_filename
        if output_path.exists() and not overwrite:
            logger.info(f"  {book_dir.name}: output exists, skipping (use --overwrite)")
            return False

        # Read from tagged entities dir if set, fall back to default
        entities_dirname = f"entities_{self.run_tag}" if self.run_tag else "entities"
        entities_dir = book_dir / entities_dirname
        if not entities_dir.exists() and self.run_tag:
            entities_dir = book_dir / "entities"
            logger.info(f"  {book_dir.name}: no tagged entities dir, using default")
        entity_files = sorted(entities_dir.glob("page_*_entities.json"))
        if not entity_files:
            logger.warning(f"  {book_dir.name}: no entity files found (run Pass 2 first)")
            return False

        source_book = book_dir.name
        logger.info(f"  {source_book}: aggregating {len(entity_files)} pages")

        if dry_run:
            return False

        # ------------------------------------------------------------------
        # Phase 1 — aggregate
        # ------------------------------------------------------------------
        agg = _aggregate_entities(entity_files)

        people_list = [
            _build_resolved_entry(n, v, "people")
            for n, v in agg["people"].items()
        ]
        places_list = [
            _build_resolved_entry(n, v, "places")
            for n, v in agg["places"].items()
        ]
        insts_list = [
            _build_resolved_entry(n, v, "institutions")
            for n, v in agg["institutions"].items()
        ]
        marks_list = [
            _build_resolved_entry(n, v, "shelf_marks")
            for n, v in agg["shelf_marks"].items()
        ]

        logger.info(
            f"  Phase 1 — {len(people_list)} people, {len(places_list)} places, "
            f"{len(insts_list)} institutions, {len(marks_list)} marks (before dedup)"
        )

        # ------------------------------------------------------------------
        # Phase 2 — LM Studio dedup (name lists only, cheap)
        # ------------------------------------------------------------------
        people_list = self._dedup_category("people",       people_list, "name")
        places_list = self._dedup_category("places",       places_list, "name")
        insts_list  = self._dedup_category("institutions", insts_list,  "name")
        marks_list  = self._dedup_category("shelf marks",  marks_list,  "mark")

        output = {
            "source_book":  source_book,
            "resolved_at":  datetime.now(timezone.utc).isoformat(),
            "model_used":   self.lms_model if self.use_llm else "aggregation_only",
            "page_count":   len(entity_files),
            "people":       sorted(people_list, key=lambda e: e["name"]),
            "places":       sorted(places_list, key=lambda e: e["name"]),
            "institutions": sorted(insts_list,  key=lambda e: e["name"]),
            "shelf_marks":  sorted(marks_list,  key=lambda e: e["mark"]),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(
            f"  {source_book}: wrote {output_path.name} — "
            f"{len(people_list)} people, {len(places_list)} places, "
            f"{len(insts_list)} institutions, {len(marks_list)} shelf marks"
        )
        return True

    def resolve_all(
        self,
        root:      Path,
        dry_run:   bool = False,
        overwrite: bool = False,
    ) -> Dict[str, int]:
        """Resolve all books found under *root*.

        :param root: Root directory to search for book directories.
        :param dry_run: Report counts without writing output.
        :param overwrite: Re-resolve books that already have output.
        :returns: Counts dict with ``resolved``, ``skipped``, ``no_entities`` keys.
        """
        book_dirs = sorted(set(
            p.parent.parent
            for p in root.rglob("entities/page_*_entities.json")
        ))

        counts = {"resolved": 0, "skipped": 0, "no_entities": 0}
        logger.info(f"Found {len(book_dirs)} book directories with entity files")

        for book_dir in book_dirs:
            logger.info(f"Book: {book_dir.relative_to(root)}")
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
        help="Report page counts without writing output.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-resolve books that already have output.",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Phase 1 only: aggregate without LM Studio deduplication.",
    )
    parser.add_argument(
        "--lms-url", default=_LMS_URL,
        help=f"LM Studio base URL (default: {_LMS_URL}).",
    )
    parser.add_argument(
        "--lms-model", default=_LMS_MODEL,
        help=f"LM Studio model name (default: {_LMS_MODEL}). Must match a loaded model ID.",
    )
    parser.add_argument(
        "--run-tag", default=None,
        help="Tag appended to output filenames (e.g. 'lms_qwen3-32b'). "
             "Auto-derived from --lms-model if omitted.",
    )
    args = parser.parse_args()

    # Auto-derive run_tag from model name
    run_tag = args.run_tag
    if run_tag is None and not args.no_llm:
        safe_model = args.lms_model.replace("/", "_").replace(":", "_")
        run_tag = f"lms_{safe_model}"

    resolver = CoreferenceResolver(
        lms_url=args.lms_url,
        lms_model=args.lms_model,
        use_llm=not args.no_llm,
        run_tag=run_tag or "",
    )

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
