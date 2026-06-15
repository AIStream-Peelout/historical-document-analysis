#!/usr/bin/env python3
"""
Pass 4 — Relation Extractor
=============================

Reads the resolved entity list (Pass 3 output) and the structured page text
(Pass 1 output) for each book, then extracts typed relationships between
entities.  Relation extraction is scoped to the pages where each entity
actually appears (from Pass 3), keeping prompts focused and evidence-grounded.

This pass is **pure file I/O** — no Neo4j dependency.  The output JSON is
consumed by ``academic_kg_import.py`` during the Neo4j import step.

Input
-----
- ``book_entities_resolved.json``  (Pass 3 output, one per book)
- ``page_NNN_structured.json``     (Pass 1 output, used for page text)

Output
------
``book_relations.json`` in the book directory::

    {
        "source_book": "india_trader_1_50",
        "extracted_at": "2026-06-08T12:00:00Z",
        "model_used":   "gemini-3.5-flash",
        "relations": [
            {
                "subject":       "Joseph b. David Lebdi",
                "subject_type":  "Person",
                "relation":      "TRAVELED_TO",
                "object":        "Aden",
                "object_type":   "Place",
                "evidence":      "Joseph Lebdi set out from Egypt via Aden to India",
                "evidence_page": 9,
                "source_book":   "india_trader_1_50",
                "confidence":    "high"
            }
        ]
    }

Allowed relations
-----------------
LIVED_IN          Person      → Place         (historical residence)
TRAVELED_TO       Person      → Place         (attested journey)
ORIGINATED_FROM   Person      → Place         (provenance / birthplace)
ORIGINATED_FROM   Fragment    → Place         (document origin)
AFFILIATED_WITH   Scholar     → Institution   (academic affiliation)
WROTE             Scholar     → BookArticle   (authorship)
TRANSCRIBED       Scholar     → Fragment      (produced transcription)
STUDIED           Scholar     → Fragment      (scholarly focus)
COLLABORATED_WITH Scholar     → Scholar       (co-authorship / acknowledgment)
MARRIED_TO        Person      → Person        (attested in documents)
RELATED_TO        Person      → Person        (family relation)
MENTIONS_PERSON   Fragment    → Person        (person named in document)
MENTIONS_PLACE    Fragment    → Place         (place named in document)
ORIGINATED_IN     Fragment    → Place         (document origin location)
HELD_AT           Fragment    → Institution   (current holding institution)
CITED_IN          Fragment    → BookArticle   (fragment discussed in article)

Usage
-----
python relationship_extractor.py --dir academic_literature/india_traders/india_trader_1_50
python relationship_extractor.py             # all books
python relationship_extractor.py --dry-run   # count without calling LLM
python relationship_extractor.py --overwrite
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import dotenv

_project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(_project_root))
dotenv.load_dotenv(_project_root / ".env")

from src.models.llm.academic.llm_client import LLMClient  # noqa: E402

logger = logging.getLogger(__name__)

_data_root     = _project_root / "src" / "datasets" / "raw_data" / "cairo_genizah" / "academic_literature"
_RELATIONS_V2  = _data_root / "relations_v2"   # output root — separate from old enriched_relations/
_OUTPUT_FILE   = "book_relations.json"
_REJECTED_FILE = "book_relations_rejected.json"
_RESOLVED_FILE = "book_entities_resolved.json"

# ---------------------------------------------------------------------------
# Allowed relation types and their expected subject/object types
# ---------------------------------------------------------------------------

ALLOWED_RELATIONS: Dict[str, tuple] = {
    "LIVED_IN":          ("Person",   "Place"),
    "TRAVELED_TO":       ("Person",   "Place"),
    "ORIGINATED_FROM":   (None,       "Place"),        # Person or Fragment
    "AFFILIATED_WITH":   ("Scholar",  "Institution"),
    "WROTE":             ("Scholar",  "BookArticle"),
    "TRANSCRIBED":       ("Scholar",  "Fragment"),
    "STUDIED":           ("Scholar",  "Fragment"),
    "COLLABORATED_WITH": ("Scholar",  "Scholar"),
    "MARRIED_TO":        ("Person",   "Person"),
    "RELATED_TO":        ("Person",   "Person"),
    "MENTIONS_PERSON":   ("Fragment", "Person"),
    "MENTIONS_PLACE":    ("Fragment", "Place"),
    "ORIGINATED_IN":     ("Fragment", "Place"),
    "HELD_AT":           ("Fragment", "Institution"),
    "CITED_IN":          ("Fragment", "BookArticle"),
}

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
/no_think
You are an expert in medieval Jewish history, the Cairo Genizah, and academic
scholarship on the medieval Mediterranean world.

Extract factual relationships from academic text.  Only extract what is directly
supported by the text — do not infer or speculate.

Output ONLY valid JSON. No markdown, no code fences, no explanation.
"""

# Valid KG node labels — constrains subject_type/object_type so the model
# can't invent types like "Book", "Work", or "Shelf Mark" (which would be
# coerced to Entity nodes at import).
_ENTITY_LABELS = ["Person", "Scholar", "Place", "Institution", "Fragment", "BookArticle"]

# JSON Schema passed to LM Studio's structured-output API for constrained
# decoding.  Guarantees valid JSON and bypasses Qwen3 "thinking" mode, which
# otherwise emits chain-of-thought prose instead of JSON.
_RELATIONS_JSON_SCHEMA: Dict = {
    "type": "object",
    "properties": {
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject":       {"type": "string"},
                    "subject_type":  {"type": "string", "enum": _ENTITY_LABELS},
                    "relation":      {"type": "string", "enum": sorted(ALLOWED_RELATIONS)},
                    "object":        {"type": "string"},
                    "object_type":   {"type": "string", "enum": _ENTITY_LABELS},
                    "evidence":      {"type": "string"},
                    "evidence_page": {"type": "integer"},
                    "confidence":    {"type": "string", "enum": ["high", "medium"]},
                },
                "required": ["subject", "subject_type", "relation", "object",
                             "object_type", "evidence", "confidence"],
            },
        },
    },
    "required": ["relations"],
}


def _build_entity_prompt(
    entity: Dict,
    entity_type: str,
    page_texts: List[str],
    all_entities: Dict,
    source_book: str,
) -> str:
    """Build a relation extraction prompt scoped to one entity.

    :param entity: Resolved entity dict (name, role/type, pages, aliases, description).
    :param entity_type: KG label (Person, Scholar, Place, Institution, Fragment).
    :param page_texts: List of page text snippets where this entity appears.
    :param all_entities: Full resolved entity dict (for object lookup hints).
    :param source_book: Book stem name.
    :returns: Prompt string.
    """
    name = entity.get("name") or entity.get("mark", "")
    aliases = entity.get("aliases", [])
    description = entity.get("description", "")
    pages = entity.get("pages", [])

    # Build a hint list of known entities for the LLM to use as objects
    known_people  = [p["name"] for p in all_entities.get("people", [])][:30]
    known_places  = [p["name"] for p in all_entities.get("places", [])][:30]
    known_insts   = [p["name"] for p in all_entities.get("institutions", [])][:20]
    known_marks   = [p["mark"] for p in all_entities.get("shelf_marks", [])][:30]

    relations_list = "\n".join(
        f"  {rel:25s} {(st or 'any'):10s} → {ot or 'any'}"
        for rel, (st, ot) in ALLOWED_RELATIONS.items()
    )

    text_block = "\n\n---\n\n".join(page_texts[:8])  # cap at 8 page excerpts

    alias_str = f" (also: {', '.join(aliases)})" if aliases else ""

    known_block = ""
    if known_people:
        known_block += f"Known people in this book: {', '.join(known_people[:15])}\n"
    if known_places:
        known_block += f"Known places in this book: {', '.join(known_places[:15])}\n"
    if known_marks:
        known_block += f"Known shelf marks: {', '.join(known_marks[:15])}\n"

    return f"""\
Book: {source_book}
Entity: "{name}"{alias_str}
Type: {entity_type}
Description: {description}
Pages where this entity appears: {pages}

{known_block}
Text passages from pages where "{name}" appears:

{text_block}

---
Extract ALL factual relationships involving "{name}" that are directly supported
by the text above.

Allowed relation types (subject_type → object_type):
{relations_list}

Rules:
- The subject of every triplet MUST be exactly "{name}" — never a short form, pronoun,
  or alias even if that is what the text uses. Same for objects: always use the full
  canonical name from the known entity lists above, not the short form in the text.
  Example: if the text says "Abraham" but the known people list has "Abraham Ben Yiju",
  use "Abraham Ben Yiju".
- evidence must be a direct quote or close paraphrase from the text above
- confidence: "high" = direct statement, "medium" = clear implication; omit "low"
- evidence_page: the page number the evidence comes from
- do NOT invent entities not mentioned in the text

Return ONLY this JSON:
{{
  "relations": [
    {{
      "subject":       "...",
      "subject_type":  "Person|Scholar|Place|Institution|Fragment|BookArticle",
      "relation":      "LIVED_IN",
      "object":        "...",
      "object_type":   "Person|Scholar|Place|Institution|Fragment|BookArticle",
      "evidence":      "direct quote or paraphrase",
      "evidence_page": 27,
      "confidence":    "high|medium"
    }}
  ]
}}

If no relationships can be extracted: {{"relations": []}}
"""


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)


def _parse_response(raw: str, source_book: str) -> List[Dict]:
    """Parse and validate relation triples from LLM output.

    :param raw: Raw LLM output.
    :param source_book: Book name to stamp on each relation.
    :returns: List of validated relation dicts.
    """
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    match = _JSON_RE.search(raw)
    if not match:
        return []
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    valid = []
    for r in (data.get("relations") or []):
        subj     = (r.get("subject")      or "").strip()
        rel      = (r.get("relation")     or "").strip().upper()
        obj      = (r.get("object")       or "").strip()
        subj_t   = (r.get("subject_type") or "").strip()
        obj_t    = (r.get("object_type")  or "").strip()
        evidence = (r.get("evidence")     or "").strip()[:600]
        conf     = (r.get("confidence")   or "medium").strip()
        ev_page  = r.get("evidence_page")

        if not (subj and rel and obj):
            continue
        if rel not in ALLOWED_RELATIONS:
            logger.debug(f"  Unknown relation '{rel}' — skipped")
            continue
        if conf == "low":
            continue

        valid.append({
            "subject":       subj,
            "subject_type":  subj_t,
            "relation":      rel,
            "object":        obj,
            "object_type":   obj_t,
            "evidence":      evidence,
            "evidence_page": ev_page,
            "source_book":   source_book,
            "confidence":    conf,
        })

    return valid


# ---------------------------------------------------------------------------
# Heuristic compaction (deterministic — no LLM)
# ---------------------------------------------------------------------------

# Person-like labels are interchangeable for type-pairing: Pass 4 can't
# reliably tell Person from Scholar (import resolves that), so we only check
# the structural bucket.
_PERSONISH = {"Person", "Scholar", "BiblicalPerson"}


def _type_ok(actual: str, expected: Optional[str]) -> bool:
    """Return True if *actual* type satisfies the *expected* slot type.

    ``None`` expected means "any".  When a person-ish type is expected, any
    of Person/Scholar/BiblicalPerson is accepted; otherwise the type must
    match exactly (Fragment, Place, Institution, BookArticle).

    :param actual: The type emitted for this subject/object.
    :param expected: The type ``ALLOWED_RELATIONS`` expects, or ``None``.
    :returns: True if the pairing is structurally valid.
    """
    if expected is None:
        return True
    if expected in _PERSONISH:
        return actual in _PERSONISH
    return actual == expected


def _compact_relations(
    relations: List[Dict],
    person_labels: Dict[str, str],
) -> Tuple[List[Dict], List[Dict]]:
    """Split raw relations into accepted (clean) and rejected sets.

    Applies deterministic rules before any Neo4j write:

    0. **Authoritative retyping** — any endpoint whose name is a resolved
       person is retyped to that person's canonical label (Scholar /
       BiblicalPerson / Person), derived from Pass-3 ``person_class``.  This
       is what fixes in-text scholars/collectors the LLM mislabelled as
       "Person" — the node label no longer depends on the model's guess.
    1. **Exact dedup** — collapse identical (subject, relation, object) triples.
    2. **Type pairing** — drop triples whose ``(subject_type, object_type)``
       violate :data:`ALLOWED_RELATIONS` (e.g. ``WROTE → Place``).  Person /
       Scholar / BiblicalPerson are interchangeable here (structural shape
       only); ``None`` means "any".
    3. **Biblical scope** — a relation touching a ``BiblicalPerson`` is kept
       only when the *other* endpoint is a ``Fragment`` (keep "fragment
       discusses Abraham"; drop biblical person↔person edges).

    :param relations: Raw validated relations from the LLM.
    :param person_labels: Map of resolved person name → authoritative label
        (``Scholar`` / ``BiblicalPerson`` / ``Person``).
    :returns: ``(accepted, rejected)``.  Each rejected relation carries a
        ``reject_reason`` field.
    """
    accepted: List[Dict] = []
    rejected: List[Dict] = []
    seen: set = set()

    for r in relations:
        subj, rel, obj = r["subject"], r["relation"], r["object"]

        # 0. authoritative retyping by name (Pass-3 person_class)
        if subj in person_labels:
            r["subject_type"] = person_labels[subj]
        if obj in person_labels:
            r["object_type"] = person_labels[obj]

        # 1. exact duplicate
        key = (subj, rel, obj)
        if key in seen:
            continue
        seen.add(key)

        # 2. type-pairing validation against ALLOWED_RELATIONS.
        # Person/Scholar/BiblicalPerson are interchangeable here: the
        # Person-vs-Scholar distinction is resolved later at import time
        # (ScholarRegistry), so we only enforce the structural shape
        # (person-ish vs Fragment/Place/Institution/BookArticle).
        exp_subj, exp_obj = ALLOWED_RELATIONS.get(rel, (None, None))
        st, ot = r["subject_type"], r["object_type"]
        if not _type_ok(st, exp_subj):
            rejected.append({**r, "reject_reason": f"subject_type {st} != expected {exp_subj} for {rel}"})
            continue
        if not _type_ok(ot, exp_obj):
            rejected.append({**r, "reject_reason": f"object_type {ot} != expected {exp_obj} for {rel}"})
            continue

        # 3b. biblical scope — only Fragment↔BiblicalPerson edges survive
        touches_biblical = "BiblicalPerson" in (st, ot)
        if touches_biblical and "Fragment" not in (st, ot):
            rejected.append({**r, "reject_reason": "biblical_non_fragment"})
            continue

        accepted.append(r)

    return accepted, rejected


# ---------------------------------------------------------------------------
# Page text loader
# ---------------------------------------------------------------------------

def _load_page_texts(book_dir: Path, page_numbers: List[int]) -> List[str]:
    """Load full_main_text for the specified pages from structured JSONs.

    :param book_dir: Book directory containing structured JSON files.
    :param page_numbers: List of page numbers to load.
    :returns: List of formatted text snippets, one per page.
    """
    # Build page_number → structured file map.  The filename sequence index
    # is the canonical page number pipeline-wide (entity files, resolved
    # files, and relations all use it); extracted_page_number is the printed
    # page and is unreliable.
    page_map: Dict[int, Path] = {}
    for p in book_dir.rglob("page_*_structured.json"):
        # Accept both single-level (book_structured_*/page.json)
        # and two-level (book_structured_*/model/page.json) layouts
        if "_structured" not in p.parent.name and "_structured" not in p.parent.parent.name:
            continue
        m = re.search(r"page_(\d+)", p.name)
        if m:
            page_map.setdefault(int(m.group(1)), p)

    texts = []
    for pg in sorted(set(page_numbers)):
        path = page_map.get(pg)
        if not path:
            continue
        try:
            d = json.load(open(path, encoding="utf-8"))
            text = (d.get("full_main_text") or "").strip()
            if text:
                texts.append(f"[Page {pg}]\n{text[:1200]}")
        except Exception:
            continue

    return texts


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------

class RelationExtractor:
    """Extracts typed relationships from resolved entities + page text (Pass 4).

    :param client: Initialised LLMClient.
    """

    def __init__(self, client: LLMClient, run_tag: str = ""):
        self.client  = client
        self.run_tag = run_tag  # e.g. "lms_qwen3-32b"; empty = default Gemini output

    def _entity_kg_type(self, entity: Dict, category: str) -> str:
        """Map a resolved entity category + role to a Neo4j label.

        :param entity: Resolved entity dict.
        :param category: Category from resolved file: people/places/institutions/shelf_marks.
        :returns: Neo4j label string.
        """
        if category == "people":
            role = (entity.get("role") or "historical_person").lower()
            if role == "scholar":
                return "Scholar"
            return "Person"   # historical_person and collector both → Person
        if category == "places":
            return "Place"
        if category == "institutions":
            return "Institution"
        if category == "shelf_marks":
            return "Fragment"
        return "Entity"

    def extract_book(
        self,
        book_dir: Path,
        dry_run: bool = False,
        overwrite: bool = False,
    ) -> int:
        """Extract relations for one book directory.

        :param book_dir: Directory containing ``book_entities_resolved.json``.
        :param dry_run: Count entities without calling the LLM.
        :param overwrite: Re-extract even if output already exists.
        :returns: Number of relations extracted.
        """
        # Read tagged resolved file if available, fall back to default
        resolved_filename = (_RESOLVED_FILE.replace(".json", f"_{self.run_tag}.json")
                             if self.run_tag else _RESOLVED_FILE)
        resolved_path = book_dir / resolved_filename
        if not resolved_path.exists() and self.run_tag:
            resolved_path = book_dir / _RESOLVED_FILE
            logger.info(f"  {book_dir.name}: no tagged resolved file, using default")

        # Output: relations_v2/<book_name>/ for Gemini,
        #         relations_v2/<book_name>/<run_tag>/ for LMS runs
        out_dir = _RELATIONS_V2 / book_dir.name
        if self.run_tag:
            out_dir = out_dir / self.run_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / _OUTPUT_FILE

        if not resolved_path.exists():
            logger.warning(f"  {book_dir.name}: no {_RESOLVED_FILE} (run Pass 3 first)")
            return 0

        if output_path.exists() and not overwrite:
            logger.info(f"  {book_dir.name}: output exists, skipping (use --overwrite)")
            return 0

        resolved = json.load(open(resolved_path, encoding="utf-8"))
        source_book = resolved.get("source_book", book_dir.name)

        # Build the flat entity list: (entity_dict, category)
        candidates = []
        for category in ("people", "places", "institutions", "shelf_marks"):
            for entity in resolved.get(category, []):
                pages = entity.get("pages", [])
                if pages:  # only entities with page attribution
                    candidates.append((entity, category))

        logger.info(f"  {source_book}: {len(candidates)} entities with page attribution")

        if dry_run:
            return len(candidates)

        all_relations: List[Dict] = []

        for entity, category in candidates:
            name  = entity.get("name") or entity.get("mark", "")
            pages = entity.get("pages", [])
            kg_type = self._entity_kg_type(entity, category)

            # Load page text scoped to this entity's pages
            page_texts = _load_page_texts(book_dir, pages)
            if not page_texts:
                logger.debug(f"  {name}: no page text found, skipping")
                continue

            prompt = _build_entity_prompt(
                entity, kg_type, page_texts, resolved, source_book
            )

            try:
                # Constrained JSON schema for LM Studio (grammar-sampled);
                # Gemini handles the JSON instruction via the prompt.
                schema = (_RELATIONS_JSON_SCHEMA
                          if self.client.backend == "lm_studio" else None)
                raw       = self.client.complete(_SYSTEM_PROMPT, prompt,
                                                 response_schema=schema)
                relations = _parse_response(raw, source_book)
            except Exception as e:
                logger.error(f"  LLM error for '{name}': {e}")
                continue

            if relations:
                all_relations.extend(relations)
                logger.info(f"  {name} ({kg_type}): {len(relations)} relation(s)")
            else:
                logger.debug(f"  {name}: no relations found")

        # ------------------------------------------------------------------
        # Heuristic compaction (deterministic): dedup + type-pairing + biblical
        # scope.  Accepted → book_relations.json; rejected → a sidecar file
        # for manual review (kept out of the automated import).
        # ------------------------------------------------------------------
        _class_to_label = {"biblical": "BiblicalPerson", "scholar": "Scholar",
                           "historical": "Person"}
        person_labels = {
            p["name"]: _class_to_label.get(p.get("person_class", "historical"), "Person")
            for p in resolved.get("people", [])
        }
        accepted, rejected = _compact_relations(all_relations, person_labels)

        model_used = (self.client.lms_model
                      if self.client.backend == "lm_studio" else "gemini")
        extracted_at = datetime.now(timezone.utc).isoformat()

        output = {
            "source_book":    source_book,
            "extracted_at":   extracted_at,
            "model_used":     model_used,
            "relation_count": len(accepted),
            "relations":      accepted,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        rejected_path = out_dir / _REJECTED_FILE
        rejected_output = {
            "source_book":    source_book,
            "extracted_at":   extracted_at,
            "model_used":     model_used,
            "relation_count": len(rejected),
            "relations":      rejected,
        }
        with open(rejected_path, "w", encoding="utf-8") as f:
            json.dump(rejected_output, f, indent=2, ensure_ascii=False)

        logger.info(
            f"  {source_book}: wrote {len(accepted)} relations → {output_path.name} "
            f"({len(all_relations) - len(accepted)} rejected → {rejected_path.name})"
        )
        return len(accepted)

    def extract_all(
        self,
        root: Path,
        dry_run: bool = False,
        overwrite: bool = False,
    ) -> Dict[str, int]:
        """Extract relations for all books under *root*.

        :param root: Root directory to search.
        :param dry_run: Count without calling the LLM.
        :param overwrite: Re-extract books that already have output.
        :returns: Counts dict.
        """
        book_dirs = sorted(set(
            p.parent for p in root.rglob(_RESOLVED_FILE)
        ))

        counts = {"books": len(book_dirs), "relations": 0, "skipped": 0}
        logger.info(f"Found {len(book_dirs)} books with resolved entities")

        for book_dir in book_dirs:
            logger.info(f"Book: {book_dir.relative_to(root)}")
            n = self.extract_book(book_dir, dry_run=dry_run, overwrite=overwrite)
            if n == 0 and not dry_run:
                counts["skipped"] += 1
            else:
                counts["relations"] += n

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
            logging.FileHandler("relation_extractor.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dir", "-d", metavar="SUBDIR", default=None)
    parser.add_argument("--dry-run", "-n", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backend", choices=["gemini", "lm_studio"], default="gemini")
    parser.add_argument("--lms-model",    default="qwen/qwen3.6-35b-a3b")
    parser.add_argument("--lms-url",      default="http://localhost:1234/v1")
    parser.add_argument("--gemini-model", default="gemini-3.5-flash")
    parser.add_argument(
        "--run-tag", default=None,
        help="Tag for output subdir (e.g. 'lms_qwen3-32b'). "
             "Auto-derived from --lms-model when --backend lm_studio if omitted.",
    )
    args = parser.parse_args()

    # Auto-derive run_tag from model name when using LM Studio
    run_tag = args.run_tag
    if run_tag is None and args.backend == "lm_studio":
        safe_model = args.lms_model.replace("/", "_").replace(":", "_")
        run_tag = f"lms_{safe_model}"

    client = LLMClient(
        backend=args.backend,
        lms_url=args.lms_url,
        lms_model=args.lms_model,
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=args.gemini_model,
    )

    extractor = RelationExtractor(client, run_tag=run_tag or "")

    root = _data_root
    if args.dir:
        root = _data_root / args.dir
        if not root.exists():
            logger.error(f"Directory not found: {root}")
            sys.exit(1)

    counts = extractor.extract_all(root, dry_run=args.dry_run, overwrite=args.overwrite)

    tag = "[DRY RUN] " if args.dry_run else ""
    print(f"\n{tag}Relation extraction complete")
    print("=" * 40)
    print(f"  Books processed: {counts['books']:,}")
    print(f"  Relations found: {counts['relations']:,}")
    print(f"  Skipped:         {counts['skipped']:,}")
    print("=" * 40)


if __name__ == "__main__":
    main()
