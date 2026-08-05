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
from src.datasets.document_models.scholar_normalizer import ScholarRegistry  # noqa: E402
from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer  # noqa: E402

logger = logging.getLogger(__name__)

_data_root     = _project_root / "src" / "datasets" / "raw_data" / "cairo_genizah" / "academic_literature"

# Pipeline data version. BUMP THIS whenever a change to Pass 2/3/4 logic would
# produce different output for the same inputs, so a re-run lands in a NEW
# directory instead of silently overwriting the previous generation's
# artifacts. The version also flows into the per-book run tag (see
# run_kg_overnight.RUN_TAG), which is what names the resolved-entity files, so
# bumping it protects BOTH the relations output and the Pass-3 output.
#
# History:
#   v2 — 2026-06-21 full-corpus run (qwen3.6-35b-a3b). Baseline currently in Neo4j.
#   v3 — adds the Hebrew citation-evidence detector, the shelfmark-as-Institution
#        reject rule, and the historical-author gazetteer (all landed 2026-07),
#        none of which have ever been executed against the corpus.
PIPELINE_VERSION = "v3"

_RELATIONS_ROOT = _data_root / f"relations_{PIPELINE_VERSION}"  # output root
_RELATIONS_V2   = _data_root / "relations_v2"  # previous generation — read-only, kept for comparison
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

# ORIGINATED_FROM means provenance — only a Person or Fragment has it. An
# Institution/BookArticle/Scholar "originating from" a place is always a
# publisher city, an institutional address, or a scholar's region (noise).
_PROVENANCE_SUBJECTS = {"Person", "Fragment", "BiblicalPerson"}

# Languages / scripts / linguistic phenomena that the tagger shoehorns into
# Person/Place/Institution/Fragment. Relations whose endpoint is one of these
# are dropped (we don't model languages as KG nodes).
_LANGUAGE_TERMS = {
    "hebrew", "arabic", "judaeo-arabic", "judeo-arabic", "classical arabic",
    "aramaic", "syriac", "greek", "latin", "persian", "coptic", "yiddish",
    "ladino", "judaeo-spanish", "middle arabic", "tiberian", "imala", "imāla",
    "judaeo-arabic literature", "vernacular arabic", "rabbinic hebrew",
}

# Bare/generic entity names that are not real fragments or entities.
_VAGUE_ENTITY_RE = re.compile(
    r'^(the |a |an )?(fragment|manuscript|geniza fragments?|genizah fragments?|'
    r'text|page|document|leaf|folio|item|the translator|the author|the scribe)s?$',
    re.IGNORECASE,
)

# Evidence strings that betray a bibliography/reference-list citation rather
# than substantive in-text content. These produce the bulk of the noise:
# co-editor COLLABORATED_WITH cliques, WROTE-per-citation, publisher cities.
_CITE_EDITOR_RE = re.compile(r'\b(ed|eds|edited by|trans|transl|translated by)\b\.?', re.IGNORECASE)
_CITE_PUBLISHER_RE = re.compile(r'\bpublisher\b|location of|[A-Z][a-z]+:\s+[A-Z][a-z]', re.IGNORECASE)
_CITE_AUTHOR_YEAR_RE = re.compile(r'[A-Z][A-Za-zäöüéè]+,\s+[A-Z]\.')
_YEAR_RE = re.compile(r'\b(1[5-9]\d\d|20\d\d)\b')

# Hebrew equivalents of the above. The Latin-script patterns above miss
# Hebrew-language academic citations entirely (no "ed."/"trans."/capitalised
# "Lastname, F." shape exists in Hebrew) — audited 2026-07-17: 28% of live
# COLLABORATED_WITH edges have Hebrew evidence text, and most are exactly
# this co-editor-citation-clique noise, just never caught because the
# detector only had English patterns.
_CITE_EDITOR_RE_HE = re.compile(r'(עורך|עורכים|עורכת|בעריכת|בעריכה|תרגם|תרגמה|מתרגם|מתרגמת|תרגום)')
# Page/siman-reference abbreviation ("עמ' 123", "עמ קכח", "סי' קט") —
# practically never appears in a substantive in-text quote, only in a
# citation/reference (עמ' = page, סי' = siman/paragraph in rabbinic
# literature citations).
_CITE_PAGE_RE_HE = re.compile(r"(עמ['׳״]?|סי['׳״])\s*[\dא-ת]")
# Publisher/city colon inside a parenthetical, e.g. "(ירושלים: יד יצחק
# בן־צבי, תשנ..." — the Hebrew analogue of "City: Publisher". Stands alone
# (mirrors _CITE_PUBLISHER_RE, which isn't AND-ed with a year either):
# evidence strings are sometimes truncated mid-citation before the year's
# gershayim mark ever appears, so requiring a co-occurring year missed this
# exact shape live. The match only requires the opening "(...:" shape, not
# a closing ")", to tolerate that truncation.
_CITE_PUBLISHER_RE_HE = re.compile(r'\([^)\n]{0,60}:')


def _is_citation_evidence(evidence: str) -> bool:
    """Return True if *evidence* looks like a bibliography/reference citation.

    Catches editor/translator credits ("ed. X, Y, Z"), publisher-location
    phrasing, and "Lastname, F. … <year>" citation shapes — the patterns that
    drive the COLLABORATED_WITH co-editor clique and per-citation WROTE noise
    — plus the Hebrew equivalents (עורכים/בעריכת, עמ' page refs, and a
    parenthetical "(City: Publisher..." shape).

    :param evidence: The relation's evidence string.
    :returns: True if it is a citation rather than substantive content.
    """
    ev = (evidence or "").strip()
    if not ev:
        return False
    if _CITE_EDITOR_RE.search(ev):
        return True
    if _CITE_PUBLISHER_RE.search(ev):
        return True
    if _CITE_AUTHOR_YEAR_RE.search(ev) and _YEAR_RE.search(ev):
        return True
    if _CITE_EDITOR_RE_HE.search(ev):
        return True
    if _CITE_PAGE_RE_HE.search(ev):
        return True
    if _CITE_PUBLISHER_RE_HE.search(ev):
        return True
    return False


def _is_vague_entity(name: str) -> bool:
    """Return True if *name* is a bare/generic entity (not a real one).

    :param name: Subject or object name.
    :returns: True for "Fragment", "the manuscript", "Geniza fragments", etc.
    """
    return bool(_VAGUE_ENTITY_RE.match((name or "").strip()))


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


_EXTENDED_DISCUSSION_CHARS = 120   # description length implying real discussion
_EXTENDED_DISCUSSION_PAGES = 3     # appearing on this many pages implies it too


def _transcription_pages(book_dir: Path) -> Set[int]:
    """Return the set of page sequence indices that edit/transcribe fragments.

    A page counts when it is classified ``transcription`` or carries a
    non-empty ``transcriptions`` block in its Pass-1 structured JSON.

    :param book_dir: Book root directory.
    :returns: Set of filename sequence indices.
    """
    pages: Set[int] = set()
    for p in book_dir.rglob("page_*_structured.json"):
        if "_structured" not in p.parent.name and "_structured" not in p.parent.parent.name:
            continue
        m = re.search(r"page_(\d+)", p.name)
        if not m:
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if d.get("classification") == "transcription" or (d.get("transcriptions") or {}):
            pages.add(int(m.group(1)))
    return pages


def _augment_studied(book_dir: Path, source_book: str, resolved: Dict) -> List[Dict]:
    """Emit deterministic ``author -STUDIED-> fragment`` relations.

    For each ground-truth author of the book and each genuine shelf-mark the
    book actually edits or discusses — gated on a transcription page OR
    extended discussion (long description / multi-page presence) — add a
    high-confidence STUDIED edge. These core scholarly relations are otherwise
    inconsistently produced by the LLM.

    :param book_dir: Book root directory.
    :param source_book: Book directory stem.
    :param resolved: The book's resolved entities dict.
    :returns: List of synthetic STUDIED relation dicts.
    """
    authors = ScholarRegistry.instance().book_authors.get(source_book) or []
    if not authors:
        return []
    transcription_pages = _transcription_pages(book_dir)

    studied: List[Dict] = []
    for sm in resolved.get("shelf_marks", []):
        mark = (sm.get("mark") or "").strip()
        if not mark:
            continue
        pages = sm.get("pages") or []
        desc = sm.get("description") or ""
        is_studied = (
            any(pg in transcription_pages for pg in pages)
            or len(desc) >= _EXTENDED_DISCUSSION_CHARS
            or len(pages) >= _EXTENDED_DISCUSSION_PAGES
        )
        if not is_studied:
            continue
        ev_page = next((pg for pg in pages if pg in transcription_pages), pages[0] if pages else None)
        for author in authors:
            studied.append({
                "subject":       author,
                "subject_type":  "Scholar",
                "relation":      "STUDIED",
                "object":        mark,
                "object_type":   "Fragment",
                "evidence":      f"{author} edits/discusses {mark} in this work.",
                "evidence_page": ev_page,
                "confidence":    "high",
                "source_book":   source_book,
                "augmented":     True,
            })
    return studied


def _compact_relations(
    relations: List[Dict],
    person_labels: Dict[str, str],
) -> Tuple[List[Dict], List[Dict]]:
    """Split raw relations into accepted (clean) and rejected sets.

    Applies deterministic rules before any Neo4j write (rejected relations are
    quarantined with a ``reject_reason``, not deleted):

    0. **Authoritative retyping** — endpoints that are resolved people get
       their canonical label (Scholar/BiblicalPerson/Person) from Pass-3.
    1. **Self-loops** — drop subject == object.
    2. **Exact dedup** — collapse identical (subject, relation, object).
    3. **Bibliography/citation noise** — drop relations whose evidence is a
       reference-list citation (co-editor COLLABORATED_WITH cliques, per-citation
       WROTE, publisher cities). The single biggest noise source.
    4. **Language/concept endpoints** — drop relations whose endpoint is a
       language/script/linguistic term (Judaeo-Arabic, imāla, Hebrew…).
    5. **Vague endpoints** — drop bare "Fragment"/"the manuscript"/etc.
    5b. **Shelfmark-as-Institution** — drop relations where an "Institution"
        endpoint is actually a shelfmark string (per
        :meth:`ShelfmarkNormalizer.is_probable_shelfmark`), e.g. a mistyped
        HELD_AT object like "ENA 3902.5 verso".
    6. **Provenance** — ORIGINATED_FROM only from Person/Fragment (drops
       publisher cities, institution addresses, scholar regions).
    7. **Type pairing** — drop triples violating :data:`ALLOWED_RELATIONS`
       (Person/Scholar/BiblicalPerson interchangeable; ``None`` = any).
    8. **Biblical scope** — a BiblicalPerson edge survives only opposite a
       Fragment.

    :param relations: Raw validated relations from the LLM.
    :param person_labels: Map of resolved person name → authoritative label.
    :returns: ``(accepted, rejected)``.
    """
    accepted: List[Dict] = []
    rejected: List[Dict] = []
    seen: set = set()

    def reject(r: Dict, reason: str) -> None:
        rejected.append({**r, "reject_reason": reason})

    for r in relations:
        subj, rel, obj = r["subject"], r["relation"], r["object"]

        # 0. authoritative retyping by name (Pass-3 person_class)
        if subj in person_labels:
            r["subject_type"] = person_labels[subj]
        if obj in person_labels:
            r["object_type"] = person_labels[obj]

        # 1. self-loop
        if subj.strip().lower() == obj.strip().lower():
            reject(r, "self_loop")
            continue

        # 2. exact duplicate
        key = (subj, rel, obj)
        if key in seen:
            continue
        seen.add(key)

        # 3. bibliography / reference-list citation noise
        if _is_citation_evidence(r.get("evidence", "")):
            reject(r, "bibliography_citation")
            continue

        # 4. language / linguistic-concept endpoint
        if subj.strip().lower() in _LANGUAGE_TERMS or obj.strip().lower() in _LANGUAGE_TERMS:
            reject(r, "language_concept")
            continue

        # 5. vague / generic endpoint
        if _is_vague_entity(subj) or _is_vague_entity(obj):
            reject(r, "vague_entity")
            continue

        st, ot = r["subject_type"], r["object_type"]

        # 5b. shelfmark mistyped as Institution — the model sometimes emits a
        # HELD_AT (Fragment -> Institution) relation where the "institution"
        # slot is actually another shelfmark string (e.g. "Bodl. MS. Heb. b.
        # 12", "ENA 3902.5 verso"). These are real shelfmarks, not repository
        # names, and geocoding them as institutions produces nonsense pins
        # (e.g. "ENA ..." matching the Japanese city Ena).
        #
        # Only the strict "standard" classification is used (anchor + a
        # digit), not classify_shelfmark's looser "berlin" bucket — that one
        # also matches real institution names like "Jüdische
        # Gemeindebibliothek (Berlin)" on a bare substring/prefix check, which
        # would wrongly reject genuine HELD_AT relations to that library.
        if st == "Institution" and ShelfmarkNormalizer.classify_shelfmark(subj) == "standard":
            reject(r, "institution_is_shelfmark")
            continue
        if ot == "Institution" and ShelfmarkNormalizer.classify_shelfmark(obj) == "standard":
            reject(r, "institution_is_shelfmark")
            continue

        # 6. provenance: ORIGINATED_FROM only from Person/Fragment
        if rel == "ORIGINATED_FROM" and st not in _PROVENANCE_SUBJECTS:
            reject(r, f"originated_from_nonprovenance ({st})")
            continue

        # 7. type-pairing validation against ALLOWED_RELATIONS
        exp_subj, exp_obj = ALLOWED_RELATIONS.get(rel, (None, None))
        if not _type_ok(st, exp_subj):
            reject(r, f"subject_type {st} != expected {exp_subj} for {rel}")
            continue
        if not _type_ok(ot, exp_obj):
            reject(r, f"object_type {ot} != expected {exp_obj} for {rel}")
            continue

        # 8. biblical scope — only Fragment↔BiblicalPerson edges survive
        if "BiblicalPerson" in (st, ot) and "Fragment" not in (st, ot):
            reject(r, "biblical_non_fragment")
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
            # HARD FAILURE, deliberately. The old behavior fell back to the
            # untagged book_entities_resolved.json — i.e. a PREVIOUS
            # generation's Pass-3 output — and still stamped the result with
            # the current pipeline_version (2026-08 audit finding). In a
            # corpus run that turned any Pass-3 failure into v3-labeled
            # relations built on stale June entities, with only an INFO log.
            # Raising makes the driver mark the book failed (no sentinel), so
            # it is retried instead of silently mixed across generations.
            raise FileNotFoundError(
                f"{book_dir.name}: tagged resolved file {resolved_filename!r} "
                f"not found — Pass 3 has not completed for this run tag. "
                f"Refusing to fall back to the untagged previous-generation "
                f"file; run Pass 3 first.")

        # Output: relations_<version>/<book_name>/ for Gemini,
        #         relations_<version>/<book_name>/<run_tag>/ for LMS runs
        out_dir = _RELATIONS_ROOT / book_dir.name
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
        # Deterministic augmentation: author -STUDIED-> fragment for fragments
        # the book actually edits/discusses (these core edges are otherwise
        # inconsistently produced by the LLM).
        augmented = _augment_studied(book_dir, source_book, resolved)
        if augmented:
            logger.info(f"  {source_book}: +{len(augmented)} augmented STUDIED relation(s)")
        all_relations.extend(augmented)

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
            "source_book":     source_book,
            "pipeline_version": PIPELINE_VERSION,
            "extracted_at":    extracted_at,
            "model_used":      model_used,
            "relation_count":  len(accepted),
            "relations":       accepted,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        rejected_path = out_dir / _REJECTED_FILE
        rejected_output = {
            "source_book":     source_book,
            "pipeline_version": PIPELINE_VERSION,
            "extracted_at":    extracted_at,
            "model_used":      model_used,
            "relation_count":  len(rejected),
            "relations":       rejected,
        }
        with open(rejected_path, "w", encoding="utf-8") as f:
            json.dump(rejected_output, f, indent=2, ensure_ascii=False)

        # len(all_relations) - accepted - rejected = exact duplicates, which
        # _compact_relations drops without recording in either list. Count
        # them separately — the old log lumped them into "rejected", which
        # made the log disagree with the rejected file (240 vs 193 confusion
        # in the 2026-08 audit).
        n_dupes = len(all_relations) - len(accepted) - len(rejected)
        logger.info(
            f"  {source_book}: wrote {len(accepted)} relations → {output_path.name} "
            f"({len(rejected)} rejected → {rejected_path.name}; "
            f"{n_dupes} exact duplicate(s) dropped)"
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
