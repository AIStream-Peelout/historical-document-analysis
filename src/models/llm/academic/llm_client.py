#!/usr/bin/env python3
"""
Entity Relation Extractor — Pass 3 LLM logic

Handles everything needed to extract entity-anchored KG triplets from
academic literature and write them to reviewable JSON files.  No Neo4j
dependency — this module is pure LLM + file I/O.

The Neo4j candidate fetch and import live in:
  src/datasets/indexing/neo4j/enrich_node_relations.py  (extraction CLI)
  src/datasets/indexing/neo4j/academic_kg_import.py     (import into Neo4j)
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file)
# ---------------------------------------------------------------------------
_project_root = Path(__file__).parent.parent.parent.parent.parent
_data_root    = _project_root / "src" / "datasets" / "raw_data" / "cairo_genizah" / "academic_literature"
output_root   = _data_root / "enriched_relations"

# ---------------------------------------------------------------------------
# Backend constants
# ---------------------------------------------------------------------------
BACKEND_LMS    = "lm_studio"
BACKEND_GEMINI = "gemini"

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------
ALLOWED_RELATIONS = {
    "LIVED_IN", "TRAVELED_TO", "AFFILIATED_WITH", "ORIGINATED_FROM",
    "MENTIONED_IN", "MENTIONS_PERSON", "MENTIONS_PLACE",
    "CONTEMPORARY_OF", "STUDENT_OF", "TEACHER_OF",
    "RELATED_TO", "MARRIED_TO", "WROTE", "COMMISSIONED",
    "SENT_TO", "RECEIVED_FROM",
}

# (subject_hint, object_hint) — used when the LLM omits a type
_REL_OBJECT_HINTS: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "LIVED_IN":        (None,       "Place"),
    "TRAVELED_TO":     (None,       "Place"),
    "ORIGINATED_FROM": (None,       "Place"),
    "AFFILIATED_WITH": (None,       "Institution"),
    "STUDENT_OF":      (None,       "Person"),
    "TEACHER_OF":      (None,       "Person"),
    "CONTEMPORARY_OF": (None,       "Person"),
    "RELATED_TO":      (None,       "Person"),
    "MARRIED_TO":      (None,       "Person"),
    "SENT_TO":         (None,       "Person"),
    "RECEIVED_FROM":   (None,       "Person"),
    "WROTE":           (None,       "BookArticle"),
    "COMMISSIONED":    (None,       None),
    "MENTIONED_IN":    (None,       "BookArticle"),
    "MENTIONS_PERSON": ("Fragment", "Person"),
    "MENTIONS_PLACE":  ("Fragment", "Place"),
}

_SYSTEM_PROMPT = """\
/no_think
You are an expert in medieval Jewish history and the Cairo Genizah.
Extract factual relationships for a specific entity from academic literature excerpts.
The text may contain Hebrew, Aramaic, Judeo-Arabic, or other languages — treat all
languages equally and extract relationships from any language content you understand.
Only extract what is directly supported by the text. Do not infer or speculate.
Output ONLY valid JSON — no markdown, no explanation, no thinking tags.
"""


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

class LLMClient:
    """Thin wrapper for LM Studio (OpenAI-compat) or Gemini."""

    def __init__(
        self,
        backend:        str = BACKEND_LMS,
        lms_url:        str = "http://localhost:1234/v1",
        lms_model:      str = "qwen3:8b",
        gemini_api_key: Optional[str] = None,
        gemini_model:   str = "gemini-3.5-flash",
        timeout:        int = 120,
    ):
        self.backend      = backend
        self.lms_url      = lms_url.rstrip("/")
        self.lms_model    = lms_model
        self.gemini_model = gemini_model
        self.timeout      = timeout
        self._gemini      = None

        if backend == BACKEND_GEMINI:
            try:
                from google import genai
                from google.genai import types as genai_types
                api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY required for Gemini backend")
                self._gemini       = genai.Client(api_key=api_key)
                self._gemini_model = gemini_model
                self._genai_types  = genai_types
                logger.info(f"Gemini client ready (model={gemini_model})")
            except ImportError:
                raise ImportError("pip install google-genai for Gemini support")
        else:
            self._verify_lms()

    def _verify_lms(self):
        try:
            r = requests.get(f"{self.lms_url}/models", timeout=5)
            models = [m["id"] for m in r.json().get("data", [])]
            logger.info(f"LM Studio connected. Available: {models}")
            if not any(self.lms_model in m for m in models):
                logger.warning(f"Model '{self.lms_model}' not in {models}")
        except Exception as e:
            logger.warning(f"Could not verify LM Studio: {e}")

    def complete(self, system: str, user: str, max_context_chars: int = 4000,
                 response_schema: Optional[Dict] = None) -> str:
        """Run a completion, optionally enforcing a JSON schema for structured output.

        :param system: System prompt.
        :param user: User prompt.
        :param max_context_chars: Soft limit on user prompt length before trimming.
        :param response_schema: Optional JSON Schema dict; when provided the LM Studio
            SDK enforces constrained decoding so output is guaranteed to match.
        :returns: Model response as a string.
        """
        if self.backend == BACKEND_GEMINI:
            resp = self._gemini.models.generate_content(
                model=self._gemini_model,
                contents=f"{system}\n\n{user}",
            )
            return resp.text
        return self._complete_lms(system, user, max_context_chars, response_schema)

    def _complete_lms(self, system: str, user: str, max_context_chars: int,
                      response_schema: Optional[Dict]) -> str:
        """Call LM Studio via the native SDK with optional structured output.

        Uses the ``lmstudio`` Python SDK which supports JSON-schema constrained
        decoding (grammar sampling), guaranteeing the output matches the schema.
        Falls back to halving the context on 400 errors.

        :param system: System prompt.
        :param user: User prompt.
        :param max_context_chars: Soft limit; halved on each 400 retry.
        :param response_schema: JSON Schema dict for constrained output, or None.
        :returns: Model response content string.
        """
        import lmstudio as lms

        # SDK connects via WebSocket; strip the /v1 REST suffix to get host:port
        host_port = (self.lms_url
                     .replace("http://", "")
                     .replace("https://", "")
                     .rstrip("/")
                     .removesuffix("/v1"))

        pred_config = lms.LlmPredictionConfig(temperature=0.1, max_tokens=4096)

        for ctx_shift in range(3):
            trimmed_user = _trim_passages(user, max_context_chars >> ctx_shift)
            chat = lms.Chat(system)
            chat.add_user_message(trimmed_user)
            try:
                with lms.Client(host_port) as client:
                    model = client.llm.model(self.lms_model)
                    result = model.respond(
                        chat,
                        config=pred_config,
                        response_format=response_schema,   # None = unstructured
                    )
                    return result.content
            except Exception as e:
                err = str(e).lower()
                if ("context" in err or "token" in err or "length" in err) and ctx_shift < 2:
                    logger.warning(f"  LMS context error (attempt {ctx_shift+1}) — halving and retrying: {e}")
                    continue
                raise
        raise RuntimeError("LMS failed after all context-reduction retries")


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _trim_passages(prompt: str, max_chars: int) -> str:
    """Shrink the passages block in a prompt to max_chars, keeping header/footer."""
    marker = "Passages from academic literature:\n"
    idx = prompt.find(marker)
    if idx == -1:
        return prompt[:max_chars]
    header   = prompt[:idx + len(marker)]
    rest     = prompt[idx + len(marker):]
    sep_idx  = rest.find("\n---\n")
    passages = rest[:sep_idx] if sep_idx != -1 else rest
    footer   = rest[sep_idx:] if sep_idx != -1 else ""
    allowed  = max(500, max_chars - len(header) - len(footer))
    return header + passages[:allowed] + footer


def build_prompt(entity_name: str, entity_label: str,
                 entity_context: str, text_passages: str) -> str:
    label_hint = {
        "Person":      "a historical person or modern scholar",
        "Scholar":     "a modern scholar",
        "Place":       "a geographic location",
        "Institution": "an institution (library, university, archive)",
    }.get(entity_label, "an entity")

    relations_list = "\n".join(f"  - {r}" for r in sorted(ALLOWED_RELATIONS))

    return f"""\
Entity: "{entity_name}"
Type: {label_hint}
Known context: {entity_context or "none"}

Passages from academic literature:
{text_passages}

---
Extract ALL factual relationships involving "{entity_name}" supported by the passages.

Allowed relation types:
{relations_list}

Rules:
- subject/object must be proper names (people, places, institutions, fragments/shelf marks)
- evidence = direct quote or close paraphrase from the text
- confidence: "high" = direct statement, "medium" = clear implication
- omit "low" confidence triplets

Return ONLY this JSON (no markdown, no code fences):
{{
  "triplets": [
    {{
      "subject":      "...",
      "subject_type": "Person|Place|Institution|Fragment|Scholar",
      "relation":     "LIVED_IN",
      "object":       "...",
      "object_type":  "Person|Place|Institution|Fragment|BookArticle",
      "evidence":     "...",
      "confidence":   "high|medium"
    }}
  ]
}}

If nothing can be extracted: {{"triplets": []}}
"""


_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)


def parse_triplets(raw: str, entity_name: str, entity_label: str) -> List[Dict]:
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    match = _JSON_RE.search(raw)
    if not match:
        logger.warning("  No JSON in LLM response")
        return []
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as e:
        logger.warning(f"  JSON parse error: {e}")
        return []

    valid: List[Dict] = []
    for t in (data.get("triplets") or []):
        subj      = (t.get("subject")      or "").strip()
        relation  = (t.get("relation")     or "").strip().upper()
        obj       = (t.get("object")       or "").strip()
        conf      = (t.get("confidence")   or "medium").strip()
        evidence  = (t.get("evidence")     or "").strip()[:500]
        subj_type = (t.get("subject_type") or entity_label).strip()
        obj_type  = (t.get("object_type")  or "").strip()

        if not (subj and relation and obj):
            continue
        if relation not in ALLOWED_RELATIONS:
            logger.debug(f"  Unknown relation '{relation}' — skipped")
            continue
        if conf == "low":
            continue
        if not obj_type:
            _, hint = _REL_OBJECT_HINTS.get(relation, (None, None))
            obj_type = hint or "Entity"

        valid.append({
            "subject":      subj,
            "subject_type": subj_type,
            "relation":     relation,
            "object":       obj,
            "object_type":  obj_type,
            "evidence":     evidence,
            "confidence":   conf,
        })

    return valid


# ---------------------------------------------------------------------------
# Context builder — assembles page text from the current pipeline outputs
# (Pass 1 structured pages + Pass 3 book_entities_resolved.json).
# Legacy *_enhanced.json fallback requires ALLOW_LEGACY_ENHANCED=1.
# ---------------------------------------------------------------------------

def _find_book_dir(source_book: str) -> Optional[Path]:
    """Locate the book directory for a source-book stem.

    A book directory is the one whose name equals *source_book* and which
    contains a ``*_structured*`` subdirectory of Pass 1 page files.

    :param source_book: Book stem (e.g. ``"PosegayandArrantpdf"``).
    :returns: Book directory path, or None when not found.
    """
    for d in _data_root.rglob(source_book):
        if d.is_dir() and any("_structured" in c.name for c in d.iterdir() if c.is_dir()):
            return d
    # Case-insensitive fallback
    target = source_book.lower()
    for d in _data_root.rglob("*_structured*"):
        if d.is_dir() and d.parent.name.lower() == target:
            return d.parent
    return None


def _load_structured_pages(book_dir: Path) -> Dict[int, Dict]:
    """Load Pass 1 structured page JSONs for a book, keyed by page number.

    Handles both single-level (``book_structured_gemini/``) and two-level
    (``book_structured_qwen/qwen3_vl_8b/``) layouts.

    :param book_dir: Book root directory.
    :returns: Mapping of page number → structured page dict.
    """
    pages: Dict[int, Dict] = {}
    for p in sorted(book_dir.rglob("page_*_structured.json")):
        if "_structured" not in p.parent.name and "_structured" not in p.parent.parent.name:
            continue
        try:
            data = json.load(open(p, encoding="utf-8"))
        except Exception as e:
            logger.warning(f"  Could not load {p}: {e}")
            continue
        # Filename sequence index is the canonical page number pipeline-wide;
        # extracted_page_number is the printed page and is unreliable.
        m = re.search(r"page_(\d+)", p.name)
        if m:
            pages[int(m.group(1))] = data
    return pages


def _get_entity_pages_resolved(
    book_dir: Path, entity_name: str, entity_label: str
) -> List[int]:
    """Look up an entity's page numbers in ``book_entities_resolved*.json``.

    :param book_dir: Book root directory.
    :param entity_name: Canonical or alias entity name.
    :param entity_label: Neo4j label (Person/Scholar/Place/Institution).
    :returns: Sorted page numbers, empty when no resolved file or no match.
    """
    resolved_files = sorted(book_dir.glob("book_entities_resolved*.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
    if not resolved_files:
        return []
    try:
        data = json.load(open(resolved_files[0], encoding="utf-8"))
    except Exception as e:
        logger.warning(f"  Could not load {resolved_files[0]}: {e}")
        return []

    if entity_label in ("Person", "Scholar"):
        categories = ["people"]
    elif entity_label == "Place":
        categories = ["places"]
    elif entity_label == "Institution":
        categories = ["institutions"]
    else:
        categories = ["people", "places", "institutions"]

    name_lower = entity_name.lower()
    pages: List[int] = []
    for cat in categories:
        for item in (data.get(cat) or []):
            names = [(item.get("name") or "").lower()] + [
                (a or "").lower() for a in (item.get("aliases") or [])
            ]
            if name_lower in names:
                pages.extend(item.get("pages") or [])
    return sorted(set(pages))


def _collect_structured_page_texts(
    page_map: Dict[int, Dict],
    target_pages: List[int],
    max_chars_per_page: int = 800,
    max_total_chars:    int = 3000,
) -> str:
    """Assemble excerpts from structured pages for the given page numbers.

    :param page_map: Page number → structured page dict.
    :param target_pages: Pages to include.
    :param max_chars_per_page: Per-page text cap.
    :param max_total_chars: Overall cap across pages.
    :returns: Joined excerpt string.
    """
    chunks, total = [], 0
    for pg in target_pages:
        if total >= max_total_chars:
            break
        page = page_map.get(pg)
        if not page:
            continue
        summary = (page.get("summary") or "").strip()
        text    = (page.get("full_main_text") or "").strip()
        excerpt = f"[Page {pg}]\n"
        if summary:
            excerpt += f"Summary: {summary}\n"
        if text:
            excerpt += f"Text: {text[:max_chars_per_page]}\n"
        remaining = max_total_chars - total
        chunks.append(excerpt[:remaining])
        total += len(excerpt)
    return "\n---\n".join(chunks)


def _find_enhanced_json(source_book: str) -> Optional[Path]:
    """Locate the canonical *_enhanced.json for a source book.

    Files inside ``*_structured*/`` subdirectories are intermediate pipeline
    artifacts.  We always prefer the copy that sits directly in the book
    directory (i.e. whose parent directory name does NOT contain
    ``_structured``).

    :param source_book: Book stem without ``_enhanced.json`` suffix.
    :returns: Path to canonical enhanced JSON, or None if not found.
    """
    hits = [
        p for p in _data_root.glob(f"**/{source_book}_enhanced.json")
        if "_structured" not in p.parent.name
    ]
    if hits:
        return hits[0]
    # Fallback: case-insensitive match, still skipping structured subdirs
    for p in _data_root.rglob("*_enhanced.json"):
        if "_structured" not in p.parent.name and \
                p.stem.replace("_enhanced", "").lower() == source_book.lower():
            return p
    return None


def _get_entity_pages(data: Dict, entity_name: str, entity_label: str) -> List[int]:
    """Return page numbers where the entity is mentioned in this enhanced JSON."""
    pages: List[int] = []
    ca = data.get("context_analysis", {}) or {}

    if entity_label in ("Person", "Scholar"):
        lists = [ca.get("people_mentioned") or [],
                 (data.get("people_locations") or {}).get("people") or []]
    elif entity_label == "Place":
        lists = [ca.get("locations_mentioned") or [],
                 (data.get("people_locations") or {}).get("locations") or []]
    else:
        lists = [ca.get("people_mentioned") or [], ca.get("locations_mentioned") or []]

    name_lower = entity_name.lower()
    for lst in lists:
        for item in lst:
            if not isinstance(item, dict):
                continue
            names = [(item.get("name") or "").lower()] + [
                (v or "").lower() for v in (item.get("name_variants") or [])
            ]
            if name_lower in names:
                pages.extend(item.get("pages_mentioned") or [])

    return sorted(set(pages))


def _collect_page_texts(
    data: Dict,
    target_pages: List[int],
    max_chars_per_page: int = 800,
    max_total_chars:    int = 3000,
) -> str:
    page_map: Dict[int, Dict] = {}
    for page in (data.get("original_pages") or []):
        meta   = page.get("metadata") or {}
        pg_num = meta.get("page_number") or (
            (page.get("extracted_page_number") or [None])[0]
        )
        if pg_num is not None:
            page_map[pg_num] = page

    chunks, total = [], 0
    for pg in target_pages:
        if total >= max_total_chars:
            break
        page = page_map.get(pg)
        if not page:
            continue
        summary = (page.get("summary") or "").strip()
        text    = (page.get("full_main_text") or "").strip()
        excerpt = f"[Page {pg}]\nSummary: {summary}\n"
        if text:
            excerpt += f"Text: {text[:max_chars_per_page]}\n"
        remaining = max_total_chars - total
        chunks.append(excerpt[:remaining])
        total += len(excerpt)

    return "\n---\n".join(chunks)


def build_context_for_entity(
    entity_name: str,
    entity_label: str,
    source_books: List[str],
    total_char_budget: int = 4000,
) -> Tuple[str, List[str]]:
    """Gather focused page text across all source books for one entity.

    Reads the current pipeline outputs: Pass 1 structured page JSONs for
    text and ``book_entities_resolved*.json`` (Pass 3) for page attribution.
    Legacy ``*_enhanced.json`` files are only consulted when the env var
    ``ALLOW_LEGACY_ENHANCED=1`` is set — never silently.

    Budget is divided evenly so entities in many books (e.g. Maimonides)
    don't exceed the model's context window.
    """
    chunks: List[str] = []
    contributing: List[str] = []
    per_book = max(500, total_char_budget // max(len(source_books), 1))
    allow_legacy = os.getenv("ALLOW_LEGACY_ENHANCED", "") == "1"

    for book in source_books:
        book_dir = _find_book_dir(book)
        page_map = _load_structured_pages(book_dir) if book_dir else {}

        if not page_map:
            if allow_legacy:
                chunk = _legacy_enhanced_context(book, entity_name, entity_label, per_book)
                if chunk:
                    logger.warning(f"  Using LEGACY enhanced JSON for '{book}' "
                                   f"(ALLOW_LEGACY_ENHANCED=1)")
                    chunks.append(f"=== Source: {book} ===\n{chunk}")
                    contributing.append(book)
            else:
                logger.warning(
                    f"  No Pass 1 structured pages found for '{book}' — skipping. "
                    f"(Set ALLOW_LEGACY_ENHANCED=1 to permit legacy *_enhanced.json.)"
                )
            continue

        pages = _get_entity_pages_resolved(book_dir, entity_name, entity_label)

        # Fallback: scan full_main_text directly
        if not pages:
            name_lower = entity_name.lower()
            pages = sorted(
                pg for pg, page in page_map.items()
                if name_lower in (page.get("full_main_text") or "").lower()
            )

        if not pages:
            continue

        chunk = _collect_structured_page_texts(page_map, pages,
                                               max_chars_per_page=per_book // 2,
                                               max_total_chars=per_book)
        if chunk:
            chunks.append(f"=== Source: {book} ===\n{chunk}")
            contributing.append(book)

    return "\n\n".join(chunks), contributing


def _legacy_enhanced_context(
    book: str, entity_name: str, entity_label: str, per_book: int
) -> str:
    """Build context from a legacy *_enhanced.json file (opt-in only).

    :param book: Source book stem.
    :param entity_name: Entity name to locate.
    :param entity_label: Neo4j label.
    :param per_book: Character budget for this book.
    :returns: Excerpt string, empty when unavailable.
    """
    path = _find_enhanced_json(book)
    if not path:
        return ""
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        logger.warning(f"  Could not load {path}: {e}")
        return ""
    pages = _get_entity_pages(data, entity_name, entity_label)
    if not pages:
        name_lower = entity_name.lower()
        for page in (data.get("original_pages") or []):
            if name_lower in (page.get("full_main_text") or "").lower():
                meta   = page.get("metadata") or {}
                pg_num = meta.get("page_number") or (
                    (page.get("extracted_page_number") or [None])[0]
                )
                if pg_num:
                    pages.append(pg_num)
        pages = sorted(set(pages))
    if not pages:
        return ""
    return _collect_page_texts(data, pages,
                               max_chars_per_page=per_book // 2,
                               max_total_chars=per_book)


# ---------------------------------------------------------------------------
# Output file I/O
# ---------------------------------------------------------------------------

def _safe_filename(name: str) -> str:
    return re.sub(r'[^\w\- ]', '', name).strip().replace(' ', '_')[:80]


def output_path(label: str, entity_name: str, base_dir: Path) -> Path:
    label_dir = base_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    return label_dir / f"{_safe_filename(entity_name)}.json"


def write_enrichment_file(
    path: Path,
    entity_name: str,
    entity_label: str,
    source_books: List[str],
    contributing_books: List[str],
    triplets: List[Dict],
    model_used: str,
    context_chars: int,
    status: str = "ok",
) -> None:
    payload = {
        "entity_name":        entity_name,
        "entity_label":       entity_label,
        "source_books":       source_books,
        "contributing_books": contributing_books,
        "enriched_at":        datetime.now(timezone.utc).isoformat(),
        "model_used":         model_used,
        "context_chars":      context_chars,
        "status":             status,
        "triplets":           triplets,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
