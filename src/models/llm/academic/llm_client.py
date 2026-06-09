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
_project_root = Path(__file__).parent.parent.parent.parent
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

    def complete(self, system: str, user: str, max_context_chars: int = 4000) -> str:
        if self.backend == BACKEND_GEMINI:
            resp = self._gemini.models.generate_content(
                model=self._gemini_model,
                contents=f"{system}\n\n{user}",
            )
            return resp.text
        return self._complete_lms_with_retry(system, user, max_context_chars)

    def _complete_lms_with_retry(self, system: str, user: str, max_context_chars: int) -> str:
        """On 400 (context too large), halve the passages block and retry up to 3x."""
        for attempt in range(3):
            trimmed_user = _trim_passages(user, max_context_chars >> attempt)
            payload = {
                "model": self.lms_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": trimmed_user},
                ],
                "temperature": 0.1,
                "max_tokens":  2048,
            }
            r = requests.post(f"{self.lms_url}/chat/completions",
                              json=payload, timeout=self.timeout)
            if r.status_code == 400 and attempt < 2:
                logger.warning(f"  400 on attempt {attempt+1} — halving context and retrying")
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        raise RuntimeError("LMS returned 400 after 3 retries")


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
# Context builder — assembles page text from _enhanced.json files
# ---------------------------------------------------------------------------

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

    Budget is divided evenly so entities in many books (e.g. Maimonides)
    don't exceed the model's context window.
    """
    chunks: List[str] = []
    contributing: List[str] = []
    per_book = max(500, total_char_budget // max(len(source_books), 1))

    for book in source_books:
        path = _find_enhanced_json(book)
        if not path:
            logger.debug(f"  No enhanced JSON for '{book}'")
            continue
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            logger.warning(f"  Could not load {path}: {e}")
            continue

        pages = _get_entity_pages(data, entity_name, entity_label)

        # Fallback: scan full_main_text directly
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
            continue

        chunk = _collect_page_texts(data, pages,
                                     max_chars_per_page=per_book // 2,
                                     max_total_chars=per_book)
        if chunk:
            chunks.append(f"=== Source: {book} ===\n{chunk}")
            contributing.append(book)

    return "\n\n".join(chunks), contributing


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
