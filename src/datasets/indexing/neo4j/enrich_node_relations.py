#!/usr/bin/env python3
"""
Node Relation Enrichment — Pass 3
===================================

A targeted LLM pass that enriches KG nodes (Person, Place, Institution, or
any label) with relationships extracted specifically *for that entity* from the
academic-literature pages that reference it.

Two-stage workflow
------------------
Stage 1 — Extract (default)
  Queries Neo4j for candidate nodes, assembles focused page-text context for
  each, calls the LLM, and writes one JSON file per entity:

    enriched_relations/<Label>/Rachel_the_Byzantine.json

  Nothing is written to Neo4j yet.  You can review, edit, or delete files
  before committing anything.

Stage 2 — Import (--import-dir / --import-file)
  Reads every JSON file and writes the triplets into Neo4j (MERGE, safe to re-run).

Why a third pass?
-----------------
Pass 1 (structured_json_llm)      — page OCR -> structured JSON
Pass 2 (secondary_llm_processing) — book-level -> people_locations + general triplets
Pass 3 (this script)              — entity-anchored -> focused relationship extraction

The 400 Bad Request errors from LM Studio are a context-size issue: the model's
context window gets exceeded when an entity appears in many books.  The budget is
capped at 4 000 chars total, split evenly across books, and the LLM client retries
with halved context on a 400 before giving up.

Usage
-----
# Stage 1: extract for all Person nodes
python enrich_node_relations.py --label Person

# Stage 1: single entity
python enrich_node_relations.py --label Person --name "Rachel the Byzantine"

# Stage 1: Place nodes via Gemini
python enrich_node_relations.py --label Place --backend gemini

# Stage 1: dry-run
python enrich_node_relations.py --label Person --dry-run

# Stage 2: import a whole label directory
python enrich_node_relations.py --import-dir enriched_relations/Person

# Stage 2: import a single file
python enrich_node_relations.py --import-file enriched_relations/Person/Rachel_the_Byzantine.json

Environment
-----------
LM_STUDIO_URL   http://localhost:1234/v1   (default)
LM_STUDIO_MODEL qwen3:8b                   (default)
GEMINI_API_KEY  ...                        (required when --backend gemini)
NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD / NEO4J_DATABASE
"""

import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dotenv
import requests
from neo4j import GraphDatabase
from tqdm import tqdm

# ---------------------------------------------------------------------------
project_root = Path(__file__).parent.parent.parent.parent.parent
data_root    = project_root / "src" / "datasets" / "raw_data" / "cairo_genizah" / "academic_literature"
output_root  = project_root / "src" / "datasets" / "raw_data" / "cairo_genizah" / "enriched_relations"
sys.path.append(str(project_root))
dotenv.load_dotenv(project_root / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("enrich_node_relations.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

BACKEND_LMS    = "lm_studio"
BACKEND_GEMINI = "gemini"

# ---------------------------------------------------------------------------
# Allowed relation types
# ---------------------------------------------------------------------------
ALLOWED_RELATIONS = {
    "LIVED_IN", "TRAVELED_TO", "AFFILIATED_WITH", "ORIGINATED_FROM",
    "MENTIONED_IN", "MENTIONS_PERSON", "MENTIONS_PLACE",
    "CONTEMPORARY_OF", "STUDENT_OF", "TEACHER_OF",
    "RELATED_TO", "MARRIED_TO", "WROTE", "COMMISSIONED",
    "SENT_TO", "RECEIVED_FROM",
}

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
        gemini_model:   str = "gemini-2.0-flash",
        timeout:        int = 120,
    ):
        self.backend   = backend
        self.lms_url   = lms_url.rstrip("/")
        self.lms_model = lms_model
        self.timeout   = timeout
        self._gemini   = None

        if backend == BACKEND_GEMINI:
            try:
                import google.generativeai as genai
                api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY required for Gemini backend")
                genai.configure(api_key=api_key)
                self._gemini = genai.GenerativeModel(gemini_model)
                logger.info(f"Gemini client ready (model={gemini_model})")
            except ImportError:
                raise ImportError("pip install google-generativeai for Gemini support")
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
            resp = self._gemini.generate_content(f"{system}\n\n{user}")
            return resp.text
        return self._complete_lms_with_retry(system, user, max_context_chars)

    def _complete_lms_with_retry(self, system: str, user: str,
                                  max_context_chars: int) -> str:
        """Try the LMS call; on 400 halve the passages block and retry up to 3x."""
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
                logger.warning(f"  400 on attempt {attempt+1} "
                               f"({len(trimmed_user):,} chars) — halving context and retrying")
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        raise RuntimeError("LMS returned 400 after 3 retries")


# ---------------------------------------------------------------------------
# Prompt trimming helper
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


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _find_enhanced_json(source_book: str) -> Optional[Path]:
    hits = list(data_root.glob(f"**/{source_book}_enhanced.json"))
    if hits:
        return hits[0]
    for p in data_root.rglob("*_enhanced.json"):
        if p.stem.replace("_enhanced", "").lower() == source_book.lower():
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
        lists = [ca.get("people_mentioned") or [],
                 ca.get("locations_mentioned") or []]

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
    """Gather focused page text across all source books.

    Budget is divided evenly across books so entities referenced in many books
    (e.g. Maimonides in 9 books) don't exceed the model's context window.
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

        # Fallback: scan full_main_text for entity name
        if not pages:
            name_lower = entity_name.lower()
            for page in (data.get("original_pages") or []):
                if name_lower in (page.get("full_main_text") or "").lower():
                    meta = page.get("metadata") or {}
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
# Prompt + parser
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert in medieval Jewish history and the Cairo Genizah.
Extract factual relationships for a specific entity from academic literature excerpts.
The text may contain Hebrew, Aramaic, Judeo-Arabic, or other languages — treat all
languages equally and extract relationships from any language content you understand.
Only extract what is directly supported by the text. Do not infer or speculate.
Output ONLY valid JSON — no markdown, no explanation, no thinking tags.
"""


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
- subject/object must be proper names (people, places, institutions, fragments)
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
    # Strip Qwen3 thinking tags
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
# Output file helpers
# ---------------------------------------------------------------------------

def _safe_filename(name: str) -> str:
    return re.sub(r'[^\w\- ]', '', name).strip().replace(' ', '_')[:80]


def _output_path(label: str, entity_name: str, base_dir: Path) -> Path:
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


# ---------------------------------------------------------------------------
# Neo4j import helpers
# ---------------------------------------------------------------------------

_VALID_LABELS = {"Person", "Scholar", "Place", "Institution",
                 "Fragment", "BookArticle", "Entity"}

_SRC = (
    "{a}.source_books = CASE "
    "WHEN $book IN coalesce({a}.source_books, []) "
    "THEN coalesce({a}.source_books, []) "
    "ELSE coalesce({a}.source_books, []) + [$book] END"
)


def _src(alias: str) -> str:
    return _SRC.replace("{a}", alias)


def _write_triplet(tx, t: Dict, book: str) -> None:
    sl  = t["subject_type"] if t["subject_type"] in _VALID_LABELS else "Entity"
    ol  = t["object_type"]  if t["object_type"]  in _VALID_LABELS else "Entity"
    rel = t["relation"].replace(" ", "_")
    tx.run(f"""
        MERGE (a:{sl} {{name: $subj}})
        SET a.data_sources = CASE
            WHEN 'enriched' IN coalesce(a.data_sources, []) THEN coalesce(a.data_sources, [])
            ELSE coalesce(a.data_sources, []) + ['enriched'] END,
            {_src('a')}
        MERGE (b:{ol} {{name: $obj}})
        SET b.data_sources = CASE
            WHEN 'enriched' IN coalesce(b.data_sources, []) THEN coalesce(b.data_sources, [])
            ELSE coalesce(b.data_sources, []) + ['enriched'] END,
            {_src('b')}
        MERGE (a)-[r:{rel}]->(b)
          ON CREATE SET
            r.source       = $book,
            r.evidence     = $evidence,
            r.confidence   = $confidence,
            r.data_sources = ['enriched'],
            r.source_books = [$book]
          ON MATCH SET {_src('r')}
    """,
        subj=t["subject"], obj=t["object"],
        book=book, evidence=t["evidence"], confidence=t["confidence"],
    )


def _mark_enriched(tx, name: str, label: str, books: List[str]) -> None:
    tx.run(f"""
        MATCH (n:{label} {{name: $name}})
        SET n.relations_enriched  = true,
            n.enriched_from_books = $books
    """, name=name, books=books)


def import_enrichment_file(path: Path, driver, database: str) -> Dict[str, int]:
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)

    if rec.get("status") != "ok":
        logger.info(f"  Skipping {path.name} (status={rec.get('status')})")
        return {"skipped": 1}

    triplets     = rec.get("triplets") or []
    entity_name  = rec["entity_name"]
    entity_label = rec["entity_label"]
    books        = rec.get("contributing_books") or rec.get("source_books") or []

    if not triplets:
        logger.info(f"  {path.name}: 0 triplets")
        return {"no_triplets": 1}

    written = 0
    with driver.session(database=database) as s:
        for t in triplets:
            for book in books:
                try:
                    s.execute_write(_write_triplet, t, book)
                    written += 1
                except Exception as e:
                    logger.warning(f"  Write failed {t}: {e}")
        s.execute_write(_mark_enriched, entity_name, entity_label, books)

    logger.info(f"  {path.name}: wrote {written} edge(s)")
    return {"triplets_written": written, "files_imported": 1}


# ---------------------------------------------------------------------------
# Main enricher class
# ---------------------------------------------------------------------------

class NodeRelationEnricher:

    def __init__(
        self,
        neo4j_uri:        str,
        neo4j_user:       str,
        neo4j_password:   str,
        neo4j_database:   str = "neo4j",
        backend:          str = BACKEND_LMS,
        lms_url:          str = "http://localhost:1234/v1",
        lms_model:        str = "qwen3:8b",
        gemini_api_key:   Optional[str] = None,
        gemini_model:     str = "gemini-2.0-flash",
        min_source_books: int = 1,
        output_dir:       Optional[Path] = None,
    ):
        self.driver   = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.database = neo4j_database
        self.min_source_books = min_source_books
        self.output_dir = output_dir or output_root

        self.llm = LLMClient(
            backend=backend, lms_url=lms_url, lms_model=lms_model,
            gemini_api_key=gemini_api_key, gemini_model=gemini_model,
        )
        self._model_id = (
            f"{backend}:{lms_model if backend == BACKEND_LMS else gemini_model}"
        )
        logger.info(f"NodeRelationEnricher ready (backend={backend}, db={neo4j_database})")

    def close(self):
        self.driver.close()

    def _fetch_candidates(
        self,
        label: str,
        name_filter: Optional[str] = None,
        force: bool = False,
    ) -> List[Dict]:
        where = [
            "n.source_books IS NOT NULL",
            f"size(n.source_books) >= {self.min_source_books}",
        ]
        if not force:
            where.append("(n.relations_enriched IS NULL OR n.relations_enriched = false)")
        if name_filter:
            where.append("n.name = $name")

        q = f"""
            MATCH (n:{label})
            WHERE {' AND '.join(where)}
            RETURN n.name         AS name,
                   n.source_books AS source_books,
                   coalesce(n.context, n.description, '') AS context
            ORDER BY size(n.source_books) DESC
        """
        params = {"name": name_filter} if name_filter else {}
        with self.driver.session(database=self.database) as s:
            return [r.data() for r in s.run(q, **params)]

    # ------------------------------------------------------------------
    # Stage 1: extract -> JSON file
    # ------------------------------------------------------------------
    def extract_entity(
        self,
        name: str,
        label: str,
        source_books: List[str],
        entity_context: str = "",
        dry_run: bool = False,
        force: bool = False,
    ) -> Dict[str, int]:
        out_path = _output_path(label, name, self.output_dir)

        if out_path.exists() and not force:
            logger.info(f"  Already extracted: {out_path.name} — skipping")
            return {"already_exists": 1}

        context_text, contributing = build_context_for_entity(
            name, label, source_books
        )

        if not context_text.strip():
            logger.info(f"  No context found for '{name}'")
            if not dry_run:
                write_enrichment_file(out_path, name, label, source_books, [],
                                      [], self._model_id, 0, status="no_context")
            return {"no_context": 1}

        logger.info(f"  '{name}': {len(context_text):,} chars from "
                    f"{len(contributing)} book(s)")

        if dry_run:
            print(f"\n{'='*60}")
            print(f"DRY RUN [{label}] {name}")
            print(f"Books: {contributing}")
            print(f"Context ({len(context_text)} chars):\n{context_text[:500]}...")
            return {"dry_run": 1}

        prompt = build_prompt(name, label, entity_context, context_text)
        try:
            raw = self.llm.complete(_SYSTEM_PROMPT, prompt,
                                    max_context_chars=len(context_text))
        except Exception as e:
            logger.error(f"  LLM error for '{name}': {e}")
            write_enrichment_file(out_path, name, label, source_books, contributing,
                                  [], self._model_id, len(context_text),
                                  status="llm_error")
            return {"llm_error": 1}

        triplets = parse_triplets(raw, name, label)
        status   = "ok" if triplets else "no_triplets"
        write_enrichment_file(out_path, name, label, source_books, contributing,
                              triplets, self._model_id, len(context_text), status=status)

        logger.info(f"  '{name}': {len(triplets)} triplet(s) -> {out_path.name}")
        return {"triplets_extracted": len(triplets), "files_written": 1}

    def extract_all(
        self,
        label: str,
        name_filter: Optional[str] = None,
        force: bool = False,
        dry_run: bool = False,
        delay: float = 0.5,
    ) -> Dict[str, int]:
        candidates = self._fetch_candidates(label, name_filter, force)
        logger.info(f"Found {len(candidates)} {label} node(s) to extract")

        if not candidates:
            logger.info("Nothing to do.")
            return {}

        totals: Dict[str, int] = defaultdict(int)
        for node in tqdm(candidates, desc=f"Extracting {label}"):
            counts = self.extract_entity(
                name=node["name"],
                label=label,
                source_books=node.get("source_books") or [],
                entity_context=node.get("context") or "",
                dry_run=dry_run,
                force=force,
            )
            for k, v in counts.items():
                totals[k] += v
            if not dry_run and delay > 0:
                time.sleep(delay)

        self._print_summary(f"Extraction summary — {label}", totals, dry_run)
        return dict(totals)

    # ------------------------------------------------------------------
    # Stage 2: import JSON files -> Neo4j
    # ------------------------------------------------------------------
    def import_dir(self, directory: Path) -> Dict[str, int]:
        files = sorted(directory.rglob("*.json"))
        logger.info(f"Importing {len(files)} file(s) from {directory}")

        totals: Dict[str, int] = defaultdict(int)
        for path in tqdm(files, desc="Importing enrichment files"):
            try:
                counts = import_enrichment_file(path, self.driver, self.database)
                for k, v in counts.items():
                    totals[k] += v
            except Exception as e:
                logger.error(f"  Failed {path.name}: {e}")
                totals["import_errors"] += 1

        self._print_summary("Import summary", totals)
        return dict(totals)

    def import_file(self, path: Path) -> Dict[str, int]:
        logger.info(f"Importing {path}")
        counts = import_enrichment_file(path, self.driver, self.database)
        self._print_summary("Import summary", counts)
        return counts

    @staticmethod
    def _print_summary(title: str, totals: Dict[str, int], dry_run: bool = False):
        tag = "[DRY RUN] " if dry_run else ""
        print(f"\n{tag}{title}")
        print("=" * 50)
        for k, v in sorted(totals.items()):
            print(f"  {k}: {v:,}")
        print("=" * 50)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--import-dir",  metavar="DIR",
                      help="Stage 2: import all JSON files under DIR into Neo4j")
    mode.add_argument("--import-file", metavar="FILE",
                      help="Stage 2: import a single enrichment JSON file into Neo4j")

    parser.add_argument("--label",    "-l", default="Person",
                        help="Node label (Person, Place, Institution, ...)")
    parser.add_argument("--name",     "-n", default=None,
                        help="Enrich only this entity name")
    parser.add_argument("--backend",  choices=[BACKEND_LMS, BACKEND_GEMINI],
                        default=BACKEND_LMS)
    parser.add_argument("--model",    default=None,
                        help="Model name override")
    parser.add_argument("--lms-url",  default="http://localhost:1234/v1")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write JSON files (default: enriched_relations/)")
    parser.add_argument("--dry-run",  "-d", action="store_true",
                        help="Show context, skip LLM call and file writes")
    parser.add_argument("--force",    "-f", action="store_true",
                        help="Re-extract even entities already processed")
    parser.add_argument("--min-books", type=int, default=1,
                        help="Minimum source_books count to qualify")
    parser.add_argument("--delay",     type=float, default=0.5,
                        help="Seconds between LLM calls (default: 0.5)")

    args = parser.parse_args()

    uri      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    user     = os.getenv("NEO4J_USER",     "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "genizah-prod")

    if not password:
        logger.error("NEO4J_PASSWORD not set")
        sys.exit(1)

    model   = args.model or os.getenv("LM_STUDIO_MODEL", "qwen3:8b")
    out_dir = Path(args.output_dir) if args.output_dir else None

    enricher = NodeRelationEnricher(
        neo4j_uri=uri, neo4j_user=user, neo4j_password=password,
        neo4j_database=database,
        backend=args.backend, lms_url=args.lms_url, lms_model=model,
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        min_source_books=args.min_books,
        output_dir=out_dir,
    )

    try:
        if args.import_dir:
            enricher.import_dir(Path(args.import_dir))
        elif args.import_file:
            enricher.import_file(Path(args.import_file))
        else:
            enricher.extract_all(
                label=args.label,
                name_filter=args.name,
                force=args.force,
                dry_run=args.dry_run,
                delay=args.delay,
            )
    finally:
        enricher.close()


if __name__ == "__main__":
    main()
