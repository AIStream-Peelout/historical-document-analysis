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
from typing import Dict, List, Optional, Set, Tuple

import requests
import dotenv

_project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(_project_root))
dotenv.load_dotenv(_project_root / ".env")

from src.datasets.document_models import biblical_person_classifier as bib  # noqa: E402
from src.datasets.document_models.scholar_normalizer import ScholarRegistry  # noqa: E402
from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer  # noqa: E402
from src.models.llm.academic.entity_tagger import _is_leaked_entity  # noqa: E402

# Roles (from Pass 2 tagging) that mark a person as a modern, post-discovery
# figure → Scholar.  Includes manuscript dealers/collectors, who align with
# scholars rather than Genizah-era historical people.
_SCHOLAR_ROLE_HINTS = {
    "scholar", "author", "editor", "translator", "co-author", "coauthor",
    "collector", "dealer", "bookseller", "philologist", "orientalist",
    "librarian", "researcher", "historian", "academic", "professor",
}

logger = logging.getLogger(__name__)

_data_root   = _project_root / "src" / "datasets" / "raw_data" / "cairo_genizah" / "academic_literature"
_OUTPUT_FILE = "book_entities_resolved.json"

# ---------------------------------------------------------------------------
# Default LM Studio settings (cheap local model for name-list dedup)
# ---------------------------------------------------------------------------
_LMS_URL     = "http://localhost:1234/v1"
_LMS_MODEL   = "qwen/qwen3-4b-2507"
# A 50-name dedup batch on the 35B reasoning model that run_kg_overnight passes
# in generates several thousand reasoning tokens before any answer, which
# reliably exceeded the old 120s read timeout — the request was killed mid-flight
# and the batch silently dropped ("Dedup skipped"). Measured: large batches sat
# at exactly 120.0s before failing. 600s leaves room for the slowest batches.
_LMS_TIMEOUT = 600  # seconds


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
        if not isinstance(pg, int):
            # Older entity files stored the printed page (often a list or
            # null) — fall back to the filename sequence index, which is
            # the canonical page number for the pipeline.
            m = re.search(r"page_(\d+)", ef.name)
            if not m:
                continue
            pg = int(m.group(1))

        for p in (data.get("people") or []):
            name = (p.get("name") or "").strip()
            if not name or _is_leaked_entity(p, "name"):
                continue
            people[name]["pages"].add(pg)
            if p.get("context"):
                people[name]["contexts"].append(p["context"])
            if p.get("role"):
                people[name]["roles"].append(p["role"])

        for pl in (data.get("places") or []):
            name = (pl.get("name") or "").strip()
            if not name or _is_leaked_entity(pl, "name"):
                continue
            places[name]["pages"].add(pg)
            if pl.get("context"):
                places[name]["contexts"].append(pl["context"])
            if pl.get("type"):
                places[name]["types"].append(pl["type"])

        for inst in (data.get("institutions") or []):
            name = (inst.get("name") or "").strip()
            if not name or _is_leaked_entity(inst, "name"):
                continue
            institutions[name]["pages"].add(pg)
            if inst.get("context"):
                institutions[name]["contexts"].append(inst["context"])
            if inst.get("type"):
                institutions[name]["types"].append(inst["type"])

        for sm in (data.get("shelf_marks") or []):
            mark = (sm.get("mark") or "").strip()
            if not mark or _is_leaked_entity(sm, "mark"):
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
    # The tagger occasionally emits a non-string context/description (e.g. a
    # JSON boolean); normalise so downstream len()/text ops never break.
    if not isinstance(desc, str):
        desc = str(desc)

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
    # NOTE on response_format: this LM Studio build validates the field and
    # rejects {"type": "json_object"} outright with
    #   'response_format.type' must be 'json_schema' or 'text'
    # so sending it burned a guaranteed-failing round trip on EVERY call and
    # relied on the retry below. We send "text" and let the regex extraction in
    # _parse_dedup_response pull the JSON out, which it already did on the retry
    # path anyway.
    #
    # NOTE on max_tokens: reasoning models (qwen3.6-35b-a3b, which
    # run_kg_overnight passes in) spend the max_tokens budget on
    # `reasoning_content` BEFORE emitting any `content`. `/no_think` is in
    # _DEDUP_SYSTEM but this model ignores it and still reasons — measured
    # ~7k characters on a 50-name batch. If the budget runs out mid-reasoning
    # the response comes back 200 OK with finish_reason="length" and an EMPTY
    # content string, so the caller's `if not raw` treats it as a failed call
    # and silently drops the batch ("Dedup skipped"). Measured on a 50-name
    # batch: max_tokens=2048 -> truncated, content empty; 8192 -> stop, valid.
    # 16384 leaves headroom for larger/messier batches, and is affordable now
    # that the model is loaded at 65536 context.
    def _payload(max_tokens: int) -> dict:
        return {
            "model": lms_model,
            "messages": [
                {"role": "system", "content": _DEDUP_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens":  max_tokens,
            "response_format": {"type": "text"},
        }

    url = f"{lms_url.rstrip('/')}/chat/completions"
    max_tokens = 16384
    try:
        for attempt in range(3):
            r = requests.post(url, json=_payload(max_tokens), timeout=timeout)

            if r.status_code == 400:
                # Genuine context overflow (prompt + reservation > window).
                # Shrink the reservation rather than dropping the batch.
                if attempt < 2:
                    max_tokens //= 2
                    logger.warning(f"  400 from LM Studio ({(r.text or '')[:160]}); "
                                   f"retrying with max_tokens={max_tokens}")
                    continue
                r.raise_for_status()

            r.raise_for_status()
            choice  = r.json()["choices"][0]
            message = choice.get("message", {})
            content = (message.get("content") or "").strip()

            if content:
                return content

            # 200 OK but no content — reasoning ate the whole budget.
            reasoning = message.get("reasoning_content") or ""
            if choice.get("finish_reason") == "length" and attempt < 2:
                max_tokens *= 2
                logger.warning(
                    f"  Empty content (finish_reason=length, "
                    f"{len(reasoning)} reasoning chars) — the model spent the "
                    f"token budget reasoning. Retrying with max_tokens={max_tokens}."
                )
                continue

            # Deliberately NO fallback to reasoning_content here. A truncated
            # reasoning trace often contains a complete-looking DRAFT grouping
            # the model then retracted ("...wait, Abraham Maimonides is his
            # SON, a different person") — cut off before the correction, that
            # draft parses cleanly and silently merges two different real
            # people (reproduced in the 2026-08 audit). A dropped batch is
            # recoverable; a wrong merge is silent data corruption.
            logger.error(f"  LM Studio returned no usable content "
                         f"(finish_reason={choice.get('finish_reason')}, "
                         f"{len(reasoning)} reasoning chars)")
            return None
    except Exception as e:
        logger.error(f"  LM Studio call failed: {e}")
        return None
    return None


_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)


_CLASSIFY_SYSTEM = """\
/no_think
You are an expert in medieval Jewish history and the Hebrew Bible.

You are given person names that appear in academic literature about the Cairo
Genizah.  Each name matches a canonical biblical, Mishnaic, or Talmudic figure
BUT the same name was also commonly borne by ordinary medieval people.

For each name decide, using its description, whether it refers to the CANONICAL
ancient figure (the patriarch, prophet, sage, biblical character) or to a
historical medieval/early-modern individual who merely shares the name.

When in doubt, choose "historical" — do not over-assign "biblical".

Output ONLY valid JSON: {"biblical": ["<names that are the canonical figure>"]}
No markdown, no explanation.
"""


def _classify_ambiguous_llm(
    names_with_ctx: List[Tuple[str, str]],
    lms_url: str,
    lms_model: str,
    timeout: int,
) -> Set[str]:
    """Ask the LLM which ambiguous names refer to the canonical biblical figure.

    :param names_with_ctx: List of ``(name, description)`` tuples to classify.
    :param lms_url: LM Studio base URL.
    :param lms_model: Model name.
    :param timeout: HTTP timeout in seconds.
    :returns: Set of names judged to be the canonical biblical/ancient figure.
        Empty on any failure (conservative — defaults everything to historical).
    """
    if not names_with_ctx:
        return set()
    lines = "\n".join(
        f'- "{name}": {(desc or "no description").strip()[:200]}'
        for name, desc in names_with_ctx
    )
    prompt = f"Names to classify:\n{lines}\n\nReturn JSON: {{\"biblical\": [...]}}"
    payload = {
        "model": lms_model,
        "messages": [
            {"role": "system", "content": _CLASSIFY_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens":  1024,
        "response_format": {"type": "json_object"},
    }
    try:
        r = requests.post(f"{lms_url.rstrip('/')}/chat/completions", json=payload, timeout=timeout)
        if r.status_code == 400 and "response_format" in payload:
            del payload["response_format"]
            r = requests.post(f"{lms_url.rstrip('/')}/chat/completions", json=payload, timeout=timeout)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning(f"  Biblical classification call failed (defaulting historical): {e}")
        return set()

    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    match = _JSON_RE.search(raw)
    if not match:
        return set()
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return set()
    return {n.strip() for n in (data.get("biblical") or []) if isinstance(n, str)}


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


def _validate_dedup_groups(
    groups: List[Dict],
    batch_names: List[str],
) -> List[Dict]:
    """Drop hallucinated content from LLM dedup groups.

    The model's response is only trusted where it references names that were
    actually in the batch it was shown (2026-08 audit: nothing previously
    validated this, so a hallucinated canonical renamed real entities to a
    string never extracted from the book, and a hallucinated variant naming a
    DIFFERENT real entity silently destroyed it).

    Rules, per group:
    * canonical not in the batch → drop the whole group (a canonical the LLM
      invented would otherwise become the output entity name);
    * variants filtered to names in the batch, excluding the canonical itself;
    * no surviving variants → drop the group (it would be a no-op anyway, and
      empty groups are where fabricated aliases used to sneak in).

    :param groups: Parsed groups from :func:`_parse_dedup_response`.
    :param batch_names: The exact names sent to the LLM for this batch.
    :returns: Groups safe to feed to :func:`_apply_dedup`.
    """
    known = {n.strip() for n in batch_names}
    valid: List[Dict] = []
    for g in groups:
        canon = (g.get("canonical") or "").strip()
        if canon not in known:
            logger.warning(
                f"  Dropping dedup group with hallucinated canonical "
                f"{canon!r} (not in the batch shown to the model)")
            continue
        variants = [v.strip() for v in (g.get("variants") or [])
                    if v and v.strip() in known and v.strip() != canon]
        dropped = [v.strip() for v in (g.get("variants") or [])
                   if v and v.strip() and v.strip() not in known]
        if dropped:
            logger.warning(
                f"  Dropping hallucinated variant(s) {dropped!r} from "
                f"group {canon!r}")
        if not variants:
            continue
        valid.append({"canonical": canon, "variants": variants})
    return valid


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
        # extend, don't assign: groups accumulate across independent batches,
        # and a canonical can legitimately appear in more than one batch's
        # groups — assignment would silently discard the earlier batch's
        # variants.
        existing = canonical_variants.setdefault(canon, [])
        existing.extend(v for v in variants if v not in existing)
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
            # Prefer longer description (coerce: the tagger can emit a non-string
            # description, e.g. a JSON boolean, which has no len()).
            if len(str(entry.get("description") or "")) > len(str(existing.get("description") or "")):
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
        # Per-book count of dedup batches that failed and were skipped.
        # Reset at the top of resolve_book; drivers (run_kg_overnight) check
        # it after Pass 3 and refuse to write a success sentinel when > 0,
        # so silently-degraded books can't masquerade as complete.
        self.dedup_skipped_batches = 0
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
    # Reasoning cost per dedup call scales with batch size, and a 50-name batch
    # on the 35B blew past both the token budget and the read timeout. 25 keeps
    # each call comfortably inside both, at the cost of ~2x more (much shorter)
    # calls. Batches are independent, so smaller batches also mean a single
    # failure loses less work.
    _BATCH_SIZE = 25

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
                self.dedup_skipped_batches += 1
                continue
            groups = _parse_dedup_response(raw)
            # Only trust groups that reference names actually in this batch
            # (guards against hallucinated canonicals/variants merging or
            # renaming real entities — see _validate_dedup_groups).
            all_groups.extend(_validate_dedup_groups(groups, batch))

        if not all_groups:
            return resolved_list

        result = _apply_dedup(resolved_list, all_groups, name_key)
        before = len(resolved_list)
        after  = len(result)
        if before != after:
            logger.info(f"    {category_label}: {before} → {after} after dedup")
        return result

    def _classify_people(self, people_list: List[Dict]) -> List[Dict]:
        """Tag each resolved person with ``person_class``.

        Three classes, in priority order:

        * ``biblical``   — unambiguous canonical figure (Haman, Vespasian …),
          or an ambiguous canonical name the LLM confirms in context.
        * ``scholar``    — modern post-discovery figure: a scholar/author/
          editor/translator role, a manuscript dealer/collector, or a known
          ground-truth author.  Wins over the ambiguous-biblical guess so a
          modern scholar named "David" is never demoted to biblical.
          A known pre-modern historical author (Maimonides, Saadia Gaon,
          Yehuda ha-Levi, ...) overrides this — they were literally authors
          of classical works, which trips the same role-hint check that
          identifies modern scholars, but they must stay ``historical``.
        * ``historical`` — default: a Genizah-era individual.

        :param people_list: Resolved people dicts (mutated in place).
        :returns: The same list, each entry given a ``person_class`` field.
        """
        registry = ScholarRegistry.instance()
        ambiguous: List[Dict] = []
        for p in people_list:
            verdict = bib.lookup(p["name"])
            role = (p.get("role") or "").lower()
            is_scholar = (not bib.is_known_historical_author(p["name"])
                          and (any(h in role for h in _SCHOLAR_ROLE_HINTS)
                               or registry.is_known_scholar(p["name"])))

            if verdict == "biblical":
                p["person_class"] = "biblical"
            elif is_scholar:
                p["person_class"] = "scholar"
            elif verdict == "ambiguous":
                p["person_class"] = "historical"     # provisional — LLM may upgrade
                ambiguous.append(p)
            else:
                p["person_class"] = "historical"

        if ambiguous and self.use_llm:
            biblical_names = _classify_ambiguous_llm(
                [(p["name"], p.get("description", "")) for p in ambiguous],
                self.lms_url, self.lms_model, self.lms_timeout,
            )
            for p in ambiguous:
                if p["name"] in biblical_names:
                    p["person_class"] = "biblical"

        counts = {"biblical": 0, "scholar": 0, "historical": 0}
        for p in people_list:
            counts[p["person_class"]] += 1
        logger.info(f"    person_class: {counts['historical']} historical, "
                    f"{counts['scholar']} scholar, {counts['biblical']} biblical "
                    f"({len(ambiguous)} needed context)")
        return people_list

    def _filter_shelfmarks(self, marks_list: List[Dict]) -> List[Dict]:
        """Keep only genuine shelf-marks, tagging each with ``mark_class``.

        Uses :meth:`ShelfmarkNormalizer.classify_shelfmark` (shared with the
        merge pipeline) so the academic-book pipeline agrees with the merge
        layer on what a real shelf-mark is.  ``standard`` and ``berlin`` are
        kept; ``bibliographic`` (Davidson poem refs) and ``reject`` (prose /
        echoes / vague descriptors) are dropped.

        :param marks_list: Resolved shelf-mark dicts (each with a ``mark`` key).
        :returns: Filtered list, each kept entry given a ``mark_class`` field.
        """
        kept: List[Dict] = []
        dropped: Dict[str, int] = {}
        for m in marks_list:
            cls = ShelfmarkNormalizer.classify_shelfmark(m.get("mark", ""))
            if cls in ("standard", "berlin"):
                m["mark_class"] = cls
                kept.append(m)
            else:
                dropped[cls] = dropped.get(cls, 0) + 1
        if dropped:
            logger.info(f"    shelf_marks: kept {len(kept)}, "
                        f"dropped {sum(dropped.values())} ({dropped})")
        return kept

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
        self.dedup_skipped_batches = 0
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

        # ------------------------------------------------------------------
        # Phase 3 — tag biblical/canonical figures (person_class) so they
        # don't conflate with Genizah-era people or modern scholars.
        # ------------------------------------------------------------------
        people_list = self._classify_people(people_list)

        # Drop shelf-marks that aren't genuine shelf-marks (LLM reasoning prose,
        # prompt echoes, vague descriptors, Davidson poem citations) so junk
        # Fragment nodes never reach Pass 4 or the KG.
        marks_list = self._filter_shelfmarks(marks_list)

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
        # Tagged runs write entities_<run_tag>/, untagged runs entities/ —
        # the literal "entities/" pattern made the standalone CLI find 0
        # books for any tagged run (2026-08 audit).
        entity_dir = f"entities_{self.run_tag}" if self.run_tag else "entities"
        book_dirs = sorted(set(
            p.parent.parent
            for p in root.rglob(f"{entity_dir}/page_*_entities.json")
        ))

        counts = {"resolved": 0, "skipped": 0, "no_entities": 0}
        logger.info(f"Found {len(book_dirs)} book directories with entity files "
                    f"(looking in {entity_dir}/)")

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
