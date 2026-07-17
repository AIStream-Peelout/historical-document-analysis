"""
error_classifier.py
LLM-judge error classification for transcription benchmark outputs.

CER/WER say HOW wrong a transcription is but not WHY.  This module has a
cheap text-only judge LLM (Gemini Flash by default, or any LM Studio model)
compare a model transcription against the scholarly ground truth and label
the error types, their severity, and a short qualitative example of each.

Per-document classifications are aggregated into benchmark-level statistics
(% of documents affected by each error type per model, mean severity, mean
occurrence count) suitable for ``wandb.summary`` — i.e. the quantitative
error analysis a reviewer expects alongside the raw CER numbers.

The judge sees text only (no image), so it can diagnose content-level error
types (confusions, omissions, canonical completion, hallucination, ...) but
not whether a divergence originates in the vision encoder — pair it with the
raw-OCR inspection columns for that.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Dict, List, Optional

from src.models.llm.transcription.genizah_fragment_agent import (
    AgentConfig,
    call_gemini_text_only,
)
from src.models.ocr.lms_transcriber import complete_text as lm_studio_complete_text


# ============================================================================
# Taxonomy
# ============================================================================

ERROR_TYPES: List[str] = [
    "character_confusion",   # visually similar letters swapped (ד/ר, ה/ח, ו/י)
    "nikud_error",           # vocalization/cantillation missing, added, or wrong
    "word_omission",         # words or whole lines missing
    "word_insertion",        # extra words not in the ground truth
    "line_order",            # right text, wrong reading order
    "canonical_completion",  # completed/"corrected" from a memorized canonical source
    "normalization",         # spelling normalized (plene/defective, abbreviations)
    "hallucination",         # invented content with no basis in the ground truth
    "truncation",            # output stops before the ground truth ends
    "script_confusion",      # wrong script or language
]

SEVERITY_SCORES: Dict[str, int] = {"minor": 1, "moderate": 2, "severe": 3}
QUALITY_SCORES: Dict[str, int] = {
    "unusable": 1, "poor": 2, "fair": 3, "good": 4, "excellent": 5,
}

#: Column layout for the per-document W&B error table.
TABLE_COLUMNS: List[str] = [
    "doc_id", "model", "error_type", "severity", "count", "example", "overall_quality",
]

_CLASSIFY_PROMPT = """\
You are evaluating a machine transcription of a historical Hebrew-script document
(Hebrew, Aramaic, or Judaeo-Arabic) against the scholarly ground-truth transcription.

GROUND TRUTH:
{gt}

MODEL TRANSCRIPTION:
{hyp}

Classify every error you find into these categories:
- character_confusion: visually similar letters swapped (e.g. ד/ר, ה/ח, ו/י, כ/ב, ם/ס)
- nikud_error: vocalization or cantillation marks missing, added, or wrong
- word_omission: words or whole lines present in the ground truth but missing
- word_insertion: extra words not present in the ground truth
- line_order: the right text but in the wrong reading order
- canonical_completion: the model completed or "corrected" the text from a memorized
  canonical source (e.g. standard Bible/Talmud text) instead of transcribing what is written
- normalization: spelling silently normalized (plene/defective changed, abbreviations
  expanded or contracted)
- hallucination: invented content with no basis in the ground truth
- truncation: the output stops before the ground truth ends
- script_confusion: output in the wrong script or language

Return ONLY a JSON object, no markdown fences, in exactly this shape:
{{"errors": [{{"type": "<category>", "severity": "minor|moderate|severe",
  "count": <approximate number of occurrences>,
  "example": "<short quote: ground truth fragment -> model fragment>"}}],
 "dominant_error": "<category with the largest impact, or 'none'>",
 "overall_quality": "excellent|good|fair|poor|unusable"}}

Only include categories that actually occur. If the transcription is essentially
perfect, return {{"errors": [], "dominant_error": "none", "overall_quality": "excellent"}}."""


# ============================================================================
# Judge call
# ============================================================================

def _parse_judge_json(raw: str) -> Optional[Dict]:
    """Extract the classification JSON object from a judge response.

    Tries the whole response, then fenced blocks, then the largest bare
    JSON object.

    :param raw: Raw judge output
    :type raw: str
    :return: Parsed dict, or None if no valid JSON was found
    :rtype: Optional[Dict]
    """
    if not raw:
        return None
    candidates = [raw.strip()]
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fence:
        candidates.append(fence.group(1))
    brace = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return None


def _validate_classification(data: Dict) -> Dict:
    """Coerce a raw judge response into the canonical classification shape.

    Drops errors with unknown types, coerces severities and counts, and
    normalizes the quality label.

    :param data: Parsed judge JSON
    :type data: Dict
    :return: Sanitized classification dict
    :rtype: Dict
    """
    errors = []
    for err in data.get("errors") or []:
        if not isinstance(err, dict):
            continue
        etype = str(err.get("type", "")).strip().lower()
        if etype not in ERROR_TYPES:
            continue
        severity = str(err.get("severity", "")).strip().lower()
        if severity not in SEVERITY_SCORES:
            severity = "moderate"
        try:
            count = max(1, int(err.get("count", 1)))
        except (TypeError, ValueError):
            count = 1
        errors.append({
            "type": etype,
            "severity": severity,
            "count": count,
            "example": str(err.get("example", ""))[:300],
        })

    dominant = str(data.get("dominant_error", "none")).strip().lower()
    if dominant not in ERROR_TYPES:
        dominant = "none"

    quality = str(data.get("overall_quality", "")).strip().lower()
    if quality not in QUALITY_SCORES:
        quality = "fair"

    return {"errors": errors, "dominant_error": dominant, "overall_quality": quality}


async def classify_transcription_errors(
    hypothesis: str,
    ground_truth: str,
    judge_model: str = AgentConfig.GEMINI_ANALYSIS_MODEL,
    judge_base_url: Optional[str] = None,
    max_chars: int = 6000,
) -> Optional[Dict]:
    """Classify the error types in one transcription using a judge LLM.

    :param hypothesis: Model transcription to diagnose
    :type hypothesis: str
    :param ground_truth: Scholarly ground-truth transcription
    :type ground_truth: str
    :param judge_model: Judge model name — a Gemini model (default Flash) or
        an LM Studio model ID when ``judge_base_url`` is set
    :type judge_model: str
    :param judge_base_url: If set, the judge is served via LM Studio at this URL
    :type judge_base_url: Optional[str]
    :param max_chars: Per-side truncation limit fed to the judge
    :type max_chars: int
    :return: ``{"errors": [...], "dominant_error": str, "overall_quality": str}``,
        or None on judge failure
    :rtype: Optional[Dict]
    """
    if not hypothesis.strip() or not ground_truth.strip():
        return None

    prompt = _CLASSIFY_PROMPT.format(
        gt=ground_truth[:max_chars],
        hyp=hypothesis[:max_chars],
    )
    if judge_base_url:
        raw = await lm_studio_complete_text(
            model_id=judge_model, prompt=prompt, base_url=judge_base_url
        )
    else:
        raw = await call_gemini_text_only(judge_model, prompt, timeout=90)

    data = _parse_judge_json(raw or "")
    if data is None:
        print(f"    ✗ [judge] Unparseable classification response "
              f"({len(raw or '')} chars)")
        return None
    return _validate_classification(data)


# ============================================================================
# Aggregation (for wandb.summary + tables)
# ============================================================================

def classification_table_rows(classified: List[Dict]) -> List[List]:
    """Flatten per-document classifications into W&B table rows.

    :param classified: List of dicts with ``doc_id``, ``model``, ``errors``,
        ``overall_quality`` keys (as produced by
        :func:`classify_transcription_errors` plus doc/model context)
    :type classified: List[Dict]
    :return: Rows matching :data:`TABLE_COLUMNS`
    :rtype: List[List]
    """
    rows: List[List] = []
    for c in classified:
        errors = c.get("errors") or []
        if not errors:
            rows.append([c["doc_id"], c["model"], "none", "", 0, "",
                         c.get("overall_quality", "")])
            continue
        for err in errors:
            rows.append([
                c["doc_id"], c["model"],
                err["type"], err["severity"], err["count"], err["example"],
                c.get("overall_quality", ""),
            ])
    return rows


def aggregate_error_stats(classified: List[Dict]) -> Dict[str, float]:
    """Aggregate per-document classifications into benchmark statistics.

    Produces, per model and error type, the fraction of classified documents
    affected (``pct_docs``), the mean severity (1=minor .. 3=severe), and the
    mean occurrence count — plus a per-model quality score (1=unusable ..
    5=excellent).  Keys are namespaced for ``wandb.summary``.

    :param classified: Per-document classifications with ``doc_id`` and
        ``model`` context
    :type classified: List[Dict]
    :return: Flat ``{summary_key: value}`` dict
    :rtype: Dict[str, float]
    """
    docs_per_model: Dict[str, set] = defaultdict(set)
    type_docs: Dict[tuple, set] = defaultdict(set)
    type_severities: Dict[tuple, List[int]] = defaultdict(list)
    type_counts: Dict[tuple, List[int]] = defaultdict(list)
    quality_scores: Dict[str, List[int]] = defaultdict(list)

    for c in classified:
        model, doc = c["model"], c["doc_id"]
        docs_per_model[model].add(doc)
        q = QUALITY_SCORES.get(c.get("overall_quality", ""))
        if q is not None:
            quality_scores[model].append(q)
        for err in c.get("errors") or []:
            key = (model, err["type"])
            type_docs[key].add(doc)
            type_severities[key].append(SEVERITY_SCORES[err["severity"]])
            type_counts[key].append(err["count"])

    out: Dict[str, float] = {}
    for (model, etype), docs in type_docs.items():
        n = len(docs_per_model[model])
        prefix = f"error_types/{model}/{etype}"
        out[f"{prefix}/pct_docs"] = len(docs) / n
        out[f"{prefix}/mean_severity"] = (
            sum(type_severities[(model, etype)]) / len(type_severities[(model, etype)])
        )
        out[f"{prefix}/mean_count"] = (
            sum(type_counts[(model, etype)]) / len(type_counts[(model, etype)])
        )
    for model, docs in docs_per_model.items():
        out[f"error_types/{model}/docs_classified"] = len(docs)
        if quality_scores[model]:
            out[f"error_types/{model}/quality_score"] = (
                sum(quality_scores[model]) / len(quality_scores[model])
            )
    return out
