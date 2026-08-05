"""Regression tests for the Pass-3 dedup integrity fixes (2026-08 audit).

Covers the four confirmed failure modes:
1. Hallucinated canonical names renaming real entities (`_validate_dedup_groups`).
2. Hallucinated variants destroying a different real entity.
3. The removed reasoning_content fallback in `_call_lms` (a truncated
   reasoning trace containing a retracted draft grouping must NOT be used).
4. Cross-batch `canonical_variants` accumulation (extend, not overwrite).
"""

import json
from unittest import mock

import pytest

from src.models.llm.academic.coreference_resolver import (
    _apply_dedup,
    _call_lms,
    _validate_dedup_groups,
)


# ── _validate_dedup_groups ────────────────────────────────────────────────────

BATCH = ["Moses Maimonides", "Abraham Maimonides", "Rambam", "Halfon b. Nethanel"]


def test_valid_group_passes():
    groups = [{"canonical": "Moses Maimonides", "variants": ["Rambam"]}]
    assert _validate_dedup_groups(groups, BATCH) == groups


def test_hallucinated_canonical_drops_whole_group():
    # A canonical the LLM invented must not become an output entity name.
    groups = [{"canonical": "Rabbi Moshe ben Maimon (RaMBaM)",
               "variants": ["Moses Maimonides", "Rambam"]}]
    assert _validate_dedup_groups(groups, BATCH) == []


def test_hallucinated_variant_is_filtered_but_group_survives():
    groups = [{"canonical": "Moses Maimonides",
               "variants": ["Rambam", "Musa ibn Maymun al-Qurtubi"]}]
    out = _validate_dedup_groups(groups, BATCH)
    assert out == [{"canonical": "Moses Maimonides", "variants": ["Rambam"]}]


def test_group_with_no_surviving_variants_dropped():
    # All-hallucinated variants -> group is a no-op and used to fabricate
    # aliases; must be dropped entirely.
    groups = [{"canonical": "Moses Maimonides", "variants": ["Invented Name"]}]
    assert _validate_dedup_groups(groups, BATCH) == []


def test_canonical_listed_as_own_variant_excluded():
    groups = [{"canonical": "Rambam", "variants": ["Rambam", "Moses Maimonides"]}]
    out = _validate_dedup_groups(groups, BATCH)
    assert out == [{"canonical": "Rambam", "variants": ["Moses Maimonides"]}]


def test_validation_is_per_batch():
    # A name from ANOTHER batch is hallucination from this batch's viewpoint.
    other_batch = ["Halfon b. Nethanel", "Halfon ha-Levi"]
    groups = [{"canonical": "Halfon b. Nethanel", "variants": ["Rambam"]}]
    assert _validate_dedup_groups(groups, other_batch) == []


# ── _apply_dedup cross-batch accumulation ─────────────────────────────────────

def _entity(name, pages):
    return {"name": name, "pages": pages, "aliases": [], "description": ""}


def test_same_canonical_across_batches_extends_variants():
    entities = [_entity("Moses Maimonides", [1]),
                _entity("Rambam", [2]),
                _entity("Musa b. Maymun", [3])]
    # Two groups for the same canonical, as produced by two independent batches.
    groups = [{"canonical": "Moses Maimonides", "variants": ["Rambam"]},
              {"canonical": "Moses Maimonides", "variants": ["Musa b. Maymun"]}]
    out = _apply_dedup(entities, groups, "name")
    assert len(out) == 1
    merged = out[0]
    assert merged["name"] == "Moses Maimonides"
    assert merged["pages"] == [1, 2, 3]
    assert set(merged["aliases"]) == {"Rambam", "Musa b. Maymun"}


# ── _call_lms: no reasoning_content recovery ──────────────────────────────────

POISON_REASONING = (
    'Let me draft the grouping first and then verify each entry... '
    '{"groups": [{"canonical": "Moses Maimonides", '
    '"variants": ["Rambam", "Abraham Maimonides"]}]} '
    '... wait, Abraham Maimonides is his SON, a different person. I must remove'
)


def _fake_response(status=200, content="", reasoning="", finish="stop"):
    r = mock.Mock()
    r.status_code = status
    r.text = ""
    r.json.return_value = {
        "choices": [{
            "finish_reason": finish,
            "message": {"content": content, "reasoning_content": reasoning},
        }]
    }
    r.raise_for_status.return_value = None
    return r


def test_empty_content_with_draft_reasoning_returns_none():
    """The retracted-draft trace must be discarded, not parsed as an answer."""
    with mock.patch("src.models.llm.academic.coreference_resolver.requests.post",
                    return_value=_fake_response(reasoning=POISON_REASONING,
                                                finish="stop")):
        assert _call_lms("prompt", "http://x/v1", "m", 5) is None


def test_length_exhaustion_on_all_attempts_returns_none():
    """finish_reason=length on every retry must give up cleanly (no fallback)."""
    with mock.patch("src.models.llm.academic.coreference_resolver.requests.post",
                    return_value=_fake_response(reasoning=POISON_REASONING,
                                                finish="length")) as post:
        assert _call_lms("prompt", "http://x/v1", "m", 5) is None
        # doubling retry means multiple attempts were made
        assert post.call_count >= 2


def test_real_content_still_returned():
    body = json.dumps({"groups": [{"canonical": "A", "variants": ["B"]}]})
    with mock.patch("src.models.llm.academic.coreference_resolver.requests.post",
                    return_value=_fake_response(content=body)):
        assert _call_lms("prompt", "http://x/v1", "m", 5) == body
