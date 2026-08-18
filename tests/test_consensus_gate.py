"""Unit tests for the consensus confidence gate (pure logic, no services)."""

import random

import pytest

from src.datasets.consensus.consensus_gate import (
    AGREE_ACCEPT,
    AGREE_HIGH,
    TIER_ESCALATE,
    TIER_HIGH,
    TIER_STANDARD,
    build_fragment_prompt,
    evaluate_pair,
)

_HEBREW = "אבגדהוזחטיכלמנסעפצקרשת"


def _hebrew_text(seed: int, n_chars: int = 400) -> str:
    """Deterministic pseudo-Hebrew text with word structure.

    :param seed: RNG seed.
    :type seed: int
    :param n_chars: Approximate letter count.
    :type n_chars: int
    :return: Space-separated pseudo-Hebrew words.
    :rtype: str
    """
    rng = random.Random(seed)
    words = []
    total = 0
    while total < n_chars:
        w = "".join(rng.choice(_HEBREW) for _ in range(rng.randint(3, 7)))
        words.append(w)
        total += len(w)
    return " ".join(words)


def test_identical_texts_reach_high_tier():
    """Two engines writing the same text is maximal cross-family evidence."""
    text = _hebrew_text(1)
    res = evaluate_pair(text, text)
    assert res.tier == TIER_HIGH
    assert res.agreement > 0.95


def test_disjoint_texts_escalate():
    """No shared n-grams means no consensus - never auto-accept."""
    res = evaluate_pair(_hebrew_text(2), _hebrew_text(3))
    assert res.tier == TIER_ESCALATE
    assert res.reason == "agreement"
    assert res.agreement < AGREE_ACCEPT


def test_loop_on_genuine_text_escalates():
    """Regression for T_S_13J8_6: a VLM loop over REAL page phrases must be
    caught by the GT-free loop screen even though n-gram agreement is high."""
    page = _hebrew_text(4, 300)
    phrase = page[:15]
    looped = page + (" " + phrase) * 120
    res = evaluate_pair(looped, page)
    assert res.vlm_loop_ratio > 0.45
    assert res.tier == TIER_ESCALATE
    assert res.reason == "loop"
    assert res.agreement >= AGREE_ACCEPT  # the screen, not agreement, caught it


def test_short_output_is_abstention():
    """Sub-25-letter output cannot be vouched for."""
    res = evaluate_pair("אבג", _hebrew_text(5))
    assert res.tier == TIER_ESCALATE
    assert res.reason == "abstain"


def test_engine_failures_escalate():
    """Empty engine output routes to escalation with the failing side named."""
    text = _hebrew_text(6)
    assert evaluate_pair(text, "").reason == "htr_failed"
    assert evaluate_pair("", text).reason == "vlm_failed"
    assert evaluate_pair("", "").tier == TIER_ESCALATE


def test_tier_matches_thresholds():
    """Tier assignment is consistent with the calibrated cutoffs."""
    shared = _hebrew_text(7, 200)
    partial = shared + " " + _hebrew_text(8, 500)
    res = evaluate_pair(partial, shared)
    if res.agreement >= AGREE_HIGH:
        assert res.tier == TIER_HIGH
    elif res.agreement >= AGREE_ACCEPT:
        assert res.tier == TIER_STANDARD
    else:
        assert res.tier == TIER_ESCALATE
    assert 0.0 < res.agreement < 1.0


def test_prompt_matches_benchmark_wording():
    """The rollout prompt must stay byte-identical to the calibrated one."""
    prompt = build_fragment_prompt("Cambridge_CUL_T_S_8_4")
    assert prompt.startswith(
        "This is a manuscript from the Cairo Genizah "
        "(Cambridge_CUL_T_S_8_4 collection)."
    )
    assert "Transcribe the text in this image exactly as written." in prompt
    assert prompt.endswith("Return ONLY the transcription with no commentary.")
    # Space-bearing shelfmarks still resolve to their series.
    assert "(T-S collection)" in build_fragment_prompt("T-S 8.4")
    assert "(T-S NS collection)" in build_fragment_prompt("T-S NS J295")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
