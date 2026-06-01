"""
Unit tests for src/datasets/evaluations/metrics.py

Covers all public functions:
  strip_nikud, normalize_whitespace,
  cer_levenshtein, cer_pair,
  wer_levenshtein, wer_pair,
  char_count_ratio, flag_failure_modes
"""

import pytest
from src.datasets.evaluations.metrics import (
    strip_nikud,
    normalize_whitespace,
    cer_levenshtein,
    cer_pair,
    wer_levenshtein,
    wer_pair,
    char_count_ratio,
    flag_failure_modes,
)


# ── strip_nikud ───────────────────────────────────────────────────────────────

class TestStripNikud:
    def test_removes_vowel_points(self):
        # שָׁלוֹם with full nikud → שלום
        assert strip_nikud("שָׁלוֹם") == "שלום"

    def test_passthrough_plain_hebrew(self):
        text = "שלום עולם"
        assert strip_nikud(text) == text

    def test_empty_string(self):
        assert strip_nikud("") == ""

    def test_preserves_non_nikud_characters(self):
        # Letters, spaces, punctuation survive
        result = strip_nikud("בְּרֵאשִׁית בָּרָא")
        assert result == "בראשית ברא"

    def test_mixed_script_passthrough(self):
        # Arabic and Latin characters are untouched
        text = "hello مرحبا"
        assert strip_nikud(text) == text


# ── normalize_whitespace ──────────────────────────────────────────────────────

class TestNormalizeWhitespace:
    def test_collapses_multiple_spaces(self):
        assert normalize_whitespace("א  ב   ג") == "א ב ג"

    def test_strips_leading_trailing(self):
        assert normalize_whitespace("  שלום  ") == "שלום"

    def test_newlines_become_spaces(self):
        assert normalize_whitespace("א\nב\nג") == "א ב ג"

    def test_tabs_become_spaces(self):
        assert normalize_whitespace("א\tב") == "א ב"

    def test_empty_string(self):
        assert normalize_whitespace("") == ""

    def test_already_normalised(self):
        assert normalize_whitespace("א ב ג") == "א ב ג"


# ── cer_levenshtein ───────────────────────────────────────────────────────────

class TestCerLevenshtein:
    def test_identical_strings(self):
        assert cer_levenshtein("שלום", "שלום") == 0.0

    def test_empty_hypothesis(self):
        # All reference chars deleted → CER = 1.0
        assert cer_levenshtein("", "שלום") == 1.0

    def test_empty_reference(self):
        # Undefined by convention → return 1.0
        assert cer_levenshtein("שלום", "") == 1.0

    def test_both_empty(self):
        assert cer_levenshtein("", "") == 1.0

    def test_single_substitution(self):
        # "שלום" vs "שלוב" — one substitution out of 4 chars
        cer = cer_levenshtein("שלוב", "שלום")
        assert cer == pytest.approx(1 / 4)

    def test_decoder_collapse_exceeds_one(self):
        # Output far longer than reference → CER > 1 is valid and expected
        ref = "שלום"
        hyp = ref * 20
        cer = cer_levenshtein(hyp, ref)
        assert cer > 1.0, "CER must NOT be clamped at 1.0 for decoder-collapse detection"

    def test_whitespace_normalised_before_comparison(self):
        # Extra spaces should not inflate CER
        assert cer_levenshtein("  שלום  ", "שלום") == 0.0

    def test_partial_match(self):
        cer = cer_levenshtein("שלום עולם", "שלום")
        # hypothesis adds 5 extra chars
        assert cer > 0.0


# ── cer_pair ──────────────────────────────────────────────────────────────────

class TestCerPair:
    def test_returns_two_floats(self):
        strict, lenient = cer_pair("שָׁלוֹם", "שָׁלוֹם")
        assert isinstance(strict, float)
        assert isinstance(lenient, float)

    def test_strict_equals_zero_on_identical(self):
        strict, _ = cer_pair("שלום", "שלום")
        assert strict == 0.0

    def test_lenient_ignores_nikud(self):
        # Reference WITH nikud, hypothesis WITHOUT — lenient should be 0
        ref = "שָׁלוֹם"
        hyp = "שלום"
        strict, lenient = cer_pair(hyp, ref)
        assert lenient == 0.0
        assert strict > 0.0  # nikud chars count as errors in strict mode

    def test_lenient_le_strict(self):
        # Stripping nikud can only reduce or maintain the error rate
        ref = "בְּרֵאשִׁית בָּרָא אֱלֹהִים"
        hyp = "בראשית ברא אלהים"
        strict, lenient = cer_pair(hyp, ref)
        assert lenient <= strict


# ── wer_levenshtein ───────────────────────────────────────────────────────────

class TestWerLevenshtein:
    def test_identical(self):
        assert wer_levenshtein("שלום עולם", "שלום עולם") == 0.0

    def test_empty_hypothesis(self):
        assert wer_levenshtein("", "שלום עולם") == 1.0

    def test_empty_reference(self):
        assert wer_levenshtein("שלום", "") == 1.0

    def test_one_word_substituted(self):
        # 1 substitution out of 2 words
        wer = wer_levenshtein("שלום עולה", "שלום עולם")
        assert wer == pytest.approx(1 / 2)

    def test_extra_words_inflate_wer(self):
        # ref = 1 word; hyp = 4 completely different words
        # edit distance = 4 (1 substitution + 3 insertions), WER = 4/1 = 4.0
        ref = "שלום"
        hyp = "א ב ג ד"
        wer = wer_levenshtein(hyp, ref)
        assert wer > 1.0, "WER must NOT be clamped for insertion-heavy outputs"

    def test_operates_on_word_boundaries(self):
        # CER and WER should differ for a single-char vs single-word change
        ref = "שלום עולם"
        hyp = "שלום ירושלים"   # one word differs
        cer = cer_levenshtein(hyp, ref)
        wer = wer_levenshtein(hyp, ref)
        # WER = 1/2 = 0.5; CER depends on char diff, so they should differ
        assert wer != pytest.approx(cer)


# ── wer_pair ──────────────────────────────────────────────────────────────────

class TestWerPair:
    def test_lenient_ignores_nikud_in_words(self):
        ref = "שָׁלוֹם עוֹלָם"
        hyp = "שלום עולם"
        strict, lenient = wer_pair(hyp, ref)
        assert lenient == 0.0
        assert strict > 0.0


# ── char_count_ratio ──────────────────────────────────────────────────────────

class TestCharCountRatio:
    def test_identical_length(self):
        assert char_count_ratio("שלום", "שלום") == pytest.approx(1.0)

    def test_hypothesis_shorter(self):
        ratio = char_count_ratio("שלום", "שלום עולם")
        assert ratio < 1.0

    def test_hypothesis_longer(self):
        ratio = char_count_ratio("שלום עולם טוב", "שלום")
        assert ratio > 1.0

    def test_decoder_collapse_threshold(self):
        # Repetitive output far exceeding reference → ratio >> 2.5
        ref = "שלום"
        hyp = ref * 30
        assert char_count_ratio(hyp, ref) > 2.5

    def test_refusal_threshold(self):
        # Near-empty output against a long reference → ratio << 0.1
        ratio = char_count_ratio("?", "א" * 200)
        assert ratio < 0.1

    def test_empty_hypothesis(self):
        assert char_count_ratio("", "שלום") == pytest.approx(0.0)

    def test_empty_reference(self):
        # Empty reference with non-empty hypothesis → inf
        import math
        ratio = char_count_ratio("שלום", "")
        assert math.isinf(ratio)

    def test_both_empty(self):
        assert char_count_ratio("", "") == pytest.approx(1.0)


# ── flag_failure_modes ────────────────────────────────────────────────────────

class TestFlagFailureModes:
    def test_empty_hypothesis_is_refusal(self):
        flags = flag_failure_modes("", "שלום עולם")
        assert "refusal" in flags

    def test_whitespace_only_is_refusal(self):
        flags = flag_failure_modes("   ", "שלום עולם")
        assert "refusal" in flags

    def test_decoder_collapse_long_repetition(self):
        ref = "שלום"
        hyp = ref * 30
        flags = flag_failure_modes(hyp, ref)
        assert "decoder_collapse" in flags

    def test_no_flags_on_good_output(self):
        ref = "כל פרשה שהיתה חביבה על דוד"
        flags = flag_failure_modes(ref, ref)   # perfect match
        assert flags == []

    def test_script_misidentification_arabic_on_hebrew(self):
        # Hebrew GT, Arabic-script output
        ref = "שלום עולם " * 5
        hyp = "مرحبا بالعالم " * 5    # Arabic script
        flags = flag_failure_modes(hyp, ref)
        assert "script_misidentification" in flags

    def test_no_script_misidentification_for_judeo_arabic(self):
        # Judaeo-Arabic: Arabic TEXT in Hebrew SCRIPT — both ref and hyp Hebrew script
        ref = "כתב בלשון ערבי"
        hyp = "כתב בלשון ערבי"
        flags = flag_failure_modes(hyp, ref)
        assert "script_misidentification" not in flags

    def test_plausible_confabulation(self):
        # Use strings with zero character overlap so CER is guaranteed > 0.85.
        # Hebrew high-frequency letters (ה, א, מ) appear in most sentences and
        # reduce Levenshtein distance even between unrelated texts, so we use
        # a single repeated character for each string.
        # CER = 30/30 = 1.0 > 0.85; ratio = 1.0, within [0.5, 2.5]
        ref = "ה" * 30
        hyp = "מ" * 30
        flags = flag_failure_modes(hyp, ref)
        assert "plausible_confabulation" in flags

    def test_metadata_anchored_confabulation_flag(self):
        flags = flag_failure_modes("שלום", "שלום", catalog_metadata_provided=True)
        assert "metadata_anchored_confabulation" in flags

    def test_metadata_flag_absent_when_false(self):
        flags = flag_failure_modes("שלום", "שלום", catalog_metadata_provided=False)
        assert "metadata_anchored_confabulation" not in flags
