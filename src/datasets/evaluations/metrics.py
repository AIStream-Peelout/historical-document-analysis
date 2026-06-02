"""
Shared evaluation metrics for the Cairo Genizah / Talmud benchmark.

All CER/WER values use Levenshtein edit distance — the HTR community standard
(OCRBench, MiDRASH paper conventions).  CER is NOT clamped at 1.0: values > 1
document decoder collapse and are first-class benchmark findings.
"""

import re
from typing import List, Tuple

# ── Hebrew Unicode ranges ─────────────────────────────────────────────────────
# Nikud (vowel points) + cantillation: U+05B0–U+05C7, U+05F0–U+05F4,
# U+FB1D–U+FB4F (presentation forms)
_NIKUD_RE = re.compile(r"[ְ-ׇװ-״יִ-ﭏ]")
_ARABIC_SCRIPT_RE = re.compile(r"[؀-ۿ]")
_HEBREW_SCRIPT_RE = re.compile(r"[֐-׿יִ-פֿ]")


# ── Text normalisation ────────────────────────────────────────────────────────

def strip_nikud(text: str) -> str:
    """Remove Hebrew vowel points and cantillation marks for lenient CER."""
    return _NIKUD_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace to single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


# ── Edit distance ─────────────────────────────────────────────────────────────

def _levenshtein(a, b) -> int:
    """Standard Levenshtein distance — works on any sequence (str or list)."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for ca in a:
        curr = [prev[0] + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


# ── Character Error Rate ──────────────────────────────────────────────────────

def cer_levenshtein(hypothesis: str, reference: str) -> float:
    """CER = edit_distance(hyp, ref) / len(ref).

    NOT clamped at 1.0.  CER > 1 is a valid finding (decoder collapse,
    confabulation) and will appear in published tables.
    Empty reference → 1.0.
    """
    ref = normalize_whitespace(reference)
    hyp = normalize_whitespace(hypothesis)
    if not ref:
        return 1.0
    return _levenshtein(hyp, ref) / len(ref)


def cer_pair(hypothesis: str, reference: str) -> Tuple[float, float]:
    """Return (cer_strict, cer_lenient) — with and without nikud."""
    return (
        cer_levenshtein(hypothesis, reference),
        cer_levenshtein(strip_nikud(hypothesis), strip_nikud(reference)),
    )


# ── Word Error Rate ───────────────────────────────────────────────────────────

def wer_levenshtein(hypothesis: str, reference: str) -> float:
    """WER = word_edit_distance(hyp, ref) / num_words(ref).

    Word-level Levenshtein (each 'character' is a whitespace-delimited token).
    NOT clamped at 1.0.
    """
    ref_words = normalize_whitespace(reference).split()
    hyp_words = normalize_whitespace(hypothesis).split()
    if not ref_words:
        return 1.0
    return _levenshtein(hyp_words, ref_words) / len(ref_words)


def wer_pair(hypothesis: str, reference: str) -> Tuple[float, float]:
    """Return (wer_strict, wer_lenient) — with and without nikud."""
    return (
        wer_levenshtein(hypothesis, reference),
        wer_levenshtein(strip_nikud(hypothesis), strip_nikud(reference)),
    )


# ── Diagnostic ratio ─────────────────────────────────────────────────────────

def char_count_ratio(hypothesis: str, reference: str) -> float:
    """len(hypothesis) / len(reference).

    > 2.5  → likely decoder collapse (repetitive output)
    < 0.1  → likely refusal or near-empty output
    ~1.0   → output is plausible length
    """
    ref_len = len(reference)
    if ref_len == 0:
        return float("inf") if hypothesis else 1.0
    return len(hypothesis) / ref_len


# ── Failure mode detection ────────────────────────────────────────────────────

def flag_failure_modes(
    hypothesis: str,
    reference: str,
    catalog_metadata_provided: bool = False,
) -> List[str]:
    """Return failure-mode labels for one model output.

    Labels match the benchmark taxonomy defined in the EACL spec:
      refusal, decoder_collapse, plausible_confabulation,
      script_misidentification, metadata_anchored_confabulation
    """
    if not hypothesis or not hypothesis.strip():
        return ["refusal"]

    flags: List[str] = []
    ratio = char_count_ratio(hypothesis, reference)

    if ratio > 2.5:
        flags.append("decoder_collapse")

    if ratio < 0.1 and len(reference) > 50:
        flags.append("refusal")

    # Script misidentification: Arabic-script output on a Hebrew-script reference
    arabic_n = len(_ARABIC_SCRIPT_RE.findall(hypothesis))
    hebrew_n = len(_HEBREW_SCRIPT_RE.findall(hypothesis))
    ref_heb_n = len(_HEBREW_SCRIPT_RE.findall(reference))
    if ref_heb_n > 20 and arabic_n > max(hebrew_n, 1) * 2:
        flags.append("script_misidentification")

    # Plausible confabulation: plausible length, wrong content
    cer = cer_levenshtein(hypothesis, reference)
    if 0.5 <= ratio <= 2.5 and cer > 0.85:
        flags.append("plausible_confabulation")

    if catalog_metadata_provided:
        flags.append("metadata_anchored_confabulation")

    return flags
