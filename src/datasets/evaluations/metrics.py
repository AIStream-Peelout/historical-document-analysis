"""
Shared evaluation metrics for the Cairo Genizah / Talmud benchmark.

All CER/WER values use Levenshtein edit distance — the HTR community standard
(OCRBench, MiDRASH paper conventions).  CER is NOT clamped at 1.0: values > 1
document decoder collapse and are first-class benchmark findings.
"""

import re
import unicodedata
from typing import List, Tuple

try:
    # C-accelerated (rapidfuzz-backed) implementation — ~1000x faster than
    # the pure-Python fallback on page-length texts.  Accepts str or lists.
    from Levenshtein import distance as _c_levenshtein
except ImportError:  # pragma: no cover
    _c_levenshtein = None

# ── Hebrew Unicode ranges ─────────────────────────────────────────────────────
# Letters only (base alphabet U+05D0–U+05EA, ligatures U+05F0–U+05F2,
# presentation forms U+FB1D–U+FB4F) — used for script counting; combining
# marks are deliberately excluded so mark-heavy text doesn't inflate counts.
_ARABIC_SCRIPT_RE = re.compile(r"[؀-ۿ]")
_HEBREW_SCRIPT_RE = re.compile(r"[א-תװ-ײיִ-ﭏ]")


# ── Text normalisation ────────────────────────────────────────────────────────

def strip_nikud(text: str) -> str:
    """Remove Hebrew vowel points and cantillation marks for lenient CER.

    Decomposes precomposed/presentation-form characters first (e.g. the
    single-codepoint yud-hiriq U+FB1D or the alef-lamed ligature U+FB4F) so
    their base letters are preserved, then drops all combining marks —
    nikud (U+05B0–U+05C7) and cantillation (U+0591–U+05AF) alike.

    :param text: Input text
    :type text: str
    :return: Text with combining marks removed, base letters intact
    :rtype: str
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace to single space and strip.

    :param text: Input text
    :type text: str
    :return: Whitespace-normalized text
    :rtype: str
    """
    return re.sub(r"\s+", " ", text).strip()


# ── Edit distance ─────────────────────────────────────────────────────────────

def _levenshtein(a, b) -> int:
    """Levenshtein distance — works on any sequence (str or list of tokens).

    Uses the C-accelerated ``python-Levenshtein`` implementation when
    installed; falls back to pure Python otherwise.

    :param a: First sequence
    :param b: Second sequence
    :return: Edit distance between the sequences
    :rtype: int
    """
    if _c_levenshtein is not None:
        return _c_levenshtein(a, b)
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

    Both sides are whitespace- and NFC-normalized so composed vs decomposed
    encodings of the same nikud'd letter don't count as edits.

    Empty reference: 0.0 if the hypothesis is also empty (correctly
    predicting nothing is not an error), else 1.0.

    :param hypothesis: Model output
    :type hypothesis: str
    :param reference: Ground-truth text
    :type reference: str
    :return: Character error rate
    :rtype: float
    """
    ref = unicodedata.normalize("NFC", normalize_whitespace(reference))
    hyp = unicodedata.normalize("NFC", normalize_whitespace(hypothesis))
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(hyp, ref) / len(ref)


def cer_pair(hypothesis: str, reference: str) -> Tuple[float, float]:
    """Return (cer_strict, cer_lenient) — with and without nikud.

    :param hypothesis: Model output
    :type hypothesis: str
    :param reference: Ground-truth text
    :type reference: str
    :return: Tuple of (strict CER, lenient CER)
    :rtype: Tuple[float, float]
    """
    return (
        cer_levenshtein(hypothesis, reference),
        cer_levenshtein(strip_nikud(hypothesis), strip_nikud(reference)),
    )


# ── Word Error Rate ───────────────────────────────────────────────────────────

def wer_levenshtein(hypothesis: str, reference: str) -> float:
    """WER = word_edit_distance(hyp, ref) / num_words(ref).

    Word-level Levenshtein (each 'character' is a whitespace-delimited token).
    NOT clamped at 1.0.  Empty reference: 0.0 if the hypothesis is also
    empty, else 1.0.

    :param hypothesis: Model output
    :type hypothesis: str
    :param reference: Ground-truth text
    :type reference: str
    :return: Word error rate
    :rtype: float
    """
    ref_words = unicodedata.normalize("NFC", normalize_whitespace(reference)).split()
    hyp_words = unicodedata.normalize("NFC", normalize_whitespace(hypothesis)).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _levenshtein(hyp_words, ref_words) / len(ref_words)


def wer_pair(hypothesis: str, reference: str) -> Tuple[float, float]:
    """Return (wer_strict, wer_lenient) — with and without nikud.

    :param hypothesis: Model output
    :type hypothesis: str
    :param reference: Ground-truth text
    :type reference: str
    :return: Tuple of (strict WER, lenient WER)
    :rtype: Tuple[float, float]
    """
    return (
        wer_levenshtein(hypothesis, reference),
        wer_levenshtein(strip_nikud(hypothesis), strip_nikud(reference)),
    )


# ── Diagnostic ratio ─────────────────────────────────────────────────────────

def char_count_ratio(hypothesis: str, reference: str) -> float:
    """len(hypothesis) / len(reference), on whitespace-normalized text
    (consistent with how CER is computed).

    > 2.5  → likely decoder collapse (repetitive output)
    < 0.1  → likely refusal or near-empty output
    ~1.0   → output is plausible length

    :param hypothesis: Model output
    :type hypothesis: str
    :param reference: Ground-truth text
    :type reference: str
    :return: Length ratio (inf when reference is empty but hypothesis is not)
    :rtype: float
    """
    ref_len = len(normalize_whitespace(reference))
    hyp_len = len(normalize_whitespace(hypothesis))
    if ref_len == 0:
        return float("inf") if hyp_len else 1.0
    return hyp_len / ref_len


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

    ``metadata_anchored_confabulation`` is only flagged when confabulation
    is actually detected AND catalog metadata was given to the model — it
    marks confabulation plausibly anchored on the metadata, not the mere
    presence of metadata.

    :param hypothesis: Model output
    :type hypothesis: str
    :param reference: Ground-truth text
    :type reference: str
    :param catalog_metadata_provided: Whether catalog metadata was in the prompt
    :type catalog_metadata_provided: bool
    :return: List of failure-mode labels (empty when output looks healthy)
    :rtype: List[str]
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

    if catalog_metadata_provided and "plausible_confabulation" in flags:
        flags.append("metadata_anchored_confabulation")

    return flags


# ── Genizah visible-ink scoring ───────────────────────────────────────────────
# The Genizah eval ground truths are diplomatic editions dense with editorial
# apparatus (uncertainty '?', bracket reconstructions, lacuna dot-runs, column
# markers, supplied letters).  Scoring raw model output against raw GT
# penalises every model for not reproducing apparatus that is not ink on the
# page.  Visible-ink scoring cleans BOTH sides: the GT keeps only visible ink
# (via the training pipeline's clean_diplomatic), and the hypothesis is
# stripped of damage markers the prompt asks for ([?]) plus markdown noise.

_MD_NOISE_RE = re.compile(r"[*#_`]+")
_BRACKET_MARK_RE = re.compile(r"\[[^\]\n]{0,4}\]")          # [?], [..], [ ]
_DOTS_RUN_RE = re.compile(r"(?:[.·…]\s*){2,}")
_EDITORIAL_CHARS_RE = re.compile(r"[¶@^§\\/|<>{}\[\]]")
_QMARK_RE = re.compile(r"\(\?+\)|\?")


def dedup_repeated_tail(text: str, min_letters: int = 60) -> str:
    """Cut a duplicated tail block (same text accidentally concatenated twice).

    Some catalog ground truths glue two copies of the same edition section
    together; the repeated tail double-counts every character of GT.  Works on
    a letters-only projection so lacuna dots and markup differences between
    the two copies do not hide the repeat.

    :param text: Ground-truth text.
    :type text: str
    :param min_letters: Minimum length (in letters) of the repeated block.
    :type min_letters: int
    :return: Text with the repeated tail removed (or unchanged).
    :rtype: str
    """
    letters = []
    positions = []
    for i, ch in enumerate(text):
        if _HEBREW_SCRIPT_RE.match(ch) or _ARABIC_SCRIPT_RE.match(ch):
            letters.append(ch)
            positions.append(i)
    n = len(letters)
    joined = "".join(letters)
    best_k = 0
    for k in range(n // 2, min_letters - 1, -1):
        if joined[:k] == joined[n - k:]:
            best_k = k
            break
    if not best_k:
        return text
    cut_at = positions[n - best_k]
    return text[:cut_at].rstrip()


def genizah_visible_ink_gt(gt_text: str) -> str:
    """Reduce a diplomatic ground truth to its visible-ink scoring form.

    :param gt_text: Raw diplomatic ground truth.
    :type gt_text: str
    :return: Visible-ink text (no apparatus, no gap tokens, deduped).
    :rtype: str
    """
    from src.datasets.cleaning.clean_genizah_transcriptions import (
        clean_diplomatic,
        GAP,
    )
    deduped = dedup_repeated_tail(gt_text)
    cleaned, _recon, _visible = clean_diplomatic(deduped)
    cleaned = cleaned.replace(GAP, " ")
    cleaned = cleaned.replace("@", "")
    return normalize_whitespace(cleaned)


def normalize_ink_hypothesis(hyp_text: str) -> str:
    """Normalize a model transcription for visible-ink scoring.

    Strips the damage markers the prompt requests ([?]), editorial characters
    models copy from their training distribution, markdown emphasis, and
    lacuna dot-runs — mirroring the GT-side cleaning so neither side is
    penalised for apparatus.

    :param hyp_text: Raw model output.
    :type hyp_text: str
    :return: Visible-ink hypothesis text.
    :rtype: str
    """
    text = unicodedata.normalize("NFC", hyp_text)
    text = _MD_NOISE_RE.sub(" ", text)
    text = _BRACKET_MARK_RE.sub(" ", text)
    text = _DOTS_RUN_RE.sub(" ", text)
    text = _QMARK_RE.sub("", text)
    text = _EDITORIAL_CHARS_RE.sub(" ", text)
    return normalize_whitespace(text)


def cer_ink_pair(hypothesis: str, reference_gt: str) -> Tuple[float, float]:
    """Visible-ink CER (strict, lenient) between raw model output and raw GT.

    :param hypothesis: Raw model transcription.
    :type hypothesis: str
    :param reference_gt: Raw diplomatic ground truth.
    :type reference_gt: str
    :return: (strict CER, nikud-stripped lenient CER) on visible-ink forms.
    :rtype: Tuple[float, float]
    """
    hyp = normalize_ink_hypothesis(hypothesis)
    ref = genizah_visible_ink_gt(reference_gt)
    return cer_pair(hyp, ref)
