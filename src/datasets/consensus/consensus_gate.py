"""Confidence gate for the Kraken x VLM consensus pipeline.

Pure decision logic — no I/O, no services — so it is unit-testable and the
corpus runner stays thin.  All text machinery is imported from the offline
scorer / metrics modules: the gate must measure agreement with EXACTLY the
functions the calibration study used, or the calibrated thresholds do not
transfer.

Calibration provenance (2026-08-11, verified 131-fragment benchmark,
``helper_eval_scripts/calibrate_consensus.py``):

* agreement < 0.2  -> Tier-A median CER 6.18 (hallucination mass);
  agreement >= 0.2 -> 0.15-0.29.  Engines hallucinate *differently*, so
  cross-family agreement rejects invented text.
* Compound gate (loop screen + agreement >= 0.10): 84% of fragments
  auto-accepted, ZERO hallucinated docs leaked, Tier-A median CER 0.208.
* agreement >= 0.35: 41% accepted at Tier-A median CER 0.166.
* The loop screen is required because a VLM can loop on GENUINE page text
  (T_S_13J8_6): n-gram agreement is multiplicity-blind and cannot see it,
  but ``loop_ratio`` needs no ground truth.
"""

from dataclasses import dataclass

from src.datasets.evaluations.metrics import normalize_ink_hypothesis
from src.datasets.evaluations.helper_eval_scripts.audit_genizah_benchmark import (
    letters_only,
)
from src.datasets.evaluations.helper_eval_scripts.score_genizah_offline import (
    ABSTAIN_MIN_CHARS,
    LOOP_REPEAT_RATIO,
    asserted_letters,
    loop_ratio,
    ngram_precision,
)

GATE_VERSION = "v1-20260811"

# Calibrated thresholds — change ONLY together with a recalibration run.
AGREE_ACCEPT = 0.10
AGREE_HIGH = 0.35

TIER_HIGH = "high"
TIER_STANDARD = "standard"
TIER_ESCALATE = "escalate"


@dataclass
class GateResult:
    """Outcome of gating one document's pair of transcriptions.

    :param tier: One of ``high`` / ``standard`` / ``escalate``.
    :type tier: str
    :param reason: Why the tier was assigned (``agreement``, ``loop``,
        ``abstain``, ``vlm_failed``, ``htr_failed``).
    :type reason: str
    :param agreement: Harmonic mean of the two directional containments.
    :type agreement: float
    :param agree_vlm_in_htr: Share of the VLM's 5-grams the HTR also saw —
        the operative hallucination check.
    :type agree_vlm_in_htr: float
    :param agree_htr_in_vlm: Share of the HTR's 5-grams the VLM also saw.
    :type agree_htr_in_vlm: float
    :param vlm_loop_ratio: Share of the VLM text covered by its single most
        repeated 12-gram (GT-free loop signal).
    :type vlm_loop_ratio: float
    :param vlm_letters: Letter count of the normalised VLM hypothesis.
    :type vlm_letters: int
    :param htr_letters: Letter count of the normalised HTR hypothesis.
    :type htr_letters: int
    """

    tier: str
    reason: str
    agreement: float
    agree_vlm_in_htr: float
    agree_htr_in_vlm: float
    vlm_loop_ratio: float
    vlm_letters: int
    htr_letters: int


def harmonic(a: float, b: float) -> float:
    """Harmonic mean of two rates.

    :param a: First rate in [0, 1].
    :type a: float
    :param b: Second rate in [0, 1].
    :type b: float
    :return: Harmonic mean, 0.0 when either rate is 0.
    :rtype: float
    """
    return (2 * a * b / (a + b)) if (a + b) else 0.0


def evaluate_pair(vlm_raw: str, htr_raw: str) -> GateResult:
    """Gate one document given both engines' raw outputs.

    Escalation is the safe default: anything the gate cannot positively
    vouch for (engine failure, abstention-length output, loop collapse,
    low cross-family agreement) goes to the escalate tier for line-wise
    re-reading or human review.

    :param vlm_raw: Raw VLM transcription (may be empty on failure).
    :type vlm_raw: str
    :param htr_raw: Raw Kraken transcription (may be empty on failure).
    :type htr_raw: str
    :return: The gate decision with its evidence.
    :rtype: GateResult
    """
    vlm_norm = normalize_ink_hypothesis(vlm_raw or "")
    htr_norm = normalize_ink_hypothesis(htr_raw or "")
    vlm_letters = letters_only(vlm_norm)
    htr_letters = letters_only(htr_norm)
    vlm_asserted = asserted_letters(vlm_norm)
    vloop = loop_ratio(vlm_asserted)

    agree_v_in_h = ngram_precision(vlm_letters, htr_letters)
    agree_h_in_v = ngram_precision(htr_letters, vlm_letters)
    agreement = harmonic(agree_v_in_h, agree_h_in_v)

    def result(tier: str, reason: str) -> GateResult:
        """Build the result with the metrics computed above.

        :param tier: Assigned tier.
        :type tier: str
        :param reason: Assignment reason.
        :type reason: str
        :return: Populated gate result.
        :rtype: GateResult
        """
        return GateResult(
            tier=tier, reason=reason, agreement=round(agreement, 4),
            agree_vlm_in_htr=round(agree_v_in_h, 4),
            agree_htr_in_vlm=round(agree_h_in_v, 4),
            vlm_loop_ratio=round(vloop, 4),
            vlm_letters=len(vlm_letters), htr_letters=len(htr_letters),
        )

    if not (htr_raw or "").strip():
        return result(TIER_ESCALATE, "htr_failed")
    if not (vlm_raw or "").strip():
        return result(TIER_ESCALATE, "vlm_failed")
    if len(vlm_asserted) < ABSTAIN_MIN_CHARS:
        return result(TIER_ESCALATE, "abstain")
    if vloop > LOOP_REPEAT_RATIO:
        return result(TIER_ESCALATE, "loop")
    if agreement >= AGREE_HIGH:
        return result(TIER_HIGH, "agreement")
    if agreement >= AGREE_ACCEPT:
        return result(TIER_STANDARD, "agreement")
    return result(TIER_ESCALATE, "agreement")


def build_fragment_prompt(doc_id: str) -> str:
    """Build the transcription prompt for one fragment.

    Byte-identical to ``fragment_evals._fragment_prompt`` — the prompt the
    VLM was benchmarked and calibrated with.  The benchmark passed the
    underscored doc_id through ``_series_from_doc_id``, which returns the
    whole id when it contains no spaces; the corpus runner reproduces that
    behaviour deliberately.  Improving the prompt is a benchmarked-track
    experiment, not a rollout-time edit.

    :param doc_id: Document identifier (underscored canonical id).
    :type doc_id: str
    :return: Prompt text.
    :rtype: str
    """
    series = _series_from_doc_id(doc_id)
    return f"""This is a manuscript from the Cairo Genizah ({series} collection).
The text may be in Hebrew, Aramaic, or Judaeo-Arabic (Arabic written in Hebrew script).

Transcribe the text in this image exactly as written. Do not normalize or correct the text.
Preserve all vocalization marks (nikud) and line structure.
Mark damaged or unclear characters with [?].

Return ONLY the transcription with no commentary."""


def _series_from_doc_id(doc_id: str) -> str:
    """Extract the collection series from a Genizah document ID.

    Copied from ``genizah_fragment_agent._series_from_doc_id`` (importing
    that module pulls the full cloud-model stack).  Must stay behaviourally
    identical to it.

    :param doc_id: Document identifier.
    :type doc_id: str
    :return: Series prefix, or the whole id when it contains no spaces.
    :rtype: str
    """
    import re

    s = doc_id.strip()
    if re.match(r"T-S\s+NS", s, re.IGNORECASE):
        return "T-S NS"
    if re.match(r"T-S", s, re.IGNORECASE):
        return "T-S"
    parts = s.split()
    return parts[0] if parts else "Unknown"
