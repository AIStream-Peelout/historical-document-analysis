# File name: scorer.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Scorer v2 for the text27b evals (``score_v2``): translation, catalogue, description and relations.

Ported from the 2026-10-18 probe scorer (``score_probe.py``) with the v2 changes of plan §5:

* Gil / leading line markers are stripped from BOTH sides before chrF++ (:func:`clean_translation`
  calls :func:`textnorm.strip_line_markers`); the v1 cleaning is kept as ``chrfpp_v1`` for comparison.
* Corpus chrF++ (sacrebleu ``CHRF(word_order=2)``) carries a document-level bootstrap CI, and two
  systems (or two conditions) on the same documents get a paired bootstrap CI of the difference
  (:func:`bootstrap_ci`, :func:`paired_bootstrap`; exact corpus statistics per resample).
* Failure counters: empty, wrong script, length ratio > 2 / < 0.5, repetition loops, copy-through
  of the source (JA→HE), meta lines; ``degenerate`` = empty or wrong script or repetition or copy-through.
* Catalogue records in the TRAINING schema (``date.year_ce_start/end``, ``origin``, ``destination``,
  ``places_mentioned``) are mapped onto the probe schema (``date_year_ce/_max``, ``origin_place``,
  ``destination_place``, ``places``) before the field checks (:func:`normalise_catalogue`).
* JSON validity (strict and lenient), document type, date ±2 / ±10 y (stated vs inferred, abstention),
  places (origin / destination / mentioned), people recall (PGP ``people.csv`` names in the
  description, cached judge names, or the gold record's people) and a deterministic grounding share
  of predicted people and of summary specifics, split into "in the reference but not the input" and
  "in neither" (:func:`specifics_split`). No judge calls are made.
* Relations: micro precision / recall / F1 over (subject, predicate, object) with name and place
  normalisation (:func:`relations_prf`).
* Canon comparison (E8, :data:`CANON_SCORER_VERSION`; deterministic, no judge): ``canon_identify`` -- exact reference
  (normalised string), same work / title, same section (same title and overlapping daf-amud / chapter ranges), anchor
  word F1 and the over-quoting rate (an anchor longer than 12 words recites the canon); ``canon_diff`` -- precision /
  recall / F1 of the predicted variant multiset against the deterministic apparatus, typed ``(type, fragment, canon)``
  (headline), untyped ``(fragment, canon)`` and strict (``+ pos``), per variant type, plus reference, agreement-ratio
  and lacuna-count errors (:func:`canon_identify_metrics`, :func:`canon_diff_metrics`).

Public API
    Text: :func:`clean_translation`, :func:`script_shares`, :func:`repetition_stats`,
    :func:`is_repetitive`, :func:`source_copy_share`, :func:`translation_metrics`,
    :func:`describe_metrics`.
    Corpus: :func:`corpus_scores`, :func:`segment_stats`, :func:`bootstrap_ci`,
    :func:`paired_bootstrap`.
    Catalogue: :func:`strict_json`, :func:`parse_json_lenient`, :func:`parse_catalogue`,
    :func:`normalise_catalogue`, :func:`reference_from_gold`, :func:`reference_date`,
    :func:`date_metrics`, :func:`place_key`, :func:`place_match`, :func:`place_metrics`,
    :func:`name_norm`, :func:`people_recall`, :func:`load_people_dictionary`,
    :func:`dictionary_people`, :class:`SourceIndex`, :func:`name_grounded`,
    :func:`people_grounding`, :func:`specifics_split`, :func:`catalogue_metrics`.
    Relations: :func:`parse_relations`, :func:`relations_prf`.
    Canon: :func:`canon_norm`, :func:`norm_ref`, :func:`parse_canon_ref`, :func:`section_key`, :func:`same_section`,
    :func:`canon_identify_metrics`, :func:`variant_keys`, :func:`multiset_prf`, :func:`canon_diff_metrics`,
    :func:`aggregate_canon_identify`, :func:`aggregate_canon_diff`.
    v0_2 rows: :func:`document_of`, :func:`score_output`, :func:`aggregate_scored`.
    Probe rescore: :func:`rescore` (CLI ``rescore``) → ``scores_v2/<system>.json``,
    ``scores_v2/summary_v2.json`` and ``BASELINES_v2.md`` under ``paths.out_dir``.

CLI::

    .venv/bin/python -m src.finetuning.text27b.scorer rescore            # all stored probe outputs
    .venv/bin/python -m src.finetuning.text27b.scorer score --rows families/translate_en.jsonl \\
        --hyps my_outputs.jsonl --out scored.json                        # outputs keyed by row_id
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sacrebleu.metrics import BLEU, CHRF

from src.finetuning.text27b import paths as _paths
from src.finetuning.text27b import rows as _rows
from src.finetuning.text27b import textnorm as tn

LOG = logging.getLogger(__name__)

SCORER_VERSION = "score_v2-20261022"
#: Prompt version of the stored 2026-10-18 probe outputs.
PROBE_PROMPT_VERSION = "probe-v1-20261018"
#: Judge version of the 2026-10-18 cache (read only; v2 makes no judge calls).
JUDGE_VERSION_V1 = "judge-v1-20261018"
#: Version of the canon-comparison metrics (E8).
CANON_SCORER_VERSION = "canon_v2-20261026"
#: Longest anchor (words) a canon_identify answer may quote; longer = recitation of the canon (over-quoting).
CANON_ANCHOR_WORDS = 12
#: The canon_diff variant types (prep pass 2026-10-26: ``orthographic`` renamed ``orthographic_or_minor``).
CANON_VARIANT_TYPES: Tuple[str, ...] = ("omission", "addition", "substitution", "orthographic_or_minor", "order")
#: Earlier / alternative type names mapped onto :data:`CANON_VARIANT_TYPES` before scoring (a model or a stored output
#: that still says ``orthographic`` is scored as ``orthographic_or_minor``).
CANON_TYPE_ALIASES: Dict[str, str] = {"orthographic": "orthographic_or_minor", "orthographic-or-minor": "orthographic_or_minor",
                                      "orthographic or minor": "orthographic_or_minor"}

CHRFPP = CHRF(word_order=2)
CHRF_PLAIN = CHRF()
BLEU_CORPUS = BLEU()

#: Minimum share of the target script among the letters of a non-empty translation.
WRONG_SCRIPT_MIN: Dict[str, Tuple[str, float]] = {"en": ("latin", 0.50), "he": ("hebrew", 0.30)}
#: An exact repeat of a ≤80-character unit (8+ times) spanning this many characters is a loop.
REPEAT_RUN_CHARS = 300
#: Distinct share of word 4-grams below which a text with enough 4-grams is a loop.
DISTINCT4_MIN = 0.25
DISTINCT4_MIN_NGRAMS = 40
#: Share of a hypothesis' word tokens copied from the source above which it is copy-through.
COPY_THROUGH_SHARE = 0.5
#: Default bootstrap resamples and seed.
N_BOOT = 1000
SEED = 3407

_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*$")
META_LOOSE = re.compile(r"^\s*(I |Note:|As an AI)")
META_STRICT = re.compile(
    r"^\s*(As an AI|Note:|Notes:|Translator'?s note|Here is|Here's|I cannot|I can't|I'm sorry|"
    r"I am sorry|I am unable|I'm unable|Sorry,)", flags=re.I)
_RUN_RE = re.compile(r"(.{1,80}?)\1{7,}", re.S)
_TOKEN_RE = re.compile(r"[^\W\d_][\w׳'״\"-]+")
_DOC_RE = re.compile(r"<document>\n?(.*?)\n?</document>", re.S)


# ================================================================================ translation
def clean_translation(text: Optional[str], strip_markers: bool = True) -> str:
    """Clean a translation (hypothesis or reference) before chrF / BLEU.

    v2: edition line markers are removed first (:func:`textnorm.strip_line_markers`); then, as in
    v1, code-fence lines and lines starting with ``§`` (side / margin labels) are dropped, each
    line is stripped and empty lines are removed.

    :param text: Raw text (None = empty).
    :param strip_markers: Apply the v2 marker strip; False reproduces the v1 cleaning exactly.
    :returns: Cleaned text, lines joined by newlines.
    """
    if not text:
        return ""
    if strip_markers:
        text = tn.strip_line_markers(text)[0]
    lines = []
    for line in text.split("\n"):
        if _FENCE.match(line) or line.lstrip().startswith("§"):
            continue
        line = line.strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def script_shares(text: str) -> Dict[str, float]:
    """Share of Hebrew / Arabic / Latin letters among all letters of a text.

    :param text: Any text.
    :returns: ``{"hebrew": f, "arabic": f, "latin": f}`` (0 when there are no letters).
    """
    counts = tn.script_counts(text)
    total = sum(counts.values())
    return {k: (v / total if total else 0.0) for k, v in counts.items()}


def target_lang(target: str) -> str:
    """Map a target name to ``en`` or ``he``.

    :param target: ``English`` / ``Modern Hebrew`` (probe), ``translate_en`` / ``translate_he``
        (family) or ``en`` / ``he``.
    :returns: ``en`` or ``he``.
    :raises ValueError: On an unknown target.
    """
    t = (target or "").lower()
    if t in ("english", "translate_en", "en"):
        return "en"
    if t in ("modern hebrew", "hebrew", "translate_he", "he"):
        return "he"
    raise ValueError(f"unknown translation target {target!r}")


def repetition_stats(text: str) -> Dict[str, Any]:
    """Loop statistics of a text.

    :param text: Any text.
    :returns: ``{"distinct4": share of distinct word 4-grams (None when fewer than 4 words),
        "n4": number of word 4-grams, "max_run": characters covered by the longest exact repeat
        of a ≤80-character unit occurring 8+ times in a row (whitespace collapsed)}``.
    """
    toks = text.split()
    grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    flat = re.sub(r"\s+", " ", text)
    run = max((len(m.group(0)) for m in _RUN_RE.finditer(flat)), default=0)
    return {"distinct4": (len(set(grams)) / len(grams)) if grams else None, "n4": len(grams), "max_run": run}


def is_repetitive(stats: Dict[str, Any]) -> bool:
    """Whether :func:`repetition_stats` describe a degenerate loop.

    Calibrated on the 10-18 probe: human references reach at most a 44-character run and a 0.85
    distinct-4-gram share; frontier outputs at most 150 characters; degenerate stock loops run for
    3,000+ characters. A legitimately repeated passage (a letter copied twice) sits near 0.5.

    :param stats: Output of :func:`repetition_stats`.
    :returns: True for a loop.
    """
    if stats["max_run"] >= REPEAT_RUN_CHARS:
        return True
    d4 = stats["distinct4"]
    return d4 is not None and stats["n4"] >= DISTINCT4_MIN_NGRAMS and d4 < DISTINCT4_MIN


def _word_tokens(text: str) -> List[str]:
    """Word tokens with at least two characters, starting with a letter (any script).

    :param text: Any text.
    :returns: Tokens in order.
    """
    return _TOKEN_RE.findall(unicodedata.normalize("NFC", text or ""))


def source_copy_share(hyp: str, source: Optional[str]) -> Optional[float]:
    """Share of a hypothesis' word tokens that occur verbatim among the source's tokens.

    Human JA→HE references copy at most 0.24 of their tokens from the edition (names, Hebrew
    formulae); frontier outputs at most 0.39; a copied source scores close to 1.

    :param hyp: Cleaned hypothesis.
    :param source: The input text the system saw (None = unknown).
    :returns: The share, or None without a source or hypothesis tokens.
    """
    if not source:
        return None
    toks = _word_tokens(hyp)
    if not toks:
        return None
    src = set(_word_tokens(source))
    return sum(1 for t in toks if t in src) / len(toks)


def translation_metrics(hyp_raw: Optional[str], ref_raw: str, target: str,
                        source: Optional[str] = None) -> Dict[str, Any]:
    """Per-document translation metrics and failure flags (v2 cleaning).

    :param hyp_raw: Raw model output.
    :param ref_raw: Human reference translation.
    :param target: Target language (see :func:`target_lang`).
    :param source: Input text the system saw, for the copy-through check (optional).
    :returns: chrF++ (v2) and chrF++ under the v1 cleaning (document level), character counts,
        length ratio, script shares, flags (``empty``, ``wrong_script``, ``len_gt2``, ``len_lt05``,
        ``repetition``, ``copy_through``, ``degenerate``, ``source_names``), meta-line counts, and the private
        fields ``_hyp`` / ``_ref`` / ``_hyp_v1`` / ``_ref_v1`` (cleaned texts) and ``_st`` / ``_st_v1``
        (chrF++ sufficient statistics), dropped before writing.
    """
    lang = target_lang(target)
    hyp, ref = clean_translation(hyp_raw), clean_translation(ref_raw)
    hyp_v1, ref_v1 = clean_translation(hyp_raw, False), clean_translation(ref_raw, False)
    shares = script_shares(hyp)
    empty = not hyp.strip()
    script, floor = WRONG_SCRIPT_MIN[lang]
    wrong_script = (not empty) and shares[script] < floor
    ratio = (len(hyp) / len(ref)) if ref else None
    rep = repetition_stats(hyp)
    copy = source_copy_share(hyp, source)
    copy_through = copy is not None and copy >= COPY_THROUGH_SHARE
    repetition = (not empty) and is_repetitive(rep)
    lines = hyp.split("\n") if hyp else []
    st = segment_stats([hyp], [ref])[0]
    st_v1 = st if (hyp_v1 == hyp and ref_v1 == ref) else segment_stats([hyp_v1], [ref_v1])[0]
    return {
        "chrfpp": _score_from_stats(CHRFPP, st),
        "chrfpp_v1": _score_from_stats(CHRFPP, st_v1),
        "hyp_chars": len(hyp), "ref_chars": len(ref), "len_ratio": ratio,
        "share_hebrew": round(shares["hebrew"], 3), "share_latin": round(shares["latin"], 3),
        "share_arabic": round(shares["arabic"], 3),
        "copy_share": None if copy is None else round(copy, 3),
        "distinct4": None if rep["distinct4"] is None else round(rep["distinct4"], 3), "max_run": rep["max_run"],
        "empty": empty, "wrong_script": wrong_script,
        "len_gt2": ratio is not None and ratio > 2, "len_lt05": ratio is not None and ratio < 0.5,
        "repetition": repetition, "copy_through": copy_through,
        "degenerate": empty or wrong_script or repetition or copy_through,
        "source_names": tn.has_source_names(hyp_raw or ""),
        "meta_lines_loose": sum(1 for ln in lines if META_LOOSE.match(ln)),
        "meta_lines_strict": sum(1 for ln in lines if META_STRICT.match(ln)),
        "_hyp": hyp, "_ref": ref, "_hyp_v1": hyp_v1, "_ref_v1": ref_v1, "_st": st, "_st_v1": st_v1,
    }


# ===================================================================================== corpus
def corpus_scores(hyps: Sequence[str], refs: Sequence[str]) -> Dict[str, float]:
    """Corpus-level chrF++, chrF and BLEU over aligned document lists.

    :param hyps: Cleaned hypotheses.
    :param refs: Cleaned references, same order.
    :returns: ``{"chrfpp", "chrf", "bleu"}``.
    """
    return {"chrfpp": CHRFPP.corpus_score(list(hyps), [list(refs)]).score,
            "chrf": CHRF_PLAIN.corpus_score(list(hyps), [list(refs)]).score,
            "bleu": BLEU_CORPUS.corpus_score(list(hyps), [list(refs)]).score}


def segment_stats(hyps: Sequence[str], refs: Sequence[str], metric: CHRF = CHRFPP) -> np.ndarray:
    """Per-document sufficient statistics of a chrF metric (rows sum to the corpus statistics).

    :param hyps: Cleaned hypotheses.
    :param refs: Cleaned references, same order.
    :param metric: A sacrebleu ``CHRF`` instance.
    :returns: Integer array of shape ``(n_docs, n_stats)``.
    """
    return np.asarray(metric._extract_corpus_statistics(list(hyps), [list(refs)]), dtype=np.int64)


def _score_from_stats(metric: CHRF, stats: np.ndarray) -> float:
    """Corpus score from summed statistics.

    :param metric: A sacrebleu ``CHRF`` instance.
    :param stats: Summed statistics vector.
    :returns: The corpus score.
    """
    return float(metric._compute_score_from_stats([int(x) for x in stats]).score)


def _resample_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    """Bootstrap index matrix (documents drawn with replacement).

    :param n: Number of documents.
    :param n_boot: Number of resamples.
    :param seed: RNG seed.
    :returns: Integer array of shape ``(n_boot, n)``.
    """
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_stats(st: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED, alpha: float = 0.05,
                    metric: CHRF = CHRFPP) -> Dict[str, Any]:
    """Corpus score with a document-level percentile bootstrap CI from per-document statistics.

    :param st: Per-document statistics (:func:`segment_stats`), shape ``(n_docs, n_stats)``.
    :param n_boot: Number of resamples.
    :param seed: RNG seed.
    :param alpha: 1 − confidence level.
    :param metric: The metric the statistics belong to.
    :returns: ``{"score", "lo", "hi", "n", "n_boot"}`` (None values for no documents).
    """
    n = len(st)
    if n == 0:
        return {"score": None, "lo": None, "hi": None, "n": 0, "n_boot": n_boot}
    idx = _resample_indices(n, n_boot, seed)
    boots = np.array([_score_from_stats(metric, st[i].sum(axis=0)) for i in idx])
    return {"score": _score_from_stats(metric, st.sum(axis=0)),
            "lo": float(np.quantile(boots, alpha / 2)), "hi": float(np.quantile(boots, 1 - alpha / 2)),
            "n": n, "n_boot": n_boot}


def bootstrap_ci(hyps: Sequence[str], refs: Sequence[str], n_boot: int = N_BOOT, seed: int = SEED,
                 alpha: float = 0.05, metric: CHRF = CHRFPP) -> Dict[str, Any]:
    """Corpus chrF++ with a document-level percentile bootstrap CI.

    :param hyps: Cleaned hypotheses.
    :param refs: Cleaned references, same order.
    :param n_boot: Number of resamples.
    :param seed: RNG seed.
    :param alpha: 1 − confidence level.
    :param metric: A sacrebleu ``CHRF`` instance.
    :returns: ``{"score", "lo", "hi", "n", "n_boot"}`` (None values for an empty list).
    """
    if not hyps:
        return bootstrap_stats(np.zeros((0, 1), dtype=np.int64), n_boot, seed, alpha, metric)
    return bootstrap_stats(segment_stats(hyps, refs, metric), n_boot, seed, alpha, metric)


def paired_bootstrap_stats(sa: np.ndarray, sb: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED,
                           alpha: float = 0.05, metric: CHRF = CHRFPP) -> Dict[str, Any]:
    """Paired document-level bootstrap of the corpus score difference ``a − b`` from per-document statistics.

    :param sa: Statistics of a, shape ``(n_docs, n_stats)``.
    :param sb: Statistics of b on the same documents, same shape and order.
    :param n_boot: Number of resamples.
    :param seed: RNG seed.
    :param alpha: 1 − confidence level.
    :param metric: The metric the statistics belong to.
    :returns: ``{"a", "b", "delta", "lo", "hi", "p_le0" (share of resamples with delta ≤ 0),
        "excludes_0", "n", "n_boot"}``.
    :raises ValueError: When the shapes differ.
    """
    if sa.shape != sb.shape:
        raise ValueError("paired bootstrap needs aligned statistics")
    n = len(sa)
    if n == 0:
        return {"a": None, "b": None, "delta": None, "lo": None, "hi": None, "p_le0": None,
                "excludes_0": False, "n": 0, "n_boot": n_boot}
    idx = _resample_indices(n, n_boot, seed)
    deltas = np.array([_score_from_stats(metric, sa[i].sum(axis=0)) - _score_from_stats(metric, sb[i].sum(axis=0))
                       for i in idx])
    a, b = _score_from_stats(metric, sa.sum(axis=0)), _score_from_stats(metric, sb.sum(axis=0))
    lo, hi = float(np.quantile(deltas, alpha / 2)), float(np.quantile(deltas, 1 - alpha / 2))
    return {"a": a, "b": b, "delta": a - b, "lo": lo, "hi": hi, "p_le0": float(np.mean(deltas <= 0)),
            "excludes_0": lo > 0 or hi < 0, "n": n, "n_boot": n_boot}


def paired_bootstrap(hyps_a: Sequence[str], hyps_b: Sequence[str], refs: Sequence[str], n_boot: int = N_BOOT,
                     seed: int = SEED, alpha: float = 0.05, metric: CHRF = CHRFPP) -> Dict[str, Any]:
    """Paired document-level bootstrap of the corpus chrF++ difference ``a − b`` on the same documents.

    :param hyps_a: Cleaned hypotheses of system / condition a.
    :param hyps_b: Cleaned hypotheses of system / condition b, same documents and order.
    :param refs: Cleaned references, same order.
    :param n_boot: Number of resamples.
    :param seed: RNG seed.
    :param alpha: 1 − confidence level.
    :param metric: A sacrebleu ``CHRF`` instance.
    :returns: See :func:`paired_bootstrap_stats`.
    :raises ValueError: When the lists differ in length.
    """
    if not (len(hyps_a) == len(hyps_b) == len(refs)):
        raise ValueError("paired bootstrap needs aligned lists")
    if not refs:
        empty = np.zeros((0, 1), dtype=np.int64)
        return paired_bootstrap_stats(empty, empty, n_boot, seed, alpha, metric)
    return paired_bootstrap_stats(segment_stats(hyps_a, refs, metric), segment_stats(hyps_b, refs, metric),
                                  n_boot, seed, alpha, metric)


# ================================================================================== catalogue
def _strip_fences(text: str) -> str:
    """Remove one surrounding markdown code fence.

    :param text: Raw text.
    :returns: The fenced content, or the stripped text when there is no fence.
    """
    m = re.match(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n\s*```\s*$", text or "", re.S)
    return m.group(1) if m else (text or "").strip()


def strict_json(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse an answer that must be exactly one JSON object (whitespace allowed, no fences or prose).

    :param text: Raw model output.
    :returns: The object, or None.
    """
    try:
        obj = json.loads((text or "").strip())
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_json_lenient(text: Optional[str], required_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Parse a JSON answer tolerating fences or prose (port of the probe's catalogue parsing).

    Tries the fenced content, then the first ``{`` … last ``}`` span, then the LAST complete JSON
    object in the text (an answer may hold a corrected second object); every candidate must carry
    ``required_key`` when one is given.

    :param text: Raw model output.
    :param required_key: Key the fallback object must carry (None = any object).
    :returns: The parsed object or None.
    """
    if not text:
        return None

    def usable(obj: Optional[Dict[str, Any]]) -> bool:
        """Whether a parsed candidate is an object carrying the required key.

        :param obj: Candidate or None.
        :returns: True when usable.
        """
        return obj is not None and (required_key is None or required_key in obj)

    first = strict_json(_strip_fences(text))
    if usable(first):
        return first
    m = re.search(r"\{.*\}", text, flags=re.S)
    span = strict_json(m.group(0)) if m else None
    if usable(span):
        return span
    decoder = json.JSONDecoder()
    last = None
    for match in re.finditer(r"\{", text):
        try:
            cand, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(cand, dict) and (required_key is None or required_key in cand):
            last = cand
    return last


def parse_catalogue(text: Optional[str], output_json: Optional[Dict[str, Any]] = None
                    ) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Catalogue object of an answer and whether the answer was strictly valid JSON.

    :param text: Raw model output.
    :param output_json: An object already parsed by the runner (probe rows), used when present.
    :returns: ``(object or None, strict)``; ``strict`` is True when the raw text alone is one JSON object.
    """
    strict = strict_json(text) is not None
    if isinstance(output_json, dict):
        return output_json, strict
    return parse_json_lenient(text, "document_type"), strict


def _as_int(value: Any) -> Optional[int]:
    """Coerce a JSON year value to int.

    :param value: int, float, numeric string or anything else.
    :returns: The integer or None.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    found = re.search(r"\d{3,4}", str(value))
    return int(found.group(0)) if found else None


def _as_list(value: Any) -> List[Any]:
    """Coerce a JSON value to a list.

    :param value: list, scalar or None.
    :returns: The list (empty for None / empty string).
    """
    if value is None or value == "":
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def normalise_catalogue(obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Map a catalogue record onto the probe schema (training schema → probe schema; probe records pass).

    ``date.year_ce_start`` / ``date.year_ce_end`` → ``date_year_ce`` / ``date_year_ce_max``;
    ``date.original`` → ``date_text``; ``date.inferred`` → ``date_inferred``; ``origin`` /
    ``destination`` → ``origin_place`` / ``destination_place``; ``places_mentioned`` → ``places``;
    ``languages`` ``{"primary", "secondary"}`` → a flat list. Keys already in the probe schema win.

    :param obj: Parsed record (None = empty).
    :returns: A new dict in the probe schema (unknown keys are kept).
    """
    obj = dict(obj or {})
    out: Dict[str, Any] = dict(obj)
    date = obj.get("date")
    if isinstance(date, dict):
        out.setdefault("date_year_ce", date.get("year_ce_start"))
        out.setdefault("date_year_ce_max", date.get("year_ce_end"))
        out.setdefault("date_text", date.get("original") or date.get("standard"))
        out.setdefault("date_inferred", date.get("inferred"))
        out.setdefault("date_standard", date.get("standard"))
    elif isinstance(date, str) and "date_text" not in out:
        out["date_text"] = date
    for src, dst in (("origin", "origin_place"), ("destination", "destination_place"),
                     ("places_mentioned", "places")):
        if src in obj and dst not in obj:
            val = obj[src]
            if isinstance(val, dict):
                val = val.get("name")
            out[dst] = val
    langs = obj.get("languages")
    if isinstance(langs, dict):
        out["languages"] = _as_list(langs.get("primary")) + _as_list(langs.get("secondary"))
    out["people"] = [p if isinstance(p, dict) else {"name": str(p)} for p in _as_list(obj.get("people"))]
    out["places"] = [p.get("name") if isinstance(p, dict) else p for p in _as_list(out.get("places"))]
    return out


def years_in(text: Optional[str]) -> List[int]:
    """CE years (3–4 digits, 500–2000) appearing in a PGP standard-date string.

    :param text: e.g. ``1065``, ``1094-09-14/1095-09-02``, ``1050/1060``, ``ca. 1050``.
    :returns: The years in order of appearance.
    """
    if not text:
        return []
    return [int(m) for m in re.findall(r"(?<!\d)\d{3,4}(?!\d)", str(text)) if 500 <= int(m) <= 2000]


def reference_from_gold(gold: Dict[str, Any]) -> Dict[str, Any]:
    """Pairs-style reference built from a gold catalogue record in the training schema.

    :param gold: Gold record (the assistant turn of a ``catalogue_json`` row, parsed).
    :returns: ``{"type", "doc_date_standard", "inferred_date_standard", "ref_years",
        "ref_date_source", "origin", "destination", "location", "mentioned", "description", "people"}``.
    """
    g = normalise_catalogue(gold)
    date = gold.get("date") if isinstance(gold.get("date"), dict) else {}
    standard = str(date.get("standard") or "")
    years = [y for y in (_as_int(date.get("year_ce_start")), _as_int(date.get("year_ce_end"))) if y is not None]
    inferred = bool(date.get("inferred"))
    return {"type": gold.get("document_type") or "", "origin": g.get("origin_place") or "",
            "destination": g.get("destination_place") or "", "location": "",
            "mentioned": [str(p) for p in g.get("places") or [] if p],
            "description": gold.get("summary") or "",
            "people": [str(p.get("name")) for p in g.get("people") or [] if p.get("name")],
            "doc_date_standard": "" if inferred else standard, "inferred_date_standard": standard if inferred else "",
            "ref_years": years or years_in(standard), "ref_date_source": "inferred" if inferred else "doc"}


def reference_date(ref: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Reference date interval of a PGP record (pairs row or :func:`reference_from_gold`).

    A gold record's ``ref_years`` (from ``year_ce_start/end``) take precedence; a pairs row uses the
    years in ``doc_date_standard``, else ``inferred_date_standard``.

    :param ref: Reference dict.
    :returns: ``{"lo", "hi", "source": "doc"|"inferred"}`` or None when there is no date.
    """
    if "ref_years" in ref:
        ys = list(ref["ref_years"] or [])
        return {"lo": min(ys), "hi": max(ys), "source": ref.get("ref_date_source", "doc")} if ys else None
    for field, source in (("doc_date_standard", "doc"), ("inferred_date_standard", "inferred")):
        years = years_in(ref.get(field))
        if years:
            return {"lo": min(years), "hi": max(years), "source": source}
    return None


def date_metrics(pred: Dict[str, Any], ref: Dict[str, Any]) -> Dict[str, Any]:
    """Date correctness of one catalogue record (probe-schema prediction).

    The prediction is the midpoint of ``date_year_ce`` .. ``date_year_ce_max``; the error is its
    distance to the reference interval (0 inside it).

    :param pred: Normalised prediction (:func:`normalise_catalogue`).
    :param ref: Reference dict.
    :returns: Reference interval, predicted point, error and ``date_ok2`` / ``date_ok10`` /
        ``date_abstained``.
    """
    rd = reference_date(ref)
    y0, y1 = _as_int(pred.get("date_year_ce")), _as_int(pred.get("date_year_ce_max"))
    point = None
    if y0 is not None:
        point = (y0 + y1) / 2 if (y1 is not None and y1 >= y0) else float(y0)
    out: Dict[str, Any] = {"ref_date": rd, "pred_year": y0, "pred_year_max": y1, "pred_point": point,
                           "date_err": None}
    if rd is not None and point is not None:
        out["date_err"] = max(0.0, rd["lo"] - point, point - rd["hi"])
    out["date_ok2"] = out["date_err"] is not None and out["date_err"] <= 2
    out["date_ok10"] = out["date_err"] is not None and out["date_err"] <= 10
    out["date_abstained"] = point is None
    return out


def strip_diacritics(text: str) -> str:
    """NFKD, drop combining marks and ʿ/ʾ-type apostrophes.

    :param text: Any text.
    :returns: Plain text.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[ʿʾ'’‘`ʻʼ]", "", text)


PLACE_ALIASES: Dict[str, Sequence[str]] = {
    "fustat": ("fustat", "fusṭāṭ", "old cairo", "misr", "miṣr", "masr", "fostat", "fusṭaṭ"),
    "qayrawan": ("qayrawan", "kairouan", "qairawan", "kairawan", "qayrawān"),
    "mahdiyya": ("al-mahdiyya", "mahdia", "mahdiya", "mahdiyya"),
    "alexandria": ("alexandria", "iskandariyya", "al-iskandariyya", "iskandariya", "alexandrie"),
    "jerusalem": ("jerusalem", "bayt al-maqdis", "al-quds", "quds", "bet ha-miqdash"),
    "aden": ("aden", "ʿadan", "adan"),
    "tyre": ("tyre", "ṣūr", "sur"),
    "ramla": ("ramla", "al-ramla", "ramle", "ramlah"),
    "damietta": ("damietta", "dimyāṭ", "dimyat"),
    "cairo": ("cairo", "al-qāhira", "qahira", "al-qahira", "new cairo"),
    "tripoli": ("tripoli", "aṭrābulus", "atrabulus", "tarabulus"),
    "palermo": ("palermo", "balarm", "bal(a)rm"),
    "sicily": ("sicily", "ṣiqilliya", "siqilliya", "siqilliyya"),
    "tinnis": ("tinnis", "tinnīs"),
    "busir": ("busir", "būṣīr", "abusir"),
    "mazara": ("mazara", "mazar", "māzar", "mazara del vallo", "māzara"),
    "aydhab": ("aydhab", "ʿaydhāb", "aidhab"),
    "ahwaz": ("ahwaz", "al-ahwāz", "ahvaz"),
}
#: City key → region key; a region-level answer for a city reference is scored ``region``.
PLACE_REGION: Dict[str, str] = {
    "fustat": "egypt", "cairo": "egypt", "alexandria": "egypt", "damietta": "egypt", "tinnis": "egypt",
    "busir": "egypt", "aydhab": "egypt", "aden": "yemen", "palermo": "sicily", "mazara": "sicily",
    "qayrawan": "ifriqiya", "mahdiyya": "ifriqiya", "jerusalem": "palestine", "ramla": "palestine",
    "tyre": "syria", "tripoli": "syria", "ahwaz": "khuzistan"}
_ALIAS_INDEX: Dict[str, str] = {}


def _place_norm(text: str) -> str:
    """String normalisation used by :func:`place_key`.

    :param text: Place name.
    :returns: Casefolded, diacritic-free, ``al-``-free, punctuation-free string.
    """
    text = strip_diacritics(text).casefold().strip()
    text = re.sub(r"^(al|el)[- ]", "", text)
    text = re.sub(r"\b(al|el)-", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def place_key(text: str) -> str:
    """Normalised key of a place string (casefold, no diacritics, no leading ``al-``, alias map).

    :param text: Place name.
    :returns: Canonical key.
    """
    if not _ALIAS_INDEX:
        for canon, aliases in PLACE_ALIASES.items():
            for alias in (canon,) + tuple(aliases):
                _ALIAS_INDEX[_place_norm(alias)] = canon
    norm = _place_norm(text)
    return _ALIAS_INDEX.get(norm, norm)


def place_candidates(pred: Any) -> Set[str]:
    """Canonical keys a predicted place string can stand for (whole string, parts, parentheticals).

    :param pred: Predicted place (None / ``null`` / ``unknown`` = abstained).
    :returns: Set of keys (empty when abstained).
    """
    if pred is None or not str(pred).strip() or str(pred).strip().lower() in {"null", "unknown", "none"}:
        return set()
    pred = str(pred)
    parts = {pred, re.sub(r"\(.*?\)", "", pred)}
    parts.update(re.findall(r"\((.*?)\)", pred))
    for piece in list(parts):
        parts.update(re.split(r",|/|;|\bor\b|\?", piece))
    return {place_key(p) for p in parts if p and p.strip() and place_key(p)}


def place_match(pred: Any, ref: str) -> bool:
    """Whether a predicted place matches the reference place (key equality or rapidfuzz ratio ≥ 90).

    :param pred: Predicted place.
    :param ref: Reference place (non-empty).
    :returns: True on a match.
    """
    rkey = place_key(ref)
    return any(c == rkey or fuzz.ratio(c, rkey) >= 90 for c in place_candidates(pred))


def place_metrics(pred_place: Any, ref_place: str) -> str:
    """Outcome of one origin / destination field.

    :param pred_place: Predicted place.
    :param ref_place: Reference place ('' = none recorded).
    :returns: ``correct``, ``region`` (only the region of the reference city), ``wrong``,
        ``abstained`` (reference exists) or ``noref_null`` / ``noref_stated`` (no reference).
    """
    cands = place_candidates(pred_place)
    if not ref_place:
        return "noref_stated" if cands else "noref_null"
    if not cands:
        return "abstained"
    if place_match(pred_place, ref_place):
        return "correct"
    region = PLACE_REGION.get(place_key(ref_place))
    return "region" if region and region in cands else "wrong"


_NAME_PARTICLES = {"b", "ibn", "ben", "bar", "bat", "bt", "bint", "bin"}


def name_norm(name: str) -> str:
    """Normalise a romanised personal name for fuzzy matching.

    Strips diacritics and ʿ/ʾ, casefolds, drops ibn/ben/bar/bat/bt./bint/b., maps abū/abi/abou/aba
    to ``abu``, drops ``al-``/``el-``/``l-`` articles and punctuation.

    :param name: Name.
    :returns: Space-joined tokens.
    """
    text = strip_diacritics(name or "").casefold()
    text = re.sub(r"\b(al|el|l)-", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    toks = []
    for tok in text.split():
        if tok in _NAME_PARTICLES:
            continue
        toks.append("abu" if tok in {"abu", "abi", "abou", "aba"} else tok)
    return " ".join(toks)


def people_recall(pred_names: Sequence[str], ref_names: Sequence[str], threshold: float = 80.0) -> Dict[str, Any]:
    """Recall of reference people among predicted names (rapidfuzz ``token_set_ratio``).

    :param pred_names: Predicted person names.
    :param ref_names: Reference person names.
    :param threshold: Minimum ``token_set_ratio`` for a match.
    :returns: ``{"n_ref", "n_found", "recall", "missed"}`` (recall None without references).
    """
    preds = [name_norm(n) for n in pred_names if n and name_norm(n)]
    found, missed = 0, []
    for ref in ref_names:
        rn = name_norm(ref)
        if rn and any(fuzz.token_set_ratio(rn, p) >= threshold for p in preds):
            found += 1
        else:
            missed.append(ref)
    n = len(ref_names)
    return {"n_ref": n, "n_found": found, "recall": (found / n) if n else None, "missed": missed}


def load_people_dictionary(people_csv: Path) -> Dict[str, str]:
    """PGP ``people.csv`` names with at least two name tokens, keyed by :func:`name_norm`.

    :param people_csv: Path to a PGP export's ``people.csv``.
    :returns: ``{normalised name: PGP name}``.
    """
    people = pd.read_csv(people_csv)
    out: Dict[str, str] = {}
    for name in people["name"].dropna():
        k = name_norm(str(name))
        if len(k.split()) >= 2:
            out[k] = str(name)
    return out


def dictionary_people(description: str, people_dict: Dict[str, str]) -> List[str]:
    """PGP people names found verbatim (normalised) in a description, longest matches only.

    :param description: PGP description.
    :param people_dict: Output of :func:`load_people_dictionary`.
    :returns: PGP names.
    """
    padded = f" {name_norm(description or '')} "
    hits = [k for k in people_dict if f" {k} " in padded]
    hits = [h for h in hits if not any(h != o and f" {h} " in f" {o} " for o in hits)]
    return [people_dict[h] for h in hits]


_RTL_WORD = re.compile(r"[א-תء-ي׳'״\"]{2,}")


@dataclass(frozen=True)
class SourceIndex:
    """Folded views of an input text for the grounding checks (built once per source)."""

    padded: str
    nospace: str
    skeletons: frozenset
    name_skeletons: frozenset

    @classmethod
    def of(cls, text: str) -> "SourceIndex":
        """Index a source text.

        :param text: Input (and context) text.
        :returns: The index: folded text with and without spaces, the ≥3-consonant skeletons of
            :func:`textnorm.rtl_skeletons` (prose candidates) and the ≥2-consonant skeletons of every
            Hebrew/Arabic-script word (name tokens such as Saʿīd / סעיד → ``sd``).
        """
        text = text or ""
        folded = " " + tn.strip_marks(text) + " "
        names = set()
        for w in _RTL_WORD.findall(text):
            for v in (w, w[1:] if w[:1] in "ובלדה" and len(w) > 3 else None):
                if v:
                    sk = tn.skeleton_rtl(v)
                    if len(sk) >= 2:
                        names.add(sk)
        return cls(padded=folded, nospace=folded.replace(" ", ""), skeletons=frozenset(tn.rtl_skeletons(text)),
                   name_skeletons=frozenset(names))


def candidate_grounded(cand: str, idx: SourceIndex) -> bool:
    """Whether one folded entity candidate is grounded in a source (the rule of :func:`textnorm.grounding`).

    :param cand: Folded candidate (from :func:`textnorm.entity_candidates` or :func:`textnorm.strip_marks`).
    :param idx: Source index.
    :returns: True when the candidate (or its form without a leading ``al``) occurs in the folded
        source, or its Latin consonant skeleton matches a Hebrew/Arabic-script word of the source.
    """
    alt = cand[2:] if cand.startswith("al") and len(cand) > 5 else cand
    if f" {cand}" in idx.padded or cand in idx.nospace or (alt != cand and alt in idx.nospace):
        return True
    if not cand.isdigit() and re.match(r"^[a-z]", cand):
        sk = tn.skeleton_latin(cand)
        if len(sk) >= 3 and sk in idx.skeletons:
            return True
    return False


def token_found(tok: str, idx: SourceIndex) -> bool:
    """One folded token found in a source: :func:`candidate_grounded`, else a ≥2-consonant skeleton match.

    The second test catches short names whose skeleton :func:`textnorm.grounding` ignores
    (Yosef / יוסף → ``sf``, Yaʿqūb / יעקב → ``kb``); a final ``s`` (a folded possessive,
    ``Avraham's`` → ``avrahams``) is retried without it.

    :param tok: Folded token.
    :param idx: Source index.
    :returns: True when found.
    """
    if candidate_grounded(tok, idx):
        return True
    if tok.isdigit() or not re.match(r"^[a-z]", tok):
        return False
    sk = tn.skeleton_latin(tok)
    if len(sk) >= 2 and sk in idx.name_skeletons:
        return True
    return tok.endswith("s") and len(tok) > 4 and token_found(tok[:-1], idx)


#: Calendar and language vocabulary that is not a "specific" (dates are scored by the date metric).
GENERIC_SPECIFICS: frozenset = frozenset("""
tishri tishre tishrei heshvan marheshvan marcheshvan kislev tevet teveth shevat shvat adar nisan iyyar iyar sivan
tammuz tamuz av elul muharram safar rabi jumada rajab shaban shaaban ramadan shawwal dhu qada qaada hijja hijjah
sunday monday tuesday wednesday thursday friday saturday sabbath shabbat sel seleucid am ah ce bce hijri anno
mundi rosh hodesh judaeo judeo arabic hebrew aramaic persian coptic greek latin
""".split())


def is_generic(cand: str) -> bool:
    """Whether a folded candidate is calendar or language vocabulary (every token in :data:`GENERIC_SPECIFICS`).

    :param cand: Folded candidate.
    :returns: True for generic vocabulary.
    """
    toks = cand.split()
    return bool(toks) and all(t in GENERIC_SPECIFICS for t in toks)


_NAME_STOP = {"abu", "umm", "ha", "the", "elder", "son", "daughter", "of", "rabbi", "rav", "r", "mar", "sitt",
              "shaykh", "sheikh", "al", "el", "kohen", "levi", "nasi", "haver", "rayyis", "unknown"}


def name_grounded(name: str, idx: SourceIndex) -> bool:
    """Whether a predicted person name is supported by the input text (deterministic proxy of the v1 judge).

    The name's significant tokens (≥3 letters, not particles or titles; a name made only of titles
    keeps them) count as found when their folded form occurs in the folded source or their Latin
    consonant skeleton (≥2 consonants) equals the skeleton of a Hebrew/Arabic-script word of the
    source. The name is grounded when at least half of its tokens are found. On the 10-18 judge cache
    (2,755 names) this agrees with the judge on 82 % of names and recovers 86 % of the names the
    judge found in the input; it is a proxy, not the judge.

    :param name: Predicted name.
    :param idx: Index of the input text.
    :returns: True when grounded (False for a name without tokens of ≥3 letters).
    """
    toks = [t for t in tn.strip_marks(name_norm(name)).split() if len(t) >= 3]
    sig = [t for t in toks if t not in _NAME_STOP] or toks
    if not sig:
        return False
    return sum(1 for t in sig if token_found(t, idx)) * 2 >= len(sig)


def people_grounding(names: Sequence[str], source: str) -> Dict[str, Any]:
    """Grounded share of predicted people names in the input text.

    :param names: Predicted names.
    :param source: Input text.
    :returns: ``{"n", "grounded", "share", "ungrounded"}`` (share None without names).
    """
    idx = SourceIndex.of(source)
    flags = [(n, name_grounded(n, idx)) for n in names if n and str(n).strip()]
    g = sum(1 for _, ok in flags if ok)
    return {"n": len(flags), "grounded": g, "share": (g / len(flags)) if flags else None,
            "ungrounded": [n for n, ok in flags if not ok][:12]}


def specifics_split(text: str, source: str, reference: str = "",
                    scholar_names: Iterable[str] = ()) -> Dict[str, Any]:
    """Specifics (names, years) of a generated text split by where they are supported.

    Candidates come from :func:`textnorm.entity_candidates` minus calendar / language vocabulary
    (:func:`is_generic`); each is ``grounded`` (in the input, :func:`token_found` per token, at least
    half of the tokens), ``in_ref_only`` (not in the input but in the reference, e.g. the PGP
    description) or ``in_neither``. Deterministic proxy of the 10-18 judge's "unsupported
    specifics": it cannot see derived facts (a CE year converted from a Seleucid date, a translated
    epithet), so it counts more than the judge did.

    :param text: Generated prose (summary, description, commentary).
    :param source: Input text the system saw (plus any context block).
    :param reference: Reference text (PGP description / gold summary); '' to skip the split.
    :param scholar_names: Folded scholar surnames (bibliographic names are not specifics).
    :returns: ``{"n_cand", "generic_skipped", "grounded", "in_ref_only", "in_neither", "grounded_share",
        "in_neither_list"}``; ``grounded_share`` is 1.0 without candidates.
    """
    raw = sorted(tn.entity_candidates(text or "", scholar_names))
    cands = [c for c in raw if not is_generic(c)]
    sidx, ridx = SourceIndex.of(source), SourceIndex.of(reference)

    def supported(cand: str, idx: SourceIndex) -> bool:
        """At least half of a candidate's tokens found in a source.

        :param cand: Folded candidate.
        :param idx: Source index.
        :returns: True when supported.
        """
        if candidate_grounded(cand, idx):
            return True
        toks = cand.split()
        return sum(1 for t in toks if token_found(t, idx)) * 2 >= len(toks)

    g, r, neither = 0, 0, []
    for c in cands:
        if supported(c, sidx):
            g += 1
        elif reference and supported(c, ridx):
            r += 1
        else:
            neither.append(c)
    n = len(cands)
    return {"n_cand": n, "generic_skipped": len(raw) - n, "grounded": g, "in_ref_only": r,
            "in_neither": len(neither), "grounded_share": (g / n) if n else 1.0, "in_neither_list": neither[:12]}


def catalogue_metrics(pred_obj: Optional[Dict[str, Any]], ref: Dict[str, Any], source: str = "",
                      ref_people: Optional[Dict[str, List[str]]] = None,
                      scholar_names: Iterable[str] = ()) -> Dict[str, Any]:
    """Field checks of one catalogue record (any schema) against a reference.

    :param pred_obj: Parsed prediction (None = unparseable).
    :param ref: Pairs row or :func:`reference_from_gold` dict (``type``, dates, ``origin``,
        ``destination``, ``location``, ``mentioned``, ``description``).
    :param source: Input text the system saw (people grounding, summary specifics).
    :param ref_people: Reference people lists by method (e.g. ``{"dict": [...], "judge": [...],
        "gold": [...]}``); each yields ``people_<method>`` recall.
    :param scholar_names: Folded scholar surnames.
    :returns: Flat dict of outcomes (``parse_ok``, ``type_ok``, date fields, place outcomes,
        mentioned places, people recall / grounding, summary specifics, ``source_names``).
    """
    pred = normalise_catalogue(pred_obj)
    ref_type = ref.get("type") or ""
    pred_type = pred.get("document_type")
    out: Dict[str, Any] = {"parse_ok": pred_obj is not None, "ref_type": ref_type, "pred_type": pred_type,
                           "type_ok": (pred_type == ref_type) if ref_type else (pred_type in (None, "Unknown"))}
    out.update(date_metrics(pred, ref))
    dt = pred.get("date_text")
    out["pred_date_text"] = dt if (dt and str(dt).strip().lower() not in {"null", "none"}) else None
    ref_origin, ref_dest = ref.get("origin") or "", ref.get("destination") or ""
    out["origin_outcome"] = place_metrics(pred.get("origin_place"), ref_origin)
    out["destination_outcome"] = place_metrics(pred.get("destination_place"), ref_dest)
    out["origin_or_location_outcome"] = place_metrics(pred.get("origin_place"), ref_origin or ref.get("location") or "")
    pred_places = [str(p) for p in pred.get("places") or [] if p]
    pred_places += [str(pred.get(k)) for k in ("origin_place", "destination_place") if pred.get(k)]
    mentioned = [str(m) for m in ref.get("mentioned") or [] if m]
    out["mentioned_places_n"] = len(mentioned)
    out["mentioned_places_found"] = sum(1 for m in mentioned if any(place_match(p, m) for p in pred_places))
    names = [str(p.get("name", "")) for p in pred.get("people") or [] if isinstance(p, dict) and p.get("name")]
    out["pred_people"] = names
    for method, ref_names in (ref_people or {}).items():
        out[f"people_{method}"] = None if ref_names is None else people_recall(names, ref_names)
    out["people_grounding"] = people_grounding(names, source) if source else None
    summary = str(pred.get("summary") or "")
    out["pred_summary"] = summary
    out["summary_specifics"] = specifics_split(summary, source, ref.get("description") or "", scholar_names) \
        if source else None
    out["source_names"] = tn.has_source_names(json.dumps(pred_obj, ensure_ascii=False)) if pred_obj else False
    return out


# =================================================================================== describe
def describe_metrics(hyp_raw: Optional[str], ref: str, source: str = "",
                     scholar_names: Iterable[str] = ()) -> Dict[str, Any]:
    """Metrics of a free-text description / commentary against a reference text.

    :param hyp_raw: Raw model output.
    :param ref: Reference text (PGP description or the book's commentary).
    :param source: Input text the system saw.
    :param scholar_names: Folded scholar surnames.
    :returns: chrF++, length ratio, ``empty`` / ``repetition`` / ``degenerate`` / ``source_names`` flags,
        ``specifics`` (:func:`specifics_split`) and ``_hyp`` / ``_ref``.
    """
    hyp = (hyp_raw or "").strip()
    ref = (ref or "").strip()
    rep = repetition_stats(hyp)
    empty = not hyp
    repetition = (not empty) and is_repetitive(rep)
    return {"chrfpp": CHRFPP.sentence_score(hyp, [ref]).score, "hyp_chars": len(hyp), "ref_chars": len(ref),
            "len_ratio": (len(hyp) / len(ref)) if ref else None, "empty": empty, "repetition": repetition,
            "degenerate": empty or repetition, "source_names": tn.has_source_names(hyp),
            "specifics": specifics_split(hyp, source, ref, scholar_names), "_hyp": hyp, "_ref": ref}


# ================================================================================== relations
PLACE_PREDICATES = {"origin", "destination", "mentions_place", "possibly_mentions_place"}
PERSON_PREDICATES = {"sender", "recipient", "mentions_person"}


def parse_relations(text: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """Relations list of an ``extract_relations`` answer.

    :param text: Raw model output (``{"relations": [...]}``; a bare list is accepted).
    :returns: The list of relation dicts, or None when unparseable.
    """
    try:
        obj: Any = json.loads(_strip_fences(text or ""))
    except json.JSONDecodeError:
        obj = parse_json_lenient(text, "relations")
    if isinstance(obj, dict):
        obj = obj.get("relations")
    if not isinstance(obj, list):
        return None
    return [r for r in obj if isinstance(r, dict)]


def _rel_match(pred: Dict[str, Any], gold: Dict[str, Any]) -> bool:
    """Whether a predicted relation matches a gold one (same predicate; normalised subject / object).

    :param pred: Predicted relation.
    :param gold: Gold relation.
    :returns: True on a match.
    """
    pp, gp = str(pred.get("predicate", "")).strip(), str(gold.get("predicate", "")).strip()
    if pp != gp:
        return False

    def same_person(a: Any, b: Any) -> bool:
        """Name equality after normalisation (``THIS_DOCUMENT`` exact).

        :param a: First name.
        :param b: Second name.
        :returns: True when equal or token_set_ratio ≥ 85.
        """
        a, b = str(a or ""), str(b or "")
        if "THIS_DOCUMENT" in (a, b):
            return a == b
        na, nb = name_norm(a), name_norm(b)
        return bool(na and nb) and (na == nb or fuzz.token_set_ratio(na, nb) >= 85)

    if not same_person(pred.get("subject"), gold.get("subject")):
        return False
    po, go = pred.get("object"), gold.get("object")
    if gp in PLACE_PREDICATES:
        return bool(str(go or "").strip()) and place_match(po, str(go))
    if gp in PERSON_PREDICATES:
        return same_person(po, go)
    return strip_diacritics(str(po or "")).casefold().strip() == strip_diacritics(str(go or "")).casefold().strip()


def relations_prf(pred: Optional[Sequence[Dict[str, Any]]], gold: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Greedy one-to-one matching of predicted to gold relations.

    :param pred: Predicted relations (None = unparseable, counted as no predictions).
    :param gold: Gold relations.
    :returns: ``{"tp", "fp", "fn", "precision", "recall", "f1", "parse_ok"}`` (None ratios on 0 denominators).
    """
    pred_l = list(pred or [])
    used: Set[int] = set()
    tp = 0
    for p in pred_l:
        for j, g in enumerate(gold):
            if j not in used and _rel_match(p, g):
                used.add(j)
                tp += 1
                break
    fp, fn = len(pred_l) - tp, len(gold) - tp
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else (0.0 if (prec is not None and rec is not None) else None)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1, "parse_ok": pred is not None}


def micro_prf(cells: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Micro-averaged precision / recall / F1 over :func:`relations_prf` results.

    :param cells: Per-document results.
    :returns: ``{"docs", "tp", "fp", "fn", "precision", "recall", "f1", "parse_ok"}``.
    """
    cells = list(cells)
    tp, fp, fn = (sum(c[k] for c in cells) for k in ("tp", "fp", "fn"))
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else 0.0
    return {"docs": len(cells), "tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1,
            "parse_ok": sum(1 for c in cells if c.get("parse_ok"))}


# ================================================================================ canon (E8)
_HE_POINTS = re.compile(r"[\u0591-\u05C7]")
_DASHES = re.compile(r"[\u2010-\u2015\u2212]")
_CANON_LOCUS = re.compile(r"^(?P<s1>\d+[ab]?)(?::(?P<l1>\d+))?(?:-(?:(?P<s2>\d+[ab]?):)?(?P<l2>\d+[ab]?))?$")


def canon_norm(text: Any) -> str:
    """Normalise a canon string for matching: NFC, Hebrew points removed, whitespace collapsed.

    :param text: Text (non-strings become '').
    :returns: Normalised text.
    """
    if not isinstance(text, str):
        return ""
    return " ".join(_HE_POINTS.sub("", unicodedata.normalize("NFC", text)).split())


def norm_ref(ref: Any) -> str:
    """Normalise a Sefaria-style reference: case-folded, dashes unified, whitespace collapsed, no space around ``-``.

    :param ref: Reference (``Niddah 27b:16-28a:7``).
    :returns: Normalised reference ('' for a non-string).
    """
    if not isinstance(ref, str):
        return ""
    t = " ".join(_DASHES.sub("-", ref).casefold().split())
    return re.sub(r"\s*-\s*", "-", t)


def section_key(section: str) -> Tuple[int, int]:
    """Order key of a section: daf + amud (``27b`` → (27, 1)) or a chapter number (``4`` → (4, 0)).

    :param section: Section label.
    :returns: ``(number, amud)``.
    """
    m = re.match(r"^(\d+)([ab]?)$", section)
    if not m:
        raise ValueError(f"not a section label: {section!r}")
    return int(m.group(1)), (1 if m.group(2) == "b" else 0)


def parse_canon_ref(ref: Any) -> Optional[Dict[str, Any]]:
    """Parse ``<title> <section>[:<line>][-[<section>:]<line>]`` (the canon builder's reference format).

    :param ref: Reference (``Niddah 27b:16-28a:7``, ``Yevamot 110b:6-12``, ``Mishneh Torah, Foundations of the
        Torah 4:11``, ``Pirkei Avot 1:14-2:2``).
    :returns: ``{"title", "start", "end"}`` with sections as :func:`section_key` tuples, or None when unparseable.
    """
    t = norm_ref(ref)
    if " " not in t:
        return None
    title, locus = t.rsplit(" ", 1)
    m = _CANON_LOCUS.match(locus)
    if not m:
        return None
    try:
        s1 = section_key(m.group("s1"))
        if m.group("s2"):
            s2 = section_key(m.group("s2"))
        elif m.group("l2") and m.group("l1") is None:  # "27a-28b" or "4-5": a range of whole sections
            s2 = section_key(m.group("l2"))
        else:
            s2 = s1
    except ValueError:
        return None
    return {"title": title, "start": s1, "end": max(s1, s2)}


def same_section(pred_ref: Any, gold_ref: Any) -> bool:
    """Whether a predicted reference names the gold title and a section range overlapping the gold one.

    :param pred_ref: Predicted reference.
    :param gold_ref: Gold reference.
    :returns: True on the same title with overlapping daf-amud / chapter ranges.
    """
    a, b = parse_canon_ref(pred_ref), parse_canon_ref(gold_ref)
    if a is None or b is None or a["title"] != b["title"]:
        return False
    return a["start"] <= b["end"] and b["start"] <= a["end"]


def _words(text: Any) -> List[str]:
    """Words of a normalised canon string.

    :param text: Text.
    :returns: Words.
    """
    return canon_norm(text).split()


def canon_identify_metrics(hyp_raw: Optional[str], gold: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic metrics of one ``canon_identify`` answer.

    :param hyp_raw: Raw model output (a JSON object with work, title, he_title, ref, he_ref, anchor).
    :param gold: The gold target object.
    :returns: ``json_ok``, ``work_ok``, ``title_ok``, ``ref_exact``, ``same_section``, ``anchor_f1`` (word bag F1
        against the gold anchor), ``anchor_words`` and ``over_quoted`` (anchor longer than :data:`CANON_ANCHOR_WORDS`).
    """
    pred = parse_json_lenient(hyp_raw, required_key="ref") or {}
    title_ok = bool(pred) and norm_ref(pred.get("title")) == norm_ref(gold.get("title"))
    pw, gw = _words(pred.get("anchor")), _words(gold.get("anchor"))
    common = sum((Counter(pw) & Counter(gw)).values())
    f1 = 2 * common / (len(pw) + len(gw)) if (pw or gw) else 1.0
    return {"json_ok": bool(pred), "work_ok": bool(pred) and norm_ref(pred.get("work")) == norm_ref(gold.get("work")),
            "title_ok": title_ok, "ref_exact": bool(pred) and norm_ref(pred.get("ref")) == norm_ref(gold.get("ref")),
            "same_section": bool(pred) and same_section(pred.get("ref"), gold.get("ref")),
            "anchor_f1": round(f1, 4), "anchor_words": len(pw), "over_quoted": len(pw) > CANON_ANCHOR_WORDS}


def variant_keys(variants: Any, mode: str = "typed") -> Counter:
    """Multiset of variant identities of an apparatus.

    :param variants: List of ``{"type", "fragment", "canon", "pos"}`` (non-dicts are skipped).
    :param mode: ``typed`` = (type, fragment, canon); ``untyped`` = (fragment, canon); ``strict`` = typed + pos.
    :returns: Counter of identities.
    :raises ValueError: On an unknown mode.
    """
    if mode not in ("typed", "untyped", "strict"):
        raise ValueError(f"unknown variant key mode {mode!r}")
    out: Counter = Counter()
    for v in variants if isinstance(variants, list) else []:
        if not isinstance(v, dict):
            continue
        frag, canon = canon_norm(v.get("fragment")), canon_norm(v.get("canon"))
        typ = str(v.get("type") or "").strip().lower()
        typ = CANON_TYPE_ALIASES.get(typ, typ)
        key: Tuple[str, ...] = (frag, canon) if mode == "untyped" else (typ, frag, canon)
        if mode == "strict":
            key = key + (str(v.get("pos") or "").strip(),)
        out[key] += 1
    return out


def multiset_prf(pred: Counter, gold: Counter) -> Dict[str, Any]:
    """Precision / recall / F1 of a predicted multiset against a gold multiset.

    :param pred: Predicted identities.
    :param gold: Gold identities.
    :returns: ``{"tp", "fp", "fn", "precision", "recall", "f1"}`` (ratios None on a 0 denominator; F1 of two empty
        sets is 1.0).
    """
    tp = sum((pred & gold).values())
    fp, fn = sum(pred.values()) - tp, sum(gold.values()) - tp
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    if not (tp + fp + fn):
        f1: Optional[float] = 1.0
    else:
        f1 = 2 * tp / (2 * tp + fp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


def canon_diff_metrics(hyp_raw: Optional[str], gold: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic metrics of one ``canon_diff`` answer against the alignment's apparatus.

    :param hyp_raw: Raw model output (JSON object with canonical_ref, agreement_ratio, lacuna_words, variants, summary).
    :param gold: The gold target object.
    :returns: ``json_ok``, ``ref_exact``, ``typed`` / ``untyped`` / ``strict`` (:func:`multiset_prf`), ``by_type``
        (typed counts per gold / predicted type), ``agreement_abs_err``, ``lacuna_abs_err``, ``n_pred``, ``n_gold``.
    """
    pred = parse_json_lenient(hyp_raw, required_key="variants")
    pv = (pred or {}).get("variants") or []
    gv = gold.get("variants") or []
    out: Dict[str, Any] = {"json_ok": pred is not None,
                           "ref_exact": pred is not None and norm_ref(pred.get("canonical_ref")) == norm_ref(gold.get("canonical_ref")),
                           "n_pred": len(pv) if isinstance(pv, list) else 0, "n_gold": len(gv)}
    for mode in ("typed", "untyped", "strict"):
        out[mode] = multiset_prf(variant_keys(pv, mode), variant_keys(gv, mode))
    pk, gk = variant_keys(pv), variant_keys(gv)
    out["by_type"] = {t: multiset_prf(Counter({k: n for k, n in pk.items() if k[0] == t}),
                                      Counter({k: n for k, n in gk.items() if k[0] == t})) for t in CANON_VARIANT_TYPES}
    for key, name in (("agreement_ratio", "agreement_abs_err"), ("lacuna_words", "lacuna_abs_err")):
        a, b = (pred or {}).get(key), gold.get(key)
        out[name] = (abs(float(a) - float(b)) if isinstance(a, (int, float)) and not isinstance(a, bool)
                     and isinstance(b, (int, float)) else None)
    return out


def aggregate_canon_identify(rs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """E8 identify accuracies over scored rows.

    :param rs: :func:`canon_identify_metrics` results (via :func:`score_output`).
    :returns: ``n`` and the rate of every boolean field, mean anchor F1.
    """
    out: Dict[str, Any] = {"n": len(rs)}
    for k in ("json_ok", "work_ok", "title_ok", "ref_exact", "same_section", "over_quoted"):
        out[k] = _rate(rs, k)
    out["anchor_f1"] = mean(r.get("anchor_f1") for r in rs)
    return out


def aggregate_canon_diff(rs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """E8 apparatus scores over scored rows: micro P / R / F1 per key mode and per type, mean errors.

    :param rs: :func:`canon_diff_metrics` results (via :func:`score_output`).
    :returns: ``n``, ``json_ok``, ``ref_exact``, ``typed`` / ``untyped`` / ``strict`` and ``by_type`` micro scores,
        mean ``agreement_abs_err`` / ``lacuna_abs_err``.
    """
    def micro(cells: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """Micro P / R / F1 of :func:`multiset_prf` cells.

        :param cells: Cells.
        :returns: Micro scores.
        """
        cells = list(cells)
        tp, fp, fn = (sum(c[k] for c in cells) for k in ("tp", "fp", "fn"))
        return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if (tp + fp) else None,
                "recall": tp / (tp + fn) if (tp + fn) else None,
                "f1": 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else None}

    out: Dict[str, Any] = {"n": len(rs), "json_ok": _rate(rs, "json_ok"), "ref_exact": _rate(rs, "ref_exact")}
    for mode in ("typed", "untyped", "strict"):
        out[mode] = micro(r[mode] for r in rs)
    out["by_type"] = {t: micro(r["by_type"][t] for r in rs) for t in CANON_VARIANT_TYPES}
    out["agreement_abs_err"] = mean(r.get("agreement_abs_err") for r in rs)
    out["lacuna_abs_err"] = mean(r.get("lacuna_abs_err") for r in rs)
    return out


# ================================================================================ v0_2 rows
def document_of(row: Dict[str, Any]) -> str:
    """The ``<document>`` text of a v0_2 row's user turn (the input the model sees).

    :param row: A family row.
    :returns: The document text ('' when absent).
    """
    user = next((m["content"] for m in row.get("messages", []) if m.get("role") == "user"), "")
    m = _DOC_RE.search(user)
    return m.group(1) if m else ""


def gold_of(row: Dict[str, Any]) -> str:
    """The assistant (gold) turn of a v0_2 row.

    :param row: A family row.
    :returns: The gold text.
    """
    return next((m["content"] for m in row.get("messages", []) if m.get("role") == "assistant"), "")


def score_output(row: Dict[str, Any], hyp: Optional[str], scholar_names: Iterable[str] = ()) -> Dict[str, Any]:
    """Score one model output against a v0_2 family row (gold = the row's assistant turn).

    :param row: Family row (``meta.family`` or ``meta.task`` picks the scorer).
    :param hyp: Raw model output.
    :param scholar_names: Folded scholar surnames.
    :returns: Scored dict with ``row_id``, ``family``, ``condition``, ``lang_class``, ``split``,
        ``pgpid`` and the family's metrics.
    :raises ValueError: On a family without a scorer.
    """
    meta = row.get("meta", {})
    family = meta.get("family") or meta.get("task")
    source, gold = document_of(row), gold_of(row)
    out: Dict[str, Any] = {k: meta.get(k) for k in ("row_id", "pgpid", "condition", "lang_class", "split",
                                                       "read_model", "input_cer", "noise_band", "reader_exposed")}
    out["family"] = family
    if family in ("translate_en", "translate_he"):
        out.update(translation_metrics(hyp, gold, family, source))
    elif family == "catalogue_json":
        gold_obj = parse_json_lenient(gold) or {}
        pred, strict = parse_catalogue(hyp)
        ref = reference_from_gold(gold_obj)
        out.update(catalogue_metrics(pred, ref, source, {"gold": ref["people"]}, scholar_names))
        out["json_strict"] = strict
    elif family in ("describe_pgp", "analyse_document"):
        out.update(describe_metrics(hyp, gold, source, scholar_names))
    elif family == "extract_relations":
        gold_rel = parse_relations(gold) or []
        out.update(relations_prf(parse_relations(hyp), gold_rel))
    elif family == "canon_identify":
        out.update(canon_identify_metrics(hyp, strict_json(gold) or {}))
        out["work"] = meta.get("work")
    elif family == "canon_diff":
        out.update(canon_diff_metrics(hyp, strict_json(gold) or {}))
        out["work"] = meta.get("work")
    else:
        raise ValueError(f"no scorer for family {family!r}")
    return out


def mean(xs: Iterable[Optional[float]]) -> Optional[float]:
    """Mean of the non-None values.

    :param xs: Values.
    :returns: Mean or None.
    """
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def _rate(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    """Share of rows whose boolean field is true.

    :param rows: Scored rows.
    :param key: Field name.
    :returns: The share or None for no rows.
    """
    return (sum(1 for r in rows if r.get(key)) / len(rows)) if rows else None


def aggregate_catalogue_cell(rs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Catalogue metrics of one cell of scored rows.

    :param rs: Rows scored with :func:`catalogue_metrics`.
    :returns: Cell dict (JSON ok / strict, type, dates, places, people recall and grounding, specifics).
    """
    withref = [r for r in rs if r["ref_date"] is not None]
    noref = [r for r in rs if r["ref_date"] is None]

    def recall(field: str) -> Dict[str, Any]:
        """Micro recall of one people-recall field.

        :param field: ``people_<method>``.
        :returns: ``{"docs", "n_ref", "n_found", "recall"}``.
        """
        got = [r[field] for r in rs if r.get(field) and r[field]["n_ref"]]
        n_ref, n_found = sum(g["n_ref"] for g in got), sum(g["n_found"] for g in got)
        return {"docs": len(got), "n_ref": n_ref, "n_found": n_found, "recall": (n_found / n_ref) if n_ref else None}

    def places(field: str) -> Dict[str, int]:
        """Outcome counts of one place field.

        :param field: Outcome field.
        :returns: ``{outcome: count}``.
        """
        return dict(Counter(r[field] for r in rs))

    pg = [r["people_grounding"] for r in rs if r.get("people_grounding")]
    sp = [r["summary_specifics"] for r in rs if r.get("summary_specifics")]
    people_fields = sorted({k for r in rs for k in r if k.startswith("people_") and k not in ("people_grounding",)})
    cell: Dict[str, Any] = {
        "n": len(rs), "json_ok": _rate(rs, "parse_ok"), "json_strict": _rate(rs, "json_strict"),
        "type_acc": _rate(rs, "type_ok"), "date_n_ref": len(withref),
        "date_n_ref_doc": sum(1 for r in withref if r["ref_date"]["source"] == "doc"),
        "date_acc2": _rate(withref, "date_ok2"), "date_acc10": _rate(withref, "date_ok10"),
        "date_acc2_doc": _rate([r for r in withref if r["ref_date"]["source"] == "doc"], "date_ok2"),
        "date_acc2_inferred": _rate([r for r in withref if r["ref_date"]["source"] == "inferred"], "date_ok2"),
        "date_abstain_with_ref": _rate(withref, "date_abstained"),
        "date_n_noref": len(noref),
        "dated_without_ref": mean((0.0 if r["date_abstained"] else 1.0) for r in noref),
        "origin": places("origin_outcome"), "destination": places("destination_outcome"),
        "origin_or_location": places("origin_or_location_outcome"),
        "mentioned_places_n": sum(r["mentioned_places_n"] for r in rs),
        "mentioned_places_recall": (sum(r["mentioned_places_found"] for r in rs)
                                    / sum(r["mentioned_places_n"] for r in rs))
        if sum(r["mentioned_places_n"] for r in rs) else None,
        "pred_people_per_doc": mean(len(r["pred_people"]) for r in rs),
        "people_grounded_n": sum(g["n"] for g in pg),
        "people_grounded_share": (sum(g["grounded"] for g in pg) / sum(g["n"] for g in pg))
        if sum(g["n"] for g in pg) else None,
        "specifics_per_summary": mean(s["n_cand"] for s in sp),
        "summary_grounded_share": mean(s["grounded_share"] for s in sp),
        "in_ref_only_per_summary": mean(s["in_ref_only"] for s in sp),
        "in_neither_per_summary": mean(s["in_neither"] for s in sp),
        "scrub_rows": sum(1 for r in rs if r.get("source_names")),
    }
    for f in people_fields:
        cell[f] = recall(f)
    return cell


def _row_stats(rs: Sequence[Dict[str, Any]], key: str = "_st") -> np.ndarray:
    """Stack the cached per-row chrF++ statistics of scored translation rows.

    :param rs: Rows from :func:`translation_metrics`.
    :param key: ``_st`` (v2 cleaning) or ``_st_v1``.
    :returns: Array of shape ``(n_rows, n_stats)``.
    """
    return np.vstack([r[key] for r in rs]) if rs else np.zeros((0, 1), dtype=np.int64)


def aggregate_translation_cell(rs: Sequence[Dict[str, Any]], n_boot: int = N_BOOT, seed: int = SEED) -> Dict[str, Any]:
    """Translation metrics of one cell of scored rows (corpus chrF++ v2 with CI, v1 corpus, chrF, BLEU, flags).

    :param rs: Rows scored with :func:`translation_metrics` (carrying the private ``_`` fields).
    :param n_boot: Bootstrap resamples.
    :param seed: RNG seed.
    :returns: Cell dict.
    """
    ci = bootstrap_stats(_row_stats(rs), n_boot=n_boot, seed=seed)
    v1 = _score_from_stats(CHRFPP, _row_stats(rs, "_st_v1").sum(axis=0)) if rs else None
    hyps, refs = [r["_hyp"] for r in rs], [r["_ref"] for r in rs]
    chrf = CHRF_PLAIN.corpus_score(hyps, [refs]).score if rs else None
    bleu = BLEU_CORPUS.corpus_score(hyps, [refs]).score if rs else None
    ratios = [r["len_ratio"] for r in rs if r["len_ratio"] is not None]
    return {
        "n": len(rs), "corpus_chrfpp": ci["score"], "ci_lo": ci["lo"], "ci_hi": ci["hi"],
        "corpus_chrfpp_v1": v1, "corpus_chrf": chrf, "corpus_bleu": bleu,
        "doc_chrfpp_mean": mean(r["chrfpp"] for r in rs),
        "doc_chrfpp_median": float(statistics.median([r["chrfpp"] for r in rs])) if rs else None,
        "len_ratio_median": float(statistics.median(ratios)) if ratios else None,
        **{k: sum(1 for r in rs if r.get(k)) for k in ("empty", "wrong_script", "len_gt2", "len_lt05",
                                                        "repetition", "copy_through", "degenerate", "source_names")},
        "degenerate_rate": _rate(rs, "degenerate"),
        "rows_meta_strict": sum(1 for r in rs if r.get("meta_lines_strict")),
        "rows_meta_loose": sum(1 for r in rs if r.get("meta_lines_loose")),
    }


def aggregate_scored(scored: Sequence[Dict[str, Any]], keys: Sequence[str] = ("family", "condition", "lang_class"),
                     n_boot: int = N_BOOT, seed: int = SEED) -> List[Dict[str, Any]]:
    """Aggregate :func:`score_output` rows per cell.

    :param scored: Scored rows.
    :param keys: Grouping fields.
    :param n_boot: Bootstrap resamples (translation cells).
    :param seed: RNG seed.
    :returns: One dict per cell (the keys plus the family's aggregate metrics).
    """
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in scored:
        groups[tuple(r.get(k) for k in keys)].append(r)
    out = []
    for gk, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        family = rs[0]["family"]
        cell = dict(zip(keys, gk))
        if family in ("translate_en", "translate_he"):
            cell.update(aggregate_translation_cell(rs, n_boot, seed))
        elif family == "catalogue_json":
            cell.update(aggregate_catalogue_cell(rs))
        elif family == "extract_relations":
            cell.update(micro_prf(rs))
        elif family == "canon_identify":
            cell.update(aggregate_canon_identify(rs))
        elif family == "canon_diff":
            cell.update(aggregate_canon_diff(rs))
        else:
            sp = [r["specifics"] for r in rs]
            cell.update({"n": len(rs), "doc_chrfpp_mean": mean(r["chrfpp"] for r in rs),
                         "degenerate": sum(1 for r in rs if r["degenerate"]), "scrub_rows": sum(1 for r in rs if r["source_names"]),
                         "grounded_share": mean(s["grounded_share"] for s in sp),
                         "in_ref_only_per_doc": mean(s["in_ref_only"] for s in sp),
                         "in_neither_per_doc": mean(s["in_neither"] for s in sp)})
        out.append(cell)
    return out


# ============================================================================== probe rescore
def discover_systems(outputs_dir: Path, arg: str = "all") -> List[str]:
    """Systems with an ``<outputs_dir>/<system>.jsonl`` file.

    :param outputs_dir: Probe outputs directory.
    :param arg: ``all`` or a comma-separated list.
    :returns: System labels.
    :raises FileNotFoundError: When a named system has no output file.
    """
    if arg == "all":
        return sorted(p.stem for p in Path(outputs_dir).glob("*.jsonl"))
    names = [s.strip() for s in arg.split(",") if s.strip()]
    for name in names:
        if not (Path(outputs_dir) / f"{name}.jsonl").exists():
            raise FileNotFoundError(f"no outputs for system {name!r}")
    return names


def probe_row_key(row: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    """Resume key of a probe output row.

    :param row: Output row.
    :returns: ``(sample_set, pgpid, task, condition, system)``.
    """
    return (row["sample_set"], str(row["pgpid"]), row["task"], row["condition"], row["system"])


def load_outputs(outputs_dir: Path, system: str, extra_dirs: Sequence[Path] = ()) -> List[Dict[str, Any]]:
    """Successful probe output rows of one system, the last successful row per key.

    :param outputs_dir: Main outputs directory.
    :param system: System label.
    :param extra_dirs: Further directories whose ``<system>.jsonl`` is read after the main file.
    :returns: Rows with ``error`` null, deduplicated by :func:`probe_row_key`.
    """
    best: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for path in [Path(outputs_dir) / f"{system}.jsonl"] + [Path(d) / f"{system}.jsonl" for d in extra_dirs]:
        if not path.exists():
            continue
        for row in _rows.read_jsonl(path):
            if row.get("error") is None:
                best[probe_row_key(row)] = row
    return list(best.values())


def load_exclusions(path: Optional[Path]) -> Dict[Tuple[str, str], List[str]]:
    """Excluded ``(sample_set, pgpid)`` rows and their reasons (``exclusions_v1.json`` format).

    :param path: Exclusions JSON (``{"entries": [{"sample_set", "pgpid", "reasons"}]}``) or None.
    :returns: ``{(sample_set, pgpid): [reason, ...]}``.
    """
    if not path:
        return {}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return {(e["sample_set"], str(e["pgpid"])): list(e["reasons"]) for e in data["entries"]}


def load_references(pairs_path: Path, pgpids: Set[str]) -> Dict[str, Dict[str, Any]]:
    """Stream a pairs file and keep the rows of the given documents.

    :param pairs_path: ``pgp_parallel`` JSONL.
    :param pgpids: pgpids to keep.
    :returns: ``{pgpid: pairs row}``.
    """
    return {str(r["pgpid"]): r for r in _rows.read_jsonl(pairs_path) if str(r["pgpid"]) in pgpids}


def sample_entries(samples: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Probe sample entries keyed by ``(sample_set, pgpid)``, each with its set's ``target``.

    :param samples: Parsed ``samples.json``.
    :returns: Entry dicts.
    """
    out = {}
    for name, spec in samples["sets"].items():
        for entry in spec["entries"]:
            out[(name, str(entry["pgpid"]))] = dict(entry, target=spec.get("target"))
    return out


def pick_read(ref: Dict[str, Any], prefer: str = "a50") -> Optional[Dict[str, Any]]:
    """The machine read the probe used for the ``read`` condition.

    :param ref: Pairs row.
    :param prefer: Preferred reader.
    :returns: The read dict or None.
    """
    reads = ref.get("reads") or []
    return next((r for r in reads if r.get("model") == prefer), reads[0] if reads else None)


def probe_source_text(condition: str, ref: Dict[str, Any], read_model: str = "a50") -> str:
    """Source text of a probe condition (re-derived; diagnostics only).

    :param condition: ``edition``, ``read``, ``translit`` or ``translit_plain``.
    :param ref: Pairs row.
    :param read_model: Preferred reader for ``read``.
    :returns: The text ('' when unavailable).
    """
    if condition == "edition":
        return ref.get("edition_text") or ""
    if condition == "translit":
        return tn.transliterate_ar_to_ja(ref.get("edition_text") or "")
    if condition == "translit_plain":
        return tn.transliterate_ar_to_ja_plain(ref.get("edition_text") or "")
    if condition == "read":
        read = pick_read(ref, read_model)
        return read["text"] if read else ""
    return ""


def load_ref_people_cache(cache_path: Optional[Path], refs: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """Reference people names from the v1 judge cache (read only), where the description is unchanged.

    :param cache_path: ``judge_cache.jsonl`` of the 10-18 probe (None / missing = empty).
    :param refs: Pairs rows by pgpid (their description must hash to the cached fingerprint).
    :returns: ``{pgpid: names}``.
    """
    if not cache_path or not Path(cache_path).exists():
        return {}
    out: Dict[str, List[str]] = {}
    for rec in _rows.read_jsonl(cache_path):
        pid = str(rec.get("pgpid", ""))
        if rec.get("kind") != "ref_people" or pid not in refs:
            continue
        desc = refs[pid].get("description") or ""
        fp = hashlib.sha1(f"{JUDGE_VERSION_V1}|{desc}".encode("utf-8")).hexdigest()
        if rec.get("fingerprint") == fp:
            out[pid] = [str(n) for n in (rec.get("result") or {}).get("people", []) if str(n).strip()]
    return out


def score_probe_row(row: Dict[str, Any], ref: Dict[str, Any], entry: Dict[str, Any],
                    people_dict: Dict[str, str], judge_people: Dict[str, List[str]],
                    scholar_names: Iterable[str] = ()) -> Dict[str, Any]:
    """Score one stored probe output row (translation or catalogue).

    :param row: Probe output row.
    :param ref: Its pairs row.
    :param entry: Its sample entry.
    :param people_dict: :func:`load_people_dictionary` output.
    :param judge_people: :func:`load_ref_people_cache` output.
    :param scholar_names: Folded scholar surnames.
    :returns: Scored row (``kind`` = ``translate`` or ``catalogue``).
    """
    read_model = entry.get("read_model") or "a50"
    source = probe_source_text(row["condition"], ref, read_model)
    out: Dict[str, Any] = {"system": row["system"], "sample_set": row["sample_set"], "pgpid": str(row["pgpid"]),
                           "kind": row["task"], "condition": row["condition"], "lang_class": ref.get("lang_class"),
                           "type": ref.get("type")}
    if row["condition"] == "read":
        read = pick_read(ref, read_model)
        out.update({"read_model": read.get("model") if read else None,
                    "read_cer": read.get("cer_vs_edition") if read else None,
                    "read_coverage": read.get("coverage") if read else None})
    if row["task"] == "translate":
        target = entry.get("target") or "English"
        ref_text = ref.get("translation_he") if target_lang(target) == "he" else ref.get("translation_en")
        out["target"] = target
        out.update(translation_metrics(row.get("output_text"), ref_text or "", target, source))
        return out
    pred, strict = parse_catalogue(row.get("output_text"), row.get("output_json"))
    ref_people = {"dict": dictionary_people(ref.get("description") or "", people_dict),
                  "judge": judge_people.get(str(row["pgpid"]))}
    out.update(catalogue_metrics(pred, ref, source, ref_people, scholar_names))
    out["json_strict"] = strict
    return out


#: Paired condition contrasts per sample set: (label, sample_set_a, condition_a, sample_set_b, condition_b).
CONDITION_CONTRASTS: List[Tuple[str, str, str, str, str]] = [
    ("S1 edition − S2 read (JA→EN)", "S1_ja_en", "edition", "S2_ja_en_read", "read"),
    ("S4 edition − translit (AR→EN)", "S4_ar_en", "edition", "S4_ar_en", "translit"),
    ("S4 edition − translit_plain (AR→EN)", "S4_ar_en", "edition", "S4_ar_en", "translit_plain"),
    ("S4 translit_plain − translit (AR→EN)", "S4_ar_en", "translit_plain", "S4_ar_en", "translit"),
]


def _by_doc(rows: Sequence[Dict[str, Any]], system: str, sset: str, cond: str) -> Dict[str, Dict[str, Any]]:
    """Scored translation rows of one (system, set, condition) by pgpid.

    :param rows: Scored translation rows.
    :param system: System.
    :param sset: Sample set.
    :param cond: Condition.
    :returns: ``{pgpid: row}``.
    """
    return {r["pgpid"]: r for r in rows if r["system"] == system and r["sample_set"] == sset and r["condition"] == cond}


def system_contrasts(rows: Sequence[Dict[str, Any]], base: str, n_boot: int = N_BOOT,
                     seed: int = SEED) -> List[Dict[str, Any]]:
    """Paired chrF++ difference of every system against a base system per (sample set, condition).

    :param rows: Scored translation rows.
    :param base: Base system (e.g. the stock 27B).
    :param n_boot: Bootstrap resamples.
    :param seed: RNG seed.
    :returns: One dict per (system, set, condition) with :func:`paired_bootstrap` fields (a = system, b = base).
    """
    out = []
    cells = sorted({(r["sample_set"], r["condition"]) for r in rows})
    systems = sorted({r["system"] for r in rows} - {base})
    for sset, cond in cells:
        b = _by_doc(rows, base, sset, cond)
        for system in systems:
            a = _by_doc(rows, system, sset, cond)
            docs = sorted(set(a) & set(b), key=lambda p: int(p) if p.isdigit() else p)
            if not docs:
                continue
            pb = paired_bootstrap_stats(_row_stats([a[d] for d in docs]), _row_stats([b[d] for d in docs]),
                                        n_boot=n_boot, seed=seed)
            out.append({"system": system, "base": base, "sample_set": sset, "condition": cond, **pb})
    return out


def condition_contrasts(rows: Sequence[Dict[str, Any]], n_boot: int = N_BOOT, seed: int = SEED) -> List[Dict[str, Any]]:
    """Paired chrF++ difference between two conditions of the same system on the same documents.

    :param rows: Scored translation rows.
    :param n_boot: Bootstrap resamples.
    :param seed: RNG seed.
    :returns: One dict per (system, contrast) with :func:`paired_bootstrap` fields.
    """
    out = []
    for system in sorted({r["system"] for r in rows}):
        for label, sa, ca, sb, cb in CONDITION_CONTRASTS:
            a, b = _by_doc(rows, system, sa, ca), _by_doc(rows, system, sb, cb)
            docs = sorted(set(a) & set(b), key=lambda p: int(p) if p.isdigit() else p)
            if not docs:
                continue
            pb = paired_bootstrap_stats(_row_stats([a[d] for d in docs]), _row_stats([b[d] for d in docs]),
                                        n_boot=n_boot, seed=seed)
            out.append({"system": system, "contrast": label, **pb})
    return out


def kept_rows(rows: Sequence[Dict[str, Any]], ignore: Iterable[str] = ()) -> List[Dict[str, Any]]:
    """Rows without an exclusion reason (reasons in ``ignore`` do not exclude).

    :param rows: Scored rows carrying ``exclusion_reasons``.
    :param ignore: Reasons to ignore.
    :returns: The kept rows.
    """
    ig = set(ignore)
    return [r for r in rows if not (set(r.get("exclusion_reasons") or []) - ig)]


def summarise_probe(trans: Sequence[Dict[str, Any]], cat: Sequence[Dict[str, Any]], base: str,
                    n_boot: int = N_BOOT, seed: int = SEED) -> Dict[str, Any]:
    """All aggregate tables of a set of scored probe rows.

    :param trans: Scored translation rows.
    :param cat: Scored catalogue rows.
    :param base: Base system for the paired system contrasts.
    :param n_boot: Bootstrap resamples.
    :param seed: RNG seed.
    :returns: ``{"translation", "catalogue", "system_contrasts", "condition_contrasts"}``.
    """
    tgroups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in trans:
        tgroups[(r["system"], r["sample_set"], r["condition"])].append(r)
    cgroups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in cat:
        cgroups[(r["system"], r["sample_set"], r["condition"])].append(r)
    return {
        "translation": [{"system": s, "sample_set": ss, "condition": c, **aggregate_translation_cell(rs, n_boot, seed)}
                        for (s, ss, c), rs in sorted(tgroups.items())],
        "catalogue": [{"system": s, "sample_set": ss, "condition": c, **aggregate_catalogue_cell(rs)}
                      for (s, ss, c), rs in sorted(cgroups.items())],
        "system_contrasts": system_contrasts(trans, base, n_boot, seed) if any(r["system"] == base for r in trans) else [],
        "condition_contrasts": condition_contrasts(trans, n_boot, seed),
    }


#: t27a bars of plan §5 that are offsets from the stock score: (label, sample_set, condition, offset).
T27A_OFFSET_BARS: List[Tuple[str, str, str, float]] = [
    ("S1 JA→EN ≥ stock +5", "S1_ja_en", "edition", 5.0),
    ("S3 JA→HE ≥ stock +5", "S3_ja_he", "edition", 5.0),
    ("S4 AR→EN within −1 of stock", "S4_ar_en", "edition", -1.0),
    ("S5 HE→EN within −2 of stock", "S5_he_en", "edition", -2.0),
    ("S4 transliterated ≥ stock +3", "S4_ar_en", "translit", 3.0),
    ("S2 read ≥ stock +3", "S2_ja_en_read", "read", 3.0),
]


def rebased_bars(translation: Sequence[Dict[str, Any]], base: str) -> List[Dict[str, Any]]:
    """The §5 t27a offset bars re-based on the stock system's v2 corpus chrF++ (clean subset).

    :param translation: Translation cells (``summary["translation"]``).
    :param base: Stock system label.
    :returns: One dict per bar: label, stock v1 and v2 scores, the v2 bar.
    """
    out = []
    for label, sset, cond, off in T27A_OFFSET_BARS:
        c = next((x for x in translation if x["system"] == base and x["sample_set"] == sset and x["condition"] == cond), None)
        if c is None:
            continue
        out.append({"bar": label, "sample_set": sset, "condition": cond, "offset": off,
                    "stock_v1_recomputed": c["corpus_chrfpp_v1"], "stock_v2": c["corpus_chrfpp"],
                    "bar_v2": c["corpus_chrfpp"] + off, "n": c["n"]})
    return out


def _fmt(x: Any, nd: int = 1, pct: bool = False) -> str:
    """Format a number for a markdown table.

    :param x: Value.
    :param nd: Decimals.
    :param pct: Multiply by 100 and add ``%``.
    :returns: String (``-`` for None).
    """
    if x is None:
        return "-"
    if pct:
        return f"{100 * x:.{nd}f}%"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _table(headers: List[str], body: List[List[str]]) -> str:
    """Markdown table.

    :param headers: Header cells.
    :param body: Body rows.
    :returns: Markdown text.
    """
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in body])


def write_baselines(summary: Dict[str, Any], path: Path, v1_clean: Optional[Dict[str, Any]] = None) -> None:
    """Write ``BASELINES_v2.md`` from a rescore summary.

    :param summary: Output of :func:`rescore` (``clean`` and ``full`` parts).
    :param path: Markdown path.
    :param v1_clean: The 10-18 ``summary_clean.json`` (for the on-file v1 column), or None.
    :returns: None
    """
    clean = summary["clean"]
    base = summary["base_system"]
    v1_cells = {(c["system"], c["sample_set"], c["condition"]): c["corpus_chrfpp"]
                for c in (v1_clean or {}).get("translation", [])}
    md = ["# Baselines under scorer v2 (stored 10-18 probe outputs, no model calls)", "",
          f"Scorer `{SCORER_VERSION}` (`src/finetuning/text27b/scorer.py rescore`), built "
          f"{summary['built_at']}. Outputs: the probe's `outputs/` (+ `outputs_fixpass/` for the "
          f"`translit_plain` arm); references: `{summary['pairs_file']}` (the pairs the probe prompts were "
          f"built from); clean subset = `{summary['exclusions_file']}` ({summary['exclusions']['rows']} "
          f"(set, document) rows dropped). Systems: {', '.join(summary['systems'])}; base for paired "
          f"contrasts: `{base}`. Bootstrap: {summary['n_boot']} document resamples, seed {summary['seed']}, "
          "95 % percentile CI on corpus chrF++.", "",
          "What changed from the v1 scorer: edition line markers (Gil `(n–m)`, U+2212 ranges, `(ln. 9)`, India "
          "Book `(III,5:4)`, line-initial `(N)` and ascending leading `N.` numbers) are stripped from BOTH "
          "hypothesis and reference before chrF++; failure counters gain repetition loops and copy-through; "
          "catalogue records in the training schema are mapped onto the probe schema; people grounding and "
          "summary specifics are deterministic (no judge). `v1 (on file)` = the 10-18 clean report; "
          "`v1 recomputed` = this module with the marker strip off (a port check: it should equal the "
          "on-file value).", "",
          "## Translation, clean subset (corpus chrF++)", ""]
    body = []
    for c in clean["translation"]:
        key = (c["system"], c["sample_set"], c["condition"])
        body.append([c["sample_set"], c["condition"], c["system"], str(c["n"]),
                     _fmt(v1_cells.get(key)), _fmt(c["corpus_chrfpp_v1"]), f"**{_fmt(c['corpus_chrfpp'])}**",
                     f"{_fmt(c['ci_lo'])} – {_fmt(c['ci_hi'])}",
                     _fmt(c["corpus_chrfpp"] - c["corpus_chrfpp_v1"]) if c["corpus_chrfpp_v1"] is not None else "-",
                     _fmt(c["doc_chrfpp_mean"]), _fmt(c["len_ratio_median"], 2)])
    md.append(_table(["set", "cond", "system", "n", "v1 (on file)", "v1 recomputed", "v2", "95 % CI", "v2 − v1",
                      "doc mean", "len ratio"], body))
    md += ["", "## Failure counters, clean subset", "",
           "empty / wrong script / repetition loop / copy-through (≥ 50 % of output tokens copied from the "
           "source) / length ratio > 2 / < 0.5; degenerate = empty ∪ wrong script ∪ repetition ∪ copy-through "
           "(t27a bar: ≤ 1 %).", ""]
    md.append(_table(["set", "cond", "system", "n", "empty", "wrong script", "repetition", "copy-through",
                      "len >2", "len <0.5", "degenerate"],
                     [[c["sample_set"], c["condition"], c["system"], str(c["n"]), str(c["empty"]),
                       str(c["wrong_script"]), str(c["repetition"]), str(c["copy_through"]), str(c["len_gt2"]),
                       str(c["len_lt05"]), f"{c['degenerate']} ({_fmt(c['degenerate_rate'], 0, True)})"]
                      for c in clean["translation"]]))
    md += ["", f"## Paired contrasts against `{base}` (same documents, clean subset)", "",
           "Δ = system − base corpus chrF++; CI from the paired document bootstrap; ✓ = CI excludes 0.", ""]
    md.append(_table(["set", "cond", "system", "n", "system", "base", "Δ", "95 % CI", ""],
                     [[c["sample_set"], c["condition"], c["system"], str(c["n"]), _fmt(c["a"]), _fmt(c["b"]),
                       _fmt(c["delta"]), f"{_fmt(c['lo'])} – {_fmt(c['hi'])}", "✓" if c["excludes_0"] else ""]
                      for c in clean["system_contrasts"]]))
    md += ["", "## Paired condition contrasts (same system, same documents, clean subset)", ""]
    md.append(_table(["system", "contrast", "n", "a", "b", "Δ", "95 % CI", ""],
                     [[c["system"], c["contrast"], str(c["n"]), _fmt(c["a"]), _fmt(c["b"]), _fmt(c["delta"]),
                       f"{_fmt(c['lo'])} – {_fmt(c['hi'])}", "✓" if c["excludes_0"] else ""]
                      for c in clean["condition_contrasts"]]))
    md += ["", "## t27a offset bars re-based on the v2 stock scores", "",
           "Plan §5 states the bars as offsets from the stock 27B (S1 ≥ +5 → 42.4, S3 ≥ +5 → 32.0 under v1). "
           "Under v2 the same offsets apply to the v2 stock scores below; compare fine-tunes under v2 only.", ""]
    md.append(_table(["bar", "n", "stock v1 (recomputed)", "stock v2", "bar under v2"],
                     [[b["bar"], str(b["n"]), _fmt(b["stock_v1_recomputed"]), _fmt(b["stock_v2"]), _fmt(b["bar_v2"])]
                      for b in summary.get("bars_rebased", [])]))
    md += ["", "## Catalogue, clean subset (C1 edition, C2 read)", "",
           "JSON ok = parseable (lenient); strict = the raw answer is exactly one JSON object. People recall: "
           "`dict` = PGP people.csv names found in the description; `judge` = names the 10-18 judge extracted "
           "from the description (cache, read only). Grounded = predicted people names supported by the input "
           "text (deterministic: ≥ half of the name's significant tokens found, by folded form or consonant "
           "skeleton). Specifics per summary: entity candidates of the predicted summary, in the input / "
           "only in the PGP description / in neither.", ""]
    md.append(_table(["set", "system", "n", "JSON ok", "strict", "type", "date ±2 (n ref)", "±10",
                      "dated w/o ref", "people dict", "people judge", "grounded", "places", "in-ref-only/summ",
                      "in-neither/summ"],
                     [[c["sample_set"], c["system"], str(c["n"]), _fmt(c["json_ok"], 0, True),
                       _fmt(c["json_strict"], 0, True), _fmt(c["type_acc"], 0, True),
                       f"{_fmt(c['date_acc2'], 0, True)} ({c['date_n_ref']})", _fmt(c["date_acc10"], 0, True),
                       _fmt(c["dated_without_ref"], 0, True),
                       _fmt((c.get("people_dict") or {}).get("recall"), 0, True),
                       _fmt((c.get("people_judge") or {}).get("recall"), 0, True),
                       _fmt(c["people_grounded_share"], 0, True), _fmt(c["mentioned_places_recall"], 0, True),
                       _fmt(c["in_ref_only_per_summary"], 2), _fmt(c["in_neither_per_summary"], 2)]
                      for c in clean["catalogue"]]))
    full = {(c["system"], c["sample_set"], c["condition"]): c for c in summary["full"]["translation"]}
    md += ["", "## Full sets (no exclusions), corpus chrF++ v2", ""]
    md.append(_table(["set", "cond", "system", "n", "v1 recomputed", "v2", "95 % CI"],
                     [[k[1], k[2], k[0], str(c["n"]), _fmt(c["corpus_chrfpp_v1"]), _fmt(c["corpus_chrfpp"]),
                       f"{_fmt(c['ci_lo'])} – {_fmt(c['ci_hi'])}"] for k, c in sorted(full.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0]))]))
    md += ["", "## Notes", ""] + [f"- {n}" for n in summary.get("notes", [])] + [""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(md), encoding="utf-8")


def repo_relative(path: Path, repo: Path) -> str:
    """A path relative to the repository (its file name when outside), for manifests without machine paths.

    :param path: Any path.
    :param repo: Repository root.
    :returns: The relative path string.
    """
    q = Path(path).resolve()
    try:
        return str(q.relative_to(Path(repo).resolve()))
    except ValueError:
        return q.name


def rescore(paths: Optional[_paths.Paths] = None, outputs_dir: Optional[Path] = None,
            extra_dirs: Optional[Sequence[Path]] = None, exclusions: Optional[Path] = None,
            pairs: Optional[Path] = None, samples: Optional[Path] = None, out_dir: Optional[Path] = None,
            baselines_md: Optional[Path] = None, systems: str = "all", base_system: str = "qwen3.8-27b",
            n_boot: int = N_BOOT, seed: int = SEED) -> Dict[str, Any]:
    """Re-score every stored probe output under scorer v2 and write the score files and baselines.

    :param paths: Build paths (default :func:`paths.load_paths`).
    :param outputs_dir: Probe outputs (default ``probe_dir/outputs``).
    :param extra_dirs: Extra output directories (default ``probe_dir/outputs_fixpass`` when present).
    :param exclusions: Exclusions JSON (default ``paths.exclusions_v1``).
    :param pairs: Pairs file the probe prompts were built from (default ``probe_dir/pgp_parallel_v1.jsonl``).
    :param samples: ``samples.json`` (default ``paths.samples``).
    :param out_dir: Score directory (default ``out_dir/scores_v2``).
    :param baselines_md: Markdown path (default ``out_dir/BASELINES_v2.md``).
    :param systems: ``all`` or comma-separated system labels.
    :param base_system: Base system of the paired contrasts and re-based bars.
    :param n_boot: Bootstrap resamples.
    :param seed: RNG seed.
    :returns: The summary written to ``summary_v2.json``.
    """
    p = paths or _paths.load_paths()
    outputs_dir = Path(outputs_dir or p.probe_dir / "outputs")
    if extra_dirs is None:
        extra_dirs = [d for d in (p.probe_dir / "outputs_fixpass",) if d.exists()]
    exclusions = Path(exclusions or p.exclusions_v1)
    pairs = Path(pairs or p.probe_dir / "pgp_parallel_v1.jsonl")
    samples_path = Path(samples or p.samples)
    out_dir = Path(out_dir or p.out("scores_v2"))
    baselines_md = Path(baselines_md or p.out("BASELINES_v2.md"))

    with open(samples_path, encoding="utf-8") as fh:
        entries = sample_entries(json.load(fh))
    excl = load_exclusions(exclusions)
    names = discover_systems(outputs_dir, systems)
    outputs = {s: load_outputs(outputs_dir, s, extra_dirs) for s in names}
    names = [s for s in names if outputs[s]]
    pgpids = {str(r["pgpid"]) for s in names for r in outputs[s]}
    refs = load_references(pairs, pgpids)
    people_csv = p.pgp_raw / "people.csv"
    people_dict = load_people_dictionary(people_csv) if people_csv.exists() else {}
    sources_csv = p.pgp_raw / "sources.csv"
    scholars = tn.scholar_surnames(sources_csv) if sources_csv.exists() else set()
    judge_people = load_ref_people_cache(p.probe_dir / "scores" / "judge_cache.jsonl", refs)
    LOG.info("rescore: systems %s, refs %d, people dict %d, judge ref-people %d", names, len(refs),
             len(people_dict), len(judge_people))

    scored: List[Dict[str, Any]] = []
    for s in names:
        for row in outputs[s]:
            key = (row["sample_set"], str(row["pgpid"]))
            r = score_probe_row(row, refs[str(row["pgpid"])], entries[key], people_dict, judge_people, scholars)
            r["exclusion_reasons"] = excl.get(key, [])
            scored.append(r)
    trans = [r for r in scored if r["kind"] == "translate"]
    cat = [r for r in scored if r["kind"] == "catalogue"]
    clean = summarise_probe(kept_rows(trans), kept_rows(cat), base_system, n_boot, seed)
    full = summarise_probe(trans, cat, base_system, n_boot, seed)
    v1_path = p.probe_dir / "scores" / "summary_clean.json"
    v1_clean = json.loads(v1_path.read_text(encoding="utf-8")) if v1_path.exists() else None
    rel = functools.partial(repo_relative, repo=p.repo)
    summary: Dict[str, Any] = {
        "scorer_version": SCORER_VERSION, "prompt_version": PROBE_PROMPT_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "systems": names, "base_system": base_system, "n_boot": n_boot, "seed": seed,
        "pairs_file": rel(pairs), "exclusions_file": rel(exclusions),
        "outputs": [rel(outputs_dir)] + [rel(d) for d in extra_dirs],
        "exclusions": {"rows": len(excl), "per_set": dict(Counter(k[0] for k in excl))},
        "scored_rows": len(scored), "kept_rows": len(kept_rows(trans)) + len(kept_rows(cat)),
        "clean": clean, "full": full, "bars_rebased": rebased_bars(clean["translation"], base_system),
        "notes": [
            "Opus has no S3/S4/S5 outputs on file (budget cut in the 10-18 run); its S1/S2/C1/C2 rows are scored.",
            "Copy-through and catalogue grounding re-derive the source text with the v0_2 transliteration "
            "(Arabic-Indic digits are no longer deleted); the model saw the 10-18 text. Diagnostics only.",
            "The judge is not called; `people judge` uses cached reference names only where the PGP description "
            "is unchanged (fingerprint check).",
            "`qwen3.8-27b-think` has no successful rows (thinking arm failed in the 10-18 run) and is skipped.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in names:
        rows_s = [{k: v for k, v in r.items() if not k.startswith("_")} for r in scored if r["system"] == s]
        (out_dir / f"{s}.json").write_text(json.dumps(rows_s, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "summary_v2.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    write_baselines(summary, baselines_md, v1_clean)
    for c in clean["translation"]:
        LOG.info("[v2] %-20s %-14s %-14s n=%3d chrF++ %.1f [%.1f, %.1f] (v1 %.1f) degenerate %d",
                 c["system"], c["sample_set"], c["condition"], c["n"], c["corpus_chrfpp"], c["ci_lo"], c["ci_hi"],
                 c["corpus_chrfpp_v1"], c["degenerate"])
    return summary


def score_file(rows_path: Path, hyps_path: Path, out_path: Optional[Path] = None, scholars_csv: Optional[Path] = None,
               n_boot: int = N_BOOT, seed: int = SEED) -> Dict[str, Any]:
    """Score model outputs (``{"row_id", "output_text"}`` JSONL) against v0_2 family rows.

    :param rows_path: A family JSONL (rows carry ``meta.row_id``).
    :param hyps_path: Outputs JSONL keyed by ``row_id``.
    :param out_path: Where to write ``{"scored": [...], "cells": [...]}`` (None = do not write).
    :param scholars_csv: PGP ``sources.csv`` for scholar surnames (optional).
    :param n_boot: Bootstrap resamples.
    :param seed: RNG seed.
    :returns: ``{"scored", "cells", "missing"}``.
    """
    hyps = {str(h["row_id"]): h.get("output_text") for h in _rows.read_jsonl(hyps_path)}
    scholars = tn.scholar_surnames(scholars_csv) if scholars_csv and Path(scholars_csv).exists() else set()
    scored, missing = [], 0
    for row in _rows.read_jsonl(rows_path):
        rid = str(row.get("meta", {}).get("row_id"))
        if rid not in hyps:
            missing += 1
            continue
        scored.append(score_output(row, hyps[rid], scholars))
    cells = aggregate_scored(scored, n_boot=n_boot, seed=seed)
    result = {"scorer_version": SCORER_VERSION, "scored": [{k: v for k, v in r.items() if not k.startswith("_")}
                                                           for r in scored], "cells": cells, "missing": missing}
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI: ``rescore`` (stored probe outputs) or ``score`` (outputs on v0_2 family rows).

    :param argv: Arguments (default ``sys.argv[1:]``).
    :returns: None
    """
    ap = argparse.ArgumentParser(description="text27b scorer v2")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rs = sub.add_parser("rescore", help="re-score the stored 10-18 probe outputs")
    rs.add_argument("--systems", default="all")
    rs.add_argument("--base", default="qwen3.8-27b", help="base system of the paired contrasts")
    rs.add_argument("--outputs", type=Path, default=None)
    rs.add_argument("--extra-outputs", type=Path, nargs="*", default=None)
    rs.add_argument("--exclusions", type=Path, default=None)
    rs.add_argument("--pairs", type=Path, default=None)
    rs.add_argument("--out", type=Path, default=None, help="score directory (default out_dir/scores_v2)")
    rs.add_argument("--baselines", type=Path, default=None)
    rs.add_argument("--n-boot", type=int, default=N_BOOT)
    rs.add_argument("--seed", type=int, default=SEED)
    sc = sub.add_parser("score", help="score outputs keyed by row_id against a family file")
    sc.add_argument("--rows", type=Path, required=True)
    sc.add_argument("--hyps", type=Path, required=True)
    sc.add_argument("--out", type=Path, required=True)
    sc.add_argument("--n-boot", type=int, default=N_BOOT)
    sc.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.cmd == "rescore":
        s = rescore(outputs_dir=args.outputs, extra_dirs=args.extra_outputs, exclusions=args.exclusions,
                    pairs=args.pairs, out_dir=args.out, baselines_md=args.baselines, systems=args.systems,
                    base_system=args.base, n_boot=args.n_boot, seed=args.seed)
        print(json.dumps({"systems": s["systems"], "scored_rows": s["scored_rows"], "kept_rows": s["kept_rows"],
                          "bars_rebased": s["bars_rebased"]}, ensure_ascii=False, indent=1))
    else:
        p = _paths.load_paths()
        r = score_file(args.rows, args.hyps, args.out, p.pgp_raw / "sources.csv", args.n_boot, args.seed)
        print(json.dumps({"scored": len(r["scored"]), "missing": r["missing"], "cells": len(r["cells"])}, indent=1))


if __name__ == "__main__":
    main()
