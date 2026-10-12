# File name: compare_descriptions.py
# Date: 10/8/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Compare runs of generated catalogue descriptions (``describe_from_reads`` output), document by document.

The vector-less pilot describes each document once from single-reader reads (pass A: v21b on the V2+V3 tranches,
pass B: the blend on V1) and once from the character-ROVER vote of the readers (:mod:`rover_reads`). This module
reports whether the voted reads change what the generator writes, and how often that changes what would be embedded.

Inputs are labelled groups of ``--out`` files of :mod:`src.datasets.consensus.describe_from_reads`, written
``LABEL=PATH[+PATH...]`` (``single=passA.jsonl+passB.jsonl voted=rover.jsonl``). The files of a label are read in
order; a document written twice keeps its newest row (``generated_at``; the later row on a tie), as regenerated rows
replace older ones in the index. Rows are joined on ``doc_id``. A partial last line (a file still being appended),
a line that is not JSON or a row without ``doc_id`` / ``json`` is skipped and listed, never fatal; a label with no
usable row is.

Per label, over each label's own documents (repeated over the shared documents when the labels cover different
ones): documents; embed-eligible count and share, and the embedding decision's reasons; the grounded confidence (the
value after the grounding cap, which gates embedding): median, mean and tiers (``< MIN_EMBED_CONFIDENCE``: not
embedded; up to ``CONFIDENCE_CAP``, where any grounding violation puts a reply; above the cap; ``>= 0.75``) and the
replies the cap lowered; genre distribution; identified works; grounded entity spans per kind and documents with one;
dropped spans and documents with one; summary quotes absent from the transcription; agreed-line share (median over
documents, and pooled) and lines per document; summary and description-text length in characters.

Paired, for every pair of labels in the order given (``Δ`` = later label minus earlier label), over the documents
both hold:

* documents in one label only (listed, never an error);
* documents whose embedding decision differs, each way, with both reasons;
* confidence deltas: median and mean with paired bootstrap 95 % percentile intervals (documents resampled with
  replacement, ``--resamples`` 10,000 by default, numpy ``default_rng(--seed)``: a seed always gives the same
  interval), and how many documents went up, stayed, went down;
* genre agreement rate and the genre changes;
* summary similarity, a character-level ratio: ``Levenshtein.ratio`` of the whitespace-collapsed summaries
  (``1 - indel distance / (len a + len b)``; 1 = identical; two empty summaries count as identical);
* the ``--sample`` documents with the largest divergence: lowest summary similarity first, ties broken by a genre
  change, an embedding-decision change, the larger confidence change, then ``doc_id``. A document whose summary,
  genre, embedding decision and confidence are all unchanged is never sampled, so a file compared with itself
  reports no divergent document.

Usage (repo root)::

    P=logs/next_round/consensus_pool
    .venv/bin/python -m src.datasets.consensus.compare_descriptions \\
        single=$P/generated_descriptions_pilot_passA.jsonl+$P/generated_descriptions_pilot_passB.jsonl \\
        voted=$P/generated_descriptions_pilot_rover.jsonl \\
        --out $P/pilot_description_comparison.md [--dry-run] [--seed 0] [--resamples 10000] [--sample 10]

``--dry-run`` validates the inputs (labels, files, rows, overlap) and writes nothing. Without ``--out`` the
markdown is printed; with it the file is replaced atomically.
"""
import argparse
import json
import os
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import Levenshtein
import numpy as np

from src.datasets.consensus.describe_from_reads import CONFIDENCE_CAP, ENTITY_KINDS, GENRES, MIN_EMBED_CONFIDENCE

HIGH_CONFIDENCE = 0.75             # lower edge of the top confidence tier
DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 0
DEFAULT_SAMPLE = 10
DEFAULT_MAX_LISTED = 50            # document ids printed per list before "... and N more"
CI_LEVEL = 0.95
BOOTSTRAP_CHUNK_CELLS = 2_000_000  # resampled values held in memory at once
ZERO = 1e-9                        # |Δ| below this is no change (confidences are rounded to 6 places first)
READER_PREFIX = "qwen3-vl-8b-heb-"
TIERS: Tuple[str, ...] = (
    f"below {MIN_EMBED_CONFIDENCE:g} (not embedded)",
    f"{MIN_EMBED_CONFIDENCE:g} to {CONFIDENCE_CAP:g} (grounding cap)",
    f"above {CONFIDENCE_CAP:g}, below {HIGH_CONFIDENCE:g}",
    f"{HIGH_CONFIDENCE:g} or more",
    "missing",
)
SIMILARITY_BINS: Tuple[Tuple[float, str], ...] = (
    (1.0, "identical (1.0)"), (0.9, "0.9 up to 1"), (0.7, "0.7 up to 0.9"), (0.5, "0.5 up to 0.7"),
    (0.3, "0.3 up to 0.5"), (0.0, "below 0.3"))


# ----------------------------------------------------------------------------- rows


@dataclass
class Description:
    """The fields of one generated-description row that the comparison uses.

    :param doc_id: Document.
    :param source: ``<file>:<line>`` of the row.
    :param generated_at: UTC timestamp of the row (empty when absent).
    :param genre: Genre of the grounded reply (``missing`` when absent).
    :param document_type: Catalogue type of the reply.
    :param confidence: Grounded confidence (after the cap), None when absent.
    :param confidence_capped: Whether the grounding cap lowered the model's confidence.
    :param eligible: ``embed_eligible``.
    :param reason: ``embed_reason``.
    :param identified_work: Whether the reply names a work.
    :param kept: Grounded entity spans per kind.
    :param dropped: Dropped entity spans.
    :param dropped_reasons: Dropped spans per reason (``absent``, ``too_short``).
    :param quotes_absent: Original-script summary quotes absent from the transcription.
    :param agreed_pct: ``agreed_lines_pct`` of the reads the description came from.
    :param n_lines: Transcription lines of those reads.
    :param n_agreed: Agreed lines among them.
    :param summary: The summary, whitespace collapsed.
    :param text_chars: Characters of ``description_text`` (the text that would be embedded).
    :param model: Generator model.
    :param prompt_version: Prompt version.
    :param reasoning: Reasoning mode.
    :param readers: VLM readers behind the description, short names joined by ``+``.
    """

    doc_id: str
    source: str
    generated_at: str
    genre: str
    document_type: str
    confidence: Optional[float]
    confidence_capped: bool
    eligible: bool
    reason: str
    identified_work: bool
    kept: Dict[str, int]
    dropped: int
    dropped_reasons: Dict[str, int]
    quotes_absent: int
    agreed_pct: Optional[float]
    n_lines: Optional[int]
    n_agreed: Optional[int]
    summary: str
    text_chars: int
    model: str
    prompt_version: str
    reasoning: str
    readers: str

    @property
    def n_kept(self) -> int:
        """Grounded entity spans of all kinds.

        :return: Count.
        :rtype: int
        """
        return sum(self.kept.values())


def _number(value: Any) -> Optional[float]:
    """A JSON number as float (booleans and anything else give None).

    :param value: Value.
    :return: Float or None.
    :rtype: Optional[float]
    """
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _count(value: Any) -> Optional[int]:
    """A JSON integer (booleans and anything else give None).

    :param value: Value.
    :return: Integer or None.
    :rtype: Optional[int]
    """
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def short_reader(name: str) -> str:
    """Reader name without the shared checkpoint prefix (``qwen3-vl-8b-heb-v21b-step1200`` -> ``v21b-step1200``).

    :param name: Reader (VLM checkpoint or ``rover<N>``).
    :return: Short name.
    :rtype: str
    """
    return name[len(READER_PREFIX):] if name.startswith(READER_PREFIX) else name


def row_problem(row: Any) -> Optional[str]:
    """Why a parsed line cannot be compared, or None when it can.

    :param row: Parsed JSON line.
    :return: Problem or None.
    :rtype: Optional[str]
    """
    if not isinstance(row, dict):
        return "not a JSON object"
    if not isinstance(row.get("doc_id"), str) or not row["doc_id"]:
        return "no doc_id"
    if not isinstance(row.get("json"), dict):
        return "no json reply (a failure row?)"
    return None


def parse_row(line: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parse one complete JSONL line into a row, or say why not.

    :param line: Line text.
    :return: ``(row, None)`` or ``(None, problem)``.
    :rtype: Tuple[Optional[Dict[str, Any]], Optional[str]]
    """
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        return None, f"not JSON ({exc.msg})"
    problem = row_problem(row)
    return (None, problem) if problem else (row, None)


def description_from_row(row: Mapping[str, Any], source: str) -> Description:
    """The compared fields of a ``describe_from_reads`` row (one that passed :func:`row_problem`).

    Entity counts come from the grounding report (``grounding.kept``), else from the grounded reply's entities; the
    confidence from the grounded reply, else from the report.

    :param row: Row.
    :param source: ``<file>:<line>``.
    :return: Description.
    :rtype: Description
    """
    reply = row["json"]
    grounding = row.get("grounding") if isinstance(row.get("grounding"), Mapping) else {}
    kept = grounding.get("kept") if isinstance(grounding.get("kept"), Mapping) else (reply.get("entities") or {})
    dropped = [d for d in grounding.get("dropped") or [] if isinstance(d, Mapping)]
    work = reply.get("identified_work")
    confidence = _number(reply.get("confidence"))
    if confidence is None:
        confidence = _number(grounding.get("confidence"))
    readers = row.get("readers") if isinstance(row.get("readers"), Mapping) else {}
    eligible = bool(row.get("embed_eligible"))
    return Description(
        doc_id=row["doc_id"], source=source, generated_at=str(row.get("generated_at") or ""),
        genre=str(reply.get("genre") or "missing"),
        document_type=" ".join(str(reply.get("document_type") or "").split()),
        confidence=None if confidence is None else round(confidence, 6),
        confidence_capped=bool(grounding.get("confidence_capped")), eligible=eligible,
        reason=str(row.get("embed_reason") or ("eligible" if eligible else "unknown")),
        identified_work=isinstance(work, Mapping) and bool(work.get("title")),
        kept={kind: len(kept.get(kind) or []) for kind in ENTITY_KINDS}, dropped=len(dropped),
        dropped_reasons=dict(Counter(str(d.get("reason") or "?") for d in dropped)),
        quotes_absent=len(grounding.get("summary_spans_absent") or []),
        agreed_pct=_number(row.get("agreed_lines_pct")), n_lines=_count(row.get("n_lines")),
        n_agreed=_count(row.get("n_agreed")), summary=" ".join(str(reply.get("summary") or "").split()),
        text_chars=len(str(row.get("description_text") or "")), model=str(row.get("model") or "?"),
        prompt_version=str(row.get("prompt_version") or "?"), reasoning=str(row.get("reasoning") or "?"),
        readers=" + ".join(sorted(short_reader(str(r)) for r in readers.get("vlm") or [])) or "?")


# ----------------------------------------------------------------------------- inputs


@dataclass
class LabelRun:
    """One labelled group of description files, loaded.

    :param label: Label.
    :param paths: Files in the order given.
    :param docs: Description per document (newest row per document).
    :param rows: Usable rows read (duplicates included).
    :param duplicates: Rows of a document already seen (the newer row is kept).
    :param problems: Skipped lines as ``<file>:<line>: <why>``.
    """

    label: str
    paths: List[Path]
    docs: Dict[str, Description] = field(default_factory=dict)
    rows: int = 0
    duplicates: int = 0
    problems: List[str] = field(default_factory=list)


def split_input_spec(raw: str) -> Tuple[str, List[str]]:
    """Split ``LABEL=PATH[+PATH...]``.

    :param raw: Argument.
    :return: ``(label, paths)``; an empty label or list when the argument does not have that form.
    :rtype: Tuple[str, List[str]]
    """
    label, sep, rest = raw.partition("=")
    if not sep:
        return "", []
    return label.strip(), [p for p in rest.split("+") if p.strip()]


def input_errors(specs: Sequence[str]) -> List[str]:
    """Problems with the input arguments: form, repeated labels, missing files, fewer than two labels.

    :param specs: ``LABEL=PATH[+PATH...]`` arguments.
    :return: Error messages (empty when the inputs can be loaded).
    :rtype: List[str]
    """
    errors: List[str] = []
    labels: List[str] = []
    for raw in specs:
        label, paths = split_input_spec(raw)
        if not label or not paths:
            errors.append(f"{raw!r} is not LABEL=PATH[+PATH...]")
            continue
        if label in labels:
            errors.append(f"label {label!r} given twice")
        labels.append(label)
        errors.extend(f"{label}: not found: {p}" for p in paths if not Path(p).is_file())
    if len(set(labels)) < 2:
        errors.append("give at least two labels")
    return errors


def load_label(label: str, paths: Sequence[Any]) -> LabelRun:
    """Read a label's files; the newest row of each document wins.

    :param label: Label.
    :param paths: Description JSONL files, in order.
    :return: The loaded label.
    :rtype: LabelRun
    """
    run = LabelRun(label=label, paths=[Path(p) for p in paths])
    for path in run.paths:
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                where = f"{path.name}:{n}"
                if not line.endswith("\n"):
                    run.problems.append(f"{where}: partial last line skipped (file still being written?)")
                    continue
                row, problem = parse_row(line)
                if row is None:
                    run.problems.append(f"{where}: {problem}")
                    continue
                run.rows += 1
                desc = description_from_row(row, where)
                old = run.docs.get(desc.doc_id)
                if old is not None:
                    run.duplicates += 1
                    if desc.generated_at < old.generated_at:
                        continue
                run.docs[desc.doc_id] = desc
    return run


# ----------------------------------------------------------------------------- metrics


def _median(values: Sequence[float]) -> Optional[float]:
    """Median, or None for no values.

    :param values: Numbers.
    :return: Median or None.
    :rtype: Optional[float]
    """
    return float(statistics.median(values)) if values else None


def _mean(values: Sequence[float]) -> Optional[float]:
    """Mean, or None for no values.

    :param values: Numbers.
    :return: Mean or None.
    :rtype: Optional[float]
    """
    return float(statistics.fmean(values)) if values else None


def confidence_tier(confidence: Optional[float]) -> str:
    """The :data:`TIERS` entry a confidence falls in.

    :param confidence: Grounded confidence, or None.
    :return: Tier name.
    :rtype: str
    """
    if confidence is None:
        return TIERS[4]
    if confidence < MIN_EMBED_CONFIDENCE:
        return TIERS[0]
    if confidence <= CONFIDENCE_CAP:
        return TIERS[1]
    if confidence < HIGH_CONFIDENCE:
        return TIERS[2]
    return TIERS[3]


def label_metrics(docs: Sequence[Description]) -> Dict[str, Any]:
    """Per-label metrics over a set of descriptions.

    :param docs: Descriptions (one per document).
    :return: ``documents``, ``eligible``, ``reasons``, ``confidence_median`` / ``_mean``, ``tiers``, ``capped``,
        ``genres``, ``identified_work``, ``kept`` (per kind), ``docs_with_kept``, ``dropped``, ``docs_with_dropped``,
        ``dropped_reasons``, ``quotes_absent``, ``agreed_pct_median``, ``agreed_pct_pooled``, ``lines_median``,
        ``agreed_lines_median``, ``summary_chars_median``, ``text_chars_median`` and ``provenance`` counters.
    :rtype: Dict[str, Any]
    """
    confidences = [d.confidence for d in docs if d.confidence is not None]
    counted = [d for d in docs if d.n_lines is not None and d.n_agreed is not None]
    pooled_lines = sum(d.n_lines for d in counted)
    dropped_reasons: Counter = Counter()
    for d in docs:
        dropped_reasons.update(d.dropped_reasons)
    return {
        "documents": len(docs),
        "eligible": sum(d.eligible for d in docs),
        "reasons": Counter(d.reason for d in docs),
        "confidence_median": _median(confidences),
        "confidence_mean": _mean(confidences),
        "tiers": Counter(confidence_tier(d.confidence) for d in docs),
        "capped": sum(d.confidence_capped for d in docs),
        "genres": Counter(d.genre for d in docs),
        "identified_work": sum(d.identified_work for d in docs),
        "kept": {kind: sum(d.kept.get(kind, 0) for d in docs) for kind in ENTITY_KINDS},
        "docs_with_kept": sum(d.n_kept > 0 for d in docs),
        "dropped": sum(d.dropped for d in docs),
        "docs_with_dropped": sum(d.dropped > 0 for d in docs),
        "dropped_reasons": dropped_reasons,
        "quotes_absent": sum(d.quotes_absent for d in docs),
        "agreed_pct_median": _median([d.agreed_pct for d in docs if d.agreed_pct is not None]),
        "agreed_pct_pooled": 100.0 * sum(d.n_agreed for d in counted) / pooled_lines if pooled_lines else None,
        "lines_median": _median([d.n_lines for d in docs if d.n_lines is not None]),
        "agreed_lines_median": _median([d.n_agreed for d in docs if d.n_agreed is not None]),
        "summary_chars_median": _median([len(d.summary) for d in docs]),
        "text_chars_median": _median([d.text_chars for d in docs]),
        "provenance": {"generator": Counter(d.model for d in docs),
                       "prompt version": Counter(d.prompt_version for d in docs),
                       "reasoning": Counter(d.reasoning for d in docs),
                       "readers": Counter(d.readers for d in docs)},
    }


def summary_similarity(a: str, b: str) -> float:
    """Character-level similarity of two summaries: ``Levenshtein.ratio`` after collapsing whitespace.

    :param a: First summary.
    :param b: Second summary.
    :return: 0 to 1 (1 = identical; two empty summaries are identical).
    :rtype: float
    """
    a, b = " ".join(a.split()), " ".join(b.split())
    return 1.0 if a == b else float(Levenshtein.ratio(a, b))


def bootstrap_ci(values: Sequence[float], resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED,
                 level: float = CI_LEVEL) -> Dict[str, Optional[Tuple[float, float]]]:
    """Paired bootstrap percentile intervals of the median and the mean of per-document deltas.

    The documents (their deltas) are resampled with replacement ``resamples`` times with numpy's
    ``default_rng(seed)``; both statistics use the same resamples. The same values, resample count and seed always
    give the same intervals.

    :param values: Per-document deltas.
    :param resamples: Bootstrap resamples.
    :param seed: Random seed.
    :param level: Interval coverage (0.95 = 2.5th to 97.5th percentile).
    :return: ``{"median": (lo, hi), "mean": (lo, hi)}``; None values without data.
    :rtype: Dict[str, Optional[Tuple[float, float]]]
    """
    data = np.asarray(values, dtype=float)
    if data.size == 0 or resamples < 1:
        return {"median": None, "mean": None}
    rng = np.random.default_rng(seed)
    chunk = max(1, BOOTSTRAP_CHUNK_CELLS // data.size)
    medians, means = [], []
    for start in range(0, resamples, chunk):
        draws = data[rng.integers(0, data.size, size=(min(chunk, resamples - start), data.size))]
        medians.append(np.median(draws, axis=1))
        means.append(draws.mean(axis=1))
    tail = 100.0 * (1.0 - level) / 2.0
    out: Dict[str, Optional[Tuple[float, float]]] = {}
    for name, stats in (("median", medians), ("mean", means)):
        lo, hi = np.percentile(np.concatenate(stats), [tail, 100.0 - tail])
        out[name] = (float(lo), float(hi))
    return out


def confidence_delta(a: Description, b: Description) -> Optional[float]:
    """``b.confidence - a.confidence`` rounded to 6 places, or None when either is missing.

    :param a: Earlier label's description.
    :param b: Later label's description.
    :return: Delta or None.
    :rtype: Optional[float]
    """
    if a.confidence is None or b.confidence is None:
        return None
    return round(b.confidence - a.confidence, 6)


def divergence_key(a: Description, b: Description, similarity: float) -> Tuple[float, int, int, float]:
    """How far two descriptions of a document diverge; larger tuples diverge more.

    :param a: Earlier label's description.
    :param b: Later label's description.
    :param similarity: Their :func:`summary_similarity`.
    :return: ``(1 - similarity, genre changed, embedding decision changed, |confidence delta|)``; all zero when
        nothing compared changed.
    :rtype: Tuple[float, int, int, float]
    """
    delta = confidence_delta(a, b)
    return (round(1.0 - similarity, 6), int(a.genre != b.genre), int(a.eligible != b.eligible),
            abs(delta) if delta is not None else 0.0)


def divergent_documents(a: Mapping[str, Description], b: Mapping[str, Description],
                        similarity: Mapping[str, float]) -> List[str]:
    """Shared documents with any divergence, most divergent first (ties: ``doc_id``).

    :param a: Earlier label's descriptions by document.
    :param b: Later label's descriptions by document.
    :param similarity: Summary similarity per shared document.
    :return: Document ids.
    :rtype: List[str]
    """
    keys = {d: divergence_key(a[d], b[d], s) for d, s in similarity.items()}
    changed = [d for d, key in keys.items() if any(v > ZERO for v in key)]
    return sorted(changed, key=lambda d: (tuple(-v for v in keys[d]), d))


def pair_metrics(a: Mapping[str, Description], b: Mapping[str, Description], resamples: int = DEFAULT_RESAMPLES,
                 seed: int = DEFAULT_SEED, sample: int = DEFAULT_SAMPLE) -> Dict[str, Any]:
    """Paired metrics of two labels over the documents both hold (``Δ`` = ``b`` minus ``a``).

    :param a: Earlier label's descriptions by document.
    :param b: Later label's descriptions by document.
    :param resamples: Bootstrap resamples.
    :param seed: Bootstrap seed.
    :param sample: Divergent documents to keep for display.
    :return: ``shared``, ``only_a``, ``only_b``, ``eligible_a_only``, ``eligible_b_only``, ``same_decision``,
        ``reason_changes``, ``deltas``, ``delta_median``, ``delta_mean``, ``delta_ci``, ``higher`` / ``equal`` /
        ``lower``, ``genre_agree``, ``genre_changes``, ``similarity``, ``n_divergent`` and ``sample``.
    :rtype: Dict[str, Any]
    """
    shared = sorted(set(a) & set(b))
    all_deltas = {d: confidence_delta(a[d], b[d]) for d in shared}
    deltas = {d: x for d, x in all_deltas.items() if x is not None}
    values = list(deltas.values())
    similarity = {d: summary_similarity(a[d].summary, b[d].summary) for d in shared}
    genre_changes = Counter((a[d].genre, b[d].genre) for d in shared if a[d].genre != b[d].genre)
    divergent = divergent_documents(a, b, similarity)
    return {
        "shared": shared,
        "only_a": sorted(set(a) - set(b)),
        "only_b": sorted(set(b) - set(a)),
        "eligible_a_only": [d for d in shared if a[d].eligible and not b[d].eligible],
        "eligible_b_only": [d for d in shared if b[d].eligible and not a[d].eligible],
        "same_decision": sum(a[d].eligible == b[d].eligible for d in shared),
        "reason_changes": Counter((a[d].reason, b[d].reason) for d in shared if a[d].reason != b[d].reason),
        "deltas": deltas,
        "delta_median": _median(values),
        "delta_mean": _mean(values),
        "delta_ci": bootstrap_ci(values, resamples, seed),
        "higher": sum(x > ZERO for x in values),
        "equal": sum(abs(x) <= ZERO for x in values),
        "lower": sum(x < -ZERO for x in values),
        "genre_agree": len(shared) - sum(genre_changes.values()),
        "genre_changes": genre_changes,
        "similarity": similarity,
        "n_divergent": len(divergent),
        "sample": divergent[:max(sample, 0)],
    }


def compare_runs(runs: Sequence[LabelRun], resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED,
                 sample: int = DEFAULT_SAMPLE) -> Dict[str, Any]:
    """All metrics: per label (own and, when the document sets differ, shared documents) and per pair of labels.

    :param runs: Loaded labels, in the order given.
    :param resamples: Bootstrap resamples.
    :param seed: Bootstrap seed.
    :param sample: Divergent documents shown per pair.
    :return: ``labels``, ``per_label``, ``shared_docs``, ``per_label_shared`` (None when every label holds the same
        documents) and ``pairs`` (each with ``a`` / ``b`` labels plus :func:`pair_metrics`).
    :rtype: Dict[str, Any]
    """
    shared = sorted(set.intersection(*(set(r.docs) for r in runs))) if runs else []
    same = all(len(r.docs) == len(shared) for r in runs)
    return {
        "labels": [r.label for r in runs],
        "per_label": {r.label: label_metrics(list(r.docs.values())) for r in runs},
        "shared_docs": len(shared),
        "per_label_shared": None if same else {r.label: label_metrics([r.docs[d] for d in shared]) for r in runs},
        "pairs": [{"a": x.label, "b": y.label, **pair_metrics(x.docs, y.docs, resamples, seed, sample)}
                  for x, y in combinations(runs, 2)],
    }


# ----------------------------------------------------------------------------- markdown


def _cell(text: Any) -> str:
    """Markdown table cell: whitespace collapsed, pipes and angle brackets escaped.

    :param text: Value.
    :return: Cell text.
    :rtype: str
    """
    return " ".join(str(text).split()).replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;")


def md_table(header: Sequence[str], rows: Sequence[Sequence[Any]], align: str) -> List[str]:
    """Markdown table lines.

    :param header: Column titles.
    :param rows: Rows (one value per column).
    :param align: One ``l`` / ``r`` per column.
    :return: Lines.
    :rtype: List[str]
    """
    out = ["| " + " | ".join(_cell(h) for h in header) + " |",
           "|" + "|".join("---:" if a == "r" else "---" for a in align) + "|"]
    out.extend("| " + " | ".join(_cell(c) for c in row) + " |" for row in rows)
    return out


def _share(k: int, n: int) -> str:
    """``k (x.x %)``.

    :param k: Count.
    :param n: Total.
    :return: Text.
    :rtype: str
    """
    return f"{k} ({100.0 * k / n:.1f} %)" if n else str(k)


def _num(x: Optional[float], digits: int = 3) -> str:
    """A number to ``digits`` places, ``-`` for None.

    :param x: Number.
    :param digits: Decimal places.
    :return: Text.
    :rtype: str
    """
    return "-" if x is None else f"{x:.{digits}f}"


def _signed(x: Optional[float], digits: int = 3) -> str:
    """A signed number to ``digits`` places (rounding noise shown as +0), ``-`` for None.

    :param x: Number.
    :param digits: Decimal places.
    :return: Text.
    :rtype: str
    """
    if x is None:
        return "-"
    return f"{0.0 if abs(x) < 0.5 * 10 ** -digits else x:+.{digits}f}"


def _ci(interval: Optional[Tuple[float, float]]) -> str:
    """A confidence interval as ``lo to hi``.

    :param interval: ``(lo, hi)`` or None.
    :return: Text.
    :rtype: str
    """
    return "-" if interval is None else f"{_signed(interval[0])} to {_signed(interval[1])}"


def _pct(x: Optional[float]) -> str:
    """A percentage to one place, ``-`` for None.

    :param x: Percentage.
    :return: Text.
    :rtype: str
    """
    return "-" if x is None else f"{x:.1f} %"


def _counts(counter: Mapping[str, int]) -> str:
    """``value n; value n`` from the most frequent.

    :param counter: Counts.
    :return: Text.
    :rtype: str
    """
    return "; ".join(f"{k} {v}" for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))) or "-"


def _ids(ids: Sequence[str], max_listed: int) -> str:
    """Document ids, comma separated, cut at ``max_listed``.

    :param ids: Ids.
    :param max_listed: Ids printed at most.
    :return: Text.
    :rtype: str
    """
    shown = ", ".join(f"`{d}`" for d in ids[:max_listed])
    return shown + (f", ... and {len(ids) - max_listed} more" if len(ids) > max_listed else "")


def per_label_rows(metrics: Mapping[str, Mapping[str, Any]], labels: Sequence[str]) -> List[List[str]]:
    """Rows of the per-label table (one column per label).

    :param metrics: :func:`label_metrics` per label.
    :param labels: Labels in column order.
    :return: Rows.
    :rtype: List[List[str]]
    """
    specs: List[Tuple[str, Any]] = [
        ("documents", lambda m: str(m["documents"])),
        ("embed-eligible", lambda m: _share(m["eligible"], m["documents"])),
        ("confidence: median / mean", lambda m: f"{_num(m['confidence_median'])} / {_num(m['confidence_mean'])}"),
    ]
    for tier in TIERS:
        if tier != TIERS[4] or any(metrics[label]["tiers"].get(tier) for label in labels):
            specs.append((f"confidence {tier}", lambda m, t=tier: _share(m["tiers"].get(t, 0), m["documents"])))
    specs += [
        ("confidence lowered by the grounding cap", lambda m: _share(m["capped"], m["documents"])),
        ("identified work", lambda m: _share(m["identified_work"], m["documents"])),
        ("grounded entity spans", lambda m: str(sum(m["kept"].values()))),
        (" / ".join(ENTITY_KINDS), lambda m: " / ".join(str(m["kept"][k]) for k in ENTITY_KINDS)),
        ("documents with a grounded span", lambda m: _share(m["docs_with_kept"], m["documents"])),
        ("dropped spans (documents with one)", lambda m: f"{m['dropped']} ({m['docs_with_dropped']})"),
        ("summary quotes absent from the transcription", lambda m: str(m["quotes_absent"])),
        ("agreed-line share: median / pooled", lambda m: f"{_pct(m['agreed_pct_median'])} / "
                                                         f"{_pct(m['agreed_pct_pooled'])}"),
        ("lines per document: median (agreed: median)", lambda m: f"{_num(m['lines_median'], 1)} "
                                                                  f"({_num(m['agreed_lines_median'], 1)})"),
        ("summary characters: median", lambda m: _num(m["summary_chars_median"], 0)),
        ("description text characters: median", lambda m: _num(m["text_chars_median"], 0)),
    ]
    return [[name] + [fn(metrics[label]) for label in labels] for name, fn in specs]


def per_label_section(metrics: Mapping[str, Mapping[str, Any]], labels: Sequence[str], title: str) -> List[str]:
    """Per-label table plus the genre and embedding-decision tables.

    :param metrics: :func:`label_metrics` per label.
    :param labels: Labels in column order.
    :param title: Section title.
    :return: Lines.
    :rtype: List[str]
    """
    out = [f"## {title}", ""]
    out += md_table(["metric", *labels], per_label_rows(metrics, labels), "l" + "r" * len(labels)) + [""]
    seen = {g for label in labels for g in metrics[label]["genres"]}
    genres = [g for g in GENRES if g in seen] + sorted(seen - set(GENRES))
    out += ["Genre:", ""]
    out += md_table(["genre", *labels], [[g] + [_share(metrics[label]["genres"].get(g, 0), metrics[label]["documents"])
                                                for label in labels] for g in genres], "l" + "r" * len(labels))
    totals: Counter = Counter()
    for label in labels:
        totals.update(metrics[label]["reasons"])
    out += ["", "Embedding decision (`embed_reason`):", ""]
    out += md_table(["reason", *labels], [[r] + [_share(metrics[label]["reasons"].get(r, 0),
                                                        metrics[label]["documents"]) for label in labels]
                                          for r, _ in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))],
                    "l" + "r" * len(labels))
    return out + [""]


def inputs_section(runs: Sequence[LabelRun], metrics: Mapping[str, Mapping[str, Any]], max_listed: int) -> List[str]:
    """Inputs, provenance per label, and the skipped lines.

    :param runs: Loaded labels.
    :param metrics: :func:`label_metrics` per label (own documents).
    :param max_listed: Problems listed at most.
    :return: Lines.
    :rtype: List[str]
    """
    out = ["## Inputs", ""]
    out += md_table(["label", "files", "rows", "documents", "duplicates replaced", "lines skipped"],
                    [[r.label, ", ".join(p.name for p in r.paths), r.rows, len(r.docs), r.duplicates, len(r.problems)]
                     for r in runs], "llrrrr") + [""]
    kinds = list(metrics[runs[0].label]["provenance"]) if runs else []
    out += md_table(["label", *kinds], [[r.label] + [_counts(metrics[r.label]["provenance"][k]) for k in kinds]
                                        for r in runs], "l" * (len(kinds) + 1)) + [""]
    setups = {(tuple(sorted(metrics[r.label]["provenance"]["generator"])),
               tuple(sorted(metrics[r.label]["provenance"]["prompt version"])),
               tuple(sorted(metrics[r.label]["provenance"]["reasoning"]))) for r in runs}
    if len(setups) > 1 or any(len(part) > 1 for setup in setups for part in setup):
        out += ["**Warning:** generator, prompt version or reasoning mode differ between or within labels, so the "
                "differences below are not due to the reads alone.", ""]
    problems = [p for r in runs for p in (f"{r.label}: {x}" for x in r.problems)]
    if problems:
        out += [f"Skipped lines ({len(problems)}):", ""]
        out += [f"- {_cell(p)}" for p in problems[:max_listed]]
        out += ([f"- ... and {len(problems) - max_listed} more"] if len(problems) > max_listed else []) + [""]
    return out


def _agreed(d: Description) -> str:
    """Agreed lines of a description's reads as ``k of n (x %)``.

    :param d: Description.
    :return: Text.
    :rtype: str
    """
    if d.n_lines is None or d.n_agreed is None:
        return _pct(d.agreed_pct)
    return f"{d.n_agreed} of {d.n_lines} ({_pct(d.agreed_pct)})"


def divergence_entry(rank: int, doc_id: str, a_label: str, b_label: str, a: Description, b: Description,
                     similarity: float) -> List[str]:
    """One sampled document with both descriptions side by side.

    :param rank: Position in the sample (1-based).
    :param doc_id: Document.
    :param a_label: Earlier label.
    :param b_label: Later label.
    :param a: Earlier label's description.
    :param b: Later label's description.
    :param similarity: Summary similarity.
    :return: Lines.
    :rtype: List[str]
    """
    rows = [["genre / type", f"{a.genre} / {a.document_type or '-'}", f"{b.genre} / {b.document_type or '-'}"],
            ["confidence", _num(a.confidence, 2), _num(b.confidence, 2)],
            ["embedding", a.reason, b.reason],
            ["agreed lines", _agreed(a), _agreed(b)],
            ["readers", a.readers, b.readers],
            ["summary", a.summary or "-", b.summary or "-"]]
    return [f"#### {rank}. `{doc_id}` (summary similarity {similarity:.2f})", "",
            *md_table(["", a_label, b_label], rows, "lll"), ""]


def pair_section(pair: Mapping[str, Any], docs_a: Mapping[str, Description], docs_b: Mapping[str, Description],
                 resamples: int, seed: int, max_listed: int) -> List[str]:
    """The paired section of one pair of labels.

    :param pair: :func:`pair_metrics` output plus ``a`` / ``b`` labels.
    :param docs_a: Earlier label's descriptions.
    :param docs_b: Later label's descriptions.
    :param resamples: Bootstrap resamples (for the text).
    :param seed: Bootstrap seed (for the text).
    :param max_listed: Ids listed at most.
    :return: Lines.
    :rtype: List[str]
    """
    a, b = pair["a"], pair["b"]
    n = len(pair["shared"])
    out = [f"## {a} → {b} (paired)", "",
           f"Documents in both: {n}. Only in {a}: {len(pair['only_a'])}. Only in {b}: {len(pair['only_b'])}.", ""]
    for name, ids in ((a, pair["only_a"]), (b, pair["only_b"])):
        if ids:
            out += [f"- only in {name}: {_ids(ids, max_listed)}"]
    if pair["only_a"] or pair["only_b"]:
        out += [""]

    flips = pair["eligible_a_only"] + pair["eligible_b_only"]
    out += ["### Embedding decision", "",
            f"Same decision for {_share(pair['same_decision'], n)} documents. Eligible in {a} only: "
            f"{len(pair['eligible_a_only'])}; eligible in {b} only: {len(pair['eligible_b_only'])}.", ""]
    if flips:
        rows = [[f"`{d}`", docs_a[d].reason, docs_b[d].reason, _num(docs_a[d].confidence, 2),
                 _num(docs_b[d].confidence, 2)] for d in flips[:max_listed]]
        out += md_table(["document", f"{a}", f"{b}", f"confidence {a}", f"confidence {b}"], rows, "lllrr")
        out += ([f"... and {len(flips) - max_listed} more"] if len(flips) > max_listed else []) + [""]

    ci = pair["delta_ci"]
    out += [f"### Confidence ({b} − {a})", "",
            *md_table(["statistic", "Δ", f"95 % CI (paired bootstrap, {resamples:,} resamples, seed {seed})"],
                      [["median", _signed(pair["delta_median"]), _ci(ci["median"])],
                       ["mean", _signed(pair["delta_mean"]), _ci(ci["mean"])]], "lrl"), "",
            f"Higher in {b}: {pair['higher']}; equal: {pair['equal']}; lower: {pair['lower']} "
            f"(of {len(pair['deltas'])} documents with both confidences).", ""]

    out += ["### Genre", "", f"Same genre: {_share(pair['genre_agree'], n)}.", ""]
    if pair["genre_changes"]:
        out += md_table([a, b, "documents"], [[ga, gb, k] for (ga, gb), k in sorted(
            pair["genre_changes"].items(), key=lambda kv: (-kv[1], kv[0]))], "llr") + [""]

    sims = sorted(pair["similarity"].values())
    out += ["### Summary similarity", ""]
    if sims:
        q1, q2, q3 = (float(np.percentile(sims, p)) for p in (25, 50, 75))
        out += [f"Character-level ratio of the summaries (`Levenshtein.ratio`, 1 = identical): median {q2:.2f}, "
                f"quartiles {q1:.2f} / {q3:.2f}, min {sims[0]:.2f}, mean {statistics.fmean(sims):.2f}.", ""]
        bins: Counter = Counter()
        for s in sims:
            bins[next(name for edge, name in SIMILARITY_BINS if s >= edge - ZERO)] += 1
        out += md_table(["similarity", "documents"], [[name, _share(bins[name], n)] for _, name in SIMILARITY_BINS],
                        "lr") + [""]
    else:
        out += ["No shared document.", ""]

    out += ["### Largest divergence", ""]
    if not pair["sample"]:
        out += ["No document differs in summary, genre, embedding decision or confidence.", ""]
    else:
        out += [f"{pair['n_divergent']} documents differ in summary, genre, embedding decision or confidence; the "
                f"{len(pair['sample'])} most divergent (lowest summary similarity; ties: genre change, "
                f"embedding-decision change, larger confidence change):", ""]
        for rank, d in enumerate(pair["sample"], 1):
            out += divergence_entry(rank, d, a, b, docs_a[d], docs_b[d], pair["similarity"][d])
    return out


def render_markdown(runs: Sequence[LabelRun], result: Mapping[str, Any], resamples: int, seed: int,
                    max_listed: int = DEFAULT_MAX_LISTED, built_at: Optional[str] = None) -> str:
    """The comparison report.

    :param runs: Loaded labels.
    :param result: :func:`compare_runs` output.
    :param resamples: Bootstrap resamples used.
    :param seed: Bootstrap seed used.
    :param max_listed: Ids listed at most per list.
    :param built_at: Timestamp for the header (default now, UTC).
    :return: Markdown text.
    :rtype: str
    """
    labels = result["labels"]
    by_label = {r.label: r for r in runs}
    out = [f"# Generated descriptions compared: {' vs '.join(labels)}", "",
           f"Built {built_at or utc_now()} by `compare_descriptions`. Rows are machine-generated descriptions from "
           f"automatic transcriptions; nothing here was written to an index. Per-label figures cover each label's own "
           f"documents; paired figures cover the documents both labels hold, with Δ = later label minus earlier "
           f"label.", ""]
    out += inputs_section(runs, result["per_label"], max_listed)
    out += per_label_section(result["per_label"], labels, "Per label")
    if result["per_label_shared"] is not None:
        out += per_label_section(result["per_label_shared"], labels,
                                 f"Per label, shared documents only ({result['shared_docs']})")
    for pair in result["pairs"]:
        out += pair_section(pair, by_label[pair["a"]].docs, by_label[pair["b"]].docs, resamples, seed, max_listed)
    return "\n".join(out).rstrip() + "\n"


# ----------------------------------------------------------------------------- CLI


def utc_now() -> str:
    """Current UTC time, ISO 8601 to the second.

    :return: Timestamp.
    :rtype: str
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_text_atomic(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` via a temp file and a rename.

    :param path: Destination.
    :param text: Content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def validation_lines(runs: Sequence[LabelRun], max_listed: int = DEFAULT_MAX_LISTED) -> List[str]:
    """Printable input check: rows, documents, duplicates and skipped lines per label; overlap per pair.

    :param runs: Loaded labels.
    :param max_listed: Skipped lines printed at most per label.
    :return: Lines.
    :rtype: List[str]
    """
    out = [f"compare_descriptions: {len(runs)} labels"]
    for r in runs:
        out.append(f"  {r.label}: {len(r.paths)} file(s) ({', '.join(p.name for p in r.paths)}); rows {r.rows}; "
                   f"documents {len(r.docs)}; duplicates replaced {r.duplicates}; lines skipped {len(r.problems)}")
        out.extend(f"    {p}" for p in r.problems[:max_listed])
    for x, y in combinations(runs, 2):
        both = set(x.docs) & set(y.docs)
        out.append(f"  {x.label} vs {y.label}: shared {len(both)}; only in {x.label} {len(set(x.docs) - both)}; "
                   f"only in {y.label} {len(set(y.docs) - both)}")
    return out


def summary_lines(result: Mapping[str, Any]) -> List[str]:
    """One printable line per pair (for logs).

    :param result: :func:`compare_runs` output.
    :return: Lines.
    :rtype: List[str]
    """
    out = []
    for p in result["pairs"]:
        a, b, n = p["a"], p["b"], len(p["shared"])
        ci = p["delta_ci"]
        sims = list(p["similarity"].values())
        out.append(f"{a} -> {b}: shared {n} (only {a} {len(p['only_a'])}, only {b} {len(p['only_b'])}); "
                   f"eligible in {a} only {len(p['eligible_a_only'])}, in {b} only {len(p['eligible_b_only'])}; "
                   f"confidence Δ median {_signed(p['delta_median'])} [{_ci(ci['median'])}], mean "
                   f"{_signed(p['delta_mean'])} [{_ci(ci['mean'])}]; same genre {_share(p['genre_agree'], n)}; "
                   f"summary similarity median {_num(_median(sims), 2)}; divergent documents {p['n_divergent']}")
    return out


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser.

    :return: Parser.
    :rtype: argparse.ArgumentParser
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", metavar="LABEL=PATH[+PATH...]",
                    help="describe_from_reads output files under a label; two or more labels")
    ap.add_argument("--out", default=None, help="markdown report, replaced atomically (default: print it)")
    ap.add_argument("--dry-run", action="store_true", help="validate the inputs only; write nothing")
    ap.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES, help="bootstrap resamples (%(default)s)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="bootstrap seed (%(default)s)")
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                    help="most divergent documents shown per pair (%(default)s)")
    ap.add_argument("--max-listed", type=int, default=DEFAULT_MAX_LISTED,
                    help="document ids / skipped lines listed at most per list (%(default)s)")
    return ap


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Load and check the inputs, then (unless ``--dry-run``) compare and write or print the report.

    :param args: Parsed CLI arguments (inputs already checked by :func:`input_errors`).
    :return: ``{"dry_run": True, "documents": {...}}`` for a dry run, else the :func:`compare_runs` output.
    :rtype: Dict[str, Any]
    :raises SystemExit: When a label has no usable row.
    """
    runs = [load_label(*split_input_spec(raw)) for raw in args.inputs]
    print("\n".join(validation_lines(runs, args.max_listed)), flush=True)
    empty = [r.label for r in runs if not r.docs]
    if empty:
        raise SystemExit(f"no usable description row for label(s): {', '.join(empty)}")
    if args.dry_run:
        print("dry run: inputs valid; nothing written", flush=True)
        return {"dry_run": True, "documents": {r.label: len(r.docs) for r in runs}}
    result = compare_runs(runs, args.resamples, args.seed, args.sample)
    print("\n".join(summary_lines(result)), flush=True)
    text = render_markdown(runs, result, args.resamples, args.seed, args.max_listed)
    if args.out:
        write_text_atomic(Path(args.out), text)
        print(f"wrote {args.out}", flush=True)
    else:
        print(text, flush=True)
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point.

    :param argv: Arguments (None = ``sys.argv``).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    errors = input_errors(args.inputs)
    if args.resamples < 1:
        errors.append("--resamples must be at least 1")
    if args.sample < 0 or args.max_listed < 0:
        errors.append("--sample and --max-listed must not be negative")
    if errors:
        parser.error("; ".join(errors))
    run(args)


if __name__ == "__main__":
    main()
