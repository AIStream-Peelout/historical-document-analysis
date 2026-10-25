# File name: rover_consensus.py
# Date: 10/7/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""ROVER-style output-space consensus over saved transcriptions (offline).

Classic ROVER (Fiscus 1997, "A post-processing system to yield reduced word
error rates: Recognizer Output Voting Error Reduction (ROVER)"), adapted to
page-level Genizah transcriptions that already sit on disk in
``transcription_raw_outputs/<doc_id>/<model>.txt``:

1. Every hypothesis is normalised with ``normalize_ink_hypothesis`` *line by
   line* (the function collapses newlines, so applying it per line is the only
   way to keep line structure) and tokenised into words, with an explicit
   newline token between lines.  ``--unit char`` uses characters instead.
2. A backbone hypothesis seeds a word transition network (WTN): a list of
   slots, each holding one entry (a token or NULL) per merged hypothesis.
   ``auto`` picks the hypothesis with the highest mean similarity to the
   others.
3. Every other hypothesis (most similar to the backbone first) is aligned to
   the current WTN by dynamic programming.  The cost of putting a token in a
   slot is the sum over the slot's entries of the normalised character edit
   distance to each entry (a NULL entry costs one gap), so similar words share
   a slot while dissimilar ones (mean distance above ``2 * gap``) become
   insert/delete pairs with NULL entries.  Costs are integers, so ties are
   exact and the alignment is deterministic.
4. Each slot is voted with per-model weights.  NULL is a candidate; ties go to
   the backbone's entry, then to the candidate backed by the higher-weight
   model, then to the earlier model on the command line.

Alignment (``align_to_network`` / ``merge_into_network``) and voting
(``vote_network``) are separate pure functions; ``rover`` composes them.

Scoring mirrors ``score_genizah_offline.py`` exactly (same benchmark loader,
same >= 50-letter filter, same normalisation, ``cer_pair(...)[0]`` strict CER,
``aligned_prf`` F1 and ``classify`` failure modes), so single-model numbers
reproduce the scorer's ``genizah_offline_long.csv``.

``--filter-failures`` drops a hypothesis before voting when ``classify``
labels it abstained / loop_collapse / hallucinated.  With the default
``--filter-source gt`` the hallucination test is the scorer's own, i.e. it
uses the ground truth's n-gram precision, so that variant is NOT deployable;
``--filter-source peer`` replaces the GT with the best n-gram precision
against the other hypotheses (GT-free).  The abstained and loop tests never
use the GT.

``--benchmark religious`` is the adapter for the religious-140 benchmark
(``genizah_religious_v1``): pages and GT come from its spec as stored (no
duplication repair, unlike the PGP loader), outputs from its
``raw_outputs/<doc_id>/<model>.txt``, and scoring is the same function calls
``run_religious_benchmark.score`` makes, so single-model numbers reproduce
``religious_scores_long.csv``.  Each page has one output per model, so the vote
is per page, as on PGP.  Pages are bucketed by column class (``single-col`` /
``multi-col``, the runner's slices) instead of the PGP script tags.

Usage (from the repo root):
    .venv/bin/python src/datasets/evaluations/helper_eval_scripts/rover_consensus.py \\
        --models qwen3_vl_8b_heb_v21b_step1200 qwen3_vl_8b_heb_v22a_step1300 \\
                 qwen3_vl_8b_heb_v22b_step1200 qwen3_vl_8b_heb_v22i_a50 \\
                 qwen3_vl_8b_heb_v23a_step1200 \\
        --filter-failures --tag word_filtered
    .venv/bin/python src/datasets/evaluations/helper_eval_scripts/rover_consensus.py \\
        --benchmark religious --unit char \\
        --models qwen3_vl_8b_heb_v21b_step1200 qwen3_vl_8b_heb_v22a_step1300 \\
                 qwen3_vl_8b_heb_v22b_step1200 qwen3_vl_8b_heb_v22i_a50 \\
        --out-dir logs/next_round/consensus/religious140 --tag char_unweighted_4vlm
"""

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.datasets.evaluations.metrics import (  # noqa: E402
    cer_pair,
    genizah_visible_ink_gt,
    normalize_ink_hypothesis,
)
from src.datasets.evaluations.helper_eval_scripts.score_genizah_offline import (  # noqa: E402
    _BENCH_DIR,
    _OUTPUTS,
    aligned_prf,
    asserted_letters,
    classify,
    letters_only,
    load_benchmark,
    ngram_precision,
)

NEWLINE = "\n"
UNITS = ("word", "char")
# Gap (insert/delete) cost per entry, in normalised-edit-distance units.  A
# token joins a slot only when its mean distance to the slot's tokens is below
# 2 * gap: 0.7 for words ("more than 70 % of the characters differ" means a
# different word); 1.0 for characters is plain unit-cost Levenshtein.
DEFAULT_GAP = {"word": 0.35, "char": 1.0}
COST_SCALE = 1000                     # integer cost units per edit-distance unit
FAILURE_MODES = ("abstained", "loop_collapse", "hallucinated")
MODES = FAILURE_MODES + ("substantive",)
BUCKETS = ("judaeo_arabic", "hebrew", "aramaic", "untagged", "arabic_script")
MIN_GT_LETTERS = 50                   # the scorer's document filter
MIN_BUCKET_DOCS = 5                   # fewer other docs in a bucket -> use all docs
MIN_CER = 1e-3                        # floor for inverse-CER weights
_DEFAULT_OUT = _REPO / "logs/next_round/consensus"
_SCRIPT_TAGS = _BENCH_DIR / "genizah_test_v1_script_tags.json"
RELIGIOUS = "religious"
RELIGIOUS_BUCKETS = ("single-col", "multi-col")   # run_religious_benchmark's column slices
# Same folder as run_religious_benchmark._BENCH.  That module is not imported here because it
# pulls in the LM Studio and Kraken clients; a test pins the two paths together.
_RELIGIOUS_DIR = _REPO / "src/datasets/raw_data/cairo_genizah/evaluations/genizah_religious_v1"

_DIAG, _UP, _LEFT = 0, 1, 2           # DP moves: token->slot, slot->NULL, new slot
_INF = np.int64(2) ** 60


# ── Hypothesis preparation ────────────────────────────────────────────────────

def normalize_lines(raw: str) -> List[str]:
    """Normalise a raw transcription line by line, keeping line structure.

    ``normalize_ink_hypothesis`` collapses every whitespace run (newlines
    included), so it is applied to each line separately.  Lines that
    normalise to nothing are dropped.

    :param raw: Raw model output as saved.
    :type raw: str
    :return: Normalised, non-empty lines in reading order.
    :rtype: List[str]
    """
    lines = (normalize_ink_hypothesis(line) for line in raw.splitlines())
    return [line for line in lines if line]


def tokenize(lines: Sequence[str], unit: str) -> List[str]:
    """Split normalised lines into voting tokens.

    :param lines: Normalised lines (see :func:`normalize_lines`).
    :type lines: Sequence[str]
    :param unit: ``word`` (words plus a newline token between lines) or
        ``char`` (every character, spaces and newlines included).
    :type unit: str
    :return: Token sequence.
    :rtype: List[str]
    :raises ValueError: For an unknown unit.
    """
    if unit == "word":
        tokens: List[str] = []
        for i, line in enumerate(lines):
            if i:
                tokens.append(NEWLINE)
            tokens.extend(line.split())
        return tokens
    if unit == "char":
        return list(NEWLINE.join(lines))
    raise ValueError(f"unknown unit {unit!r}; expected one of {UNITS}")


def detokenize(tokens: Sequence[str], unit: str) -> str:
    """Join voted tokens back into text, one transcription line per row.

    Consecutive or leading/trailing newline tokens collapse, so the result has
    no empty lines.

    :param tokens: Voted tokens in network order.
    :type tokens: Sequence[str]
    :param unit: ``word`` or ``char``.
    :type unit: str
    :return: Consensus text.
    :rtype: str
    """
    if unit == "word":
        lines: List[str] = []
        current: List[str] = []
        for token in tokens:
            if token == NEWLINE:
                if current:
                    lines.append(" ".join(current))
                current = []
            else:
                current.append(token)
        if current:
            lines.append(" ".join(current))
        return NEWLINE.join(lines)
    rows = (" ".join(row.split()) for row in "".join(tokens).split(NEWLINE))
    return NEWLINE.join(row for row in rows if row)


# ── Word transition network and alignment ─────────────────────────────────────

@dataclass(frozen=True)
class Network:
    """Word transition network built by progressive alignment.

    :param slots: One tuple per slot with one entry per merged hypothesis, in
        merge order; ``None`` marks a NULL (that hypothesis has nothing here).
    :param members: Hypothesis index (position in the caller's list) of each
        entry column, in merge order.
    """

    slots: Tuple[Tuple[Optional[str], ...], ...]
    members: Tuple[int, ...]


def seed_network(tokens: Sequence[str], member: int) -> Network:
    """Start a network from the backbone hypothesis.

    :param tokens: Backbone tokens.
    :type tokens: Sequence[str]
    :param member: Hypothesis index of the backbone.
    :type member: int
    :return: Network with one slot per backbone token.
    :rtype: Network
    """
    return Network(tuple((t,) for t in tokens), (member,))


def token_costs(left: Sequence[str], right: Sequence[str]) -> np.ndarray:
    """Pairwise normalised character edit distance in integer cost units.

    :param left: Row tokens.
    :type left: Sequence[str]
    :param right: Column tokens.
    :type right: Sequence[str]
    :return: ``len(left) x len(right)`` int32 matrix, 0 (identical) to
        ``COST_SCALE`` (nothing in common).
    :rtype: np.ndarray
    """
    if not left or not right:
        return np.zeros((len(left), len(right)), dtype=np.int32)
    dist = process.cdist(list(left), list(right),
                         scorer=Levenshtein.normalized_distance, dtype=np.float64)
    return np.rint(dist * COST_SCALE).astype(np.int32)


def align_to_network(network: Network, tokens: Sequence[str],
                     gap: int) -> List[Tuple[Optional[int], Optional[int]]]:
    """Globally align a token sequence to a network (sum-of-pairs costs).

    Moves and their integer costs (``k`` = merged hypotheses):

    * token ``j`` into slot ``i``: sum over the slot's entries of the token
      edit-distance cost (a NULL entry costs ``gap``);
    * slot ``i`` gets NULL from this hypothesis: ``gap`` per non-NULL entry;
    * token ``j`` opens a new slot (all existing entries NULL): ``gap * k``.

    Ties prefer token->slot, then slot->NULL, then a new slot.  Each DP row
    is vectorised; the in-row insertion recurrence is solved exactly with a
    cumulative minimum.

    :param network: Current network.
    :type network: Network
    :param tokens: Tokens of the hypothesis being added.
    :type tokens: Sequence[str]
    :param gap: Gap cost per entry, in integer cost units.
    :type gap: int
    :return: Alignment path of ``(slot index or None, token index or None)``
        pairs in order; every slot and every token appears exactly once.
    :rtype: List[Tuple[Optional[int], Optional[int]]]
    """
    n_slots, n_tok = len(network.slots), len(tokens)
    if n_slots == 0 or n_tok == 0:
        return [(i, None) for i in range(n_slots)] + [(None, j) for j in range(n_tok)]
    k = len(network.members)
    vocab = sorted({e for slot in network.slots for e in slot if e is not None})
    position = {tok: v for v, tok in enumerate(vocab)}
    hyp_vocab = sorted(set(tokens))
    hyp_position = {tok: v for v, tok in enumerate(hyp_vocab)}
    columns = np.fromiter((hyp_position[t] for t in tokens), dtype=np.int64, count=n_tok)
    cost_rows = token_costs(vocab, hyp_vocab)[:, columns]          # |vocab| x n_tok

    offsets = np.arange(n_tok + 1, dtype=np.int64) * (gap * k)    # cumulative insert cost
    prev = offsets.copy()                                          # DP row 0
    back = np.empty((n_slots + 1, n_tok + 1), dtype=np.int8)
    back[0, :] = _LEFT
    diag = np.empty(n_tok + 1, dtype=np.int64)
    for i, slot in enumerate(network.slots, start=1):
        ids = [position[e] for e in slot if e is not None]
        sub = cost_rows[ids].sum(axis=0, dtype=np.int64) + gap * (k - len(ids))
        up = prev + gap * len(ids)
        diag[0] = _INF
        np.add(prev[:-1], sub, out=diag[1:])
        best = np.minimum(diag, up)
        move = np.where(diag <= up, _DIAG, _UP).astype(np.int8)
        row = np.minimum.accumulate(best - offsets) + offsets
        move[row < best] = _LEFT
        back[i] = move
        prev = row

    path: List[Tuple[Optional[int], Optional[int]]] = []
    i, j = n_slots, n_tok
    while i > 0 or j > 0:
        move = back[i, j] if i > 0 else _LEFT
        if move == _DIAG:
            i, j = i - 1, j - 1
            path.append((i, j))
        elif move == _UP:
            i -= 1
            path.append((i, None))
        else:
            j -= 1
            path.append((None, j))
    path.reverse()
    return path


def merge_into_network(network: Network, tokens: Sequence[str],
                       path: Sequence[Tuple[Optional[int], Optional[int]]],
                       member: int) -> Network:
    """Add an aligned hypothesis to the network as a new entry column.

    :param network: Current network.
    :type network: Network
    :param tokens: Tokens of the hypothesis being added.
    :type tokens: Sequence[str]
    :param path: Alignment from :func:`align_to_network`.
    :type path: Sequence[Tuple[Optional[int], Optional[int]]]
    :param member: Hypothesis index of the added hypothesis.
    :type member: int
    :return: New network (the input is not modified).
    :rtype: Network
    """
    empty = (None,) * len(network.members)
    slots = []
    for slot_i, tok_j in path:
        entry = tokens[tok_j] if tok_j is not None else None
        slots.append((network.slots[slot_i] if slot_i is not None else empty) + (entry,))
    return Network(tuple(slots), network.members + (member,))


def build_network(hypotheses: Sequence[Sequence[str]], order: Sequence[int],
                  gap: int) -> Network:
    """Progressively align hypotheses into one network.

    :param hypotheses: Token sequences, indexed by hypothesis.
    :type hypotheses: Sequence[Sequence[str]]
    :param order: Merge order; ``order[0]`` is the backbone.
    :type order: Sequence[int]
    :param gap: Gap cost per entry, in integer cost units.
    :type gap: int
    :return: Network containing every hypothesis in ``order``.
    :rtype: Network
    """
    network = seed_network(hypotheses[order[0]], order[0])
    for member in order[1:]:
        path = align_to_network(network, hypotheses[member], gap)
        network = merge_into_network(network, hypotheses[member], path, member)
    return network


# ── Voting ────────────────────────────────────────────────────────────────────

def vote_slot(slot: Sequence[Optional[str]], members: Sequence[int],
              weights: Sequence[float], backbone: int) -> Optional[str]:
    """Weighted vote over one slot (NULL is a candidate).

    :param slot: Entries of the slot, aligned with ``members``.
    :type slot: Sequence[Optional[str]]
    :param members: Hypothesis index of each entry.
    :type members: Sequence[int]
    :param weights: Weight of each hypothesis, indexed by hypothesis index.
    :type weights: Sequence[float]
    :param backbone: Hypothesis index of the backbone.
    :type backbone: int
    :return: Winning token, or ``None`` when NULL wins.
    :rtype: Optional[str]
    """
    tally: Dict[Optional[str], float] = {}
    strongest: Dict[Optional[str], Tuple[float, int]] = {}
    for entry, member in zip(slot, members):
        tally[entry] = tally.get(entry, 0.0) + weights[member]
        support = (weights[member], -member)
        if entry not in strongest or support > strongest[entry]:
            strongest[entry] = support
    top = max(tally.values())
    tied = [c for c, w in tally.items() if math.isclose(w, top, rel_tol=1e-9, abs_tol=1e-12)]
    if len(tied) == 1:
        return tied[0]
    if backbone in members:
        backbone_entry = slot[list(members).index(backbone)]
        if backbone_entry in tied:
            return backbone_entry
    return max(tied, key=lambda c: strongest[c])


def vote_network(network: Network, weights: Sequence[float], backbone: int) -> List[str]:
    """Vote every slot and keep the winning tokens (NULL winners vanish).

    :param network: Network from :func:`build_network`.
    :type network: Network
    :param weights: Weight of each hypothesis, indexed by hypothesis index.
    :type weights: Sequence[float]
    :param backbone: Hypothesis index of the backbone (first tie-break).
    :type backbone: int
    :return: Voted tokens in network order.
    :rtype: List[str]
    """
    voted = []
    for slot in network.slots:
        winner = vote_slot(slot, network.members, weights, backbone)
        if winner is not None:
            voted.append(winner)
    return voted


# ── Backbone and merge order ──────────────────────────────────────────────────

def similarity_matrix(texts: Sequence[str]) -> np.ndarray:
    """Pairwise normalised Levenshtein similarity of whole texts.

    :param texts: Hypotheses as flat strings.
    :type texts: Sequence[str]
    :return: Symmetric matrix with ones on the diagonal.
    :rtype: np.ndarray
    """
    n = len(texts)
    sim = np.eye(n)
    for a in range(n):
        for b in range(a + 1, n):
            sim[a, b] = sim[b, a] = Levenshtein.normalized_similarity(texts[a], texts[b])
    return sim


def choose_backbone(sim: np.ndarray, weights: Sequence[float]) -> int:
    """Hypothesis with the highest mean similarity to the others.

    Ties go to the higher weight, then to the lower index (deterministic).

    :param sim: Matrix from :func:`similarity_matrix`.
    :type sim: np.ndarray
    :param weights: Weight of each hypothesis.
    :type weights: Sequence[float]
    :return: Backbone index.
    :rtype: int
    """
    n = sim.shape[0]
    if n == 1:
        return 0
    mean = (sim.sum(axis=1) - 1.0) / (n - 1)
    return min(range(n), key=lambda i: (-round(float(mean[i]), 12), -weights[i], i))


def merge_order(sim: np.ndarray, backbone: int) -> List[int]:
    """Backbone first, then the others by similarity to it (ties: lower index).

    :param sim: Matrix from :func:`similarity_matrix`.
    :type sim: np.ndarray
    :param backbone: Backbone index.
    :type backbone: int
    :return: Merge order.
    :rtype: List[int]
    """
    others = [i for i in range(sim.shape[0]) if i != backbone]
    others.sort(key=lambda i: (-round(float(sim[backbone, i]), 12), i))
    return [backbone] + others


@dataclass(frozen=True)
class RoverResult:
    """Outcome of one ROVER combination.

    :param text: Consensus text (one transcription line per row).
    :param backbone: Index of the backbone hypothesis.
    :param order: Merge order.
    :param n_slots: Number of network slots.
    """

    text: str
    backbone: int
    order: Tuple[int, ...]
    n_slots: int


def rover(hypotheses: Sequence[str], weights: Optional[Sequence[float]] = None,
          unit: str = "word", backbone: Optional[int] = None,
          gap: Optional[float] = None) -> RoverResult:
    """Combine raw transcriptions of one page by ROVER voting.

    :param hypotheses: Raw model outputs (normalised here, line by line).
    :type hypotheses: Sequence[str]
    :param weights: One weight per hypothesis (default all 1.0).
    :type weights: Optional[Sequence[float]]
    :param unit: ``word`` or ``char``.
    :type unit: str
    :param backbone: Fixed backbone index, or ``None`` for automatic choice.
    :type backbone: Optional[int]
    :param gap: Gap cost in edit-distance units (default per unit, see
        ``DEFAULT_GAP``).
    :type gap: Optional[float]
    :return: Consensus text plus backbone, merge order and network size.
    :rtype: RoverResult
    :raises ValueError: When no hypothesis is given or lengths disagree.
    """
    if not hypotheses:
        raise ValueError("rover needs at least one hypothesis")
    weights = [1.0] * len(hypotheses) if weights is None else list(weights)
    if len(weights) != len(hypotheses):
        raise ValueError("one weight per hypothesis is required")
    lines = [normalize_lines(h) for h in hypotheses]
    tokens = [tokenize(ls, unit) for ls in lines]
    sim = similarity_matrix([" ".join(ls) for ls in lines])
    chosen = choose_backbone(sim, weights) if backbone is None else backbone
    order = merge_order(sim, chosen)
    gap_units = int(round((DEFAULT_GAP[unit] if gap is None else gap) * COST_SCALE))
    network = build_network(tokens, order, gap_units)
    voted = vote_network(network, weights, chosen)
    return RoverResult(detokenize(voted, unit), chosen, tuple(order), len(network.slots))


# ── Failure filtering ─────────────────────────────────────────────────────────

def failure_modes(raws: Sequence[str], gt_letters: Optional[str], source: str) -> List[str]:
    """Classify every hypothesis with the scorer's ``classify``.

    :param raws: Raw model outputs.
    :type raws: Sequence[str]
    :param gt_letters: Ground truth reduced to letters (needed for ``gt``).
    :type gt_letters: Optional[str]
    :param source: ``gt`` — hallucination = n-gram precision against the GT
        below the scorer's threshold (uses the answer key, not deployable);
        ``peer`` — the same test against the best-matching other hypothesis
        (GT-free).  Abstention and loop tests never use the GT.
    :type source: str
    :return: One failure mode per hypothesis.
    :rtype: List[str]
    :raises ValueError: For an unknown source or ``gt`` without GT letters.
    """
    hyps = [normalize_ink_hypothesis(r) for r in raws]
    letters = [letters_only(h) for h in hyps]
    if source == "gt":
        if gt_letters is None:
            raise ValueError("source='gt' needs gt_letters")
        precisions = [ngram_precision(lt, gt_letters) for lt in letters]
    elif source == "peer":
        precisions = [max((ngram_precision(lt, other) for j, other in enumerate(letters) if j != i),
                          default=1.0)
                      for i, lt in enumerate(letters)]
    else:
        raise ValueError(f"unknown filter source {source!r}")
    return [classify(raw, asserted_letters(h), gt_letters or "", p)
            for raw, h, p in zip(raws, hyps, precisions)]


def select_voters(modes: Sequence[str], weights: Sequence[float]) -> List[int]:
    """Indices of hypotheses that survive failure filtering.

    :param modes: Failure mode per hypothesis.
    :type modes: Sequence[str]
    :param weights: Weight per hypothesis (fallback choice).
    :type weights: Sequence[float]
    :return: Indices whose mode is ``substantive``; when none is, the single
        highest-weight hypothesis (ties: lowest index).
    :rtype: List[int]
    """
    keep = [i for i, mode in enumerate(modes) if mode not in FAILURE_MODES]
    if keep:
        return keep
    return [min(range(len(modes)), key=lambda i: (-weights[i], i))]


# ── Weights ───────────────────────────────────────────────────────────────────

def inverse_median_weights(cers: Dict[str, Dict[str, float]], models: Sequence[str],
                           pool: Sequence[str]) -> Dict[str, float]:
    """Weight each model by 1 / its median CER over a pool of documents.

    :param cers: ``{doc_id: {model: strict CER}}``.
    :type cers: Dict[str, Dict[str, float]]
    :param models: Models to weight.
    :type models: Sequence[str]
    :param pool: Documents the medians are taken over.
    :type pool: Sequence[str]
    :return: ``{model: weight}``.
    :rtype: Dict[str, float]
    """
    return {m: 1.0 / max(statistics.median(cers[d][m] for d in pool), MIN_CER) for m in models}


def leave_one_out_weights(cers: Dict[str, Dict[str, float]], models: Sequence[str],
                          buckets: Dict[str, str], per_bucket: bool) -> Dict[str, Dict[str, float]]:
    """Inverse-median-CER weights for each document, excluding that document.

    :param cers: ``{doc_id: {model: strict CER}}``.
    :type cers: Dict[str, Dict[str, float]]
    :param models: Models to weight.
    :type models: Sequence[str]
    :param buckets: ``{doc_id: script bucket}``.
    :type buckets: Dict[str, str]
    :param per_bucket: Take medians within the document's bucket (falls back
        to all other documents when fewer than ``MIN_BUCKET_DOCS`` remain).
    :type per_bucket: bool
    :return: ``{doc_id: {model: weight}}``.
    :rtype: Dict[str, Dict[str, float]]
    """
    out = {}
    for doc_id in cers:
        pool = [d for d in cers if d != doc_id]
        if per_bucket:
            same = [d for d in pool if buckets[d] == buckets[doc_id]]
            pool = same if len(same) >= MIN_BUCKET_DOCS else pool
        out[doc_id] = inverse_median_weights(cers, models, pool)
    return out


# ── Scoring (mirrors score_genizah_offline.py) ────────────────────────────────

@dataclass(frozen=True)
class Score:
    """Scorer-equivalent numbers for one output on one document.

    :param cer: Strict CER (``cer_pair(...)[0]``), unrounded.
    :param mode: Failure mode from ``classify``.
    :param ngram_p: Order-independent n-gram precision against the GT.
    :param f1: Aligned F1 (``None`` when not computed).
    """

    cer: float
    mode: str
    ngram_p: float
    f1: Optional[float] = None


def score_output(raw: str, gt_ink: str, gt_letters: str, with_f1: bool = False) -> Score:
    """Score one raw output exactly as ``score_genizah_offline.main`` does.

    ``run_religious_benchmark.score`` makes the same calls for its ``cer``,
    ``failure_mode``, ``ngram_precision`` and ``aligned_f1`` columns.

    :param raw: Raw output text (model file, or consensus text).
    :type raw: str
    :param gt_ink: Visible-ink ground truth.
    :type gt_ink: str
    :param gt_letters: Ground truth reduced to letters.
    :type gt_letters: str
    :param with_f1: Also compute aligned F1 (difflib; slow on long texts).
    :type with_f1: bool
    :return: CER, failure mode, n-gram precision and optional F1.
    :rtype: Score
    """
    hyp = normalize_ink_hypothesis(raw)
    ngram_p = ngram_precision(letters_only(hyp), gt_letters)
    mode = classify(raw, asserted_letters(hyp), gt_letters, ngram_p)
    f1 = aligned_prf(hyp, gt_ink)[2] if with_f1 else None
    return Score(cer=cer_pair(hyp, gt_ink)[0], mode=mode, ngram_p=ngram_p, f1=f1)


@dataclass(frozen=True)
class BenchDoc:
    """One scored benchmark document and its saved outputs.

    :param doc_id: Benchmark document id.
    :param bucket: Script/language bucket (eval-only sidecar).
    :param gt_ink: Visible-ink ground truth.
    :param gt_letters: Ground truth reduced to letters.
    :param raws: ``{model: raw output}`` for the requested models.
    """

    doc_id: str
    bucket: str
    gt_ink: str
    gt_letters: str
    raws: Dict[str, str]


def load_script_buckets(path: Path) -> Dict[str, str]:
    """Read the eval-only script/language sidecar.

    :param path: ``genizah_test_v1_script_tags.json``.
    :type path: Path
    :return: ``{doc_id: bucket}``; empty when the file is missing.
    :rtype: Dict[str, str]
    """
    if not path.exists():
        return {}
    return {k: v.get("bucket", "untagged") for k, v in json.loads(path.read_text()).items()}


def load_documents(benchmark: str, models: Sequence[str], outputs_dir: Path,
                   buckets: Dict[str, str]) -> Tuple[List[BenchDoc], List[str]]:
    """Load the scored documents that have an output from every model.

    Mirrors the scorer: documents without an output folder or with fewer than
    ``MIN_GT_LETTERS`` GT letters are not scored at all.

    :param benchmark: ``verified`` or ``frozen``.
    :type benchmark: str
    :param models: Model file stems.
    :type models: Sequence[str]
    :param outputs_dir: ``transcription_raw_outputs`` folder (read only).
    :type outputs_dir: Path
    :param buckets: ``{doc_id: bucket}``.
    :type buckets: Dict[str, str]
    :return: Documents with every output, and ids of scored documents skipped
        because an output is missing.
    :rtype: Tuple[List[BenchDoc], List[str]]
    """
    docs, skipped = [], []
    for spec in load_benchmark(benchmark):
        folder = outputs_dir / spec["doc_id"]
        if not folder.is_dir():
            continue
        gt_ink = genizah_visible_ink_gt(spec["gt"])
        gt_letters = letters_only(gt_ink)
        if len(gt_letters) < MIN_GT_LETTERS:
            continue
        files = {m: folder / f"{m}.txt" for m in models}
        if not all(f.exists() for f in files.values()):
            skipped.append(spec["doc_id"])
            continue
        raws = {m: f.read_text(errors="replace") for m, f in files.items()}
        docs.append(BenchDoc(spec["doc_id"], buckets.get(spec["doc_id"], "untagged"),
                             gt_ink, gt_letters, raws))
    return docs, skipped


def religious_bucket(spec_doc: dict) -> str:
    """Column-class bucket of a religious-140 page, as the runner slices it.

    :param spec_doc: One entry of the spec's ``docs`` list.
    :type spec_doc: dict
    :return: ``multi-col`` for two or more columns, otherwise ``single-col``.
    :rtype: str
    """
    return "multi-col" if spec_doc["n_columns"] >= 2 else "single-col"


def load_religious_documents(models: Sequence[str], spec_path: Path,
                             outputs_dir: Path) -> Tuple[List[BenchDoc], List[str]]:
    """Load the religious-140 pages that have an output from every model.

    Mirrors ``run_religious_benchmark.score``: the GT is used as stored (no
    duplication repair, unlike the PGP loader), pages with fewer than
    ``MIN_GT_LETTERS`` GT letters are not scored, and each page has one output
    per model in ``<outputs_dir>/<doc_id>/<model>.txt``.

    :param models: Model file stems.
    :type models: Sequence[str]
    :param spec_path: ``genizah_religious_v1.json``.
    :type spec_path: Path
    :param outputs_dir: The benchmark's ``raw_outputs`` folder (read only).
    :type outputs_dir: Path
    :return: Pages with every output, and ids of scored pages skipped because
        an output is missing.
    :rtype: Tuple[List[BenchDoc], List[str]]
    """
    docs, skipped = [], []
    for spec in json.loads(spec_path.read_text())["docs"]:
        gt_ink = genizah_visible_ink_gt(spec["gt"])
        gt_letters = letters_only(gt_ink)
        if len(gt_letters) < MIN_GT_LETTERS:
            continue
        files = {m: outputs_dir / spec["doc_id"] / f"{m}.txt" for m in models}
        if not all(f.exists() for f in files.values()):
            skipped.append(spec["doc_id"])
            continue
        raws = {m: f.read_text(errors="replace") for m, f in files.items()}
        docs.append(BenchDoc(spec["doc_id"], religious_bucket(spec), gt_ink, gt_letters, raws))
    return docs, skipped


@dataclass(frozen=True)
class BenchmarkSource:
    """Where one ``--benchmark`` choice keeps its outputs, and how it buckets pages.

    :param outputs_dir: Folder holding ``<doc_id>/<model>.txt`` (read only).
    :param buckets: Bucket labels in report order.
    :param spec: Spec JSON for the religious loader; ``None`` for PGP, whose
        spec the scorer's ``load_benchmark`` picks.
    """

    outputs_dir: Path
    buckets: Tuple[str, ...]
    spec: Optional[Path] = None


SOURCES = {
    "verified": BenchmarkSource(_OUTPUTS, BUCKETS),
    "frozen": BenchmarkSource(_OUTPUTS, BUCKETS),
    RELIGIOUS: BenchmarkSource(_RELIGIOUS_DIR / "raw_outputs", RELIGIOUS_BUCKETS,
                               _RELIGIOUS_DIR / "genizah_religious_v1.json"),
}


def load_benchmark_documents(benchmark: str,
                             models: Sequence[str]) -> Tuple[List[BenchDoc], List[str]]:
    """Benchmark adapter: load the documents of any ``--benchmark`` choice.

    :param benchmark: A key of ``SOURCES`` (``verified``, ``frozen`` or
        ``religious``).
    :type benchmark: str
    :param models: Model file stems.
    :type models: Sequence[str]
    :return: Documents with every output, and ids of scored documents skipped
        because an output is missing.
    :rtype: Tuple[List[BenchDoc], List[str]]
    """
    source = SOURCES[benchmark]
    if benchmark == RELIGIOUS:
        return load_religious_documents(models, source.spec, source.outputs_dir)
    return load_documents(benchmark, models, source.outputs_dir, load_script_buckets(_SCRIPT_TAGS))


def oracle(cers: Dict[str, float], models: Sequence[str]) -> Tuple[float, str]:
    """Best single model on one document.

    :param cers: ``{model: CER}`` for the document.
    :type cers: Dict[str, float]
    :param models: Candidate models (ties: the earlier one).
    :type models: Sequence[str]
    :return: (minimum CER, winning model).
    :rtype: Tuple[float, str]
    """
    best = min(models, key=lambda m: (cers[m], models.index(m)))
    return cers[best], best


# ── Aggregation and reporting ─────────────────────────────────────────────────

def aggregate(rows: Sequence[dict], cer_key: str, mode_key: Optional[str],
              buckets: Sequence[str] = BUCKETS) -> dict:
    """Mean/median CER, failure-mode counts and per-bucket CER of one system.

    :param rows: Per-document rows (each with a ``bucket`` key).
    :type rows: Sequence[dict]
    :param cer_key: Column holding the system's CER.
    :type cer_key: str
    :param mode_key: Column holding its failure mode (``None`` to skip).
    :type mode_key: Optional[str]
    :param buckets: Bucket labels to break out, in order (default: the PGP
        script buckets).
    :type buckets: Sequence[str]
    :return: Summary dict (n, mean, median, modes, buckets).
    :rtype: dict
    """
    cers = [r[cer_key] for r in rows]
    out = {"n": len(cers), "mean": statistics.fmean(cers), "median": statistics.median(cers)}
    if mode_key:
        counts = Counter(r[mode_key] for r in rows)
        out["modes"] = {m: counts.get(m, 0) for m in MODES}
    out["buckets"] = {}
    for bucket in buckets:
        values = [r[cer_key] for r in rows if r["bucket"] == bucket]
        if values:
            out["buckets"][bucket] = {"n": len(values), "mean": statistics.fmean(values),
                                      "median": statistics.median(values)}
    return out


def summary_table(summary: dict) -> str:
    """Render a run summary as a plain-text table.

    :param summary: Dict written to ``summary.json``.
    :type summary: dict
    :return: Printable table.
    :rtype: str
    """
    systems = summary["systems"]
    lines = [f"{'system':34s} {'n':>4s} {'meanCER':>8s} {'medCER':>7s} "
             f"{'abst':>5s} {'loop':>5s} {'hall':>5s} {'subst':>5s}"]
    for name, rec in systems.items():
        modes = rec.get("modes", {})
        lines.append(f"{name:34s} {rec['n']:4d} {rec['mean']:8.3f} {rec['median']:7.3f} "
                     + " ".join(f"{modes.get(m, '-'):>5}" for m in MODES))
    present = list(dict.fromkeys(b for r in systems.values() for b in r["buckets"]))
    lines.append("")
    lines.append(f"{'mean CER by bucket':34s} "
                 + " ".join(f"{b[:13]:>13s}" for b in present))
    first = next(iter(systems.values()))
    lines.append(f"{'(n docs)':34s} "
                 + " ".join(f"{first['buckets'].get(b, {}).get('n', 0):>13d}" for b in present))
    for name, rec in systems.items():
        cells = [f"{rec['buckets'][b]['mean']:13.3f}" if b in rec["buckets"] else f"{'-':>13s}"
                 for b in present]
        lines.append(f"{name:34s} " + " ".join(cells))
    return "\n".join(lines)


def default_tag(args: argparse.Namespace) -> str:
    """Descriptive output folder name built from the run options.

    :param args: Parsed CLI arguments.
    :type args: argparse.Namespace
    :return: Tag such as ``word_unweighted_5m`` (``religious_`` prefix on that
        benchmark).
    :rtype: str
    """
    weighting = args.auto_weights or ("explicit" if args.weights else "unweighted")
    parts = [args.unit, weighting, f"{len(args.models)}m"]
    if args.filter_failures:
        parts.append(f"filter_{args.filter_source}")
    if args.benchmark == RELIGIOUS:
        parts.insert(0, RELIGIOUS)
    return "_".join(parts)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (default ``sys.argv[1:]``).
    :type argv: Optional[Sequence[str]]
    :return: Parsed arguments.
    :rtype: argparse.Namespace
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--models", nargs="+", required=True, help="model file stems")
    p.add_argument("--weights", nargs="+", type=float, help="one weight per model (default 1.0)")
    p.add_argument("--auto-weights", choices=("inv_median", "inv_median_bucket"),
                   help="weights = 1 / median CER on this benchmark, leave-one-out per "
                        "document (inv_median_bucket: within the document's script bucket)")
    p.add_argument("--unit", choices=UNITS, default="word")
    p.add_argument("--gap", type=float, help="gap cost (default: 0.35 word, 1.0 char)")
    p.add_argument("--filter-failures", action="store_true",
                   help="drop abstained / loop_collapse / hallucinated hypotheses before voting")
    p.add_argument("--filter-source", choices=("gt", "peer"), default="gt",
                   help="hallucination test against the GT (scorer's classify; not "
                        "deployable) or against the other hypotheses (GT-free)")
    p.add_argument("--backbone", default="auto", help="'auto' or a model file stem")
    p.add_argument("--benchmark", choices=tuple(SOURCES), default="verified",
                   help="PGP-131 (verified) or the frozen PGP-150 set via the offline scorer, or "
                        "religious-140 via run_religious_benchmark's spec and raw_outputs")
    p.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    p.add_argument("--tag", help="output sub-folder (default: built from the options)")
    args = p.parse_args(argv)
    if args.weights and len(args.weights) != len(args.models):
        p.error("--weights needs one value per --models entry")
    if args.weights and args.auto_weights:
        p.error("--weights and --auto-weights are exclusive")
    if args.backbone != "auto" and args.backbone not in args.models:
        p.error("--backbone must be 'auto' or one of --models")
    if len(set(args.models)) != len(args.models):
        p.error("--models has duplicates")
    return args


def consensus_for_document(raws: Sequence[str], models: Sequence[str], weights: Dict[str, float],
                           gt_modes: Sequence[str], unit: str = "word",
                           gap: Optional[float] = None, backbone: str = "auto",
                           filter_source: Optional[str] = None
                           ) -> Tuple[RoverResult, List[str], bool]:
    """Optionally drop failed readings of one page, then ROVER the rest.

    :param raws: Raw output of each model, aligned with ``models``.
    :type raws: Sequence[str]
    :param models: Model file stems.
    :type models: Sequence[str]
    :param weights: ``{model: weight}`` for this page.
    :type weights: Dict[str, float]
    :param gt_modes: The scorer's (GT-based) failure mode of each reading,
        used when ``filter_source`` is ``gt``.
    :type gt_modes: Sequence[str]
    :param unit: ``word`` or ``char``.
    :type unit: str
    :param gap: Gap cost (``None`` = unit default).
    :type gap: Optional[float]
    :param backbone: ``auto`` or a model stem (falls back to ``auto`` when
        that model was filtered out).
    :type backbone: str
    :param filter_source: ``None`` (no filtering), ``gt`` or ``peer``.
    :type filter_source: Optional[str]
    :return: (ROVER result, models that voted, whether every reading failed
        and the heaviest one was kept as a fallback).
    :rtype: Tuple[RoverResult, List[str], bool]
    """
    voters = list(range(len(models)))
    fallback = False
    if filter_source is not None:
        modes = failure_modes(raws, None, "peer") if filter_source == "peer" else list(gt_modes)
        voters = select_voters(modes, [weights[m] for m in models])
        fallback = all(mode in FAILURE_MODES for mode in modes)
    voter_models = [models[i] for i in voters]
    fixed = voter_models.index(backbone) if backbone in voter_models else None
    result = rover([raws[i] for i in voters], [weights[m] for m in voter_models],
                   unit=unit, backbone=fixed, gap=gap)
    return result, voter_models, fallback


def summarise_systems(rows: Sequence[dict], models: Sequence[str],
                      buckets: Sequence[str] = BUCKETS) -> Dict[str, dict]:
    """Aggregate every input model, the oracle and the consensus.

    :param rows: Per-document rows built by :func:`run`.
    :type rows: Sequence[dict]
    :param models: Input model stems.
    :type models: Sequence[str]
    :param buckets: The benchmark's bucket labels (see ``SOURCES``).
    :type buckets: Sequence[str]
    :return: ``{system name: aggregate}``; the consensus also has ``f1_median``.
    :rtype: Dict[str, dict]
    """
    systems = {m: aggregate(rows, f"cer_{m}", f"mode_{m}", buckets) for m in models}
    systems[f"oracle (best of {len(models)})"] = aggregate(rows, "oracle_cer", "oracle_mode",
                                                           buckets)
    systems["ROVER consensus"] = aggregate(rows, "consensus_cer", "consensus_mode", buckets)
    systems["ROVER consensus"]["f1_median"] = statistics.median(r["consensus_f1"] for r in rows)
    return systems


def run(args: argparse.Namespace) -> dict:
    """Run one ROVER configuration over the benchmark and write its outputs.

    :param args: Parsed CLI arguments.
    :type args: argparse.Namespace
    :return: The summary written to ``summary.json``.
    :rtype: dict
    """
    started = time.time()
    models = list(args.models)
    tag = args.tag or default_tag(args)
    out_dir = args.out_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    source = SOURCES[args.benchmark]
    docs, skipped = load_benchmark_documents(args.benchmark, models)
    if not docs:
        absent = [m for m in models if not any(source.outputs_dir.glob(f"*/{m}.txt"))]
        raise SystemExit(f"no document has an output from every requested model "
                         f"(no outputs at all for: {absent or 'none'})")

    singles = {d.doc_id: {m: score_output(d.raws[m], d.gt_ink, d.gt_letters) for m in models}
               for d in docs}
    cers = {doc_id: {m: s.cer for m, s in per.items()} for doc_id, per in singles.items()}
    doc_bucket = {d.doc_id: d.bucket for d in docs}
    if args.auto_weights:
        doc_weights = leave_one_out_weights(cers, models, doc_bucket,
                                            per_bucket=args.auto_weights == "inv_median_bucket")
    else:
        fixed = dict(zip(models, args.weights or [1.0] * len(models)))
        doc_weights = {d.doc_id: fixed for d in docs}

    rows, dropped, fallbacks = [], Counter(), 0
    for doc in docs:
        result, voter_models, fallback = consensus_for_document(
            [doc.raws[m] for m in models], models, doc_weights[doc.doc_id],
            [singles[doc.doc_id][m].mode for m in models], unit=args.unit, gap=args.gap,
            backbone=args.backbone,
            filter_source=args.filter_source if args.filter_failures else None)
        fallbacks += int(fallback)
        dropped.update(m for m in models if m not in voter_models)
        (out_dir / f"consensus_{doc.doc_id}.txt").write_text(result.text + "\n")
        score = score_output(result.text, doc.gt_ink, doc.gt_letters, with_f1=True)
        best_cer, best_model = oracle(cers[doc.doc_id], models)
        row = {"doc_id": doc.doc_id, "bucket": doc.bucket}
        row.update({f"cer_{m}": cers[doc.doc_id][m] for m in models})
        row.update({f"mode_{m}": singles[doc.doc_id][m].mode for m in models})
        row.update({
            "oracle_cer": best_cer, "oracle_model": best_model,
            "oracle_mode": singles[doc.doc_id][best_model].mode,
            "consensus_cer": score.cer, "consensus_f1": score.f1,
            "consensus_mode": score.mode, "backbone": voter_models[result.backbone],
            "voters": ";".join(voter_models),
            "dropped": ";".join(m for m in models if m not in voter_models),
            "n_slots": result.n_slots,
        })
        rows.append(row)

    with open(out_dir / "per_doc.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                          for r in rows])

    systems = summarise_systems(rows, models, source.buckets)
    summary = {
        "tag": tag,
        "config": {"models": models, "weights": args.weights, "auto_weights": args.auto_weights,
                   "unit": args.unit, "gap": args.gap if args.gap is not None else DEFAULT_GAP[args.unit],
                   "filter_failures": args.filter_failures,
                   "filter_source": args.filter_source if args.filter_failures else None,
                   "backbone": args.backbone, "benchmark": args.benchmark},
        "n_docs": len(docs), "skipped_missing_output": skipped,
        "systems": systems,
        "oracle_wins": dict(Counter(r["oracle_model"] for r in rows)),
        "backbones": dict(Counter(r["backbone"] for r in rows)),
        "filter": {"dropped": dict(dropped), "all_failed_fallbacks": fallbacks},
        "seconds": round(time.time() - started, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point: run one configuration and print its summary.

    :param argv: Arguments (default ``sys.argv[1:]``).
    :type argv: Optional[Sequence[str]]
    """
    args = parse_args(argv)
    summary = run(args)
    cfg = summary["config"]
    print(f"[{summary['tag']}] ROVER unit={cfg['unit']} gap={cfg['gap']} "
          f"weights={cfg['auto_weights'] or cfg['weights'] or 'equal'} "
          f"filter={cfg['filter_source'] or 'off'} backbone={cfg['backbone']} "
          f"docs={summary['n_docs']} (skipped {len(summary['skipped_missing_output'])} "
          f"missing an output)  {summary['seconds']}s\n")
    print(summary_table(summary))
    if cfg["filter_failures"]:
        print(f"\ndropped before voting: {summary['filter']['dropped']}  "
              f"all-failed fallbacks: {summary['filter']['all_failed_fallbacks']}")
    print(f"\nWrote {args.out_dir / summary['tag']}/(per_doc.csv, summary.json, consensus_*.txt)")


if __name__ == "__main__":
    main()
