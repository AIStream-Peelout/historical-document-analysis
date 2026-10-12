# File name: slot_oracle.py
# Date: 10/7/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Per-slot oracle over the ROVER network: the ceiling for a learned voter (offline).

For each page the readers are combined into the same word/character transition
network that ``rover_consensus.py`` builds (same line-wise normalisation,
tokenisation, backbone rule, merge order and gap costs, equal weights).  The
ground truth is then aligned into that network as one more hypothesis with the
tool's own DP (``align_to_network``) and merged as a last entry column, but it
never votes.  Per slot:

* **slot oracle**: emit the reader entry equal to the GT entry when any reader
  has it.  NULL counts as an entry, so a reader that skipped the slot lets the
  oracle drop an insertion.  Otherwise emit the reader entry with the smallest
  normalised edit distance to the GT entry (ties: the backbone's entry, then
  the entry more readers wrote, then merge order).  Slots that only the GT
  fills (every reader NULL) emit nothing.  At word level this is the
  *attested-word oracle*: its output only holds words some reader wrote at
  that slot.
* **decomposition**: a slot where the plain, unweighted ROVER vote differs from
  the GT entry is *learnable* when some reader had the GT entry there and
  *unreachable* when none did.  The slot oracle is wrong exactly on the
  unreachable slots.  Two diagnostics qualify the split: how often a losing
  candidate written by a single reader is the GT entry (the base rate a voter
  faces when it overrules the majority), and how much unreachable GT text sits
  in *blocks* (stretches where almost nothing was read right: missed or extra
  lines, another text layer, reading order) rather than in local misreadings.

Entries are compared in the scorer's whitespace semantics (``cer_pair``
collapses whitespace): a newline token equals a space at character level and
equals NULL (a word separator) at word level.

The GT side is the scoring reference itself.  ``gt_ink_lines`` reruns
``genizah_visible_ink_gt``'s cleaning without its final whitespace collapse, so
the reference keeps its line breaks, and a page whose lines do not rejoin to
the reference exactly falls back to one line.  It is tokenised like a reading
(``tokenize``) but not passed through ``normalize_ink_hypothesis``: that would
delete the few ``?``, ``*`` and ``/`` characters some references keep, which
the CER still counts, so here they are unreachable.

Approximations (reported by the run):

* The GT is aligned by the tool's sum-of-pairs DP against the whole network,
  not optimally against each reader.  A GT entry can land in a slot of its own
  while a reader wrote it one or two slots away (reading-order swaps,
  ambiguous gaps).  ``near_miss`` counts unreachable GT entries that some
  reader wrote within ``NEAR_MISS_WINDOW`` slots, in a slot where the GT is
  NULL.
* The choice is greedy per slot.  For words the fallback ranks by normalised
  edit distance, as specified, not by the edit cost it adds to the page CER.

ROVER and oracle texts are scored with ``rover_consensus.score_output`` (the
offline scorer's strict CER and aligned F1); best single reader and per-page
oracle use the same scores.

``--dump-blocks OUT.jsonl`` writes the unreachable *blocks* instead (char
level): stretches of at least ``--min-block-chars`` GT characters where
:func:`block_mask`'s density rule holds, each with the GT text (visible ink and
the raw GT around it with its editorial marks), what every reader wrote over
the stretch, its line index and the page's reconstruction share.

Usage (from the repo root):
    .venv/bin/python src/datasets/evaluations/helper_eval_scripts/slot_oracle.py \\
        --models qwen3_vl_8b_heb_v21b_step1200 qwen3_vl_8b_heb_v22a_step1300 \\
                 qwen3_vl_8b_heb_v22b_step1200 qwen3_vl_8b_heb_v22i_a50 \\
                 qwen3_vl_8b_heb_v23a_step1200 qwen3_vl_8b_heb_v23a_step1500 \\
        --tag pgp131_6vlm
    .venv/bin/python src/datasets/evaluations/helper_eval_scripts/slot_oracle.py \\
        --benchmark religious --models qwen3_vl_8b_heb_v21b_step1200 \\
        qwen3_vl_8b_heb_v22a_step1300 qwen3_vl_8b_heb_v22b_step1200 qwen3_vl_8b_heb_v22i_a50 \\
        kraken_seg_k7_ktivseg_e10_rgb_mv --tag religious_4vlm_kseg
    .venv/bin/python src/datasets/evaluations/helper_eval_scripts/slot_oracle.py \\
        --models qwen3_vl_8b_heb_v21b_step1200 qwen3_vl_8b_heb_v22a_step1300 \\
                 qwen3_vl_8b_heb_v22b_step1200 qwen3_vl_8b_heb_v22i_a50 \\
                 qwen3_vl_8b_heb_v23a_step1200 qwen3_vl_8b_heb_v23a_step1500 \\
        --dump-blocks logs/next_round/consensus/unreachable/blocks_pgp131_6vlm.jsonl
"""

import argparse
import csv
import json
import statistics
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from rapidfuzz.distance import Indel, Levenshtein

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.datasets.cleaning.clean_genizah_transcriptions import GAP, clean_diplomatic  # noqa: E402
from src.datasets.evaluations.metrics import (  # noqa: E402
    dedup_repeated_tail,
    normalize_ink_hypothesis,
    normalize_whitespace,
)
from src.datasets.evaluations.helper_eval_scripts.rover_consensus import (  # noqa: E402
    COST_SCALE,
    DEFAULT_GAP,
    NEWLINE,
    RELIGIOUS,
    SOURCES,
    UNITS,
    BenchDoc,
    Network,
    aggregate,
    align_to_network,
    build_network,
    choose_backbone,
    detokenize,
    load_benchmark_documents,
    merge_into_network,
    merge_order,
    normalize_lines,
    oracle,
    score_output,
    similarity_matrix,
    tokenize,
    vote_slot,
)
from src.datasets.evaluations.helper_eval_scripts.score_genizah_offline import (  # noqa: E402
    load_benchmark,
)

CLASSES = ("right", "learnable", "unreachable")
ERROR_TYPES = ("substitution", "deletion", "insertion")
NEAR_MISS_WINDOW = 2                  # slots on each side searched for a misaligned GT entry
# An unreachable GT entry is in a "block" when at least BLOCK_SHARE of the GT entries within
# BLOCK_WINDOW GT entries on each side are unreachable too: a stretch nobody read right (missed or
# extra lines, another text layer or reading order), as opposed to a local misreading.
BLOCK_WINDOW = {"word": 2, "char": 10}
BLOCK_SHARE = 0.7
MIN_BLOCK_CHARS = 8                   # shortest block --dump-blocks writes (GT characters)
RAW_CONTEXT = 20                      # raw GT characters shown on each side of a dumped block
_DEFAULT_OUT = _REPO / "logs/next_round/consensus/slot_oracle"


# ── Ground truth with line structure ──────────────────────────────────────────

def gt_ink_lines(gt_raw: str) -> List[str]:
    """Visible-ink scoring reference of a raw GT, one string per line.

    Same steps as ``genizah_visible_ink_gt`` (tail dedup, ``clean_diplomatic``,
    gap tokens to spaces, ``@`` removed) except that whitespace is collapsed per
    line instead of over the whole text, so ``" ".join(lines)`` is the
    reference whenever that function is unchanged.

    :param gt_raw: Raw diplomatic ground truth (as the benchmark loader returns it).
    :type gt_raw: str
    :return: Non-empty reference lines in order.
    :rtype: List[str]
    """
    cleaned, _recon, _visible = clean_diplomatic(dedup_repeated_tail(gt_raw))
    cleaned = cleaned.replace(GAP, " ").replace("@", "")
    lines = (normalize_whitespace(line) for line in cleaned.split("\n"))
    return [line for line in lines if line]


def reference_lines(gt_raw: str, gt_ink: str) -> Tuple[List[str], bool]:
    """Reference lines that rejoin exactly to the scored reference.

    :param gt_raw: Raw diplomatic ground truth.
    :type gt_raw: str
    :param gt_ink: Scoring reference (``genizah_visible_ink_gt`` of ``gt_raw``).
    :type gt_ink: str
    :return: (lines, whether the one-line fallback was used).
    :rtype: Tuple[List[str], bool]
    """
    lines = gt_ink_lines(gt_raw)
    if " ".join(lines) == gt_ink:
        return lines, False
    return ([gt_ink] if gt_ink else []), True


def load_raw_gts(benchmark: str) -> Dict[str, str]:
    """Raw GT per document from the same spec the ROVER loaders read.

    :param benchmark: A key of ``rover_consensus.SOURCES``.
    :type benchmark: str
    :return: ``{doc_id: raw GT}`` (PGP: after the scorer's duplication repair).
    :rtype: Dict[str, str]
    """
    if benchmark == RELIGIOUS:
        spec = json.loads(SOURCES[RELIGIOUS].spec.read_text())
        return {d["doc_id"]: d["gt"] for d in spec["docs"]}
    return {d["doc_id"]: d["gt"] for d in load_benchmark(benchmark)}


def alignment_raw(gt_raw: str) -> str:
    """The raw GT as the visible-ink cleaning sees it (tail dedup, NFC).

    :param gt_raw: Raw diplomatic ground truth.
    :type gt_raw: str
    :return: Text whose characters :func:`raw_positions` indexes.
    :rtype: str
    """
    return unicodedata.normalize("NFC", dedup_repeated_tail(gt_raw))


def page_metadata(benchmark: str) -> Dict[str, dict]:
    """Shelf mark, link fields and reconstruction share of every benchmark page.

    :param benchmark: A key of ``rover_consensus.SOURCES``.
    :type benchmark: str
    :return: ``{doc_id: fields}``.  PGP: ``shelf_mark``, ``institution``,
        ``image_url``, ``recon_frac`` and ``gaps`` from the spec.  Religious:
        ``shelf_mark``, ``sys_num``, ``fl``, ``n_columns``, ``genre`` and
        ``recon_frac`` recomputed as the cleaning computes it (restored letters
        over restored plus visible letters; the spec has no such field).
    :rtype: Dict[str, dict]
    """
    if benchmark != RELIGIOUS:
        return {d["doc_id"]: {"shelf_mark": d.get("shelf_mark"), "institution": d.get("institution"),
                              "image_url": d.get("image_url"), "recon_frac": d.get("recon_frac"),
                              "gaps": d.get("gaps")}
                for d in load_benchmark(benchmark)}
    out = {}
    for d in json.loads(SOURCES[RELIGIOUS].spec.read_text())["docs"]:
        _, recon, visible = clean_diplomatic(dedup_repeated_tail(d["gt"]))
        out[d["doc_id"]] = {"shelf_mark": d.get("shelf_mark"), "sys_num": d.get("sys_num"),
                            "fl": d.get("fl"), "n_columns": d.get("n_columns"),
                            "genre": d.get("genre"),
                            "recon_frac": round(recon / max(recon + visible, 1), 4)}
    return out


# ── Network with the reference column ─────────────────────────────────────────

@dataclass(frozen=True)
class PageNetwork:
    """ROVER network of one page, built as ``rover_consensus.rover`` builds it.

    :param network: Reader network; entry 0 of every slot is the backbone's.
    :param backbone: Hypothesis index of the backbone.
    :param order: Merge order.
    :param gap_units: Gap cost per entry in integer cost units.
    """

    network: Network
    backbone: int
    order: Tuple[int, ...]
    gap_units: int


def build_page_network(raws: Sequence[str], unit: str,
                       gap: Optional[float] = None) -> PageNetwork:
    """Build the unweighted ROVER network of one page.

    The steps and calls are those of ``rover_consensus.rover`` with equal
    weights, so voting this network reproduces its consensus text.

    :param raws: Raw reader outputs (normalised here, line by line).
    :type raws: Sequence[str]
    :param unit: ``word`` or ``char``.
    :type unit: str
    :param gap: Gap cost in edit-distance units (default per unit).
    :type gap: Optional[float]
    :return: Network, backbone, merge order and gap cost.
    :rtype: PageNetwork
    """
    weights = [1.0] * len(raws)
    lines = [normalize_lines(h) for h in raws]
    tokens = [tokenize(ls, unit) for ls in lines]
    sim = similarity_matrix([" ".join(ls) for ls in lines])
    backbone = choose_backbone(sim, weights)
    order = merge_order(sim, backbone)
    gap_units = int(round((DEFAULT_GAP[unit] if gap is None else gap) * COST_SCALE))
    return PageNetwork(build_network(tokens, order, gap_units), backbone, tuple(order), gap_units)


def add_reference(page: PageNetwork, reference: Sequence[str]) -> Network:
    """Align the GT tokens into the network as a last, non-voting entry column.

    :param page: Reader network of the page.
    :type page: PageNetwork
    :param reference: GT tokens (``tokenize`` of the reference lines).
    :type reference: Sequence[str]
    :return: Network with one more column; slots the GT alone fills have NULL
        in every reader column.  The reference's member index is the number of
        readers.
    :rtype: Network
    """
    path = align_to_network(page.network, reference, page.gap_units)
    return merge_into_network(page.network, reference, path, len(page.network.members))


# ── Per-slot rules ────────────────────────────────────────────────────────────

def token_key(token: Optional[str], unit: str) -> Optional[str]:
    """Scoring-equivalent form of a slot entry.

    :param token: Slot entry (``None`` = NULL).
    :type token: Optional[str]
    :param unit: ``word`` or ``char``.
    :type unit: str
    :return: A newline as a space (char) or as NULL (word, where it is only a
        separator); any other entry unchanged.
    :rtype: Optional[str]
    """
    if token == NEWLINE:
        return " " if unit == "char" else None
    return token


def edit_distance_ratio(key: Optional[str], target: Optional[str]) -> Fraction:
    """Normalised edit distance as an exact fraction (NULL = empty string).

    :param key: Candidate entry key.
    :type key: Optional[str]
    :param target: GT entry key.
    :type target: Optional[str]
    :return: Levenshtein distance / length of the longer string (0 for two NULLs).
    :rtype: Fraction
    """
    a, b = key or "", target or ""
    longest = max(len(a), len(b))
    return Fraction(Levenshtein.distance(a, b), longest) if longest else Fraction(0)


def oracle_entry(entries: Sequence[Optional[str]], reference: Optional[str],
                 unit: str) -> Optional[str]:
    """The slot oracle's choice among the reader entries of one slot.

    :param entries: Reader entries in merge order (index 0 = backbone).
    :type entries: Sequence[Optional[str]]
    :param reference: GT entry of the slot (``None`` = GT NULL).
    :type reference: Optional[str]
    :param unit: ``word`` or ``char``.
    :type unit: str
    :return: The entry equal to the GT (exact, then scoring-equivalent); else
        the closest entry by normalised edit distance, ties to the backbone's
        entry, then to the entry more readers wrote, then to merge order.
        ``None`` when every reader is NULL (a GT-only slot).
    :rtype: Optional[str]
    """
    if reference in entries:
        return reference
    target = token_key(reference, unit)
    keys = [token_key(e, unit) for e in entries]
    if target in keys:
        return entries[keys.index(target)]
    distance = [edit_distance_ratio(k, target) for k in keys]
    best = min(distance)
    tied = [i for i, d in enumerate(distance) if d == best]
    if tied[0] == 0:
        return entries[0]
    support = Counter(keys)
    return entries[min(tied, key=lambda i: (-support[keys[i]], i))]


def classify_slot(entries: Sequence[Optional[str]], reference: Optional[str],
                  voted: Optional[str], unit: str) -> Tuple[str, Optional[str]]:
    """Class and error type of the ROVER vote at one slot.

    :param entries: Reader entries of the slot.
    :type entries: Sequence[Optional[str]]
    :param reference: GT entry.
    :type reference: Optional[str]
    :param voted: ROVER's winning entry.
    :type voted: Optional[str]
    :param unit: ``word`` or ``char``.
    :type unit: str
    :return: (``right`` | ``learnable`` | ``unreachable``, error type or
        ``None`` when right).  Error types: ``insertion`` (GT NULL),
        ``deletion`` (vote NULL), ``substitution``.
    :rtype: Tuple[str, Optional[str]]
    """
    target, got = token_key(reference, unit), token_key(voted, unit)
    if got == target:
        return "right", None
    reachable = any(token_key(e, unit) == target for e in entries)
    error = "insertion" if target is None else "deletion" if got is None else "substitution"
    return ("learnable" if reachable else "unreachable"), error


# ── One page ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PageResult:
    """Slot oracle and ROVER outputs of one page plus its slot statistics.

    :param rover_text: Unweighted ROVER consensus (equals ``rover(...).text``).
    :param oracle_text: Slot-oracle text.
    :param backbone: Hypothesis index of the backbone.
    :param stats: Slot counts (see :func:`slot_statistics`).
    """

    rover_text: str
    oracle_text: str
    backbone: int
    stats: Dict[str, int]


def near_misses(targets: Sequence[Optional[str]], reader_keys: Sequence[set],
                unreachable: Sequence[int], window: int = NEAR_MISS_WINDOW) -> List[int]:
    """Unreachable slots whose GT entry a reader wrote in a nearby GT-NULL slot.

    :param targets: GT key per slot.
    :type targets: Sequence[Optional[str]]
    :param reader_keys: Set of reader entry keys per slot.
    :type reader_keys: Sequence[set]
    :param unreachable: Indices of unreachable slots.
    :type unreachable: Sequence[int]
    :param window: Slots searched on each side.
    :type window: int
    :return: Indices of the near-miss slots (GT entry not NULL).
    :rtype: List[int]
    """
    hits = []
    for i in unreachable:
        target = targets[i]
        if target is None:
            continue
        lo, hi = max(0, i - window), min(len(targets), i + window + 1)
        if any(targets[j] is None and target in reader_keys[j] for j in range(lo, hi) if j != i):
            hits.append(i)
    return hits


def block_mask(flags: Sequence[bool], window: int, share: float = BLOCK_SHARE) -> List[bool]:
    """Flags that sit in a dense stretch of flags.

    :param flags: One flag per GT entry, in order (``True`` = unreachable).
    :type flags: Sequence[bool]
    :param window: Entries on each side (truncated at the page edges).
    :type window: int
    :param share: Minimum share of flagged entries in the window.
    :type share: float
    :return: ``True`` where the entry is flagged and its window is at least
        ``share`` flagged.
    :rtype: List[bool]
    """
    prefix = [0]
    for flag in flags:
        prefix.append(prefix[-1] + int(flag))
    n = len(flags)
    out = []
    for i, flag in enumerate(flags):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        out.append(bool(flag) and prefix[hi] - prefix[lo] >= share * (hi - lo))
    return out


def count_dissent(stats: Counter, keys: Sequence[Optional[str]], winner: Optional[str],
                  target: Optional[str], members: Sequence[int]) -> None:
    """Tally the losing candidates of one slot and how often they match the GT.

    ``dissent_lone`` / ``dissent_multi``: candidate keys other than the
    winner's written by one / several readers; ``*_right``: of those, the ones
    equal to the GT key.  ``lone_<member>`` / ``lone_right_<member>`` split the
    lone dissents by reader.

    :param stats: Counter updated in place.
    :type stats: Counter
    :param keys: Reader entry keys of the slot.
    :type keys: Sequence[Optional[str]]
    :param winner: Key of the ROVER winner.
    :type winner: Optional[str]
    :param target: GT key.
    :type target: Optional[str]
    :param members: Hypothesis index of each reader entry.
    :type members: Sequence[int]
    """
    for key, n in Counter(keys).items():
        if key == winner:
            continue
        kind = "lone" if n == 1 else "multi"
        right = int(key == target)
        stats[f"dissent_{kind}"] += 1
        stats[f"dissent_{kind}_right"] += right
        if n == 1:
            member = members[keys.index(key)]
            stats[f"lone_{member}"] += 1
            stats[f"lone_right_{member}"] += right


def slot_statistics(network: Network, backbone: int, unit: str
                    ) -> Tuple[List[Optional[str]], List[Optional[str]], Dict[str, int]]:
    """Vote, oracle and classify every slot of a network with a reference column.

    Statistics keys: ``slots``; ``gt_chars`` (characters of the GT entries:
    one per character token, a word's length per word token, nothing for word
    separators); ``<class>_slots`` and ``<class>_chars`` per class;
    ``<class>_<error type>`` slot counts for the two wrong classes;
    ``oracle_wrong_slots`` (equals ``unreachable_slots`` by construction);
    ``learnable_lost_tie`` (the GT key had as many readers as the winner) and
    ``learnable_single`` (one reader had it); ``gt_only_slots`` /
    ``gt_only_chars`` (every reader NULL); ``near_miss_slots`` /
    ``near_miss_chars`` (see :func:`near_misses`); ``unreachable_block_chars``
    (see :func:`block_mask`); and the dissent tallies of :func:`count_dissent`.

    :param network: Network from :func:`add_reference` (reference last).
    :type network: Network
    :param backbone: Hypothesis index of the backbone (tie-break of the vote).
    :type backbone: int
    :param unit: ``word`` or ``char``.
    :type unit: str
    :return: (ROVER entries, oracle entries, statistics); the entry lists hold
        one entry per slot, ``None`` where nothing is emitted.
    :rtype: Tuple[List[Optional[str]], List[Optional[str]], Dict[str, int]]
    """
    members = network.members[:-1]
    weights = [1.0] * len(members)
    stats: Counter = Counter()
    voted_all, chosen_all, targets, reader_keys, unreachable = [], [], [], [], []
    gt_entries: List[Tuple[int, bool]] = []          # (characters, unreachable) per GT entry
    for i, slot in enumerate(network.slots):
        entries, reference = slot[:-1], slot[-1]
        voted = vote_slot(entries, members, weights, backbone)
        chosen = oracle_entry(entries, reference, unit)
        voted_all.append(voted)
        chosen_all.append(chosen)
        keys = [token_key(e, unit) for e in entries]
        target = token_key(reference, unit)
        targets.append(target)
        reader_keys.append(set(keys))
        chars = len(target) if target else 0
        cls, error = classify_slot(entries, reference, voted, unit)
        stats["slots"] += 1
        stats["gt_chars"] += chars
        stats[f"{cls}_slots"] += 1
        stats[f"{cls}_chars"] += chars
        stats["oracle_wrong_slots"] += int(classify_slot(entries, reference, chosen, unit)[0] != "right")
        if error is not None:
            stats[f"{cls}_{error}"] += 1
        if all(e is None for e in entries):
            stats["gt_only_slots"] += 1
            stats["gt_only_chars"] += chars
        if cls == "learnable":
            support = Counter(keys)
            stats["learnable_lost_tie"] += int(support[target] == support[token_key(voted, unit)])
            stats["learnable_single"] += int(support[target] == 1)
        elif cls == "unreachable":
            unreachable.append(i)
        count_dissent(stats, keys, token_key(voted, unit), target, members)
        if target is not None:
            gt_entries.append((chars, cls == "unreachable"))
    for i in near_misses(targets, reader_keys, unreachable):
        stats["near_miss_slots"] += 1
        stats["near_miss_chars"] += len(targets[i])
    in_block = block_mask([flag for _, flag in gt_entries], BLOCK_WINDOW[unit])
    stats["unreachable_block_chars"] += sum(c for (c, _), b in zip(gt_entries, in_block) if b)
    return voted_all, chosen_all, dict(stats)


def analyse_page(raws: Sequence[str], gt_lines: Sequence[str], unit: str,
                 gap: Optional[float] = None) -> PageResult:
    """ROVER, slot oracle and slot decomposition of one page.

    :param raws: Raw reader outputs.
    :type raws: Sequence[str]
    :param gt_lines: Reference lines (see :func:`reference_lines`).
    :type gt_lines: Sequence[str]
    :param unit: ``word`` or ``char``.
    :type unit: str
    :param gap: Gap cost in edit-distance units (default per unit).
    :type gap: Optional[float]
    :return: Texts, backbone and statistics.
    :rtype: PageResult
    :raises ValueError: For an unknown unit or no readers.
    """
    if unit not in UNITS:
        raise ValueError(f"unknown unit {unit!r}; expected one of {UNITS}")
    if not raws:
        raise ValueError("analyse_page needs at least one reader")
    page = build_page_network(raws, unit, gap)
    merged = add_reference(page, tokenize(gt_lines, unit))
    voted, chosen, stats = slot_statistics(merged, page.backbone, unit)
    rover_text = detokenize([t for t in voted if t is not None], unit)
    oracle_text = detokenize([t for t in chosen if t is not None], unit)
    return PageResult(rover_text, oracle_text, page.backbone, stats)


# ── Unreachable blocks (char level) ───────────────────────────────────────────

@dataclass(frozen=True)
class GtEntry:
    """ROVER outcome at the slot of one GT entry.

    :param slot: Slot index in the network with the reference column.
    :param cls: ``right``, ``learnable`` or ``unreachable``.
    :param error: Error type (``None`` when right).
    """

    slot: int
    cls: str
    error: Optional[str]


def gt_entries(network: Network, backbone: int, unit: str) -> List[GtEntry]:
    """Classify the ROVER vote at every slot the GT fills, in GT order.

    The vote and the classes are those of :func:`slot_statistics`; at char
    level there is one entry per GT character (newlines included, as spaces).

    :param network: Network from :func:`add_reference` (reference last).
    :type network: Network
    :param backbone: Hypothesis index of the backbone (tie-break of the vote).
    :type backbone: int
    :param unit: ``word`` or ``char``.
    :type unit: str
    :return: One entry per non-NULL GT key, in order.
    :rtype: List[GtEntry]
    """
    members = network.members[:-1]
    weights = [1.0] * len(members)
    out = []
    for i, slot in enumerate(network.slots):
        entries, reference = slot[:-1], slot[-1]
        if token_key(reference, unit) is None:
            continue
        voted = vote_slot(entries, members, weights, backbone)
        cls, error = classify_slot(entries, reference, voted, unit)
        out.append(GtEntry(i, cls, error))
    return out


def block_spans(flags: Sequence[bool], window: int, share: float = BLOCK_SHARE,
                min_len: int = 1, extend: bool = False) -> List[Tuple[int, int]]:
    """Stretches of dense flags, as half-open index ranges.

    A position is dense when at least ``share`` of the flags within ``window``
    positions on each side are set (the window of :func:`block_mask`, applied
    to every position).  A block core is a maximal run of dense positions
    trimmed to start and end on a flag, so the flags inside cores are exactly
    the ones :func:`block_mask` marks; an unflagged position inside a core (a
    character some reader got right amid the stretch) stays in it.  The window
    reaches the density only a few positions into a stretch, so with
    ``extend`` each core also takes the run of flags touching it on either
    side (cores that then meet are merged): the whole stretch nobody read
    right, as shown to a reader of the dump.

    :param flags: One flag per GT entry, in order (``True`` = unreachable).
    :type flags: Sequence[bool]
    :param window: Entries on each side (truncated at the page edges).
    :type window: int
    :param share: Minimum share of flagged entries in the window.
    :type share: float
    :param min_len: Shortest block kept (entries, after trimming/extending).
    :type min_len: int
    :param extend: Grow each core over the flags adjacent to it.
    :type extend: bool
    :return: ``(start, end)`` pairs in order, disjoint.
    :rtype: List[Tuple[int, int]]
    """
    prefix = [0]
    for flag in flags:
        prefix.append(prefix[-1] + int(flag))
    n = len(flags)
    dense = [prefix[min(n, i + window + 1)] - prefix[max(0, i - window)]
             >= share * (min(n, i + window + 1) - max(0, i - window)) for i in range(n)]
    spans: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if not dense[i]:
            i += 1
            continue
        j = i
        while j < n and dense[j]:
            j += 1
        start, end = i, j
        while start < end and not flags[start]:
            start += 1
        while end > start and not flags[end - 1]:
            end -= 1
        if extend:
            while start > 0 and flags[start - 1]:
                start -= 1
            while end < n and flags[end]:
                end += 1
        if end > start:
            if spans and start <= spans[-1][1]:
                start = spans.pop()[0]
            spans.append((start, end))
        i = j
    return [(s, e) for s, e in spans if e - s >= min_len]


def reader_texts(network: Network, entries: Sequence[GtEntry], start: int, end: int) -> List[str]:
    """What each reader wrote over GT entries ``start`` to ``end - 1`` (char level).

    The slot range runs from just after the previous GT entry's slot to just
    before the next one's, so reader insertions next to the stretch count.

    :param network: Char network from :func:`add_reference` (reference last).
    :type network: Network
    :param entries: Its GT entries (:func:`gt_entries`).
    :type entries: Sequence[GtEntry]
    :param start: First GT entry of the stretch.
    :type start: int
    :param end: One past its last GT entry.
    :type end: int
    :return: One string per reader column, in merge order (``network.members``
        without the reference); NULL entries skipped, newline tokens kept.
    :rtype: List[str]
    """
    lo = entries[start - 1].slot + 1 if start > 0 else 0
    hi = entries[end].slot if end < len(entries) else len(network.slots)
    return ["".join(slot[c] for slot in network.slots[lo:hi] if slot[c] is not None)
            for c in range(len(network.members) - 1)]


def token_lines(lines: Sequence[str]) -> List[int]:
    """Line index of every character token of the reference.

    :param lines: Reference lines; the tokens are ``tokenize(lines, "char")``.
    :type lines: Sequence[str]
    :return: One index per token; a newline token belongs to the line it ends.
    :rtype: List[int]
    """
    out: List[int] = []
    for i, line in enumerate(lines):
        out.extend([i] * len(line))
        if i < len(lines) - 1:
            out.append(i)
    return out


def aligned_positions(text: str, other: str) -> List[Optional[int]]:
    """Position in ``other`` of each character of ``text`` (indel / LCS alignment).

    :param text: Text whose characters are placed (e.g. the visible-ink GT).
    :type text: str
    :param other: Text they are placed in (e.g. the raw GT).
    :type other: str
    :return: Index into ``other`` for each matched character, ``None`` for a
        character ``other`` does not have at that point.
    :rtype: List[Optional[int]]
    """
    out: List[Optional[int]] = [None] * len(text)
    for tag, i1, i2, j1, _j2 in Indel.opcodes(text, other):
        if tag == "equal":
            out[i1:i2] = range(j1, j1 + i2 - i1)
    return out


def raw_positions(text: str, raw: str) -> List[int]:
    """Raw-GT position of every character of the visible-ink text.

    :param text: Visible-ink text (``"\\n".join`` of the reference lines).
    :type text: str
    :param raw: Raw GT (:func:`alignment_raw`).
    :type raw: str
    :return: The matched raw index; a character the raw text lacks (a space
        left by a removed mark) takes the position of the next matched
        character (the last raw index at the end).
    :rtype: List[int]
    """
    matched = aligned_positions(text, raw)
    out = [0] * len(text)
    following = max(len(raw) - 1, 0)
    for i in range(len(text) - 1, -1, -1):
        if matched[i] is not None:
            following = matched[i]
        out[i] = following
    return out


def raw_context(raw: str, lo: int, hi: int, context: int = RAW_CONTEXT) -> str:
    """Raw GT around a stretch, the stretch marked with ``⟦ ⟧``.

    :param raw: Raw GT.
    :type raw: str
    :param lo: Raw index of the stretch's first character.
    :type lo: int
    :param hi: Raw index of its last character (inclusive).
    :type hi: int
    :param context: Characters shown on each side.
    :type context: int
    :return: The excerpt.
    :rtype: str
    """
    hi = max(hi, lo)
    return (raw[max(0, lo - context):lo] + "⟦" + raw[lo:hi + 1] + "⟧"
            + raw[hi + 1:hi + 1 + context])


@dataclass(frozen=True)
class PageChars:
    """Char-level ROVER network of one page with the GT column and its GT entries.

    :param page: Reader network (backbone, merge order, gap).
    :param network: The network with the reference merged in as the last column.
    :param entries: One :class:`GtEntry` per GT character.
    :param tokens: GT character tokens (``tokenize(lines, "char")``).
    """

    page: PageNetwork
    network: Network
    entries: List[GtEntry]
    tokens: List[str]


def page_chars(raws: Sequence[str], gt_lines: Sequence[str],
               gap: Optional[float] = None) -> PageChars:
    """Build the char network of one page and classify every GT character.

    :param raws: Raw reader outputs.
    :type raws: Sequence[str]
    :param gt_lines: Reference lines (see :func:`reference_lines`).
    :type gt_lines: Sequence[str]
    :param gap: Gap cost in edit-distance units (default for chars).
    :type gap: Optional[float]
    :return: Network, entries and GT tokens.
    :rtype: PageChars
    """
    page = build_page_network(raws, "char", gap)
    tokens = tokenize(gt_lines, "char")
    network = add_reference(page, tokens)
    return PageChars(page, network, gt_entries(network, page.backbone, "char"), tokens)


def page_blocks(chars: PageChars, models: Sequence[str], lines: Sequence[str], raw: str,
                min_len: int = MIN_BLOCK_CHARS) -> List[dict]:
    """The unreachable blocks of one page with what every reader wrote there.

    :param chars: Output of :func:`page_chars`.
    :type chars: PageChars
    :param models: Reader stems, indexed by hypothesis index.
    :type models: Sequence[str]
    :param lines: Reference lines the network was built with.
    :type lines: Sequence[str]
    :param raw: Raw GT (:func:`alignment_raw`).
    :type raw: str
    :param min_len: Shortest block kept (GT characters).
    :type min_len: int
    :return: One dict per block (:func:`block_spans` with ``extend``): GT
        entry range ``start``/``end``, ``length``, ``unreachable`` and
        ``gt_only`` characters (every reader NULL), line indices, relative
        position, ``gt_text`` (visible ink, newlines kept), ``raw_span`` /
        ``raw_context`` (raw GT with its marks) and ``readers``
        (``{model: text}``).
    :rtype: List[dict]
    """
    flags = [e.cls == "unreachable" for e in chars.entries]
    ink = "".join(chars.tokens)
    line_of = token_lines(lines)
    positions = raw_positions(ink, raw)
    members = chars.network.members[:-1]
    blocks = []
    for start, end in block_spans(flags, BLOCK_WINDOW["char"], min_len=min_len, extend=True):
        texts = reader_texts(chars.network, chars.entries, start, end)
        lo, hi = positions[start], positions[end - 1]
        blocks.append({
            "start": start, "end": end, "length": end - start,
            "unreachable": sum(flags[start:end]),
            "gt_only": sum(all(e is None for e in chars.network.slots[x.slot][:-1])
                           for x in chars.entries[start:end]),
            "line_start": line_of[start], "line_end": line_of[end - 1], "n_lines": len(lines),
            "rel_start": round(start / len(flags), 4), "rel_end": round(end / len(flags), 4),
            "gt_text": ink[start:end],
            "raw_span": [lo, hi],
            "raw_context": raw_context(raw, lo, hi),
            "readers": {models[m]: texts[c] for c, m in enumerate(members)},
        })
    return blocks


def iter_pages(benchmark: str, models: Sequence[str]
               ) -> Iterator[Tuple[BenchDoc, List[str], str, PageChars]]:
    """Char networks of every scored page that has an output from every reader.

    :param benchmark: A key of ``rover_consensus.SOURCES``.
    :type benchmark: str
    :param models: Reader stems (ROVER order).
    :type models: Sequence[str]
    :return: ``(document, reference lines, raw GT, page chars)`` per page.
    :rtype: Iterator[Tuple[BenchDoc, List[str], str, PageChars]]
    :raises SystemExit: When no document has an output from every reader.
    """
    docs, _skipped = load_benchmark_documents(benchmark, models)
    if not docs:
        raise SystemExit("no document has an output from every requested model")
    raw_gts = load_raw_gts(benchmark)
    for doc in docs:
        lines, _fallback = reference_lines(raw_gts[doc.doc_id], doc.gt_ink)
        yield (doc, lines, alignment_raw(raw_gts[doc.doc_id]),
               page_chars([doc.raws[m] for m in models], lines))


def collect_blocks(benchmark: str, models: Sequence[str],
                   min_len: int = MIN_BLOCK_CHARS) -> List[dict]:
    """Unreachable blocks of every page, with page fields added.

    :param benchmark: A key of ``rover_consensus.SOURCES``.
    :type benchmark: str
    :param models: Reader stems (ROVER order).
    :type models: Sequence[str]
    :param min_len: Shortest block kept (GT characters).
    :type min_len: int
    :return: Blocks of :func:`page_blocks` with ``block_id``, ``doc_id``,
        ``bucket``, ``backbone`` and the :func:`page_metadata` fields.
    :rtype: List[dict]
    """
    meta = page_metadata(benchmark)
    out = []
    for doc, lines, raw, chars in iter_pages(benchmark, models):
        for block in page_blocks(chars, models, lines, raw, min_len):
            out.append({"block_id": f"{doc.doc_id}:{block['start']}", "doc_id": doc.doc_id,
                        "bucket": doc.bucket, "backbone": models[chars.page.backbone],
                        **meta.get(doc.doc_id, {}), **block})
    return out


def write_jsonl(path: Path, records: Sequence[dict]) -> None:
    """Write records as JSON lines (UTF-8, Hebrew unescaped).

    :param path: Output file (parent folders created).
    :type path: Path
    :param records: JSON-serialisable dicts.
    :type records: Sequence[dict]
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Benchmark run ─────────────────────────────────────────────────────────────

def system_key(name: str, unit: str) -> str:
    """Per-document column name of a combined system.

    :param name: ``rover`` or ``slot_oracle``.
    :type name: str
    :param unit: ``word`` or ``char``.
    :type unit: str
    :return: Column stem such as ``rover_char``.
    :rtype: str
    """
    return f"{name}_{unit}"


def best_single(rows: Sequence[dict], models: Sequence[str]) -> Dict[str, Tuple[str, float]]:
    """Best single reader by mean CER and by median CER.

    :param rows: Per-document rows with ``cer_<model>`` columns.
    :type rows: Sequence[dict]
    :param models: Reader stems (ties: the earlier one).
    :type models: Sequence[str]
    :return: ``{"mean": (model, mean), "median": (model, median)}``.
    :rtype: Dict[str, Tuple[str, float]]
    """
    out = {}
    for stat, fn in (("mean", statistics.fmean), ("median", statistics.median)):
        values = {m: fn(r[f"cer_{m}"] for r in rows) for m in models}
        winner = min(models, key=lambda m: (values[m], models.index(m)))
        out[stat] = (winner, values[winner])
    return out


def micro_cer(rows: Sequence[dict], cer_key: str) -> float:
    """Pooled CER: total edits over total reference characters.

    :param rows: Per-document rows with ``gt_len`` and the CER column.
    :type rows: Sequence[dict]
    :param cer_key: CER column.
    :type cer_key: str
    :return: Sum of CER x reference length over the sum of reference lengths.
    :rtype: float
    """
    return sum(r[cer_key] * r["gt_len"] for r in rows) / sum(r["gt_len"] for r in rows)


def run_configuration(benchmark: str, models: Sequence[str],
                      units: Sequence[str] = UNITS) -> Tuple[dict, List[dict]]:
    """Slot oracle, ROVER and decomposition for one reader set on one benchmark.

    :param benchmark: A key of ``rover_consensus.SOURCES``.
    :type benchmark: str
    :param models: Reader stems (order fixes tie-breaks, as in ROVER).
    :type models: Sequence[str]
    :param units: Units to run.
    :type units: Sequence[str]
    :return: (summary dict, per-document rows).
    :rtype: Tuple[dict, List[dict]]
    :raises SystemExit: When no document has an output from every reader.
    """
    started = time.time()
    models = list(models)
    docs, skipped = load_benchmark_documents(benchmark, models)
    if not docs:
        raise SystemExit("no document has an output from every requested model")
    raw_gts = load_raw_gts(benchmark)
    buckets = SOURCES[benchmark].buckets
    rows, decomposition, fallbacks = [], {u: Counter() for u in units}, 0
    for doc in docs:
        lines, fallback = reference_lines(raw_gts[doc.doc_id], doc.gt_ink)
        fallbacks += int(fallback)
        cers = {m: score_output(doc.raws[m], doc.gt_ink, doc.gt_letters).cer for m in models}
        page_cer, page_model = oracle(cers, models)
        row = {"doc_id": doc.doc_id, "bucket": doc.bucket, "gt_len": len(doc.gt_ink),
               "gt_lines": len(lines),
               "gt_unreproducible": Levenshtein.distance(doc.gt_ink,
                                                         normalize_ink_hypothesis(doc.gt_ink))}
        row.update({f"cer_{m}": cers[m] for m in models})
        row.update({"page_oracle_cer": page_cer, "page_oracle_model": page_model})
        raws = [doc.raws[m] for m in models]
        for unit in units:
            result = analyse_page(raws, lines, unit)
            for name, text in (("rover", result.rover_text), ("slot_oracle", result.oracle_text)):
                score = score_output(text, doc.gt_ink, doc.gt_letters, with_f1=True)
                row[f"{system_key(name, unit)}_cer"] = score.cer
                row[f"{system_key(name, unit)}_f1"] = score.f1
                row[f"{system_key(name, unit)}_mode"] = score.mode
            row[f"backbone_{unit}"] = models[result.backbone]
            for key in ("slots", "gt_chars", "learnable_slots", "unreachable_slots",
                        "learnable_chars", "unreachable_chars"):
                row[f"{unit}_{key}"] = result.stats.get(key, 0)
            decomposition[unit].update(result.stats)
        rows.append(row)
    lone = {unit: split_lone_dissent(decomposition[unit], models) for unit in units}

    systems = {m: aggregate(rows, f"cer_{m}", None, buckets) for m in models}
    systems["page_oracle"] = aggregate(rows, "page_oracle_cer", None, buckets)
    for unit in units:
        for name in ("rover", "slot_oracle"):
            key = system_key(name, unit)
            systems[key] = aggregate(rows, f"{key}_cer", None, buckets)
            systems[key]["f1_median"] = statistics.median(r[f"{key}_f1"] for r in rows)
            systems[key]["micro_cer"] = micro_cer(rows, f"{key}_cer")
            systems[key]["modes"] = dict(Counter(r[f"{key}_mode"] for r in rows))
    summary = {
        "benchmark": benchmark, "models": models, "units": list(units),
        "gap": {u: DEFAULT_GAP[u] for u in units},
        "n_docs": len(docs), "skipped_missing_output": skipped,
        "gt_line_fallbacks": fallbacks,
        "gt_lines_median": statistics.median(r["gt_lines"] for r in rows),
        "gt_chars_total": sum(r["gt_len"] for r in rows),
        # edits normalize_ink_hypothesis would make to the references: characters no normalised
        # reading can contain (e.g. '?', '*', '/'), counted as unreachable
        "gt_unreproducible_edits": sum(r["gt_unreproducible"] for r in rows),
        "gt_unreproducible_pages": sum(r["gt_unreproducible"] > 0 for r in rows),
        "best_single": best_single(rows, models),
        "page_oracle_wins": dict(Counter(r["page_oracle_model"] for r in rows)),
        "systems": systems,
        "decomposition": {u: dict(c) for u, c in decomposition.items()},
        "lone_dissent_by_model": lone,
        "seconds": round(time.time() - started, 1),
    }
    return summary, rows


def split_lone_dissent(stats: Counter, models: Sequence[str]) -> Dict[str, Dict[str, int]]:
    """Move the per-reader lone-dissent tallies out of pooled statistics.

    :param stats: Pooled statistics of one unit; ``lone_<i>`` and
        ``lone_right_<i>`` keys are removed in place.
    :type stats: Counter
    :param models: Reader stems indexed by hypothesis index.
    :type models: Sequence[str]
    :return: ``{model: {"n": lone dissents, "right": of those equal to the GT}}``.
    :rtype: Dict[str, Dict[str, int]]
    """
    out = {}
    for i, model in enumerate(models):
        out[model] = {"n": stats.pop(f"lone_{i}", 0), "right": stats.pop(f"lone_right_{i}", 0)}
    return out


def headroom(summary: dict, unit: str) -> Dict[str, float]:
    """ROVER minus slot oracle at one unit, on the mean and on the median.

    :param summary: Output of :func:`run_configuration`.
    :type summary: dict
    :param unit: ``word`` or ``char``.
    :type unit: str
    :return: ``{"mean": ..., "median": ...}`` (positive = room for a voter).
    :rtype: Dict[str, float]
    """
    rover, best = summary["systems"][f"rover_{unit}"], summary["systems"][f"slot_oracle_{unit}"]
    return {stat: rover[stat] - best[stat] for stat in ("mean", "median")}


def decomposition_shares(stats: Dict[str, int]) -> Dict[str, float]:
    """Shares behind the learnable / unreachable split of ROVER's wrong slots.

    :param stats: Pooled statistics of one unit (see :func:`slot_statistics`).
    :type stats: Dict[str, int]
    :return: ``wrong_slots``; ``learnable_share`` / ``unreachable_share`` of
        the wrong slots; ``<class>_char_share`` of the GT characters;
        ``lost_tie_share`` / ``single_share`` of the learnable slots;
        ``near_miss_share``, ``gt_only_share`` and ``block_share`` of the
        unreachable GT characters; ``lone_precision`` / ``multi_precision``:
        share of losing candidates (one reader / several readers) equal to the GT.
    :rtype: Dict[str, float]
    """
    wrong = stats.get("learnable_slots", 0) + stats.get("unreachable_slots", 0)
    chars = stats.get("gt_chars", 0) or 1
    learnable = stats.get("learnable_slots", 0) or 1
    unreachable_chars = stats.get("unreachable_chars", 0) or 1
    out = {"wrong_slots": wrong,
           "learnable_share": stats.get("learnable_slots", 0) / (wrong or 1),
           "unreachable_share": stats.get("unreachable_slots", 0) / (wrong or 1),
           "lost_tie_share": stats.get("learnable_lost_tie", 0) / learnable,
           "single_share": stats.get("learnable_single", 0) / learnable,
           "near_miss_share": stats.get("near_miss_chars", 0) / unreachable_chars,
           "gt_only_share": stats.get("gt_only_chars", 0) / unreachable_chars,
           "block_share": stats.get("unreachable_block_chars", 0) / unreachable_chars}
    for kind in ("lone", "multi"):
        out[f"{kind}_precision"] = (stats.get(f"dissent_{kind}_right", 0)
                                    / (stats.get(f"dissent_{kind}", 0) or 1))
    out.update({f"{c}_char_share": stats.get(f"{c}_chars", 0) / chars for c in CLASSES})
    return out


def summary_table(summary: dict) -> str:
    """Plain-text table of one configuration.

    :param summary: Output of :func:`run_configuration`.
    :type summary: dict
    :return: Printable table.
    :rtype: str
    """
    systems = summary["systems"]
    best = summary["best_single"]
    lines = [f"{'system':28s} {'mean':>7s} {'median':>7s} {'F1 med':>7s}",
             f"{'best single by mean':28s} {best['mean'][1]:7.3f} {'':>7s}   ({best['mean'][0]})",
             f"{'best single by median':28s} {'':>7s} {best['median'][1]:7.3f}   ({best['median'][0]})"]
    for name in ["page_oracle"] + [system_key(n, u) for n in ("rover", "slot_oracle")
                                   for u in summary["units"]]:
        rec = systems[name]
        f1 = f"{rec['f1_median']:7.3f}" if "f1_median" in rec else f"{'':>7s}"
        lines.append(f"{name:28s} {rec['mean']:7.3f} {rec['median']:7.3f} {f1}")
    for unit in summary["units"]:
        room = headroom(summary, unit)
        shares = decomposition_shares(summary["decomposition"][unit])
        lines.append(f"\n{unit}: headroom mean {room['mean']:.3f} median {room['median']:.3f}; "
                     f"ROVER-wrong slots {shares['wrong_slots']}: learnable "
                     f"{shares['learnable_share']:.1%} / unreachable {shares['unreachable_share']:.1%}; "
                     f"GT chars right {shares['right_char_share']:.1%} learnable "
                     f"{shares['learnable_char_share']:.1%} unreachable "
                     f"{shares['unreachable_char_share']:.1%}; lone-dissent precision "
                     f"{shares['lone_precision']:.1%}; unreachable chars in blocks "
                     f"{shares['block_share']:.1%}")
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (default ``sys.argv[1:]``).
    :type argv: Optional[Sequence[str]]
    :return: Parsed arguments.
    :rtype: argparse.Namespace
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--models", nargs="+", required=True, help="reader file stems (ROVER order)")
    p.add_argument("--benchmark", choices=tuple(SOURCES), default="verified")
    p.add_argument("--units", nargs="+", choices=UNITS, default=list(UNITS))
    p.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    p.add_argument("--tag", help="output stem: <out-dir>/<tag>.json and <tag>_per_doc.csv "
                                 "(required unless --dump-blocks)")
    p.add_argument("--dump-blocks", type=Path, metavar="OUT.jsonl",
                   help="instead of the oracle run, write the char-level unreachable blocks")
    p.add_argument("--min-block-chars", type=int, default=MIN_BLOCK_CHARS,
                   help="shortest block --dump-blocks writes (GT characters)")
    args = p.parse_args(argv)
    if len(set(args.models)) != len(args.models):
        p.error("--models has duplicates")
    if args.tag is None and args.dump_blocks is None:
        p.error("--tag is required unless --dump-blocks is given")
    return args


def write_outputs(summary: dict, rows: Sequence[dict], out_dir: Path, tag: str) -> None:
    """Write the summary JSON and the per-document CSV of one configuration.

    :param summary: Output of :func:`run_configuration`.
    :type summary: dict
    :param rows: Per-document rows.
    :type rows: Sequence[dict]
    :param out_dir: Output folder (created if missing).
    :type out_dir: Path
    :param tag: File stem.
    :type tag: str
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    with open(out_dir / f"{tag}_per_doc.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                          for r in rows])


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point: run one reader set and print its summary.

    :param argv: Arguments (default ``sys.argv[1:]``).
    :type argv: Optional[Sequence[str]]
    """
    args = parse_args(argv)
    if args.dump_blocks is not None:
        blocks = collect_blocks(args.benchmark, args.models, args.min_block_chars)
        write_jsonl(args.dump_blocks, blocks)
        print(f"{args.benchmark}: {len(blocks)} unreachable blocks of >= {args.min_block_chars} GT "
              f"characters on {len({b['doc_id'] for b in blocks})} pages "
              f"({sum(b['unreachable'] for b in blocks)} unreachable characters) -> {args.dump_blocks}")
        return
    summary, rows = run_configuration(args.benchmark, args.models, args.units)
    summary["tag"] = args.tag
    write_outputs(summary, rows, args.out_dir, args.tag)
    print(f"[{args.tag}] {args.benchmark}: {summary['n_docs']} docs, {len(args.models)} readers, "
          f"{summary['seconds']}s (GT one-line fallbacks: {summary['gt_line_fallbacks']})\n")
    print(summary_table(summary))
    print(f"\nWrote {args.out_dir / args.tag}.json and {args.tag}_per_doc.csv")


if __name__ == "__main__":
    main()
