"""Novelty tiers and evaluation hold-outs, version 2, of the ja_ar_v1 build (``GATE_V2_SPEC.md`` section 7).

PRIVATE / INTERNAL: names benchmark sets, registries and internal paths. No edition text lives in this module.

Two judgements are made for every target side (and, later, for every gate-v2 pair), each with its witnesses.

**1. Novelty against the seen registry** (``logs/next_round/unseen_v1/seen/*.jsonl``: one row per training row of every
prior fine-tune, keyed by ``registry_lib``; ``genizah_novel_v1`` was registered on 2026-11-12). Every registry row gets a
class (:func:`row_class`, the ledger's ``subtract.witness_name`` rule with ``genizah_novel_v1`` added to the definite
datasets):

* ``trained``: a row of a materialised mixture the trainer consumed or of a fully drawn direct dataset
  (:data:`DEFINITE_DATASETS`; a ``partial`` mixture row only with ``consumed=1``);
* ``pool``: a source pool or an undrawn part of a mixture (whether it was drawn is not recorded): uncertain evidence;
* ``eval_only``: a validation row of a definite dataset (a split evaluated during a run). Never novelty evidence; an
  evaluation hold-out (family ``registry_val``, ACCEPTANCE.md rule 10);
* ``pool_val``: a validation row of a pool dataset. Such a split was never evaluated as such (a pool row that a
  mixture drew into its val split is registered again as that mixture's ``eval_only`` row), so it is neither evidence
  nor a hold-out; it is recorded as a flag (``--pool-val-holdout`` turns these rows into ``eval_only``);
* ``released_val``: the val rows of ``pgp_editions_v1`` whose PGP id is in ``held_out_edition_pages_v0``, the reserve
  that decision 3 released (recorded only).

Rows of never-trained datasets (:data:`NEVER_TRAINED_DATASETS`) or with ``trained_status=not_trained`` are dropped.

Evidence axes of a subject (:class:`Subject`):

* page (this side's own photograph or its text): the photograph's sha256, URL, KTIV ``FL`` id, or a pHash within
  :data:`PHASH_MAX_BITS` bits of a registry image *confirmed* by :class:`PhashConfirmer`; the side's text against one
  registry page under the long-run text rule (:class:`TextRule`). A registry page is every row of one image (or one
  text-only row) of one class, unioned.

  pHash confirmation: Genizah photographs share mounts, colour bars and rulers, so different fragments often lie within
  6 bits of each other (measured on W1 photographs against the registry: every sampled match at 4-6 bits was another
  document, about half at 2 bits; at 0 bits all were the same photograph). A match within :data:`PHASH_MAX_BITS` of
  a different file counts only when the 256-bit pHash of the two files differs by at most
  :data:`PHASH256_MAX_BITS` (re-encodes and resizes of one photograph measured at most 6). When a file is missing, a
  0-bit match counts and a 1-6-bit match does not. Unconfirmed matches are kept in the record (class
  ``phash_rejected`` / ``phash_unverified``) but are never evidence.
* document, definite keys: PGP id, ES canonical id (``canonical_id``, ``canonical_ids``, ``cids_es``), the
  institution-qualified ``shelf_keys_clean``, the KTIV system number, the ledger ``document_group`` (novel_v1 rows
  record theirs); and the other members of the subject's document group (another side, section or edition of the same
  document, :func:`group_evidence`).
* document, weak keys (uncertain evidence only): the legacy institution-free ``shelf_key`` (it collides across
  institutions), loose ids (``pgpids_on_shelf``, ``alt_canonical_ids``), parent / child shelf keys, and a trained
  verified join partner (the ledger's decision d: never more than ``uncertain_history``).

Tier of a side (:func:`tier_of`), first match wins:

1. ``previously_trained``: a ``trained`` page witness (the side's own photograph, or its text);
2. ``uncertain_history``: a ``pool`` page witness;
3. ``novel_page_level``: the document was trained (definite ``trained`` document witness or a trained group member)
   but this page's image and text never were; reported apart, never the headline (ACCEPTANCE.md rule 2);
4. ``uncertain_history``: only uncertain document evidence;
5. ``novel_strict``: no evidence at all (the headline).

The side's own photograph is the one novel_v1's stage-2 placement accepted, or the single photograph of a one-page
candidate (for a whole ledger candidate: its own image keys). Gate v2 re-places sides, so every record also carries
``pair_tiers``: the tier of each (side, photograph) pair of its candidate (:func:`pair_tiers`). A trained or pool
photograph of the candidate that is not the side's own is recorded as a flag (``photo_trained_unplaced`` /
``photo_pool_unplaced``), never as this side's page evidence.

**2. Evaluation hold-outs** (:class:`EvalIndex`), against: the registered benchmarks that are still evaluation sets
(``arabic_script_benchmark_v0``, ``sealed_v1``), the frozen PGP-131 list, religious-140, the named exclusions, the APD
records of Arabic-v0 documents (``apd_exclusions.jsonl`` reason ``benchmark_holdout``), the KTIV manuscript
``FR 9-002.2`` (a re-photograph of a sealed_v1 literary manuscript), the registry's eval-only rows, and every new eval
set registered under ``W2/eval/registered/*.json``. Released, recorded under ``released`` but never holding out
(decisions 3-4): the three PGP analysis reserves ``two_edition_documents_v0``, ``held_out_edition_pages_v0``,
``offbench_sample_v0`` (and the ``two_edition_footnotes`` GT text, not loaded), the text LLM's hold-out set (not loaded
at all), and the APD exclusions whose only reason is ``pgp_arabic_editions_v2`` (PGP's edition wins, not an
evaluation). A page of a released set that a new eval set draws is held out through that eval set's own file.

* identity (any member of the document group matching holds out every side of the group): ES id (as recorded, as the
  merged-corpus index resolves it, and the candidate's verified join partners), PGP id, KTIV system number, facade
  loose key, KTIV ``FL`` page id, row/stem names, image sha256, pHash within :data:`PHASH_MAX_BITS`; registry
  eval-only rows by their definite document keys and photographs;
* text (per side): the long-run rule against one evaluation document's ground truth or one eval-only registry page.

The short-shingle count rule of novel_v1 (two shared 25-letter positions, with its boilerplate amendment) is gone.

**Long-run text rule** (:class:`TextRule`). A side's text is reduced to letters and cut into letter 5-grams in reading
order (against the registry: the ``registry_lib`` letter class without folding and xxh64 of each 5-gram, because the
registry stores only those hashes; against evaluation ground truth: ``textnorm.skeleton``, finals and Arabic letter
variants folded). Against one page (registry) or one document (evaluation set), a 5-gram position is shared when its
5-gram occurs in the page. The rule holds the side when

* the longest run of consecutive shared positions spans at least :data:`LONG_RUN_LETTERS` (100) letters, or
* at least :data:`SHARE_THRESHOLD` (30 %) of the side's :data:`SHINGLE_LETTERS`-letter (25) shingles are shared with
  the page (``share_basis="shingles"``, the default), and the side has at least :data:`MIN_SIDE_GRAMS` distinct
  5-grams. A 25-letter shingle counts as shared when the 21 consecutive 5-grams it spans all are (the registry keeps
  5-gram hashes only), so a shared run of ``L`` letters contributes ``L - 24`` shingles: the unit of the share guard of
  the 2026-11-04 boilerplate amendment, now against one document instead of the union of a set.

Scattered short matches (formulaic openings, deed clauses, a basmala, common letter 5-grams) barely count towards the
share: a 44-letter epistolary formula in a 120-letter address side is 20 % of its shingles. ``share_basis="grams"`` is
the literal reading (30 % of the side's distinct 5-grams anywhere in the page); it is kept for comparison only:
measured on the W1 sides against the evaluation ground truth it holds 172 sides (Arabic script 9.6 %), mostly on
epistolary and petition formulae, against 18 for the default (``logs/next_round/ja_ar_v1/build/NOVELTY_HOLDOUT_V2.md``).

**New evaluation sets** (``W2/eval/registered/<name>.json``; the directory may be empty or missing)::

    {"eval_set": "ja_dev_v1",                 # required; witnesses read set=<eval_set>, item=<item_id>
     "role": "dev" | "sealed",                # informational
     "built": "YYYY-MM-DD",
     "items": [{"item_id": "...",             # required, unique in the file
                "canonical_ids": [...], "pgpids": [...], "shelfmarks": [...],   # document identity (any may be empty)
                "ktiv_sys_nums": [...], "ktiv_fl": [...],
                "image_sha256": [...], "phash": [...],                          # the item's photographs
                "row_ids": [...], "document_group": "..."}],                    # ledger / build names, optional
     "gt_jsonl": "/Volumes/.../gt.jsonl"}     # optional: {"item_id": ..., "text": ...} per line, for the text rule

Every listed key is a hold-out key; shelf marks are split into parts, keyed by the facade and resolved through the
merged-corpus index like the registered benchmarks. Edition text never goes into this file (only into ``gt_jsonl`` on
the NAS).

**Usage** (from the repo root)::

    PYTHONPATH=. .venv/bin/python -m src.finetuning.qwen_hebrew.ja_ar_build.holdout_v2 \\
        [--w1 W1] [--w2 W2] [--only w1|uncertain|both] [--no-index] [--limit N]

writes ``W2/novelty/verdicts_w1.jsonl`` (one record per W1 target side), ``W2/novelty/verdicts_uncertain.jsonl`` (one
record per ledger ``uncertain_history`` candidate, re-run with these rules) and ``W2/novelty/summary.json``.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import orjson
import xxhash

from src.datasets import shelfmarks as shelf
from src.datasets.evaluations import benchmark_registry as br
from src.finetuning.qwen_hebrew import build_legacy_store_rows as lsr
from src.finetuning.qwen_hebrew.ja_ar_build import textnorm
from src.finetuning.qwen_hebrew.novel_build import holdout as nh
from src.finetuning.qwen_hebrew.novel_build.images import image_keys
from src.finetuning.qwen_hebrew.novel_build.schema import Candidate, from_dict
from src.finetuning.qwen_hebrew.novel_build.targets import measure_script

# ===================================================================================================================
# Paths and parameters
# ===================================================================================================================

REPO = Path(__file__).resolve().parents[4]
NAS = Path("/Volumes/home/studio_offload/datasets")
W1 = NAS / "genizah_novel_v1_work"
W2 = NAS / "genizah_ja_ar_v1_work"
LEDGER_DIR = REPO / "logs" / "next_round" / "unseen_v1"
SEEN_DIR = LEDGER_DIR / "seen"
REKEY_SEEN_DIR = LEDGER_DIR / "rekey" / "seen"
LEDGER = LEDGER_DIR / "unseen_pool.jsonl"
CANDIDATES_DIR = LEDGER_DIR / "candidates"
JOIN_PARTNERS = LEDGER_DIR / "rekey" / "join_partners.jsonl"
NAMED_EXCLUSIONS = REPO / "logs" / "next_round" / "novel_v1" / "named_exclusions.json"
APD_EXCLUSIONS = REPO / "logs" / "next_round" / "arabic_v25a" / "apd_exclusions.jsonl"
EVAL_DIR = lsr.EVAL_DIR
ARABIC_V0_DIR = NAS / "arabic_script_benchmark_v0"

PHASH_MAX_BITS = 6
#: Largest 256-bit pHash distance (16 x 16 DCT grid) that confirms a 64-bit pHash match as the same photograph.
PHASH256_MAX_BITS = 10
SHINGLE_N = 5
LONG_RUN_LETTERS = 100
SHARE_THRESHOLD = 0.30
SHINGLE_LETTERS = 25
MIN_SIDE_GRAMS = 40
SHARE_BASES = ("shingles", "grams")
#: Postings of a 5-gram are kept for at most this many pages; a 5-gram that reaches the cap is "common" and is not
#: used to FIND candidate pages (it is still counted once a page is found).
MAX_POSTING = 4000
#: Registry pages per side checked exactly (the pages sharing the most informative 5-grams).
TOP_PAGES = 64
#: Witnesses kept per list in a record (counts are kept in full).
MAX_WITNESSES = 6
#: Row ids kept per (key, class, dataset) in the indexes.
MAX_REFS = 2

TRAINED, POOL, EVAL_ONLY, POOL_VAL, RELEASED_VAL = "trained", "pool", "eval_only", "pool_val", "released_val"
PHASH_REJECTED, PHASH_UNVERIFIED = "phash_rejected", "phash_unverified"
PREVIOUSLY_TRAINED, UNCERTAIN, PAGE_LEVEL, STRICT = ("previously_trained", "uncertain_history", "novel_page_level",
                                                     "novel_strict")
TIERS = (PREVIOUSLY_TRAINED, UNCERTAIN, PAGE_LEVEL, STRICT)

#: Materialised mixtures the trainer consumed and fully drawn direct datasets (``subtract.DEFINITE_DATASETS`` plus
#: novel_v1, whose 2,608 train rows were consumed by the Lambda and ROCm runs).
DEFINITE_DATASETS = frozenset({
    "genizah_v22_pilot", "genizah_v23a_arabic", "genizah_v24a_arabic", "genizah_v25a_arabic", "genizah_v22_full",
    "genizah_clean_v2", "genizah_ktiv_v1", "genizah_novel_v1",
})
#: Registered but never trained (``subtract.NEVER_TRAINED_DATASETS``).
NEVER_TRAINED_DATASETS = frozenset({"genizah_v23h_hebonly", "genizah_v23q_vqa", "pgp_vqa_parse_v1"})
#: Registries released to training by the user's decisions 3-4 (2026-11-12): recorded, never holding out.
RELEASED_REGISTRIES = frozenset({"two_edition_documents_v0", "held_out_edition_pages_v0", "offbench_sample_v0"})
#: The registry dataset whose val pages are the released ``held_out_edition_pages_v0`` reserve.
RELEASED_VAL_DATASET = "pgp_editions_v1"
#: KTIV manuscript that re-photographs a sealed_v1 literary manuscript (survey ktiv_new, s09/s10).
FR_9_002_2 = {"sys_num": "990001996710205171",
              "shelf_mark": "University of Toronto Library, Toronto, ON, Canada Ms. FR 9-002.2"}
#: APD exclusion reasons that are evaluation hold-outs (the rest is "PGP's edition wins", D15).
APD_EVAL_REASONS = frozenset({"benchmark_holdout"})

_VAL_SPLIT = re.compile(r"(^|_)(val|dev|test|eval)", re.I)
_STATUS = re.compile(r"trained_status=([a-z_]+)")
_FL = re.compile(r"(?<![A-Za-z0-9])FL(\d{6,})(?!\d)")
_HEX16 = re.compile(r"[0-9a-fA-F]{16}")
_LOOSE = (("loose_pgpid", re.compile(r"pgpids?_on_shelf=([0-9,; ]+)")),
          ("loose_canonical_id", re.compile(r"alt_canonical_ids?=([A-Za-z0-9_,; ]+)")))
#: The ``registry_lib`` letter class and marks, written as code points (the registry's text shingles use them).
REGISTRY_LETTERS = re.compile("[א-תؠ-يٱ-ۓ]")
REGISTRY_MARKS = re.compile("[֑-ׇؐ-ًؚ-ٰٟۖ-ۭ]")


# ===================================================================================================================
# Small helpers
# ===================================================================================================================

def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Records of a JSON-lines file (orjson: unsigned 64-bit shingle hashes survive).

    :param path: File.
    :type path: Path
    :return: Records.
    :rtype: Iterator[Dict[str, Any]]
    """
    with open(path, "rb") as fh:
        for line in fh:
            if line.strip():
                yield orjson.loads(line)


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Write records as JSON lines (parent directories created).

    :param path: Output file.
    :type path: Path
    :param records: Records.
    :type records: Iterable[Mapping[str, Any]]
    :return: Records written.
    :rtype: int
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "wb") as fh:
        for rec in records:
            fh.write(orjson.dumps(rec, option=orjson.OPT_NON_STR_KEYS) + b"\n")
            n += 1
    return n


def listify(value: Any) -> List[str]:
    """A value as a list of non-empty strings (None, a scalar or a list).

    :param value: Anything.
    :type value: Any
    :return: Strings.
    :rtype: List[str]
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def notes_text(row: Mapping[str, Any]) -> str:
    """A registry row's notes as one string (they are a string or a list).

    :param row: Registry row.
    :type row: Mapping[str, Any]
    :return: The notes, list items joined by ``;``.
    :rtype: str
    """
    notes = row.get("notes")
    if isinstance(notes, (list, tuple)):
        return ";".join(str(x) for x in notes)
    return str(notes or "")


def note_value(notes: str, key: str) -> Optional[str]:
    """The value of a ``key=value`` item of ``;``-separated notes.

    :param notes: Notes string.
    :type notes: str
    :param key: Item name.
    :type key: str
    :return: The value, or None.
    :rtype: Optional[str]
    """
    for item in notes.split(";"):
        name, sep, value = item.strip().partition("=")
        if sep and name == key and value.strip():
            return value.strip()
    return None


def fl_ids(*values: Any) -> List[str]:
    """KTIV page ids (``FL<digits>``) found in strings.

    :param values: Strings (or None).
    :type values: Any
    :return: Distinct ``FL<digits>`` ids in order of appearance.
    :rtype: List[str]
    """
    out: List[str] = []
    for value in values:
        for m in _FL.finditer(str(value or "")):
            fl = "FL" + m.group(1)
            if fl not in out:
                out.append(fl)
    return out


def cap(witnesses: Sequence[Mapping[str, Any]], n: int = MAX_WITNESSES) -> List[Mapping[str, Any]]:
    """The first ``n`` witnesses (records stay small; full counts are kept beside them).

    :param witnesses: Witnesses.
    :type witnesses: Sequence[Mapping[str, Any]]
    :param n: Kept at most.
    :type n: int
    :return: The kept witnesses.
    :rtype: List[Mapping[str, Any]]
    """
    return list(witnesses[:n])


# ===================================================================================================================
# Text: letters, 5-gram positions and the long-run rule
# ===================================================================================================================

def registry_letters(text: str) -> str:
    """Letters of a text as ``registry_lib.letters_only`` keeps them (marks dropped, no folding).

    :param text: Any text.
    :type text: str
    :return: Hebrew and Arabic letters in order.
    :rtype: str
    """
    return "".join(REGISTRY_LETTERS.findall(REGISTRY_MARKS.sub("", text or "")))


def gram_positions(letters: str, n: int = SHINGLE_N) -> np.ndarray:
    """xxh64 hash of every letter n-gram of a letter string, in reading order (duplicates kept).

    :param letters: Letters only.
    :type letters: str
    :param n: n-gram length.
    :type n: int
    :return: ``uint64`` array of ``len(letters) - n + 1`` hashes (empty when shorter than ``n``).
    :rtype: np.ndarray
    """
    count = max(0, len(letters) - n + 1)
    return np.fromiter((xxhash.xxh64_intdigest(letters[i:i + n]) for i in range(count)), dtype=np.uint64,
                       count=count)


def registry_positions(text: str, n: int = SHINGLE_N) -> np.ndarray:
    """Positions of a text hashed like the registry's ``text_shingles`` (``registry_lib.text_shingles``).

    :param text: Any text.
    :type text: str
    :param n: n-gram length.
    :type n: int
    :return: ``uint64`` hashes in reading order.
    :rtype: np.ndarray
    """
    return gram_positions(registry_letters(text), n)


def skeleton_positions(text: str, n: int = SHINGLE_N) -> np.ndarray:
    """Positions of a text over its ``textnorm.skeleton`` (finals and Arabic letter variants folded).

    :param text: Any text.
    :type text: str
    :param n: n-gram length.
    :type n: int
    :return: ``uint64`` hashes in reading order.
    :rtype: np.ndarray
    """
    return gram_positions(textnorm.skeleton(text or ""), n)


def streaks(present: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Starts and lengths of the runs of True in a boolean array.

    :param present: Boolean array.
    :type present: np.ndarray
    :return: ``(starts, lengths)``, both int arrays (empty when nothing is True).
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    if not len(present):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    edges = np.diff(np.concatenate(([0], present.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return starts, ends - starts


def contains(sorted_unique: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Membership of each value in a sorted array of distinct values.

    :param sorted_unique: Sorted distinct values.
    :type sorted_unique: np.ndarray
    :param values: Values to test.
    :type values: np.ndarray
    :return: Boolean array, one per value.
    :rtype: np.ndarray
    """
    if not len(sorted_unique) or not len(values):
        return np.zeros(len(values), dtype=bool)
    idx = np.minimum(np.searchsorted(sorted_unique, values), len(sorted_unique) - 1)
    return sorted_unique[idx] == values


@dataclass(frozen=True)
class Overlap:
    """How much of a side's text one page (or ground-truth document) holds.

    :ivar run_letters: Letters spanned by the longest run of consecutive shared 5-gram positions (0 when none).
    :ivar share: Share of the side's distinct 5-grams found in the page (the literal "share of 5-grams").
    :ivar shingle_share: Share of the side's ``TextRule.shingle_letters``-letter shingle positions whose 5-grams are
        all shared (a shared run of ``L`` letters holds ``L - shingle_letters + 1`` of them).
    :ivar shared: Distinct shared 5-grams.
    :ivar grams: Distinct 5-grams of the side.
    """

    run_letters: int
    share: float
    shingle_share: float
    shared: int
    grams: int


@dataclass(frozen=True)
class TextRule:
    """The long-run text rule (module docstring).

    :ivar long_run_letters: A shared run of this many letters holds the side.
    :ivar share_threshold: This share of the side's shingles shared with one page holds the side.
    :ivar share_basis: ``shingles`` (the share of the side's ``shingle_letters``-letter shingles) or ``grams`` (the
        share of its distinct 5-grams anywhere in the page; for comparison).
    :ivar shingle_letters: Shingle length of the ``shingles`` share.
    :ivar min_grams: Distinct 5-grams a side needs before the share clause applies.
    :ivar n: 5-gram length.
    """

    long_run_letters: int = LONG_RUN_LETTERS
    share_threshold: float = SHARE_THRESHOLD
    share_basis: str = "shingles"
    shingle_letters: int = SHINGLE_LETTERS
    min_grams: int = MIN_SIDE_GRAMS
    n: int = SHINGLE_N

    def __post_init__(self) -> None:
        """Validate the parameters.

        :raises ValueError: On an unknown share basis or thresholds that cannot fire.
        """
        if self.share_basis not in SHARE_BASES:
            raise ValueError(f"share_basis {self.share_basis!r}: expected one of {SHARE_BASES}")
        if self.long_run_letters < self.n or self.shingle_letters < self.n:
            raise ValueError("run lengths shorter than one 5-gram")
        if not 0.0 < self.share_threshold <= 1.0:
            raise ValueError(f"share_threshold {self.share_threshold}: expected a share in (0, 1]")

    def reason(self, ov: Overlap) -> Optional[str]:
        """Which clause holds the side, if any.

        :param ov: The side's overlap with one page.
        :type ov: Overlap
        :return: ``text_run``, ``text_share`` or None.
        :rtype: Optional[str]
        """
        if ov.run_letters >= self.long_run_letters:
            return "text_run"
        if ov.grams >= self.min_grams:
            share = ov.shingle_share if self.share_basis == "shingles" else ov.share
            if share >= self.share_threshold:
                return "text_share"
        return None


def overlap(positions: np.ndarray, page: np.ndarray, rule: TextRule) -> Overlap:
    """Overlap of a side's 5-gram positions with one page's 5-gram set.

    :param positions: The side's hashes in reading order.
    :type positions: np.ndarray
    :param page: The page's distinct hashes, sorted.
    :type page: np.ndarray
    :param rule: Text rule (n and the shingle length).
    :type rule: TextRule
    :return: The overlap.
    :rtype: Overlap
    """
    if not len(positions):
        return Overlap(0, 0.0, 0.0, 0, 0)
    present = contains(page, positions)
    _, lengths = streaks(present)
    run = int(lengths.max()) + rule.n - 1 if len(lengths) else 0
    span = rule.shingle_letters - rule.n          # extra 5-gram positions one shingle spans
    shingles = len(positions) - span
    held = float(np.maximum(lengths - span, 0).sum())
    grams = int(np.unique(positions).size)
    shared = int(np.unique(positions[present]).size)
    return Overlap(run, shared / grams if grams else 0.0, held / shingles if shingles > 0 else 0.0, shared, grams)


def overlap_summary(ov: Overlap) -> Dict[str, Any]:
    """The figures of an overlap a record keeps.

    :param ov: Overlap.
    :type ov: Overlap
    :return: ``{"run_letters", "share", "shingle_share"}``.
    :rtype: Dict[str, Any]
    """
    return {"run_letters": ov.run_letters, "share": round(ov.share, 4), "shingle_share": round(ov.shingle_share, 4)}


# ===================================================================================================================
# Indexes: keys, perceptual hashes, page 5-grams
# ===================================================================================================================

class KeyIndex:
    """Identity keys of registry rows: ``(kind, value) -> {class: {dataset: [rows, [row ids]]}}``."""

    def __init__(self, max_refs: int = MAX_REFS) -> None:
        """Create an empty index.

        :param max_refs: Row ids kept per (key, class, dataset).
        :type max_refs: int
        """
        self.map: Dict[Tuple[str, str], Dict[str, Dict[str, List[Any]]]] = {}
        self.max_refs = max_refs

    def add(self, kind: str, value: Any, cls: str, dataset: str, row_id: str) -> None:
        """Record that a row carries a key.

        :param kind: Key kind (``pgpid``, ``image_sha256``, ...).
        :type kind: str
        :param value: Key value (empty values are ignored).
        :type value: Any
        :param cls: Row class.
        :type cls: str
        :param dataset: Registry dataset.
        :type dataset: str
        :param row_id: Registry row id.
        :type row_id: str
        """
        if value is None or not str(value).strip():
            return
        by_cls = self.map.setdefault((kind, str(value).strip()), {})
        by_ds = by_cls.setdefault(cls, {})
        entry = by_ds.get(dataset)
        if entry is None:
            by_ds[dataset] = [1, [row_id]]
            return
        entry[0] += 1
        if len(entry[1]) < self.max_refs:
            entry[1].append(row_id)

    def has(self, kind: str, value: Any) -> bool:
        """Whether any row carries a key.

        :param kind: Key kind.
        :type kind: str
        :param value: Key value.
        :type value: Any
        :return: True when present.
        :rtype: bool
        """
        return value is not None and (kind, str(value).strip()) in self.map

    def lookup(self, kind: str, value: Any, axis: str,
               extra: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
        """Witnesses of one key: one per (class, dataset).

        :param kind: Key kind.
        :type kind: str
        :param value: Key value.
        :type value: Any
        :param axis: ``image`` or ``document`` (copied into the witness).
        :type axis: str
        :param extra: Extra fields for every witness.
        :type extra: Optional[Mapping[str, Any]]
        :return: ``{"axis", "key", "value", "class", "dataset", "row_ids", "n_rows", ...}`` dicts.
        :rtype: List[Dict[str, Any]]
        """
        if value is None or not str(value).strip():
            return []
        by_cls = self.map.get((kind, str(value).strip()))
        if not by_cls:
            return []
        out = []
        for cls in sorted(by_cls):
            for dataset in sorted(by_cls[cls]):
                n_rows, row_ids = by_cls[cls][dataset]
                w = {"axis": axis, "key": kind, "value": str(value), "class": cls, "dataset": dataset,
                     "row_ids": list(row_ids), "n_rows": n_rows}
                if extra:
                    w.update(extra)
                out.append(w)
        return out


@dataclass
class PhashHit:
    """One stored perceptual hash within reach of a query.

    :ivar distance: 64-bit Hamming distance to the query.
    :ivar owners: ``{(class, owner): [count, refs]}`` of the rows (or items) carrying the hash.
    :ivar files: ``{sha256: path}`` of the images carrying it (at most a few; path may be None).
    """

    distance: int
    owners: Dict[Tuple[str, str], List[Any]]
    files: Dict[str, Optional[str]]


class PhashIndex:
    """Perceptual hashes with their owners and image files, searchable by Hamming distance."""

    def __init__(self, max_refs: int = MAX_REFS, max_files: int = 3) -> None:
        """Create an empty index.

        :param max_refs: References kept per (hash, class, owner).
        :type max_refs: int
        :param max_files: Image files kept per hash (for confirmation).
        :type max_files: int
        """
        self._owners: Dict[int, Dict[Tuple[str, str], List[Any]]] = {}
        self._files: Dict[int, Dict[str, Optional[str]]] = {}
        self._order: List[int] = []
        self.values = np.zeros(0, dtype=np.uint64)
        self.max_refs = max_refs
        self.max_files = max_files

    def add(self, phash: Optional[str], cls: str, owner: str, ref: str, sha256: Optional[str] = None,
            path: Optional[str] = None) -> None:
        """Record one hash.

        :param phash: 16 hex characters (ignored when empty or malformed).
        :type phash: Optional[str]
        :param cls: Owner class (registry row class, or an evaluation set index).
        :type cls: str
        :param owner: Dataset (registry) or family (evaluation).
        :type owner: str
        :param ref: Row id or evaluation item.
        :type ref: str
        :param sha256: Digest of the image file carrying the hash, if known.
        :type sha256: Optional[str]
        :param path: Path of that file, if known.
        :type path: Optional[str]
        """
        if not phash or not _HEX16.fullmatch(str(phash)):
            return
        value = int(str(phash), 16)
        by_owner = self._owners.setdefault(value, {})
        entry = by_owner.get((cls, owner))
        if entry is None:
            by_owner[(cls, owner)] = [1, [ref]]
        else:
            entry[0] += 1
            if len(entry[1]) < self.max_refs:
                entry[1].append(ref)
        files = self._files.setdefault(value, {})
        key = str(sha256 or path or "")
        if key and (key not in files or (files[key] is None and path)) and (key in files or len(files) < self.max_files):
            files[key] = path

    def finish(self) -> None:
        """Freeze the hash array."""
        self._order = sorted(self._owners)
        self.values = np.asarray(self._order, dtype=np.uint64)

    def within(self, phash: Optional[str], max_bits: int = PHASH_MAX_BITS) -> List[PhashHit]:
        """Hashes within ``max_bits`` of a query, nearest first.

        :param phash: Query hash (16 hex characters) or None.
        :type phash: Optional[str]
        :param max_bits: Largest Hamming distance.
        :type max_bits: int
        :return: Hits.
        :rtype: List[PhashHit]
        """
        if not phash or not len(self.values) or not _HEX16.fullmatch(str(phash)):
            return []
        dist = np.bitwise_count(self.values ^ np.uint64(int(str(phash), 16)))
        hits = np.flatnonzero(dist <= max_bits)
        hits = hits[np.argsort(dist[hits], kind="stable")]
        return [PhashHit(int(dist[i]), self._owners[self._order[int(i)]], self._files.get(self._order[int(i)], {}))
                for i in hits]


def phash256(path: str) -> str:
    """256-bit DCT perceptual hash of an image file (the ledger's algorithm on a 16 x 16 grid of a 64 x 64 resize).

    :param path: Image file.
    :type path: str
    :return: 64 hex characters.
    :rtype: str
    :raises OSError: When the file cannot be read or decoded.
    """
    from PIL import Image
    from scipy.fft import dct

    with Image.open(path) as img:
        grid = np.asarray(img.convert("L").resize((64, 64), Image.Resampling.LANCZOS), dtype=np.float64)
    low = dct(dct(grid, axis=0, norm="ortho"), axis=1, norm="ortho")[:16, :16]
    bits = (low > np.median(low)).flatten()
    return "".join(f"{int(''.join('1' if b else '0' for b in bits[i:i + 4]), 2):x}" for i in range(0, 256, 4))


class PhashConfirmer:
    """Confirms a 64-bit pHash match by the 256-bit pHash of the two files (cached by path, size and mtime)."""

    def __init__(self, cache_path: Optional[Path] = None, max_bits: int = PHASH256_MAX_BITS) -> None:
        """Create a confirmer.

        :param cache_path: JSON cache of 256-bit hashes (never under ``raw_data``), or None.
        :type cache_path: Optional[Path]
        :param max_bits: Largest 256-bit distance of the same photograph.
        :type max_bits: int
        :raises ValueError: When the cache path lies under ``raw_data``.
        """
        if cache_path is not None and "raw_data" in Path(cache_path).resolve().parts:
            raise ValueError(f"the pHash cache must not live under raw_data: {cache_path}")
        self.cache_path = cache_path
        self.max_bits = max_bits
        self.cache: Dict[str, List[Any]] = {}
        self.changed = False
        self.stats: Counter = Counter()
        if cache_path is not None and cache_path.is_file():
            try:
                self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.cache = {}

    def hash_of(self, path: Optional[str]) -> Optional[str]:
        """256-bit pHash of a file, or None when it is missing or does not decode.

        :param path: Image file.
        :type path: Optional[str]
        :return: 64 hex characters or None.
        :rtype: Optional[str]
        """
        if not path:
            return None
        try:
            st = Path(path).stat()
            hit = self.cache.get(path)
            if hit and hit[0] == st.st_size and hit[1] == st.st_mtime_ns:
                return hit[2]
            value = phash256(path)
        except (OSError, ValueError):
            return None
        self.cache[path] = [st.st_size, st.st_mtime_ns, value]
        self.changed = True
        return value

    def distance(self, path_a: Optional[str], paths_b: Iterable[Optional[str]]) -> Optional[int]:
        """Smallest 256-bit distance between one file and any of several files.

        :param path_a: The query photograph's file.
        :type path_a: Optional[str]
        :param paths_b: Candidate files.
        :type paths_b: Iterable[Optional[str]]
        :return: The distance, or None when no pair of files could be hashed.
        :rtype: Optional[int]
        """
        a = self.hash_of(path_a)
        if a is None:
            return None
        best: Optional[int] = None
        for path in paths_b:
            b = self.hash_of(path)
            if b is not None:
                d = bin(int(a, 16) ^ int(b, 16)).count("1")
                best = d if best is None else min(best, d)
        return best

    def judge(self, distance64: int, path_a: Optional[str], paths_b: Iterable[Optional[str]]) -> Tuple[str, Optional[int]]:
        """Verdict on a 64-bit pHash match.

        :param distance64: The 64-bit distance.
        :type distance64: int
        :param path_a: The query photograph's file.
        :type path_a: Optional[str]
        :param paths_b: The stored image's files.
        :type paths_b: Iterable[Optional[str]]
        :return: ``("confirmed" | "rejected" | "unverified", 256-bit distance or None)``; a 0-bit match whose files
            cannot be compared is confirmed.
        :rtype: Tuple[str, Optional[int]]
        """
        d256 = self.distance(path_a, paths_b)
        if d256 is None:
            verdict = "confirmed" if distance64 == 0 else "unverified"
        else:
            verdict = "confirmed" if d256 <= self.max_bits else "rejected"
        self.stats[verdict] += 1
        return verdict, d256

    def save(self) -> None:
        """Write the cache when it changed."""
        if self.changed and self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")
            self.changed = False


class ShingleIndex:
    """Distinct 5-gram hashes per page, with postings to find the pages that share a side's 5-grams.

    A page is keyed by ``(class, page key)``; the rows added under one key are unioned. With a ``vocabulary`` (the
    sorted distinct hashes of every text that will be judged) only those hashes are stored: no other hash can be shared
    with a judged text, so overlaps stay exact and the index stays small.
    """

    def __init__(self, vocabulary: Optional[np.ndarray] = None, max_refs: int = MAX_REFS,
                 max_posting: int = MAX_POSTING) -> None:
        """Create an empty index.

        :param vocabulary: Sorted distinct ``uint64`` hashes to keep, or None to keep all.
        :type vocabulary: Optional[np.ndarray]
        :param max_refs: References kept per (page, dataset).
        :type max_refs: int
        :param max_posting: Posting cap per 5-gram (:data:`MAX_POSTING`).
        :type max_posting: int
        """
        self.vocabulary = vocabulary
        self.max_refs = max_refs
        self.max_posting = max_posting
        self.page_of: Dict[Tuple[str, str], int] = {}
        self.page_keys: List[Tuple[str, str]] = []
        self.page_refs: List[Dict[str, List[Any]]] = []
        self._parts: List[List[np.ndarray]] = []
        self.ptr = np.zeros(1, dtype=np.int64)
        self.hashes = np.zeros(0, dtype=np.uint64)
        self.post_h = np.zeros(0, dtype=np.uint64)
        self.post_p = np.zeros(0, dtype=np.int32)

    def add(self, page: Tuple[str, str], hashes: np.ndarray, dataset: str, ref: str) -> None:
        """Add a row's 5-gram hashes to a page.

        :param page: ``(class, page key)``.
        :type page: Tuple[str, str]
        :param hashes: The row's hashes (any order, duplicates allowed).
        :type hashes: np.ndarray
        :param dataset: Registry dataset (or evaluation family).
        :type dataset: str
        :param ref: Row id (or evaluation item).
        :type ref: str
        """
        hashes = np.asarray(hashes, dtype=np.uint64)
        if self.vocabulary is not None and len(hashes):
            hashes = hashes[contains(self.vocabulary, hashes)]
        pid = self.page_of.get(page)
        if pid is None:
            pid = self.page_of[page] = len(self.page_keys)
            self.page_keys.append(page)
            self.page_refs.append({})
            self._parts.append([])
        if len(hashes):
            self._parts[pid].append(hashes)
        entry = self.page_refs[pid].get(dataset)
        if entry is None:
            self.page_refs[pid][dataset] = [1, [ref]]
        else:
            entry[0] += 1
            if len(entry[1]) < self.max_refs:
                entry[1].append(ref)

    def finish(self) -> None:
        """Union each page's hashes and build the capped postings."""
        sizes = np.zeros(len(self.page_keys), dtype=np.int64)
        merged: List[np.ndarray] = []
        for pid, parts in enumerate(self._parts):
            arr = np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.uint64)
            sizes[pid] = len(arr)
            merged.append(arr)
            self._parts[pid] = []
        self._parts = []
        self.ptr = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
        self.hashes = np.concatenate(merged) if merged else np.zeros(0, dtype=np.uint64)
        del merged
        order = np.argsort(self.hashes, kind="stable")
        post_h = self.hashes[order]
        page_ids = np.searchsorted(self.ptr, order, side="right").astype(np.int32) - 1
        del order
        if len(post_h):
            starts = np.concatenate(([0], np.flatnonzero(np.diff(post_h)) + 1))
            lens = np.diff(np.concatenate((starts, [len(post_h)])))
            rank = np.arange(len(post_h)) - np.repeat(starts, lens)
            keep = rank < self.max_posting
            post_h, page_ids = post_h[keep], page_ids[keep]
        self.post_h, self.post_p = post_h, page_ids

    def __len__(self) -> int:
        """Number of pages.

        :return: Pages.
        :rtype: int
        """
        return len(self.page_keys)

    def page(self, pid: int) -> np.ndarray:
        """The sorted distinct hashes of one page.

        :param pid: Page id.
        :type pid: int
        :return: Hashes.
        :rtype: np.ndarray
        """
        return self.hashes[self.ptr[pid]:self.ptr[pid + 1]]

    def candidate_pages(self, positions: np.ndarray, top: int = TOP_PAGES) -> np.ndarray:
        """Pages sharing the most informative distinct 5-grams with a side, most first.

        Common 5-grams (postings at the cap) are used only when the side has no informative one.

        :param positions: The side's hashes.
        :type positions: np.ndarray
        :param top: Pages returned at most.
        :type top: int
        :return: Page ids.
        :rtype: np.ndarray
        """
        if not len(positions) or not len(self.post_h):
            return np.zeros(0, dtype=np.int64)
        uniq = np.unique(positions)
        lo = np.searchsorted(self.post_h, uniq, side="left")
        hi = np.searchsorted(self.post_h, uniq, side="right")
        n = hi - lo
        keep = (n > 0) & (n < self.max_posting)
        if not keep.any():
            keep = n > 0
            if not keep.any():
                return np.zeros(0, dtype=np.int64)
        starts, lens = lo[keep], n[keep]
        offsets = np.repeat(starts - np.concatenate(([0], np.cumsum(lens)[:-1])), lens) + np.arange(int(lens.sum()))
        counts = np.bincount(self.post_p[offsets], minlength=len(self.page_keys))
        nonzero = np.flatnonzero(counts)
        if len(nonzero) > top:
            nonzero = nonzero[np.argpartition(-counts[nonzero], top - 1)[:top]]
        return nonzero[np.argsort(-counts[nonzero], kind="stable")]

    def overlaps(self, positions: np.ndarray, rule: TextRule, top: int = TOP_PAGES) -> List[Tuple[int, Overlap]]:
        """Exact overlaps of a side with its candidate pages, the most overlapping first.

        :param positions: The side's hashes in reading order.
        :type positions: np.ndarray
        :param rule: Text rule.
        :type rule: TextRule
        :param top: Candidate pages checked.
        :type top: int
        :return: ``[(page id, overlap)]`` sorted by run letters, then shingle share, then share (descending).
        :rtype: List[Tuple[int, Overlap]]
        """
        out = [(int(pid), overlap(positions, self.page(int(pid)), rule))
               for pid in self.candidate_pages(positions, top)]
        out.sort(key=lambda x: (-x[1].run_letters, -x[1].shingle_share, -x[1].share, x[0]))
        return out


# ===================================================================================================================
# The seen registry
# ===================================================================================================================

def row_class(row: Mapping[str, Any], released_val_pgpids: Optional[Set[str]] = None,
              pool_val_holdout: bool = False) -> Optional[str]:
    """Class of a registry row (module docstring); None when the row was never trained and does not count.

    :param row: Registry row.
    :type row: Mapping[str, Any]
    :param released_val_pgpids: PGP ids of the released ``held_out_edition_pages_v0`` reserve: a val row of
        :data:`RELEASED_VAL_DATASET` with one of them is ``released_val``.
    :type released_val_pgpids: Optional[Set[str]]
    :param pool_val_holdout: Treat the val rows of pool datasets as ``eval_only`` instead of ``pool_val``.
    :type pool_val_holdout: bool
    :return: ``trained``, ``pool``, ``eval_only``, ``pool_val``, ``released_val`` or None.
    :rtype: Optional[str]
    """
    dataset = str(row.get("dataset") or "")
    if dataset in NEVER_TRAINED_DATASETS:
        return None
    notes = notes_text(row)
    m = _STATUS.search(notes)
    status = m.group(1) if m else None
    if status in ("not_trained", "never_trained"):
        return None
    if status == "eval_only" or _VAL_SPLIT.search(str(row.get("split") or "")):
        if (dataset == RELEASED_VAL_DATASET and released_val_pgpids
                and set(listify(row.get("pgpid"))) & set(released_val_pgpids)):
            return RELEASED_VAL
        if dataset in DEFINITE_DATASETS or pool_val_holdout:
            return EVAL_ONLY
        return POOL_VAL
    if dataset not in DEFINITE_DATASETS:
        return POOL
    if status == "partial" and "consumed=0" in notes:
        return POOL
    return TRAINED


@dataclass
class Photo:
    """One photograph of a candidate.

    :ivar sha256: File digest.
    :ivar phash: Perceptual hash (16 hex characters).
    :ivar url: Remote URL.
    :ivar fl_ids: KTIV page ids found in its URL, path or note.
    :ivar label: Canvas or file label (record only).
    :ivar path: Local file (for pHash confirmation), if any.
    """

    sha256: Optional[str] = None
    phash: Optional[str] = None
    url: Optional[str] = None
    fl_ids: List[str] = field(default_factory=list)
    label: Optional[str] = None
    path: Optional[str] = None

    @property
    def key(self) -> str:
        """The photograph's record key: sha256, else URL, else pHash.

        :return: Key (empty when the photograph has none).
        :rtype: str
        """
        return str(self.sha256 or self.url or self.phash or "")


@dataclass
class Subject:
    """A target side (or a whole ledger candidate) to judge.

    :ivar row_id: Candidate id.
    :ivar side_key: Target side (``""`` for a whole ledger candidate).
    :ivar source: Ledger source.
    :ivar document_group: Ledger document group.
    :ivar text: Target text.
    :ivar pgpids: PGP ids.
    :ivar canonical_ids: ES canonical ids (as recorded and ``cids_es``).
    :ivar shelf_keys_clean: Institution-qualified shelf keys.
    :ivar shelf_keys_legacy: Legacy institution-free shelf keys.
    :ivar ktiv_id: KTIV system number.
    :ivar fl_ids: KTIV page ids of the candidate.
    :ivar join_partners: Verified join partners (ES ids).
    :ivar photos: The candidate's photographs.
    :ivar own_photos: Keys (:attr:`Photo.key`) of the photographs that are this side's page.
    :ivar own_basis: Why those are its own (``stage2_accepted``, ``single_page_candidate``, ``ledger_image``, ``none``).
    :ivar candidate: The candidate record (for the facade identity of the hold-out).
    :ivar meta: Record-only fields.
    """

    row_id: str
    side_key: str
    source: str
    document_group: str
    text: str
    pgpids: List[str] = field(default_factory=list)
    canonical_ids: List[str] = field(default_factory=list)
    shelf_keys_clean: List[str] = field(default_factory=list)
    shelf_keys_legacy: List[str] = field(default_factory=list)
    ktiv_id: Optional[str] = None
    fl_ids: List[str] = field(default_factory=list)
    join_partners: List[str] = field(default_factory=list)
    photos: List[Photo] = field(default_factory=list)
    own_photos: List[str] = field(default_factory=list)
    own_basis: str = "none"
    candidate: Optional[Candidate] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def subject_id(self) -> str:
        """``row_id|side_key``.

        :return: The id.
        :rtype: str
        """
        return f"{self.row_id}|{self.side_key}"


class SeenRegistry:
    """Every identity key and page text of the seen registry, with the rows (witnesses) that carry them."""

    def __init__(self, vocabulary: Optional[np.ndarray] = None, released_val_pgpids: Optional[Set[str]] = None,
                 pool_val_holdout: bool = False, max_refs: int = MAX_REFS,
                 confirmer: Optional[PhashConfirmer] = None) -> None:
        """Create an empty registry index.

        :param vocabulary: Hashes to keep in the text index (see :class:`ShingleIndex`), or None for all.
        :type vocabulary: Optional[np.ndarray]
        :param released_val_pgpids: PGP ids of the released val reserve (:func:`row_class`).
        :type released_val_pgpids: Optional[Set[str]]
        :param pool_val_holdout: Val rows of pool datasets are hold-outs too (:func:`row_class`).
        :type pool_val_holdout: bool
        :param max_refs: Row ids kept per key, class and dataset.
        :type max_refs: int
        :param confirmer: pHash confirmation (a fresh uncached one when None).
        :type confirmer: Optional[PhashConfirmer]
        """
        self.keys = KeyIndex(max_refs)
        self.phash = PhashIndex(max_refs)
        self.confirmer = confirmer or PhashConfirmer()
        self.text = ShingleIndex(vocabulary, max_refs)
        self.released_val_pgpids = set(released_val_pgpids or ())
        self.pool_val_holdout = pool_val_holdout
        self.rows: Counter = Counter()
        self.dropped: Counter = Counter()

    def add_row(self, row: Mapping[str, Any]) -> Optional[str]:
        """Index one registry row.

        :param row: Registry row (with the rekey overlay's ``cids_es`` / ``shelf_keys_clean`` when known).
        :type row: Mapping[str, Any]
        :return: The row's class, or None when it was dropped.
        :rtype: Optional[str]
        """
        cls = row_class(row, self.released_val_pgpids, self.pool_val_holdout)
        dataset = str(row.get("dataset") or "?")
        if cls is None:
            self.dropped[dataset] += 1
            return None
        self.rows[(dataset, cls)] += 1
        rid = str(row.get("row_id"))
        notes = notes_text(row)
        add = self.keys.add
        sha, url = row.get("image_sha256"), row.get("image_url")
        add("image_sha256", sha, cls, dataset, rid)
        add("image_url", url, cls, dataset, rid)
        for fl in fl_ids(row.get("row_id"), row.get("stem"), url, row.get("image_path"), note_value(notes, "ktiv_fl")):
            add("ktiv_fl", fl, cls, dataset, rid)
        self.phash.add(row.get("phash"), cls, dataset, rid, sha, row.get("image_path"))
        for v in listify(row.get("pgpid")) + listify(row.get("pgpids")):
            add("pgpid", v, cls, dataset, rid)
        for v in listify(row.get("canonical_id")) + listify(row.get("canonical_ids")) + listify(row.get("cids_es")):
            add("canonical_id", v, cls, dataset, rid)
        for v in listify(row.get("shelf_keys_clean")):
            add("shelf_key_clean", v, cls, dataset, rid)
        for v in listify(row.get("shelf_key")) + listify(row.get("shelf_keys")):
            add("shelf_key_legacy", v, cls, dataset, rid)
            parts = v.split("|")
            for i in range(1, len(parts)):
                add("shelf_prefix", "|".join(parts[:i]), cls, dataset, rid)
        add("ktiv_id", row.get("ktiv_id"), cls, dataset, rid)
        add("document_group", note_value(notes, "document_group"), cls, dataset, rid)
        for kind, pat in _LOOSE:
            for m in pat.finditer(notes):
                for v in re.split(r"[,; ]+", m.group(1).strip()):
                    add(kind, v, cls, dataset, rid)
        shingles = row.get("text_shingles") or []
        if len(shingles) >= 5:
            page_key = str(sha or url or f"row:{dataset}:{rid}")
            self.text.add((cls, page_key), np.asarray(shingles, dtype=np.uint64), dataset, rid)
        return cls

    def finish(self) -> None:
        """Freeze the numeric indexes."""
        self.phash.finish()
        self.text.finish()

    def image_witnesses(self, photo: Photo, max_bits: int = PHASH_MAX_BITS) -> List[Dict[str, Any]]:
        """Registry rows of the same photograph: sha256, URL, KTIV FL id, pHash within ``max_bits``.

        A pHash hit on a file of the same sha256 is left to the sha256 witness; any other hit is judged by
        :meth:`PhashConfirmer.judge`, and an unconfirmed one is kept with the class ``phash_rejected`` /
        ``phash_unverified`` (never evidence).

        :param photo: The photograph.
        :type photo: Photo
        :param max_bits: pHash tolerance.
        :type max_bits: int
        :return: Witnesses (``axis`` ``image``; pHash witnesses carry ``distance`` and ``distance256``).
        :rtype: List[Dict[str, Any]]
        """
        out = self.keys.lookup("image_sha256", photo.sha256, "image")
        out += self.keys.lookup("image_url", photo.url, "image")
        for fl in photo.fl_ids:
            out += self.keys.lookup("ktiv_fl", fl, "image")
        for hit in self.phash.within(photo.phash, max_bits):
            if photo.sha256 and photo.sha256 in hit.files:
                continue
            verdict, d256 = self.confirmer.judge(hit.distance, photo.path, hit.files.values())
            for (cls, dataset), (n_rows, refs) in sorted(hit.owners.items()):
                out.append({"axis": "image", "key": "phash", "value": photo.phash,
                            "class": cls if verdict == "confirmed" else
                            (PHASH_REJECTED if verdict == "rejected" else PHASH_UNVERIFIED),
                            "row_class": cls, "dataset": dataset, "row_ids": list(refs), "n_rows": n_rows,
                            "distance": hit.distance, "distance256": d256})
        return out

    def text_witnesses(self, positions: np.ndarray,
                       rule: TextRule) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Registry pages the rule says hold the side's text, and the nearest page per class for the record.

        :param positions: The side's registry-hashed 5-gram positions.
        :type positions: np.ndarray
        :param rule: Text rule.
        :type rule: TextRule
        :return: ``(witnesses, nearest)``: witnesses with ``key`` ``text_run`` / ``text_share``, the page's datasets
            and rows and its overlap figures; ``nearest`` = per class the highest run and share seen.
        :rtype: Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]
        """
        out: List[Dict[str, Any]] = []
        nearest: Dict[str, Dict[str, Any]] = {}
        for pid, ov in self.text.overlaps(positions, rule):
            cls, page_key = self.text.page_keys[pid]
            best = nearest.setdefault(cls, {"run_letters": 0, "share": 0.0, "shingle_share": 0.0})
            best["run_letters"] = max(best["run_letters"], ov.run_letters)
            best["share"] = max(best["share"], round(ov.share, 4))
            best["shingle_share"] = max(best["shingle_share"], round(ov.shingle_share, 4))
            reason = rule.reason(ov)
            if reason is None:
                continue
            for dataset, (n_rows, refs) in sorted(self.text.page_refs[pid].items()):
                w = {"axis": "text", "key": reason, "value": page_key[:24], "class": cls, "dataset": dataset,
                     "row_ids": list(refs), "n_rows": n_rows}
                w.update(overlap_summary(ov))
                out.append(w)
        return out, nearest

    def document_witnesses(self, subject: Subject) -> List[Dict[str, Any]]:
        """Registry rows of the subject's document (definite keys) and weak document evidence.

        Witnesses of the definite keys carry ``strength`` ``definite``; the legacy shelf key, loose ids, parent /
        child shelf keys and join partners carry ``strength`` ``weak`` (uncertain evidence whatever the row's class).

        :param subject: The side.
        :type subject: Subject
        :return: Witnesses (``axis`` ``document``).
        :rtype: List[Dict[str, Any]]
        """
        out: List[Dict[str, Any]] = []
        definite = {"strength": "definite"}
        weak = {"strength": "weak"}
        for v in subject.pgpids:
            out += self.keys.lookup("pgpid", v, "document", definite)
            out += self.keys.lookup("loose_pgpid", v, "document", weak)
        for v in subject.canonical_ids:
            out += self.keys.lookup("canonical_id", v, "document", definite)
            out += self.keys.lookup("loose_canonical_id", v, "document", weak)
        for v in subject.shelf_keys_clean:
            out += self.keys.lookup("shelf_key_clean", v, "document", definite)
        if subject.ktiv_id:
            out += self.keys.lookup("ktiv_id", subject.ktiv_id, "document", definite)
        if subject.document_group:
            out += self.keys.lookup("document_group", subject.document_group, "document", definite)
        for v in subject.shelf_keys_legacy:
            if self.keys.has("shelf_key_legacy", v):
                out += self.keys.lookup("shelf_key_legacy", v, "document", weak)
                continue
            parts = v.split("|")
            parent = next((p for p in ("|".join(parts[:i]) for i in range(len(parts) - 1, 0, -1))
                           if self.keys.has("shelf_key_legacy", p)), None)
            if parent:
                out += self.keys.lookup("shelf_key_legacy", parent, "document",
                                        {"strength": "weak", "relation": "shelf_parent"})
            out += self.keys.lookup("shelf_prefix", v, "document", {"strength": "weak", "relation": "shelf_child"})
        for partner in subject.join_partners:
            for w in self.keys.lookup("canonical_id", partner, "document",
                                      {"strength": "weak", "relation": "join_partner"}):
                w["key"] = "join_partner"
                out.append(w)
        return out


def read_overlay(path: Path) -> Dict[str, Tuple[List[str], List[str]]]:
    """The rekey overlay of one registry file: ``row_id -> (cids_es, shelf_keys_clean)`` (empty when absent).

    :param path: Overlay file.
    :type path: Path
    :return: The overlay.
    :rtype: Dict[str, Tuple[List[str], List[str]]]
    """
    out: Dict[str, Tuple[List[str], List[str]]] = {}
    if path.is_file():
        for o in iter_jsonl(path):
            out[str(o["row_id"])] = (list(o.get("cids_es") or []), list(o.get("shelf_keys_clean") or []))
    return out


def load_seen(seen_dir: Path = SEEN_DIR, rekey_dir: Path = REKEY_SEEN_DIR, vocabulary: Optional[np.ndarray] = None,
              released_val_pgpids: Optional[Set[str]] = None, pool_val_holdout: bool = False,
              confirmer: Optional[PhashConfirmer] = None, log: bool = True) -> SeenRegistry:
    """Index every registry file (read only).

    :param seen_dir: Registry directory (``*.jsonl``).
    :type seen_dir: Path
    :param rekey_dir: Rekey overlay directory (same file names).
    :type rekey_dir: Path
    :param vocabulary: Hashes to keep in the text index, or None.
    :type vocabulary: Optional[np.ndarray]
    :param released_val_pgpids: PGP ids of the released val reserve.
    :type released_val_pgpids: Optional[Set[str]]
    :param pool_val_holdout: Val rows of pool datasets are hold-outs too.
    :type pool_val_holdout: bool
    :param confirmer: pHash confirmation, or None for a fresh uncached one.
    :type confirmer: Optional[PhashConfirmer]
    :param log: Print one line per file to stderr.
    :type log: bool
    :return: The index.
    :rtype: SeenRegistry
    """
    reg = SeenRegistry(vocabulary, released_val_pgpids, pool_val_holdout, confirmer=confirmer)
    t0 = time.time()
    for path in sorted(seen_dir.glob("*.jsonl")):
        overlay = read_overlay(rekey_dir / path.name)
        n = 0
        for row in iter_jsonl(path):
            o = overlay.get(str(row.get("row_id")))
            if o:
                row["cids_es"], row["shelf_keys_clean"] = o
            reg.add_row(row)
            n += 1
        if log:
            print(f"seen {path.name}: {n:,} rows ({time.time() - t0:.0f} s)", file=sys.stderr)
    reg.finish()
    if log:
        print(f"seen registry: {sum(reg.rows.values()):,} rows kept, {sum(reg.dropped.values()):,} dropped, "
              f"{len(reg.text):,} text pages, {len(reg.text.hashes):,} page 5-grams, {len(reg.text.post_h):,} "
              f"postings ({time.time() - t0:.0f} s)", file=sys.stderr)
    return reg


# ===================================================================================================================
# Subjects (what is judged)
# ===================================================================================================================

def usable_photos(cand: Candidate) -> List[Photo]:
    """The photographs of a stage-1 candidate that have a file digest (present or fetched; one per sha256).

    :param cand: Stage-1 candidate.
    :type cand: Candidate
    :return: Photos.
    :rtype: List[Photo]
    """
    out: List[Photo] = []
    seen: Set[str] = set()
    for img in cand.images:
        if not img.sha256 or img.status not in ("present", "fetched") or img.sha256 in seen:
            continue
        seen.add(img.sha256)
        out.append(Photo(img.sha256, img.phash, img.url, fl_ids(img.url, img.path, img.note), img.label, img.path))
    return out


def subjects_from_candidate(cand: Candidate, placed: Mapping[Tuple[str, str], str],
                            partners: Mapping[Tuple[str, str], List[str]]) -> List[Subject]:
    """One subject per target side of a stage-1 candidate.

    :param cand: Stage-1 candidate (``schema.Candidate``).
    :type cand: Candidate
    :param placed: ``{(row_id, side_key): sha256}`` of the stage-2 accepted placements.
    :type placed: Mapping[Tuple[str, str], str]
    :param partners: Verified join partners per ``(source, row_id)``.
    :type partners: Mapping[Tuple[str, str], List[str]]
    :return: Subjects.
    :rtype: List[Subject]
    """
    photos = usable_photos(cand)
    notes = cand.notes or {}
    legacy = list(dict.fromkeys(listify(notes.get("shelf_key")) + listify(notes.get("shelf_keys"))))
    fls = fl_ids(cand.row_id, *[json.dumps(t.pointer) for t in cand.targets], *[p.url for p in photos])
    out = []
    for target in cand.targets:
        own, basis = [], "none"
        sha = placed.get((cand.row_id, target.side_key))
        if sha and any(p.sha256 == sha for p in photos):
            own, basis = [sha], "stage2_accepted"
        elif cand.unit == "page" and len(cand.targets) == 1 and len(photos) == 1:
            own, basis = [photos[0].key], "single_page_candidate"
        out.append(Subject(
            row_id=cand.row_id, side_key=target.side_key, source=cand.source, document_group=cand.document_group,
            text=target.text, pgpids=listify(cand.pgpids), canonical_ids=listify(cand.canonical_ids),
            shelf_keys_clean=listify(notes.get("shelf_keys_clean")), shelf_keys_legacy=legacy,
            ktiv_id=str(cand.ktiv_id) if cand.ktiv_id else None, fl_ids=fls,
            join_partners=list(partners.get((cand.source, cand.row_id), [])), photos=photos, own_photos=own,
            own_basis=basis, candidate=cand,
            meta={"unit": cand.unit, "shelfmark": cand.shelfmark, "rnl": bool(cand.rnl)}))
    return out


def candidate_from_ledger(row: Mapping[str, Any]) -> Candidate:
    """A ``schema.Candidate`` view of a ledger row (identity fields only; no targets, no images).

    :param row: ``unseen_pool.jsonl`` record.
    :type row: Mapping[str, Any]
    :return: The candidate.
    :rtype: Candidate
    """
    cids = list(dict.fromkeys(listify(row.get("canonical_id")) + listify(row.get("canonical_ids"))
                              + listify(row.get("cids_es"))))
    pgpids = list(dict.fromkeys(listify(row.get("pgpid")) + listify(row.get("pgpids"))))
    flags = row.get("flags") or {}
    return Candidate(row_id=str(row.get("row_id")), source=str(row.get("source_file") or row.get("source") or ""),
                     document_group=str(row.get("document_group") or ""), canonical_ids=cids, pgpids=pgpids,
                     ktiv_id=row.get("ktiv_id"), shelfmark=str(row.get("shelfmark") or ""),
                     script=str(row.get("script") or ""), languages_stated=row.get("languages"),
                     verdict=str(flags.get("verdict") or ""), witnesses=listify(flags.get("witnesses")),
                     unit=str(row.get("unit") or ""), n_letters=int(row.get("n_letters") or 0),
                     notes={"stem": row.get("stem")})


def subject_from_ledger(row: Mapping[str, Any], text: str, partners: Mapping[Tuple[str, str], List[str]]) -> Subject:
    """A subject for a whole ledger candidate (the uncertain_history re-run): its own image keys are its page.

    :param row: ``unseen_pool.jsonl`` record.
    :type row: Mapping[str, Any]
    :param text: The candidate's edition text (``candidates/<source>.jsonl``).
    :type text: str
    :param partners: Verified join partners per ``(source, row_id)``.
    :type partners: Mapping[Tuple[str, str], List[str]]
    :return: The subject (``side_key`` ``""``).
    :rtype: Subject
    """
    cand = candidate_from_ledger(row)
    photo = Photo(row.get("image_sha256"), row.get("phash"), row.get("image_url"),
                  fl_ids(row.get("image_url"), row.get("image_path"), row.get("row_id")), None, row.get("image_path"))
    photos = [photo] if photo.key else []
    legacy = list(dict.fromkeys(listify(row.get("shelf_key")) + listify(row.get("shelf_keys"))))
    return Subject(
        row_id=cand.row_id, side_key="", source=cand.source, document_group=cand.document_group, text=text or "",
        pgpids=cand.pgpids, canonical_ids=cand.canonical_ids, shelf_keys_clean=listify(row.get("shelf_keys_clean")),
        shelf_keys_legacy=legacy, ktiv_id=str(row["ktiv_id"]) if row.get("ktiv_id") else None,
        fl_ids=fl_ids(row.get("row_id"), row.get("image_url"), row.get("stem")),
        join_partners=list(partners.get((cand.source, cand.row_id), [])), photos=photos,
        own_photos=[photo.key] if photo.key else [], own_basis="ledger_image" if photo.key else "none",
        candidate=cand, meta={"unit": cand.unit, "shelfmark": cand.shelfmark})


# ===================================================================================================================
# Novelty decisions
# ===================================================================================================================

@dataclass
class NoveltyEvidence:
    """What the registry knows about one subject (every class; the accessors split them).

    :ivar text: Text witnesses.
    :ivar text_nearest: Highest run and shares per class (record only).
    :ivar photos: Per photograph key, its image witnesses.
    :ivar document: Document witnesses (definite and weak).
    """

    text: List[Dict[str, Any]] = field(default_factory=list)
    text_nearest: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    photos: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    document: List[Dict[str, Any]] = field(default_factory=list)

    def text_of(self, cls: str) -> List[Dict[str, Any]]:
        """Text witnesses of one class.

        :param cls: Row class.
        :type cls: str
        :return: Witnesses.
        :rtype: List[Dict[str, Any]]
        """
        return [w for w in self.text if w["class"] == cls]

    def photo_of(self, key: str, cls: str) -> List[Dict[str, Any]]:
        """Image witnesses of one photograph and class.

        :param key: Photograph key.
        :type key: str
        :param cls: Row class.
        :type cls: str
        :return: Witnesses.
        :rtype: List[Dict[str, Any]]
        """
        return [w for w in self.photos.get(key, []) if w["class"] == cls]

    def photos_of(self, cls: str) -> List[Dict[str, Any]]:
        """Image witnesses of one class over every photograph of the candidate (each tagged with its photograph).

        :param cls: Row class.
        :type cls: str
        :return: Witnesses.
        :rtype: List[Dict[str, Any]]
        """
        return [dict(w, photo=key) for key in self.photos for w in self.photo_of(key, cls)]

    def doc_definite(self, cls: str) -> List[Dict[str, Any]]:
        """Definite-key document witnesses of one class.

        :param cls: Row class.
        :type cls: str
        :return: Witnesses.
        :rtype: List[Dict[str, Any]]
        """
        return [w for w in self.document if w["class"] == cls and w.get("strength") == "definite"]

    def doc_uncertain(self) -> List[Dict[str, Any]]:
        """Uncertain document evidence: definite keys of pool rows, weak keys of trained or pool rows.

        :return: Witnesses.
        :rtype: List[Dict[str, Any]]
        """
        return [w for w in self.document if (w["class"] == POOL and w.get("strength") == "definite")
                or (w.get("strength") == "weak" and w["class"] in (TRAINED, POOL))]


def novelty_evidence(subject: Subject, seen: SeenRegistry, rule: TextRule,
                     positions: Optional[np.ndarray] = None) -> NoveltyEvidence:
    """Collect the registry evidence of a subject.

    :param subject: The side.
    :type subject: Subject
    :param seen: Registry index.
    :type seen: SeenRegistry
    :param rule: Text rule.
    :type rule: TextRule
    :param positions: Registry-hashed 5-gram positions of the subject's text (computed when None).
    :type positions: Optional[np.ndarray]
    :return: The evidence.
    :rtype: NoveltyEvidence
    """
    if positions is None:
        positions = registry_positions(subject.text)
    ev = NoveltyEvidence()
    ev.text, ev.text_nearest = seen.text_witnesses(positions, rule)
    for photo in subject.photos:
        ev.photos[photo.key] = seen.image_witnesses(photo)
    ev.document = seen.document_witnesses(subject)
    return ev


def tier_of(page_trained: bool, page_pool: bool, doc_trained: bool, doc_uncertain: bool) -> str:
    """The tier from the four evidence flags (module docstring, first match wins).

    :param page_trained: A trained page witness (own photograph or text).
    :type page_trained: bool
    :param page_pool: A pool page witness.
    :type page_pool: bool
    :param doc_trained: The document was trained (definite key or group member).
    :type doc_trained: bool
    :param doc_uncertain: Uncertain document evidence.
    :type doc_uncertain: bool
    :return: A tier of :data:`TIERS`.
    :rtype: str
    """
    if page_trained:
        return PREVIOUSLY_TRAINED
    if page_pool:
        return UNCERTAIN
    if doc_trained:
        return PAGE_LEVEL
    if doc_uncertain:
        return UNCERTAIN
    return STRICT


@dataclass
class GroupEvidence:
    """Evidence a subject inherits from the other members of its document group.

    :ivar trained: Witnesses that another member's own page or document was trained.
    :ivar uncertain: Witnesses that another member has only uncertain evidence.
    """

    trained: List[Dict[str, Any]] = field(default_factory=list)
    uncertain: List[Dict[str, Any]] = field(default_factory=list)


def side_tier(subject: Subject, ev: NoveltyEvidence, group: Optional[GroupEvidence] = None) -> str:
    """A subject's tier (with its group's evidence when given; without it, what it passes on to its group).

    :param subject: The side.
    :type subject: Subject
    :param ev: Its evidence.
    :type ev: NoveltyEvidence
    :param group: What its document group passes on, or None.
    :type group: Optional[GroupEvidence]
    :return: A tier.
    :rtype: str
    """
    group = group or GroupEvidence()
    return tier_of(bool(ev.text_of(TRAINED)) or any(ev.photo_of(k, TRAINED) for k in subject.own_photos),
                   bool(ev.text_of(POOL)) or any(ev.photo_of(k, POOL) for k in subject.own_photos),
                   bool(ev.doc_definite(TRAINED)) or bool(group.trained),
                   bool(ev.doc_uncertain()) or bool(group.uncertain))


def pair_tiers(subject: Subject, ev: NoveltyEvidence, group: Optional[GroupEvidence] = None) -> Dict[str, str]:
    """The tier of every (side, photograph) pair of the subject's candidate.

    :param subject: The side.
    :type subject: Subject
    :param ev: Its evidence.
    :type ev: NoveltyEvidence
    :param group: What its document group passes on, or None.
    :type group: Optional[GroupEvidence]
    :return: ``{photograph key: tier}``.
    :rtype: Dict[str, str]
    """
    group = group or GroupEvidence()
    doc_trained = bool(ev.doc_definite(TRAINED)) or bool(group.trained)
    doc_unc = bool(ev.doc_uncertain()) or bool(group.uncertain)
    return {p.key: tier_of(bool(ev.text_of(TRAINED)) or bool(ev.photo_of(p.key, TRAINED)),
                           bool(ev.text_of(POOL)) or bool(ev.photo_of(p.key, POOL)), doc_trained, doc_unc)
            for p in subject.photos if p.key}


def group_evidence(subjects: Sequence[Subject], own: Mapping[str, Tuple[str, NoveltyEvidence]],
                   ledger_members: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, GroupEvidence]:
    """What each subject inherits from the other members of its document group.

    A member whose own tier (:func:`side_tier` without group evidence) is ``previously_trained`` or
    ``novel_page_level`` makes the group's document trained; a member whose own tier is ``uncertain_history`` passes
    uncertainty on. Ledger candidates of the group that are not judged here contribute what their ledger verdict and
    witnesses say (:func:`ledger_member_evidence`), with the ledger's witnesses. Sibling sides
    of one candidate are members too; a subject never inherits from itself, and only members' own evidence is passed
    on (no chains).

    :param subjects: Every judged subject.
    :type subjects: Sequence[Subject]
    :param own: ``{subject_id: (own tier, evidence)}``.
    :type own: Mapping[str, Tuple[str, NoveltyEvidence]]
    :param ledger_members: ``{document_group: [ledger rows of the group that are not judged]}``.
    :type ledger_members: Mapping[str, Sequence[Mapping[str, Any]]]
    :return: ``{subject_id: GroupEvidence}``.
    :rtype: Dict[str, GroupEvidence]
    """
    by_group: Dict[str, List[Subject]] = defaultdict(list)
    for s in subjects:
        if s.document_group:
            by_group[s.document_group].append(s)
    out: Dict[str, GroupEvidence] = {}
    for s in subjects:
        g = GroupEvidence()
        for m in by_group.get(s.document_group, []) if s.document_group else []:
            if m.subject_id == s.subject_id:
                continue
            tier = own[m.subject_id][0]
            w = {"axis": "group", "key": "group_member", "value": m.subject_id, "member_tier": tier}
            if tier in (PREVIOUSLY_TRAINED, PAGE_LEVEL):
                g.trained.append(w)
            elif tier == UNCERTAIN:
                g.uncertain.append(w)
        for row in ledger_members.get(s.document_group, []) if s.document_group else []:
            flags = row.get("flags") or {}
            kind = ledger_member_evidence(row)
            w = {"axis": "group", "key": "ledger_member", "value": str(row.get("row_id")),
                 "member_tier": flags.get("verdict"), "witnesses": listify(flags.get("witnesses"))[:MAX_WITNESSES]}
            if kind == TRAINED:
                g.trained.append(w)
            elif kind == UNCERTAIN:
                g.uncertain.append(w)
        out[s.subject_id] = g
    return out


def ledger_member_evidence(row: Mapping[str, Any]) -> Optional[str]:
    """What a ledger candidate that is not judged here says about its document, from its ledger verdict and witnesses.

    :param row: ``unseen_pool.jsonl`` record.
    :type row: Mapping[str, Any]
    :return: ``trained`` (verdict ``previously_trained``, or a ``held_out`` candidate with a definite witness),
        ``uncertain_history`` (verdict ``uncertain_history``, or a ``held_out`` candidate with only pool witnesses),
        or None.
    :rtype: Optional[str]
    """
    flags = row.get("flags") or {}
    verdict = flags.get("verdict")
    if verdict == PREVIOUSLY_TRAINED:
        return TRAINED
    if verdict == UNCERTAIN:
        return UNCERTAIN
    if verdict == "held_out":
        labels = listify(flags.get("witnesses"))
        if any("~" not in w and w in DEFINITE_DATASETS for w in labels):
            return TRAINED
        if any(w.endswith("~pool") for w in labels):
            return UNCERTAIN
    return None


def novelty_record(subject: Subject, ev: NoveltyEvidence, group: GroupEvidence) -> Dict[str, Any]:
    """The novelty part of a record: tier, its basis, witnesses, flags, pair tiers.

    :param subject: The side.
    :type subject: Subject
    :param ev: Its evidence.
    :type ev: NoveltyEvidence
    :param group: What its group passes on.
    :type group: GroupEvidence
    :return: ``{"tier", "tier_basis", "witnesses", "n_witnesses", "flags", "pair_tiers", "text_nearest",
        "evidence_counts"}``.
    :rtype: Dict[str, Any]
    """
    tier = side_tier(subject, ev, group)
    img_trained = [dict(w, photo=k) for k in subject.own_photos for w in ev.photo_of(k, TRAINED)]
    img_pool = [dict(w, photo=k) for k in subject.own_photos for w in ev.photo_of(k, POOL)]
    if tier == PREVIOUSLY_TRAINED:
        kinds = (["text"] if ev.text_of(TRAINED) else []) + (["image"] if img_trained else [])
        basis, witnesses = "page:" + "+".join(kinds), ev.text_of(TRAINED) + img_trained
    elif tier == UNCERTAIN and (ev.text_of(POOL) or img_pool):
        basis, witnesses = "page_pool", ev.text_of(POOL) + img_pool
    elif tier == PAGE_LEVEL:
        basis = "document" if ev.doc_definite(TRAINED) else "group"
        witnesses = ev.doc_definite(TRAINED) + group.trained
    elif tier == UNCERTAIN:
        basis = "document_uncertain" if ev.doc_uncertain() else "group_uncertain"
        witnesses = ev.doc_uncertain() + group.uncertain
    else:
        basis, witnesses = "none", []
    flags = []
    own = set(subject.own_photos)
    for p in subject.photos:
        if p.key in own:
            continue
        if ev.photo_of(p.key, TRAINED):
            flags.append(f"photo_trained_unplaced:{p.key[:16]}")
        elif ev.photo_of(p.key, POOL):
            flags.append(f"photo_pool_unplaced:{p.key[:16]}")
    if tier == PREVIOUSLY_TRAINED and (ev.doc_definite(TRAINED) or group.trained):
        flags.append("document_trained")
    page_ws = img_trained + img_pool
    if page_ws and all(w.get("key") == "phash" for w in page_ws) and not ev.text_of(TRAINED) and not ev.text_of(POOL):
        flags.append("page_evidence_confirmed_phash_only")
    unconfirmed = [w for p in subject.photos for w in ev.photos.get(p.key, [])
                   if w["class"] in (PHASH_REJECTED, PHASH_UNVERIFIED)]
    if unconfirmed:
        flags.append(f"phash_unconfirmed:{sum(w['class'] == PHASH_REJECTED for w in unconfirmed)}_rejected,"
                     f"{sum(w['class'] == PHASH_UNVERIFIED for w in unconfirmed)}_unverified")
    if ev.text_of(POOL_VAL) or ev.doc_definite(POOL_VAL) or ev.photos_of(POOL_VAL):
        flags.append("pool_val_evidence")
    return {"tier": tier, "tier_basis": basis, "witnesses": cap(witnesses), "n_witnesses": len(witnesses),
            "flags": flags, "pair_tiers": pair_tiers(subject, ev, group), "text_nearest": ev.text_nearest,
            "evidence_counts": {
                "text_trained": len(ev.text_of(TRAINED)), "text_pool": len(ev.text_of(POOL)),
                "own_image_trained": len(img_trained), "own_image_pool": len(img_pool),
                "doc_trained": len(ev.doc_definite(TRAINED)), "doc_uncertain": len(ev.doc_uncertain()),
                "group_trained": len(group.trained), "group_uncertain": len(group.uncertain)}}


# ===================================================================================================================
# Evaluation sets
# ===================================================================================================================

@dataclass
class EvalSet:
    """One evaluation set (or a released reserve) as hold-out keys.

    :ivar name: Set name (the witness's ``set``).
    :ivar family: Witness family (``pgp131``, ``religious140``, ``sealed_v1``, ``arabic_v0``, ``named_exclusion``,
        ``ktiv_twin``, ``new_eval``, ``released_reserve``, ...).
    :ivar bench: ES ids / PGP ids / KTIV manuscripts / loose keys (``novel_build.holdout.BenchmarkSet``).
    :ivar released: Recorded only, never holding out.
    :ivar fl: KTIV page id -> item.
    :ivar names: Other identifiers (row ids, stems, namespaced ids, document groups) -> item.
    :ivar images: ``(sha256, phash, item, path)`` of the set's photographs.
    :ivar gt: ``(item, ground-truth text)``.
    """

    name: str
    family: str
    bench: nh.BenchmarkSet
    released: bool = False
    fl: Dict[str, str] = field(default_factory=dict)
    names: Dict[str, str] = field(default_factory=dict)
    images: List[Tuple[Optional[str], Optional[str], str, Optional[str]]] = field(default_factory=list)
    gt: List[Tuple[str, str]] = field(default_factory=list)


class EvalIndex:
    """Every evaluation set's identity keys, photographs and ground-truth texts."""

    def __init__(self, sets: Sequence[EvalSet], rule: TextRule, confirmer: Optional[PhashConfirmer] = None) -> None:
        """Index the sets.

        :param sets: Evaluation sets (released ones included, flagged).
        :type sets: Sequence[EvalSet]
        :param rule: Text rule.
        :type rule: TextRule
        :param confirmer: pHash confirmation, or None for a fresh uncached one.
        :type confirmer: Optional[PhashConfirmer]
        """
        self.sets = list(sets)
        self.rule = rule
        self.confirmer = confirmer or PhashConfirmer()
        self.unconfirmed: List[Dict[str, Any]] = []
        self.sha: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        self.phash = PhashIndex(max_refs=8, max_files=8)
        self.text = ShingleIndex()
        for si, es in enumerate(self.sets):
            for sha, ph, item, path in es.images:
                if sha and (si, item) not in self.sha[str(sha)]:
                    self.sha[str(sha)].append((si, item))
                self.phash.add(ph, str(si), es.family, item, sha, path)
            for item, gt in es.gt:
                self.text.add((str(si), item), skeleton_positions(gt), es.family, item)
        self.phash.finish()
        self.text.finish()

    def _witness(self, si: int, item: str, key: str, value: Any, **extra: Any) -> Dict[str, Any]:
        """One evaluation witness.

        :param si: Set index.
        :type si: int
        :param item: Evaluation item.
        :type item: str
        :param key: Key kind that matched.
        :type key: str
        :param value: Matched value.
        :type value: Any
        :param extra: Extra fields.
        :type extra: Any
        :return: ``{"family", "set", "item", "key", "value", "released", ...}``.
        :rtype: Dict[str, Any]
        """
        es = self.sets[si]
        w = {"family": es.family, "set": es.name, "item": item, "key": key, "value": str(value),
             "released": es.released}
        w.update(extra)
        return w

    def identity_witnesses(self, ident: nh.Identity, fls: Iterable[str]) -> List[Dict[str, Any]]:
        """Evaluation items a candidate's document identity matches.

        :param ident: The candidate's identity (:func:`candidate_ident`).
        :type ident: nh.Identity
        :param fls: The candidate's KTIV page ids.
        :type fls: Iterable[str]
        :return: Witnesses.
        :rtype: List[Dict[str, Any]]
        """
        out = []
        fls = set(fls)
        for si, es in enumerate(self.sets):
            b = es.bench
            out += [self._witness(si, b.ids[c], "canonical_id", c) for c in sorted(ident.ids) if c in b.ids]
            out += [self._witness(si, f"pgpid:{p}", "pgpid", p) for p in sorted(ident.pgpids & b.pgpids)]
            out += [self._witness(si, b.sys_nums[s], "ktiv_sys_num", s) for s in sorted(ident.sys_nums)
                    if s in b.sys_nums]
            out += [self._witness(si, b.keys[k], "loose_key", k) for k in sorted(ident.keys) if k in b.keys]
            out += [self._witness(si, es.fl[f], "ktiv_fl", f) for f in sorted(fls) if f in es.fl]
            out += [self._witness(si, es.names[n], "name", n) for n in sorted(ident.names) if n in es.names]
        return out

    def image_witnesses(self, photo: Photo, max_bits: int = PHASH_MAX_BITS) -> List[Dict[str, Any]]:
        """Evaluation items with the same photograph (sha256, or pHash within ``max_bits``).

        pHash hits are judged by :meth:`PhashConfirmer.judge`; unconfirmed ones go to :attr:`unconfirmed` only.

        :param photo: The photograph.
        :type photo: Photo
        :param max_bits: pHash tolerance.
        :type max_bits: int
        :return: Witnesses.
        :rtype: List[Dict[str, Any]]
        """
        out = [self._witness(si, item, "image_sha256", photo.sha256)
               for si, item in self.sha.get(str(photo.sha256), [])] if photo.sha256 else []
        for hit in self.phash.within(photo.phash, max_bits):
            if photo.sha256 and photo.sha256 in hit.files:
                continue
            verdict, d256 = self.confirmer.judge(hit.distance, photo.path, hit.files.values())
            found = [self._witness(int(si), item, "phash", photo.phash, distance=hit.distance, distance256=d256)
                     for (si, _family), (_n, items) in sorted(hit.owners.items()) for item in items]
            if verdict == "confirmed":
                out += found
            else:
                self.unconfirmed += [dict(w, verdict=verdict, photo=photo.key) for w in found]
        return out

    def text_witnesses(self, text: str) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Evaluation documents whose ground truth the rule says holds the text, and the nearest one.

        :param text: Target text.
        :type text: str
        :return: ``(witnesses, nearest)``: witnesses with ``key`` ``text_run`` / ``text_share`` and the overlap
            figures; ``nearest`` = the most overlapping document (``set``, ``item`` and figures) or None.
        :rtype: Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]
        """
        out = []
        nearest = None
        for pid, ov in self.text.overlaps(skeleton_positions(text), self.rule):
            si, item = self.text.page_keys[pid]
            if nearest is None:
                nearest = dict(overlap_summary(ov), set=self.sets[int(si)].name, item=item)
            reason = self.rule.reason(ov)
            if reason is not None:
                out.append(self._witness(int(si), item, reason, item, **overlap_summary(ov)))
        return out, nearest


def image_keys_cached(paths: Sequence[Path], cache_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """sha256 and pHash of image files (``novel_build.images.image_keys``), cached by path, size and mtime.

    :param paths: Image files (read only).
    :type paths: Sequence[Path]
    :param cache_path: JSON cache file (never under ``raw_data``), or None.
    :type cache_path: Optional[Path]
    :return: ``{str(path): {"sha256", "phash", "size", "mtime"}}`` for the files that exist and decode.
    :rtype: Dict[str, Dict[str, Any]]
    :raises ValueError: When the cache path lies under ``raw_data``.
    """
    if cache_path is not None and "raw_data" in Path(cache_path).resolve().parts:
        raise ValueError(f"the image-key cache must not live under raw_data: {cache_path}")
    cache: Dict[str, Dict[str, Any]] = {}
    if cache_path is not None and cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}
    out: Dict[str, Dict[str, Any]] = {}
    changed = False
    for path in paths:
        try:
            st = path.stat()
            hit = cache.get(str(path))
            if hit and hit.get("size") == st.st_size and hit.get("mtime") == st.st_mtime_ns:
                out[str(path)] = hit
                continue
            keys = image_keys(path.read_bytes())
        except OSError:
            continue
        cache[str(path)] = out[str(path)] = {"sha256": keys["sha256"], "phash": keys["phash"], "size": st.st_size,
                                             "mtime": st.st_mtime_ns}
        changed = True
    if changed and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=0), encoding="utf-8")
    return out


def _docs(path: Path) -> List[Dict[str, Any]]:
    """The ``docs`` of a benchmark JSON.

    :param path: Benchmark JSON.
    :type path: Path
    :return: Documents.
    :rtype: List[Dict[str, Any]]
    """
    return list(json.loads(path.read_text(encoding="utf-8"))["docs"])


def _ktiv_page_items(docs: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """``FL id -> doc id`` of KTIV page documents (``KTIV_<sys>_<FL>`` ids or an ``fl`` field).

    :param docs: Benchmark documents.
    :type docs: Sequence[Mapping[str, Any]]
    :return: The map.
    :rtype: Dict[str, str]
    """
    out: Dict[str, str] = {}
    for d in docs:
        for fl in fl_ids(d.get("doc_id"), d.get("fl")):
            out.setdefault(fl, str(d.get("doc_id")))
    return out


def _ktiv_image_name(doc_id: str) -> str:
    """``<sys>_<FL>.jpg`` of a ``KTIV_<sys>_<FL>`` page id (the religious / sealed literary image files).

    :param doc_id: Page id.
    :type doc_id: str
    :return: File name.
    :rtype: str
    """
    return "_".join(str(doc_id).split("_")[1:3]) + ".jpg"


def load_new_eval_sets(directory: Path, index: Optional[shelf.ShelfmarkIndex]) -> List[EvalSet]:
    """The new evaluation sets registered under ``W2/eval/registered/*.json`` (format in the module docstring).

    :param directory: Registration directory (missing or empty: no sets).
    :type directory: Path
    :param index: Merged-corpus index for resolving ids and shelf marks, or None.
    :type index: Optional[shelf.ShelfmarkIndex]
    :return: The sets.
    :rtype: List[EvalSet]
    :raises ValueError: On a file without ``eval_set`` or an item without ``item_id``.
    """
    out: List[EvalSet] = []
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.json")):
        spec = json.loads(path.read_text(encoding="utf-8"))
        name = spec.get("eval_set")
        if not name:
            raise ValueError(f"{path}: no eval_set name")
        es = EvalSet(name=str(name), family="new_eval", bench=nh.BenchmarkSet(str(name)))
        for item in spec.get("items") or []:
            label = str(item.get("item_id") or "")
            if not label:
                raise ValueError(f"{path}: an item has no item_id")
            for cid in listify(item.get("canonical_ids")):
                es.bench.add_id(cid, label, index)
            for pgpid in listify(item.get("pgpids")):
                es.bench.pgpids.add(pgpid)
                if index is not None:
                    for resolved in sorted(lsr.es_ids(f"pgpid:{pgpid}", index)):
                        es.bench.ids.setdefault(resolved, label)
            for mark in listify(item.get("shelfmarks")):
                es.bench.add_mark(mark, label, index)
                for part in nh.mark_parts(mark):
                    cid = shelf.canonical_id(part)
                    if cid:
                        es.bench.ids.setdefault(cid, label)
            for sys_num in listify(item.get("ktiv_sys_nums")):
                es.bench.sys_nums.setdefault(sys_num, label)
            for fl in listify(item.get("ktiv_fl")):
                es.fl.setdefault(fl, label)
            shas, phs = listify(item.get("image_sha256")), listify(item.get("phash"))
            for i in range(max(len(shas), len(phs))):
                es.images.append((shas[i] if i < len(shas) else None, phs[i] if i < len(phs) else None, label, None))
            for n in listify(item.get("row_ids")) + listify(item.get("document_group")):
                es.names.setdefault(n, label)
        gt_path = spec.get("gt_jsonl")
        if gt_path and Path(gt_path).is_file():
            es.gt = [(str(r["item_id"]), str(r.get("text") or "")) for r in iter_jsonl(Path(gt_path))]
        out.append(es)
    return out


def load_eval_sets(index: Optional[shelf.ShelfmarkIndex], *, registry_dir: Path = br.REGISTRY_DIR,
                   eval_dir: Path = EVAL_DIR, arabic_v0_dir: Path = ARABIC_V0_DIR,
                   named_path: Path = NAMED_EXCLUSIONS, apd_path: Path = APD_EXCLUSIONS,
                   registered_dir: Path = W2 / "eval" / "registered", pgp_export: Path = lsr.PGP_EXPORT,
                   image_cache: Optional[Path] = None, with_images: bool = True) -> List[EvalSet]:
    """Every evaluation set of the module docstring, and the released reserves (flagged); read only.

    :param index: Merged-corpus index, or None (ids as recorded and loose keys only).
    :type index: Optional[shelf.ShelfmarkIndex]
    :param registry_dir: ``decontam`` registry directory.
    :type registry_dir: Path
    :param eval_dir: ``raw_data/cairo_genizah/evaluations``.
    :type eval_dir: Path
    :param arabic_v0_dir: Arabic-script benchmark v0 directory (``benchmark.jsonl``, ``images/``).
    :type arabic_v0_dir: Path
    :param named_path: Named exclusions JSON.
    :type named_path: Path
    :param apd_path: APD exclusion list of v2.5a.
    :type apd_path: Path
    :param registered_dir: New evaluation sets.
    :type registered_dir: Path
    :param pgp_export: Pinned PGP export (``documents.csv``) for the registered PGP documents' shelf marks.
    :type pgp_export: Path
    :param image_cache: Cache file of the evaluation photographs' keys.
    :type image_cache: Optional[Path]
    :param with_images: Hash the evaluation photographs.
    :type with_images: bool
    :return: The sets.
    :rtype: List[EvalSet]
    :raises FileNotFoundError: When a registry or benchmark file is missing.
    """
    documents = pgp_export / "documents.csv"
    marks = lsr.pgp_shelf_marks(documents) if documents.exists() else {}
    sets: List[EvalSet] = []
    image_files: List[Tuple[int, Path, str]] = []
    family = {"arabic_script_benchmark_v0": "arabic_v0", "sealed_v1": "sealed_v1"}
    for name in br.BENCHMARK_REGISTRIES:
        released = name in RELEASED_REGISTRIES
        sets.append(EvalSet(name, "released_reserve" if released else family.get(name, "registered_benchmark"),
                            nh.registry_set(name, registry_dir, marks, index), released=released))
    position = {s.name: i for i, s in enumerate(sets)}
    if "arabic_script_benchmark_v0" in position:
        si = position["arabic_script_benchmark_v0"]
        for r in iter_jsonl(arabic_v0_dir / "benchmark.jsonl"):
            label = str(r["id"])
            gt = r.get("gt") or {}
            sets[si].gt.append((label, "\n".join(line for sec in gt.get("sections") or []
                                                  for line in sec.get("lines") or [])))
            imgs = r.get("images")
            if isinstance(imgs, str):
                imgs = ast.literal_eval(imgs) if imgs.strip() else []   # the benchmark stores a Python-literal list
            for img in imgs or []:
                if isinstance(img, Mapping) and img.get("file"):
                    image_files.append((si, arabic_v0_dir / "images" / str(img["file"]), label))
    if "sealed_v1" in position:
        si = position["sealed_v1"]
        for part in ("documentary", "literary"):
            docs = _docs(eval_dir / "sealed_v1" / part / f"sealed_v1_{part}.json")
            sets[si].gt += [(str(d["doc_id"]), str(d.get("gt") or "")) for d in docs]
            sets[si].fl.update(_ktiv_page_items(docs))
            for d in docs:
                doc_id = str(d["doc_id"])
                if part == "documentary":
                    if d.get("image_sha256"):
                        sets[si].images.append((str(d["image_sha256"]), None, doc_id, None))
                    image_files.append((si, eval_dir / "sealed_v1" / part / "images" / f"{doc_id}.jpg", doc_id))
                else:
                    image_files.append((si, eval_dir / "sealed_v1" / part / "images" / _ktiv_image_name(doc_id),
                                        doc_id))
    pgp131 = EvalSet("pgp131", "pgp131", nh.frozen_set(nh.FROZEN_IDS, nh.FROZEN_KEYS, index))
    pgp131.gt = [(str(d["doc_id"]), str(d.get("gt") or ""))
                 for d in _docs(eval_dir / "genizah_test_v1" / "genizah_test_v1.json")]
    for d in _docs(eval_dir / "genizah_test_v1" / "genizah_test_v1_verified.json"):
        image_files.append((len(sets), eval_dir / "genizah_test_v1" / "images" / f"{d['doc_id']}.jpg",
                            str(d["doc_id"])))
    sets.append(pgp131)
    religious = EvalSet("religious140", "religious140", nh.religious_set(nh.RELIGIOUS))
    rel_docs = _docs(nh.RELIGIOUS)
    religious.gt = [(str(d["doc_id"]), str(d.get("gt") or "")) for d in rel_docs]
    religious.fl = _ktiv_page_items(rel_docs)
    for d in rel_docs:
        image_files.append((len(sets), nh.RELIGIOUS.parent / "images" / _ktiv_image_name(str(d["doc_id"])),
                            str(d["doc_id"])))
    sets.append(religious)
    named = json.loads(named_path.read_text(encoding="utf-8")) if named_path.is_file() else {}
    for entry in nh.compile_named(named):
        es = EvalSet(f"named:{entry.name}", "named_exclusion", nh.BenchmarkSet(f"named:{entry.name}"))
        es.bench.pgpids = set(entry.pgpids)
        es.bench.ids = {c: entry.name for c in entry.ids}
        es.bench.keys = {k: entry.name for k in entry.keys}
        es.names = {entry.name: entry.name}
        sets.append(es)
    twin = EvalSet("ktiv_FR_9-002.2", "ktiv_twin", nh.BenchmarkSet("ktiv_FR_9-002.2"))
    twin.bench.sys_nums[FR_9_002_2["sys_num"]] = "FR 9-002.2"
    twin.bench.add_mark(FR_9_002_2["shelf_mark"], "FR 9-002.2", index)
    sets.append(twin)
    if apd_path.is_file():
        apd_eval = EvalSet("apd_exclusions:benchmark_holdout", "arabic_v0", nh.BenchmarkSet("apd_benchmark_holdout"))
        apd_rel = EvalSet("apd_exclusions:pgp_arabic_editions_v2", "released_apd_exclusion",
                          nh.BenchmarkSet("apd_pgp_arabic_editions_v2"), released=True)
        for rec in iter_jsonl(apd_path):
            doc = str(rec.get("doc_id") or "")
            if doc:
                reasons = set(listify(rec.get("reasons")) or listify(rec.get("reason")))
                target = apd_eval if reasons & APD_EVAL_REASONS else apd_rel
                target.names.setdefault(doc, doc)
                target.names.setdefault(f"apd:{doc}", doc)
        sets += [apd_eval, apd_rel]
    sets += load_new_eval_sets(registered_dir, index)
    if with_images and image_files:
        keys = image_keys_cached([p for _, p, _ in image_files], image_cache)
        for si, path, label in image_files:
            k = keys.get(str(path))
            if k:
                sets[si].images.append((k["sha256"], k["phash"], label, str(path)))
    return sets


def candidate_ident(subject: Subject, index: Optional[shelf.ShelfmarkIndex]) -> nh.Identity:
    """The facade identity of a subject's candidate (``novel_build.holdout.candidate_identity``), plus its verified
    join partners, its KTIV manuscript and its document group (as a name).

    :param subject: The side.
    :type subject: Subject
    :param index: Merged-corpus index, or None.
    :type index: Optional[shelf.ShelfmarkIndex]
    :return: The identity.
    :rtype: nh.Identity
    """
    cand = subject.candidate or Candidate(subject.row_id, subject.source, subject.document_group,
                                          list(subject.canonical_ids), list(subject.pgpids), subject.ktiv_id, "", "",
                                          None, "", [], "", 0)
    ident = nh.candidate_identity(cand, index)
    for partner in subject.join_partners:
        ident.ids.add(partner)
        key = nh.key_of(partner)
        if key:
            ident.keys.add(key)
    if subject.ktiv_id:
        ident.sys_nums.add(str(subject.ktiv_id))
    if subject.document_group:
        ident.names.add(subject.document_group)
    return ident


# ===================================================================================================================
# Judging a run
# ===================================================================================================================

def registry_val_witnesses(ev: NoveltyEvidence, text_level: bool) -> List[Dict[str, Any]]:
    """Registry eval-only evidence of a subject as hold-out witnesses (family ``registry_val``).

    :param ev: The subject's registry evidence.
    :type ev: NoveltyEvidence
    :param text_level: True: the text witnesses (side-level); False: the definite document keys and the candidate's
        photographs (group-level, like an identity).
    :type text_level: bool
    :return: Witnesses.
    :rtype: List[Dict[str, Any]]
    """
    found = ev.text_of(EVAL_ONLY) if text_level else ev.doc_definite(EVAL_ONLY) + ev.photos_of(EVAL_ONLY)
    return [dict(w, family="registry_val", set=w["dataset"], item=(w.get("row_ids") or [""])[0], released=False)
            for w in found]


def released_val_witnesses(ev: NoveltyEvidence) -> List[Dict[str, Any]]:
    """Evidence of the released val reserve (recorded only).

    :param ev: The subject's registry evidence.
    :type ev: NoveltyEvidence
    :return: Witnesses of ``released_val`` rows.
    :rtype: List[Dict[str, Any]]
    """
    found = ev.text_of(RELEASED_VAL) + ev.doc_definite(RELEASED_VAL) + ev.photos_of(RELEASED_VAL)
    return [dict(w, released=True, family="released_reserve", set="held_out_edition_pages_v0") for w in found]


def ledger_val_witnesses(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Eval-only (``~val``) witnesses the ledger recorded for a candidate that is not judged here.

    :param row: ``unseen_pool.jsonl`` record.
    :type row: Mapping[str, Any]
    :return: ``registry_val`` witnesses of definite datasets (pool datasets' val rows are not hold-outs).
    :rtype: List[Dict[str, Any]]
    """
    flags = row.get("flags") or {}
    out = []
    for axis in ("image", "document"):
        for entry in flags.get(axis) or []:
            kind, _, datasets = str(entry).partition(":")
            for ds in datasets.split(","):
                name = ds.split("@")[0]
                if name.endswith("~val") and name[:-4] in DEFINITE_DATASETS:
                    out.append({"family": "registry_val", "set": name[:-4], "item": str(row.get("row_id")),
                                "key": f"ledger_{axis}_{kind}", "value": str(row.get("row_id")), "released": False})
    return out


def holdout_record(group_ws: Sequence[Mapping[str, Any]], text_ws: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The evaluation hold-out part of a record.

    :param group_ws: Identity, image and registry-val identity witnesses of every member of the document group.
    :type group_ws: Sequence[Mapping[str, Any]]
    :param text_ws: Text witnesses of the side (evaluation ground truth and eval-only registry pages).
    :type text_ws: Sequence[Mapping[str, Any]]
    :return: ``{"eval_holdout", "eval_families", "eval_witnesses", "n_eval_witnesses", "released",
        "released_sets"}``.
    :rtype: Dict[str, Any]
    """
    every = list(group_ws) + list(text_ws)
    held = [w for w in every if not w.get("released")]
    released = [w for w in every if w.get("released")]
    held.sort(key=lambda w: (str(w.get("family")), str(w.get("key"))))
    return {"eval_holdout": bool(held), "eval_families": sorted({str(w.get("family")) for w in held}),
            "eval_witnesses": cap(held, 2 * MAX_WITNESSES), "n_eval_witnesses": len(held),
            "eval_text_holdout": any(not w.get("released") for w in text_ws),
            "released": cap(released), "released_sets": sorted({str(w.get("set")) for w in released})}


@dataclass
class RunResult:
    """Records of a judged run.

    :ivar records: One record per subject, in input order.
    """

    records: List[Dict[str, Any]]


def judge_run(subjects: Sequence[Subject], seen: SeenRegistry, evals: EvalIndex,
              index: Optional[shelf.ShelfmarkIndex], rule: TextRule,
              ledger_members: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
              log_every: int = 1000) -> RunResult:
    """Judge every subject: registry evidence, group inheritance, tiers, evaluation hold-outs.

    Identity hold-outs (identity keys, photographs, registry eval-only document keys and photographs) of any member of
    a document group (the judged subjects and the ledger members not judged here) hold out every judged side of the
    group; text hold-outs hold out the side whose text matched.

    :param subjects: Subjects (W1 sides, ledger candidates, or both: groups span the run).
    :type subjects: Sequence[Subject]
    :param seen: Registry index.
    :type seen: SeenRegistry
    :param evals: Evaluation index.
    :type evals: EvalIndex
    :param index: Merged-corpus index for the facade identity, or None.
    :type index: Optional[shelf.ShelfmarkIndex]
    :param rule: Text rule.
    :type rule: TextRule
    :param ledger_members: ``{document_group: [ledger rows not judged here]}``.
    :type ledger_members: Optional[Mapping[str, Sequence[Mapping[str, Any]]]]
    :param log_every: Progress line every this many subjects (0: none).
    :type log_every: int
    :return: The records.
    :rtype: RunResult
    """
    ledger_members = ledger_members or {}
    t0 = time.time()
    own: Dict[str, Tuple[str, NoveltyEvidence]] = {}
    ident_cache: Dict[str, List[Dict[str, Any]]] = {}
    text_ws: Dict[str, List[Dict[str, Any]]] = {}
    nearest: Dict[str, Optional[Dict[str, Any]]] = {}
    for i, s in enumerate(subjects):
        ev = novelty_evidence(s, seen, rule)
        own[s.subject_id] = (side_tier(s, ev), ev)
        if s.row_id not in ident_cache:
            ws = evals.identity_witnesses(candidate_ident(s, index), s.fl_ids)
            for p in s.photos:
                ws += evals.image_witnesses(p)
            ident_cache[s.row_id] = ws
        found, nearest[s.subject_id] = evals.text_witnesses(s.text)
        text_ws[s.subject_id] = found + registry_val_witnesses(ev, text_level=True)
        if log_every and (i + 1) % log_every == 0:
            print(f"judged {i + 1:,}/{len(subjects):,} ({time.time() - t0:.0f} s)", file=sys.stderr)
    groups = group_evidence(subjects, own, ledger_members)
    by_group: Dict[str, List[Subject]] = defaultdict(list)
    for s in subjects:
        by_group[s.document_group or s.subject_id].append(s)
    group_ws: Dict[str, List[Dict[str, Any]]] = {}
    for g, members in by_group.items():
        ws: List[Dict[str, Any]] = []
        for m in members:
            ws += [dict(w, member=m.row_id) for w in ident_cache[m.row_id]]
            ws += [dict(w, member=m.subject_id) for w in registry_val_witnesses(own[m.subject_id][1], text_level=False)]
        for row in ledger_members.get(g, []):
            member = str(row.get("row_id"))
            ident = nh.candidate_identity(candidate_from_ledger(row), index)
            ws += [dict(w, member=member)
                   for w in evals.identity_witnesses(ident, fl_ids(row.get("row_id"), row.get("image_url")))]
            ws += [dict(w, member=member) for w in ledger_val_witnesses(row)]
        group_ws[g] = list({json.dumps(w, sort_keys=True, default=str): w for w in ws}.values())
    records = []
    for s in subjects:
        tier_own, ev = own[s.subject_id]
        rec: Dict[str, Any] = {
            "row_id": s.row_id, "side_key": s.side_key, "source": s.source, "document_group": s.document_group,
            "script": measure_script(s.text or ""), "letters": textnorm.letters(s.text or ""),
            "ja_signal": bool(textnorm.ja_signal(s.text or "")), "photos": [p.key for p in s.photos],
            "own_photos": list(s.own_photos), "own_basis": s.own_basis, "own_tier": tier_own}
        rec.update(s.meta)
        rec.update(novelty_record(s, ev, groups[s.subject_id]))
        rec.update(holdout_record(group_ws[s.document_group or s.subject_id], text_ws[s.subject_id]))
        rec["released"] = rec["released"] + cap(released_val_witnesses(ev))
        rec["eval_text_nearest"] = nearest[s.subject_id]
        records.append(rec)
    return RunResult(records)


# ===================================================================================================================
# Inputs and the CLI
# ===================================================================================================================

def load_join_partners(path: Path = JOIN_PARTNERS) -> Dict[Tuple[str, str], List[str]]:
    """Verified join partners per ``(source, row_id)`` (``join_axis.py``'s overlay; empty when absent).

    :param path: Overlay file.
    :type path: Path
    :return: The partners.
    :rtype: Dict[Tuple[str, str], List[str]]
    """
    out: Dict[Tuple[str, str], List[str]] = {}
    if path.is_file():
        for o in iter_jsonl(path):
            if o.get("verified") and o.get("partners"):
                out[(str(o["source"]), str(o["row_id"]))] = [str(p) for p in o["partners"]]
    return out


def load_placements(path: Path) -> Dict[Tuple[str, str], str]:
    """Accepted stage-2 placements: ``{(row_id, side_key): sha256}``.

    :param path: ``stage2_sides.jsonl``.
    :type path: Path
    :return: The placements.
    :rtype: Dict[Tuple[str, str], str]
    """
    out: Dict[Tuple[str, str], str] = {}
    for r in iter_jsonl(path):
        if r.get("outcome") == "accepted" and r.get("image_sha256"):
            out[(str(r["row_id"]), str(r["side_key"]))] = str(r["image_sha256"])
    return out


def load_stage1(path: Path) -> List[Candidate]:
    """The stage-1 candidates (``schema.Candidate``).

    :param path: ``stage1_images.jsonl``.
    :type path: Path
    :return: Candidates.
    :rtype: List[Candidate]
    """
    return [from_dict(Candidate, r) for r in iter_jsonl(path)]


def ledger_texts(candidates_dir: Path, wanted: Mapping[str, Set[str]]) -> Dict[Tuple[str, str], str]:
    """Edition texts of ledger candidates (``candidates/<source>.jsonl`` field ``edition_text``).

    :param candidates_dir: Candidate directory.
    :type candidates_dir: Path
    :param wanted: ``{source file: {row_id}}``.
    :type wanted: Mapping[str, Set[str]]
    :return: ``{(source file, row_id): text}``.
    :rtype: Dict[Tuple[str, str], str]
    """
    out: Dict[Tuple[str, str], str] = {}
    for src, ids in wanted.items():
        path = candidates_dir / f"{src}.jsonl"
        if ids and path.is_file():
            for r in iter_jsonl(path):
                rid = str(r.get("row_id"))
                if rid in ids:
                    out[(src, rid)] = str(r.get("edition_text") or "")
    return out


def summarize(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Counts of a run: tiers, hold-outs and their families, by source and script.

    :param records: Records.
    :type records: Sequence[Mapping[str, Any]]
    :return: Counts (JSON-ready).
    :rtype: Dict[str, Any]
    """
    by_source = Counter((r["source"], r["script"], r["tier"]) for r in records)
    return {"records": len(records), "tiers": dict(Counter(r["tier"] for r in records)),
            "eval_holdout": sum(bool(r["eval_holdout"]) for r in records),
            "eval_holdout_by_tier": dict(Counter(r["tier"] for r in records if r["eval_holdout"])),
            "eval_families": dict(Counter(f for r in records for f in r["eval_families"])),
            "released_sets": dict(Counter(s for r in records for s in r.get("released_sets") or [])),
            "by_source_script_tier": {"|".join(k): v for k, v in sorted(by_source.items())}}


def build_parser() -> argparse.ArgumentParser:
    """The command line.

    :return: The parser.
    :rtype: argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(description="Novelty tiers and evaluation hold-outs v2 (GATE_V2_SPEC.md section 7).")
    p.add_argument("--w1", type=Path, default=W1, help="frozen novel_v1 work dir (read only)")
    p.add_argument("--w2", type=Path, default=W2, help="ja_ar_v1 work dir (outputs under novelty/, caches under cache/)")
    p.add_argument("--seen-dir", type=Path, default=SEEN_DIR, help="seen registry directory")
    p.add_argument("--ledger", type=Path, default=LEDGER, help="unseen_pool.jsonl")
    p.add_argument("--candidates-dir", type=Path, default=CANDIDATES_DIR, help="ledger candidate files (edition text)")
    p.add_argument("--registered-dir", type=Path, default=None, help="new eval sets (default <w2>/eval/registered)")
    p.add_argument("--verdict", default=UNCERTAIN, help="ledger verdict re-run beside W1 (default uncertain_history)")
    p.add_argument("--only", choices=("w1", "uncertain", "both"), default="both", help="which verdict files to write")
    p.add_argument("--no-index", action="store_true", help="skip the merged-corpus index (identity as recorded)")
    p.add_argument("--limit", type=int, default=None, help="judge at most this many W1 candidates (smoke runs)")
    p.add_argument("--pool-val-holdout", action="store_true", help="val rows of pool datasets are hold-outs too")
    p.add_argument("--share-basis", choices=SHARE_BASES, default="shingles", help="share clause of the text rule")
    p.add_argument("--shingle-letters", type=int, default=SHINGLE_LETTERS, help="shingle length of the share clause")
    p.add_argument("--long-run-letters", type=int, default=LONG_RUN_LETTERS, help="long-run length (letters)")
    p.add_argument("--share-threshold", type=float, default=SHARE_THRESHOLD, help="share threshold")
    p.add_argument("--out-suffix", default="", help="suffix of the output file names (comparison runs)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Judge the W1 target sides and the ledger's uncertain_history candidates; write the verdict files.

    :param argv: Command line (None: ``sys.argv``).
    :type argv: Optional[List[str]]
    :return: Exit status.
    :rtype: int
    """
    args = build_parser().parse_args(argv)
    t0 = time.time()
    rule = TextRule(args.long_run_letters, args.share_threshold, args.share_basis, args.shingle_letters)
    out_dir, cache_dir, sfx = args.w2 / "novelty", args.w2 / "cache", args.out_suffix
    partners = load_join_partners()
    cands = load_stage1(args.w1 / "stages" / "stage1_images.jsonl")
    if args.limit is not None:
        cands = cands[:args.limit]
    placed = load_placements(args.w1 / "stages" / "stage2_sides.jsonl")
    w1_subjects = [s for c in cands for s in subjects_from_candidate(c, placed, partners)]
    w1_ids = {c.row_id for c in cands}
    ledger_rows = list(iter_jsonl(args.ledger))
    unc_rows = ([r for r in ledger_rows if (r.get("flags") or {}).get("verdict") == args.verdict
                 and str(r.get("row_id")) not in w1_ids] if args.only != "w1" else [])
    wanted: Dict[str, Set[str]] = defaultdict(set)
    for r in unc_rows:
        wanted[str(r.get("source_file"))].add(str(r.get("row_id")))
    texts = ledger_texts(args.candidates_dir, wanted)
    unc_subjects = [subject_from_ledger(r, texts.get((str(r.get("source_file")), str(r.get("row_id"))), ""),
                                        partners) for r in unc_rows]
    subjects = w1_subjects + unc_subjects
    judged = {s.row_id for s in subjects}
    groups = {s.document_group for s in subjects if s.document_group}
    ledger_members: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for r in ledger_rows:
        if r.get("document_group") in groups and str(r.get("row_id")) not in judged:
            ledger_members[str(r["document_group"])].append(r)
    del ledger_rows
    print(f"{len(w1_subjects):,} W1 sides, {len(unc_subjects):,} {args.verdict} candidates ({len(texts):,} with "
          f"text), {sum(len(v) for v in ledger_members.values()):,} other ledger members ({time.time() - t0:.0f} s)",
          file=sys.stderr)
    vocab = np.unique(np.concatenate([registry_positions(s.text) for s in subjects] + [np.zeros(0, np.uint64)]))
    _, released_pgpids = br.registered_benchmark_documents(["held_out_edition_pages_v0"])
    confirmer = PhashConfirmer(cache_dir / "phash256.json")
    seen = load_seen(args.seen_dir, REKEY_SEEN_DIR, vocab, set(released_pgpids), args.pool_val_holdout, confirmer)
    index = None
    if not args.no_index:
        index = shelf.ShelfmarkIndex(shelf.MERGED_SHELFMARKS, fragments=nh.pgp_fragments(lsr.PGP_EXPORT),
                                     cache_dir=cache_dir / "shelfmark_index")
    sets = load_eval_sets(index, registered_dir=args.registered_dir or args.w2 / "eval" / "registered",
                          image_cache=cache_dir / "eval_image_keys.json")
    evals = EvalIndex(sets, rule, confirmer)
    print("eval sets: " + ", ".join(f"{s.name}({len(s.gt)} gt, {len(s.images)} img)" for s in sets)
          + f" ({time.time() - t0:.0f} s)", file=sys.stderr)
    result = judge_run(subjects, seen, evals, index, rule, ledger_members)
    confirmer.save()
    w1_ids_side = {s.subject_id for s in w1_subjects}
    w1_records = [r for r in result.records if f"{r['row_id']}|{r['side_key']}" in w1_ids_side]
    unc_records = [r for r in result.records if f"{r['row_id']}|{r['side_key']}" not in w1_ids_side]
    old = {str(r["row_id"]): r for r in unc_rows}
    for r in unc_records:
        o = old.get(r["row_id"]) or {}
        f = o.get("flags") or {}
        r["old_verdict"] = f.get("verdict")
        r["old_evidence"] = {k: listify(f.get(k))[:4] for k in ("image", "image_uncertain", "document",
                                                                "document_uncertain")}
        r["old_text_best"] = {"definite": f.get("text_best_definite"), "uncertain": f.get("text_best_uncertain")}
        r["has_text"] = bool(texts.get((str(o.get("source_file")), r["row_id"])))
    summary: Dict[str, Any] = {
        "rule": asdict(rule), "phash_max_bits": PHASH_MAX_BITS, "pool_val_holdout": args.pool_val_holdout,
        "registry_rows": {f"{d}|{c}": n for (d, c), n in sorted(seen.rows.items())},
        "registry_dropped": dict(seen.dropped), "registry_text_pages": len(seen.text),
        "eval_sets": [{"name": s.name, "family": s.family, "released": s.released, "gt": len(s.gt),
                       "images": len(s.images), "ids": len(s.bench.ids), "pgpids": len(s.bench.pgpids),
                       "sys_nums": len(s.bench.sys_nums), "keys": len(s.bench.keys), "fl": len(s.fl),
                       "names": len(s.names)} for s in sets]}
    if args.only in ("w1", "both"):
        write_jsonl(out_dir / f"verdicts_w1{sfx}.jsonl", w1_records)
        summary["w1"] = summarize(w1_records)
    if args.only in ("uncertain", "both"):
        write_jsonl(out_dir / f"verdicts_uncertain{sfx}.jsonl", unc_records)
        summary["uncertain"] = summarize(unc_records)
        summary["uncertain"]["old_to_new"] = dict(Counter(f"{r['old_verdict']}->{r['tier']}" for r in unc_records))
    summary["phash_confirmation"] = {"phash256_max_bits": confirmer.max_bits, "verdicts": dict(confirmer.stats),
                                     "eval_unconfirmed": len(evals.unconfirmed),
                                     "eval_unconfirmed_sample": evals.unconfirmed[:20]}
    summary["seconds"] = round(time.time() - t0)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"summary{sfx}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: (summary.get(k) or {}).get("tiers") for k in ("w1", "uncertain")}, ensure_ascii=False),
          file=sys.stderr)
    print(f"wrote {out_dir} ({summary['seconds']} s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
