# File name: unreachable_profile.py
# Date: 10/7/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Where the readers' unreachable text sits, and whether two scholars disagree there too (offline).

``slot_oracle.py`` splits ROVER's wrong characters into *learnable* (some
reader wrote the GT character at that slot) and *unreachable* (no reader did).
This module profiles the unreachable characters of one reader set on one
benchmark from the same char networks (``slot_oracle.iter_pages``: same
normalisation, alignment and vote; nothing is re-aligned and no model is run):

* **damage proxy**: distance in the raw GT from every character to the nearest
  editorial mark: lacuna / restoration brackets and dot runs, parenthesised
  lacunae and Weiss-format restorations (``lacuna``), ``?`` (``uncertain``);
  sic marks (``sic``) and other parenthesised notes (``paren``) are kept apart
  because they do not mark damage;
* **page level**: the unreachable share of a page against the spec's
  ``recon_frac`` and its gap density, and how the unreachable text is spread
  over pages (Gini, top-10 share);
* **line position**: reference lines (religious-140), or, for PGP-131 whose
  references are flattened to one line, the lines of the PGP edition the GT
  matches, with the margin and address regions the edition labels;
* **blocks** (``slot_oracle.page_blocks``): what each reader did over the
  block (dropped it, wrote it elsewhere, wrote other text of the page, looped,
  misread) and whether any reader, or a non-voting extra reader, wrote the
  block's text anywhere in its output;
* **scholar dissent**: for documents whose PGP export holds editions from two
  different sources (``two_edition_documents``), the two editions are put in a
  char network with the ROVER tool's own DP and the GT is merged in as
  ``slot_oracle.add_reference`` merges it; every GT character is labelled
  ``agree`` / ``dissent`` / ``passage`` / ``gt_differs`` / ``gt_only`` and
  the readers' unreachable characters are compared with that labelling.
  Every two-edition pair is also profiled the same way (difference runs and
  their block share).

A stratified sample of blocks is drawn for human review.  Outputs (small JSON
files): ``blocks_<tag>.jsonl`` (the ``--dump-blocks`` records plus the
analysis fields), ``profile.json`` and ``sample.json``; the report is built by
``logs/next_round/consensus/build_unreachable_sample_report.py``.

Usage (from the repo root):
    .venv/bin/python src/datasets/evaluations/helper_eval_scripts/unreachable_profile.py
"""

import argparse
import bisect
import json
import random
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Collection, Dict, List, Optional, Sequence, Tuple

from rapidfuzz import fuzz

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.datasets.cleaning.clean_genizah_transcriptions import (  # noqa: E402
    DOTS_RUN_RE,
    EMPTY_PARENS_RE,
    GAP,
    LOSS_NOTE_RE,
    PARENTHESES_RE,
    SEMITIC_RE,
    clean_diplomatic,
)
from src.datasets.evaluations import arabic_script as ar  # noqa: E402
from src.datasets.evaluations import two_edition_documents as ted  # noqa: E402
from src.datasets.evaluations.helper_eval_scripts import rover_consensus as rc  # noqa: E402
from src.datasets.evaluations.helper_eval_scripts import slot_oracle as so  # noqa: E402
from src.datasets.evaluations.helper_eval_scripts.build_religious_benchmark import IIIF  # noqa: E402
from src.datasets.evaluations.metrics import (  # noqa: E402
    dedup_repeated_tail,
    genizah_visible_ink_gt,
)

DAMAGE_WINDOW = 15                     # raw GT characters: "near a mark"
MARK_KINDS = ("lacuna", "uncertain", "paren", "sic")
DISTANCE_GROUPS = {"damage": ("lacuna", "uncertain"), "lacuna": ("lacuna",),
                   "uncertain": ("uncertain",), "paren": ("paren",), "sic": ("sic",),
                   "any": MARK_KINDS}
NGRAM = 4                              # letter n-grams for "did the reader write this anywhere"
ELSEWHERE_SHARE = 0.5                  # share of a text's n-grams found in another text
NOWHERE_SHARE = 0.2                    # below this for every reader: nobody wrote the text
SHIFT_SIMILARITY = 60.0                # fuzz.ratio of span vs GT letters: the reader wrote it here
DROP_SHARE = 0.25                      # span letters <= this share of the GT's: dropped
LOOP_SHARE = 0.5                       # span n-grams repeated in the reader's output: looped
ONE_SIDED_MIN = 21                     # a passage one edition lacks: more than 20 slots
SHORT_READ = 0.6                       # most prolific reader wrote < 60 % of the GT's letters
CLEAN_PAIR = 0.9                       # GT characters an edition matches: the GT is that edition
SAME_COVERAGE = 0.85                   # ... and the other edition matches this much: same text covered
GT_BEYOND_EDITION = 0.8                # best PGP edition matches less of the GT: the GT holds other text
DISSENT_DILATION = 3                   # slots on each side for "near a dissent"
LENGTH_CLASSES = ((8, 20, "8-20"), (21, 60, "21-60"), (61, 10 ** 9, ">60"))
SAMPLE = {"verified": (23, 7), "religious": (12, 3)}   # (far from damage, near damage) blocks
SAMPLE_PER_PAGE = 2
SAMPLE_SEED = 20261007
_CHAR_GAP = int(round(rc.DEFAULT_GAP["char"] * rc.COST_SCALE))
_LETTER = ar.SCRIPT_LETTER
_STRAY_BRACKET = re.compile(r"[\[\]]")
_BRACKET_SPAN = re.compile(r"\[[^\[\]\n]*\]")
_SIC_INNER = re.compile(r"^[ \t]*[!?]*![!?]*[ \t]*$")
_UNCERTAIN_INNER = re.compile(r"^[ \t]*\?+[ \t]*$")
_SIDE = re.compile(r"^\s*(recto|verso)\b", re.IGNORECASE)
_LATIN = re.compile(r"[A-Za-z]")
_APPARATUS = re.compile(r"[0-9A-Za-z]")       # digits / Latin in a GT block: footnote or line numbers
PGP_DOC_URL = "https://geniza.princeton.edu/en/documents/{pgpid}/"
KTIV_ITEM_URL = ("https://www.nli.org.il/en/discover/manuscripts/hebrew-manuscripts/itempage"
                 "?vid=KTIV&scope=KTIV&docId=PNX_MANUSCRIPTS{sys_num}")
P = "qwen3_vl_8b_heb_"
KSEG = "kraken_seg_k7_ktivseg_e10_rgb_mv"
V6 = tuple(P + m for m in ("v21b_step1200", "v22a_step1300", "v22b_step1200", "v22i_a50",
                           "v23a_step1200", "v23a_step1500"))
_DEFAULT_OUT = _REPO / "logs/next_round/consensus/unreachable"


@dataclass(frozen=True)
class Config:
    """One reader set on one benchmark.

    :param tag: Output stem.
    :param benchmark: A key of ``rover_consensus.SOURCES``.
    :param models: Voting readers (ROVER order).
    :param extra: Non-voting readers, used only to ask whether a block's text
        was written anywhere.
    """

    tag: str
    benchmark: str
    models: Tuple[str, ...]
    extra: Tuple[str, ...] = ()


CONFIGS = (Config("pgp131_6vlm", "verified", V6, (KSEG,)),
           Config("religious_4vlm_kseg", rc.RELIGIOUS, V6[:4] + (KSEG,)))


# ── Editorial marks in the raw GT ─────────────────────────────────────────────

def damage_marks(raw: str) -> List[Tuple[int, int, str]]:
    """Editorial marks of a raw GT as half-open character spans.

    Kinds: ``lacuna`` (``[...]`` and ``[text]`` brackets, stray edge brackets,
    dot runs, parenthesised dots / loss notes / empty pairs, and in a
    Weiss-format text (one with an empty ``( )``) parenthesised letters, which
    are restorations there), ``uncertain`` (``?`` and ``(?)``), ``sic``
    (``(!)`` and a bare ``!``) and ``paren`` (any other parenthesised span:
    expansions and glosses).  A ``?``, ``!``, bracket or dot run inside a
    parenthesised or bracketed span belongs to that span.

    :param raw: Raw GT (``slot_oracle.alignment_raw``).
    :type raw: str
    :return: ``(start, end, kind)`` spans, sorted.
    :rtype: List[Tuple[int, int, str]]
    """
    marks: List[Tuple[int, int, str]] = []
    covered = bytearray(len(raw))
    weiss = bool(EMPTY_PARENS_RE.search(raw))

    def add(start: int, end: int, kind: str) -> None:
        marks.append((start, end, kind))
        covered[start:end] = b"\x01" * (end - start)

    for m in PARENTHESES_RE.finditer(raw):
        inner = m.group(1)
        if _SIC_INNER.match(inner):
            kind = "sic"
        elif _UNCERTAIN_INNER.match(inner):
            kind = "uncertain"
        elif (not inner.strip() or DOTS_RUN_RE.fullmatch(inner.strip()) or LOSS_NOTE_RE.search(inner)
              or (weiss and SEMITIC_RE.search(inner))):
            kind = "lacuna"
        else:
            kind = "paren"
        add(m.start(), m.end(), kind)
    for m in _BRACKET_SPAN.finditer(raw):
        if not covered[m.start()]:
            add(m.start(), m.end(), "lacuna")
    for pattern, kind in ((_STRAY_BRACKET, "lacuna"), (DOTS_RUN_RE, "lacuna"),
                          (re.compile(r"\?"), "uncertain"), (re.compile(r"!"), "sic")):
        for m in pattern.finditer(raw):
            if not covered[m.start()]:
                add(m.start(), m.end(), kind)
    return sorted(marks)


def mark_distance(n: int, marks: Sequence[Tuple[int, int, str]], kinds: Collection[str]) -> List[int]:
    """Distance from every raw position to the nearest mark of some kinds.

    :param n: Length of the raw GT.
    :type n: int
    :param marks: Output of :func:`damage_marks`.
    :type marks: Sequence[Tuple[int, int, str]]
    :param kinds: Mark kinds that count.
    :type kinds: Collection[str]
    :return: One distance per position: 0 inside a mark, ``n + 1`` when the
        text has no mark of those kinds.
    :rtype: List[int]
    """
    far = n + 1
    dist = [far] * n
    for start, end, kind in marks:
        if kind in kinds:
            dist[start:min(end, n)] = [0] * (min(end, n) - start)
    for p in range(1, n):
        dist[p] = min(dist[p], dist[p - 1] + 1)
    for p in range(n - 2, -1, -1):
        dist[p] = min(dist[p], dist[p + 1] + 1)
    return dist


def gap_count(raw: str) -> int:
    """Gap tokens the visible-ink cleaning leaves in a raw GT.

    :param raw: Raw GT.
    :type raw: str
    :return: Number of lacuna / restoration gaps after merging neighbours.
    :rtype: int
    """
    return clean_diplomatic(dedup_repeated_tail(raw))[0].count(GAP)


# ── Letters, n-grams and what a reader did ────────────────────────────────────

def letters(text: str) -> str:
    """Hebrew and Arabic letters of a text, in order.

    :param text: Any text.
    :type text: str
    :return: Letters only.
    :rtype: str
    """
    return "".join(_LETTER.findall(text))


def grams(text: str, n: int = NGRAM) -> Counter:
    """Letter n-grams of a letters-only string, with counts.

    :param text: Letters-only string.
    :type text: str
    :param n: N-gram length.
    :type n: int
    :return: ``Counter`` of n-grams.
    :rtype: Counter
    """
    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


def gram_share(text: str, pool: Collection[str], n: int = NGRAM) -> float:
    """Share of a letters-only string's distinct n-grams present in a pool.

    :param text: Letters-only string.
    :type text: str
    :param pool: N-grams of the other text.
    :type pool: Collection[str]
    :param n: N-gram length.
    :type n: int
    :return: Share in [0, 1]; 0.0 when ``text`` is shorter than ``n``.
    :rtype: float
    """
    own = set(grams(text, n))
    return len(own & set(pool)) / len(own) if own else 0.0


BEHAVIOURS = ("shifted", "dropped, wrote it elsewhere", "dropped", "other page text",
              "wrote it elsewhere", "looped", "misread")


def reader_behaviour(gt_block: str, span: str, full: str, page_rest: str) -> str:
    """What one reader did over an unreachable block.

    Tests in order: ``shifted`` (the span's letters are at least
    ``SHIFT_SIMILARITY`` similar to the GT's: the reader wrote the text here
    but the slot alignment put it beside the GT entries); ``dropped`` (at most
    ``DROP_SHARE`` as many letters), qualified ``wrote it elsewhere`` when at
    least ``ELSEWHERE_SHARE`` of the block's n-grams are elsewhere in the
    reader's output; ``other page text`` (the span is mostly text of the GT
    outside the block: a neighbouring line, reading order, a repeat);
    ``wrote it elsewhere``; ``looped`` (the span repeats the reader's own
    output); else ``misread``.

    :param gt_block: GT text of the block.
    :type gt_block: str
    :param span: What the reader wrote over the block.
    :type span: str
    :param full: The reader's whole normalised output.
    :type full: str
    :param page_rest: The GT outside the block.
    :type page_rest: str
    :return: One of :data:`BEHAVIOURS`.
    :rtype: str
    """
    g, r, f = letters(gt_block), letters(span), letters(full)
    if r and fuzz.ratio(r, g) >= SHIFT_SIMILARITY:
        return "shifted"
    elsewhere = gram_share(g, grams(f)) >= ELSEWHERE_SHARE
    if len(r) <= DROP_SHARE * len(g):
        return "dropped, wrote it elsewhere" if elsewhere else "dropped"
    if gram_share(r, grams(letters(page_rest))) >= ELSEWHERE_SHARE:
        return "other page text"
    if elsewhere:
        return "wrote it elsewhere"
    counts = grams(f)
    own = set(grams(r))
    if own and sum(counts[x] >= 2 for x in own) >= LOOP_SHARE * len(own):
        return "looped"
    return "misread"


def block_note(behaviour: Dict[str, str], readers: Dict[str, str], gt_block: str,
               short: Callable[[str], str]) -> str:
    """One line on what the readers did over a block.

    :param behaviour: ``{model: behaviour}``.
    :type behaviour: Dict[str, str]
    :param readers: ``{model: span text}``.
    :type readers: Dict[str, str]
    :param gt_block: GT text of the block.
    :type gt_block: str
    :param short: Display name of a model.
    :type short: Callable[[str], str]
    :return: Counts per behaviour (most common first), the readers that differ
        from the majority, and a shared alternative reading when three or more
        readers wrote the same letters.
    :rtype: str
    """
    n = len(behaviour)
    counts = Counter(behaviour.values())
    top, top_n = counts.most_common(1)[0]
    parts = [f"{top_n}/{n} {top}"]
    if top_n < n:
        parts.append(", ".join(f"{short(m)} {b}" for m, b in behaviour.items() if b != top))
    same = Counter(letters(t) for t in readers.values() if letters(t))
    if same:
        text, k = same.most_common(1)[0]
        if k >= 3 and text != letters(gt_block) and len(text) >= 3:
            parts.append(f"{k} readers wrote the same letters «{text[:24]}»")
    return "; ".join(parts)


# ── PGP edition lines (PGP-131 references are one line) ───────────────────────

@dataclass(frozen=True)
class EditionLine:
    """One text line of a PGP edition.

    :param side: ``""``, ``recto`` or ``verso`` (from the edition's labels).
    :param region: ``main``, ``margin`` or ``address`` (from its labels).
    :param raw: The line as stored in ``footnotes.csv``.
    :param ink: Its visible ink (the benchmark's cleaning).
    """

    side: str
    region: str
    raw: str
    ink: str


def edition_lines(content: str) -> List[EditionLine]:
    """Text lines of a PGP edition with their side and region.

    Lines are kept and cleaned as ``arabic_script.split_sections`` keeps and
    cleans them, then reduced to visible ink with ``genizah_visible_ink_gt``
    (``two_edition_documents.edition_visible_text`` per line).  A label line
    (no Hebrew or Arabic letter) starting with Recto / Verso opens a side; a
    label with Latin letters sets the region: ``margin`` or ``address`` when
    it says so, ``main`` for a side label or "main", else unchanged.

    :param content: ``content`` cell of a ``footnotes.csv`` edition row.
    :type content: str
    :return: Text lines in edition order.
    :rtype: List[EditionLine]
    """
    side, region, out = "", "main", []
    for raw in content.splitlines():
        if not ar.SCRIPT_LETTER.search(unicodedata.normalize("NFKC", raw)):
            label = raw.strip().lower()
            opens = _SIDE.match(raw)
            if opens:
                side, region = opens.group(1).lower(), "main"
            if _LATIN.search(label):
                region = ("margin" if "margin" in label else "address" if "address" in label
                          else "main" if opens or "main" in label else region)
            continue
        cleaned = ar.clean_edition_line(raw)
        if not ar.SCRIPT_LETTER.search(cleaned):
            continue
        ink = unicodedata.normalize("NFC", genizah_visible_ink_gt(cleaned))
        if ink:
            out.append(EditionLine(side, region, raw.strip(), ink))
    return out


def line_classes(lines: Sequence[EditionLine]) -> List[str]:
    """Position class of every edition line.

    :param lines: Output of :func:`edition_lines`.
    :type lines: Sequence[EditionLine]
    :return: ``margin`` / ``address`` for those regions; for main-text lines of
        a side, ``first`` / ``last`` / ``middle`` (``short`` when the side has
        fewer than three main lines).
    :rtype: List[str]
    """
    out = [line.region for line in lines]
    by_side: Dict[str, List[int]] = {}
    for i, line in enumerate(lines):
        if line.region == "main":
            by_side.setdefault(line.side, []).append(i)
    for idx in by_side.values():
        for rank, i in enumerate(idx):
            out[i] = ("short" if len(idx) < 3 else "first" if rank == 0
                      else "last" if rank == len(idx) - 1 else "middle")
    return out


def edition_char_lines(gt_ink: str, lines: Sequence[EditionLine]) -> Tuple[List[Optional[int]], float]:
    """Edition line of every GT character (indel alignment, lines joined by spaces).

    :param gt_ink: One-line scoring reference.
    :type gt_ink: str
    :param lines: Edition lines.
    :type lines: Sequence[EditionLine]
    :return: (line index or ``None`` per GT character, share matched).
    :rtype: Tuple[List[Optional[int]], float]
    """
    text = " ".join(line.ink for line in lines)
    starts, offset = [], 0
    for line in lines:
        starts.append(offset)
        offset += len(line.ink) + 1
    matched = so.aligned_positions(gt_ink, text)
    out = [None if p is None else bisect.bisect_right(starts, p) - 1 for p in matched]
    return out, (sum(p is not None for p in matched) / len(gt_ink) if gt_ink else 0.0)


def best_edition(gt_ink: str, editions: Dict[str, str]) -> Tuple[Optional[str], List[EditionLine], List[Optional[int]], float]:
    """The PGP edition that matches the GT best.

    :param gt_ink: One-line scoring reference.
    :type gt_ink: str
    :param editions: ``{source: content}`` of one document.
    :type editions: Dict[str, str]
    :return: (source, its lines, line per GT character, share of GT characters matched);
        ``(None, [], [None...], 0.0)`` when no edition has a text line.
    :rtype: Tuple[Optional[str], List[EditionLine], List[Optional[int]], float]
    """
    best: Tuple[Optional[str], List[EditionLine], List[Optional[int]], float] = (
        None, [], [None] * len(gt_ink), 0.0)
    for source, content in sorted(editions.items()):
        lines = edition_lines(content)
        if not lines:
            continue
        per_char, coverage = edition_char_lines(gt_ink, lines)
        if coverage > best[3]:
            best = (source, lines, per_char, coverage)
    return best


# ── Scholar dissent ───────────────────────────────────────────────────────────

DISSENT_LABELS = ("agree", "dissent", "passage", "gt_differs", "gt_only")


@dataclass(frozen=True)
class EditionSlot:
    """What the two editions do at the slot of one GT character.

    :param label: ``agree`` (both editions have the GT character), ``dissent``
        (they differ at this slot: a word-level disagreement), ``passage``
        (inside a stretch of more than 20 slots that one edition lacks:
        coverage, an omitted or relocated line; see :func:`edition_kinds`),
        ``gt_differs`` (the editions agree, the GT has something else),
        ``gt_only`` (neither edition has anything here).
    :param follows: ``a`` / ``b`` when the GT character equals exactly that
        edition's entry, ``both``, or ``""``.
    :param other: Where the GT follows exactly one edition: the other
        edition's entry key (``None`` = it has nothing here).
    :param near_dissent: A ``dissent`` slot lies within ``DISSENT_DILATION``
        slots (GT-NULL slots included).
    """

    label: str
    follows: str
    other: Optional[str]
    near_dissent: bool


def edition_kinds(keys: Sequence[Tuple[Optional[str], Optional[str]]]) -> List[str]:
    """Kind of every slot of a two-edition network.

    A slot where exactly one edition is NULL is *lopsided*.  Stretches of
    lopsided slots found with the readers' block rule
    (``slot_oracle.block_spans`` with the char window, extended, at least
    ``ONE_SIDED_MIN`` slots) are passages one edition lacks: every slot inside
    is ``passage``, including chance matches of the shorter edition's stray
    letters.  Any other slot where the two keys differ is ``dissent``
    (word-level disagreement); the rest are ``agree``.

    :param keys: ``(edition A key, edition B key)`` per slot.
    :type keys: Sequence[Tuple[Optional[str], Optional[str]]]
    :return: ``agree``, ``dissent`` or ``passage`` per slot.
    :rtype: List[str]
    """
    kind = ["agree" if ka == kb else "dissent" for ka, kb in keys]
    lopsided = [(ka is None) != (kb is None) for ka, kb in keys]
    for start, end in so.block_spans(lopsided, so.BLOCK_WINDOW["char"], min_len=ONE_SIDED_MIN,
                                     extend=True):
        kind[start:end] = ["passage"] * (end - start)
    return kind


def edition_network(a: str, b: str) -> rc.Network:
    """Char network of two editions (A is the backbone).

    :param a: Visible text of edition A.
    :type a: str
    :param b: Visible text of edition B.
    :type b: str
    :return: Network with columns A, B.
    :rtype: rc.Network
    """
    return rc.build_network([list(a), list(b)], [0, 1], _CHAR_GAP)


def edition_slots(gt_tokens: Sequence[str], a: str, b: str) -> List[EditionSlot]:
    """Label every GT character by what two editions do at its slot.

    :param gt_tokens: GT character tokens (``tokenize(lines, "char")``).
    :type gt_tokens: Sequence[str]
    :param a: Visible text of edition A.
    :type a: str
    :param b: Visible text of edition B.
    :type b: str
    :return: One :class:`EditionSlot` per GT character, in order.
    :rtype: List[EditionSlot]
    """
    network = edition_network(a, b)
    path = rc.align_to_network(network, list(gt_tokens), _CHAR_GAP)
    merged = rc.merge_into_network(network, list(gt_tokens), path, 2)
    keys = [(so.token_key(s[0], "char"), so.token_key(s[1], "char")) for s in merged.slots]
    kind = edition_kinds(keys)
    dissent_prefix = [0]
    for k in kind:
        dissent_prefix.append(dissent_prefix[-1] + (k == "dissent"))
    out = []
    for i, slot in enumerate(merged.slots):
        g = so.token_key(slot[2], "char")
        if g is None:
            continue
        ka, kb = keys[i]
        if ka is None and kb is None:
            label = "gt_only"
        elif kind[i] != "agree":
            label = kind[i]
        else:
            label = "agree" if g == ka else "gt_differs"
        follows = ("both" if g == ka == kb else "a" if g == ka else "b" if g == kb else "")
        other = kb if follows == "a" else ka if follows == "b" else None
        lo, hi = max(0, i - DISSENT_DILATION), min(len(kind), i + DISSENT_DILATION + 1)
        out.append(EditionSlot(label, follows, other, dissent_prefix[hi] - dissent_prefix[lo] > 0))
    return out


def pair_profile(a: str, b: str) -> Dict[str, object]:
    """How two editions of one document differ, in the readers' block terms.

    :param a: Visible text of edition A (the longer one).
    :type a: str
    :param b: Visible text of edition B.
    :type b: str
    :return: ``slots``, ``differ_slots``, ``passage_slots``, the lengths of
        the differing runs (``runs``), ``a_chars``, ``a_differ`` (A characters
        at differing slots), ``a_block`` (of those, the ones in a block by
        ``slot_oracle.block_mask`` with the char window), ``a_block_dissent``
        (the same with one-sided runs counted as agreement: reading-level
        differences only) and ``subs`` (letter-for-letter substitutions as
        sorted letter pairs).
    :rtype: Dict[str, object]
    """
    network = edition_network(a, b)
    keys = [(so.token_key(s[0], "char"), so.token_key(s[1], "char")) for s in network.slots]
    kind = edition_kinds(keys)
    runs, i = [], 0
    while i < len(kind):
        if kind[i] == "agree":
            i += 1
            continue
        j = i
        while j < len(kind) and kind[j] != "agree":
            j += 1
        runs.append(j - i)
        i = j
    on_a = [(k, s) for k, s in zip(kind, network.slots) if s[0] is not None]
    a_flags = [k != "agree" for k, _ in on_a]
    a_dissent = [k == "dissent" for k, _ in on_a]
    subs = Counter("".join(sorted((ka, kb))) for ka, kb in keys
                   if ka and kb and ka != kb and _LETTER.match(ka) and _LETTER.match(kb))
    return {"slots": len(kind), "differ_slots": sum(k != "agree" for k in kind),
            "passage_slots": kind.count("passage"), "runs": runs, "a_chars": len(a_flags),
            "a_differ": sum(a_flags), "a_dissent": sum(a_dissent),
            "a_block": sum(so.block_mask(a_flags, so.BLOCK_WINDOW["char"])),
            "a_block_dissent": sum(so.block_mask(a_dissent, so.BLOCK_WINDOW["char"])),
            "subs": subs}


def edition_pair(per_source: Dict[str, str]) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    """The two longest editions from different sources, as ``two_edition_documents`` pairs them.

    :param per_source: ``{source: content}`` of one document (two or more sources).
    :type per_source: Dict[str, str]
    :return: ``((source A, visible text A), (source B, visible text B))``, A the longer content.
    :rtype: Tuple[Tuple[str, str], Tuple[str, str]]
    """
    (sa, ca), (sb, cb) = sorted(per_source.items(), key=lambda kv: -len(kv[1]))[:2]
    return (sa, ted.edition_visible_text(ca)), (sb, ted.edition_visible_text(cb))


# ── Per page ──────────────────────────────────────────────────────────────────

@dataclass
class PageProfile:
    """Per-character view of one page for one reader set.

    :param doc_id: Benchmark document id.
    :param bucket: Script bucket (PGP) or column class (religious).
    :param meta: ``slot_oracle.page_metadata`` fields.
    :param chars: The page's char network (``slot_oracle.PageChars``).
    :param cls: ROVER class per GT character.
    :param in_block: ``slot_oracle.block_mask`` per GT character (the slot-oracle report's blocks).
    :param block_of: Index of the dumped block holding the character, ``-1`` if none.
    :param dist: Raw-GT distance to the nearest mark per :data:`DISTANCE_GROUPS` group.
    :param line: Reference line index per GT character.
    :param n_lines: Reference lines.
    :param blocks: ``slot_oracle.page_blocks`` records (annotated later).
    :param texts: Normalised one-line output per reader (voting and extra).
    :param gaps: Gap tokens of the cleaned GT.
    :param read_ratio: Letters of the most prolific voting reader over GT letters.
    :param line_class: Line-position class per GT character.
    :param edition: PGP edition fields (source, coverage, lines) when matched.
    """

    doc_id: str
    bucket: str
    meta: dict
    chars: so.PageChars
    cls: List[str]
    in_block: List[bool]
    block_of: List[int]
    dist: Dict[str, List[int]]
    line: List[int]
    n_lines: int
    blocks: List[dict]
    texts: Dict[str, str]
    gaps: int
    read_ratio: float
    line_class: List[str] = field(default_factory=list)
    edition: dict = field(default_factory=dict)


def reference_line_classes(line: Sequence[int], n_lines: int) -> List[str]:
    """Line-position class of every character from reference line indices.

    :param line: Line index per character.
    :type line: Sequence[int]
    :param n_lines: Number of reference lines.
    :type n_lines: int
    :return: ``first`` / ``last`` / ``middle``, or ``short`` for pages of fewer than three lines.
    :rtype: List[str]
    """
    if n_lines < 3:
        return ["short"] * len(line)
    return ["first" if i == 0 else "last" if i == n_lines - 1 else "middle" for i in line]


def profile_page(doc: rc.BenchDoc, lines: Sequence[str], raw: str, chars: so.PageChars,
                 config: Config, meta: dict) -> PageProfile:
    """Per-character profile and blocks of one page.

    :param doc: Benchmark document.
    :type doc: rc.BenchDoc
    :param lines: Reference lines.
    :type lines: Sequence[str]
    :param raw: Raw GT (``slot_oracle.alignment_raw``).
    :type raw: str
    :param chars: Its char network (``slot_oracle.page_chars``).
    :type chars: so.PageChars
    :param config: Reader set.
    :type config: Config
    :param meta: ``slot_oracle.page_metadata`` fields of the page.
    :type meta: dict
    :return: The page profile (blocks not yet annotated).
    :rtype: PageProfile
    """
    flags = [e.cls == "unreachable" for e in chars.entries]
    blocks = so.page_blocks(chars, config.models, lines, raw)
    block_of = [-1] * len(flags)
    for b, block in enumerate(blocks):
        block_of[block["start"]:block["end"]] = [b] * block["length"]
    positions = so.raw_positions("".join(chars.tokens), raw)
    marks = damage_marks(raw)
    dist = {}
    for group, kinds in DISTANCE_GROUPS.items():
        per_raw = mark_distance(len(raw), marks, kinds)
        dist[group] = [per_raw[p] if per_raw else 1 for p in positions]
    folder = rc.SOURCES[config.benchmark].outputs_dir / doc.doc_id
    texts = {m: " ".join(rc.normalize_lines(doc.raws[m])) for m in config.models}
    for m in config.extra:
        path = folder / f"{m}.txt"
        if path.exists():
            texts[m] = " ".join(rc.normalize_lines(path.read_text(errors="replace")))
    line = so.token_lines(lines)
    gt_letters = len(letters(doc.gt_ink)) or 1
    read_ratio = max(len(letters(texts[m])) for m in config.models) / gt_letters
    return PageProfile(doc.doc_id, doc.bucket, meta, chars, [e.cls for e in chars.entries],
                       so.block_mask(flags, so.BLOCK_WINDOW["char"]), block_of, dist, line,
                       len(lines), blocks, texts, gap_count(raw), round(read_ratio, 3),
                       reference_line_classes(line, len(lines)))


def attach_edition(page: PageProfile, editions: Dict[str, str]) -> None:
    """Replace a PGP page's line classes with those of its best-matching PGP edition.

    Also records which GT characters lie on the edition's second labelled
    side (recto / verso, in GT order) when the GT covers both.

    :param page: Page profile (``line_class`` is overwritten, ``edition`` set).
    :type page: PageProfile
    :param editions: ``{source: content}`` of the page's PGP document (may be empty).
    :type editions: Dict[str, str]
    """
    ink = "".join(page.chars.tokens)
    source, lines, per_char, coverage = best_edition(ink, editions)
    if source is None:
        page.line_class = ["no edition"] * len(ink)
        return
    classes = line_classes(lines)
    sides = [s for s in dict.fromkeys(lines[i].side for i in per_char if i is not None)
             if s in ("recto", "verso")]
    second = sides[1] if len(sides) > 1 else None
    page.line_class = ["unmapped" if i is None else classes[i] for i in per_char]
    page.edition = {"source": source, "coverage": round(coverage, 4), "lines": lines,
                    "per_char": per_char, "sides": sides,
                    "second_side": [second is not None and i is not None and lines[i].side == second
                                    for i in per_char]}


STRUCTURE_CATEGORIES = ("GT text beyond its PGP edition (a second copy in the GT)", "second side of the edition",
                        "margin or address region", "block on a page every reader read short")
DIFFERS_CATEGORY = "GT wording the matched PGP edition lacks"


def char_category(page: PageProfile, k: int, kinds: Sequence[str]) -> str:
    """Where one unreachable character sits (first category that applies).

    :param page: Page profile.
    :type page: PageProfile
    :param k: GT character index.
    :type k: int
    :param kinds: :func:`block_kind` of each of the page's blocks.
    :type kinds: Sequence[str]
    :return: One of :data:`STRUCTURE_CATEGORIES` (PGP edition structure, or a
        block on a page where the most prolific reader wrote under
        ``SHORT_READ`` of the GT's letters), :data:`DIFFERS_CATEGORY` (a GT
        character the matched edition does not have, on a page whose edition
        covers at least ``GT_BEYOND_EDITION`` of the GT: the benchmark's GT and
        the PGP edition read differently here), ``near damage``,
        ``block: <kind>`` or ``local``.
    :rtype: str
    """
    if page.line_class[k] == "unmapped":
        beyond = page.edition.get("coverage", 1.0) < GT_BEYOND_EDITION
        return STRUCTURE_CATEGORIES[0] if beyond else DIFFERS_CATEGORY
    if page.edition and page.edition["second_side"][k]:
        return STRUCTURE_CATEGORIES[1]
    if page.line_class[k] in ("margin", "address"):
        return STRUCTURE_CATEGORIES[2]
    if page.block_of[k] >= 0 and page.read_ratio < SHORT_READ:
        return STRUCTURE_CATEGORIES[3]
    if page.dist["damage"][k] <= DAMAGE_WINDOW:
        return "near damage"
    if page.block_of[k] >= 0:
        return f"block: {kinds[page.block_of[k]]}"
    return "local"


def annotate_blocks(page: PageProfile, config: Config, link: Dict[str, str],
                    short: Callable[[str], str]) -> None:
    """Add damage distance, line class, reader behaviour, note, category and links to a page's blocks.

    :param page: Page profile.
    :type page: PageProfile
    :param config: Reader set.
    :type config: Config
    :param link: Link fields of the page (``link`` and optional ``image``).
    :type link: Dict[str, str]
    :param short: Display name of a model.
    :type short: Callable[[str], str]
    """
    ink = "".join(page.chars.tokens)
    for block in page.blocks:
        k0, k1 = block["start"], block["end"]
        damage = page.dist["damage"][k0:k1]
        gt_block = block["gt_text"]
        rest = ink[:k0] + " " + ink[k1:]
        behaviour = {m: reader_behaviour(gt_block, block["readers"][m], page.texts[m], rest)
                     for m in config.models}
        g = letters(gt_block)
        block.update({
            "near_damage_share": round(sum(d <= DAMAGE_WINDOW for d in damage) / len(damage), 3),
            "min_damage_distance": min(damage),
            "line_class": Counter(page.line_class[k0:k1]).most_common(1)[0][0],
            "read_ratio": page.read_ratio,
            "apparatus": bool(_APPARATUS.search(gt_block)),
            "behaviour": behaviour,
            "elsewhere": {m: round(gram_share(g, grams(letters(page.texts[m]))), 3)
                          for m in page.texts},
            "note": block_note(behaviour, block["readers"], gt_block, short),
            **link,
        })
        if page.edition:
            lines = page.edition["lines"]
            idx = sorted({i for i in page.edition["per_char"][k0:k1] if i is not None})
            block["edition_source"] = page.edition["source"]
            block["edition_lines"] = [lines[i].raw for i in idx]
            block["edition_line_numbers"] = [i + 1 for i in idx]
            block["edition_n_lines"] = len(lines)
            block["edition_places"] = sorted({f"{lines[i].side or 'unlabelled'} {lines[i].region}" for i in idx})
    kinds = [block_kind(b) for b in page.blocks]
    for block, kind in zip(page.blocks, kinds):
        cats = Counter(char_category(page, k, kinds) for k in range(block["start"], block["end"])
                       if page.cls[k] == "unreachable")
        block["kind"] = kind
        block["category"] = cats.most_common(1)[0][0] if cats else "none"


# ── Characterisation ──────────────────────────────────────────────────────────

def gini(values: Sequence[float]) -> float:
    """Gini coefficient of non-negative values.

    :param values: Per-page amounts.
    :type values: Sequence[float]
    :return: 0 (even) to (n - 1) / n (all on one page); 0.0 for no mass.
    :rtype: float
    """
    xs = sorted(values)
    total = sum(xs)
    if not xs or total == 0:
        return 0.0
    n = len(xs)
    return 2 * sum((i + 1) * x for i, x in enumerate(xs)) / (n * total) - (n + 1) / n


def concentration(per_page: Sequence[float]) -> Dict[str, float]:
    """How a per-page amount is spread over pages.

    :param per_page: Amount per page.
    :type per_page: Sequence[float]
    :return: ``gini``, ``top10_share``, ``pages_with_any`` and the number of
        pages holding half / 80 % of the total (``pages_for_50`` / ``pages_for_80``).
    :rtype: Dict[str, float]
    """
    xs = sorted(per_page, reverse=True)
    total = sum(xs) or 1
    cum, need = 0.0, {}
    for i, x in enumerate(xs, start=1):
        cum += x
        for q in (50, 80):
            if f"pages_for_{q}" not in need and cum >= q / 100 * total:
                need[f"pages_for_{q}"] = i
    return {"gini": round(gini(xs), 3), "top10_share": round(sum(xs[:10]) / total, 3),
            "pages_with_any": sum(x > 0 for x in xs), "pages": len(xs), **need}


def class_table(pages: Sequence[PageProfile], key: Callable[[PageProfile, int], str]) -> Dict[str, Dict[str, float]]:
    """GT, unreachable and block characters per class of some per-character key.

    :param pages: Page profiles.
    :type pages: Sequence[PageProfile]
    :param key: ``(page, character index) -> class``.
    :type key: Callable[[PageProfile, int], str]
    :return: ``{class: {gt, unreachable, block, learnable, unreachable_rate,
        share_of_gt, share_of_unreachable, share_of_block}}``; ``block`` =
        unreachable characters in blocks (``block_mask``).
    :rtype: Dict[str, Dict[str, float]]
    """
    counts: Dict[str, Counter] = {}
    for page in pages:
        for k, cls in enumerate(page.cls):
            c = counts.setdefault(key(page, k), Counter())
            c["gt"] += 1
            c[cls] += 1
            c["block"] += page.in_block[k]
    totals = Counter()
    for c in counts.values():
        totals.update(c)
    out = {}
    for name, c in sorted(counts.items(), key=lambda kv: -kv[1]["gt"]):
        out[name] = {"gt": c["gt"], "unreachable": c["unreachable"], "learnable": c["learnable"],
                     "block": c["block"],
                     "unreachable_rate": round(c["unreachable"] / c["gt"], 4),
                     "share_of_gt": round(c["gt"] / (totals["gt"] or 1), 4),
                     "share_of_unreachable": round(c["unreachable"] / (totals["unreachable"] or 1), 4),
                     "share_of_block": round(c["block"] / (totals["block"] or 1), 4)}
    return out


def near_class(group: str) -> Callable[[PageProfile, int], str]:
    """Key function: within ``DAMAGE_WINDOW`` of a mark of one distance group, or not.

    :param group: A :data:`DISTANCE_GROUPS` key.
    :type group: str
    :return: ``(page, k) -> "near" | "far"``.
    :rtype: Callable[[PageProfile, int], str]
    """
    return lambda page, k: "near" if page.dist[group][k] <= DAMAGE_WINDOW else "far"


def correlations(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Pearson and Spearman correlation.

    :param x: First variable.
    :type x: Sequence[float]
    :param y: Second variable.
    :type y: Sequence[float]
    :return: ``{"pearson": r, "spearman": rho, "n": n}`` (NaN-free: 0.0 for a constant variable).
    :rtype: Dict[str, float]
    """
    if len(set(x)) < 2 or len(set(y)) < 2:
        return {"pearson": 0.0, "spearman": 0.0, "n": len(x)}
    return {"pearson": round(statistics.correlation(x, y), 3),
            "spearman": round(statistics.correlation(x, y, method="ranked"), 3), "n": len(x)}


def block_inventory(pages: Sequence[PageProfile]) -> Dict[str, object]:
    """Counts and lengths of the dumped blocks, and how their text was handled.

    :param pages: Page profiles with annotated blocks.
    :type pages: Sequence[PageProfile]
    :return: Block counts, length quantiles, characters per majority behaviour,
        characters whose text no voting reader wrote anywhere (n-gram share
        below ``NOWHERE_SHARE`` for every voting reader), and the same with the
        extra readers.
    :rtype: Dict[str, object]
    """
    blocks = [b for p in pages for b in p.blocks]
    lengths = sorted(b["length"] for b in blocks) or [0]
    by_majority: Counter = Counter()
    nowhere = nowhere_all = anywhere = 0
    for page in pages:
        for b in page.blocks:
            majority = Counter(b["behaviour"].values()).most_common(1)[0][0]
            by_majority[majority] += b["unreachable"]
            voting = [b["elsewhere"][m] for m in b["behaviour"]]
            every = list(b["elsewhere"].values())
            nowhere += b["unreachable"] * (max(voting) < NOWHERE_SHARE)
            nowhere_all += b["unreachable"] * (max(every) < NOWHERE_SHARE)
            anywhere += b["unreachable"] * (max(voting) >= ELSEWHERE_SHARE)
    total = sum(b["unreachable"] for b in blocks) or 1
    q = lambda f: lengths[int(f * (len(lengths) - 1))]  # noqa: E731
    return {"blocks": len(blocks), "pages": len({p.doc_id for p in pages if p.blocks}),
            "chars": sum(b["length"] for b in blocks), "unreachable": sum(b["unreachable"] for b in blocks),
            "gt_only": sum(b["gt_only"] for b in blocks),
            "length_quantiles": {"min": q(0), "p25": q(0.25), "median": q(0.5), "p75": q(0.75),
                                 "p90": q(0.9), "max": q(1)},
            "majority_behaviour_chars": dict(by_majority.most_common()),
            "written_nowhere_share": round(nowhere / total, 3),
            "written_nowhere_incl_extra_share": round(nowhere_all / total, 3),
            "written_elsewhere_share": round(anywhere / total, 3)}


BLOCK_KINDS = ("shifted by the alignment", "written elsewhere by a reader", "other page text",
               "dropped, written by no reader", "dropped", "looped", "misread")


def block_kind(block: dict) -> str:
    """One kind per annotated block, from its readers' majority behaviour.

    :param block: Annotated block (``behaviour`` and ``elsewhere`` fields).
    :type block: dict
    :return: ``shifted by the alignment`` (most readers wrote the text over
        the block); ``written elsewhere by a reader`` (some voting reader wrote
        at least ``ELSEWHERE_SHARE`` of its n-grams somewhere: reading order or
        alignment, not a reading failure); then by majority behaviour:
        ``other page text``, ``dropped, written by no reader`` (every voting
        reader below ``NOWHERE_SHARE``), ``dropped``, ``looped``, ``misread``.
    :rtype: str
    """
    majority = Counter(block["behaviour"].values()).most_common(1)[0][0]
    voting = max(block["elsewhere"][m] for m in block["behaviour"])
    if majority == "shifted":
        return "shifted by the alignment"
    if voting >= ELSEWHERE_SHARE:
        return "written elsewhere by a reader"
    if majority == "other page text":
        return "other page text"
    if majority.startswith("dropped"):
        return "dropped, written by no reader" if voting < NOWHERE_SHARE else "dropped"
    return "looped" if majority == "looped" else "misread"


def unreachable_decomposition(pages: Sequence[PageProfile]) -> Dict[str, int]:
    """Every unreachable character in one category (:func:`char_category`).

    :param pages: Page profiles with annotated blocks.
    :type pages: Sequence[PageProfile]
    :return: ``{category: characters}``, largest first.
    :rtype: Dict[str, int]
    """
    out: Counter = Counter()
    for page in pages:
        kinds = [block_kind(b) for b in page.blocks]
        out.update(char_category(page, k, kinds) for k, cls in enumerate(page.cls) if cls == "unreachable")
    return dict(out.most_common())


def structure_pages(pages: Sequence[PageProfile]) -> Dict[str, Dict[str, int]]:
    """Pages behind each structural category (and the GT-differs category).

    :param pages: Page profiles with annotated blocks.
    :type pages: Sequence[PageProfile]
    :return: ``{category: {doc_id: unreachable characters}}``, largest page first.
    :rtype: Dict[str, Dict[str, int]]
    """
    out: Dict[str, Counter] = {}
    for page in pages:
        kinds = [block_kind(b) for b in page.blocks]
        for k, cls in enumerate(page.cls):
            if cls != "unreachable":
                continue
            category = char_category(page, k, kinds)
            if category in STRUCTURE_CATEGORIES or category == DIFFERS_CATEGORY:
                out.setdefault(category, Counter())[page.doc_id] += 1
    return {category: dict(c.most_common()) for category, c in out.items()}


def characterise(pages: Sequence[PageProfile], config: Config) -> Dict[str, object]:
    """Every characterisation number of one reader set.

    :param pages: Page profiles with annotated blocks.
    :type pages: Sequence[PageProfile]
    :param config: Reader set.
    :type config: Config
    :return: Totals, damage-proxy tables, page-level correlations,
        line-position and bucket tables, concentration and the block inventory.
    :rtype: Dict[str, object]
    """
    out: Dict[str, object] = {"models": list(config.models), "extra": list(config.extra),
                              "benchmark": config.benchmark, "pages": len(pages)}
    gt = sum(len(p.cls) for p in pages)
    unreachable = sum(c == "unreachable" for p in pages for c in p.cls)
    in_block = sum(sum(p.in_block) for p in pages)
    out["totals"] = {"gt": gt, "unreachable": unreachable, "block": in_block,
                     "unreachable_share": round(unreachable / gt, 4),
                     "block_share_of_unreachable": round(in_block / (unreachable or 1), 4)}
    out["damage"] = {group: class_table(pages, near_class(group)) for group in DISTANCE_GROUPS}
    # unreachable outside blocks, near / far from damage
    near = near_class("damage")
    out["damage_nonblock"] = class_table(
        pages, lambda p, k: f"{near(p, k)} {'block' if p.in_block[k] else 'other'}")
    rows = []
    for p in pages:
        n = len(p.cls)
        u = sum(c == "unreachable" for c in p.cls)
        rows.append({"doc_id": p.doc_id, "gt": n, "unreachable": u, "block": sum(p.in_block),
                     "unreachable_share": u / n, "recon_frac": p.meta.get("recon_frac") or 0.0,
                     "gap_density": 100 * p.gaps / n,
                     "near_damage_share": sum(d <= DAMAGE_WINDOW for d in p.dist["damage"]) / n})
    share = [r["unreachable_share"] for r in rows]
    out["page_correlations"] = {
        "recon_frac": correlations([r["recon_frac"] for r in rows], share),
        "gap_density": correlations([r["gap_density"] for r in rows], share),
        "near_damage_share": correlations([r["near_damage_share"] for r in rows], share)}
    out["page_quantiles"] = {k: [round(statistics.quantiles([r[k] for r in rows], n=4)[i], 4)
                                 for i in range(3)]
                             for k in ("unreachable_share", "recon_frac", "gap_density", "near_damage_share")}
    out["concentration"] = {"unreachable": concentration([r["unreachable"] for r in rows]),
                            "block": concentration([r["block"] for r in rows])}
    out["line_position"] = class_table(pages, lambda p, k: p.line_class[k])
    out["relative_position"] = class_table(
        pages, lambda p, k: ("first 10 %" if k < 0.1 * len(p.cls)
                             else "last 10 %" if k >= 0.9 * len(p.cls) else "middle 80 %"))
    out["bucket"] = class_table(pages, lambda p, k: p.bucket)
    if config.benchmark == rc.RELIGIOUS:
        out["line_position_by_columns"] = class_table(pages, lambda p, k: f"{p.bucket} {p.line_class[k]}")
        out["genre"] = class_table(pages, lambda p, k: str(p.meta.get("genre")))
    else:
        out["edition_matching"] = {
            "pages_with_edition": sum(bool(p.edition) for p in pages),
            "coverage_quantiles": [round(x, 3) for x in statistics.quantiles(
                [p.edition["coverage"] for p in pages if p.edition], n=4)],
            "pages_gt_on_two_sides": sum(len(p.edition.get("sides", [])) > 1 for p in pages)}
        two_sided = [p for p in pages if len(p.edition.get("sides", [])) > 1]
        out["side_rank"] = class_table(two_sided, lambda p, k: (
            "second side" if p.edition["second_side"][k] else "first side or unlabelled"))
        out["side_rank_pages"] = [p.doc_id for p in two_sided]
    out["blocks"] = block_inventory(pages)
    out["block_kinds"] = dict(Counter(
        {kind: sum(b["unreachable"] for p in pages for b in p.blocks if block_kind(b) == kind)
         for kind in BLOCK_KINDS}).most_common())
    out["decomposition"] = unreachable_decomposition(pages)
    out["structure_pages"] = structure_pages(pages)
    out["read_ratio"] = {"short_read_pages": sorted(p.doc_id for p in pages if p.read_ratio < SHORT_READ),
                         "quantiles": [round(x, 3) for x in statistics.quantiles(
                             [p.read_ratio for p in pages], n=10)]}
    out["local_confusions"] = confusion_pairs(pages)
    out["apparatus_blocks"] = {"blocks": sum(b["apparatus"] for p in pages for b in p.blocks),
                               "unreachable": sum(b["unreachable"] for p in pages for b in p.blocks
                                                  if b["apparatus"])}
    return out


def edition_coverage(ink: str, text: str) -> float:
    """Share of the GT's characters an edition's text matches (indel alignment).

    :param ink: Scoring reference.
    :type ink: str
    :param text: Visible text of an edition.
    :type text: str
    :return: Matched characters over GT characters (0.0 for an empty GT).
    :rtype: float
    """
    matched = so.aligned_positions(ink, text)
    return sum(p is not None for p in matched) / len(matched) if matched else 0.0


def dissent_overlap(pages: Sequence[PageProfile], editions: Dict[str, Dict[str, str]],
                    two: Collection[str], pgpid_of: Dict[str, str]) -> Dict[str, object]:
    """Readers' unreachable characters against two scholars' disagreements.

    Every page whose PGP id is a two-edition document is labelled with
    :func:`edition_slots`.  Group ``all`` pools every such page; group
    ``clean`` only those whose GT is one of the two editions (one edition
    matches at least ``CLEAN_PAIR`` of the GT characters), where ``gt_only``
    and ``passage`` labels cannot come from the GT holding other text; group
    ``same_coverage`` the clean pages where the other edition also matches at
    least ``SAME_COVERAGE`` of the GT (both scholars transcribed the same
    stretch of text, so the labels are word-level readings).

    :param pages: PGP page profiles (six VLMs).
    :type pages: Sequence[PageProfile]
    :param editions: ``{pgpid: {source: content}}`` (``two_edition_documents.editions_by_source``).
    :type editions: Dict[str, Dict[str, str]]
    :param two: PGP ids of the two-edition documents (``two_edition_documents.two_edition_documents``).
    :type two: Collection[str]
    :param pgpid_of: ``{doc_id: pgpid}``.
    :type pgpid_of: Dict[str, str]
    :return: ``documents`` (one row per page) and, per group, the label
        counts of all GT characters / unreachable ones / unreachable ones in
        and outside blocks (``labels``), the near-dissent table and what ROVER
        voted at the dissent slots where the GT follows one edition
        (``votes``: by class, by whether the other edition has a different
        reading or nothing, and by outcome) and the labels of the unreachable
        characters per :func:`char_category` (``by_category``).
    :rtype: Dict[str, object]
    """
    docs: List[dict] = []
    groups = {g: {"labels": {}, "near_dissent": Counter(), "votes": Counter(), "by_category": Counter(),
                  "n_docs": 0}
              for g in ("all", "clean", "same_coverage")}
    for page in pages:
        pgpid = pgpid_of.get(page.doc_id)
        if pgpid not in two:
            continue
        (sa, a), (sb, b) = edition_pair(editions[pgpid])
        ink = "".join(page.chars.tokens)
        labels = edition_slots(page.chars.tokens, a, b)
        cover = {"a": edition_coverage(ink, a), "b": edition_coverage(ink, b)}
        clean = max(cover.values()) >= CLEAN_PAIR
        same = clean and min(cover.values()) >= SAME_COVERAGE
        member_groups = ([groups["all"]] + ([groups["clean"]] if clean else [])
                         + ([groups["same_coverage"]] if same else []))
        prof = pair_profile(a, b)
        members = page.chars.network.members[:-1]
        weights = [1.0] * len(members)
        row: Counter = Counter()
        kinds = [block_kind(b) for b in page.blocks]
        for group in member_groups:
            group["n_docs"] += 1
        for k, lab in enumerate(labels):
            cls = page.cls[k]
            category = char_category(page, k, kinds) if cls == "unreachable" else None
            scopes = ["all"]
            if cls == "unreachable":
                scopes += ["unreachable", "unreachable_block" if page.in_block[k] else "unreachable_other"]
            for scope in scopes:
                row[f"{scope}_{lab.label}"] += 1
                row[scope] += 1
            outcome = None
            if lab.label == "dissent" and lab.follows in ("a", "b"):
                slot = page.chars.network.slots[page.chars.entries[k].slot]
                vote = so.token_key(rc.vote_slot(slot[:-1], members, weights,
                                                 page.chars.page.backbone), "char")
                keys = [so.token_key(e, "char") for e in slot[:-1]]
                other = "nothing" if lab.other is None else "a different reading"
                outcome = (other, "gt" if vote == so.token_key(slot[-1], "char")
                           else "other scholar" if vote == lab.other else "neither",
                           lab.other in keys)
            for group in member_groups:
                for scope in scopes:
                    group["labels"].setdefault(scope, Counter())[lab.label] += 1
                group["near_dissent"][f"{'near' if lab.near_dissent else 'far'} | {cls}"] += 1
                if category:
                    group["by_category"][f"{category} | {lab.label}"] += 1
                    group["by_category"][f"{category} | near dissent"] += lab.near_dissent
                if outcome:
                    group["votes"][f"{cls} | other edition has {outcome[0]} | ROVER voted {outcome[1]}"] += 1
                    group["votes"][f"{cls} | other edition has {outcome[0]} | some reader wrote the other "
                                   f"edition's entry"] += outcome[2]
        docs.append({"doc_id": page.doc_id, "pgpid": pgpid, "source_a": sa, "source_b": sb,
                     "len_a": len(a), "len_b": len(b), "gt_len": len(ink),
                     "gt_coverage_a": round(cover["a"], 3), "gt_coverage_b": round(cover["b"], 3),
                     "clean": clean, "same_coverage": same,
                     "pair_difference": round(prof["differ_slots"] / (prof["slots"] or 1), 4),
                     "pair_passage": round(prof["passage_slots"] / (prof["slots"] or 1), 4),
                     **dict(row)})
    out: Dict[str, object] = {"documents": docs}
    for name, group in groups.items():
        out[name] = {"n_docs": group["n_docs"],
                     "labels": {scope: {lab: c.get(lab, 0) for lab in DISSENT_LABELS}
                                for scope, c in group["labels"].items()},
                     "near_dissent": dict(sorted(group["near_dissent"].items())),
                     "votes": dict(sorted(group["votes"].items())),
                     "by_category": dict(sorted(group["by_category"].items()))}
    return out


_PAIR_KEYS = ("slots", "differ_slots", "passage_slots", "a_chars", "a_differ", "a_dissent",
              "a_block", "a_block_dissent")


def summarise_pairs(totals: Counter, runs: Sequence[int], rates: Sequence[float]) -> Dict[str, object]:
    """Pooled numbers of a set of edition pairs.

    :param totals: Sums of the :func:`pair_profile` counts.
    :type totals: Counter
    :param runs: Lengths of every differing run.
    :type runs: Sequence[int]
    :param rates: Per-pair difference rate (differing slots over slots).
    :type rates: Sequence[float]
    :return: Counts, rates, block shares and the run-length profile.
    :rtype: Dict[str, object]
    """
    run_slots = sum(runs) or 1
    return {"pairs": len(rates), **{k: totals[k] for k in _PAIR_KEYS},
            "difference_rate": round(totals["differ_slots"] / (totals["slots"] or 1), 4),
            "median_pair_difference_rate": round(statistics.median(rates), 4) if rates else 0.0,
            "passage_share_of_differences": round(totals["passage_slots"] / (totals["differ_slots"] or 1), 4),
            "a_block_share_of_differences": round(totals["a_block"] / (totals["a_differ"] or 1), 4),
            "a_block_share_of_dissent": round(totals["a_block_dissent"] / (totals["a_dissent"] or 1), 4),
            "dissent_rate_of_a": round(totals["a_dissent"] / (totals["a_chars"] or 1), 4),
            "run_slots_by_length": {name: round(sum(r for r in runs if lo <= r <= hi) / run_slots, 4)
                                    for lo, hi, name in ((1, 1, "1"), (2, 7, "2-7"), (8, 20, "8-20"),
                                                         (21, 60, "21-60"), (61, 10 ** 9, ">60"))},
            "runs": len(runs), "median_run": statistics.median(runs) if runs else 0}


def two_edition_profile(editions: Dict[str, Dict[str, str]], two: Collection[str]) -> Dict[str, object]:
    """Profile every two-edition pair: difference rate, run lengths, block share, letter pairs.

    :param editions: ``{pgpid: {source: content}}``.
    :type editions: Dict[str, Dict[str, str]]
    :param two: PGP ids of the two-edition documents.
    :type two: Collection[str]
    :return: :func:`summarise_pairs` for ``all`` pairs and for ``similar_length``
        pairs (shorter text at least 90 % of the longer), and the most common
        letter-for-letter substitutions (``letter_pairs``: pair -> share of all
        letter substitutions).
    :rtype: Dict[str, object]
    """
    totals = {"all": Counter(), "similar_length": Counter()}
    runs: Dict[str, List[int]] = {"all": [], "similar_length": []}
    rates: Dict[str, List[float]] = {"all": [], "similar_length": []}
    subs: Counter = Counter()
    for pgpid in sorted(two, key=int):
        (_, a), (_, b) = edition_pair(editions[pgpid])
        prof = pair_profile(a, b)
        names = ["all"] + (["similar_length"] if min(len(a), len(b)) >= 0.9 * max(len(a), len(b)) else [])
        for name in names:
            totals[name].update({k: prof[k] for k in _PAIR_KEYS})
            runs[name].extend(prof["runs"])
            rates[name].append(prof["differ_slots"] / (prof["slots"] or 1))
        subs.update(prof["subs"])
    n_subs = sum(subs.values()) or 1
    return {**{name: summarise_pairs(totals[name], runs[name], rates[name]) for name in totals},
            "letter_pairs": {pair: round(n / n_subs, 4) for pair, n in subs.most_common(12)},
            "letter_substitutions": sum(subs.values())}


def confusion_pairs(pages: Sequence[PageProfile]) -> Dict[str, object]:
    """Letter pairs of the readers' unreachable substitutions outside blocks and far from damage.

    :param pages: Page profiles.
    :type pages: Sequence[PageProfile]
    :return: ``letter_substitutions`` (unreachable characters where the GT and
        ROVER's vote are two different letters) and ``letter_pairs`` (the most
        common sorted pairs with their share).
    :rtype: Dict[str, object]
    """
    subs: Counter = Counter()
    for page in pages:
        members = page.chars.network.members[:-1]
        weights = [1.0] * len(members)
        for k, cls in enumerate(page.cls):
            if (cls != "unreachable" or page.block_of[k] >= 0
                    or page.dist["damage"][k] <= DAMAGE_WINDOW):
                continue
            slot = page.chars.network.slots[page.chars.entries[k].slot]
            g = slot[-1]
            v = rc.vote_slot(slot[:-1], members, weights, page.chars.page.backbone)
            if g and v and g != v and _LETTER.match(g) and _LETTER.match(v):
                subs["".join(sorted((g, v)))] += 1
    n = sum(subs.values()) or 1
    return {"letter_substitutions": sum(subs.values()),
            "letter_pairs": {pair: round(c / n, 4) for pair, c in subs.most_common(12)}}


# ── Sample ────────────────────────────────────────────────────────────────────

def length_class(length: int) -> str:
    """Length class of a block.

    :param length: GT characters.
    :type length: int
    :return: A :data:`LENGTH_CLASSES` name.
    :rtype: str
    """
    return next(name for lo, hi, name in LENGTH_CLASSES if lo <= length <= hi)


def allocate(sizes: Dict[str, int], quota: int) -> Dict[str, int]:
    """Split a quota over strata in proportion to size, at least one each when possible.

    :param sizes: ``{stratum: blocks available}``.
    :type sizes: Dict[str, int]
    :param quota: Blocks to draw.
    :type quota: int
    :return: ``{stratum: blocks to draw}`` (largest remainder, never above the size).
    :rtype: Dict[str, int]
    """
    total = sum(sizes.values())
    quota = min(quota, total)
    if not quota:
        return {s: 0 for s in sizes}
    exact = {s: quota * n / total for s, n in sizes.items()}
    out = {s: int(x) for s, x in exact.items()}
    for s in sorted(sizes, key=lambda s: (-(exact[s] - out[s]), s))[:quota - sum(out.values())]:
        out[s] += 1
    for s in sorted(sizes):                       # every non-empty stratum gets one if the quota allows
        if sizes[s] and not out[s]:
            donor = max(out, key=lambda t: (out[t], t))
            if out[donor] > 1:
                out[donor] -= 1
                out[s] += 1
    return out


def draw_sample(blocks: Sequence[dict], quota: int, seed: int,
                per_page: Dict[str, int], cap: int = SAMPLE_PER_PAGE) -> List[dict]:
    """Stratified draw of blocks (stratum = bucket x length class).

    :param blocks: Candidate blocks.
    :type blocks: Sequence[dict]
    :param quota: Blocks to draw.
    :type quota: int
    :param seed: Random seed.
    :type seed: int
    :param per_page: Blocks already drawn per page (updated in place).
    :type per_page: Dict[str, int]
    :param cap: Most blocks drawn from one page.
    :type cap: int
    :return: Drawn blocks; a stratum short of eligible blocks passes its
        remainder to the largest strata.
    :rtype: List[dict]
    """
    rng = random.Random(seed)
    strata: Dict[str, List[dict]] = {}
    for block in sorted(blocks, key=lambda b: b["block_id"]):
        strata.setdefault(f"{block['bucket']} | {length_class(block['length'])}", []).append(block)
    for name in sorted(strata):
        rng.shuffle(strata[name])
    chosen: List[dict] = []
    taken = set()

    def take(name: str, n: int) -> int:
        got = 0
        for block in strata[name]:
            if got >= n:
                break
            if block["block_id"] in taken or per_page.get(block["doc_id"], 0) >= cap:
                continue
            chosen.append(block)
            taken.add(block["block_id"])
            per_page[block["doc_id"]] = per_page.get(block["doc_id"], 0) + 1
            got += 1
        return got

    want = allocate({s: len(v) for s, v in strata.items()}, quota)
    short = sum(want[name] - take(name, want[name]) for name in sorted(strata))
    for name in sorted(strata, key=lambda s: (-len(strata[s]), s)):   # shortfall: largest strata
        if short <= 0:
            break
        short -= take(name, short)
    return chosen


def sample_blocks(blocks: Sequence[dict], far: int, near: int, seed: int = SAMPLE_SEED) -> List[dict]:
    """Blocks for human review: mostly far from damage marks, some near them.

    :param blocks: Annotated blocks of one reader set.
    :type blocks: Sequence[dict]
    :param far: Blocks with no character within ``DAMAGE_WINDOW`` of a damage mark.
    :type far: int
    :param near: Blocks with at least one.
    :type near: int
    :param seed: Random seed.
    :type seed: int
    :return: Drawn blocks, far ones first, each with ``sample_group``.
    :rtype: List[dict]
    """
    per_page: Dict[str, int] = {}
    groups = (("far from damage", [b for b in blocks if b["min_damage_distance"] > DAMAGE_WINDOW], far),
              ("near damage", [b for b in blocks if b["min_damage_distance"] <= DAMAGE_WINDOW], near))
    out = []
    for i, (name, pool, quota) in enumerate(groups):
        for block in draw_sample(pool, quota, seed + i, per_page):
            out.append({**block, "sample_group": name})
    return out


# ── Run ───────────────────────────────────────────────────────────────────────

def short_name(model: str) -> str:
    """Display name of a reader stem.

    :param model: Reader file stem.
    :type model: str
    :return: e.g. ``v21b-1200``, ``v22i-a50``, ``Kraken seg``.
    :rtype: str
    """
    if model == KSEG:
        return "Kraken seg"
    stem = model.removeprefix(P)
    return stem.replace("_step", "-").replace("_a", "-a")


def page_link(benchmark: str, meta: dict, pgpid: Optional[str]) -> Dict[str, str]:
    """Links of one page (PGP document page, else the image; KTIV item and page image).

    :param benchmark: Benchmark key.
    :type benchmark: str
    :param meta: ``slot_oracle.page_metadata`` fields.
    :type meta: dict
    :param pgpid: PGP id (PGP pages only).
    :type pgpid: Optional[str]
    :return: ``{"link": ..., "image": ...}`` (``image`` only where a page image differs from ``link``).
    :rtype: Dict[str, str]
    """
    if benchmark == rc.RELIGIOUS:
        return {"link": KTIV_ITEM_URL.format(sys_num=meta.get("sys_num")),
                "image": IIIF.format(fl=meta.get("fl"))}
    if pgpid:
        return {"link": PGP_DOC_URL.format(pgpid=pgpid), "image": meta.get("image_url") or ""}
    return {"link": meta.get("image_url") or ""}


def run(out_dir: Path) -> Dict[str, object]:
    """Profile both reader sets, the scholar dissent and the sample; write the outputs.

    :param out_dir: Output folder.
    :type out_dir: Path
    :return: The profile written to ``profile.json``.
    :rtype: Dict[str, object]
    """
    from src.finetuning.qwen_hebrew import build_pgp_editions as bpe   # heavy: datasets, PIL

    started = time.time()
    editions = ted.editions_by_source()
    two = set(ted.two_edition_documents())
    merged = bpe.load_merged(bpe.MERGED_JSONL, set())
    profile: Dict[str, object] = {"damage_window": DAMAGE_WINDOW, "ngram": NGRAM,
                                  "block_window": so.BLOCK_WINDOW["char"], "block_share": so.BLOCK_SHARE,
                                  "min_block_chars": so.MIN_BLOCK_CHARS}
    samples: Dict[str, List[dict]] = {}
    pgp_pages: List[PageProfile] = []
    pgpid_of: Dict[str, str] = {}
    for config in CONFIGS:
        meta = so.page_metadata(config.benchmark)
        pages = []
        for doc, lines, raw, chars in so.iter_pages(config.benchmark, config.models):
            page = profile_page(doc, lines, raw, chars, config, meta.get(doc.doc_id, {}))
            pgpid = None
            if config.benchmark != rc.RELIGIOUS:
                ids = merged.pgpids_of.get(doc.doc_id) or []
                pgpid = ids[0] if ids else None
                if pgpid:
                    pgpid_of[doc.doc_id] = pgpid
                attach_edition(page, editions.get(pgpid, {}) if pgpid else {})
            annotate_blocks(page, config, page_link(config.benchmark, page.meta, pgpid), short_name)
            pages.append(page)
        blocks = [{"block_id": f"{p.doc_id}:{b['start']}", "doc_id": p.doc_id, "bucket": p.bucket,
                   "backbone": config.models[p.chars.page.backbone], **p.meta, **b}
                  for p in pages for b in p.blocks]
        so.write_jsonl(out_dir / f"blocks_{config.tag}.jsonl", blocks)
        profile[config.tag] = characterise(pages, config)
        far, near = SAMPLE[config.benchmark]
        samples[config.tag] = sample_blocks(blocks, far, near)
        if config.benchmark != rc.RELIGIOUS:
            pgp_pages = pages
        print(f"[{config.tag}] {len(pages)} pages, {len(blocks)} blocks ({time.time() - started:.0f}s)")
    profile["dissent"] = dissent_overlap(pgp_pages, editions, two, pgpid_of)
    profile["two_edition_profile"] = two_edition_profile(editions, two)
    profile["pgp_pages_with_pgpid"] = len(pgpid_of)
    profile["seconds"] = round(time.time() - started, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "profile.json").write_text(json.dumps(profile, indent=1, ensure_ascii=False))
    (out_dir / "sample.json").write_text(json.dumps(samples, indent=1, ensure_ascii=False))
    return profile


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point.

    :param argv: Arguments (default ``sys.argv[1:]``).
    :type argv: Optional[Sequence[str]]
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    args = p.parse_args(argv)
    profile = run(args.out_dir)
    print(f"wrote {args.out_dir}/(profile.json, sample.json, blocks_*.jsonl) in {profile['seconds']}s")


if __name__ == "__main__":
    main()
