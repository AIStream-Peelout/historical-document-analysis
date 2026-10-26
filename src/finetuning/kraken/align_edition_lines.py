# File name: align_edition_lines.py
# Date: 10/19/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Align page-level PGP edition lines onto Kraken segmenter lines and write kraken-trainable PAGE XML.

Genizah Arabic documentary hands have no line-level ground truth, but the PGP Arabic-edition page images carry a human
edition of the page (``pgp_arabic_editions_v2``) and the production Genizah segmenter (``ktivseg_e10``)
has found line baselines and polygons on them, each with a weak zero-shot read (PP-OCRv6). This module
places each edition line on the segmenter line(s) that hold it and writes PAGE XML whose text is ALWAYS
the human edition line (:func:`~src.finetuning.kraken.build_arabic_lines.clean_target`). The zero-shot
read only steers placement; it is never a training target (no self-labels).

Method, per page:

1. **Lines.** Segmenter lines whose baseline is tiny and whose read is (nearly) empty are noise
   (:func:`is_noise`). Edition lines are the page's ``answer`` lines; a line whose target cleans to ``""``
   (a ``[...]`` gap, a Latin letter, ...) is still aligned as a placeholder for order but never written.
2. **Order.** Horizontal lines are sorted by column, then baseline mean y; steep (rotated, marginal)
   lines go last. Two-column candidates come from a gap in the baseline x-centres wider than
   :data:`COLUMN_GAP` of the page width (:func:`column_split`) or from a vertical gutter no baseline
   crosses (:func:`gutter_split`, e.g. a bifolium); a block candidate reads side-by-side blocks of short
   lines (witness clauses under a full-width text) right block first (:func:`block_order`). Every
   candidate ordering is aligned and the one with the most anchors (then accepted lines, then score) wins.
3. **Fragments.** Adjacent sorted lines on the same baseline height (``|dy| <`` :data:`MERGE_DY` x the
   median line height) whose x-ranges overlap by at most :data:`MERGE_X_OVERLAP` may be merged by the
   DP (2:1 or 3:1); the merged read is the parts' reads right to left.
4. **Similarity.** Rasm skeletons (:func:`skeleton`: dots, hamza carriers and marks folded away, spaces
   kept) compared by a normalised Levenshtein (:func:`similarity`).
5. **Length.** Characters per baseline pixel (``cpp``) starts as the page's edition characters over its
   segmenter baseline length and is refit on the anchors (at least :data:`MIN_REFIT_ANCHORS`).
6. **DP.** Monotone alignment (:func:`align`) with match, merge, skip-segmenter-line and skip-edition-line
   moves; pair score ``0.6 * sim + 0.4 * length score`` for reads with at least :data:`TEXT_MIN_LETTERS`
   letters, else ``0.5 * length score``.
7. **Span evidence.** For a read of at least :data:`COVER_MIN_LETTERS` letters, :func:`span_similarity`
   scores it against every contiguous word span (at least :data:`COVER_WORD_SHARE` of the words) of each
   edition line. The segmenter's polygon often stops before the left end of an Arabic line; such a
   segment reads like a sub-span of its edition line, and a read of half a neighbouring line still scores
   high against that neighbour.
8. **Tiers** (:func:`assign_tiers`): a **partial** segment (best span beats the whole line by
   :data:`COVER_GAIN` and leaves :data:`COVER_LOST` of it out, :func:`is_partial`) is rejected in every
   tier, anchors included, because the full edition line is not in its image. Otherwise ``anchor``
   (sim >= :data:`SIM_ANCHOR`, read >= :data:`ANCHOR_MIN_LETTERS` letters), ``count_match`` (the path is
   the identity over the merged lines and passes the wrong-side guard: over at least
   :data:`GUARD_MIN_PAIRS` pairs with a readable read, mean sim >= :data:`SIM_PAGE` and mean sim minus the
   same reads' mean sim to the page's other edition lines >= :data:`SIM_MARGIN`; the absolute sim alone
   cannot separate sides: unrelated Arabic lines score ~0.21; a page with no anchor also needs
   margin >= :data:`SIM_MARGIN` + :data:`NO_ANCHOR_MARGIN_BUFFER` or :data:`NO_ANCHOR_OWN_BEST` reads per
   column whose best whole-line match is their own line), ``interpolated`` (between two position anchors with no
   skip in between, i.e. equal numbers of edition and segmenter lines), else ``rejected``. Span vetoes
   (:func:`span_vetoes`): a read whose best span sim to another edition line reaches
   :data:`CONTRADICT_MIN_SIM` and beats its own by :data:`CONTRADICT_MARGIN`, or a non-anchor read that
   beats its own by :data:`NEIGHBOR_MARGIN` against edition line e-1 or e+1, cancels the page's count
   match and the interpolation of its anchor-to-anchor stretch. Count-match and interpolated lines must
   also pass the local length guard: width x local cpp (:func:`local_cpps`: median of the bracketing
   covering anchors and the page value) over the edition length within [:data:`LEN_SHORT_MIN`,
   :data:`LEN_LONG_MAX`], stricter on the short side (a truncated segment) than on the long one. Anchors
   are exempt from the length guard (their length scores mostly reflect handwriting-size variation; the
   partial test bounds their coverage). The **lost guard** rejects a pair of any tier, merged segments
   included, whose read leaves more than ``--max-lost`` (:data:`MAX_LOST`; 0.0 = strict) of its edition line
   outside its best span: a line whose read misses words usually has a polygon that misses them.
   ``--drop-count-match`` drops the count-match tier (between two anchors such a line is interpolated);
   ``--drop-interpolated`` drops the interpolated tier; ``--drop-duplicate-texts`` rejects every edition
   line whose NFC target occurs more than once on the page (pen trials: the placement is ambiguous).
   **Text guards** (round 3): the long-read guard (:func:`long_read`) rejects a pair whose read has
   :data:`LONG_READ_WORDS` more words than its edition line (and :data:`LONG_READ_CHARS` times its skeleton
   letters: zero-shot reads split words), or more words and a best span sim of at most
   :data:`LONG_READ_SPAN` (:data:`REJECT_LONG_READ`: the segment holds another line's words too); an accepted
   pair whose read misses an edition word (:func:`missing_words`: a word of at least
   :data:`MISSING_MIN_LETTERS` letters aligned to nothing) is checked for a supralinear insertion below.
9. **Ink extent** (``--extend-to-ink``, on by default; :mod:`~src.finetuning.kraken.line_extent`,
   :mod:`~src.finetuning.kraken.page_layout`). The segmenter's polygon often stops before the line-final
   (left) word, written raised or compressed or segmented as a separate small box, or cuts the first word. On
   the page's ink (background-normalised Otsu plus a local adaptive rule for faint ink, mount masked) each
   target line is walked outward in its baseline band (clamped to the neighbouring lines' baselines, segmented
   or detected from the row profile) to the last ink reachable without a gap of 0.4 line heights, at most 2.0
   line heights per end. The page's **column blocks** (:func:`~src.finetuning.kraken.page_layout.page_blocks`;
   ``--no-column-blocks``: the round-2 page-wide block, :func:`page_block`) bound the walk: one block per
   column group (two-page spreads, two-column pages), its right edge (where the lines start) the
   ``--block-start-percentile`` of the main lines' starts and its left edge the 5th percentile of their ends,
   plus ``--block-slack`` line heights; short or steep margin lines beside the block and gutters before
   unsegmented side material are hard limits without slack. Past the bound only ink connected to the line's
   own letters moves an end. Only components that belong to the line count (centroid in its own band and
   nearer its baseline than any other, not pencil-thin or faint, not side material beyond the bound), and every
   other segmenter line's polygon (grown 2 px) is subtracted from what the line gains. The polygon is rebuilt
   per column; with ``--raise-top`` (on with the extension) every column covers the band up to 0.55 and down
   to 0.45 of the page's measured baseline spacing (:func:`page_spacing`; at most 0.75 / 0.3 line heights),
   pulled back from foreign ink, plus the admitted components (own letters' ascenders and descenders down to
   0.9 line heights, clear of the next line's ink), without narrow stubs. **End checks**: an edition line
   ending in ':' moves its left end along connected ink only (no filler glyph), and a box beside an end whose
   read matches none of the edition line's end words is not absorbed. **Supralinear**: an accepted pair whose
   read misses an edition word merges a short unplaced segment just above it (:func:`supralinear_box`) or the
   ink there (:func:`supralinear_ink`).
10. **Geometry gate** (text-free): an accepted line whose walk reached a segment of another edition line
    (placed there, or its read matches it) is rejected (:data:`REJECT_STEAL`), and so is one whose final
    polygon newly covers more than ``--max-overlap`` of another line's polygon (:data:`REJECT_OVERLAP`), one
    whose ink still continues beyond an end by more than ``--max-ink-beyond`` line heights
    (:data:`REJECT_INK_BEYOND`), one whose end took detached ink more than ``--max-end-shift`` line heights
    above or below its own (:data:`REJECT_END_SHIFT`, an L-shaped end), one whose missing word was found
    nowhere (:data:`REJECT_MISSING_WORD`), and one that leaves more than ``--max-own-ink-outside`` of the own
    ink in its band (its x-range plus ``--own-ink-reach`` line heights, inside its column block) outside its
    polygon (:data:`REJECT_OWN_INK`). Then the **final polygons** of the accepted lines are compared pairwise
    (:func:`final_overlaps`): a region shared beyond ``--max-overlap`` of the smaller polygon is trimmed from the
    line farther from the shared ink, or that line is rejected when the region holds its own ink
    (:data:`REJECT_OVERLAP_TRIM`). A page left without an accepted line is skipped. ``--no-raise-top
    --block-slack inf --max-own-ink-outside inf --max-overlap inf`` switches off the round-2 bound, rebuild and
    gates (the masked walk, the neighbour clamps and the per-column end geometry stay); the segmenter's
    geometry with no gate is ``--no-extend-to-ink --max-ink-beyond inf --max-own-ink-outside inf --max-lost 1
    --long-read-words 0 --no-supralinear``.

Output (``--out-root``)::

    page_xml/genizah/<stem>.xml            one TextLine per segmenter line (merged fragments = one line);
                                           edition text on accepted lines, empty Unicode elsewhere;
                                           custom="alignment {tier:...; sim:...; len:...; edition:...;}"
    lists/genizah_train.lst, genizah_val.lst   absolute XML paths, split by canonical id (else PGP id), SHA-1;
                                           ``--split-by-page``: by page stem instead
    alignment_stats.json                   pages seen / aligned / skipped by reason, lines per tier, yield,
                                           ink-walk totals ("extension")
    pairs.jsonl                            one row per matched pair: page, line id, tier, reason, lost,
                                           extended px and ink beyond per end, absorbed segments,
                                           own_ink_outside, overlap, overlap_final, trimmed, end shifts,
                                           hard limits, spacing, detected neighbours, missing words,
                                           supralinear merge, text block bound, text
    gate/genizah_alignment/index.html      audit sheet: seeded sample of accepted + rejected line crops
                                           (1.5 line heights of margin, polygon outlines before / after)
    gate/genizah_alignment/pages.jsonl     per-page alignment summary

Usage (repo root)::

    .venv/bin/python -m src.finetuning.kraken.align_edition_lines \\
        --pages-dir /Volumes/home/studio_offload/datasets/kraken_arabic_lines/edition_pages/ppocrv6m__ktivseg_e10 \\
        --editions /Volumes/home/studio_offload/datasets/pgp_arabic_editions_v2 \\
        --out-root /Volumes/home/studio_offload/datasets/kraken_arabic_lines/v1 [--limit N] [--sim-anchor 0.5]

Strict geometry (round 2): add ``--max-lost 0.0 --drop-count-match --drop-interpolated --drop-duplicate-texts``
(extension, vertical rebuild, text block bound and the overlap / own-ink gates are on by default).

Compiling for training (``finetune_arabic.sh``): add ``genizah`` to ``SETS`` and mount the page images,
which live outside the data root, read-only:
``EXTRA_RO+=/Volumes/home/studio_offload/datasets/pgp_arabic_editions_v2_images``. Every PAGE file's
``imageFilename`` points there; without the mount ``compile`` stops with "image ... is not visible in the
container".
"""
import argparse
import dataclasses
import datetime
import html
import json
import logging
import math
import random
import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from lxml import etree
from PIL import Image, ImageDraw

from src.datasets.evaluations.arabic_script import normalize_arabic
from src.datasets.evaluations.benchmark_registry import registered_benchmark_documents
from src.finetuning.kraken import line_extent, page_layout
from src.finetuning.kraken.build_arabic_lines import arabic_letter_count, clean_target, hash_split

logger = logging.getLogger(__name__)

PAGES_DIR = Path("/Volumes/home/studio_offload/datasets/kraken_arabic_lines/edition_pages/ppocrv6m__ktivseg_e10")
EDITIONS_DIR = Path("/Volumes/home/studio_offload/datasets/pgp_arabic_editions_v2")
OUT_ROOT = Path("/Volumes/home/studio_offload/datasets/kraken_arabic_lines/v1")
SET_NAME = "genizah"
PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
GAP_TOKEN = "[...]"

# ----------------------------------------------------------------------------- thresholds (CLI-overridable)
SIM_ANCHOR = 0.5            # text sim of an anchor
ANCHOR_MIN_LETTERS = 6      # Arabic letters an anchor's read needs
SIM_PAGE = 0.15             # wrong-side guard: mean sim of readable pairs on a count-match page ...
SIM_MARGIN = 0.07           # ... and that mean minus the same reads' mean sim to the page's OTHER edition lines
GUARD_MIN_PAIRS = 3         # ... over at least this many readable pairs
COVER_MIN_LETTERS = 6       # span evidence (coverage, contradiction, neighbour preference) needs a read this long
COVER_WORD_SHARE = 0.5      # a span of an edition line covers at least this share of its words
COVER_GAIN = 0.15           # partial segment: the best span beats the whole line by at least this ...
COVER_LOST = 0.25           # ... and leaves at least this share of the edition line's characters outside the span
CONTRADICT_MARGIN = 0.1     # a read whose best span sim to ANOTHER edition line beats its own by this much ...
CONTRADICT_MIN_SIM = 0.5    # ... and reaches this, contradicts its placement
NEIGHBOR_MARGIN = 0.05      # a non-anchor read closer by this much to edition line e-1 or e+1 (span sims) vetoes too
NO_ANCHOR_MARGIN_BUFFER = 0.02  # a count match without anchors needs margin >= SIM_MARGIN + this ...
NO_ANCHOR_OWN_BEST = 2      # ... or this many readable lines per column whose best whole-line match is their own line
WIDE_LINE_SHARE = 0.6       # block order: lines narrower than this share of the text width may form side-by-side blocks
MIN_BLOCK_LINES = 2         # block order: lines needed on each side of a block gutter
LEN_SHORT_MIN = 0.8         # local length guard on count-match / interpolated lines: width x local cpp / edition
LEN_LONG_MAX = 1.5          # characters within [LEN_SHORT_MIN, LEN_LONG_MAX] (a too-short segment truncates the line)
TEXT_MIN_LETTERS = 4        # a read with fewer letters carries no text evidence
TEXT_WEIGHT = 0.6           # pair score = TEXT_WEIGHT * sim + LENGTH_WEIGHT * length score (readable read)
LENGTH_WEIGHT = 0.4
SHORT_READ_WEIGHT = 0.5     # pair score = SHORT_READ_WEIGHT * length score (unreadable read)
SKIP_S_COST = 0.05          # skipping a segmenter line whose read is short or empty
SKIP_S_LONG_COST = 0.4      # skipping a segmenter line whose read holds >= SKIP_S_LONG_LETTERS letters
SKIP_S_LONG_LETTERS = 8
SKIP_E_COST = 0.3           # an edition line the segmenter did not find
MERGE_DY = 0.5              # fragments: |baseline mean y difference| < MERGE_DY * median line height
MERGE_X_OVERLAP = 0.2       # fragments: x-range overlap at most this share of the narrower part
MAX_MERGE = 3               # at most this many fragments make one line
COLUMN_GAP = 0.25           # x-centre gap (share of page width) that makes two columns
GUTTER_MIN_SHARE = 0.02     # a gutter no baseline crosses must be at least this share of the page width
MIN_COLUMN_LINES = 3        # each column needs this many lines
STEEP_DEGREES = 45.0        # a baseline steeper than this is a rotated (marginal) line, ordered last
NOISE_WIDTH_SHARE = 0.15    # noise: baseline shorter than this share of the page's median baseline ...
NOISE_MAX_LETTERS = 1       # ... and a read of at most this many letters
MIN_REFIT_ANCHORS = 3
REFIT_ROUNDS = 2
MIN_TARGET_LINES = 2        # a page with fewer cleaned target lines is skipped
MIN_EDITION_LETTERS = 2     # an edition line with fewer letters (a lone "[...]") is left out of the alignment
VAL_SHARE = 0.08
AUDIT_N = 40
AUDIT_REJECTED = 10
AUDIT_SEED = 1019
AUDIT_PAGE_CAP = 2          # audit sheet: at most this many lines per (page, tier) unless a tier runs short
AUDIT_MAX_WIDTH = 1600
AUDIT_PAD = 12
AUDIT_MARGIN = 1.5          # audit crops: horizontal margin in line heights beyond the polygon
MAX_LOST = 0.25             # lost guard (every tier): a read whose best span leaves more than this share out
MAX_INK_BEYOND = 0.4        # geometry gate: ink still continuing beyond an end by more than this many line heights
MAX_OVERLAP = 0.05          # geometry gate: the polygon newly covers more than this share of another line's polygon
MAX_OWN_INK_OUTSIDE = 0.15  # geometry gate: more than this share of the band's ink (in the text block) is outside
MAX_END_SHIFT = 0.5         # geometry gate: detached ink an end took sits this many line heights off the line's own
LONG_READ_WORDS = 3         # text guard: a read with this many more words than its edition line ...
LONG_READ_SPAN = 0.5        # ... or with more words and a best span sim of at most this is much longer
LONG_READ_CHARS = 1.15      # ... the word count needs the read's skeleton letters >= this x the edition line's
                            # (zero-shot reads split words: more words alone are no evidence)
MISSING_MIN_LETTERS = 3     # supralinear: an edition word of at least this many skeleton letters ...
MISSING_SHARE = 0.8         # ... with at least this share of them aligned to nothing in the read is missing
SUPRA_MIN = 0.3             # supralinear: a segment / ink just above the line, between this many spacings ...
SUPRA_MAX = 0.8             # ... and this many above its baseline ...
SUPRA_WIDTH = 0.5           # ... a segment no wider than this share of the line ...
SUPRA_NEAR = 1.5            # ... ink within this many line heights of the missing word's estimated position
END_READ_SIM = 0.34         # an absorbable end box's read must reach this sim to one of the edition line's end words
FILLER_ENDINGS = (":", "׃")  # an edition line ending in one of these takes no detached glyph at its (left) end

ACCEPTED_TIERS = ("anchor", "count_match", "interpolated")
TIER_REJECTED = "rejected"
TIER_UNMATCHED = "unmatched"
TIER_NOISE = "noise"

REJECT_LENGTH = "length guard"
REJECT_UNEQUAL = "between anchors, unequal counts"
REJECT_OUTSIDE = "outside the anchor span"
REJECT_NO_ANCHORS = "no anchor on the path"
REJECT_GUARD = "identity path failed the wrong-side guard"
REJECT_CONTRADICTED = "a read in its stretch prefers another edition line"
REJECT_NEIGHBOR = "a read in its stretch prefers a neighbouring edition line"
REJECT_NO_SUPPORT = "count match without anchors lacks per-column support"
REJECT_PARTIAL = "segment covers only part of the edition line"
REJECT_LOST = "read leaves part of the edition line out (lost guard)"
REJECT_COUNT_MATCH_DROPPED = "count match dropped"
REJECT_INK_BEYOND = "ink beyond segment end"
REJECT_STEAL = "extension reaches another edition line's segment"
REJECT_OVERLAP = "extension overlaps another line"
REJECT_OWN_INK = "own ink outside the polygon"
REJECT_INTERPOLATED_DROPPED = "interpolated dropped"
REJECT_DUPLICATE = "edition text repeated on the page"
REJECT_LONG_READ = "read much longer than edition line"
REJECT_MISSING_WORD = "edition word not found"
REJECT_END_SHIFT = "end box leaves the line band"
REJECT_OVERLAP_TRIM = "overlap trim would cut own ink"

SKIP_REGISTRY = "benchmark registry"
SKIP_NO_EDITION = "no edition text"
SKIP_FEW_TARGETS = f"fewer than {MIN_TARGET_LINES} target lines"
SKIP_NO_LINES = "no segmenter lines"
SKIP_NO_ANCHORS = "no anchors and no count match"
SKIP_NO_ACCEPTED = "no accepted target line"
SKIP_IMAGE_SIZE = "image size mismatch"

# ----------------------------------------------------------------------------- skeleton

_SKELETON_CLASSES = {
    "ب": "ب", "ت": "ب", "ث": "ب", "ٮ": "ب",                       # beh class (incl. dotless beh U+066E)
    "ن": "ن", "ں": "ن",                                             # noon (incl. dotless noon)
    "ي": "ي", "ى": "ي", "ئ": "ي", "ی": "ي", "ے": "ي",              # yeh, alef maqsura, yeh-hamza
    "ج": "ح", "ح": "ح", "خ": "ح",
    "د": "د", "ذ": "د",
    "ر": "ر", "ز": "ر",
    "س": "س", "ش": "س",
    "ص": "ص", "ض": "ص",
    "ط": "ط", "ظ": "ط",
    "ع": "ع", "غ": "ع",
    "ف": "ف", "ق": "ف", "ڡ": "ف", "ٯ": "ف",                       # feh / qaf (incl. dotless forms)
    "ه": "ه", "ة": "ه", "ۀ": "ه", "ہ": "ه", "ە": "ه",
    "ا": "ا", "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "و": "و", "ؤ": "و",
    "ك": "ك", "ک": "ك",
    "ل": "ل", "م": "م",
}
_SKELETON_DROP = re.compile(r"[ً-ٰٟـ]")
_EDITION_MARKS = re.compile(r"\[\.\.\.\]|[\[\]<>{}()]")


def skeleton(text: str) -> str:
    """Rasm skeleton of a text: the letter classes that look alike without dots, spaces kept.

    ب/ت/ث (and dotless ٮ) are one class, ن its own, ي/ى/ئ one; ج/ح/خ, د/ذ, ر/ز, س/ش, ص/ض, ط/ظ, ع/غ, ف/ق
    are folded; ة becomes ه, hamza-on-alef forms become ا, ؤ becomes و; vowel marks, tatweel, a lone
    hamza and everything that is not an Arabic letter (Hebrew letters, digits, punctuation) are dropped.

    :param text: Any text.
    :type text: str
    :return: Skeleton letters with single spaces between words, stripped.
    :rtype: str
    """
    norm = _SKELETON_DROP.sub("", normalize_arabic(text))
    out = [_SKELETON_CLASSES.get(ch, " " if ch.isspace() else "") for ch in norm]
    return " ".join("".join(out).split())


def edition_skeleton(line: str) -> str:
    """Skeleton of an edition line without its gap tokens and editorial brackets.

    :param line: One edition line (may hold ``[...]``).
    :type line: str
    :return: :func:`skeleton` of the visible text.
    :rtype: str
    """
    return skeleton(_EDITION_MARKS.sub(" ", line))


def visible_text(line: str) -> str:
    """What an edition line puts on the page, for the length model.

    :param line: One edition line (may hold ``[...]``).
    :type line: str
    :return: The line without gap tokens and editorial brackets, presentation forms unfolded, vowel marks
        and tatweel removed, whitespace collapsed; letters of every script, digits and punctuation kept.
    :rtype: str
    """
    return " ".join(normalize_arabic(_EDITION_MARKS.sub(" ", line)).split())


def levenshtein(a: str, b: str) -> int:
    """Edit distance (unit insert / delete / substitute costs).

    :param a: First string.
    :type a: str
    :param b: Second string.
    :type b: str
    :return: Minimum number of edits turning ``a`` into ``b``.
    :rtype: int
    """
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def similarity(a: str, b: str) -> float:
    """``1 - levenshtein / max length``.

    :param a: First (skeleton) string.
    :type a: str
    :param b: Second (skeleton) string.
    :type b: str
    :return: Similarity in [0, 1]; 0.0 when either string is empty (no evidence).
    :rtype: float
    """
    if not a or not b:
        return 0.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))


def prefix_distances(pattern: str, text: str) -> List[int]:
    """Edit distance of ``pattern`` to every prefix of ``text`` in one pass (Myers / Hyyrö bit-parallel, global).

    Equal to ``[levenshtein(pattern, text[:j]) for j in range(len(text) + 1)]`` but O(len(text)) integer
    operations instead of a full table per prefix.

    :param pattern: The string held fixed (the zero-shot read skeleton).
    :type pattern: str
    :param text: The string whose prefixes are compared (an edition skeleton from some word on).
    :type text: str
    :return: Distances for prefix lengths ``0 .. len(text)``.
    :rtype: List[int]
    """
    m = len(pattern)
    if m == 0:
        return list(range(len(text) + 1))
    peq: Dict[str, int] = {}
    for i, ch in enumerate(pattern):
        peq[ch] = peq.get(ch, 0) | (1 << i)
    full = (1 << m) - 1
    high = 1 << (m - 1)
    pv, mv, score = full, 0, m
    out = [m]
    for ch in text:
        eq = peq.get(ch, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | (~(xh | pv) & full)
        mh = pv & xh
        if ph & high:
            score += 1
        elif mh & high:
            score -= 1
        ph = ((ph << 1) | 1) & full
        mh = (mh << 1) & full
        pv = mh | (~(xv | ph) & full)
        mv = ph & xv
        out.append(score)
    return out


def span_similarity(read: str, line: str, min_word_share: float = COVER_WORD_SHARE) -> Tuple[float, float]:
    """Best :func:`similarity` of a read to a contiguous word span of an edition line.

    Only spans of at least ``ceil(min_word_share * words)`` words are tried, the whole line included, so the
    result is never below ``similarity(read, line)``. A segment that holds only part of its edition line
    reads like a sub-span of it (the partial-segment test), and a read of half of a neighbouring line still
    scores high against that line (the contradiction test).

    :param read: Read skeleton.
    :type read: str
    :param line: Edition line skeleton (single spaces between words).
    :type line: str
    :param min_word_share: Smallest share of the line's words a span covers.
    :type min_word_share: float
    :return: ``(sim, lost)``: the best span's similarity and the share of the line's characters outside it;
        ties go to the longer span; ``(0.0, 0.0)`` when either string is empty.
    :rtype: Tuple[float, float]
    """
    if not read or not line:
        return 0.0, 0.0
    words = line.split(" ")
    starts: List[int] = []
    ends: List[int] = []
    pos = 0
    for word in words:
        starts.append(pos)
        pos += len(word)
        ends.append(pos)
        pos += 1
    n = len(words)
    need = min(n, max(1, math.ceil(min_word_share * n - 1e-9)))
    total = len(line)
    best_sim, best_lost = -1.0, 1.0
    for a in range(n - need + 1):
        dist = prefix_distances(read, line[starts[a]:])
        for b in range(a + need - 1, n):
            span = ends[b] - starts[a]
            sim = 1.0 - dist[span] / max(len(read), span)
            lost = (total - span) / total
            if sim > best_sim + 1e-12 or (sim > best_sim - 1e-12 and lost < best_lost):
                best_sim, best_lost = sim, lost
    return best_sim, best_lost


def missing_words(read: str, line: str, min_letters: int = MISSING_MIN_LETTERS, share: float = MISSING_SHARE
                  ) -> List[Tuple[int, float]]:
    """Edition words that the read has no counterpart for (a word written above the line, outside the segment).

    The read and the edition line are aligned by a global Levenshtein alignment (diagonal moves preferred on
    ties, then deletions of edition characters). An edition word of at least ``min_letters`` letters whose
    letters are at least ``share`` deletions is missing: substitutions (a misread word) do not count.

    :param read: Read skeleton.
    :type read: str
    :param line: Edition line skeleton (single spaces between words).
    :type line: str
    :param min_letters: Shorter words are never reported.
    :type min_letters: int
    :param share: Share of a word's letters that must be deletions.
    :type share: float
    :return: ``[(word index, centre position along the line, 0 = start .. 1 = end)]``; empty when either
        string is empty.
    :rtype: List[Tuple[int, float]]
    """
    if not read or not line:
        return []
    n, m = len(line), len(read)
    dist = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dist[i][0] = i
    for j in range(m + 1):
        dist[0][j] = j
    for i in range(1, n + 1):
        row, prev, ci = dist[i], dist[i - 1], line[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (ci != read[j - 1]))
    deleted = [False] * n
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dist[i][j] == dist[i - 1][j - 1] + (line[i - 1] != read[j - 1]):
            i, j = i - 1, j - 1
        elif i > 0 and dist[i][j] == dist[i - 1][j] + 1:
            deleted[i - 1] = True
            i -= 1
        else:
            j -= 1
    out: List[Tuple[int, float]] = []
    pos = 0
    for k, word in enumerate(line.split(" ")):
        span = range(pos, pos + len(word))
        if len(word) >= min_letters and sum(deleted[c] for c in span) >= share * len(word):
            out.append((k, (pos + len(word) / 2) / n))
        pos += len(word) + 1
    return out


def long_read(pair: "Pair", params: "Params") -> bool:
    """Whether a read is much longer than its edition line (it holds another line's words too).

    :param pair: Pair with :attr:`Pair.read_words`, :attr:`Pair.edition_words` and span evidence.
    :type pair: Pair
    :param params: Thresholds (``long_read_words``, ``long_read_span``, ``cover_min_letters``).
    :type params: Params
    :return: True for a read of at least ``cover_min_letters`` letters with ``long_read_words`` more words than
        the edition line and at least :data:`LONG_READ_CHARS` times its skeleton letters, or with more words and
        a best span sim of at most ``long_read_span``; always False with ``long_read_words`` 0.
    :rtype: bool
    """
    if params.long_read_words <= 0 or pair.letters < params.cover_min_letters:
        return False
    extra = pair.read_words - pair.edition_words
    longer = pair.read_chars >= LONG_READ_CHARS * pair.edition_chars
    return (extra >= params.long_read_words and longer) or (extra >= 1 and pair.own_span <= params.long_read_span)


# ----------------------------------------------------------------------------- records

Point = Tuple[float, float]


@dataclass(frozen=True)
class SegLine:
    """One segmenter line with the geometry the alignment needs.

    :param index: Index in the segmenter output.
    :param baseline: Baseline points (full-resolution pixels).
    :param boundary: Boundary polygon points.
    :param text: Zero-shot read (steers placement only).
    :param confidence: Mean read confidence (None when the read is empty).
    :param x_min: Smallest baseline x.
    :param x_max: Largest baseline x.
    :param y_mean: Mean baseline y.
    :param width: Baseline polyline length in pixels.
    :param height: Boundary polygon height (y extent).
    :param angle: Baseline angle from the horizontal, degrees in [0, 90].
    :param letters: Arabic letters in the read.
    :param skeleton: :func:`skeleton` of the read.
    """
    index: int
    baseline: Tuple[Point, ...]
    boundary: Tuple[Point, ...]
    text: str
    confidence: Optional[float]
    x_min: float
    x_max: float
    y_mean: float
    width: float
    height: float
    angle: float
    letters: int
    skeleton: str

    @property
    def x_center(self) -> float:
        """Middle of the baseline's x-range.

        :return: ``(x_min + x_max) / 2``.
        :rtype: float
        """
        return (self.x_min + self.x_max) / 2


@dataclass(frozen=True)
class EditionLine:
    """One edition line of a page.

    :param index: Line number in the page's edition text (0-based, empty lines not counted).
    :param raw: The line as the edition dataset stores it.
    :param target: :func:`clean_target` text; ``""`` when the line is not a training target.
    :param skeleton: :func:`edition_skeleton` of the line.
    :param visible: :func:`visible_text` of the line.
    """
    index: int
    raw: str
    target: str
    skeleton: str
    visible: str = ""

    @property
    def length(self) -> int:
        """Visible length (letters of any script, digits, spaces), the unit of the length model.

        Digits and Hebrew words take room on the page although the skeleton drops them.

        :return: ``len(visible)``.
        :rtype: int
        """
        return len(self.visible)


@dataclass(frozen=True)
class Step:
    """One move of an alignment path.

    :param kind: ``"match"`` (1:1 or a merge), ``"skip_s"`` or ``"skip_e"``.
    :param e: Edition line position (None for ``skip_s``).
    :param s: Segmenter line positions in the ordered list (empty for ``skip_e``).
    """
    kind: str
    e: Optional[int]
    s: Tuple[int, ...]


@dataclass
class Pair:
    """A matched edition line and its segmenter line(s), with the evidence.

    :param step: Index of the path step.
    :param e: Edition line position.
    :param s: Ordered segmenter positions.
    :param sim: Text similarity of the (merged) read and the edition line.
    :param len_score: Length score under the page's final cpp.
    :param letters: Arabic letters in the (merged) read.
    :param off_sim: Mean whole-line sim of the read to the page's other edition lines (the guard's control).
    :param best_other: Highest :func:`span_similarity` of the read to any other edition line of the page
        (reads of at least ``cover_min_letters`` letters; else 0.0).
    :param best_other_line: Highest whole-line sim of the read to any other edition line (readable reads).
    :param width: Baseline length of the (merged) segment in pixels.
    :param length: Visible length of the edition line (:attr:`EditionLine.length`).
    :param cover_sim: :func:`span_similarity` of the read to its own edition line (0.0 when not computed).
    :param cover_lost: Share of the own edition line's characters outside that best span.
    :param neighbor_sim: Highest span sim of the read to edition line ``e - 1`` or ``e + 1``.
    :param column: Column of the segment under the alignment's ordering (0 for one column).
    :param anchored: Position anchor (sim >= ``sim_anchor`` on a read of >= ``anchor_min_letters`` letters),
        whether or not its target is accepted.
    :param partial: The segment covers only part of its edition line (:func:`is_partial`).
    :param len_ratio: ``width x local cpp / length`` checked by the local length guard (0.0 when unchecked).
    :param tier: Acceptance tier.
    :param reason: Why a rejected pair was rejected (``REJECT_*``; ``""`` otherwise).
    :param read_words: Words in the (merged) read's skeleton.
    :param edition_words: Words in the edition line's skeleton.
    :param read_chars: Letters in the (merged) read's skeleton (spaces left out).
    :param edition_chars: Letters in the edition line's skeleton.
    :param missing: Edition words the read has no counterpart for (:func:`missing_words`): ``(word index,
        position along the line from its start, 0..1)``.
    """
    step: int
    e: int
    s: Tuple[int, ...]
    sim: float
    len_score: float
    letters: int
    off_sim: float = 0.0
    best_other: float = 0.0
    best_other_line: float = 0.0
    width: float = 0.0
    length: int = 0
    cover_sim: float = 0.0
    cover_lost: float = 0.0
    neighbor_sim: float = 0.0
    column: int = 0
    anchored: bool = False
    partial: bool = False
    len_ratio: float = 0.0
    tier: str = TIER_REJECTED
    reason: str = ""
    read_words: int = 0
    edition_words: int = 0
    read_chars: int = 0
    edition_chars: int = 0
    missing: Tuple[Tuple[int, float], ...] = ()

    @property
    def own_span(self) -> float:
        """Span similarity of the read to its own edition line (never below the whole-line sim).

        :return: ``max(sim, cover_sim)``.
        :rtype: float
        """
        return max(self.sim, self.cover_sim)


@dataclass
class Params:
    """Every threshold of the alignment (CLI-overridable).

    :param sim_anchor: :data:`SIM_ANCHOR`.
    :param anchor_min_letters: :data:`ANCHOR_MIN_LETTERS`.
    :param sim_page: :data:`SIM_PAGE`.
    :param sim_margin: :data:`SIM_MARGIN`.
    :param guard_min_pairs: :data:`GUARD_MIN_PAIRS`.
    :param cover_min_letters: :data:`COVER_MIN_LETTERS`.
    :param cover_word_share: :data:`COVER_WORD_SHARE`.
    :param cover_gain: :data:`COVER_GAIN`.
    :param cover_lost: :data:`COVER_LOST`.
    :param contradict_margin: :data:`CONTRADICT_MARGIN`.
    :param contradict_min_sim: :data:`CONTRADICT_MIN_SIM`.
    :param neighbor_margin: :data:`NEIGHBOR_MARGIN`.
    :param no_anchor_margin_buffer: :data:`NO_ANCHOR_MARGIN_BUFFER`.
    :param no_anchor_own_best: :data:`NO_ANCHOR_OWN_BEST`.
    :param wide_line_share: :data:`WIDE_LINE_SHARE`.
    :param min_block_lines: :data:`MIN_BLOCK_LINES`.
    :param len_short_min: :data:`LEN_SHORT_MIN`.
    :param len_long_max: :data:`LEN_LONG_MAX`.
    :param text_min_letters: :data:`TEXT_MIN_LETTERS`.
    :param text_weight: :data:`TEXT_WEIGHT`.
    :param length_weight: :data:`LENGTH_WEIGHT`.
    :param short_read_weight: :data:`SHORT_READ_WEIGHT`.
    :param skip_s_cost: :data:`SKIP_S_COST`.
    :param skip_s_long_cost: :data:`SKIP_S_LONG_COST`.
    :param skip_s_long_letters: :data:`SKIP_S_LONG_LETTERS`.
    :param skip_e_cost: :data:`SKIP_E_COST`.
    :param merge_dy: :data:`MERGE_DY`.
    :param merge_x_overlap: :data:`MERGE_X_OVERLAP`.
    :param max_merge: :data:`MAX_MERGE`.
    :param column_gap: :data:`COLUMN_GAP`.
    :param gutter_min_share: :data:`GUTTER_MIN_SHARE`.
    :param min_column_lines: :data:`MIN_COLUMN_LINES`.
    :param steep_degrees: :data:`STEEP_DEGREES`.
    :param noise_width_share: :data:`NOISE_WIDTH_SHARE`.
    :param noise_max_letters: :data:`NOISE_MAX_LETTERS`.
    :param min_refit_anchors: :data:`MIN_REFIT_ANCHORS`.
    :param refit_rounds: :data:`REFIT_ROUNDS`.
    :param min_target_lines: :data:`MIN_TARGET_LINES`.
    :param min_edition_letters: :data:`MIN_EDITION_LETTERS`.
    :param val_share: :data:`VAL_SHARE`.
    :param audit_n: :data:`AUDIT_N`.
    :param audit_rejected: :data:`AUDIT_REJECTED`.
    :param audit_seed: :data:`AUDIT_SEED`.
    :param audit_page_cap: :data:`AUDIT_PAGE_CAP`.
    :param max_lost: :data:`MAX_LOST`; the lost guard rejects a pair of any tier (merged segments included)
        whose read leaves more than this share of its edition line outside its best span (0.0: strict).
    :param drop_count_match: Never accept the ``count_match`` tier (a count-match line between two anchors
        falls back to ``interpolated``).
    :param extend_to_ink: Extend every accepted line's polygon to the ink beyond its ends
        (:mod:`~src.finetuning.kraken.line_extent`).
    :param max_ink_beyond: :data:`MAX_INK_BEYOND`; ``inf`` disables the geometry gate.
    :param band_above: :data:`~src.finetuning.kraken.line_extent.BAND_ABOVE`.
    :param band_below: :data:`~src.finetuning.kraken.line_extent.BAND_BELOW`.
    :param extend_gap: :data:`~src.finetuning.kraken.line_extent.EXTEND_GAP`.
    :param extend_max: :data:`~src.finetuning.kraken.line_extent.EXTEND_MAX`.
    :param audit_margin: :data:`AUDIT_MARGIN`.
    :param split_by_page: Split by page stem instead of by document (:func:`split_key`), for a small set whose
        few documents leave the val list empty; pages of one document can then land in both splits.
    :param raise_top: Whole-line vertical rebuild (every column covers the band, clamped to the neighbouring
        lines); effective only with ``extend_to_ink``.
    :param block_slack: :data:`~src.finetuning.kraken.line_extent.BLOCK_SLACK`; ``inf`` removes the text
        block bound.
    :param block_thirds: Estimate the text block per vertical third of the page (slanted blocks).
    :param neighbour_clamp: :data:`~src.finetuning.kraken.line_extent.NEIGHBOUR_CLAMP`.
    :param max_overlap: :data:`MAX_OVERLAP`; ``inf`` disables the overlap gate.
    :param max_own_ink_outside: :data:`MAX_OWN_INK_OUTSIDE`; ``inf`` disables the own-ink gate.
    :param drop_interpolated: Never accept the ``interpolated`` tier.
    :param drop_duplicate_texts: Reject every edition line whose NFC target occurs more than once on its page
        (:func:`duplicate_positions`).
    :param spacing_above: :data:`~src.finetuning.kraken.line_extent.SPACING_ABOVE` (``inf``: the round-2 raise).
    :param spacing_below: :data:`~src.finetuning.kraken.line_extent.SPACING_BELOW`.
    :param detect_unsegmented: Detect unsegmented neighbour lines from the ink row profile.
    :param foreign_components: Components nearer another baseline are never a line's ink.
    :param pencil_share: :data:`~src.finetuning.kraken.line_extent.PENCIL_SHARE` (0: off).
    :param stub_width: :data:`~src.finetuning.kraken.line_extent.STUB_WIDTH` (0: off).
    :param own_down_cap: :data:`~src.finetuning.kraken.line_extent.OWN_DOWN_CAP`.
    :param own_ink_reach: :data:`~src.finetuning.kraken.line_extent.OWN_INK_REACH` (``inf``: the whole block).
    :param column_blocks: Text blocks per column group with margin material and gutters
        (:func:`~src.finetuning.kraken.page_layout.page_blocks`); off: the round-2 page-wide block.
    :param gutters: Detect gutters (with ``column_blocks``).
    :param block_start_percentile: :data:`~src.finetuning.kraken.page_layout.START_PERCENTILE`: a column
        block's right edge, where right-to-left lines start (95: the round-2 percentile).
    :param max_end_shift: :data:`MAX_END_SHIFT`; ``inf`` disables the end-shift gate.
    :param long_read_words: :data:`LONG_READ_WORDS` (0: the long-read guard is off).
    :param long_read_span: :data:`LONG_READ_SPAN`.
    :param supralinear: Look for a supralinear insertion when an edition word is missing from the read
        (:func:`missing_words`), else reject the line (:data:`REJECT_MISSING_WORD`).
    :param end_checks: An absorbable end box's read must match the edition line's end words, and an edition line
        ending in ':' takes no detached glyph at its end.
    """
    sim_anchor: float = SIM_ANCHOR
    anchor_min_letters: int = ANCHOR_MIN_LETTERS
    sim_page: float = SIM_PAGE
    sim_margin: float = SIM_MARGIN
    guard_min_pairs: int = GUARD_MIN_PAIRS
    cover_min_letters: int = COVER_MIN_LETTERS
    cover_word_share: float = COVER_WORD_SHARE
    cover_gain: float = COVER_GAIN
    cover_lost: float = COVER_LOST
    contradict_margin: float = CONTRADICT_MARGIN
    contradict_min_sim: float = CONTRADICT_MIN_SIM
    neighbor_margin: float = NEIGHBOR_MARGIN
    no_anchor_margin_buffer: float = NO_ANCHOR_MARGIN_BUFFER
    no_anchor_own_best: int = NO_ANCHOR_OWN_BEST
    wide_line_share: float = WIDE_LINE_SHARE
    min_block_lines: int = MIN_BLOCK_LINES
    len_short_min: float = LEN_SHORT_MIN
    len_long_max: float = LEN_LONG_MAX
    text_min_letters: int = TEXT_MIN_LETTERS
    text_weight: float = TEXT_WEIGHT
    length_weight: float = LENGTH_WEIGHT
    short_read_weight: float = SHORT_READ_WEIGHT
    skip_s_cost: float = SKIP_S_COST
    skip_s_long_cost: float = SKIP_S_LONG_COST
    skip_s_long_letters: int = SKIP_S_LONG_LETTERS
    skip_e_cost: float = SKIP_E_COST
    merge_dy: float = MERGE_DY
    merge_x_overlap: float = MERGE_X_OVERLAP
    max_merge: int = MAX_MERGE
    column_gap: float = COLUMN_GAP
    gutter_min_share: float = GUTTER_MIN_SHARE
    min_column_lines: int = MIN_COLUMN_LINES
    steep_degrees: float = STEEP_DEGREES
    noise_width_share: float = NOISE_WIDTH_SHARE
    noise_max_letters: int = NOISE_MAX_LETTERS
    min_refit_anchors: int = MIN_REFIT_ANCHORS
    refit_rounds: int = REFIT_ROUNDS
    min_target_lines: int = MIN_TARGET_LINES
    min_edition_letters: int = MIN_EDITION_LETTERS
    val_share: float = VAL_SHARE
    audit_n: int = AUDIT_N
    audit_rejected: int = AUDIT_REJECTED
    audit_seed: int = AUDIT_SEED
    audit_page_cap: int = AUDIT_PAGE_CAP
    max_lost: float = MAX_LOST
    drop_count_match: bool = False
    extend_to_ink: bool = True
    max_ink_beyond: float = MAX_INK_BEYOND
    band_above: float = line_extent.BAND_ABOVE
    band_below: float = line_extent.BAND_BELOW
    extend_gap: float = line_extent.EXTEND_GAP
    extend_max: float = line_extent.EXTEND_MAX
    audit_margin: float = AUDIT_MARGIN
    split_by_page: bool = False
    raise_top: bool = True
    block_slack: float = line_extent.BLOCK_SLACK
    block_thirds: bool = False
    neighbour_clamp: float = line_extent.NEIGHBOUR_CLAMP
    max_overlap: float = MAX_OVERLAP
    max_own_ink_outside: float = MAX_OWN_INK_OUTSIDE
    drop_interpolated: bool = False
    drop_duplicate_texts: bool = False
    spacing_above: float = line_extent.SPACING_ABOVE
    spacing_below: float = line_extent.SPACING_BELOW
    detect_unsegmented: bool = True
    foreign_components: bool = True
    pencil_share: float = line_extent.PENCIL_SHARE
    stub_width: float = line_extent.STUB_WIDTH
    own_down_cap: float = line_extent.OWN_DOWN_CAP
    own_ink_reach: float = line_extent.OWN_INK_REACH
    column_blocks: bool = True
    gutters: bool = True
    block_start_percentile: float = page_layout.START_PERCENTILE
    max_end_shift: float = MAX_END_SHIFT
    long_read_words: int = LONG_READ_WORDS
    long_read_span: float = LONG_READ_SPAN
    supralinear: bool = True
    end_checks: bool = True


@dataclass
class Alignment:
    """Best alignment of one page under one line ordering.

    :param ordering: Ordering name (``one_column``, ``two_columns_right_first``, ``two_columns_left_first``).
    :param order: Segmenter lines in that order (noise excluded).
    :param steps: Best path.
    :param pairs: Matched pairs with tiers.
    :param score: Path score.
    :param cpp: Final characters per baseline pixel.
    :param identity: Whether the path has no skip move (equal line counts after merging).
    :param page_sim: Mean sim of the pairs with a readable read (the wrong-side guard's statistic).
    :param sim_margin: ``page_sim`` minus the mean :attr:`Pair.off_sim` of the same pairs.
    :param guard_pairs: Number of pairs with a readable read.
    """
    ordering: str
    order: List[SegLine]
    steps: List[Step]
    pairs: List[Pair]
    score: float
    cpp: float
    identity: bool
    page_sim: float
    sim_margin: float = 0.0
    guard_pairs: int = 0

    def tier_counts(self) -> Counter:
        """Pairs per tier.

        :return: ``Counter`` of tiers.
        :rtype: Counter
        """
        return Counter(p.tier for p in self.pairs)

    def anchored(self) -> int:
        """Position anchors on the path (accepted or rejected as partial).

        :return: Number of pairs with :attr:`Pair.anchored`.
        :rtype: int
        """
        return sum(1 for p in self.pairs if p.anchored)


@dataclass
class PageResult:
    """What became of one page.

    :param stem: Page stem.
    :param pgpid: PGP document id.
    :param canonical_id: Canonical id (``""`` when unknown).
    :param image_path: Absolute image path.
    :param image_size: ``(width, height)`` of the full-resolution image.
    :param edition: Edition lines of the page.
    :param segments: All segmenter lines (noise included).
    :param noise: Indices (segmenter numbering) of noise lines.
    :param alignment: Chosen alignment (None for a skipped page).
    :param skip: Why nothing was written (``""`` when the page was written).
    :param split: ``"train"`` or ``"val"``.
    :param xml_path: Written PAGE XML.
    :param geometry: Line geometry after the ink walk (:func:`extend_pairs`), by path step of the pair; pairs
        without an entry keep their segmenter geometry.
    :param line_height: Median line height of the page used by the ink walk (0.0 before it ran).
    :param spacing: Baseline spacing of the page used by the raise (0.0 before the walk ran).
    :param notes: Per path step of a walked pair: what the aligner decided around the walk (``supralinear``:
        ``"box:<i>"``, ``"ink:<n>"`` or ``"missing"``; ``connected_only_end``; ``not_absorbable``: box keys).
    """
    stem: str
    pgpid: str
    canonical_id: str
    image_path: str
    image_size: Tuple[int, int]
    edition: List[EditionLine] = field(default_factory=list)
    segments: List[SegLine] = field(default_factory=list)
    noise: List[int] = field(default_factory=list)
    alignment: Optional[Alignment] = None
    skip: str = ""
    split: str = ""
    xml_path: Optional[Path] = None
    geometry: Dict[int, line_extent.LineExtent] = field(default_factory=dict)
    line_height: float = 0.0
    spacing: float = 0.0
    notes: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    def absorbed_segments(self) -> Set[int]:
        """Segmenter indices absorbed into a written line by the ink walk (not written on their own).

        :return: Indices.
        :rtype: Set[int]
        """
        return {k for p in self.accepted() if p.step in self.geometry
                for k in self.geometry[p.step].absorbed + self.geometry[p.step].forced}

    def accepted(self) -> List[Pair]:
        """Accepted pairs that carry a training target.

        :return: Pairs in an accepted tier whose edition line has a non-empty target.
        :rtype: List[Pair]
        """
        if self.alignment is None:
            return []
        return [p for p in self.alignment.pairs if p.tier in ACCEPTED_TIERS and self.edition[p.e].target]


# ----------------------------------------------------------------------------- geometry

def polyline_length(points: Sequence[Point]) -> float:
    """Length of a polyline.

    :param points: Points in order.
    :type points: Sequence[Point]
    :return: Sum of segment lengths (0.0 for fewer than two points).
    :rtype: float
    """
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def baseline_angle(points: Sequence[Point]) -> float:
    """Angle of a baseline's end-to-end chord from the horizontal.

    :param points: Baseline points.
    :type points: Sequence[Point]
    :return: Degrees in [0, 90]; 0.0 for fewer than two points.
    :rtype: float
    """
    if len(points) < 2:
        return 0.0
    dx = abs(points[-1][0] - points[0][0])
    dy = abs(points[-1][1] - points[0][1])
    return math.degrees(math.atan2(dy, dx))


def make_seg_line(raw: Dict[str, Any]) -> SegLine:
    """Build a :class:`SegLine` from one entry of a segmenter JSON's ``lines``.

    :param raw: ``{"index", "baseline", "boundary", "text", "mean_confidence"}``.
    :type raw: Dict[str, Any]
    :return: The line with its derived geometry and read statistics.
    :rtype: SegLine
    """
    baseline = tuple((float(x), float(y)) for x, y in raw["baseline"])
    boundary = tuple((float(x), float(y)) for x, y in raw["boundary"])
    anchor_pts = baseline or boundary or ((0.0, 0.0),)
    xs = [p[0] for p in anchor_pts]
    ys = [p[1] for p in anchor_pts]
    bys = [p[1] for p in boundary] or ys
    text = raw.get("text") or ""
    return SegLine(index=int(raw["index"]), baseline=baseline, boundary=boundary, text=text,
                   confidence=raw.get("mean_confidence"), x_min=min(xs), x_max=max(xs), y_mean=sum(ys) / len(ys),
                   width=polyline_length(baseline), height=max(bys) - min(bys), angle=baseline_angle(baseline),
                   letters=arabic_letter_count(text), skeleton=skeleton(text))


def median_width(lines: Sequence[SegLine]) -> float:
    """Median baseline length of a page's lines.

    :param lines: Segmenter lines.
    :type lines: Sequence[SegLine]
    :return: Median width (0.0 for no lines).
    :rtype: float
    """
    return statistics.median(l.width for l in lines) if lines else 0.0


def line_height(lines: Sequence[SegLine]) -> float:
    """Median line height of a page (boundary polygon y extent).

    :param lines: Segmenter lines.
    :type lines: Sequence[SegLine]
    :return: Median height, at least 1.0.
    :rtype: float
    """
    return max(1.0, statistics.median(l.height for l in lines)) if lines else 1.0


def is_noise(line: SegLine, page_median_width: float, params: Params) -> bool:
    """Whether a segmenter line is noise (a speck: tiny baseline, no read).

    :param line: Segmenter line.
    :type line: SegLine
    :param page_median_width: Median baseline length of the page.
    :type page_median_width: float
    :param params: Thresholds.
    :type params: Params
    :return: True for fewer than two baseline points, or a baseline under
        ``noise_width_share * page_median_width`` with at most ``noise_max_letters`` letters read.
    :rtype: bool
    """
    if len(line.baseline) < 2:
        return True
    return line.width < params.noise_width_share * page_median_width and line.letters <= params.noise_max_letters


def x_overlap(a: SegLine, b: SegLine) -> float:
    """Overlap of two lines' baseline x-ranges as a share of the narrower range.

    :param a: First line.
    :type a: SegLine
    :param b: Second line.
    :type b: SegLine
    :return: Overlap share in [0, 1].
    :rtype: float
    """
    overlap = min(a.x_max, b.x_max) - max(a.x_min, b.x_min)
    narrower = max(1.0, min(a.x_max - a.x_min, b.x_max - b.x_min))
    return max(0.0, overlap) / narrower


def mergeable(a: SegLine, b: SegLine, height: float, params: Params) -> bool:
    """Whether two lines are fragments of one physical line.

    :param a: First line.
    :type a: SegLine
    :param b: Second line.
    :type b: SegLine
    :param height: Median line height of the page.
    :type height: float
    :param params: Thresholds.
    :type params: Params
    :return: True when their baselines sit at the same height (``|dy| < merge_dy * height``) and their
        x-ranges overlap by at most ``merge_x_overlap``.
    :rtype: bool
    """
    return abs(a.y_mean - b.y_mean) < params.merge_dy * height and x_overlap(a, b) <= params.merge_x_overlap


def group_mergeable(order: Sequence[SegLine], start: int, size: int, height: float, params: Params) -> bool:
    """Whether ``order[start:start + size]`` may be merged into one line.

    :param order: Ordered segmenter lines.
    :type order: Sequence[SegLine]
    :param start: First position.
    :type start: int
    :param size: Number of lines (1 is always mergeable).
    :type size: int
    :param height: Median line height.
    :type height: float
    :param params: Thresholds.
    :type params: Params
    :return: True when every pair of the group is :func:`mergeable`.
    :rtype: bool
    """
    group = order[start:start + size]
    if len(group) != size:
        return False
    return all(mergeable(group[i], group[j], height, params) for i in range(size) for j in range(i + 1, size))


def merged_read(group: Sequence[SegLine]) -> str:
    """The reads of fragments joined in Arabic reading order (rightmost fragment first).

    :param group: Fragments.
    :type group: Sequence[SegLine]
    :return: Reads joined by spaces, by descending baseline x-centre.
    :rtype: str
    """
    return " ".join(l.text for l in sorted(group, key=lambda l: -l.x_center) if l.text.strip())


def merged_skeleton(group: Sequence[SegLine]) -> str:
    """Skeleton of the merged read.

    :param group: Fragments.
    :type group: Sequence[SegLine]
    :return: :func:`skeleton` of :func:`merged_read`.
    :rtype: str
    """
    return skeleton(merged_read(group))


def convex_hull(points: Iterable[Point]) -> List[Point]:
    """Convex hull (Andrew's monotone chain).

    :param points: Any points.
    :type points: Iterable[Point]
    :return: Hull vertices counter-clockwise (in y-up terms), no repeated first point; collinear points
        dropped; fewer than three distinct points are returned sorted.
    :rtype: List[Point]
    """
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def cross(o: Point, a: Point, b: Point) -> float:
        """Z component of ``(a - o) x (b - o)``.

        :param o: Origin.
        :param a: First point.
        :param b: Second point.
        :return: Cross product.
        """
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Point] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def merged_geometry(group: Sequence[SegLine]) -> Tuple[List[Point], List[Point]]:
    """Baseline and boundary of a (possibly merged) line.

    :param group: One line or its fragments.
    :type group: Sequence[SegLine]
    :return: ``(baseline, boundary)``: a single line keeps its own points; fragments get their
        baselines concatenated left to right (each oriented left to right) and the convex hull of their
        boundaries.
    :rtype: Tuple[List[Point], List[Point]]
    """
    if len(group) == 1:
        return list(group[0].baseline), list(group[0].boundary)
    baseline: List[Point] = []
    for line in sorted(group, key=lambda l: l.x_min):
        pts = list(line.baseline)
        if pts[0][0] > pts[-1][0]:
            pts.reverse()
        baseline.extend(pts)
    return baseline, convex_hull(p for line in group for p in line.boundary)


# ----------------------------------------------------------------------------- order

def column_split(lines: Sequence[SegLine], page_width: float, params: Params) -> Optional[float]:
    """Two-column split from a gap in the baseline x-centres.

    :param lines: Horizontal segmenter lines.
    :type lines: Sequence[SegLine]
    :param page_width: Page width in pixels.
    :type page_width: float
    :param params: Thresholds.
    :type params: Params
    :return: The x of the middle of the widest centre gap when it exceeds ``column_gap * page_width``
        and leaves at least ``min_column_lines`` lines on each side; else None.
    :rtype: Optional[float]
    """
    centres = sorted(l.x_center for l in lines)
    k = params.min_column_lines
    best: Optional[Tuple[float, float]] = None
    for i in range(k - 1, len(centres) - k):
        gap = centres[i + 1] - centres[i]
        if best is None or gap > best[0]:
            best = (gap, (centres[i] + centres[i + 1]) / 2)
    if best is None or best[0] <= params.column_gap * page_width:
        return None
    return best[1]


def gutter_split(lines: Sequence[SegLine], page_width: float, params: Params) -> Optional[float]:
    """Two-column split at a vertical gutter that no baseline crosses (e.g. the fold of a bifolium).

    :param lines: Horizontal segmenter lines.
    :type lines: Sequence[SegLine]
    :param page_width: Page width in pixels.
    :type page_width: float
    :param params: Thresholds.
    :type params: Params
    :return: The middle of the widest gap in the union of the baselines' x-ranges that is at least
        ``gutter_min_share * page_width`` wide and has ``min_column_lines`` lines on each side; else None.
    :rtype: Optional[float]
    """
    spans = sorted((l.x_min, l.x_max) for l in lines)
    if not spans:
        return None
    best: Optional[Tuple[float, float]] = None
    reach = spans[0][1]
    for x0, x1 in spans[1:]:
        if x0 > reach:
            left = sum(1 for l in lines if l.x_max <= reach)
            right = sum(1 for l in lines if l.x_min >= x0)
            gap = x0 - reach
            if (left >= params.min_column_lines and right >= params.min_column_lines
                    and gap >= params.gutter_min_share * page_width and (best is None or gap > best[0])):
                best = (gap, (reach + x0) / 2)
        reach = max(reach, x1)
    return None if best is None else best[1]


def order_lines(lines: Sequence[SegLine], split_x: Optional[float], right_first: bool, params: Params) -> List[SegLine]:
    """Reading order: columns, then baseline mean y; steep (rotated) lines last.

    :param lines: Segmenter lines (noise excluded).
    :type lines: Sequence[SegLine]
    :param split_x: Column boundary (None: one column).
    :type split_x: Optional[float]
    :param right_first: With two columns, read the right column first (Arabic); else the left one.
    :type right_first: bool
    :param params: Thresholds.
    :type params: Params
    :return: Ordered lines. Within a column ties in y go right to left.
    :rtype: List[SegLine]
    """
    flat = [l for l in lines if l.angle <= params.steep_degrees]
    steep = sorted((l for l in lines if l.angle > params.steep_degrees), key=lambda l: (l.y_mean, -l.x_center))

    def by_y(group: Iterable[SegLine]) -> List[SegLine]:
        """Lines sorted by baseline mean y, right to left on ties.

        :param group: Lines.
        :return: Sorted lines.
        """
        return sorted(group, key=lambda l: (l.y_mean, -l.x_center))

    if split_x is None:
        return by_y(flat) + steep
    right = by_y(l for l in flat if l.x_center >= split_x)
    left = by_y(l for l in flat if l.x_center < split_x)
    return (right + left if right_first else left + right) + steep


def block_gutter(lines: Sequence[SegLine], page_width: float, params: Params) -> Optional[float]:
    """Gutter between side-by-side blocks of short lines (e.g. two witness clauses under a full-width text).

    :param lines: Horizontal segmenter lines.
    :type lines: Sequence[SegLine]
    :param page_width: Page width in pixels.
    :type page_width: float
    :param params: Thresholds.
    :type params: Params
    :return: The middle of the widest gap in the union of the x-ranges of the lines narrower than
        ``wide_line_share`` of the text width, at least ``gutter_min_share * page_width`` wide with
        ``min_block_lines`` short lines on each side; else None.
    :rtype: Optional[float]
    """
    if not lines:
        return None
    text_width = max(l.x_max for l in lines) - min(l.x_min for l in lines)
    short = [l for l in lines if l.x_max - l.x_min < params.wide_line_share * text_width]
    return gutter_split(short, page_width, dataclasses.replace(params, min_column_lines=params.min_block_lines))


def block_order(lines: Sequence[SegLine], split_x: float, params: Params) -> List[SegLine]:
    """Reading order with side-by-side blocks: runs of lines between lines that cross the gutter are read
    right block first, then left block; steep lines last.

    :param lines: Segmenter lines (noise excluded).
    :type lines: Sequence[SegLine]
    :param split_x: Block gutter x.
    :type split_x: float
    :param params: Thresholds.
    :type params: Params
    :return: Ordered lines.
    :rtype: List[SegLine]
    """
    flat = sorted((l for l in lines if l.angle <= params.steep_degrees), key=lambda l: (l.y_mean, -l.x_center))
    steep = [l for l in order_lines(lines, None, True, params) if l.angle > params.steep_degrees]
    out: List[SegLine] = []
    run: List[SegLine] = []

    def flush() -> None:
        """Append the current run, right block first.

        :return: None.
        """
        out.extend([l for l in run if l.x_center >= split_x] + [l for l in run if l.x_center < split_x])
        run.clear()

    for line in flat:
        if line.x_min < split_x < line.x_max:
            flush()
            out.append(line)
        else:
            run.append(line)
    flush()
    return out + steep


def candidate_orders(lines: Sequence[SegLine], page_width: float, params: Params
                     ) -> Dict[str, Tuple[List[SegLine], Optional[float]]]:
    """Every ordering worth aligning: one column, two columns either way when a page-wide split exists,
    and side-by-side blocks when short lines leave a gutter (:func:`block_gutter`).

    :param lines: Segmenter lines (noise excluded).
    :type lines: Sequence[SegLine]
    :param page_width: Page width in pixels.
    :type page_width: float
    :param params: Thresholds.
    :type params: Params
    :return: ``{ordering name: (ordered lines, column split x or None)}``; the split is set for the
        two-column orderings only (the block ordering counts as one column).
    :rtype: Dict[str, Tuple[List[SegLine], Optional[float]]]
    """
    orders: Dict[str, Tuple[List[SegLine], Optional[float]]] = {
        "one_column": (order_lines(lines, None, True, params), None)}
    flat = [l for l in lines if l.angle <= params.steep_degrees]
    split = column_split(flat, page_width, params)
    if split is None:
        split = gutter_split(flat, page_width, params)
    if split is not None:
        orders["two_columns_right_first"] = (order_lines(lines, split, True, params), split)
        orders["two_columns_left_first"] = (order_lines(lines, split, False, params), split)
    block_x = block_gutter(flat, page_width, params)
    if block_x is not None:
        blocks = block_order(lines, block_x, params)
        if all([l.index for l in blocks] != [l.index for l in o] for o, _ in orders.values()):
            orders["blocks"] = (blocks, None)
    return orders


# ----------------------------------------------------------------------------- scores and DP

def length_score(width: float, cpp: float, length: int) -> float:
    """How well a line's baseline length fits an edition line's length.

    :param width: Baseline length in pixels.
    :type width: float
    :param cpp: Characters per pixel.
    :type cpp: float
    :param length: Edition line length in skeleton characters.
    :type length: int
    :return: ``1 - min(1, |width * cpp - length| / max(length, 1))``.
    :rtype: float
    """
    return 1.0 - min(1.0, abs(width * cpp - length) / max(length, 1))


def pair_score(sim: float, len_score: float, letters: int, params: Params) -> float:
    """Score of placing an edition line on a (merged) segmenter line.

    :param sim: Text similarity.
    :type sim: float
    :param len_score: Length score.
    :type len_score: float
    :param letters: Arabic letters in the read.
    :type letters: int
    :param params: Weights.
    :type params: Params
    :return: ``text_weight * sim + length_weight * len_score`` for a read of at least ``text_min_letters``
        letters, else ``short_read_weight * len_score``.
    :rtype: float
    """
    if letters >= params.text_min_letters:
        return params.text_weight * sim + params.length_weight * len_score
    return params.short_read_weight * len_score


def skip_s_cost(line: SegLine, params: Params) -> float:
    """Cost of leaving a segmenter line without an edition line.

    :param line: Segmenter line.
    :type line: SegLine
    :param params: Costs.
    :type params: Params
    :return: ``skip_s_long_cost`` for a read of at least ``skip_s_long_letters`` letters, else ``skip_s_cost``.
    :rtype: float
    """
    return params.skip_s_long_cost if line.letters >= params.skip_s_long_letters else params.skip_s_cost


def align(n_e: int, n_s: int, pair: Callable[[int, int, int], float], skip_s: Sequence[float], skip_e: float,
          can_merge: Callable[[int, int], bool], max_merge: int) -> Tuple[float, List[Step]]:
    """Best monotone alignment of edition lines onto ordered segmenter lines.

    Moves: match (edition line ``j`` on segmenter lines ``i .. i+k-1``, ``k = 1`` or a merge up to
    ``max_merge`` when ``can_merge(i, k)``), skip a segmenter line, skip an edition line.

    :param n_e: Number of edition lines.
    :type n_e: int
    :param n_s: Number of ordered segmenter lines.
    :type n_s: int
    :param pair: ``pair(j, i, k)``: score of edition line ``j`` on segmenter lines ``i .. i+k-1``.
    :type pair: Callable[[int, int, int], float]
    :param skip_s: Cost of skipping each segmenter line.
    :type skip_s: Sequence[float]
    :param skip_e: Cost of skipping an edition line.
    :type skip_e: float
    :param can_merge: ``can_merge(i, k)``: whether lines ``i .. i+k-1`` may form one line (k >= 2).
    :type can_merge: Callable[[int, int], bool]
    :param max_merge: Largest group.
    :type max_merge: int
    :return: ``(score, steps)``; ties keep the move tried first (skip segmenter line, skip edition line,
        then matches by group size).
    :rtype: Tuple[float, List[Step]]
    """
    neg = -math.inf
    best = [[neg] * (n_e + 1) for _ in range(n_s + 1)]
    back: List[List[Optional[Tuple[int, int, Step]]]] = [[None] * (n_e + 1) for _ in range(n_s + 1)]
    best[0][0] = 0.0

    def relax(i: int, j: int, value: float, prev: Tuple[int, int], step: Step) -> None:
        """Keep a better path into cell ``(i, j)``.

        :param i: Segmenter lines consumed.
        :param j: Edition lines consumed.
        :param value: Path score.
        :param prev: Previous cell.
        :param step: Move into the cell.
        """
        if value > best[i][j]:
            best[i][j] = value
            back[i][j] = (prev[0], prev[1], step)

    for i in range(n_s + 1):
        for j in range(n_e + 1):
            cur = best[i][j]
            if cur == neg:
                continue
            if i < n_s:
                relax(i + 1, j, cur - skip_s[i], (i, j), Step("skip_s", None, (i,)))
            if j < n_e:
                relax(i, j + 1, cur - skip_e, (i, j), Step("skip_e", j, ()))
                for k in range(1, max_merge + 1):
                    if i + k > n_s or (k > 1 and not can_merge(i, k)):
                        continue
                    relax(i + k, j + 1, cur + pair(j, i, k), (i, j), Step("match", j, tuple(range(i, i + k))))
    steps: List[Step] = []
    i, j = n_s, n_e
    while (i, j) != (0, 0):
        pi, pj, step = back[i][j]
        steps.append(step)
        i, j = pi, pj
    steps.reverse()
    return best[n_s][n_e], steps


def span_fields(spans: Sequence[Tuple[float, float]], j: int) -> Tuple[float, float, float, float]:
    """The span evidence of a read placed on edition line ``j``.

    :param spans: :func:`span_similarity` of the read to every edition line, by position.
    :type spans: Sequence[Tuple[float, float]]
    :param j: Edition position the read is placed on.
    :type j: int
    :return: ``(cover_sim, cover_lost, best_other, neighbor_sim)``: the own line's span sim and lost share,
        the best span sim to any other line, the best span sim to line ``j - 1`` or ``j + 1`` (0.0 if none).
    :rtype: Tuple[float, float, float, float]
    """
    best_other = max((s for k, (s, _) in enumerate(spans) if k != j), default=0.0)
    neighbor = max((spans[k][0] for k in (j - 1, j + 1) if 0 <= k < len(spans)), default=0.0)
    return spans[j][0], spans[j][1], best_other, neighbor


def is_partial(pair: Pair, params: Params) -> bool:
    """Whether a segment holds only part of its edition line (the segmenter's polygon stops short).

    The read of such a segment matches a contiguous word span of the edition line clearly better than the
    whole line, and that span leaves a large share of the line out. Writing the whole line as the target
    of a truncated segment would teach the recogniser text that is not in the image.

    :param pair: Pair with its span evidence (:attr:`Pair.cover_sim`, :attr:`Pair.cover_lost`).
    :type pair: Pair
    :param params: Thresholds.
    :type params: Params
    :return: True for a read of at least ``cover_min_letters`` letters whose best span beats the whole line
        by at least ``cover_gain`` and leaves at least ``cover_lost`` of the line's characters outside it.
    :rtype: bool
    """
    return (pair.letters >= params.cover_min_letters and pair.cover_sim - pair.sim >= params.cover_gain
            and pair.cover_lost >= params.cover_lost)


def span_vetoes(pairs: Sequence[Pair], params: Params) -> Dict[int, str]:
    """Pairs whose read says they sit on another edition line.

    A read (at least ``cover_min_letters`` letters) whose best span sim to another edition line reaches
    ``contradict_min_sim`` and beats its own line's by ``contradict_margin`` is contradicted; a non-anchor
    read whose best span sim to edition line ``e - 1`` or ``e + 1`` beats its own by ``neighbor_margin``
    prefers a neighbour (a one-line shift). Span sims are used so that a read of half a neighbouring line
    still counts.

    :param pairs: Pairs with span evidence and :attr:`Pair.anchored` set.
    :type pairs: Sequence[Pair]
    :param params: Thresholds.
    :type params: Params
    :return: ``{step: REJECT_CONTRADICTED or REJECT_NEIGHBOR}``.
    :rtype: Dict[int, str]
    """
    out: Dict[int, str] = {}
    for p in pairs:
        if p.letters < params.cover_min_letters:
            continue
        if p.best_other >= params.contradict_min_sim and p.best_other - p.own_span >= params.contradict_margin:
            out[p.step] = REJECT_CONTRADICTED
        elif not p.anchored and p.neighbor_sim >= p.own_span + params.neighbor_margin:
            out[p.step] = REJECT_NEIGHBOR
    return out


def own_best_support(pairs: Sequence[Pair], params: Params) -> bool:
    """Whether every column holds enough reads whose best-matching edition line is their own.

    Whole-line sims are compared here (the argmax over a page's ~15-40 lines; by chance a read's own line
    wins about once per page length), not span sims: the max over many sub-spans of every other line
    inflates the competition and makes the argmax noisy.

    :param pairs: Pairs with :attr:`Pair.best_other_line` set.
    :type pairs: Sequence[Pair]
    :param params: Thresholds.
    :type params: Params
    :return: True when each column present on the path has at least ``no_anchor_own_best`` pairs with a read
        of at least ``cover_min_letters`` letters whose whole-line sim to its own edition line beats every
        other edition line's.
    :rtype: bool
    """
    counts: Counter = Counter({p.column: 0 for p in pairs})
    for p in pairs:
        if p.letters >= params.cover_min_letters and p.sim > p.best_other_line:
            counts[p.column] += 1
    return bool(counts) and min(counts.values()) >= params.no_anchor_own_best


def local_cpps(pairs: Sequence[Pair]) -> Dict[int, float]:
    """Characters per baseline pixel near each pair, for the local length guard.

    The page value is the median over the covering anchors' ``length / width`` plus one vote for the median
    over every pair of the path (so a page with no or one covering anchor still has a robust value). A
    pair's local value is the median of the nearest covering anchor before it, the nearest after it and
    the page value. Partial anchors never vote: their segment is too short for their text.

    :param pairs: Pairs with :attr:`Pair.anchored` and :attr:`Pair.partial` set.
    :type pairs: Sequence[Pair]
    :return: ``{step: cpp}`` for every pair (0.0 when no pair has a positive width).
    :rtype: Dict[int, float]
    """
    anchors = {p.step: p.length / p.width for p in pairs if p.anchored and not p.partial and p.width > 0}
    path = [p.length / p.width for p in pairs if p.width > 0]
    if not path:
        return {p.step: 0.0 for p in pairs}
    page = statistics.median(list(anchors.values()) + [statistics.median(path)])
    steps = sorted(anchors)
    out = {}
    for p in pairs:
        before = [s for s in steps if s < p.step]
        after = [s for s in steps if s > p.step]
        votes = ([anchors[before[-1]]] if before else []) + ([anchors[after[0]]] if after else []) + [page]
        out[p.step] = statistics.median(votes)
    return out


def assign_tiers(steps: Sequence[Step], pairs: Sequence[Pair], params: Params) -> Tuple[bool, float, float, int]:
    """Set the tier of every matched pair (in place).

    Order of the tests: a partial segment (:func:`is_partial`) is rejected in every tier; a position anchor is
    an ``anchor``; a non-anchor on a count-match page or between two position anchors with no skip in
    between is ``count_match`` / ``interpolated`` when its stretch has no span veto (:func:`span_vetoes`) and
    it passes the local length guard (``width x local cpp / length`` within [``len_short_min``,
    ``len_long_max``], :func:`local_cpps`). With ``drop_count_match`` a count-match line between two
    anchors is ``interpolated`` and any other count-match line is rejected. Last, the lost guard turns any
    accepted pair (single or merged segment, every tier) whose read leaves more than ``max_lost`` of its
    edition line outside its best span into a rejection (:data:`REJECT_LOST`; it needs span evidence, i.e. a
    read of at least ``cover_min_letters`` letters; like a partial pair it stays a position anchor for
    bridging), the long-read guard (:func:`long_read`) one whose read is much longer than its edition line
    into :data:`REJECT_LONG_READ`, and with ``drop_interpolated`` an interpolated pair into
    :data:`REJECT_INTERPOLATED_DROPPED`.
    A count match needs the identity path, at least
    ``guard_min_pairs`` readable pairs, ``page_sim >= sim_page``, ``margin >= sim_margin``, no span veto on
    the page and, without any position anchor, either ``margin >= sim_margin + no_anchor_margin_buffer`` or
    :func:`own_best_support`.

    :param steps: Alignment path.
    :type steps: Sequence[Step]
    :param pairs: Matched pairs of the path, in path order (``pair.step`` indexes ``steps``), with their
        evidence (sim, letters, width, length, span sims, column).
    :type pairs: Sequence[Pair]
    :param params: Thresholds.
    :type params: Params
    :return: ``(identity, page_sim, sim_margin, readable pairs)``: whether the path has no skip move; the
        mean sim of the pairs whose read has at least ``text_min_letters`` letters (0.0 when there is
        none); that mean minus their mean :attr:`Pair.off_sim`; how many such pairs there are.
    :rtype: Tuple[bool, float, float, int]
    """
    identity = bool(pairs) and all(s.kind == "match" for s in steps)
    readable = [p for p in pairs if p.letters >= params.text_min_letters]
    page_sim = sum(p.sim for p in readable) / len(readable) if readable else 0.0
    margin = page_sim - sum(p.off_sim for p in readable) / len(readable) if readable else 0.0
    for p in pairs:
        p.anchored = p.sim >= params.sim_anchor and p.letters >= params.anchor_min_letters
        p.partial = is_partial(p, params)
    vetoes = span_vetoes(pairs, params)
    anchor_steps = [p.step for p in pairs if p.anchored]
    guard = (identity and len(readable) >= params.guard_min_pairs and page_sim >= params.sim_page
             and margin >= params.sim_margin)
    supported = (bool(anchor_steps) or margin >= params.sim_margin + params.no_anchor_margin_buffer
                 or own_best_support(pairs, params))
    count_match = guard and not vetoes and supported

    bridged: Set[int] = set()
    vetoed: Dict[int, str] = {}
    for a, b in zip(anchor_steps, anchor_steps[1:]):
        if all(steps[k].kind == "match" for k in range(a + 1, b)):
            stretch = set(range(a + 1, b))
            hits = sorted(stretch & set(vetoes))
            if hits:
                vetoed.update({k: vetoes[hits[0]] for k in stretch})
            else:
                bridged.update(stretch)
    cpps = local_cpps(pairs)
    use_count = count_match and not params.drop_count_match
    for p in pairs:
        p.tier, p.reason, p.len_ratio = TIER_REJECTED, "", 0.0
        if p.partial:
            p.reason = REJECT_PARTIAL
        elif p.anchored:
            p.tier = "anchor"
        elif use_count or p.step in bridged:
            p.len_ratio = p.width * cpps[p.step] / max(p.length, 1)
            if params.len_short_min <= p.len_ratio <= params.len_long_max:
                p.tier = "count_match" if use_count else "interpolated"
            else:
                p.reason = REJECT_LENGTH
        elif count_match:
            p.reason = REJECT_COUNT_MATCH_DROPPED
        elif p.step in vetoed:
            p.reason = vetoed[p.step]
        elif anchor_steps and anchor_steps[0] < p.step < anchor_steps[-1]:
            p.reason = REJECT_UNEQUAL
        elif identity:
            p.reason = (vetoes[min(vetoes)] if guard and vetoes
                        else REJECT_NO_SUPPORT if guard else REJECT_GUARD)
        else:
            p.reason = REJECT_OUTSIDE if anchor_steps else REJECT_NO_ANCHORS
        if p.tier in ACCEPTED_TIERS and p.letters >= params.cover_min_letters and p.cover_lost > params.max_lost:
            p.tier, p.reason = TIER_REJECTED, REJECT_LOST
        if p.tier in ACCEPTED_TIERS and long_read(p, params):
            p.tier, p.reason = TIER_REJECTED, REJECT_LONG_READ
        if p.tier == "interpolated" and params.drop_interpolated:
            p.tier, p.reason = TIER_REJECTED, REJECT_INTERPOLATED_DROPPED
    return identity, page_sim, margin, len(readable)


def align_order(edition: Sequence[EditionLine], order: Sequence[SegLine], ordering: str, height: float,
                params: Params, sim_cache: Dict[Tuple[int, Tuple[int, ...]], float],
                split_x: Optional[float] = None,
                span_cache: Optional[Dict[Tuple[int, ...], List[Tuple[float, float]]]] = None) -> Alignment:
    """Align a page's edition lines onto one ordering of its segmenter lines, refitting cpp on anchors.

    :param edition: Edition lines (placeholders included).
    :type edition: Sequence[EditionLine]
    :param order: Ordered segmenter lines.
    :type order: Sequence[SegLine]
    :param ordering: Ordering name.
    :type ordering: str
    :param height: Median line height of the page.
    :type height: float
    :param params: Thresholds.
    :type params: Params
    :param sim_cache: ``{(edition position, segmenter indices): sim}`` shared across orderings.
    :type sim_cache: Dict[Tuple[int, Tuple[int, ...]], float]
    :param split_x: Column split of a two-column ordering (sets :attr:`Pair.column`); None for one column.
    :type split_x: Optional[float]
    :param span_cache: ``{segmenter indices: [(span sim, lost) per edition position]}`` shared across
        orderings (a fresh one when None).
    :type span_cache: Optional[Dict[Tuple[int, ...], List[Tuple[float, float]]]]
    :return: The best alignment (:func:`alignment_rank`) over the initial cpp and at most ``refit_rounds``
        refits.
    :rtype: Alignment
    """
    span_cache = {} if span_cache is None else span_cache
    total_width = sum(l.width for l in order)
    cpp = sum(e.length for e in edition) / total_width if total_width > 0 else 0.0
    merge_ok: Dict[Tuple[int, int], bool] = {}

    def can_merge(i: int, k: int) -> bool:
        """Cached :func:`group_mergeable`.

        :param i: First position.
        :param k: Group size.
        :return: Whether the group may be merged.
        """
        if (i, k) not in merge_ok:
            merge_ok[(i, k)] = group_mergeable(order, i, k, height, params)
        return merge_ok[(i, k)]

    def evidence(j: int, i: int, k: int) -> Tuple[float, int, float]:
        """Sim, read letters and width of edition line ``j`` on lines ``i .. i+k-1``.

        :param j: Edition position.
        :param i: First segmenter position.
        :param k: Group size.
        :return: ``(sim, letters, width)``.
        """
        group = order[i:i + k]
        key = (j, tuple(l.index for l in group))
        letters = sum(l.letters for l in group)
        if key not in sim_cache:
            sim_cache[key] = similarity(merged_skeleton(group), edition[j].skeleton) if letters else 0.0
        return sim_cache[key], letters, sum(l.width for l in group)

    off_cache: Dict[Tuple[int, Tuple[int, ...]], Tuple[float, float]] = {}

    def off_sim(j: int, group: Sequence[SegLine]) -> Tuple[float, float]:
        """Mean and highest whole-line sim of a (merged) read to every edition line but ``j``.

        :param j: Matched edition position.
        :param group: Segmenter lines.
        :return: ``(mean, max)``; ``(0.0, 0.0)`` for a one-line edition.
        """
        key = (j, tuple(l.index for l in group))
        if key not in off_cache:
            read = merged_skeleton(group)
            others = [similarity(read, e.skeleton) for n, e in enumerate(edition) if n != j]
            off_cache[key] = (sum(others) / len(others), max(others)) if others else (0.0, 0.0)
        return off_cache[key]

    def spans(group: Sequence[SegLine]) -> List[Tuple[float, float]]:
        """Cached :func:`span_similarity` of a (merged) read to every edition line.

        :param group: Segmenter lines.
        :return: ``[(sim, lost)]`` by edition position.
        """
        key = tuple(l.index for l in group)
        if key not in span_cache:
            read = merged_skeleton(group)
            span_cache[key] = [span_similarity(read, e.skeleton, params.cover_word_share) for e in edition]
        return span_cache[key]

    def make_pair(n: int, step: Step, cpp_now: float) -> Pair:
        """A matched step with its evidence.

        :param n: Step index.
        :param step: Match step.
        :param cpp_now: cpp of the length score.
        :return: The pair (tier not yet assigned).
        """
        group = order[step.s[0]:step.s[-1] + 1]
        sim, letters, width = evidence(step.e, step.s[0], len(step.s))
        off, other_line = off_sim(step.e, group) if letters >= params.text_min_letters else (0.0, 0.0)
        x_mid = sum(l.x_center for l in group) / len(group)
        pair = Pair(n, step.e, step.s, sim, length_score(width, cpp_now, edition[step.e].length), letters, off,
                    width=width, length=edition[step.e].length, best_other_line=other_line,
                    column=0 if split_x is None or x_mid >= split_x else 1)
        if letters >= params.cover_min_letters:
            pair.cover_sim, pair.cover_lost, pair.best_other, pair.neighbor_sim = span_fields(spans(group), step.e)
            read = merged_skeleton(group)
            pair.read_words = len(read.split())
            pair.edition_words = len(edition[step.e].skeleton.split())
            pair.read_chars = len(read.replace(" ", ""))
            pair.edition_chars = len(edition[step.e].skeleton.replace(" ", ""))
            pair.missing = tuple(missing_words(read, edition[step.e].skeleton))
        return pair

    skips = [skip_s_cost(l, params) for l in order]
    best: Optional[Alignment] = None
    for _ in range(params.refit_rounds + 1):
        fixed_cpp = cpp

        def pair(j: int, i: int, k: int) -> float:
            """Pair score under the current cpp.

            :param j: Edition position.
            :param i: First segmenter position.
            :param k: Group size.
            :return: :func:`pair_score`.
            """
            sim, letters, width = evidence(j, i, k)
            return pair_score(sim, length_score(width, fixed_cpp, edition[j].length), letters, params)

        score, steps = align(len(edition), len(order), pair, skips, params.skip_e_cost, can_merge, params.max_merge)
        pairs = [make_pair(n, step, fixed_cpp) for n, step in enumerate(steps) if step.kind == "match"]
        identity, page_sim, margin, readable = assign_tiers(steps, pairs, params)
        alignment = Alignment(ordering, list(order), steps, pairs, score, fixed_cpp, identity, page_sim, margin,
                              readable)
        if best is None or alignment_rank(alignment) > alignment_rank(best):
            best = alignment
        anchors = [p for p in pairs if p.tier == "anchor"]
        if len(anchors) < params.min_refit_anchors:
            break
        anchor_width = sum(order[i].width for p in anchors for i in p.s)
        cpp = sum(edition[p.e].length for p in anchors) / anchor_width if anchor_width > 0 else fixed_cpp
        if abs(cpp - fixed_cpp) <= 1e-9 * max(cpp, 1e-12):
            break
    return best


def alignment_rank(alignment: Alignment) -> Tuple[int, int, float]:
    """Sort key of competing orderings (higher is better).

    :param alignment: An alignment.
    :type alignment: Alignment
    :return: ``(position anchors, accepted pairs, score)``; position anchors include partial ones, whose
        placement is still evidence for the ordering.
    :rtype: Tuple[int, int, float]
    """
    tiers = alignment.tier_counts()
    return alignment.anchored(), sum(tiers[t] for t in ACCEPTED_TIERS), alignment.score


# ----------------------------------------------------------------------------- page

def edition_lines(answer: str, params: Params) -> List[EditionLine]:
    """A page's edition lines that take part in the alignment.

    :param answer: Edition text of the page (lines joined by ``\\n``).
    :type answer: str
    :param params: Thresholds.
    :type params: Params
    :return: Non-empty lines with at least ``min_edition_letters`` Arabic letters outside gap tokens, with
        their :func:`clean_target` (``""`` for a placeholder) and skeleton.
    :rtype: List[EditionLine]
    """
    out = []
    for raw in (l.strip() for l in answer.split("\n")):
        if not raw:
            continue
        visible = _EDITION_MARKS.sub(" ", raw)
        if arabic_letter_count(visible) < params.min_edition_letters:
            continue
        out.append(EditionLine(index=len(out), raw=raw, target=clean_target(raw)[0], skeleton=edition_skeleton(raw),
                               visible=visible_text(raw)))
    return out


def duplicate_positions(edition: Sequence[EditionLine]) -> Set[int]:
    """Edition lines whose target text occurs more than once on the page.

    :param edition: Edition lines of one page.
    :type edition: Sequence[EditionLine]
    :return: Positions of the lines with a non-empty target whose NFC form (whitespace collapsed) is shared
        with another line of the page.
    :rtype: Set[int]
    """
    keys = [" ".join(unicodedata.normalize("NFC", e.target).split()) for e in edition]
    counts = Counter(k for k in keys if k)
    return {n for n, k in enumerate(keys) if k and counts[k] > 1}


def drop_duplicates(alignment: Alignment, edition: Sequence[EditionLine]) -> None:
    """Reject the accepted pairs of repeated edition texts (in place).

    :param alignment: Chosen alignment of the page.
    :type alignment: Alignment
    :param edition: Edition lines of the page.
    :type edition: Sequence[EditionLine]
    :return: None. Accepted pairs on :func:`duplicate_positions` become :data:`REJECT_DUPLICATE`; position
        anchors stay anchors (they steered the path).
    :rtype: None
    """
    repeated = duplicate_positions(edition)
    for p in alignment.pairs:
        if p.e in repeated and p.tier in ACCEPTED_TIERS:
            p.tier, p.reason = TIER_REJECTED, REJECT_DUPLICATE


def align_page(page: Dict[str, Any], answer: str, params: Params) -> Tuple[List[SegLine], List[int], List[EditionLine],
                                                                         Optional[Alignment], str]:
    """Align one page (pure: no I/O).

    :param page: Segmenter JSON of the page.
    :type page: Dict[str, Any]
    :param answer: Edition text of the page.
    :type answer: str
    :param params: Thresholds.
    :type params: Params
    :return: ``(all segmenter lines, noise indices, edition lines, best alignment or None, skip reason)``;
        the reason is ``""`` when the alignment has at least one accepted target line.
    :rtype: Tuple[List[SegLine], List[int], List[EditionLine], Optional[Alignment], str]
    """
    segments = [make_seg_line(raw) for raw in page.get("lines", [])]
    edition = edition_lines(answer, params)
    if sum(1 for e in edition if e.target) < params.min_target_lines:
        return segments, [], edition, None, SKIP_FEW_TARGETS
    page_median = median_width(segments)
    noise = [l.index for l in segments if is_noise(l, page_median, params)]
    noise_set = set(noise)
    kept = [l for l in segments if l.index not in noise_set]
    if not kept:
        return segments, noise, edition, None, SKIP_NO_LINES
    height = line_height(kept)
    sim_cache: Dict[Tuple[int, Tuple[int, ...]], float] = {}
    span_cache: Dict[Tuple[int, ...], List[Tuple[float, float]]] = {}
    candidates = [align_order(edition, order, name, height, params, sim_cache, split_x, span_cache)
                  for name, (order, split_x) in candidate_orders(kept, float(page["image_size"][0]), params).items()]
    best = max(candidates, key=alignment_rank)
    if params.drop_duplicate_texts:
        drop_duplicates(best, edition)
    if not best.anchored() and not best.tier_counts()["count_match"]:
        return segments, noise, edition, best, SKIP_NO_ANCHORS
    if not any(p.tier in ACCEPTED_TIERS and edition[p.e].target for p in best.pairs):
        return segments, noise, edition, best, SKIP_NO_ACCEPTED
    return segments, noise, edition, best, ""


# ----------------------------------------------------------------------------- ink extent

def extent_config(params: Params) -> line_extent.ExtentConfig:
    """Band-walk thresholds from the alignment parameters.

    :param params: Thresholds.
    :type params: Params
    :return: The walk configuration; ``max_extend`` is 0 and the vertical rebuild is off when ``extend_to_ink``
        is off (the walk then only measures the ink beyond the segmenter's ends for the gate).
    :rtype: line_extent.ExtentConfig
    """
    return line_extent.ExtentConfig(above=params.band_above, below=params.band_below, gap=params.extend_gap,
                                    max_extend=params.extend_max if params.extend_to_ink else 0.0,
                                    raise_top=params.raise_top and params.extend_to_ink,
                                    clamp=params.neighbour_clamp, block_slack=params.block_slack,
                                    spacing_above=params.spacing_above, spacing_below=params.spacing_below,
                                    detect=params.detect_unsegmented, foreign=params.foreign_components,
                                    pencil_share=params.pencil_share, stub_width=params.stub_width,
                                    own_down_cap=params.own_down_cap, own_ink_reach=params.own_ink_reach)


def is_steep(line: SegLine, params: Params) -> bool:
    """Whether a segmenter line is rotated (a marginal line written at an angle).

    :param line: Segmenter line.
    :type line: SegLine
    :param params: Thresholds (``steep_degrees``).
    :type params: Params
    :return: True when its baseline is steeper than ``steep_degrees``.
    :rtype: bool
    """
    return line.angle > params.steep_degrees


def page_block(result: PageResult, params: Params) -> Optional[line_extent.TextBlock]:
    """Text block of a page from the segmenter's own line ends (every non-noise, non-rotated line, written
    or not).

    :param result: Page result with segments and noise.
    :type result: PageResult
    :param params: Thresholds (``block_thirds``).
    :type params: Params
    :return: :func:`~src.finetuning.kraken.line_extent.text_block` of the lines' polygon x-ranges at their mean
        baseline heights; None without such lines.
    :rtype: Optional[line_extent.TextBlock]
    """
    noise = set(result.noise)
    spans = []
    for line in result.segments:
        if line.index in noise or is_steep(line, params):
            continue
        xs = [p[0] for p in line.boundary] if len(line.boundary) >= 3 else [line.x_min, line.x_max]
        spans.append((min(xs), max(xs), line.y_mean))
    return line_extent.text_block(spans, params.block_thirds)


def line_block(block: Optional[line_extent.TextBlock], group: Sequence[SegLine], params: Params
               ) -> Optional[Tuple[float, float]]:
    """The text block margins that bound one (merged) line.

    :param block: Page text block.
    :type block: Optional[line_extent.TextBlock]
    :param group: The line's segments.
    :type group: Sequence[SegLine]
    :param params: Thresholds (``block_slack``).
    :type params: Params
    :return: ``(left, right)`` at the group's mean baseline height; None without a block or with an infinite
        slack.
    :rtype: Optional[Tuple[float, float]]
    """
    if block is None or not math.isfinite(params.block_slack):
        return None
    return block.bounds(sum(l.y_mean for l in group) / len(group))


def layout_lines(result: PageResult) -> List[page_layout.LayoutLine]:
    """The page's non-noise segmenter lines for the page layout (rotated ones included).

    :param result: Page result with segments and noise.
    :type result: PageResult
    :return: One :class:`~src.finetuning.kraken.page_layout.LayoutLine` per line (polygon x/y ranges, or the
        baseline's without a polygon).
    :rtype: List[page_layout.LayoutLine]
    """
    noise = set(result.noise)
    out = []
    for line in result.segments:
        if line.index in noise:
            continue
        pts = line.boundary if len(line.boundary) >= 3 else line.baseline
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        out.append(page_layout.LayoutLine(line.index, line.baseline, min(xs), max(xs), min(ys), max(ys),
                                          line.y_mean, line.angle))
    return out


def page_spacing(result: PageResult, params: Params) -> float:
    """Baseline spacing of a page (:func:`~src.finetuning.kraken.page_layout.baseline_spacing` over its
    non-noise, non-rotated lines; the line height with fewer than four measured pairs).

    :param result: Page result with segments and noise.
    :type result: PageResult
    :param params: Thresholds (``steep_degrees``).
    :type params: Params
    :return: Spacing in pixels.
    :rtype: float
    """
    lines = [l for l in layout_lines(result) if l.angle <= params.steep_degrees]
    return page_layout.baseline_spacing(lines, page_line_height(result))


def line_bounds(blocks: Optional[page_layout.PageBlocks], group: Sequence[SegLine], params: Params
                ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[Optional[float], Optional[float]]]]:
    """Soft (block, with slack) and hard (margin material, gutters) limits of one (merged) line.

    :param blocks: Column blocks of the page.
    :type blocks: Optional[page_layout.PageBlocks]
    :param group: The line's segments.
    :type group: Sequence[SegLine]
    :param params: Thresholds (``block_slack``).
    :type params: Params
    :return: ``(block or None, hard or None)`` at the group's mean baseline in the column block holding its
        x-centre; the block is None with an infinite slack.
    :rtype: Tuple[Optional[Tuple[float, float]], Optional[Tuple[Optional[float], Optional[float]]]]
    """
    if blocks is None:
        return None, None
    x = (min(l.x_min for l in group) + max(l.x_max for l in group)) / 2
    y = sum(l.y_mean for l in group) / len(group)
    column = blocks.column_of(x)
    soft = column.block.bounds(y) if math.isfinite(params.block_slack) else None
    return soft, column.hard_bounds(y)


def needs_geometry(params: Params) -> bool:
    """Whether a run must read page images for the ink walk.

    :param params: Thresholds.
    :type params: Params
    :return: True when lines are extended or the geometry gate is on (finite ``max_ink_beyond``).
    :rtype: bool
    """
    return params.extend_to_ink or math.isfinite(params.max_ink_beyond)


def page_line_height(result: PageResult) -> float:
    """Median line height of the page's non-noise segmenter lines (the alignment's own unit).

    :param result: Page result with segments and noise.
    :type result: PageResult
    :return: :func:`line_height` of the kept lines.
    :rtype: float
    """
    noise = set(result.noise)
    return line_height([l for l in result.segments if l.index not in noise])


def segment_owners(alignment: Alignment) -> Dict[int, int]:
    """Which edition line the path placed on each segmenter line.

    :param alignment: Page alignment.
    :type alignment: Alignment
    :return: ``{segmenter index: edition position}`` for every member of a matched pair (any tier).
    :rtype: Dict[int, int]
    """
    return {alignment.order[i].index: p.e for p in alignment.pairs for i in p.s}


def read_matches(segments: Sequence[SegLine], edition: Sequence[EditionLine], skip: Set[int],
                 params: Params) -> Dict[int, Tuple[int, float]]:
    """Best edition line of each readable segmenter read.

    :param segments: Segmenter lines.
    :type segments: Sequence[SegLine]
    :param edition: Edition lines.
    :type edition: Sequence[EditionLine]
    :param skip: Segmenter indices not to score (those the path already placed).
    :type skip: Set[int]
    :param params: Thresholds (``cover_min_letters``, ``cover_word_share``).
    :type params: Params
    :return: ``{segmenter index: (edition position, span sim)}`` for reads of at least ``cover_min_letters``
        letters (:func:`span_similarity`, ties to the earlier line).
    :rtype: Dict[int, Tuple[int, float]]
    """
    out: Dict[int, Tuple[int, float]] = {}
    for line in segments:
        if line.index in skip or line.letters < params.cover_min_letters or not edition:
            continue
        sims = [span_similarity(line.skeleton, e.skeleton, params.cover_word_share)[0] for e in edition]
        best = max(range(len(sims)), key=lambda k: (sims[k], -k))
        out[line.index] = (best, sims[best])
    return out


def other_boxes(result: PageResult, pair: Pair, owners: Dict[int, int], matches: Dict[int, Tuple[int, float]],
                params: Params) -> List[line_extent.Box]:
    """Every other segmenter line of the page as the ink walk of one pair sees it.

    :param result: Page result.
    :type result: PageResult
    :param pair: The pair being extended.
    :type pair: Pair
    :param owners: :func:`segment_owners`.
    :type owners: Dict[int, int]
    :param matches: :func:`read_matches` of the unplaced lines.
    :type matches: Dict[int, Tuple[int, float]]
    :param params: Thresholds (``contradict_min_sim``).
    :type params: Params
    :return: Boxes of all lines but the pair's members (polygons of at least three points), flagged ``steal``
        when the path placed another edition line on them or their read matches another edition line with a
        span sim of at least ``contradict_min_sim``; non-noise, non-rotated lines carry their baseline (they
        clamp the pair's band as neighbours). With ``end_checks`` a box beside an end whose read has no
        counterpart among the edition line's end words (:func:`end_read_matches`) is not absorbable.
    :rtype: List[line_extent.Box]
    """
    group = [result.alignment.order[i] for i in pair.s]
    members = {l.index for l in group}
    x_lo, x_hi = min(l.x_min for l in group), max(l.x_max for l in group)
    edition = result.edition[pair.e]
    noise = set(result.noise)
    boxes = []
    for line in result.segments:
        if line.index in members or len(line.boundary) < 3:
            continue
        owner = owners.get(line.index)
        match = matches.get(line.index)
        steal = ((owner is not None and owner != pair.e)
                 or (match is not None and match[0] != pair.e and match[1] >= params.contradict_min_sim))
        neighbour = line.index not in noise and not is_steep(line, params)
        absorbable = True
        if params.end_checks and line.letters and not steal and (line.x_center < x_lo or line.x_center > x_hi):
            absorbable = end_read_matches(line.skeleton, edition.skeleton, line.x_center < x_lo)
        boxes.append(line_extent.Box(line.index, line.boundary, steal, line.baseline if neighbour else (),
                                     absorbable))
    return boxes


def end_read_matches(read: str, line: str, tail: bool, words: int = 2, min_sim: float = END_READ_SIM) -> bool:
    """Whether a box's read has a counterpart among the edition line's end words.

    :param read: The box's read skeleton.
    :type read: str
    :param line: Edition line skeleton.
    :type line: str
    :param tail: Compare with the line's last words (a box beyond the left end of a right-to-left line); else
        with its first words.
    :type tail: bool
    :param words: End words compared (each alone and all of them together).
    :type words: int
    :param min_sim: Smallest sim that counts.
    :type min_sim: float
    :return: True when the read is empty, the line is empty, or some end word (or the end words together)
        reaches ``min_sim``.
    :rtype: bool
    """
    parts = line.split()
    if not read or not parts:
        return True
    end = parts[-words:] if tail else parts[:words]
    candidates = end + [" ".join(end)]
    return max(similarity(read.replace(" ", ""), c.replace(" ", "")) for c in candidates) >= min_sim


def missing_x(pair: Pair, group: Sequence[SegLine]) -> List[float]:
    """Estimated page columns of the pair's missing edition words (the line runs right to left).

    :param pair: Pair with :attr:`Pair.missing`.
    :type pair: Pair
    :param group: The pair's segments.
    :type group: Sequence[SegLine]
    :return: ``x_max - position x width`` per missing word.
    :rtype: List[float]
    """
    x_lo, x_hi = min(l.x_min for l in group), max(l.x_max for l in group)
    return [x_hi - frac * (x_hi - x_lo) for _, frac in pair.missing]


def supralinear_box(result: PageResult, pair: Pair, group: Sequence[SegLine], owners: Dict[int, int],
                    matches: Dict[int, Tuple[int, float]], spacing: float, params: Params) -> Optional[int]:
    """A short unplaced segment just above the line that can hold its missing word (a supralinear insertion).

    :param result: Page result.
    :type result: PageResult
    :param pair: Pair whose read misses an edition word.
    :type pair: Pair
    :param group: The pair's segments.
    :type group: Sequence[SegLine]
    :param owners: :func:`segment_owners`.
    :type owners: Dict[int, int]
    :param matches: :func:`read_matches` of the unplaced lines.
    :type matches: Dict[int, Tuple[int, float]]
    :param spacing: Baseline spacing in pixels.
    :type spacing: float
    :param params: Thresholds.
    :type params: Params
    :return: Segmenter index of the segment nearest to a missing word's estimated position among those not
        placed on any edition line (nor reading as another one), not rotated, no wider than :data:`SUPRA_WIDTH`
        of the line, with their polygon centre inside the line's x-range and between :data:`SUPRA_MIN` and
        :data:`SUPRA_MAX` spacings above its baseline; None without one.
    :rtype: Optional[int]
    """
    baseline, _ = merged_geometry(group)
    members = {l.index for l in group}
    x_lo, x_hi = min(l.x_min for l in group), max(l.x_max for l in group)
    targets = missing_x(pair, group)
    best: Optional[Tuple[float, int]] = None
    for line in result.segments:
        if line.index in members or line.index in owners or len(line.boundary) < 3 or is_steep(line, params):
            continue
        match = matches.get(line.index)
        if match is not None and match[0] != pair.e and match[1] >= params.contradict_min_sim:
            continue
        bx = [p[0] for p in line.boundary]
        by = [p[1] for p in line.boundary]
        cx, cy = (min(bx) + max(bx)) / 2, (min(by) + max(by)) / 2
        if not x_lo <= cx <= x_hi or max(bx) - min(bx) > SUPRA_WIDTH * (x_hi - x_lo):
            continue
        rise = float(line_extent.baseline_y(baseline, [cx])[0]) - cy
        if not SUPRA_MIN * spacing <= rise <= SUPRA_MAX * spacing:
            continue
        dist = min(abs(cx - x) for x in targets)
        if best is None or dist < best[0]:
            best = (dist, line.index)
    return None if best is None else best[1]


def supralinear_ink(ink: line_extent.PageInk, pair: Pair, group: Sequence[SegLine], boxes: Sequence[line_extent.Box],
                    spacing: float, height: float) -> List[int]:
    """Ink components just above the line near a missing word (an insertion the segmenter did not box).

    :param ink: Page ink components.
    :type ink: line_extent.PageInk
    :param pair: Pair whose read misses an edition word.
    :type pair: Pair
    :param group: The pair's segments.
    :type group: Sequence[SegLine]
    :param boxes: Other lines (a component whose centroid lies inside one of their polygons is theirs).
    :type boxes: Sequence[line_extent.Box]
    :param spacing: Baseline spacing in pixels.
    :type spacing: float
    :param height: Line height in pixels.
    :type height: float
    :return: Labels of the components at least half a walk component large, centroid inside the line's x-range
        and within :data:`SUPRA_NEAR` line heights of a missing word's estimated position, between
        :data:`SUPRA_MIN` and :data:`SUPRA_MAX` spacings above the baseline; empty when they span less than a
        quarter line height.
    :rtype: List[int]
    """
    baseline, _ = merged_geometry(group)
    x_lo, x_hi = int(min(l.x_min for l in group)), int(math.ceil(max(l.x_max for l in group)))
    base = line_extent.baseline_y(baseline, np.array([x_lo, x_hi], dtype=np.float64))
    y0 = int(max(0, math.floor(base.min() - SUPRA_MAX * spacing)))
    y1 = int(min(ink.shape[0], math.ceil(base.max() - SUPRA_MIN * spacing) + 1))
    if y1 <= y0 or x_hi <= x_lo:
        return []
    ids = np.unique(ink.labels[y0:y1, max(0, x_lo):x_hi + 1])
    targets = missing_x(pair, group)
    contours = [np.array(b.polygon, dtype=np.float32).reshape(-1, 1, 2) for b in boxes if len(b.polygon) >= 3]
    out: List[int] = []
    for k in ids[ids > 0]:
        cx, cy = float(ink.centroid_x[k]), float(ink.centroid_y[k])
        rise = float(line_extent.baseline_y(baseline, np.array([cx]))[0]) - cy
        if not (x_lo <= cx <= x_hi and SUPRA_MIN * spacing <= rise <= SUPRA_MAX * spacing):
            continue
        if ink.area[k] < 0.5 * (line_extent.WALK_MIN_AREA * height) ** 2:
            continue
        if min(abs(cx - x) for x in targets) > SUPRA_NEAR * height:
            continue
        if any(cv2.pointPolygonTest(c, (cx, cy), False) >= 0 for c in contours):
            continue
        out.append(int(k))
    if not out:
        return []
    span = ink.bbox[out, 2].max() - ink.bbox[out, 0].min()
    return out if span >= 0.25 * height else []


def extend_pairs(result: PageResult, ink: line_extent.PageInk, params: Params) -> None:
    """Run the ink walk on every matched pair with a target line (in place: :attr:`PageResult.geometry`).

    Steep (rotated) lines are left alone. The page's baseline spacing (:func:`page_spacing`) sets the raise.
    Each line's walk is bounded by its column block, with hard limits at margin material and gutters
    (:func:`~src.finetuning.kraken.page_layout.page_blocks`, :func:`line_bounds`; without ``column_blocks`` the
    round-2 page-wide block, :func:`page_block`). With ``end_checks`` an edition line ending in ':' moves its
    left end along own-connected ink only, and boxes whose read has no counterpart at the edition line's end
    are not absorbed (:func:`other_boxes`). With ``supralinear`` an accepted pair whose read misses an edition
    word merges a short unplaced segment above it (:func:`supralinear_box`) or the ink there
    (:func:`supralinear_ink`); with neither it is noted ``missing`` for the gate. With ``extend_to_ink`` off the
    walk only measures the ink beyond the segmenter's ends, the geometry stays the segmenter's and the own-ink
    share is measured on it.

    :param result: Aligned page.
    :type result: PageResult
    :param ink: Page ink components (:func:`~src.finetuning.kraken.line_extent.page_ink`).
    :type ink: line_extent.PageInk
    :param params: Thresholds.
    :type params: Params
    :return: None.
    :rtype: None
    """
    alignment = result.alignment
    result.line_height = page_line_height(result)
    result.spacing = page_spacing(result, params)
    result.geometry = {}
    result.notes = {}
    config = extent_config(params)
    owners = segment_owners(alignment)
    matches = read_matches(result.segments, result.edition, set(owners), params)
    block = page_block(result, params)
    blocks = None
    if params.column_blocks:
        blocks = page_layout.page_blocks(layout_lines(result), result.line_height, result.spacing,
                                         ink.labels > 0 if params.gutters else None, params.block_thirds,
                                         params.gutters, params.steep_degrees, params.block_start_percentile)
    for p in alignment.pairs:
        group = [alignment.order[i] for i in p.s]
        if not result.edition[p.e].target or any(is_steep(l, params) for l in group):
            continue
        baseline, boundary = merged_geometry(group)
        boxes = other_boxes(result, p, owners, matches, params)
        if params.column_blocks:
            bounds, hard = line_bounds(blocks, group, params)
        else:
            bounds, hard = line_block(block, group, params), None
        note: Dict[str, Any] = {}
        connected_only = (False, False)
        if params.end_checks and result.edition[p.e].raw.rstrip().endswith(FILLER_ENDINGS):
            connected_only = (True, False)
            note["connected_only_end"] = "left"
        blocked_boxes = [b.key for b in boxes if not b.absorbable]
        if blocked_boxes:
            note["not_absorbable"] = blocked_boxes
        forced: Tuple[int, ...] = ()
        extra: Tuple[int, ...] = ()
        if params.supralinear and p.missing and p.tier in ACCEPTED_TIERS:
            key = supralinear_box(result, p, group, owners, matches, result.spacing, params) \
                if params.extend_to_ink else None
            comps = supralinear_ink(ink, p, group, boxes, result.spacing, result.line_height) \
                if params.extend_to_ink and key is None else []
            if key is not None:
                forced, note["supralinear"] = (key,), f"box:{key}"
            elif comps:
                extra, note["supralinear"] = tuple(comps), f"ink:{len(comps)}"
            else:
                note["supralinear"] = "missing"
        ext = line_extent.extend_line(ink, baseline, [l.boundary for l in group], result.line_height, boxes,
                                      config, bounds, result.spacing, hard, connected_only, forced, extra)
        if not params.extend_to_ink:
            own = line_extent.measure_polygon(ink, baseline, boundary, result.line_height, boxes, config, bounds,
                                              result.spacing)
            ext = line_extent.LineExtent(tuple(baseline), tuple(boundary), ext.left, ext.right,
                                         own_ink_outside=own, block=ext.block, spacing=ext.spacing,
                                         detected=ext.detected, hard=ext.hard)
        result.geometry[p.step] = ext
        if note:
            result.notes[p.step] = note


def ink_beyond(ext: line_extent.LineExtent, height: float) -> float:
    """Ink still continuing beyond the farther-reaching end, in line heights.

    :param ext: Line geometry after the walk.
    :type ext: line_extent.LineExtent
    :param height: Line height in pixels.
    :type height: float
    :return: ``max(left, right) ink_beyond_px / height``.
    :rtype: float
    """
    return max(ext.left.ink_beyond_px, ext.right.ink_beyond_px) / max(height, 1.0)


def gate_extents(result: PageResult, params: Params) -> None:
    """The text-free geometry gate on accepted target pairs (in place).

    A pair is rejected, in this order, with :data:`REJECT_STEAL` when its walk reached a segment of another
    edition line or absorbed (or merged) a segment that another accepted line took too; with
    :data:`REJECT_OVERLAP` when its polygon newly covers more than ``max_overlap`` of another line's polygon
    area; with :data:`REJECT_INK_BEYOND` when ink still continues beyond an end by more than ``max_ink_beyond``
    line heights; with :data:`REJECT_END_SHIFT` when the detached ink an end took sits more than
    ``max_end_shift`` line heights above or below the line's own ink there (an L-shaped end that took the
    neighbouring line's last word); with :data:`REJECT_MISSING_WORD` when its read misses an edition word and no
    supralinear segment or ink was found for it; with :data:`REJECT_OWN_INK` when more than
    ``max_own_ink_outside`` of the ink in its band lies outside its polygon. The final-polygon overlaps are
    gated afterwards (:func:`final_overlaps`).

    :param result: Page with :attr:`PageResult.geometry` set.
    :type result: PageResult
    :param params: Thresholds.
    :type params: Params
    :return: None.
    :rtype: None
    """
    accepted = [p for p in result.accepted() if p.step in result.geometry]
    shared = Counter(k for p in accepted for k in result.geometry[p.step].absorbed + result.geometry[p.step].forced)
    for p in accepted:
        ext = result.geometry[p.step]
        if ext.stolen or any(shared[k] > 1 for k in ext.absorbed + ext.forced):
            p.tier, p.reason = TIER_REJECTED, REJECT_STEAL
        elif ext.overlap > params.max_overlap:
            p.tier, p.reason = TIER_REJECTED, REJECT_OVERLAP
        elif ink_beyond(ext, result.line_height) > params.max_ink_beyond:
            p.tier, p.reason = TIER_REJECTED, REJECT_INK_BEYOND
        elif max(abs(ext.end_shift[0]), abs(ext.end_shift[1])) > params.max_end_shift:
            p.tier, p.reason = TIER_REJECTED, REJECT_END_SHIFT
        elif result.notes.get(p.step, {}).get("supralinear") == "missing":
            p.tier, p.reason = TIER_REJECTED, REJECT_MISSING_WORD
        elif ext.own_ink_outside > params.max_own_ink_outside:
            p.tier, p.reason = TIER_REJECTED, REJECT_OWN_INK


def final_overlaps(result: PageResult, ink: line_extent.PageInk, params: Params) -> None:
    """The overlap gate on the final polygons of the accepted lines (in place).

    Runs :func:`~src.finetuning.kraken.page_layout.resolve_overlaps` on the accepted target lines' polygons
    (trimming only when ``extend_to_ink`` rebuilt them; ``max_overlap`` inf trims nothing): a line whose trim
    would cut its own ink is rejected with :data:`REJECT_OVERLAP_TRIM`, one still sharing more than
    ``max_overlap`` of the smaller polygon with another with :data:`REJECT_OVERLAP`. Every walked line's
    geometry records ``overlap_final`` (with the other line's first segmenter index) and whether it was
    trimmed.

    :param result: Page after :func:`gate_extents`.
    :type result: PageResult
    :param ink: Page ink components.
    :type ink: line_extent.PageInk
    :param params: Thresholds.
    :type params: Params
    :return: None.
    :rtype: None
    """
    accepted = [p for p in result.accepted() if p.step in result.geometry]
    if not accepted:
        return
    lines = {p.step: (result.geometry[p.step].baseline, result.geometry[p.step].boundary) for p in accepted}
    limit = params.max_overlap if params.extend_to_ink else math.inf
    resolved = page_layout.resolve_overlaps(ink, lines, limit, result.line_height or page_line_height(result))
    first_index = {p.step: min(result.alignment.order[i].index for i in p.s) for p in accepted}
    for p in accepted:
        r = resolved[p.step]
        ext = result.geometry[p.step]
        result.geometry[p.step] = dataclasses.replace(
            ext, boundary=r.polygon, overlap_final=r.overlap,
            overlap_final_key=first_index.get(r.key) if r.key is not None else None,
            trimmed=r.trimmed, trim_cut_own=r.cut_own)
        if r.cut_own:
            p.tier, p.reason = TIER_REJECTED, REJECT_OVERLAP_TRIM
        elif r.overlap > params.max_overlap:
            p.tier, p.reason = TIER_REJECTED, REJECT_OVERLAP


# ----------------------------------------------------------------------------- PAGE XML

@dataclass(frozen=True)
class OutLine:
    """One ``TextLine`` to write.

    :param line_id: XML id.
    :param baseline: Baseline points.
    :param boundary: Polygon points.
    :param text: Unicode text (``""`` for a line that is not a training target).
    :param custom: ``custom`` attribute (PAGE ``tag {key:value;}`` syntax).
    """
    line_id: str
    baseline: Tuple[Point, ...]
    boundary: Tuple[Point, ...]
    text: str
    custom: str


def format_points(points: Sequence[Point], width: int, height: int) -> str:
    """PAGE ``points`` string, rounded and clamped into the image.

    :param points: Points.
    :type points: Sequence[Point]
    :param width: Image width.
    :type width: int
    :param height: Image height.
    :type height: int
    :return: ``"x1,y1 x2,y2 ..."``.
    :rtype: str
    """
    return " ".join(f"{min(max(int(round(x)), 0), width - 1)},{min(max(int(round(y)), 0), height - 1)}"
                    for x, y in points)


def custom_attr(tier: str, sim: Optional[float] = None, len_score: Optional[float] = None,
                edition_index: Optional[int] = None, span: Optional[float] = None, lost: Optional[float] = None,
                ratio: Optional[float] = None, extended: Optional[Tuple[int, int]] = None,
                absorbed: Sequence[int] = ()) -> str:
    """Audit attribute of a line in PAGE ``custom`` syntax (kraken parses ``tag {k:v;}`` and ignores this tag).

    :param tier: Tier.
    :type tier: str
    :param sim: Text similarity.
    :type sim: Optional[float]
    :param len_score: Length score.
    :type len_score: Optional[float]
    :param edition_index: Edition line number.
    :type edition_index: Optional[int]
    :param span: Best span similarity to the own edition line.
    :type span: Optional[float]
    :param lost: Share of the edition line outside that span.
    :type lost: Optional[float]
    :param ratio: Local length ratio checked by the length guard.
    :type ratio: Optional[float]
    :param extended: Pixels the ink walk moved the ``(left, right)`` ends.
    :type extended: Optional[Tuple[int, int]]
    :param absorbed: Segmenter indices absorbed into the line.
    :type absorbed: Sequence[int]
    :return: e.g. ``"alignment {tier:anchor; sim:0.62; len:0.81; edition:3;}"`` (``span``, ``lost``,
        ``ratio``, ``extended:<left>/<right>`` and ``absorbed:<i>/<j>`` appended when given).
    :rtype: str
    """
    parts = [f"tier:{tier}"]
    if sim is not None:
        parts.append(f"sim:{sim:.2f}")
    if len_score is not None:
        parts.append(f"len:{len_score:.2f}")
    if edition_index is not None:
        parts.append(f"edition:{edition_index}")
    for key, value in (("span", span), ("lost", lost), ("ratio", ratio)):
        if value is not None:
            parts.append(f"{key}:{value:.2f}")
    if extended is not None:
        parts.append(f"extended:{extended[0]}/{extended[1]}")
    if absorbed:
        parts.append("absorbed:" + "/".join(str(k) for k in absorbed))
    return "alignment {" + "; ".join(parts) + ";}"


def page_xml_bytes(image_path: str, width: int, height: int, lines: Sequence[OutLine], created: str) -> bytes:
    """A PAGE 2019-07-15 document with one text region (bytes out).

    :param image_path: Absolute image path for ``imageFilename``.
    :type image_path: str
    :param width: Image width.
    :type width: int
    :param height: Image height.
    :type height: int
    :param lines: Lines in reading order.
    :type lines: Sequence[OutLine]
    :param created: ISO timestamp for ``Metadata/Created``.
    :type created: str
    :return: UTF-8 XML with a declaration. The region's ``Coords`` is the convex hull of the lines.
    :rtype: bytes
    """
    ns = "{%s}" % PAGE_NS
    root = etree.Element(ns + "PcGts", nsmap={None: PAGE_NS, "xsi": XSI_NS})
    root.set(f"{{{XSI_NS}}}schemaLocation", f"{PAGE_NS} {PAGE_NS}/pagecontent.xsd")
    meta = etree.SubElement(root, ns + "Metadata")
    etree.SubElement(meta, ns + "Creator").text = "align_edition_lines (PGP edition text on ktivseg_e10 lines)"
    etree.SubElement(meta, ns + "Created").text = created
    etree.SubElement(meta, ns + "LastChange").text = created
    page = etree.SubElement(root, ns + "Page", imageFilename=image_path, imageWidth=str(width), imageHeight=str(height))
    region = etree.SubElement(page, ns + "TextRegion", id="r0")
    hull = convex_hull(p for line in lines for p in line.boundary) or [(0, 0), (width - 1, 0), (width - 1, height - 1)]
    etree.SubElement(region, ns + "Coords", points=format_points(hull, width, height))
    for line in lines:
        el = etree.SubElement(region, ns + "TextLine", id=line.line_id, custom=line.custom)
        etree.SubElement(el, ns + "Coords", points=format_points(line.boundary, width, height))
        etree.SubElement(el, ns + "Baseline", points=format_points(line.baseline, width, height))
        equiv = etree.SubElement(el, ns + "TextEquiv")
        etree.SubElement(equiv, ns + "Unicode").text = line.text
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", pretty_print=True)


def line_id(group: Sequence[SegLine]) -> str:
    """XML id of a (merged) line from its segmenter indices.

    :param group: Lines.
    :type group: Sequence[SegLine]
    :return: ``l<i>`` or ``l<i>_<j>...`` (indices ascending).
    :rtype: str
    """
    return "l" + "_".join(str(i) for i in sorted(l.index for l in group))


def out_lines(result: PageResult) -> List[OutLine]:
    """The ``TextLine`` records of an aligned page: path order, then noise lines.

    A line that carries text takes its geometry from the ink walk (:attr:`PageResult.geometry`) when there is
    one; segmenter lines the walk absorbed into such a line are not written on their own.

    :param result: Page with an alignment.
    :type result: PageResult
    :return: One record per path line (matched groups, skipped segmenter lines) and per noise line.
    :rtype: List[OutLine]
    """
    alignment = result.alignment
    by_step = {p.step: p for p in alignment.pairs}
    absorbed = result.absorbed_segments()
    lines: List[OutLine] = []
    for n, step in enumerate(alignment.steps):
        if step.kind == "skip_e":
            continue
        group = [alignment.order[i] for i in step.s]
        baseline, boundary = merged_geometry(group)
        if step.kind == "skip_s":
            if group[0].index not in absorbed:
                lines.append(OutLine(line_id(group), tuple(baseline), tuple(boundary), "",
                                     custom_attr(TIER_UNMATCHED)))
            continue
        p = by_step[n]
        target = result.edition[p.e].target
        tier = p.tier if target else f"{p.tier}_untargeted"
        text = target if p.tier in ACCEPTED_TIERS else ""
        ext = result.geometry.get(p.step) if text else None
        if ext is not None:
            baseline, boundary = list(ext.baseline), list(ext.boundary)
        lines.append(OutLine(line_id(group), tuple(baseline), tuple(boundary), text,
                             custom_attr(tier, p.sim, p.len_score, result.edition[p.e].index,
                                         p.cover_sim if p.cover_sim else None,
                                         p.cover_lost if p.cover_sim else None,
                                         p.len_ratio if p.len_ratio else None,
                                         (ext.left.extended_px, ext.right.extended_px) if ext else None,
                                         ext.absorbed if ext else ())))
    by_index = {l.index: l for l in result.segments}
    for idx in result.noise:
        if idx in absorbed:
            continue
        line = by_index[idx]
        lines.append(OutLine(line_id([line]), line.baseline, line.boundary, "", custom_attr(TIER_NOISE)))
    return lines


# ----------------------------------------------------------------------------- I/O

def load_editions(editions_dir: Path) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    """Edition texts and manifest rows by stem (no image is decoded).

    :param editions_dir: ``pgp_arabic_editions_v2`` (a ``DatasetDict`` saved to disk + ``manifest.jsonl``).
    :type editions_dir: Path
    :return: ``({stem: answer}, {stem: manifest row})``.
    :rtype: Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]
    """
    import datasets
    answers: Dict[str, str] = {}
    for split in datasets.load_from_disk(str(editions_dir)).values():
        for row in split.select_columns(["stem", "answer"]):
            answers[row["stem"]] = row["answer"]
    manifest: Dict[str, Dict[str, Any]] = {}
    manifest_path = editions_dir / "manifest.jsonl"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    manifest[row["stem"]] = row
    return answers, manifest


def page_files(pages_dir: Path, limit: Optional[int] = None) -> List[Path]:
    """Segmenter JSON files of a pages directory, sorted.

    :param pages_dir: Directory of ``<stem>.json`` files.
    :type pages_dir: Path
    :param limit: Keep only the first N.
    :type limit: Optional[int]
    :return: Paths (``summary.jsonl`` and hidden files excluded).
    :rtype: List[Path]
    """
    files = sorted(p for p in pages_dir.glob("*.json") if not p.name.startswith("."))
    return files[:limit] if limit is not None else files


def split_key(canonical_id: str, pgpid: str, stem: str = "") -> str:
    """Split key of a page: its document, or the page itself when ``stem`` is given (``--split-by-page``).

    :param canonical_id: Canonical id (may be ``""``).
    :type canonical_id: str
    :param pgpid: PGP id.
    :type pgpid: str
    :param stem: Page stem; non-empty keys the page alone, so pages of one document can split apart.
    :type stem: str
    :return: ``genizah/page:<stem>``, ``genizah/<canonical id>`` or ``genizah/pgpid:<id>``.
    :rtype: str
    """
    if stem:
        return f"{SET_NAME}/page:{stem}"
    return f"{SET_NAME}/{canonical_id}" if canonical_id else f"{SET_NAME}/pgpid:{pgpid}"


def process_page(path: Path, answers: Dict[str, str], manifest: Dict[str, Dict[str, Any]],
                 held_out: Tuple[Set[str], Set[str]], params: Params) -> PageResult:
    """Read and align one page (no output is written).

    :param path: Segmenter JSON.
    :type path: Path
    :param answers: Edition texts by stem.
    :type answers: Dict[str, str]
    :param manifest: Manifest rows by stem.
    :type manifest: Dict[str, Dict[str, Any]]
    :param held_out: ``(canonical ids, PGP ids)`` of the registered benchmarks.
    :type held_out: Tuple[Set[str], Set[str]]
    :param params: Thresholds.
    :type params: Params
    :return: The page result with its split assigned.
    :rtype: PageResult
    """
    page = json.loads(path.read_text(encoding="utf-8"))
    stem = page.get("stem") or path.stem
    row = manifest.get(stem, {})
    pgpid = str(page.get("pgpid") or row.get("pgpid") or "")
    canonical = page.get("canonical_id") or row.get("canonical_id") or ""
    size = tuple(int(v) for v in page["image_size"])
    result = PageResult(stem, pgpid, canonical, page.get("image_path") or row.get("image_path", ""), size)
    result.split = hash_split(split_key(canonical, pgpid, stem if params.split_by_page else ""), params.val_share)
    if canonical in held_out[0] or pgpid in held_out[1]:
        result.skip = SKIP_REGISTRY
        return result
    if stem not in answers:
        result.skip = SKIP_NO_EDITION
        return result
    if not page.get("lines"):
        result.skip = SKIP_NO_LINES
        result.edition = edition_lines(answers[stem], params)
        return result
    result.segments, result.noise, result.edition, result.alignment, result.skip = align_page(page, answers[stem], params)
    return result


def check_image_size(result: PageResult) -> bool:
    """Whether the image file has the size the segmenter recorded (header read only).

    :param result: Page result.
    :type result: PageResult
    :return: True when ``Image.open(path).size == image_size``.
    :rtype: bool
    """
    with Image.open(result.image_path) as im:
        return tuple(im.size) == tuple(result.image_size)


def apply_geometry(result: PageResult, params: Params) -> None:
    """Read the page image, extend the target lines to their ink, gate them and resolve the final polygons'
    overlaps (in place: :func:`extend_pairs`, :func:`gate_extents`, :func:`final_overlaps`).

    :param result: Aligned page with no skip reason.
    :type result: PageResult
    :param params: Thresholds.
    :type params: Params
    :return: None. The skip reason becomes :data:`SKIP_NO_ACCEPTED` when the gate leaves no accepted target.
    :rtype: None
    """
    height = page_line_height(result)
    with Image.open(result.image_path) as im:
        ink = line_extent.page_ink(im, height)
    extend_pairs(result, ink, params)
    gate_extents(result, params)
    final_overlaps(result, ink, params)
    if not result.accepted():
        result.skip = SKIP_NO_ACCEPTED


def write_page(result: PageResult, xml_dir: Path, created: str) -> Path:
    """Write the PAGE XML of an aligned page.

    :param result: Page with an alignment and no skip reason.
    :type result: PageResult
    :param xml_dir: ``<out-root>/page_xml/genizah``.
    :type xml_dir: Path
    :param created: Timestamp.
    :type created: str
    :return: Written path.
    :rtype: Path
    """
    xml_dir.mkdir(parents=True, exist_ok=True)
    path = xml_dir / f"{result.stem}.xml"
    path.write_bytes(page_xml_bytes(result.image_path, result.image_size[0], result.image_size[1],
                                    out_lines(result), created))
    return path


def write_lists(results: Sequence[PageResult], out_root: Path) -> Dict[str, Path]:
    """Write ``lists/genizah_train.lst`` and ``genizah_val.lst`` (absolute XML paths).

    :param results: Page results.
    :type results: Sequence[PageResult]
    :param out_root: Output root.
    :type out_root: Path
    :return: ``{split: list path}``.
    :rtype: Dict[str, Path]
    """
    lists_dir = out_root / "lists"
    lists_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in ("train", "val"):
        paths = [str(r.xml_path) for r in results if r.xml_path is not None and r.split == split]
        out[split] = lists_dir / f"{SET_NAME}_{split}.lst"
        out[split].write_text("".join(f"{p}\n" for p in paths), encoding="utf-8")
    return out


def page_summary(result: PageResult) -> Dict[str, Any]:
    """One row of ``pages.jsonl``.

    :param result: Page result.
    :type result: PageResult
    :return: Stem, ids, split, skip reason, line counts, ordering, cpp, identity, page sim, tiers.
    :rtype: Dict[str, Any]
    """
    a = result.alignment
    return {"stem": result.stem, "pgpid": result.pgpid, "canonical_id": result.canonical_id, "split": result.split,
            "skip": result.skip, "edition_lines": len(result.edition),
            "target_lines": sum(1 for e in result.edition if e.target), "segmenter_lines": len(result.segments),
            "noise": len(result.noise), "ordering": a.ordering if a else None, "cpp": round(a.cpp, 5) if a else None,
            "identity": a.identity if a else None, "page_sim": round(a.page_sim, 3) if a else None,
            "sim_margin": round(a.sim_margin, 3) if a else None, "guard_pairs": a.guard_pairs if a else None,
            "merges": sum(1 for p in a.pairs if len(p.s) > 1) if a else 0,
            "position_anchors": a.anchored() if a else 0, "partial": sum(1 for p in a.pairs if p.partial) if a else 0,
            "tiers": dict(a.tier_counts()) if a else {}, "accepted_targets": len(result.accepted()),
            "line_height": round(result.line_height, 1),
            "extended_lines": sum(1 for p in result.accepted() if p.step in result.geometry
                                  and extended_px(result.geometry[p.step])),
            "xml": str(result.xml_path) if result.xml_path else None}


def extended_px(ext: line_extent.LineExtent) -> int:
    """Pixels the ink walk added to a line at both ends.

    :param ext: Line geometry after the walk.
    :type ext: line_extent.LineExtent
    :return: ``left + right`` extension.
    :rtype: int
    """
    return ext.left.extended_px + ext.right.extended_px


def pair_records(result: PageResult) -> List[Dict[str, Any]]:
    """Rows of ``pairs.jsonl`` for one page: every matched pair with its evidence and geometry.

    :param result: Page result (pages without an alignment give no rows).
    :type result: PageResult
    :return: One dict per pair in path order: page, written line id, segmenter indices, edition line, tier,
        reason, whether the PAGE line carries the text, sim / span / lost (None without span evidence) /
        local length ratio, line height, extension and ink beyond per end (pixels and line heights; None
        when the walk did not run), absorbed and stolen segments, own ink outside the polygon, new overlap
        with another line (share and line), the text block bound and whether it stopped each end (with the
        ink past it, in line heights), the target text and the zero-shot read.
    :rtype: List[Dict[str, Any]]
    """
    a = result.alignment
    if a is None:
        return []
    rows = []
    height = result.line_height or page_line_height(result)
    for p in a.pairs:
        group = [a.order[i] for i in p.s]
        ext = result.geometry.get(p.step)
        target = result.edition[p.e].target
        row: Dict[str, Any] = {
            "stem": result.stem, "page_skip": result.skip, "line_id": line_id(group),
            "segments": sorted(l.index for l in group), "merged": len(group) > 1,
            "edition_index": result.edition[p.e].index, "tier": p.tier, "reason": p.reason,
            "written": bool(target) and p.tier in ACCEPTED_TIERS and result.xml_path is not None,
            "sim": round(p.sim, 3), "span": round(p.cover_sim, 3) if p.cover_sim else None,
            "lost": round(p.cover_lost, 3) if p.cover_sim else None,
            "len_ratio": round(p.len_ratio, 3) if p.len_ratio else None, "line_height": round(height, 1),
            "extended_px_left": None, "extended_px_right": None, "extended_lh_left": None,
            "extended_lh_right": None, "ink_beyond_left": None, "ink_beyond_right": None,
            "absorbed": [], "stolen": [], "capped": None, "polygon_rebuilt": None, "hull_fallback": None,
            "raised": None, "own_ink_outside": None, "overlap": None, "overlap_with": None, "block": None,
            "block_limited": None, "past_block_left": None, "past_block_right": None,
            "overlap_final": None, "overlap_final_with": None, "trimmed": None, "trim_cut_own": None,
            "end_shift_left": None, "end_shift_right": None, "crossed_lh_left": None, "crossed_lh_right": None,
            "hard": None, "spacing": round(result.spacing, 1) if result.spacing else None,
            "detected_above": None, "detected_below": None, "stubs_removed": None, "foreign_components": None,
            "pencil_components": None, "forced": [], "read_words": p.read_words, "edition_words": p.edition_words,
            "missing_words": [k for k, _ in p.missing], **result.notes.get(p.step, {}),
            "text": target, "read": merged_read(group)}
        if ext is not None:
            row.update({"extended_px_left": ext.left.extended_px, "extended_px_right": ext.right.extended_px,
                        "extended_lh_left": round(ext.left.extended_px / height, 3),
                        "extended_lh_right": round(ext.right.extended_px / height, 3),
                        "ink_beyond_left": round(ext.left.ink_beyond_px / height, 3),
                        "ink_beyond_right": round(ext.right.ink_beyond_px / height, 3),
                        "absorbed": list(ext.absorbed), "stolen": list(ext.stolen),
                        "capped": [ext.left.capped, ext.right.capped], "polygon_rebuilt": ext.rebuilt,
                        "hull_fallback": ext.hull_fallback, "raised": ext.raised,
                        "own_ink_outside": round(ext.own_ink_outside, 4), "overlap": round(ext.overlap, 4),
                        "overlap_with": ext.overlap_key,
                        "block": [round(v, 1) for v in ext.block] if ext.block else None,
                        "block_limited": [ext.left.limited, ext.right.limited],
                        "past_block_left": round(ext.left.past_block_px / height, 3),
                        "past_block_right": round(ext.right.past_block_px / height, 3),
                        "overlap_final": round(ext.overlap_final, 4), "overlap_final_with": ext.overlap_final_key,
                        "trimmed": ext.trimmed, "trim_cut_own": ext.trim_cut_own,
                        "end_shift_left": round(ext.end_shift[0], 3), "end_shift_right": round(ext.end_shift[1], 3),
                        "crossed_lh_left": round(ext.left.crossed_px / height, 3),
                        "crossed_lh_right": round(ext.right.crossed_px / height, 3),
                        "hard": ([None if v is None else round(v, 1) for v in ext.hard] if ext.hard else None),
                        "detected_above": ext.detected[0], "detected_below": ext.detected[1],
                        "stubs_removed": ext.stubs_removed, "foreign_components": ext.foreign_dropped,
                        "pencil_components": ext.pencil_dropped, "forced": list(ext.forced)})
        rows.append(row)
    return rows


def extension_stats(results: Sequence[PageResult], params: Params) -> Dict[str, Any]:
    """Ink-walk totals over the written pages.

    :param results: Page results.
    :type results: Sequence[PageResult]
    :param params: Thresholds used.
    :type params: Params
    :return: Whether extension / the gate were on, written lines with any extension (and with at least a
        quarter line height), pixels added per side, absorbed segments, rebuilt and raised polygons,
        convex-hull fallbacks, capped ends, ends the text block stopped, and target pairs rejected by the gate
        (ink beyond, steal, overlap, own ink outside).
    :rtype: Dict[str, Any]
    """
    out: Counter = Counter()
    for r in results:
        if r.alignment is None:
            continue
        for p in r.alignment.pairs:
            ext = r.geometry.get(p.step)
            if ext is None:
                continue
            out["walked_pairs"] += 1
            out["ink_beyond_rejections"] += p.reason == REJECT_INK_BEYOND
            out["steal_rejections"] += p.reason == REJECT_STEAL
            out["overlap_rejections"] += p.reason in (REJECT_OVERLAP, REJECT_OVERLAP_TRIM)
            out["overlap_trim_rejections"] += p.reason == REJECT_OVERLAP_TRIM
            out["own_ink_rejections"] += p.reason == REJECT_OWN_INK
            out["end_shift_rejections"] += p.reason == REJECT_END_SHIFT
            out["missing_word_rejections"] += p.reason == REJECT_MISSING_WORD
            note = r.notes.get(p.step, {})
            out["supralinear_boxes"] += str(note.get("supralinear", "")).startswith("box:")
            out["supralinear_ink"] += str(note.get("supralinear", "")).startswith("ink:")
            if r.xml_path is None or p.tier not in ACCEPTED_TIERS:
                continue
            out["block_limited_ends"] += ext.left.limited + ext.right.limited
            out["crossed_ends"] += (ext.left.crossed_px > 0) + (ext.right.crossed_px > 0)
            out["hard_limited_lines"] += bool(ext.hard and any(v is not None for v in ext.hard))
            out["trimmed_lines"] += ext.trimmed
            out["detected_above"] += ext.detected[0]
            out["detected_below"] += ext.detected[1]
            out["stubs_removed"] += ext.stubs_removed
            out["connected_only_ends"] += "connected_only_end" in note
            out["written_overlap_final_over_max"] += ext.overlap_final > params.max_overlap
            out["raised_lines"] += ext.raised
            height = r.line_height or 1.0
            out["written_lines_walked"] += 1
            out["extended_lines"] += extended_px(ext) > 0
            out["extended_lines_quarter_lh"] += max(ext.left.extended_px, ext.right.extended_px) >= 0.25 * height
            out["extended_px_left"] += ext.left.extended_px
            out["extended_px_right"] += ext.right.extended_px
            out["absorbed_segments"] += len(ext.absorbed)
            out["polygons_rebuilt"] += ext.rebuilt
            out["hull_fallbacks"] += ext.hull_fallback
            out["capped_ends"] += ext.left.capped + ext.right.capped
    return {"extend_to_ink": params.extend_to_ink, "max_ink_beyond": params.max_ink_beyond,
            "max_lost": params.max_lost, "drop_count_match": params.drop_count_match,
            "raise_top": params.raise_top and params.extend_to_ink, "block_slack": params.block_slack,
            "max_overlap": params.max_overlap, "max_own_ink_outside": params.max_own_ink_outside,
            "drop_interpolated": params.drop_interpolated, "drop_duplicate_texts": params.drop_duplicate_texts,
            "column_blocks": params.column_blocks, "max_end_shift": params.max_end_shift,
            "supralinear": params.supralinear, "own_ink_reach": params.own_ink_reach,
            **{k: int(v) for k, v in sorted(out.items())}, "own_ink_calibration": own_ink_calibration(results)}


def own_ink_calibration(results: Sequence[PageResult]) -> Dict[str, Any]:
    """Distribution of ``own_ink_outside`` over the walked target pairs, written vs rejected by the gate.

    :param results: Page results.
    :type results: Sequence[PageResult]
    :return: ``{"written": {n, median, p90, p99, max}, "own_ink_rejected": {...}}`` (empty dicts without
        values).
    :rtype: Dict[str, Any]
    """
    written: List[float] = []
    rejected: List[float] = []
    for r in results:
        if r.alignment is None:
            continue
        for p in r.alignment.pairs:
            ext = r.geometry.get(p.step)
            if ext is None:
                continue
            if r.xml_path is not None and p.tier in ACCEPTED_TIERS:
                written.append(ext.own_ink_outside)
            elif p.reason == REJECT_OWN_INK:
                rejected.append(ext.own_ink_outside)

    def summary(values: List[float]) -> Dict[str, Any]:
        """Quantiles of a list.

        :param values: Values.
        :return: ``{n, median, p90, p99, max}`` (empty without values).
        """
        if not values:
            return {}
        arr = np.asarray(values)
        return {"n": int(arr.size), "median": round(float(np.median(arr)), 4),
                "p90": round(float(np.percentile(arr, 90)), 4), "p99": round(float(np.percentile(arr, 99)), 4),
                "max": round(float(arr.max()), 4)}

    return {"written": summary(written), "own_ink_rejected": summary(rejected)}


def alignment_stats(results: Sequence[PageResult], params: Params, sources: Dict[str, str]) -> Dict[str, Any]:
    """Summary of a run.

    :param results: Page results.
    :type results: Sequence[PageResult]
    :param params: Thresholds used.
    :type params: Params
    :param sources: ``{"pages_dir": ..., "editions": ...}``.
    :type sources: Dict[str, str]
    :return: Pages seen / aligned / skipped by reason, orderings, line counts (pairs per tier and path
        moves on the written pages), edition lines total vs target vs placed, yield, splits.
    :rtype: Dict[str, Any]
    """
    written = [r for r in results if r.xml_path is not None]
    tiers: Counter = Counter()
    targeted: Counter = Counter()
    reasons: Counter = Counter()
    lines: Counter = Counter()
    for r in results:
        lines["segmenter_lines"] += len(r.segments)
        lines["noise_lines"] += len(r.noise)
    for r in written:
        lines["written_segmenter_lines"] += len(r.segments)
        for p in r.alignment.pairs:
            tiers[p.tier] += 1
            if r.edition[p.e].target:
                targeted[p.tier] += 1
                reasons[p.reason] += p.tier == TIER_REJECTED
        lines["merged_pairs"] += sum(1 for p in r.alignment.pairs if len(p.s) > 1)
        lines["unmatched_segmenter_lines"] += sum(1 for s in r.alignment.steps if s.kind == "skip_s")
        lines["unmatched_edition_lines"] += sum(1 for s in r.alignment.steps if s.kind == "skip_e")
    eligible = [r for r in results if r.skip not in (SKIP_REGISTRY, SKIP_NO_EDITION)]
    edition_total = sum(len(r.edition) for r in eligible)
    target_total = sum(sum(1 for e in r.edition if e.target) for r in eligible)
    placed = sum(len(r.accepted()) for r in written)
    target_written = sum(sum(1 for e in r.edition if e.target) for r in written)
    splits = {s: {"pages": sum(1 for r in written if r.split == s),
                  "lines": sum(len(r.accepted()) for r in written if r.split == s)} for s in ("train", "val")}
    return {
        "built": datetime.datetime.now().isoformat(timespec="seconds"), **sources,
        "params": dataclasses.asdict(params),
        "pages_seen": len(results), "pages_aligned": len(written),
        "pages_skipped": dict(Counter(r.skip for r in results if r.skip).most_common()),
        "orderings": dict(Counter(r.alignment.ordering for r in written).most_common()),
        "count_match_pages": sum(1 for r in written if r.alignment.tier_counts()["count_match"]),
        "lines": dict(lines),
        "pairs_by_tier": dict(tiers), "target_pairs_by_tier": dict(targeted),
        "rejected_target_pairs_by_reason": dict(Counter({k: v for k, v in reasons.items() if v}).most_common()),
        "edition_lines_total": edition_total, "edition_target_lines": target_total, "placed_lines": placed,
        "yield_pct_of_targets": round(100 * placed / target_total, 2) if target_total else 0.0,
        "yield_pct_of_targets_on_aligned_pages": round(100 * placed / target_written, 2) if target_written else 0.0,
        "splits": splits,
        "extension": extension_stats(results, params),
    }


# ----------------------------------------------------------------------------- audit sheet

def cap_per_page(pool: Sequence[Tuple[PageResult, Pair]], cap: int) -> List[Tuple[PageResult, Pair]]:
    """Reorder a shuffled pool so that no page contributes more than ``cap`` items before every page has had
    its turn: the first ``cap`` items of each page in pool order, then the overflow in pool order.

    :param pool: ``(page, pair)`` items in random order.
    :type pool: Sequence[Tuple[PageResult, Pair]]
    :param cap: Items per page in the first pass (0 or less: no cap).
    :type cap: int
    :return: The reordered items (same multiset).
    :rtype: List[Tuple[PageResult, Pair]]
    """
    if cap <= 0:
        return list(pool)
    seen: Counter = Counter()
    first: List[Tuple[PageResult, Pair]] = []
    overflow: List[Tuple[PageResult, Pair]] = []
    for item in pool:
        seen[item[0].stem] += 1
        (first if seen[item[0].stem] <= cap else overflow).append(item)
    return first + overflow


def audit_sample(results: Sequence[PageResult], n_accepted: int, n_rejected: int, seed: int,
                 page_cap: int = AUDIT_PAGE_CAP) -> List[Tuple[PageResult, Pair]]:
    """Seeded sample of accepted target lines spread across tiers and pages, plus rejected target lines.

    :param results: Page results.
    :type results: Sequence[PageResult]
    :param n_accepted: Accepted lines wanted (shared round-robin across tiers).
    :type n_accepted: int
    :param n_rejected: Rejected lines wanted.
    :type n_rejected: int
    :param seed: Random seed.
    :type seed: int
    :param page_cap: At most this many lines per (page, tier) while other pages of the tier remain
        (:func:`cap_per_page`).
    :type page_cap: int
    :return: ``(page, pair)`` items, accepted first.
    :rtype: List[Tuple[PageResult, Pair]]
    """
    rng = random.Random(seed)
    pools: Dict[str, List[Tuple[PageResult, Pair]]] = {t: [] for t in ACCEPTED_TIERS + (TIER_REJECTED,)}
    for r in results:
        if r.alignment is None or r.skip == SKIP_REGISTRY:
            continue
        for p in r.alignment.pairs:
            if r.edition[p.e].target:
                pools[p.tier].append((r, p))
    for tier, pool in pools.items():
        rng.shuffle(pool)
        pools[tier] = cap_per_page(pool, page_cap)
    picked: List[Tuple[PageResult, Pair]] = []
    cursor = {t: 0 for t in ACCEPTED_TIERS}
    while len(picked) < n_accepted and any(cursor[t] < len(pools[t]) for t in ACCEPTED_TIERS):
        for t in ACCEPTED_TIERS:
            if len(picked) < n_accepted and cursor[t] < len(pools[t]):
                picked.append(pools[t][cursor[t]])
                cursor[t] += 1
    return picked + pools[TIER_REJECTED][:n_rejected]


def crop_box(points: Sequence[Point], width: int, height: int, pad: int = AUDIT_PAD) -> Tuple[int, int, int, int]:
    """Padded bounding box of a polygon, clamped into the image.

    :param points: Polygon points.
    :type points: Sequence[Point]
    :param width: Image width.
    :type width: int
    :param height: Image height.
    :type height: int
    :param pad: Padding in pixels.
    :type pad: int
    :return: ``(left, top, right, bottom)`` for ``Image.crop``.
    :rtype: Tuple[int, int, int, int]
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad),
            min(width, int(math.ceil(max(xs))) + pad), min(height, int(math.ceil(max(ys))) + pad))


def audit_box(polygons: Sequence[Sequence[Point]], width: int, height: int, line_px: float,
              margin: float = AUDIT_MARGIN) -> Tuple[int, int, int, int]:
    """Crop box of an audit line: wide enough to show ink the polygon missed beyond its ends.

    :param polygons: Polygons to show (segmenter and extended).
    :type polygons: Sequence[Sequence[Point]]
    :param width: Image width.
    :type width: int
    :param height: Image height.
    :type height: int
    :param line_px: Line height in pixels.
    :type line_px: float
    :param margin: Horizontal margin in line heights.
    :type margin: float
    :return: ``(left, top, right, bottom)``: the polygons' bounding box widened by ``margin`` line heights
        on each side and by ``max(AUDIT_PAD, line_px / 4)`` above and below, clamped into the image.
    :rtype: Tuple[int, int, int, int]
    """
    xs = [p[0] for poly in polygons for p in poly]
    ys = [p[1] for poly in polygons for p in poly]
    dx, dy = int(round(margin * line_px)), int(round(max(AUDIT_PAD, line_px / 4)))
    return (max(0, int(min(xs)) - dx), max(0, int(min(ys)) - dy),
            min(width, int(math.ceil(max(xs))) + dx), min(height, int(math.ceil(max(ys))) + dy))


def outline_panel(im: Image.Image, box: Tuple[int, int, int, int], polygon: Sequence[Point],
                  colour: Tuple[int, int, int], line_px: float) -> Image.Image:
    """A crop with one polygon drawn as a closed outline.

    :param im: Page image (RGB).
    :type im: Image.Image
    :param box: Crop box.
    :type box: Tuple[int, int, int, int]
    :param polygon: Polygon in page coordinates.
    :type polygon: Sequence[Point]
    :param colour: Outline colour.
    :type colour: Tuple[int, int, int]
    :param line_px: Line height in pixels (sets the stroke width).
    :type line_px: float
    :return: The annotated crop.
    :rtype: Image.Image
    """
    crop = im.crop(box)
    pts = [(x - box[0], y - box[1]) for x, y in polygon]
    ImageDraw.Draw(crop).line(pts + pts[:1], fill=colour, width=max(2, int(round(line_px / 40))))
    return crop


def audit_image(im: Image.Image, r: PageResult, p: Pair, margin: float) -> Image.Image:
    """Audit crop of one line: the segmenter polygon in red; when the ink walk changed it, a second panel
    below with the written polygon in green (same crop box, so the two align).

    :param im: Page image (RGB).
    :type im: Image.Image
    :param r: Page result.
    :type r: PageResult
    :param p: Pair.
    :type p: Pair
    :param margin: Horizontal margin in line heights.
    :type margin: float
    :return: The crop (at most :data:`AUDIT_MAX_WIDTH` wide).
    :rtype: Image.Image
    """
    group = [r.alignment.order[i] for i in p.s]
    _, before = merged_geometry(group)
    ext = r.geometry.get(p.step)
    after = list(ext.boundary) if ext is not None and ext.rebuilt else None
    line_px = r.line_height or page_line_height(r)
    box = audit_box([before] + ([after] if after else []), im.width, im.height, line_px, margin)
    out = outline_panel(im, box, before, (220, 30, 30), line_px)
    if after:
        second = outline_panel(im, box, after, (20, 160, 60), line_px)
        both = Image.new("RGB", (out.width, out.height * 2 + 6), (128, 128, 128))
        both.paste(out, (0, 0))
        both.paste(second, (0, out.height + 6))
        out = both
    if out.width > AUDIT_MAX_WIDTH:
        out = out.resize((AUDIT_MAX_WIDTH, max(1, round(out.height * AUDIT_MAX_WIDTH / out.width))))
    return out


def write_audit(sample: Sequence[Tuple[PageResult, Pair]], gate_dir: Path, margin: float = AUDIT_MARGIN) -> Path:
    """Write the audit sheet: one crop per sampled line (:func:`audit_image`), ``margin`` line heights wider
    than the polygon on each side so that ink beyond the polygon ends is visible.

    :param sample: ``(page, pair)`` items.
    :type sample: Sequence[Tuple[PageResult, Pair]]
    :param gate_dir: ``<out-root>/gate/genizah_alignment``.
    :type gate_dir: Path
    :param margin: Horizontal crop margin in line heights.
    :type margin: float
    :return: Path of ``index.html``.
    :rtype: Path
    """
    crops = gate_dir / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    for old in crops.glob("*.png"):
        old.unlink()
    by_page: Dict[str, List[Tuple[int, PageResult, Pair]]] = {}
    for n, (r, p) in enumerate(sample):
        by_page.setdefault(r.image_path, []).append((n, r, p))
    rows: Dict[int, str] = {}
    for image_path, entries in by_page.items():
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            for n, r, p in entries:
                group = [r.alignment.order[i] for i in p.s]
                name = f"{n:03d}_{p.tier}_{r.stem}_{line_id(group)}.png"
                audit_image(im, r, p, margin).save(crops / name)
                read = merged_read(group)
                ext = r.geometry.get(p.step)
                height = r.line_height or 1.0
                walk = (f'extended L {ext.left.extended_px} px / R {ext.right.extended_px} px'
                        f'{" (absorbed " + ", ".join(map(str, ext.absorbed)) + ")" if ext.absorbed else ""} · '
                        f'ink beyond L {ext.left.ink_beyond_px / height:.2f} / R {ext.right.ink_beyond_px / height:.2f}'
                        f' line heights · own ink outside {ext.own_ink_outside:.2f} · overlap {ext.overlap:.2f} · '
                        if ext is not None else '')
                rows[n] = (f'<figure class="{p.tier}"><img src="crops/{html.escape(name)}" alt="">'
                           f'<p class="ed" dir="rtl" lang="ar">{html.escape(r.edition[p.e].target)}</p>'
                           f'<p class="meta">#{n} · <b>{p.tier}</b>{" (" + p.reason + ")" if p.reason else ""} · '
                           f'sim {p.sim:.2f} · len {p.len_score:.2f} · '
                           + (f'span {p.cover_sim:.2f} (lost {p.cover_lost:.2f}) · ' if p.cover_sim else '')
                           + (f'local length ratio {p.len_ratio:.2f} · ' if p.len_ratio else '') + walk +
                           f'{r.stem} edition line {r.edition[p.e].index} · segments {line_id(group)} · '
                           f'{r.alignment.ordering}</p>'
                           f'<p class="read" dir="rtl" lang="ar">zero-shot read (placement only): '
                           f'{html.escape(read) or "∅"}</p></figure>')
    items = [rows[n] for n in sorted(rows)]
    page = ("<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
            "<title>Genizah alignment audit</title><style>"
            ":root{--bg:#faf8f4;--fg:#222;--muted:#666;--line:#ddd}"
            "@media (prefers-color-scheme: dark){:root{--bg:#1c1b19;--fg:#eee;--muted:#aaa;--line:#444}}"
            "body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;"
            "max-width:1700px;padding:16px}figure{margin:0 0 28px;border-bottom:1px solid var(--line);"
            "padding-bottom:12px}img{max-width:100%;height:auto;display:block;background:#fff}"
            ".ed{font-size:22px;margin:6px 0}.meta,.read{color:var(--muted);font-size:13px;margin:2px 0}"
            ".rejected .meta b{color:#c33}</style></head><body>"
            f"<h1>Genizah edition → segmenter alignment audit</h1><p>{len(sample)} lines "
            f"(seed {AUDIT_SEED}); text under each crop is the human edition line written to the PAGE XML. "
            "Rejected lines are shown for contrast and are NOT written as targets. Red outline: the segmenter's "
            "polygon; a second panel with a green outline: the polygon after the ink walk (written for accepted "
            f"lines). Crops reach {margin:g} line heights beyond the polygon on each side.</p>" + "".join(items)
            + "</body></html>")
    index = gate_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    return index


# ----------------------------------------------------------------------------- run

def run(pages_dir: Path, editions_dir: Path, out_root: Path, params: Params, limit: Optional[int] = None,
        held_out: Optional[Tuple[Set[str], Set[str]]] = None) -> Dict[str, Any]:
    """Align every page, write PAGE XML, lists, statistics and the audit sheet.

    PAGE files in ``page_xml/genizah`` left over from an earlier run that this run did not write are
    removed, so the directory always matches the lists.

    :param pages_dir: Segmenter JSON directory.
    :type pages_dir: Path
    :param editions_dir: Edition dataset directory.
    :type editions_dir: Path
    :param out_root: Output root.
    :type out_root: Path
    :param params: Thresholds.
    :type params: Params
    :param limit: Pages to process (first N by name).
    :type limit: Optional[int]
    :param held_out: Registered benchmark documents (default: :func:`registered_benchmark_documents`).
    :type held_out: Optional[Tuple[Set[str], Set[str]]]
    :return: The statistics as written.
    :rtype: Dict[str, Any]
    """
    out_root = out_root.absolute()
    held_out = held_out if held_out is not None else registered_benchmark_documents()
    answers, manifest = load_editions(editions_dir)
    files = page_files(pages_dir, limit)
    logger.info("%d pages, %d edition pages", len(files), len(answers))
    created = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    xml_dir = out_root / "page_xml" / SET_NAME
    results = []
    for n, path in enumerate(files, 1):
        result = process_page(path, answers, manifest, held_out, params)
        if not result.skip and not check_image_size(result):
            result.skip = SKIP_IMAGE_SIZE
        if not result.skip and needs_geometry(params):
            apply_geometry(result, params)
        if not result.skip:
            result.xml_path = write_page(result, xml_dir, created)
        results.append(result)
        if n % 25 == 0:
            logger.info("  %d / %d pages, %d written", n, len(files), sum(1 for r in results if r.xml_path))
    written = {r.xml_path for r in results if r.xml_path}
    for stale in xml_dir.glob("*.xml") if xml_dir.exists() else []:
        if stale not in written:
            stale.unlink()
    write_lists(results, out_root)
    stats = alignment_stats(results, params, {"pages_dir": str(pages_dir), "editions": str(editions_dir)})
    gate_dir = out_root / "gate" / f"{SET_NAME}_alignment"
    gate_dir.mkdir(parents=True, exist_ok=True)
    with open(gate_dir / "pages.jsonl", "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(page_summary(r), ensure_ascii=False) + "\n")
    with open(out_root / "pairs.jsonl", "w", encoding="utf-8") as fh:
        for r in results:
            for row in pair_records(r):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats["pairs"] = str(out_root / "pairs.jsonl")
    stats["audit_sheet"] = str(write_audit(audit_sample(results, params.audit_n, params.audit_rejected,
                                                        params.audit_seed, params.audit_page_cap), gate_dir,
                                           params.audit_margin))
    (out_root / "alignment_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
    return stats


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser; every :class:`Params` field is an option (a boolean field is a
    ``--name`` / ``--no-name`` pair).

    :return: Parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pages-dir", type=Path, default=PAGES_DIR)
    parser.add_argument("--editions", type=Path, default=EDITIONS_DIR)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    for f in dataclasses.fields(Params):
        flag = "--" + f.name.replace("_", "-")
        if isinstance(f.default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            parser.add_argument(flag, type=type(f.default), default=f.default)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Command-line entry point.

    :param argv: Arguments (default ``sys.argv[1:]``).
    :type argv: Optional[List[str]]
    :return: None.
    :rtype: None
    """
    args = build_parser().parse_args(argv)
    params = Params(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Params)})
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stats = run(args.pages_dir, args.editions, args.out_root, params, args.limit)
    print(json.dumps({k: v for k, v in stats.items() if k != "params"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
