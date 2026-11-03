"""Stage 2 (sides) of the novel-rows build: place every target side on a photograph by reading the photographs.

A side is placed by reads only (never by a file name or a canvas order). The statistics are those of the legacy-store
row builder (:mod:`src.finetuning.qwen_hebrew.build_legacy_store_rows`), generalised over the script:

1. **Skeletons and n-grams.** Hebrew-script targets: letters only, finals folded, points dropped
   (``build_legacy_store_rows.hebrew_skeleton``), letter 4-grams with 3-grams as the second signal. Arabic-script
   targets: the rasm (``arabic_bench_coverage.rasm_skeleton``) of ``arabic_script.arabic_letters``, rasm 5-grams (the
   side matcher's length) with 3-grams as the second signal. A target holding both scripts is scored on the script
   with more skeleton letters.
2. **Reads.** Per photograph (sha256) the best usable read for the side's script: Kraken first for Hebrew script, the
   page reader (``vlm``) first for Arabic script, Gemini only when nothing else is usable. A read is unusable when it
   errored, is blank (or says ``NO TEXT``), loops (``finish`` looping / length, or ``arabic_script.repeat_share`` at or
   above ``side_match.MAX_REPEAT_SHARE``), has no line, or has fewer than ``MIN_READ_LETTERS`` letters of the script.
3. **Null.** One pool per (script, reader family): the best usable read of that family of every photograph, exact
   duplicate texts once. A side's null is its clipped recall against the pool reads of photographs of OTHER document
   groups (its own group, every group sharing one of its candidate's photographs or its text, and its candidate's own
   photographs are left out), per read-length bin, p95 / p99 with bootstrap CIs (``threshold_stats``); a bin with
   fewer than ``MIN_BIN_NULL`` reads falls back to the unstratified null, and a null of fewer reads than that makes the
   pair unusable (``null_too_small``).
4. **Duplicates and shared texts.** Two targets of one document group whose skeletons share at least
   :data:`DUPLICATE_SHARE` of their letters (Dice on the letters of a Levenshtein alignment) are versions of one text:
   the longer is kept, the shorter is ``dropped`` (``duplicate_version_shorter``). A skeleton found in several
   document groups is a shared text: ``build_legacy_store_rows.decide_ownership`` (called with adapted inputs) names
   the owner; the other copies are ``dropped`` (``shared_text_not_owner``) or, with no owner, a tie or a contested
   lead, go to review.
5. **Assignment** per decision unit (:func:`decision_units`): one candidate, or the candidates of one document group
   that share a photograph (the two faces of a leaf edited as two candidates), whose sides compete for the union of
   their photographs, as the legacy-store build places the sections of one document. Hungarian assignment of sides
   to distinct photographs on the primary recall (an agreeing prior and the photograph's origin only break ties). A
   pair is accepted at p99, or at p95 when an agreeing prior exists (the edition's own side marker and the
   photograph's canvas label or slip side name the same side; never for a single image). A photograph of another
   photo campaign (origin family) on which the side clears its acceptance rule too, no other side clears p95 and
   which no other side holds, is another photograph of the same page: among those the preferred origin wins
   (:data:`ORIGIN_PREFERENCE`). Within the chosen photograph's campaign the side must clear p95 on no other
   photograph (``side_clears_p95_on_another_image``), unless it clears p99 on its photograph, recalls at least
   ``side_match.RECALL_RATIO`` times as much there as on every such photograph, and each such photograph is held by
   another side with a higher recall there or by none (basis ``p99_ratio2``, flag-only code ``spill_ratio_ok``).
   Arabic-script sides must also beat the best other photograph of the campaign by ``side_match.MIN_MARGIN``.
   Review also when another side clears p95 on the photograph (``other_side_clears_p95_on_image``), the 3-gram recall
   misses its p95 or fewer than ``MIN_MATCHED_GRAMS`` primary n-grams matched. The mirror of the ratio rule excuses
   that conflict when it is the only blocking reason, the side clears p99 on its photograph, and every conflicting
   side is accepted on its own photograph with at least ``side_match.RECALL_RATIO`` times its recall here and recalls
   less here than the side does (basis ``p99_ratio2_mirror``, flag-only code ``conflict_ratio_ok``; two faces that
   excuse each other are accepted together). A side with a single readable photograph needs p99.
6. **Gate** (pass-through sources and KTIV pages, :data:`GATE_SOURCES`): the photograph is given; the read-vs-target
   recall must reach the p95 of the side's null (``read_recall_below_p95`` otherwise).
7. A photograph accepted for two targets goes to review (``image_accepted_for_another_document`` across document
   groups, ``image_accepted_for_another_target`` inside one).

Every decision carries ``competitors``: one entry per photograph considered (role, read, recalls, null, clears flags),
the chosen one first with a ``coverage`` key (read letters / target letters, read lines / target lines, and the Kraken
line count of the photograph) for the quality stage.

Usage::

    python -m src.finetuning.qwen_hebrew.novel_build.side_assign --work-dir DIR [--n-boot 500] [--seed 1030]
        [--sources pgp,ktiv] [--limit N]
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import Levenshtein
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.datasets.evaluations import arabic_script as ar
from src.datasets.evaluations.helper_eval_scripts import build_arabic_benchmark as bab
from src.datasets.evaluations.helper_eval_scripts.arabic_bench_coverage import ngram_counts, rasm_skeleton
from src.finetuning.qwen_hebrew import build_legacy_store_rows as lsr
from src.finetuning.qwen_hebrew import side_match as sm
from src.finetuning.qwen_hebrew.novel_build import paths
from src.finetuning.qwen_hebrew.novel_build.schema import (Candidate, ImageRef, ReadRecord, SideDecision, TargetSide,
                                                           read_jsonl, write_jsonl)

HEBREW, ARABIC = lsr.HEBREW, lsr.ARABIC
SCRIPTS = (HEBREW, ARABIC)
#: (primary, secondary) letter n-gram lengths per script
SCRIPT_NGRAMS: Dict[str, Tuple[int, int]] = {HEBREW: (lsr.PRIMARY_N, 3), ARABIC: (sm.NGRAM, 3)}
#: reader families in order of preference per script (Gemini only when nothing else is usable)
READER_PREFERENCE: Dict[str, Tuple[str, ...]] = {HEBREW: ("kraken", "vlm", "gemini"),
                                                 ARABIC: ("vlm", "kraken", "gemini")}
#: preferred photograph origin when several photographs of one page pass (first is best)
ORIGIN_PREFERENCE: Tuple[str, ...] = ("ktiv_zip", "iiif_nli", "iiif_cudl", "bodleian", "iiif_pgp", "cudl_local",
                                      "fjp_local", "fjp_bucket", "pgpid_mirror")
#: origins of one photo campaign (copies of one library's photographs); any other origin is its own campaign
CAMPAIGNS: Dict[str, str] = {"ktiv_zip": "nli", "iiif_nli": "nli", "iiif_cudl": "cudl", "cudl_local": "cudl",
                             "fjp_local": "legacy_store", "fjp_bucket": "legacy_store", "iiif_pgp": "pgp",
                             "pgpid_mirror": "pgp"}
#: sources whose photograph is given: the read-vs-target recall is a gate, not an assignment
GATE_SOURCES: Tuple[str, ...] = ("apd_v3", "aodeh_v2_1", "ktiv")
#: the origin of the given photograph of a gate source
GATE_ORIGINS: Dict[str, str] = {"apd_v3": "apd_local", "aodeh_v2_1": "aodeh_local", "ktiv": "ktiv_zip"}
READABLE_STATUSES: Tuple[str, ...] = ("present", "fetched")
DUPLICATE_SHARE = 0.9                          # Dice of two skeletons' aligned letters: one text in two versions
MIN_SHARED_TEXT_LETTERS = lsr.MIN_TARGET_LETTERS
MIN_MARGIN = sm.MIN_MARGIN
RECALL_RATIO = sm.RECALL_RATIO                 # the chosen photograph's recall over a same-campaign spill's (2.0)
RATIO_BASIS = "p99_ratio2"
MIRROR_BASIS = "p99_ratio2_mirror"
MAX_REPEAT_SHARE = sm.MAX_REPEAT_SHARE
MIN_READ_LETTERS = lsr.MIN_READ_LETTERS
PRIOR_EPS = lsr.PRIOR_EPS["side_label"]        # an agreeing prior breaks a tie of recalls, never more
ORIGIN_EPS = 1e-9                              # per preference rank; smaller than the prior's tie-break
READS_FILE = Path("reads") / "reads.jsonl"
CONFLICT_FREE_ROLES = ("chosen", "assigned", "alternative")

# reason codes
NO_IMAGE = "no_image"
NO_USABLE_IMAGE = "no_usable_image"
NO_READ = "no_read"
NO_USABLE_READ = "no_usable_read"
READ_LOOPING = "read_looping"
NULL_TOO_SMALL = "null_too_small"
SIDE_WITHOUT_GRAMS = "side_without_grams"
NOT_FOUND = "not_found"
SINGLE_BELOW_P99 = "single_image_below_p99"
NOT_P99_WITHOUT_PRIOR = "clears_p95_not_p99_without_agreeing_prior"
ON_IMAGE_OF_OTHER_SIDE = "clears_p95_on_image_assigned_to_another_side"
ON_ANOTHER_IMAGE = "side_clears_p95_on_another_image"
OTHER_SIDE_ON_IMAGE = "other_side_clears_p95_on_image"
N3_BELOW_P95 = "n3_below_p95"
MARGIN_REASON = "runner_up_within_margin"
GATE_BELOW_P95 = "read_recall_below_p95"
DUPLICATE_SHORTER = "duplicate_version_shorter"
NOT_OWNER = "shared_text_not_owner"
IMAGE_TWICE_DOCUMENT = "image_accepted_for_another_document"
IMAGE_TWICE_GROUP = "image_accepted_for_another_target"
SPILL_RATIO_OK = "spill_ratio_ok"
CONFLICT_RATIO_OK = "conflict_ratio_ok"
NULL_PROBLEMS = ("no_null", NULL_TOO_SMALL)
#: codes that never send a side to review (an accepted decision may carry them; they mark rule-recovered acceptances)
FLAG_ONLY = frozenset({SPILL_RATIO_OK, CONFLICT_RATIO_OK})

_HEBREW_SIDE = re.compile(r"ע\s*[\"״'׳]?\s*([אב])")
_BARE_SIDE = re.compile(r"^\(?\s*(?:f(?:ol)?\.?\s*)?\d*\s*([rv])\s*\.?\s*\)?$")


def under_matched_reason(n: int) -> str:
    """Reason code of an acceptance resting on too few matched primary n-grams.

    :param n: Primary n-gram length of the side's script.
    :type n: int
    :return: ``under_<MIN_MATCHED_GRAMS>_matched_<n>grams`` (``under_5_matched_4grams`` for Hebrew script).
    :rtype: str
    """
    return f"under_{lsr.MIN_MATCHED_GRAMS}_matched_{n}grams"


# ===================================================================================================================
# Text, reads, photographs
# ===================================================================================================================

def skeletons_of(text: str) -> Dict[str, str]:
    """Skeleton of a text in each script.

    :param text: A target or a read.
    :type text: str
    :return: ``{"hebrew": hebrew_skeleton(text), "arabic": rasm of arabic_letters(text)}``.
    :rtype: Dict[str, str]
    """
    return {HEBREW: lsr.hebrew_skeleton(text or ""), ARABIC: rasm_skeleton(ar.arabic_letters(text or ""))}


def family_of(reader: str) -> str:
    """Reader family of a reader id.

    :param reader: ``kraken:<model>``, ``vlm:<key>``, ``gemini:<model>``.
    :type reader: str
    :return: The part before the first colon (``kraken``, ``vlm``, ``gemini``).
    :rtype: str
    """
    return (reader or "").split(":", 1)[0]


@dataclass
class ReadView:
    """One read with what the scorer needs.

    :ivar record: The read.
    :ivar family: Reader family (:func:`family_of`).
    :ivar skeletons: Skeleton per script (:func:`skeletons_of`).
    :ivar repeat_share: ``arabic_script.repeat_share`` of the text.
    :ivar fingerprint: ``side_match.text_fingerprint`` of the text (exact duplicate texts enter a pool once).
    """
    record: ReadRecord
    family: str
    skeletons: Dict[str, str]
    repeat_share: float
    fingerprint: str

    def problem(self, script: str) -> str:
        """Why the read cannot be compared with a side of this script.

        :param script: Side script.
        :type script: str
        :return: ``read_error``, ``looping``, ``blank``, ``zero_lines``, ``under_20_<script>_letters`` or ``""``.
        :rtype: str
        """
        rec = self.record
        text = (rec.text or "").strip()
        if rec.finish == "error":
            return "read_error"
        if rec.finish in ("looping", "length") or self.repeat_share >= MAX_REPEAT_SHARE:
            return "looping"
        if rec.finish == "blank" or not text or lsr.NO_TEXT.match(text):
            return "blank"
        if rec.n_lines <= 0:
            return "zero_lines"
        if len(self.skeletons.get(script, "")) < MIN_READ_LETTERS:
            return f"under_{MIN_READ_LETTERS}_{script}_letters"
        return ""


def make_view(record: ReadRecord) -> ReadView:
    """Wrap a read.

    :param record: The read.
    :type record: ReadRecord
    :return: The view.
    :rtype: ReadView
    """
    text = record.text or ""
    return ReadView(record, family_of(record.reader), skeletons_of(text), ar.repeat_share(text),
                    sm.text_fingerprint(text))


def views_by_sha(reads: Iterable[ReadRecord]) -> Dict[str, List[ReadView]]:
    """Reads grouped by photograph.

    :param reads: Read records.
    :type reads: Iterable[ReadRecord]
    :return: ``{sha256: [ReadView]}`` in input order.
    :rtype: Dict[str, List[ReadView]]
    """
    out: Dict[str, List[ReadView]] = defaultdict(list)
    for record in reads:
        out[record.sha256].append(make_view(record))
    return dict(out)


def _family_rank(family: str, script: str) -> int:
    """Preference rank of a reader family for a script.

    :param family: Reader family.
    :type family: str
    :param script: Side script.
    :type script: str
    :return: Index in :data:`READER_PREFERENCE` (unknown families last).
    :rtype: int
    """
    order = READER_PREFERENCE.get(script, READER_PREFERENCE[HEBREW])
    return order.index(family) if family in order else len(order)


def choose_read(views: Sequence[ReadView], script: str,
                family: Optional[str] = None) -> Tuple[Optional[ReadView], Optional[ReadView]]:
    """The best usable read of one photograph for a script, and the read that explains why there is none.

    Usable reads rank by reader family (:data:`READER_PREFERENCE`), then by letters of the script, then by the latest
    ``read_at``. Without a usable read, the fallback is the most preferred read (for its problem).

    :param views: Reads of the photograph.
    :type views: Sequence[ReadView]
    :param script: Side script.
    :type script: str
    :param family: Only reads of this family (a pool), or None for all.
    :type family: Optional[str]
    :return: ``(usable read or None, fallback read or None)``.
    :rtype: Tuple[Optional[ReadView], Optional[ReadView]]
    """
    pool = [v for v in views if family is None or v.family == family]
    if not pool:
        return None, None
    ranked = sorted(pool, key=lambda v: (_family_rank(v.family, script), -len(v.skeletons.get(script, "")),
                                         _neg_text(v.record.read_at or ""), v.record.reader))
    usable = [v for v in ranked if not v.problem(script)]
    return (usable[0], usable[0]) if usable else (None, ranked[0])


def _neg_text(text: str) -> Tuple[int, ...]:
    """A key that sorts strings in descending order inside an ascending sort.

    :param text: A string (an ISO timestamp).
    :type text: str
    :return: Negated code points.
    :rtype: Tuple[int, ...]
    """
    return tuple(-ord(ch) for ch in text)


def origin_rank(origin: str) -> int:
    """Preference rank of a photograph origin.

    :param origin: ``ImageRef.origin``.
    :type origin: str
    :return: Index in :data:`ORIGIN_PREFERENCE`; unlisted origins rank after every listed one.
    :rtype: int
    """
    return ORIGIN_PREFERENCE.index(origin) if origin in ORIGIN_PREFERENCE else len(ORIGIN_PREFERENCE)


def campaign_of(origin: str) -> str:
    """Photo campaign of an origin (copies of one library's photographs share one).

    :param origin: ``ImageRef.origin``.
    :type origin: str
    :return: :data:`CAMPAIGNS` entry, or the origin itself.
    :rtype: str
    """
    return CAMPAIGNS.get(origin, origin)


def distinct_images(images: Sequence[ImageRef]) -> List[ImageRef]:
    """The readable photographs of a candidate, byte-identical copies folded onto the preferred origin.

    :param images: ``Candidate.images``.
    :type images: Sequence[ImageRef]
    :return: Photographs with a sha256 and status ``present`` / ``fetched``, one per sha256, in first-seen order.
    :rtype: List[ImageRef]
    """
    best: Dict[str, ImageRef] = {}
    for image in images:
        if image.status not in READABLE_STATUSES or not image.sha256:
            continue
        if image.sha256 not in best or origin_rank(image.origin) < origin_rank(best[image.sha256].origin):
            best[image.sha256] = image
    order = list(dict.fromkeys(im.sha256 for im in images if im.sha256 in best))
    return [best[sha] for sha in order]


def side_of(label: Optional[str]) -> str:
    """Recto / verso named by a canvas label, a slip side or an edition's side marker.

    :param label: ``1r``, ``f. 2v``, ``Recto``, ``ע"א``, ``r`` ... or None.
    :type label: Optional[str]
    :return: ``recto``, ``verso`` or ``""``.
    :rtype: str
    """
    text = (label or "").strip()
    if not text:
        return ""
    hebrew = _HEBREW_SIDE.search(text)
    if hebrew:
        return "recto" if hebrew.group(1) == "א" else "verso"
    named = bab.side_of_label(text)
    if named:
        return named
    bare = _BARE_SIDE.match(text.lower())
    if bare:
        return "recto" if bare.group(1) == "r" else "verso"
    return ""


# ===================================================================================================================
# Sides
# ===================================================================================================================

@dataclass
class Side:
    """One target side with its skeleton and n-grams.

    :ivar key: ``<row_id>#<target index>``.
    :ivar cand: The candidate.
    :ivar index: Position among the candidate's targets.
    :ivar target: The target side.
    :ivar script: Majority script of the target's skeletons (``""`` without letters).
    :ivar skeleton: Skeleton in that script.
    :ivar ngrams: ``(primary, secondary)`` n-gram lengths.
    :ivar grams: n-gram counts of the skeleton per n.
    """
    key: str
    cand: Candidate
    index: int
    target: TargetSide
    script: str
    skeleton: str
    ngrams: Tuple[int, int]
    grams: Dict[int, Counter] = field(default_factory=dict)

    @property
    def primary(self) -> int:
        """Primary n-gram length.

        :return: 4 (Hebrew script) or 5 (Arabic script).
        :rtype: int
        """
        return self.ngrams[0]

    @property
    def secondary(self) -> int:
        """Second-signal n-gram length.

        :return: 3.
        :rtype: int
        """
        return self.ngrams[1]

    @property
    def group(self) -> str:
        """Document group of the candidate.

        :return: ``Candidate.document_group``.
        :rtype: str
        """
        return self.cand.document_group

    @property
    def has_grams(self) -> bool:
        """Whether the side can be scored.

        :return: True when the skeleton has a primary n-gram.
        :rtype: bool
        """
        return bool(self.script) and bool(self.grams.get(self.primary))


def make_side(cand: Candidate, index: int, target: TargetSide) -> Side:
    """A side with its skeleton and n-grams.

    :param cand: Candidate.
    :type cand: Candidate
    :param index: Position among the candidate's targets.
    :type index: int
    :param target: The target side.
    :type target: TargetSide
    :return: The side.
    :rtype: Side
    """
    skeletons = skeletons_of(target.text)
    script = lsr.majority_script(skeletons)
    skeleton = skeletons.get(script, "") if script else ""
    ngrams = SCRIPT_NGRAMS[script] if script else SCRIPT_NGRAMS[HEBREW]
    return Side(f"{cand.row_id}#{index}", cand, index, target, script, skeleton, ngrams,
                {n: ngram_counts(skeleton, n) for n in ngrams})


def edition_side(target: TargetSide) -> str:
    """The side the edition itself names: its side marker (or, when the marker is a bare flag, its side key).

    :param target: Target side.
    :type target: TargetSide
    :return: ``recto``, ``verso`` or ``""`` (no marker: no prior).
    :rtype: str
    """
    if not target.side_marker:
        return ""
    return side_of(target.side_marker) or side_of(target.side_key)


def prior_agrees(side: Side, image: ImageRef) -> bool:
    """Whether the edition's own side marker and the photograph's canvas label or slip side name the same side.

    :param side: Side.
    :type side: Side
    :param image: Photograph.
    :type image: ImageRef
    :return: True only when both are known and agree.
    :rtype: bool
    """
    named = edition_side(side.target)
    return bool(named) and named == side_of(image.label)


def shared_letters(a: str, b: str) -> int:
    """Letters two skeletons share in a Levenshtein alignment.

    :param a: Skeleton.
    :type a: str
    :param b: Skeleton.
    :type b: str
    :return: Total length of the alignment's ``equal`` blocks.
    :rtype: int
    """
    return sum(i2 - i1 for tag, i1, i2, _, _ in Levenshtein.opcodes(a, b) if tag == "equal")


def duplicate_share(a: str, b: str) -> float:
    """Dice share of the letters two skeletons share.

    :param a: Skeleton.
    :type a: str
    :param b: Skeleton.
    :type b: str
    :return: ``2 * shared / (len(a) + len(b))``; 0.0 for two empty strings.
    :rtype: float
    """
    total = len(a) + len(b)
    return 2.0 * shared_letters(a, b) / total if total else 0.0


def duplicate_targets(sides: Sequence[Side]) -> Dict[str, Tuple[str, float]]:
    """Versions of one text inside a document group: every version but the longest.

    Two sides of one group in the same script whose skeletons share at least :data:`DUPLICATE_SHARE` of their letters
    are one text; versions are grouped transitively (union-find, as ``build_legacy_store_rows.duplicate_versions``
    does) and the one with the most skeleton letters is kept (the first on a tie).

    :param sides: All sides.
    :type sides: Sequence[Side]
    :return: ``{dropped key: (kept key, share with the kept version)}``.
    :rtype: Dict[str, Tuple[str, float]]
    """
    by_group: Dict[str, List[Side]] = defaultdict(list)
    for side in sides:
        if side.has_grams:
            by_group[side.group].append(side)
    dropped: Dict[str, Tuple[str, float]] = {}
    for members in by_group.values():
        parent = {s.key: s.key for s in members}

        def find(key: str) -> str:
            """Union-find root.

            :param key: Side key.
            :type key: str
            :return: Root key.
            :rtype: str
            """
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        for i, side in enumerate(members):
            for other in members[:i]:
                if side.script == other.script and duplicate_share(side.skeleton, other.skeleton) >= DUPLICATE_SHARE:
                    parent[find(side.key)] = find(other.key)
        groups: Dict[str, List[Side]] = defaultdict(list)
        for side in members:
            groups[find(side.key)].append(side)
        for versions in groups.values():
            if len(versions) < 2:
                continue
            keep = max(versions, key=lambda s: (len(s.skeleton), -versions.index(s)))
            for side in versions:
                if side is not keep:
                    dropped[side.key] = (keep.key, duplicate_share(side.skeleton, keep.skeleton))
    return dropped


def shared_text_groups(sides: Sequence[Side], skip: Set[str]) -> List[List[Side]]:
    """Sides holding the same skeleton in different document groups.

    :param sides: All sides.
    :type sides: Sequence[Side]
    :param skip: Keys already dropped.
    :type skip: Set[str]
    :return: Groups of at least two sides from at least two document groups (skeletons of at least
        :data:`MIN_SHARED_TEXT_LETTERS` letters).
    :rtype: List[List[Side]]
    """
    by_text: Dict[Tuple[str, str], List[Side]] = defaultdict(list)
    for side in sides:
        if side.key not in skip and len(side.skeleton) >= MIN_SHARED_TEXT_LETTERS:
            by_text[(side.script, side.skeleton)].append(side)
    return [members for members in by_text.values() if len({s.group for s in members}) > 1]


def ownership(group: Sequence[Side], pairs: Mapping[str, Mapping[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Which document group owns a shared text: ``build_legacy_store_rows.decide_ownership`` on adapted inputs.

    The legacy function reads a side's key and document (``canonical_id``, here the document group) and, per pair,
    ``usable``, ``clears_p99`` and the primary recall under the key ``n4``; the adapter passes this side's primary
    recall (4-grams for Hebrew script, 5-grams for Arabic script) under that key.

    :param group: Sides of one shared text.
    :type group: Sequence[Side]
    :param pairs: ``{side key: {sha256: score_pair result}}``.
    :type pairs: Mapping[str, Mapping[str, Dict[str, Any]]]
    :return: Per side key: ``role`` (owner, not_owner, no_owner, tie, contested), ``best_n4``, ``owner``,
        ``group_docs``.
    :rtype: Dict[str, Dict[str, Any]]
    """
    records = [lsr.SideRecord(s.key, s.group, s.index, {}, s.script, s.skeleton, {}, None) for s in group]
    primary_key = f"n{lsr.PRIMARY_N}"
    adapted = {s.key: {sha: {"usable": bool(p.get("usable")), "clears_p99": bool(p.get("clears_p99")),
                             primary_key: float(p.get("recall") or 0.0)}
                       for sha, p in pairs.get(s.key, {}).items()}
               for s in group}
    return lsr.decide_ownership(records, adapted)


# ===================================================================================================================
# Null
# ===================================================================================================================

@dataclass
class Pool:
    """A null pool (``build_legacy_store_rows.NullPool``) with lookups for leaving reads out.

    :ivar null: The pool.
    :ivar family: Reader family of its reads.
    :ivar by_group: Pool positions per owning document group.
    :ivar by_sha: Pool position per photograph.
    """
    null: lsr.NullPool
    family: str
    by_group: Dict[str, np.ndarray]
    by_sha: Dict[str, int]


def build_pool(script: str, family: str, reads: Mapping[str, Sequence[ReadView]],
               owners: Mapping[str, Set[str]]) -> Pool:
    """The null pool of one script and reader family: the best usable read of that family per photograph, exact
    duplicate texts once (``build_legacy_store_rows.build_null_pool`` generalised over n-gram lengths and readers).

    :param script: :data:`HEBREW` or :data:`ARABIC`.
    :type script: str
    :param family: Reader family.
    :type family: str
    :param reads: Reads per photograph.
    :type reads: Mapping[str, Sequence[ReadView]]
    :param owners: Document groups per photograph.
    :type owners: Mapping[str, Set[str]]
    :return: The pool.
    :rtype: Pool
    """
    kept: List[ReadView] = []
    dropped: Counter = Counter()
    seen: Set[str] = set()
    for sha in sorted(reads):
        usable, fallback = choose_read(reads[sha], script, family)
        if usable is None:
            if fallback is not None:
                dropped[fallback.problem(script)] += 1
            continue
        if usable.fingerprint in seen:
            dropped["duplicate_text"] += 1
            continue
        seen.add(usable.fingerprint)
        kept.append(usable)
    index: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for n in SCRIPT_NGRAMS[script]:
        postings: Dict[str, Tuple[List[int], List[int]]] = defaultdict(lambda: ([], []))
        for i, view in enumerate(kept):
            for gram, count in ngram_counts(view.skeletons[script], n).items():
                postings[gram][0].append(i)
                postings[gram][1].append(count)
        index[n] = {g: (np.array(i, dtype=np.int64), np.array(c, dtype=np.float64)) for g, (i, c) in postings.items()}
    letters = np.array([len(v.skeletons[script]) for v in kept], dtype=np.int64)
    shas = [v.record.sha256 for v in kept]
    group_of = [frozenset(owners.get(sha, ())) for sha in shas]
    bins = np.array([lsr.length_bin(int(x)) for x in letters], dtype=object)
    null = lsr.NullPool(script, shas, group_of, letters, bins, index, dict(dropped))
    by_group: Dict[str, List[int]] = defaultdict(list)
    for i, groups in enumerate(group_of):
        for group in groups:
            by_group[group].append(i)
    return Pool(null, family, {g: np.array(v, dtype=np.int64) for g, v in by_group.items()},
                {sha: i for i, sha in enumerate(shas)})


def side_null(side: Side, pool: Pool, excluded_groups: Set[str], excluded_shas: Set[str], rng: np.random.Generator,
              n_boot: int = lsr.N_BOOT) -> Dict[str, Any]:
    """The null of one side: its recall against every pool read of another document group, per read-length bin
    (``build_legacy_store_rows.side_null`` generalised over the side's n-gram lengths; same output shape, so
    ``pick_threshold`` reads it).

    :param side: Side.
    :type side: Side
    :param pool: Its script's pool for one reader family.
    :type pool: Pool
    :param excluded_groups: Document groups whose reads never count.
    :type excluded_groups: Set[str]
    :param excluded_shas: Photographs that never count.
    :type excluded_shas: Set[str]
    :param rng: Random generator (bootstrap).
    :type rng: np.random.Generator
    :param n_boot: Bootstrap resamples for the primary n (the second signal gets none).
    :type n_boot: int
    :return: ``{"reads": k, "family": f, "n<primary>": {bin: stats}, "n<secondary>": {bin: stats}}``.
    :rtype: Dict[str, Any]
    """
    size = len(pool.null)
    mask = np.ones(size, dtype=bool)
    for group in excluded_groups:
        positions = pool.by_group.get(group)
        if positions is not None:
            mask[positions] = False
    for sha in excluded_shas:
        position = pool.by_sha.get(sha)
        if position is not None:
            mask[position] = False
    out: Dict[str, Any] = {"reads": int(mask.sum()), "family": pool.family}
    for n in side.ngrams:
        values = lsr.pool_recall(side.grams[n], pool.null.index[n], size)[mask] if size else np.zeros(0)
        bins = pool.null.bins[mask] if size else np.zeros(0, dtype=object)
        boot = n_boot if n == side.primary else 0
        per = {lsr.ALL_BINS: lsr.threshold_stats(values, rng, boot)}
        for b in lsr.BINS:
            per[b] = lsr.threshold_stats(values[bins == b], rng, boot)
        out[f"n{n}"] = per
    return out


class NullCache:
    """Null pools built on first use, and side nulls computed once per (side, reader family).

    :param reads: Reads per photograph.
    :type reads: Mapping[str, Sequence[ReadView]]
    :param owners: Document groups per photograph.
    :type owners: Mapping[str, Set[str]]
    :param rng: Random generator (bootstrap).
    :type rng: np.random.Generator
    :param n_boot: Bootstrap resamples.
    :type n_boot: int
    """

    def __init__(self, reads: Mapping[str, Sequence[ReadView]], owners: Mapping[str, Set[str]],
                 rng: np.random.Generator, n_boot: int) -> None:
        """Keep the inputs.

        :param reads: Reads per photograph.
        :type reads: Mapping[str, Sequence[ReadView]]
        :param owners: Document groups per photograph.
        :type owners: Mapping[str, Set[str]]
        :param rng: Random generator.
        :type rng: np.random.Generator
        :param n_boot: Bootstrap resamples.
        :type n_boot: int
        """
        self.reads = reads
        self.owners = owners
        self.rng = rng
        self.n_boot = n_boot
        self.pools: Dict[Tuple[str, str], Pool] = {}
        self.nulls: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def pool(self, script: str, family: str) -> Pool:
        """The pool of a script and reader family.

        :param script: Script.
        :type script: str
        :param family: Reader family.
        :type family: str
        :return: The pool (built on first use).
        :rtype: Pool
        """
        key = (script, family)
        if key not in self.pools:
            self.pools[key] = build_pool(script, family, self.reads, self.owners)
        return self.pools[key]

    def null(self, side: Side, family: str, excluded_groups: Set[str], excluded_shas: Set[str]) -> Dict[str, Any]:
        """The side's null against one reader family's pool.

        :param side: Side.
        :type side: Side
        :param family: Reader family of the read being scored.
        :type family: str
        :param excluded_groups: Document groups left out.
        :type excluded_groups: Set[str]
        :param excluded_shas: Photographs left out.
        :type excluded_shas: Set[str]
        :return: :func:`side_null` result (computed once per side and family).
        :rtype: Dict[str, Any]
        """
        key = (side.key, family)
        if key not in self.nulls:
            self.nulls[key] = side_null(side, self.pool(side.script, family), excluded_groups, excluded_shas, self.rng,
                                        self.n_boot)
        return self.nulls[key]


def score_pair(side: Side, read: Optional[ReadView], null: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Recall of a side in one photograph's read against the side's null (``build_legacy_store_rows.score_pair``
    generalised over the side's n-gram lengths).

    :param side: Side.
    :type side: Side
    :param read: The photograph's read (its best usable one, or the fallback that explains why none is usable).
    :type read: Optional[ReadView]
    :param null: :func:`side_null` result for the read's family, or None.
    :type null: Optional[Mapping[str, Any]]
    :return: ``usable`` False with ``why``; or ``recall`` (primary), ``secondary``, ``n<k>`` per n, the thresholds of
        the bin used (``null_n<k>``), ``matched`` (clipped primary n-grams found), ``z``, ``clears_p95``,
        ``clears_p99``, ``clears_p99_ci_hi``, ``n3_clears_p95``, ``reader``, ``read_letters``, ``read_lines``, ``bin``.
    :rtype: Dict[str, Any]
    """
    if read is None:
        return {"usable": False, "why": NO_READ}
    problem = read.problem(side.script or HEBREW)
    if problem:
        return {"usable": False, "why": problem, "reader": read.record.reader}
    if not side.has_grams:
        return {"usable": False, "why": SIDE_WITHOUT_GRAMS}
    if null is None:
        return {"usable": False, "why": "no_null", "reader": read.record.reader}
    skeleton = read.skeletons[side.script]
    bin_name = lsr.length_bin(len(skeleton))
    out: Dict[str, Any] = {"usable": True, "reader": read.record.reader, "family": read.family,
                           "read_letters": len(skeleton), "read_lines": read.record.n_lines, "bin": bin_name}
    for n in side.ngrams:
        read_grams = ngram_counts(skeleton, n)
        stats, used = lsr.pick_threshold(null, n, bin_name)
        if stats is None or stats["n"] < lsr.MIN_BIN_NULL:
            return {"usable": False, "why": NULL_TOO_SMALL, "reader": read.record.reader,
                    "null_reads": null.get("reads")}
        out[f"n{n}"] = lsr.clipped_recall(side.grams[n], read_grams)
        out[f"null_n{n}"] = {"bin_used": used, "n": stats["n"], "p95": stats["p95"], "p99": stats["p99"],
                             "mean": stats["mean"], "std": stats["std"],
                             **({"p95_ci": stats["p95_ci"], "p99_ci": stats["p99_ci"]} if "p99_ci" in stats else {})}
        if n == side.primary:
            out["matched"] = sum(min(count, read_grams.get(gram, 0)) for gram, count in side.grams[n].items())
    main = out[f"null_n{side.primary}"]
    out["recall"] = out[f"n{side.primary}"]
    out["secondary"] = out[f"n{side.secondary}"]
    out["z"] = (out["recall"] - main["mean"]) / main["std"] if main["std"] > 0 else None
    out["clears_p95"] = out["recall"] > main["p95"]
    out["clears_p99"] = out["recall"] > main["p99"]
    out["clears_p99_ci_hi"] = out["recall"] > main["p99_ci"][1] if "p99_ci" in main else None
    out["n3_clears_p95"] = out["secondary"] > out[f"null_n{side.secondary}"]["p95"]
    return out


# ===================================================================================================================
# Decisions
# ===================================================================================================================

def kraken_view(views: Sequence[ReadView]) -> Optional[ReadView]:
    """The Kraken read of a photograph with the most lines (errors left out), for the line-count signal.

    :param views: Reads of the photograph.
    :type views: Sequence[ReadView]
    :return: The read, or None.
    :rtype: Optional[ReadView]
    """
    kraken = [v for v in views if v.family == "kraken" and v.record.finish != "error"]
    return max(kraken, key=lambda v: v.record.n_lines) if kraken else None


def coverage_of(side: Side, read: ReadView, kraken: Optional[ReadView]) -> Dict[str, Any]:
    """Coverage signals of a placed side for the quality stage.

    :param side: Side.
    :type side: Side
    :param read: The read used as evidence.
    :type read: ReadView
    :param kraken: The photograph's Kraken read, if any (:func:`kraken_view`).
    :type kraken: Optional[ReadView]
    :return: ``reader``, ``read_letters``, ``target_letters`` (skeleton letters in the side's script),
        ``letters_ratio``, ``read_lines``, ``target_lines``, ``lines_ratio``, ``kraken_reader``, ``kraken_lines``,
        ``kraken_lines_ratio`` (ratios None when the denominator is 0).
    :rtype: Dict[str, Any]
    """
    target_letters = len(side.skeleton)
    read_letters = len(read.skeletons.get(side.script, ""))
    target_lines = len([line for line in (side.target.lines or side.target.text.splitlines()) if line.strip()])
    kraken_lines = kraken.record.n_lines if kraken is not None else None
    return {"reader": read.record.reader, "read_letters": read_letters, "target_letters": target_letters,
            "letters_ratio": read_letters / target_letters if target_letters else None,
            "read_lines": read.record.n_lines, "target_lines": target_lines,
            "lines_ratio": read.record.n_lines / target_lines if target_lines else None,
            "kraken_reader": kraken.record.reader if kraken is not None else None, "kraken_lines": kraken_lines,
            "kraken_lines_ratio": kraken_lines / target_lines if kraken_lines is not None and target_lines else None}


def _pair_entry(image: ImageRef, pair: Mapping[str, Any], role: str, prior: bool,
                holder: Optional[str]) -> Dict[str, Any]:
    """One ``competitors`` entry.

    :param image: Photograph.
    :type image: ImageRef
    :param pair: :func:`score_pair` result.
    :type pair: Mapping[str, Any]
    :param role: ``chosen``, ``assigned``, ``alternative``, ``spill`` or ``other``.
    :type role: str
    :param prior: Whether an agreeing prior names this photograph.
    :type prior: bool
    :param holder: Side key the assignment gave the photograph to, if any.
    :type holder: Optional[str]
    :return: Flat summary.
    :rtype: Dict[str, Any]
    """
    entry: Dict[str, Any] = {"role": role, "image_id": image.image_id, "sha256": image.sha256, "origin": image.origin,
                             "campaign": campaign_of(image.origin), "label": image.label, "prior_agrees": prior,
                             "assigned_to": holder, "usable": bool(pair.get("usable"))}
    if not pair.get("usable"):
        entry.update(why=pair.get("why"), reader=pair.get("reader"))
        return entry
    keys = ("reader", "read_letters", "read_lines", "bin", "recall", "secondary", "matched", "z", "clears_p95",
            "clears_p99", "clears_p99_ci_hi", "n3_clears_p95")
    entry.update({k: pair.get(k) for k in keys})
    for k, v in pair.items():
        if k.startswith("null_n"):
            entry[k] = v
    return entry


def _decision(outcome: str, reasons: Sequence[str], **extra: Any) -> Dict[str, Any]:
    """An internal decision record.

    :param outcome: One of the schema's outcomes.
    :type outcome: str
    :param reasons: Reason codes.
    :type reasons: Sequence[str]
    :param extra: ``image`` (chosen), ``assigned``, ``basis``, ``margin``, ``roles``, ``info``.
    :type extra: Any
    :return: The record.
    :rtype: Dict[str, Any]
    """
    out: Dict[str, Any] = {"outcome": outcome, "reasons": list(reasons), "image": None, "assigned": None, "basis": None,
                           "margin": None, "roles": {}, "info": []}
    out.update(extra)
    return out


def _unreadable_decision(side: Side, images: Sequence[ImageRef], pairs: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Decision of a side none of whose photographs has a usable pair.

    :param side: Side.
    :type side: Side
    :param images: The candidate's distinct photographs.
    :type images: Sequence[ImageRef]
    :param pairs: ``{sha256: score_pair result}`` of the side (an unusable pair may list the ``problems`` of every
        read of its photograph).
    :type pairs: Mapping[str, Dict[str, Any]]
    :return: ``unassigned`` (no read at all, no grams) or ``needs_review`` (unusable reads, with ``read_looping`` when
        any read of a photograph loops; a small null).
    :rtype: Dict[str, Any]
    """
    found = [pairs.get(im.sha256) or {} for im in images]
    whys = [p.get("why") for p in found]
    if not side.has_grams:
        return _decision("unassigned", [SIDE_WITHOUT_GRAMS])
    if not whys or all(w == NO_READ for w in whys):
        return _decision("unassigned", [NO_READ])
    if any(w in NULL_PROBLEMS for w in whys):
        return _decision("needs_review", [NULL_TOO_SMALL])
    looping = any(w == "looping" or "looping" in (p.get("problems") or []) for p, w in zip(found, whys))
    return _decision("needs_review", [NO_USABLE_READ] + ([READ_LOOPING] if looping else []))


def _basis(pair: Mapping[str, Any], single: bool, prior: bool) -> Optional[str]:
    """The rule that would accept a side on a photograph.

    :param pair: :func:`score_pair` result.
    :type pair: Mapping[str, Any]
    :param single: The side has one readable photograph.
    :type single: bool
    :param prior: An agreeing prior names the photograph (:func:`prior_agrees`).
    :type prior: bool
    :return: ``p99``, ``single_image_p99``, ``p95_with_prior`` or None.
    :rtype: Optional[str]
    """
    if not pair.get("usable"):
        return None
    if pair["clears_p99"]:
        return "single_image_p99" if single else "p99"
    if not single and pair["clears_p95"] and prior:
        return "p95_with_prior"
    return None


def decide_candidate(sides: Sequence[Side], images: Sequence[ImageRef],
                     pairs: Mapping[str, Mapping[str, Dict[str, Any]]],
                     blocked: Mapping[str, Sequence[str]]) -> Dict[str, Dict[str, Any]]:
    """Place the sides of one decision unit on its photographs (``build_legacy_store_rows.decide_document`` with the
    novel build's rules).

    The unit is one candidate, or the candidates of one document group that share a photograph
    (:func:`decision_units`): their sides compete for the union of their photographs in one assignment, and the
    conflict rules see every side of the unit. On top of ``decide_document``: a side with a single readable photograph
    needs p99; another photo campaign's photograph passing the same rule is an alternative and the preferred origin
    wins; the absolute margin holds for Arabic-script sides only; a spill (p95 on another photograph of the chosen
    photograph's campaign) is excused when the side clears p99 on its photograph, recalls at least
    :data:`RECALL_RATIO` times as much there as on every spill photograph, and every spill photograph is held by
    another side with a higher recall there or by none (basis ``p99_ratio2``, flag ``spill_ratio_ok``). Photographs
    are first chosen for every side, then the spills are judged against those choices. A side whose only blocking
    reason is ``other_side_clears_p95_on_image`` and which clears p99 on its photograph is accepted by the mirrored
    rule (basis ``p99_ratio2_mirror``, flag ``conflict_ratio_ok``) when every conflicting side is accepted on its own
    photograph with at least :data:`RECALL_RATIO` times its recall on this one and recalls less on this one than the
    side does; the accepted set is the largest that satisfies this for all its members (two faces that excuse each
    other pass together). Every conflict is recorded in ``info`` (the conflicting side's row id, side key, recall here
    and on its own photograph) for the eye sheet.

    :param sides: The unit's sides still competing (dropped versions and texts owned elsewhere removed).
    :type sides: Sequence[Side]
    :param images: The unit's distinct readable photographs.
    :type images: Sequence[ImageRef]
    :param pairs: ``{side key: {sha256: score_pair result}}``.
    :type pairs: Mapping[str, Mapping[str, Dict[str, Any]]]
    :param blocked: Sides that may not be accepted (shared text without an owner), with their review reasons.
    :type blocked: Mapping[str, Sequence[str]]
    :return: Per side key: ``outcome``, ``reasons`` (flag-only codes of :data:`FLAG_ONLY` included), ``image``
        (chosen photograph), ``assigned`` (the assignment's photograph), ``basis``, ``margin``, ``roles``
        (``{sha256: role}``).
    :rtype: Dict[str, Dict[str, Any]]
    """
    by_key = {s.key: s for s in sides}

    def pair(side: Side, image: ImageRef) -> Dict[str, Any]:
        """Pair score (unusable when missing).

        :param side: Side.
        :type side: Side
        :param image: Photograph.
        :type image: ImageRef
        :return: The score.
        :rtype: Dict[str, Any]
        """
        return pairs.get(side.key, {}).get(image.sha256) or {"usable": False, "why": NO_READ}

    def clears95(side: Side, image: ImageRef) -> bool:
        """Whether a side clears its p95 on a photograph.

        :param side: Side.
        :type side: Side
        :param image: Photograph.
        :type image: ImageRef
        :return: True for a usable pair above p95.
        :rtype: bool
        """
        p = pair(side, image)
        return bool(p.get("usable") and p["clears_p95"])

    def conflicts_on(side: Side, image: ImageRef) -> List[str]:
        """Other sides of the unit that clear their p95 on a photograph.

        :param side: Side.
        :type side: Side
        :param image: Photograph.
        :type image: ImageRef
        :return: Their keys.
        :rtype: List[str]
        """
        return [o.key for o in sides if o.key != side.key and clears95(o, image)]

    def basis_for(side: Side, image: ImageRef, single: bool) -> Optional[str]:
        """The rule that would accept a side on a photograph (:func:`_basis`).

        :param side: Side.
        :type side: Side
        :param image: Photograph.
        :type image: ImageRef
        :param single: The side has one readable photograph.
        :type single: bool
        :return: The basis or None.
        :rtype: Optional[str]
        """
        return _basis(pair(side, image), single, prior_agrees(side, image))

    competing = [s for s in sides if s.key not in blocked and s.has_grams
                 and any(pair(s, im).get("usable") for im in images)]
    weights = np.zeros((len(competing), len(images)))
    for r, side in enumerate(competing):
        for c, image in enumerate(images):
            p = pair(side, image)
            if p.get("usable"):
                weights[r, c] = (p["recall"] + (PRIOR_EPS if prior_agrees(side, image) else 0.0)
                                 + ORIGIN_EPS * (len(ORIGIN_PREFERENCE) - origin_rank(image.origin)))
    assigned: Dict[str, ImageRef] = {}
    if weights.size:
        rows, cols = linear_sum_assignment(weights, maximize=True)
        assigned = {competing[r].key: images[c] for r, c in zip(rows.tolist(), cols.tolist())
                    if pair(competing[r], images[c]).get("usable")}   # never a photograph the side cannot be read on
    holder = {image.sha256: key for key, image in assigned.items()}
    out: Dict[str, Dict[str, Any]] = {}
    plans: Dict[str, Dict[str, Any]] = {}
    for side in sides:
        side_pairs = pairs.get(side.key, {})
        if side.key in blocked:
            out[side.key] = _decision("needs_review", blocked[side.key])
            continue
        readable = [im for im in images if pair(side, im).get("usable")]
        if not side.has_grams or not readable:
            out[side.key] = _unreadable_decision(side, images, side_pairs)
            continue
        single = len(readable) == 1
        a = assigned.get(side.key)
        cleared = [im for im in readable if clears95(side, im)]
        if a is None or basis_for(side, a, single) is None:
            roles = {im.sha256: "spill" for im in cleared}
            if a is not None:
                roles[a.sha256] = "assigned"
            if a is not None and single:
                out[side.key] = _decision("needs_review", [SINGLE_BELOW_P99], assigned=a, roles=roles)
            elif not cleared:
                out[side.key] = _decision("unassigned", [NOT_FOUND], assigned=a, roles=roles)
            else:
                reasons: Set[str] = set()
                if a is not None and a in cleared:
                    reasons.add(NOT_P99_WITHOUT_PRIOR)
                for other in cleared:
                    if a is not None and other.sha256 == a.sha256:
                        continue
                    reasons.add(ON_IMAGE_OF_OTHER_SIDE if other.sha256 in holder else ON_ANOTHER_IMAGE)
                out[side.key] = _decision("needs_review", sorted(reasons), assigned=a, roles=roles)
            continue
        alternatives = [im for im in readable
                        if im.sha256 != a.sha256 and campaign_of(im.origin) != campaign_of(a.origin)
                        and basis_for(side, im, single) is not None and im.sha256 not in holder
                        and not conflicts_on(side, im)]
        clean = [im for im in [a] + alternatives if not conflicts_on(side, im)]
        final = min(clean or [a], key=lambda im: (origin_rank(im.origin), 0 if prior_agrees(side, im) else 1,
                                                  -pair(side, im)["recall"], images.index(im)))
        plans[side.key] = {"a": a, "final": final, "alternatives": alternatives, "readable": readable,
                           "single": single}
    taken = {plan["final"].sha256: key for key, plan in plans.items()}

    def spill_excused(side: Side, chosen: Mapping[str, Any], image: ImageRef) -> bool:
        """Whether a spill photograph is background next to the side's own photograph (the ratio rule).

        :param side: Side.
        :type side: Side
        :param chosen: The side's pair on its chosen photograph (clears p99).
        :type chosen: Mapping[str, Any]
        :param image: A photograph of the same campaign on which the side clears p95.
        :type image: ImageRef
        :return: True when the chosen recall is at least :data:`RECALL_RATIO` times the recall here and the
            photograph is chosen by (else assigned to) another side recalling more here, or by no side.
        :rtype: bool
        """
        recall = pair(side, image)["recall"]
        if chosen["recall"] < RECALL_RATIO * recall:
            return False
        other = taken.get(image.sha256) or holder.get(image.sha256)
        if other is None:
            return True
        if other == side.key:
            return False
        return float(pair(by_key[other], image).get("recall") or 0.0) > recall

    staged: Dict[str, Dict[str, Any]] = {}
    for key, plan in plans.items():
        side, a, final, single = by_key[key], plan["a"], plan["final"], plan["single"]
        p = pair(side, final)
        siblings = [im for im in plan["readable"]
                    if im.sha256 != final.sha256 and campaign_of(im.origin) == campaign_of(final.origin)]
        spill = [im for im in siblings if clears95(side, im)]
        excused = bool(spill) and bool(p["clears_p99"]) and all(spill_excused(side, p, im) for im in spill)
        runner = max((pair(side, im)["recall"] for im in siblings), default=None)
        margin = p["recall"] - runner if runner is not None else None
        reasons_list: List[str] = []
        if conflicts_on(side, final):
            reasons_list.append(OTHER_SIDE_ON_IMAGE)
        if spill and not excused:
            reasons_list.append(ON_ANOTHER_IMAGE)
        if not p["n3_clears_p95"]:
            reasons_list.append(N3_BELOW_P95)
        if p["matched"] < lsr.MIN_MATCHED_GRAMS:
            reasons_list.append(under_matched_reason(side.primary))
        # The absolute margin is the Arabic side matcher's rule (rasm 5-gram recall of a correct side runs 0.1-0.5).
        # Hebrew letter 4-gram recall of a correct side is often under 0.05 (legacy-store build: median 0.275,
        # min 0.011 at p99), so Hebrew sides keep the p95/p99 and conflict rules only.
        if margin is not None and margin < MIN_MARGIN and side.script == "arabic":
            reasons_list.append(MARGIN_REASON)
        basis = basis_for(side, final, single)
        if excused:
            reasons_list.append(SPILL_RATIO_OK)
            basis = RATIO_BASIS
        roles = {im.sha256: "alternative" for im in plan["alternatives"]}
        roles.update({im.sha256: "spill" for im in spill})
        if final.sha256 != a.sha256:
            roles[a.sha256] = "assigned"
        roles[final.sha256] = "chosen"
        staged[key] = {"reasons": reasons_list, "basis": basis, "margin": margin, "roles": roles,
                       "conflicts": conflicts_on(side, final), "clears_p99": bool(p["clears_p99"])}

    def blocking(key: str) -> List[str]:
        """Reasons of a staged side that send it to review.

        :param key: Side key.
        :type key: str
        :return: Its reasons outside :data:`FLAG_ONLY`.
        :rtype: List[str]
        """
        return [r for r in staged[key]["reasons"] if r not in FLAG_ONLY]

    def conflict_excused(key: str, other: str, accepted: Set[str]) -> bool:
        """The mirrored ratio rule for one conflicting side on a side's chosen photograph.

        :param key: The side.
        :type key: str
        :param other: A side clearing p95 on its chosen photograph.
        :type other: str
        :param accepted: Sides accepted (or still tentatively accepted).
        :type accepted: Set[str]
        :return: True when the other side is accepted on its own photograph with at least :data:`RECALL_RATIO` times
            its recall on this photograph, and recalls less on this photograph than the side does.
        :rtype: bool
        """
        if other not in plans or other not in accepted:
            return False
        here = plans[key]["final"]
        theirs = float(pair(by_key[other], here).get("recall") or 0.0)
        own = float(pair(by_key[other], plans[other]["final"]).get("recall") or 0.0)
        return own >= RECALL_RATIO * theirs and theirs < float(pair(by_key[key], here)["recall"])

    mirrored = {k for k in staged if blocking(k) == [OTHER_SIDE_ON_IMAGE] and staged[k]["clears_p99"]}
    accepted_set = {k for k in staged if not blocking(k)} | mirrored
    changed = True
    while changed:
        changed = False
        for key in sorted(mirrored & accepted_set):
            if not all(conflict_excused(key, other, accepted_set) for other in staged[key]["conflicts"]):
                accepted_set.discard(key)
                changed = True
    for key in sorted(mirrored & accepted_set):
        staged[key]["reasons"] = [r for r in staged[key]["reasons"] if r != OTHER_SIDE_ON_IMAGE] + [CONFLICT_RATIO_OK]
        staged[key]["basis"] = MIRROR_BASIS
    for key, stage in staged.items():
        here = plans[key]["final"]
        info = [{"role": "conflict", "row_id": by_key[o].cand.row_id, "side_key": by_key[o].target.side_key,
                 "recall_here": pair(by_key[o], here).get("recall"),
                 "own_sha256": plans[o]["final"].sha256 if o in plans else None,
                 "recall_own": pair(by_key[o], plans[o]["final"]).get("recall") if o in plans else None,
                 "placed": o in plans, "accepted": o in accepted_set}
                for o in stage["conflicts"]]
        out[key] = _decision("needs_review" if blocking(key) else "accepted", stage["reasons"], image=here,
                             assigned=plans[key]["a"], basis=stage["basis"], margin=stage["margin"],
                             roles=stage["roles"], info=info)
    return {side.key: out[side.key] for side in sides}


def gate_image(cand: Candidate, images: Sequence[ImageRef]) -> Optional[ImageRef]:
    """The given photograph of a gate source: the first of the source's own origin, else the preferred origin.

    :param cand: Candidate.
    :type cand: Candidate
    :param images: Its distinct readable photographs.
    :type images: Sequence[ImageRef]
    :return: The photograph, or None.
    :rtype: Optional[ImageRef]
    """
    own = [im for im in images if im.origin == GATE_ORIGINS.get(cand.source)]
    if own:
        return own[0]
    return min(images, key=lambda im: (origin_rank(im.origin), images.index(im))) if images else None


def decide_gate(side: Side, image: ImageRef, pair: Mapping[str, Any]) -> Dict[str, Any]:
    """The read-vs-target gate of a side whose photograph is given.

    :param side: Side.
    :type side: Side
    :param image: The given photograph.
    :type image: ImageRef
    :param pair: :func:`score_pair` result on it.
    :type pair: Mapping[str, Any]
    :return: ``accepted`` (basis ``gate_p95``) when the recall reaches the p95 of the side's null (and is above 0);
        ``needs_review`` with ``read_recall_below_p95``, unusable reads or a small null; ``unassigned`` without a read.
    :rtype: Dict[str, Any]
    """
    roles = {image.sha256: "chosen"}
    if not pair.get("usable"):
        decision = _unreadable_decision(side, [image], {image.sha256: dict(pair)})
        decision["roles"] = roles
        return decision
    p95 = pair[f"null_n{side.primary}"]["p95"]
    if pair["recall"] >= p95 and pair["recall"] > 0.0:
        return _decision("accepted", [], image=image, assigned=image, basis="gate_p95", roles=roles)
    return _decision("needs_review", [GATE_BELOW_P95], assigned=image, roles=roles)


def image_conflicts(decisions: Mapping[str, Dict[str, Any]], sides: Mapping[str, Side]) -> List[str]:
    """Send accepted pairs to review when one photograph is accepted for two targets.

    :param decisions: Decisions by side key (modified in place).
    :type decisions: Mapping[str, Dict[str, Any]]
    :param sides: Sides by key.
    :type sides: Mapping[str, Side]
    :return: Keys moved to review.
    :rtype: List[str]
    """
    by_sha: Dict[str, List[str]] = defaultdict(list)
    for key, decision in decisions.items():
        if decision["outcome"] == "accepted":
            by_sha[decision["image"].sha256].append(key)
    moved = []
    for keys in by_sha.values():
        if len(keys) < 2:
            continue
        reason = IMAGE_TWICE_DOCUMENT if len({sides[k].group for k in keys}) > 1 else IMAGE_TWICE_GROUP
        for key in keys:
            decisions[key].update(outcome="needs_review", reasons=decisions[key]["reasons"] + [reason])
            decisions[key]["info"].append({"role": "image_shared_with", "keys": [k for k in keys if k != key]})
            moved.append(key)
    return moved


# ===================================================================================================================
# Whole set
# ===================================================================================================================

def decision_units(cands: Sequence[Candidate]) -> List[List[Candidate]]:
    """Candidates decided together: those of one document group linked by a shared readable photograph (sha256).

    Candidates of one group that share no photograph stay apart, and so do candidates of different groups.

    :param cands: Candidates (the gate sources are left to the caller).
    :type cands: Sequence[Candidate]
    :return: Units in the order of their first candidate, candidates in input order; a candidate sharing no
        photograph is a unit of its own.
    :rtype: List[List[Candidate]]
    """
    parent = {c.row_id: c.row_id for c in cands}

    def find(row_id: str) -> str:
        """Union-find root.

        :param row_id: Row id.
        :type row_id: str
        :return: Root row id.
        :rtype: str
        """
        while parent[row_id] != row_id:
            parent[row_id] = parent[parent[row_id]]
            row_id = parent[row_id]
        return row_id

    first: Dict[Tuple[str, str], str] = {}
    for cand in cands:
        for image in distinct_images(cand.images):
            slot = (cand.document_group, image.sha256)
            if slot in first:
                parent[find(cand.row_id)] = find(first[slot])
            else:
                first[slot] = cand.row_id
    units: Dict[str, List[Candidate]] = {}
    for cand in cands:
        units.setdefault(find(cand.row_id), []).append(cand)
    return list(units.values())


def _to_record(side: Side, decision: Mapping[str, Any], images: Sequence[ImageRef], pairs: Mapping[str, Dict[str, Any]],
               holder: Mapping[str, str], reads: Mapping[str, Sequence[ReadView]],
               sides: Mapping[str, Side]) -> SideDecision:
    """The output record of one side.

    :param side: Side.
    :type side: Side
    :param decision: Internal decision.
    :type decision: Mapping[str, Any]
    :param images: Photographs the side was scored on.
    :type images: Sequence[ImageRef]
    :param pairs: ``{sha256: score_pair result}`` of the side.
    :type pairs: Mapping[str, Dict[str, Any]]
    :param holder: ``{sha256: side key}`` of the unit's assignment.
    :type holder: Mapping[str, str]
    :param reads: Reads per photograph.
    :type reads: Mapping[str, Sequence[ReadView]]
    :param sides: Sides by key (to name holders by row id and side key).
    :type sides: Mapping[str, Side]
    :return: The record.
    :rtype: SideDecision
    """
    roles = decision.get("roles") or {}
    shown = decision.get("image") or decision.get("assigned")
    order = {"chosen": 0, "assigned": 1, "alternative": 2, "spill": 3, "other": 4}

    def label(key: Optional[str]) -> Optional[str]:
        """``<row_id>#<side_key>`` of a side key.

        :param key: Side key or None.
        :type key: Optional[str]
        :return: The label or None.
        :rtype: Optional[str]
        """
        return f"{sides[key].cand.row_id}#{sides[key].target.side_key}" if key in sides else None

    entries = []
    for image in images:
        p = pairs.get(image.sha256) or {"usable": False, "why": NO_READ}
        entries.append(_pair_entry(image, p, roles.get(image.sha256, "other"), prior_agrees(side, image),
                                   label(holder.get(image.sha256))))
    entries.sort(key=lambda e: (order.get(e["role"], 9), -(e.get("recall") or 0.0)))
    competitors: List[Dict[str, Any]] = lsr.round_floats(entries)
    p = pairs.get(shown.sha256) if shown is not None else None
    if competitors and shown is not None and p is not None and p.get("usable"):
        read, _ = choose_read(reads.get(shown.sha256, []), side.script)
        if read is not None:
            competitors[0]["coverage"] = coverage_of(side, read, kraken_view(reads.get(shown.sha256, [])))
    competitors.extend(lsr.round_floats(list(decision.get("info") or [])))
    usable = bool(p and p.get("usable"))
    null = p.get(f"null_n{side.primary}") if usable else None
    return SideDecision(row_id=side.cand.row_id, side_key=side.target.side_key, outcome=decision["outcome"],
                        reasons=list(decision["reasons"]),
                        image_sha256=decision["image"].sha256 if decision["outcome"] == "accepted" else None,
                        basis=decision.get("basis") if decision["outcome"] == "accepted" else None,
                        reader=p.get("reader") if usable else None, recall=p.get("recall") if usable else None,
                        null_p95=null["p95"] if null else None, null_p99=null["p99"] if null else None,
                        margin=decision.get("margin"), competitors=competitors)


def assign_sides(cands: Sequence[Candidate], reads: Iterable[ReadRecord], *, n_boot: int = lsr.N_BOOT,
                 seed: int = lsr.SEED, decide: Optional[Set[str]] = None) -> List[SideDecision]:
    """Decide every side of the candidates to decide.

    Every candidate names the owners of its photographs (so the null leaves them out for the right sides); only the
    candidates in ``decide`` get decisions, and duplicates, shared texts and decision units are looked for among them.
    A side is scored on every photograph of its decision unit (:func:`decision_units`; the given photograph for a
    gate source), and the unit's photographs are left out of its null.

    :param cands: Candidates with their targets and photographs.
    :type cands: Sequence[Candidate]
    :param reads: Every read.
    :type reads: Iterable[ReadRecord]
    :param n_boot: Bootstrap resamples of the null thresholds.
    :type n_boot: int
    :param seed: Random seed (bootstrap).
    :type seed: int
    :param decide: Row ids to decide; None for all.
    :type decide: Optional[Set[str]]
    :return: One decision per (candidate, target side), in candidate and target order.
    :rtype: List[SideDecision]
    """
    reads_by_sha = views_by_sha(reads)
    owners: Dict[str, Set[str]] = defaultdict(set)
    for cand in cands:
        for image in cand.images:
            if image.sha256:
                owners[image.sha256].add(cand.document_group)
    chosen = [c for c in cands if decide is None or c.row_id in decide]
    sides = [make_side(cand, i, target) for cand in chosen for i, target in enumerate(cand.targets)]
    by_key = {s.key: s for s in sides}
    duplicates = duplicate_targets(sides)
    shared = shared_text_groups(sides, set(duplicates))
    partners: Dict[str, Set[str]] = defaultdict(set)
    for members in shared:
        for side in members:
            partners[side.key] |= {s.group for s in members if s.group != side.group}
    cache = NullCache(reads_by_sha, owners, np.random.default_rng(seed), n_boot)
    units = decision_units([c for c in chosen if c.source not in GATE_SOURCES])
    unit_of = {c.row_id: i for i, unit in enumerate(units) for c in unit}
    unit_images = [distinct_images([im for c in unit for im in c.images]) for unit in units]
    unit_shas = [{im.sha256 for c in unit for im in c.images if im.sha256} for unit in units]
    scored: Dict[str, List[ImageRef]] = {}
    pairs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for side in sides:
        if side.key in duplicates:
            continue
        if side.cand.source in GATE_SOURCES:
            given = gate_image(side.cand, distinct_images(side.cand.images))
            images = [given] if given is not None else []
            own_shas = {im.sha256 for im in side.cand.images if im.sha256}
        else:
            images = unit_images[unit_of[side.cand.row_id]]
            own_shas = unit_shas[unit_of[side.cand.row_id]]
        scored[side.key] = images
        excluded = {side.group} | partners.get(side.key, set()) | {g for sha in own_shas for g in owners.get(sha, ())}
        pairs[side.key] = {}
        for image in images:
            views = reads_by_sha.get(image.sha256, [])
            usable, fallback = choose_read(views, side.script or HEBREW)
            null = (cache.null(side, usable.family, excluded, own_shas)
                    if usable is not None and side.has_grams else None)
            scored_pair = score_pair(side, usable or fallback, null)
            if not scored_pair.get("usable"):
                scored_pair["problems"] = sorted({v.problem(side.script or HEBREW) for v in views} - {""})
            pairs[side.key][image.sha256] = scored_pair
    decisions: Dict[str, Dict[str, Any]] = {}
    for key, (kept, share) in duplicates.items():
        decisions[key] = _decision("dropped", [DUPLICATE_SHORTER],
                                   info=[{"role": "kept_version", "row_id": by_key[kept].cand.row_id,
                                          "side_key": by_key[kept].target.side_key, "share": share}])
    roles: Dict[str, Dict[str, Any]] = {}
    for members in shared:
        roles.update(ownership(members, pairs))
    for key, own in roles.items():
        if own["role"] == "not_owner":
            owner = by_key[own["owner"]]
            decisions[key] = _decision("dropped", [NOT_OWNER],
                                       info=[{"role": "shared_text", "owner_row_id": owner.cand.row_id,
                                              "owner_side_key": owner.target.side_key,
                                              "group_docs": own["group_docs"]}])
    holders: Dict[str, Dict[str, str]] = {}
    by_cand: Dict[str, List[Side]] = defaultdict(list)
    for side in sides:
        if side.key not in decisions:
            by_cand[side.cand.row_id].append(side)
    for cand in chosen:
        if cand.source not in GATE_SOURCES or not by_cand.get(cand.row_id):
            continue
        if not distinct_images(cand.images):
            reason = NO_IMAGE if not cand.images else NO_USABLE_IMAGE
            for side in by_cand[cand.row_id]:
                decisions[side.key] = _decision("unassigned", [reason])
            continue
        for side in by_cand[cand.row_id]:
            given = scored[side.key][0]
            decisions[side.key] = decide_gate(side, given, pairs[side.key][given.sha256])
        holders[cand.row_id] = {}
    for unit, images in zip(units, unit_images):
        unit_sides = [s for c in unit for s in by_cand.get(c.row_id, [])]
        if not unit_sides:
            continue
        if not images:                      # a unit without a readable photograph is a single candidate
            for cand in unit:
                reason = NO_IMAGE if not cand.images else NO_USABLE_IMAGE
                for side in by_cand.get(cand.row_id, []):
                    decisions[side.key] = _decision("unassigned", [reason])
            continue
        blocked = {s.key: [lsr.OWNERSHIP_REVIEW_REASONS[roles[s.key]["role"]]] for s in unit_sides
                   if roles.get(s.key, {}).get("role") in lsr.OWNERSHIP_REVIEW_REASONS}
        result = decide_candidate(unit_sides, images, pairs, blocked)
        for key, decision in result.items():
            if key in blocked:
                own = roles[key]
                decision["info"].append({"role": "shared_text", "ownership": own["role"],
                                         "group_docs": own["group_docs"]})
        decisions.update(result)
        unit_holder = {d["assigned"].sha256: k for k, d in result.items() if d.get("assigned") is not None}
        for cand in unit:
            holders[cand.row_id] = unit_holder
    image_conflicts(decisions, by_key)
    out = []
    for side in sides:
        decision = decisions[side.key]
        out.append(_to_record(side, decision, scored.get(side.key, []), pairs.get(side.key, {}),
                              holders.get(side.cand.row_id, {}), reads_by_sha, by_key))
    return out


# ===================================================================================================================
# CLI
# ===================================================================================================================

def select_rows(cands: Sequence[Candidate], sources: Optional[Sequence[str]], limit: Optional[int]) -> Set[str]:
    """Row ids to decide.

    :param cands: Candidates.
    :type cands: Sequence[Candidate]
    :param sources: Sources to keep, or None for all.
    :type sources: Optional[Sequence[str]]
    :param limit: At most this many candidates (in file order), or None.
    :type limit: Optional[int]
    :return: Row ids.
    :rtype: Set[str]
    """
    rows = [c.row_id for c in cands if not sources or c.source in sources]
    return set(rows[:limit] if limit is not None else rows)


def summarize(decisions: Sequence[SideDecision], cands: Sequence[Candidate]) -> str:
    """Outcome counts per source and the most frequent reasons.

    :param decisions: Decisions.
    :type decisions: Sequence[SideDecision]
    :param cands: Candidates (for the source of each row id).
    :type cands: Sequence[Candidate]
    :return: Printable text.
    :rtype: str
    """
    source_of = {c.row_id: c.source for c in cands}
    per: Dict[str, Counter] = defaultdict(Counter)
    reasons: Counter = Counter()
    for d in decisions:
        per[source_of.get(d.row_id, "?")][d.outcome] += 1
        reasons.update(d.reasons)
    lines = [f"{len(decisions):,} side decisions"]
    for source in sorted(per):
        lines.append(f"  {source:12s} " + "  ".join(f"{o} {per[source][o]:,}" for o in ("accepted", "needs_review",
                                                                                        "unassigned", "dropped")))
    lines.append("  reasons: " + ", ".join(f"{r} {n:,}" for r, n in reasons.most_common(15)))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """The command line.

    :return: The parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description="Stage 2 (sides): place target sides on photographs by their reads.")
    parser.add_argument("--work-dir", type=Path, default=paths.DEFAULT_WORK_DIR, help="build work dir")
    parser.add_argument("--candidates", type=Path, default=None,
                        help="candidates with photographs (default: the images stage file of the work dir)")
    parser.add_argument("--reads", type=Path, default=None, help="reads file (default: <work dir>/reads/reads.jsonl)")
    parser.add_argument("--out", type=Path, default=None, help="output (default: the sides stage file)")
    parser.add_argument("--n-boot", type=int, default=lsr.N_BOOT, help="bootstrap resamples of the null thresholds")
    parser.add_argument("--seed", type=int, default=lsr.SEED, help="random seed of the bootstrap")
    parser.add_argument("--sources", default=None, help="comma-separated sources to decide (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="decide at most this many candidates")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Read the candidates and reads, decide every side, write the sides stage file.

    The stage is recomputed from its inputs on every run (the null depends on every read), so there is nothing to
    resume. A missing reads file is reported and every side goes ``unassigned`` (``no_read``).

    :param argv: Command-line arguments (None: ``sys.argv``).
    :type argv: Optional[List[str]]
    """
    args = build_parser().parse_args(argv)
    cand_path = args.candidates or paths.stage_path(args.work_dir, "images")
    reads_path = args.reads or args.work_dir / READS_FILE
    out = args.out or paths.stage_path(args.work_dir, "sides")
    cands = list(read_jsonl(cand_path, Candidate))
    if reads_path.exists():
        reads = list(read_jsonl(reads_path, ReadRecord))
    else:
        print(f"no reads file at {reads_path}: every side is unassigned (no_read)")
        reads = []
    sources = [s.strip() for s in args.sources.split(",") if s.strip()] if args.sources else None
    decisions = assign_sides(cands, reads, n_boot=args.n_boot, seed=args.seed,
                             decide=select_rows(cands, sources, args.limit))
    write_jsonl(out, decisions)
    print(summarize(decisions, cands))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
