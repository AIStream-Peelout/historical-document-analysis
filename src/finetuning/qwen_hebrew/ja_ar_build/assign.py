"""Gate v2 decisions: ordered chains of aligned lines, line-to-photograph assignment, coverage both ways, pairs.

For every target side and every read of every photograph of its document group (:mod:`align` gives the line scores
and the null), the anchored lines (z >= ``z_line`` and score >= ``s_min``) form a chain: the heaviest subsequence in
target order whose read positions do not go backwards. The chain is significant when its weight clears the chains
the same target forms against its null reads (run-level null). A photograph supports the lines of its significant
chains (plus strongly anchored lines read out of order, e.g. margins). Photographs that show the same face (pHash or
shared support) form one face. Each target line goes to the face that supports it; unread lines inside a run stay
with it; unread lines at a run's edge stay only when the photograph has unexplained read letters to hold them
("room"), else they are trimmed. Coverage the other way: read letters of the photograph left unexplained by the
assigned lines; a separate block of them is cropped away, an interleaved one sends the pair to review.

See ``logs/next_round/ja_ar_v1/build/GATE_V2_SPEC.md`` §3-§4. Thresholds live in :class:`GateParams` and are tuned on
the acceptance fixtures and planted negatives only.
"""
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from rapidfuzz import fuzz

from src.finetuning.qwen_hebrew.ja_ar_build import align, textnorm
from src.finetuning.qwen_hebrew.ja_ar_build.align import NullPool, ReadStream

#: A target line longer than this (letters) is cut into several evidence units.
UNIT_MAX = 70
#: Target size of a unit cut from a long line (letters).
UNIT_TARGET = 50


@dataclass
class GateParams:
    """Thresholds of gate v2 (defaults = the spec's starting values).

    :ivar z_line: Line anchor z.
    :ivar s_min: Line anchor minimum score (0-100).
    :ivar z_strong: z of a strong anchor kept although out of chain order.
    :ivar s_strong: Score of a strong out-of-order anchor.
    :ivar z_cover: z at which a line marks the read lines it touches as covered.
    :ivar k_cover_min: k-gram chain matches a read line needs to count as covered ...
    :ivar k_cover_share: ... covering at least this share of the read line's letters.
    :ivar z_weak: z of weak evidence used to place unread boundary lines.
    :ivar w_cap: Cap of one line's weight in a chain.
    :ivar pos_tol: Allowed backward step of a chain's read position (letters).
    :ivar min_chain: Units a significant chain needs (or all units with evidence when fewer).
    :ivar q_min: Minimum chain weight of a significant chain.
    :ivar line_support_share: Share of a line's letters in supported units that makes the line supported.
    :ivar weak_face_lines: Lines a side's support on a weaker face needs to stand on its own ...
    :ivar edge_keep_max_share: Unread edge letters kept at most as this share of the run's letters.
    :ivar edge_small_share: An edge block this small (share of the run's letters) is kept ...
    :ivar edge_small_room: ... when the ink read has at least this many unexplained letters per edge letter.
    :ivar margin_line_ratio: Edge lines this short (mean letters / median run line letters) form a margin
        block: editions print margins last as runs of short lines, and Kraken does not segment sideways
        writing at 0 degrees, so such a block is kept without room.
    :ivar no_ink_share: Room below this share of the edge letters means the photograph has no ink for them.
    :ivar weak_face_share: ... unless it holds this share of the strongest face's supported letters.
    :ivar order_ooo_max: Share of a pair's anchored letters allowed out of read order (margins) ...
    :ivar order_runs_max: ... in at most this many runs of consecutive target lines.
    :ivar order_ooo_free: Out-of-order share at or below which any number of runs is allowed.
    :ivar dup_text_ratio: Text similarity (0-100) at which two texts on one photograph are editions of the same text.
    :ivar cover_match_min: Matched letters a read line needs to count as covered by an anchored unit ...
    :ivar cover_match_share: ... and this share of its letters.
    :ivar k_min: Minimum length of a significant ordered k-gram chain.
    :ivar k_margin: k-gram chain length above the longest null k-gram chain.
    :ivar k_sd_mult: k-gram chain length above the null mean in null standard deviations.
    :ivar k_unit_min: Chain k-grams a unit needs to count as supported by the k-gram channel ...
    :ivar k_unit_share: ... or this share of its k-grams, whichever is larger.
    :ivar q_margin: Chain weight above the best null chain.
    :ivar q_sd_mult: Chain weight above the null mean in null standard deviations.
    :ivar q_abs_no_null: Chain weight needed when no null read exists.
    :ivar phash_same_face: pHash distance at or below which two photographs show the same face.
    :ivar overlap_same_face: Share of shared supported lines that makes two photographs one face.
    :ivar room_factor: Unexplained read letters needed per unexplained target letter to keep edge lines.
    :ivar exceed_ratio: Edge letters above this multiple of the run's letters need ``exceed_room`` to be kept.
    :ivar exceed_room: Room factor demanded when the edge dominates the target.
    :ivar slack: Read letters allowed per unexplained target letter before the rest counts as uncovered ink.
    :ivar substantial_letters: Letters of a read line that counts as ink in the coverage check.
    :ivar substantial_conf: Mean confidence of such a line (Kraken).
    :ivar uncovered_max: Share of unexplained substantial read letters a covered pair may have.
    :ivar uncovered_max_vlm: The same share measured on the photograph's VLM read (whole-page reader).
    :ivar support_kraken: Supported share needed when a Kraken read supports the face (Hebrew script).
    :ivar support_vlm: Supported share needed otherwise.
    :ivar interior_max: Share of interior unread letters allowed.
    :ivar review_band: Distance below a threshold that sends a pair to review instead of reject-by-test.
    :ivar min_pair_letters: Letters a pair needs.
    """

    z_line: float = 2.5
    s_min: float = 50.0
    z_strong: float = 5.0
    s_strong: float = 65.0
    z_cover: float = 2.5
    k_cover_min: int = 2
    k_cover_share: float = 0.15
    z_weak: float = 1.5
    w_cap: float = 8.0
    pos_tol: float = 2.0
    min_chain: int = 2
    q_min: float = 10.0
    q_margin: float = 2.0
    q_sd_mult: float = 4.0
    q_abs_no_null: float = 12.0
    line_support_share: float = 0.5
    weak_face_lines: int = 3
    edge_keep_max_share: float = 0.35
    edge_small_share: float = 0.10
    edge_small_room: float = 0.5
    margin_line_ratio: float = 0.5
    no_ink_share: float = 0.2
    weak_face_share: float = 0.25
    order_ooo_max: float = 0.4
    order_ooo_free: float = 0.2
    order_runs_max: int = 2
    dup_text_ratio: float = 60.0
    cover_match_min: int = 2
    cover_match_share: float = 0.3
    k_min: float = 6.0
    k_margin: float = 3.0
    k_sd_mult: float = 4.0
    k_unit_min: int = 2
    k_unit_share: float = 0.08
    phash_same_face: int = 2
    overlap_same_face: float = 0.6
    room_factor: float = 0.4
    exceed_ratio: float = 1.0
    exceed_room: float = 0.8
    slack: float = 1.25
    substantial_letters: int = 8
    substantial_conf: float = 0.5
    uncovered_max: float = 0.15
    uncovered_max_vlm: float = 0.25
    support_kraken: float = 0.55
    support_vlm: float = 0.40
    interior_max: float = 0.35
    review_band: float = 0.10
    min_pair_letters: int = 20


@dataclass
class Photo:
    """A photograph of a document group.

    :ivar sha256: Digest of the stored file.
    :ivar path: File path.
    :ivar label: Library label (``1r``, ``verso`` ...) or None.
    :ivar phash: 64-bit perceptual hash (hex) or None.
    :ivar width: Pixel width or None.
    :ivar height: Pixel height or None.
    """

    sha256: str
    path: str
    label: Optional[str] = None
    phash: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None

    @property
    def area(self) -> int:
        """Pixel area (0 when unknown).

        :return: width x height.
        """
        return int(self.width or 0) * int(self.height or 0)


@dataclass
class Side:
    """One target side of one candidate.

    :ivar row_id: Candidate id.
    :ivar side_key: Side key of the target.
    :ivar source: Candidate source.
    :ivar script: Ledger script of the candidate.
    :ivar lines: Target lines (cleaned by the candidate builder).
    :ivar skels: Letter skeletons of the lines.
    :ivar old_sha: Photograph the novel_v1 stage-2 decision chose for this side, if any.
    """

    row_id: str
    side_key: str
    source: str
    script: str
    lines: List[str]
    skels: List[str]
    old_sha: Optional[str] = None
    unit_line: List[int] = field(default_factory=list)
    unit_skels: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Cut lines into evidence units: a line up to :data:`UNIT_MAX` letters is one unit; a longer line (a flattened
        text, a long verse) is cut into units of about :data:`UNIT_TARGET` letters, so a partial match still counts.
        """
        if self.unit_skels:
            return
        for i, sk in enumerate(self.skels):
            if len(sk) <= UNIT_MAX:
                self.unit_line.append(i)
                self.unit_skels.append(sk)
                continue
            n = -(-len(sk) // UNIT_TARGET)
            size = -(-len(sk) // n)
            for k in range(n):
                self.unit_line.append(i)
                self.unit_skels.append(sk[k * size:(k + 1) * size])

    @property
    def sid(self) -> str:
        """Side id ``row_id|side_key``.

        :return: The id.
        """
        return f"{self.row_id}|{self.side_key}"

    def letters(self, idx: Sequence[int]) -> int:
        """Letters of a set of lines.

        :param idx: Line indices.
        :return: Summed skeleton letters.
        """
        return sum(len(self.skels[i]) for i in idx)

    def lines_of_units(self, units: Set[int], share: float) -> Set[int]:
        """Lines whose letters lie at least ``share`` in the given units.

        :param units: Unit indices.
        :param share: Required share of the line's letters.
        :return: Line indices.
        """
        got: Dict[int, int] = {}
        for u in units:
            got[self.unit_line[u]] = got.get(self.unit_line[u], 0) + len(self.unit_skels[u])
        return {i for i, n in got.items() if len(self.skels[i]) and n >= share * len(self.skels[i])}


@dataclass
class Group:
    """Everything gate v2 needs for one document group.

    :ivar document_group: Group id.
    :ivar sides: Target sides of all candidates of the group.
    :ivar photos: Photographs by sha256.
    :ivar reads: Usable reads per photograph sha256.
    :ivar null_exclude: Further photographs never drawn as null evidence for this group (a planted negative's true
        photographs).
    """

    document_group: str
    sides: List[Side]
    photos: Dict[str, Photo]
    reads: Dict[str, List[ReadStream]]
    null_exclude: Set[str] = field(default_factory=set)


@dataclass
class ReadEval:
    """One target side against one read.

    :ivar read: The read.
    :ivar scores: ``(n, 3)`` line scores and spans.
    :ivar z: ``(n,)`` z per line.
    :ivar chain: Line indices of the best chain.
    :ivar q: Chain weight.
    :ivar q_null: Chain weights against the null reads.
    :ivar significant: Whether the chain clears the run-level null.
    :ivar out_of_order: Strong anchors outside the chain.
    """

    read: ReadStream
    scores: np.ndarray
    z: np.ndarray
    chain: List[int]
    q: float
    q_null: np.ndarray
    significant: bool
    out_of_order: List[int] = field(default_factory=list)
    kunits: Optional[np.ndarray] = None
    kchain: List[Tuple[int, int]] = field(default_factory=list)
    k_null: Optional[np.ndarray] = None
    ksig: bool = False

    @property
    def klen(self) -> int:
        """Length of the ordered k-gram chain.

        :return: Chain k-grams.
        """
        return len(self.kchain)

    @property
    def any_sig(self) -> bool:
        """Whether either evidence channel (line chain or k-gram chain) is significant.

        :return: True when one of them clears its null.
        """
        return self.significant or self.ksig

    def k_supported(self, side: "Side", params: "GateParams") -> Set[int]:
        """Units the k-gram chain supports (only when the k-gram chain is significant).

        :param side: The side (unit skeletons).
        :param params: Thresholds.
        :return: Unit indices.
        """
        if not self.ksig or self.kunits is None:
            return set()
        k = align.KGRAM_K.get(self.read.script, 4)
        out: Set[int] = set()
        for u, n in enumerate(self.kunits):
            grams = max(0, len(side.unit_skels[u]) - k + 1)
            if grams and n >= max(params.k_unit_min, params.k_unit_share * grams):
                out.add(u)
            elif 0 < grams < params.k_unit_min * 2 and n >= 1:
                out.add(u)
        return out

    def anchored(self, params: GateParams, z_min: Optional[float] = None) -> List[int]:
        """Lines at or above an anchor threshold on this read.

        :param params: Gate thresholds.
        :param z_min: z threshold (default ``params.z_line``).
        :return: Line indices.
        """
        zt = params.z_line if z_min is None else z_min
        with np.errstate(invalid="ignore"):
            mask = (self.z >= zt) & (self.scores[:, 0] >= params.s_min)
        return [int(i) for i in np.nonzero(np.nan_to_num(mask, nan=False))[0]]


# --- chains ------------------------------------------------------------------------------------------------------------

def best_chain(anchors: Sequence[int], pos: np.ndarray, weight: np.ndarray, pos_tol: float) -> Tuple[List[int], float]:
    """Heaviest subsequence of anchored lines whose read positions do not go backwards (beyond ``pos_tol``).

    :param anchors: Anchored line indices in increasing order.
    :param pos: Read position per line (NaN allowed for non-anchors).
    :param weight: Weight per line.
    :param pos_tol: Allowed backward step in letters.
    :return: ``(chain line indices, chain weight)``; empty chain and 0.0 when there are no anchors.
    """
    m = len(anchors)
    if m == 0:
        return [], 0.0
    best = [float(weight[i]) for i in anchors]
    prev = [-1] * m
    for a in range(m):
        ia = anchors[a]
        for b in range(a):
            ib = anchors[b]
            if pos[ib] <= pos[ia] + pos_tol and best[b] + float(weight[ia]) > best[a]:
                best[a] = best[b] + float(weight[ia])
                prev[a] = b
    end = int(np.argmax(best))
    chain: List[int] = []
    k = end
    while k >= 0:
        chain.append(anchors[k])
        k = prev[k]
    chain.reverse()
    return chain, float(best[end])


def chain_of(scores: np.ndarray, z: np.ndarray, params: GateParams) -> Tuple[List[int], float, List[int]]:
    """Chain, weight and strong out-of-order anchors of one (target, read) evaluation.

    :param scores: ``(n, 3)`` scores and spans.
    :param z: ``(n,)`` z values.
    :param params: Thresholds.
    :return: ``(chain, weight, out_of_order)``.
    """
    with np.errstate(invalid="ignore"):
        mask = np.nan_to_num((z >= params.z_line) & (scores[:, 0] >= params.s_min), nan=False)
    anchors = [int(i) for i in np.nonzero(mask)[0]]
    pos = (scores[:, 1] + scores[:, 2]) / 2.0
    weight = np.minimum(np.nan_to_num(z, nan=0.0), params.w_cap)
    chain, q = best_chain(anchors, pos, weight, params.pos_tol)
    in_chain = set(chain)
    strong = [i for i in anchors if i not in in_chain and z[i] >= params.z_strong and scores[i, 0] >= params.s_strong]
    return chain, q, strong


def is_significant(q: float, q_null: np.ndarray, chain_len: int, n_evidence: int, params: GateParams) -> bool:
    """Run-level test of a chain against the chains of the same target on null reads.

    :param q: Chain weight.
    :param q_null: Null chain weights.
    :param chain_len: Lines in the chain.
    :param n_evidence: Lines of the target with own evidence.
    :param params: Thresholds.
    :return: True when the chain is significant.
    """
    if q < params.q_min or chain_len < min(params.min_chain, max(1, n_evidence)):
        return False
    if q_null.size == 0:
        return q >= params.q_abs_no_null
    mx = float(np.max(q_null))
    mu = float(np.mean(q_null))
    sd = float(np.std(q_null))
    return q >= mx + params.q_margin and q >= mu + params.q_sd_mult * max(sd, 1.0)


def is_k_significant(klen: int, k_null: Optional[np.ndarray], params: GateParams) -> bool:
    """Run-level test of an ordered k-gram chain against the chains of the same target on null reads.

    :param klen: Chain length (k-grams).
    :param k_null: Null chain lengths.
    :param params: Thresholds.
    :return: True when the chain is significant.
    """
    if klen < params.k_min:
        return False
    if k_null is None or k_null.size == 0:
        return klen >= 3 * params.k_min
    mx = float(np.max(k_null))
    mu = float(np.mean(k_null))
    sd = float(np.std(k_null))
    return klen >= mx + params.k_margin and klen >= mu + params.k_sd_mult * max(sd, 1.0)


def null_chain_weights(null: np.ndarray, params: GateParams) -> np.ndarray:
    """Chain weight of the target against every null read (leave-one-out z).

    :param null: ``(K, n, 3)`` null scores and spans.
    :param params: Thresholds.
    :return: ``(K,)`` chain weights.
    """
    k = null.shape[0]
    if k == 0:
        return np.zeros(0)
    zl = align.loo_z(null[:, :, 0])
    return np.array([chain_of(null[j], zl[j], params)[1] for j in range(k)])


# --- evaluation of one group -------------------------------------------------------------------------------------------

def evaluate_side(side: Side, group: Group, pool: NullPool, params: GateParams,
                  null_cache: Dict[Tuple[str, Tuple[str, str, int]], Tuple[np.ndarray, np.ndarray, np.ndarray]]
                  ) -> Dict[str, List[ReadEval]]:
    """Evaluate one side against every read of its group, on both evidence channels.

    :param side: The target side.
    :param group: Its group.
    :param pool: Null reads.
    :param params: Thresholds.
    :param null_cache: ``(sid, null key) -> (null scores (K, n, 3), null chain weights (K,), null k-gram chain
        lengths (K,))``, filled in place.
    :return: Read evaluations per photograph sha256.
    """
    n_ev = sum(1 for s in side.unit_skels if len(s) >= align.MIN_LINE_LETTERS)
    out: Dict[str, List[ReadEval]] = {}
    exclude = set(group.photos) | set(group.null_exclude)
    for sha, reads in group.reads.items():
        for read in reads:
            key = align.null_key_of(read)
            ck = (side.sid, key.as_tuple())
            if ck not in null_cache:
                nulls = pool.draw(key, exclude, f"{side.sid}|{key.as_tuple()}")
                ns = align.null_scores(side.unit_skels, nulls)
                null_cache[ck] = (ns, null_chain_weights(ns, params), align.kgram_null(side.unit_skels, nulls, pool.stops))
            ns, qn, kn = null_cache[ck]
            sc = align.line_scores(side.unit_skels, read)
            z = align.z_scores(sc[:, 0], ns[:, :, 0]) if ns.shape[0] else np.full(len(side.unit_skels), np.nan)
            chain, q, strong = chain_of(sc, z, params)
            sig = is_significant(q, qn, len(chain), n_ev, params)
            kunits, kchain = align.kgram_chain(side.unit_skels, read, pool.stops.get(read.script))
            ksig = is_k_significant(len(kchain), kn, params)
            if ksig and pool.targets:
                # read-side null: chains this read forms with other documents' targets of similar length
                letters = sum(len(u) for u in side.unit_skels)
                rk = ("read_null", read.key, align.length_bin(letters), side.script)
                if rk not in null_cache:
                    others = pool.draw_targets(_script_family(side.script), letters, group.document_group,
                                               f"{read.key}|{rk[2]}")
                    stop = pool.stops.get(read.script)
                    rn = np.array([len(align.kgram_chain(t, read, stop)[1]) for t in others], dtype=np.float64)
                    null_cache[rk] = (rn, rn, rn)
                rn = null_cache[rk][0]
                ksig = is_k_significant(len(kchain), rn, params) if rn.size else ksig
            out.setdefault(sha, []).append(ReadEval(read=read, scores=sc, z=z, chain=chain, q=q, q_null=qn,
                                                    significant=sig, out_of_order=strong, kunits=kunits,
                                                    kchain=kchain, k_null=kn, ksig=ksig))
    return out


def _script_family(script: str) -> str:
    """Script family of a ledger script (``mixed`` counts as Hebrew script).

    :param script: ``hebrew`` / ``arabic`` / ``mixed``.
    :return: ``hebrew`` or ``arabic``.
    """
    return "arabic" if script == "arabic" else "hebrew"


def phash_distance(a: Optional[str], b: Optional[str]) -> Optional[int]:
    """Hamming distance of two hex pHashes.

    :param a: First hash.
    :param b: Second hash.
    :return: Distance or None when either is missing.
    """
    if not a or not b:
        return None
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def side_label(text: Optional[str]) -> Optional[str]:
    """Normalised face of a label or side key: ``recto`` / ``verso`` / None.

    :param text: Photograph label (``1r``, ``verso``, ``b`` ...) or side key.
    :return: The face or None when the label does not say.
    """
    if not text:
        return None
    t = text.strip().lower()
    if t in ("recto", "r", "a", "1r", "fol. 1r", "ע\"א") or t.endswith("r") and t[:-1].strip().isdigit():
        return "recto"
    if t in ("verso", "v", "b", "1v", "fol. 1v", "ע\"ב") or t.endswith("v") and t[:-1].strip().isdigit():
        return "verso"
    if "recto" in t:
        return "recto"
    if "verso" in t:
        return "verso"
    return None


class _UnionFind:
    """Minimal union-find over strings."""

    def __init__(self, items: Sequence[str]) -> None:
        """Create singleton sets.

        :param items: Elements.
        """
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        """Root of an element.

        :param x: Element.
        :return: Root element.
        """
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        """Merge two sets.

        :param a: Element.
        :param b: Element.
        """
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


@dataclass
class Unit:
    """Lines of one candidate assigned to one face (one future pair).

    :ivar row_id: Candidate id.
    :ivar face: Face id (representative sha256).
    :ivar parts: ``[(side, core supported lines, interior lines, edge candidate lines, boundary lines)]``; boundary
        lines are the edge candidates that lie between this face's run and another face's run of the same side.
    """

    row_id: str
    face: str
    parts: List[Tuple[Side, List[int], List[int], List[int], List[int]]] = field(default_factory=list)


def evaluate_group(group: Group, pool: NullPool, params: Optional[GateParams] = None
                   ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run gate v2 on one document group.

    :param group: The group.
    :param pool: Null reads.
    :param params: Thresholds (defaults when None).
    :return: ``(pair records, line evidence records)`` as JSON-ready dicts (schema GATE_V2_SPEC §4).
    """
    params = params or GateParams()
    null_cache: Dict[Tuple[str, Tuple[str, str, int]], Tuple[np.ndarray, np.ndarray]] = {}
    evals: Dict[str, Dict[str, List[ReadEval]]] = {s.sid: evaluate_side(s, group, pool, params, null_cache)
                                                   for s in group.sides}
    # photo support per side: units of significant chains + strong out-of-order anchors -> supported lines
    support: Dict[str, Dict[str, Set[int]]] = {}
    for side in group.sides:
        per_photo: Dict[str, Set[int]] = {}
        for sha, evs in evals[side.sid].items():
            units: Set[int] = set()
            for ev in evs:
                if ev.significant:
                    units.update(ev.chain)
                    units.update(ev.out_of_order)
                units.update(ev.k_supported(side, params))
            lines = side.lines_of_units(units, params.line_support_share) if units else set()
            if lines:
                per_photo[sha] = lines
        support[side.sid] = per_photo
    faces = _faces(group, support, params)
    line_records: List[Dict[str, Any]] = []
    pairs: List[Dict[str, Any]] = []
    units: Dict[Tuple[str, str], Unit] = {}
    side_outcome: Dict[str, Dict[str, Any]] = {}
    assigns: Dict[str, List[Optional[str]]] = {}
    weaks: Dict[str, List[Dict[str, float]]] = {}
    for side in group.sides:
        assigns[side.sid], weaks[side.sid] = _assign_lines(side, support[side.sid], faces, evals[side.sid], params)
    _apply_ownership(group, assigns)
    for side in group.sides:
        assign, weak = assigns[side.sid], weaks[side.sid]
        line_records.extend(_line_evidence(side, evals[side.sid], faces, assign))
        if not any(a is not None for a in assign):
            side_outcome[side.sid] = {"placed": False}
            continue
        side_outcome[side.sid] = {"placed": True}
        for face, (core, interior, edge, boundary) in _runs(side, assign, weak, params).items():
            unit = units.setdefault((side.row_id, face), Unit(row_id=side.row_id, face=face))
            unit.parts.append((side, core, interior, edge, boundary))
    for (_row_id, face), unit in sorted(units.items()):
        for part_unit in (_split_duplicate_parts(unit, params) if len(unit.parts) > 1 else [unit]):
            pairs.append(_decide_unit(part_unit, group, faces, evals, support, params))
    _rank_duplicates(pairs, params)
    for side in group.sides:
        if not side_outcome.get(side.sid, {}).get("placed"):
            pairs.append(_reject_record(side, group, evals[side.sid]))
    return pairs, line_records


def _apply_ownership(group: Group, assigns: Dict[str, List[Optional[str]]]) -> None:
    """A face belongs to the side with the most supported letters on it; other sides keep lines there only when
    they own no face of their own and are strongly supported on it (at least half of their lines with evidence, as
    in a two-page opening). A short side that shares names or formulae with the owner (an address, notes on the
    back) never joins its text.

    :param group: The group.
    :param assigns: Face (or None) per line per side id; updated in place.
    """
    # Ownership is decided among the sides of ONE candidate (one edition): it removes one face's spill onto another.
    # Texts of different candidates on one photograph are editions of the same side and are ranked as rivals later
    # (no source wins by rule).
    by_face: Dict[Tuple[str, str], Dict[str, int]] = {}
    sides = {s.sid: s for s in group.sides}
    for sid, assign in assigns.items():
        side = sides[sid]
        for i, f in enumerate(assign):
            if f is not None:
                key = (f, side.row_id)
                by_face.setdefault(key, {})
                by_face[key][sid] = by_face[key].get(sid, 0) + len(side.skels[i])
    owner = {key: max(c, key=lambda sid: (c[sid], sid)) for key, c in by_face.items()}
    owns: Dict[str, Set[str]] = {}
    for (f, _row), sid in owner.items():
        owns.setdefault(sid, set()).add(f)
    for sid, assign in assigns.items():
        side = sides[sid]
        n_ev = sum(1 for sk in side.skels if len(sk) >= align.MIN_LINE_LETTERS)
        for f in {x for x in assign if x is not None}:
            if owner.get((f, side.row_id)) == sid:
                continue
            on_f = [i for i, x in enumerate(assign) if x == f]
            strong = len(on_f) >= max(2, 0.5 * n_ev)
            if owns.get(sid) or not strong:
                for i in on_f:
                    assign[i] = None


def _faces(group: Group, support: Mapping[str, Mapping[str, Set[int]]], params: GateParams) -> Dict[str, str]:
    """Cluster supporting photographs into faces.

    :param group: The group.
    :param support: Supported lines per side per photograph.
    :param params: Thresholds.
    :return: ``{sha256: face id}``; the face id is the representative photograph's sha256.
    """
    shas = sorted({sha for per in support.values() for sha in per})
    uf = _UnionFind(shas)
    for i, a in enumerate(shas):
        for b in shas[i + 1:]:
            d = phash_distance(group.photos[a].phash, group.photos[b].phash)
            if d is not None and d <= params.phash_same_face:
                uf.union(a, b)
                continue
            # two photographs of one face both carry most of a side's text: each supports at least half of the side's
            # lines (and at least 3), and their supported lines largely coincide. A short side's weak support on the
            # other face (names it shares with it) never joins the two faces.
            for side in group.sides:
                per = support.get(side.sid, {})
                la, lb = per.get(a), per.get(b)
                if not la or not lb:
                    continue
                need = max(3, 0.5 * len(side.lines))
                if len(la) >= need and len(lb) >= need and \
                        len(la & lb) >= params.overlap_same_face * len(la | lb):
                    uf.union(a, b)
                    break
    members: Dict[str, List[str]] = {}
    for sha in shas:
        members.setdefault(uf.find(sha), []).append(sha)
    face_of: Dict[str, str] = {}
    for root, shas_ in members.items():
        def strength(sha: str) -> Tuple[int, int, str]:
            total = 0
            for side in group.sides:
                total += side.letters(sorted(support[side.sid].get(sha, set())))
            return (total, group.photos[sha].area, sha)
        rep = max(shas_, key=strength)
        for sha in shas_:
            face_of[sha] = rep
    return face_of


def _assign_lines(side: Side, support: Mapping[str, Set[int]], faces: Mapping[str, str],
                  evals: Mapping[str, List[ReadEval]], params: GateParams
                  ) -> Tuple[List[Optional[str]], List[Dict[str, float]]]:
    """Face of every supported line, and weak evidence per face for the unread ones.

    :param side: The side.
    :param support: Supported lines per photograph.
    :param faces: Face of every supporting photograph.
    :param evals: Read evaluations per photograph.
    :param params: Thresholds.
    :return: ``(face or None per line, {face: best z} per line)``.
    """
    n = len(side.lines)
    raw: Dict[str, Set[int]] = {}
    for sha, lines in support.items():
        raw.setdefault(faces[sha], set()).update(lines)
    # A side's support on a face counts only as a block of its own: support scattered strictly inside the span of a
    # stronger face (the other face of the same document shares names and formulae with it) is dropped, and what is
    # left on a weak face must be either a real block (>= weak_face_lines lines) or a fair share of the strongest.
    face_lines: Dict[str, Set[int]] = {}
    for f in sorted(raw, key=lambda x: (-side.letters(sorted(raw[x])), x)):
        lines = set(raw[f])
        for stronger in face_lines.values():
            lo, hi = min(stronger), max(stronger)
            lines = {i for i in lines if not lo < i < hi}
        if not lines:
            continue
        strongest = max((side.letters(sorted(v)) for v in face_lines.values()), default=0)
        if face_lines and len(lines) < params.weak_face_lines and \
                side.letters(sorted(lines)) < params.weak_face_share * strongest:
            continue
        face_lines[f] = lines
    face_letters = {f: side.letters(sorted(ls)) for f, ls in face_lines.items()}
    assign: List[Optional[str]] = [None] * n
    for i in range(n):
        cands = [f for f, ls in face_lines.items() if i in ls]
        if len(cands) == 1:
            assign[i] = cands[0]
        elif cands:
            assign[i] = max(cands, key=lambda f: (face_letters[f], f))
    weak: List[Dict[str, float]] = [dict() for _ in range(n)]
    for sha, evs in evals.items():
        face = faces.get(sha)
        if face is None:
            continue
        for ev in evs:
            for u, i in enumerate(side.unit_line):
                zi = ev.z[u]
                if not np.isnan(zi):
                    weak[i][face] = max(weak[i].get(face, -99.0), float(zi))
    return assign, weak


def _runs(side: Side, assign: Sequence[Optional[str]], weak: Sequence[Mapping[str, float]], params: GateParams
          ) -> Dict[str, Tuple[List[int], List[int], List[int], List[int]]]:
    """Core, interior, edge-candidate and boundary lines per face for one side.

    :param side: The side.
    :param assign: Face per line (None = unread).
    :param weak: Best z per face per line.
    :param params: Thresholds.
    :return: ``{face: (core, interior, edge candidates, boundary lines)}``; boundary lines are edge candidates that sit
        between this face's run and another face's run (they may belong to either face).
    """
    n = len(assign)
    out: Dict[str, Tuple[List[int], List[int], List[int], List[int]]] = {}
    for i, f in enumerate(assign):
        if f is not None:
            out.setdefault(f, ([], [], [], []))[0].append(i)
    for i in range(n):
        if assign[i] is not None:
            continue
        p = next((j for j in range(i - 1, -1, -1) if assign[j] is not None), None)
        q = next((j for j in range(i + 1, n) if assign[j] is not None), None)
        fp = assign[p] if p is not None else None
        fq = assign[q] if q is not None else None
        if fp is not None and fp == fq:
            out[fp][1].append(i)
            continue
        options = [f for f in (fp, fq) if f is not None]
        if not options:
            continue
        if len(options) == 2:
            zp = weak[i].get(fp, -99.0)
            zq = weak[i].get(fq, -99.0)
            target = fq if zq > zp and zq >= params.z_weak else fp
            out[target][2].append(i)
            out[target][3].append(i)
        else:
            out[options[0]][2].append(i)
    return out


def _coverage_read(unit: Unit, group: Group, faces: Mapping[str, str],
                   evals: Mapping[str, Mapping[str, List[ReadEval]]]) -> Tuple[str, ReadStream]:
    """The photograph and read used for the coverage check of a unit.

    The face's photographs are tried in order of the unit's chain weight; a significant Kraken read with boxes is
    preferred (geometry for crops), then the heaviest significant read of any reader.

    :param unit: The unit.
    :param group: Its group.
    :param faces: Face of every supporting photograph.
    :param evals: Read evaluations per side per photograph.
    :return: ``(photograph sha256, read)``.
    """
    members = [sha for sha, f in faces.items() if f == unit.face]
    best: Optional[Tuple[Tuple[int, float, float], str, ReadStream]] = None
    for sha in members:
        for side, *_ in unit.parts:
            for ev in evals[side.sid].get(sha, []):
                if not ev.any_sig:
                    continue
                score = (1 if ev.read.bboxes is not None else 0, float(ev.klen), ev.q)
                if best is None or score > best[0]:
                    best = (score, sha, ev.read)
    if best is None:
        sha = unit.face
        return sha, group.reads[sha][0]
    return best[1], best[2]


def _coverage_on(unit: Unit, read: ReadStream, sha: str, evals: Mapping[str, Mapping[str, List[ReadEval]]],
                 params: GateParams) -> Tuple[Set[int], Dict[str, Set[int]], Dict[str, Dict[int, float]], Dict[str, Any]]:
    """Read lines a unit's lines cover on one read, the lines anchored there, their read positions, chain summary.

    A read line counts as covered only on real evidence: letters an anchored unit actually matches on it (InDel
    alignment inside the unit's window), or k-gram chain matches dense enough on it; stray matches in short garbage
    lines do not cover them.

    :param unit: The unit.
    :param read: The read.
    :param sha: Its photograph.
    :param evals: Read evaluations per side per photograph.
    :param params: Thresholds.
    :return: ``(covered read lines, anchored lines per side, median read position per anchored line per side,
        evidence summary of the best significant evaluation on this read)``.
    """
    covered: Set[int] = set()
    here_by: Dict[str, Set[int]] = {}
    pos_by: Dict[str, Dict[int, float]] = {}
    info: Dict[str, Any] = {}
    for side, *_ in unit.parts:
        ev = next((e for e in evals[side.sid].get(sha, []) if e.read.key == read.key), None)
        here: Set[int] = set()
        positions: Dict[int, List[float]] = {}
        if ev is not None:
            units_here = set(ev.anchored(params, params.z_cover))
            for u in units_here:
                matched = align.matched_read_lines(side.unit_skels[u], read, ev.scores[u, 1], ev.scores[u, 2])
                for j, m in matched.items():
                    if m >= params.cover_match_min and m >= params.cover_match_share * max(1, read.line_letters[j]):
                        covered.add(j)
                positions.setdefault(side.unit_line[u], []).append(float(ev.scores[u, 1] + ev.scores[u, 2]) / 2.0)
            k_here = ev.k_supported(side, params)
            if ev.ksig:
                kk = align.KGRAM_K.get(read.script, 4)
                per_line: Dict[int, int] = {}
                for u, r in ev.kchain:
                    if u in k_here and r < len(read.owner):
                        per_line[read.owner[r]] = per_line.get(read.owner[r], 0) + 1
                        positions.setdefault(side.unit_line[u], []).append(float(r))
                for j, n in per_line.items():
                    if n >= params.k_cover_min and n * kk >= params.k_cover_share * max(1, read.line_letters[j]):
                        covered.add(j)
            units_here |= k_here
            here = side.lines_of_units(units_here, params.line_support_share)
            if ev.any_sig and (not info or (ev.klen, ev.q) > (info.get("k_chain", -1), info.get("chain_q", -1))):
                info = _chain_info(ev)
        here_by[side.sid] = here
        pos_by[side.sid] = {i: float(np.median(v)) for i, v in positions.items() if i in here and v}
    return covered, here_by, pos_by, info


def _short_line_block(unit: Unit, params: GateParams) -> bool:
    """Whether every side's unread (non-boundary) edge lines are short lines, as margins are printed.

    :param unit: The unit.
    :param params: Thresholds.
    :return: True when each side with edge lines has a mean edge-line length of at most ``margin_line_ratio`` times
        the median length of its run lines (and at least two edge lines or one line under 20 letters).
    """
    seen = False
    for side, core, interior, edge, boundary in unit.parts:
        plain = [i for i in edge if i not in boundary]
        if not plain:
            continue
        seen = True
        run = sorted(core + interior)
        if not run:
            return False
        run_len = statistics.median([len(side.skels[i]) for i in run])
        mean_edge = sum(len(side.skels[i]) for i in plain) / len(plain)
        if mean_edge > params.margin_line_ratio * run_len:
            return False
        if len(plain) < 2 and mean_edge >= 20:
            return False
    return seen


def _ink_read(group: Group, sha: str, read: ReadStream) -> ReadStream:
    """The read that best shows where a photograph's ink is.

    Kraken segments written lines in any script and orientation it reads, so its read with the most text lines (boxes
    preferred, for crops) shows the ink better than a VLM read, which may skip what it cannot read.

    :param group: The group.
    :param sha: The photograph.
    :param read: The coverage read (returned when it is a Kraken read or no Kraken read exists).
    :return: The ink read.
    """
    if read.family == "kraken":
        return read
    krakens = [r for r in group.reads.get(sha, []) if r.family == "kraken" and r.n_lines]
    if not krakens:
        return read
    return max(krakens, key=lambda r: (sum(1 for n in r.line_letters if n >= 3), r.bboxes is not None, r.letters, r.key))


def _decide_unit(unit: Unit, group: Group, faces: Mapping[str, str], evals: Mapping[str, Mapping[str, List[ReadEval]]],
                 support: Mapping[str, Mapping[str, Set[int]]], params: GateParams) -> Dict[str, Any]:
    """Coverage both ways, edge decisions, crop and decision for one unit.

    Unread edge lines (no evidence on any read of the face) are kept only when they are plausibly on the photograph:
    a small block (at most ``edge_keep_max_share`` of the run's letters, e.g. a margin or an address), not at the
    boundary with another face's run, with room for them on the ink read. Boundary lines are trimmed. Trimmed edge
    lines while the photograph shows unexplained ink send the pair to review: the photograph may hold them (written
    sideways or upside down; rotated reads can settle it).

    :param unit: The unit (one candidate on one face).
    :param group: Its group.
    :param faces: Face of every supporting photograph.
    :param evals: Read evaluations per side per photograph.
    :param support: Supported lines per side per photograph.
    :param params: Thresholds.
    :return: The pair record.
    """
    cov_sha, read = _coverage_read(unit, group, faces, evals)
    covered, here, line_pos, chain_info = _coverage_on(unit, read, cov_sha, evals, params)
    if not chain_info:
        for side, *_ in unit.parts:
            for sha, evs in evals[side.sid].items():
                if faces.get(sha) != unit.face:
                    continue
                for ev in evs:
                    if ev.any_sig and (ev.klen, ev.q) > (chain_info.get("k_chain", -1), chain_info.get("chain_q", -1)):
                        chain_info = _chain_info(ev)
    order_ok, order_stats = _order_check(unit, line_pos, params)
    ink = _ink_read(group, cov_sha, read)
    if ink.key == read.key:
        ink_cov, ink_here = covered, here
    else:
        ink_cov, ink_here, _pos, _info = _coverage_on(unit, ink, cov_sha, evals, params)
    room = sum(ink.line_letters[j] for j in range(ink.n_lines) if j not in ink_cov and ink.line_letters[j] >= 3)
    demand_core = 0
    edge_letters = 0
    run_letters = 0
    for side, core, interior, edge, boundary in unit.parts:
        h = ink_here[side.sid]
        demand_core += side.letters([i for i in core + interior if i not in h])
        edge_letters += side.letters([i for i in edge if i not in boundary])
        run_letters += side.letters(core + interior)
    no_ink = room < params.no_ink_share * max(1, edge_letters)
    small = edge_letters <= params.edge_keep_max_share * run_letters
    margin_block = small and _short_line_block(unit, params)
    keep_edges = edge_letters > 0 and small and (
        margin_block
        or (not no_ink and room >= params.room_factor * (demand_core + edge_letters))
        or (edge_letters <= params.edge_small_share * run_letters and room >= params.edge_small_room * edge_letters))
    edge_review = edge_letters > 0 and not keep_edges and not no_ink
    kept_parts: List[Tuple[Side, List[int], List[int], List[int], List[int]]] = []
    boundary_trimmed: Dict[str, List[int]] = {}
    for side, core, interior, edge, boundary in unit.parts:
        plain_edge = [i for i in edge if i not in boundary]
        kept_edge = sorted(plain_edge) if keep_edges else []
        trimmed = sorted(set(edge) - set(kept_edge))
        if boundary:
            boundary_trimmed[side.sid] = sorted(boundary)
        kept_parts.append((side, core, interior, kept_edge, trimmed))
    # unexplained ink: substantial uncovered letters of the ink read beyond what unanchored target lines can hold
    unexplained_target = 0
    for side, core, interior, kept_edge, _tr in kept_parts:
        h = ink_here[side.sid]
        unexplained_target += side.letters([i for i in core + interior + kept_edge if i not in h])
    substantial = [j for j in range(ink.n_lines) if ink.line_letters[j] >= params.substantial_letters
                   and (ink.line_conf[j] is None or ink.line_conf[j] >= params.substantial_conf)]
    total_sub = sum(ink.line_letters[j] for j in substantial)
    uncovered_sub = [j for j in substantial if j not in ink_cov]
    residual = max(0.0, sum(ink.line_letters[j] for j in uncovered_sub) - params.slack * unexplained_target)
    uncovered_share = residual / total_sub if total_sub else 0.0
    crop, block, crop_reason = None, "none", None
    if uncovered_share > params.uncovered_max:
        # a crop is drawn from the boxes of the kept lines on the ink read: every kept line must have one, or the
        # crop could cut lines supported only by another (box-less) read
        unplaced = any(i not in ink_here[side.sid] for side, core, interior, kept_edge, _tr in kept_parts
                       for i in core + interior + kept_edge)
        if unplaced:
            crop, block, crop_reason = None, "unknown", "crop_cannot_place_all_lines"
        else:
            crop, block, crop_reason = _crop_box(ink, ink_cov, uncovered_sub, group.photos.get(cov_sha), params,
                                                 any(p[3] for p in kept_parts))
    # a second look at uncovered text through the VLM read of the photograph: it reads whole pages, so text the target
    # lacks shows up there even when Kraken read it with too little confidence to count as ink (a deed whose edition
    # covers only its lower part)
    vlm_share: Optional[float] = None
    for vread in group.reads.get(cov_sha, []):
        if vread.family != "vlm" or vread.rotation != 0 or vread.key == ink.key:
            continue
        v_cov, v_here, _vp, _vi = _coverage_on(unit, vread, cov_sha, evals, params)
        v_sub = [j for j in range(vread.n_lines) if vread.line_letters[j] >= params.substantial_letters]
        v_total = sum(vread.line_letters[j] for j in v_sub)
        if not v_total:
            continue
        v_unexpl = 0
        for side, core, interior, kept_edge, _tr in kept_parts:
            v_unexpl += side.letters([i for i in core + interior + kept_edge if i not in v_here[side.sid]])
        v_res = max(0.0, sum(vread.line_letters[j] for j in v_sub if j not in v_cov) - params.slack * v_unexpl)
        share = v_res / v_total
        vlm_share = share if vlm_share is None else max(vlm_share, share)
    supported_letters = 0
    target_letters = 0
    interior_letters = 0
    line_indices: List[Tuple[str, int]] = []
    trimmed_all: Dict[str, List[int]] = {}
    for side, core, interior, kept_edge, trimmed in kept_parts:
        supported_letters += side.letters(core)
        interior_letters += side.letters(interior)
        target_letters += side.letters(core + interior + kept_edge)
        for i in sorted(core + interior + kept_edge):
            line_indices.append((side.sid, i))
        trimmed_all[side.sid] = trimmed
    support_share = supported_letters / target_letters if target_letters else 0.0
    interior_share = interior_letters / target_letters if target_letters else 0.0
    kraken_face = any(ev.any_sig and ev.read.family == "kraken"
                      for side, *_ in unit.parts for sha, evs in evals[side.sid].items()
                      if faces.get(sha) == unit.face for ev in evs)
    script = unit.parts[0][0].script
    need = params.support_kraken if (kraken_face and script != "arabic") else params.support_vlm
    covered_ok = uncovered_share <= params.uncovered_max or crop is not None
    reasons: List[str] = []
    if support_share < need:
        reasons.append("support_share_low")
    if interior_share > params.interior_max:
        reasons.append("interior_unread_high")
    if not covered_ok:
        reasons.append(crop_reason or "uncovered_ink")
    if target_letters < params.min_pair_letters:
        reasons.append("too_few_letters")
    if not order_ok:
        reasons.append("order_conflict")
    if edge_review:
        reasons.append("unread_ink_may_hold_trimmed_lines")
    if vlm_share is not None and vlm_share > params.uncovered_max_vlm and crop is None:
        reasons.append("uncovered_text_vlm")
    decision = "accept" if not reasons else "review"
    remedies, flags, moved = _remedies(unit, kept_parts, faces, support, group, crop, read)
    if boundary_trimmed:
        flags = sorted(set(flags) | {"boundary_trimmed"})
    if keep_edges and margin_block:
        flags = sorted(set(flags) | {"margin_block_kept"})
    first_side = unit.parts[0][0]
    photo = group.photos.get(cov_sha) or group.photos.get(unit.face)
    dups = sorted(sha for sha, f in faces.items() if f == unit.face and sha != cov_sha)
    sids = [p[0].sid for p in kept_parts]
    lines_out: List[str] = []
    line_refs: List[List[Any]] = []
    for sid, i in _ordered_lines(line_indices, kept_parts, evals, cov_sha, read):
        side = next(p[0] for p in kept_parts if p[0].sid == sid)
        lines_out.append(side.lines[i])
        line_refs.append([side.side_key, i])
    line_out_idx: Optional[List[int]] = [r[1] for r in line_refs] if len(sids) == 1 else None
    # a photograph is stored upside down only when its best read is a 180-degree read and no unrotated read of it
    # is significant (a page with blocks in several orientations stays as stored)
    upright_sig = any(ev.any_sig and ev.read.rotation != 180 for p in unit.parts
                      for ev in evals[p[0].sid].get(cov_sha, []))
    rotation = 180 if read.rotation == 180 and not upright_sig else 0
    if rotation == 0 and "rotate180" in remedies:
        remedies.remove("rotate180")
    return {
        "pair_id": f"{first_side.row_id}|{'+'.join(p[0].side_key for p in kept_parts)}|{cov_sha[:12]}",
        "row_id": first_side.row_id, "side_key": first_side.side_key,
        "side_keys": [p[0].side_key for p in kept_parts], "source": first_side.source,
        "document_group": group.document_group, "script": script, "ja_signal": None, "tier": None,
        "image_sha256": cov_sha, "image_path": photo.path if photo else None, "image_label": photo.label if photo else None,
        "duplicate_photos": dups, "decision": decision, "remedies": remedies,
        "line_indices": line_out_idx, "line_refs": line_refs, "lines": lines_out,
        "trimmed_lines": {sid: tr for sid, tr in trimmed_all.items() if tr}, "moved_to": moved,
        "rotation": rotation, "crop": crop,
        "evidence": {**chain_info, "coverage_reader": read.reader, "coverage_read_rotation": read.rotation,
                     "ink_reader": ink.reader, "ink_read_rotation": ink.rotation, "ink_read_lines": ink.n_lines,
                     "covered_read_lines": sorted(covered), **order_stats, "supported_letters": supported_letters,
                     "target_letters": target_letters, "support_share": round(support_share, 4),
                     "interior_unread": {p[0].sid: p[2] for p in kept_parts if p[2]},
                     "edge_kept": {p[0].sid: p[3] for p in kept_parts if p[3]},
                     "boundary_trimmed": boundary_trimmed, "interior_share": round(interior_share, 4),
                     "room_letters": room, "room_demand_core": demand_core, "edge_letters": edge_letters,
                     "uncovered_share": round(uncovered_share, 4), "uncovered_block": block,
                     "uncovered_share_vlm": None if vlm_share is None else round(vlm_share, 4),
                     "n_read_lines": read.n_lines, "n_substantial": len(substantial), "support_needed": need},
        "dup_rank": None, "flags": flags, "reasons": reasons,
    }


def _order_check(unit: Unit, line_pos: Mapping[str, Mapping[int, float]], params: GateParams
                 ) -> Tuple[bool, Dict[str, Any]]:
    """Whether the lines anchored on the coverage read follow the target's line order.

    Per side, the anchored lines (in target order) keep their median read position; the heaviest chain whose read
    positions do not go backwards is the in-order part. Lines outside it are out of order: allowed when they are few
    (margins and an address read before the main text) and form at most ``order_runs_max`` runs of consecutive
    anchored lines; a target in the wrong line order (planted shuffles, a mis-assembled edition) fails.

    :param unit: The unit.
    :param line_pos: Median read position per anchored line, per side.
    :param params: Thresholds.
    :return: ``(ok, stats)``.
    """
    total = 0
    ooo_letters = 0
    runs = 0
    for side, *_ in unit.parts:
        pos = line_pos.get(side.sid, {})
        if len(pos) < 3:
            continue
        idx = sorted(pos)
        weights = np.array([float(len(side.skels[i])) for i in idx])
        chain, _q = best_chain(list(range(len(idx))), np.array([pos[i] for i in idx]), weights, params.pos_tol)
        in_chain = {idx[k] for k in chain}
        total += int(weights.sum())
        prev_out = False
        for i in idx:
            out = i not in in_chain
            if out:
                ooo_letters += len(side.skels[i])
                if not prev_out:
                    runs += 1
            prev_out = out
    share = ooo_letters / total if total else 0.0
    ok = total < 30 or share <= params.order_ooo_free or (share <= params.order_ooo_max and runs <= params.order_runs_max)
    return ok, {"order_anchored_letters": total, "order_ooo_letters": ooo_letters, "order_ooo_runs": runs}


def _texts_duplicate(a: str, b: str, params: GateParams) -> bool:
    """Whether two texts are editions (copies) of the same text.

    :param a: First text (any form; compared as letter skeletons).
    :param b: Second text.
    :param params: Thresholds.
    :return: True when their skeletons are similar, or the shorter one is largely contained in the longer one.
    """
    sa, sb = textnorm.skeleton(a), textnorm.skeleton(b)
    if not sa or not sb:
        return False
    if fuzz.ratio(sa, sb) >= params.dup_text_ratio:
        return True
    short, long_ = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    return len(short) >= 30 and fuzz.partial_ratio(short, long_) >= params.dup_text_ratio + 15


def _split_duplicate_parts(unit: Unit, params: GateParams) -> List[Unit]:
    """Keep parts that are editions of the same text out of one merged unit (they become rival pairs instead).

    :param unit: A unit with several parts (target sides of one candidate on one face).
    :param params: Thresholds.
    :return: Units whose parts are pairwise distinct texts.
    """
    def text_of(part: Tuple[Side, List[int], List[int], List[int], List[int]]) -> str:
        side, core, interior, edge, _boundary = part
        return "\n".join(side.lines[i] for i in sorted(core + interior + edge))
    ordered = sorted(unit.parts, key=lambda p: -p[0].letters(p[1] + p[2] + p[3]))
    out: List[Unit] = []
    for part in ordered:
        home = next((u for u in out if not any(_texts_duplicate(text_of(part), text_of(q), params) for q in u.parts)),
                    None)
        if home is None:
            home = Unit(row_id=unit.row_id, face=unit.face)
            out.append(home)
        home.parts.append(part)
    return out


def _chain_info(ev: ReadEval) -> Dict[str, Any]:
    """Evidence summary of one read evaluation for a pair record.

    :param ev: The evaluation.
    :return: Reader, line-chain and k-gram-chain statistics with their nulls.
    """
    def stats(a: Optional[np.ndarray]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if a is None or a.size == 0:
            return None, None, None
        return round(float(np.max(a)), 2), round(float(np.mean(a)), 2), round(float(np.std(a)), 2)
    qmx, qmu, qsd = stats(ev.q_null)
    kmx, kmu, ksd = stats(ev.k_null)
    return {"reader": ev.read.reader, "rotation_read": ev.read.rotation, "chain_lines": len(ev.chain),
            "chain_q": round(ev.q, 2), "chain_sig": ev.significant, "q_null_max": qmx, "q_null_mean": qmu,
            "q_null_sd": qsd, "k_chain": ev.klen, "k_sig": ev.ksig, "k_null_max": kmx, "k_null_mean": kmu,
            "k_null_sd": ksd}


def _ordered_lines(line_indices: Sequence[Tuple[str, int]], kept_parts: Sequence[Tuple[Side, List[int], List[int],
                   List[int], List[int]]], evals: Mapping[str, Mapping[str, List[ReadEval]]], sha: str,
                   read: ReadStream) -> List[Tuple[str, int]]:
    """Order of a unit's lines: target order within a side; sides ordered by their mean read position.

    :param line_indices: ``(sid, line index)`` pairs.
    :param kept_parts: The unit's parts.
    :param evals: Read evaluations.
    :param sha: Coverage photograph.
    :param read: Coverage read.
    :return: Ordered ``(sid, line index)`` pairs.
    """
    if len(kept_parts) == 1:
        return sorted(line_indices, key=lambda x: x[1])
    mean_pos: Dict[str, float] = {}
    for side, core, *_ in kept_parts:
        ev = next((e for e in evals[side.sid].get(sha, []) if e.read.key == read.key), None)
        core_set = set(core)
        units = [u for u, i in enumerate(side.unit_line) if i in core_set]
        if ev is not None and units:
            mid = (ev.scores[units, 1] + ev.scores[units, 2]) / 2.0
            mean_pos[side.sid] = float(np.nanmean(mid)) if np.any(~np.isnan(mid)) else float("inf")
        else:
            mean_pos[side.sid] = float("inf")
    order = {sid: k for k, sid in enumerate(sorted(mean_pos, key=lambda s: (mean_pos[s], s)))}
    return sorted(line_indices, key=lambda x: (order[x[0]], x[1]))


def _crop_box(read: ReadStream, covered: Set[int], uncovered_sub: Sequence[int], photo: Optional[Photo],
              params: GateParams, edges_kept: bool) -> Tuple[Optional[List[int]], str, Optional[str]]:
    """Crop to the covered block when the unexplained ink forms a separate block.

    :param read: Coverage read (needs Kraken boxes).
    :param covered: Covered read lines.
    :param uncovered_sub: Substantial uncovered read lines.
    :param photo: The photograph (for bounds).
    :param params: Thresholds.
    :param edges_kept: Whether edge lines were kept (their read lines are unknown, so a crop could cut them).
    :return: ``(crop box or None, block side, reason when no crop)``.
    """
    if read.bboxes is None:
        return None, "unknown", "uncovered_ink_no_geometry"
    boxes = [read.bboxes[j] for j in covered if j < len(read.bboxes) and not np.isnan(read.bboxes[j][0])]
    if not boxes:
        return None, "unknown", "uncovered_ink_no_covered_box"
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    h = statistics.median([max(1.0, b[3] - b[1]) for b in boxes])
    outside: List[int] = []
    inside: List[int] = []
    for j in uncovered_sub:
        b = read.bboxes[j]
        if np.isnan(b[0]):
            continue
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        if cx < x0 - h or cx > x1 + h or cy < y0 - h or cy > y1 + h:
            outside.append(j)
        else:
            inside.append(j)
    out_letters = sum(read.line_letters[j] for j in outside)
    in_letters = sum(read.line_letters[j] for j in inside)
    if out_letters == 0 or in_letters > 0.2 * (out_letters + in_letters):
        return None, "interleaved", "uncovered_ink_interleaved"
    if edges_kept:
        return None, "outside", "crop_would_risk_edge_lines"
    cxs = [(read.bboxes[j][0] + read.bboxes[j][2]) / 2.0 for j in outside]
    cys = [(read.bboxes[j][1] + read.bboxes[j][3]) / 2.0 for j in outside]
    mx, my = statistics.mean(cxs), statistics.mean(cys)
    dx = min(abs(mx - x0), abs(mx - x1))
    dy = min(abs(my - y0), abs(my - y1))
    if mx < x0 or mx > x1:
        block = "left" if mx < x0 else "right"
    elif my < y0 or my > y1:
        block = "top" if my < y0 else "bottom"
    else:
        block = "left" if dx < dy and mx < (x0 + x1) / 2 else "right" if dx < dy else "top" if my < (y0 + y1) / 2 \
            else "bottom"
    w = photo.width if photo and photo.width else int(x1 + 2 * h)
    hh = photo.height if photo and photo.height else int(y1 + 2 * h)
    pad = h
    box = [int(max(0, x0 - pad)), int(max(0, y0 - pad)), int(min(w, x1 + pad)), int(min(hh, y1 + pad))]
    for j in outside:
        b = read.bboxes[j]
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
            return None, block, "crop_cannot_separate"
    return box, block, None


def _remedies(unit: Unit, kept_parts: Sequence[Tuple[Side, List[int], List[int], List[int], List[int]]],
              faces: Mapping[str, str], support: Mapping[str, Mapping[str, Set[int]]], group: Group,
              crop: Optional[List[int]], read: ReadStream) -> Tuple[List[str], List[str], Dict[str, List[int]]]:
    """Remedies, flags and moved lines of a unit.

    :param unit: The unit.
    :param kept_parts: Its parts after the edge decision.
    :param faces: Face of every supporting photograph.
    :param support: Supported lines per side per photograph.
    :param group: Its group.
    :param crop: Crop box or None.
    :param read: Coverage read.
    :return: ``(remedies, flags, {other face: lines of these sides placed there})``.
    """
    remedies: List[str] = []
    flags: List[str] = []
    moved: Dict[str, List[int]] = {}
    face_label = side_label(group.photos[unit.face].label) if unit.face in group.photos else None
    trimmed_any = False
    re_pair = False
    for side, core, interior, kept_edge, trimmed in kept_parts:
        here = set(core + interior + kept_edge)
        if trimmed:
            trimmed_any = True
        for sha, lines in support[side.sid].items():
            f = faces.get(sha)
            if f is not None and f != unit.face:
                other = sorted(set(lines) - here)
                if other:
                    moved.setdefault(f, [])
                    moved[f] = sorted(set(moved[f]) | set(other))
        if len(here) < len(side.lines):
            trimmed_any = True
        lab = side_label(side.side_key)
        if lab and face_label and lab != face_label:
            re_pair = True
        if side.old_sha and side.old_sha in group.photos and faces.get(side.old_sha) != unit.face:
            re_pair = True
        if interior:
            flags.append("interior_unread")
        if kept_edge:
            flags.append("edge_kept")
        if trimmed:
            flags.append("edge_trimmed")
    if trimmed_any:
        remedies.append("trim")
    if crop is not None:
        remedies.append("crop")
    if re_pair:
        remedies.append("re_pair")
    if len(kept_parts) > 1:
        remedies.append("merge")
    if read.rotation == 180:
        remedies.append("rotate180")
    if moved:
        flags.append("lines_on_other_face")
    return remedies, sorted(set(flags)), moved


def _rank_duplicates(pairs: List[Dict[str, Any]], params: GateParams) -> None:
    """Rank editions of the same text on one photograph: best evidence first (no source priority).

    Pairs on one photograph whose texts are copies of each other (:func:`_texts_duplicate`) are rivals: the best is
    ``dup_rank`` 0, the next ones fall back in order. Pairs with different texts on one photograph (two documents on
    one sheet) are not ranked against each other.

    :param pairs: Pair records of one group (updated in place: ``dup_rank``).
    :param params: Thresholds.
    """
    by_photo: Dict[str, List[Dict[str, Any]]] = {}
    for p in pairs:
        if p["decision"] == "reject":
            continue
        by_photo.setdefault(p["image_sha256"], []).append(p)

    def rank_key(p: Dict[str, Any]) -> Tuple[int, float, int, int]:
        ev = p["evidence"]
        score = ev["support_share"] * (1.0 - ev["uncovered_share"])
        return (0 if p["decision"] == "accept" else 1, -score, len(p["flags"]), -ev["target_letters"])

    for photo_pairs in by_photo.values():
        if len(photo_pairs) < 2:
            continue
        clusters: List[List[Dict[str, Any]]] = []
        for p in sorted(photo_pairs, key=rank_key):
            text = "\n".join(p["lines"])
            home = next((c for c in clusters if _texts_duplicate(text, "\n".join(c[0]["lines"]), params)), None)
            if home is None:
                clusters.append([p])
            else:
                home.append(p)
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            for k, p in enumerate(cluster):
                p["dup_rank"] = k


def _reject_record(side: Side, group: Group, evals: Mapping[str, List[ReadEval]]) -> Dict[str, Any]:
    """Record of a side that no photograph supports.

    :param side: The side.
    :param group: Its group.
    :param evals: Its read evaluations.
    :return: The reject record.
    """
    best: Optional[ReadEval] = None
    for evs in evals.values():
        for ev in evs:
            if best is None or (ev.klen, ev.q) > (best.klen, best.q):
                best = ev
    reasons = ["no_significant_chain"] if evals else ["no_usable_read"]
    if not group.photos:
        reasons = ["no_photograph"]
    ev_info: Dict[str, Any] = {}
    if best is not None:
        ev_info = {**_chain_info(best), "best_sha256": best.read.sha256}
    return {"pair_id": f"{side.row_id}|{side.side_key}|none", "row_id": side.row_id, "side_key": side.side_key,
            "side_keys": [side.side_key], "source": side.source, "document_group": group.document_group,
            "script": side.script, "ja_signal": None, "tier": None, "image_sha256": None, "image_path": None,
            "image_label": None, "duplicate_photos": [], "decision": "reject", "remedies": [], "line_indices": [],
            "lines": [], "trimmed_lines": {}, "moved_to": {}, "rotation": 0, "crop": None,
            "evidence": {**ev_info, "target_letters": side.letters(range(len(side.lines)))}, "dup_rank": None,
            "flags": [], "reasons": reasons}


def _line_evidence(side: Side, evals: Mapping[str, List[ReadEval]], faces: Mapping[str, str],
                   assign: Sequence[Optional[str]]) -> List[Dict[str, Any]]:
    """Per-line audit records: best photograph and read, score, z, assigned face.

    :param side: The side.
    :param evals: Its read evaluations.
    :param faces: Face of every supporting photograph.
    :param assign: Face per line.
    :return: One record per line.
    """
    out: List[Dict[str, Any]] = []
    for i, line in enumerate(side.lines):
        best: Optional[Tuple[float, str, str, float, float]] = None
        units = [u for u, li in enumerate(side.unit_line) if li == i]
        for sha, evs in evals.items():
            for ev in evs:
                for u in units:
                    zi = ev.z[u]
                    if np.isnan(zi):
                        continue
                    if best is None or zi > best[0]:
                        best = (float(zi), sha, ev.read.reader, float(ev.scores[u, 0]), float(ev.scores[u, 1]))
        out.append({"sid": side.sid, "line": i, "letters": len(side.skels[i]), "face": assign[i],
                    "best_z": round(best[0], 2) if best else None, "best_sha256": best[1] if best else None,
                    "best_reader": best[2] if best else None, "best_score": round(best[3], 1) if best else None,
                    "pos": best[4] if best else None})
    return out


def params_dict(params: GateParams) -> Dict[str, Any]:
    """Thresholds as a dict (for run records).

    :param params: Thresholds.
    :return: Field values.
    """
    return asdict(params)
