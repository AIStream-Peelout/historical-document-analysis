"""Line evidence for gate v2: target lines aligned against the reads of a photograph, scored against a null.

A read becomes a :class:`ReadStream`: the letter skeletons of its lines joined into one stream, with the read line
that owns every stream position, per-line letters and confidences, and Kraken's line boxes in the ORIGINAL photograph's
pixel frame. A target line (skeleton of at least :data:`MIN_LINE_LETTERS` letters) is matched against the stream with
``rapidfuzz.fuzz.partial_ratio_alignment`` (best-matching substring, InDel similarity 0-100); the span gives its
position in the read and the read lines it touches.

The null of a line is its score against :data:`K_NULL` reads of photographs of OTHER document groups made by the same
reader family, of the same script family and in the same read-length bin as the read under test, so formulaic lines
(a basmala, a letter opening) score high against the null too and get a low z. Draws are deterministic (seeded by the
target and the null key). See ``logs/next_round/ja_ar_v1/build/GATE_V2_SPEC.md`` §2.
"""
from __future__ import annotations

import bisect
import hashlib
import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Indel

from src.finetuning.qwen_hebrew.ja_ar_build import textnorm

#: A target line shorter than this (letters) has no evidence of its own.
MIN_LINE_LETTERS = 5
#: Null reads per (target side, null key).
K_NULL = 40
#: Floor of the null standard deviation of a line's score (score units, 0-100).
SD_FLOOR = 3.0
#: Read-length bins (letters) of the null; a read falls in the first bin whose upper bound exceeds its letters.
BIN_EDGES = (150, 400, 900, 1800)
#: Minimum reads in a null bin before neighbouring bins are pooled in.
MIN_BIN_READS = K_NULL


@dataclass
class ReadStream:
    """One usable read of one photograph, as a letter stream.

    :ivar key: Unique key ``<sha256>|<reader>|<rotation>|<crop>``.
    :ivar sha256: Photograph digest.
    :ivar reader: Reader id (``kraken:<model>``, ``vlm:<model>``, ``gemini:<model>``).
    :ivar family: ``kraken`` / ``vlm`` / ``gemini``.
    :ivar rotation: Rotation of the image the reader saw, relative to the stored photograph (degrees).
    :ivar stream: Joined letter skeletons of the read lines.
    :ivar owner: Read-line index of every stream position (same length as ``stream``).
    :ivar line_letters: Skeleton letters per read line.
    :ivar line_conf: Mean confidence per read line (Kraken), None per line otherwise.
    :ivar bboxes: Line boxes ``[x0, y0, x1, y1]`` in the stored photograph's pixel frame (Kraken, rotation 0 and no
        crop only), else None.
    :ivar script: ``hebrew`` or ``arabic`` by the read's letters.
    :ivar crop: Crop box the reader saw (original pixels) or None.
    """

    key: str
    sha256: str
    reader: str
    family: str
    rotation: int
    stream: str
    owner: List[int]
    line_letters: List[int]
    line_conf: List[Optional[float]]
    bboxes: Optional[List[List[float]]]
    script: str
    crop: Optional[List[float]] = None
    mstream: str = ""
    _kindex: Optional[Dict[int, Dict[str, List[int]]]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Fill the matching form of the stream (Arabic letters as rasm)."""
        if not self.mstream:
            self.mstream = textnorm.match_form(self.stream)

    def kgram_index(self, k: int) -> Dict[str, List[int]]:
        """Positions of every k-gram of the matching stream (built once per k, then cached).

        :param k: k-gram length.
        :return: ``{k-gram: [start positions]}``.
        """
        if self._kindex is None:
            self._kindex = {}
        if k not in self._kindex:
            idx: Dict[str, List[int]] = {}
            s = self.mstream
            for p in range(len(s) - k + 1):
                idx.setdefault(s[p:p + k], []).append(p)
            self._kindex[k] = idx
        return self._kindex[k]

    @property
    def letters(self) -> int:
        """Letters in the read.

        :return: Length of the stream.
        """
        return len(self.stream)

    @property
    def n_lines(self) -> int:
        """Read lines.

        :return: Number of read lines (including lines without letters).
        """
        return len(self.line_letters)


def reader_family(reader: str) -> str:
    """Reader family of a reader id (the part before the colon).

    :param reader: Reader id such as ``kraken:MiDRASH_Gen_01``.
    :return: ``kraken``, ``vlm``, ``gemini`` or the raw prefix.
    """
    return reader.split(":", 1)[0]


def length_bin(n_letters: int) -> int:
    """Read-length bin of a read.

    :param n_letters: Letters in the read.
    :return: Bin index 0..len(BIN_EDGES).
    """
    for i, edge in enumerate(BIN_EDGES):
        if n_letters < edge:
            return i
    return len(BIN_EDGES)


def stream_from_read(sha256: str, reader: str, lines: Sequence[str], meta: Mapping[str, object]) -> ReadStream:
    """Build the letter stream of one read.

    Kraken boxes are scaled from the uploaded copy back to the stored photograph (``meta['scale']`` = copy width /
    original width) when the service did not rotate the page and the client sent no rotation or crop; otherwise the
    boxes are dropped (no geometry), the text is still used.

    :param sha256: Photograph digest.
    :param reader: Reader id.
    :param lines: Read lines in reader order.
    :param meta: The read record's ``meta`` (``confidences``, ``bboxes``, ``scale``, ``view``, ``crop``,
        ``rotation_client`` ...).
    :return: The stream.
    """
    view = meta.get("view") or {}
    service_rotation = int(view.get("rotation") or 0) if isinstance(view, dict) else 0
    client_rotation = int(meta.get("rotation_client") or 0)
    crop = meta.get("crop")
    rotation = (service_rotation + client_rotation) % 360
    skels = textnorm.line_skeletons(lines)
    stream_parts: List[str] = []
    owner: List[int] = []
    for idx, sk in enumerate(skels):
        stream_parts.append(sk)
        owner.extend([idx] * len(sk))
    confs_raw = meta.get("confidences")
    confs: List[Optional[float]] = []
    for idx in range(len(skels)):
        value = confs_raw[idx] if isinstance(confs_raw, list) and idx < len(confs_raw) else None
        confs.append(float(value) if isinstance(value, (int, float)) else None)
    bboxes: Optional[List[List[float]]] = None
    raw_boxes = meta.get("bboxes")
    scale = meta.get("scale")
    prepared = meta.get("prepared_size")
    flip = (service_rotation == 0 and client_rotation == 180 and isinstance(prepared, (list, tuple))
            and len(prepared) == 2)
    if (isinstance(raw_boxes, list) and len(raw_boxes) == len(skels) and (rotation == 0 or flip) and not crop
            and isinstance(scale, (int, float)) and scale > 0):
        bboxes = []
        width = float(prepared[0]) / float(scale) if flip else 0.0
        height = float(prepared[1]) / float(scale) if flip else 0.0
        for box in raw_boxes:
            if isinstance(box, (list, tuple)) and len(box) == 4:
                b = [float(v) / float(scale) for v in box]
                # a 180-degree client rotation: map the box back to the stored photograph's frame
                bboxes.append([width - b[2], height - b[3], width - b[0], height - b[1]] if flip else b)
            else:
                bboxes.append([float("nan")] * 4)
    stream = "".join(stream_parts)
    family = reader_family(reader)
    crop_key = ",".join(str(int(v)) for v in crop) if isinstance(crop, (list, tuple)) else ""
    return ReadStream(key=f"{sha256}|{reader}|{rotation}|{crop_key}", sha256=sha256, reader=reader, family=family,
                      rotation=rotation, stream=stream, owner=owner, line_letters=[len(s) for s in skels],
                      line_conf=confs, bboxes=bboxes, script=textnorm.script_family(stream),
                      crop=list(crop) if isinstance(crop, (list, tuple)) else None)


def line_scores(target_skels: Sequence[str], read: ReadStream) -> np.ndarray:
    """Score every target line against one read.

    :param target_skels: Letter skeletons of the target's lines.
    :param read: The read.
    :return: Array ``(n_lines, 3)``: score (0-100), span start, span end in the read stream; NaN rows for lines with
        fewer than :data:`MIN_LINE_LETTERS` letters or when the read has fewer letters than that.
    """
    out = np.full((len(target_skels), 3), np.nan, dtype=np.float64)
    if read.letters < MIN_LINE_LETTERS:
        return out
    for i, sk in enumerate(target_skels):
        if len(sk) < MIN_LINE_LETTERS:
            continue
        a = fuzz.partial_ratio_alignment(textnorm.match_form(sk), read.mstream)
        out[i, 0] = a.score
        out[i, 1] = a.dest_start
        out[i, 2] = a.dest_end
    return out


#: k of the ordered k-gram chain, per script family of the read (rasm folding shrinks the Arabic alphabet).
KGRAM_K = {"hebrew": 4, "arabic": 5}
#: Occurrences of one k-gram kept per read (repetitive texts).
KGRAM_MAX_OCC = 8


#: A k-gram found in at least this share of the reads of its script is boilerplate (formulae, common words) and never
#: counts as evidence in the k-gram channel ...
STOP_DF_SHARE = 0.02
#: ... with at least this many reads.
STOP_DF_MIN = 5


def stop_kgrams(streams: Iterable[ReadStream]) -> Dict[str, Set[str]]:
    """Boilerplate k-grams per script family: k-grams present in many reads.

    :param streams: Reads (one per photograph and reader is enough).
    :return: ``{script family: set of k-grams}``.
    """
    df: Dict[str, Dict[str, int]] = {}
    n_reads: Dict[str, int] = {}
    for st in streams:
        k = KGRAM_K.get(st.script, 4)
        grams = {st.mstream[p:p + k] for p in range(len(st.mstream) - k + 1)}
        counts = df.setdefault(st.script, {})
        for g in grams:
            counts[g] = counts.get(g, 0) + 1
        n_reads[st.script] = n_reads.get(st.script, 0) + 1
    out: Dict[str, Set[str]] = {}
    for script, counts in df.items():
        cut = max(STOP_DF_MIN, STOP_DF_SHARE * n_reads[script])
        out[script] = {g for g, n in counts.items() if n >= cut}
    return out


def kgram_chain(target_skels: Sequence[str], read: ReadStream, stop: Optional[Set[str]] = None
                ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Ordered k-gram chain of a target against one read: the longest set of shared k-grams in the same order.

    The target's units are joined (k-grams never cross a unit boundary); every target k-gram that is not boilerplate
    (``stop``) is matched to the read's occurrences of it (at most :data:`KGRAM_MAX_OCC`); the longest chain strictly
    increasing in both target and read position is kept (patience sorting).

    :param target_skels: Unit skeletons of the target.
    :param read: The read.
    :param stop: Boilerplate k-grams of the read's script (ignored), or None.
    :return: ``(matched k-grams per unit, [(unit index, read position)] of the chain)``.
    """
    k = KGRAM_K.get(read.script, 4)
    per_unit = np.zeros(len(target_skels), dtype=np.int64)
    if read.letters < k:
        return per_unit, []
    idx = read.kgram_index(k)
    pairs: List[Tuple[int, int, int]] = []
    t = 0
    for u, sk in enumerate(target_skels):
        m = textnorm.match_form(sk)
        for p in range(len(m) - k + 1):
            gram = m[p:p + k]
            if stop is not None and gram in stop:
                continue
            occ = idx.get(gram)
            if occ:
                # descending read positions per target position: at most one is used by a strictly increasing chain
                for r in sorted(occ[:KGRAM_MAX_OCC], reverse=True):
                    pairs.append((t + p, r, u))
        t += len(m) + k
    if not pairs:
        return per_unit, []
    tails: List[int] = []
    tail_idx: List[int] = []
    prev = [-1] * len(pairs)
    for j, (_tp, r, _u) in enumerate(pairs):
        pos = bisect.bisect_left(tails, r)
        if pos == len(tails):
            tails.append(r)
            tail_idx.append(j)
        else:
            tails[pos] = r
            tail_idx[pos] = j
        prev[j] = tail_idx[pos - 1] if pos > 0 else -1
    chain: List[Tuple[int, int]] = []
    j = tail_idx[-1]
    while j >= 0:
        _tp, r, u = pairs[j]
        chain.append((u, r))
        per_unit[u] += 1
        j = prev[j]
    chain.reverse()
    return per_unit, chain


def kgram_null(target_skels: Sequence[str], nulls: Sequence[ReadStream],
               stops: Optional[Mapping[str, Set[str]]] = None) -> np.ndarray:
    """Ordered k-gram chain lengths of the target against every null read.

    :param target_skels: Unit skeletons.
    :param nulls: Null reads.
    :param stops: Boilerplate k-grams per script family, or None.
    :return: ``(K,)`` chain lengths.
    """
    return np.array([len(kgram_chain(target_skels, r, (stops or {}).get(r.script))[1]) for r in nulls],
                    dtype=np.float64)


def matched_read_lines(target_skel: str, read: ReadStream, start: float, end: float) -> Dict[int, int]:
    """Read lines that a target unit's alignment actually matches, with the matched letters on each.

    The unit is aligned (InDel opcodes) against the read window its best-substring match found; only letters inside
    ``equal`` blocks count, so short garbage lines that merely fall inside the window are not matched.

    :param target_skel: The unit's skeleton.
    :param read: The read.
    :param start: Window start in the read stream.
    :param end: Window end (exclusive).
    :return: ``{read line index: matched letters}``.
    """
    if np.isnan(start) or np.isnan(end) or read.letters == 0 or not target_skel:
        return {}
    s = max(0, int(start))
    e = min(read.letters, max(s + 1, int(end)))
    window = read.mstream[s:e]
    out: Dict[int, int] = {}
    for op in Indel.opcodes(textnorm.match_form(target_skel), window):
        if op.tag != "equal":
            continue
        for pos in range(s + op.dest_start, s + op.dest_end):
            line = read.owner[pos]
            out[line] = out.get(line, 0) + 1
    return out


def touched_lines(read: ReadStream, start: float, end: float) -> List[int]:
    """Read lines covered by a stream span.

    :param read: The read.
    :param start: Span start (stream index, may be float from an array).
    :param end: Span end (exclusive).
    :return: Sorted distinct read-line indices.
    """
    if np.isnan(start) or np.isnan(end) or read.letters == 0:
        return []
    s = max(0, int(start))
    e = min(read.letters, max(s + 1, int(end)))
    return sorted(set(read.owner[s:e]))


@dataclass
class NullKey:
    """What a null read must share with the read under test.

    :ivar family: Reader family.
    :ivar script: Script family of the read.
    :ivar bin: Read-length bin.
    """

    family: str
    script: str
    bin: int

    def as_tuple(self) -> Tuple[str, str, int]:
        """Hashable form.

        :return: ``(family, script, bin)``.
        """
        return (self.family, self.script, self.bin)


def null_key_of(read: ReadStream) -> NullKey:
    """Null key of a read.

    :param read: The read under test.
    :return: Its null key.
    """
    return NullKey(read.family, read.script, length_bin(read.letters))


@dataclass
class NullPool:
    """Reads available as null evidence, indexed by (family, script, bin).

    :ivar by_key: Reads per key.
    """

    by_key: Dict[Tuple[str, str, int], List[ReadStream]] = field(default_factory=dict)
    stops: Dict[str, Set[str]] = field(default_factory=dict)
    targets: Dict[Tuple[str, int], List[Tuple[str, List[str]]]] = field(default_factory=dict)

    def add_targets(self, items: Iterable[Tuple[str, str, List[str]]]) -> None:
        """Register targets of all document groups for the read-side null of the k-gram channel.

        :param items: ``(document group, script family, unit skeletons)`` per target side.
        """
        for gid, script, units in items:
            letters = sum(len(u) for u in units)
            if letters < MIN_LINE_LETTERS:
                continue
            self.targets.setdefault((script, length_bin(letters)), []).append((gid, list(units)))
        for lst in self.targets.values():
            lst.sort(key=lambda x: (x[0], len(x[1])))

    def draw_targets(self, script: str, letters: int, exclude_group: str, seed_text: str, k: int = K_NULL
                     ) -> List[List[str]]:
        """Deterministic draw of other documents' targets of similar length (read-side null).

        :param script: Script family of the target under test.
        :param letters: Its letters.
        :param exclude_group: Its document group (never drawn).
        :param seed_text: Seed material.
        :param k: Targets wanted.
        :return: Unit-skeleton lists.
        """
        b = length_bin(letters)
        cands = [u for gid, u in self.targets.get((script, b), []) if gid != exclude_group]
        radius = 1
        while len(cands) < k and radius <= len(BIN_EDGES):
            for bb in (b - radius, b + radius):
                if 0 <= bb <= len(BIN_EDGES):
                    cands.extend(u for gid, u in self.targets.get((script, bb), []) if gid != exclude_group)
            radius += 1
        rng = random.Random(int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:12], 16))
        return cands if len(cands) <= k else rng.sample(cands, k)

    @classmethod
    def from_streams(cls, streams: Iterable[ReadStream]) -> "NullPool":
        """Index reads by their null key (unrotated, uncropped reads only, one per (photo, reader)) and derive the
        boilerplate k-grams of each script family from the same reads.

        :param streams: Usable reads.
        :return: The pool.
        """
        pool = cls()
        seen: Set[Tuple[str, str]] = set()
        kept: List[ReadStream] = []
        for st in streams:
            if st.rotation != 0 or st.crop or st.letters < MIN_LINE_LETTERS:
                continue
            if (st.sha256, st.reader) in seen:
                continue
            seen.add((st.sha256, st.reader))
            kept.append(st)
            pool.by_key.setdefault(null_key_of(st).as_tuple(), []).append(st)
        for reads in pool.by_key.values():
            reads.sort(key=lambda r: r.key)
        pool.stops = stop_kgrams(kept)
        return pool

    def candidates(self, key: NullKey) -> List[ReadStream]:
        """Reads of the key's bin, widened to neighbouring bins until at least :data:`MIN_BIN_READS` are available.

        :param key: Null key.
        :return: Candidate null reads (may be fewer than K_NULL when the family/script has few reads).
        """
        out = list(self.by_key.get(key.as_tuple(), []))
        radius = 1
        while len(out) < MIN_BIN_READS and radius <= len(BIN_EDGES):
            for b in (key.bin - radius, key.bin + radius):
                if 0 <= b <= len(BIN_EDGES):
                    out.extend(self.by_key.get((key.family, key.script, b), []))
            radius += 1
        return out

    def draw(self, key: NullKey, exclude_shas: Set[str], seed_text: str, k: int = K_NULL) -> List[ReadStream]:
        """Deterministic draw of null reads for one target side and key.

        :param key: Null key of the read under test.
        :param exclude_shas: Photographs of the target's own document group (never null evidence).
        :param seed_text: Seed material (target id + key).
        :param k: Reads wanted.
        :return: Up to ``k`` reads.
        """
        cands = [r for r in self.candidates(key) if r.sha256 not in exclude_shas]
        seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:12], 16)
        rng = random.Random(seed)
        if len(cands) <= k:
            return cands
        return rng.sample(cands, k)


def null_scores(target_skels: Sequence[str], nulls: Sequence[ReadStream]) -> np.ndarray:
    """Scores of the target lines against every null read.

    :param target_skels: Line skeletons of the target.
    :param nulls: Null reads.
    :return: Array ``(K, n_lines, 3)`` like :func:`line_scores` per null read.
    """
    if not nulls:
        return np.full((0, len(target_skels), 3), np.nan)
    return np.stack([line_scores(target_skels, r) for r in nulls])


def z_scores(scores: np.ndarray, null: np.ndarray) -> np.ndarray:
    """z of each line's score against the null scores of that line.

    :param scores: ``(n,)`` scores of the lines against the read under test (NaN = no evidence).
    :param null: ``(K, n)`` null scores.
    :return: ``(n,)`` z values (NaN where the line has no evidence or the null is empty).
    """
    if null.shape[0] == 0:
        return np.full(scores.shape, np.nan)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(null, axis=0)
        sd = np.maximum(np.nanstd(null, axis=0), SD_FLOOR)
        return (scores - mu) / sd


def loo_z(null: np.ndarray) -> np.ndarray:
    """Leave-one-out z of every null read's line scores against the other null reads.

    :param null: ``(K, n)`` null scores.
    :return: ``(K, n)`` z values.
    """
    k = null.shape[0]
    if k < 3:
        return np.full(null.shape, np.nan)
    with np.errstate(invalid="ignore"):
        total = np.nansum(null, axis=0)
        total_sq = np.nansum(null ** 2, axis=0)
        counts = np.sum(~np.isnan(null), axis=0)
        mu = (total[None, :] - np.nan_to_num(null)) / np.maximum(counts[None, :] - 1, 1)
        var = (total_sq[None, :] - np.nan_to_num(null) ** 2) / np.maximum(counts[None, :] - 1, 1) - mu ** 2
        sd = np.maximum(np.sqrt(np.maximum(var, 0.0)), SD_FLOOR)
        return (null - mu) / sd
