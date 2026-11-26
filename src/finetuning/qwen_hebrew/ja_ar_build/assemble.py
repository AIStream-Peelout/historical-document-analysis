# File name: assemble.py
# Date: 11/19/26
# Author: Isaac Godfried. Coded by Claude.
"""Photo-centric line assembly (round ja_ar_v1): which transcription lines stand on which photograph.

Why. Gate v2 (:mod:`.assign`) places a transcription *side* on a photograph and repairs the fit (trim, merge, crop).
That model holds when an edition's sides are the leaf's faces. The legacy-store transcriptions are typeset pages of a
document: their page breaks are the typesetter's, not the manuscript's. A photographed page then holds the tail of
one typeset page, all of the next and the head of a third; an opening holds two manuscript pages; bound leaves are
photographed out of text order. The blind audit of 2026-11-19 (64 accepted pairs of gate run ``novel1119_00``) found
78 % complete pairs, and 8 of the 14 failures were exactly this: a few lines too many or too few at a page boundary,
or a whole page of text inside a merge that the photograph does not show. The same mismatch sent most sides of the
long Judaeo-Arabic manuscripts to review.

Model. One candidate (one transcription of one document) is ONE sequence of lines, in the edition's order. Every
photograph of the document holds zero or more runs of consecutive lines of that sequence. The module

1. scores every line against every read of every photograph with the gate's line channel (:func:`align.line_scores`,
   z against null reads of other documents: :func:`score_candidate`);
2. assigns each line a photograph by dynamic programming (:func:`viterbi`): a line's weight on a photograph is its
   capped z; changing photograph between consecutive lines costs :attr:`AssembleParams.switch`, so a run needs
   real evidence and a single stray match never moves a line;
3. finds the manuscript pages a photograph shows (:func:`page_columns`: one, or two side by side) from the boxes of
   the read lines the supported lines match;
4. proposes a place for the lines no read supports (:func:`settle_unsupported`) by the ROOM the page has for them,
   measured in line pitches (the usual distance between consecutive supported lines of that page): a stretch
   between two supported lines of one page is put into the text when the page has that many line slots between
   them (damaged lines the reader could not read); a stretch at the end of a page (between two pages, before the
   first or after the last supported line) is shared out by the read lines left unexplained below the last
   supported line of the first page and above the first supported line of the second, weak evidence choosing the
   cut. A line without room, an unsupported line with a weak sign of an editor's wording
   (:func:`fgp_notes.soft_marker`, Latin letters, digits) and a run left with a single weakly supported line
   (:func:`drop_stray_runs`) are put into no text. That is a proposal, never a finding: a reader that could not
   read a line says nothing about whether the line is on the photograph (integrity audit of 2026-11-22: the
   earlier rule cut such lines silently and accepted the shortened text);
5. builds each photograph's text (:func:`assemble_photo`): its pages right to left, each page's runs top to bottom,
   lines in edition order; and decides it: accepted when enough of its letters are supported, few unread letters
   sit inside it and the read holds little ink the text does not explain; review otherwise, with reasons. When
   the unexplained ink lies apart from the explained lines (another page, another text on the sheet), the
   photograph is cropped to the explained block instead (:func:`crop_to_text`, remedy ``crop``). A photograph
   that shows two pages with a clear gutter between their lines gives one pair per page, cropped at the gutter
   (:func:`page_crops`, remedy ``split``): half the text at twice the letter size on the same image budget;
6. accounts for EVERY line of the candidate (:func:`line_status`): in a pair's text (``read`` or ``kept_unread``),
   left out on the evidence of the document itself (``note``: an editor's label or note by its wording,
   :func:`fgp_notes.classify_line`; ``no_letters``), or unresolved (``open``, and ``doubled`` for a line that
   matches a read line which already has its text: the edition may set the line twice, or the manuscript may
   repeat it, a refrain or a copied block, and the alignment cannot tell which). Every
   stretch of unresolved lines is attached to the pairs whose text it borders (:func:`attach_open_lines`, field
   ``open_lines``), and such a pair is never accepted here (reason ``unresolved_lines``): a judge who looks at
   the photograph says whether the lines stand on it.

The output is gate-v2 pair records (remedy ``assemble``), one per photograph and candidate (``unread_lines`` gives
the numbers of the text lines no read supports), plus the per-line audit records (each with its ``status`` and the
pair that holds it), so :mod:`.build_rows`, the blind-audit tools and the
tallies read them. Reads are evidence for placement only: no read ever becomes text, and no read's silence removes
text.

Example::

    PYTHONPATH=. .venv/bin/python -m src.finetuning.qwen_hebrew.ja_ar_build.assemble \\
        --candidates W2/novel1119/stages/stage1_images.jsonl --reads-dir W2/novel1119 W1 \\
        --null-candidates W1/stages/stage1_images.jsonl --out W2/gate/novel1119_asm01 --workers 6
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Indel

from src.finetuning.qwen_hebrew.ja_ar_build import align, fgp_notes, textnorm
from src.finetuning.qwen_hebrew.ja_ar_build.align import NullPool, ReadStream
from src.finetuning.qwen_hebrew.ja_ar_build.assign import Group, Photo


@dataclass
class AssembleParams:
    """Thresholds of the assembly.

    :ivar z_sup: z at which a line is supported on a photograph ...
    :ivar s_sup: ... with at least this alignment score (0-100).
    :ivar z_cap: Cap of one line's weight in the assignment.
    :ivar z_floor: Floor of one line's weight (a mismatch never costs more than this).
    :ivar switch: Cost of changing photograph between consecutive lines.
    :ivar ink_letters: Letters of a read line that counts as ink ...
    :ivar ink_conf: ... with at least this mean confidence (Kraken; reads without confidences always count).
    :ivar cover_match_min: Matched letters a read line needs to count as explained by a text line ...
    :ivar cover_match_share: ... and this share of its letters.
    :ivar room_factor: Line slots a page needs per unsupported text line to keep a whole stretch of them.
    :ivar read_line_letters: Letters of a read line that marks written lines (the page's text extent, its pitch).
    :ivar stray_z: z a lone supported line needs to stand as a run of its own ...
    :ivar stray_letters: ... with at least this many letters.
    :ivar weak_z: z of weak evidence used to choose among unsupported lines when room is short.
    :ivar tail_letters: A line of at most this many letters may be the wrapped end of the line before it ...
    :ivar tail_score: ... when it matches the read right after that line's match at least this well (0-100).
    :ivar crop_overlap_max: Share of an unexplained line's box a crop may cut into.
    :ivar crop_gap: Clear distance (line heights) unexplained ink must keep from the explained block for a crop.
    :ivar unread_ends_max: Share of an accepted pair's letters in unread lines at the ends of its runs.
    :ivar unread_max: Share of an accepted pair's letters in unread lines of any kind. Blind audits asm01-asm03
        (2026-11-19, 177 pairs): pairs at or under 10 % were complete in 65 of 68 cases after the page-end rules,
        pairs above it in 12 of 29 (lines the edition leaves out stand next to lines the reader could not read).
    :ivar dup_ratio: Similarity (0-100) at which two lines matching the same read line are one line set twice.
    :ivar gutter_min: Clear width (share of the photograph's width) between the lines of two pages that allows one
        pair per page.
    :ivar unmatched_end_max: Median share of a supported line that may lie outside its match, at the line's start
        or end, in an accepted pair (an edition that sets a translation or a commentary beside each line matches
        only one half of every line; a poor read misses letters all along the line instead).
    :ivar crop_min_area: Share of the photograph a crop must keep.
    :ivar column_gap: Horizontal gap (share of the photograph's width) between matched boxes that splits two pages.
    :ivar column_min_lines: Supported lines each of the two pages needs.
    :ivar support_min: Supported share of an accepted photograph's letters (Kraken evidence) ...
    :ivar support_min_vlm: ... and when only a page-reader read supports it.
    :ivar interior_max: Share of unread letters inside an accepted photograph's text.
    :ivar uncovered_max: Share of the read's ink letters an accepted photograph may leave unexplained.
    :ivar slack: Read letters one kept unread text letter explains.
    :ivar min_lines: Lines a pair needs.
    :ivar min_letters: Letters a pair needs.
    """

    z_sup: float = 3.5
    s_sup: float = 55.0
    z_cap: float = 8.0
    z_floor: float = -1.0
    switch: float = 6.0
    ink_letters: int = 8
    ink_conf: float = 0.5
    cover_match_min: int = 2
    cover_match_share: float = 0.3
    room_factor: float = 0.6
    read_line_letters: int = 3
    stray_z: float = 6.0
    stray_letters: int = 15
    weak_z: float = 1.5
    tail_letters: int = 15
    tail_score: float = 80.0
    crop_overlap_max: float = 0.3
    crop_gap: float = 0.6
    unread_ends_max: float = 0.10
    unread_max: float = 0.10
    dup_ratio: float = 90.0
    unmatched_end_max: float = 0.45
    gutter_min: float = 0.02
    crop_min_area: float = 0.05
    column_gap: float = 0.22
    column_min_lines: int = 2
    support_min: float = 0.55
    support_min_vlm: float = 0.40
    interior_max: float = 0.35
    uncovered_max: float = 0.15
    slack: float = 1.25
    min_lines: int = 3
    min_letters: int = 60


@dataclass
class LineSeq:
    """The lines of one candidate as one sequence.

    :ivar row_id: Candidate id.
    :ivar source: Candidate source.
    :ivar script: Candidate script.
    :ivar refs: ``(side key, line index in the side)`` per line.
    :ivar lines: Line texts.
    :ivar skels: Letter skeletons.
    """

    row_id: str
    source: str
    script: str
    refs: List[Tuple[str, int]] = field(default_factory=list)
    lines: List[str] = field(default_factory=list)
    skels: List[str] = field(default_factory=list)

    @property
    def letters(self) -> np.ndarray:
        """Skeleton letters per line.

        :return: ``(N,)`` integers.
        """
        return np.array([len(s) for s in self.skels], dtype=np.int64)


@dataclass
class PhotoEval:
    """Scores of one candidate's lines on one photograph.

    :ivar sha: Photograph digest.
    :ivar streams: Its usable reads.
    :ivar z: ``(S, N)`` z per read and line (NaN: no evidence).
    :ivar score: ``(S, N)`` alignment scores.
    :ivar start: ``(S, N)`` span starts in the read streams.
    :ivar end: ``(S, N)`` span ends.
    """

    sha: str
    streams: List[ReadStream]
    z: np.ndarray
    score: np.ndarray
    start: np.ndarray
    end: np.ndarray

    def best(self) -> np.ndarray:
        """Best z per line over the photograph's reads.

        :return: ``(N,)`` z (NaN where no read gives evidence).
        """
        with np.errstate(invalid="ignore"):
            out = np.nanmax(np.where(np.isnan(self.z), -np.inf, self.z), axis=0)
        return np.where(np.isinf(out), np.nan, out)


def line_sequences(group: Group) -> List[LineSeq]:
    """One line sequence per candidate of a group, sides in file order.

    :param group: The document group.
    :return: Sequences (candidates without lines are left out).
    """
    by_row: Dict[str, LineSeq] = {}
    for side in group.sides:
        seq = by_row.setdefault(side.row_id, LineSeq(side.row_id, side.source, side.script))
        for i, (line, skel) in enumerate(zip(side.lines, side.skels)):
            seq.refs.append((side.side_key, i))
            seq.lines.append(line)
            seq.skels.append(skel)
    return [s for s in by_row.values() if s.lines]


def read_speaks_for(script: str, stream: align.ReadStream) -> bool:
    """Whether a read can say anything about a candidate's lines.

    A line reader trained on Hebrew script turns an Arabic-script page into Hebrew letters. Such a read neither
    confirms Arabic-script lines nor shows writing they leave unexplained, so for an Arabic-script candidate only a
    read in Arabic script is evidence (user, 2026-11-19: nothing is held or left out because a reader cannot read
    the script; such documents go to the judges instead).

    :param script: The candidate's script.
    :param stream: A read of one of its photographs.
    :return: False for a read in another script than an Arabic-script candidate's.
    """
    return script != "arabic" or stream.script == "arabic"


def score_candidate(seq: LineSeq, group: Group, pool: NullPool) -> List[PhotoEval]:
    """Line scores and z of a candidate on every read of every photograph of its group.

    The null of a read is drawn once per (candidate, reader family, script, read-length bin). Reads that cannot
    speak for the candidate's script are left out (:func:`read_speaks_for`).

    :param seq: The candidate's lines.
    :param group: Its document group (photographs and reads).
    :param pool: Null reads of other documents.
    :return: One evaluation per photograph that has a usable read.
    """
    exclude = set(group.photos) | set(group.null_exclude)
    null_stats: Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray]] = {}
    out: List[PhotoEval] = []
    for sha in sorted(group.reads):
        streams = [st for st in group.reads[sha]
                   if st.letters >= align.MIN_LINE_LETTERS and read_speaks_for(seq.script, st)]
        if not streams:
            continue
        zs, scs, sts, ens = [], [], [], []
        for st in streams:
            sc = align.line_scores(seq.skels, st)
            key = align.null_key_of(st)
            if key.as_tuple() not in null_stats:
                nulls = pool.draw(key, exclude, f"{seq.row_id}|{key.as_tuple()}")
                null = align.null_scores(seq.skels, nulls)[:, :, 0]
                if null.shape[0]:
                    with np.errstate(invalid="ignore"):
                        mu = np.nanmean(null, axis=0)
                        sd = np.maximum(np.nanstd(null, axis=0), align.SD_FLOOR)
                else:  # no null read of this kind: a fixed, cautious scale
                    mu, sd = np.full(len(seq.skels), 50.0), np.full(len(seq.skels), 8.0)
                null_stats[key.as_tuple()] = (mu, sd)
            mu, sd = null_stats[key.as_tuple()]
            zs.append((sc[:, 0] - mu) / sd)
            scs.append(sc[:, 0])
            sts.append(sc[:, 1])
            ens.append(sc[:, 2])
        out.append(PhotoEval(sha, streams, np.vstack(zs), np.vstack(scs), np.vstack(sts), np.vstack(ens)))
    return out


def viterbi(weights: np.ndarray, switch: float) -> np.ndarray:
    """The photograph of every line that maximises summed weights minus the switch costs.

    :param weights: ``(N, M)`` weight of line i on photograph m.
    :param switch: Cost of changing photograph between consecutive lines.
    :return: ``(N,)`` photograph index per line.
    """
    n, m = weights.shape
    if n == 0 or m == 0:
        return np.zeros(n, dtype=np.int64)
    score = weights[0].astype(np.float64).copy()
    back = np.zeros((n, m), dtype=np.int64)
    states = np.arange(m)
    for i in range(1, n):
        top = int(np.argmax(score))
        move = score[top] - switch
        take = move > score
        back[i] = np.where(take, top, states)
        score = np.where(take, move, score) + weights[i]
    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


@dataclass
class PhotoState:
    """Read-side facts of one photograph for one candidate.

    :ivar sha: Photograph digest.
    :ivar stream: Its primary read (the one that supports most of the lines assigned to the photograph).
    :ivar rotation: Rotation of that read relative to the stored photograph.
    :ivar order: Reading-order key per read line (top to bottom in the reading orientation, or the read's own order
        when it has no boxes).
    :ivar xcenter: Horizontal centre per read line in the reading orientation, as a share of the width (NaN: no box).
    :ivar ink: Whether a read line counts as ink.
    :ivar covered: Whether a text line explains the read line.
    :ivar column: Page of each read line (0 = the right page, 1 = the left page).
    :ivar split: Width share that separates the two pages, or None for a single page.
    :ivar pitch: Line pitch per page, in units of ``order``.
    :ivar top: Reading-order key of the first written line per page.
    :ivar bottom: Reading-order key of the last written line per page.
    """

    sha: str
    stream: ReadStream
    rotation: int
    order: np.ndarray
    xcenter: np.ndarray
    ink: np.ndarray
    covered: np.ndarray
    column: np.ndarray
    split: Optional[float] = None
    pitch: Dict[int, float] = field(default_factory=dict)
    top: Dict[int, float] = field(default_factory=dict)
    bottom: Dict[int, float] = field(default_factory=dict)

    def slots(self, column: int, low: float, high: float) -> int:
        """Line slots of a page strictly between two reading-order keys.

        :param column: The page.
        :param low: Key of the line above (``-inf``: from the page's first written line, that line included).
        :param high: Key of the line below (``inf``: to the page's last written line, that line included).
        :return: Whole line pitches that fit, never negative.
        """
        pitch = self.pitch.get(column)
        if not pitch or column not in self.top:
            return 0
        edge = 0
        if np.isinf(low):
            low, edge = self.top[column], edge + 1
        if np.isinf(high):
            high, edge = self.bottom[column], edge + 1
        return max(0, int(round((high - low) / pitch)) - 1 + edge)

    def free_lines(self, column: int, low: float, high: float, min_letters: int) -> int:
        """Read lines of a page that no text line explains, strictly between two reading-order keys.

        :param column: The page.
        :param low: Lower key (exclusive).
        :param high: Upper key (exclusive).
        :param min_letters: Letters a read line needs to count.
        :return: Number of read lines.
        """
        sel = (np.array(self.stream.line_letters) >= min_letters) & ~self.covered & (self.column == column) & \
              (self.order > low) & (self.order < high)
        return int(sel.sum())


def read_geometry(stream: ReadStream, photo: Optional[Photo], params: AssembleParams
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reading-order key, horizontal centre and ink flag of every line of a read.

    Boxes are in the stored photograph's frame; a read made on the photograph turned by 180 degrees is read in
    that orientation, so both axes are mirrored for it.

    :param stream: The read.
    :param photo: Its photograph (for the size), or None.
    :param params: Thresholds.
    :return: ``(order, xcenter, ink)`` per read line.
    """
    n = len(stream.line_letters)
    order = np.arange(n, dtype=np.float64)                              # no boxes: the read's own line order
    xcenter = np.full(n, np.nan)
    width = float(photo.width) if photo and photo.width else 0.0
    height = float(photo.height) if photo and photo.height else 0.0
    if stream.bboxes is not None and width > 0 and height > 0:
        boxes = np.array(stream.bboxes, dtype=np.float64)
        ok = ~np.isnan(boxes).any(axis=1)
        if ok.any():
            xc = (boxes[:, 0] + boxes[:, 2]) / 2.0 / width
            yc = (boxes[:, 1] + boxes[:, 3]) / 2.0 / height
            if stream.rotation == 180:
                xc, yc = 1.0 - xc, 1.0 - yc
            xcenter = np.where(ok, xc, np.nan)
            if ok.all():
                order = yc
    conf = np.array([1.0 if c is None else c for c in stream.line_conf], dtype=np.float64)
    ink = (np.array(stream.line_letters) >= params.ink_letters) & (conf >= params.ink_conf)
    return order, xcenter, ink


def page_columns(xcenters: Sequence[float], params: AssembleParams) -> Optional[float]:
    """Whether matched read lines lie on two pages side by side.

    :param xcenters: Horizontal centres (share of the width) of the read lines that supported text lines match.
    :param params: Thresholds.
    :return: The width share that separates the pages (lines right of it are the first page), or None for one page.
    """
    xs = np.sort(np.array([x for x in xcenters if not np.isnan(x)], dtype=np.float64))
    if len(xs) < 2 * params.column_min_lines:
        return None
    gaps = np.diff(xs)
    k = int(np.argmax(gaps))
    if gaps[k] < params.column_gap or k + 1 < params.column_min_lines or len(xs) - k - 1 < params.column_min_lines:
        return None
    return float((xs[k] + xs[k + 1]) / 2.0)


@dataclass
class Assignment:
    """Where every line of a candidate stands.

    :ivar photo: Photograph index per line after the dynamic programme (index into the evaluation list).
    :ivar supported: Whether the line is supported on its photograph.
    :ivar anchor: Read line (of the photograph's primary read) the line matches most, or -1.
    :ivar unit: Proposed page per line as ``(photograph index, column)``, or None for a line put on no page.
    :ivar states: Read-side facts per photograph index.
    :ivar z: ``(M, N)`` best z of every line on every photograph.
    :ivar span_end: End of each supported line's match in its photograph's primary read stream (NaN: none).
    :ivar match_share: Share of each supported line's letters that its read lines match (NaN: none).
    :ivar unmatched_end: Larger of the unmatched shares at the start and at the end of each supported line (NaN: none).
    """

    photo: np.ndarray
    supported: np.ndarray
    anchor: np.ndarray
    unit: List[Optional[Tuple[int, int]]]
    states: Dict[int, PhotoState]
    z: np.ndarray
    span_end: np.ndarray
    match_share: np.ndarray
    unmatched_end: np.ndarray


def _unmatched_end(skel: str, stream: ReadStream, start: float, end: float) -> float:
    """Share of a text line that lies outside its match with the read, at the line's start or end.

    :param skel: The line's skeleton.
    :param stream: The read.
    :param start: Start of the line's match window in the read stream.
    :param end: End of the window.
    :return: The larger of the two shares (NaN when nothing matches); blocks of fewer than four letters are ignored.
    """
    if np.isnan(start) or np.isnan(end) or not skel:
        return float("nan")
    s0 = max(0, int(start))
    window = stream.mstream[s0:min(stream.letters, max(s0 + 1, int(end)))]
    blocks = [(op.src_start, op.src_end) for op in Indel.opcodes(textnorm.match_form(skel), window)
              if op.tag == "equal" and op.src_end - op.src_start >= 4]
    if not blocks:
        return float("nan")
    return max(blocks[0][0], len(skel) - blocks[-1][1]) / len(skel)


def assign_lines(seq: LineSeq, evals: Sequence[PhotoEval], group: Group, params: AssembleParams) -> Assignment:
    """Assign every line a photograph and collect the read-side facts of each photograph.

    :param seq: The candidate's lines.
    :param evals: Its scores per photograph.
    :param group: The document group (photograph sizes).
    :param params: Thresholds.
    :return: The assignment before the unsupported lines are settled (``unit`` is filled for supported lines only).
    """
    n = len(seq.skels)
    best = np.vstack([ev.best() for ev in evals])                       # (M, N)
    weights = np.clip(np.nan_to_num(best, nan=0.0), params.z_floor, params.z_cap).T
    path = viterbi(weights, params.switch)
    supported = np.zeros(n, dtype=bool)
    anchor = np.full(n, -1, dtype=np.int64)
    span_end = np.full(n, np.nan)
    match_share = np.full(n, np.nan)
    unmatched_end = np.full(n, np.nan)
    states: Dict[int, PhotoState] = {}
    unit: List[Optional[Tuple[int, int]]] = [None] * n
    for m, ev in enumerate(evals):
        mine = np.where(path == m)[0]
        if not len(mine):
            continue
        gain = np.clip(np.nan_to_num(ev.z[:, mine], nan=0.0), 0.0, params.z_cap).sum(axis=1)
        boxed = np.array([st.bboxes is not None for st in ev.streams])
        s = int(np.argmax(gain + 1e-6 * boxed))                         # ties: a read with boxes
        stream = ev.streams[s]
        order, xcenter, ink = read_geometry(stream, group.photos.get(ev.sha), params)
        covered = np.zeros(len(stream.line_letters), dtype=bool)
        for i in mine:
            z_any = best[m, i]
            if np.isnan(z_any) or z_any < params.z_sup or np.nanmax(ev.score[:, i]) < params.s_sup:
                continue
            supported[i] = True
            if np.isnan(ev.z[s, i]) or ev.z[s, i] < params.z_sup / 2:
                continue                                                # supported by another read: no geometry
            matched = align.matched_read_lines(seq.skels[i], stream, ev.start[s, i], ev.end[s, i])
            if not matched:
                continue
            anchor[i] = max(matched, key=lambda j: matched[j])
            span_end[i] = ev.end[s, i]
            match_share[i] = sum(matched.values()) / max(1, len(seq.skels[i]))
            unmatched_end[i] = _unmatched_end(seq.skels[i], stream, ev.start[s, i], ev.end[s, i])
            for j, got in matched.items():
                if got >= params.cover_match_min and got >= params.cover_match_share * stream.line_letters[j]:
                    covered[j] = True
        split = page_columns([xcenter[anchor[i]] for i in mine if anchor[i] >= 0], params)
        column = np.zeros(len(stream.line_letters), dtype=np.int64)
        if split is not None:
            column = np.where(np.nan_to_num(xcenter, nan=1.0) < split, 1, 0)
        state = PhotoState(ev.sha, stream, stream.rotation, order, xcenter, ink, covered, column, split)
        written = np.array(stream.line_letters) >= params.read_line_letters
        for col in sorted(set(column.tolist())):
            keys = np.sort(order[written & (column == col)])
            if not len(keys):
                continue
            state.top[col], state.bottom[col] = float(keys[0]), float(keys[-1])
            steps = [order[anchor[j]] - order[anchor[i]] for i, j in zip(mine[:-1], mine[1:])
                     if j == i + 1 and anchor[i] >= 0 and anchor[j] >= 0 and anchor[i] != anchor[j]
                     and column[anchor[i]] == col and column[anchor[j]] == col]
            steps = [d for d in steps if d > 0]
            if len(steps) < 3:                                          # few anchors: the read's own spacing
                steps = [d for d in np.diff(keys).tolist() if d > 0]
            if steps:
                state.pitch[col] = float(np.median(steps))
        states[m] = state
        for i in mine:
            if supported[i]:
                unit[i] = (m, int(column[anchor[i]]) if anchor[i] >= 0 else -1)
    # a supported line without geometry takes the page of the nearest supported line of its photograph that has one
    for i in range(n):
        if unit[i] is not None and unit[i][1] < 0:
            m = unit[i][0]
            near = [j for j in range(n) if unit[j] is not None and unit[j][0] == m and unit[j][1] >= 0]
            unit[i] = (m, unit[min(near, key=lambda j: abs(j - i))][1] if near else 0)
    return Assignment(path, supported, anchor, unit, states, best, span_end, match_share, unmatched_end)


_LATIN_OR_DIGIT = re.compile(r"[A-Za-z0-9]")


def _evident_run(z: Sequence[float], need: Sequence[bool], weak_z: float) -> int:
    """Length of the longest prefix of a stretch whose lines all show weak evidence of standing on a page.

    A line long enough to carry evidence must reach ``weak_z``; a shorter line counts only when a line with
    evidence follows it inside the prefix (it cannot end the run).

    :param z: z of the stretch's lines on the page, nearest line first.
    :param need: Whether each line is long enough to carry evidence.
    :param weak_z: The weak-evidence z.
    :return: Number of lines.
    """
    last = 0
    for k, (value, needs) in enumerate(zip(z, need)):
        if needs:
            if np.isnan(value) or value < weak_z:
                break
            last = k + 1
    return last


def _split_point(zl: Sequence[float], zr: Sequence[float], max_left: int, max_right: int,
                 need: Optional[Sequence[bool]] = None, weak_z: Optional[float] = None) -> Tuple[int, int]:
    """How many lines of a stretch at the end of a page go to the page before and the page after.

    The first ``k_left`` lines go to the earlier page, the last ``k_right`` to the later one and the middle is
    dropped; each side is capped by its room and, when ``weak_z`` is given, by the lines that show weak evidence
    of standing there (:func:`_evident_run`). Among the splits that place the most lines the evidence decides.

    :param zl: z of every line of the stretch on the earlier page (NaN: none).
    :param zr: z of every line on the later page.
    :param max_left: Unexplained read lines at the foot of the earlier page.
    :param max_right: Unexplained read lines at the head of the later page.
    :param need: Whether each line is long enough to carry evidence (all, when None).
    :param weak_z: The weak-evidence z, or None for no evidence requirement.
    :return: ``(k_left, k_right)``.
    """
    n = len(zl)
    left = np.nan_to_num(np.array(zl, dtype=np.float64), nan=0.0)
    right = np.nan_to_num(np.array(zr, dtype=np.float64), nan=0.0)
    if weak_z is not None:
        needs = list(need) if need is not None else [True] * n
        max_left = min(max_left, _evident_run(list(zl), needs, weak_z))
        max_right = min(max_right, _evident_run(list(zr)[::-1], needs[::-1], weak_z))
    best: Tuple[int, float, int, int] = (-1, 0.0, 0, 0)
    for k_left in range(0, min(n, max_left) + 1):
        k_right = min(n - k_left, max_right)
        gain = float(left[:k_left].sum() + right[n - k_right:].sum())
        cand = (k_left + k_right, gain, k_left, k_right)
        if cand[:2] > best[:2]:
            best = cand
    return best[2], best[3]


def _is_tail(seq: LineSeq, a: Assignment, left: int, stretch: Sequence[int], params: AssembleParams) -> bool:
    """Whether a single short unsupported line is the wrapped end of the supported line before it.

    Editions wrap the last word of a long manuscript line onto a typeset line of its own. Such a tail has no line
    slot of its own on the page; it is on the photograph when the read goes on with it right after the match of
    the line before.

    :param seq: The candidate's lines.
    :param a: The assignment.
    :param left: Index of the supported line before the stretch.
    :param stretch: The unsupported lines.
    :param params: Thresholds.
    :return: True when the stretch is one short line the read continues with.
    """
    if len(stretch) != 1 or np.isnan(a.span_end[left]):
        return False
    skel = seq.skels[stretch[0]]
    if not align.MIN_LINE_LETTERS <= len(skel) <= params.tail_letters:
        return False
    stream = a.states[a.unit[left][0]].stream
    start = int(a.span_end[left])
    window = stream.mstream[max(0, start - 2):start + 2 * len(skel) + 4]
    return bool(window) and fuzz.partial_ratio(textnorm.match_form(skel), window) >= params.tail_score


def settle_unsupported(seq: LineSeq, a: Assignment, params: AssembleParams) -> None:
    """Propose a page for every unsupported line, or none, by the room its neighbours' pages have for it.

    See the module docstring (step 4). Only lines long enough to carry evidence need room; shorter lines follow
    their stretch. ``a.unit`` is completed in place; a line left without a page is unresolved, not found absent
    (:func:`line_status`).

    :param seq: The candidate's lines.
    :param a: The assignment (supported lines already placed).
    :param params: Thresholds.
    """
    n = len(seq.skels)
    need = seq.letters >= align.MIN_LINE_LETTERS
    editorial = [bool(fgp_notes.soft_marker(line)) or bool(_LATIN_OR_DIGIT.search(line)) for line in seq.lines]

    def key_of(i: int) -> float:
        """Reading-order key of a supported line's anchor (NaN without one).

        :param i: Line index.
        :return: The key.
        """
        m = a.unit[i][0]
        return float(a.states[m].order[a.anchor[i]]) if a.anchor[i] >= 0 else float("nan")

    i = 0
    while i < n:
        if a.supported[i]:
            i += 1
            continue
        j = i
        while j < n and not a.supported[j]:
            j += 1
        stretch = list(range(i, j))
        wanting = int(sum(1 for k in stretch if need[k] and not editorial[k]))
        left = i - 1 if i > 0 else None
        right = j if j < n else None
        ul = a.unit[left] if left is not None else None
        ur = a.unit[right] if right is not None else None
        keep: List[Optional[Tuple[int, int]]] = [None] * len(stretch)
        if ul is not None and ur is not None and ul == ur:
            state = a.states[ul[0]]
            kl, kr = key_of(left), key_of(right)
            slots = 0 if np.isnan(kl) or np.isnan(kr) else state.slots(ul[1], min(kl, kr), max(kl, kr))
            if wanting == 0 or slots >= params.room_factor * wanting or _is_tail(seq, a, left, stretch, params):
                keep = [ul] * len(stretch)
            elif slots > 0:                                             # room for some: the best weakly supported
                z = np.nan_to_num(a.z[ul[0], i:j], nan=-9.0)
                ranked = [k for k in np.argsort(-z) if z[k] >= params.weak_z][:slots]
                for k in ranked:
                    keep[k] = ul
        else:
            room_l = room_r = 0
            if ul is not None:
                kl = key_of(left)
                room_l = 0 if np.isnan(kl) else a.states[ul[0]].free_lines(ul[1], kl, np.inf, params.read_line_letters)
            if ur is not None:
                kr = key_of(right)
                room_r = 0 if np.isnan(kr) else a.states[ur[0]].free_lines(ur[1], -np.inf, kr,
                                                                           params.read_line_letters)
            zl = a.z[ul[0], i:j] if ul is not None else np.full(len(stretch), np.nan)
            zr = a.z[ur[0], i:j] if ur is not None else np.full(len(stretch), np.nan)
            k_left, k_right = _split_point(zl, zr, room_l, room_r, [bool(x) for x in need[i:j]], params.weak_z)
            for k in range(k_left):
                keep[k] = ul
            for k in range(len(stretch) - k_right, len(stretch)):
                keep[k] = ur
        for k, idx in enumerate(stretch):
            a.unit[idx] = None if editorial[idx] else keep[k]
        i = j


def drop_stray_runs(seq: LineSeq, a: Assignment, params: AssembleParams) -> None:
    """Take runs that rest on one weakly supported line off their page.

    A run is a stretch of consecutive lines on one page. One supported line keeps a run only when its z reaches
    :attr:`AssembleParams.stray_z` and it has :attr:`AssembleParams.stray_letters` letters: a formula or a quoted
    verse matches a wrong page that well, a run of lines does not. The lines are left unresolved
    (:func:`line_status`).

    :param seq: The candidate's lines.
    :param a: The settled assignment (changed in place).
    :param params: Thresholds.
    """
    n = len(seq.skels)
    letters = seq.letters
    i = 0
    while i < n:
        if a.unit[i] is None:
            i += 1
            continue
        j = i
        while j < n and a.unit[j] == a.unit[i]:
            j += 1
        sup = [k for k in range(i, j) if a.supported[k]]
        m = a.unit[i][0]
        strong = [k for k in sup if a.z[m, k] >= params.stray_z and letters[k] >= params.stray_letters]
        if len(sup) < 2 and not strong:
            for k in range(i, j):
                a.unit[k] = None
        i = j


def crop_to_text(state: PhotoState, photo: Optional[Photo], edge_lines: int, unread_letters: int,
                 params: AssembleParams) -> Optional[Tuple[List[int], float]]:
    """A crop around the explained read lines when the unexplained ink lies clearly apart from them.

    The box is the union of the explained lines' boxes, padded by up to one line height but never more than 0.4
    of the gap to the nearest unexplained ink line on that side. No crop is made when unread lines were kept at
    the ends of the text (they need room the boxes do not show), when unexplained ink comes closer than
    :attr:`AssembleParams.crop_gap` line heights to the explained block, when a crop would cut an unexplained
    line's box by more than :attr:`AssembleParams.crop_overlap_max`, or when the ink left inside exceeds
    :attr:`AssembleParams.uncovered_max`.

    :param state: The photograph's read-side facts (boxes in the stored photograph's frame).
    :param photo: The photograph (bounds).
    :param edge_lines: Unread lines kept at the ends of the text's runs.
    :param unread_letters: Letters of all unread lines kept.
    :param params: Thresholds.
    :return: ``(crop box [x0, y0, x1, y1] in stored pixels, unexplained share inside it)`` or None.
    """
    if state.stream.bboxes is None or edge_lines:
        return None
    arr = np.array(state.stream.bboxes, dtype=np.float64)
    valid = ~np.isnan(arr).any(axis=1)
    cov = state.covered & valid
    free = state.ink & ~state.covered & valid
    if cov.sum() < 2 or not free.any():
        return None
    line_h = max(1.0, float(np.median(arr[cov, 3] - arr[cov, 1])))
    width = float(photo.width) if photo and photo.width else float(arr[valid, 2].max() + 2 * line_h)
    height = float(photo.height) if photo and photo.height else float(arr[valid, 3].max() + 2 * line_h)
    x0, y0 = float(arr[cov, 0].min()), float(arr[cov, 1].min())
    x1, y1 = float(arr[cov, 2].max()), float(arr[cov, 3].max())
    over_x = (arr[:, 2] > x0) & (arr[:, 0] < x1)                         # shares columns with the block
    over_y = (arr[:, 3] > y0) & (arr[:, 1] < y1)                         # shares rows with the block
    gaps = {"top": y0 - arr[free & over_x & (arr[:, 3] <= y0 + 0.3 * line_h), 3],
            "bottom": arr[free & over_x & (arr[:, 1] >= y1 - 0.3 * line_h), 1] - y1,
            "left": x0 - arr[free & over_y & (arr[:, 2] <= x0 + 0.3 * line_h), 2],
            "right": arr[free & over_y & (arr[:, 0] >= x1 - 0.3 * line_h), 0] - x1}
    pad: Dict[str, float] = {}
    for side, values in gaps.items():
        gap = float(values.min()) if len(values) else float("inf")
        if gap < params.crop_gap * line_h:
            return None                                                 # unexplained ink right next to the block
        pad[side] = min(line_h, 0.4 * gap)
    box = [max(0.0, x0 - pad["left"]), max(0.0, y0 - pad["top"]), min(width, x1 + pad["right"]),
           min(height, y1 + pad["bottom"])]
    if (box[2] - box[0]) * (box[3] - box[1]) < params.crop_min_area * width * height:
        return None
    cx, cy = (arr[:, 0] + arr[:, 2]) / 2.0, (arr[:, 1] + arr[:, 3]) / 2.0
    within = valid & (cx >= box[0]) & (cx <= box[2]) & (cy >= box[1]) & (cy <= box[3])
    letters = np.array(state.stream.line_letters)
    if not int(letters[free & ~within].sum()):
        return None
    for j in np.where(free & ~within)[0]:
        ox = max(0.0, min(arr[j, 2], box[2]) - max(arr[j, 0], box[0]))
        oy = max(0.0, min(arr[j, 3], box[3]) - max(arr[j, 1], box[1]))
        if ox * oy > params.crop_overlap_max * max(1.0, (arr[j, 2] - arr[j, 0]) * (arr[j, 3] - arr[j, 1])):
            return None
    ink_in = int(letters[state.ink & within].sum())
    share = max(0.0, float(letters[free & within].sum()) - params.slack * unread_letters) / ink_in if ink_in else 0.0
    if share > params.uncovered_max:
        return None
    return [int(v) for v in box], share


def page_crops(state: PhotoState, photo: Optional[Photo], params: AssembleParams) -> Optional[Dict[int, List[int]]]:
    """Crop boxes of the two pages of a photograph, cut in the gutter between their explained lines.

    :param state: The photograph's read-side facts.
    :param photo: The photograph (its size).
    :param params: Thresholds.
    :return: ``{page: [x0, y0, x1, y1]}`` in stored pixels (page 0 = the right page as read), or None when the
        photograph shows one page, has no boxes or size, or the pages' lines leave no clear gutter.
    """
    if state.split is None or state.stream.bboxes is None or not photo or not photo.width or not photo.height:
        return None
    arr = np.array(state.stream.bboxes, dtype=np.float64)
    ok = state.covered & ~np.isnan(arr).any(axis=1)
    width, height = float(photo.width), float(photo.height)
    left_edge, right_edge = arr[:, 0] / width, arr[:, 2] / width         # as read: mirrored for a 180-degree read
    if state.rotation == 180:
        left_edge, right_edge = 1.0 - arr[:, 2] / width, 1.0 - arr[:, 0] / width
    first, second = ok & (state.column == 0), ok & (state.column == 1)   # right page, left page
    if first.sum() < params.column_min_lines or second.sum() < params.column_min_lines:
        return None
    clear = float(left_edge[first].min()) - float(right_edge[second].max())
    if clear < params.gutter_min:
        return None
    gutter = float(right_edge[second].max()) + clear / 2.0
    cut = int(round((1.0 - gutter if state.rotation == 180 else gutter) * width))
    low, high = [0, 0, cut, int(height)], [cut, 0, int(width), int(height)]
    return {0: low, 1: high} if state.rotation == 180 else {0: high, 1: low}


def assemble_photo(seq: LineSeq, a: Assignment, m: int, group: Group, params: AssembleParams,
                   page: Optional[int] = None, page_crop: Optional[List[int]] = None) -> Optional[Dict[str, Any]]:
    """The pair record of one photograph (or of one of its two pages) for one candidate.

    :param seq: The candidate's lines.
    :param a: The settled assignment.
    :param m: Photograph index.
    :param group: The document group.
    :param params: Thresholds.
    :param page: Build the record of this page only (0 = the right page), with ``page_crop`` as its crop.
    :param page_crop: Crop box of that page (:func:`page_crops`).
    :return: The record, or None when no line stands on the photograph (or page).
    """
    state = a.states.get(m)
    mine = [i for i, u in enumerate(a.unit) if u is not None and u[0] == m and (page is None or u[1] == page)]
    if state is None or not mine:
        return None
    runs: List[List[int]] = []
    for i in mine:
        if runs and runs[-1][-1] == i - 1 and a.unit[runs[-1][-1]] == a.unit[i]:
            runs[-1].append(i)
        else:
            runs.append([i])

    def run_key(run: List[int]) -> Tuple[int, float, int]:
        """Sort key of a run: its page, then its height on the page, then edition order.

        :param run: Line indices.
        :return: The key.
        """
        keys = [state.order[a.anchor[i]] for i in run if a.anchor[i] >= 0]
        return (a.unit[run[0]][1], float(np.median(keys)) if keys else float("inf"), run[0])

    runs.sort(key=run_key)
    seen: Dict[int, List[int]] = defaultdict(list)                       # read line -> text lines kept on it
    doubled: Dict[int, int] = {}                                         # line set twice -> the line kept for it
    for run in runs:
        for i in run:
            if not a.supported[i] or a.anchor[i] < 0:
                continue
            twin = next((j for j in seen[int(a.anchor[i])]
                         if fuzz.ratio(seq.skels[i], seq.skels[j]) >= params.dup_ratio), None)
            if twin is None:
                seen[int(a.anchor[i])].append(i)
            else:
                doubled[i] = twin                                       # the edition sets this line twice
    runs = [[i for i in run if i not in doubled] for run in runs]
    runs = [run for run in runs if run]
    order = [i for run in runs for i in run]
    if not order:
        return None
    letters = seq.letters
    total = int(letters[order].sum())
    sup = int(sum(letters[i] for i in order if a.supported[i]))
    edge = set()
    for run in runs:                                                    # unread lines at the ends of a run
        for seq_run in (run, run[::-1]):
            for i in seq_run:
                if a.supported[i]:
                    break
                edge.add(i)
    interior = int(sum(letters[i] for i in order if not a.supported[i] and i not in edge))
    unread = int(sum(letters[i] for i in order if not a.supported[i]))
    on_page = np.ones(len(state.column), dtype=bool) if page is None else state.column == page
    ink_total = int(np.array(state.stream.line_letters)[state.ink & on_page].sum())
    free = int(np.array(state.stream.line_letters)[state.ink & ~state.covered & on_page].sum())
    uncovered = max(0.0, free - params.slack * unread) / ink_total if ink_total else 0.0
    support_share = sup / total if total else 0.0
    interior_share = interior / total if total else 0.0
    reasons: List[str] = []
    need = params.support_min if state.stream.family == "kraken" else params.support_min_vlm
    if len(order) < params.min_lines or total < params.min_letters:
        reasons.append("too_few_letters")
    if support_share < need:
        reasons.append("support_share_low")
    if interior_share > params.interior_max:
        reasons.append("interior_unread_high")
    ends = int(sum(letters[i] for i in edge))
    if total and ends > params.unread_ends_max * total:
        reasons.append("unread_ends")
    elif total and unread > params.unread_max * total:
        reasons.append("unread_lines")
    shares = [a.match_share[i] for i in order if a.supported[i] and not np.isnan(a.match_share[i])]
    match_share = float(np.median(shares)) if shares else float("nan")
    gaps = [a.unmatched_end[i] for i in order if a.supported[i] and not np.isnan(a.unmatched_end[i])]
    end_gap = float(np.median(gaps)) if gaps else float("nan")
    if gaps and end_gap > params.unmatched_end_max:
        reasons.append("lines_partly_matched")
    photo = group.photos.get(state.sha)
    crop: Optional[List[int]] = list(page_crop) if page_crop else None
    uncovered_photo = uncovered
    if uncovered > params.uncovered_max:
        cropped = crop_to_text(state, photo, len(edge), unread, params) if page is None else None
        if cropped is None:
            reasons.append("uncovered_ink")
        else:
            crop, uncovered = cropped
    sides: List[str] = []
    for i in order:
        if seq.refs[i][0] not in sides:
            sides.append(seq.refs[i][0])
    flags = (["two_pages"] if state.split is not None and page is None else []) + \
            (["several_runs"] if len(runs) > 1 else []) + \
            (["doubled_lines_dropped"] if doubled else []) + (["unread_lines_kept"] if unread else [])
    tag = "" if page is None else f"|{'rl'[page]}"
    return {"_order": order, "_doubled": doubled,                       # taken off again by evaluate_group
            "pair_id": f"{seq.row_id}|asm|{state.sha[:12]}{tag}", "row_id": seq.row_id, "side_key": sides[0],
            "side_keys": sides, "source": seq.source, "document_group": group.document_group, "script": seq.script,
            "ja_signal": None, "tier": None, "image_sha256": state.sha, "image_path": photo.path if photo else None,
            "image_label": photo.label if photo else None, "duplicate_photos": [],
            "decision": "review" if reasons else "accept",
            "remedies": ["assemble"] + (["split"] if page is not None else ["crop"] if crop else []) +
                        (["rotate180"] if state.rotation == 180 else []),
            "line_indices": [seq.refs[i][1] for i in order], "line_refs": [list(seq.refs[i]) for i in order],
            "lines": [seq.lines[i] for i in order], "trimmed_lines": {}, "moved_to": {},
            "unread_lines": [k for k, i in enumerate(order, 1) if not a.supported[i]],   # numbers of ``lines``
            "rotation": 180 if state.rotation == 180 else 0, "crop": crop,
            "evidence": {"reader": state.stream.reader, "rotation_read": state.rotation, "target_letters": total,
                         "support_share": round(support_share, 4), "interior_share": round(interior_share, 4),
                         "uncovered_share": round(uncovered, 4), "uncovered_share_photo": round(uncovered_photo, 4),
                         "chain_lines": int(sum(a.supported[i] for i in order)),
                         "chain_q": None, "q_null_max": None, "k_chain": None, "k_null_max": None,
                         "runs": [[seq.refs[r[0]][0], seq.refs[r[0]][1], seq.refs[r[-1]][0], seq.refs[r[-1]][1]]
                                  for r in runs],
                         "pages": 2 if state.split is not None and page is None else 1,
                         "page": None if page is None else ("right", "left")[page],
                         "unread_letters": unread, "unread_end_letters": ends, "doubled_lines": len(doubled),
                         "match_share": None if np.isnan(match_share) else round(match_share, 3),
                         "unmatched_end": None if np.isnan(end_gap) else round(end_gap, 3),
                         "ink_letters": ink_total, "free_ink_letters": free},
            "dup_rank": None, "flags": flags, "reasons": reasons}


#: Line statuses that leave a line unresolved: it is in no pair's text and nothing but a reader's silence (or an
#: ambiguous match) put it there.
UNRESOLVED = ("open", "doubled")
#: Line statuses of lines in a pair's text.
IN_TEXT = ("read", "kept_unread")


def line_status(seq: LineSeq, a: Assignment, made: Sequence[Dict[str, Any]]
                ) -> Tuple[List[str], List[Optional[str]]]:
    """Account for every line of a candidate once its pairs are built.

    Statuses: ``read`` (in a pair's text, supported by a read), ``kept_unread`` (in a pair's text by the room the
    page has for it), ``note`` (in no text: an editor's label or note by its own wording, and no read supports
    it), ``no_letters`` (in no text: a line without a letter, such as a lone lacuna mark), ``doubled`` (in no
    text: it matches a read line that already has its text; whether the edition sets the line twice or the
    manuscript repeats it, a refrain or a whole copied block, the alignment cannot tell) and ``open`` (in no text
    for lack of room or evidence). The last two are unresolved (:data:`UNRESOLVED`).

    :param seq: The candidate's lines.
    :param a: The settled assignment.
    :param made: The candidate's pair records, each still carrying ``_order`` and ``_doubled``.
    :return: ``(status per line, id of the pair holding the line or None)``.
    :raises ValueError: When a pair holds a line twice or two pairs hold one line.
    """
    n = len(seq.lines)
    owner: List[Optional[str]] = [None] * n
    twin: Dict[int, int] = {}
    for rec in made:
        twin.update(rec["_doubled"])
        for i in rec["_order"]:
            if owner[i] is not None:
                raise ValueError(f"line {seq.refs[i]} of {seq.row_id} stands in {owner[i]} and {rec['pair_id']}")
            owner[i] = rec["pair_id"]
    status: List[str] = []
    for i in range(n):
        if owner[i] is not None:
            status.append("read" if a.supported[i] else "kept_unread")
        elif i in twin:
            status.append("doubled")
        elif not seq.skels[i]:
            status.append("no_letters")
        elif not a.supported[i] and fgp_notes.classify_line(seq.lines[i]):
            status.append("note")
        else:
            status.append("open")
    return status, owner


def attach_open_lines(seq: LineSeq, status: Sequence[str], owner: Sequence[Optional[str]],
                      made: Sequence[Dict[str, Any]]) -> None:
    """Attach every stretch of unresolved lines to the pairs whose text it borders, and hold those pairs.

    A stretch is a maximal run of consecutive lines (edition order) that stand in no pair's text and hold at least
    one unresolved line (:data:`UNRESOLVED`). It borders the pair that holds the line before it and the pair that
    holds the line after it (one pair when both are the same). Each of them gets an entry in ``open_lines``::

        {"after_line": number of the pair's line the stretch follows (1-based, None: it precedes the text),
         "before_line": number of the pair's line the stretch precedes (None: it follows the text),
         "refs": [[side key, line index], ...], "lines": [...], "status": [...]   # the unresolved lines only
         "letters": their letters, "stretch": [first, end) in the candidate's sequence,
         "left_out": lines of the stretch left out on the document's own evidence (notes, lone lacuna marks),
         "other_pair": the pair on the stretch's other side, or None}

    A pair with an entry gets the reason ``unresolved_lines`` and the decision ``review``: its text may be short
    of lines the photograph shows, and only a look at the photograph settles that. ``evidence.lines_open`` and
    ``evidence.letters_open`` count the pair's own entries (a stretch between two pairs is counted for both, so
    these never add up over pairs; :func:`evaluate_group`'s line records count every line once).

    :param seq: The candidate's lines.
    :param status: Status per line (:func:`line_status`).
    :param owner: Pair per line.
    :param made: The candidate's pair records (changed in place).
    """
    n = len(seq.lines)
    by_id = {rec["pair_id"]: rec for rec in made}
    number = {rec["pair_id"]: {i: k for k, i in enumerate(rec["_order"], 1)} for rec in made}
    letters = seq.letters
    for rec in made:
        rec["open_lines"] = []
    i = 0
    while i < n:
        if owner[i] is not None:
            i += 1
            continue
        j = i
        while j < n and owner[j] is None:
            j += 1
        unresolved = [k for k in range(i, j) if status[k] in UNRESOLVED]
        before = owner[i - 1] if i > 0 else None
        after = owner[j] if j < n else None
        if unresolved:
            entry = {"refs": [list(seq.refs[k]) for k in unresolved], "lines": [seq.lines[k] for k in unresolved],
                     "status": [status[k] for k in unresolved], "letters": int(letters[unresolved].sum()),
                     "stretch": [i, j], "left_out": j - i - len(unresolved)}
            if before is not None:
                by_id[before]["open_lines"].append({
                    "after_line": number[before][i - 1],
                    "before_line": number[before][j] if after == before else None,
                    **entry, "other_pair": None if after == before else after})
            if after is not None and after != before:
                by_id[after]["open_lines"].append({"after_line": None, "before_line": number[after][j], **entry,
                                                   "other_pair": before})
        i = j
    for rec in made:
        entries = rec["open_lines"]
        rec["evidence"]["lines_open"] = sum(len(e["lines"]) for e in entries)
        rec["evidence"]["letters_open"] = sum(e["letters"] for e in entries)
        if entries:
            rec["reasons"] = list(rec["reasons"]) + ["unresolved_lines"]
            rec["decision"] = "review"
            rec["flags"] = list(rec["flags"]) + ["open_lines"]


def evaluate_group(group: Group, pool: NullPool, params: Optional[AssembleParams] = None
                   ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Assemble every candidate of a document group.

    :param group: The group.
    :param pool: Null reads.
    :param params: Thresholds (defaults when None).
    :return: ``(pair records, per-line audit records)``; a line record carries the line's ``status``
        (:func:`line_status`) and the ``pair_id`` of the pair whose text holds it (None: in no text).
    """
    params = params or AssembleParams()
    pairs: List[Dict[str, Any]] = []
    lines: List[Dict[str, Any]] = []
    for seq in line_sequences(group):
        evals = score_candidate(seq, group, pool)
        if not evals:
            pairs.append(_unplaced(seq, group, "no_usable_read" if group.photos else "no_photograph"))
            continue
        a = assign_lines(seq, evals, group, params)
        settle_unsupported(seq, a, params)
        drop_stray_runs(seq, a, params)
        made = []
        for m in sorted(a.states):
            crops = page_crops(a.states[m], group.photos.get(a.states[m].sha), params)
            if crops:                                                   # an opening with a clear gutter: per page
                recs = [assemble_photo(seq, a, m, group, params, page, crops[page]) for page in (0, 1)]
            else:
                recs = [assemble_photo(seq, a, m, group, params)]
            made.extend(rec for rec in recs if rec)
        status, owner = line_status(seq, a, made)
        attach_open_lines(seq, status, owner, made)
        for rec in made:
            del rec["_order"], rec["_doubled"]
        if not made:
            pairs.append(_unplaced(seq, group, "no_supported_line"))
        pairs.extend(made)
        best = np.vstack([ev.best() for ev in evals])
        for i, (side_key, idx) in enumerate(seq.refs):
            m = int(np.nanargmax(np.where(np.isnan(best[:, i]), -np.inf, best[:, i]))) if not np.isnan(best[:, i]).all() else -1
            lines.append({"sid": f"{seq.row_id}|{side_key}", "line": idx, "letters": len(seq.skels[i]),
                          "face": evals[a.unit[i][0]].sha if a.unit[i] is not None else None,
                          "best_z": round(float(best[m, i]), 2) if m >= 0 else None,
                          "best_sha256": evals[m].sha if m >= 0 else None,
                          "best_reader": None, "best_score": None, "pos": None,
                          "supported": bool(a.supported[i]), "status": status[i], "pair_id": owner[i]})
    by_photo: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in pairs:
        if p.get("image_sha256"):                                       # the two pages of an opening are two places
            by_photo[f"{p['image_sha256']}|{p['evidence'].get('page')}"].append(p)
    for recs in by_photo.values():                                      # editions sharing a photograph: best first
        recs.sort(key=lambda p: (p["decision"] != "accept", -p["evidence"]["target_letters"], p["pair_id"]))
        if len(recs) > 1:
            for rank, p in enumerate(recs):
                p["dup_rank"] = rank
    return pairs, lines


def _unplaced(seq: LineSeq, group: Group, reason: str) -> Dict[str, Any]:
    """Record of a candidate none of whose lines stands on a photograph.

    :param seq: The candidate's lines.
    :param group: Its group.
    :param reason: Why.
    :return: A reject record.
    """
    sides: List[str] = []
    for side_key, _ in seq.refs:
        if side_key not in sides:
            sides.append(side_key)
    return {"pair_id": f"{seq.row_id}|asm|none", "row_id": seq.row_id, "side_key": sides[0], "side_keys": sides,
            "source": seq.source, "document_group": group.document_group, "script": seq.script, "ja_signal": None,
            "tier": None, "image_sha256": None, "image_path": None, "image_label": None, "duplicate_photos": [],
            "decision": "reject", "remedies": [], "line_indices": [], "line_refs": [], "lines": [],
            "trimmed_lines": {}, "moved_to": {}, "rotation": 0, "crop": None,
            "evidence": {"target_letters": int(seq.letters.sum())}, "dup_rank": None, "flags": [], "reasons": [reason]}


_POOL: Optional[NullPool] = None
_PARAMS: Optional[AssembleParams] = None


def _init(pool: NullPool, params: AssembleParams) -> None:
    """Worker initialiser (fork: the pool is inherited).

    :param pool: Null pool.
    :param params: Thresholds.
    """
    global _POOL, _PARAMS
    _POOL, _PARAMS = pool, params


def _run_one(group: Group) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Assemble one group in a worker.

    :param group: The group.
    :return: Pair records and line records.
    """
    assert _POOL is not None and _PARAMS is not None
    return evaluate_group(group, _POOL, _PARAMS) if group.sides else ([], [])


def main(argv: Optional[List[str]] = None) -> None:
    """Run the assembly over candidate files and cached reads.

    :param argv: Arguments (None: sys.argv).
    """
    from src.finetuning.qwen_hebrew.ja_ar_build import run_gate

    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["edition-lines"]:
        from src.finetuning.qwen_hebrew.ja_ar_build import edition_assembly

        edition_assembly.main(argv[1:])
        return

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 epilog="For the edition-led PDF dry run, use: assemble edition-lines --help")
    ap.add_argument("--candidates", type=Path, nargs="+", required=True, help="candidate JSONL files")
    ap.add_argument("--reads-dir", type=Path, nargs="+", required=True, help="work dirs with reads/reads.jsonl")
    ap.add_argument("--null-candidates", type=Path, nargs="*", default=None,
                    help="candidate files whose photographs feed the null (default: --candidates)")
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-groups", type=int, default=None, help="first N groups only (smoke runs)")
    ap.add_argument("--only-groups", nargs="*", default=None, help="document groups to run (substring match)")
    ap.add_argument("--params", type=Path, default=None, help="JSON file overriding AssembleParams fields")
    args = ap.parse_args(argv)
    t0 = time.time()
    params = AssembleParams()
    if args.params:
        for k, v in json.loads(args.params.read_text()).items():
            setattr(params, k, v)
    cands = run_gate.load_candidates(args.candidates)
    streams = run_gate.load_streams(args.reads_dir)
    null_cands = run_gate.load_candidates(args.null_candidates) if args.null_candidates else cands
    null_shas = {im.get("sha256") for c in null_cands for im in (c.get("images") or []) if im.get("sha256")}
    pool = NullPool.from_streams(st for sha, sts in streams.items() if sha in null_shas for st in sts)
    groups = run_gate.build_groups(cands, streams, {})
    if args.only_groups:
        groups = [g for g in groups if any(s in g.document_group for s in args.only_groups)]
    if args.limit_groups:
        groups = groups[: args.limit_groups]
    print(f"candidates {len(cands)}, groups {len(groups)}, photos with reads {len(streams)}, null keys "
          f"{len(pool.by_key)}, load {time.time() - t0:.0f}s", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    all_pairs: List[Dict[str, Any]] = []
    line_count: Counter = Counter()                                     # every line once, by status
    letter_count: Counter = Counter()
    with open(args.out / "pairs.jsonl", "w", encoding="utf-8") as pf, \
            open(args.out / "line_evidence.jsonl", "w", encoding="utf-8") as lf:

        def emit(pairs: List[Dict[str, Any]], lines: List[Dict[str, Any]]) -> None:
            """Write one group's records and count its lines.

            :param pairs: Pair records.
            :param lines: Line records.
            """
            all_pairs.extend(pairs)
            pf.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs)
            lf.writelines(json.dumps(ln, ensure_ascii=False) + "\n" for ln in lines)
            for ln in lines:
                line_count[ln["status"]] += 1
                letter_count[ln["status"]] += ln["letters"]

        if args.workers <= 1:
            _init(pool, params)
            for pairs, lines in map(_run_one, groups):
                emit(pairs, lines)
        else:
            with mp.get_context("fork").Pool(args.workers, initializer=_init, initargs=(pool, params)) as wp:
                for done, (pairs, lines) in enumerate(wp.imap(_run_one, groups, chunksize=2), 1):
                    emit(pairs, lines)
                    if done % 100 == 0:
                        print(f"groups {done}/{len(groups)} ({time.time() - t0:.0f}s)", flush=True)
    run = {"inputs": {"candidates": [str(p) for p in args.candidates], "reads_dirs": [str(p) for p in args.reads_dir]},
           "params": asdict(params), "groups": len(groups), "pairs": len(all_pairs),
           "decisions": dict(Counter(p["decision"] for p in all_pairs)),
           "lines_by_status": dict(sorted(line_count.items())),
           "letters_by_status": dict(sorted(letter_count.items())),
           "pairs_with_open_lines": sum(1 for p in all_pairs if p.get("open_lines")),
           "accepted_letters": sum(p["evidence"]["target_letters"] for p in all_pairs
                                   if p["decision"] == "accept" and not p.get("dup_rank")),
           "review_letters": sum(p["evidence"]["target_letters"] for p in all_pairs
                                 if p["decision"] == "review" and not p.get("dup_rank")),
           "seconds": round(time.time() - t0, 1)}
    (args.out / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    print(json.dumps({k: run[k] for k in ("decisions", "accepted_letters", "review_letters", "lines_by_status",
                                           "pairs_with_open_lines")}), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
