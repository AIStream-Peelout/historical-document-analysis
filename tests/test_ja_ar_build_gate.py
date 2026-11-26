"""Synthetic tests of gate v2's core (text skeletons, evidence channels, assignment, coverage). No edition text."""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

import numpy as np
import pytest

from src.finetuning.qwen_hebrew.ja_ar_build import align, assign, textnorm
from src.finetuning.qwen_hebrew.ja_ar_build.align import NullPool, ReadStream
from src.finetuning.qwen_hebrew.ja_ar_build.assign import GateParams, Group, Photo, Side

HEB = [chr(c) for c in range(0x05D0, 0x05EB) if chr(c) not in "ךםןףץ"]


def rand_line(rng: random.Random, n: int = 30) -> str:
    """A random Hebrew-letter line with spaces.

    :param rng: Random source.
    :param n: Letters.
    :return: The line.
    """
    letters = [rng.choice(HEB) for _ in range(n)]
    for k in range(5, n, 6):
        letters[k] = letters[k] + " "
    return "".join(letters)


def noisy(line: str, rng: random.Random, rate: float = 0.2) -> str:
    """Copy of a line with letter substitutions (a reader's errors).

    :param line: Source line.
    :param rng: Random source.
    :param rate: Substitution rate.
    :return: The noisy copy.
    """
    return "".join(rng.choice(HEB) if ch != " " and rng.random() < rate else ch for ch in line)


def kraken_read(sha: str, lines: Sequence[str], height: int = 40) -> ReadStream:
    """A Kraken-like read with one box per line, stacked vertically.

    :param sha: Photograph id.
    :param lines: Read lines.
    :param height: Line height in pixels.
    :return: The read stream.
    """
    meta = {"confidences": [0.9] * len(lines), "bboxes": [[100, 50 + i * height, 900, 50 + i * height + 30]
                                                          for i in range(len(lines))], "scale": 1.0,
            "view": {"rotation": 0}}
    return align.stream_from_read(sha, "kraken:test", lines, meta)


def null_pool(rng: random.Random, n: int = 120) -> NullPool:
    """Null pool of random reads of several lengths.

    :param rng: Random source.
    :param n: Reads.
    :return: The pool.
    """
    reads = []
    for k in range(n):
        lines = [rand_line(rng) for _ in range(rng.choice([4, 8, 14, 24]))]
        reads.append(kraken_read(f"null{k}", lines))
    return NullPool.from_streams(reads)


def make_side(lines: List[str], row_id: str = "r1", side_key: str = "recto") -> Side:
    """A target side.

    :param lines: Target lines.
    :param row_id: Candidate id.
    :param side_key: Side key.
    :return: The side.
    """
    return Side(row_id=row_id, side_key=side_key, source="test", script="hebrew", lines=lines,
                skels=textnorm.line_skeletons(lines))


def make_group(sides: List[Side], reads: Dict[str, ReadStream], labels: Optional[Dict[str, str]] = None) -> Group:
    """A group from sides and one read per photograph.

    :param sides: Target sides.
    :param reads: Read per photograph id.
    :param labels: Photograph labels.
    :return: The group.
    """
    labels = labels or {}
    photos = {sha: Photo(sha256=sha, path=f"/tmp/{sha}.jpg", label=labels.get(sha), width=1000, height=2000)
              for sha in reads}
    return Group(document_group="g1", sides=sides, photos=photos, reads={sha: [r] for sha, r in reads.items()})


def test_skeleton_folds_finals_marks_and_arabic() -> None:
    """Finals fold, marks and non-letters drop, Arabic variants fold."""
    assert textnorm.skeleton("שָׁלוֹם 12 [...] abc") == "שלומ"
    assert textnorm.skeleton("أحمد إلى مدرسة") == "احمداليمدرسه"
    assert textnorm.match_form("بنت") == textnorm.match_form("ثيب")


def test_ja_signal_needs_three_markers() -> None:
    """Three distinct markers give the signal; two do not; a prefixed marker counts."""
    assert textnorm.ja_signal("אלדי כתבת ואלתי ענדי")
    assert not textnorm.ja_signal("אלדי כתבת ואלתי")


def test_kgram_chain_rewards_order() -> None:
    """The ordered k-gram chain is long for the right read and short for a shuffled one."""
    rng = random.Random(1)
    lines = [rand_line(rng) for _ in range(10)]
    side = make_side(lines)
    read = kraken_read("a", [noisy(ln, rng, 0.15) for ln in lines])
    shuffled = kraken_read("b", [noisy(ln, rng, 0.15)[::-1] for ln in lines])
    _, chain = align.kgram_chain(side.unit_skels, read)
    _, chain_s = align.kgram_chain(side.unit_skels, shuffled)
    assert len(chain) > 3 * max(1, len(chain_s))


def test_long_line_is_cut_into_units() -> None:
    """A flattened line of 200 letters becomes several units of one line."""
    rng = random.Random(2)
    side = make_side([rand_line(rng, 200)])
    assert len(side.unit_skels) >= 3
    assert set(side.unit_line) == {0}


def test_matching_pair_is_accepted_as_is() -> None:
    """A target read by its own photograph is accepted without remedies."""
    rng = random.Random(3)
    lines = [rand_line(rng) for _ in range(12)]
    group = make_group([make_side(lines)], {"a": kraken_read("a", [noisy(ln, rng) for ln in lines])})
    pairs, _ = assign.evaluate_group(group, null_pool(rng), GateParams())
    acc = [p for p in pairs if p["decision"] == "accept"]
    assert len(acc) == 1 and acc[0]["image_sha256"] == "a"
    assert acc[0]["remedies"] == [] and len(acc[0]["lines"]) == 12


def test_cross_document_swap_is_rejected() -> None:
    """A target paired only with another document's photograph is rejected."""
    rng = random.Random(4)
    lines = [rand_line(rng) for _ in range(12)]
    other = [rand_line(rng) for _ in range(12)]
    group = make_group([make_side(lines)], {"b": kraken_read("b", [noisy(ln, rng) for ln in other])})
    pairs, _ = assign.evaluate_group(group, null_pool(rng), GateParams())
    assert [p["decision"] for p in pairs] == ["reject"]


def test_two_faces_split_and_repair() -> None:
    """A target holding both faces' text is split over the two photographs."""
    rng = random.Random(5)
    recto = [rand_line(rng) for _ in range(8)]
    verso = [rand_line(rng) for _ in range(8)]
    group = make_group([make_side(recto + verso)],
                       {"a": kraken_read("a", [noisy(ln, rng) for ln in recto]),
                        "b": kraken_read("b", [noisy(ln, rng) for ln in verso])}, labels={"a": "1r", "b": "1v"})
    pairs, _ = assign.evaluate_group(group, null_pool(rng), GateParams())
    acc = {p["image_sha256"]: p for p in pairs if p["decision"] == "accept"}
    assert set(acc) == {"a", "b"}
    assert acc["a"]["line_indices"] == list(range(8))
    assert acc["b"]["line_indices"] == list(range(8, 16))
    assert "trim" in acc["a"]["remedies"] and "re_pair" in acc["b"]["remedies"]


def test_appended_text_without_room_is_trimmed() -> None:
    """Lines appended from another document are trimmed when the photograph has no ink for them."""
    rng = random.Random(6)
    own = [rand_line(rng) for _ in range(10)]
    extra = [rand_line(rng) for _ in range(6)]
    group = make_group([make_side(own + extra)], {"a": kraken_read("a", [noisy(ln, rng) for ln in own])})
    pairs, _ = assign.evaluate_group(group, null_pool(rng), GateParams())
    acc = [p for p in pairs if p["decision"] == "accept"]
    assert len(acc) == 1
    assert acc[0]["line_indices"] == list(range(10))
    assert "trim" in acc[0]["remedies"]


def test_unread_edge_lines_kept_when_photo_has_room() -> None:
    """Damaged last lines (read as garbage) stay with the pair: the photograph has unexplained ink for them."""
    rng = random.Random(7)
    lines = [rand_line(rng) for _ in range(12)]
    read_lines = [noisy(ln, rng) for ln in lines[:10]] + [rand_line(rng) for _ in range(2)]
    group = make_group([make_side(lines)], {"a": kraken_read("a", read_lines)})
    pairs, _ = assign.evaluate_group(group, null_pool(rng), GateParams())
    acc = [p for p in pairs if p["decision"] == "accept"]
    assert len(acc) == 1 and len(acc[0]["lines"]) == 12
    assert "edge_kept" in acc[0]["flags"]


def test_separate_block_of_foreign_ink_is_cropped() -> None:
    """A second text beside the target's block is cropped away."""
    rng = random.Random(8)
    lines = [rand_line(rng) for _ in range(10)]
    foreign = [rand_line(rng) for _ in range(8)]
    read = kraken_read("a", [noisy(ln, rng) for ln in lines] + foreign)
    # move the foreign block to the right half of the page
    for j in range(10, 18):
        read.bboxes[j] = [1200, 50 + (j - 10) * 40, 1900, 80 + (j - 10) * 40]
    group = make_group([make_side(lines)], {"a": read})
    group.photos["a"].width = 2000
    pairs, _ = assign.evaluate_group(group, null_pool(rng), GateParams())
    acc = [p for p in pairs if p["decision"] == "accept"]
    assert len(acc) == 1 and "crop" in acc[0]["remedies"]
    x0, y0, x1, y1 = acc[0]["crop"]
    assert x1 < 1200


@pytest.mark.parametrize("anchors,pos,expected", [([0, 1, 2], [1.0, 5.0, 9.0], [0, 1, 2]),
                                                    ([0, 1, 2], [9.0, 5.0, 1.0], [0])])
def test_best_chain_order(anchors: List[int], pos: List[float], expected: List[int]) -> None:
    """The chain keeps only lines whose read positions do not go backwards.

    :param anchors: Anchored lines.
    :param pos: Their read positions.
    :param expected: Expected chain (first of equal-weight options).
    """
    chain, q = assign.best_chain(anchors, np.array(pos), np.ones(3), 0.0)
    assert len(chain) == len(expected)
