"""Tests for the photo-centric line assembly (which transcription lines stand on which photograph)."""
from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pytest

from src.finetuning.qwen_hebrew.ja_ar_build import align, assemble as A, textnorm
from src.finetuning.qwen_hebrew.ja_ar_build.align import NullPool
from src.finetuning.qwen_hebrew.ja_ar_build.assign import Group, Photo, Side

LETTERS = "אבגדהוזחטיכלמנסעפצקרשת"
WIDTH, HEIGHT = 1000, 2000


def make_lines(n: int, seed: int, words: int = 7) -> List[str]:
    """Distinct pseudo-Hebrew lines.

    :param n: Number of lines.
    :param seed: Random seed.
    :param words: Words per line.
    :return: Lines.
    """
    rng = random.Random(seed)
    return [" ".join("".join(rng.choice(LETTERS) for _ in range(rng.randint(3, 6))) for _ in range(words))
            for _ in range(n)]


def read_of(sha: str, lines: Sequence[str], x: Sequence[float] = (), top: float = 100.0, pitch: float = 40.0,
            rows: Sequence[int] = ()) -> align.ReadStream:
    """A Kraken-like read whose lines sit on a regular grid.

    :param sha: Photograph digest.
    :param lines: Read lines.
    :param x: Horizontal centre per line (pixels; default: one page in the middle).
    :param top: Baseline of the first row.
    :param pitch: Row distance.
    :param rows: Row of each line (default: one row per line, in order).
    :return: The read stream.
    """
    xs = list(x) or [500.0] * len(lines)
    rs = list(rows) or list(range(len(lines)))
    boxes = [[cx - 200, top + pitch * r, cx + 200, top + pitch * r + 30] for cx, r in zip(xs, rs)]
    return align.stream_from_read(sha, "kraken:test", list(lines),
                                  {"bboxes": boxes, "scale": 1.0, "confidences": [0.9] * len(lines)})


def group_of(sides: Sequence[Tuple[str, List[str]]], reads: Dict[str, align.ReadStream]) -> Group:
    """A one-candidate document group.

    :param sides: ``(side key, lines)`` in edition order.
    :param reads: Read per photograph digest.
    :return: The group.
    """
    return Group(document_group="doc", sides=[Side("row", key, "fgp_targeted", "hebrew", lines,
                                                   textnorm.line_skeletons(lines)) for key, lines in sides],
                 photos={sha: Photo(sha, f"/{sha}.jpg", width=WIDTH, height=HEIGHT) for sha in reads},
                 reads={sha: [st] for sha, st in reads.items()})


@pytest.fixture(scope="module")
def pool() -> NullPool:
    """Null reads of other documents.

    :return: A pool of 60 unrelated reads.
    """
    return NullPool.from_streams(read_of(f"null{k}", make_lines(20, 1000 + k)) for k in range(60))


def by_photo(pairs: Sequence[dict]) -> Dict[str, dict]:
    """Pair records by photograph digest.

    :param pairs: Pair records.
    :return: Mapping.
    """
    return {p["image_sha256"]: p for p in pairs if p["image_sha256"]}


def test_viterbi_keeps_a_run_through_a_stray_match_and_switches_at_a_page_break() -> None:
    """One stray line never moves; a run of lines on another photograph does."""
    w = np.zeros((10, 2))
    w[:5, 0] = 8.0
    w[2] = [0.0, 8.0]                 # a stray strong match on the other photograph
    w[5:, 1] = 8.0
    assert A.viterbi(w, 6.0).tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


def test_page_columns_split_only_two_separate_pages() -> None:
    """Matched boxes in two clusters are two pages; one cluster is one page."""
    params = A.AssembleParams()
    assert A.page_columns([0.7, 0.72, 0.71, 0.69], params) is None
    split = A.page_columns([0.75, 0.74, 0.76, 0.25, 0.26, 0.24], params)
    assert split is not None and 0.26 < split < 0.74
    assert A.page_columns([0.75, 0.74, 0.76, 0.25], params) is None       # one stray box is not a page


def test_split_point_respects_room_and_weak_evidence() -> None:
    """A stretch between two pages is shared out by room; the weak evidence chooses the cut."""
    assert A._split_point([0, 0, 0], [0, 0, 0], 1, 1) == (1, 1)            # the middle line has no room: dropped
    assert A._split_point([2, 2, 0, 0], [0, 0, 2, 2], 4, 4) == (2, 2)
    assert A._split_point([0, 0], [0, 0], 0, 0) == (0, 0)


def test_lines_follow_the_photographed_pages_not_the_typeset_pages(pool: NullPool) -> None:
    """A manuscript page holding one typeset page and the head of the next gets exactly those lines."""
    text = make_lines(20, 1)
    group = group_of([("p1", text[:10]), ("p2", text[10:])],
                     {"a" * 64: read_of("a" * 64, text[:13]), "b" * 64: read_of("b" * 64, text[13:])})
    pairs, lines = A.evaluate_group(group, pool)
    got = by_photo(pairs)
    assert got["a" * 64]["lines"] == text[:13] and got["a" * 64]["side_keys"] == ["p1", "p2"]
    assert got["b" * 64]["lines"] == text[13:] and got["b" * 64]["decision"] == "accept"
    assert got["a" * 64]["line_refs"][9:11] == [["p1", 9], ["p2", 0]]
    assert len(lines) == 20 and all(r["supported"] for r in lines)


def test_unread_line_with_room_stays_and_one_without_room_is_left_open(pool: NullPool) -> None:
    """A damaged line the reader missed stays (the page has a slot); a line without a slot is unresolved, not cut."""
    text = make_lines(12, 2)
    damaged, extra = text[4], make_lines(1, 77)[0]
    edition = text[:8] + [extra] + text[8:]
    read_lines = text[:4] + text[5:]                                       # line 4 unread, its row left empty
    rows = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11]
    group = group_of([("p1", edition)], {"a" * 64: read_of("a" * 64, read_lines, rows=rows)})
    pairs, records = A.evaluate_group(group, pool)
    pair = by_photo(pairs)["a" * 64]
    assert damaged in pair["lines"] and pair["lines"] == text
    assert pair["decision"] == "review" and "unresolved_lines" in pair["reasons"]
    (entry,) = pair["open_lines"]
    assert entry["lines"] == [extra] and (entry["after_line"], entry["before_line"]) == (8, 9)
    assert entry["refs"] == [["p1", 8]] and pair["evidence"]["lines_open"] == 1
    assert pair["unread_lines"] == [5]                                     # the damaged line, by its number in the text
    assert [r["status"] for r in records].count("open") == 1 and records[4]["status"] == "kept_unread"


def test_a_line_in_the_editors_own_wording_is_left_out_on_the_documents_evidence(pool: NullPool) -> None:
    """An unread line that is an editor's note by its wording is accounted for and does not hold the pair."""
    text = make_lines(12, 2)
    note = "הערה של המהדיר שאינה בכתב היד כלל ועיקר"
    group = group_of([("p1", text[:8] + [note] + text[8:])], {"a" * 64: read_of("a" * 64, text)})
    pairs, records = A.evaluate_group(group, pool)
    pair = by_photo(pairs)["a" * 64]
    assert pair["lines"] == text and pair["decision"] == "accept" and pair["open_lines"] == []
    assert records[8]["status"] == "note" and records[8]["pair_id"] is None


def test_two_page_opening_reads_right_page_first(pool: NullPool) -> None:
    """On an opening the right page's lines come first whatever their place in the edition."""
    text = make_lines(16, 3)
    right, left = text[8:], text[:8]                                       # the later text is on the right page
    read = read_of("a" * 64, left + right, x=[250.0] * 8 + [750.0] * 8, rows=list(range(8)) * 2)
    group = group_of([("p1", text[:8]), ("p2", text[8:])], {"a" * 64: read})
    pairs = [p for p in A.evaluate_group(group, pool)[0] if p["image_sha256"]]
    assert [p["evidence"]["page"] for p in pairs] == ["right", "left"]     # a clear gutter: one pair per page
    assert pairs[0]["lines"] == right and pairs[1]["lines"] == left
    assert pairs[0]["remedies"] == ["assemble", "split"] and pairs[0]["decision"] == "accept"
    assert pairs[0]["crop"][0] == pairs[1]["crop"][2] and 450 <= pairs[0]["crop"][0] <= 550
    whole = by_photo(A.evaluate_group(group, pool, A.AssembleParams(gutter_min=2.0))[0])["a" * 64]
    assert whole["lines"] == right + left and whole["evidence"]["pages"] == 2   # no clear gutter: right page first


def test_wrapped_line_end_is_kept_with_its_line(pool: NullPool) -> None:
    """The typeset wrap of a line's last word has no slot of its own but is on the photograph."""
    text = make_lines(8, 4)
    tail = "ואחרית"
    edition = text[:3] + [tail] + text[3:]
    read_lines = text[:2] + [text[2] + " " + tail] + text[3:]
    group = group_of([("p1", edition)], {"a" * 64: read_of("a" * 64, read_lines)})
    pair = by_photo(A.evaluate_group(group, pool)[0])["a" * 64]
    assert pair["lines"] == edition


def test_uncovered_ink_sends_a_photograph_to_review(pool: NullPool) -> None:
    """A photograph with a block of writing the text does not explain is not accepted."""
    text = make_lines(8, 5)
    extra = make_lines(8, 6)
    group = group_of([("p1", text)], {"a" * 64: read_of("a" * 64, text + extra)})
    pair = by_photo(A.evaluate_group(group, pool)[0])["a" * 64]
    assert pair["decision"] == "review" and pair["reasons"] == ["uncovered_ink"] and pair["lines"] == text


def test_ink_of_another_page_is_cropped_away(pool: NullPool) -> None:
    """Writing on the other page of an opening, which the text does not cover, is cut off by a crop."""
    text = make_lines(8, 7)
    other = make_lines(8, 8)
    read = read_of("a" * 64, text + other, x=[750.0] * 8 + [250.0] * 8, rows=list(range(8)) * 2)
    group = group_of([("p1", text)], {"a" * 64: read})
    pair = by_photo(A.evaluate_group(group, pool)[0])["a" * 64]
    assert pair["decision"] == "accept" and pair["remedies"] == ["assemble", "crop"] and pair["lines"] == text
    x0, y0, x1, y1 = pair["crop"]
    assert 450 < x0 < 550 and x1 >= 950 and y0 < 100 and y1 > 100 + 40 * 7       # the right page only


def test_unread_lines_at_the_foot_of_the_text_are_left_open(pool: NullPool) -> None:
    """Lines after the last supported line that no read confirms are neither put into the text nor found absent.

    They may be an editor's apparatus or the foot of the page that the reader could not read: the reads cannot
    tell, so the pair is held and carries the lines for a judge.
    """
    text = make_lines(10, 9)
    more = make_lines(4, 10)
    group = group_of([("p1", text + more)], {"a" * 64: read_of("a" * 64, text)})
    pair = by_photo(A.evaluate_group(group, pool)[0])["a" * 64]
    assert pair["lines"] == text and pair["decision"] == "review" and pair["reasons"] == ["unresolved_lines"]
    (entry,) = pair["open_lines"]
    assert entry["lines"] == more and (entry["after_line"], entry["before_line"]) == (10, None)
    unrelated = make_lines(1, 12)[0]                                       # the read has one more line of other ink
    group = group_of([("p1", text + more[:1])], {"a" * 64: read_of("a" * 64, text + [unrelated])})
    pair = by_photo(A.evaluate_group(group, pool)[0])["a" * 64]
    assert pair["lines"] == text                                           # room alone puts no line into the text
    assert pair["decision"] == "review" and pair["open_lines"][0]["lines"] == more[:1]


def test_a_reader_that_misses_the_end_of_a_page_never_certifies_the_shortened_text(pool: NullPool) -> None:
    """The integrity audit's fixture (2026-11-22): 20 true lines, the reader reads 16, the pair is not accepted."""
    text = make_lines(20, 791)
    group = group_of([("p1", text)], {"a" * 64: read_of("a" * 64, text[:16])})
    pairs, records = A.evaluate_group(group, pool)
    assert all(pair["decision"] != "accept" or pair["lines"] == text for pair in pairs)
    pair = by_photo(pairs)["a" * 64]
    assert pair["lines"] == text[:16] and pair["open_lines"][0]["lines"] == text[16:]
    assert pair["evidence"]["lines_open"] == 4 and "open_lines" in pair["flags"]
    assert [r["status"] for r in records] == ["read"] * 16 + ["open"] * 4


def test_unread_lines_between_two_photographs_hold_both_pairs(pool: NullPool) -> None:
    """A stretch no read places borders the pair before it and the pair after it; both carry it, counted once."""
    text = make_lines(20, 31)
    group = group_of([("p1", text)], {"a" * 64: read_of("a" * 64, text[:8]), "b" * 64: read_of("b" * 64, text[12:])})
    pairs, records = A.evaluate_group(group, pool)
    got = by_photo(pairs)
    first, second = got["a" * 64], got["b" * 64]
    assert first["lines"] == text[:8] and second["lines"] == text[12:]
    assert first["decision"] == second["decision"] == "review"
    (after,), (before,) = first["open_lines"], second["open_lines"]
    assert after["lines"] == before["lines"] == text[8:12] and after["stretch"] == before["stretch"] == [8, 12]
    assert (after["after_line"], after["before_line"], after["other_pair"]) == (8, None, second["pair_id"])
    assert (before["after_line"], before["before_line"], before["other_pair"]) == (None, 1, first["pair_id"])
    assert [r["status"] for r in records].count("open") == 4               # each line once, whatever pairs it borders
    assert [r["pair_id"] for r in records[:8]] == [first["pair_id"]] * 8 and records[9]["pair_id"] is None


def test_open_lines_count_for_the_page_they_border_not_for_the_whole_photograph(pool: NullPool) -> None:
    """On an opening cut into two pages only the page whose text the unread lines border carries them."""
    text = make_lines(16, 3)
    more = make_lines(2, 41)
    right, left = text[8:], text[:8]
    read = read_of("a" * 64, left + right, x=[250.0] * 8 + [750.0] * 8, rows=list(range(8)) * 2)
    group = group_of([("p1", text[:8]), ("p2", text[8:] + more)], {"a" * 64: read})
    pairs = {p["evidence"]["page"]: p for p in A.evaluate_group(group, pool)[0] if p["image_sha256"]}
    assert pairs["right"]["evidence"]["lines_open"] == 2 and pairs["right"]["decision"] == "review"
    assert pairs["left"]["evidence"]["lines_open"] == 0 and pairs["left"]["decision"] == "accept"
    assert pairs["left"]["open_lines"] == [] and "lines_dropped_no_room" not in pairs["left"]["evidence"]


def test_end_lines_need_weak_evidence_as_well_as_room() -> None:
    """At a page's end a line is kept only up to the last line that weakly matches, and never beyond the room."""
    need = [True, True, True, True]
    assert A._evident_run([2.0, 3.0, 0.2, 2.5], need, 1.5) == 2
    assert A._evident_run([2.0, float("nan"), 2.5, 0.0], [True, False, True, True], 1.5) == 3   # a short line inside
    assert A._evident_run([2.0, float("nan")], [True, False], 1.5) == 1                        # ... never at the end
    assert A._split_point([2, 2, 0, 0], [0, 0, 0, 2], 4, 4, need, 1.5) == (2, 1)
    assert A._split_point([2, 2, 2, 2], [0, 0, 0, 0], 1, 4, need, 1.5) == (1, 0)                # capped by room


def test_unsupported_line_with_a_weak_sign_of_a_note_is_left_open(pool: NullPool) -> None:
    """An unread line with a word editors use (and manuscripts too) is put into no text, and that is not final."""
    text = make_lines(10, 13)
    note = "כנראה שיש כאן שיבוש של הסופר"
    rows = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]                                 # an empty slot after the fourth line
    group = group_of([("p1", text[:4] + [note] + text[4:])], {"a" * 64: read_of("a" * 64, text, rows=rows)})
    pair = by_photo(A.evaluate_group(group, pool)[0])["a" * 64]
    assert note not in pair["lines"] and pair["lines"] == text
    assert pair["decision"] == "review" and pair["open_lines"][0]["lines"] == [note]
    assert (pair["open_lines"][0]["after_line"], pair["open_lines"][0]["before_line"]) == (4, 5)


def test_a_block_set_twice_is_in_the_text_once_and_left_to_a_judge(pool: NullPool) -> None:
    """Whether the edition repeats a block or the manuscript does, the alignment cannot tell: nothing is certified."""
    text = make_lines(8, 14)
    group = group_of([("p1", text), ("p2", text)], {"a" * 64: read_of("a" * 64, text)})
    pairs, records = A.evaluate_group(group, pool)
    pair = by_photo(pairs)["a" * 64]
    assert pair["lines"] == text and "doubled_lines_dropped" in pair["flags"]
    assert pair["decision"] == "review" and pair["open_lines"][0]["lines"] == text
    assert [r["status"] for r in records[8:]] == ["doubled"] * 8


def test_a_block_the_manuscript_itself_repeats_is_never_cut_silently(pool: NullPool) -> None:
    """36 lines of which four are written again further down, all 40 read: the pair is not accepted short of four."""
    text = make_lines(36, 55)
    page = text[:20] + text[:4] + text[20:]                                # as written, and as the edition has it
    group = group_of([("p1", page)], {"a" * 64: read_of("a" * 64, page, pitch=20.0)})
    pairs, records = A.evaluate_group(group, pool)
    pair = by_photo(pairs)["a" * 64]
    assert pair["decision"] != "accept" or pair["lines"] == page
    assert sum(len(e["lines"]) for e in pair["open_lines"]) + len(pair["lines"]) == 40


def test_a_single_line_set_twice_is_left_open(pool: NullPool) -> None:
    """One repeated line may be the edition's slip or a refrain on the page: it is not cut on a read's word."""
    text = make_lines(10, 15)
    group = group_of([("p1", text[:5] + [text[4]] + text[5:])], {"a" * 64: read_of("a" * 64, text)})
    pairs, records = A.evaluate_group(group, pool)
    pair = by_photo(pairs)["a" * 64]
    assert pair["lines"] == text and pair["decision"] == "review" and pair["reasons"] == ["unresolved_lines"]
    (entry,) = pair["open_lines"]
    assert entry["status"] == ["doubled"] and (entry["after_line"], entry["before_line"]) == (5, 6)
    assert records[5]["status"] == "doubled"


def test_lines_set_beside_a_translation_send_the_pair_to_review(pool: NullPool) -> None:
    """An edition that prints a translation after every line matches only the first half of each line."""
    text = make_lines(10, 15)
    gloss = make_lines(10, 16)
    edition = [f"{line} {extra}" for line, extra in zip(text, gloss)]
    group = group_of([("p1", edition)], {"a" * 64: read_of("a" * 64, text)})
    pair = by_photo(A.evaluate_group(group, pool)[0])["a" * 64]
    assert pair["decision"] == "review" and "lines_partly_matched" in pair["reasons"]
    assert pair["evidence"]["unmatched_end"] > 0.4


ARABIC_LETTERS = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"


def make_arabic_lines(n: int, seed: int, words: int = 6) -> List[str]:
    """Distinct pseudo-Arabic lines.

    :param n: Number of lines.
    :param seed: Random seed.
    :param words: Words per line.
    :return: Lines.
    """
    rng = random.Random(seed)
    return [" ".join("".join(rng.choice(ARABIC_LETTERS) for _ in range(rng.randint(3, 6))) for _ in range(words))
            for _ in range(n)]


def test_a_hebrew_script_read_says_nothing_about_arabic_script_lines() -> None:
    """For an Arabic-script candidate a read in Hebrew letters is no evidence and no unexplained writing."""
    lines = make_arabic_lines(12, 7)
    kraken = read_of("p1", make_lines(14, 8))                               # a Hebrew-script reader's lines of the page
    vlm = align.stream_from_read("p1", "vlm:test", list(lines), {})
    assert kraken.script == "hebrew" and vlm.script == "arabic"
    assert not A.read_speaks_for("arabic", kraken) and A.read_speaks_for("arabic", vlm)
    assert A.read_speaks_for("hebrew", kraken) and A.read_speaks_for("mixed", kraken)
    sides = [Side("row", "1r", "fgp_targeted", "arabic", lines, textnorm.line_skeletons(lines))]
    photos = {"p1": Photo("p1", "/p1.jpg", width=WIDTH, height=HEIGHT)}
    null = NullPool.from_streams(align.stream_from_read(f"n{k}", "vlm:test", make_arabic_lines(12, 500 + k), {})
                                 for k in range(40))
    only_kraken = Group(document_group="doc", sides=sides, photos=photos, reads={"p1": [kraken]})
    pairs, _ = A.evaluate_group(only_kraken, null)
    assert [p["reasons"] for p in pairs] == [["no_usable_read"]]            # left to the judges, not judged by Kraken
    both = Group(document_group="doc", sides=sides, photos=photos, reads={"p1": [kraken, vlm]})
    pairs, _ = A.evaluate_group(both, null)
    pair = by_photo(pairs)["p1"]
    assert pair["decision"] == "accept" and pair["lines"] == lines
    assert pair["evidence"]["reader"].startswith("vlm") and "uncovered_ink" not in pair["reasons"]
