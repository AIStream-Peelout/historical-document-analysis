"""Tests for the read-time health check of VLM page reads (src/datasets/consensus/read_health.py)."""
import json
import random
from typing import Dict, List, Optional, Sequence, Tuple

import pytest

from src.datasets.consensus import read_health as rh
from src.datasets.consensus.two_reader_lines import parse_grounded

CAP = 3500
HEBREW = "אבגדהוזחטיכלמנסעפצקרשת"


def random_text(n_letters: int, seed: int) -> str:
    """Words of pseudo-random Hebrew letters: no 12-letter run repeats, like real text.

    :param n_letters: Letters in the text.
    :type n_letters: int
    :param seed: Random seed.
    :type seed: int
    :return: Text of exactly ``n_letters`` letters in words of 3-6 letters.
    :rtype: str
    """
    rng = random.Random(seed)
    letters = [rng.choice(HEBREW) for _ in range(n_letters)]
    words, i = [], 0
    while i < len(letters):
        k = rng.randint(3, 6)
        words.append("".join(letters[i:i + k]))
        i += k
    return " ".join(words)


def text_line(i: int, n_letters: int = 30) -> str:
    """A line of distinct pseudo-random Hebrew words.

    :param i: Line number (the seed).
    :type i: int
    :param n_letters: Letters in the line.
    :type n_letters: int
    :return: Line text.
    :rtype: str
    """
    return random_text(n_letters, seed=1000 + i)


def reply_of(texts: Sequence[str], closed: bool = True) -> str:
    """A grounded reply in the reader's format.

    :param texts: Line texts.
    :type texts: Sequence[str]
    :param closed: Close the array (a capped reply is cut mid-entry instead).
    :type closed: bool
    :return: Reply text.
    :rtype: str
    """
    body = ", ".join(json.dumps({"text": t, "bbox_2d": [50, 40 + 20 * i, 900, 58 + 20 * i]}, ensure_ascii=False)
                     for i, t in enumerate(texts))
    return f"[{body}]" if closed else f'[{body}, {{"text": "{texts[-1][:5]}'


def judge(texts: Sequence[str], kraken: Sequence[str], closed: bool = True, tokens: Optional[int] = None,
          reply: Optional[str] = None, lines: Optional[List[str]] = None) -> rh.ReadHealth:
    """Run the detector on a synthetic read.

    :param texts: Line texts of the reply.
    :type texts: Sequence[str]
    :param kraken: Kraken fragment texts.
    :type kraken: Sequence[str]
    :param closed: Close the reply's array.
    :type closed: bool
    :param tokens: Reply token count (None = unknown).
    :type tokens: Optional[int]
    :param reply: Use this reply instead of building one.
    :type reply: Optional[str]
    :param lines: Parsed lines (default: ``texts``).
    :type lines: Optional[List[str]]
    :return: Verdict.
    :rtype: rh.ReadHealth
    """
    reply = reply if reply is not None else reply_of(texts, closed)
    return rh.read_health(reply, list(texts) if lines is None else lines, list(kraken), CAP, tokens)


def page(n: int = 12) -> Tuple[List[str], List[str]]:
    """A healthy page: the VLM's lines and Kraken's fragments of the same text.

    :param n: Lines.
    :type n: int
    :return: ``(vlm texts, kraken texts)``.
    :rtype: Tuple[List[str], List[str]]
    """
    texts = [text_line(i) for i in range(n)]
    return texts, list(texts)


# ----------------------------------------------------------------------------- ok


def test_healthy_read_is_ok() -> None:
    texts, kraken = page()
    h = judge(texts, kraken, tokens=600)
    assert h.label == rh.LABEL_OK and h.flags == () and not h.failed
    assert h.facts.closed and h.facts.n_entries == len(texts) and h.facts.letter_ratio == pytest.approx(1.0)


def test_classify_read_returns_the_label_only() -> None:
    texts, kraken = page()
    assert rh.classify_read(reply_of(texts), texts, kraken, CAP, 600) == rh.LABEL_OK


def test_code_fences_do_not_unclose_a_reply() -> None:
    texts, kraken = page()
    fenced = "```json\n" + reply_of(texts) + "\n```"
    assert judge(texts, kraken, reply=fenced, tokens=3480).label == rh.LABEL_OK


def test_blank_image_with_empty_reply_is_ok() -> None:
    h = judge([], ["ab", "--"], reply="[]")
    assert h.label == rh.LABEL_OK and h.facts.letter_ratio is None


# ----------------------------------------------------------------------------- loop


def test_single_letter_run_is_a_loop_by_loop_ratio() -> None:
    texts, kraken = page(6)
    looped = texts + ["לי" + "י" * 400]
    h = judge(looped, kraken, tokens=1200)
    assert h.label == rh.LABEL_LOOP and h.facts.loop_ratio >= rh.LOOP_RATIO


def test_long_period_loop_is_caught_by_the_tail_measure() -> None:
    texts, kraken = page(10)
    phrase = "שמדרךבניורבלישברביבררביוסףהכהן"            # 30 letters: too long a period for loop_ratio
    h = judge(texts + [phrase * 25], kraken, tokens=2400)
    assert h.facts.loop_ratio < rh.LOOP_RATIO
    assert h.facts.tail_repeat_share >= rh.TAIL_REPEAT_SHARE
    assert h.label == rh.LABEL_LOOP


def test_repeated_short_lines_are_a_loop() -> None:
    texts, kraken = page(10)
    h = judge(texts + ["א א"] * 9, kraken, tokens=900)
    assert h.facts.dup_lines == 8 and h.facts.dup_share == pytest.approx(8 / 19, abs=1e-3)
    assert h.label == rh.LABEL_LOOP


@pytest.mark.parametrize("n_repeats, n_other, looped", [
    (5, 10, False),    # 4 later repeats: one below DUP_LINES_MIN
    (6, 10, True),     # 5 later repeats, 5/16 = 0.31 of the lines
    (6, 20, False),    # 5 later repeats, but only 5/26 = 0.19 of the lines
])
def test_duplicate_line_boundaries(n_repeats: int, n_other: int, looped: bool) -> None:
    texts, kraken = page(n_other)
    h = judge(texts + ["ככה"] * n_repeats, kraken, tokens=900)
    assert (h.label == rh.LABEL_LOOP) is looped


def test_loop_wins_over_capped_and_both_are_flagged() -> None:
    texts, kraken = page(6)
    h = judge(texts + ["הא׳ע׳פ׳"] * 60, kraken, closed=False, tokens=3499)
    assert h.label == rh.LABEL_LOOP
    assert h.flags[:2] == (rh.LABEL_LOOP, rh.LABEL_CAPPED)


def test_formulaic_but_genuine_text_is_not_a_loop() -> None:
    texts, kraken = page(20)
    texts[5] = texts[12] = "ולא חוב ולא מלוה ולא שום דבר"           # one repeated formula line
    assert judge(texts, kraken, tokens=1500).label == rh.LABEL_OK


# ----------------------------------------------------------------------------- capped


@pytest.mark.parametrize("tokens, capped", [(3400, True), (3399, False), (3500, True)])
def test_cap_boundary_with_a_token_count(tokens: int, capped: bool) -> None:
    texts, kraken = page(40)
    h = judge(texts, kraken, closed=False, tokens=tokens)
    assert (h.label == rh.LABEL_CAPPED) is capped
    assert rh.is_capped(h.facts, CAP) is capped


def test_closed_reply_at_the_cap_is_not_capped() -> None:
    texts, kraken = page(40)
    assert judge(texts, kraken, closed=True, tokens=3480).label == rh.LABEL_OK


def test_cap_without_a_token_count_uses_the_reply_length() -> None:
    limit = rh.MIN_CHARS_PER_TOKEN * (CAP - rh.CAP_MARGIN_TOKENS)
    texts, kraken = page(80)
    reply = reply_of(texts, closed=False)
    assert len(reply) >= limit
    assert judge(texts, kraken, reply=reply).label == rh.LABEL_CAPPED
    cut = reply[: int(limit) - 1]                        # one character short of the limit, still unclosed
    assert judge(texts, kraken, reply=cut).label != rh.LABEL_CAPPED
    assert judge(texts, kraken, reply=reply[: int(limit) + 1]).label == rh.LABEL_CAPPED


def test_a_reply_that_only_forgot_its_closing_bracket_is_not_capped_without_tokens() -> None:
    texts, kraken = page(80)
    slip = reply_of(texts)[:-1] + "\n```"                # every entry finished, "]" missing, fence closed
    assert len(slip) >= rh.MIN_CHARS_PER_TOKEN * (CAP - rh.CAP_MARGIN_TOKENS)
    h = judge(texts, kraken, reply=slip)
    assert not h.facts.closed and h.facts.ends_entry and h.label == rh.LABEL_OK
    assert judge(texts, kraken, reply=slip, tokens=3450).label == rh.LABEL_CAPPED     # a token count overrides


# ----------------------------------------------------------------------------- near-empty


def test_one_short_line_on_a_full_page_is_near_empty() -> None:
    _, kraken = page(20)
    h = judge(["בשמך רחמנא"], kraken, tokens=35)
    assert h.label == rh.LABEL_NEAR_EMPTY
    assert rh.LABEL_SKIPPED in h.flags                       # also far below Kraken, but near-empty comes first


@pytest.mark.parametrize("letters, kraken_letters, label", [
    (24, 400, rh.LABEL_NEAR_EMPTY),
    (25, 400, rh.LABEL_SKIPPED),        # not near-empty any more, still far below Kraken
    (5, 99, rh.LABEL_OK),               # Kraken read too little to call the image a text page
    (5, 100, rh.LABEL_NEAR_EMPTY),
])
def test_near_empty_boundaries(letters: int, kraken_letters: int, label: str) -> None:
    kraken = [random_text(kraken_letters, seed=7)]
    assert judge([random_text(letters, seed=8)], kraken, tokens=40).label == label


def test_empty_reply_on_a_text_page_is_near_empty_not_capped() -> None:
    _, kraken = page(10)
    h = judge([], kraken, reply="")
    assert h.label == rh.LABEL_NEAR_EMPTY and not h.facts.closed


# ----------------------------------------------------------------------------- skipped text


@pytest.mark.parametrize("vlm_letters, kraken_letters, skipped", [
    (79, 100, True),       # 0.79 of Kraken's letters
    (80, 100, False),      # exactly at SKIPPED_RATIO
    (60, 99, False),       # Kraken read too little to compare
    (500, 1000, True),
    (950, 1000, False),
])
def test_skipped_text_boundaries(vlm_letters: int, kraken_letters: int, skipped: bool) -> None:
    h = judge([random_text(vlm_letters, seed=3)], [random_text(kraken_letters, seed=4)], tokens=500)
    assert (h.label == rh.LABEL_SKIPPED) is skipped


def test_parse_loss_does_not_look_like_skipped_text() -> None:
    texts, kraken = page(12)
    h = judge(texts, kraken, lines=texts[:2], tokens=700)    # the parser kept 2 of 12 lines
    assert h.facts.reply_letters == h.facts.kraken_letters
    assert h.label == rh.LABEL_PARSE_LOSS


# ----------------------------------------------------------------------------- parse loss


def test_unescaped_quotes_no_longer_cost_lines_and_lost_lines_are_still_reported() -> None:
    texts = ['הודה ס"ז לתת לו', "שכר שביר ראיינו", 'כ"ז שכר מתעטף', "יום שערבה מערב", 'ה"ק על פנינו']
    reply = "[" + ", ".join('{"text": "%s", "bbox_2d": [57, %d, 453, %d]}' % (t, 90 + 25 * i, 115 + 25 * i)
                            for i, t in enumerate(texts)) + "]"
    parsed, lines = parse_grounded(reply)
    assert len(rh.reply_entries(reply)) == 5 and [ln["text"] for ln in lines] == texts    # fixed 2026-10-05: all five kept
    kraken = [t.replace('"', "") for t in texts]
    assert rh.read_health(reply, [ln["text"] for ln in lines], kraken, CAP, 120).label != rh.LABEL_PARSE_LOSS
    h = rh.read_health(reply, [ln["text"] for ln in lines[:2]], kraken, CAP, 120)           # a read stored with lines missing
    assert h.label == rh.LABEL_PARSE_LOSS and not h.reread
    assert h.facts.n_entries == 5


@pytest.mark.parametrize("parsed_lines, lost", [(8, False), (7, False), (6, True)])
def test_parse_loss_boundaries(parsed_lines: int, lost: bool) -> None:
    texts, kraken = page(8)                                  # 8 entries; losing 1 is the normal truncated tail
    h = judge(texts, kraken, lines=texts[:parsed_lines], tokens=500)
    assert (h.label == rh.LABEL_PARSE_LOSS) is lost


def test_reply_entries_ignore_an_entry_cut_before_its_box() -> None:
    texts, _ = page(4)
    reply = reply_of(texts, closed=False)                    # ends with '{"text": "...' and no box
    assert len(rh.reply_entries(reply)) == 4


# ----------------------------------------------------------------------------- helpers and overrides


def test_thresholds_can_be_overridden() -> None:
    _, kraken = page(20)
    short = random_text(30, seed=5)
    h = judge([short], kraken, tokens=40)
    assert h.label == rh.LABEL_SKIPPED
    strict = rh.Thresholds(near_empty_letters=50)
    assert rh.read_health(reply_of([short]), [short], kraken, CAP, 40, strict).label == rh.LABEL_NEAR_EMPTY


def test_health_of_entry_reads_a_raw_cache_entry() -> None:
    texts, kraken = page(10)
    entry: Dict[str, object] = {"vlm_raw": reply_of(texts), "vlm_lines": [{"text": t, "box": [0, 0, 1, 1]} for t in texts],
                                "frags": [{"text": t, "box": [0, 0, 1, 1], "conf": 0.9} for t in kraken]}
    h = rh.health_of_entry(entry, CAP, 700)
    assert h.label == rh.LABEL_OK and h.facts.kraken_lines == 10
    assert rh.health_of_entry({"vlm_raw": None, "vlm_lines": [], "frags": entry["frags"]}, CAP).label == rh.LABEL_NEAR_EMPTY


def test_to_dict_is_json_ready() -> None:
    texts, kraken = page()
    d = judge(texts, kraken, tokens=500).to_dict()
    assert json.loads(json.dumps(d))["label"] == rh.LABEL_OK and d["flags"] == []


def test_duplicate_lines_ignores_whitespace_and_empty_lines() -> None:
    assert rh.duplicate_lines(["א  ב", "א ב", "", "", "ג"]) == 1


def test_tail_repeat_share_of_short_text_is_zero() -> None:
    assert rh.tail_repeat_share("[{\"text\": \"שלום\"}]") == 0.0
