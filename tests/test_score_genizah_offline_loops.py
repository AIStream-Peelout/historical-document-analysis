"""Tests of the failure classifier's loop rules in ``score_genizah_offline`` (legacy 12-gram and period-aware).

Texts are random words over the Hebrew alphabet (no real page text); a loop is one synthetic line repeated.
"""
import json
import random
from pathlib import Path
from typing import List

import pytest

from src.datasets.evaluations.helper_eval_scripts import run_religious_benchmark as rr
from src.datasets.evaluations.helper_eval_scripts import score_genizah_offline as s

HEBREW = "אבגדהוזחטיכלמנסעפצקרשת"


def words(letters: int, seed: int) -> str:
    """Random words over the Hebrew alphabet.

    :param letters: Approximate letter count.
    :param seed: Random seed.
    :return: The text.
    """
    rng = random.Random(seed)
    out: List[str] = []
    while sum(len(w) for w in out) < letters:
        out.append("".join(rng.choice(HEBREW) for _ in range(rng.randint(2, 6))))
    return " ".join(out)


def letters(text: str) -> str:
    """Hebrew letters of a text, in order.

    :param text: Text.
    :return: Its letters.
    """
    return "".join(ch for ch in text if ch in HEBREW)


GT = words(600, seed=1)
LONG_LINE = words(45, seed=2)            # a 45-letter line: one 12-gram covers ~12/45 of a loop of it
SHORT_LINE = "אבגדהוזח"                  # an 8-letter line: one 12-gram covers well over half a loop of it


def test_repeated_window_share_separates_loops_from_text() -> None:
    """A long-period loop is almost all repeated windows; plain text has none; short texts score 0."""
    loop = letters("\n".join([LONG_LINE] * 200))
    assert s.repeated_window_share(loop) > 0.95
    assert s.repeated_window_share(letters(GT)) == 0.0
    assert s.repeated_window_share("אבג") == 0.0
    assert s.loop_ratio(loop) < s.LOOP_REPEAT_RATIO        # the legacy rule cannot see it


def test_long_period_loop_is_a_loop_only_under_the_period_aware_rule() -> None:
    """Part of the page read, then one long line repeated: substantive under legacy, loop under period_aware."""
    raw = GT[:300] + "\n" + "\n".join([LONG_LINE] * 120)
    hyp = s.asserted_letters(raw)
    gt_letters = letters(GT)
    assert len(hyp) >= s.LOOP_LENGTH_RATIO * len(gt_letters)
    assert s.classify(raw, hyp, gt_letters, ngram_p=0.5, loop_rule="legacy") == "substantive"
    assert s.classify(raw, hyp, gt_letters, ngram_p=0.5, loop_rule="period_aware") == "loop_collapse"
    assert s.classify(raw, hyp, gt_letters, ngram_p=0.5) == "loop_collapse"        # the default


def test_short_period_loop_is_caught_by_both_rules() -> None:
    """The legacy 12-gram rule still fires first, whatever the rule."""
    raw = "\n".join([SHORT_LINE] * 300)
    hyp = s.asserted_letters(raw)
    for rule in s.LOOP_RULES:
        assert s.classify(raw, hyp, letters(GT), ngram_p=0.5, loop_rule=rule) == "loop_collapse"


def test_a_repetitive_page_read_at_its_length_is_not_a_loop() -> None:
    """A page whose own text repeats (a refrain) read at its true length stays substantive."""
    refrain_page = "\n".join([LONG_LINE] * 14)
    gt_letters = letters(refrain_page)
    hyp = s.asserted_letters(refrain_page)
    assert s.repeated_window_share(hyp) > s.LOOP_WINDOW_SHARE      # repeated, but ...
    assert len(hyp) < s.LOOP_LENGTH_RATIO * len(gt_letters)       # ... not longer than the page
    assert s.classify(refrain_page, hyp, gt_letters, ngram_p=0.9) == "substantive"


def test_no_ground_truth_means_legacy_only() -> None:
    """Without a ground truth the period-aware rule does not apply."""
    raw = "\n".join([LONG_LINE] * 120)
    assert not s.is_long_period_loop(s.asserted_letters(raw), "")
    assert s.classify(raw, s.asserted_letters(raw), "", ngram_p=0.5) == "substantive"


def test_unknown_rule_is_refused() -> None:
    """A typo in the rule name raises instead of scoring silently."""
    with pytest.raises(ValueError):
        s.classify("x", "אבג" * 20, letters(GT), 0.5, loop_rule="period")


def test_scoring_rule_record(tmp_path: Path) -> None:
    """The sidecar records the rule and every threshold.

    :param tmp_path: Pytest temporary directory.
    """
    path = s.write_scoring_rule(tmp_path / "out", "legacy")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["loop_rule"] == "legacy" and record["loop_window"] == s.LOOP_WINDOW
    assert record["loop_length_ratio"] == s.LOOP_LENGTH_RATIO and record["loop_repeat_ratio"] == s.LOOP_REPEAT_RATIO


def test_religious_scorer_threads_the_rule(tmp_path: Path) -> None:
    """``run_religious_benchmark.score`` labels a long-period loop by the rule it is given.

    :param tmp_path: Pytest temporary directory.
    """
    doc = {"doc_id": "KTIV_000_FL1", "gt": GT, "is_talmud": False, "n_columns": 1, "genre": "other"}
    out = tmp_path / "raw_outputs" / doc["doc_id"]
    out.mkdir(parents=True)
    page_line = " ".join(GT.split()[40:52])           # a loop repeats a line of the page itself
    (out / "qwen3_vl_test.txt").write_text(GT[:300] + "\n" + "\n".join([page_line] * 120), encoding="utf-8")
    legacy = rr.score([doc], tmp_path / "raw_outputs", loop_rule="legacy")
    aware = rr.score([doc], tmp_path / "raw_outputs", loop_rule="period_aware")
    assert legacy[0]["failure_mode"] == "substantive" and aware[0]["failure_mode"] == "loop_collapse"
