"""Tests of the blind-audit rules: no verdict mixes two judges, complete means complete, edits need two judges."""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional

import pytest

from src.finetuning.qwen_hebrew.ja_ar_build import adjudicate, approval, audit_contract, build_rows

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "logs" / "next_round" / "ja_ar_v1" / "novel1119" / "audit"
ALPHABET = "אבגדהוזחטיכלמנסעפצקרשת"


def make_lines(n: int, seed: int) -> List[str]:
    """Distinct pseudo-Hebrew lines (unlike enough that no text rule takes one for a repeat of another).

    :param n: Number of lines.
    :param seed: Random seed.
    :return: Lines.
    """
    rng = random.Random(seed)
    return [" ".join("".join(rng.choice(ALPHABET) for _ in range(rng.randint(3, 6))) for _ in range(6))
            for _ in range(n)]


LINES = make_lines(10, 5)
MORE = make_lines(3, 6)
#: What a judge reports who sees further lines of the main text that the pair leaves out.
SEEN_MORE = {"uncovered_writing": "substantial", "uncovered_kind": "main_text_lines"}


def record(verdict: str = "complete_pair", **fields: Any) -> Dict[str, Any]:
    """One judge's record, whole: nothing uncovered, nothing supplied, sure of it, unless said otherwise.

    :param verdict: The verdict.
    :param fields: Fields that differ.
    :return: The record.
    """
    return {"item_id": "x_001", "verdict": verdict, "confidence": 4, "share_on_photo": "all",
            "uncovered_writing": "none", "uncovered_kind": "none", "lines_not_on_photo": [], "supplied_text": "none",
            "order_problem": "none", "garbled_lines": [], **fields}


def result(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]] = None, audit: str = "x", **key: Any
           ) -> Dict[str, Any]:
    """The scored result of an item, as :func:`adjudicate.load_audit` returns it.

    :param a: Judge A's record.
    :param b: Judge B's record.
    :param audit: Audit name.
    :param key: Further key fields (``contract``, ``fingerprint``, ``open_shown``, ``open_digest``, ``lines_match``).
    :return: The result record (judged on the whole photograph ``a...a``, unrotated).
    """
    base = {"item_id": f"{audit}_001", "pair_id": "r1|asm|aaaaaaaaaaaa", "stratum": "assemble_1p_one|ja",
            "decision": "review", "letters": 200}
    return {**adjudicate.score_item({**base, **key}, a, b), "audit": audit, "lines": list(LINES),
            "judged_image": ["a" * 64, 0, None]}


def make_pair(lines: Optional[List[str]] = None, **fields: Any) -> Dict[str, Any]:
    """An assembled pair of candidate ``r1`` whose text is the first lines of its side.

    :param lines: The pair's lines (default: :data:`LINES`).
    :param fields: Fields that differ.
    :return: The pair record.
    """
    lines = list(LINES if lines is None else lines)
    return {"pair_id": "r1|asm|aaaaaaaaaaaa", "row_id": "r1", "side_key": "p1", "side_keys": ["p1"], "source": "pgp",
            "document_group": "g:r1", "script": "hebrew", "image_sha256": "a" * 64, "image_path": "/a.jpg",
            "decision": "review", "remedies": ["assemble"], "line_indices": list(range(len(lines))),
            "line_refs": [["p1", i] for i in range(len(lines))], "lines": lines, "rotation": 0, "crop": None,
            "dup_rank": None, "flags": [], "reasons": ["unread_lines"], "open_lines": [],
            "evidence": {"target_letters": 200}, **fields}


def candidates(lines: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """The candidate behind :func:`make_pair`.

    :param lines: Lines of its one side (default: :data:`LINES` and :data:`MORE`).
    :return: Candidates by row id.
    """
    return {"r1": {"row_id": "r1", "source": "pgp", "targets": [{"side_key": "p1", "lines": lines or LINES + MORE}]}}


def open_after(count: int = 3, lines: Optional[List[str]] = None) -> Dict[str, Any]:
    """Unresolved lines that follow the pair's last line.

    :param count: Number of lines.
    :param lines: The lines (default: the first of :data:`MORE`).
    :return: The ``open_lines`` entry.
    """
    return {"after_line": len(LINES), "before_line": None, "lines": list(lines or MORE[:count]),
            "refs": [["p1", len(LINES) + i] for i in range(count)], "status": ["open"] * count, "letters": 20 * count,
            "stretch": [len(LINES), len(LINES) + count], "left_out": 0, "other_pair": None}


def second_contract(pair: Dict[str, Any], cands: Dict[str, Dict[str, Any]], shown: Optional[List[int]] = None
                    ) -> Dict[str, Any]:
    """The key fields of a second-contract item made from a pair.

    :param pair: The pair as the judges were shown it.
    :param cands: Candidates by row id.
    :param shown: Lines shown per unresolved entry (default: all).
    :return: ``contract``, ``fingerprint``, ``open_shown``, ``open_digest`` and ``lines_match``.
    """
    answer = build_rows.final_answer(pair, cands, {}, set())
    return {"contract": "v2", "fingerprint": build_rows.pair_fingerprint(pair, answer, cands),
            "open_shown": shown or audit_contract.open_shown(pair), "open_digest": audit_contract.open_digest(pair),
            "lines_match": len(answer.lines) == len(pair["lines"])}


def test_a_result_keeps_each_judges_own_record() -> None:
    """The verdict is the worse of the two; every other observation stays with the judge who made it."""
    a = record(uncovered_writing="minor", uncovered_kind="marginal_or_other_hand")
    b = record("photo_has_extra", **SEEN_MORE)
    got = result(a, b)
    assert got["final"] == "photo_has_extra" and (got["judge_a"], got["judge_b"]) == ("complete_pair", "photo_has_extra")
    assert got["records"]["A"]["uncovered_kind"] == "marginal_or_other_hand"
    assert got["records"]["B"]["uncovered_kind"] == "main_text_lines"
    assert "uncovered_kind" not in got and "lines_not_on_photo" not in got      # no field mixes the judges
    assert got["concerns"] == {"A": ["uncovered:marginal_or_other_hand"], "B": ["uncovered:main_text_lines"]}


def test_one_judges_marginalia_never_overrides_the_others_missing_main_text() -> None:
    """The three rows the integrity audit found: accepted as marginalia against a judge. Now held."""
    for a, b in [(record(uncovered_writing="minor", uncovered_kind="marginal_or_other_hand"),
                  record("photo_has_extra", **SEEN_MORE)),
                 (record("cannot_judge", uncovered_kind="marginal_or_other_hand"),
                  record("photo_has_extra", **SEEN_MORE)),
                 (record("photo_has_extra", uncovered_writing="substantial", uncovered_kind="marginal_or_other_hand"),
                  record("photo_has_extra", uncovered_writing="substantial", uncovered_kind="other_page_or_text"))]:
        assert adjudicate.decide(make_pair(), result(a, b)) == ("review", "audit_hold:photo_has_extra")


def test_marginal_writing_left_out_holds_a_pair_even_when_both_judges_call_it_marginal() -> None:
    """Margins are in scope: a text that leaves them out is not the photograph's text."""
    note = record("photo_has_extra", uncovered_writing="substantial", uncovered_kind="marginal_or_other_hand")
    assert adjudicate.decide(make_pair(), result(note, dict(note))) == ("review", "audit_hold:photo_has_extra")


def test_complete_needs_every_judge_and_no_concern() -> None:
    """A complete verdict with a warning attached approves nothing."""
    assert adjudicate.decide(make_pair(), result(record())) == ("review", "audit_hold:one_judge_complete")
    assert adjudicate.decide(make_pair(), result(record(), record())) == ("accept", "audit_complete")
    for concern in ({"uncovered_writing": "minor", "uncovered_kind": "main_text_lines"},
                    {"uncovered_writing": "minor", "uncovered_kind": "marginal_or_other_hand"},
                    {"uncovered_kind": "main_text_lines"}, {"lines_not_on_photo": [3]}, {"supplied_text": "minor"},
                    {"order_problem": "clear"}, {"garbled_lines": [2]}, {"confidence": 2}, {"share_on_photo": "most"}):
        assert adjudicate.decide(make_pair(), result(record(), record(**concern))) == \
            ("review", "audit_hold:complete_with_concerns"), concern
    assert adjudicate.decide(make_pair(), result(record(order_problem="suspected"), record())) == \
        ("accept", "audit_complete")
    assert adjudicate.decide(make_pair(), result(record(), record("cannot_judge"))) == \
        ("review", "audit_hold:cannot_judge")
    assert adjudicate.decide(make_pair(), {"final": "complete_pair", "records": {}}) == \
        ("review", "audit_hold:no_judge_record")


def test_a_record_that_does_not_answer_what_it_was_asked_approves_nothing() -> None:
    """A missing or malformed answer is never read as 'no concern'."""
    bare = {"item_id": "x_001", "verdict": "complete_pair"}
    assert adjudicate.decide(make_pair(), result(bare)) == ("review", "audit_hold:complete_with_concerns")
    assert "record:uncovered_writing" in adjudicate.concerns(bare) and "record:confidence" in adjudicate.concerns(bare)
    for broken in ({"confidence": "4"}, {"confidence": True}, {"uncovered_writing": None},
                   {"lines_not_on_photo": "none"}, {"supplied_text": "a little"}, {"share_on_photo": None}):
        assert adjudicate.decide(make_pair(), result(record(**broken))) == \
            ("review", "audit_hold:complete_with_concerns"), broken
    whole = record()
    assert adjudicate.record_problems(whole, "v1") == [] and adjudicate.record_problems(whole, "v2") == []
    assert adjudicate.record_problems(record(uncovered_writing="minor"), "v2") == ["record:uncovered_writing"]
    del whole["garbled_lines"]                                           # the first contract never asked for it
    assert adjudicate.record_problems(whole, "v1") == [] and adjudicate.record_problems(whole, "v2") != []


def test_lines_leave_a_text_only_on_two_judges_naming_the_same_lines() -> None:
    """One judge's list removes nothing; nor does a list the other judge does not share, nor a number out of range."""
    extra = record("text_has_extra", lines_not_on_photo=[9, 10], share_on_photo="most")
    pair = make_pair()
    assert adjudicate.decide(pair, result(extra)) == ("review", "audit_hold:one_judge_edit")
    assert adjudicate.decide(pair, result(extra, record())) == ("review", "audit_hold:text_has_extra")
    other = record("text_has_extra", lines_not_on_photo=[10], share_on_photo="most")
    assert adjudicate.decide(pair, result(extra, other)) == ("review", "audit_hold:text_has_extra")
    noted = record("text_has_extra", lines_not_on_photo=[9, 10], share_on_photo="most", supplied_text="minor")
    assert adjudicate.decide(pair, result(extra, noted)) == ("review", "audit_hold:text_has_extra")
    beyond = record("text_has_extra", lines_not_on_photo=[9, 10, 11], share_on_photo="most")
    assert adjudicate.decide(pair, result(beyond, extra)) == ("review", "audit_hold:text_has_extra")
    assert pair["lines"] == LINES and "lines_removed_by_judges" not in pair
    assert adjudicate.decide(pair, result(extra, dict(extra))) == ("accept", "audit_lines_dropped")
    assert pair["lines"] == LINES[:8] and pair["line_refs"] == [["p1", i] for i in range(8)]
    assert pair["lines_removed_by_judges"] == [{"audit": "x", "numbers": [9, 10], "refs": [["p1", 8], ["p1", 9]],
                                                "lines": LINES[8:]}]
    many = record("text_has_extra", lines_not_on_photo=[1, 2, 3, 4, 5], share_on_photo="part")
    assert adjudicate.decide(make_pair(), result(many, dict(many))) == ("review", "audit_hold:text_has_extra")


def test_first_contract_judges_settle_unresolved_lines_only_by_seeing_nothing_uncovered() -> None:
    """They were not shown the lines; 'complete' with no uncovered writing at all means the lines are not there."""
    pair = make_pair(open_lines=[open_after()])
    assert adjudicate.decide(pair, result(record(), record())) == ("accept", "audit_complete")
    assert pair["open_lines"][0]["settled"] == "absent" and approval.unsettled_open_lines(pair) == 0
    pair = make_pair(open_lines=[open_after()])
    minor = record(uncovered_writing="minor", uncovered_kind="main_text_lines")
    assert adjudicate.decide(pair, result(minor, record())) == ("review", "audit_hold:complete_with_concerns")
    assert "settled" not in pair["open_lines"][0]


def answered(found: str, count: int = 0, **fields: Any) -> Dict[str, Any]:
    """A second-contract record that answers one entry of unresolved lines.

    :param found: ``none``, ``all``, ``some`` or ``cannot_tell``.
    :param count: Lines found, for ``some``.
    :param fields: Fields that differ.
    :return: The record.
    """
    return record(open_lines=[{"entry": 1, "on_photo": found, "count": count}], **fields)


def test_second_contract_judges_settle_the_unresolved_lines_they_are_shown() -> None:
    """Absent on every judge's word; put back only on two judges' agreement; unsure means held."""
    key = {"contract": "v2", "fingerprint": "f", "open_shown": [3], "lines_match": True}
    pair = make_pair(open_lines=[open_after()])
    assert adjudicate.decide(pair, result(answered("none"), **key)) == ("review", "audit_hold:one_judge_complete")
    assert "settled" not in pair["open_lines"][0]
    assert adjudicate.decide(pair, result(answered("none"), answered("none"), **key)) == ("accept", "audit_complete")
    assert pair["lines"] == LINES and pair["open_lines"][0]["settled"] == "absent"
    for a, b, why in [(answered("all"), None, "audit_hold:one_judge_complete"),
                      (answered("all"), answered("none"), "audit_hold:open_lines_disagree"),
                      (answered("some", 2), answered("some", 1), "audit_hold:open_lines_disagree"),
                      (answered("cannot_tell"), answered("none"), "audit_hold:open_lines_uncertain"),
                      (record(), record(), "audit_hold:open_lines_unanswered")]:
        pair = make_pair(open_lines=[open_after()])
        assert adjudicate.decide(pair, result(a, b, **key)) == ("review", why), why
        assert pair["lines"] == LINES and "settled" not in pair["open_lines"][0]
    pair = make_pair(open_lines=[open_after()], unread_lines=[3])
    assert adjudicate.decide(pair, result(answered("some", 2), answered("some", 2), **key)) == \
        ("accept", "audit_lines_restored")
    assert pair["lines"] == LINES + MORE[:2] and pair["line_refs"][-2:] == [["p1", 10], ["p1", 11]]
    assert pair["line_indices"][-2:] == [10, 11] and pair["open_lines"][0]["settled"] == "restored:2"
    assert "unread_lines" not in pair                                 # its numbers were the old text's
    pair = make_pair(open_lines=[open_after()])                       # 3 lines exist, 2 were shown, both found
    assert adjudicate.decide(pair, result(answered("all"), answered("all"), **{**key, "open_shown": [2]})) == \
        ("review", "audit_hold:open_lines_longer_than_shown")


def test_no_text_is_changed_on_line_numbers_of_another_text() -> None:
    """When the text rules changed the number of lines the judges saw, their numbers are not the pair's."""
    key = {"contract": "v2", "fingerprint": "f", "open_shown": [3], "lines_match": False}
    pair = make_pair(open_lines=[open_after()])
    assert adjudicate.decide(pair, result(answered("all"), answered("all"), **key)) == \
        ("review", "audit_hold:line_numbers_shifted")
    extra = record("text_has_extra", lines_not_on_photo=[10], share_on_photo="most")
    pair = make_pair()
    assert adjudicate.decide(pair, result(extra, dict(extra), **{**key, "open_shown": None})) == \
        ("review", "audit_hold:line_numbers_shifted")
    assert pair["lines"] == LINES
    assert adjudicate.decide(make_pair(), result(record(), record(), **{**key, "open_shown": None})) == \
        ("accept", "audit_complete")                                  # approving the text as shown needs no numbers


def test_lines_before_a_text_are_put_back_next_to_it_and_bring_their_side() -> None:
    """Of a stretch that precedes the text the nearest lines are the last ones; their side joins the pair's sides."""
    entry = {"after_line": None, "before_line": 1, "lines": MORE, "refs": [["p0", i] for i in range(3)],
             "status": ["open"] * 3, "letters": 60, "stretch": [0, 3], "left_out": 0, "other_pair": None}
    pair = make_pair(open_lines=[dict(entry)])
    key = {"contract": "v2", "fingerprint": "f", "open_shown": [3], "lines_match": True}
    assert adjudicate.decide(pair, result(answered("some", 1), answered("some", 1), **key)) == \
        ("accept", "audit_lines_restored")
    assert pair["lines"] == MORE[2:] + LINES and pair["line_refs"][0] == ["p0", 2]
    assert pair["side_keys"] == ["p0", "p1"] and pair["side_key"] == "p0"   # hold-outs and source of p0 now count
    assert audit_contract.judge_open_lines(make_pair(open_lines=[dict(entry)]), shown=2) == \
        [{"entry": 1, "position": "before text line 1", "lines": MORE[1:], "not_shown": 1}]


def test_every_accepted_pair_gets_its_approval_bound_or_goes_back_to_review() -> None:
    """Rule accepts and judged accepts alike; unresolved lines and runs without line accounting approve nothing."""
    cands = candidates()
    clean = make_pair(decision="accept", reasons=[])
    open_pair = make_pair(decision="accept", reasons=[], open_lines=[open_after()], pair_id="r1|asm|open")
    old = make_pair(decision="accept", reasons=[], pair_id="r1|asm|old")
    del old["open_lines"]
    held = make_pair(pair_id="r1|asm|held")
    counts = adjudicate.apply_verdicts([clean, open_pair, old, held], {}, cands, {}, set())
    assert clean["decision"] == "accept" and clean["approval"]["by"] == "rule"
    assert clean["approval"]["scope"] == approval.SCOPE
    assert open_pair["decision"] == "review" and "unresolved_lines" in open_pair["reasons"]
    assert old["decision"] == "review" and "no_line_accounting" in old["reasons"] and "approval" not in old
    assert held["decision"] == "review" and "approval" not in held
    assert counts["approved_by_rule"] == 1 and counts["accept->review:unresolved_lines"] == 1


def test_a_judged_pair_is_approved_with_the_answer_the_judges_saw() -> None:
    """A complete verdict binds the approval; the row builder then finds the fingerprint it computes itself."""
    cands = candidates()
    pair = make_pair(open_lines=[open_after()])
    counts = adjudicate.apply_verdicts([pair], {pair["pair_id"]: [result(record(), record())]}, cands, {}, set())
    assert pair["decision"] == "accept" and pair["flags"] == ["audit_complete"] and pair["reasons"] == []
    assert pair["approval"]["by"] == "judges" and pair["approval"]["judges"] == ["A", "B"]
    assert pair["approval"]["contract"] == "v1" and pair["open_lines"][0]["settled"] == "absent"
    assert pair["audit"]["judges"] == {"A": "complete_pair", "B": "complete_pair"}
    answer = build_rows.final_answer(pair, cands, {}, set())
    assert approval.approval_state(pair, build_rows.pair_fingerprint(pair, answer, cands)) is None
    assert counts["review->accept:audit_complete"] == 1 and counts["approved_by_judges"] == 1


def test_a_verdict_on_another_version_of_a_pair_does_not_apply() -> None:
    """First contract: other lines or another crop. Second: another fingerprint or other unresolved lines."""
    cands = candidates()
    pair = make_pair(lines=LINES[:8])
    counts = adjudicate.apply_verdicts([pair], {pair["pair_id"]: [result(record())]}, cands, {}, set())
    assert counts["judged_but_pair_changed"] == 1 and pair["decision"] == "review" and "audit" not in pair
    cut = make_pair(crop=[0, 0, 10, 10])                              # the judges saw the whole photograph
    counts = adjudicate.apply_verdicts([cut], {cut["pair_id"]: [result(record())]}, cands, {}, set())
    assert counts["judged_but_pair_changed"] == 1 and "audit" not in cut
    keyless = {**result(record()), "judged_image": None}             # the audit's key is lost: nothing is bound
    assert not adjudicate.verdict_applies(make_pair(), keyless, cands, {}, set())
    pair = make_pair()
    seen = second_contract(pair, cands)
    assert adjudicate.verdict_applies(pair, result(record(), **seen), cands, {}, set())
    assert not adjudicate.verdict_applies(make_pair(crop=[0, 0, 10, 10]), result(record(), **seen), cands, {}, set())
    shown = make_pair(open_lines=[open_after()])
    seen = second_contract(shown, cands)
    assert adjudicate.verdict_applies(make_pair(open_lines=[open_after()]), result(record(), **seen), cands, {}, set())
    moved = make_pair(open_lines=[open_after(2)])                     # same answer, other lines beside it
    assert not adjudicate.verdict_applies(moved, result(record(), **seen), cands, {}, set())


def moved_marks(lines: List[str]) -> List[str]:
    """The same lines with a colon moved from the start of the first line to its end and a space doubled.

    :param lines: Lines whose first one starts with a colon.
    :return: Lines with the same letters in the same order.
    """
    return [lines[0].lstrip(":").strip() + " :"] + [lines[1].replace(" ", "  ", 1)] + list(lines[2:])


def test_a_verdict_stays_with_a_text_whose_letters_did_not_change() -> None:
    """A re-ingest that only moves spaces and punctuation leaves the judged text what it was; any letter does not."""
    old_lines = [": " + LINES[0]] + LINES[1:]
    new_lines = moved_marks(old_lines)
    old_c, new_c = candidates(old_lines + MORE), candidates(new_lines + MORE)
    judged = make_pair(lines=old_lines, open_lines=[open_after()])
    seen = second_contract(judged, old_c)
    verdict = {**result(answered("none"), answered("none"), **seen),
               "lines": build_rows.final_answer(judged, old_c, {}, set()).lines,
               "open_lines_shown": audit_contract.judge_open_lines(judged)}
    pair = make_pair(lines=new_lines, open_lines=[open_after()])
    assert adjudicate.verdict_match(judged, verdict, old_c, {}, set()) == "exact"
    assert adjudicate.verdict_match(pair, verdict, new_c, {}, set()) == "same_letters"
    counts = adjudicate.apply_verdicts([pair], {pair["pair_id"]: [verdict]}, new_c, {}, set())
    assert pair["decision"] == "accept" and counts["judged_same_letters"] == 1
    assert pair["audit"]["match"] == "same_letters" and pair["approval"]["text_judged"] == "same_letters"
    answer = build_rows.final_answer(pair, new_c, {}, set())            # bound to the text as it is now
    assert approval.approval_state(pair, build_rows.pair_fingerprint(pair, answer, new_c)) is None
    words = new_lines[0].split()
    swapped = [" ".join([words[1], words[0]] + words[2:])] + new_lines[1:]          # two words changed places
    assert adjudicate.verdict_match(make_pair(lines=swapped, open_lines=[open_after()]), verdict,
                                    candidates(swapped + MORE), {}, set()) is None
    shorter = [new_lines[0][2:]] + new_lines[1:]                                    # a letter is gone
    assert adjudicate.verdict_match(make_pair(lines=shorter, open_lines=[open_after()]), verdict,
                                    candidates(shorter + MORE), {}, set()) is None
    cut = make_pair(lines=new_lines, open_lines=[open_after()], crop=[0, 0, 10, 10])   # another crop
    assert adjudicate.verdict_match(cut, verdict, new_c, {}, set()) is None
    other_open = make_pair(lines=new_lines, open_lines=[open_after(2)])             # other lines beside the text
    assert adjudicate.verdict_match(other_open, verdict, new_c, {}, set()) is None


def test_a_first_contract_verdict_stays_with_the_same_letters_too() -> None:
    """Punctuation moved on a text judged under the first contract: the verdict applies, final letters aside."""
    lines = moved_marks([": " + LINES[0]] + LINES[1:])
    pair = make_pair(lines=lines)
    verdict = {**result(record(), record()), "lines": [": " + LINES[0]] + LINES[1:]}
    assert adjudicate.verdict_match(pair, verdict, candidates(lines + MORE), {}, set()) == "same_letters"
    assert adjudicate.verdict_match(make_pair(), result(record(), record()), candidates(), {}, set()) == "exact"


def test_the_rule_does_not_approve_a_pair_judges_turned_down_in_another_version() -> None:
    """An adverse verdict that no longer applies still means: this pair needs a look, not a rule."""
    cands = candidates()
    cut = make_pair(crop=[0, 0, 10, 10], decision="accept", reasons=[])
    bad = result(record("photo_has_extra", **SEEN_MORE), record("photo_has_extra", **SEEN_MORE))
    counts = adjudicate.apply_verdicts([cut], {cut["pair_id"]: [bad]}, cands, {}, set())
    assert cut["decision"] == "review" and "earlier_verdict_on_other_version" in cut["reasons"]
    assert "approval" not in cut and counts["accept->review:earlier_verdict_on_other_version"] == 1
    fine = make_pair(crop=[0, 0, 10, 10], decision="accept", reasons=[])
    adjudicate.apply_verdicts([fine], {fine["pair_id"]: [result(record())]}, cands, {}, set())
    assert fine["decision"] == "accept" and fine["approval"]["by"] == "rule"


def test_every_audit_of_a_pair_counts() -> None:
    """A later judge who finds the pair complete does not erase two earlier judges who did not."""
    cands = candidates()
    bad = record("photo_has_extra", **SEEN_MORE)
    pair = make_pair()
    verdicts = {pair["pair_id"]: [result(bad, dict(bad), audit="rev01"), result(record(), audit="rev02")]}
    adjudicate.apply_verdicts([pair], verdicts, cands, {}, set())
    assert pair["decision"] == "review" and "audit_hold:photo_has_extra" in pair["reasons"]
    assert sorted(pair["audit"]["judges"]) == ["rev01:A", "rev01:B", "rev02:A"] and "approval" not in pair
    both = make_pair()
    verdicts = {both["pair_id"]: [result(record(), audit="rev01"), result(record(), audit="rev02")]}
    adjudicate.apply_verdicts([both], verdicts, cands, {}, set())
    assert both["decision"] == "accept" and both["approval"]["judges"] == ["rev01:A", "rev02:A"]


def test_later_judges_must_answer_what_earlier_judges_saw() -> None:
    """Second-contract judges take the first's place, but an approval that ignores what the first saw is held."""
    cands = candidates()
    earlier = result(record("photo_has_extra", **SEEN_MORE), audit="rev01")
    pair = make_pair(open_lines=[open_after()])
    key = second_contract(pair, cands)
    nothing = result(answered("none"), answered("none"), audit="v2a", **key)
    adjudicate.apply_verdicts([pair], {pair["pair_id"]: [earlier, nothing]}, cands, {}, set())
    assert pair["decision"] == "review" and "audit_hold:earlier_judges_differ" in pair["reasons"]
    assert pair["lines"] == LINES and "settled" not in pair["open_lines"][0] and "approval" not in pair
    pair = make_pair(open_lines=[open_after()])
    found = result(answered("all"), answered("all"), audit="v2a", **key)
    adjudicate.apply_verdicts([pair], {pair["pair_id"]: [earlier, found]}, cands, {}, set())
    assert pair["decision"] == "accept" and pair["lines"] == LINES + MORE
    assert pair["flags"] == ["audit_lines_restored"] and pair["approval"]["contract"] == "v2"
    restored = build_rows.final_answer(pair, cands, {}, set())
    assert approval.approval_state(pair, build_rows.pair_fingerprint(pair, restored, cands)) is None
    words = result(record(supplied_text="minor"), audit="rev01")     # one later judge approves nothing alone
    plain = make_pair()
    key = second_contract(plain, cands)
    adjudicate.apply_verdicts([plain], {plain["pair_id"]: [words, result(record(), audit="v2a", **key)]},
                              cands, {}, set())
    assert plain["decision"] == "review" and "audit_hold:one_judge_complete" in plain["reasons"]
    assert adjudicate.conflict_with_earlier(words, "audit_complete", 1)
    assert not adjudicate.conflict_with_earlier(words, "audit_complete", 2)
    plain = make_pair()
    two = result(record(), record(), audit="v2a", **key)             # two judges who looked for restored words do
    adjudicate.apply_verdicts([plain], {plain["pair_id"]: [words, two]}, cands, {}, set())
    assert plain["decision"] == "accept" and plain["approval"]["answer_seen"] is True


def test_first_contract_approval_does_not_cover_an_answer_the_text_rules_shortened() -> None:
    """Those judges saw the assembled lines; when cleaning then drops a line, nobody approved the answer."""
    doubled = [LINES[0], LINES[0]] + LINES[1:]
    cands = candidates(doubled)
    pair = make_pair(lines=doubled)
    verdict = {**result(record(), record()), "lines": doubled}
    adjudicate.apply_verdicts([pair], {pair["pair_id"]: [verdict]}, cands, {}, set())
    assert pair["decision"] == "review" and "answer_changed_by_cleaning" in pair["reasons"]
    assert "approval" not in pair


def test_a_change_the_judges_prescribed_is_kept_only_with_its_approval() -> None:
    """Lines put back whose answer then cannot be approved leave the pair as the assembly made it."""
    again = [LINES[-1]] + MORE[1:]                                    # the first line to put back repeats the last
    cands = candidates(LINES + again)
    pair = make_pair(open_lines=[open_after(lines=again)])
    key = second_contract(pair, cands)
    verdict = result(answered("all"), answered("all"), audit="v2a", **key)
    counts = adjudicate.apply_verdicts([pair], {pair["pair_id"]: [verdict]}, cands, {}, set())
    assert pair["decision"] == "review" and "answer_changed_by_cleaning" in pair["reasons"]
    assert pair["lines"] == LINES and "settled" not in pair["open_lines"][0] and "approval" not in pair
    assert pair["audit"]["note"] == "audit_lines_restored" and counts["accept->review:answer_changed_by_cleaning"] == 1


def test_a_line_in_two_accepted_pairs_holds_both() -> None:
    """Two photographs, or the two pages of one, cannot both show the same line of the edition."""
    cands = candidates()
    for other in ({"pair_id": "r1|asm|bbbbbbbbbbbb", "image_sha256": "b" * 64},
                  {"pair_id": "r1|asm|aaaaaaaaaaaa|l"}):               # another photograph; the other half of this one
        first = make_pair(lines=LINES[:5], decision="accept", reasons=[])
        second = make_pair(lines=LINES[4:], decision="accept", reasons=[], **other)
        second["line_refs"] = [["p1", i] for i in range(4, 10)]
        second["line_indices"] = list(range(4, 10))
        counts = adjudicate.apply_verdicts([first, second], {}, cands, {}, set())
        assert first["decision"] == second["decision"] == "review", other
        assert counts["accept->review:line_in_two_pairs"] == 2
        assert "approval" not in first and "line_in_two_pairs" in second["reasons"]


def test_the_strict_rate_counts_only_pairs_complete_without_a_concern() -> None:
    """What a release audit's target is measured on."""
    results = [result(record()), result(record(), record(uncovered_writing="minor")),
               result(record("photo_has_extra", uncovered_writing="substantial")), result(record("cannot_judge")),
               result({"item_id": "x_001", "verdict": "complete_pair"})]
    assert adjudicate.strict_rate(results) == (4, 1)


def module_at(name: str, path: Path) -> ModuleType:
    """Import a script by path without running its command line.

    :param name: Module name to register.
    :param path: Source file.
    :return: The module.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not (AUDIT_DIR / "score.py").exists(), reason="the audit scripts live in the unversioned logs/")
def test_the_audit_scripts_hold_a_pair_the_judges_disagree_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The integrity audit's regression, through the real scripts: B's verdict never borrows A's exception.

    :param tmp_path: Isolated audit inputs and outputs.
    :param monkeypatch: Redirects the scripts' folders and arguments.
    """
    score = module_at("test_audit_score", AUDIT_DIR / "score.py")
    apply = module_at("test_audit_apply", AUDIT_DIR / "apply_audit.py")
    keys = tmp_path / "audit1119_key"
    keys.mkdir()
    key = {"item_id": "case_001", "pair_id": "p", "stratum": "assemble_1p_one|ja", "decision": "review", "letters": 100}
    (keys / "case_key.jsonl").write_text(json.dumps(key) + "\n")
    for judge, verdict, kind in [("A", "complete_pair", "marginal_or_other_hand"),
                                 ("B", "photo_has_extra", "main_text_lines")]:
        item = {"item_id": "case_001", "verdict": verdict, "uncovered_kind": kind, "confidence": 4}
        (tmp_path / f"judge{judge}_case_batch01.json").write_text(json.dumps([item]))
    monkeypatch.setattr(score, "HERE", tmp_path)
    monkeypatch.setattr(score, "SCRATCH", tmp_path)
    monkeypatch.setattr(sys, "argv", ["score.py", "--name", "case"])
    score.main()
    verdict = json.loads((tmp_path / "audit_case_results.jsonl").read_text())
    assert verdict["records"]["B"]["uncovered_kind"] == "main_text_lines"
    assert apply.decide({"lines": ["a"] * 10}, verdict) == ("review", "audit_hold:photo_has_extra")
    monkeypatch.setattr(sys, "argv", ["score.py", "--name", "case", "--require-complete", "0.95"])
    with pytest.raises(SystemExit):
        score.main()
