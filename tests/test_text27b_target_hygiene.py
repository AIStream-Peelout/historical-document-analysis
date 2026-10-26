# File name: test_text27b_target_hygiene.py
# Date: 10/26/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for text27b ``target_hygiene`` (prep pass 2026-10-26): the sentence-level strip of analysis targets that cite
evidence the input lacks (physical description, numbered cross-references, bibliographic apparatus), the uncertainty
protection, the 120-character floor in ``builder.analysis_evidence_strip`` and the before/after audit. Synthetic
strings only; no real work, shelfmark or holding is named.
"""
import json
from pathlib import Path
from typing import Any, Dict, List

from src.finetuning.text27b import builder as B
from src.finetuning.text27b import rows as R
from src.finetuning.text27b import target_hygiene as H

ARGUMENT = ("The writer asks his partner to sell the flax before the caravan leaves and to send the price with the "
            "next messenger. He complains that the judge has delayed the case of the widow.")


def test_physical_cross_ref_and_biblio_sentences_go() -> None:
    """Dimensions, material, ink, hand, layout, numbered cross-references and citations are removed; argument stays."""
    text = ("Paper about 10 x 33 inches, tapering towards the top. Arabic in cursive Hebrew characters. " + ARGUMENT +
            " This fragment is described in the introduction to our No. XXIII. The verso is blank. "
            "See Smith, Journal of Things 12 (1999), pp. 33-45.")
    out, ops, removed = H.strip_unavailable_evidence(text)
    assert out == ARGUMENT
    assert ops == ["E:physical", "E:cross_ref", "E:biblio"]
    assert [w for w, _ in removed] == ["physical", "physical", "cross_ref", "physical", "biblio"]
    assert H.strip_unavailable_evidence(ARGUMENT) == (ARGUMENT, [], [])
    assert H.strip_unavailable_evidence("") == ("", [], [])


def test_hand_rules_and_idioms() -> None:
    """Handwriting claims go; 'on the other hand', a messenger 'in the hand of', trade goods and lists stay."""
    for s in ("The letter is written in the handwriting of the cantor.", "A later hand added the address.",
              "The verso is in a similar hand to the recto.", "It is in the hand of Ḥalfon the scribe.",
              "Written with gummy ink in a cursive hand."):
        assert H.decide(s) == "physical", s
    for s in ("On the other hand, the price rose in the autumn.", "He sent the money in the hand of the messenger.",
              "They traded flax, silk and cloth in the western ports.", "Scrying with oil, ink, egg and mirrors was common.",
              "He complained about the quality of the paper and the script in the books he bought."):
        assert H.decide(s) is None, s


def test_cross_references_and_exemptions() -> None:
    """Numbered documents / plates / letters of the book are cross-references; line references and MS numbers are not."""
    for s in ("Compare Nos. XLI and XLVIII.", "See doc. 12 for the same merchant.", "The seal is shown on pl. IV.",
              "He is also mentioned in letter P.4.f.", "Similar clauses occur ([no. 4, document of sale]).",
              "השטר נדפס אצל המהדיר, מס' 195."):
        assert H.decide(s) == "cross_ref", s
    for s in ("In line 12 the writer names his brother.", "The treatise of MS No.1751 uses a classical register.",
              "He paid 12 dinars to the courier."):
        assert H.decide(s) is None, s


def test_uncertainty_protects_against_soft_triggers_only() -> None:
    """A layout / form sentence with reading-uncertainty language, a line reference or a quoted word stays; a hard
    trigger removes it anyway; long arguments mentioning a torn letter in passing stay."""
    assert H.decide("The margin is torn and the reading of the last word is uncertain.") is None
    assert H.decide("The margin is torn.") == "physical"
    assert H.decide("In the right margin the partner attests the payment (Margin, l. 2).") is None
    assert H.decide("The reading is uncertain because the ink is faded.") == "physical"
    long_arg = ("This torn letter tells us little about the trade itself, but it shows how the community leader was "
                "seen by his peers and why the merchants turned to him when the governor raised the tax on flax.")
    assert H.decide(long_arg) is None
    assert H.decide("השטר קרוע לכל אורכו.") == "physical"
    assert H.decide("הקטע קרוע והקריאה מסופקת במספר מקומות.") is None
    assert H.is_uncertainty("perhaps the name is to be read differently") and not H.is_uncertainty(ARGUMENT)


def test_units_split_dated_heading() -> None:
    """A dated heading in brackets starts a new unit, so the sentence after it can go alone."""
    text = "Petition to a governor (Sixth century A.H./twelfth century A.D.) The leaf has been trimmed on the left."
    spans = H.units(text)
    assert [text[a:b] for a, b in spans] == ["Petition to a governor (Sixth century A.H./twelfth century A.D.)",
                                             "The leaf has been trimmed on the left."]
    out, ops, _ = H.strip_unavailable_evidence(text)
    assert out == "Petition to a governor (Sixth century A.H./twelfth century A.D.)" and ops == ["E:physical"]


def test_builder_floor_and_info() -> None:
    """``analysis_evidence_strip`` drops a target left under 120 characters and reports counts only."""
    kept, ops, info = B.analysis_evidence_strip("Paper, one leaf. " + ARGUMENT)
    assert kept == ARGUMENT and ops == ["E:physical"]
    assert info == {"removed_by_reason": {"physical": 1}, "chars_before": len("Paper, one leaf. " + ARGUMENT),
                    "chars_after": len(ARGUMENT)}
    short, ops2, info2 = B.analysis_evidence_strip("Paper. The verso is blank. A short note on the date.")
    assert short is None and ops2 == ["E:physical"] and info2["chars_after"] < B.ANALYSIS_EVIDENCE_MIN_CHARS == 120
    assert B.analysis_evidence_strip(ARGUMENT) == (ARGUMENT, [], {"removed_by_reason": {}, "chars_before": len(ARGUMENT),
                                                                  "chars_after": len(ARGUMENT)})


def _row(rid: str, target: str, split: str = "train", family: str = "analyse_document") -> Dict[str, Any]:
    """A minimal family row.

    :param rid: Row id.
    :param target: Assistant text.
    :param split: Split.
    :param family: Family.
    :returns: The row.
    """
    return {"messages": R.make_messages("u", target), "meta": {"row_id": rid, "family": family, "split": split,
                                                               "doc_id": f"d{rid}", "source": "book:x",
                                                               "target_tokens": len(target.split())}}


def test_audit_family_files(tmp_path: Path) -> None:
    """The audit counts dropped / changed / unchanged / new rows, keeps non-analysis rows out and samples changed rows
    (the included ids first) with the units the strip removes from the before text."""
    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    after.mkdir()
    b_rows = [_row("1", "Paper. " + ARGUMENT), _row("2", ARGUMENT), _row("3", "The verso is blank. Short."),
              _row("9", "x", family="translate_en")]
    out1, ops1, _ = H.strip_unavailable_evidence("Paper. " + ARGUMENT)
    a_rows = [_row("1", out1), _row("2", ARGUMENT), _row("4", ARGUMENT + " New.")]
    a_rows[0]["meta"].update({"strip_ops": ops1, "evidence_strip": {"removed_by_reason": {"physical": 1}}})
    R.write_jsonl(before / "book_pairs.jsonl", b_rows)
    R.write_jsonl(after / "book_pairs.jsonl", a_rows)
    report, examples = H.audit_family_files(before, after, ["book_pairs", "missing_file"], n=5, include=["1"])
    f = report["files"]["book_pairs"]
    assert f["rows"] == {"changed": 1, "unchanged": 1, "dropped": 1, "new": 1}
    assert f["before"]["rows"] == 3 and f["after"]["rows"] == 3
    assert f["removed_sentences_by_reason"] == {"physical": 1} and f["ops"] == {"E:physical": 1}
    assert [e["row_id"] for e in examples] == ["1"] and examples[0]["removed"] == [{"reason": "physical", "text": "Paper."}]
    assert report["files"]["missing_file"]["rows"] == {"new": 0}
    out = tmp_path / "audit" / "sample.jsonl"
    assert H.main(["audit", "--before", str(before), "--after", str(after), "--files", "book_pairs",
                   "--out", str(out)]) == 0
    assert len(R.load_jsonl(out)) == 1 and json.loads(out.with_name("analysis_strip_report.json").read_text())["examples"] == 1
