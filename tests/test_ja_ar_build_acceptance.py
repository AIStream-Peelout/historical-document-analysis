# File name: test_ja_ar_build_acceptance.py
# Date: 11/12/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for the gate v2 acceptance harness: fixture scoring per remedy class, planted scoring per class, planting on a
synthetic pool, CLI. Synthetic data only (random letter strings), no edition text."""
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

from src.finetuning.qwen_hebrew.ja_ar_build import acceptance as acc

HEB = "אבגדהוזחטיכלמנסעפצקרשת"
ARA = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
P = "p" * 64          # audited photo
Q = "q" * 64          # another face
D = "d" * 64          # same-face copy of P


def fake_line(rng: random.Random, alphabet: str = HEB, words: int = 5) -> str:
    """A pseudo line of random words.

    :param rng: Generator.
    :param alphabet: Letters to draw from.
    :param words: Words per line.
    :return: The line.
    """
    return " ".join("".join(rng.choice(alphabet) for _ in range(rng.randint(3, 6))) for _ in range(words))


def make_fixture(remedy: str, truth: Sequence[str], item_id: str = "X001", side: str = "recto",
                 acceptable: Optional[List[str]] = None, **expected: Any) -> Dict[str, Any]:
    """A fixture record with the fields the scorer reads.

    :param remedy: Remedy class.
    :param truth: Per-line truth.
    :param item_id: Item id.
    :param side: Side key.
    :param acceptable: Acceptable outcomes (default: the remedy's pass outcome).
    :param expected: Extra ``expected`` fields.
    :return: Fixture.
    """
    exp = {"acceptable": acceptable if acceptable is not None else
           ([acc.PASS_OUTCOME[remedy]] if remedy in acc.PASS_OUTCOME else []),
           "same_face_photos": [], "siblings_not_on_photo": [], "rotation": None, "photo_should_get": None,
           "line_check": True, "text": "synthetic"}
    exp.update(expected)
    return {"item_id": item_id, "stratum": "S", "row_id": "r:1", "side_key": side, "document_group": "g:1",
            "remedy": remedy, "remedy_judgeB": None, "confidence": 5, "image_sha256": P, "line_truth": list(truth),
            "expected": exp}


def rec(decision: str, sha: Optional[str] = P, side: str = "recto", row: str = "r:1", remedies: Sequence[str] = (),
        line_indices: Optional[List[int]] = None, **kw: Any) -> Dict[str, Any]:
    """A gate pair record (spec §4 fields used by the scorer).

    :param decision: accept / review / reject.
    :param sha: Photograph (None for a side-level reject).
    :param side: Side key.
    :param row: Row id.
    :param remedies: Remedies.
    :param line_indices: Kept lines.
    :param kw: Other fields.
    :return: Record.
    """
    out = {"pair_id": f"{row}|{side}|{(sha or 'none')[:12]}", "row_id": row, "side_key": side, "document_group": "g:1",
           "image_sha256": sha, "duplicate_photos": [], "decision": decision, "remedies": list(remedies),
           "line_indices": line_indices, "trimmed_lines": [], "moved_to": {}, "rotation": 0, "crop": None,
           "flags": [], "reasons": []}
    out.update(kw)
    return out


def score_one(fixture: Dict[str, Any], records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Score one fixture.

    :param fixture: Fixture.
    :param records: Gate records.
    :return: Scored item.
    """
    return acc.score_fixtures([fixture], records)[0]


# --- text helpers ---------------------------------------------------------------------------------------------------

def test_skeleton_folds_finals_and_arabic_forms() -> None:
    """Finals fold, Arabic carriers fold, tatweel, marks, digits and Latin drop."""
    assert acc.skeleton("שלום ך 12 abc [...]") == "שלומכ"
    assert acc.skeleton("أإآٱ ى ة ؤ ئ ـ") == "اااايهوي"
    assert acc.letters("בְּרֵאשִׁית") == 6
    assert acc.script_of("אבג") == "hebrew" and acc.script_of("ابت") == "arabic" and acc.script_of("12") == "none"


def test_parse_and_compress_ranges() -> None:
    """Range strings round-trip through 0-based indices."""
    assert acc.parse_ranges("1-3, 7; 9–10 and 12") == [1, 2, 3, 7, 9, 10, 12]
    assert acc.parse_ranges("") == []
    assert acc.compress_ranges([0, 1, 2, 6, 8, 9]) == "1-3,7,9-10"
    assert acc.compress_ranges([]) == "-"


def test_judge_line_map_skips_blank_split_lines() -> None:
    """Judges number the split text; the gate indexes non-blank lines."""
    mapping = acc.judge_line_map("a\n\nb\nc", ["a", "b", "c"])
    assert mapping == {1: 0, 2: None, 3: 1, 4: 2}
    assert acc.line_truth(mapping, off=[4], part=[1]) == ["part", "on", "off"]
    with pytest.raises(ValueError):
        acc.line_truth(mapping, off=[9])
    with pytest.raises(ValueError):
        acc.judge_line_map("a\nb", ["a", "x"])


def test_remedy_class_rules() -> None:
    """Remedy classes follow the audit's rule."""
    def j(verdict: str, best: str = "T1", on: str = "all") -> Dict[str, Any]:
        return {"verdict": verdict, "best_text_id": best, "texts": [{"text_id": "T1", "on_photo": on}]}
    assert acc.remedy_class(j("complete_pair"), "T1") == "keep"
    assert acc.remedy_class(j("complete_pair", best="T2"), "T1") == "re_pair"
    assert acc.remedy_class(j("text_has_extra", on="part"), "T1") == "trim"
    assert acc.remedy_class(j("text_has_extra", on="none"), "T1") == "re_pair"
    assert acc.remedy_class(j("photo_has_extra", on="all"), "T1") == "crop"
    assert acc.remedy_class(j("both_partial", on="part"), "T1") == "trim_and_crop"
    assert acc.remedy_class(j("wrong_text", on="none"), "T1") == "drop"
    assert acc.remedy_class(j("cannot_judge", on="part"), "T1") == "unsure"


# --- record parsing -------------------------------------------------------------------------------------------------

def test_record_parsing_accepts_spec_and_implementation_shapes() -> None:
    """line_refs of a merged record, trimmed_lines keyed by side id, moved_to with pairs."""
    merged = rec("accept", side="recto", side_keys=["recto", "verso"], line_indices=None,
                 line_refs=[["verso", 0], ["recto", 2], ["recto", 3]],
                 trimmed_lines={"r:1|recto": [0, 1]}, moved_to={"face2": [["recto", 0]]})
    assert acc.record_side_keys(merged) == ["recto", "verso"]
    assert acc.kept_lines_for(merged, "recto") == [2, 3]
    assert acc.kept_lines_for(merged, "verso") == [0]
    assert acc.trimmed_lines_for(merged, "recto") == [0, 1]
    assert acc.moved_lines_of(merged) == {"face2": [0]}
    flat = rec("accept", line_indices=[1, 2], trimmed_lines=[0])
    assert acc.kept_lines_for(flat, "recto") == [1, 2] and acc.trimmed_lines_for(flat, "recto") == [0]
    assert acc.kept_lines_for(rec("accept", side_keys=["recto", "verso"]), "recto") is None


def test_classify_record_trim_and_crop_signals() -> None:
    """Trim from remedies, from trimmed lines, or from a dropped off-line; crop from a box or the remedy."""
    assert acc.classify_record(rec("accept", line_indices=[0, 1]), "recto") == "accept_as_is"
    assert acc.classify_record(rec("accept", remedies=["trim"]), "recto") == "accept_trim"
    assert acc.classify_record(rec("accept", trimmed_lines={"r:1|recto": [2]}), "recto") == "accept_trim"
    assert acc.classify_record(rec("accept", line_indices=[0, 1]), "recto", off_lines=[2]) == "accept_trim"
    assert acc.classify_record(rec("accept", crop=[0, 0, 5, 5]), "recto") == "accept_crop"
    assert acc.classify_record(rec("accept", remedies=["trim", "crop"]), "recto") == "accept_trim_crop"
    assert acc.classify_record(rec("review"), "recto") == "review"
    assert acc.classify_record(rec("reject", sha=None), "recto") == "reject"


# --- fixture scoring, one test per remedy class ---------------------------------------------------------------------

def test_keep_fixture_outcomes() -> None:
    """keep: as-is pass, trim mismatch, review loss, moved loss, missing."""
    fx = make_fixture("keep", ["on"] * 4)
    assert score_one(fx, [rec("accept", line_indices=[0, 1, 2, 3])])["verdict"] == "pass"
    s = score_one(fx, [rec("accept", remedies=["trim"], line_indices=[0, 1, 2])])
    assert (s["verdict"], s["kind"], s["line_check"]["status"], s["line_check"]["lost_on"]) == \
        ("mismatch", "over_trim", "lost_on", [3])
    assert score_one(fx, [rec("review")])["verdict"] == "loss"
    s = score_one(fx, [rec("accept", sha=Q)])
    assert (s["outcome"], s["verdict"], s["kind"]) == ("elsewhere", "loss", "moved_away")
    assert score_one(fx, [rec("reject", sha=None)])["outcome"] == "reject"
    assert score_one(fx, [])["verdict"] == "missing"


def test_keep_matches_same_face_copy_and_duplicates() -> None:
    """A same-face copy or a duplicate_photos entry counts as the audited photo."""
    fx = make_fixture("keep", ["on"] * 2, same_face_photos=[D])
    assert score_one(fx, [rec("accept", sha=D, line_indices=[0, 1])])["verdict"] == "pass"
    s = score_one(make_fixture("keep", ["on"] * 2), [rec("accept", sha=Q, duplicate_photos=[P], line_indices=[0, 1])])
    assert (s["verdict"], s["match"]) == ("pass", "duplicate")


def test_keep_rotation_and_alternative_crop() -> None:
    """Rotation the judges asked for is checked; an extra acceptable outcome passes."""
    fx = make_fixture("keep", ["on"] * 2, rotation=180)
    assert score_one(fx, [rec("accept", rotation=180, remedies=["rotate180"])])["rotation"]["ok"] is True
    assert score_one(fx, [rec("accept")])["rotation"]["ok"] is False
    fx = make_fixture("keep", ["on"] * 2, acceptable=["accept_as_is", "accept_crop"])
    assert score_one(fx, [rec("accept", crop=[1, 1, 9, 9])])["verdict"] == "pass"


def test_trim_fixture_outcomes() -> None:
    """trim: trim pass (explicit or implied), untrimmed accept harm, reject loss, line check."""
    fx = make_fixture("trim", ["off", "off", "on", "on"])
    s = score_one(fx, [rec("accept", remedies=["trim"], line_indices=[2, 3])])
    assert (s["verdict"], s["line_check"]["status"]) == ("pass", "ok")
    assert score_one(fx, [rec("accept", line_indices=[2, 3])])["verdict"] == "pass"
    s = score_one(fx, [rec("accept", line_indices=[0, 1, 2, 3])])
    assert (s["verdict"], s["kind"]) == ("harm", "accepted_untrimmed")
    s = score_one(fx, [rec("accept", remedies=["trim"], line_indices=[1, 2])])
    assert s["line_check"]["status"] == "kept_off+lost_on" and s["verdict"] == "pass" and not s["strict_pass"]
    summ = acc.summarize_fixtures([s])
    assert summ["pass_kept_off"] == ["X001"] and summ["pass_lost_on"] == ["X001"] and summ["strict_pass"] == []
    assert score_one(fx, [rec("reject", sha=None)])["verdict"] == "loss"
    assert score_one(fx, [rec("accept", remedies=["trim", "crop"])])["kind"] == "over_crop"


def test_trim_flattened_fixture_has_no_line_check() -> None:
    """A flattened fixture is scored on the class only."""
    fx = make_fixture("trim", ["part"], line_check=False)
    s = score_one(fx, [rec("accept", remedies=["reline", "trim"], line_indices=[0, 1, 2])])
    assert (s["verdict"], s["line_check"]["status"]) == ("pass", "n/a")


def test_crop_fixture_outcomes() -> None:
    """crop: crop pass, uncropped accept mismatch, trim-instead mismatch, review loss, merge alternative."""
    fx = make_fixture("crop", ["on"] * 3)
    assert score_one(fx, [rec("accept", crop=[0, 0, 10, 10])])["verdict"] == "pass"
    assert score_one(fx, [rec("accept")])["kind"] == "crop_missed"
    assert score_one(fx, [rec("accept", remedies=["trim"])])["kind"] == "trim_instead_of_crop"
    assert score_one(fx, [rec("review")])["verdict"] == "loss"
    fx = make_fixture("crop", ["on"] * 3, acceptable=["accept_crop", "accept_as_is+merge"],
                      merge_partners={"verso": [0]})
    merged = rec("accept", remedies=["merge"], side_keys=["recto", "verso"],
                 line_refs=[["recto", 0], ["recto", 1], ["recto", 2], ["verso", 0]])
    assert score_one(fx, [merged])["verdict"] == "pass"
    assert score_one(fx, [rec("accept")])["verdict"] == "mismatch"


def test_trim_and_crop_fixture_outcomes() -> None:
    """trim_and_crop: both pass, trim only = crop missed, crop only = harm (untrimmed text)."""
    fx = make_fixture("trim_and_crop", ["on", "off"])
    assert score_one(fx, [rec("accept", remedies=["trim", "crop"], line_indices=[0])])["verdict"] == "pass"
    assert score_one(fx, [rec("accept", remedies=["trim"], line_indices=[0])])["kind"] == "crop_missed"
    assert score_one(fx, [rec("accept", crop=[0, 0, 4, 4], line_indices=[0, 1])])["verdict"] == "harm"
    assert score_one(fx, [rec("review")])["verdict"] == "loss"


def test_re_pair_fixture_outcomes() -> None:
    """re_pair: accept on the photo is harm; moved text or the photo's own text passes; nothing = not repaired."""
    fx = make_fixture("re_pair", ["off"] * 3, side="verso",
                      photo_should_get={"row_id": "r:1", "side_key": "recto"}, siblings_not_on_photo=[])
    assert score_one(fx, [rec("accept", side="verso")])["verdict"] == "harm"
    s = score_one(fx, [rec("accept", side="verso", sha=Q)])
    assert (s["verdict"], s["kind"]) == ("pass", "text_moved")
    s = score_one(fx, [rec("reject", side="verso", sha=None), rec("accept", side="recto")])
    assert (s["verdict"], s["kind"], s["photo_got_expected_text"]) == ("pass", "photo_got_other_text", True)
    s = score_one(fx, [rec("reject", side="verso", sha=None), rec("accept", side="other")])
    assert (s["verdict"], s["photo_got_expected_text"]) == ("not_repaired", False)
    s = score_one(fx, [rec("review", side="verso")])
    assert (s["verdict"], s["kind"]) == ("not_repaired", "review")
    assert score_one(fx, [])["verdict"] == "missing"


def test_drop_and_unsure_fixture_outcomes() -> None:
    """drop: any accept on the photo is harm; unsure is never scored."""
    fx = make_fixture("drop", ["off"])
    assert score_one(fx, [rec("accept", remedies=["trim"])])["verdict"] == "harm"
    assert score_one(fx, [rec("reject", sha=None)])["verdict"] == "pass"
    assert score_one(fx, [rec("accept", sha=Q)])["verdict"] == "pass"
    assert score_one(make_fixture("unsure", ["unknown"]), [rec("accept")])["verdict"] == "unscored"


def test_sibling_text_accepted_on_photo_is_flagged() -> None:
    """Another text the judges saw was not on the photo, accepted there, is reported (a merge without its lines is
    not)."""
    fx = make_fixture("keep", ["on"], siblings_not_on_photo=["verso"])
    s = score_one(fx, [rec("accept"), rec("accept", side="verso", line_indices=[0, 1])])
    assert [a["side_key"] for a in s["sibling_accepts"]] == ["verso"] and s["verdict"] == "pass"
    assert s["sibling_accepts"][0]["kept_lines"] == 2
    empty = rec("accept", side_keys=["recto", "verso"], line_refs=[["recto", 0]])
    assert score_one(fx, [empty])["sibling_accepts"] == []


def test_merged_record_trim_and_wrong_text_per_side() -> None:
    """In a merged record the other side's trim does not trim this side; merged-in lines of a text that is not on
    the photo make the accept harmful."""
    fx = make_fixture("keep", ["on"] * 2)
    merged = rec("accept", remedies=["trim", "merge"], side_keys=["recto", "verso"],
                 line_refs=[["recto", 0], ["recto", 1], ["verso", 3]], trimmed_lines={"r:1|verso": [0, 1, 2]})
    s = score_one(fx, [merged])
    assert (s["outcome"], s["verdict"]) == ("accept_as_is", "pass")
    short = rec("accept", remedies=["merge"], side_keys=["recto", "verso"], line_refs=[["recto", 0], ["verso", 0]])
    assert score_one(fx, [short])["outcome"] == "accept_trim"
    fx = make_fixture("keep", ["on"] * 2, siblings_not_on_photo=["verso"])
    s = score_one(fx, [merged])
    assert (s["verdict"], s["kind"], s["merged_text_not_on_photo"]) == ("harm", "merged_text_not_on_photo", ["verso"])
    fx = make_fixture("keep", ["on"] * 2, duplicate_editions=["verso"])
    assert score_one(fx, [merged])["kind"] == "merged_duplicate_edition"
    assert score_one(make_fixture("keep", ["on"] * 2), [merged])["merged_unverified"] == ["verso"]


def test_merge_alternative_needs_allowed_partner_lines() -> None:
    """``accept_as_is+merge`` passes only when the merged partner keeps lines the fixture allows."""
    fx = make_fixture("crop", ["on"] * 2, acceptable=["accept_crop", "accept_as_is+merge"],
                      merge_partners={"verso": [0, 1]})
    good = rec("accept", remedies=["merge"], side_keys=["recto", "verso"],
               line_refs=[["verso", 0], ["recto", 0], ["recto", 1]])
    bad = rec("accept", remedies=["merge"], side_keys=["recto", "verso"],
              line_refs=[["verso", 0], ["verso", 5], ["recto", 0], ["recto", 1]])
    assert score_one(fx, [good])["verdict"] == "pass"
    assert (score_one(fx, [bad])["verdict"], score_one(fx, [bad])["kind"]) == ("mismatch", "crop_missed")


def test_summary_harm_and_loss_follow_spec() -> None:
    """Harm and loss lists, spec subsets and the agreement matrix."""
    fixtures = [make_fixture("trim", ["off", "on"], item_id="T1"), make_fixture("drop", ["off"], item_id="D1"),
                make_fixture("keep", ["on"], item_id="K1"), make_fixture("trim_and_crop", ["on", "off"], item_id="C1")]
    fixtures[1]["row_id"] = "r:2"
    fixtures[2]["row_id"] = "r:3"
    fixtures[3]["row_id"] = "r:4"
    records = [rec("accept", line_indices=[0, 1]), rec("accept", row="r:2"), rec("review", row="r:3"),
               rec("accept", row="r:4", crop=[0, 0, 1, 1], line_indices=[0, 1])]
    summ = acc.summarize_fixtures(acc.score_fixtures(fixtures, records))
    assert summ["harm"] == ["T1", "D1", "C1"] and summ["harm_spec"] == ["T1", "D1"]
    assert summ["loss"] == ["K1"] and summ["loss_spec"] == ["K1"]
    assert summ["matrix"]["trim"] == {"accept_as_is": 1} and summ["matrix"]["keep"] == {"review": 1}


# --- planted scoring, one test per class ----------------------------------------------------------------------------

def entry(cls: str, line_photos: List[List[str]], permutation: Optional[List[int]] = None) -> Dict[str, Any]:
    """A manifest entry.

    :param cls: Planted class.
    :param line_photos: Photographs per line.
    :param permutation: Shuffle permutation.
    :return: Entry.
    """
    return {"row_id": f"planted_{cls}:001", "class": cls, "side_key": "recto", "n_lines": len(line_photos),
            "line_photos": line_photos, "permutation": permutation}


def planted(cls: str, decision: str, sha: Optional[str] = P, line_indices: Optional[List[int]] = None,
            **kw: Any) -> Dict[str, Any]:
    """A gate record of a planted row.

    :param cls: Planted class.
    :param decision: Decision.
    :param sha: Photograph.
    :param line_indices: Kept lines.
    :param kw: Other fields.
    :return: Record.
    """
    return rec(decision, sha=sha, row=f"planted_{cls}:001", line_indices=line_indices, **kw)


@pytest.mark.parametrize("cls", ["cross_doc", "same_genre"])
def test_planted_swaps(cls: str) -> None:
    """Swapped texts: any accept is a false accept; review / reject / missing are not."""
    e = entry(cls, [[], [], []])
    assert acc.score_planted_record(e, [planted(cls, "accept", line_indices=[0, 2])])["outcome"] == "false_accept"
    assert acc.score_planted_record(e, [planted(cls, "accept")])["accepts"][0]["assumed_all"] is True
    assert acc.score_planted_record(e, [planted(cls, "review")])["outcome"] == "review"
    assert acc.score_planted_record(e, [planted(cls, "reject", sha=None)])["outcome"] == "reject"
    assert acc.score_planted_record(e, [])["outcome"] == "missing"


def test_planted_other_face_append() -> None:
    """Appended other-face lines must be trimmed on this face or accepted only on the other face."""
    e = entry("other_face_append", [[P], [P], [Q], [Q]])
    ok = [planted("other_face_append", "accept", line_indices=[0, 1], remedies=["trim"], moved_to={Q: [2, 3]}),
          planted("other_face_append", "accept", sha=Q, line_indices=[2, 3])]
    assert acc.score_planted_record(e, ok)["outcome"] == "trimmed_ok"
    bad = [planted("other_face_append", "accept", line_indices=[0, 1, 2, 3])]
    s = acc.score_planted_record(e, bad)
    assert s["outcome"] == "false_accept" and s["accepts"][0]["wrong_lines"] == [2, 3]
    wrong_face = [planted("other_face_append", "accept", sha=Q, line_indices=[0, 2])]
    assert acc.score_planted_record(e, wrong_face)["outcome"] == "false_accept"


def test_planted_concat_docs() -> None:
    """Lines of the appended unrelated document belong to no photo."""
    e = entry("concat_docs", [[P], [P], [], []])
    assert acc.score_planted_record(e, [planted("concat_docs", "accept", line_indices=[0, 1])])["outcome"] == \
        "trimmed_ok"
    assert acc.score_planted_record(e, [planted("concat_docs", "accept", line_indices=[0, 1, 3])])["outcome"] == \
        "false_accept"
    s = acc.score_planted_record(e, [planted("concat_docs", "accept", line_indices=[0, 1, 2], remedies=["reline"])])
    assert s["outcome"] == "false_accept"
    s = acc.score_planted_record(e, [planted("concat_docs", "accept", line_indices=[0, 7])])
    assert s["outcome"] == "unverifiable"


def test_planted_shuffled() -> None:
    """Shuffled lines: planted order accepted = false accept; true order restored or an in-order subset is not."""
    perm = [2, 0, 3, 1]                         # planted line k holds true line perm[k]
    e = entry("shuffled", [[P]] * 4, permutation=perm)
    assert acc.score_planted_record(e, [planted("shuffled", "accept", line_indices=[0, 1, 2, 3])])["outcome"] == \
        "false_accept"
    assert acc.score_planted_record(e, [planted("shuffled", "accept", line_indices=[1, 3, 0, 2])])["outcome"] == \
        "restored"
    assert acc.score_planted_record(e, [planted("shuffled", "accept", line_indices=[1, 0])])["outcome"] == \
        "subset_accept"
    assert acc.score_planted_record(e, [planted("shuffled", "review")])["outcome"] == "review"


def test_planted_rates_per_class() -> None:
    """False-accept rate over the records the gate wrote anything for."""
    scored = [{"class": "cross_doc", "outcome": "false_accept"}, {"class": "cross_doc", "outcome": "reject"},
              {"class": "cross_doc", "outcome": "missing"}, {"class": "shuffled", "outcome": "unverifiable"}]
    rates = acc.planted_rates(scored)
    assert rates["cross_doc"]["fa_rate"] == 0.5 and rates["cross_doc"]["scored"] == 2
    assert rates["shuffled"]["fa_rate"] == 0.0 and rates["shuffled"]["fa_rate_upper"] == 1.0


# --- planting on a synthetic pool -----------------------------------------------------------------------------------

def synthetic_pool(n_docs: int = 24, seed: int = 7) -> Dict[str, Any]:
    """Candidates and stage-2 sides of a synthetic pool: two faces per document, four collections, two formula
    families (each formula line shared by six documents) and four Arabic-script documents.

    :param n_docs: Hebrew-script documents.
    :param seed: Seed.
    :return: ``{candidates, sides, reads}``.
    """
    rng = random.Random(seed)
    formula = {"letter": "אבגד הוזח טיכל מנסע", "deed": "פצקר שתאב גדהו זחטי"}
    cands: Dict[str, Dict[str, Any]] = {}
    sides: List[Dict[str, Any]] = []
    shas = set()
    docs = [(i, HEB) for i in range(n_docs)] + [(n_docs + i, ARA) for i in range(4)]
    for i, alphabet in docs:
        row = f"src:{i}"
        images, targets = [], []
        for face, (side, phash) in enumerate((("recto", "0" * 16), ("verso", "f" * 16))):
            sha = f"{i:02d}{face}".ljust(64, "a" if face == 0 else "b")
            shas.add(sha)
            images.append({"image_id": f"x:{face}", "origin": "x", "url": None, "path": f"/fake/{sha}.jpg",
                           "sha256": sha, "phash": phash, "width": 100, "height": 100, "canvas_index": face,
                           "label": "1r" if face == 0 else "1v", "slip_outcome": None, "status": "present",
                           "note": ""})
            lines = [fake_line(rng, alphabet) for _ in range(8)]
            if alphabet == HEB and face == 0 and i < 12:
                lines[0] = formula["letter" if i < 6 else "deed"]
            targets.append({"side_key": side, "text": "\n".join(lines), "raw_text": "\n".join(lines), "lines": lines,
                            "pointer": {}, "side_marker": None, "letters_hebrew": 0, "letters_arabic": 0})
            sides.append({"row_id": row, "side_key": side, "outcome": "accepted", "reasons": [],
                          "image_sha256": sha})
        cands[row] = {"row_id": row, "source": "src", "document_group": f"g:{i}",
                      "canonical_ids": [f"Coll{i % 4}_Lib_Box_{i}"], "pgpids": [], "ktiv_id": None,
                      "shelfmark": f"Box {i}", "script": "hebrew" if alphabet == HEB else "arabic",
                      "languages_stated": [], "verdict": "novel", "witnesses": [], "unit": "page", "n_letters": 0,
                      "targets": targets, "images": images, "rnl": False, "notes": {}}
    return {"candidates": cands, "sides": sides, "reads": shas}


@pytest.fixture()
def planter(monkeypatch: pytest.MonkeyPatch) -> acc.Planter:
    """A planter over the synthetic pool, with synthetic genre markers and every path present.

    :param monkeypatch: pytest monkeypatch.
    :return: Planter.
    """
    monkeypatch.setattr(acc, "GENRE_MARKERS", {"hebrew": {"letter": frozenset({"אבגד", "הוזח"}),
                                                          "deed": frozenset({"פצקר", "שתאב"})}, "arabic": {}})
    pool = synthetic_pool()
    exists = lambda path: True  # noqa: E731
    donors = acc.build_donors(pool["candidates"], pool["sides"], pool["reads"], set(), exists)
    by_row: Dict[str, List[Dict[str, Any]]] = {}
    for s in pool["sides"]:
        by_row.setdefault(s["row_id"], []).append(s)
    return acc.Planter(pool["candidates"], donors, acc.shingle_stats(donors), by_row, pool["reads"], seed=11,
                       exists=exists)


def test_build_donors_filters() -> None:
    """Reasons, RNL marks, flattened and short targets, missing reads or files are left out."""
    pool = synthetic_pool(n_docs=6)
    c = pool["candidates"]
    c["src:1"]["shelfmark"] = "Yevr.-Arab. I 12"
    c["src:2"]["targets"][0]["lines"] = ["א" * 130] + c["src:2"]["targets"][0]["lines"][1:]
    c["src:3"]["targets"][0]["lines"] = ["אב", "גד"]
    sides = [dict(s, reasons=["spill_ratio_ok"]) if s["row_id"] == "src:4" else s for s in pool["sides"]]
    donors = acc.build_donors(c, sides, pool["reads"] - {c["src:5"]["images"][0]["sha256"]}, {"g:0"},
                              lambda path: True)
    keys = {(d.row_id, d.side_key) for d in donors}
    assert ("src:0", "recto") not in keys and ("src:1", "recto") not in keys
    assert ("src:2", "recto") not in keys and ("src:3", "recto") not in keys and ("src:2", "verso") in keys
    assert not any(k[0] == "src:4" for k in keys) and ("src:5", "recto") not in keys
    assert acc.build_donors(c, sides, pool["reads"], set(), lambda path: False) == []


def test_collection_prefix_and_rnl() -> None:
    """Prefixes from canonical ids or shelf-marks; RNL marks are recognised."""
    assert acc.collection_prefix({"canonical_ids": ["Coll1_Lib_Box_7"]}) == "Coll1_Lib_Box"
    assert acc.collection_prefix({"canonical_ids": ["apd:X_1"], "shelfmark": "P.Vind.inv. A.Ch. 1"}) == \
        "shelfmark:P.Vind"
    assert acc.is_rnl({"shelfmark": "Firkovich II 1"}) and acc.is_rnl({"rnl": True})
    assert not acc.is_rnl({"shelfmark": "T-S 8J1.1"})


def test_no_shuffle_adjacency_breaks_every_succession() -> None:
    """No planted neighbour pair keeps its original succession."""
    rng = random.Random(3)
    for n in range(4, 15):
        perm = acc.no_shuffle_adjacency(n, rng)
        assert sorted(perm) == list(range(n)) and all(b != a + 1 for a, b in zip(perm, perm[1:]))


@pytest.mark.parametrize("cls", acc.PLANTED_CLASSES)
def test_plant_class_records(planter: acc.Planter, cls: str) -> None:
    """Every class plants candidates in stage-1 format with a consistent manifest entry."""
    cands, entries = planter.plant_class(cls, 4, {"ar": 1, "ja": 1, "heb": 2})
    assert len(cands) == len(entries) >= 3
    groups = set()
    for c, e in zip(cands, entries):
        assert c["row_id"].startswith(f"planted_{cls}:") and c["document_group"].startswith(f"planted:{cls}:")
        assert c["verdict"] == "planted" and c["notes"]["planted"]["never_build"] is True
        t = c["targets"][0]
        assert t["lines"] == t["text"].split("\n") and len(e["line_photos"]) == e["n_lines"] == len(t["lines"])
        shas = {im["sha256"] for im in c["images"]}
        assert shas == set(e["photos"]) and set().union(*map(set, e["line_photos"])) <= shas
        if cls in ("cross_doc", "same_genre"):
            assert all(not lp for lp in e["line_photos"]) and e["donors"][0]["row_id"] != e["donors"][1]["row_id"]
            assert e["donors"][0]["collection"] != e["donors"][1]["collection"]
        if cls == "same_genre":
            assert e["selection"]["formula"] >= 2 and e["selection"]["genre"] in ("letter", "deed")
        if cls == "other_face_append":
            assert e["segments"][1]["photos"] and set(e["segments"][1]["photos"]).isdisjoint(e["segments"][0]["photos"])
        if cls == "concat_docs":
            assert e["segments"][1]["photos"] == [] and e["donors"][1]["row_id"] != e["donors"][0]["row_id"]
        if cls == "shuffled":
            assert sorted(e["permutation"]) == list(range(e["n_lines"]))
        if cls != "same_genre":
            assert e["donors"][0]["document_group"] not in groups
        groups.add(e["donors"][0]["document_group"])


def test_plant_class_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same seed plants the same records."""
    def run() -> List[str]:
        monkeypatch.setattr(acc, "GENRE_MARKERS", {"hebrew": {}, "arabic": {}})
        pool = synthetic_pool()
        donors = acc.build_donors(pool["candidates"], pool["sides"], pool["reads"], set(), lambda path: True)
        by_row: Dict[str, List[Dict[str, Any]]] = {}
        for s in pool["sides"]:
            by_row.setdefault(s["row_id"], []).append(s)
        pl = acc.Planter(pool["candidates"], donors, acc.shingle_stats(donors), by_row, pool["reads"], seed=5,
                         exists=lambda path: True)
        return [json.dumps(c, sort_keys=True) for c in pl.plant_class("shuffled", 5, acc.default_quotas(5))[0]]
    assert run() == run()


def test_write_planted_refuses_other_directories(tmp_path: Path) -> None:
    """Planted records go only into a pool_planted directory."""
    with pytest.raises(ValueError):
        acc.write_planted(tmp_path / "pool", {"shuffled": []}, [])
    counts = acc.write_planted(tmp_path / "pool_planted", {"shuffled": [{"row_id": "planted_shuffled:001"}]}, [{}])
    assert counts == {"shuffled.jsonl": 1, acc.MANIFEST_NAME: 1}


# --- CLI ------------------------------------------------------------------------------------------------------------

def test_cli_writes_report_and_json(tmp_path: Path) -> None:
    """End to end on synthetic files: fixtures + pairs, manifest + planted pairs."""
    fixtures = [make_fixture("trim", ["off", "on"]), make_fixture("drop", ["off"], item_id="X002")]
    fixtures[1]["row_id"] = "r:2"
    manifest = [entry("cross_doc", [[], []])]
    paths = {name: tmp_path / name for name in ("fx.jsonl", "pairs.jsonl", "man.jsonl", "planted.jsonl")}
    acc.write_jsonl(paths["fx.jsonl"], fixtures)
    acc.write_jsonl(paths["pairs.jsonl"], [rec("accept", remedies=["trim"], line_indices=[1]),
                                           rec("reject", sha=None, row="r:2"),
                                           rec("accept", row="planted_cross_doc:009")])
    acc.write_jsonl(paths["man.jsonl"], manifest)
    acc.write_jsonl(paths["planted.jsonl"], [planted("cross_doc", "accept", line_indices=[0, 1])])
    out, js = tmp_path / "report.md", tmp_path / "scores.json"
    assert acc.main(["--pairs", str(paths["pairs.jsonl"]), "--planted-pairs", str(paths["planted.jsonl"]),
                     "--fixtures", str(paths["fx.jsonl"]), "--manifest", str(paths["man.jsonl"]), "--out", str(out),
                     "--json", str(js)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "judged remedy x gate outcome" in text and "false accepts per class" in text
    assert "WARNING planted rows in the real pairs file" in text
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert [s["verdict"] for s in payload["fixtures"]] == ["pass", "pass"]
    assert payload["planted_rates"]["cross_doc"]["fa_rate"] == 1.0
    with pytest.raises(SystemExit):
        acc.main(["--out", str(out)])
