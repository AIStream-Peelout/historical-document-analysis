# File name: test_arabic_bench_coverage.py
# Date: 10/24/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests of the Arabic-script benchmark coverage map, on synthetic documents."""
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import pytest

from src.datasets.evaluations.helper_eval_scripts import arabic_bench_coverage as abc

RECTO = ("بسم الله الرحمن الرحيم هذا ما اشترى الشيخ ابو الفضل بن يوسف من القاضي ابي علي دارا في سوق الوراقين "
         "بمدينة مصر بثمن مبلغه عشرون دينارا")
VERSO = ("وصل كتابك ايها الاخ العزيز وفهمت ما ذكرته من امر القماش والحرير والفلفل الذي حملته الى الاسكندرية "
         "في المركب الكبير")
NULL_TEXTS = [
    "شهد عندي الشهود العدول ان فلانا باع جميع الحصة الشائعة من البستان الكائن بالجيزة مع شجره وبئره وساقيته",
    "اما بعد فان امير المؤمنين امر بان يطلق للتجار ما وجب لهم من الزكاة والخراج عن غلات الضياع بالصعيد",
    "تزوج على بركة الله تعالى وسنة نبيه محمد صلى الله عليه وسلم امراة بكرا بالغا على صداق مبلغه خمسون",
    "يا سيدي ومولاي اطال الله بقاك وادام عزك وتاييدك وسعادتك ونعمتك وكبت اعداءك وحسادك وجعلني فداك",
    "ورد على الديوان كتاب العامل بتنيس يذكر فيه ان المراكب وصلت سالمة وفيها الكتان والصوف والزيت",
]
UNEDITED = ("قال الشاعر في وصف الليل الطويل والنجوم الزاهرة والقمر المنير فوق الجبال العالية والبحار الواسعة "
            "والسحاب الماطر")
HEBREW = "שלום עליכם אחי היקר קיבלתי את מכתבך ושמחתי מאד לשמוע ממך ומשלומך ומשלום כל בני ביתך"
SETTINGS = abc.Settings()
NULLS = [abc.ngram_counts(abc.rasm_skeleton(t), SETTINGS.ngram) for t in NULL_TEXTS]


def _record(sections: Sequence[tuple], images: int = 2, sides: Optional[Sequence[str]] = None, pgp_side: str = "") -> Dict[str, Any]:
    """A benchmark record with the given edition sections and images.

    :param sections: ``(side, text)`` per section.
    :type sections: Sequence[tuple]
    :param images: Number of images.
    :type images: int
    :param sides: Side label per image (default unlabelled).
    :type sides: Optional[Sequence[str]]
    :param pgp_side: PGP's record-level side field.
    :type pgp_side: str
    :return: Record.
    :rtype: Dict[str, Any]
    """
    sides = list(sides or [""] * images)
    return {"id": "doc", "pgp_side": pgp_side,
            "images": [{"file": f"doc__{i}.jpg", "image_index": i, "label": "", "side": sides[i]} for i in range(images)],
            "gt": {"sections": [{"side": side, "lines": [text]} for side, text in sections]}}


def _reads(per_image: Sequence[Optional[str]], readers: int = 3) -> Dict[int, Dict[str, str]]:
    """The same read of each image by several readers (None: no read of that image).

    :param per_image: Text per image.
    :type per_image: Sequence[Optional[str]]
    :param readers: Number of readers.
    :type readers: int
    :return: ``{image_index: {tag: text}}``.
    :rtype: Dict[int, Dict[str, str]]
    """
    return {i: {f"reader{k}": text for k in range(readers)} for i, text in enumerate(per_image) if text is not None}


def _decide(record: Dict[str, Any], reads: Dict[int, Dict[str, str]], digests: Optional[List[Optional[str]]] = None):
    """Run :func:`abc.decide_document` with the synthetic null units and labels taken from the record.

    :param record: Record.
    :type record: Dict[str, Any]
    :param reads: Reads.
    :type reads: Dict[int, Dict[str, str]]
    :param digests: Content digests per image.
    :type digests: Optional[List[Optional[str]]]
    :return: ``(image rows, document row)``.
    """
    sides = [(im["side"], "canvas_label" if im["side"] else "") for im in record["images"]]
    return abc.decide_document(record, sides, reads, NULLS, SETTINGS, None, digests)


# --- text helpers -----------------------------------------------------------------------------------------------------

def test_rasm_skeleton_folds_dots_and_drops_other_scripts():
    assert abc.rasm_skeleton("بتثنيى") == abc.rasm_skeleton("ٮٮٮٮٮٮ") == "ٮٮٮٮٮٮ"
    assert abc.rasm_skeleton("جحخ دذ رز سش صض طظ عغ فق") == "ححح" + "دد" + "رر" + "سس" + "صص" + "طط" + "عع" + "فف"
    assert abc.rasm_skeleton("أإآ ة ؤ ء") == "ااا" + "ه" + "و"
    assert abc.rasm_skeleton("שלום abc 123 [...]") == ""
    assert abc.rasm_skeleton("بِسْمِ") == abc.rasm_skeleton("بسم")          # vowel marks go


def test_pooled_ngrams_take_the_per_gram_maximum_over_reads():
    one = "كتب"            # rasm "كٮٮ": too short for a 4-gram
    assert abc.pooled_ngrams([one], 4) == Counter()
    a, b = "سلام سلام", "سلام"
    pooled = abc.pooled_ngrams([a, b, b], 4)
    assert pooled[abc.rasm_skeleton("سلام")] == 2          # the read with two copies decides, three reads do not add up


def test_clipped_recall_and_chance_correction():
    ref = Counter({"abcd": 2, "bcde": 2})
    assert abc.clipped_recall(ref, Counter({"abcd": 5})) == 0.5
    assert abc.clipped_recall(Counter(), Counter({"abcd": 1})) == 0.0
    assert abc.chance_corrected(0.5, 0.2) == pytest.approx(0.375)
    assert abc.chance_corrected(0.1, 0.2) == 0.0 and abc.chance_corrected(0.5, 1.0) == 0.0


def test_quantile_and_exclusive_ngrams():
    assert abc.quantile([], 0.95) == 0.0
    assert abc.quantile([0.0, 1.0], 0.5) == 0.5 and abc.quantile([3.0, 1.0, 2.0], 1.0) == 3.0
    units = [Counter({"aaaa": 1, "bbbb": 2}), Counter({"bbbb": 1, "cccc": 1})]
    assert abc.exclusive_ngrams(units, 0) == Counter({"aaaa": 1})
    assert abc.exclusive_ngrams(units, 1) == Counter({"cccc": 1})


# --- labels and duplicates --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name, side", [("Halper_359_1r.jpg", "recto"), ("https://host/images/T_S_Misc_29_24_1v.jpg", "verso"),
                                        ("MS_HEB_b_18_23a.jpg", "recto"), ("MS_ARAB_c_56_29b.jpg", "verso"),
                                        ("0001_FL167082537.jpg", ""), ("19642_image.jpg", ""), ("", "")])
def test_side_from_filename(name, side):
    assert abc.side_from_filename(name) == side


def test_image_sides_prefer_the_canvas_label():
    record = _record([("", RECTO)], images=3, sides=["verso", "", ""])
    urls = {"doc__0.jpg": "x/doc_1r.jpg", "doc__1.jpg": "x/doc_1v.jpg", "doc__2.jpg": "x/123_image.jpg"}
    assert abc.image_sides(record, urls) == [("verso", "canvas_label"), ("verso", "filename"), ("", "")]


def test_duplicate_groups_by_bytes_and_by_hash():
    assert abc.duplicate_groups([None, None, None], ["a", "b", "a"], 24) == [[0, 2], [1]]
    far = (1 << 200) - 1                      # 200 bits away from 0
    assert abc.duplicate_groups([0, far, 0b111], [None, None, None], 24) == [[0, 2], [1]]
    assert abc.duplicate_groups([0, far], [None, None], 24) == [[0], [1]]


def test_group_sides_and_conflicts():
    assert abc.group_sides([[0, 2], [1]], ["recto", "verso", ""]) == (["recto", "verso"], False)
    assert abc.group_sides([[0, 1]], ["recto", "verso"]) == ([""], True)


def test_side_units_merge_sections_of_one_side():
    units = abc.side_units([{"side": "recto", "lines": ["a"]}, {"side": "recto", "lines": ["b"]}, {"side": "verso", "lines": ["c"]}])
    assert [(u["side"], u["sections"], u["text"]) for u in units] == [("recto", [0, 1], "a\nb"), ("verso", [2], "c")]


@pytest.mark.parametrize("tags, sides, pgp_side, expected", [
    (["recto"], ["recto", "verso"], "", ("side_label", [[0], []])),
    (["recto", "verso"], ["verso", "recto"], "", ("side_label", [[1], [0]])),
    (["recto"], ["verso"], "", None),                                    # the edition's side is not among the labels
    (["recto"], ["recto", "recto"], "", None),                           # a side labelled twice
    (["recto"], ["recto", ""], "", None),                                # an unlabelled image
    ([""], ["recto", "verso"], "verso", ("pgp_side_label", [[], [0]])),
    ([""], ["recto", "verso"], "", None),
    ([""], ["recto", "verso"], "recto and verso", None),
    (["", "verso"], ["recto", "verso"], "", None),                       # untagged text next to a tagged side
])
def test_label_assignment(tags, sides, pgp_side, expected):
    units = [{"side": t, "sections": [k], "text": ""} for k, t in enumerate(tags)]
    assert abc.label_assignment(units, sides, pgp_side) == expected


# --- reads decisions ---------------------------------------------------------------------------------------------------

def _stats(letters: int = 500, share: float = 1.0) -> Dict[str, Any]:
    """Read statistics for :func:`abc.reads_decisions`.

    :param letters: Pooled script letters.
    :type letters: int
    :param share: Arabic share.
    :type share: float
    :return: Stats dict.
    :rtype: Dict[str, Any]
    """
    return {"read_letters": letters, "arabic_share": share}


def test_reads_decisions_place_hebrew_blank_and_guard():
    decisions = abc.reads_decisions([[0.4, 0.05], [0.0, 0.3], [0.0, 0.0], [0.0, 0.0], [0.2, 0.0]],
                                    [[0.4, 0.05], [0.0, 0.3], [0.0, 0.0], [0.0, 0.0], [0.2, 0.0]],
                                    [_stats(), _stats(), _stats(share=0.2), _stats(), _stats(letters=10)], [True, True], SETTINGS)
    assert [(d["status"], d["units"]) for d in decisions] == [("covered", [0]), ("covered", [1]), ("uncovered", []),
                                                             ("uncovered", []), ("blank_or_faint", [])]
    assert decisions[2]["reason"] == "hebrew-script read" and decisions[3]["reason"] == "below floor"


def test_reads_decisions_uncovered_only_when_the_edition_is_placed_elsewhere():
    decisions = abc.reads_decisions([[0.01], [0.02]], [[0.01], [0.02]], [_stats(), _stats()], [True], SETTINGS)
    assert [d["status"] for d in decisions] == ["ambiguous", "ambiguous"]
    nothing_to_place = abc.reads_decisions([[0.0], [0.0]], [[0.0], [0.0]], [_stats(), _stats()], [False], SETTINGS)
    assert [d["status"] for d in nothing_to_place] == ["ambiguous", "ambiguous"]
    middle = abc.reads_decisions([[0.05], [0.5]], [[0.05], [0.5]], [_stats(), _stats()], [True], SETTINGS)
    assert [d["status"] for d in middle] == ["ambiguous", "covered"] and middle[0]["reason"] == "between thresholds"


def test_reads_decisions_ratio_and_several_units():
    shared = abc.reads_decisions([[0.3, 0.2]], [[0.3, 0.01]], [_stats()], [True, True], SETTINGS)
    assert shared[0]["units"] == [0] and shared[0]["reason"] == "one unit above the floor on its own n-grams"   # unit 1 holds only on shared n-grams
    tie = abc.reads_decisions([[0.3, 0.2]], [[0.3, 0.05]], [_stats()], [True, True], SETTINGS)
    assert tie[0]["status"] == "ambiguous" and tie[0]["reason"] == "no unit clears the ratio"
    both = abc.reads_decisions([[0.3, 0.2]], [[0.3, 0.15]], [_stats()], [True, True], SETTINGS)
    assert both[0]["status"] == "covered" and both[0]["units"] == [0, 1]
    elsewhere = abc.reads_decisions([[0.3, 0.2], [0.0, 0.2]], [[0.3, 0.15], [0.0, 0.2]], [_stats(), _stats()], [True, True], SETTINGS)
    assert elsewhere[0]["status"] == "ambiguous"             # unit 1 is as strong on the other image
    short = abc.reads_decisions([[0.3, 0.9]], [[0.3, 0.9]], [_stats()], [True, False], SETTINGS)
    assert short[0]["units"] == [0]                          # a unit too short to place is ignored
    rest = abc.reads_decisions([[0.3, 0.0], [0.0, 0.0]], [[0.3, 0.0], [0.0, 0.0]], [_stats(), _stats()], [True, False], SETTINGS)
    assert rest[1]["status"] == "uncovered" and "too short" in rest[1]["reason"]


def test_reads_check_and_label_contradiction():
    assert abc.reads_check("covered", [0], None) == "no_reads"
    assert abc.reads_check("covered", [0], {"status": "ambiguous", "units": []}) == "inconclusive"
    assert abc.reads_check("covered", [0, 1], {"status": "covered", "units": [1]}) == "agree"
    assert abc.reads_check("uncovered", [], {"status": "covered", "units": [0]}) == "contradict"
    assert abc.labels_contradicted([[0], []], [{"status": "uncovered", "units": []}, {"status": "covered", "units": [0]}])
    assert not abc.labels_contradicted([[0], []], [{"status": "covered", "units": [0]}, {"status": "ambiguous", "units": []}])
    assert not abc.labels_contradicted([[0], []], None)


@pytest.mark.parametrize("statuses, placed, placeable, expected", [
    (["covered", "covered"], [0, 1], [True, True], True),
    (["covered"], [0], [True, True], False),                 # a side of the edition on no image
    (["covered"], [0], [True, False], True),                 # ... unless it is too short to place
    (["covered", "uncovered"], [0], [True], False),
    (["covered", "ambiguous"], [0], [True], None),
    (["blank_or_faint", "ambiguous"], [], [True], False),
])
def test_coverage_match(statuses, placed, placeable, expected):
    assert abc.coverage_match(statuses, placed, placeable) is expected


# --- whole documents ---------------------------------------------------------------------------------------------------

def test_two_unlabelled_images_with_recto_and_verso_are_placed_by_reads():
    rows, doc = _decide(_record([("recto", RECTO), ("verso", VERSO)]), _reads([VERSO, RECTO]))
    assert [(r["method"], r["status"], r["units"], r["sections"]) for r in rows] == [("reads", "covered", [1], [1]), ("reads", "covered", [0], [0])]
    assert rows[0]["corrected_units"][1] > 0.5 and rows[0]["recall_sections"] == rows[0]["recall_units"]
    assert doc["coverage_match"] is True and doc["sides_match"] is True and doc["consistent_with_sides_match"] is True


def test_an_image_of_unedited_arabic_is_uncovered_once_the_edition_is_placed():
    rows, doc = _decide(_record([("", RECTO)]), _reads([RECTO, UNEDITED]))
    assert [(r["status"], r["reason"]) for r in rows] == [("covered", "one unit above the floor"), ("uncovered", "below floor")]
    assert doc["coverage_match"] is False and doc["sides_match"] is False and doc["consistent_with_sides_match"] is True


def test_failed_reads_leave_both_images_ambiguous():
    rows, doc = _decide(_record([("", RECTO)]), _reads([UNEDITED, NULL_TEXTS[4]]))
    assert [r["status"] for r in rows] == ["ambiguous", "ambiguous"] and doc["coverage_match"] is None
    assert doc["consistent_with_sides_match"] is None and doc["units_unplaced"] == [0]


def test_hebrew_and_blank_sides():
    rows, doc = _decide(_record([("", RECTO)], images=3), _reads([RECTO, HEBREW, "NO TEXT"]))
    assert [(r["status"], r["reason"]) for r in rows] == [("covered", "one unit above the floor"), ("uncovered", "hebrew-script read"),
                                                         ("blank_or_faint", "pooled reads under the letter minimum")]
    assert rows[1]["arabic_share"] == 0.0 and rows[1]["hebrew_reads"] == 3 and doc["uncovered"] == 1 and doc["blank_or_faint"] == 1


def test_side_labels_decide_and_are_checked_by_the_reads():
    rows, doc = _decide(_record([("verso", VERSO)], sides=["recto", "verso"]), _reads([UNEDITED, VERSO]))
    assert [(r["method"], r["status"], r["units"], r["reads_check"]) for r in rows] == [
        ("side_label", "uncovered", [], "agree"), ("side_label", "covered", [0], "agree")]
    assert doc["label_conflict"] is None and doc["coverage_match"] is False


def test_labels_contradicted_by_the_reads_fall_through_to_the_reads():
    record = _record([("", RECTO)], sides=["recto", "verso"], pgp_side="recto")
    rows, doc = _decide(record, _reads([UNEDITED, RECTO]))
    assert [(r["method"], r["status"]) for r in rows] == [("reads", "uncovered"), ("reads", "covered")]
    assert doc["label_conflict"] and all(r["label_conflict"] for r in rows)
    rows, _ = _decide(record, {})
    assert [(r["method"], r["status"], r["reads_check"]) for r in rows] == [("pgp_side_label", "covered", "no_reads"),
                                                                            ("pgp_side_label", "uncovered", "no_reads")]


def test_single_image_and_missing_reads():
    rows, doc = _decide(_record([("recto", RECTO)], images=1, sides=["verso"]), {})
    assert [(r["method"], r["status"], r["reads_check"]) for r in rows] == [("single_image", "covered", "no_reads")]
    assert doc["coverage_match"] is True and doc["sides_match"] is True
    rows, _ = _decide(_record([("", RECTO)]), _reads([RECTO, UNEDITED], readers=1))      # one model alone never decides
    assert [(r["method"], r["status"]) for r in rows] == [("no_reads", "ambiguous")] * 2
    rows, _ = _decide(_record([("", RECTO)]), _reads([RECTO, None]))                    # an image without reads
    assert {r["method"] for r in rows} == {"no_reads"}


def test_copies_of_one_photograph_share_a_decision():
    rows, doc = _decide(_record([("recto", RECTO)], images=2), {}, digests=["same", "same"])
    assert [(r["method"], r["status"], r["duplicate_of"]) for r in rows] == [("single_image", "covered", None), ("single_image", "covered", 0)]
    assert doc["distinct_images"] == 1 and doc["duplicates"] == [[0, 1]] and doc["coverage_match"] is True and doc["sides_match"] is False
    rows, doc = _decide(_record([("recto", RECTO), ("verso", VERSO)], sides=["recto", "verso"]), {}, digests=["same", "same"])
    assert doc["label_conflict"] and [r["method"] for r in rows] == ["no_reads", "no_reads"]


def test_summarise_counts_methods_and_consistency():
    rows_a, doc_a = _decide(_record([("recto", RECTO), ("verso", VERSO)]), _reads([RECTO, VERSO]))
    rows_b, doc_b = _decide(_record([("", RECTO)]), _reads([UNEDITED, NULL_TEXTS[4]]))
    doc_b = dict(doc_b, doc_id="doc_b")
    rows_b = [dict(r, doc_id="doc_b") for r in rows_b]
    summary = abc.summarise(rows_a + rows_b, [doc_a, doc_b])
    assert summary["images"] == 4 and summary["method"] == {"reads": {"covered": 2, "ambiguous": 2}}
    assert (summary["consistent"], summary["inconsistent"], summary["undetermined"]) == (1, 0, 1)
    assert abc.summarise(rows_a + rows_b, [doc_a, doc_b], only=["doc_b"])["documents"] == 1


# --- v2: images ---------------------------------------------------------------------------------------------------------

import json
import random

import numpy as np
from PIL import Image, ImageDraw

IMAGE_SETTINGS = abc.ImageSettings()


def _fragment_image(path, text: bool, backing=(30, 70, 160), paper=(200, 180, 140), seed: int = 0) -> None:
    """Write a synthetic photograph: a paper rectangle on a coloured backing, with or without writing.

    :param path: Output file.
    :param text: Draw rows of dark letter-sized strokes on the paper.
    :type text: bool
    :param backing: Backing colour.
    :param paper: Paper colour.
    :param seed: Random seed (texture and strokes).
    :type seed: int
    """
    rng = random.Random(seed)
    image = Image.new("RGB", (600, 800), backing)
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 100, 500, 700), fill=paper)
    if text:
        for row in range(150, 660, 40):
            x = 140
            while x < 460:
                width, height = rng.randint(6, 22), rng.randint(4, 9)
                draw.rectangle((x, row, x + width, row + height), fill=(60, 40, 30))
                x += width + rng.randint(4, 10)
    pixels = np.asarray(image).astype(np.int16)
    noise = np.random.default_rng(seed).integers(-4, 5, size=pixels.shape)
    Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8)).save(path)


def test_ink_separates_a_blank_side_from_a_written_one(tmp_path):
    _fragment_image(tmp_path / "blank.png", text=False)
    _fragment_image(tmp_path / "text.png", text=True)
    blank = abc.analyse_image(tmp_path / "blank.png", IMAGE_SETTINGS)["ink"]
    text = abc.analyse_image(tmp_path / "text.png", IMAGE_SETTINGS)["ink"]
    assert blank["mask_ok"] and text["mask_ok"]
    assert 0.25 < blank["inner_share"] < 0.5                       # the paper, eroded, without the blue backing
    assert blank["ink_permille"] < 1.0 < IMAGE_SETTINGS.blank_max_permille < text["ink_permille"]
    assert abc.ink_class(blank, IMAGE_SETTINGS) == abc.BLANK and abc.ink_class(text, IMAGE_SETTINGS) == "text"
    assert text["line_components"] >= 5


def test_ink_class_rules():
    ok = {"mask_ok": True, "paper_grey": 150.0, "ink_permille": 2.0, "loose_ink_permille": 3.0}
    assert abc.ink_class(ok, IMAGE_SETTINGS) == abc.BLANK
    assert abc.ink_class(dict(ok, ink_permille=9.0), IMAGE_SETTINGS) == "text"
    assert abc.ink_class(dict(ok, loose_ink_permille=30.0), IMAGE_SETTINGS) == "unknown"   # ink outside the fragment mask
    assert abc.ink_class(dict(ok, paper_grey=30.0), IMAGE_SETTINGS) == "unknown"          # the "paper" is the black backing
    assert abc.ink_class({"mask_ok": False}, IMAGE_SETTINGS) == "unknown"


def test_a_black_image_has_no_fragment(tmp_path):
    Image.new("RGB", (400, 400), (0, 0, 0)).save(tmp_path / "black.png")
    analysis = abc.analyse_image(tmp_path / "black.png", IMAGE_SETTINGS)
    assert analysis["ink"]["mask_ok"] is False and analysis["crop_hashes"] is None
    assert abc.ink_class(analysis["ink"], IMAGE_SETTINGS) == "unknown"


def test_transform_hashes_are_rotation_aware_and_mirror_aware():
    rng = np.random.default_rng(3)
    grey = Image.fromarray(rng.integers(0, 255, size=(64, 48)).astype(np.uint8)).resize((96, 128))
    hashes = {k: format(v, "x") for k, v in abc.transform_hashes(grey, 16).items()}
    for turn, name in ((Image.Transpose.ROTATE_90, "90"), (Image.Transpose.ROTATE_180, "180"), (Image.Transpose.ROTATE_270, "270")):
        rotated = {k: format(v, "x") for k, v in abc.transform_hashes(grey.transpose(turn), 16).items()}
        rotation, mirrored, best = abc.hash_distances(rotated, hashes)
        assert rotation <= 4 and mirrored > 60 and not best.startswith("m")
    mirror = {k: format(v, "x") for k, v in abc.transform_hashes(grey.transpose(Image.Transpose.FLIP_LEFT_RIGHT), 16).items()}
    rotation, mirrored, best = abc.hash_distances(mirror, hashes)
    assert mirrored <= 4 and rotation > 60 and best.startswith("m")


def _hashes(rng: random.Random, base: Optional[int] = None, turn: str = "0") -> Dict[str, str]:
    """Fake transform hashes: random, except that ``turn`` equals ``base`` (a rotated or mirrored copy).

    :param rng: Random generator.
    :param base: Hash of the original as it is (None: an unrelated image).
    :param turn: Transform under which this image matches the original.
    :return: ``{transform: hex}``.
    """
    out = {t: format(rng.getrandbits(256), "x") for t in abc.TRANSFORMS}
    if base is not None:
        out[turn] = format(base, "x")
    return out


def _analysis(hashes: Dict[str, str], permille: float = 50.0, crop: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """A fake :func:`abc.analyse_image` result.

    :param hashes: Whole-image hashes.
    :param permille: Ink per mille (also used as the loose value).
    :param crop: Crop hashes.
    :return: Analysis dict.
    """
    return {"hashes": hashes, "crop_hashes": crop,
            "ink": {"mask_ok": True, "paper_grey": 150.0, "ink_permille": permille, "loose_ink_permille": permille, "line_components": 0}}


def test_is_copy_accepts_rotations_and_rejects_mirrors():
    rng = random.Random(1)
    base = rng.getrandbits(256)
    original = _analysis(_hashes(rng, base, "0"))
    upside_down = _analysis(_hashes(rng, base, "180"))
    other_side = _analysis(_hashes(rng, base, "m0"))
    unrelated = _analysis(_hashes(rng))
    assert abc.is_copy(original, upside_down, 24, 10)[0]
    copy, evidence = abc.is_copy(original, other_side, 24, 10)
    assert not copy and evidence["whole"][1] == 0
    assert not abc.is_copy(original, unrelated, 24, 10)[0]
    crop_only = _analysis(_hashes(rng), crop=_hashes(rng, base, "90"))
    assert abc.is_copy(_analysis(_hashes(rng), crop=_hashes(rng, base, "0")), crop_only, 24, 10)[0]   # different framing, same fragment


def test_copy_groups_rotation_override_and_representative():
    rng = random.Random(2)
    base = rng.getrandbits(256)
    analyses = [_analysis(_hashes(rng, base, "0")), _analysis(_hashes(rng)), _analysis(_hashes(rng, base, "180")),
                _analysis(_hashes(rng, base, "m180"))]
    groups, evidence, mirrored = abc.copy_groups(analyses, [None] * 4, {}, SETTINGS)
    assert groups == [[0, 2], [1], [3]] and evidence[2]["by"] == ["hash"] and evidence[2]["whole"][2] == "180"
    assert mirrored == [[0, 3]] or [0, 3] in mirrored
    groups, evidence, _ = abc.copy_groups(analyses, [None] * 4, {0: 2, 1: 2}, SETTINGS)   # override: 0 and 1 copy 2
    assert groups[0] == [2, 0, 1] and evidence[1]["by"] == ["override"]
    groups, _, _ = abc.copy_groups([None, None], ["x", "x"], {}, SETTINGS)
    assert groups == [[0, 1]]


# --- v2: overrides, scripts, decisions ---------------------------------------------------------------------------------

def test_load_overrides_validates(tmp_path):
    path = tmp_path / "o.jsonl"
    good = {"doc_id": "doc", "image_index": 1, "script": "hebrew", "role": "uncovered", "reason": "Hebrew"}
    path.write_text(json.dumps(good) + "\n\n", encoding="utf-8")
    assert abc.load_overrides(path)["doc"][1]["script"] == "hebrew"
    assert abc.load_overrides(tmp_path / "missing.jsonl") == {}
    for bad in (dict(good, role="covered"), dict(good, role="copy_of"), dict(good, script="latin"), dict(good, role="maybe")):
        path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
        with pytest.raises(ValueError):
            abc.load_overrides(path)
    path.write_text(json.dumps(good) + "\n" + json.dumps(good) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        abc.load_overrides(path)


@pytest.mark.parametrize("text, script", [(RECTO, "arabic"), (HEBREW, "hebrew"), (RECTO[:40] + HEBREW[:40], "mixed"), ("123 ...", "none")])
def test_edition_script(text, script):
    assert abc.edition_script(text) == script


def _decide_v2(record: Dict[str, Any], reads: Dict[int, Dict[str, str]], analyses: Optional[List[Dict[str, Any]]] = None,
               overrides: Optional[Dict[int, Dict[str, Any]]] = None):
    """Run :func:`abc.decide_document_v2` with the synthetic null units.

    :param record: Record.
    :param reads: Reads.
    :param analyses: Fake image analyses (default: distinct images with writing).
    :param overrides: Overrides by image index.
    :return: ``(image rows, document row)``.
    """
    rng = random.Random(9)
    n = len(record["images"])
    analyses = analyses or [_analysis(_hashes(rng)) for _ in range(n)]
    sides = [(im["side"], "canvas_label" if im["side"] else "") for im in record["images"]]
    return abc.decide_document_v2(record, sides, reads, NULLS, SETTINGS, IMAGE_SETTINGS, analyses, None, overrides or {})


def _override(index: int, role: str, script: str, units: Optional[List[int]] = None, copy_of: Optional[int] = None) -> Dict[str, Any]:
    """An override row.

    :param index: Image index.
    :param role: Role.
    :param script: Script.
    :param units: Units (covered).
    :param copy_of: Original (copy_of).
    :return: Row.
    """
    row = {"doc_id": "doc", "image_index": index, "role": role, "script": script, "reason": f"by eye {index}", "edited_text_visible": None}
    if units is not None:
        row["units"] = units
    if copy_of is not None:
        row["copy_of"] = copy_of
    return row


def test_v2_rotated_copy_inherits_and_takes_the_best_member_reads():
    rng = random.Random(4)
    base = rng.getrandbits(256)
    analyses = [_analysis(_hashes(rng, base, "0")), _analysis(_hashes(rng, base, "180")), _analysis(_hashes(rng))]
    rows, doc = _decide_v2(_record([("", RECTO)], images=3), _reads([RECTO, NULL_TEXTS[0] + NULL_TEXTS[1], UNEDITED]), analyses)
    assert [(r["role"], r["status"], r["copy_of"]) for r in rows] == [("covered", "covered", None), ("copy_of", "covered", 0),
                                                                     ("uncovered", "uncovered", None)]
    assert rows[0]["evidence_image"] == 0 and doc["copies"] == [[0, 1]]
    assert rows[1]["copy_evidence"]["by"] == ["hash"] and rows[0]["script"] == "arabic" and rows[0]["script_source"] == "edition"
    assert rows[2]["script"] == "unknown"                       # never taken from the reads


def test_v2_blank_image_from_ink_and_single_text_image():
    analyses = [_analysis(_hashes(random.Random(5)), 60.0), _analysis(_hashes(random.Random(6)), 1.0)]
    rows, doc = _decide_v2(_record([("", RECTO)]), {}, analyses)
    assert [(r["method"], r["role"]) for r in rows] == [("single_text_image", "covered"), ("single_text_image", "blank_or_faint")]
    assert rows[1]["script"] == "none" and rows[1]["script_source"] == "ink" and doc["blank_or_faint"] == 1
    rows, _ = _decide_v2(_record([("", RECTO)]), _reads([RECTO, "NO TEXT AT ALL HERE"]), analyses)
    assert rows[1]["role"] == "blank_or_faint"
    rows, _ = _decide_v2(_record([("recto", RECTO), ("verso", VERSO + " " + UNEDITED)], sides=["recto", "verso"]), {}, analyses)
    assert [(r["method"], r["role"], r["ink_conflict"]) for r in rows] == [("side_label", "covered", False), ("side_label", "covered", True)]
    rows, _ = _decide_v2(_record([("", RECTO)], sides=["recto", "verso"], pgp_side="recto"), {}, analyses)
    assert [(r["method"], r["role"]) for r in rows] == [("pgp_side_label", "covered"), ("ink", "blank_or_faint")]


def test_v2_reads_place_only_long_units():
    short = "الى الشيخ ابي علي"                                  # an address: far under 100 rasm letters
    rows, doc = _decide_v2(_record([("recto", RECTO), ("verso", short)]), _reads([RECTO, UNEDITED]))
    assert rows[0]["role"] == "covered" and rows[1]["role"] == "ambiguous"
    assert "too short to place by reads" in rows[1]["reason"] and doc["short_units"] == [1] and doc["units_unplaced"] == [1]
    long_verso = VERSO + " والسلام عليكم ورحمة الله وبركاته"     # 95 + rasm letters: placeable
    rows, doc = _decide_v2(_record([("recto", RECTO), ("verso", long_verso)], images=3), _reads([RECTO, long_verso, UNEDITED]))
    assert abc.Settings().min_place_letters == 100 and [r["role"] for r in rows] == ["covered", "covered", "uncovered"]
    assert doc["short_units"] == [] and rows[1]["units"] == [1]


def test_v2_overrides_take_precedence_and_join_sections():
    overrides = {1: _override(1, "covered", "arabic", units=[0]), 2: _override(2, "uncovered", "hebrew")}
    rows, doc = _decide_v2(_record([("", RECTO + " " + VERSO)], images=3), _reads([RECTO, VERSO, HEBREW]), overrides=overrides)
    assert [(r["role"], r["method"], r["units"]) for r in rows] == [("covered", "reads", [0]), ("covered", "manual", [0]),
                                                                   ("uncovered", "manual", [])]
    assert doc["joined_units"] == {"0": [0, 1]} and rows[0]["joined_with"] == [1] and rows[1]["joined_with"] == [0]
    assert rows[1]["override_reason"] == "by eye 1" and rows[2]["script"] == "hebrew" and rows[2]["script_source"] == "manual"
    assert doc["overrides"] == 2 and doc["units"][0]["images"] == [0, 1]


def test_v2_copy_override_and_errors():
    overrides = {0: _override(0, "copy_of", "arabic", copy_of=1)}
    rows, doc = _decide_v2(_record([("", RECTO)]), _reads([RECTO, RECTO]), overrides=overrides)
    assert [(r["role"], r["copy_of"]) for r in rows] == [("copy_of", 1), ("covered", None)] and doc["copies"] == [[1, 0]]
    assert rows[0]["copy_evidence"]["by"] == ["override"] and rows[0]["script"] == "arabic"
    with pytest.raises(ValueError):
        _decide_v2(_record([("", RECTO)]), {}, overrides={5: _override(5, "uncovered", "hebrew")})
    clash = {0: _override(0, "covered", "arabic", units=[0]), 1: _override(1, "uncovered", "arabic"),
             2: _override(2, "copy_of", "arabic", copy_of=0)}
    clash[1]["copy_of"] = None
    clash = {0: _override(0, "covered", "arabic", units=[0]), 1: _override(1, "uncovered", "arabic"),
             2: _override(2, "copy_of", "arabic", copy_of=0)}
    rng = random.Random(8)
    base = rng.getrandbits(256)
    analyses = [_analysis(_hashes(rng, base, "0")), _analysis(_hashes(rng, base, "180")), _analysis(_hashes(rng))]
    with pytest.raises(ValueError):                              # images 0 and 1 are copies, but their overrides differ
        _decide_v2(_record([("", RECTO)], images=3), {}, analyses, overrides=clash)


def test_v2_summary_and_comparison_with_v1():
    v1_rows, _ = _decide(_record([("", RECTO)], images=3), _reads([RECTO, UNEDITED, RECTO]), digests=[None, None, None])
    rng = random.Random(7)
    base = rng.getrandbits(256)
    analyses = [_analysis(_hashes(rng, base, "0")), _analysis(_hashes(rng), 1.0), _analysis(_hashes(rng, base, "90"))]
    rows, doc = _decide_v2(_record([("", RECTO)], images=3), _reads([RECTO, UNEDITED, RECTO]), analyses)
    assert [r["role"] for r in rows] == ["covered", "blank_or_faint", "copy_of"]
    changes = abc.compare_maps(v1_rows, rows)
    assert [(c["image_index"], c["v1_role"], c["v2_role"]) for c in changes] == [(1, "uncovered", "blank_or_faint"), (2, "covered", "copy_of")]
    assert abc.v1_role({"status": "covered", "duplicate_of": 0}) == "copy_of"
    summary = abc.summarise_v2(rows, [doc])
    assert summary["role"] == {"covered": 1, "blank_or_faint": 1, "copy_of": 1} and summary["copies"] == 1
    assert summary["method"]["single_text_image"] == {"covered": 1, "blank_or_faint": 1, "copy_of": 1}
    assert summary["script_by_role"]["blank_or_faint"] == {"none (ink)": 1}
    assert abc.summarise_v2(rows, [doc], only=["other"])["images"] == 0


def test_blank_calibration_counts():
    analyses = {"b.jpg": _analysis({}, 1.0), "t.jpg": _analysis({}, 40.0), "h.jpg": _analysis({}, 9.0)}
    report = abc.blank_calibration(analyses, IMAGE_SETTINGS, ["b.jpg"], ["t.jpg"], {"h.jpg": "a hard one"})
    assert (report["blank_ok"], report["text_ok"], report["max_blank_permille"], report["min_text_permille"]) == (1, 1, 1.0, 40.0)
    assert report["images"][-1]["note"] == "a hard one" and report["images"][-1]["class"] == "text"
