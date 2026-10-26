# File name: test_score_arabic_covered.py
# Date: 10/24/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests of the coverage-aware Arabic-script benchmark scorer, on synthetic documents."""
import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from src.datasets.evaluations.helper_eval_scripts import score_arabic_benchmark as sab
from src.datasets.evaluations.helper_eval_scripts import score_arabic_covered as sac

RECTO = ("بسم الله الرحمن الرحيم هذا ما اشترى الشيخ ابو الفضل بن يوسف من القاضي ابي علي دارا في سوق الوراقين "
         "بمدينة مصر بثمن مبلغه عشرون دينارا")
VERSO = ("وصل كتابك ايها الاخ العزيز وفهمت ما ذكرته من امر القماش والحرير والفلفل الذي حملته الى الاسكندرية "
         "في المركب الكبير")
MARGIN = "شهد عندي الشهود العدول ان فلانا باع جميع الحصة الشائعة من البستان الكائن بالجيزة"
UNEDITED = ("قال الشاعر في وصف الليل الطويل والنجوم الزاهرة والقمر المنير فوق الجبال العالية والبحار الواسعة "
            "والسحاب الماطر")
HEBREW = "שלום עליכם אחי היקר קיבלתי את מכתבך ושמחתי מאד לשמוע ממך ומשלומך ומשלום כל בני ביתך"
COMPARED = ["gt_letters", "answer_letters", "arabic_share", "status", "ngram_precision", "ngram_recall", "ngram_f1", "ler",
            "length_ratio", "floor_f1"]


def _record(doc_id: str, sections: Sequence[Tuple[str, str]], images: int) -> Dict[str, Any]:
    """A benchmark record.

    :param doc_id: Document id.
    :type doc_id: str
    :param sections: ``(side, text)`` per edition section.
    :type sections: Sequence[Tuple[str, str]]
    :param images: Number of images.
    :type images: int
    :return: Record.
    :rtype: Dict[str, Any]
    """
    return {"id": doc_id, "pgp_side": "",
            "images": [{"file": f"{doc_id}__{i}.jpg", "image_index": i, "label": "", "side": ""} for i in range(images)],
            "gt": {"sections": [{"side": side, "lines": [text]} for side, text in sections]}}


def _coverage(doc_id: str, per_image: Sequence[Tuple[str, Sequence[int]]], duplicate_of: Optional[Dict[int, int]] = None,
              arabic_share: Optional[Sequence[Optional[float]]] = None) -> Dict[int, Dict[str, Any]]:
    """Coverage rows of a document.

    :param doc_id: Document id.
    :type doc_id: str
    :param per_image: ``(status, sections)`` per image.
    :type per_image: Sequence[Tuple[str, Sequence[int]]]
    :param duplicate_of: ``{image: representative}`` for copies.
    :type duplicate_of: Optional[Dict[int, int]]
    :param arabic_share: Pooled-read Arabic share per image (None: no reads).
    :type arabic_share: Optional[Sequence[Optional[float]]]
    :return: ``{image_index: row}``.
    :rtype: Dict[int, Dict[str, Any]]
    """
    duplicate_of = duplicate_of or {}
    shares = list(arabic_share or [1.0] * len(per_image))
    return {i: {"doc_id": doc_id, "image_index": i, "status": status, "sections": list(sections), "method": "side_label",
                "duplicate_of": duplicate_of.get(i), "arabic_share": shares[i]}
            for i, (status, sections) in enumerate(per_image)}


# --- recognition ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sections, pages, per_image", [
    ([("", RECTO)], {0: RECTO[:60] + " خطا "}, [("covered", [0])]),
    ([("recto", RECTO), ("verso", VERSO)], {0: VERSO, 1: RECTO + " زيادة"}, [("covered", [0]), ("covered", [1])]),
    ([("recto", RECTO), ("recto", MARGIN), ("verso", VERSO)], {0: RECTO + MARGIN, 1: VERSO[:50]},
     [("covered", [0, 1]), ("covered", [2])]),
    ([("recto", RECTO), ("verso", VERSO)], {0: RECTO + VERSO}, [("covered", [0, 1])]),
])
def test_every_image_covered_equals_the_scorer(sections, pages, per_image):
    record = _record("doc", sections, len(per_image))
    floor = _record("other", [("", UNEDITED)], 1)
    row, _ = sac.score_covered_document(record, _coverage("doc", per_image), pages, floor)
    expected = sab.score_document(record, pages, floor)
    assert row["full_coverage"] is True
    assert {k: row[k] for k in COMPARED} == {k: expected[k] for k in COMPARED}


def test_uncovered_image_leaves_recognition_and_goes_to_overread():
    record = _record("doc", [("", RECTO)], 2)
    pages = {0: RECTO, 1: UNEDITED}
    row, images = sac.score_covered_document(record, _coverage("doc", [("covered", [0]), ("uncovered", [])]), pages)
    whole = sab.score_document(record, pages)
    assert row["ngram_f1"] == 1.0 and row["ler"] == 0.0 and row["full_coverage"] is False
    assert whole["ngram_f1"] < 1.0 and whole["ler"] > 0.5            # the scorer counts the unedited side as error
    assert row["covered_images"] == 1 and row["overread_images"] == 1 and row["overread_answered"] == 1
    assert row["overread_ge40"] == 1 and row["overread_arabic_letters"] == len(sac.ar.arabic_letters(UNEDITED))
    assert [im["role"] for im in images] == [sac.RECOGNITION, sac.OVERREAD]


def test_reference_is_reduced_to_the_sections_the_covered_images_carry():
    record = _record("doc", [("recto", RECTO), ("verso", VERSO)], 2)
    pages = {0: RECTO, 1: "[...]"}
    row, _ = sac.score_covered_document(record, _coverage("doc", [("covered", [0]), ("ambiguous", [])]), pages)
    assert row["sections"] == "0" and row["sections_total"] == 2
    assert row["ngram_recall"] == 1.0 and row["ler"] == 0.0 and row["gt_letters"] == len(sac.ar.arabic_letters(RECTO))
    assert row["ambiguous_images"] == 1 and row["overread_images"] == 0


def test_document_without_covered_images_has_no_recognition_fields():
    record = _record("doc", [("", RECTO)], 2)
    row, images = sac.score_covered_document(record, _coverage("doc", [("ambiguous", []), ("uncovered", [])]), {0: RECTO, 1: HEBREW})
    assert row["ngram_f1"] is None and row["ler"] is None and row["status"] is None
    assert row["ambiguous_images"] == 1 and row["ambiguous_arabic_letters"] == len(sac.ar.arabic_letters(RECTO))
    assert row["overread_hebrew_output"] == 1 and row["overread_ge40"] == 0
    assert images[1]["hebrew_output"] and not images[1]["ge40"]


def test_copies_are_left_out_unless_kept():
    record = _record("doc", [("", RECTO)], 2)
    pages = {0: RECTO, 1: RECTO}
    coverage = _coverage("doc", [("covered", [0]), ("covered", [0])], duplicate_of={1: 0})
    row, images = sac.score_covered_document(record, coverage, pages)
    assert row["duplicate_images"] == 1 and row["covered_images"] == 1 and row["ler"] == 0.0 and row["full_coverage"] is False
    assert images[1]["role"] == sac.DUPLICATE
    kept, _ = sac.score_covered_document(record, coverage, pages, keep_duplicates=True)
    assert kept["covered_images"] == 2 and kept["ler"] == pytest.approx(1.0) and kept["full_coverage"] is True
    assert kept["ngram_f1"] == sab.score_document(record, pages)["ngram_f1"]


def test_unanswered_covered_image_scores_like_the_scorer_and_unanswered_uncovered_is_not_counted():
    record = _record("doc", [("recto", RECTO), ("verso", VERSO)], 3)
    coverage = _coverage("doc", [("covered", [0]), ("covered", [1]), ("uncovered", [])])
    row, images = sac.score_covered_document(record, coverage, {0: RECTO})
    assert row["covered_answered"] == 1 and row["ngram_precision"] == 1.0 and row["ngram_recall"] < 0.6
    assert row["overread_images"] == 1 and row["overread_answered"] == 0 and not images[2]["answered"]


# --- per-image measures -----------------------------------------------------------------------------------------------

def test_image_measures():
    loop = sac.image_measures("الحمد لله رب العالمين " * 20)
    assert loop["loop"] and loop["ge40"] and not loop["hebrew_output"]
    hebrew = sac.image_measures(HEBREW)
    assert hebrew["hebrew_output"] and hebrew["arabic_letters"] == 0 and hebrew["hebrew_letters"] > 40
    short = sac.image_measures("قال")
    assert not short["ge40"] and not short["hebrew_output"] and not short["loop"]
    none = sac.image_measures(None)
    assert not none["answered"] and none["arabic_letters"] == 0


def test_side_script_and_roles():
    assert sac.side_script({"arabic_share": 0.2}) == "hebrew"
    assert sac.side_script({"arabic_share": 0.9}) == "arabic"
    assert sac.side_script({"arabic_share": None}) == "" and sac.side_script({}) == ""
    assert sac.image_role({"status": "blank_or_faint"}) == sac.OVERREAD
    assert sac.image_role({"status": "ambiguous"}) == sac.EXCLUDED
    assert sac.image_role({"status": "covered", "duplicate_of": 0}) == sac.DUPLICATE
    assert sac.image_role({"status": "covered", "duplicate_of": 0}, keep_duplicates=True) == sac.RECOGNITION


# --- whole model ------------------------------------------------------------------------------------------------------

def _benchmark() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[int, Dict[str, Any]]], Dict[str, Dict[int, str]]]:
    """Three synthetic documents with coverage and answers.

    :return: ``(benchmark, coverage, answers)``.
    :rtype: Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[int, Dict[str, Any]]], Dict[str, Dict[int, str]]]
    """
    bench = {"a": _record("a", [("", RECTO)], 1), "b": _record("b", [("recto", VERSO), ("verso", MARGIN)], 2),
             "c": _record("c", [("", UNEDITED)], 2)}
    coverage = {"a": _coverage("a", [("covered", [0])]), "b": _coverage("b", [("covered", [0]), ("covered", [1])]),
                "c": _coverage("c", [("covered", [0]), ("uncovered", [])], arabic_share=[1.0, 0.1])}
    answers = {"a": {0: RECTO[:70]}, "b": {0: VERSO, 1: MARGIN[:40]}, "c": {0: UNEDITED, 1: RECTO}}
    return bench, coverage, answers


def test_score_covered_model_matches_the_scorer_with_its_floor():
    bench, coverage, answers = _benchmark()
    docs, images = sac.score_covered_model(bench, coverage, answers, "m")
    scorer = {r["doc_id"]: r for r in sab.score_model(bench, answers, "m")}
    for row in docs[:2]:
        assert {k: row[k] for k in COMPARED} == {k: scorer[row["doc_id"]][k] for k in COMPARED}
    assert docs[2]["ngram_f1"] == 1.0 and scorer["c"]["ngram_f1"] < 1.0
    check = sac.equivalence_check(docs, list(scorer.values()))
    assert check == {"documents": 2, "identical": 2, "mismatches": []}
    summary = sac.summarise_covered(docs, images)
    assert summary["recognition_documents"] == 3 and summary["overread_answered"] == 1
    assert summary["overread_ge40_share"] == 1.0 and summary["overread_hebrew_side_images"] == 1
    assert summary["overread_ge40_share_hebrew_side"] == 1.0 and summary["overread_arabic_side_images"] == 0
    only, _ = sac.score_covered_model(bench, coverage, answers, "m", only=["a", "c"])
    assert [r["doc_id"] for r in only] == ["a", "c"]
    with pytest.raises(ValueError):
        sac.score_covered_model(bench, {"a": coverage["a"]}, answers, "m")


def test_equivalence_check_reads_scorer_csv_strings():
    bench, coverage, answers = _benchmark()
    docs, _ = sac.score_covered_model(bench, coverage, answers, "m")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=sab.FIELDS)
    writer.writeheader()
    writer.writerows(sab.score_model(bench, answers, "m"))
    buffer.seek(0)
    on_disk = list(csv.DictReader(buffer))
    assert sac.equivalence_check(docs, on_disk)["identical"] == 2
    on_disk[0]["ler"] = "0.9999"
    check = sac.equivalence_check(docs, on_disk)
    assert check["identical"] == 1 and check["mismatches"][0][:2] == ["a", "ler"]


# --- paired comparison ------------------------------------------------------------------------------------------------

def _rows(values: Dict[str, Optional[float]], metric: str) -> List[Dict[str, Any]]:
    """Document rows with one metric.

    :param values: ``{doc_id: value}``.
    :type values: Dict[str, Optional[float]]
    :param metric: Column name.
    :type metric: str
    :return: Rows.
    :rtype: List[Dict[str, Any]]
    """
    return [{"doc_id": d, metric: v} for d, v in values.items()]


def test_paired_comparison_counts_and_intervals():
    new = _rows({"a": 0.5, "b": 0.4, "c": 0.3, "d": None, "e": 0.2}, "ngram_f1")
    old = _rows({"a": 0.3, "b": 0.4, "c": 0.4, "d": 0.1}, "ngram_f1")
    c = sac.paired_comparison(new, old, "ngram_f1", higher_is_better=True, resamples=2000)
    assert c["documents"] == 3 and (c["wins"], c["losses"], c["ties"]) == (1, 1, 1)
    assert c["mean_delta"] == pytest.approx(round((0.2 + 0.0 - 0.1) / 3, 4))
    assert c["mean_ci"][0] <= c["mean_delta"] <= c["mean_ci"][1]
    ler = sac.paired_comparison(_rows({"a": 0.5, "b": 0.9}, "ler"), _rows({"a": 0.7, "b": 0.8}, "ler"), "ler", higher_is_better=False,
                                resamples=500)
    assert (ler["wins"], ler["losses"]) == (1, 1)
    assert sac.paired_comparison(new, _rows({"z": 0.1}, "ngram_f1"), "ngram_f1", True) == {"documents": 0}


def test_bootstrap_ci_is_deterministic_and_degenerate_for_constant_deltas():
    assert sac.bootstrap_ci([0.1] * 10, "mean", 1000) == pytest.approx((0.1, 0.1))
    values = [0.3, -0.1, 0.2, 0.0, 0.5, -0.2, 0.1]
    assert sac.bootstrap_ci(values, "median", 3000, seed=7) == sac.bootstrap_ci(values, "median", 3000, seed=7)
    low, high = sac.bootstrap_ci(values, "mean", 3000)
    assert low < sum(values) / len(values) < high


# --- v2 maps --------------------------------------------------------------------------------------------------------------

def _coverage_v2(doc_id: str, per_image: Sequence[Tuple[str, Sequence[int], str]], copy_of: Optional[Dict[int, int]] = None,
                 joined: Optional[Dict[int, List[int]]] = None, status: Optional[Dict[int, str]] = None) -> Dict[int, Dict[str, Any]]:
    """v2 coverage rows (role, script, joined_with).

    :param doc_id: Document id.
    :type doc_id: str
    :param per_image: ``(role, sections, script)`` per image.
    :type per_image: Sequence[Tuple[str, Sequence[int], str]]
    :param copy_of: ``{copy: original}``.
    :type copy_of: Optional[Dict[int, int]]
    :param joined: ``{image: [partner images]}``.
    :type joined: Optional[Dict[int, List[int]]]
    :param status: Group status of copies (default covered).
    :type status: Optional[Dict[int, str]]
    :return: ``{image_index: row}``.
    :rtype: Dict[int, Dict[str, Any]]
    """
    copy_of, joined, status = copy_of or {}, joined or {}, status or {}
    rows = {}
    for i, (role, sections, script) in enumerate(per_image):
        rows[i] = {"doc_id": doc_id, "image_index": i, "role": role, "status": status.get(i, "covered") if role == "copy_of" else role,
                   "sections": list(sections), "method": "manual", "copy_of": copy_of.get(i), "duplicate_of": copy_of.get(i),
                   "joined_with": joined.get(i, []), "script": script}
    return rows


def test_v2_roles_and_model_independent_script():
    rows = _coverage_v2("doc", [("covered", [0], "arabic"), ("copy_of", [0], "arabic"), ("blank_or_faint", [], "none"),
                                ("uncovered", [], "hebrew"), ("ambiguous", [], "unknown")], copy_of={1: 0})
    assert [sac.image_role(rows[i]) for i in range(5)] == [sac.RECOGNITION, sac.DUPLICATE, sac.OVERREAD, sac.OVERREAD, sac.EXCLUDED]
    assert sac.image_role(rows[1], keep_duplicates=True) == sac.RECOGNITION
    assert [sac.side_script(rows[i]) for i in range(5)] == ["arabic", "arabic", "none", "hebrew", ""]
    assert sac.side_script({"script": "mixed", "arabic_share": 1.0}) == "mixed"        # the v2 script wins over the reads


def test_v2_joined_section_is_scored_once_against_the_concatenation():
    first, second = RECTO[:60], RECTO[60:]
    record = _record("doc", [("", RECTO)], 3)
    coverage = _coverage_v2("doc", [("covered", [0], "arabic"), ("uncovered", [], "hebrew"), ("covered", [0], "arabic")],
                            joined={0: [2], 2: [0]})
    row, images = sac.score_covered_document(record, coverage, {0: first, 1: HEBREW, 2: second})
    assert row["joined_images"] == 2 and row["covered_images"] == 2 and row["sections"] == "0"
    assert row["ngram_f1"] == 1.0 and row["ler"] < 0.05              # the two halves joined in image order give the section back
    assert [im["joined_with"] for im in images] == ["2", "", "0"]
    assert (row["answer_order"], row["ler_order"]) == ("0;2", "0;2")
    swapped, _ = sac.score_covered_document(record, coverage, {0: second, 1: HEBREW, 2: first})
    assert swapped["ler"] == row["ler"] and swapped["ler_order"] == "2;0"     # which image starts the unit is not in the map
    assert swapped["answer_order"] == "0;2" and swapped["ngram_f1"] < 1.0     # the other fields keep the canonical order
    broken = _coverage_v2("doc", [("covered", [0], "arabic"), ("uncovered", [], "hebrew"), ("ambiguous", [], "arabic")],
                          joined={0: [2]})
    with pytest.raises(ValueError):
        sac.score_covered_document(record, broken, {0: first})


def test_v2_overread_split_by_script_and_blank():
    bench = {"a": _record("a", [("", RECTO)], 4)}
    coverage = {"a": _coverage_v2("a", [("covered", [0], "arabic"), ("uncovered", [], "hebrew"), ("blank_or_faint", [], "none"),
                                        ("uncovered", [], "mixed")])}
    answers = {"a": {0: RECTO, 1: UNEDITED, 2: UNEDITED + UNEDITED, 3: "قال"}}
    docs, images = sac.score_covered_model(bench, coverage, answers, "m")
    assert docs[0]["blank_images"] == 1 and docs[0]["overread_images"] == 3 and docs[0]["joined_images"] == 0
    summary = sac.summarise_covered(docs, images)
    assert (summary["overread_hebrew_side_images"], summary["overread_mixed_side_images"], summary["overread_blank_images"]) == (1, 1, 1)
    assert summary["overread_ge40_share_blank"] == 1.0 and summary["overread_ge40_share_mixed_side"] == 0.0
    assert summary["overread_arabic_side_images"] == 0 and summary["overread_arabic_letters_mean_blank"] == 2 * len(sac.ar.arabic_letters(UNEDITED))
    assert "blank images" in sac.table_b({"m": summary}).splitlines()[0]


@pytest.mark.parametrize("name, out", [("coverage_v1.jsonl", "scores_covered"), ("coverage_v2.jsonl", "scores_covered_v2"),
                                       ("other.jsonl", "scores_covered_other")])
def test_default_out_dir_never_reuses_an_older_maps_directory(name, out):
    assert sac.default_out_dir(Path(name)) == out


R1, R2 = RECTO[:55], RECTO[55:]
V1, V2 = VERSO[:50], VERSO[50:]


def _pieces() -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]], Dict[int, str]]:
    """A leaf photographed in two pieces: each side's unit is joined across the pieces, the second half first.

    :return: ``(record, v2 coverage, answers)``; images 0 / 1 = piece 1 recto / verso (second halves),
        2 / 3 = piece 2 recto / verso (first halves).
    :rtype: Tuple[Dict[str, Any], Dict[int, Dict[str, Any]], Dict[int, str]]
    """
    record = _record("doc", [("recto", R1), ("recto", R2), ("verso", V1), ("verso", V2)], 4)
    coverage = _coverage_v2("doc", [("covered", [0, 1], "arabic"), ("covered", [2, 3], "arabic"), ("covered", [0, 1], "arabic"),
                                    ("covered", [2, 3], "arabic")], joined={0: [2], 2: [0], 1: [3], 3: [1]})
    return record, coverage, {0: R2, 1: V2, 2: R1, 3: V1}


def test_joined_units_are_ordered_by_edition_and_not_interleaved():
    record, coverage, pages = _pieces()
    assert sac.joined_blocks(coverage, [0, 1, 2, 3]) == [[0, 2], [1, 3]]
    orders = sac.answer_orders(coverage, [0, 1, 2, 3])
    assert orders[0] == [0, 2, 1, 3] and len(orders) == 4 and [2, 0, 3, 1] in orders
    row, _ = sac.score_covered_document(record, coverage, pages)
    assert row["ler"] == 0.0 and row["ler_order"] == "2;0;3;1" and row["answer_order"] == "0;2;1;3"
    assert row["joined_images"] == 4 and row["full_coverage"] is True
    image_order = sab.score_document(record, pages)                  # the scorer's image order interleaves the units
    assert image_order["ler"] > 0.3
    assert sac.answer_orders(coverage, [0, 1, 2, 3], max_orders=3) == [[0, 2, 1, 3]]


def test_answer_orders_without_joined_units_is_image_order():
    coverage = _coverage_v2("doc", [("covered", [1], "arabic"), ("covered", [0], "arabic")])
    assert sac.answer_orders(coverage, [0, 1]) == [[0, 1]]           # unjoined documents keep the scorer's order
    assert sac.joined_blocks(coverage, [0, 1]) == [[0], [1]]


def test_equivalence_check_skips_documents_with_joined_units():
    record, coverage, pages = _pieces()
    row, _ = sac.score_covered_document(record, coverage, pages)
    scorer = sab.score_document(record, pages)
    assert sac.equivalence_check([row], [scorer]) == {"documents": 0, "identical": 0, "mismatches": []}


def test_overread_groups_split_uncovered_by_script_and_keep_blank_apart():
    bench = {"a": _record("a", [("", RECTO)], 6)}
    coverage = {"a": _coverage_v2("a", [("covered", [0], "arabic"), ("uncovered", [], "hebrew"), ("uncovered", [], "hebrew"),
                                        ("uncovered", [], "arabic"), ("blank_or_faint", [], "none"), ("uncovered", [], "unknown")])}
    answers = {"a": {0: RECTO, 1: HEBREW, 2: UNEDITED, 3: UNEDITED, 4: "قال"}}
    docs, images = sac.score_covered_model(bench, coverage, answers, "m")
    assert [sac.overread_group_of(im) for im in images[1:]] == ["hebrew", "hebrew", "arabic", "blank", "unknown"]
    s = sac.summarise_covered(docs, images)
    assert (s["overread_uncovered_images"], s["overread_uncovered_answered"]) == (4, 3)
    assert (s["overread_hebrew_answered"], s["overread_hebrew_ge40"], s["overread_hebrew_hebrew_output"]) == (2, 0.5, 0.5)
    hebrew_letters = sab.ar.script_share(HEBREW)[1]
    unedited = len(sac.ar.arabic_letters(UNEDITED))
    assert s["overread_hebrew_hebrew_letter_share"] == round(hebrew_letters / (hebrew_letters + unedited), 4)
    assert s["overread_hebrew_arabic_mean"] == round(unedited / 2, 1) and s["overread_arabic_arabic_median"] == unedited
    assert (s["overread_blank_answered"], s["overread_blank_arabic_mean"], s["overread_blank_ge40"]) == (1, 3, 0.0)
    assert (s["overread_unknown_images"], s["overread_unknown_answered"], s["overread_unknown_ge40"]) == (1, 0, None)
    assert s["overread_mixed_images"] == 0 and s["overread_mixed_arabic_mean"] is None
    for table in (sac.table_b_uncovered, sac.table_b_hebrew, sac.table_b_blank):
        text = table({"qwen3-vl-8b-heb-v25a-step900__trained": s})
        assert len(text.splitlines()) == 3 and "| v25a-step900 raw |" in text


def test_v1_rows_group_by_pooled_reads_and_have_no_blank():
    bench, coverage, answers = _benchmark()
    docs, images = sac.score_covered_model(bench, coverage, answers, "m")
    s = sac.summarise_covered(docs, images)
    assert s["overread_uncovered_answered"] == 1 and s["overread_hebrew_answered"] == 1 and s["overread_blank_images"] == 0
    assert [r["answer_order"] for r in docs] == ["0", "0;1", "0"] and all(r["answer_order"] == r["ler_order"] for r in docs)


@pytest.mark.parametrize("tag, name", [("qwen3-vl-8b-heb-v25a-step1200__trained__guarded", "v25a-step1200 guarded"),
                                       ("qwen3-vl-8b-heb-v24a-step2000__trained", "v24a-step2000 raw"),
                                       ("kraken_seg", "kraken_seg")])
def test_short_name(tag, name):
    assert sac.short_name(tag) == name


def test_recognition_changes_classifies_each_document():
    def row(doc_id, order, sections, f1=0.5, ler=0.5, ler_order=None):
        return {"doc_id": doc_id, "answer_order": order, "ler_order": ler_order or order, "sections": sections, "ngram_f1": f1,
                "ler": ler}
    old = [row("same", "0", "0"), row("enter", None, None, None, None), row("leave", "0", "0"), row("imgs", "0", "0"),
           row("secs", "0", "1"), row("order", "0;1", "0;1"), row("none", None, None, None, None)]
    new = [row("same", "0", "0", 0.6), row("enter", "1", "0"), row("leave", None, None, None, None), row("imgs", "0;2", "0"),
           row("secs", "0", "0;1"), row("order", "1;0", "0;1", ler_order="0;1"), row("none", None, None, None, None)]
    changes = sac.recognition_changes(old, new)
    assert [(c["doc_id"], c["change"]) for c in changes] == [("enter", "entered"), ("leave", "left"), ("imgs", "images"),
                                                             ("secs", "sections"), ("order", "order")]
    assert changes[2]["old_images"] == "0" and changes[2]["new_images"] == "0;2"


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write rows as JSON lines.

    :param path: Output file.
    :type path: Path
    :param rows: Rows.
    :type rows: Sequence[Dict[str, Any]]
    """
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def test_cli_defaults_to_the_v2_map_and_reports_the_diff(tmp_path):
    record, coverage, pages = _pieces()
    other = _record("other", [("", UNEDITED)], 2)
    _write_jsonl(tmp_path / "benchmark.jsonl", [record, other])
    v2 = [dict(r, kind="image") for r in coverage.values()]
    v2 += [dict(r, kind="image") for r in _coverage_v2("other", [("covered", [0], "arabic"), ("blank_or_faint", [], "none")]).values()]
    v1 = [dict(r, kind="image") for r in _coverage("doc", [("covered", [0, 1]), ("ambiguous", []), ("ambiguous", []), ("ambiguous", [])]).values()]
    v1 += [dict(r, kind="image") for r in _coverage("other", [("covered", [0]), ("uncovered", [])]).values()]
    _write_jsonl(tmp_path / "coverage_v2.jsonl", v2)
    _write_jsonl(tmp_path / "coverage_v1.jsonl", v1)
    (tmp_path / "outputs").mkdir()
    for tag in ("m-v1a-step1__trained", "m-v1a-step2__trained"):
        rows = [{"doc_id": "doc", "image_index": i, "text": t} for i, t in pages.items()]
        rows += [{"doc_id": "other", "image_index": 0, "text": UNEDITED}, {"doc_id": "other", "image_index": 1, "text": "قال"}]
        _write_jsonl(tmp_path / "outputs" / f"{tag}.jsonl", rows)
    report = tmp_path / "report.md"
    sac.main(["--benchmark", str(tmp_path), "--tags", "m-v1a-step2__trained", "m-v1a-step1__trained", "--sets", "all",
              "--compare", "m-v1a-step2__trained:m-v1a-step1__trained", "--report", str(report),
              "--diff-coverage", str(tmp_path / "coverage_v1.jsonl")])
    assert (tmp_path / "scores_covered_v2" / "m-v1a-step2__trained.csv").exists() and not (tmp_path / "scores_covered").exists()
    with open(tmp_path / "scores_covered_v2" / "m-v1a-step2__trained.csv", encoding="utf-8") as fh:
        doc = next(r for r in csv.DictReader(fh) if r["doc_id"] == "doc")
    assert (doc["ler"], doc["ler_order"], doc["joined_images"]) == ("0.0", "2;0;3;1", "4")
    text = report.read_text(encoding="utf-8")
    assert "(coverage_v2.jsonl)" in text and "Diff against `coverage_v1.jsonl`" in text
    assert "| doc | images | 0 (0;1) → 0;1;2;3 (0;1;2;3) |" in text
    sac.main(["--benchmark", str(tmp_path), "--coverage", str(tmp_path / "coverage_v1.jsonl"), "--tags", "m-v1a-step1__trained",
              "--sets", "all"])
    assert (tmp_path / "scores_covered" / "m-v1a-step1__trained.csv").exists()
