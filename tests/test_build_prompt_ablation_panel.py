# File name: test_build_prompt_ablation_panel.py
# Date: 10/24/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests of the stratified image panel for the Arabic-script prompt ablation, on a synthetic coverage map."""
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

from src.datasets.evaluations.helper_eval_scripts import build_prompt_ablation_panel as bpp


def _image(doc_id: str, index: int, role: str, script: str = "arabic") -> Dict[str, Any]:
    """A coverage-map image row.

    :param doc_id: Document id.
    :type doc_id: str
    :param index: Image index.
    :type index: int
    :param role: Coverage role (``covered``, ``uncovered``, ``blank_or_faint``, ``copy_of``, ``ambiguous``).
    :type role: str
    :param script: Side script.
    :type script: str
    :return: Row.
    :rtype: Dict[str, Any]
    """
    return {"kind": "image", "doc_id": doc_id, "image_index": index, "image_file": f"{doc_id}__{index}.jpg", "role": role,
            "script": script}


# --- image classes ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("role, script, expected", [
    ("covered", "arabic", bpp.COVERED),
    ("covered", "hebrew", bpp.COVERED),
    ("blank_or_faint", "none", bpp.BLANK),
    ("uncovered", "hebrew", bpp.HEBREW_SIDE),
    ("uncovered", "arabic", bpp.ARABIC_UNEDITED),
    ("uncovered", "mixed", bpp.OTHER_UNCOVERED),
    ("uncovered", "unknown", bpp.OTHER_UNCOVERED),
    ("uncovered", None, bpp.OTHER_UNCOVERED),
    ("copy_of", "arabic", None),
    ("ambiguous", "arabic", None),
])
def test_image_class(role: str, script: Optional[str], expected: Optional[str]) -> None:
    assert bpp.image_class({"role": role, "script": script}) == expected


# --- quantile bins ----------------------------------------------------------------------------------------------------

def test_quantile_bins_sizes_and_order() -> None:
    items = [("e", 0.5), ("a", 0.1), ("d", 0.4), ("b", 0.2), ("c", 0.3), ("f", 0.6), ("g", 0.7)]
    bins = bpp.quantile_bins(items, 3)
    assert [len(b) for b in bins] == [3, 2, 2]                  # the remainder goes to the lowest bins
    assert bins == [["a", "b", "c"], ["d", "e"], ["f", "g"]]     # lowest score first, ordered within a bin
    ties = bpp.quantile_bins([("y", 0.5), ("x", 0.5)], 1)
    assert ties == [["x", "y"]]                                  # ties broken by id


def test_quantile_bins_never_makes_more_bins_than_items() -> None:
    assert bpp.quantile_bins([("a", 0.1), ("b", 0.2)], 5) == [["a"], ["b"]]
    assert bpp.quantile_bins([("a", 0.1)], 0) == [["a"]]
    assert bpp.quantile_bins([], 4) == [[]]


# --- covered documents ------------------------------------------------------------------------------------------------

def _bin_of(doc_id: str, bins: Sequence[Sequence[str]]) -> int:
    """Index of the bin holding a document.

    :param doc_id: Document id.
    :type doc_id: str
    :param bins: Bins of document ids.
    :type bins: Sequence[Sequence[str]]
    :return: Bin index.
    :rtype: int
    """
    return next(n for n, b in enumerate(bins) if doc_id in b)


def test_pick_covered_documents_round_robin_across_bins() -> None:
    candidates = {f"d{n}": [0] for n in range(8)}
    reference = {f"d{n}": n / 10 for n in range(8)}
    chosen = bpp.pick_covered_documents(candidates, reference, 4, random.Random(1), bins=4)
    bins = bpp.quantile_bins(list(reference.items()), 4)
    assert len(chosen) == 4
    assert [_bin_of(d, bins) for d in chosen] == [0, 1, 2, 3]   # one document from each bin, lowest bin first


def test_pick_covered_documents_unscored_form_their_own_bin() -> None:
    candidates = {"s1": [0], "s2": [0], "u1": [0], "u2": [0]}
    reference = {"s1": 0.1, "s2": 0.9}
    chosen = bpp.pick_covered_documents(candidates, reference, 3, random.Random(3), bins=2)
    assert chosen[:2] == ["s1", "s2"] and chosen[2] in ("u1", "u2")   # bins s1 / s2 / unscored, in that turn order


def test_pick_covered_documents_takes_whole_documents_and_never_overshoots() -> None:
    candidates = {"big": [0, 1, 2, 3, 4], "three": [0, 1, 2], "two": [0, 1], "one": [0]}
    reference = {"big": 0.9, "three": 0.5, "two": 0.3, "one": 0.1}
    for seed in range(20):
        chosen = bpp.pick_covered_documents(candidates, reference, 4, random.Random(seed), bins=2)
        images = sum(len(candidates[d]) for d in chosen)
        assert images <= 4 and "big" not in chosen
        assert len(set(chosen)) == len(chosen)
    assert bpp.pick_covered_documents(candidates, reference, 0, random.Random(0)) == []
    assert bpp.pick_covered_documents({"big": [0, 1, 2]}, {}, 2, random.Random(0)) == []     # nothing fits


def test_pick_covered_documents_is_deterministic_for_a_seed() -> None:
    candidates = {f"d{n}": list(range(1 + n % 3)) for n in range(20)}
    reference = {f"d{n}": (n * 7 % 11) / 11 for n in range(15)}
    first = bpp.pick_covered_documents(candidates, reference, 12, random.Random(42))
    again = bpp.pick_covered_documents(candidates, reference, 12, random.Random(42))
    assert first == again and sum(len(candidates[d]) for d in first) == 12


# --- other classes ----------------------------------------------------------------------------------------------------

def test_pick_images_prefers_documents_and_respects_the_quota() -> None:
    rows = [{"doc_id": d, "image_index": i} for d in ("a", "b", "c", "d") for i in range(2)]
    picked = bpp.pick_images(rows, 3, ["c"], random.Random(0))
    assert len(picked) == 3
    assert {r["doc_id"] for r in picked[:2]} == {"c"}             # both images of the preferred document first
    assert picked[2]["doc_id"] != "c"
    assert len(bpp.pick_images(rows, 50, [], random.Random(0))) == len(rows)
    assert bpp.pick_images(rows, 0, ["a"], random.Random(0)) == []


# --- whole panel ------------------------------------------------------------------------------------------------------

def _synthetic() -> tuple:
    """A 30-document benchmark and its coverage rows.

    Documents ``doc00``..``doc29``; every document has one or two covered images, plus (by position) a Hebrew side,
    a blank image, an unedited Arabic side, a mixed side, a copy and an ambiguous image.

    :return: ``(benchmark, image_rows, reference)``.
    :rtype: tuple
    """
    benchmark: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    reference: Dict[str, float] = {}
    for n in range(30):
        doc = f"doc{n:02d}"
        benchmark[doc] = {"id": doc}
        idx = 0
        for _ in range(1 + n % 2):
            rows.append(_image(doc, idx, "covered"))
            idx += 1
        for role, script in (("uncovered", "hebrew"), ("blank_or_faint", "none"), ("uncovered", "arabic"),
                             ("uncovered", "mixed" if n % 2 else "unknown"), ("copy_of", "arabic"), ("ambiguous", "arabic")):
            rows.append(_image(doc, idx, role, script))
            idx += 1
        if n % 5:
            reference[doc] = round(n / 30, 3)
    rows.append(_image("not_in_benchmark", 0, "uncovered", "hebrew"))
    rows.append(_image("doc00", 99, "covered"))                    # a second covered image, listed out of order
    return benchmark, rows, reference


QUOTAS = {bpp.COVERED: 6, bpp.HEBREW_SIDE: 5, bpp.BLANK: 4, bpp.ARABIC_UNEDITED: 3, bpp.OTHER_UNCOVERED: 2}


def test_build_panel_end_to_end() -> None:
    benchmark, rows, reference = _synthetic()
    panel = bpp.build_panel(benchmark, rows, reference, QUOTAS, first=10, seed=7, bins=2)
    counts = Counter(r["image_class"] for r in panel)
    assert {c: counts[c] for c in bpp.CLASSES} == QUOTAS
    assert all(set(r) == set(bpp.PANEL_FIELDS) for r in panel)

    order = list(benchmark)
    covered = [r for r in panel if r["image_class"] == bpp.COVERED]
    covered_docs = {r["doc_id"] for r in covered}
    assert all(order.index(d) < 10 for d in covered_docs)           # covered documents only from the first N
    for doc in covered_docs:                                         # every covered image of a chosen document
        expected = {r["image_index"] for r in rows if r["doc_id"] == doc and r["role"] == "covered"}
        assert {r["image_index"] for r in covered if r["doc_id"] == doc} == expected
    assert all(r["reference_f1"] == reference.get(r["doc_id"]) for r in covered)
    assert all(r["reference_f1"] is None for r in panel if r["image_class"] != bpp.COVERED)

    for cls in bpp.CLASSES[1:]:                                      # the quota fits in the preferred documents
        assert all(r["benchmark_position"] < 10 for r in panel if r["image_class"] == cls)
    assert "not_in_benchmark" not in {r["doc_id"] for r in panel}
    excluded = {(r["doc_id"], r["image_index"]) for r in rows if r["role"] in ("copy_of", "ambiguous")}
    assert not excluded & {(r["doc_id"], r["image_index"]) for r in panel}   # no copies, no ambiguous images

    keys = [(bpp.CLASSES.index(r["image_class"]), r["benchmark_position"], r["image_index"]) for r in panel]
    assert keys == sorted(keys)                                       # by class, then benchmark position, then image
    assert all(r["benchmark_position"] == order.index(r["doc_id"]) for r in panel)
    assert panel == bpp.build_panel(benchmark, rows, reference, QUOTAS, first=10, seed=7, bins=2)


def test_build_panel_falls_back_to_other_documents_when_the_preferred_run_short() -> None:
    benchmark, rows, reference = _synthetic()
    panel = bpp.build_panel(benchmark, rows, reference, dict(QUOTAS, hebrew_side=8), first=3, seed=1)
    hebrew = [r for r in panel if r["image_class"] == bpp.HEBREW_SIDE]
    assert len(hebrew) == 8
    assert sum(r["benchmark_position"] < 3 for r in hebrew) == 3       # all three preferred Hebrew sides are in
    covered_docs = {r["doc_id"] for r in panel if r["image_class"] == bpp.COVERED}
    assert covered_docs and all(list(benchmark).index(d) < 3 for d in covered_docs)


# --- files ------------------------------------------------------------------------------------------------------------

def test_write_and_load_panel_round_trip(tmp_path: Path) -> None:
    benchmark, rows, reference = _synthetic()
    panel = bpp.build_panel(benchmark, rows, reference, QUOTAS, first=10, seed=7, bins=2)
    panel_path, docs_path = bpp.write_panel(panel, tmp_path / "panels" / "ablation_v1")
    assert panel_path == tmp_path / "panels" / "ablation_v1.jsonl" and docs_path == tmp_path / "panels" / "ablation_v1.docs.csv"
    assert bpp.load_panel(panel_path) == panel
    with open(docs_path, encoding="utf-8") as fh:
        docs = [r["doc_id"] for r in csv.DictReader(fh)]
    assert docs == sorted({r["doc_id"] for r in panel if r["image_class"] == bpp.COVERED})


def test_load_image_rows_and_reference_f1(tmp_path: Path) -> None:
    coverage = tmp_path / "coverage_v2.jsonl"
    coverage.write_text("\n".join(json.dumps(r) for r in [{"kind": "document", "doc_id": "a"}, _image("a", 0, "covered"),
                                                          _image("a", 1, "uncovered", "hebrew")]) + "\n\n", encoding="utf-8")
    assert [r["image_index"] for r in bpp.load_image_rows(coverage)] == [0, 1]
    scores = tmp_path / "scores.csv"
    scores.write_text("doc_id,ngram_f1\na,0.25\nb,\nc,None\n", encoding="utf-8")
    assert bpp.load_reference_f1(scores) == {"a": 0.25}
    assert bpp.load_reference_f1(None) == {}


def test_parse_quotas() -> None:
    quotas = bpp.parse_quotas(["covered=10", "blank_faint=0"])
    assert quotas == dict(bpp.DEFAULT_QUOTAS, covered=10, blank_faint=0)
    assert bpp.parse_quotas([]) == bpp.DEFAULT_QUOTAS and bpp.parse_quotas([]) is not bpp.DEFAULT_QUOTAS
    with pytest.raises(ValueError):
        bpp.parse_quotas(["hebrew=3"])


def test_summary() -> None:
    panel = [{"doc_id": "a", "image_class": bpp.COVERED}, {"doc_id": "a", "image_class": bpp.COVERED},
             {"doc_id": "b", "image_class": bpp.COVERED}, {"doc_id": "a", "image_class": bpp.HEBREW_SIDE},
             {"doc_id": "c", "image_class": bpp.BLANK}]
    assert bpp.summary(panel) == {"images": 5, "covered_documents": 2,
                                  "by_class": {bpp.COVERED: 3, bpp.HEBREW_SIDE: 1, bpp.BLANK: 1, bpp.ARABIC_UNEDITED: 0,
                                               bpp.OTHER_UNCOVERED: 0}}
    assert bpp.summary([]) == {"images": 0, "covered_documents": 0, "by_class": {c: 0 for c in bpp.CLASSES}}
