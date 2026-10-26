# File name: test_score_prompt_ablation.py
# Date: 10/24/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests of the Arabic-script prompt-ablation scorer, on a synthetic benchmark, coverage map, panel and arms."""
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from src.datasets.evaluations.helper_eval_scripts import score_arabic_covered as sac
from src.datasets.evaluations.helper_eval_scripts import score_prompt_ablation as spa

RECTO = ("بسم الله الرحمن الرحيم هذا ما اشترى الشيخ ابو الفضل بن يوسف من القاضي ابي علي دارا في سوق الوراقين "
         "بمدينة مصر بثمن مبلغه عشرون دينارا")
VERSO = ("وصل كتابك ايها الاخ العزيز وفهمت ما ذكرته من امر القماش والحرير والفلفل الذي حملته الى الاسكندرية "
         "في المركب الكبير")
UNEDITED = ("قال الشاعر في وصف الليل الطويل والنجوم الزاهرة والقمر المنير فوق الجبال العالية والبحار الواسعة "
            "والسحاب الماطر")
HEBREW = "שלום עליכם אחי היקר קיבלתי את מכתבך ושמחתי מאד לשמוע ממך ומשלומך ומשלום כל בני ביתך"
MODEL = "qwen3-vl-8b-heb-v25a-step1200"


def _record(doc_id: str, sections: Sequence[Tuple[str, str]], images: int) -> Dict[str, Any]:
    """A benchmark record (as in ``test_score_arabic_covered``).

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


def _coverage(doc_id: str, per_image: Sequence[Tuple[str, Sequence[int], str]]) -> Dict[int, Dict[str, Any]]:
    """v2 coverage rows of a document (as in ``test_score_arabic_covered``).

    :param doc_id: Document id.
    :type doc_id: str
    :param per_image: ``(role, sections, script)`` per image.
    :type per_image: Sequence[Tuple[str, Sequence[int], str]]
    :return: ``{image_index: row}``.
    :rtype: Dict[int, Dict[str, Any]]
    """
    return {i: {"kind": "image", "doc_id": doc_id, "image_index": i, "role": role, "status": role, "sections": list(sections),
                "method": "manual", "copy_of": None, "duplicate_of": None, "joined_with": [], "script": script}
            for i, (role, sections, script) in enumerate(per_image)}


def _fixture() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[int, Dict[str, Any]]], List[Dict[str, Any]]]:
    """Three documents: ``a`` (one covered image), ``b`` (covered recto, Hebrew verso), ``c`` (blank, unedited Arabic).

    :return: ``(benchmark, coverage, panel)``.
    :rtype: Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[int, Dict[str, Any]]], List[Dict[str, Any]]]
    """
    benchmark = {"a": _record("a", [("", RECTO)], 1), "b": _record("b", [("recto", VERSO)], 2), "c": _record("c", [("", RECTO)], 2)}
    coverage = {"a": _coverage("a", [("covered", [0], "arabic")]),
                "b": _coverage("b", [("covered", [0], "arabic"), ("uncovered", [], "hebrew")]),
                "c": _coverage("c", [("blank_or_faint", [], "none"), ("uncovered", [], "arabic")])}
    classes = [("a", 0, "covered"), ("b", 0, "covered"), ("b", 1, "hebrew_side"), ("c", 0, "blank_faint"), ("c", 1, "arabic_unedited")]
    panel = [{"doc_id": d, "image_index": i, "image_file": f"{d}__{i}.jpg", "image_class": c, "script": None, "reference_f1": None,
              "benchmark_position": "abc".index(d)} for d, i, c in classes]
    return benchmark, coverage, panel


def _arm(tag: str, texts: Dict[Tuple[str, int], Optional[str]], finish: Optional[str] = "stop") -> spa.Arm:
    """An arm whose rows carry the given texts.

    :param tag: Arm tag (``<model>__<prompt>__mt<n>``).
    :type tag: str
    :param texts: Text per image.
    :type texts: Dict[Tuple[str, int], Optional[str]]
    :param finish: Finish reason of every row.
    :type finish: Optional[str]
    :return: Arm.
    :rtype: spa.Arm
    """
    model, prompt, budget = spa.parse_arm_tag(tag)
    rows = {k: {"doc_id": k[0], "image_index": k[1], "text": t, "finish_reason": finish, "prompt_tokens": 800,
                "completion_tokens": 10 * len(t or ""), "secs": 1.0} for k, t in texts.items()}
    return spa.Arm(tag, model, prompt, budget, rows)


TRAINED = {("a", 0): RECTO[:70], ("b", 0): VERSO, ("b", 1): UNEDITED, ("c", 0): "[...] [...]", ("c", 1): UNEDITED}
NOSCRIPT = {("a", 0): spa.ABSTAIN_MARKER, ("b", 0): VERSO[:50], ("b", 1): HEBREW, ("c", 0): "", ("c", 1): UNEDITED}


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write rows as JSON lines.

    :param path: Output file.
    :type path: Path
    :param rows: Rows.
    :type rows: Sequence[Dict[str, Any]]
    """
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


# --- names ------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("model, short", [(MODEL, "v2.5a-1200"), ("qwen3-vl-8b-heb-v24a-step2000", "v2.4a-2000"),
                                          (MODEL + "-ctx16k", "v2.5a-1200-ctx16k"), ("kraken_seg", "kraken_seg"),
                                          ("qwen/qwen3-vl-8b", "qwen/qwen3-vl-8b")])
def test_short_model(model: str, short: str) -> None:
    assert spa.short_model(model) == short


def test_parse_arm_tag() -> None:
    assert spa.parse_arm_tag(f"{MODEL}__trained__mt2500") == (MODEL, "trained", 2500)
    assert spa.parse_arm_tag("a__b__noscript__mt1200") == ("a__b", "noscript", 1200)
    assert spa.parse_arm_tag("v25a-1200-cached") is None
    assert spa.parse_arm_tag(f"{MODEL}__trained") is None
    assert spa.parse_arm_tag(f"{MODEL}__trained__mt") is None


# --- per image --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, kind", [(None, None), ("", "empty"), ("  \n", "empty"), ("[NO_READABLE_TEXT]", "marker"),
                                        (" [NO_READABLE_TEXT]\n", "marker"), ("[...] [...]", "no_letters"), ("[?] 12", "no_letters"),
                                        (UNEDITED, None), (HEBREW, None), ("[NO_READABLE_TEXT] قال", None)])
def test_abstain_kind(text: Optional[str], kind: Optional[str]) -> None:
    assert spa.abstain_kind(text) == kind


def test_measure() -> None:
    text = "[...] قال الشيخ […] ابو"
    m = spa.measure({"text": text, "finish_reason": "length", "prompt_tokens": 5, "completion_tokens": 7, "secs": 1.5})
    assert set(m) == set(sac.image_measures(text)) | {"abstain", "chars", "gap_markers", "finish_reason", "prompt_tokens",
                                                      "completion_tokens", "secs"}
    assert m["answered"] and m["abstain"] is None and m["chars"] == len(text) and m["gap_markers"] == 2
    assert (m["finish_reason"], m["prompt_tokens"], m["completion_tokens"], m["secs"]) == ("length", 5, 7, 1.5)
    assert m["arabic_letters"] == sac.image_measures(text)["arabic_letters"]
    for missing in (None, {"text": None, "finish_reason": "length"}, {"text": 7}):
        n = spa.measure(missing)
        assert not n["answered"] and n["abstain"] is None and n["chars"] == 0 and n["gap_markers"] == 0
    assert spa.measure({"text": None, "finish_reason": "length"})["finish_reason"] == "length"
    assert spa.measure(None)["finish_reason"] is None and spa.measure({"text": ""})["abstain"] == "empty"


# --- files and arms ---------------------------------------------------------------------------------------------------

def test_load_rows(tmp_path: Path) -> None:
    path = tmp_path / "arm.jsonl"
    _write_jsonl(path, [{"doc_id": "a", "image_index": 0, "text": "old"}, {"doc_id": "b", "image_index": "1", "text": "b"},
                        {"doc_id": "a", "image_index": 0, "text": "new"}])
    assert spa.load_rows(path) == {("a", 0): {"doc_id": "a", "image_index": 0, "text": "new"},
                                   ("b", 1): {"doc_id": "b", "image_index": "1", "text": "b"}}
    assert list(spa.load_rows(path, keep=[("b", 1), ("z", 0)])) == [("b", 1)]
    assert spa.load_rows(path, keep=[]) == {}


def test_discover_arms_order_and_panel_restriction(tmp_path: Path) -> None:
    rows = [{"doc_id": "a", "image_index": 0, "text": "x"}, {"doc_id": "z", "image_index": 9, "text": "off panel"}]
    stems = ["m2__trained__mt2500", "m1__stopping__mt2500", "m1__trained__mt1200", "m1__noscript__mt2500", "m1__trained__mt2500",
             "m1__custom__mt100", "notes", f"{MODEL}__trained"]
    for stem in stems:
        _write_jsonl(tmp_path / f"{stem}.jsonl", rows)
    (tmp_path / "m1.loaded.json").write_text("{}", encoding="utf-8")
    arms = spa.discover_arms(tmp_path, [("a", 0)])
    assert [(a.model, a.prompt, a.max_tokens) for a in arms] == [("m1", "trained", 2500), ("m1", "trained", 1200),
                                                                 ("m1", "noscript", 2500), ("m1", "stopping", 2500),
                                                                 ("m1", "custom", 100), ("m2", "trained", 2500)]
    assert arms[0].tag == "m1__trained__mt2500" and all(list(a.rows) == [("a", 0)] for a in arms)


def test_reference_arm(tmp_path: Path) -> None:
    path = tmp_path / f"{MODEL}__trained.jsonl"
    _write_jsonl(path, [{"doc_id": "a", "image_index": 0, "text": RECTO}, {"doc_id": "a", "image_index": 1, "text": "x"}])
    arm = spa.reference_arm("v25a-cached", path, [("a", 0), ("b", 0)])
    assert (arm.tag, arm.model, arm.prompt, arm.max_tokens) == ("v25a-cached", MODEL, "trained(cached)", 2500)
    assert list(arm.rows) == [("a", 0)]


# --- summaries --------------------------------------------------------------------------------------------------------

def test_summarise_class_shares_over_answered_only() -> None:
    _, _, panel = _fixture()
    texts = {("b", 1): spa.ABSTAIN_MARKER, ("c", 0): UNEDITED, ("c", 1): ""}
    arm = _arm(f"{MODEL}__stopping__mt2500", texts)
    arm.rows[("c", 0)]["finish_reason"] = "length"
    rows = [r for r in spa.image_table(arm, panel) if r["image_class"] != "covered"]
    s = spa.summarise_class(rows + [dict(rows[0], answered=False, abstain=None, finish_reason=None)])
    assert (s["images"], s["answered"]) == (4, 3)
    assert s["abstain"] == round(2 / 3, 3) and s["abstain_kinds"] == {"marker": 1, "empty": 1}
    letters = len(sac.ar.arabic_letters(UNEDITED))
    assert s["arabic_mean"] == round(letters / 3, 1) and s["arabic_median"] == 0
    assert s["ge40"] == round(1 / 3, 3) and s["loop"] == 0.0 and s["hebrew_output"] == 0.0
    assert s["finish"] == {"stop": 2, "length": 1} and s["length"] == round(1 / 3, 3)
    assert s["chars_mean"] == round((len(spa.ABSTAIN_MARKER) + len(UNEDITED)) / 3, 1)
    assert s["tokens_mean"] == round(10 * (len(spa.ABSTAIN_MARKER) + len(UNEDITED)) / 3, 1)
    for r in rows:
        r["finish_reason"] = None
    plain = spa.summarise_class(rows)
    assert plain["length"] is None and plain["finish"] == {"None": 3}
    empty = spa.summarise_class([])
    assert empty["answered"] == 0 and empty["abstain"] is None and empty["arabic_mean"] is None and empty["length"] is None


def test_covered_documents_score_like_the_covered_scorer() -> None:
    benchmark, coverage, panel = _fixture()
    arm = _arm(f"{MODEL}__trained__mt2500", TRAINED)
    docs = spa.covered_documents(benchmark, coverage, arm, panel)
    expected, _ = sac.score_covered_model(benchmark, coverage, {"a": {0: TRAINED[("a", 0)]}, "b": {0: TRAINED[("b", 0)]}},
                                          arm.tag, only=["a", "b"])
    assert [r["doc_id"] for r in docs] == ["a", "b"]
    assert docs == [{"arm": arm.tag, "model": MODEL, "prompt": "trained", "max_tokens": 2500,
                     **{k: v for k, v in e.items() if k != "model"}} for e in expected]
    assert docs[1]["overread_answered"] == 0                     # the Hebrew side of b is not part of the covered score
    assert set(docs[0]) == set(spa.DOC_FIELDS)


def test_one_covered_image_and_an_abstention() -> None:
    benchmark, coverage, panel = _fixture()
    only_a = [p for p in panel if p["doc_id"] == "a"]
    read = spa.covered_documents(benchmark, coverage, _arm(f"{MODEL}__trained__mt2500", {("a", 0): RECTO[:70]}), only_a)
    direct, _ = sac.score_covered_model(benchmark, coverage, {"a": {0: RECTO[:70]}}, "m")
    assert len(read) == 1 and (read[0]["ngram_f1"], read[0]["ler"]) == (direct[0]["ngram_f1"], direct[0]["ler"])
    assert 0 < read[0]["ngram_f1"] < 1 and read[0]["ngram_precision"] == 1.0
    abstained = spa.covered_documents(benchmark, coverage, _arm(f"{MODEL}__stopping__mt2500", {("a", 0): spa.ABSTAIN_MARKER}), only_a)
    assert (abstained[0]["ngram_f1"], abstained[0]["ler"], abstained[0]["status"]) == (0.0, 1.0, "empty")
    unanswered = spa.covered_documents(benchmark, coverage, _arm(f"{MODEL}__stopping__mt2500", {("a", 0): None}), only_a)
    assert unanswered == []                                      # a document with no answered covered image is not scored


def test_summarise_covered() -> None:
    benchmark, coverage, panel = _fixture()
    arm = _arm(f"{MODEL}__noscript__mt2500", NOSCRIPT)
    docs = spa.covered_documents(benchmark, coverage, arm, panel)
    images = [r for r in spa.image_table(arm, panel) if r["image_class"] == "covered"]
    s = spa.summarise_covered(docs, images)
    f1 = sorted(r["ngram_f1"] for r in docs)
    assert s["documents"] == 2 and f1[0] == 0.0
    assert s["f1"] == round((f1[0] + f1[1]) / 2, 4) and s["f1_mean"] == round(sum(f1) / 2, 4)
    assert s["docs_f1_ge_0_2"] == sum(v >= 0.2 for v in f1) and s["status"]["empty"] == 1
    assert s["ler"] == sac._median([r["ler"] for r in docs])
    assert s["images"]["answered"] == 2 and s["images"]["abstain"] == 0.5 and s["images"]["abstain_kinds"] == {"marker": 1}
    none = spa.summarise_covered([], [])
    assert none["documents"] == 0 and none["f1"] is None and none["f1_mean"] is None and none["docs_f1_ge_0_2"] == 0


# --- report and files -------------------------------------------------------------------------------------------------

def test_build_report_smoke(tmp_path: Path) -> None:
    benchmark, coverage, panel = _fixture()
    arms = [_arm(f"{MODEL}__trained__mt2500", TRAINED), _arm(f"{MODEL}__noscript__mt2500", NOSCRIPT),
            _arm(f"{MODEL}__trained__mt1200", {k: v for k, v in TRAINED.items() if k != ("c", 1)}, finish="length")]
    scored = {a.tag: spa.score_arm(a, panel, benchmark, coverage) for a in arms}
    assert set(scored[arms[0].tag]["by_class"]) == set(spa.CLASSES)
    report = spa.build_report(arms, scored, panel, tmp_path / "panel_v1.jsonl")
    for header in ("## Table 1: prompt × model × image class", "## Table 2: how generations ended",
                   "## Table 3: paired comparisons on the covered documents", "## Table 4: uncovered classes in detail"):
        assert header in report
    assert "Panel `panel_v1.jsonl`: 5 images" in report and spa.panel_summary(panel) in report
    assert spa.panel_summary(panel).startswith("covered 2 images / 2 docs; hebrew_side 1 images / 1 docs")
    assert f"partial: {arms[2].tag} (4/5)" in report
    assert "| v2.5a-1200 / noscript / 2500 vs v2.5a-1200 / trained / 2500 | 2 |" in report
    assert "| v2.5a-1200 / trained / 1200 vs v2.5a-1200 / trained / 2500 | 2 |" in report


def test_write_csv(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    spa.write_csv(path, [{"a": 1, "b": None, "extra": "dropped"}, {"a": "ب"}], ["a", "b"])
    with open(path, encoding="utf-8") as fh:
        assert fh.readline().strip() == "a,b"
        fh.seek(0)
        assert list(csv.DictReader(fh)) == [{"a": "1", "b": ""}, {"a": "ب", "b": ""}]


def test_main_end_to_end(tmp_path: Path) -> None:
    benchmark, coverage, panel = _fixture()
    _write_jsonl(tmp_path / "benchmark.jsonl", list(benchmark.values()))
    _write_jsonl(tmp_path / "coverage_v2.jsonl", [r for doc in coverage.values() for r in doc.values()])
    panel_path = tmp_path / "panels" / "panel_v1.jsonl"
    panel_path.parent.mkdir()
    _write_jsonl(panel_path, panel)
    arms_dir = tmp_path / "arms"
    arms_dir.mkdir()
    tag = f"{MODEL}__stopping__mt2500"
    _write_jsonl(arms_dir / f"{tag}.jsonl", [{"doc_id": d, "image_index": i, "text": t, "finish_reason": "stop", "completion_tokens": 3}
                                             for (d, i), t in TRAINED.items()])
    reference = tmp_path / f"{MODEL}__trained.jsonl"
    _write_jsonl(reference, [{"doc_id": d, "image_index": i, "text": t} for (d, i), t in NOSCRIPT.items()])
    out_dir = tmp_path / "scores"
    spa.main(["--panel", str(panel_path), "--benchmark", str(tmp_path), "--arms-dir", str(arms_dir), "--out-dir", str(out_dir),
              "--reference", f"cached={reference}"])
    with open(out_dir / "images.csv", encoding="utf-8") as fh:
        images = list(csv.DictReader(fh))
    assert list(images[0]) == spa.IMAGE_FIELDS and len(images) == 2 * len(panel)
    assert [r["arm"] for r in images] == [tag] * 5 + ["cached"] * 5
    with open(out_dir / "documents.csv", encoding="utf-8") as fh:
        documents = list(csv.DictReader(fh))
    assert list(documents[0]) == spa.DOC_FIELDS and [(r["arm"], r["doc_id"]) for r in documents] == [
        (tag, "a"), (tag, "b"), ("cached", "a"), ("cached", "b")]
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert list(summary) == [tag, "cached"] and summary["cached"]["prompt"] == "trained(cached)"
    assert summary[tag]["covered"]["documents"] == 2 and summary[tag]["hebrew_side"]["ge40"] == 1.0
    assert summary["cached"]["hebrew_side"]["hebrew_output"] == 1.0 and summary["cached"]["blank_faint"]["abstain"] == 1.0
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert report.startswith("# Arabic-script prompt ablation on the fixed panel") and "Arms with every panel image answered: 2" in report
