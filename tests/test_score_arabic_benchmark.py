# File name: test_score_arabic_benchmark.py
# Date: 10/2/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Tests for the Arabic-script benchmark scorer."""
import json

import pytest

from src.datasets.evaluations import arabic_script as ar
from src.datasets.evaluations.helper_eval_scripts import score_arabic_benchmark as sab

RECTO = ["بسم الله الرحمن الرحيم", "هذا ما استاجر افرايم بن عالي الاسرائيلي من ابي الفتوح"]
VERSO = ["الى الشيخ ابو سعد اطال الله بقاه وادام عزه"]
OTHER = ["وصل كتاب مولاي الشيخ الجليل وفهمت ما ذكره من امر البضاعه المحموله"]


def _record(doc_id, sections, images=2):
    lines = [l for _, ls in sections for l in ls]
    return {"id": doc_id, "images": [{"file": f"{doc_id}__{i}.jpg"} for i in range(images)],
            "gt": {"sections": [{"side": s, "lines": ls} for s, ls in sections], "text": "\n".join(lines)}}


TWO_SIDED = _record("doc_a", [("recto", RECTO), ("verso", VERSO)])
ONE_SIDED = _record("doc_b", [("", OTHER)], images=1)


def test_reference_orders_allows_the_reverse_for_two_sides_only():
    recto, verso = ar.arabic_letters("\n".join(RECTO)), ar.arabic_letters("\n".join(VERSO))
    assert sab.reference_orders(TWO_SIDED) == [recto + verso, verso + recto]
    assert sab.reference_orders(ONE_SIDED) == [ar.arabic_letters(OTHER[0])]


def test_a_perfect_answer_in_either_side_order():
    exact = sab.score_document(TWO_SIDED, {0: "\n".join(RECTO), 1: "\n".join(VERSO)}, ONE_SIDED)
    assert exact["status"] == "read" and exact["ngram_f1"] == 1.0 and exact["ler"] == 0.0 and exact["length_ratio"] == 1.0
    assert exact["images_expected"] == 2 and exact["images_answered"] == 2 and exact["floor_f1"] < 0.15
    flipped = sab.score_document(TWO_SIDED, {0: "\n".join(VERSO), 1: "\n".join(RECTO)})
    assert flipped["ler"] == 0.0 and flipped["ngram_f1"] > 0.9       # the images came verso first


def test_one_side_read_keeps_precision_and_loses_recall():
    row = sab.score_document(TWO_SIDED, {0: "\n".join(RECTO)})
    assert row["images_answered"] == 1 and row["ngram_precision"] == 1.0 and 0.5 < row["ngram_recall"] < 0.8
    assert 0.2 < row["ler"] < 0.5


def test_wrong_script_empty_and_hallucinated_answers():
    hebrew = sab.score_document(TWO_SIDED, {0: "בסם אללה אלרחמן אלרחים הדא מא אסתאגר אפרים בן עאלי"})
    assert hebrew["status"] == "wrong_script" and hebrew["ngram_f1"] == 0.0 and hebrew["arabic_share"] == 0.0 and hebrew["ler"] == 1.0
    empty = sab.score_document(TWO_SIDED, {0: "...", 1: ""})
    assert empty["status"] == "empty" and empty["answer_letters"] == 0
    unrelated = OTHER[0] + " ثم ان التاجر المذكور سافر الى الاسكندريه في المركب وباع جميع ما معه من القماش والحرير والفلفل ورجع سالما"
    fluent_but_wrong = sab.score_document(TWO_SIDED, {0: unrelated})
    assert fluent_but_wrong["status"] == "read" and fluent_but_wrong["ngram_f1"] < 0.15 and fluent_but_wrong["ler"] > 0.8
    assert fluent_but_wrong["length_ratio"] > 1.2
    assert sab.score_document(TWO_SIDED, {0: OTHER[0] * 3})["status"] == "loop"                    # the same wrong text, repeated


@pytest.mark.parametrize("arabic, hebrew, gt_letters, status", [
    (0, 0, 500, "empty"), (10, 5, 500, "empty"),
    (0, 400, 500, "wrong_script"), (60, 400, 500, "wrong_script"),      # the Arabic page came back in Hebrew letters
    (480, 900, 500, "read"),                                            # Arabic side read in Arabic, Hebrew side read in Hebrew
    (480, 0, 500, "read"), (30, 0, 500, "read"), (200, 150, 500, "read"),
])
def test_answer_status(arabic, hebrew, gt_letters, status):
    assert sab.answer_status(arabic, hebrew, gt_letters) == status


def test_a_collapsed_image_marks_the_document_as_a_loop():
    assert sab.answer_status(480, 0, 500, worst_repeat_share=0.8) == "loop"
    assert sab.answer_status(0, 900, 500, worst_repeat_share=0.8) == "loop"            # collapse is reported before the script
    assert sab.answer_status(5, 0, 500, worst_repeat_share=0.8) == "empty"
    row = sab.score_document(TWO_SIDED, {0: "\n".join(RECTO), 1: (VERSO[0] + "\n") * 40})
    assert row["status"] == "loop" and row["ngram_recall"] == 1.0 and row["ngram_precision"] < 0.2
    assert sab.summarise([dict(row, model="m")])["loop"] == 1


def test_images_match_edition_only_when_the_record_can_tell():
    assert sab.images_match_edition(TWO_SIDED) and sab.images_match_edition(ONE_SIDED)
    assert sab.images_match_edition(_record("d", [("verso", VERSO)], images=1))
    assert sab.images_match_edition(_record("d", [("recto", RECTO), ("recto", OTHER), ("verso", VERSO)], images=2))
    assert not sab.images_match_edition(_record("d", [("recto", RECTO), ("verso", VERSO)], images=1))     # one side is not in the images
    assert not sab.images_match_edition(_record("d", [("", OTHER), ("verso", VERSO)], images=1))
    assert not sab.images_match_edition(_record("d", [("recto", RECTO)], images=2))                       # what is on the other image?
    assert not sab.images_match_edition(_record("d", [("recto", RECTO), ("verso", VERSO)], images=3))


def test_score_model_summary_and_pairing():
    benchmark = {"doc_a": TWO_SIDED, "doc_b": ONE_SIDED}
    good = {"doc_a": {0: "\n".join(RECTO), 1: "\n".join(VERSO)}, "doc_b": {0: OTHER[0]}, "not_in_benchmark": {0: "x"}}
    partial = {"doc_a": {0: "בסם אללה אלרחמן אלרחים הדא מא אסתאגר"}}
    rows = sab.score_model(benchmark, good, "good")
    assert [r["doc_id"] for r in rows] == ["doc_a", "doc_b"] and all(r["model"] == "good" for r in rows)
    summary = sab.summarise(rows)
    assert summary["documents"] == 2 and summary["ngram_f1_median"] == 1.0 and summary["wrong_script"] == 0 and summary["f1_at_least_0.5"] == 2
    assert summary["matched_documents"] == 2 and summary["matched_f1_median"] == 1.0
    unmatched = sab.summarise(sab.score_model({"d": _record("d", [("recto", RECTO)], images=2)}, {"d": {0: RECTO[0]}}, "m"))
    assert unmatched["matched_documents"] == 0 and unmatched["matched_f1_median"] is None and unmatched["documents"] == 1
    weak = sab.summarise(sab.score_model(benchmark, partial, "weak"))
    assert weak == {**weak, "documents": 1, "wrong_script": 1, "incomplete": 1, "f1_below_0.1": 1}
    assert [r["doc_id"] for r in sab.score_model(benchmark, good, "good", only=["doc_a"])] == ["doc_a"]
    assert sab.summarise([]) == {"documents": 0}
    table = sab.markdown_table({"good": summary, "weak": weak})
    assert table.splitlines()[2].startswith("| good | 2 | 1.0 |") and table.splitlines()[3].startswith("| weak | 1 |")


def test_cli_scores_every_model_and_writes_files(tmp_path, capsys):
    (tmp_path / "outputs").mkdir()
    with open(tmp_path / "benchmark.jsonl", "w", encoding="utf-8") as fh:
        for record in (TWO_SIDED, ONE_SIDED):
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    rows = [{"doc_id": "doc_a", "image_index": 0, "text": "first attempt"}, {"doc_id": "doc_a", "image_index": 0, "text": "\n".join(RECTO)},
            {"doc_id": "doc_a", "image_index": 1, "text": "\n".join(VERSO)}, {"doc_id": "doc_b", "image_index": 0, "text": None, "error": "timeout"}]
    (tmp_path / "outputs" / "model_x.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    (tmp_path / "outputs" / "model_y.jsonl").write_text(json.dumps({"doc_id": "doc_b", "image_index": 0, "text": OTHER[0]}, ensure_ascii=False), encoding="utf-8")
    assert sab.load_outputs(tmp_path / "outputs" / "model_x.jsonl") == {"doc_a": {0: "\n".join(RECTO), 1: "\n".join(VERSO)}}   # last line wins, failures skipped
    sab.main(["--benchmark", str(tmp_path)])
    summary = json.loads((tmp_path / "scores" / "summary.json").read_text())
    assert summary["model_x"]["documents"] == 1 and summary["model_x"]["ngram_f1_median"] == 1.0 and summary["model_y"]["documents"] == 1
    assert (tmp_path / "scores" / "model_x.csv").read_text().splitlines()[0] == ",".join(sab.FIELDS)
    sab.main(["--benchmark", str(tmp_path), "--paired"])
    assert json.loads((tmp_path / "scores" / "summary.paired.json").read_text()) == {"model_x": {"documents": 0}, "model_y": {"documents": 0}}
    assert "paired on 0" in capsys.readouterr().out
    sab.main(["--benchmark", str(tmp_path), "--models", "model_x", "--paired", "--tag", "primary"])
    assert json.loads((tmp_path / "scores" / "summary.primary.json").read_text())["model_x"]["documents"] == 1
    assert (tmp_path / "scores" / "model_x.primary.csv").exists()
    (tmp_path / "outputs_other").mkdir()
    (tmp_path / "outputs_other" / "kraken.jsonl").write_text(json.dumps({"doc_id": "doc_b", "image_index": 0, "text": OTHER[0]}, ensure_ascii=False), encoding="utf-8")
    sab.main(["--benchmark", str(tmp_path), "--outputs", "outputs_other", "--tag", "other"])
    assert list(json.loads((tmp_path / "scores" / "summary.other.json").read_text())) == ["kraken"]
