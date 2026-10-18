# File name: score_arabic_benchmark.py
# Date: 10/2/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Score model outputs on the Arabic-script benchmark (offline, no model needed).

Inputs: the benchmark directory written by ``build_arabic_benchmark.py`` and, under
``<benchmark>/outputs/``, one ``<model>.jsonl`` per model with a line per image::

    {"doc_id": ..., "image_index": 0, "text": "..."}

(the layout ``run_arabic_benchmark.py`` writes; any other reader's output can be converted to it).

A document's answer is the text of its images in image order. Everything is measured on Arabic
letters after the folding in ``arabic_script.arabic_letters``:

* ``ngram_precision / recall / f1``: clipped 5-gram overlap with the edition. Order-robust, so a
  verso read before the recto is not punished; precision is the cleanest reading-accuracy signal
  because an edition covers the whole document while the images may not.
* ``ler``: letter error rate (edit distance over the edition length), taking the better of the
  edition's side order and its reverse for two-sided documents. Only comparable across models
  on the same documents; above 1.0 means the answer is longer than the text and mostly wrong.
* ``arabic_share`` and ``status``: how much of the answer is in Arabic letters. ``wrong_script``
  means the Arabic text was answered in Hebrew letters, ``loop`` that the answer to at least one
  image collapsed into repeating one phrase (see :func:`answer_status`).
* ``floor_f1``: the same answer scored against another document's edition. Formulaic openings
  give unrelated documents a small overlap; a model has to clear this floor to have read anything.

An edition and the images of a fragment do not always cover the same sides: an edition of both
sides with one image caps recall, a second image with unedited Arabic text lowers precision. Model
comparisons are unaffected (every model sees the same images), but absolute values are only
meaningful on the documents where the two demonstrably match (``sides_match``, see
:func:`images_match_edition`); the summary reports that subset separately.

Usage::

    .venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.score_arabic_benchmark \\
        [--benchmark DIR] [--outputs SUBDIR] [--models a b ...] [--paired] [--tag NAME]
"""
import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.datasets.evaluations import arabic_script as ar

DEFAULT_BENCHMARK = Path("/Volumes/home/studio_offload/datasets/arabic_script_benchmark_v0")
MIN_ANSWER_LETTERS = 20
WRONG_SCRIPT_MAX_ARABIC = 0.25          # of the edition's letters
LOOP_REPEAT_SHARE = 0.5                 # more than half of an answer's 12-letter runs repeat an earlier one
FIELDS = ["doc_id", "model", "images_expected", "images_answered", "sides_match", "gt_letters", "answer_letters", "arabic_share", "status",
          "ngram_precision", "ngram_recall", "ngram_f1", "ler", "length_ratio", "floor_f1"]


def load_benchmark(benchmark: Path) -> Dict[str, Dict[str, Any]]:
    """Benchmark records by document id.

    :param benchmark: Benchmark directory.
    :type benchmark: Path
    :return: ``{doc_id: record}`` in file order.
    :rtype: Dict[str, Dict[str, Any]]
    """
    with open(benchmark / "benchmark.jsonl", encoding="utf-8") as fh:
        return {r["id"]: r for r in (json.loads(line) for line in fh if line.strip())}


def load_outputs(path: Path) -> Dict[str, Dict[int, str]]:
    """A model's answers: text per document and image (the last line for an image wins).

    :param path: ``outputs/<model>.jsonl``.
    :type path: Path
    :return: ``{doc_id: {image_index: text}}``; lines without text (failed requests) are skipped.
    :rtype: Dict[str, Dict[int, str]]
    """
    answers: Dict[str, Dict[int, str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row.get("text"), str):
                answers.setdefault(row["doc_id"], {})[int(row["image_index"])] = row["text"]
    return answers


def reference_orders(record: Dict[str, Any]) -> List[str]:
    """Letters-only ground truth in the side orders an answer may legitimately follow.

    :param record: Benchmark record.
    :type record: Dict[str, Any]
    :return: The edition order, plus the reverse order when the document has exactly two sides.
    :rtype: List[str]
    """
    sides = [ar.arabic_letters("\n".join(section["lines"])) for section in record["gt"]["sections"]]
    orders = ["".join(sides)]
    if len(sides) == 2:
        orders.append("".join(reversed(sides)))
    return orders


def answer_status(arabic: int, hebrew: int, gt_letters: int, worst_repeat_share: float = 0.0) -> str:
    """Classify an answer by how it failed, if it did.

    :param arabic: Arabic letters in the answer (all images of the document).
    :type arabic: int
    :param hebrew: Hebrew letters in the answer.
    :type hebrew: int
    :param gt_letters: Arabic letters in the edition.
    :type gt_letters: int
    :param worst_repeat_share: Largest ``arabic_script.repeat_share`` among the document's images.
    :type worst_repeat_share: float
    :return: ``"empty"`` (hardly any letters of either script), ``"loop"`` (the answer to an image
        collapsed into repeating one phrase), ``"wrong_script"`` (more Hebrew than Arabic letters
        and fewer Arabic letters than a quarter of the edition: the Arabic text was answered in
        Hebrew letters) or ``"read"``. A Hebrew-script side read correctly next to an Arabic side
        read in Arabic is ``"read"``: such fragments are common, Arabic documents were reused for
        Hebrew texts.
    :rtype: str
    """
    if arabic + hebrew < MIN_ANSWER_LETTERS:
        return "empty"
    if worst_repeat_share > LOOP_REPEAT_SHARE:
        return "loop"
    if hebrew > arabic and arabic < WRONG_SCRIPT_MAX_ARABIC * gt_letters:
        return "wrong_script"
    return "read"


def images_match_edition(record: Dict[str, Any]) -> bool:
    """Whether the images and the edition demonstrably cover the same sides.

    :param record: Benchmark record.
    :type record: Dict[str, Any]
    :return: True for one image with an edition of a single side, and for two images with an
        edition that has both a recto and a verso. False otherwise (an edition of one side with
        two images, three or more images): the other image may be blank, hold a Hebrew text, or
        hold Arabic text nobody edited, and the record cannot tell which.
    :rtype: bool
    """
    sides = {section["side"] for section in record["gt"]["sections"]}
    if len(record["images"]) == 1:
        return len(sides) == 1
    return len(record["images"]) == 2 and {"recto", "verso"} <= sides


def score_document(record: Dict[str, Any], pages: Dict[int, str], floor_record: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Score one document's answer.

    :param record: Benchmark record of the document.
    :type record: Dict[str, Any]
    :param pages: The model's text per image index (missing images count as unanswered).
    :type pages: Dict[int, str]
    :param floor_record: Another document, for the wrong-document floor.
    :type floor_record: Optional[Dict[str, Any]]
    :return: Row with the fields of :data:`FIELDS` except ``model``.
    :rtype: Dict[str, Any]
    """
    answer = "\n".join(pages[i] for i in sorted(pages))
    letters = ar.arabic_letters(answer)
    arabic, hebrew, share = ar.script_share(answer)
    orders = reference_orders(record)
    overlap = ar.ngram_overlap(letters, orders[0])
    return {"doc_id": record["id"], "images_expected": len(record["images"]), "images_answered": len(pages),
            "sides_match": images_match_edition(record), "gt_letters": len(orders[0]), "answer_letters": len(letters),
            "arabic_share": round(share, 4),
            "status": answer_status(arabic, hebrew, len(orders[0]), max((ar.repeat_share(text) for text in pages.values()), default=0.0)),
            "ngram_precision": round(overlap["precision"], 4), "ngram_recall": round(overlap["recall"], 4),
            "ngram_f1": round(overlap["f1"], 4), "ler": round(min(ar.letter_error_rate(letters, o) for o in orders), 4),
            "length_ratio": round(len(letters) / max(1, len(orders[0])), 3),
            "floor_f1": round(ar.ngram_overlap(letters, reference_orders(floor_record)[0])["f1"], 4) if floor_record else 0.0}


def score_model(benchmark: Dict[str, Dict[str, Any]], answers: Dict[str, Dict[int, str]], model: str,
                only: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Per-document rows for one model.

    :param benchmark: :func:`load_benchmark` result.
    :type benchmark: Dict[str, Dict[str, Any]]
    :param answers: :func:`load_outputs` result.
    :type answers: Dict[str, Dict[int, str]]
    :param model: Model name for the rows.
    :type model: str
    :param only: Restrict to these document ids (paired comparison).
    :type only: Optional[Sequence[str]]
    :return: One row per benchmark document the model answered at least one image of.
    :rtype: List[Dict[str, Any]]
    """
    ids = [d for d in benchmark if d in answers and (only is None or d in set(only))]
    rows = []
    for position, doc_id in enumerate(ids):
        other = benchmark[ids[(position + 1) % len(ids)]] if len(ids) > 1 else None
        rows.append(dict(score_document(benchmark[doc_id], answers[doc_id], other), model=model))
    return rows


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate one model's rows.

    :param rows: Output of :func:`score_model`.
    :type rows: List[Dict[str, Any]]
    :return: Counts, medians and means of the per-document metrics; ``matched_*`` are the same
        medians on the documents whose images and edition cover the same sides.
    :rtype: Dict[str, Any]
    """
    if not rows:
        return {"documents": 0}
    column = lambda name: [row[name] for row in rows]
    matched = [row for row in rows if row["sides_match"]]
    matched_median = lambda name: round(statistics.median(row[name] for row in matched), 4) if matched else None
    return {"documents": len(rows),
            "ngram_f1_median": round(statistics.median(column("ngram_f1")), 4), "ngram_f1_mean": round(statistics.mean(column("ngram_f1")), 4),
            "ngram_precision_median": round(statistics.median(column("ngram_precision")), 4),
            "ngram_recall_median": round(statistics.median(column("ngram_recall")), 4),
            "ler_median": round(statistics.median(column("ler")), 4),
            "floor_f1_median": round(statistics.median(column("floor_f1")), 4),
            "f1_at_least_0.5": sum(v >= 0.5 for v in column("ngram_f1")), "f1_at_least_0.2": sum(v >= 0.2 for v in column("ngram_f1")),
            "f1_below_0.1": sum(v < 0.1 for v in column("ngram_f1")),
            "wrong_script": sum(s == "wrong_script" for s in column("status")), "empty": sum(s == "empty" for s in column("status")),
            "loop": sum(s == "loop" for s in column("status")),
            "incomplete": sum(row["images_answered"] < row["images_expected"] for row in rows),
            "length_ratio_median": round(statistics.median(column("length_ratio")), 3),
            "matched_documents": len(matched), "matched_f1_median": matched_median("ngram_f1"),
            "matched_precision_median": matched_median("ngram_precision"), "matched_recall_median": matched_median("ngram_recall")}


def markdown_table(summaries: Dict[str, Dict[str, Any]]) -> str:
    """Model comparison as a Markdown table.

    :param summaries: ``{model: summarise(rows)}``.
    :type summaries: Dict[str, Dict[str, Any]]
    :return: Table text.
    :rtype: str
    """
    head = ["model", "docs", "5-gram F1 (median)", "precision", "recall", "LER (median)", "floor F1", "F1>=0.5", "F1>=0.2",
            "loops", "wrong script", "empty", "sides-matched docs", "their F1", "their precision", "their recall"]
    keys = ["documents", "ngram_f1_median", "ngram_precision_median", "ngram_recall_median", "ler_median", "floor_f1_median",
            "f1_at_least_0.5", "f1_at_least_0.2", "loop", "wrong_script", "empty", "matched_documents", "matched_f1_median",
            "matched_precision_median", "matched_recall_median"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for model, summary in sorted(summaries.items(), key=lambda kv: -kv[1].get("ngram_f1_median", 0)):
        lines.append("| " + " | ".join([model] + [str(summary.get(k, "")) for k in keys]) + " |")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI.

    :param argv: Arguments.
    :type argv: Optional[List[str]]
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--models", nargs="*", default=None, help="model names (outputs/<name>.jsonl); default: all found")
    parser.add_argument("--paired", action="store_true", help="only documents every listed model answered")
    parser.add_argument("--tag", default=None, help="suffix of the files written under scores/ (default: 'paired' with --paired)")
    parser.add_argument("--outputs", default="outputs", help="subdirectory of the benchmark that holds the answer files "
                                                              "(readers that cover part of the benchmark go in their own one)")
    args = parser.parse_args(argv)
    suffix = f".{args.tag}" if args.tag else (".paired" if args.paired else "")
    benchmark = load_benchmark(args.benchmark)
    models = args.models or sorted(p.stem for p in (args.benchmark / args.outputs).glob("*.jsonl"))
    answers = {m: load_outputs(args.benchmark / args.outputs / f"{m}.jsonl") for m in models}
    only = None
    if args.paired and models:
        only = sorted(set.intersection(*(set(a) & set(benchmark) for a in answers.values())))
    (args.benchmark / "scores").mkdir(exist_ok=True)
    summaries = {}
    for model in models:
        rows = score_model(benchmark, answers[model], model, only)
        summaries[model] = summarise(rows)
        with open(args.benchmark / "scores" / f"{model}{suffix}.csv", "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    with open(args.benchmark / "scores" / f"summary{suffix}.json", "w", encoding="utf-8") as fh:
        json.dump(summaries, fh, ensure_ascii=False, indent=1)
    print(f"benchmark: {len(benchmark)} documents" + (f"; paired on {len(only)}" if only is not None else ""))
    print(markdown_table(summaries))


if __name__ == "__main__":
    main()
