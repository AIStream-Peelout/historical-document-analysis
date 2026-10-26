# File name: score_arabic_covered.py
# Date: 10/24/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Score the Arabic-script benchmark on its coverage map (offline, no model needed).

``score_arabic_benchmark.py`` concatenates the answers to every image of a document and compares
them with the whole edition, so a reader is penalised for reading a side the edition does not cover
and cannot be told apart from a reader that stops early. With a coverage map (written by
``arabic_bench_coverage.py``) this module splits a model's answers into three parts:

* **A, recognition** on covered images: per document, the answers to the covered images only against
  only the edition sections those images carry (in edition order, with the scorer's reverse-order
  alternative for a two-section reference). The metrics are the scorer's own
  (:func:`score_arabic_benchmark.score_document` on a reduced record): clipped 5-gram precision /
  recall / F1, LER, status, length ratio and the wrong-document floor. Answers are joined in image
  order; when every image of a document is covered, no image is a copy, no unit is joined and every
  section is carried, the row equals the scorer's.
* **B, over-reading** on ``uncovered`` and ``blank_or_faint`` images (no edited text on them): Arabic
  and Hebrew letters emitted per image, whether at least :data:`OVERREAD_MIN_ARABIC` Arabic letters
  were emitted, loops (``arabic_script.repeat_share`` above the scorer's loop threshold) and
  Hebrew-script output, overall and per group: uncovered images by the script of the side (Arabic,
  Hebrew, mixed, unknown) and blank / faint images. Reading an unedited Arabic side is not an error of
  the reader; Arabic letters on a Hebrew-script side or on a blank image are.
* **C, ambiguous** images (the coverage map could not decide) are excluded from both and counted.

Two map versions are read:

* **v1** (``coverage_v1.jsonl``): per image ``status`` (``covered`` with the edition sections it carries,
  ``uncovered``, ``blank_or_faint`` or ``ambiguous``) and ``duplicate_of`` for copies of one
  photograph; the script of a side comes from the coverage map's pooled reads (``arabic_share``), which
  are the scored models' own reads. Behaviour unchanged from the first version of this module
  (``--coverage <benchmark>/coverage_v1.jsonl`` reproduces ``scores_covered/``).
* **v2** (``coverage_v2.jsonl``, the default): per image a ``role`` (``covered`` / ``uncovered`` /
  ``blank_or_faint`` / ``ambiguous`` / ``copy_of``) and a ``script`` that does not come from the
  scored models (by-eye overrides, the edition's script on covered images, ``none`` on blank ones).
  A unit spread over several images (``joined_with``, e.g. one section photographed as two pieces) is
  scored once: all its images must be covered (a map that leaves one out of recognition is rejected),
  the document's covered answers are joined unit by unit in edition order, the images of a joined unit
  in image order, and LER is the minimum over the orders of each joined unit's images, because the
  map does not record which image holds the start of the unit (the counterpart of the scorer's
  reverse-order alternative). Other documents keep image order.

Copies of one photograph (v1 ``duplicate_of``, v2 role ``copy_of``) are left out of A and B by default
(the scorer counts each copy again) and counted; ``keep_duplicates`` uses them like any other image.

The module also holds the paired comparison used in the report (per-document deltas, bootstrap
confidence intervals, win / loss counts), the equivalence check against the scorer and the
comparison of two maps' scores (which documents' recognition inputs changed, and how the tables moved).

Usage::

    .venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.score_arabic_covered \\
        --tags T1 T2 ... [--sets all paired] [--paired-ids CSV] [--benchmark DIR] [--coverage FILE] \\
        [--out-dir DIR] [--compare NEW:OLD ...] [--reference-suffixes v25a ...] [--report FILE] \\
        [--diff-coverage OLD_MAP]
"""
import argparse
import csv
import itertools
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.datasets.evaluations import arabic_script as ar
from src.datasets.evaluations.helper_eval_scripts import score_arabic_benchmark as sab

DEFAULT_BENCHMARK = sab.DEFAULT_BENCHMARK
COVERED, UNCOVERED, AMBIGUOUS, BLANK = "covered", "uncovered", "ambiguous", "blank_or_faint"
RECOGNITION, OVERREAD, EXCLUDED, DUPLICATE = "recognition", "overread", "ambiguous", "duplicate"
OVERREAD_MIN_ARABIC = 40                # Arabic letters on an image without edited text that count as "read"
HEBREW_SIDE_MAX_ARABIC_SHARE = 0.5      # coverage's pooled reads below this Arabic share: a Hebrew-script side
BOOTSTRAP_RESAMPLES = 20000
BOOTSTRAP_SEED = 20261024
MAX_JOINED_ORDERS = 24                  # answer orders tried for a document with joined units (beyond: the canonical order only)
DEFAULT_COVERAGE = "coverage_v2.jsonl"
DOC_FIELDS = ["doc_id", "model", "images", "covered_images", "covered_answered", "overread_images", "blank_images", "ambiguous_images",
              "duplicate_images", "joined_images", "sections", "sections_total", "full_coverage", "sides_match",
              "gt_letters", "answer_letters", "arabic_share", "status", "ngram_precision", "ngram_recall", "ngram_f1", "ler",
              "length_ratio", "floor_f1",
              "overread_answered", "overread_arabic_letters", "overread_hebrew_letters", "overread_ge40", "overread_loops",
              "overread_hebrew_output", "ambiguous_arabic_letters", "answer_order", "ler_order"]
IMAGE_FIELDS = ["doc_id", "model", "image_index", "coverage_status", "method", "duplicate_of", "role", "side_script", "joined_with",
                "blank", "answered", "arabic_letters", "hebrew_letters", "repeat_share", "loop", "hebrew_output", "ge40"]
SIDE_SCRIPTS = ("arabic", "hebrew", "mixed", "none")
OVERREAD_GROUPS = ("uncovered", "arabic", "hebrew", "mixed", "unknown", "blank")


# --- loading ----------------------------------------------------------------------------------------------------------

def load_coverage(path: Path) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Image rows of a coverage map.

    :param path: ``coverage_v1.jsonl``.
    :type path: Path
    :return: ``{doc_id: {image_index: image row}}``.
    :rtype: Dict[str, Dict[int, Dict[str, Any]]]
    """
    coverage: Dict[str, Dict[int, Dict[str, Any]]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "image":
                coverage.setdefault(row["doc_id"], {})[int(row["image_index"])] = row
    return coverage


def read_ids(path: Path) -> List[str]:
    """Document ids of a per-document scores CSV (the paired set).

    :param path: CSV with a ``doc_id`` column.
    :type path: Path
    :return: Ids in file order.
    :rtype: List[str]
    """
    with open(path, encoding="utf-8") as fh:
        return [row["doc_id"] for row in csv.DictReader(fh)]


# --- one document -----------------------------------------------------------------------------------------------------

def image_role(row: Mapping[str, Any], keep_duplicates: bool = False) -> str:
    """Which part of the scoring an image goes to.

    :param row: The image's coverage row (v1: ``status`` and ``duplicate_of``; v2: also ``role``).
    :type row: Mapping[str, Any]
    :param keep_duplicates: Score copies of a photograph like any other image.
    :type keep_duplicates: bool
    :return: ``recognition`` (covered), ``overread`` (uncovered or blank), ``ambiguous`` or ``duplicate``.
    :rtype: str
    """
    copy = row.get("role") == "copy_of" if "role" in row else row.get("duplicate_of") is not None
    if copy and not keep_duplicates:
        return DUPLICATE
    if row["status"] == COVERED:
        return RECOGNITION
    if row["status"] in (UNCOVERED, BLANK):
        return OVERREAD
    return EXCLUDED


def side_script(row: Mapping[str, Any]) -> str:
    """Script of an image: the v2 map's model-independent ``script``, else (v1) its pooled reads' script.

    :param row: The image's coverage row.
    :type row: Mapping[str, Any]
    :return: v2: ``"arabic"``, ``"hebrew"``, ``"mixed"``, ``"none"`` (blank) or ``""`` (unknown); v1:
        ``"hebrew"``, ``"arabic"`` or ``""`` (no pooled reads).
    :rtype: str
    """
    if "script" in row:
        return row["script"] if row["script"] in SIDE_SCRIPTS else ""
    share = row.get("arabic_share")
    if share is None:
        return ""
    return "hebrew" if share < HEBREW_SIDE_MAX_ARABIC_SHARE else "arabic"


def covered_record(record: Mapping[str, Any], indices: Sequence[int], sections: Iterable[int]) -> Dict[str, Any]:
    """The record reduced to some images and the edition sections they carry.

    :param record: Benchmark record.
    :type record: Mapping[str, Any]
    :param indices: Image indices to keep.
    :type indices: Sequence[int]
    :param sections: Section positions to keep (kept in edition order).
    :type sections: Iterable[int]
    :return: A shallow copy with ``images`` and ``gt.sections`` reduced.
    :rtype: Dict[str, Any]
    """
    keep = set(indices)
    wanted = sorted(set(sections))
    return {**record, "images": [im for im in record["images"] if im["image_index"] in keep],
            "gt": {**record["gt"], "sections": [record["gt"]["sections"][s] for s in wanted]}}


def image_measures(text: Optional[str]) -> Dict[str, Any]:
    """Letters, loop and script of one answer.

    :param text: The answer to the image (None: not answered).
    :type text: Optional[str]
    :return: ``answered``, ``arabic_letters`` (after the scorer's folding), ``hebrew_letters``,
        ``repeat_share``, ``loop`` (repeat share above the scorer's loop threshold), ``hebrew_output``
        (at least the scorer's minimum letters, more Hebrew than Arabic) and ``ge40`` (at least
        :data:`OVERREAD_MIN_ARABIC` Arabic letters).
    :rtype: Dict[str, Any]
    """
    if text is None:
        return {"answered": False, "arabic_letters": 0, "hebrew_letters": 0, "repeat_share": 0.0, "loop": False,
                "hebrew_output": False, "ge40": False}
    arabic = len(ar.arabic_letters(text))
    _, hebrew, _ = ar.script_share(text)
    repeat = ar.repeat_share(text)
    return {"answered": True, "arabic_letters": arabic, "hebrew_letters": hebrew, "repeat_share": round(repeat, 4),
            "loop": repeat > sab.LOOP_REPEAT_SHARE, "hebrew_output": arabic + hebrew >= sab.MIN_ANSWER_LETTERS and hebrew > arabic,
            "ge40": arabic >= OVERREAD_MIN_ARABIC}


def joined_images(coverage: Mapping[int, Mapping[str, Any]], covered: Sequence[int]) -> List[int]:
    """Covered images that carry a unit spread over several images, checked for consistency.

    :param coverage: The document's coverage rows by image index.
    :type coverage: Mapping[int, Mapping[str, Any]]
    :param covered: Images in recognition.
    :type covered: Sequence[int]
    :return: The covered images with a non-empty ``joined_with`` (v2), in image order.
    :rtype: List[int]
    :raises ValueError: An image of a joined unit is not in recognition (the map and the roles disagree),
        so the unit would be compared with only part of its text.
    """
    out = []
    for i in covered:
        partners = coverage[i].get("joined_with") or []
        missing = [j for j in partners if j not in covered]
        if missing:
            raise ValueError(f"{coverage[i]['doc_id']}: image {i} carries a unit joined with images {missing} that are not scored")
        if partners:
            out.append(i)
    return out


def joined_blocks(coverage: Mapping[int, Mapping[str, Any]], covered: Sequence[int]) -> List[List[int]]:
    """Covered images grouped by the units they carry jointly (``joined_with``, connected components).

    :param coverage: The document's coverage rows by image index.
    :type coverage: Mapping[int, Mapping[str, Any]]
    :param covered: Images in recognition, in image order.
    :type covered: Sequence[int]
    :return: One block per joined unit (its images in image order) and one single-image block per other
        covered image, in order of their first image.
    :rtype: List[List[int]]
    :raises ValueError: As :func:`joined_images`.
    """
    joined_images(coverage, covered)
    seen: set = set()
    blocks = []
    for i in covered:
        if i in seen:
            continue
        block, stack = set(), [i]
        while stack:
            j = stack.pop()
            if j not in block:
                block.add(j)
                stack.extend(coverage[j].get("joined_with") or [])
        seen |= block
        blocks.append(sorted(block))
    return blocks


def answer_orders(coverage: Mapping[int, Mapping[str, Any]], covered: Sequence[int],
                  max_orders: int = MAX_JOINED_ORDERS) -> List[List[int]]:
    """Orders in which the covered answers may be joined; the first is the canonical one.

    Without a joined unit the only order is image order (the scorer's). With one, the blocks of
    :func:`joined_blocks` are put in edition order (by the first section they carry; ties by image) so
    that a document whose units are photographed in pieces is not scored as interleaved units, and
    each joined unit's images may come in any order: the map does not record which image holds the
    start of the unit.

    :param coverage: The document's coverage rows by image index.
    :type coverage: Mapping[int, Mapping[str, Any]]
    :param covered: Images in recognition, in image order.
    :type covered: Sequence[int]
    :param max_orders: Most orders to return; beyond it only the canonical order is returned.
    :type max_orders: int
    :return: Orders of image indices; the first has every joined unit's images in image order.
    :rtype: List[List[int]]
    """
    blocks = joined_blocks(coverage, covered)
    if all(len(block) == 1 for block in blocks):
        return [list(covered)]
    blocks.sort(key=lambda block: (min((s for i in block for s in coverage[i]["sections"]), default=math.inf), block[0]))
    choices = [list(itertools.permutations(block)) for block in blocks]
    if math.prod(len(c) for c in choices) > max_orders:
        return [[i for block in blocks for i in block]]
    return [[i for part in combo for i in part] for combo in itertools.product(*choices)]


def score_covered_document(record: Mapping[str, Any], coverage: Mapping[int, Mapping[str, Any]], pages: Mapping[int, str],
                           floor_record: Optional[Mapping[str, Any]] = None,
                           keep_duplicates: bool = False) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Score one document's answer on its coverage map.

    :param record: Benchmark record.
    :type record: Mapping[str, Any]
    :param coverage: The document's coverage rows by image index (:func:`load_coverage`).
    :type coverage: Mapping[int, Mapping[str, Any]]
    :param pages: The model's text per image index.
    :type pages: Mapping[int, str]
    :param floor_record: Another document, for the wrong-document floor (as in the scorer).
    :type floor_record: Optional[Mapping[str, Any]]
    :param keep_duplicates: Score copies of a photograph like any other image.
    :type keep_duplicates: bool
    :return: ``(document row with the fields of DOC_FIELDS except model, image rows)``. The
        recognition fields are None when no image of the document is covered. Recognition is scored
        on the answers joined in :func:`answer_orders`' canonical order, except LER, the minimum over
        its orders; ``answer_order`` and ``ler_order`` record both.
    :rtype: Tuple[Dict[str, Any], List[Dict[str, Any]]]
    """
    indices = [im["image_index"] for im in record["images"]]
    roles = {i: image_role(coverage[i], keep_duplicates) for i in indices}
    covered = [i for i in indices if roles[i] == RECOGNITION]
    joined = joined_images(coverage, covered)
    sections = sorted({s for i in covered for s in coverage[i]["sections"]})
    total_sections = len(record["gt"]["sections"])
    images = []
    for i in indices:
        measures = image_measures(pages.get(i))
        images.append({"doc_id": record["id"], "image_index": i, "coverage_status": coverage[i]["status"],
                       "method": coverage[i]["method"], "duplicate_of": coverage[i].get("duplicate_of"), "role": roles[i],
                       "side_script": side_script(coverage[i]), "joined_with": ";".join(map(str, coverage[i].get("joined_with") or [])),
                       "blank": coverage[i]["status"] == BLANK, **measures})
    overread = [im for im in images if im["role"] == OVERREAD and im["answered"]]
    row: Dict[str, Any] = {
        "doc_id": record["id"], "images": len(indices), "covered_images": len(covered),
        "covered_answered": sum(i in pages for i in covered),
        "overread_images": sum(r == OVERREAD for r in roles.values()),
        "blank_images": sum(roles[i] == OVERREAD and coverage[i]["status"] == BLANK for i in indices),
        "ambiguous_images": sum(r == EXCLUDED for r in roles.values()),
        "duplicate_images": sum(r == DUPLICATE for r in roles.values()), "joined_images": len(joined),
        "sections": ";".join(str(s) for s in sections), "sections_total": total_sections,
        "full_coverage": len(covered) == len(indices) and len(sections) == total_sections,
        "sides_match": sab.images_match_edition(record),
        "overread_answered": len(overread), "overread_arabic_letters": sum(im["arabic_letters"] for im in overread),
        "overread_hebrew_letters": sum(im["hebrew_letters"] for im in overread), "overread_ge40": sum(im["ge40"] for im in overread),
        "overread_loops": sum(im["loop"] for im in overread), "overread_hebrew_output": sum(im["hebrew_output"] for im in overread),
        "ambiguous_arabic_letters": sum(im["arabic_letters"] for im in images if im["role"] == EXCLUDED)}
    recognition_keys = ["gt_letters", "answer_letters", "arabic_share", "status", "ngram_precision", "ngram_recall", "ngram_f1",
                        "ler", "length_ratio", "floor_f1"]
    if covered:
        reduced = covered_record(record, covered, sections)
        orders = answer_orders(coverage, covered)
        scored = [sab.score_document(reduced, {p: pages[i] for p, i in enumerate(order) if i in pages}, floor_record)
                  for order in orders]
        best = min(range(len(orders)), key=lambda k: scored[k]["ler"])
        row.update({k: scored[0][k] for k in recognition_keys})
        row.update({"ler": scored[best]["ler"], "answer_order": ";".join(map(str, orders[0])),
                    "ler_order": ";".join(map(str, orders[best]))})
    else:
        row.update({k: None for k in recognition_keys + ["answer_order", "ler_order"]})
    return row, images


def score_covered_model(benchmark: Mapping[str, Mapping[str, Any]], coverage: Mapping[str, Mapping[int, Mapping[str, Any]]],
                        answers: Mapping[str, Mapping[int, str]], model: str, only: Optional[Sequence[str]] = None,
                        keep_duplicates: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Per-document and per-image rows for one model.

    Documents are taken in benchmark order, restricted as in :func:`score_arabic_benchmark.score_model`
    (answered by the model, and in ``only`` when given), and the floor document is the next one in that
    list, so the floor matches the scorer's.

    :param benchmark: :func:`score_arabic_benchmark.load_benchmark` result.
    :type benchmark: Mapping[str, Mapping[str, Any]]
    :param coverage: :func:`load_coverage` result.
    :type coverage: Mapping[str, Mapping[int, Mapping[str, Any]]]
    :param answers: :func:`score_arabic_benchmark.load_outputs` result.
    :type answers: Mapping[str, Mapping[int, str]]
    :param model: Model name for the rows.
    :type model: str
    :param only: Restrict to these document ids.
    :type only: Optional[Sequence[str]]
    :param keep_duplicates: Score copies of a photograph like any other image.
    :type keep_duplicates: bool
    :return: ``(document rows, image rows)``.
    :rtype: Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
    """
    keep = None if only is None else set(only)
    ids = [d for d in benchmark if d in answers and (keep is None or d in keep)]
    missing = [d for d in ids if d not in coverage]
    if missing:
        raise ValueError(f"{len(missing)} documents have no coverage rows, e.g. {missing[:3]}")
    doc_rows, image_rows = [], []
    for position, doc_id in enumerate(ids):
        other = benchmark[ids[(position + 1) % len(ids)]] if len(ids) > 1 else None
        row, images = score_covered_document(benchmark[doc_id], coverage[doc_id], answers[doc_id], other, keep_duplicates)
        doc_rows.append(dict(row, model=model))
        image_rows.extend(dict(im, model=model) for im in images)
    return doc_rows, image_rows


# --- aggregation ------------------------------------------------------------------------------------------------------

def _median(values: Sequence[float]) -> Optional[float]:
    """Median rounded to four places.

    :param values: Values.
    :type values: Sequence[float]
    :return: The median, None for no values.
    :rtype: Optional[float]
    """
    return round(statistics.median(values), 4) if values else None


def _share(part: int, whole: int) -> Optional[float]:
    """A count as a share of another.

    :param part: Count.
    :type part: int
    :param whole: Denominator.
    :type whole: int
    :return: ``part / whole`` rounded to four places, None for an empty denominator.
    :rtype: Optional[float]
    """
    return round(part / whole, 4) if whole else None


def overread_group_of(image: Mapping[str, Any]) -> str:
    """The over-reading group of an image in B: its side script, or ``blank``.

    :param image: Image row of :func:`score_covered_document` with role ``overread``.
    :type image: Mapping[str, Any]
    :return: ``blank`` for a ``blank_or_faint`` image, else ``arabic`` / ``hebrew`` / ``mixed`` from the
        image's ``side_script``, ``unknown`` when the script is not known.
    :rtype: str
    """
    if image.get("blank"):
        return "blank"
    script = image.get("side_script") or ""
    return script if script in ("arabic", "hebrew", "mixed") else "unknown"


def overread_stats(images: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Over-reading measures of a group of images (only answered images enter the means and shares).

    :param images: Image rows with role ``overread``.
    :type images: Sequence[Mapping[str, Any]]
    :return: ``images``, ``answered``, ``arabic_mean`` / ``arabic_median`` and ``hebrew_mean`` /
        ``hebrew_median`` (letters per answered image), ``ge40`` / ``loop`` / ``hebrew_output`` (shares of
        answered images) and ``hebrew_letter_share`` (Hebrew letters among all script letters emitted).
    :rtype: Dict[str, Any]
    """
    answered = [im for im in images if im["answered"]]
    arabic = [im["arabic_letters"] for im in answered]
    hebrew = [im["hebrew_letters"] for im in answered]
    return {"images": len(images), "answered": len(answered),
            "arabic_mean": round(statistics.mean(arabic), 1) if arabic else None, "arabic_median": _median(arabic),
            "hebrew_mean": round(statistics.mean(hebrew), 1) if hebrew else None, "hebrew_median": _median(hebrew),
            "ge40": _share(sum(im["ge40"] for im in answered), len(answered)),
            "loop": _share(sum(im["loop"] for im in answered), len(answered)),
            "hebrew_output": _share(sum(im["hebrew_output"] for im in answered), len(answered)),
            "hebrew_letter_share": _share(sum(hebrew), sum(arabic) + sum(hebrew))}


def summarise_covered(doc_rows: Sequence[Mapping[str, Any]], image_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate one model's rows.

    :param doc_rows: Document rows of :func:`score_covered_model`.
    :type doc_rows: Sequence[Mapping[str, Any]]
    :param image_rows: Image rows of :func:`score_covered_model`.
    :type image_rows: Sequence[Mapping[str, Any]]
    :return: ``documents``; A (``recognition_*``: medians and counts over documents with a covered
        image); B (``overread_*``: over answered uncovered / blank images; ``_arabic_side`` /
        ``_hebrew_side`` / ``_mixed_side`` split by the image's script (v2: model-independent; v1: the
        coverage reads' script), ``_blank`` over ``blank_or_faint`` images; ``overread_<group>_<stat>`` with
        the :func:`overread_stats` of every group of :data:`OVERREAD_GROUPS`, ``uncovered`` being every
        uncovered image that is not blank); C (``ambiguous_*``) and the copies left out.
    :rtype: Dict[str, Any]
    """
    rec = [r for r in doc_rows if r["ngram_f1"] is not None]
    col = lambda name: [r[name] for r in rec]
    over = [im for im in image_rows if im["role"] == OVERREAD and im["answered"]]
    letters = [im["arabic_letters"] for im in over]
    ambiguous = [im for im in image_rows if im["role"] == EXCLUDED]
    side = {s: [im for im in over if im["side_script"] == s and not im.get("blank")] for s in ("arabic", "hebrew", "mixed")}
    blank = [im for im in over if im.get("blank")]
    script_letters = sum(im["arabic_letters"] + im["hebrew_letters"] for im in over)
    pool = [im for im in image_rows if im["role"] == OVERREAD]
    members = {"uncovered": [im for im in pool if overread_group_of(im) != "blank"]}
    members.update({g: [im for im in pool if overread_group_of(im) == g] for g in OVERREAD_GROUPS[1:]})
    groups = {f"overread_{g}_{k}": v for g in OVERREAD_GROUPS for k, v in overread_stats(members[g]).items()}
    return {"documents": len(doc_rows), "recognition_documents": len(rec),
            "recognition_images": sum(r["covered_images"] for r in rec),
            "recognition_f1_median": _median(col("ngram_f1")),
            "recognition_f1_mean": round(statistics.mean(col("ngram_f1")), 4) if rec else None,
            "recognition_precision_median": _median(col("ngram_precision")),
            "recognition_recall_median": _median(col("ngram_recall")), "recognition_ler_median": _median(col("ler")),
            "recognition_floor_f1_median": _median(col("floor_f1")),
            "recognition_length_ratio_median": _median(col("length_ratio")),
            "recognition_f1_at_least_0.2": sum(v >= 0.2 for v in col("ngram_f1")),
            "recognition_f1_at_least_0.5": sum(v >= 0.5 for v in col("ngram_f1")),
            "recognition_loop": sum(s == "loop" for s in col("status")),
            "recognition_wrong_script": sum(s == "wrong_script" for s in col("status")),
            "recognition_empty": sum(s == "empty" for s in col("status")),
            "recognition_full_coverage_documents": sum(bool(r["full_coverage"]) for r in rec),
            "overread_images": sum(im["role"] == OVERREAD for im in image_rows), "overread_answered": len(over),
            "overread_arabic_letters_mean": round(statistics.mean(letters), 1) if letters else None,
            "overread_arabic_letters_median": _median(letters),
            "overread_ge40_share": _share(sum(im["ge40"] for im in over), len(over)),
            "overread_loop_share": _share(sum(im["loop"] for im in over), len(over)),
            "overread_hebrew_output_share": _share(sum(im["hebrew_output"] for im in over), len(over)),
            "overread_hebrew_letter_share": _share(sum(im["hebrew_letters"] for im in over), script_letters),
            "overread_arabic_side_images": len(side["arabic"]),
            "overread_ge40_share_arabic_side": _share(sum(im["ge40"] for im in side["arabic"]), len(side["arabic"])),
            "overread_hebrew_side_images": len(side["hebrew"]),
            "overread_ge40_share_hebrew_side": _share(sum(im["ge40"] for im in side["hebrew"]), len(side["hebrew"])),
            "overread_mixed_side_images": len(side["mixed"]),
            "overread_ge40_share_mixed_side": _share(sum(im["ge40"] for im in side["mixed"]), len(side["mixed"])),
            "overread_blank_images": len(blank),
            "overread_ge40_share_blank": _share(sum(im["ge40"] for im in blank), len(blank)),
            "overread_arabic_letters_mean_blank": round(statistics.mean(im["arabic_letters"] for im in blank), 1) if blank else None,
            "ambiguous_images": len(ambiguous),
            "ambiguous_arabic_letters_mean": round(statistics.mean(im["arabic_letters"] for im in ambiguous), 1) if ambiguous else None,
            "duplicate_images_left_out": sum(im["role"] == DUPLICATE for im in image_rows),
            **groups}


# --- paired comparison ------------------------------------------------------------------------------------------------

def bootstrap_ci(values: Sequence[float], statistic: str = "mean", resamples: int = BOOTSTRAP_RESAMPLES,
                 seed: int = BOOTSTRAP_SEED, level: float = 0.95) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval of a statistic of paired deltas (documents resampled).

    :param values: Per-document deltas.
    :type values: Sequence[float]
    :param statistic: ``"mean"`` or ``"median"``.
    :type statistic: str
    :param resamples: Number of resamples.
    :type resamples: int
    :param seed: Random seed.
    :type seed: int
    :param level: Confidence level.
    :type level: float
    :return: ``(low, high)``.
    :rtype: Tuple[float, float]
    """
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = data[rng.integers(0, len(data), size=(resamples, len(data)))]
    stats = draws.mean(axis=1) if statistic == "mean" else np.median(draws, axis=1)
    tail = (1.0 - level) / 2 * 100
    low, high = np.percentile(stats, [tail, 100 - tail])
    return float(low), float(high)


def paired_comparison(new_rows: Sequence[Mapping[str, Any]], old_rows: Sequence[Mapping[str, Any]], metric: str,
                      higher_is_better: bool, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> Dict[str, Any]:
    """Per-document deltas ``new - old`` of one metric on the documents both rows score.

    :param new_rows: Document rows of the newer model.
    :type new_rows: Sequence[Mapping[str, Any]]
    :param old_rows: Document rows of the older model.
    :type old_rows: Sequence[Mapping[str, Any]]
    :param metric: Column, e.g. ``ngram_f1`` or ``ler``.
    :type metric: str
    :param higher_is_better: Whether a larger value is better (F1) or worse (LER).
    :type higher_is_better: bool
    :param resamples: Bootstrap resamples.
    :type resamples: int
    :param seed: Bootstrap seed.
    :type seed: int
    :return: ``documents``, medians of both, ``mean_delta`` / ``median_delta`` with bootstrap 95 %
        intervals, and ``wins`` / ``losses`` / ``ties`` of the newer model.
    :rtype: Dict[str, Any]
    """
    old = {r["doc_id"]: r[metric] for r in old_rows if r[metric] is not None}
    pairs = [(r[metric], old[r["doc_id"]]) for r in new_rows if r[metric] is not None and r["doc_id"] in old]
    if not pairs:
        return {"documents": 0}
    deltas = [n - o for n, o in pairs]
    better = [d > 0 if higher_is_better else d < 0 for d in deltas]
    mean_ci, median_ci = bootstrap_ci(deltas, "mean", resamples, seed), bootstrap_ci(deltas, "median", resamples, seed)
    return {"documents": len(pairs), "new_median": _median([n for n, _ in pairs]), "old_median": _median([o for _, o in pairs]),
            "mean_delta": round(statistics.mean(deltas), 4), "mean_ci": [round(v, 4) for v in mean_ci],
            "median_delta": round(statistics.median(deltas), 4), "median_ci": [round(v, 4) for v in median_ci],
            "wins": sum(better), "losses": sum(d != 0 and not b for d, b in zip(deltas, better)), "ties": sum(d == 0 for d in deltas)}


# --- equivalence with the scorer --------------------------------------------------------------------------------------

def equivalence_check(doc_rows: Sequence[Mapping[str, Any]], scorer_rows: Sequence[Mapping[str, Any]],
                      fields: Sequence[str] = ("gt_letters", "answer_letters", "arabic_share", "status", "ngram_precision",
                                               "ngram_recall", "ngram_f1", "ler", "length_ratio", "floor_f1")) -> Dict[str, Any]:
    """Compare covered rows with the scorer's rows on fully covered documents without copies or joined units.

    :param doc_rows: Document rows of :func:`score_covered_model`.
    :type doc_rows: Sequence[Mapping[str, Any]]
    :param scorer_rows: The scorer's rows for the same model (in memory, or read from its CSV: values
        are compared as strings of floats rounded to four places).
    :type scorer_rows: Sequence[Mapping[str, Any]]
    :param fields: Fields to compare.
    :type fields: Sequence[str]
    :return: ``documents`` compared, ``identical`` documents and ``mismatches`` (``[doc_id, field, covered, scorer]``).
    :rtype: Dict[str, Any]
    """
    scorer = {r["doc_id"]: r for r in scorer_rows}
    full = [r for r in doc_rows if r["full_coverage"] and not r["duplicate_images"] and not r.get("joined_images")
            and r["doc_id"] in scorer]
    mismatches = []
    for row in full:
        for field in fields:
            mine, theirs = row[field], scorer[row["doc_id"]][field]
            if isinstance(theirs, str) and not isinstance(mine, str):
                equal = theirs == str(mine) or (theirs.replace(".", "", 1).isdigit() and float(theirs) == float(mine))
            else:
                equal = mine == theirs
            if not equal:
                mismatches.append([row["doc_id"], field, mine, theirs])
    return {"documents": len(full), "identical": len(full) - len({m[0] for m in mismatches}), "mismatches": mismatches}


# --- report -----------------------------------------------------------------------------------------------------------

def _fmt(value: Any) -> str:
    """A table cell.

    :param value: Value.
    :type value: Any
    :return: ``"–"`` for None, else the value as text.
    :rtype: str
    """
    return "–" if value is None else str(value)


def table_a(summaries: Mapping[str, Mapping[str, Any]], scorer: Mapping[str, Mapping[str, Any]]) -> str:
    """Markdown table of covered-only recognition per model, next to the whole-document scorer.

    :param summaries: ``{model: summarise_covered(...)}``.
    :type summaries: Mapping[str, Mapping[str, Any]]
    :param scorer: ``{model: score_arabic_benchmark.summarise(...)}`` on the same documents.
    :type scorer: Mapping[str, Mapping[str, Any]]
    :return: Table text.
    :rtype: str
    """
    head = ["model", "docs", "covered imgs", "P", "R", "F1", "LER", "F1>=0.2", "F1>=0.5", "loop", "wrong script", "len ratio",
            "floor F1", "scorer F1", "scorer LER", "scorer F1>=0.2"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for model, s in summaries.items():
        o = scorer.get(model, {})
        cells = [model, s["recognition_documents"], s["recognition_images"], s["recognition_precision_median"],
                 s["recognition_recall_median"], s["recognition_f1_median"], s["recognition_ler_median"],
                 s["recognition_f1_at_least_0.2"], s["recognition_f1_at_least_0.5"], s["recognition_loop"],
                 s["recognition_wrong_script"], s["recognition_length_ratio_median"], s["recognition_floor_f1_median"],
                 o.get("ngram_f1_median"), o.get("ler_median"), o.get("f1_at_least_0.2")]
        lines.append("| " + " | ".join(_fmt(c) for c in cells) + " |")
    return "\n".join(lines)


def table_b(summaries: Mapping[str, Mapping[str, Any]]) -> str:
    """Markdown table of over-reading per model.

    :param summaries: ``{model: summarise_covered(...)}``.
    :type summaries: Mapping[str, Mapping[str, Any]]
    :return: Table text.
    :rtype: str
    """
    head = ["model", "uncovered imgs", "mean Arabic letters/img", "median", ">=40 Arabic", "loop", "Hebrew output",
            "Hebrew letter share", ">=40 on Arabic sides (n)", ">=40 on Hebrew sides (n)", ">=40 on mixed sides (n)",
            ">=40 on blank images (n, mean letters)", "ambiguous imgs (mean letters)", "copies left out"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for model, s in summaries.items():
        cells = [model, s["overread_answered"], s["overread_arabic_letters_mean"], s["overread_arabic_letters_median"],
                 s["overread_ge40_share"], s["overread_loop_share"], s["overread_hebrew_output_share"],
                 s["overread_hebrew_letter_share"],
                 f"{_fmt(s['overread_ge40_share_arabic_side'])} ({s['overread_arabic_side_images']})",
                 f"{_fmt(s['overread_ge40_share_hebrew_side'])} ({s['overread_hebrew_side_images']})",
                 f"{_fmt(s['overread_ge40_share_mixed_side'])} ({s['overread_mixed_side_images']})",
                 f"{_fmt(s['overread_ge40_share_blank'])} ({s['overread_blank_images']}, {_fmt(s['overread_arabic_letters_mean_blank'])})",
                 f"{s['ambiguous_images']} ({_fmt(s['ambiguous_arabic_letters_mean'])})", s["duplicate_images_left_out"]]
        lines.append("| " + " | ".join(_fmt(c) for c in cells) + " |")
    return "\n".join(lines)


def short_name(tag: str) -> str:
    """A short model label for the tables.

    :param tag: Answer-file tag, e.g. ``qwen3-vl-8b-heb-v25a-step1200__trained__guarded``.
    :type tag: str
    :return: e.g. ``v25a-step1200 guarded`` (``raw`` for ``__trained``); the tag itself when it has no
        ``vNN-stepNNN`` part.
    :rtype: str
    """
    base, _, variant = tag.partition("__")
    match = re.search(r"v\d+[a-z]*-step\d+", base)
    if not match:
        return tag
    return f"{match.group(0)} {'guarded' if variant.endswith('guarded') else 'raw'}" if variant else match.group(0)


def _group(s: Mapping[str, Any], group: str, *stats: str) -> str:
    """One table cell from several statistics of an over-reading group.

    :param s: Summary of :func:`summarise_covered`.
    :type s: Mapping[str, Any]
    :param group: Group of :data:`OVERREAD_GROUPS`.
    :type group: str
    :param stats: Statistics of :func:`overread_stats`, joined with " / ".
    :type stats: str
    :return: Cell text.
    :rtype: str
    """
    return " / ".join(_fmt(s.get(f"overread_{group}_{stat}")) for stat in stats)


def table_b_uncovered(summaries: Mapping[str, Mapping[str, Any]]) -> str:
    """Markdown table of over-reading on uncovered images (not blank), overall and by side script.

    :param summaries: ``{model: summarise_covered(...)}``.
    :type summaries: Mapping[str, Mapping[str, Any]]
    :return: Table text.
    :rtype: str
    """
    head = ["model", "uncovered imgs (answered)", "mean Arabic letters/img", "median", ">=40 Arabic", "loop",
            "Arabic sides: n / mean / median / >=40", "mixed sides: n / mean / median / >=40",
            "Hebrew sides: n / mean / median / >=40", "unknown script: n / mean / >=40"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for model, s in summaries.items():
        cells = [short_name(model), s["overread_uncovered_answered"], s["overread_uncovered_arabic_mean"],
                 s["overread_uncovered_arabic_median"], s["overread_uncovered_ge40"], s["overread_uncovered_loop"],
                 _group(s, "arabic", "answered", "arabic_mean", "arabic_median", "ge40"),
                 _group(s, "mixed", "answered", "arabic_mean", "arabic_median", "ge40"),
                 _group(s, "hebrew", "answered", "arabic_mean", "arabic_median", "ge40"),
                 _group(s, "unknown", "answered", "arabic_mean", "ge40")]
        lines.append("| " + " | ".join(_fmt(c) for c in cells) + " |")
    return "\n".join(lines)


def table_b_hebrew(summaries: Mapping[str, Mapping[str, Any]]) -> str:
    """Markdown table of what readers emit on uncovered Hebrew-script sides.

    :param summaries: ``{model: summarise_covered(...)}``.
    :type summaries: Mapping[str, Mapping[str, Any]]
    :return: Table text.
    :rtype: str
    """
    head = ["model", "Hebrew sides (answered)", "mean Arabic letters/img", "median Arabic", "mean Hebrew letters/img",
            "median Hebrew", ">=40 Arabic", "Hebrew output (share of imgs)", "Hebrew letter share", "loop"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for model, s in summaries.items():
        cells = [short_name(model)] + [s.get(f"overread_hebrew_{k}") for k in
                                       ("answered", "arabic_mean", "arabic_median", "hebrew_mean", "hebrew_median", "ge40",
                                        "hebrew_output", "hebrew_letter_share", "loop")]
        lines.append("| " + " | ".join(_fmt(c) for c in cells) + " |")
    return "\n".join(lines)


def table_b_blank(summaries: Mapping[str, Mapping[str, Any]]) -> str:
    """Markdown table of letters emitted on blank / faint images, with the ambiguous images and copies left out.

    :param summaries: ``{model: summarise_covered(...)}``.
    :type summaries: Mapping[str, Mapping[str, Any]]
    :return: Table text.
    :rtype: str
    """
    head = ["model", "blank/faint imgs (answered)", "mean Arabic letters/img", "median Arabic", "mean Hebrew letters/img",
            ">=40 Arabic", "loop", "ambiguous imgs (mean Arabic letters)", "copies left out"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for model, s in summaries.items():
        cells = [short_name(model)] + [s.get(f"overread_blank_{k}") for k in ("answered", "arabic_mean", "arabic_median",
                                                                               "hebrew_mean", "ge40", "loop")]
        cells += [f"{s['ambiguous_images']} ({_fmt(s['ambiguous_arabic_letters_mean'])})", s["duplicate_images_left_out"]]
        lines.append("| " + " | ".join(_fmt(c) for c in cells) + " |")
    return "\n".join(lines)


def comparison_table(comparisons: Sequence[Tuple[str, str, str, Mapping[str, Any]]]) -> str:
    """Markdown table of paired comparisons.

    :param comparisons: ``(new model, old model, metric, paired_comparison(...))``.
    :type comparisons: Sequence[Tuple[str, str, str, Mapping[str, Any]]]
    :return: Table text.
    :rtype: str
    """
    head = ["new", "old", "metric", "docs", "old median", "new median", "mean Δ [95% CI]", "median Δ [95% CI]", "W/L/T (new better)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for new, old, metric, c in comparisons:
        cells = [new, old, metric, c["documents"], c.get("old_median"), c.get("new_median"),
                 f"{c['mean_delta']:+.4f} [{c['mean_ci'][0]:+.4f}, {c['mean_ci'][1]:+.4f}]",
                 f"{c['median_delta']:+.4f} [{c['median_ci'][0]:+.4f}, {c['median_ci'][1]:+.4f}]",
                 f"{c['wins']}/{c['losses']}/{c['ties']}"]
        lines.append("| " + " | ".join(_fmt(x) for x in cells) + " |")
    return "\n".join(lines)


# --- two maps -------------------------------------------------------------------------------------------------------

def _covered_set(row: Mapping[str, Any]) -> Tuple[Tuple[int, ...], str]:
    """The covered images and sections a document row was scored on.

    :param row: Document row of :func:`score_covered_document`.
    :type row: Mapping[str, Any]
    :return: ``(sorted covered image indices, sections)``.
    :rtype: Tuple[Tuple[int, ...], str]
    """
    order = row.get("answer_order") or ""
    return tuple(sorted(int(i) for i in order.split(";") if i != "")), row.get("sections") or ""


def recognition_changes(old_rows: Sequence[Mapping[str, Any]], new_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Documents whose recognition input differs between two maps (one model's rows on each).

    :param old_rows: Document rows on the older map.
    :type old_rows: Sequence[Mapping[str, Any]]
    :param new_rows: Document rows on the newer map, same model and documents.
    :type new_rows: Sequence[Mapping[str, Any]]
    :return: Per changed document (in the newer rows' order): ``doc_id``, ``change`` (``entered``,
        ``left``, ``images``, ``sections`` or ``order``), the covered images / sections / answer order
        and F1 / LER on both maps.
    :rtype: List[Dict[str, Any]]
    """
    old = {r["doc_id"]: r for r in old_rows}
    out = []
    for new in new_rows:
        prev = old.get(new["doc_id"])
        if prev is None:
            continue
        (old_images, old_sections), (new_images, new_sections) = _covered_set(prev), _covered_set(new)
        if not old_images and not new_images:
            continue
        if not old_images:
            change = "entered"
        elif not new_images:
            change = "left"
        elif old_images != new_images:
            change = "images"
        elif old_sections != new_sections:
            change = "sections"
        elif (prev.get("answer_order"), prev.get("ler_order")) != (new.get("answer_order"), new.get("ler_order")):
            change = "order"
        else:
            continue
        out.append({"doc_id": new["doc_id"], "change": change,
                    "old_images": ";".join(map(str, old_images)), "new_images": ";".join(map(str, new_images)),
                    "old_sections": old_sections, "new_sections": new_sections,
                    "old_order": prev.get("answer_order"), "new_order": new.get("answer_order"), "new_ler_order": new.get("ler_order"),
                    "old_f1": prev["ngram_f1"], "new_f1": new["ngram_f1"], "old_ler": prev["ler"], "new_ler": new["ler"]})
    return out


def _arrow(old: Any, new: Any) -> str:
    """``old → new`` cell.

    :param old: Value on the older map.
    :type old: Any
    :param new: Value on the newer map.
    :type new: Any
    :return: Cell text.
    :rtype: str
    """
    return f"{_fmt(old)} → {_fmt(new)}"


def diff_tables(old: Mapping[str, Any], new: Mapping[str, Any], tags: Sequence[str], compare: Sequence[str]) -> List[str]:
    """Markdown lines comparing one document set scored on two maps.

    :param old: :func:`run_set` result on the older map.
    :type old: Mapping[str, Any]
    :param new: :func:`run_set` result on the newer map.
    :type new: Mapping[str, Any]
    :param tags: Models, in table order.
    :type tags: Sequence[str]
    :param compare: ``NEW:OLD`` model pairs; the first pair's models get per-document F1 / LER columns.
    :type compare: Sequence[str]
    :return: Lines (headings, tables).
    :rtype: List[str]
    """
    lines = ["#### Table A medians (older map → this map)", "",
             "| model | docs | covered imgs | F1 | LER | F1>=0.2 | P | R |", "|---|---|---|---|---|---|---|---|"]
    for tag in tags:
        o, n = old["summary"][tag], new["summary"][tag]
        cells = [short_name(tag)] + [_arrow(o[k], n[k]) for k in ("recognition_documents", "recognition_images", "recognition_f1_median",
                                                                   "recognition_ler_median", "recognition_f1_at_least_0.2",
                                                                   "recognition_precision_median", "recognition_recall_median")]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "#### Table B, uncovered images that are not blank (older map → this map)", "",
              "| model | uncovered imgs | mean Arabic letters | median | >=40 Arabic | loop | Arabic sides n (>=40) | Hebrew sides n (>=40) | blank imgs |",
              "|---|---|---|---|---|---|---|---|---|"]
    for tag in tags:
        o, n = old["summary"][tag], new["summary"][tag]
        cells = [short_name(tag)] + [_arrow(o[f"overread_uncovered_{k}"], n[f"overread_uncovered_{k}"])
                                     for k in ("answered", "arabic_mean", "arabic_median", "ge40", "loop")]
        cells += [_arrow(f"{o[f'overread_{g}_answered']} ({_fmt(o[f'overread_{g}_ge40'])})",
                         f"{n[f'overread_{g}_answered']} ({_fmt(n[f'overread_{g}_ge40'])})") for g in ("arabic", "hebrew")]
        cells.append(_arrow(o["overread_blank_answered"], n["overread_blank_answered"]))
        lines.append("| " + " | ".join(cells) + " |")
    if compare:
        lines += ["", "#### Paired deltas (older map → this map)", "",
                  "| new | old | metric | docs | median Δ [95% CI] | mean Δ [95% CI] | W/L/T |", "|---|---|---|---|---|---|---|"]
        for pair in compare:
            a, b = pair.split(":")
            for metric, higher in (("ngram_f1", True), ("ler", False)):
                co = paired_comparison(old["docs"][a], old["docs"][b], metric, higher)
                cn = paired_comparison(new["docs"][a], new["docs"][b], metric, higher)
                fmt = lambda c, k: f"{c[k + '_delta']:+.4f} [{c[k + '_ci'][0]:+.4f}, {c[k + '_ci'][1]:+.4f}]"
                lines.append(f"| {short_name(a)} | {short_name(b)} | {metric} | {_arrow(co['documents'], cn['documents'])} | "
                             f"{fmt(co, 'median')} → {fmt(cn, 'median')} | {fmt(co, 'mean')} → {fmt(cn, 'mean')} | "
                             f"{co['wins']}/{co['losses']}/{co['ties']} → {cn['wins']}/{cn['losses']}/{cn['ties']} |")
        a, b = compare[0].split(":")
        changes_a = recognition_changes(old["docs"][a], new["docs"][a])
        changes_b = {c["doc_id"]: c for c in recognition_changes(old["docs"][b], new["docs"][b])}
        lines += ["", f"#### Documents whose recognition input changed ({len(changes_a)})", "",
                  f"| doc | change | covered imgs (sections) | answer order (LER order) | {short_name(b)} F1 | {short_name(a)} F1 | "
                  f"{short_name(b)} LER | {short_name(a)} LER | ΔF1 new−old model |", "|---|---|---|---|---|---|---|---|---|"]
        for c in changes_a:
            d = changes_b.get(c["doc_id"], {})
            delta = lambda x, y: f"{x - y:+.4f}" if x is not None and y is not None else "–"
            lines.append(f"| {c['doc_id']} | {c['change']} | {_arrow(f'{c['old_images']} ({c['old_sections']})', f'{c['new_images']} ({c['new_sections']})')} | "
                         f"{_fmt(c['new_order'])} ({_fmt(c['new_ler_order'])}) | {_arrow(d.get('old_f1'), d.get('new_f1'))} | "
                         f"{_arrow(c['old_f1'], c['new_f1'])} | {_arrow(d.get('old_ler'), d.get('new_ler'))} | {_arrow(c['old_ler'], c['new_ler'])} | "
                         f"{_arrow(delta(c['old_f1'], d.get('old_f1')), delta(c['new_f1'], d.get('new_f1')))} |")
    return lines


# --- CLI --------------------------------------------------------------------------------------------------------------

def default_out_dir(coverage: Path) -> str:
    """Output directory name for a coverage map, so a new map never writes into an older map's scores.

    :param coverage: Coverage map file.
    :type coverage: Path
    :return: ``scores_covered`` for ``coverage_v1.jsonl``, else ``scores_covered_<suffix>`` with the part of
        the stem after ``coverage_`` (``coverage_v2.jsonl`` -> ``scores_covered_v2``).
    :rtype: str
    """
    stem = coverage.stem
    suffix = stem.split("coverage_", 1)[1] if stem.startswith("coverage_") else stem
    return "scores_covered" if suffix == "v1" else f"scores_covered_{suffix}"


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    """Write rows to a CSV.

    :param path: Output file.
    :type path: Path
    :param rows: Rows.
    :type rows: Sequence[Mapping[str, Any]]
    :param fields: Columns.
    :type fields: Sequence[str]
    """
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def reference_csv(scores: Path, model: str, suffixes: Sequence[str]) -> Optional[Path]:
    """The first existing scorer CSV of a model among several suffixes.

    :param scores: ``<benchmark>/scores``.
    :type scores: Path
    :param model: Model name.
    :type model: str
    :param suffixes: Run suffixes to try, in order.
    :type suffixes: Sequence[str]
    :return: Path, or None.
    :rtype: Optional[Path]
    """
    for suffix in suffixes:
        path = scores / f"{model}.{suffix}.csv"
        if path.exists():
            return path
    return None


def run_set(benchmark: Mapping[str, Mapping[str, Any]], coverage: Mapping[str, Mapping[int, Mapping[str, Any]]],
            answers: Mapping[str, Mapping[str, Mapping[int, str]]], only: Optional[Sequence[str]],
            keep_duplicates: bool = False) -> Dict[str, Any]:
    """Score every model on one document set, with the whole-document scorer alongside.

    :param benchmark: Records by id.
    :type benchmark: Mapping[str, Mapping[str, Any]]
    :param coverage: :func:`load_coverage`.
    :type coverage: Mapping[str, Mapping[int, Mapping[str, Any]]]
    :param answers: ``{model: load_outputs(...)}``.
    :type answers: Mapping[str, Mapping[str, Mapping[int, str]]]
    :param only: Document ids (None: every document a model answered).
    :type only: Optional[Sequence[str]]
    :param keep_duplicates: Score copies of a photograph like any other image.
    :type keep_duplicates: bool
    :return: ``{"docs": {model: rows}, "images": {model: rows}, "summary": {model: ...}, "scorer_rows": {...},
        "scorer": {model: scorer summary}}``.
    :rtype: Dict[str, Any]
    """
    out: Dict[str, Any] = {"docs": {}, "images": {}, "summary": {}, "scorer_rows": {}, "scorer": {}}
    for model, model_answers in answers.items():
        docs, images = score_covered_model(benchmark, coverage, model_answers, model, only, keep_duplicates)
        scorer_rows = sab.score_model(dict(benchmark), dict(model_answers), model, only)
        out["docs"][model], out["images"][model] = docs, images
        out["summary"][model] = summarise_covered(docs, images)
        out["scorer_rows"][model], out["scorer"][model] = scorer_rows, sab.summarise(scorer_rows)
    return out


def main(argv: Optional[List[str]] = None) -> None:
    """CLI: score the tags on one or both document sets, optionally writing the Markdown report.

    :param argv: Arguments.
    :type argv: Optional[List[str]]
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--coverage", type=Path, default=None,
                        help=f"default: <benchmark>/{DEFAULT_COVERAGE}; <benchmark>/coverage_v1.jsonl gives the v1 behaviour")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--tags", nargs="+", required=True, help="answer files (outputs/<tag>.jsonl)")
    parser.add_argument("--sets", nargs="+", default=["all", "paired"], choices=["all", "paired"])
    parser.add_argument("--paired-ids", type=Path, default=None, help="per-document scores CSV whose doc_ids form the paired set")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: <benchmark>/scores_covered for coverage_v1.jsonl, <benchmark>/scores_covered_<v> for coverage_<v>.jsonl")
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--compare", nargs="*", default=[], help="NEW:OLD tag pairs for paired deltas")
    parser.add_argument("--reference-suffixes", nargs="*", default=[], help="scorer CSV suffixes to check equivalence against")
    parser.add_argument("--report", type=Path, default=None, help="Markdown report to write")
    parser.add_argument("--diff-coverage", type=Path, default=None,
                        help="another coverage map: the same tags and sets are scored on it in memory (nothing written) and the "
                             "report gets a section on what moved")
    args = parser.parse_args(argv)
    if "paired" in args.sets and args.paired_ids is None:
        parser.error("--paired-ids is required for the paired set")
    benchmark = sab.load_benchmark(args.benchmark)
    args.coverage = args.coverage or args.benchmark / DEFAULT_COVERAGE
    coverage = load_coverage(args.coverage)
    answers = {t: sab.load_outputs(args.benchmark / args.outputs / f"{t}.jsonl") for t in args.tags}
    out_dir = args.out_dir or args.benchmark / default_out_dir(args.coverage)
    out_dir.mkdir(exist_ok=True)
    results: Dict[str, Dict[str, Any]] = {}
    baseline: Dict[str, Dict[str, Any]] = {}
    old_coverage = load_coverage(args.diff_coverage) if args.diff_coverage else None
    for name in args.sets:
        only = read_ids(args.paired_ids) if name == "paired" else None
        result = run_set(benchmark, coverage, answers, only, args.keep_duplicates)
        suffix = "" if name == "all" else f".{name}"
        for model in args.tags:
            write_rows(out_dir / f"{model}{suffix}.csv", result["docs"][model], DOC_FIELDS)
            write_rows(out_dir / f"{model}{suffix}.images.csv", result["images"][model], IMAGE_FIELDS)
        summary_rows = [{"model": m, **s} for m, s in result["summary"].items()]
        write_rows(out_dir / f"summary{suffix}.csv", summary_rows, list(summary_rows[0]))
        with open(out_dir / f"summary{suffix}.json", "w", encoding="utf-8") as fh:
            json.dump({"set": name, "coverage": str(args.coverage),
                       "documents": sorted({r["doc_id"] for rows in result["docs"].values() for r in rows}),
                       "summary": result["summary"], "scorer": result["scorer"]}, fh, ensure_ascii=False, indent=1)
        results[name] = result
        if old_coverage is not None:
            baseline[name] = run_set(benchmark, old_coverage, answers, only, args.keep_duplicates)
        print(f"[{name}] {len(result['docs'][args.tags[0]])} documents")
        print(table_a(result["summary"], result["scorer"]))
        print(table_b_uncovered(result["summary"]))
    if args.report:
        args.report.write_text(build_report(args, results, benchmark, baseline), encoding="utf-8")
        print(f"wrote {args.report}")


def build_report(args: argparse.Namespace, results: Mapping[str, Mapping[str, Any]], benchmark: Mapping[str, Any],
                 baseline: Optional[Mapping[str, Mapping[str, Any]]] = None) -> str:
    """The Markdown report: tables A and B per set, paired comparisons, equivalence checks, and the diff
    against another map when one was scored.

    :param args: Parsed CLI arguments (``coverage`` set).
    :type args: argparse.Namespace
    :param results: ``{set: run_set(...)}``.
    :type results: Mapping[str, Mapping[str, Any]]
    :param benchmark: Records by id.
    :type benchmark: Mapping[str, Any]
    :param baseline: ``{set: run_set(...)}`` on ``args.diff_coverage``, or None.
    :type baseline: Optional[Mapping[str, Mapping[str, Any]]]
    :return: Report text.
    :rtype: str
    """
    parts = [f"# Arabic-script benchmark scored on the coverage map ({Path(args.coverage).name})", ""]
    for name, result in results.items():
        docs = result["docs"][args.tags[0]]
        parts += [f"## Set `{name}`: {len(docs)} documents, {sum(r['images'] for r in docs)} images "
                  f"(benchmark {len(benchmark)} documents)", "",
                  "### Table A: recognition on covered images (medians over documents with a covered image)", "",
                  table_a({short_name(m): s for m, s in result["summary"].items()},
                          {short_name(m): s for m, s in result["scorer"].items()}), "",
                  "### Table B1: over-reading on uncovered images that are not blank (shares of answered images; "
                  "cells n / mean / median / >=40 by the side's script)", "",
                  table_b_uncovered(result["summary"]), "",
                  "### Table B2: uncovered Hebrew-script sides", "", table_b_hebrew(result["summary"]), "",
                  "### Table B3: blank / faint images; ambiguous images and copies left out of A and B", "",
                  table_b_blank(result["summary"]), ""]
        comparisons = []
        for pair in args.compare:
            new, old = pair.split(":")
            for metric, higher in (("ngram_f1", True), ("ler", False)):
                comparisons.append((short_name(new), short_name(old), metric,
                                    paired_comparison(result["docs"][new], result["docs"][old], metric, higher)))
        if comparisons:
            parts += [f"### Paired deltas (covered-only, {BOOTSTRAP_RESAMPLES:,} bootstrap resamples of documents, seed {BOOTSTRAP_SEED})", "",
                      comparison_table(comparisons), ""]
        lines = ["### Equivalence on fully covered documents (every image covered, no copies, no joined unit, every section carried)", "",
                 "| model | docs | identical (in-memory scorer) | identical (scorer CSV) |", "|---|---|---|---|"]
        for model in args.tags:
            memory = equivalence_check(result["docs"][model], result["scorer_rows"][model])
            ref = reference_csv(args.benchmark / "scores", model, args.reference_suffixes) if name == "paired" else None
            on_disk = "–"
            if ref is not None:
                with open(ref, encoding="utf-8") as fh:
                    disk = equivalence_check(result["docs"][model], list(csv.DictReader(fh)))
                on_disk = f"{disk['identical']}/{disk['documents']} (`{ref.name}`)"
            lines.append(f"| {short_name(model)} | {memory['documents']} | {memory['identical']}/{memory['documents']} | {on_disk} |")
        parts += lines + [""]
        if baseline and name in baseline:
            parts += [f"### Diff against `{Path(args.diff_coverage).name}` (same tags, same documents, scored in memory)", ""]
            parts += diff_tables(baseline[name], result, args.tags, args.compare) + [""]
    return "\n".join(parts)


if __name__ == "__main__":
    main()
