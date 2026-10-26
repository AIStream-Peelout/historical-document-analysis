# File name: build_prompt_ablation_panel.py
# Date: 10/24/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""A fixed, stratified image panel for the Arabic-script prompt ablation (offline, no model needed).

The ablation decodes the same images under several prompts and output budgets, so the images must be
chosen once and stratified by what they carry, read off the coverage map (``coverage_v2.jsonl`` of
``arabic_bench_coverage.py``):

* ``covered``: images with edited Arabic text (role ``covered``). Whole documents are taken, every covered
  image of a chosen document, so the covered-only scorer (``score_arabic_covered.py``) scores the panel's
  documents exactly as it scores a full run. Documents are drawn from quantile bins of a reference
  per-document F1 (one earlier run's covered-only scores) so the panel spans easy and hard pages instead
  of leaning on either end.
* ``hebrew_side``: uncovered images whose side is in Hebrew script (role ``uncovered``, script ``hebrew``):
  Arabic letters written here are hallucinations.
* ``blank_faint``: images with no readable ink (role ``blank_or_faint``).
* ``arabic_unedited``: uncovered sides in Arabic script that the edition does not cover (reading them is
  right; the benchmark cannot credit it).
* ``other_uncovered``: uncovered images of mixed or unknown script.

Copies of a photograph (role ``copy_of``) and ambiguous images are never in the panel. Within a class,
images of the benchmark's first ``--first`` documents (the paired set other runs report on) are taken
before the rest. Drawing is seeded and the panel is written once; later runs read the file.

Output: ``<out>.jsonl`` (one row per image: ``doc_id``, ``image_index``, ``image_file``, ``image_class``,
``script``, ``reference_f1``) and ``<out>.docs.csv`` (the covered documents, for the scorer's ``--paired-ids``).

Usage::

    .venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.build_prompt_ablation_panel \\
        --out <benchmark>/panels/prompt_ablation_v1 [--benchmark DIR] [--coverage FILE] \\
        [--reference-scores CSV] [--first 60] [--seed N] [--quota covered=24 hebrew_side=15 ...]
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.datasets.evaluations.helper_eval_scripts.score_arabic_benchmark import DEFAULT_BENCHMARK, load_benchmark

COVERED, HEBREW_SIDE, BLANK, ARABIC_UNEDITED, OTHER_UNCOVERED = ("covered", "hebrew_side", "blank_faint",
                                                                 "arabic_unedited", "other_uncovered")
CLASSES = (COVERED, HEBREW_SIDE, BLANK, ARABIC_UNEDITED, OTHER_UNCOVERED)
DEFAULT_QUOTAS = {COVERED: 24, HEBREW_SIDE: 15, BLANK: 10, ARABIC_UNEDITED: 5, OTHER_UNCOVERED: 6}
DEFAULT_SEED = 20261024
DEFAULT_FIRST = 60
DEFAULT_BINS = 4
DEFAULT_COVERAGE = "coverage_v2.jsonl"
DEFAULT_REFERENCE = "scores_covered_v2/qwen3-vl-8b-heb-v25a-step1200__trained.csv"
PANEL_FIELDS = ("doc_id", "image_index", "image_file", "image_class", "script", "reference_f1", "benchmark_position")


def image_class(row: Mapping[str, Any]) -> Optional[str]:
    """The panel class of a coverage-map image row, or None when the image is not eligible.

    :param row: An image row of ``coverage_v2.jsonl`` (``role`` and ``script``).
    :type row: Mapping[str, Any]
    :return: One of :data:`CLASSES`, or None for ambiguous images and copies of a photograph.
    :rtype: Optional[str]
    """
    role, script = row.get("role"), row.get("script")
    if role == "covered":
        return COVERED
    if role == "blank_or_faint":
        return BLANK
    if role == "uncovered":
        if script == "hebrew":
            return HEBREW_SIDE
        if script == "arabic":
            return ARABIC_UNEDITED
        return OTHER_UNCOVERED
    return None


def load_image_rows(coverage: Path) -> List[Dict[str, Any]]:
    """The image rows of a coverage map.

    :param coverage: ``coverage_v*.jsonl``.
    :type coverage: Path
    :return: Rows with ``kind == "image"`` in file order.
    :rtype: List[Dict[str, Any]]
    """
    with open(coverage, encoding="utf-8") as fh:
        return [r for r in (json.loads(line) for line in fh if line.strip()) if r.get("kind") == "image"]


def load_reference_f1(scores_csv: Optional[Path]) -> Dict[str, float]:
    """Per-document reference F1 from a covered-only scores CSV.

    :param scores_csv: ``scores_covered*/<tag>.csv`` (columns ``doc_id``, ``ngram_f1``), or None.
    :type scores_csv: Optional[Path]
    :return: ``{doc_id: f1}``; empty when no file is given. Documents without a score are left out.
    :rtype: Dict[str, float]
    """
    if scores_csv is None:
        return {}
    with open(scores_csv, encoding="utf-8") as fh:
        return {r["doc_id"]: float(r["ngram_f1"]) for r in csv.DictReader(fh) if r.get("ngram_f1") not in (None, "", "None")}


def quantile_bins(items: Sequence[Tuple[str, float]], bins: int) -> List[List[str]]:
    """Split documents into equal-count bins of a score, lowest first.

    :param items: ``(doc_id, score)`` pairs.
    :type items: Sequence[Tuple[str, float]]
    :param bins: Number of bins (at least 1).
    :type bins: int
    :return: Document ids per bin, ordered by score within a bin.
    :rtype: List[List[str]]
    """
    ordered = [d for d, _ in sorted(items, key=lambda x: (x[1], x[0]))]
    bins = max(1, min(bins, len(ordered)))
    size, rest = divmod(len(ordered), bins)
    out, start = [], 0
    for b in range(bins):
        end = start + size + (1 if b < rest else 0)
        out.append(ordered[start:end])
        start = end
    return out


def pick_covered_documents(candidates: Mapping[str, Sequence[int]], reference: Mapping[str, float], target_images: int,
                           rng: random.Random, bins: int = DEFAULT_BINS) -> List[str]:
    """Covered documents whose covered images add up to the target, drawn round-robin from F1 bins.

    Documents without a reference score form their own bin. A document whose images would overshoot the
    target is skipped in favour of a smaller one; the draw stops when no document fits.

    :param candidates: ``{doc_id: covered image indices}``.
    :type candidates: Mapping[str, Sequence[int]]
    :param reference: ``{doc_id: reference F1}``.
    :type reference: Mapping[str, float]
    :param target_images: Number of covered images wanted.
    :type target_images: int
    :param rng: Seeded generator.
    :type rng: random.Random
    :param bins: Number of F1 bins.
    :type bins: int
    :return: Chosen document ids in draw order.
    :rtype: List[str]
    """
    scored = [(d, reference[d]) for d in candidates if d in reference]
    pools = [list(b) for b in quantile_bins(scored, bins)]
    unscored = sorted(d for d in candidates if d not in reference)
    if unscored:
        pools.append(unscored)
    for pool in pools:
        rng.shuffle(pool)
    chosen: List[str] = []
    images = 0
    progress = True
    while images < target_images and progress:
        progress = False
        for pool in pools:
            room = target_images - images
            fit = next((d for d in pool if len(candidates[d]) <= room), None)
            if fit is None:
                continue
            pool.remove(fit)
            chosen.append(fit)
            images += len(candidates[fit])
            progress = True
            if images >= target_images:
                break
    return chosen


def pick_images(rows: Sequence[Mapping[str, Any]], quota: int, preferred: Iterable[str], rng: random.Random) -> List[Mapping[str, Any]]:
    """Up to ``quota`` images, those of the preferred documents first, each part drawn at random.

    :param rows: Candidate image rows of one class.
    :type rows: Sequence[Mapping[str, Any]]
    :param quota: Images wanted.
    :type quota: int
    :param preferred: Document ids taken before the others.
    :type preferred: Iterable[str]
    :param rng: Seeded generator.
    :type rng: random.Random
    :return: Chosen rows, preferred ones first.
    :rtype: List[Mapping[str, Any]]
    """
    wanted = set(preferred)
    first = [r for r in rows if r["doc_id"] in wanted]
    rest = [r for r in rows if r["doc_id"] not in wanted]
    rng.shuffle(first)
    rng.shuffle(rest)
    return (first + rest)[:quota]


def build_panel(benchmark: Mapping[str, Mapping[str, Any]], image_rows: Sequence[Mapping[str, Any]], reference: Mapping[str, float],
                quotas: Mapping[str, int] = DEFAULT_QUOTAS, first: int = DEFAULT_FIRST, seed: int = DEFAULT_SEED,
                bins: int = DEFAULT_BINS) -> List[Dict[str, Any]]:
    """The panel rows.

    :param benchmark: :func:`score_arabic_benchmark.load_benchmark` result (document order matters).
    :type benchmark: Mapping[str, Mapping[str, Any]]
    :param image_rows: :func:`load_image_rows` result.
    :type image_rows: Sequence[Mapping[str, Any]]
    :param reference: :func:`load_reference_f1` result.
    :type reference: Mapping[str, float]
    :param quotas: Images per class (``covered`` counts images, taken as whole documents).
    :type quotas: Mapping[str, int]
    :param first: Documents of the benchmark whose images are preferred (the paired set); covered documents
        come only from these.
    :type first: int
    :param seed: Random seed.
    :type seed: int
    :param bins: F1 bins for the covered draw.
    :type bins: int
    :return: Panel rows (:data:`PANEL_FIELDS`), covered documents first, in benchmark order within a class.
    :rtype: List[Dict[str, Any]]
    """
    position = {d: n for n, d in enumerate(benchmark)}
    preferred = list(benchmark)[:first]
    rng = random.Random(seed)
    by_class: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in image_rows:
        cls = image_class(row)
        if cls is not None and row["doc_id"] in position:
            by_class[cls].append(row)
    covered_candidates: Dict[str, List[int]] = defaultdict(list)
    for row in by_class[COVERED]:
        if row["doc_id"] in set(preferred):
            covered_candidates[row["doc_id"]].append(int(row["image_index"]))
    chosen_docs = set(pick_covered_documents(covered_candidates, reference, quotas.get(COVERED, 0), rng, bins))
    chosen: List[Mapping[str, Any]] = [r for r in by_class[COVERED] if r["doc_id"] in chosen_docs]
    for cls in CLASSES[1:]:
        chosen.extend(pick_images(by_class[cls], quotas.get(cls, 0), preferred, rng))

    def key(row: Mapping[str, Any]) -> Tuple[int, int, int]:
        return CLASSES.index(image_class(row)), position[row["doc_id"]], int(row["image_index"])

    panel = []
    for row in sorted(chosen, key=key):
        panel.append({"doc_id": row["doc_id"], "image_index": int(row["image_index"]), "image_file": row["image_file"],
                      "image_class": image_class(row), "script": row.get("script"),
                      "reference_f1": reference.get(row["doc_id"]) if image_class(row) == COVERED else None,
                      "benchmark_position": position[row["doc_id"]]})
    return panel


def write_panel(panel: Sequence[Mapping[str, Any]], out: Path) -> Tuple[Path, Path]:
    """Write the panel and its covered-document list.

    :param panel: :func:`build_panel` result.
    :type panel: Sequence[Mapping[str, Any]]
    :param out: Output stem (``.jsonl`` and ``.docs.csv`` are appended).
    :type out: Path
    :return: ``(panel path, documents CSV path)``.
    :rtype: Tuple[Path, Path]
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    panel_path, docs_path = out.with_suffix(".jsonl"), out.with_suffix(".docs.csv")
    with open(panel_path, "w", encoding="utf-8") as fh:
        for row in panel:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    docs = sorted({r["doc_id"] for r in panel if r["image_class"] == COVERED})
    with open(docs_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["doc_id"])
        writer.writeheader()
        writer.writerows({"doc_id": d} for d in docs)
    return panel_path, docs_path


def load_panel(path: Path) -> List[Dict[str, Any]]:
    """Read a panel file.

    :param path: ``<panel>.jsonl``.
    :type path: Path
    :return: Rows in file order.
    :rtype: List[Dict[str, Any]]
    """
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def summary(panel: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Counts per class and the covered documents.

    :param panel: Panel rows.
    :type panel: Sequence[Mapping[str, Any]]
    :return: ``{"images": n, "by_class": {class: n}, "covered_documents": n}``.
    :rtype: Dict[str, Any]
    """
    by_class = {c: sum(r["image_class"] == c for r in panel) for c in CLASSES}
    return {"images": len(panel), "by_class": by_class,
            "covered_documents": len({r["doc_id"] for r in panel if r["image_class"] == COVERED})}


def parse_quotas(values: Sequence[str]) -> Dict[str, int]:
    """``class=n`` arguments over the defaults.

    :param values: ``["covered=24", ...]``.
    :type values: Sequence[str]
    :return: Quotas per class.
    :rtype: Dict[str, int]
    :raises ValueError: On an unknown class.
    """
    quotas = dict(DEFAULT_QUOTAS)
    for value in values:
        cls, _, n = value.partition("=")
        if cls not in CLASSES:
            raise ValueError(f"unknown class {cls!r}; expected one of {CLASSES}")
        quotas[cls] = int(n)
    return quotas


def main(argv: Optional[List[str]] = None) -> None:
    """CLI.

    :param argv: Arguments.
    :type argv: Optional[List[str]]
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--coverage", type=Path, default=None, help=f"default: <benchmark>/{DEFAULT_COVERAGE}")
    parser.add_argument("--reference-scores", type=Path, default=None,
                        help=f"covered-only per-document CSV for the F1 bins; default: <benchmark>/{DEFAULT_REFERENCE}; 'none' for no bins")
    parser.add_argument("--out", type=Path, required=True, help="output stem (.jsonl and .docs.csv are written)")
    parser.add_argument("--first", type=int, default=DEFAULT_FIRST)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument("--quota", nargs="*", default=[], help="class=n overrides")
    args = parser.parse_args(argv)
    benchmark = load_benchmark(args.benchmark)
    image_rows = load_image_rows(args.coverage or args.benchmark / DEFAULT_COVERAGE)
    ref_path = None if str(args.reference_scores) == "none" else (args.reference_scores or args.benchmark / DEFAULT_REFERENCE)
    panel = build_panel(benchmark, image_rows, load_reference_f1(ref_path), parse_quotas(args.quota), args.first, args.seed, args.bins)
    paths = write_panel(panel, args.out)
    print(json.dumps(summary(panel), indent=1))
    print(f"wrote {paths[0]} and {paths[1]}")


if __name__ == "__main__":
    main()
