# File name: verify_decontamination.py
# Date: 8/9/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.
"""Verify the Genizah training manifest is disjoint from the frozen benchmark.

Checks three identity levels between every training doc and every benchmark
fragment (both the frozen 150 and the verified 131):

1. exact ``doc_id``,
2. canonical shelfmark (:meth:`ShelfmarkNormalizer.to_canonical_id`),
3. loose fuzzy key (:meth:`ShelfmarkNormalizer.loose_key` of the canonical
   ID — digit-boundary-safe, so dotted classmark numbers cannot collide the
   way plain alphanumeric squashing did with ENA NS 2.26 vs 22.6).

Exits nonzero on any overlap so it can gate dataset rebuilds.

Usage::

    python -m src.datasets.cleaning.verify_decontamination [--manifest PATH]
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer  # noqa: E402

_BENCH_DIR = _REPO / "src/datasets/raw_data/cairo_genizah/evaluations/genizah_test_v1"
DEFAULT_MANIFEST = _REPO / "src/datasets/raw_data/genizah_cleanup/train_clean_manifest_repaired.jsonl"


def _keys(doc_id: str, shelf_mark: str) -> Tuple[str, str, str]:
    """Build the three identity keys for one document.

    :param doc_id: Pipeline document id.
    :param shelf_mark: Raw shelfmark string.
    :returns: (doc_id, canonical id, loose key) tuple.
    """
    canonical = ShelfmarkNormalizer.to_canonical_id(shelf_mark)
    return doc_id, canonical, ShelfmarkNormalizer.loose_key(canonical)


def check(manifest_path: Path) -> List[str]:
    """Compare training docs against both benchmark files.

    :param manifest_path: Training manifest JSONL.
    :returns: List of human-readable overlap descriptions (empty = clean).
    """
    train: Dict[str, Set[str]] = {"doc_id": set(), "canonical": set(), "loose": set()}
    for line in open(manifest_path):
        rec = json.loads(line)
        d, c, l = _keys(rec["doc_id"], rec["shelf_mark"])
        train["doc_id"].add(d)
        train["canonical"].add(c)
        train["loose"].add(l)

    problems: List[str] = []
    for bench_name in ("genizah_test_v1.json", "genizah_test_v1_verified.json"):
        bench_path = _BENCH_DIR / bench_name
        if not bench_path.exists():
            continue
        spec = json.loads(bench_path.read_text())
        for doc in spec["docs"]:
            d, c, l = _keys(doc["doc_id"], doc["shelf_mark"])
            if d in train["doc_id"]:
                problems.append(f"{bench_name}: doc_id overlap {d}")
            elif c and c in train["canonical"]:
                problems.append(f"{bench_name}: canonical overlap {c} ({d})")
            elif l and l in train["loose"]:
                problems.append(f"{bench_name}: loose-key overlap {l} ({d})")
    return problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    problems = check(args.manifest)
    if problems:
        print(f"CONTAMINATION: {len(problems)} overlap(s)")
        for p in problems:
            print(f"  {p}")
        sys.exit(1)
    print("decontamination clean: no doc_id / canonical / loose-key overlap "
          "with genizah_test_v1 or the verified subset")
