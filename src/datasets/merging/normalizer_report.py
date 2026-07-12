#!/usr/bin/env python3
"""Shelfmark normalization coverage report.

Measures how well :class:`ShelfmarkNormalizer.to_canonical_id` unifies
shelfmarks across the three Cairo Genizah sources. Three metrics:

1. **Ground-truth self-consistency** — every ``shelfmarks_historic`` variant in
   PGP ``fragments.csv`` should normalize to the same canonical id as its
   fragment's primary shelfmark. This is the objective regression metric the
   normalizer is tuned against.
2. **FJP -> PGP match rate** — fraction of FJP records whose canonical id exists
   in the PGP fragment set.
3. **KTIV -> PGP match rate** — same for KTIV.

Run::

    python -m src.datasets.merging.normalizer_report
    python -m src.datasets.merging.normalizer_report --show-misses 40

The script is read-only; it never writes merged output.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import sys
from typing import Dict, List, Set, Tuple

# Anchor everything to the repo root (three levels up from this file) so the
# script works regardless of the current working directory, and add it to
# sys.path so ``from src...`` resolves even when run by file path (e.g. the
# PyCharm debugger) rather than as ``python -m``.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer  # noqa: E402

RAW_DIR = os.path.join(_REPO_ROOT, "src", "datasets", "raw_data", "cairo_genizah")
PGP_FRAGMENTS = os.path.join(RAW_DIR, "pgp_raw", "data", "fragments.csv")
FJP_FILE = os.path.join(
    RAW_DIR, "fjp_all", "merged_princeton_friedberger_all_documents_final.json"
)
KTIV_GLOB = os.path.join(RAW_DIR, "ktiv", "*.json")


def load_pgp_canonical_ids(path: str = PGP_FRAGMENTS) -> Set[str]:
    """Return the set of canonical ids for every PGP fragment primary shelfmark.

    :param path: Path to PGP ``fragments.csv`` (BOM-encoded ``utf-8-sig``).
    :returns: Set of non-empty canonical ids.
    """
    ids: Set[str] = set()
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            cid = ShelfmarkNormalizer.to_canonical_id(row["shelfmark"])
            if cid:
                ids.add(cid)
    return ids


def load_pgp_alias_index(path: str = PGP_FRAGMENTS) -> Dict[str, str]:
    """Build a ``variant_cid -> primary_cid`` index from PGP fragments.

    Includes each fragment's primary shelfmark plus every ``;``-separated entry
    in ``shelfmarks_historic``. A primary id never gets overwritten by a variant.

    :param path: Path to PGP ``fragments.csv``.
    :returns: Mapping from any known canonical-id variant to its primary id.
    """
    alias: Dict[str, str] = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            primary = ShelfmarkNormalizer.to_canonical_id(row["shelfmark"])
            if not primary:
                continue
            alias[primary] = primary
            historic = row.get("shelfmarks_historic") or ""
            for variant in (v.strip() for v in historic.split(";") if v.strip()):
                cid = ShelfmarkNormalizer.to_canonical_id(variant)
                if cid:
                    alias.setdefault(cid, primary)
    return alias


def ground_truth_consistency(
    path: str = PGP_FRAGMENTS, limit_misses: int = 0
) -> Tuple[int, int, List[Tuple[str, str, str, str]]]:
    """Measure historic-variant self-consistency against PGP primary shelfmarks.

    For each fragment, every ``;``-separated entry in ``shelfmarks_historic`` is
    normalized and compared to the canonical id of the primary ``shelfmark``.

    :param path: Path to PGP ``fragments.csv``.
    :param limit_misses: Max number of disagreement samples to collect (0 = none).
    :returns: ``(agree, total, samples)`` where ``samples`` is a list of
        ``(primary_shelfmark, historic_variant, primary_cid, variant_cid)``.
    """
    agree = 0
    total = 0
    samples: List[Tuple[str, str, str, str]] = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            primary = ShelfmarkNormalizer.to_canonical_id(row["shelfmark"])
            if not primary:
                continue
            historic = row.get("shelfmarks_historic") or ""
            for variant in (v.strip() for v in historic.split(";") if v.strip()):
                total += 1
                cid = ShelfmarkNormalizer.to_canonical_id(variant)
                if cid == primary:
                    agree += 1
                elif len(samples) < limit_misses:
                    samples.append((row["shelfmark"], variant, primary, cid))
    return agree, total, samples


def fjp_match_rate(
    alias: Dict[str, str], path: str = FJP_FILE, limit_misses: int = 0
) -> Tuple[int, int, collections.Counter, List[Tuple[str, str]]]:
    """Measure how many FJP records map (via alias index) to a PGP fragment.

    :param alias: PGP alias index from :func:`load_pgp_alias_index`.
    :param path: Path to the FJP merged JSON.
    :param limit_misses: Max number of miss samples to collect.
    :returns: ``(matched, total, miss_by_institution, samples)``.
    """
    with open(path, encoding="utf-8") as fh:
        fjp = json.load(fh)
    matched = 0
    total = 0
    miss_by_inst: collections.Counter = collections.Counter()
    samples: List[Tuple[str, str]] = []
    for rec in fjp.values():
        sm = rec.get("shelf_mark") or ""
        cid = ShelfmarkNormalizer.to_canonical_id(sm)
        if not cid:
            continue
        total += 1
        if cid in alias:
            matched += 1
        else:
            miss_by_inst[rec.get("institution") or "?"] += 1
            if len(samples) < limit_misses:
                samples.append((sm, cid))
    return matched, total, miss_by_inst, samples


def ktiv_match_rate(
    alias: Dict[str, str], pattern: str = KTIV_GLOB, limit_misses: int = 0
) -> Tuple[int, int, List[Tuple[str, str]]]:
    """Measure how many distinct KTIV shelfmarks map (via alias) to a PGP fragment.

    :param alias: PGP alias index.
    :param pattern: Glob for KTIV per-manuscript JSON files.
    :param limit_misses: Max number of miss samples to collect.
    :returns: ``(matched, total_distinct, samples)``.
    """
    seen: Dict[str, str] = {}
    for fpath in glob.glob(pattern):
        with open(fpath, encoding="utf-8") as fh:
            doc = json.load(fh)
        sm = doc.get("shelf_mark") or ""
        cid = ShelfmarkNormalizer.to_canonical_id(sm)
        if cid:
            seen.setdefault(cid, sm)
    matched = sum(1 for cid in seen if cid in alias)
    samples = [(sm, cid) for cid, sm in seen.items() if cid not in alias][
        :limit_misses
    ]
    return matched, len(seen), samples


def main() -> None:
    """Print the three coverage metrics and optional miss samples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-misses",
        type=int,
        default=0,
        metavar="N",
        help="Show N sample misses per metric.",
    )
    args = parser.parse_args()

    pgp_ids = load_pgp_canonical_ids()
    alias = load_pgp_alias_index()
    print(f"PGP fragments (distinct canonical ids): {len(pgp_ids)}")
    print(f"PGP alias index (primary + historic variants): {len(alias)}")

    agree, total, gt_samples = ground_truth_consistency(
        limit_misses=args.show_misses
    )
    pct = 100 * agree / total if total else 0.0
    print(f"\n[1] Ground-truth historic-variant consistency: "
          f"{agree}/{total} = {pct:.1f}%")
    for primary_sm, variant, p_cid, v_cid in gt_samples:
        print(f"    MISS {primary_sm!r} | {variant!r}")
        print(f"         primary={p_cid!r}  variant={v_cid!r}")

    fjp_matched, fjp_total, fjp_miss_inst, fjp_samples = fjp_match_rate(
        alias, limit_misses=args.show_misses
    )
    fpct = 100 * fjp_matched / fjp_total if fjp_total else 0.0
    print(f"\n[2] FJP -> PGP match (via alias): {fjp_matched}/{fjp_total} = {fpct:.1f}%"
          f"  ({fjp_total - fjp_matched} become union-only records)")
    if args.show_misses:
        print("    top unmatched institutions:")
        for inst, c in fjp_miss_inst.most_common(10):
            print(f"      {c:6}  {inst}")
        for sm, cid in fjp_samples:
            print(f"    MISS {sm!r} -> {cid!r}")

    ktiv_matched, ktiv_total, ktiv_samples = ktiv_match_rate(
        alias, limit_misses=args.show_misses
    )
    kpct = 100 * ktiv_matched / ktiv_total if ktiv_total else 0.0
    print(f"\n[3] KTIV -> PGP match (via alias): {ktiv_matched}/{ktiv_total} = {kpct:.1f}%")
    for sm, cid in ktiv_samples:
        print(f"    MISS {sm!r} -> {cid!r}")


if __name__ == "__main__":
    main()
