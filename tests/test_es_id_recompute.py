"""T6 skeleton (slow): recompute the ids of merged records with the merge recipe and with the facade.

Skipped when the merged corpus is absent. A fixed-seed sample of :data:`SAMPLE_SIZE` records is re-derived from
each record's own source blocks with the merge's per-source recipes (``pgp_canonical_id`` + ``pgp_variant_aliases``,
``split_shelfmarks`` + the legacy-store recipe, ``ktiv_record_ids`` / ``ktiv_mark_cid``, ``bodleian_canonical_id``),
exactly as ``merge_shelfmarks`` calls them. Every source block must give back the record's id byte for byte.

The facade's ``canonical_id(shelfmark_display, institution)`` is compared too. In PR-1 its disagreement count is
only printed: the facade does not infer institutions yet and the source recipes differ from it by design (§1.4
of the migration plan). Later PRs turn it into an assertion against the known re-key list.
"""

from __future__ import annotations

import collections
import json
import random
from pathlib import Path
from typing import Any, Counter, Dict, List, Set

import pytest

from src.datasets import shelfmarks
from src.datasets.merging import merge_shelfmarks as ms

REPO = Path(__file__).resolve().parents[1]
MERGED = REPO / "src/datasets/raw_data/cairo_genizah/merged/merged_shelfmarks.jsonl"
SAMPLE_SIZE = 2000
SEED = 20261103

pytestmark = pytest.mark.skipif(not MERGED.exists(), reason="merged corpus not present")


def sample_records(path: Path, size: int, seed: int) -> List[Dict[str, Any]]:
    """A fixed-seed sample of records, parsing only the sampled lines.

    :param path: Merged JSONL file.
    :type path: Path
    :param size: Number of records.
    :type size: int
    :param seed: Random seed.
    :type seed: int
    :return: The sampled records in file order.
    :rtype: List[Dict[str, Any]]
    """
    with open(path, "rb") as fh:
        total = sum(1 for _ in fh)
    wanted = set(random.Random(seed).sample(range(total), min(size, total)))
    out: List[Dict[str, Any]] = []
    with open(path, "rb") as fh:
        for i, line in enumerate(fh):
            if i in wanted:
                out.append(json.loads(line))
    return out


def _pgp_ids(block: Dict[str, Any]) -> Dict[str, str]:
    """Primary id and accepted historic-variant ids of a record's PGP fragment rows (the merge's alias map).

    :param block: ``sources.pgp`` of a merged record.
    :type block: Dict[str, Any]
    :return: variant id -> primary id (the primary maps to itself).
    :rtype: Dict[str, str]
    """
    alias: Dict[str, str] = {}
    for row in [block["fragment"]] + list(block.get("_extra_fragments") or []):
        token, primary = ms.pgp_canonical_id(row)
        if not primary:
            continue
        alias.setdefault(primary, primary)
        accepted, _ = ms.pgp_variant_aliases(token, primary, row["shelfmark"],
                                             ms._split_semicolons(row.get("shelfmarks_historic")))
        for variant in accepted:
            alias.setdefault(variant, primary)
    return alias


def recipe_ids(record: Dict[str, Any]) -> Dict[str, Set[str]]:
    """The ids each source block of a merged record re-derives with the merge recipe (after aliasing).

    :param record: Merged record.
    :type record: Dict[str, Any]
    :return: source name -> the ids that source maps to (a PGP block yields only its primary id).
    :rtype: Dict[str, Set[str]]
    """
    sources = record.get("sources") or {}
    alias = _pgp_ids(sources["pgp"]) if sources.get("pgp") else {}
    out: Dict[str, Set[str]] = {}
    if sources.get("pgp"):
        out["pgp"] = {ms.pgp_canonical_id(sources["pgp"]["fragment"])[1]}
    if sources.get("fjp"):
        for i, rec in enumerate(sources["fjp"]):
            cids = {ms.fjp_canonical_id(mark, rec) for mark in ms.split_shelfmarks(rec.get("shelf_mark") or "")}
            out[f"store[{i}]"] = {alias.get(c, c) for c in cids if c}
    if sources.get("ktiv"):
        doc = sources["ktiv"]
        marks = [doc["shelf_mark"]] if doc.get("shelf_mark") else ms._ktiv_fallback_shelfmarks(doc)
        out["ktiv"] = {alias.get(cid, cid) for _, cid, _ in ms.ktiv_record_ids(doc, marks) if cid}
    if sources.get("bodleian"):
        rec = sources["bodleian"]
        out["bodleian"] = {ms.bodleian_canonical_id(rec) or rec.get("canonical_id") or ""}
    return out


def test_merge_recipe_reproduces_every_sampled_id() -> None:
    """Every source block of every sampled record re-derives the record's id; the facade count is printed.

    :return: None.
    :rtype: None
    """
    records = sample_records(MERGED, SAMPLE_SIZE, SEED)
    assert records
    misses: List[str] = []
    facade_off: Counter[str] = collections.Counter()
    for record in records:
        cid = record["canonical_id"]
        derived = recipe_ids(record)
        assert derived, f"{cid}: no source block"
        misses.extend(f"{cid} <- {src}: {sorted(ids)}" for src, ids in derived.items() if cid not in ids)
        if shelfmarks.canonical_id(record.get("shelfmark_display") or "", record.get("institution")) != cid:
            facade_off["+".join(record.get("sources_present") or [])] += 1
    print(f"\nT6: {len(records)} records; merge recipe misses {len(misses)}; "
          f"facade canonical_id(shelfmark_display, institution) disagrees on {sum(facade_off.values())} "
          f"(by sources: {dict(facade_off.most_common())})")
    assert not misses, misses[:20]
