# File name: reserve_test_documents.py
# Date: 10/4/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Reserve documentary test documents before the next training set is built.

Documents with a human edition that no training set has used are both the best new training data
and the best test data. This writes three registries (see
:mod:`src.datasets.evaluations.benchmark_registry`) so that every training-set builder leaves
them out:

* ``two_edition_documents_v0``: documents with two editions from different sources
  (:mod:`src.datasets.evaluations.two_edition_documents`) that no training set contains;
* ``held_out_edition_pages_v0``: the documents of the validation pages of ``pgp_editions_v1``
  (side-verified page text, never trained);
* ``offbench_sample_v0``: the documents of the fixed off-benchmark page sample that each new
  checkpoint reads.

A registry is a plain list; releasing documents later means rewriting the file. After writing,
the three names must be listed in ``benchmark_registry.BENCHMARK_REGISTRIES``.

Usage::

    .venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.reserve_test_documents \\
        --sample logs/v22b_diagnosis/offbench/sample.json
"""
import argparse
import datetime
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from src.datasets.evaluations import two_edition_documents as ted
from src.datasets.evaluations.benchmark_registry import REGISTRY_DIR, write_registry
from src.datasets.evaluations.helper_eval_scripts import build_arabic_benchmark as bab
from src.finetuning.qwen_hebrew import build_pgp_editions as bpe

EDITIONS_MANIFEST = bpe.NAS_DATASETS / "pgp_editions_v1" / "manifest.jsonl"
TWO_EDITION_NAME = "two_edition_documents_v0"
HELD_OUT_PAGES_NAME = "held_out_edition_pages_v0"
SAMPLE_NAME = "offbench_sample_v0"


def training_exposure() -> Set[str]:
    """Keys of every document some training set has used.

    :return: Canonical ids and ``pgp:<pgpid>`` keys (edition pages, documentary grounding,
        clean_v1 / clean_v2).
    :rtype: Set[str]
    """
    return set(bab.training_documents(bab.TRAINING_MANIFESTS, bab.TRAINING_ID_LISTS)) | set(bpe.load_trained_ids())


def never_trained(pgpids: Iterable[str], canonical_ids: Dict[str, List[str]], exposure: Set[str]) -> Set[str]:
    """Documents without any training exposure, by PGP id and by every canonical id they have.

    :param pgpids: Candidate PGP document ids.
    :type pgpids: Iterable[str]
    :param canonical_ids: ``{pgpid: canonical ids}``.
    :type canonical_ids: Dict[str, List[str]]
    :param exposure: Result of :func:`training_exposure`.
    :type exposure: Set[str]
    :return: The candidates no training set has used.
    :rtype: Set[str]
    """
    return {pid for pid in pgpids
            if f"pgp:{pid}" not in exposure and not any(cid in exposure for cid in canonical_ids.get(pid, []))}


def with_canonical_ids(pgpids: Iterable[str], canonical_ids: Dict[str, List[str]]) -> Tuple[Set[str], Set[str]]:
    """A registry's two lists for a set of PGP documents.

    :param pgpids: PGP document ids to reserve.
    :type pgpids: Iterable[str]
    :param canonical_ids: ``{pgpid: canonical ids}``.
    :type canonical_ids: Dict[str, List[str]]
    :return: ``(canonical ids, pgpids)``; a document missing from the merged index keeps its PGP id only.
    :rtype: Tuple[Set[str], Set[str]]
    """
    pids = {str(p) for p in pgpids}
    return {cid for pid in pids for cid in canonical_ids.get(pid, [])}, pids


def manifest_documents(manifest: Path, split: str = "val") -> Tuple[Set[str], Set[str]]:
    """Documents of one split of an edition-pages manifest.

    :param manifest: ``pgp_editions_v1/manifest.jsonl``.
    :type manifest: Path
    :param split: Split name.
    :type split: str
    :return: ``(canonical ids, pgpids)``.
    :rtype: Tuple[Set[str], Set[str]]
    """
    ids: Set[str] = set()
    pgpids: Set[str] = set()
    with open(manifest, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["split"] == split:
                ids.add(row["canonical_id"])
                pgpids.add(str(row["pgpid"]))
    return ids, pgpids


def sample_documents(sample_json: Path) -> Tuple[Set[str], Set[str]]:
    """Documents of the off-benchmark page sample.

    :param sample_json: ``offbench/sample.json`` (jobs with ``doc_id`` and ``pgpid``).
    :type sample_json: Path
    :return: ``(canonical ids, pgpids)``.
    :rtype: Tuple[Set[str], Set[str]]
    """
    jobs = json.loads(Path(sample_json).read_text(encoding="utf-8"))
    return {j["doc_id"] for j in jobs}, {str(j["pgpid"]) for j in jobs}


def main() -> None:
    """Write the three registries and print what they hold."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--registry-dir", type=Path, default=REGISTRY_DIR)
    parser.add_argument("--manifest", type=Path, default=EDITIONS_MANIFEST)
    parser.add_argument("--sample", type=Path, required=True, help="offbench/sample.json")
    parser.add_argument("--date", default=datetime.date.today().isoformat())
    args = parser.parse_args()

    two = ted.two_edition_documents()
    merged = bpe.load_merged(bpe.MERGED_JSONL, set(two))
    untouched = never_trained(two, merged.canonicals_of, training_exposure())
    ids, pgpids = with_canonical_ids(untouched, merged.canonicals_of)
    write_registry(args.registry_dir / f"{TWO_EDITION_NAME}.json", TWO_EDITION_NAME, ids, pgpids, args.date)
    print(f"{TWO_EDITION_NAME}: {len(two)} two-edition documents, {len(pgpids)} never trained ({len(ids)} canonical ids)")

    ids, pgpids = manifest_documents(args.manifest)
    write_registry(args.registry_dir / f"{HELD_OUT_PAGES_NAME}.json", HELD_OUT_PAGES_NAME, ids, pgpids, args.date)
    print(f"{HELD_OUT_PAGES_NAME}: {len(pgpids)} documents ({len(ids)} canonical ids)")

    ids, pgpids = sample_documents(args.sample)
    write_registry(args.registry_dir / f"{SAMPLE_NAME}.json", SAMPLE_NAME, ids, pgpids, args.date)
    print(f"{SAMPLE_NAME}: {len(pgpids)} documents ({len(ids)} canonical ids)")


if __name__ == "__main__":
    main()
