# File name: benchmark_registry.py
# Date: 10/2/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Registry of held-out benchmark documents, for training-set builders.

The frozen 131-fragment benchmark keeps its own inventory files (``decontam/benchmark_ids.json``,
``benchmark_keys.json``). Every later benchmark registers its documents in a
``decontam/<name>.json`` file::

    {"benchmark": "<name>", "ids": [canonical ids], "pgpids": [PGP document ids]}

and its name is listed in :data:`BENCHMARK_REGISTRIES`. A builder of training rows adds
:func:`registered_benchmark_documents` to the documents it excludes. A registry that is listed but
missing raises: a training set must not be built without knowing what is held out.

This module has no dependency beyond the standard library so that every builder can import it.
"""
import json
from pathlib import Path
from typing import Iterable, Set, Tuple

REGISTRY_DIR = Path(__file__).resolve().parents[3] / "src/datasets/raw_data/cairo_genizah/decontam"
BENCHMARK_REGISTRIES = ("arabic_script_benchmark_v0",
                        # reserved 2026-10-04 by helper_eval_scripts/reserve_test_documents.py
                        "two_edition_documents_v0", "held_out_edition_pages_v0", "offbench_sample_v0")


def registered_benchmark_documents(names: Iterable[str] = BENCHMARK_REGISTRIES,
                                   registry_dir: Path = REGISTRY_DIR) -> Tuple[Set[str], Set[str]]:
    """Documents held out by the registered benchmarks.

    :param names: Registry names (file stems under ``registry_dir``).
    :type names: Iterable[str]
    :param registry_dir: Directory of the registry files.
    :type registry_dir: Path
    :return: ``(canonical ids, PGP document ids)`` over all the registries.
    :rtype: Tuple[Set[str], Set[str]]
    :raises FileNotFoundError: When a listed registry file does not exist.
    """
    ids: Set[str] = set()
    pgpids: Set[str] = set()
    for name in names:
        with open(registry_dir / f"{name}.json", encoding="utf-8") as fh:
            registry = json.load(fh)
        ids |= set(registry["ids"])
        pgpids |= {str(pid) for pid in registry.get("pgpids", [])}
    return ids, pgpids


def write_registry(path: Path, benchmark: str, ids: Iterable[str], pgpids: Iterable[str], built: str) -> None:
    """Write a benchmark's registry file.

    :param path: ``decontam/<name>.json``.
    :type path: Path
    :param benchmark: Benchmark name.
    :type benchmark: str
    :param ids: Canonical ids of the benchmark documents.
    :type ids: Iterable[str]
    :param pgpids: PGP document ids of the benchmark documents.
    :type pgpids: Iterable[str]
    :param built: Build date (``YYYY-MM-DD``).
    :type built: str
    """
    registry = {"benchmark": benchmark, "built": built, "ids": sorted(ids), "pgpids": sorted({str(p) for p in pgpids}, key=int)}
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=1), encoding="utf-8")
