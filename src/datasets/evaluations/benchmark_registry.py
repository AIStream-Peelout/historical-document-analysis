# File name: benchmark_registry.py
# Date: 10/2/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Registry of held-out benchmark documents, for training-set builders.

The frozen 131-fragment benchmark keeps its own inventory files (``decontam/benchmark_ids.json``,
``benchmark_keys.json``). Every later benchmark registers its documents in a
``decontam/<name>.json`` file::

    {"benchmark": "<name>", "ids": [canonical ids], "pgpids": [PGP document ids],
     "sys_nums": [KTIV manuscripts]}

and its name is listed in :data:`BENCHMARK_REGISTRIES`. ``sys_nums`` is optional (documentary
registries have none). A registry may instead live next to the benchmark it seals when that
directory is the only place its builder writes; :data:`REGISTRY_LOCATIONS` maps such a name to
its directory (``sealed_v1``, whose registry sits in ``evaluations/sealed_v1/``).

A builder of training rows adds :func:`registered_benchmark_documents` to the documents it
excludes, and a KTIV builder adds :func:`registered_benchmark_manuscripts` to the manuscripts it
excludes. A registry that is listed but missing raises: a training set must not be built without
knowing what is held out.

This module has no dependency beyond the standard library so that every builder can import it.
"""
import json
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

_CAIRO_GENIZAH = Path(__file__).resolve().parents[3] / "src/datasets/raw_data/cairo_genizah"
REGISTRY_DIR = _CAIRO_GENIZAH / "decontam"
#: Registries that live next to their benchmark instead of under ``decontam/`` (used with the default ``registry_dir``).
REGISTRY_LOCATIONS: Dict[str, Path] = {"sealed_v1": _CAIRO_GENIZAH / "evaluations/sealed_v1"}
BENCHMARK_REGISTRIES = ("arabic_script_benchmark_v0",
                        # reserved 2026-10-04 by helper_eval_scripts/reserve_test_documents.py
                        "two_edition_documents_v0", "held_out_edition_pages_v0", "offbench_sample_v0",
                        # sealed confirmation set, 2026-10-12, helper_eval_scripts/build_sealed_benchmark.py
                        "sealed_v1")


def registry_path(name: str, registry_dir: Path = REGISTRY_DIR) -> Path:
    """File of one registry.

    :param name: Registry name.
    :type name: str
    :param registry_dir: Directory of the registry files; when it is the default, a name listed in
        :data:`REGISTRY_LOCATIONS` resolves to its own directory.
    :type registry_dir: Path
    :return: ``<directory>/<name>.json``.
    :rtype: Path
    """
    if registry_dir == REGISTRY_DIR and name in REGISTRY_LOCATIONS:
        return REGISTRY_LOCATIONS[name] / f"{name}.json"
    return registry_dir / f"{name}.json"


def _load(name: str, registry_dir: Path) -> Dict:
    """Read one registry.

    :param name: Registry name.
    :type name: str
    :param registry_dir: Directory of the registry files.
    :type registry_dir: Path
    :return: The registry object.
    :rtype: Dict
    :raises FileNotFoundError: When the registry file does not exist.
    """
    with open(registry_path(name, registry_dir), encoding="utf-8") as fh:
        return json.load(fh)


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
        registry = _load(name, registry_dir)
        ids |= set(registry["ids"])
        pgpids |= {str(pid) for pid in registry.get("pgpids", [])}
    return ids, pgpids


def registered_benchmark_manuscripts(names: Iterable[str] = BENCHMARK_REGISTRIES,
                                     registry_dir: Path = REGISTRY_DIR) -> Set[str]:
    """KTIV manuscripts (NLI system numbers) held out by the registered benchmarks.

    :param names: Registry names (file stems under ``registry_dir``).
    :type names: Iterable[str]
    :param registry_dir: Directory of the registry files.
    :type registry_dir: Path
    :return: System numbers over all the registries (registries without ``sys_nums`` add none).
    :rtype: Set[str]
    :raises FileNotFoundError: When a listed registry file does not exist.
    """
    return {str(s) for name in names for s in _load(name, registry_dir).get("sys_nums", [])}


def write_registry(path: Path, benchmark: str, ids: Iterable[str], pgpids: Iterable[str], built: str,
                   sys_nums: Iterable[str] = ()) -> None:
    """Write a benchmark's registry file.

    :param path: ``decontam/<name>.json`` (or the location in :data:`REGISTRY_LOCATIONS`).
    :type path: Path
    :param benchmark: Benchmark name.
    :type benchmark: str
    :param ids: Canonical ids of the benchmark documents.
    :type ids: Iterable[str]
    :param pgpids: PGP document ids of the benchmark documents.
    :type pgpids: Iterable[str]
    :param built: Build date (``YYYY-MM-DD``).
    :type built: str
    :param sys_nums: KTIV manuscripts of the benchmark; the key is written only when there are any.
    :type sys_nums: Iterable[str]
    """
    registry = {"benchmark": benchmark, "built": built, "ids": sorted(ids), "pgpids": sorted({str(p) for p in pgpids}, key=int)}
    manuscripts = sorted({str(s) for s in sys_nums})
    if manuscripts:
        registry["sys_nums"] = manuscripts
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=1), encoding="utf-8")
