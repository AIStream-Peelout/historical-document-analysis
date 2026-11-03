"""T4 snapshots of the FROZEN shelf-mark key functions (shelf-mark migration plan §2.5, PR-7).

Some key formats must stay byte-identical because finished artefacts are keyed on them: the v2.5a held-out sets
(``derive_v25a.shelfmark_key`` / ``pgp_shelfmark_index``), the APD v3 exclusions (``arabic_external.shelfmarks``) and
the image file names on disk of the legacy store's PDF transcriptions (``pdf_transcriptions.series_key_for`` /
``shelfmark_from_filename``). Each of those functions carries a ``FROZEN 2026-11`` docstring line; new code uses
``src.datasets.shelfmarks``. Their outputs over every non-empty probe mark and expected ES id of
``tests/fixtures/shelfmark_probes.json`` were captured in ``tests/fixtures/shelfmark_frozen_keys.json`` (one section
per module) and must not drift. The text27b keys have their own snapshot (``test_text27b_shelfmarks_frozen.py``).

``derive_v25a`` and ``arabic_external`` are private modules that a public clone does not have: their sections are
skipped, never failed, when the module is absent. Regenerate the snapshot only with an explicit decision to re-key
the artefact named in the module's ``FROZEN`` line.
"""

from __future__ import annotations

import csv
import importlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PROBES = FIXTURES / "shelfmark_probes.json"
SNAPSHOT = FIXTURES / "shelfmark_frozen_keys.json"

#: Section name -> (module, private: skipped when absent).
MODULES: Dict[str, tuple] = {
    "derive_v25a": ("src.finetuning.qwen_hebrew.colab.derive_v25a", True),
    "arabic_external": ("src.datasets.indexing.arabic_external.shelfmarks", True),
    "pdf_transcriptions": ("src.datasets.indexing.legacy_store.pdf_transcriptions", False),
}
#: The docstring line every frozen function carries (its artefact named in between).
FROZEN_LINE = ("FROZEN 2026-11: key format of ", "; new code must use src.datasets.shelfmarks")
#: Section -> the frozen functions it pins.
FROZEN_FUNCTIONS: Dict[str, tuple] = {
    "derive_v25a": ("shelfmark_key", "pgp_shelfmark_index"),
    "arabic_external": ("fold", "tokens", "make_key", "base_key", "apd_inventory_keys", "shelfmark_key",
                        "shelfmark_keys", "canonical_id_key", "comparable_keys"),
    "pdf_transcriptions": ("series_key_for", "shelfmark_from_filename"),
}


def probe_inputs() -> List[Dict[str, str]]:
    """Every non-empty probe mark and expected ES id, in probe order.

    :return: ``[{"id", "input", "value"}]`` (``input`` is ``mark`` or ``expected_es_id``).
    :rtype: List[Dict[str, str]]
    """
    probes = json.loads(PROBES.read_text(encoding="utf-8"))["probes"]
    return [{"id": p["id"], "input": field, "value": p[field]}
            for p in probes for field in ("mark", "expected_es_id") if p[field]]


def module_present(name: str) -> bool:
    """Whether a dotted module exists, its parent packages included (a missing private package is not an error).

    :param name: Dotted module name.
    :type name: str
    :return: True when the module and every parent package can be found.
    :rtype: bool
    """
    parent = name.rpartition(".")[0]
    if parent and not module_present(parent):
        return False
    return importlib.util.find_spec(name) is not None


def load_module(section: str) -> ModuleType:
    """Import a section's module, skipping the test when a private module is absent.

    :param section: Key of :data:`MODULES`.
    :type section: str
    :return: The module.
    :rtype: ModuleType
    """
    name, private = MODULES[section]
    if private and not module_present(name):
        pytest.skip(f"{name} is a private module (git-excluded) and is absent from this clone")
    return importlib.import_module(name)


def file_name_of(value: str) -> str:
    """The old-style transcription file name a shelf mark would be stored under (spaces as underscores).

    :param value: Probe value.
    :type value: str
    :return: ``transcription_manual_<value with underscores>.pdf``.
    :rtype: str
    """
    return f"transcription_manual_{value.replace(' ', '_')}.pdf"


def _rows(fns: Dict[str, Callable[[str], Any]]) -> List[Dict[str, Any]]:
    """One row per probe input with each function's output.

    :param fns: Column name -> function of the probe value (outputs must be JSON values: tuples become lists).
    :type fns: Dict[str, Callable[[str], Any]]
    :return: Rows.
    :rtype: List[Dict[str, Any]]
    """
    out = []
    for item in probe_inputs():
        row: Dict[str, Any] = dict(item)
        for col, fn in fns.items():
            value = fn(item["value"])
            row[col] = list(value) if isinstance(value, tuple) else value
        out.append(row)
    return out


def current_section(section: str, tmp_dir: Path) -> Dict[str, Any]:
    """The frozen functions' outputs today for one section (what the snapshot holds).

    :param section: Key of :data:`MODULES`.
    :type section: str
    :param tmp_dir: Directory for the synthetic PGP ``documents.csv`` of ``pgp_shelfmark_index``.
    :type tmp_dir: Path
    :return: ``{"module", "rows"}`` (plus ``pgp_shelfmark_index`` for ``derive_v25a``).
    :rtype: Dict[str, Any]
    """
    mod = load_module(section)
    if section == "derive_v25a":
        documents = tmp_dir / "documents.csv"
        with open(documents, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["pgpid", "shelfmark"])
            writer.writeheader()
            for item in probe_inputs():
                if item["input"] == "mark":
                    writer.writerow({"pgpid": item["id"], "shelfmark": item["value"]})
        index = mod.pgp_shelfmark_index(documents)
        return {"module": MODULES[section][0], "rows": _rows({"shelfmark_key": mod.shelfmark_key}),
                "pgp_shelfmark_index": {k: sorted(v) for k, v in sorted(index.items())}}
    if section == "arabic_external":
        return {"module": MODULES[section][0], "rows": _rows({
            "fold": mod.fold, "tokens": mod.tokens,
            "make_key": lambda v: mod.make_key(mod.UNKNOWN, mod.tokens(v)),
            "base_key": lambda v: [mod.base_key(k) for k in mod.comparable_keys(v)],
            "apd_inventory_keys": mod.apd_inventory_keys, "shelfmark_key": mod.shelfmark_key,
            "shelfmark_keys": mod.shelfmark_keys, "canonical_id_key": mod.canonical_id_key,
            "comparable_keys": mod.comparable_keys})}
    return {"module": MODULES[section][0], "rows": _rows({
        "series_key_for": mod.series_key_for, "file_name": file_name_of,
        "shelfmark_from_filename": lambda v: mod.shelfmark_from_filename(file_name_of(v))})}


def snapshot_section(section: str) -> Dict[str, Any]:
    """A section of the committed snapshot.

    :param section: Key of :data:`MODULES`.
    :type section: str
    :return: The section.
    :rtype: Dict[str, Any]
    """
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))["sections"][section]


@pytest.mark.parametrize("section", sorted(MODULES))
def test_frozen_functions_match_snapshot(section: str, tmp_path: Path) -> None:
    """Every frozen function gives its snapshot value on every probe input.

    :param section: Key of :data:`MODULES`.
    :type section: str
    :param tmp_path: pytest's temporary directory.
    :type tmp_path: Path
    """
    now, then = current_section(section, tmp_path), snapshot_section(section)
    drift = [(a["id"], a["input"], col, b[col], a[col]) for a, b in zip(now["rows"], then["rows"])
             for col in a if a[col] != b.get(col)]
    assert len(now["rows"]) == len(then["rows"]) and not drift, drift[:5]
    assert now.get("pgp_shelfmark_index") == then.get("pgp_shelfmark_index")


@pytest.mark.parametrize("section", sorted(MODULES))
def test_frozen_functions_carry_the_frozen_line(section: str) -> None:
    """Each pinned function's docstring names the artefact its key format belongs to.

    :param section: Key of :data:`MODULES`.
    :type section: str
    """
    mod = load_module(section)
    head, tail = FROZEN_LINE
    missing = [fn for fn in FROZEN_FUNCTIONS[section]
               if not any(line.strip().startswith(head) and line.strip().endswith(tail)
                          for line in (getattr(mod, fn).__doc__ or "").splitlines())]
    assert not missing, missing


def test_snapshot_covers_every_probe_input() -> None:
    """Every section has one row per non-empty probe mark and expected ES id (309 inputs), in probe order."""
    expected = [(i["id"], i["input"], i["value"]) for i in probe_inputs()]
    assert len(expected) == 309
    for section in MODULES:
        rows = snapshot_section(section)["rows"]
        assert [(r["id"], r["input"], r["value"]) for r in rows] == expected, section
